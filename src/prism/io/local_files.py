"""Local-file ingestion under a declared mapping (engine specification §3.2, "Local ingestion").

``prism data ingest --format parquet|csv --mapping <toml> <path>`` reads files
this workspace already holds and writes them into the tables of §3.2. The
mapping file is the single declaration point, and nothing here is inferred from
a file name or a column heading:

* ``[source]`` declares the source id, its availability lag in sessions, its
  revision policy, the price basis of its bars, the timezone its timestamps are
  written in, and whether it may be redistributed.
* ``[columns]`` declares which column of the file carries each canonical bar
  field. The same mapping read against a CSV and against a Parquet of the same
  content produces the same rows, which is what
  ``tests/test_local_ingest.py::test_a_csv_and_a_parquet_of_one_series_ingest_identically``
  asserts.
* ``[identity]`` declares which column carries the vendor symbol and how a
  series is matched to an instrument. Matching itself is
  :class:`prism.io.security_master.SecurityMaster`'s: a series that matches no
  existing sid mints a provisional one and is flagged for attestation.

Three rules the module exists to enforce.

*A revision is a whole row.* §3.2: "Every revision is a complete replacement of
its row: an explicit null in a later revision means the value is null. Partial
updates are not supported in the first version; ingestion builds full rows." So
a re-ingested row that differs in one field is written as the next revision
carrying *every* field of the incoming row, and a field the incoming file does
not carry is null in that revision rather than inherited from the previous one.
:func:`merge_bar_revisions` is that rule, and the three ``revision_policy``
values choose when a revision is written at all.

*A lag is a property of a source, never of a row.* ``lag_bars`` is copied from
the mapping into the manifest's ``sources`` entry and read back from there by
:class:`prism.io.dataset.DatasetView`, so two rows of one source can never
carry different lags.

*A key absent from a later ingest is not a deletion.* Deletion is an explicit
null-valued full revision (worked example B of §3.2). An ingest that stops
covering a date leaves the stored row exactly where it was.

One consequence of the identity rule worth stating, because it decides how a
tail must be fetched. A series matches an existing sid when its ``(ticker,
interval)`` overlaps a stored ``listings`` row (§3.2), so a delta-fetched tail
that begins *after* the last stored bar overlaps nothing, matches nothing, and
mints a second provisional sid flagged for attestation. That is the same rule
that makes a ticker reused after a delisting two instruments, and it is why the
incremental store re-requests ``DEFAULT_OVERLAP_BARS`` of overlap with every
delta fetch (``prism.io.store``). Ingest a tail that overlaps the stored series.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.io import sessions
from prism.io.dataset import BAR_TZ, DEFAULT_BAR_INTERVAL
from prism.io.events import ADJUSTMENT_COLUMNS, EVENT_COLUMNS, PRICE_BASES
from prism.io.manifest import (
    REVISION_POLICIES,
    Manifest,
    canonical_json,
    hash_tree,
    sha256_bytes,
)
from prism.io.security_master import DEFAULT_GAP_FLAG_SESSIONS, SecurityMaster

logger = logging.getLogger(__name__)

#: The mapping-file schema this module reads. A file declaring another schema
#: is refused rather than read under this one's assumptions.
MAPPING_SCHEMA = 1

#: The canonical bar fields ``[columns]`` maps. All six are required: a series
#: without a volume is not a bar table, and a close-only exogenous series
#: belongs in the ``series`` table of §3.2, not here.
BAR_FIELDS: tuple[str, ...] = ("ts", "open", "high", "low", "close", "volume")

#: Column order of the ``bars`` table (§3.2).
BAR_COLUMNS: tuple[str, ...] = (
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_basis",
    "available_at",
    "ingested_at",
    "source",
    "revision",
)

#: The fields compared when deciding whether a re-ingested row equals the
#: stored one. ``revision`` is the answer rather than an input, and
#: ``ingested_at`` records when this workspace stored the row rather than what
#: the row claims, so an unchanged bar re-read tomorrow keeps its first date.
_COMPARED_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "price_basis", "available_at")

#: The only ``[identity] match`` value the first version implements. Anything
#: else is refused rather than silently treated as the default: the alternative
#: to overlapping-ticker matching is an attestation in
#: ``security_overrides.toml``, not another automatic rule.
MATCH_RULES: tuple[str, ...] = ("ticker_interval_overlap",)

_SOURCE_REQUIRED: tuple[str, ...] = ("id", "lag_bars", "revision_policy", "price_basis")
_SOURCE_ALLOWED: frozenset[str] = frozenset(
    {
        "id",
        "kind",
        "lag_bars",
        "revision_policy",
        "price_basis",
        "timezone",
        "gap_flag_sessions",
        "redistributable",
        "url",
        "sha256",
        "interval",
        "asset_class",
    }
)
_IDENTITY_ALLOWED: frozenset[str] = frozenset({"vendor_symbol", "match", "ticker"})
_MAPPING_SECTIONS: frozenset[str] = frozenset({"schema", "source", "columns", "identity"})


class IngestError(ValueError):
    """A local ingest cannot be performed as declared, and guessing would forge provenance."""


# ---------------------------------------------------------------------------
# the mapping file
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceDeclaration:
    """The ``[source]`` block: everything about the source that is not its bytes."""

    id: str
    lag_bars: int
    revision_policy: str
    price_basis: str
    kind: str = "local_file"
    timezone: str = BAR_TZ
    gap_flag_sessions: int = DEFAULT_GAP_FLAG_SESSIONS
    redistributable: bool = False
    url: str | None = None
    sha256: str | None = None
    interval: str = DEFAULT_BAR_INTERVAL
    asset_class: str = "equity"

    def manifest_entry(self) -> dict[str, Any]:
        """The manifest ``sources`` entry §3.2 hashes into the dataset identity."""
        return {
            "lag_bars": int(self.lag_bars),
            "revision_policy": self.revision_policy,
            "price_basis": self.price_basis,
            "url": self.url,
            "sha256": self.sha256,
            "redistributable": bool(self.redistributable),
        }


@dataclass(frozen=True)
class IngestMapping:
    """One parsed mapping file (§3.2, "Local ingestion")."""

    source: SourceDeclaration
    columns: dict[str, str]
    vendor_symbol: str | None = None
    ticker: str | None = None
    match: str = MATCH_RULES[0]
    path: Path | None = None

    @classmethod
    def read(cls, path: Path | str) -> "IngestMapping":
        """Parse a mapping file, refusing an unknown schema, section, or key."""
        path = Path(path)
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise IngestError(f"no mapping file at {path}; `prism data ingest` declares its columns in one") from exc
        except tomllib.TOMLDecodeError as exc:
            raise IngestError(f"{path} does not parse as TOML: {exc}") from exc
        return cls.from_dict(payload, path=path)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, path: Path | None = None) -> "IngestMapping":
        where = str(path) if path is not None else "the mapping"
        unknown = sorted(set(payload) - _MAPPING_SECTIONS)
        if unknown:
            raise IngestError(
                f"{where} carries unknown section(s) {unknown}; §3.2 declares {sorted(_MAPPING_SECTIONS)}"
            )
        schema = payload.get("schema")
        if schema != MAPPING_SCHEMA:
            raise IngestError(f"{where} declares schema {schema!r}; this reader implements schema {MAPPING_SCHEMA}")
        source = dict(payload.get("source") or {})
        if not source:
            raise IngestError(f"{where} has no [source] block; the source id, lag and revision policy are declared")
        unknown = sorted(set(source) - _SOURCE_ALLOWED)
        if unknown:
            raise IngestError(
                f"{where} [source] carries unknown key(s) {unknown}; a misspelled key would declare nothing"
            )
        missing = [key for key in _SOURCE_REQUIRED if source.get(key) in (None, "")]
        if missing:
            raise IngestError(f"{where} [source] is missing {missing}")
        lag_bars = source["lag_bars"]
        if not isinstance(lag_bars, int) or isinstance(lag_bars, bool) or lag_bars < 0:
            raise IngestError(f"{where} [source] lag_bars must be a non-negative integer, got {lag_bars!r}")
        if source["revision_policy"] not in REVISION_POLICIES:
            raise IngestError(
                f"{where} [source] revision_policy must be one of {list(REVISION_POLICIES)}, "
                f"got {source['revision_policy']!r}"
            )
        if source["price_basis"] not in PRICE_BASES:
            raise IngestError(
                f"{where} [source] price_basis must be one of {list(PRICE_BASES)}, got {source['price_basis']!r}"
            )
        gap_flag_sessions = int(source.get("gap_flag_sessions", DEFAULT_GAP_FLAG_SESSIONS))
        if gap_flag_sessions < 0:
            raise IngestError(f"{where} [source] gap_flag_sessions must be non-negative, got {gap_flag_sessions}")
        declaration = SourceDeclaration(
            id=str(source["id"]),
            lag_bars=int(lag_bars),
            revision_policy=str(source["revision_policy"]),
            price_basis=str(source["price_basis"]),
            kind=str(source.get("kind", "local_file")),
            timezone=str(source.get("timezone", BAR_TZ)),
            gap_flag_sessions=gap_flag_sessions,
            redistributable=bool(source.get("redistributable", False)),
            url=_optional_text(source.get("url")),
            sha256=_optional_text(source.get("sha256")),
            interval=str(source.get("interval", DEFAULT_BAR_INTERVAL)),
            asset_class=str(source.get("asset_class", "equity")),
        )
        columns = {str(k): str(v) for k, v in dict(payload.get("columns") or {}).items()}
        unknown = sorted(set(columns) - set(BAR_FIELDS))
        if unknown:
            raise IngestError(f"{where} [columns] maps {unknown}, which are not bar fields {list(BAR_FIELDS)}")
        missing = [field_name for field_name in BAR_FIELDS if field_name not in columns]
        if missing:
            raise IngestError(
                f"{where} [columns] does not map {missing}; the bars table of §3.2 carries all of "
                f"{list(BAR_FIELDS)}, and a column this workspace guessed would not be a declaration"
            )
        identity = dict(payload.get("identity") or {})
        unknown = sorted(set(identity) - _IDENTITY_ALLOWED)
        if unknown:
            raise IngestError(f"{where} [identity] carries unknown key(s) {unknown}")
        match = str(identity.get("match", MATCH_RULES[0]))
        if match not in MATCH_RULES:
            raise IngestError(
                f"{where} [identity] match must be one of {list(MATCH_RULES)}, got {match!r}; a mapping that "
                "overlapping-ticker matching cannot establish is asserted in security_overrides.toml, not by "
                "another automatic rule"
            )
        return cls(
            source=declaration,
            columns=columns,
            vendor_symbol=_optional_text(identity.get("vendor_symbol")),
            ticker=_optional_text(identity.get("ticker")),
            match=match,
            path=path,
        )


# ---------------------------------------------------------------------------
# reading a file under the mapping
# ---------------------------------------------------------------------------


def read_mapped_file(path: Path | str, *, fmt: Literal["parquet", "csv"], mapping: IngestMapping) -> pd.DataFrame:
    """One file's bars in canonical columns, ts-indexed and timezone-resolved.

    The declared ``ts`` column may be a column or the frame's index — a Parquet
    written by :class:`prism.io.store.IncrementalBarStore` stores the
    timestamps in the index — so a non-range index is reset into a column
    before the mapping is applied. Naive timestamps are localized to the
    ``[source] timezone``; aware ones are converted to it.
    """
    path = Path(path)
    if fmt == "parquet":
        frame = pd.read_parquet(path)
    elif fmt == "csv":
        frame = pd.read_csv(path)
    else:  # pragma: no cover - argparse constrains the value
        raise IngestError(f"format must be 'parquet' or 'csv', got {fmt!r}")
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    frame.columns = [str(column) for column in frame.columns]
    wanted = {field_name: mapping.columns[field_name] for field_name in BAR_FIELDS}
    missing = sorted({column for column in wanted.values()} - set(frame.columns))
    if missing:
        raise IngestError(
            f"{path} does not carry the column(s) {missing} the mapping declares; the file's columns are "
            f"{sorted(frame.columns)}"
        )
    stamps = _timestamps(frame[wanted["ts"]], timezone=mapping.source.timezone, where=path)
    out = pd.DataFrame(index=stamps)
    out.index.name = "ts"
    for field_name in BAR_FIELDS[1:]:
        out[field_name] = pd.to_numeric(frame[wanted[field_name]], errors="coerce").to_numpy(dtype=float)
    symbols = _symbol_column(frame, mapping, path=path)
    if symbols is not None:
        out["vendor_symbol"] = symbols.to_numpy()
    if out.empty:
        raise IngestError(f"{path} carries no rows; an empty file is a failed fetch, not an empty result (N7)")
    return out


def split_by_symbol(frame: pd.DataFrame, *, default_symbol: str | None = None) -> dict[str, pd.DataFrame]:
    """Group a mapped frame by ``vendor_symbol``, refusing a duplicate ``(symbol, ts)``.

    Two rows of one file for one symbol and one timestamp cannot be two
    revisions of each other — one file has one ingestion instant — so they are
    an ingestion defect and are refused with the key, never deduplicated (N7).
    """
    if "vendor_symbol" in frame.columns:
        blank = frame["vendor_symbol"].isna() | (frame["vendor_symbol"].astype(str).str.strip() == "")
        if bool(blank.any()):
            raise IngestError(
                f"{int(blank.sum())} of {len(frame)} rows carry no vendor symbol; a bar with no instrument cannot "
                "be resolved to a sid"
            )
        groups = {str(symbol): part.drop(columns=["vendor_symbol"]) for symbol, part in frame.groupby("vendor_symbol")}
    else:
        if default_symbol is None:
            raise IngestError(
                "the mapping declares no [identity] vendor_symbol column and no symbol was supplied; a bar file "
                "with no symbol is a series with no instrument"
            )
        groups = {str(default_symbol): frame}
    for symbol, part in groups.items():
        duplicated = part.index.duplicated()
        if bool(duplicated.any()):
            stamps = [str(ts) for ts in part.index[duplicated][:3]]
            raise IngestError(
                f"{symbol} carries more than one row for {stamps}; two rows of one file for one timestamp are an "
                "ingestion defect, not two revisions — one file has one ingestion instant"
            )
        groups[symbol] = part.sort_index()
    return groups


# ---------------------------------------------------------------------------
# revisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarBatch:
    """One file's bars for one sid, with the instant this workspace stored them."""

    sid: str
    frame: pd.DataFrame
    ingested_at: str
    source: str
    price_basis: str
    origin: str = ""
    available_at: str | None = None


@dataclass(frozen=True)
class RevisionOutcome:
    """What :func:`merge_bar_revisions` did, per batch, so a report can state it."""

    written: int = 0
    deduped: int = 0
    revised: int = 0

    def __add__(self, other: "RevisionOutcome") -> "RevisionOutcome":
        return RevisionOutcome(
            written=self.written + other.written,
            deduped=self.deduped + other.deduped,
            revised=self.revised + other.revised,
        )


def merge_bar_revisions(
    stored: pd.DataFrame | None,
    batches: Sequence[BarBatch],
    *,
    policy: str,
) -> tuple[pd.DataFrame, RevisionOutcome]:
    """Fold ``batches`` into a §3.2 ``bars`` table under one ``revision_policy``.

    Every revision is a complete row. A differing re-ingest appends the next
    revision within ``(ts, source)`` carrying every field of the *incoming*
    row, so a field the new file does not carry is null in the new revision
    rather than inherited: a partial update would silently mix two vintages of
    one bar.

    ``dedupe_identical`` writes nothing when the incoming row equals the stored
    one field for field; ``always_new_revision`` writes a revision on every
    ingest, which is what a source that corrects silently requires when the
    history of what this workspace held must be exact; ``conflict_raise``
    refuses a differing row and names the key.

    A key absent from a later batch is never a deletion — deletion is an
    explicit null-valued full revision (worked example B).
    """
    if policy not in REVISION_POLICIES:
        raise IngestError(f"revision_policy must be one of {list(REVISION_POLICIES)}, got {policy!r}")
    rows: list[dict[str, Any]] = []
    latest: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
    if stored is not None and not stored.empty:
        for row in _normalize_stored(stored).to_dict("records"):
            rows.append(row)
            key = (pd.Timestamp(row["ts"]), str(row["source"]))
            previous = latest.get(key)
            if previous is None or int(row["revision"]) > int(previous["revision"]):
                latest[key] = row
    outcome = RevisionOutcome()
    for batch in batches:
        for stamp, values in zip(batch.frame.index, batch.frame.to_dict("records")):
            candidate = {
                "ts": pd.Timestamp(stamp),
                "open": _float(values.get("open")),
                "high": _float(values.get("high")),
                "low": _float(values.get("low")),
                "close": _float(values.get("close")),
                "volume": _float(values.get("volume")),
                "price_basis": batch.price_basis,
                "available_at": batch.available_at,
                "ingested_at": batch.ingested_at,
                "source": batch.source,
                "revision": 1,
            }
            key = (candidate["ts"], batch.source)
            previous = latest.get(key)
            if previous is None:
                rows.append(candidate)
                latest[key] = candidate
                outcome = outcome + RevisionOutcome(written=1)
                continue
            identical = _same_row(previous, candidate)
            if identical and policy != "always_new_revision":
                outcome = outcome + RevisionOutcome(deduped=1)
                continue
            if not identical and policy == "conflict_raise":
                differing = [
                    name for name in _COMPARED_FIELDS if not _same_value(previous.get(name), candidate.get(name))
                ]
                raise IngestError(
                    f"{batch.sid} at {candidate['ts']} from {batch.origin or 'this ingest'} differs from the stored "
                    f"revision {previous['revision']} in {differing}; the source declares revision_policy "
                    "'conflict_raise', which refuses a differing row rather than revising it"
                )
            candidate["revision"] = int(previous["revision"]) + 1
            rows.append(candidate)
            latest[key] = candidate
            outcome = outcome + RevisionOutcome(revised=1)
    frame = pd.DataFrame(rows, columns=list(BAR_COLUMNS))
    if frame.empty:
        return _empty_bars(), outcome
    frame["ts"] = pd.DatetimeIndex(pd.to_datetime(frame["ts"], utc=True)).tz_convert(BAR_TZ)
    frame = frame.sort_values(["ts", "source", "revision"], kind="stable").reset_index(drop=True)
    frame["revision"] = frame["revision"].astype("int64")
    return frame[list(BAR_COLUMNS)], outcome


# ---------------------------------------------------------------------------
# the ingest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestReport:
    """What one ``prism data ingest`` wrote, and what it refused to write."""

    root: Path
    manifest: Manifest
    source: str
    files: tuple[str, ...]
    sids: tuple[str, ...]
    symbols: dict[str, str]
    rows_written: int
    rows_deduped: int
    rows_revised: int
    minted_sids: tuple[str, ...]
    gap_flags: tuple[dict[str, Any], ...] = ()
    gap_scan_skipped: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "data_id": self.manifest.id,
            "vintage": self.manifest.vintage,
            "as_of": self.manifest.as_of,
            "source": self.source,
            "n_files": len(self.files),
            "n_sids": len(self.sids),
            "symbols": dict(sorted(self.symbols.items())),
            "rows_written": int(self.rows_written),
            "rows_deduped": int(self.rows_deduped),
            "rows_revised": int(self.rows_revised),
            "minted_sids": list(self.minted_sids),
            "gap_flags": [dict(flag) for flag in self.gap_flags],
            "gap_scan_skipped": [dict(entry) for entry in self.gap_scan_skipped],
        }


def ingest_local_files(
    root: Path | str,
    paths: Sequence[Path | str],
    *,
    mapping: IngestMapping,
    fmt: Literal["parquet", "csv"],
    vintage: str,
    ingested_at: str | None = None,
    symbol_from_filename: bool = False,
    provenance: Literal["market", "synthetic", "mixed"] = "market",
) -> IngestReport:
    """Ingest local bar files into the tables of §3.2 under ``mapping``.

    Identity is :class:`SecurityMaster`'s: each vendor series is resolved to a
    sid, minting a provisional one where nothing matches, and the gap flag of
    §3.2 is computed from the series' own session list. The bars of each sid
    are folded into whatever the workspace already holds for it under the
    source's declared revision policy, and the manifest is rewritten so it
    describes exactly the files that are there.
    """
    root = Path(root)
    files = [Path(p) for p in paths]
    if not files:
        raise IngestError("no files to ingest; `prism data ingest` names a file or a directory of them")
    for path in files:
        if not path.is_file():
            raise IngestError(f"{path} is not a file")
    stamp = _instant(ingested_at) if ingested_at is not None else _now_instant()
    interval = mapping.source.interval

    batches: dict[str, list[BarBatch]] = {}
    dates: dict[str, set[dt.date]] = {}
    symbols_seen: dict[str, list[Path]] = {}
    for path in files:
        frame = read_mapped_file(path, fmt=fmt, mapping=mapping)
        default = _symbol_from_filename(path, interval) if symbol_from_filename else mapping.ticker
        for symbol, part in split_by_symbol(frame, default_symbol=default).items():
            symbols_seen.setdefault(symbol, []).append(path)
            batches.setdefault(symbol, []).append(
                BarBatch(
                    sid=symbol,
                    frame=part,
                    ingested_at=stamp,
                    source=mapping.source.id,
                    price_basis=mapping.source.price_basis,
                    origin=path.name,
                )
            )
            dates.setdefault(symbol, set()).update(ts.date() for ts in part.index)

    master = SecurityMaster.open(root, gap_flag_sessions=mapping.source.gap_flag_sessions)
    before = set(master.sids())
    resolution_for: dict[str, str] = {}
    gap_flags: list[dict[str, Any]] = []
    skipped_scans: list[dict[str, str]] = []
    for symbol in sorted(batches):
        days = sorted(dates[symbol])
        scan = _scannable(days)
        if not scan:
            skipped_scans.append(
                {
                    "vendor_symbol": symbol,
                    "reason": (
                        f"bar dates {days[0].isoformat()}..{days[-1].isoformat()} fall outside the bundled session "
                        f"table {sessions.TABLE_VERSION}, which cannot count the missing sessions a gap flag needs"
                    ),
                }
            )
        resolution = master.ingest_series(
            mapping.source.id,
            symbol,
            ingested_at=stamp,
            ticker=mapping.ticker if len(batches) == 1 else None,
            bar_dates=days if scan else None,
            first_bar=days[0],
            last_bar=days[-1],
            asset_class=mapping.source.asset_class,
        )
        sids = resolution.sids
        if len(sids) != 1:
            raise IngestError(
                f"{mapping.source.id}:{symbol} resolves to {len(sids)} identities {list(sids)} — the series carries a "
                "known identity break, so its bars belong to more than one instrument and this ingest cannot write "
                "one bar file for it. Split the file at the break and ingest the parts."
            )
        resolution_for[symbol] = sids[0]
        gap_flags.extend(flag.to_coverage_entry() for flag in resolution.gap_flags)

    written: list[Path] = []
    outcome = RevisionOutcome()
    (root / "bars").mkdir(parents=True, exist_ok=True)
    for symbol, sid in sorted(resolution_for.items()):
        path = root / "bars" / f"{sid}_{interval}.parquet"
        stored = pd.read_parquet(path) if path.is_file() else None
        frame, one = merge_bar_revisions(
            stored,
            [
                BarBatch(
                    sid=sid,
                    frame=b.frame,
                    ingested_at=b.ingested_at,
                    source=b.source,
                    price_basis=b.price_basis,
                    origin=b.origin,
                )
                for b in batches[symbol]
            ],
            policy=mapping.source.revision_policy,
        )
        _write_bars(frame, path)
        outcome = outcome + one
        written.append(path)

    written.extend(master.write(root))
    written.extend(ensure_event_tables(root))
    report = master.report()
    manifest = rebuild_manifest(
        root,
        vintage=vintage,
        provenance=provenance,
        sources={mapping.source.id: mapping.source.manifest_entry()},
        coverage={
            **report.coverage_entries(),
            "gap_scan_skipped": skipped_scans,
            "ingested_files": [path.name for path in files],
            "ingested_format": fmt,
        },
        assumptions={
            "ingestion_time_quality": "ingest_wall_clock" if ingested_at is None else "declared",
            "availability": (
                "available_at is empty on every ingested row; availability derives from the session close of the "
                f"bar's own session plus the source's declared lag_bars ({mapping.source.lag_bars})"
            ),
            "session_table_version": sessions.TABLE_VERSION,
            "gap_flag_sessions": int(mapping.source.gap_flag_sessions),
            "revision_policy": mapping.source.revision_policy,
            "mapping": None if mapping.path is None else Path(mapping.path).name,
        },
        transformations=[f"local_ingest_v1:{mapping.source.id}", f"interval={interval}"],
    )
    minted = tuple(sorted(set(master.sids()) - before))
    logger.info(
        "ingested %d file(s) into %d sid(s) under %s (data_id %s)",
        len(files),
        len(resolution_for),
        mapping.source.id,
        manifest.id,
    )
    return IngestReport(
        root=root,
        manifest=manifest,
        source=mapping.source.id,
        files=tuple(sorted(path.name for path in files)),
        sids=tuple(sorted(resolution_for.values())),
        symbols={symbol: sid for symbol, sid in sorted(resolution_for.items())},
        rows_written=outcome.written,
        rows_deduped=outcome.deduped,
        rows_revised=outcome.revised,
        minted_sids=minted,
        gap_flags=tuple(gap_flags),
        gap_scan_skipped=tuple(skipped_scans),
    )


# ---------------------------------------------------------------------------
# tables and manifest
# ---------------------------------------------------------------------------


def ensure_event_tables(root: Path) -> list[Path]:
    """Write empty, typed ``events`` and ``adjustments`` tables where none exist.

    Both are written rather than omitted because a view reads only files the
    manifest lists, and "no events table" and "no events" are different
    statements: with no table, ``DatasetView.total_return`` refuses (§3.2).
    """
    root = Path(root)
    written: list[Path] = []
    for name, columns in (("events", EVENT_COLUMNS), ("adjustments", ADJUSTMENT_COLUMNS)):
        path = root / f"{name}.parquet"
        if not path.is_file():
            pd.DataFrame({column: pd.Series(dtype="object") for column in columns}).to_parquet(path, index=False)
        written.append(path)
    return written


def rebuild_manifest(
    root: Path,
    *,
    vintage: str,
    provenance: Literal["market", "synthetic", "mixed"],
    sources: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, Any] | None = None,
    assumptions: Mapping[str, Any] | None = None,
    transformations: Sequence[str] = (),
    universe: str | None = None,
) -> Manifest:
    """Rewrite ``manifest.json`` so it describes exactly the tables under ``root``.

    A manifest is written whole, never patched. Sources already declared by an
    earlier ingest are carried forward and keep their position in every
    ``source_priority`` order, so adding a vendor never silently reorders the
    ones already there; a source re-declared with different terms replaces its
    own entry, which changes the manifest id, which is the point.
    """
    root = Path(root)
    previous = None
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        previous = Manifest.read(manifest_path)
    merged: dict[str, dict[str, Any]] = {} if previous is None else {k: dict(v) for k, v in previous.sources.items()}
    order = list(merged)
    for source_id, entry in sources.items():
        merged[source_id] = dict(entry)
        if source_id not in order:
            order.append(source_id)
    files = sorted(
        path for path in root.rglob("*.parquet") if path.is_file() and _is_table_path(path.relative_to(root))
    )
    if not files:
        raise IngestError(f"{root} holds no tables to describe; an ingest that wrote nothing writes no manifest")
    as_of = _latest_bar_ts(root, files)
    tables = sorted({_table_name(path.relative_to(root)) for path in files})
    coverage_out = dict(coverage or {})
    assumptions_out = dict(assumptions or {})
    if previous is not None:
        carried = {k: v for k, v in previous.assumptions.items() if k not in assumptions_out}
        assumptions_out = {**carried, **assumptions_out}
        coverage_out = {**previous.coverage, **coverage_out}
    manifest = Manifest(
        provenance=provenance,
        as_of=as_of,
        files=tuple(hash_tree(root, files)),
        sources=merged,
        source_priority={table: list(order) for table in tables},
        universe=universe or _universe_hash(root, files),
        vintage=vintage,
        coverage=coverage_out,
        transformations=list(previous.transformations if previous is not None else []) + list(transformations),
        assumptions=assumptions_out,
    )
    manifest.write(manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_table_path(relative: Path) -> bool:
    if relative.parts[0] == "bars":
        return len(relative.parts) == 2
    return len(relative.parts) == 1


def _table_name(relative: Path) -> str:
    return "bars" if relative.parts[0] == "bars" else relative.stem


def _latest_bar_ts(root: Path, files: Sequence[Path]) -> str:
    """The manifest ``as_of``: the latest bar ``ts`` any bar file carries."""
    latest: pd.Timestamp | None = None
    for path in files:
        if path.parent.name != "bars":
            continue
        frame = pd.read_parquet(path, columns=["ts"])
        if frame.empty:
            continue
        stamps = pd.DatetimeIndex(pd.to_datetime(frame["ts"], utc=True)).tz_convert(BAR_TZ)
        candidate = stamps.max()
        latest = candidate if latest is None or candidate > latest else latest
    if latest is None:
        raise IngestError(
            f"{root} carries no bar rows, so the manifest has no as_of; §3.2 defines it as the latest bar ts"
        )
    return str(latest.isoformat())


def _universe_hash(root: Path, files: Sequence[Path]) -> str:
    """A hash of the sids the dataset holds bars for — the resolved eligible universe of §3.2."""
    sids = sorted(path.stem.rsplit("_", 1)[0] for path in files if path.parent.name == "bars")
    return sha256_bytes(canonical_json(sids).encode("utf-8"))


def _write_bars(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    # The ``ts`` column keeps a tz-aware datetime dtype rather than an ISO
    # string: the caches cross a daylight-saving boundary, and a column of
    # mixed-offset strings parses to object dtype, which the view's record-time
    # parse cannot use.
    out["ts"] = pd.DatetimeIndex(pd.to_datetime(out["ts"], utc=True)).tz_convert(BAR_TZ)
    out.to_parquet(path, index=False)
    return path


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in BAR_COLUMNS})


def _normalize_stored(stored: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in BAR_COLUMNS if column not in stored.columns]
    if missing:
        raise IngestError(f"the stored bar table is missing {missing}; it is not a §3.2 bars table")
    frame = stored.copy()
    frame["ts"] = pd.DatetimeIndex(pd.to_datetime(frame["ts"], utc=True)).tz_convert(BAR_TZ)
    frame["revision"] = frame["revision"].astype("int64")
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(float)
    for name in ("price_basis", "available_at", "ingested_at", "source"):
        frame[name] = frame[name].where(frame[name].notna(), other=None)
    return frame[list(BAR_COLUMNS)]


def _same_row(stored: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    return all(_same_value(stored.get(name), candidate.get(name)) for name in _COMPARED_FIELDS)


def _same_value(left: Any, right: Any) -> bool:
    left_null = left is None or (isinstance(left, float) and math.isnan(left)) or left is pd.NaT
    right_null = right is None or (isinstance(right, float) and math.isnan(right)) or right is pd.NaT
    if left_null or right_null:
        return left_null and right_null
    if isinstance(left, (int, float, np.floating)) and isinstance(right, (int, float, np.floating)):
        return float(left) == float(right)
    return str(left) == str(right)


def _float(value: Any) -> float:
    if value is None or value is pd.NaT:
        return float("nan")
    return float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _timestamps(values: pd.Series, *, timezone: str, where: Path | str) -> pd.DatetimeIndex:
    try:
        parsed = pd.to_datetime(values, errors="raise")
    except (ValueError, TypeError) as exc:
        raise IngestError(f"{where}: the declared ts column does not parse as timestamps: {exc}") from exc
    index = pd.DatetimeIndex(parsed)
    if index.isna().any():
        raise IngestError(f"{where}: the declared ts column carries an unparseable timestamp")
    if index.tz is None:
        index = index.tz_localize(timezone, nonexistent="raise", ambiguous="raise")
    return index.tz_convert(BAR_TZ)


def _symbol_column(frame: pd.DataFrame, mapping: IngestMapping, *, path: Path) -> pd.Series | None:
    if mapping.vendor_symbol is None:
        return None
    if mapping.vendor_symbol not in frame.columns:
        raise IngestError(
            f"{path} does not carry the [identity] vendor_symbol column {mapping.vendor_symbol!r}; the file's "
            f"columns are {sorted(frame.columns)}"
        )
    column = frame[mapping.vendor_symbol]
    text = column.astype("string").str.strip().str.upper()
    blank = column.isna() | text.isna() | (text == "")
    if bool(blank.any()):
        raise IngestError(
            f"{path}: {int(blank.sum())} of {len(frame)} rows carry no vendor symbol in column "
            f"{mapping.vendor_symbol!r}; a bar with no instrument cannot be resolved to a sid"
        )
    return text.astype(str)


def _symbol_from_filename(path: Path, interval: str) -> str:
    """``{SYMBOL}_{interval}...`` — the repository's own cache naming, used only when asked."""
    stem = path.stem
    marker = f"_{interval}"
    if marker in stem:
        return stem.split(marker)[0].upper()
    return stem.upper()


def _scannable(days: Sequence[dt.date]) -> bool:
    start, end = sessions.coverage()
    return bool(days) and start <= days[0] and days[-1] <= end


def _instant(value: str) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tz is None:
        stamp = stamp.tz_localize(BAR_TZ)
    return str(stamp.tz_convert(BAR_TZ).isoformat())


def _now_instant() -> str:
    return str(pd.Timestamp.now(tz=BAR_TZ).isoformat())


__all__ = [
    "BAR_COLUMNS",
    "BAR_FIELDS",
    "BarBatch",
    "IngestError",
    "IngestMapping",
    "IngestReport",
    "MAPPING_SCHEMA",
    "MATCH_RULES",
    "RevisionOutcome",
    "SourceDeclaration",
    "ensure_event_tables",
    "ingest_local_files",
    "merge_bar_revisions",
    "read_mapped_file",
    "rebuild_manifest",
    "split_by_symbol",
]
