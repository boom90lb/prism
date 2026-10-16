"""``prism data import-legacy``: this repository's own data as the tables of §3.2.

Two entry points live here, and they are separate on purpose.

:func:`build_legacy_dataset` is the **stage 2 slice**, described below and
unchanged: the ticker-keyed dataset the recorded reproduction of
``results/demotion_b1`` consumes. §3.2 says so in as many words — the legacy
import "is a stage 3 addition, since the tables it writes are defined there;
stage 2's reproduction of the recorded momentum and trend results reads the
caches directly through the existing loader and is unaffected by it" — so the
stage 3 command below is written beside it rather than through it. Run: rebuilt
with the stage 3 code present, the reproduction dataset hashes to the same
``data_id`` (``bedf8c4c0f0c2195ce5aace938f4ad38e23e6b69b57075080073004692d46eef``)
as the frozen one the certified reproduction runs on.

:func:`import_legacy` is the **stage 3 command**. It writes every table of §3.2
from everything the repository holds — the range-encoded caches, the incremental
store, the quarantined caches, the three membership files, the three reviewed
tables of :mod:`prism.io.universe_sp500`, and the issuer distribution histories
— minting surrogate sids through
:class:`prism.io.security_master.SecurityMaster` (decision D2) and folding
overlapping caches into revisions through
:func:`prism.io.local_files.merge_bar_revisions` (§3.2). Its own contract is
documented at the function.

--- the stage 2 slice ---

Stage 2 has to reproduce a recorded evaluation, and the evaluator of §3.8 reads
a :class:`~prism.io.dataset.Dataset`: a manifest-listed set of Parquet tables
with a content-hashed identity, which is the ``data_id`` every
:class:`~prism.validation.trials.ResultRecord` carries. The repository's own
market data is not in that shape — it is the range-encoded per-symbol cache
``data/{SYMBOL}_{interval}_{start}_{end}.parquet`` that
:class:`prism.io.loader.DataLoader` reads, plus the point-in-time membership
Parquet that ``scripts/build_sp500_universe.py`` writes.

That slice writes the second from the first, for the two tables a stage 2
reproduction consumes — ``bars`` and ``membership`` — plus the minimal
``listings``, ``securities`` and ``events`` tables the view's accessors need to
answer at all. It is deliberately *not* the whole legacy import: the security
master, the events and adjustments derived from ``RENAME_TABLE``,
``QUARANTINE_TABLE`` and ``CHANGES_PATCH_TABLE``, the distribution CSVs, the
revision ordering across overlapping caches, and sid minting are
:func:`import_legacy`'s (§3.2, "Legacy import"). What the slice does is read the
caches through the existing loader, which is what §3.2 says a stage 2
reproduction does; it mints no identifier beyond the ticker and claims no
provenance the files do not carry.

Two conventions are declared rather than inferred, and both are recorded in the
manifest's ``assumptions``:

*Sessions.* The recorded driver dropped non-trading dates from the union of the
per-symbol calendars with ``prism.residual.factors.consensus_trading_days`` — a
presence heuristic, because it had no session table. This import filters on
``prism.io.sessions`` instead, which is derived from the exchange's published
rules. On the recorded momentum panel the two agree exactly: 1,621 sessions
from 2020-01-02 to 2026-06-15, and every one of the 148 dates the heuristic
dropped is a non-session under the table (Run;
``tests/test_sessions.py::test_the_recorded_momentum_panel_agrees_with_the_table``).

*Membership intervals.* They are translated verbatim, ``[start, end]`` closed,
because that is what ``prism.io.universe_sp500.build_membership_mask`` applies
and reproducing it is the point. Stage 3's import turns an ``end`` equal to the
file's as-of into an open interval; that rewrite changes no day inside any
recorded sample and belongs to :func:`import_legacy`, not here.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from prism.io import sessions
from prism.io.dataset import BAR_TZ, DEFAULT_BAR_INTERVAL
from prism.io.events import (
    ISSUER_SOURCE,
    Event,
    adjustments_frame,
    events_frame,
    issuer_distribution_events,
)
from prism.io.local_files import (
    BarBatch,
    IngestMapping,
    RevisionOutcome,
    merge_bar_revisions,
    read_mapped_file,
    rebuild_manifest,
)
from prism.io.manifest import Manifest, canonical_json, hash_tree, sha256_bytes
from prism.io.security_master import (
    ATTESTATION_SOURCE,
    DEFAULT_GAP_FLAG_SESSIONS,
    OVERRIDES_FILENAME,
    OVERRIDES_SCHEMA,
    Attestation,
    SecurityMaster,
    SecurityOverrides,
)
from prism.io.universe_sp500 import (
    CHANGES_PATCH_TABLE,
    QUARANTINE_TABLE,
    RENAME_TABLE,
    normalize_ticker,
)

logger = logging.getLogger(__name__)

#: The source id every imported bar row carries. Vendor bars reach this
#: workspace split-adjusted and not dividend-adjusted
#: (``prism.io.loader.DataLoader.fetch_historical_data``), which is the
#: ``split_adjusted_at_ingest`` price basis of §3.2.
BAR_SOURCE = "legacy_cache"

#: The source id of the point-in-time membership table.
MEMBERSHIP_SOURCE = "universe_build"

#: Neither source may be redistributed: the bars are licensed to this workspace
#: by the vendor, and the membership file is derived from them plus a scraped
#: constituent history (§3.2, "Non-redistributable inputs").
REDISTRIBUTABLE = False

#: Bars and membership rows are knowable at the close of the session they
#: describe: the caches hold end-of-day bars and the membership file is a
#: reconstruction of intervals, not a feed.
LAG_BARS = 0

TABLE_NAMES: tuple[str, ...] = ("securities", "listings", "membership", "events")

_BAR_COLUMNS: tuple[str, ...] = (
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


class LegacyImportError(ValueError):
    """The caches cannot be imported as asked, and guessing would forge provenance."""


@dataclass(frozen=True)
class LegacyImportReport:
    """What one import wrote, and what it refused to write."""

    root: Path
    manifest: Manifest
    sids: tuple[str, ...]
    sessions: tuple[str, ...]
    requested: tuple[str, ...]
    without_bars: tuple[str, ...]
    dropped_non_sessions: int
    membership_intervals: int
    membership_sids: tuple[str, ...]
    files: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "data_id": self.manifest.id,
            "vintage": self.manifest.vintage,
            "as_of": self.manifest.as_of,
            "n_sids": len(self.sids),
            "n_sessions": len(self.sessions),
            "first_session": self.sessions[0] if self.sessions else None,
            "last_session": self.sessions[-1] if self.sessions else None,
            "n_requested": len(self.requested),
            "without_bars": list(self.without_bars),
            "dropped_non_sessions": int(self.dropped_non_sessions),
            "membership_intervals": int(self.membership_intervals),
            "n_membership_sids": len(self.membership_sids),
            "n_files": len(self.files),
        }


def read_legacy_panels(
    symbols: Sequence[str],
    *,
    cache_dir: Path,
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Per-symbol OHLCV frames from the range-encoded caches, through the existing loader.

    The loader's own file selection is used unchanged — the first covering
    file in sorted-name order — so the panel is the one the recorded driver read,
    `research/scripts/stat_arb_residual_wfo.py:228-243` (removed; last present at commit 5fcac33869e9),
    duplicated here with this citation because ``prism`` imports no research
    module. A symbol the cache cannot answer for is reported, never silently
    skipped.
    """
    from prism.io.loader import (
        DataLoader,  # local: constructing one reads the environment
    )

    loader = DataLoader(api_key="", cache_dir=Path(cache_dir))
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = loader.fetch_historical_data(symbol, DEFAULT_BAR_INTERVAL, start, end)
        if frame.empty:
            continue
        missing = {"open", "high", "low", "close", "volume"} - set(frame.columns)
        if missing:
            raise LegacyImportError(
                f"the cache for {symbol!r} is missing {sorted(missing)}; a bar table with no price column is a "
                "defect in the cache, not an empty result (N7)"
            )
        if not frame.index.is_unique:
            # The vendor occasionally returns an overlapping bar; the driver
            # kept the last, and a duplicate index cannot be pivoted.
            frame = frame[~frame.index.duplicated(keep="last")]
        frames[symbol] = frame.sort_index()
    return frames


def panel_matrices(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(closes, opens, volumes)`` on the union calendar, with no session filter.

    The recorded driver's own panel construction,
    `research/scripts/stat_arb_residual_wfo.py::_panel_matrices` (removed; last present at commit 5fcac33869e9),
    duplicated here with that citation because ``prism`` imports no research
    module: the close panel drops rows where every name is absent and the other
    two are reindexed onto it, so all three share one index.

    No trading-day filter is applied. That is deliberate — this is the union
    calendar a recorded run consumed, and a diagnostic recomputing something
    that run reported has to see the same rows. :func:`session_index` is the
    filter, and :func:`build_legacy_dataset` is where it belongs.
    """
    closes = pd.DataFrame({sid: frame["close"].astype(float) for sid, frame in frames.items()}).sort_index()
    closes = closes.dropna(how="all")
    opens = pd.DataFrame({sid: frame["open"].astype(float) for sid, frame in frames.items()}).reindex(closes.index)
    volumes = pd.DataFrame({sid: frame["volume"].astype(float) for sid, frame in frames.items()}).reindex(closes.index)
    return closes, opens, volumes


def session_index(frames: dict[str, pd.DataFrame]) -> tuple[list[pd.Timestamp], int]:
    """The exchange sessions the union of the per-symbol calendars covers, and how many dates were dropped."""
    union = pd.DatetimeIndex(sorted({ts for frame in frames.values() for ts in frame.index}))
    if len(union) == 0:
        raise LegacyImportError("the caches carry no bars at all; there is nothing to import")
    kept = [ts for ts in union if sessions.is_session(_day(ts))]
    if not kept:
        raise LegacyImportError(
            f"none of the {len(union)} cached dates is a session on the bundled table "
            f"({sessions.TABLE_VERSION}); the table and the caches disagree about the calendar"
        )
    return kept, len(union) - len(kept)


def build_legacy_dataset(
    root: Path,
    *,
    symbols: Sequence[str],
    cache_dir: Path,
    start: str,
    end: str,
    vintage: str,
    index_id: str,
    membership_file: Path | None = None,
    universe_note: str | None = None,
    coverage: dict[str, object] | None = None,
    extra_indexes: Mapping[str, Path] | None = None,
) -> LegacyImportReport:
    """Write a manifest-listed dataset over the legacy caches and return its report.

    ``extra_indexes`` maps further index ids to their membership files; their
    rows are written to the same ``membership`` table under their own
    ``index_id``, so one dataset (one set of bars, one ``data_id``) can serve
    arms that differ only in the index they trade.

    ``symbols`` is the resolved universe of the run being reproduced. Every sid
    is its ticker: the caches are ticker-keyed and this import mints no
    identifier, which is why every imported security is ``provisional`` in the
    ``securities`` table it writes.
    """
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise LegacyImportError(
            f"{root} already holds files; an import writes a dataset whole so its manifest describes exactly what "
            "is there. Remove the directory or name another one."
        )
    requested = [str(s) for s in dict.fromkeys(symbols)]
    frames = read_legacy_panels(requested, cache_dir=cache_dir, start=start, end=end)
    without_bars = tuple(s for s in requested if s not in frames)
    if not frames:
        raise LegacyImportError(
            f"none of the {len(requested)} requested symbols has a cached bar file under {cache_dir}"
        )
    index, dropped = session_index(frames)
    labels = tuple(_day(ts).isoformat() for ts in index)
    ingested_at = _mtime_instant(cache_dir)

    (root / "bars").mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    sids = sorted(frames)
    for sid in sids:
        path = root / "bars" / f"{sid}_{DEFAULT_BAR_INTERVAL}.parquet"
        _bar_frame(frames[sid], index, ingested_at=ingested_at).to_parquet(path, index=False)
        written.append(path)

    membership_rows, n_intervals = _membership_rows(
        membership_file,
        sids,
        index_id=index_id,
        first_session=_day(index[0]),
        last_session=_day(index[-1]),
    )
    index_sids: dict[str, list[str]] = {
        index_id: sorted(set(membership_rows["sid"].astype(str))) if len(membership_rows) else []
    }
    membership_hashes: dict[str, str | None] = {
        index_id: None if membership_file is None else _sha256_file(Path(membership_file))
    }
    for extra_id, extra_file in dict(extra_indexes or {}).items():
        if extra_id in index_sids:
            raise LegacyImportError(f"index id {extra_id!r} is declared twice")
        extra_rows, extra_n = _membership_rows(
            Path(extra_file),
            sids,
            index_id=extra_id,
            first_session=_day(index[0]),
            last_session=_day(index[-1]),
        )
        membership_rows = pd.concat([membership_rows, extra_rows], ignore_index=True)
        n_intervals += extra_n
        index_sids[extra_id] = sorted(set(extra_rows["sid"].astype(str))) if len(extra_rows) else []
        membership_hashes[extra_id] = _sha256_file(Path(extra_file))
    tables = {
        "securities": _securities_rows(sids, ingested_at=ingested_at),
        "listings": _listing_rows(sids, ingested_at=ingested_at),
        "membership": membership_rows,
        "events": _empty_events(),
    }
    for name, rows in tables.items():
        path = root / f"{name}.parquet"
        rows.to_parquet(path, index=False)
        written.append(path)

    membership_sids = tuple(index_sids[index_id])
    manifest = Manifest(
        provenance="market",
        as_of=str(pd.Timestamp(index[-1]).isoformat()),
        files=tuple(hash_tree(root, written)),
        sources={
            BAR_SOURCE: {
                "lag_bars": LAG_BARS,
                "revision_policy": "conflict_raise",
                "price_basis": "split_adjusted_at_ingest",
                "url": None,
                "sha256": None,
                "redistributable": REDISTRIBUTABLE,
            },
            MEMBERSHIP_SOURCE: {
                "lag_bars": LAG_BARS,
                "revision_policy": "conflict_raise",
                "price_basis": "raw",
                "url": None,
                "sha256": None if membership_file is None else _sha256_file(Path(membership_file)),
                "redistributable": REDISTRIBUTABLE,
            },
        },
        source_priority={table: [BAR_SOURCE, MEMBERSHIP_SOURCE] for table in (*TABLE_NAMES, "bars")},
        universe=sha256_bytes(
            canonical_json(
                {index_id: list(membership_sids)} if not extra_indexes else {k: list(v) for k, v in index_sids.items()}
            ).encode("utf-8")
        ),
        vintage=vintage,
        coverage={
            "sessions": [labels[0], labels[-1]],
            "n_sessions": len(labels),
            "index_membership": (
                {index_id: len(membership_sids)} if not extra_indexes else {k: len(v) for k, v in index_sids.items()}
            ),
            "requested_symbols": len(requested),
            "symbols_without_bars": list(without_bars),
            "dropped_non_session_dates": int(dropped),
            "universe_note": universe_note,
            "recorded_coverage": dict(coverage or {}),
        },
        transformations=[
            "legacy_import_v1",
            f"cache_dir={Path(cache_dir).name}",
            f"span={start}..{end}",
            f"session_table={sessions.TABLE_VERSION}",
        ],
        assumptions={
            "reader": (
                "bars read through prism.io.loader.DataLoader, whose file selection is the first covering "
                "range-encoded cache in sorted-name order; duplicate bars for one date keep the last"
            ),
            "ingestion_time_quality": "file_mtime",
            "availability": "available_at is empty on every row; availability derives from the session table",
            "session_table_version": sessions.TABLE_VERSION,
            "session_filter": (
                "union dates that are not sessions on the bundled table are dropped before any estimator sees "
                "them; on the recorded momentum panel this reproduces "
                "prism.residual.factors.consensus_trading_days exactly"
            ),
            "membership_intervals": (
                "translated as closed [valid_from, valid_to]; an interval covering no imported session is "
                "not written, an interval opening on or before the first session becomes floor_unknown "
                "(valid_from null), and no as-of end is opened"
            ),
            "sid_minting": "sid = ticker; every security is provisional and no identifier is minted",
            **({"membership_files_sha256": membership_hashes} if extra_indexes else {}),
            "not_imported": (
                "adjustments, cash distributions, the rename/quarantine/changes tables and revision ordering "
                "across overlapping caches arrive with prism data import-legacy at stage 3"
            ),
        },
    )
    manifest.write(root / "manifest.json")
    logger.info(
        "imported %d sid(s) over %d session(s) into %s (data_id %s)", len(sids), len(labels), root, manifest.id
    )
    return LegacyImportReport(
        root=root,
        manifest=manifest,
        sids=tuple(sids),
        sessions=labels,
        requested=tuple(requested),
        without_bars=without_bars,
        dropped_non_sessions=int(dropped),
        membership_intervals=int(n_intervals),
        membership_sids=membership_sids,
        files=tuple(sorted(str(p.relative_to(root)) for p in written)),
    )


# ---------------------------------------------------------------------------
# table builders
# ---------------------------------------------------------------------------


def _bar_frame(frame: pd.DataFrame, index: Iterable[pd.Timestamp], *, ingested_at: str) -> pd.DataFrame:
    """One sid's bars on the session index, in the ``bars`` schema of §3.2.

    Rows the symbol has no bar for are absent rather than NaN-filled: a missing
    bar is missing, and the panel reader reindexes it to NaN, which is what the
    driver's union panel held.
    """
    aligned = frame.reindex(pd.DatetimeIndex(index))
    aligned = aligned.loc[aligned["close"].notna()]
    out = pd.DataFrame(
        {
            "ts": aligned.index,
            "open": pd.to_numeric(aligned["open"], errors="coerce").to_numpy(dtype=float),
            "high": pd.to_numeric(aligned["high"], errors="coerce").to_numpy(dtype=float),
            "low": pd.to_numeric(aligned["low"], errors="coerce").to_numpy(dtype=float),
            "close": pd.to_numeric(aligned["close"], errors="coerce").to_numpy(dtype=float),
            "volume": pd.to_numeric(aligned["volume"], errors="coerce").to_numpy(dtype=float),
        }
    )
    out["price_basis"] = "split_adjusted_at_ingest"
    out["available_at"] = None
    out["ingested_at"] = ingested_at
    out["source"] = BAR_SOURCE
    out["revision"] = 1
    return out[list(_BAR_COLUMNS)].reset_index(drop=True)


def _membership_rows(
    membership_file: Path | None,
    sids: Sequence[str],
    *,
    index_id: str,
    first_session: dt.date | None = None,
    last_session: dt.date | None = None,
) -> tuple[pd.DataFrame, int]:
    """The ``membership`` table, restricted to sids the import wrote bars for.

    Restriction is not a screen on the index: it is the same restriction the
    recorded run made when it resolved 574 of 633 ever-members and recorded the
    other 59 as a coverage skip-list. A membership row for a sid with no bar
    file would make ``DatasetView.bars`` refuse for a name the universe names.

    Two translations, both of which leave every in-sample membership decision
    unchanged and both of which are recorded in the manifest's assumptions.

    An interval that closes before the first session or opens after the last
    one covers no day the dataset holds and is not written: the dataset covers
    the sample, and a row about a period outside it would be a claim the bars
    cannot support.

    An interval opening on or before ``first_session`` is written with
    ``valid_from = None``, which §3.2 calls ``floor_unknown``: the interval
    opened before the sample. The source file floors those starts at a history
    sentinel (1990-01-01, and older on a few rows), and a record time that far
    outside the session table is not a date the availability derivation can
    walk. An interval starting before the first session contains every sample
    day it would have contained anyway.

    Two surviving intervals for one sid that both open before the sample would
    overlap inside it, which is a defect in the source rather than a revision
    to choose between; it is refused here rather than deduplicated.
    """
    columns = [
        "index_id",
        "sid",
        "valid_from",
        "valid_to",
        "announce_date",
        "floor_unknown",
        "source",
        "ingested_at",
        "revision",
    ]
    if membership_file is None:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns}), 0
    frame = pd.read_parquet(Path(membership_file))
    for required in ("ticker", "start", "end"):
        if required not in frame.columns:
            raise LegacyImportError(
                f"{membership_file} has no {required!r} column; the membership file this import reads is the one "
                "prism.scripts.build_sp500_universe writes, with [ticker, start, end]"
            )
    wanted = set(sids)
    seen: set[tuple[str, dt.date | None]] = set()
    rows: list[dict[str, object]] = []
    ingested_at = _mtime_instant(Path(membership_file))
    for record in frame.itertuples(index=False):
        sid = normalize_ticker(str(record.ticker))
        if sid not in wanted:
            continue
        opened = pd.Timestamp(record.start).date()
        closed = pd.Timestamp(record.end).date()
        if first_session is not None and closed < first_session:
            continue
        if last_session is not None and opened > last_session:
            continue
        floor_unknown = first_session is not None and opened <= first_session
        key = (sid, None if floor_unknown else opened)
        if key in seen:
            raise LegacyImportError(
                f"{membership_file} carries two intervals for {sid!r} that both open on or before the first "
                f"imported session ({first_session}); inside the sample they overlap, which is a defect in the "
                "source, not a revision to choose between"
            )
        seen.add(key)
        rows.append(
            {
                "index_id": index_id,
                "sid": sid,
                "valid_from": None if floor_unknown else opened.isoformat(),
                "valid_to": pd.Timestamp(record.end).date().isoformat(),
                "announce_date": None,
                "floor_unknown": bool(floor_unknown),
                "source": MEMBERSHIP_SOURCE,
                "ingested_at": ingested_at,
                "revision": 1,
            }
        )
    if not rows:
        raise LegacyImportError(
            f"{membership_file} names none of the {len(sids)} imported sids; a membership table that gates every "
            "name out is a defect, not an empty universe (N7)"
        )
    return pd.DataFrame(rows, columns=columns), len(frame)


def _securities_rows(sids: Sequence[str], *, ingested_at: str) -> pd.DataFrame:
    """One ``securities`` row per sid, every one ``provisional`` (§3.2, "Legacy import")."""
    return pd.DataFrame(
        {
            "sid": list(sids),
            "valid_at": [_EPOCH] * len(sids),
            "asset_class": ["equity"] * len(sids),
            "entity_status": ["provisional"] * len(sids),
            "currency": ["USD"] * len(sids),
            "source": [BAR_SOURCE] * len(sids),
            "ingested_at": [ingested_at] * len(sids),
            "revision": [1] * len(sids),
        }
    )


def _listing_rows(sids: Sequence[str], *, ingested_at: str) -> pd.DataFrame:
    """One open ``listings`` interval per sid, ticker equal to the sid.

    ``valid_from`` is null, which §3.2 calls the ``floor_unknown`` case: the
    cache does not record when the listing opened, and inventing a date would
    make the listing-seasoning gate answer a question the data cannot.
    """
    return pd.DataFrame(
        {
            "sid": list(sids),
            "vendor": [BAR_SOURCE] * len(sids),
            "ticker": list(sids),
            "valid_from": [None] * len(sids),
            "valid_to": [None] * len(sids),
            "announce_date": [None] * len(sids),
            "source": [BAR_SOURCE] * len(sids),
            "ingested_at": [ingested_at] * len(sids),
            "revision": [1] * len(sids),
        }
    )


def _empty_events() -> pd.DataFrame:
    """An empty, typed ``events`` table.

    It is written rather than omitted because the view reads only files the
    manifest lists, and "no events table" and "no events" are different
    statements. This import applies no corporate action, which the manifest's
    ``not_imported`` assumption says out loud.
    """
    columns = ["sid", "event_type", "effective_date", "announce_date", "payload", "source", "ingested_at", "revision"]
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_EPOCH = "1970-01-01T00:00:00-05:00"


def _day(ts: pd.Timestamp) -> dt.date:
    stamp = pd.Timestamp(ts)
    if stamp.tz is not None:
        stamp = stamp.tz_convert(BAR_TZ)
    return stamp.date()


def _mtime_instant(path: Path) -> str:
    """The source's modification time, which §3.2 declares as the imported ingestion time."""
    stamp = pd.Timestamp(Path(path).stat().st_mtime, unit="s", tz="UTC").tz_convert(BAR_TZ)
    return str(stamp.isoformat())


def _sha256_file(path: Path) -> str:
    from prism.io.manifest import sha256_file

    return sha256_file(Path(path))


# ===========================================================================
# The stage 3 import: `prism data import-legacy` (§3.2, "Legacy import")
# ===========================================================================

#: The vendor id every imported bar row carries. All three bar locations — the
#: range caches, ``data/store`` and ``data/quarantine`` — are the same vendor's
#: split-adjusted series, so they share a source and their overlapping rows are
#: revisions of one another rather than rows of two sources that a priority
#: order would have to choose between. §3.2 names the id: "``import-legacy``
#: writes ``sources.twelvedata.lag_bars = 0``".
LEGACY_VENDOR = "twelvedata"

#: The source id of the issuer distribution histories (§3.2), matching
#: :data:`prism.io.events.ISSUER_SOURCE`.
DISTRIBUTION_SOURCE = ISSUER_SOURCE

#: The source id of the events derived from the three reviewed tables in
#: ``src/prism/io/universe_sp500.py``. They are a human review recorded in
#: code, not a vendor feed, and §3.2 says they "become data, not code
#: branches".
REVIEWED_SOURCE = "reviewed_tables"

#: The prefix of a membership source id. Each membership file is its own
#: source ``wikipedia@{asof}`` (§3.2), ordered newest first in the manifest.
MEMBERSHIP_SOURCE_PREFIX = "wikipedia@"

#: The date the three reviewed tables were reviewed, and the evidence that
#: records the review. Both are read off the tables themselves
#: (``src/prism/io/universe_sp500.py:52`` "Reviewed 2026-07-14
#: (docs/data_integrity_diagnostic.md §6)") and the diagnostic's own §7,
#: "Remediation executed (2026-07-14, owner decision)". Nothing here dates an
#: attestation from anywhere else.
REVIEW_DATE = dt.date(2026, 7, 14)
REVIEW_ATTESTED_BY = "owner"
QUARANTINE_EVIDENCE = "docs/data_integrity_diagnostic.md §3"

#: The reviewed ticker changes that carry an effective date. ``RENAME_TABLE``
#: is a dateless ticker-to-vendor-symbol remap — it is applied that way by
#: ``prism.io.universe_sp500.resolve_coverage`` — and only the EchoStar entry
#: states an effective date in the reviewed record (2026-06-24, company release
#: 2026-06-22; ``universe_sp500.py:60`` and ``docs/data_integrity_diagnostic.md
#: §8). An undated rename is recorded in the manifest ``coverage`` as a
#: remapped ticker and is *not* written as a ``ticker_change`` event, because an
#: event's key is its effective date and inventing one would date a corporate
#: action from nothing.
DATED_TICKER_CHANGES: dict[str, dict[str, str]] = {
    "SATS": {
        "to_ticker": "ECHO",
        "effective_date": "2026-06-24",
        "announce_date": "2026-06-22",
        "evidence": "docs/data_integrity_diagnostic.md §8",
    },
}

#: The column mapping the repository's own caches are written in: a tz-aware
#: ``DatetimeIndex`` named ``datetime`` plus the five OHLCV columns (Run over
#: ``data/AAPL_1d_2020-01-01_2026-06-16.parquet``). It is declared here in the
#: same shape ``prism data ingest`` reads from a mapping file, so the legacy
#: import and a local ingest read a file through one code path.
LEGACY_MAPPING_TOML: str = """
schema = 1

[source]
id = "twelvedata"
kind = "vendor_cache"
lag_bars = 0
revision_policy = "dedupe_identical"
price_basis = "split_adjusted_at_ingest"
timezone = "America/New_York"
gap_flag_sessions = 15
redistributable = false

[columns]
ts = "datetime"
open = "open"
high = "high"
low = "low"
close = "close"
volume = "volume"

[identity]
match = "ticker_interval_overlap"
"""


def legacy_mapping(*, gap_flag_sessions: int = DEFAULT_GAP_FLAG_SESSIONS) -> IngestMapping:
    """The declared mapping the caches are read under."""
    mapping = IngestMapping.from_dict(tomllib.loads(LEGACY_MAPPING_TOML))
    if gap_flag_sessions == mapping.source.gap_flag_sessions:
        return mapping
    return replace(mapping, source=replace(mapping.source, gap_flag_sessions=int(gap_flag_sessions)))


@dataclass(frozen=True)
class LegacyInputs:
    """Every input §3.2 enumerates, found on disk and counted."""

    cache_dir: Path
    interval: str
    files: dict[str, tuple[Path, ...]]
    quarantined_symbols: tuple[str, ...]
    stored_symbols: tuple[str, ...]
    unranged: tuple[Path, ...]
    other_intervals: tuple[str, ...]
    membership_files: tuple[Path, ...]
    coverage_files: tuple[Path, ...]
    distribution_symbols: tuple[str, ...]
    #: The per-symbol file list the caller pinned, when it pinned one. Empty
    #: when the caches were discovered by glob, so a dataset built the
    #: discovered way keeps the identity it already has.
    pinned_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        pinned: dict[str, object] = {"pinned_cache_files": list(self.pinned_files)} if self.pinned_files else {}
        return {
            **pinned,
            "cache_dir": str(self.cache_dir),
            "interval": self.interval,
            "n_symbols": len(self.files),
            "n_files": sum(len(paths) for paths in self.files.values()),
            "n_multi_file_symbols": sum(1 for paths in self.files.values() if len(paths) > 1),
            "quarantined_symbols": list(self.quarantined_symbols),
            "stored_symbols": list(self.stored_symbols),
            "unranged_files_not_imported": [p.name for p in self.unranged],
            "other_interval_groups": list(self.other_intervals),
            "membership_files": [p.name for p in self.membership_files],
            "coverage_files": [p.name for p in self.coverage_files],
            "distribution_symbols": list(self.distribution_symbols),
        }


def discover_legacy_inputs(
    cache_dir: Path | str,
    *,
    interval: str = DEFAULT_BAR_INTERVAL,
    cache_files: Mapping[str, Sequence[str]] | None = None,
) -> LegacyInputs:
    """Enumerate the inputs of §3.2's "Legacy import", in mtime order per symbol.

    The four bar locations are enumerated separately because they mean
    different things. The range-encoded caches and ``data/store`` are the live
    read paths; ``data/quarantine`` holds the caches the integrity sweep pulled
    out of the universe, which are imported because the certified record that
    stage 2 reproduces consumed them (§3.2) and are marked quarantined rather
    than pretended absent; the un-ranged ``{SYMBOL}_{interval}.parquet`` files
    are *not* imported, because their coverage is unknown and the loader
    already skips them (``prism.io.loader:71-79``), and their count is recorded.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise LegacyImportError(f"{cache_dir} is not a directory; there is nothing to import")
    if cache_files is not None:
        return _pinned_legacy_inputs(cache_dir, interval=interval, cache_files=cache_files)
    files: dict[str, list[Path]] = {}
    other: set[str] = set()
    unranged: list[Path] = []
    for path in sorted(cache_dir.glob("*.parquet")):
        symbol, found = _parse_cache_name(path.name)
        if symbol is None:
            continue
        if found is None:
            unranged.append(path)
            continue
        if found != interval:
            other.add(f"{symbol}_{found}")
            continue
        files.setdefault(symbol, []).append(path)
    quarantined: set[str] = set()
    for path in sorted((cache_dir / "quarantine").glob("*.parquet")):
        symbol, found = _parse_cache_name(path.name)
        if symbol is None or found != interval:
            continue
        files.setdefault(symbol, []).append(path)
        quarantined.add(symbol)
    stored: set[str] = set()
    for path in sorted((cache_dir / "store").glob(f"*_{interval}_bars.parquet")):
        symbol = path.name[: -len(f"_{interval}_bars.parquet")]
        files.setdefault(symbol, []).append(path)
        stored.add(symbol)
    universe = cache_dir / "universe"
    membership = tuple(sorted(universe.glob("sp500_membership_*.parquet"), key=_mtime))
    coverage = tuple(sorted(universe.glob("sp500_coverage_*.json")))
    distributions = cache_dir / "distributions"
    symbols = tuple(sorted(p.stem for p in distributions.glob("*.csv"))) if distributions.is_dir() else ()
    return LegacyInputs(
        cache_dir=cache_dir,
        interval=interval,
        files={symbol: tuple(sorted(paths, key=_mtime)) for symbol, paths in sorted(files.items())},
        quarantined_symbols=tuple(sorted(quarantined)),
        stored_symbols=tuple(sorted(stored)),
        unranged=tuple(unranged),
        other_intervals=tuple(sorted(other)),
        membership_files=membership,
        coverage_files=coverage,
        distribution_symbols=symbols,
    )


def _pinned_legacy_inputs(
    cache_dir: Path,
    *,
    interval: str,
    cache_files: Mapping[str, Sequence[str]],
) -> LegacyInputs:
    """The inputs a caller pinned by name, in the order it pinned them.

    A recorded configuration that names its cache files is reproduced from
    those files and from nothing else: ``results/trend_t0/config.json``
    ``cache_files``, under the "Explicit file lists only; never glob" rule of
    ``research/arbitrage/trend_walk_forward.py`` (removed; last present at commit 5fcac33869e9).
    The glob would be wrong here in a way no number would show: the live cache holds a
    dozen later ``SPY_1d_min_*.parquet`` files the recorded run never read, and
    a discovery that swept them in would build a different panel under the
    recorded name. Every named file must exist and must parse as that symbol at
    this interval; a missing or misfiled name is a refusal, never a smaller
    import.

    The declared order is preserved rather than sorted by mtime, because the
    order is what decides which file wins where two overlap.
    """
    files: dict[str, list[Path]] = {}
    for symbol, names in cache_files.items():
        key = normalize_ticker(symbol)
        listed = [str(name) for name in names]
        if not listed:
            raise LegacyImportError(
                f"{symbol!r} pins an empty cache-file list; a symbol with no file is not an import"
            )
        paths: list[Path] = []
        for name in listed:
            path = cache_dir / name
            if not path.is_file():
                raise LegacyImportError(
                    f"{symbol!r} pins {name!r}, which is not a file under {cache_dir}; the reproduction reads the "
                    "files the record names and mints nothing in their place"
                )
            found_symbol, found_interval = _parse_cache_name(path.name)
            if found_symbol != key or found_interval != interval:
                raise LegacyImportError(
                    f"{symbol!r} pins {name!r}, which names {found_symbol!r} at interval {found_interval!r}; a "
                    "pinned file is filed under the symbol its name carries or the panel is silently wrong"
                )
            paths.append(path)
        files[key] = paths
    distributions = cache_dir / "distributions"
    universe = cache_dir / "universe"
    return LegacyInputs(
        cache_dir=cache_dir,
        interval=interval,
        files={symbol: tuple(paths) for symbol, paths in sorted(files.items())},
        quarantined_symbols=(),
        stored_symbols=(),
        unranged=(),
        other_intervals=(),
        membership_files=(
            tuple(sorted(universe.glob("sp500_membership_*.parquet"), key=_mtime)) if universe.is_dir() else ()
        ),
        coverage_files=tuple(sorted(universe.glob("sp500_coverage_*.json"))) if universe.is_dir() else (),
        distribution_symbols=(
            tuple(sorted(p.stem for p in distributions.glob("*.csv"))) if distributions.is_dir() else ()
        ),
        pinned_files=tuple(f"{symbol}:{name}" for symbol, names in sorted(cache_files.items()) for name in names),
    )


@dataclass(frozen=True)
class LegacyDatasetReport:
    """What one ``prism data import-legacy`` wrote, and what it refused to write."""

    root: Path
    manifest: Manifest
    inputs: LegacyInputs
    symbol_to_sid: dict[str, str]
    rows_written: int
    rows_deduped: int
    rows_revised: int
    duplicate_bars_dropped: int
    skipped_series: tuple[dict[str, str], ...]
    truncated_series: tuple[dict[str, object], ...]
    membership: tuple[dict[str, object], ...]
    remapped_tickers: tuple[dict[str, str], ...]
    quarantined: tuple[dict[str, str], ...]
    events_written: int
    events_outside_session_table: tuple[dict[str, object], ...]
    gap_flags: tuple[dict[str, object], ...]
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "data_id": self.manifest.id,
            "vintage": self.manifest.vintage,
            "as_of": self.manifest.as_of,
            "inputs": self.inputs.to_dict(),
            "n_sids": len(set(self.symbol_to_sid.values())),
            "rows_written": int(self.rows_written),
            "rows_deduped": int(self.rows_deduped),
            "rows_revised": int(self.rows_revised),
            "duplicate_bars_dropped": int(self.duplicate_bars_dropped),
            "skipped_series": [dict(entry) for entry in self.skipped_series],
            "truncated_series": [dict(entry) for entry in self.truncated_series],
            "membership": [dict(entry) for entry in self.membership],
            "remapped_tickers": [dict(entry) for entry in self.remapped_tickers],
            "quarantined": [dict(entry) for entry in self.quarantined],
            "events_written": int(self.events_written),
            "events_outside_session_table": [dict(entry) for entry in self.events_outside_session_table],
            "n_gap_flags": len(self.gap_flags),
            "gap_flags": [dict(flag) for flag in self.gap_flags],
            "n_files": len(self.files),
        }


def import_legacy(
    root: Path | str,
    *,
    cache_dir: Path | str,
    vintage: str,
    index_id: str = "sp500",
    interval: str = DEFAULT_BAR_INTERVAL,
    symbols: Sequence[str] | None = None,
    membership_files: Sequence[Path] | None = None,
    distributions_dir: Path | str | None = None,
    cache_files: Mapping[str, Sequence[str]] | None = None,
    gap_flag_sessions: int = DEFAULT_GAP_FLAG_SESSIONS,
) -> LegacyDatasetReport:
    """Write the tables of §3.2 from what this repository already holds.

    The mappings §3.2 states in words, and where each is enforced. Ingestion
    time is each source file's modification time (:func:`_mtime_instant`);
    ``available_at`` is empty on every row, so availability is derived; every
    minted sid starts ``provisional``; and every bar is labelled
    ``split_adjusted_at_ingest`` — decision D3, because both vendors in this
    repository fetch split-adjusted, dividend-unadjusted bars (``loader.py``
    passes ``adjust=splits``; ``live/alpaca_data.py`` defaults to
    ``adjustment="split"``), so no share change is applied on a split date.

    Four decisions this import makes that §3.2 leaves to it, each recorded in
    the manifest's ``assumptions`` and, where it is a count, in ``coverage``:

    *The reviewed tables become data, not code branches.* ``QUARANTINE_TABLE``
    becomes a ``security_overrides.toml`` attestation per sid, dated by the
    review the tables themselves record; a quarantined ticker resolves to no
    index seat, because the vendor series under it is a known *different*
    instrument, while its bars stay readable so an earlier record that consumed
    them still reproduces. ``RENAME_TABLE`` is a dateless remap and is applied
    as one, exactly as ``prism.io.universe_sp500.resolve_coverage`` applies it;
    only a rename whose effective date the reviewed record states — the
    2026-06-24 EchoStar change — is written as a ``ticker_change`` event, since
    an event's key is its effective date and inventing one would date a
    corporate action from nothing. ``CHANGES_PATCH_TABLE`` becomes a
    ``membership_add``/``membership_remove`` pair.

    *Each membership file is its own source.* §3.2 gives the three files
    ``source = "wikipedia@{asof}"`` and also calls them revisions 1, 2 and 3 of
    one table. The two cannot both hold: a revision is monotone within
    ``(key, source)`` and is never compared across sources. The source id is
    taken literally and the ordering is expressed where a reader will act on
    it — ``source_priority``, newest file first.

    *Two intervals of one ticker that overlap are coalesced.* The key is
    ``(index_id, sid, valid_from)`` and two intervals that both open before the
    sample share it. The union answers the membership question identically, and
    it is the repository's own convention (``docs/data_integrity_diagnostic.md``
    §7). Every coalesce is listed in ``coverage.membership``.

    *A row the session table cannot date is not written.* Availability is the
    session close of a row's time of record plus the source's lag, and
    :mod:`prism.io.sessions` covers 2006-2027 only. Bars outside it are dropped
    with the series truncated and counted; a series wholly outside is skipped
    and named; an event outside it is not written and is counted. Decision D12
    widened the table from 2020 to 2006 for exactly this reason: the trend
    reproduction's sample opens on 2006-09-27, and the ETF distribution
    histories open in 2007, so before the widening neither could be read.
    """
    root = Path(root)
    existing = [p for p in (root / "manifest.json", root / "securities.parquet", root / "bars") if p.exists()]
    if existing:
        raise LegacyImportError(
            f"{root} already holds {[p.name for p in existing]}; an import writes a dataset whole so its manifest "
            "describes exactly what is there. Remove them or name another directory."
        )
    inputs = discover_legacy_inputs(cache_dir, interval=interval, cache_files=cache_files)
    wanted = None if symbols is None else {normalize_ticker(s) for s in symbols}
    selected = {s: paths for s, paths in inputs.files.items() if wanted is None or s in wanted}
    if not selected:
        raise LegacyImportError(
            f"none of the requested symbols has a cache under {inputs.cache_dir}; an empty import is a failed "
            "lookup, not an empty result (N7)"
        )
    mapping = legacy_mapping(gap_flag_sessions=gap_flag_sessions)
    master = SecurityMaster.open(root, gap_flag_sessions=gap_flag_sessions)

    # -- bars and identity ------------------------------------------------
    symbol_to_sid: dict[str, str] = {}
    skipped: list[dict[str, str]] = []
    truncated: list[dict[str, object]] = []
    gap_flags: list[dict[str, object]] = []
    duplicates = 0
    outcome = RevisionOutcome()
    written: list[Path] = []
    (root / "bars").mkdir(parents=True, exist_ok=True)
    coverage_start, coverage_end = sessions.coverage()
    for symbol, paths in selected.items():
        batches: list[BarBatch] = []
        days: set[dt.date] = set()
        outside: set[dt.date] = set()
        for path in paths:
            frame = read_mapped_file(path, fmt="parquet", mapping=mapping)
            repeated = frame.index.duplicated(keep="last")
            if bool(repeated.any()):
                # The vendor occasionally returns an overlapping bar. The
                # existing replay reader keeps the last duplicate
                # (`src/prism/live/replay.py:153`) and this import follows it,
                # counting every dropped row rather than passing a duplicate
                # key into a table whose revisions are ordered by file.
                duplicates += int(repeated.sum())
                frame = frame.loc[~repeated]
            # A bar outside the bundled session table cannot be dated: its
            # availability is the session close of its own date plus the
            # source's lag, and the table refuses a date it does not cover.
            # The rows inside the table are imported and the rest are recorded,
            # rather than dropping a whole series that is usable from 2020 on.
            inside = frame.index.map(lambda ts: coverage_start <= ts.date() <= coverage_end).to_numpy(dtype=bool)
            if not inside.all():
                outside.update(ts.date() for ts in frame.index[~inside])
                frame = frame.loc[inside]
            if frame.empty:
                continue
            batches.append(
                BarBatch(
                    sid=symbol,
                    frame=frame,
                    ingested_at=_mtime_instant(path),
                    source=LEGACY_VENDOR,
                    price_basis=mapping.source.price_basis,
                    origin=_origin(path, inputs.cache_dir),
                )
            )
            days.update(ts.date() for ts in frame.index)
        if not days:
            skipped.append(
                {
                    "vendor_symbol": symbol,
                    "reason": (
                        f"every bar it carries ({min(outside).isoformat()}..{max(outside).isoformat()}) lies outside "
                        f"the bundled session table {sessions.TABLE_VERSION}; availability is derived by walking "
                        "that table, so none of the rows can be dated"
                    ),
                }
            )
            continue
        if outside:
            truncated.append(
                {
                    "vendor_symbol": symbol,
                    "n_rows_outside": len(outside),
                    "first_outside": min(outside).isoformat(),
                    "last_outside": max(outside).isoformat(),
                    "kept": f"{min(days).isoformat()}..{max(days).isoformat()}",
                    "reason": f"outside the bundled session table {sessions.TABLE_VERSION}",
                }
            )
        resolution = master.ingest_series(
            LEGACY_VENDOR,
            symbol,
            ingested_at=min(batch.ingested_at for batch in batches),
            bar_dates=sorted(days),
            asset_class="etf" if symbol in inputs.distribution_symbols else "equity",
        )
        sids = resolution.sids
        if len(sids) != 1:
            raise LegacyImportError(
                f"{LEGACY_VENDOR}:{symbol} resolves to {len(sids)} identities {list(sids)}; the cache holds bars for "
                "more than one instrument and one bar file cannot carry both"
            )
        sid = sids[0]
        symbol_to_sid[symbol] = sid
        gap_flags.extend(flag.to_coverage_entry() for flag in resolution.gap_flags)
        frame, one = merge_bar_revisions(None, batches, policy=mapping.source.revision_policy)
        path = root / "bars" / f"{sid}_{interval}.parquet"
        _write_bar_table(frame, path)
        outcome = outcome + one
        written.append(path)
    if not symbol_to_sid:
        raise LegacyImportError(
            f"every one of the {len(selected)} selected series was skipped; nothing was imported (N7)"
        )
    first_session = min(
        day
        for day in (
            _first_bar_date(root / "bars" / f"{sid}_{interval}.parquet") for sid in set(symbol_to_sid.values())
        )
    )

    # -- membership -------------------------------------------------------
    files = tuple(membership_files) if membership_files is not None else inputs.membership_files
    memberships: list[dict[str, object]] = []
    remapped: list[dict[str, str]] = []
    membership_sources: list[str] = []
    for path in files:
        as_of = _membership_asof(path)
        source = f"{MEMBERSHIP_SOURCE_PREFIX}{as_of.isoformat()}"
        intervals, remaps, quarantined_skips, out_of_sample, coalesced = _legacy_membership_intervals(
            path, symbol_to_sid, first_session=first_session
        )
        remapped.extend(remaps)
        result = master.import_membership_intervals(
            index_id,
            intervals,
            as_of=as_of,
            source=source,
            ingested_at=_mtime_instant(path),
            floor=first_session,
            on_unresolved="skip",
        )
        membership_sources.append(source)
        memberships.append(
            {
                **result.to_dict(),
                "file": path.name,
                "as_of": as_of.isoformat(),
                "source": source,
                "ingested_at": _mtime_instant(path),
                "n_quarantined_skipped": len(quarantined_skips),
                "quarantined_skipped": quarantined_skips,
                "n_closed_before_sample": int(out_of_sample),
                "n_coalesced_intervals": len(coalesced),
                "coalesced_intervals": coalesced,
            }
        )
    membership_sources.reverse()  # newest file first: the later scrape is the better statement

    # -- attestations from the reviewed quarantine table -------------------
    attestations = []
    quarantined_report: list[dict[str, str]] = []
    for ticker, reason in sorted(QUARANTINE_TABLE.items()):
        sid = symbol_to_sid.get(normalize_ticker(ticker))
        if sid is None:
            continue
        attestations.append(
            Attestation(
                sid=sid,
                action="quarantine",
                attested_by=REVIEW_ATTESTED_BY,
                attested_at=REVIEW_DATE,
                attested_identity="unknown",
                reason=reason,
                evidence=QUARANTINE_EVIDENCE,
            )
        )
        quarantined_report.append({"ticker": ticker, "sid": sid, "reason": reason})
    master.overrides = SecurityOverrides(attestations=tuple(attestations))
    _write_overrides_file(root / OVERRIDES_FILENAME, attestations)

    # -- events -----------------------------------------------------------
    events, outside, event_sources = _legacy_events(
        symbol_to_sid,
        distributions_dir=(
            Path(distributions_dir) if distributions_dir is not None else inputs.cache_dir / "distributions"
        ),
        distribution_symbols=inputs.distribution_symbols,
        index_id=index_id,
        master=master,
    )
    events_frame(events).to_parquet(root / "events.parquet", index=False)
    adjustments_frame([]).to_parquet(root / "adjustments.parquet", index=False)
    written.extend([root / "events.parquet", root / "adjustments.parquet"])
    written.extend(master.write(root))

    # -- manifest ---------------------------------------------------------
    report = master.report()
    sources: dict[str, dict[str, object]] = {
        LEGACY_VENDOR: {
            "lag_bars": LAG_BARS,
            "revision_policy": mapping.source.revision_policy,
            "price_basis": mapping.source.price_basis,
            "url": None,
            "sha256": None,
            "redistributable": False,
        }
    }
    for source in event_sources:
        sources[source] = _non_bar_source()
    sources[ATTESTATION_SOURCE] = _non_bar_source()
    for source in membership_sources:
        sources[source] = _non_bar_source()
    manifest = rebuild_manifest(
        root,
        vintage=vintage,
        provenance="market",
        sources=sources,
        coverage={
            **report.coverage_entries(),
            "inputs": inputs.to_dict(),
            "skipped_series": skipped,
            "truncated_series": truncated,
            "membership": memberships,
            "remapped_tickers": remapped,
            "quarantined_tickers": quarantined_report,
            "events_outside_session_table": outside,
            "duplicate_bars_dropped": int(duplicates),
            "recorded_coverage_files": [p.name for p in inputs.coverage_files],
        },
        assumptions=_LEGACY_ASSUMPTIONS
        | {
            "session_table_version": sessions.TABLE_VERSION,
            "gap_flag_sessions": int(gap_flag_sessions),
            "membership_floor": first_session.isoformat(),
        },
        transformations=[
            "import_legacy_v1",
            f"cache_dir={inputs.cache_dir.name}",
            f"interval={interval}",
            f"session_table={sessions.TABLE_VERSION}",
        ],
        universe=sha256_bytes(canonical_json(sorted(set(symbol_to_sid.values()))).encode("utf-8")),
    )
    logger.info(
        "imported %d series into %d sid(s) at %s (data_id %s)",
        len(symbol_to_sid),
        len(set(symbol_to_sid.values())),
        root,
        manifest.id,
    )
    return LegacyDatasetReport(
        root=root,
        manifest=manifest,
        inputs=inputs,
        symbol_to_sid=dict(sorted(symbol_to_sid.items())),
        rows_written=outcome.written,
        rows_deduped=outcome.deduped,
        rows_revised=outcome.revised,
        duplicate_bars_dropped=duplicates,
        skipped_series=tuple(skipped),
        truncated_series=tuple(truncated),
        membership=tuple(memberships),
        remapped_tickers=tuple(remapped),
        quarantined=tuple(quarantined_report),
        events_written=len(events),
        events_outside_session_table=tuple(outside),
        gap_flags=tuple(gap_flags),
        files=tuple(sorted(str(p.relative_to(root)) for p in written)),
    )


#: The assumptions every legacy import records, verbatim from §3.2 where §3.2
#: states them.
_LEGACY_ASSUMPTIONS: dict[str, object] = {
    "ingestion_time_quality": "file_mtime",
    "availability": (
        "available_at is empty on every row; availability is the session close of the row's own time of record "
        "plus the source's lag_bars, which is zero for every source this import writes"
    ),
    "price_basis": (
        "every imported bar is split_adjusted_at_ingest: both vendors in this repository fetch split-adjusted, "
        "dividend-unadjusted bars (loader.py passes adjust=splits, live/alpaca_data.py defaults to "
        "adjustment='split'), so no share change is applied on a split effective date (decision D3)"
    ),
    "sid_minting": (
        "every sid is a minted surrogate sec_%06d with entity_status provisional, matched across files by "
        "overlapping (ticker, interval) through prism.io.security_master; a ticker is never a sid"
    ),
    "overlapping_caches": (
        "rows for one (sid, ts) from two files dedupe when identical and otherwise become successive revisions "
        "ordered by file modification time; a duplicate timestamp inside one file keeps the last row, matching "
        "prism.live.replay"
    ),
    "membership_intervals": (
        "an end equal to the file's own as-of becomes an open interval; an end after it is a scheduled removal and "
        "is kept; a start at or before the first imported session becomes floor_unknown with a null valid_from, "
        "because the availability derivation cannot walk a date outside the bundled session table"
    ),
    "membership_revisions": (
        "each membership file is its own source wikipedia@{asof} ordered newest first in source_priority; §3.2 also "
        "calls the three files revisions 1, 2 and 3 of one table, which is not expressible at once because a "
        "revision is monotone within (key, source) and is never compared across sources"
    ),
    "reviewed_tables": (
        "RENAME_TABLE is applied as a dateless membership-ticker remap, exactly as "
        "prism.io.universe_sp500.resolve_coverage applies it, and is recorded in coverage.remapped_tickers; only a "
        "rename whose effective date the reviewed record states is written as a ticker_change event. QUARANTINE_TABLE "
        "becomes a security_overrides.toml attestation per sid, and its tickers resolve to no index seat"
    ),
    "non_bar_price_basis": (
        "a source that carries no prices records price_basis 'raw' because the manifest schema requires a value; it "
        "is a placeholder for those sources and never a claim about a price series"
    ),
    "session_table_truncation": (
        "a bar dated outside the bundled session table is not imported, because its availability is the session "
        "close of its own date plus the source's lag and the table refuses a date it does not cover; the rows inside "
        "the table are imported and every truncation is counted in coverage.truncated_series"
    ),
    "not_imported": (
        "the un-ranged {SYMBOL}_{interval}.parquet files are not imported because their coverage is unknown and the "
        "loader already skips them; events dated outside the bundled session table are not written because their "
        "availability cannot be derived, and both counts are in coverage"
    ),
}


def sid_by_symbol(root: Path | str, *, vendor: str = LEGACY_VENDOR) -> dict[str, str]:
    """The ticker-to-sid mapping an import assigned, read back from ``listings``.

    §3.2: "The import records the ticker-to-sid mapping it assigned, so a later
    reader can trace any imported row back to the file and symbol it came from."
    """
    frame = pd.read_parquet(Path(root) / "listings.parquet")
    if frame.empty:
        return {}
    rows = frame.loc[frame["vendor"].astype(str) == vendor]
    return {str(r.vendor_symbol): str(r.sid) for r in rows.itertuples(index=False)}


@dataclass(frozen=True)
class RoundTrip:
    """The round-trip of §3.2: the imported panel against the loader's own file selection."""

    symbols: tuple[str, ...]
    n_cells: int
    max_abs_difference: float
    disagreements: tuple[dict[str, object], ...]
    only_imported: tuple[dict[str, object], ...]
    only_loader: tuple[dict[str, object], ...]
    missing_symbols: tuple[str, ...]

    @property
    def exact(self) -> bool:
        """Whether every cell of the two panels agrees and neither side holds a row the other does not."""
        return not (self.disagreements or self.only_imported or self.only_loader or self.missing_symbols)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_symbols": len(self.symbols),
            "n_cells": int(self.n_cells),
            "exact": bool(self.exact),
            "max_abs_difference": float(self.max_abs_difference),
            "n_disagreements": len(self.disagreements),
            "disagreements": [dict(entry) for entry in self.disagreements[:50]],
            "n_only_imported": len(self.only_imported),
            "only_imported": [dict(entry) for entry in self.only_imported[:20]],
            "n_only_loader": len(self.only_loader),
            "only_loader": [dict(entry) for entry in self.only_loader[:20]],
            "missing_symbols": list(self.missing_symbols),
        }


def round_trip_panel(
    root: Path | str,
    *,
    cache_dir: Path | str,
    symbols: Sequence[str],
    start: str,
    end: str,
    interval: str = DEFAULT_BAR_INTERVAL,
    field: str = "close",
) -> RoundTrip:
    """Compare the imported bars against the loader's own selection, cell by cell.

    §3.2 declares the comparison: the loader's file selection is "the first
    covering file in sorted-name order", so for a symbol with one cache file
    the comparison is exact by construction, and for a symbol with more than
    one the test "reports every overlapping-row disagreement as a finding
    rather than tolerating it, since a disagreement means the two caches carry
    different adjustment bases".

    Three kinds of finding are reported separately and none is tolerated: a
    cell both sides carry and disagree about, a row only the import holds (a
    second cache file covers a date the loader's selected file does not), and a
    row only the loader holds. The imported side takes the highest revision per
    ``(sid, ts)``, which is the selection rule of §3.2 restricted to the one
    source this import writes.
    """
    root = Path(root)
    mapping = sid_by_symbol(root)
    wanted = [normalize_ticker(s) for s in dict.fromkeys(symbols)]
    missing = tuple(s for s in wanted if s not in mapping)
    reference = read_legacy_panels(
        [s for s in wanted if s in mapping], cache_dir=Path(cache_dir), start=start, end=end
    )
    lower = pd.Timestamp(start).tz_localize(BAR_TZ)
    upper = pd.Timestamp(end).tz_localize(BAR_TZ)
    disagreements: list[dict[str, object]] = []
    only_imported: list[dict[str, object]] = []
    only_loader: list[dict[str, object]] = []
    cells = 0
    worst = 0.0
    for symbol in wanted:
        sid = mapping.get(symbol)
        if sid is None:
            continue
        imported = _imported_series(root / "bars" / f"{sid}_{interval}.parquet", field=field)
        imported = imported.loc[(imported.index >= lower) & (imported.index <= upper)]
        expected = reference.get(symbol)
        if expected is None:
            only_imported.extend(
                {"symbol": symbol, "sid": sid, "ts": str(stamp), "imported": float(value)}
                for stamp, value in imported.items()
            )
            continue
        left = pd.to_numeric(expected[field], errors="coerce").astype(float)
        left = left.loc[(left.index >= lower) & (left.index <= upper)]
        left = left[~left.index.duplicated(keep="last")]
        for stamp in left.index.difference(imported.index):
            only_loader.append({"symbol": symbol, "sid": sid, "ts": str(stamp), "loader": float(left[stamp])})
        for stamp in imported.index.difference(left.index):
            only_imported.append({"symbol": symbol, "sid": sid, "ts": str(stamp), "imported": float(imported[stamp])})
        shared = left.index.intersection(imported.index)
        cells += len(shared)
        for stamp in shared:
            a = float(left[stamp])
            b = float(imported[stamp])
            if _close_enough(a, b):
                continue
            worst = max(worst, abs(a - b))
            disagreements.append({"symbol": symbol, "sid": sid, "ts": str(stamp), "loader": a, "imported": b})
    return RoundTrip(
        symbols=tuple(wanted),
        n_cells=cells,
        max_abs_difference=worst,
        disagreements=tuple(disagreements),
        only_imported=tuple(only_imported),
        only_loader=tuple(only_loader),
        missing_symbols=missing,
    )


def legacy_summary_json(report: LegacyDatasetReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# stage 3 helpers
# ---------------------------------------------------------------------------


_CACHE_NAME = re.compile(
    r"^(?P<symbol>[A-Za-z0-9.\-]+)_(?P<interval>[A-Za-z0-9]+)(?:_(?P<start>[^_]+)_(?P<end>[^_]+))?\.parquet$"
)


def _parse_cache_name(name: str) -> tuple[str | None, str | None]:
    """``(symbol, interval)`` for a range-encoded cache; ``(symbol, None)`` for an un-ranged one."""
    match = _CACHE_NAME.match(name)
    if match is None:
        return None, None
    if match.group("start") is None:
        return match.group("symbol").upper(), None
    return match.group("symbol").upper(), match.group("interval")


def _origin(path: Path, cache_dir: Path) -> str:
    try:
        return str(path.relative_to(cache_dir))
    except ValueError:
        return path.name


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _write_bar_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    # The ``ts`` column keeps a tz-aware datetime dtype rather than an ISO
    # string: the caches cross a daylight-saving boundary, and a column of
    # mixed-offset strings parses to object dtype, which the view's record-time
    # parse cannot use.
    out["ts"] = pd.DatetimeIndex(pd.to_datetime(out["ts"], utc=True)).tz_convert(BAR_TZ)
    out.to_parquet(path, index=False)
    return path


def _first_bar_date(path: Path) -> dt.date:
    frame = pd.read_parquet(path, columns=["ts"])
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["ts"], utc=True)).tz_convert(BAR_TZ)
    return stamps.min().date()


def _imported_series(path: Path, *, field: str) -> pd.Series:
    """One sid's imported bars, highest revision per ts, as a ts-indexed series."""
    frame = pd.read_parquet(path)
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["ts"], utc=True)).tz_convert(BAR_TZ)
    frame = frame.assign(_ts=stamps).sort_values(["_ts", "revision"], kind="stable")
    frame = frame.drop_duplicates(subset=["_ts"], keep="last")
    return pd.Series(pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float), index=frame["_ts"])


def _close_enough(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return left == right


def _membership_asof(path: Path) -> dt.date:
    stem = Path(path).stem
    try:
        return dt.date.fromisoformat(stem.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise LegacyImportError(
            f"{path.name} does not carry an as-of date in its name; §3.2 reads it as "
            "sp500_membership_{asof}.parquet and the as-of decides which interval ends are open"
        ) from exc


def _legacy_membership_intervals(
    path: Path,
    symbol_to_sid: Mapping[str, str],
    *,
    first_session: dt.date,
) -> tuple[list[tuple[str, object, object]], list[dict[str, str]], list[dict[str, str]], int]:
    """One membership file's intervals, under the reviewed tables' own conventions.

    ``RENAME_TABLE`` is applied as the remap it is: the index seat of a renamed
    ticker belongs to the successor's instrument, which is exactly how
    ``prism.io.universe_sp500.resolve_coverage`` resolves an ever-member to a
    vendor symbol.

    Two intervals of one resolved ticker that overlap or touch are coalesced
    into their union, and every coalesce is recorded. This is lossless for the
    only question the table answers — the days on which the sid is a member are
    the union either way — and it is required, because the key of §3.2 is
    ``(index_id, sid, valid_from)`` and two intervals that both open before the
    sample share it. It is also the repository's own convention: "``RENAME_TABLE``
    is seeded with the three reviewed remaps ... which also merges the old
    symbols' membership intervals into their successors'"
    (``docs/data_integrity_diagnostic.md`` §7). Run over the three tracked files:
    the rename splits one seat in two in every one of them — ``FB``/``META``,
    ``PCLN``/``BKNG`` and ``WLTW``/``WTW`` in the 2026-06-16 file, and the same
    seats relabelled but still split in the two post-remediation files.

    ``QUARANTINE_TABLE`` tickers resolve to no instrument at all — the vendor
    series under that ticker is a known different instrument — so their seats
    join the counted skip list rather than being attributed to the wrong sid,
    which is the same measured-survivorship-leak discipline the coverage ledger
    already applies.
    """
    frame = pd.read_parquet(path)
    for required in ("ticker", "start", "end"):
        if required not in frame.columns:
            raise LegacyImportError(
                f"{path} has no {required!r} column; the membership file this import reads is the one "
                "prism.scripts.build_sp500_universe writes, with [ticker, start, end]"
            )
    remapped: list[dict[str, str]] = []
    quarantined: list[dict[str, str]] = []
    closed_before_sample = 0
    groups: dict[str, list[dict[str, object]]] = {}
    for record in frame.itertuples(index=False):
        ticker = normalize_ticker(str(record.ticker))
        if ticker in QUARANTINE_TABLE:
            quarantined.append({"ticker": ticker, "reason": QUARANTINE_TABLE[ticker]})
            continue
        opened = pd.Timestamp(record.start).date()
        closed = pd.Timestamp(record.end).date()
        if closed < first_session:
            closed_before_sample += 1
            continue
        resolved = normalize_ticker(RENAME_TABLE.get(ticker, ticker))
        if resolved != ticker:
            remapped.append(
                {
                    "file": path.name,
                    "ticker": ticker,
                    "resolves_to": resolved,
                    "sid": symbol_to_sid.get(resolved, ""),
                    "note": "RENAME_TABLE (src/prism/io/universe_sp500.py); the reviewed remap carries no date",
                }
            )
        groups.setdefault(resolved, []).append(
            {"ticker": ticker, "start": opened, "end": closed, "remapped": resolved != ticker}
        )
    intervals: list[tuple[str, object, object]] = []
    coalesced: list[dict[str, str]] = []
    for resolved, entries in groups.items():
        for merged in _coalesce_intervals(resolved, entries, coalesced, file_name=path.name):
            intervals.append((resolved, merged["start"], merged["end"]))
    return intervals, remapped, quarantined, closed_before_sample, coalesced


def _coalesce_intervals(
    resolved: str,
    entries: list[dict[str, object]],
    coalesced: list[dict[str, str]],
    *,
    file_name: str,
) -> list[dict[str, object]]:
    """Union the intervals of one resolved ticker that overlap or touch, recording each union."""
    ordered = sorted(entries, key=lambda entry: (entry["start"], entry["end"]))
    if len(ordered) == 1:
        return ordered
    out: list[dict[str, object]] = []
    for entry in ordered:
        if out and entry["start"] <= out[-1]["end"] + dt.timedelta(days=1):
            previous = out[-1]
            merged = {
                "ticker": f"{previous['ticker']}+{entry['ticker']}",
                "start": min(previous["start"], entry["start"]),
                "end": max(previous["end"], entry["end"]),
                "remapped": bool(previous["remapped"] or entry["remapped"]),
            }
            coalesced.append(
                {
                    "file": file_name,
                    "resolves_to": resolved,
                    "from": (
                        f"{previous['ticker']} {previous['start']}..{previous['end']} + "
                        f"{entry['ticker']} {entry['start']}..{entry['end']}"
                    ),
                    "to": f"{merged['start']}..{merged['end']}",
                    "remapped": "yes" if merged["remapped"] else "no",
                }
            )
            out[-1] = merged
        else:
            out.append(entry)
    return out


def _legacy_events(
    symbol_to_sid: Mapping[str, str],
    *,
    distributions_dir: Path,
    distribution_symbols: Sequence[str],
    index_id: str,
    master: SecurityMaster,
) -> tuple[list[Event], list[dict[str, object]], list[str]]:
    """The ``events`` table: issuer distributions plus the dated reviewed events.

    An event dated outside the bundled session table is not written. Its
    availability is the session close of its effective date plus the source's
    lag, and the table cannot walk a date it does not cover, so one 1993 ex-date
    would make every ``events`` query on this dataset refuse. Every such event
    is counted in the manifest ``coverage`` with its date range.
    """
    start, end = sessions.coverage()
    events: list[Event] = []
    outside: list[dict[str, object]] = []
    sources: list[str] = []
    for symbol in distribution_symbols:
        sid = symbol_to_sid.get(normalize_ticker(symbol))
        if sid is None:
            continue
        found = issuer_distribution_events(distributions_dir, sid, symbol=symbol)
        kept = [e for e in found if start <= dt.date.fromisoformat(e.effective_date) <= end]
        dropped = [e for e in found if e not in kept]
        events.extend(kept)
        if dropped:
            outside.append(
                {
                    "sid": sid,
                    "symbol": symbol,
                    "event_type": "cash_distribution",
                    "n_dropped": len(dropped),
                    "first": min(e.effective_date for e in dropped),
                    "last": max(e.effective_date for e in dropped),
                    "reason": f"outside the bundled session table {sessions.TABLE_VERSION}",
                }
            )
    if events:
        sources.append(DISTRIBUTION_SOURCE)
    reviewed: list[Event] = []
    for ticker, entry in sorted(DATED_TICKER_CHANGES.items()):
        sid = symbol_to_sid.get(normalize_ticker(ticker))
        if sid is None:
            continue
        effective = dt.date.fromisoformat(entry["effective_date"])
        if not start <= effective <= end:
            outside.append(
                {
                    "sid": sid,
                    "symbol": ticker,
                    "event_type": "ticker_change",
                    "n_dropped": 1,
                    "first": entry["effective_date"],
                    "last": entry["effective_date"],
                    "reason": f"outside the bundled session table {sessions.TABLE_VERSION}",
                }
            )
            continue
        reviewed.append(
            Event(
                sid=sid,
                event_type="ticker_change",
                effective_date=entry["effective_date"],
                announce_date=entry.get("announce_date"),
                payload={"from_ticker": ticker, "to_ticker": entry["to_ticker"]},
                ingested_at=_review_instant(),
                source=REVIEWED_SOURCE,
            )
        )
    for patch in CHANGES_PATCH_TABLE:
        effective = dt.date.fromisoformat(str(patch["date"]))
        for column, event_type in (("added", "membership_add"), ("removed", "membership_remove")):
            ticker = normalize_ticker(str(patch.get(column, "")))
            if not ticker:
                continue
            sid = symbol_to_sid.get(ticker)
            if sid is None or not start <= effective <= end:
                outside.append(
                    {
                        "sid": sid or "",
                        "symbol": ticker,
                        "event_type": event_type,
                        "n_dropped": 1,
                        "first": patch["date"],
                        "last": patch["date"],
                        "reason": (
                            f"outside the bundled session table {sessions.TABLE_VERSION}"
                            if sid is not None
                            else "the ticker resolves to no imported instrument"
                        ),
                    }
                )
                continue
            reviewed.append(
                Event(
                    sid=sid,
                    event_type=event_type,
                    effective_date=str(patch["date"]),
                    payload={"index_id": index_id},
                    ingested_at=_review_instant(),
                    source=REVIEWED_SOURCE,
                )
            )
    if reviewed:
        sources.append(REVIEWED_SOURCE)
    events.extend(reviewed)
    return events, outside, sources


def _review_instant() -> str:
    """The reviewed tables' ingestion time: the close of the day they were reviewed."""
    return f"{REVIEW_DATE.isoformat()}T16:00:00-04:00"


def _non_bar_source() -> dict[str, object]:
    """The manifest entry of a source that carries no prices.

    ``price_basis`` is required by the manifest schema and is recorded as
    ``raw`` for these sources; it is a placeholder, never a claim about a price
    series, and the manifest ``assumptions`` says so.
    """
    return {
        "lag_bars": LAG_BARS,
        "revision_policy": "dedupe_identical",
        "price_basis": "raw",
        "url": None,
        "sha256": None,
        "redistributable": False,
    }


def _write_overrides_file(path: Path, attestations: Sequence[Attestation]) -> Path:
    """Write ``security_overrides.toml`` so the import's decisions are inspectable data.

    The file is the attestation file of §3.2 and re-reads through
    :meth:`prism.io.security_master.SecurityOverrides.read`, so opening the
    workspace again applies exactly the decisions this import applied.
    """
    lines = [f"schema = {OVERRIDES_SCHEMA}", ""]
    for attestation in attestations:
        lines.append("[[attestation]]")
        lines.append(f'sid = "{_toml(attestation.sid)}"')
        lines.append(f'action = "{_toml(attestation.action)}"')
        if attestation.attested_identity:
            lines.append(f'attested_identity = "{_toml(attestation.attested_identity)}"')
        if attestation.reason:
            lines.append(f'reason = "{_toml(attestation.reason)}"')
        lines.append(f'attested_by = "{_toml(attestation.attested_by)}"')
        lines.append(f'attested_at = "{attestation.attested_at.isoformat()}"')
        if attestation.evidence:
            lines.append(f'evidence = "{_toml(attestation.evidence)}"')
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _toml(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def summary_json(report: LegacyImportReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


__all__ = [
    "BAR_SOURCE",
    "DATED_TICKER_CHANGES",
    "DISTRIBUTION_SOURCE",
    "LEGACY_MAPPING_TOML",
    "LEGACY_VENDOR",
    "MEMBERSHIP_SOURCE",
    "MEMBERSHIP_SOURCE_PREFIX",
    "REVIEWED_SOURCE",
    "REVIEW_DATE",
    "LegacyDatasetReport",
    "LegacyImportError",
    "LegacyImportReport",
    "LegacyInputs",
    "RoundTrip",
    "build_legacy_dataset",
    "discover_legacy_inputs",
    "import_legacy",
    "legacy_mapping",
    "legacy_summary_json",
    "panel_matrices",
    "read_legacy_panels",
    "round_trip_panel",
    "session_index",
    "sid_by_symbol",
    "summary_json",
]
