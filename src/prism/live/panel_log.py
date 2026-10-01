"""Per-session record of the bar panel a live decision was made on.

The nightly fetches the whole trailing OHLCV panel from the venue's own feed,
scores it, submits orders and discards it. Nothing in the run directory said
what was scored, so no past decision could be re-derived and the engine
upgrade's decision-parity gate compared zero sessions
(``docs/engine_overhaul_operating_changes.md`` §3.2 item 7, ``prism.live.parity_gate``).
This module is the per-session snapshot that item asks for: before a cycle
decides, the panel it holds is written under ``<run_dir>/panels/`` with a
content digest, and the refresh row in ``targets.jsonl`` cites that digest, so a
decision and the exact inputs it was made on are tied together durably.

Storage is sized for a 750-bar by 500-symbol panel every trading day. A full
snapshot (a *base*, parquet, zstd) is written first; each later session writes
a *delta* holding only the cells that are new, changed or gone relative to the
previous session's reconstructed panel: one appended bar per symbol, the bar
that fell off the trailing window, and any vendor revision of history. A new
base is written whenever the delta would exceed half the panel, whenever the
fields or the timezone change, every ``BASE_EVERY_SESSIONS`` sessions, or when
the previous panel cannot be reconstructed (that last case is logged at ERROR
and recorded in the index row's ``reason``: the record of what was decided on
stays complete, only the diff chain restarts). Measured on a synthetic
752 x 500 close/volume panel the base is about 2.6 MB and a one-day delta about
8 KB, so a year of sessions costs on the order of 12 MB.

The digest is over a canonical serialization, so two panels that agree cell
for cell have the same digest whatever order the vendor returned them in:
fields in a fixed order, symbols sorted, bars sorted and expressed as int64
nanoseconds, every field a float64 grid of bars by symbols with NaN and signed
zero normalized. Reconstruction replays the chain from the base and refuses to
return a panel whose digest does not match the recorded one. The index
(``index.jsonl``) is append-only and monotone in the decision bar; recording
the same bar twice is a no-op for identical content and an error for different
content, because a run directory that holds two different panels for one
decision describes two different decisions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prism.live import ledgers

logger = logging.getLogger(__name__)

#: Subdirectory of the run directory that holds the log.
PANEL_LOG_DIRNAME = "panels"
#: The append-only index inside that directory.
INDEX_FILENAME = "index.jsonl"
#: A new base is written once this many sessions (base included) share a base.
BASE_EVERY_SESSIONS = 63
#: A delta touching more than this fraction of the panel's cells becomes a base.
DELTA_BASE_FRACTION = 0.5
#: Version of the canonical serialization the digest is computed over.
CANONICAL_VERSION = 1

_PARQUET_COMPRESSION = "zstd"
_FIELD_ORDER = ("open", "high", "low", "close", "volume")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UPSERT = "upsert"
_DELETE = "delete"


class PanelLogError(RuntimeError):
    """The log cannot be read or written as asked."""


class PanelLogConflict(PanelLogError):
    """A decision bar already recorded with different content."""


class PanelDigestMismatch(PanelLogError):
    """A reconstructed panel, or a stored file, does not hash to its recorded digest."""


# ----------------------------------------------------------------- canonical


@dataclass(frozen=True)
class CanonicalPanel:
    """The digest-defining form of a panel: fixed field order, sorted symbols,
    sorted bars as int64 nanoseconds, one float64 grid per field."""

    fields: tuple[str, ...]
    symbols: tuple[str, ...]
    bars: np.ndarray
    tz: str | None
    values: Mapping[str, np.ndarray]

    @property
    def n_bars(self) -> int:
        return int(len(self.bars))

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    def index(self) -> pd.DatetimeIndex:
        stamps = pd.to_datetime(self.bars, unit="ns", utc=self.tz is not None)
        index = pd.DatetimeIndex(stamps)
        return index.tz_convert(self.tz) if self.tz is not None else index

    def sha256(self) -> str:
        """Digest of the canonical bytes: a JSON header, then each field's grid."""
        digest = hashlib.sha256()
        header = {
            "version": CANONICAL_VERSION,
            "fields": list(self.fields),
            "tz": self.tz,
            "symbols": list(self.symbols),
            "bars": [int(b) for b in self.bars],
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        for field in self.fields:
            digest.update(np.ascontiguousarray(self.values[field], dtype="<f8").tobytes())
        return digest.hexdigest()

    def to_frames(self) -> dict[str, pd.DataFrame]:
        index = self.index()
        columns = pd.Index(list(self.symbols), dtype=object)
        return {field: pd.DataFrame(self.values[field].copy(), index=index, columns=columns) for field in self.fields}

    def bar_dates(self) -> tuple[str, str]:
        index = self.index()
        return str(index[0].date()), str(index[-1].date())


def _normalize_floats(grid: np.ndarray) -> np.ndarray:
    """One bit pattern per value: -0.0 becomes 0.0 and every NaN the default NaN."""
    grid = np.asarray(grid, dtype="float64") + 0.0
    return np.where(np.isnan(grid), np.nan, grid)


def _order_fields(fields: list[str]) -> tuple[str, ...]:
    known = [f for f in _FIELD_ORDER if f in fields]
    return tuple(known + sorted(f for f in fields if f not in _FIELD_ORDER))


def canonicalize(panel: Mapping[str, pd.DataFrame]) -> CanonicalPanel:
    """The canonical form of ``panel`` (``{field: wide frame}``, bars by symbols).

    Frames whose indexes or columns differ are placed on the union of both, NaN
    where a frame has no cell, so the canonical shape is the same whichever
    field the caller consults. Frames that already share index and columns
    round-trip exactly (up to index frequency and axis names, which the
    canonical form does not carry).
    """
    if not isinstance(panel, Mapping) or not panel:
        raise TypeError("panel must be a non-empty mapping of field name -> wide DataFrame")
    frames: dict[str, pd.DataFrame] = {}
    indexes: dict[str, pd.DatetimeIndex] = {}
    for name, frame in panel.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"panel field {name!r} is {type(frame).__name__}, expected a DataFrame")
        if frame.empty:
            raise ValueError(f"panel field {name!r} is empty; refusing to record nothing (N7)")
        bars_index = frame.index
        if not isinstance(bars_index, pd.DatetimeIndex):
            raise TypeError(f"panel field {name!r} is not indexed by a DatetimeIndex")
        if bars_index.hasnans or not bars_index.is_unique:
            raise ValueError(f"panel field {name!r} has NaT or duplicate bars")
        renamed = frame.copy()
        renamed.columns = pd.Index([str(c) for c in frame.columns], dtype=object)
        if not renamed.columns.is_unique:
            raise ValueError(f"panel field {name!r} has duplicate symbols")
        frames[str(name)] = renamed
        indexes[str(name)] = bars_index
    tzs = {None if ix.tz is None else str(ix.tz) for ix in indexes.values()}
    if len(tzs) != 1:
        raise ValueError(f"panel fields carry different timezones {sorted(map(str, tzs))}; one panel, one clock")
    tz = next(iter(tzs))
    union: pd.DatetimeIndex | None = None
    symbols: set[str] = set()
    for name, frame in frames.items():
        union = indexes[name] if union is None else pd.DatetimeIndex(union.union(indexes[name]))
        symbols |= set(frame.columns)
    assert union is not None
    index = pd.DatetimeIndex(union.sort_values())
    ordered_symbols = tuple(sorted(symbols))
    fields = _order_fields(list(frames))
    values = {
        field: _normalize_floats(
            frames[field]
            .reindex(index=index, columns=list(ordered_symbols))
            .to_numpy(dtype="float64", na_value=np.nan)
        )
        for field in fields
    }
    # int64 nanoseconds; for a tz-aware index these are UTC-based, as asi8 is.
    bars = index.to_numpy(dtype="datetime64[ns]").astype("int64")
    return CanonicalPanel(fields=fields, symbols=ordered_symbols, bars=bars, tz=tz, values=values)


# -------------------------------------------------------------------- records


@dataclass(frozen=True)
class PanelRecord:
    """One index row: what was recorded for a decision bar and where."""

    decision_bar: str
    kind: str
    sha256: str
    n_symbols: int
    n_bars: int
    first_bar: str
    last_bar: str
    file: str
    bytes: int
    base_bar: str
    fields: tuple[str, ...]
    tz: str | None
    file_sha256: str
    n_upserts: int
    n_deletes: int
    reason: str
    recorded_at: str

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["fields"] = list(self.fields)
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "PanelRecord":
        try:
            return cls(
                decision_bar=str(row["decision_bar"]),
                kind=str(row["kind"]),
                sha256=str(row["sha256"]),
                n_symbols=int(row["n_symbols"]),
                n_bars=int(row["n_bars"]),
                first_bar=str(row["first_bar"]),
                last_bar=str(row["last_bar"]),
                file=str(row["file"]),
                bytes=int(row["bytes"]),
                base_bar=str(row["base_bar"]),
                fields=tuple(str(f) for f in row["fields"]),
                tz=None if row.get("tz") is None else str(row["tz"]),
                file_sha256=str(row["file_sha256"]),
                n_upserts=int(row["n_upserts"]),
                n_deletes=int(row["n_deletes"]),
                reason=str(row.get("reason", "")),
                recorded_at=str(row.get("recorded_at", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PanelLogError(f"malformed panel index row {dict(row)!r}: {exc}") from exc


def panel_log_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / PANEL_LOG_DIRNAME


def read_index(run_dir: Path | str) -> list[PanelRecord]:
    """Every index row, oldest first (empty when nothing was recorded)."""
    return [PanelRecord.from_row(row) for row in ledgers.read_rows(panel_log_dir(run_dir) / INDEX_FILENAME)]


def latest_record(run_dir: Path | str) -> PanelRecord | None:
    """The newest record, or ``None`` on an empty log."""
    records = read_index(run_dir)
    return records[-1] if records else None


# ---------------------------------------------------------------------- files


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet(table: pa.Table, path: Path, metadata: Mapping[str, str]) -> None:
    """Write ``table`` atomically: temp file, fsync, rename, directory fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    existing = table.schema.metadata or {}
    schema_metadata = {**existing, **{k.encode(): v.encode() for k, v in metadata.items()}}
    pq.write_table(table.replace_schema_metadata(schema_metadata), tmp, compression=_PARQUET_COMPRESSION)
    with open(tmp, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _base_table(panel: CanonicalPanel) -> pa.Table:
    bars = np.repeat(panel.bars, panel.n_symbols)
    symbols = np.tile(np.asarray(panel.symbols, dtype=object), panel.n_bars)
    columns: dict[str, pa.Array] = {
        "bar": pa.array(bars, type=pa.int64()),
        "symbol": pa.array(symbols, type=pa.string()),
    }
    for field in panel.fields:
        columns[field] = pa.array(np.ascontiguousarray(panel.values[field]).ravel(), type=pa.float64())
    return pa.table(columns)


def _delta_table(
    current: CanonicalPanel,
    upserts: np.ndarray,
    previous: CanonicalPanel,
    deletes: np.ndarray,
) -> pa.Table:
    up_rows, up_cols = np.nonzero(upserts)
    del_rows, del_cols = np.nonzero(deletes)
    symbols_cur = np.asarray(current.symbols, dtype=object)
    symbols_prev = np.asarray(previous.symbols, dtype=object)
    columns: dict[str, pa.Array] = {
        "bar": pa.array(np.concatenate([current.bars[up_rows], previous.bars[del_rows]]), type=pa.int64()),
        "symbol": pa.array(np.concatenate([symbols_cur[up_cols], symbols_prev[del_cols]]), type=pa.string()),
        "op": pa.array([_UPSERT] * len(up_rows) + [_DELETE] * len(del_rows), type=pa.string()),
    }
    for field in current.fields:
        values = np.concatenate([current.values[field][up_rows, up_cols], np.full(len(del_rows), np.nan)])
        columns[field] = pa.array(values, type=pa.float64())
    return pa.table(columns)


def _read_long(path: Path, fields: tuple[str, ...], *, delta: bool) -> pd.DataFrame:
    wanted = ["bar", "symbol", *(["op"] if delta else []), *fields]
    try:
        table = pq.read_table(path, columns=wanted)
    except (OSError, pa.ArrowException, KeyError) as exc:
        raise PanelLogError(f"cannot read panel file {path}: {exc}") from exc
    frame = table.to_pandas()
    frame["bar"] = frame["bar"].to_numpy(dtype="int64")
    frame["symbol"] = frame["symbol"].astype(str).to_numpy(dtype=object)
    for field in fields:
        frame[field] = frame[field].to_numpy(dtype="float64")
    return frame


# ---------------------------------------------------------------- the diff


def _diff(previous: CanonicalPanel, current: CanonicalPanel) -> tuple[np.ndarray, np.ndarray]:
    """``(upserts, deletes)``: boolean grids over the current and previous panels.

    An upsert is a current cell absent from the previous grid or holding a
    different value (NaN equals NaN); a delete is a previous cell whose bar or
    symbol is no longer in the current grid.
    """
    bar_pos = pd.Index(previous.bars).get_indexer(pd.Index(current.bars))
    sym_pos = pd.Index(list(previous.symbols)).get_indexer(pd.Index(list(current.symbols)))
    bar_in_prev = bar_pos >= 0
    sym_in_prev = sym_pos >= 0
    present = bar_in_prev[:, None] & sym_in_prev[None, :]
    changed = np.zeros(present.shape, dtype=bool)
    cur_rows = np.nonzero(bar_in_prev)[0]
    cur_cols = np.nonzero(sym_in_prev)[0]
    for field in current.fields:
        aligned = np.full(present.shape, np.nan)
        aligned[np.ix_(cur_rows, cur_cols)] = previous.values[field][np.ix_(bar_pos[cur_rows], sym_pos[cur_cols])]
        now = current.values[field]
        equal = (now == aligned) | (np.isnan(now) & np.isnan(aligned))
        changed |= present & ~equal
    upserts = ~present | changed
    prev_bar_kept = np.isin(previous.bars, current.bars)
    prev_sym_kept = np.isin(np.asarray(previous.symbols, dtype=object), np.asarray(current.symbols, dtype=object))
    deletes = ~(prev_bar_kept[:, None] & prev_sym_kept[None, :])
    return upserts, deletes


# ------------------------------------------------------------ reconstruction


def _chain_to(records: list[PanelRecord], target: PanelRecord) -> list[PanelRecord]:
    chain = [r for r in records if r.base_bar == target.base_bar and r.decision_bar <= target.decision_bar]
    if not chain or chain[0].kind != "base" or chain[0].decision_bar != target.base_bar:
        raise PanelLogError(
            f"panel index has no base {target.base_bar} for decision bar {target.decision_bar}; "
            "the chain cannot be replayed"
        )
    if any(r.kind != "delta" for r in chain[1:]):
        raise PanelLogError(
            f"panel chain from base {target.base_bar} holds a second base before {target.decision_bar}"
        )
    return chain


def _verify_file(directory: Path, record: PanelRecord) -> Path:
    path = directory / record.file
    if not path.is_file():
        raise PanelLogError(f"panel index names {record.file} for {record.decision_bar} but the file is missing")
    actual = _sha256_file(path)
    if actual != record.file_sha256:
        raise PanelDigestMismatch(
            f"panel file {record.file} hashes to {actual}, index recorded {record.file_sha256} (N7)"
        )
    return path


def _canonical_from_long(long: pd.DataFrame, fields: tuple[str, ...], tz: str | None, what: str) -> CanonicalPanel:
    """A full bars-by-symbols grid from a ``(bar, symbol)``-indexed long frame, or raise."""
    if not long.index.is_unique:
        raise PanelDigestMismatch(f"{what}: duplicate (bar, symbol) cells after replaying the chain")
    bars = np.array(sorted(set(long.index.get_level_values("bar"))), dtype="int64")
    symbols = tuple(sorted(set(long.index.get_level_values("symbol"))))
    expected = pd.MultiIndex.from_product([bars, list(symbols)], names=["bar", "symbol"])
    if len(long) != len(expected):
        raise PanelDigestMismatch(
            f"{what}: {len(long)} cells do not form a full {len(bars)} x {len(symbols)} grid after replaying the chain"
        )
    ordered = long.reindex(expected)
    values = {
        field: _normalize_floats(ordered[field].to_numpy(dtype="float64").reshape(len(bars), len(symbols)))
        for field in fields
    }
    return CanonicalPanel(fields=fields, symbols=symbols, bars=bars, tz=tz, values=values)


def _apply_delta(current: pd.DataFrame, delta: pd.DataFrame, fields: tuple[str, ...], what: str) -> pd.DataFrame:
    ops = delta["op"].to_numpy(dtype=object)
    deletes = delta[ops == _DELETE]
    upserts = delta[ops == _UPSERT]
    if len(deletes) + len(upserts) != len(delta):
        raise PanelLogError(f"{what}: unknown op in delta file (expected {_UPSERT!r}/{_DELETE!r})")
    if len(deletes):
        keys = pd.MultiIndex.from_arrays([deletes["bar"], deletes["symbol"]], names=["bar", "symbol"])
        if not keys.isin(current.index).all():
            raise PanelDigestMismatch(f"{what}: delta deletes a cell the chain never held")
        current = current.drop(index=keys)
    if len(upserts):
        incoming = upserts.set_index(["bar", "symbol"])[list(fields)]
        current = pd.concat([current.drop(index=incoming.index.intersection(current.index)), incoming])
    return current


def _reconstruct_canonical(directory: Path, records: list[PanelRecord], target: PanelRecord) -> CanonicalPanel:
    chain = _chain_to(records, target)
    base = chain[0]
    long = _read_long(_verify_file(directory, base), base.fields, delta=False).set_index(["bar", "symbol"])
    for record in chain[1:]:
        if record.fields != base.fields or record.tz != base.tz:
            raise PanelLogError(f"delta {record.file} changes fields or timezone against base {base.file}")
        delta = _read_long(_verify_file(directory, record), record.fields, delta=True)
        long = _apply_delta(long, delta, record.fields, record.file)
    panel = _canonical_from_long(long, base.fields, base.tz, target.file)
    actual = panel.sha256()
    if actual != target.sha256:
        raise PanelDigestMismatch(
            f"panel for {target.decision_bar} reconstructs to {actual}, index recorded {target.sha256} (N7)"
        )
    return panel


def reconstruct_panel(run_dir: Path | str, decision_bar: str) -> dict[str, pd.DataFrame]:
    """The panel recorded for ``decision_bar`` as ``{field: wide frame}``.

    Replays the chain from its base, verifies every file's digest and the
    reconstructed panel's digest, and raises :class:`PanelDigestMismatch` on
    any disagreement. Raises :class:`PanelLogError` for a bar never recorded.
    """
    directory = panel_log_dir(run_dir)
    records = read_index(run_dir)
    matches = [r for r in records if r.decision_bar == str(decision_bar)]
    if not matches:
        raise PanelLogError(f"no panel recorded for decision bar {decision_bar} under {directory}")
    return _reconstruct_canonical(directory, records, matches[-1]).to_frames()


# ------------------------------------------------------------------ recording


def _sessions_since_base(records: list[PanelRecord], latest: PanelRecord) -> int:
    return sum(1 for r in records if r.base_bar == latest.base_bar)


def record_panel(run_dir: Path | str, decision_bar: str, panel: Mapping[str, pd.DataFrame]) -> PanelRecord:
    """Record the panel ``decision_bar`` is decided on; return its index row.

    Identical content for an already-recorded bar returns the existing record
    without writing; different content raises :class:`PanelLogConflict`. A bar
    earlier than the newest recorded one raises: the index is monotone. Any
    failure to write raises, so a caller that proceeds has a recorded panel.
    """
    decision_bar = str(decision_bar)
    if not _ISO_DATE.match(decision_bar):
        raise ValueError(f"decision_bar must be an ISO date (YYYY-MM-DD), got {decision_bar!r}")
    directory = panel_log_dir(run_dir)
    current = canonicalize(panel)
    digest = current.sha256()
    records = read_index(run_dir)
    latest = records[-1] if records else None

    if latest is not None and decision_bar < latest.decision_bar:
        raise PanelLogError(
            f"decision bar {decision_bar} is earlier than the newest recorded {latest.decision_bar}; "
            "the panel log is append-only and monotone"
        )
    if latest is not None and decision_bar == latest.decision_bar:
        if latest.sha256 == digest:
            _verify_file(directory, latest)
            logger.info("panel for %s already recorded (sha256 %s); not rewriting", decision_bar, digest[:12])
            return latest
        raise PanelLogConflict(
            f"decision bar {decision_bar} is already recorded with sha256 {latest.sha256}; "
            f"refusing to record different content {digest} for the same decision (N7)"
        )

    kind, reason = "base", "first"
    previous: CanonicalPanel | None = None
    upserts = deletes = None
    if latest is not None:
        if latest.fields != current.fields or latest.tz != current.tz:
            reason = "fields_changed"
        elif _sessions_since_base(records, latest) >= BASE_EVERY_SESSIONS:
            reason = "cadence"
        else:
            try:
                previous = _reconstruct_canonical(directory, records, latest)
            except PanelLogError as exc:
                logger.error(
                    "panel log: cannot reconstruct the previous panel (%s) for %s: %s; writing a fresh base so "
                    "today's decision is still recorded in full (the delta chain restarts here) (N7)",
                    latest.decision_bar,
                    decision_bar,
                    exc,
                )
                reason = "previous_unreadable"
            else:
                upserts, deletes = _diff(previous, current)
                touched = int(upserts.sum()) + int(deletes.sum())
                if touched > DELTA_BASE_FRACTION * current.n_bars * current.n_symbols:
                    reason = "delta_exceeds_half"
                else:
                    kind, reason = "delta", "delta"

    if kind == "delta":
        assert previous is not None and upserts is not None and deletes is not None
        table = _delta_table(current, upserts, previous, deletes)
        n_upserts, n_deletes = int(upserts.sum()), int(deletes.sum())
        base_bar = latest.base_bar if latest is not None else decision_bar
    else:
        table = _base_table(current)
        n_upserts, n_deletes = current.n_bars * current.n_symbols, 0
        base_bar = decision_bar

    file_name = f"{decision_bar}.{kind}.parquet"
    path = directory / file_name
    first_bar, last_bar = current.bar_dates()
    _write_parquet(
        table,
        path,
        {
            "prism_panel_log": json.dumps(
                {
                    "kind": kind,
                    "decision_bar": decision_bar,
                    "fields": list(current.fields),
                    "tz": current.tz,
                    "sha256": digest,
                    "base_bar": base_bar,
                },
                sort_keys=True,
            )
        },
    )
    record = PanelRecord(
        decision_bar=decision_bar,
        kind=kind,
        sha256=digest,
        n_symbols=current.n_symbols,
        n_bars=current.n_bars,
        first_bar=first_bar,
        last_bar=last_bar,
        file=file_name,
        bytes=int(path.stat().st_size),
        base_bar=base_bar,
        fields=current.fields,
        tz=current.tz,
        file_sha256=_sha256_file(path),
        n_upserts=n_upserts,
        n_deletes=n_deletes,
        reason=reason,
        recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    ledgers.append_rows(directory / INDEX_FILENAME, [record.to_row()])
    logger.info(
        "panel for %s recorded: %s %s (%d bars x %d symbols, %d upserts, %d deletes, %d bytes, sha256 %s)",
        decision_bar,
        kind,
        file_name,
        record.n_bars,
        record.n_symbols,
        n_upserts,
        n_deletes,
        record.bytes,
        digest[:12],
    )
    return record
