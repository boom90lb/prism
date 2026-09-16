"""``Dataset`` and ``DatasetView``: the as-of read path over a manifest (engine spec §3.2).

A ``DatasetView`` is a dataset pinned to one decision date. Every accessor
applies the same visibility predicate and the same selection rule, so the
eligible universe, ticker resolution and dated capabilities cannot bypass the
as-of cut: a value attributed to bar ``t`` uses only what was knowable at
``t``'s close (SPEC.md N2).

Two visibility modes, and they answer different questions:

``replay_recorded`` answers *what did this workspace possess*. A row is visible
when ``ingested_at <= cutoff(d)``, where ``cutoff(d)`` is the wall clock the
decision was taken at: the recorded ``decided_at`` for ``d`` when the run
recorder wrote one (§3.12), otherwise the ``decision_time`` clock and IANA zone
the evaluation declares. With neither, the view refuses rather than inventing a
clock. A decision date earlier than a table's first ingestion is refused and the
table is named, because the honest answer is that the workspace held nothing.

``research_vintage`` answers *what had the world published*. A frozen manifest
vintage is fixed for the whole evaluation and only files that manifest lists are
read. A row is visible when its time of record is on or before ``d`` and its
availability is at or before ``cutoff(d)``, the close of session ``d`` plus the
registration's ``decision_offset_minutes``. Availability is the recorded
``available_at`` when the source publishes one and otherwise the derivation of
§3.2's availability table. Corrections a source made silently before the vintage
are taken as known at the row's own time of record; that assumption is recorded
on the view and copied into the result record, because it is the one place this
mode can flatter a strategy.

The two clocks genuinely disagree, and the disagreement is not a defect: a
correction published at 09:00 and stored here at 20:00 is visible to a
``research_vintage`` decision taken at that day's 16:00 close and invisible to a
``replay_recorded`` decision taken at the same instant. Worked example B of §3.2
is exactly that case and ``tests/test_dataset_asof.py`` pins both answers.

Selection among visible rows for one key is deterministic (§3.2): within each
source the highest ``revision`` wins, a tie there is an ingestion defect and is
refused, and across sources the highest-priority source with a visible row wins,
where priority is position in the manifest's ``source_priority`` list for that
table. Revisions are never compared across sources, and a row whose source the
manifest does not order makes the view refuse rather than pick a lexical winner.

Every revision is a complete replacement of its row, so a null-valued revision
returns null and never falls back to the previous value. A null close is a
missing bar, and a missing bar never erases a holding or simulates a sale:
:meth:`DatasetView.marks` carries the holding at its last valuation with
``valuation_status = "stale"``. Resolving that state through events, and the
accounting transitions built on it, are §3.6 and arrive at stage 3b.

Column conventions, fixed here because §3.2 declares fields and not dtypes:
``ts`` is a timezone-aware timestamp in ``America/New_York`` (the bar
convention of ``prism.io.store``); ``valid_from``, ``valid_to``,
``effective_date`` and ``announce_date`` are ISO ``YYYY-MM-DD`` date strings,
nullable where §3.2 says the interval is open; ``valid_at``, ``available_at``
and ``ingested_at`` are ISO 8601 instants with an offset, because they are
compared against ``cutoff(d)``, which is an instant. Membership and listing
intervals are inclusive of ``valid_to``, matching the reconstruction already
in ``prism.io.universe_sp500`` (``[start, end]`` at ``:280`` and ``:389``).
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from prism.io import sessions
from prism.io.events import TerminalClaim, distribution_amounts, terminal_claims, total_return_close
from prism.io.manifest import Manifest, ManifestError

Mode = Literal["replay_recorded", "research_vintage"]

MODES: tuple[str, ...] = ("replay_recorded", "research_vintage")

# How ``cutoff(d)`` was arrived at. Recorded on the view and copied into the
# result record: the last query of worked example B returns a different revision
# under a 16:00 clock than under a 21:30 clock, so a recorded cutoff without its
# basis is not enough to say why.
CutoffBasis = Literal["session_close", "recorded_decided_at", "declared_decision_time"]
CUTOFF_BASES: tuple[str, ...] = ("session_close", "recorded_decided_at", "declared_decision_time")

ValuationStatus = Literal["priced", "stale"]
VALUATION_STATUSES: tuple[str, ...] = ("priced", "stale")

#: The date a whole-history scan opens at. Earlier than any bar this repository
#: holds, so a terminal-claim scan never has to choose a lookback.
HISTORY_FLOOR = "1900-01-01"

BAR_TZ = "America/New_York"
_ET = ZoneInfo(BAR_TZ)

DEFAULT_BAR_INTERVAL = "1d"

# The nightly decision clock of the current paper book
# (``ops/paper_loop_nightly.sh:3-4``). It is the reference value §3.2 names for
# sessions recorded before the run-recording protocol added ``decided_at``; it
# is deliberately not a default, because a clock nobody declared is an
# assumption about what a past decision saw.
REFERENCE_DECISION_TIME = "21:30 America/New_York"

SELECTION_RULE = (
    "Within each source the highest revision of a key wins; a tie at (key, source, revision) is an ingestion "
    "defect and is refused. Across sources the highest-priority source with a visible row wins, where priority "
    "is position in the manifest source_priority list for that table. Revisions are never compared across "
    "sources. A row whose source the manifest does not order makes the view refuse."
)

SILENT_CORRECTION_ASSUMPTION = (
    "research_vintage: corrections the source made silently before this vintage was frozen are taken as known "
    "at the row's own time of record, because the vintage holds only the corrected value and no record of when "
    "it changed. Only corrections carried as their own revision are dated."
)

REPLAY_INGESTION_STATEMENT = (
    "replay_recorded: a row is visible when this workspace stored it at or before the decision's wall clock. "
    "Availability at the source is not consulted; a row the source published earlier and this workspace fetched "
    "later was not possessed and is not visible."
)

# numpy stores NaT as the smallest int64; the availability derivation fills
# unresolvable rows with it so the whole column converts in one pass.
_NAT_NS = np.iinfo("int64").min


class DatasetError(ValueError):
    """A read cannot be answered as asked: a missing file, a bad date, an unordered source."""


class StageNotAvailableError(NotImplementedError):
    """The capability is defined by the specification and delivered by a later stage.

    Carries the stage number so a caller can report it. Raised instead of an
    approximate answer: answering a question the engine cannot yet answer with
    the nearest thing it can is how a silent wrong number enters a record.
    """

    def __init__(self, message: str, *, stage: int) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True, slots=True)
class TableSpec:
    """How one table of §3.2 is keyed and when its rows become visible.

    ``availability`` names the derivation used when ``available_at`` is empty:
    ``announce_or_lag`` takes the announcement date when the table carries one,
    ``lag`` takes the session close of the time of record plus the source's
    ``lag_bars`` sessions, and ``time_of_record`` takes the record time itself
    (the ``valid_at`` tables, whose record time is already an instant).
    """

    name: str
    key: tuple[str, ...]
    time_of_record: str
    availability: Literal["announce_or_lag", "lag", "time_of_record"]
    record_kind: Literal["date", "instant", "timestamp"]


TABLES: dict[str, TableSpec] = {
    "securities": TableSpec("securities", ("sid", "valid_at"), "valid_at", "time_of_record", "instant"),
    "listings": TableSpec("listings", ("sid", "vendor", "valid_from"), "valid_from", "announce_or_lag", "date"),
    "membership": TableSpec("membership", ("index_id", "sid", "valid_from"), "valid_from", "announce_or_lag", "date"),
    "events": TableSpec(
        "events", ("sid", "event_type", "effective_date"), "effective_date", "announce_or_lag", "date"
    ),
    "adjustments": TableSpec("adjustments", ("sid", "effective_date", "kind"), "effective_date", "lag", "date"),
    "capabilities": TableSpec("capabilities", ("sid", "valid_at"), "valid_at", "time_of_record", "instant"),
    "account_state": TableSpec("account_state", ("account_id", "valid_at"), "valid_at", "time_of_record", "instant"),
    "series": TableSpec("series", ("series_id", "ts"), "ts", "lag", "timestamp"),
    "bars": TableSpec("bars", ("ts",), "ts", "lag", "timestamp"),
}


def _as_date(value: str | dt.date | pd.Timestamp) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    return dt.date.fromisoformat(str(value))


@lru_cache(maxsize=4096)
def _session_close_instant(date: dt.date) -> pd.Timestamp:
    """The close of ``date``'s own session as a tz-aware instant, early closes included."""
    return pd.Timestamp(dt.datetime.combine(date, sessions.session_close(date), tzinfo=_ET))


@lru_cache(maxsize=4096)
def _availability_instant(date: dt.date, lag_bars: int) -> pd.Timestamp:
    """Session close of ``date`` plus ``lag_bars`` sessions.

    A record time that is not a session (an announcement on a weekend, a pay
    date on a holiday) rolls forward to the next session before the lag is
    counted: rolling back would publish the row before the event it describes.
    The rule is recorded in the manifest ``assumptions``.
    """
    session = date if sessions.is_session(date) else sessions.next_session(date)
    for _ in range(lag_bars):
        session = sessions.next_session(session)
    return _session_close_instant(session)


def _dates(values: pd.Series) -> pd.Series:
    """Parse an ISO-date column into naive ``datetime64[ns]``, leaving nulls as ``NaT``."""
    parsed = pd.to_datetime(values, errors="raise", format="ISO8601")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _instants(values: pd.Series) -> pd.Series:
    """Parse an ISO-instant column into tz-aware ``America/New_York`` timestamps."""
    parsed = pd.to_datetime(values, errors="raise", format="ISO8601", utc=True)
    return parsed.dt.tz_convert(BAR_TZ)


def _blank(column: pd.Series) -> pd.Series:
    """True where a nullable string column carries nothing: NA, NaT, or the empty string."""
    return column.isna() | (column.astype("string").fillna("") == "")


def parse_decision_time(value: str) -> tuple[dt.time, ZoneInfo]:
    """Parse a declared ``replay_recorded`` clock: ``"21:30 America/New_York"``.

    A clock without a zone is refused. A wall clock is not an instant until a
    zone says which instant it is, and the difference is a whole DST hour twice
    a year — long enough to move a row across the cutoff.
    """
    parts = str(value).split()
    if len(parts) != 2:
        raise DatasetError(
            f"decision_time {value!r} must be a clock and an IANA zone, as in {REFERENCE_DECISION_TIME!r}; a clock "
            "without a zone does not name an instant (§3.2)"
        )
    clock, zone = parts
    try:
        parsed = dt.time.fromisoformat(clock)
    except ValueError as exc:
        raise DatasetError(f"decision_time {value!r} carries {clock!r}, which is not an ISO clock") from exc
    if parsed.tzinfo is not None:
        raise DatasetError(f"decision_time {value!r} carries an offset on the clock; declare the IANA zone instead")
    try:
        info = ZoneInfo(zone)
    except Exception as exc:  # ZoneInfoNotFoundError and the loader's own errors
        raise DatasetError(f"decision_time {value!r} names {zone!r}, which is not an IANA zone") from exc
    return parsed, info


def read_decided_at(path: Path) -> dict[str, str]:
    """Recorded decision wall clocks from a JSONL decision ledger: date -> ``decided_at``.

    The run-recording protocol of §3.12 writes ``decided_at`` on every decision
    row going forward. Rows written before it — the existing paper-loop ledgers,
    which carry only ``decision_bar``, ``equity``, ``reference_prices``,
    ``refresh_bar`` and ``targets`` — carry no wall clock and are skipped here,
    so the evaluation's declared ``decision_time`` covers them and a date with
    neither is refused by the view rather than defaulted.
    """
    recorded: dict[str, str] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{number} is not JSON: {exc}") from exc
        stamp = row.get("decided_at")
        if stamp in (None, ""):
            continue
        bar = row.get("decision_bar", row.get("decision_date"))
        if bar in (None, ""):
            raise DatasetError(f"{path}:{number} carries decided_at but names no decision bar")
        recorded[_as_date(bar).isoformat()] = str(stamp)
    return recorded


@dataclass(frozen=True, slots=True)
class BarPanels:
    """Every listed bar file as wide ``ts x sid`` frames, plus what decides their visibility.

    ``available`` is aligned to the panel index and is the instant each bar
    became knowable. ``ingested_at`` is the single ingestion instant shared by
    every bar row, or ``None`` when the files do not share one — in which case
    the ``replay_recorded`` predicate is not a property of ``ts`` alone and the
    cache cannot answer that mode.
    """

    fields: dict[str, pd.DataFrame]
    available: pd.Series
    ingested_at: pd.Timestamp | None


class Dataset:
    """A data root plus the manifest that says what it contains.

    ``root`` is ``workspace.data``; manifest file paths are relative to it.
    Tables are read once and cached on the dataset, so several views of one
    dataset share one parse.

    ``decided_at`` maps a decision date onto the wall clock the decision was
    recorded at, as :func:`read_decided_at` reads it from a decision ledger. It
    is consulted only by ``replay_recorded`` views and is the first source of
    ``cutoff(d)``; a declared ``decision_time`` covers the dates it does not
    carry.
    """

    def __init__(
        self,
        root: Path,
        manifest: Manifest,
        *,
        verify_files: bool = False,
        decided_at: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.bar_interval = DEFAULT_BAR_INTERVAL
        if not self.root.is_dir():
            raise DatasetError(f"dataset root {self.root} is not a directory")
        if verify_files:
            manifest.verify_files(self.root)
        self.decided_at: dict[str, str] = {
            _as_date(day).isoformat(): str(stamp) for day, stamp in (decided_at or {}).items()
        }
        self._tables: dict[str, pd.DataFrame] = {}
        self.visibility_cache: dict[str, tuple[pd.Series, pd.Series]] = {}
        self.ingestion_cache: dict[str, pd.Series] = {}
        self._bar_panels: BarPanels | None = None
        self._bar_panels_built = False

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        verify_files: bool = False,
        decided_at: Mapping[str, str] | None = None,
    ) -> "Dataset":
        """Open the dataset whose ``manifest.json`` lives at ``root``."""
        return cls(
            root,
            Manifest.read(Path(root) / "manifest.json"),
            verify_files=verify_files,
            decided_at=decided_at,
        )

    def read_table(self, relative_path: str) -> pd.DataFrame:
        """Read one manifest-listed Parquet file, refusing anything the manifest does not list."""
        if not self.manifest.lists_file(relative_path):
            raise DatasetError(
                f"{relative_path} is not listed in manifest {self.manifest.id}; a view reads only the files its "
                "manifest froze"
            )
        cached = self._tables.get(relative_path)
        if cached is not None:
            return cached
        path = self.root / relative_path
        if not path.is_file():
            raise DatasetError(f"manifest {self.manifest.id} lists {relative_path}, which is absent from {self.root}")
        frame = pd.read_parquet(path)
        self._tables[relative_path] = frame
        return frame

    def has_table(self, name: str) -> bool:
        return self.manifest.lists_file(f"{name}.parquet")

    def bars_path(self, sid: str) -> str:
        return f"bars/{sid}_{self.bar_interval}.parquet"

    def view(
        self,
        mode: Mode,
        decision_date: str,
        *,
        vintage: str | None = None,
        decision_time: str | None = None,
        decision_offset_minutes: int = 0,
    ) -> "DatasetView":
        """Pin the dataset to one decision date under one visibility mode.

        ``decision_offset_minutes`` is the ``[sample]`` block's offset from the
        session close (§3.13); it defaults to 0, which is the value an
        evaluation without a registration takes, and belongs to
        ``research_vintage`` alone.
        """
        if mode not in MODES:
            raise DatasetError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "research_vintage":
            if vintage is None:
                raise DatasetError("vintage is required when mode == 'research_vintage' (§3.2)")
            if self.manifest.vintage is None:
                raise DatasetError(
                    f"manifest {self.manifest.id} carries no vintage label; it cannot be read as of one"
                )
            if vintage != self.manifest.vintage:
                raise DatasetError(
                    f"vintage {vintage!r} does not name this manifest, whose vintage is {self.manifest.vintage!r}"
                )
            if decision_time is not None:
                raise DatasetError(
                    "decision_time belongs to replay_recorded; research_vintage cuts at the session close"
                )
        else:
            if vintage is not None:
                raise DatasetError(
                    "vintage is refused in replay_recorded: that mode reproduces what this workspace possessed at a "
                    "wall clock, which is not a frozen file list. Passing it is an error, not a silent no-op (§3.2)"
                )
            if decision_offset_minutes:
                raise DatasetError(
                    "decision_offset_minutes is the [sample] block's offset from the session close and belongs to "
                    "research_vintage; replay_recorded cuts at the recorded decision wall clock"
                )
        return DatasetView(
            self,
            decision_date,
            mode=mode,
            vintage=vintage,
            decision_time=decision_time,
            decision_offset_minutes=decision_offset_minutes,
        )

    # -- the wide-bar fast path -------------------------------------------

    def bar_panels(self) -> BarPanels | None:
        """Every listed bar file as wide ``ts x sid`` frames, read once, or ``None``.

        A walk-forward asks for the same panel on every decision bar, and
        rebuilding it from the per-sid files each time is the whole cost of an
        evaluation on a real universe: 574 file filters, a concat and two
        pivots, per bar. The frames are identical between decision bars — only
        the visible *prefix* changes — so they are built once here and the view
        slices them.

        ``None`` means the fast path does not apply and the caller must read
        the files. It applies only when the shape makes the visible set a
        prefix of ``ts`` that is the same for every sid: one revision per
        ``(sid, ts)``, no row declaring its own ``available_at``, and one
        availability lag across the bar sources. Anything else — a revision to
        choose between, a vendor that publishes late for some names only — is
        the general case, which the per-file predicate answers exactly and this
        cache must not approximate.
        """
        if self._bar_panels_built:
            return self._bar_panels
        self._bar_panels_built = True
        self._bar_panels = self._build_bar_panels()
        return self._bar_panels

    def _build_bar_panels(self) -> BarPanels | None:
        prefix = "bars/"
        suffix = f"_{self.bar_interval}.parquet"
        paths = [path for path, _ in self.manifest.files if path.startswith(prefix) and path.endswith(suffix)]
        if not paths:
            return None
        lags: set[int] = set()
        ingestions: set[pd.Timestamp] = set()
        ingestion_uniform = True
        seen = False
        fields = ("open", "high", "low", "close", "volume")
        columns: dict[str, dict[str, pd.Series]] = {field: {} for field in fields}
        for relative in paths:
            sid = relative[len(prefix) : -len(suffix)]
            frame = self.read_table(relative)
            if frame.empty:
                continue
            declared = frame.get("available_at")
            if declared is not None and not bool(_blank(declared).all()):
                return None
            if "source" not in frame.columns:
                return None
            lags |= {self.manifest.lag_bars(source) for source in frame["source"].astype(str).unique()}
            if len(lags) > 1:
                return None
            if ingestion_uniform and "ingested_at" in frame.columns and not bool(_blank(frame["ingested_at"]).any()):
                ingestions |= set(_instants(frame["ingested_at"]).unique())
                ingestion_uniform = len(ingestions) == 1
            else:
                ingestion_uniform = False
            stamps = pd.DatetimeIndex(pd.to_datetime(frame["ts"], errors="raise", utc=True)).tz_convert(BAR_TZ)
            if not stamps.is_unique:
                return None
            for field in fields:
                if field in frame.columns:
                    columns[field][sid] = pd.Series(
                        pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float), index=stamps
                    )
            seen = True
        if not seen or not lags:
            return None
        panels = {field: pd.DataFrame(series).sort_index() for field, series in columns.items() if series}
        if not panels:
            return None
        index = panels["close"].index if "close" in panels else next(iter(panels.values())).index
        panels = {field: panel.reindex(index=index) for field, panel in panels.items()}
        lag = next(iter(lags))
        available = pd.Series(
            [_availability_instant(stamp.date(), lag) for stamp in index], index=index, dtype="object"
        )
        ingested = next(iter(ingestions)) if ingestion_uniform and len(ingestions) == 1 else None
        return BarPanels(fields=panels, available=available, ingested_at=ingested)


class DatasetView:
    """One dataset as of one decision date, under one visibility mode (§3.2).

    The mode decides the predicate and nothing else: both modes run the same
    selection rule over the rows they admit, and every accessor bounds its own
    horizon at the decision date, so no accessor can return a bar the decision
    could not have seen.
    """

    def __init__(
        self,
        dataset: Dataset,
        decision_date: str,
        *,
        mode: Mode = "research_vintage",
        vintage: str | None = None,
        decision_time: str | None = None,
        decision_offset_minutes: int = 0,
    ) -> None:
        self.dataset = dataset
        self.mode: Mode = mode
        self.vintage: str | None = vintage
        self.decision_time: str | None = decision_time
        self.decision_offset_minutes = int(decision_offset_minutes)
        self.decision_day = _as_date(decision_date)
        if not sessions.is_session(self.decision_day):
            raise DatasetError(
                f"decision date {self.decision_day.isoformat()} is not a session on the bundled table "
                f"({sessions.status(self.decision_day)}); a decision is taken at a session close"
            )
        self.decision_date = self.decision_day.isoformat()
        self.cutoff_ts, self.cutoff_basis = self._resolve_cutoff()
        self.cutoff = self.cutoff_ts.isoformat()

    def _resolve_cutoff(self) -> tuple[pd.Timestamp, CutoffBasis]:
        """``cutoff(d)``, per mode, never left free (§3.2)."""
        if self.mode == "research_vintage":
            close = _session_close_instant(self.decision_day)
            return close + pd.Timedelta(minutes=self.decision_offset_minutes), "session_close"
        recorded = self.dataset.decided_at.get(self.decision_date)
        if recorded is not None:
            stamp = pd.Timestamp(recorded)
            if stamp.tzinfo is None:
                raise DatasetError(
                    f"the recorded decided_at for {self.decision_date} is {recorded!r}, which carries no offset; a "
                    "wall clock without a zone does not name an instant"
                )
            return stamp.tz_convert(BAR_TZ), "recorded_decided_at"
        if self.decision_time is not None:
            clock, zone = parse_decision_time(self.decision_time)
            stamp = pd.Timestamp(dt.datetime.combine(self.decision_day, clock, tzinfo=zone))
            return stamp.tz_convert(BAR_TZ), "declared_decision_time"
        raise DatasetError(
            f"replay_recorded needs cutoff({self.decision_date}) and has neither a recorded decided_at nor a "
            f"declared decision_time; declare one as a clock and an IANA zone (the current book's nightly schedule "
            f"is {REFERENCE_DECISION_TIME!r}) or supply the decision ledger. The view refuses rather than assuming "
            "a clock (§3.2)"
        )

    def __repr__(self) -> str:
        return (
            f"DatasetView(mode={self.mode!r}, decision_date={self.decision_date!r}, cutoff={self.cutoff!r}, "
            f"cutoff_basis={self.cutoff_basis!r})"
        )

    # ---- visibility ----------------------------------------------------

    def _availability(self, spec: TableSpec, frame: pd.DataFrame) -> pd.Series:
        """The instant each row became knowable: ``available_at`` when the source declares it, else derived.

        The derivation walks the session table, so it is done once per distinct
        ``(basis date, lag)`` pair rather than once per row: a hundred bar
        files share a few hundred distinct session dates.
        """
        record = self._record_time(spec, frame)
        if spec.availability == "time_of_record":
            derived = _instants(frame[spec.time_of_record])
        else:
            basis = record
            if spec.availability == "announce_or_lag" and "announce_date" in frame.columns:
                announce = _dates(frame["announce_date"])
                basis = announce.fillna(record)
                # An announced row is knowable on its announcement, with no
                # further source lag; only the derived-from-record rows lag.
                lags = pd.Series(0, index=frame.index, dtype="int64").where(announce.notna(), self._lags(frame))
            else:
                lags = self._lags(frame)
            nanos = np.full(len(frame), _NAT_NS, dtype="int64")
            days = basis.dt.date.to_numpy()
            for lag in sorted(set(lags.unique())):
                selected = ((lags == lag) & basis.notna()).to_numpy()
                if not bool(selected.any()):
                    continue
                unique, inverse = np.unique(days[selected], return_inverse=True)
                mapped = np.array([_availability_instant(day, int(lag)).value for day in unique], dtype="int64")
                nanos[selected] = mapped[inverse]
            derived = pd.Series(
                pd.DatetimeIndex(nanos.view("datetime64[ns]"), tz="UTC").tz_convert(BAR_TZ), index=frame.index
            )
        declared = frame.get("available_at")
        if declared is None:
            return derived
        parsed = _instants(declared.where(~_blank(declared), other=pd.NaT))
        return parsed.fillna(derived)

    def _lags(self, frame: pd.DataFrame) -> pd.Series:
        """Each row's source lag in sessions, read from the manifest and never from a row."""
        manifest = self.dataset.manifest
        lookup = {source: manifest.lag_bars(source) for source in frame["source"].astype(str).unique()}
        return frame["source"].astype(str).map(lookup).astype("int64")

    def _record_time(self, spec: TableSpec, frame: pd.DataFrame) -> pd.Series:
        column = frame[spec.time_of_record]
        if spec.record_kind == "instant":
            return _instants(column).dt.tz_localize(None)
        if spec.record_kind == "timestamp":
            parsed = pd.to_datetime(column, errors="raise")
            return parsed.dt.tz_localize(None) if getattr(parsed.dt, "tz", None) is not None else parsed
        return _dates(column)

    def _check_table(self, spec: TableSpec, frame: pd.DataFrame, relative_path: str) -> None:
        """Refuse a table the manifest does not order, or one carrying an ingestion defect.

        Both are properties of the file rather than of a decision date, so they
        run once per file and refuse for every view of the dataset. A duplicate
        ``(key, source, revision)`` is checked over the whole file and not over
        the visible slice: two such rows can differ in their availability, so
        checking the slice would let a defect hide until the day it changed an
        answer.
        """
        order = set(self.dataset.manifest.priority_for(spec.name))
        unknown = sorted(set(frame["source"].astype(str)) - order)
        if unknown:
            raise DatasetError(
                f"table {spec.name!r} carries rows from {unknown}, which manifest {self.dataset.manifest.id} does "
                "not order; the view refuses rather than picking a lexical winner"
            )
        key = [column for column in spec.key if column in frame.columns]
        if not key:
            raise DatasetError(f"{relative_path} carries none of the key columns {spec.key} table {spec.name!r} needs")
        if "revision" not in frame.columns:
            raise DatasetError(f"{relative_path} carries no revision column; §3.2 keys every row by its revision")
        duplicated = frame.duplicated(subset=key + ["source", "revision"], keep=False)
        if bool(duplicated.any()):
            offending = (
                frame.loc[duplicated, key + ["source", "revision"]].drop_duplicates().head(3).to_dict("records")
            )
            raise DatasetError(
                f"{relative_path} carries more than one row for {offending}; two rows of one table with the same "
                "(key, source, revision) are an ingestion defect and the view refuses to read them (§3.2)"
            )

    def _visibility(self, spec: TableSpec, relative_path: str, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Record time and availability for one file, cached on the dataset.

        Neither depends on the decision date — only on the row and the
        manifest — so several views of one dataset parse each file once.
        """
        cached = self.dataset.visibility_cache.get(relative_path)
        if cached is None:
            self._check_table(spec, frame, relative_path)
            cached = (self._record_time(spec, frame), self._availability(spec, frame))
            self.dataset.visibility_cache[relative_path] = cached
        return cached

    def _ingested(self, spec: TableSpec, relative_path: str, frame: pd.DataFrame) -> pd.Series:
        """When this workspace stored each row, cached on the dataset. ``replay_recorded`` only."""
        cached = self.dataset.ingestion_cache.get(relative_path)
        if cached is None:
            self._check_table(spec, frame, relative_path)
            column = frame.get("ingested_at")
            if column is None:
                raise DatasetError(
                    f"{relative_path} carries no ingested_at column; replay_recorded visibility is a statement about "
                    "when this workspace stored a row and cannot be answered without it"
                )
            blank = _blank(column)
            if bool(blank.any()):
                raise DatasetError(
                    f"{relative_path} leaves ingested_at empty on {int(blank.sum())} row(s); a row with no ingestion "
                    "time has no replay_recorded visibility and the view refuses rather than guessing one"
                )
            cached = _instants(column)
            self.dataset.ingestion_cache[relative_path] = cached
        return cached

    def _require_first_ingestion(self, first: pd.Timestamp, label: str) -> None:
        """Refuse a decision date earlier than the first ingestion of a table it reads (§3.2)."""
        if first > self.cutoff_ts:
            raise DatasetError(
                f"replay_recorded refuses decision date {self.decision_date}: the first ingestion of {label} is "
                f"{first.isoformat()}, later than cutoff {self.cutoff} ({self.cutoff_basis}). This workspace held "
                "nothing of that table at the decision, and an empty frame would read as a universe with no data "
                "rather than as a question that cannot be asked"
            )

    def _visible(self, spec: TableSpec, relative_path: str) -> pd.DataFrame:
        """Rows of one manifest-listed file visible at this view's decision date and cutoff."""
        frame = self.dataset.read_table(relative_path)
        if frame.empty:
            return frame
        if self.mode == "replay_recorded":
            ingested = self._ingested(spec, relative_path, frame)
            self._require_first_ingestion(ingested.min(), self._label(spec, relative_path))
            keep = (ingested <= self.cutoff_ts).to_numpy()
        else:
            record, available = self._visibility(spec, relative_path, frame)
            # A null record time is the floor_unknown case of §3.2: the interval
            # opened before the sample, so the row was knowable throughout it.
            # Its availability is null for the same reason and is treated the
            # same way.
            on_or_before = record.isna() | (record.dt.normalize() <= pd.Timestamp(self.decision_day))
            known = available.isna() | (available <= self.cutoff_ts)
            keep = (on_or_before & known).to_numpy()
        return self._select(spec, frame.loc[keep], relative_path)

    @staticmethod
    def _label(spec: TableSpec, relative_path: str) -> str:
        return spec.name if relative_path == f"{spec.name}.parquet" else f"{spec.name} ({relative_path})"

    def _select(self, spec: TableSpec, frame: pd.DataFrame, relative_path: str) -> pd.DataFrame:
        """The deterministic selection of §3.2: highest revision within a source, highest-priority source.

        Sorting by key, then by priority rank descending, then by revision
        ascending puts the winning row last within each key group: the last
        block is the smallest rank, and within it the largest revision. The
        selected rows are returned in the file's own order, so a read is
        reproducible byte for byte.
        """
        if frame.empty:
            return frame
        key = [column for column in spec.key if column in frame.columns]
        rank = {source: index for index, source in enumerate(self.dataset.manifest.priority_for(spec.name))}
        sources = frame["source"].astype(str)
        unordered = sorted(set(sources) - set(rank))
        if unordered:
            raise DatasetError(
                f"table {spec.name!r} carries rows from {unordered}, which manifest {self.dataset.manifest.id} does "
                "not order; the view refuses rather than picking a lexical winner"
            )
        work = frame.assign(
            _priority=sources.map(rank).astype("int64"),
            _revision=pd.to_numeric(frame["revision"], errors="raise").astype("int64"),
        )
        if not work.duplicated(subset=key, keep=False).any():
            return frame
        work = work.sort_values(
            key + ["_priority", "_revision"],
            ascending=[True] * len(key) + [False, True],
            kind="mergesort",
        )
        chosen = work.groupby(key, as_index=False, sort=False, dropna=False).tail(1)
        return frame.loc[sorted(chosen.index)]

    def _table(self, name: str) -> pd.DataFrame:
        return self._visible(TABLES[name], f"{name}.parquet")

    def _require_within_horizon(self, end: str, what: str) -> dt.date:
        end_day = _as_date(end)
        if end_day > self.decision_day:
            raise DatasetError(
                f"{what} asked for {end_day.isoformat()}, later than decision date {self.decision_date}; a view "
                "never returns a bar beyond the decision it was opened for"
            )
        return end_day

    # ---- accessors -----------------------------------------------------

    def bars(self, sids: Sequence[str], end: str) -> pd.DataFrame:
        """Visible bars for ``sids`` up to and including ``end`` (``end <= decision_date``).

        Long form, one row per ``(sid, ts)``, sorted by ``(sid, ts)``, carrying
        ``price_basis`` so a caller can tell a raw series from one the vendor
        had already rebased. A row selected from a null-valued revision is
        returned with its nulls: the revision is a complete replacement, so the
        previous value is gone and the bar is missing (§3.2).
        """
        end_day = self._require_within_horizon(end, "bars")
        spec = TABLES["bars"]
        frames: list[pd.DataFrame] = []
        for sid in sids:
            relative = self.dataset.bars_path(sid)
            if not self.dataset.manifest.lists_file(relative):
                raise DatasetError(f"manifest {self.dataset.manifest.id} lists no bars for sid {sid!r}")
            visible = self._visible(spec, relative)
            visible = visible.loc[visible["ts"].dt.tz_convert(BAR_TZ).dt.date <= end_day]
            if "sid" not in visible.columns:
                visible = visible.assign(sid=sid)
            frames.append(visible)
        if not frames:
            return pd.DataFrame(columns=["sid", "ts", "open", "high", "low", "close", "volume", "price_basis"])
        out = pd.concat(frames, ignore_index=True)
        columns = ["sid"] + [c for c in out.columns if c != "sid"]
        return out[columns].sort_values(["sid", "ts"]).reset_index(drop=True)

    def panel(self, sids: Sequence[str], end: str, field: str = "close") -> pd.DataFrame:
        """One column per sid of ``field``, indexed by ``ts``.

        The wide form the retained reproduction engine consumes
        (``prism.execution.target_weights``). Not part of the §3.2 accessor
        list: it is a reshaping of ``bars`` and applies the same predicate —
        through :meth:`Dataset.bar_panels` when that cache can answer exactly,
        and by reading the files when it cannot.
        """
        fast = self._panel_from_cache(list(sids), end, field)
        if fast is not None:
            return fast
        long = self.bars(list(sids), end)
        if long.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz=BAR_TZ), columns=list(sids), dtype=float)
        wide = long.pivot(index="ts", columns="sid", values=field)
        wide = wide.reindex(columns=list(sids))
        wide.columns.name = None
        return wide.sort_index()

    def _panel_from_cache(self, sids: list[str], end: str, field: str) -> pd.DataFrame | None:
        """The wide panel from :meth:`Dataset.bar_panels`, or ``None`` if the cache cannot answer."""
        end_day = self._require_within_horizon(end, "bars")
        cached = self.dataset.bar_panels()
        if cached is None:
            return None
        panel = cached.fields.get(field)
        if panel is None:
            return None
        missing = [sid for sid in sids if sid not in panel.columns]
        if missing:
            # The manifest may still list the file (an empty one, say); the
            # per-file path answers and refuses for the right reason.
            for sid in missing:
                if not self.dataset.manifest.lists_file(self.dataset.bars_path(sid)):
                    raise DatasetError(f"manifest {self.dataset.manifest.id} lists no bars for sid {sid!r}")
            return None
        index = panel.index
        days = index.tz_convert(BAR_TZ).date
        if self.mode == "replay_recorded":
            if cached.ingested_at is None:
                return None
            self._require_first_ingestion(cached.ingested_at, "bars")
            known = np.ones(len(index), dtype=bool)
        else:
            known = cached.available.to_numpy() <= self.cutoff_ts
        out = panel.loc[(days <= end_day) & known, list(sids)]
        out.columns.name = None
        return out

    def total_return(self, sids: Sequence[str], end: str) -> pd.DataFrame:
        """The as-traded closes reinvested at each visible ``cash_distribution`` ex-date.

        The wide panel :meth:`panel` returns, on the same index and columns, so
        a component declaring ``scoring_basis = "total_return"`` scores on the
        same calendar it would have scored on under ``price``. Scoring only:
        costs, participation, fills and every limit are computed on the
        as-traded close (§3.6).

        Only distributions this view can see enter it — the events accessor
        applies the same visibility predicate — so a total-return level is
        never revised upward by a distribution announced after the decision.
        """
        close = self.panel(list(sids), end, "close")
        if close.empty:
            return close
        if not self.dataset.has_table("events"):
            raise DatasetError(
                f"manifest {self.dataset.manifest.id} lists no events table, so a total-return series would be the "
                "price series under a total-return label; ingest cash distributions before scoring on them (§3.2)"
            )
        start = close.index[0].date().isoformat()
        distributions = distribution_amounts(self.events(list(sids), start, end), close)
        return total_return_close(close, distributions)

    def marks(self, sids: Sequence[str], end: str | None = None) -> pd.DataFrame:
        """Each sid's last visible priced bar, with how stale it is (§3.2, §3.6).

        The read side of "a missing bar never erases a holding or simulates a
        sale". A sid whose bar for ``end`` is absent — never ingested, halted,
        or withdrawn by a null-valued revision — is carried at its last
        valuation with ``valuation_status = "stale"`` and ``sessions_stale``
        counting the sessions since that valuation on the bundled session
        table. Nothing here is zero and nothing is forward-filled into the
        panel: the caller marks the holding, it does not sell it.

        A sid with no priced bar at or before ``end`` is refused rather than
        returned as a null mark, because a holding valued at nothing is exactly
        the silent erasure this rule exists to prevent. Resolving a stale
        holding through a terminal payout, a cessation or an OTC continuation
        is §3.6 and arrives at stage 3b with ``execution/accounting.py``; the
        ``unresolved_claim`` status that goes with it is set there.
        """
        horizon = self.decision_date if end is None else end
        end_day = self._require_within_horizon(horizon, "marks")
        wanted = list(sids)
        frame = self.bars(wanted, horizon)
        rows: list[dict[str, Any]] = []
        for sid in wanted:
            priced = frame.loc[(frame["sid"].astype(str) == sid) & frame["close"].notna()]
            if priced.empty:
                raise DatasetError(
                    f"sid {sid!r} has no priced bar visible at {self.cutoff} on or before {end_day.isoformat()}; the "
                    "view refuses rather than valuing a holding at nothing (§3.2)"
                )
            last = priced.iloc[-1]
            bar_day = last["ts"].tz_convert(BAR_TZ).date() if last["ts"].tzinfo else last["ts"].date()
            elapsed = [day for day in sessions.sessions_between(bar_day, end_day) if day > bar_day]
            rows.append(
                {
                    "sid": sid,
                    "last_price": float(last["close"]),
                    "last_price_bar": bar_day.isoformat(),
                    "sessions_stale": len(elapsed),
                    "valuation_status": "priced" if not elapsed else "stale",
                }
            )
        return pd.DataFrame(rows).set_index("sid")

    def events(self, sids: Sequence[str], start: str, end: str) -> pd.DataFrame:
        """Visible ``events`` rows for ``sids`` with ``effective_date`` in ``[start, end]``."""
        return self._dated_rows("events", sids, start, end, "effective_date")

    def terminal_claims(self, sids: Sequence[str], end: str | None = None) -> dict[str, TerminalClaim]:
        """What this view can see about each sid's terminal claim (§3.2, §3.6).

        The companion to :meth:`marks`. ``marks`` answers "how old is the price
        I am carrying this holding at"; this answers "is there still an
        instrument to carry, and does anyone know what the claim is worth".
        A delisted sid with no ``terminal_payout`` comes back ``open``, one
        whose payout says ``{"status": "unknown"}`` comes back ``unresolved``,
        and a record built while either holds is written ``incomplete``
        (:func:`prism.io.events.incomplete_reasons`).

        The visibility predicate is the events accessor's, so a payout recorded
        after the decision does not retroactively resolve a claim the decision
        could not have known was resolved. ``end`` defaults to the decision
        date and is never beyond it.
        """
        horizon = self.decision_date if end is None else end
        self._require_within_horizon(horizon, "terminal_claims")
        wanted = list(sids)
        if not self.dataset.has_table("events"):
            raise DatasetError(
                f"manifest {self.dataset.manifest.id} lists no events table, so every claim would read as still "
                "trading and a delisted holding would be marked at a price that no longer exists (§3.2)"
            )
        # A claim is a statement about the whole history of the sid, not about
        # a window: a cessation five years before the decision still ends the
        # series, so the scan opens at the history floor rather than at a
        # lookback the caller would have to choose.
        return terminal_claims(self.events(wanted, HISTORY_FLOOR, horizon), wanted)

    def adjustments(self, sids: Sequence[str], start: str, end: str) -> pd.DataFrame:
        """Visible ``adjustments`` rows for ``sids`` with ``effective_date`` in ``[start, end]``."""
        return self._dated_rows("adjustments", sids, start, end, "effective_date")

    def _dated_rows(self, name: str, sids: Sequence[str], start: str, end: str, column: str) -> pd.DataFrame:
        end_day = self._require_within_horizon(end, name)
        start_day = _as_date(start)
        frame = self._table(name)
        if frame.empty:
            return frame
        dates = _dates(frame[column]).dt.date
        selected = frame.loc[frame["sid"].isin(list(sids)) & (dates >= start_day) & (dates <= end_day)]
        return selected.sort_values(["sid", column]).reset_index(drop=True)

    def membership(self, index_id: str) -> set[str]:
        """Sids whose membership interval contains the decision date under this view.

        Intervals are inclusive of ``valid_to``; a null ``valid_from`` is the
        ``floor_unknown`` case (the interval opened before the sample) and a
        null ``valid_to`` is an open interval.
        """
        frame = self._table("membership")
        if frame.empty:
            return set()
        rows = frame.loc[frame["index_id"] == index_id]
        if rows.empty:
            return set()
        day = pd.Timestamp(self.decision_day)
        starts = _dates(rows["valid_from"])
        ends = _dates(rows["valid_to"])
        contains = (starts.isna() | (starts <= day)) & (ends.isna() | (ends >= day))
        return set(rows.loc[contains, "sid"].astype(str))

    def listings(self, sids: Sequence[str]) -> pd.DataFrame:
        frame = self._table("listings")
        if frame.empty:
            return frame
        return (
            frame.loc[frame["sid"].isin(list(sids))]
            .sort_values(["sid", "vendor", "valid_from"])
            .reset_index(drop=True)
        )

    def resolve(self, ticker: str, on: str) -> str:
        """The single sid whose listing for ``ticker`` is valid on ``on``; refuses zero or several."""
        on_day = self._require_within_horizon(on, "resolve")
        frame = self._table("listings")
        if frame.empty:
            raise DatasetError(f"no listing resolves {ticker!r} on {on_day.isoformat()}: the listings table is empty")
        day = pd.Timestamp(on_day)
        starts = _dates(frame["valid_from"])
        ends = _dates(frame["valid_to"])
        matches = frame.loc[
            (frame["ticker"].astype(str) == ticker) & (starts.isna() | (starts <= day)) & (ends.isna() | (ends >= day))
        ]
        candidates = sorted(set(matches["sid"].astype(str)))
        if len(candidates) != 1:
            raise DatasetError(
                f"{ticker!r} on {on_day.isoformat()} resolves to {candidates or 'no sid'}; a ticker must be "
                "single-valued on a date under this view"
            )
        return candidates[0]

    def securities(self, sids: Sequence[str]) -> pd.DataFrame:
        """Latest visible ``securities`` row per sid (``valid_at <= cutoff``), indexed by sid."""
        return self._latest_by("securities", "sid", sids)

    def capabilities(self, sids: Sequence[str]) -> pd.DataFrame:
        """Latest visible ``capabilities`` row per sid (``valid_at <= cutoff``), indexed by sid."""
        return self._latest_by("capabilities", "sid", sids)

    def _latest_by(self, name: str, key: str, keys: Sequence[str]) -> pd.DataFrame:
        """The greatest ``valid_at`` at or before the cutoff, after the selection rule (§3.2).

        The ``valid_at`` bound is applied here rather than left to the mode's
        predicate because ``replay_recorded`` cuts on ingestion: a capability
        row this workspace stored before the decision but stamped valid after
        it would otherwise be treated as current.
        """
        frame = self._table(name)
        wanted = list(keys)
        if frame.empty:
            return frame
        selected = frame.loc[frame[key].isin(wanted)].copy()
        if selected.empty:
            return selected
        selected["_valid_at"] = _instants(selected["valid_at"])
        selected = selected.loc[selected["_valid_at"] <= self.cutoff_ts]
        selected = selected.sort_values([key, "_valid_at"]).groupby(key, as_index=False).tail(1)
        missing = sorted(set(wanted) - set(selected[key].astype(str)))
        if missing:
            raise DatasetError(
                f"table {name!r} has no row visible at {self.cutoff} for {missing}; the view refuses rather than "
                "returning a partial frame"
            )
        return selected.drop(columns="_valid_at").set_index(key).loc[wanted]

    def series(self, series_id: str, start: str, end: str) -> pd.Series:
        """Visible rows of the ``series`` table for ``series_id``, indexed by ``ts``."""
        end_day = self._require_within_horizon(end, "series")
        start_day = _as_date(start)
        if not self.dataset.has_table("series"):
            raise DatasetError(
                f"manifest {self.dataset.manifest.id} lists no series table; the benchmark series a registration "
                "names must be ingested before it can be read"
            )
        frame = self._table("series")
        rows = frame.loc[frame["series_id"] == series_id]
        if rows.empty:
            raise DatasetError(f"series {series_id!r} has no visible row at {self.cutoff}")
        stamps = pd.to_datetime(rows["ts"])
        selected = rows.loc[(stamps.dt.date >= start_day) & (stamps.dt.date <= end_day)].sort_values("ts")
        return pd.Series(
            selected["value"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(selected["ts"])),
            name=series_id,
        )

    def describe(self) -> dict[str, Any]:
        """The view fields a result record copies (§3.2, §3.12).

        The cutoff alone does not say why it is what it is, and the two modes
        disagree by construction, so the basis, the declared clock, the source
        lags and the mode's own assumption travel with it. A record that
        carried the cutoff without the lag assumptions could not be checked.
        """
        manifest = self.dataset.manifest
        return {
            "mode": self.mode,
            "decision_date": self.decision_date,
            "cutoff": self.cutoff,
            "cutoff_basis": self.cutoff_basis,
            "vintage": self.vintage,
            "data_id": manifest.id,
            "decision_offset_minutes": self.decision_offset_minutes,
            "decision_time": self.decision_time,
            "lag_bars": {source: manifest.lag_bars(source) for source in sorted(manifest.sources)},
            "selection_rule": SELECTION_RULE,
            "availability_assumption": (
                SILENT_CORRECTION_ASSUMPTION if self.mode == "research_vintage" else REPLAY_INGESTION_STATEMENT
            ),
        }


__all__ = [
    "BarPanels",
    "CUTOFF_BASES",
    "CutoffBasis",
    "Dataset",
    "DatasetError",
    "DatasetView",
    "HISTORY_FLOOR",
    "MODES",
    "Manifest",
    "ManifestError",
    "Mode",
    "REFERENCE_DECISION_TIME",
    "REPLAY_INGESTION_STATEMENT",
    "SELECTION_RULE",
    "SILENT_CORRECTION_ASSUMPTION",
    "StageNotAvailableError",
    "TABLES",
    "TableSpec",
    "TerminalClaim",
    "VALUATION_STATUSES",
    "ValuationStatus",
    "parse_decision_time",
    "read_decided_at",
]
