"""Purged + embargoed walk-forward cross-validation for time-series.

Implements López de Prado (AFML §7.4) walk-forward with two distinct
leakage controls:

  * **purge_horizon** — drop training samples whose label window overlaps
    the test fold. With a forward-return target `y_t = (p_{t+h} - p_t)/p_t`,
    every training label for `t ∈ [test_start - h, test_start - 1]` peeks
    into the test region; those rows must be removed from train.

  * **embargo_pct** — additional buffer (as fraction of total bars) appended
    after each test fold and excluded from subsequent training. Targets
    serial correlation in model residuals across adjacent folds, not label
    overlap (purge already handles that).

The splitter yields integer-position index arrays compatible with
`pd.DataFrame.iloc`. It does not assume the input has a monotonic
DatetimeIndex (the embargo is in bar count, not wall-clock).

Two more things live here, added at engine stage 2 (spec §3.8, §3.13):

  * :class:`RollingWalkForward` — the rolling *formation / test / step*
    geometry the recorded evidence actually used. ``PurgedWalkForward``
    partitions a sample into ``n_splits + 1`` equal chunks and cannot express
    a fixed 312-bar formation window rolled forward 63 bars at a time, so it
    cannot reproduce ``results/demotion_b1/folds.json`` (21 folds) or
    ``results/trend_t0/folds.json`` (74 folds) at all. This class reproduces
    both, date for date.

  * :class:`FoldTable` — a *frozen* table of train and test date ranges,
    content-hashed and recorded as ``fold_geometry`` in every result. The
    registration declares a table, never splitter keyword arguments; the
    splitter is one way to *generate* a table, and
    :func:`resolve_fold_geometry` refuses a call that supplies both, so a
    parameter set can never quietly stand in for a recorded geometry.

Why the geometry is what it is. The rolling loop is a duplication (not an
import — ``prism`` never imports ``research``) of
``research/arbitrage/walk_forward.py::iter_walk_forward_slices``, the only
implementation that produced the recorded folds. Three details there are
load-bearing and are reproduced exactly:

  * the last usable test bar is ``n_obs - 1``, not ``n_obs``: one bar is
    reserved past every fold so the fold-end flatten can fill at the *next*
    open (SPEC: next-open fills, backtest and live alike);
  * a fold is emitted only while ``test_start + min_test_bars <= n_obs - 1``,
    so a stub tail shorter than ``min_test_bars`` is dropped rather than
    scored;
  * the final fold's test window is truncated at that reserved bar, which is
    why the momentum table's 21st fold holds 48 test bars and not 63.

Getting any one of the three wrong changes the fold count, and a fold count is
a denominator: it moves every per-fold statistic downstream.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, Sized, Tuple

import numpy as np
import pandas as pd

from prism.validation.trials import stable_json_dumps

#: Meaning of the four dates in a :class:`FoldBoundary`: every one of them is a
#: session on which the strategy is live, so a window ``[start, end]`` includes
#: both ends. The recorded ``folds.json`` files use this convention (their
#: ``formation_end`` is the last formation bar, not the first test bar), and it
#: is written into every table so a future convention cannot be read as this one.
INCLUSIVE_SESSION_DATES = "inclusive_session_dates"

#: Default minimum test-window length, in bars. A shorter tail is dropped rather
#: than scored: ``results/demotion_b1/config.json`` and
#: ``results/trend_t0/config.json`` both record ``walk.min_test_bars = 20``.
DEFAULT_MIN_TEST_BARS = 20


@dataclass
class PurgedWalkForward:
    """Walk-forward splitter with purge and embargo.

    Parameters
    ----------
    n_splits : int
        Number of test folds. The series is partitioned into ``n_splits + 1``
        equal chunks; chunk 0 is reserved as the initial training pool and
        chunks ``1..n_splits`` each become one test fold.
    purge_horizon : int
        Number of bars whose forward-looking labels overlap a test fold.
        Set to the label horizon `h` (e.g. 5 for a 5-day forward return).
        Training rows whose label window touches the test region are dropped.
    embargo_pct : float, default 0.01
        Fraction of total bars to skip *after* each test fold. The embargoed
        region is excluded from any subsequent fold's training set. Set to
        ``0.0`` to disable.
    expanding : bool, default True
        If True, each fold's train set starts at index 0 (expanding window).
        If False, train is a rolling window of length ``test_size * 2``
        ending immediately before the purge zone.
    strict : bool, default False
        If True, a fold whose training set is emptied by purge/embargo
        raises ``ValueError`` instead of being skipped. Use on any path
        where the fold count is a claim denominator.

    Notes
    -----
    Fold counts are conserved quantities: a skipped fold silently changes
    every per-fold denominator downstream. Skips therefore always emit a
    ``RuntimeWarning``, and after the iterator returned by :meth:`split` is
    fully consumed, ``n_yielded_`` and ``n_skipped_`` hold the realized
    counts for downstream assertions.
    """

    n_splits: int
    purge_horizon: int
    embargo_pct: float = 0.01
    expanding: bool = True
    strict: bool = False

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {self.n_splits}")
        if self.purge_horizon < 0:
            raise ValueError(f"purge_horizon must be >= 0, got {self.purge_horizon}")
        if not 0.0 <= self.embargo_pct < 1.0:
            raise ValueError(f"embargo_pct must be in [0, 1), got {self.embargo_pct}")
        self.n_yielded_: int | None = None
        self.n_skipped_: int | None = None

    def _skip_fold(self, k: int, reason: str) -> None:
        """Record a skipped fold: raise in strict mode, warn otherwise."""
        msg = (
            f"PurgedWalkForward: fold {k} of {self.n_splits} skipped "
            f"({reason}). The realized fold count no longer matches the "
            f"requested count; every per-fold denominator downstream moves."
        )
        if self.strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        self.n_skipped_ = (self.n_skipped_ or 0) + 1

    def get_n_splits(self, X: Sized | None = None) -> int:
        return self.n_splits

    def split(self, X: Sized) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` integer arrays for each fold."""
        n = len(X)
        if n < (self.n_splits + 1) * 2:
            raise ValueError(
                f"Need at least {(self.n_splits + 1) * 2} samples for " f"n_splits={self.n_splits}; got {n}"
            )
        embargo = int(self.embargo_pct * n)
        test_size = n // (self.n_splits + 1)

        self.n_yielded_ = 0
        self.n_skipped_ = 0
        prior_embargo_zones: list[tuple[int, int]] = []

        for k in range(self.n_splits):
            test_start = (k + 1) * test_size
            if k < self.n_splits - 1:
                test_end = test_start + test_size
            else:
                test_end = n
            train_end = test_start - self.purge_horizon
            if train_end <= 0:
                self._skip_fold(
                    k,
                    f"purge_horizon={self.purge_horizon} consumes the "
                    f"entire training pool before test_start={test_start}",
                )
                prior_embargo_zones.append((test_end, min(test_end + embargo, n)))
                continue

            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - test_size * 2)
            train_mask = np.zeros(n, dtype=bool)
            train_mask[train_start:train_end] = True

            for emb_start, emb_end in prior_embargo_zones:
                train_mask[emb_start:emb_end] = False

            train_idx = np.flatnonzero(train_mask)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) == 0:
                self._skip_fold(k, "training set emptied by prior embargo zones")
                prior_embargo_zones.append((test_end, min(test_end + embargo, n)))
                continue

            self.n_yielded_ += 1
            yield train_idx, test_idx
            prior_embargo_zones.append((test_end, min(test_end + embargo, n)))


# --------------------------------------------------------------------------
# Rolling formation / test / step geometry (engine spec §3.8, stage 2)
# --------------------------------------------------------------------------


class FoldGeometryError(ValueError):
    """A fold geometry is malformed, ambiguous, or does not match its record.

    Raised rather than repaired. A fold table is a denominator: silently
    dropping a fold, silently shortening one, or silently substituting a
    parameter set for a recorded table all change what every per-fold number
    downstream is an average *of*, without changing anything visible.
    """


@dataclass(frozen=True)
class RollingFold:
    """One fold as integer positions into the session index.

    ``formation``/``test`` are half-open in the Python sense
    (``stop`` excluded); the *dates* recorded in a :class:`FoldTable` are the
    inclusive endpoints, i.e. ``index[formation_stop - 1]`` and
    ``index[test_stop - 1]``.

    ``next_open_position`` is the bar whose open fills the fold-end flatten. It
    always exists (the geometry reserves it), and it is deliberately *not* part
    of the test window: the flatten trades at that open, it is not scored as a
    decision on it.
    """

    fold: int
    formation_start: int
    formation_stop: int
    test_start: int
    test_stop: int
    next_open_position: int

    @property
    def formation_rows(self) -> int:
        return self.formation_stop - self.formation_start

    @property
    def test_rows(self) -> int:
        return self.test_stop - self.test_start

    @property
    def formation_positions(self) -> np.ndarray:
        return np.arange(self.formation_start, self.formation_stop)

    @property
    def test_positions(self) -> np.ndarray:
        return np.arange(self.test_start, self.test_stop)


@dataclass(frozen=True)
class RollingWalkForward:
    """Rolling formation / test / step walk-forward splitter.

    Fold ``k`` forms on bars ``[k * step, k * step + formation_bars)`` and
    tests on the ``test_bars`` bars immediately after, with no gap: the
    formation window ends on the bar before the first test bar, so a decision
    taken on the first test bar uses only closes at or before it. There is no
    purge and no embargo, because there is no forward-looking label — the
    formation window fits estimators, and the test window is traded forward.

    Parameters
    ----------
    formation_bars : int
        Length of the in-sample formation window, in bars. Every fold uses the
        same length; the window rolls, it does not expand.
    test_bars : int
        Length of the out-of-sample test window, in bars. The final fold's
        window is truncated when the sample ends (see the module docstring).
    step_bars : int or None, default None
        Bars between consecutive formation starts. ``None`` means "step by
        ``test_bars``", which tiles the test windows edge to edge with no gap
        and no overlap. Both recorded configurations record ``step_bars: null``.
    min_test_bars : int, default 20
        Shortest test window that may be emitted. A tail shorter than this is
        dropped, not scored.
    close_positions_at_fold_end : bool, default True
        Every fold is flattened before the next one may trade. ``False`` is
        refused: carry across folds is an accounting rule the recorded evidence
        never used and this splitter cannot express, and silently ignoring the
        flag would misstate the position path.

    Notes
    -----
    ``split`` yields ``(formation_idx, test_idx)`` position arrays, so the class
    is drop-in wherever ``PurgedWalkForward.split`` is consumed. Use
    :meth:`iter_folds` when the fold's identity and its reserved next-open bar
    matter, and :meth:`fold_table` to turn the geometry into a datestamped,
    content-hashed :class:`FoldTable`.
    """

    formation_bars: int
    test_bars: int
    step_bars: int | None = None
    min_test_bars: int = DEFAULT_MIN_TEST_BARS
    close_positions_at_fold_end: bool = True

    def __post_init__(self) -> None:
        if self.formation_bars < 1:
            raise ValueError(f"formation_bars must be >= 1, got {self.formation_bars}")
        if self.test_bars < 2:
            raise ValueError(f"test_bars must be >= 2, got {self.test_bars}")
        if not 1 <= self.min_test_bars <= self.test_bars:
            raise ValueError(f"min_test_bars must be in [1, test_bars={self.test_bars}], got {self.min_test_bars}")
        if self.step_bars is not None and self.step_bars < self.test_bars:
            raise ValueError(
                f"step_bars must be >= test_bars={self.test_bars} so test windows do not overlap, "
                f"got {self.step_bars}"
            )
        if not self.close_positions_at_fold_end:
            raise ValueError(
                "close_positions_at_fold_end=False is refused: carry across folds is not implemented "
                "and no recorded configuration used it"
            )

    @property
    def effective_step_bars(self) -> int:
        """Bars between consecutive formation starts, with ``None`` resolved to ``test_bars``."""
        return self.test_bars if self.step_bars is None else self.step_bars

    def min_obs(self) -> int:
        """Fewest bars from which at least one fold can be cut."""
        return self.formation_bars + self.min_test_bars + 1

    def iter_folds(self, n_obs: int) -> Iterator[RollingFold]:
        """Yield each fold's integer positions, in order.

        ``n_obs`` is the length of the *session index the strategy trades*, not
        of the raw download: dropping non-sessions before the split is what
        keeps the fold boundaries on real bars.
        """
        if n_obs < self.min_obs():
            raise ValueError(
                f"need at least formation_bars + min_test_bars + 1 = {self.min_obs()} bars, got {n_obs} "
                f"(formation_bars={self.formation_bars}, min_test_bars={self.min_test_bars}); the trailing "
                "bar is reserved for the fold-end flatten's next-open fill"
            )
        step = self.effective_step_bars
        last_test_end = n_obs - 1
        fold = 0
        formation_start = 0
        while True:
            formation_stop = formation_start + self.formation_bars
            test_start = formation_stop
            if test_start + self.min_test_bars > last_test_end:
                return
            test_stop = min(test_start + self.test_bars, last_test_end)
            yield RollingFold(
                fold=fold,
                formation_start=formation_start,
                formation_stop=formation_stop,
                test_start=test_start,
                test_stop=test_stop,
                next_open_position=test_stop,
            )
            fold += 1
            formation_start += step

    def get_n_splits(self, X: Sized | None = None) -> int:
        """Number of folds this geometry cuts from ``X``.

        Unlike :class:`PurgedWalkForward`, the count is a function of the
        sample length, so ``X`` is required: answering without it would be a
        guess dressed as a denominator.
        """
        if X is None:
            raise ValueError("RollingWalkForward.get_n_splits needs the sample: the fold count depends on its length")
        return sum(1 for _ in self.iter_folds(len(X)))

    def split(self, X: Sized) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(formation_idx, test_idx)`` integer arrays for each fold."""
        for fold in self.iter_folds(len(X)):
            yield fold.formation_positions, fold.test_positions

    def fold_table(
        self,
        index: Sequence[Any] | pd.Index,
        *,
        table_id: str,
        source: str | None = None,
        timezone: str | None = None,
    ) -> FoldTable:
        """Cut ``index`` into a datestamped, content-hashed :class:`FoldTable`.

        ``index`` must be the strictly increasing session index the strategy
        trades. The returned table carries the four inclusive boundary dates of
        every fold plus its row counts; the row counts are recorded but do not
        enter the content hash (the registration's TOML form declares dates
        only, and the same geometry must hash the same in both forms).
        """
        sessions = _as_session_index(index)
        boundaries = [
            FoldBoundary(
                index=fold.fold,
                formation_start=sessions[fold.formation_start],
                formation_end=sessions[fold.formation_stop - 1],
                test_start=sessions[fold.test_start],
                test_end=sessions[fold.test_stop - 1],
                formation_rows=fold.formation_rows,
                test_rows=fold.test_rows,
            )
            for fold in self.iter_folds(len(sessions))
        ]
        if timezone is None and sessions.tz is not None:
            timezone = str(sessions.tz)
        return FoldTable(table_id=table_id, folds=tuple(boundaries), source=source, timezone=timezone)


@dataclass(frozen=True)
class FoldBoundary:
    """One fold's train and test windows as inclusive session dates.

    ``formation_end`` is the last formation bar and ``test_start`` the first
    test bar; under a contiguous session index they are adjacent sessions.
    ``formation_rows``/``test_rows`` are the recorded bar counts where a source
    carries them and ``None`` where it does not (the TOML registration form
    declares dates only). They are checked, never hashed.
    """

    index: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    formation_rows: int | None = None
    test_rows: int | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise FoldGeometryError(f"fold index must be >= 0, got {self.index}")
        order = (
            ("formation_start", self.formation_start),
            ("formation_end", self.formation_end),
            ("test_start", self.test_start),
            ("test_end", self.test_end),
        )
        for name, value in order:
            if not isinstance(value, pd.Timestamp) or value is pd.NaT:
                raise FoldGeometryError(f"fold {self.index} {name} must be a pandas Timestamp, got {value!r}")
        if not self.formation_start <= self.formation_end:
            raise FoldGeometryError(
                f"fold {self.index}: formation_start {self.formation_start} is after formation_end "
                f"{self.formation_end}"
            )
        if not self.formation_end < self.test_start:
            raise FoldGeometryError(
                f"fold {self.index}: test_start {self.test_start} does not follow formation_end "
                f"{self.formation_end}; a test bar inside the formation window is leakage"
            )
        if not self.test_start <= self.test_end:
            raise FoldGeometryError(
                f"fold {self.index}: test_start {self.test_start} is after test_end {self.test_end}"
            )
        for name, rows in (("formation_rows", self.formation_rows), ("test_rows", self.test_rows)):
            if rows is not None and rows < 1:
                raise FoldGeometryError(f"fold {self.index} {name} must be >= 1, got {rows}")

    def identity(self) -> dict[str, Any]:
        """The part of the boundary that enters the content hash: index + four dates."""
        return {
            "index": int(self.index),
            "formation_start": _canonical_timestamp(self.formation_start),
            "formation_end": _canonical_timestamp(self.formation_end),
            "test_start": _canonical_timestamp(self.test_start),
            "test_end": _canonical_timestamp(self.test_end),
        }

    def to_record(self) -> dict[str, Any]:
        """A JSON-ready row: the identity plus the recorded row counts."""
        record = self.identity()
        record["formation_rows"] = None if self.formation_rows is None else int(self.formation_rows)
        record["test_rows"] = None if self.test_rows is None else int(self.test_rows)
        return record


@dataclass(frozen=True)
class FoldTable:
    """A frozen table of train and test date ranges (engine spec §3.8, §3.13).

    This is the authority on fold geometry. A registration declares a table; a
    result records :meth:`fold_geometry`, whose ``sha256`` is the table's
    content hash. Splitter keyword arguments are never an accepted substitute —
    see :func:`resolve_fold_geometry`.

    The content hash covers the ``convention`` and, per fold, the index and the
    four boundary dates canonicalized to UTC instants. It therefore:

      * is stable across runs, processes, and machines (no dict ordering, no
        float, no local timezone);
      * is identical for the same geometry expressed in different but
        equivalent forms (a ``-05:00`` offset and ``America/New_York``, the
        imported JSON and the round-tripped TOML);
      * changes if any boundary moves by a single session.

    ``table_id`` and ``source`` are provenance and stay out of the hash: renaming
    a table must not revoke the identity of the geometry it holds, and — the
    point of the stage 2 reproduction — a table generated by
    :meth:`RollingWalkForward.fold_table` must hash *equal* to the one imported
    from the recorded ``folds.json`` when the geometry reproduces.
    """

    table_id: str
    folds: tuple[FoldBoundary, ...]
    source: str | None = None
    timezone: str | None = None
    convention: str = INCLUSIVE_SESSION_DATES

    def __post_init__(self) -> None:
        if not str(self.table_id).strip():
            raise FoldGeometryError("a fold table needs a non-empty table_id")
        if not self.folds:
            raise FoldGeometryError(f"fold table {self.table_id!r} holds no folds")
        if self.convention != INCLUSIVE_SESSION_DATES:
            raise FoldGeometryError(
                f"fold table {self.table_id!r} declares convention {self.convention!r}; only "
                f"{INCLUSIVE_SESSION_DATES!r} is implemented"
            )
        aware = {b.formation_start.tzinfo is not None for b in self.folds}
        if len(aware) > 1:
            raise FoldGeometryError(
                f"fold table {self.table_id!r} mixes timezone-aware and naive boundaries; the two are not "
                "comparable and would hash as different geometries"
            )
        for position, boundary in enumerate(self.folds):
            if boundary.index != position:
                raise FoldGeometryError(
                    f"fold table {self.table_id!r} is not a contiguous 0..n-1 run: row {position} carries "
                    f"index {boundary.index}. A gap means a fold was dropped, which moves every per-fold "
                    "denominator"
                )
        for previous, current in zip(self.folds, self.folds[1:]):
            if current.formation_start <= previous.formation_start:
                raise FoldGeometryError(
                    f"fold table {self.table_id!r}: fold {current.index} formation_start "
                    f"{current.formation_start} does not advance past fold {previous.index}'s "
                    f"{previous.formation_start}"
                )
            if current.test_start <= previous.test_end:
                raise FoldGeometryError(
                    f"fold table {self.table_id!r}: fold {current.index} test window starts "
                    f"{current.test_start}, on or before fold {previous.index}'s test_end "
                    f"{previous.test_end}; overlapping test windows double-count bars"
                )

    def __len__(self) -> int:
        return len(self.folds)

    def __iter__(self) -> Iterator[FoldBoundary]:
        return iter(self.folds)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def first_decision(self) -> pd.Timestamp:
        """First decision date: the table's first ``test_start`` (spec §3.13)."""
        return self.folds[0].test_start

    @property
    def last_decision(self) -> pd.Timestamp:
        """Last decision date: the table's last ``test_end`` (spec §3.13)."""
        return self.folds[-1].test_end

    @property
    def content_hash(self) -> str:
        """``sha256`` hex digest of the canonical geometry. See the class docstring."""
        payload = {
            "convention": self.convention,
            "folds": [boundary.identity() for boundary in self.folds],
        }
        return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()

    def to_records(self) -> list[dict[str, Any]]:
        """JSON-ready rows, one per fold."""
        return [boundary.to_record() for boundary in self.folds]

    def fold_geometry(self) -> dict[str, Any]:
        """The ``evaluation.fold_geometry`` block of a ``ResultRecord`` (spec §3.12).

        ``{ table id, sha256, per-fold train and test ranges }``, plus the
        provenance a reader needs to find the table again.
        """
        return {
            "table_id": self.table_id,
            "sha256": self.content_hash,
            "convention": self.convention,
            "timezone": self.timezone,
            "source": self.source,
            "n_folds": self.n_folds,
            "first_decision": _canonical_timestamp(self.first_decision),
            "last_decision": _canonical_timestamp(self.last_decision),
            "folds": [
                {
                    "index": int(boundary.index),
                    "train": {
                        "start": _canonical_timestamp(boundary.formation_start),
                        "end": _canonical_timestamp(boundary.formation_end),
                    },
                    "test": {
                        "start": _canonical_timestamp(boundary.test_start),
                        "end": _canonical_timestamp(boundary.test_end),
                    },
                }
                for boundary in self.folds
            ],
        }

    def positions(self, index: Sequence[Any] | pd.Index) -> tuple[RollingFold, ...]:
        """Resolve the table against a session index, as integer positions.

        Every boundary date must be a session in ``index``; a date that is not
        raises rather than snapping to a neighbour, because a snapped boundary
        is a different fold wearing the recorded one's name. The reserved
        next-open bar must exist too — the fold-end flatten has to fill
        somewhere.
        """
        sessions = _as_session_index(index)
        lookup = {timestamp: position for position, timestamp in enumerate(sessions)}
        folds: list[RollingFold] = []
        for boundary in self.folds:
            located: dict[str, int] = {}
            for name in ("formation_start", "formation_end", "test_start", "test_end"):
                value = getattr(boundary, name)
                if value not in lookup:
                    raise FoldGeometryError(
                        f"fold table {self.table_id!r} fold {boundary.index}: {name} {value} is not a "
                        f"session in the supplied index ({sessions[0]} .. {sessions[-1]}, "
                        f"{len(sessions)} bars)"
                    )
                located[name] = lookup[value]
            test_stop = located["test_end"] + 1
            if test_stop >= len(sessions):
                raise FoldGeometryError(
                    f"fold table {self.table_id!r} fold {boundary.index}: no bar after test_end "
                    f"{boundary.test_end} for the fold-end flatten to fill at the next open"
                )
            folds.append(
                RollingFold(
                    fold=boundary.index,
                    formation_start=located["formation_start"],
                    formation_stop=located["formation_end"] + 1,
                    test_start=located["test_start"],
                    test_stop=test_stop,
                    next_open_position=test_stop,
                )
            )
        return tuple(folds)

    def differences(self, other: FoldTable) -> list[str]:
        """Every way ``other``'s geometry differs from this one, as readable lines.

        Empty means the two tables carry the same geometry — which is what the
        stage 2 reproduction has to show. Row counts are compared too, where
        both tables record them; they are diagnosis, not identity.
        """
        problems: list[str] = []
        if self.convention != other.convention:
            problems.append(f"convention: {self.convention!r} vs {other.convention!r}")
        if self.n_folds != other.n_folds:
            problems.append(f"n_folds: {self.n_folds} vs {other.n_folds}")
        for mine, theirs in zip(self.folds, other.folds):
            for name in ("formation_start", "formation_end", "test_start", "test_end"):
                left, right = getattr(mine, name), getattr(theirs, name)
                if _canonical_timestamp(left) != _canonical_timestamp(right):
                    problems.append(f"fold {mine.index} {name}: {left} vs {right}")
            for name in ("formation_rows", "test_rows"):
                left, right = getattr(mine, name), getattr(theirs, name)
                if left is not None and right is not None and left != right:
                    problems.append(f"fold {mine.index} {name}: {left} vs {right}")
        return problems

    def assert_matches(self, other: FoldTable) -> None:
        """Raise :class:`FoldGeometryError` naming every difference, or return."""
        problems = self.differences(other)
        if problems:
            raise FoldGeometryError(
                f"fold table {self.table_id!r} does not match {other.table_id!r}:\n  " + "\n  ".join(problems)
            )

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        table_id: str,
        source: str | None = None,
        timezone: str | None = None,
        convention: str = INCLUSIVE_SESSION_DATES,
    ) -> FoldTable:
        """Build a table from ``folds.json``-shaped rows.

        Accepts either ``index`` or the recorded drivers' ``fold`` as the fold
        number, and ignores every other key a recorded row carries (metrics,
        diagnostics): a fold table is boundaries, and importing a row's
        performance alongside them would invite comparing a reproduction
        against the record's own numbers by accident.
        """
        boundaries: list[FoldBoundary] = []
        for position, row in enumerate(records):
            if "index" in row:
                fold_index = int(row["index"])
            elif "fold" in row:
                fold_index = int(row["fold"])
            else:
                raise FoldGeometryError(f"fold row {position} carries neither 'index' nor 'fold'")
            missing = [k for k in ("formation_start", "formation_end", "test_start", "test_end") if k not in row]
            if missing:
                raise FoldGeometryError(f"fold row {position} (index {fold_index}) is missing {missing}")
            boundaries.append(
                FoldBoundary(
                    index=fold_index,
                    formation_start=_as_timestamp(row["formation_start"], timezone),
                    formation_end=_as_timestamp(row["formation_end"], timezone),
                    test_start=_as_timestamp(row["test_start"], timezone),
                    test_end=_as_timestamp(row["test_end"], timezone),
                    formation_rows=None if row.get("formation_rows") is None else int(row["formation_rows"]),
                    test_rows=None if row.get("test_rows") is None else int(row["test_rows"]),
                )
            )
        return cls(
            table_id=table_id,
            folds=tuple(boundaries),
            source=source,
            timezone=timezone,
            convention=convention,
        )

    @classmethod
    def from_json(
        cls,
        path: Path | str,
        *,
        table_id: str | None = None,
        source: str | None = None,
        timezone: str | None = None,
    ) -> FoldTable:
        """Import a recorded ``folds.json`` (a JSON list of fold rows).

        ``source`` defaults to the path as given, so the imported table says
        where it came from without the caller having to repeat it. The file is
        only ever read.
        """
        file_path = Path(path)
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("folds")
            if rows is None:
                raise FoldGeometryError(f"{file_path} is a JSON object with no 'folds' key")
        else:
            rows = payload
        if not isinstance(rows, list):
            raise FoldGeometryError(f"{file_path} does not hold a list of fold rows")
        return cls.from_records(
            rows,
            table_id=table_id or file_path.parent.name or file_path.stem,
            source=source if source is not None else str(path),
            timezone=timezone,
        )

    @classmethod
    def from_toml(cls, path: Path | str) -> FoldTable:
        """Read a declared fold table (the registration form, spec §3.13).

        The file carries ``id``, an optional ``source``, an optional
        ``timezone`` that bare dates are localized to, ``convention``, and one
        ``[[fold]]`` entry per fold.
        """
        file_path = Path(path)
        payload = tomllib.loads(file_path.read_text(encoding="utf-8"))
        table_id = payload.get("id")
        if not table_id:
            raise FoldGeometryError(f"{file_path} declares no fold table id")
        rows = payload.get("fold")
        if not rows:
            raise FoldGeometryError(f"{file_path} declares no [[fold]] entries")
        return cls.from_records(
            rows,
            table_id=str(table_id),
            source=payload.get("source"),
            timezone=payload.get("timezone"),
            convention=payload.get("convention", INCLUSIVE_SESSION_DATES),
        )

    def to_toml(self) -> str:
        """Render the registration form. Round-trips through :meth:`from_toml`.

        Boundaries are written as bare dates, so every one of them must fall on
        local midnight; a boundary carrying a time of day is refused rather than
        truncated, because truncating it would silently move the geometry.
        """
        if self.timezone is None and self.folds[0].formation_start.tzinfo is not None:
            raise FoldGeometryError(
                f"fold table {self.table_id!r} carries timezone-aware boundaries but declares no timezone; "
                "the TOML form writes bare dates and would lose the zone that says which sessions they are"
            )
        lines = [
            f"id = {json.dumps(self.table_id)}",
        ]
        if self.source is not None:
            lines.append(f"source = {json.dumps(self.source)}")
        if self.timezone is not None:
            lines.append(f"timezone = {json.dumps(self.timezone)}")
        lines.append(f"convention = {json.dumps(self.convention)}")
        for boundary in self.folds:
            lines.append("")
            lines.append("[[fold]]")
            lines.append(f"index = {boundary.index}")
            for name in ("formation_start", "formation_end", "test_start", "test_end"):
                value: pd.Timestamp = getattr(boundary, name)
                if (value.hour, value.minute, value.second, value.nanosecond) != (0, 0, 0, 0):
                    raise FoldGeometryError(
                        f"fold table {self.table_id!r} fold {boundary.index} {name} is {value}, not local "
                        "midnight; the TOML form writes bare dates and will not truncate a time of day"
                    )
                lines.append(f"{name} = {value.date().isoformat()}")
        return "\n".join(lines) + "\n"

    def write_toml(self, path: Path | str) -> Path:
        """Write the registration form to ``path`` and return it."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.to_toml(), encoding="utf-8")
        return file_path


def resolve_fold_geometry(
    *,
    fold_table: FoldTable | None = None,
    splitter: RollingWalkForward | None = None,
    index: Sequence[Any] | pd.Index | None = None,
    table_id: str | None = None,
    source: str | None = None,
) -> FoldTable:
    """Return the one fold table a run may use, or refuse.

    Exactly one of ``fold_table`` and ``splitter`` may be supplied. Supplying
    both is the failure the spec names outright: splitter keyword arguments are
    never silently substituted for a recorded table, and a caller that passes
    both has not said which one the run is authorized against. Supplying
    neither leaves the fold count undefined.

    A ``splitter`` also needs ``index`` (the session index it cuts) and
    ``table_id`` (what the generated table will be recorded as).
    """
    if fold_table is not None and splitter is not None:
        raise FoldGeometryError(
            "a run declares either a frozen fold table or a splitter, never both: splitter keyword "
            f"arguments ({splitter}) may not stand in for the recorded table {fold_table.table_id!r}"
        )
    if fold_table is not None:
        if index is not None:
            fold_table.positions(index)
        return fold_table
    if splitter is None:
        raise FoldGeometryError("no fold geometry declared: supply either a frozen fold table or a splitter")
    if index is None:
        raise FoldGeometryError("a splitter needs the session index it cuts")
    if not table_id:
        raise FoldGeometryError("a generated fold table needs a table_id to be recorded under")
    return splitter.fold_table(index, table_id=table_id, source=source)


def _canonical_timestamp(value: pd.Timestamp) -> str:
    """A boundary's hash token: the UTC instant, or the naive stamp as written.

    Converting to UTC is what makes ``2020-01-02T00:00:00-05:00`` and the same
    session expressed in ``America/New_York`` one geometry rather than two.
    """
    if value.tzinfo is not None:
        return value.tz_convert("UTC").isoformat()
    return value.isoformat()


def _as_timestamp(value: Any, timezone: str | None) -> pd.Timestamp:
    """Coerce a recorded boundary to a ``Timestamp`` expressed in ``timezone``.

    A declared ``timezone`` says how these boundaries are *written*, not which
    instants they are: a naive stamp is localized to it, an aware one (a
    recorded ``-05:00`` offset, say) is converted to it. Conversion leaves the
    instant alone, so declaring a timezone never changes the content hash — it
    only makes the bare-date TOML form expressible.
    """
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise FoldGeometryError(f"fold boundary {value!r} is not a timestamp")
    if timezone is None:
        return timestamp
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def _as_session_index(index: Sequence[Any] | pd.Index) -> pd.DatetimeIndex:
    """Validate a session index: datetimes, strictly increasing, no duplicates."""
    sessions = index if isinstance(index, pd.DatetimeIndex) else pd.DatetimeIndex(list(index))
    if len(sessions) == 0:
        raise FoldGeometryError("the session index is empty")
    if sessions.hasnans:
        raise FoldGeometryError("the session index carries NaT")
    if not sessions.is_monotonic_increasing or not sessions.is_unique:
        raise FoldGeometryError(
            "the session index must be strictly increasing and unique; fold boundaries cut by position "
            "are meaningless otherwise"
        )
    return sessions
