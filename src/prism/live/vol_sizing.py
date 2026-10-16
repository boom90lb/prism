"""The paper book's volatility scale: the Barroso and Santa-Clara rule, live, off by default.

``prism.live.daily.run_daily_cycle(vol_scale=...)`` sizes the decile momentum
book by the constant-volatility rule of :mod:`prism.portfolio.vol_scale`. At a
refresh at close *t* the book's scale is ``min(max_scale, target_vol /
sigma)``, where sigma is the annualized standard deviation of the last
``lookback_bars`` daily returns of the unscaled *base book*; the account target
is ``scale`` times the unscaled book, capped at the account's gross ceiling, and
fills from the next open (N1, N2). Between refreshes the book holds its filled
weights, so the scale is held until the next refresh.

**The base book.** The paper account's own returns are not the base book: they
are scaled, rounded to whole shares, banded and charged costs. The base book is
rebuilt at every refresh from the panel the cycle already holds:

* the construct the cycle trades (``prism.assembly.heuristic_book_rule``, the
  decile book at raw gross 1.0) applied to the signal's scores at each past
  refresh date of the book's schedule, each score taken on the panel truncated
  at that date, so nothing after the date enters it (N1);
* held at constant weights from one refresh date to the next, close to close,
  gross of costs (:func:`prism.portfolio.vol_scale.base_book_returns`).

The schedule is the book's own. Under a seeded phase it is the panel bars whose
session-grid position is the book's offset modulo the cadence
(``prism.io.sessions.grid_position``). Under the first-session rule it is every
``decision_every``-th panel bar counting back from the refresh being decided,
which is the schedule that refresh puts the book on.

Two simplifications:

* the unrankable (spin-off) mask applies to the decision being made, not to the
  base book's past refresh dates, because the mask provider is consulted for
  the decision bar only. A corporate action the vendor's bars do not adjust for
  therefore enters sigma as a return of the name that carries it. A jump of
  that kind almost always raises the estimate, and so lowers the scale, but a
  jump that offsets an opposite move on the same bar can lower it;
* the base book carries none of the paper book's rounding, band or costs; it
  is the book the rule is defined on, as in the registration.

**History.** Scoring the earliest refresh date the window reaches needs the
signal's own ``required_history`` bars before it, so the panel must hold
:func:`required_panel_bars` bars (399 for 12-1 momentum at a 126-session window
and a 21-session cadence). A shorter panel is refused before any state moves
(N7); the paper loop's default fetch of three years holds about 750.

**The failure path.** :func:`prism.portfolio.vol_scale.decide_scale` falls back
to ``min(1, last good scale)``, or 1 with none, when its window cannot be
trusted: a panel whose last bar is not the decision bar, fewer than
``lookback_bars`` finite returns, more than 5% of them exactly zero (a stale or
flat panel), or a degenerate estimate. The last good scale is read from the
refresh rows of the targets ledger, each of which carries the scale decision it
was persisted with (``vol_scale``), so only decisions that reached the
write-ahead save can bound a later one: a decision the order guard or the
authorization gate refused never becomes a bound. The search stops at the
newest refresh row that carries no decision, because before it the book was
not scaled.

**The record.** Every refresh appends one row to ``{run-dir}/sizing.jsonl``
before the write-ahead step (:func:`append_sizing_decision`): the rule, the
decision and its reason, the base book's refresh dates, the unscaled and target
gross, the long side over equity and the cash-sleeve fraction ``max(0, 1 - 0.02
- L/E)``. A same-bar re-run that decides again (a crash before the write-ahead
save) decides on the same recorded panel and finds an identical row, which it
leaves alone; a row that disagrees raises, because one bar cannot carry two
scale decisions.

**Tranches.** The tranched book (:mod:`prism.live.tranches`) decides a scale
per tranche at its refresh, on the tranched base book of
:func:`tranched_base_book_window`, and bounds a failure by the tranche's own
last good scale, persisted in its component state. Its sizing row has
``schedule = "tranches"`` and carries the due tranches' decisions under
``tranches``, which is part of the decision a re-run must reproduce.

Production-import-path safe (N8): numpy and pandas only.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.assembly import heuristic_book_rule
from prism.io import sessions
from prism.live import ledgers
from prism.portfolio.vol_scale import (
    SLEEVE_BUFFER,
    ScaleDecision,
    VolScaleRule,
    base_book_returns,
    cap_gross,
    decide_scale,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism.live.daily import DailyBookConfig
    from prism.signal.base import Signal

logger = logging.getLogger(__name__)

#: The per-refresh scale record, beside the other ledgers in the run directory.
SIZING_LEDGER_FILENAME = "sizing.jsonl"

#: The key a refresh row of the targets ledger carries its scale decision under.
TARGETS_LEDGER_KEY = "vol_scale"

#: The one book the rule sizes. The decile book is built at raw gross 1.0, so
#: its scale is its gross; the inverse-vol trend book is already scaled to
#: ``max_gross`` and the directional book's gross is its scores', so a scale on
#: either would multiply a gross that is not the rule's unit.
SCALED_BOOK = "decile_neutral"

#: The fields of a sizing row that make up the decision. A same-bar re-run must
#: reproduce them exactly; the rest of the row describes the target the decision
#: was applied to.
DECISION_FIELDS: tuple[str, ...] = (
    "decision_bar",
    "rule",
    "scale",
    "sigma",
    "status",
    "reason",
    "n_valid",
    "zero_fraction",
    "last_good",
    "schedule",
    "base_refresh_bars",
    "tranches",
)


class VolSizingError(ValueError):
    """The live volatility scale cannot run on these inputs (N7)."""


class SizingLedgerConflict(RuntimeError):
    """The sizing ledger already holds a different scale decision for this bar (N7)."""


@dataclass(frozen=True)
class LiveScale:
    """One refresh's scale decision and the base book it was taken on.

    ``schedule`` is ``"grid"`` (the seeded phase) or ``"panel"`` (the
    first-session rule); ``base_refresh_bars`` are the refresh dates whose
    books were held over the window, oldest first; ``ceiling`` is the account
    gross ceiling the scaled book is capped at.
    """

    rule: VolScaleRule
    decision: ScaleDecision
    schedule: str
    base_refresh_bars: tuple[str, ...]
    ceiling: float

    @property
    def scale(self) -> float:
        return float(self.decision.scale)


def required_panel_bars(required_history: int, rule: VolScaleRule, decision_every: int) -> int:
    """The panel bars a refresh needs to rebuild the base book over a full window.

    The window's ``lookback_bars`` returns use the weights held at the closes
    ``t - lookback_bars`` through ``t - 1``. The weights at the oldest of those
    closes were set at the latest refresh date at or before it, at most
    ``decision_every - 1`` bars earlier, and scoring there needs
    ``required_history`` bars ending at that date.
    """
    return int(required_history) + int(rule.lookback_bars) + max(1, int(decision_every)) - 1


def require_vol_scale_inputs(
    rule: VolScaleRule,
    *,
    config: "DailyBookConfig",
    signal: "Signal",
    close: pd.DataFrame,
    regime_gross_scale: object | None,
    targets_ledger: Path | None,
) -> None:
    """Refuse a volatility scale the cycle cannot apply correctly, before any state moves (N7)."""
    if not isinstance(rule, VolScaleRule):
        raise VolSizingError(f"vol_scale must be a VolScaleRule, got {type(rule).__name__}")
    if config.book != SCALED_BOOK:
        raise VolSizingError(
            f"the volatility scale sizes the {SCALED_BOOK!r} book only, whose raw gross is 1.0; book "
            f"{config.book!r} has a different gross unit and a scale would multiply it"
        )
    if regime_gross_scale is not None:
        raise VolSizingError(
            "the volatility scale and the regime gross-scale hook both size the book's gross; a regime hook is "
            "refused while the volatility scale is on"
        )
    if not config.max_gross >= 1.0:
        raise VolSizingError(
            f"the volatility scale needs max_gross >= 1.0 so the unscaled decile book keeps its raw gross 1.0; "
            f"got {config.max_gross}"
        )
    declared = getattr(signal, "fitted_state", None)
    if declared != "none":
        raise VolSizingError(
            f"the base book re-scores the signal on the panel truncated at each past refresh date, which is "
            f"point in time only for a signal with no fitted state; {type(signal).__name__} declares "
            f"fitted_state={declared!r}, not 'none'"
        )
    if targets_ledger is None:
        raise VolSizingError(
            "the volatility scale needs the targets ledger: each refresh row carries the scale it was persisted "
            "with, and the failure path is bounded by the last good one"
        )
    needed = required_panel_bars(signal.required_history, rule, config.decision_every)
    if len(close) < needed:
        raise VolSizingError(
            f"the panel has {len(close)} bars and the volatility scale needs {needed} (the signal's "
            f"{signal.required_history} bars of history before the oldest refresh date a "
            f"{rule.lookback_bars}-return window reaches at a {config.decision_every}-session cadence); fetch "
            "more history rather than estimate the volatility on a short window"
        )


def base_book_window(
    signal: "Signal",
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    config: "DailyBookConfig",
    *,
    max_gross: float,
    lookback_bars: int,
    phase_offset: int | None,
) -> tuple[np.ndarray, tuple[str, ...], str]:
    """The base book's last ``lookback_bars`` returns through the panel's last bar.

    Returns ``(returns, refresh_bars, schedule)``. A window bar with no
    scoreable refresh date at or before it holds no book, and its return is
    NaN, which :func:`prism.portfolio.vol_scale.decide_scale` reads as a short
    window.
    """
    n = len(close.index)
    last = n - 1
    first = last - int(lookback_bars)
    if first < 0:
        raise VolSizingError(f"a {lookback_bars}-return window needs {lookback_bars + 1} bars, the panel has {n}")
    every = max(1, int(config.decision_every))
    # Refresh dates are looked for one full cycle before the window's first close,
    # which is as far back as the oldest held book can have been set.
    low = max(0, first - every + 1)
    candidates = np.arange(low, last + 1)
    schedule = "panel" if phase_offset is None else "grid"
    if phase_offset is None:
        refreshes = candidates[(last - candidates) % every == 0]
    else:
        positions = np.array([_grid_position(close.index[int(p)]) for p in candidates], dtype=np.int64)
        refreshes = candidates[(positions - int(phase_offset)) % every == 0]

    books: dict[int, np.ndarray | None] = {}
    held = np.full((lookback_bars + 1, close.shape[1]), np.nan)
    used: list[int] = []
    # The weights held at the last close enter no return through it, so the
    # window's books are the ones held at closes first .. last - 1.
    for row, bar in enumerate(range(first, last)):
        at = int(np.searchsorted(refreshes, bar, side="right")) - 1
        if at < 0:
            continue
        refresh = int(refreshes[at])
        if refresh not in books:
            books[refresh] = _book_at(signal, close, volume, config, max_gross=max_gross, position=refresh)
        book = books[refresh]
        if book is None:
            continue
        if refresh not in used:
            used.append(refresh)
        held[row] = book
    window = close.iloc[first : last + 1]
    weights = pd.DataFrame(held, index=window.index, columns=close.columns)
    returns = base_book_returns(window, weights).to_numpy(dtype=float)[1:]
    return returns, tuple(_iso(close.index[p]) for p in used), schedule


def tranched_base_book_window(
    signal: "Signal",
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    config: "DailyBookConfig",
    *,
    max_gross: float,
    lookback_bars: int,
    offsets: Sequence[int],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The tranched base book's last ``lookback_bars`` returns through the panel's last bar.

    The tranched book of :mod:`prism.live.tranches` holds ``len(offsets)``
    tranches; tranche *j* refreshes on the panel bars whose session-grid
    position is ``offsets[j]`` modulo ``config.decision_every``. Its base book at
    a close is the plain mean of every tranche's unscaled book as last refreshed
    at or before that close, ``sum_j D_j / K`` at raw gross 1.0, which is the
    book the volatility scale is defined on whatever share of the tranches the
    paper account has built. Each tranche book is the cycle's own construct
    applied to the signal scored on the panel truncated at its refresh date (N1),
    exactly as :func:`base_book_window` builds the single book.

    A window bar at which some tranche has no scoreable refresh date at or
    before it holds no base book, and its return is NaN, which
    :func:`prism.portfolio.vol_scale.decide_scale` reads as a short window.
    Returns ``(returns, refresh_bars)``: the refresh dates whose books entered
    the window, oldest first.
    """
    if not offsets:
        raise VolSizingError("a tranched base book needs at least one tranche offset")
    n = len(close.index)
    last = n - 1
    first = last - int(lookback_bars)
    if first < 0:
        raise VolSizingError(f"a {lookback_bars}-return window needs {lookback_bars + 1} bars, the panel has {n}")
    every = max(1, int(config.decision_every))
    low = max(0, first - every + 1)
    candidates = np.arange(low, last + 1)
    positions = np.array([_grid_position(close.index[int(p)]) for p in candidates], dtype=np.int64)
    schedules = [candidates[(positions - int(offset)) % every == 0] for offset in offsets]

    books: dict[int, np.ndarray | None] = {}
    held = np.full((lookback_bars + 1, close.shape[1]), np.nan)
    used: set[int] = set()
    count = float(len(offsets))
    for row, bar in enumerate(range(first, last)):
        dates: list[int] = []
        for refreshes in schedules:
            at = int(np.searchsorted(refreshes, bar, side="right")) - 1
            if at < 0:
                break
            dates.append(int(refreshes[at]))
        if len(dates) < len(schedules):
            continue
        vectors: list[np.ndarray] = []
        for refresh in dates:
            if refresh not in books:
                books[refresh] = _book_at(signal, close, volume, config, max_gross=max_gross, position=refresh)
            book = books[refresh]
            if book is None:
                break
            vectors.append(book)
        if len(vectors) < len(dates):
            continue
        total = vectors[0].copy()
        for vector in vectors[1:]:
            total = total + vector
        held[row] = total / count
        used.update(dates)
    window = close.iloc[first : last + 1]
    weights = pd.DataFrame(held, index=window.index, columns=close.columns)
    returns = base_book_returns(window, weights).to_numpy(dtype=float)[1:]
    return returns, tuple(_iso(close.index[p]) for p in sorted(used))


def decide_live_scale(
    signal: "Signal",
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    config: "DailyBookConfig",
    rule: VolScaleRule,
    *,
    decision_bar: str,
    phase_offset: int | None,
    last_good: float | None,
) -> LiveScale:
    """The scale a refresh at ``decision_bar`` sets, from the base book rebuilt on ``close``."""
    returns, refresh_bars, schedule = base_book_window(
        signal,
        close,
        volume,
        config,
        max_gross=float(config.max_gross),
        lookback_bars=rule.lookback_bars,
        phase_offset=phase_offset,
    )
    decision = decide_scale(
        returns,
        rule,
        last_good=last_good,
        decision_bar=decision_bar,
        panel_last_bar=close.index[-1],
    )
    return LiveScale(
        rule=rule,
        decision=decision,
        schedule=schedule,
        base_refresh_bars=refresh_bars,
        ceiling=min(float(config.max_gross), float(rule.max_scale if rule.max_gross is None else rule.max_gross)),
    )


def scaled_book_rule(
    config: "DailyBookConfig",
    close: pd.DataFrame,
    max_gross: float,
    scale: LiveScale,
) -> tuple[Callable[[pd.Series], pd.Series], dict[str, float]]:
    """The cycle's construct times the decided scale, capped at the account ceiling.

    Returns the rule composition applies in place of
    :func:`prism.assembly.heuristic_book_rule` and a dict it fills when it runs:
    the unscaled book's gross and the ceiling factor :func:`cap_gross` applied.
    At a scale of 1.0 under a slack ceiling every weight is the unscaled
    construct's own float.
    """
    base = heuristic_book_rule(config, close, max_gross)
    applied: dict[str, float] = {}

    def rule(scores: pd.Series) -> pd.Series:
        weights = base(scores)
        applied["unscaled_gross"] = float(np.nansum(np.abs(weights.to_numpy(dtype=float))))
        target, factor = cap_gross(weights * scale.scale, scale.ceiling)
        applied["ceiling_factor"] = float(factor)
        return target

    return rule, applied


def last_good_scale(targets_ledger: Path | None, before_bar: str) -> float | None:
    """The scale of the newest persisted ``ok`` decision before ``before_bar``, or None.

    Read from the refresh rows of the targets ledger, newest first. Fallback and
    warm-up decisions are passed over; a refresh row with no decision ends the
    search, because the book was not scaled at that refresh.
    """
    if targets_ledger is None:
        return None
    for row in reversed(ledgers.read_rows(Path(targets_ledger))):
        refresh_bar = str(row.get("refresh_bar", ""))
        if not refresh_bar or refresh_bar >= str(before_bar):
            continue
        record = row.get(TARGETS_LEDGER_KEY)
        if not isinstance(record, Mapping):
            return None
        if record.get("status") != "ok":
            continue
        scale = record.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
            raise VolSizingError(f"targets ledger row {refresh_bar} carries a scale of {scale!r} (N7)")
        if float(scale) <= 0.0:
            raise VolSizingError(f"targets ledger row {refresh_bar} carries a non-positive scale {scale!r} (N7)")
        return float(scale)
    return None


def targets_ledger_record(scale: LiveScale) -> dict[str, Any]:
    """What the refresh row of the targets ledger carries under ``vol_scale``."""
    decision = scale.decision
    return {
        "scale": _plain(decision.scale),
        "status": decision.status,
        "reason": decision.reason,
        "sigma": _plain(decision.sigma),
    }


def sizing_row(scale: LiveScale, targets: pd.Series, applied: Mapping[str, float]) -> dict[str, Any]:
    """The sizing ledger row for one refresh: the decision, then the target it sized."""
    values = pd.to_numeric(pd.Series(targets, dtype=float), errors="coerce").to_numpy(dtype=float)
    finite = np.where(np.isfinite(values), values, 0.0)
    long_side = float(np.clip(finite, 0.0, None).sum())
    row: dict[str, Any] = {
        **{key: _plain(value) for key, value in scale.decision.to_dict().items()},
        "rule": scale.rule.to_dict(),
        "schedule": scale.schedule,
        "base_refresh_bars": list(scale.base_refresh_bars),
        "ceiling": scale.ceiling,
        "unscaled_gross": _plain(applied.get("unscaled_gross", math.nan)),
        "ceiling_factor": _plain(applied.get("ceiling_factor", math.nan)),
        "target_gross": float(np.abs(finite).sum()),
        "long_over_equity": long_side,
        "sleeve_fraction": max(0.0, 1.0 - SLEEVE_BUFFER - long_side),
        "sleeve_buffer": SLEEVE_BUFFER,
    }
    return row


def append_sizing_decision(path: Path, row: Mapping[str, Any]) -> bool:
    """Append ``row`` unless the ledger already records this bar's decision.

    A same-bar re-run that decides again finds its own row and writes nothing
    (returns False). A recorded row whose decision differs raises
    :class:`SizingLedgerConflict` before anything is persisted or submitted.
    """
    bar = str(row["decision_bar"])
    for recorded in ledgers.read_rows(Path(path)):
        if str(recorded.get("decision_bar")) != bar:
            continue
        if _decision_of(recorded) != _decision_of(row):
            raise SizingLedgerConflict(
                f"{path} already records a different scale decision for {bar}: recorded "
                f"{_decision_of(recorded)}, decided now {_decision_of(row)}. One bar cannot carry two scale "
                "decisions; the panel log pins the bars, so a difference means the rule or the targets ledger "
                "changed between the two runs (N7)"
            )
        return False
    return ledgers.append_monotone(Path(path), "decision_bar", dict(row))


def read_sizing_ledger(path: Path) -> pd.DataFrame:
    """The sizing ledger as a frame, one row per refresh (empty when absent)."""
    return ledgers.read_frame(Path(path))


def row_decisions(row: Mapping[str, Any]) -> list[tuple[str, str | None]]:
    """The ``(status, reason)`` of every scale decision a sizing row records.

    A single-book row records one decision at its top level; a tranched row
    (``schedule = "tranches"``) records one per due tranche under ``tranches``.
    Read-only, for ``prism-doctor``'s fallback-streak check.
    """
    tranches = row.get("tranches")
    if isinstance(tranches, Mapping):
        return [
            (str(entry.get("status")), None if entry.get("reason") is None else str(entry.get("reason")))
            for entry in tranches.values()
            if isinstance(entry, Mapping)
        ]
    return [(str(row.get("status")), None if row.get("reason") is None else str(row.get("reason")))]


def _book_at(
    signal: "Signal",
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    config: "DailyBookConfig",
    *,
    max_gross: float,
    position: int,
) -> np.ndarray | None:
    """The unscaled book decided at the close of panel bar ``position``, from bars through it only."""
    if position + 1 < int(signal.required_history):
        return None
    panel = close.iloc[: position + 1]
    scores = signal.score(panel, None if volume is None else volume.iloc[: position + 1])
    weights = heuristic_book_rule(config, panel, max_gross)(scores)
    return weights.reindex(close.columns).to_numpy(dtype=float)


def _grid_position(stamp: Any) -> int:
    day = pd.Timestamp(stamp).date()
    try:
        return sessions.grid_position(day)
    except ValueError as exc:
        raise VolSizingError(
            f"panel bar {day.isoformat()} has no session-grid position, so the seeded base book cannot place its "
            f"refresh dates: {exc} (N7)"
        ) from exc


def _iso(stamp: Any) -> str:
    return pd.Timestamp(stamp).date().isoformat()


def _plain(value: Any) -> Any:
    """A JSON-safe scalar: NaN and infinities become None, numpy scalars plain Python."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _decision_of(row: Mapping[str, Any]) -> str:
    return json.dumps({key: row.get(key) for key in DECISION_FIELDS}, sort_keys=True)
