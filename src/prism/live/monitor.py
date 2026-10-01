"""Live monitor over the paper-loop ledgers: return-level and tracking statistics.

The daily loop appends one mark-to-market NAV per decision bar to an equity
ledger (``prism.live.loop.append_equity_ledger``). This module turns that
accruing ledger into two live reads, both registered in
``docs/live_monitor_design.md`` (2026-09-30):

* **Return monitor.** The paper book's mean daily net return against a hurdle,
  read through a time-uniform confidence sequence so the stream can be
  inspected after every cycle and acted on at any data-dependent time without
  inflating type-I error. Returns are formed only over *consecutive exchange
  sessions*: a gap in the ledger (the 2026-08-14 to 2026-09-25 outage entered
  the previous monitor as one "daily" return of six weeks) is reported, not
  averaged in. The default sequence is the empirical-Bernstein one of
  Waudby-Smith and Ramdas (2023), whose width adapts to the realised variance;
  the earlier normal-mixture sequence, whose width is set by the assumed support
  alone, stays available for comparison.

* **Tracking monitor.** The paper book's return minus the return of the book it
  was supposed to hold (the last refresh's targets marked on the same closes:
  the *shadow*). A return-level monitor cannot resolve a 0.4-Sharpe book in a
  year, because the half-width of even a fixed-sample interval on the mean daily
  return is about ``2 * 0.7% / sqrt(252) = 0.09%``, about two annualized Sharpe
  units. The paper-minus-shadow difference has a dispersion of a few basis
  points a day, so it can say within months whether the paper stream is
  evidence about the strategy at all.

Neither read gates an order. ``docs/forward_design.md`` and
``docs/momentum_design.md`` consume the return verdict under their own
registered wording; the tracking verdict applies to registrations whose first
made after 2026-09-30 (``docs/live_monitor_design.md`` §4).
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from prism.io import sessions
from prism.live.loop import read_equity_ledger, read_shadow_ledger
from prism.validation.anytime import (
    anytime_monitor_read,
    empirical_bernstein_confidence_sequence,
    empirical_bernstein_monitor_read,
)

logger = logging.getLogger(__name__)

# Daily NAV returns of a paper book never approach ±10%; a bound tighter than the
# generic 0.5 default in prism.validation.anytime yields a more informative
# interval while staying winsorisation-safe (coverage holds for any bound).
DEFAULT_NAV_RETURN_BOUND = 0.1

#: Support assumed for the daily paper-minus-shadow difference (20 bp). Its
#: realised dispersion is a few basis points; the bound sets the sequence's
#: width floor (about ``2 * bound * log(2/alpha) / (0.75 n)``), so a tighter bound
#: resolves sooner at the cost of clipping a day whose difference exceeds it.
#: Clipped days are counted in the read (``n_winsorised``).
TRACKING_DIFFERENCE_BOUND = 0.002
#: Mean daily difference beyond which the paper stream contradicts the shadow:
#: 2 bp a day is about 5% a year, far beyond unmodelled cash, dividend or
#: rounding effects.
TRACKING_DIFFERENCE_TOLERANCE = 0.0002
#: Sessions before the correlation and the "consistent" verdict are read.
TRACKING_MIN_SESSIONS = 42
#: Pearson correlation of paper and shadow daily returns below which the paper
#: book is not holding the decided book.
TRACKING_CORR_FLOOR = 0.90
#: Fraction of the baseline's gross that may be missing from the close panel
#: before the shadow return is refused.
SHADOW_MAX_MISSING_GROSS = 0.10

METHODS = ("empirical_bernstein", "normal_mixture")


# ---------------------------------------------------------------------------
# Return monitor
# ---------------------------------------------------------------------------


def bar_date(bar: Any) -> dt.date:
    """The calendar date of an ISO bar label (``2026-09-30`` or ``2026-09-30T...``)."""
    return dt.date.fromisoformat(str(bar)[:10])


def consecutive_session_returns(equity: pd.DataFrame) -> dict[str, Any]:
    """Per-session NAV returns over ledger rows whose bars are consecutive exchange sessions.

    A pair ``(prev, cur)`` of adjacent ledger rows counts only when ``prev`` is a
    session and ``cur`` is the next session after it
    (``prism.io.sessions.next_session``). Every other adjacent pair is a *gap*
    (an outage, a holiday row, or a row dated off the calendar) and is reported
    in ``skipped`` with both bars and the sessions between them, never folded
    into the return series as if it were one day. Returns ``returns`` (array),
    ``bars`` (the ``cur`` bar of each return), ``prev_bars``, ``n_gaps`` and
    ``skipped``. A bar outside the bundled session table raises
    (:class:`prism.io.sessions.SessionTableRangeError`): a monitor that guessed
    about a calendar it does not hold would be wrong exactly when unwatched.
    """
    if equity.empty or len(equity) < 2:
        return {"returns": np.array([], dtype=float), "bars": [], "prev_bars": [], "n_gaps": 0, "skipped": []}
    frame = equity.sort_values("decision_bar").reset_index(drop=True)
    bars = [str(b) for b in frame["decision_bar"]]
    values = frame["equity"].astype(float).to_numpy()
    returns: list[float] = []
    kept: list[str] = []
    prev_kept: list[str] = []
    skipped: list[dict[str, Any]] = []
    for i in range(1, len(bars)):
        prev, cur = bars[i - 1], bars[i]
        prev_date, cur_date = bar_date(prev), bar_date(cur)
        consecutive = sessions.is_session(prev_date) and sessions.next_session(prev_date) == cur_date
        if not consecutive:
            between = sessions.sessions_between(prev_date, cur_date)
            missing = [d.isoformat() for d in between if prev_date < d < cur_date]
            skipped.append({"prev_bar": prev, "bar": cur, "sessions_between": len(missing)})
            continue
        if values[i - 1] <= 0.0 or not math.isfinite(values[i - 1]) or not math.isfinite(values[i]):
            skipped.append(
                {"prev_bar": prev, "bar": cur, "sessions_between": 0, "reason": "non-positive or non-finite NAV"}
            )
            continue
        returns.append(float(values[i] / values[i - 1] - 1.0))
        kept.append(cur)
        prev_kept.append(prev)
    return {
        "returns": np.asarray(returns, dtype=float),
        "bars": kept,
        "prev_bars": prev_kept,
        "n_gaps": len(skipped),
        "skipped": skipped,
    }


def paper_monitor_read(
    equity_ledger_path: Path | str,
    *,
    hurdle: float = 0.0,
    alpha: float = 0.05,
    bound: float = DEFAULT_NAV_RETURN_BOUND,
    opt_horizon: int = 252,
    method: str = "empirical_bernstein",
    shadow_ledger_path: Path | str | None = None,
) -> dict:
    """Anytime-valid verdict on the paper book's mean daily net return vs ``hurdle``.

    Loads the equity ledger, forms the per-session return series over consecutive
    exchange sessions (:func:`consecutive_session_returns`) and reads it through
    the confidence sequence ``method`` names: ``"empirical_bernstein"`` (default;
    :func:`prism.validation.anytime.empirical_bernstein_monitor_read`) or
    ``"normal_mixture"`` (the pre-2026-09-30 read,
    :func:`prism.validation.anytime.anytime_monitor_read`, kept for comparison).
    Verdict semantics are the same for both: ``confirmed`` when the whole
    sequence is above ``hurdle``, ``refuted`` when entirely below, else
    ``inconclusive``.

    The dict carries ``n``, ``mean``, ``ci_lower``, ``ci_upper``, ``verdict``,
    ``hurdle``, ``alpha``, ``n_equity_points`` and ``latest_equity`` as before,
    plus ``method``, ``bound``, ``n_gaps`` and ``skipped_pairs``. With
    ``shadow_ledger_path`` the tracking read (:func:`shadow_tracking_read`) is
    attached under ``tracking``. With fewer than two equity points, or no
    consecutive pair, there is no return yet: ``inconclusive`` with ``n = 0``,
    accruing, not failing.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    df = read_equity_ledger(Path(equity_ledger_path))
    n_points = int(len(df))
    series = consecutive_session_returns(df) if n_points >= 2 else None
    returns = series["returns"] if series is not None else np.array([], dtype=float)
    if returns.size == 0:
        read: dict[str, Any] = {
            "n": 0,
            "mean": None,  # None, not NaN: the read is logged as JSON and compared across runs
            "ci_lower": None,
            "ci_upper": None,
            "hurdle": float(hurdle),
            "alpha": float(alpha),
            "edge_confirmed": False,
            "edge_refuted": False,
            "verdict": "inconclusive",
        }
    elif method == "normal_mixture":
        read = dict(anytime_monitor_read(returns, hurdle=hurdle, alpha=alpha, bound=bound, opt_horizon=opt_horizon))
    else:
        read = dict(empirical_bernstein_monitor_read(returns, hurdle=hurdle, alpha=alpha, bound=bound))
    read["method"] = method
    read["bound"] = float(bound)
    read["n_equity_points"] = n_points
    read["latest_equity"] = float(df["equity"].iloc[-1]) if n_points else float("nan")
    read["n_gaps"] = int(series["n_gaps"]) if series is not None else 0
    read["skipped_pairs"] = list(series["skipped"]) if series is not None else []
    if shadow_ledger_path is not None:
        read["tracking"] = shadow_tracking_read(equity_ledger_path, shadow_ledger_path, alpha=alpha)
    return read


# ---------------------------------------------------------------------------
# Shadow book and tracking monitor
# ---------------------------------------------------------------------------


def _close_row(close: pd.DataFrame, bar: str) -> pd.Series:
    """The close row dated ``bar`` (ISO date) in a wide panel, whatever its index timezone."""
    dates = [str(stamp.date()) if hasattr(stamp, "date") else str(stamp)[:10] for stamp in close.index]
    matches = [i for i, d in enumerate(dates) if d == bar]
    if len(matches) != 1:
        raise KeyError(f"close panel has {len(matches)} row(s) dated {bar}; need exactly one")
    return close.iloc[matches[0]]


def shadow_session_return(
    close: pd.DataFrame,
    baseline_targets: Mapping[str, float] | pd.Series,
    decision_bar: str,
    prev_bar: str,
) -> dict[str, Any]:
    """Return of the decided book held unchanged from ``prev_bar``'s close to ``decision_bar``'s close.

    ``sum_i w_i (close_i[decision_bar] / close_i[prev_bar] - 1)`` over the
    baseline's names present with finite positive closes on both bars. Names
    absent from the panel or unpriced on either bar are left out and counted
    (``n_names_missing``, ``gross_missing``); when more than
    :data:`SHADOW_MAX_MISSING_GROSS` of the baseline's gross is missing the
    shadow would describe a different book, so the call raises rather than
    returning a number (N7). The baseline is the ``targets`` mapping of a refresh
    row in ``targets.jsonl``; weights are fractions of equity, so the result is
    a NAV return comparable with the equity ledger's.
    """
    weights = pd.Series(dict(baseline_targets), dtype=float)
    weights = weights[np.isfinite(weights) & (weights != 0.0)]
    if weights.empty:
        raise ValueError("baseline targets hold no finite nonzero weight; no book to shadow")
    gross = float(weights.abs().sum())
    now = _close_row(close, decision_bar)
    before = _close_row(close, prev_bar)
    used: dict[str, float] = {}
    missing: list[str] = []
    for symbol, weight in weights.items():
        p1 = now.get(symbol)
        p0 = before.get(symbol)
        if p1 is None or p0 is None:
            missing.append(str(symbol))
            continue
        p1f, p0f = float(p1), float(p0)
        if not (math.isfinite(p1f) and math.isfinite(p0f) and p0f > 0.0 and p1f > 0.0):
            missing.append(str(symbol))
            continue
        used[str(symbol)] = float(weight) * (p1f / p0f - 1.0)
    gross_missing = float(weights.reindex(missing).abs().sum()) if missing else 0.0
    if gross_missing > SHADOW_MAX_MISSING_GROSS * gross:
        raise ValueError(
            f"shadow {prev_bar}->{decision_bar}: {len(missing)} of {len(weights)} baseline names are unpriced "
            f"({gross_missing:.4f} of {gross:.4f} gross, over the {SHADOW_MAX_MISSING_GROSS:.0%} limit): "
            f"{sorted(missing)[:8]} (N7)"
        )
    return {
        "shadow_return": float(sum(used.values())),
        "n_names_used": len(used),
        "n_names_missing": len(missing),
        "gross_missing": gross_missing,
        "gross_baseline": gross,
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else None


def shadow_tracking_read(
    equity_ledger_path: Path | str,
    shadow_ledger_path: Path | str,
    *,
    alpha: float = 0.05,
    bound: float = TRACKING_DIFFERENCE_BOUND,
    tolerance: float = TRACKING_DIFFERENCE_TOLERANCE,
    min_sessions: int = TRACKING_MIN_SESSIONS,
    corr_floor: float = TRACKING_CORR_FLOOR,
) -> dict[str, Any]:
    """How faithfully the paper stream reproduces the decided book, read off two ledgers.

    Joins the paper book's consecutive-session returns (equity ledger) with the
    shadow returns (``shadow.jsonl``, one row per session the daily cycle could
    shadow) by decision bar, requiring the same previous bar. *Transition*
    sessions, the session right after a refresh bar, are set aside: next-open
    fills mean the held book switched mid-session, so paper minus shadow there
    is the realised execution wedge of that refresh (reported per refresh under
    ``wedge_by_refresh``, keyed by the refresh bar, which is the transition
    session's previous bar; the row's own ``refresh_bar`` names the baseline, the
    book that was shadowed), not tracking error. Over the other sessions it reports
    the empirical-Bernstein confidence sequence on the mean daily difference
    ``paper - shadow`` (support ``bound``, winsorised and counted), the Pearson
    correlation of the two series, the least-squares slope of paper on shadow,
    and the verdict:

    * ``contradicted`` when the sequence lies entirely below ``-tolerance``, or
      when ``n >= min_sessions`` and the correlation is below ``corr_floor``;
    * ``consistent`` when the sequence lies within ``[-tolerance, +tolerance]``,
      ``n >= min_sessions`` and the correlation is at least ``corr_floor``;
    * ``inconclusive`` otherwise.

    The correlation conjunct is a fixed-threshold screen on the point Pearson
    correlation re-read every session; unlike the confidence sequence it carries
    no time-uniform error control (its single-look false-trip rates are stated
    in the registration), and a registration reads the verdict as of its read
    date.

    A sequence entirely *above* ``+tolerance`` is reported (``above_tolerance``)
    but is not a contradiction: a book that earns more than it was supposed to
    is equally not evidence about the strategy, and the registration treats it
    as a defect to investigate, not a verdict.
    """
    paper = consecutive_session_returns(read_equity_ledger(Path(equity_ledger_path)))
    paper_by_bar = {bar: (prev, r) for bar, prev, r in zip(paper["bars"], paper["prev_bars"], paper["returns"])}
    shadow_rows = read_shadow_ledger(Path(shadow_ledger_path))
    joined: list[dict[str, Any]] = []
    unmatched = 0
    if not shadow_rows.empty:
        for row in shadow_rows.sort_values("decision_bar").to_dict("records"):
            bar = str(row["decision_bar"])
            found = paper_by_bar.get(bar)
            if found is None or str(row.get("prev_bar")) != found[0]:
                unmatched += 1
                continue
            shadow_value = float(row["shadow_return"])
            if not math.isfinite(shadow_value):
                unmatched += 1
                continue
            joined.append(
                {
                    "decision_bar": bar,
                    "prev_bar": found[0],
                    "paper": float(found[1]),
                    "shadow": shadow_value,
                    "transition": bool(row.get("transition", False)),
                    "refresh_bar": row.get("refresh_bar"),
                }
            )
    steady = [j for j in joined if not j["transition"]]
    transitions = [j for j in joined if j["transition"]]
    p = np.array([j["paper"] for j in steady], dtype=float)
    s = np.array([j["shadow"] for j in steady], dtype=float)
    diff = p - s
    n = int(diff.size)

    read: dict[str, Any] = {
        "n": n,
        "n_transition": len(transitions),
        "n_unmatched": unmatched,
        "alpha": float(alpha),
        "bound": float(bound),
        "tolerance": float(tolerance),
        "min_sessions": int(min_sessions),
        "corr_floor": float(corr_floor),
        "mean_difference": float(diff.mean()) if n else None,
        "ci_lower": None,
        "ci_upper": None,
        "n_winsorised": int(np.count_nonzero(np.clip(diff, -bound, bound) != diff)) if n else 0,
        "correlation": _pearson(p, s),
        "slope": None,
        "above_tolerance": False,
        "wedge_by_refresh": {},
        "verdict": "inconclusive",
    }
    if n >= 1:
        lower, upper = empirical_bernstein_confidence_sequence(diff, alpha=alpha, bound=bound)
        read["ci_lower"], read["ci_upper"] = float(lower), float(upper)
        if n >= 3 and float(np.var(s)) > 0.0:
            read["slope"] = float(np.cov(p, s, ddof=0)[0, 1] / np.var(s))
        read["above_tolerance"] = bool(lower > tolerance)
    wedge: dict[str, float] = {}
    for j in transitions:
        # A transition's previous bar IS the refresh bar whose next-open fills switched the
        # book; the row's refresh_bar names the baseline (the OLD book) and is not the key.
        key = str(j["prev_bar"])
        wedge[key] = wedge.get(key, 0.0) + (j["paper"] - j["shadow"])
    read["wedge_by_refresh"] = wedge

    corr = read["correlation"]
    lower, upper = read["ci_lower"], read["ci_upper"]
    enough = n >= min_sessions
    if n and upper is not None and upper < -tolerance:
        read["verdict"] = "contradicted"
    elif enough and corr is not None and corr < corr_floor:
        read["verdict"] = "contradicted"
    elif (
        enough
        and corr is not None
        and corr >= corr_floor
        and lower is not None
        and upper is not None
        and lower >= -tolerance
        and upper <= tolerance
    ):
        read["verdict"] = "consistent"
    return read


# ---------------------------------------------------------------------------
# Book concordance (weights, not returns)
# ---------------------------------------------------------------------------


def book_concordance(held_weights: pd.Series, target_weights: pd.Series) -> dict:
    """How faithfully the held book tracks the last refresh's target book.

    The promotion read (docs/momentum_design.md §3) is defined on the paper
    equity stream, but the stream is only evidence about the *strategy* to the
    extent the instrument actually holds the strategy's book — a 0.19-gross
    partial-fill book and a full B1 book produce indistinguishable monitor
    output. This read makes the gap first-order visible at any sample size:

    * ``active_share``  — ``0.5 * Σ|w_held − w_target|`` over the union of
      names (absent = explicit 0.0); 0 is a perfect replication, 1 is a
      disjoint book.
    * ``weight_corr``   — Pearson correlation of the two weight vectors over
      the union (``None`` when either side is degenerate).
    * ``gross_held`` / ``gross_target`` / ``gross_ratio`` — how much of the
      decided gross is actually deployed.

    Pure telemetry: it gates nothing and moves no ratified statistic.
    """
    union = held_weights.index.union(target_weights.index)
    held = held_weights.reindex(union).fillna(0.0).astype(float)
    target = target_weights.reindex(union).fillna(0.0).astype(float)
    gross_held = float(held.abs().sum())
    gross_target = float(target.abs().sum())
    corr: float | None = None
    if len(union) >= 2 and float(held.std()) > 0.0 and float(target.std()) > 0.0:
        raw_corr = float(held.corr(target))
        corr = raw_corr if math.isfinite(raw_corr) else None
    return {
        "active_share": float(0.5 * (held - target).abs().sum()),
        "weight_corr": corr,
        "gross_held": gross_held,
        "gross_target": gross_target,
        "gross_ratio": gross_held / gross_target if gross_target > 0.0 else None,
        "n_held": int((held != 0.0).sum()),
        "n_target": int((target != 0.0).sum()),
    }
