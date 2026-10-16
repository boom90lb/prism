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

* **T-bill-referenced return read (report only, off by default).** The return
  monitor again, against the per-session bill rate instead of zero
  (:func:`tbill_monitor_read`). It is added beside the registered hurdle-0 read
  and never replaces it: every registered conjunct reads the hurdle-0 verdict.
  It answers the question the hurdle-0 read does not, whether the paper account
  beats holding its equity in Treasury bills, and it states how long a stream
  must run before it could tell (:func:`detection_horizon_years`).
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import tomllib
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.io import sessions
from prism.live.loop import (
    read_equity_ledger,
    read_shadow_ledger,
    read_targets_ledger,
)
from prism.validation.anytime import (
    anytime_monitor_read,
    empirical_bernstein_confidence_sequence,
    empirical_bernstein_monitor_read,
)
from prism.validation.hurdle import tbill_hurdle

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
# T-bill-referenced return read (report only)
# ---------------------------------------------------------------------------

#: Sessions per year the T-bill read divides its annual hurdle by.
TBILL_PERIODS_PER_YEAR = 252

#: Default annual fee of the T-bill ETF a cash holding buys, as a fraction:
#: ``[criteria] sleeve_fee_annual`` of ``registrations/2026-10-public-baseline.toml``
#: (SGOV's expense ratio is about 0.09% by secondary sources, not checked at the issuer).
DEFAULT_TBILL_ETF_FEE = 0.0010

_TBILL_REQUIRED = ("rf_annual", "rf_source")
_TBILL_OPTIONAL = ("cash_yield", "cash_yield_source", "etf_fee_annual", "planning_sharpe", "planning_source")

#: How the detection horizon is computed, recorded beside every horizon stated.
DETECTION_HORIZON_METHOD = (
    "fixed-sample one-sided z-test on daily returns: years = (z_(1-alpha) / S)^2, the horizon at which the "
    "expected t-statistic of a stream whose true annualized Sharpe is S reaches the critical value, so about half "
    "of such streams would have rejected zero by then; a time-uniform confidence sequence pays for being valid at "
    "every n and needs longer, so this is a lower bound for the anytime read"
)


def _real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


@dataclass(frozen=True)
class TbillReference:
    """The inputs of the report-only T-bill-referenced return read, each with where it came from.

    ``rf_annual`` is the bill rate the account is measured against (0.0421 for
    4.21%) and ``rf_source`` names it. There is no default rate: a stale one
    would pass silently.

    ``cash_yield`` is what idle cash earns that the paper NAV does not record.
    The default, 0, assumes nothing beyond the NAV. Whether Alpaca's paper
    accounts credit interest on cash is not verified, and a yield the NAV
    already carries would be counted twice if it were entered here.

    ``etf_fee_annual`` matters only when the targets ledger records a T-bill
    holding (:mod:`prism.live.cash_holding`). Paper accounts do not simulate
    dividends, so the ETF's distributions are missing from the paper NAV while
    the price drop at each ex-date is in it. The read credits the ETF's mean
    share of equity at ``rf_annual - etf_fee_annual``: an assumption that the
    ETF distributes at the bill rate less its fee.

    ``planning_sharpe`` (with ``planning_source``) is the annualized Sharpe of
    the account's excess over bills at which the detection horizon is stated.
    It is a planning figure, not a measurement; without it no horizon is stated.
    """

    rf_annual: float
    rf_source: str
    cash_yield: float = 0.0
    cash_yield_source: str = "none assumed beyond the paper NAV"
    etf_fee_annual: float = DEFAULT_TBILL_ETF_FEE
    planning_sharpe: float | None = None
    planning_source: str = ""

    def __post_init__(self) -> None:
        _real("rf_annual", self.rf_annual)
        for name in ("cash_yield", "etf_fee_annual"):
            if _real(name, getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        for name in ("rf_source", "cash_yield_source", "planning_source"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be text, got {getattr(self, name)!r}")
        if not self.rf_source.strip():
            raise ValueError("rf_source must name where the bill rate came from")
        if self.planning_sharpe is not None:
            if _real("planning_sharpe", self.planning_sharpe) <= 0.0:
                raise ValueError(f"planning_sharpe must be > 0, got {self.planning_sharpe!r}")
            if not self.planning_source.strip():
                raise ValueError("planning_source must name where planning_sharpe came from")


def load_tbill_reference(path: Path | str) -> TbillReference:
    """A :class:`TbillReference` from a TOML file whose top-level keys are its fields.

    Read strictly: an unknown key is refused, so a misspelled field cannot pass
    as its default (N7). ``rf_annual`` and ``rf_source`` are required.
    """
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"T-bill reference {path} cannot be read: {exc}") from exc
    unknown = sorted(set(raw) - set(_TBILL_REQUIRED) - set(_TBILL_OPTIONAL))
    if unknown:
        raise ValueError(f"T-bill reference {path} has unknown key(s) {unknown}; allowed: "
                         f"{sorted(_TBILL_REQUIRED + _TBILL_OPTIONAL)}")
    missing = [key for key in _TBILL_REQUIRED if key not in raw]
    if missing:
        raise ValueError(f"T-bill reference {path} is missing {missing}")
    try:
        return TbillReference(**raw)
    except ValueError as exc:
        raise ValueError(f"T-bill reference {path}: {exc}") from exc


def detection_horizon_years(sharpe_annual: float, *, alpha: float = 0.05) -> float | None:
    """Years of daily returns before a stream with true annualized Sharpe ``sharpe_annual`` is expected to reject zero.

    ``(z_(1-alpha) / S)^2`` (:data:`DETECTION_HORIZON_METHOD`): about 30 years at
    S = 0.30 and about 245 at S = 0.105, at ``alpha = 0.05``. ``None`` when S is
    not positive, because such a stream is not expected to reject zero at all.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    sharpe = _real("sharpe_annual", sharpe_annual)
    if sharpe <= 0.0:
        return None
    return (NormalDist().inv_cdf(1.0 - alpha) / sharpe) ** 2


def held_book_exposure(rows: Sequence[Mapping[str, Any]] | None, prev_bars: Sequence[str]) -> dict[str, Any]:
    """The stock long side and T-bill ETF share of the book held through each return session.

    For the session whose previous bar is ``prev``, the book is the latest
    targets-ledger row whose ``refresh_bar`` is on or before ``prev``: decided
    at or before the previous close and filled at the session's open. Its long
    side is the sum of its positive weights other than the ETF's; the ETF's
    share is the ``cash_holding`` decision's ``weight`` (zero without one).
    Before the first row no book is held, so both are zero. ``rows=None`` (no
    targets ledger) gives zeros and says so in ``source``.
    """
    n = len(prev_bars)
    long_side = np.zeros(n, dtype=float)
    sleeve = np.zeros(n, dtype=float)
    symbols: set[str] = set()
    without = n
    if rows:
        ordered = sorted(rows, key=lambda row: str(row["refresh_bar"])[:10])
        keys = [str(row["refresh_bar"])[:10] for row in ordered]
        without = 0
        for i, prev in enumerate(prev_bars):
            at = bisect_right(keys, str(prev)[:10]) - 1
            if at < 0:
                without += 1
                continue
            row = ordered[at]
            holding = row.get("cash_holding") or {}
            symbol = holding.get("symbol")
            if symbol is not None:
                symbols.add(str(symbol))
                sleeve[i] = float(holding.get("weight", 0.0))
            long_side[i] = sum(max(0.0, float(w)) for s, w in (row.get("targets") or {}).items() if s != symbol)
    return {
        "source": None if rows is None else "targets ledger",
        "n_sessions": n,
        "n_sessions_without_book": int(without),
        "long_over_equity_mean": float(long_side.mean()) if n else 0.0,
        "sleeve_fraction_mean": float(sleeve.mean()) if n else 0.0,
        "etf_symbols": sorted(symbols),
    }


def tbill_monitor_read(
    equity_ledger_path: Path | str,
    reference: TbillReference,
    *,
    targets_ledger_path: Path | str | None = None,
    alpha: float = 0.05,
    bound: float = DEFAULT_NAV_RETURN_BOUND,
    method: str = "empirical_bernstein",
) -> dict:
    """The return monitor against the per-session T-bill hurdle instead of zero. Report only.

    SPEC §3's hurdle (:func:`prism.validation.hurdle.tbill_hurdle`) stated for
    the paper NAV. The account beats bills when its mean daily NAV return
    exceeds ``return_hurdle_annual / 252``, where

        return_hurdle_annual = rf - (rf - fee) * s - y * max(0, 1 - L/E - s)

    with ``L/E`` the stock long side and ``s`` the T-bill ETF's share of equity
    of the book held through each session (:func:`held_book_exposure`),
    averaged over the sessions the read covers; ``y`` is
    ``reference.cash_yield`` and ``fee`` ``reference.etf_fee_annual``. With no
    ETF and ``y = 0`` the hurdle is the bill rate itself.

    The read is :func:`paper_monitor_read` at that hurdle, with the same
    sequence, support and alpha as the registered read, plus
    ``reference = "tbill"``, ``report_only = True``, ``hurdle_annual``,
    ``hurdle_inputs`` (every input and its source), ``exposure`` and
    ``detection_horizon``. The hurdle's Sharpe form needs a positive realized
    volatility; before two returns exist it is omitted (``None``), and the
    return hurdle, which does not depend on volatility, is still stated.

    Raises ``ValueError`` when ``reference.cash_yield`` is not zero and no
    targets ledger is given (the idle fraction it applies to is unknown), and
    when the ETF's mean share exceeds the idle equity it is bought with
    (:func:`prism.validation.hurdle.tbill_hurdle` refuses it).
    """
    df = read_equity_ledger(Path(equity_ledger_path))
    series = consecutive_session_returns(df) if len(df) >= 2 else None
    returns = series["returns"] if series is not None else np.array([], dtype=float)
    prev_bars = series["prev_bars"] if series is not None else []
    if targets_ledger_path is None and reference.cash_yield != 0.0:
        raise ValueError("a non-zero cash yield needs the targets ledger: the idle fraction it applies to is unknown")
    rows = read_targets_ledger(Path(targets_ledger_path)) if targets_ledger_path is not None else None
    exposure = held_book_exposure(rows, prev_bars)
    sd = float(np.std(returns, ddof=1)) if returns.size >= 2 else float("nan")
    vol = sd * math.sqrt(TBILL_PERIODS_PER_YEAR) if math.isfinite(sd) and sd > 0.0 else None
    sleeve = exposure["sleeve_fraction_mean"]
    etf_yield = reference.rf_annual - reference.etf_fee_annual if sleeve > 0.0 else 0.0
    found = tbill_hurdle(
        reference.rf_annual,
        reference.cash_yield,
        exposure["long_over_equity_mean"],
        1.0 if vol is None else vol,
        periods_per_year=TBILL_PERIODS_PER_YEAR,
        rf_source=reference.rf_source,
        cash_yield_source=reference.cash_yield_source,
        vol_basis="" if vol is None else "realized_annualized",
        sleeve_fraction=sleeve,
        sleeve_yield=etf_yield,
        sleeve_yield_source=(
            f"ETF distributions the paper NAV does not record, assumed at rf less a {reference.etf_fee_annual} fee"
            if sleeve > 0.0
            else ""
        ),
    )
    read = paper_monitor_read(
        equity_ledger_path,
        hurdle=found.return_hurdle_annual / TBILL_PERIODS_PER_YEAR,
        alpha=alpha,
        bound=bound,
        method=method,
    )
    inputs = found.to_dict()
    if vol is None:  # tbill_hurdle needs a volatility for the Sharpe form; none is measured yet
        inputs.update(vol_annual=None, sharpe_annual=None, sharpe_periodic=None)
    if reference.planning_sharpe is None:
        horizon: dict[str, Any] = {"planning_sharpe": None, "years": None,
                                   "note": "no planning Sharpe was stated in the T-bill reference"}
    else:
        horizon = {"planning_sharpe": reference.planning_sharpe, "planning_source": reference.planning_source,
                   "alpha": float(alpha), "years": detection_horizon_years(reference.planning_sharpe, alpha=alpha),
                   "method": DETECTION_HORIZON_METHOD}
    read.update(
        reference="tbill",
        report_only=True,
        hurdle_annual=found.return_hurdle_annual,
        hurdle_inputs=inputs,
        exposure=exposure,
        detection_horizon=horizon,
    )
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
