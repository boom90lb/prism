"""Capacity and cost-toll diagnostics (SPEC.md §3 law 5, §10).

Two distinct cost lenses, deliberately separated because they bind at different
scales:

* ``capacity_curve`` — net performance as a function of deployed AUM, driven by
  the square-root ADV participation-impact term in
  ``prism.execution.target_weights``. This is a **scaling-readiness** ruler: at a
  retail account (~$2k-$10k) trading liquid names the sqrt-ADV term is
  negligible and the curve is flat, so it does not bind *today*. It exists so a
  future net-positive strategy is never size-blind — it answers "at what AUM does
  this edge die?".

* ``cost_toll`` — the cost the operator actually pays now: turnover times
  effective spread, and the fraction of periods where cost exceeds the gross
  return (the "toll-booth" regime the residual slice fell into). This is the
  primary retail cost lens and the input to the SPEC.md §10 kill-criterion.

Both are pure functions over already-computed backtest artifacts / panels; they
add no new cost model, they only re-read the shipped one at different scales.

A third lens prices **crowding**: what an account's own trades cost when other
deployments of the same book trade the same names in the same direction on the
same session. ``crowd_impact_cost``, ``crowd_capacity_table`` and
``breakeven_follower_capital`` re-price a run's recorded per-name trades under the
square-root impact law with the other deployments' flow added. Cost per own dollar
traded in name *i* on session *t* is

    Y * sigma_ti * sqrt(f * q_ti * (1 + phi * F / E_t) / (V_ti * w))

for each execution window that takes a fraction *f* of the order and a share *w*
of the day's dollar volume, where *q* is the account's own order notional, *E* its
equity, *F* the aggregate equity of the other deployments, *phi* the fraction of
that equity trading on the same session, *sigma* the name's daily volatility and
*V* its daily dollar volume. The other deployments are assumed to hold scaled
copies of the account's book, so their flow in a name is ``phi * F / E`` times the
account's. With ``F = 0``, one window and ``w = 1`` this is the participation term
``CostModel.charge_trade`` charges at ``participation_coeff = Y * sigma * 1e4``.
In a book whose targets are set without reference to costs, the re-pricing keeps
every decision; it ignores only the small change that different costs make to the
equity path, and so to later order sizes. Its inputs (*Y*, *sigma*, the window
shares, *F*, *phi*) are scenario assumptions: paper fills cannot measure impact,
and the results must not be used to choose among configurations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from prism.config import ExecutionConfig
from prism.execution.target_weights import backtest_target_weights
from prism.validation.trials import compute_trial_metrics

_NAN_TOLL: dict[str, float] = {
    "avg_turnover": float("nan"),
    "avg_cost": float("nan"),
    "total_cost": float("nan"),
    "cost_to_gross": float("nan"),
    "toll_booth_fraction": float("nan"),
}


def _metric(value: float | None) -> float:
    return float(value) if value is not None else float("nan")


def capacity_curve(
    open_prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    aum_levels: list[float],
    execution: ExecutionConfig | None = None,
    *,
    max_gross_exposure: float | None = None,
    dividends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Net/gross Sharpe vs deployed AUM under the sqrt-ADV impact model.

    Re-runs ``backtest_target_weights`` at each AUM in ``aum_levels`` (as
    ``initial_capital``), which is the only quantity the ADV participation-impact
    term scales with (``participation = trade_abs * initial_capital / adv``). The
    same target weights are used at every level, so the curve isolates the cost
    of size: larger AUM -> larger participation -> larger sqrt-ADV impact -> lower
    net Sharpe. The AUM where net Sharpe crosses zero (or an operator hurdle) is
    the capacity ceiling. Per-level numbers come from the canonical
    ``compute_trial_metrics`` — this module owns no Sharpe/gross math of its own.

    Requires ``execution.adv_impact_coeff > 0`` and a ``dollar_volume`` panel; with
    ``adv_impact_coeff == 0`` the curve is flat by construction (a correct, and
    informative, result: impact does not bind). Returns a DataFrame indexed by
    ``aum`` with columns ``net_sharpe_ann``, ``gross_sharpe_ann``,
    ``ann_return``, ``total_cost``, ``avg_turnover``.
    """
    if not aum_levels:
        raise ValueError("aum_levels must be non-empty")
    if any(a <= 0 for a in aum_levels):
        raise ValueError(f"aum_levels must all be > 0, got {aum_levels}")
    execution = execution or ExecutionConfig()

    rows: list[dict[str, float]] = []
    for aum in sorted(float(a) for a in aum_levels):
        result = backtest_target_weights(
            open_prices=open_prices,
            target_weights=target_weights,
            execution=execution,
            initial_capital=aum,
            max_gross_exposure=max_gross_exposure,
            dividends=dividends,
            dollar_volume=dollar_volume,
        )
        trial = compute_trial_metrics(result.returns, result.costs)
        rows.append(
            {
                "aum": aum,
                "net_sharpe_ann": _metric(trial["annualized_sharpe"]),
                "gross_sharpe_ann": _metric(trial["gross_annualized_sharpe"]),
                "ann_return": float(result.metrics.get("annualized_return", float("nan"))),
                "total_cost": _metric(trial["total_cost"]),
                "avg_turnover": _metric(trial["avg_turnover"]),
            }
        )
    return pd.DataFrame(rows).set_index("aum")


def cost_toll(returns: pd.Series, costs: pd.DataFrame) -> dict[str, float]:
    """Turnover x spread vs gross — the primary retail cost lens (SPEC.md §10).

    ``returns`` is the net periodic return series; ``costs`` is the per-period
    cost frame from ``backtest_target_weights`` (columns ``total``, ``turnover``,
    ``gross``). Reconstructs the gross return as ``net + total_cost`` and reports:

    * ``avg_turnover`` — mean one-way turnover per period.
    * ``avg_cost`` / ``total_cost`` — mean / summed per-period cost drag.
    * ``cost_to_gross`` — total cost divided by total |gross return| (the share of
      the gross edge consumed by cost).
    * ``toll_booth_fraction`` — fraction of periods where per-period cost exceeds
      the magnitude of that period's gross return. A value near/above ~0.4 is the
      toll-booth regime the SPEC.md §10 kill-criterion fires on.

    NB: ``cost_to_gross`` divides by ``sum(|gross_t|)`` (per-period magnitude,
    robust to sign-cancelling PnL). The fold-level ``cost_to_gross_pnl`` from
    ``research.arbitrage.residual_walk_forward._fold_cost_share`` divides by
    ``|sum(gross_t)|`` (net PnL magnitude) — a deliberately different statistic;
    the two will not reconcile numerically.

    Returns NaN fields on empty input rather than raising, so a monitor can call
    it on a fresh strategy's first (empty) window.
    """
    net = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if net.dropna().empty or "total" not in costs.columns:
        return dict(_NAN_TOLL)
    total_cost = pd.to_numeric(costs["total"], errors="coerce").reindex(net.index).fillna(0.0)
    aligned = pd.DataFrame({"net": net, "cost": total_cost}).dropna(subset=["net"])
    gross = aligned["net"] + aligned["cost"]
    avg_turnover = (
        float(pd.to_numeric(costs["turnover"], errors="coerce").mean())
        if "turnover" in costs.columns
        else float("nan")
    )
    gross_abs_sum = float(gross.abs().sum())
    total_cost_sum = float(aligned["cost"].sum())
    return {
        "avg_turnover": avg_turnover,
        "avg_cost": float(aligned["cost"].mean()),
        "total_cost": total_cost_sum,
        "cost_to_gross": total_cost_sum / gross_abs_sum if gross_abs_sum > 1e-12 else float("nan"),
        "toll_booth_fraction": float((aligned["cost"] > gross.abs()).mean()),
    }


# ---------------------------------------------------------------------------
# Crowding re-pricing (see the module docstring)
# ---------------------------------------------------------------------------

#: Allowed error in the sum of a scenario's order fractions.
_FRACTION_SUM_TOLERANCE = 1e-9


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


@dataclass(frozen=True)
class CrowdScenario:
    """Who else trades the account's book, and where its orders execute.

    ``follower_capital`` is *F*, the aggregate equity of the other deployments
    in dollars (>= 0), held fixed over the run. ``synchronized_fraction`` is
    *phi*, the fraction of *F* that trades on the same session as the account,
    in (0, 1]: 1 when every deployment trades on the same sessions as the
    account, 1/21 for single-phase monthly books spread evenly over 21
    phases. ``windows`` lists ``(order_fraction, volume_share)`` pairs: the
    fraction of each order executed in a window and that window's share of the
    day's dollar volume, for example ``((1.0, 0.01),)`` for an opening auction
    at 1% of volume or ``((0.5, 0.01), (0.5, 0.08))`` for half at the open and
    half at the close. Order fractions are positive and sum to 1; volume shares
    lie in (0, 1]. ``impact_y`` is *Y*, the coefficient of the square-root law
    (> 0). ``label`` is free text carried into the table.
    """

    follower_capital: float = 0.0
    synchronized_fraction: float = 1.0
    windows: tuple[tuple[float, float], ...] = ((1.0, 1.0),)
    impact_y: float = 0.7
    label: str = ""

    def __post_init__(self) -> None:
        capital = _finite_number("follower_capital", self.follower_capital)
        if capital < 0.0:
            raise ValueError(f"follower_capital must be >= 0, got {capital!r}")
        phi = _finite_number("synchronized_fraction", self.synchronized_fraction)
        if not 0.0 < phi <= 1.0:
            raise ValueError(f"synchronized_fraction must lie in (0, 1], got {phi!r}")
        y = _finite_number("impact_y", self.impact_y)
        if y <= 0.0:
            raise ValueError(f"impact_y must be > 0, got {y!r}")
        try:
            pairs = tuple(tuple(pair) for pair in self.windows)
        except TypeError as exc:
            raise ValueError(f"windows must be (order_fraction, volume_share) pairs, got {self.windows!r}") from exc
        if not pairs or any(len(pair) != 2 for pair in pairs):
            raise ValueError(f"windows must be a non-empty sequence of (order_fraction, volume_share) pairs, "
                             f"got {self.windows!r}")
        windows: list[tuple[float, float]] = []
        for fraction, share in pairs:
            f = _finite_number("window order_fraction", fraction)
            w = _finite_number("window volume_share", share)
            if not 0.0 < f <= 1.0:
                raise ValueError(f"window order_fraction must lie in (0, 1], got {f!r}")
            if not 0.0 < w <= 1.0:
                raise ValueError(f"window volume_share must lie in (0, 1], got {w!r}")
            windows.append((f, w))
        total = sum(f for f, _ in windows)
        if abs(total - 1.0) > _FRACTION_SUM_TOLERANCE:
            raise ValueError(f"window order fractions must sum to 1, got {total!r}: every order executes in full")
        object.__setattr__(self, "follower_capital", capital)
        object.__setattr__(self, "synchronized_fraction", phi)
        object.__setattr__(self, "impact_y", y)
        object.__setattr__(self, "windows", tuple(windows))
        object.__setattr__(self, "label", str(self.label))

    @property
    def window_factor(self) -> float:
        """``sum_k f_k ** 1.5 / sqrt(w_k)`` over the windows.

        An order of notional *q* split as ``f_k * q`` into windows of volume
        ``w_k * V`` costs ``Y * sigma * q ** 1.5 * sqrt(m / V)`` times this factor
        in dollars, where *m* is the crowd multiplier ``1 + phi * F / E``. One
        window at the full day's volume gives 1.
        """
        return float(sum(f**1.5 / math.sqrt(w) for f, w in self.windows))

    def describe_windows(self) -> str:
        """``"0.5@0.01+0.5@0.08"``: each window's order fraction at its volume share."""
        return "+".join(f"{f:g}@{w:g}" for f, w in self.windows)


@dataclass(frozen=True)
class _CrowdInputs:
    notional: np.ndarray  # |own order notional|, bars x names
    traded: np.ndarray  # notional > 0
    dollar_volume: np.ndarray  # floored, 1.0 where not traded
    sigma: np.ndarray  # 1.0 where not traded
    equity: np.ndarray  # one value per bar


def _aligned_panel(name: str, value: object, trades: pd.DataFrame) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=trades.index, columns=trades.columns).to_numpy(dtype=float)
    if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
        raise ValueError(f"{name} must be a bar-by-name DataFrame aligned with trades, or one number for every cell")
    return np.full(trades.shape, _finite_number(name, value))


def _crowd_inputs(
    trades: pd.DataFrame,
    dollar_volume: pd.DataFrame | float,
    sigma: pd.DataFrame | float,
    equity: pd.Series | float,
    floor_dollars: float,
) -> _CrowdInputs:
    """Align and check the inputs; every refusal names the offending cell (SPEC.md N7)."""
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        raise ValueError("trades must be a non-empty bar-by-name DataFrame of the account's own order notional")
    try:
        notional = np.abs(trades.to_numpy(dtype=float))
    except (TypeError, ValueError) as exc:
        raise ValueError("trades must hold numbers: the account's order notional in dollars") from exc
    if not np.isfinite(notional).all():
        row, col = np.argwhere(~np.isfinite(notional))[0]
        raise ValueError(
            f"trades is {float(notional[row, col])!r} for {trades.columns[col]!r} on {trades.index[row]}; "
            "pass 0.0 for a name not traded on a bar, so that a missing trade cannot pass as a free one"
        )
    traded = notional > 0.0
    if not traded.any():
        raise ValueError("trades holds no non-zero order: a run with no trades has no impact cost to price")
    floor = _finite_number("floor_dollars", floor_dollars)
    if floor < 0.0:
        raise ValueError(f"floor_dollars must be >= 0, got {floor!r}")

    checked: dict[str, np.ndarray] = {}
    for name, value in (("dollar_volume", dollar_volume), ("sigma", sigma)):
        panel = _aligned_panel(name, value, trades)
        bad = traded & ~(np.isfinite(panel) & (panel > 0.0))
        if bad.any():
            row, col = np.argwhere(bad)[0]
            raise ValueError(
                f"{name} is {float(panel[row, col])!r} for {trades.columns[col]!r} on {trades.index[row]}, where the "
                "account trades; impact on a name with no positive, finite statistic cannot be priced"
            )
        checked[name] = np.where(traded, panel, 1.0)
    # The floor follows CostModel._adv_impact: a non-positive statistic is refused above,
    # and a positive one below the floor is priced at the floor.
    volume = np.where(traded, np.maximum(checked["dollar_volume"], floor), 1.0)

    if isinstance(equity, pd.Series):
        marks = equity.reindex(trades.index).to_numpy(dtype=float)
    elif isinstance(equity, (pd.DataFrame, np.ndarray, list, tuple)):
        raise ValueError("equity must be one number or a Series indexed like trades")
    else:
        marks = np.full(len(trades.index), _finite_number("equity", equity))
    bad_marks = ~(np.isfinite(marks) & (marks > 0.0))
    if bad_marks.any():
        row = int(np.argmax(bad_marks))
        raise ValueError(
            f"equity is {float(marks[row])!r} on {trades.index[row]}; every bar needs a positive, finite mark"
        )
    return _CrowdInputs(notional, traded, volume, checked["sigma"], marks)


def _own_cost(inputs: _CrowdInputs, scenario: CrowdScenario) -> np.ndarray:
    """Dollar cost of each own order with no other deployment trading (``F = 0``)."""
    cost = scenario.impact_y * inputs.sigma * inputs.notional**1.5 / np.sqrt(inputs.dollar_volume)
    return np.where(inputs.traded, cost * scenario.window_factor, 0.0)


def _crowd_multiplier(inputs: _CrowdInputs, scenario: CrowdScenario) -> np.ndarray:
    """``sqrt(1 + phi * F / E_t)`` per bar: how much the other deployments' flow raises cost per own dollar."""
    return np.sqrt(1.0 + scenario.synchronized_fraction * scenario.follower_capital / inputs.equity)


def _periods(periods_per_year: int) -> int:
    if not isinstance(periods_per_year, int) or isinstance(periods_per_year, bool) or periods_per_year < 1:
        raise ValueError(f"periods_per_year must be an integer >= 1, got {periods_per_year!r}")
    return periods_per_year


def _impact_budget(margin_annual: float, charged_annual: float) -> float:
    margin = _finite_number("margin_annual", margin_annual)
    charged = _finite_number("charged_annual", charged_annual)
    if charged < 0.0:
        raise ValueError(f"charged_annual must be >= 0, got {charged!r}")
    return margin + charged


def _breakeven(own_per_bar: np.ndarray, equity: np.ndarray, phi: float, budget: float, periods: int) -> float:
    """The *F* at which the impact cost equals ``budget``; 0.0 when the cost at ``F = 0`` already reaches it.

    The annual impact cost is ``periods * mean_t(a_t * sqrt(1 + phi * F / E_t))``,
    with ``a_t`` the own cost of bar *t* over its equity: increasing in *F* and
    unbounded. With ``A = periods * mean_t(a_t)`` and ``u = (budget / A) ** 2 - 1``,
    the root lies in ``[min(E) * u / phi, max(E) * u / phi]``, and is the closed
    form ``E * u / phi`` when equity is constant. Bisection closes the bracket
    otherwise.
    """
    own_annual = float(own_per_bar.mean()) * periods
    if budget <= own_annual:
        return 0.0
    growth = (budget / own_annual) ** 2 - 1.0
    lo, hi = float(equity.min()) * growth / phi, float(equity.max()) * growth / phi
    if lo == hi:
        return lo

    def excess(capital: float) -> float:
        return budget - periods * float(np.mean(own_per_bar * np.sqrt(1.0 + phi * capital / equity)))

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if excess(mid) >= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


def _own_capacity_multiple(own_annual: float, budget: float) -> float:
    """The largest multiple *k* of the account's size at which its own impact stays within ``budget``.

    Scaling the account by *k* scales every order by *k*, so own impact cost per
    unit of equity scales by ``sqrt(k)``: ``own_annual * sqrt(k) = budget`` gives
    ``k = (budget / own_annual) ** 2``, and a budget at or below zero gives 0.0.
    """
    return (budget / own_annual) ** 2 if budget > 0.0 else 0.0


def crowd_impact_cost(
    trades: pd.DataFrame,
    dollar_volume: pd.DataFrame | float,
    sigma: pd.DataFrame | float,
    equity: pd.Series | float,
    scenario: CrowdScenario,
    *,
    floor_dollars: float = 0.0,
) -> pd.DataFrame:
    """Dollar impact cost of each of the account's orders under one crowding scenario.

    ``trades`` is the account's own order notional in dollars, one row per bar
    and one column per name; the sign is ignored, because the other deployments
    trade in the same direction. Pass 0.0 where nothing is traded: a NaN is
    refused rather than read as no trade. ``dollar_volume`` (*V*) and ``sigma``
    (the daily volatility) are bar-by-name frames, reindexed onto ``trades``, or
    one number for every cell; both must be positive and finite wherever the
    account trades. ``floor_dollars`` prices a positive *V* below it at the floor,
    as ``CostModel`` does with ``dollar_volume.impact.floor_dollars``. ``equity``
    (*E*) is the account's equity, one number or a Series per bar of ``trades``.

    Returns a frame shaped like ``trades``: for each order of notional *q*,
    ``Y * sigma * q ** 1.5 * sqrt((1 + phi * F / E) / V) * scenario.window_factor``,
    and 0.0 where nothing was traded. With ``F = 0`` and one window at the full
    day's volume this is ``CostModel.charge_trade``'s participation term at
    ``participation_coeff = Y * sigma * 1e4`` on the same *V*.
    """
    inputs = _crowd_inputs(trades, dollar_volume, sigma, equity, floor_dollars)
    cost = _own_cost(inputs, scenario) * _crowd_multiplier(inputs, scenario)[:, None]
    return pd.DataFrame(cost, index=trades.index, columns=trades.columns)


def breakeven_follower_capital(
    trades: pd.DataFrame,
    dollar_volume: pd.DataFrame | float,
    sigma: pd.DataFrame | float,
    equity: pd.Series | float,
    scenario: CrowdScenario,
    *,
    margin_annual: float,
    charged_annual: float = 0.0,
    periods_per_year: int = 252,
    floor_dollars: float = 0.0,
) -> float:
    """The follower capital *F* at which the book's excess over its hurdle reaches zero.

    ``margin_annual`` is the book's expected return in excess of its hurdle, as
    an annual fraction of equity, net of the costs it was computed with.
    ``charged_annual`` is the part of those costs that this re-pricing replaces:
    the participation impact the run was charged, as an annual fraction of
    equity. The impact the book can bear is then ``margin_annual +
    charged_annual``, and the break-even *F* is where the annual impact cost of
    the account's orders under the scenario (the mean over the bars of
    ``trades``, traded or not, of :func:`crowd_impact_cost` over equity, times
    ``periods_per_year``) equals it. The default ``charged_annual = 0.0`` treats
    the margin as containing no impact charge, so the account's own impact in
    the scenario's windows is counted against it in full; a margin computed net
    of a participation term should pass that term here, or its own impact is
    counted twice.

    ``scenario.follower_capital`` is ignored, because *F* is the unknown; the
    other fields and the inputs are as in :func:`crowd_impact_cost`. Returns 0.0
    when the account's own impact already uses up the budget at ``F = 0``.
    Otherwise the result is finite and positive, since the cost grows without
    bound in *F*.
    """
    inputs = _crowd_inputs(trades, dollar_volume, sigma, equity, floor_dollars)
    periods = _periods(periods_per_year)
    budget = _impact_budget(margin_annual, charged_annual)
    own_per_bar = _own_cost(inputs, scenario).sum(axis=1) / inputs.equity
    return _breakeven(own_per_bar, inputs.equity, scenario.synchronized_fraction, budget, periods)


def crowd_capacity_table(
    trades: pd.DataFrame,
    dollar_volume: pd.DataFrame | float,
    sigma: pd.DataFrame | float,
    equity: pd.Series | float,
    scenarios: Sequence[CrowdScenario],
    *,
    margin_annual: float,
    charged_annual: float = 0.0,
    users_at: Sequence[float] = (100_000.0, 1_000_000.0),
    thin_dollar_volume: float = 100_000_000.0,
    periods_per_year: int = 252,
    floor_dollars: float = 0.0,
) -> pd.DataFrame:
    """One row per scenario: impact cost, excess over the hurdle and break-even capital.

    Inputs are as in :func:`crowd_impact_cost` and
    :func:`breakeven_follower_capital`. All rates are annual fractions of
    equity. Columns:

    * ``label``, ``follower_capital``, ``synchronized_fraction``, ``impact_y``
      and ``windows`` (:meth:`CrowdScenario.describe_windows`): the scenario;
    * ``margin_annual`` and ``charged_annual``: the inputs;
    * ``own_cost_annual``: the impact cost of the account's own orders in this
      scenario's windows with no other deployment trading;
    * ``cost_annual``: the same cost with the scenario's follower capital;
    * ``crowding_drag_annual``: ``cost_annual - own_cost_annual``, the part the
      other deployments add;
    * ``excess_over_hurdle_annual``: ``margin_annual + charged_annual -
      cost_annual``;
    * ``breakeven_follower_capital`` and one ``breakeven_users_at_<size>``
      column per entry of ``users_at``: the break-even *F*, and that capital
      as a number of deployments of each size;
    * ``own_capacity_multiple`` and ``own_capacity_initial_equity``: with no
      other deployment, the largest multiple of the account's size at which the
      excess is still >= 0, and that multiple times the first bar's equity. Every
      order is scaled by the multiple, so whole-share rounding is ignored;
    * ``thin_name_cost_share``: the share of ``cost_annual`` incurred in orders
      on names whose dollar volume on that bar is below ``thin_dollar_volume``.
    """
    if not scenarios:
        raise ValueError("scenarios must be non-empty")
    if any(not isinstance(scenario, CrowdScenario) for scenario in scenarios):
        raise ValueError("every scenario must be a CrowdScenario")
    sizes = [_finite_number("users_at", size) for size in users_at]
    if any(size <= 0.0 for size in sizes):
        raise ValueError(f"users_at must all be > 0, got {list(users_at)!r}")
    thin = _finite_number("thin_dollar_volume", thin_dollar_volume)
    inputs = _crowd_inputs(trades, dollar_volume, sigma, equity, floor_dollars)
    periods = _periods(periods_per_year)
    budget = _impact_budget(margin_annual, charged_annual)
    thin_cells = inputs.traded & (inputs.dollar_volume < thin)

    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        own = _own_cost(inputs, scenario)
        multiplier = _crowd_multiplier(inputs, scenario)
        own_per_bar = own.sum(axis=1) / inputs.equity
        own_annual = float(own_per_bar.mean()) * periods
        cost_annual = float((own_per_bar * multiplier).mean()) * periods
        breakeven = _breakeven(own_per_bar, inputs.equity, scenario.synchronized_fraction, budget, periods)
        cells = own * multiplier[:, None]
        multiple = _own_capacity_multiple(own_annual, budget)
        row: dict[str, object] = {
            "label": scenario.label,
            "follower_capital": scenario.follower_capital,
            "synchronized_fraction": scenario.synchronized_fraction,
            "impact_y": scenario.impact_y,
            "windows": scenario.describe_windows(),
            "margin_annual": float(margin_annual),
            "charged_annual": float(charged_annual),
            "own_cost_annual": own_annual,
            "cost_annual": cost_annual,
            "crowding_drag_annual": cost_annual - own_annual,
            "excess_over_hurdle_annual": budget - cost_annual,
            "breakeven_follower_capital": breakeven,
        }
        for size in sizes:
            row[f"breakeven_users_at_{size:.0f}"] = breakeven / size
        row["own_capacity_multiple"] = multiple
        row["own_capacity_initial_equity"] = multiple * float(inputs.equity[0])
        row["thin_name_cost_share"] = float(cells[thin_cells].sum() / cells.sum())
        rows.append(row)
    return pd.DataFrame(rows)
