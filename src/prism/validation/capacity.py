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
"""

from __future__ import annotations

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
