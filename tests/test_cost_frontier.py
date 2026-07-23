"""Unit tests for the cost-frontier re-pricing arithmetic (research tier)."""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.research

from research.scripts.cost_frontier import (  # noqa: E402
    aggregate_frontier,
    break_even_one_way_spread_bps,
    daily_frontier,
    decompose_costs,
    realized_effective_spread_bps,
    reprice,
    require_finite,
    validate_against_summary,
)

COMMISSION_BPS = 1.0
SPREAD_BPS = 2.5


def synthetic_run(n: int = 504, spread_bps: float = SPREAD_BPS) -> tuple[pd.Series, pd.DataFrame]:
    """A stored-run-shaped (returns, costs) pair priced at a known flat spread.

    Gross returns and turnover vary day to day so the frontier arithmetic is
    exercised off the constant-series special case; the cost columns follow the
    engine's accounting identity exactly (total = commission_spread + impact +
    borrow, net = gross - total).
    """
    rng = np.random.default_rng(20260719)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    gross = pd.Series(rng.normal(3e-4, 1.2e-3, n), index=idx)
    turnover = pd.Series(rng.uniform(0.1, 0.4, n), index=idx)
    commission_spread = turnover * (COMMISSION_BPS + spread_bps) / 1e4
    impact = pd.Series(np.full(n, 2e-6), index=idx)
    borrow = pd.Series(np.full(n, 1e-6), index=idx)
    total = commission_spread + impact + borrow
    costs = pd.DataFrame(
        {
            "commission_spread": commission_spread,
            "impact": impact,
            "borrow": borrow,
            "total": total,
            "turnover": turnover,
            "gross": 1.0,
            "net": 0.0,
        }
    )
    returns = (gross - total).rename("daily_return")
    return returns, costs


def test_validation_gates_reject_non_finite_inputs_loudly() -> None:
    # NaN compares False against any tolerance, so a NaN-bearing column would
    # sail through every `if error > tol: raise` gate and then be dropped
    # silently downstream — the gates assert finiteness first and name the
    # offending column.
    returns, costs = synthetic_run(n=40)
    bad_costs = costs.copy()
    bad_costs.loc[bad_costs.index[5], "impact"] = np.nan
    with pytest.raises(ValueError, match="'impact'.*non-finite"):
        decompose_costs(returns, bad_costs, COMMISSION_BPS)

    bad_returns = returns.copy()
    bad_returns.iloc[3] = np.nan
    with pytest.raises(ValueError, match="'daily_return'.*non-finite"):
        decompose_costs(bad_returns, costs, COMMISSION_BPS)

    summary = {
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)),
        "total_cost": float(costs["total"].sum()),
        "avg_turnover": float(costs["turnover"].mean()),
    }
    bad_turnover = costs.copy()
    bad_turnover.loc[bad_turnover.index[2], "turnover"] = np.nan
    with pytest.raises(ValueError, match="'turnover'.*non-finite"):
        validate_against_summary(returns, bad_turnover, summary)

    # The helper's message carries the count and location for the operator.
    with pytest.raises(ValueError, match="1 non-finite"):
        require_finite(pd.Series([1.0, np.nan, 2.0]), "leg", "unit context")


def test_decomposition_recovers_effective_spread_exactly() -> None:
    returns, costs = synthetic_run()
    decomp = decompose_costs(returns, costs, COMMISSION_BPS)
    assert realized_effective_spread_bps(decomp) == pytest.approx(SPREAD_BPS, abs=1e-12)
    # The decomposition reconstructs the stored net stream (gate inside decompose_costs).
    rebuilt = decomp["gross_return"] - decomp["commission"] - decomp["spread_cost"] - decomp["fixed"]
    assert float((rebuilt - returns).abs().max()) == pytest.approx(0.0, abs=1e-15)


def test_reprice_at_as_run_spread_reproduces_stored_net() -> None:
    returns, costs = synthetic_run()
    decomp = decompose_costs(returns, costs, COMMISSION_BPS)
    net, repriced = reprice(decomp, SPREAD_BPS)
    assert float((net - returns).abs().max()) == pytest.approx(0.0, abs=1e-15)
    assert float((repriced["total"] - costs["total"]).abs().max()) == pytest.approx(0.0, abs=1e-15)


def test_decomposition_rejects_wrong_commission() -> None:
    returns, costs = synthetic_run(spread_bps=0.0)
    # Commission larger than the whole commission_spread leg -> negative spread cost.
    with pytest.raises(ValueError, match="commission_bps does not match"):
        decompose_costs(returns, costs, commission_bps=5.0)


def test_break_even_is_closed_form_zero_of_mean_net() -> None:
    returns, costs = synthetic_run()
    decomp = decompose_costs(returns, costs, COMMISSION_BPS)
    s_star = break_even_one_way_spread_bps(decomp)
    assert s_star > 0.0  # seeded gross edge clears commission + fixed costs
    net, _ = reprice(decomp, s_star)
    assert float(net.mean()) == pytest.approx(0.0, abs=1e-15)
    # Hand-check the closed form on the decomposition itself.
    base = decomp["gross_return"] - decomp["commission"] - decomp["fixed"]
    assert s_star == pytest.approx(1e4 * float(base.mean()) / float(decomp["turnover"].mean()))


def test_break_even_negative_when_gross_below_fixed_costs() -> None:
    returns, costs = synthetic_run()
    # Push gross down so even zero spread cannot rescue the config.
    starved = (returns - 5e-4).rename("daily_return")
    decomp = decompose_costs(starved, costs, COMMISSION_BPS)
    assert break_even_one_way_spread_bps(decomp) < 0.0


def test_daily_frontier_monotone_with_invariant_gross() -> None:
    returns, costs = synthetic_run()
    result = daily_frontier(returns, costs, COMMISSION_BPS, levels=(0.0, 0.5, 1.0, 2.0))
    rows = {row["level"]: row for row in result["frontier"]}
    flat = [rows[f"flat_{s:g}bp"] for s in (0.0, 0.5, 1.0, 2.0)]
    sharpes = [row["net_annualized_sharpe"] for row in flat]
    assert sharpes == sorted(sharpes, reverse=True), "net Sharpe must fall as spread rises"
    gross_sharpes = {round(row["gross_annualized_sharpe"], 12) for row in flat}
    assert len(gross_sharpes) == 1, "gross Sharpe is spread-invariant by construction"
    # Costs rise monotonically with the spread level.
    assert [row["total_cost"] for row in flat] == sorted(row["total_cost"] for row in flat)


def test_daily_frontier_as_run_and_ceiling_rows() -> None:
    returns, costs = synthetic_run()
    result = daily_frontier(returns, costs, COMMISSION_BPS)
    rows = {row["level"]: row for row in result["frontier"]}

    as_run = rows["bucket_v1_as_run"]
    assert as_run["one_way_spread_bps"] == pytest.approx(SPREAD_BPS, abs=1e-12)
    # Flat synthetic pricing: the as-run row must match the flat re-pricing at 2.5 bp.
    net_25, costs_25 = reprice(decompose_costs(returns, costs, COMMISSION_BPS), SPREAD_BPS)
    assert as_run["net_annualized_sharpe"] == pytest.approx(
        float(net_25.mean() / net_25.std(ddof=1) * np.sqrt(252.0))
    )

    ceiling = rows["zero_cost_gross_ceiling"]
    assert ceiling["gross_ceiling"] is True
    assert "DSR is 0.145" in ceiling["deflated_read"]
    assert ceiling["one_way_spread_bps"] is None
    assert ceiling["total_cost"] == pytest.approx(0.0)
    assert ceiling["cost_to_gross"] == pytest.approx(0.0)
    assert ceiling["net_annualized_sharpe"] == pytest.approx(ceiling["gross_annualized_sharpe"])
    assert ceiling["net_annualized_sharpe"] > as_run["net_annualized_sharpe"]


def aggregate_record(**overrides) -> dict:
    """A self-consistent aggregate record with round numbers, then overridden."""
    record = {
        "config_hash": "cafecafecafe",
        "n_obs": 504,
        "avg_daily_turnover": 0.25,
        "commission_bps": 1.0,
        "spread_bps_as_run": 1.0,
        "gross_cum_return_arith": 0.04,
        "cost_cum_arith": 0.03,
        "net_annualized_sharpe": 0.25,
        "residual_set_dsr_recorded": 0.05,
    }
    record.update(overrides)
    # Make the documented gross Sharpe exactly consistent unless overridden.
    years = record["n_obs"] / 252.0
    net_ann = (record["gross_cum_return_arith"] - record["cost_cum_arith"]) / years
    vol = net_ann / record["net_annualized_sharpe"]
    record.setdefault("doc_gross_annualized_sharpe", record["gross_cum_return_arith"] / years / vol)
    return record


def test_aggregate_frontier_reproduces_as_run_and_break_even() -> None:
    record = aggregate_record()
    result = aggregate_frontier(record, levels=(0.0, 0.5, 1.0, 2.0))
    assert result["granularity"] == "aggregate"
    assert result["as_run_net_annualized_sharpe"] == pytest.approx(record["net_annualized_sharpe"], abs=1e-12)

    # Break-even: net_ann(s*) == 0 by the linear model.
    years = record["n_obs"] / 252.0
    tau = record["avg_daily_turnover"]
    gross_ann = record["gross_cum_return_arith"] / years
    fixed_ann = record["cost_cum_arith"] / years - 252.0 * tau * 2.0 / 1e4
    s_star = result["break_even_one_way_spread_bps"]
    assert gross_ann - 252.0 * tau * (1.0 + s_star) / 1e4 - fixed_ann == pytest.approx(0.0, abs=1e-15)

    rows = {row["level"]: row for row in result["frontier"]}
    sharpes = [rows[f"flat_{s:g}bp"]["net_annualized_sharpe"] for s in (0.0, 0.5, 1.0, 2.0)]
    assert sharpes == sorted(sharpes, reverse=True)
    ceiling = rows["zero_cost_gross_ceiling"]
    assert ceiling["gross_ceiling"] is True
    assert ceiling["net_annualized_sharpe"] == pytest.approx(result["implied_gross_annualized_sharpe"])


def test_aggregate_frontier_rejects_inconsistent_gross_sharpe() -> None:
    record = aggregate_record()
    record["doc_gross_annualized_sharpe"] = record["doc_gross_annualized_sharpe"] + 0.2
    with pytest.raises(ValueError, match="cross-check FAILED"):
        aggregate_frontier(record)
