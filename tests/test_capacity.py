"""Tests for capacity + cost-toll diagnostics (SPEC.md §3 law 5, §10)."""

import numpy as np
import pandas as pd
import pytest

from prism.config import ExecutionConfig
from prism.execution.target_weights import backtest_target_weights
from prism.validation.capacity import capacity_curve, cost_toll


def _panel(n=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    syms = ["A", "B", "C"]
    prices = pd.DataFrame(
        100 * np.cumprod(1 + 0.001 * rng.standard_normal((n, 3)), axis=0), index=dates, columns=syms
    )
    tw = pd.DataFrame(0.2 * np.sign(rng.standard_normal((n, 3))), index=dates, columns=syms)
    dvol = pd.DataFrame(1e6, index=dates, columns=syms)
    return prices, tw, dvol


class TestCapacityCurve:
    def test_cost_monotone_increasing_in_aum(self):
        prices, tw, dvol = _panel()
        exe = ExecutionConfig(adv_impact_coeff=10.0, adv_impact_model="sqrt", adv_floor_dollars=1.0)
        cc = capacity_curve(prices, tw, dvol, [1e3, 1e5, 1e7, 1e9], execution=exe)
        costs = cc["total_cost"].to_numpy()
        assert np.all(np.diff(costs) >= -1e-9)  # non-decreasing in AUM

    def test_gross_sharpe_invariant_to_aum(self):
        # AUM scales only the sqrt-ADV impact; gross (pre-cost) is unchanged up to
        # float reconstruction round-off (gross = net + cost re-adds the AUM term).
        prices, tw, dvol = _panel()
        exe = ExecutionConfig(adv_impact_coeff=10.0, adv_impact_model="sqrt", adv_floor_dollars=1.0)
        cc = capacity_curve(prices, tw, dvol, [1e3, 1e9], execution=exe)
        g = cc["gross_sharpe_ann"].to_numpy()
        assert g[0] == pytest.approx(g[1], rel=1e-9)

    def test_flat_when_impact_off(self):
        # adv_impact_coeff=0 -> AUM does not enter cost -> net flat.
        prices, tw, dvol = _panel()
        exe = ExecutionConfig(adv_impact_coeff=0.0)
        cc = capacity_curve(prices, tw, dvol, [1e3, 1e9], execution=exe)
        assert cc["net_sharpe_ann"].nunique() == 1

    def test_index_sorted_and_named(self):
        prices, tw, dvol = _panel()
        cc = capacity_curve(prices, tw, dvol, [1e7, 1e3, 1e5])
        assert list(cc.index) == sorted(cc.index)
        assert cc.index.name == "aum"

    def test_rejects_bad_aum(self):
        prices, tw, dvol = _panel()
        with pytest.raises(ValueError):
            capacity_curve(prices, tw, dvol, [])
        with pytest.raises(ValueError):
            capacity_curve(prices, tw, dvol, [1e3, -1.0])


class TestCostToll:
    def test_reconstructs_gross_and_reports_toll(self):
        prices, tw, dvol = _panel()
        exe = ExecutionConfig(adv_impact_coeff=10.0, adv_impact_model="sqrt", adv_floor_dollars=1.0)
        res = backtest_target_weights(prices, tw, execution=exe, initial_capital=1e7, dollar_volume=dvol)
        toll = cost_toll(res.returns, res.costs)
        assert 0.0 <= toll["toll_booth_fraction"] <= 1.0
        assert toll["cost_to_gross"] >= 0.0
        assert toll["total_cost"] == pytest.approx(res.costs["total"].sum())

    def test_zero_cost_gives_zero_toll(self):
        prices, tw, dvol = _panel()
        exe = ExecutionConfig(commission_bps=0.0, spread_bps=0.0, slippage_coeff=0.0, borrow_rate_bps_annual=0.0)
        res = backtest_target_weights(prices, tw, execution=exe, initial_capital=1.0)
        toll = cost_toll(res.returns, res.costs)
        assert toll["total_cost"] == pytest.approx(0.0)
        assert toll["toll_booth_fraction"] == pytest.approx(0.0)

    def test_empty_input_returns_nan_fields(self):
        toll = cost_toll(pd.Series(dtype=float), pd.DataFrame())
        assert np.isnan(toll["toll_booth_fraction"])
        assert np.isnan(toll["cost_to_gross"])
