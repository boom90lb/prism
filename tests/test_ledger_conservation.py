"""N4 ledger-conservation property test (SPEC.md §1 N4, §3 law 1).

The single defensible residue of the "capital conservation" law: equity moves
only by realized PnL minus charged costs, with no cash created or destroyed
across a rebalance. This reconstructs the equity path independently from the
backtest's own fills / prices / dividends / costs and asserts it matches — a
cash-leak bug (the most common source of phantom Sharpe) would break it.
"""

import numpy as np
import pandas as pd
import pytest

from prism.config import ExecutionConfig
from prism.execution.target_weights import backtest_target_weights


def _panel(n=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    syms = ["A", "B", "C"]
    prices = pd.DataFrame(
        100 * np.cumprod(1 + 0.002 * rng.standard_normal((n, 3)), axis=0), index=dates, columns=syms
    )
    tw = pd.DataFrame(0.25 * rng.standard_normal((n, 3)), index=dates, columns=syms)
    dvol = pd.DataFrame(1e7, index=dates, columns=syms)
    return prices, tw, dvol, dates, syms


def _reconstruct_returns(result, prices):
    """Independently rebuild net returns from fills + prices + recorded costs."""
    fills = result.fill_weights
    idx = fills.index
    aligned_prices = prices.reindex(index=result.target_weights.index, columns=fills.columns).ffill()
    open_ret = aligned_prices.shift(-1) / aligned_prices - 1.0
    open_ret = open_ret.reindex(idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = (fills * open_ret).sum(axis=1)
    cost_total = pd.to_numeric(result.costs["total"], errors="coerce").reindex(idx).fillna(0.0)
    return gross - cost_total


class TestLedgerConservation:
    def test_cost_total_is_sum_of_components(self):
        prices, tw, dvol, _, _ = _panel()
        exe = ExecutionConfig(adv_impact_coeff=5.0, adv_impact_model="sqrt", adv_floor_dollars=1.0)
        res = backtest_target_weights(prices, tw, execution=exe, initial_capital=1e6, dollar_volume=dvol)
        c = res.costs
        recomposed = c["commission_spread"] + c["impact"] + c["borrow"]
        pd.testing.assert_series_equal(c["total"], recomposed, check_names=False)

    def test_equity_is_cumprod_of_returns(self):
        prices, tw, _, _, _ = _panel()
        res = backtest_target_weights(prices, tw, initial_capital=10_000.0)
        rebuilt = (1.0 + res.returns).cumprod() * 10_000.0
        pd.testing.assert_series_equal(res.equity, rebuilt, check_names=False)

    def test_ledger_closes_no_dividends(self):
        # No unexplained cash term: net return == fills·open_ret − costs.
        prices, tw, _, _, _ = _panel(seed=1)
        exe = ExecutionConfig(borrow_rate_bps_annual=200.0)
        res = backtest_target_weights(prices, tw, execution=exe, initial_capital=1.0)
        rebuilt = _reconstruct_returns(res, prices)
        pd.testing.assert_series_equal(res.returns, rebuilt, check_names=False, rtol=1e-9, atol=1e-12)

    def test_zero_cost_conserves_pure_pnl(self):
        # With no costs and no dividends, net return is exactly position·open_ret.
        prices, tw, _, _, _ = _panel(seed=2)
        exe = ExecutionConfig(
            commission_bps=0.0, spread_bps=0.0, slippage_coeff=0.0, borrow_rate_bps_annual=0.0
        )
        res = backtest_target_weights(prices, tw, execution=exe, initial_capital=1.0)
        assert res.costs["total"].abs().max() == pytest.approx(0.0)
        rebuilt = _reconstruct_returns(res, prices)
        pd.testing.assert_series_equal(res.returns, rebuilt, check_names=False, rtol=1e-9, atol=1e-12)

    def test_flat_book_has_zero_return_and_flat_equity(self):
        prices, _, _, dates, syms = _panel(seed=3)
        flat = pd.DataFrame(0.0, index=dates, columns=syms)
        res = backtest_target_weights(prices, flat, initial_capital=1234.0)
        assert res.returns.abs().max() == pytest.approx(0.0)
        assert res.equity.round(6).nunique() == 1  # equity never moves off initial capital
