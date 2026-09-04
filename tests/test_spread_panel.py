"""Day x symbol spread accounting (``backtest_target_weights(spread_bps_per_name=DataFrame)``)
and its use by the trend_v1 core: each fold's SPREAD_BUCKET_SCHEDULE_V1 buckets price
the fills inside that fold's test window (``spread_accounting="per_fold_bucket"``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.config import ExecutionConfig
from prism.execution.target_weights import backtest_target_weights
from research.arbitrage.residual_walk_forward import bucket_spread_bps
from research.arbitrage.trend_walk_forward import TrendSleeveConfig, run_trend_walk_forward
from research.arbitrage.walk_forward import StatArbWalkForwardConfig, iter_walk_forward_slices


def _two_trade_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.bdate_range("2024-01-01", periods=12, tz="America/New_York")
    opens = pd.DataFrame({"A": np.linspace(100.0, 111.0, 12), "B": np.linspace(50.0, 61.0, 12)}, index=idx)
    # NaN rows are no-op markers (hold); an all-zero row would be an explicit exit.
    targets = pd.DataFrame(np.nan, index=idx, columns=["A", "B"])
    targets.iloc[0] = [0.5, -0.5]  # decided at close 0, fills at open 1
    targets.iloc[6] = [0.2, 0.3]  # decided at close 6, fills at open 7
    return opens, targets


def test_constant_frame_is_bit_identical_to_series() -> None:
    opens, targets = _two_trade_panel()
    ex = ExecutionConfig()
    series = pd.Series({"A": 3.0, "B": 7.0})
    frame = pd.DataFrame([[3.0, 7.0]] * len(opens), index=opens.index, columns=["A", "B"])
    a = backtest_target_weights(opens, targets, execution=ex, spread_bps_per_name=series)
    b = backtest_target_weights(opens, targets, execution=ex, spread_bps_per_name=frame)
    pd.testing.assert_series_equal(a.returns, b.returns)
    pd.testing.assert_frame_equal(a.costs, b.costs)
    pd.testing.assert_frame_equal(a.fill_weights, b.fill_weights)


def test_time_varying_frame_prices_each_fill_day() -> None:
    opens, targets = _two_trade_panel()
    ex = ExecutionConfig()
    frame = pd.DataFrame(10.0, index=opens.index, columns=["A", "B"])
    frame.iloc[7:] = 3.0  # re-priced from the FILL row 7 onward; decision row 6 stays 10.0 so the fill-day row is what is pinned
    res = backtest_target_weights(opens, targets, execution=ex, spread_bps_per_name=frame)
    cs = res.costs["commission_spread"]
    # fill at open 1: |0.5| + |-0.5| = 1.0 traded at (commission 1 + spread 10) bps
    assert np.isclose(cs.iloc[1], 1.0 * (ex.commission_bps + 10.0) / 1e4)
    # fill at open 7: |0.2-0.5| + |0.3+0.5| = 1.1 traded at (1 + 3) bps
    assert np.isclose(cs.iloc[7], 1.1 * (ex.commission_bps + 3.0) / 1e4)
    # rows with no trade carry no commission_spread
    assert cs.iloc[2:7].abs().max() == 0.0
    # a missing (NaN) day falls back to the flat configured spread, never to zero
    frame_nan = frame.copy()
    frame_nan.iloc[7] = np.nan
    res_nan = backtest_target_weights(opens, targets, execution=ex, spread_bps_per_name=frame_nan)
    assert np.isclose(res_nan.costs["commission_spread"].iloc[7], 1.1 * (ex.commission_bps + ex.spread_bps) / 1e4)
    # a name absent from the frame falls back to the flat spread for that name only
    frame_a = frame[["A"]]
    res_a = backtest_target_weights(opens, targets, execution=ex, spread_bps_per_name=frame_a)
    assert np.isclose(
        res_a.costs["commission_spread"].iloc[7], (0.3 * (ex.commission_bps + 3.0) + 0.8 * (ex.commission_bps + ex.spread_bps)) / 1e4
    )


def _synthetic_trend_panel(n: int, symbols: list[str], *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n, tz="America/New_York")
    rets = rng.normal(0.0004, 0.01, size=(n, len(symbols)))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=symbols)
    open_ = close.shift(1).bfill() * (1.0 + rng.normal(0.0, 0.001, size=close.shape))
    # dollar volume ~ $2M/day for the first 250 sessions (10 bps bucket), ~$1B/day after (1 bps bucket)
    vol = pd.DataFrame(2.0e4, index=idx, columns=symbols)
    vol.iloc[250:] = 1.0e7
    return close, open_, vol


def test_core_prices_each_fold_with_its_own_buckets() -> None:
    symbols = ["A", "B", "C"]
    close, open_, volume = _synthetic_trend_panel(700, symbols, seed=11)
    walk = StatArbWalkForwardConfig(
        formation_bars=126,
        test_bars=63,
        min_test_bars=20,
        max_gross=1.0,
        max_symbol_abs_weight=1.0,
        no_trade_band=0.0,
        band_mode="fixed",
        spread_mode="bucket",
        max_participation=0.0,
    )
    sleeve = TrendSleeveConfig(price_convention="price_return", lookback_bars=60, skip_bars=5, decision_every=21, entry_bars=60)
    ex = ExecutionConfig()
    res = run_trend_walk_forward(close, open_, volume, score_close=close, dividends=None, sleeve=sleeve, walk=walk, execution=ex)
    assert res.spread_bps_frame is not None
    raw_dv = close * volume
    folds = list(iter_walk_forward_slices(len(close), walk))
    assert len(folds) == len(res.folds) >= 3
    seen_buckets: set[float] = set()
    for slices, fold in zip(folds, res.folds):
        expected = bucket_spread_bps(raw_dv.iloc[slices.formation].median(axis=0, skipna=True)).reindex(symbols)
        assert fold.spread_bps == {s: float(expected[s]) for s in symbols}
        test_index = close.index[slices.test]
        frame_rows = res.spread_bps_frame.loc[test_index, symbols]
        assert (frame_rows.to_numpy() == expected.to_numpy()[None, :]).all()
        seen_buckets.update(float(v) for v in expected.to_numpy())
    assert {1.0, 10.0} <= seen_buckets  # the schedule really re-bucketed across folds
    # accounting: every fill row's commission_spread equals sum |trade| * (commission + that day's bucket)
    fills = res.portfolio.fill_weights
    trade = fills.diff().fillna(fills.iloc[[0]])
    for day in trade.index:
        row_trade = trade.loc[day].abs().to_numpy()
        if row_trade.sum() == 0.0:
            continue
        spread_row = res.spread_bps_frame.loc[day, symbols].fillna(ex.spread_bps).to_numpy(dtype=float)
        expected_cs = float((row_trade * (ex.commission_bps + spread_row)).sum() / 1e4)
        assert np.isclose(res.portfolio.costs.loc[day, "commission_spread"], expected_cs, rtol=0, atol=1e-15)
