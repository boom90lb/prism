"""Day x symbol spread accounting: ``backtest_target_weights(spread_bps_per_name=DataFrame)``.

The third test here drove the retired trend core to show that each fold's
``SPREAD_BUCKET_SCHEDULE_V1`` buckets price the fills inside that fold's test
window (``spread_accounting = "per_fold_bucket"``). That property is now
asserted against the recorded record itself, per fold and per bucket, in
``tests/test_trend_t0_reproduction.py``; what remains here is the engine
contract the frame form has to keep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.config import ExecutionConfig
from prism.execution.target_weights import backtest_target_weights


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
    frame.iloc[7:] = (
        3.0  # re-priced from the FILL row 7 onward; decision row 6 stays 10.0 so the fill-day row is what is pinned
    )
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
        res_a.costs["commission_spread"].iloc[7],
        (0.3 * (ex.commission_bps + 3.0) + 0.8 * (ex.commission_bps + ex.spread_bps)) / 1e4,
    )
