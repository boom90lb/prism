"""The trailing dollar-volume statistic reads only its last window (prism.evaluate._trailing_dollar_volume).

The reference below is the full-history computation the function used before it sliced
to its last ``window`` rows, restated by value: the median must match it exactly, the
mean to rounding (pandas' running sum rounds differently over a longer history).
"""

import numpy as np
import pandas as pd
import pytest

from prism.evaluate import _trailing_dollar_volume


def _reference(close, volume, window, statistic):
    rolling = (close * volume).rolling(window, min_periods=window)
    return (rolling.median() if statistic == "median" else rolling.mean()).iloc[-1]


@pytest.fixture
def panel():
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2021-01-04", periods=400)
    close = pd.DataFrame(rng.lognormal(4, 0.3, (400, 6)), index=index, columns=list("ABCDEF"))
    volume = pd.DataFrame(rng.lognormal(14, 0.5, (400, 6)), index=index, columns=list("ABCDEF"))
    close.iloc[395, 1] = np.nan  # a gap inside the last window: no statistic for B
    close.iloc[:390, 2] = np.nan  # listed 10 bars ago: no full window for C
    close.iloc[100, 3] = np.nan  # a gap long before the window changes nothing for D
    return close, volume


@pytest.mark.parametrize("window", [1, 20, 60])
def test_the_median_equals_the_full_history_computation_exactly(panel, window):
    close, volume = panel
    got = _trailing_dollar_volume(close, volume, window=window, statistic="median")
    pd.testing.assert_series_equal(got, _reference(close, volume, window, "median"), check_exact=True)


def test_the_mean_equals_it_to_rounding_and_the_missing_names_stay_missing(panel):
    close, volume = panel
    got = _trailing_dollar_volume(close, volume, window=20, statistic="mean")
    pd.testing.assert_series_equal(got, _reference(close, volume, 20, "mean"), rtol=1e-12)
    assert np.isnan(got["B"]) and np.isnan(got["C"]) and np.isfinite(got["D"])


def test_a_history_shorter_than_the_window_has_no_statistic(panel):
    close, volume = panel
    got = _trailing_dollar_volume(close.iloc[:5], volume.iloc[:5], window=20, statistic="median")
    assert got.isna().all()
