"""Unit tests for the IEX-eligibility pre-flight's pure helpers (research tier)."""

import pandas as pd
import pytest

pytestmark = pytest.mark.research

from prism.residual.factors import (  # noqa: E402
    ResidualStatArbConfig,
    compute_eligibility,
)
from research.scripts.iex_eligibility_check import (  # noqa: E402
    SCREEN_FLOOR_DOLLARS,
    median_dollar_volume,
    screen_comparison,
    share_stats,
    universe_symbols,
    volume_share,
)


def make_bars(index: pd.DatetimeIndex, close, volume) -> pd.DataFrame:
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=index)


def med_series(values: dict[str, float]) -> pd.Series:
    """A warm rolling-median series stub: date-string keys -> median values."""
    ser = pd.Series(values)
    ser.index = pd.DatetimeIndex(pd.to_datetime(ser.index))
    return ser


def test_universe_symbols_strips_comments_and_blanks() -> None:
    text = "AAPL\n# a comment line\nMSFT  # trailing note\n\n  NVDA \n"
    assert universe_symbols(text) == ["AAPL", "MSFT", "NVDA"]


def test_median_dollar_volume_matches_compute_eligibility_volume_leg() -> None:
    # The helper must reproduce the screen's volume leg bit-for-bit: on a panel
    # where price and history legs are satisfied, compute_eligibility's mask
    # equals thresholding the helper's series at the floor.
    idx = pd.bdate_range("2024-01-02", periods=60)
    volumes = pd.DataFrame(
        {"BIG": [100_000.0 + 1_000.0 * i for i in range(60)], "THIN": [10_000.0] * 60}, index=idx
    )
    closes = pd.DataFrame({"BIG": 50.0, "THIN": 50.0}, index=idx)
    config = ResidualStatArbConfig(corr_window=20, regr_window=20, min_price=0.0)
    eligible = compute_eligibility(closes, volumes, config)
    for sym in ("BIG", "THIN"):
        med = median_dollar_volume(make_bars(idx, closes[sym], volumes[sym]))
        assert len(med) == 60 - 20 + 1  # warm-up head dropped, min_periods == window
        # full_history binds through bar 20 (pct_change eats bar 0), so compare after it.
        for t in idx[20:]:
            assert eligible.loc[t, sym] == (med.loc[t] >= config.min_median_dollar_volume)


def test_median_dollar_volume_dedups_keep_last_and_clips_negative_volume() -> None:
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=6, tz="America/New_York"))
    doubled = idx.append(idx[-1:])  # duplicated final date, tz-aware like a real cache
    bars = make_bars(doubled, 10.0, [1_000.0] * 5 + [9_999_999.0, 2_000.0])
    bars.iloc[0, bars.columns.get_loc("volume")] = -5.0  # clipped to 0, not a negative dollar volume
    med = median_dollar_volume(bars, window=6)
    assert len(med) == 1
    assert med.index.tz is None
    # keep-last: the duplicate date contributes 2_000 shares, not 9_999_999.
    assert med.iloc[0] == pytest.approx(pd.Series([0.0, 1e4, 1e4, 1e4, 1e4, 2e4]).median())


def test_volume_share_median_and_min_overlap() -> None:
    idx = pd.bdate_range("2024-01-02", periods=12)
    cons = make_bars(idx, 100.0, 1_000_000.0)
    iex = make_bars(idx, 100.0, 50_000.0)
    assert volume_share(iex, cons, min_overlap=10) == pytest.approx(0.05)
    # A zero consolidated print is excluded from the ratio, not divided by.
    cons_zero = cons.copy()
    cons_zero.iloc[0, cons_zero.columns.get_loc("volume")] = 0.0
    assert volume_share(iex, cons_zero, min_overlap=10) == pytest.approx(0.05)
    # Under the overlap minimum the share is NaN, not a noisy estimate.
    assert volume_share(iex.iloc[:5], cons, min_overlap=10) != volume_share(iex.iloc[:5], cons, min_overlap=10)


def test_screen_comparison_masks_lag_tolerance_and_flap_risk() -> None:
    floor = SCREEN_FLOOR_DOLLARS
    cons = {
        # Passes consolidated; its IEX series lags the cache end by one session.
        "ACTX": med_series({"2026-06-15": 12e6, "2026-06-16": 12e6}),
        "SAFE": med_series({"2026-06-15": 40e6, "2026-06-16": 40e6}),
        "NOIEX": med_series({"2026-06-16": 5e6}),
    }
    iex = {
        "ACTX": med_series({"2026-06-15": 0.74e6}),
        "SAFE": med_series({"2026-06-15": 2.5e6, "2026-06-16": 2.5e6, "2026-07-01": 1.5e6}),
        "NEW": med_series({"2026-07-01": 0.9e6}),  # no consolidated cache at all
    }
    result = screen_comparison(cons, iex, floor=floor)
    summary = result["summary"]
    assert summary["n_names"] == 4
    assert summary["n_with_both_feeds_asof"] == 2
    assert summary["fail_cons_asof"] == []
    assert summary["fail_iex_asof"] == ["ACTX"]  # read at-or-before the cache end, tolerating the lag
    assert summary["mask_symmetric_difference_asof"] == ["ACTX"]
    assert summary["fail_iex_last"] == ["ACTX", "NEW"]
    assert summary["near_floor_iex_last"] == ["SAFE"]
    actx = result["per_name"]["ACTX"]
    assert actx["asof"] == "2026-06-16"
    assert actx["iex_med_asof"] == pytest.approx(0.74e6)
    assert "iex_med_asof" not in result["per_name"]["NOIEX"]


def test_share_stats_reports_distribution_and_implied_floor() -> None:
    stats = share_stats({"A": 0.04, "B": 0.05, "C": 0.06, "D": float("nan")}, floor=1e6)
    assert stats["n"] == 3
    assert stats["median"] == pytest.approx(0.05)
    assert stats["implied_consolidated_floor_at_median"] == pytest.approx(2e7)
    assert share_stats({}, floor=1e6) == {"n": 0}
