"""Unit tests for the data-integrity sweep's pure helpers (research tier)."""

import pandas as pd
import pytest

pytestmark = pytest.mark.research

from research.scripts.data_integrity_sweep import (  # noqa: E402
    NYSE_FULL_CLOSURES,
    cache_metrics,
    classify,
    run_exposure,
    screen_max_within,
)


def us_sessions(start: str, end: str) -> pd.DatetimeIndex:
    """Weekdays minus the hardcoded NYSE closures, tz-aware ET like the caches."""
    days = pd.bdate_range(start, end, tz="America/New_York")
    return pd.DatetimeIndex([d for d in days if d.strftime("%Y-%m-%d") not in NYSE_FULL_CLOSURES])


def make_bars(index: pd.DatetimeIndex, close: float = 100.0, volume: float = 1_000_000.0) -> pd.DataFrame:
    data = {"open": close, "high": close, "low": close, "close": close, "volume": volume}
    return pd.DataFrame(data, index=index)


def test_clean_us_cache_has_no_flags() -> None:
    bars = make_bars(us_sessions("2024-01-02", "2024-12-31"))
    metrics = cache_metrics(bars)
    assert metrics["duplicate_dates"] == 0
    assert metrics["n_closure_bars"] == 0
    assert metrics["missing_sessions"] == 0
    assert classify(metrics, min_last_date="2024-12-01") == []


def test_foreign_calendar_and_thin_volume_flagged() -> None:
    # A plain business-day index includes the NYSE closures — the signature of
    # a foreign-venue series — and a foreign local line prints thin volume.
    bars = make_bars(pd.bdate_range("2024-01-01", "2024-12-31", tz="America/New_York"), volume=3_000.0)
    metrics = cache_metrics(bars)
    assert metrics["n_closure_bars"] >= 3
    flags = classify(metrics, min_last_date="2024-12-01")
    assert "off_calendar_bars" in flags
    assert "thin_volume" in flags


def test_duplicate_dates_flagged() -> None:
    idx = us_sessions("2024-01-02", "2024-06-28")
    doubled = pd.DatetimeIndex(idx.append(idx[:5]).sort_values())
    metrics = cache_metrics(make_bars(doubled))
    assert metrics["duplicate_dates"] == 5
    assert "duplicate_dates" in classify(metrics, min_last_date="2024-06-01")


def test_gappy_series_flagged() -> None:
    idx = us_sessions("2024-01-02", "2024-12-31")
    sparse = idx.delete(range(40, 80))  # a 40-session hole, far past any plausible halt
    metrics = cache_metrics(make_bars(sparse))
    assert metrics["missing_sessions"] == 40
    assert "gappy_series" in classify(metrics, min_last_date="2024-12-01")


def test_screen_max_respects_membership_and_window() -> None:
    idx = us_sessions("2024-01-02", "2024-12-31")
    bars = make_bars(idx, close=10.0, volume=50_000.0)  # dollar volume 5e5, below the 1e6 floor
    bars.loc[bars.index[-40:], "volume"] = 200_000.0  # last 40 sessions print 2e6
    intervals = [{"start": "2024-01-02", "end": "2024-12-31"}]
    full = screen_max_within(bars, intervals, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31"))
    assert full == pytest.approx(2_000_000.0)
    # Membership that ends before the volume ramp never sees the passable regime.
    early = screen_max_within(
        bars, [{"start": "2024-01-02", "end": "2024-06-28"}], pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31")
    )
    assert early == pytest.approx(500_000.0)


def test_run_exposure_reports_held_days_and_contribution(tmp_path) -> None:
    dates = ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"]
    weights = pd.DataFrame(
        {"XYZ": [0.0, 0.5, 0.5, 0.5, 0.0], "OTHER": [0.1, 0.1, 0.1, 0.1, 0.1]},
        index=[f"{d} 00:00:00-05:00" for d in dates],
    )
    weights.to_csv(tmp_path / "target_weights.csv")
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize("America/New_York")
    bars = make_bars(idx)
    bars["close"] = [100.0, 101.0, 102.0, 103.0, 104.0]
    exposure, lo, hi = run_exposure(tmp_path, {"XYZ": bars, "ABSENT": bars})
    assert exposure["ABSENT"] == {"in_universe": False}
    xyz = exposure["XYZ"]
    assert xyz["days_held"] == 3
    assert xyz["max_abs_weight"] == pytest.approx(0.5)
    assert xyz["first_held"] == "2024-03-05" and xyz["last_held"] == "2024-03-07"
    # w[t-1] * r[t]: three nonzero contribution days (03-06, 03-07, 03-08), each ~0.5 * ~1%.
    assert xyz["return_contribution_days"] == 3
    assert xyz["return_contribution_total"] == pytest.approx(
        0.5 * (102 / 101 - 1) + 0.5 * (103 / 102 - 1) + 0.5 * (104 / 103 - 1)
    )
    assert str(lo.date()) == "2024-03-04" and str(hi.date()) == "2024-03-08"
