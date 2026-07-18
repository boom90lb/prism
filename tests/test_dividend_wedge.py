"""Unit tests for the dividend-wedge accrual arithmetic (research tier)."""

import pandas as pd
import pytest

pytestmark = pytest.mark.research

from research.scripts.dividend_wedge import dividend_panel, wedge_report  # noqa: E402


def frame(index, data):
    return pd.DataFrame(data, index=index)


def test_dividend_panel_maps_ex_dates_and_counts_drops() -> None:
    idx = pd.DatetimeIndex(pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-06"]))
    records = [
        {"ex_date": "2024-03-05", "symbol": "AAA", "rate": 0.5, "special": False, "foreign": False},
        {"ex_date": "2024-03-05", "symbol": "AAA", "rate": 0.25, "special": True, "foreign": False},  # same-day sums
        {"ex_date": "2024-03-09", "symbol": "AAA", "rate": 1.0, "special": False, "foreign": False},  # off calendar
        {"ex_date": "2024-03-05", "symbol": "ZZZ", "rate": 9.0, "special": False, "foreign": True},  # not in universe
    ]
    panel, meta = dividend_panel(records, idx, pd.Index(["AAA", "BBB"]))
    assert panel.loc[pd.Timestamp("2024-03-05"), "AAA"] == pytest.approx(0.75)
    assert panel["BBB"].sum() == 0.0
    assert meta["n_records"] == 4
    assert meta["n_records_in_universe"] == 3
    assert meta["n_ex_dates_off_calendar"] == 1
    assert meta["n_special"] == 1 and meta["n_foreign"] == 1


def test_wedge_report_accrues_to_prior_day_book_and_splits_legs() -> None:
    idx = pd.DatetimeIndex(pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"]))
    # Long AAA, short BBB; both pay $1 on 2024-03-06 with prior close 100.
    weights = frame(idx, {"AAA": [0.5, 0.5, 0.5, 0.5], "BBB": [-0.25, -0.25, -0.25, -0.25]})
    closes = frame(idx, {"AAA": [100.0, 100.0, 101.0, 102.0], "BBB": [100.0, 100.0, 99.0, 98.0]})
    dividends = frame(idx, {"AAA": [0.0, 0.0, 1.0, 0.0], "BBB": [0.0, 0.0, 1.0, 0.0]})
    report = wedge_report(weights, closes, dividends)
    # Net flow on the ex-date: 0.5*1/100 - 0.25*1/100 = 0.0025 of NAV.
    assert report["total_wedge_return"] == pytest.approx(0.0025)
    assert report["long_leg"]["flow_annualized_bps"] > 0.0
    assert report["short_leg"]["flow_annualized_bps"] < 0.0
    # Short-leg portfolio yield is reported as a positive yield of the leg.
    assert report["short_leg"]["portfolio_yield_pct"] > 0.0
    assert report["long_leg"]["avg_gross"] == pytest.approx(0.5)
    assert report["short_leg"]["avg_gross"] == pytest.approx(0.25)
    assert report["n_ex_date_days_hit"] == 1


def test_wedge_sign_flips_with_anti_yield_tilt() -> None:
    idx = pd.DatetimeIndex(pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-06"]))
    # Short leg yields more than the long leg: price-return ledger flatters the book.
    weights = frame(idx, {"LOW": [0.5, 0.5, 0.5], "HIGH": [-0.5, -0.5, -0.5]})
    closes = frame(idx, {"LOW": [100.0, 100.0, 100.0], "HIGH": [100.0, 100.0, 100.0]})
    dividends = frame(idx, {"LOW": [0.0, 0.0, 0.2], "HIGH": [0.0, 0.0, 1.0]})
    report = wedge_report(weights, closes, dividends)
    assert report["total_wedge_return"] == pytest.approx(0.5 * 0.002 - 0.5 * 0.01)
    assert report["annualized_wedge_bps"] < 0.0
