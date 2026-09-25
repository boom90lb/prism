"""Offline tests for point-in-time earnings announcements (prism.io.earnings_events)."""

import numpy as np
import pandas as pd
import pytest

from prism.io import earnings_events as ee

NY = "America/New_York"


def _sessions(start="2024-01-02", end="2024-03-29"):
    return pd.bdate_range(start, end)


def _releases(rows):
    return pd.DataFrame([{"cik": cik, "accn": accn, "form": form, "acceptance": pd.Timestamp(ts, tz=NY)}
                         for cik, accn, form, ts in rows])


def test_day_zero_is_the_acceptance_date_or_the_next_session():
    sessions = _sessions()
    dates = ee.announcement_dates(_releases([
        (1, "a", "8-K", "2024-01-12 18:05"),  # Friday after the close: day 0 is that Friday
        (2, "b", "8-K", "2024-01-13 09:00"),  # Saturday: day 0 is Monday
        (3, "c", "8-K", "2024-01-16 07:30"),  # before the open
        (3, "d", "8-K/A", "2024-01-17 07:30"),  # amendments are never events
    ]), sessions)
    got = dict(zip(dates["accn"], dates["d0"]))
    assert got == {"a": pd.Timestamp("2024-01-12"), "b": pd.Timestamp("2024-01-15"), "c": pd.Timestamp("2024-01-16")}


def test_a_release_before_the_first_session_is_dropped_and_does_not_shadow_a_later_one():
    sessions = _sessions()
    dates = ee.announcement_dates(_releases([
        (1, "old", "8-K", "2023-10-30 16:05"),  # before the sample: not mapped onto its first session
        (1, "early", "8-K", "2024-01-04 16:05"),  # 2 sessions into the sample: kept
    ]), sessions)
    assert list(dates["accn"]) == ["early"]


def test_a_second_release_within_the_gap_belongs_to_the_first():
    sessions = _sessions()
    dates = ee.announcement_dates(_releases([
        (1, "first", "8-K", "2024-01-10 16:05"),
        (1, "supplement", "8-K", "2024-01-12 08:00"),  # 2 sessions later: dropped
        (1, "next", "8-K", "2024-01-17 16:05"),  # 5 sessions after the kept one: kept
    ]), sessions)
    assert list(dates["accn"]) == ["first", "next"]


def test_announcement_return_sums_the_four_session_window_of_excess_returns():
    stock = np.array([0.0, 0.01, 0.02, 0.05, -0.01, 0.03, 0.0])
    market = np.array([0.0, 0.005, 0.0, 0.01, 0.0, 0.0, 0.0])
    got = ee.announcement_returns(np.array([3, 1, 6]), stock, market)
    assert got[0] == pytest.approx((0.01 - 0.005) + 0.02 + (0.05 - 0.01) + (-0.01))  # sessions 1..4
    assert np.isnan(got[1]) and np.isnan(got[2])  # window leaves the sample on either side
    gapped = stock.copy()
    gapped[2] = np.nan
    assert np.isnan(ee.announcement_returns(np.array([3]), gapped, market)[0])


def test_market_return_is_the_mean_over_members_only(monkeypatch):
    monkeypatch.setattr(ee, "MIN_MARKET_NAMES", 2)
    index = pd.bdate_range("2024-01-02", periods=2)
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, np.nan], "C": [0.5, 0.5]}, index=index)
    members = pd.DataFrame({"A": [True, True], "B": [True, True], "C": [False, False]}, index=index)
    market = ee.market_returns(returns, members)
    assert market.iloc[0] == pytest.approx(0.02)
    assert np.isnan(market.iloc[1])  # one member reporting is below the minimum


def _panel_inputs(stock):
    sessions = _sessions("2024-01-02", "2024-01-31")
    dates = ee.announcement_dates(_releases([(9, "q4", "8-K", "2024-01-10 16:05")]), sessions)
    intervals = pd.DataFrame({"start": [pd.Timestamp("2020-01-01")], "end": [pd.Timestamp("2026-01-01")],
                              "cik": [9]})
    market = np.zeros(len(sessions))
    events = ee.security_events("XYZ", intervals, {9: dates}, sessions, stock, market)
    bars = pd.DataFrame({"ts": sessions.tz_localize(NY), "close": 10.0, "volume": 1000.0})
    return sessions, events, ee.security_panel("XYZ", bars, intervals, events, {}, sessions)


def test_the_value_is_first_live_at_the_close_of_day_plus_one():
    sessions = _sessions("2024-01-02", "2024-01-31")
    d0 = sessions.get_loc(pd.Timestamp("2024-01-10"))
    stock = np.zeros(len(sessions))
    stock[d0 - 3:d0 + 3] = [0.3, 0.01, 0.01, 0.04, 0.02, 0.3]  # the window is d0-2 .. d0+1; its neighbours are out
    sessions, events, panel = _panel_inputs(stock)
    assert events.loc[0, "signal_date"] == sessions[d0 + 1]
    assert events.loc[0, "ear"] == pytest.approx(stock[d0 - 2:d0 + 2].sum())
    by_date = panel.set_index("date")
    assert pd.Timestamp("2024-01-10") not in by_date.index  # not yet known at day 0's close
    assert by_date.loc[sessions[d0 + 1], "age_sessions"] == 0
    assert by_date.loc[sessions[d0 + 4], "age_sessions"] == 3


def test_returns_after_the_signal_date_never_change_the_value():
    base = np.random.default_rng(3).normal(0, 0.01, len(_sessions("2024-01-02", "2024-01-31")))
    _, _, panel = _panel_inputs(base)
    changed = base.copy()
    sessions = _sessions("2024-01-02", "2024-01-31")
    d0 = sessions.get_loc(pd.Timestamp("2024-01-10"))
    changed[d0 + 2:] += 0.5
    _, _, panel_changed = _panel_inputs(changed)
    pd.testing.assert_series_equal(panel["ear"], panel_changed["ear"])


def test_an_announcement_outside_the_mapping_interval_is_not_the_securitys():
    sessions = _sessions()
    dates = ee.announcement_dates(_releases([(9, "q4", "8-K", "2024-01-10 16:05")]), sessions)
    intervals = pd.DataFrame({"start": [pd.Timestamp("2024-02-01")], "end": [pd.Timestamp("2026-01-01")], "cik": [9]})
    events = ee.security_events("XYZ", intervals, {9: dates}, sessions, np.zeros(len(sessions)),
                                np.zeros(len(sessions)))
    assert events.empty


def _history(rows):
    return ee.company_history(_releases([(5, f"r{i}", "8-K", ts) for i, ts in enumerate(rows)]))


def test_the_expected_date_is_the_unrealized_anniversary_of_last_years_quarter():
    history = _history(["2023-04-27 16:05", "2023-07-27 16:05", "2023-10-26 16:05", "2024-01-30 16:05"])
    dates = pd.DatetimeIndex(["2024-04-19", "2024-05-01"])
    got = ee.days_to_expected(history, dates)
    assert got[0] == pytest.approx(6.0)  # 2023-04-27 + 364 days = 2024-04-25
    # 2024-04-25 passed unannounced by more than five days: the next anniversary, 2024-07-25
    assert got[1] == pytest.approx((pd.Timestamp("2024-07-25") - pd.Timestamp("2024-05-01")).days)


def test_an_announcement_realizes_its_anniversary_only_once_it_is_accepted():
    history = _history(["2023-04-27 16:05", "2023-07-27 16:05", "2024-04-24 16:05"])
    before, same_day, after = ee.days_to_expected(history, pd.DatetimeIndex(["2024-04-23", "2024-04-24",
                                                                             "2024-04-25"]))
    assert before == pytest.approx(2.0)
    assert same_day == pytest.approx(1.0)  # accepted at 16:05, after that session's close: not yet known
    assert after == pytest.approx((pd.Timestamp("2024-07-25") - pd.Timestamp("2024-04-25")).days)


def test_a_slightly_late_announcer_ranks_as_overdue_and_no_history_means_no_value():
    history = _history(["2023-04-27 16:05"])
    assert ee.days_to_expected(history, pd.DatetimeIndex(["2024-04-28"]))[0] == pytest.approx(-3.0)
    assert np.isnan(ee.days_to_expected(history.iloc[:0], pd.DatetimeIndex(["2024-04-28"]))[0])
