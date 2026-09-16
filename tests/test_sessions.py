"""The bundled exchange session table (engine spec §3.11, stage 0).

The table is derived from the exchange's rules, so the test's job is to pin
the derived rows against dates written down independently here — the NYSE and
Nasdaq published 2020 through 2027 holiday and early-close calendars — and to
attack the three places rule-derivation goes wrong: the direction of a
weekend observance, a Christmas Eve that is itself the Christmas closure, and
a Saturday New Year's leaking a closure back into the previous December.

The 2020-2025 half of the table also has an external check the later years
cannot have: ``test_the_recorded_momentum_panel_agrees_with_the_table``
compares every session in 2020-01-02..2026-06-15 against the trading days the
recorded B1 evaluation actually ran on, read from
``tests/fixtures/folds/demotion_b1_sessions.json``. Two independent
derivations — the exchange's published rules and a 574-name vendor panel —
have to name the same 1,621 days.

Stage 3b widened the table to 2006 (decision D12) and brought the strongest
check in the file with it. SPY has traded every US equity session since 1993,
so over any span its bar dates *are* the calendar. ``_SPY_SESSION_SOURCES``
reads those dates from two independent recordings — the SPY column of the
recorded trend ledger, 2006-09-27..2026-07-17, and the cached SPY daily bars,
2018-01-02..2026-07-21 — and compares each against the table in both
directions. A date the table calls a session that SPY did not trade is a
missing closure; a date SPY traded that the table calls a closure is a false
holiday. Neither is a tolerance. Both recordings live outside the repository
(vendor bars are not redistributable), so those tests skip when absent, and
the transcribed literal lists below stay the check that always runs.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest

from prism.io.sessions import (
    COVERAGE_END,
    COVERAGE_START,
    EARLY_CLOSE,
    JUNETEENTH_FIRST_OBSERVED,
    REGULAR_CLOSE,
    NotASessionError,
    SessionTableRangeError,
    coverage,
    coverage_end,
    is_early_close,
    is_session,
    next_session,
    previous_session,
    session_close,
    session_row,
    sessions_between,
    status,
)


def _d(iso: str) -> dt.date:
    return dt.date.fromisoformat(iso)


def _dates(*iso: str) -> set[dt.date]:
    return {_d(s) for s in iso}


def _statuses(year: int, wanted: str) -> set[dt.date]:
    """Every covered date in ``year`` whose table status is ``wanted``."""
    found = set()
    date = dt.date(year, 1, 1)
    while date.year == year:
        if status(date) == wanted:
            found.add(date)
        date += dt.timedelta(days=1)
    return found


# Transcribed from the published NYSE/Nasdaq calendars, not computed.
HOLIDAYS_2026 = _dates(
    "2026-01-01",  # New Year's Day, Thursday
    "2026-01-19",  # Martin Luther King Jr. Day
    "2026-02-16",  # Washington's Birthday
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth, Friday
    "2026-07-03",  # Independence Day observed (July 4 is a Saturday)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving Day
    "2026-12-25",  # Christmas Day, Friday
)

HOLIDAYS_2027 = _dates(
    "2027-01-01",  # New Year's Day, Friday
    "2027-01-18",  # Martin Luther King Jr. Day
    "2027-02-15",  # Washington's Birthday
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial Day
    "2027-06-18",  # Juneteenth observed (June 19 is a Saturday)
    "2027-07-05",  # Independence Day observed (July 4 is a Sunday)
    "2027-09-06",  # Labor Day
    "2027-11-25",  # Thanksgiving Day
    "2027-12-24",  # Christmas Day observed (December 25 is a Saturday)
)

EARLY_CLOSES = _dates(
    "2026-11-27",  # day after Thanksgiving
    "2026-12-24",  # Christmas Eve, a Thursday and not itself the closure
    "2027-11-26",  # day after Thanksgiving
)

# Easter Sunday, written down independently; Good Friday is two days earlier.
EASTER = {2026: _d("2026-04-05"), 2027: _d("2027-03-28")}


@pytest.mark.parametrize("year,expected", [(2026, HOLIDAYS_2026), (2027, HOLIDAYS_2027)])
def test_holidays_match_the_published_calendar(year: int, expected: set[dt.date]) -> None:
    assert _statuses(year, "holiday") == expected
    for holiday in expected:
        assert is_session(holiday) is False
        assert session_row(holiday).close_time is None


@pytest.mark.parametrize("year", [2026, 2027])
def test_good_friday_is_two_days_before_easter(year: int) -> None:
    good_friday = EASTER[year] - dt.timedelta(days=2)
    assert good_friday.weekday() == 4
    assert status(good_friday) == "holiday"
    assert session_row(good_friday).name == "Good Friday"


def test_saturday_holiday_is_observed_the_preceding_friday() -> None:
    assert _d("2026-07-04").weekday() == 5
    assert status(_d("2026-07-04")) == "weekend"
    assert status(_d("2026-07-03")) == "holiday"
    assert session_row(_d("2026-07-03")).name == "Independence Day"

    assert _d("2027-06-19").weekday() == 5
    assert status(_d("2027-06-18")) == "holiday"


def test_sunday_holiday_is_observed_the_following_monday() -> None:
    assert _d("2027-07-04").weekday() == 6
    assert status(_d("2027-07-05")) == "holiday"
    assert session_row(_d("2027-07-05")).name == "Independence Day"
    # The Friday before stays a full session: the observance moved forward only.
    assert status(_d("2027-07-02")) == "open"


def test_new_years_saturday_does_not_close_the_previous_december() -> None:
    """January 1, 2028 is a Saturday and the exchange trades December 31, 2027."""
    assert _d("2028-01-01").weekday() == 5
    assert status(_d("2027-12-31")) == "open"
    assert session_close(_d("2027-12-31")) == REGULAR_CLOSE


def test_early_closes_are_exactly_the_published_set() -> None:
    found = _statuses(2026, "early_close") | _statuses(2027, "early_close")
    assert found == EARLY_CLOSES
    for date in EARLY_CLOSES:
        assert is_session(date) is True
        assert is_early_close(date) is True
        assert session_close(date) == EARLY_CLOSE


def test_christmas_eve_2027_is_the_holiday_not_an_early_close() -> None:
    """December 24, 2027 is the observed Christmas closure, so it cannot also be a 1:00 pm session."""
    assert status(_d("2027-12-24")) == "holiday"
    assert _d("2027-12-24") not in EARLY_CLOSES


def test_full_sessions_close_at_four() -> None:
    assert session_close(_d("2026-08-17")) == REGULAR_CLOSE
    assert is_early_close(_d("2026-08-17")) is False


def test_the_missed_gap_is_fifteen_full_sessions_and_labor_day_is_not_one() -> None:
    """The stage 0 assessment counts exactly this: 2026-08-17..2026-09-04 missed, 2026-09-07 a holiday."""
    gap = sessions_between(_d("2026-08-17"), _d("2026-09-04"))
    assert len(gap) == 15
    assert gap[0] == _d("2026-08-17")
    assert gap[-1] == _d("2026-09-04")
    assert all(status(d) == "open" for d in gap)

    assert status(_d("2026-09-07")) == "holiday"
    assert session_row(_d("2026-09-07")).name == "Labor Day"
    assert is_session(_d("2026-09-07")) is False
    assert _d("2026-09-07") not in sessions_between(_d("2026-08-17"), _d("2026-09-08"))


def test_sessions_between_is_closed_on_both_ends_and_ordered() -> None:
    single = sessions_between(_d("2026-08-17"), _d("2026-08-17"))
    assert single == [_d("2026-08-17")]
    # A holiday endpoint contributes nothing but is a legal bound.
    assert sessions_between(_d("2026-09-07"), _d("2026-09-07")) == []
    assert sessions_between(_d("2026-09-05"), _d("2026-09-08")) == [_d("2026-09-08")]
    with pytest.raises(ValueError, match="is after end"):
        sessions_between(_d("2026-09-08"), _d("2026-09-07"))


def test_neighbouring_sessions_skip_weekends_and_holidays() -> None:
    assert next_session(_d("2026-09-04")) == _d("2026-09-08")  # Fri -> past Labor Day
    assert previous_session(_d("2026-09-08")) == _d("2026-09-04")
    assert next_session(_d("2026-09-07")) == _d("2026-09-08")  # from the holiday itself
    assert previous_session(_d("2026-09-07")) == _d("2026-09-04")
    # Strictly after / strictly before, never the date itself.
    assert next_session(_d("2026-08-17")) == _d("2026-08-18")
    assert previous_session(_d("2026-08-18")) == _d("2026-08-17")


def test_neighbour_beyond_the_table_raises_rather_than_guessing() -> None:
    with pytest.raises(SessionTableRangeError, match="no session after"):
        next_session(COVERAGE_END)
    with pytest.raises(SessionTableRangeError, match="no session before"):
        previous_session(COVERAGE_START)


@pytest.mark.parametrize("outside", [_d("2005-12-30"), _d("2028-01-01"), _d("2030-06-03")])
def test_dates_outside_coverage_raise_on_every_entry_point(outside: dt.date) -> None:
    for call in (
        lambda: status(outside),
        lambda: is_session(outside),
        lambda: is_early_close(outside),
        lambda: session_close(outside),
        lambda: session_row(outside),
        lambda: next_session(outside),
        lambda: previous_session(outside),
        lambda: sessions_between(outside, _d("2026-06-01")),
        lambda: sessions_between(_d("2026-06-01"), outside),
    ):
        with pytest.raises(SessionTableRangeError):
            call()


def test_coverage_is_reported_so_the_health_check_can_warn_before_the_table_ends() -> None:
    assert coverage() == (COVERAGE_START, COVERAGE_END)
    assert coverage_end() == _d("2027-12-31")
    assert is_session(coverage_end()) is True
    days_left = (coverage_end() - _d("2027-11-01")).days
    assert days_left < 90  # the shape the ninety-day warning consumes


@pytest.mark.parametrize("closed", [_d("2026-09-07"), _d("2026-09-05")])
def test_session_attributes_of_a_closed_day_raise(closed: dt.date) -> None:
    with pytest.raises(NotASessionError):
        is_early_close(closed)
    with pytest.raises(NotASessionError):
        session_close(closed)


def test_weekends_carry_no_row_and_every_weekday_does() -> None:
    date = COVERAGE_START
    weekend_days = 0
    while date <= COVERAGE_END:
        if date.weekday() >= 5:
            weekend_days += 1
            assert status(date) == "weekend"
            with pytest.raises(NotASessionError):
                session_row(date)
        else:
            assert session_row(date).date == date
            assert status(date) in ("open", "early_close", "holiday")
        date += dt.timedelta(days=1)
    # Twenty-two years, 2006-01-01 through 2027-12-31: 8,035 days, of which
    # 2,295 are Saturdays or Sundays.
    assert weekend_days == 2295


@pytest.mark.parametrize("year", [2026, 2027])
def test_each_year_has_251_sessions(year: int) -> None:
    """261 weekdays minus 10 holidays, all of which are observed on weekdays in both years."""
    year_sessions = sessions_between(dt.date(year, 1, 1), dt.date(year, 12, 31))
    assert len(year_sessions) == 251
    assert len(_statuses(year, "holiday")) == 10


# ---------------------------------------------------------------------------
# The 2020-2025 extension (stage 2)
# ---------------------------------------------------------------------------

# Transcribed from the published NYSE calendars, not computed.
HOLIDAYS_2020 = _dates(
    "2020-01-01",
    "2020-01-20",
    "2020-02-17",
    "2020-04-10",
    "2020-05-25",
    "2020-07-03",  # Independence Day observed (July 4 is a Saturday)
    "2020-09-07",
    "2020-11-26",
    "2020-12-25",
)
HOLIDAYS_2021 = _dates(
    "2021-01-01",
    "2021-01-18",
    "2021-02-15",
    "2021-04-02",
    "2021-05-31",
    "2021-07-05",  # Independence Day observed (July 4 is a Sunday)
    "2021-09-06",
    "2021-11-25",
    "2021-12-24",  # Christmas Day observed (December 25 is a Saturday)
)
HOLIDAYS_2022 = _dates(
    "2022-01-17",
    "2022-02-21",
    "2022-04-15",
    "2022-05-30",
    "2022-06-20",  # first observed Juneteenth (June 19 is a Sunday)
    "2022-07-04",
    "2022-09-05",
    "2022-11-24",
    "2022-12-26",  # Christmas Day observed (December 25 is a Sunday)
)  # New Year's Day 2022 is a Saturday and is observed nowhere
HOLIDAYS_2023 = _dates(
    "2023-01-02",  # New Year's Day observed (January 1 is a Sunday)
    "2023-01-16",
    "2023-02-20",
    "2023-04-07",
    "2023-05-29",
    "2023-06-19",
    "2023-07-04",
    "2023-09-04",
    "2023-11-23",
    "2023-12-25",
)
HOLIDAYS_2024 = _dates(
    "2024-01-01",
    "2024-01-15",
    "2024-02-19",
    "2024-03-29",
    "2024-05-27",
    "2024-06-19",
    "2024-07-04",
    "2024-09-02",
    "2024-11-28",
    "2024-12-25",
)
HOLIDAYS_2025 = _dates(
    "2025-01-01",
    "2025-01-09",  # national day of mourning for President Carter (announced)
    "2025-01-20",
    "2025-02-17",
    "2025-04-18",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",
)

EARLY_CLOSES_2020_2025 = _dates(
    "2020-11-27",
    "2020-12-24",
    "2021-11-26",
    "2022-11-25",
    "2023-07-03",
    "2023-11-24",
    "2024-07-03",
    "2024-11-29",
    "2024-12-24",
    "2025-07-03",
    "2025-11-28",
    "2025-12-24",
)


@pytest.mark.parametrize(
    "year,expected",
    [
        (2020, HOLIDAYS_2020),
        (2021, HOLIDAYS_2021),
        (2022, HOLIDAYS_2022),
        (2023, HOLIDAYS_2023),
        (2024, HOLIDAYS_2024),
        (2025, HOLIDAYS_2025),
    ],
)
def test_earlier_years_match_the_published_calendar(year: int, expected: set[dt.date]) -> None:
    assert _statuses(year, "holiday") == expected


def test_juneteenth_is_not_a_closure_before_the_exchange_observed_it() -> None:
    """June 19, 2021 was a Saturday and the exchange traded Friday June 18."""
    assert status(_d("2021-06-18")) == "open"
    assert status(_d("2020-06-19")) == "open"
    assert status(_d("2022-06-20")) == "holiday"


def test_the_carter_mourning_closure_is_carried_because_no_rule_predicts_it() -> None:
    assert status(_d("2025-01-09")) == "holiday"
    assert session_row(_d("2025-01-09")).name == "National Day of Mourning"


def test_early_closes_of_the_earlier_years_are_exactly_the_published_set() -> None:
    found: set[dt.date] = set()
    for year in range(2020, 2026):
        found |= _statuses(year, "early_close")
    assert found == EARLY_CLOSES_2020_2025


def test_the_recorded_momentum_panel_agrees_with_the_table() -> None:
    """The rule-derived table and a 574-name vendor panel name the same sessions.

    ``demotion_b1_sessions.json`` holds the trading days the recorded B1
    evaluation ran on: the union of the panel's dates filtered by
    ``prism.residual.factors.consensus_trading_days``. Nothing about the
    exchange's rules produced them, so an agreement over 1,621 days is an
    external check on the 2020-2025 extension rather than a restatement of it.
    """
    payload = json.loads((Path(__file__).parent / "fixtures" / "folds" / "demotion_b1_sessions.json").read_text())
    recorded = [_d(str(day)[:10]) for day in payload["sessions"]]
    assert len(recorded) == 1621
    derived = sessions_between(recorded[0], recorded[-1])
    assert derived == recorded


# ---------------------------------------------------------------------------
# The 2006-2019 extension (stage 3b, decision D12)
# ---------------------------------------------------------------------------

# Transcribed from the published NYSE calendars, not computed. Each year's
# comment names the reason a date is not the obvious one, because a weekend
# observance in the wrong direction is the failure mode these lists exist for.
HOLIDAYS_2006 = _dates(
    "2006-01-02",  # New Year's Day observed (January 1 is a Sunday)
    "2006-01-16",
    "2006-02-20",
    "2006-04-14",  # Good Friday (Easter April 16)
    "2006-05-29",
    "2006-07-04",
    "2006-09-04",
    "2006-11-23",
    "2006-12-25",
)  # no Juneteenth: June 19, 2006 was an ordinary Monday session
HOLIDAYS_2007 = _dates(
    "2007-01-01",
    "2007-01-02",  # national day of mourning for President Ford (announced)
    "2007-01-15",
    "2007-02-19",
    "2007-04-06",  # Good Friday (Easter April 8)
    "2007-05-28",
    "2007-07-04",
    "2007-09-03",
    "2007-11-22",
    "2007-12-25",
)
HOLIDAYS_2008 = _dates(
    "2008-01-01",
    "2008-01-21",
    "2008-02-18",
    "2008-03-21",  # Good Friday (Easter March 23)
    "2008-05-26",
    "2008-07-04",
    "2008-09-01",
    "2008-11-27",
    "2008-12-25",
)
HOLIDAYS_2009 = _dates(
    "2009-01-01",
    "2009-01-19",
    "2009-02-16",
    "2009-04-10",
    "2009-05-25",
    "2009-07-03",  # Independence Day observed (July 4 is a Saturday)
    "2009-09-07",
    "2009-11-26",
    "2009-12-25",
)
HOLIDAYS_2010 = _dates(
    "2010-01-01",
    "2010-01-18",
    "2010-02-15",
    "2010-04-02",
    "2010-05-31",
    "2010-07-05",  # Independence Day observed (July 4 is a Sunday)
    "2010-09-06",
    "2010-11-25",
    "2010-12-24",  # Christmas Day observed (December 25 is a Saturday)
)
HOLIDAYS_2011 = _dates(
    "2011-01-17",
    "2011-02-21",
    "2011-04-22",
    "2011-05-30",
    "2011-07-04",
    "2011-09-05",
    "2011-11-24",
    "2011-12-26",  # Christmas Day observed (December 25 is a Sunday)
)  # New Year's Day 2011 is a Saturday and is observed nowhere
HOLIDAYS_2012 = _dates(
    "2012-01-02",  # New Year's Day observed (January 1 is a Sunday)
    "2012-01-16",
    "2012-02-20",
    "2012-04-06",
    "2012-05-28",
    "2012-07-04",
    "2012-09-03",
    "2012-10-29",  # Hurricane Sandy
    "2012-10-30",  # Hurricane Sandy
    "2012-11-22",
    "2012-12-25",
)
HOLIDAYS_2013 = _dates(
    "2013-01-01",
    "2013-01-21",
    "2013-02-18",
    "2013-03-29",  # Good Friday (Easter March 31)
    "2013-05-27",
    "2013-07-04",
    "2013-09-02",
    "2013-11-28",
    "2013-12-25",
)
HOLIDAYS_2014 = _dates(
    "2014-01-01",
    "2014-01-20",
    "2014-02-17",
    "2014-04-18",
    "2014-05-26",
    "2014-07-04",
    "2014-09-01",
    "2014-11-27",
    "2014-12-25",
)
HOLIDAYS_2015 = _dates(
    "2015-01-01",
    "2015-01-19",
    "2015-02-16",
    "2015-04-03",
    "2015-05-25",
    "2015-07-03",  # Independence Day observed (July 4 is a Saturday)
    "2015-09-07",
    "2015-11-26",
    "2015-12-25",
)
HOLIDAYS_2016 = _dates(
    "2016-01-01",
    "2016-01-18",
    "2016-02-15",
    "2016-03-25",  # Good Friday (Easter March 27)
    "2016-05-30",
    "2016-07-04",
    "2016-09-05",
    "2016-11-24",
    "2016-12-26",  # Christmas Day observed (December 25 is a Sunday)
)
HOLIDAYS_2017 = _dates(
    "2017-01-02",  # New Year's Day observed (January 1 is a Sunday)
    "2017-01-16",
    "2017-02-20",
    "2017-04-14",
    "2017-05-29",
    "2017-07-04",
    "2017-09-04",
    "2017-11-23",
    "2017-12-25",
)
HOLIDAYS_2018 = _dates(
    "2018-01-01",
    "2018-01-15",
    "2018-02-19",
    "2018-03-30",  # Good Friday (Easter April 1)
    "2018-05-28",
    "2018-07-04",
    "2018-09-03",
    "2018-11-22",
    "2018-12-05",  # national day of mourning for President George H. W. Bush
    "2018-12-25",
)
HOLIDAYS_2019 = _dates(
    "2019-01-01",
    "2019-01-21",
    "2019-02-18",
    "2019-04-19",
    "2019-05-27",
    "2019-07-04",
    "2019-09-02",
    "2019-11-28",
    "2019-12-25",
)

EARLY_CLOSES_2006_2019 = _dates(
    "2006-07-03",
    "2006-11-24",
    # 2006-12-24 is a Sunday: no Christmas Eve early close.
    "2007-07-03",
    "2007-11-23",
    "2007-12-24",
    "2008-07-03",
    "2008-11-28",
    "2008-12-24",
    # 2009-07-03 is the observed Independence Day closure, not an early close.
    "2009-11-27",
    "2009-12-24",
    # 2010: July 4 is a Sunday (observed the 5th) and December 24 is the
    # observed Christmas closure, so only the day after Thanksgiving remains.
    "2010-11-26",
    # 2011: July 3 is a Sunday and December 24 is a Saturday.
    "2011-11-25",
    "2012-07-03",
    "2012-11-23",
    "2012-12-24",
    "2013-07-03",
    "2013-11-29",
    "2013-12-24",
    "2014-07-03",
    "2014-11-28",
    "2014-12-24",
    # 2015-07-03 is the observed Independence Day closure.
    "2015-11-27",
    "2015-12-24",
    # 2016: July 3 is a Sunday and December 24 is a Saturday.
    "2016-11-25",
    "2017-07-03",
    "2017-11-24",
    # 2017-12-24 is a Sunday.
    "2018-07-03",
    "2018-11-23",
    "2018-12-24",
    "2019-07-03",
    "2019-11-29",
    "2019-12-24",
)

# The published NYSE session counts. 2007 lost January 2 to the Ford mourning
# closure, 2012 lost two days to Hurricane Sandy, and 2018 lost December 5 to
# the Bush mourning closure, which is why those years fall below 252.
SESSION_COUNTS_2006_2019 = {
    2006: 251,
    2007: 251,
    2008: 253,
    2009: 252,
    2010: 252,
    2011: 252,
    2012: 250,
    2013: 252,
    2014: 252,
    2015: 252,
    2016: 252,
    2017: 251,
    2018: 251,
    2019: 252,
}


@pytest.mark.parametrize(
    "year,expected",
    [
        (2006, HOLIDAYS_2006),
        (2007, HOLIDAYS_2007),
        (2008, HOLIDAYS_2008),
        (2009, HOLIDAYS_2009),
        (2010, HOLIDAYS_2010),
        (2011, HOLIDAYS_2011),
        (2012, HOLIDAYS_2012),
        (2013, HOLIDAYS_2013),
        (2014, HOLIDAYS_2014),
        (2015, HOLIDAYS_2015),
        (2016, HOLIDAYS_2016),
        (2017, HOLIDAYS_2017),
        (2018, HOLIDAYS_2018),
        (2019, HOLIDAYS_2019),
    ],
)
def test_the_2006_2019_years_match_the_published_calendar(year: int, expected: set[dt.date]) -> None:
    assert _statuses(year, "holiday") == expected


@pytest.mark.parametrize("year,expected", sorted(SESSION_COUNTS_2006_2019.items()))
def test_the_2006_2019_years_have_the_published_session_count(year: int, expected: int) -> None:
    assert len(sessions_between(dt.date(year, 1, 1), dt.date(year, 12, 31))) == expected


def test_early_closes_of_2006_2019_are_exactly_the_published_set() -> None:
    """The three early-close rules held constant across the whole span.

    Every derived early close in 2006-2019 is checked against the published
    calendars here rather than assumed from the fact that the rules produce
    2020-2025 correctly. The list is 32 days; a fourth rule appearing, or one
    of the three lapsing, moves the count.
    """
    found: set[dt.date] = set()
    for year in range(2006, 2020):
        found |= _statuses(year, "early_close")
    assert found == EARLY_CLOSES_2006_2019
    assert len(found) == 32
    for date in found:
        assert is_early_close(date) is True
        assert session_close(date) == EARLY_CLOSE


def test_juneteenth_is_an_ordinary_session_in_every_year_the_exchange_had_not_yet_observed_it() -> None:
    """Sixteen sessions the ten-holiday rule would delete if applied uniformly.

    June 19 falls on a weekday in twelve of 2006-2021 and on a weekend in the
    other four; a table that observed the holiday from the start would close
    the weekday directly in the twelve and its Friday/Monday observance in the
    four. All sixteen days traded. This is the detail most likely to be got
    wrong when a modern calendar is applied to sixteen years of history.
    """
    would_be_closed: list[dt.date] = []
    for year in range(2006, JUNETEENTH_FIRST_OBSERVED):
        june_19 = dt.date(year, 6, 19)
        if june_19.weekday() == 5:  # Saturday, observed the Friday before
            would_be_closed.append(june_19 - dt.timedelta(days=1))
        elif june_19.weekday() == 6:  # Sunday, observed the Monday after
            would_be_closed.append(june_19 + dt.timedelta(days=1))
        else:
            would_be_closed.append(june_19)
    assert len(would_be_closed) == 16
    assert sum(1 for date in would_be_closed if date.month == 6 and date.day == 19) == 12
    for date in would_be_closed:
        assert is_session(date), f"{date.isoformat()} was a real session before the exchange observed Juneteenth"
    # The first observance and every year after it does close.
    assert status(_d("2022-06-20")) == "holiday"
    for year in range(JUNETEENTH_FIRST_OBSERVED, COVERAGE_END.year + 1):
        holidays = _statuses(year, "holiday")
        assert any(
            session_row(date).name.startswith("Juneteenth") for date in holidays
        ), f"{year} carries no Juneteenth closure"


def test_the_earlier_ad_hoc_closures_are_carried_because_no_rule_predicts_them() -> None:
    """Four announced closures in 2006-2019, and the sessions that bracket them."""
    for date, name in (
        ("2007-01-02", "National Day of Mourning"),
        ("2012-10-29", "Hurricane Sandy"),
        ("2012-10-30", "Hurricane Sandy"),
        ("2018-12-05", "National Day of Mourning"),
    ):
        assert status(_d(date)) == "holiday", date
        assert session_row(_d(date)).name == name

    # 2007 opens on January 3: January 1 is the holiday and January 2 the
    # mourning closure, so a table that only knew the rules would open a day early.
    assert sessions_between(_d("2007-01-01"), _d("2007-01-05")) == [
        _d("2007-01-03"),
        _d("2007-01-04"),
        _d("2007-01-05"),
    ]
    # Sandy: Friday the 26th, then nothing until Wednesday the 31st, a full session.
    assert next_session(_d("2012-10-26")) == _d("2012-10-31")
    assert is_early_close(_d("2012-10-31")) is False
    # Bush: Tuesday the 4th to Thursday the 6th.
    assert next_session(_d("2018-12-04")) == _d("2018-12-06")
    assert previous_session(_d("2018-12-06")) == _d("2018-12-04")


# ---------------------------------------------------------------------------
# Two-directional agreement with the market (decision D12)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bar_cache_dir() -> Path | None:
    """Where the vendor bar caches live, or ``None`` when they are not present.

    The caches are vendor data and are not redistributable, so they sit
    outside the repository. ``PRISM_BAR_CACHE`` overrides the search; failing
    that the workspace's own ``data/`` is tried, then the directory the
    recorded paper-replay manifest already names.
    """
    candidates: list[Path] = []
    override = os.environ.get("PRISM_BAR_CACHE")
    if override:
        candidates.append(Path(override))
    candidates.append(REPO_ROOT / "data")
    manifest = Path(__file__).parent / "fixtures" / "paper_replay" / "manifest.json"
    if manifest.is_file():
        store_dir = json.loads(manifest.read_text()).get("source", {}).get("store_dir")
        if store_dir:
            candidates.append(Path(store_dir).parent)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("SPY_1d_2*.parquet")):
            return candidate
    return None


def _spy_dates_from_the_trend_ledger() -> list[dt.date] | None:
    """The dates SPY traded, read from the recorded trend ledger's close panel.

    ``results/trend_t0/closes.csv`` is an immutable recorded artifact. Its SPY
    column is complete over the whole 2006-09-27..2026-07-17 span, so its
    index is the exchange calendar for those years, produced by a vendor panel
    rather than by any rule this table knows.
    """
    ledger = REPO_ROOT / "results" / "trend_t0" / "closes.csv"
    if not ledger.is_file():
        return None
    import pandas as pd

    frame = pd.read_csv(ledger, index_col=0, parse_dates=True)
    if "SPY" not in frame.columns or frame["SPY"].isna().any():
        return None
    return [stamp.date() for stamp in frame.index]


def _spy_dates_from_the_bar_cache() -> list[dt.date] | None:
    """The dates SPY traded, read from the widest cached SPY daily bar file."""
    cache = _bar_cache_dir()
    if cache is None:
        return None
    import pandas as pd

    widest: list[dt.date] | None = None
    for path in sorted(cache.glob("SPY_1d_2*.parquet")):
        frame = pd.read_parquet(path)
        dates = sorted({stamp.date() for stamp in frame.index})
        if widest is None or len(dates) > len(widest):
            widest = dates
    return widest


_SPY_SESSION_SOURCES = {
    "recorded trend ledger": _spy_dates_from_the_trend_ledger,
    "cached SPY daily bars": _spy_dates_from_the_bar_cache,
}


@pytest.mark.parametrize("source", sorted(_SPY_SESSION_SOURCES))
def test_the_table_and_the_dates_spy_traded_are_the_same_set(source: str) -> None:
    """Two directions, no tolerance: the table's sessions are SPY's bar dates.

    SPY has traded every US equity session since 1993, so over any span its
    bar dates are the calendar. A date the table calls a session that SPY did
    not trade is a missing closure; a date SPY traded that the table calls a
    closure is a false holiday. This is the only check in the file that could
    catch a closure nobody thought to transcribe.
    """
    traded_list = _SPY_SESSION_SOURCES[source]()
    if traded_list is None:
        pytest.skip(f"{source} is not available (vendor data is not redistributable)")

    traded = set(traded_list)
    assert len(traded) == len(traded_list), f"{source} carries duplicate dates"
    derived = set(sessions_between(min(traded), max(traded)))

    missing_closures = sorted(derived - traded)
    false_holidays = sorted(traded - derived)
    assert (
        not missing_closures
    ), f"{len(missing_closures)} dates the table calls sessions that SPY did not trade: " + ", ".join(
        date.isoformat() for date in missing_closures[:20]
    )
    assert not false_holidays, f"{len(false_holidays)} dates SPY traded that the table calls closures: " + ", ".join(
        date.isoformat() for date in false_holidays[:20]
    )


def test_the_trend_ledger_span_is_the_one_the_reproduction_needs() -> None:
    """The reference is only evidence for 2006 if it actually starts in 2006."""
    traded = _spy_dates_from_the_trend_ledger()
    if traded is None:
        pytest.skip("the recorded trend ledger is not available")
    assert traded[0] == _d("2006-09-27")
    assert traded[-1] == _d("2026-07-17")
    assert len(traded) == 4981
    assert is_session(traded[0]) and COVERAGE_START <= traded[0]


def test_derived_early_closes_traded_materially_less_than_full_sessions() -> None:
    """A half day leaves a mark in the tape, so the early closes get a market check too.

    The transcribed list above is the authority; this is the corroboration it
    cannot give itself. Per-day volume is far too noisy to assert on — the
    2018-12-24 half day out-traded its neighbours during that December's
    selloff — so the claim is about the median across every derived early
    close, against a control median over the full sessions of the same file.
    """
    cache = _bar_cache_dir()
    if cache is None:
        pytest.skip("the bar caches are not available (vendor data is not redistributable)")
    import pandas as pd

    checked = 0
    for name in ("SPY_1d_2018-01-01_2026-07-22.parquet", "GLD_1d_2007-01-01_2017-12-31.parquet"):
        path = cache / name
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        volume = pd.Series(
            frame["volume"].to_numpy(dtype=float),
            index=pd.DatetimeIndex([stamp.date() for stamp in frame.index]),
        )
        typical = volume.rolling(21, center=True, min_periods=11).median()
        ratio = volume / typical
        early = [stamp for stamp in ratio.index if status(stamp.date()) == "early_close"]
        full = [stamp for stamp in ratio.index if status(stamp.date()) == "open"]
        assert len(early) >= 12, f"{name} covers too few early closes to say anything"
        assert ratio[early].median() < 0.75, f"{name}: derived early closes do not read as half days"
        assert 0.9 < ratio[full].median() < 1.1, f"{name}: the full-session control is not centred"
        checked += 1
    if checked == 0:
        pytest.skip("neither reference bar file is present in the cache")
