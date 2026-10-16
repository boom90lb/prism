"""The bundled US equity exchange session table (engine spec §3.11, stage 0).

NYSE and Nasdaq share one regular-session calendar, so one table serves both.
It answers four questions and nothing else: is this date a session, which
sessions lie in an interval, what is the neighbouring session, and does this
session close early. Freshness, the ingest gap counter, and the research
vintage cutoff all count on this table rather than on weekday arithmetic,
because a weekday count is wrong about every holiday.

Scope. Regular US equity sessions only, for the declared coverage years. It
does not describe intraday halts (those are events, and reach freshness
through halt status), other venues, or extended-hours trading. Outside the
coverage range every entry point raises: a table that guesses about a year it
does not hold would be silently wrong exactly when nobody is watching.

Derivation, not transcription. The rows are computed from the published
exchange rules — the nine or ten annual holidays with their weekend
observances, Good Friday from the Gregorian computus, and the three
early-close rules — because
a rule that is executed is a rule that can be checked, while a hand-typed
table fails silently on its first wrong entry. ``tests/test_sessions.py``
pins the resulting dates against an independently written literal list, which
is what makes the derivation evidence rather than assertion. Closures the
rules cannot predict (a national day of mourning, a weather or systems
closure) are not derivable and are carried in ``AD_HOC_CLOSURES``, which the
exchange's announcement is the only source for.

Coverage runs 2006-2028. It was 2026-2027 at stage 0, which is all the
operating assessment needed; stage 2 widened it to 2020 because the recorded
momentum evaluation decides on every session from 2021-03-30 to 2026-06-12,
and stage 3b widened it to 2006 (decision D12) because the recorded trend
evaluation decides from 2006-09-27 and ``DatasetView`` refuses a decision date
the table does not cover. Two corrections come with the earlier years:
Juneteenth is not a closure before ``JUNETEENTH_FIRST_OBSERVED``, which alone
governs sixteen of the twenty-three years; and five announced closures no rule
predicts are carried in ``AD_HOC_CLOSURES``.

2028 was added on 2026-10-14, and only because the exchange had published it:
NYSE's "Holidays & Trading Hours" page
(https://www.nyse.com/markets/hours-calendars, retrieved 2026-10-14) lists the
2026, 2027 and 2028 holidays and 1:00 p.m. early closes. ``HOLIDAYS_2028`` and
``EARLY_CLOSES_2028`` in ``tests/test_sessions.py`` are transcribed from that
page, not computed. The rules needed no change to reproduce them: New Year's
Day 2028 falls on a Saturday and NYSE observes no holiday for it, and July 3
and November 24 are the early closes (December 24 is a Sunday). Extending the
end of the range cannot move an earlier row, because each year's rows are
built from that year's rules alone. ``tests/test_sessions.py`` checks this
anyway by comparing every 2006-2027 row with a digest taken from the table
before the extension. Do not extend the table past the last year the exchange
has published. The weekly CI job fails once the table covers fewer than 180
days ahead; that is the signal to add the next year. Each extension appends an
entry to ``TABLE_RELEASES``. A dataset's manifest records the earliest release
that covers its last date, so adding a year leaves the ``data_id`` of every
dataset that ends before it unchanged.

The extension is checked two-directionally against market data rather than
against a transcription. ``tests/test_sessions.py`` compares the table's
sessions over 2006-09-27..2026-07-21 against the dates SPY actually traded,
read from the recorded trend ledger and from the cached SPY bars. A date the
table calls a session that SPY did not trade is a missing closure; a date SPY
traded that the table calls a closure is a false holiday. Both are failures.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

# Wall-clock zone of every ``close_time`` below. The times are naive local ET;
# callers that need an instant localize them, because the table itself must
# stay free of any DST reasoning.
SESSION_TZ = "America/New_York"

REGULAR_CLOSE = dt.time(16, 0)
EARLY_CLOSE = dt.time(13, 0)

COVERAGE_START = dt.date(2006, 1, 1)
COVERAGE_END = dt.date(2028, 12, 31)

TABLE_SOURCE = "NYSE/Nasdaq published holiday and early-close rules for US equities (rule-derived; 2006-2028)"

#: Every release of this table that a dataset may record, oldest first, as
#: (version, last covered date). A release only appends years the exchange has
#: published, so the rows an earlier release covers are identical in every
#: later one (``tests/test_sessions.py`` checks 2006-2027 against a digest
#: taken before 2028 was added). A dataset therefore records the earliest
#: release that covers its last date (``table_version_for``), and its
#: ``data_id`` survives an extension that changed nothing it depends on.
#: Releases before 2006.2027-1 started later and were not compared row by row
#: with this chain, so a dataset built under one keeps the version it recorded.
TABLE_RELEASES: tuple[tuple[str, dt.date], ...] = (
    ("2006.2027-1", dt.date(2027, 12, 31)),
    ("2006.2028-1", COVERAGE_END),
)
TABLE_VERSION = TABLE_RELEASES[-1][0]

# The exchange first closed for Juneteenth on 2022-06-20: the federal holiday
# was signed on 2021-06-17 and June 19, 2021 fell on a Saturday, which the
# NYSE did not observe on the preceding Friday. June 19 is an ordinary session
# in every year of the table before this one. Applying today's ten-holiday
# rule uniformly across the span would delete one real session in each of the
# sixteen years 2006-2021 — June 19 itself in twelve of them and its weekend
# observance in the other four — the single largest way this table can be
# wrong.
JUNETEENTH_FIRST_OBSERVED = 2022

# Announced closures no rule predicts. An entry here is a dated exchange
# announcement, never an inference.
AD_HOC_CLOSURES: dict[dt.date, str] = {
    # NYSE and Nasdaq closed for the national day of mourning for President
    # Gerald R. Ford, whose state funeral was held that day. January 1, 2007
    # was the New Year's Day closure and January 2 this one, so the year's
    # first session is January 3.
    dt.date(2007, 1, 2): "National Day of Mourning",
    # Hurricane Sandy. The exchanges closed Monday and Tuesday — the first
    # consecutive two-day weather closure since 1888 — and reopened Wednesday
    # October 31 for a full session.
    dt.date(2012, 10, 29): "Hurricane Sandy",
    dt.date(2012, 10, 30): "Hurricane Sandy",
    # NYSE and Nasdaq closed for the national day of mourning for President
    # George H. W. Bush (exchange notice, 2018-12-03).
    dt.date(2018, 12, 5): "National Day of Mourning",
    # NYSE and Nasdaq closed for the national day of mourning for President
    # Jimmy Carter (exchange notice, 2024-12-30).
    dt.date(2025, 1, 9): "National Day of Mourning",
}

SessionStatus = Literal["open", "early_close", "holiday", "weekend"]

MONDAY = 0
THURSDAY = 3
FRIDAY = 4
SATURDAY = 5
SUNDAY = 6


class SessionTableRangeError(ValueError):
    """A date was asked about that the bundled table does not cover.

    Raised rather than answered, because the caller's alternative — a False
    that means "not covered" and a False that means "closed" — is the silent
    wrong answer this table exists to prevent.
    """


class NotASessionError(ValueError):
    """A session attribute (close time, early-close status) was asked of a closed day."""


@dataclass(frozen=True, slots=True)
class Session:
    """One row of the exchange calendar: a covered weekday and how it trades."""

    date: dt.date
    status: SessionStatus
    close_time: dt.time | None
    name: str
    source: str = TABLE_SOURCE
    version: str = TABLE_VERSION


def _easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (the anonymous computus)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lu = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lu) // 451
    month, day = divmod(h + lu - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The ``n``-th ``weekday`` (Mon=0) of a month, 1-based."""
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    """The last ``weekday`` (Mon=0) of a month."""
    if month == 12:
        last = dt.date(year, 12, 31)
    else:
        last = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(holiday: dt.date) -> dt.date:
    """The observed closure for a fixed-date holiday: Saturday moves back to Friday, Sunday forward to Monday."""
    if holiday.weekday() == SATURDAY:
        return holiday - dt.timedelta(days=1)
    if holiday.weekday() == SUNDAY:
        return holiday + dt.timedelta(days=1)
    return holiday


def _holidays(year: int) -> dict[dt.date, str]:
    """The exchange holidays observed within ``year``.

    New Year's Day is the one exception to the Saturday-back-to-Friday rule:
    the exchange does not close on December 31 for a Saturday January 1, so a
    Saturday New Year's is observed nowhere rather than in the prior year.
    """
    new_year = dt.date(year, 1, 1)
    observed: dict[dt.date, str] = {}
    if new_year.weekday() != SATURDAY:
        observed[_observed(new_year)] = "New Year's Day"
    observed[_nth_weekday(year, 1, MONDAY, 3)] = "Martin Luther King Jr. Day"
    observed[_nth_weekday(year, 2, MONDAY, 3)] = "Washington's Birthday"
    observed[_easter(year) - dt.timedelta(days=2)] = "Good Friday"
    observed[_last_weekday(year, 5, MONDAY)] = "Memorial Day"
    if year >= JUNETEENTH_FIRST_OBSERVED:
        observed[_observed(dt.date(year, 6, 19))] = "Juneteenth National Independence Day"
    observed[_observed(dt.date(year, 7, 4))] = "Independence Day"
    observed[_nth_weekday(year, 9, MONDAY, 1)] = "Labor Day"
    observed[_nth_weekday(year, 11, THURSDAY, 4)] = "Thanksgiving Day"
    observed[_observed(dt.date(year, 12, 25))] = "Christmas Day"
    for date, reason in AD_HOC_CLOSURES.items():
        if date.year == year:
            observed[date] = reason
    return observed


def _early_closes(year: int, holidays: dict[dt.date, str]) -> dict[dt.date, str]:
    """The 1:00 pm closes observed within ``year``.

    Three rules: the day after Thanksgiving; Christmas Eve when it is a
    weekday and not itself the observed Christmas closure; and July 3 when
    Independence Day is observed on the 4th and the 3rd is a weekday.
    """
    early: dict[dt.date, str] = {}
    early[_nth_weekday(year, 11, THURSDAY, 4) + dt.timedelta(days=1)] = "Day after Thanksgiving"

    christmas_eve = dt.date(year, 12, 24)
    if christmas_eve.weekday() <= FRIDAY and christmas_eve not in holidays:
        early[christmas_eve] = "Christmas Eve"

    july_third = dt.date(year, 7, 3)
    independence = dt.date(year, 7, 4)
    if independence.weekday() <= FRIDAY and july_third.weekday() <= FRIDAY and july_third not in holidays:
        early[july_third] = "Day before Independence Day"

    return {date: reason for date, reason in early.items() if date not in holidays}


def _build_table() -> dict[dt.date, Session]:
    """Every weekday in the coverage range, with its status. Weekends carry no row."""
    table: dict[dt.date, Session] = {}
    for year in range(COVERAGE_START.year, COVERAGE_END.year + 1):
        holidays = _holidays(year)
        early = _early_closes(year, holidays)
        date = dt.date(year, 1, 1)
        while date.year == year:
            if date.weekday() <= FRIDAY:
                if date in holidays:
                    table[date] = Session(date, "holiday", None, holidays[date])
                elif date in early:
                    table[date] = Session(date, "early_close", EARLY_CLOSE, early[date])
                else:
                    table[date] = Session(date, "open", REGULAR_CLOSE, "Regular session")
            date += dt.timedelta(days=1)
    return table


_TABLE: dict[dt.date, Session] = _build_table()
_SESSIONS: tuple[dt.date, ...] = tuple(
    date for date in sorted(_TABLE) if _TABLE[date].status in ("open", "early_close")
)


def coverage() -> tuple[dt.date, dt.date]:
    """The closed date interval the table can answer for."""
    return COVERAGE_START, COVERAGE_END


def coverage_end() -> dt.date:
    """The last date the table covers — what the health check compares against its ninety-day warning."""
    return COVERAGE_END


def table_version_for(last: dt.date) -> str:
    """The table version a dataset whose dates end on ``last`` records in its manifest.

    That is the earliest entry of ``TABLE_RELEASES`` covering ``last``: every row
    such a dataset could have read is the same in that release and in every
    later one. A date past all of them gets ``TABLE_VERSION``, the release that
    answered for every date it could.
    """
    for version, through in TABLE_RELEASES:
        if last <= through:
            return version
    return TABLE_VERSION


def _require_covered(date: dt.date) -> None:
    if not COVERAGE_START <= date <= COVERAGE_END:
        raise SessionTableRangeError(
            f"{date.isoformat()} is outside the bundled session table "
            f"{COVERAGE_START.isoformat()}..{COVERAGE_END.isoformat()} (version {TABLE_VERSION})"
        )


def status(date: dt.date) -> SessionStatus:
    """How ``date`` trades: ``open``, ``early_close``, ``holiday``, or ``weekend``."""
    _require_covered(date)
    row = _TABLE.get(date)
    return "weekend" if row is None else row.status


def is_session(date: dt.date) -> bool:
    """Whether the exchange trades a regular session on ``date`` (early closes included)."""
    return status(date) in ("open", "early_close")


def is_early_close(date: dt.date) -> bool:
    """Whether ``date`` closes at 1:00 pm ET. Raises for a day the exchange does not trade."""
    if not is_session(date):
        raise NotASessionError(f"{date.isoformat()} is not a session ({status(date)}); early close is undefined")
    return _TABLE[date].status == "early_close"


def session_close(date: dt.date) -> dt.time:
    """The closing wall-clock time on ``date`` in ``SESSION_TZ``. Raises for a non-session."""
    if not is_session(date):
        raise NotASessionError(f"{date.isoformat()} is not a session ({status(date)}); it has no close time")
    close = _TABLE[date].close_time
    if close is None:  # unreachable by construction; a session row always carries a close time
        raise NotASessionError(f"{date.isoformat()} is marked a session but carries no close time")
    return close


def session_row(date: dt.date) -> Session:
    """The full table row for a covered weekday. Raises for a weekend, which carries no row."""
    _require_covered(date)
    row = _TABLE.get(date)
    if row is None:
        raise NotASessionError(f"{date.isoformat()} is a weekend and carries no calendar row")
    return row


def sessions_between(start: dt.date, end: dt.date) -> list[dt.date]:
    """The sessions in the closed interval ``[start, end]``, ascending."""
    _require_covered(start)
    _require_covered(end)
    if start > end:
        raise ValueError(f"start {start.isoformat()} is after end {end.isoformat()}")
    return [date for date in _SESSIONS if start <= date <= end]


def next_session(date: dt.date) -> dt.date:
    """The first session strictly after ``date``. Raises if the table ends first."""
    _require_covered(date)
    for candidate in _SESSIONS:
        if candidate > date:
            return candidate
    raise SessionTableRangeError(
        f"no session after {date.isoformat()} within the bundled table (ends {COVERAGE_END.isoformat()})"
    )


def previous_session(date: dt.date) -> dt.date:
    """The last session strictly before ``date``. Raises if the table starts later."""
    _require_covered(date)
    for candidate in reversed(_SESSIONS):
        if candidate < date:
            return candidate
    raise SessionTableRangeError(
        f"no session before {date.isoformat()} within the bundled table (starts {COVERAGE_START.isoformat()})"
    )
