# Earnings announcement dates from 8-K item 2.02

Date: 2026-10-15. This page covers `prism.io.earnings_events` and the panel builder
`prism.scripts.build_earnings_panel`. They build an earnings-announcement calendar from SEC
filing histories, which are free, and date each announcement by the moment its filing was
accepted by EDGAR. From the calendar and daily prices they compute two per-session values that
use only what was known at that session's close:

- the **announcement return**: the stock's return less the market's over the four sessions
  around the latest announcement (Chan, Jegadeesh and Lakonishok 1996; Chen and Zimmermann's
  `AnnouncementReturn`);
- the **days to the next expected announcement**, for the announcement premium (Beaver 1968;
  Frazzini and Lamont 2007).

The filing histories come from the EDGAR store described in `docs/edgar_point_in_time.md`,
which also covers SEC's access rules, the `PRISM_SEC_USER_AGENT` requirement and the mapping
from tickers to companies. Install from a clone of this repository;
`docs/depending_on_prism.md` explains why.

## Runnable example

`examples/point_in_time/earnings_dates.py` runs every rule on this page on made-up filings and
prices, with sessions from the bundled exchange calendar (`prism.io.sessions`). It needs no
network and no User-Agent, and it writes only inside a temporary directory that it deletes:

```sh
PYTHONPATH=src python examples/point_in_time/earnings_dates.py
```

It shows an after-close release keeping its own date as day 0, a release on Good Friday (EDGAR
open, exchanges closed) moving to the next session, an 8-K/A and a same-week supplement being
dropped, the four-session window, and the expected date changing only once a release is known
at a close. Part 2 writes the event list and the daily panel the way the builder does and prints
their schemas; the tables in [Files the builder writes](#files-the-builder-writes) are checked
against that output by `tests/test_point_in_time_examples.py`.

With `--live` it lists one real company's announcement dates instead. The cache directory must
be outside any git checkout:

```sh
export PRISM_SEC_USER_AGENT="Your Name you@example.com"
PYTHONPATH=src python examples/point_in_time/earnings_dates.py --live --cik 320193 --cache-dir ~/edgar-cache
```

## Quick start

```python
import datetime as dt

import pandas as pd

from prism.io import earnings_events as ee
from prism.io import edgar, sessions

# `filings` is edgar.submissions_filings(...) for one or more companies (docs/edgar_point_in_time.md).
releases = edgar.earnings_releases(filings)           # 8-K and 8-K/A filings naming item 2.02
days = pd.DatetimeIndex(pd.to_datetime(sessions.sessions_between(dt.date(2015, 1, 2), dt.date(2025, 12, 31))))
dates = ee.announcement_dates(releases, days)          # cik, accn, acceptance, d0_pos, d0
history = ee.company_history(releases)                 # every original release, for expected dates
to_expected = {cik: ee.days_to_expected(part, days)    # calendar days, known at each close
               for cik, part in history.groupby("cik")}
```

The announcement return needs daily returns: `ee.market_returns(returns, members)` gives the
market leg, and `ee.security_events(...)` and `ee.security_panel(...)` assemble one security's
events and daily rows, as the example does (events in Part 1, the panel in Part 2).

## What counts as an announcement

An announcement is an **original 8-K that lists item 2.02**, "Results of Operations and
Financial Condition" (`edgar.earnings_releases`, then `ee.announcement_dates`):

- An **8-K/A is never an announcement.** It amends an earlier filing.
- An 8-K **without item 2.02** is not one, whatever else it reports.
- A second item 2.02 8-K from the same company **within 5 sessions** (`MIN_GAP_SESSIONS`) of the
  last one kept is treated as part of it (a supplemental or corrected release) and dropped.

In the maintainer's EDGAR store built 2026-09-24, 107,663 original 8-Ks and 963 8-K/As list
item 2.02, the earliest filed on 2004-08-24.

## Day 0 and the reaction session

**Day 0 is the New York calendar date of the filing's acceptance**, or the next session when
that date is not a session. EDGAR accepts filings on some days the exchanges are closed: the
store holds 554 filings accepted on Good Friday 2024. Of the 83,098 original item 2.02 8-Ks
filed since 2010 in the store, 45.7% were accepted before the 09:30 open, 13.4% during trading
hours and 40.9% after the 16:00 close.

A release before the open or during trading moves the price on day 0. A release after the
close keeps its own date as day 0, and the price moves on day 0 + 1. The window below covers
both, so day 0 does not have to say which session reacted. This is the convention of Compustat's `rdq` for releases after the close (`docs/earnings_design.md`,
section 3). The same record measured where the price reaction falls on the 2026-09-25 build: on
the first session closing after acceptance, the mean market-adjusted absolute return was 4.3%,
against about 1.2% on the sessions around it, and that session held the largest move within
three sessions either side for 57% of 13,842 releases.

## The announcement return

`ee.announcement_returns` sums the stock's close-to-close return less the market's over sessions
day 0 − 2 through day 0 + 1 (`WINDOW = (-2, 1)`):

- **The market leg** is the equal-weighted mean return of that session's index members
  (`ee.market_returns`), known at the close. A session with fewer than 100 reporting members
  (`MIN_MARKET_NAMES`) has no market return. The published series use a value-weighted CRSP
  index instead, which is released with a lag and cannot be used at the close.
- **A window with any missing return, or one that runs past either end of the sample, gives no
  value.** Nothing is filled.
- **The value is first usable at the close of day 0 + 1**, its signal date, and trades at the
  next open.

## The next expected announcement

For each session, `ee.days_to_expected` counts the calendar days to the company's next expected
announcement, using only releases accepted by that session's 16:00 close:

- Each known release's anniversary, 364 days later (`ANNIVERSARY_DAYS`), is a candidate.
- A candidate is dropped when it falls no later than 30 days after the latest known release
  (`REALIZED_WITHIN_DAYS`), because that release has realized it, and when it is more than 5
  days overdue (`OVERDUE_DAYS`).
- The earliest remaining candidate is the expected date.

`ee.company_history` supplies the releases over the whole store, including those before the
dataset's first session. Each is dated by its acceptance date rolled forward to a weekday
(holidays are not skipped), and a release within 7 calendar days of the last one kept is dropped
(`DUPLICATE_DAYS`). These releases only set anniversaries; events come from
`ee.announcement_dates`.

## One security per company

When several securities map to one company on a session (share classes, or a renamed company
listed under two tickers), index members are kept first, then the more liquid security
(trailing 20-session median dollar volume), then the lower security id. The others carry no
row for that session, so a company counts once in any cross-section.

## Files the builder writes

```sh
python -m prism.scripts.build_earnings_panel --dataset <dataset directory> \
    --edgar-dir <EDGAR store directory> --out <output directory>
```

- `--dataset` is a Prism dataset directory, the layout the `prism data` commands write:
  `manifest.json`, `membership.parquet` (index membership by security id),
  `listings.parquet` (security id to ticker) and `bars/<sid>_1d.parquet` (daily bars). The
  sessions are the dates on which the bars have a close, and the market leg uses the members in
  `membership.parquet`.
- `--edgar-dir` is a store from `prism.scripts.build_fundamentals`; the builder reads its
  `filings.parquet`, `ticker_cik.parquet` and `manifest.json`. The default is `data/edgar`.
- Everything is written under `--out`, under `.partial` names renamed into place at the end.

This command has not been run while writing this page. The maintainer's build is recorded in
`docs/earnings_design.md`, section 3, and Part 2 of the example produces the two parquet files
with the same functions on made-up inputs.

### `events.parquet`

One row per announcement of a security's company while the security mapped to it.

| column | type | meaning |
|---|---|---|
| `sid` | `string` | The dataset's security id. |
| `cik` | `int64` | SEC Central Index Key of the announcing company. |
| `accn` | `string` | Accession number of the announcing 8-K. |
| `acceptance` | `timestamp[ns, tz=America/New_York]` | EDGAR acceptance time of that 8-K. |
| `d0` | `timestamp[ns]` | Day 0, a session date. |
| `signal_date` | `timestamp[ns]` | Day 0 + 1: the first close at which `ear` is known. |
| `ear` | `double` | Announcement return over day 0 − 2 … day 0 + 1; null when the window leaves the sample or has a missing return. |

### `panel.parquet`

One row per session a security traded while mapped to a company, after the one-security-per-
company rule. Rows that would carry neither an announcement return nor an expected date are
left out.

| column | type | meaning |
|---|---|---|
| `date` | `timestamp[ns]` | The session. |
| `sid` | `string` | The dataset's security id. |
| `cik` | `int64` | The company the security mapped to on that session. |
| `member` | `bool` | Whether the security was an index member on that session. |
| `liquidity` | `double` | Trailing 20-session median dollar volume (close × volume) through that session. |
| `accn` | `string` | Accession number of the latest announcement whose return is known at that close; null before the first. |
| `d0` | `timestamp[ns]` | Its day 0. |
| `signal_date` | `timestamp[ns]` | Its signal date. |
| `age_sessions` | `double` | Sessions since `signal_date`, 0 on the signal date itself. |
| `ear` | `double` | Its announcement return. An announcement without a return is skipped, and the previous one carries on. |
| `days_to_expected` | `double` | Calendar days to the next expected announcement, known at that close; null without history. |

### `freeze.json`

Records the build: `built_at`; `dataset` (`path`, `id`, `as_of`); `inputs` (`path` and
`sha256` of the store's `filings`, `ticker_cik` and `edgar_manifest`); `definitions` (the
constants above); `events` (`path`, `sha256`, `rows`, `securities`, `without_return`,
`per_security_year`); `panel` (`path`, `sha256`, `rows`, `securities`, `companies`, `first`,
`last`); `median_daily_member_coverage` (`ear_age_le_60`, `ear_age_le_126`,
`days_to_expected`: the median share of each session's index members with a value);
`market_sessions_without_value`; and `peak_rss_mb`.

## Limits

- **Item 2.02 starts on 2004-08-23**, the effective date of SEC Release 33-8400, which
  renumbered Form 8-K's items. Earlier releases used the old numbering, in which item 12 carried
  results of operations, and are not matched: 4,393 8-Ks in the store, filed from 2003-04-25 to
  2004-08-20, list item 12. The bundled
  session table (`prism.io.sessions`) covers 2006–2027.
- **Not every item 2.02 filing is a quarterly result.** Companies also furnish preliminary
  results, guidance and restatement notices under it. Only releases within 5 sessions of a kept
  one are merged, and the 2026-09-25 build counted 3.8 announcements per security-year, against
  4 quarterly reports.
- **The date is the filing's, not the press release's.** Form 8-K is due within four business
  days, so a company that publishes a press release and files later is dated late. In the store,
  44 8-Ks and 8-K/As were accepted on a later date than their filing date; day 0 follows the
  acceptance. The dates have been checked against price reactions only (above), not
  against an independent calendar such as Compustat or I/B/E/S.
- **The market leg is the dataset's own index**, equal-weighted, not the value-weighted index of
  the published series. A session on which fewer than 100 members have a return has no market
  return, so no window containing it has an announcement return.
- **Expected dates assume a steady 364-day cadence.** A change of fiscal year or a skipped
  quarter gives a wrong expected date until the next release, and anniversaries more than 5
  days overdue are dropped.
- **Early closes are ignored.** The cut-off for "known at the close" is always 16:00 New York
  time, as in `docs/edgar_point_in_time.md`.
- **Ticker-to-company mapping** has the limits described in `docs/edgar_point_in_time.md`.
