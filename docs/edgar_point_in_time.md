# Point-in-time fundamentals from SEC EDGAR

Date: 2026-10-15. This page covers `prism.io.edgar` and the store builder
`prism.scripts.build_fundamentals`. Together they turn SEC's free XBRL data into company
fundamentals that a backtest can read without look-ahead: a value is visible at a decision
time only if the filing that reported it had been accepted by EDGAR by then.

Three properties make that possible:

- **One row per fact per filing.** A number reported in a 10-Q and repeated, or restated, in a
  later 10-K appears twice, once for each filing. Restatements are kept, never overwritten.
- **Each row is timed by its filing's EDGAR acceptance timestamp**, not by the period it
  describes and not by the date SEC's database was downloaded.
- **Gaps stay gaps.** A trailing-twelve-month value needs four contiguous quarters; when they
  are missing there is no value, never a filled one.

No API key is needed. SEC asks every automated client to identify itself with a User-Agent
that names the requester and gives a contact e-mail address, and to stay under 10 requests per
second ([Accessing EDGAR Data](https://www.sec.gov/os/accessing-edgar-data), checked
2026-10-15). The client reads the User-Agent from `PRISM_SEC_USER_AGENT`, refuses to start
without an `@` in it, and sends at most 5 requests per second.

Install from a clone of this repository; `docs/depending_on_prism.md` explains why.
`docs/earnings_dates.md` covers the earnings-announcement calendar built from the same filing
histories.

## Runnable example

`examples/point_in_time/edgar_fundamentals.py` runs every rule on this page on made-up
filings. It needs no network and no User-Agent, and it writes only inside a temporary
directory that it deletes:

```sh
PYTHONPATH=src python examples/point_in_time/edgar_fundamentals.py
```

Part 1 shows a restatement appearing only after the restating 10-K/A's acceptance, a fact
dated by the fallback rule, quarters derived from year-to-date values, and a company refused a
trailing-twelve-month value. Part 2 runs the store builder on the same payloads through the
real client and prints the schema of every file it writes; the tables in
[Files the builder writes](#files-the-builder-writes) are checked against that output by
`tests/test_point_in_time_examples.py`.

With `--live` it reads one real company from SEC instead. Run it with a cache directory outside
any git checkout; the example refuses one inside a checkout:

```sh
export PRISM_SEC_USER_AGENT="Your Name you@example.com"
PYTHONPATH=src python examples/point_in_time/edgar_fundamentals.py --live --cik 320193 \
    --cache-dir ~/edgar-cache --as-of 2025-06-30
```

## Quick start

```python
from pathlib import Path

from prism.io import edgar

client = edgar.SecClient(edgar.user_agent_from_env(), Path("~/edgar-cache").expanduser())
history, _ = client.submissions(320193)                       # recent filings; older ones in pages
pages = [client.submissions_page(p["name"])[0] for p in history["filings"].get("files", [])]
filings = edgar.submissions_filings(history, pages)           # one row per filing, with acceptance time
payload, _ = client.companyfacts(320193)
facts = edgar.companyfacts_frame(payload, edgar.concept_names())  # one row per fact per filing
facts = edgar.attach_availability(facts, filings)             # adds available_at

pit = edgar.point_in_time(facts, edgar.session_close("2025-06-30"))  # known at that 16:00 close
edgar.trailing_twelve_months(pit, "revenue")    # columns cik, ttm, period_end, derived_quarters
edgar.latest_instant(pit, "assets")             # columns cik, value, period_end
```

Every request is cached under the cache directory, gzip-compressed, with a sidecar recording
the URL, the fetch time and the SHA-256 of the body. A cached body whose hash no longer matches
its sidecar is refused. Pass `refresh=True` to fetch again; a vendor revision then shows as a
changed hash.

## When a fact becomes usable

`attach_availability` gives each fact an `available_at` timestamp in New York time:

1. **The filing's acceptance time**, joined by accession number from the filing history. EDGAR
   publishes it as `acceptanceDateTime` in UTC; it is converted to New York time. The UTC
   reading was checked against Apple's earnings 8-Ks, released at 16:30 New York time: they
   show 20:30Z in summer and 21:30Z in winter.
2. **Otherwise, 23:59:59 New York time on the filing date**, with `availability_basis` set to
   `filed_date_end`. This happens when a fact's accession number is missing from the filing
   history. The fallback is meant to err late: EDGAR dates a filing whose transmission starts
   after 17:30 New York time on the next business day (Regulation S-T Rule 13(a)(2)), so the
   acceptance usually falls on or before the end of the filing date. Usually is not always:
   see the measurements below.

A decision made at the close of session *t* may use a fact when `available_at` is at or before
16:00 New York time on *t* (`edgar.session_close`). A 10-K accepted at 16:20 is therefore first
usable at the next session's close.

Measured on the maintainer's store built 2026-09-24 (the coverage ledger
`data/universe/edgar_coverage_2026-09-24.json` is tracked; the store itself is not):

- 3,133,899 of 3,133,978 facts are dated by acceptance, and 79, from two filings, by the
  fallback. Neither of those two filings appears in any filing history in the store, so their
  acceptance times could not be checked.
- Of the 24,793 8-Ks accepted after 17:30, 23,613 carry a later filing date: the Rule 13(a)
  behaviour the fallback relies on.
- The rule has exceptions in the other direction. 117 of the 530,777 filings on forms 10-K,
  10-Q and 8-K (with their amendments), 10-KT, 20-F and 40-F were accepted on a date after
  their filing date, from one day to several years later. Had one of those been missing from
  its filing history, the fallback would have dated its facts too early.

For most companies the acceptance time is not when the numbers first became public. A
quarter's revenue is usually published first in an earnings press release (an 8-K with item
2.02, whose exhibit is not XBRL-tagged) and reaches the XBRL data only with the 10-Q or 10-K,
days or weeks later. For those numbers the stamp is late, which costs timeliness but does not
leak the future.

## Point-in-time semantics

`point_in_time(facts, as_of)` keeps the facts whose `available_at <= as_of`, then, for each
(company, taxonomy, concept, unit, period start, period end), the row from the latest filing.
Ties in `available_at` are broken by accession number. `as_of` must be timezone-aware.

The result is the company as an investor could have known it at `as_of`: the latest reported
value for every period, including restated values once the restatement was accepted and the
original before. A later filing that stops reporting a period does not delete it: the last value
reported stays.

## Canonical fields

`edgar.FIELDS` names 15 fields and the XBRL concepts that report each, in order of preference.
For each period, the first concept with a value wins, so a company that moved from
`SalesRevenueNet` to the ASC 606 revenue concept in 2018 reads as one series.
`field_facts(pit, name)` applies that rule. `concept_names()` lists the concepts the store keeps.

| field | concepts, in order of preference |
|---|---|
| `revenue` | RevenueFromContractWithCustomerExcludingAssessedTax, Revenues, SalesRevenueNet, RevenueFromContractWithCustomerIncludingAssessedTax, SalesRevenueGoodsNet |
| `cost_of_revenue` | CostOfRevenue, CostOfGoodsAndServicesSold, CostOfGoodsSold |
| `gross_profit` | GrossProfit |
| `operating_income` | OperatingIncomeLoss |
| `net_income` | NetIncomeLoss, ProfitLoss |
| `eps_diluted` | EarningsPerShareDiluted, EarningsPerShareBasicAndDiluted |
| `cfo` | NetCashProvidedByUsedInOperatingActivities, NetCashProvidedByUsedInOperatingActivitiesContinuingOperations |
| `capex` | PaymentsToAcquirePropertyPlantAndEquipment |
| `rd_expense` | ResearchAndDevelopmentExpense |
| `dividends_paid` | PaymentsOfDividends, PaymentsOfDividendsCommonStock |
| `buybacks` | PaymentsForRepurchaseOfCommonStock |
| `assets` | Assets |
| `liabilities` | Liabilities |
| `equity` | StockholdersEquity, StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest |
| `shares_outstanding` | dei:EntityCommonStockSharesOutstanding, CommonStockSharesOutstanding |

All concepts are in the `us-gaap` taxonomy except the one marked `dei:` (document and entity
information, the cover-page data).

## Quarters and trailing twelve months

`period_kind` classifies each duration fact by its length in days (end − start + 1): `quarter`
80–100, `half` 170–195, `nine_months` 260–285, `annual` 350–380; facts without a start date are
`instant`, and anything else is `other`.

Cash-flow statements are reported year to date only, and a 10-K seldom reports the fourth
quarter on its own. `quarterly_values` therefore takes a quarter as reported where a
three-month fact exists, and otherwise as the difference of two reports that share a start
date and end one quarter apart: six months less three, nine less six, twelve less nine. The
column `basis` says which (`reported` or `derived`).

`trailing_twelve_months(pit, name)` sums a company's latest four quarters when they are
contiguous: each starts the day after the previous one ends, give or take 7 days for 52/53-week
fiscal years. A company whose latest four quarters are not contiguous is absent from the result.
`derived_quarters` counts how many of the four were derived. `latest_instant(pit, name)` returns
each company's latest-dated instant value of a balance-sheet or share-count field.

## Files the builder writes

`prism.scripts.build_fundamentals` builds the store for every ticker of a membership file. It
maps each ticker to a company (next section), fetches every accepted company's filing history,
all pages of it, and its company facts, and writes one company at a time, so memory holds one
company's payloads at a time (the 2026-09-24 build peaked at 376 MB). Every output is written under a
`.partial` name and renamed into place only when the whole build succeeds.

**Its default directories are inside the repository** (`data/edgar/raw`, `data/edgar` and
`data/universe`), which is how the maintainer's store is kept. Anyone else should pass all three
directory flags. This smoke run fetches five companies and writes nothing into the checkout:

```sh
DIR=$HOME/prism-edgar        # any directory outside the checkout
export PRISM_SEC_USER_AGENT="Your Name you@example.com"
python -m prism.scripts.build_fundamentals \
    --membership data/universe/sp500_membership_2026-09-25.parquet --asof 2026-10-15 --limit 5 \
    --cache-dir "$DIR/raw" --out-dir "$DIR" --ledger-dir "$DIR"
```

This command has not been run while writing this page; Part 2 of the example runs the same
`build` function offline. `--membership` is a parquet file with columns `ticker`, `start` and
`end` (inclusive dates). `--limit 5` keeps the first five tickers in alphabetical order. For
scale, the 2026-09-24 build of 1,565 tickers fetched 3,189 filing-history pages and 1,477
company-facts files (5.3 GB uncompressed, 386 MB in the gzip cache): about 4,670 requests,
which take at least 15 minutes at 5 per second.

### `facts.parquet`

One row per reported fact per filing, for the concepts behind `FIELDS` only.

| column | type | meaning |
|---|---|---|
| `cik` | `int64` | SEC Central Index Key of the filer. |
| `taxonomy` | `string` | `us-gaap` or `dei`; facts in other taxonomies are dropped. |
| `concept` | `string` | XBRL concept name without its prefix, for example `Assets`. |
| `unit` | `string` | Unit as reported: `USD`, `USD/shares`, `shares` and occasionally others. |
| `start` | `timestamp[ns]` | First day of a duration fact; null for an instant (balance sheet, share count). |
| `end` | `timestamp[ns]` | Last day of the period, or the date of the instant. |
| `val` | `double` | The value as reported, in `unit`, unscaled. |
| `accn` | `string` | Accession number of the filing that reported this value. |
| `fy` | `int64` | Fiscal year of the reporting filing, not of the fact's period: a 10-K's prior-year comparatives carry the 10-K's year. |
| `fp` | `string` | Fiscal period of the reporting filing: `FY`, `Q1`, `Q2` or `Q3`, rarely `Q4`, `CY`, empty or null. |
| `form` | `string` | Form type of the reporting filing, for example `10-K`, `10-Q` or `10-K/A`. |
| `filed` | `timestamp[ns]` | EDGAR filing date of the reporting filing. |
| `frame` | `string` | SEC's calendar-period label (for example `CY2023`, `CY2023Q1`, `CY2023Q4I`), set on at most one row per concept, unit and calendar period, null on the others. |
| `available_at` | `timestamp[ns, tz=America/New_York]` | When the fact became usable: the filing's acceptance time, else 23:59:59 on `filed`. |
| `availability_basis` | `string` | `acceptance` or `filed_date_end`, the rule that set `available_at`. |

### `filings.parquet`

Every filing in each accepted company's history, all forms.

| column | type | meaning |
|---|---|---|
| `cik` | `int64` | SEC Central Index Key of the filer. |
| `accn` | `string` | Accession number. |
| `form` | `string` | Form type, for example `10-K`, `8-K`, `4`. |
| `filing_date` | `timestamp[ns]` | EDGAR filing date (Regulation S-T Rule 13: after 17:30 New York time, the next business day, with a 22:00 cut-off for Forms 3, 4, 5, 144 and some schedules). |
| `acceptance` | `timestamp[ns, tz=America/New_York]` | EDGAR acceptance time, converted from UTC. |
| `report_date` | `timestamp[ns]` | Period of report where EDGAR gives one, else null. |
| `items` | `string` | 8-K item numbers, comma-separated (for example `2.02,9.01`). Empty for most other forms; a few (for example `EFFECT`, `CT ORDER`, `D`) carry other EDGAR codes. |

### `ticker_cik.parquet`

The accepted ticker intervals and the company each maps to.

| column | type | meaning |
|---|---|---|
| `ticker` | `string` | Ticker as in the membership file, in upper case without spaces. |
| `start` | `timestamp[ns]` | First day of the interval (inclusive). |
| `end` | `timestamp[ns]` | Last day of the interval (inclusive). |
| `cik` | `int64` | The company the ticker referred to over the interval. |
| `title` | `string` | Company name: SEC's current title, or the name recorded by a name match or a review decision. |
| `source` | `string` | How the CIK was found: `sec_current_ticker`, `rename_successor:<ticker>`, `name_match` or `review_decision`. |
| `reason` | `string` | The reviewer's reason for a review decision, else null. When no accepted row has a reason, Arrow stores this column with type `null`. |
| `status` | `string` | Always `accepted` in this file; the coverage ledger lists the other intervals. |

### `manifest.json`, the coverage ledger and the cache

- **`manifest.json`** has the keys `company_tickers` (one record), `submissions` (per CIK, a
  list with one record per history page), `companyfacts` (per CIK, one record) and `built_at`.
  Each record holds `url`, `fetched_at`, `sha256`, `bytes` and `cache_path`.
- **The coverage ledger**, `{ledger-dir}/edgar_coverage_{asof}.json`, records what the build
  covered:

  | key | meaning |
  |---|---|
  | `asof` | The `--asof` label. |
  | `membership_rows` | Rows in the membership file. |
  | `tickers` | Distinct tickers processed. |
  | `xbrl_era_tickers` | Tickers with an interval ending on or after 2009-06-15. |
  | `accepted_tickers` | XBRL-era tickers with at least one accepted interval. |
  | `coverage_xbrl_era` | `accepted_tickers / xbrl_era_tickers`. |
  | `accepted_ciks` | Companies with at least one accepted interval. |
  | `ciks_without_companyfacts` | Accepted companies for which SEC returned no company facts (HTTP 404). |
  | `facts_rows` | Rows in `facts.parquet`. |
  | `filings_rows` | Rows in `filings.parquet`. |
  | `availability_basis` | Fact counts per `availability_basis`. |
  | `peak_rss_mb` | Peak resident memory of the build. |
  | `status_counts` | Interval counts per status: `accepted`, `pre_xbrl`, `unresolved`, `no_periodic_filing_in_interval`. |
  | `not_accepted` | Every XBRL-era interval not accepted, with ticker, dates, CIK, status and reason. |

- **The cache** (`--cache-dir`) holds `company_tickers.json.gz`,
  `submissions/CIK##########.json.gz` (and `CIK##########-submissions-NNN.json.gz` for older
  pages) and `companyfacts/CIK##########.json.gz`, each with a `.meta.json` sidecar.

## Mapping tickers to companies

EDGAR is keyed by CIK, index membership by ticker, and tickers are reused. The builder maps a
ticker in three steps and records each decision:

1. **SEC's current ticker map** (`company_tickers.json`), directly or through a reviewed rename
   successor (`prism.io.universe_sp500.RENAME_TABLE`).
2. **Acceptance per interval.** A mapping is accepted for a membership interval only if that
   CIK filed a 10-K or 10-Q inside it, or in the quarter before it. This rejects a reused
   ticker's current holder for an earlier holder's interval. Intervals that end before
   2009-06-15 are `pre_xbrl`; names the current map does not carry are `unresolved`. Nothing is
   guessed.
3. **Reviewed overrides.** `--name-matches` takes the output of
   `prism.scripts.match_delisted_ciks`, which matches unresolved names to SEC entities by exact
   normalized name and accepts a match only when exactly one candidate filed during the
   interval. `--review-decisions` takes files of decisions made by a person, each assigning one
   CIK to an interval or splitting the interval between a predecessor and a successor filer.
   Decisions take precedence over name matches, and an interval decided in two files is
   refused.

The tracked files under `data/universe/` are the maintainer's: `cik_name_matches_2026-09-24.json`
(374 accepted matches, 81 sent to review), the review sheets `cik_review_2026-09-24.md` and
`cik_nearmiss_review_2026-09-24.md` with the evidence for each case, and the decisions
`cik_review_decisions_2026-09-24.json` and `cik_nearmiss_decisions_2026-09-24.json`. They were
made for `sp900_membership_2026-09-23.parquet`. With another membership file, a decision whose
interval ends on a different date is refused ("re-review it"), and one whose interval starts on
a different date is not applied. A decision file looks like this (example Part 2 writes one):

```json
{"decisions": [
  {"ticker": "ACE", "start": "1990-01-01", "end": "2016-01-19", "kind": "cik", "cik": 896159,
   "name": "Chubb Ltd", "reason": "..."},
  {"ticker": "ARRS", "start": "2014-06-20", "end": "2019-04-11", "kind": "split", "reason": "...",
   "segments": [{"from": "2014-06-20", "to": "2015-11-15", "cik": 1141107, "name": "ARRIS GROUP INC"},
                {"from": "2015-11-16", "to": "2019-04-11", "cik": 1645494, "name": "ARRIS International plc"}]}
]}
```

Split segments must tile the interval exactly, day after day.

## Limits

- **History starts in 2009–2011.** SEC phased XBRL in by filer size: the largest filers from
  fiscal periods ending on or after 2009-06-15, other large accelerated filers from 2010-06-15
  and everyone else from 2011-06-15 (Release 33-9002, *Interactive Data to Improve Financial
  Reporting*, 2009). In the 2026-09-24 store, 391 of 1,477 companies first filed XBRL in 2009,
  566 in 2010 and 210 in 2011; the other 310 first filed later. Comparatives reach back further
  (1% of facts describe periods ending before 2009), but they are timed by the filing that
  reported them.
- **Only the concepts behind `FIELDS`.** The builder keeps 27 concepts. A field needs a new
  entry in `FIELDS` and a rebuild. Company-specific extension concepts and filers reporting in
  IFRS (`ifrs-full` taxonomy) produce no facts.
- **Units are not converted.** Values stay in the unit the filer used, and `unit` says which.
  `field_facts` does not filter by unit, and the outputs of `trailing_twelve_months` and
  `latest_instant` carry no unit column. In the 2026-09-24 store, 99.99% of facts are in `USD`,
  `USD/shares` or `shares`, and 80 of 25,738 company–concept pairs appear in more than one unit;
  for those, one unit is kept per period without regard to which.
- **Share counts.** `dei:EntityCommonStockSharesOutstanding` is dated at the cover page, usually
  a few weeks after the balance sheet. Splits are not in EDGAR: market equity and share issuance
  in `prism.io.fundamental_panel` apply split history from Alpaca's corporate-actions endpoint
  (`prism.live.split_history`), which needs Alpaca keys.
- **Early closes are ignored.** `session_close` is always 16:00. On an early-close day (13:00;
  `prism.io.sessions.is_early_close` identifies them), a filing accepted between 13:00 and 16:00
  counts as known at that day's close although trading had ended. With fills at the next open
  it was still public before the trade.
- **Ticker-to-company mapping** depends on the reviewed tables above. Membership is keyed by
  ticker, so a ticker reused within one interval is handled only by a review decision.
- **Fair access applies to every user.** Each person running the builder sends their own
  traffic to SEC under their own User-Agent. The client retries HTTP 429 and 5xx responses with
  exponential backoff, and fails on 403 (SEC refused the request) and 404.
