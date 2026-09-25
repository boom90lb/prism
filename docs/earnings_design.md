# Earnings announcements — design and registration (`earnings_v1`, `earnings_daily_v1`)

**Status: registered 2026-09-24.** The registration is the commit that adds this file
with:

- `registrations/2026-09-earnings.toml` (the monthly arm);
- `registrations/2026-09-earnings-replication.toml` (its criteria and the combination);
- `registrations/2026-09-earnings-daily.toml` (the two daily arms);
- the books `examples/earnings/monthly.toml` and `examples/earnings/daily.toml`;
- the reads `research/scripts/earnings_replication.py` and `research/scripts/earnings_read.py`.

At registration, no earnings book's return has been computed, and no relation between an
earnings score and any later return.

## 1. What is registered

| selection set | route | arm | score | decisions | held |
|---|---|---|---|---|---|
| `earnings_v1` | replication | `ear_monthly` | announcement return (Chan, Jegadeesh and Lakonishok 1996) | every 21 sessions | while ≤ 126 sessions old |
| `earnings_v1` | replication | combination | `ear_monthly` + vol-scaled momentum, equal risk at 12% | — | — |
| `earnings_daily_v1` | discovery | `ear_event` | announcement return | every session | 60 sessions from the day it is known |
| `earnings_daily_v1` | discovery | `announcement_premium` | minus the days to the next expected announcement (Beaver 1968; Frazzini and Lamont 2007) | every session | while it ranks |

Every arm is a top/bottom-decile long/short book on the point-in-time S&P 500.

- **`ear_monthly`** is built the way the public series is built, so it can be tested
  against it (SPEC §4, replication route).
- **The daily arms have no free public series.** They are discovery-route
  configurations: this in-house read can kill them, and only data after this
  registration can promote them (§6).
- **The two sets use separate books.** In one book, a zero-weight daily component would
  re-trade the monthly arm every session.

## 2. Why, and what to expect

- **New information (SPEC §5.3).** Prices and quarterly accounts are saturated on this
  sample. Announcement dates from 8-K item 2.02 are a new, point-in-time source.
- **Why the announcement return, and not the earnings surprise.**
  - XBRL figures arrive with the 10-Q or 10-K, weeks after the announcement.
  - Our `niq_su` could only be known at filing, and it tracked its public series at 0.29
    (`docs/fundamentals_read_2026-09-24.md`).
  - The number itself is announced in an untagged press-release exhibit, which is not
    parsed here.
  - The announcement return needs only prices and the date, and both are known at the
    close after the announcement. It is the market's own reading of the surprise.
- **The public record.**
  - Chen and Zimmermann classify `AnnouncementReturn` as a clear predictor.
  - Their October 2024 release fixed a look-ahead bug: month-end announcements used the
    next month's returns, and the factor had looked like the best in the zoo
    (CrossSection issue #158). The 2025.10 release used here postdates the fix.
  - Martineau (2022, *Critical Finance Review*) argues that post-earnings-announcement
    drift has largely disappeared, first in large caps.
  - **The central expectation is a small or zero prior after 1996 for large-cap,
    value-weighted books.**
- **Expectations for the daily arms.**
  - `ear_event` enters on average about ten sessions sooner than a monthly
    re-formation. It gains only if the drift is front-loaded.
  - `announcement_premium` replaces most of its long leg every week. Its cost drag will
    be high, and it survives only if the premium is large in large caps.

## 3. Data and implementation

- **Announcements.** Definitions are in `prism.io.earnings_events`.
  - An announcement is an original 8-K with item 2.02 from the EDGAR store, dated by its
    acceptance time. Amendments are never announcements.
  - Day 0 is the acceptance date in New York, or the next session if that date is not
    one. That matches Compustat's `rdq` for a release after the close.
  - A second item 2.02 8-K within 5 sessions of a kept one is dropped as part of it.
  - The set includes some pre-announcements and guidance releases: 3.8 per security-year.
- **The dates locate the price reaction.** This was checked on absolute returns only:
  - On the first session closing after acceptance, the mean market-adjusted absolute
    return is 4.3%, against about 1.2% on the sessions around it.
  - That session holds the largest move in [−3, +3] for 57% of 13,842 releases.
- **Announcement return.**
  - It is the sum over sessions d0−2 … d0+1 of the stock's close-to-close return, minus
    the equal-weighted return of that session's index members.
  - The window is Chen and Zimmermann's.
  - The market leg departs from their CRSP value-weighted index: French's series is
    published with a lag and could not be used live (N1).
  - It is known at d0+1's close and traded at the next open (N2).
- **Expected announcement date, for the premium.**
  - Candidates are each known announcement's anniversary, +364 days.
  - An anniversary is dropped once a later actual announcement has realized it (it falls
    within 30 days after the latest known one), or when it is more than 5 days overdue.
  - The earliest remaining candidate is the expected date.
  - Only releases accepted by the decision's 16:00 close count. History before 2020 is
    included (weekday-dated, near-duplicates within 7 days dropped) to set anniversaries.
- **Frozen panel.** `data/frozen/earnings_v1/`, built by
  `prism.scripts.build_earnings_panel`, whose `freeze.json` hashes every input.
  - Events file: sha256 `9f005f1d…54e7`, 13,963 events on 569 securities.
  - Panel file: sha256 `d4703a6c…701e`, 564 securities and 560 companies, one security per
    company per session.
  - 98 events have no return. They are renamed tickers after their price series ends; the
    company is carried under its other ticker.
  - Median daily coverage of index members: 95.2% with an announcement return ≤ 60
    sessions old; 97.1% at ≤ 126; 97.3% with an expected date.
- **Nodes** (`prism.signal.earnings_node`):
  - `EarningsAnnouncementNode(panel, panel_sha256, max_age_sessions)`;
  - `AnnouncementProximityNode(panel, panel_sha256)`.
  - Both refuse a panel with the wrong hash and decision dates outside the panel.
- **Engine runs.** Identical to the fundamentals arms: frozen dataset
  `data/frozen/demotion_b1`, folds `folds/demotion_b1.toml`, cost stack `retail_v1`,
  execution profile `execution_alpaca`, canonical engine, $100,000, and the same
  eligibility screen and construction.
  - The engine's trailing dollar-volume statistic reads only its last window since the
    commit before this one. Its output is unchanged: exact for the median every
    configuration uses (`tests/test_trailing_dollar_volume.py`).
- **Mechanics check.** The engine's freeze step was run to scratch; it computes scores,
  not returns.
  - Monthly book: 63 decision rows in 26 s.
  - Daily book: 1,308 rows in 351 s.
  - No warnings.

## 4. The monthly arm: SPEC §4 replication criteria

The reference is Chen and Zimmermann's daily value-weighted `AnnouncementReturn`
long−short: quintile 5 minus 1, re-formed monthly, 1971-09 → 2024-12, pinned by sha256.

**Value weight, not the paper's equal weight:**
- the arm trades S&P 500 large caps, which dominate the value-weighted series;
- an equal-weighted prior would be set by stocks the arm cannot hold.

The source is Chan, Jegadeesh and Lakonishok, "Momentum strategies", *Journal of Finance*
51(5), December 1996.

1. **Publication bar and prior.**
   - The bar is t ≥ 3 over ≥ 20 years on the full series.
   - The prior is the post-publication Sharpe, 1996-12 → 2024-12 (28 years).
   - A prior ≤ 0 fails.
2. **Fidelity.**
   - Criterion: the daily correlation of the arm's net return with the public series,
     over the scored sample's overlap with it, must be ≥ 0.5. The overlap ends 2024-12;
     the admission rule sets its start.
   - A negative correlation is recorded as a direction mismatch and fails.
   - Weekly correlation is reported beside it.
3. **Economics at our costs.**
   - F3 = ρ × prior − cost drag must exceed rf ÷ min(12%, 2 × the arm's own volatility).
   - rf is French RF over the last 252 sessions of the pinned file.
4. **No contradiction.** The in-house net Sharpe's 95% interval (studentized circular
   block bootstrap, 42 sessions, 2,000 resamples) must not lie entirely below F3.

## 5. The combination (SPEC §4, Combinations)

- **Parts.** `ear_monthly`, if it passes criteria 1 and 2, and the promoted vol-scaled
  momentum (`results/2026-09-vol-momentum/series.csv`, pinned).
  - Two parts are required; otherwise the combination is not evaluated.
  - Construction and costs are the fundamentals combination's, unchanged: each part at
    12% volatility, equal weights, re-scaled to 12% within a total gross of 2.
- **Prior.** The window from momentum's 2015-04 to the public series' end is 2,455
  sessions (9.74 years at 252), short of ten. So by SPEC §4 the prior is 0.4 × the
  public combination's Sharpe over the whole common window (1971 on). This follows from
  the public series ending in 2024 and is stated before any of it is computed.
- **Economics.** ρ × prior − cost drag must exceed rf ÷ 12%.
- **No contradiction.** As in criterion 4.
- **Trial log.** One row in `trials/earnings_v1_combination.jsonl`.

## 6. The daily arms: discovery route

- **Kill on this sample** (`research/scripts/earnings_read.py`). An arm whose studentized
  95% interval for net periodic Sharpe lies entirely below zero is dead. The interval
  uses 42-session circular blocks, 2,000 resamples, seed 20260929 and daily decisions.
- **Reported beside it:**
  - each arm's daily correlation with vol-scaled momentum's in-house net return (§5.5
    judges a fast signal by its effect on the combined book);
  - N_eff and held-book IC from `registered_read`.
- **Forward read.** It uses `forward_v1`'s geometry and data rules (`docs/forward_design.md`
  §2–3): the first test window opens on or after 2026-09-24. The announcement panel is
  rebuilt at each read's as-of by `prism.scripts.build_earnings_panel`, with the
  definitions above.
  - **Interim**, at ≥ 126 forward sessions: the kill rule above, on forward sessions only.
    It cannot promote.
  - **Promotion**, at ≥ 252 forward sessions. All of these must hold on forward sessions
    only:
    - `net_edge` under bucket spreads;
    - DSR above 0.5, deflated against **three** rows (both daily arms and `ear_monthly`,
      since all three test the same announcement information);
    - periodic net Sharpe above the T-bill hurdle;
    - the live monitor not in contradiction.
    - The correlation with the momentum book's forward stream is reported beside it.
  - The forward reader is not built. It is due before the first interim read, about
    2027-03.

## 7. Horizon profile (SPEC §5.5)

For all three scores, `earnings_read` reports the cross-sectional rank IC of each
session's score with the open-to-open return over the next 1, 2, 5, 10, 21 and 63
sessions:
- the return starts at the next open (N2);
- the cross-section is that session's index members, with at least 30 names;
- the sessions run from 2021-03-30;
- the standard error is Newey–West with lag = horizon − 1.

It is a diagnostic, not a criterion.

## 8. Run order

```
python -m prism.scripts.construction_experiment --registration registrations/2026-09-earnings.toml \
    --dataset data/frozen/demotion_b1 --freeze
python -m prism.scripts.construction_experiment --registration registrations/2026-09-earnings.toml \
    --dataset data/frozen/demotion_b1 --arm ear_monthly --capital 100000
python -m research.scripts.registered_read --registration registrations/2026-09-earnings.toml \
    --dataset data/frozen/demotion_b1 --out results/2026-09-earnings
python -m research.scripts.earnings_replication

python -m prism.scripts.construction_experiment --registration registrations/2026-09-earnings-daily.toml \
    --dataset data/frozen/demotion_b1 --freeze
python -m prism.scripts.construction_experiment --registration registrations/2026-09-earnings-daily.toml \
    --dataset data/frozen/demotion_b1 --arm <arm> --capital 100000     # ear_event, announcement_premium
python -m research.scripts.registered_read --registration registrations/2026-09-earnings-daily.toml \
    --dataset data/frozen/demotion_b1 --out results/2026-09-earnings-daily
python -m research.scripts.earnings_read
```

## 9. Prior exposure, disclosed

- **The public series.**
  - The archive listing and the `AnnouncementReturn` member's columns and date range were
    read. Its first (1971-09-01) and last (2024-12-31) rows were printed while checking
    the schema.
  - The count of sessions from 2015-04 to 2024-12 (2,455) used dates only.
  - No statistic of the series has been computed.
  - Chen and Zimmermann's signal documentation, with the original paper's regression
    t of 9.3, was read.
- **The date check** used absolute market-adjusted returns only: no signed return, and
  no relation to any later return.
- **The panel's values** (announcement returns, expected dates) were computed and their
  coverage counted. They have not been set against any later return.
- **Known results:**
  - the fundamentals read: `niq_su` in-house net Sharpe 0.356, fidelity 0.29; JKP
    `niq_su` scaled Sharpe since 2015 0.354;
  - the vol-scaled momentum read;
  - the literature cited in §2.
- **Value weight was chosen before any return of either weighting was computed.**

## 10. Limits

- **Sample.** The in-house sample is 2021 → 2026, one regime family. The fidelity overlap
  ends 2024-12, about 880 sessions.
- **Universe.** S&P 500 large caps against all-stock, value-weighted quintiles.
- **Dates and market leg.**
  - Item 2.02 includes some non-earnings releases.
  - The market leg is equal-weighted over members; Chen and Zimmermann's is the CRSP
    value-weighted index.
- **Construction band.** The record reports the band as `closed_form`, but its width is
  0.0 on this path (`src/prism/evaluate.py:2515`), so nothing damps the daily arms'
  re-sorting. This is recorded, not fixed here; it is an audit item.
- **The combination is an overlay** of its parts' returns and costs.
- **Conventions.** Modeled fills and the price-return convention apply throughout.
- **Runtime.** Daily arms run about 20 times as many decisions as monthly ones. Their
  runtime is measured only for the freeze step (§3).

## 11. What this does not do

- It does not change the live book, arm any sizing, or edit SPEC §6.
- It does not use analyst forecasts, parse press releases, or search windows, weights or
  universes: each is pinned above, and a change is a new registration.
- It does not register a trade-timing overlay on the momentum book (SPEC §5.5). That is a
  later registration, which this panel makes possible.
