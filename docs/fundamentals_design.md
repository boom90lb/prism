# Fundamentals — design and registration (`fundamentals_v1`, replication route)

**Status: registered 2026-09-24.** The registration is the commit that adds this
file with `registrations/2026-09-fundamentals.toml` (the six in-house arms),
`registrations/2026-09-fundamentals-replication.toml` (the criteria),
`examples/fundamentals/control.toml` (the component book) and
`research/scripts/fundamentals_replication.py` (the read). No fundamental book's
return has been computed at registration.

## 1. What is registered

**Arms.** Six published accounting characteristics, each run as its own monthly
top/bottom-decile long/short book on the point-in-time S&P 500:

| arm | characteristic | source (publication) | long leg |
|---|---|---|---|
| `be_me` | book-to-market | Fama and French 1992 (1992-06) | high |
| `gp_at` | gross profitability | Novy-Marx 2013 (2013-04) | high |
| `at_gr1` | asset growth | Cooper, Gulen and Schill 2008 (2008-08) | low |
| `oaccruals_at` | operating accruals | Sloan 1996 (1996-07) | low |
| `niq_su` | earnings surprise | Foster, Olsen and Shevlin 1984 (1984-10) | high |
| `chcsho_12m` | net share issuance | Pontiff and Woodgate 2008 (2008-04) | low |

**Combination.** The arms that pass SPEC §4 criteria 1 and 2, together with the promoted
volatility-scaled momentum (`vol_momentum_v1`), each at 12% volatility and equal weight
(§5).

Each arm and the combination is decided on the replication route (SPEC §4).

## 2. Why, and what to expect

Characteristics built from the daily price path are saturated on this sample (SPEC §5.3).
Accounting data is new information: value moves against momentum, correlation about
−0.18 in 2000–2025 (`docs/public_factor_lab_read_2026-09-24.md`).

The public record sets modest expectations. The factor lab's post-publication Sharpes
were HML 0.21, RMW 0.24 and CMA 0.03. The hurdle at a 12% volatility target is about
0.35 (T-bills at ~4.2%). **Most arms are expected to fail the hurdle on their own.** The
combination is the point: several small, weakly correlated premia at equal risk, next to
momentum, on the evidence the amended SPEC §4 allows (public long-sample data).

## 3. Data and implementation

- **Fundamentals.** SEC EDGAR XBRL, point in time by filing acceptance
  (`prism.io.edgar`; store built by `prism.scripts.build_fundamentals`).
  - Tickers are mapped to filers through SEC's current ticker map, validated company
    names, and owner-approved review decisions
    (`data/universe/cik_review_decisions_2026-09-24.json`,
    `cik_nearmiss_decisions_2026-09-24.json`).
  - 98.7% of post-2009 index tickers are mapped; S&P 500 coverage is 98.6% at
    2022-03-29.
- **Characteristics.** Defined once in `prism.io.fundamental_panel`.
  - Book equity is stockholders' equity (book-to-market is NaN when it is ≤ 0).
  - Gross profit is reported gross profit, or revenue less cost of revenue.
  - Fiscal-year values are the latest annual net-income period; balance-sheet values
    are the instants at its end.
  - Accruals take the cash-flow form: (net income − operating cash flow) / assets.
  - Earnings surprise is on quarterly net income: the year-on-year change over its
    8-quarter standard deviation, with at least 6 quarters required.
- **Market equity and splits.** Reported share counts are carried by every split
  between their date and the dataset's price basis. Splits come from Alpaca's
  corporate actions, frozen as `data/frozen/fundamentals_v1/splits.parquet` (114
  splits).
- **Point in time and deduplication.** A value for session *t* uses only filings
  accepted by *t*'s close. Each company is counted once per session: index members
  first, then the more liquid security.
- **Frozen panel.** `data/frozen/fundamentals_v1/panel.parquet`, sha256
  `06ea92de…5df3`. It covers 565 securities and 561 companies, 2020-01-02 →
  2026-06-15. Median daily coverage: asset growth, accruals and earnings surprise 99%;
  issuance 94%; book-to-market 91%; gross profitability 61% (banks and insurers report
  no cost of revenue). The freeze record in `freeze.json` hashes every input.
- **Node.** `prism.signal.fundamental_node:FundamentalCharacteristicNode` checks the
  panel's hash, signs each characteristic to its published direction, and refuses
  decision dates outside the panel.
- **Engine runs.** Identical to the momentum control of
  `registrations/2026-09-characteristics.toml`:
  - frozen dataset `data/frozen/demotion_b1`, folds `folds/demotion_b1.toml`;
  - cost stack `retail_v1`, execution profile `execution_alpaca`, canonical engine,
    $100,000;
  - the same eligibility screen and construction.

  Scored sample: 1,056 sessions, 2022-03-29 → 2026-06-12 (folds 0–3 excluded by the
  shared admission rule, as before). A mechanics check of the engine's freeze step,
  written to scratch, scored all six nodes on the 63 decision rows (2021-03-30 →
  2026-06-05) with no warnings. That check computes scores, not returns.

## 4. Per-arm criteria (SPEC §4, replication route)

The reference for each arm is JKP's daily US series for the characteristic: capped
value weight, direction-adjusted, pinned by sha256. They run from 1950–1964 (1927 for
issuance) to 2025-12-31.

1. **Publication bar and prior.**
   - The bar is t ≥ 3 over ≥ 20 years.
   - The prior is the post-publication Sharpe (every source is ten or more years old).
   - A prior ≤ 0 fails.
2. **Fidelity.**
   - Criterion: the daily correlation of the in-house arm's net returns with the JKP
     series, over 2022-03-29 → 2025-12-31, must be ≥ 0.5.
   - A negative correlation is recorded as a direction mismatch and fails.
   - Weekly correlation is reported beside it.
3. **Economics at our costs.**
   - Figure: F3 = ρ × prior − cost drag. The drag is the arm's charged cost over its
     volatility, both annualized.
   - F3 must exceed the hurdle rf / min(12%, 2 × the arm's own volatility). That is the
     T-bill rate (French RF, last 252 sessions of the pinned file) at the volatility the
     arm can run at within the Reg T 2× gross ceiling.
4. **No contradiction.** The in-house net Sharpe's 95% interval (studentized circular
   block bootstrap, 42-session blocks, 2,000 resamples) must not lie entirely below F3.

An arm passing all four is promoted, which makes it eligible for the live book. Changing
the live book is the owner's act.

## 5. The combination (SPEC §4, Combinations)

- **Parts.**
  - Every arm passing criteria 1 and 2, plus vol-scaled momentum. Both criteria read
    public data and tracking, never in-house performance.
  - At least three parts are required, momentum among them; otherwise the combination
    is not evaluated.
  - Momentum enters through its promoted record
    (`results/2026-09-vol-momentum/series.csv`, sha256 pinned): in-house scaled net
    returns and costs, and scaled French momentum as its public reference.
- **Construction.**
  - Each fundamental part is scaled to 12% volatility by momentum's rule: 126-session
    volatility, re-set every 21 sessions, scale capped at 2, 10 bp one-way per scale
    change.
  - The parts are averaged at equal weights, which is equal risk.
  - The average is scaled to 12% again by the same rule, with its scale capped so
    that total gross (Σ weight × part scale × part gross) stays within 2.
  - Costs are the parts' own, averaged with no credit for trades that would net across
    parts (conservative).
  - The public combination is built the same way from the parts' public series.
- **Criteria.**
  - Prior: the public combination's Sharpe from the latest publication among the parts
    to the common end (2025-12-31), if that spans ten years or more, else 0.4 × its
    Sharpe over the whole common window. With momentum's 2015-04, the window is
    10.7 years.
  - Economics: ρ (daily correlation of the in-house and public combinations) × prior −
    in-house cost drag must exceed rf / 12%.
  - No contradiction: as in criterion 4, on the in-house combination.
- **Trial log.** One row in `trials/fundamentals_v1_combination.jsonl`, kept apart from
  the engine's six-arm budget.

## 6. Run order

```
python -m prism.scripts.construction_experiment --registration registrations/2026-09-fundamentals.toml \
    --dataset data/frozen/demotion_b1 --freeze
python -m prism.scripts.construction_experiment --registration registrations/2026-09-fundamentals.toml \
    --dataset data/frozen/demotion_b1 --arm <arm> --capital 100000        # each of the six arms
python -m research.scripts.registered_read --registration registrations/2026-09-fundamentals.toml \
    --dataset data/frozen/demotion_b1 --out results/2026-09-fundamentals
python -m research.scripts.fundamentals_replication
```

## 7. Forward kill rule

The rule covers the arms and the combination, run forward on sessions from 2026-09-24
with the `forward_v1` geometry. The first read is at ≥ 126 forward sessions, then every
63. Any one of these kills:

- the forward net Sharpe's 95% interval lies entirely below zero;
- drawdown exceeds 25%;
- daily correlation with the reference falls below 0.5.

The forward reader is not yet built and is due before the first read, about 2027-03.

## 8. Prior exposure, disclosed

- **Theme-level public results are known.** The factor lab computed JKP theme-level and
  French HML/RMW/CMA Sharpes and correlations (§2).
- **Characteristic-level public series have not been examined.** The six JKP series
  were fetched on 2026-09-24; only their schemas, date ranges and direction flags were
  read.
- **The panel was checked on values, not returns.** Market equity, book equity and
  characteristic values were checked for about eight companies on 2024-06-28, plus
  cross-sectional medians. No return of any fundamental book or JKP characteristic
  series has been computed.
- **The in-house momentum book's results are known** (net Sharpe 0.68; vol-scaled 0.66;
  `docs/vol_momentum_read_2026-09-24.md`).

## 9. Limits

- The in-house sample is 2022–2026, one regime family.
- The in-house books are S&P 500 large caps. The JKP references are all-stock, capped
  value-weight terciles, and fidelity measures how far apart they are.
- The combination is an overlay of the parts' returns and costs, not one engine book.
  Netting across parts would lower its costs.
- Gross profitability misses most financials.
- The universe files list some renamed companies under two tickers at once. The panel
  counts each company once per session, but the price data's own treatment of those
  pairs is not changed here.
- Modeled fills and the price-return convention apply throughout.

## 10. What this does not do

- It does not change the live book, arm any sizing, or edit SPEC §6.
- It does not search characteristics, windows, weights or universes: each is pinned
  above, and a change is a new registration.
- It does not test post-earnings drift around announcement dates, factor or industry
  momentum, or any faster signal. Those are later registrations.
