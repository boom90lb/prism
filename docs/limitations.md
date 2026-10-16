# Public limitations of every result this project has published

Updated: 2026-10-16. This page sets out what the published reference results do and do not
establish, using only evidence recorded in this repository. Read it before the numbers, not
after. Corrections to how recorded results are described are in
[`erratum_2026-10-14.md`](erratum_2026-10-14.md). In-house terms are explained in the
[glossary](glossary.md).

**The short version.** No configuration in this repository has met the project's conditions for
real money. One configuration, volatility-scaled momentum, was approved under the project's rules
on 2026-09-24. It cleared its T-bill benchmark by 0.018 Sharpe, a margin that rests on January
to July 2026, and it is not traded (section 10). Every published number comes from a simulation
with modeled fills, not from realized trading. The momentum reference record has a measured
coverage gap of nine percent of its universe, and it excludes dividends. No record charges a
cost that grows with assets, so none says anything about capacity (section 13). The
trend-following strategy was refused. The construction experiment was inconclusive on all
eighteen of its comparisons. The uncertainty method that experiment used is measured to
undercover. The public baseline is designed, and its registration and evaluation are pending.
Arithmetic on the recorded figures puts its mid-case edge over T-bills at about +1.26% a year
with idle equity in a T-bill ETF, and at −0.55% a year if idle cash earns nothing (section 12).
Each of these points is stated below with its artifact.

---

## 1. What none of it establishes

**No real-money track record.** Nothing here reports realized profit or loss from a funded
account. The conditions for real money are `SPEC.md` section 6, and opening, funding or resizing
a real-money account is a decision reserved to the account owner that has not been taken for any
of these configurations.

**Modeled fills throughout.** Every performance number below comes from a simulation that decides
at the close and fills at the next open under declared cost assumptions. The construction
comparison record carries this on its face: all eighteen of its runs are flagged
`fill_source=simulated; fills are not actual fills` in the record's own `warnings` list
(`results/2026-09-construction/comparison.json`). A modeled fill is not a fill.

**No configuration has reached the top evidence tier.** The tiers are `no_claim`,
`mechanics_clean` (the pipeline ran), `gross_edge` (profitable before costs), `net_edge`
(profitable after modeled costs) and `robust_edge` (also significant after the multiple-testing
adjustment), and `robust_edge` requires a deflated Sharpe ratio at or above 0.95
(`results/demotion_b1/claim_packet.json`, `claim_rules`). The best result here is at `net_edge`
with a deflated Sharpe ratio of 0.19.

**One published negative result already exists.** Daily residual reversion on the S&P
cross-section was found uneconomic at retail cost over a completed experiment of 17 trials, and
the finding was published with artifacts that may never be rewritten
(`docs/certifications/001-residual-reversion-daily-negative.md`). It is included here because a
limitations statement that lists only the surviving candidate is a selection effect in itself.

---

## 2. The momentum reference record: coverage, dividends, and a thin margin

The reference book is monthly 12-1 cross-sectional momentum on the S&P 500, recorded at
`results/demotion_b1/`.

**What it reports.** Net annualized Sharpe 0.465 (periodic 0.0293), total return 27.6 percent,
annualized 4.80 percent, maximum drawdown 14.0 percent, average turnover 5.0 percent, over 1,308
daily observations and 274 rebalances across 21 folds, 2020-01-01 to 2026-06-16, against a 3.71
percent nominal T-bill hurdle (periodic 0.0203). Claim tier `net_edge`
(`results/demotion_b1/summary.json`).

**Deflation.** Deflated Sharpe ratio 0.19 against the 17 configurations tried in the experiments
that produced it. That is the number to read, not the 0.465. A deflated Sharpe ratio of 0.19 says
the probability that this Sharpe survives the search that produced it is low, and the record says
so rather than reporting the raw figure alone.

**Coverage, which is the largest single limitation.** The universe resolved 574 of 633
ever-members over the membership window, a coverage fraction of 0.907, with 59 names skipped.
Recorded in `results/demotion_b1/config.json` under `universe.coverage`, and enumerated name by
name in `data/universe/sp500_coverage_2026-06-16.json`.

The reason the 59 are missing is mechanical and is the reason it matters: free vendors do not
always retain prices for delisted tickers, so the builder emits an explicit coverage ledger
rather than quietly dropping the dead names (`src/prism/io/universe_sp500.py`, module docstring
lines 11 to 16). Names absent for that reason are not a random nine percent of the index. They
are disproportionately the names that left it, by delisting or acquisition, which is exactly the
population whose exclusion flatters a backtest. The direction of the bias is therefore known even
though its size is not measured here. Nothing in this repository estimates what the record would
have been with full coverage; the honest statement is that the gap exists, is 9.3 percent of
ever-members, is enumerated, and is not corrected.

**Dividends are excluded.** The record's data convention is
`split_adjusted_open_close_price_return_no_dividends`
(`results/demotion_b1/claim_packet.json`, `data.data_convention`). It is a price-return record on
a long-short S&P book, so both dividends received on longs and dividend liabilities
on shorts are absent. Their net effect is not established by this price-return
record. This changes the return series, and the comparison to a cash hurdle
compares a price return against a total return.

**Sample.** Six and a half years, one market, one universe definition, one vendor. The bar source
is Twelve Data (`results/demotion_b1/claim_packet.json`, `data.source`), so vendor-specific
adjustment and revision behaviour is not controlled for.

---

## 3. Trend following: a negative result

Recorded verdict: **"positive-carry, non-convex; portfolio admission refused"**
(`docs/trend_t_series_read_2026-09-03.md`, run 2026-09-03, five counted trials T0 to T4 in
`results/trend_v1_trials.jsonl`): a small positive return, no protection in the momentum book's
bad periods, and so not added to the portfolio. Three facts from that report:

- **There is no standalone carry to speak of.** T0 reports net annualized Sharpe 0.042 and net
  total return +1.4 percent over 18.5 scored years, which is 0.08 percent a year and 6.6 percent
  of the cash hurdle it was measured against.
- **The signal's price leg is negative.** The +1.4 percent decomposes into distribution cash of
  +28.9 percent on a net-long book, a price leg of **-13.5 percent**, and -13.1 percent of cost.
  The nominal positive result is dividend and distribution income on a long bias, not trend
  timing.
- **The diversification test failed.** In the momentum book's worst decile of windows, the trend
  book returned -1.70 percent against an unconditional +0.32 percent over the same 62 windows, a
  shortfall of 2.02 percentage points at 2.9 standard errors, with a permutation p of about
  0.002. The trend book is positively exposed to the momentum book's drawdowns, which is the
  opposite of the property it was tested for. The sample contains no momentum crash, so even
  this is measured on a period that never posed the question hardest.

---

## 4. The construction experiment: a null result on all eighteen comparisons

The comparison of three construction variants (`decile_control`, `convex_point`,
`convex_p80_one_sided`) across six account sizes from $10k to $1m is recorded at
`results/2026-09-construction/comparison.json`.

**All eighteen Sharpe-difference intervals straddle zero.** Recomputed from the record on
2026-09-09 (Run): 18 of 18 intervals contain zero, with two-sided p-values from 0.262 to 0.999.
The widest point difference is 0.030 in periodic Sharpe against interval half-widths of 0.037 to
0.062. The experiment does not show that the convex constructor is better, and it does not show it
is worse. It shows the data cannot tell.

**No variant can be approved, by design.** The record's own `promotion_check` says it: "a
same-sample comparison can kill a configuration and can never promote one; the reference book's
promotion stays under the existing prospective rules". Reading the null as a licence to adopt
either constructor is a misreading of the design, not of the numbers.

**The intervals are not what they are often taken for.** The record's own method block states it:
"Endpoints are directional statements under the resampling assumptions ... They are not a
simultaneous interval and say nothing about realized portfolio return coverage." Eighteen
intervals at 0.95 each are not a 0.95 statement about the family.

---

## 5. The calibration bootstrap undercovers, and this was measured before the experiment ran

The uncertainty variant of the construction experiment uses a one-sided shrunk-slope endpoint at a
nominal confidence of 0.80. The declared block bootstrap does not achieve 0.80 on dependent data.

Measured over 120 replications at nominal 0.80 per side
(`tests/test_calibration.py::test_the_block_scheme_carries_dependence_an_iid_resample_misses`,
whose docstring records the runs): with AR(1) scores at rho 0.9 and MA(2) label errors, achieved
coverage is lower 0.725 and upper 0.758 with `block_periods = 3`; with an independent resample of
whole periods it falls to lower 0.675 and upper 0.692; and on the frozen reference fold geometry
(13 periods of 96 names, 5 blocks) it is lower 0.708 and upper 0.700. So **at nominal 0.80 the
achieved one-sided coverage is roughly 0.71 to 0.76 on dependent data, and about 0.70 at the
reference fold geometry.** The test asserts the shortfall rather than hiding it
(`assert blocked_upper < 0.80, "and it still undercovers; that is reported, not repaired"`).

The declared block scheme widens the interval by about a third under dependence relative to an
independent resample, so it recovers part of the shortfall. It does not close it. The p80
endpoints in the construction experiment are therefore optimistic, and were known to be
optimistic before the experiment was run. This is reported as a property of the declared method,
not patched after seeing an outcome.

---

## 6. A constructor honouring a limit is not the portfolio satisfying it

Deterministic quantity rounding runs after the solve, and it can introduce a breach of a
preference limit the solver honoured exactly.

Recomputed on 2026-09-15 (Run) over exactly the eighteen ids in
`results/2026-09-construction/comparison.json` → `runs.read_by_this_comparison`:
918 decisions, of which **111, or 12.1 percent, introduced a preference excess
through rounding**. The precise field is
`evaluation.construction.decisions[].rounding.validation.violations.introduced`
in each run's `record.json`. All 136 excesses there are attributed to `rounding`.
By limit: `max_gross` 53, `net_band` 43, `max_name_weight` 25, `variance_budget`
15. Magnitudes ran from 2.5e-06 to 6.7e-02 in each limit's own units, median 2.6e-03.

The previously reported 112 decisions / 138 excesses traversed every nested
`introduced` list. It also included one separate heuristic `max_gross` excess
at decision index 44 of `run-333d02e707af4c6f`, repeated in both
`diagnostics.validation.violations` and the decision's top-level `violations`.
That is one constructor violation recorded twice, not two additional rounding
excesses. The correction changes no run, trial row, or reference policy.

The general point recurs wherever continuous construction is followed by integer rounding: a
constructor enforcing a limit exactly and the implemented portfolio satisfying that limit are
different claims, and only the first is inside the solver's reach. The excesses are recorded in
the run artifacts rather than suppressed, which is why they can be counted here at all.

---

## 7. What cannot currently be checked at all

**Whether an engine upgrade changes the recorded sessions' decisions.** The upgrade comparison is
real and tested, but the recorded paper capture carries no bar panels, so the candidate side
cannot be produced: 0 of 5 sessions compare end to end, and the engine overhaul's parity check
reports decision parity NOT COMPARED (`docs/engine_overhaul_parity_gate.json`). What is demonstrated
instead is order-sizing and extraction parity on those sessions. The distinction, and what would
close it, is `docs/engine_overhaul_operating_changes.md` section 1.3; the user-facing consequence
is `docs/versioning.md` section 3.

**The published-factor evaluations, byte for byte.** The volatility-scaled momentum approval and
the public factor lab read Kenneth French's daily files of the 202607 vintage. French rewrites
those files in place at fixed URLs, the 202607 bytes are no longer served, and the Internet
Archive holds no copy of them (capture index checked 2026-10-14). The two registered drivers
behave differently from a clone. The volatility-scaled momentum driver
(`research/scripts/vol_momentum_replication.py`) refuses any file hash other than the registered
one, so it cannot be rerun. The factor-lab driver (`research/scripts/factor_lab.py`) checks no
hash: it downloads whatever vintage is served and records its hash. Rerunning any of its parts
other than `--part fetch` rewrites the tracked `results/2026-09-factor-lab/sources.json`, and the
`priors`, `fidelity` and default `all` parts also rewrite `priors.json`, `correlations/` and
`fidelity.json` there, with numbers from another vintage. The `sizing` part refuses a second run at its ledger
check, but only after `sources.json` has been rewritten. Do not rerun either driver to check the
record. What a clone can check is
`python -m research.scripts.reproduce_public` (`docs/public_reproduction_design.md`). Offline, it
recomputes the four recorded criteria from the tracked series, rebuilds the in-house side from
its tracked inputs, and checks that the recorded reference follows the registered rule. With
network access, it downloads the current files into a directory of the caller's choosing and
measures how far they differ from the record. Its output is a reproduction, not a new evaluation:
it adds no ledger row and changes no verdict, and a number computed on a later vintage describes
that vintage only.

**The minimum supported live account size.** Undetermined. The engine design brief
(`docs/prism_engine_decisions.md`, Q22) declines to encode a figure from one synthetic rounding
exercise, and the account sizes in the construction comparison are scenarios, not a supported
range. What is measured is narrower: whole-share rounding holds the roughly 100-name momentum
book faithfully at about $100k and trades a structurally different portfolio at $10k
(`docs/account_size_floor.md`).

---

## 8. The paper-trading record does not execute or accrue like the backtests (erratum E11)

The live paper account is the only forward record. It differs from the simulated books in how
orders fill and in which cash flows it books.

**Most paper fills are not opening-auction fills.** The backtests fill every order at the next
session's open. The tracked capture of the paper run (`tests/fixtures/paper_replay/fills.jsonl`,
the rebalances decided 2026-07-13 and 2026-08-11) has 249 fills. Of these, 185 (74%) are market
orders sent at 09:50 ET by the morning completion sweep, identified by client order ids ending in
`:S1` (`src/prism/scripts/paper_sweep.py`, `ops/paper_sweep_morning.sh`): 91 of 119 in July and
94 of 130 in August. The sweep exists because the paper venue fills only part of the
market-on-open orders. Paper fills therefore price most of each rebalance about twenty minutes
after the open, and the gap between paper and simulated fills mostly measures that delay, not
auction cost. Alpaca's documentation also lists "Market impact of your orders" among the things
paper trading "does not account for", so paper fills cannot measure impact at all. How
real-money market-on-open orders would fill is not known.

**Paper accounts book no dividends and no borrow fees.** Alpaca's paper-trading page
(docs.alpaca.markets/docs/paper-trading, read 2026-10-14, last modified 2026-07-07) states that
the paper account "does NOT simulate dividends", and its paper-versus-live table marks borrow fees
as not available on paper ("Coming Soon!"). Two consequences:

- Paper equity omits dividends received on long positions and dividends owed on short ones. For
  the momentum reference book that net flow was measured at −36.1 bp a year
  (`docs/program_review_2026-07.md`; erratum E3), so paper equity overstates a real account's
  return by about that much. A T-bill ETF held on paper would show its price falling at each
  distribution and no income.
- Paper equity omits stock-loan fees on the short side, which the backtests charge (50 bp a year
  in `examples/reference/costs/retail_v2.toml`). Paper therefore looks cheaper than both the
  backtest and a real account.

Anything that reads paper equity as the broker reports it, such as the live monitor
(`prism.live.monitor.paper_monitor_read`), sees these uncorrected figures. A forward evaluation
of the paper stream has to add the dividend flows and charge modelled borrow before it compares
the stream with a backtest or with a T-bill hurdle.

---

## 9. Standing conditions on any published number

- Compare returns with returns. The Sharpe figures here are on the excess-return conventions each
  record declares; a Sharpe is not comparable with a Treasury yield.
- Labels are not interchangeable. Synthetic demonstration, exploratory historical analysis,
  prospective evaluation, paper fills and actual fills are distinct categories and every record
  carries which one it is.
- Where an uncertainty estimate or a search history is not established, the record says
  `unavailable` rather than omitting the field.
- The licence is MIT and carries no warranty. Nothing in this repository is investment advice, and
  no result here is a forecast of future return.

---

## 10. Volatility-scaled momentum: approved by a thin margin, not traded

`vol_momentum_v1` scales the momentum book to a 12 percent volatility target by the Barroso and
Santa-Clara (2015) rule (126-session window, gross capped at 2×). It was approved on 2026-09-24
under the published-factor route of `SPEC.md` section 4, which judges a published factor by its
long public return series rather than by an in-house backtest
(`docs/vol_momentum_read_2026-09-24.md`; criteria in `results/2026-09-vol-momentum/summary.json`).

- **The approval figure.** Tracking correlation 0.895 with the scaled public momentum series ×
  post-publication Sharpe 0.465 (2015-04 to 2026-07, t 1.56 over 11.3 years) − cost drag 0.047 =
  0.368, against a T-bill benchmark of 0.351 computed as if idle cash earns nothing. The margin is
  0.018 Sharpe, about 0.2 percent a year at the 12 percent target volatility. The results report's
  "about half a percent a year" over T-bills (its section 3) is on another basis: the 4.7 percent a
  year that 0.368 gives at the book's 12.7 percent realized volatility, less 4.2 percent bills.
- **It rests on seven months.** Cut at 2025-12, the post-publication Sharpe is 0.401 and the figure
  would be 0.312, below the benchmark (`docs/fundamentals_read_2026-09-24.md`; erratum E6). The
  pinned Ken French file has since been replaced by a newer vintage, so the registered inputs
  cannot be downloaded again.
- **The margin depends on what idle cash earns.** Restated at the provisional 3.3 percent broker
  cash yield, the benchmark is 0.224 and the margin about 0.14 Sharpe, roughly 1.7 percent a year
  at 12 percent volatility, or about 2.0 percent on the results report's 12.7 percent basis, its
  addendum's "about 2%" (`docs/hurdle_restatement_2026-09-30.md`). That yield is a listed rate,
  not one taken from an account statement.
- **It is not traded.** The paper account trades the unscaled momentum book. Arming the scaling
  rule, and replacing the crash-conditional sizing rule `SPEC.md` section 6 still names, are
  decisions reserved to the account owner. Its stop rule runs on later data; the first forward
  evaluation is due about 2027-03.

---

## 11. Published-factor and comparison results since the momentum record

None of these produced an approved configuration. Each results report states its own limits.

- **Sizing rules on public data** (`docs/public_factor_lab_read_2026-09-24.md`). On Kenneth French
  and JKP daily data, the crash-conditional exposure cut named in `SPEC.md` section 6 lowers Sharpe
  before 2020 and holds half gross on about 45 percent of sessions; constant-volatility scaling is
  supported. Post-publication Sharpe: momentum 0.35, residual momentum 0.18. Residual momentum's
  in-house tracking correlation, 0.495, is below the floor of 0.5.
- **Fundamentals** (`docs/fundamentals_read_2026-09-24.md`). Six accounting-based factors and one
  combination. Tracking correlation × post-publication Sharpe is 0.08 to 0.24 per factor against
  benchmarks of 0.35 to 0.48, before any cost.
- **Earnings announcements** (`docs/earnings_read_2026-09-24.md`). The monthly announcement return
  fails economics (0.117 against 0.442). Its combination with volatility-scaled momentum fails
  after costs: 0.275 against 0.351, and against 0.321 at the provisional 3.3 percent cash yield
  with the corrected long side (erratum E10). Two daily variants carry to a forward evaluation.
- **Rank buffer** (`docs/buffer_read_2026-09-25.md`). Turnover fell 12 to 36 percent; paired
  net-Sharpe differences of −0.09 to +0.07 are indistinguishable from zero.
- **Comparisons against the momentum control** (`docs/characteristics_read_2026-09-23.md`,
  `docs/breadth_read_2026-09-23.md`, `docs/learned_xsection_read_2026-09-23.md`). Residual momentum
  could not be told apart from the control (p 0.79); adding the S&P 400 raised effective breadth
  37 percent and not net Sharpe; a learned ranking was explained by momentum. These are backtests
  on data the project had already studied, which can reject a configuration and can never approve
  one. The two residual-momentum candidates carry to a forward evaluation
  (`docs/forward_design.md`).

---

<a id="public-baseline"></a>

## 12. The public baseline: designed, registration and evaluation pending

**What it is.** `public_baseline_v1` is the configuration this repository intends to offer as its
public baseline. It is the volatility-scaled momentum book of section 10, with its signal,
universe and parameters unchanged, split into 21 overlapping tranches. Each tranche holds 1/21 of
capital and rebalances every 21 sessions on its own date, so exactly one tranche rebalances
each session. A new deployment starts with empty tranches and reaches full size over 21 sessions.
Idle own equity above a 2% cash buffer is held in a T-bill ETF. Short-sale proceeds earn nothing.

**Status.** No version of it has been run on market data. Its registration, a design document
and parameter file committed before any evaluation (`SPEC.md` section 4), is not yet in the
repository. Until it is committed and the evaluation runs, nothing in this section is a result.

**What to expect, by arithmetic.** The table uses a normal approximation. Every input is a recorded
figure or a stated assumption, and every output is derived from them; no market data was read to
make it. The recorded inputs are:

- the volatility-scaled momentum record, `results/2026-09-vol-momentum/summary.json`:
  - post-publication Sharpe 0.4646, 2015-04 to 2026-07, t 1.562 (`publication_bar.post_publication`);
  - full-sample Sharpe 0.9013 (`publication_bar.full`);
  - daily tracking correlation 0.8951 (`fidelity.daily_corr`);
  - cost drag 0.0474 Sharpe (`economics.cost_drag_sharpe`);
  - rf 4.21% (`economics.rf_annual`);
  - mean scale 1.0786 (`in_house.mean_scale`), so the long side averages 0.539 of equity;
- post-publication Sharpe 0.401 when the series is cut at 2025-12
  (`docs/fundamentals_read_2026-09-24.md`);
- the provisional broker cash yield of 3.3% (`examples/reference/costs/retail_v2.toml`,
  `cash_income_bps_annual`).

The assumptions are:

- volatility of 12%;
- a T-bill ETF return of rf less 10 bp. SGOV's expense ratio is about 0.09% according to secondary
  sources, not yet checked with the issuer;
- a 2% cash buffer earning nothing, so the T-bill holding is 1 − 0.539 − 0.02 = 0.441 of equity;
- an allowance for fills that come later than the open, because 74% of paper fills are 09:50 ET
  sweep orders (section 8);
- in the low case, a tracking correlation and cost drag worse than recorded.

| Derived | Low | Mid | High |
|---|---|---|---|
| Post-publication Sharpe used | 0.361 (0.4 × 0.901) | 0.41 (between 0.401 and 0.465) | 0.465 |
| × tracking correlation | 0.85 (assumed) | 0.895 | 0.895 |
| − cost drag | 0.055 (assumed) | 0.047 | 0.047 |
| − late-fill allowance (assumed) | 0.03 | 0.015 | 0.005 |
| **Expected net Sharpe of the long–short book** | **0.22** | **0.30** | **0.36** |
| Long–short return at 12% volatility | 2.7%/yr | 3.7%/yr | 4.4%/yr |
| Excess over bills, idle equity in the T-bill ETF (benchmark 0.200) | +0.26%/yr | **+1.26%/yr** | +1.96%/yr |
| Excess over bills, idle cash at a 3.3% broker yield (benchmark 0.224) | −0.03%/yr | +0.97%/yr | +1.67%/yr |
| Excess over bills, idle cash earning nothing (benchmark 0.351) | −1.55%/yr | **−0.55%/yr** | +0.15%/yr |
| Probability the true expected excess is positive, T-bill ETF | 0.53 | 0.65 | 0.73 |
| Probability the true expected excess is positive, idle cash earning nothing | 0.31 | 0.43 | 0.52 |
| Years of a deployment's own returns needed to reject a zero Sharpe (one-sided 95%) | 55 | 29 | 20 |

How to read it:

- **The account's edge, not the book's Sharpe.** In the mid case the long–short book earns 3.66%,
  and the T-bill holding earns 0.441 × 4.11% = 1.81%. The account earns 5.47% against bills at
  4.21%, which is +1.26% a year. The holding adds almost no volatility, so the account's
  excess-over-bills Sharpe is about 1.26% / 12% ≈ **0.10**. The 0.30 long–short Sharpe is not the
  edge over bills.
- **The benchmarks.** They follow the formula in `SPEC.md` section 3: (rf − y·max(0, 1 − L/E)) /
  volatility. With the T-bill holding, the ETF counts as a long position earning rf less its fee,
  and only the 2% buffer is idle: (4.21% − 4.11% × 0.441) / 12% = 0.200. The account owner has not
  yet confirmed this reading of section 3, and the design calls for every verdict to be reported
  under all three cash cases.
- **Uncertainty.** The standard error of the post-publication Sharpe is 0.465 / 1.562 = 0.297.
  After tracking it is 0.266 Sharpe, about ±3.2% a year at 12% volatility, which is larger than
  the mid-case edge. Distinguishing the mid-case margin over the benchmark (0.105 Sharpe) from zero
  would take about 250 years of the deployment's own returns.
- **Break-even.** To break even, the post-publication Sharpe must be at least (benchmark + cost
  drag + allowance) / tracking. That is 0.29 with the T-bill holding, 0.32 at a 3.3% cash yield,
  and 0.46 if idle cash earns nothing. The recorded figures are 0.401 to 0.465. With idle cash
  earning nothing, the requirement equals the approved figure, and the outcome is a coin flip.
- **Rates.** With the T-bill holding, the benchmark is 0.143 at rf 3% and 0.237 at rf 5%.
- **Decay.** Momentum was published in 1993. Kenneth French's momentum factor had a Sharpe of 0.629
  before publication and 0.346 after it, a fall of 45% (`results/2026-09-factor-lab/priors.json`,
  `priors.UMD`). The scaling rule, published in 2015, is the live decay risk. Its post-publication
  Sharpe is 0.465 against 0.901 over the full sample, and the low case allows a further fall to
  0.36. Tranching leaves the average age of the signal unchanged. Momentum's information
  coefficient by horizon has never been computed on this engine (`docs/buffer_read_2026-09-25.md`).
- **Not applied.**
  - Taxes. A reference overlay at a 37% federal and 9.3% state rate puts the drag from tax
    asymmetries, beyond ordinary tax on gains, at about −37 bp a year
    (`results/tax_wedge_2026-07-19.json`, `outputs.steady_state_wedge_bps_per_year`).
  - Survivorship. The in-house tracking and cost figures come from a universe with 0.907
    membership coverage (section 2), which flatters both.
  - Deflation for choosing the scaling rule. It was chosen in a sizing experiment of 8
    configurations (`sizing_lab_v1`, `docs/public_factor_lab_read_2026-09-24.md`).
  - In-house backtest Sharpes on the data the rule was chosen on are never used as expectations.
- **Paper accounts.** The T-bill ETF's distributions do not appear on paper (section 8). A paper
  account would show its price falling at each distribution and no income. A forward evaluation
  of the paper stream must add them back.

**What cannot be promised.**

1. *A high Sharpe.* Nothing on record supports a forward net Sharpe above about 0.36 for the
   long–short book, or above about 0.16 for the account's excess over bills. A forward 0.5 would
   need a post-publication Sharpe of at least 0.63 at the recorded tracking and cost, and no
   recorded momentum series shows one.
2. *Beating T-bills with confidence.* The mid case is positive only if idle equity earns close to
   the bill rate, through a T-bill ETF or a verified broker yield of about 3.3% or more in a margin
   account that holds shorts. Even then the probability is about 0.65.
3. *Confirmed low decay.* A deployment's own returns cannot confirm it on any useful horizon: about
   29 years against zero, and about 250 years against bills. Decay can only be argued in advance
   from long public series and checked again on each new release of the public data.
4. *Deployments that do not compete.* See section 13.

---

## 13. Capacity and crowding: no record measures them

**No recorded cost grows with assets.** Market impact that depends on an order's size relative to
the market's traded volume is the term that would bring in capacity. It is
`adv_impact_coeff` in the older records and `participation_coeff` in the engine's cost stacks.
Its coefficient is 0.0 everywhere it is recorded. Checked on 2026-10-16 (Run): all 166 such
fields across the 65 tracked JSON and JSONL files under `results/` and `trials/` that carry one
are 0.0. So are both shipped cost stacks (`examples/reference/costs/retail_v1.toml` and
`examples/reference/costs/retail_v2.toml`, `participation_coeff = 0.0`). The remaining impact
term is `quadratic_notional_coeff` (`slippage_coeff` in the older records). It is quadratic in an
order's size relative to the account's own equity (`prism.execution.cost_model`), so modeled cost
per dollar traded is the same at every account size. The momentum record was simulated on an
initial capital of 1.0 (`results/demotion_b1/summary.json`, `initial_capital`). No recorded figure
therefore says how much capital the strategy can hold, or what other users' orders cost it.

**Every deployment computes the same portfolio.** The universe file is tracked, the parameters are
fixed and the decile rule is deterministic. The paper loop sets its rebalance date from each
deployment's first session (`src/prism/live/daily.py`, `_is_refresh_session`). Its rebalance
orders are market-on-open orders (`src/prism/live/alpaca.py`), and the engine knows only
market-on-open and market-day orders (`src/prism/execution/portfolio_state.py`, `ORDER_TYPES`).
Users who start on the same day therefore send the same orders into the same opening auction.
Spreading users over different days spreads their order flow, but their holdings still overlap
almost completely. A momentum unwind hits every deployment on the same day.

**Timing luck is unmeasured.** Every record uses a single rebalance phase. How much the outcome
depends on which sessions a monthly book happens to rebalance on has not been measured here.

**An illustration, not a measurement.** The table assumes square-root impact. The annual
crowding cost, as a fraction of capital, is

    8.5 × Y × σ × √(0.011 × F × φ / (a × V))

with these inputs:

- 8.5: times capital traded a year (assumed);
- Y = 0.7: impact coefficient (assumed);
- σ = 1.8%: daily volatility (assumed);
- 0.011: each order in a name is one decile-book position of 1.1% of capital (assumed);
- F: the combined capital of users who trade together;
- φ: 1 when every user rebalances on the same day, 1/21 with 21 tranches;
- a: the share of the day's volume available at the execution time, all assumed: 1% or 3% in
  the opening auction, 8% in the closing auction, 100% over the full day;
- V: the name's 20-session median consolidated dollar volume as of 2026-06-15
  (`results/iex_eligibility_2026-07-17.json`, `per_name.*.cons_med_asof`). Over the 501 names
  with a value, the median is $356M and the 10th percentile is $142M.

Each entry is the F at which that cost equals the mid-case +1.26% a year of section 12. In the
split row each half of an order trades at its own share of volume.

| Case | Median name ($356M/day) | 10th-percentile name ($142M/day) | Users at $100k (10th-percentile name) |
|---|---|---|---|
| Every user on the same day, opening auction at 1% of volume | $4.5M | $1.8M | 18 |
| 21 tranches, opening auction at 1% | $94M | $37M | 370 |
| 21 tranches, opening auction at 3% | $281M | $112M | 1,100 |
| 21 tranches, half at the open (1%), half at the close (8%) | $409M | $163M | 1,600 |
| 21 tranches, all at the close (8%) | $749M | $299M | 3,000 |
| 21 tranches, full-day execution | about $9.4B | about $3.7B | about 37,000 |

Above these levels users compete whatever the timing scheme. The engine has no closing-auction
order and no way to spread an order over the day: its only orders are market-on-open and plain
market orders. The last three rows therefore describe execution it cannot do today. Three things
could raise the limits:

- more liquidity per dollar traded: legs weighted by liquidity, full-day execution, or a larger
  share in the closing auction;
- a second strategy with a positive expected excess over bills at retail cash terms. None on
  record has one (section 11);
- broker terms that pay interest on short-sale proceeds.
