# Public limitations of every result this project has published

Updated: 2026-09-15. This is the statement the engine's stage 6 exit requires: what the published
reference results do and do not establish, drawn only from evidence recorded in this repository.
It is written to be read before the numbers, not after. Corrections to how recorded results are
described are in [`erratum_2026-10-14.md`](erratum_2026-10-14.md).

**The short version.** No configuration in this repository has cleared the project's own evidence
bar. Every published number comes from a simulation with modeled fills, not from realized
trading. The one reference result with a positive verdict has a measured coverage gap of nine
percent of its universe and excludes dividends. The second sleeve was refused. The construction
experiment was inconclusive on every one of its eighteen comparisons. The uncertainty method used
in that experiment is measured to undercover. Each of those is stated with its artifact below.

---

## 1. What none of it establishes

**No real-money track record.** Nothing here reports realized profit or loss from a funded
account. The deployment gate is `SPEC.md` section 6, and opening, funding or resizing a
real-money account is an owner act that has not been taken for any of these configurations.

**Modeled fills throughout.** Every performance number below comes from a simulation that decides
at the close and fills at the next open under a declared cost stack. The stage 4 comparison
record carries this on its face: all eighteen of its runs are flagged
`fill_source=simulated; fills are not actual fills` in the record's own `warnings` list
(`results/2026-09-construction/comparison.json`). A modeled fill is not a fill.

**No configuration has reached the top claim tier.** The tiers are `no_claim`,
`mechanics_clean`, `gross_edge`, `net_edge`, `robust_edge`, and `robust_edge` requires a Deflated
Sharpe Ratio at or above 0.95 (`results/demotion_b1/claim_packet.json`, `claim_rules`). The best
result here is at `net_edge` with a DSR of 0.19.

**One certified negative already exists.** Daily residual reversion on the S&P cross-section was
certified uneconomic at retail cost over a closed 17-trial selection set
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

**Deflation.** DSR 0.19 against a 17-trial selection set. That is the number to read, not the
0.465. A DSR of 0.19 says the probability that this Sharpe survives the search that produced it
is low, and the record says so rather than reporting the raw figure alone.

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

## 3. The trend sleeve: a negative result

Registered verdict: **"positive-carry, non-convex; portfolio admission refused"**
(`docs/trend_t_series_read_2026-09-03.md`, run 2026-09-03, five counted cells T0 to T4 in
`results/trend_v1_trials.jsonl`). The sleeve was not adopted. Three facts from that read:

- **There is no standalone carry to speak of.** T0 reports net annualized Sharpe 0.042 and net
  total return +1.4 percent over 18.5 scored years, which is 0.08 percent a year and 6.6 percent
  of the cash hurdle it was measured against.
- **The signal's price leg is negative.** The +1.4 percent decomposes into distribution cash of
  +28.9 percent on a net-long book, a price leg of **-13.5 percent**, and -13.1 percent of cost.
  The nominal positive result is dividend and distribution income on a long bias, not trend
  timing.
- **The convexity admission failed.** On the momentum book leg, conditional sleeve return in the
  worst decile was -1.70 percent against an unconditional +0.32 percent over the same 62 windows,
  a shortfall of 2.02 percentage points at 2.9 standard errors, with a permutation p of about
  0.002. The sleeve is positively exposed to the momentum book's drawdowns, which is the opposite
  of the property it was admitted to test for. The sample contains no momentum crash, so even
  this is measured on a period that never posed the question hardest.

---

## 4. The construction experiment: a null result on all eighteen comparisons

The stage 4 three-arm comparison (`decile_control`, `convex_point`, `convex_p80_one_sided`) across
six capital scenarios from $10k to $1m is recorded at
`results/2026-09-construction/comparison.json`.

**All eighteen Sharpe-difference intervals straddle zero.** Recomputed from the record on
2026-09-09 (Run): 18 of 18 intervals contain zero, with two-sided p-values from 0.262 to 0.999.
The widest point difference is 0.030 in periodic Sharpe against interval half-widths of 0.037 to
0.062. The experiment does not show that the convex constructor is better, and it does not show it
is worse. It shows the data cannot tell.

**No arm promotes, by construction.** The record's own `promotion_check` says it: "a same-sample
comparison can kill a configuration and can never promote one; the reference book's promotion
stays under the existing prospective rules". Reading the null as a licence to adopt either
constructor is a misreading of the design, not of the numbers.

**The intervals are not what they are often taken for.** The record's own method block states it:
"Endpoints are directional statements under the resampling assumptions ... They are not a
simultaneous interval and say nothing about realized portfolio return coverage." Eighteen
intervals at 0.95 each are not a 0.95 statement about the family.

---

## 5. The calibration bootstrap undercovers, and this was measured before the experiment ran

The uncertainty arm of the construction experiment uses a one-sided shrunk-slope endpoint at a
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
cannot be produced: 0 of 5 sessions compare end to end, and the stage 5 parity gate reports
decision-parity NOT COMPARED (`docs/engine_overhaul_parity_gate.json`). What is demonstrated
instead is order-sizing and extraction parity on those sessions. The distinction, and what would
close it, is `docs/engine_overhaul_operating_changes.md` section 1.3; the user-facing consequence
is `docs/versioning.md` section 3.

**The minimum supported live account size.** Undetermined. The launch brief refuses to encode a
figure from one synthetic rounding exercise, and the capital scenarios recorded in the stage 4
comparison are scenarios, not a supported range.

---

## 8. Standing conditions on any published number

- Compare returns with returns. The Sharpe figures here are on the excess-return conventions each
  record declares; a Sharpe is not comparable with a Treasury yield.
- Labels are not interchangeable. Synthetic demonstration, exploratory historical analysis,
  prospective evaluation, paper fills and actual fills are distinct categories and every record
  carries which one it is.
- Where an uncertainty estimate or a search history is not established, the record says
  `unavailable` rather than omitting the field.
- The licence is MIT and carries no warranty. Nothing in this repository is investment advice, and
  no result here is a forecast of future return.
