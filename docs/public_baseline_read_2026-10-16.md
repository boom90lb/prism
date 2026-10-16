# Public baseline (tranched, volatility-scaled momentum) — read record, 2026-10-16

**Kind:** read record for `registrations/2026-10-public-baseline.toml`
([design](public_baseline_design.md)), replication route ([SPEC §4](../SPEC.md)).
**Registration commit:** `47a42c2` (2026-10-16). It is also the code commit of every
row: each row record under `results/2026-10-public-baseline/rows/` carries it, and
the driver refused to run unless the registration files were committed unchanged.
**Selection set:** `public_baseline_v1`, budget 30, spent 30. The trial log
`trials/public_baseline_v1.jsonl` holds 60 rows: 30 started and 30 finished, all of
kind `new`, no retries. Its hash chain verifies (`TrialLog.verify()`, re-run for
this record).
**Outputs:** `results/2026-10-public-baseline/`: `summary.json` (every criterion,
case and input), `series.csv`, `parity.json`, `phase_dispersion.json`, `flow.json`,
`capacity/`, and `rows/<id>/` for each of the 30 rows.
**Inputs:** Kenneth French's daily momentum factor and three-factor files, 202608
CRSP vintage, rows through 2026-08-31 (sha256 `ecf82b11…` and `2f29e225…`, both
matching the pins); the frozen dataset `demotion_b1_2026-06-16` (manifest id
`bedf8c4c…`); cost stack `examples/reference/costs/retail_v3.toml`.
**Scored window:** 2022-10-27 → 2026-06-12, 909 sessions, the same for every row
(`summary.json` → `window`).

"Approved" here is what SPEC §4 calls promotion. T21 is the candidate: 12−1
momentum, top and bottom decile of the S&P 500, scaled to 12% volatility by the
Barroso–Santa-Clara rule, held as 21 overlapping tranches (sub-portfolios that each
rebalance every 21 sessions, one tranche per session), with idle equity in a T-bill
ETF. P00–P20 are the same book rebalanced on a single date every 21 sessions, at
offsets of 0 to 20 sessions ("phases"). The other rows are diagnostics.

## Verdict

**Approved: all six criteria hold.** T21, with idle equity above a 2% cash buffer
held in a T-bill ETF, is eligible for the live paper book. Switching on the volatility
scale, holding the ETF, choosing a risk profile, replacing the crash-triggered
gross reduction that SPEC §6 still names, and changing the live book are the
account owner's decisions. This record is the evidence for them; it makes none of
them.

The verdict holds under all three cash conventions the registration reports, so it
does not depend on the owner's pending confirmation that a T-bill ETF counts as a
long position under SPEC §3. The size of the edge does depend on it:

- idle equity in a T-bill ETF: expected excess over T-bills **+1.95%/yr**;
- idle cash earning a 3.3% broker yield: **+1.76%/yr**;
- idle cash earning nothing: **+0.30%/yr**.

These are expectations from the public series, not in-house results. Their
uncertainty is larger than the edge: one standard error of the expected net Sharpe
is about 0.27, about ±3.2%/yr at 12% volatility (§3).

## 1. The six criteria

| # | Criterion | Computed | Threshold | Result |
|---|---|---|---|---|
| 1 | Publication bar | Scaled French momentum, gross, 1931-02-02 → 2026-08-31: Sharpe 0.899, t 8.94 over 99.0 years | t ≥ 3 over ≥ 20 years | pass |
| 1 | Prior | Sharpe from 2015-04-01 to 2026-08-31 (11.4 years): **0.445** (t 1.50). Reported beside it: 0.401 cut at 2025-12-31 (t 1.31); 0.4 × the full-sample Sharpe, 0.359 | > 0 | pass |
| 2 | Tracking | Daily correlation of T21's net returns with the scaled reference: **0.903** over 909 sessions. Weekly: 0.920 | ≥ 0.75 | pass |
| 3 | Economics | 0.903 × 0.445 − cost drag 0.044 = **0.358** | > governing hurdle 0.195 | pass, by 0.163 |
| 4 | No in-house contradiction | T21's in-house net Sharpe 0.563, 95% interval [−0.244, +1.350] | upper end ≥ 0.358 | pass |
| 5 | Mechanics | Capital conservation (N4) on all 1,620 session steps of the simulation, largest residual $1.1 × 10⁻¹⁰, no held name ever unmarked. Largest decided gross 1.499, largest held gross 1.502, no session above 2.0. No scale fallback. P00's parity gate: daily correlation 0.99928 with `results/2026-09-vol-momentum/series.csv` `in_house|net` over its 929 sessions, Sharpe 0.624 against the recorded 0.663 (difference 0.038) | correlation ≥ 0.99, difference ≤ 0.05, gross ≤ 2.0 | pass |
| 6 | Tranching does not cost return | T21 − P̄ net Sharpe +0.014, paired 95% interval [−0.009, +0.037], where P̄ is the equal-weight average of P00–P20's daily returns. Beside it: T21 − P00 −0.008 [−0.152, +0.118] | upper end ≥ 0 | pass |

The cost drag of 0.044 Sharpe is T21's charged cost of 0.56%/yr divided by its
realized volatility of 12.86%. rf is 4.00%, the mean French one-month bill rate
over the 252 sessions 2025-08-29 → 2026-08-31, annualized. The hurdle is the
T-bill benchmark expressed as a Sharpe ratio (§2).

Every value in the table is in `summary.json` → `criteria` and
`rows.T21`. The criterion-1 prior and the rf were also computed on these same two
files by another package before the registration commit, which the design
discloses (design §2); this run's figures agree with that disclosure (0.4446).

## 2. The three cash conventions

T21's stock long side averaged 0.556 of equity (L/E), so 0.444 of equity was idle on
average. The governing case holds 0.424 of equity in the T-bill ETF, earning rf less
a 10 bp fee (3.90%), and keeps the 2% buffer at zero yield.

| Case | Idle equity earns | Hurdle at 12% target vol | Hurdle at 12.86% realized vol | 0.358 − hurdle | Expected excess over bills | Criterion 3 |
|---|---|---|---|---|---|---|
| T-bill ETF (governing) | ETF 3.90% on 0.424, buffer 0 on 0.020 | **0.195** | 0.182 | +0.163 | **+1.95%/yr** | pass |
| Broker cash, no ETF | 3.3% on 0.444 | 0.211 | 0.197 | +0.147 | +1.76%/yr | pass |
| Idle cash earns nothing | 0 | 0.333 | 0.311 | +0.025 | +0.30%/yr | pass |

In each case the hurdle at the 12% target is the larger one, so it governs
(registered rule). The expected excess over bills is (0.358 − hurdle) × 12%: the
long–short book's expected return of 0.358 × 12% = 4.30%/yr, plus what idle equity
earns, less the 4.00% bill rate. The 3.3% yield is the provisional broker figure in
`examples/reference/costs/retail_v2.toml`, not a verified rate for a margin account
holding shorts. `summary.json` → `criteria.economics.cases`.

## 3. What the margin means

Everything in this section is plain arithmetic on the figures above (derived, not
measured).

- **The account's edge, not the book's Sharpe.** The long–short book is expected to
  earn 4.30%/yr and the T-bill holding 0.424 × 3.90% = 1.65%/yr, so the account earns
  5.95%/yr against bills at 4.00%. The holding adds almost no volatility, so the
  account's excess-over-bills Sharpe is about 1.95% / 12% ≈ **0.16**. The 0.358 is the
  long–short book's Sharpe, not the edge over bills.
- **Uncertainty.** The standard error of the prior is 0.4446 / 1.501 = 0.296. After
  tracking it is 0.268 Sharpe, about ±3.2%/yr at 12% volatility. Counting only that
  error, under a normal approximation, the probability that the true expected excess
  over bills is positive is about 0.73 with the T-bill ETF, 0.71 at a 3.3% yield and
  0.54 if idle cash earns nothing. Error in the tracking, cost and decay inputs would
  widen these.
- **Detection.** Rejecting a zero Sharpe for the long–short book at one-sided 95%
  from a deployment's own returns would take (1.645 / 0.358)² ≈ 21 years. Telling
  the 0.163 margin over the governing hurdle apart from zero would take about 100
  years.
- **What the prior rests on.**

  | Prior used | 0.903 × prior − 0.044 | T-bill ETF | 3.3% cash | Idle earns nothing |
  |---|---|---|---|---|
  | 0.445, through 2026-08-31 (registered) | 0.358 | +1.95%/yr | +1.76%/yr | +0.30%/yr |
  | 0.401, cut at 2025-12-31 | 0.319 | +1.48%/yr | +1.29%/yr | **−0.17%/yr** |
  | 0.359, 0.4 × full sample | 0.281 | +1.03%/yr | +0.84%/yr | −0.63%/yr |

  With idle cash earning nothing, the pass rests on January–August 2026: the prior
  needed to break even in that case is (0.333 + 0.044) / 0.903 = 0.417. With the
  T-bill ETF it is 0.265, and at a 3.3% yield 0.282.
- **Rates.** With the T-bill ETF the hurdle is 0.147 at rf 3% and 0.243 at rf 5%. With
  idle cash earning nothing it is 0.250 and 0.417, so that case fails at rf 5%. At
  `vol_momentum_v1`'s recorded rf of 4.21% the zero-yield hurdle is 0.351, still
  below 0.358.
- **Late fills.** The economics figure carries no allowance for fills later than the
  open. 74% of the paper account's fills are morning sweep orders, not
  opening-auction fills (§14). Each 0.01 of Sharpe lost to execution costs
  0.12%/yr; a loss of 0.025 would erase the zero-yield case.

## 4. Timing luck across rebalance dates (P00–P20)

In-house, price return, the 909-session window. These are same-sample figures,
reported to measure dispersion, not as expectations.

| | Median | Interquartile range | Range |
|---|---|---|---|
| Net Sharpe | 0.555 | 0.494 – 0.581 | 0.418 – 0.676 |
| Annual return | 6.56% | 5.68% – 6.93% | 4.66% – 8.18% |
| Maximum drawdown | 14.4% | 13.8% – 15.2% | 13.2% – 16.2% |

- The standard deviation of annual return across the 21 phases is **0.97%/yr**. A
  user who rebalances on one date carries this dispersion: over these 3.6 years the
  same signal returned between 4.7% and 8.2% a year depending only on the date.
- T21: net Sharpe 0.563, 6.62%/yr, drawdown 14.1%, at the 52nd percentile of the
  phases' Sharpes. P̄: 0.549, 6.38%/yr, 14.0%, at the 48th. A T21 user holds close
  to the average of the phases and carries almost none of the dispersion.
- 3.6 years of data estimate the dispersion noisily.

`phase_dispersion.json`.

## 5. Tranching mechanics (T21 against the single-date books)

| | T21 | Mean of P00–P20 |
|---|---|---|
| Turnover, times equity per year | 8.50 | 8.43 |
| Cost drag, Sharpe | 0.044 | 0.045 |
| Orders per session | 38.5 | 4.8 |
| Median order notional | $72 | $887 |
| Wash-sale pairs per year | 1,102 | 274 |

- **Trades are not netted away.** One tranche refreshes each session, so turnover
  is the same as a single-date book's. Tranching spreads trading over the month; it
  does not reduce it.
- **Wash-sale pairs** are consecutive trades of opposite sign in one name within 30
  calendar days, counted from the fills: T21 made 3,995 (1,102 a year; 1,999 of them
  a sell followed by a buy), about four times the single-date books' rate (P00: 1,063,
  293 a year). Each session's orders come from a different tranche's refresh, so
  the account often buys a name and sells some of it within a few sessions. In a
  taxable account, a loss realized on a sale within 30 days of buying the same stock
  is deferred (the wash-sale rule). The tax effect is not estimated here.
- **The volatility scale.** Mean 1.110, minimum 0.727, maximum 1.418; it never reached
  the 2.0 cap. 126 warm-up decisions at scale 1, 930 full-window decisions, no
  fallback (`rows/T21/row.json` → `sizing`).
- **Cash and margin.** Mean L/E 0.556 and mean T-bill ETF share 0.424. Counting the
  ETF as a long position, Reg T initial excess was 0.232 of equity on average and
  0.116 at its lowest; maintenance excess 0.539 on average and 0.470 at its lowest.
  The 50% initial and 30% maintenance rates are Alpaca's documented minimums; the
  ETF's margin treatment is not verified.

`summary.json` → `mechanics_of_tranching`.

## 6. Account size

| Row | Capital | Names held / targeted | Share held | Tracking error to continuous-share target | Net Sharpe against T21 at $100k, paired 95% interval |
|---|---|---|---|---|---|
| T21-25k | $25,000 | 115.2 / 147.4 | 0.78 | 1.53%/yr | −0.048 [−0.195, +0.087] |
| T3-25k (3 tranches) | $25,000 | 112.1 / 124.2 | 0.90 | 1.51%/yr | −0.003 [−0.128, +0.103] |
| T21 | $100,000 | 139.4 / 147.4 | 0.95 | 0.51%/yr | — |
| T21-1m | $1,000,000 | 147.3 / 147.4 | 1.00 | 0.38%/yr | +0.0003 [−0.031, +0.029] |

- At $25,000, whole-share rounding drops about a fifth of T21's targeted names: a
  name that only a few tranches hold adds up to less than half a share and rounds
  to zero. T3 at $25,000 keeps 90% of its (fewer) names.
- None of the paired differences is distinguishable from zero on 3.6 years.
- The 0.38%/yr at $1M is mostly the gap between the open fill and the close mark,
  which every account size shares (`summary.json` → `rounding.*.tracking_error_note`).
- T3 is a diagnostic. Offering it as a small-account mode would need its own
  registration, approved on data after that registration (design §4).

## 7. Execution venue, liquidity weighting and holdings diversity

| Row | Daily tracking | Cost drag | 0.4446 × tracking − drag | Governing hurdle | Excess, T-bill ETF | Excess, 3.3% | Excess, idle at 0 | Net Sharpe against T21, paired 95% interval |
|---|---|---|---|---|---|---|---|---|
| T21 | 0.903 | 0.044 | 0.358 | 0.195 | +1.95% | +1.76% | +0.30% | — |
| D-split | 0.901 | 0.044 | 0.357 | 0.195 | +1.94% | +1.75% | +0.29% | +0.003 [−0.014, +0.024] |
| D-liq | 0.891 | 0.035 | 0.361 | 0.172 | +2.27% | +2.04% | +0.33% | −0.014 [−0.369, +0.279] |
| D-part1 | 0.894 | 0.042 | 0.355 | 0.189 | +2.00% | +1.80% | +0.27% | +0.013 [−0.187, +0.198] |
| D-part2 | 0.901 | 0.043 | 0.358 | 0.191 | +2.01% | +1.81% | +0.30% | −0.095 [−0.238, +0.038] |
| D-part3 | 0.901 | 0.043 | 0.357 | 0.192 | +1.98% | +1.79% | +0.29% | +0.077 [−0.057, +0.216] |

Excess figures are %/yr over bills at the 12% target. `summary.json` →
`diagnostics`.

- **Venue (D-split).** Filling half of each order at the next open and half at the
  next close changed nothing measurable: tracking −0.002, the same cost drag, and a
  paired difference of +0.003. Both legs are charged the same bucket spreads; a real
  closing-auction cost and paper closing-auction fills are unverified.
  Closing-auction orders are not adopted (design §12).
- **Liquidity weighting (D-liq).** Weighting names within each leg by the square
  root of their 20-session median dollar volume lowered tracking by 0.012 (0.005 of
  Sharpe in the economics figure) and cost drag by 0.008, a net +0.003. Its long side
  was smaller (L/E 0.482), which lowers its hurdle to 0.172. Its in-house drawdown
  was larger (17.9% against 14.1%). In the orders whose name is at or below the 10th
  percentile of median dollar volume, the median order was 5.0 × 10⁻⁷ of that volume
  against T21's 8.6 × 10⁻⁷, and the 95th percentile 1.9 × 10⁻⁶ against 3.0 × 10⁻⁶.
  Its crowding break-even is about 2.4 times T21's (§9).
- **Holdings diversity (D-part1–3).** Three seeded 80% subsets of the universe.
  Their universes overlap with a Jaccard index of 0.64–0.67 and their held names
  with 0.62–0.63 on an average day. Tracking fell by 0.002–0.009, which costs
  0.001–0.004 of Sharpe (0.01–0.05%/yr) in the economics figure. With followers
  split evenly over the three subsets, aggregate per-name flow is 0.970 of everyone
  on T21, and flow weighted the way square-root impact weights it (flow^1.5) is
  0.757 (`summary.json` → `diagnostics.holdings_diversity`). The paired intervals
  (−0.095 to +0.077) show how much dropping a random fifth of the names moves a
  3.6-year in-house Sharpe.

None of these diagnostics can be adopted on this evidence. Each would need a new
registration approved on data after its own date (design §4).

## 8. Flow across deployments

Share of every deployment's combined traded notional by session (`flow.json`):

| Case | Busiest 1/21 of sessions' share of flow | Peak session over mean | Concentration against an even spread |
|---|---|---|---|
| Single date, everyone starting the same day (P00) | 98.9% | 30.5 | 21.6 |
| Single date, uniformly random dates (mean of P00–P20) | 6.9% | 1.63 | 1.04 |
| T21 | 7.5% | 2.03 | 1.07 |

- A burst of T21 deployments starting on one day builds over its first 21 sessions;
  its busiest session is 1.98 times T21's steady mean.
- Two single-date deployments on different dates never traded the same name in the
  same direction on the same session (mean and maximum 0 over all 420 ordered pairs
  of P rows). Two deployments on the same date, or two T21 deployments of the same
  size, make identical trades.
- Holdings overlap is unchanged by any timing scheme: every deployment holds the same
  book, so a momentum unwind hits all of them the same day.

## 9. Crowding and capacity (uncounted; cannot choose a configuration)

The recorded trades of T21, P00, D-split and D-liq were re-priced with square-root
impact: the cost per dollar traded in name *i* is
Y · σᵢ · √[f · q · (1 + φ · F / E) / (Vᵢ · w)], where q is the order's notional, f the
fraction of it in the window, E own equity, F is the capital of other
users trading the same book, φ the fraction of it trading on the same session, w
the share of daily volume available in the execution window, and σᵢ and Vᵢ each
name's trailing 20-session volatility and median dollar volume. The stack's own
participation charge is replaced, not added. Every coefficient and window share is
an assumption; the auction shares come from 2015–17 industry notes. `capacity/`.

**Number of other users at which crowding uses up the whole expected excess over
bills**, users at $100,000 each, at Y = 0.7 (in brackets: Y = 1.0 to Y = 0.5). At
$1,000,000 each, divide by ten.

| Execution window | Single date, everyone on one date (P00, φ = 1) | Single date, uniformly random dates (P00, φ = 1/21) | T21, everyone on one grid (φ = 1) | D-liq (φ = 1) |
|---|---|---|---|---|
| Opening auction at 1% of volume | **65** (32–129) | 1,375 (662–2,717) | **548** (268–1,075) | 1,334 (653–2,616) |
| Opening auction at 3% | 199 (97–390) | 4,170 (2,032–8,194) | 1,646 (806–3,227) | 4,005 (1,962–7,850) |
| Closing auction at 8% | 531 (260–1,042) | 11,157 (5,455–21,889) | 4,391 (2,151–8,608) | 10,680 (5,233–20,935) |
| Full day | 6,653 (3,260–13,042) | 139,720 (68,451–273,872) | 54,904 (26,902–107,612) | 133,518 (65,423–261,696) |

D-split, half at the open and half at the close (8%): 2,376 (1,164–4,658) users with
the open at 1%, and 5,025 (2,462–9,850) with the open at 3%.

- **The edge erodes well before break-even.** Cost grows with the square root of
  follower capital. For T21 in a 1% opening auction at Y = 0.7, $1M of other users (10
  at $100k) costs 0.19%/yr, $10M (100 users) 0.75%/yr, and $100M (1,000 users)
  2.56%/yr, more than the whole edge. With everyone rebalancing a single-date book on
  the same day, $1M costs 0.55%/yr and $10M 2.15%/yr.
- **Own orders.** With no other users, pricing T21's own orders against 1% of daily
  volume, rather than the full day as the cost stack does, costs 0.08%/yr more than
  the stack charged. The largest account that keeps a positive expected excess with no
  other users is about $54M for T21 and $6.7M for a single-date book at that setting.
- **Why T21 prices below uniformly random dates.** At $100,000 a tranche's slice of a
  name is often a fraction of a share, so the account trades in whole-share lumps
  rather than in 21 even slices: 78% of T21's fills are one share. Over the common
  window, the sum of each fill's notional to the power 1.5 (what square-root impact
  charges) is 0.373 of P00's, against 1/√21 = 0.218 for even slices; scaled to the
  same capital it is 0.234 at $1M and 0.677 at $25k (derived from `rows/*/fills.parquet`;
  §15). The random-dates column also assumes users actually spread their dates. The
  live loop today sets a book's date by its first session, so users who start
  together are in the first column, and a single-date user carries the dispersion of
  §4. T21 needs no coordination to be in its column.
- **Binding names.** T21 traded 479 names, 53 of them with median dollar volume under
  $100M; the 5th and 10th percentiles of traded names' median dollar volume were $75M
  and $97M. Names under $100M carry about 13% of T21's crowding cost (D-liq: about 7%).
- Paper fills cannot measure impact. These are scenarios under stated assumptions,
  not measurements.

## 10. Deflation, breadth and horizon

- **Deflated Sharpe ratio** of T21: 0.790 against all 30 rows, 0.840 against their
  effective count. The participation ratio of the 30 rows' daily return correlations
  is 1.045, so the floor of 2 applies: the rows are almost one strategy. Approval
  does not depend on it (replication route). `summary.json` → `deflation`.
- **Claim tier.** The engine records every row at `net_edge`. Under SPEC §3, T21
  reaches `robust_edge` through the replication tests of §4, not through its deflated
  Sharpe ratio. No other row is described above `net_edge`.
- **Breadth (N6).** The effective number of independent positions, N_eff, is 43.4:
  the participation ratio of the covariance of
  position-weighted daily return contributions, over 491 names ever held and 147
  targeted on an average day (design §7). It is not the post-residualization
  covariance SPEC N6 names, because T21 has no residualization step. With the
  21-session rank IC of 0.045, the fundamental-law ceiling on the Sharpe ratio is
  0.045 × √43.4 × √12 = 1.02. The realized 0.563 is below it; no flag.
- **Horizon profile.** Rank IC of T21's target weights against each name's next
  h-session return (`summary.json` → `breadth.ic_profile`):

  | h (sessions) | 1 | 2 | 5 | 10 | 21 | 63 |
  |---|---|---|---|---|---|---|
  | Mean IC | 0.029 | 0.033 | 0.030 | 0.030 | 0.045 | 0.062 |
  | Standard error (non-overlapping) | 0.010 | 0.014 | 0.020 | 0.026 | 0.040 | 0.063 |
  | Non-overlapping observations | 908 | 454 | 181 | 90 | 43 | 14 |

## 11. The cap on public data

The reference scale was at the 2.0 cap on 25.7% of sessions since 1931 (at 1.8 or
more on 35.9%), and on none since 2015-04-01 (`summary.json` → `reference_cap`). Over
the post-publication window the cap never bound. This describes the volatility
series; it is not a criterion.

## 12. Every row

In-house net figures on the 909-session window (`summary.json` → `rows`). All 30
rows passed the N4, ceiling and fallback checks.

| Row | Net Sharpe | Annual return | Volatility | Max drawdown | Mean scale | Mean L/E | Turnover | Charged cost |
|---|---|---|---|---|---|---|---|---|
| T21 | 0.563 | 6.62% | 12.86% | 14.10% | 1.110 | 0.556 | 8.50 | 0.560% |
| P00 | 0.571 | 6.75% | 12.91% | 14.40% | 1.096 | 0.552 | 8.46 | 0.575% |
| P01 | 0.418 | 4.66% | 12.90% | 14.87% | 1.081 | 0.544 | 8.47 | 0.573% |
| P02 | 0.495 | 5.72% | 12.95% | 14.94% | 1.079 | 0.546 | 8.38 | 0.568% |
| P03 | 0.465 | 5.33% | 13.01% | 16.17% | 1.074 | 0.543 | 8.68 | 0.581% |
| P04 | 0.555 | 6.56% | 12.95% | 15.01% | 1.081 | 0.546 | 8.41 | 0.569% |
| P05 | 0.565 | 6.69% | 12.97% | 15.64% | 1.097 | 0.551 | 8.54 | 0.579% |
| P06 | 0.576 | 6.78% | 12.83% | 15.30% | 1.094 | 0.548 | 8.40 | 0.575% |
| P07 | 0.494 | 5.68% | 12.86% | 16.11% | 1.098 | 0.551 | 8.24 | 0.572% |
| P08 | 0.534 | 6.20% | 12.80% | 14.47% | 1.094 | 0.546 | 8.35 | 0.573% |
| P09 | 0.511 | 5.91% | 12.84% | 14.20% | 1.088 | 0.546 | 8.27 | 0.570% |
| P10 | 0.548 | 6.40% | 12.82% | 14.04% | 1.112 | 0.559 | 8.57 | 0.586% |
| P11 | 0.585 | 6.94% | 12.89% | 13.80% | 1.117 | 0.560 | 8.64 | 0.586% |
| P12 | 0.676 | 8.15% | 12.82% | 13.72% | 1.109 | 0.555 | 8.62 | 0.587% |
| P13 | 0.595 | 7.05% | 12.86% | 13.82% | 1.109 | 0.556 | 8.56 | 0.587% |
| P14 | 0.674 | 8.18% | 12.89% | 13.82% | 1.109 | 0.557 | 8.34 | 0.577% |
| P15 | 0.581 | 6.93% | 12.98% | 13.70% | 1.104 | 0.555 | 8.27 | 0.577% |
| P16 | 0.614 | 7.35% | 12.90% | 13.19% | 1.111 | 0.559 | 8.35 | 0.582% |
| P17 | 0.444 | 5.00% | 12.84% | 13.65% | 1.104 | 0.557 | 8.21 | 0.577% |
| P18 | 0.473 | 5.43% | 12.93% | 13.45% | 1.104 | 0.556 | 8.34 | 0.582% |
| P19 | 0.449 | 5.08% | 12.90% | 15.20% | 1.098 | 0.553 | 8.38 | 0.578% |
| P20 | 0.572 | 6.77% | 12.93% | 15.49% | 1.097 | 0.551 | 8.52 | 0.576% |
| D-split | 0.566 | 6.66% | 12.85% | 14.18% | 1.110 | 0.556 | 8.49 | 0.560% |
| D-liq | 0.549 | 6.50% | 13.02% | 17.92% | 0.972 | 0.482 | 7.45 | 0.462% |
| D-part1 | 0.576 | 6.77% | 12.79% | 15.03% | 1.072 | 0.535 | 8.09 | 0.539% |
| D-part2 | 0.468 | 5.28% | 12.74% | 14.21% | 1.079 | 0.541 | 8.27 | 0.542% |
| D-part3 | 0.640 | 7.65% | 12.81% | 14.45% | 1.090 | 0.545 | 8.36 | 0.553% |
| T21-25k | 0.515 | 5.76% | 12.37% | 14.42% | 1.110 | 0.517 | 8.36 | 0.553% |
| T21-1m | 0.563 | 6.61% | 12.83% | 14.04% | 1.110 | 0.559 | 8.55 | 0.570% |
| T3-25k | 0.560 | 6.42% | 12.50% | 14.20% | 1.111 | 0.528 | 8.39 | 0.560% |

The trial log's per-row Sharpes differ from this table because they cover all 1,056
scored sessions from 2022-03-29, including the 21-session build and the warm-up at
scale 1 (T21: 0.633 annualized there). The criteria use the common window.

## 13. What this changes

- **SPEC §6 deployment gate.** "A configuration is promoted under §4": already met
  by `vol_momentum_v1`; T21 is a second. The sizing rule §6 names is still the
  crash-triggered gross reduction that public data contradicted; replacing it with
  the volatility scale is the account owner's decision. The clean-session count and
  the risk profile are untouched by this record.
- **The live book is unchanged.** The scheduled paper account still trades unscaled,
  single-date momentum. Running T21 on paper needs the tranched book, the volatility
  scale and the T-bill holding in the live loop, all off by default, and the owner's
  decision to switch them on, preferably on a separate paper account so that
  `forward_v1`'s stream is not disturbed.
- **The forward stop rule runs from the registration commit** (design §10). The
  first read is at 126 forward sessions counted from 2026-10-19, about
  **2027-04-20**, then every 63 sessions (next about 2027-07-21). Eligibility is
  withdrawn if the forward net Sharpe's 95% interval lies entirely below zero, the
  forward drawdown exceeds 25%, the daily correlation with the scaled reference on
  the latest French vintage falls below 0.5, or, on two consecutive reads, forward
  tracking × the latest post-publication Sharpe − forward cost drag falls below the
  governing hurdle. The reader must be committed before the first read.
- **Unchanged:** `vol_momentum_v1`'s verdict and stop rule, `forward_v1`, M6 and T5.
  This set is closed: all 30 rows ran.
- **For users**, [baseline.md](baseline.md) states the forward expectation, the
  conditions it rests on and the capacity warning.

## 14. Limits

- The in-house sample is 3.6 years; every interval above is wide.
- In-house books are price returns and the public series total returns. The forward
  figure takes its level from the total-return public series; the in-house returns
  enter it only through the tracking correlation. The missing dividends (about
  −36 bp/yr for the unscaled book, [erratum E3](erratum_2026-10-14.md)) bear mainly on
  the in-house Sharpes (criteria 4 and 6, §§4–7).
- Membership coverage is 0.907 (574 of 633 names); the missing names flatter tracking
  and cost.
- The French files were read by another package before the registration commit
  (design §2).
- P00 is not `vol_momentum_v1`'s computation (design §13); the parity gate bounds the
  difference (0.038 of Sharpe).
- The T-bill holding is not simulated. Its 10 bp fee is an assumption, and its margin
  treatment is unverified.
- Paper fills are not the modelled open fills: 185 of the 249 fills in
  `tests/fixtures/paper_replay/fills.jsonl` are 09:50 ET sweep orders. No allowance
  for late fills is in the economics figure (§3).
- Taxes are not modelled; T21's wash-sale rate is about four times a single-date
  book's (§5).
- The experiment profile's preference limits (including max gross 1.0) were recorded,
  not enforced. No batch in any row was repaired, scaled or dropped (each run record's
  `performance.repair`; the run records under `results/run-*/` are git-ignored, and
  their sha256 values are in `summary.json` → `inputs.rows`, all re-verified for this
  record).
- §9 is driven by its assumptions.

## 15. How to check these numbers

From the repository root (no network, no prism import):

```sh
python3 -I -c "import json;d=json.load(open('results/2026-10-public-baseline/summary.json'));c=d['criteria'];e=c['economics'];print(d['verdict'],c['publication_bar']['prior_sharpe'],c['tracking']['daily_corr'],e['cost_drag_sharpe'],e['figure_f3'],*[(k,v['sharpe_annual'],v['excess_over_bills_annual']) for k,v in e['cases'].items()])"
python3 -I -c "import json;d=json.load(open('results/2026-10-public-baseline/summary.json'));p=d['phase_dispersion'];print(p['sharpe'],p['annual_return'],p['annual_return_sd_pct'])"
python3 -I -c "import json;d=json.load(open('results/2026-10-public-baseline/summary.json'));print(d['mechanics_of_tranching']['T21']['wash_sale'],{k:v['held_over_targeted_mean'] for k,v in d['rounding'].items()})"
python3 -I -c "import csv;[print(r['label'],r['breakeven_users_at_100000']) for r in csv.DictReader(open('results/2026-10-public-baseline/capacity/T21.csv')) if r['follower_capital']=='0.0']"
```

The tracking correlation, Sharpe, cost drag, mean L/E and ETF share recompute exactly
from `series.csv` columns `in_house|net`, `reference|gross`, `in_house|cost` and
`in_house|long_over_equity` on 2022-10-27 → 2026-06-12, and the three priors from
`reference|gross`; rf is the mean of the last 252 `RF` rows of the pinned
three-factor file, times 252. The lumpiness ratios of §9 come from
`rows/{P00,T21,T21-1m,T21-25k}/fills.parquet`: the sum of |`notional`|^1.5 over fills
dated 2022-10-28 → 2026-06-15, with T21-1m's divided by 10^1.5 and T21-25k's
multiplied by 4^1.5 before dividing by P00's. The share of one-share fills is over
the same fills.

The per-row `targets.parquet`, `holdings.parquet` and `fills.parquet` files (about 61 MB)
are not in the public repository: they hold per-name prices and holdings derived from
vendor bars whose redistribution terms are unconfirmed (`NOTICE`), and they would roughly
triple the size of a clone. They are kept by the operator and regenerate exactly from the
driver at the registration commit with the registered dataset. Every other figure in this
record recomputes from tracked files.
