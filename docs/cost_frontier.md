# Cost frontier — net performance vs effective one-way spread (measurement record)

**Status: uncounted, read-only diagnostic. Recorded 2026-07-19.** It
searches nothing (every spread level is reported, none is selected), invokes
no counted machinery, appends no ledger row, and moves no ratified
statistic — certification 001 stands exactly as published. It
**retroactively closes the open attachment promised in `docs/r2_design.md`
§5** (`docs/r2_design.md:85-86`: the §10 adjudication happens on trial 3's
claim packet "with cost_toll and capacity_curve attached"): the stored
packet (`results/r2_t3_full/claim_packet.json`) carries neither attachment.
This record supplies the cost lens for that adjudication packet on the
corrected axis, and documents why the `capacity_curve` half is degenerate
for these runs (§1). References to
`docs/certifications/001-residual-reversion-daily-negative.md` are one-way
reads; the certified doc is not edited.

## 1. Question, and the corrected axis

The desk-critique capacity question ("at what size does this die?") is
mis-axed for the shipped cost model. Spread cost per dollar traded is
**AUM-invariant**: the spread leg is charged on weight-space turnover —
`commission_spread = Σ|trade| · (commission_bps + spread_bps) / 1e4` with
weights as fractions of capital
(`src/prism/execution/target_weights.py:152-168`) — so doubling capital
doubles dollars traded and dollars of spread cost in exact proportion. The
only AUM-dependent term in the model is the ADV participation-impact leg,
and every certified residual run disabled it (`adv_impact_coeff: 0.0`,
`results/r2_t3_full/config.json`), which makes the `capacity_curve` AUM
axis flat by construction — `src/prism/validation/capacity.py:55-68` states
this explicitly (a correct, and informative, result: impact does not bind
at these sizes). The meaningful axis is therefore the **effective one-way
spread level**. The artifact an outside reader needs answers: *does this
edge net positive below some spread level X, or does no spread level rescue
it?*

## 2. Method

**Instrument:** `research/scripts/cost_frontier.py` (tests in
`tests/test_cost_frontier.py`; evidence
`results/cost_frontier_2026-07-19.json`). Pure arithmetic over stored
artifacts — no backtest rerun, no WFO driver, no trials-ledger writes.

**Daily granularity — frozen-stack trial 3** (`results/r2_t3_full`,
`config_hash d4c93af68855`, cert-001 §4 row 9, the adjudicated
configuration). The run's `returns.csv`/`costs.csv` record net daily
returns and the per-day cost decomposition. Gross is reconstructed as
`net + total`; the flat 1 bp commission leg is split out of
`commission_spread`, leaving the realized bucket-schedule spread cost.
Re-pricing at a flat one-way spread `s` is then

    net(s) = gross − turnover · (commission_bps + s)/1e4 − impact − borrow

— commission (1 bp), quadratic impact (`slippage_coeff` 10), and borrow
(50 bp/yr on shorts) are held at the shipped model; **only the spread axis
moves**. Validation gates, asserted before any frontier row is read: the
decomposition must reconstruct the stored net stream exactly (max abs
residual at float precision), and the loaded series must reproduce the
stored summary — measured: Sharpe reproduced to 1.6e-14 (−0.7507369107978005
vs stored −0.750736910797816), total cost to 1.7e-14, avg turnover exact.

**Aggregate granularity — ETF-factor run** (`config_hash a2c5538e70b1`,
`docs/stat_arb.md:218-232`: gross annualized Sharpe **+0.54**, the only
configuration in the selection history with a positive gross edge). Its
per-day artifacts no longer exist on disk — the trials-ledger row's
`output_dir` pointed into a since-deleted worktree — so the frontier is
computed from the surviving records: the ledger row
(`results/stat_arb_residual_trials.jsonl`: net annualized Sharpe
−0.2261 exact, 1,304 obs, flat 1 bp spread + 1 bp commission as run) and
the doc aggregates (gross cumulative +6.1%, cumulative costs 8.6%,
~0.26/day turnover). Stated approximations: annualized vol is implied from
the ledger Sharpe and held constant across levels (the daily-granularity
computation on trial 3 measures the vol shift across the whole frontier at
0.1% relative, 2.1095% → 2.1116% annualized);
returns are arithmetic per-year means. Cross-check gate: the implied gross
annualized Sharpe (0.552) must match the documented +0.54 within rounding
of the three 2-significant-figure inputs (tolerance 0.03) — it does, and
the as-run row reproduces the ledger Sharpe by construction.

`cost/gross` below is the `cost_toll` convention (total cost over
Σ|gross_t|, `src/prism/validation/capacity.py:103-146`); the daily
toll-booth fraction here is deliberately a different statistic from
cert-001's fold-level `cost_to_gross_pnl` conjunct and does not reconcile
with it numerically (documented at `capacity.py:118-122`).

## 3. Results

Evidence: `results/cost_frontier_2026-07-19.json`.

**Frozen-stack trial 3** (daily granularity; realized effective one-way
spread as run: **2.57 bp**, turnover-weighted through
`SPREAD_BUCKET_SCHEDULE_V1`):

| one-way spread level | net Sharpe (ann.) | net total return | cost/gross |
|---|---|---|---|
| 0 bp | −0.062 | −0.8% | 0.033 |
| 0.5 bp | −0.196 | −2.2% | 0.044 |
| 1 bp | −0.330 | −3.7% | 0.055 |
| 2 bp | −0.598 | −6.4% | 0.078 |
| bucket V1 as run (2.57 bp eff.) | **−0.751** | −8.0% | 0.091 |
| zero-cost gross ceiling | **+0.325** | +3.5% | 0 |

**Break-even one-way spread: −0.23 bp.** No spread level rescues the
frozen stack: it is net-negative even at *zero* spread, with commission,
impact, and borrow still at the shipped model. (Corroboration: the counted
flat-1 bp trial `r2_t1_cfband_flat` — different weights, since its band saw
flat costs — landed at −0.350, cert-001 §4 row 7, mechanism-consistent
with the −0.330 re-pricing here.)

**ETF-factor run** (aggregate granularity; implied ann. vol 2.14%):

| one-way spread level | net Sharpe (ann.) | net return (arith./yr) | cost/gross PnL |
|---|---|---|---|
| 0 bp | **+0.081** | +0.17% | 0.85 |
| 0.5 bp | −0.073 | −0.16% | 1.13 |
| 1 bp (as run) | −0.226 | −0.48% | 1.41 |
| 2 bp | −0.533 | −1.14% | 1.97 |
| zero-cost gross ceiling | **+0.552** | +1.18% | 0 |

**Break-even one-way spread: +0.26 bp.** The one configuration with a real
gross edge nets positive only below ~0.26 bp effective one-way spread at
its ~0.26/day turnover — a quarter of the *tightest* bucket in the
pre-registered conservative schedule (1 bp, ≥$500M ADV names) and an order
of magnitude under the schedule's realized pricing on the frozen-stack
book (2.57 bp). The frontier therefore puts the lever on
holding-period/turnover, not execution: at this turnover no measured
spread in the repo's cost evidence (schedule buckets, fills-ledger
calibration floors) comes near 0.26 bp one-way — the same redirect
`docs/stat_arb.md:218-232` recorded from the gross/net split.

**The zero-cost ceiling rows are flagged, not claimable.** Even with every
cost leg off, the frontier tops out at gross Sharpe +0.325 (trial 3) /
+0.552 (ETF factors) — and the multiplicity bar survives at zero cost: the
best residual selection-set DSR is **0.145**
(`docs/certifications/001-residual-reversion-daily-negative.md:108-109`),
far below the 0.5 a positive deflated Sharpe requires. The cost frontier
and the deflation verdict fail the sleeve independently.

## 4. What this does and does not license

It licenses: attaching this record to the §10 adjudication packet as the
cost lens `docs/r2_design.md:85-86` promised (the `capacity_curve` half
discharged as degenerate per §1); quoting the break-even spreads
(−0.23 bp frozen stack / +0.26 bp ETF factors) as the corrected-axis
answer to desk-critique point 4; the outside-reader claim "no effective
spread level rescues the adjudicated configuration, and the one positive
gross configuration requires sub-0.3 bp one-way costs at its recorded
turnover."

It does not license: any change to certification 001 or its verdict (the
DSR bar fails the sleeve before costs are charged); treating either
gross-ceiling row as an edge claim; reading the aggregate-granularity ETF
rows as if per-day artifacts backed them (they are reconstructions from
the ledger row and `docs/stat_arb.md:218-232`, exact only at the as-run
point and linear elsewhere); or re-running any counted machinery to
sharpen them.
