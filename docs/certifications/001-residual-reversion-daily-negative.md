# Certification 001 — Residual reversion on the S&P cross-section is uneconomic at retail cost

**Kind:** negative result (the harness's first certification).
**Status:** final. The selection set is closed at 17 counted trials; per the
pre-registered adjudication rule (`docs/demotion_design.md §4`) there are no
appeals inside it.
**Date:** 2026-07-06. **Certifying artifacts:** the trial ledger
`results/stat_arb_residual_trials.jsonl` (17 rows, append-only) and the
per-trial claim packets under `results/` (§8).

---

## 1. What is certified

The Avellaneda–Lee residual-reversion sleeve (frozen-v1 signal stack, §2) on
the point-in-time S&P 500 universe, 2020-01-01 → 2026-06-16, is **uneconomic
net of realistic retail costs at every rebalance cadence tried (daily through
monthly)**: across a pre-registered, exhaustively counted 17-trial selection
set spanning construction (bands, participation, spread calibration tiers),
cadence demotion, score smoothing, conviction sizing, and a two-speed netting
book, **no configuration achieves a deflated net Sharpe above zero** (best
selection-set DSR 0.191, on a trial containing no residual signal at all; the
best *residual* trial reaches DSR 0.145 and also fails the cash-hurdle
viability lens). The SPEC §10 kill-criterion fired and the else-branch of the
demotion budget is in force: the sleeve is archived.

This is the pre-registered *expected* outcome (`docs/demotion_design.md §5`),
published as designed. The harness's purpose is to make exactly this claim
bankable: the negative is certified under the same evidence bar a positive
would have needed.

## 2. Strategy under test

Residual statistical arbitrage (`src/prism/residual/`, research WFO driver
`research/scripts/stat_arb_residual_wfo.py`): PCA factor model (15 factors,
252-bar correlation window), OU s-scores on 60-bar residual regressions,
threshold state machine (entry ±1.25, exits −0.5/+0.75), dollar-neutral book,
`position_unit=0.02`, `max_gross=1.0`, `max_symbol_abs_weight=0.35`,
participation-capped at 5% ADV. Demotion-arm variants moved only the
pre-registered knobs: decision cadence (1/3/5/21 bars), s-score EWMA
(halflife 3/5), strength sizing, and the Arm-B momentum sleeve / two-speed
book (`docs/demotion_design.md §§2–3`).

## 3. Evidence base

- **Sample:** 2020-01-01 → 2026-06-16, 1,308 daily bars, 21 purged
  walk-forward folds (formation 312 bars, test 63, positions closed at fold
  end).
- **Universe:** point-in-time S&P 500 membership (891 membership intervals,
  `membership_sha 4d2fef91f80f`), 574 of 633 ever-members resolved — the
  ~9.3% survivorship coverage leak is **counted and disclosed** (I-7), not
  hidden.
- **Data convention (I-7):** split-adjusted price returns, dividends *not*
  credited in this research path; Twelve Data daily bars.
- **Fills and costs (N2, N3):** next-open market-on-open fills; commission
  1 bp; borrow 50 bp/yr on shorts; quadratic slippage coefficient 10;
  **per-liquidity-bucket one-way spread** `SPREAD_BUCKET_SCHEDULE_V1`
  (≥$500M ADV: 1 bp · ≥$100M: 2 bp · ≥$25M: 5 bp · below: 10 bp) — a
  pre-registered *conservative-upper* schedule (I-9), see §6.
- **Hurdle (SPEC §10):** after-cost viability is read against the T-bill
  cash hurdle, 3.71%/yr nominal (`basis: tbill_nominal`), converted to a
  per-trial periodic-Sharpe hurdle at each trial's realized vol and recorded
  in its summary.
- **Deflation (N5):** every searched configuration is a counted ledger trial
  — including net-negative and degenerate outcomes. Each packet's DSR is
  deflated against the ledger count at its creation (recorded per packet as
  `trial_count`); the terminal count is 17, and re-deflating earlier packets
  against 17 only lowers them, so the verdict is robust to the running count.
  "Deflated net Sharpe > 0" ⟺ DSR > 0.5.

## 4. The selection set (all 17 trials)

Ledger rows 1–4 are legacy-era residual trials (flat 1 bp spread; the
band-grid era whose −0.65 → −0.01 arc SPEC §0 records). They carry no
bucket-cost packets and enter through the deflation count. Rows 5–17 are
packet-backed (annualized net Sharpe; net total return over the sample;
DSR against the ledger at creation):

| row | run | configuration | net Sharpe (ann.) | net return | DSR | tier |
|---|---|---|---|---|---|---|
| 5 | `phaseA_rerun_noband` | daily, no band, flat 1 bp | −0.648 | −6.8% | 0.002 | gross_edge |
| 6 | `phaseA_rerun_band0.004` | daily, fixed band, flat 1 bp | +0.009 | −0.02% | 0.075 | gross_edge |
| 7 | `r2_t1_cfband_flat` | daily, closed-form band, flat | −0.350 | −3.8% | 0.008 | gross_edge |
| 8 | `r2_t2_cfband_bucket` | daily, closed-form band, buckets | −0.752 | −8.0% | 0.0004 | gross_edge |
| 9 | `r2_t3_full` | t2 + participation gate (the frozen stack) | −0.751 | −8.0% | 0.0005 | gross_edge |
| 10 | `r2_t4_sqrt_bucket` | daily, sqrt cost-aware band, buckets | −0.100 | −0.2% | 0.041 | gross_edge |
| 11 | `demotion_d1` | weekly decisions | −0.384 | −4.3% | 0.005 | gross_edge |
| 12 | `demotion_d2` | weekly + EWMA(5) | +0.091 | +0.9% | 0.069 | net_edge |
| 13 | `demotion_d3` | 3-day + EWMA(3) | −0.120 | −1.5% | 0.028 | gross_edge |
| 14 | `demotion_d4` | daily + strength sizing | −0.951 | −12.6% | 0.000 | gross_edge |
| 15 | `demotion_d5` | weekly + EWMA(5) + strength | **+0.335** | +4.3% | **0.145** | net_edge |
| 16 | `demotion_b1` | momentum sleeve *alone*, monthly | **+0.465** | +27.6% | **0.191** | net_edge |
| 17 | `demotion_b2` | two-speed book (D1 + B1), weekly | +0.081 | +1.1% | 0.042 | net_edge |

Rows 8–17 are priced under the bucket schedule; rows 5–7 under the flat 1 bp
spread (recorded per packet, I-7). All demotion trials (11–17) ran the frozen
trial-3 stack with only the pre-registered knobs moved.

## 5. The verdict

**SPEC §10 kill-criterion — fired** (adjudicated in two stages, both
recorded before the runs that resolved them):

1. *Best deflated net Sharpe < 0.* The maximum DSR anywhere in the
   selection set is 0.191 (row 16) — far below the 0.5 that a positive
   deflated Sharpe requires. Row 16 contains **zero residual signal** (it is
   the momentum sleeve alone), so the residual sleeve's own best case is row
   15 (DSR 0.145), which additionally fails viability: periodic net Sharpe
   0.0211 vs its cash hurdle 0.0936.
2. *Structural toll-booth.* At the frozen stack (row 9), cost ≥ gross PnL in
   11 of 21 folds (52%, above the ~40% line).
3. *Budget exhausted.* The demotion budget pre-registered exactly 7 trials
   (`docs/demotion_design.md §3`); the ledger closed at 17 of 17 with the §4
   success condition unmet.

**Demotion rule (§4) — else-branch in force:** no demotion trial reached
`net_edge` under bucket spreads *with* deflated net Sharpe > 0 (rows 12, 15,
16, 17 reach the raw `net_edge` tier; none clears deflation). Therefore: the
sleeve archives, this certification publishes, and no further
cadence/smoothing/sizing value may be tried against this selection set. Any
new idea is a new signal entering at `mechanics_clean` with its own
pre-registered budget.

## 6. Why these numbers can be believed

The claim inherits the full harness discipline, each element tested in-tree:

- Purged/embargoed walk-forward; no fit sees test data (I-1, I-2).
- Next-open fills; costs charged before any claim (N2, N3); the ledger
  conserves capital — the accounting algebra is machine-checked in Lean 4
  (`formal/PrismFormal/Ledger.lean`) and the float implementation is
  property-tested against it (N4).
- The online no-trade band is applied statefully against held weights; the
  batch-replay-from-flat defect that produced the legacy −0.01 headline is
  pinned as a checked counterexample
  (`formal/PrismFormal/Band.lean:batch_replay_from_zero_diverges`).
- Point-in-time universe with the survivorship leak counted (I-7).
- Every searched knob is a counted trial, including degenerate outcomes;
  the deflation is recomputable from the ledger (N5).
- Fold counts are conserved and asserted (silent count changes are treated
  as leaks).

## 7. What this does **not** certify

- **Not** "mean reversion is dead." The claim is scoped to this signal stack
  (PCA-residual OU s-scores, threshold state machine), this universe (liquid
  S&P names), this sample (2020–2026), and retail-scale costs
  (turnover × spread dominant; sqrt-ADV impact inactive at this AUM).
- **The spread schedule is a conservative upper proxy, not measured fills**
  (I-9). The pre-registered decision standing (`docs/handoff.md §8`): if
  paper-loop fills contradict the schedule, the historical verdict is
  recomputed under calibrated buckets before any new work. The paper-loop
  instrument and per-bucket estimator are built
  (`prism.live`, `prism.execution.spread`) and awaiting fills. Note the
  asymmetry: rows 5–7 show the sleeve failing even under the *optimistic*
  flat 1 bp spread, so calibration would have to undercut megacap costs
  dramatically to move the verdict.
- **Dividends are not credited** in this research path (price-return
  convention, I-7). For a dollar-neutral book the long/short dividend
  asymmetry is second-order but nonzero.
- The B2 anti-netting finding (§8) is a *construction* measurement under one
  pre-registered combination rule, not a refutation of internal netting in
  general.

## 8. Side findings (recorded, not certified)

- **B1 (row 16) is the program's next candidate signal.** Monthly 12−1
  cross-sectional momentum, decile long/short, is the only configuration in
  program history to clear the cash-hurdle viability lens (periodic 0.0293
  vs hurdle 0.0203; ≈+5.0%/yr at ~11% realized vol), at one-third the
  residual sleeve's turnover (0.050/day) and the best DSR ever recorded
  (0.191). It **fails deflation against this selection set** and was
  discovered as the best of 17 searched trials — its numbers are
  selection-biased upward and prove little by themselves. It enters at
  `mechanics_clean` under its own pre-registered budget
  (`docs/momentum_design.md`), where B1's provenance is carried as the
  honest prior.
- **The netting hypothesis failed, measurably backwards.** B2's realized
  cost (0.0761) exceeds the gross-weighted sum of its half-sized standalone
  sleeves (≈0.0593) by ~28%; turnover by ~38% (0.1358 vs ≈0.0987). Leading
  diagnosis (hypothesis, unverified): the pre-registered "sum then cap as
  one book" rule re-scales the combined row daily, so residual-sleeve
  fluctuations wiggle the frozen momentum component through the
  proportional cap, manufacturing trades in a sleeve that should be dormant
  20 of 21 days. Worth resolving before any future multi-sleeve book; per
  pre-registration it buys B2 no re-run here.

## 9. Consequences in force

- The residual sleeve is **archived**: the code remains in
  `src/prism/residual/` and `research/arbitrage/` for reproduction (deleting
  evidence would be falsification, not housekeeping), but it is no longer a
  candidate; no counted trial may target this selection set again.
- The trial ledger `results/stat_arb_residual_trials.jsonl` is closed at 17
  rows for this program and is retained append-only.
- The engine moves to the next candidate (B1-derived momentum) at
  `mechanics_clean` under a fresh pre-registered budget — the STOP branch of
  `docs/handoff.md` Phase C, executing as written.

## 10. Reproduction

The evidence travels with the repository: the trial ledger and every cited
run's `claim_packet.json` / `summary.json` / `folds.json` / `config.json`
are committed (force-added past the `results/` scratch ignore). The
time-series CSVs (equity, returns, target weights; ~5 MB per run) stay
local and are regenerable from each run's `config.json`. Every packet embeds
its full config, git commit, config hash, universe provenance (membership
SHA, coverage), spread schedule, and claim-tier rules (`schema_version: 2`).
Key anchors:

| run | packet config hash | code commit |
|---|---|---|
| `r2_t3_full` (frozen stack) | `d4c93af68855` | `d272c93` |
| `demotion_d5` | `9a876fba0196` | `194d7f9` |
| `demotion_b1` | `000b74941cfd` | `a2f4478` (clean tree) |
| `demotion_b2` | `e77aa678b157` | `a2f4478` (clean tree) |

Driver: `python -m research.scripts.stat_arb_residual_wfo` with the config
recorded in each run's `config.json`. Known wart: the ledger rows and the
claim packets hash the same config through two different writers and disagree
(e.g. B1: ledger `301c449b9173` vs packet `000b74941cfd`); linkage is via the
ledger row's `output_dir`. Flagged for unification; the underlying configs
compare byte-identical.
