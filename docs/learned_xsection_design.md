# Learned cross-section pre-registration — frontier axis, monthly-cadence learned ranking (DRAFT)

> **Status: DRAFT — not ratified.** Drafted 2026-07-18 under the owner's
> frontier directive. Ratification is the owner's dedicated commit; from that
> commit the design freezes, the amendment rule applies (seams may be pinned,
> no trial value may move), and every subsequent bar accrues as out-of-sample
> by construction while the program queues — the binding act is ratification,
> not the first run (`docs/trend_design.md` banner precedent). **No backtest
> of any configuration in this family runs before ratification, and none runs
> after it outside the §3 counted set** — the budget-amnesia failure mode
> (`docs/handoff.md` §7.2) is blocked here exactly as in the sibling designs.

## 0. Provenance and the honest prior

This family is the routed successor to the residual-reversion post-mortem,
entering at `mechanics_clean` per the demotion adjudication's own routing
sentence: "any new idea is a new signal entering at `mechanics_clean`"
(`docs/demotion_design.md` §4). It is **not a stat-arb revival**: no residual
formation, no OU state machine, no daily cadence. The autopsy pinned the
constraint that shapes this design — the daily sleeve died on a cost wall
(cost ≥ gross PnL in 52% of folds, `docs/demotion_design.md` §0), with a
real gross signal underneath (ETF-factor residuals gross +0.54 annualized,
`docs/stat_arb.md` v1 result) — so the frontier question this family asks is
posed at the cadence where the measured cost regime is survivable: the
sibling momentum sleeve's `decision_every=21` grid. The question: **does
learning the combination of slow cross-sectional characteristics add net
edge over the fixed single-characteristic score?** The sibling *is* this
family's null hypothesis.

The prior, stated so it cannot inflate: linear characteristic combinations
are heavily published and decay after publication (McLean–Pontiff); the
learned-cross-section literature (Lewellen's linear combinations;
Gu–Kelly–Xiu's ML zoo) reports gross or generous-cost results discovered
over enormous effective search spaces. This program charges the bucket
schedule and deflates against its own ledger — the exact discipline under
which the demotion set's raw net +0.465 read as DSR 0.191
(`docs/demotion_design.md` §7). The central case is that X0 lands within
noise of the sibling and admission fails; that branch is publishable as a
certification, same as cert 001. A surprising positive is treated as a
falsification-gate event first (`docs/demotion_design.md` §5 register).

**Prior in-repo exposure, disclosed:** feature 1 below (12−1 momentum) is
the sibling's entire score, certified and in flight — its inclusion is the
nesting that makes the null exact, not a discovery. The retired forecasting
harness trained learners over price features inside its own folds; it is in
the armed-retired state, appears in no ledger this family counts against,
and is not consulted. Discovery here is literature-blind, not history-blind:
publication multiplicity is the residual contamination, and it is why §2
pins canonical published characteristics rather than any tuned variant.

## 1. Universe, data, era

Identical to the sibling program in every respect: the PIT S&P 500 universe
artifacts and eligibility screens of the certified momentum lineage
(currently 37ed61308aca, `docs/data_integrity_diagnostic.md` §11), the same
spine-vendor caches, the same era as the certified window. Zero new data
dependencies; free tier; no fundamentals. The replication family owns era
extension; this family claims nothing about pre-2020. One honest
consequence recorded now: the §2 training window consumes ~36 refresh
events of runway, so X0's tradable segment is shorter than the sibling's
full window — the §4 spanning read is defined on the overlapping segment,
and the shorter sample is the recorded price of walk-forward fitting, not a
free choice.

## 2. Features, learner, construction — pinned

**F0, exactly these seven, all causal and daily-spine-derivable.** At each
decision bar t, per name, computed from closes/volume through t, then
cross-sectionally rank-transformed to [−0.5, +0.5] on that bar (monotone,
outlier-robust; rank preserves the decile sort, which is what makes the
parity nesting below exact):

1. `mom_12_1` — close[t−21]/close[t−252] − 1 (the sibling's score, verbatim)
2. `mom_6_1` — close[t−21]/close[t−126] − 1
3. `rev_1` — close[t]/close[t−21] − 1 (monthly reversal)
4. `vol_63` — std of daily returns over 63 bars
5. `maxret_21` — max daily return over trailing 21 bars
6. `high_52w` — close[t]/max(close over (t−252, t])
7. `dv_63` — median daily dollar volume over 63 bars

A name lacking history for a feature carries NaN: NaN rows never enter
training and a NaN-featured name takes no position — never "trade anyway"
(`docs/demotion_design.md` §2b masking discipline).

**L0 — cross-sectional ridge, every value pinned now.** At each decision bar
t (the sibling's `decision_every=21` grid), pool the 36 most recent
cross-sections at 21-bar spacing whose targets are fully realized (formation
bars s with s + 21 ≤ t; the target is the name's (s, s+21] return,
cross-sectionally rank-transformed on s+21's realized panel — demeaned by
construction). Fit ridge with λ = 1.0 on the rank-transformed features; the
score at t is the fitted linear combination of t's feature ranks. Fewer than
36 realizable cross-sections → no fit, no positions (runway, not an error).

**Construction: the learned score replaces the raw score, nothing else
moves.** Decile long−short, equal weight, raw gross 1.0, and the frozen
stack of the sibling: `max_gross=1.0`, `max_symbol_abs_weight=0.35`,
closed-form band, `SPREAD_BUCKET_SCHEDULE_V1`, 5% participation,
`decision_every=21` (`docs/demotion_design.md` §1–§2b). **Parity nesting:**
with the learner bypassed and the score pinned to `mom_12_1` alone, the book
must reproduce the sibling's bit-for-bit — this is the family's central
parity test, and it is what makes "learning added X" a measurement rather
than an attribution story.

## 3. Counted trial set (ratify or amend, then frozen)

Namespace `learned_xsection_v1`; budget **exactly 6 counted trials**, never
refilled; degenerate and NaN outcomes count. Every run appends to the global
trials ledger; any rerun redirects `--trial_ledger` (the recorded
unconditional-append gotcha). Prior counted programs at drafting: four —
residual reversion (closed at 17, cert 001), momentum (in flight, holding
the slot), `momentum_replication_v1` (≤ 3, concurrency exception granted),
`trend_v1` (≤ 6, queued). This family queues behind the momentum verdict
under amended SPEC §10 and **claims no concurrency exception** — a budget-6
counted search is exactly what the replication family's budget-3/zero-dof
grounds exclude (`docs/trend_design.md` §0 took the same position).

| id | delta vs X0 | probes |
|----|-------------|--------|
| X0 | — (the pinned §2 configuration, first run) | the primary read |
| X1 | λ = 10 | does the result survive shrinkage toward equal weights? |
| X2 | drop `rev_1` | is the highest-turnover feature load-bearing? |
| X3 | training window 18 cross-sections | window sensitivity |
| X4 | pinned GBM in place of ridge (depth 2, 200 trees, lr 0.05, min leaf 50, no subsampling, seed 20260718; implementation seam to mechanics, hyperparameters frozen here) | is nonlinearity load-bearing? |
| X5 | X0 re-run, sample extended ≥ 1 year past ratification, OOS segment reported separately | the promotion read |

X1–X4 are **fragility detection only** — they can kill, they cannot
promote, and the pinned configuration remains X0's regardless of which probe
scores best; adopting a different cell would be a new discovery event
(`docs/trend_design.md` §3 clause, transplanted whole).

## 4. Adjudication (pre-committed)

- **Fragility kill (readable after X1–X4):** dropped without appeal if the
  median net annualized Sharpe of X1–X4 is negative, or if any single probe
  flips the sign of the net result at magnitude greater than X0's own point
  estimate.
- **Cost-wall inheritance:** any trial with `cost_to_gross_pnl` ≥ 1 in ≥
  40% of folds records the cert-001 condition-2 read — a structural
  toll-booth verdict for that configuration, whatever its Sharpe.
- **Incrementality read (the admission case; readable on X0, re-read on
  X5):** the claim on offer is *learning adds edge over the fixed score*,
  never standalone Sharpe. Admission requires (a) `net_edge` under the
  frozen stack with DSR > 0.5 against the `learned_xsection_v1` selection
  set, and (b) the spanning read: X0's daily net returns regressed on the
  certified sibling book's daily net returns over the overlapping segment
  show positive intercept with t ≥ 2. A positive-Sharpe book spanned by the
  sibling fails its admission case at any Sharpe: the recorded verdict is
  "positive-carry, spanned; admission refused," and repurposing as a
  standalone sleeve is a new pre-registration, not a reinterpretation.
- **Promotion (readable only at X5):** `net_edge` with DSR > 0.5, periodic
  net Sharpe above the then-current cash hurdle, and the spanning read
  passing on the post-ratification OOS segment. Deployment and blending then
  follow their own law: `docs/handoff.md` §8 GO branch at minimum size; any
  combination with the momentum or trend sleeves is the aim-portfolio's
  jurisdiction.
- **Neither fires:** the program waits for more out-of-sample data; the
  budget does not refill.
- **M6 barrier:** outputs of this family are BARRED from the momentum
  program's M-series adjudication in both directions
  (`docs/replication_preregistration.md` precedent).

## 5. Mechanics and exit criteria (tests that gate the mechanics landing)

Mechanics land **default-off** under the base signal contract
(`src/prism/signal/momentum_node.py` precedent); landing mechanics is not a
trial and may happen any time. Gates:

- Parity: learner bypassed, score = `mom_12_1` → reproduces the sibling
  pipeline bit-for-bit (pinned test).
- Causality: appending future bars never changes any past score; the
  training-panel inequality (s + 21 ≤ t) is property-tested, not assumed.
- NaN discipline: a name with any NaN feature takes no position; NaN rows
  never enter the fit; smoothing/ranking never resurrects an invalid name.
- Determinism: identical inputs → identical fitted weights and book (X4's
  seed pinned above).
- Every knob hashes into the config — an unhashed knob is an uncounted
  trial (SPEC N5).

## 6. What is deliberately left to owner decisions

- **Ratification** — the dedicated commit, or an amendment first: the two
  most contestable pinned values are flagged for the free pre-ratification
  window: whether `rev_1` stays in F0 (it buys the X2 probe its subject but
  raises expected turnover), and whether X4 stays in budget or defers
  nonlinearity to a v2 family.
- **Paper instrument:** same posture as `docs/trend_design.md` §5 — neither
  requested nor foreclosed here; it would be its own seam amendment.

## 7. What this document does not do

It does not touch the momentum program, its budget, or M6. It does not
blend anything. It does not buy data or add dependencies beyond the frozen
stack's. It does not revive the residual sleeve or any daily-cadence trade.
It runs nothing now. Its budget is fixed at 6 and does not refill.
