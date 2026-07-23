# Learned cross-section pre-registration — frontier axis, monthly-cadence learned ranking (DRAFT)

> **Status: RATIFIED 2026-07-19** (this commit is the dedicated ratification
> commit; the decision was delegated by the owner to the session in the same
> turn — the `docs/momentum_design.md` delegation precedent; drafted
> 2026-07-18 under the owner's frontier directive). **The design is frozen
> from this commit: the out-of-sample clock runs.** The §6 flagged values are
> resolved as written: `rev_1` stays in F0, X4 stays in budget. From this
> commit the amendment rule applies: seams may be pinned, no trial value may
> move. **No backtest of any configuration in this family runs outside the
> §3 counted set** — the budget-amnesia failure mode (`docs/handoff.md`
> §7.2) stays blocked exactly as in the sibling designs.
>
> **A4 concurrency opt-in: RATIFIED 2026-07-22** (this commit is the
> dedicated opt-in commit; owner-authorized on `main`; mechanism
> `docs/amendments_2026-07.md` A4; this banner contemplated the opt-in at
> family ratification; queue item under `docs/v040_program.md` §5).
> **No trial value in §1–§4 moves under this seam.** Binding posture:
>
> 1. **Kill/fragility reads X0–X4** may run concurrently with the
>    adjudication-slot holder under A4's three criteria: (a) this
>    pre-registration is ratified and frozen; (b) X0–X4 cannot promote
>    ("they can kill, they cannot promote", §3); (c) outputs are
>    **firewalled** — barred from every cross-family adjudication
>    (momentum M6, trend T5, aim-portfolio G4b, any other promotion) until
>    this family holds the serial promotion slot for X5.
> 2. **Promotion read X5 always requires the serial adjudication slot.**
> 3. Every A4-concurrent read still appends to this family's ledger and
>    counts against the frozen budget of 6 exactly as if it ran in the slot.
> 4. Concurrency remains *permitted, not mandated* (operator bandwidth;
>    equity attention ladder in `docs/v040_program.md` §7 still ranks this
>    family after trend when bandwidth is scarce).
>
> Uncounted mechanics and paper were never gated by the slot.

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
`trend_v1` (≤ 6, A4 opt-in RATIFIED 2026-07-22). **A4 opt-in RATIFIED
2026-07-22 (banner box):** X0–X4 may run A4-concurrent and firewalled; X5
still requires the serial adjudication slot. A4 is not a replication-style
zero-dof exception; it is the standing criterion for non-promoting reads
from any ratified frozen family. A budget-6 counted search still counts
every cell against this family's never-refilled budget.

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
(`docs/trend_design.md` §3 clause, transplanted whole). X0 is the primary
read of the pinned cell; it is not a promotion read (promotion is X5 only)
and therefore sits with X1–X4 under the A4 "cannot promote" criterion
(banner opt-in RATIFIED 2026-07-22).

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

- **A4 concurrency opt-in:** RATIFIED 2026-07-22 (banner box above). No
  further owner act is required for kill-class X0–X4 concurrency under A4;
  operator bandwidth may still choose to run nothing concurrent. The §6
  free-window pins (`rev_1` stays; X4 stays) were already resolved at family
  ratification and are not reopened by this seam.
- **Paper instrument:** same posture as `docs/trend_design.md` §5 — neither
  requested nor foreclosed here; it would be its own seam amendment.

## 7. What this document does not do

It does not touch the momentum program, its budget, or M6. It does not
blend anything. It does not buy data or add dependencies beyond the frozen
stack's. It does not revive the residual sleeve or any daily-cadence trade.
It runs nothing now. Its budget is fixed at 6 and does not refill.
