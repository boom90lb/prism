# Handoff — the long-horizon doctrine

**Standing.** `SPEC.md` is the constitution: rules, contracts, gates. This
document is the jurisprudence and the strategy — *why* the rules are what they
are, what to build over the next 18–24 months, which decisions are already
made, and the ranked ways the project can fail. It binds nothing; where it
disagrees with `SPEC.md`, the spec wins. It exists because the constitution
deliberately does not editorialize, and a future maintainer (human or agent)
needs the editorial record — the reasoning that produced the rules — so the
rules can be amended on their merits rather than cargo-culted or quietly
abandoned.

Code-level claims below originate in the 2026-07 deep audit and are tagged
*(audit)*; anchors are re-verified against the current tree before being acted
on.

---

## 1. The asset ledger

Ranked by durable value, most to least:

1. **The evaluation methodology, made executable.** Purged/embargoed WFO,
   next-open costed fills, claim tiers, per-selection-set DSR deflation, trial
   ledgers, cost-toll and breadth diagnostics, ledger-conservation property
   tests, leakage regression tests. Retail backtests are almost universally
   lies; a harness that refuses to lie is rare and does not decay on
   publication. This is the moat.
2. **The constitution itself.** A spec that carries a numeric kill-criterion
   for its own flagship strategy, pre-registered trial budgets, and a
   non-goals section that survives contact with enthusiasm — that discipline
   is an asset independent of any code.
3. **The PIT universe builder** (`prism.scripts.build_sp500_universe` +
   coverage ledger). Point-in-time membership with the survivorship leak
   *counted* instead of hidden is production-grade and genuinely uncommon at
   $0.
4. **The execution/accounting spine** (`execution/target_weights.py`,
   `costs.py`): t+1 fills, borrow, dividends-as-cash, sqrt-ADV impact — the
   best-tested subsystem, now backed by machine-checked accounting algebra
   (`formal/`, §5).
5. **The residual signal core** (`residual/factors.py`, `residual/residual.py`)
   — real engineering with known numeric-integrity defects *(audit: NaN→0
   imputation into factor returns; stale betas on mid-window ineligibility;
   eigenvector sign discontinuity under degenerate sigma)*. Valuable after the
   fix pass, not before.
6. **The alpha.** Best net ≈ −0.01, un-deflated, under an uncalibrated cost
   model. Not an asset yet. Possibly never. The project's design assumption is
   that items 1–5 outlive item 6 in either case.

The strategic consequence: **protect the harness's credibility above every
feature.** Any change that would let a number be reported above its evidence
tier is a worse bug than any crash.

## 2. The strategy in one paragraph

Prism is the honest certification engine that can also trade. Deployment is
the mandate and it is gated hard (`SPEC §10`); the certification machinery is
the durable public asset either way. Near-term work therefore has one shape:
make the kill-criterion's own inputs trustworthy (calibrated costs, fixed
deflation accounting, fixed residual numerics), run the cheap experiments the
criterion contemplates (frequency demotion, slow-signal netting), and let the
verdict fire honestly. If the daily residual sleeve clears `net_edge`, deploy
at trivial size and scale by the capacity curve. If it dies, the sleeve is
demoted or archived as a first-class ledgered negative result, the harness's
first public certification — and the engine moves to the next candidate
signal with the same bar. Either branch is a success state; only an
un-fireable criterion is failure.

## 3. Economic doctrine

The load-bearing insight of the whole program: **at this scale the binding
cost is turnover × effective spread, and turnover is a property of the signal
*set*, not of construction.** Everything below follows from it.

- **A single fast signal is the degenerate case.** A ~5-day-half-life
  reversion signal is close to maximum turnover per unit of IC; no band can
  rescue it beyond what is already measured (−0.65 → −0.01). Gârleanu–Pedersen
  — the framework `SPEC §7.3` canonizes — is derived for *multiple signals
  with heterogeneous decay rates*; running it with one fast signal uses the
  framework where it has the least to give.
- **Slow signals are cost machinery, not new alpha.** A 1–3 month
  cross-sectional momentum sleeve (from bars already on disk) reduces cost per
  unit of gross through internal netting: reversion buys what momentum sells
  into, and the trades cross before touching the market. This is why the
  gating doctrine says "no new *fast* alpha" rather than "no new alpha"
  (`SPEC §13` as amended): a slow, negatively-turnover-correlated signal is
  construction-adjacent and belongs with R2, not behind it.
- **Frequency is a free parameter that was never searched.** The
  kill-criterion's own demotion clause (2–3 day or weekly cadence, EWMA-smoothed
  scores, continuous sizing in s instead of the ±1.25/−0.5 threshold state
  machine) costs hours and is admissible *now* as counted construction trials.
  The threshold state machine is a 2010-paper artifact that manufactures
  turnover at band edges; expected information per engineering hour here beats
  the closed-form band derivation.
- **Capacity is insurance, not the constraint.** At $2k–$10k, sqrt-ADV impact
  does not bind (the −0.65 was produced with `adv_impact_coeff=0`). Keep the
  capacity curve as scaling-readiness, but fix its denominator first — it is
  referenced to static `initial_capital` *(audit; `SPEC §7.4` already flags
  it)*, so its output is wrong at every AUM except the first dollar.
- **The cost-calibration circularity is the deepest structural trap.** I-9
  wants spreads calibrated from fills; fills are gated behind `net_edge`;
  `net_edge` is judged under the uncalibrated flat `spread_bps=1.0`. The
  verdict −0.01 sits inside the cost model's error bars *in both directions*
  (ETB megacap retail fills often price-improve inside the half-spread;
  small-caps cost multiples of 1bp). **Break the loop with the paper loop:**
  Alpaca paper, trivial size, trading the current zero-edge book purely as a
  cost-measurement instrument. It costs $0, it forces `live/` (durable order
  state, reconciliation) into existence, and every fill is I-9 calibration
  data. Nothing about a paper loop needs an edge. This is why it moves from
  R4 into R2 (`SPEC §13` as amended).

## 4. Evidence doctrine

The claim-tier ladder is the public API of the project. Long-term, a *claim
packet* — config hash, code commit, data convention, trial ledger, tier — is
what Prism certifies; strategies are just inputs. Doctrine:

- **The ledger is append-only and hostile to amnesia.** Every searched knob is
  a counted trial *including degenerate outcomes*: NaN-Sharpe configs were
  searched and must deflate the survivor *(audit: the residual sweep deflates
  against finite-Sharpe trials only, so the false-strategy benchmark sits too
  low)*. A deflation that cannot be recomputed from the ledger is not a
  deflation.
- **Silent count changes are leaks.** `PurgedWalkForward` skipping empty folds
  without a downstream assertion *(audit)* is the same defect class as
  uncounted trials: the denominator moved and nobody signed the change. Fold
  counts, trial counts, and universe counts are all conserved quantities —
  treat a count discrepancy exactly like a ledger-conservation failure.
- **A headline number must have a reproduction path in-tree.** The −0.01
  currently does not: the online band (`step_no_trade_band`) is implemented
  and tested but wired into no pipeline, and the batch replay is the wrong
  semantics for continuing a book — a divergence now machine-checked as a
  theorem (`formal/PrismFormal/Band.lean:batch_replay_from_zero_diverges`).
  Wire it or de-claim the number; those are the only two honest states.
- **Fix the residual numerics before the verdict fires.** The known integrity
  defects *(audit: NaN→0 imputation damping factor variance; stale frozen
  betas; sign-flip discontinuity)* mostly *inflate* gross — so the fix pass
  strengthens whatever verdict follows, and skipping it taints either verdict.
- **The kill-criterion is a feature, not a threat.** The single most likely
  long-term failure mode of this project is the operator (or an agent)
  iterating the dying sleeve past its pre-registered budget because a fix is
  always one experiment away. The budget (≤ ~30 counted construction trials)
  exists precisely for that moment. When it exhausts: fire the criterion,
  publish the negative packet, demote or archive. No appeals inside the same
  selection set.

## 5. The formal foundation (Lean 4 charter)

`formal/` is a core-only Lean 4 package (no Mathlib — builds in seconds)
machine-checking the spec's *checkable algebra* over exact integer arithmetic:

| Theorem | Invariant | What it pins |
|---|---|---|
| `Ledger.rebalance_conserves`, `Ledger.run_conserves` | N4 | Trading and multi-period accounting create/destroy no cash; equity moves only by PnL − costs |
| `Ledger.mark_to_market_pnl` | N4 | PnL is attributable entirely to held quantity × price move |
| `Band.stepBand_idem`, `Band.stepBand_trades_only_past_band` | §7.3 | Band hysteresis is stable and trades only strictly past the width |
| `Band.batch_replay_from_zero_diverges` | §7.3 | The batch-replay-from-zero defect, as a checked counterexample — the online form is not optional |
| `Purge.purge_label_disjoint`, `Purge.train_avoids_embargo` | I-1 | Purge/embargo index geometry: no label window reaches a test slice; embargoed rows excluded |
| `Participation.capTrade_le_cap` / `_down_only` / `_sign_*` | §7.4 | The gate is a pure attenuator: capped, never amplifying, never sign-flipping |

**The division of labor is the whole design.** Lean proves the *algorithm*
over ℤ; pytest proves the float64 *implementation* tracks the algorithm
(property tests + tolerance). The Lean proofs are never to be cited as
verifying the Python — they verify the algebra the Python is tested against.
Both halves together are the guarantee; neither alone is.

**What is deliberately not formalized** (pre-registered, to keep proof effort
where it pays):
float error propagation (bound it empirically, per-run, in the property
tests); the DSR/PBO statistics derivations (analytic results from the
literature — formalizing expected-max-Sharpe asymptotics is a Mathlib-scale
research project with no engineering payoff); anything about market behavior.
Formal methods here verify *bookkeeping*, never *beliefs*.

**Next formal targets, in value order:** (1) the live-loop crash-safety state
machine once `live/` exists — decide/submit/fill/reconcile with a crash
transition between any two steps; prove no order is lost or doubled from
durable state (this is where formal methods will catch what tests won't,
because the failure needs a crash at an exact seam); (2) the G-P band's
down-only/monotonicity properties when R2 lands it; (3) trial-ledger
append-only monotonicity as a tiny state machine. The package stays
core-only; any proposal to import Mathlib must name the theorem that needs it
and why a property test is not the right tool instead.

## 6. The long-term roadmap

Extends `SPEC §13`; phase letters continue the R-series. Each phase has an
exit artifact, and the standing rule is unchanged: cost-bound before
signal-bound, everything counted.

**Phase A — make the verdict trustworthy (now → +2 months).**
Honesty plumbing: grid-size-aware DSR deflation (count searched, not
surviving, trials); fold-count assertion in the WFO consumers; wire
`step_no_trade_band` into the residual WFO (or formally de-claim −0.01);
residual numerics fix pass (NaN imputation, stale betas, sign continuity);
re-run the residual slice so −0.65/−0.01/+0.23 become load-bearing numbers.
Frequency-demotion experiments as counted trials. **Paper loop up** (Alpaca,
trivial size) as the I-9 cost instrument, with durable order state — the
embryo of `live/`. *Exit: a claim packet whose every number has an in-tree
reproduction path and a calibrated (or explicitly bracketed) spread
assumption.*

**Phase B — the two-speed book (+2 → +6 months).**
Slow sleeve (1–3 month cross-sectional momentum) admitted as construction
machinery; G-P multi-signal band with heterogeneous decay rates (its native
case); internal netting measured as a cost-toll delta; incremental data store
+ fail-loud data layer (429/Retry-After handling, dividend negative-cache TTL,
error-vs-absence distinction — a rate-limit failure must never be bookable as
a delisting). *Exit: the kill-criterion fires — GO or STOP — on calibrated
inputs, and either verdict ships as a public claim packet.*

**Phase C — deployment or certification (+6 → +12 months).**
GO branch: live at minimum size, rolling PSR/DSR live monitor armed as the
kill-switch, scale strictly along the (fixed-denominator) capacity curve.
STOP branch: the negative result is written up as the harness's first public
certification; the residual sleeve archives; the next candidate signal enters
at `mechanics_clean` and climbs. Either branch: stabilize the Signal/Construct
/Execute contracts as a documented public API — external users bring signals,
Prism certifies them. That is the open-source product regardless of the
alpha's fate.

**Phase D — the platform (+12 → +24 months).**
Second market lane (crypto time-series book) only after the equity verdict,
under its own `net_edge` bar and venue-priced fees. Contributor surface:
CONTRIBUTING.md, claim-packet reader docs, mypy + multi-Python CI, hypothesis
property tests for N4/N1 paths. (Prophet left the core dependencies in the
R1 quarantine — `uv pip install prism` no longer builds cmdstan; xgboost
stays core, `prism.signal.ensemble_node` uses it.) Delete the RL
trio outright (git history preserves it; the seed study extracted its value;
carrying known-broken members in a public repo is a standing credibility
tax). The regime layer is wired through I-8 IC gates or explicitly demoted to
diagnostics — zero-importer modules do not count as shipped capability.

**The data-budget expiry condition** (pre-registered, so the decision is not
rationalized in the moment): the $0 constraint is a discipline, not an
identity. Buy data or
infrastructure exactly when a `net_edge`-tier claim's capacity curve shows the
edge funding the purchase at deployable AUM — never to *find* the edge, only
to *feed* a proven one.

## 7. Failure modes, ranked by expected damage

1. **Verdict on untrustworthy inputs.** Firing (or passing) the kill-criterion
   under the flat 1bp spread and the current deflation accounting. Phase A
   exists to prevent this; do not reorder it away.
2. **Budget amnesia.** Un-ledgered "quick checks" during construction work.
   Every sweep counts, including the degenerate ones. The ledger is the
   memory the operator's enthusiasm doesn't have.
3. **Tested-but-not-wired.** `step_no_trade_band` is the canonical instance: a
   correct, tested, *unreachable* fix under a headline number produced by
   different semantics. Any "shipped" claim now requires an importer on the
   claimed path; shelf-ware is counted as such (the `regime/` rule).
4. **Silent data rot.** Empty-frame-on-failure, poisoned negative caches,
   429s booked as delistings — each one manufactures survivorship bias on top
   of the disclosed ~9%. Fail-loud is a merge gate (N7), not a preference.
5. **The dying-sleeve loop.** Emotional sunk-cost iteration past the trial
   budget. Pre-registered STOP; no appeals within the selection set.
6. **Docs drifting below the code.** The README describing the retired system
   for months was scope *minimization* — undersold capability is also a
   truthfulness failure and it costs contributors. The supremacy chain
   (SPEC → README/ARCHITECTURE → code comments) gets re-audited at every
   release tag.
7. **Constitution creep.** Amending N1–N8 casually, or — worse — routing
   around them "temporarily." Amendments happen in dedicated commits with
   written rationale, never bundled with feature work.

## 8. Pre-registered decisions

So they are not relitigated under pressure:

| Trigger | Decision |
|---|---|
| Kill-criterion fires STOP | Weekly-cadence demotion + slow-sleeve netting book gets *one* pre-registered budget; else archive the sleeve, publish the negative packet |
| Kill-criterion fires GO | Deploy at minimum viable size; scale only along the capacity curve; live monitor armed from day one |
| Paper fills contradict the 1bp assumption | Recompute the entire historical verdict under calibrated buckets before any new work — the past numbers change meaning |
| Crypto lane proposal before the equity verdict | No. One verdict at a time; the evidence bar machinery is shared and the ledger discipline doesn't parallelize across an operator of one |
| PyPI name collision on `prism` | Qualify the distribution name only; the import package stays `prism` (`SPEC §12`, already decided) |
| Anyone proposes intraday anything | `SPEC §8`. Structural, not empirical. Closed. |

## 9. Onboarding protocol

Read order for a new maintainer or agent: `SPEC.md` → this file → `MARKETS.md` →
`docs/audit.md` (historical, for the *why* behind R0–R4) → `ARCHITECTURE.md`
→ the claim packets under `results/`. The non-goals (`SPEC §8`) and the
pre-registered decisions (§8 above) are settled; spend disagreement budget on
open questions (hurdle basis, slow-sleeve design, live-loop state schema),
not settled ones. Amendments to `SPEC.md`: dedicated commit, rationale in the
commit body, invariant changes flagged in the PR title. The trial ledger and
claim packets are the project's lab notebook — they transfer with the repo,
and deleting or "cleaning" them is falsification, not housekeeping.

The project's one non-replicable asset is that its numbers can be believed. Every future decision that trades credibility for
convenience — an uncounted trial, a silent fallback, an overclaimed tier — is
the project ending in slow motion. Guard the bar; the alpha is optional.
