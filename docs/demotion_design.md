# §10 STOP adjudication + demotion-budget pre-registration (2026-07-06)

> **Outcome (2026-07-06, recorded after B2's packet — see §7):** the budget
> ran to completion (ledger 17/17) and the §4 else-branch fired. The sleeve
> is archived; the negative certification is
> `docs/certifications/001-residual-reversion-daily-negative.md`. Sections
> 0–6 below are the pre-registration as committed before the runs and are
> unmodified.

This document does two things, in order: it records the SPEC §10 kill-criterion
verdict on the daily residual sleeve, and it pre-registers the *one* demotion
budget the verdict routes to (handoff §8 decision table). It is committed
before any demotion mechanics exist and before any counted demotion run.

## 0. The adjudication record

Adjudicated on trial 3's claim packet (`results/r2_t3_full/claim_packet.json`,
config `d4c93af68855`, code commit `d272c93`, created 2026-07-06), per
`docs/r2_design.md` §5.

**Condition 1 — fired.** Best deflated net Sharpe across the counted set is
< 0 under the calibrated bucket schedule. Annualized net Sharpe: t2 −0.75,
t3 −0.75, t4 −0.10 (all `spread_mode=bucket`). The only non-negative counted
config (phaseA band=0.004, +0.009 net annualized, total return −0.0002) ran
under the flat 1 bp spread and is indistinguishable from zero; its
selection-set DSR is 0.075. Trial 3's DSR is 0.0005 against 9 counted trials.

**Condition 2 — fired.** `cost_to_gross_pnl` ≥ 1 in 11 of 21 folds (52%,
above the ~40% line): the sleeve is a structural toll-booth at daily cadence.

**Condition 3 — not fired.** 10 counted trials of the ~30 budget.

**Verdict: STOP at daily frequency.** The residual-reversion sleeve is
declared uneconomic at daily cadence — demoted, not iterated further. Routing
follows the pre-registered decision table (handoff §8): the weekly-cadence
demotion + slow-sleeve netting book receives **one** pre-registered budget —
this document. Its failure fires the else-branch automatically: archive the
sleeve, publish the negative claim packet. No appeals inside the selection
set.

**Standing caveat, recorded with the verdict:** the bucket schedule is a
conservative *upper* proxy (r2_design §3, I-9) — net-negative under it is not
yet net-negative under measured fills. The paper-loop fill calibration
(handoff Phase A) proceeds in parallel and can recontextualize this verdict;
it does not suspend it.

## 1. Baseline: everything frozen at the trial-3 stack

All demotion trials run the exact trial-3 configuration — closed-form band
(`gamma_risk=1.0`), `spread_mode=bucket` (SPREAD_BUCKET_SCHEDULE_V1),
`max_participation=0.05`, frozen-v1 signal parameters, same universe and date
range — with only the §2 knobs moving. Every combination below is one counted
ledger trial; `--design_trials` on every run counts the full set here plus the
10 already ledgered.

## 2. Demotion mechanics (all default-off; defaults are bit-identical parity)

- **`decision_every`** (int ≥ 1, default **1** = frozen parity). State-machine
  decisions and trade execution are restricted to *decision bars*: bars where
  `(t − window_start) % decision_every == 0`, so the first bar of every
  window is a decision bar. Between decision bars the target row is frozen at
  the last decision row and the online loop executes nothing (`target :=
  held`; the book drifts). This is the semantics of a weekly-*traded* book,
  not a weekly-sampled signal on a daily-traded book. The closed-form band's
  formation replay applies the same rule, so `sigma2_target` reflects
  cadence-consistent target changes.
- **`sscore_ewma_halflife_bars`** (float, default **0** = off). Causal
  columnwise EWMA of the s-score panel before the state machine. Days where
  the raw s-score is NaN stay NaN in the smoothed panel (smoothing never
  resurrects an invalid OU fit); EWMA state carries across NaN gaps.
- **`sizing_mode="strength"`** — already implemented
  (`strength_multiplier`, construct.py) and never counted; enabling it is a
  trial like any other.

## 2b. Arm-B mechanics semantics (amended 2026-07-06, before any counted run)

Fixes the seams §3's one-line B1/B2 descriptions left open. No §3 trial value
changes; this section only pins how the mechanics behave, and it is committed
before any demotion trial runs.

- **Score**: `close[t-21] / close[t-252] - 1` (knobs `mom_lookback_bars=252`,
  `mom_skip_bars=21`), masked to the SAME eligibility screen the residual
  sleeve trades under. Insufficient history or ineligibility → NaN → no
  position, never "trade anyway".
- **Row**: `n_dec = floor(n_finite × mom_decile)` with `mom_decile=0.10`;
  winners `+0.5/n_dec` each, losers `-0.5/n_dec` each — raw gross exactly
  1.0. Stable sort, so ties are deterministic. `n_dec < 1` emits a zero row.
- **Cadence seam**: the momentum component refreshes at window bars where
  `(t - window_start) % mom_decision_every == 0` (`mom_decision_every=21`);
  the emitted row picks a refresh up at the *next trading decision bar*. In
  B1 (`decision_every=21`) the two cadences coincide exactly; in B2
  (`decision_every=5`) a refresh at offset 21 executes at offset 25. This lag
  is a recorded consequence of the two pre-registered cadences, not a knob.
- **B2 gross split**: raw target rows are summed pre-cap (residual raw gross
  ≤ 1.0 by construction, momentum raw gross 1.0), then the combined book is
  capped to the same `max_gross=1.0` / `max_symbol_abs_weight=0.35` and
  banded/gated as one. The sleeve split is therefore an emergent fixed rule
  (≈ equal at typical residual gross), never a searched weight.
- **Band consistency**: the closed-form band's formation replay includes the
  sleeve, so `sigma2_target` reflects combined-book target changes; formation
  bars without `mom_lookback_bars` of history emit zero momentum rows.

## 3. The counted trial set (exactly these, in this order)

Arm A — cadence / smoothing / sizing on the trial-3 stack:

| id | decision_every | ewma_halflife | sizing |
|----|----------------|---------------|--------|
| D1 | 5 | off | unit |
| D2 | 5 | 5 | unit |
| D3 | 3 | 3 | unit |
| D4 | 1 | off | strength |
| D5 | 5 | 5 | strength |

Arm B — the slow-sleeve netting book (mechanics are Phase-B scale; the arm is
fixed *now*, before any Arm-A result is seen, to kill adaptive selection):

- **B1** — 12−1 cross-sectional momentum sleeve alone: formation return over
  bars (t−252, t−21], top/bottom decile, equal-weight long−short,
  `decision_every=21`, same cap/band/gate/cost stack. Diagnostic *and*
  counted.
- **B2** — the two-speed book: D1's weekly residual sleeve + B1's momentum
  sleeve, target rows summed, then capped/banded/gated as one book at the
  same `max_gross=1.0`. Fixed to D1's cadence regardless of Arm-A outcomes.
  The headline diagnostic is the internal-netting cost delta: B2's cost toll
  vs the gross-weighted sum of its sleeves' standalone tolls.

Budget: **exactly 7 counted trials**, taking the ledger to 17 of ~30. No
other knob moves. Any value not in this table is out of budget.

## 4. The adjudication rule (pre-committed)

After B2's packet: the demotion succeeds iff some demotion trial's claim
packet reaches **`net_edge` under bucket spreads with its deflated (selection-
set DSR) net Sharpe > 0**. Otherwise the else-branch fires: the sleeve is
archived, the negative packet is published as the harness's first
certification, and no further cadence/smoothing/sizing value may be tried —
any new idea is a new signal entering at `mechanics_clean`. A Arm-A-positive
that appears *without* Arm B's netting must additionally survive the §10
falsification gate with extra scrutiny (see §5).

## 5. Expectation, recorded honestly

The prior is that Arm A alone fails: at trial 3 the toll is ~3.4× gross
(0.90 bp/day cost vs 0.27 bp/day gross), weekly sampling of a ~5-day-half-life
signal decays gross alongside cost, and trial 4's quasi-experiment (activity
cut ~15×, avg gross 0.049) converged to ≈0 net, not positive. The doctrine's
mechanism lives in Arm B (internal netting; G-P's native multi-signal case —
handoff §3). This paragraph exists so a surprising Arm-A positive is treated
as a falsification-gate event, not a celebration, and so the negative branch
is recognized as the pre-registered *expected* outcome, publishable without
embarrassment.

## 6. Exit criteria (tests that gate the mechanics landing)

- Parity: `decision_every=1`, EWMA off, `sizing_mode=unit` reproduces the
  trial-3 pipeline bit-for-bit (pinned test).
- Cadence property: on the synthetic panel, realized turnover is
  non-increasing in `decision_every`.
- Causality: appending future bars never changes past smoothed s-scores; NaN
  days stay NaN through the EWMA.
- Arm-B mechanics (momentum sleeve, book summation) land default-off with
  their own parity + property tests before B1 runs: `sleeve_mode="off"`
  reproduces frozen-v1 bit-for-bit; the raw momentum row has gross exactly
  1.0 and balanced legs; momentum scores are causal (appending future bars
  never changes past scores); `two_speed` targets equal the sum of the
  standalone sleeves' targets when caps/band/gate are inactive; and every
  Arm-B knob moves the config hash (an unhashed knob would be an uncounted
  trial, SPEC N5).

## 7. Outcome record (added 2026-07-06, after B2's packet; §§0–6 unmodified)

All 7 trials ran (D1–D5 at commit `194d7f9`, B1–B2 at `a2f4478`, clean
trees), taking the ledger to 17 of 17. Applying §4: rows D2 (+0.091 ann.),
D5 (+0.335), B1 (+0.465) and B2 (+0.081) reach raw `net_edge`; none clears
deflation (best DSR 0.191 = B1, which contains no residual signal; best
residual DSR 0.145 = D5, which also sits under its cash hurdle, periodic
0.0211 vs 0.0936). **The else-branch fires: sleeve archived, negative packet
published, no appeals inside this selection set.** The §5 expectation held —
Arm A alone did not rescue the sleeve, and the surprise was in the opposite
direction: B2's netting diagnostic came out *anti*-netting (cost ~+28%,
turnover ~+38% vs the gross-weighted standalone sum), recorded as a
construction finding in the certification (§8 there). B1 is routed per §4's
last sentence: a new signal entering at `mechanics_clean`
(`docs/momentum_design.md`). Full evidence, scope, and caveats:
`docs/certifications/001-residual-reversion-daily-negative.md`.
