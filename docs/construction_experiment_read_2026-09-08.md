# Stage 4 construction experiment — read record, 2026-09-08

**Kind:** read record for the registered construction experiment
`registrations/2026-09-construction.toml` (spec §3.13; decisions D5, D6, D8, D9).
**Selection set:** `momentum_construction_v1`. **Budget:** 3 declared, 3 spent — one
attempt is one `arm_id`, so an arm's six capital scenarios spend one attempt between
them. **Records:** 18, one per (arm, scenario) pair. **Comparison record:**
`results/2026-09-construction/comparison.json`, with the same tables as
`comparison.txt`. **Trial log:** `trials/momentum_construction_v1.jsonl`.
**Frozen score panel:** `data/frozen/momentum_construction_v1.freeze.json`
(63 decision rows × 574 names, `panel_sha256` `23fda286…`).

## Verdict: INCONCLUSIVE on the Sharpe difference, and one arm is structurally different

All eighteen studentized Sharpe-difference intervals straddle zero; the smallest
two-sided p is 0.262. On this sample, at this precision, **the experiment cannot
distinguish any arm from any other on net Sharpe.** That is the headline and it is not a
disappointment to be worked around: 1,056 daily observations do not resolve annualized
Sharpe differences of the size these arms produce. The annualized intervals run about 1.3
to 2.0 wide, against point differences of 0.00 to 0.48.

What the point estimates say, reported because the contract says report them whatever
they are:

- `convex_point` is **lower than the control at every one of the six capital levels**
  (annualized net Sharpe 0.344–0.433 against 0.525–0.679), and the difference is never
  distinguishable from zero (p 0.31–0.79). It is the arm this experiment comes closest to
  killing, and it is not killed: the interval contains both zero and a control advantage
  of roughly 0.9 annualized.
- `convex_p80_one_sided` is **higher than the control at three capital levels and lower at
  three** (0.397–0.915 against 0.525–0.679), with no interval excluding zero. It holds
  about half the control's gross, takes about half its turnover, and pays about half its
  charged cost.
- The two convex arms hold nearly the same *names* and different *sizes*: at $100,000 the
  name-level Jaccard between them is 0.9995 while the mean L1 weight distance is 0.484.
  Both differ structurally from the control, whose Jaccard against either is 0.198. They
  are different books, not near-copies: at $100,000 the control's target holds 96 names at
  a decision (94 to 100 across decisions), `convex_point`'s holds 36 (0 to 458) and
  `convex_p80_one_sided`'s 27 (0 to 497). The convex arms concentrate into fewer names than
  the decile rule on a typical date and spread into many on a few, which is what the
  variance budget and the per-name cap between them produce. That the arms are structurally
  different is what makes the inconclusiveness a statement about precision rather than
  about the arms being the same book twice.

## Per-arm, per-scenario results

Annualized net Sharpe, annualized net return, charged cost as a fraction of equity over
the scored sample, total turnover, median at-cap fraction, median ex-ante standard
deviation, batch-rounding repair outcome counts (none / scaled / dropped) and the number of
decisions the arm was flat because its endpoint interval straddled zero:

```
   capital arm                      sharpe  ann.ret     cost  turnover  at_cap   ex_sd  none/scale/drop  flat
    10,000 decile_control           0.5253   0.0331   0.0198     28.97   0.000     n/a         51/0/0      0
    25,000 decile_control           0.6232   0.0393   0.0209     31.63   0.000     n/a         51/0/0      0
    50,000 decile_control           0.6151   0.0387   0.0210     31.86   0.000     n/a         51/0/0      0
   100,000 decile_control           0.6789   0.0428   0.0209     31.64   0.000     n/a         51/0/0      0
   250,000 decile_control           0.6541   0.0412   0.0209     31.72   0.000     n/a         51/0/0      0
 1,000,000 decile_control           0.6562   0.0413   0.0209     31.75   0.000     n/a         51/0/0      0
    10,000 convex_point             0.4331   0.0273   0.0190     30.25   0.000  0.0332         48/1/2      0
    25,000 convex_point             0.3490   0.0220   0.0196     31.96   0.000  0.0332         50/0/1      0
    50,000 convex_point             0.3961   0.0250   0.0197     31.92   0.000  0.0332         51/0/0      0
   100,000 convex_point             0.3973   0.0250   0.0197     32.05   0.000  0.0332         51/0/0      0
   250,000 convex_point             0.3436   0.0216   0.0196     31.67   0.000  0.0332         50/0/1      0
 1,000,000 convex_point             0.3981   0.0251   0.0198     32.13   0.000  0.0332         51/0/0      0
    10,000 convex_p80_one_sided     0.9150   0.0576   0.0105     16.04   0.000  0.0166         44/0/4     18
    25,000 convex_p80_one_sided     0.4435   0.0279   0.0111     17.26   0.000  0.0332         48/0/3     18
    50,000 convex_p80_one_sided     0.3969   0.0250   0.0114     18.29   0.000  0.0332         50/0/1     18
   100,000 convex_p80_one_sided     0.4939   0.0311   0.0109     16.69   0.000  0.0166         47/0/1     18
   250,000 convex_p80_one_sided     0.6911   0.0435   0.0112     17.62   0.000  0.0166         48/0/0     18
 1,000,000 convex_p80_one_sided     0.8103   0.0510   0.0076      8.30   0.000  0.0000         34/0/0     18
```

Every arm made 51 decisions in the seventeen admitted folds; `n/a` in `ex_sd` is recorded,
not missing — see "the budget binds only the convex arms" below.

## The Sharpe differences

Studentized paired block bootstrap, every parameter declared in the registration before
the run: daily frequency, circular moving blocks of 42 bars (the decision interval is 21),
2,000 resamples, seed 20260908, HAC lag 41. Both arms of a pair are resampled on the same
blocks. Eighteen intervals were computed and the count is in the record's selection block,
because the multiplicity is part of the selection history.

```
   capital difference                                       diff    lower    upper      p  diff/yr  lower/yr  upper/yr
    10,000 decile_control_minus_convex_point              0.0058  -0.0367   0.0462  0.789   0.0921   -0.5829    0.7337
    10,000 decile_control_minus_convex_p80_one_sided     -0.0246  -0.0763   0.0204  0.297  -0.3898   -1.2115    0.3241
    10,000 convex_point_minus_convex_p80_one_sided       -0.0304  -0.0931   0.0169  0.262  -0.4819   -1.4772    0.2689
    25,000 decile_control_minus_convex_point              0.0173  -0.0262   0.0587  0.422   0.2743   -0.4154    0.9324
    25,000 decile_control_minus_convex_p80_one_sided      0.0113  -0.0372   0.0573  0.635   0.1797   -0.5911    0.9092
    25,000 convex_point_minus_convex_p80_one_sided       -0.0060  -0.0565   0.0348  0.781  -0.0945   -0.8969    0.5525
    50,000 decile_control_minus_convex_point              0.0138  -0.0271   0.0516  0.483   0.2190   -0.4301    0.8185
    50,000 decile_control_minus_convex_p80_one_sided      0.0137  -0.0317   0.0565  0.532   0.2182   -0.5038    0.8968
    50,000 convex_point_minus_convex_p80_one_sided       -0.0000  -0.0408   0.0367  0.999  -0.0008   -0.6480    0.5826
   100,000 decile_control_minus_convex_point              0.0177  -0.0222   0.0543  0.369   0.2817   -0.3516    0.8628
   100,000 decile_control_minus_convex_p80_one_sided      0.0117  -0.0363   0.0553  0.601   0.1850   -0.5758    0.8777
   100,000 convex_point_minus_convex_p80_one_sided       -0.0061  -0.0482   0.0317  0.753  -0.0966   -0.7655    0.5025
   250,000 decile_control_minus_convex_point              0.0196  -0.0192   0.0545  0.314   0.3105   -0.3047    0.8657
   250,000 decile_control_minus_convex_p80_one_sided     -0.0023  -0.0489   0.0398  0.919  -0.0370   -0.7769    0.6317
   250,000 convex_point_minus_convex_p80_one_sided       -0.0219  -0.0641   0.0177  0.308  -0.3475   -1.0172    0.2807
 1,000,000 decile_control_minus_convex_point              0.0163  -0.0240   0.0533  0.412   0.2582   -0.3807    0.8461
 1,000,000 decile_control_minus_convex_p80_one_sided     -0.0097  -0.0686   0.0412  0.727  -0.1541   -1.0895    0.6541
 1,000,000 convex_point_minus_convex_p80_one_sided       -0.0260  -0.0937   0.0309  0.435  -0.4122   -1.4876    0.4912
```

The statistic is the per-bar Sharpe difference; the `/yr` columns are the same interval
scaled by `sqrt(252)`. The endpoints are directional statements under the resampling
assumptions and are not a simultaneous interval across the eighteen.

## Folds excluded, for every arm alike

Folds 0, 1, 2 and 3 are excluded, in every one of the eighteen records, for
`min_training_periods`: their formation-window training pools carry 1, 4, 7 and 10 decision
periods with a complete observation against the declared minimum of 12. The cause is
geometric and was decided before any result was seen. A 312-bar formation window holds
about 15 monthly decision periods; the 12-1 momentum score needs 253 bars of history, so a
fold whose formation window begins at the sample's own first bar can score only its last
few periods, and the labelled pool is shorter still because a period's open-to-open label
must also end inside the window. From fold 4 on there is enough earlier history and every
fold is admitted with exactly 13 usable periods — one above the declared minimum of 12,
drawing 5 blocks of 3 against the declared `min_blocks = 4`. The margin is one period wide,
which is worth knowing: a geometry change that shortened the formation window by a month
would start excluding admitted folds.

Admission is read off the shared calibration family (`calibrations/shared_slope_point.toml`)
and never off an arm's own `confidence`, which is decision D6. The control arm fits no
slope and reads the same count through `prism.decisions.usable_periods`; `calibrate_fold`
refuses to proceed if that count and a fitted arm's own `n_periods` disagree, and
`tests/test_decisions.py` pins the two against each other. The comparison record carries
`exit_conditions.excluded_folds.shared = true`, which is the checkable form of "excluded
for all three arms alike". The scored sample is therefore 1,056 bars, identical for every
arm, which is what makes the paired resampling legitimate.

## Identical charged cost conventions, and what that does not mean

Held identical across every arm and every capital level: the frozen cost stack
(`examples/reference/costs/retail_v1.toml`, sha256 in the comparison record), the execution
profile, the accounting engine, the fold geometry and its hash, the dataset and visibility
mode, the experiment profile's limits, and the rules that apply all of them.

Not identical, by construction: **charged dollars**. Participation is trade dollars over
dollar volume from current equity; impact is charged on filled notional; rounding at the
per-share increment changes filled notional. The sweep exists to measure that, and the
measurement is in the table above: the control pays 0.0198 of equity at $10,000 and
0.0209–0.0210 at every larger account. At $10,000 a decile weight of about 1% of equity is
$104, a fraction of a share in the book's more expensive names, and the order does not
survive rounding — the run logs name the drops. What that costs the book is visible
directly in its gross: the control carries 0.888 average gross at $10,000 against 1.002 at
$100,000, an eleven-point shortfall against a target the larger account reaches. Its
annualized Sharpe is 0.15 lower in the same scenario; the two facts are recorded side by
side and the causal share is not separated here.

## Repair outcomes at each capital level

Batch rounding repair is recorded per decision, per arm, per scenario
(`performance.repair` in each record, `exit_conditions.arms[...].by_capital` in the
comparison record). Every decision that reached rounding is counted, and the counts are in
the table above.

Fourteen repair events across the eighteen runs, and **none of them on the control**,
which rounded 51 of 51 batches cleanly at every capital level. Thirteen are dropped
batches and one is a scaled batch:

- twelve drops with `binding_condition = max_name_weight`: the *rounded* batch breached
  the 0.10 per-name cap that the solved target satisfied, because rounding a small
  account's position up by one share moves its weight past the cap, and the repair could
  not reduce it. `implemented_gross_fraction` is 0.0 on every one — the batch was not
  implementable at all, not merely shrunk. They fall at $10,000 (2 for `convex_point`, 3
  for `convex_p80_one_sided`), $25,000 (1 and 3), $50,000 (1), $100,000 (1) and $250,000
  (1), so this is mostly but not only a small-account effect;
- one drop with `max_gross` binding (`convex_p80_one_sided` at $10,000, 2026-05-06);
- one scaled batch, `convex_point` at $10,000 on 2025-05-05, with `variance_budget`
  binding: scaled to 0.75 and implementing 0.721 of the intended gross, which cleared the
  declared `min_implemented_gross_fraction = 0.80` floor only after the scale schedule ran.

That is what "how often repair fails" reads as here: of the 918 decisions the experiment
made, 13 had their rounded batch dropped and 1 was scaled, all on convex arms, and the
mechanism behind twelve of the thirteen drops is rounding against the per-name cap rather
than the account being unable to afford the book. A further 26 decisions submitted no
orders for a different reason — the no-action path below — and the two are counted
separately in the record.

`convex_p80_one_sided` also took the **no-action** path on 26 decisions in total — 3 each
at $10,000, $100,000 and $250,000 and 17 at $1,000,000, none at $25,000 or $50,000 — all
with `binding_condition = economic_objective`: the program's own objective determined the
outcome, no batch was justified, and no orders were submitted. Nothing is rounded on the
no-action path, so those decisions are counted separately from a dropped batch, and the
`no_action` count is in `performance.repair` beside the rounding outcomes.

## The per-name cap, and whether the budget bound

The declared acceptance check is `max_at_cap_fraction = 0.35`: an arm whose median
fraction of names at the per-name cap exceeds it is reported as concentration-dominated.
**Neither convex arm is.** The median at-cap fraction is 0.000 at every capital level for
both, and the maximum over decisions is 0.12 (`convex_point`) and 0.094
(`convex_p80_one_sided`). The decile control's is zero by construction.

The variance budget did bind, which is the point of declaring it (decision D8): the median
ex-ante standard deviation of `convex_point` is 0.033200 against a declared budget of
0.0332 at every capital level, and `variance_budget` is the binding condition on 28 to 34
of its 51 decisions. `convex_p80_one_sided`'s recorded median of 0.0166 is an artifact of
a bimodal set and not a risk level the arm ran at: at $100,000 its 48 measured decisions
are 24 at exactly zero (the arm holds nothing) and 24 at exactly the budget, so the median
of an even-length list is their midpoint. Read the two modes, not the median. Its binding
condition is the variance budget on 9 to 23 decisions.

The control's ex-ante standard deviation is **not measured**, and the record says so rather
than reporting it satisfied. The control's declared construction carries no risk model, so
its path has no snapshot, and §3.7's own table says the heuristic path does not enforce
`variance_budget` or `beta_band`; `prism.decisions.measurable_limits` drops exactly those
two and records them in `unchecked_limits`. Reporting a satisfied budget for a book whose
risk was never evaluated would be the silent-zero defect this engine refuses. The
consequence for the reading is explicit: **the budget constrains the convex arms and not
the control**, so the convex arms did not lose by being allowed more risk, and the control
was not held to the model risk the convex arms were held to.

## The uncertainty arm goes flat, as declared

`convex_p80_one_sided` is flat on 18 of its 51 decisions at every capital level, because
its 80% one-sided lower endpoint is negative on those dates and the robust term's maximum
then sets `a'w = 0`. §3.7 states this in advance and classifies it as uncertainty rather
than a sign contradiction, so the arm stays enabled and holds nothing. It is a reportable
property of the calibration's precision on this book, not a defect to patch and not a
reason to widen the arm after seeing it.

Separately, the shared slope is *negative* on the first two admitted folds
(`theta_shrunk` −0.0234 and −0.0049 on folds 4 and 5 against a declared
`expected_sign = "positive"`), so both convex arms take no new directional exposure on
those folds' decisions under the §3.4 sign policy, and the record carries
`sign_contradiction` per decision.

## No arm promotes

The registration's `[promotion_rule]` is `kind = "none"` and
`Registration.promotion_rule.promotes()` is `False`; the comparison record carries that
outcome rather than evaluating a threshold. This is a same-sample comparison: three arms
read against each other on one frozen score panel, one fold geometry and one cost stack,
after the fact. Such a comparison can kill a configuration and can never promote one. The
reference book's promotion stays under the existing prospective rules of `SPEC.md`, with
its own profile, its own record and its own forward evidence; nothing recorded here may be
cited for it.

## How the arms were run

Every arm decides **inside** the simulation. The constructor of §3.7 cannot be evaluated by
a pass that produces every target before the first fill: the participation restriction is
trade dollars over dollar volume from *current* equity, batch rounding is at the per-share
increment against current equity, and the reduce-only box is written on the held position.
So `walk_forward_evaluate` gained `decision_coupling = "account_state"`, which calls the
constructor at the close of each bar with the account as it stands
(`prism.execution.simulate.DecisionCall`); the coupling is part of the run's
`evaluation_context` and therefore of its `config_id`. All three arms run under it, and all
three go through the same limit-evaluation shim, the same independent post-solve
validation, the same batch rounding and the same second validation, so "repair scaled the
batch" means the same thing in the control's record as in a convex arm's.

The decile control is not the convex program: it is the rank-to-weight rule, the
participation restriction applied where the live path applies it, and the §3.7 shim, which
records every holding-state limit it can measure and repairs none. `solver` is `null` and
`path` is `normal` on all 51 of its decisions, at every capital level.

## Reproducing

```
python -m prism.scripts.construction_experiment --freeze
python -m prism.scripts.construction_experiment --arm <arm> --capital <scenario>   # 18 times
python -m prism.scripts.construction_experiment --compare --records <18 record paths> \
    --out results/2026-09-construction/comparison.json
```

Rerun determinism was measured, not assumed. Every one of the eighteen configurations was
executed twice — a shakedown pass and the recorded pass — and all eighteen pairs carry the
same `config_id` and return paths identical to 0.0 (max absolute difference, all 1,056
bars, all 18 cells). `decile_control` at $100,000 was executed three times and all three
agree the same way. The shakedown pass of `decile_control` and `convex_point` ran under an
earlier build of the decision path whose only behavioral difference was how a held name
with no close on the decision bar would be marked; the identical return paths show that
case never arose on this dataset.

## What this read does not establish

- The comparison is same-sample and cannot promote anything.
- The intervals are wide enough to contain both a large control advantage and a large
  convex advantage in most cells. Nothing here is evidence that the constructor helps, and
  nothing here is evidence that it does not; the correct reading is that this sample cannot
  tell.
- The calibration's training cross-section is the formation window's point-in-time
  membership at the fit cutoff, without the per-bar eligibility screen the decision path
  applies bar by bar. The screen is applied at every decision bar.
- The frozen dataset carries no cash-distribution events, so the `credited` label of §3.4
  credits zero and the accounting credits zero. The two agree; neither says what a
  dividend-bearing panel would produce.
- The calibration label reads the run's own open panel, truncated to the formation window
  so that no label reaches a test bar. On a dataset carrying bar revisions that panel is
  not identical to the one visible at the fit cutoff; this dataset imports no revisions,
  and the same panel is what the accounting fills against.
- Fills are simulated at the next open under the declared `[fills]` policy. No panel here
  is evidence about actual fills, and every panel in the comparison record is labeled
  `fill_source = simulated`.
- The trial log carries 74 rows — 37 started and 37 finished — where the comparison reads
  eighteen: a shakedown pass of all three arms is kept, because the contract records every
  evaluated configuration and a re-run is a row rather than an attempt. The comparison
  record's `runs` block names the eighteen it read and the rows it did not, and an attempt
  is one `arm_id`, so the repetitions spend no budget: three attempts against a budget of
  three.
- Those repeat rows carry `kind = "new"` and should carry `kind = "reproduction"`, the
  §3.12 kind for re-executing a recorded configuration with no search. The kind is a
  driver argument and the driver defaulted it; the log is append-only and is not rewritten
  to hide the fact. Nothing downstream is affected here — the attempt count is over
  distinct `arm_id` and is 3 either way — but a driver that defaulted the kind by looking
  for a finished row at the same `config_id` would have labelled them correctly, and that
  is the fix.
