# Learned cross-section X0-X4 — read record, 2026-09-23

**Kind:** read record for the kill-class probes X0-X4 of the frozen
`docs/learned_xsection_design.md` (ratified 2026-07-19; A4 opt-in 2026-07-22),
run under `registrations/2026-09-learned-xsection.toml` (seam: design §8).
**Selection set:** `learned_xsection_v1`. **Registration commit:** `04dcb34`.
**Budget:** 6 for the family; 5 spent here, X5 remains. **Records:**
`results/2026-09-learned-xsection/` (`comparison.json`, `read.json`,
`returns/`). **Trial log:** `trials/learned_xsection_v1.jsonl`.

## Verdict

**Fragility kill: survives. Cost wall: X3 and X4 record the condition-2 read.
Admission: refused. The recorded verdict is the design's own sentence:
"positive-carry, spanned; admission refused."** X0's daily net returns are
spanned by the momentum sibling (intercept t = 0.59 against the certified
book, −0.28 against the same-engine book; the design requires t ≥ 2). On the
same sessions, the fixed 12-1 momentum book and the residual-momentum
candidate both out-earned the learner. Learning the combination of the seven
F0 characteristics did not add edge over the fixed score on this sample.
X5, the promotion read, stays on its clock (on or after 2027-07-19) with its
budget of one; nothing here is cited for it.

## The attempt history, including the failure

The first attempt of every arm (`kind = "new"`, five `failed` rows) refused to
score during fold admission: `prism.evaluate._training_panels` scored each
component over the formation window without the point-in-time membership
mask, and the node fails loud without it rather than train on a pool that
includes future index joiners. The mask was bound there (`b51b51f`) and each
arm reran as `kind = "retry"`, which spends no budget. No failed attempt
reached a scored bar, so no outcome was seen before the retries.

## An exclusion the seam did not anticipate

The seam (design §8) chose folds 12-20 of the recorded geometry, 552 test
sessions. The shared calibration family's admission rule, which every
registration in this program applies to every arm alike, then excluded the
first four of those folds: their formation windows begin before the learner's
36-cross-section runway is met, so they hold 1, 4, 7 and 10 usable periods
against a declared minimum of 12. **The scored sample is 300 sessions,
2025-04-03 to 2026-06-12, and 14 non-overlapping decision cross-sections.**
The exclusion is identical across X0-X4, so the arms remain comparable with
one another; it makes every statistic below very noisy, and it was not
foreseen when the seam was pinned. It is recorded rather than repaired: a
geometry chosen after seeing this would be a new configuration.

## Results

Canonical engine, $100,000, experiment profile, frozen dataset
`data/frozen/demotion_b1`, modeled fills.

| probe | net Sharpe | gross Sharpe | ann. return | ann. vol | max DD | charged cost | N_eff | held-book IC ± se (n = 14) | DSR vs set |
|---|---|---|---|---|---|---|---|---|---|
| X0 (pinned) | 1.005 | 1.051 | 13.4% | 13.4% | 6.6% | 0.73% | 24.5 | 0.058 ± 0.066 | 0.769 |
| X1 λ = 10 | 1.009 | 1.054 | 13.5% | 13.5% | 6.7% | 0.73% | 24.5 | 0.060 ± 0.067 | 0.770 |
| X2 drop `rev_1` | 1.142 | 1.185 | 16.0% | 13.8% | 7.1% | 0.72% | 23.6 | 0.078 ± 0.070 | 0.809 |
| X3 18 cross-sections | 0.475 | 0.525 | 5.7% | 13.6% | 8.1% | 0.81% | 25.0 | −0.004 ± 0.073 | 0.571 |
| X4 pinned GBM | 0.812 | 0.884 | 8.0% | 10.1% | 5.4% | 0.87% | 35.8 | 0.052 ± 0.066 | 0.703 |

On the same 300 sessions, for context (raw daily Sharpe, annualized): X0
1.005; `characteristics_v1` `momentum_control` 1.441; its
`residual_momentum_only` 1.753; the certified B1 book 1.265. 2025-04 to
2026-06 was a strong period for momentum, and every book here shares it.

No pairwise interval excludes zero except X2 − X3 (lower +0.0012 per bar,
p 0.092); λ = 1 against λ = 10 differs by −0.0035 annualized, the two ridges
being nearly the same fit at these sample sizes.

## The pre-committed reads (design §4)

- **Fragility kill.** Median net annualized Sharpe of X1-X4 is 0.91, not
  negative. No probe flips the sign of X0's net result. **Survives.**
- **Cost-wall inheritance.** Charged cost is at or above gross PnL in 1 of 5
  scored folds for X0, X1 and X2, and 2 of 5 (40%) for X3 and X4. X3 and X4
  therefore record the cert-001 condition-2 read. Both of their walled folds
  are folds with negative gross, not folds with large costs: charged cost is
  under 0.2% of equity in every fold.
- **Incrementality (the admission case).** (a) `net_edge` with DSR 0.769
  against this set: met. (b) Spanning: X0's daily net returns regressed on the
  sibling's over the 300 common sessions, Newey-West with 21 lags:

  | regressor | alpha, annualized | alpha t | beta | correlation |
  |---|---|---|---|---|
  | certified B1 (`results/demotion_b1/returns.csv`), the pinned read | +4.7% | 0.59 | 0.51 | 0.52 |
  | same-engine sibling (`characteristics_v1` `momentum_control`) | −1.9% | −0.28 | 0.73 | 0.80 |

  Neither reaches t ≥ 2. **Admission refused: positive-carry, spanned.**

Repurposing any probe as a standalone sleeve would be a new pre-registration,
not a reinterpretation of this one (design §4).
