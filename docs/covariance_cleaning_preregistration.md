# Covariance cleaning — pre-registration + design (nonlinear shrinkage / RMT)

**Status: drafted 2026-07-08, GATED on a second sleeve existing — no production
wiring now.** This document registers *analytical nonlinear shrinkage* (Ledoit–Wolf
2020) as the covariance/correlation cleaner for Prism's residual and risk-model
surfaces, and pins the trigger under which it may be built and wired. It is a
**diagnostic / risk-model estimator, not an alpha**: it moves no signal, adds no
configuration to any strategy's selection set, and therefore **introduces no
counted ledger trial by itself** (contrast `docs/r2_design.md`, where every
band/spread/gate value is a counted trial). Nothing here deflates a Sharpe; it only
makes the spectrum a book is measured against honest.

Scope (SPEC §7.2 residualize guarantees; SPEC §3 law 3 artifact (b)): replace the
raw sample correlation eigendecomposition that feeds the factor model and the
$N_{\text{eff}}$ breadth diagnostic with an RMT-aware cleaned estimate. Lands
default-off; the SPEC-native Marchenko–Pastur (MP) clip is the cheaper first option
and analytical NLS is its successor. **Everything below is design, not a build
order — the build gate is §5.**

## 0. Governance framing (why this is gated, and what it is not)

The cleaner's own justification is multi-consumer: it is worth building when a
covariance estimate is *shared* across more than one sleeve's construction or risk
model, because that is when eigenvalue mis-estimation stops being a single-sleeve
internal detail and becomes a cross-sleeve allocation error. Today there is
**exactly one would-be consumer, and it is archived.** The Avellaneda–Lee
residual-reversion sleeve — the only production surface that builds a sample
correlation matrix and takes its top eigenvectors — was declared uneconomic and
archived: the SPEC §10 kill-criterion fired
(`docs/certifications/001-residual-reversion-daily-negative.md §1, §5, §9`; "the
sleeve is archived"; SPEC.md §10 lines 543–553; `docs/handoff.md:216`). The
momentum program (`docs/momentum_design.md`) is the sole live candidate and is a
*single* sleeve trading decile long–short — it neither residualizes nor shares a
risk model with a second book.

So the natural trigger — **"clean the covariance when a second sleeve exists"** — is
**unmet**. Building the cleaner now would wire it into either (a) an archived
consumer (falsification if we resurrected it silently — cert 001 §9 forbids
re-targeting that selection set) or (b) a single-sleeve diagnostic where the cheaper
MP clip already captures most of the benefit. This document settles the design and makes the trigger explicit; the code lands
only at the §5 gate.

## 1. The current production gap (verified against HEAD)

Three distinct surfaces read an **uncleaned** sample spectrum today.

- **The factor model builds a raw sample correlation and clips it by count, not by
  the MP edge.** `estimate_eigenportfolios` in `src/prism/residual/factors.py:347`
  forms `corr = standardized.T @ standardized / (n_obs - 1)` and immediately
  eigendecomposes it (`np.linalg.eigh(corr)`, `factors.py:348`), keeping the top
  `n_factors`. That count is **hard-fixed at 15**
  (`ResidualStatArbConfig.n_factors: int = 15`, `factors.py:63`; the frozen-v1
  correlation window is `corr_window: int = 252`, `factors.py:61`). No shrinkage, no
  OAS, no eigenvalue clipping, and **no MP factor-count selection** — the retained
  rank is a constant, not a function of the aspect ratio. With a 252-bar window
  against an S&P-scale eligible cross-section, $q = N/T$ is $O(1)$, so the RMT bulk
  is wide and a fixed 15-factor cut has no Marchenko–Pastur justification; some
  retained "factors" are bulk noise and some discarded eigenvalues carry structure.
  (The eligible $N$ varies per rebalance; the point is qualitative and holds for any
  $q$ not $\ll 1$.)

- **Even the SPEC-specified MP-edge cleaning is unbuilt.** SPEC §7.2 states the
  guarantee explicitly: "**factor count chosen by the Marchenko–Pastur edge**
  $\lambda_+$, not a fixed `n_factors=15`; bulk eigenvalues clipped when residual
  covariance is used downstream" (`SPEC.md:358–360`). SPEC §3's load-bearing law 3
  table confirms the status split: MP factor-count selection via
  $\lambda_+ = (1+\sqrt{N/T})^2$ is marked **TO-BUILD in 0.3.0**, and the only
  **SHIPPED** member of that law is James–Stein *signal* shrinkage in
  `arbitrage/pairs.py` — not covariance cleaning (`SPEC.md:190`). So the gap is not
  merely "NLS is missing"; the cheaper SPEC-native clip is also missing.

- **The $N_{\text{eff}}$ diagnostic therefore ingests an uncleaned spectrum.**
  `effective_breadth_from_cov` (`src/prism/validation/metrics.py:393`) computes the
  participation ratio $N_{\text{eff}} = (\sum\lambda_i)^2 / \sum\lambda_i^2$ over the
  eigenvalues of whatever covariance it is handed (`metrics.py:412`), and its
  docstring already prescribes computing it "on the covariance of the *traded
  residuals* … (SPEC.md §7.2)" (`metrics.py:401`). But the matrix it receives is a
  raw sample covariance: the finished momentum breadth reading reports the
  participation ratio of the raw held-name return covariance at $\approx 11.8$ (top
  eigenvalue 7.3%) and the PnL-contribution breadth at $N_{\text{eff}} \approx 52.7$
  — both "in sample-covariance terms" (`docs/momentum_design.md §0`, lines 42–48). A
  sample spectrum at $q=O(1)$ systematically *spreads* eigenvalues (top eigenvalue
  biased up, small ones down), which biases $N_{\text{eff}}$ **downward** — the
  participation ratio, and hence the $IC\sqrt{N_{\text{eff}}}$ ceiling and the SPEC
  §10 viability gate (`metrics.py:420` `information_ratio_ceiling`; SPEC.md §10 lines
  517–524), is read off noise. This is the concrete harm the cleaner removes, and it
  is a *measurement* harm on a live diagnostic, not a strategy change.

## 2. Estimand

Let $R$ be the $T \times N$ window of standardized residual (post-neutralization)
returns for the book(s) under measurement, $C = \tfrac{1}{T} R^\top R$ its sample
correlation with eigenpairs $(\lambda_i, u_i)$. The estimand is the **population
correlation** $\Sigma$ (equivalently its cleaned eigenvalues $\{d_i\}$ under the
Frobenius-optimal rotation-equivariant estimator), of which $C$ — the object built
at `factors.py:347` — is the $q=O(1)$-biased sample. The cleaner is a map
$C \mapsto \hat{\Sigma} = \sum_i d_i\, u_i u_i^\top$ that keeps the sample
eigenvectors (rotation-equivariant, no side information) and replaces the sample
eigenvalues $\lambda_i$ with cleaned $d_i$. Two things consume $\hat\Sigma$: (a) the
$N_{\text{eff}}$ participation-ratio diagnostic (`effective_breadth_from_cov`),
which becomes a de-biased breadth estimate; and (b) any future multi-sleeve risk
model that inverts or factorizes a shared residual covariance for construction. The
estimand is identical for both; only the downstream use differs.

## 3. The cleaner: MP clip (cheaper, SPEC-native) → analytical NLS (successor)

Two rungs, registered together; the second strictly dominates the first in the
limit and reduces to it as a special case.

1. **MP clipping (the SPEC-native option, `SPEC.md:358–360`).** Set the retained
   factor count by the MP edge $\lambda_+ = (1+\sqrt{N/T})^2$ rather than the
   constant 15: eigenvalues above $\lambda_+$ are signal, the bulk at or below is
   noise. Clip the bulk — collapse sub-edge eigenvalues to their mean
   (trace-preserving) or to $\lambda_+$ — so the downstream covariance is not
   inverted through noise directions. This is a few lines over the existing `eigh`
   (`factors.py:348`), introduces one derived quantity ($\lambda_+$ from the
   realized aspect ratio), and directly discharges the SPEC §7.2 guarantee. Cost: it
   is a hard, discontinuous cut — it discards intra-bulk structure and is sensitive
   to the edge estimate when the spectrum has no clean gap.

2. **Analytical nonlinear shrinkage (the successor, Ledoit–Wolf 2020).** Replace the
   hard clip with the asymptotically optimal *continuous* shrinkage: each
   $\lambda_i$ is mapped by the oracle nonlinear shrinkage function
   $d_i = \lambda_i / \lvert 1 - q - q\,\lambda_i\, m(\lambda_i)\rvert^2$, where $m$
   is the Stieltjes transform of the limiting sample spectral density, estimated
   **directly and in closed form** from $\{\lambda_i\}$ by a kernel (the analytical
   estimator replaces the earlier QuEST numerical inversion — no optimization loop,
   $O(N^2)$ after the eigendecomposition we already pay). NLS needs **no
   factor-count choice at all**: there is no $\lambda_+$ threshold and no discarded
   directions, so it is robust when the spectrum has no gap — exactly the
   residual-covariance regime. In the degenerate no-bulk limit it returns the sample
   eigenvalues; under a wide bulk it shrinks the top toward the bulk and lifts the
   bottom, which is precisely the correction $N_{\text{eff}}$ needs.

Recommendation: register **MP clip as the first build** (it is cheap, it closes the
literal SPEC §7.2 guarantee, and it is auditable by hand) and **analytical NLS as
the ratified successor** to be swapped in once a multi-sleeve risk model actually
inverts the matrix (where the hard clip's discarded structure starts to cost real
allocation error). Both are default-off and parity-pinned: with cleaning disabled,
`factors.py` must reproduce the frozen-v1 eigenportfolios bit-for-bit.

## 4. Consumers and wiring (design only)

- **$N_{\text{eff}}$ diagnostic (immediate on trigger).** Feed
  `effective_breadth_from_cov` (`metrics.py:393`) the cleaned $\hat\Sigma$ instead of
  the raw sample covariance. This is a pure measurement upgrade: it changes no
  traded weight, only the reported breadth, ceiling, and the SPEC §10 viability read.
  It is the single change with the best benefit-to-risk ratio and the one the
  momentum breadth accounting (`momentum_design.md §0`) would consume first.

- **Multi-sleeve risk model (the actual gated consumer).** When a second live sleeve
  exists, a shared residual covariance is inverted/factorized for cross-sleeve
  construction. There, uncleaned eigenvalues produce the classic Markowitz
  error-maximization: the inverse loads on the smallest (most downward-biased)
  eigenvalues. NLS is Frobenius-optimal for exactly this use. This is the consumer
  whose *existence* is the trigger (§5); it does not exist today.

## 5. The second-sleeve trigger (the build gate)

This design is authorized to become code when **both** hold:

1. **A second live sleeve exists** — a second candidate program past
   `mechanics_clean` running concurrently with momentum, such that a covariance
   estimate is genuinely *shared* across construction or risk models (not the
   archived residual sleeve; cert 001 §9 closes that selection set). A single sleeve
   does not trip this — the cheaper MP clip already suffices for a single-book
   diagnostic.
2. **A consumer actually inverts or factorizes the shared covariance** for
   sizing/allocation, i.e. the estimate's *conditioning* — not just its
   participation ratio — is load-bearing.

Until both hold, the standing decision is: **the MP clip may be built
opportunistically to discharge the SPEC §7.2 guarantee on the $N_{\text{eff}}$
diagnostic** (it is cheap and introduces no trial), but **analytical NLS stays
drafted**. Neither rung is wired into any strategy's construction path while
momentum is the only sleeve.

## 6. Relation to the shipped James–Stein experiment (not the same object)

`docs/stat_arb_shrinkage_experiment.md` shrinks **signals** — per-pair evidence
weights via one empirical-Bayes / James–Stein step across a fold's admitted
cointegrated set, the SHIPPED law-3 member (`arbitrage/pairs.py`, `SPEC.md:190`).
That is shrinkage of a *portfolio-weight vector*, and its pre-registered megacap run
was a **clean negative after deflation**: raw net Sharpe and PSR rose, but the wider
candidate family lifted the trial count and `pair_set_dsr` collapsed, so the default
stays `fdr_hard` (that doc, Status + `construction_mode` default). **This
pre-registration is a different object**: shrinkage of the *eigenvalue spectrum of a
covariance matrix*, consumed by a risk model / breadth diagnostic, with no signal
on/off decision and no admission to any selection set. The James–Stein result
therefore neither supports nor refutes covariance cleaning — the two touch different
failure modes (signal-density vs. spectrum-conditioning). The only shared lesson is
the honesty discipline: because covariance cleaning changes a *measurement* and not
a traded weight, it escapes the DSR ledger entirely, which is precisely why it must
be gated on genuine multi-sleeve need rather than adopted because a number looks
nicer.

---

**Status: drafted 2026-07-08, GATED on a second sleeve — no production wiring now.**
No counted trial is introduced; on trigger, MP clipping (`SPEC.md:358–360`) is the
first build and analytical NLS (Ledoit–Wolf 2020) its ratified successor, both
default-off and parity-pinned against frozen-v1 `factors.py`.
