# Pinned interpretations

Where the constitution's text underdetermines a reading and the ambiguity is
load-bearing, the reading is pinned here — before the data that would make
the choice self-serving exists. A pin is an interpretation of existing text,
never a change to a trial value, budget, or gate threshold; anything beyond
interpretation is an amendment (`docs/amendments_2026-07.md` pattern). Each
entry records what was ambiguous, the pinned reading, and why it was pinned
when it was.

## PI-1 — The claim-tier ladder is cumulative, and the viability bound is the one-sided lower-95 (ratified 2026-07-19)

**Ratification:** owner, 2026-07-19, in-turn, after adversarial review of the
projection arithmetic under both candidate bounds. Pinned before any M6
extension data exists; adjudicating this in 2027 with the numbers visible
would have tainted whichever way it went.

**Ambiguity (i) — ladder cumulativity.** SPEC §7.6's five-tier ladder states
`robust_edge` = "`net_edge` and DSR ≥ threshold" but the `net_edge` row never
states whether it carries `gross_edge`'s conditions (falsification gate,
viability gate). The M6 promotion conjunct is written in tier vocabulary
("`net_edge` under bucket spreads on the extended sample",
`docs/momentum_design.md` §3) and contains no explicit viability term — so an
uncumulative reading would let a book promote while its own fundamental-law
ceiling cannot clear cash.

**Pinned: the ladder is cumulative.** Each tier carries every condition of
the tiers below it. `net_edge` at M6 therefore includes the `gross_edge`
viability gate, read on the extended sample.

**Ambiguity (ii) — the bound.** SPEC §7.6 gates viability "on the **lower**
CI bound" of the rank IC with a bootstrap CI — no level, no sidedness. The
discovery-run record (`docs/momentum_design.md` §0) exhibits two different
bounds in adjacent bullets: the IC's one-sided lower-95 (−0.009) and a
viability-lens bound whose recorded ceiling (0.069) implies IC ≈ +0.0095 — a
z ≈ 0.86 bound corresponding to no level named anywhere. The two readings
differ by an order of magnitude in what promotion requires (below).

**Pinned: the bound is the one-sided lower-95** (bootstrap where the
diagnostic computes one; Gaussian-on-the-mean otherwise, stated in the
artifact). The §0 lens bullet's z ≈ 0.86 figure is superseded for gating
purposes; it stands in the record as a discovery-time diagnostic at an
unpinned level.

**Ambiguity (iii) — the extension read.** Pinned: at M6 the bound is
re-estimated on the extended sample under the same protocol (rank IC at the
stated horizon, IC never estimated on the gated fold, periodic units on both
sides — SPEC §7.6's existing conventions, unchanged).

**Consequence, recorded at pin time.** At the discovery-run point estimate
(IC 0.030, SE 0.024, n = 62, √N_eff ≈ 7.3, hurdle 0.093 periodic): the
one-sided lower-95 is −0.0095 today, −0.0061 at n = 74, −0.0035 at n = 86;
CI shrinkage alone clears the hurdle only near n ≈ 324. Promotion at M6
under this pin therefore requires the extension sample to raise the point
IC materially — an on-trend year routes to §3's "waits for more OOS data"
branch, which is the apparatus working, not failing. This price was computed
and accepted at ratification; the rejected alternative (the z ≈ 0.86 lens
bound, under which two on-trend years reach the line) was rejected because
no constitutional text names its level and because a real-money promotion
gate should not be the weaker of two available readings.

**Routing.** A same-sample viability failure with passing conjuncts routes
to the already-ratified wait branch (`docs/momentum_design.md` §3: "waits
for more OOS data — it does not buy more same-sample trials"). Neither this
pin nor its consequence moves any trial value, budget, or threshold.
