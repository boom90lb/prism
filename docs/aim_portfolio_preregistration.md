# Aim-portfolio pre-registration — the multi-signal Gârleanu–Pedersen implementable frontier (DRAFTED, GATED)

**Status: drafted 2026-07-08, GATED on the momentum verdict — no counted trial
until the gate opens.** This document registers the design and the trial-accounting
*intent* of the multi-signal aim-portfolio; it spends nothing. It is not ratified
and not runnable. The gate, the trial budget it draws against, and the frozen trial
table below all become live only when the momentum promotion read
(`docs/momentum_design.md §3`) returns a verdict. Until then this is a design
record that pins the construction now so it is not improvised when the gate opens.

## 0. Scope and the gate

Scope (`SPEC §7.3`/`§13`, R2 multi-signal): replace the current "sum the per-signal
target rows, then re-cap the combined book daily" combination rule with a
Gârleanu–Pedersen aim-portfolio that combines heterogeneous-decay signals *inside*
the optimization, and — one step further out — weights those signals against
realized after-cost utility (the JKMP implementable frontier) rather than against
IC. Everything here lands default-off behind the shipped single-signal seed;
enabling any piece is a counted ledger trial.

**The gate is the momentum verdict, and it is hard.** The aim-portfolio is a
*multi-signal* construction; it has nothing to blend until a second signal survives
on its own bar. The residual daily reversion sleeve was archived as the harness's
first negative certification
(`docs/certifications/001-residual-reversion-daily-negative.md`), so the only live
slow-signal candidate is 12−1 cross-sectional momentum, and its status is
`mechanics_clean` under its own pre-registered budget. Its promotion is readable
**only** at M6 + paper: `net_edge` under bucket spreads on the sample extended
through ≥ 2027-06, DSR > 0.5 against the momentum ledger, periodic net Sharpe above
the then-current cash hurdle, and a live-monitor read not in contradiction
(`docs/momentum_design.md §3`). This aim-portfolio sits at Phase B of the
sequencing — "the two-speed book (+2 → +6 months)… G-P multi-signal band with
heterogeneous decay rates (its native case)" (`docs/handoff.md §5`) — which is
gated behind Phase A's trustworthy verdict and, per the pre-registered decision
table, is the "slow-sleeve netting book [that] gets *one* pre-registered budget"
only after the kill-criterion resolves (`docs/handoff.md §8`). Until the momentum
verdict opens that gate, **zero counted trials run for this item.** A blend built on
an unpromoted signal would be spending deflation budget to combine noise.

## 1. The failure this fixes: sum-then-cap is anti-netting

The doctrine that justifies a slow sleeve at all is internal netting — a slow,
negatively-turnover-correlated signal buys what the fast signal sells into, so the
trades cross before touching the market (`SPEC §13`; `docs/handoff.md §3`). The
shipped combination rule does not deliver it. In
`research/arbitrage/residual_walk_forward.py` the `two_speed` sleeve forms each
signal's target row independently and combines them by addition, then caps the
combined book:

```
row = row + mom_current          # residual_walk_forward.py:328
...
return cap_book(targets, walk_config.max_gross, walk_config.max_symbol_abs_weight)   # :331
```

This is combination *outside* the optimizer. Costs never enter the blend; the daily
`cap_book` re-scales the summed row to the fixed gross budget every session, so the
fast sleeve's day-to-day fluctuations wiggle the frozen slow component through the
proportional cap and manufacture trades in a sleeve that should be dormant 20 of 21
days. The measurement is on the record and it ran the wrong way: the B2 two-speed
book's realized cost (0.0761) *exceeded* the gross-weighted sum of its half-sized
standalone sleeves (≈0.0593) by **~28%**, and turnover by ~38% (0.1358 vs ≈0.0987)
— an *anti*-netting result, recorded as a construction finding, not a signal finding
(`docs/certifications/001-residual-reversion-daily-negative.md §8`;
`docs/demotion_design.md §7`). The certification's leading diagnosis (recorded
there as hypothesis, unverified) names the candidate mechanism: the pre-registered
"sum then cap as one book" rule re-scales the combined row daily, so the dormant
sleeve is churned by the active one. The certified finding is the anti-netting
result; sum-then-cap is the rule that produced it, and the aim-portfolio is the fix.

## 2. The design: the Gârleanu–Pedersen aim-portfolio

Gârleanu–Pedersen (dynamic trading with predictable returns and
proportional/quadratic costs) does not sum target rows. It trades a fixed fraction
of the way from the *current* book toward an **aim portfolio** — a cost-weighted
blend of the per-signal Markowitz targets — where the trade rate is set by the
cost/risk ratio and the blend weights are set by each signal's persistence:

- Let each signal $s$ contribute its own Markowitz target $M_s$ (the position that
  signal alone would want, risk-scaled). The aim is
  $\text{aim}_t = \sum_s \omega_s M_{s,t}$, with $\omega_s$ increasing in the
  signal's decay half-life: a slow signal earns nearly its full Markowitz weight in
  the aim; a fast signal is down-weighted, because the position will decay before
  the cost of establishing it is recouped. **Heterogeneous decay enters here** —
  not as a post-hoc sum, but as signal-specific aim weights.
- The book then moves $x_t = (1-a)\,x_{t-1} + a\cdot\text{aim}_t$, with the single
  trade rate $a$ determined by round-trip cost against risk aversion — the
  multivariate generalization of the shipped scalar no-trade band.

Because the fast and slow targets are blended *before* the single trade decision,
the slow sleeve's contribution to $\text{aim}_t$ is stable across its dormant bars
and the fast sleeve's fluctuations net against it inside one optimization step,
rather than each being formed, summed, and re-capped independently. This is the
netting the sum-then-cap rule failed to produce.

This is the primary lever by the spec's own accounting. `SPEC §7.3` canonizes the
"cost-aware no-trade band sized from the OU half-life and round-trip cost by a
**closed-form Gârleanu–Pedersen rule**" and states flatly, "**This is the primary
lever (§2).**" `SPEC §13`'s closing paragraph frames the slow momentum sleeve as
the case the framework is *for*: "That is the heterogeneous-decay case
Gârleanu–Pedersen is actually derived for; a single fast signal is the framework's
degenerate case." The shipped seed occupies exactly that degenerate corner:
`step_no_trade_band` (`src/prism/portfolio/construct.py:100`, the online
single-step band) and `closed_form_band` (`:146`, the Martin-2012 cube-root
proportional-cost half-width — the single-name G-P analogue registered in
`docs/r2_design.md §1`). Both are correct, tested, and single-signal. The
aim-portfolio is their multivariate completion, not a replacement: it recovers the
shipped band exactly when the signal set has one member.

## 3. The JKMP frontier extension: weight signals against after-cost utility

The aim weights $\omega_s$ of §2 are still, in the base G-P construction, functions
of decay and cost — they are not fit to realized net performance. The principled
generalization is Jensen–Kelly–Malamud–Pedersen, *Machine Learning and the
Implementable Efficient Frontier* (RFS): choose the signal combination to maximize
the mean–variance utility of the *implementable* portfolio — the portfolio net of
the transaction-cost function — rather than to maximize each signal's predictive
accuracy (IC).

The objective, schematically, is to select combination weights maximizing

$$\mathbb{E}[r_p] - \tfrac{\gamma}{2}\,\mathrm{Var}[r_p] - \mathbb{E}\big[\mathrm{TC}(\Delta x)\big],$$

with $\mathrm{TC}(\cdot)$ the same cost stack the harness charges (bucket spreads,
participation gate), placed *inside* the objective. The frontier is "implementable"
precisely because the cost term is optimized against, not subtracted after. Signals
then earn weight by marginal contribution to after-cost utility: a high-IC,
high-turnover fast signal that a low-cost world would load heavily is correctly
discounted once its churn is priced, and a modest-IC slow signal that nets against
the fast one earns weight it would never earn on IC alone. This is the disciplined
answer to the question the +28% finding forced — *how* to combine heterogeneous-decay
signals — replacing the pre-registered sum-then-cap rule with an optimization whose
objective is the number the kill-criterion actually reads.

## 4. Pre-registered trial intent (counted, unspent)

No trials are budgeted here. What is registered is the *accounting* they will obey
when the gate opens:

- **Every enable is one counted construction trial** against the cumulative `SPEC
  §10` budget: "the cumulative construction-trial budget (pre-registered, ≤ **~30
  counted trials**)." That ledger stood at **17 of ~30** after the residual
  selection set closed (`docs/demotion_design.md §3`), with the momentum program
  drawing ≤ 8 more against the remainder (`docs/momentum_design.md §2`). The
  aim-portfolio's trials must fit inside whatever survives both; the exact count and
  axes are **not fixed by this draft** and will be pre-registered — in a fresh DSR
  selection set, mirroring the `momentum_v1` namespace — at ratification, when the
  gate opens.
- **Frozen-off parity is a landing requirement, not a trial.** The aim-portfolio
  must reproduce the shipped single-signal band bit-for-bit at signal-set size one,
  and reproduce frozen behavior with the sleeve disabled, before any counted enable
  — the `sleeve_mode="off"` / parity discipline the Arm-B mechanics already
  established (`docs/demotion_design.md §6`).
- **The +28% finding is the pre-registered baseline.** The first counted
  aim-portfolio trial is adjudicated against the sum-then-cap cost toll it replaces;
  the claim it must beat is the anti-netting delta, measured as a cost-toll
  difference under identical signals and cost stack. A blend that does not turn the
  toll negative relative to the standalone gross-weighted sum has not earned its
  trial.

The budget is drawn only after
ratification, and ratification is downstream of the momentum verdict.

## 5. Exit criteria and adjudication

Pinned now so they are not improvised at the gate:

- **Gate check (precedes everything).** No counted trial runs until
  `docs/momentum_design.md §3`'s promotion read returns — and if that read fires the
  fragility-kill instead of promotion (`docs/momentum_design.md §3`), there is no
  second signal to blend and this document is shelved, not run. The gate is
  momentum's verdict, full stop.
- **Netting mechanics (test-gated, before any counted run).** On a synthetic
  two-signal panel with one fast and one slow component, the aim-portfolio's
  realized turnover is no greater than the sum-then-cap turnover on the same
  targets, and strictly less when the components are negatively turnover-correlated.
  Single-signal reduction to `step_no_trade_band`/`closed_form_band` is pinned by
  parity test. The synthetic panel must include the *confirmed* anti-netting
  regime (`docs/b2_anti_netting_diagnostic.md`, 2026-07-11): cap-binding pressure
  — raw combined gross above `max_gross` while the slow component is static — and
  the test asserts **zero slow-component turnover on fast-only days**, since the
  measured B2 excess was entirely the proportional re-cap wiggling the dormant
  sleeve on days only the fast sleeve moved (and netting actually *worked* on
  refresh days). An exit test that only bounds aggregate turnover can pass while
  the confirmed failure mode survives.
- **Promotion (readable only post-gate, on the counted trials).** A blended book
  reaches `net_edge` under bucket spreads with deflated (selection-set DSR) net
  Sharpe > 0, its cost toll strictly below the sum-then-cap baseline of §4, and a
  live-monitor read not in contradiction — then deployment follows the GO branch of
  `docs/handoff.md §8` at minimum size.
- **Kill.** If the aim-portfolio cannot beat sum-then-cap's toll on netting-favorable
  signal pairs, or exhausts its post-gate budget without a `net_edge` claim, the
  multi-signal construction is recorded as uneconomic at this scale and the program
  does not buy more trials — the `SPEC §10` budget does not refill.

**Status: drafted, GATED on the momentum verdict — no counted trial until the
gate opens.**
