# Sizing pre-registration — GO-branch deployment (crash-conditional de-gross)

> **Status: PROPOSED (drafted 2026-07-19).** Ratification is the owner push of
> the dedicated commit flipping this banner, per the repo's ratification
> convention. This document is GO-branch precondition (a) (docs/handoff.md §8,
> the kill-criterion-fires-GO row): no real-money order sizes itself except
> through the rules pinned here. It pins deployment arithmetic only — it
> searches nothing, starts no counted trial, and moves no ratified statistic.
> Every number below is either a pin (chosen ex ante, marked **pin**) or a
> measurement citation (tracked artifact named inline).
>
> Filename note: the GO row as ratified pointed at
> `docs/aim_portfolio_preregistration.md`; that name was already held by the
> gated Gârleanu–Pedersen multi-signal construction design (2026-07-08),
> which lives on a different gate (the momentum verdict) and is shelved if
> that gate fails. Sizing must survive for a single-signal GO book, so it
> lives here; the pointer correction lands as a dedicated commit named in the
> same chain as this draft (owner-directed 2026-07-19).

## 1. Scope

Governs the sizing of the first real-money deployment book if and when the
kill-criterion fires GO and both handoff §8 preconditions are met. Nothing
here runs before that. Explicitly out of scope: the A3 micro-account (≤ $2k
carve-out) — that is a cost-measurement instrument under A3's own terms, not
the deployment book, and its sizing is A3's; and the multi-signal
aim-portfolio construction (its own document, its own gate).

## 2. Base size and share-sizing mode

Two admitted deployment modes, from docs/account_size_floor.md:

- **Whole-share OPG** (the certified replay convention): admitted at
  **≥ $100,000** only. Measured there: zero censoring, mean active share
  0.027 against the 0.05 concordance bar (§3 of that doc). The $50k–$100k
  band measured at or over the bar (0.062 mean at $50k) and is **not
  admitted** — a book that is marginally not the decided book is the wrong
  place to put the first real dollar. **Pin: no whole-share deployment below
  $100k.**
- **Fractional DAY** (floor-free at every size tested, §6 of that doc):
  admitted at any size **conditional on both venue checks passing and being
  recorded first**: (i) fractional *short* acceptance and per-name
  fractionability on the live venue — a venue taking fractional longs but
  whole-share shorts re-introduces half the floor on a long−short book;
  (ii) the day-vs-auction execution-quality delta read from the paper
  stream's fill telemetry. Either check failing closes this mode and leaves
  whole-share OPG ≥ $100k as the only admitted path.

Initial deployment is the **minimum viable size within the admitted mode** —
scaling beyond it moves only along the capacity curve (SPEC §7.6), which is
the existing rule, not a new one.

## 3. Crash-conditional de-gross term

The measured problem (results/beta_telemetry_2026-07-19.json): the book's
market beta is state-dependent in exactly the way its dollar-neutrality
hides. Unconditional: −0.081 (equal-weight proxy, n=1308) / +0.237 (SPY
overlap, n=259). Conditional on the market crash state defined below:
**+0.420** (SPY cell, n=24), with the replay stream cross-checking at +0.689
(n=112, too short to stand alone). A dollar-neutral book that is
0.42-correlated with the market precisely when the market is falling has a
convexity problem that average-beta telemetry cannot see and sizing must
price.

### 3.1 State definition

S_t = 1 (crash state) iff the trailing 21-session SPY close-to-close simple
return, measured through the **prior** session's close, is below

> **θ = −0.017981751321183567** (pin)

θ is pinned to the measured worst-decile threshold of the telemetry cell
(`conditional.worst_decile_market_21bar.spy.threshold_trailing_return` in the
tracked JSON), under the same strictly-prior conditioning the cell was
measured with. The state is computed from SPY daily closes fetched by the
live loop's own loader, same return convention as the instrument. Wiring the
SPY series into the loop is part of the arming change (§5), not a new
discretionary input: the state definition is closed under this section.

### 3.2 Action

- **Entry (fast):** on the first session with S_t = 1, scale the held book
  proportionally to gross multiplier **g = 0.5** (pin) at that session's
  decision — every position halved, rank-preserving, no re-scoring, no
  cadence change for anything else. This is a pre-registered risk action, not
  an alpha decision, which is why it may act off the B1 refresh cadence.
- **Exit (slow):** after **5 consecutive sessions** (pin) with S_t = 0, the
  book restores to full gross at the **next scheduled refresh** — re-grossing
  is a buy program whose turnover belongs inside scheduled turnover, not a
  reflex. Asymmetry is deliberate: fast out, slow back in.
- g never exceeds 1.0; the term only ever reduces gross below the
  construction's own `max_gross`.

### 3.3 Why g = 0.5 and not 0 or a curve

The same telemetry cell that measures the +0.420 crash-conditional beta
measures **+28 bps/day in-state alpha** (alpha_daily +0.0028, n=24): on the
measured sample the book *earns* through crash windows on average while
carrying the beta excursion. Halving cuts the crash-conditional beta
contribution to ≈ +0.21 — the neighborhood of the unconditional SPY cell —
while keeping half the in-state alpha and halving re-entry turnover.
Recorded and rejected:

- **Full flatten (g = 0):** forfeits the measured in-state alpha entirely,
  doubles the round-trip turnover of every episode, and converts a hedged
  book into a timing bet on the state variable.
- **Continuous rolling-beta scaling:** the 63-session rolling estimator
  (range −0.17…+0.50 on SPY) lags the state the telemetry says matters and
  makes the multiplier itself a noisy estimated quantity; a binary rule on a
  pinned threshold is auditable ex post.
- **Book-drawdown trigger:** measured weaker — the book's own >5% drawdown
  state carries SPY beta +0.154 vs the market state's +0.420. The market
  state, not the book's drawdown, is where the conditional loading lives.
- **Vol-target overlay:** unmeasured here and interacting with the regime
  layer; a named follow-up that would amend this document through
  re-ratification, not a silent extension.

Honesty note: n=24 is a small cell. The term's justification is convexity
protection under a measured sign, not point-estimate optimization; the
re-open triggers in §6 are the guard against the cell having been noise.

## 4. After-tax read

Deployment sizing consumes the **crash-year conditional** tax wedge, not the
steady-state average (docs/tax_wedge_spec.md §3) — the loss-netting cap is a
convexity penalty in after-tax space that co-moves with exactly the state
§3.1 conditions on. The tracked tax-wedge JSON's parameter vector must be
the operator's actual rates before this document's arithmetic is executed;
the committed reference vector is for reproducibility only.

## 5. Execution path and seniority

The term executes through the SPEC §7.7 regime step's gross-scale hook,
armed only by the dedicated commit that also flips this banner (until then
the hook stays `None` and the paper stream is bit-identical to the certified
convention). Seniority: the safety halt (kill switch, drawdown bound) beats
the de-gross — a halted cycle trades nothing regardless of state; the
participation gate applies to de-gross orders like any others. Precondition
(b) is unchanged by this document: ≥ 21 consecutive clean regime-step
sessions before any real-money order, a clean session being one whose
regime telemetry carries zero named block failures.

## 6. Re-open triggers (pre-registered)

This document re-opens — deployment blocked until re-ratified — if, before
the first real-money order:

1. the accruing paper stream's crash-conditional beta contradicts the
   **sign** of the telemetry cell (the pin protects against a +0.42 that was
   never real);
2. spin-adjustment remediation of the certified spine (the
   docs/bar_vendor_divergence.md §6 finding) materially moves certified B1 —
   the beta telemetry inherits the spine, so a rank-integrity fix upstream
   invalidates the measurements this document consumes;
3. either §2 venue check fails in a way that leaves no admitted mode at the
   owner's intended capital.

## 7. Parameter table (all pins)

| Parameter | Value | Source of the choice |
|---|---|---|
| Whole-share floor | $100,000 | account_size_floor §3 (0.027 vs 0.05 bar) |
| Fractional path | any size, both venue checks first | account_size_floor §6 |
| Crash threshold θ | −0.017981751321183567 | telemetry cell's measured worst-decile bound |
| Trailing window | 21 sessions, strictly prior close | telemetry cell definition |
| De-gross multiplier g | 0.5 | §3.3 |
| Exit hysteresis | 5 consecutive S=0 sessions, restore at next refresh | §3.2 |
| Tax input | crash-year conditional wedge, operator rates | tax_wedge_spec §3–4 |
