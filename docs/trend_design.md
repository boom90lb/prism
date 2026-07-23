# Trend-sleeve pre-registration — P3, crash-convex ETF time-series momentum (RATIFIED)

**Status: RATIFIED 2026-07-18** (this commit is the dedicated ratification
commit; the decision was delegated by the owner to the session in the same
turn — the `docs/momentum_design.md` delegation precedent; drafted
2026-07-17 under the owner's scope-expansion directive; in-repo convention:
`docs/handoff.md` §7). The §1 universe is resolved and recorded below per
the ADV rule, exactly as this document requires at ratification. **The
design is frozen from this commit: the out-of-sample clock runs.** From this
commit the amendment rule applies: seams may be pinned, no trial value may
move. **The binding act is family ratification, not the first run:
ratification freezes this design, and every bar after it accrues as
out-of-sample by construction.**

> **A4 concurrency opt-in: RATIFIED 2026-07-22** (this commit is the
> dedicated opt-in commit; owner-authorized on `main`; mechanism
> `docs/amendments_2026-07.md` A4; queue item under
> `docs/v040_program.md` §5). **No trial value in §1–§4 moves under this
> seam.** Binding posture:
>
> 1. **Kill/fragility reads T0–T4** may run concurrently with the
>    adjudication-slot holder under A4's three criteria: (a) this
>    pre-registration is ratified and frozen; (b) T0–T4 cannot promote
>    ("they can kill, they cannot promote", §3); (c) outputs are
>    **firewalled** — barred from every cross-family adjudication
>    (momentum M6, aim-portfolio G4b, any other promotion) until this
>    family holds the serial promotion slot for T5.
> 2. **Promotion read T5 always requires the serial adjudication slot** —
>    A4 never grants concurrent promotion.
> 3. Every A4-concurrent read still appends to this family's ledger and
>    counts against the frozen budget of 6 exactly as if it ran in the slot.
> 4. Concurrency remains *permitted, not mandated* (operator bandwidth).
>
> Uncounted mechanics and paper were never gated by the slot and remain
> free to build. This document spends nothing until a counted run under §3.

## 0. Provenance and the honest prior

The sleeve is selected for **crash convexity, not standalone Sharpe** — the
program review's finding #2: the momentum sample structurally cannot contain
a momentum crash, the extension year is equally unlikely to contain one, and
"the real mitigation is portfolio-level: the second sleeve should be chosen
for crash-convexity, which argues for time-series ETF trend over any
standalone-Sharpe comparison" (`docs/program_review_2026-07.md` §1.2). The
sibling sleeve's own pre-registration defers crash exposure to sizing
(`docs/momentum_design.md` §4); this sleeve is the instrument that sizing
doctrine wants to exist.

The prior, stated so it cannot inflate: time-series momentum is documented
across a century and dozens of markets (Moskowitz–Ooi–Pedersen 2012;
Hurst–Ooi–Pedersen's century evidence), and its positive tail behavior in
equity drawdowns — the "CTA smile" — is exactly the property purchased here.
Both facts decay on publication: the 2010s were a famously poor decade for
published TSMOM implementations, and a retail ETF implementation truncates
the asset breadth (no futures, no term-structure carry, wrapper fees and
roll drag in commodity/FX ETFs) that powers the published Sharpes. The
realistic central case is a low-Sharpe sleeve whose admission case is the
tail complement and future netting value, both priced elsewhere: netting by
the aim-portfolio when its own gate opens
(`docs/aim_portfolio_preregistration.md`), tail value by the §4 convexity
read.

**Prior in-repo exposure, disclosed:** a single-series TSMOM baseline exists
in the retired research tree (`research/baselines/tsmom.py:20`, sign of the
trailing K-bar return, default lookback 60) and ran inside the retired
forecasting harness's folds as an ensemble comparator
(`research/scripts/backtest.py:500`). It was never a sleeve, never formed a
cross-asset book, and appears in no counted ledger. This design is specified
from the literature, not from any in-repo result — but the discovery step is
*literature-blind, not history-blind*: publication multiplicity is the
residual contamination and it is why §1's configuration is the canonical
published convention rather than any tuned variant. From this drafting
forward, **no trend backtest of any configuration runs before ratification**,
and none runs after it outside the §3 counted set — the budget-amnesia
failure mode (`docs/handoff.md` §7.2) is the one this paragraph exists to
block.

## 1. Universe — pinned by rule, not by list

US-listed ETFs on the admitted $0 daily-bar vendor, tradable at the live
venue. Ten asset-class buckets, spanning the standard TSMOM span in ETF
form: US broad equity, developed ex-US equity, EM equity, long US duration,
intermediate US duration, IG credit, HY credit, gold, broad commodity, US
dollar. Within each bucket the universe member is **the single US-listed ETF
with the highest median dollar volume over the 252 bars ending at the last
bar before ratification** — a deterministic liquidity rule, resolved and
recorded in this document at ratification (the free pre-ratification
amendment window, `docs/momentum_design.md` banner precedent), so no ticker
is ever hand-picked.

**Resolved at ratification (2026-07-18).** Measurement: median of
close × volume over the 252 bars ending 2026-07-17, consolidated daily bars
from the spine vendor; the two credit buckets were measured on the Alpaca
feed instead (the spine vendor returns no bond-ETF bars on the current
tier) — the ranking is intra-bucket, so the tape choice touches no other
bucket, and both credit margins (3.3× and 5.7×) dwarf feed-share noise.
Contenders are recorded so the resolution is auditable; every winner leads
its runner-up by ≥ 1.5×.

| bucket | winner | median $DV (M) | contenders measured ($M) |
|---|---|---|---|
| US broad equity | **SPY** | 46,062 | IVV 4,463 · VOO 4,437 · VTI 1,290 |
| Developed ex-US equity | **EFA** | 1,517 | IEFA 999 · VEA 837 |
| EM equity | **EEM** | 1,614 | IEMG 859 · VWO 492 |
| Long US duration | **TLT** | 2,695 | SPTL 151 · VGLT 106 |
| Intermediate US duration | **IEF** | 734 | VGIT 157 · SPTI 53 |
| IG credit | **LQD** | 252\* | VCIT 75\* · IGIB 10\* |
| HY credit | **HYG** | 224\* | JNK 39\* · SHYG 3\* |
| Gold | **GLD** | 4,081 | IAU 653 · GLDM 430 |
| Broad commodity | **PDBC** | 77 | GSG 16 · DBC 14 |
| US dollar | **UUP** | 38 | USDU 6 |

\* Alpaca-feed volume (intra-bucket basis note above).

**The universe: SPY, EFA, EEM, TLT, IEF, LQD, HYG, GLD, PDBC, UUP.**

Honesty notes, recorded now: (i) the rule evaluated at ratification carries
mild survivorship with respect to the *backtest* era — a fund that won its
bucket today may not have existed in 2010; a name enters its cell only from
its listing date + 252 bars, an empty cell holds cash, and no proxy splicing
is permitted (N7 posture: absence is absence, not a substitute series);
(ii) ETF wrappers embed expense ratios and roll costs that futures-based
literature results do not carry — this is a feature of the estimand (the
retail-implementable premium), not a defect of the data.

## 2. Signal and construction, pinned

- **Signal:** per-ETF time-series momentum — the sign of the trailing
  252-bar total return excluding the most recent 21 bars (12−1). The
  canonical MOP convention is 12-0; the skip is adopted deliberately for
  cross-family consistency with the sibling sleeve's reversal-avoidance
  discipline (`docs/momentum_design.md` §0) and is itself probed at T2. This
  is the kind of contestable choice that must be pinned before any data is
  seen, and here it is.
- **Sizing:** inverse-volatility to equal per-name risk contribution
  (63-bar EWMA of daily returns, annualized), scaled to the book-level gross
  cap; signs from the signal. No cross-sectional ranking — every name holds
  its own sign.
- **Cadence:** decisions on the same `decision_every=21` grid as the
  momentum sleeve. Cadence alignment is what makes future internal netting
  real rather than notional — the two books trade the same bars
  (`docs/handoff.md` §3).
- **Cost stack:** frozen, identical to the sibling program — closed-form
  band, `SPREAD_BUCKET_SCHEDULE_V1`, 5% participation cap.
- **Mechanics:** `trend_node.py` under the base signal contract
  (`src/prism/signal/momentum_node.py` precedent), landing **default-off**
  with parity and property tests green before any counted run — the
  `mechanics_clean` entry discipline (`docs/demotion_design.md` §6, Arm-B).
  Mechanics may land any time; landing mechanics is not a trial.

## 3. Counted trial set (ratify or amend, then frozen)

Namespace `trend_v1`; budget **exactly 6 counted trials**, never refilled;
degenerate and NaN outcomes count. Prior counted programs, recorded at
ratification 2026-07-18 per amended SPEC §10: **three** — residual
reversion (closed at 17, cert 001); momentum (≤ 8, in flight, holding the
counted slot); `momentum_replication_v1` (≤ 3, ratified 2026-07-18 with a
granted concurrency exception).

| id | delta vs T0 | probes |
|----|-------------|--------|
| T0 | — (the pinned §1/§2 configuration, first run) | the primary read |
| T1 | lookback 126 | horizon sensitivity |
| T2 | no skip (252-0, the MOP convention) | is the transplanted skip load-bearing? |
| T3 | `decision_every=63` | cadence sensitivity |
| T4 | equal-notional sizing (inverse-vol off) | is the result a vol-timing artifact? |
| T5 | T0 re-run, sample extended ≥ 1 year past ratification, post-ratification OOS segment reported separately | the promotion read |

T1–T4 are **fragility detection only** — they can kill, they cannot promote,
and the pinned configuration remains T0's regardless of which probe scores
best (adopting a different cell would be a new discovery event; the §1/§2
logic of `docs/momentum_design.md` transplants whole). T0 is the primary
read of the pinned cell; it is not a promotion read (promotion is T5 only)
and therefore sits with T1–T4 under the A4 "cannot promote" criterion
(banner opt-in RATIFIED 2026-07-22). T5 still requires the serial
adjudication slot. **OOS accrual is independent of concurrency:** a
2026-07-18 family ratification means every subsequent bar is out-of-sample
for the frozen design whether the first counted run is serial or
A4-concurrent; T5's ≥ 1-year OOS segment is defined against family
ratification, not against the first kill-class print.

## 4. Adjudication (pre-committed)

- **Fragility kill (readable after T1–T4):** the sleeve is dropped without
  appeal if the median net annualized Sharpe of T1–T4 is negative, or if any
  single probe flips the sign of the net result at magnitude greater than
  T0's own point estimate.
- **Convexity read (the admission case; readable on T0 and re-read on T5,
  full sample and OOS segment separately):** the sleeve's mean 21-bar return
  conditional on (a) the worst-decile 21-bar returns of the US broad-equity
  bucket member and (b) the worst-decile 21-bar returns of the sibling
  momentum sleeve's backtest book must be **no worse than its unconditional
  mean** — the left wing of the smile. A sleeve that fails this read fails
  its admission case *even at positive Sharpe*: the recorded verdict is
  "positive-carry, non-convex; portfolio admission refused," and any
  repurposing as a standalone-Sharpe sleeve is a new pre-registration, not a
  reinterpretation.
- **Promotion (readable only at T5):** `net_edge` under the frozen stack
  with DSR > 0.5 against the `trend_v1` selection set, periodic net Sharpe
  above the then-current cash hurdle, the convexity read passing on the OOS
  segment, and a live-monitor read not in contradiction if a paper
  instrument exists by then (§5). Deployment and blending then follow their
  own law: the GO branch of `docs/handoff.md` §8 at minimum size, and the
  aim-portfolio's own gated pre-registration for any combination with the
  momentum sleeve.
- **Neither fires:** the program waits for more out-of-sample data; the
  budget does not refill.

## 5. What is deliberately left to owner decisions

- **A4 concurrency opt-in:** RATIFIED 2026-07-22 (banner box above). No
  further owner act is required for kill-class T0–T4 concurrency under A4;
  operator bandwidth may still choose to run nothing concurrent.
- **Paper instrument:** whether the trend book joins the paper loop before
  its counted trials (a second I-9 instrument at trivial size, order-ids
  under their own book prefix — the namespacing seam already supports it,
  `docs/momentum_design.md` §5.4) is an owner decision requiring its own
  seam amendment; this document neither requests nor forecloses it.
  (Mechanics and a default-off paper path may exist on the execution branch
  without that seam; the seam gates *joining the certified I-9 instrument*,
  not building.)

## 6. What this document does not do

It does not touch the momentum program, its budget, or M6. It does not blend
anything — combination is the aim-portfolio's jurisdiction and that document
stays gated on the momentum verdict as written. It does not buy data (the §1
universe lives on the free tier). It does not run anything now. Its budget
is fixed at 6 and does not refill.
