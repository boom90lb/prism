# Prism v0.4.0 — the deploy-first program

> **Status: DRAFT — awaiting owner ratification (= owner push of the dedicated
> commit).** This document re-founds the *objective ranking* of the project for
> v0.4.0 and the *product surface* required to run it. It does **not** re-open
> any frozen mid-flight design: the momentum M-series (`docs/momentum_design.md`),
> trend T-series (`docs/trend_design.md`), learned-XS X-series
> (`docs/learned_xsection_design.md`), replication C-series
> (`docs/replication_preregistration.md`), and the ratified sizing
> pre-registration (`docs/sizing_preregistration.md`) stand exactly as ratified.
> What changes is what the program *optimizes*, what runs in parallel, which
> markets execute, how risk and hedging compose for an operator, and what the
> standing doctrine tells agents to protect.
>
> **Product charter (not a return SLA).** Prism is a near-frontier-conditional
> systematic trading bot for operators under retail constraints: free/cheap
> data, solo or small operator, modest AUM, daily-to-weekly horizon, large
> local compute. **Product priority is US equity and liquid-ETF markets first**
> (cross-sectional B1 / X, trend ETFs, multi-sleeve construction). Crypto is a
> second, independent market — useful as a *validation lane* (venue-true fees,
> 24/7 bars, no equity-history gap), never the ranking prize or the default
> owner-attention sink. The consistent deliverable is an *honest, deployable
> process* with intuitive risk profiles and composed hedging — not a promise of
> alpha. Alpha is the output of the filter when a sleeve clears its bar; kill
> branches remain first-class certified results.

## 0. The operating sentence

**The harness is a hard filter that never moves. The objective is multi-sleeve
net after-cost IR inside an operator-declared risk envelope, deployed at the
operator's real capital mode. Incomplete history is accepted via liquid ETFs,
crypto, and forward validation. Compute buys purged learning and construction
search, not model species.**

Two rankings, never conflated:

| Ranking | Prize | Role |
|---|---|---|
| **Objective** | Multi-sleeve net after-cost IR at real capital mode | What we optimize |
| **Integrity** | Claim credibility, N1–N8, trial accounting, claim tiers | Hard filter — never traded for features |

Handoff's historic "protect the harness above every feature" was right as
*integrity* and wrong as *objective*. Agents reading "deploy-first" as
permission to soft-bar are wrong: deploy-first re-ranks the prize, not the
bar.

**Frontier vocabulary is conditional.** Target class: best-in-class systematic
trading under {free/cheap data, solo operator, modest AUM, daily-to-weekly
horizon, large local compute}. Absolute-frontier claims (HFT, SIP-depth,
multi-venue latency) are barred from docs — that is a different firm
(`MARKETS.md` §1; SPEC §3 law 6 already deleted the latency pillar). For
product language: "frontier" means honest, risk-configurable, deployable
process — not edge secrecy or SOTA returns.

**Barred product vocabulary:** "guaranteed," "consistent profits," "set and
forget alpha," "hedge" without naming the instrument (de-gross vs convexity vs
factor), "calibrated costs" without saying micro vs deploy AUM.

## 1. What v0.4.0 reverses, and what it keeps

**Reversed:**

1. **The asset-ledger inversion.** `docs/handoff.md` §1 ranks the alpha last —
   "Not an asset yet. Possibly never." That line is both strategically inverted
   (it instructs every future agent to protect process over IR) and factually
   stale (it describes the archived residual sleeve at net ≈ −0.01; the live
   book is B1 momentum, certified price-return Sharpe ≈ 0.43–0.52 with the
   −17 bps/yr live-vs-certified pin). v0.4.0 rewrites the ledger: the objective
   is deployable multi-sleeve economics inside a risk envelope; the harness is
   the *enabling filter* — still rare, still non-decaying, no longer the prize.
   Integrity remains absolute; process is not the thing we maximize.
2. **Serialization of the second sleeve.** Counted adjudication stays
   accountable, but trend and learned-XS take A4 seam amendments (the PI-2
   mechanism: a family whose ratified text declines concurrency opts in by
   seam amendment). Uncounted mechanics and paper never needed permission —
   the constitution gates counted adjudication, never building.
3. **Single-market serialization (softened, not inverted).** Crypto spot is
   already CORE-candidate in SPEC §4 with its own residualize-bypassed lane
   (§7.1) and its own `net_edge` bar. v0.4.0 *registers* it as an independent
   family so it may run without waiting on the equity promotion verdict — it
   is the one execution market where incomplete US equity history is
   irrelevant, and venue-true fees make cost calibration easier than A3 equity
   micro-fills. **Product ranking and owner attention still prioritize stock
   markets first** (B1, trend ETFs, X, G0 equity costs, multi-sleeve). Crypto
   is second-market / optional validation, not the deploy-first prize. This
   rewrites the handoff §8 row from hard "crypto before equity verdict → No"
   to **"equity owns product priority; crypto may proceed as independent
   validation when bandwidth allows"** (ratification queue item 1).
4. **Single-path GO.** The preferred GO shape becomes multi-sleeve
   (§3 below). Single-sleeve B1 GO remains legal under its own docs but must
   be labeled **`single-premium, de-gross-only`** (convexity-unhedged) — not
   "crash-unhedged." Sizing already pins crash-conditional de-gross
   (`docs/sizing_preregistration.md` §3); that is risk reduction, not
   CTA-style convexity. See §8.
5. **Compute posture.** The big box is aimed at purged walk-forward learning
   (X-family), construction counterfactuals (bands, cadence, netting), and
   stress libraries — not at model species. N8 keeps all of it off the
   production import path.
6. **Aim-portfolio gate (split).** The drafted aim-portfolio gate
   (`docs/aim_portfolio_preregistration.md` §0) is the momentum *promotion*
   verdict. v0.4.0 does **not** silently replace that with "two sleeves at
   `mechanics_clean`." Instead: **G4a** uncounted joint construction
   diagnostics open at two `mechanics_clean` sleeves; **G4b** counted
   aim-portfolio trials open only when ≥1 sleeve is at its ratified bar (or
   under a firewalled A4 non-promotion kill still open). Rationale: blending
   unpromoted signals into *counted* construction spends deflation budget on
   noise; joint *engineering* diagnostics do not.

**Kept, untouched:**

- N1–N8, claim tiers, per-selection-set deflation, kill criteria. No bar
  moves down anywhere in this document.
- All frozen designs and their clocks: B1 M-series (M6 read at its ratified
  window), trend OOS accrual (running since ratification 2026-07-18),
  X-family budget-6, C-family budget-3.
- Certification 001: daily residual reversion stays archived. No revival, no
  new knobs on its closed selection set.
- The research quarantine (RL trio, Prophet zoo) — no compute allocated.
- The data doctrine as settled: $0-for-users is product identity; the
  observatory is the sole survivorship-free channel; buy-vs-build is closed.
- GO preconditions as ratified: (a) sizing pre-registration RATIFIED
  2026-07-20; (b) §7.7 regime step wired + 21 clean paper sessions (clock
  running). Gross-scale-hook arming remains its own deployment commit after
  both preconditions — decoupled from any banner flip.

## 2. Workstreams

Sequenced by **gates**, not calendar. Every workstream is *legally* buildable
the day this document ratifies; none touches an armed clock. **Attention is
not parallel** — see §7. Owner focus order binds solo-operator bandwidth even
when A4 permits concurrent kill-class reads.

### W0 — Ground truth: capital, costs, operations

The sharpest constraint is capital mode, and every `net_edge` stays
semi-fiction until real fills calibrate cost tables (I-9 contemplates exactly
this replacement of the conservative proxy).

- **A3 micro-account funding** (ratified carve-out) and first real fills →
  per-bucket realized spread + participation truth at *micro size*. Standing
  pre-checks fold in here: fractional-short acceptance and fractionability
  flags on the venue.
- **Capital-mode adjudication** per the ratified sizing pre-registration:
  $100k whole-share floor, or fractional-DAY gated on the two venue checks.
  One of these must be selected before any GO commit.
- **Operational hardening.** The disk event of 2026-07-18 is a measured
  threat; off-box artifact sync is live. Remaining: boot-resilient scheduling,
  a filesystem health check in the nightly wrapper, and a thin deploy runbook
  (capital mode × risk profile × venue × custody sweep × live kill-switch)
  so "anyone to run" is an operations claim, not a vibe.

**Gate G0 (micro):** first A3 fill ledger spanning ≥1 full rebalance cycle →
spread tables re-calibrated from realized fills at micro size + venue checks
recorded.

**Gate G0b (deploy-mode):** first fills at the *admitted capital mode*
(≥$100k whole-share, or fractional-DAY at operator size) before GO claims
"cost tables calibrated for deployment." A3 alone cannot calibrate capacity /
sqrt-ADV impact, whole-share concordance of the full B1 book, or multi-sleeve
crash economics (`docs/account_size_floor.md`). Micro fills break the
I-9 circularity; they do not finish I-9 for the deploy book.

### W1 — Trend sleeve online, uncounted first

Build trend_v1 mechanics end-to-end on the frozen design (signal → construct →
paper, on the 10-ETF universe pinned at ratification), running as a parallel
paper sleeve ledger beside B1.

- Uncounted diagnostics first: fill quality, turnover, and **sleeve-alone**
  crash-window replays (2020-03, 2022) — ETF survivorship is ~free, so these
  reads carry none of the equity-panel caveats.
- **Joint crash diagnostic (uncounted, required before multi-sleeve GO
  narrative):** B1 alone vs B1+trend over the same stress windows — joint max
  DD, crash-window return, turnover interaction, and whether sum-then-cap
  anti-netting reappears (B2 / cert-001 lesson). Fixed capital-allocation
  sensitivity (not optimized weights) is the product receipt for risk
  profiles.
- Counted T1–T4 runs require the **trend A4 seam amendment** (ratification
  queue item 2).
- Trend's role is not standalone Sharpe: it is crash convexity B1 never
  samples, a second decay rate for netting, and a capital-light book
  (`docs/trend_design.md` §0; program review finding #2).

**Gate G1:** trend `mechanics_clean` + A4 opt-in ratified → counted T-runs.

### W2 — Learned cross-section: the compute-side equity alpha bet

The X-family is the only in-repo program asking a frontier-conditional
question under the certified cost regime: does a learned combination of slow
characteristics beat fixed 12−1 at monthly cadence, net of bucket costs,
after family DSR? Prior stays skeptical (McLean–Pontiff; the published ML
zoo is gross/search-inflated) — but it is the correct *compute* bet relative
to any alternative use of the box.

- Counted X-runs proceed under the frozen budget-6 once the **learned-XS A4
  seam amendment** ratifies (queue item 3), rather than waiting on the
  momentum slot.
- Admission stays exactly as ratified: spanning read, incremental-to-B1
  t ≥ 2, never standalone Sharpe.
- Branches (all success states for the *program*):
  - **X wins** → registered supersession of the B1 score in the live spine.
  - **X loses** → family closes; compute re-aims at W4 construction search;
    **B1 remains the equity product path** if/when it promotes.
  - **Neither promotes** → cert packets; operator surface (W6) and harness
    still ship as signal-certification product (handoff Phase C shape).

Product must not hang "the alpha" solely on X. B1 is the live spine today.

**Gate G2:** A4 opt-in ratified → counted X-runs under the frozen budget.

### W3 — Crypto long/flat: second market, optional fast validation

**Priority: after equity product path.** Stock markets (W0 equity costs, W1
trend, W2 X, W4 multi-sleeve, W6 profiles for the equity book) own ranking and
default owner attention. Crypto does **not** lead v0.4.0 GO or the product
story.

**Why it is still useful early:** validation friction is lower than equities —
no incomplete US equity history, no Norgate-shaped gap, 24/7 bars, one named
venue priced at true fee (~2 bp taker at small size on Binance.US-class),
residualize bypassed (rank-1 TS lane). Real fills and fee-honest PnL can
prove the live spine / cost discipline on an independent book without waiting
on equity clocks. That is a *lab advantage*, not a product re-rank: treat
crypto as a secondary fill-and-custody gym if owner bandwidth allows, never as
a substitute for G0 equity cost truth or B1/trend evidence.

Requires a new family pre-registration (ratification queue item 4 — **after**
equity A4 seams). Frame per SPEC §4/§7.1: one named US venue priced at its
true fee, long/flat time-series lane, own `net_edge` bar, hard budget —
recommended: TSMOM/vol-target class, frozen stack, budget ≤ 6, T0-style zero
free parameters at ratification. Venue pin is frozen *at* that
pre-registration, not improvised later.

- Custody doctrine per MARKETS §2: sweep profits, minimize float —
  counterparty solvency is the survival risk.
- One *promotion adjudication* at a time still stands (A4); independent family
  ≠ concurrent promotion; equity promotion stays the serial prize.

**Gate G3:** crypto pre-reg ratified → mechanics → paper/live-shadow at
venue-true fees. **Non-blocking** of equity G0–G2 / preferred GO.

### W4 — Multi-sleeve construction: where the marginal IR lives

`docs/handoff.md` §3's own economic doctrine: turnover is a property of the
signal *set*; slow heterogeneous-decay signals net against each other before
touching the market; G-P construction has the most to give with multiple
signals. One sleeve is the degenerate case.

Construction search optimizes IR **inside** the operator risk envelope
(§8), not against it — otherwise pure IR systematically undervalues trend
(low standalone Sharpe, high convexity prior).

- **G4a:** when ≥ 2 sleeves reach `mechanics_clean`, uncounted joint
  construction diagnostics and engineering (including the joint crash
  diagnostic from W1) are unblocked. No deflation budget spent.
- **G4b:** the aim-portfolio pre-registration
  (`docs/aim_portfolio_preregistration.md`) un-gates for review →
  ratification → *counted* construction trials (netting, cadence, band
  counterfactuals — all counted, all deflated) only when ≥1 sleeve is at its
  ratified bar. Seam amendment on the aim-portfolio banner records the split
  gate (queue item 6).
- Highest-EV allocation of the large box once G4b opens: construction search
  at scale under real cost tables, not new model species.

### W5 — Parallel, non-gating: expectations capture + the factory doctrine

Two forward-compounding programs that never touch a counted clock:

- **Observatory expectations lane.** Extend the append-only capture
  discipline (the corp-actions lane pattern: jsonl.gz, capture timestamps,
  verbatim-stored payloads) to point-in-time *expectation state*: news flow
  (Polygon news — already keyed), EDGAR 8-K/filing cadence, coverage-intensity
  counts. Capture is time-irreversible — every uncaptured day is gone —
  so it starts at build-readiness; **modeling is deferred** until a family
  registers under the factory doctrine. No purchase, no license cage: the
  accumulated PIT expectation record is proprietary by construction, the same
  doctrine already ratified for delisting accumulation. Capture does **not**
  wait on factory ratification.
- **The factory amendment (SPEC §10 semantics).** Pre-registration of a
  *pipeline* — frozen feature space, frozen search procedure, selection-set
  accounting, promotion rule — inside which trials are industrial and every
  one is counted, deflated by effective independent trial count (the N5
  machinery as built). The X-family is already this shape at budget 6; the
  amendment generalizes the shape so future families (including the
  event-time/surprise family over the expectations lane) can run
  industrial search without either lying or begging for slots. Draft text
  may parallel A4 seams; ratification may lag capture.

### W6 — Operator surface: risk profiles + hedge composition

The product gap the research gates alone do not close. Operators never
free-knob a counted statistic. They choose a **named risk profile** that maps
onto already-ratified pins (or explicitly *stricter* subsets). Profiles are
product; pins are constitution.

| Profile (provisional) | Gross / participation | Crash term | Sleeve mix | Who |
|---|---|---|---|---|
| `research_paper` | as certified | telemetry only | B1 only | free tier; learn the loop |
| `conservative` | tightened max_gross allowed | g=0.5 (pin) when GO-armed | B1 + trend weight floor if available | first real money default |
| `balanced` | certified gross | g=0.5 | multi-sleeve preferred weights | default deploy |
| `assertive` | certified gross | only as later-ratified pins allow — **never freer than ratified** | multi-sleeve | not a free boost |

Rules:

1. Profiles may only **tighten** relative to ratified pins, never loosen
   (no g > 0.5 without a new sizing ratification).
2. Hedge composition is a declared **policy**
   `{equity_sleeve, trend_sleeve, crypto_book, de_gross}` with capital
   allocation bands — not ad-hoc daily discretion (§8).
3. Paper path with `research_paper` is bit-identical to today's certified B1
   loop (no silent fork of the promotion instrument).

**Gate G6 (soft):** documented risk-profile schema + paper loop honors a
profile without bit-breaking certified B1 under `research_paper`. No counted
trial spent. Schema may draft with queue item 1; freeze can lag.

## 3. GO doctrine for v0.4.0

GO preconditions (a) and (b) stand exactly as ratified. On top of them, the
**preferred GO shape is equity multi-sleeve** (stock markets first):

- ≥ 1 equity sleeve at its ratified bar, **plus** the trend (ETF convexity)
  sleeve at its own bar → label **`multi-premium, convexity-complemented`**;
- equity cost tables: **G0 done**; **G0b preferred** before material deploy AUM;
- capital mode selected and valid ($100k whole-share, or fractional-DAY with
  both venue checks passed);
- sizing armed by its own deployment commit;
- operator risk profile chosen (default **`conservative`** on first real
  money);
- joint B1+trend crash diagnostic on record.

Crypto at its own bar is an **optional second-market GO**, not a substitute for
the equity+trend preferred shape and not required for preferred GO. A crypto-
only GO is out of product ranking for v0.4.0 (may paper/shadow for validation).

Single-sleeve B1 GO remains legal per its ratified docs, but the GO commit
must label it **`single-premium, de-gross-only`** in those words
(convexity-unhedged). M6 stays frozen and is *not* the deployment bottleneck:
A3 runs under its ratified carve-out now; trend carries the equity-side
convexity evidence bar.

## 4. Cut list

Standing prohibitions (all pre-existing, restated so the ranking is
unambiguous): no residual-daily revival (cert-001 stands); no new knobs on
closed selection sets; no compute to the quarantined RL/Prophet stack; no
continuous hyperparameter roaming (counted or not run); no absolute-frontier
or SOTA vocabulary in docs; no event/news *modeling* before its factory
pre-registration (capture is exempt and encouraged); no free risk knobs that
loosen ratified pins; no "consistent alpha" / return-SLA language in product
docs.

## 5. Ratification queue

Each item is its own dedicated commit, per the standing convention (anchor
tag at ratification; never deleted):

1. **This document + doc coherence set** — handoff §1 asset ledger re-ranked;
   handoff §2 restated around the operating sentence (**equity-first product
   priority**); handoff §6 Phase D order: equity multi-sleeve before crypto;
   handoff §8 crypto row rewritten to **"equity owns product priority; crypto
   may run as independent validation when bandwidth allows"** (not a hard ban,
   not a re-rank to crypto-first); README status banner pointed at this program
   once ratified (honest "no cleared bar" kept); aim-portfolio banner note that
   G4 is split pending its seam amendment.
2. **Trend A4 seam amendment** (small — banner amendment via the PI-2
   mechanism). Equity path.
3. **Learned-XS A4 seam amendment** (small — same mechanism; its banner
   already contemplates the opt-in). Equity path.
4. **Factory amendment to SPEC §10** (substantive — pipeline-level
   pre-registration semantics under N5; draft may parallel items 2–3). Serves
   equity search industrialization first.
5. **Aim-portfolio pre-registration review + split-gate seam** — triggered
   at G4a for engineering; counted G4b only when ≥1 sleeve at bar. Equity
   multi-sleeve.
6. **Risk-profile schema (W6)** — may draft with item 1; freeze when the
   paper path honors `research_paper` bit-identity. Equity operator surface.
7. **Crypto family pre-registration** (substantive — venue, fee, family,
   budget frozen). **Queued after equity A4 seams** unless owner explicitly
   pulls validation-lane work forward; never steals serial promotion slot from
   equity.

## 6. What v0.4.0 does not claim

No absolute frontier. No return SLA and no "consistent alpha" promise —
alpha is deployable *when certified*, absent when killed. Every family keeps
its kill branch; a fired kill remains a first-class certified result. No bar
is lowered anywhere in this program. Micro-calibrated spreads (G0) are not
deploy-AUM costs (G0b). Profiles never loosen pins. What changes is that the
program stops optimizing the least interesting coordinate of a multi-signal
system and starts optimizing the objective it was chartered for — inside a
risk envelope an operator can actually configure.

## 7. Attention ladder (solo operator)

Legality is parallel; attention is not. Standing owner protocol for this
release:

1. **Owner now (equity):** ratify this program + doc coherence (queue 1);
   fund A3 equity micro-account; equity venue checks (fractional short /
   fractionability); decide capital-mode *path* (selection before GO).
2. **Agents parallel (uncounted, equity-first):** trend mechanics end-to-end;
   A4 banner patches for T/X; factory draft; W6 schema draft; joint-crash
   instrument design; observatory capture plumbing.
3. **Owner ratifies equity seams:** queue items 2–3 (T/X A4), then 4–6 as
   ready.
4. **After G0 (equity):** re-read economic narratives under micro-calibrated
   spreads before any GO story.
5. **After G4a:** uncounted joint equity construction + B1+trend crash
   receipts; G4b only with ≥1 sleeve at bar.
6. **Crypto (optional, bandwidth-gated):** pre-reg + paper/shadow only when
   steps 1–2 are not starved. Validation convenience does not reorder product
   priority.

A4 concurrency remains *permitted, not mandated*
(`docs/amendments_2026-07.md` A4). Operator bandwidth is a legitimate reason
to run nothing concurrently — and the default spend of that bandwidth is
**equity**.

## 8. Hedging doctrine

Three instruments, one composition policy. "Elegant hedging" means the
operator configures *one* policy surface, not three mental ledgers.

| Instrument | What it is | What it is not | Source |
|---|---|---|---|
| **De-gross** | State-conditional gross scale (θ, g=0.5, hysteresis) | Convexity / positive crash tail | `docs/sizing_preregistration.md` §3 |
| **Convexity sleeve** | Trend (TSMOM ETF book) admitted for crash complement + second decay | Standalone Sharpe engine | `docs/trend_design.md` §0 |
| **Construction netting** | G-P aim-portfolio blends heterogeneous-decay signals *inside* the optimizer | Sum-then-cap (measured anti-netting) | `docs/aim_portfolio_preregistration.md` |

Policy shape (product, W6): capital bands and enable flags over
`{equity_sleeve, trend_sleeve, crypto_book, de_gross}` with profile defaults.
No discretionary day-trading of hedge ratios. Joint stress receipts (W1/W4)
are the demo that the composition works.

IR is maximized **subject to** this envelope (max gross, de-gross policy,
custody float, optional live max-DD / PSR kill-switch). Hedging is not IR's
enemy when the objective is constrained correctly.
