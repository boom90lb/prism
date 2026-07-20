# Bar-vendor divergence — measurement record

**Status: uncounted, read-only diagnostic (§1–§4, recorded 2026-07-19); §5 is
the owner-approved remediation mechanic, landed default-off.** No ratified
statistic moves here; no counted machinery invoked. Like the dividend wedge
and the carry-flatten counterfactual, the measurement is **divergence-ledger
context for M6, not an amendment** — folding any of it into
`docs/momentum_design.md` §3 is an owner decision.

## 1. Question

The live paper loop decides and marks on Alpaca IEX daily bars
(`src/prism/live/alpaca_data.py`, `DEFAULT_FEED = "iex"` — chosen so decision
and fill share one venue and one clock), while certification, backtests, and
the replay concordance all price off the spine-vendor parquet caches
(`data/*_1d_*.parquet`). `docs/replay_concordance_diagnostic.md` established
CONCORDANT with the **same prices on both sides by construction**, and its
closing residual-wedge list (§"Consequence for M6", lines 129–135) names venue
fill rates and the IEX-volume eligibility screen — bar-vendor divergence is
absent from it. Decile membership (`floor(n·0.10)` per leg,
`prism.portfolio.construct._decile_row`) and NAV marks are rank- and
level-sensitive, so a close-level disagreement between the two vendors is a
live-vs-certified wedge the concordance instrument is structurally blind to.
This record measures it.

## 2. Method

**Instrument:** `research/scripts/bar_vendor_divergence.py` (tests in
`tests/test_bar_vendor_divergence.py`; the network seam is an injectable
requests-compatible session, the `AlpacaBarSource` pattern, so every mapping
runs offline). Evidence: `results/bar_vendor_divergence_2026-07-19.json`.

**Panels.** Spine closes from the frozen bar caches
(`_1d_2020-01-01_2026-06-16.parquet`); IEX closes via
`AlpacaBarSource.fetch_batch` (feed `iex`, `adjustment=split`), fetched from
2025-01-02 so every in-window decision bar's 252-bar lookback endpoint lands
inside IEX history. The fetch covers the **union of the universe file and the
held book** — the live loop itself prices file ∪ book (the 64d7ea1 lesson;
POOL is held, exit-only, and absent from `sp500_current.txt`). 21 paginated
multi-symbol requests for 503 names, well inside the IEX budget.

**Close-diff panel.** `(iex/spine − 1)·1e4` bps over the common sessions of
2026-01-02 → latest common session, per-name and pooled distributions, tail
fractions at 5/25 bps. Names or sessions on one side only, and name-days NaN
on one side, are counted and listed, never dropped (N7). (Per-name listing of
spine-side NaN name-days — `names_partial_spine`, symmetric to
`names_partial_iex` — exists from this revision; the published 2026-07-19 run
predates it and carries the spine-side count in aggregate only.)

**Adjustment-basis flagging.** The spine caches are frozen at 2026-06-16;
Alpaca serves a *current* split-adjusted series. A corporate action after the
freeze rebases Alpaca's whole history for that name, so a level-stable ratio
(|median diff| > 250 bps) is an adjustment-basis artifact, not price
divergence: flagged, reported separately, excluded from the distribution and
rank reads.

**Rank impact.** At each month-end decision bar in the window (Jan–May 2026;
the truncated June stub is dropped, and month-end is a representative grid —
the live cadence is a 21-bar refresh), the 12-1 score
`close[t−21]/close[t−252] − 1` (the `MomentumSignalNode` convention) is
computed from both vendors' closes at the **same two endpoint sessions** on
the spine calendar, and top/bottom-decile membership (`floor(n·decile)` per
leg, stable sort over the symbol-sorted cross-section) is compared on the
names finite on both sides. One-sided names are excluded loudly. The
eligibility screen the live book applies is *not* reproduced here; the rank
read is over the current-universe cross-section.

**Decile-boundary sensitivity (fallback/robustness).** Spine endpoint closes
perturbed multiplicatively by draws from the measured empirical diff
distribution (iid per name per endpoint, 200 draws, seed 20260719), leg flips
recounted against the unperturbed legs. With the 2025-01-02 fetch start the
direct read covered every refresh (PSKY the one exclusion, IEX-history-short),
so this served as a robustness band rather than the primary read.

**NAV mark.** The held 98-name book (`runs/paper_loop_momentum2/state.json`,
last settled 2026-07-13) marked session by session on IEX vs spine closes,
difference in bps of NAV (cash included). Two reads: **raw** (every held
name — what a naive cross-vendor mark would show) and **ex-flagged** (genuine
per-session vendor divergence, the headline), because a held
adjustment-flagged name marks hugely differently for frozen-cache reasons,
not vendor disagreement. Marking a book decided 2026-07-13 over the Jan–Jun
overlap window is a counterfactual mark of the current book, stated as such.

## 3. Results

Window effective 2026-01-02 → 2026-06-15, 113 common sessions, 503 common
names (502-name universe plus held POOL). Zero names missing on either side;
22 sessions are IEX-only (post-freeze, expected); 4 name-days NaN on IEX
inside the window (FDXF 3, HONA 1 — new listings' first days). HONA has
**zero** sessions priced by both vendors — the spine cache holds only its
2026-06-15 debut bar and IEX lacks exactly that bar — so it is absent from
every distribution, reported not silent; hence 497 distribution names
(502 − 4 flagged − HONA).

**Adjustment-basis flags (4):** CRWD −7,500 bps (ratio ×0.25), DD +19,998
(×3.0), HON +10,000 (×2.0), FDX +2,409 — level-stable across all 113
sessions, i.e. corporate actions between the 2026-06-16 cache freeze and the
fetch date, rebasings not price disagreements. (Corrected in §6: FDX is a
*pre-freeze spin-off* of the genuine-divergence class, not a post-freeze
rebasing, and its shift is not in fact level-stable across all 113 sessions.)

**Close-diff distribution** (497 universe names ex-flagged, 56,059
name-days): median |diff| **2.14 bps**, mean |diff| 4.29, p95 8.84, p99
16.0 bps; signed mean +1.25 bps (IEX marginally above spine). Exactly equal
closes: 8.4% of name-days. Tails: **17.0%** of name-days beyond 5 bps,
**0.28%** beyond 25 bps, max 2,730 bps (BDX). (§6 reclassifies FDX — excluded
from these stats as adjustment-flagged — as genuine convention divergence;
restoring its ~102 in-window divergent name-days at ~2,400 bps puts the
>25 bps tail near 0.46%, roughly double the ex-flagged figure quoted here,
and lifts p99. Cite the ex-flagged numbers only with that caveat.) The tail
is not noise: BDX
(~2,730 bps level shift across late Jan–Feb 2026), CMCSA (670 bps on
2026-01-02 only), DOW (149 bps on 2026-03-30 only) are
corporate-action-window disagreements — the spine back-adjusts spin-offs
while Alpaca's split-only series does not, so the two series part company for
bars before an action date until the event clears the comparison window.
(§6 withdraws DOW from this attribution — no action record exists for it —
and qualifies the universal: spine spin-adjustment is not uniform; APTV's
spin sits raw on the spine too.)

**Rank impact** (5 refreshes, ~494 names, 49 per leg): **0 long-leg flips, 4
short-leg flips** — one per refresh Jan–Apr, zero in May. Spearman
0.9972–0.9996; median |score diff| 3.6–4.6 bps per refresh. The flip driver
is the spin-off adjustment convention, not close noise: FTV's 2025-01-30
lookback close is 61.51 on the spine (spin-adjusted) vs 81.62 on IEX (raw),
depressing its IEX-side score into the short decile (short-in at the Jan and
Feb refreshes; NCLH and CRM displaced); BDX enters short on the Mar refresh
the same way; the Apr flip (KMB in, DPZ out) is genuine boundary noise. WDC's
Jan-refresh score differs by 8,648 bps (lookback endpoint predates the SNDK
separation) without flipping a leg — it sits nowhere near the boundary.
Excluded per refresh: PSKY (spine score only, IEX history short) — 1 name,
listed. The spine's spin-adjusted series is the economically correct basis
for a price-ratio score, so on affected names the live Alpaca-fed rank is
the distorted one until the lookback window clears the event. (§6 qualifies:
spine spin-adjustment is non-uniform — on APTV-class names the raw
distribution step sits on *both* bases and the distortion is shared, not
live-only; "spine correct / Alpaca distorted" holds for the BDX/CMCSA/FDX/FTV
class, not universally.)

**Decile-boundary sensitivity** (200 draws/refresh): mean 0.10–0.82 flips per
refresh, p95 ≤ 2, max 2 — consistent with the direct read's ~1 flip per
refresh at 49 names per leg.

**NAV mark** (113 sessions, min coverage 97/97 ex-flagged): headline
ex-flagged median |ΔNAV| **0.45 bps**, mean 0.54, max **1.78 bps** — the
long-short book nets the already-small per-name diffs. Raw including held DD:
median **84.6 bps**, max 111.8 — entirely the DD ×3.0 rebasing, i.e. an
artifact any naive cross-vendor comparison against the frozen caches would
show for as long as a post-freeze-rebased name is held. No held name was
unpriceable on either vendor.

## 4. Reading and limits

The wedge is bimodal. The bulk is venue close noise — IEX official closes vs
the spine's consolidated closes — at ~2 bps median, which nets to sub-2 bps
on the hedged book's NAV and moves essentially nothing. The consequential
component is **corporate-action adjustment-convention divergence**
(spin-offs chiefly): it produced every systematic rank flip measured here
(~1 name per leg per refresh at the short boundary, ~2% of a leg) and, for
post-freeze actions, a level artifact that a naive live-vs-certified equity
comparison would misread as tens of bps of NAV divergence. For the M6
divergence ledger: (a) live decile membership can legitimately differ from a
spine-computed book by ~1 boundary name per leg on refreshes where a recent
spin-off sits inside the lookback window — a conjunct-#4 read should check
the flip lists here before attributing such a difference to spine mechanics;
(b) any cross-vendor NAV or price comparison must exclude or re-base
adjustment-flagged names first, or it measures the freeze, not the vendors.
(§6 qualifies rule (b): the flag set mixes mechanisms — for FDX-class names
the median gate auto-flags *genuine* convention divergence, so excluding
them removes real vendor-divergence evidence under a freeze-artifact label;
separate the classes per §6 before excluding.)

Limits, stated: the eligibility screen is not reproduced (rank read is the
raw current-universe cross-section); the month-end grid approximates the
21-bar live cadence; the universe file is the 2026-07-17 regeneration read
back over Jan–Jun (survivorship in the cross-section, fine for a vendor-diff
read, not a backtest); the NAV read marks the current book counterfactually
over the overlap window; and the spine freeze bounds the comparison at
2026-06-15 — the diagnostic is rerunnable against any fresher cache vintage
by pointing `--cache_suffix` at it.

## 5. Remediation — spin-off eligibility mask (owner-approved; mechanic landed 2026-07-19)

**Status: mechanic landed default-off; live activation recorded in the
nightly runner.** With the flag off the loop is bit-identical to the
unmasked loop (`tests/test_spinoff_mask.py::test_flag_off_is_bit_identical_to_empty_mask`).

**Mechanism.** A name with a spin-off ex-date inside the trailing 252-bar
lookback at a refresh is **unrankable**: it is scored NaN before construction
(so it cannot distort decile membership — the decile recomputes over the
rankable cross-section) and its target carries *no decision* (NaN), which the
live loop's pinned NaN-target semantics resolve to hold-never-liquidate
(`step_no_trade_band`, `prism/portfolio/construct.py`; `targets_to_orders`,
`prism/live/loop.py`). Net: no NEW position may open on a divergent rank; an
already-held flagged name is HELD until the event clears the lookback, then
ranks normally. Valuation-only extras (index leavers still held) stay
exit-only — an index-leaver's exit is a universe decision, not a rank
decision.

**Detection.** Alpaca corporate-actions `spin_off` records (the dividend-wedge
surface, paper key, $0) over the fetched universe (file ∪ held) and the
trailing lookback window, at each refresh session only
(`prism/live/spinoff_mask.py`; applied at the `unrankable` seam of
`run_daily_cycle`; CLI `--spinoff-mask` on `prism.scripts.paper_loop`). The
per-decision-bar answer is cached as `spinoff_mask_<bar>.json` in the run
dir — a same-bar rerun never refetches, and only events with ex-date ≤ the
decision bar flag (causal). A detection failure is a loud N7 warning naming
every unchecked symbol and the refresh proceeds UNMASKED: the mask is a
protection, not a correctness precondition. **Endpoint shape live-verified
2026-07-19** (direct probe, current universe, trailing year): 10 records,
`source_symbol`/`ex_date` fields as coded, and the in-window flag set
{APTV, BDX, CMCSA, DD, FDX, HON, SPGI} independently matches this
document's §1 adjustment-flagged names — two instruments concordant; WDC/FTV
correctly absent (their events have aged out of the trailing window).
(Superseded in part by §6: the set relation is superset-not-match, and "§1"
should read §3.) Replay: the library seam
(`replay_daily_cycles(unrankable=...)`) accepts an injected offline provider;
the replay CLI carries no flag because a replay has no event source.

**Residual, stated.** A position entered on a divergent rank before masking
(or during an unmasked refresh, e.g. after a detection failure) persists at
most 252 bars — the mask blocks entries and holds; it never forces an exit.
FTV's two short-boundary entries (§3) are exactly this class. A second
residual — the mask's rename-window limitation: detection exact-matches
record `source_symbol` against current panel symbols, so if the vendor keys
historical records to event-time symbology, a parent renamed after an
in-window spin-off goes unmasked (which convention Alpaca uses is
unverified; `prism/live/spinoff_mask.py` module docstring).

**M6.** The per-refresh masked-name lists (the run-dir `spinoff_mask_*.json`
records) join the divergence ledger §4 routes: a conjunct-#4 read consults
them — alongside the §3 flip lists — before attributing a live-vs-spine book
difference to spine mechanics.

## 6. Correction record — post-publication probes complete the action taxonomy (2026-07-19)

Two attributions above, and the §5 addendum's concordance sentence, do not
survive a full-action-type probe (all corporate-action types, trailing year,
the §5 endpoint and key; spine cache reads and Alpaca quotes inline below).
The superseded text stands in place with pointers here — the record of the
record is part of the record.

**The §3 flag set mixes two mechanisms.** CRWD (1:4 forward split ex
2026-07-02), DD (3:1 reverse ex 2026-06-24), HON (2:1 reverse ex 2026-06-29)
are post-freeze split rebasings exactly as §3 frames them — each ratio
reproduces its constant (×0.25 → −7,500 bps; ×3.0 → +19,998; ×2.0 →
+10,000), and DD's/HON's own spin-offs (Q ex 2025-11-03, SOLS ex 2025-10-30)
predate the window and leave no in-window signature. **FDX does not belong
in that class**: its event is a spin-off (FDXF, ex 2026-06-01) *before* the
cache freeze, and its panel signature is not level-stable — mean 2,174 vs
median 2,409 bps is the arithmetic of 102 divergent pre-event sessions and
11 near-zero post-event ones (2,408.69 × 102/113 = 2,174.2, the recorded
mean; ex 2026-06-01 → window end 2026-06-15 spans 11 sessions). FDX is the
BDX/CMCSA/FTV class (spine
back-adjusted, Alpaca raw), auto-flagged only because a mid-window event
this large drives the median past the 250 bps gate. Its exclusion from the
§3 rank read cost nothing for Jan–May: a cross-vendor *score* diverges only
while the ex-date sits **between** the two score endpoints (with both
endpoints pre-event the adjustment factor cancels in the ratio), and FDX's
divergence window opens after 2026-06-01 — the one live decision inside it
so far is 2026-07-08, pre-mask, which is §5-residual class.

**DOW is withdrawn from the §3 spin sentence.** No spin-off or split exists
for DOW in the trailing year (cash dividends only); its single divergent day
(149.3 bps, 2026-03-30) is quarterly-dividend-sized (0.35/share). Mechanism
unattributed; it is one day of the 0.28% > 25 bps tail, not established
convention divergence.

**New finding — the spine's spin-adjustment is not uniform.** APTV (spin-off
VGNT ex 2026-04-01, 1/3 share per share) shows *zero* panel divergence (max
8.4 bps) because **both** series are raw across the event: the spine's own
2026-03-31 → 04-01 return is −1,058 bps against a mechanical
distribution fraction of −13.3% (0.333 × VGNT's 27.77 debut close / APTV's
69.44 pre-event close). The ~275 bps residual is attributed to the ex-day
move of the post-distribution stub and is the one uncited link in this
chain — the directional inference does not depend on it (−10.6% is
incompatible with a back-adjusted series, and the 8.4 bps max panel diff
pins both vendors to the same basis), but the follow-up sweep must set its
match tolerance to absorb ex-day moves rather than inheriting a tight fit
from this example. The spine did not
back-adjust this spin while it did back-adjust BDX, CMCSA, FDX, and FTV.
Consequence: the certified price basis is not uniformly spin-adjusted, and
every refresh whose score endpoints straddle 2026-04-01 scores APTV against
a raw distribution step *on both vendors* — not a live-vs-certified wedge,
but a scoring distortion the certified side shares. Named follow-up
(owner-sequenced): a spin-adjustment-consistency dimension for
`research/scripts/data_integrity_sweep.py` — for each in-universe `spin_off`
record, test the cache's cross-event return against the distribution
fraction; this probe, mechanized.

**The §5 addendum sentence is superseded.** The probe flag set
{APTV, BDX, CMCSA, DD, FDX, HON, SPGI} is "names with a spin-off ex-date in
the trailing window" — a **superset** of "names whose vendors diverge," not
a match (and "§1" should have read §3). Per name: BDX, CMCSA, FDX diverge
(convention class, above); DD and HON enter via pre-window spins whose panel
shifts are post-freeze splits; APTV enters with both vendors agreeing (raw,
above); SPGI (MBGL ex 2026-07-01, post-freeze) diverges only prospectively —
and whether it materializes at all depends on which path the next
post-event spine vintage takes: the BDX path (back-adjusted → cross-vendor
divergence appears) or the APTV path (raw → no divergence, and the
certified basis silently carries another shared distribution step — the
worse outcome, invisible to any cross-vendor read; the named sweep, not
vendor comparison, is the instrument that catches it). Either way SPGI is
the case the mask exists for. The mask stays correct as a
protection — every over-flag is a conservative hold, and APTV is unrankable
for the deeper reason that its lookback spans a raw distribution step on
both vendors — but an M6 divergence-ledger read must not equate "masked"
with "vendor-divergent": the §3 flip lists carry divergence evidence; the
mask lists carry event exposure.
