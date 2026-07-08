# EDGE bracketing-diagnostic pre-registration (DRAFT)

**Status: drafted 2026-07-08, awaiting owner ratification.** The estimator code
lands in the same change (`src/prism/execution/edge.py`: `edge_spread`,
`edge_spread_by_symbol`, `edge_bracketing_diagnostic`); this document fixes its
evidentiary status *before* any output is read against a certified number, per the
repo precondition that a diagnostic's governance is settled ahead of its first
reading, not after. Ratification follows the momentum precedent
(`docs/momentum_design.md`): the agent drafts, the owner ratifies. Nothing in §2
(governance) or §3 (adjudication) is live until that ratification; no value in this
document moves once it is.

## 0. Provenance and rationale

EDGE is the Ardia–Guidotti–Kroencke (2024, *JFE*) effective-spread estimator: it
recovers the effective bid–ask spread from open/high/low/close bars alone, with no
quote or trade tape. It is registered here as a **second, $0, OHLC-only instrument
sitting beside the fills ledger** — not as a replacement for it.

The fills-ledger instrument it sits beside is `src/prism/execution/spread.py`,
which its own module docstring frames as *"the estimator half of the R2
cost-measurement instrument"*: the paper loop's fills ledger
(`prism.live.read_fills_ledger`) records every fill against its decision-close
reference price, and `spread.py` turns those rows into the per-bucket spread table
that **replaces** the pre-registered conservative-upper schedule once fills exist
(`arrival_slippage_bps`, `calibrated_bucket_schedule`). That instrument is
quote-grade but slow: it accrues one row per fill and, at N2 next-open fills,
carries an overnight-drift variance that only large $n$ retires (hence `se_bps` on
every estimate and the `min_fills` promotion floor).

EDGE fills the complementary gap. It needs no fills, no capital, and no waiting —
it reads the same OHLC bars the WFO already consumes — and it answers exactly one
question: **does the pre-registered `SPREAD_BUCKET_SCHEDULE_V1` sit above, inside,
or below the EDGE per-bucket effective-spread distribution?** That is a bracketing
question, not a calibration question, and the distinction is the whole of §2.

The schedule under test (`docs/r2_design.md §3`, conservative-upper, one-way bps by
dollar-volume bucket):

    ADV >= $500M   : 1.0 bps      $100M-$500M : 2.0 bps
    $25M-$100M     : 5.0 bps      < $25M      : 10.0 bps

with dollar-volume floors `(500e6, 100e6, 25e6, 0.0)`
(`spread.py::DEFAULT_BUCKET_FLOORS`, duplicated by value from the research WFO's
`SPREAD_BUCKET_SCHEDULE_V1`). These are the numbers I-9 governs.

**The relevant invariant (SPEC.md §6, I-9 · *Cost is calibrated, not a flat
constant*):** the effective spread is calibrated per liquidity bucket, *"plus an
arrival-slippage / adverse-selection term estimated from paper/live fills once they
exist; until then half-spread stands as a conservative upper proxy on liquid ETB
retail-notional fills. Every net claim records its spread assumption."* I-9 names
exactly one thing that replaces the proxy: fills. EDGE is not that thing, and this
document does not make it that thing.

## 1. What this diagnostic can and cannot show

Every EDGE reading is computed from the same historical OHLC as the backtest and
carries the estimator's own market-wide interpretation. It therefore exists for
**bracketing only** — it can flag that the schedule is optimistic against a
market-wide effective-spread reference; it cannot set the schedule.

- **It can bracket.** For each liquidity bucket it produces an effective-spread
  distribution over the constituent names, and reports the pre-registered floor's
  position relative to that distribution (ABOVE / INSIDE / BELOW, §3). This is a
  cheap, continuous sanity read on whether the conservative-upper schedule is in
  fact conservative.
- **It cannot calibrate.** Moving `SPREAD_BUCKET_SCHEDULE_V1`, or any per-bucket
  value entering a net claim, is reserved to the fills path (I-9;
  `spread.py::calibrated_bucket_schedule`). EDGE has no fill-direction signal, no
  adverse-selection term, and no venue/fee content (I-7), so it cannot produce the
  quantity I-9 requires a net claim to record.

**Known caveat, recorded so it is not lost.** EDGE estimates the effective spread
of *all* trades transacting in the name — the market-wide round-trip. This book's
paper/live flow is marketable retail notional routed through Alpaca, which
frequently price-improves *inside* the effective spread. For this book, therefore,
**EDGE is an upper-ish bound on realized execution cost, not an unbiased estimate
of it** — which is the *conservative* direction for a cost harness: an EDGE reading
that sits above a bucket floor does not by itself indict the floor, because our
realized fills are expected to land below EDGE's market-wide number. This asymmetry
is why EDGE is registered as a bracket and the fills ledger as the authority, not
the reverse.

**Conversion (so the bracket is apples-to-apples).** `edge_spread` returns the
**full effective spread as a fraction of price** (round-trip). The schedule is
stated in **one-way bps** (half-spread; `spread.py` documents the schedule shape as
*"(dollar-volume floor, one-way bps)"*, and I-9's proxy is explicitly the
half-spread). The comparison is made in one-way bps:

    one_way_bps = edge_fraction / 2 * 1e4

Worked check: an EDGE effective spread of `0.0004` (4 bps round-trip) →
`0.0004 / 2 * 1e4 = 2.0` one-way bps → compared against the `$100M–$500M` floor of
`2.0` bps → INSIDE (at the floor). `edge_bracketing_diagnostic` performs this
conversion before comparing; no bracket is ever read from the raw fraction against
a one-way floor.

## 2. Evidentiary status / governance (pre-committed)

- **EDGE is registered as a bracketing diagnostic only.** It is a monitored read,
  in the same posture I-8 assigns a regime feature that has not earned its way into
  sizing: logged as a diagnostic, not fed to the machinery that moves money.
- **Fills remain the sole calibration authority.** The only pre-registered event
  that recomputes a certified number under new spreads is a fills contradiction.
  Per `docs/handoff.md §8` (Pre-registered decisions), the trigger row reads
  verbatim: *"Paper fills contradict the 1bp assumption → Recompute the entire
  historical verdict under calibrated buckets before any new work — the past
  numbers change meaning."* **EDGE does not trigger this recompute. Only paper/live
  fills do.** An EDGE bracket, in any direction, is never the "fills contradict"
  condition and never initiates the historical-verdict recompute.
- **EDGE does not enter claim packets as a calibration source.** A `net_edge`-tier
  claim records its spread assumption as the calibrated per-bucket spread (SPEC.md
  §6 tier table; I-9), which is the schedule or its fills-calibrated successor —
  never an EDGE number. EDGE may be attached to a packet as a diagnostic
  annotation; it may not appear as the basis of a net-of-cost figure.
- **EDGE moves no ratified statistic, introduces no counted trial, and needs no new
  selection set.** It sweeps nothing, fits nothing, and adds no configuration hash.
  It is not a trial in any ledger and is exempt from `--design_trials` accounting.
  Reading EDGE cannot spend, refill, or reweight any budget, and cannot deflate or
  inflate any Sharpe/DSR against any selection set.

The governance claim in one line: **EDGE changes what we *know* about the
schedule's conservatism; it changes no *value* and fires no *trigger*.** Every
branch that could move a certified number runs through fills.

## 3. How it is read (adjudication)

For each liquidity bucket $b$ with pre-registered floor $s_b$ (one-way bps):

1. Compute the per-name EDGE effective spread over the evaluation window
   (`edge_spread_by_symbol`), map each name to its bucket by the same
   formation-window median dollar-volume screen the WFO uses, and convert to
   one-way bps (`/2 * 1e4`).
2. Summarize the bucket's EDGE distribution (report the median and an inner
   interval — the interquartile band — so a single wide-spread name does not drive
   the verdict).
3. Emit the bracket (`edge_bracketing_diagnostic`):
   - **ABOVE** — $s_b$ sits above the bucket's EDGE central tendency: the schedule
     is (at least) as conservative as the market-wide reference. Expected and
     unremarkable; no action.
   - **INSIDE** — $s_b$ falls within the bucket's EDGE distribution: the schedule
     is plausibly close to the market-wide effective spread. Logged; still no
     action on the schedule.
   - **BELOW** — $s_b$ sits below the bucket's EDGE central tendency: the
     conservative-upper schedule reads *optimistic* against the market-wide
     reference. This is the informative case — and even here, **the schedule does
     not move and no verdict recomputes.** A BELOW bracket is a prompt to
     prioritize accruing *fills* in that bucket (the only instrument that can move
     the number under I-9), not a warrant to edit `SPREAD_BUCKET_SCHEDULE_V1`.

**No branch of this adjudication changes a ratified value or fires a recompute.**
The bracket is read, logged, and — at most — used to sequence where fill collection
is most urgent. The historical verdict changes meaning under exactly one condition,
and it is not on this page: paper fills contradicting the 1bp assumption
(`docs/handoff.md §8`). EDGE informs the eye that watches the schedule; the fills
ledger, and only the fills ledger, holds the pen.
