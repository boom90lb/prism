# Carry-mode fold-flatten counterfactual — measurement record

**Status: uncounted, read-only diagnostic. Recorded 2026-07-17.** This
closes the open item in `docs/program_review_2026-07.md` §1 finding 3 (§5
"still open" list). No ratified statistic moves here; no B1 configuration
changes; the walk-forward machinery was never invoked (the counterfactual
replays recorded targets through the frozen accounting engine). Like the
dividend wedge (program review §4), the result is **divergence-ledger
context for M6, not an amendment** — folding any of it into
`docs/momentum_design.md` §3 is an owner decision.

## 1. Question

The certified equity process force-flattens the book on the last two bars
of every fold (`research/arbitrage/walk_forward.py:264-271`; 20 interior
boundaries in B1), charging full round-trip costs; the live loop never
flattens. Two consumers care about the size of that artifact: the M6
divergence read (live equity vs certified, where flattening makes the
certified number *conservative*), and the eventual sizing decision (which
consumes the certified drawdown/vol/skew statistics — are they artifacts
of the flatten?). The prior "~63% of B1 turnover" figure was a derivation
from fold aggregates, known to overstate the incremental cost
(program review §2); this record replaces it with a measurement.

## 2. Method

**Instrument:** `research/scripts/carry_flatten_diagnostic.py` (tests in
`tests/test_carry_flatten_diagnostic.py`). The run's recorded
`target_weights.csv` (1,308 decided rows) already encodes every
signal/band/cap/cadence decision, so counterfactuals need only weight-space
accounting: the frozen engine
(`prism.execution.target_weights.backtest_target_weights`) is replayed on
the run's own bar caches (quarantine included — the run consumed those
bars before quarantine) with the run's spread schedule and execution
config.

**Validation gate, asserted before any counterfactual is read:** replaying
the recorded targets must reproduce the recorded `returns.csv`. Measured:
max abs daily-return diff **3.4e-16** (tolerance 1e-12), Sharpe reproduced
to 15 significant figures. The replay *is* the run's accounting.

**Counterfactual:** at each of the 20 interior fold boundaries, the two
zeroed target rows are replaced by the fold's last held book, so the next
fold's opening targets trade from that book instead of from flat.
Variant **A** keeps the terminal (window-edge) flatten and is the
headline — it isolates the recurring systematic component a
live-vs-certified comparison consumes. Variant **B** carries every
boundary including the terminal one (pure mark-to-market), reported for
completeness.

**Known approximation:** the recorded fold-opening targets were themselves
band-stepped from flat, so a true carry regime would additionally hold
small legacy deviations these targets trade away. The measured saving is a
**lower bound** on the artifact.

## 3. Results

Evidence: `results/carry_flatten_diagnostic_2026-07-17.json`.

| | certified (replayed) | carry-A | carry-B |
|---|---|---|---|
| Sharpe | 0.4654 | **0.5199** | 0.5461 |
| annualized vol | 11.51% | 11.65% | 11.67% |
| max drawdown | 14.00% | 14.26% | 14.26% |
| daily skew | −0.491 | −0.452 | −0.447 |
| worst day | −3.98% | −3.98% (identical) | −3.98% (identical) |
| avg gross | 0.968 | 0.999 | 0.999 |
| avg daily turnover | 0.0500 | 0.0281 | 0.0274 |
| total cost (bps/yr) | 71.2 | 51.8 | 51.0 |

**Turnover decomposition (per interior boundary):** the flatten round-trip
is ~2.00 gross vs 0.57 of genuine repositioning under carry — carrying
does 28.6% of the flatten's boundary trading. Aggregated, the
flatten-attributable churn is **43.7% of certified total turnover**
(0.0218 of 0.0500/day), replacing the "63%" derivation, which counted the
full flatten round-trip as incremental although carry must still do 0.57
per boundary of it.

**Flatten-attributable cost: −19.4 bps/yr** on the certified stream
(−20.2 under variant B) — cost the certified process pays that the live
process never will.

**The tail statistics are not flattered by the flatten.** Worst day is
identical, skew moves slightly *less* negative under carry, kurtosis
slightly lower, max drawdown +0.26pp. The quarterly forced flatten was
providing no crash truncation; the certified risk statistics the sizing
decision will consume are properties of the strategy, not of the fold
mechanics. (The sample still contains no momentum crash — finding 2
stands untouched.)

## 4. Combined M6 divergence pin (stated now, before any comparison is read)

Two measured systematic wedges now sit between the live/paper equity
stream and the certified price-return process:

- **fold-flatten cost** (this record): certified pays **+19.4 bps/yr** the
  live process does not → live runs *above* certified by that much;
- **dividend wedge** (program review §4): live is total-return on an
  anti-yield book → live runs **−36.1 bps/yr** below certified
  price-return.

**Net pin: live ≈ certified − 17 bps/yr**, at ~3% higher average gross
(0.968 → 0.999) and ~1.2% (relative) higher vol. A marginal conjunct-#4
(rolling PSR) read at M6 must be read against this pin before any
mechanical or venue explanation is entertained — the spine is concordant
(`docs/replay_concordance_diagnostic.md`), both wedge components are
measured, and they mostly cancel.

## 5. What this does and does not license

It licenses: the M6 pin above; using carry-A (not certified) statistics as
the better *description* of live-process risk when sizing is eventually
argued; retiring the "63%" figure. It does not license: any change to the
certified numbers (Sharpe 0.4654 under lineage `583b9155eab7` stands), any
edit to the ratified pre-registration, or treating the +0.054 carry
Sharpe uplift as edge — it is an accounting artifact of where fold
boundaries fall, measured precisely so it can be *subtracted*, not
claimed.
