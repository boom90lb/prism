# Account-size floor — whole-share discretization of the B1 book vs capital

> **Status: uncounted diagnostic (run 2026-07-18).** Measures an execution
> mechanic, searches nothing, moves no ratified statistic. Instrument:
> `research/scripts/account_size_floor.py` over five replay runs; artifact:
> `results/account_size_floor_2026-07-18.json` (tracked). Modeled fills
> (prism/live/replay.py): the cross-size *relative* comparison is the
> measurement; the absolute window return is one membership-blind sample and
> claims nothing.

## 1. Question

The live loop sizes whole-share OPG orders (venue requirement). Below what
capital does the ~100-name decile book stop being the book the strategy
decided? Per-name targets are ~1% of equity, so at $10k a $100+ stock cannot
enter at all — the dust filter censors it loudly
(`prism.live.loop.targets_to_orders`, N7) — and the instrument holds a
different portfolio than the decision constructed.

## 2. Method

Five replays of the B1 momentum book (`prism.scripts.replay_loop`), identical
in every respect — universe `data/universe/sp500_current.txt`, window
2026-01-02 → 2026-06-15 (113 sessions, 6 refreshes), consensus 0.6,
membership-blind, whole shares — except `--cash`. For each refresh, the
achieved post-fill book is reconstructed from the run's own ledgers and scored
against the constructed targets with the same `book_concordance` metric the
live loop logs.

## 3. Results

| Starting cash | Active share (mean / max) | Gross deployed | Names censored per refresh | Window return | Gap vs $1M |
|---|---|---|---|---|---|
| $1,000,000 | 0.003 / 0.003 | 1.000 | 0.0 | +26.7% | — |
| $100,000 | 0.027 / 0.031 | 1.002 | 0.0 | +27.0% | +0.3pp |
| $50,000 | 0.062 / 0.070 | 1.016 | 1.7 | +26.0% | −0.7pp |
| $25,000 | 0.113 / 0.124 | 0.942 | 9.3 | +23.4% | −3.3pp |
| $10,000 | 0.281 / 0.295 | 0.705 | 39.7 | +12.0% | −14.7pp |

Cross-check: the $100k mean active share (0.027) matches the 0.026 quantum
measured independently on the live concordance instrument
(docs/replay_concordance_diagnostic.md), which is the consistency you would
demand of two instruments measuring the same mechanic.

## 4. Reading

Anchor: the live concordance bar is 0.05 active share (the threshold under
which the paper stream counts as evidence about the strategy,
docs/momentum_design.md).

- **$100k and above:** clean. Zero censoring; divergence is pure rounding
  granularity, comfortably under the bar.
- **$50k:** borderline. Mean 0.062 sits just over the bar; a couple of
  high-priced names drop out per refresh. Runnable, but the stream is
  marginally a measurement of a slightly different book.
- **$25k:** degraded. ~9 names censored per refresh, 94% gross deployed —
  the book is visibly not the decided book.
- **$10k:** structurally different. 40 of ~98 names never enter, 70% gross,
  and in this (hot) window the under-deployment cost 14.7pp of the move. The
  sign of the gap is window-dependent; its existence is not.

**Floor statement:** with whole-share OPG orders, ~$100k holds the B1 book
faithfully; $50k is the edge of the concordance bar; below that the account
trades a materially different portfolio. The in-tree escape hatch for small
accounts is `--tif day` (fractional shares admitted, at the cost of trading
the open market instead of the auction) — unmeasured here; measuring the
DAY-order fractional path at $10–25k is the natural follow-up.

## 5. Limits

One window, one universe snapshot, membership-blind, modeled fills that
print everything (live OPG auctions print ~20–25% before the sweep, which
adds a size-independent divergence on top). Censoring depends on the price
distribution of the current membership — a future high-priced index cohort
moves the floor up.
