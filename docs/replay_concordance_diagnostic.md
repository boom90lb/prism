# Replay↔backtest concordance diagnostic — read frame

**Status: uncounted mechanics diagnostic. Frame pinned 2026-07-13, before
the runs; results appended below the line after.** Boundaries per
`src/prism/live/replay.py`: modeled fills, never I-9 calibration, never the
live-monitor concordance stream; no ratified statistic can move on any
outcome here. A finding justifies code/ops fixes or an owner decision —
never a B1 configuration change (that would be a new discovery event,
`docs/momentum_design.md` §2).

## 1. Question

The M6 promotion read re-runs the *research backtest* unmodified, while the
paper stream conjunct reads the *live spine*. The two implement B1 with
twelve known mechanical divergences (audit 2026-07-10); signal and decile
construction are parity-pinned by test
(`tests/test_momentum_node.py`), so any book divergence lives in the
execution machinery around them. This diagnostic measures the divergence at
the **book level** — per-refresh target weights — where the comparison is
near-deterministic (same prices both sides), rather than at the equity
level, which is confounded by zero-cost replay fills, close-vs-open
marking, and the backtest's fold-boundary flattening.

## 2. Design

Two replays over the discovery window, driving the real
`run_daily_cycle` via `prism.scripts.replay_loop`:

- **Run A (mechanics):** `--cash 1e8` — whole-share quantum negligible, so
  A-vs-backtest isolates spine mechanics (cadence anchoring, band
  difference, flip-clamp, membership handling).
- **Run B (paper scale):** `--cash 1e5` — B-vs-A isolates the integer-share
  quantum and dust censoring at the (pre-reset) paper NAV. Descriptive
  only; the $1M reset decision is already made.

Shared config, matching the frozen B1 backtest
(`results/demotion_b1/config.json`): universe
`data/universe/sp500_pit_resolved_2026-06-16.txt` (574 names) with
`--membership-file data/universe/sp500_membership_2026-06-16.parquet`
(PIT gate ANDed into eligibility, as the backtest does), lookback 252 /
skip 21 / decile 0.10 / `decision_every` 21, `--max-symbol-weight 0.35`,
`--max-gross 1.0`, `--band 0.0`, window `--start 2021-03-30` (the
backtest's first test bar, fold 0 `test_start`, anchoring the 21-bar
refresh grid to the backtest's global grid `312 + 21k`) `--end 2026-06-12`
(the backtest's last test bar).

**Comparison object:** each replay `targets.jsonl` refresh row vs the
backtest's `results/demotion_b1/target_weights.csv` row on the same date
(nearest within ±2 trading days when the calendars drift; unmatched
refreshes reported). Backtest rows inside a fold's final 2 bars (the
fold-flatten zeros) are excluded — that divergence is already known,
one-sided, and conservative on the backtest side. Per matched refresh:
active share (0.5·Σ|Δw|), long/short membership overlap (Jaccard per leg),
gross both sides. Aggregate: mean/max active share, trend over time,
per-name attribution of the largest gaps.

## 3. Pre-pinned reading

Known, accepted divergence sources the attribution should land on: the
backtest's per-name closed-form band vs the replay's band-off path (holds
small reweights the replay trades), PIT-mask edge days, calendar drift,
flip-through-zero clamp (replay defers a leg-crossing to the next refresh;
the backtest flips in one step).

- **Concordant:** mean active share (A vs backtest) ≤ **0.05** — spine
  mechanics reproduce the backtest book to within band-scale noise; the M6
  conjunct-#4 read carries no material mechanical bias.
- **Investigate:** mean in **(0.05, 0.15]**, or any single refresh > 0.30 —
  decompose per-name, attribute to the named divergence sources before any
  further conclusion; fix what is fixable in the *instrument* (never the
  ratified config).
- **Escalate:** mean > **0.15** sustained, or a systematic one-sided drift
  (gross ratio trending, or one leg persistently under-represented) — a
  book gap of that size is the order of B1's edge (Sharpe 0.465 at 11.5%
  vol ≈ 480 bps/yr) and mechanically threatens the promotion conjunct;
  owner decides between spine changes (instrument repair) and accepting a
  documented wedge between the two evidence streams.
- B vs A: descriptive report (censored names, weight quantization error at
  $100k). No threshold.

---

## 4. Results (appended after the runs; frame above unchanged)

*Pending.*
