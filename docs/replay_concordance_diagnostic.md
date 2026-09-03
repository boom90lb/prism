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

## 4. Results (recorded 2026-07-13; frame above unchanged)

Runs: `runs/replay_concordance_a` ($100M) and `_b` ($100k), 1308 cycles
each, 2021-03-30 → 2026-06-12, 63 refreshes on an exactly regular 21-bar
grid; `--consensus 0.6` (the 0.95 default drops the early calendar — the
574 ever-member caches thin out before later IPOs list; 0.6 mirrors the
backtest's own peak-presence filter). Every replay refresh matched a
backtest row on the **exact date** (1308 rows both sides, calendars
identical); zero fold-flatten rows were hit.

**Verdict: CONCORDANT.** Run A vs backtest, per-refresh active share:
**mean 0.0067, median 0.0000, max 0.0319** (2021-12-28) — well inside the
≤0.05 threshold. Leg Jaccard: long mean 0.992 (min 0.959), short mean
0.992 (min 0.938). Gross 1.0000 vs 1.0002; book size 95.3 vs 95.8 names.
No trend across eras (thirds: 0.0076 / 0.0040 / 0.0084). The live spine
reproduces the frozen backtest book at refresh level to boundary noise.

Attribution of the residual divergence (the 2021-12-28 max): marginal
decile membership at the rank/eligibility boundary — a handful of names
swap in/out at the decile edge (TDG vs FISV/FLT/RJF) and the per-name
weight differs in the fourth decimal (0.0109 vs 0.0106 = one fewer
eligible name that day, `floor(n·0.10)` per-leg count 46 vs 47). The
feared closed-form-band divergence does not materialize: decile entry/exit
moves (~1% weight) dwarf band widths, so the band never separates the
books at refresh scale. Median-zero says most refreshes are *identical*.

**Flip-through-zero clamp: zero events.** Across all 62 refresh
transitions (3,510 weight changes), no name's target crossed sign — the
one-refresh flip deferral the live spine imposes never binds for 12-1
monthly momentum in this sample. The largest suspected live-vs-backtest
wedge is empirically a non-event for B1.

**Quantum at $100k (B vs A, realized final books):** constructed targets
are identical by construction; the realized books differ by active share
**0.026**, per-name |weight error| mean 0.0005 / max 0.0034 (FIX, $1,952 —
held at ~⅓ weight error from single-share rounding). No name was fully
censored in this sample (no held name crossed the ~$2,000 zero-round line;
NVR/AZO never entered a decile here). Consistent with — and second-order
support for — the $1M reset already executed.

Equity, for completeness only (confounded per §1): replay +39.5%/+40.7%
(A/B, zero modeled costs, close-marked) vs backtest +27.6% net — the gap
is the modeled cost stack (~3.7% total), the backtest's quarterly
fold-flatten round-trips, and marking differences; no reading is taken
from it.

**Consequence for M6:** the promotion read's two evidence streams (research
backtest re-run; live paper stream) are mechanically concordant at the
book level to <1% active share. A conjunct-#4 failure at M6 therefore
cannot be blamed on spine mechanics measured here; remaining
instrument-side wedges are the live-only ones a replay cannot see — venue
fill rates (addressed by the completion sweep, momentum_design.md §5) and
the IEX-volume eligibility screen (open item).
