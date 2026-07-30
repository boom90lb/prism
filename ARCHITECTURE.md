# Architecture

What calls what, end to end. The README covers status and method;
`docs/operations.md` covers operational sharp edges; `SPEC.md` defines the
contracts this wiring implements.

The organizing abstraction is a cross-sectional engine, not a per-symbol
ensemble:

```
DATA → SIGNAL → RESIDUALIZE → CONSTRUCT → EXECUTE → LEDGER
         (ensemble = one node)  (breadth/cost-aware)  (next-open fills,
                                                       participation gate)
     conditioned throughout by a REGIME layer (curve · vol · liquidity),
     gated by the HARNESS (purged WFO · deflated metrics · claim tiers).
```

The production/research boundary (`SPEC.md` §9): everything importable from
`prism` is the production spine and is free of JAX, torch, prophet, mlflow,
and matplotlib (enforced by `tests/test_import_hygiene.py`). Everything
heavier — the legacy forecaster stack, RL policy members, batch walk-forward
CLIs, baselines — lives under `research/`, which imports `prism` and never
the reverse.

## Three paths through the engine

**The nightly live path** (`src/prism/live/`) runs the production spine every
trading session — this is the path that matters:

```
broker book ─► universe panels ─► signal node ─► construct ─► safety ─► broker ─► settle + ledgers
   (Alpaca IEX bars)
```

The path opens at the *broker*, not the universe file: the mark step values
what the venue holds, so `daily.resolve_fetch_universe` reads the account
before the fetch universe is decided and fetches configured ∪ held. Held names
outside the configured universe are valuation/exit-only — priceable, never
rankable — and may not be dropped by the missing-bar tolerance. Persisted state
is a reconciliation cache, never the authority for holdings
(`docs/operations.md`, postmortem 2026-07-23).

**The residual/stat-arb batch path** exercises the same spine offline:
`src/prism/residual/` (factor model, causal s-scores) plus the shared
construction and execution modules, driven by
`research/scripts/stat_arb_residual_wfo.py` — the momentum evidence path.

**The legacy directional path** (`research/scripts/training.py` →
`backtest.py`) runs the demoted per-symbol forecaster ensemble through the
same fold structure and accounting. It survives as a diagnostic surface, not
the production book.

## Modules

### Ingestion — `src/prism/io/`
`DataLoader` pulls split-adjusted daily bars from Twelve Data into a
range-keyed parquet cache and returns tz-aware (America/New_York) frames;
dividends are fetched separately and credited as cash in the backtest, never
back-adjusted into prices. Credential-bearing URLs pass through a redactor
before any log line (`docs/security.md` §2.4). `observatory.py` is the
append-only capture store for point-in-time membership and expectation data:
compressed JSONL with capture timestamps and verbatim payloads. Capture runs
unconditionally; modeling over captured data is separately gated
(`docs/v040_program.md` W5).

### Validation — `src/prism/validation/`
`PurgedWalkForward` yields train/test index pairs with purge and embargo; one
splitter drives both training and the meta-learner's out-of-fold predictions.
`metrics.py` implements the deflation-adjusted metrics and effective-breadth
diagnostics; `capacity.py` the capacity curve and cost-toll lens; `trials.py`
the canonical claim packet; `joint_crash.py` the uncounted B1-plus-trend
stress diagnostic.

### Signals — `src/prism/signal/`
The typed Signal contract (`SPEC.md` §7.1) and its nodes: `momentum_node.py`
(the production book's alpha since v0.3.2 — strictly trailing 12−1 momentum
feeding a decile long/short construct), `trend_node.py` (default-off ETF
trend), `ensemble_node.py` (the JAX-free XGBoost + ARIMA blend, one node
among several), and the residual-reversion node (archived sleeve; its
machinery remains as the live book's eligibility screen).

### Construction and execution — `src/prism/portfolio/`, `src/prism/execution/`
`portfolio/` builds books: caps, the single-step online no-trade band, and
the inverse-vol construct for the trend sleeve. `execution/target_weights.py`
is the accounting path: close-time targets fill at the next open, small
rebalances are suppressed by band, fold-last pending targets are dropped,
and spread, commission, impact, borrow, and dividend contributions are
charged on a fold-aligned equity curve. `spread.py` calibrates per-liquidity-
bucket effective spreads from the paper loop's fills ledger and refuses
under-sampled buckets; `edge.py` estimates spreads from bars alone
(a bracketing diagnostic, never a calibration authority); `participation.py`
hard-caps order size by ADV participation.

### Regime — `src/prism/regime/`
Curve, volatility, inflation, and net-liquidity state from free sources
(FRED, Treasury, CBOE), consumed as conditioning and de-gross triggers —
never a traded book.

### Live loop — `src/prism/live/`
`daily.py` runs one decide-at-close / fill-at-next-open cycle: settle the
prior decision first, then decide once. `loop.py` implements the write-ahead
protocol — reconcile to broker truth, decide, persist the pending decision
*before* the first submission; a restart resumes the persisted decision and
never re-decides. `state.py` is the atomic durable store; corrupt state
refuses to start rather than starting flat. `alpaca.py` submits idempotently
(per-book client order ids, next-open auction orders, whole shares enforced
loudly). `regime_step.py` writes regime telemetry each cycle; its de-gross
action hook stays unarmed until a dedicated arming commit.
`risk_profile.py` is the frozen operator surface
(`docs/risk_profile_schema.md`); profiles only tighten ratified pins.
`safety.py` (halt / size / exposure rails), `monitor.py` (rolling
deflation-adjusted metrics over the equity ledger), and `replay.py`
(diagnostic replay from local bars — never calibration evidence) complete
the loop. Unfilled auction orders are completed next morning by
`prism.scripts.paper_sweep`, and the fills ledger feeds spread calibration.

## Entry points

Production console scripts (`src/prism/scripts/`): `prism-doctor`
(preflight *and* operational health — ledger freshness, nightly verdict,
regime clock, and venue-book reconciliation; its exit code is the alert
condition), `prism-build-universe`, plus the module-run paper loop, sweep,
monitor, replay, and spread diagnostic. The scheduled wrappers that drive the
paper session and route alerts are versioned under `ops/`.

Research CLIs run as `python -m research.scripts.<name>` from the repo root —
training, backtest, sweep, the stat-arb walk-forwards, and a set of one-shot
diagnostics (data integrity, dividend wedge, breadth, cost frontier, tax
wedge, and others; each writes its receipt under `results/`). Flags and
outputs are documented by each script's `--help`.

## Invariants worth knowing before editing

- Bars are tz-aware (ET) from ingestion onward.
- The ensemble output is a position, not a price; forecast members are mapped
  to positions before blending.
- Directional results are portfolio-level by default: cite the root claim
  packet and cost/equity artifacts, not per-symbol logs.
- Arbitrage-style claims require cross-asset hedged target weights and
  portfolio-level PnL — independent per-symbol signals do not qualify.
- The full invariant set (causality, next-open fills, fail-loud, import
  hygiene, and the rest) is `SPEC.md` §1 and §6; cite invariants by tag.
