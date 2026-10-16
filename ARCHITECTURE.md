# Architecture

What calls what, end to end. The README covers status and method;
[`docs/operations.md`](docs/operations.md) covers operational sharp edges;
[`SPEC.md`](SPEC.md) states the rules this wiring implements; the
[glossary](docs/glossary.md) explains the in-house terms that code comments and
records use.

The organizing abstraction is a cross-sectional engine, not a per-symbol
forecaster: score every name, remove the factor exposures, build one portfolio,
and execute it.

```
DATA → SIGNAL → RESIDUALIZE → CONSTRUCT → EXECUTE → LEDGER
                               (caps, costs,  (next-open fills,
                                breadth)       participation limit)
     conditioned throughout by a REGIME layer (curve · volatility · liquidity),
     judged by the EVALUATION harness (purged walk-forward · deflated
     statistics · evidence tiers).
```

The production/research boundary (SPEC §2, N8, and §7): everything importable
from `prism` is the production code path and is free of JAX, torch, prophet,
mlflow and matplotlib, which `tests/test_import_hygiene.py` enforces.
Everything heavier, including the legacy forecaster stack, the reinforcement
learning members, batch walk-forward drivers and baselines, lives under
`research/`, which imports `prism` and is never imported by it.

## Three paths through the engine

**The nightly paper loop** (`src/prism/live/`, driven by
`python -m prism.scripts.paper_loop`) runs the production code every trading
session. This is the path that trades:

```
broker book ─► universe panels ─► signal node ─► construct ─► safety ─► broker ─► settle + ledgers
   (Alpaca IEX bars)
```

The path opens at the broker, not the universe file. The mark step values what
the account holds, so `daily.resolve_fetch_universe` reads the account before
it decides which names to fetch, and fetches the configured universe plus every
held name. Held names outside the configured universe can be valued and exited
but are never ranked, and the missing-bar tolerance may not drop them. The
saved state is a cache to reconcile against, never the authority for holdings
(`docs/operations.md`, postmortem 2026-07-23).

**The specification-driven engine** is what the `prism` command and the Python
interfaces run. A specification (TOML, `prism.spec`) declares the universe, the
components and their schedules, how scores become weights, and the cost, risk
and execution policies; it resolves to a `config_id` that also covers the
engine version and the content of every policy file. `prism.io.dataset` reads
the workspace's dataset as of each decision date, so a component sees only what
was known then. `prism.conformance` checks components against the contract the
engine relies on (`prism conform`). `prism.composition` combines components in
position space, `prism.assembly` turns one decision bar into orders for both
the evaluator and the operating loop, and `prism.evaluate` runs the
walk-forward evaluation. Two accounting engines sit under it: the
constant-weight simulator (`historical_constant_weight`,
`execution/target_weights.py`), which reproduces the recorded results, and the
share-holding simulator (`canonical`, `execution/simulate.py`,
`portfolio_state.py`, `accounting.py`), in which weights drift with prices and
cash is a real balance. `walk_forward_evaluate` writes a result record and
hash-chained trial-log rows; `smoke_evaluate` writes nothing. The command line
runs the smoke path only; the [onboarding guide](docs/onboarding.md) shows the
rest from Python.

**The research drivers** under `research/scripts/` produced most recorded
results, each run as `python -m research.scripts.<name>` from the repository
root. The momentum reference record in `results/demotion_b1/` is now
reproduced through `prism.evaluate.walk_forward_evaluate` by
`src/prism/scripts/reproduce.py`; the driver that first produced it,
`research/scripts/stat_arb_residual_wfo.py`, was removed in the engine overhaul
(removed; last present at commit 5fcac33869e9). The legacy directional path
(`research/scripts/training.py` → `backtest.py`) runs the retired per-symbol
forecaster ensemble through the same fold structure and accounting. It survives
as a diagnostic, not as a trading book. [`docs/research_tree.md`](docs/research_tree.md)
lists every research module.

## Modules

### Ingestion — `src/prism/io/`
`DataLoader` downloads split-adjusted daily bars from Twelve Data into a
range-keyed parquet cache and returns time-zone-aware (America/New_York)
frames. Dividends are fetched separately and can be credited as cash by the
accounting path, never back-adjusted into prices; the momentum reference record
was run without them (erratum E3). URLs that carry a credential pass through a
redactor before any log line (`docs/security.md`, section 2, rule 4).
`dataset.py`, `manifest.py`, `security_master.py`, `local_files.py` and
`legacy_import.py` are the dataset layer behind `prism data`; `synthetic.py`
writes the credential-free test fixture; `sessions.py` is the NYSE session
table; `universe_sp500.py` reconstructs point-in-time index membership;
`edgar.py`, `fundamental_panel.py` and `earnings_events.py` build point-in-time
fundamentals and earnings dates from SEC filings. The append-only store for
captured point-in-time observations (compressed JSONL with capture timestamps
and verbatim payloads) moved out of this package in the engine overhaul and is
now `research/observatory.py`, beside its capture script. Capture runs
unconditionally; modeling over captured data needs its own registration
(SPEC §4).

### Validation — `src/prism/validation/`
`walk_forward.py` holds the purged, embargoed splitters (`PurgedWalkForward`,
`RollingWalkForward`) and the frozen `FoldTable`. `metrics.py` implements the deflated statistics and the breadth
diagnostics; `hurdle.py` the T-bill benchmark; `compare.py` the paired
comparison of recorded evaluations; `anytime.py` the anytime-valid monitor
statistics; `capacity.py` the capacity curve and cost lens; `trials.py` the
result record, the claim packet and the trial log; `joint_crash.py` the
uncounted stress diagnostic of the momentum book with the trend book.

### Signals — `src/prism/signal/`
The typed component contract (`base.py`, the `Signal` class) and its
implementations, one `*_node.py` each: `momentum_node.py` (the paper book's
signal: strictly trailing 12−1 momentum feeding a decile long/short book),
`trend_node.py` (default-off ETF trend following), `residual_node.py` (daily
residual reversion, found uneconomic; its machinery remains the live book's
eligibility screen), and the research nodes for residual momentum,
idiosyncratic volatility, learned ranking, fundamentals and earnings. The
XGBoost and ARIMA blend moved out in the engine overhaul to the optional
`prism-ensemble` distribution under `packages/prism-ensemble/`, which took
xgboost and statsmodels out of the engine's dependencies. The `ensemble` book
still dispatches when that package is installed.

### Residualization — `src/prism/residual/`
The factor model and causal residual scores that remove market and industry
exposure from a cross-section.

### Construction and execution — `src/prism/portfolio/`, `src/prism/execution/`
`portfolio/` builds books: caps, the single-step no-trade band, the
inverse-volatility construction used by trend following, the convex
constructor with its recovery path when a solve fails, whole-share rounding,
the limit definitions and an independent validation of every solved and rounded
book, and the obligations a book must reduce first. The legacy
per-order fill engine moved out in the engine overhaul with its only two users
and is now `research/execution_model.py`. In `execution/`,
`target_weights.py` is the constant-weight accounting path: close-time targets
fill at the next open, small rebalances are suppressed by the band, targets
still pending at the end of a fold are dropped, and spread, commission, impact,
borrow and dividend charges are applied on a fold-aligned equity curve. `simulate.py`, `portfolio_state.py` and `accounting.py` are the
share-holding engine; `cost_model.py` and `costs.py` hold the charging
arithmetic and the declared cost stacks. `spread.py` calibrates
per-liquidity-bucket effective spreads from the paper loop's fills ledger and
refuses buckets with too few fills; `edge.py` estimates spreads from bars alone
as a cross-check, never as a calibration; `participation.py` caps order size as
a share of average daily volume.

### Regime — `src/prism/regime/`
Yield curve, volatility, inflation and net-liquidity state from free sources
(FRED, the Treasury, Cboe), recorded as telemetry. A regime feature may enter
sizing or a signal only after a purged walk-forward test shows it adds
information (SPEC §3). It is never a traded book.

### Live loop — `src/prism/live/`
`daily.py` runs one decide-at-close, fill-at-next-open cycle: settle the prior
decision first, then decide once. `loop.py` implements the write-ahead
protocol: reconcile to what the broker holds, decide, and save the pending
decision before the first submission, so a restart resumes the saved decision
and never decides twice. `state.py` is the atomic durable store; corrupt state
refuses to start rather than starting flat. `alpaca.py` submits idempotently
(client order ids per book, next-open auction orders, whole shares enforced
loudly) and `alpaca_data.py` fetches IEX daily bars. `authorization.py`,
`lockfile.py` and `freshness.py` are the controls that refuse a session before
any broker request: the owner's authorization record, the per-account
execution lock, and the data, broker-state and clock freshness checks.
`regime_step.py` writes regime telemetry each cycle; its exposure-cut hook is
not armed. `risk_profile.py` is the operator's settings
(`docs/risk_profile_schema.md`), which may only tighten the project's fixed
limits. `safety.py` (halt, size and exposure rails), `monitor.py` (rolling
statistics over the equity ledger), `reconcile.py` (state against the broker),
`replay.py` (diagnostic replay from local bars and the upgrade comparison,
never calibration evidence) and `parity_gate.py` (decision parity across engine
versions) complete the loop. Unfilled auction orders are completed the next
morning by `prism.scripts.paper_sweep`, and the fills ledger feeds spread
calibration.

## Entry points

Console scripts: `prism` (the command surface: `init`, `data`, `conform`,
`evaluate`, `run`, `check`, `replay`, `upgrade-diff`; commands the release does
not run from the command line say so and name what does), `prism-doctor` (the
health check before and after each session: ledger freshness, the nightly
result, the regime clock and, with `--network`, the broker's positions against
saved state; its exit code is the alert condition) and `prism-build-universe`. The paper loop,
the morning sweep, the monitor, the replay driver and the reproduction run as
`python -m prism.scripts.<name>`. The scheduled wrappers that run the paper
session and route alerts are versioned under `ops/`.

Research drivers run as `python -m research.scripts.<name>` from the
repository root: training, backtest, sweeps, the published-factor lab and
replications, and one-off diagnostics (data integrity, dividend wedge, breadth,
cost frontier, tax wedge and others), each writing its record under `results/`.
Each script's `--help` documents its options and outputs.

## Invariants worth knowing before editing

- Bars are time-zone-aware (US Eastern) from ingestion onward.
- A signal emits a standardized score with its horizon, never a price, and
  components are combined in position space.
- Results are portfolio-level: cite the run's claim packet and its cost and
  equity artifacts, not per-symbol logs.
- An arbitrage-style claim needs hedged target weights across assets and
  portfolio-level profit and loss; independent per-symbol signals do not
  qualify.
- The full invariant set (no look-ahead, next-open fills, costs before edge,
  capital conservation, deflation, breadth accounting, failing loudly and
  import hygiene) is SPEC §2, N1–N8, and the [glossary](docs/glossary.md#the-invariants-n1n8)
  states each in one line. Cite them by tag.
