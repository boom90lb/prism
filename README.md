# Prism

Prism is a cross-sectional systematic trading engine for US equities at a
daily-to-weekly horizon: **score → residualize → construct → execute**,
conditioned by a regime layer, and gated by an evaluation harness built to
produce honest out-of-sample numbers. The harness — purged walk-forward
validation, next-open fills with realistic costs, deflation-adjusted metrics,
append-only trial ledgers, and a tiered claim vocabulary — is the point of the
project. Capital is risked only on results that clear an explicit evidence
bar, and no result has cleared it yet.

The governing documents, in order of authority:

| Document | Role |
|---|---|
| [`SPEC.md`](SPEC.md) | The constitution: invariants, component contracts, claim tiers, kill-criterion. Read first. |
| [`docs/v040_program.md`](docs/v040_program.md) | The v0.4.0 program: objective ranking, workstreams, deployment gates (RATIFIED 2026-07-22). |
| [`docs/handoff.md`](docs/handoff.md) | Standing doctrine: why the rules are what they are. |
| [`MARKETS.md`](MARKETS.md) | Market-structure analysis: which markets execute, which only supply signals. |
| [`AGENTS.md`](AGENTS.md) | Agent conduct and tooling. |

## Status (v0.3.4, 2026-07)

v0.3.4 is a **direction freeze, not a deployable product**. The law, the
program ranking, and most of the equity product surface are ratified; what
remains before an honest v0.4.0 is owner capital, operational reality, and
clocks — not more design documents.

**Closed:**

- The first alpha candidate — daily residual reversion — exhausted its
  pre-registered 17-trial budget without a positive deflated net Sharpe. The
  kill-criterion fired 2026-07-06 and the sleeve is archived; the negative
  verdict is the harness's first certification
  ([cert 001](docs/certifications/001-residual-reversion-daily-negative.md)).
- The live candidate is monthly cross-sectional momentum
  ([design](docs/momentum_design.md), ratified 2026-07-06), paper-trading
  nightly from 2026-07-13. Its promotion verdict is unreadable before the
  pre-registered window (≥ 2027-06 data). The stream is **interrupted**: the
  last completed session is 2026-07-17, and every scheduled cycle from
  2026-07-23 to 07-29 failed on an unvaluable held position. The defect is
  fixed and pinned; restarting the stream needs an owner run-directory
  decision ([operations](docs/operations.md), "Reattaching a run directory").
- Successor pre-registrations ratified: replication, trend, learned
  cross-section, and sizing with crash-conditional de-grossing (2026-07-18
  through 2026-07-20). The v0.4.0 program, its amendment set, and the operator
  risk-profile schema are ratified or frozen. Trend mechanics, regime
  telemetry, and the joint-crash diagnostic are built but uncounted.
- Test suite: 1008 passed, 1 skipped.

**Open — what actually blocks v0.4.0:**

- **Real fills.** The cost-calibration micro-account (amendment A3) is
  unfunded; per-bucket spread tables are still calibrated from paper fills,
  which the venue simulates. First gate: one full rebalance cycle of real
  fills (G0).
- **Capital mode.** Whole-share auction orders at ≥ $100k, or fractional
  day orders if both venue checks pass — neither selected yet.
- **The regime clock.** Deployment requires ≥ 21 consecutive clean paper
  sessions with regime telemetry recorded; the streak has not started. Four
  sessions ran 2026-07-13 → 07-17 with one fail-loud event on 07-15, then the
  nightly went dark 07-23 → 07-29. `prism-doctor` now reports the count off the
  regime ledger instead of anyone remembering it.
- **Operations.** Boot-resilient nightly scheduling and a one-page deploy
  runbook. The health signal itself now exists: the scheduled wrappers live in
  [`ops/`](ops/), each runs `prism-doctor` after its work, and either going
  nonzero raises an alert — a green morning sweep no longer implies a live book
  ([operations](docs/operations.md)).

**Nothing is authorized for real-money deployment today.** No sleeve has
cleared its evidence bar; the crash de-gross hook is wired but deliberately
unarmed. That statement is doctrine, not modesty: shipping "deploy-first"
without real fill data and a cleared bar would violate the project's own
rules.

## Method

Every methodological choice exists because the naive alternative makes a
backtest look better than the strategy is. The short version — details in
`SPEC.md` §6–§7 and the cited modules:

- **Purged, embargoed walk-forward validation**
  (`src/prism/validation/walk_forward.py`). Training rows whose label windows
  overlap a test slice are dropped; a buffer after each test slice is excluded
  from later folds. Training and backtest iterate the same fold structure.
  There is no 80/20 split anywhere.
- **Decide at close, fill at next open**
  (`src/prism/execution/target_weights.py`). Nothing fills same-bar, in
  backtest or live. Reported PnL is net of half-spread, impact, commission,
  and short borrow.
- **Overfitting-adjusted metrics** (`src/prism/validation/metrics.py`).
  Probabilistic and Deflated Sharpe Ratios, and the probability of backtest
  overfitting across the real selection set. When a grid was searched, the
  deflated number is the one to read — never the raw Sharpe.
- **Claim packets** (`src/prism/validation/trials.py`). Every result artifact
  records its config hash, code commit, data convention, trial count, and a
  claim tier (`mechanics_clean` → `gross_edge` → `net_edge` → `robust_edge`).
  No result is described above the tier its metrics support, and no capital
  moves below `net_edge`.
- **Dividends as cash, prices split-adjusted only.** The close is a faithful
  tradeable price; dividend credits make positions total-return correct
  without rewriting price history.
- **Survivorship is counted, not hidden.** The point-in-time universe is
  best-effort on included names and does not recover delisted tickers; every
  claim carries that caveat. The forward fix is prospective in-house
  accumulation ([evaluation](docs/data_purchase_evaluation.md)).

## Running it

To go from a clone to a nightly paper loop on a free Alpaca paper account,
see [`docs/quickstart.md`](docs/quickstart.md). Credential rules:
[`docs/security.md`](docs/security.md). What a $0 setup does and does not
reproduce: [`docs/free_tier_profile.md`](docs/free_tier_profile.md).

For the research side (Python ≥ 3.12, [uv](https://github.com/astral-sh/uv)):

```bash
git clone https://github.com/boom90lb/prism.git
cd prism
uv sync --extra research   # Linux only; bare `uv sync` elsewhere (core + dev)
uv run pytest -q -m "not research"
```

API keys go in a gitignored `.env` (Twelve Data for research bars, Alpaca and
FRED for the live loop and regime fetch; see `docs/security.md` §1 for the
inventory). Research entry points run as modules from the repo root:

```bash
# Train per-symbol purged-WFO models, then backtest the run
python -m research.scripts.training --symbols AAPL,MSFT,GOOG --start_date 2018-01-01
python -m research.scripts.backtest --training_run runs/{run_name}

# Hyperparameter sweep → trial Sharpes for honest deflation
python -m research.scripts.sweep --symbols AAPL,MSFT

# The B1 evidence path: momentum-sleeve walk-forward over a universe file
python -m research.scripts.stat_arb_residual_wfo --help
```

Each script's `--help` is authoritative for flags; the backtest writes a
`claim_packet.json` plus target/fill/cost/equity artifacts under `results/`,
and the packet is what to read first. Operational sharp edges (vendor tiers,
interval strings, which ensemble members actually contribute, per-bar compute
cost) are collected in [`docs/operations.md`](docs/operations.md).

## Layout

```
src/prism/     the shipped package — production import path (JAX/torch-free, SPEC N8)
  io/          bars + dividends loader, caching, point-in-time universe, capture store
  signal/      the Signal contract and its nodes (momentum, trend, ensemble, residual)
  residual/    factor model, causal s-scores, hedged book construction
  portfolio/   book construction: caps, no-trade bands, inverse-vol
  execution/   target-weight accounting, costs, participation gate, spread calibration
  regime/      curve / vol / liquidity regime state from free sources
  live/        nightly loop: durable order state, broker adapter, safety rails
  validation/  purged WFO, metrics, capacity, claim packets, joint-crash diagnostic
  conformal/   EnbPI + ACI prediction bands
  scripts/     prism-doctor, paper_loop / paper_sweep / paper_monitor, replay
ops/           scheduled paper-session wrappers + alerting (docs/operations.md)
research/      quarantined research tree (imports prism; never the reverse — SPEC §9)
formal/        Lean 4 machine-checked kernel invariants (see formal/README.md)
tests/         offline suite; the slim subset runs without the [research] extra
```

Configuration is split at the production/research boundary:
`src/prism/config.py` (directories, keys, cost dataclasses) and
`research/config.py` (ensemble members, training, MLflow). Both fail fast on
invalid values at construction.

## License

MIT License, Copyright (c) 2025 Brendon Reperttang. Nothing in this
repository is investment advice, and the project's own evidence bar has never
been cleared by any configuration.
