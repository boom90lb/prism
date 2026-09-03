# Prism

Prism is a cross-sectional systematic trading engine for US equities at a
daily-to-weekly horizon: **score → residualize → construct → execute**,
conditioned by a regime layer, and gated by an evaluation harness built to
produce honest out-of-sample numbers. The harness — purged walk-forward
validation, next-open fills with realistic costs, deflation-adjusted metrics,
append-only trial ledgers, and a tiered claim vocabulary — is the point of the
project. Capital is risked only on results that clear an explicit evidence
bar, and no result has cleared it yet.

[`SPEC.md`](SPEC.md) is the single operating contract: current state,
invariants, the trial-ledger rules, research direction, deployment gate, and
working rules. Read it first. [`ARCHITECTURE.md`](ARCHITECTURE.md) is the
call graph; [`MARKETS.md`](MARKETS.md) the market-structure analysis.

## Status (2026-09-02)

- **Live paper book:** monthly 12−1 cross-sectional momentum on the S&P 500
  ([design](docs/momentum_design.md)), trading an Alpaca paper account
  nightly since 2026-07-13. Last completed session 2026-08-14; the scheduler
  has not fired since. Out-of-sample evidence: net annualized Sharpe 0.465,
  DSR 0.19 against the 17-trial set it came from — a real but thin premium
  that is signal-bound, not cost-bound, at monthly cadence.
- **Certified negative:** daily residual reversion on the S&P cross-section
  is uneconomic at retail cost
  ([cert 001](docs/certifications/001-residual-reversion-daily-negative.md)).
- **Registered, unrun:** the momentum fragility reads M1–M5 and the trend
  sleeve reads T0–T4. They are the next research work.
- **Nothing is authorized for real money.** The gate is `SPEC.md` §6.

## Method

Every methodological choice exists because the naive alternative makes a
backtest look better than the strategy is. The short version — details in
`SPEC.md` §2–§4 and the cited modules:

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
research/      quarantined research tree (imports prism; never the reverse — SPEC N8)
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
