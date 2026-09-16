# Prism

Prism is a cross-sectional systematic trading engine for US equities at a
daily-bar resolution: **score → residualize → construct → execute**,
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

## Status (2026-09-15)

- **Live paper book:** monthly 12−1 cross-sectional momentum on the S&P 500
  ([design](docs/momentum_design.md)), trading an Alpaca paper account
  nightly since 2026-07-13. Last completed session 2026-08-14; the scheduler
  has not fired since. Out-of-sample evidence: net annualized Sharpe 0.465,
  DSR 0.19 against the 17-trial set it came from — a real but thin premium
  that is signal-bound, not cost-bound, at monthly cadence.
- **Certified negative:** daily residual reversion on the S&P cross-section
  is uneconomic at retail cost
  ([cert 001](docs/certifications/001-residual-reversion-daily-negative.md)).
- **Completed registered reads:** momentum M1–M5 survived their kill rule
  ([read](docs/momentum_m_series_read_2026-09-03.md)); trend T0–T4 did not
  earn portfolio admission ([read](docs/trend_t_series_read_2026-09-03.md)).
  M6 and T5 remain on their prospective clocks.
- **Engine 0.4.0:** distribution `prism-trading`, import `prism`. Private
  projects can use the installed engine, specification and dataset formats,
  evaluation APIs, and replay reports without a fork. Publication and the
  operating-contract migration remain pending. The
  [24 operating changes](docs/engine_overhaul_operating_changes.md) require
  owner review before operational adoption; recorded decision parity is
  incomplete because the session bar panels are unavailable.
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

Start with the [four onboarding journeys](docs/onboarding.md): a first
evaluation, a private signal, interpreting a comparison, and the operating
workflow. They identify the commands and Python APIs available in this release,
including the CLI commands that still refuse unsupported paths. Use Python
≥ 3.12 and [uv](https://github.com/astral-sh/uv).

- [Depending on a release](docs/depending_on_prism.md): exact pins,
  lockfiles, fresh installation, artifact hashes, and dependency provenance.
- [Versions and upgrades](docs/versioning.md): compatibility, deprecation,
  and decision comparisons before adoption.
- [Contributing](docs/contributing.md): engine and reference-policy changes,
  tests, and the distinction between merging code and authorizing an account.
- [Public limitations](docs/limitations.md): coverage, deflation, null results,
  calibration shortfalls, rounding excesses, and missing decision parity.
- [Research archive](docs/research_archive.md): preserved tags, reproduction
  inputs, trial history, and retained legacy experiments.
- [Owner migration demonstration](docs/owner_migration.md): a private project
  pinned to a wheel, with the existing research history preserved.

For engine development:

```bash
git clone https://github.com/boom90lb/prism.git
cd prism
uv sync --extra construction   # core + dev and the optional solver tests
uv run --extra construction pytest -q -m "not research"
```

The optional `research` extra contains the archived heavy model stack and is
Linux-only. It is unnecessary for the onboarding examples. Existing registered
cases reproduce through the installed engine's evaluator; start with
`python -m prism.scripts.reproduce --help` and the archive's data notes.

The [paper quickstart](docs/quickstart.md) documents the earlier operating path.
Read the onboarding operating journey and [security rules](docs/security.md)
before using it: evaluation directories must be outside live credential
ancestry, and an existing account needs its owner's operating decisions.
[Operations](docs/operations.md) and the [free-tier profile](docs/free_tier_profile.md)
describe vendor and data constraints.

## Layout

```
src/prism/     the shipped package — production import path (JAX/torch-free, SPEC N8)
  io/          causal datasets, bars, events, sessions, security master, PIT universe
  signal/      the Signal contract and its nodes (momentum, trend, residual)
  residual/    factor model, causal s-scores, hedged book construction
  portfolio/   heuristic/convex construction, limits, obligations, rounding, recovery
  execution/   canonical portfolio accounting, simulation, costs, participation
  regime/      curve / vol / liquidity regime state from free sources
  live/        nightly loop: durable order state, broker adapter, safety rails
  validation/  purged WFO, metrics, capacity, claim packets, joint-crash diagnostic
  composition.py   component updates, score combination, position blending
  evaluate.py      purged walk-forward evaluation and result recording
  scripts/     prism-doctor, paper_loop / paper_sweep / paper_monitor, replay
ops/           scheduled paper-session wrappers + alerting (docs/operations.md)
research/      quarantined research tree (imports prism; never the reverse — SPEC N8)
packages/      optional strategy distributions, including prism-ensemble
examples/      reference specifications, profiles, policies, and cost stacks
trials/        append-only selection-set ledgers and imported history
formal/        Lean 4 machine-checked kernel invariants (see formal/README.md)
tests/         offline suite; the slim subset runs without the [research] extra
```

Configuration is split at the production/research boundary:
`src/prism/config.py` (cost dataclasses) and `research/config.py` (ensemble
members, training, MLflow). Both fail fast on invalid values at construction.
Writable directories come from `prism.workspace.resolve_workspace()`, not from
either config module: importing `prism` creates nothing and reads no `.env`.

## License

MIT License, Copyright (c) 2025 Brendon Reperttang. Nothing in this
repository is investment advice, and the project's own evidence bar has never
been cleared by any configuration.
