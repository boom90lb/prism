# Research tree inventory

`research/` holds the scripts that produced the project's recorded evidence, plus an
older per-symbol forecasting stack. It is not part of the `prism-trading` wheel. It
imports `prism`, `prism` never imports it (enforced by `tests/test_import_hygiene.py`),
and its scripts run from the repository root as `python -m research.scripts.<name>`.

This page lists every module in the tree, one row each, with:

- what the module does;
- a proposed status, **keep** or **archive candidate**;
- the tracked records it produced;
- the tests that import it;
- the third-party packages it needs beyond the core install.

The statuses are proposals. Nothing has been moved or deleted. Archiving is the owner's
decision, and the steps it would take are listed at the end.

**How this was checked (2026-10-15, tree as of `70e8332`).**

- *Tests that import a module* come from parsing every tracked `tests/*.py` and resolving
  its `import` and `from … import` statements and `importlib.import_module` /
  `pytest.importorskip` calls. A test that only names a module in a string or a
  docstring is not counted.
- *Records* are tracked files under `results/`, `trials/` and `data/universe/` that the
  module's own docstring, a `"script"` or `"command"` field in the record, or the
  document describing the record names as its source.
- *Packages needed at import time* were measured by importing each module in two
  environments. One has the core install plus the `construction` extra and the dev group
  (Python 3.12). The other has every extra (Python 3.14). In both, `PYTHON_DOTENV_DISABLED=1`
  was set; see the `research/config.py` row for why. *Packages needed when called* come
  from imports inside function bodies.

"Heavy" means a package that the core install does not provide. That is everything in
the `research` extra (prophet, scikit-learn, statsmodels, xgboost, tqdm, jax, flax,
distrax, optax, torch, transformers, gymnasium, mlflow, matplotlib) plus the
`construction` extra (cvxpy, clarabel). statsmodels and xgboost are also in the dev
group, so a plain `uv sync` installs them.

In the test columns, † marks a file whose tests are all skipped without the `research`
extra or carry the `research` marker, so CI's core job does not run them. Every other
listed file runs in the core job.

**Summary.**

- The 23 modules marked keep (21 scripts and two package files) need no package from
  the `research` extra. One of them, `solver_profile.py`, imports cvxpy and clarabel
  from the `construction` extra inside its functions.
- No keep module imports an archive candidate.
- Within `research/`, every package in the `research` extra is imported only by archive
  candidates. Archiving them would therefore leave that extra with no importer in this
  tree.
- `packages/prism-ensemble` is outside `research/` and is not covered here.

## Evidence and published-factor replication scripts (keep)

These produced the records that the project's results reports cite, or they run
evaluations that are scheduled but not yet due.

| Module | What it does | Status | Tracked records it produced | Tests that import it | Heavy dependencies |
|---|---|---|---|---|---|
| `research/scripts/factor_lab.py` | Published-factor lab: full-sample and post-publication Sharpe ratios of public long–short series, how closely in-house books track them, and four sizing rules tested on public momentum series (`registrations/2026-09-factor-lab.toml`) | keep | `results/2026-09-factor-lab/` (priors, fidelity, sizing, correlations, sources); `trials/sizing_lab_v1.jsonl` | `test_factor_lab.py`, `test_earnings_replication.py` | none |
| `research/scripts/vol_momentum_replication.py` | Replication test of volatility-scaled 12−1 momentum against the public series (`registrations/2026-09-vol-momentum.toml`) | keep | `results/2026-09-vol-momentum/` (`series.csv`, `summary.json`); `trials/vol_momentum_v1.jsonl` | `test_vol_momentum_replication.py`, `test_forward_read.py` | none |
| `research/scripts/fundamentals_replication.py` | Replication tests of six fundamentals factors against JKP series, and of their fixed-weight combination with volatility-scaled momentum (`registrations/2026-09-fundamentals*.toml`) | keep | `results/2026-09-fundamentals/replication/`; `trials/fundamentals_v1_combination.jsonl` | `test_fundamentals_replication.py` | none |
| `research/scripts/earnings_replication.py` | Replication test of the earnings-announcement return against the Chen–Zimmermann series, and of its combination with volatility-scaled momentum (`registrations/2026-09-earnings*.toml`) | keep | `results/2026-09-earnings/replication/`; `trials/earnings_v1_combination.jsonl` | `test_earnings_replication.py` | none |
| `research/scripts/earnings_read.py` | Stop-rule test for the two daily earnings strategies, and the rank IC of the three earnings scores at 1–63 session horizons | keep | `results/2026-09-earnings-daily/earnings_read.json` | `test_earnings_read.py` | none |
| `research/scripts/registered_read.py` | Evaluates a pre-registered set of runs from their own records: paired Sharpe differences, deflated Sharpe ratio against the set, effective number of independent bets, spanning regressions | keep | `read.json`, `comparison.json`, `comparison.txt` and `returns/` under `results/2026-09-{breadth,buffer-daily,buffer-monthly,characteristics,earnings,earnings-daily,fundamentals,learned-xsection}/` | `test_registered_read.py` | none |
| `research/scripts/buffer_read.py` | Evaluates the rank buffer (enter at the 10% tail, hold to 20%) against each unbuffered counterpart | keep | `results/2026-09-buffer/buffer_read.json` | `test_buffer_read.py` | none |
| `research/scripts/momentum_fragility_read.py` | Stop-rule test of the momentum robustness runs M1–M5 against the baseline momentum backtest | keep | `results/momentum_m_series_read_2026-09-03.json` | none | none |
| `research/scripts/forward_read.py` | Out-of-sample evaluations of every pre-registered forward test, on the schedule their registrations fix; the earliest is 2027-03-29, and `status` prints the schedule without reading any data | keep | none yet (writes under `results/forward/` once an evaluation is due) | `test_forward_read.py` | none |
| `research/scripts/build_breadth_dataset.py` | Builds the S&P 500 plus S&P 400 dataset used by the breadth evaluation | keep | `data/universe/sp400_exclusive_membership_2026-09-23.parquet`, `data/universe/sp900_membership_2026-09-23.parquet` (the dataset itself, `data/frozen/breadth_v1/`, is local and untracked) | none | none |
| `research/scripts/solver_profile.py` | Measures solve time and memory of the convex portfolio constructor at 500 to 3,000 names | keep | `results/solver_profile_2026-09-08.json` | `test_solver_profile.py` | when called: cvxpy, clarabel |

## Diagnostics with tracked records (keep)

| Module | What it does | Status | Tracked records it produced | Tests that import it | Heavy dependencies |
|---|---|---|---|---|---|
| `research/scripts/account_size_floor.py` | Whole-share rounding of the decile book as a function of account size, measured on replay runs | keep | `results/account_size_floor_2026-07-18.json`, `results/account_size_floor_day_2026-07-18.json` ([write-up](account_size_floor.md)) | `test_account_size_floor.py` | none |
| `research/scripts/bar_vendor_divergence.py` | Close-price differences between Alpaca IEX bars and the backtest's bar caches, and their effect on decile membership and marks | keep | `results/bar_vendor_divergence_2026-07-19.json` ([write-up](bar_vendor_divergence.md)) | `test_bar_vendor_divergence.py` | none |
| `research/scripts/beta_telemetry.py` | Full-sample, rolling and conditional market beta of the baseline momentum backtest's daily returns | keep | `results/beta_telemetry_2026-07-19.json` ([write-up](beta_telemetry.md)) | `test_beta_telemetry.py` | none |
| `research/scripts/carry_flatten_diagnostic.py` | What the backtest's flattening at each fold end costs, against carrying the book across folds | keep | `results/carry_flatten_diagnostic_2026-07-17.json` ([write-up](carry_flatten_diagnostic.md)) | `test_carry_flatten_diagnostic.py` | none |
| `research/scripts/cost_frontier.py` | Net performance as a function of the one-way spread, recomputed from stored backtest artifacts | keep | `results/cost_frontier_2026-07-19.json` ([write-up](cost_frontier.md)) | `test_cost_frontier.py` | none |
| `research/scripts/data_integrity_sweep.py` | Vendor symbol collisions, duplicate dates and other defects across the bar caches, checked against what each backtest held | keep | `results/data_integrity_sweep_2026-07-13.json`, `results/data_integrity_sweep_2026-07-14.json` ([write-up](data_integrity_diagnostic.md)) | `test_data_integrity_sweep.py` | none |
| `research/scripts/dividend_wedge.py` | Dividend difference between a price-return backtest and the same book on a total-return basis, from Alpaca corporate actions | keep | `results/dividend_wedge_2026-07-13.json` ([write-up](program_review_2026-07.md)) | `test_dividend_wedge.py` | none |
| `research/scripts/iex_eligibility_check.py` | How the $1M median-dollar-volume screen changes when computed on IEX-feed volume instead of consolidated volume | keep | `results/iex_eligibility_2026-07-17.json` ([write-up](program_review_2026-07.md)) | `test_iex_eligibility_check.py` | none |
| `research/scripts/spin_adjustment_sweep.py` | Checks that spin-offs and splits are adjusted consistently across the backtest's price panel | keep | `results/spin_adjustment_sweep_2026-07-19.json` ([write-up](spin_adjustment_sweep.md)) | `test_spin_adjustment_sweep.py` | none |
| `research/scripts/tax_wedge.py` | Tax asymmetries measured over the recorded ledgers: wash-sale deferral, payments in lieu of dividends on the short leg, the capital-loss netting cap, and state tax | keep | `results/tax_wedge_2026-07-19.json` ([write-up](tax_wedge.md)) | `test_tax_wedge.py` | none |

## Light modules with no tracked record (archive candidates)

| Module | What it does | Status | Tracked records it produced | Tests that import it | Heavy dependencies |
|---|---|---|---|---|---|
| `research/scripts/joint_crash_receipt.py` | Momentum alone against momentum plus ETF trend over fixed stress windows. Its output is gitignored scratch ([write-up](joint_crash_diagnostic.md)). The trend strategy it compares was refused admission on 2026-09-03 ([results report](trend_t_series_read_2026-09-03.md)) | archive candidate | none | none (`tests/test_joint_crash.py` tests `prism.validation.joint_crash`, not this script) | none |
| `research/observatory.py` | Append-only store for captured news-count and filing-count observations | archive candidate | none (captures are local) | `test_observatory.py` | none |
| `research/scripts/observatory_capture.py` | Captures Polygon news counts and SEC EDGAR filing counts into that store. No scheduled job or other module uses it. A day that is not captured cannot be recovered later, so archive it only if nobody is capturing | archive candidate | none | `test_observatory_capture.py` | none |

## Legacy per-symbol forecasting, reinforcement-learning and sentiment stack (archive candidates)

None of these has a tracked record. The [research archive](research_archive.md) found no
complete numerical run bundle for this line of work, and the [2026-06 audit](audit.md)
describes its limits, including path memorization in the xLSTM-GRPO member. Several
core-job tests import these modules, so archiving them means moving or retiring those
tests too (see the last section).

| Module | What it does | Status | Tracked records it produced | Tests that import it | Heavy dependencies |
|---|---|---|---|---|---|
| `research/config.py` | Settings for the legacy stack: ensemble, member and training dataclasses, artifact directories, the MLflow URI, the Polygon key. On import it creates `<repo>/models/` and calls `dotenv.load_dotenv()`, which loads the first `.env` found walking up from `research/` and ignores `PRISM_DISABLE_DOTENV` | archive candidate | none | `test_conformal.py`, `test_wfo_pipeline.py`, `test_real_pipeline_regressions.py`, `test_rl_seed_eval.py`† | none |
| `research/features.py` | Per-symbol feature engineering, with scaling and outlier clipping fitted on training data only | archive candidate | none | `test_feature_engineer_leakage.py`, `test_wfo_pipeline.py`, `test_real_pipeline_regressions.py`, `test_sweep.py`†, `test_script_claim_packet_smokes.py`† | none |
| `research/trading.py` | Per-symbol order-based backtester that trades the ensemble's or a baseline's positions | archive candidate | none | `test_baselines.py`, `test_dividends.py`, `test_execution.py`, `test_wfo_pipeline.py`, `test_real_pipeline_regressions.py`, `test_rl_seed_eval.py`† | when called: matplotlib |
| `research/execution_model.py` | Order queue, next-bar fill timing and per-fill costs for `trading.py` (moved out of `src/prism/execution/` in the engine overhaul) | archive candidate | none | `test_execution.py`, `test_wfo_pipeline.py`, `test_rl_seed_eval.py`† | none |
| `research/sentiment_analysis.py` | News sentiment (Polygon news, FinBERT) joined to bars point in time | archive candidate | none | `test_sentiment_leakage.py`, `test_transformer_sentiment.py`† | when called: torch, transformers |
| `research/models/base.py` | Base class for the forecast models | archive candidate | none | `test_dividends.py`, `test_execution.py`, `test_wfo_pipeline.py`, `test_rl_seed_eval.py`†, `test_script_claim_packet_smokes.py`† | when called: scikit-learn |
| `research/models/ensemble.py` | Ensemble of forecast and reinforcement-learning members; emits positions in [−1, 1] with conformal bands | archive candidate | none | `test_conformal.py`, `test_wfo_pipeline.py`, `test_real_pipeline_regressions.py`, `test_rl_seed_eval.py`† | when called: scikit-learn, plus each member's packages as it is loaded |
| `research/models/arima.py` | ARIMA forecast member | archive candidate | none | none | at import: statsmodels |
| `research/models/prophet.py` | Prophet forecast member | archive candidate | none | `test_real_pipeline_regressions.py` | at import: prophet (which loads matplotlib and tqdm) |
| `research/models/xgboost_model.py` | XGBoost forecast member | archive candidate | none | none | at import: xgboost (which loads scikit-learn when it is installed) |
| `research/models/lstm_ppo.py` | LSTM policy trained with PPO (JAX/Flax) | archive candidate | none | `test_rl_seed_eval.py`† | at import: jax, flax, distrax, optax, gymnasium; when called: tqdm |
| `research/models/xlstm_ppo.py` | xLSTM policy trained with PPO | archive candidate | none | none | at import: jax, flax, distrax, optax, gymnasium; when called: tqdm |
| `research/models/xlstm_grpo.py` | xLSTM policy trained with GRPO | archive candidate | none | none | at import: jax, flax, distrax, optax, gymnasium; when called: tqdm |
| `research/models/mapping.py` | Maps forecasts to positions by inverse-volatility scaling | archive candidate | none | none | none |
| `research/models/registry.py` | Registry of model type names | archive candidate | none | none | none |
| `research/baselines/buy_and_hold.py` | Buy-and-hold baseline for comparison with the ensemble | archive candidate | none | none directly (through `research/baselines/__init__.py`) | none |
| `research/baselines/ma_crossover.py` | Moving-average crossover baseline | archive candidate | none | none directly (through `research/baselines/__init__.py`) | none |
| `research/baselines/tsmom.py` | Time-series momentum baseline. `docs/trend_design.md` cites it by line | archive candidate | none | none directly (through `research/baselines/__init__.py`) | none |
| `research/scripts/_cli_common.py` | Command-line flags and data loading shared by the four legacy scripts below | archive candidate | none | `test_script_claim_packet_smokes.py`† | none |
| `research/scripts/training.py` | Purged walk-forward training of the ensemble per symbol, logged to MLflow | archive candidate | none | `test_data_loader.py`, `test_universe.py`, `test_wfo_pipeline.py`, `test_real_pipeline_regressions.py`, `test_sweep.py`† | at import, optional: mlflow; when called: torch |
| `research/scripts/backtest.py` | Walk-forward backtest of trained ensembles and the baselines per symbol, with the probability of backtest overfitting across them | archive candidate | none | `test_wfo_pipeline.py`, `test_real_pipeline_regressions.py`, `test_script_claim_packet_smokes.py`† | at import, optional: mlflow; when called: matplotlib |
| `research/scripts/sweep.py` | Grid search over three ensemble settings, with a deflated Sharpe ratio over the grid | archive candidate | none | `test_sweep.py`† | at import: mlflow |
| `research/scripts/rl_seed_eval.py` | Multi-seed study of how much the reinforcement-learning members' Sharpe ratios vary with the random seed | archive candidate | none | `test_rl_seed_eval.py`†, `test_script_claim_packet_smokes.py`† | at import: mlflow; when called: the JAX packages through the ensemble |
| `research/tracking/mlflow_utils.py` | Thin MLflow wrappers for the training loop | archive candidate | none | `test_wfo_pipeline.py`, `test_sweep.py`† | at import, optional: mlflow |

"Optional" means the import is wrapped in `try`/`except ImportError`, so the module loads
without the package and fails only when a function that needs it runs.

## Modules with no remaining importer (archive candidates)

Every module that used these was removed in the engine overhaul.

| Module | What it does | Status | Tracked records it produced | Tests that import it | Heavy dependencies |
|---|---|---|---|---|---|
| `research/arbitrage/pairs.py` | Engle–Granger pair selection and spread signals on a formation window. Its importers (`research/arbitrage/walk_forward.py`, `research/scripts/stat_arb_wfo.py` and `tests/test_arbitrage.py`) were last present at commit `5fcac33869e9`. `docs/stat_arb_shrinkage_experiment.md` records an opt-in shrinkage variant of its selection, whose first run was negative after deflation | archive candidate | none | none | at import: statsmodels |
| `research/scripts/stat_arb_data.py` | Loads close and open price matrices for the statistical-arbitrage scripts. Its only importer, `research/scripts/stat_arb_wfo.py`, was last present at commit `5fcac33869e9` | archive candidate | none | none | none |

## Package files

| Module | What it does | Status | Tracked records it produced | Tests that import it | Heavy dependencies |
|---|---|---|---|---|---|
| `research/__init__.py` | Marks the research tree as a package | keep | none | every test that imports a research module | none |
| `research/scripts/__init__.py` | Marks the scripts package. Its docstring still describes only the legacy scripts | keep | none | every test that imports a script | none |
| `research/arbitrage/__init__.py` | Marks the package. Its docstring still describes the removed walk-forward harness | archive candidate (with `pairs.py`) | none | none | none |
| `research/baselines/__init__.py` | Re-exports the three baselines | archive candidate | none | `test_baselines.py`, `test_wfo_pipeline.py` | none |
| `research/models/__init__.py` | Marks the models package | archive candidate | none | through its submodules | none |
| `research/tracking/__init__.py` | Re-exports `mlflow_utils` | archive candidate | none | through `mlflow_utils.py` | at import, optional: mlflow |

## Removed modules that docstrings and documents still cite

Eight research files were deleted by commit `52281e1` (2026-09-15, the engine overhaul).
Its parent, `5fcac33869e9`, is on `main` and holds all eight. Read one with
`git show 5fcac33869e9:<path>`. Citations of these files in `src/prism` say
"(removed; last present at commit 5fcac33869e9)", and the line numbers they cite refer
to that commit.

- `research/arbitrage/walk_forward.py`: shared walk-forward core (fold geometry, fold-end flattening, test-row slicing).
- `research/arbitrage/residual_walk_forward.py`: residual and momentum walk-forward core (spread buckets, no-trade band, decile rows).
- `research/arbitrage/trend_walk_forward.py`: trend walk-forward core (pinned cache files, distributions, total-return closes).
- `research/scripts/stat_arb_residual_wfo.py`: driver that produced the baseline momentum backtest, `results/demotion_b1/` ([research archive](research_archive.md), "Momentum B1").
- `research/scripts/trend_wfo.py`: driver that produced the trend record, `results/trend_t0/` ([research archive](research_archive.md), "Trend T0").
- `research/scripts/trend_adjudicate.py`: the trend set's stop rule and convexity test, read from finished runs.
- `research/scripts/breadth_diagnostic.py`: the earlier effective-breadth diagnostic. `research/scripts/registered_read.py` reuses its definition and cites it at `c08e3d0^`, a commit that only local tags reach.
- `research/scripts/stat_arb_wfo.py`: the pairs walk-forward driver.

The [research archive](research_archive.md) also names three annotated tags that
preserve these files: `pre-driver-retirement-stage2`, `pre-driver-retirement-stage3b` and
`pre-legacy-retirement-stage6`. Those tags exist only in the owner's local clone, and
`git ls-remote --tags origin` does not list them, so a public clone should use
`5fcac33869e9`. The copies differ in three places:

- `walk_forward.py`, `residual_walk_forward.py`, `trend_walk_forward.py`, `trend_wfo.py`
  and `trend_adjudicate.py` are byte-identical at `5fcac33869e9` and at
  `pre-driver-retirement-stage3b`.
- `stat_arb_residual_wfo.py` is absent from `pre-driver-retirement-stage3b`. Its copy at
  `pre-driver-retirement-stage2` has one extra import line (line 47), so every later
  line number there is one higher than at `5fcac33869e9`. The line numbers cited in
  `src/prism` match `5fcac33869e9`.
- `breadth_diagnostic.py` and `stat_arb_wfo.py` differ between `5fcac33869e9` and the tags.

## What archiving the candidates would take

1. **Tests.** These core-job test files import archive candidates and would have to move
   with them, be retired, or be split:
   - `test_baselines.py`, `test_dividends.py` and `test_execution.py` (the legacy
     backtester; `test_execution.py` also covers `prism.execution`);
   - `test_conformal.py` (also covers `prism.conformal`; only its ensemble tests depend
     on the legacy stack);
   - `test_feature_engineer_leakage.py` and `test_sentiment_leakage.py`;
   - `test_data_loader.py` (one test imports `clean_data_for_training`);
   - `test_universe.py` (imports `load_universe` and `filter_universe_asof` from
     `training.py`);
   - `test_wfo_pipeline.py`, `test_real_pipeline_regressions.py`, `test_observatory.py`
     and `test_observatory_capture.py`.

   The research-only files (`test_rl_seed_eval.py`, `test_script_claim_packet_smokes.py`,
   `test_sweep.py`, `test_transformer_sentiment.py`) would go with the modules.
2. **Dependencies.** The `research` extra in `pyproject.toml` would have no importer left
   in this tree. The dev group's statsmodels and xgboost stay, for
   `packages/prism-ensemble`'s tests.
3. **Documents.** These cite archive candidates by path: `ARCHITECTURE.md`, `README.md`,
   `docs/operations.md`, `docs/research_archive.md`, `docs/joint_crash_diagnostic.md`,
   `docs/stat_arb_shrinkage_experiment.md` and `docs/trend_design.md`. The last is a
   frozen design document, so its citation would be resolved by a pinned commit, as for
   the removed files above.
4. **A hazard to fix whether or not anything is archived.** `research/config.py` loads the
   nearest `.env` at import, and the core-job tests listed in its row import it. Running
   the suite in a checkout that holds broker credentials, or in a worktree nested inside
   one, therefore loads those credentials into the test process despite
   `PRISM_DISABLE_DOTENV=1`. Until that is fixed, set `PYTHON_DOTENV_DISABLED=1`, which
   python-dotenv honours, when running the suite in such a checkout. That variable also
   disables `prism.workspace.load_env`, so `tests/test_dotenv_guard.py`,
   `tests/test_workspace.py` and `tests/test_release_check.py` have to be run without it.

## Keeping this page in step with the tree

Every tracked `.py` file under `research/` has exactly one row above. This prints nothing
when that holds:

```bash
comm -3 <(git ls-files 'research/*.py' | sort) \
        <(grep -o '^| `research/[^`]*\.py` |' docs/research_tree.md | sed 's/^| `//; s/` |$//' | sort)
```
