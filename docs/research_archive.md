# Research archive

This index preserves the source, dependency records, input identities, and reproduction
routes for the implementations retired or moved by the engine overhaul. It implements the
archive requirement in [the launch decisions, Q13 and Q20](prism_engine_decisions.md).
The [operating contract](../SPEC.md) still governs research and deployment; an archive
entry is not permission to reopen a closed selection set or change an operating book.

**Status: Run.** The tags, source paths, tracked artifacts, ledger row counts, and hashes
below were inspected locally on 2026-09-15. The historical reproduction reports are
recorded executions; this indexing pass did not rerun evaluations, training, captures,
or broker operations. The commands below are **Drafted** instructions, with input and
environment limits stated beside them.

## Source preservation and disposition

The named tags preserve source before retirement. Inspect any archived file with
`git show TAG:PATH`, and the annotation with `git cat-file -p refs/tags/TAG`. To execute
historical code, extract the tagged tree into a separate research directory with its own
environment and inputs. Do not restore retired modules over an operating checkout.

| Archive key | Annotated tag | Commit held by the tag |
|---|---|---|
| A2 | `pre-driver-retirement-stage2` | `29a6c679c3fb2eae8e9ef24107afd4a37b989c2a` |
| A3 | `pre-driver-retirement-stage3b` | `e33ab01cba526646bd1052ab515571c0883adbfe` |
| A6 | `pre-legacy-retirement-stage6` | `19054769993202dd59a7159c299931531d851afc` |

Each row below names its source tag and the dependency/input/reproduction entry that
applies to it. A moved module has a current home; an archived model remains in the
research tree. RL and forecast model source was preserved, not deleted.

| Source path at retirement | Archive | Current disposition | Reproduction and inputs |
|---|---|---|---|
| `research/scripts/stat_arb_residual_wfo.py` | A2 | Driver deleted after the B1 reproduction | Momentum B1 below |
| `research/scripts/trend_wfo.py` | A3 | Driver deleted after the T0 reproduction | Trend T0 below |
| `research/arbitrage/walk_forward.py` | A3 | Retired shared WFO core | Trend T0 below; archived source retains the shared geometry |
| `research/arbitrage/residual_walk_forward.py` | A3 | Retired shared residual/momentum core | B1 and T0 below; residual certification remains closed |
| `research/arbitrage/trend_walk_forward.py` | A3 | Retired trend core | Trend T0 below |
| `src/prism/io/observatory.py` | A6 | Moved to [`research/observatory.py`](../research/observatory.py) | Observatory below |
| `src/prism/execution/execution_model.py` | A6 | Moved to [`research/execution_model.py`](../research/execution_model.py) | Legacy fill engine below |
| `src/prism/signal/ensemble_node.py` | A6 | Moved to [`prism_ensemble/ensemble_node.py`](../packages/prism-ensemble/prism_ensemble/ensemble_node.py) | Optional ensemble below |
| `research/models/arima.py` | A6 | Retained in `research/models/` | Forecast/RL archive below; statsmodels member |
| `research/models/prophet.py` | A6 | Retained in `research/models/` | Forecast/RL archive below; Prophet member |
| `research/models/lstm_ppo.py` | A6 | Retained in `research/models/` | Forecast/RL archive below; LSTM-PPO and its environment |
| `research/models/xlstm_ppo.py` | A6 | Retained in `research/models/` | Forecast/RL archive below; xLSTM-PPO |
| `research/models/xlstm_grpo.py` | A6 | Retained in `research/models/` | Forecast/RL archive below; xLSTM-GRPO |
| `research/scripts/rl_seed_eval.py` | A6 | Retained in `research/scripts/` | Forecast/RL archive below; consumes existing training folds |

The shared-core retirement was supported by representative B1 and T0 reproductions.
It does not claim that every historical residual configuration was rerun. The
[residual signal and factor implementations](../src/prism/residual/) remain in the engine.

## Dependency records

All three archive tags contain `uv.lock` with SHA-256
`8d7be1e58501807b5ceaf373d210059f03535218dd3a39a8c70e9a058f0f350e`.
Retrieve the exact file with `git show TAG:uv.lock`; retrieve the accompanying declarations
with `git show TAG:pyproject.toml`.

**Recorded limitation:** these archived locks still name the project `prism` 0.3.4 while
the archived source declares `prism-trading` 0.4.0. They preserve the dependency resolution
file that existed at retirement, but they do not establish a complete, matching snapshot
of each historical installed environment. The A6 annotation's description of this lock
as the resolution the code last ran under does not supply an installed-package inventory.
Do not silently refresh an archived lock and describe the result as the original runtime.
A reconstructed environment needs its own resolution and provenance record.

The original B1 claim packet identifies source commit
`a2f44781ff1d2f87e1746fc70060eb17ee9d2918`; its `uv.lock` remains recoverable and hashes to
`e1d3e9e47ed7c02a4db4713fa41ae7780c5aeef973d349ce43ed43305730dcb5`.
The original T0 packet identifies `56eb3bf83c8f3d2e55705ee264f8c95d9fafc039`; its lock has
the `8d7be1e…` hash above. These are source-associated locks, subject to the same limit
on proof of installed dependencies. T0's packet reports a dirty source state with no
tracked diff and eleven untracked files; its commit alone does not describe those files.

For current reproduction, use a separately installed release and the matching project
files; [depending on Prism](depending_on_prism.md) explains version pins, wheel hashes,
lockfiles, and release provenance. A lockfile supplies identity, not a dependency audit.

## Momentum B1: retired stat-arb driver, surviving evidence

The [B1 record](../results/demotion_b1/) contains its tracked configuration, summary,
folds, claim packet, and breadth diagnostic. The original source is the A2 driver above;
its replacement route is [`prism.scripts.reproduce`](../src/prism/scripts/reproduce.py),
using [`examples/momentum/demotion_b1.toml`](../examples/momentum/demotion_b1.toml) and
the frozen [21-fold table](../folds/demotion_b1.toml).

Inputs are Twelve Data daily split-adjusted open/close prices, 2020-01-01 through the
configured 2026-06-16 cutoff, with no dividend cash credits. The recorded panel's last
session is 2026-06-15. The configuration names 574 symbols and the tracked point-in-time
membership, resolved-symbol list, and coverage artifacts dated 2026-06-16 under
`data/universe/`; membership identity is `4d2fef91f80f`. Coverage is 574 of 633
ever-members, 0.906793, with 59 missing names. The selection-related coverage bias is
disclosed and its size is not estimated.

The [recorded reproduction](../results/demotion_b1_reproduction/report.json) identifies
dataset `bedf8c4c0f0c2195ce5aace938f4ad38e23e6b69b57075080073004692d46eef`
and fold-table SHA-256
`fec02ca011e8c66309ab9e414403d1a6412a10569ff47bac775fa754ef3384f8`.
Its bucket arm's nine reported headline differences have maximum relative magnitude
`8.050956875898769e-16`. This records the stage-2 run, not a fresh result on the final
release. [`tests/test_demotion_b1_reproduction.py`](../tests/test_demotion_b1_reproduction.py)
separately checks the configuration, geometry, mechanics, and an available-data golden
slice with declared relative tolerance `1e-9`.

The original packet names the vendor and sample but has no complete hash inventory of
the original bar files. The reproduction's manifest supplies a later dataset identity;
it cannot retroactively prove the original vendor snapshot. Vendor caches and the
per-session output CSVs are local inputs/artifacts, excluded from the distributed
repository. Reproduction needs access to the matching licensed caches. A fresh vendor
download can contain revised adjustments and is not guaranteed to reproduce them.

The bucket arm reproduces the original cost convention; the separately declared flat
arm is reported without a historical comparison. Both are reproduction rows, not new
independent evidence. The [closed residual selection set](../results/stat_arb_residual_trials.jsonl)
retains the denominator behind B1's original selection-adjusted statistics.

## Trend T0: retired trend driver and shared WFO cores

The [T0 record](../results/trend_t0/) contains the tracked configuration, summary,
folds, claim packet, and adjudication. The replacement uses
[`examples/trend/trend_t0.toml`](../examples/trend/trend_t0.toml) and the frozen
[74-fold table](../folds/trend_t0.toml). A3 preserves all four retired driver/core paths.

The fixed ten-ETF list is SPY, EFA, EEM, TLT, IEF, LQD, HYG, GLD, PDBC, and UUP.
The recorded window is 2006-09-27 through 2026-07-17, 4,981 sessions and 4,662 test rows.
Scoring uses total returns; costs use split-adjusted traded prices; distributions are
credited as cash. Each fold prices its own formation-period liquidity buckets.

`config.json.cache_files` pins twenty vendor files, including the special SPY file
`SPY_1d_min_2026-08-14.parquet`. The packet's `data.cache_files` records individual
SHA-256 values and overlap handling. Its `data.distributions` records CSV and provenance
hashes for each ETF; those [distribution files](../data/distributions/) are tracked.
The vendor bar caches and recorded output CSVs remain local. Obtain the matching
licensed cache snapshots; changing an adjustment vintage changes the input identity.

The [recorded reproduction](../results/trend_t0_reproduction/report.json) identifies
dataset `b705acbfe8363749b3b92cf7d9dc4e904a3290f28698ba65998381040f19234b`
and fold-table SHA-256
`c17447526a06363f8edb33f9921b81e5d707769bad3ce84230d3acf276b0884c`.
All twelve headline differences in that report are zero. The golden checks in
[`tests/test_trend_t0_reproduction.py`](../tests/test_trend_t0_reproduction.py) declare
`1e-9` relative and `1e-12` absolute tolerances and use three folds to exercise changing
spread buckets.

The reproduced result remains negative as an admission decision: annualized net Sharpe
0.042, no standalone carry above the cash hurdle, and refused convexity admission.
See the [T-series read](trend_t_series_read_2026-09-03.md) and
[trend trial ledger](../results/trend_v1_trials.jsonl). A reproduction does not reopen
the decision or spend a fresh independent trial.

## Reproduction commands for B1 and T0

**Drafted.** Use a new, isolated research workspace with an installed `prism-trading`
release. Copy `examples/` and `folds/` from the corresponding release source into that
workspace: the runner resolves its specification, referenced policies, and fold paths
from the workspace root. Set `PRISM_ARCHIVE_SOURCE` to that source tree,
`PRISM_ARCHIVE_WORKSPACE` to the new workspace, and `PRISM_ARCHIVE_CACHE` to the directory
of matching licensed cache files. The source must include the original tracked
`results/`, `data/universe/`, and `data/distributions/` paths.

```bash
mkdir -p "$PRISM_ARCHIVE_WORKSPACE"
cp -R "$PRISM_ARCHIVE_SOURCE/examples" "$PRISM_ARCHIVE_SOURCE/folds" "$PRISM_ARCHIVE_WORKSPACE/"
python -m prism.scripts.reproduce --case demotion_b1 \
  --workspace "$PRISM_ARCHIVE_WORKSPACE" \
  --recorded "$PRISM_ARCHIVE_SOURCE/results/demotion_b1" \
  --cache-dir "$PRISM_ARCHIVE_CACHE" \
  --out "$PRISM_ARCHIVE_WORKSPACE/results/b1-reproduction.json"
python -m prism.scripts.reproduce --case trend_t0 \
  --workspace "$PRISM_ARCHIVE_WORKSPACE" \
  --recorded "$PRISM_ARCHIVE_SOURCE/results/trend_t0" \
  --cache-dir "$PRISM_ARCHIVE_CACHE" \
  --out "$PRISM_ARCHIVE_WORKSPACE/results/t0-reproduction.json"
```

These commands use `historical_constant_weight` accounting, which holds weights constant
between decisions without charging the implicit rebalance. They reproduce a historical
convention; a forward claim uses the canonical fixed-share simulator. The
[engine comparison script](../src/prism/scripts/engine_comparison.py) measures the distinction;
its module docstring records the 2026-09-08 run on `results/demotion_b1`. That run's JSON
output was never committed.
Outputs and new reproduction rows go to the isolated workspace, not the certified
record. Existing output directories are not a license to overwrite historical evidence.

The tracked reports contain absolute paths to the original local per-run records and
trial logs. Those files existed when this index was checked, but are untracked and not
available in a fresh clone. The tracked reports remain readable; full run-artifact
inspection requires the local archive or a newly identified reproduction.

## Stage-6 module moves

**Observatory (A6).** The store now lives in [`research/observatory.py`](../research/observatory.py),
used by [`research.scripts.observatory_capture`](../research/scripts/observatory_capture.py).
Its archived dependencies are described by A6's `pyproject.toml` and lock; the capture
driver directly uses `requests`. Inputs are Polygon reference-news responses and SEC
EDGAR search responses over the pinned 10-K/10-Q/8-K form set. Capture records carry
their observation time; there is no strategy result or trial claim to reproduce.
Polygon access requires an operator's own credentials, which are not archived. Raw
news payloads are not redistributed with this repository. Uncaptured historical days
cannot be regenerated as original observations by asking the API today. Offline
mechanics can be reproduced with:

```bash
python -m pytest -q tests/test_observatory.py tests/test_observatory_capture.py
```

**Legacy fill engine (A6).** The per-order implementation now lives in
[`research/execution_model.py`](../research/execution_model.py), consumed by
[`research/trading.py`](../research/trading.py) and
[`research/scripts/backtest.py`](../research/scripts/backtest.py). The A6 lock and
declarations cover the historical dependencies; a historical backtest additionally
needs the saved training run, fitted models, and licensed bars described below.
Deterministic local fixtures reproduce its fill/cost mechanics without those inputs:

```bash
python -m pytest -q tests/test_execution.py tests/test_wfo_pipeline.py -m "not research"
```

**Optional ensemble (A6).** The learned node is now the separate
[`prism-ensemble` package](../packages/prism-ensemble/pyproject.toml), version 0.4.0,
with `prism-trading>=0.4.0`, `statsmodels>=0.14.0`, and `xgboost>=1.7.0` declared as
dependencies. Inverse-OOF-MAE weighting and conformal-band combination primitives
remain in [`prism.composition`](../src/prism/composition.py). The optional node consumes
bar-derived features; model inputs from a historical forecast run follow the archive
entry below. A6 preserves the node before the move. In an isolated environment with
this optional package installed, the preserved fixture route is:

```bash
python -m pytest -q tests/test_signal_node.py tests/test_component_declarations.py
```

These are **Drafted** verification commands, not claims that a historical model fit or
external capture was reproduced by the archive-indexing pass.

## Forecast and reinforcement-learning archive

A6 preserves the five model files and the seed-evaluation driver listed above. Their
current source remains in `research/`; the [historical audit](audit.md) supplies the
limitations and provenance of that research direction. It is not a new result packet.
No immutable, complete numerical run bundle was identified for these retired research
lines in this indexing pass. Their source is recoverable; exact historical fits and
returns are not promised.

At A6, the research dependency declarations include JAX/Flax/Distrax/Optax, torch and
Gymnasium for RL; statsmodels for ARIMA; Prophet for the Prophet member; XGBoost for
the tree member; and scikit-learn, tqdm and MLflow for the surrounding research workflow.
The sentiment path uses transformers and `ProsusAI/finbert`. Consult A6's actual
declarations and lock for versions, subject to the lock limitation above. A reconstruction
needs Linux and the research dependency stack; it does not belong in the operating
environment.

Historical input dependencies are:

| Input | Identity retained | Availability limit |
|---|---|---|
| Price/volume bars | Vendor, symbol, timeframe and sample in the training configuration; Twelve Data cache/store paths or the declared Alpaca source | Licensed bars are not distributed; no complete frozen model-run bar inventory identified |
| Training folds | `runs/<run_name>/training_config.json`, saved per-symbol fold indices and transforms | Local, untracked; an absent original fold bundle cannot be replaced with a new split while claiming exact reproduction |
| Fitted models | Saved ensemble/member files under the training run and `models/` | Local, untracked; rerunning training makes new fit artifacts |
| MLflow provenance | Local `mlruns/` tracking store | Local, untracked; absent records leave run settings and history incomplete |
| Sentiment, when enabled | Polygon reference-news source and Hugging Face model id `ProsusAI/finbert` | News history and article payloads are external; no frozen model-weight revision or historical news bundle is identified here |

The A6 annotation records these historical driver recipes. They are **Drafted** recovery
templates, not an executed experiment or an exact run reconstruction. A training rerun
needs a registered research scope and must preserve all trial accounting; an archive
entry does not authorize a new search. Run only in an isolated research environment,
with the chosen historical source tree and a separately recorded environment resolution:

```bash
python -m research.scripts.training \
  --symbols AAPL,MSFT,GOOG,AMZN --timeframe 1d \
  --start_date 2020-01-01 --horizon 5 \
  --models arima,prophet,xgboost --n_splits 5 --experiment prism
python -m research.scripts.training \
  --models xgboost,lstm_ppo,xlstm_ppo,xlstm_grpo --rl_timesteps 100000
python -m research.scripts.rl_seed_eval \
  --training_run "$PRISM_ARCHIVE_TRAINING_RUN" \
  --members lstm_ppo,xlstm_ppo,xlstm_grpo --seeds 0,1,2,3,4
python -m research.scripts.backtest --training_run "$PRISM_ARCHIVE_TRAINING_RUN"
```

The training examples above have no frozen end date or saved fit identity; they explain
the available entry points. For exact reconstruction, use the original complete
training configuration and saved fold/model artifacts. The seed evaluator deliberately
consumes the original training run's folds instead of choosing another split.

## Research records and lineage

| Record | Where to start | Interpretation retained |
|---|---|---|
| Residual reversion and demotion | [Certification 001](certifications/001-residual-reversion-daily-negative.md), [demotion design](demotion_design.md), [17-row ledger](../results/stat_arb_residual_trials.jsonl) | Closed selection set; negative residual verdict; B1 is one of its searched trials |
| Momentum fragility | [Momentum design](momentum_design.md), [M-series read](momentum_m_series_read_2026-09-03.md), [6-row ledger](../results/momentum_v1_trials.jsonl) | Same-sample fragility read; prospective M6 remains on its registered clock |
| ETF trend | [Trend design](trend_design.md), [T-series read](trend_t_series_read_2026-09-03.md), [5-row ledger](../results/trend_v1_trials.jsonl) | Refused admission; prospective T5 remains on its registered clock |
| Universe construction | [4-row density ledger](../data/universe/sp500_density_trials.jsonl) | Provenance, not a strategy selection set |
| Construction experiment | [Registration](../registrations/2026-09-construction.toml), [comparison](../results/2026-09-construction/comparison.json), [limitations](limitations.md) | Three arms across six capital scenarios; no arm promotes |
| Engine behavior and operation | [Operating changes](engine_overhaul_operating_changes.md), [parity gate](engine_overhaul_parity_gate.json) | Reproduction evidence does not close missing decision parity or approve operating changes |

[`prism.history_import`](../src/prism/history_import.py) imports the four historical
ledgers without altering them. Strategy rows keep their original selection sets and
legacy digests, with `config_id = null` and `data_id = "unavailable"`; unavailable
artifacts remain explicitly unavailable. Imported rows count toward attempt history,
not independent new evidence. [`tests/test_history_import.py`](../tests/test_history_import.py)
checks source-byte preservation and the complete row mapping. The source row counts
above are the archive inventory, not a reset of any trial ledger.

Certified artifacts remain immutable. Old citations to removed driver paths resolve
through this index and the preserved tags. Their historical wording and code hashes
are retained rather than rewritten into a different claim.
