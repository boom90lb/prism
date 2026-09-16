# Owner research from an installed release

The owner migration check executes the registered B1 momentum bucket reproduction in a
private project, with `prism-trading==0.4.0` installed from a local wheel. The project keeps
its strategy specifications, fold tables, policy files and historical evidence outside the
engine. It does not maintain a copy of `src/prism`, an editable engine install, or a fork.
The distribution has not been published; the wheel is the release artifact used here.

The executable receipt is [owner_migration.json](owner_migration.json). The reusable procedure
is [ops/verify_owner_migration.py](../ops/verify_owner_migration.py). The private project, its
wheel, lockfile, dataset and full run outputs live outside the engine checkout; their exact
locations are recorded in the receipt.

## Recorded verification

**Tested, 2026-09-15:** the complete installed-package check exited 0. All 21 B1 folds
and 1,308 out-of-sample observations reproduced across nine headline metrics, with maximum
absolute difference `2.220446049250313e-16` against the predeclared `1e-9` tolerance.
Evaluation took 286.1 seconds. All 11 history files, all 579 frozen dataset files and all six
native ledger chains retained their original contents; no trial row was appended.

| Release identity | Verified value |
| --- | --- |
| Distribution and exact project pin | `prism-trading==0.4.0` |
| Wheel SHA-256 | `ad479c6a006ab28f91b722675e69fb38c95bd7eb0171e95184f9c492a85060fc` |
| Installed RECORD SHA-256 | `db11e3f95768a04e1cd6951114bb4cdaea2a21e2d7cd10838bc1c6ec3c443e5b` |
| Private lockfile SHA-256 | `919f7c8ff21b3cee5e01a0f4dd5eda73d933a9f114d060283f23816d89676787` |
| Run ID | `run-f4c613fb26574d41` |

The receipt records source base commit `f9a8a3a3a3d900ee8140d6e7690d58cef135b299`,
`source_dirty=true`, and the tracked-diff hash. That base commit alone does not name the
candidate source state. The wheel hash above identifies the exact artifact that was installed
and exercised, including the subsequent installed-identity and pandas compatibility fixes.

The output-location guard also passed three refusal tests: a destination equal to the frozen
dataset, a new child directory inside it, and a symlink resolving inside it. Each test checks
that the source and dataset retain their original bytes and that no output directory or
virtual environment is created. See [test_owner_migration.py](../tests/test_owner_migration.py).

## Reproduce the check

Use Python 3.12 or later, `uv`, the local release wheel, and an existing frozen B1 dataset.
The dataset is private vendor data and is not shipped with the wheel. The supplied output
path must be new and outside the engine checkout.

```sh
python ops/verify_owner_migration.py \
  --source /path/to/prism \
  --wheel /path/to/prism_trading-0.4.0-py3-none-any.whl \
  --dataset /path/to/frozen/demotion_b1 \
  --out /path/to/new-private-migration
```

The script copies the tracked history, specifications and policy inputs into
`private-project/`, and copies the frozen B1 dataset without rewriting the original. It
places the wheel in the private project's `vendor/` directory and writes:

```toml
[project]
name = "prism-owner-research"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["prism-trading==0.4.0"]

[tool.uv]
package = false

[tool.uv.sources]
prism-trading = { path = "vendor/prism_trading-0.4.0-py3-none-any.whl" }
```

`uv lock` records the local wheel and the resolved dependency versions; `uv sync --frozen`
installs into a new project-local environment. Dependency resolution may contact the package
index. Evaluation then uses that environment, with network connections and `.env` file opens
refused by a Python audit hook. The child environment inherits no broker/vendor credential
variables or Python import-path override. The evaluator calls no operating command or broker.

The relative wheel path and private lockfile travel with the project. Move the project with
its wheel and private dataset, recreate `.venv` with `uv sync --frozen`, and retain the receipts
and immutable history. A lockfile identifies installed code; it does not establish that a
dependency is safe. See [Depending on Prism](depending_on_prism.md).

## What the verification establishes

The script asserts the installed distribution's name, version, wheel direct URL and
noneditable status, and checks that `prism.__file__` resolves inside the private project's
`.venv`. It records the installed distribution's `RECORD` digest separately from the wheel
SHA-256, and asserts that the engine-reported identity equals that independently computed
installed `RECORD` digest even when the private project is inside another Git repository.
The source checkout's virtual environment is used only to launch the orchestration;
the evaluator imports the installed wheel from the fresh environment.

Every JSONL history file copied from tracked `trials/`, `results/` and `data/universe/` gets a
byte count, physical row count and SHA-256 before evaluation. The check rehashes both the
private copies and the originals afterward. All six native trial logs also pass hash-chain
verification, and their native attempt-count outputs remain identical. Historical registrations,
read records and certified artifacts are copied with their original paths and contents.

The 2026-09-15 snapshot contains 11 JSONL history files and 209 physical rows. That number is
not a count of independent trials: it includes raw legacy evidence, imported records, genesis
rows, started/completed events and reproductions. The raw ledgers retain the 17 residual-set
rows, six momentum-set rows and five trend-set rows. The imported registrations retain the
closed residual set and the momentum/trend clocks. Preserving native imported rows does not
replace the legacy independent counts used to deflate those historical claims.

The evaluation uses the existing `examples/momentum/demotion_b1.toml`, its recorded execution
cost parameters, the 21 frozen folds and the copied market dataset. It runs
`historical_constant_weight`, with reset portfolios and costs refreshed once, exactly as the
registered B1 bucket reproduction declares. Relative and absolute tolerance are both `1e-9`,
declared before evaluation. The nine headline metrics are compared with values read directly
from the preserved `results/demotion_b1/summary.json`; the observation count is checked against
the preserved fold records.

The run declares `kind="reproduction"` and `run_channel="engineering"`. It writes a started
and a terminal row to the private `runs/engineering.jsonl`, with no append to any trial ledger.
The original run artifacts remain untouched. The reproduction answers whether the installed
release executes the owner's existing configuration; it supplies no new independent evidence,
opens no selection set and authorizes no order.

## Scope and limits

This is the historical B1 reproduction engine, which holds weights constant between decision
bars and leaves that implied rebalancing uncharged. Agreement with the old record establishes
migration compatibility. It is not evidence for the canonical engine's execution performance
or a new alpha result. The underlying momentum record retains its own DSR, survivorship and
coverage limits; see [Limitations](limitations.md).

The private project carries the recorded history of other research sets, but this check executes
only the existing B1 bucket configuration. It does not restart the scheduler, run a paper
session, settle decision parity, alter the live profile, apply the operating-contract migration,
publish to PyPI or complete the pending owner operating decisions. The online observation
window remains outstanding.
