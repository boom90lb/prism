# Four journeys through Prism 0.4.0

Use a private project for strategy code and policy, and install the `prism-trading`
engine as a dependency. The Python import stays `prism`. These journeys cover a
first evaluation, a private signal, a comparison, and an operating review. The
[owner migration](owner_migration.md) demonstrates the corresponding move of an
existing research project and its trial history to an installed wheel.

The examples below use synthetic data or read existing artifacts. Their results
establish engineering behavior; they do not establish market edge or authorize
an account. Read [limitations](limitations.md) before interpreting the published
reference results and [versioning](versioning.md) before choosing an upgrade.

## 1. First evaluation from an installed wheel

Start with Python 3.12 or later and `uv`. Obtain the wheel and its hash from the
release check described in [depending on Prism](depending_on_prism.md). The
unpublished local artifact path is the supported installation route for this
release; this walkthrough does not assume a PyPI upload.

Create a new directory outside any trading checkout. Replace the wheel path
with the verified artifact you obtained:

```sh
PRISM_WHEEL=/absolute/path/to/prism_trading-0.4.0-py3-none-any.whl
mkdir prism-onboarding
cd prism-onboarding
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "$PRISM_WHEEL"
.venv/bin/prism --version
.venv/bin/prism init my-strategy
cd my-strategy
export PYTHON_DOTENV_DISABLED=1
../.venv/bin/prism data synthetic --seed 7 --scenarios all --equities 12 --etfs 2
../.venv/bin/prism conform specs/book.toml
../.venv/bin/prism conform specs/etf_core.toml
../.venv/bin/prism evaluate specs/book.toml --engine historical_constant_weight --smoke
../.venv/bin/prism evaluate specs/etf_core.toml --engine historical_constant_weight --smoke
```

Run these in an environment without broker credentials. The CLI normally searches
for `.env` upward from the current directory; a nested worktree can therefore
inherit the parent checkout's credentials. `PYTHON_DOTENV_DISABLED=1` disables
loading with the release's installed `python-dotenv`; it does not clear credentials
already exported in the shell. The synthetic commands require no credentials.

`prism init` writes a private package with an exact `prism-trading==0.4.0` pin,
example signal modules, two specifications, policies, costs, and a fold table.
Its default workspace is the project directory, so generated data, logs, results,
and state stay there. `--workspace` is a global option: place it before the command
when overriding the workspace. Lock your private project's dependencies before
using it for research; the local wheel source and exact pin procedure are in
[depending on Prism](depending_on_prism.md) and [owner migration](owner_migration.md).

Read the evaluation output in this order:

1. Confirm the specification, synthetic manifest identity, vintage, universe,
   and span. `book.toml` uses synthetic equities; `etf_core.toml` is an illustrative
   long-only ETF configuration whose minimum suitable account size is undetermined.
2. Read the holding-process label. `historical_constant_weight` holds weights
   constant between decisions and re-levers without charging for that re-levering.
   It exists to reproduce historical records. The canonical engine holds shares
   and lets weights drift, so its returns answer a different question.
3. Read the unexercised and unapplied scenario lists. Generating all scenarios
   does not mean a given universe and accounting engine exercised all of them.
4. Read the recording statement. A smoke evaluation writes no `ResultRecord`
   and no trial row. Its printed Sharpe is an engineering diagnostic.

**Status: Tested** in a fresh private directory using an installed 0.4.0 wheel,
seed 7, 12 equities, 2 ETFs, and fatal outbound-socket guards. The conformance checks and both smoke
commands completed; [the verification receipt](onboarding_verification.json)
records exact command arguments, outcomes, and the installed distribution identity.
This is a local-artifact check, not an index installation or a prospective result.

## 2. Add and evaluate a private signal

The template's `strategies/my_signal.py` implements `TrailingReturnRank`.
Keep engine code in the installed package and put your change in `strategies/`.
This deliberately simple example reverses the score to exercise a private import
and a distinct configuration identity. It is not a new research registration.

From the `my-strategy` directory above:

```sh
../.venv/bin/python - <<'PY'
from pathlib import Path

source = Path("strategies/my_signal.py").read_text()
source = source.replace("TrailingReturnRank", "PrivateReversalRank")
source = source.replace(
    "return close.iloc[-1] / close.iloc[-1 - self.lookback_bars] - 1.0",
    "return -(close.iloc[-1] / close.iloc[-1 - self.lookback_bars] - 1.0)",
)
Path("strategies/private_signal.py").write_text(source)
spec = Path("specs/book.toml").read_text()
spec = spec.replace('name = "trailing_rank"', 'name = "private_reversal_demo"')
spec = spec.replace(
    "strategies.my_signal:TrailingReturnRank",
    "strategies.private_signal:PrivateReversalRank",
)
Path("specs/private.toml").write_text(spec)
PY
../.venv/bin/prism conform specs/private.toml
../.venv/bin/prism evaluate specs/private.toml --engine historical_constant_weight --smoke
```

A component declares its output kind, units, scoring basis, required history,
fitted-state behavior, horizon, update cadence, and validity. A panel with too
little history must raise; an individual name without enough history inside a
sufficiently long panel must emit NaN. The specification selects the component
by `module:class`, passes its parameters, and declares how scores become weights.
The engine combines component positions inside construction.

`prism conform` checks the declared behavior on the available fixture, including
history handling, determinism, and sampled causality. Inspect each PASS, FAIL,
and SKIP. Conformance executes your Python in the engine process; use reviewed
code. It does not turn the component into a trading claim or prove causality on
all possible inputs.

**Status: Tested.** The private component conformed, completed its synthetic
smoke evaluation, and produced a different `config_id` from the unchanged template.
The verification asserted that the smoke journeys created neither trial rows
nor result records. The reference template remained available beside the private
specification.

For research, first commit a registration describing the selection set, sample,
universe, cost stack, search procedure, kill rule, and promotion rule under the
[operating contract](../SPEC.md). Then use the package's
`prism.evaluate.walk_forward_evaluate` with a frozen
`prism.validation.walk_forward.FoldTable`, the registration, and declared accounting
settings. It records started and terminal rows, train-only fits, fold geometry,
input identities, costs, and provenance. Synthetic datasets are forced onto the
engineering channel. The function and its focused examples are in
[`evaluate.py`](../src/prism/evaluate.py) and
[`test_walk_forward_evaluate.py`](../tests/test_walk_forward_evaluate.py).

**CLI boundary:** `prism evaluate` currently delivers the smoke path above.
Full recorded evaluation, registered evaluation, and `--engine canonical` remain
refused by the CLI even though their Python APIs exist. A command appearing in
`--help` is not evidence that its handler executes it. The owner migration shows
a released-package reproduction through the full Python evaluator.

## 3. Interpret a comparison and recompute one

The [construction comparison](../results/2026-09-construction/comparison.json)
and its [text rendering](../results/2026-09-construction/comparison.txt) are a
worked example: three registered arms across six capital scenarios. All eighteen
studentized Sharpe-difference intervals include zero, and the registration permits
no promotion from this same-sample read. The largest displayed return is not a
selection rule.

For any comparison, check these fields before ranking performance:

| Read | Meaning |
| --- | --- |
| `registration`, `runs`, and each arm's `run_id` | The actual evaluated selection set and source records. |
| Provenance flags | Whether fills were simulated, data synthetic, the channel engineering, or the sample prospective. Missing values remain unavailable. |
| `conditions_held_fixed` and `conditions_that_differ` | Data, vintage, visibility, accounting engine, cost refresh, and capital must be interpreted explicitly. Capital scenarios are separate comparison blocks. |
| Fold coverage and exclusions | Different samples cannot be treated as a paired comparison. |
| Net returns, cost decomposition, turnover, and repairs | Gross performance omits the costs and implementation behavior needed to judge the result. |
| Sharpe difference, interval, and method | Check frequency, block length, seed, resamples, and uncertainty assumptions. A Sharpe is not a Treasury yield. |
| `promotion_check` and the trial ledger | A comparison reads evidence; it neither creates an independent trial nor promotes a strategy. |

`prism compare` currently exits 2 with a deferred-handler message. The implemented
comparison API is `prism.validation.compare.compare_reevaluation`. Save the
following as `compare_records.py` in your private project. It reads two existing
post-overhaul `record.json` files and their adjacent `returns.csv` files, and
writes a diagnostic to the explicitly chosen path:

```python
import argparse
import json
from pathlib import Path

import pandas as pd
from prism.validation.compare import (
    ComparisonArm,
    SharpeBootstrapSpec,
    compare_reevaluation,
)

parser = argparse.ArgumentParser()
parser.add_argument("record_a", type=Path)
parser.add_argument("record_b", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()


def read_arm(record_path):
    record = json.loads(record_path.read_text())
    returns = pd.read_csv(record_path.parent / "returns.csv", index_col=0).iloc[:, 0]
    returns.index = pd.to_datetime(returns.index, utc=True)
    return ComparisonArm(record=record, net_returns=returns)


# Frozen parameters of the recorded construction experiment.
# Use the registered parameters of your own experiment when reading its arms.
method = SharpeBootstrapSpec(
    block_bars=42,
    n_resamples=2000,
    seed=20260908,
    decision_interval_bars=21,
    frequency="daily",
    confidence=0.95,
)
report = compare_reevaluation(
    [read_arm(args.record_a), read_arm(args.record_b)], method
)
payload = report.to_dict()
args.out.write_text(json.dumps(payload, indent=2) + "\n")
assert payload["is_evidence_row"] is False
assert payload["writes_trial_row"] is False
print(json.dumps(payload["blocks"], indent=2))
```

The committed JSON and text comparison reports linked above can be read from a
fresh clone. The four source `record.json` and `returns.csv` files used in this
recomputation are private, untracked research artifacts: they are not included
in the wheel or a fresh clone. Obtain authorized copies of those existing files
before running this example, or supply two complete, comparable records from your
own registered evaluation. The example cannot reconstruct missing returns from
the published summary.

To recompute the $10,000 decile-control versus convex-point comparison when those
artifacts are available, point to their containing results directory:

```sh
PRISM_RECORDS=/absolute/path/to/prism/results
../.venv/bin/python compare_records.py \
  "$PRISM_RECORDS/run-333d02e707af4c6f/record.json" \
  "$PRISM_RECORDS/run-14401bdc29584ed1/record.json" \
  --out comparison-two-arms.json
```

**Status: Tested.** The example recomputed the committed periodic Sharpe difference
`0.005804596637678579`, interval `[-0.03671615541323037, 0.04622124273219694]`,
and two-sided p `0.789` exactly from the recorded 1,056 return observations.
The interval spans zero. It establishes neither superiority nor equivalence.
This smaller report deliberately supplies no holdings or cost panels, so those
fields are unavailable; the complete recorded report includes them. Reading a
report does not evaluate a new configuration or append to a trial ledger.

Do not feed the preceding smoke runs to this API: they write no records, and
`compare_reevaluation` refuses engineering-channel records. The receipt checks
that refusal. For repricing, `compare_repricing` holds recorded fills fixed and
reports a counterfactual cost diagnostic. Changing a constructor's cost inputs
requires a new registered evaluation, because its decisions may change.

## 4. Review operations and assess an upgrade

An operator needs a pinned engine, versioned project inputs, durable state,
recorded decisions, independent missed-session checks, and an authorization bound
to the account and configuration. Read [operations](operations.md),
[operating differences](engine_overhaul_operating_changes.md), and the
[deployment gate](../SPEC.md) before adopting the overhaul for a book.
The current overhaul's decision-parity gate remains incomplete and the owner
review of the operating differences remains outstanding.

Inspect the supported command surface without starting a session:

```sh
../.venv/bin/prism run --help
../.venv/bin/prism check --help
../.venv/bin/prism upgrade-diff --help
```

`run` requires an explicit paper or live account. The default broker is synthetic.
The Alpaca path implements submission and sweep stages against previously prepared
intent; the full specification-driven decide stage and resident loop are still
refused. `prism reconcile` and `prism check --monitor` are also deferred CLI
handlers. Paper operation still requires the separately reviewed operating setup;
this walkthrough starts no broker session and changes no scheduler.

Exercise the independent missed-session check against an empty synthetic run:

```sh
mkdir runs/empty
../.venv/bin/prism check runs/empty --expect-session --asof 2026-09-15
```

**Expected exit: 1.** An empty run has decided no session. The report names the
failure and lists the data-source, broker-state, and clock checks it cannot
establish offline. Without `--expect-session`, warnings do not by themselves make
`check` fail, so always read the individual rows. In an approved operating setup,
invoke the expected-session check from a service independent of the trading
scheduler; a stopped scheduler cannot run its own missed-run alert.

An upgrade review compares an incumbent capture with a separately produced
candidate capture over the same sessions. It does not install another engine or
produce the candidate for you. This synthetic example makes the report's three
outcomes visible without running a broker:

```sh
../.venv/bin/python - <<'PY'
import json
from pathlib import Path

for name in ("baseline", "candidate"):
    path = Path("runs") / name
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({
        "source": {"traded_commit": "synthetic-onboarding"},
        "caveats": ["Synthetic capture; upgrade-report mechanics only."],
        "sessions": [{"bar": "2026-09-14", "kind": "decision"}],
    }))
    (path / "targets.jsonl").write_text(json.dumps({
        "decision_bar": "2026-09-14", "targets": {"DEMO": 0.1}, "equity": 1000.0,
    }) + "\n")
PY
../.venv/bin/prism upgrade-diff --sessions runs/baseline --json upgrade-missing.json
../.venv/bin/prism upgrade-diff --sessions runs/baseline --candidate runs/candidate --json upgrade-equal.json
```

The first command exits **2**, with 0/1 sessions compared. The second exits **0**,
with 1/1 target decisions compared and unchanged. This is an identical fabricated
capture comparison, not evidence about two engine versions; the captures contain
no order ledger, so `orders_compared` is false.

Now edit only the disposable candidate and use the equivalent replay command:

```sh
../.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("runs/candidate/targets.jsonl")
row = json.loads(path.read_text())
row["targets"]["DEMO"] = 0.2
path.write_text(json.dumps(row) + "\n")
PY
../.venv/bin/prism replay runs/baseline --upgrade-diff --candidate runs/candidate --json upgrade-changed.json
```

**Expected exit: 1.** One of one compared sessions has a changed target decision.
No live or research record was edited. **Status: Tested:** the three exit codes,
comparison counts, changed target, and empty-run refusal were asserted against
synthetic captures with outbound sockets disabled.

For an actual upgrade, keep the incumbent capture immutable, run the candidate
engine in a separate pinned environment on the full recorded input panel, and
compare the resulting captures. Read every changed decision and every session that could not be
compared before choosing the version. The shipped paper capture lacks the full
historical panels needed to produce its candidate; 0/5 compared sessions remain
missing evidence. See [versioning](versioning.md#3-the-upgrade-comparison-and-what-it-can-and-cannot-compare) for the
command and the bounds of the claim. Merging engine code does not authorize
live configuration changes, and an offline review does not complete the required
paper observation window.
