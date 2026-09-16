# Engine overhaul: stage 6 completion record

Date: 2026-09-15, Pacific time. This records the release engineering and
maintainability work completed before consolidating the overhaul into one
commit on `main`. Operational adoption remains subject to the owner decisions
and missing evidence below. This record does not migrate the operating contract.

## Completed release workflows

**Tested — four onboarding journeys.** [The walkthrough](onboarding.md) and
[its receipt](onboarding_verification.json) cover an installed-wheel evaluation,
a private signal, reading and recomputing a comparison, and an offline operating
review. Fourteen CLI-handler outcomes were asserted, including both template
evaluations, private conformance and evaluation, missed-session refusal, and
upgrade reports for missing, matching, and changed captures. Smoke runs wrote
neither result records nor trial rows. The comparison API reproduced the
committed two-arm result exactly and refused engineering-channel inputs.

**Tested — owner research from a pinned release.** [The migration guide](owner_migration.md),
[receipt](owner_migration.json), and [verification script](../ops/verify_owner_migration.py)
demonstrate a separate private project with `prism-trading==0.4.0`, a local wheel
source, and a private lockfile. The existing B1 bucket configuration reproduced
all 21 folds and 1,308 out-of-sample observations. Nine headline metrics matched
the preserved summary at the predeclared relative and absolute tolerance of
`1e-9`; the maximum absolute difference was `2.220446049250313e-16`.
All 11 JSONL histories, containing 209 physical rows, and all 579 frozen dataset
files were byte-identical before and after. Six native ledger chains and their
attempt counts were unchanged. No trial row was appended; the private run wrote
only two engineering reproduction rows. Physical row count is not independent
trial count.

Both workflows used this wheel:

```text
prism_trading-0.4.0-py3-none-any.whl
sha256:ad479c6a006ab28f91b722675e69fb38c95bd7eb0171e95184f9c492a85060fc
830393 bytes
Python 3.12.8; pandas 3.0.5; NumPy 2.5.3
```

The receipts identify the executed bytes separately from the dirty source base
used to build them. Engine source fixes are recorded by commit `a336d68`.
Each private installation independently checked its installed `RECORD` digest.
These digests describe installations and can differ between environments even
when the wheel is identical. Evaluation refused outbound connections and `.env`
opens or disabled dotenv loading; no broker session was started.

**Run — indexed research archive.** [The index](research_archive.md) maps 14
retired or moved source paths to the three preservation tags, dependency
records, data identities, reproduction commands, trial lineage and negative
results. It states where historical installed environments, original bar hashes
or private artifacts are incomplete. RL and forecaster source remains retained.

**Tested — release metadata and navigation.** `uv lock --check` accepts the
renamed distribution and its 209-package resolution. The
[release check](depending_on_prism.md#check-after-the-lock-refresh) passed all
six checks after the lock refresh. README now links the release, contribution,
limitations, archive, migration and onboarding guides; local links were checked.
Quickstart's two deployment-gate citations now name `SPEC.md` section 6.
The versioning guide describes the implemented upgrade-review commands.

## Defects exposed by the installed workflows

These fixes restore the intended release behavior; they introduce no new
reference strategy or research arm.

- A wheel installed beneath a private Git project now identifies its own
  installed `RECORD`, rather than the surrounding project's commit. Source and
  editable imports retain Git identity. Missing matching identity fails closed.
  [Identity tests](../tests/test_distribution_identity.py) cover changed wheel
  records, unchanged private-strategy edits, missing records, and unrelated
  installations; the existing authorization refusal suite also passes.
- Dividend placement now owns the array it accumulates into. The installed
  pandas 3 workflow had failed with `ValueError: assignment destination is
  read-only`; [the regression](../tests/test_price_basis.py) exercises
  copy-on-write and checks that the input price panel remains unchanged.
- A fresh T0 import accepts its recorded distributions directory while ordinary
  recorded inputs still require files. Existing frozen datasets had masked the
  failure. [The new tests](../tests/test_reproduce_input_paths.py) cover the
  uncached entry path, external inputs, missing inputs and wrong input kinds.
- CI installs the construction extra for solver tests and imports the shipped
  momentum and composition modules in its slim-wheel smoke. The previous smoke
  failed because it imported the retired in-engine ensemble module.
- The owner migration script refuses output inside its source dataset before
  creating files. [Three refusal tests](../tests/test_owner_migration.py) include
  a symlink alias and prove the original inputs remain unchanged.

The public rounding account was also corrected by reading only the 18 named
comparison runs: 111 of 918 decisions have 136 rounding excesses. One separate
heuristic excess appears twice in the JSON; traversing every nested violation
list double-counted it. [Limitations section 6](limitations.md#6-a-constructor-honouring-a-limit-is-not-the-portfolio-satisfying-it)
states the exact field and distinction. No recorded run or trial row was edited.

## Final source validation

```sh
.venv/bin/python -m pytest -q -m "not research" -p no:cacheprovider
```

**Tested:** 3223 passed, 7 skipped, 85 deselected, 52 warnings in 865.23s (0:14:25). Exit 0. The source suite ran with pandas 2.3.3;
the installed-wheel workflows above separately exercised pandas 3.0.5.
The selected item count is 3,226: four module-level skips are reported in the
seven-skip summary without becoming collected test items.

The non-research suite excludes 85 cases marked research. Its seven skips
comprise four research modules skipped during collection, the optional Prophet
runtime check, the unavailable range-encoded vendor-cache round trip, and the
full paper replay whose recorded input panels are absent. The historical
momentum and trend golden reproductions using the available frozen datasets
remain selected. The earlier handoff's description of all seven skips as vendor
cache skips was incorrect.

Full-suite validation applies to the engine source in `a336d68`; the subsequent
changes are documentation and receipts. Final wheel verification compares its
shipped Python files byte-for-byte with that source. No remote CI execution or
index-served package verification is claimed.

## Operating adoption remains separate

- No scheduler restart, order submission, PyPI publication, real-money account
  change, de-gross arming or live-profile change was performed.
- [The 24 intentional operating changes](engine_overhaul_operating_changes.md)
  await owner review. `SPEC.md` remains the pre-overhaul contract, including its
  previously recorded dangling retired-driver citation; its migration was not applied.
- [Decision parity](engine_overhaul_parity_gate.json) remains INCOMPLETE: 209
  recorded parent orders rebuild share-for-share, but 0 of 5 sessions have a
  comparable candidate decision because their full input panels are missing.
- Authorization, execution-host and freshness choices, `PRISM_RELEASE`, and
  the observation window remain outstanding owner operating work.
- The known upward dotenv lookup and the legacy heuristic's full-exit behavior
  remain limitations. The rounding-repair question remains open. The onboarding
  guide also names CLI handlers that still require Python APIs or remain deferred.

The original live run directory and live environment metadata were checked
against a pre-session snapshot. Existing result artifacts and trial histories
were preserved; design scratch remained untracked. The overhaul branch and
preservation tags retain historical commit references after the single-commit
consolidation on `main`.
