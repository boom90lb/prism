"""The project template `prism init` writes, and the offline CI job it ships.

The template is a contract with a user we cannot see: the files it writes are
the first thing a private project runs, so every TOML must parse, the example
component must satisfy the component contract, both specifications must load,
and the pin in the project's own ``pyproject.toml`` must name a distribution
that exists at the version the wheel carries.

The last test in the file runs the shipped ``ci.yml`` job for real, with every
outbound socket refused and every credential unset, because "offline CI" is a
claim about behaviour and not about a comment in a YAML file.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

import prism
from prism.signal.base import Signal, check_declarations
from prism.spec import load_spec
from prism.template import (
    ENGINE_DISTRIBUTION,
    ENGINE_PIN,
    TEMPLATE_DIRECTORIES,
    TEMPLATE_FILES,
    TemplateError,
    write_template,
)
from prism.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The paths section 3.14's `prism init` row promises, spelled as it spells them.
DECLARED_PATHS = (
    "pyproject.toml",
    "strategies",
    "specs",
    "profiles",
    "costs",
    "calibrations",
    "risk",
    "policies",
    "mappings",
    "registrations",
    "folds",
    "trials",
    "runs",
    "data/frozen",
    ".gitignore",
    ".env.example",
    ".github/workflows/ci.yml",
)

#: The stage 1 exit sequence of section 4, after `prism init`. The CI job runs
#: exactly these commands and no others.
EXIT_SEQUENCE = (
    "prism data synthetic --seed 7 --scenarios all",
    "prism conform specs/book.toml",
    "prism conform specs/etf_core.toml",
    "prism evaluate specs/book.toml --engine historical_constant_weight --smoke",
    "prism evaluate specs/etf_core.toml --engine historical_constant_weight --smoke",
    "prism run --spec specs/book.toml --account paper --once --broker synthetic",
)


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One written template, shared by the read-only checks."""
    root = tmp_path_factory.mktemp("template") / "my-strategy"
    write_template(root)
    return root


# --------------------------------------------------------------------------
# What init writes
# --------------------------------------------------------------------------


def test_init_writes_every_declared_path(project: Path) -> None:
    missing = [name for name in DECLARED_PATHS if not (project / name).exists()]
    assert not missing, f"`prism init` did not write {missing}"


def test_empty_input_directories_survive_a_clone(project: Path) -> None:
    # An empty directory does not exist in git; the four input directories a
    # later stage fills need a placeholder, and the two workspace-output ones
    # deliberately do not (they are ignored, and a placeholder there invites
    # someone to commit evidence rows).
    for name in ("calibrations", "risk", "registrations", "data/frozen"):
        assert (project / name / ".gitkeep").is_file()
    for name in ("trials", "runs"):
        assert (project / name).is_dir()
        assert not list((project / name).iterdir())


def test_gitignore_excludes_workspace_output_and_credentials(project: Path) -> None:
    ignored = (project / ".gitignore").read_text().splitlines()
    for entry in ("trials/", "runs/", "results/", "state/", "logs/", ".env"):
        assert entry in ignored, f"the template .gitignore must exclude {entry}"
    assert "!data/frozen/" in ignored, "frozen score-panel records are project input, not output"


def test_env_example_names_no_value(project: Path) -> None:
    # A template that ships a real-looking credential is a template that gets
    # committed with one. Every assignment in the example is commented out.
    for line in (project / ".env.example").read_text().splitlines():
        assert not re.match(r"^\s*[A-Z_]+=\S", line), f"the example .env carries a value: {line!r}"


def test_init_refuses_an_existing_project_unless_forced(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    write_template(root)
    (root / "strategies" / "my_signal.py").write_text("# the operator's edits\n")
    with pytest.raises(TemplateError, match="already holds"):
        write_template(root)
    write_template(root, force=True)
    assert "TrailingReturnRank" in (root / "strategies" / "my_signal.py").read_text()


def test_init_leaves_unrelated_files_alone(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "notes.md").write_text("mine\n")
    write_template(root)
    assert (root / "notes.md").read_text() == "mine\n"


# --------------------------------------------------------------------------
# Every template TOML parses, and the two specifications load
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(n for n in TEMPLATE_FILES if n.endswith(".toml")))
def test_every_template_toml_parses(project: Path, name: str) -> None:
    tomllib.loads((project / name).read_text(encoding="utf-8"))


def test_project_pyproject_pins_the_engine(project: Path) -> None:
    table = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    assert table["project"]["dependencies"] == [ENGINE_PIN]
    assert table["project"]["requires-python"] == ">=3.12"


def test_both_specifications_load_without_importing_components(project: Path) -> None:
    workspace = Workspace(project)
    for name in ("book.toml", "etf_core.toml"):
        spec = load_spec(project / "specs" / name, workspace=workspace, instantiate=False)
        assert spec.strategy.combination == "position_space"
        assert spec.construction.kind == "heuristic"
        for key, path in spec.policy.paths().items():
            assert path.is_file(), f"{name} names {key} = {path}, which the template does not write"


def test_both_specifications_instantiate_and_have_distinct_identities(project: Path) -> None:
    workspace = Workspace(project).ensure()
    book = load_spec(project / "specs" / "book.toml", workspace=workspace)
    etf = load_spec(project / "specs" / "etf_core.toml", workspace=workspace)
    assert [c.component_id for c in book.components] == ["trailing"]
    assert book.components[0].spec.to_weights == "decile_long_short"
    assert etf.components[0].spec.to_weights == "top_k_equal_weight"
    # Two configurations of one component: the parameters differ, so the
    # identities must differ, or one record could be read as the other.
    assert book.config_id() != etf.config_id()


def test_the_costs_stack_declares_a_class_for_every_source(project: Path) -> None:
    costs = tomllib.loads((project / "costs" / "retail_v1.toml").read_text())
    assert set(costs["source"]) == {"commission", "spread", "impact", "borrow", "financing", "cash_income"}
    for name, block in costs["source"].items():
        assert block["class"] in ("assumption", "calibrated"), name
        assert block["reference"], f"{name} declares a class with no reference"
    assert costs["spread"]["bound"] == "inclusive_lower"
    assert costs["spread"]["unknown_liquidity"] == "widest_step"
    # The three uses of a dollar-volume statistic are separate declarations.
    assert set(costs["dollar_volume"]) == {"spread_bucket", "participation", "impact"}


def test_the_etf_profile_claims_no_capital_suitability(project: Path) -> None:
    profile = tomllib.loads((project / "profiles" / "etf_small.toml").read_text())
    assert profile["min_account_status"] == "undetermined"
    assert profile["long_only"] is True
    assert "illustrative" in profile["name"]


def test_the_paper_profile_carries_only_its_certified_limits(project: Path) -> None:
    # The stage 1 parity fixture. New limits here change what the certified
    # book replays as, so the file is pinned rather than merely parsed.
    profile = tomllib.loads((project / "profiles" / "paper.toml").read_text())
    limits = {entry["name"]: entry["value"] for entry in profile["limits"]}
    assert limits == {"max_gross": 1.0, "max_name_weight": 0.10}
    assert "cash_buffer" not in profile


def test_the_smoke_fold_table_lies_inside_the_fixture_span(project: Path) -> None:
    from prism.io import sessions

    folds = tomllib.loads((project / "folds" / "smoke.toml").read_text())
    covered_start, covered_end = sessions.coverage()
    assert len(folds["fold"]) == 2
    previous_test_end = None
    for fold in folds["fold"]:
        assert covered_start <= fold["formation_start"] <= fold["formation_end"] <= covered_end
        assert fold["formation_end"] < fold["test_start"] <= fold["test_end"] <= covered_end
        if previous_test_end is not None:
            assert fold["test_start"] > previous_test_end, "the two test windows must not overlap"
        previous_test_end = fold["test_end"]


# --------------------------------------------------------------------------
# The example component
# --------------------------------------------------------------------------


def _load_example(project: Path):
    spec = importlib.util.spec_from_file_location("template_example_signal", project / "strategies" / "my_signal.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TrailingReturnRank


def test_example_signal_imports_instantiates_and_declares(project: Path) -> None:
    cls = _load_example(project)
    component = cls(lookback_bars=126)
    assert isinstance(component, Signal)
    declarations = check_declarations(component, configured_id="trailing")
    assert declarations["output_kind"] == "rank_score"
    assert declarations["units"] == "rank"
    assert declarations["scoring_basis"] == "price"
    assert declarations["fitted_state"] == "none"
    assert declarations["price_scale_invariant"] is True
    assert declarations["required_history"] == 127


def test_example_signal_raises_on_a_short_panel_and_scores_nan_on_a_short_name(project: Path) -> None:
    import numpy as np
    import pandas as pd

    cls = _load_example(project)
    component = cls(lookback_bars=5)
    index = pd.bdate_range("2026-01-05", periods=6)
    close = pd.DataFrame({"a": np.linspace(10.0, 15.0, 6), "b": np.linspace(20.0, 25.0, 6)}, index=index)

    # Check 4 of section 3.3, both halves: a panel below required_history is a
    # refusal, a name below it inside a long enough panel is NaN.
    with pytest.raises(ValueError, match="needs 6 bars"):
        component.score(close.iloc[:5])
    scores = component.score(close)
    assert scores["a"] == pytest.approx(15.0 / 10.0 - 1.0)
    blanked = close.copy()
    blanked.loc[blanked.index[0], "b"] = float("nan")
    assert np.isnan(component.score(blanked)["b"])
    assert not np.isnan(component.score(blanked)["a"])


def test_example_signal_is_price_scale_invariant(project: Path) -> None:
    import numpy as np
    import pandas as pd

    cls = _load_example(project)
    component = cls(lookback_bars=5)
    index = pd.bdate_range("2026-01-05", periods=6)
    close = pd.DataFrame({"a": np.linspace(10.0, 15.0, 6), "b": np.linspace(20.0, 21.0, 6)}, index=index)
    np.testing.assert_allclose(component.score(close * 7.5).to_numpy(), component.score(close).to_numpy())


# --------------------------------------------------------------------------
# Packaging: the pin, the version, and the built wheel
# --------------------------------------------------------------------------


def _engine_pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_pin_the_version_and_the_distribution_agree() -> None:
    table = _engine_pyproject()["project"]
    assert table["name"] == ENGINE_DISTRIBUTION == "prism-trading"
    assert table["version"] == prism.__version__
    assert ENGINE_PIN == f"{table['name']}=={table['version']}"


def test_the_console_script_is_declared() -> None:
    scripts = _engine_pyproject()["project"]["scripts"]
    assert scripts["prism"] == "prism.cli:main"
    # The two older entry points stay as aliases through stage 6.
    assert scripts["prism-build-universe"] == "prism.scripts.build_sp500_universe:main"
    assert scripts["prism-doctor"] == "prism.scripts.doctor:main"


def test_the_dependency_audit_left_core_to_what_the_wheel_imports() -> None:
    project_table = _engine_pyproject()["project"]
    core = {re.split(r"[<>=!\[]", entry)[0] for entry in project_table["dependencies"]}
    # scipy is imported at module scope on the production path
    # (validation/metrics.py) and must be declared.
    assert "scipy" in core
    # Nothing under src/prism imports these four at module scope. At stage 1 two
    # left core entirely and two moved behind an [ensemble] extra; at stage 6 the
    # learned node moved to the optional prism-ensemble distribution and took its
    # two members with it, so the extra is gone and all four are out of core.
    assert core.isdisjoint({"scikit-learn", "tqdm", "xgboost", "statsmodels"})
    extras = project_table["optional-dependencies"]
    assert "ensemble" not in extras, "the [ensemble] extra retired at stage 6 with the node it served"
    assert {re.split(r"[<>=!\[]", e)[0] for e in extras["construction"]} == {"cvxpy", "clarabel"}
    assert {"scikit-learn", "tqdm"} <= {re.split(r"[<>=!\[]", e)[0] for e in extras["research"]}
    # xgboost and statsmodels are declared in exactly two places now, and neither
    # is the engine: the [research] extra, whose research/models/ tree imports
    # them, and the optional strategy package that carries the node.
    engine_declared = {
        name
        for group in extras.values()
        for name in (re.split(r"[<>=!\[]", e)[0] for e in group)
        if group is not extras["research"]
    }
    assert engine_declared.isdisjoint({"xgboost", "statsmodels"})
    strategy_package = tomllib.loads((REPO_ROOT / "packages" / "prism-ensemble" / "pyproject.toml").read_text())
    assert strategy_package["project"]["name"] == "prism-ensemble"
    assert {re.split(r"[<>=!\[]", e)[0] for e in strategy_package["project"]["dependencies"]} == {
        "prism-trading",
        "statsmodels",
        "xgboost",
    }


def _wheel_for_this_version(tmp_path: Path) -> Path:
    """A wheel built from this tree, preferring a fresh offline build."""
    expected = f"prism_trading-{prism.__version__}-py3-none-any.whl"
    if shutil.which("uv") is not None:
        out = tmp_path / "dist"
        built = subprocess.run(
            ["uv", "build", "--offline", "--wheel", "--out-dir", str(out)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if built.returncode == 0 and (out / expected).is_file():
            return out / expected
    existing = REPO_ROOT / "dist" / expected
    if existing.is_file():
        return existing
    pytest.skip(f"no {expected}; build one with `uv build` (an offline build needs a warm uv cache)")


def test_the_built_wheel_carries_the_renamed_distribution(tmp_path: Path) -> None:
    wheel = _wheel_for_this_version(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(f"prism_trading-{prism.__version__}.dist-info/METADATA").decode("utf-8")
        entry_points = archive.read(f"prism_trading-{prism.__version__}.dist-info/entry_points.txt").decode("utf-8")
        names = set(archive.namelist())
    assert "Name: prism-trading" in metadata
    assert f"Version: {prism.__version__}" in metadata
    assert ENGINE_PIN == f"prism-trading=={prism.__version__}"
    assert "prism = prism.cli:main" in entry_points
    # The import name is unchanged, and the template travels inside the wheel.
    assert "prism/__init__.py" in names
    assert "prism/template.py" in names
    assert "prism/cli.py" in names


# --------------------------------------------------------------------------
# The shipped CI job
# --------------------------------------------------------------------------


def _offline_step_commands(workflow: str) -> list[str]:
    """The command lines of the workflow's ``offline smoke`` step.

    Parsed by hand rather than with a YAML library: the engine does not depend
    on one, and the block this reads is a fixed-shape literal in a file this
    repository writes.
    """
    lines = workflow.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "- name: offline smoke")
    run_at = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    indent = len(lines[run_at]) - len(lines[run_at].lstrip()) + 2
    body: list[str] = []
    for line in lines[run_at + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) < indent:
            break
        if line.strip():
            body.append(line[indent:])
    return body


def test_ci_workflow_runs_the_exit_sequence_and_nothing_else(project: Path) -> None:
    workflow = (project / ".github" / "workflows" / "ci.yml").read_text()
    body = _offline_step_commands(workflow)
    engine_commands = tuple(line for line in body if line.startswith("prism "))
    assert engine_commands == EXIT_SEQUENCE
    # No live account and no credential anywhere in the job.
    assert "--broker synthetic" in workflow
    assert "--account live" not in workflow
    assert "secrets." not in workflow, "an offline job needs no repository secret"
    # The workspace is a throwaway directory outside the checkout, so the
    # project's own trial log cannot receive a CI row.
    assert "PRISM_WORKSPACE: ${{ runner.temp }}" in workflow


@pytest.mark.parametrize("shell", [shutil.which("bash") or "/bin/sh"])
def test_ci_workflow_runs_offline(tmp_path: Path, shell: str) -> None:
    """Run the shipped CI job body with every outbound socket refused.

    This is the stage 1 exit sequence, executed: `prism init`, the synthetic
    fixture, both conformance runs, both smoke evaluations, and one synthetic
    session. If any of them reached the network — a vendor call, a broker call,
    a package index — the socket shim below turns it into a failure.
    """
    project = tmp_path / "checkout"
    write_template(project)
    workspace = tmp_path / "runner-temp" / "prism-workspace"

    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "prism"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m prism "$@"\n')
    shim.chmod(0o755)

    blocker = tmp_path / "sitecustomize"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(
        "import socket\n"
        "def _refuse(*args, **kwargs):\n"
        "    raise OSError('outbound socket refused: this job is offline')\n"
        "socket.socket.connect = _refuse\n"
        "socket.socket.connect_ex = _refuse\n"
        "socket.create_connection = _refuse\n"
    )

    env = {k: v for k, v in os.environ.items() if not re.match(r"^(TWELVEDATA|ALPACA|AWS|GITHUB)_", k)}
    env.update(
        {
            "PATH": f"{shim_dir}:{env.get('PATH', '')}",
            "PYTHONPATH": os.pathsep.join([str(blocker), str(REPO_ROOT / "src"), str(REPO_ROOT)]),
            "PRISM_WORKSPACE": str(workspace),
            "HOME": str(tmp_path / "home"),
        }
    )
    (tmp_path / "home").mkdir()

    body = _offline_step_commands((project / ".github" / "workflows" / "ci.yml").read_text())
    script = "set -eu\n" + "\n".join(body) + "\n"
    trials_before = sorted((project / "trials").iterdir())

    completed = subprocess.run(
        [shell, "-c", script], cwd=project, env=env, capture_output=True, text=True, timeout=1800
    )
    assert (
        completed.returncode == 0
    ), f"the offline CI job failed:\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
    # It ran the whole sequence, and it ran it on the synthetic fixture.
    assert "0 failed check(s)" in completed.stdout
    assert "unexercised scenarios" in completed.stdout
    assert "synthetic broker" in completed.stdout
    # The checkout is untouched: no dataset, no results, and a trial log that
    # is byte-identical before and after (spec section 3.1).
    assert sorted((project / "trials").iterdir()) == trials_before
    assert not (project / "results").exists()
    assert not (project / "data" / "manifest.json").exists()
    assert (workspace / "data" / "manifest.json").is_file()
