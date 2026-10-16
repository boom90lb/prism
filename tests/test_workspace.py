"""Workspace resolution and the import-purity invariant (spec §3.1).

The failure this file exists to catch: an engine that writes beside the
installed package, or that reads a credential the moment something imports it.
Before the correction, ``import prism.config`` created ``data/`` and
``results/`` next to the package, ran ``dotenv.load_dotenv()``, and captured
``TWELVEDATA_API_KEY`` into a module attribute -- so any process that so much
as touched a cost dataclass inherited the operator's keys and left directories
behind. ``test_import_closure_is_pure`` runs the whole package's import closure
in a subprocess against a *copied* package tree, with a planted ``.env`` in the
working directory, in the package's parent, and in its grandparent, and proves
none of the three happens.

The rest pins the resolution order (``--workspace``, ``PRISM_WORKSPACE``,
``[tool.prism].workspace``, cwd), the layout, and the callers that lost their
package-relative defaults: ``DataLoader``, ``configure_logging``, and the
doctor's data-dir check.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from prism.io.loader import DataLoader
from prism.logging_utils import configure_logging
from prism.workspace import SUBDIRECTORIES, Workspace, load_env, resolve_workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "src" / "prism"

_SENTINEL_VAR = "PRISM_DOTENV_SENTINEL"
_SENTINEL_VALUE = "planted-by-test-must-not-be-read"
_PLANTED_KEY = "planted-twelvedata-key-must-not-be-read"

# Import every module under ``prism`` (the walk used by
# tests/test_import_hygiene.py), then report what the process picked up: the
# environment variables a stray ``load_dotenv()`` would have set, and any
# module attribute holding the planted key.
_PURITY_PROBE = """
import importlib, json, os, pkgutil, sys
import prism
for module in [m.name for m in pkgutil.walk_packages(prism.__path__, "prism.")]:
    importlib.import_module(module)
holders = []
for name, module in sorted(sys.modules.items()):
    if not name.startswith("prism"):
        continue
    for attr in dir(module):
        try:
            value = getattr(module, attr)
        except Exception:
            continue
        if isinstance(value, str) and value in (%(key)r, %(sentinel)r):
            holders.append(f"{name}.{attr}")
print(json.dumps({
    "sentinel": os.environ.get(%(sentinel_var)r),
    "key": os.environ.get("TWELVEDATA_API_KEY"),
    "holders": holders,
}))
""" % {"key": _PLANTED_KEY, "sentinel": _SENTINEL_VALUE, "sentinel_var": _SENTINEL_VAR}


def _plant_env(directory: Path) -> None:
    (directory / ".env").write_text(
        f"{_SENTINEL_VAR}={_SENTINEL_VALUE}\nTWELVEDATA_API_KEY={_PLANTED_KEY}\n",
        encoding="utf-8",
    )


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


@pytest.fixture
def restored_environ():
    """Snapshot/restore os.environ around a test that really loads a .env.

    ``monkeypatch.delenv`` would *restore* the loaded value at teardown, which
    is the opposite of what a planted credential needs.
    """
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_import_closure_is_pure(tmp_path):
    """Importing every prism module creates no directory, reads no .env, holds no key.

    The package is copied so its ancestors are inside ``tmp_path``: the
    pre-correction code wrote ``data/`` and ``results/`` three levels above
    ``config.py``, which is exactly what a wheel install would turn into a
    write inside ``site-packages``.
    """
    site = tmp_path / "site"
    shutil.copytree(PACKAGE_DIR, site / "prism", ignore=shutil.ignore_patterns("__pycache__"))
    work = tmp_path / "work"
    work.mkdir()
    for planted in (tmp_path, site, work):
        _plant_env(planted)
    before = _tree(tmp_path)
    work.chmod(0o555)  # a read-only working directory: a write here is an error, not a mess
    try:
        proc = subprocess.run(
            [sys.executable, "-P", "-c", _PURITY_PROBE],
            capture_output=True,
            text=True,
            cwd=work,
            # -P and an explicit PYTHONPATH: the closure is the copied package's,
            # not the checkout's. PYTHONDONTWRITEBYTECODE keeps __pycache__ out
            # of the directory diff.
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(site),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            timeout=600,
        )
    finally:
        work.chmod(0o755)
    assert proc.returncode == 0, f"import probe failed:\n{proc.stderr}"
    report = json.loads(proc.stdout.strip().splitlines()[-1])

    assert report["sentinel"] is None, "an import read a .env into the environment"
    assert report["key"] is None, "an import read a credential into the environment"
    assert report["holders"] == [], f"module attributes hold a planted credential: {report['holders']}"
    assert _tree(tmp_path) == before, "an import created something on disk"


def test_resolve_workspace_prefers_the_cli_then_env_then_pyproject_then_cwd(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("PRISM_WORKSPACE", raising=False)

    assert resolve_workspace(None).root == cwd

    (cwd / "pyproject.toml").write_text('[tool.prism]\nworkspace = "declared"\n', encoding="utf-8")
    assert resolve_workspace(None).root == (cwd / "declared").resolve()

    monkeypatch.setenv("PRISM_WORKSPACE", str(tmp_path / "from_env"))
    assert resolve_workspace(None).root == (tmp_path / "from_env").resolve()

    assert resolve_workspace(str(tmp_path / "from_cli")).root == (tmp_path / "from_cli").resolve()


def test_resolve_workspace_ignores_empty_sources(tmp_path, monkeypatch):
    """An unset flag and a blank env var fall through; they do not name "" as a root."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PRISM_WORKSPACE", "")
    assert resolve_workspace(None).root == tmp_path
    assert resolve_workspace("").root == tmp_path


def test_resolve_workspace_never_returns_the_package_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PRISM_WORKSPACE", raising=False)
    root = resolve_workspace(None).root
    assert root == tmp_path
    assert PACKAGE_DIR not in root.parents and root != PACKAGE_DIR


def test_resolve_workspace_refuses_a_malformed_or_non_string_declaration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PRISM_WORKSPACE", raising=False)
    (tmp_path / "pyproject.toml").write_text("[tool.prism\nworkspace = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid TOML"):
        resolve_workspace(None)
    (tmp_path / "pyproject.toml").write_text("[tool.prism]\nworkspace = 7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        resolve_workspace(None)


def test_resolve_workspace_tolerates_a_pyproject_without_the_table(tmp_path, monkeypatch):
    """The live checkout's own pyproject.toml has no [tool.prism]; cwd wins."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PRISM_WORKSPACE", raising=False)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "unrelated"\n', encoding="utf-8")
    assert resolve_workspace(None).root == tmp_path


def test_ensure_creates_the_layout_and_is_idempotent(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    assert not workspace.root.exists()
    assert workspace.ensure() is workspace
    for name in SUBDIRECTORIES:
        assert getattr(workspace, name).is_dir()
    assert workspace.data == workspace.root / "data"
    marker = workspace.results / "keep.txt"
    marker.write_text("kept", encoding="utf-8")
    workspace.ensure()
    assert marker.read_text(encoding="utf-8") == "kept"


def test_ensure_refuses_a_root_that_is_not_a_directory(tmp_path):
    file_root = tmp_path / "a_file"
    file_root.write_text("", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        Workspace(file_root).ensure()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the write bit")
def test_ensure_refuses_an_unwritable_root(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            Workspace(locked).ensure()
    finally:
        locked.chmod(0o755)


def test_run_dir_fits_and_trial_log_stay_inside_the_workspace(tmp_path):
    workspace = Workspace(tmp_path).ensure()
    assert workspace.run_dir("paper_loop_momentum2") == tmp_path / "state" / "paper_loop_momentum2"
    assert workspace.fits() == tmp_path / "state" / "fits"
    assert workspace.trial_log("momentum_v1") == tmp_path / "trials" / "momentum_v1.jsonl"
    for escape in ("..", "../elsewhere", "a/b", ""):
        with pytest.raises(ValueError):
            workspace.run_dir(escape)
        with pytest.raises(ValueError):
            workspace.trial_log(escape)


def test_relative_names_a_file_inside_the_workspace_without_the_machine_path(tmp_path, monkeypatch):
    root = tmp_path / "project"
    workspace = Workspace(root).ensure()
    record = workspace.results / "run-1" / "record.json"
    assert workspace.relative(record) == "results/run-1/record.json"
    assert workspace.relative(str(record)) == "results/run-1/record.json"
    assert workspace.relative(root) == "."

    # A relative path is read from the current directory.
    monkeypatch.chdir(root)
    assert workspace.relative(Path("specs") / "book.toml") == "specs/book.toml"
    assert Workspace(Path(".")).relative(record) == "results/run-1/record.json"

    # A path that reaches the workspace through a symlink or ``..`` still resolves into it.
    (tmp_path / "alias").symlink_to(root, target_is_directory=True)
    assert workspace.relative(tmp_path / "alias" / "trials" / "x.jsonl") == "trials/x.jsonl"
    assert workspace.relative(root / "results" / ".." / "trials" / "x.jsonl") == "trials/x.jsonl"

    # A file outside the workspace keeps the path it was given: it is the only way to find it.
    outside = tmp_path / "elsewhere" / "spec.toml"
    assert workspace.relative(outside) == str(outside)
    assert workspace.relative(root / ".." / "elsewhere" / "spec.toml") == str(root / ".." / "elsewhere" / "spec.toml")


def test_config_serves_the_deprecated_directories(tmp_path, monkeypatch):
    """The research drivers still bind PROJECT_DIR/DATA_DIR/RESULTS_DIR (spec §3.1)."""
    import prism.config as config

    with pytest.warns(DeprecationWarning):
        project_dir = config.PROJECT_DIR
    assert project_dir == REPO_ROOT
    with pytest.warns(DeprecationWarning):
        assert config.DATA_DIR == REPO_ROOT / "data"
    with pytest.warns(DeprecationWarning):
        assert config.RESULTS_DIR == REPO_ROOT / "results"
    # The names that carried a credential or a log directory are gone outright.
    for gone in ("TWELVEDATA_API_KEY", "LOGS_DIR"):
        with pytest.raises(AttributeError):
            getattr(config, gone)


def test_data_loader_takes_its_key_and_cache_from_the_caller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PRISM_WORKSPACE", raising=False)
    monkeypatch.setenv("TWELVEDATA_API_KEY", "from-environment")

    # No import-time capture: the environment is read now, at construction.
    assert DataLoader().api_key == "from-environment"
    assert DataLoader(api_key="explicit").api_key == "explicit"
    monkeypatch.delenv("TWELVEDATA_API_KEY")
    assert DataLoader().api_key is None

    # The cache follows the workspace, never the package location.
    assert DataLoader().cache_dir == tmp_path / "data"
    monkeypatch.setenv("PRISM_WORKSPACE", str(tmp_path / "ws"))
    assert DataLoader().cache_dir == tmp_path / "ws" / "data"
    assert DataLoader(cache_dir=tmp_path / "explicit").cache_dir == tmp_path / "explicit"


def test_configure_logging_requires_a_caller_supplied_directory(tmp_path):
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        with pytest.raises(TypeError):
            configure_logging(verbose=True)  # type: ignore[call-arg]
        path = configure_logging(tmp_path / "logs", run_name="unit")
        assert path.parent == tmp_path / "logs"
    finally:
        for handler in list(root.handlers):
            if getattr(handler, "_te_managed", False):
                root.removeHandler(handler)
                handler.close()
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def test_doctor_data_dir_defaults_to_the_workspace(tmp_path, monkeypatch):
    """The doctor checks the cache the loop will write, not one beside the package."""
    from prism.scripts.doctor import run_checks

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PRISM_WORKSPACE", str(tmp_path / "ws"))
    universe = tmp_path / "u.txt"
    universe.write_text("AAPL\n", encoding="utf-8")
    results = {r.name: r for r in run_checks(run_dir=tmp_path / "run", universe_file=universe, env={})}
    assert str(tmp_path / "ws" / "data") in results["data-dir"].detail


def test_load_env_reads_the_project_dotenv(tmp_path, monkeypatch, restored_environ):
    """Discovery starts at the current directory, and only a command calls it."""
    monkeypatch.delenv("PRISM_DISABLE_DOTENV", raising=False)
    monkeypatch.chdir(tmp_path)
    os.environ.pop(_SENTINEL_VAR, None)
    assert load_env() is None
    assert _SENTINEL_VAR not in os.environ

    _plant_env(tmp_path)
    assert load_env() == tmp_path / ".env"
    assert os.environ[_SENTINEL_VAR] == _SENTINEL_VALUE

    with pytest.raises(FileNotFoundError):
        load_env(tmp_path / "absent.env")


def test_load_env_never_crosses_a_checkout_boundary(tmp_path, monkeypatch, restored_environ):
    """A linked worktree under the live checkout must not inherit its ``.env``.

    Layout mirrors the 2026-09-09 incident: ``<repo>/.env`` beside
    ``<repo>/.git/``, and a worktree at ``<repo>/.claude/worktrees/wt`` whose
    ``.git`` is a file. From inside the worktree discovery returns nothing;
    from a plain subdirectory of the repo it still finds the repo's file.
    """
    monkeypatch.delenv("PRISM_DISABLE_DOTENV", raising=False)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    _plant_env(repo)
    worktree = repo / ".claude" / "worktrees" / "wt"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text("gitdir: ../../../.git/worktrees/wt\n", encoding="utf-8")
    nested = worktree / "src" / "pkg"
    nested.mkdir(parents=True)

    os.environ.pop(_SENTINEL_VAR, None)
    monkeypatch.chdir(nested)
    assert load_env() is None
    assert _SENTINEL_VAR not in os.environ

    _plant_env(worktree)
    assert load_env() == worktree / ".env"

    plain = repo / "src" / "prism"
    plain.mkdir(parents=True)
    monkeypatch.chdir(plain)
    assert load_env() == repo / ".env"


_WIKI_HTML = """
<table>
  <tr><th>Symbol</th><th>Security</th></tr>
  <tr><td>AAA</td><td>Alpha</td></tr>
  <tr><td>BBB</td><td>Beta</td></tr>
</table>
<table>
  <tr><th>Date</th><th>Added</th><th>Added</th><th>Removed</th><th>Removed</th><th>Reason</th></tr>
  <tr><td>Date</td><td>Ticker</td><td>Security</td><td>Ticker</td><td>Security</td><td>Reason</td></tr>
  <tr><td>2020-01-02</td><td>AAA</td><td>Alpha</td><td>CCC</td><td>Gamma</td><td>test</td></tr>
</table>
"""


def test_universe_builder_writes_into_the_workspace_and_not_beside_the_package(tmp_path, monkeypatch):
    """``prism-build-universe`` used to write to ``<package>/../../data/universe``.

    Offline route: a local Wikipedia snapshot and ``--skip_prices``, so the
    only thing under test is where the artifacts land.
    """
    from prism.scripts import build_sp500_universe as builder

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("PRISM_WORKSPACE", raising=False)
    html = tmp_path / "wiki.html"
    html.write_text(_WIKI_HTML, encoding="utf-8")
    workspace = tmp_path / "ws"

    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_sp500_universe",
            "--workspace",
            str(workspace),
            "--html_file",
            str(html),
            "--skip_prices",
            "--asof",
            "2024-01-31",
        ],
    )
    try:
        builder.main()
    finally:
        for handler in list(root.handlers):
            if getattr(handler, "_te_managed", False):
                root.removeHandler(handler)
                handler.close()
        root.handlers = saved_handlers
        root.setLevel(saved_level)

    assert (workspace / "data" / "universe" / "sp500_pit_2024-01-31.txt").is_file()
    assert list((workspace / "logs").glob("run_*.log")), "the run log did not land in the workspace"
    assert _tree(elsewhere) == set(), "the builder wrote into the current directory"
    assert not (PACKAGE_DIR.parents[1] / "data" / "universe" / "sp500_pit_2024-01-31.txt").exists()


def test_current_members_file_reads_the_workspace_bar_cache(tmp_path):
    """The 'has a local bar cache' filter globs the workspace, not the package."""
    import pandas as pd

    from prism.scripts.build_sp500_universe import _write_current_members

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "AAA_1d_2020-01-01_2024-01-31.parquet").write_bytes(b"")
    membership = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "start": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-02")],
            "end": [pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30")],
        }
    )
    out = tmp_path / "sp500_current.txt"
    _write_current_members(membership, "2024-01-31", out, data_dir)
    names = [line for line in out.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    assert names == ["AAA"], "only names with a bar cache in the given data dir belong in the live universe file"
