"""The suite-wide guard keeps ``.env`` discovery off inside a checkout."""

import os
import subprocess
import sys
from pathlib import Path

from prism.workspace import DOTENV_DISABLE_ENV, load_env

_SENTINEL = "PRISM_TEST_DOTENV_GUARD_SENTINEL"


def test_discovery_is_off_under_the_suite_guard(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".env").write_text(f"{_SENTINEL}=leaked\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(_SENTINEL, raising=False)
    assert os.environ.get(DOTENV_DISABLE_ENV) == "1", "tests/conftest.py must set the guard"

    assert load_env() is None
    assert _SENTINEL not in os.environ

    monkeypatch.delenv(DOTENV_DISABLE_ENV)
    assert load_env() == tmp_path / ".env"
    assert os.environ[_SENTINEL] == "leaked"
    monkeypatch.delenv(_SENTINEL)


def test_an_explicit_path_still_loads_under_the_guard(tmp_path, monkeypatch):
    env_file = tmp_path / "explicit.env"
    env_file.write_text(f"{_SENTINEL}=explicit\n", encoding="utf-8")
    monkeypatch.delenv(_SENTINEL, raising=False)
    assert load_env(env_file) == env_file
    assert os.environ[_SENTINEL] == "explicit"
    monkeypatch.delenv(_SENTINEL)


def _sentinel_after_importing_research_config(cwd, **env_overrides):
    """Import ``research.config`` in a fresh process at ``cwd``; return what the sentinel reads.

    A subprocess, because ``research.config`` is already imported (and its
    dataclasses already bound by other modules) in this one.
    """
    repo = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items() if key not in (_SENTINEL, "PYTHON_DOTENV_DISABLED")}
    env.pop(DOTENV_DISABLE_ENV, None)
    env.update(env_overrides)
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), str(repo)])
    code = f"import os, research.config; print(os.environ.get({_SENTINEL!r}, ''))"
    done = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def test_research_config_honours_the_guard_and_the_checkout_boundary(tmp_path):
    outer = tmp_path / "deployment"
    (outer / ".git").mkdir(parents=True)
    (outer / ".env").write_text(f"{_SENTINEL}=leaked\n", encoding="utf-8")
    inner = outer / ".worktrees" / "dev"
    inner.mkdir(parents=True)
    (inner / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")

    # The switch the suite sets keeps the research stack off a checkout's .env.
    assert _sentinel_after_importing_research_config(outer, **{DOTENV_DISABLE_ENV: "1"}) == ""
    # Without it, a worktree nested in a deployment checkout still stops at its own root.
    assert _sentinel_after_importing_research_config(inner) == ""
    # A driver run from the checkout itself still reads that checkout's .env.
    assert _sentinel_after_importing_research_config(outer) == "leaked"
