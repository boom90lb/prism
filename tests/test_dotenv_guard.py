"""The suite-wide guard keeps ``.env`` discovery off inside a checkout."""

import os

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
