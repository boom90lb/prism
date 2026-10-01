"""The ops wrappers' release switch (§7 contract migration, stage 5).

``SPEC.md`` still says "the working tree is the live system": the nightly runs
the editable ``.venv`` install of whatever is in the tree, and uncommitted
changes trade that night. The migration row replaces that with a pinned
installed release. This file pins the shape of that change rather than the
change itself:

* **both paths work**, and the default is exactly what runs today, so upgrading
  the wrapper alone changes nothing about the session that trades;
* **the switch is one line**, and throwing it is the account owner's act — no
  test here asserts a pinned release is selected, because selecting one is not
  this stage's work;
* **it never falls back**: a release that does not resolve refuses the session
  and alerts, because a wrapper that quietly reverted to the checkout after the
  owner pinned a release would trade code nobody approved;
* the provenance stamp the health check and the paper-state assessment parse
  still parses, with the release and engine version added beside it.

The wrappers are exercised for real, against a temporary repository holding
stub interpreters. Nothing here touches a venue, a credential or the operating
run directory: the Alpaca variables the arming gate reads are passed as obvious
placeholders in the subprocess environment and no ``.env`` is written.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from prism.scripts.doctor import _WRAPPER_STAMP
from prism.scripts.paper_state_assessment import _COMMIT_FIELD

OPS = Path(__file__).resolve().parents[1] / "ops"
WRAPPERS = {"nightly": "paper_loop_nightly.sh", "sweep": "paper_sweep_morning.sh"}
RUN_DIR = "runs/test_book"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="the ops wrappers are bash scripts")


def _stub_interpreter(path: Path, name: str, log: Path) -> None:
    """A stand-in for a release's ``bin/python3`` that records how it was called."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-c" ]; then echo "9.9.9+' + name + '"; exit 0; fi\n'
        'printf "%s %s\\n" "' + name + '" "$*" >> "' + str(log) + '"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A throwaway repository holding the real wrappers and stub interpreters."""
    root = tmp_path / "repo"
    (root / "ops").mkdir(parents=True)
    for name in ("_alert.sh", *WRAPPERS.values()):
        shutil.copy2(OPS / name, root / "ops" / name)
        (root / "ops" / name).chmod(0o755)
    (root / "runs").mkdir()
    (root / "runs" / "ACTIVE_RUN_DIR").write_text(RUN_DIR + "\n", encoding="utf-8")
    (root / "data" / "universe").mkdir(parents=True)
    (root / "data" / "universe" / "sp500_current.txt").write_text("AAA\nBBB\n", encoding="utf-8")
    _stub_interpreter(root / ".venv" / "bin" / "python3", "checkout", tmp_path / "calls.log")
    _stub_interpreter(root / "releases" / "pinned" / "bin" / "python3", "pinned", tmp_path / "calls.log")
    return root


# Variables the wrappers read from the shell. Scrubbed so an operator's exported
# heartbeat URL or pinned release cannot reach a real monitor or select a
# foreign interpreter from a test run (the same list test_artifacts_sync uses).
_WRAPPER_ENV = (
    "PRISM_REPO",
    "PRISM_ARTIFACT_SYNC",
    "PRISM_RELEASE",
    "PRISM_RELEASE_ROOT",
    "PRISM_HEARTBEAT_URL",
    "PRISM_HEARTBEAT_URL_SWEEP",
    "PRISM_ALERT_CMD",
    "PRISM_ALERT_URL",
    "FRED_API_KEY",
)


def _stub_curl(root: Path) -> Path:
    """A ``curl`` ahead of the real one on PATH: the wrappers end with a heartbeat ping, and no test
    here may reach a monitor even if a URL leaks into the environment."""
    bin_dir = root / "stubbin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "curl"
    if not stub.exists():
        stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
    return bin_dir


def _run(repo: Path, which: str, **env: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    for name in _WRAPPER_ENV:
        environment.pop(name, None)
    environment.update(
        {
            # The arming gate only asks whether the variables are present.
            "APCA_API_KEY_ID": "placeholder-not-a-key",
            "APCA_API_SECRET_KEY": "placeholder-not-a-key",
            "PRISM_ARTIFACT_SYNC": "/dev/null",  # not executable: the sync is skipped
            "PRISM_REPO": str(repo),
            "PATH": f"{_stub_curl(repo.parent)}:{environment.get('PATH', '')}",
        }
    )
    environment.update(env)
    return subprocess.run(
        ["bash", str(repo / "ops" / WRAPPERS[which])],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _log(repo: Path, which: str) -> str:
    name = "nightly.log" if which == "nightly" else "sweep.log"
    return (repo / RUN_DIR / name).read_text(encoding="utf-8")


def _calls(repo: Path) -> list[str]:
    path = repo.parent / "calls.log"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


# ---------------------------------------------------------------------------
# the switch itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wrapper", sorted(WRAPPERS.values()))
def test_the_switch_is_one_line_and_defaults_to_the_editable_checkout(wrapper) -> None:
    lines = (OPS / wrapper).read_text(encoding="utf-8").splitlines()
    selections = [line for line in lines if line.startswith("PRISM_RELEASE=")]
    assert selections == ['PRISM_RELEASE="${PRISM_RELEASE:-checkout}"'], (
        "the release is selected on exactly one line, and its default is the editable checkout — " "what runs today"
    )


@pytest.mark.parametrize("which", sorted(WRAPPERS))
def test_the_default_runs_the_working_trees_venv_exactly_as_today(repo, which) -> None:
    result = _run(repo, which)
    assert result.returncode == 0, result.stderr
    calls = _calls(repo)
    assert calls and all(call.startswith("checkout ") for call in calls)
    assert "release=checkout" in _log(repo, which)


def test_the_nightly_passes_the_same_command_line_it_passes_today(repo) -> None:
    assert _run(repo, "nightly").returncode == 0
    loop, doctor = _calls(repo)
    assert "-m prism.scripts.paper_loop" in loop
    assert "--book momentum" in loop
    assert "--submit" in loop and "--no-submit" not in loop  # the wrapper is the one caller that trades
    assert "--universe-file data/universe/sp500_current.txt" in loop
    assert f"--run-dir {RUN_DIR}" in loop
    assert "--spinoff-mask" in loop
    assert "--regime" not in loop  # no FRED key in the environment: unchanged behaviour
    assert "-m prism.scripts.doctor" in doctor and "--network" in doctor


def test_the_sweep_passes_the_same_command_line_it_passes_today(repo) -> None:
    assert _run(repo, "sweep").returncode == 0
    sweep, doctor = _calls(repo)
    assert "-m prism.scripts.paper_sweep" in sweep and f"--run-dir {RUN_DIR}" in sweep
    assert "--submit" in sweep and "--no-submit" not in sweep
    assert "-m prism.scripts.doctor" in doctor and "--network" not in doctor


@pytest.mark.parametrize("which", sorted(WRAPPERS))
def test_a_pinned_release_runs_that_release_and_nothing_from_the_checkout(repo, which) -> None:
    result = _run(repo, which, PRISM_RELEASE="pinned")
    assert result.returncode == 0, result.stderr
    calls = _calls(repo)
    assert calls and all(call.startswith("pinned ") for call in calls)
    log = _log(repo, which)
    assert "release=pinned" in log and "engine=9.9.9+pinned" in log


@pytest.mark.parametrize("which", sorted(WRAPPERS))
def test_an_absolute_release_root_is_taken_as_given(repo, which) -> None:
    root = repo / "releases" / "pinned"
    assert _run(repo, which, PRISM_RELEASE=str(root)).returncode == 0
    assert all(call.startswith("pinned ") for call in _calls(repo))


@pytest.mark.parametrize("which", sorted(WRAPPERS))
def test_a_release_that_does_not_resolve_refuses_the_session_rather_than_falling_back(repo, which) -> None:
    result = _run(repo, which, PRISM_RELEASE="never-installed")
    assert result.returncode == 4
    assert _calls(repo) == [], "no interpreter ran: the refusal happens before the session"
    log = _log(repo, which)
    assert "FAIL: PRISM_RELEASE=never-installed" in log
    assert "refusing to run rather than falling back to the checkout" in log
    alert = (repo / RUN_DIR / "ALERT").read_text(encoding="utf-8")
    assert "never-installed" in alert


@pytest.mark.parametrize("which", sorted(WRAPPERS))
@pytest.mark.parametrize("state", ["missing", "empty"])
def test_a_missing_active_run_dir_refuses_the_session_rather_than_falling_back(repo, which, state) -> None:
    """Until 2026-09-30 a missing switch fell back to the retired runs/paper_loop_momentum."""
    pointer = repo / "runs" / "ACTIVE_RUN_DIR"
    if state == "missing":
        pointer.unlink()
    else:
        pointer.write_text("\n", encoding="utf-8")
    result = _run(repo, which)
    assert result.returncode == 4
    assert _calls(repo) == [], "no interpreter ran: the refusal happens before the session"
    assert not (repo / "runs" / "paper_loop_momentum").exists(), "nothing fell back to the retired directory"
    # With no run directory to log into, the refusal is recorded beside the switch.
    log = (repo / "runs" / ("nightly.log" if which == "nightly" else "sweep.log")).read_text(encoding="utf-8")
    assert "FAIL:" in log and "ACTIVE_RUN_DIR is missing or empty" in log
    alert = (repo / "runs" / "ALERT").read_text(encoding="utf-8")
    assert "no active run directory" in alert


@pytest.mark.parametrize("which", sorted(WRAPPERS))
def test_an_absolute_active_run_dir_pointer_logs_into_that_directory(repo, which) -> None:
    """The pointer may be repo-relative or absolute (prism.live.safety.require_active_run_dir); the shell
    must not compose $REPO/<absolute path> and log into a tree inside the checkout."""
    run_dir = repo / RUN_DIR
    (repo / "runs" / "ACTIVE_RUN_DIR").write_text(str(run_dir) + "\n", encoding="utf-8")
    result = _run(repo, which)
    assert result.returncode == 0, result.stderr
    assert (run_dir / ("nightly.log" if which == "nightly" else "sweep.log")).exists()
    assert not (repo / str(repo).lstrip("/")).exists(), "no $REPO//abs/path tree may be created"
    assert any(f"--run-dir {run_dir}" in call for call in _calls(repo))


# ---------------------------------------------------------------------------
# the provenance stamp the health readers parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", sorted(WRAPPERS))
def test_the_provenance_stamp_still_parses_with_the_release_added(repo, which) -> None:
    _run(repo, which)
    line = [row for row in _log(repo, which).splitlines() if " RUN: " in row or " SWEEP: " in row][-1]
    stamp = _WRAPPER_STAMP.search(line)
    assert stamp is not None and stamp.group("path").endswith(WRAPPERS[which])
    commit = _COMMIT_FIELD.search(line)
    assert commit is not None and commit.group("commit")
    assert re.search(r"\brelease=\S+", line) and re.search(r"\bengine=\S+", line)
