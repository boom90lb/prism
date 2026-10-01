"""The off-box backup (ops/artifacts_sync.sh) and the wrappers' handling of it.

The predecessor script mirrored results/ whole into a GitHub-backed repository,
and once a record.json crossed GitHub's 100 MB blob limit every push was
refused — from 2026-09-25, each night, while the script logged PUSH-FAILED,
exited 0, and the nightly wrapper discarded even that. These tests run the
versioned replacement for real against temporary directories: a fake checkout,
a backup clone, and a bare remote on the local filesystem. Nothing here touches
the operating run directory, the real backup repository, a credential or the
network: the heartbeat and alert transports are exercised through a stub
``curl`` placed ahead of the real one on PATH, which records its arguments and
sends nothing.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[1] / "ops"
SYNC = OPS / "artifacts_sync.sh"
RUN_DIR = "runs/test_book"
LIMIT_BYTES = 1024 * 1024  # PRISM_SYNC_MAX_MB=1 in every run below

pytestmark = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("bash", "git", "rsync", "gzip")),
    reason="the sync script needs bash, git, rsync and gzip",
)


def _git(*args: str, cwd: Path, env: dict[str, str]) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True)
    return done.stdout.strip()


@pytest.fixture()
def env(tmp_path: Path) -> dict[str, str]:
    """A git identity under a throwaway HOME, and none of the operator's sync, release or alert settings.

    The wrappers read every one of these from the shell. An exported heartbeat
    URL would make a test run ping the real dead-man's monitor; an exported
    release pin would select a foreign interpreter.
    """
    home = tmp_path / "home"
    home.mkdir()
    environment = dict(os.environ)
    for name in (
        "PRISM_REPO",
        "PRISM_ARTIFACTS_REPO",
        "PRISM_SYNC_LOG",
        "PRISM_ARTIFACT_SYNC",
        "PRISM_RELEASE",
        "PRISM_RELEASE_ROOT",
        "PRISM_HEARTBEAT_URL",
        "PRISM_HEARTBEAT_URL_SWEEP",
        "PRISM_ALERT_CMD",
        "PRISM_ALERT_URL",
        "FRED_API_KEY",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "HOME": str(home),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "prism-test",
            "GIT_AUTHOR_EMAIL": "prism-test@example.invalid",
            "GIT_COMMITTER_NAME": "prism-test",
            "GIT_COMMITTER_EMAIL": "prism-test@example.invalid",
            "PRISM_SYNC_MAX_MB": "1",
        }
    )
    return environment


@pytest.fixture()
def source(tmp_path: Path) -> Path:
    """A fake checkout: a run directory, a log, and results/ with one oversize file."""
    src = tmp_path / "src"
    (src / RUN_DIR).mkdir(parents=True)
    (src / "runs" / "ACTIVE_RUN_DIR").write_text(RUN_DIR + "\n", encoding="utf-8")
    (src / RUN_DIR / "equity.jsonl").write_text('{"decision_bar": "2026-09-29"}\n', encoding="utf-8")
    (src / "logs").mkdir()
    (src / "logs" / "a.log").write_text("log\n", encoding="utf-8")
    (src / "results" / "run-big").mkdir(parents=True)
    (src / "results" / "run-big" / "record.json").write_bytes(os.urandom(LIMIT_BYTES + 500_000))
    (src / "results" / "run-big" / "small.json").write_text("{}\n", encoding="utf-8")
    (src / "results" / "note.txt").write_text("small\n", encoding="utf-8")
    return src


@pytest.fixture()
def remote(tmp_path: Path, env: dict[str, str]) -> Path:
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], env=env, check=True)
    return bare


@pytest.fixture()
def backup(tmp_path: Path, remote: Path, env: dict[str, str]) -> Path:
    clone = tmp_path / "backup"
    subprocess.run(["git", "clone", "-q", str(remote), str(clone)], env=env, check=True, capture_output=True)
    return clone


def _sync(source: Path, backup: Path, env: dict[str, str], **extra: str) -> subprocess.CompletedProcess:
    environment = dict(env)
    environment.update({"PRISM_REPO": str(source), "PRISM_ARTIFACTS_REPO": str(backup)})
    environment.update(extra)
    return subprocess.run(["bash", str(SYNC)], env=environment, capture_output=True, text=True, timeout=120)


def _status(source: Path) -> dict:
    return json.loads((source / RUN_DIR / "sync_status.json").read_text(encoding="utf-8"))


def _log(source: Path) -> str:
    return (source / RUN_DIR / "sync.log").read_text(encoding="utf-8")


def _debug(result: subprocess.CompletedProcess, source: Path) -> str:
    log = _log(source) if (source / RUN_DIR / "sync.log").exists() else "<no log>"
    return f"rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}\nlog=\n{log}"


# ---------------------------------------------------------------------------
# the sync itself
# ---------------------------------------------------------------------------


def test_first_sync_mirrors_compresses_commits_and_pushes(source, backup, remote, env) -> None:
    # What the old script left behind: a raw oversize copy in the backup worktree.
    stale_raw = backup / "results" / "run-big" / "record.json"
    stale_raw.parent.mkdir(parents=True)
    stale_raw.write_bytes(b"x" * (LIMIT_BYTES + 1))

    result = _sync(source, backup, env)
    assert result.returncode == 0, _debug(result, source)

    status = _status(source)
    head = _git("rev-parse", "--short", "HEAD", cwd=backup, env=env)
    assert status["ok"] is True and status["stage"] == "done"
    assert status["compressed"] == 1 and status["pushed"] is True and status["head"] == head
    assert "at" in status

    # runs/ and logs/ are mirrored; the sync's own bookkeeping stays out of the mirror.
    assert (backup / RUN_DIR / "equity.jsonl").read_text(encoding="utf-8").startswith('{"decision_bar"')
    assert (backup / "logs" / "a.log").exists()
    assert not (backup / RUN_DIR / "sync_status.json").exists()
    assert not (backup / RUN_DIR / "sync.log").exists()
    # Small results travel raw; the oversize file travels only as .gz, byte-identical when expanded.
    assert (backup / "results" / "note.txt").exists() and (backup / "results" / "run-big" / "small.json").exists()
    assert not stale_raw.exists()
    archived = backup / "results" / "run-big" / "record.json.gz"
    assert gzip.decompress(archived.read_bytes()) == (source / "results" / "run-big" / "record.json").read_bytes()
    # The remote advanced to the backup's commit and the worktree is clean.
    assert _git("rev-parse", "main", cwd=remote, env=env) == _git("rev-parse", "HEAD", cwd=backup, env=env)
    assert _git("status", "--porcelain", cwd=backup, env=env) == ""
    assert "nightly sync" in _git("log", "-1", "--format=%s", cwd=backup, env=env)
    assert "COMPRESSED results/run-big/record.json" in _log(source) and "SYNCED" in _log(source)


def test_second_sync_with_nothing_new_exits_0_and_says_no_change(source, backup, remote, env) -> None:
    assert _sync(source, backup, env).returncode == 0
    head = _git("rev-parse", "HEAD", cwd=backup, env=env)
    result = _sync(source, backup, env)
    assert result.returncode == 0, _debug(result, source)
    status = _status(source)
    assert status["ok"] is True and status["stage"] == "done"
    assert status["detail"].startswith("no-change")
    assert status["compressed"] == 0  # the existing .gz is newer than its source: not redone
    assert _git("rev-parse", "HEAD", cwd=backup, env=env) == head
    assert "NO-CHANGE" in _log(source)


def test_deletions_propagate_in_runs_and_logs_but_results_is_additive(source, backup, remote, env) -> None:
    assert _sync(source, backup, env).returncode == 0
    (source / "logs" / "a.log").unlink()
    (source / "results" / "note.txt").unlink()
    (source / RUN_DIR / "equity.jsonl").unlink()
    result = _sync(source, backup, env)
    assert result.returncode == 0, _debug(result, source)
    assert not (backup / "logs" / "a.log").exists()
    assert not (backup / RUN_DIR / "equity.jsonl").exists()
    assert (backup / "results" / "note.txt").exists()
    # The retired file is still in the backup's history, which is the recovery path docs/operations.md names.
    assert _git("rev-list", "--count", "HEAD", cwd=backup, env=env) == "2"


def test_a_results_file_that_grows_past_the_cap_leaves_no_stale_raw_copy(source, backup, remote, env) -> None:
    # record.json files grow across runs: mirrored raw while small, then skipped by
    # rsync --max-size once over the cap, which left the small stale copy in place.
    small = source / "results" / "run-grow" / "record.json"
    small.parent.mkdir()
    small.write_bytes(os.urandom(500_000))
    assert _sync(source, backup, env).returncode == 0
    raw = backup / "results" / "run-grow" / "record.json"
    assert raw.exists()
    time.sleep(1.1)  # the .gz freshness test is mtime-based
    small.write_bytes(os.urandom(LIMIT_BYTES + 200_000))
    result = _sync(source, backup, env)
    assert result.returncode == 0, _debug(result, source)
    assert not raw.exists(), "a raw copy of the pre-growth file must not survive beside the .gz"
    archived = backup / "results" / "run-grow" / "record.json.gz"
    assert gzip.decompress(archived.read_bytes()) == small.read_bytes()
    assert "DROPPED stale raw copy results/run-grow/record.json" in _log(source)
    assert _git("status", "--porcelain", cwd=backup, env=env) == ""
    # A third sync with nothing new must not resurrect it.
    assert _sync(source, backup, env).returncode == 0
    assert not raw.exists()


def test_a_stale_raw_beside_an_already_current_gz_is_also_dropped(source, backup, remote, env) -> None:
    # The production backup's state: a .gz newer than its source, and the stale raw
    # copy from before the file crossed the cap still at the canonical path.
    assert _sync(source, backup, env).returncode == 0
    stale = backup / "results" / "run-big" / "record.json"
    stale.write_bytes(b"old" * 1000)
    result = _sync(source, backup, env)
    assert result.returncode == 0, _debug(result, source)
    assert not stale.exists()
    assert _status(source)["compressed"] == 0  # the skip branch dropped it; nothing was re-gzipped


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file, so rsync cannot be made to fail")
def test_a_results_rsync_error_is_reported_after_the_ledgers_are_committed_and_pushed(
    source, backup, remote, env
) -> None:
    assert _sync(source, backup, env).returncode == 0
    with (source / RUN_DIR / "equity.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"decision_bar": "2026-09-30"}\n')
    locked = source / "results" / "scratch" / "locked.txt"
    locked.parent.mkdir()
    locked.write_text("unreadable\n", encoding="utf-8")
    locked.chmod(0o000)
    try:
        result = _sync(source, backup, env)
    finally:
        locked.chmod(0o644)
    assert result.returncode != 0, _debug(result, source)
    status = _status(source)
    assert status["ok"] is False and status["stage"] == "rsync" and status["pushed"] is True
    assert "rsync results/ exited" in status["detail"] and "committed and pushed" in status["detail"]
    # The irreplaceable ledgers landed on the remote despite the results/ problem.
    assert _git("status", "--porcelain", "--", "runs", cwd=backup, env=env) == ""
    shown = _git("show", f"main:{RUN_DIR}/equity.jsonl", cwd=remote, env=env)
    assert '"2026-09-30"' in shown


def test_an_absolute_active_run_dir_pointer_is_taken_as_given(source, backup, remote, env) -> None:
    # prism.live.safety.require_active_run_dir accepts a repo-relative or absolute
    # pointer; the shell must not compose $REPO/<absolute path>.
    (source / "runs" / "ACTIVE_RUN_DIR").write_text(str(source / RUN_DIR) + "\n", encoding="utf-8")
    result = _sync(source, backup, env)
    assert result.returncode == 0, _debug(result, source)
    assert _status(source)["ok"] is True
    assert (source / RUN_DIR / "sync.log").exists()
    assert not (source / str(source).lstrip("/")).exists(), "no $REPO//abs/path tree may be created"


def test_a_control_byte_in_an_input_still_yields_valid_status_json(source, backup, remote, env) -> None:
    # The detail strings embed operator-controlled text; JSON forbids raw control bytes.
    result = _sync(source, backup, env, PRISM_SYNC_MAX_MB="45\x01")
    assert result.returncode != 0
    status = _status(source)  # json.loads must succeed
    assert status["ok"] is False and "not a whole number" in status["detail"]


def test_a_failed_push_exits_nonzero_and_keeps_the_commit(source, backup, remote, env) -> None:
    assert _sync(source, backup, env).returncode == 0
    _git("remote", "set-url", "origin", str(source.parent / "nowhere" / "remote.git"), cwd=backup, env=env)
    (source / "results" / "new.txt").write_text("new\n", encoding="utf-8")
    result = _sync(source, backup, env)
    assert result.returncode != 0, _debug(result, source)
    status = _status(source)
    assert status["ok"] is False and status["stage"] == "push" and status["pushed"] is False
    assert status["head"] is not None
    assert "1 local commit(s) unpushed" in status["detail"]
    assert _git("rev-list", "--count", "HEAD", cwd=backup, env=env) == "2"  # the commit is kept locally


def _install_size_limit_hook(remote: Path) -> None:
    """GitHub's pre-receive refusal, as it reads in the backup repo's sync.log from 2026-09-25 on."""
    hook = remote / "hooks" / "pre-receive"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"remote: error: File results/run-a195851811a34615/record.json is 128.17 MB; "
        "this exceeds GitHub's file size limit of 100.00 MB\" >&2\n"
        "echo 'remote: error: GH001: Large files detected. You may want to try Git Large File Storage' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)


def test_a_remote_size_limit_refusal_names_the_operator_repair(source, backup, remote, env) -> None:
    _install_size_limit_hook(remote)
    result = _sync(source, backup, env)
    assert result.returncode != 0, _debug(result, source)
    status = _status(source)
    assert status["ok"] is False and status["stage"] == "push"
    assert "GH001" in status["detail"] and "operator repair" in status["detail"]
    assert "is 128.17 MB" in status["detail"]  # the remote's own line, carried into the report
    assert "PUSH-REFUSED" in _log(source) and "does not rewrite history" in _log(source)


def test_a_size_limit_refusal_with_an_oversize_file_in_the_current_tree_does_not_blame_history(
    source, backup, remote, env
) -> None:
    # runs/ and logs/ mirror uncapped, so a blob over the remote's 100 MiB limit can
    # be in the commit this run made; the reset --soft repair would not clear that.
    _install_size_limit_hook(remote)
    with (source / RUN_DIR / "nightly.log").open("wb") as handle:
        handle.truncate(101 * 1024 * 1024)
    result = _sync(source, backup, env)
    assert result.returncode != 0, _debug(result, source)
    status = _status(source)
    assert status["ok"] is False and status["stage"] == "push"
    assert "oversize file(s) in the current tree" in status["detail"]
    assert f"{RUN_DIR}/nightly.log" in status["detail"]
    assert "operator repair" not in status["detail"]
    assert "reset --soft will not clear it" in _log(source)


def test_missing_logs_is_tolerated_and_missing_results_is_not(source, backup, remote, env) -> None:
    shutil.rmtree(source / "logs")
    result = _sync(source, backup, env)
    assert result.returncode == 0, _debug(result, source)
    assert "logs/ absent at source; skipped" in _log(source)
    shutil.rmtree(source / "results")
    result = _sync(source, backup, env)
    assert result.returncode != 0
    status = _status(source)
    assert status["ok"] is False and status["stage"] == "repo"


def test_a_backup_path_that_is_not_a_repository_fails_at_the_repo_stage(source, tmp_path, env) -> None:
    result = _sync(source, tmp_path / "not-a-repo", env)
    assert result.returncode != 0
    status = _status(source)
    assert status["ok"] is False and status["stage"] == "repo" and status["head"] is None


def test_status_falls_back_to_runs_when_no_active_run_dir_is_named(source, backup, remote, env) -> None:
    (source / "runs" / "ACTIVE_RUN_DIR").unlink()
    result = _sync(source, backup, env)
    assert result.returncode == 0, result.stderr
    status = json.loads((source / "runs" / "sync_status.json").read_text(encoding="utf-8"))
    assert status["ok"] is True and status["stage"] == "done"
    assert not (source / RUN_DIR / "sync_status.json").exists()


def test_a_non_numeric_limit_is_refused(source, backup, remote, env) -> None:
    result = _sync(source, backup, env, PRISM_SYNC_MAX_MB="45m")
    assert result.returncode != 0
    assert _status(source)["stage"] == "repo"


# ---------------------------------------------------------------------------
# transports: a stub curl records what would have been sent
# ---------------------------------------------------------------------------


def _stub_curl(tmp_path: Path) -> tuple[Path, Path]:
    """A ``curl`` ahead of the real one on PATH that records its arguments and exits 0.

    One line per call holds the argument vector; when the data argument is
    ``@-`` (the body on stdin, as ``alert_curl.sh`` sends it) the body follows
    on its own line. Stdin is read only in that case, so the heartbeat's call,
    which passes no body, can never block on an inherited descriptor.
    """
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / "curl_calls.log"
    stub = bin_dir / "curl"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{record}"\n'
        f'case " $* " in *" @- "*) cat >> "{record}"; printf "\\n" >> "{record}";; esac\n'
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return bin_dir, record


def _alert_curl(bin_dir: Path, env: dict[str, str], subject: str, body: str) -> subprocess.CompletedProcess:
    environment = dict(env, PATH=f"{bin_dir}:{env['PATH']}", PRISM_ALERT_URL="https://ntfy.example/topic")
    return subprocess.run(
        ["bash", str(OPS / "alert_curl.sh"), subject, body], env=environment, capture_output=True, text=True
    )


def test_alert_curl_posts_title_and_body_to_the_configured_url(tmp_path, env) -> None:
    bin_dir, record = _stub_curl(tmp_path)
    done = _alert_curl(bin_dir, env, "prism nightly unhealthy (loop=1 doctor=0)", "read the log")
    assert done.returncode == 0, done.stderr
    args, body = record.read_text(encoding="utf-8").splitlines()
    assert "Title: prism nightly unhealthy (loop=1 doctor=0)" in args
    assert "--data-binary @-" in args and args.endswith("https://ntfy.example/topic")
    assert body == "read the log"


def test_alert_curl_sends_a_body_starting_with_at_as_text_not_as_a_file_name(tmp_path, env) -> None:
    # As a --data-binary argument, "@/etc/passwd" would tell curl to read that file.
    bin_dir, record = _stub_curl(tmp_path)
    done = _alert_curl(bin_dir, env, "subject", "@/etc/passwd")
    assert done.returncode == 0, done.stderr
    args, body = record.read_text(encoding="utf-8").splitlines()
    assert "--data-binary @-" in args and "@/etc/passwd" not in args
    assert body == "@/etc/passwd"


def test_alert_curl_refuses_without_a_url_and_never_prints_it(tmp_path, env) -> None:
    bin_dir, record = _stub_curl(tmp_path)
    environment = dict(env, PATH=f"{bin_dir}:{env['PATH']}")
    done = subprocess.run(
        ["bash", str(OPS / "alert_curl.sh"), "subject", "body"], env=environment, capture_output=True, text=True
    )
    assert done.returncode != 0 and "PRISM_ALERT_URL is unset" in done.stderr
    assert not record.exists()


# ---------------------------------------------------------------------------
# the wrappers: a failed backup is alerted and folded into the exit code; the
# heartbeat carries that code last
# ---------------------------------------------------------------------------

WRAPPERS = {"nightly": "paper_loop_nightly.sh", "sweep": "paper_sweep_morning.sh"}


def _stub_interpreter(path: Path) -> None:
    """A stand-in for ``.venv/bin/python3`` that answers the version probe and exits 0 for the session."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\n" 'if [ "$1" = "-c" ]; then echo 9.9.9+stub; exit 0; fi\n' "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _fake_sync(path: Path, rc: int, record: Path) -> None:
    """Stands in for ops/artifacts_sync.sh at the wrapper's default path, exiting ``rc``."""
    path.write_text(
        "#!/usr/bin/env bash\n" f'echo "sync ran" >> "{record}"\n' f"exit {rc}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.fixture()
def wrapper_repo(tmp_path: Path) -> Path:
    """A throwaway checkout with the real wrappers and _alert.sh, stub interpreter, no .env."""
    root = tmp_path / "repo"
    (root / "ops").mkdir(parents=True)
    for name in ("_alert.sh", *WRAPPERS.values()):
        shutil.copy2(OPS / name, root / "ops" / name)
        (root / "ops" / name).chmod(0o755)
    (root / "runs").mkdir()
    (root / "runs" / "ACTIVE_RUN_DIR").write_text(RUN_DIR + "\n", encoding="utf-8")
    (root / "data" / "universe").mkdir(parents=True)
    (root / "data" / "universe" / "sp500_current.txt").write_text("AAA\nBBB\n", encoding="utf-8")
    _stub_interpreter(root / ".venv" / "bin" / "python3")
    return root


def _run_wrapper(repo: Path, which: str, env: dict[str, str], **extra: str) -> subprocess.CompletedProcess:
    environment = dict(env)
    environment.update(
        {
            # The arming gate only asks whether the variables are present.
            "APCA_API_KEY_ID": "placeholder-not-a-key",
            "APCA_API_SECRET_KEY": "placeholder-not-a-key",
            "PRISM_REPO": str(repo),
        }
    )
    environment.update(extra)
    return subprocess.run(
        ["bash", str(repo / "ops" / WRAPPERS[which])], env=environment, capture_output=True, text=True, timeout=120
    )


def test_the_nightly_runs_the_versioned_sync_by_default_and_folds_its_failure_into_the_exit(
    wrapper_repo, tmp_path, env
) -> None:
    record = tmp_path / "sync_calls.log"
    _fake_sync(wrapper_repo / "ops" / "artifacts_sync.sh", 7, record)
    result = _run_wrapper(wrapper_repo, "nightly", env)
    assert result.returncode == 7, result.stderr
    assert record.read_text(encoding="utf-8").strip() == "sync ran"
    log = (wrapper_repo / RUN_DIR / "nightly.log").read_text(encoding="utf-8")
    assert "SYNC EXIT 7" in log and "ALERT: prism artifact backup failed" in log
    alert = (wrapper_repo / RUN_DIR / "ALERT").read_text(encoding="utf-8")
    assert "prism artifact backup failed" in alert and "sync_status.json" in alert


def test_the_nightly_with_a_green_sync_exits_0_and_raises_no_alert(wrapper_repo, tmp_path, env) -> None:
    _fake_sync(wrapper_repo / "ops" / "artifacts_sync.sh", 0, tmp_path / "sync_calls.log")
    result = _run_wrapper(wrapper_repo, "nightly", env)
    assert result.returncode == 0, result.stderr
    assert not (wrapper_repo / RUN_DIR / "ALERT").exists()
    assert "SYNC EXIT 0" in (wrapper_repo / RUN_DIR / "nightly.log").read_text(encoding="utf-8")


def test_the_sync_override_still_disables_it(wrapper_repo, tmp_path, env) -> None:
    record = tmp_path / "sync_calls.log"
    _fake_sync(wrapper_repo / "ops" / "artifacts_sync.sh", 7, record)
    result = _run_wrapper(wrapper_repo, "nightly", env, PRISM_ARTIFACT_SYNC="/dev/null")
    assert result.returncode == 0, result.stderr
    assert not record.exists()


def test_the_heartbeat_carries_the_final_exit_code_and_a_failed_ping_changes_nothing(
    wrapper_repo, tmp_path, env
) -> None:
    bin_dir, record = _stub_curl(tmp_path)
    _fake_sync(wrapper_repo / "ops" / "artifacts_sync.sh", 7, tmp_path / "sync_calls.log")
    path = f"{bin_dir}:{env['PATH']}"
    result = _run_wrapper(wrapper_repo, "nightly", env, PATH=path, PRISM_HEARTBEAT_URL="https://hc.example/ping/")
    assert result.returncode == 7
    calls = record.read_text(encoding="utf-8").splitlines()
    assert calls[-1].endswith("https://hc.example/ping/7")  # healthchecks.io: /<rc>, trailing slash folded
    log = (wrapper_repo / RUN_DIR / "nightly.log").read_text(encoding="utf-8")
    assert log.rstrip().splitlines()[-1].endswith("HEARTBEAT rc=7")
    assert "hc.example" not in log  # the ping URL is a secret and never reaches the log

    # The sweep pings its own monitor with /0 on a healthy morning.
    result = _run_wrapper(wrapper_repo, "sweep", env, PATH=path, PRISM_HEARTBEAT_URL_SWEEP="https://hc.example/sweep")
    assert result.returncode == 0, result.stderr
    assert record.read_text(encoding="utf-8").splitlines()[-1].endswith("https://hc.example/sweep/0")

    # A ping transport that fails is logged and does not touch the exit code.
    (bin_dir / "curl").write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    result = _run_wrapper(wrapper_repo, "sweep", env, PATH=path, PRISM_HEARTBEAT_URL_SWEEP="https://hc.example/sweep")
    assert result.returncode == 0
    assert "HEARTBEAT-FAILED rc=0 (curl exit 7)" in (wrapper_repo / RUN_DIR / "sweep.log").read_text(encoding="utf-8")


def test_without_a_heartbeat_url_nothing_is_pinged(wrapper_repo, tmp_path, env) -> None:
    bin_dir, record = _stub_curl(tmp_path)
    _fake_sync(wrapper_repo / "ops" / "artifacts_sync.sh", 0, tmp_path / "sync_calls.log")
    result = _run_wrapper(wrapper_repo, "nightly", env, PATH=f"{bin_dir}:{env['PATH']}")
    assert result.returncode == 0, result.stderr
    assert not record.exists()
    assert "HEARTBEAT" not in (wrapper_repo / RUN_DIR / "nightly.log").read_text(encoding="utf-8")
