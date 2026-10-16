"""The run-recording protocol and the trial log (engine specification §3.12).

Four properties are worth more than the row shapes and are tested as such:

* **A started configuration counts as an attempt whether or not it finished.**
  The crash case — a process that dies between the ``started`` row and its
  terminal row — is closed by ``reconcile``, which writes ``incomplete`` with
  the reason. The attempt was always counted; what the crash costs is the
  metrics, not the denominator.
* **A run that is still executing is never marked incomplete by another
  invocation.** That would write a false terminal row for a live run and leave
  two terminal rows in an append-only log, so reconcile requires *both* a stale
  heartbeat and a dead process on the recorded host before it closes anything.
* **The chain detects a removed row.** Deleting a row from an append-only log
  is falsification; the chain makes it detectable relative to a trusted copy
  rather than deniable.
* **Two concurrent appenders cannot fork the chain.** Each append takes the
  exclusive lock and re-reads the log inside it, so the row it chains onto is
  the row that is last at the moment it writes.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from prism import history_import
from prism.validation import trials
from prism.validation.trials import (
    HOST_LABEL_ENV,
    RECORD_SCHEMA,
    RunRecorder,
    TrialLog,
    TrialLogError,
    attempt_counts,
    chain_hash,
    host_id_path,
    host_label,
    reconcile_trial_log,
    record_schema_problems,
    this_host_names,
    utc_now,
    validate_result_record,
)

HOST = "test-host"


def _row(run_id: str, event: str = "started", **overrides: object) -> dict[str, object]:
    """One protocol row with the fields the log requires."""
    row: dict[str, object] = {
        "ts": utc_now(),
        "event": event,
        "run_id": run_id,
        "config_id": "c" * 64,
        "data_id": "d" * 64,
        "kind": "new",
        "arm_id": None,
        "selection_set_id": "unit_set",
        "capital": 1.0,
        "synthetic": False,
        "run_channel": "evidence",
        "summary": None,
        "reason": None,
        "pid": os.getpid(),
        "host": HOST,
        "heartbeat_at": utc_now() if event == "started" else None,
    }
    row.update(overrides)
    return row


def _dead_pid() -> int:
    """A pid that has certainly exited (its process is reaped before we return)."""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return int(process.pid)


def _aged(seconds: float) -> str:
    return (
        (datetime.now(timezone.utc) - timedelta(seconds=seconds))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


# ---- ordering --------------------------------------------------------------


def test_a_started_row_is_followed_by_exactly_one_terminal_row(tmp_path: Path) -> None:
    log = TrialLog(tmp_path / "set.jsonl")
    log.append(_row("run-1"))
    log.append(_row("run-1", "finished", summary={"n_obs": 10}))

    rows = log.verify()
    assert [row["event"] for row in rows] == ["started", "finished"]

    with pytest.raises(TrialLogError, match="already has the terminal row"):
        log.append(_row("run-1", "failed", reason="a second outcome"))
    with pytest.raises(TrialLogError, match="already has a started row"):
        log.append(_row("run-1"))
    assert len(log.rows()) == 2, "a refused append writes nothing"


def test_a_terminal_row_needs_the_attempt_it_ends(tmp_path: Path) -> None:
    """Except ``failed``: identity resolution can fail before the started row."""
    log = TrialLog(tmp_path / "set.jsonl")

    with pytest.raises(TrialLogError, match="no started row"):
        log.append(_row("run-1", "finished", summary={}))
    with pytest.raises(TrialLogError, match="no started row"):
        log.append(_row("run-1", "incomplete", reason="never began"))

    log.append(_row("run-2", "failed", reason="identity resolution failed"))
    assert [row["event"] for row in log.verify()] == ["failed"]


def test_a_row_outside_the_schema_is_refused(tmp_path: Path) -> None:
    log = TrialLog(tmp_path / "set.jsonl")

    with pytest.raises(TrialLogError, match="outside the row schema"):
        log.append(dict(_row("run-1"), extra_field="not in §3.12"))
    with pytest.raises(TrialLogError, match="must carry the reason"):
        log.append(_row("run-1", "failed"))


# ---- the crash case --------------------------------------------------------


def test_the_crash_case_between_a_started_row_and_a_terminal_row(tmp_path: Path) -> None:
    """A dead process with a stale heartbeat is closed as ``incomplete``, with its reason."""
    log = TrialLog(tmp_path / "set.jsonl")
    log.append(_row("run-crashed", pid=_dead_pid(), host=HOST, heartbeat_at=_aged(4000)))

    report = reconcile_trial_log(log.path, host=HOST, max_heartbeat_age_seconds=900)

    assert report.n_open == 1
    assert [verdict.verdict for verdict in report.closed] == ["closed_incomplete"]
    rows = log.verify()
    assert [row["event"] for row in rows] == ["started", "incomplete"]
    reason = str(rows[1]["reason"])
    assert "never recorded a terminal row" in reason
    assert "is gone" in reason
    assert rows[1]["run_id"] == "run-crashed"
    # The attempt was always counted; the crash costs the metrics, not the
    # denominator (§3.12, "Attempt counting").
    assert attempt_counts(rows, registered=True)["n_trials_observed"] == 1

    # Reconciling again changes nothing: the run now has a terminal row.
    again = reconcile_trial_log(log.path, host=HOST, max_heartbeat_age_seconds=900)
    assert again.n_open == 0 and not again.closed
    assert len(log.rows()) == 2


def test_a_run_still_executing_is_not_marked_incomplete(tmp_path: Path) -> None:
    log = TrialLog(tmp_path / "set.jsonl")
    log.append(_row("run-live", pid=os.getpid(), host=HOST, heartbeat_at=_aged(4000)))

    report = reconcile_trial_log(log.path, host=HOST, max_heartbeat_age_seconds=1)

    assert not report.closed
    assert [verdict.verdict for verdict in report.in_flight] == ["process_alive"]
    assert [row["event"] for row in log.verify()] == ["started"]


def test_a_fresh_heartbeat_defers_to_the_clock_even_when_the_pid_is_gone(tmp_path: Path) -> None:
    """Pid reuse is real, so a dead pid alone never closes a run."""
    log = TrialLog(tmp_path / "set.jsonl")
    log.append(_row("run-fresh", pid=_dead_pid(), host=HOST, heartbeat_at=utc_now()))

    report = reconcile_trial_log(log.path, host=HOST, max_heartbeat_age_seconds=900)

    assert not report.closed
    assert [verdict.verdict for verdict in report.in_flight] == ["heartbeat_fresh"]


def test_a_run_on_another_host_is_decided_by_heartbeat_age_alone(tmp_path: Path) -> None:
    log = TrialLog(tmp_path / "set.jsonl")
    log.append(_row("run-elsewhere", pid=os.getpid(), host="another-host", heartbeat_at=utc_now()))

    fresh = reconcile_trial_log(log.path, host=HOST, max_heartbeat_age_seconds=900)
    assert [verdict.verdict for verdict in fresh.in_flight] == ["unverifiable_host"]
    assert len(log.rows()) == 1

    log.append(_row("run-old-elsewhere", pid=os.getpid(), host="another-host", heartbeat_at=_aged(4000)))
    stale = reconcile_trial_log(log.path, host=HOST, max_heartbeat_age_seconds=900)
    assert [verdict.run_id for verdict in stale.closed] == ["run-old-elsewhere"]
    assert "cannot be checked" in str(stale.closed[0].reason)


# ---- the host label --------------------------------------------------------


@pytest.fixture
def generated_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No ``PRISM_HOST_LABEL``, and a config directory of the test's own."""
    monkeypatch.delenv(HOST_LABEL_ENV, raising=False)
    config = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    return config / "prism" / "host_id"


def test_the_host_label_is_the_environment_variable_when_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(HOST_LABEL_ENV, "  lab-1\n")
    assert host_label() == "lab-1"
    log = TrialLog(tmp_path / "set.jsonl")
    recorder = RunRecorder(log=log, run_id="run-1", config_id=None, data_id="d", kind="new", selection_set_id=None)
    assert recorder.host == "lab-1", "the recorder's default host is the label, not the hostname"


@pytest.mark.parametrize("value", ["two words", "a/b", "x" * 65, "unavailable"])
def test_a_host_label_that_is_not_a_label_is_refused(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(HOST_LABEL_ENV, value)
    with pytest.raises(TrialLogError, match=HOST_LABEL_ENV):
        host_label()


def test_the_generated_label_is_created_once_and_read_back(generated_label: Path, monkeypatch) -> None:
    monkeypatch.setattr("socket.gethostname", lambda: "named-machine")

    first = host_label()

    assert re.fullmatch(r"host-[0-9a-f]{16}", first)
    assert host_id_path() == generated_label
    assert generated_label.read_text(encoding="utf-8") == first + "\n"
    assert host_label() == first, "a second call reads the file rather than minting a new label"
    assert "named-machine" not in first
    assert sorted(path.name for path in generated_label.parent.iterdir()) == ["host_id"], "no temporary left behind"


def test_an_unset_empty_or_relative_xdg_config_home_means_the_home_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = tmp_path / ".config" / "prism" / "host_id"
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert host_id_path() == expected
    for value in ("", "relative/config"):
        monkeypatch.setenv("XDG_CONFIG_HOME", value)
        assert host_id_path() == expected


def test_a_host_label_that_cannot_be_read_or_created_fails_loudly(generated_label: Path, monkeypatch) -> None:
    generated_label.parent.parent.mkdir(parents=True)
    generated_label.parent.write_text("a file where the prism config directory belongs\n", encoding="utf-8")
    with pytest.raises(TrialLogError, match="cannot read the host label"):
        host_label()

    generated_label.parent.unlink()

    def no_hard_links(source: str, target: str) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(trials.os, "link", no_hard_links)
    with pytest.raises(TrialLogError, match="cannot create the host label"):
        host_label()
    assert list(generated_label.parent.iterdir()) == [], "neither a label nor a temporary is left behind"


def test_a_label_file_that_holds_no_label_is_refused(generated_label: Path) -> None:
    generated_label.parent.mkdir(parents=True)
    generated_label.write_text("\n", encoding="utf-8")

    with pytest.raises(TrialLogError, match="holds the host label"):
        host_label()


def test_two_processes_creating_the_label_both_get_the_one_linked(generated_label: Path, monkeypatch) -> None:
    """The process that loses the race reads the winner's label instead of writing its own."""
    real_link = os.link

    def link_after_a_rival(source: str, target: str) -> None:
        Path(target).write_text("host-rival\n", encoding="utf-8")  # the rival links first
        real_link(source, target)

    monkeypatch.setattr(trials.os, "link", link_after_a_rival)

    assert host_label() == "host-rival"
    assert generated_label.read_text(encoding="utf-8") == "host-rival\n"
    assert sorted(path.name for path in generated_label.parent.iterdir()) == ["host_id"]


def test_reconcile_reads_rows_under_the_label_or_the_hostname_as_this_host(tmp_path: Path, monkeypatch) -> None:
    """Rows written before the label carry the hostname and stay reconcilable on the machine that wrote them."""
    monkeypatch.setenv(HOST_LABEL_ENV, "lab-1")
    monkeypatch.setattr("socket.gethostname", lambda: "legacy-box")
    assert this_host_names() == ("lab-1", "legacy-box")
    log = TrialLog(tmp_path / "set.jsonl")
    log.append(_row("run-legacy-dead", pid=_dead_pid(), host="legacy-box", heartbeat_at=_aged(4000)))
    log.append(_row("run-labelled-dead", pid=_dead_pid(), host="lab-1", heartbeat_at=_aged(4000)))
    log.append(_row("run-labelled-live", pid=os.getpid(), host="lab-1", heartbeat_at=_aged(4000)))
    log.append(_row("run-legacy-live", pid=os.getpid(), host="legacy-box", heartbeat_at=_aged(4000)))
    log.append(_row("run-elsewhere", pid=os.getpid(), host="another-host", heartbeat_at=utc_now()))

    report = reconcile_trial_log(log.path, max_heartbeat_age_seconds=900)

    assert {verdict.run_id: verdict.verdict for verdict in report.closed} == {
        "run-legacy-dead": "closed_incomplete",
        "run-labelled-dead": "closed_incomplete",
    }
    assert {verdict.run_id: verdict.verdict for verdict in report.in_flight} == {
        "run-labelled-live": "process_alive",
        "run-legacy-live": "process_alive",
        "run-elsewhere": "unverifiable_host",
    }
    written = [row for row in log.verify() if row["event"] == "incomplete"]
    assert [row["host"] for row in written] == ["lab-1", "lab-1"], "new rows carry the label, never the hostname"
    assert all("is gone on this host ('lab-1')" in str(row["reason"]) for row in written)
    assert not any("legacy-box" in json.dumps(row) for row in written)


def test_reconcile_of_a_log_with_no_open_run_needs_no_label(tmp_path: Path, generated_label: Path) -> None:
    generated_label.parent.parent.mkdir(parents=True)
    generated_label.parent.write_text("not a directory\n", encoding="utf-8")
    log = TrialLog(tmp_path / "set.jsonl")
    log.append(_row("run-1"))
    log.append(_row("run-1", "finished", summary={}))

    report = reconcile_trial_log(log.path)

    assert report.n_open == 0 and not report.closed and not report.in_flight


# ---- the hash chain --------------------------------------------------------


def test_the_hash_chain_detects_a_removed_row(tmp_path: Path) -> None:
    log = TrialLog(tmp_path / "set.jsonl")
    for number in range(4):
        log.append(_row(f"run-{number}"))
        log.append(_row(f"run-{number}", "finished", summary={"n": number}))
    trusted = log.verify()
    assert len(trusted) == 8

    lines = log.path.read_text(encoding="utf-8").splitlines()
    log.path.write_text("\n".join(lines[:3] + lines[4:]) + "\n", encoding="utf-8")

    with pytest.raises(TrialLogError, match="a row has been removed, reordered or rewritten") as excinfo:
        log.verify()
    assert f"{log.path}:4" in str(excinfo.value), "the break is reported at the successor of the removed row"


def test_the_hash_chain_detects_a_rewritten_row(tmp_path: Path) -> None:
    log = TrialLog(tmp_path / "set.jsonl")
    log.append(_row("run-1"))
    log.append(_row("run-1", "finished", summary={"periodic_sharpe": 0.01}))

    lines = log.path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["summary"]["periodic_sharpe"] = 0.9
    log.path.write_text(lines[0] + "\n" + json.dumps(tampered) + "\n", encoding="utf-8")

    with pytest.raises(TrialLogError, match="does not cover its content"):
        log.verify()


def test_one_chain_hash_implementation_serves_every_writer() -> None:
    """A second implementation is a second format, discovered when a log fails."""
    assert history_import.chain_hash is chain_hash


# ---- concurrency -----------------------------------------------------------


def _append_runs(job: tuple[str, str, int]) -> int:
    """Append ``count`` complete runs to one log, from one process."""
    path, tag, count = job
    log = TrialLog(Path(path))
    for number in range(count):
        run_id = f"{tag}-{number}"
        log.append(_row(run_id, host=tag))
        log.append(_row(run_id, "finished", summary={"worker": tag}, host=tag))
    return count


def test_two_concurrent_runs_do_not_corrupt_each_others_rows(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    workers = 4
    per_worker = 5
    jobs = [(str(path), f"worker{index}", per_worker) for index in range(workers)]

    with multiprocessing.get_context("fork").Pool(workers) as pool:
        assert sum(pool.map(_append_runs, jobs)) == workers * per_worker

    log = TrialLog(path)
    rows = log.verify()  # raises if any appender forked the chain
    assert len(rows) == workers * per_worker * 2
    starts = [row["run_id"] for row in rows if row["event"] == "started"]
    finishes = [row["run_id"] for row in rows if row["event"] == "finished"]
    assert sorted(starts) == sorted(finishes)
    assert len(set(starts)) == workers * per_worker, "every run appears exactly once"
    # Each row is written whole: a torn line would not parse, and every run's
    # start precedes its finish even with the appends interleaved.
    for run_id in set(starts):
        events = [row["event"] for row in rows if row["run_id"] == run_id]
        assert events == ["started", "finished"]


# ---- counting and cross-writer agreement -----------------------------------


def test_uncounted_kinds_are_attempts_but_not_independent_evidence(tmp_path: Path) -> None:
    log = TrialLog(tmp_path / "set.jsonl")
    log.append(_row("run-a", arm_id="arm_a"))
    log.append(_row("run-b", arm_id="arm_b", kind="reproduction"))
    log.append(_row("run-c", arm_id="arm_c", kind="replay"))
    log.append(_row("run-d", arm_id="arm_d", synthetic=True))
    log.append(_row("run-e", arm_id="arm_a"))  # the same arm run twice is one attempt

    counts = attempt_counts(log.verify(), registered=True)

    assert counts["attempt_basis"] == "arm_id"
    assert counts["n_trials_observed"] == 4
    assert counts["n_independent"] == 1
    assert counts["excluded_rows"] == {"replay": 1, "reproduction": 1, "synthetic": 1}


def test_imported_records_and_logs_satisfy_the_record_schema_and_the_chain(tmp_path: Path) -> None:
    """The two writers of ``record_schema = 1`` agree, and their logs verify here.

    ``prism.history_import`` writes records and rows for the legacy ledgers;
    the evaluator writes them for new evaluations. One schema and one chain, or
    the migration's history and the engine's evidence are two incompatible
    stores wearing one version number.
    """
    workspace = history_import.Workspace(tmp_path / "ws").ensure()
    repository = Path(__file__).resolve().parents[1]

    history_import.import_history(workspace, source_root=repository)

    records = [
        json.loads(line)
        for line in (workspace.results / "history_import" / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records, "the import wrote no record"
    for record in records:
        assert record["record_schema"] == RECORD_SCHEMA
        assert record_schema_problems(record) == [], record["ids"]["run_id"]
        validate_result_record(record)
    for path in sorted(workspace.trials.glob("*.jsonl")):
        assert TrialLog(path).verify(), path
