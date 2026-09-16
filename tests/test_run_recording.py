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
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from prism import history_import
from prism.validation.trials import (
    RECORD_SCHEMA,
    TrialLog,
    TrialLogError,
    attempt_counts,
    chain_hash,
    reconcile_trial_log,
    record_schema_problems,
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
