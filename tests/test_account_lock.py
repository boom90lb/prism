"""The account-keyed execution lock (§3.10, stage 5).

Two run directories can name one account, and the run-directory lock of
``tests/test_lockfile.py`` says nothing about that; this lock is keyed on the
account, which is the thing the broker sees.

What is asserted here is deliberately narrower than "no two writers ever":

* on one host the lock excludes a second writer, under both protocols;
* a holder that is gone is reclaimed, a holder that is alive is not;
* a host that is not the declared ``execution_host`` is refused outright;
* a *record from another host* is judged on heartbeat age alone, because its
  pid cannot be checked from here.

Nothing here can assert cross-host exclusion, because the lock does not
provide it — see the module docstring. The final test states that in code.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from prism.live import lockfile
from prism.live.lockfile import (
    DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    LockHandle,
    LockLost,
    LockRefused,
    account_execution_lock,
    account_lock_path,
    break_account_lock,
    process_start_token,
    read_lock_record,
    run_dir_lock,
)

THIS_HOST = socket.gethostname()
ACCOUNT = "paper"


@pytest.fixture(params=["flock", "o_excl"])
def protocol(request, monkeypatch) -> str:
    """Run the body under each lock protocol the module can take."""
    if request.param == "o_excl":
        monkeypatch.setattr(lockfile, "fcntl", None)
    elif lockfile.fcntl is None:  # pragma: no cover — non-POSIX platform
        pytest.skip("this platform has no fcntl, so there is no flock path to exercise")
    return request.param


@pytest.fixture
def no_fcntl(monkeypatch) -> None:
    """The platform stage 5 corrects: no ``fcntl``, so ``O_EXCL`` is the lock."""
    monkeypatch.setattr(lockfile, "fcntl", None)


@pytest.fixture
def no_liveness_probe(monkeypatch) -> None:
    """The platform where a pid cannot be asked about at all.

    ``os.kill(pid, 0)`` is the POSIX probe; on Windows ``os.kill`` terminates
    the process whatever signal it is handed, so the module never calls it
    there. Simulated, because the operating host is POSIX.
    """
    monkeypatch.setattr(lockfile, "LIVENESS_IS_CHECKABLE", False)


def _stamp(age_seconds: float = 0.0) -> str:
    moment = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _plant(lock_dir: Path, **overrides) -> Path:
    """Write a lock record for ``ACCOUNT`` by hand, as a crashed or foreign holder would leave it."""
    payload = {
        "schema": 1,
        "kind": "account",
        "key": ACCOUNT,
        "host": THIS_HOST,
        "pid": os.getpid(),
        "start_token": process_start_token(os.getpid()),
        "acquired_at": _stamp(1.0),
        "heartbeat_at": _stamp(0.0),
        "released_at": None,
    }
    payload.update(overrides)
    path = account_lock_path(ACCOUNT, lock_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _reaped_pid() -> int:
    """A pid that has exited. Paired everywhere with a start token that cannot
    match, so a recycled pid still reads as gone."""
    process = subprocess.Popen([sys.executable, "-c", ""])
    process.wait()
    return process.pid


def _gone_holder(lock_dir: Path, *, heartbeat_age: float, host: str = THIS_HOST) -> Path:
    return _plant(
        lock_dir,
        host=host,
        pid=_reaped_pid(),
        start_token="proc:0",
        heartbeat_at=_stamp(heartbeat_age),
    )


def _hold(lock_dir: Path) -> AbstractContextManager[LockHandle]:
    return account_execution_lock(ACCOUNT, THIS_HOST, lock_dir=lock_dir)


# --- exclusion on one host -------------------------------------------------


def test_a_second_acquisition_for_the_same_account_fails(tmp_path, protocol) -> None:
    with _hold(tmp_path):
        with pytest.raises(LockRefused, match="another prism process holds"):
            with _hold(tmp_path):
                pass  # pragma: no cover — must not be reached


def test_a_refused_acquisition_never_yields(tmp_path, protocol) -> None:
    """There is no platform on which a failed acquisition proceeds.

    The ``fcntl``-absent platform is simulated rather than reasoned about: the
    branch this replaces logged a warning and yielded, which would pass every
    other test in this file.
    """
    entered: list[str] = []
    with _hold(tmp_path):
        with pytest.raises(LockRefused):
            with _hold(tmp_path):
                entered.append(protocol)  # pragma: no cover
    assert entered == []


def test_two_run_directories_one_account(tmp_path, protocol) -> None:
    """The scenario the run-directory lock cannot see (§6 acceptance, "Two run directories, one account")."""
    first, second = tmp_path / "run-a", tmp_path / "run-b"
    with run_dir_lock(first), _hold(tmp_path / "locks"):
        # A second run directory is free — that lock is keyed on the directory.
        with run_dir_lock(second):
            # The account is not.
            with pytest.raises(LockRefused):
                with _hold(tmp_path / "locks"):
                    pass  # pragma: no cover


def test_the_lock_is_released_after_the_body(tmp_path, protocol) -> None:
    with _hold(tmp_path):
        pass
    with _hold(tmp_path):
        pass


def test_the_lock_is_released_when_the_holding_process_raises(tmp_path, protocol) -> None:
    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with _hold(tmp_path):
            raise Boom("settle blew up")
    # Not wedged: settle writes state and ledgers inside the lock, and an
    # exception there must not lock the account out of its next invocation.
    with _hold(tmp_path) as handle:
        assert handle.record.released_at is None


def test_release_is_recorded_and_the_file_is_never_unlinked(tmp_path, protocol) -> None:
    with _hold(tmp_path):
        path = account_lock_path(ACCOUNT, tmp_path)
    assert path.exists(), "unlinking races a concurrent opener (§3.10)"
    record = read_lock_record(path)
    assert record is not None and record.released_at is not None


def test_the_record_carries_the_pid_the_host_and_a_heartbeat(tmp_path, protocol) -> None:
    with _hold(tmp_path) as handle:
        record = read_lock_record(account_lock_path(ACCOUNT, tmp_path))
        assert record is not None
        assert (record.kind, record.key) == ("account", ACCOUNT)
        assert record.pid == os.getpid() == handle.pid
        assert record.host == THIS_HOST
        assert record.start_token == process_start_token(os.getpid())
        beat = handle.heartbeat()
        assert beat > record.heartbeat_at
        assert read_lock_record(account_lock_path(ACCOUNT, tmp_path)).heartbeat_at == beat


# --- the declared execution host ------------------------------------------


def test_an_undeclared_execution_host_is_refused(tmp_path, protocol) -> None:
    with pytest.raises(LockRefused, match="undeclared host"):
        with account_execution_lock(ACCOUNT, "some-other-box", lock_dir=tmp_path):
            pass  # pragma: no cover
    assert not account_lock_path(ACCOUNT, tmp_path).exists(), "refused before any file is created"


def test_an_account_id_that_is_a_path_is_refused(tmp_path, protocol) -> None:
    with pytest.raises(ValueError, match="single path segment"):
        with account_execution_lock("../elsewhere", THIS_HOST, lock_dir=tmp_path):
            pass  # pragma: no cover


# --- staleness -------------------------------------------------------------


def test_a_stale_lock_whose_process_is_gone_is_reclaimed(tmp_path, no_fcntl) -> None:
    path = _gone_holder(tmp_path, heartbeat_age=DEFAULT_MAX_HEARTBEAT_AGE_SECONDS + 60)
    with _hold(tmp_path) as handle:
        assert handle.pid == os.getpid()
    assert read_lock_record(path).key == ACCOUNT


def test_a_stale_lock_whose_process_still_lives_is_not_reclaimed(tmp_path, no_fcntl) -> None:
    # This process's own pid and start token stand in for a live holder that
    # has stopped beating: a long decision, or a stopped scheduler.
    _plant(tmp_path, heartbeat_at=_stamp(DEFAULT_MAX_HEARTBEAT_AGE_SECONDS + 60))
    with pytest.raises(LockRefused, match="that process is running on this host"):
        with _hold(tmp_path):
            pass  # pragma: no cover


def test_a_gone_process_with_a_fresh_heartbeat_is_not_reclaimed_without_fcntl(tmp_path, no_fcntl) -> None:
    """Both conditions are required (§3.10): stale heartbeat *and* a gone process."""
    _gone_holder(tmp_path, heartbeat_age=1.0)
    with pytest.raises(LockRefused, match="a pid can be recycled"):
        with _hold(tmp_path):
            pass  # pragma: no cover


def test_a_gone_process_is_reclaimed_immediately_under_flock(tmp_path) -> None:
    """The kernel mutex is stronger than the heartbeat and is allowed to be.

    A crashed holder's ``flock`` died with it, so waiting out the heartbeat
    window would refuse a lock nothing holds. That is the whole difference
    between the two protocols, and it is why ``fcntl`` is not optional here.
    """
    if lockfile.fcntl is None:  # pragma: no cover — non-POSIX platform
        pytest.skip("no fcntl on this platform")
    _gone_holder(tmp_path, heartbeat_age=1.0)
    with _hold(tmp_path) as handle:
        assert handle.pid == os.getpid()


def test_a_live_record_without_the_kernel_lock_is_refused(tmp_path) -> None:
    """A live holder is never displaced, even when it does not hold the flock."""
    if lockfile.fcntl is None:  # pragma: no cover — non-POSIX platform
        pytest.skip("no fcntl on this platform")
    _plant(tmp_path)  # this process's pid and start token, beating now
    with pytest.raises(LockRefused, match="that process is running on this host"):
        with _hold(tmp_path):
            pass  # pragma: no cover


def test_a_record_from_another_host_is_judged_on_heartbeat_age_alone(tmp_path, protocol) -> None:
    """Its pid cannot be checked from here, so age is all there is."""
    _plant(tmp_path, host="another-box", pid=1, start_token="proc:1", heartbeat_at=_stamp(60))
    with pytest.raises(LockRefused, match="not this host"):
        with _hold(tmp_path):
            pass  # pragma: no cover

    _plant(
        tmp_path,
        host="another-box",
        pid=1,
        start_token="proc:1",
        heartbeat_at=_stamp(DEFAULT_MAX_HEARTBEAT_AGE_SECONDS + 60),
    )
    with _hold(tmp_path) as handle:
        assert handle.host == THIS_HOST


def test_a_declared_heartbeat_age_shorter_than_the_default_is_honoured(tmp_path, no_fcntl) -> None:
    _gone_holder(tmp_path, heartbeat_age=30.0)
    with pytest.raises(LockRefused):
        with account_execution_lock(ACCOUNT, THIS_HOST, lock_dir=tmp_path, max_heartbeat_age_seconds=60.0):
            pass  # pragma: no cover
    with account_execution_lock(ACCOUNT, THIS_HOST, lock_dir=tmp_path, max_heartbeat_age_seconds=10.0):
        pass


def test_an_unreadable_record_is_refused_until_it_goes_quiet(tmp_path, no_fcntl) -> None:
    """A lock file with no readable record is somebody's, until nothing touches it.

    ``O_EXCL`` creation followed by a crash before the write leaves exactly
    this. Refusing on sight would be a lock nobody can clear; reclaiming on
    sight would race the writer that is mid-create.
    """
    path = account_lock_path(ACCOUNT, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    with pytest.raises(LockRefused, match="no readable record"):
        with _hold(tmp_path):
            pass  # pragma: no cover
    old = datetime.now(timezone.utc).timestamp() - (DEFAULT_MAX_HEARTBEAT_AGE_SECONDS + 60)
    os.utime(path, (old, old))
    with _hold(tmp_path):
        pass


# --- what the lock does not enforce ---------------------------------------


def test_a_holder_whose_lock_was_reclaimed_finds_out_at_its_next_heartbeat(tmp_path, no_fcntl) -> None:
    """Without a kernel mutex a reclaim cannot be prevented, only detected.

    A holder on the ``O_EXCL`` path is not notified when another writer renames
    a fresh lock over its file. It finds out when it next beats — which is why
    the heartbeat verifies the file is still the one it acquired.
    """
    with pytest.raises(LockLost, match="reclaimed the lock"):
        with _hold(tmp_path) as handle:
            handle.heartbeat()
            replacement = tmp_path / "stolen"
            replacement.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "kind": "account",
                        "key": ACCOUNT,
                        "host": "another-box",
                        "pid": 1,
                        "start_token": "proc:1",
                        "acquired_at": _stamp(),
                        "heartbeat_at": _stamp(),
                        "released_at": None,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(replacement, account_lock_path(ACCOUNT, tmp_path))
            handle.heartbeat()


def test_a_second_lock_directory_excludes_nothing(tmp_path, protocol) -> None:
    """Stated in code because an operator must not assume otherwise.

    Two lock directories on one host, or one per host on shared storage, are
    two different locks for one account. Nothing in this module can see that;
    cross-host exclusion rests on ``execution_host`` in the authorization
    record, and on nothing else.
    """
    with _hold(tmp_path / "locks-a"):
        with _hold(tmp_path / "locks-b"):
            pass


def test_a_pid_cannot_be_asked_about_where_there_is_no_probe(no_liveness_probe) -> None:
    """None, not False: "unknown" must not read as "gone"."""
    assert lockfile.process_is_alive(os.getpid()) is None


def test_without_a_liveness_probe_a_beating_holder_is_still_refused(tmp_path, no_fcntl, no_liveness_probe) -> None:
    _plant(tmp_path, heartbeat_at=_stamp(1.0))
    with pytest.raises(LockRefused, match="cannot be asked"):
        with _hold(tmp_path):
            pass  # pragma: no cover


def test_without_a_liveness_probe_a_live_holder_that_stops_beating_is_displaced(
    tmp_path, no_fcntl, no_liveness_probe
) -> None:
    """The honest weakness, asserted rather than left to be discovered.

    The planted record is this process: alive, and provably so on POSIX. With
    no probe available the lock cannot tell it from a corpse, so once the
    heartbeat goes stale the lock is taken. Heartbeat age alone decides, the
    same rule as for a holder on another host — and it is why a holder on such
    a platform must beat.
    """
    _plant(tmp_path, heartbeat_at=_stamp(DEFAULT_MAX_HEARTBEAT_AGE_SECONDS + 60))
    assert lockfile.LIVENESS_IS_CHECKABLE is False
    with _hold(tmp_path) as handle:
        assert handle.pid == os.getpid()


def test_a_lock_record_at_an_unknown_schema_is_refused(tmp_path, protocol) -> None:
    """A newer engine's lock is a holder, not garbage to age out.

    An unreadable record is displaced once nothing has touched it for a
    heartbeat window; a record at a schema this build does not know is not
    unreadable in that sense — some other build wrote it on purpose — so it is
    refused outright, on both protocols.
    """
    _plant(tmp_path, schema=2)
    with pytest.raises(LockRefused, match="lock schema 2"):
        with _hold(tmp_path):
            pass  # pragma: no cover


# --- the manual break (§3.10: who broke it, when, and why) -----------------


def test_breaking_a_lock_records_before_it_takes_effect(tmp_path, protocol) -> None:
    """The order is the control: the account of the break precedes the break.

    A foreign-host holder with a fresh heartbeat is refused by every automatic
    path — heartbeat age is all this host can judge it on — which is exactly
    the lock `--break-lock` exists for.
    """
    planted = _plant(tmp_path, host="some-other-host", pid=4321)
    before = planted.read_bytes()
    written: list[str] = []

    with pytest.raises(LockRefused):  # the automatic path will not touch it
        with _hold(tmp_path):
            pass  # pragma: no cover

    path, displaced = break_account_lock(
        ACCOUNT,
        reason="the declared host was rebuilt and its lock outlived it",
        broken_by="operator@here:1",
        lock_dir=tmp_path,
        before_break=lambda record: written.append(record.describe()),
    )
    assert path == planted
    assert displaced is not None and displaced.host == "some-other-host"
    assert written == [displaced.describe()]  # ran, and ran with the record being displaced
    assert planted.read_bytes() != before
    assert read_lock_record(planted).released_at is not None

    # The break took effect: the next acquisition succeeds where it just failed.
    with _hold(tmp_path) as handle:
        assert handle.pid == os.getpid()


def test_a_break_whose_record_cannot_be_written_does_not_break_the_lock(tmp_path, protocol) -> None:
    """If the account of the break fails, the break does not happen."""
    planted = _plant(tmp_path, host="some-other-host", pid=4321)
    before = planted.read_bytes()

    def refuse_to_record(record) -> None:
        raise OSError("the run directory is read-only")

    with pytest.raises(OSError, match="read-only"):
        break_account_lock(
            ACCOUNT,
            reason="a reason nobody could record",
            broken_by="operator@here:1",
            lock_dir=tmp_path,
            before_break=refuse_to_record,
        )
    assert planted.read_bytes() == before


def test_a_lock_held_by_a_live_process_on_this_host_is_not_broken(tmp_path, protocol) -> None:
    """Not a stale lock — a live writer. Breaking it would make two of them."""
    with _hold(tmp_path):
        with pytest.raises(LockRefused, match="running on this host"):
            break_account_lock(
                ACCOUNT,
                reason="I am impatient",
                broken_by="operator@here:1",
                lock_dir=tmp_path,
                before_break=lambda record: pytest.fail("the break was recorded before it was refused"),
            )


def test_breaking_a_lock_that_nothing_holds_is_refused(tmp_path, protocol) -> None:
    with pytest.raises(LockRefused, match="no lock file"):
        break_account_lock(ACCOUNT, reason="tidying up", broken_by="operator@here:1", lock_dir=tmp_path)
    with _hold(tmp_path):
        pass  # released, and a released lock holds nothing either
    with pytest.raises(LockRefused, match="nothing to break"):
        break_account_lock(ACCOUNT, reason="tidying up", broken_by="operator@here:1", lock_dir=tmp_path)


def test_a_break_with_no_reason_is_refused(tmp_path, protocol) -> None:
    _plant(tmp_path, host="some-other-host", pid=4321)
    with pytest.raises(LockRefused, match="requires a reason"):
        break_account_lock(ACCOUNT, reason="   ", broken_by="operator@here:1", lock_dir=tmp_path)
