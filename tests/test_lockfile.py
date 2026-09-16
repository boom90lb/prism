"""live/ run-directory writer lock (SPEC §7.7 durable-state protection).

The write-ahead protocol survives crashes, not concurrency: two writers on
one run directory (nightly loop + morning sweep + a manual invocation) can
interleave decide/settle. The lock must be exclusive, non-blocking, loud on
contention, and reusable after release.

Every contention case runs twice: once over ``flock``, and once with
``fcntl`` removed from the module, which is the platform the ``O_EXCL``
protocol exists for (§3.10). The old warn-and-proceed branch would have
passed the first pass and yielded control in the second; that is the whole
point of running both.
"""

from __future__ import annotations

import os
import socket

import pytest

from prism.live import lockfile
from prism.live.lockfile import LOCK_FILENAME, LockRefused, read_lock_record, run_dir_lock


@pytest.fixture(params=["flock", "o_excl"])
def protocol(request, monkeypatch) -> str:
    """Run the body under each lock protocol the module can take."""
    if request.param == "o_excl":
        monkeypatch.setattr(lockfile, "fcntl", None)
    elif lockfile.fcntl is None:  # pragma: no cover — non-POSIX platform
        pytest.skip("this platform has no fcntl, so there is no flock path to exercise")
    return request.param


def test_lock_acquires_creates_file_and_releases(tmp_path, protocol) -> None:
    run_dir = tmp_path / "run"
    with run_dir_lock(run_dir):
        assert (run_dir / LOCK_FILENAME).exists()
    # Released: a second acquisition succeeds.
    with run_dir_lock(run_dir):
        pass


def test_second_concurrent_writer_raises_loudly(tmp_path, protocol) -> None:
    # flock is per open-file-description, so a nested acquisition models a
    # second process exactly; under O_EXCL the file simply already exists.
    with run_dir_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another prism process holds"):
            with run_dir_lock(tmp_path):
                pass  # pragma: no cover — must not be reached


def test_contention_failure_does_not_break_the_holder(tmp_path, protocol) -> None:
    with run_dir_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with run_dir_lock(tmp_path):
                pass  # pragma: no cover
        # The failed second acquisition must not have released the holder's lock.
        with pytest.raises(RuntimeError):
            with run_dir_lock(tmp_path):
                pass  # pragma: no cover
    with run_dir_lock(tmp_path):
        pass


def test_a_refused_acquisition_never_yields(tmp_path, protocol) -> None:
    """No platform has a branch where the body runs without the lock.

    This is the correction stage 5 makes: the module used to log a warning and
    yield when ``fcntl`` was missing, which claimed a protection that was not
    there. The absence is simulated rather than reasoned about.
    """
    entered = []
    with run_dir_lock(tmp_path):
        with pytest.raises(LockRefused):
            with run_dir_lock(tmp_path):
                entered.append(protocol)  # pragma: no cover
    assert entered == []


def test_the_lock_file_records_its_holder(tmp_path, protocol) -> None:
    with run_dir_lock(tmp_path) as handle:
        record = read_lock_record(tmp_path / LOCK_FILENAME)
        assert record is not None
        assert record.kind == "run_dir"
        assert record.pid == os.getpid() == handle.pid
        assert record.host == socket.gethostname()
        assert record.released_at is None
        assert record.heartbeat_at >= record.acquired_at


def test_release_is_recorded_and_the_file_is_never_unlinked(tmp_path, protocol) -> None:
    with run_dir_lock(tmp_path):
        pass
    path = tmp_path / LOCK_FILENAME
    assert path.exists(), "unlinking the lock file races a concurrent opener; release is recorded in it"
    record = read_lock_record(path)
    assert record is not None and record.released_at is not None


def test_the_lock_is_released_when_the_body_raises(tmp_path, protocol) -> None:
    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with run_dir_lock(tmp_path):
            raise Boom("the decision blew up mid-cycle")
    with run_dir_lock(tmp_path):
        pass  # the next invocation is not wedged by the failed one
