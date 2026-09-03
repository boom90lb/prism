"""live/ run-directory writer lock (SPEC §7.7 durable-state protection).

The write-ahead protocol survives crashes, not concurrency: two writers on
one run directory (nightly loop + morning sweep + a manual invocation) can
interleave decide/settle. The lock must be exclusive, non-blocking, loud on
contention, and reusable after release.
"""

from __future__ import annotations

import pytest

from prism.live.lockfile import LOCK_FILENAME, run_dir_lock


def test_lock_acquires_creates_file_and_releases(tmp_path) -> None:
    run_dir = tmp_path / "run"
    with run_dir_lock(run_dir):
        assert (run_dir / LOCK_FILENAME).exists()
    # Released: a second acquisition succeeds.
    with run_dir_lock(run_dir):
        pass


def test_second_concurrent_writer_raises_loudly(tmp_path) -> None:
    # flock is per open-file-description, so a nested acquisition models a
    # second process exactly.
    with run_dir_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another prism process holds"):
            with run_dir_lock(tmp_path):
                pass  # pragma: no cover — must not be reached


def test_contention_failure_does_not_break_the_holder(tmp_path) -> None:
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
