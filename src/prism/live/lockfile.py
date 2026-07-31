"""Advisory cross-process lock on a run directory (SPEC.md §7.7).

The write-ahead protocol makes the loop safe against *crashes*; nothing made
it safe against *concurrency*. The nightly loop, the morning sweep, and any
manual invocation all mutate the same ``state.json``, and two writers
interleaving decide/settle can tear the pending set or double-submit — the
runtime analogue of the single-checkout hazard AGENTS.md §6 records for git.

The lock is ``flock`` on ``{run_dir}/.lock``, non-blocking and fail-loud: a
second writer raises immediately (N7) instead of queueing, because "wait
your turn" is not a remedy when the first writer may be mid-decision on the
same book. Read-only consumers (the doctor, the monitor) do not take it.
The lock file is left in place on release — unlinking under ``flock`` races
a concurrent opener; a stale file with no holder locks nothing.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover — non-POSIX platform
    fcntl = None  # type: ignore[assignment]

LOCK_FILENAME = ".lock"


@contextlib.contextmanager
def run_dir_lock(run_dir: Path) -> Generator[None, None, None]:
    """Hold the exclusive writer lock for ``run_dir``, or raise loudly.

    Raises :class:`RuntimeError` when another process already holds it. On a
    platform without ``fcntl`` the lock is a loud no-op — the current
    operator environment is POSIX (WSL), and a silent pass elsewhere would
    claim a protection that is not there.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / LOCK_FILENAME
    if fcntl is None:  # pragma: no cover — non-POSIX platform
        logger.warning(
            "fcntl unavailable on this platform — the run-dir writer lock %s is NOT enforced",
            path,
        )
        yield
        return
    handle = open(path, "w", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"another prism process holds {path} — refusing a second concurrent "
                "writer on this run directory (N7; the write-ahead protocol protects "
                "against crashes, not concurrent writers)"
            ) from exc
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield
    finally:
        handle.close()  # closing the fd releases the flock
