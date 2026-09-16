"""Advisory writer locks for the engine's mutating paths (§3.10; SPEC.md §7.7).

Two locks ride one protocol. ``run_dir_lock`` excludes a second writer on one
run directory: the write-ahead protocol makes the loop safe against *crashes*
and never against *concurrency*, and the nightly loop, the morning sweep and
any manual invocation all mutate the same ``state.json``, where two writers
interleaving decide/settle tear the pending set or double-submit — the runtime
analogue of the single-checkout hazard AGENTS.md §6 records for git.
``account_execution_lock`` excludes a second writer on one *account*, which is
what the broker actually sees: two run directories on one host can name one
account, and the run-directory lock says nothing about that. Both are held.

Be exact about what this buys, because an operator will rely on it.

**What it enforces.** On one host, where ``fcntl`` exists, ``flock`` on the
lock file is a kernel mutex. A second acquisition fails immediately rather than
queueing — "wait your turn" is not a remedy when the first writer may be
mid-decision on the same book — and the kernel drops the lock when the holding
process dies, however it dies. That is the whole of the strong guarantee: one
host, one filesystem, live processes.

**What it enforces without ``fcntl``.** The lock file is created with
``O_EXCL``, and that creation *is* the lock; there is no warn-and-proceed, on
any platform, in any branch. Nothing releases an ``O_EXCL`` lock when its
holder is killed, so a dead holder's lock is reclaimed only once its heartbeat
is older than ``max_heartbeat_age_seconds`` **and** the recorded process is
gone from this host (§3.10). Between a hard kill and the end of that window the
account is locked by a process that no longer exists. That is a real cost of
the weaker protocol, not a defect to paper over: the engine refuses, and an
operator who has confirmed no writer is running removes the file.

**What it merely documents.** Across hosts it enforces nothing. The lock
directory is host-local by construction, and putting it on a shared filesystem
would not help: cross-host exclusion rests entirely on ``execution_host`` in
the authorization record, which this module checks against
``socket.gethostname()`` at acquisition, in both modes. That check refuses a
host that is not the declared one. It cannot refuse a *second* host whose own
record names itself.

**What it cannot detect.** Two hosts each holding a record naming themselves
for the same account: each acquires its own lock, each decides, and because
client order ids are ``{prefix}{decision_bar}:{symbol}`` one host's order is
accepted while the other's is treated as an already-submitted duplicate, which
the protocol reads as success. Nothing here surfaces that at submission time;
the reconciliation comparison of open orders on quantity and side does, after
the fact (§3.11). Nor is the file lock reliable on every filesystem: neither
``flock`` nor ``O_EXCL`` is atomic on NFS without a modern server, or on some
Windows-mounted paths — which matters here, because this engine runs under WSL
and ``/mnt/c`` is such a path. A lock directory on one of those excludes
nothing, silently. Keep it on local disk.

Two further limits sit in the liveness check the ``O_EXCL`` path leans on.
Where no process start token can be read (any platform without ``/proc``), a
recycled pid reads as the original holder, so a dead holder's lock is refused
rather than reclaimed — the safe direction, and still an operator's problem to
clear. Where the pid cannot be asked about at all — Windows, where ``os.kill``
ignores the signal and terminates the process, so it is never called there —
heartbeat age alone decides, exactly as it does for a holder recorded on
another host, and a *live* holder that stops beating for longer than
``max_heartbeat_age_seconds`` is displaced. On that platform the heartbeat is
the entire protection and a holder that does not beat has none.

The lock file is never unlinked by the engine, on either path: unlinking races
a concurrent opener, and a stale file with no holder locks nothing. Release is
recorded *in* the record instead. Read-only consumers (the doctor, the monitor)
take neither lock. Neither lock says anything about whether the holder is
authorized to trade the account; that is ``live/authorization.py`` (§3.10).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import tempfile
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generator, Mapping

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover — non-POSIX platform
    fcntl = None  # type: ignore[assignment]

#: The run-directory lock's filename, unchanged since SPEC §7.7.
LOCK_FILENAME = ".lock"

#: One lock file per account, in the host-local lock directory.
ACCOUNT_LOCK_SUFFIX = ".lock"

#: On-disk schema of a lock record. A build that finds a version it does not
#: know refuses: a newer engine's lock is a real holder, not garbage to step
#: over, and guessing which fields mean what is how two writers end up on one
#: account.
LOCK_SCHEMA = 1

#: ``[freshness].max_heartbeat_age_seconds`` of the execution profile (§3.9).
#: The default is a fallback for callers that hold no profile; the operating
#: paths pass the declared value.
DEFAULT_MAX_HEARTBEAT_AGE_SECONDS = 900.0

_MAX_RECORD_BYTES = 65536


class LockRefused(RuntimeError):
    """The lock is held, or its state cannot be verified, so we do not run.

    A ``RuntimeError`` because every existing caller of ``run_dir_lock``
    already turns that into a refusal at the CLI boundary.
    """


class LockLost(RuntimeError):
    """The lock file we hold is no longer the file at our path.

    Only reachable where the ``O_EXCL`` protocol is in use or the lock
    directory is shared: someone reclaimed the lock out from under a live
    holder. The holder finds out at its next heartbeat, not never.
    """


def utc_now_text() -> str:
    """The timestamp format every lock record carries."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


#: Whether a pid's liveness can be asked about at all on this platform.
#: ``os.kill(pid, 0)`` is the POSIX probe; on Windows ``os.kill`` ignores the
#: signal and calls ``TerminateProcess``, so a "liveness check" there would
#: kill the process it asked about. It is never called there, and a recorded
#: holder is then judged on heartbeat age alone — the same rule, and the same
#: weakness, as a holder recorded on another host.
LIVENESS_IS_CHECKABLE = os.name != "nt"


def process_is_alive(pid: Any) -> bool | None:
    """True when a process with ``pid`` exists on *this* host; None when this
    platform cannot be asked.

    ``EPERM`` counts as alive: the process exists and belongs to someone else.
    A pid that cannot be interpreted as a pid is not a live process — a
    malformed record is handled by the staleness rules, not here. None is the
    Windows answer, and callers must not read it as either yes or no.

    This is checkable on one host only. It duplicates
    ``prism.validation.trials.process_is_alive`` deliberately: a lock in the
    live write path does not import the evaluation stack (which pulls numpy
    and pandas) to ask whether a pid exists.
    """
    if not LIVENESS_IS_CHECKABLE:  # pragma: no cover — non-POSIX platform
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def process_start_token(pid: Any) -> str | None:
    """A token identifying *this* run of ``pid``, or None when unobtainable.

    Field 22 of ``/proc/<pid>/stat`` is the process start time in clock ticks
    since boot, so ``proc:<starttime>`` distinguishes a live holder from an
    unrelated process that inherited its pid. None where ``/proc`` is absent,
    and the caller must then treat a live pid as the original holder, which
    refuses rather than reclaims.
    """
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
    except (OSError, TypeError, ValueError):
        return None
    try:
        fields = raw[raw.rindex(")") + 1 :].split()
        return f"proc:{fields[19]}"
    except (IndexError, ValueError):
        return None


def default_lock_dir() -> Path:
    """The host-local directory the account locks live in.

    ``{tempdir}/prism-locks``, created 0700. It is deliberately not the run
    directory: the run directory may sit on shared storage, where neither
    ``flock`` nor ``O_EXCL`` is dependable, and one account may be reached
    from more than one run directory on the same host.

    The one thing a deployment must guarantee is that every invocation on the
    host resolves the *same* directory. ``tempfile.gettempdir()`` honours
    ``TMPDIR``, so a cron job and an interactive shell with different
    environments would take different locks and exclude nothing. Pass
    ``lock_dir`` explicitly wherever that is a real risk.
    """
    return Path(tempfile.gettempdir()) / "prism-locks"


def account_lock_path(account_id: str, lock_dir: Path | str | None = None) -> Path:
    """Where the execution lock for ``account_id`` lives on this host."""
    directory = Path(lock_dir).expanduser() if lock_dir is not None else default_lock_dir()
    return directory / f"{_account_segment(account_id)}{ACCOUNT_LOCK_SUFFIX}"


def _account_segment(account_id: str) -> str:
    """Validate that ``account_id`` names one directory entry, not a path.

    A lock whose file can be steered out of the lock directory by its own key
    is not an exclusion boundary.
    """
    if not isinstance(account_id, str) or not account_id or account_id in (".", ".."):
        raise ValueError(f"account id must be a non-empty name; got {account_id!r}")
    if os.sep in account_id or (os.altsep and os.altsep in account_id) or "/" in account_id:
        raise ValueError(f"account id must be a single path segment (no separators); got {account_id!r}")
    return account_id


@dataclass(frozen=True)
class LockRecord:
    """What a lock file says about the process that holds it.

    ``host`` and ``pid`` name the holder; ``start_token`` is what defeats pid
    reuse; ``heartbeat_at`` is what a *remote* reclaimer has to go on, since it
    cannot check the pid. ``released_at`` is how a normal release is recorded
    without unlinking the file.
    """

    schema: int
    kind: str
    key: str
    host: str
    pid: int
    start_token: str | None
    acquired_at: str
    heartbeat_at: str
    released_at: str | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "key": self.key,
            "host": self.host,
            "pid": self.pid,
            "start_token": self.start_token,
            "acquired_at": self.acquired_at,
            "heartbeat_at": self.heartbeat_at,
            "released_at": self.released_at,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LockRecord":
        """Parse a record, or raise ``ValueError`` on anything unrecognised."""
        if int(payload["schema"]) != LOCK_SCHEMA:
            raise ValueError(f"unknown lock record schema {payload['schema']!r}")
        return cls(
            schema=LOCK_SCHEMA,
            kind=str(payload["kind"]),
            key=str(payload["key"]),
            host=str(payload["host"]),
            pid=int(payload["pid"]),
            start_token=None if payload.get("start_token") is None else str(payload["start_token"]),
            acquired_at=str(payload["acquired_at"]),
            heartbeat_at=str(payload["heartbeat_at"]),
            released_at=None if payload.get("released_at") is None else str(payload["released_at"]),
        )

    def describe(self) -> str:
        return (
            f"held by pid {self.pid} on host {self.host!r} since {self.acquired_at} "
            f"(heartbeat {self.heartbeat_at})"
        )


def read_lock_record(path: Path | str) -> LockRecord | None:
    """The record at ``path``, or None when it is absent, empty or malformed.

    For diagnostics — the doctor and ``prism run``'s refusal messages. It takes
    no lock, so what it returns may be one heartbeat out of date.
    """
    try:
        raw = Path(path).read_bytes()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return None
    return _parse_record(raw)


def _parse_record(raw: bytes) -> LockRecord | None:
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    try:
        return LockRecord.from_mapping(payload)
    except (KeyError, TypeError, ValueError):
        return None


def _record_blob(record: LockRecord) -> bytes:
    return (json.dumps(record.to_mapping(), sort_keys=True) + "\n").encode("utf-8")


def _write_record(fd: int, record: LockRecord) -> None:
    """Rewrite the record in place, on the descriptor that holds the lock.

    In place, not write-temp-and-rename: on the ``fcntl`` path the ``flock``
    belongs to this descriptor's open file description, and renaming a new file
    over the path would leave us holding a lock on an orphaned inode while the
    next acquirer locked the new one. ``os.pread``/``os.pwrite`` are Unix-only,
    so this seeks — the ``O_EXCL`` path exists for platforms that are not Unix.
    """
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, _record_blob(record))
    os.fsync(fd)


def _read_raw(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    return os.read(fd, _MAX_RECORD_BYTES)


def _read_record(fd: int) -> LockRecord | None:
    return _parse_record(_read_raw(fd))


def _foreign_schema(raw: bytes) -> int | None:
    """The schema of a well-formed record this build cannot read, else None.

    A record at an unknown version is the one kind of unparseable file that is
    evidence of a holder rather than evidence of a crash: some other build
    wrote it deliberately. It is refused, not aged out.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    try:
        version = int(payload["schema"])
    except (KeyError, TypeError, ValueError):
        return None
    return None if version == LOCK_SCHEMA else version


def _refuse_a_foreign_schema(raw: bytes, path: Path, context: str) -> None:
    version = _foreign_schema(raw)
    if version is not None:
        raise LockRefused(
            _refusal(
                path,
                f"its record is at lock schema {version} and this build reads schema {LOCK_SCHEMA}; a newer "
                "engine wrote it and may be holding it",
                context,
            )
        )


def _age_seconds(stamp: str, *, now: datetime) -> float | None:
    """Seconds since ``stamp``, or None when it cannot be read as a time.

    An unreadable heartbeat is treated by the callers as infinitely old, the
    same rule the trial-log reconciler applies to a row with no usable
    heartbeat: a timestamp nobody can parse is not evidence of a live holder.
    """
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        return None
    return (now - moment).total_seconds()


def _holder_is_live(record: LockRecord) -> bool | None:
    """True when the recorded pid is running *and* is the recorded process;
    False when it is gone; None when this platform cannot be asked.

    Conservative in the refusing direction: where no start token can be read,
    on either side, a live pid counts as the original holder.
    """
    alive = process_is_alive(record.pid)
    if alive is None or alive is False:  # pragma: no branch — None only off POSIX
        return alive
    if record.start_token is None:
        return True
    current = process_start_token(record.pid)
    return current is None or current == record.start_token


def _displaceable(
    record: LockRecord | None,
    *,
    path: Path,
    this_host: str,
    max_heartbeat_age_seconds: float,
    kernel_mutex_held: bool,
) -> tuple[bool, str]:
    """May this process take the lock, and why (or why not)?

    ``kernel_mutex_held`` is True on the ``flock`` path, where the kernel has
    already proved no live process on this host holds the lock; there the
    record only has to rule out a *remote* holder, which is why a crashed
    holder costs no waiting there and costs a full heartbeat window on the
    ``O_EXCL`` path.
    """
    now = datetime.now(timezone.utc)
    limit = float(max_heartbeat_age_seconds)
    if record is None:
        # An empty or unreadable lock file is either brand new, or a holder
        # that died between the create and the write. Under the kernel mutex
        # that is settled: whoever wrote it is not holding the lock now.
        if kernel_mutex_held:
            return True, "the lock file carries no readable record and the kernel lock on this host is free"
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return True, "the lock file vanished between the failed create and the read"
        age = (now - mtime).total_seconds()
        if age > limit:
            return True, (
                f"the lock file carries no readable record and has not been written for {age:.0f}s "
                f"(limit {limit:.0f}s), so no live holder is maintaining it"
            )
        return False, (
            f"the lock file carries no readable record and was written {age:.0f}s ago (limit {limit:.0f}s); "
            "a holder may be mid-write. Refusing rather than guessing whose lock this is"
        )
    if record.released_at is not None:
        return True, f"the previous holder released it at {record.released_at}"
    heartbeat_age = _age_seconds(record.heartbeat_at, now=now)
    stale = heartbeat_age is None or heartbeat_age > limit
    age_text = (
        f"its heartbeat is {heartbeat_age:.0f}s old (limit {limit:.0f}s)"
        if heartbeat_age is not None
        else "its heartbeat cannot be read as a time, which is not evidence of a live holder"
    )
    if record.host != this_host:
        # The pid belongs to another host's process table. Nothing here can
        # ask about it, and a lock directory shared between hosts does not
        # make the kernel mutex mean anything across them either.
        live: bool | None = None
        unverifiable = f"that is not this host ({this_host!r}), so its pid cannot be checked from here"
    else:
        live = _holder_is_live(record)
        unverifiable = "this platform cannot be asked whether that pid is running"
    if live is True:
        return False, f"{record.describe()}; that process is running on this host"
    if kernel_mutex_held and record.host == this_host:
        settled = "that process is gone" if live is False else unverifiable
        return True, f"{record.describe()}; {settled}, and the kernel lock on this host is free"
    if live is False:
        if not stale:
            return False, (
                f"{record.describe()}; pid {record.pid} is gone but {age_text}, and without fcntl a pid can "
                "be recycled, so the lock is not broken"
            )
        return True, f"{record.describe()}; pid {record.pid} is gone on this host and {age_text}"
    if stale:
        return True, f"{record.describe()}; {unverifiable}, and {age_text}"
    return False, f"{record.describe()}; {unverifiable}, and {age_text}"


class LockHandle:
    """A held lock. Yielded by both context managers; released on exit."""

    def __init__(self, path: Path, fd: int, record: LockRecord) -> None:
        self.path = path
        self.record = record
        self._fd = fd
        stat = os.fstat(fd)
        self._identity = (stat.st_dev, stat.st_ino)
        self._released = False

    @property
    def pid(self) -> int:
        return self.record.pid

    @property
    def host(self) -> str:
        return self.record.host

    def _confirm_ours(self) -> None:
        try:
            stat = os.stat(self.path)
        except OSError as exc:
            raise LockLost(f"the lock file {self.path} is gone while this process still held it: {exc}") from exc
        if (stat.st_dev, stat.st_ino) != self._identity:
            raise LockLost(
                f"{self.path} was replaced while this process held it — another writer reclaimed the lock, "
                "so this process is no longer the exclusive writer"
            )

    def heartbeat(self) -> str:
        """Refresh the heartbeat, or raise ``LockLost`` if we no longer hold it.

        A holder that never beats still cannot be displaced on its own host
        while its process lives, on any platform whose pids can be asked about;
        the heartbeat is what a reclaimer that cannot ask — another host, or a
        platform with no probe — has to go on, and there it is the only thing
        standing between a live holder and a second writer.
        """
        self._confirm_ours()
        self.record = replace(self.record, heartbeat_at=utc_now_text())
        _write_record(self._fd, self.record)
        return self.record.heartbeat_at

    def release(self) -> None:
        """Record the release and drop the lock. Idempotent.

        The file is not unlinked, on either path: unlinking races a concurrent
        opener. ``released_at`` is what makes the next acquisition immediate
        instead of waiting out a heartbeat window.
        """
        if self._released:
            return
        self._released = True
        try:
            try:
                self._confirm_ours()
            except LockLost as exc:
                # Do not stamp our release over the record of whoever holds it
                # now. Raising here would mask the exception being unwound.
                logger.error("releasing %s: %s", self.path, exc)
            else:
                self.record = replace(self.record, released_at=utc_now_text())
                _write_record(self._fd, self.record)
        finally:
            os.close(self._fd)  # on the fcntl path, closing the fd releases the flock


def _refusal(path: Path, why: str, context: str) -> str:
    return f"another prism process holds {path}: {why}. {context}"


def _acquire_flock(
    path: Path,
    record: LockRecord,
    *,
    max_heartbeat_age_seconds: float,
    context: str,
) -> LockHandle:
    """Take the kernel mutex, then rule out a holder the kernel cannot see."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            existing = _read_record(fd)
            why = existing.describe() if existing is not None else "the holder's record is unreadable"
            raise LockRefused(_refusal(path, why, context)) from exc
        raw = _read_raw(fd)
        _refuse_a_foreign_schema(raw, path, context)
        free, why = _displaceable(
            _parse_record(raw),
            path=path,
            this_host=record.host,
            max_heartbeat_age_seconds=max_heartbeat_age_seconds,
            kernel_mutex_held=True,
        )
        if not free:
            raise LockRefused(_refusal(path, why, context))
        _write_record(fd, record)
    except BaseException:
        os.close(fd)  # releases the flock if we took it
        raise
    return LockHandle(path, fd, record)


def _acquire_o_excl(
    path: Path,
    record: LockRecord,
    *,
    max_heartbeat_age_seconds: float,
    context: str,
) -> LockHandle:
    """Create the lock file with ``O_EXCL``, or reclaim a provably dead one.

    Reclaiming writes a fresh temp file and renames it over the stale one after
    re-reading and confirming the bytes are unchanged. There is no
    unlink-then-create step, which would open a window in which two writers
    both believe they hold the lock; the rename narrows that window to the
    interval between the confirming read and the rename itself, and the
    post-rename read is what tells a loser of that race that it lost.
    """
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    except FileExistsError:
        pass
    else:
        try:
            _write_record(fd, record)
        except BaseException:
            os.close(fd)
            raise
        return LockHandle(path, fd, record)

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LockRefused(_refusal(path, f"it exists but cannot be read ({exc})", context)) from exc
    _refuse_a_foreign_schema(raw, path, context)
    free, why = _displaceable(
        _parse_record(raw),
        path=path,
        this_host=record.host,
        max_heartbeat_age_seconds=max_heartbeat_age_seconds,
        kernel_mutex_held=False,
    )
    if not free:
        raise LockRefused(_refusal(path, why, context))

    temp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        temp_fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            _write_record(temp_fd, record)
        finally:
            os.close(temp_fd)
        if path.read_bytes() != raw:
            raise LockRefused(
                _refusal(
                    path, "it changed while this process was reclaiming it, so another writer got there first", context
                )
            )
        os.replace(temp, path)
    finally:
        with contextlib.suppress(OSError):
            temp.unlink()

    fd = os.open(path, os.O_RDWR)
    try:
        if _read_record(fd) != record:
            raise LockRefused(
                _refusal(path, "another writer reclaimed it in the same instant, and won the rename", context)
            )
    except BaseException:
        os.close(fd)
        raise
    return LockHandle(path, fd, record)


def _acquire(path: Path, record: LockRecord, *, max_heartbeat_age_seconds: float, context: str) -> LockHandle:
    """One entry point, two protocols, no third branch that proceeds.

    Where ``fcntl`` exists ``flock`` is the lock, as it has always been here.
    Where it does not, the ``O_EXCL`` creation *is* the lock. There is no
    platform on which a failed acquisition continues.
    """
    if fcntl is not None:
        return _acquire_flock(path, record, max_heartbeat_age_seconds=max_heartbeat_age_seconds, context=context)
    return _acquire_o_excl(path, record, max_heartbeat_age_seconds=max_heartbeat_age_seconds, context=context)


def _new_record(kind: str, key: str) -> LockRecord:
    pid = os.getpid()
    now = utc_now_text()
    return LockRecord(
        schema=LOCK_SCHEMA,
        kind=kind,
        key=key,
        host=socket.gethostname(),
        pid=pid,
        start_token=process_start_token(pid),
        acquired_at=now,
        heartbeat_at=now,
        released_at=None,
    )


@contextlib.contextmanager
def run_dir_lock(
    run_dir: Path | str,
    *,
    max_heartbeat_age_seconds: float = DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
) -> Generator[LockHandle, None, None]:
    """Hold the exclusive writer lock for ``run_dir``, or raise loudly.

    Raises :class:`LockRefused` (a ``RuntimeError``) when another process holds
    it, on every platform. This is the weaker of the two locks: one run
    directory, not one account. It is retained alongside the account lock
    because the ledgers and ``state.json`` of a run directory are a second
    thing two writers can tear.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / LOCK_FILENAME
    handle = _acquire(
        path,
        _new_record("run_dir", str(run_dir)),
        max_heartbeat_age_seconds=max_heartbeat_age_seconds,
        context=(
            "Refusing a second concurrent writer on this run directory (N7; the write-ahead protocol "
            "protects against crashes, not concurrent writers)."
        ),
    )
    try:
        yield handle
    finally:
        handle.release()


def break_account_lock(
    account_id: str,
    *,
    reason: str,
    broken_by: str,
    lock_dir: Path | str | None = None,
    before_break: Callable[[LockRecord | None], None] | None = None,
) -> tuple[Path, LockRecord | None]:
    """Break one account's execution lock, on the record (§3.10).

    A non-stale lock is never broken *automatically*; this is the manual break,
    and what makes it accountable rather than reckless is the order.
    ``before_break`` is called first, with the record about to be displaced, and
    only a return from it lets the break happen — so a caller that writes the
    operator's record there has written it *before* the lock changed, and a
    crash mid-break leaves evidence of the attempt rather than an unexplained
    free lock. ``reason`` and ``broken_by`` are what that record has to carry;
    they are required here so that no caller can break a lock without having
    them, and they are logged at WARNING. They are not written into the lock
    file itself: the lock record's schema is what a concurrent acquirer parses,
    and the durable account of the break belongs in the run directory, which is
    where §3.10 puts it.

    The lock file is never unlinked — unlinking races a concurrent opener — so
    the break writes a *released* record over it through the same
    temp-and-rename the reclaim path uses. The next acquisition reads a released
    record and takes the lock; nothing has to be cleaned up by hand. A displaced
    holder that turns out to be alive finds out the way a reclaimed one does: its
    next heartbeat sees a different inode at the path and raises
    :class:`LockLost`.

    One case is refused rather than broken: a lock whose record names **this**
    host and whose pid is running here. That is not a stale lock, it is a live
    writer, and displacing it would leave two processes believing they hold one
    account — the exact condition the lock exists to prevent. Stop that process,
    or wait for it.

    Returns the lock's path and the record it displaced (``None`` when the file
    held no readable record — a holder that died between the create and the
    write).
    """
    if not str(reason).strip():
        raise LockRefused("breaking a lock requires a reason: §3.10 records who broke it, when, and why")
    path = account_lock_path(account_id, lock_dir)
    if not path.exists():
        raise LockRefused(
            f"there is no lock file at {path} to break; nothing holds account {account_id!r} on this host"
        )
    raw = path.read_bytes()
    _refuse_a_foreign_schema(raw, path, "Refusing to break a lock this build cannot read.")
    existing = _parse_record(raw)
    if existing is not None and existing.released_at is not None:
        raise LockRefused(
            f"the lock at {path} was released at {existing.released_at}; nothing holds it, so there is "
            "nothing to break"
        )
    this_host = socket.gethostname()
    if existing is not None and existing.host == this_host and _holder_is_live(existing) is True:
        raise LockRefused(
            f"{existing.describe()}; that process is running on this host, so the lock is held rather than "
            "stale. Refusing to break it: two writers on one account is the condition this lock exists to "
            "prevent. Stop pid " + f"{existing.pid} first, or wait for it (§3.10)"
        )
    if before_break is not None:
        before_break(existing)
    now = utc_now_text()
    displacing = LockRecord(
        schema=LOCK_SCHEMA,
        kind=existing.kind if existing is not None else "account",
        key=_account_segment(account_id),
        host=this_host,
        pid=os.getpid(),
        start_token=process_start_token(os.getpid()),
        acquired_at=existing.acquired_at if existing is not None else now,
        heartbeat_at=now,
        released_at=now,
    )
    temp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        temp_fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            _write_record(temp_fd, displacing)
        finally:
            os.close(temp_fd)
        os.replace(temp, path)
    finally:
        with contextlib.suppress(OSError):
            temp.unlink()
    logger.warning(
        "execution lock for account %r broken by %s: %s (displaced %s)",
        account_id,
        broken_by,
        reason,
        existing.describe() if existing is not None else "an unreadable record",
    )
    return path, existing


@contextlib.contextmanager
def account_execution_lock(
    account_id: str,
    execution_host: str,
    *,
    lock_dir: Path | str | None = None,
    max_heartbeat_age_seconds: float = DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
) -> Generator[LockHandle, None, None]:
    """Hold the execution lock for one account on its declared host (§3.10).

    ``account_id`` and ``execution_host`` are the authorization record's
    ``account_id`` and ``execution_host``; they are taken as plain values so
    this module does not depend on the record type. Acquire before the first
    state load of any invocation that may write to the broker or to state, and
    release after that invocation's last state save — settle writes state and
    ledgers, so settle is *inside* the lock.

    Refuses when this host is not ``execution_host``: one execution host per
    account is a declaration, and running the account from an undeclared host
    is exactly the condition the declaration exists to catch. It does not, and
    cannot, refuse a second host that declares itself the execution host for
    the same account — see this module's docstring.
    """
    account_id = _account_segment(account_id)
    record = _new_record("account", account_id)
    if not execution_host or record.host != execution_host:
        raise LockRefused(
            f"this host is {record.host!r} and the authorization record for account {account_id!r} names "
            f"{execution_host!r} as its execution host; refusing to take the execution lock, and therefore "
            "to write, from an undeclared host (§3.10)"
        )
    path = account_lock_path(account_id, lock_dir)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle = _acquire(
        path,
        record,
        max_heartbeat_age_seconds=max_heartbeat_age_seconds,
        context=(
            f"Refusing a second concurrent writer on account {account_id!r} (§3.10: one execution host per "
            "account, one writer per host; this lock excludes writers on this host only)."
        ),
    )
    try:
        yield handle
    finally:
        handle.release()
