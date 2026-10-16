"""The per-deployment seed: what makes one deployment's schedule its own.

Every deployment of this repository computes the same book from the same public
data. Before this module the only thing that set *when* a monthly book traded
was its first session (``prism.live.daily._is_refresh_session``), so every user
who started on the same day refreshed on the same day, and their orders landed
together. A deployment profile is a random seed drawn once, when a deployment
starts, and stored in its run directory. Values derived from it are fixed for
that deployment and independent across deployments:

* the **rebalance phase** of a single-phase book (``--rebalance-phase seeded``
  on ``prism.scripts.paper_loop``): a grid offset in ``[0, decision_every)``,
  so the book refreshes on its own sessions of the fixed session grid
  (``prism.io.sessions.grid_position``) whenever it started;
* any later per-session draw a schedule needs, through :meth:`DeploymentProfile.draw`.

A draw is HMAC-SHA256 keyed by the seed over ``"<purpose>:<bar>"``, read as a
uniform value. The seed is never derived from the run directory's name
(``prism.scripts.paper_books.run_dir_tag`` hashes a basename that many
deployments share, such as ``paper_loop_momentum``), and it is never written
anywhere else.

The file is ``{run-dir}/deployment.json``, written atomically with mode 0600.
The seed is not a credential, and the phase it sets is not a secret: the
offset is written in plain text to the nightly log, to ``state.json`` (each
tranche's ``phase_offset_bars``), to the ``tranches`` record of
``targets.jsonl`` and to ``sizing.jsonl``, all of which the off-box backup
mirrors, and anyone who sees the account's orders sees it from the first
refresh. What the seed buys is
independence: two deployments draw unrelated phases. The 0600 mode keeps the
seed itself to the owner, and so every later draw it would make, such as a
per-session draw a schedule might take (none is used yet).

**Who creates it.** Only the first submitting session of a new deployment:
``--submit --allow-new-run-dir`` on a run directory that held no prior session
when the process started (:func:`establish_deployment`). Every later session,
and every rehearsal, reads it, and a seeded mode with no file raises (N7): a
phase drawn afresh on a later night would move the book to a different
schedule, and a rehearsal that drew one would show a schedule its real first
session would not use. An existing file is never overwritten.

Production-import-path safe (N8): standard library only.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism.live.safety import RunDirHistory

logger = logging.getLogger(__name__)

__all__ = [
    "DEPLOYMENT_FILENAME",
    "DEPLOYMENT_FILE_MODE",
    "DEPLOYMENT_SCHEMA",
    "REBALANCE_PHASE_PURPOSE",
    "DeploymentProfile",
    "DeploymentProfileError",
    "DeploymentProfileMissing",
    "establish_deployment",
    "read_deployment",
    "write_deployment",
]

#: The profile's file name inside the run directory.
DEPLOYMENT_FILENAME = "deployment.json"

#: The profile document's schema version.
DEPLOYMENT_SCHEMA = 1

#: Owner read and write only.
DEPLOYMENT_FILE_MODE = 0o600

#: The draw purpose the rebalance phase is taken under.
REBALANCE_PHASE_PURPOSE = "rebalance_phase"

#: A seed is 32 random bytes, stored as 64 lowercase hex characters.
SEED_BYTES = 32

_SEED_PATTERN = re.compile(rf"[0-9a-f]{{{2 * SEED_BYTES}}}")
_KEYS = frozenset({"schema", "seed_hex", "created_bar"})


class DeploymentProfileError(ValueError):
    """A deployment profile is malformed, or an operation on one is refused (N7)."""


class DeploymentProfileMissing(DeploymentProfileError):
    """A seeded mode was requested and the run directory holds no deployment profile."""


@dataclass(frozen=True)
class DeploymentProfile:
    """One deployment's seed, the bar on which it was created, and the document schema.

    ``created_bar`` is the last published session when the deployment's first
    submitting session ran (an ISO date). It is part of the record, and it is
    the bar the rebalance phase is drawn under, so the phase never changes
    after creation.
    """

    seed_hex: str
    created_bar: str
    schema: int = DEPLOYMENT_SCHEMA

    def __post_init__(self) -> None:
        if isinstance(self.schema, bool) or self.schema != DEPLOYMENT_SCHEMA:
            raise DeploymentProfileError(
                f"deployment profile schema {self.schema!r} is not the supported {DEPLOYMENT_SCHEMA}"
            )
        if not isinstance(self.seed_hex, str) or _SEED_PATTERN.fullmatch(self.seed_hex) is None:
            raise DeploymentProfileError(
                f"deployment seed must be {2 * SEED_BYTES} lowercase hex characters ({SEED_BYTES} bytes)"
            )
        if not isinstance(self.created_bar, str):
            raise DeploymentProfileError(f"created_bar must be an ISO date string, got {self.created_bar!r}")
        try:
            parsed = dt.date.fromisoformat(self.created_bar)
        except ValueError as exc:
            raise DeploymentProfileError(f"created_bar {self.created_bar!r} is not an ISO date: {exc}") from exc
        if parsed.isoformat() != self.created_bar:
            raise DeploymentProfileError(f"created_bar {self.created_bar!r} is not in YYYY-MM-DD form")

    @classmethod
    def new(cls, created_bar: str) -> DeploymentProfile:
        """A profile with a fresh seed from the operating system's random source."""
        return cls(seed_hex=secrets.token_hex(SEED_BYTES), created_bar=str(created_bar))

    def __repr__(self) -> str:
        # The seed determines every draw this deployment makes; keep it out of logs.
        return (
            f"DeploymentProfile(seed_hex=<{2 * SEED_BYTES} hex>, created_bar={self.created_bar!r}, "
            f"schema={self.schema})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "seed_hex": self.seed_hex, "created_bar": self.created_bar}

    @classmethod
    def from_dict(cls, payload: Any) -> DeploymentProfile:
        if not isinstance(payload, dict):
            raise DeploymentProfileError(f"a deployment profile is a JSON object, got {type(payload).__name__}")
        keys = set(payload)
        if keys != _KEYS:
            missing, extra = sorted(_KEYS - keys), sorted(keys - _KEYS)
            raise DeploymentProfileError(
                f"deployment profile keys must be exactly {sorted(_KEYS)}; missing {missing}, unexpected {extra}"
            )
        return cls(seed_hex=payload["seed_hex"], created_bar=payload["created_bar"], schema=payload["schema"])

    # ----------------------------------------------------------------- draws

    def _word(self, purpose: str, bar: str) -> int:
        """The first 64 bits of HMAC-SHA256(seed, ``"<purpose>:<bar>"``), big-endian."""
        if not isinstance(purpose, str) or not purpose or ":" in purpose:
            # A colon in the purpose would let two (purpose, bar) pairs share one message.
            raise DeploymentProfileError(f"a draw purpose is a non-empty string without ':', got {purpose!r}")
        message = f"{purpose}:{bar}".encode()
        digest = hmac.new(bytes.fromhex(self.seed_hex), message, hashlib.sha256).digest()
        return int.from_bytes(digest[:8], "big")

    def draw(self, purpose: str, bar: str) -> float:
        """A uniform value in ``[0, 1)`` for ``purpose`` on ``bar``, fixed by the seed.

        The top 53 bits of the 64-bit word, so the value is exactly representable
        and never reaches 1.
        """
        return (self._word(purpose, bar) >> 11) * 2.0**-53

    def draw_index(self, purpose: str, bar: str, n: int) -> int:
        """A uniform integer in ``[0, n)`` for ``purpose`` on ``bar``: ``floor(n * word / 2**64)``.

        Computed in integers on the same 64-bit word :meth:`draw` reads, so it
        is exact and below ``n`` by construction, with no floating-point step.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise DeploymentProfileError(f"draw_index needs a positive integer n, got {n!r}")
        return (self._word(purpose, bar) * n) >> 64

    def phase_offset(self, decision_every: int) -> int:
        """The book's grid offset in ``[0, decision_every)``, drawn under ``created_bar``.

        A pure function of the seed, the creation bar and the cadence, so it is
        the same on every session of the deployment.
        """
        return self.draw_index(REBALANCE_PHASE_PURPOSE, self.created_bar, decision_every)


# ----------------------------------------------------------------- the file


def read_deployment(run_dir: Path | str) -> DeploymentProfile:
    """Load ``{run_dir}/deployment.json``, or raise (N7).

    A missing file raises :class:`DeploymentProfileMissing`; an unreadable or
    malformed one raises :class:`DeploymentProfileError`. A file that the group
    or others can read is loaded with a warning, because the mode is the
    operator's to fix and the seed is still the deployment's own.
    """
    path = Path(run_dir) / DEPLOYMENT_FILENAME
    if not path.is_file():
        raise DeploymentProfileMissing(f"no deployment profile at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentProfileError(f"unreadable deployment profile at {path}: {exc}") from exc
    try:
        profile = DeploymentProfile.from_dict(payload)
    except DeploymentProfileError as exc:
        raise DeploymentProfileError(f"malformed deployment profile at {path}: {exc}") from exc
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        logger.warning(
            "deployment profile %s has mode %o; it should be %o (owner only), because its seed determines every "
            "draw this deployment makes (the phase offset itself is recorded in plain text elsewhere)",
            path,
            mode,
            DEPLOYMENT_FILE_MODE,
        )
    return profile


def write_deployment(run_dir: Path | str, profile: DeploymentProfile) -> Path:
    """Write ``profile`` to ``{run_dir}/deployment.json`` atomically with mode 0600; never overwrite.

    Temp file (created 0600 and set to 0600 whatever the umask), fsync, rename,
    then an fsync of the directory so the rename itself survives a power loss.
    The caller holds the run directory's writer lock, which is what makes the
    existence check and the rename one step.
    """
    path = Path(run_dir) / DEPLOYMENT_FILENAME
    if path.exists():
        raise DeploymentProfileError(
            f"refusing to overwrite the deployment profile at {path}: a deployment's seed is drawn once"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, DEPLOYMENT_FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            # A leftover temp file keeps its old mode through O_CREAT; set it explicitly.
            os.fchmod(handle.fileno(), DEPLOYMENT_FILE_MODE)
            handle.write(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return path


def establish_deployment(
    run_dir: Path | str,
    *,
    history: RunDirHistory,
    submit: bool,
    allow_new_run_dir: bool,
    created_bar: str,
) -> DeploymentProfile:
    """The run directory's deployment profile: read it, or create it on a new deployment's first session.

    Returns the existing profile whenever ``{run_dir}/deployment.json`` exists
    (a first session that crashed after writing it is re-run on the same seed).
    Otherwise creates one only when this invocation submits, names
    ``--allow-new-run-dir``, and ``history`` (sampled before the writer lock
    created the directory) shows no prior session. Any other case raises
    :class:`DeploymentProfileMissing`, saying why.
    """
    path = Path(run_dir) / DEPLOYMENT_FILENAME
    if path.exists():
        return read_deployment(run_dir)
    if submit and allow_new_run_dir and not history.established:
        profile = DeploymentProfile.new(created_bar)
        write_deployment(run_dir, profile)
        logger.info("created deployment profile %s (created_bar %s, mode %o)", path, created_bar, DEPLOYMENT_FILE_MODE)
        return profile
    if not submit:
        why = (
            "A rehearsal never creates one: the seed is drawn once, by the deployment's first submitting "
            "session (--submit --allow-new-run-dir), and a rehearsal that drew its own would show a schedule "
            "the real first session would not use."
        )
    elif history.established:
        why = (
            f"This run directory already holds sessions ({', '.join(history.session_files)}) that ran without a "
            "seeded phase, and a deployment's phase is drawn when it starts. Start a new run directory with "
            "--submit --allow-new-run-dir --rebalance-phase seeded, or keep this one on "
            "--rebalance-phase first-session."
        )
    else:
        why = "Only the first session of a new deployment, run with --submit --allow-new-run-dir, creates one."
    raise DeploymentProfileMissing(
        f"--rebalance-phase seeded needs the deployment profile {path}, and it does not exist. {why} "
        "Nothing has been read from the venue and nothing has been sent."
    )
