"""The workspace: the only root the engine writes to (spec §3.1).

Nothing the engine writes lands beside the installed package. A command
resolves a :class:`Workspace` from ``--workspace``, ``PRISM_WORKSPACE``,
``[tool.prism].workspace`` in the current directory's ``pyproject.toml``, or
the current directory, then calls :meth:`Workspace.ensure` before its first
write. The package location is never consulted: an installed wheel has no
business writing into ``site-packages``, and the previous package-relative
``DATA_DIR``/``RESULTS_DIR`` made the checkout the implicit workspace.

``.env`` loading lives here too, for the same reason: the credentials belong
to the project the operator is standing in, not to the installed package.
:func:`load_env` is called by command entry points after argument parsing —
never at import (the import-purity invariant of §3.1).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import dotenv

# The directories ``ensure`` creates under the root, in layout order.
SUBDIRECTORIES = ("data", "results", "state", "trials", "runs", "logs")

# The two named places ``resolve_workspace`` consults between the CLI flag and
# the current directory.
WORKSPACE_ENV_VAR = "PRISM_WORKSPACE"
PYPROJECT_FILENAME = "pyproject.toml"

# Fitted component artifacts live under one directory inside ``state`` so a
# ``--stage decide`` process and a later ``--stage submit`` process exchange
# fitted state through the filesystem and nothing else.
FITS_DIRNAME = "fits"


def _segment(value: str, *, kind: str) -> str:
    """Validate that ``value`` names one directory entry, not a path.

    A workspace that can be escaped by an account name or a selection-set id
    is not a containment boundary. Fail loud rather than write outside root.
    """
    if not value or value in (".", ".."):
        raise ValueError(f"{kind} must be a non-empty name; got {value!r}")
    if os.sep in value or (os.altsep and os.altsep in value) or "/" in value:
        raise ValueError(f"{kind} must be a single path segment (no separators); got {value!r}")
    return value


@dataclass(frozen=True)
class Workspace:
    """The writable root and the six directories the engine owns.

    Constructed from ``root`` alone; every other attribute is derived, so a
    caller cannot hand two halves of the layout to different roots.
    """

    root: Path
    data: Path = field(init=False)
    results: Path = field(init=False)
    state: Path = field(init=False)
    trials: Path = field(init=False)
    runs: Path = field(init=False)
    logs: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())
        for name in SUBDIRECTORIES:
            object.__setattr__(self, name, self.root / name)

    def ensure(self) -> "Workspace":
        """Create root and the six directories if absent and return self.

        Idempotent, never destructive, never called at import. Creates
        nothing outside root. Raises when root exists and is not a writable
        directory.
        """
        if self.root.exists():
            if not self.root.is_dir():
                raise NotADirectoryError(f"workspace root {self.root} exists and is not a directory")
            if not os.access(self.root, os.W_OK | os.X_OK):
                raise PermissionError(f"workspace root {self.root} is not writable")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        for name in SUBDIRECTORIES:
            getattr(self, name).mkdir(exist_ok=True)
        return self

    def run_dir(self, account: str) -> Path:
        """``state / account``: state.json, the ledgers, and the account lock."""
        return self.state / _segment(account, kind="account")

    def fits(self) -> Path:
        """``state / "fits"``: fitted component artifacts, one directory per
        ``component_id``, each holding one file per content hash.
        """
        return self.state / FITS_DIRNAME

    def trial_log(self, selection_set_id: str) -> Path:
        """``trials / f"{selection_set_id}.jsonl"``."""
        return self.trials / f"{_segment(selection_set_id, kind='selection_set_id')}.jsonl"


def _pyproject_workspace(directory: Path) -> Path | None:
    """``[tool.prism].workspace`` from ``directory/pyproject.toml``, resolved
    relative to that file, or None when the file or the key is absent.

    A malformed ``pyproject.toml`` or a non-string value raises: the operator
    asked for a workspace here and silently falling through to the current
    directory would write the run somewhere they did not name.
    """
    path = directory / PYPROJECT_FILENAME
    if not path.is_file():
        return None
    try:
        table = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path} is not valid TOML: {exc}") from exc
    value = table.get("tool", {}).get("prism", {}).get("workspace")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"[tool.prism].workspace in {path} must be a non-empty string; got {value!r}")
    return (path.parent / Path(value).expanduser()).resolve()


def resolve_workspace(cli_value: str | None = None) -> Workspace:
    """Resolve the workspace root and return an *unensured* :class:`Workspace`.

    Order: ``--workspace``, ``PRISM_WORKSPACE``, ``[tool.prism].workspace`` in
    a ``pyproject.toml`` in the current directory (resolved relative to that
    file), then the current directory. Never the package location. The caller
    decides whether to create it.
    """
    if cli_value is not None and str(cli_value).strip():
        return Workspace(Path(str(cli_value)).expanduser().resolve())
    env_value = os.environ.get(WORKSPACE_ENV_VAR, "")
    if env_value.strip():
        return Workspace(Path(env_value).expanduser().resolve())
    cwd = Path.cwd()
    declared = _pyproject_workspace(cwd)
    if declared is not None:
        return Workspace(declared)
    return Workspace(cwd)


#: Set to ``1`` to switch ``.env`` discovery off. ``tests/conftest.py`` sets it so
#: no test process reads a deployment checkout's credentials; an explicit
#: ``path`` still loads.
DOTENV_DISABLE_ENV = "PRISM_DISABLE_DOTENV"


def load_env(path: Path | str | None = None) -> Path | None:
    """Load the project's ``.env`` into the process environment.

    Called by command entry points after argument parsing, never at import.
    Existing environment variables win, as ``python-dotenv`` defaults: an
    exported credential is not overwritten by a stale file.

    With no explicit ``path`` the search starts at the *current directory* and
    walks up, but never past a checkout boundary: the walk stops at the first
    directory holding a ``.git`` entry (a directory in a primary checkout, a
    file in a linked worktree). It deliberately does not walk up from the
    package: an installed wheel's neighbours are not the operator's project.
    The boundary exists because a worktree under ``<repo>/.claude/worktrees/``
    otherwise inherits ``<repo>/.env`` — the live account's credentials — which
    is how 116 unauthorized paper orders were submitted on 2026-09-09
    (``docs/incident_2026-09-09_unauthorized_paper_orders.md``, root cause 2).
    Discovery is skipped when ``PRISM_DISABLE_DOTENV=1`` (test suites, CI, and
    operators who export credentials themselves); an explicit ``path`` still
    loads. Returns the file loaded, or None when there is none.
    """
    if path is None and os.environ.get(DOTENV_DISABLE_ENV) == "1":
        return None
    if path is not None:
        found = Path(path)
        if not found.is_file():
            raise FileNotFoundError(f"no .env at {found}")
    else:
        discovered = find_dotenv_within_checkout(Path.cwd())
        if discovered is None:
            return None
        found = discovered
    dotenv.load_dotenv(found)
    return found


def find_dotenv_within_checkout(start: Path) -> Path | None:
    """Walk up from ``start`` for a ``.env``, stopping at the checkout root.

    The first directory that contains ``.env`` wins. The walk ends after
    examining a directory that contains ``.git`` (file or directory), so a
    nested checkout or linked worktree never reads a parent checkout's
    credentials. With no checkout boundary at all the walk runs to the
    filesystem root, as ``python-dotenv`` does.
    """
    directory = Path(start).resolve()
    while True:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            return None
        parent = directory.parent
        if parent == directory:
            return None
        directory = parent
