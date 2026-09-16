"""Fitted component state as a real artifact (spec §3.3, §3.11; decision D4).

A JSON-serialized dataclass cannot hold a fitted object, so the fit does not
live in the state document: it lives in one file under
``workspace.state / "fits" / <component_id> /``, written by the component's own
``save_state`` in the component's own format, opaque to the engine. What the
engine keeps is the *reference*: ``"sha256:" + sha256(file bytes)``.

Three properties that reference buys, and this module exists to enforce:

* **A ``--stage decide`` process and a later ``--stage submit`` process resolve
  the same fit.** The reference is content-addressed, so the second process
  reads the artifact the first one wrote or fails; it never silently scores an
  unfitted component.
* **A restart restores the fit.** ``load`` verifies the bytes against the
  reference before handing them to ``load_state``, and a mismatch is refused
  rather than tolerated — a fit that is not the fit the record names is not a
  fallback, it is a different model.
* **A record can be checked.** ``fitted_state_ref`` in a result record
  (§3.12) names a file whose hash anyone can recompute.

A component declaring ``fitted_state = "none"`` has no artifact and its
reference is ``None`` (§3.3). Nothing here writes an empty file to stand in
for one: an empty artifact is a fit that a later load would silently accept.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, Protocol

#: The one digest the reference carries, spelled out in the reference itself so
#: a future second algorithm is a different prefix rather than an ambiguity.
FIT_REF_PREFIX = "sha256:"

#: Artifact filename suffix under ``fits/<component_id>/``. The stem is the
#: hex digest, which is what makes the store content-addressed: writing the
#: same fit twice writes the same path.
ARTIFACT_SUFFIX = ".state"


class FitArtifactError(ValueError):
    """A fit artifact is missing, unreadable, or not the artifact the reference names."""


class StatefulComponent(Protocol):
    """The half of the §3.3 contract this module uses."""

    fitted_state: str

    def save_state(self, path: Path) -> None: ...

    def load_state(self, path: Path) -> Any: ...


def is_stateless(component: object) -> bool:
    """True when the component declares ``fitted_state = "none"`` (§3.3).

    An undeclared ``fitted_state`` is *not* read as stateless: the declaration
    is mandatory (``check_declarations``), and guessing "none" for a component
    that never said so would drop its fit without a word.
    """
    declared = getattr(component, "fitted_state", None)
    if declared is None:
        raise FitArtifactError(
            f"{type(component).__name__} declares no fitted_state; every component declares it (§3.3), "
            "and a fit store that guesses would drop a fitted model in silence"
        )
    return str(declared) == "none"


def digest_bytes(data: bytes) -> str:
    """``"sha256:" + sha256(data)`` — the reference form of §3.11."""
    return FIT_REF_PREFIX + hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    """The reference of an artifact already on disk."""
    path = Path(path)
    if not path.is_file():
        raise FitArtifactError(f"{path} is not a file, so it has no content hash")
    return digest_bytes(path.read_bytes())


def normalize_ref(ref: str) -> str:
    """The canonical ``"sha256:<hex>"`` form of a reference.

    A bare 64-character hex digest is accepted and canonicalized: the evaluator
    recorded fit references that way before this module existed
    (``evaluate.py::_save_fit``), and refusing to resolve an artifact this
    store can address would strand those records.
    """
    text = str(ref).strip()
    if not text:
        raise FitArtifactError("an empty string is not a fit reference")
    body = text[len(FIT_REF_PREFIX) :] if text.startswith(FIT_REF_PREFIX) else text
    if len(body) != 64 or any(character not in "0123456789abcdef" for character in body.lower()):
        raise FitArtifactError(
            f"{ref!r} is not a fit reference: the reference is {FIT_REF_PREFIX!r} followed by a "
            "64-character sha256 hex digest of the artifact's bytes (§3.11)"
        )
    return FIT_REF_PREFIX + body.lower()


def _component_segment(component_id: str) -> str:
    """One directory name, never a path: a component id is not a place to escape the store from."""
    text = str(component_id)
    if not text or text in (".", ".."):
        raise FitArtifactError(f"component_id must be a non-empty name; got {component_id!r}")
    if "/" in text or os.sep in text or (os.altsep and os.altsep in text):
        raise FitArtifactError(f"component_id must be a single path segment (no separators); got {component_id!r}")
    return text


def component_dir(fits_dir: Path, component_id: str) -> Path:
    """``fits_dir / component_id`` — one directory per component (§3.11)."""
    return Path(fits_dir) / _component_segment(component_id)


def artifact_path(fits_dir: Path, component_id: str, ref: str) -> Path:
    """Where the artifact named by ``ref`` lives. The path is derived, never stored."""
    return component_dir(fits_dir, component_id) / f"{normalize_ref(ref)[len(FIT_REF_PREFIX):]}{ARTIFACT_SUFFIX}"


def save(component: object, *, component_id: str, fits_dir: Path) -> str | None:
    """Write ``component``'s fit and return its ``fitted_state_ref``.

    Returns ``None``, and writes nothing, for a component declaring
    ``fitted_state = "none"``: a stateless component has no fit, and an empty
    artifact standing in for one is exactly the dangling reference §3.11 rules
    out. A stateful component that does not implement ``save_state``, or writes
    no file, is refused loudly — its record's reference would name nothing.

    The write is staged and renamed, so a crash mid-write cannot leave a
    truncated artifact at a name that claims to be a content hash.
    """
    if is_stateless(component):
        return None
    directory = component_dir(fits_dir, component_id)
    directory.mkdir(parents=True, exist_ok=True)
    staging = directory / f".staging-{uuid.uuid4().hex}"
    try:
        try:
            component.save_state(staging)  # type: ignore[attr-defined]
        except NotImplementedError as exc:
            raise FitArtifactError(
                f"component {component_id!r} declares fitted_state="
                f"{getattr(component, 'fitted_state', None)!r} and must implement save_state (§3.3): a fit "
                "that cannot be persisted cannot be replayed, and the record's fitted_state_ref would name "
                "nothing (§3.11)"
            ) from exc
        if not staging.is_file():
            raise FitArtifactError(
                f"component {component_id!r}.save_state wrote no file at {staging}; the fit artifact is one "
                "file per content hash (§3.11)"
            )
        ref = digest_file(staging)
        target = directory / f"{ref[len(FIT_REF_PREFIX):]}{ARTIFACT_SUFFIX}"
        if target.exists():
            staging.unlink()
        else:
            staging.replace(target)
        return ref
    finally:
        if staging.exists():
            staging.unlink()


def load(component: object, ref: str, *, component_id: str, fits_dir: Path) -> Any:
    """Restore ``ref`` into ``component`` after verifying the artifact's bytes.

    The hash is checked before ``load_state`` sees the file. A mismatch means
    the artifact at that path is not the one the reference names — a rewritten
    file, a truncated write, a collision of ids across two runs — and it is
    refused, never loaded with a warning.
    """
    reference = normalize_ref(ref)
    path = artifact_path(fits_dir, component_id, reference)
    if not path.is_file():
        raise FitArtifactError(
            f"component {component_id!r}: no fit artifact at {path} for {reference}; the reference names a "
            "file that is not in this workspace's fit store"
        )
    observed = digest_file(path)
    if observed != reference:
        raise FitArtifactError(
            f"component {component_id!r}: {path} hashes to {observed}, not the {reference} the reference "
            "names; the artifact is not the fit that was recorded and is refused rather than loaded"
        )
    try:
        restored = component.load_state(path)  # type: ignore[attr-defined]
    except NotImplementedError as exc:
        raise FitArtifactError(
            f"component {component_id!r} declares fitted_state="
            f"{getattr(component, 'fitted_state', None)!r} and must implement load_state (§3.3): the fit "
            f"artifact {path} exists and cannot be restored"
        ) from exc
    if restored is None:
        raise FitArtifactError(
            f"component {component_id!r}.load_state returned None; the contract is 'the inverse of save_state', "
            "which returns the restored component (§3.3)"
        )
    return restored


def exists(ref: str, *, component_id: str, fits_dir: Path) -> bool:
    """Whether the artifact ``ref`` names is present in this store."""
    return artifact_path(fits_dir, component_id, ref).is_file()
