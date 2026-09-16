"""Dataset manifests and the content-hashed dataset identity (engine spec §3.2).

A manifest is the complete statement of what a dataset *is*: the files it
covers with their hashes, the sources that produced them with their lags and
price bases, the priority order that decides between two sources for one key,
the as-of boundary, and the assumptions the reader must accept to use it. Its
``id`` is the sha256 of that statement in canonical form, and that id is the
``data_id`` of every result derived from the dataset (§3.12).

Why a hash and not a name. Two evaluations that read different bytes must not
be comparable by accident. A new source revision, a corrected file, a changed
availability assumption, a different feature-cache training boundary — each
produces a different id, so nothing is silently rewritten and no result can
claim a provenance it does not have. Hashes establish identity, not truth.

What the id deliberately excludes. ``vintage`` is a human label mapped
one-to-one onto an id, so hashing it would make the mapping circular.
``coverage`` is descriptive of files already hashed. ``redistributable`` is a
licensing fact about a source, not a fact about its content: relabelling a
source's redistribution rights must not invalidate a recorded result. Every
other field of the manifest enters the id, and ``tests/test_manifest.py``
asserts that field by field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Provenance = Literal["market", "synthetic", "mixed"]
PriceBasis = Literal["raw", "split_adjusted_at_ingest"]
RevisionPolicy = Literal["dedupe_identical", "always_new_revision", "conflict_raise"]

PROVENANCES: tuple[str, ...] = ("market", "synthetic", "mixed")
PRICE_BASES: tuple[str, ...] = ("raw", "split_adjusted_at_ingest")
REVISION_POLICIES: tuple[str, ...] = ("dedupe_identical", "always_new_revision", "conflict_raise")

# The source-entry keys that enter ``Manifest.id``. ``redistributable`` is
# carried on the entry and stays out of the identity (see the module docstring).
SOURCE_IDENTITY_KEYS: tuple[str, ...] = ("lag_bars", "revision_policy", "price_basis", "url", "sha256")
SOURCE_KEYS: tuple[str, ...] = SOURCE_IDENTITY_KEYS + ("redistributable",)

_HASH_CHUNK_BYTES = 1 << 20


class ManifestError(ValueError):
    """A manifest is internally inconsistent, or does not describe the files on disk.

    Raised rather than repaired: a manifest that cannot be trusted about its
    own sources cannot be trusted about the data they produced.
    """


def canonical_json(payload: Any) -> str:
    """Serialize ``payload`` so that equal content always gives equal bytes.

    Sorted keys, no insignificant whitespace, ASCII escapes, and no NaN or
    Infinity — a non-finite float in an assumption would hash differently
    across platforms and is refused instead.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Content hash of one file, read in chunks so a large bar file costs no more memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, paths: list[Path]) -> list[tuple[str, str]]:
    """``(path relative to root, sha256)`` pairs, sorted by path.

    ``paths`` are absolute or root-relative; each must exist and lie under
    ``root``, because a manifest entry pointing outside the data root would
    name a file the view is not allowed to read.
    """
    root = Path(root).resolve()
    pairs: list[tuple[str, str]] = []
    for candidate in paths:
        absolute = (root / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate).resolve()
        try:
            relative = absolute.relative_to(root)
        except ValueError as exc:
            raise ManifestError(f"manifest file {absolute} is not under the data root {root}") from exc
        if not absolute.is_file():
            raise ManifestError(f"manifest file {absolute} does not exist")
        pairs.append((relative.as_posix(), sha256_file(absolute)))
    pairs.sort()
    return pairs


@dataclass(frozen=True)
class Manifest:
    """The frozen description of one dataset (§3.2, "Interfaces").

    ``files`` are ``(path relative to the data root, sha256)`` pairs and are
    the only files a ``research_vintage`` view may read. ``sources`` maps a
    source id onto its ``lag_bars``, ``revision_policy``, ``price_basis``,
    ``url``, ``sha256`` and ``redistributable`` flag. ``source_priority`` maps
    a table name onto the total order over the sources that may supply it; a
    row whose source is missing from that order makes the view refuse, because
    a lexical fallback would silently change results when a vendor is renamed.
    """

    provenance: Provenance
    as_of: str
    files: tuple[tuple[str, str], ...]
    sources: dict[str, dict[str, Any]]
    source_priority: dict[str, list[str]]
    universe: str
    vintage: str | None = None
    coverage: dict[str, Any] = field(default_factory=dict)
    transformations: list[str] = field(default_factory=list)
    assumptions: dict[str, Any] = field(default_factory=dict)
    fit_cutoff: str | None = None
    feature_code_sha: str | None = None
    feature_params: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(sorted((str(p), str(h)) for p, h in self.files)))
        self._validate()

    def _validate(self) -> None:
        if self.provenance not in PROVENANCES:
            raise ManifestError(f"provenance must be one of {PROVENANCES}, got {self.provenance!r}")
        if not self.as_of:
            raise ManifestError("as_of is required: it is the latest bar ts the manifest covers")
        seen: set[str] = set()
        for path, digest in self.files:
            if path in seen:
                raise ManifestError(f"manifest lists {path!r} twice")
            seen.add(path)
            if len(digest) != 64:
                raise ManifestError(f"manifest file {path!r} carries {digest!r}, which is not a sha256 hex digest")
        if not self.sources:
            raise ManifestError("a manifest declares at least one source")
        for source_id, entry in self.sources.items():
            missing = [k for k in SOURCE_KEYS if k not in entry]
            if missing:
                raise ManifestError(f"source {source_id!r} is missing {missing}")
            unknown = sorted(set(entry) - set(SOURCE_KEYS))
            if unknown:
                raise ManifestError(f"source {source_id!r} carries unknown keys {unknown}")
            if not isinstance(entry["lag_bars"], int) or isinstance(entry["lag_bars"], bool) or entry["lag_bars"] < 0:
                raise ManifestError(f"source {source_id!r} lag_bars must be a non-negative int")
            if entry["revision_policy"] not in REVISION_POLICIES:
                raise ManifestError(f"source {source_id!r} revision_policy must be one of {REVISION_POLICIES}")
            if entry["price_basis"] not in PRICE_BASES:
                raise ManifestError(f"source {source_id!r} price_basis must be one of {PRICE_BASES}")
            if not isinstance(entry["redistributable"], bool):
                raise ManifestError(f"source {source_id!r} redistributable must be a bool")
        ordered: set[str] = set()
        for table, order in self.source_priority.items():
            if len(set(order)) != len(order):
                raise ManifestError(f"source_priority[{table!r}] repeats a source: {order}")
            for source_id in order:
                if source_id not in self.sources:
                    raise ManifestError(f"source_priority[{table!r}] names unknown source {source_id!r}")
            ordered.update(order)
        unordered = sorted(set(self.sources) - ordered)
        if unordered:
            raise ManifestError(
                f"sources {unordered} appear in no source_priority order; a manifest that does not order its "
                "own sources is an error, not a case for a lexical fallback"
            )

    @property
    def id(self) -> str:
        """sha256 of the canonical identity payload — the ``data_id`` of §3.12."""
        return sha256_bytes(canonical_json(self.identity_payload()).encode("utf-8"))

    def identity_payload(self) -> dict[str, Any]:
        """Exactly the fields §3.2 hashes, in the shape it hashes them."""
        return {
            "files": [list(pair) for pair in self.files],
            "as_of": self.as_of,
            "provenance": self.provenance,
            "universe": self.universe,
            "transformations": list(self.transformations),
            "assumptions": self.assumptions,
            "sources": {
                source_id: {key: entry[key] for key in SOURCE_IDENTITY_KEYS}
                for source_id, entry in self.sources.items()
            },
            "source_priority": {table: list(order) for table, order in self.source_priority.items()},
            "fit_cutoff": self.fit_cutoff,
            "feature_code_sha": self.feature_code_sha,
            "feature_params": self.feature_params,
        }

    def priority_for(self, table: str) -> list[str]:
        """The declared source order for ``table``; refuses a table the manifest does not order."""
        try:
            return list(self.source_priority[table])
        except KeyError as exc:
            raise ManifestError(
                f"manifest declares no source_priority for table {table!r}; the view cannot decide between sources"
            ) from exc

    def lag_bars(self, source_id: str) -> int:
        """The source's declared availability lag in sessions, read from the manifest and never from a row."""
        try:
            return int(self.sources[source_id]["lag_bars"])
        except KeyError as exc:
            raise ManifestError(f"manifest declares no source {source_id!r}") from exc

    def lists_file(self, relative_path: str) -> bool:
        return any(path == relative_path for path, _ in self.files)

    def verify_files(self, root: Path) -> None:
        """Re-hash every listed file under ``root`` and refuse on the first disagreement."""
        root = Path(root)
        for relative, digest in self.files:
            actual = sha256_file(root / relative)
            if actual != digest:
                raise ManifestError(f"{relative} hashes to {actual}, the manifest records {digest}")

    def to_dict(self) -> dict[str, Any]:
        """The on-disk form: the identity payload plus the fields outside the identity, plus the id itself."""
        payload = self.identity_payload()
        payload["id"] = self.id
        payload["vintage"] = self.vintage
        payload["coverage"] = self.coverage
        payload["sources"] = {source_id: dict(entry) for source_id, entry in self.sources.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Manifest":
        """Rebuild a manifest and verify the recorded id against the recomputed one."""
        recorded = payload.get("id")
        manifest = cls(
            provenance=payload["provenance"],
            as_of=payload["as_of"],
            files=tuple((str(p), str(h)) for p, h in payload["files"]),
            sources={str(k): dict(v) for k, v in payload["sources"].items()},
            source_priority={str(k): list(v) for k, v in payload["source_priority"].items()},
            universe=payload["universe"],
            vintage=payload.get("vintage"),
            coverage=payload.get("coverage", {}),
            transformations=list(payload.get("transformations", [])),
            assumptions=payload.get("assumptions", {}),
            fit_cutoff=payload.get("fit_cutoff"),
            feature_code_sha=payload.get("feature_code_sha"),
            feature_params=payload.get("feature_params"),
        )
        if recorded is not None and recorded != manifest.id:
            raise ManifestError(
                f"manifest records id {recorded} but its content hashes to {manifest.id}; "
                "the file was edited after it was written"
            )
        return manifest

    def write(self, path: Path) -> Path:
        """Write ``manifest.json``. The file is written whole; a manifest is never patched."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
