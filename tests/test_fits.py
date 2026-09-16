"""Fitted state as a real artifact (spec §3.3, §3.11; decision D4).

The property that matters is not "a file was written": it is that the reference
recorded beside a decision resolves to *that* fit and to no other. So the tests
here are mostly about refusal — a rewritten artifact, a reference that names
nothing, a component that declares a fit it cannot persist — because a fit store
that silently returns the wrong model is worse than one that has nothing in it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism import fits
from prism.fits import FitArtifactError
from prism.workspace import Workspace


class Stateless:
    fitted_state = "none"


class Stateful:
    """A component whose fit is one number, persisted as JSON."""

    fitted_state = "train_only"

    def __init__(self, level: float | None = None) -> None:
        self.level = level

    def fit(self, level: float) -> "Stateful":
        self.level = float(level)
        return self

    def save_state(self, path: Path) -> None:
        path.write_text(json.dumps({"level": self.level}, sort_keys=True), encoding="utf-8")

    def load_state(self, path: Path) -> "Stateful":
        self.level = json.loads(path.read_text(encoding="utf-8"))["level"]
        return self


class Unpersistable(Stateful):
    def save_state(self, path: Path) -> None:
        raise NotImplementedError("no serialization for the fitted members yet")


class WritesNothing(Stateful):
    def save_state(self, path: Path) -> None:
        return None


def test_a_stateless_component_has_no_artifact_and_a_null_reference(tmp_path: Path) -> None:
    ref = fits.save(Stateless(), component_id="momentum", fits_dir=tmp_path)

    assert ref is None
    assert list(tmp_path.iterdir()) == [], "no empty artifact stands in for a fit that does not exist"


def test_a_fit_is_written_once_per_content_hash_and_reloads(tmp_path: Path) -> None:
    ref = fits.save(Stateful(2.5), component_id="ensemble", fits_dir=tmp_path)

    assert ref is not None and ref.startswith("sha256:") and len(ref) == len("sha256:") + 64
    path = fits.artifact_path(tmp_path, "ensemble", ref)
    assert path.is_file() and path.parent.name == "ensemble"
    assert fits.digest_file(path) == ref
    assert fits.exists(ref, component_id="ensemble", fits_dir=tmp_path)

    # The same fit written again is the same artifact, not a second file.
    again = fits.save(Stateful(2.5), component_id="ensemble", fits_dir=tmp_path)
    assert again == ref
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]

    restored = fits.load(Stateful(), ref, component_id="ensemble", fits_dir=tmp_path)
    assert restored.level == 2.5


def test_two_different_fits_are_two_references(tmp_path: Path) -> None:
    first = fits.save(Stateful(1.0), component_id="ensemble", fits_dir=tmp_path)
    second = fits.save(Stateful(2.0), component_id="ensemble", fits_dir=tmp_path)

    assert first != second
    assert fits.load(Stateful(), first, component_id="ensemble", fits_dir=tmp_path).level == 1.0
    assert fits.load(Stateful(), second, component_id="ensemble", fits_dir=tmp_path).level == 2.0


def test_an_artifact_that_no_longer_hashes_to_its_reference_is_refused(tmp_path: Path) -> None:
    """The whole point of a content hash: a rewritten file is a different model, not a warning."""
    ref = fits.save(Stateful(1.0), component_id="ensemble", fits_dir=tmp_path)
    assert ref is not None
    path = fits.artifact_path(tmp_path, "ensemble", ref)
    path.write_text(json.dumps({"level": 999.0}), encoding="utf-8")

    with pytest.raises(FitArtifactError, match="not the fit that was recorded"):
        fits.load(Stateful(), ref, component_id="ensemble", fits_dir=tmp_path)


def test_a_reference_naming_nothing_in_this_store_is_refused(tmp_path: Path) -> None:
    ref = "sha256:" + "0" * 64

    with pytest.raises(FitArtifactError, match="no fit artifact"):
        fits.load(Stateful(), ref, component_id="ensemble", fits_dir=tmp_path)


def test_a_component_that_declares_a_fit_it_cannot_persist_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FitArtifactError, match="must implement save_state"):
        fits.save(Unpersistable(1.0), component_id="ensemble", fits_dir=tmp_path)

    with pytest.raises(FitArtifactError, match="wrote no file"):
        fits.save(WritesNothing(1.0), component_id="ensemble", fits_dir=tmp_path)
    assert not any(fits.component_dir(tmp_path, "ensemble").iterdir()), "no staging file is left behind"


def test_a_component_that_declares_no_fitted_state_at_all_is_refused(tmp_path: Path) -> None:
    class Undeclared:
        pass

    with pytest.raises(FitArtifactError, match="declares no fitted_state"):
        fits.save(Undeclared(), component_id="x", fits_dir=tmp_path)


@pytest.mark.parametrize("component_id", ["", ".", "..", "a/b", "../escape"])
def test_a_component_id_that_is_not_one_segment_cannot_escape_the_store(component_id: str, tmp_path: Path) -> None:
    with pytest.raises(FitArtifactError, match="component_id"):
        fits.save(Stateful(1.0), component_id=component_id, fits_dir=tmp_path)


@pytest.mark.parametrize("ref", ["", "sha256:", "not-a-hash", "sha256:" + "z" * 64, "0" * 63])
def test_a_malformed_reference_is_refused(ref: str) -> None:
    with pytest.raises(FitArtifactError):
        fits.normalize_ref(ref)


def test_a_bare_hex_digest_still_resolves(tmp_path: Path) -> None:
    """The evaluator recorded fit references without the prefix; those records still resolve."""
    ref = fits.save(Stateful(3.0), component_id="ensemble", fits_dir=tmp_path)
    assert ref is not None
    bare = ref[len("sha256:") :]

    assert fits.normalize_ref(bare) == ref
    assert fits.load(Stateful(), bare, component_id="ensemble", fits_dir=tmp_path).level == 3.0


def test_the_store_lives_under_the_workspace_state_directory(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path).ensure()

    assert workspace.fits() == workspace.state / "fits"
    ref = fits.save(Stateful(1.0), component_id="momentum", fits_dir=workspace.fits())
    assert ref is not None
    assert fits.artifact_path(workspace.fits(), "momentum", ref).is_file()
    assert fits.artifact_path(workspace.fits(), "momentum", ref).is_relative_to(workspace.root)
