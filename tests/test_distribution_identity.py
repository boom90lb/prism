"""Installed engine identity follows its own RECORD inside private git projects."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import pytest

from prism.live.authorization import AuthorizationRefused, distribution_id


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def private_project(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "private-project"
    project.mkdir()
    _git(project, "init", "-q", "-b", "main")
    (project / "strategy.py").write_text("SCORE = 1\n")
    _git(project, "add", "strategy.py")
    _git(project, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "strategy")
    return project, _git(project, "rev-parse", "HEAD")


def _install(site: Path, monkeypatch: pytest.MonkeyPatch, *, editable: bool = False) -> tuple[Path, Path]:
    package = site / "prism"
    (package / "live").mkdir(parents=True)
    metadata = site / "prism_trading-0.4.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: prism-trading\nVersion: 0.4.0\n")
    record = metadata / "RECORD"
    record.write_text("prism/live/authorization.py,sha256=engine-one,100\n")
    if editable:
        (metadata / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}}))
    distribution = importlib.metadata.PathDistribution(metadata)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    return package / "live", record


def _record_id(record: Path) -> str:
    return "sha256:" + hashlib.sha256(record.read_bytes()).hexdigest()


def test_installed_wheel_inside_private_git_uses_record_and_tracks_record_changes(
    private_project: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, head = private_project
    package, record = _install(project / ".venv/lib/python3.12/site-packages", monkeypatch)

    first = distribution_id(package_root=package)
    assert first == _record_id(record)
    assert first != f"git:{head}"
    # Strategy edits change the enclosing checkout, not the installed engine.
    (project / "strategy.py").write_text("SCORE = 2\n")
    assert distribution_id(package_root=package) == first
    record.write_text("prism/live/authorization.py,sha256=engine-two,101\n")
    assert distribution_id(package_root=package) == _record_id(record)
    assert distribution_id(package_root=package) != first


def test_installed_wheel_missing_record_cannot_fall_back_to_private_git(
    private_project: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = private_project
    package, record = _install(project / ".venv/lib/python3.12/site-packages", monkeypatch)
    record.unlink()

    with pytest.raises(AuthorizationRefused, match="has no RECORD"):
        distribution_id(package_root=package)


def test_source_outside_git_cannot_borrow_another_installation_record(tmp_path: Path, monkeypatch) -> None:
    _install(tmp_path / "different-environment/site-packages", monkeypatch)
    source = tmp_path / "untracked-source/prism/live"
    source.mkdir(parents=True)

    with pytest.raises(AuthorizationRefused, match="no matching installed RECORD"):
        distribution_id(package_root=source)


def test_source_inside_git_keeps_git_identity_when_another_wheel_is_installed(
    private_project: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, head = private_project
    _install(tmp_path / "different-environment/site-packages", monkeypatch)
    source = project / "src/prism/live"
    source.mkdir(parents=True)

    assert distribution_id(package_root=source) == f"git:{head}"
    (project / "strategy.py").write_text("SCORE = 2\n")
    assert distribution_id(package_root=source) == f"git:{head}+dirty"


def test_editable_install_keeps_source_git_identity(
    private_project: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, head = private_project
    package, _ = _install(project, monkeypatch, editable=True)

    assert distribution_id(package_root=package) == f"git:{head}"
