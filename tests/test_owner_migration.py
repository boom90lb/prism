"""The private migration refuses output paths that would mutate its inputs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ops" / "verify_owner_migration.py"


@pytest.mark.parametrize("location", ["dataset", "nested", "symlink"])
def test_output_inside_external_dataset_is_refused_before_any_write(tmp_path: Path, location: str) -> None:
    source = tmp_path / "engine"
    source.mkdir()
    (source / "sentinel").write_bytes(b"source must not change")
    dataset = tmp_path / "private-vendor-dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_bytes(b"{}\n")
    (dataset / "bars.parquet").write_bytes(b"vendor bytes must not change")
    alias = tmp_path / "dataset-alias"
    alias.symlink_to(dataset, target_is_directory=True)
    output = {"dataset": dataset, "nested": dataset / "new-output", "symlink": alias / "new-output"}[location]
    before = {str(path.relative_to(tmp_path)): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(source), "--dataset", str(dataset),
         "--wheel", str(tmp_path / "deliberately-absent.whl"), "--out", str(output)],
        text=True, capture_output=True, check=False,
    )

    assert result.returncode == 2, result.stderr
    expected = "refusing existing output" if location == "dataset" else "outside the frozen input dataset"
    assert expected in result.stderr
    after = {str(path.relative_to(tmp_path)): path.read_bytes()
             for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
    assert not (dataset / "new-output").exists()
    assert not (source / ".venv").exists()
