"""A fresh reproduction resolves its recorded files and distribution directory.

The source archive can be outside the evaluation workspace. Existing frozen
datasets bypass these lookups, so testing only a cached reproduction hides a
directory rejected by the file-only resolver.
"""

from pathlib import Path

import pytest

from prism.scripts.reproduce import ReproductionError, _recorded_input, open_dataset
from prism.workspace import Workspace


@pytest.fixture
def recorded_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    recorded = source / "results" / "trend_t0"
    recorded.mkdir(parents=True)
    return source, recorded


def test_recorded_file_resolves_from_external_source(recorded_source: tuple[Path, Path]) -> None:
    source, recorded = recorded_source
    membership = source / "data" / "membership.parquet"
    membership.parent.mkdir()
    membership.write_bytes(b"recorded membership")

    assert _recorded_input(recorded, "data/membership.parquet") == membership


def test_recorded_directory_resolves_from_external_source(recorded_source: tuple[Path, Path]) -> None:
    source, recorded = recorded_source
    distributions = source / "data" / "distributions"
    distributions.mkdir(parents=True)

    assert _recorded_input(recorded, "data/distributions", directory=True) == distributions


@pytest.mark.parametrize("want_directory", [False, True])
def test_recorded_input_refuses_the_wrong_kind(
    recorded_source: tuple[Path, Path], want_directory: bool
) -> None:
    source, recorded = recorded_source
    wrong = source / "data" / "wrong-kind"
    wrong.parent.mkdir()
    if want_directory:
        wrong.write_bytes(b"not a directory")
    else:
        wrong.mkdir()

    with pytest.raises(ReproductionError, match="not a directory" if want_directory else "not a file"):
        _recorded_input(recorded, "data/wrong-kind", directory=want_directory)


@pytest.mark.parametrize("want_directory", [False, True])
def test_recorded_input_refuses_missing_inputs(
    recorded_source: tuple[Path, Path], want_directory: bool
) -> None:
    _, recorded = recorded_source

    with pytest.raises(ReproductionError, match="not a directory" if want_directory else "not a file"):
        _recorded_input(recorded, "data/missing", directory=want_directory)


def test_fresh_trend_import_receives_the_recorded_distribution_directory(
    recorded_source: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual uncached entry path must request a directory, before import."""
    source, recorded = recorded_source
    distributions = source / "data" / "distributions"
    distributions.mkdir(parents=True)
    workspace = Workspace(root=tmp_path / "separate-workspace")
    cache = tmp_path / "licensed-cache"
    config = {
        "end_date": "2026-07-17",
        "symbols": ["SPY"],
        "distributions_dir": "data/distributions",
        "cache_files": {"SPY": ["frozen-spy.parquet"]},
    }

    class ImportReached(Exception):
        pass

    def import_spy(root: Path, **kwargs: object) -> None:
        assert root == workspace.data / "frozen" / "trend_t0"
        assert kwargs["distributions_dir"] == distributions
        assert kwargs["cache_dir"] == cache
        assert kwargs["cache_files"] == config["cache_files"]
        assert kwargs["symbols"] == config["symbols"]
        raise ImportReached

    monkeypatch.setattr("prism.scripts.reproduce.import_legacy", import_spy)
    with pytest.raises(ImportReached):
        open_dataset(
            workspace,
            case="trend_t0",
            recorded_config=config,
            cache_dir=cache,
            recorded_dir=recorded,
        )
