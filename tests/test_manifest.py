"""The dataset manifest and its content-hashed identity (engine spec §3.2).

The manifest id is the ``data_id`` every result record carries, so the failure
that matters is a *stable* id across a change that alters what was read: two
evaluations would then claim one provenance while reading different bytes.
The mutation table below therefore walks every field §3.2 hashes and asserts
the id moves, and the second table pins the three fields it deliberately does
not hash, so a future edit that quietly folds one of them in fails here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prism.io.manifest import Manifest, ManifestError, canonical_json, hash_tree, sha256_file


def _source(**overrides: Any) -> dict[str, Any]:
    entry = {
        "lag_bars": 0,
        "revision_policy": "dedupe_identical",
        "price_basis": "split_adjusted_at_ingest",
        "url": "https://example.invalid/bars",
        "sha256": "0" * 64,
        "redistributable": False,
    }
    entry.update(overrides)
    return entry


def _manifest(**overrides: Any) -> Manifest:
    fields: dict[str, Any] = {
        "provenance": "market",
        "as_of": "2026-06-30T00:00:00-04:00",
        "files": (("bars/sec_000001_1d.parquet", "a" * 64), ("securities.parquet", "b" * 64)),
        "sources": {"twelvedata": _source(), "issuer_csv": _source(lag_bars=1)},
        "source_priority": {"bars": ["twelvedata", "issuer_csv"], "events": ["issuer_csv", "twelvedata"]},
        "universe": "c" * 64,
        "vintage": "v1",
        "coverage": {"skipped": []},
        "transformations": ["import-legacy"],
        "assumptions": {"gap_flag_sessions": 15},
        "fit_cutoff": None,
        "feature_code_sha": None,
        "feature_params": None,
    }
    fields.update(overrides)
    return Manifest(**fields)


def _mutations() -> dict[str, dict[str, Any]]:
    """One mutation per field §3.2 puts inside the id."""
    return {
        "file_hash": {"files": (("bars/sec_000001_1d.parquet", "f" * 64), ("securities.parquet", "b" * 64))},
        "file_added": {
            "files": (
                ("bars/sec_000001_1d.parquet", "a" * 64),
                ("bars/sec_000002_1d.parquet", "d" * 64),
                ("securities.parquet", "b" * 64),
            )
        },
        "file_renamed": {"files": (("bars/sec_000009_1d.parquet", "a" * 64), ("securities.parquet", "b" * 64))},
        "as_of": {"as_of": "2026-07-01T00:00:00-04:00"},
        "provenance": {"provenance": "mixed"},
        "universe": {"universe": "e" * 64},
        "transformations": {"transformations": ["import-legacy", "resample"]},
        "assumptions": {"assumptions": {"gap_flag_sessions": 20}},
        "source_lag_bars": {"sources": {"twelvedata": _source(lag_bars=2), "issuer_csv": _source(lag_bars=1)}},
        "source_revision_policy": {
            "sources": {"twelvedata": _source(revision_policy="conflict_raise"), "issuer_csv": _source(lag_bars=1)}
        },
        "source_price_basis": {
            "sources": {"twelvedata": _source(price_basis="raw"), "issuer_csv": _source(lag_bars=1)}
        },
        "source_url": {
            "sources": {"twelvedata": _source(url="https://example.invalid/other"), "issuer_csv": _source(lag_bars=1)}
        },
        "source_sha256": {"sources": {"twelvedata": _source(sha256="9" * 64), "issuer_csv": _source(lag_bars=1)}},
        "source_added": {
            "sources": {"twelvedata": _source(), "issuer_csv": _source(lag_bars=1), "alpaca": _source()},
            "source_priority": {
                "bars": ["twelvedata", "issuer_csv", "alpaca"],
                "events": ["issuer_csv", "twelvedata"],
            },
        },
        "source_priority_order": {
            "source_priority": {"bars": ["issuer_csv", "twelvedata"], "events": ["issuer_csv", "twelvedata"]}
        },
        "fit_cutoff": {"fit_cutoff": "2025-12-31"},
        "feature_code_sha": {"feature_code_sha": "7" * 40},
        "feature_params": {"feature_params": {"lookback_bars": 126}},
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_manifest_id_changes_with_every_hashed_input(name: str) -> None:
    base = _manifest()
    mutated = _manifest(**_mutations()[name])
    assert mutated.id != base.id, f"changing {name} left the manifest id unchanged"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"vintage": "v2"}, id="vintage"),
        pytest.param({"coverage": {"skipped": ["AAPL"]}}, id="coverage"),
        pytest.param(
            {"sources": {"twelvedata": _source(redistributable=True), "issuer_csv": _source(lag_bars=1)}},
            id="redistributable",
        ),
    ],
)
def test_manifest_id_excludes_the_three_declared_non_identity_fields(overrides: dict[str, Any]) -> None:
    """A vintage relabel, a coverage note, or a licence flag must not invalidate a recorded result."""
    assert _manifest(**overrides).id == _manifest().id


def test_file_order_does_not_change_the_id() -> None:
    reversed_files = tuple(reversed(_manifest().files))
    assert _manifest(files=reversed_files).id == _manifest().id


def test_id_is_stable_across_processes() -> None:
    """The id is a hash of canonical JSON, not of a Python repr with a salted hash seed."""
    payload = _manifest().identity_payload()
    assert canonical_json(payload) == canonical_json(json.loads(canonical_json(payload)))
    assert _manifest().id == _manifest().id


def test_round_trip_through_json_preserves_the_id(tmp_path: Path) -> None:
    manifest = _manifest()
    path = manifest.write(tmp_path / "manifest.json")
    assert Manifest.read(path).id == manifest.id


def test_reading_an_edited_manifest_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    path = manifest.write(tmp_path / "manifest.json")
    payload = json.loads(path.read_text())
    payload["as_of"] = "2027-01-04T00:00:00-05:00"
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestError, match="hashes to"):
        Manifest.read(path)


def test_a_source_outside_source_priority_is_refused() -> None:
    with pytest.raises(ManifestError, match="appear in no source_priority order"):
        _manifest(source_priority={"bars": ["twelvedata"]})


def test_source_priority_naming_an_unknown_source_is_refused() -> None:
    with pytest.raises(ManifestError, match="unknown source"):
        _manifest(source_priority={"bars": ["twelvedata", "issuer_csv", "polygon"], "events": ["issuer_csv"]})


def test_priority_for_an_unordered_table_is_refused() -> None:
    with pytest.raises(ManifestError, match="no source_priority for table"):
        _manifest().priority_for("membership")


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"provenance": "vendor"}, "provenance must be one of"),
        ({"sources": {"x": _source(lag_bars=-1)}}, "lag_bars must be a non-negative int"),
        ({"sources": {"x": _source(revision_policy="overwrite")}}, "revision_policy must be one of"),
        ({"sources": {"x": _source(price_basis="total_return")}}, "price_basis must be one of"),
        ({"as_of": ""}, "as_of is required"),
    ],
)
def test_manifest_validation_refuses_a_malformed_field(overrides: dict[str, Any], match: str) -> None:
    if "sources" in overrides:
        overrides["source_priority"] = {"bars": ["x"]}
    with pytest.raises(ManifestError, match=match):
        _manifest(**overrides)


def test_verify_files_catches_a_changed_byte(tmp_path: Path) -> None:
    target = tmp_path / "securities.parquet"
    target.write_bytes(b"one")
    manifest = _manifest(files=tuple(hash_tree(tmp_path, [target])))
    manifest.verify_files(tmp_path)
    target.write_bytes(b"two")
    with pytest.raises(ManifestError, match="the manifest records"):
        manifest.verify_files(tmp_path)


def test_hash_tree_refuses_a_file_outside_the_data_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"x")
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(ManifestError, match="not under the data root"):
        hash_tree(root, [outside])


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "blob.bin"
    payload = b"prism" * 10_000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_canonical_json_refuses_a_non_finite_number() -> None:
    with pytest.raises(ValueError):
        canonical_json({"assumption": float("nan")})
