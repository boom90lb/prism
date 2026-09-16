"""Legacy evidence-history import (engine specification §3.12, stage 2).

The three conditions of "history retained" are the first three sections here:
the four source ledgers stay byte-identical and nothing in this stage writes to
them; every source row has exactly one imported record and every imported
record has a source row; and the map from ``(source path, line number, legacy
config hash)`` to the imported ``run_id`` is a recorded artifact under
``workspace.results`` rather than an implicit product of the import code.

The sources are governed evidence: ``results/stat_arb_residual_trials.jsonl`` is
the closed 17-trial set the live book's DSR 0.191 deflates against and a
certifying artifact of ``docs/certifications/001-residual-reversion-daily-negative.md``.
Every test that runs an import runs it against a *copy* under ``tmp_path``, and
the copies are compared byte for byte afterwards.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import tomllib
from pathlib import Path

import pytest

from prism.history_import import (
    GENESIS_EVENT,
    HISTORY_IMPORT_DIRNAME,
    KIND_IMPORTED,
    KIND_PROVENANCE,
    LEGACY_SOURCES,
    PROVENANCE_LOG_NAME,
    RECORDS_FILENAME,
    REGISTRATIONS_DIRNAME,
    REPORT_FILENAME,
    RUN_ID_MAP_FILENAME,
    HistoryImportError,
    LegacySource,
    chain_hash,
    compare_config_id,
    import_history,
    legacy_run_key,
    mint_run_id,
    verify_trial_log,
)
from prism.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The stage 2 inventory, restated here so a silent edit to the module's
#: declaration fails a test rather than passing one.
EXPECTED_INVENTORY = {
    "results/momentum_v1_trials.jsonl": ("momentum_v1", 6, KIND_IMPORTED),
    "results/trend_v1_trials.jsonl": ("trend_v1", 5, KIND_IMPORTED),
    "results/stat_arb_residual_trials.jsonl": ("stat_arb_residual", 17, KIND_IMPORTED),
    "data/universe/sp500_density_trials.jsonl": (None, 4, KIND_PROVENANCE),
}

TOTAL_ROWS = sum(rows for _, rows, _ in EXPECTED_INVENTORY.values())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def source_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A copy of the four tracked ledgers and the results tree they cite.

    A copy, because the originals are append-only governed evidence: a test
    that imported from the repository in place would be one bug away from
    writing to a certified ledger.
    """
    root = tmp_path_factory.mktemp("source_tree")
    for relative in EXPECTED_INVENTORY:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    for directory in sorted((REPO_ROOT / "results").iterdir()):
        if directory.is_dir():
            shutil.copytree(directory, root / "results" / directory.name)
    return root


@pytest.fixture(scope="module")
def imported(tmp_path_factory: pytest.TempPathFactory, source_tree: Path):
    """One import of the copied tree into a fresh workspace."""
    workspace = Workspace(tmp_path_factory.mktemp("workspace"))
    report = import_history(
        workspace,
        source_root=source_tree,
        imported_at=dt.datetime(2026, 9, 8, 12, 0, tzinfo=dt.timezone.utc),
    )
    directory = workspace.results / HISTORY_IMPORT_DIRNAME
    records = [
        json.loads(line)
        for line in (directory / RECORDS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_id_map = json.loads((directory / RUN_ID_MAP_FILENAME).read_text(encoding="utf-8"))
    return workspace, report, records, run_id_map


# --------------------------------------------------------------------------
# The declared inventory is the tree's
# --------------------------------------------------------------------------


def test_declared_sources_match_the_tracked_ledgers() -> None:
    """Every declared digest, row count and selection set is the tree's."""
    assert {source.path for source in LEGACY_SOURCES} == set(EXPECTED_INVENTORY)
    for source in LEGACY_SOURCES:
        selection_set_id, rows, kind = EXPECTED_INVENTORY[source.path]
        path = REPO_ROOT / source.path
        assert path.is_file(), f"{source.path} is tracked and must be present"
        assert _sha256(path) == source.sha256, f"{source.path}: declared digest is not the tree's"
        assert len(path.read_bytes().splitlines()) == rows == source.rows
        assert source.selection_set_id == selection_set_id
        assert source.kind == kind


def test_declared_budgets_match_the_tracked_records() -> None:
    """Budgets, statuses and denominators come from the tracked records."""
    by_path = {source.path: source for source in LEGACY_SOURCES}
    momentum = by_path["results/momentum_v1_trials.jsonl"]
    assert (momentum.declared_budget, momentum.rows_spent, momentum.status) == (8, 6, "open")
    assert momentum.legacy_independent_count == 8
    trend = by_path["results/trend_v1_trials.jsonl"]
    assert (trend.declared_budget, trend.rows_spent, trend.status) == (6, 5, "open")
    residual = by_path["results/stat_arb_residual_trials.jsonl"]
    assert (residual.rows_spent, residual.status, residual.legacy_independent_count) == (17, "closed", 17)
    density = by_path["data/universe/sp500_density_trials.jsonl"]
    assert density.selection_set_id is None and density.legacy_independent_count is None


# --------------------------------------------------------------------------
# Condition 1: the sources stay byte-identical and nothing writes to them
# --------------------------------------------------------------------------


def test_import_leaves_every_source_byte_identical(tmp_path: Path, source_tree: Path) -> None:
    before = {source.path: (source_tree / source.path).read_bytes() for source in LEGACY_SOURCES}
    stats = {source.path: (source_tree / source.path).stat().st_mtime_ns for source in LEGACY_SOURCES}
    import_history(Workspace(tmp_path / "ws"), source_root=source_tree)
    for source in LEGACY_SOURCES:
        path = source_tree / source.path
        assert path.read_bytes() == before[source.path], f"{source.path} was modified by the import"
        assert _sha256(path) == source.sha256
        assert path.stat().st_mtime_ns == stats[source.path], f"{source.path} was rewritten with identical bytes"


def test_the_import_writes_only_under_the_workspace(tmp_path: Path, source_tree: Path) -> None:
    """No artifact lands beside a source ledger."""
    listing_before = sorted(p.relative_to(source_tree) for p in source_tree.rglob("*"))
    import_history(Workspace(tmp_path / "ws"), source_root=source_tree)
    assert sorted(p.relative_to(source_tree) for p in source_tree.rglob("*")) == listing_before


def test_a_changed_source_is_a_hard_failure(tmp_path: Path, source_tree: Path) -> None:
    """A digest mismatch refuses the import; it never re-baselines."""
    tampered = tmp_path / "tampered"
    shutil.copytree(source_tree, tampered)
    ledger = tampered / "results/stat_arb_residual_trials.jsonl"
    ledger.write_bytes(ledger.read_bytes()[:-1])
    with pytest.raises(HistoryImportError, match="does not match the declared"):
        import_history(Workspace(tmp_path / "ws"), source_root=tampered)


def test_a_row_count_mismatch_is_a_hard_failure(tmp_path: Path, source_tree: Path) -> None:
    """The declared row count is checked independently of the digest."""
    declared = LEGACY_SOURCES[0]
    wrong = LegacySource(
        path=declared.path,
        sha256=declared.sha256,
        rows=declared.rows + 1,
        selection_set_id=declared.selection_set_id,
        kind=declared.kind,
        declared_budget=declared.declared_budget,
        rows_spent=declared.rows + 1,
        status=declared.status,
        legacy_independent_count=declared.legacy_independent_count,
    )
    with pytest.raises(HistoryImportError, match="declared 7"):
        import_history(Workspace(tmp_path / "ws"), source_root=source_tree, sources=[wrong])


def test_a_second_import_is_refused(tmp_path: Path, source_tree: Path) -> None:
    """Re-importing would duplicate rows in an append-only log."""
    workspace = Workspace(tmp_path / "ws")
    import_history(workspace, source_root=source_tree)
    with pytest.raises(HistoryImportError, match="already exists"):
        import_history(workspace, source_root=source_tree)


# --------------------------------------------------------------------------
# Condition 2: one record per source row, in both directions
# --------------------------------------------------------------------------


def test_every_source_row_has_exactly_one_record(imported, source_tree: Path) -> None:
    _, report, records, _ = imported
    assert len(records) == TOTAL_ROWS == report.records
    seen = [
        (record["legacy"]["source_ledger"]["path"], record["legacy"]["source_ledger"]["line"]) for record in records
    ]
    assert len(set(seen)) == len(seen)
    expected = {
        (source.path, line)
        for source in LEGACY_SOURCES
        for line in range(1, len((source_tree / source.path).read_bytes().splitlines()) + 1)
    }
    assert set(seen) == expected


def test_every_record_reproduces_its_source_row(imported, source_tree: Path) -> None:
    """A record's row digest is the digest of the line it names."""
    _, _, records, _ = imported
    for record in records:
        cite = record["legacy"]["source_ledger"]
        raw = (source_tree / cite["path"]).read_bytes().splitlines()[cite["line"] - 1]
        assert record["legacy"]["legacy_row_sha256"] == hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        assert record["legacy"]["legacy_ledger_config_hash"] == payload["config_hash"]
        assert record["legacy"]["legacy_row"]["ts"] == payload["ts"]


def test_run_ids_are_unique_and_minted_from_path_line_and_timestamp(imported, source_tree: Path) -> None:
    _, _, records, _ = imported
    run_ids = [record["ids"]["run_id"] for record in records]
    assert len(set(run_ids)) == len(run_ids) == TOTAL_ROWS
    for record in records:
        cite = record["legacy"]["source_ledger"]
        assert record["ids"]["run_id"] == mint_run_id(cite["path"], cite["line"], record["legacy"]["legacy_row"]["ts"])


def test_trial_logs_hold_one_row_per_record_under_a_genesis_block(imported) -> None:
    workspace, report, records, _ = imported
    logs = {source.selection_set_id or PROVENANCE_LOG_NAME for source in LEGACY_SOURCES}
    assert {Path(path).stem for path in report.trial_log_paths} == logs
    total = 0
    for name in logs:
        rows = verify_trial_log(workspace.trial_log(name))
        genesis, *events = rows
        assert genesis["event"] == GENESIS_EVENT and genesis["prev_hash"] is None
        source = next(s for s in LEGACY_SOURCES if (s.selection_set_id or PROVENANCE_LOG_NAME) == name)
        assert (genesis["source_sha256"], genesis["source_rows"]) == (source.sha256, source.rows)
        assert len(events) == source.rows
        assert {row["run_id"] for row in events} == {
            record["ids"]["run_id"] for record in records if record["legacy"]["source_ledger"]["path"] == source.path
        }
        total += len(events)
    assert total == TOTAL_ROWS


def test_the_chain_detects_a_removed_row(imported, tmp_path: Path) -> None:
    workspace, _, _, _ = imported
    log = workspace.trial_log("stat_arb_residual")
    lines = log.read_text(encoding="utf-8").splitlines()
    shortened = tmp_path / "shortened.jsonl"
    shortened.write_text("\n".join(lines[:5] + lines[6:]) + "\n", encoding="utf-8")
    with pytest.raises(HistoryImportError, match="does not chain to the preceding row"):
        verify_trial_log(shortened)


def test_the_chain_detects_a_rewritten_row(imported, tmp_path: Path) -> None:
    workspace, _, _, _ = imported
    log = workspace.trial_log("momentum_v1")
    lines = log.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[2])
    row["summary"]["annualized_sharpe"] = 9.99
    lines[2] = json.dumps(row, sort_keys=True)
    rewritten = tmp_path / "rewritten.jsonl"
    rewritten.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(HistoryImportError, match="does not cover its content"):
        verify_trial_log(rewritten)


# --------------------------------------------------------------------------
# Condition 3: the mapping is a recorded artifact, not implicit in the code
# --------------------------------------------------------------------------


def test_the_mapping_artifact_is_written_under_workspace_results(imported) -> None:
    workspace, report, _, run_id_map = imported
    path = workspace.results / HISTORY_IMPORT_DIRNAME / RUN_ID_MAP_FILENAME
    assert path.is_file() and Path(report.run_id_map_path) == path
    assert path.is_relative_to(workspace.results)
    assert run_id_map["key"] == ["source_path", "line", "legacy_ledger_config_hash"]


def test_the_mapping_is_complete_in_both_directions(imported, source_tree: Path) -> None:
    _, _, records, run_id_map = imported
    entries = run_id_map["entries"]
    assert len(entries) == TOTAL_ROWS
    keys = {(entry["source_path"], entry["line"], entry["legacy_ledger_config_hash"]) for entry in entries}
    source_keys = set()
    for source in LEGACY_SOURCES:
        for line, raw in enumerate((source_tree / source.path).read_bytes().splitlines(), start=1):
            source_keys.add((source.path, line, json.loads(raw)["config_hash"]))
    assert keys == source_keys, "every source row is keyed in the mapping, and nothing else is"
    assert {entry["run_id"] for entry in entries} == {record["ids"]["run_id"] for record in records}
    by_run_id = {entry["run_id"]: entry for entry in entries}
    for record in records:
        entry = by_run_id[record["ids"]["run_id"]]
        assert (entry["source_path"], entry["line"]) == (
            record["legacy"]["source_ledger"]["path"],
            record["legacy"]["source_ledger"]["line"],
        )
        assert entry["selection_set_id"] == record["ids"]["registration_id"]


def test_the_report_and_registrations_are_recorded_artifacts(imported) -> None:
    workspace, report, _, _ = imported
    directory = workspace.results / HISTORY_IMPORT_DIRNAME
    assert (directory / REPORT_FILENAME).is_file()
    registrations = sorted((directory / REGISTRATIONS_DIRNAME).glob("*.toml"))
    assert [path.stem for path in registrations] == ["momentum_v1", "stat_arb_residual", "trend_v1"]
    tables = {path.stem: tomllib.loads(path.read_text(encoding="utf-8")) for path in registrations}
    assert tables["momentum_v1"]["declared_budget"] == 8
    assert tables["momentum_v1"]["rows_imported"] == 6
    assert tables["momentum_v1"]["legacy_independent_count"] == 8
    assert tables["momentum_v1"]["clock_conditions"] == ["M6 reads on data through 2027-06 or later"]
    assert tables["stat_arb_residual"]["status"] == "closed"
    assert tables["stat_arb_residual"]["rows_imported"] == 17
    assert tables["trend_v1"]["rows_spent"] == 5
    for table in tables.values():
        assert table["kind"] == KIND_IMPORTED
        assert isinstance(table["imported_at"], dt.date)
        assert "arms" not in table, "an imported registration never admits a run"
    assert json.loads((directory / REPORT_FILENAME).read_text(encoding="utf-8"))["records"] == report.records


# --------------------------------------------------------------------------
# Imported row identity (§3.12)
# --------------------------------------------------------------------------


def test_imported_identity_is_explicitly_non_identity(imported) -> None:
    _, _, records, _ = imported
    for record in records:
        assert record["ids"]["config_id"] is None
        assert record["ids"]["data_id"] == "unavailable"
        assert record["evaluation"]["accounting_engine"] == "unavailable"
        assert record["data"]["visibility_mode"] == "unavailable"
        assert record["ids"]["kind"] in (KIND_IMPORTED, KIND_PROVENANCE)
        assert record["status"]["status"] == "finished"


def test_a_null_config_id_is_never_compared(imported) -> None:
    _, _, records, _ = imported
    with pytest.raises(HistoryImportError, match="never comparable"):
        compare_config_id(records[0]["ids"]["config_id"], "0" * 64)
    with pytest.raises(HistoryImportError):
        compare_config_id("0" * 64, None)
    assert compare_config_id("a" * 64, "a" * 64) is True
    assert compare_config_id("a" * 64, "b" * 64) is False


def test_unsourced_fields_are_the_literal_unavailable_never_null(imported) -> None:
    """Only the declared nullable fields are null anywhere in a record."""
    #: ``legacy.legacy_row`` is a verbatim copy of the source row and is not
    #: part of the record's own schema, so a null the ledger wrote stays a null.
    allowed = {
        ("ids", "config_id"),
        ("ids", "registration_id"),
        ("flags", "prospective"),
        ("selection", "legacy_independent_count"),
    }
    _, _, records, _ = imported

    verbatim = ("legacy", "legacy_row")

    def walk(node, path):
        if node is None:
            assert (
                path[:2] == verbatim or path in allowed
            ), f"null at {path}: an unsourced field is the literal 'unavailable'"
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + (key,))
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    for record in records:
        walk(record, ())


def test_both_legacy_digests_are_preserved_verbatim(imported) -> None:
    """The ledger digest and the packet digest differ by construction."""
    _, _, records, _ = imported
    b1 = [record for record in records if record["legacy"]["legacy_row"].get("output_dir") == "results/demotion_b1"]
    assert len(b1) == 2, "the B1 run is row 1 of momentum_v1 and row 16 of stat_arb_residual"
    for record in b1:
        assert record["legacy"]["legacy_ledger_config_hash"] == "301c449b9173"
        assert record["legacy"]["legacy_packet_config_hash"] == "000b74941cfd"
        assert record["legacy"]["legacy_code_commit"] == "a2f44781ff1d2f87e1746fc70060eb17ee9d2918"


def test_a_run_in_two_sets_is_one_record_per_set_sharing_a_run_key(imported) -> None:
    _, report, records, _ = imported
    b1 = [record for record in records if record["legacy"]["legacy_row"].get("output_dir") == "results/demotion_b1"]
    assert len({record["legacy"]["legacy_run_key"] for record in b1}) == 1
    assert {record["ids"]["run_id"] for record in b1} == {record["ids"]["run_id"] for record in b1}
    assert len({record["ids"]["run_id"] for record in b1}) == 2
    for record in b1:
        assert record["ids"]["selection_set_ids"] == ["momentum_v1", "stat_arb_residual"]
    assert {record["ids"]["registration_id"] for record in b1} == {"momentum_v1", "stat_arb_residual"}
    assert report.shared_runs == (
        {
            "legacy_run_key": b1[0]["legacy"]["legacy_run_key"],
            "selection_set_ids": ["momentum_v1", "stat_arb_residual"],
        },
    )
    assert legacy_run_key({"a": 1}) != legacy_run_key({"a": 2})


def test_rows_of_one_set_are_never_pooled_across_sets(imported) -> None:
    """Each set's log holds its own rows, counted once per set."""
    workspace, _, records, _ = imported
    for source in LEGACY_SOURCES:
        if source.selection_set_id is None:
            continue
        rows = verify_trial_log(workspace.trial_log(source.selection_set_id))[1:]
        assert all(row["selection_set_id"] == source.selection_set_id for row in rows)
        assert len(rows) == source.rows
    momentum = [r for r in records if r["ids"]["registration_id"] == "momentum_v1"]
    assert all(record["selection"]["n_trials_observed"] == 6 for record in momentum)


def test_imported_rows_count_as_attempts_and_not_as_independent_evidence(imported) -> None:
    _, _, records, _ = imported
    for record in records:
        selection = record["selection"]
        assert selection["counted_as_independent_evidence"] is False
        if record["ids"]["kind"] == KIND_IMPORTED:
            assert selection["counts_as_attempt"] is True
            assert (
                selection["n_trials_observed"]
                == {
                    "momentum_v1": 6,
                    "trend_v1": 5,
                    "stat_arb_residual": 17,
                }[record["ids"]["registration_id"]]
            )
            assert (
                selection["legacy_independent_count"]
                == {
                    "momentum_v1": 8,
                    "trend_v1": 6,
                    "stat_arb_residual": 17,
                }[record["ids"]["registration_id"]]
            )
        else:
            assert selection["counts_as_attempt"] is False
            assert selection["n_trials_observed"] == "unavailable"


def test_provenance_rows_belong_to_no_selection_set(imported) -> None:
    workspace, report, records, _ = imported
    provenance = [record for record in records if record["ids"]["kind"] == KIND_PROVENANCE]
    assert len(provenance) == 4
    for record in provenance:
        assert record["ids"]["registration_id"] is None
        assert record["ids"]["selection_set_ids"] == []
        assert record["selection"]["legacy_independent_count"] is None
    assert not any(PROVENANCE_LOG_NAME in Path(path).stem for path in report.registration_paths)
    rows = verify_trial_log(workspace.trial_log(PROVENANCE_LOG_NAME))[1:]
    assert len(rows) == 4
    assert all(row["selection_set_id"] is None and row["kind"] == KIND_PROVENANCE for row in rows)


def test_absent_cited_artifacts_are_recorded_as_unavailable(imported) -> None:
    """The eight rows citing ``/home/b90/trading-ensemble/...`` keep their paths."""
    _, report, records, _ = imported
    missing = [
        record
        for record in records
        if any(
            entry["kind"] == "output_dir" and entry["artifact_status"] == "unavailable"
            for entry in record["artifacts"]
        )
    ]
    assert len(missing) == report.rows_with_unavailable_output_dir == 8
    for record in missing:
        entry = record["artifacts"][0]
        assert entry["path"] == record["legacy"]["legacy_row"]["output_dir"]
        assert entry["path"].startswith("/home/b90/trading-ensemble/")
        assert entry["sha256"] == "unavailable"
        assert record["legacy"]["legacy_packet_config_hash"] == "unavailable"
        assert record["legacy"]["legacy_code_commit"] == "unavailable"
    residual_missing = [r for r in missing if r["ids"]["registration_id"] == "stat_arb_residual"]
    assert len(residual_missing) == 4, "four of the seventeen residual rows are counted evidence with no artifacts"


def test_present_artifacts_carry_their_digest(imported, source_tree: Path) -> None:
    _, _, records, _ = imported
    b1 = next(
        record
        for record in records
        if record["legacy"]["source_ledger"]["path"] == "results/momentum_v1_trials.jsonl"
        and record["legacy"]["source_ledger"]["line"] == 1
    )
    packet = next(entry for entry in b1["artifacts"] if entry["kind"] == "claim_packet")
    assert packet["artifact_status"] == "present"
    assert packet["path"] == "results/demotion_b1/claim_packet.json"
    assert packet["sha256"] == hashlib.sha256((source_tree / packet["path"]).read_bytes()).hexdigest()


def test_the_only_derived_values_are_the_row_s_own(imported) -> None:
    """``fold_portfolio`` is derived; the walk block of every row says reset."""
    _, _, records, _ = imported
    assert {record["evaluation"]["fold_portfolio"] for record in records} == {"reset"}
    momentum_b1 = next(
        record
        for record in records
        if record["legacy"]["source_ledger"]["path"] == "results/momentum_v1_trials.jsonl"
        and record["legacy"]["source_ledger"]["line"] == 1
    )
    assert momentum_b1["evaluation"]["capital"] == 1.0
    assert momentum_b1["performance"]["sharpe"]["annualized"] == pytest.approx(0.46544245642237375)
    assert momentum_b1["performance"]["sample"]["n_obs"] == 1308
    assert momentum_b1["coverage"]["membership_coverage"] == pytest.approx(0.9067930489731437)
    assert momentum_b1["flags"]["fill_source"] == "simulated"
    residual_row_one = next(
        record
        for record in records
        if record["legacy"]["source_ledger"]["path"] == "results/stat_arb_residual_trials.jsonl"
        and record["legacy"]["source_ledger"]["line"] == 1
    )
    assert residual_row_one["evaluation"]["capital"] == "unavailable"
    assert residual_row_one["coverage"]["membership_coverage"] == "unavailable"


def test_a_trial_row_hash_covers_its_content(imported) -> None:
    workspace, _, _, _ = imported
    rows = verify_trial_log(workspace.trial_log("trend_v1"))
    for row in rows:
        assert row["hash"] == chain_hash(row)
    assert [row["event"] for row in rows[1:]] == ["finished"] * 5


def test_the_record_names_the_tree_availability_was_read_from(imported, source_tree: Path) -> None:
    """``artifact_status`` is a statement about a tree, and the record says which.

    A checkout without the untracked time-series CSVs is not evidence that a
    certified run's artifacts were destroyed.
    """
    _, report, records, _ = imported
    assert {record["artifact_root"] for record in records} == {str(source_tree.resolve())}
    assert report.source_root == str(source_tree.resolve())


def test_a_row_that_states_nothing_yields_unavailable_never_a_default(tmp_path: Path) -> None:
    """A null or absent legacy value becomes ``"unavailable"``, not a guess.

    The failure this guards against is a null or a plausible default leaking
    into an imported record and being read later as a configured setting.
    """
    ledger = tmp_path / "sparse" / "results" / "sparse_trials.jsonl"
    ledger.parent.mkdir(parents=True)
    row = {
        "ts": "2026-01-02T03:04:05+00:00",
        "config_hash": "0123456789ab",
        "config": {},
        "oos_annualized_sharpe": None,
        "output_dir": "results/gone",
    }
    ledger.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    source = LegacySource(
        path="results/sparse_trials.jsonl",
        sha256=_sha256(ledger),
        rows=1,
        selection_set_id="sparse_set",
        kind=KIND_IMPORTED,
        declared_budget=1,
        rows_spent=1,
        status="open",
        legacy_independent_count=1,
    )
    workspace = Workspace(tmp_path / "ws")
    import_history(workspace, source_root=ledger.parents[1], sources=[source])
    record = json.loads(
        (workspace.results / HISTORY_IMPORT_DIRNAME / RECORDS_FILENAME).read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["evaluation"]["fold_portfolio"] == "unavailable"
    assert record["evaluation"]["capital"] == "unavailable"
    assert record["coverage"]["membership_coverage"] == "unavailable"
    assert record["performance"]["sharpe"]["annualized"] == "unavailable"
    assert record["performance"]["sample"]["n_obs"] == "unavailable"
    assert record["legacy"]["legacy_packet_config_hash"] == "unavailable"
    assert record["legacy"]["legacy_claim_packet"] == "unavailable"
    assert record["artifacts"] == [
        {"path": "results/gone", "kind": "output_dir", "sha256": "unavailable", "artifact_status": "unavailable"}
    ]
