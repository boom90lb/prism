"""Legacy evidence-history import: the four tracked ledgers (§3.12, stage 2).

The migration inherits four append-only research ledgers written by the
retiring drivers. They are the denominators the reference book's certified
statistics deflate against, so they are imported rather than restated: each
source row becomes exactly one ``imported`` (or ``provenance``) record, the
sources stay byte-identical and read-only, and the map from a source row to the
``run_id`` it was given is written to a recorded artifact under
``workspace.results`` instead of living implicitly in this code.

**Identity of an imported row** (§3.12, "Imported records"). ``config_id`` is
``None`` — never comparable with a post-overhaul ``config_id``, which
:func:`compare_config_id` enforces by raising rather than answering; ``data_id``
is the literal ``"unavailable"``; ``run_id`` is minted from the source path, the
line number and the row's own timestamp; ``selection_set_id`` comes from the
declared mapping in :data:`LEGACY_SOURCES` and from nowhere else; both legacy
digests are preserved verbatim under distinct names,
``legacy_ledger_config_hash`` (the driver's ``stable_config_hash``) and
``legacy_packet_config_hash`` (the claim packet's digest of the same run), with
``legacy_code_commit`` where a packet exists. Every other field with no source
in the legacy row takes the literal string ``"unavailable"``, never a null and
never a plausible default, so an imported record cannot be read as a configured
setting. ``accounting_engine`` is ``"unavailable"``: the legacy driver is not
the reproduction engine.

**What is derived, and from what.** Only these, each from a value present in the
row itself: ``fold_portfolio`` from ``config.walk.close_positions_at_fold_end``;
``capital`` from ``config.initial_capital``; ``coverage.membership_coverage``
from ``config.universe.coverage.coverage_fraction``; the sample counts and the
two Sharpe figures the ledger records, carried under ``performance`` and labeled
with their source; and the artifact list, from ``output_dir`` plus the claim
packet's own artifact map when that packet exists. ``fill_source`` is
``"simulated"`` because every one of these rows is a walk-forward simulation and
displaying it unlabeled beside broker fills is the failure this field exists to
prevent (§3.12, "Comparison recording").

**Counting.** An imported row counts in its selection set's attempt accounting
and is excluded from independent-evidence counts, the treatment replay rows
receive; each set's own denominator is preserved verbatim as
``legacy_independent_count`` on the record and in the imported registration this
module writes, so no certified denominator shrinks. ``provenance`` rows carry
``selection_set_id = None``, write no registration and enter no count at all.

**Why there is no ``prism`` subcommand.** The command surface of §3.14 is closed
and declares no import command; the import is a one-time stage 2 migration, run
as ``python -m prism.history_import --workspace <dir>`` and thereafter evidenced
by the artifacts it wrote. ``prism trials reconcile --source <ledger>`` is the
declared command for what happens *after* an import: divergence between a source
ledger and its genesis block.

:func:`genesis_block` lives here because the import *is* the chain's genesis
(§3.12, "Hash chain"). The row hash itself is
:func:`prism.validation.trials.chain_hash`, re-exported here and shared with the
evaluator's trial-log writer: two implementations of a chain hash are two
formats, and the second one is discovered when a log fails to verify.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from prism.spec import UNAVAILABLE, engine_version
from prism.validation.trials import chain_hash, stable_json_dumps
from prism.workspace import Workspace, resolve_workspace

#: ``workspace.results / HISTORY_IMPORT_DIRNAME`` holds every artifact the
#: import writes. Nothing is written beside a source ledger.
HISTORY_IMPORT_DIRNAME = "history_import"
RUN_ID_MAP_FILENAME = "run_id_map.json"
RECORDS_FILENAME = "records.jsonl"
REPORT_FILENAME = "report.json"
REGISTRATIONS_DIRNAME = "registrations"

RECORD_SCHEMA = 1
RUN_ID_MAP_SCHEMA = 1
REPORT_SCHEMA = 1

#: Record kinds of §3.12's run-recording protocol used by this module.
KIND_IMPORTED = "imported"
KIND_PROVENANCE = "provenance"

#: The chain anchor is not a run event: the ``{started, finished, failed,
#: incomplete}`` enum describes runs, and a genesis block describes a file.
GENESIS_EVENT = "import_genesis"

#: Provenance rows belong to no selection set, so they cannot route to
#: ``trials/<selection_set_id>.jsonl``. They go to a log of their own, which is
#: in no selection set and enters no attempt or independent count.
PROVENANCE_LOG_NAME = "provenance"

#: Legacy ledger rows are appended by the driver *after* a completed walk
#: forward, so an imported row's terminal event is ``finished``; the record's
#: ``status`` says the same thing, as §3.12 requires of the pair.
IMPORTED_EVENT = "finished"
IMPORTED_STATUS = "finished"


class HistoryImportError(RuntimeError):
    """The import cannot proceed: a source, a count, or a target is wrong.

    Raised rather than warned. A source ledger whose digest or row count does
    not match the declaration below is not imported at all: the identity of
    every record depends on the file being the one that was declared.
    """


@dataclass(frozen=True)
class LegacySource:
    """One tracked source ledger and everything declared about it.

    ``sha256`` and ``rows`` are the checked identity of the file. The
    registration keys (``declared_budget`` through ``clock_conditions``) are the
    tracked records' statement of the set's budget, status and clock, and are
    what :func:`registration_toml` writes so that a closed set stays closed and
    an open set's remaining space survives the migration (§3.13).
    """

    path: str
    sha256: str
    rows: int
    selection_set_id: str | None
    kind: str
    declared_budget: int | None = None
    rows_spent: int | None = None
    status: str | None = None
    legacy_independent_count: int | None = None
    registration_doc: str | None = None
    clock_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in (KIND_IMPORTED, KIND_PROVENANCE):
            raise HistoryImportError(f"{self.path}: kind must be {KIND_IMPORTED!r} or {KIND_PROVENANCE!r}")
        if (self.selection_set_id is None) != (self.kind == KIND_PROVENANCE):
            raise HistoryImportError(
                f"{self.path}: a provenance source has no selection set and an imported one must have one"
            )
        if self.kind == KIND_IMPORTED and self.rows_spent != self.rows:
            raise HistoryImportError(
                f"{self.path}: declared rows_spent {self.rows_spent} does not equal declared rows {self.rows}"
            )

    @property
    def log_name(self) -> str:
        """The trial log this source's rows append to."""
        return self.selection_set_id or PROVENANCE_LOG_NAME


#: The four tracked ledgers of the stage 2 inventory, with the digests and row
#: counts checked at import and the budgets, statuses and clocks the tracked
#: records state: ``docs/momentum_m_series_read_2026-09-03.md`` (8 registered, 6
#: spent, M6 on its clock), ``docs/trend_design.md`` and ``SPEC.md`` (6 counted
#: trials, 5 spent, T5 on its clock), and
#: ``docs/certifications/001-residual-reversion-daily-negative.md`` (closed at
#: 17 rows; the 17-trial set the live book's DSR 0.191 deflates against).
LEGACY_SOURCES: tuple[LegacySource, ...] = (
    LegacySource(
        path="results/momentum_v1_trials.jsonl",
        sha256="6e1cfdf877d5e75a77ec3fd5ccd82d50ae6f2bc1a84207c356cc2ec718eee6d9",
        rows=6,
        selection_set_id="momentum_v1",
        kind=KIND_IMPORTED,
        declared_budget=8,
        rows_spent=6,
        status="open",
        legacy_independent_count=8,
        registration_doc="docs/momentum_m_series_read_2026-09-03.md",
        clock_conditions=("M6 reads on data through 2027-06 or later",),
    ),
    LegacySource(
        path="results/trend_v1_trials.jsonl",
        sha256="a4c1f245bd730c89203bc354ffea9390dd2a82f83a434267bd4c97e89bf43687",
        rows=5,
        selection_set_id="trend_v1",
        kind=KIND_IMPORTED,
        declared_budget=6,
        rows_spent=5,
        status="open",
        legacy_independent_count=6,
        registration_doc="docs/trend_design.md",
        clock_conditions=("T5 reads on data through 2027-07-17 or later",),
    ),
    LegacySource(
        path="results/stat_arb_residual_trials.jsonl",
        sha256="92a1a5e478e8318cb546b5971038a641d4cb0c9bccd53cb4b2b6142efbde3a96",
        rows=17,
        selection_set_id="stat_arb_residual",
        kind=KIND_IMPORTED,
        declared_budget=17,
        rows_spent=17,
        status="closed",
        legacy_independent_count=17,
        registration_doc="docs/certifications/001-residual-reversion-daily-negative.md",
        clock_conditions=(),
    ),
    LegacySource(
        path="data/universe/sp500_density_trials.jsonl",
        sha256="a27db9c31705d972b238274b2bb47e2b2032966dab2cbf02d06a645f12ea91fa",
        rows=4,
        selection_set_id=None,
        kind=KIND_PROVENANCE,
    ),
)


@dataclass(frozen=True)
class LegacyRow:
    """One parsed source row, kept beside the bytes it was parsed from."""

    source: LegacySource
    line: int
    raw: bytes
    payload: dict[str, Any]

    @property
    def ts(self) -> str:
        value = self.payload.get("ts")
        if not isinstance(value, str) or not value:
            raise HistoryImportError(
                f"{self.source.path}:{self.line}: row has no 'ts'; the run_id is minted from it and cannot be faked"
            )
        return value

    @property
    def ledger_config_hash(self) -> str:
        value = self.payload.get("config_hash")
        if not isinstance(value, str) or not value:
            raise HistoryImportError(
                f"{self.source.path}:{self.line}: row has no 'config_hash'; the mapping artifact is keyed on it"
            )
        return value

    @property
    def output_dir(self) -> str | None:
        value = self.payload.get("output_dir")
        return value if isinstance(value, str) and value else None

    @property
    def run_id(self) -> str:
        return mint_run_id(self.source.path, self.line, self.ts)

    @property
    def run_key(self) -> str:
        return legacy_run_key(self.payload)


@dataclass(frozen=True)
class HistoryImportReport:
    """What one import read and wrote. Serialized to ``report.json``."""

    imported_at: str
    engine_version: str
    source_root: str
    workspace_root: str
    sources: tuple[dict[str, Any], ...]
    records: int
    run_id_map_path: str
    records_path: str
    report_path: str
    registration_paths: tuple[str, ...]
    trial_log_paths: tuple[str, ...]
    shared_runs: tuple[dict[str, Any], ...]
    rows_with_unavailable_output_dir: int
    artifacts_present: int
    artifacts_unavailable: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_schema": REPORT_SCHEMA,
            "imported_at": self.imported_at,
            "engine_version": self.engine_version,
            "source_root": self.source_root,
            "workspace_root": self.workspace_root,
            "sources": [dict(entry) for entry in self.sources],
            "records": self.records,
            "run_id_map_path": self.run_id_map_path,
            "records_path": self.records_path,
            "report_path": self.report_path,
            "registration_paths": list(self.registration_paths),
            "trial_log_paths": list(self.trial_log_paths),
            "shared_runs": [dict(entry) for entry in self.shared_runs],
            "rows_with_unavailable_output_dir": self.rows_with_unavailable_output_dir,
            "artifacts_present": self.artifacts_present,
            "artifacts_unavailable": self.artifacts_unavailable,
        }


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def mint_run_id(source_path: str, line: int, ts: str) -> str:
    """The imported ``run_id``: source path, line number and row timestamp.

    Deterministic, so re-running the import on the same file mints the same
    ids and a divergence is a real divergence rather than a fresh identity.
    """
    payload = {"source_path": str(source_path), "line": int(line), "ts": str(ts)}
    return "imported-" + hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()[:16]


def legacy_run_key(payload: Mapping[str, Any]) -> str:
    """The key identifying one legacy *run* across the sets it appears in.

    Row 1 of ``results/momentum_v1_trials.jsonl`` is a byte copy of row 16 of
    ``results/stat_arb_residual_trials.jsonl``: one run, two selection sets. The
    import mints one record per (run, set) pair; both carry this key, and both
    list every set the run belongs to in ``selection_set_ids``. Such rows count
    once per set and are never pooled across sets.
    """
    return "legacyrun-" + hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()[:16]


def compare_config_id(left: str | None, right: str | None) -> bool:
    """Compare two ``config_id`` values, refusing a null one.

    An imported record's ``config_id`` is null by construction and is never
    comparable with a post-overhaul ``config_id``: the legacy driver hashed a
    different payload with a different canonicalization. Answering ``False``
    would let a caller read "different configuration" out of "no configuration
    identity", so this raises instead.
    """
    if left is None or right is None:
        raise HistoryImportError(
            "a null config_id belongs to an imported legacy row and is never comparable with a "
            "post-overhaul config_id (§3.12); compare legacy_ledger_config_hash instead"
        )
    return left == right


def genesis_block(source: LegacySource, *, imported_at: str, version: str) -> dict[str, Any]:
    """The chain anchor for one imported source (§3.12, "Hash chain").

    Its hash covers the source file's sha256 and row count at import, so a
    later ``prism trials reconcile --source <path>`` can detect that the file
    has grown or changed without rewriting the block.
    """
    row: dict[str, Any] = {
        "ts": imported_at,
        "event": GENESIS_EVENT,
        "kind": source.kind,
        "selection_set_id": source.selection_set_id,
        "source_path": source.path,
        "source_sha256": source.sha256,
        "source_rows": source.rows,
        "engine_version": version,
        "imported_by": "prism.history_import",
        "prev_hash": None,
    }
    row["hash"] = chain_hash(row)
    return row


# --------------------------------------------------------------------------
# Reading the sources (read-only, always)
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """``sha256`` of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(source_root: Path, source: LegacySource) -> tuple[LegacyRow, ...]:
    """Read and verify one source ledger. Never opens it for writing.

    A digest or row-count mismatch is a hard failure: these files are the
    denominators certified claims deflate against, and importing a file that is
    not the declared one would silently re-found those claims on other evidence.
    """
    path = Path(source_root) / source.path
    if not path.is_file():
        raise HistoryImportError(f"source ledger {path} is missing; the import cannot invent it")
    digest = sha256_file(path)
    if digest != source.sha256:
        raise HistoryImportError(
            f"{source.path}: sha256 {digest} does not match the declared {source.sha256}; "
            "a source ledger is immutable evidence, so a mismatch is a finding, not a new baseline"
        )
    raw_lines = path.read_bytes().splitlines()
    rows: list[LegacyRow] = []
    for index, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            raise HistoryImportError(f"{source.path}:{index}: blank line in an append-only ledger")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HistoryImportError(f"{source.path}:{index}: unreadable row: {exc}") from exc
        if not isinstance(payload, dict):
            raise HistoryImportError(f"{source.path}:{index}: row is not a JSON object")
        rows.append(LegacyRow(source=source, line=index, raw=bytes(raw), payload=payload))
    if len(rows) != source.rows:
        raise HistoryImportError(
            f"{source.path}: read {len(rows)} rows, declared {source.rows}; "
            "an appended source is reported by 'prism trials reconcile --source', never absorbed here"
        )
    return tuple(rows)


def _claim_packet(source_root: Path, output_dir: str | None) -> tuple[dict[str, Any] | None, Path | None]:
    """The run's ``claim_packet.json``, or ``(None, None)`` when absent."""
    if output_dir is None:
        return None, None
    directory = _resolve_output_dir(source_root, output_dir)
    packet_path = directory / "claim_packet.json"
    if not packet_path.is_file():
        return None, None
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoryImportError(f"{packet_path}: unreadable claim packet: {exc}") from exc
    if not isinstance(packet, dict):
        raise HistoryImportError(f"{packet_path}: claim packet is not a JSON object")
    return packet, packet_path


def _resolve_output_dir(source_root: Path, output_dir: str) -> Path:
    candidate = Path(output_dir)
    return candidate if candidate.is_absolute() else Path(source_root) / candidate


def _artifacts(source_root: Path, row: LegacyRow, packet: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The row's cited artifacts, each with its status on disk *now*.

    The original path string is preserved whatever its status: a cited artifact
    that no longer exists is recorded as ``artifact_status = "unavailable"``,
    never silently dropped and never re-created (§3.12). The directory entry
    carries no digest — a directory has no content hash — and the files the
    claim packet cites carry theirs when they are present.

    "Exists" is relative to the tree the import read, which the record names in
    ``artifact_root`` and the report in ``source_root``: a checkout without the
    untracked time-series CSVs is not evidence that they were destroyed.
    """
    output_dir = row.output_dir
    if output_dir is None:
        return []
    directory = _resolve_output_dir(source_root, output_dir)
    present = directory.is_dir()
    entries: list[dict[str, Any]] = [
        {
            "path": output_dir,
            "kind": "output_dir",
            "sha256": UNAVAILABLE,
            "artifact_status": "present" if present else "unavailable",
        }
    ]
    cited = packet.get("artifacts") if isinstance(packet, Mapping) else None
    if isinstance(cited, Mapping):
        for name in sorted(cited):
            filename = cited[name]
            if not isinstance(filename, str) or not filename:
                continue
            file_path = directory / filename
            exists = file_path.is_file()
            entries.append(
                {
                    "path": f"{output_dir.rstrip('/')}/{filename}",
                    "kind": name,
                    "sha256": sha256_file(file_path) if exists else UNAVAILABLE,
                    "artifact_status": "present" if exists else "unavailable",
                }
            )
    return entries


# --------------------------------------------------------------------------
# The imported record (§3.12)
# --------------------------------------------------------------------------


def _derived_fold_portfolio(row: LegacyRow) -> str:
    """``"reset"`` when the row's walk closes positions at a fold end.

    The one derivable value of §3.12. Absent, it is ``"unavailable"``: the
    accounting a driver used is not guessed from a default.
    """
    value = _from_config(row, "walk", "close_positions_at_fold_end")
    if not isinstance(value, bool):
        return UNAVAILABLE
    return "reset" if value else "carry"


def _cited_path(path: Path | None, source_root: Path) -> str:
    """A cited file as a path relative to the imported tree when it is inside it."""
    if path is None:
        return UNAVAILABLE
    try:
        return str(path.relative_to(source_root))
    except ValueError:
        return str(path)


def _stated(value: Any) -> Any:
    """A legacy value, or ``"unavailable"`` when the row states none.

    A JSON null in a legacy row is an absent value, not a configured one, and
    §3.12 admits only one spelling for that.
    """
    return UNAVAILABLE if value is None else value


def _packet_field(packet: Mapping[str, Any] | None, key: str) -> Any:
    """One claim-packet digest or commit, or ``"unavailable"`` with no packet."""
    if not isinstance(packet, Mapping):
        return UNAVAILABLE
    value = packet.get(key)
    return value if isinstance(value, str) and value else UNAVAILABLE


def _from_config(row: LegacyRow, *keys: str) -> Any:
    value: Any = row.payload.get("config", {})
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return UNAVAILABLE
        value = value[key]
    return _stated(value)


def build_record(
    row: LegacyRow,
    *,
    source_root: Path,
    selection_set_ids: Sequence[str],
    n_trials_observed: int | str,
) -> dict[str, Any]:
    """One ``ResultRecord`` for one source row (§3.12, "Imported records")."""
    packet, packet_path = _claim_packet(source_root, row.output_dir)
    artifacts = _artifacts(source_root, row, packet)
    payload = row.payload
    warnings = [
        f"imported legacy row {row.source.path}:{row.line}; config_id is null and is never comparable with a "
        "post-overhaul config_id"
    ]
    missing = [entry["path"] for entry in artifacts if entry["artifact_status"] == "unavailable"]
    if missing:
        warnings.append("cited artifacts absent at import: " + ", ".join(missing))
    coverage_fraction = _from_config(row, "universe", "coverage", "coverage_fraction")
    record: dict[str, Any] = {
        "record_schema": RECORD_SCHEMA,
        "ids": {
            "run_id": row.run_id,
            "config_id": None,
            "data_id": UNAVAILABLE,
            "kind": row.source.kind,
            "arm_id": UNAVAILABLE,
            "registration_id": row.source.selection_set_id,
            "registration_hash": UNAVAILABLE,
            "selection_set_ids": list(selection_set_ids),
            "reproduces": [],
            "supersedes": [],
        },
        "status": {"status": IMPORTED_STATUS, "incomplete_reasons": []},
        "versions": {name: UNAVAILABLE for name in ("engine", "solver", "numpy", "pandas", "python")},
        "conventions": {
            name: UNAVAILABLE
            for name in (
                "returns",
                "scoring_basis",
                "price_basis",
                "executable_return",
                "excess_return_definition",
                "hurdle_series_id",
                "hurdle_conversion",
                "cash_and_financing",
                "execution_profile",
            )
        },
        "data": {
            name: UNAVAILABLE
            for name in ("visibility_mode", "vintage", "manifest_id", "lag_assumptions", "score_panel")
        },
        "coverage": {
            "membership_coverage": coverage_fraction,
            "missing_securities": UNAVAILABLE,
            "terminal_outcomes": UNAVAILABLE,
            "provisional_sid_count": UNAVAILABLE,
            "quarantined_sid_count": UNAVAILABLE,
            "untested_conclusions": UNAVAILABLE,
            "bias_status": "unquantified",
        },
        "costs": {
            name: UNAVAILABLE for name in ("stack", "calibration_source", "cost_refresh", "dollar_volume_statistics")
        },
        "evaluation": {
            "fold_geometry": UNAVAILABLE,
            "excluded_folds": UNAVAILABLE,
            "fold_portfolio": _derived_fold_portfolio(row),
            "train_only_transforms": UNAVAILABLE,
            "calibration": UNAVAILABLE,
            "risk_model": UNAVAILABLE,
            "accounting_engine": UNAVAILABLE,
            "capital": _from_config(row, "initial_capital"),
            "components": UNAVAILABLE,
        },
        "performance": {
            "periodic_returns_path": UNAVAILABLE,
            "gross": UNAVAILABLE,
            "net": UNAVAILABLE,
            "sharpe": {
                "annualized": _stated(payload.get("oos_annualized_sharpe", UNAVAILABLE)),
                "periodic": _stated(payload.get("oos_periodic_sharpe", UNAVAILABLE)),
                "se_method": UNAVAILABLE,
                "source": "legacy_ledger_row",
            },
            "sample": {
                "n_obs": _stated(payload.get("n_obs", UNAVAILABLE)),
                "n_folds": _stated(payload.get("n_folds", UNAVAILABLE)),
                "n_symbols": _stated(payload.get("n_symbols", UNAVAILABLE)),
                "source": "legacy_ledger_row",
            },
            "drawdown": UNAVAILABLE,
            "tails": UNAVAILABLE,
            "turnover": UNAVAILABLE,
            "exposures": UNAVAILABLE,
            "violations": UNAVAILABLE,
            "repair": UNAVAILABLE,
            "binding_conditions": UNAVAILABLE,
        },
        "selection": {
            "n_trials_observed": n_trials_observed,
            "attempt_basis": "legacy_ledger_row",
            "dependence_adjustment": {"method": UNAVAILABLE, "value": UNAVAILABLE},
            "unobserved_search_note": (
                "the legacy ledger records the runs that were appended; it does not prove the search was exhaustive"
            ),
            "legacy_independent_count": row.source.legacy_independent_count,
            "counts_as_attempt": row.source.kind == KIND_IMPORTED,
            "counted_as_independent_evidence": False,
        },
        "flags": {
            "synthetic": False,
            "exploratory": False,
            "prospective": None,
            "fill_source": "simulated",
            "run_channel": "evidence",
            "evidence_tier": UNAVAILABLE,
        },
        "legacy": {
            "legacy_ledger_config_hash": row.ledger_config_hash,
            "legacy_packet_config_hash": _packet_field(packet, "config_hash"),
            "legacy_code_commit": _packet_field(packet, "code_commit"),
            "legacy_run_key": row.run_key,
            "legacy_claim_packet": _cited_path(packet_path, source_root),
            "legacy_row_sha256": hashlib.sha256(row.raw).hexdigest(),
            "legacy_row": {key: value for key, value in payload.items() if key != "config"},
            "legacy_config_sha256": hashlib.sha256(
                stable_json_dumps(payload.get("config", {})).encode("utf-8")
            ).hexdigest(),
            "source_ledger": {"path": row.source.path, "line": row.line, "sha256": row.source.sha256},
        },
        "artifact_root": str(source_root),
        "artifacts": artifacts,
        "warnings": warnings,
    }
    return record


def trial_row(record: Mapping[str, Any], *, prev_hash: str | None) -> dict[str, Any]:
    """The run-recording protocol row for one imported record (§3.12).

    ``pid``, ``host`` and ``heartbeat_at`` are ``"unavailable"``: the legacy row
    records no process, and a null would read as "no process was recorded by a
    protocol that records one".
    """
    legacy_row = record["legacy"]["legacy_row"]
    row: dict[str, Any] = {
        "ts": legacy_row.get("ts", UNAVAILABLE),
        "event": IMPORTED_EVENT,
        "run_id": record["ids"]["run_id"],
        "config_id": None,
        "data_id": UNAVAILABLE,
        "kind": record["ids"]["kind"],
        "arm_id": UNAVAILABLE,
        "selection_set_id": record["ids"]["registration_id"],
        "capital": record["evaluation"]["capital"],
        "synthetic": False,
        "run_channel": record["flags"]["run_channel"],
        "summary": {
            "annualized_sharpe": record["performance"]["sharpe"]["annualized"],
            "periodic_sharpe": record["performance"]["sharpe"]["periodic"],
            "n_obs": record["performance"]["sample"]["n_obs"],
            "output_dir": legacy_row.get("output_dir", UNAVAILABLE),
        },
        "reason": None,
        "pid": UNAVAILABLE,
        "host": UNAVAILABLE,
        "heartbeat_at": UNAVAILABLE,
        "prev_hash": prev_hash,
    }
    row["hash"] = chain_hash(row)
    return row


# --------------------------------------------------------------------------
# The imported registration (§3.13)
# --------------------------------------------------------------------------


def registration_toml(source: LegacySource, *, imported_at: dt.date) -> str:
    """The imported registration for one selection set (§3.13).

    The reduced key set: it never admits a run. A closed set refuses every new
    configuration; an open set requires the owner to declare its remaining space
    as a new registration with ``[[arms]]`` before a counted run may target it.
    """
    if source.selection_set_id is None:
        raise HistoryImportError(f"{source.path}: a provenance source writes no registration")

    def _string(value: str) -> str:
        return json.dumps(value)

    lines = [
        f"# registrations/imported/{source.selection_set_id}.toml — written by prism.history_import (§3.13).",
        "# An imported registration never admits a run: it records the set's spent rows, remaining",
        "# budget, status and clock so the migration loses none of them.",
        f"id = {_string(source.selection_set_id)}",
        f"kind = {_string(KIND_IMPORTED)}",
        f"imported_at = {imported_at.isoformat()}",
        f"source_ledger = {_string(source.path)}",
        f"source_sha256 = {_string(source.sha256)}",
        f"rows_imported = {source.rows}",
    ]
    if source.declared_budget is not None:
        lines.append(f"declared_budget = {source.declared_budget}")
    if source.rows_spent is not None:
        lines.append(f"rows_spent = {source.rows_spent}")
    if source.status is not None:
        lines.append(f"status = {_string(source.status)}")
    if source.legacy_independent_count is not None:
        lines.append(f"legacy_independent_count = {source.legacy_independent_count}")
    if source.registration_doc is not None:
        lines.append(f"registration_doc = {_string(source.registration_doc)}")
    conditions = ", ".join(_string(item) for item in source.clock_conditions)
    lines.append(f"clock_conditions = [{conditions}]")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# The import
# --------------------------------------------------------------------------


def _write_new(path: Path, text: str) -> None:
    """Create ``path`` with ``text``; refuse to overwrite.

    ``"x"`` is ``O_CREAT | O_EXCL``: it is both the exclusive creation the
    append protocol requires and the refusal a second import of the same
    history deserves. Re-importing over a written history would duplicate rows
    in an append-only log.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise HistoryImportError(
            f"{path} already exists; the history has been imported already. Re-importing would duplicate rows "
            "in an append-only log — use 'prism trials reconcile --source <ledger>' for a source that has grown"
        ) from exc


def import_history(
    workspace: Workspace,
    *,
    source_root: Path | str | None = None,
    sources: Iterable[LegacySource] = LEGACY_SOURCES,
    imported_at: dt.datetime | None = None,
) -> HistoryImportReport:
    """Import the declared source ledgers into ``workspace``.

    Writes, all under the workspace and never beside a source: the imported
    records, the ``(source path, line number, legacy config hash) -> run_id``
    map, the report, one imported registration per selection set, and one trial
    log per set (and one for provenance rows) opened with a genesis block whose
    hash covers the source file's digest and row count.
    """
    workspace.ensure()
    root = Path(source_root) if source_root is not None else workspace.root
    root = Path(root).expanduser().resolve()
    moment = imported_at or dt.datetime.now(dt.timezone.utc)
    stamp = moment.isoformat()
    version = engine_version()
    declared = tuple(sources)
    if not declared:
        raise HistoryImportError("no source ledgers declared; the import has nothing to read")

    rows_by_source: dict[str, tuple[LegacyRow, ...]] = {}
    for source in declared:
        rows_by_source[source.path] = read_source(root, source)

    # Which selection sets does each legacy run belong to? One record per
    # (run, set) pair, every record listing every set (§3.12).
    sets_by_run_key: dict[str, list[str]] = {}
    for source in declared:
        for row in rows_by_source[source.path]:
            if source.selection_set_id is None:
                sets_by_run_key.setdefault(row.run_key, [])
                continue
            entry = sets_by_run_key.setdefault(row.run_key, [])
            if source.selection_set_id not in entry:
                entry.append(source.selection_set_id)

    records: list[dict[str, Any]] = []
    map_entries: list[dict[str, Any]] = []
    rows_by_log: dict[str, list[dict[str, Any]]] = {}
    minted: dict[str, str] = {}
    unavailable_dirs = 0
    artifacts_present = 0
    artifacts_unavailable = 0

    for source in declared:
        source_rows = rows_by_source[source.path]
        observed: int | str = len(source_rows) if source.kind == KIND_IMPORTED else UNAVAILABLE
        for row in source_rows:
            record = build_record(
                row,
                source_root=root,
                selection_set_ids=sorted(sets_by_run_key.get(row.run_key, [])),
                n_trials_observed=observed,
            )
            run_id = record["ids"]["run_id"]
            collision = minted.get(run_id)
            if collision is not None:
                raise HistoryImportError(
                    f"run_id {run_id} minted twice, at {collision} and {source.path}:{row.line}; "
                    "two rows share a path, line and timestamp, so the identity rule cannot separate them"
                )
            minted[run_id] = f"{source.path}:{row.line}"
            records.append(record)
            map_entries.append(
                {
                    "source_path": source.path,
                    "line": row.line,
                    "legacy_ledger_config_hash": row.ledger_config_hash,
                    "legacy_packet_config_hash": record["legacy"]["legacy_packet_config_hash"],
                    "legacy_run_key": row.run_key,
                    "run_id": run_id,
                    "kind": source.kind,
                    "selection_set_id": source.selection_set_id,
                    "selection_set_ids": record["ids"]["selection_set_ids"],
                }
            )
            for entry in record["artifacts"]:
                if entry["artifact_status"] == "present":
                    artifacts_present += 1
                else:
                    artifacts_unavailable += 1
                    if entry["kind"] == "output_dir":
                        unavailable_dirs += 1
            rows_by_log.setdefault(source.log_name, []).append(record)

    # Trial logs: one genesis block per source, then that source's rows.
    trial_log_paths: list[str] = []
    for source in declared:
        log_path = workspace.trial_log(source.log_name)
        block = genesis_block(source, imported_at=stamp, version=version)
        lines = [json.dumps(block, sort_keys=True)]
        prev = block["hash"]
        for record in rows_by_log.get(source.log_name, []):
            if record["legacy"]["source_ledger"]["path"] != source.path:
                continue
            row_payload = trial_row(record, prev_hash=prev)
            prev = row_payload["hash"]
            lines.append(json.dumps(row_payload, sort_keys=True))
        _write_new(log_path, "\n".join(lines) + "\n")
        trial_log_paths.append(str(log_path))

    directory = workspace.results / HISTORY_IMPORT_DIRNAME
    records_path = directory / RECORDS_FILENAME
    _write_new(records_path, "".join(json.dumps(record, sort_keys=True) + "\n" for record in records))

    run_id_map = {
        "run_id_map_schema": RUN_ID_MAP_SCHEMA,
        "imported_at": stamp,
        "engine_version": version,
        "source_root": str(root),
        "key": ["source_path", "line", "legacy_ledger_config_hash"],
        "sources": [
            {
                "path": source.path,
                "sha256": source.sha256,
                "rows": len(rows_by_source[source.path]),
                "selection_set_id": source.selection_set_id,
                "kind": source.kind,
                "trial_log": str(workspace.trial_log(source.log_name)),
            }
            for source in declared
        ],
        "entries": map_entries,
    }
    run_id_map_path = directory / RUN_ID_MAP_FILENAME
    _write_new(run_id_map_path, json.dumps(run_id_map, indent=2, sort_keys=True) + "\n")

    registration_paths: list[str] = []
    for source in declared:
        if source.selection_set_id is None:
            continue
        path = directory / REGISTRATIONS_DIRNAME / f"{source.selection_set_id}.toml"
        _write_new(path, registration_toml(source, imported_at=moment.date()))
        registration_paths.append(str(path))

    shared = [
        {"legacy_run_key": key, "selection_set_ids": sorted(value)}
        for key, value in sorted(sets_by_run_key.items())
        if len(value) > 1
    ]
    report_path = directory / REPORT_FILENAME
    report = HistoryImportReport(
        imported_at=stamp,
        engine_version=version,
        source_root=str(root),
        workspace_root=str(workspace.root),
        sources=tuple(
            {
                "path": source.path,
                "sha256": source.sha256,
                "rows": len(rows_by_source[source.path]),
                "selection_set_id": source.selection_set_id,
                "kind": source.kind,
                "declared_budget": source.declared_budget,
                "rows_spent": source.rows_spent,
                "status": source.status,
                "legacy_independent_count": source.legacy_independent_count,
            }
            for source in declared
        ),
        records=len(records),
        run_id_map_path=str(run_id_map_path),
        records_path=str(records_path),
        report_path=str(report_path),
        registration_paths=tuple(registration_paths),
        trial_log_paths=tuple(trial_log_paths),
        shared_runs=tuple(shared),
        rows_with_unavailable_output_dir=unavailable_dirs,
        artifacts_present=artifacts_present,
        artifacts_unavailable=artifacts_unavailable,
    )
    _write_new(report_path, json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def verify_trial_log(path: Path) -> tuple[dict[str, Any], ...]:
    """Read a trial log and verify its hash chain, raising on any break.

    The chain detects change relative to a trusted copy only; it does not prove
    the search was exhaustive (§3.12). A removed row breaks it at the successor
    whose ``prev_hash`` no longer matches, which is what makes deleting a row
    from an append-only log detectable rather than deniable.
    """
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise HistoryImportError(f"{path}:{index}: blank line in an append-only log")
        rows.append(json.loads(line))
    if not rows:
        raise HistoryImportError(f"{path}: empty trial log; an imported log opens with a genesis block")
    if rows[0].get("event") != GENESIS_EVENT or rows[0].get("prev_hash") is not None:
        raise HistoryImportError(f"{path}:1: an imported trial log opens with a genesis block and no predecessor")
    previous: str | None = None
    for index, row in enumerate(rows, start=1):
        if "hash" not in row:
            raise HistoryImportError(f"{path}:{index}: row carries no hash")
        expected = chain_hash(row)
        if row["hash"] != expected:
            raise HistoryImportError(f"{path}:{index}: row hash {row['hash']} does not cover its content ({expected})")
        if row.get("prev_hash") != previous:
            raise HistoryImportError(
                f"{path}:{index}: prev_hash {row.get('prev_hash')} does not chain to the preceding row ({previous}); "
                "a row has been removed, reordered or rewritten"
            )
        previous = row["hash"]
    return tuple(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m prism.history_import [--workspace DIR] [--source-root DIR]``."""
    parser = argparse.ArgumentParser(
        prog="python -m prism.history_import",
        description="Import the four tracked legacy trial ledgers into a workspace (engine specification §3.12).",
    )
    parser.add_argument("--workspace", default=None, help="workspace root (default: the usual resolution order)")
    parser.add_argument("--source-root", default=None, help="tree holding the source ledgers (default: the workspace)")
    args = parser.parse_args(argv)
    workspace = resolve_workspace(args.workspace)
    report = import_history(workspace, source_root=args.source_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
