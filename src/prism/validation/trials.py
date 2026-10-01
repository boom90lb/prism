"""Canonical research-trial packets and claim-tier classification.

The project has several strategy surfaces that already write their own JSON/CSV
artifacts. This module gives them one shared, strict packet shape for the
methodology layer: what was run, which data convention it used, which artifacts
prove it, and what claim tier the observed result supports.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - the branch taken depends on the platform, not the test
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platform
    fcntl = None  # type: ignore[assignment]

from prism.validation.hurdle import PACKET_BLOCK_KEYS as HURDLE_PACKET_KEYS
from prism.validation.hurdle import HurdleResult
from prism.validation.metrics import periodic_sharpe

CLAIM_NO_CLAIM = "no_claim"
CLAIM_MECHANICS_CLEAN = "mechanics_clean"
CLAIM_GROSS_EDGE = "gross_edge"
CLAIM_NET_EDGE = "net_edge"
CLAIM_ROBUST_EDGE = "robust_edge"

CLAIM_TIERS = (
    CLAIM_NO_CLAIM,
    CLAIM_MECHANICS_CLEAN,
    CLAIM_GROSS_EDGE,
    CLAIM_NET_EDGE,
    CLAIM_ROBUST_EDGE,
)

TRIAL_PACKET_SCHEMA_VERSION = 2
DEFAULT_ROBUST_DSR_THRESHOLD = 0.95
DEFAULT_MIN_OBS = 20


@dataclass(frozen=True)
class GitCodeState:
    """Hashed git provenance for a research artifact."""

    available: bool
    commit: str | None
    dirty: bool | None
    tracked_diff_hash: str | None
    untracked_count: int | None
    untracked_paths_hash: str | None
    status_hash: str | None


@dataclass(frozen=True)
class ResearchClaimPacket:
    """Strict JSON-compatible summary of one research trial."""

    schema_version: int
    created_at_utc: str
    strategy: str
    claim_tier: str
    config_hash: str
    code_commit: str | None
    code_state: GitCodeState
    data: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    claim_rules: dict[str, Any]
    #: Optional since 2026-09-30: the after-cost hurdle the packet's net claim
    #: was read against, with its rate sources and the volatility basis it was
    #: stated at (:mod:`prism.validation.hurdle`). ``None`` in every packet
    #: written before the block existed, and in any packet whose surface
    #: states no hurdle; the schema version stays at 2 because readers of the
    #: old shape lose nothing.
    hurdle: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible packet dict."""
        return _to_jsonable(asdict(self))


def _to_jsonable(value: Any) -> Any:
    """Convert common scientific Python values into strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if dataclass_is_instance(value):
        return _to_jsonable(asdict(value))
    return str(value)


def dataclass_is_instance(value: Any) -> bool:
    """True for dataclass instances, false for dataclass classes."""
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def stable_json_dumps(payload: Mapping[str, Any]) -> str:
    """Canonical JSON used for research config identity."""
    return json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_config_hash(payload: Mapping[str, Any], length: int = 12) -> str:
    """Short SHA-256 digest for a canonicalized trial config payload."""
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()[:length]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_stdout(args: list[str], cwd: Path | str | None) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        capture_output=True,
    )
    return completed.stdout or b""


def current_git_state(cwd: Path | str | None = None) -> GitCodeState:
    """Return hashed git state for the current checkout, or unavailable state."""
    try:
        commit = _git_stdout(["rev-parse", "HEAD"], cwd).decode("utf-8", errors="replace").strip() or None
        tracked_diff = _git_stdout(["diff", "--binary", "--no-ext-diff", "HEAD", "--"], cwd)
        untracked_paths = _git_stdout(["ls-files", "--others", "--exclude-standard", "-z"], cwd)
        status = _git_stdout(["status", "--porcelain=v1", "-z"], cwd)
    except Exception:
        return GitCodeState(
            available=False,
            commit=None,
            dirty=None,
            tracked_diff_hash=None,
            untracked_count=None,
            untracked_paths_hash=None,
            status_hash=None,
        )

    untracked_count = sum(1 for item in untracked_paths.split(b"\0") if item)
    return GitCodeState(
        available=True,
        commit=commit,
        dirty=bool(status),
        tracked_diff_hash=_sha256_bytes(tracked_diff),
        untracked_count=untracked_count,
        untracked_paths_hash=_sha256_bytes(untracked_paths),
        status_hash=_sha256_bytes(status),
    )


def current_git_commit(cwd: Path | str | None = None) -> str | None:
    """Return the current git commit, or None when unavailable."""
    return current_git_state(cwd).commit


def _coerce_git_code_state(value: GitCodeState | Mapping[str, Any] | None) -> GitCodeState:
    if value is None:
        return current_git_state()
    if isinstance(value, GitCodeState):
        return value
    return GitCodeState(
        available=bool(value.get("available", False)),
        commit=value.get("commit"),
        dirty=value.get("dirty"),
        tracked_diff_hash=value.get("tracked_diff_hash"),
        untracked_count=value.get("untracked_count"),
        untracked_paths_hash=value.get("untracked_paths_hash"),
        status_hash=value.get("status_hash"),
    )


def _finite_returns(returns: pd.Series | np.ndarray | list[float]) -> pd.Series:
    series = pd.Series(returns, dtype=float)
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _total_return(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    value = float((1.0 + returns).prod() - 1.0)
    return value if math.isfinite(value) else None


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_metric(summary: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in summary:
            value = _optional_float(summary[key])
            if value is not None:
                return value
    return None


def _gross_returns(net_returns: pd.Series, costs: pd.DataFrame | None) -> pd.Series | None:
    if costs is None or "total" not in costs.columns:
        return None
    total_cost = pd.to_numeric(costs["total"], errors="coerce")
    aligned_cost = total_cost.reindex(net_returns.index).fillna(0.0)
    return net_returns.add(aligned_cost, fill_value=0.0)


def _cost_column_mean(costs: pd.DataFrame | None, column: str) -> float | None:
    if costs is None or column not in costs.columns:
        return None
    return _optional_float(pd.to_numeric(costs[column], errors="coerce").mean())


def _cost_column_sum(costs: pd.DataFrame | None, column: str) -> float | None:
    if costs is None or column not in costs.columns:
        return None
    return _optional_float(pd.to_numeric(costs[column], errors="coerce").sum())


def _target_weight_metrics(target_weights: pd.DataFrame | None) -> dict[str, Any]:
    if target_weights is None or target_weights.empty:
        return {}
    weights = target_weights.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = weights.abs().sum(axis=1)
    return {
        "n_target_rows": int(len(weights)),
        "n_target_symbols": int(len(weights.columns)),
        "avg_target_gross": _optional_float(gross.mean()),
        "max_target_gross": _optional_float(gross.max()),
    }


def compute_trial_metrics(
    returns: pd.Series | np.ndarray | list[float],
    costs: pd.DataFrame | None = None,
    target_weights: pd.DataFrame | None = None,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute canonical gross/net/cost/trial metrics for one run."""
    summary = summary or {}
    net = _finite_returns(returns)
    net_periodic = periodic_sharpe(net)
    gross = _gross_returns(net, costs)
    gross_periodic = periodic_sharpe(gross) if gross is not None else float("nan")
    total_cost = _cost_column_sum(costs, "total")
    avg_turnover = _cost_column_mean(costs, "turnover")
    avg_gross = _cost_column_mean(costs, "gross")
    avg_net = _cost_column_mean(costs, "net")

    metrics: dict[str, Any] = {
        "n_obs": int(len(net)),
        "total_return": _total_return(net),
        "periodic_sharpe": _optional_float(net_periodic),
        "annualized_sharpe": _optional_float(net_periodic * math.sqrt(252.0)),
        "gross_total_return": _total_return(gross) if gross is not None else None,
        "gross_periodic_sharpe": _optional_float(gross_periodic),
        "gross_annualized_sharpe": _optional_float(gross_periodic * math.sqrt(252.0)),
        "total_cost": total_cost if total_cost is not None else _first_metric(summary, ("total_cost",)),
        "avg_turnover": (
            avg_turnover if avg_turnover is not None else _first_metric(summary, ("avg_turnover", "turnover"))
        ),
        "avg_gross": avg_gross if avg_gross is not None else _first_metric(summary, ("avg_gross",)),
        "avg_net": avg_net if avg_net is not None else _first_metric(summary, ("avg_net",)),
        "dsr": _first_metric(summary, ("dsr", "pair_set_dsr", "residual_set_dsr")),
        "trial_count": _first_metric(
            summary,
            ("trial_count", "n_trials", "ledger_trials", "pair_trial_count", "finite_pair_trial_sharpes"),
        ),
    }
    metrics.update(_target_weight_metrics(target_weights))
    return _to_jsonable(metrics)


def classify_claim_tier(
    metrics: Mapping[str, Any],
    *,
    robust_dsr_threshold: float = DEFAULT_ROBUST_DSR_THRESHOLD,
    min_obs: int = DEFAULT_MIN_OBS,
) -> str:
    """Classify the strongest claim supported by canonical trial metrics."""
    if min_obs < 1:
        raise ValueError(f"min_obs must be >= 1, got {min_obs}")
    if not 0.0 <= robust_dsr_threshold <= 1.0:
        raise ValueError(f"robust_dsr_threshold must be in [0, 1], got {robust_dsr_threshold}")

    n_obs = int(metrics.get("n_obs") or 0)
    if n_obs < min_obs:
        return CLAIM_NO_CLAIM

    net_return = _optional_float(metrics.get("total_return"))
    net_sharpe = _optional_float(metrics.get("periodic_sharpe"))
    gross_return = _optional_float(metrics.get("gross_total_return"))
    gross_sharpe = _optional_float(metrics.get("gross_periodic_sharpe"))
    dsr = _optional_float(metrics.get("dsr"))

    tier = CLAIM_MECHANICS_CLEAN
    if gross_return is not None and gross_sharpe is not None and gross_return > 0 and gross_sharpe > 0:
        tier = CLAIM_GROSS_EDGE
    if net_return is not None and net_sharpe is not None and net_return > 0 and net_sharpe > 0:
        tier = CLAIM_NET_EDGE
    if tier == CLAIM_NET_EDGE and dsr is not None and dsr >= robust_dsr_threshold:
        tier = CLAIM_ROBUST_EDGE
    return tier


def build_research_claim_packet(
    *,
    strategy: str,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    returns: pd.Series | np.ndarray | list[float],
    costs: pd.DataFrame | None = None,
    target_weights: pd.DataFrame | None = None,
    summary: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    code_commit: str | None = None,
    code_state: GitCodeState | Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
    robust_dsr_threshold: float = DEFAULT_ROBUST_DSR_THRESHOLD,
    min_obs: int = DEFAULT_MIN_OBS,
    hurdle: HurdleResult | Mapping[str, Any] | None = None,
) -> ResearchClaimPacket:
    """Build one canonical packet for a strategy trial.

    ``hurdle`` is optional: a :class:`~prism.validation.hurdle.HurdleResult`
    or a mapping with every key of ``HURDLE_PACKET_KEYS``. A surface that read
    its net claim against a stated hurdle records it here; one that did not
    leaves it ``None``.
    """
    if not strategy:
        raise ValueError("strategy must be non-empty")
    hurdle_block = _coerce_hurdle_block(hurdle)
    metrics = compute_trial_metrics(
        returns=returns,
        costs=costs,
        target_weights=target_weights,
        summary=summary,
    )
    tier = classify_claim_tier(
        metrics,
        robust_dsr_threshold=robust_dsr_threshold,
        min_obs=min_obs,
    )
    config_payload = {"strategy": strategy, "config": config, "data": data}
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    resolved_code_state = _coerce_git_code_state(code_state)
    if code_commit is None:
        code_commit = resolved_code_state.commit
    elif resolved_code_state.commit is None:
        resolved_code_state = replace(resolved_code_state, commit=code_commit)
    return ResearchClaimPacket(
        schema_version=TRIAL_PACKET_SCHEMA_VERSION,
        created_at_utc=timestamp.astimezone(timezone.utc).isoformat(),
        strategy=strategy,
        claim_tier=tier,
        config_hash=stable_config_hash(config_payload),
        code_commit=code_commit,
        code_state=resolved_code_state,
        data=_to_jsonable(data),
        config=_to_jsonable(config),
        metrics=metrics,
        artifacts={str(k): str(v) for k, v in sorted((artifacts or {}).items())},
        claim_rules={
            "claim_tiers": list(CLAIM_TIERS),
            "min_obs": int(min_obs),
            "robust_dsr_threshold": float(robust_dsr_threshold),
            "gross_edge": "gross_total_return > 0 and gross_periodic_sharpe > 0",
            "net_edge": "total_return > 0 and periodic_sharpe > 0",
            "robust_edge": "net_edge and dsr >= robust_dsr_threshold",
        },
        hurdle=hurdle_block,
    )


def _coerce_hurdle_block(value: HurdleResult | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The packet's ``hurdle`` block from a result or a mapping, or None when absent.

    A mapping must carry every key of ``HURDLE_PACKET_KEYS``; a block with a
    key missing is refused rather than written short, because a hurdle whose
    volatility basis or rate source is unrecorded is the ambiguity the block
    exists to remove.
    """
    if value is None:
        return None
    if isinstance(value, HurdleResult):
        return value.packet_block()
    if not isinstance(value, Mapping):
        raise ValueError(f"hurdle must be a HurdleResult or a mapping, got {type(value).__name__}")
    missing = [key for key in HURDLE_PACKET_KEYS if key not in value]
    if missing:
        raise ValueError(f"hurdle block missing keys {missing}; required keys are {list(HURDLE_PACKET_KEYS)}")
    return {str(k): v for k, v in value.items()}


def write_research_claim_packet(packet: ResearchClaimPacket, path: Path | str) -> None:
    """Write a strict JSON claim packet."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")


def emit_research_claim_packet(
    out_dir: Path | str,
    *,
    filename: str = "claim_packet.json",
    **packet_kwargs: Any,
) -> ResearchClaimPacket:
    """Build a packet and write it as ``out_dir/filename``; return the packet.

    Thin wrapper over :func:`build_research_claim_packet` +
    :func:`write_research_claim_packet` so every strategy surface emits packets
    through one identical path.
    """
    packet = build_research_claim_packet(**packet_kwargs)
    write_research_claim_packet(packet, Path(out_dir) / filename)
    return packet


def summary_claim_fields(
    packet: ResearchClaimPacket,
    *,
    packet_filename: str = "claim_packet.json",
) -> dict[str, Any]:
    """Canonical claim-packet fields to fold into a strategy's summary JSON."""
    return {
        "claim_tier": packet.claim_tier,
        "config_hash": packet.config_hash,
        "claim_packet_schema_version": packet.schema_version,
        "claim_packet": packet_filename,
    }


REQUIRED_PACKET_KEYS: tuple[str, ...] = (
    "schema_version",
    "created_at_utc",
    "strategy",
    "claim_tier",
    "config_hash",
    "code_commit",
    "code_state",
    "data",
    "config",
    "metrics",
    "artifacts",
    "claim_rules",
)


def _reject_non_finite_json(token: str) -> Any:
    """``json.loads`` ``parse_constant`` hook: a strict packet has no NaN/Inf."""
    raise ValueError(f"non-finite JSON token {token!r} is not allowed in a strict claim packet")


def validate_claim_packet_dir(
    out_dir: Path | str,
    *,
    packet_filename: str = "claim_packet.json",
    require_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate a strategy output dir against the research-claim contract.

    Asserts the dir holds a strict-JSON packet (no NaN/Inf), with every required
    key, a known ``claim_tier``, the current schema version, a complete
    ``hurdle`` block when one is present (the block is optional; packets
    written before it existed carry none), and — when ``require_artifacts`` —
    that every file named in the packet's ``artifacts`` manifest actually
    exists in ``out_dir``. Returns the parsed packet dict.
    Raises ``FileNotFoundError``/``ValueError`` on the first violation.
    """
    out = Path(out_dir)
    packet_path = out / packet_filename
    if not packet_path.is_file():
        raise FileNotFoundError(f"claim packet not found: {packet_path}")

    packet = json.loads(packet_path.read_text(), parse_constant=_reject_non_finite_json)
    if not isinstance(packet, dict):
        raise ValueError(f"claim packet is not a JSON object: {packet_path}")

    missing_keys = [key for key in REQUIRED_PACKET_KEYS if key not in packet]
    if missing_keys:
        raise ValueError(f"claim packet missing required keys {missing_keys}: {packet_path}")

    if packet["schema_version"] != TRIAL_PACKET_SCHEMA_VERSION:
        raise ValueError(
            f"claim packet schema_version {packet['schema_version']} != "
            f"{TRIAL_PACKET_SCHEMA_VERSION}: {packet_path}"
        )
    if packet["claim_tier"] not in CLAIM_TIERS:
        raise ValueError(f"claim packet has unknown claim_tier {packet['claim_tier']!r}: {packet_path}")

    code_state = packet["code_state"]
    if not isinstance(code_state, Mapping):
        raise ValueError(f"claim packet code_state is not an object: {packet_path}")
    missing_code_state = [key for key in GitCodeState.__dataclass_fields__ if key not in code_state]
    if missing_code_state:
        raise ValueError(f"claim packet code_state missing keys {missing_code_state}: {packet_path}")

    hurdle = packet.get("hurdle")
    if hurdle is not None:
        if not isinstance(hurdle, Mapping):
            raise ValueError(f"claim packet hurdle is not an object: {packet_path}")
        missing_hurdle = [key for key in HURDLE_PACKET_KEYS if key not in hurdle]
        if missing_hurdle:
            raise ValueError(f"claim packet hurdle block missing keys {missing_hurdle}: {packet_path}")

    artifacts = packet["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"claim packet artifacts is not an object: {packet_path}")
    if require_artifacts:
        missing = sorted(rel for rel in artifacts.values() if rel != packet_filename and not (out / rel).is_file())
        if missing:
            raise ValueError(f"claim packet references missing artifacts {missing} under {out}")
    return packet


# ==========================================================================
# The result record and the run-recording protocol (engine specification §3.12)
# ==========================================================================
#
# ``TRIAL_PACKET_SCHEMA_VERSION`` above stays at 2 and its reader keeps every
# cited claim packet readable. What follows is the *distinct* record type new
# evidence is written as, plus the append-only trial log that records the
# attempt whether or not it finished. The two coexist; no packet is rewritten.

#: ``ResultRecord.record_schema``. A different number from the claim packet's,
#: because it is a different object with a different reader.
RECORD_SCHEMA = 1

#: The literal a field with no source takes. Never a null, never a plausible
#: default: a null reads as "not applicable" and a default reads as a setting.
UNAVAILABLE = "unavailable"

RUN_KINDS = ("new", "retry", "replay", "reproduction", "exploratory", "imported", "provenance")

#: Kinds excluded from a selection set's *independent* evidence count. They
#: still count as attempts (§3.12, "Attempt counting").
UNCOUNTED_KINDS = ("retry", "replay", "reproduction", "imported")

RUN_CHANNELS = ("evidence", "engineering")
FILL_SOURCES = ("simulated", "paper", "live")
RECORD_STATUSES = ("finished", "incomplete")
ACCOUNTING_ENGINES = ("historical_constant_weight", "canonical")

#: The reproduction engine of §3.8: ``backtest_target_weights`` unchanged.
REPRODUCTION_ENGINE = "historical_constant_weight"
#: The fixed-share simulator of §3.6, on ``PortfolioState``.
CANONICAL_ENGINE = "canonical"

#: What each engine holds between decision bars, recorded as
#: ``evaluation.holding_process``. This is the whole difference between the two,
#: and it is written into the record as a value rather than left implicit in the
#: engine's name, so a reader who does not know the engine's history can still
#: tell what the number describes.
ENGINE_HOLDING_PROCESS: Mapping[str, str] = {
    REPRODUCTION_ENGINE: "constant_weight",
    CANONICAL_ENGINE: "fixed_share",
}

#: The marker every ``historical_constant_weight`` record carries in its
#: warnings, and no ``canonical`` record may carry. §3.8 requires the
#: reproduction engine to be "labeled in every record as a reproduction mode
#: with the implicit-rebalancing caveat"; a label nothing checks is a label that
#: goes missing, so :func:`record_schema_problems` checks it both ways.
ENGINE_CAVEAT_MARKER = "accounting_engine=historical_constant_weight:"

#: ``conventions.executable_return`` per engine. The reproduction engine's
#: return is open-to-open on a weight book; the canonical engine's is the change
#: in marked equity across two closes, with the fill at the open between them.
#: They are different measurements of different processes and never interchanged.
ENGINE_EXECUTABLE_RETURN: Mapping[str, str] = {
    REPRODUCTION_ENGINE: "next_open_to_open",
    CANONICAL_ENGINE: "next_open_fill_marked_close_to_close",
}

FOLD_PORTFOLIOS = ("reset", "carry")
BIAS_STATUSES = ("unquantified", "estimated")
ARTIFACT_STATUSES = ("present", "unavailable")

#: The tiers a synthetic or engineering-channel record is capped at, whatever
#: its metrics: the classifier is provenance-blind, so the cap belongs to the
#: record (§3.12, "Evidence tiers").
CAPPED_EVIDENCE_TIER = CLAIM_MECHANICS_CLEAN

#: The two tiers an ``incomplete`` record cannot support (§3.12).
TOP_EVIDENCE_TIERS = (CLAIM_NET_EDGE, CLAIM_ROBUST_EDGE)

#: Every field of every section, in the order §3.12 lists them. Presence is
#: required; a field with no source carries :data:`UNAVAILABLE`.
RECORD_SECTIONS: Mapping[str, tuple[str, ...]] = {
    "ids": (
        "run_id",
        "config_id",
        "data_id",
        "kind",
        "arm_id",
        "registration_id",
        "registration_hash",
        "selection_set_ids",
        "reproduces",
        "supersedes",
    ),
    "status": ("status", "incomplete_reasons"),
    "versions": ("engine", "solver", "numpy", "pandas", "python"),
    "conventions": (
        "returns",
        "scoring_basis",
        "price_basis",
        "executable_return",
        "excess_return_definition",
        "hurdle_series_id",
        "hurdle_conversion",
        "cash_and_financing",
        "execution_profile",
    ),
    "data": ("visibility_mode", "vintage", "manifest_id", "lag_assumptions", "score_panel"),
    "coverage": (
        "membership_coverage",
        "missing_securities",
        "terminal_outcomes",
        "provisional_sid_count",
        "quarantined_sid_count",
        "untested_conclusions",
        "bias_status",
    ),
    "costs": ("stack", "calibration_source", "cost_refresh", "dollar_volume_statistics"),
    "evaluation": (
        "fold_geometry",
        "excluded_folds",
        "fold_portfolio",
        "train_only_transforms",
        "calibration",
        "risk_model",
        "accounting_engine",
        "capital",
        "components",
    ),
    "performance": (
        "periodic_returns_path",
        "gross",
        "net",
        "sharpe",
        "drawdown",
        "tails",
        "turnover",
        "exposures",
        "violations",
        "repair",
        "binding_conditions",
    ),
    "selection": (
        "n_trials_observed",
        "attempt_basis",
        "dependence_adjustment",
        "unobserved_search_note",
        "legacy_independent_count",
    ),
    "flags": ("synthetic", "exploratory", "prospective", "fill_source", "run_channel", "evidence_tier"),
    "legacy": (
        "legacy_ledger_config_hash",
        "legacy_packet_config_hash",
        "legacy_code_commit",
        "legacy_run_key",
        "source_ledger",
    ),
}

#: The keys of the repair block, which is a recorded *count* per outcome and
#: not a narrative (§3.8, "Capital scenarios").
REPAIR_OUTCOMES = ("attempted", "succeeded", "scaled_batch", "dropped_batch", "no_action")


class RecordSchemaError(ValueError):
    """A payload does not satisfy ``record_schema = 1``.

    Raised with *every* problem, not the first: a record is written once and
    inspected later, so a validator that stops at the first missing field costs
    a round trip per field.
    """


def empty_repair_counts() -> dict[str, int]:
    """The five repair outcome counts, all zero."""
    return {name: 0 for name in REPAIR_OUTCOMES}


def cap_evidence_tier(tier: str, *, synthetic: bool, run_channel: str) -> str:
    """Cap ``tier`` at "pipeline runs" for a synthetic or engineering record.

    ``classify_claim_tier`` is provenance-blind by construction; this is the
    exclusion the *record* applies afterwards (§3.12).
    """
    if tier not in CLAIM_TIERS:
        raise RecordSchemaError(f"unknown evidence tier {tier!r}; the tiers are {CLAIM_TIERS}")
    if not (bool(synthetic) or str(run_channel) == "engineering"):
        return tier
    if CLAIM_TIERS.index(tier) <= CLAIM_TIERS.index(CAPPED_EVIDENCE_TIER):
        return tier
    return CAPPED_EVIDENCE_TIER


def _bias_status_value(value: Any) -> str | None:
    """The bias status out of either accepted shape, or None when malformed."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        status = value.get("status")
        return status if isinstance(status, str) else None
    return None


def record_schema_problems(payload: Any) -> list[str]:
    """Every way ``payload`` fails ``record_schema = 1``, as readable lines."""
    if not isinstance(payload, Mapping):
        return [f"record is not a mapping but a {type(payload).__name__}"]
    problems: list[str] = []
    if payload.get("record_schema") != RECORD_SCHEMA:
        problems.append(f"record_schema: expected {RECORD_SCHEMA}, got {payload.get('record_schema')!r}")

    for section, fields in RECORD_SECTIONS.items():
        block = payload.get(section)
        if not isinstance(block, Mapping):
            problems.append(f"{section}: missing section (or not a mapping)")
            continue
        for name in fields:
            if name not in block:
                problems.append(f"{section}.{name}: missing")

    ids = payload.get("ids") if isinstance(payload.get("ids"), Mapping) else {}
    if not str(ids.get("run_id") or "").strip():
        problems.append("ids.run_id: a record needs a non-empty run_id")
    if "config_id" in ids and not (ids["config_id"] is None or isinstance(ids["config_id"], str)):
        problems.append(f"ids.config_id: must be a string or null, got {ids['config_id']!r}")
    if "kind" in ids and ids["kind"] not in RUN_KINDS:
        problems.append(f"ids.kind: {ids['kind']!r} is not one of {RUN_KINDS}")
    for name in ("selection_set_ids", "reproduces", "supersedes"):
        if name in ids and not isinstance(ids[name], (list, tuple)):
            problems.append(f"ids.{name}: must be a list, got {type(ids[name]).__name__}")
    for entry in ids.get("reproduces", ()) or ():
        if not (isinstance(entry, Mapping) and "run_id" in entry and "reason" in entry):
            problems.append(f"ids.reproduces: each entry needs run_id and reason, got {entry!r}")

    status_block = payload.get("status") if isinstance(payload.get("status"), Mapping) else {}
    status = status_block.get("status")
    if "status" in status_block and status not in RECORD_STATUSES:
        problems.append(f"status.status: {status!r} is not one of {RECORD_STATUSES}")
    reasons = status_block.get("incomplete_reasons")
    if "incomplete_reasons" in status_block and not isinstance(reasons, (list, tuple)):
        problems.append(f"status.incomplete_reasons: must be a list, got {type(reasons).__name__}")
    elif status == "incomplete" and not reasons:
        problems.append("status.incomplete_reasons: an incomplete record must say why")

    flags = payload.get("flags") if isinstance(payload.get("flags"), Mapping) else {}
    if "run_channel" in flags and flags["run_channel"] not in RUN_CHANNELS:
        problems.append(f"flags.run_channel: {flags['run_channel']!r} is not one of {RUN_CHANNELS}")
    if "fill_source" in flags and flags["fill_source"] not in FILL_SOURCES:
        problems.append(f"flags.fill_source: {flags['fill_source']!r} is not one of {FILL_SOURCES}")
    for name in ("synthetic", "exploratory"):
        if name in flags and not isinstance(flags[name], bool):
            problems.append(f"flags.{name}: must be a bool, got {flags[name]!r}")
    prospective = flags.get("prospective")
    if prospective is not None and not (isinstance(prospective, Mapping) and "measured_from" in prospective):
        problems.append(f"flags.prospective: must be null or carry measured_from, got {prospective!r}")
    tier = flags.get("evidence_tier")
    if "evidence_tier" in flags and tier != UNAVAILABLE:
        if tier not in CLAIM_TIERS:
            problems.append(f"flags.evidence_tier: {tier!r} is not one of {CLAIM_TIERS} (or {UNAVAILABLE!r})")
        else:
            capped = cap_evidence_tier(
                tier, synthetic=bool(flags.get("synthetic")), run_channel=flags.get("run_channel", "evidence")
            )
            if capped != tier:
                problems.append(
                    f"flags.evidence_tier: {tier!r} on a synthetic or engineering-channel record; the cap is "
                    f"{capped!r} (§3.12)"
                )
            if status == "incomplete" and tier in TOP_EVIDENCE_TIERS:
                problems.append(f"flags.evidence_tier: an incomplete record cannot support {tier!r} (§3.12)")

    evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), Mapping) else {}
    engine = evaluation.get("accounting_engine")
    if "accounting_engine" in evaluation and engine != UNAVAILABLE and engine not in ACCOUNTING_ENGINES:
        problems.append(f"evaluation.accounting_engine: {engine!r} is not one of {ACCOUNTING_ENGINES}")
    problems.extend(_engine_label_problems(payload, evaluation))
    portfolio = evaluation.get("fold_portfolio")
    if "fold_portfolio" in evaluation and portfolio != UNAVAILABLE and portfolio not in FOLD_PORTFOLIOS:
        problems.append(f"evaluation.fold_portfolio: {portfolio!r} is not one of {FOLD_PORTFOLIOS}")
    capital = evaluation.get("capital")
    if "capital" in evaluation and capital is not None and not isinstance(capital, (int, float, str)):
        problems.append(f"evaluation.capital: must be a number, null or {UNAVAILABLE!r}, got {capital!r}")

    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), Mapping) else {}
    if "bias_status" in coverage:
        bias = _bias_status_value(coverage["bias_status"])
        if bias not in BIAS_STATUSES:
            problems.append(f"coverage.bias_status: {coverage['bias_status']!r} is not one of {BIAS_STATUSES}")
        elif bias == "estimated":
            method = coverage["bias_status"].get("method") if isinstance(coverage["bias_status"], Mapping) else None
            if not method:
                problems.append("coverage.bias_status: an estimated bias must name its method (§3.12)")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, (list, tuple)):
        problems.append("artifacts: missing (a list, possibly empty)")
    else:
        for position, entry in enumerate(artifacts):
            if not isinstance(entry, Mapping):
                problems.append(f"artifacts[{position}]: not a mapping")
                continue
            for name in ("path", "sha256", "artifact_status"):
                if name not in entry:
                    problems.append(f"artifacts[{position}].{name}: missing")
            if entry.get("artifact_status") not in ARTIFACT_STATUSES:
                problems.append(
                    f"artifacts[{position}].artifact_status: {entry.get('artifact_status')!r} is not one of "
                    f"{ARTIFACT_STATUSES}"
                )
    if not isinstance(payload.get("warnings"), (list, tuple)):
        problems.append("warnings: missing (a list, possibly empty)")

    try:
        stable_json_dumps(payload)
    except ValueError as exc:
        # ``allow_nan=False``: a NaN in a record is a number that cannot be
        # read back, which is worse than a missing field.
        problems.append(f"record does not canonicalize: {exc}")
    return problems


def _engine_label_problems(payload: Mapping[str, Any], evaluation: Mapping[str, Any]) -> list[str]:
    """Every way a record's engine label disagrees with the record's own contents (§3.8).

    ``accounting_engine`` is a claim about which holding process produced every
    number in the record, and the two processes differ by thousands of basis
    points on the same targets. So the label is cross-checked rather than
    trusted: the record must also declare the holding process the engine has,
    state the executable return that engine measures, and carry the
    implicit-rebalancing caveat if and only if it is a reproduction-mode record.
    A record whose label says one engine and whose contents say the other is
    refused here, at the point it is written.

    An imported record (``accounting_engine = "unavailable"``) is exempt: the
    legacy driver is neither engine and §3.12 forbids naming one for it.
    """
    engine = evaluation.get("accounting_engine")
    if "accounting_engine" not in evaluation or engine == UNAVAILABLE or engine not in ACCOUNTING_ENGINES:
        return []
    problems: list[str] = []
    expected_process = ENGINE_HOLDING_PROCESS[str(engine)]
    process = evaluation.get("holding_process")
    if process is None:
        problems.append(
            f"evaluation.holding_process: missing on a record naming accounting_engine {engine!r}; the engine's "
            f"holding process is what the record's numbers describe and is recorded as {expected_process!r} (§3.8)"
        )
    elif process != expected_process:
        problems.append(
            f"evaluation.holding_process: {process!r} on a record naming accounting_engine {engine!r}, whose "
            f"holding process is {expected_process!r}; the two engines are never silently substituted (§3.8)"
        )

    conventions = payload.get("conventions") if isinstance(payload.get("conventions"), Mapping) else {}
    executable = conventions.get("executable_return")
    expected_return = ENGINE_EXECUTABLE_RETURN[str(engine)]
    if executable is not None and executable != UNAVAILABLE and executable != expected_return:
        problems.append(
            f"conventions.executable_return: {executable!r} on a record naming accounting_engine {engine!r}, "
            f"which measures {expected_return!r} (§3.8)"
        )

    warnings = payload.get("warnings")
    carried = (
        any(ENGINE_CAVEAT_MARKER in str(entry) for entry in warnings) if isinstance(warnings, (list, tuple)) else False
    )
    if engine == REPRODUCTION_ENGINE and not carried:
        problems.append(
            "warnings: a historical_constant_weight record carries the implicit-rebalancing caveat in every "
            f"record it writes (§3.8); no warning contains {ENGINE_CAVEAT_MARKER!r}"
        )
    if engine == CANONICAL_ENGINE and carried:
        problems.append(
            "warnings: a canonical record carries the reproduction engine's implicit-rebalancing caveat; the "
            "canonical engine holds shares and rebalances nothing between decisions (§3.8)"
        )
    return problems


def validate_result_record(payload: Any) -> None:
    """Raise :class:`RecordSchemaError` naming every problem, or return."""
    problems = record_schema_problems(payload)
    if problems:
        raise RecordSchemaError(
            f"record_schema {RECORD_SCHEMA} violated by {len(problems)} problem(s):\n  " + "\n  ".join(problems)
        )


@dataclass(frozen=True)
class ResultRecord:
    """One evaluation's evidence (§3.12), distinct from the claim packet.

    Every section of §3.12 is a required field and every field of every section
    is required to be *present*; a field with no source carries
    :data:`UNAVAILABLE` rather than a null or a plausible default, so a reader
    can never mistake "not recorded" for "configured that way".

    ``extra`` carries top-level keys outside the fourteen sections (the history
    import's ``artifact_root``, for one) so a record round-trips through
    :meth:`from_dict` and :meth:`to_dict` unchanged.
    """

    ids: Mapping[str, Any]
    status: Mapping[str, Any]
    versions: Mapping[str, Any]
    conventions: Mapping[str, Any]
    data: Mapping[str, Any]
    coverage: Mapping[str, Any]
    costs: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    performance: Mapping[str, Any]
    selection: Mapping[str, Any]
    flags: Mapping[str, Any]
    legacy: Mapping[str, Any]
    artifacts: Sequence[Mapping[str, Any]] = ()
    warnings: Sequence[str] = ()
    record_schema: int = RECORD_SCHEMA
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The JSON-ready payload, sections in §3.12 order."""
        payload: dict[str, Any] = {"record_schema": int(self.record_schema)}
        for section in RECORD_SECTIONS:
            payload[section] = dict(getattr(self, section))
        payload["artifacts"] = [dict(entry) for entry in self.artifacts]
        payload["warnings"] = list(self.warnings)
        for key, value in dict(self.extra).items():
            payload.setdefault(key, value)
        return payload

    def validate(self) -> "ResultRecord":
        """Validate and return self, so a builder can end with ``.validate()``."""
        validate_result_record(self.to_dict())
        return self

    @property
    def run_id(self) -> str:
        return str(self.ids["run_id"])

    @property
    def config_id(self) -> str | None:
        value = self.ids.get("config_id")
        return None if value is None else str(value)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResultRecord":
        """Build a record from a payload, validating it first."""
        validate_result_record(payload)
        known = set(RECORD_SECTIONS) | {"record_schema", "artifacts", "warnings"}
        return cls(
            **{section: dict(payload[section]) for section in RECORD_SECTIONS},
            artifacts=[dict(entry) for entry in payload.get("artifacts", ())],
            warnings=list(payload.get("warnings", ())),
            record_schema=int(payload["record_schema"]),
            extra={key: value for key, value in payload.items() if key not in known},
        )

    @classmethod
    def read(cls, path: Path | str) -> "ResultRecord":
        """Read a record written by :meth:`write`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write(self, path: Path | str) -> Path:
        """Write the validated record as indented JSON and return the path."""
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_to_jsonable(self.to_dict()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


# --------------------------------------------------------------------------
# The trial log: one row schema, appended under an exclusive lock
# --------------------------------------------------------------------------

STARTED = "started"
FINISHED = "finished"
FAILED = "failed"
INCOMPLETE = "incomplete"

TRIAL_EVENTS = (STARTED, FINISHED, FAILED, INCOMPLETE)
TERMINAL_EVENTS = (FINISHED, FAILED, INCOMPLETE)

#: One row schema serves every event (§3.12, "Run recording protocol").
TRIAL_ROW_FIELDS = (
    "ts",
    "event",
    "run_id",
    "config_id",
    "data_id",
    "kind",
    "arm_id",
    "selection_set_id",
    "capital",
    "synthetic",
    "run_channel",
    "summary",
    "reason",
    "pid",
    "host",
    "heartbeat_at",
    "prev_hash",
    "hash",
)

#: The event a genesis block carries. Written by ``prism.history_import`` when
#: a source ledger is imported; this module reads it and never writes one.
GENESIS_EVENT = "import_genesis"

#: The log an unregistered evaluation appends to. It belongs to no selection
#: set and cannot be cited for promotion (§3.12, "Log routing").
EXPLORATORY_LOG = "exploratory"

#: The log a smoke run, a synthetic fixture run, or a synthetic broker session
#: appends to. In no selection set, in no attempt count.
ENGINEERING_LOG = "engineering"

#: How old a heartbeat must be before ``reconcile`` may consider a run dead.
DEFAULT_MAX_HEARTBEAT_AGE_SECONDS = 900.0

LOCK_SUFFIX = ".lock"


class TrialLogError(RuntimeError):
    """The trial log cannot be appended to, or its chain does not verify."""


def chain_hash(row: Mapping[str, Any]) -> str:
    """``sha256`` over a trial-log row's canonical JSON, excluding ``hash``.

    One implementation, used by every writer of the log: the history import's
    genesis block, its imported rows, and the evaluator's protocol rows. A
    second implementation would fork the format silently.
    """
    payload = {key: value for key, value in row.items() if key != "hash"}
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def utc_now() -> str:
    """The timestamp format every protocol row carries."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def process_is_alive(pid: Any) -> bool:
    """True when a process with ``pid`` exists on *this* host.

    ``EPERM`` counts as alive: the process exists and belongs to someone else.
    This cannot defeat pid reuse — a recycled pid reads as the original process
    — which is why the reconciler also requires a stale heartbeat before it
    calls a run dead. The start token that defeats pid reuse arrives with the
    account lock (§3.10, stage 5).
    """
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrialLogError(f"cannot check liveness of pid {pid!r}: {exc}") from exc
    return True


@contextlib.contextmanager
def append_lock(path: Path | str, *, timeout_seconds: float = 30.0, poll_seconds: float = 0.01) -> Iterator[None]:
    """Hold the exclusive append lock for a trial log (§3.12, "Concurrency").

    ``flock`` where ``fcntl`` exists, the ``O_EXCL`` protocol where it does not
    (§3.10). Unlike the execution lock this one *waits*: two evaluations
    appending to the same log are not a conflict to refuse, they are two rows
    to serialize. It waits with a deadline rather than forever, and raises when
    it expires, because a lock nobody releases must surface as an error rather
    than as a hung run.
    """
    lock_path = Path(str(path) + LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + float(timeout_seconds)
    if fcntl is not None:
        handle = open(lock_path, "a+", encoding="utf-8")
        try:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise TrialLogError(
                            f"waited {timeout_seconds}s for the append lock {lock_path} and it is still held; "
                            "another appender is stuck or the lock was abandoned"
                        ) from exc
                    time.sleep(poll_seconds)
            yield
        finally:
            handle.close()  # closing the fd releases the flock
        return
    while True:  # pragma: no cover - non-POSIX platform
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise TrialLogError(
                    f"waited {timeout_seconds}s for the append lock {lock_path} and it still exists; remove it "
                    "only after confirming no appender is running"
                ) from exc
            time.sleep(poll_seconds)
    try:  # pragma: no cover - non-POSIX platform
        os.write(descriptor, f"{os.getpid()}\n".encode("utf-8"))
        yield
    finally:  # pragma: no cover - non-POSIX platform
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def verify_chain(rows: Sequence[Mapping[str, Any]], *, where: str = "trial log") -> None:
    """Raise :class:`TrialLogError` at the first broken link.

    The chain detects change relative to a trusted copy only; it does not prove
    the search was exhaustive (§3.12). A *removed* row breaks it at the
    successor whose ``prev_hash`` no longer names the row before it, which is
    what makes deleting a row from an append-only log detectable rather than
    deniable.
    """
    previous: str | None = None
    for number, row in enumerate(rows, start=1):
        if "hash" not in row:
            raise TrialLogError(f"{where}:{number}: row carries no hash")
        expected = chain_hash(row)
        if row["hash"] != expected:
            raise TrialLogError(
                f"{where}:{number}: row hash {row['hash']} does not cover its content (recomputed {expected}); "
                "the row has been rewritten"
            )
        if row.get("prev_hash") != previous:
            raise TrialLogError(
                f"{where}:{number}: prev_hash {row.get('prev_hash')!r} does not chain to the preceding row "
                f"({previous!r}); a row has been removed, reordered or rewritten"
            )
        previous = row["hash"]


@dataclass(frozen=True)
class TrialLog:
    """One append-only, hash-chained trial log under ``workspace.trials``.

    Rows are appended, never edited (`SPEC.md:96-100`). Every append takes the
    exclusive lock, re-reads the log inside it, and chains onto whatever row is
    last at that moment, so two concurrent appenders cannot fork the chain.
    """

    path: Path
    lock_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def exists(self) -> bool:
        return self.path.exists()

    def rows(self) -> tuple[dict[str, Any], ...]:
        """Every row, parsed, in file order. No chain check."""
        if not self.path.exists():
            return ()
        parsed: list[dict[str, Any]] = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                raise TrialLogError(f"{self.path}:{number}: blank line in an append-only log")
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TrialLogError(f"{self.path}:{number}: not valid JSON: {exc}") from exc
        return tuple(parsed)

    def verify(self) -> tuple[dict[str, Any], ...]:
        """Parse the log and verify its hash chain, raising on any break."""
        rows = self.rows()
        verify_chain(rows, where=str(self.path))
        return rows

    def last_hash(self) -> str | None:
        rows = self.rows()
        return str(rows[-1]["hash"]) if rows else None

    def open_runs(self) -> tuple[dict[str, Any], ...]:
        """Every ``started`` row with no terminal row, oldest first."""
        rows = self.rows()
        terminal = {row.get("run_id") for row in rows if row.get("event") in TERMINAL_EVENTS}
        return tuple(row for row in rows if row.get("event") == STARTED and row.get("run_id") not in terminal)

    def append(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Append one protocol row, returning it with ``prev_hash`` and ``hash``.

        The whole read-check-write sequence runs inside the lock: the ordering
        rules (one ``started`` per run, at most one terminal row per run) are
        decided against the log as it is at the moment of the append, not as it
        was when the caller was built.
        """
        candidate = _protocol_row(row)
        with append_lock(self.path, timeout_seconds=self.lock_timeout_seconds):
            existing = self.rows()
            _check_ordering(existing, candidate, where=str(self.path))
            candidate["prev_hash"] = str(existing[-1]["hash"]) if existing else None
            candidate["hash"] = chain_hash(candidate)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(_to_jsonable(candidate), sort_keys=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return candidate


def _protocol_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one row's fields and return it in the declared field order."""
    unknown = sorted(set(row) - set(TRIAL_ROW_FIELDS))
    if unknown:
        raise TrialLogError(
            f"trial row carries field(s) {unknown} outside the row schema {TRIAL_ROW_FIELDS}; the schema is "
            "hashed into the chain, so an extra field forks the format"
        )
    event = row.get("event")
    if event not in TRIAL_EVENTS:
        raise TrialLogError(f"event {event!r} is not one of {TRIAL_EVENTS}")
    if not str(row.get("run_id") or "").strip():
        raise TrialLogError("a trial row needs a non-empty run_id")
    kind = row.get("kind")
    if kind not in RUN_KINDS:
        raise TrialLogError(f"kind {kind!r} is not one of {RUN_KINDS}")
    channel = row.get("run_channel")
    if channel not in RUN_CHANNELS:
        raise TrialLogError(f"run_channel {channel!r} is not one of {RUN_CHANNELS}")
    if not isinstance(row.get("synthetic"), bool):
        raise TrialLogError(f"synthetic must be a bool, got {row.get('synthetic')!r}")
    if event in TERMINAL_EVENTS and event != FINISHED and not str(row.get("reason") or "").strip():
        raise TrialLogError(f"a {event!r} row must carry the reason it is not a finish (§3.12)")
    built = {name: row.get(name) for name in TRIAL_ROW_FIELDS}
    built["ts"] = str(row.get("ts") or utc_now())
    built["prev_hash"] = None
    built.pop("hash", None)
    built["hash"] = None
    return {name: built[name] for name in TRIAL_ROW_FIELDS}


def _check_ordering(existing: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any], *, where: str) -> None:
    """The protocol's ordering rules, checked against the log under the lock."""
    run_id = candidate["run_id"]
    events = [row.get("event") for row in existing if row.get("run_id") == run_id]
    if candidate["event"] == STARTED:
        if STARTED in events:
            raise TrialLogError(
                f"{where}: run_id {run_id!r} already has a started row; a repeated execution is a new run_id, "
                "not a second start"
            )
        return
    terminal = [event for event in events if event in TERMINAL_EVENTS]
    if terminal:
        raise TrialLogError(
            f"{where}: run_id {run_id!r} already has the terminal row {terminal[0]!r}; at most one terminal row "
            "per run (§3.12), and a second would let a log report two outcomes for one attempt"
        )
    if STARTED not in events and candidate["event"] != FAILED:
        # ``failed`` is the exception the protocol names: identity resolution
        # runs *before* the started row, and a failure there is still an
        # attempt, so it writes a lone ``failed`` row (§3.12, "Ordering").
        raise TrialLogError(
            f"{where}: run_id {run_id!r} has no started row, so a {candidate['event']!r} row would record the "
            "end of an attempt the log never saw begin"
        )


@dataclass
class RunRecorder:
    """The run-recording protocol for one run (§3.12).

    The order is fixed and the class enforces it: the stop-rule and budget
    checks of §3.13 happen *before* this object is asked for a ``started`` row;
    identity resolution happens next, and a failure there still writes a
    ``failed`` row, because a specification that never resolved is still an
    attempt. Then ``started``, then exactly one terminal row.
    """

    log: TrialLog
    run_id: str
    config_id: str | None
    data_id: str
    kind: str
    selection_set_id: str | None
    arm_id: str | None = None
    capital: float | None = None
    synthetic: bool = False
    run_channel: str = "evidence"
    pid: int = field(default_factory=os.getpid)
    host: str = field(default_factory=socket.gethostname)
    started_row: dict[str, Any] | None = field(default=None, init=False)
    terminal_row: dict[str, Any] | None = field(default=None, init=False)

    def _row(self, event: str, *, summary: Any = None, reason: str | None = None) -> dict[str, Any]:
        now = utc_now()
        return {
            "ts": now,
            "event": event,
            "run_id": self.run_id,
            "config_id": self.config_id,
            "data_id": self.data_id,
            "kind": self.kind,
            "arm_id": self.arm_id,
            "selection_set_id": self.selection_set_id,
            "capital": None if self.capital is None else float(self.capital),
            "synthetic": bool(self.synthetic),
            "run_channel": self.run_channel,
            "summary": summary,
            "reason": reason,
            "pid": int(self.pid),
            "host": self.host,
            "heartbeat_at": now if event == STARTED else None,
        }

    def started(self) -> dict[str, Any]:
        """Append the ``started`` row. Nothing is computed before it."""
        if self.started_row is not None:
            raise TrialLogError(f"run {self.run_id!r} has already started")
        self.started_row = self.log.append(self._row(STARTED))
        return self.started_row

    def _terminal(self, event: str, *, summary: Any = None, reason: str | None = None) -> dict[str, Any]:
        if self.terminal_row is not None:
            raise TrialLogError(
                f"run {self.run_id!r} already recorded {self.terminal_row['event']!r}; at most one terminal row"
            )
        self.terminal_row = self.log.append(self._row(event, summary=summary, reason=reason))
        return self.terminal_row

    def finished(self, summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._terminal(FINISHED, summary=dict(summary or {}))

    def failed(self, reason: str) -> dict[str, Any]:
        return self._terminal(FAILED, reason=str(reason))

    def incomplete(self, reason: str, summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._terminal(INCOMPLETE, summary=dict(summary or {}) or None, reason=str(reason))


@dataclass(frozen=True)
class OrphanVerdict:
    """One ``started`` row with no terminal row, and what was decided about it."""

    run_id: str
    verdict: str
    reason: str
    age_seconds: float | None
    row: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "age_seconds": self.age_seconds,
            "pid": self.row.get("pid"),
            "host": self.row.get("host"),
            "started_at": self.row.get("ts"),
        }


@dataclass(frozen=True)
class ReconcileReport:
    """What ``prism trials reconcile`` did to one log."""

    path: Path
    n_open: int
    closed: tuple[OrphanVerdict, ...]
    in_flight: tuple[OrphanVerdict, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "n_open": self.n_open,
            "closed": [verdict.to_dict() for verdict in self.closed],
            "in_flight": [verdict.to_dict() for verdict in self.in_flight],
        }


def reconcile_trial_log(
    path: Path | str,
    *,
    now: datetime | None = None,
    max_heartbeat_age_seconds: float = DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    host: str | None = None,
    process_alive: Any = None,
    reason: str | None = None,
) -> ReconcileReport:
    """Close orphaned ``started`` rows, and only the ones proven dead (§3.12).

    A ``started`` row with no terminal row is *not* reconciled by the next
    invocation on sight: that would write a false ``incomplete`` for a live run
    and leave two terminal rows in an append-only log. An ``incomplete`` row is
    written only when the heartbeat is older than ``max_heartbeat_age_seconds``
    **and** the recorded process on the recorded host is gone. A run on another
    host cannot have its pid checked from here, so heartbeat age alone decides
    there, exactly as the execution lock's staleness rule does (§3.10).

    The chain is verified before anything is appended: reconciling onto a log
    whose chain is already broken would put a valid row on top of an invalid
    one and make the break look older than it is.
    """
    log = TrialLog(Path(path))
    log.verify()
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise TrialLogError("reconcile needs an aware timestamp: a naive 'now' compares against nothing")
    this_host = host if host is not None else socket.gethostname()
    alive = process_is_alive if process_alive is None else process_alive

    closed: list[OrphanVerdict] = []
    in_flight: list[OrphanVerdict] = []
    orphans = log.open_runs()
    for row in orphans:
        age = _row_age_seconds(row, moment)
        stale = age is None or age > float(max_heartbeat_age_seconds)
        row_host = row.get("host")
        if row_host != this_host:
            if not stale:
                in_flight.append(
                    OrphanVerdict(
                        run_id=str(row.get("run_id")),
                        verdict="unverifiable_host",
                        reason=(
                            f"the run was started on {row_host!r} and this is {this_host!r}; its pid cannot be "
                            f"checked from here and its heartbeat is {age:.0f}s old, inside the "
                            f"{max_heartbeat_age_seconds:.0f}s window"
                        ),
                        age_seconds=age,
                        row=row,
                    )
                )
                continue
            verdict_reason = (
                f"heartbeat {age:.0f}s old (limit {max_heartbeat_age_seconds:.0f}s) and the recorded host "
                f"{row_host!r} is not this host, so the pid cannot be checked"
                if age is not None
                else f"no usable heartbeat on the row, and the recorded host {row_host!r} is not this host"
            )
        else:
            if alive(row.get("pid")):
                in_flight.append(
                    OrphanVerdict(
                        run_id=str(row.get("run_id")),
                        verdict="process_alive",
                        reason=f"pid {row.get('pid')} is running on {row_host!r}; the run is still executing",
                        age_seconds=age,
                        row=row,
                    )
                )
                continue
            if not stale:
                in_flight.append(
                    OrphanVerdict(
                        run_id=str(row.get("run_id")),
                        verdict="heartbeat_fresh",
                        reason=(
                            f"pid {row.get('pid')} is gone but the heartbeat is only {age:.0f}s old (limit "
                            f"{max_heartbeat_age_seconds:.0f}s); a pid can be recycled, so the run is reported, "
                            "not closed"
                        ),
                        age_seconds=age,
                        row=row,
                    )
                )
                continue
            verdict_reason = (
                f"heartbeat {age:.0f}s old (limit {max_heartbeat_age_seconds:.0f}s) and pid {row.get('pid')} is "
                f"gone on {row_host!r}"
                if age is not None
                else f"no usable heartbeat on the row and pid {row.get('pid')} is gone on {row_host!r}"
            )
        text = reason or (
            "the run started and never recorded a terminal row: "
            + verdict_reason
            + ". The attempt is counted; its metrics are not."
        )
        appended = log.append(
            {
                "ts": utc_now(),
                "event": INCOMPLETE,
                "run_id": row.get("run_id"),
                "config_id": row.get("config_id"),
                "data_id": row.get("data_id"),
                "kind": row.get("kind"),
                "arm_id": row.get("arm_id"),
                "selection_set_id": row.get("selection_set_id"),
                "capital": row.get("capital"),
                "synthetic": bool(row.get("synthetic")),
                "run_channel": row.get("run_channel"),
                "summary": None,
                "reason": text,
                "pid": os.getpid(),
                "host": this_host,
                "heartbeat_at": None,
            }
        )
        closed.append(
            OrphanVerdict(
                run_id=str(row.get("run_id")),
                verdict="closed_incomplete",
                reason=str(appended["reason"]),
                age_seconds=age,
                row=row,
            )
        )
    return ReconcileReport(path=Path(path), n_open=len(orphans), closed=tuple(closed), in_flight=tuple(in_flight))


def _row_age_seconds(row: Mapping[str, Any], now: datetime) -> float | None:
    """Age of a row's heartbeat (or its timestamp), in seconds, or None."""
    for key in ("heartbeat_at", "ts"):
        raw = row.get(key)
        if not isinstance(raw, str) or not raw.strip() or raw == UNAVAILABLE:
            continue
        try:
            stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return (now - stamp).total_seconds()
    return None


def attempt_counts(rows: Sequence[Mapping[str, Any]], *, registered: bool) -> dict[str, Any]:
    """Attempt and independent-evidence counts over a log's rows (§3.12).

    Within a registered selection set an attempt is one distinct ``arm_id``;
    outside one it is one distinct ``config_id``. A ``started`` configuration
    counts whether or not it finished. Rows of kind ``retry``, ``replay``,
    ``reproduction`` and ``imported``, and any synthetic row, are excluded from
    the independent count and the exclusion is reported, so the count is
    reproducible rather than asserted.

    An engineering-channel row is excluded from *both* counts: that channel is
    in no selection set and enters no attempt count at all (§3.12, "Which
    commands write rows"), which is what lets private CI run on every push
    without evidence history growing with commit traffic.
    """
    key = "arm_id" if registered else "config_id"
    attempts: set[Any] = set()
    independent: set[Any] = set()
    excluded: dict[str, int] = {}
    for row in rows:
        if row.get("event") == GENESIS_EVENT:
            continue
        identity = row.get(key)
        if identity is None:
            identity = f"run:{row.get('run_id')}"
        if str(row.get("run_channel")) == "engineering":
            excluded["engineering"] = excluded.get("engineering", 0) + 1
            continue
        attempts.add(identity)
        kind = str(row.get("kind"))
        if bool(row.get("synthetic")):
            excluded["synthetic"] = excluded.get("synthetic", 0) + 1
            continue
        if kind in UNCOUNTED_KINDS:
            excluded[kind] = excluded.get(kind, 0) + 1
            continue
        independent.add(identity)
    return {
        "attempt_basis": key,
        "n_trials_observed": len(attempts),
        "n_independent": len(independent),
        "excluded_rows": dict(sorted(excluded.items())),
    }
