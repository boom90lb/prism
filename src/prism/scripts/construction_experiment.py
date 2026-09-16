"""The stage 4 construction experiment: three declared arms, six capital scenarios (§5).

What this runs. ``registrations/2026-09-construction.toml`` declares three arms --
``decile_control``, ``convex_point`` and ``convex_p80_one_sided`` -- on one frozen score
panel, one fold geometry, one cost stack and one accounting engine, across six capital
scenarios. Eighteen evaluation records come out, one per (arm, scenario) pair, and one
comparison record holding six blocks, one per capital level, each with the three pairwise
arm-versus-arm Sharpe differences at that level (§3.12).

What it does not do, and cannot. **No arm promotes.** The registration's
``[promotion_rule]`` is ``kind = "none"`` and this is a same-sample comparison: three arms
read against each other on the sample they were chosen against. Such a comparison can kill
a configuration -- an arm that loses to the control under identical charged cost
conventions is dead -- and it can never promote one. The reference book's promotion stays
under the existing prospective rules of ``SPEC.md``, with its own profile, its own record
and its own forward evidence, and nothing recorded here may be cited for it.

**Identical charged cost conventions** means the frozen ``cost_stack`` and the rules that
apply it, not identical charged dollars. Charged dollars are not identical across capital
levels and cannot be: participation is trade dollars over dollar volume from current
equity, impact is charged on filled notional, and rounding at the per-sid increment
changes filled notional. Freezing charged costs at one capital level would defeat the
sweep, which exists to measure exactly that variation.

Usage::

    python -m prism.scripts.construction_experiment --freeze
    python -m prism.scripts.construction_experiment --arm decile_control --capital 100000
    python -m prism.scripts.construction_experiment --compare --out results/2026-09-construction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from prism.evaluate import CanonicalSettings, EvaluationResult, walk_forward_evaluate
from prism.execution.cost_model import CostStack
from prism.execution.simulate import FillPolicy
from prism.io.dataset import Dataset
from prism.logging_utils import configure_logging
from prism.portfolio.rounding import RoundingSettings
from prism.registration import Registration, load_registration
from prism.validation.compare import (
    ComparisonArm,
    SharpeBootstrapSpec,
    compare_reevaluation,
)
from prism.validation.trials import stable_json_dumps
from prism.validation.uncertainty import BLOCK_SCHEMES, CIRCULAR_MOVING
from prism.workspace import Workspace, resolve_workspace

logger = logging.getLogger(__name__)

#: The registration this module runs. It is named here rather than defaulted from a
#: directory scan: a driver that picked up whichever registration it found could run an
#: arm of one experiment under the budget of another.
REGISTRATION = "registrations/2026-09-construction.toml"

#: The frozen dataset the registration's ``data_id`` names. It is the recorded momentum
#: configuration's own inputs, built at stage 2 and reused unchanged.
DATASET = "data/frozen/demotion_b1"

#: What the runs cannot establish and the record must say so (§3.12, ``coverage``).
UNTESTED: tuple[str, ...] = (
    "The comparison is same-sample: three arms read against each other on one frozen score "
    "panel, one fold geometry and one cost stack. It can kill a configuration and it cannot "
    "promote one; no arm promotes out of this registration (§3.13, [promotion_rule]).",
    "The calibration's training cross-section is the formation window's point-in-time "
    "membership at the fit cutoff, without the per-bar eligibility screen the decision path "
    "applies; the screen is applied at every decision bar and to the labels' universe, not "
    "bar by bar inside the training pool.",
    "The frozen dataset carries no cash-distribution events, so the 'credited' label of §3.4 "
    "credits zero and the accounting credits zero; the two agree, and neither establishes "
    "what a dividend-bearing panel would have produced.",
    "Fills are simulated at the next open under the declared [fills] policy. No panel here "
    "is evidence about actual fills.",
)


class ExperimentError(RuntimeError):
    """The experiment cannot be run as asked, and running it anyway would forge the record."""


def registration_hash(path: Path) -> str:
    """``sha256`` of the registration's bytes, recorded beside ``registration_id`` (§3.12).

    The registration's content is deliberately *not* hashed into ``config_id``: it declares
    which runs are admitted, not what was run, so an edit to a sweep list must not revoke a
    live authorization. It is recorded here instead.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_settings(registration: Registration) -> CanonicalSettings:
    """The charged policy every arm runs under: the registration's frozen stack and profile.

    One object, built once, shared by all eighteen runs. The quantity increment, the dust
    floor and the zero-crossing clamp are the execution profile's; every charged component
    including its zeros is the stack's (§3.6).
    """
    stack = CostStack.from_mapping(tomllib.loads(registration.frozen.cost_stack.read_text(encoding="utf-8")))
    profile = tomllib.loads(registration.frozen.execution_profile.read_text(encoding="utf-8"))
    rounding = RoundingSettings.from_profile(profile)
    return CanonicalSettings(
        cost_stack=stack,
        share_increment=float(profile.get("quantity_increment_default", 1.0)),
        min_notional=float(profile.get("min_notional", 0.0)),
        zero_crossing_clamp=bool(rounding.zero_crossing_clamp),
        fills=FillPolicy(),
    )


# --------------------------------------------------------------------------
# The opening step: freeze the score panel
# --------------------------------------------------------------------------


def freeze_scores(
    *,
    workspace: Workspace,
    registration: Registration,
    dataset: Dataset,
    panel_out: Path | None = None,
) -> dict[str, Any]:
    """Materialize the score panel the three arms share and write the freeze record.

    The registration names ``[frozen].freeze_record`` as an input that the run's own
    opening step writes, and refuses every run until it exists, so that no arm can freeze a
    panel implicitly and a later arm read a different one (§3.13).

    What is frozen is the *inputs* the score is a function of, content-hashed: the dataset
    manifest id, the visibility mode and vintage the panel was cut under, the universe, the
    component's import path and source digest, and the fold geometry. The panel itself is
    materialized and hashed so a reader can re-derive it; the arms recompute the score at
    each decision bar from the same frozen dataset under the same point-in-time view rather
    than reading the file, and the record says so instead of implying otherwise.
    """
    from prism.evaluate import _resolve, _score_bar  # noqa: PLC0415

    spec = registration.arm_specification(registration.arm_ids[0], workspace=workspace)
    resolved = _resolve(
        spec,
        workspace=workspace,
        dataset=dataset,
        fold_table=registration.fold_table,
        splitter=None,
        fold_table_id=None,
        engine="canonical",
        capital=float(registration.frozen.capital_scenarios[0]),
        fold_portfolio=registration.frozen.fold_portfolio,
        cost_refresh=registration.frozen.cost_refresh,
        synthetic=None,
        start=None,
        end=None,
        visibility_mode=registration.frozen.visibility_mode,
        decision_time=None,
        canonical=canonical_settings(registration),
        price_basis=None,
        decision_coupling="account_state",
    )
    labels = resolved.session_labels
    universe = list(resolved.universe)
    plans = tuple(
        component.plan(combination=spec.strategy.combination)
        for component in _fit_for_freeze(spec, resolved, workspace)
    )
    step = max(int(plan.update_every_bars) for plan in plans)
    first = min(fold.test_start for fold in resolved.folds)
    last = max(fold.test_stop for fold in resolved.folds)
    rows: dict[str, pd.Series] = {}
    warnings: list[str] = []
    states: dict[str, Any] = {}
    for position in range(first, last, step):
        scored = _score_bar(
            spec,
            resolved,
            labels[position],
            position=position,
            universe=universe,
            plans=plans,
            states=states,
            needs_total_return=False,
            history=max(int(loaded.declarations["required_history"]) for loaded in spec.components),
            warnings=warnings,
            construction=None,
        )
        if scored is None:
            continue
        states = dict(scored.composition.states)
        for output in scored.composition.component_outputs:
            if output.raw is not None:
                rows[labels[position]] = output.raw.reindex(pd.Index(universe))
    panel = pd.DataFrame(rows).T.sort_index()
    digest = hashlib.sha256(panel.round(12).to_csv(float_format="%.12g").encode("utf-8")).hexdigest()
    universe_digest = hashlib.sha256(stable_json_dumps(sorted(universe)).encode("utf-8")).hexdigest()
    record = {
        "registration_id": registration.id,
        "selection_set": registration.selection_set,
        "data_id": resolved.data_id,
        "visibility_mode": resolved.visibility.mode,
        "vintage": resolved.vintage,
        "fold_table_sha256": resolved.fold_table.content_hash,
        "universe_sha256": universe_digest,
        "n_universe": len(universe),
        "panel_sha256": digest,
        "n_decision_rows": int(len(panel)),
        "first_row": None if panel.empty else str(panel.index[0]),
        "last_row": None if panel.empty else str(panel.index[-1]),
        "components": [
            {
                "component_id": loaded.component_id,
                "import_path": loaded.spec.path,
                "source_sha256": loaded.declarations.get("source_sha256"),
                "update_every_bars": int(loaded.spec.update_every_bars),
                "horizon_bars": int(loaded.declarations["horizon_bars"]),
            }
            for loaded in spec.components
        ],
        "note": (
            "the panel is materialized from this dataset, this universe and this component identity under "
            "the recorded visibility mode; every arm recomputes the same score at each decision bar from the "
            "same frozen dataset rather than reading this file, and panel_sha256 is what a reader re-derives"
        ),
        "warnings": warnings,
    }
    target = Path(registration.frozen.freeze_record)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if panel_out is not None:
        Path(panel_out).parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(panel_out)
    logger.info("froze the score panel: %d rows x %d names -> %s", len(panel), len(universe), target)
    return record


def _fit_for_freeze(spec: Any, resolved: Any, workspace: Workspace) -> tuple[Any, ...]:
    """Components fitted on the first fold's formation window, for the freeze pass only."""
    from prism.evaluate import _members, prepare_components  # noqa: PLC0415

    fold = resolved.folds[0]
    train_start = resolved.session_labels[fold.formation_start]
    train_end = resolved.session_labels[fold.formation_stop - 1]
    view = resolved.view_at(train_end)
    members = _members(spec, view, list(resolved.universe))
    return prepare_components(
        spec,
        view,
        train_end,
        fits_dir=workspace.fits(),
        members=members,
        window=(train_start, train_end),
    )


# --------------------------------------------------------------------------
# The runs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmRun:
    """One (arm, capital) evaluation and what it cost to produce."""

    arm_id: str
    capital: float
    result: EvaluationResult
    seconds: float


def run_arm(
    *,
    workspace: Workspace,
    registration: Registration,
    dataset: Dataset,
    arm_id: str,
    capital: float,
    kind: str = "new",
    reconcile: bool = True,
) -> ArmRun:
    """Evaluate one declared arm at one declared capital scenario, under strict mode.

    ``walk_forward_evaluate`` runs the registration's checks itself, after identity
    resolution and before the ``started`` row: a run whose resolved ``config_id`` is not a
    declared arm at a declared scenario is refused, writes no trial row, and is not an
    attempt (§3.12, §3.13).
    """
    spec = registration.arm_specification(arm_id, workspace=workspace)
    started = time.time()
    result = walk_forward_evaluate(
        spec,
        workspace=workspace,
        dataset=dataset,
        fold_table=registration.fold_table,
        engine=registration.frozen.accounting_engine,
        capital=float(capital),
        kind=kind,
        selection_set_id=registration.selection_set,
        registration_id=registration.id,
        registration_hash=registration_hash(registration.path),
        arm_id=arm_id,
        run_channel="evidence",
        canonical=canonical_settings(registration),
        visibility_mode=registration.frozen.visibility_mode,
        fold_portfolio=registration.frozen.fold_portfolio,
        cost_refresh=registration.frozen.cost_refresh,
        decision_coupling="account_state",
        calibration_family=registration.shared.calibration_family,
        registration=registration,
        reconcile=reconcile,
        untested_conclusions=UNTESTED,
    )
    seconds = time.time() - started
    logger.info(
        "arm %s at capital %.0f: run_id %s config_id %s in %.1fs",
        arm_id,
        capital,
        result.run_id,
        result.config_id[:12],
        seconds,
    )
    return ArmRun(arm_id=arm_id, capital=float(capital), result=result, seconds=seconds)


# --------------------------------------------------------------------------
# The comparison record
# --------------------------------------------------------------------------


def _record_of(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def comparison_arm(record_path: Path, *, weight_index: pd.Index | None = None) -> ComparisonArm:
    """One arm of the comparison, read from the record and artifacts ``prism evaluate`` wrote.

    ``gross`` is ``net + cost`` per bar, which the canonical engine's own record writer
    states as an identity rather than an approximation: marked equity is additive, so
    ``E_t = E_{t-1} + pnl_t - cost_t``. Nothing is recomputed from prices here.
    """
    record = _record_of(record_path)
    run_dir = Path(record_path).parent
    net = pd.read_csv(run_dir / "returns.csv", index_col=0, parse_dates=True).iloc[:, 0].rename("net")
    costs = pd.read_csv(run_dir / "costs.csv", index_col=0, parse_dates=True)
    weights = pd.read_csv(run_dir / "target_weights.csv", index_col=0, parse_dates=True)
    if weight_index is not None:
        weights = weights.reindex(index=weight_index)
    return ComparisonArm(
        record=record,
        net_returns=net,
        gross_returns=(net + costs["total"].reindex(net.index).fillna(0.0)).rename("gross"),
        costs=costs["total"].reindex(net.index).fillna(0.0).rename("cost"),
        turnover=costs["turnover"].reindex(net.index).fillna(0.0).rename("turnover"),
        weights=weights,
        by_fold=None,
    )


def common_decision_index(records: Sequence[Path]) -> tuple[pd.Index, dict[str, int]]:
    """The decision bars every arm decided on, and how many each arm decided outside it.

    The holdings-overlap panel compares weight rows bar by bar, so it needs one index. An
    arm that decided on a bar another did not is a difference the record states rather than
    a row silently dropped from one side of a comparison. It is taken *within* a capital
    block, because that is where the overlap is computed: intersecting across capital
    levels would drop a bar from a block in which every arm decided on it.
    """
    frames = {
        str(path): pd.read_csv(Path(path).parent / "target_weights.csv", index_col=0, parse_dates=True).index
        for path in records
    }
    shared: pd.Index | None = None
    for index in frames.values():
        shared = index if shared is None else shared.intersection(index)
    assert shared is not None  # a comparison has at least one arm
    return shared, {name: int(len(index) - len(shared)) for name, index in frames.items()}


def bootstrap_spec(registration: Registration, *, decision_interval_bars: int = 1) -> SharpeBootstrapSpec:
    """The declared ``[sharpe_bootstrap]`` block, before the run and not after it.

    ``paired`` is not a field of :class:`SharpeBootstrapSpec` because the pairing is
    structural: :func:`sharpe_difference_interval` resamples both arms on the same blocks
    and has no unpaired mode. The registration declares ``paired = true`` and a
    declaration of ``false`` is refused here rather than silently ignored.

    ``decision_interval_bars`` is the arms' own decision cadence, so the declared rule
    "a block spans at least one decision" is checked against the run rather than against a
    default of one bar.
    """
    declared = dict(registration.sharpe_bootstrap)
    scheme = str(declared["scheme"])
    if scheme not in BLOCK_SCHEMES:
        # The registration is frozen and its sha256 is recorded in every run that already
        # cites it, so a spelling the engine's enum does not carry is mapped here and the
        # mapping is recorded, never edited into the file after the runs were made. Only
        # the one synonym is accepted; anything else is refused rather than guessed.
        if scheme != "circular":
            raise ExperimentError(
                f"{registration.path} [sharpe_bootstrap] declares scheme={scheme!r}; the engine's block "
                f"schemes are {BLOCK_SCHEMES} and only 'circular' is accepted as a spelling of "
                "'circular_moving' (§3.12)"
            )
        scheme = CIRCULAR_MOVING
    if not bool(declared.get("paired", True)):
        raise ExperimentError(
            f"{registration.path} [sharpe_bootstrap] declares paired = false; the difference interval "
            "resamples both arms on the same blocks and has no unpaired mode (§3.12)"
        )
    return SharpeBootstrapSpec(
        frequency=str(declared["frequency"]),
        block_bars=int(declared["block_bars"]),
        n_resamples=int(declared["n_resamples"]),
        seed=int(declared["seed"]),
        scheme=scheme,
        decision_interval_bars=int(decision_interval_bars),
    )


def exit_conditions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The stage 4 exit paragraph's own fields, read off the eighteen records (§5).

    Excluded folds, which must be the same for every arm; the fraction of names at the
    per-name cap for the convex arms; the repair outcome at each capital level; and the
    charged cost at each capital level, which is what "identical charged cost conventions"
    does *not* mean.
    """
    by_arm: dict[str, dict[str, Any]] = {}
    exclusions: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        arm = str(record["ids"]["arm_id"])
        capital = float(record["evaluation"]["capital"])
        construction = record["evaluation"]["construction"]
        entry = by_arm.setdefault(arm, {"kind": construction.get("kind"), "by_capital": {}})
        entry["by_capital"][f"{capital:.0f}"] = {
            "at_cap_fraction_median": construction["at_cap_fraction"]["median"],
            "at_cap_fraction_max": construction["at_cap_fraction"]["max"],
            "max_at_cap_fraction": construction["at_cap_fraction"]["threshold"],
            "concentration_dominated": construction["at_cap_fraction"]["concentration_dominated"],
            "ex_ante_sd_median": construction["ex_ante_sd"]["median"],
            "n_decisions": construction["n_decisions"],
            "paths": construction["paths"],
            "repair": record["performance"]["repair"],
            "straddling_decisions": construction["straddling_decisions"],
            "flat_from_straddling_endpoints": construction["flat_from_straddling_endpoints"],
            "binding_conditions": record["performance"]["binding_conditions"],
            "charged_cost_fraction": record["performance"]["cost"]["total"],
            "turnover": record["performance"]["turnover"],
            "net_sharpe": record["performance"]["sharpe"],
            "n_obs": record["performance"]["sample"]["n_obs"],
        }
        exclusions[f"{arm}@{capital:.0f}"] = [dict(row) for row in record["evaluation"]["excluded_folds"]]
    distinct = {stable_json_dumps(value) for value in exclusions.values()}
    return {
        "excluded_folds": {
            "shared": len(distinct) == 1,
            "per_run": exclusions,
            "note": (
                "folds excluded by min_training_periods or min_blocks are excluded for all three arms alike; "
                "admission is decided from the shared calibration family and never from an arm's own "
                "confidence (decision D6)"
            ),
        },
        "arms": by_arm,
    }


def build_comparison(
    *,
    workspace: Workspace,
    registration: Registration,
    records: Sequence[Path],
) -> dict[str, Any]:
    """The comparison record of §3.12: six capital blocks, three differences in each."""
    by_capital: dict[str, list[Path]] = {}
    for path in records:
        by_capital.setdefault(f"{float(_record_of(path)['evaluation']['capital']):.0f}", []).append(path)
    shared: dict[str, int] = {}
    dropped: dict[str, int] = {}
    arms: list[ComparisonArm] = []
    for capital, block in sorted(by_capital.items(), key=lambda item: float(item[0])):
        index, outside = common_decision_index(block)
        shared[capital] = int(len(index))
        dropped.update(outside)
        arms.extend(comparison_arm(path, weight_index=index) for path in block)
    control = registration.arm_specification(registration.arm_ids[0], workspace=workspace)
    cadence = max(int(loaded.spec.update_every_bars) for loaded in control.components)
    report = compare_reevaluation(arms, bootstrap_spec(registration, decision_interval_bars=cadence))
    payload = report.to_dict()
    payload["holdings_overlap_index"] = {
        "n_decision_bars_by_capital": shared,
        "rows_outside_the_shared_index": dropped,
        "note": (
            "the holdings-overlap panel is computed on the decision bars every arm decided on; a row an arm "
            "decided and another did not is counted here rather than dropped silently"
        ),
    }
    payload["registration"] = {
        "id": registration.id,
        "sha256": registration_hash(registration.path),
        "selection_set": registration.selection_set,
        "arms": list(registration.arm_ids),
        "capital_scenarios": list(registration.frozen.capital_scenarios),
        "trial_budget": registration.trial_budget,
    }
    payload["promotion_check"] = {
        "kind": registration.promotion_rule.kind,
        "promotes": bool(registration.promotion_rule.promotes()),
        "outcome": "no arm promotes",
        "note": registration.promotion_rule.note,
        "recorded_by": "prism compare --promotion-check",
        "reason": (
            "a same-sample comparison can kill a configuration and can never promote one; the reference "
            "book's promotion stays under the existing prospective rules of SPEC.md"
        ),
    }
    payload["exit_conditions"] = exit_conditions([_record_of(path) for path in records])
    payload["runs"] = trial_log_runs(workspace, registration, records)
    payload["charged_cost_conventions"] = {
        "cost_stack": str(registration.frozen.cost_stack),
        "cost_stack_sha256": hashlib.sha256(registration.frozen.cost_stack.read_bytes()).hexdigest(),
        "execution_profile": str(registration.frozen.execution_profile),
        "cost_refresh": registration.frozen.cost_refresh,
        "held_identical": (
            "the frozen cost stack and the rules that apply it, at every capital level and for every arm"
        ),
        "not_identical": (
            "charged dollars: participation is trade dollars over dollar volume from current equity, impact "
            "is charged on filled notional, and rounding at the per-sid increment changes filled notional, so "
            "charged cost varies with account size by construction. Freezing charged costs at one capital "
            "level would defeat the sweep."
        ),
    }
    return payload


def trial_log_runs(
    workspace: Workspace,
    registration: Registration,
    records: Sequence[Path],
) -> dict[str, Any]:
    """Which runs the comparison reads, and which other rows the selection set holds.

    A repeated execution of one configuration is a row, not a new attempt (§3.12), so the
    log legitimately carries more finished rows than the eighteen the comparison reads --
    a shakedown pass under an earlier build of the decision path, and any retry. Naming
    both sets here is what lets a reader check that the comparison read one pass rather
    than the best rows of several.
    """
    path = workspace.trial_log(registration.selection_set)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    used = {str(_record_of(record)["ids"]["run_id"]) for record in records}
    finished = [row for row in rows if str(row.get("event")) == "finished"]
    return {
        "trial_log": str(path),
        "n_rows": len(rows),
        "attempts": sorted({str(row.get("arm_id")) for row in rows if row.get("arm_id")}),
        "read_by_this_comparison": sorted(used),
        "other_finished_rows": sorted({str(row["run_id"]) for row in finished if str(row["run_id"]) not in used}),
        "note": (
            "an attempt is one arm_id; six capital scenarios of one arm spend one attempt between them, and a "
            "repeated execution of the same configuration is a row and not an attempt (§3.12)"
        ),
    }


def _short(value: Any, *, width: int = 72) -> str:
    """A condition value as one scannable token; a long structure is named, not printed.

    The cost-refresh condition is a whole per-fold table. Printing it inside a one-line
    summary would bury the twelve conditions that fit beside it, and the full value is in
    the JSON record either way.
    """
    text = str(value)
    if len(text) <= width:
        return text
    if isinstance(value, Mapping):
        return "{" + ", ".join(sorted(str(key) for key in value)) + "} (see the record)"
    return text[: width - 3] + "..."


def report_text(payload: Mapping[str, Any]) -> str:
    """The comparison as a table a reader can scan, with every panel labeled (§3.12).

    Every number here is read off the record; nothing is recomputed. The provenance flags
    are printed with the arms rather than in a footnote, because a simulated-fill panel and
    an actual-fill panel must never be displayed alike.
    """
    lines: list[str] = []
    registration = payload["registration"]
    lines.append(f"registration {registration['id']}  sha256 {registration['sha256'][:12]}")
    lines.append(f"selection set {registration['selection_set']}  arms {', '.join(registration['arms'])}")
    held = payload["conditions_held_fixed"]
    lines.append("held fixed: " + ", ".join(f"{key}={_short(value)}" for key, value in sorted(held.items())))
    if payload["conditions_that_differ"]:
        lines.append("differs by design: " + ", ".join(sorted(payload["conditions_that_differ"])))
    promotion = payload["promotion_check"]
    lines.append(f"promotion check: {promotion['outcome']} (rule kind={promotion['kind']!r})")
    exits = payload["exit_conditions"]
    excluded = exits["excluded_folds"]
    first = next(iter(excluded["per_run"].values()), [])
    lines.append(
        f"excluded folds (shared across every arm: {excluded['shared']}): "
        + (", ".join(str(entry["index"]) for entry in first) or "none")
    )
    lines.append("")
    header = (
        f"{'capital':>10} {'arm':<22}{'sharpe':>9}{'ann.ret':>9}{'cost':>9}{'turnover':>10}"
        f"{'at_cap':>8}{'ex_sd':>8}{'repair none/scale/drop':>24}{'flat':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for arm, entry in sorted(exits["arms"].items()):
        for capital in sorted(entry["by_capital"], key=float):
            row = entry["by_capital"][capital]
            repair = row["repair"]
            lines.append(
                f"{float(capital):>10,.0f} {arm:<22}"
                f"{row['net_sharpe']['annualized']:>9.4f}"
                f"{row['net_sharpe']['periodic']:>9.4f}"
                f"{row['charged_cost_fraction']:>9.4f}"
                f"{row['turnover']['total']:>10.2f}"
                f"{(row['at_cap_fraction_median'] if row['at_cap_fraction_median'] is not None else float('nan')):>8.3f}"
                f"{(row['ex_ante_sd_median'] if row['ex_ante_sd_median'] is not None else float('nan')):>8.4f}"
                f"{repair['succeeded']:>10d}/{repair['scaled_batch']:d}/{repair['dropped_batch']:d}".ljust(0)
                + f"{row['flat_from_straddling_endpoints']:>6d}"
            )
    lines.append("")
    lines.append(
        f"studentized Sharpe differences, {payload['method']['block_scheme']} blocks of "
        f"{payload['method']['block_bars']} bars, {payload['method']['n_resamples']} resamples, "
        f"seed {payload['method']['seed']}, hac_lags {payload['method']['hac_lags']}, "
        f"decision interval {payload['method']['decision_interval_bars']} bars, "
        f"{payload['method']['interval_method']}"
    )
    lines.append(
        "  the statistic is the per-bar (periodic) Sharpe difference; the annualized column is the same "
        "interval scaled by sqrt(252)"
    )
    head = (
        f"{'capital':>10} {'difference':<44}{'diff':>9}{'lower':>9}{'upper':>9}{'p':>7}"
        f"{'diff/yr':>9}{'lower/yr':>10}{'upper/yr':>10}"
    )
    lines.append(head)
    lines.append("-" * len(head))
    scale = math.sqrt(252.0)
    for block in sorted(payload["blocks"], key=lambda entry: float(entry["capital"])):
        for entry in block["sharpe_differences"]:
            lines.append(
                f"{float(block['capital']):>10,.0f} {entry['label']:<44}"
                f"{entry['difference']:>9.4f}{entry['lower']:>9.4f}{entry['upper']:>9.4f}"
                f"{entry['p_value_two_sided']:>7.3f}"
                f"{entry['difference'] * scale:>9.4f}{entry['lower'] * scale:>10.4f}"
                f"{entry['upper'] * scale:>10.4f}"
            )
    lines.append("")
    lines.append(f"intervals computed: {payload['selection']['n_intervals']}")
    for warning in payload.get("warnings", ()):
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--registration", default=REGISTRATION)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--freeze", action="store_true", help="write the frozen score-panel record and stop")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--kind", default="new")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--records", nargs="*", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    workspace = resolve_workspace(args.workspace).ensure()
    configure_logging(workspace.logs, verbose=args.verbose, run_name="construction_experiment")
    registration = load_registration(args.registration, workspace=workspace)
    dataset = Dataset.open(Path(args.dataset))

    if args.freeze:
        record = freeze_scores(workspace=workspace, registration=registration, dataset=dataset)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    if args.compare:
        if not args.records:
            raise ExperimentError("--compare needs the eighteen record paths, one per (arm, scenario) pair")
        payload = build_comparison(
            workspace=workspace, registration=registration, records=[Path(p) for p in args.records]
        )
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
            out.with_suffix(".txt").write_text(report_text(payload) + "\n", encoding="utf-8")
        print(report_text(payload))
        return 0

    if args.arm is None or args.capital is None:
        raise ExperimentError("name the arm and the capital scenario, or pass --freeze or --compare")
    run = run_arm(
        workspace=workspace,
        registration=registration,
        dataset=dataset,
        arm_id=args.arm,
        capital=args.capital,
        kind=args.kind,
    )
    print(
        json.dumps(
            {
                "arm_id": run.arm_id,
                "capital": run.capital,
                "run_id": run.result.run_id,
                "config_id": run.result.config_id,
                "record": str(run.result.record_path),
                "seconds": round(run.seconds, 1),
                "metrics": {key: float(value) for key, value in run.result.metrics.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "DATASET",
    "REGISTRATION",
    "ArmRun",
    "ExperimentError",
    "build_comparison",
    "bootstrap_spec",
    "canonical_settings",
    "comparison_arm",
    "exit_conditions",
    "freeze_scores",
    "main",
    "registration_hash",
    "report_text",
    "trial_log_runs",
    "run_arm",
]


if __name__ == "__main__":  # pragma: no cover - console entry
    raise SystemExit(main())
