"""The recovery policy file, the three recovery stages, and the no-action path (spec §3.7).

This module holds no solver. It orchestrates: it receives a *program* -- an object that
knows how to write one convex expression per limit, one per obligation, and how to solve
what it is handed -- and it drives that object through the declared sequence.
``portfolio/convex.py`` supplies the only program this engine has, and the split is what
keeps ``cvxpy`` out of the policy layer entirely: nothing here imports it, directly or
transitively.

**The three stages, and why the order is not negotiable.**

``R0`` makes the largest progress on obligations the *action restrictions* allow, taking
the obligations in the total order of :func:`prism.portfolio.obligations.order_obligations`
one ``(priority, deadline)`` group at a time. Each group maximizes
``sum_i progress_i / remaining_qty_i`` -- a concave objective, since ``progress_i`` is the
minimum of an affine function of the traded weight and a constant -- subject to every
action restriction and to the achieved progress of every earlier group fixed as a lower
bound. Obligation progress outranks every preference limit: an obligation leg may make a
preference violation worse, and R1 minimizes that violation from there but may never undo
the progress.

``R1`` walks every holding-state limit in :func:`prism.portfolio.limits.reduction_order`,
**satisfied limits included with an achieved target of zero**, and minimizes each limit's
violation measure ``v(w)`` subject to all action restrictions, R0's achieved progress as a
bound, and each higher-priority limit's achieved violation plus ``bound_slack`` as a
bound. Every ``v`` in the §3.7 table is convex, so each achieved-value bound is a convex
constraint and every stage of the sequence is a convex program.

``R2`` maximizes the economic objective subject to all action restrictions, R0's achieved
progress, and every R1 achieved value plus ``bound_slack``. The guarantee is therefore
checkable and is checked, by ``portfolio/validate.py`` and not by the solver's own status:
**R2 cannot increase any violation beyond the value R1 achieved**. It is not the stronger
claim that R2 introduces nothing -- R0 and R1 can each introduce a violation of a limit the
marked state satisfied, and a path that promised otherwise could not honor it.

**The no-action path.** When no executable improvement exists -- a stage that does not
solve, or a solved target the independent validation refuses -- the declared
``on_infeasible`` response is taken and *no orders are submitted*. Neither no-trade nor the
zero portfolio is assumed executable: this module never falls back to "hold ``w0``" or "go
flat" as though either were a validated action. It returns the response the file declares
and the reason, and the caller acts on it.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence

import pandas as pd

from prism.portfolio.limits import (
    MEASURED_LIMITS,
    Limit,
    LimitContext,
    reduction_order,
    violation,
)
from prism.portfolio.obligations import (
    Obligation,
    group_by_priority_deadline,
    order_obligations,
)

__all__ = [
    "RecoveryError",
    "EscalationPolicy",
    "RecoveryPolicy",
    "Escalation",
    "StageSolve",
    "ObligationProgress",
    "RecoveryResult",
    "StageProgram",
    "recovery_entry",
    "run_recovery",
    "no_action",
    "finalize_batch",
    "PROGRAM_BOUND_FRACTION",
    "ON_INFEASIBLE",
    "ON_SIGN_CONTRADICTION",
]


class RecoveryError(ValueError):
    """A recovery declaration or transition that must not be silently repaired."""


ON_INFEASIBLE: tuple[str, ...] = ("cancel", "hold", "escalate")
ON_SIGN_CONTRADICTION: tuple[str, ...] = ("hold", "unwind")
_ON_BLOCKED_OBLIGATION: tuple[str, ...] = ("record", "escalate")
_ON_DEADLINE_UNDER_KILL_SWITCH: tuple[str, ...] = ("record", "escalate")
_ON_ROUNDING_FAILURE: tuple[str, ...] = ("scale_batch", "drop_batch")
_ON_PARTIAL_BATCH: tuple[str, ...] = ("cancel_accepted", "hold_and_escalate")
_PREFERENCE_RESPONSE: tuple[str, ...] = ("record", "block")

_ENUMS: Mapping[str, tuple[str, ...]] = {
    "on_infeasible": ON_INFEASIBLE,
    "on_sign_contradiction": ON_SIGN_CONTRADICTION,
    "on_blocked_obligation": _ON_BLOCKED_OBLIGATION,
    "on_deadline_under_kill_switch": _ON_DEADLINE_UNDER_KILL_SWITCH,
    "on_rounding_failure": _ON_ROUNDING_FAILURE,
    "on_partial_batch": _ON_PARTIAL_BATCH,
    "preference_response": _PREFERENCE_RESPONSE,
}

_POLICY_KEYS = frozenset(_ENUMS) | {"bound_slack", "recovery_entry_margin", "component_liquidate_priority"}

#: The fraction of the declared ``bound_slack`` the *program* is allowed to spend. The
#: declared slack is what the independent validation grants for solver error
#: (``portfolio/validate.py``: a violation beyond a bound R1 achieved plus ``bound_slack``
#: fails); if the program were written at the full slack, a solution sitting on that bound
#: to within the solver's own tolerance would cross the validated threshold and send a
#: sound recovery to the no-action path. The stages therefore keep half of it in reserve,
#: which leaves the sequence strictly feasible and the guarantee checkable.
PROGRAM_BOUND_FRACTION: float = 0.5


@dataclass(frozen=True)
class EscalationPolicy:
    """The ``[escalation]`` block: what an ``escalate`` response actually does."""

    write_alert: bool = True
    exit_code: int = 3
    require_acknowledgement: bool = True

    def to_dict(self) -> dict[str, Any]:
        """The record shape."""
        return {
            "write_alert": bool(self.write_alert),
            "exit_code": int(self.exit_code),
            "require_acknowledgement": bool(self.require_acknowledgement),
        }


@dataclass(frozen=True)
class RecoveryPolicy:
    """``policies/recovery.toml`` (spec §3.7), so the safety-critical branches are values.

    Every response key is a closed enumeration and an unknown spelling is refused at load,
    not defaulted: a mis-spelled ``on_infeasible`` that silently kept ``cancel`` would
    decide, unrecorded, what happens on the one path where nothing is executable.
    """

    on_infeasible: Literal["cancel", "hold", "escalate"] = "cancel"
    on_sign_contradiction: Literal["hold", "unwind"] = "hold"
    on_blocked_obligation: Literal["record", "escalate"] = "escalate"
    on_deadline_under_kill_switch: Literal["record", "escalate"] = "escalate"
    on_rounding_failure: Literal["scale_batch", "drop_batch"] = "scale_batch"
    on_partial_batch: Literal["cancel_accepted", "hold_and_escalate"] = "cancel_accepted"
    preference_response: Literal["record", "block"] = "record"
    bound_slack: float = 1.0e-6
    recovery_entry_margin: float = 0.002
    component_liquidate_priority: int = 1
    escalation: EscalationPolicy = field(default_factory=EscalationPolicy)
    source: str = ""

    def __post_init__(self) -> None:
        for key, allowed in _ENUMS.items():
            value = getattr(self, key)
            if value not in allowed:
                raise RecoveryError(f"{key} is one of {allowed}, got {value!r}")
        for key in ("bound_slack", "recovery_entry_margin"):
            value = float(getattr(self, key))
            if not math.isfinite(value) or value < 0.0:
                raise RecoveryError(f"{key} must be finite and >= 0, got {getattr(self, key)!r}")
            object.__setattr__(self, key, value)
        if not isinstance(self.component_liquidate_priority, int):
            raise RecoveryError(
                f"component_liquidate_priority must be an integer, got {self.component_liquidate_priority!r}"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, source: str = "") -> "RecoveryPolicy":
        """Parse the file's contents. Unknown keys are refused, never ignored."""
        table = dict(mapping)
        escalation_table = table.pop("escalation", {}) or {}
        if not isinstance(escalation_table, Mapping):
            raise RecoveryError(f"{source or 'recovery policy'}: [escalation] must be a table")
        unknown_escalation = set(escalation_table) - {"write_alert", "exit_code", "require_acknowledgement"}
        if unknown_escalation:
            raise RecoveryError(
                f"{source or 'recovery policy'}: unknown [escalation] keys {sorted(unknown_escalation)}"
            )
        unknown = set(table) - _POLICY_KEYS
        if unknown:
            raise RecoveryError(
                f"{source or 'recovery policy'}: unknown keys {sorted(unknown)}; declared keys are "
                f"{sorted(_POLICY_KEYS)} plus the [escalation] table"
            )
        kwargs: dict[str, Any] = {key: table[key] for key in table}
        return cls(
            **kwargs,
            escalation=EscalationPolicy(
                write_alert=bool(escalation_table.get("write_alert", True)),
                exit_code=int(escalation_table.get("exit_code", 3)),
                require_acknowledgement=bool(escalation_table.get("require_acknowledgement", True)),
            ),
            source=source,
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "RecoveryPolicy":
        """Read ``policies/recovery.toml``."""
        location = Path(path)
        with location.open("rb") as handle:
            table = tomllib.load(handle)
        return cls.from_mapping(table, source=str(location))

    def declaration(self) -> dict[str, Any]:
        """The declared content, for the result record and for ``config_id``."""
        out: dict[str, Any] = {key: getattr(self, key) for key in sorted(_ENUMS)}
        out["bound_slack"] = float(self.bound_slack)
        out["recovery_entry_margin"] = float(self.recovery_entry_margin)
        out["component_liquidate_priority"] = int(self.component_liquidate_priority)
        out["escalation"] = self.escalation.to_dict()
        return out


@dataclass(frozen=True)
class Escalation:
    """One escalation the caller must act on: alert, exit code, acknowledgement gate."""

    reason: str
    kind: str
    write_alert: bool
    exit_code: int
    require_acknowledgement: bool

    def to_dict(self) -> dict[str, Any]:
        """The record shape."""
        return {
            "reason": self.reason,
            "kind": self.kind,
            "write_alert": bool(self.write_alert),
            "exit_code": int(self.exit_code),
            "require_acknowledgement": bool(self.require_acknowledgement),
        }


@dataclass(frozen=True)
class StageSolve:
    """One solved (or failed) stage of the sequence, as the record carries it."""

    stage: str
    subject: str
    status: str
    optimal: bool
    value: float | None
    solve_seconds: float
    compile_seconds: float
    iterations: int | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The record shape."""
        return {
            "stage": self.stage,
            "subject": self.subject,
            "status": self.status,
            "optimal": bool(self.optimal),
            "value": None if self.value is None else float(self.value),
            "solve_seconds": float(self.solve_seconds),
            "compile_seconds": float(self.compile_seconds),
            "iterations": self.iterations,
            "note": self.note,
        }


@dataclass(frozen=True)
class ObligationProgress:
    """What R0 achieved on one obligation, in shares, and whether it is blocked."""

    sid: str
    priority: int
    deadline: str | None
    remaining_qty: float
    achieved_qty: float
    blocked: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """The record shape."""
        return {
            "sid": self.sid,
            "priority": int(self.priority),
            "deadline": self.deadline,
            "remaining_qty": float(self.remaining_qty),
            "achieved_qty": float(self.achieved_qty),
            "blocked": bool(self.blocked),
            "reason": self.reason,
        }


@dataclass
class RecoveryResult:
    """The outcome of the three-stage sequence."""

    w: pd.Series | None
    stages: tuple[StageSolve, ...]
    achieved_bounds: dict[str, float]
    obligations: tuple[ObligationProgress, ...]
    escalations: tuple[Escalation, ...]
    binding_condition: str | None
    failed_stage: str | None
    attribution: dict[str, str]

    @property
    def ok(self) -> bool:
        """True when every stage solved and a target came out of R2."""
        return self.failed_stage is None and self.w is not None

    def to_dict(self) -> dict[str, Any]:
        """The record shape: per-stage attribution, achieved bounds, obligation progress."""
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "achieved_bounds": {name: float(value) for name, value in self.achieved_bounds.items()},
            "obligations": [progress.to_dict() for progress in self.obligations],
            "escalations": [escalation.to_dict() for escalation in self.escalations],
            "binding_condition": self.binding_condition,
            "failed_stage": self.failed_stage,
            "attribution": dict(self.attribution),
        }


class StageProgram(Protocol):
    """What :func:`run_recovery` needs from a convex program.

    Declared structurally so this module depends on no solver: ``portfolio/convex.py``
    implements it, and a test double implements it in a few lines.
    """

    sids: tuple[str, ...]

    def action_constraints(self) -> list[Any]:  # pragma: no cover - protocol declaration
        ...

    def violation_expr(self, limit: Limit) -> Any:  # pragma: no cover - protocol declaration
        ...

    def obligation_progress_expr(
        self, obligation: Obligation, current_qty: float
    ) -> tuple[Any, float, str | None]:  # pragma: no cover - protocol declaration
        ...

    def economic_objective(self) -> Any:  # pragma: no cover - protocol declaration
        ...

    def solve(
        self, objective: Any, constraints: Sequence[Any], *, sense: str, stage: str, subject: str
    ) -> Any:  # pragma: no cover - protocol declaration
        ...


def recovery_entry(
    limits: Iterable[Limit],
    ctx: LimitContext,
    obligations: Sequence[Obligation] = (),
    *,
    limit_tolerance: float,
    recovery_entry_margin: float,
) -> tuple[bool, str | None]:
    """``(enters_recovery, reason)`` for the marked state (spec §3.7).

    Recovery is entered when an obligation exists, or when the marked state breaches a
    holding-state limit by more than ``limit_tolerance + recovery_entry_margin``. The
    margin exists so a marginal drift over a preference limit does not flip the path with
    no declared band: a profile whose ``max_gross`` equals the gross its construction
    targets would otherwise enter recovery on roughly half of decision bars.
    """
    live = [obligation for obligation in obligations if obligation.status != "done" and obligation.remaining_qty > 0.0]
    if live:
        first = live[0]
        return True, f"obligation {first.sid} ({first.kind}, priority {first.priority})"
    for limit in reduction_order(limits):
        if limit.name not in MEASURED_LIMITS:
            continue
        measure = violation(limit, ctx.w0, ctx)
        threshold = limit.effective_tolerance(limit_tolerance) + float(recovery_entry_margin)
        if measure > threshold:
            return True, f"{limit.name} breached by the marked state at v={measure:.6g} > {threshold:.6g}"
    return False, None


def run_recovery(
    program: StageProgram,
    *,
    limits: Sequence[Limit],
    obligations: Sequence[Obligation],
    policy: RecoveryPolicy,
    positions: Mapping[str, float] | pd.Series,
    reference_prices: Mapping[str, float] | pd.Series | None = None,
    limit_tolerance: float = 1.0e-6,
    ctx: LimitContext | None = None,
) -> RecoveryResult:
    """Drive R0, R1 and R2 in order and return what each achieved (spec §3.7).

    The stages share one variable and one set of action restrictions; what changes between
    them is the objective and the accumulated bounds. A stage that does not solve stops the
    sequence and the result carries ``failed_stage``, which puts the decision on the
    no-action path -- never on a silently truncated recovery.
    """
    slack = float(policy.bound_slack) * PROGRAM_BOUND_FRACTION
    action = list(program.action_constraints())
    stages: list[StageSolve] = []
    escalations: list[Escalation] = []
    attribution: dict[str, str] = {}
    progress_records: list[ObligationProgress] = []
    positions = pd.Series(positions, dtype=float)
    marked: dict[str, float] = {}
    if ctx is not None:
        for limit in reduction_order(limits):
            if limit.name in MEASURED_LIMITS:
                marked[limit.name] = violation(limit, ctx.w0, ctx)
    w_after_r0: pd.Series | None = None

    # -- R0: obligations, one (priority, deadline) group at a time ----------
    progress_bounds: list[Any] = []
    live = [obligation for obligation in obligations if obligation.status != "done" and obligation.remaining_qty > 0.0]
    ordered = order_obligations(live, reference_prices)
    for key, group in group_by_priority_deadline(list(ordered)):
        terms: list[Any] = []
        members: list[tuple[Obligation, Any, float]] = []
        for obligation in group:
            current = float(positions.get(obligation.sid, 0.0))
            expr, remaining, reason = program.obligation_progress_expr(obligation, current)
            if expr is None and reason is None:
                # Nothing left to do: a broker-initiated fill already moved the position to
                # its target. Satisfied, not blocked, and never an escalation.
                progress_records.append(
                    ObligationProgress(
                        sid=obligation.sid,
                        priority=int(obligation.priority),
                        deadline=obligation.deadline,
                        remaining_qty=0.0,
                        achieved_qty=0.0,
                        blocked=False,
                        reason="already at or past the obligation target before this decision",
                    )
                )
                continue
            if expr is None:
                progress_records.append(
                    ObligationProgress(
                        sid=obligation.sid,
                        priority=int(obligation.priority),
                        deadline=obligation.deadline,
                        remaining_qty=float(obligation.remaining_qty),
                        achieved_qty=0.0,
                        blocked=True,
                        reason=reason or "the obligation is not expressible against this decision bar",
                    )
                )
                if policy.on_blocked_obligation == "escalate":
                    escalations.append(
                        Escalation(
                            reason=f"{obligation.sid}: {reason or 'obligation not expressible'}",
                            kind="blocked_obligation",
                            write_alert=policy.escalation.write_alert,
                            exit_code=policy.escalation.exit_code,
                            require_acknowledgement=policy.escalation.require_acknowledgement,
                        )
                    )
                continue
            terms.append(expr / max(remaining, 1e-12))
            members.append((obligation, expr, remaining))
        if not members:
            continue
        objective = terms[0]
        for term in terms[1:]:
            objective = objective + term
        subject = f"priority {key[0]}, deadline {key[1]}"
        # Progress is never negative. Without this, a coupling action restriction (a buying
        # power bound, say) would let the group's total be maximized by moving one name
        # *away* from its target to make room for another, which is not "progress".
        floors = [expr >= 0.0 for _, expr, _ in members]
        solved = program.solve(
            objective, action + progress_bounds + floors, sense="maximize", stage="R0", subject=subject
        )
        stages.append(_stage_solve(solved, "R0", subject))
        if not solved.optimal or solved.w is None:
            return RecoveryResult(
                w=None,
                stages=tuple(stages),
                achieved_bounds={},
                obligations=tuple(progress_records),
                escalations=tuple(escalations),
                binding_condition=f"R0 {subject}",
                failed_stage="R0",
                attribution=attribution,
            )
        w_after_r0 = solved.w
        for obligation, expr, remaining in members:
            achieved = float(expr.value)
            # The achieved value is fixed as a lower bound for every later group and for
            # R1 and R2: no preference limit ever reduces obligation progress.
            progress_bounds.append(expr >= achieved - slack)
            blocked = achieved <= limit_tolerance
            reason = "" if not blocked else "the action restrictions allow no progress in this name this bar"
            progress_records.append(
                ObligationProgress(
                    sid=obligation.sid,
                    priority=int(obligation.priority),
                    deadline=obligation.deadline,
                    remaining_qty=float(remaining),
                    achieved_qty=achieved,
                    blocked=blocked,
                    reason=reason,
                )
            )
            if blocked and policy.on_blocked_obligation == "escalate":
                escalations.append(
                    Escalation(
                        reason=f"{obligation.sid}: {reason}",
                        kind="blocked_obligation",
                        write_alert=policy.escalation.write_alert,
                        exit_code=policy.escalation.exit_code,
                        require_acknowledgement=policy.escalation.require_acknowledgement,
                    )
                )
            if achieved > limit_tolerance:
                attribution[f"obligation:{obligation.sid}"] = "R0"

    # Obligation progress outranks every preference limit, so a violation R0 introduced is
    # attributed to R0 and named with its obligation; R1 minimizes it from there but may
    # never undo the progress.
    if ctx is not None and w_after_r0 is not None:
        for limit in reduction_order(limits):
            if limit.name not in MEASURED_LIMITS:
                continue
            tolerance = limit.effective_tolerance(limit_tolerance)
            after = violation(limit, w_after_r0, ctx)
            if after > tolerance and marked.get(limit.name, 0.0) <= tolerance:
                attribution[limit.name] = "R0"

    # -- R1: every holding-state limit, in reduction order ------------------
    achieved_bounds: dict[str, float] = {}
    bound_constraints: list[Any] = []
    for limit in reduction_order(limits):
        if limit.name not in MEASURED_LIMITS:
            continue
        expr = program.violation_expr(limit)
        subject = limit.name
        solved = program.solve(
            expr,
            action + progress_bounds + bound_constraints,
            sense="minimize",
            stage="R1",
            subject=subject,
        )
        stages.append(_stage_solve(solved, "R1", subject))
        if not solved.optimal or solved.w is None:
            return RecoveryResult(
                w=None,
                stages=tuple(stages),
                achieved_bounds=achieved_bounds,
                obligations=tuple(progress_records),
                escalations=tuple(escalations),
                binding_condition=subject,
                failed_stage="R1",
                attribution=attribution,
            )
        achieved = max(0.0, float(solved.value if solved.value is not None else expr.value))
        achieved_bounds[limit.name] = achieved
        bound_constraints.append(expr <= achieved + slack)
        if achieved > limit.effective_tolerance(limit_tolerance):
            attribution.setdefault(limit.name, "R1")

    # -- R2: the economic objective inside what R1 achieved -----------------
    solved = program.solve(
        program.economic_objective(),
        action + progress_bounds + bound_constraints,
        sense="maximize",
        stage="R2",
        subject="economic_objective",
    )
    stages.append(_stage_solve(solved, "R2", "economic_objective"))
    if not solved.optimal or solved.w is None:
        return RecoveryResult(
            w=None,
            stages=tuple(stages),
            achieved_bounds=achieved_bounds,
            obligations=tuple(progress_records),
            escalations=tuple(escalations),
            binding_condition="economic_objective",
            failed_stage="R2",
            attribution=attribution,
        )
    binding = None
    introduced_by_r0 = sorted(name for name, stage in attribution.items() if stage == "R0")
    made_progress = [record for record in progress_records if not record.blocked]
    if introduced_by_r0 and made_progress:
        # "That violation is recorded as introduced by R0 with the obligation named in
        # binding_condition" -- the obligation, not the limit, is what determined it.
        binding = f"obligation:{made_progress[0].sid}"
    elif achieved_bounds:
        worst = max(achieved_bounds.items(), key=lambda item: (item[1], item[0]))
        binding = worst[0] if worst[1] > limit_tolerance else None
    if binding is None and progress_records:
        binding = f"obligation:{progress_records[0].sid}"
    return RecoveryResult(
        w=solved.w,
        stages=tuple(stages),
        achieved_bounds=achieved_bounds,
        obligations=tuple(progress_records),
        escalations=tuple(escalations),
        binding_condition=binding,
        failed_stage=None,
        attribution=attribution,
    )


def _stage_solve(solved: Any, stage: str, subject: str) -> StageSolve:
    return StageSolve(
        stage=stage,
        subject=subject,
        status=str(solved.status),
        optimal=bool(solved.optimal),
        value=None if solved.value is None else float(solved.value),
        solve_seconds=float(solved.solve_seconds),
        compile_seconds=float(solved.compile_seconds),
        iterations=solved.iterations,
        note=solved.note,
    )


def no_action(policy: RecoveryPolicy, reason: str, *, stage: str = "construction") -> dict[str, Any]:
    """The declared ``on_infeasible`` response, with no orders (spec §3.7).

    Returns the response and, when it is ``escalate``, the escalation the caller must act
    on: no orders are submitted, the reason is recorded with the limit or obligation that
    caused it, an alert is written through ``ops/_alert.sh::alert``, and the command exits
    with the declared code. This function decides *what* the response is; writing the alert
    file and exiting belong to the run driver, which owns the process.

    Neither no-trade nor the zero portfolio is assumed executable here. The caller carries
    the marked state forward because no order was submitted, not because holding it was
    validated as an action.
    """
    record: dict[str, Any] = {
        "path": "no_action",
        "response": policy.on_infeasible,
        "reason": reason,
        "stage": stage,
        "orders_submitted": False,
        "assumes_no_trade_executable": False,
    }
    if policy.on_infeasible == "escalate":
        record["escalation"] = Escalation(
            reason=reason,
            kind="infeasible",
            write_alert=policy.escalation.write_alert,
            exit_code=policy.escalation.exit_code,
            require_acknowledgement=policy.escalation.require_acknowledgement,
        ).to_dict()
    return record


def finalize_batch(
    output: Any,
    *,
    state: Any,
    prices: pd.Series,
    execution_profile: Any,
    limits: Sequence[Limit],
    policy: RecoveryPolicy,
    decision_bar: str,
    obligations: Sequence[Obligation] = (),
    capabilities: Any = None,
    adv_dollars: pd.Series | None = None,
    risk: Any = None,
    cost_model: Any = None,
    validation: Any = None,
    executable: Mapping[str, Any] | None = None,
) -> Any:
    """Round a construction output into an order batch, revalidating the whole batch.

    This is the second of the two independent validations §3.7 requires: ``construct``
    validates the solved target, and ``round_batch`` validates the portfolio the rounded
    batch implies. The bounds R1 achieved are carried across, so a rounding repair that
    increased a violation beyond what recovery achieved fails there too.

    A construction output whose own post-solve validation blocked is refused rather than
    rounded: solver status never authorizes an order, and neither does a validation the
    engine already refused.
    """
    from prism.portfolio.rounding import (
        round_batch,  # local: rounding pulls the whole batch stack
    )

    if output.path == "no_action":
        raise RecoveryError(
            "the decision is on the no-action path; no orders are submitted, so there is no batch to round "
            f"(response {policy.on_infeasible!r}, reason {output.diagnostics.get('no_action', {}).get('reason')!r})"
        )
    report = output.diagnostics.get("validation")
    if isinstance(report, Mapping) and not report.get("ok", True):
        raise RecoveryError(
            "the solved target failed independent post-solve validation; a batch is never rounded from it "
            f"(binding condition {report.get('binding_condition')!r})"
        )
    return round_batch(
        output.w_target,
        state,
        prices,
        execution_profile,
        limits,
        obligations,
        capabilities,
        decision_bar=decision_bar,
        validation=validation,
        adv_dollars=adv_dollars,
        risk=risk,
        cost_model=cost_model,
        executable=dict(executable or {}),
        achieved_bounds=output.diagnostics.get("achieved_bounds") or None,
        on_rounding_failure=policy.on_rounding_failure,
        on_blocked_obligation=policy.on_blocked_obligation,
    )
