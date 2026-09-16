"""Flow F1, position-space combination (engine specification §3.3).

Components do not vote on a score and they do not share a cross-section: each
one keeps its own *attributed sub-target* per security, and the account target
is the weighted sum of those sub-targets. The rank-to-weight rule runs first,
over the whole cross-section, and every state rule below acts on the resulting
weight vector rather than on the raw score. That ordering is the point of the
flow: at score level a zero is mid-rank rather than flat, and an ineligible NaN
score would hold a stale position instead of exiting.

The state machine, at decision bar *t*, per component and per security:

======================================  ==========================================
Condition                               Sub-target
======================================  ==========================================
not due, output unexpired               carried forward; the component is not scored
due, rule value finite                  replaced by the rule value
due, rule value NaN (no opinion)        prior sub-target held
due, rule value 0.0                     explicit flat
``liquidate`` instruction for that sid  explicit flat, and band/dust exempt
output expired                          set to zero and recorded, for every sid
======================================  ==========================================

An expired output is an explicit **zero**, never "no opinion": the two states
have opposite consequences and the term is not reused here. A late refresh is
not expiry — the grace is ``valid_for_bars`` — and expiry never scales an
order, it only flattens. Opposing sub-targets net, and with ``renormalize =
false`` (the default) a component that drops out reduces the account's target
gross by its own contribution: cash absorbs the difference, no surviving
component is scaled up to replace it.

What this module does **not** do: it never fetches the unrankable mask (the
caller supplies it, §3.8), it never fits (``assemble``'s caller does, §3.8),
and it never caps or bands — the constructor of §3.7 does that to the account
target this flow returns.

``ComponentState`` is the §3.11 state record. It lives here until ``LoopState``
reaches schema version 3, because persisting it under schema 2 — the version
the live book runs on — would change what a running loop reads back.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.signal.base import Signal
from prism.spec import TO_WEIGHTS_RULES, apply_to_weights, resolve_to_weights_params

logger = logging.getLogger(__name__)

#: The combination this module implements. ``calibrated_convex`` (flow F2) does
#: not pass through position-space combination at all (§3.3).
COMBINATION = "position_space"

#: Component output kinds flow F1 supports (§3.3). ``standardized_forecast`` is
#: an F2 input and is refused here.
SUPPORTED_OUTPUT_KINDS: tuple[str, ...] = ("rank_score", "target_weights")

#: The ``disabled_reason`` enumeration of §3.8. ``sign_contradiction`` is set by
#: the sign policy of §3.4 and ``missing_inputs`` by the risk/calibration inputs
#: of flow F2; neither exists at this stage, so this flow sets only ``expired``
#: and ``unrankable_coverage``.
DISABLED_REASONS: tuple[str, ...] = (
    "sign_contradiction",
    "expired",
    "missing_inputs",
    "unrankable_coverage",
)

#: Per-sid marks a component may emit from ``Signal.instructions`` (§3.3).
INSTRUCTIONS: tuple[str, ...] = ("liquidate", "no_decision")

_EMPTY_PARAMS: Mapping[str, Any] = MappingProxyType({})


class CompositionError(ValueError):
    """A composition input is unsupported, inconsistent, or self-contradictory."""


@dataclass(frozen=True)
class ComponentState:
    """One component's persisted composition state (§3.11's declaration).

    ``last_output_bar`` is the last bar an output was produced — an entirely
    NaN output counts, because it advances the anchor and restarts the validity
    clock — and it anchors both the update schedule and expiry.
    ``decision_bar`` is the last bar an output actually changed a sub-target.
    ``sub_targets`` never stores NaN: absence is "no sub-target", which is a
    different state from an explicit 0.0.

    ``expires_bar`` is a label resolved on the supplied decision calendar and
    is None when that calendar does not reach ``valid_for_bars`` past the
    decision (the live loop's calendar ends at the decision bar). Expiry itself
    is decided by counting elapsed bars, never by reading this label back.
    """

    component_id: str
    last_output_bar: str | None = None
    decision_bar: str | None = None
    expires_bar: str | None = None
    expired: bool = False
    sub_targets: Mapping[str, float] = field(default_factory=dict)
    fitted_state_ref: str | None = None
    fitted_through_bar: str | None = None
    fit_scope: str = "trailing_window"


@dataclass(frozen=True)
class ComponentPlan:
    """A component paired with the composition policy declared for it.

    ``weights_fn`` is the seam the heuristic live books need: the four
    rank-to-weight rules of §3.3 are named by ``to_weights`` and applied by
    ``prism.spec.apply_to_weights``, but the recorded directional / decile /
    inverse-vol books are constructed by ``prism.assembly`` against the whole
    close panel and a regime-scaled cap, which no rule name can carry. When
    ``weights_fn`` is given it *is* the rule, and ``to_weights`` is the name
    recorded for it.
    """

    component: Signal
    component_id: str
    weight: float = 1.0
    to_weights: str = "none"
    to_weights_params: Mapping[str, Any] = _EMPTY_PARAMS
    update_every_bars: int = 1
    valid_for_bars: int = 1
    output_kind: str = "rank_score"
    scoring_basis: str = "price"
    weights_fn: Callable[[pd.Series], pd.Series] | None = None
    #: The flow this plan is composed under (§3.3). Under ``calibrated_convex`` (F2) the
    #: component's score is an input to the calibration and the constructor, so there is
    #: no rank-to-weight rule to declare and ``to_weights = "none"`` is the *required*
    #: value rather than a missing one. Everything else about the plan -- the refresh
    #: cadence, the expiry, the mask handling, the state machine -- is the same in both
    #: flows, which is why one plan type serves both.
    combination: str = COMBINATION

    def __post_init__(self) -> None:
        if not str(self.component_id):
            raise CompositionError("a component plan needs a non-empty component_id")
        where = f"component {self.component_id!r}"
        if not math.isfinite(float(self.weight)):
            raise CompositionError(f"{where}: weight must be finite, got {self.weight!r}")
        if int(self.update_every_bars) < 1:
            raise CompositionError(f"{where}: update_every_bars must be >= 1, got {self.update_every_bars!r}")
        if int(self.valid_for_bars) < int(self.update_every_bars):
            # §3.3: an output that expires before its own next refresh has no
            # reachable valid state, so the configuration is refused.
            raise CompositionError(
                f"{where}: valid_for_bars {self.valid_for_bars} < update_every_bars "
                f"{self.update_every_bars}; an output that expires before its own next refresh "
                "has no reachable valid state"
            )
        if self.output_kind not in SUPPORTED_OUTPUT_KINDS:
            raise CompositionError(
                f"{where}: flow F1 supports output_kind {SUPPORTED_OUTPUT_KINDS}, not "
                f"{self.output_kind!r}; a standardized_forecast is an input to flow F2 "
                "(calibrated_convex), which does not pass through position-space combination"
            )
        if self.weights_fn is None:
            if self.to_weights not in TO_WEIGHTS_RULES:
                raise CompositionError(
                    f"{where}: unknown to_weights rule {self.to_weights!r}; permitted rules are "
                    f"{', '.join(TO_WEIGHTS_RULES)} (or supply weights_fn)"
                )
            if self.output_kind == "target_weights" and self.to_weights != "none":
                raise CompositionError(
                    f"{where}: a target_weights component emits an account-relative weight "
                    f"vector already, so it requires to_weights='none', not {self.to_weights!r}"
                )
            if self.output_kind == "rank_score" and self.to_weights == "none" and self.combination == COMBINATION:
                raise CompositionError(
                    f"{where}: a rank_score component must declare a rank-to-weight rule; "
                    "to_weights='none' leaves a rank in weight space, which is not a position"
                )
            if self.output_kind == "rank_score" and self.to_weights != "none" and self.combination != COMBINATION:
                raise CompositionError(
                    f"{where}: combination={self.combination!r} (flow F2) feeds the component's score to the "
                    f"calibration and the constructor, so to_weights must be 'none', not {self.to_weights!r}; a "
                    "rank-to-weight rule would map the scores to weights before the constructor ever saw them"
                )
            # Validate and default the parameters now: a rule whose caps are
            # only checked at the first decision fails in operation, not at load.
            resolve_to_weights_params(self.to_weights, self.to_weights_params, where=where)

    @classmethod
    def from_loaded(
        cls,
        loaded: Any,
        *,
        weights_fn: Callable[[pd.Series], pd.Series] | None = None,
        combination: str = COMBINATION,
    ) -> "ComponentPlan":
        """Build a plan from a ``prism.spec.LoadedComponent``."""
        spec = loaded.spec
        declarations = loaded.declarations
        return cls(
            component=loaded.component,
            component_id=loaded.component_id,
            weight=float(spec.weight),
            to_weights=str(spec.to_weights),
            to_weights_params=dict(spec.to_weights_params),
            update_every_bars=int(spec.update_every_bars),
            valid_for_bars=int(spec.valid_for_bars),
            output_kind=str(declarations["output_kind"]),
            scoring_basis=str(declarations["scoring_basis"]),
            combination=str(combination),
            weights_fn=weights_fn,
        )


@dataclass(frozen=True)
class ComponentOutput:
    """What one component contributed to one decision (§3.8, recorded per §3.3).

    ``raw`` is the component's own output in its own units, ``None`` when the
    component was not due and therefore not scored. ``sub_target`` is the
    attributed sub-target after the state rules — NaN for a sid carrying no
    decision at all, which is not the same as an explicit 0.0.

    ``disabled_reason`` is ``"expired"`` whenever the state entering this bar
    had expired, whether or not a fresh output also arrived: it is the record
    of why the carried sub-targets were zeroed. ``due`` and ``raw`` say
    separately whether the component produced anything at this bar.
    """

    component_id: str
    raw: pd.Series | None
    standardized: pd.Series | None
    sub_target: pd.Series | None
    due: bool
    expired: bool
    disabled_reason: str | None
    decision_bar: str | None = None
    expires_bar: str | None = None
    instructions: pd.Series | None = None
    masked: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.disabled_reason is not None and self.disabled_reason not in DISABLED_REASONS:
            raise CompositionError(
                f"component {self.component_id}: disabled_reason {self.disabled_reason!r} is not one "
                f"of the declared enumeration {DISABLED_REASONS} (§3.8)"
            )


@dataclass(frozen=True)
class CompositionResult:
    """The account target and everything the record needs about how it was made."""

    targets: pd.Series
    component_outputs: tuple[ComponentOutput, ...]
    states: Mapping[str, ComponentState]
    renormalization: float
    liquidate: frozenset[str]
    warnings: tuple[str, ...]


def compose(
    plans: Sequence[ComponentPlan],
    *,
    decision_bar: str,
    scoring_panels: Mapping[str, pd.DataFrame],
    sessions: Sequence[str] | None = None,
    volume: pd.DataFrame | None = None,
    states: Mapping[str, ComponentState] | None = None,
    unrankable: Collection[str] = (),
    ineligible: Collection[str] = (),
    renormalize: bool = False,
) -> CompositionResult:
    """Combine components in position space at ``decision_bar`` (flow F1).

    ``scoring_panels`` maps a declared ``scoring_basis`` to the close panel the
    engine supplies for it; each component scores on the panel its declaration
    names, and the panel's last row is the decision bar (N1). ``sessions`` is
    the decision calendar used to count elapsed bars for due-ness and expiry;
    it defaults to the price panel's own index, which is the calendar the live
    loop already holds.

    ``states`` carries each component's persisted ``ComponentState``; absent
    state means the component has never produced an output, which makes it due
    and gives it no sub-target to carry. ``unrankable`` is the supplied mask
    (§3.8): a masked sid's score becomes NaN *before* the rank-to-weight rule,
    so it cannot distort cross-sectional membership, and is marked no-decision
    *after* it, so a held name is held rather than pinned flat.

    ``ineligible`` is the strategy screen's failures (§3.3, ``[strategy.eligibility]``).
    It differs from ``unrankable`` in exactly one place and the difference is the
    whole point: an ineligible sid's score also goes NaN before the rule, but it
    is **not** marked no-decision, so the rank-to-weight rule assigns it an
    explicit flat and a held name that has just failed the screen is exited
    rather than frozen at its last weight. Dropping such a sid from the scored
    panel instead would freeze it, which is why the screen is a mask here and
    never a column filter.

    Returns the account target (NaN where no component holds a sub-target for a
    sid), the per-component record, and the state to persist.
    """
    if not plans:
        raise CompositionError("no components: there is nothing to combine (§3.3)")
    ids = [plan.component_id for plan in plans]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicates:
        raise CompositionError(
            f"duplicate component id(s) {duplicates}: a sub-target is attributed per component, "
            "so ids must be unique"
        )
    calendar = _calendar(sessions, scoring_panels, plans)
    masked_input = {str(sid) for sid in unrankable}
    screened_out = {str(sid) for sid in ineligible} - masked_input
    prior = dict(states or {})

    outputs: list[ComponentOutput] = []
    new_states: dict[str, ComponentState] = {}
    warnings: list[str] = []
    liquidate: set[str] = set()

    for plan in plans:
        state = prior.get(plan.component_id, ComponentState(component_id=plan.component_id))
        output, new_state, flat = _compose_one(
            plan,
            state,
            decision_bar=decision_bar,
            calendar=calendar,
            scoring_panels=scoring_panels,
            volume=volume,
            masked=masked_input,
            screened=screened_out,
            warnings=warnings,
        )
        outputs.append(output)
        new_states[plan.component_id] = new_state
        liquidate |= flat

    targets, scale = _account_target(plans, outputs, new_states, renormalize=renormalize)
    return CompositionResult(
        targets=targets,
        component_outputs=tuple(outputs),
        states=new_states,
        renormalization=scale,
        liquidate=frozenset(liquidate),
        warnings=tuple(warnings),
    )


def _compose_one(
    plan: ComponentPlan,
    state: ComponentState,
    *,
    decision_bar: str,
    calendar: Sequence[str],
    scoring_panels: Mapping[str, pd.DataFrame],
    volume: pd.DataFrame | None,
    masked: Collection[str],
    warnings: list[str],
    screened: Collection[str] = (),
) -> tuple[ComponentOutput, ComponentState, set[str]]:
    """One component's sub-target vector at ``decision_bar``, and its new state."""
    elapsed = _bars_elapsed(calendar, state.last_output_bar, decision_bar)
    due = state.last_output_bar is None or elapsed >= int(plan.update_every_bars)
    expired = state.last_output_bar is not None and elapsed > int(plan.valid_for_bars)

    sub_targets = dict(state.sub_targets)
    if expired and sub_targets:
        # An expired output is an explicit zero for every sid the component
        # holds, never "no opinion": a stale forecast de-risks the book (a C1
        # reduction, §3.10) rather than freezing it.
        sub_targets = {sid: 0.0 for sid in sub_targets}
        logger.warning(
            "component %s expired at %s: last output %s is %d bar(s) old > valid_for_bars %d; "
            "its %d sub-target(s) go flat",
            plan.component_id,
            decision_bar,
            state.last_output_bar,
            elapsed,
            int(plan.valid_for_bars),
            len(sub_targets),
        )

    if not due:
        # Not due and unexpired: carried forward unchanged, and NOT scored.
        carried = pd.Series(sub_targets, dtype=float)
        output = ComponentOutput(
            component_id=plan.component_id,
            raw=None,
            standardized=None,
            sub_target=carried if len(carried) else None,
            due=False,
            expired=expired,
            disabled_reason="expired" if expired else None,
            decision_bar=state.decision_bar,
            expires_bar=state.expires_bar,
        )
        new_state = replace(state, expired=expired, sub_targets=dict(sub_targets))
        return output, new_state, set()

    close = _panel_for(plan, scoring_panels)
    raw = _score(plan, close, volume)
    applied_mask = tuple(sid for sid in raw.index if sid in masked)
    applied_screen = tuple(sid for sid in raw.index if sid in screened)
    if applied_mask or applied_screen:
        raw = raw.copy()
    if applied_mask:
        raw.loc[list(applied_mask)] = np.nan
    if applied_screen:
        # No opinion before the rule, explicit flat after it: the screen's
        # failures are exited, not held (§3.3).
        raw.loc[list(applied_screen)] = np.nan

    instructions = _instructions(plan, close, volume)
    no_decision = set(applied_mask)
    flat: set[str] = set()
    if instructions is not None:
        for sid, mark in instructions.items():
            if mark == "no_decision":
                no_decision.add(str(sid))
            elif mark == "liquidate":
                flat.add(str(sid))

    values = _rule_values(plan, raw, close, no_decision=no_decision)
    if flat:
        values = values.copy()
    for sid in flat:
        # A liquidate instruction is an explicit flat whether or not the rule
        # had an opinion, and it exempts the sid from the band and the dust
        # rules downstream (§3.3). It never creates an obligation.
        values.loc[sid] = 0.0

    for sid, value in values.items():
        key = str(sid)
        if math.isnan(float(value)):
            continue  # no opinion: the prior sub-target (if any) stands
        sub_targets[key] = float(value)

    changed = (
        any(key not in state.sub_targets or state.sub_targets[key] != value for key, value in sub_targets.items())
        or expired
    )
    expires_bar = _bar_at_offset(calendar, decision_bar, int(plan.valid_for_bars))
    new_state = ComponentState(
        component_id=plan.component_id,
        last_output_bar=decision_bar,
        decision_bar=decision_bar if changed else state.decision_bar,
        expires_bar=expires_bar,
        expired=False,
        sub_targets=dict(sub_targets),
        fitted_state_ref=state.fitted_state_ref,
        fitted_through_bar=state.fitted_through_bar,
        fit_scope=state.fit_scope,
    )
    if not values.notna().any():
        # An entirely NaN output is still a produced output: it advances the
        # anchor and restarts the validity clock (§3.3). Worth saying out loud.
        warnings.append(
            f"component {plan.component_id} produced an entirely NaN output at {decision_bar}: "
            "its sub-targets are held, and the validity clock restarts"
        )
    output = ComponentOutput(
        component_id=plan.component_id,
        raw=raw,
        standardized=None,
        sub_target=values,
        due=True,
        expired=expired,
        disabled_reason="expired" if expired else None,
        decision_bar=new_state.decision_bar,
        expires_bar=expires_bar,
        instructions=instructions,
        masked=applied_mask,
    )
    return output, new_state, flat


def _account_target(
    plans: Sequence[ComponentPlan],
    outputs: Sequence[ComponentOutput],
    states: Mapping[str, ComponentState],
    *,
    renormalize: bool,
) -> tuple[pd.Series, float]:
    """``sum_c weight_c * subtarget_c``; NaN where no component holds one.

    Opposing sub-targets net. The index is the union of every sid any component
    scored or holds a sub-target for, in first-seen order, so the target of a
    single-component book is exactly that component's own cross-section.
    """
    index: list[str] = []
    seen: set[str] = set()
    for output in outputs:
        for source in (output.sub_target, output.raw):
            if source is None:
                continue
            for sid in source.index:
                key = str(sid)
                if key not in seen:
                    seen.add(key)
                    index.append(key)
    for plan in plans:
        for sid in states[plan.component_id].sub_targets:
            if sid not in seen:
                seen.add(sid)
                index.append(sid)

    total = pd.Series(np.nan, index=pd.Index(index, dtype=object), dtype=float)
    for plan in plans:
        held = states[plan.component_id].sub_targets
        if not held:
            continue
        weight = float(plan.weight)
        for sid, value in held.items():
            contribution = weight * float(value)
            current = total.at[sid]
            total.at[sid] = contribution if math.isnan(current) else current + contribution

    names = {output.sub_target.name for output in outputs if output.sub_target is not None}
    if len(names) == 1:
        # A single-component book's account target IS that component's vector,
        # so it keeps its label (the decision bar) — what the live path emits.
        total.name = names.pop()

    scale = _renormalization(plans, states) if renormalize else 1.0
    if scale != 1.0:
        total = total * scale
    return total, scale


def _renormalization(plans: Sequence[ComponentPlan], states: Mapping[str, ComponentState]) -> float:
    """The declared-weight scale that restores a fully-populated target gross.

    ``sum of declared weights over all declared components / sum over the
    components holding a sub-target at this decision`` (§3.3). A component that
    dropped out — expired to zero, or gone flat by decision — holds no exposure,
    so it does not count towards the denominator; that is what "restoring the
    fully-populated target gross" requires. The specification does not spell
    the predicate out, so it is stated here rather than left to be inferred.
    """
    declared = sum(abs(float(plan.weight)) for plan in plans)
    holding = sum(
        abs(float(plan.weight))
        for plan in plans
        if any(float(value) != 0.0 for value in states[plan.component_id].sub_targets.values())
    )
    if holding <= 0.0:
        return 1.0
    return declared / holding


def _rule_values(
    plan: ComponentPlan,
    raw: pd.Series,
    close: pd.DataFrame,
    *,
    no_decision: Collection[str],
) -> pd.Series:
    """The rank-to-weight rule's output for this cross-section.

    The rule runs over the whole cross-section before any state rule, so a name
    the rule declines gets an explicit 0.0 (a full exit) and only a no-decision
    sid carries NaN.
    """
    if plan.weights_fn is not None:
        values = plan.weights_fn(raw)
        if not isinstance(values, pd.Series):
            raise CompositionError(
                f"component {plan.component_id}: weights_fn must return a pandas Series, got "
                f"{type(values).__name__}"
            )
        values = values.astype(float)
        marked = [sid for sid in dict.fromkeys(no_decision) if sid in values.index]
        if marked:
            values = values.copy()
            values.loc[marked] = np.nan
        return values
    return apply_to_weights(
        plan.to_weights,
        raw,
        plan.to_weights_params,
        close=close,
        no_decision=no_decision,
    )


def _score(plan: ComponentPlan, close: pd.DataFrame, volume: pd.DataFrame | None) -> pd.Series:
    """The component's own output at the decision bar, fail-loud (N7)."""
    scores = plan.component.score(close, volume)
    if not isinstance(scores, pd.Series):
        raise CompositionError(
            f"component {plan.component_id}: score must return a pandas Series indexed by sid, "
            f"got {type(scores).__name__}"
        )
    if scores.index.has_duplicates:
        raise CompositionError(
            f"component {plan.component_id}: score returned a duplicated index; a cross-section "
            "holds one value per name"
        )
    return scores


def _instructions(plan: ComponentPlan, close: pd.DataFrame, volume: pd.DataFrame | None) -> pd.Series | None:
    """The component's per-sid instruction marks, validated against §3.3's vocabulary."""
    marks = plan.component.instructions(close, volume)
    if marks is None:
        return None
    if not isinstance(marks, pd.Series):
        raise CompositionError(
            f"component {plan.component_id}: instructions must return a pandas Series or None, "
            f"got {type(marks).__name__}"
        )
    unknown = sorted({str(v) for v in marks.to_numpy()} - set(INSTRUCTIONS))
    if unknown:
        raise CompositionError(
            f"component {plan.component_id}: unknown instruction(s) {unknown}; the vocabulary is " f"{INSTRUCTIONS}"
        )
    return marks


def _panel_for(plan: ComponentPlan, scoring_panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The close panel the component's declared ``scoring_basis`` names (§3.8)."""
    panel = scoring_panels.get(plan.scoring_basis)
    if panel is None:
        raise CompositionError(
            f"component {plan.component_id} declares scoring_basis={plan.scoring_basis!r} but no "
            f"such panel was supplied; scoring_panels carries {sorted(scoring_panels)}"
        )
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise CompositionError(
            f"component {plan.component_id}: the {plan.scoring_basis!r} panel is empty — there is "
            "nothing to score (N7)"
        )
    return panel


def _calendar(
    sessions: Sequence[str] | None,
    scoring_panels: Mapping[str, pd.DataFrame],
    plans: Sequence[ComponentPlan],
) -> tuple[str, ...]:
    """The decision calendar elapsed bars are counted on."""
    if sessions is not None:
        calendar = tuple(str(bar) for bar in sessions)
        if not calendar:
            raise CompositionError("sessions is empty: elapsed bars cannot be counted (§3.3)")
        return calendar
    panel = _panel_for(plans[0], scoring_panels)
    return tuple(str(getattr(bar, "date", lambda: bar)()) for bar in panel.index)


def _bars_elapsed(calendar: Sequence[str], last_bar: str | None, decision_bar: str) -> int:
    """Trading sessions strictly after ``last_bar`` and through ``decision_bar``.

    Counted on the decision calendar rather than on wall-clock days, which is
    the same rule the live cadence gate uses (``live/daily.py``); a bar the
    calendar does not carry contributes nothing, so a calendar that starts after
    ``last_bar`` under-counts rather than inventing sessions.
    """
    if last_bar is None:
        return 0
    return sum(1 for bar in calendar if last_bar < bar <= decision_bar)


def _bar_at_offset(calendar: Sequence[str], decision_bar: str, offset: int) -> str | None:
    """``decision_bar + offset`` sessions on the calendar, or None past its end."""
    later = [bar for bar in calendar if bar > decision_bar]
    if offset <= 0:
        return decision_bar
    if len(later) < offset:
        return None
    return later[offset - 1]
