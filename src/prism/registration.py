"""Registrations: the declared arms of a search, and strict mode over them (spec §3.13, D5).

A registration declares the space a search may occupy — the arms inside it, the
inputs frozen across them, and the rules that stop the search and gate
promotion — before the first configuration is evaluated. This module is the
loader for that file and the enforcement of what it declares.

Three things are deliberate here.

**Strict mode is defined against the declaration, not against a vocabulary of
allowed edits.** :meth:`Registration.check_strict` resolves every declared arm
into a full :class:`~prism.spec.Specification`, computes its ``config_id``
under the run's own evaluation context, and refuses the run unless its
``config_id`` equals one of them. A standardization or weighting alternative is
therefore a declared arm or it is not run: there is no free diagnostic, because
there is no code path that accepts a configuration nobody wrote down (§3.13,
decision D5).

**Sharing is structural.** The brief requires the two convex arms to share one
mapping and one risk model. Prose cannot enforce that, so the loader does: an
arm's calibration file must differ from ``[shared].calibration_family`` in no
key but ``confidence``; an arm may not name a risk model other than
``[shared].risk_model``; a key restated in ``[shared]`` for enumerability must
equal the value in the file it restates; and the file, never the restatement,
is what a run reads. The two convex arms cannot drift apart without the
registration refusing to load.

**An arm is a composition, not a second specification file.** The registration
names the control's specification, and each arm is that specification with the
overrides the arm declares: its profile, its construction block, its
calibration, and the registration's frozen cost stack and execution profile.
Anything the registration restates about the control — the component's import
path, its ``scoring_basis``, the control arm's rank-to-weight rule and its
parameters — is checked against the control specification and refuses on
disagreement rather than silently overriding it. Writing three near-identical
specification files instead would put three chances to drift where there is one
fact.

What this module does not do: it does not evaluate a stop or promotion rule's
thresholds against a result. It parses and validates them, and answers whether
the promotion rule can promote at all (``kind = "none"`` cannot). Rule outcomes
are recorded by ``prism compare --promotion-check``, which the engine never
runs on its own.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from prism.live.risk_profile import load_declared_profile
from prism.spec import Specification, load_spec_table
from prism.validation.trials import stable_json_dumps
from prism.validation.walk_forward import FoldTable
from prism.workspace import Workspace, resolve_workspace

logger = logging.getLogger(__name__)

#: Every top-level key a ``registrations/*.toml`` file carries (§3.13). An
#: unknown key is refused: a mis-spelled ``strict`` that silently ran a search
#: outside its declaration is the defect this loader exists to prevent.
REGISTRATION_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "created",
        "selection_set",
        "strict",
        "trial_budget",
        "components",
        "shared",
        "arms",
        "sample",
        "folds",
        "frozen",
        "evidence",
        "sharpe_bootstrap",
        "stop_rule",
        "promotion_rule",
        "benchmark",
    }
)

#: ``[shared]`` keys that restate a value of the calibration family for
#: enumerability. Each must equal the value in ``calibration_family``; the file
#: is what a run reads (§3.13).
SHARED_CALIBRATION_KEYS: tuple[str, ...] = (
    "standardize",
    "weighting",
    "shrink_pseudo_periods",
    "block_periods",
    "min_training_periods",
    "min_blocks",
)

#: ``[shared]`` construction settings that reach the convex program only. A
#: heuristic arm has no objective for ``risk_aversion`` to scale and no cap
#: fraction to accept against, so carrying them into its ``[construction]``
#: block would put settings it cannot act on into its ``config_id``.
SHARED_CONVEX_CONSTRUCTION_KEYS: tuple[str, ...] = ("risk_aversion", "eps", "max_at_cap_fraction")

#: ``[shared]`` construction settings every arm carries: how the constructor
#: treats a held name that has left the eligible set is a property of the
#: decision, not of the constructor kind (§3.7).
SHARED_CONSTRUCTION_KEYS: tuple[str, ...] = ("eligibility_exit",)

#: The stop- and promotion-rule keys §3.13 supports. ``callback`` names an
#: importable ``module:function``; its import path and source hash are recorded
#: in the rule outcome, so an opaque policy is still an identified one.
RULE_KEYS: frozenset[str] = frozenset(
    {
        "kind",
        "note",
        "min_obs",
        "min_net_return",
        "min_net_sharpe",
        "min_excess_sharpe",
        "hurdle_series",
        "min_dsr",
        "max_drawdown",
        "require_prospective_from",
        "require_monitor_not_contradicting",
        "callback",
    }
)

RULE_KINDS: tuple[str, ...] = ("structured", "none")

#: Evaluation-context keys a registered run must agree with the registration on
#: (§3.12's ``evaluation_context``, §3.13's ``[frozen]``): the same key name
#: carries the same string on both sides.
#:
#: ``executable_return`` is deliberately *not* here. The registration freezes
#: the label convention the calibration is fitted on — ``open_to_open``, the
#: return the fills can earn over the horizon (§3.4) — while the evaluation
#: context records the accounting engine's own convention for the same key
#: (``next_open_fill_marked_close_to_close`` for the canonical engine,
#: ``prism.validation.trials.ENGINE_EXECUTABLE_RETURN``). Comparing the two
#: strings would refuse every canonical run over a collision of vocabularies.
#: The registration's value is checked against the calibration files instead,
#: which is where it means something.
FROZEN_CONTEXT_KEYS: tuple[str, ...] = (
    "accounting_engine",
    "fold_portfolio",
    "cost_refresh",
    "visibility_mode",
)


class RegistrationError(ValueError):
    """A registration is malformed, internally inconsistent, or refuses a run.

    Raised rather than warned. Every check in this module exists because the
    alternative is a run that looks registered and is not: an arm whose
    calibration quietly diverged from the family, a capital scenario nobody
    declared, a fold table that moved under a set already half spent.
    """


# --------------------------------------------------------------------------
# The parsed file
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RegisteredComponent:
    """One ``[[components]]`` entry: the component every arm runs."""

    id: str
    path: str
    expected_sign: str
    scoring_basis: str


@dataclass(frozen=True)
class Shared:
    """The ``[shared]`` block: the files and settings the arms hold in common."""

    risk_model: Path
    calibration_family: Path
    calibration_restated: Mapping[str, Any]
    construction: Mapping[str, Any]
    convex_construction: Mapping[str, Any]


@dataclass(frozen=True)
class Arm:
    """One declared arm. ``construction`` is the ``[arms.construction]`` block."""

    id: str
    profile: Path
    kind: str
    construction: Mapping[str, Any]
    calibration: Path | None = None
    confidence: float | None = None
    #: ``[arms.components]``: the weight each declared component carries in this
    #: arm's position-space composition, or None to run the control's own
    #: weights. Every declared component is listed when the table is present,
    #: so a zero is an explicit exclusion and never an omission.
    component_weights: Mapping[str, float] | None = None
    #: ``universe_index``: the index this arm trades, replacing the control's.
    #: None runs the control's own universe. The dataset must carry membership
    #: for it, which the evaluator's universe resolution refuses loudly if not.
    universe_index: str | None = None


@dataclass(frozen=True)
class Sample:
    """The ``[sample]`` block: the span, the first and last decision, the warm-up."""

    history_start: dt.date
    first_decision: dt.date
    last_decision: dt.date
    warmup_bars: int
    decision_offset_minutes: int


@dataclass(frozen=True)
class Frozen:
    """The ``[frozen]`` block: what is held identical across every arm and scenario."""

    freeze_record: Path
    executable_return: str
    cost_stack: Path
    execution_profile: Path
    accounting_engine: str
    visibility_mode: str
    fold_portfolio: str
    cost_refresh: str
    capital_scenarios: tuple[float, ...]
    capital_unit: str


@dataclass(frozen=True)
class Rule:
    """A parsed ``[stop_rule]`` or ``[promotion_rule]`` (§3.13)."""

    kind: str
    note: str | None
    thresholds: Mapping[str, Any]
    callback: str | None

    def promotes(self) -> bool:
        """Whether this rule can promote anything at all.

        ``kind = "none"`` is a promotion rule that never promotes, which is what
        the stage 4 registration declares: a same-sample comparison can kill a
        configuration and can never promote one.
        """
        return self.kind != "none"


@dataclass(frozen=True)
class Benchmark:
    """The ``[benchmark]`` block: the cash hurdle and the control specification."""

    cash_series: str
    cash_source: str
    cash_conversion: str
    day_count: str
    control: Path


@dataclass(frozen=True)
class Registration:
    """A loaded registration and everything strict mode is enforced from."""

    path: Path
    root: Path
    id: str
    created: dt.date
    selection_set: str
    strict: bool
    trial_budget: int
    components: tuple[RegisteredComponent, ...]
    shared: Shared
    arms: tuple[Arm, ...]
    sample: Sample
    fold_table_path: Path
    fold_table: FoldTable
    frozen: Frozen
    evidence: Mapping[str, Any]
    sharpe_bootstrap: Mapping[str, Any]
    stop_rule: Rule
    promotion_rule: Rule
    benchmark: Benchmark

    @property
    def arm_ids(self) -> tuple[str, ...]:
        """The declared arm ids, in declaration order."""
        return tuple(arm.id for arm in self.arms)

    def arm(self, arm_id: str) -> Arm:
        """The declared arm of that id."""
        for arm in self.arms:
            if arm.id == arm_id:
                return arm
        raise RegistrationError(
            f"{self.path} declares no arm {arm_id!r}; the declared arms are {list(self.arm_ids)}. "
            "An arm that is not declared is not run (§3.13, decision D5)"
        )

    # ---------------------------------------------------------------- arms

    def arm_specification(self, arm_id: str, *, workspace: Workspace | None = None) -> Specification:
        """Resolve one declared arm into the specification a run of it would load.

        The control specification named by ``[benchmark].control`` supplies the
        universe, the components with their parameters and schedules, and the
        eligibility screen; the arm supplies its profile and its construction
        block; ``[shared]`` supplies the risk model and the construction
        settings; ``[frozen]`` supplies the cost stack and the execution
        profile. Everything the registration restates about the control is
        checked against it, never merged over it.
        """
        ws = workspace if workspace is not None else Workspace(self.root)
        arm = self.arm(arm_id)
        control = _read_toml(self.benchmark.control)
        strategy = _control_strategy(control, registration=self, arm=arm)
        construction = _arm_construction(self, arm, control=control)
        policy = _arm_policy(self, arm, control=control)
        table = {"strategy": strategy, "construction": construction, "policy": policy}
        specification = load_spec_table(
            table,
            path=self.path.parent / f"{self.id}#{arm.id}.toml",
            workspace=ws,
            instantiate=True,
        )
        declared = {component.id: component.scoring_basis for component in self.components}
        for loaded in specification.components:
            basis = loaded.declarations["scoring_basis"]
            if basis != declared[loaded.component_id]:
                raise RegistrationError(
                    f"{self.path} declares component {loaded.component_id!r} on scoring_basis "
                    f"{declared[loaded.component_id]!r} and {loaded.spec.path} declares {basis!r}. The "
                    "basis fixes the panel the score, the label and the horizon volatility are computed "
                    "on, so a run under the registration's reading would be a different experiment (§3.3)"
                )
        return specification

    def arm_config_ids(
        self,
        *,
        evaluation_context: Mapping[str, Any] | None,
        workspace: Workspace | None = None,
        engine_version: str | None = None,
    ) -> dict[str, str]:
        """``{arm_id: config_id}`` for every declared arm under one evaluation context.

        The context is the run's own: two configurations are the same
        configuration only when they were evaluated the same way, and §3.12
        puts capital, the data id, the fold hash and the accounting engine in
        the identity for exactly that reason.
        """
        return {
            arm.id: self.arm_specification(arm.id, workspace=workspace).config_id(
                evaluation_context=evaluation_context, engine_version=engine_version
            )
            for arm in self.arms
        }

    # -------------------------------------------------------------- strict

    def check_strict(
        self,
        *,
        config_id: str,
        evaluation_context: Mapping[str, Any],
        workspace: Workspace | None = None,
        engine_version: str | None = None,
        check_freeze: bool = True,
    ) -> str:
        """Refuse a run that is not a declared arm at a declared capital scenario.

        Returns the matching ``arm_id``. Raises :class:`RegistrationError`
        naming the arm ids it accepts otherwise.

        ``check_freeze=False`` omits the frozen score panel binding, which is
        the one declared path that the run's own opening step writes rather than
        an input that exists beforehand. Pass it only where no run is being
        authorized — enumerating the arms of a registration, say. A run passes
        the default.
        """
        if not self.strict:
            raise RegistrationError(
                f"{self.path} declares strict = false; check_strict enforces a declaration this "
                "registration did not make. Refusing rather than silently accepting the run"
            )
        self.check_frozen_inputs(evaluation_context)
        if check_freeze:
            self.check_freeze_record(evaluation_context)
        declared = self.arm_config_ids(
            evaluation_context=evaluation_context, workspace=workspace, engine_version=engine_version
        )
        for arm_id, declared_id in declared.items():
            if declared_id == config_id:
                return arm_id
        listed = ", ".join(f"{arm_id}={declared_id[:12]}" for arm_id, declared_id in declared.items())
        raise RegistrationError(
            f"{self.path} is strict and config_id {config_id[:12]} is not a declared arm: it accepts "
            f"{listed}. A standardization or weighting alternative is a declared arm or it is not run "
            "(§3.13, decision D5); adding one after a result is a new registration, never an edit of this one"
        )

    def check_frozen_inputs(self, evaluation_context: Mapping[str, Any]) -> None:
        """Refuse a run whose frozen inputs are not the ones this registration declares."""
        for key in FROZEN_CONTEXT_KEYS:
            declared = getattr(self.frozen, key)
            if key not in evaluation_context:
                raise RegistrationError(
                    f"{self.path}: the run's evaluation_context declares no {key!r}; a registered run "
                    f"states it, and this registration freezes {key}={declared!r}"
                )
            actual = evaluation_context[key]
            if actual != declared:
                raise RegistrationError(
                    f"{self.path} freezes {key}={declared!r} across every arm; this run declares "
                    f"{actual!r}. The arms are compared under identical conventions or they are not "
                    "compared (§3.13)"
                )
        capital = evaluation_context.get("capital")
        if (
            isinstance(capital, bool)
            or not isinstance(capital, (int, float))
            or float(capital) not in self.frozen.capital_scenarios
        ):
            raise RegistrationError(
                f"{self.path} declares capital scenarios {list(self.frozen.capital_scenarios)} in "
                f"{self.frozen.capital_unit}; this run declares capital={capital!r}. A scenario outside "
                "the declaration is outside the registered space"
            )
        recorded = evaluation_context.get("fold_table_sha256")
        expected = self.fold_table.content_hash
        if recorded is None:
            raise RegistrationError(
                f"{self.path}: the run's evaluation_context carries no fold_table_sha256, so the "
                f"geometry it ran cannot be compared with the frozen table at {self.fold_table_path}. "
                "A registered run records the hash it ran (§3.13)"
            )
        if recorded != expected:
            raise RegistrationError(
                f"{self.path} freezes the fold geometry of {self.fold_table_path} "
                f"(sha256 {expected[:12]}); this run carries {str(recorded)[:12]}. Fold geometry is a "
                "frozen table, not a splitter argument (decision D7): moving a boundary moves every "
                "per-fold denominator in a set already partly spent"
            )

    def check_freeze_record(self, evaluation_context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Refuse a run that is not on the frozen score panel this registration names.

        The record is written by ``prism evaluate <spec> --freeze-scores``, the
        step that materializes the panel the arms share. Nothing else produces
        it, so a registration whose panel has not been frozen yet refuses every
        run rather than letting the first arm freeze one implicitly and the
        second arm read a different one.
        """
        record = _read_json(self.frozen.freeze_record, what="freeze record")
        for context_key, record_key in (("data_id", "data_id"), ("visibility_mode", "visibility_mode")):
            declared = record.get(record_key)
            actual = evaluation_context.get(context_key)
            if declared is not None and actual is not None and declared != actual:
                raise RegistrationError(
                    f"{self.frozen.freeze_record} froze the score panel on {record_key}={declared!r}; "
                    f"this run declares {actual!r}. The panel supplies component outputs only and the "
                    "accounting engine prices from the vintage's bars, which is consistent only while "
                    "the two are the same dataset (§3.13)"
                )
        return record

    def check_budget(self, attempts_spent: int) -> None:
        """Refuse a run beyond the declared budget. One attempt is one ``arm_id``."""
        if attempts_spent >= self.trial_budget:
            raise RegistrationError(
                f"{self.path} declares trial_budget={self.trial_budget} and {attempts_spent} "
                "attempt(s) are already spent; strict mode refuses a run beyond budget rather than "
                "warning (§3.13). A retry after a crash carries kind='retry' and spends no budget"
            )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_registration(path: Path | str, *, workspace: Workspace | None = None) -> Registration:
    """Load and validate a ``registrations/*.toml`` file (§3.13).

    Every declared input path must exist and is resolved against the workspace
    root. The one exception is ``[frozen].freeze_record``, which the run's own
    opening step writes; :meth:`Registration.check_freeze_record` is where that
    one is required, so a registration can be read, reviewed and tested before
    the panel it will run on has been frozen.
    """
    ws = workspace if workspace is not None else resolve_workspace()
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise RegistrationError(f"no registration at {file_path}")
    try:
        table = tomllib.loads(file_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RegistrationError(f"{file_path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise RegistrationError(f"{file_path} cannot be read: {exc}") from exc

    where = str(file_path)
    _reject_unknown(table, REGISTRATION_KEYS, where=where)
    registration_id = _require_str(table, "id", where=where)
    created = _require_date(table, "created", where=where)
    selection_set = _require_str(table, "selection_set", where=where)
    strict = _require_bool(table, "strict", where=where)
    trial_budget = _require_int(table, "trial_budget", where=where, minimum=1)

    components = _parse_components(table, where=where)
    shared = _parse_shared(table, root=ws.root, where=where)
    arms = _parse_arms(table, shared=shared, root=ws.root, where=where, component_ids=tuple(c.id for c in components))
    sample = _parse_sample(table, where=where)
    folds_block = _require_table(table, "folds", where=where)
    _reject_unknown(folds_block, {"table"}, where=f"{where} [folds]")
    fold_table_path = _resolve_file(
        _require_str(folds_block, "table", where=f"{where} [folds]"),
        root=ws.root,
        key="folds.table",
        where=where,
    )
    fold_table = FoldTable.from_toml(fold_table_path)
    frozen = _parse_frozen(table, root=ws.root, where=where)
    stop_rule = _parse_rule(table, key="stop_rule", where=where)
    promotion_rule = _parse_rule(table, key="promotion_rule", where=where)
    benchmark = _parse_benchmark(table, root=ws.root, where=where)

    registration = Registration(
        path=file_path,
        root=ws.root,
        id=registration_id,
        created=created,
        selection_set=selection_set,
        strict=strict,
        trial_budget=trial_budget,
        components=components,
        shared=shared,
        arms=arms,
        sample=sample,
        fold_table_path=fold_table_path,
        fold_table=fold_table,
        frozen=frozen,
        evidence=dict(table.get("evidence") or {}),
        sharpe_bootstrap=dict(table.get("sharpe_bootstrap") or {}),
        stop_rule=stop_rule,
        promotion_rule=promotion_rule,
        benchmark=benchmark,
    )
    _check_sample_against_folds(registration, where=where)
    _check_calibration_sharing(registration, where=where)
    _check_variance_budget(registration, where=where)
    _check_budget_admits_arms(registration, where=where)
    return registration


def _parse_components(table: Mapping[str, Any], *, where: str) -> tuple[RegisteredComponent, ...]:
    rows = table.get("components")
    if not isinstance(rows, list) or not rows:
        raise RegistrationError(f"{where} declares no [[components]]; a registration runs at least one")
    components: list[RegisteredComponent] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        label = f"{where} [[components]] #{index}"
        if not isinstance(row, Mapping):
            raise RegistrationError(f"{label} is {row!r}, not a table")
        _reject_unknown(row, {"id", "path", "expected_sign", "scoring_basis"}, where=label)
        component_id = _require_str(row, "id", where=label)
        if component_id in seen:
            raise RegistrationError(f"{label} repeats component id {component_id!r}; component state is keyed by id")
        seen.add(component_id)
        components.append(
            RegisteredComponent(
                id=component_id,
                path=_require_str(row, "path", where=label),
                expected_sign=_require_choice(row, "expected_sign", ("positive", "negative"), where=label),
                scoring_basis=_require_str(row, "scoring_basis", where=label),
            )
        )
    return tuple(components)


def _parse_shared(table: Mapping[str, Any], *, root: Path, where: str) -> Shared:
    block = _require_table(table, "shared", where=where)
    label = f"{where} [shared]"
    risk_model = _resolve_file(
        _require_str(block, "risk_model", where=label), root=root, key="risk_model", where=label
    )
    family = _resolve_file(
        _require_str(block, "calibration_family", where=label), root=root, key="calibration_family", where=label
    )
    restated = {key: block[key] for key in SHARED_CALIBRATION_KEYS if key in block}
    construction = {key: block[key] for key in SHARED_CONSTRUCTION_KEYS if key in block}
    convex = {key: block[key] for key in SHARED_CONVEX_CONSTRUCTION_KEYS if key in block}
    known = (
        {"risk_model", "calibration_family"}
        | set(SHARED_CALIBRATION_KEYS)
        | set(SHARED_CONSTRUCTION_KEYS)
        | set(SHARED_CONVEX_CONSTRUCTION_KEYS)
    )
    _reject_unknown(block, known, where=label)
    return Shared(
        risk_model=risk_model,
        calibration_family=family,
        calibration_restated=restated,
        construction=construction,
        convex_construction=convex,
    )


def _parse_arms(
    table: Mapping[str, Any],
    *,
    shared: Shared,
    root: Path,
    where: str,
    component_ids: tuple[str, ...] = (),
) -> tuple[Arm, ...]:
    rows = table.get("arms")
    if not isinstance(rows, list) or not rows:
        raise RegistrationError(
            f"{where} declares no [[arms]]; strict mode is defined against the declared arms and is "
            "unimplementable without them (decision D5)"
        )
    arms: list[Arm] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        label = f"{where} [[arms]] #{index}"
        if not isinstance(row, Mapping):
            raise RegistrationError(f"{label} is {row!r}, not a table")
        _reject_unknown(
            row,
            {"id", "profile", "calibration", "confidence", "risk_model", "construction", "components", "universe_index"},
            where=label,
        )
        arm_id = _require_str(row, "id", where=label)
        if arm_id in seen:
            raise RegistrationError(f"{label} repeats arm id {arm_id!r}; an arm id is one attempt against the budget")
        seen.add(arm_id)
        label = f"{where} [[arms]] {arm_id!r}"
        construction = _require_table(row, "construction", where=label)
        kind = _require_choice(construction, "kind", ("heuristic", "convex"), where=f"{label} [arms.construction]")
        calibration = (
            _resolve_file(row["calibration"], root=root, key="calibration", where=label)
            if row.get("calibration") is not None
            else None
        )
        if kind == "convex" and calibration is None:
            raise RegistrationError(
                f"{label} declares a convex construction and names no calibration; a convex "
                "configuration cannot be evaluated without the file its forecast comes from (§3.13)"
            )
        if kind == "heuristic" and calibration is not None:
            raise RegistrationError(
                f"{label} declares a heuristic construction and a calibration file; the heuristic path "
                "reads no calibration, and a file in the identity that nothing reads is a false claim "
                "about what the arm ran"
            )
        if row.get("risk_model") is not None:
            declared = _resolve_file(row["risk_model"], root=root, key="risk_model", where=label)
            if declared != shared.risk_model:
                raise RegistrationError(
                    f"{label} names risk model {declared}, which is not [shared].risk_model "
                    f"{shared.risk_model}. The arms share one risk model by construction, so a "
                    "difference between them can only be the calibration's confidence (§3.13)"
                )
        confidence = row.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise RegistrationError(f"{label} declares confidence={confidence!r}, which is not a number")
            confidence = float(confidence)
            if not 0.5 < confidence < 1.0:
                raise RegistrationError(
                    f"{label} declares confidence={confidence}; a one-sided level lies strictly between "
                    "0.5 and 1.0 (§3.4, decision D6)"
                )
        weights = _parse_component_weights(row.get("components"), component_ids=component_ids, where=label)
        universe_index = row.get("universe_index")
        if universe_index is not None and (not isinstance(universe_index, str) or not universe_index.strip()):
            raise RegistrationError(f"{label} declares universe_index={universe_index!r}; it names one index id")
        arms.append(
            Arm(
                id=arm_id,
                profile=_resolve_file(
                    _require_str(row, "profile", where=label), root=root, key="profile", where=label
                ),
                kind=kind,
                construction={key: value for key, value in construction.items() if key != "kind"},
                calibration=calibration,
                confidence=confidence,
                component_weights=weights,
                universe_index=None if universe_index is None else str(universe_index),
            )
        )
    return tuple(arms)


def _parse_component_weights(
    value: Any, *, component_ids: tuple[str, ...], where: str
) -> dict[str, float] | None:
    """``[arms.components]``: one finite non-negative weight per declared component.

    The table is the arm's whole difference from the control when the arms of a
    registration vary *which* characteristics the book holds rather than how it
    is constructed (SPEC §5.3: one registered selection set, deflated as a
    whole). It lists every declared component, so an arm that drops one says so
    with a zero; a partial table is refused because an unlisted component would
    silently keep the control's weight. At least one weight is positive: an arm
    that holds nothing is not a configuration.
    """
    if value is None:
        return None
    label = f"{where} [arms.components]"
    if not isinstance(value, Mapping):
        raise RegistrationError(f"{label} is {value!r}, not a table of component weights")
    declared = set(component_ids)
    listed = set(str(key) for key in value)
    if listed != declared:
        raise RegistrationError(
            f"{label} weights {sorted(listed)} and the registration declares components "
            f"{sorted(declared)}; every declared component is weighted explicitly (zero excludes it)"
        )
    weights: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise RegistrationError(f"{label} declares {key}={raw!r}, which is not a finite number")
        if float(raw) < 0.0:
            raise RegistrationError(f"{label} declares {key}={raw!r}; a component weight is non-negative")
        weights[str(key)] = float(raw)
    if not any(weight > 0.0 for weight in weights.values()):
        raise RegistrationError(f"{label} weights every component at zero; the arm would hold nothing")
    return weights


def _parse_sample(table: Mapping[str, Any], *, where: str) -> Sample:
    block = _require_table(table, "sample", where=where)
    label = f"{where} [sample]"
    _reject_unknown(
        block,
        {"history_start", "first_decision", "last_decision", "warmup_bars", "decision_offset_minutes"},
        where=label,
    )
    sample = Sample(
        history_start=_require_date(block, "history_start", where=label),
        first_decision=_require_date(block, "first_decision", where=label),
        last_decision=_require_date(block, "last_decision", where=label),
        warmup_bars=_require_int(block, "warmup_bars", where=label, minimum=0),
        decision_offset_minutes=_require_int(block, "decision_offset_minutes", where=label, minimum=0),
    )
    if not sample.history_start < sample.first_decision <= sample.last_decision:
        raise RegistrationError(
            f"{label} declares history_start={sample.history_start}, first_decision="
            f"{sample.first_decision}, last_decision={sample.last_decision}; history must open before "
            "the first decision and the last decision may not precede the first"
        )
    return sample


def _parse_frozen(table: Mapping[str, Any], *, root: Path, where: str) -> Frozen:
    block = _require_table(table, "frozen", where=where)
    label = f"{where} [frozen]"
    _reject_unknown(
        block,
        {
            "freeze_record",
            "executable_return",
            "cost_stack",
            "execution_profile",
            "accounting_engine",
            "visibility_mode",
            "fold_portfolio",
            "cost_refresh",
            "capital_scenarios",
            "capital_unit",
        },
        where=label,
    )
    scenarios = block.get("capital_scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise RegistrationError(f"{label} declares no capital_scenarios; each scenario is one evaluation of an arm")
    values: list[float] = []
    for entry in scenarios:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)) or entry <= 0:
            raise RegistrationError(f"{label} declares capital scenario {entry!r}; each is a positive dollar amount")
        values.append(float(entry))
    if len(set(values)) != len(values):
        raise RegistrationError(f"{label} repeats a capital scenario; each scenario is one evaluation of an arm")
    freeze_record = Path(_require_str(block, "freeze_record", where=label))
    return Frozen(
        freeze_record=freeze_record if freeze_record.is_absolute() else root / freeze_record,
        executable_return=_require_str(block, "executable_return", where=label),
        cost_stack=_resolve_file(
            _require_str(block, "cost_stack", where=label), root=root, key="cost_stack", where=label
        ),
        execution_profile=_resolve_file(
            _require_str(block, "execution_profile", where=label), root=root, key="execution_profile", where=label
        ),
        accounting_engine=_require_choice(
            block, "accounting_engine", ("canonical", "historical_constant_weight"), where=label
        ),
        visibility_mode=_require_str(block, "visibility_mode", where=label),
        fold_portfolio=_require_choice(block, "fold_portfolio", ("carry", "reset"), where=label),
        cost_refresh=_require_str(block, "cost_refresh", where=label),
        capital_scenarios=tuple(values),
        capital_unit=_require_str(block, "capital_unit", where=label),
    )


def _parse_rule(table: Mapping[str, Any], *, key: str, where: str) -> Rule:
    block = _require_table(table, key, where=where)
    label = f"{where} [{key}]"
    _reject_unknown(block, RULE_KEYS, where=label)
    kind = _require_choice(block, "kind", RULE_KINDS, where=label)
    callback = block.get("callback")
    if callback is not None:
        if not isinstance(callback, str) or ":" not in callback:
            raise RegistrationError(
                f"{label} declares callback={callback!r}; it names an importable 'module:function' whose "
                "import path and source hash are recorded in the rule outcome (§3.13)"
            )
    thresholds = {k: v for k, v in block.items() if k not in ("kind", "note", "callback")}
    if kind == "none" and (thresholds or callback is not None):
        carried = sorted(thresholds) + (["callback"] if callback is not None else [])
        raise RegistrationError(
            f"{label} declares kind='none' and also carries {carried}; a rule that never fires may not "
            "also carry conditions, because a reader cannot tell which of the two the author meant"
        )
    note = block.get("note")
    if note is not None and not isinstance(note, str):
        raise RegistrationError(f"{label} declares note={note!r}, which is not a string")
    return Rule(kind=kind, note=note, thresholds=thresholds, callback=callback)


def _parse_benchmark(table: Mapping[str, Any], *, root: Path, where: str) -> Benchmark:
    block = _require_table(table, "benchmark", where=where)
    label = f"{where} [benchmark]"
    _reject_unknown(block, {"cash_series", "cash_source", "cash_conversion", "day_count", "control"}, where=label)
    return Benchmark(
        cash_series=_require_str(block, "cash_series", where=label),
        cash_source=_require_str(block, "cash_source", where=label),
        cash_conversion=_require_str(block, "cash_conversion", where=label),
        day_count=_require_str(block, "day_count", where=label),
        control=_resolve_file(_require_str(block, "control", where=label), root=root, key="control", where=label),
    )


# --------------------------------------------------------------------------
# Structural checks
# --------------------------------------------------------------------------


def _check_sample_against_folds(registration: Registration, *, where: str) -> None:
    """``[sample]`` states the fold table's own first and last decision, or refuses."""
    first = registration.fold_table.first_decision.date()
    last = registration.fold_table.last_decision.date()
    declared = registration.sample
    if declared.first_decision != first or declared.last_decision != last:
        raise RegistrationError(
            f"{where} [sample] declares first_decision={declared.first_decision} and "
            f"last_decision={declared.last_decision}; {registration.fold_table_path} runs "
            f"{first}..{last}. The table is the authority on fold geometry (decision D7) and the "
            "registration restates it for readers, so a disagreement is a defect in one of the two"
        )


def _check_calibration_sharing(registration: Registration, *, where: str) -> None:
    """The two convex arms share one mapping by construction (§3.13, decision D5).

    Three checks, each closing a way the arms could drift apart while the file
    still read as if they had not: a restated ``[shared]`` key that no longer
    matches the family it restates; an arm calibration that differs from the
    family in something other than ``confidence``; and an arm whose declared
    ``confidence`` is not the one its own calibration file carries.
    """
    family_path = registration.shared.calibration_family
    family = _read_toml(family_path)
    for key, value in registration.shared.calibration_restated.items():
        if key not in family:
            raise RegistrationError(
                f"{where} [shared] restates {key}={value!r}, which {family_path} does not declare. "
                "[shared] restates calibration keys for enumerability only; a key that is not in the "
                "file describes a calibration nothing will run (§3.13)"
            )
        if family[key] != value:
            raise RegistrationError(
                f"{where} [shared] restates {key}={value!r} and {family_path} declares {family[key]!r}. "
                "The file, not the registration, is what the run reads, so a disagreement is refused "
                "rather than resolved in favour of either (§3.13)"
            )
    paths = {family_path} | {arm.calibration for arm in registration.arms if arm.calibration is not None}
    for path in sorted(paths):
        declared = _read_toml(path).get("executable_return")
        if declared is not None and declared != registration.frozen.executable_return:
            raise RegistrationError(
                f"{where} [frozen] declares executable_return={registration.frozen.executable_return!r} "
                f"and {path} fits its slope on {declared!r}. The label the mapping is fitted on and the "
                "return the registration freezes are one convention, or the forecast is not "
                "commensurable with the outcome it is read against (§3.4)"
            )
    if "confidence" in family:
        raise RegistrationError(
            f"{where} [shared].calibration_family {family_path} declares confidence="
            f"{family['confidence']!r}; the family is the point form of the mapping and the "
            "uncertainty arm's file is the one that adds a level (decision D6)"
        )
    expected_sign = {component.expected_sign for component in registration.components}
    for arm in registration.arms:
        if arm.calibration is None:
            if arm.confidence is not None:
                raise RegistrationError(
                    f"{where} [[arms]] {arm.id!r} declares confidence={arm.confidence} and no calibration "
                    "file; a confidence level with nothing to apply it to is not a declaration of anything"
                )
            continue
        payload = _read_toml(arm.calibration)
        difference = _table_difference(family, payload)
        if difference - {"confidence"}:
            raise RegistrationError(
                f"{where} [[arms]] {arm.id!r} names {arm.calibration}, which differs from "
                f"[shared].calibration_family {family_path} in {sorted(difference - {'confidence'})}. The "
                "two convex arms share one mapping and may differ in confidence alone, which is what "
                "makes the comparison a comparison of the uncertainty adjustment (§3.13, decision D5)"
            )
        declared_confidence = payload.get("confidence")
        if declared_confidence is None and arm.confidence is not None:
            raise RegistrationError(
                f"{where} [[arms]] {arm.id!r} declares confidence={arm.confidence} and {arm.calibration} "
                "declares none, so the run would take the point form under an uncertainty arm's name"
            )
        if declared_confidence is not None and (
            arm.confidence is None or abs(float(declared_confidence) - arm.confidence) > 1e-12
        ):
            raise RegistrationError(
                f"{where} [[arms]] {arm.id!r} declares confidence={arm.confidence!r} and "
                f"{arm.calibration} declares {declared_confidence!r}; the restatement is enumerability "
                "only and must equal the file (§3.13, decision D6)"
            )
        sign = payload.get("expected_sign")
        if sign is not None and expected_sign and sign not in expected_sign:
            raise RegistrationError(
                f"{where} [[arms]] {arm.id!r} names {arm.calibration}, which declares expected_sign="
                f"{sign!r} against the registration's {sorted(expected_sign)}. expected_sign is declared "
                "once, in [[components]]; a calibration file that conflicts refuses the run rather than "
                "silently winning (§3.13, Q9)"
            )


def _check_variance_budget(registration: Registration, *, where: str) -> None:
    """The risk model must have a channel to act through, and the convex arms must share it.

    Decision D8. ``risk_aversion`` at its reference value of zero keeps the
    variance term out of the objective, so with no variance budget and
    negligible cost curvature the convex program is linear in expected return
    and collapses to a maximal-concentration corner: the full factor covariance
    would carry weight zero in the objective and appear in no constraint, and
    the comparison would say nothing about the risk model. A registration that
    declares one without the other is refused here rather than discovered in the
    at-cap fraction of a run that has already spent its budget.

    The profile is read with the reference project's tighten-only discipline
    off: a profile's pins are its own project's policy, and this loader's
    business is the relationship between the arms.
    """
    risk_horizon = _read_toml(registration.shared.risk_model).get("horizon_bars")
    budgets: dict[str, Mapping[str, Any]] = {}
    for arm in registration.arms:
        if arm.kind != "convex":
            continue
        profile = load_declared_profile(arm.profile, tighten_only=False)
        limit = profile.limit("variance_budget")
        risk_aversion = registration.shared.convex_construction.get("risk_aversion")
        if limit is None:
            if risk_aversion is not None and float(risk_aversion) == 0.0:
                raise RegistrationError(
                    f"{where} [[arms]] {arm.id!r} runs a convex construction at risk_aversion=0 under "
                    f"{arm.profile}, which declares no variance_budget limit. The budget is the only "
                    "channel through which the risk model acts on the solution; without it the program "
                    "is linear in expected return and collapses to a maximal-concentration corner "
                    "(decision D8)"
                )
            continue
        budgets[arm.id] = dict(limit.params)
        horizon = limit.params.get("horizon_bars")
        if horizon is not None and risk_horizon is not None and int(horizon) != int(risk_horizon):
            raise RegistrationError(
                f"{where} [[arms]] {arm.id!r}: {arm.profile} declares a variance_budget over "
                f"{horizon} bars and {registration.shared.risk_model} declares horizon_bars="
                f"{risk_horizon}. The budget bounds `sqrt(w' Sigma w)` under that snapshot, so a "
                "budget stated over a different horizon bounds a different quantity (§3.5, §3.7)"
            )
    distinct = {stable_json_dumps(params) for params in budgets.values()}
    if len(distinct) > 1:
        raise RegistrationError(
            f"{where}: the convex arms declare different variance budgets {budgets}. The budget is "
            "fixed before the run and shared by both convex arms (decision D8); an arm allowed more "
            "ex-ante risk than another is not the same experiment"
        )


def _check_budget_admits_arms(registration: Registration, *, where: str) -> None:
    """``trial_budget`` admits every declared arm; one attempt is one ``arm_id``."""
    if registration.trial_budget < len(registration.arms):
        raise RegistrationError(
            f"{where} declares trial_budget={registration.trial_budget} and {len(registration.arms)} "
            "arms; the budget must admit the arms the registration itself declares, or the last arm is "
            "refused by the registration that asked for it"
        )


def _table_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    """Top-level keys on which two parsed calibration files disagree, nested tables included."""
    keys = set(left) | set(right)
    return {key for key in keys if left.get(key) != right.get(key)}


# --------------------------------------------------------------------------
# Arm composition
# --------------------------------------------------------------------------


def _control_strategy(control: Mapping[str, Any], *, registration: Registration, arm: Arm) -> dict[str, Any]:
    """The arm's ``[strategy]`` block: the control's, checked against the registration."""
    where = f"{registration.path} [benchmark].control {registration.benchmark.control}"
    strategy = control.get("strategy")
    if not isinstance(strategy, Mapping):
        raise RegistrationError(f"{where} declares no [strategy] block")
    components = strategy.get("components")
    if not isinstance(components, list) or not components:
        raise RegistrationError(f"{where} declares no [[strategy.components]]")
    declared = {component.id: component for component in registration.components}
    control_ids = [str(entry.get("id")) for entry in components]
    if sorted(control_ids) != sorted(declared):
        raise RegistrationError(
            f"{where} runs components {sorted(control_ids)} and the registration declares "
            f"{sorted(declared)}. The registration's [[components]] name the control's components; a "
            "set that differs means the arms would not run the book the control runs"
        )
    resolved: list[dict[str, Any]] = []
    for entry in components:
        row = dict(entry)
        component = declared[str(row["id"])]
        if row.get("path") != component.path:
            raise RegistrationError(
                f"{where} runs component {component.id!r} from {row.get('path')!r} and the registration "
                f"declares {component.path!r}. The registration restates the import path and never "
                "overrides it: an arm that runs different code is a different experiment"
            )
        if arm.kind == "convex":
            # The calibrated convex flow consumes the component's scores through
            # the calibration; a rank-to-weight rule would map them to weights
            # before the constructor ever saw them (§3.3).
            row["to_weights"] = "none"
            row.pop("to_weights_params", None)
        else:
            _check_heuristic_restatement(row, arm=arm, component_id=component.id, where=where)
        if arm.component_weights is not None:
            # The arm's declared weight replaces the control's for this
            # component; the parser has already required every declared
            # component to be listed, so there is no partial override.
            row["weight"] = float(arm.component_weights[component.id])
        resolved.append(row)
    out = dict(strategy)
    out["components"] = resolved
    if arm.universe_index is not None:
        # The arm trades another index of the same dataset; everything else
        # about the control's strategy is unchanged.
        out["universe_index"] = arm.universe_index
    out["combination"] = "calibrated_convex" if arm.kind == "convex" else "position_space"
    return out


def _check_heuristic_restatement(row: Mapping[str, Any], *, arm: Arm, component_id: str, where: str) -> None:
    """A heuristic arm restates the control's rank-to-weight rule; it never redefines it."""
    for key, value in arm.construction.items():
        if key == "to_weights":
            if row.get("to_weights") != value:
                raise RegistrationError(
                    f"{arm.id!r} declares to_weights={value!r} and {where} maps component "
                    f"{component_id!r} with {row.get('to_weights')!r}. The control arm is the control's "
                    "own construction: the registration restates it so a reader can see it, and a "
                    "disagreement is refused rather than merged"
                )
            continue
        params = row.get("to_weights_params")
        params = params if isinstance(params, Mapping) else {}
        if key not in params:
            raise RegistrationError(
                f"{arm.id!r} declares {key}={value!r} and {where} declares no such rank-to-weight "
                f"parameter for component {component_id!r}"
            )
        if params[key] != value:
            raise RegistrationError(
                f"{arm.id!r} declares {key}={value!r} and {where} declares {params[key]!r} for "
                f"component {component_id!r}; the registration restates and never overrides"
            )


def _arm_construction(registration: Registration, arm: Arm, *, control: Mapping[str, Any]) -> dict[str, Any]:
    """The arm's ``[construction]`` block: its kind, the shared settings, the shared files."""
    block: dict[str, Any] = {"kind": arm.kind}
    control_construction = control.get("construction")
    control_construction = control_construction if isinstance(control_construction, Mapping) else {}
    # The no-trade band is a construction setting, not a limit, and it changes
    # which trades are taken. The control declares one; carrying it across the
    # arms is what keeps the comparison about the constructor rather than about
    # hysteresis one arm has and another does not.
    if "band" in control_construction:
        block["band"] = control_construction["band"]
    if arm.kind == "heuristic":
        for key, value in control_construction.items():
            if key not in block:
                block[key] = value
    else:
        block["calibration"] = str(arm.calibration)
        block["risk_model"] = str(registration.shared.risk_model)
        block.update(registration.shared.convex_construction)
    block.update(registration.shared.construction)
    return block


def _arm_policy(registration: Registration, arm: Arm, *, control: Mapping[str, Any]) -> dict[str, Any]:
    """The arm's ``[policy]`` block: the arm's profile, the frozen stack, the control's policies."""
    control_policy = control.get("policy")
    if not isinstance(control_policy, Mapping):
        raise RegistrationError(f"{registration.benchmark.control} declares no [policy] block")
    for key in ("monitoring", "recovery"):
        if control_policy.get(key) is None:
            raise RegistrationError(
                f"{registration.benchmark.control} [policy] declares no {key!r}; the arms take the "
                "control's monitoring and recovery policies, and a policy nobody names is a silent default"
            )
    return {
        "profile": str(arm.profile),
        "costs": str(registration.frozen.cost_stack),
        "execution": str(registration.frozen.execution_profile),
        "monitoring": str(control_policy["monitoring"]),
        "recovery": str(control_policy["recovery"]),
    }


# --------------------------------------------------------------------------
# Small parsing helpers
# --------------------------------------------------------------------------


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RegistrationError(f"{path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise RegistrationError(f"{path} cannot be read: {exc}") from exc


def _read_json(path: Path, *, what: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistrationError(
            f"no {what} at {path}; it is written by `prism evaluate <spec> --freeze-scores`, the step "
            "that materializes the score panel the arms share, and a registered run is refused until it "
            "exists (§3.13)"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RegistrationError(f"the {what} at {path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise RegistrationError(f"the {what} at {path} cannot be read: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RegistrationError(f"the {what} at {path} is {type(payload).__name__}, not an object")
    return dict(payload)


def _resolve_file(value: Any, *, root: Path, key: str, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RegistrationError(f"{where} declares {key}={value!r}; it must be a non-empty path string")
    candidate = Path(value).expanduser()
    path = candidate if candidate.is_absolute() else (root / candidate)
    if not path.is_file():
        raise RegistrationError(
            f"{where} names {key}={value!r}, which does not exist at {path}; relative references "
            f"resolve against the workspace root ({root})"
        )
    return path


def _reject_unknown(table: Mapping[str, Any], permitted: Iterable[str], *, where: str) -> None:
    unknown = sorted(set(table) - set(permitted))
    if unknown:
        raise RegistrationError(f"{where}: unknown key(s) {unknown}; declared keys are {sorted(set(permitted))}")


def _require_table(table: Mapping[str, Any], key: str, *, where: str) -> Mapping[str, Any]:
    value = table.get(key)
    if not isinstance(value, Mapping):
        raise RegistrationError(f"{where} declares no [{key}] table")
    return value


def _require_str(table: Mapping[str, Any], key: str, *, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RegistrationError(f"{where} declares {key}={value!r}; it must be a non-empty string")
    return value


def _require_choice(table: Mapping[str, Any], key: str, permitted: Sequence[str], *, where: str) -> str:
    value = _require_str(table, key, where=where)
    if value not in permitted:
        raise RegistrationError(f"{where} declares {key}={value!r}; the declared values are {list(permitted)}")
    return value


def _require_bool(table: Mapping[str, Any], key: str, *, where: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise RegistrationError(f"{where} declares {key}={value!r}; it must be a boolean")
    return value


def _require_int(table: Mapping[str, Any], key: str, *, where: str, minimum: int) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegistrationError(f"{where} declares {key}={value!r}; it must be an integer")
    if value < minimum:
        raise RegistrationError(f"{where} declares {key}={value}; the minimum is {minimum}")
    return value


def _require_date(table: Mapping[str, Any], key: str, *, where: str) -> dt.date:
    value = table.get(key)
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    raise RegistrationError(
        f"{where} declares {key}={value!r}; it must be a bare TOML date (2026-09-08), not a string"
    )
