"""State-coupled decisions: one bar, the account as it stands, one target row (§3.7, §3.8).

The evaluator's stage-2 loop decides every bar of a fold before the first fill, then hands
the whole target panel to an accounting engine. That is sound for a constructor whose
output depends only on prices, and it is *unsound* for the constructor of §3.7: the
participation restriction is trade dollars over dollar volume **from current equity**,
batch rounding is at the per-sid increment against current equity, and the reduce-only box
is written against the marked position. None of the three is evaluable before the account
exists, which is why stage 4's arms decide inside the simulation
(:class:`prism.execution.simulate.DecisionCall`) rather than in front of it.

This module owns one bar of that loop and nothing else. It does not fit, it does not fetch,
it does not account: it takes the account, the scores, the fold's calibration and a risk
snapshot, and returns the target row the simulator submits plus the record §3.12 asks for.

**Both arms come through here, and that is the point.** The decile control does not use the
convex program (§3.7: "the heuristic constructor returns ``solver = None`` and
``path = 'normal'`` always"), but it *does* go through the same limit-evaluation shim, the
same independent post-solve validation and the same batch rounding, so "repair scaled the
batch" means the same thing in the control's record as in a convex arm's. A comparison in
which one arm's costs were rounded and another's were not would be a comparison of
rounding.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.calibration import (
    CalibrationFit,
    CalibrationSpec,
    FoldExclusion,
    fold_exclusion,
    horizon_sigma,
    load_calibration,
    standardize_cross_section,
    standardize_panel,
)
from prism.execution.participation import participation_capped_targets
from prism.portfolio.convex import (
    ComponentForecast,
    ConstructionInputs,
    ConstructionOutput,
    ConvexConstructor,
)
from prism.portfolio.limits import Limit, LimitContext, load_limits
from prism.portfolio.recovery import RecoveryPolicy, finalize_batch
from prism.portfolio.validate import ValidationSettings, validate_targets
from prism.risk_model import RiskPolicy, RiskSnapshot, build_snapshot, validate_snapshot
from prism.spec import Specification

__all__ = [
    "ArmPolicies",
    "DecisionError",
    "DecisionOutcome",
    "FoldCalibration",
    "calibrate_fold",
    "decide_bar",
    "usable_periods",
]


class DecisionError(ValueError):
    """The bar cannot be decided as asked, and deciding it anyway would forge the record."""


# ---------------------------------------------------------------------------------
# The policy stack one arm runs under
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmPolicies:
    """Every declared file one arm's constructor reads, parsed once per run.

    The profile's ``[[limits]]`` rows, the execution profile's ``[validation]`` and
    ``[rounding]`` blocks, the recovery policy, and (on a convex arm) the calibration and
    risk-model files. They are loaded once because they are frozen for the run: a limit
    that changed mid-run would make the recorded ``config_id`` a description of nothing.
    """

    profile_path: Path
    profile: Mapping[str, Any]
    execution_path: Path
    execution: Mapping[str, Any]
    recovery_path: Path
    recovery: RecoveryPolicy
    limits: tuple[Limit, ...]
    limit_notes: tuple[str, ...]
    validation: ValidationSettings
    cash_buffer: float
    min_regime_scale: float | None
    kind: str
    calibration_path: Path | None = None
    calibration: CalibrationSpec | None = None
    risk_path: Path | None = None
    risk_policy: RiskPolicy | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: Specification) -> "ArmPolicies":
        """Resolve the policy stack of ``spec``; every path is the specification's own."""
        profile = _read_toml(spec.policy.profile)
        execution = _read_toml(spec.policy.execution)
        recovery_raw = _read_toml(spec.policy.recovery)
        recovery = RecoveryPolicy.from_mapping(recovery_raw, source=str(spec.policy.recovery))
        limits, notes = load_limits(profile.get("limits") or (), source=str(spec.policy.profile))
        validation = ValidationSettings.from_profile(execution, recovery_policy=recovery_raw)
        kind = str(spec.construction.kind)
        calibration_spec = None
        risk_policy = None
        if spec.construction.calibration is not None:
            calibration_spec = load_calibration(spec.construction.calibration)
        if spec.construction.risk_model is not None:
            risk_policy = RiskPolicy.from_toml(spec.construction.risk_model)
        if kind == "convex" and (calibration_spec is None or risk_policy is None):
            raise DecisionError(
                f"{spec.path} declares construction.kind='convex' with "
                f"calibration={spec.construction.calibration!r} and risk_model={spec.construction.risk_model!r}; "
                "the convex constructor reads both files and neither is defaulted (§3.7)"
            )
        return cls(
            profile_path=Path(spec.policy.profile),
            profile=profile,
            execution_path=Path(spec.policy.execution),
            execution=execution,
            recovery_path=Path(spec.policy.recovery),
            recovery=recovery,
            limits=limits,
            limit_notes=notes,
            validation=validation,
            cash_buffer=float(profile.get("cash_buffer", 0.0) or 0.0),
            min_regime_scale=(None if profile.get("min_regime_scale") is None else float(profile["min_regime_scale"])),
            kind=kind,
            calibration_path=spec.construction.calibration,
            calibration=calibration_spec,
            risk_path=spec.construction.risk_model,
            risk_policy=risk_policy,
            settings=dict(spec.construction.options),
        )

    def limit(self, name: str) -> Limit | None:
        """The declared limit called ``name``, or ``None`` when the profile declares none."""
        for limit in self.limits:
            if limit.name == name:
                return limit
        return None

    def declaration(self) -> dict[str, Any]:
        """What the record carries about the files this arm ran under (§3.12)."""
        return {
            "profile": str(self.profile_path),
            "execution_profile": str(self.execution_path),
            "recovery_policy": str(self.recovery_path),
            "limits": [
                {
                    "name": limit.name,
                    "class": limit.cls.value,
                    "priority": limit.priority,
                    "params": dict(limit.params),
                }
                for limit in self.limits
            ],
            "ignored_limit_keys": list(self.limit_notes),
            "validation": {
                "limit_tolerance": self.validation.limit_tolerance,
                "cash_tolerance": self.validation.cash_tolerance,
                "cost_tolerance_bps": self.validation.cost_tolerance_bps,
                "mode": self.validation.mode,
                "preference_response": self.validation.preference_response,
                "bound_slack": self.validation.bound_slack,
            },
            "recovery": self.recovery.declaration(),
            "cash_buffer": self.cash_buffer,
            "min_regime_scale": self.min_regime_scale,
            "calibration": None if self.calibration_path is None else str(self.calibration_path),
            "risk_model": None if self.risk_path is None else str(self.risk_path),
            "construction": dict(self.settings),
        }


def _read_toml(path: Path | str) -> dict[str, Any]:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


#: The holding-state limits whose measure reads a §3.5 snapshot. A path that carries no
#: snapshot cannot measure them, and §3.7's table says so in as many words: on the
#: heuristic path "net_band, beta_band, variance_budget, turnover_cap ... not enforced --
#: no risk snapshot and no post-band check enter this path". They are recorded as
#: unchecked rather than silently reported satisfied, which is the defect the limit module
#: refuses outright.
RISK_MEASURED_LIMITS: tuple[str, ...] = ("variance_budget", "beta_band")


def measurable_limits(limits: Sequence[Limit], risk: Any) -> tuple[tuple[Limit, ...], tuple[str, ...]]:
    """``(limits that can be measured here, names of those that cannot)``."""
    if risk is not None:
        return tuple(limits), ()
    keep = tuple(limit for limit in limits if limit.name not in RISK_MEASURED_LIMITS)
    dropped = tuple(limit.name for limit in limits if limit.name in RISK_MEASURED_LIMITS)
    return keep, dropped


# ---------------------------------------------------------------------------------
# Fold admission and the shared-slope fit (§3.4, decision D6)
# ---------------------------------------------------------------------------------


def usable_periods(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    sigma_h: pd.DataFrame,
    spec: CalibrationSpec,
) -> int:
    """The count of decision periods that carry at least one complete observation.

    This is ``CalibrationFit.n_periods`` computed without fitting a slope, and it is what
    the decile control arm's admission reads (decision D6): the control fits nothing, so it
    must not compute its own admission rule, and the three arms must exclude the same
    folds. ``tests/test_decisions.py`` pins this against ``fit_calibration`` on the same
    panels, because a divergence here would drop a fold for one arm and keep it for
    another with nothing in the record saying so.
    """
    standardized = standardize_panel(scores, spec.standardize)
    z = standardized.to_numpy(dtype=float)
    r = returns.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    sigma = sigma_h.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    complete = np.isfinite(z) & np.isfinite(r) & np.isfinite(sigma) & (sigma > 0.0)
    return int((complete.sum(axis=1) > 0).sum())


@dataclass(frozen=True)
class FoldCalibration:
    """One fold's calibration: its admission, and the fit each component's arm consumes."""

    fold: int
    excluded: bool
    exclusion_reason: str | None
    exclusion: FoldExclusion
    n_periods: int
    fit_cutoff: str
    fits: Mapping[str, CalibrationFit] = field(default_factory=dict)
    panels: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "fold": int(self.fold),
            "excluded": bool(self.excluded),
            "exclusion_reason": self.exclusion_reason,
            "n_periods": int(self.n_periods),
            "fit_cutoff": self.fit_cutoff,
            "admission": {
                "excluded": bool(self.exclusion.excluded),
                "reason": self.exclusion.reason,
                "n_periods": int(self.exclusion.n_periods),
                "blocks_drawn": int(self.exclusion.blocks_drawn),
                "min_training_periods": int(self.exclusion.min_training_periods),
                "min_blocks": int(self.exclusion.min_blocks),
            },
            "periods": dict(self.panels),
        }
        record["fits"] = {cid: fit.to_record() for cid, fit in self.fits.items()}
        return record


def calibrate_fold(
    *,
    fold: int,
    component_id: str,
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    sigma_h: pd.DataFrame,
    family: CalibrationSpec,
    arm_spec: CalibrationSpec | None,
    fit_cutoff: str,
    horizon_bars: int,
    update_every_bars: int,
) -> FoldCalibration:
    """Admit or exclude the fold, and fit the arm's slope when it is admitted.

    ``family`` is the registration's ``[shared] calibration_family``, which decides
    admission for every arm alike; ``arm_spec`` is the arm's own file and is ``None`` for
    an arm that fits no slope. The two differ only in ``confidence`` (the registration's
    loader refuses anything else), so the admission a convex arm's own fit reports and the
    admission read off the family here are the same decision (D6).
    """
    n_periods = usable_periods(scores, returns, sigma_h, family)
    exclusion = fold_exclusion(n_periods, family)
    periods = {
        "n_periods_supplied": int(scores.shape[0]),
        "n_periods_used": int(n_periods),
        "first_period": None if not len(scores.index) else str(scores.index[0]),
        "last_period": None if not len(scores.index) else str(scores.index[-1]),
    }
    if exclusion.excluded or arm_spec is None:
        return FoldCalibration(
            fold=fold,
            excluded=bool(exclusion.excluded),
            exclusion_reason=exclusion.reason,
            exclusion=exclusion,
            n_periods=n_periods,
            fit_cutoff=fit_cutoff,
            fits={},
            panels=periods,
        )
    from prism.calibration import (
        fit_calibration,  # local: keeps the import graph of the control arm's path small
    )

    fit = fit_calibration(
        scores,
        returns,
        sigma_h,
        arm_spec,
        fit_cutoff=fit_cutoff,
        horizon_bars=horizon_bars,
        update_every_bars=update_every_bars,
    )
    if int(fit.n_periods) != int(n_periods):
        raise DecisionError(
            f"fold {fold}: the arm's fit counted {fit.n_periods} usable periods and the shared family counted "
            f"{n_periods}. The two read the same panels and must agree, or a fold would be dropped for one arm "
            "and kept for another (D6, §3.4)"
        )
    if bool(fit.excluded) != bool(exclusion.excluded):
        raise DecisionError(
            f"fold {fold}: the arm's fit and the shared family disagree on admission "
            f"({fit.excluded} against {exclusion.excluded}); admission is shared by declaration (D6)"
        )
    return FoldCalibration(
        fold=fold,
        excluded=bool(fit.excluded),
        exclusion_reason=fit.exclusion_reason,
        exclusion=exclusion,
        n_periods=n_periods,
        fit_cutoff=fit_cutoff,
        fits={component_id: fit},
        panels=periods,
    )


# ---------------------------------------------------------------------------------
# One decision bar
# ---------------------------------------------------------------------------------


def _aligned_holding(values: Any, index: pd.Index, *, name: str) -> pd.Series:
    """The caller's holding vector on the constructor universe, or a refusal.

    A sid the caller does not mention is not held, so 0.0 is its weight and not an
    imputation. A sid the caller *does* mention as NaN is a different thing: it is a name
    whose holding nobody could compute, and reading it as flat would let the program open
    a position against a book it cannot see. The house rule is that a silent zero is a
    defect, so the second case raises where the first fills (SPEC.md §1; §3.7,
    "a name with no marked weight is flat only if the caller says so").
    """
    series = pd.Series(values, dtype=float)
    series.index = pd.Index([str(sid) for sid in series.index])
    if series.index.has_duplicates:
        offenders = sorted({str(sid) for sid in series.index[series.index.duplicated()]})
        raise DecisionError(
            f"{name} names {offenders[:5]} more than once; a holding vector with a duplicated sid has no "
            "unambiguous weight per name"
        )
    stated = series.reindex(index)
    supplied = index.isin(series.index)
    unusable = sorted(
        {
            str(sid)
            for sid, seen, value in zip(index, supplied, stated.to_numpy(dtype=float))
            if seen and not np.isfinite(value)
        }
    )
    if unusable:
        raise DecisionError(
            f"{name} carries a non-finite value for {unusable[:5]}, which the constructor universe "
            "holds. A name the caller omits carries 0.0 because it is not held; a name the caller states as "
            "NaN is a holding nobody could compute, and reading it as flat would let the program trade "
            "against a book it cannot see"
        )
    return stated.fillna(0.0)


@dataclass
class DecisionOutcome:
    """What one decision bar produced: the row the simulator submits, and the record."""

    row: pd.Series | None
    construction: ConstructionOutput | None
    rounding: Any | None
    record: dict[str, Any]


def decide_bar(
    *,
    decision_bar: str,
    policies: ArmPolicies,
    state: Any,
    equity: float,
    prices: pd.Series,
    sids: Sequence[str],
    w0: pd.Series,
    positions: pd.Series,
    adv_dollars: pd.Series,
    costs: Any,
    heuristic_target: pd.Series | None = None,
    forecasts: Sequence[ComponentForecast] = (),
    risk: RiskSnapshot | None = None,
    component_outputs: Mapping[str, Any] | None = None,
    fold: int | None = None,
) -> DecisionOutcome:
    """Construct, validate, round and revalidate one decision bar.

    Returns ``row = None`` when nothing is submitted -- the no-action path, a dropped
    batch, or an empty batch -- which the simulator reads as "hold every position", never
    as "go flat". A silent zero here would liquidate the book on a bar the constructor
    declined to trade, which §3.7 refuses in as many words.
    """
    index = pd.Index([str(sid) for sid in sids])
    w0 = _aligned_holding(w0, index, name="w0")
    positions = _aligned_holding(positions, index, name="positions")
    ctx = LimitContext(w0=w0, equity=float(equity), adv_dollars=adv_dollars.reindex(index), risk=risk)
    limits, unchecked = measurable_limits(policies.limits, risk)

    if policies.kind == "heuristic":
        output = _heuristic_construction(
            heuristic_target,
            policies=policies,
            limits=limits,
            unchecked=unchecked,
            ctx=ctx,
            index=index,
            adv_dollars=adv_dollars,
            equity=float(equity),
            decision_bar=decision_bar,
            risk=risk,
            costs=costs,
        )
    else:
        inputs = ConstructionInputs(
            sids=list(index),
            w0=w0,
            equity=float(equity),
            reference_prices=prices.reindex(index),
            limits=limits,
            costs=costs,
            decision_bar=decision_bar,
            adv_dollars=adv_dollars.reindex(index),
            forecasts=tuple(forecasts),
            risk=risk,
            positions=positions,
            component_outputs=dict(component_outputs or {}),
            settings=dict(policies.settings),
            cash_buffer=policies.cash_buffer,
            min_regime_scale=policies.min_regime_scale,
            policy=policies.recovery,
            validation=policies.validation,
            calibration_spec=policies.calibration,
            risk_policy=policies.risk_policy,
        )
        output = ConvexConstructor(policy=policies.recovery).construct(inputs)

    record: dict[str, Any] = {
        "decision_bar": decision_bar,
        "fold": fold,
        "equity": float(equity),
        "n_names": int(len(index)),
        "path": output.path,
        "binding_condition": output.binding_condition,
        "solver": output.solver,
        "violations": output.violations,
        "unchecked_limits": list(unchecked),
        "diagnostics": output.diagnostics,
    }
    if output.path == "no_action":
        record["rounding"] = None
        record["repair"] = {"response": "no_action", "n_intents": 0}
        return DecisionOutcome(row=None, construction=output, rounding=None, record=record)

    rounded = finalize_batch(
        output,
        state=state,
        prices=prices.reindex(index),
        execution_profile=policies.execution,
        limits=limits,
        policy=policies.recovery,
        decision_bar=decision_bar,
        adv_dollars=adv_dollars.reindex(index),
        risk=risk,
        cost_model=costs,
        validation=policies.validation,
    )
    record["rounding"] = rounded.to_dict()
    record["repair"] = {
        "response": rounded.response,
        "scale_applied": float(rounded.scale_applied),
        "n_intents": int(len(rounded.intents)),
        "n_repair_moves": int(len(rounded.repair_moves)),
        "implemented_gross_fraction": float(rounded.implemented_gross_fraction),
        "not_implementable": bool(rounded.not_implementable),
        "binding_condition": rounded.binding_condition,
        "validation_ok": bool(rounded.validation.get("ok", True)),
    }
    row = _row_from_intents(rounded.intents, positions=positions, prices=prices, equity=float(equity))
    return DecisionOutcome(row=row, construction=output, rounding=rounded, record=record)


def _row_from_intents(
    intents: Sequence[Any],
    *,
    positions: pd.Series,
    prices: pd.Series,
    equity: float,
) -> pd.Series | None:
    """The submitted intents as the weight row the simulator sizes against.

    The simulator sizes ``target_shares = w * equity / price`` and trades the rounded
    difference from what the account holds, so a row built as
    ``(held + qty) * price / equity`` at *this bar's* equity and prices reproduces exactly
    the share quantities batch rounding already validated. Names with no intent are absent
    from the row, which the simulator reads as "no decision, hold the position" -- so a
    name the batch declined to trade is held rather than sold.
    """
    if not intents:
        return None
    values: dict[str, float] = {}
    for intent in intents:
        sid = str(intent.sid)
        price = float(prices.get(sid, float("nan")))
        if not math.isfinite(price) or price <= 0.0:
            raise DecisionError(
                f"{sid}: an intent was sized at a reference price of {price!r}; rounding priced it and the "
                "simulator cannot, which would submit a quantity nobody sized"
            )
        held = float(positions.get(sid, 0.0))
        values[sid] = (held + float(intent.qty)) * price / float(equity)
    return pd.Series(values, dtype=float)


def _heuristic_construction(
    target: pd.Series | None,
    *,
    policies: ArmPolicies,
    limits: Sequence[Limit],
    unchecked: Sequence[str],
    ctx: LimitContext,
    index: pd.Index,
    adv_dollars: pd.Series,
    equity: float,
    decision_bar: str,
    risk: RiskSnapshot | None,
    costs: Any,
) -> ConstructionOutput:
    """The decile control's target, participation-bounded, through the §3.7 shim.

    The shim evaluates every declared holding-state limit on ``w0`` and on the produced
    target and fills ``violations`` and ``binding_condition`` without changing the target;
    it is not a repair (§3.7). What *is* enforced here is the participation restriction,
    because that is what the live path enforces and where it enforces it
    (``execution/participation.py::participation_capped_targets``, called after the
    construct); an action restriction is never merely recorded.
    """
    if target is None:
        raise DecisionError("the heuristic arm was asked to decide a bar with no constructed target")
    proposed = pd.Series(target, dtype=float).reindex(index)
    # A held name the rank-to-weight rule gave no value carries no decision: under
    # `eligibility_exit = "reduce_only"` a held name that has left the tradeable set is
    # pinned reduce-only rather than exited, so it is held here rather than forced flat
    # on a bar the exit may not be executable (§3.7, the stage 3 correction).
    proposed = proposed.fillna(ctx.w0)
    limit = policies.limit("participation_per_name")
    if limit is not None:
        proposed = participation_capped_targets(
            ctx.w0,
            proposed,
            adv_dollars.reindex(index),
            aum=float(equity),
            max_participation=float(limit.value()),
            adv_floor=0.0,
        ).reindex(index)
    report = validate_targets(
        proposed,
        limits,
        ctx,
        settings=policies.validation,
        attributed_to="heuristic",
    )
    values = proposed.to_numpy(dtype=float)
    trade = values - ctx.w0.to_numpy(dtype=float)
    cap = policies.limit("max_name_weight")
    tolerance = policies.validation.limit_tolerance
    at_cap = None
    if cap is not None:
        held = np.abs(values) > tolerance
        at_cap = float((np.abs(values) >= float(cap.value()) - tolerance).sum() / held.sum()) if held.sum() else 0.0
    diagnostics: dict[str, Any] = {
        "decision_bar": decision_bar,
        "n_names": int(len(index)),
        "at_cap_fraction": at_cap,
        "max_at_cap_fraction": policies.settings.get("max_at_cap_fraction"),
        "gross": float(np.abs(values).sum()),
        "net": float(values.sum()),
        "turnover": float(np.abs(trade).sum()),
        "n_held": int((np.abs(values) > tolerance).sum()),
        "validation": report.to_dict(),
        "unchecked_limits": list(unchecked),
        "eligibility_exit": policies.settings.get("eligibility_exit"),
        "shim": (
            "the heuristic path evaluates every holding-state limit and repairs none (§3.7); the "
            "participation restriction is enforced, because an action restriction is never merely recorded"
        ),
    }
    if risk is not None:
        diagnostics["ex_ante_sd"] = math.sqrt(max(float(risk.variance(proposed)), 0.0))
        diagnostics["risk_model"] = risk.record()
        diagnostics["substitutions"] = risk.substitutions
        diagnostics["reduce_only"] = list(risk.reduce_only)
    if costs is not None:
        terms = costs.convex_terms(list(index), adv_dollars=adv_dollars.reindex(index))
        diagnostics["trade_cost_fraction"] = float(terms.trade_costs(trade))
        diagnostics["holding_cost_fraction"] = float(terms.holding_costs(values))
        diagnostics["modeled_cost_dollars"] = float(terms.total_dollars(trade, values))
    return ConstructionOutput(
        w_target=proposed,
        violations={key: [dict(entry) for entry in value] for key, value in report.violations.items()},
        path="normal",
        solver=None,
        binding_condition=report.binding_condition,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------------
# The risk snapshot and the decision-bar forecast
# ---------------------------------------------------------------------------------


def snapshot_at(
    *,
    returns: pd.DataFrame,
    policy: RiskPolicy,
    decision_bar: str,
    universe: Sequence[str],
    held: Sequence[str] = (),
    eligible: Mapping[str, bool] | pd.Series | None = None,
    valuation_status: Mapping[str, str] | None = None,
    halted: Sequence[str] = (),
) -> RiskSnapshot:
    """Build and validate one decision bar's snapshot; a rejected snapshot raises."""
    snapshot = build_snapshot(
        returns,
        policy,
        as_of=decision_bar,
        universe=list(universe),
        held=list(held),
        eligible=eligible,
        valuation_status=valuation_status,
        halted=list(halted),
    )
    validate_snapshot(snapshot, universe=list(universe), policy=policy).require()
    return snapshot


def forecast_at(
    *,
    component_id: str,
    fit: CalibrationFit,
    raw: pd.Series,
    prices: pd.DataFrame,
    spec: CalibrationSpec,
    horizon_bars: int,
) -> ComponentForecast:
    """The decision bar's standardized cross-section and its horizon volatilities.

    ``z`` is standardized by the rule the fit used and ``sigma_h`` estimated by the
    ``[sigma_h]`` block the fit used, on the same index -- the constructor forms
    ``a = z * sigma_h`` itself. A name with no finite score, or no positive volatility, is
    absent from both: it is given ``a_i = 0`` inside the program rather than carried in as
    a NaN (§3.4, §3.7).
    """
    sigma = horizon_sigma(prices, spec.sigma_h, horizon_bars).iloc[-1]
    scores = pd.Series(raw, dtype=float)
    usable = scores.index[np.isfinite(scores.to_numpy(dtype=float))]
    sigma = sigma.reindex(usable)
    keep = sigma.index[np.isfinite(sigma.to_numpy(dtype=float)) & (sigma.to_numpy(dtype=float) > 0.0)]
    z = standardize_cross_section(scores.reindex(keep), spec.standardize, label=component_id)
    return ComponentForecast(component_id=component_id, fit=fit, z=z, sigma_h=sigma.reindex(keep))
