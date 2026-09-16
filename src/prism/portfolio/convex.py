"""The convex constructor: one program, the declared objective, and the recovery path (§3.7).

**What this module is.** The second of the two constructors §3.7 names. The heuristic one
wraps ``portfolio/construct.py`` and enforces caps before a band that can breach them; this
one states every action restriction and every holding-state limit *inside* one convex
program, where no post-processing step can breach them, and hands the solved target to an
independent validator that never reads the solver's status.

**The objective, stated once and not paraphrased.**

    maximize  min(lower * s, upper * s)
              - trade_costs(w - w0) - holding_costs(w)
              - risk_aversion * variance(w) - eps * ||w||_2^2                with s = a' w

It is a concave maximization over a convex set. The robust term is the pointwise minimum of
two linear functions of ``w``, hence concave. The cost terms ``CostModel.convex_terms``
returns are linear, quadratic and power-1.5 in the absolute traded weight, hence convex, and
enter negated. ``variance`` and ``||w||_2^2`` are convex and enter negated. Every constraint
of the §3.7 limit table is convex, as is the reduce-only box. Writing the same objective as
a minimization of the negated terms is the identical program; writing it as a minimization
of the terms as printed is a different, non-concave program that CVXPY refuses.

**The two units contracts, both of which are silent when broken.**

``a`` is ``z * sigma_h``, formed by :func:`prism.calibration.apply_calibration` and never
here: ``theta`` is dimensionless because the regressand was an executable return divided by
a volatility at the same horizon, so an objective on ``z`` alone is wrong by a factor of
order ``sigma_h`` and would re-weight the cross-section by inverse volatility with nothing
looking broken. And the endpoints come from
:func:`prism.calibration.objective_endpoints`, never off ``fit.lower`` / ``fit.upper``:
that function is what applies the §3.4 sign policy by returning ``(0.0, 0.0)`` on a sign
contradiction, so the arm takes no new directional exposure while the evaluation continues.

**The straddling interval, which is a result and not a bug.** ``min(lower*s, upper*s)``
equals ``lower*s`` for ``s >= 0`` and ``upper*s`` for ``s < 0``. When ``lower < 0 < upper``
the term is at most zero for every ``s`` and is zero only at ``s = 0``, so the adverse
objective penalizes aggregate signal exposure in *both* directions and its maximum sets
``a' w = 0``. With any positive trade cost the cheapest way to hold ``a' w = 0`` is to hold
nothing at all. On a decision date whose 80% one-sided lower endpoint is negative the
uncertainty arm therefore goes flat, stays enabled, and is *not* a defect to patch. Every
such date is recorded: :attr:`ConstructionOutput.diagnostics` carries ``straddles_zero``
per component, the achieved ``s``, the resulting gross, and the flag
``flat_from_straddling_endpoints``, so a flat arm shows in the comparison as what it is --
a finding about the calibration's precision.

**cvxpy stays out of the core import path.** It belongs to the declared ``construction``
extra, so it is imported inside :class:`ConvexProgram`'s methods, never at module scope, and
``portfolio/__init__.py`` does not import this module. The import-hygiene probe in
``tests/test_import_hygiene.py`` imports every module under ``prism`` in a subprocess with a
controlled path; a module-level ``import cvxpy`` would make that probe fail in any install
without the extra.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.calibration import (
    CalibrationFit,
    CalibrationSpec,
    apply_calibration,
    objective_endpoints,
    require_sigma_agreement,
)
from prism.portfolio.limits import (
    MEASURED_LIMITS,
    Limit,
    LimitContext,
    binding_condition,
    is_action_restriction,
    is_holding_state,
    partition_violations,
)
from prism.portfolio.obligations import Obligation
from prism.portfolio.recovery import (
    RecoveryPolicy,
    RecoveryResult,
    no_action,
    recovery_entry,
    run_recovery,
)
from prism.portfolio.validate import ValidationSettings, validate_targets

__all__ = [
    "ACTION_RESTRICTION_NAMES",
    "ConstructionError",
    "ComponentForecast",
    "ConstructionInputs",
    "ConstructionOutput",
    "ConvexProgram",
    "ConvexConstructor",
    "SolveOutcome",
    "SOLVER_TOLERANCES",
]


class ConstructionError(ValueError):
    """A construction input, declaration, or solver outcome that must not be repaired silently."""


#: The action restrictions this constructor knows how to write as a convex constraint, plus
#: the two it deliberately does not: ``quantity_increment`` and ``permitted_order_types``
#: are not expressions on a continuous weight and are enforced by ``portfolio/rounding.py``
#: and by the submission revalidation. A restriction outside this set is **refused**, never
#: skipped: silently ignoring an action restriction is how an engine submits an order the
#: venue forbids.
ACTION_RESTRICTION_NAMES: frozenset[str] = frozenset(
    {
        "participation_per_name",
        "halted_no_trade",
        "reduce_only_ineligible",
        "no_new_short_without_borrow",
        "account_prohibits_shorting",
        "buying_power",
        "regulatory_restriction",
        "quantity_increment",
        "permitted_order_types",
    }
)

#: Action restrictions enforced downstream, recorded here rather than solved here.
_DEFERRED_ACTION_RESTRICTIONS: frozenset[str] = frozenset({"quantity_increment", "permitted_order_types"})

#: The program is held to a slightly tighter book than the declared limit: each hard bound
#: is pulled in by this fraction of the limit's own ``limit_tolerance``. The reason is that
#: the independent validation, not the solver, is the authority, and an interior-point
#: solution sits *at* an active constraint to within the solver's feasibility tolerance --
#: which, summed over a few thousand names in a measure like ``max_gross``, can land just
#: outside a tolerance of 1e-6 and send a perfectly good batch to the no-action path. Half
#: the declared tolerance is a margin the profile itself calls immaterial, and it makes the
#: validated threshold reachable rather than marginal. It is applied to the *bound*, never
#: to the measure: ``portfolio/limits.py`` still measures against the declared value.
_CONSTRAINT_MARGIN_FRACTION: float = 0.5

#: CLARABEL's declared tolerances, recorded in the solver block of every result so that a
#: tolerance-driven outcome is readable rather than inferred.
SOLVER_TOLERANCES: dict[str, float] = {
    "tol_gap_abs": 1.0e-9,
    "tol_gap_rel": 1.0e-9,
    "tol_feas": 1.0e-9,
}


@dataclass(frozen=True)
class ComponentForecast:
    """One component's decision-bar forecast, as the constructor consumes it (§3.4, §3.7).

    ``fit`` is the fold's :class:`~prism.calibration.CalibrationFit`; ``z`` the standardized
    cross-section at the decision bar, produced by
    :func:`prism.calibration.standardize_cross_section`; ``sigma_h`` the per-name horizon
    volatility on the same index, under the same ``[sigma_h]`` declaration the fit used.
    The constructor forms ``a = z * sigma_h`` through
    :func:`prism.calibration.apply_calibration` -- it never multiplies by ``sigma`` itself
    and never consumes ``z`` alone.

    A name the component does not score is simply absent from ``z``; it is given ``a_i = 0``
    explicitly in the program, after the call, rather than carried into it as a NaN.
    """

    component_id: str
    fit: CalibrationFit
    z: pd.Series
    sigma_h: pd.Series
    weight: float = 1.0
    disabled: bool = False
    disabled_reason: str | None = None

    def __post_init__(self) -> None:
        if not str(self.component_id).strip():
            raise ConstructionError("a component forecast needs a component_id")
        weight = float(self.weight)
        if not math.isfinite(weight):
            raise ConstructionError(f"{self.component_id}: weight must be finite, got {self.weight!r}")
        object.__setattr__(self, "weight", weight)
        z = pd.Series(self.z, dtype=float)
        sigma = pd.Series(self.sigma_h, dtype=float)
        if not z.index.equals(sigma.index):
            raise ConstructionError(
                f"{self.component_id}: z and sigma_h must share one index; a silent reindex misprices names"
            )
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "sigma_h", sigma)


@dataclass
class ConstructionInputs:
    """Everything one decision bar hands the constructor (§3.7).

    Two departures from the section's prose sketch, both recorded in the module's tests:
    ``forecasts`` replaces the ``component_outputs`` / ``calibration_fits`` pair, because the
    constructor must form ``a`` itself through
    :func:`prism.calibration.apply_calibration` and therefore needs ``z`` and ``sigma_h``
    alongside the fit; and ``positions`` is accepted rather than derived, because obligation
    quantities are share-denominated and deriving shares from ``w0 * equity / price`` is
    exact only under this bar's own marks. ``component_outputs`` is still carried and
    recorded verbatim.

    Two fields are carried for the record and deliberately not consumed. ``account_target``
    is the F1 path's weighted sum of sub-targets; this constructor derives its target from
    the objective, so accepting one and then solving past it would leave two answers in the
    record with nothing saying which was implemented. ``regime_scale`` is recorded next to
    ``min_regime_scale`` because §3.7 states that *the constructor receives the scaled limit
    values*: scaling the limits is the caller's step, and doing it again here would apply
    the de-gross twice.
    """

    sids: Sequence[str]
    w0: pd.Series
    equity: float
    reference_prices: pd.Series
    limits: Sequence[Limit]
    costs: Any
    decision_bar: str
    adv_dollars: pd.Series | None = None
    forecasts: Sequence[ComponentForecast] = ()
    risk: Any = None
    obligations: Sequence[Obligation] = ()
    positions: pd.Series | None = None
    account_target: pd.Series | None = None
    component_outputs: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    regime_scale: float | None = None
    min_regime_scale: float | None = None
    cash_buffer: float = 0.0
    policy: RecoveryPolicy | None = None
    validation: ValidationSettings | None = None
    calibration_spec: CalibrationSpec | None = None
    risk_policy: Any = None
    capabilities: Any = None

    def __post_init__(self) -> None:
        sids = tuple(str(sid) for sid in self.sids)
        if not sids:
            raise ConstructionError("the constructor universe is empty; there is no cross-section to solve")
        if len(set(sids)) != len(sids):
            duplicates = sorted({sid for sid in sids if sids.count(sid) > 1})
            raise ConstructionError(f"the constructor universe repeats {duplicates}; sids fixes the order of w")
        object.__setattr__(self, "sids", sids)
        equity = float(self.equity)
        if not math.isfinite(equity) or equity <= 0.0:
            raise ConstructionError(f"equity must be finite and > 0, got {self.equity!r}")
        self.equity = equity
        w0 = pd.Series(self.w0, dtype=float).reindex(pd.Index(sids))
        if w0.isna().any():
            missing = [sid for sid in sids if pd.isna(w0.get(sid))]
            raise ConstructionError(
                f"the marked weights cover no value for {missing[:5]}; a name in the universe with no marked "
                "weight is flat only if the caller says so, never by imputation here"
            )
        self.w0 = w0
        self.reference_prices = pd.Series(self.reference_prices, dtype=float)
        if self.adv_dollars is not None:
            self.adv_dollars = pd.Series(self.adv_dollars, dtype=float)
        if self.positions is not None:
            self.positions = pd.Series(self.positions, dtype=float)


@dataclass(frozen=True)
class SolveOutcome:
    """One solver call's outcome. ``optimal`` is this module's verdict, not CLARABEL's."""

    status: str
    optimal: bool
    value: float | None
    w: pd.Series | None
    solve_seconds: float
    compile_seconds: float
    iterations: int | None = None
    note: str | None = None


@dataclass
class ConstructionOutput:
    """What the constructor returns (§3.7).

    ``w_target`` carries ``NaN`` for no decision on a sid -- never a silent zero, which the
    downstream band would read as "go flat". ``violations`` is the
    ``{existing, remaining, introduced}`` partition, each entry carrying ``limit_name``,
    ``class``, ``priority``, the measure ``v`` and ``attributed_to``.
    """

    w_target: pd.Series
    violations: dict[str, list[dict[str, Any]]]
    path: Literal["normal", "recovery", "no_action"]
    solver: dict[str, Any] | None
    binding_condition: str | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """The JSON-safe record."""
        return {
            "w_target": {str(sid): (None if pd.isna(value) else float(value)) for sid, value in self.w_target.items()},
            "violations": {key: [dict(entry) for entry in value] for key, value in self.violations.items()},
            "path": self.path,
            "solver": None if self.solver is None else dict(self.solver),
            "binding_condition": self.binding_condition,
            "diagnostics": dict(self.diagnostics),
        }


# --------------------------------------------------------------------------
# The program
# --------------------------------------------------------------------------


class ConvexProgram:
    """One decision bar's convex program: the variable, the hard restrictions, the terms.

    The class is deliberately expression-shaped rather than problem-shaped. Recovery needs
    the *same* variable and the *same* action restrictions across four or more solves with
    different objectives and accumulating bounds, and a class that returned finished
    problems could not offer that. Each :meth:`solve` compiles one problem for the shape it
    is handed; the universe dimension is never fixed at import time.
    """

    def __init__(self, inputs: ConstructionInputs, *, limit_tolerance: float = 1.0e-6) -> None:
        self.inputs = inputs
        self.limit_tolerance = float(limit_tolerance)
        self.sids: tuple[str, ...] = tuple(inputs.sids)
        self.n = len(self.sids)
        self.index = pd.Index(self.sids, name="sid")
        self.equity = float(inputs.equity)
        self.w0 = inputs.w0.reindex(self.index).to_numpy(dtype=float)
        if not np.isfinite(self.w0).all():
            raise ConstructionError("the marked weights carry a non-finite entry; there is nothing to solve against")
        settings = dict(inputs.settings or {})
        self.settings = settings
        self.risk_aversion = float(settings.get("risk_aversion", 0.0))
        self.eps = float(settings.get("eps", 0.0))
        if self.risk_aversion < 0.0 or not math.isfinite(self.risk_aversion):
            raise ConstructionError(f"risk_aversion must be finite and >= 0, got {self.risk_aversion!r}")
        if self.eps < 0.0 or not math.isfinite(self.eps):
            raise ConstructionError(f"eps must be finite and >= 0, got {self.eps!r}")
        self.eligibility_exit = str(settings.get("eligibility_exit", "reduce_only"))
        if self.eligibility_exit != "reduce_only":
            raise ConstructionError(
                f"eligibility_exit={self.eligibility_exit!r} is not a convex-constructor value. "
                "`exit` reproduces the recorded decile behavior, which assigns an explicit 0.0 to an "
                "ineligible held name; against a hard participation restriction that target is frequently "
                "infeasible, and the whole batch would fall to the no-action path. The correction of §3.7 is "
                "`reduce_only`; the reproduction path is the heuristic constructor (portfolio/construct.py)"
            )
        band = dict(settings.get("band", {}) or {})
        band_mode = str(band.get("mode", "off"))
        if band_mode != "off":
            raise ConstructionError(
                f"[construction.band] mode={band_mode!r} is refused by the convex constructor. The no-trade "
                "band is a post-construction step that rebalances a name after the caps, which is exactly how "
                "the heuristic path breaches max_gross and max_name_weight (§3.7). This constructor enforces "
                "the limits inside the program, so a band would have to enter the program as a term, and none "
                "is declared. The registration carries mode = 'off' for both convex arms"
            )
        self.cash_buffer = float(inputs.cash_buffer or 0.0)
        if not math.isfinite(self.cash_buffer) or self.cash_buffer < 0.0:
            raise ConstructionError(f"cash_buffer must be finite and >= 0, got {inputs.cash_buffer!r}")

        self.limits = tuple(inputs.limits)
        self._classify_limits()
        self.risk = inputs.risk
        self._check_risk_before_any_solve()
        self.cost_terms = self._resolve_cost_terms()
        self.notes: list[str] = []
        self.deferred: list[str] = []

        cp = self._cvxpy()
        self.cp = cp
        self.w = cp.Variable(self.n, name="w")
        self._cone: tuple[np.ndarray, np.ndarray] | None = None
        self._forecasts = self._resolve_forecasts()
        self._action = self._build_action_constraints()
        self._compile_seconds_first: float | None = None

    # -- lazy solver import -------------------------------------------------

    @staticmethod
    def _cvxpy() -> Any:
        """Import cvxpy here and nowhere else (§3.7): it is the ``construction`` extra."""
        try:
            import cvxpy  # noqa: PLC0415 - deliberately lazy; see the module docstring
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ConstructionError(
                "the convex constructor needs cvxpy and clarabel, which are the declared `construction` "
                "extra; install prism[construction]"
            ) from exc
        return cvxpy

    # -- setup --------------------------------------------------------------

    def _classify_limits(self) -> None:
        action: list[Limit] = []
        holding: list[Limit] = []
        for limit in self.limits:
            if is_action_restriction(limit):
                if limit.name not in ACTION_RESTRICTION_NAMES:
                    raise ConstructionError(
                        f"{limit.name!r} is declared {limit.cls.value} and this constructor has no convex "
                        f"form for it; the restrictions it writes are {sorted(ACTION_RESTRICTION_NAMES)}. An "
                        "action restriction is never waived, so an unrecognized one refuses the solve rather "
                        "than being skipped"
                    )
                action.append(limit)
            elif is_holding_state(limit):
                if limit.name not in MEASURED_LIMITS:
                    raise ConstructionError(
                        f"{limit.name!r} is a holding-state limit with no violation measure in the §3.7 table; "
                        "recovery could neither bound it nor report it"
                    )
                holding.append(limit)
        self.action_limits = tuple(action)
        self.holding_limits = tuple(holding)

    def _check_risk_before_any_solve(self) -> None:
        """§3.4 and D9: refuse the run before the first solve, not after it."""
        needs_risk = any(limit.name in ("variance_budget", "beta_band") for limit in self.limits)
        if self.risk is None:
            if needs_risk:
                raise ConstructionError(
                    "a variance_budget or beta_band is declared and no risk snapshot was supplied; the budget "
                    "is the only channel through which the risk model acts on the solution (D8), so solving "
                    "without one would silently drop it"
                )
            return
        from prism.risk_model import (
            validate_snapshot,  # local: keeps the limit layer solver-free
        )

        validation = validate_snapshot(self.risk, universe=self.sids, policy=self.inputs.risk_policy)
        if not validation.accepted:
            raise ConstructionError(
                f"the risk snapshot is rejected before any solve ({validation.outcome}): "
                f"{'; '.join(validation.findings)}"
            )
        spec = self.inputs.calibration_spec
        if spec is not None:
            require_sigma_agreement(
                spec,
                scaling=str(self.risk.scaling),
                horizon_bars=int(self.risk.horizon_bars),
            )

    def _resolve_cost_terms(self) -> Any:
        costs = self.inputs.costs
        if costs is None:
            raise ConstructionError(
                "the objective charges trade and holding costs; a constructor with no cost model would "
                "propose a book no cost stack was ever applied to"
            )
        if hasattr(costs, "convex_terms"):
            return costs.convex_terms(self.sids, adv_dollars=self.inputs.adv_dollars)
        terms = costs
        if list(map(str, terms.sids)) != list(self.sids):
            raise ConstructionError(
                "the cost terms are stated on a different universe than the constructor's; the two orderings "
                "must be identical, since every coefficient is positional"
            )
        return terms

    def _resolve_forecasts(self) -> list[dict[str, Any]]:
        """``a`` per component, with the §3.4 sign policy applied through the calibration."""
        resolved: list[dict[str, Any]] = []
        for forecast in self.inputs.forecasts:
            record: dict[str, Any] = {
                "component_id": forecast.component_id,
                "weight": float(forecast.weight),
                "disabled": bool(forecast.disabled),
                "disabled_reason": forecast.disabled_reason or forecast.fit.disabled_reason,
                "confidence": forecast.fit.confidence,
                "straddles_zero": bool(forecast.fit.straddles_zero),
                "sign_ok": bool(forecast.fit.sign_ok),
                "theta_shrunk": forecast.fit.theta_shrunk,
                "n_scored": 0,
                "n_zero_coefficient": self.n,
            }
            if forecast.disabled:
                record["lower"] = 0.0
                record["upper"] = 0.0
                record["a"] = np.zeros(self.n, dtype=float)
                record["reason"] = "the component is disabled; it contributes no return term (§3.7)"
                resolved.append(record)
                continue
            lower, upper = objective_endpoints(forecast.fit)
            record["lower"], record["upper"] = float(lower), float(upper)
            record["sign_contradiction"] = not forecast.fit.sign_ok
            usable = forecast.z.reindex(self.index)
            sigma = forecast.sigma_h.reindex(self.index)
            mask = np.isfinite(usable.to_numpy(dtype=float)) & np.isfinite(sigma.to_numpy(dtype=float))
            mask &= sigma.to_numpy(dtype=float) > 0.0
            a = np.zeros(self.n, dtype=float)
            if mask.any():
                # apply_calibration refuses a non-finite z or a non-positive sigma rather
                # than imputing; the held-name-without-an-estimate case of §3.7 is given
                # a_i = 0 explicitly *here*, after the call, never carried in as a NaN.
                sub_a, _ = apply_calibration(forecast.fit, usable[mask], sigma[mask])
                a[mask] = sub_a.to_numpy(dtype=float) * float(forecast.weight)
            record["a"] = a
            record["n_scored"] = int(mask.sum())
            record["n_zero_coefficient"] = int((~mask).sum())
            resolved.append(record)
        return resolved

    # -- action restrictions ------------------------------------------------

    def _reduce_only_mask(self) -> np.ndarray:
        """The names pinned reduce-only: substituted, ineligible, or without an estimate."""
        mask = np.zeros(self.n, dtype=bool)
        position = {sid: i for i, sid in enumerate(self.sids)}
        named: set[str] = set()
        for limit in self.action_limits:
            if limit.name == "reduce_only_ineligible":
                named |= {str(sid) for sid in limit.params.get("sids", ())}
        if self.risk is not None:
            named |= {str(sid) for sid in getattr(self.risk, "reduce_only", ())}
        scored = np.zeros(self.n, dtype=bool)
        for record in self._forecasts:
            scored |= record["a"] != 0.0
        for sid in named:
            if sid in position:
                mask[position[sid]] = True
        if self._forecasts:
            # A held name with no current estimate stays in the program under the
            # reduce-only box rather than being forced to a liquidation the market may not
            # take today; a zero holding mask is not used (§3.7, Q15).
            mask |= ~scored
        return mask

    def _build_action_constraints(self) -> list[Any]:
        cp = self.cp
        w, w0 = self.w, self.w0
        constraints: list[Any] = []
        position = {sid: i for i, sid in enumerate(self.sids)}

        no_trade = np.zeros(self.n, dtype=bool)
        no_new_short = np.zeros(self.n, dtype=bool)
        for limit in self.action_limits:
            name = limit.name
            if name in _DEFERRED_ACTION_RESTRICTIONS:
                self.deferred.append(
                    f"{name} is enforced by portfolio/rounding.py and the submission revalidation, not by a "
                    "constraint on a continuous weight"
                )
                continue
            if name == "participation_per_name":
                constraints.append(cp.abs(w - w0) <= self._participation_trade_bound(limit))
            elif name in ("halted_no_trade", "regulatory_restriction"):
                for sid in limit.params.get("sids", ()):
                    if str(sid) in position:
                        no_trade[position[str(sid)]] = True
            elif name == "no_new_short_without_borrow":
                for sid in limit.params.get("sids", ()):
                    if str(sid) in position:
                        no_new_short[position[str(sid)]] = True
            elif name == "account_prohibits_shorting":
                no_new_short |= True
            elif name == "buying_power":
                dollars = limit.value()
                if dollars < 0.0:
                    raise ConstructionError(f"buying_power must be >= 0 dollars, got {dollars}")
                constraints.append(cp.sum(cp.abs(w)) * self.equity <= dollars)
            elif name == "reduce_only_ineligible":
                continue  # handled by the reduce-only box below
        if no_trade.any():
            constraints.append(w[no_trade] == w0[no_trade])
        if no_new_short.any():
            # No *new* short: a held short may be reduced, never deepened.
            floor = np.minimum(w0, 0.0)
            constraints.append(w[no_new_short] >= floor[no_new_short])

        reduce_only = self._reduce_only_mask()
        self.reduce_only_mask = reduce_only
        longs = reduce_only & (w0 > 0.0)
        shorts = reduce_only & (w0 < 0.0)
        flats = reduce_only & (w0 == 0.0)
        if longs.any():
            constraints.append(w[longs] >= 0.0)
            constraints.append(w[longs] <= w0[longs])
        if shorts.any():
            constraints.append(w[shorts] <= 0.0)
            constraints.append(w[shorts] >= w0[shorts])
        if flats.any():
            constraints.append(w[flats] == 0.0)

        unwind = self._unwind_mask()
        self.unwind_mask = unwind
        if unwind.any():
            allowance = self._max_reduction()
            ceiling = np.maximum(0.0, np.abs(w0) - allowance)
            constraints.append(cp.abs(w[unwind]) <= ceiling[unwind])
        return constraints

    def _unwind_mask(self) -> np.ndarray:
        """Names a sign-contradicted component must unwind, under ``on_sign_contradiction``.

        ``hold`` adds nothing: the return term is already zero for that component, so trade
        cost keeps its existing exposure where it is. ``unwind`` requires the largest
        reduction the action restrictions allow this bar -- not an instant exit, which the
        participation restriction may forbid.
        """
        policy = self.inputs.policy
        mask = np.zeros(self.n, dtype=bool)
        if policy is None or policy.on_sign_contradiction != "unwind":
            return mask
        for record in self._forecasts:
            if record.get("sign_contradiction") or record.get("disabled"):
                # `a` is non-zero wherever the component scored, whatever its endpoints;
                # a component that scored nothing has nothing to unwind.
                mask |= record["a"] != 0.0
        return mask & (self.w0 != 0.0)

    def _participation_bound(self, limit: Limit) -> np.ndarray:
        """``value * adv_i / equity``: the traded weight the participation cap allows."""
        adv = self.inputs.adv_dollars
        if adv is None:
            raise ConstructionError(
                "participation_per_name needs the declared dollar_volume.participation statistic; without it "
                "every trade would be unrestricted"
            )
        values = adv.reindex(self.index).to_numpy(dtype=float)
        unusable = ~np.isfinite(values) | (values <= 0.0)
        if unusable.any():
            # The fail-safe direction, as in limits.py: an unknown dollar volume restricts
            # the name to nothing, never to everything.
            values = np.where(unusable, 0.0, values)
            self.notes.append(
                f"{int(unusable.sum())} name(s) carry no usable participation dollar volume and are pinned "
                "to no trade this bar"
            )
        return limit.value() * values / self.equity

    def _participation_trade_bound(self, limit: Limit) -> np.ndarray:
        """The bound the *program* imposes: the declared allowance less the safety margin."""
        return np.maximum(0.0, self._participation_bound(limit) - self.margin(limit))

    def _max_reduction(self) -> np.ndarray:
        """The magnitude each name may move toward zero under the action restrictions.

        Stated against the same margined bound the trade constraint uses, so that an unwind
        ceiling and the participation restriction cannot disagree by the margin and make the
        program infeasible over a number no limit declares.
        """
        allowance = np.abs(self.w0).copy()
        for limit in self.action_limits:
            if limit.name == "participation_per_name":
                allowance = np.minimum(allowance, self._participation_trade_bound(limit))
        return allowance

    def action_constraints(self) -> list[Any]:
        """Every action restriction, hard, on every path."""
        return list(self._action)

    # -- terms --------------------------------------------------------------

    def cone(self) -> tuple[np.ndarray, np.ndarray]:
        """The snapshot's cone operator, computed once per program (D9)."""
        if self._cone is None:
            if self.risk is None:
                raise ConstructionError("no risk snapshot: the variance cone has nothing to stand for")
            factor, residual = self.risk.cone_operator()
            self._cone = (np.asarray(factor, dtype=float), np.asarray(residual, dtype=float))
        return self._cone

    def _variance_norm(self) -> Any:
        """``sqrt(w' Sigma w)`` as one second-order cone; no ``N x N`` matrix is formed."""
        cp = self.cp
        factor, residual = self.cone()
        return cp.norm(cp.hstack([factor.T @ self.w, cp.multiply(residual, self.w)]), 2)

    def _per_name_cap(self, limit: Limit) -> np.ndarray:
        raw = limit.params.get("value")
        if isinstance(raw, (Mapping, pd.Series)):
            caps = pd.Series(raw, dtype=float).reindex(self.index)
            if caps.isna().any():
                missing = [sid for sid in self.index[caps.isna()]][:5]
                raise ConstructionError(f"{limit.name}: no per-name cap declared for {missing}")
            return caps.to_numpy(dtype=float)
        return np.full(self.n, limit.value(), dtype=float)

    def _beta_column(self, limit: Limit) -> np.ndarray:
        factor_id = limit.params.get("factor_id")
        if not isinstance(factor_id, str) or not factor_id:
            raise ConstructionError(f"{limit.name}: a beta band declares params.factor_id")
        if self.risk is None:
            raise ConstructionError(f"{limit.name}: a beta band needs a risk snapshot")
        frame = self.risk.loadings_frame()
        if factor_id not in frame.columns:
            raise ConstructionError(
                f"{limit.name}: factor {factor_id!r} is not one of the snapshot's factors "
                f"{[str(col) for col in frame.columns]}"
            )
        column = frame[factor_id].reindex(self.index)
        if column.isna().any():
            missing = [sid for sid in self.index[column.isna()]][:5]
            raise ConstructionError(f"{limit.name}: the snapshot carries no {factor_id!r} loading for {missing}")
        return column.to_numpy(dtype=float)

    def violation_expr(self, limit: Limit) -> Any:
        """The §3.7 violation measure ``v(w)`` as a convex expression -- the same measure
        :func:`prism.portfolio.limits.violation` computes numerically, term for term."""
        cp = self.cp
        w, w0 = self.w, self.w0
        name = limit.name
        if name == "max_gross":
            return cp.pos(cp.sum(cp.abs(w)) - limit.value())
        if name == "max_name_weight":
            return cp.sum(cp.pos(cp.abs(w) - self._per_name_cap(limit)))
        if name == "net_band":
            centre, half = limit.band()
            return cp.pos(cp.abs(cp.sum(w) - centre) - half)
        if name == "beta_band":
            centre, half = limit.band()
            return cp.pos(cp.abs(self._beta_column(limit) @ w - centre) - half)
        if name == "variance_budget":
            return cp.pos(self._variance_norm() - limit.value())
        if name == "turnover_cap":
            return cp.pos(cp.sum(cp.abs(w - w0)) - limit.value())
        if name == "participation_per_name":
            adv = self._participation_bound(limit) * self.equity
            return cp.sum(cp.pos(cp.abs(w - w0) * self.equity - adv)) / self.equity
        raise ConstructionError(f"{name!r} has no declared violation measure in the §3.7 table")

    def margin(self, limit: Limit) -> float:
        """The bound's safety margin: half this limit's declared tolerance (see the constant)."""
        return _CONSTRAINT_MARGIN_FRACTION * limit.effective_tolerance(self.limit_tolerance)

    def limit_constraints(self, limit: Limit) -> list[Any]:
        """The limit as a *hard* constraint, which is what the normal path imposes."""
        cp = self.cp
        w, w0 = self.w, self.w0
        name = limit.name
        margin = self.margin(limit)
        if name == "max_gross":
            available = limit.value() - self.cash_buffer
            if available <= 0.0:
                raise ConstructionError(
                    f"max_gross {limit.value()} less the {self.cash_buffer} cash buffer leaves "
                    f"{available} of gross available; the program would admit only the zero book"
                )
            return [cp.sum(cp.abs(w)) <= max(0.0, available - margin)]
        if name == "max_name_weight":
            return [cp.abs(w) <= np.maximum(0.0, self._per_name_cap(limit) - margin)]
        if name == "net_band":
            centre, half = limit.band()
            return [cp.abs(cp.sum(w) - centre) <= max(0.0, half - margin)]
        if name == "beta_band":
            centre, half = limit.band()
            return [cp.abs(self._beta_column(limit) @ w - centre) <= max(0.0, half - margin)]
        if name == "variance_budget":
            return [self._variance_norm() <= max(0.0, limit.value() - margin)]
        if name == "turnover_cap":
            return [cp.sum(cp.abs(w - w0)) <= max(0.0, limit.value() - margin)]
        if name == "participation_per_name":
            return [cp.abs(w - w0) <= self._participation_trade_bound(limit)]
        raise ConstructionError(f"{name!r} has no declared hard form")

    def holding_state_constraints(self) -> list[Any]:
        """Every holding-state limit, hard: the normal path's whole difference from recovery."""
        out: list[Any] = []
        for limit in self.holding_limits:
            out.extend(self.limit_constraints(limit))
        return out

    def obligation_progress_expr(
        self, obligation: Obligation, current_qty: float
    ) -> tuple[Any | None, float, str | None]:
        """``(progress_expr, remaining, reason)`` for one obligation, in shares.

        Progress is the magnitude moved toward ``target_qty``, clipped to ``remaining_qty``:
        ``min(direction * (w_i - w0_i) * equity / price_i, remaining_i)``, the minimum of an
        affine function and a constant, hence concave and maximizable.

        ``expr = None`` with a ``reason`` is an obligation the decision bar cannot express --
        an unpriced name, a name outside the universe -- which is *blocked*. ``expr = None``
        with ``reason = None`` is an obligation with nothing left to do, because a
        broker-initiated fill already moved the position to its target; that is satisfied,
        not blocked, and must not escalate.
        """
        cp = self.cp
        sid = str(obligation.sid)
        if sid not in self.index:
            return None, float(obligation.remaining_qty), f"{sid} is not in the constructor universe"
        price = float(self.inputs.reference_prices.get(sid, float("nan")))
        if not math.isfinite(price) or price <= 0.0:
            return None, float(obligation.remaining_qty), f"{sid} has no usable reference price ({price!r})"
        direction = obligation.direction(float(current_qty))
        if direction == 0.0:
            return None, 0.0, None
        remaining = obligation.shortfall_qty(float(current_qty))
        if remaining <= 0.0:
            return None, 0.0, None
        position = self.index.get_loc(sid)
        shares = direction * (self.w[position] - self.w0[position]) * self.equity / price
        return cp.minimum(shares, remaining), float(remaining), None

    def return_term(self) -> Any:
        """``sum_c min(lower_c * s_c, upper_c * s_c)`` -- concave, and zero when nothing is
        forecast.

        §3.7 writes the term for one component. With several, each contributes its own
        robust minimum on its own ``s_c = a_c' w``: the sum of concave terms is concave, and
        a single joint interval would have to invent a covariance between the components'
        endpoint intervals that no calibration declares.
        """
        cp = self.cp
        terms: list[Any] = []
        for record in self._forecasts:
            lower, upper = float(record.get("lower", 0.0)), float(record.get("upper", 0.0))
            a = record["a"]
            if lower == 0.0 and upper == 0.0:
                continue
            if not np.any(a != 0.0):
                continue
            s = a @ self.w
            terms.append(cp.minimum(lower * s, upper * s) if lower != upper else lower * s)
        if not terms:
            return cp.Constant(0.0)
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total

    def trade_cost_expr(self) -> Any:
        """The one-time cost fraction of equity, exactly ``ConvexCostTerms.trade_costs``."""
        cp = self.cp
        terms = self.cost_terms
        dw = self.w - self.w0
        cost = (np.asarray(terms.linear_bps, dtype=float) @ cp.abs(dw)) / 1e4
        quadratic = np.asarray(terms.quadratic_coeff, dtype=float)
        if np.any(quadratic != 0.0):
            cost = cost + (quadratic @ cp.square(dw)) / 1e4
        power = np.asarray(terms.power_three_halves_coeff, dtype=float)
        if np.any(power != 0.0):
            cost = cost + (power @ cp.power(cp.abs(dw), 1.5)) / 1e4
        return cost

    def holding_cost_expr(self) -> Any:
        """The holding cost fraction over the horizon, as a *convex* expression.

        ``ConvexCostTerms.holding_costs`` is
        ``borrow' max(-w,0) + f*max(-cf,0) - c*max(cf,0)`` with ``cf`` the cash fraction.
        The financing pair is written as ``max(-f*cf, -c*cf)``, which is the identical
        function when ``f >= c`` and is DCP-verifiable: each branch is a non-positive
        multiple of a concave ``cf``, hence convex, and the maximum of convex terms is
        convex. When ``c > f`` the function is not convex in ``w`` and the run is refused
        rather than silently modeled as something else -- a cash rate above the financing
        rate is an arbitrage, not a rounding detail.
        """
        cp = self.cp
        terms = self.cost_terms
        borrow = np.asarray(terms.borrow_per_horizon, dtype=float)
        cost: Any = cp.Constant(0.0)
        if np.any(borrow != 0.0):
            cost = cost + borrow @ cp.pos(-self.w)
        financing = float(terms.financing_per_horizon)
        income = float(terms.cash_income_per_horizon)
        if financing == 0.0 and income == 0.0:
            return cost
        if income > financing:
            raise ConstructionError(
                f"the cash income rate ({income}) exceeds the financing rate ({financing}) over the horizon; "
                "the holding cost is then not convex in w and this program cannot state it"
            )
        if str(terms.financing_base) == "proceeds_to_cash_with_rebate":
            cash_fraction = 1.0 - cp.sum(self.w)  # long - short weight is exactly sum(w)
        else:
            cash_fraction = 1.0 - cp.sum(cp.pos(self.w))
        return cost + cp.maximum(-financing * cash_fraction, -income * cash_fraction)

    def economic_objective(self) -> Any:
        """The §3.7 objective, as a concave expression to be maximized."""
        cp = self.cp
        objective = self.return_term() - self.trade_cost_expr() - self.holding_cost_expr()
        if self.risk_aversion > 0.0:
            # sqrt(risk_aversion) folds into the cone rather than multiplying it: the
            # product of two parameters is not DPP and would fall back silently (§3.7).
            factor, residual = self.cone()
            scale = math.sqrt(self.risk_aversion)
            objective = objective - cp.sum_squares(
                cp.hstack([(scale * factor).T @ self.w, cp.multiply(scale * residual, self.w)])
            )
        if self.eps > 0.0:
            objective = objective - self.eps * cp.sum_squares(self.w)
        return objective

    # -- solving ------------------------------------------------------------

    def solve(
        self,
        objective: Any,
        constraints: Sequence[Any],
        *,
        sense: str = "maximize",
        stage: str = "normal",
        subject: str = "economic_objective",
    ) -> SolveOutcome:
        """Compile and solve one problem with CLARABEL, selected explicitly.

        The problem is compiled for the shape it is handed; the universe dimension is fixed
        nowhere. A solver failure is an outcome, not an exception: it becomes
        ``optimal = False`` and the caller puts the decision on the no-action path. A DCP
        violation is *not* caught, because it is a defect in this module and not a property
        of the data.
        """
        cp = self.cp
        started = time.perf_counter()
        target = cp.Maximize(objective) if sense == "maximize" else cp.Minimize(objective)
        problem = cp.Problem(target, list(constraints))
        compile_seconds = time.perf_counter() - started
        if self._compile_seconds_first is None:
            self._compile_seconds_first = compile_seconds
        solve_started = time.perf_counter()
        note: str | None = None
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False, **SOLVER_TOLERANCES)
        except cp.error.SolverError as exc:
            return SolveOutcome(
                status="solver_error",
                optimal=False,
                value=None,
                w=None,
                solve_seconds=time.perf_counter() - solve_started,
                compile_seconds=compile_seconds,
                note=f"{stage}/{subject}: {exc}",
            )
        wall = time.perf_counter() - solve_started
        status = str(problem.status)
        stats = getattr(problem, "solver_stats", None)
        iterations = getattr(stats, "num_iters", None) if stats is not None else None
        solve_seconds = float(getattr(stats, "solve_time", None) or wall)
        optimal = status in ("optimal", "optimal_inaccurate")
        if status == "optimal_inaccurate":
            note = (
                f"{stage}/{subject}: CLARABEL returned optimal_inaccurate; the independent validation of "
                "portfolio/validate.py decides whether the output is usable, not this status"
            )
        values = None if self.w.value is None else np.asarray(self.w.value, dtype=float).reshape(-1)
        if optimal and (values is None or not np.isfinite(values).all()):
            optimal = False
            status = "non_finite_solution"
            note = f"{stage}/{subject}: the solver returned a non-finite weight vector"
            values = None
        series = None if values is None or not optimal else pd.Series(values, index=self.index, dtype=float)
        return SolveOutcome(
            status=status,
            optimal=optimal,
            value=None if problem.value is None or not np.isfinite(problem.value) else float(problem.value),
            w=series,
            solve_seconds=solve_seconds,
            compile_seconds=compile_seconds,
            iterations=None if iterations is None else int(iterations),
            note=note,
        )

    # -- reporting ----------------------------------------------------------

    def signal_exposure(self, w: pd.Series) -> dict[str, dict[str, Any]]:
        """``s_c = a_c' w`` per component, with the endpoints that priced it."""
        values = w.reindex(self.index).to_numpy(dtype=float)
        out: dict[str, dict[str, Any]] = {}
        for record in self._forecasts:
            out[str(record["component_id"])] = {
                "s": float(record["a"] @ values),
                "lower": record.get("lower"),
                "upper": record.get("upper"),
                "straddles_zero": bool(record["straddles_zero"]),
                "sign_ok": bool(record["sign_ok"]),
                "sign_contradiction": bool(record.get("sign_contradiction", False)),
                "disabled": bool(record["disabled"]),
                "disabled_reason": record.get("disabled_reason"),
                "confidence": record.get("confidence"),
                "n_scored": int(record["n_scored"]),
                "n_zero_coefficient": int(record["n_zero_coefficient"]),
            }
        return out

    def at_cap_fraction(self, w: pd.Series, *, limit_tolerance: float) -> float | None:
        """The fraction of held names at the per-name cap, to ``limit_tolerance`` (§3.7).

        ``None`` when no ``max_name_weight`` is declared, because there is then no cap to be
        at and reporting ``0.0`` would read as "no concentration" rather than "not measured".
        """
        caps = None
        for limit in self.holding_limits:
            if limit.name == "max_name_weight":
                caps = self._per_name_cap(limit)
        if caps is None:
            return None
        values = np.abs(w.reindex(self.index).to_numpy(dtype=float))
        held = values > limit_tolerance
        if not held.any():
            return 0.0
        at_cap = held & (values >= caps - limit_tolerance)
        return float(at_cap.sum()) / float(held.sum())


# --------------------------------------------------------------------------
# The constructor
# --------------------------------------------------------------------------


class ConvexConstructor:
    """The §3.7 convex constructor: normal path, recovery path, no-action path.

    ``construct`` is the whole interface. It chooses the path from the *marked* state, never
    from how the solve went, and it validates independently afterwards: a solver status is
    never an authorization, so an output CLARABEL called optimal and the validator refuses
    goes to the no-action path with the declared response.
    """

    def __init__(self, *, policy: RecoveryPolicy | None = None) -> None:
        self.policy = policy or RecoveryPolicy()

    def construct(self, x: ConstructionInputs) -> ConstructionOutput:
        """Solve one decision bar (§3.7)."""
        policy = x.policy or self.policy
        settings = x.validation or ValidationSettings(
            preference_response=policy.preference_response,
            bound_slack=policy.bound_slack,
        )
        program = ConvexProgram(x, limit_tolerance=settings.limit_tolerance)
        ctx = LimitContext(w0=x.w0, equity=x.equity, adv_dollars=x.adv_dollars, risk=x.risk)
        enters, reason = recovery_entry(
            x.limits,
            ctx,
            tuple(x.obligations),
            limit_tolerance=settings.limit_tolerance,
            recovery_entry_margin=policy.recovery_entry_margin,
        )

        diagnostics: dict[str, Any] = {
            "decision_bar": x.decision_bar,
            "n_names": program.n,
            "risk_aversion": program.risk_aversion,
            "eps": program.eps,
            "cash_buffer": program.cash_buffer,
            "eligibility_exit": program.eligibility_exit,
            "regime_scale": None if x.regime_scale is None else float(x.regime_scale),
            "min_regime_scale": None if x.min_regime_scale is None else float(x.min_regime_scale),
            "reduce_only": [sid for sid, flag in zip(program.sids, program.reduce_only_mask) if flag],
            "unwind_on_sign_contradiction": [sid for sid, flag in zip(program.sids, program.unwind_mask) if flag],
            "deferred_action_restrictions": list(program.deferred),
            "notes": sorted(set(program.notes)),
            "component_outputs": dict(x.component_outputs or {}),
            "recovery_entry": {"entered": bool(enters), "reason": reason},
            "account_target_supplied": x.account_target is not None,
            "solver_name": "CLARABEL",
        }
        if x.risk is not None:
            diagnostics["risk_model"] = x.risk.record()
            diagnostics["substitutions"] = x.risk.substitutions

        if enters:
            return self._recovery_path(program, x, ctx, policy, settings, diagnostics, reason)
        return self._normal_path(program, x, ctx, policy, settings, diagnostics)

    # -- the normal path ----------------------------------------------------

    def _normal_path(
        self,
        program: ConvexProgram,
        x: ConstructionInputs,
        ctx: LimitContext,
        policy: RecoveryPolicy,
        settings: ValidationSettings,
        diagnostics: dict[str, Any],
    ) -> ConstructionOutput:
        constraints = program.action_constraints() + program.holding_state_constraints()
        outcome = program.solve(
            program.economic_objective(),
            constraints,
            sense="maximize",
            stage="normal",
            subject="economic_objective",
        )
        solver = _solver_record(outcome, stage="normal")
        diagnostics["stages"] = [dict(solver, stage="normal", subject="economic_objective")]
        if not outcome.optimal or outcome.w is None:
            diagnostics["no_action"] = no_action(
                policy,
                f"the normal-path program did not solve (status {outcome.status}); "
                "neither no-trade nor the zero portfolio is assumed executable",
                stage="normal",
            )
            return self._no_action_output(x, ctx, settings, diagnostics, solver, binding="normal")
        return self._finish(program, x, ctx, policy, settings, diagnostics, outcome, solver, path="normal")

    # -- the recovery path --------------------------------------------------

    def _recovery_path(
        self,
        program: ConvexProgram,
        x: ConstructionInputs,
        ctx: LimitContext,
        policy: RecoveryPolicy,
        settings: ValidationSettings,
        diagnostics: dict[str, Any],
        reason: str | None,
    ) -> ConstructionOutput:
        positions = x.positions
        if positions is None:
            prices = x.reference_prices.reindex(pd.Index(program.sids))
            positions = pd.Series(
                x.w0.to_numpy(dtype=float) * x.equity / prices.to_numpy(dtype=float),
                index=pd.Index(program.sids),
                dtype=float,
            )
        result: RecoveryResult = run_recovery(
            program,
            limits=x.limits,
            obligations=tuple(x.obligations),
            policy=policy,
            positions=positions,
            reference_prices=x.reference_prices,
            limit_tolerance=settings.limit_tolerance,
            ctx=ctx,
        )
        diagnostics["recovery"] = result.to_dict()
        diagnostics["stages"] = [stage.to_dict() for stage in result.stages]
        diagnostics["achieved_bounds"] = dict(result.achieved_bounds)
        diagnostics["escalations"] = [escalation.to_dict() for escalation in result.escalations]
        solver = _solver_record_from_stages(result)
        if not result.ok or result.w is None:
            diagnostics["no_action"] = no_action(
                policy,
                f"recovery stopped at {result.failed_stage} on {result.binding_condition!r}; "
                "no executable improvement exists and no orders are submitted",
                stage=str(result.failed_stage),
            )
            return self._no_action_output(x, ctx, settings, diagnostics, solver, binding=result.binding_condition)
        outcome = SolveOutcome(
            status=result.stages[-1].status,
            optimal=True,
            value=result.stages[-1].value,
            w=result.w,
            solve_seconds=sum(stage.solve_seconds for stage in result.stages),
            compile_seconds=sum(stage.compile_seconds for stage in result.stages),
        )
        return self._finish(
            program,
            x,
            ctx,
            policy,
            settings,
            diagnostics,
            outcome,
            solver,
            path="recovery",
            achieved_bounds=result.achieved_bounds,
            attribution=result.attribution,
            binding=result.binding_condition,
        )

    # -- shared tail --------------------------------------------------------

    def _finish(
        self,
        program: ConvexProgram,
        x: ConstructionInputs,
        ctx: LimitContext,
        policy: RecoveryPolicy,
        settings: ValidationSettings,
        diagnostics: dict[str, Any],
        outcome: SolveOutcome,
        solver: dict[str, Any],
        *,
        path: Literal["normal", "recovery"],
        achieved_bounds: Mapping[str, float] | None = None,
        attribution: Mapping[str, str] | None = None,
        binding: str | None = None,
    ) -> ConstructionOutput:
        assert outcome.w is not None  # guarded by both callers
        w = outcome.w
        stage_label = "R2" if path == "recovery" else "normal"
        report = validate_targets(
            w,
            x.limits,
            ctx,
            settings=settings,
            achieved_bounds=achieved_bounds,
            attributed_to=stage_label,
        )
        violations = {key: [dict(entry) for entry in value] for key, value in report.violations.items()}
        if attribution:
            # §3.7: `attributed_to` names the stage that produced the violation. R0 and R1
            # can each introduce one, so an entry the recovery sequence attributed to an
            # earlier stage is relabeled from the solve stage to that one.
            for entry in violations["introduced"]:
                stage = attribution.get(str(entry["limit_name"]))
                if stage is not None:
                    entry["attributed_to"] = stage
        report = report.__class__(
            stage=report.stage,
            mode=report.mode,
            violations=violations,
            findings=report.findings,
            unchecked=report.unchecked,
            cash=report.cash,
            cost=report.cost,
            binding_condition=report.binding_condition,
        )
        diagnostics["validation"] = report.to_dict()
        # The measures read the risk snapshot and the cost model, both of which refuse a
        # non-finite weight. Validation has already recorded that refusal, so a target the
        # validator rejected for finiteness is described, not re-measured.
        if np.isfinite(w.to_numpy(dtype=float)).all():
            diagnostics.update(self._measures(program, x, w, settings))
        else:
            diagnostics["measures_skipped"] = "the solved target is not finite; validation refused it"
        diagnostics["achieved_bounds"] = dict(achieved_bounds or {})
        if attribution:
            diagnostics["attribution"] = dict(attribution)
        if not report.ok:
            # Solver status never authorizes an order. A target the independent validation
            # refuses is not submitted, whatever CLARABEL called it.
            diagnostics["no_action"] = no_action(
                policy,
                "the solved target failed independent validation: "
                + "; ".join(finding.detail for finding in report.fatal_findings),
                stage=stage_label,
            )
            return self._no_action_output(
                x, ctx, settings, diagnostics, solver, binding=report.binding_condition or binding
            )
        return ConstructionOutput(
            w_target=w,
            violations={key: [dict(entry) for entry in value] for key, value in report.violations.items()},
            path=path,
            solver=solver,
            # The recovery sequence's own binding condition wins when it has one: §3.7 asks
            # for the *obligation* to be named when R0's progress is what introduced the
            # violation, and the limit measure alone cannot say that.
            binding_condition=binding or report.binding_condition,
            diagnostics=diagnostics,
        )

    def _no_action_output(
        self,
        x: ConstructionInputs,
        ctx: LimitContext,
        settings: ValidationSettings,
        diagnostics: dict[str, Any],
        solver: dict[str, Any] | None,
        *,
        binding: str | None,
    ) -> ConstructionOutput:
        """No orders, the declared response, and the marked state's own violations.

        ``w_target`` is all-NaN: NaN is "no decision for that sid", which is exactly the
        state of every name on this path. Returning ``w0`` would assert that holding it was
        decided and validated, and returning zeros would assert that a liquidation is
        executable; §3.7 permits neither assumption.
        """
        marked = partition_violations(
            x.limits, x.w0, ctx, limit_tolerance=settings.limit_tolerance, attributed_to="marked"
        )
        w_target = pd.Series(np.nan, index=pd.Index(tuple(x.sids), name="sid"), dtype=float)
        return ConstructionOutput(
            w_target=w_target,
            violations={key: list(value) for key, value in marked.items()},
            path="no_action",
            solver=solver,
            binding_condition=binding or binding_condition(marked),
            diagnostics=diagnostics,
        )

    def _measures(
        self,
        program: ConvexProgram,
        x: ConstructionInputs,
        w: pd.Series,
        settings: ValidationSettings,
    ) -> dict[str, Any]:
        values = w.reindex(pd.Index(program.sids)).to_numpy(dtype=float)
        trade = values - program.w0
        exposure = program.signal_exposure(w)
        gross = float(np.abs(values).sum())
        straddling = [
            component
            for component, entry in exposure.items()
            if float(entry.get("lower") or 0.0) < 0.0 < float(entry.get("upper") or 0.0)
        ]
        measures: dict[str, Any] = {
            "at_cap_fraction": program.at_cap_fraction(w, limit_tolerance=settings.limit_tolerance),
            "max_at_cap_fraction": program.settings.get("max_at_cap_fraction"),
            "gross": gross,
            "net": float(values.sum()),
            "turnover": float(np.abs(trade).sum()),
            "n_held": int((np.abs(values) > settings.limit_tolerance).sum()),
            "signal_exposure": exposure,
            "straddling_components": straddling,
            "flat_from_straddling_endpoints": bool(straddling) and gross <= settings.limit_tolerance,
            "trade_cost_fraction": float(program.cost_terms.trade_costs(trade)),
            "holding_cost_fraction": float(program.cost_terms.holding_costs(values)),
            "modeled_cost_dollars": float(program.cost_terms.total_dollars(trade, values)),
            "no_trade": bool(np.abs(trade).max(initial=0.0) <= settings.limit_tolerance),
        }
        if x.risk is not None:
            variance = float(x.risk.variance(w))
            factor, residual = program.cone()
            from prism.risk_model import (
                ConeOperator,  # local: the solver-free record path
            )

            cone_value = ConeOperator(factor, residual).quadratic_form(values)
            measures["ex_ante_sd"] = math.sqrt(max(variance, 0.0))
            measures["cone_vs_variance_relative"] = abs(cone_value - variance) / variance if variance > 0.0 else 0.0
        return measures


def _solver_record(outcome: SolveOutcome, *, stage: str) -> dict[str, Any]:
    """The ``{status, iters, time, tol}`` record §3.7 names, plus what it omits."""
    return {
        "status": outcome.status,
        "iters": outcome.iterations,
        "time": float(outcome.solve_seconds),
        "compile_time": float(outcome.compile_seconds),
        "tol": dict(SOLVER_TOLERANCES),
        "solver": "CLARABEL",
        "stage": stage,
        "note": outcome.note,
    }


def _solver_record_from_stages(result: RecoveryResult) -> dict[str, Any] | None:
    if not result.stages:
        return None
    last = result.stages[-1]
    return {
        "status": last.status,
        "iters": last.iterations,
        "time": float(sum(stage.solve_seconds for stage in result.stages)),
        "compile_time": float(sum(stage.compile_seconds for stage in result.stages)),
        "tol": dict(SOLVER_TOLERANCES),
        "solver": "CLARABEL",
        "stage": last.stage,
        "note": last.note,
        "n_solves": len(result.stages),
    }
