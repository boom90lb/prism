"""The convex constructor: the objective, the three paths, and what may authorize an order.

The failure this file exists to catch is an order the program forbade. Four ways that
happens, and one test class each:

* the objective is built on the wrong vector -- ``z`` instead of ``a = z * sigma_h``, or the
  fit's raw endpoints instead of :func:`prism.calibration.objective_endpoints` -- and the
  book is silently re-weighted or takes exposure a sign contradiction disabled;
* a constraint is stated as a soft term, or omitted, and the solution breaches it;
* the solver's own status is trusted, so an output the independent validator refuses is
  submitted anyway;
* a held name without an estimate is forced flat by a zero mask on a bar when the exit is
  not executable.

The straddling-interval case is tested as a *result*, not a bug: with ``lower < 0 < upper``
the adverse term's maximum sets ``a' w = 0`` and the arm holds nothing. The test asserts the
flat book **and** the diagnostics that explain it, because an unexplained flat arm in the
stage 4 comparison is indistinguishable from a defect.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.calibration import CalibrationFit
from prism.execution.cost_model import ConvexCostTerms, CostInputs, CostModel, CostStack
from prism.portfolio.convex import (
    ComponentForecast,
    ConstructionError,
    ConstructionInputs,
    ConvexConstructor,
    ConvexProgram,
    SolveOutcome,
)
from prism.portfolio.limits import Limit, LimitClass, LimitContext, violation
from prism.portfolio.obligations import obligation_from_current
from prism.portfolio.recovery import RecoveryPolicy
from prism.portfolio.validate import ValidationSettings
from prism.risk_model import RiskSnapshot

EQUITY = 1_000_000.0
HORIZON = 21
TOL = 1e-6

SETTINGS = {
    "kind": "convex",
    "risk_aversion": 0.0,
    "eps": 1.0e-8,
    "eligibility_exit": "reduce_only",
    "max_at_cap_fraction": 0.35,
    "band": {"mode": "off"},
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def universe(n: int = 20) -> pd.Index:
    return pd.Index([f"S{i:02d}" for i in range(n)], name="sid")


def scores(index: pd.Index, seed: int = 0) -> pd.Series:
    raw = pd.Series(np.random.default_rng(seed).normal(size=len(index)), index=index)
    return (raw - raw.mean()) / raw.std()


def fit(
    *,
    lower: float = 0.06,
    upper: float = 0.06,
    theta: float = 0.06,
    sign_ok: bool = True,
    confidence: float | None = None,
) -> CalibrationFit:
    return CalibrationFit(
        theta_hat=theta,
        theta_shrunk=theta,
        lower=lower,
        upper=upper,
        straddles_zero=lower < 0.0 < upper,
        n_periods=13,
        n_observations=1_248,
        shrink_factor=0.52,
        blocks_drawn=2_000,
        fit_cutoff="2026-05-01",
        sign_ok=sign_ok,
        excluded=False,
        exclusion_reason=None,
        disabled_reason=None,
        confidence=confidence,
        method={"estimator": "pooled_slope"},
    )


def cost_terms(index: pd.Index, *, linear_bps: float = 5.0) -> ConvexCostTerms:
    n = len(index)
    return ConvexCostTerms(
        sids=index,
        linear_bps=np.full(n, linear_bps),
        quadratic_coeff=np.zeros(n),
        power_three_halves_coeff=np.zeros(n),
        borrow_per_horizon=np.zeros(n),
        financing_per_horizon=0.0,
        cash_income_per_horizon=0.0,
        equity=EQUITY,
        horizon_bars=HORIZON,
    )


def full_cost_model(index: pd.Index, adv: pd.Series) -> CostModel:
    """A real cost stack, so the quadratic and power-1.5 impact terms enter the program."""
    dollar_volume = {
        key: {
            "estimator": "median",
            "window": "trailing",
            "window_bars": 20,
            "refresh": "decision",
            "floor_dollars": 0.0,
        }
        for key in ("spread_bucket", "participation")
    }
    dollar_volume["participation"]["base"] = "current_equity"
    dollar_volume["impact"] = {
        "estimator": "same_bar",
        "window": "trailing",
        "window_bars": 1,
        "refresh": "bar",
        "floor_dollars": 100_000.0,
    }
    stack = CostStack.from_mapping(
        {
            "spread": {"mode": "flat", "flat_bps": 2.0},
            "impact": {
                "quadratic_notional_coeff": 10.0,
                "participation_model": "sqrt",
                "participation_coeff": 5.0,
            },
            "dollar_volume": dollar_volume,
            "commission_bps": 1.0,
            "borrow_bps_annual": 50.0,
            "cash_income_bps_annual": 100.0,
            "financing_bps_annual": 200.0,
            "short_proceeds": "collateral_no_rebate",
            "holding_cost_settlement": "per_bar",
            "trading_days_per_year": 252,
        }
    )
    inputs = CostInputs(as_of="2026-05-01", adv_impact=adv, adv_participation=adv, adv_bucket=adv)
    return CostModel(stack, inputs, equity=EQUITY, horizon_bars=HORIZON)


def snapshot(index: pd.Index, *, seed: int = 1, singular: bool = False, drop: str | None = None) -> RiskSnapshot:
    rng = np.random.default_rng(seed)
    sids = [sid for sid in index if sid != drop]
    n = len(sids)
    loadings = rng.normal(size=(n, 2)) * 0.5
    omega = np.array([[4.0e-4, 3.6e-4], [3.6e-4, 4.0e-4]])
    if singular:
        # A PCA factor covariance is routinely PSD-singular; Cholesky raises on it and
        # `eigh` does not (D9).
        omega = np.array([[4.0e-4, 4.0e-4], [4.0e-4, 4.0e-4]])
    return RiskSnapshot(
        sids=tuple(sids),
        B=loadings,
        omega=omega,
        d=np.full(n, 4.0e-4),
        horizon_bars=HORIZON,
        scaling="per_bar_times_h",
        as_of="2026-06-01",
        method={
            "eigenvalue_floor_relative": 1.0e-10,
            "factor_ids": ["pca_1", "pca_2"],
            "estimator": "pca",
            "substitutions": {},
        },
    )


def limit(name: str, cls: str, priority: int | None = None, **params) -> Limit:
    return Limit(name=name, cls=LimitClass(cls), priority=priority, params=params, source="test")


def profile_limits(
    *,
    gross: float = 1.0,
    cap: float = 0.10,
    net: float = 0.02,
    budget: float | None = 0.0332,
    participation: float | None = 0.05,
) -> list[Limit]:
    """The stage 4 experiment profile's limits (examples/reference/profiles/experiment_v1.toml)."""
    rows = [
        limit("max_gross", "preference", 1, value=gross),
        limit("max_name_weight", "preference", 2, value=cap),
        limit("net_band", "preference", 3, value=net),
    ]
    if budget is not None:
        rows.append(limit("variance_budget", "preference", 4, value=budget))
    if participation is not None:
        rows.append(limit("participation_per_name", "physical", None, value=participation))
    return rows


NO_RISK = object()


def inputs(
    index: pd.Index,
    *,
    w0: pd.Series | None = None,
    z: pd.Series | None = None,
    sigma: pd.Series | None = None,
    fit_: CalibrationFit | None = None,
    limits: list[Limit] | None = None,
    costs=None,
    risk=None,
    adv: pd.Series | None = None,
    settings: dict | None = None,
    **kwargs,
) -> ConstructionInputs:
    z = scores(index) if z is None else z
    sigma = pd.Series(0.0722, index=index) if sigma is None else sigma
    adv = pd.Series(5.0e7, index=index) if adv is None else adv
    forecasts = kwargs.pop("forecasts", None)
    if forecasts is None:
        forecasts = [ComponentForecast("momentum", fit_ or fit(), z, sigma)]
    return ConstructionInputs(
        sids=list(index),
        w0=pd.Series(0.0, index=index) if w0 is None else w0,
        equity=EQUITY,
        reference_prices=pd.Series(100.0, index=index),
        limits=profile_limits() if limits is None else limits,
        costs=cost_terms(index) if costs is None else costs,
        decision_bar="2026-06-01",
        adv_dollars=adv,
        forecasts=forecasts,
        risk=None if risk is NO_RISK else (snapshot(index) if risk is None else risk),
        settings=dict(SETTINGS if settings is None else settings),
        cash_buffer=kwargs.pop("cash_buffer", 0.02),
        policy=kwargs.pop("policy", RecoveryPolicy()),
        **kwargs,
    )


def solve(x: ConstructionInputs, *, policy: RecoveryPolicy | None = None):
    return ConvexConstructor(policy=policy).construct(x)


# ---------------------------------------------------------------------------
# The objective: the two units contracts
# ---------------------------------------------------------------------------


def test_the_coefficient_vector_is_z_times_sigma_h_and_not_z_alone() -> None:
    """``a = z * sigma_h``. An objective on ``z`` alone reweights by inverse volatility.

    Two names carry the identical standardized score and different horizon volatility. The
    declared objective is indifferent between them only if the ``sigma`` multiplication is
    missing; with it, the high-volatility name carries the whole position.
    """
    index = pd.Index(["LOWVOL", "HIGHVOL"], name="sid")
    z = pd.Series([1.0, 1.0], index=index)
    sigma = pd.Series([0.05, 0.20], index=index)
    x = inputs(
        index,
        z=z,
        sigma=sigma,
        limits=[
            limit("max_gross", "preference", 1, value=0.10),
            limit("max_name_weight", "preference", 2, value=0.10),
        ],
        risk=snapshot(index),
        cash_buffer=0.0,
    )
    out = solve(x)
    assert out.path == "normal"
    assert out.w_target["HIGHVOL"] == pytest.approx(0.10, abs=1e-4)
    assert out.w_target["LOWVOL"] == pytest.approx(0.0, abs=1e-4)
    exposure = out.diagnostics["signal_exposure"]["momentum"]
    assert exposure["s"] == pytest.approx(float((z * sigma * out.w_target).sum()), rel=1e-6)


def test_a_sign_contradiction_disables_new_directional_exposure_and_records_it() -> None:
    """The endpoints come from ``objective_endpoints``, which applies the §3.4 sign policy.

    Reading ``fit.lower`` / ``fit.upper`` directly would take full exposure on a fit whose
    slope contradicts the declared sign; the policy is ``(0.0, 0.0)``, so the arm takes no
    new directional exposure and the evaluation continues.
    """
    index = universe(8)
    contradicted = fit(lower=0.06, upper=0.06, sign_ok=False)
    out = solve(inputs(index, fit_=contradicted))
    assert out.path == "normal"
    assert out.diagnostics["gross"] == pytest.approx(0.0, abs=1e-6)
    entry = out.diagnostics["signal_exposure"]["momentum"]
    assert entry["sign_contradiction"] is True
    assert entry["lower"] == 0.0 and entry["upper"] == 0.0
    assert entry["s"] == pytest.approx(0.0, abs=1e-9)
    # The fit's own fields are untouched: the record keeps the estimate that was made.
    assert contradicted.lower == 0.06


def test_a_sign_contradiction_with_unwind_reduces_the_existing_book_as_far_as_it_can() -> None:
    """``on_sign_contradiction`` is a policy value, and ``unwind`` is not an instant exit."""
    index = universe(4)
    held = pd.Series([0.20, 0.20, -0.20, -0.20], index=index)
    adv = pd.Series(2.0e6, index=index)  # 5% of 2e6 is 100k, i.e. 0.10 of a 1e6 book
    out = solve(
        inputs(
            index,
            w0=held,
            fit_=fit(sign_ok=False),
            adv=adv,
            policy=RecoveryPolicy(on_sign_contradiction="unwind"),
        )
    )
    assert set(out.diagnostics["unwind_on_sign_contradiction"]) == set(index)
    for sid in index:
        # Reduced by exactly the participation allowance, never further and never flipped.
        assert abs(out.w_target[sid]) == pytest.approx(0.10, abs=1e-4)
        assert np.sign(out.w_target[sid]) == np.sign(held[sid])


def test_the_straddling_interval_holds_nothing_and_the_diagnostics_say_why() -> None:
    """§3.7: with ``lower < 0 < upper`` the adverse term's maximum sets ``a' w = 0``.

    This is the designed output of the declared method, not a defect, so the assertion is
    on the *record* as much as on the book: an arm that is flat on most dates must be
    readable in the stage 4 comparison as a finding about the calibration's precision.
    """
    index = universe(20)
    positive = solve(inputs(index, fit_=fit(lower=0.4, upper=1.2, confidence=0.80)))
    straddling = solve(inputs(index, fit_=fit(lower=-0.3, upper=1.2, confidence=0.80)))

    assert positive.diagnostics["gross"] == pytest.approx(0.98, abs=1e-4)
    assert positive.diagnostics["flat_from_straddling_endpoints"] is False

    assert straddling.diagnostics["gross"] == pytest.approx(0.0, abs=1e-6)
    assert straddling.diagnostics["signal_exposure"]["momentum"]["s"] == pytest.approx(0.0, abs=1e-8)
    assert straddling.diagnostics["flat_from_straddling_endpoints"] is True
    assert straddling.diagnostics["straddling_components"] == ["momentum"]
    assert straddling.diagnostics["signal_exposure"]["momentum"]["straddles_zero"] is True
    assert straddling.diagnostics["signal_exposure"]["momentum"]["confidence"] == 0.80
    # Flat is not disabled: the arm stays enabled and keeps solving.
    assert straddling.path == "normal"
    assert straddling.diagnostics["validation"]["ok"] is True


def test_the_uncertainty_arm_prices_the_same_book_at_the_lower_endpoint() -> None:
    """The two arms differ only through the endpoints, and the difference has one sign.

    ``min(lower*s, upper*s)`` is ``lower*s`` for ``s >= 0``, so a lower endpoint below the
    point estimate scales the whole return term down against an unchanged cost term. The
    uncertainty arm therefore takes weakly less exposure and never a different direction --
    it does not merely reproduce the point arm's book, which is why the stage 4 comparison
    can read a smaller gross as the endpoint's doing rather than as noise.
    """
    index = universe(12)
    point = solve(inputs(index, fit_=fit(lower=0.06, upper=0.06)))
    wide = solve(inputs(index, fit_=fit(lower=0.02, upper=0.10, confidence=0.80)))
    s_point = point.diagnostics["signal_exposure"]["momentum"]["s"]
    s_wide = wide.diagnostics["signal_exposure"]["momentum"]["s"]
    assert 0.0 < s_wide <= s_point + 1e-9
    assert wide.diagnostics["gross"] <= point.diagnostics["gross"] + 1e-9
    held = point.w_target.abs() > 1e-4
    signs_point = np.sign(point.w_target[held].to_numpy())
    signs_wide = np.sign(wide.w_target[held].to_numpy())
    assert np.all((signs_wide == signs_point) | (np.abs(wide.w_target[held].to_numpy()) <= 1e-4))


# ---------------------------------------------------------------------------
# Constraint honoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_every_declared_limit_is_honored_on_the_normal_path(seed: int) -> None:
    """Property: over random cross-sections, no limit of the §3.7 table is breached.

    Measured by ``portfolio/limits.py``, which shares no code with the program's expression
    builders -- the point is that two independent statements of the same measure agree.
    """
    rng = np.random.default_rng(seed)
    index = universe(25)
    held = pd.Series(rng.normal(scale=0.02, size=len(index)), index=index)
    held = (held - held.mean()).clip(-0.08, 0.08)
    held = held - held.mean()  # inside the net band
    held = held * min(1.0, 0.6 / float(held.abs().sum()))  # inside max_gross and the per-name cap
    adv = pd.Series(rng.uniform(5.0e6, 1.0e8, size=len(index)), index=index)
    limits = profile_limits()
    x = inputs(index, z=scores(index, seed=seed), w0=held, adv=adv, limits=limits)
    ctx = LimitContext(w0=held, equity=EQUITY, adv_dollars=adv, risk=x.risk)
    for row in limits:  # the fixture's marked book must be legal, or the path is not the one under test
        assert violation(row, held, ctx) == 0.0, row.name
    out = solve(x)
    assert out.path == "normal", out.diagnostics.get("no_action")
    for row in limits:
        measure = violation(row, out.w_target, ctx)
        assert measure <= TOL, f"{row.name} breached by {measure}"
    assert out.diagnostics["validation"]["ok"] is True
    assert out.violations["introduced"] == []


def test_a_reduce_only_name_may_shrink_and_may_never_grow_or_flip() -> None:
    index = universe(6)
    held = pd.Series([0.30, 0.0, 0.0, -0.30, 0.0, 0.0], index=index)
    z = scores(index, seed=3)
    # The two held names carry no current estimate: they leave the scored cross-section.
    z = z.drop(["S00", "S03"])
    sigma = pd.Series(0.0722, index=z.index)
    out = solve(inputs(index, w0=held, z=z, sigma=sigma))
    assert set(out.diagnostics["reduce_only"]) >= {"S00", "S03"}
    assert 0.0 <= out.w_target["S00"] <= 0.30 + TOL
    assert -0.30 - TOL <= out.w_target["S03"] <= 0.0
    assert out.diagnostics["signal_exposure"]["momentum"]["n_zero_coefficient"] == 2


def test_a_held_name_without_an_estimate_is_never_forced_to_an_unavailable_liquidation() -> None:
    """§3.7, Q15: a zero holding mask is not used. The name reduces at the rate the
    participation restriction allows, and no further."""
    index = universe(4)
    held = pd.Series([0.40, 0.0, 0.0, 0.0], index=index)
    z = scores(index, seed=4).drop(["S00"])
    adv = pd.Series(2.0e6, index=index)  # 5% of 2e6 on a 1e6 book is a 0.10 weight move
    out = solve(inputs(index, w0=held, z=z, sigma=pd.Series(0.0722, index=z.index), adv=adv))
    assert out.w_target["S00"] == pytest.approx(0.30, abs=1e-4)
    assert not np.isnan(out.w_target["S00"])


def test_an_account_that_prohibits_shorting_opens_no_short_and_may_still_reduce_one() -> None:
    index = universe(4)
    held = pd.Series([0.0, 0.0, 0.0, -0.20], index=index)
    rows = profile_limits(cap=0.5, net=0.5, budget=None) + [
        limit("account_prohibits_shorting", "mandatory_action", None),
    ]
    out = solve(inputs(index, w0=held, limits=rows))
    assert (out.w_target[["S00", "S01", "S02"]] >= -TOL).all()
    assert -0.20 - TOL <= out.w_target["S03"] <= TOL


def test_a_name_without_borrow_may_not_be_shorted_further() -> None:
    index = universe(4)
    held = pd.Series([0.0, 0.0, 0.0, -0.05], index=index)
    rows = profile_limits(cap=0.5, net=0.5, budget=None) + [
        limit("no_new_short_without_borrow", "physical", None, sids=["S03"]),
    ]
    out = solve(inputs(index, w0=held, limits=rows))
    assert out.w_target["S03"] >= -0.05 - TOL


def test_a_buying_power_restriction_bounds_the_gross_in_dollars() -> None:
    index = universe(4)
    rows = profile_limits(cap=0.5, net=0.5, budget=None) + [
        limit("buying_power", "mandatory_action", None, value=300_000.0),
    ]
    out = solve(inputs(index, limits=rows, cash_buffer=0.0))
    assert out.diagnostics["gross"] <= 0.30 + 1e-6


def test_a_restriction_the_rounding_step_enforces_is_recorded_rather_than_solved() -> None:
    index = universe(4)
    rows = profile_limits() + [limit("quantity_increment", "physical", None, value=1.0)]
    out = solve(inputs(index, limits=rows))
    deferred = " ".join(out.diagnostics["deferred_action_restrictions"])
    assert "quantity_increment" in deferred
    assert "rounding" in deferred


def test_the_cash_buffer_reduces_the_gross_available_to_the_objective() -> None:
    index = universe(20)
    with_buffer = solve(inputs(index, cash_buffer=0.02))
    without = solve(inputs(index, cash_buffer=0.0))
    assert with_buffer.diagnostics["gross"] == pytest.approx(0.98, abs=1e-4)
    assert without.diagnostics["gross"] == pytest.approx(1.00, abs=1e-4)


def test_a_gross_limit_swallowed_by_the_cash_buffer_is_refused_rather_than_solved() -> None:
    index = universe(4)
    with pytest.raises(ConstructionError, match="cash buffer"):
        solve(inputs(index, limits=profile_limits(gross=0.01), cash_buffer=0.02))


# ---------------------------------------------------------------------------
# The variance budget: the channel through which the risk model acts
# ---------------------------------------------------------------------------


def test_the_variance_budget_is_the_only_channel_through_which_the_risk_model_acts() -> None:
    """D8, reproduced in the direction §3.7's Run states.

    With ``risk_aversion = 0`` and negligible cost curvature the program is linear in
    expected return and its optimum is a maximal-concentration corner. Tightening the budget
    is what moves it off that corner: the at-cap fraction and the signal exposure both fall.
    """
    index = universe(20)
    unbudgeted = solve(inputs(index, limits=profile_limits(budget=None)))
    sd = unbudgeted.diagnostics["ex_ante_sd"]
    assert unbudgeted.diagnostics["at_cap_fraction"] >= 0.8

    exposures, at_cap = [], []
    for multiple in (0.75, 0.50, 0.35, 0.25):
        out = solve(inputs(index, limits=profile_limits(budget=multiple * sd)))
        exposures.append(out.diagnostics["signal_exposure"]["momentum"]["s"])
        at_cap.append(out.diagnostics["at_cap_fraction"])
        assert out.diagnostics["ex_ante_sd"] <= multiple * sd + 1e-9
    assert exposures == sorted(exposures, reverse=True), exposures
    assert at_cap[-1] < unbudgeted.diagnostics["at_cap_fraction"]


def test_the_at_cap_fraction_is_recorded_on_every_decision() -> None:
    index = universe(20)
    out = solve(inputs(index))
    assert 0.0 <= out.diagnostics["at_cap_fraction"] <= 1.0
    assert out.diagnostics["max_at_cap_fraction"] == 0.35
    # No declared per-name cap means "not measured", never a reassuring zero.
    without = solve(inputs(index, limits=[limit("max_gross", "preference", 1, value=1.0)]))
    assert without.diagnostics["at_cap_fraction"] is None


def test_a_singular_factor_covariance_factorizes_rather_than_raising() -> None:
    """D9: ``eigh`` with a clipped floor, never Cholesky."""
    index = universe(10)
    singular = snapshot(index, singular=True)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(singular.omega)
    out = solve(inputs(index, risk=singular))
    assert out.path == "normal"
    assert out.diagnostics["validation"]["ok"] is True


def test_a_snapshot_that_does_not_cover_the_universe_is_rejected_before_any_solve(monkeypatch) -> None:
    index = universe(6)
    calls: list[str] = []
    monkeypatch.setattr(ConvexProgram, "solve", lambda *a, **k: calls.append("solved"))
    with pytest.raises(ConstructionError, match="rejected before any solve"):
        solve(inputs(index, risk=snapshot(index, drop="S02")))
    assert calls == []


def test_the_variance_cone_and_the_snapshot_agree_on_the_same_book() -> None:
    index = universe(15)
    out = solve(inputs(index))
    assert out.diagnostics["cone_vs_variance_relative"] < 1e-10


# ---------------------------------------------------------------------------
# Solver status never authorizes an order
# ---------------------------------------------------------------------------


def _forged(w: pd.Series) -> SolveOutcome:
    return SolveOutcome(
        status="optimal",
        optimal=True,
        value=1.0,
        w=w,
        solve_seconds=0.001,
        compile_seconds=0.001,
        iterations=7,
    )


def test_an_invalid_solver_output_is_rejected_even_when_the_solver_calls_it_optimal(monkeypatch) -> None:
    """The whole point of the independent validation: CLARABEL's status is not a permission.

    The forged output breaches the participation restriction, which is ``physical``: an
    introduced action-restriction violation fails on every path and under every
    ``preference_response``.
    """
    index = universe(4)
    forged = pd.Series([0.9, -0.9, 0.9, -0.9], index=index)
    adv = pd.Series(2.0e6, index=index)  # a 5% cap on 2e6 is a 0.10 weight move, not 0.9
    monkeypatch.setattr(ConvexProgram, "solve", lambda self, *a, **k: _forged(forged))
    out = solve(inputs(index, adv=adv))
    assert out.path == "no_action"
    assert out.w_target.isna().all(), "no orders means no decision, not a decision to hold"
    assert out.diagnostics["no_action"]["response"] == "cancel"
    assert out.diagnostics["validation"]["ok"] is False
    breached = {entry["limit_name"] for entry in out.diagnostics["validation"]["violations"]["introduced"]}
    assert "participation_per_name" in breached


def test_an_introduced_preference_violation_blocks_when_the_policy_says_block(monkeypatch) -> None:
    index = universe(4)
    forged = pd.Series([0.02, 0.02, 0.02, 0.02], index=index)  # net 0.08 against a 0.02 band
    monkeypatch.setattr(ConvexProgram, "solve", lambda self, *a, **k: _forged(forged))
    blocking = ValidationSettings(preference_response="block")
    recording = ValidationSettings(preference_response="record")
    index_inputs = dict(limits=profile_limits(participation=None, budget=None))
    assert solve(inputs(index, validation=blocking, **index_inputs)).path == "no_action"
    assert solve(inputs(index, validation=recording, **index_inputs)).path == "normal"


def test_solver_tolerance_is_measured_against_the_declared_limit_tolerance(monkeypatch) -> None:
    """A breach inside ``limit_tolerance`` is solver noise; one outside it is a violation."""
    index = universe(2)
    limits = [limit("max_gross", "preference", 1, value=1.0)]
    for excess, expected in ((1e-9, "normal"), (1e-3, "no_action")):
        forged = pd.Series([0.5 + excess / 2, -0.5 - excess / 2], index=index)
        monkeypatch.setattr(ConvexProgram, "solve", lambda self, *a, _w=forged, **k: _forged(_w))
        out = solve(inputs(index, limits=limits, validation=ValidationSettings(preference_response="block")))
        assert out.path == expected, (excess, out.violations)


def test_a_non_finite_solver_output_is_never_returned_as_a_target(monkeypatch) -> None:
    index = universe(3)

    def broken(self, objective, constraints, **kwargs):
        return SolveOutcome(
            status="optimal",
            optimal=True,
            value=1.0,
            w=pd.Series([np.nan, 0.0, 0.0], index=self.index),
            solve_seconds=0.0,
            compile_seconds=0.0,
        )

    monkeypatch.setattr(ConvexProgram, "solve", broken)
    out = solve(inputs(index))
    assert out.path == "no_action"
    assert out.diagnostics["validation"]["ok"] is False


# ---------------------------------------------------------------------------
# Costs, determinism, and the infeasible case
# ---------------------------------------------------------------------------


def test_cost_consistency_in_financial_units() -> None:
    """The program's cost terms and the cost model's own expression agree, in dollars."""
    index = universe(8)
    adv = pd.Series(2.0e7, index=index)
    model = full_cost_model(index, adv)
    x = inputs(index, costs=model, adv=adv)
    out = solve(x)
    terms = model.convex_terms(list(index), adv_dollars=adv)
    target = out.w_target.reindex(index).to_numpy(dtype=float)
    trade = target - x.w0.reindex(index).to_numpy(dtype=float)
    assert out.diagnostics["trade_cost_fraction"] == pytest.approx(terms.trade_costs(trade), rel=1e-12)
    assert out.diagnostics["holding_cost_fraction"] == pytest.approx(terms.holding_costs(target), rel=1e-12)
    assert out.diagnostics["modeled_cost_dollars"] == pytest.approx(terms.total(trade, target) * EQUITY, rel=1e-12)
    assert terms.power_three_halves_coeff.max() > 0.0, "the fixture must exercise the power-1.5 term"


def test_a_cash_rate_above_the_financing_rate_is_refused_rather_than_modeled() -> None:
    index = universe(4)
    terms = cost_terms(index)
    terms = ConvexCostTerms(
        sids=terms.sids,
        linear_bps=terms.linear_bps,
        quadratic_coeff=terms.quadratic_coeff,
        power_three_halves_coeff=terms.power_three_halves_coeff,
        borrow_per_horizon=terms.borrow_per_horizon,
        financing_per_horizon=0.001,
        cash_income_per_horizon=0.002,
        equity=EQUITY,
        horizon_bars=HORIZON,
    )
    with pytest.raises(ConstructionError, match="not convex in w"):
        solve(inputs(index, costs=terms))


def test_rerun_determinism_to_tolerance() -> None:
    index = universe(20)
    first = solve(inputs(index))
    second = solve(inputs(index))
    pd.testing.assert_series_equal(first.w_target, second.w_target, atol=1e-9, check_names=False)
    assert first.diagnostics["gross"] == pytest.approx(second.diagnostics["gross"], abs=1e-9)


def test_the_one_asset_infeasible_case_takes_the_declared_response_and_submits_nothing() -> None:
    """One name, an unwind the sign policy requires, and a turnover cap that forbids it.

    The two constraints have no common point, so the program is infeasible. Neither
    no-trade nor the zero portfolio is assumed executable: the output is ``no_action`` with
    the declared response and an all-NaN target, not ``w0`` and not zeros.
    """
    index = pd.Index(["ONE"], name="sid")
    held = pd.Series([0.30], index=index)
    limits = [
        limit("turnover_cap", "preference", 1, value=0.0),
        limit("participation_per_name", "physical", None, value=0.05),
    ]
    x = inputs(
        index,
        w0=held,
        z=pd.Series([1.0], index=index),
        sigma=pd.Series([0.0722], index=index),
        fit_=fit(sign_ok=False),
        limits=limits,
        policy=RecoveryPolicy(on_sign_contradiction="unwind", on_infeasible="escalate"),
    )
    out = solve(x)
    assert out.path == "no_action"
    assert out.solver["status"] in ("infeasible", "infeasible_inaccurate")
    assert out.w_target.isna().all()
    record = out.diagnostics["no_action"]
    assert record["response"] == "escalate"
    assert record["orders_submitted"] is False
    assert record["escalation"]["exit_code"] == 3


# ---------------------------------------------------------------------------
# Declarations the program refuses rather than approximating
# ---------------------------------------------------------------------------


def test_an_unrecognized_action_restriction_refuses_the_solve() -> None:
    index = universe(4)
    rows = profile_limits() + [limit("some_new_broker_rule", "mandatory_action", None, value=1.0)]
    with pytest.raises(ConstructionError, match="no convex form"):
        solve(inputs(index, limits=rows))


def test_a_no_trade_band_is_refused_because_it_would_breach_the_limits_it_follows() -> None:
    index = universe(4)
    settings = dict(SETTINGS, band={"mode": "constant", "value": 0.5})
    with pytest.raises(ConstructionError, match="band"):
        solve(inputs(index, settings=settings))


def test_the_exit_flavour_of_eligibility_exit_is_refused_by_the_convex_constructor() -> None:
    index = universe(4)
    settings = dict(SETTINGS, eligibility_exit="exit")
    with pytest.raises(ConstructionError, match="eligibility_exit"):
        solve(inputs(index, settings=settings))


def test_a_variance_budget_without_a_snapshot_is_refused_before_any_solve() -> None:
    index = universe(4)
    with pytest.raises(ConstructionError, match="only channel"):
        solve(inputs(index, risk=NO_RISK))


def test_a_calibration_and_a_snapshot_that_scale_volatility_differently_are_refused() -> None:
    from prism.calibration import CalibrationSpec

    index = universe(4)
    spec = CalibrationSpec.from_mapping(
        {
            "kind": "shared_slope",
            "standardize": "rank_normal",
            "weighting": "per_date_equal",
            "shrink_pseudo_periods": 12.0,
            "expected_sign": "positive",
            "uncertainty": "time_block_bootstrap",
            "block_scheme": "circular_moving",
            "block_periods": 3,
            "n_resamples": 2000,
            "seed": 20260906,
            "min_training_periods": 12,
            "min_blocks": 4,
            "inner_oos_required": False,
            "executable_return": "open_to_open",
            "label_distributions": "credited",
            "sigma_h": {"estimator": "ewma_daily", "window_bars": 63, "scaling": "per_bar_times_h"},
        },
        source="test",
    )
    # The snapshot of this fixture is stated at the horizon; the shipped calibration is not.
    at_horizon = snapshot(index)
    at_horizon = RiskSnapshot(
        sids=at_horizon.sids,
        B=at_horizon.B,
        omega=at_horizon.omega,
        d=at_horizon.d,
        horizon_bars=at_horizon.horizon_bars,
        scaling="estimated_at_horizon",
        as_of=at_horizon.as_of,
        method=dict(at_horizon.method),
    )
    with pytest.raises(Exception, match="commensurable"):
        solve(inputs(index, risk=at_horizon, calibration_spec=spec))


def test_cvxpy_is_not_imported_when_the_module_is(tmp_path: Path) -> None:
    """§3.7: cvxpy belongs to the ``construction`` extra and stays out of the import path."""
    probe = "import sys, prism.portfolio.convex, prism.portfolio.recovery; print('cvxpy' in sys.modules)"
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-P", "-c", probe],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_the_portfolio_package_does_not_import_the_convex_constructor() -> None:
    source = (Path(__file__).resolve().parents[1] / "src/prism/portfolio/__init__.py").read_text()
    assert "convex" not in source
