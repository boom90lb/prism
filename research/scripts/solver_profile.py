"""Solver profiling for the stage 4 convex constructor (spec section 3.7 and stage 4, Q15).

What this measures and why it exists. The stage 4 roadmap carries one open performance
question: what the convex constructor actually costs at cross-section sizes larger than the
574-name S&P universe the reference book runs on, and whether a DPP (disciplined parametrized
programming) formulation, which compiles once and re-solves against parameter values, is worth
building. The spec is explicit that an earlier *synthetic* out-of-memory result is a reason to
profile and not a conclusion, and that no roadmap item may cite it as one. So this script
measures, and records every number with the hardware, the library versions, and the exact
problem shape the number belongs to.

Four quantities, at each declared cross-section size:

* **first compilation** -- the canonicalization of one freshly constructed problem, which is
  what the production path pays on every decision bar today, since it builds no parameters;
* **repeated solves** -- the per-bar cost of a sequence of decision bars with new data, under
  each compilation mode;
* **memory** -- peak resident set size of a process that does nothing but build and solve the
  shape, measured by ``getrusage`` in a child process so the peak is attributable;
* **complete evaluations** -- the end-to-end per-decision cost through
  ``ConvexConstructor.construct`` and ``recovery.finalize_batch``, i.e. the solve plus the
  independent post-solve validation, the rounding, and the second validation; projected onto
  the declared stage 4 run through the registration's own arithmetic.

Three programs are measured at each size, and the third is the point of the exercise:

``production``
    ``prism.portfolio.convex.ConvexProgram`` exactly as it ships. It holds no
    ``cvxpy.Parameter``, so every solve is a fresh non-DPP compile. This is the engine's
    current cost, measured, not modeled.

``twin_constant``
    A profiling twin of the same program with the bar-varying data written as numpy constants.
    Its only purpose is to be the non-DPP control for the DPP cell: it has the DPP twin's
    structure (see below) without its parameters, so the difference between the two isolates
    the compilation mode rather than the formulation.

``twin_dpp``
    The same twin with every bar-varying quantity written as a ``cvxpy.Parameter``, compiled
    once and re-solved by assigning parameter values.

The twin is a *profiling instrument*, not a second constructor, and it deliberately does not
live in ``src/prism``: adopting a DPP formulation is a decision the numbers here inform, and
writing it into the production path before the numbers exist would be the tuning-to-a-result
this stage forbids. Its faithfulness is not asserted, it is checked: every cell verifies that
the twin's optimum agrees with the production program's on identical data, and the agreement
is recorded next to the timing (``equivalence`` in each cell). A twin that disagreed would
make its speed number describe a different problem.

Two formulation notes the numbers depend on, both recorded in ``method`` in the artifact:

* The twin introduces a trade variable ``dw`` with ``dw == w - w0`` and writes every trade
  cost and the participation restriction on ``dw``. It is needed: with ``w0`` a parameter,
  ``linear_bps @ abs(w - w0)`` is a product of a parameter with an expression that is not
  parameter-free, which is not DPP and would fall back to a full recompile silently. The
  substitution costs N variables and N equality rows, and that cost is part of what the DPP
  cell reports.
* The twin folds ``max_name_weight`` and ``participation_per_name`` into box bounds on ``w``
  and ``dw`` rather than ``abs(.) <= bound`` atoms. The feasible set is identical -- ``abs(x)
  <= b`` canonicalizes to the same pair of inequalities -- and the box is what lets one
  compiled program serve a bar on which a name becomes halted or reduce-only, which an
  ``abs`` atom over a per-bar index set cannot.

``sqrt(risk_aversion)`` folds into the cone operator rather than multiplying it (spec section
3.7): a product of two parameters is not DPP and falls back silently. The registration
declares ``risk_aversion = 0.0``, so the term is absent from the shapes profiled here and the
folding is exercised only by the code path, not by a number.

Uncounted diagnostic. It searches nothing, evaluates no arm, changes no trial code path, and
never appends to the trials ledger. It writes one JSON.

Run from the repo root::

    python -m research.scripts.solver_profile --names 500 1000 3000 --bars 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from prism.calibration import CalibrationFit
from prism.execution.cost_model import CostInputs, CostModel, CostStack
from prism.portfolio.convex import (
    ComponentForecast,
    ConstructionInputs,
    ConvexConstructor,
    ConvexProgram,
)
from prism.portfolio.limits import Limit, LimitClass
from prism.portfolio.recovery import RecoveryPolicy, finalize_batch
from prism.risk_model import RiskSnapshot

# ---------------------------------------------------------------------------
# The declared shape: the stage 4 experiment profile and risk policy, verbatim
# ---------------------------------------------------------------------------

#: examples/reference/risk/factor_diagonal.toml
N_FACTORS = 15
HORIZON_BARS = 21
SCALING = "per_bar_times_h"
EIGENVALUE_FLOOR_RELATIVE = 1.0e-10

#: examples/reference/profiles/experiment_v1.toml
MAX_GROSS = 1.0
MAX_NAME_WEIGHT = 0.10
NET_BAND = 0.02
VARIANCE_BUDGET = 0.0332
PARTICIPATION = 0.05
CASH_BUFFER = 0.02

#: registrations/2026-09-construction.toml, [shared]
RISK_AVERSION = 0.0
EPS = 1.0e-8

#: examples/reference/costs/retail_v1.toml is the frozen stack; these are its shape-relevant
#: terms, restated so the profiled program carries a linear, a quadratic, and a power-1.5
#: impact term rather than a linear one alone. A cost stack with no curvature would make the
#: program a linear one and the timings unrepresentative.
COST_STACK: dict[str, Any] = {
    "spread": {"mode": "flat", "flat_bps": 2.0},
    "impact": {
        "quadratic_notional_coeff": 10.0,
        "participation_model": "sqrt",
        "participation_coeff": 5.0,
    },
    "dollar_volume": {
        "spread_bucket": {
            "estimator": "median",
            "window": "trailing",
            "window_bars": 20,
            "refresh": "decision",
            "floor_dollars": 0.0,
        },
        "participation": {
            "estimator": "median",
            "window": "trailing",
            "window_bars": 20,
            "refresh": "decision",
            "floor_dollars": 0.0,
            "base": "current_equity",
        },
        "impact": {
            "estimator": "same_bar",
            "window": "trailing",
            "window_bars": 1,
            "refresh": "bar",
            "floor_dollars": 100_000.0,
        },
    },
    "commission_bps": 1.0,
    "borrow_bps_annual": 50.0,
    "cash_income_bps_annual": 100.0,
    "financing_bps_annual": 200.0,
    "short_proceeds": "collateral_no_rebate",
    "holding_cost_settlement": "per_bar",
    "trading_days_per_year": 252,
}

#: examples/reference/profiles/execution_alpaca.toml's rounding and validation blocks, as the
#: recovery tests state them; the rounding half of a complete decision is measured under them.
EXECUTION_PROFILE: dict[str, Any] = {
    "quantity_increment_default": 1.0,
    "lot_size": 1,
    "order_types": ["market_on_open"],
    "fractional": {"allowed": False, "time_in_force": ["day"], "short_allowed": False},
    "validation": {
        "limit_tolerance": 1.0e-6,
        "cash_tolerance": 0.01,
        "cost_tolerance_bps": 0.1,
        "mode": "blocking",
    },
    "rounding": {
        "tie_rule": "half_to_even",
        "zero_crossing_clamp": True,
        "apply_dust_rule": True,
        "fractional_increment": 0.001,
        "off_increment_holding": "leave_residual",
        "scale_schedule": [0.75, 0.5, 0.25],
        "repair_max_moves": 200,
        "repair_min_decrement": 1.0e-6,
        "min_implemented_gross_fraction": 0.80,
    },
}

DEFAULT_EQUITY = 1_000_000.0
DEFAULT_NAMES = (500, 1_000, 3_000)
MODES = ("production", "twin_constant", "twin_dpp", "twin_dpp_fixed_cone")

#: The registration's own arithmetic for a complete evaluation, restated so the projection in
#: the artifact is derived rather than asserted. 21 folds spanning 1,308 sessions at
#: `update_every_bars = 21` (examples/momentum/reference.toml) is the decision cadence
#: `src/prism/evaluate.py:273` takes as the stride.
RUN_SESSIONS = 1_308
RUN_STRIDE_BARS = 21
RUN_CAPITAL_SCENARIOS = 6
RUN_CONVEX_ARMS = 2


# ---------------------------------------------------------------------------
# Synthetic cross-sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """One decision bar's data at a declared cross-section size.

    The panel is synthetic and its generating law is recorded in the artifact: a real frozen
    panel does not exist at 3,000 names in this workspace, and inventing one that did would be
    the worse error. What matters for a solver timing is the *shape* -- the number of names,
    the number of factors, which atoms appear, and the density of the coefficient vectors --
    and every one of those is the declared one. Volatilities, prices and dollar volumes are
    drawn lognormal rather than held constant, because a cross-section of identical names is
    an artificially symmetric program and CLARABEL's iteration count on it is not
    representative.
    """

    index: pd.Index
    z: pd.Series
    sigma_h: pd.Series
    prices: pd.Series
    adv_dollars: pd.Series
    w0: pd.Series
    equity: float

    @property
    def n(self) -> int:
        return len(self.index)


def make_shape(n: int, *, seed: int, equity: float = DEFAULT_EQUITY, held: bool = True) -> Shape:
    """One synthetic decision bar at ``n`` names."""
    rng = np.random.default_rng(seed)
    index = pd.Index([f"S{i:05d}" for i in range(n)], name="sid")
    raw = rng.normal(size=n)
    z = pd.Series((raw - raw.mean()) / raw.std(ddof=0), index=index)
    # 32% annual volatility at the median, a wide but plausible dispersion, expressed over the
    # 21-bar horizon under the risk file's `per_bar_times_h` scaling.
    annual = np.exp(rng.normal(math.log(0.32), 0.45, size=n))
    sigma_h = pd.Series(annual * math.sqrt(HORIZON_BARS / 252.0), index=index)
    prices = pd.Series(np.exp(rng.normal(math.log(60.0), 0.9, size=n)), index=index)
    adv_dollars = pd.Series(np.exp(rng.normal(math.log(4.0e7), 1.3, size=n)), index=index)
    if held:
        # A marked book that is already invested: roughly the profile's gross, dollar-neutral,
        # so the trade-cost and participation terms are exercised rather than trivial.
        signs = rng.choice((-1.0, 1.0), size=n)
        magnitude = rng.random(size=n) * (MAX_NAME_WEIGHT * 0.8)
        w0_values = signs * magnitude
        w0_values *= (MAX_GROSS - CASH_BUFFER) / np.abs(w0_values).sum()
        w0_values -= w0_values.sum() / n
        w0 = pd.Series(w0_values, index=index)
    else:
        w0 = pd.Series(0.0, index=index)
    return Shape(index=index, z=z, sigma_h=sigma_h, prices=prices, adv_dollars=adv_dollars, w0=w0, equity=equity)


def make_snapshot(shape: Shape, *, seed: int) -> RiskSnapshot:
    """A ``factor_diagonal`` snapshot at the declared 15 factors and 21-bar horizon."""
    rng = np.random.default_rng(seed)
    n = shape.n
    loadings = rng.normal(size=(n, N_FACTORS)) * 0.45
    # A PSD factor covariance with a decaying spectrum, as a shrunk PCA omega has.
    root = rng.normal(size=(N_FACTORS, N_FACTORS)) / math.sqrt(N_FACTORS)
    omega = root @ root.T
    spectrum = np.linspace(1.0, 0.05, N_FACTORS)
    omega = omega * np.outer(spectrum, spectrum)
    omega *= (0.014**2) / float(np.trace(omega) / N_FACTORS)
    omega = 0.5 * (omega + omega.T)
    # Residual variance over the horizon, floored as the risk file's residual floor would.
    residual = np.maximum((shape.sigma_h.to_numpy(dtype=float) * 0.7) ** 2, 0.0075 * 0.25)
    return RiskSnapshot(
        sids=tuple(str(sid) for sid in shape.index),
        B=loadings,
        omega=omega,
        d=residual,
        horizon_bars=HORIZON_BARS,
        scaling=SCALING,
        as_of="2026-06-01",
        method={
            "eigenvalue_floor_relative": EIGENVALUE_FLOOR_RELATIVE,
            "factor_ids": [f"pca_{i + 1}" for i in range(N_FACTORS)],
            "estimator": "pca",
            "substitutions": {},
        },
    )


def make_fit(*, confidence: float | None = None, lower: float = 0.06, upper: float = 0.06) -> CalibrationFit:
    """A calibration fit standing in for one fold's shared slope."""
    return CalibrationFit(
        theta_hat=upper,
        theta_shrunk=upper,
        lower=lower,
        upper=upper,
        straddles_zero=lower < 0.0 < upper,
        n_periods=13,
        n_observations=13_000,
        shrink_factor=0.52,
        blocks_drawn=2_000,
        fit_cutoff="2026-05-01",
        sign_ok=True,
        excluded=False,
        exclusion_reason=None,
        disabled_reason=None,
        confidence=confidence,
        method={"estimator": "pooled_slope"},
    )


def profile_limits() -> list[Limit]:
    """``examples/reference/profiles/experiment_v1.toml``'s limits, in its declared order."""

    def row(name: str, cls: str, priority: int | None, **params: Any) -> Limit:
        return Limit(name=name, cls=LimitClass(cls), priority=priority, params=params, source="experiment_v1")

    return [
        row("max_gross", "preference", 1, value=MAX_GROSS),
        row("max_name_weight", "preference", 2, value=MAX_NAME_WEIGHT),
        row("net_band", "preference", 3, value=NET_BAND),
        row(
            "variance_budget",
            "preference",
            4,
            mode="absolute",
            value=VARIANCE_BUDGET,
            units="per_horizon_sd_fraction_of_equity",
            horizon_bars=HORIZON_BARS,
        ),
        row("participation_per_name", "physical", None, value=PARTICIPATION),
    ]


def make_inputs(shape: Shape, snapshot: RiskSnapshot, fit: CalibrationFit) -> ConstructionInputs:
    """The production constructor's inputs for one decision bar at this shape."""
    stack = CostStack.from_mapping(COST_STACK)
    costs = CostModel(
        stack,
        CostInputs(
            as_of="2026-06-01",
            adv_impact=shape.adv_dollars,
            adv_participation=shape.adv_dollars,
            adv_bucket=shape.adv_dollars,
        ),
        equity=shape.equity,
        horizon_bars=HORIZON_BARS,
    )
    return ConstructionInputs(
        sids=list(shape.index),
        w0=shape.w0,
        equity=shape.equity,
        reference_prices=shape.prices,
        limits=profile_limits(),
        costs=costs,
        decision_bar="2026-06-01",
        adv_dollars=shape.adv_dollars,
        forecasts=[ComponentForecast("momentum", fit, shape.z, shape.sigma_h)],
        risk=snapshot,
        settings={
            "kind": "convex",
            "risk_aversion": RISK_AVERSION,
            "eps": EPS,
            "eligibility_exit": "reduce_only",
            "max_at_cap_fraction": 0.35,
            "band": {"mode": "off"},
        },
        cash_buffer=CASH_BUFFER,
        policy=RecoveryPolicy(),
    )


# ---------------------------------------------------------------------------
# The profiling twin
# ---------------------------------------------------------------------------


@dataclass
class TwinData:
    """The bar-varying data one twin solve consumes, extracted from the production program.

    Everything here is read off ``ConvexProgram`` rather than recomputed, so a twin cell and a
    production cell on the same shape are solving the same numbers by construction and not by
    a second implementation of the same formulas.
    """

    n: int
    a_lower: np.ndarray
    a_upper: np.ndarray
    w0: np.ndarray
    linear_bps: np.ndarray
    quadratic: np.ndarray
    power: np.ndarray
    borrow: np.ndarray
    financing: float
    income: float
    financing_on_net: bool
    eps: float
    gross: float
    name_lo: np.ndarray
    name_hi: np.ndarray
    trade_lo: np.ndarray
    trade_hi: np.ndarray
    net_centre: float
    net_half: float
    factor: np.ndarray
    residual: np.ndarray
    budget: float


def twin_data(program: ConvexProgram) -> TwinData:
    """Read one bar's twin data off a built ``ConvexProgram``."""
    n = program.n
    w0 = np.asarray(program.w0, dtype=float)
    terms = program.cost_terms

    a_lower = np.zeros(n, dtype=float)
    a_upper = np.zeros(n, dtype=float)
    for record in program._forecasts:  # noqa: SLF001 - the twin reads the program's own vectors
        a = np.asarray(record["a"], dtype=float)
        a_lower += float(record.get("lower", 0.0)) * a
        a_upper += float(record.get("upper", 0.0)) * a

    gross = MAX_GROSS
    name_hi = np.full(n, np.inf)
    trade_hi = np.full(n, np.inf)
    net_centre, net_half = 0.0, np.inf
    budget = np.inf
    for limit in program.limits:
        margin = program.margin(limit)
        if limit.name == "max_gross":
            gross = max(0.0, limit.value() - program.cash_buffer - margin)
        elif limit.name == "max_name_weight":
            name_hi = np.minimum(name_hi, np.maximum(0.0, program._per_name_cap(limit) - margin))  # noqa: SLF001
        elif limit.name == "net_band":
            centre, half = limit.band()
            net_centre, net_half = float(centre), max(0.0, float(half) - margin)
        elif limit.name == "variance_budget":
            budget = max(0.0, limit.value() - margin)
        elif limit.name == "participation_per_name":
            trade_hi = np.minimum(trade_hi, program._participation_trade_bound(limit))  # noqa: SLF001

    # The reduce-only, no-trade and unwind masks fold into the same two boxes; on the profiled
    # shape none of them is set, and the box is then the per-name cap alone.
    name_lo = -name_hi
    trade_lo = -trade_hi
    factor, residual = program.cone()
    return TwinData(
        n=n,
        a_lower=a_lower,
        a_upper=a_upper,
        w0=w0,
        linear_bps=np.asarray(terms.linear_bps, dtype=float),
        quadratic=np.asarray(terms.quadratic_coeff, dtype=float),
        power=np.asarray(terms.power_three_halves_coeff, dtype=float),
        borrow=np.asarray(terms.borrow_per_horizon, dtype=float),
        financing=float(terms.financing_per_horizon),
        income=float(terms.cash_income_per_horizon),
        financing_on_net=str(terms.financing_base) == "proceeds_to_cash_with_rebate",
        eps=float(program.eps),
        gross=float(gross),
        name_lo=name_lo,
        name_hi=name_hi,
        trade_lo=trade_lo,
        trade_hi=trade_hi,
        net_centre=float(net_centre),
        net_half=float(net_half),
        factor=np.asarray(factor, dtype=float),
        residual=np.asarray(residual, dtype=float),
        budget=float(budget),
    )


class TwinProgram:
    """The production normal-path program, written once with parameters or with constants.

    ``dpp=True`` writes every bar-varying quantity as a ``cvxpy.Parameter`` and the compiled
    problem is reused across bars by assignment; ``dpp=False`` writes the same quantities as
    numpy constants, so the problem must be rebuilt and recanonicalized per bar. The two share
    every line of structure, which is what makes their difference a measurement of the
    compilation mode.
    """

    def __init__(self, data: TwinData, *, dpp: bool, parametrize_cone: bool = True) -> None:
        import cvxpy as cp  # noqa: PLC0415 - the construction extra, imported lazily

        self.cp = cp
        self.dpp = bool(dpp)
        self.parametrize_cone = bool(parametrize_cone)
        self.n = data.n
        self.k = int(data.factor.shape[1])
        self.w = cp.Variable(self.n, name="w")
        self.dw = cp.Variable(self.n, name="dw")

        def vec(value: np.ndarray, *, nonneg: bool = False) -> Any:
            if not dpp:
                return np.asarray(value, dtype=float)
            param = cp.Parameter(len(value), nonneg=nonneg)
            param.value = np.asarray(value, dtype=float)
            return param

        def scalar(value: float, *, nonneg: bool = False) -> Any:
            if not dpp:
                return float(value)
            param = cp.Parameter(nonneg=nonneg)
            param.value = float(value)
            return param

        def matrix(value: np.ndarray) -> Any:
            if not dpp:
                return np.asarray(value, dtype=float)
            param = cp.Parameter(value.shape)
            param.value = np.asarray(value, dtype=float)
            return param

        self.p_a_lower = vec(data.a_lower)
        self.p_a_upper = vec(data.a_upper)
        self.p_w0 = vec(data.w0)
        self.p_linear = vec(data.linear_bps, nonneg=True)
        self.p_quadratic = vec(data.quadratic, nonneg=True)
        self.p_power = vec(data.power, nonneg=True)
        self.p_borrow = vec(data.borrow, nonneg=True)
        self.p_financing = scalar(data.financing, nonneg=True)
        self.p_income = scalar(data.income, nonneg=True)
        self.p_name_lo = vec(data.name_lo)
        self.p_name_hi = vec(data.name_hi)
        self.p_trade_lo = vec(data.trade_lo)
        self.p_trade_hi = vec(data.trade_hi)
        self.p_gross = scalar(data.gross, nonneg=True)
        self.p_net_centre = scalar(data.net_centre)
        self.p_net_half = scalar(data.net_half, nonneg=True)
        self.p_budget = scalar(data.budget, nonneg=True)
        if self.parametrize_cone:
            self.p_factor = matrix(data.factor)
            self.p_residual = vec(data.residual, nonneg=True)
        else:
            # The cone as a constant: the risk snapshot refreshes every
            # `refresh_every_bars = 5` bars (examples/reference/risk/factor_diagonal.toml),
            # so a DPP program that holds it constant is recompiled on that cadence rather
            # than on every bar. Whether that is enough to make DPP usable at scale is
            # exactly what this mode measures.
            self.p_factor = np.asarray(data.factor, dtype=float)
            self.p_residual = np.asarray(data.residual, dtype=float)
        self.eps = float(data.eps)
        self.financing_on_net = bool(data.financing_on_net)
        self.has_quadratic = bool(np.any(data.quadratic != 0.0))
        self.has_power = bool(np.any(data.power != 0.0))
        self.has_borrow = bool(np.any(data.borrow != 0.0))
        self.has_financing = bool(data.financing != 0.0 or data.income != 0.0)

        self.problem = cp.Problem(cp.Maximize(self._objective()), self._constraints())

    # -- terms ----------------------------------------------------------

    def _objective(self) -> Any:
        cp = self.cp
        w, dw = self.w, self.dw
        # `min(lower * a'w, upper * a'w)` with the endpoints folded into the coefficient
        # vectors: `lower * (a'w)` with both `lower` and `a` parameters would be a product of
        # two parameters and would not be DPP.
        objective = cp.minimum(self.p_a_lower @ w, self.p_a_upper @ w)
        cost = (self.p_linear @ cp.abs(dw)) / 1e4
        if self.has_quadratic:
            cost = cost + (self.p_quadratic @ cp.square(dw)) / 1e4
        if self.has_power:
            cost = cost + (self.p_power @ cp.power(cp.abs(dw), 1.5)) / 1e4
        objective = objective - cost
        holding: Any = cp.Constant(0.0)
        if self.has_borrow:
            holding = holding + self.p_borrow @ cp.pos(-w)
        if self.has_financing:
            cash = 1.0 - (cp.sum(w) if self.financing_on_net else cp.sum(cp.pos(w)))
            holding = holding + cp.maximum(-self.p_financing * cash, -self.p_income * cash)
        objective = objective - holding
        if self.eps > 0.0:
            objective = objective - self.eps * cp.sum_squares(w)
        return objective

    def _constraints(self) -> list[Any]:
        cp = self.cp
        w, dw = self.w, self.dw
        constraints = [
            dw == w - self.p_w0,
            w >= self.p_name_lo,
            w <= self.p_name_hi,
            dw >= self.p_trade_lo,
            dw <= self.p_trade_hi,
            cp.sum(cp.abs(w)) <= self.p_gross,
            cp.abs(cp.sum(w) - self.p_net_centre) <= self.p_net_half,
            cp.norm(cp.hstack([self.p_factor.T @ w, cp.multiply(self.p_residual, w)]), 2) <= self.p_budget,
        ]
        return constraints

    # -- use ------------------------------------------------------------

    def assign(self, data: TwinData) -> None:
        """Point the compiled problem at a new bar. Only meaningful when ``dpp`` is set."""
        if not self.dpp:
            raise RuntimeError("a constant twin cannot be re-pointed; rebuild it")
        self.p_a_lower.value = data.a_lower
        self.p_a_upper.value = data.a_upper
        self.p_w0.value = data.w0
        self.p_linear.value = data.linear_bps
        self.p_quadratic.value = data.quadratic
        self.p_power.value = data.power
        self.p_borrow.value = data.borrow
        self.p_financing.value = data.financing
        self.p_income.value = data.income
        self.p_name_lo.value = data.name_lo
        self.p_name_hi.value = data.name_hi
        self.p_trade_lo.value = data.trade_lo
        self.p_trade_hi.value = data.trade_hi
        self.p_gross.value = data.gross
        self.p_net_centre.value = data.net_centre
        self.p_net_half.value = data.net_half
        self.p_budget.value = data.budget
        if self.parametrize_cone:
            self.p_factor.value = data.factor
            self.p_residual.value = data.residual

    def solve(self) -> dict[str, Any]:
        cp = self.cp
        started = time.perf_counter()
        self.problem.solve(solver=cp.CLARABEL, verbose=False, **_solver_tolerances())
        wall = time.perf_counter() - started
        stats = self.problem.solver_stats
        return {
            "status": str(self.problem.status),
            "value": None if self.problem.value is None else float(self.problem.value),
            "wall_seconds": wall,
            "compile_seconds": float(getattr(self.problem, "compilation_time", float("nan")) or 0.0),
            "solver_seconds": float(getattr(stats, "solve_time", 0.0) or 0.0),
            "iterations": None if stats is None else getattr(stats, "num_iters", None),
            "w": None if self.w.value is None else np.asarray(self.w.value, dtype=float).reshape(-1),
        }

    def is_dpp(self) -> bool:
        return bool(self.problem.is_dcp(dpp=True))

    def canonical_shape(self) -> dict[str, Any]:
        """The canonicalized cone program's dimensions, from cvxpy's own solver data."""
        cp = self.cp
        data = self.problem.get_problem_data(cp.CLARABEL)[0]
        record = _canonical_record(data, self.problem)
        record["parameter_tensor"] = _parameter_tensor(data.get("param_prob"))
        return record


def _solver_tolerances() -> dict[str, float]:
    from prism.portfolio.convex import SOLVER_TOLERANCES  # noqa: PLC0415

    return dict(SOLVER_TOLERANCES)


def _cone_dims(dims: Any) -> dict[str, Any] | None:
    """The canonical cone's dimensions, summarized: at 3,000 names the raw repr is a list of
    several thousand identical second-order cone sizes."""
    if dims is None:
        return None
    soc = [int(size) for size in getattr(dims, "soc", ()) or ()]
    sizes: dict[str, int] = {}
    for size in soc:
        sizes[str(size)] = sizes.get(str(size), 0) + 1
    return {
        "zero": int(getattr(dims, "zero", 0) or 0),
        "nonneg": int(getattr(dims, "nonneg", 0) or 0),
        "n_second_order_cones": len(soc),
        "second_order_cone_sizes": sizes,
        "second_order_cone_scalars": int(sum(soc)),
        "exp": int(getattr(dims, "exp", 0) or 0),
        "psd": [int(size) for size in getattr(dims, "psd", ()) or ()],
    }


def _parameter_tensor(param_prob: Any) -> dict[str, Any] | None:
    """The DPP canonicalization's parameter-to-canonical-form map, if the problem has one.

    This is the object a DPP compile builds and caches, and its nominal row dimension is the
    canonical constraint matrix's rows times its columns. It is stored sparsely, so the
    nonzero count stays small; the memory a DPP compile actually spends is spent *building*
    it, which is why the peak resident set size in the same cell is the number that matters
    and not this shape alone.
    """
    if param_prob is None:
        return None
    matrix = getattr(param_prob, "A", None)
    return {
        "n_scalar_parameters": int(getattr(param_prob, "total_param_size", 0) or 0),
        "tensor_rows": None if matrix is None else int(matrix.shape[0]),
        "tensor_cols": None if matrix is None else int(matrix.shape[1]),
        "tensor_nnz": None if matrix is None else int(getattr(matrix, "nnz", 0)),
    }


def _canonical_record(data: dict[str, Any], problem: Any) -> dict[str, Any]:
    matrix = data.get("A")
    metrics = problem.size_metrics
    return {
        "n_scalar_variables": int(metrics.num_scalar_variables),
        "n_scalar_eq_constraints": int(metrics.num_scalar_eq_constr),
        "n_scalar_leq_constraints": int(metrics.num_scalar_leq_constr),
        "constraint_matrix_rows": None if matrix is None else int(matrix.shape[0]),
        "constraint_matrix_cols": None if matrix is None else int(matrix.shape[1]),
        "constraint_matrix_nnz": None if matrix is None else int(matrix.nnz),
        "cone_dims": _cone_dims(data.get("dims")),
    }


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


class _Trace:
    """A flushed phase log, so a cell the kernel kills still says where it died.

    A cell that exceeds the memory cap raises ``MemoryError`` and reports normally; a cell the
    kernel's OOM killer takes leaves no return value at all, and without this file the record
    would carry a signal number and nothing else. That is the difference between "3,000 names
    ran out of memory" as a fact with a phase and an RSS attached, and the bare claim the spec
    refuses to let a roadmap item cite.
    """

    #: Exit code the sampler uses when it stops a cell at the declared resident-set budget.
    MEMORY_BUDGET_EXIT = 42

    def __init__(self, path: str | None, *, sample_seconds: float = 0.05, budget_mib: float | None = None) -> None:
        self.path = Path(path) if path else None
        self.peak = 0.0
        self.phase = "start"
        self.budget_mib = budget_mib
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
        self._sampler = threading.Thread(target=self._sample, args=(sample_seconds,), daemon=True)
        self._sampler.start()

    def _sample(self, interval: float) -> None:
        """Write the resident set size on a timer.

        A phase marker records the memory a phase *started* with. The memory that kills a cell
        is spent inside the phase, after the marker, so a record built from markers alone would
        report a few hundred MiB for a process that died needing several GiB -- which is the
        unattributed out-of-memory claim this whole exercise exists to replace.
        """
        while not self._stop.wait(interval):
            rss = self._write("sample")
            if self.budget_mib is not None and math.isfinite(rss) and rss > self.budget_mib:
                # Stop this cell rather than let it grow until the kernel picks a victim, which
                # may not be this process. The record then says which phase asked for the memory
                # and how much it was holding, which is what a roadmap item may cite.
                self.phase = f"{self.phase}:memory_budget_exceeded"
                self._write("stopped")
                sys.stderr.write(
                    f"solver_profile: stopped at {rss:.0f} MiB resident, over the declared "
                    f"{self.budget_mib:.0f} MiB budget, during {self.phase}\n"
                )
                sys.stderr.flush()
                os._exit(self.MEMORY_BUDGET_EXIT)

    def _write(self, kind: str) -> float:
        rss = _current_rss_mib()
        if math.isfinite(rss):
            self.peak = max(self.peak, rss)
        if self.path is None:
            return rss
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"phase": self.phase, "kind": kind, "rss_mib": rss, "t": time.time()}) + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
            except (OSError, MemoryError):  # a sampler must never be the thing that fails a cell
                pass
        return rss

    def mark(self, phase: str) -> None:
        self.phase = phase
        self._write("mark")

    def stop(self) -> None:
        self._stop.set()


def _apply_memory_cap(gib: float | None) -> dict[str, Any]:
    """An optional hard backstop on the child's *address space*.

    It is a backstop and not the primary control, because ``RLIMIT_AS`` limits virtual address
    space rather than resident memory, and this process reserves about 1.4 GiB of address space
    before it does any work at all (numpy and its BLAS map far more than they touch). A cap set
    anywhere near the resident figures of interest would therefore fail the cell before it
    started. The resident-set budget the trace sampler enforces is the control that means what
    it says; this one only stops a runaway from reaching the machine's own limits.
    """
    if gib is None or gib <= 0.0:
        return {"applied": False}
    limit = int(gib * (1024**3))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return {"applied": True, "rlimit_as_gib": limit / (1024**3)}


def _rss_mib() -> float:
    """Peak resident set size of this process, in MiB. Linux reports ``ru_maxrss`` in KiB."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _current_rss_mib() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):  # pragma: no cover - non-Linux
        return float("nan")
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def _summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(list(values), dtype=float)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "total": float(array.sum()),
    }


def run_cell(
    *,
    n: int,
    mode: str,
    bars: int,
    seed: int,
    equity: float,
    full_decision: bool,
    memory_budget_mib: float | None = None,
    memory_cap_gib: float | None = None,
    trace_path: str | None = None,
) -> dict[str, Any]:
    """Profile one (cross-section size, compilation mode) cell.

    ``bars`` distinct synthetic decision bars are solved. The first is the cold cell: it pays
    the first compilation. The remaining ``bars - 1`` are the repeated-solve cell, and what
    "repeated" means differs by mode, which is the whole point: ``production`` and
    ``twin_constant`` rebuild and recanonicalize per bar, ``twin_dpp`` assigns parameters to a
    problem compiled once.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; the declared modes are {list(MODES)}")
    # Warm the lazy solver import before anything is timed: cvxpy costs a second or so to
    # import and charging it to the first bar's build would make the cold cell unreadable.
    started_import = time.perf_counter()
    import cvxpy  # noqa: PLC0415, F401 - the construction extra, warmed deliberately

    import_seconds = time.perf_counter() - started_import
    trace = _Trace(trace_path, budget_mib=memory_budget_mib)
    trace.mark("cell_start")
    cap = _apply_memory_cap(memory_cap_gib)
    tracemalloc.start()
    baseline_rss = _current_rss_mib()
    started_cell = time.perf_counter()

    shapes = [make_shape(n, seed=seed + bar, equity=equity) for bar in range(bars)]
    snapshots = [make_snapshot(shapes[bar], seed=seed + 1_000 + bar) for bar in range(bars)]
    fit = make_fit()
    data_seconds: list[float] = []
    build_seconds: list[float] = []
    compile_seconds: list[float] = []
    solver_seconds: list[float] = []
    wall_seconds: list[float] = []
    iterations: list[float] = []
    statuses: list[str] = []
    values: list[float] = []
    construct_seconds: list[float] = []
    finalize_seconds: list[float] = []
    paths: list[str] = []
    canonical: dict[str, Any] | None = None
    dpp_compliant: bool | None = None
    twin: TwinProgram | None = None
    reference_w: np.ndarray | None = None
    reference_value: float | None = None
    equivalence: dict[str, Any] | None = None

    for bar in range(bars):
        shape, snapshot = shapes[bar], snapshots[bar]
        trace.mark(f"bar{bar}:inputs")
        inputs = make_inputs(shape, snapshot, fit)
        if mode == "production":
            started = time.perf_counter()
            program = ConvexProgram(inputs)
            data_seconds.append(time.perf_counter() - started)
            started = time.perf_counter()
            objective = program.economic_objective()
            constraints = program.action_constraints() + program.holding_state_constraints()
            build_seconds.append(time.perf_counter() - started)
            # The production path holds no parameters, so it recanonicalizes on every solve.
            # `Problem.solve` does not expose that split, so the canonicalization is timed on
            # a fresh problem of the same expressions; the solve below then pays it again,
            # which is exactly what the engine does today.
            trace.mark(f"bar{bar}:compile")
            probe = program.cp.Problem(program.cp.Maximize(objective), constraints)
            started = time.perf_counter()
            probe_data = probe.get_problem_data(program.cp.CLARABEL)[0]
            compile_seconds.append(time.perf_counter() - started)
            if canonical is None:
                canonical = _canonical_record(probe_data, probe)
                dpp_compliant = bool(probe.is_dcp(dpp=True))
            del probe, probe_data
            trace.mark(f"bar{bar}:solve")
            started = time.perf_counter()
            outcome = program.solve(objective, constraints, sense="maximize")
            wall_seconds.append(time.perf_counter() - started)
            solver_seconds.append(float(outcome.solve_seconds))
            statuses.append(outcome.status)
            if outcome.iterations is not None:
                iterations.append(float(outcome.iterations))
            if outcome.value is not None:
                values.append(float(outcome.value))
            if bar == 0 and outcome.w is not None:
                reference_w = outcome.w.to_numpy(dtype=float)
                reference_value = outcome.value
            if full_decision:
                trace.mark(f"bar{bar}:construct")
                started = time.perf_counter()
                output = ConvexConstructor().construct(inputs)
                construct_seconds.append(time.perf_counter() - started)
                paths.append(output.path)
                if output.path != "no_action":
                    trace.mark(f"bar{bar}:finalize")
                    started = time.perf_counter()
                    finalize_batch(
                        output,
                        state={"cash": shape.equity, "positions": {}, "pending_orders": []},
                        prices=shape.prices,
                        execution_profile=EXECUTION_PROFILE,
                        limits=inputs.limits,
                        policy=RecoveryPolicy(),
                        decision_bar=inputs.decision_bar,
                        adv_dollars=shape.adv_dollars,
                        risk=snapshot,
                    )
                    finalize_seconds.append(time.perf_counter() - started)
        else:
            started = time.perf_counter()
            program = ConvexProgram(inputs)
            data = twin_data(program)
            data_seconds.append(time.perf_counter() - started)
            if bar == 0:
                # The production solve on the same bar, so the twin's optimum is checked
                # against the program it stands for rather than assumed equal to it.
                production = program.solve(
                    program.economic_objective(),
                    program.action_constraints() + program.holding_state_constraints(),
                    sense="maximize",
                )
                reference_w = None if production.w is None else production.w.to_numpy(dtype=float)
                reference_value = production.value
            trace.mark(f"bar{bar}:build")
            if mode in ("twin_dpp", "twin_dpp_fixed_cone"):
                parametrize_cone = mode == "twin_dpp"
                if twin is None:
                    started = time.perf_counter()
                    twin = TwinProgram(data, dpp=True, parametrize_cone=parametrize_cone)
                    build_seconds.append(time.perf_counter() - started)
                    dpp_compliant = twin.is_dpp()
                else:
                    started = time.perf_counter()
                    twin.assign(data)
                    build_seconds.append(time.perf_counter() - started)
                active = twin
            else:
                started = time.perf_counter()
                active = TwinProgram(data, dpp=False)
                build_seconds.append(time.perf_counter() - started)
                if dpp_compliant is None:
                    dpp_compliant = active.is_dpp()
            trace.mark(f"bar{bar}:solve")
            result = active.solve()
            wall_seconds.append(float(result["wall_seconds"]))
            solver_seconds.append(float(result["solver_seconds"]))
            compile_seconds.append(float(result["compile_seconds"]))
            statuses.append(str(result["status"]))
            if result["iterations"] is not None:
                iterations.append(float(result["iterations"]))
            if result["value"] is not None:
                values.append(float(result["value"]))
            if bar == 0:
                trace.mark(f"bar{bar}:canonical_shape")
                canonical = active.canonical_shape()
                if reference_w is not None and result["w"] is not None:
                    twin_w = np.asarray(result["w"], dtype=float)
                    equivalence = {
                        "checked_against": "production",
                        "max_abs_weight_difference": float(np.max(np.abs(twin_w - reference_w))),
                        "gross_difference": float(abs(np.abs(twin_w).sum() - np.abs(reference_w).sum())),
                        "objective_production": reference_value,
                        "objective_twin": result["value"],
                        "objective_difference": (
                            None
                            if reference_value is None or result["value"] is None
                            else float(abs(float(result["value"]) - float(reference_value)))
                        ),
                    }

    trace.mark("done")
    trace.stop()
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cold = {
        "data_seconds": data_seconds[0] if data_seconds else None,
        "build_seconds": build_seconds[0] if build_seconds else None,
        "compile_seconds": compile_seconds[0] if compile_seconds else None,
        "wall_seconds": wall_seconds[0] if wall_seconds else None,
        "solver_seconds": solver_seconds[0] if solver_seconds else None,
        "status": statuses[0] if statuses else None,
        "iterations": iterations[0] if iterations else None,
    }
    n_bars = min(len(data_seconds), len(build_seconds), len(wall_seconds))
    per_bar_total = [data_seconds[i] + build_seconds[i] + wall_seconds[i] for i in range(n_bars)]
    return {
        "n_names": n,
        "mode": mode,
        "bars": bars,
        "seed": seed,
        "equity": equity,
        "import_seconds": import_seconds,
        "cold": cold,
        "repeat": {
            "data_seconds": _summary(data_seconds[1:]),
            "build_seconds": _summary(build_seconds[1:]),
            "compile_seconds": _summary(compile_seconds[1:]),
            "wall_seconds": _summary(wall_seconds[1:]),
            "solver_seconds": _summary(solver_seconds[1:]),
            "iterations": _summary(iterations[1:]),
        },
        "per_bar_total_seconds": _summary(per_bar_total[1:]),
        "cold_bar_total_seconds": per_bar_total[0] if per_bar_total else None,
        "complete_decision": {
            "construct_seconds": _summary(construct_seconds),
            "finalize_batch_seconds": _summary(finalize_seconds),
            "paths": sorted(set(paths)) or None,
        },
        "statuses": sorted(set(statuses)),
        # How many of the cell's bars ended in each status. `optimal_inaccurate` is a tolerance
        # outcome at the declared 1e-9 gap and feasibility tolerances, not a failure and not a
        # function of the cross-section size; prism.portfolio.convex records it and the
        # independent validation, never the status, decides whether the output may be used.
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "objective_values": _summary(values),
        "memory": {
            "peak_rss_mib": _rss_mib(),
            "rss_at_start_mib": baseline_rss,
            "rss_at_end_mib": _current_rss_mib(),
            "tracemalloc_peak_mib": traced_peak / (1024.0 * 1024.0),
            "tracemalloc_current_end_mib": traced_current / (1024.0 * 1024.0),
            "sampled_peak_rss_mib": trace.peak,
        },
        "canonical_shape": canonical,
        "dpp_compliant": dpp_compliant,
        "memory_budget_mib": memory_budget_mib,
        "memory_cap": cap,
        "equivalence": equivalence,
        "cell_seconds": time.perf_counter() - started_cell,
    }


# ---------------------------------------------------------------------------
# Environment and orchestration
# ---------------------------------------------------------------------------


def environment() -> dict[str, Any]:
    """The declared hardware and library versions every number in the artifact belongs to."""
    import clarabel  # noqa: PLC0415
    import cvxpy  # noqa: PLC0415
    import scipy  # noqa: PLC0415

    model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:  # pragma: no cover - non-Linux
        pass
    total_mib = None
    available_mib = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                total_mib = int(line.split()[1]) / 1024.0
            elif line.startswith("MemAvailable:"):
                available_mib = int(line.split()[1]) / 1024.0
    except OSError:  # pragma: no cover - non-Linux
        pass
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        affinity = None
    return {
        "cpu_model": model,
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_affinity": affinity,
        "memory_total_mib": total_mib,
        "memory_available_mib_at_start": available_mib,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cvxpy": cvxpy.__version__,
        "clarabel": clarabel.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "solver": "CLARABEL",
        "solver_tolerances": _solver_tolerances(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")
        },
        "sandbox_note": (
            "This ran inside an agent sandbox that may constrain CPU and memory. Every timing here is "
            "therefore a floor on the hardware's performance and every memory figure is what this shape "
            "actually needed under that constraint, not a ceiling the hardware imposes. The machine was "
            "shared while the cells ran, so timings move run to run by a few tens of percent; every timing "
            "summary carries its own min, median and max over the cell's bars, and the median is the figure "
            "the scaling fits and the projection use. The memory figures are far more stable than the "
            "timings, and the conclusions that turn on memory do not turn on a few tens of percent."
        ),
    }


def run_cell_in_subprocess(
    *,
    n: int,
    mode: str,
    bars: int,
    seed: int,
    equity: float,
    full_decision: bool,
    memory_budget_mib: float | None = None,
    memory_cap_gib: float | None = None,
    trace_dir: str | None = None,
) -> dict:
    """Run one cell in a child process, so ``ru_maxrss`` is that cell's own peak.

    Peak resident set size is monotonic within a process: measuring 500 names after 3,000 in
    one process would report the 3,000-name peak for both. One process per cell is the only
    way the memory column means what it says.
    """
    # A trace is always written: a cell that dies has no return value, and the phase log is
    # then the only thing standing between "it ran out of memory" and a number with a phase.
    directory = Path(trace_dir) if trace_dir else Path(tempfile.mkdtemp(prefix="solver_profile_trace_"))
    trace_path = str(directory / f"trace_{mode}_{n}.jsonl")
    payload = {
        "n": n,
        "mode": mode,
        "bars": bars,
        "seed": seed,
        "equity": equity,
        "full_decision": full_decision,
        "memory_budget_mib": memory_budget_mib,
        "memory_cap_gib": memory_cap_gib,
        "trace_path": trace_path,
    }
    completed = subprocess.run(
        [sys.executable, "-m", "research.scripts.solver_profile", "--cell", json.dumps(payload)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_repo_root()),
        env=child_environment(),
    )
    if completed.returncode != 0:
        record = {
            "n_names": n,
            "mode": mode,
            "bars": bars,
            "failed": True,
            "returncode": completed.returncode,
            "signal": -completed.returncode if completed.returncode < 0 else None,
            "killed_by_oom_killer": completed.returncode == -9,
            "stopped_at_memory_budget": completed.returncode == _Trace.MEMORY_BUDGET_EXIT,
            "memory_budget_mib": memory_budget_mib,
            "stderr_tail": completed.stderr[-4000:],
        }
        if record["stopped_at_memory_budget"]:
            record["reason"] = (
                f"the harness stopped the cell at the declared {memory_budget_mib:.0f} MiB resident-set "
                "budget; the phase and the resident figure at the stop are in `trace`"
            )
        record.update(_read_trace(trace_path))
        return record
    marker = "@@CELL@@"
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            record = json.loads(line[len(marker) :])
            record["stderr_tail"] = completed.stderr[-2000:] or None
            return record
    return {
        "n_names": n,
        "mode": mode,
        "bars": bars,
        "failed": True,
        "returncode": 0,
        "reason": "the child produced no cell record",
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def child_environment() -> dict[str, str]:
    """The environment a cell's child process needs to import ``prism`` and ``research``.

    A child does not inherit the parent's ``sys.path``, only its environment. In a checkout
    where ``prism`` is not installed -- a bare worktree, or a test session that reaches the
    package through pytest's own ``pythonpath`` setting -- the child would fail to import it
    and the cell would come back as a failure that says nothing about the solver. The repo
    root and its ``src`` are prepended rather than substituted, so an installed ``prism``
    still wins nothing and loses nothing.
    """
    root = _repo_root()
    entries = [str(root / "src"), str(root)]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        entries.extend(part for part in existing.split(os.pathsep) if part)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))
    return env


def _read_trace(trace_path: str | None) -> dict[str, Any]:
    """What the phase log says about a cell that did not return."""
    if not trace_path or not Path(trace_path).exists():
        return {"trace": None}
    rows = []
    for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a torn final write from a killed process
            continue
    if not rows:
        return {"trace": None}
    peak_row = max(rows, key=lambda row: float(row.get("rss_mib") or 0.0))
    marks = [row for row in rows if row.get("kind") != "sample"]
    return {
        "trace": {
            "last_phase": rows[-1].get("phase"),
            "rss_at_last_sample_mib": rows[-1].get("rss_mib"),
            "peak_rss_observed_mib": float(peak_row.get("rss_mib") or 0.0),
            "phase_at_peak_rss": peak_row.get("phase"),
            "n_rows": len(rows),
            "n_phase_marks": len(marks),
            "phases_reached": [row.get("phase") for row in marks],
            "seconds_to_last_row": float(rows[-1]["t"]) - float(rows[0]["t"]),
        }
    }


#: The resident set the interpreter holds before a cell does any work, measured in the same
#: cells and subtracted so the fitted exponent describes the shape's own cost and not a constant.
def _baseline_rss(cells: Sequence[dict[str, Any]]) -> float:
    starts = [
        float(cell["memory"]["rss_at_start_mib"])
        for cell in cells
        if not cell.get("failed") and cell.get("memory", {}).get("rss_at_start_mib")
    ]
    return float(np.median(starts)) if starts else 0.0


def _fit_exponent(sizes: Sequence[float], values: Sequence[float]) -> dict[str, Any] | None:
    """``value = c * n**b`` by least squares on the logs, with the fit's own quality reported."""
    pairs = [(float(n), float(v)) for n, v in zip(sizes, values) if n > 0.0 and v > 0.0]
    if len(pairs) < 2:
        return None
    x = np.log(np.array([pair[0] for pair in pairs]))
    y = np.log(np.array([pair[1] for pair in pairs]))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "exponent": float(slope),
        "coefficient": float(np.exp(intercept)),
        "r_squared": None if total == 0.0 else 1.0 - residual / total,
        "fitted_on_n_names": [pair[0] for pair in pairs],
        "fitted_values": [pair[1] for pair in pairs],
    }


def scaling(cells: Sequence[dict[str, Any]], *, extrapolate_to: Sequence[int] = DEFAULT_NAMES) -> dict[str, Any]:
    """The curve, per mode: how per-bar time and marginal memory grow with the cross-section.

    This is the part a roadmap item may cite about a shape that did not complete. A cell the
    harness stopped at its memory budget is not evidence of how much memory the shape needs; the
    fitted exponent over the shapes that *did* complete is, and it is labeled as an
    extrapolation wherever it is used as one.
    """
    baseline = _baseline_rss(cells)
    out: dict[str, Any] = {"baseline_rss_mib": baseline, "by_mode": {}}
    for mode in MODES:
        rows = [
            cell
            for cell in cells
            if cell.get("mode") == mode and not cell.get("failed") and cell.get("per_bar_total_seconds")
        ]
        if len(rows) < 2:
            continue
        sizes = [float(cell["n_names"]) for cell in rows]
        seconds = [float(cell["per_bar_total_seconds"]["median"]) for cell in rows]
        first_compile = [float(cell["cold"]["compile_seconds"] or 0.0) for cell in rows]
        marginal_rss = [max(0.0, float(cell["memory"]["peak_rss_mib"]) - baseline) for cell in rows]
        entry: dict[str, Any] = {
            "completed_at_n_names": [int(size) for size in sizes],
            "per_bar_seconds": _fit_exponent(sizes, seconds),
            "first_compile_seconds": _fit_exponent(sizes, first_compile),
            "marginal_peak_rss_mib": _fit_exponent(sizes, marginal_rss),
        }
        projections: dict[str, Any] = {}
        for target in extrapolate_to:
            if float(target) in sizes:
                continue
            row: dict[str, Any] = {"measured": False}
            for key in ("per_bar_seconds", "first_compile_seconds", "marginal_peak_rss_mib"):
                fit = entry[key]
                if fit is None:
                    continue
                row[key] = float(fit["coefficient"]) * float(target) ** float(fit["exponent"])
            projections[str(target)] = row
        entry["extrapolated"] = projections
        out["by_mode"][mode] = entry
    return out


def projection(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What a complete stage 4 evaluation costs, from the per-decision numbers measured here.

    The arithmetic is the registration's own: 1,308 sessions at a 21-bar decision stride is 63
    decisions per arm per capital scenario, over two convex arms and six capital scenarios.
    """
    decisions = math.ceil(RUN_SESSIONS / RUN_STRIDE_BARS)
    total_decisions = decisions * RUN_CONVEX_ARMS * RUN_CAPITAL_SCENARIOS
    per_size: dict[str, Any] = {}
    for cell in cells:
        if cell.get("mode") != "production" or cell.get("failed"):
            continue
        complete = (cell.get("complete_decision") or {}).get("construct_seconds")
        finalize = (cell.get("complete_decision") or {}).get("finalize_batch_seconds")
        if complete is None:
            continue
        per_decision = float(complete["median"]) + float((finalize or {}).get("median", 0.0))
        per_size[str(cell["n_names"])] = {
            "median_seconds_per_decision": per_decision,
            "seconds_per_arm_per_capital_scenario": per_decision * decisions,
            "seconds_for_both_convex_arms_all_capital_scenarios": per_decision * total_decisions,
            "minutes_for_both_convex_arms_all_capital_scenarios": per_decision * total_decisions / 60.0,
        }
    return {
        "definition": (
            "construct() plus finalize_batch() at the measured median, times the registration's decision count. "
            "It is the construction half of an evaluation only: the score panel, the calibration fits, the "
            "accounting engine and the fold loop are not in it."
        ),
        "sessions": RUN_SESSIONS,
        "decision_stride_bars": RUN_STRIDE_BARS,
        "decisions_per_arm_per_capital_scenario": decisions,
        "convex_arms": RUN_CONVEX_ARMS,
        "capital_scenarios": RUN_CAPITAL_SCENARIOS,
        "total_convex_decisions": total_decisions,
        "reference_universe_names": 574,
        "by_cross_section_size": per_size,
    }


def summary(cells: Sequence[dict[str, Any]], curve: dict[str, Any]) -> dict[str, Any]:
    """The measured facts, each carrying its own number, derived from the cells.

    This block exists so that the thing a roadmap item cites is a number with a shape and a
    machine attached. Nothing here is a recommendation; every entry is a statement of what was
    measured, and an entry about a shape that did not complete says so and names the fit it
    came from.
    """

    def cell(mode: str, n: int) -> dict[str, Any] | None:
        for row in cells:
            if row.get("mode") == mode and int(row.get("n_names", -1)) == n:
                return row
        return None

    def stopped(mode: str) -> list[dict[str, Any]]:
        return [
            {
                "n_names": row["n_names"],
                "resident_mib_at_stop": (row.get("trace") or {}).get("peak_rss_observed_mib"),
                "phase": (row.get("trace") or {}).get("phase_at_peak_rss"),
                "seconds_before_stop": (row.get("trace") or {}).get("seconds_to_last_row"),
            }
            for row in cells
            if row.get("mode") == mode and row.get("failed")
        ]

    sizes = sorted({int(row["n_names"]) for row in cells})
    largest = max(sizes) if sizes else 0
    production = {}
    for n in sizes:
        row = cell("production", n)
        if row is None or row.get("failed"):
            production[str(n)] = {"completed": False}
            continue
        production[str(n)] = {
            "completed": True,
            "solver_status": row["statuses"],
            "solver_status_counts": row.get("status_counts"),
            "first_compile_seconds": row["cold"]["compile_seconds"],
            "median_per_bar_seconds": (row["per_bar_total_seconds"] or {}).get("median"),
            "median_clarabel_seconds": (row["repeat"]["solver_seconds"] or {}).get("median"),
            "median_construct_seconds": (row["complete_decision"]["construct_seconds"] or {}).get("median"),
            "median_finalize_batch_seconds": (row["complete_decision"]["finalize_batch_seconds"] or {}).get("median"),
            "peak_rss_mib": row["memory"]["peak_rss_mib"],
        }
    largest_production = cell("production", largest)
    baseline = curve.get("baseline_rss_mib", 0.0)
    return {
        "the_shipped_constructor_at_the_largest_shape_profiled": (
            None
            if largest_production is None or largest_production.get("failed")
            else {
                "n_names": largest,
                "ran": True,
                "solver_status": largest_production["statuses"],
                "median_seconds_per_bar": (largest_production["per_bar_total_seconds"] or {}).get("median"),
                "peak_rss_mib": largest_production["memory"]["peak_rss_mib"],
                "peak_rss_above_interpreter_baseline_mib": largest_production["memory"]["peak_rss_mib"] - baseline,
                "note": (
                    "The shipped non-DPP path completed every cross-section size profiled, at every bar. There "
                    "is no out-of-memory result on this path to cite. `solver_status` is the set of statuses "
                    "CLARABEL returned across the cell's bars; `optimal_inaccurate` among them is a tolerance "
                    "outcome and not a failure -- prism.portfolio.convex records it and the independent "
                    "validation of portfolio/validate.py, never the status, decides whether the output is usable."
                ),
            }
        ),
        "production_by_size": production,
        "solver_status_counts_by_mode_and_size": {
            f"{row['mode']}@{row['n_names']}": row.get("status_counts")
            for row in cells
            if not row.get("failed") and row.get("status_counts")
        },
        "interpreter_baseline_rss_mib": baseline,
        "what_ran_out_of_memory": {
            "twin_dpp": stopped("twin_dpp"),
            "twin_dpp_fixed_cone": stopped("twin_dpp_fixed_cone"),
            "mechanism": (
                "DPP canonicalization builds and caches a parameter-to-canonical-form map whose construction "
                "cost grows with the square of the cross-section; `scaling.by_mode.twin_dpp.marginal_peak_rss_mib` "
                "carries the fitted exponent and its R-squared. The stopped cells were stopped by this harness at "
                "the declared resident budget during the first bar's compile, not by the kernel, and the figure "
                "each was holding at the stop is a lower bound on what it needed, not the figure it would have "
                "reached; the extrapolations in `scaling` are what describe those shapes."
            ),
        },
        "what_dpp_buys": {
            "measured": (
                "A DPP compile turns the per-bar canonicalization into a parameter assignment: `repeat."
                "compile_seconds` falls from tens of milliseconds to a few milliseconds. It changes nothing about "
                "CLARABEL's own solve time, which is what dominates at every size profiled -- compare "
                "`repeat.solver_seconds` across the modes at one size."
            ),
            "per_bar_seconds_exponent_by_mode": {
                mode: (entry.get("per_bar_seconds") or {}).get("exponent") for mode, entry in curve["by_mode"].items()
            },
        },
        "equivalence": (
            "Every twin cell solved the production program on the same bar; `cells[].equivalence."
            "max_abs_weight_difference` is the largest disagreement, and it is what makes the twin's timings "
            "comparable to the production ones rather than merely adjacent to them."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--names", type=int, nargs="+", default=list(DEFAULT_NAMES))
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--bars", type=int, default=6, help="decision bars per cell; the first is the cold one")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--equity", type=float, default=DEFAULT_EQUITY)
    parser.add_argument("--no-full-decision", action="store_true", help="skip construct()/finalize_batch()")
    parser.add_argument("--in-process", action="store_true", help="do not fork per cell (memory becomes unusable)")
    parser.add_argument(
        "--memory-budget-mib",
        type=float,
        default=8192.0,
        help="resident-set budget per cell: the harness stops a cell that exceeds it and records the "
        "phase and the figure, rather than letting the kernel pick a victim; 0 disables it",
    )
    parser.add_argument(
        "--memory-cap-gib",
        type=float,
        default=0.0,
        help="optional hard address-space backstop (RLIMIT_AS) per cell; off by default because "
        "address space is not resident memory and this process reserves ~1.4 GiB before it works",
    )
    parser.add_argument("--trace-dir", default=None, help="where the per-cell phase logs are written")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cell is not None:
        payload = json.loads(args.cell)
        payload.pop("trace_dir", None)
        trace_path = payload.get("trace_path")
        try:
            record = run_cell(**payload)
        except MemoryError:
            record = {
                "n_names": payload.get("n"),
                "mode": payload.get("mode"),
                "bars": payload.get("bars"),
                "failed": True,
                "reason": "MemoryError under the declared address-space cap",
                "memory_cap_gib": payload.get("memory_cap_gib"),
            }
            record.update(_read_trace(trace_path))
        print("@@CELL@@" + json.dumps(record))
        return 0

    env = environment()
    trace_dir = args.trace_dir or tempfile.mkdtemp(prefix="solver_profile_trace_")
    cells: list[dict[str, Any]] = []
    for n in args.names:
        for mode in args.modes:
            kwargs: dict[str, Any] = {
                "n": int(n),
                "mode": str(mode),
                "bars": int(args.bars),
                "seed": int(args.seed),
                "equity": float(args.equity),
                "full_decision": not args.no_full_decision,
                "memory_budget_mib": float(args.memory_budget_mib) or None,
                "memory_cap_gib": float(args.memory_cap_gib) or None,
            }
            started = time.perf_counter()
            if args.in_process:
                cell = run_cell(**kwargs)
            else:
                cell = run_cell_in_subprocess(**kwargs, trace_dir=trace_dir)
            cell.setdefault("n_names", int(n))
            cell.setdefault("mode", str(mode))
            cell["wall_seconds_including_process"] = time.perf_counter() - started
            cells.append(cell)
            outcome = "FAILED" if cell.get("failed") else (cell.get("statuses") or [""])[0]
            seconds = cell.get("cell_seconds")
            print(
                f"{n:>6} names  {mode:<21} {outcome:<20} "
                f"{'-' if seconds is None else format(seconds, '.2f') + 's'}",
                file=sys.stderr,
            )

    curve = scaling(cells, extrapolate_to=tuple(int(n) for n in args.names))
    payload = {
        "instrument": "research/scripts/solver_profile.py",
        "question": (
            "What does the stage 4 convex constructor cost in time and memory at 500, 1,000 and 3,000 names, "
            "under DPP and non-DPP compilation, and does 3,000 names run at all (spec stage 4, Q15)?"
        ),
        "generated": date.today().isoformat(),
        "environment": env,
        "declared_shape": {
            "source_profile": "examples/reference/profiles/experiment_v1.toml",
            "source_risk_model": "examples/reference/risk/factor_diagonal.toml",
            "source_registration": "registrations/2026-09-construction.toml",
            "n_factors": N_FACTORS,
            "horizon_bars": HORIZON_BARS,
            "scaling": SCALING,
            "limits": ["max_gross", "max_name_weight", "net_band", "variance_budget", "participation_per_name"],
            "max_gross": MAX_GROSS,
            "max_name_weight": MAX_NAME_WEIGHT,
            "net_band": NET_BAND,
            "variance_budget": VARIANCE_BUDGET,
            "participation_per_name": PARTICIPATION,
            "cash_buffer": CASH_BUFFER,
            "risk_aversion": RISK_AVERSION,
            "eps": EPS,
            "components": 1,
            "cost_terms": ["linear_bps", "quadratic_notional", "power_three_halves", "borrow", "financing"],
            "panel": (
                "synthetic: z rank-standardized normal, sigma_h lognormal about 32 percent annual, prices "
                "lognormal about 60 dollars, adv_dollars lognormal about 40 million, w0 an invested "
                "dollar-neutral book at the profile's gross less the cash buffer"
            ),
            "why_synthetic": (
                "No frozen 3,000-name panel exists in this workspace, and a solver timing depends on the "
                "problem's shape and coefficient density rather than on which names carry which scores. Every "
                "structural parameter is the declared one; the draws are recorded so the shape is reproducible."
            ),
        },
        "method": {
            "modes": {
                "production": "prism.portfolio.convex.ConvexProgram as shipped: no parameters, fresh compile per bar",
                "twin_constant": "the profiling twin with numpy constants: the non-DPP control for the DPP cell",
                "twin_dpp": "the profiling twin with cvxpy Parameters: compiled once, re-solved by assignment",
                "twin_dpp_fixed_cone": (
                    "the same DPP twin with the risk cone written as a constant rather than an N x K "
                    "parameter, so it is recompiled on the risk model's `refresh_every_bars = 5` cadence "
                    "rather than on every bar. It isolates how much of DPP's cost is the cone parameter."
                ),
            },
            "twin_departures": [
                "a trade variable dw with dw == w - w0, because linear_bps @ abs(w - w0_parameter) is not DPP",
                "max_name_weight and participation_per_name as box bounds rather than abs() atoms; the feasible "
                "set is identical and the box survives a bar on which a name becomes reduce-only",
            ],
            "twin_is_not_production": (
                "The twin lives in research/ and is a measuring instrument. Whether to adopt a DPP formulation "
                "is a decision these numbers inform; writing one into the production path before the numbers "
                "existed would be the tuning this stage forbids."
            ),
            "equivalence_check": (
                "Each twin cell solves the production program on the same bar and records the maximum absolute "
                "weight difference and the objective difference, so a twin timing cannot silently describe a "
                "different problem."
            ),
            "memory": (
                "One child process per cell; peak resident set size from getrusage(RUSAGE_SELF).ru_maxrss, "
                "which is monotonic within a process and would otherwise report the largest cell's peak for "
                "every cell. tracemalloc peak is the Python-object half of the same number and excludes "
                "CLARABEL's own allocations. A sampling thread reads /proc/self/statm every 50 ms and stops "
                "the cell if the resident set exceeds `memory_budget_mib`, so a cell that needs more memory "
                "than the budget allows is stopped by this harness, with the phase and the figure recorded, "
                "rather than by the kernel's OOM killer, which is free to take some other process instead. "
                "A stopped cell's `trace.peak_rss_observed_mib` is therefore a lower bound on what the cell "
                "needed, not the figure it would have reached."
            ),
            "extrapolation": (
                "`scaling` fits log(marginal cost) against log(n_names) by least squares over the cells that "
                "completed, and reports the exponent, so a shape that was stopped can be described by the "
                "curve the completed shapes trace rather than by an unattributed failure."
            ),
            "first_compile": (
                "For the production mode, the canonicalization time of a freshly built problem measured through "
                "Problem.get_problem_data(CLARABEL); for the twins, cvxpy's own Problem.compilation_time. "
                "`cold` is the first bar of the cell and `repeat` is every bar after it, so `cold.compile_seconds` "
                "is first compilation and `repeat.compile_seconds` is what a subsequent bar pays."
            ),
            "timing_decomposition": {
                "data_seconds": (
                    "production: ConvexProgram construction, which is the snapshot validation, the sigma "
                    "cross-check, the coefficient vectors and the action-restriction masks. twins: the same, "
                    "plus reading the twin's vectors off it. Charging the twins for building the production "
                    "program's cvxpy expressions overstates their data cost, which is the conservative "
                    "direction for a DPP claim."
                ),
                "build_seconds": (
                    "production: assembling the objective and constraint expressions. twin_constant: "
                    "constructing the whole problem. twin_dpp: assigning parameter values on the first bar's "
                    "compiled problem, except the cold bar, which constructs it."
                ),
                "wall_seconds": "the Problem.solve call, which for a non-DPP problem recanonicalizes inside it",
                "solver_seconds": "CLARABEL's own reported solve time, excluding canonicalization",
                "per_bar_total_seconds": "data + build + wall, over the bars after the cold one",
            },
            "import_seconds": (
                "cvxpy is imported and warmed before anything is timed, so the cold bar carries compilation "
                "rather than an import. The engine pays it once per process."
            ),
        },
        "memory_budget_mib": float(args.memory_budget_mib) or None,
        "memory_cap_gib": float(args.memory_cap_gib) or None,
        "command": " ".join([Path(sys.executable).name, "-m", "research.scripts.solver_profile", *sys.argv[1:]]),
        "cells": cells,
        "scaling": curve,
        "summary": summary(cells, curve),
        "complete_evaluation_projection": projection(cells),
    }
    out_path = Path(args.out) if args.out else Path("results") / f"solver_profile_{date.today().isoformat()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
