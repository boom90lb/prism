"""The cost model: what a decision is charged, and under which declared spread mode.

Spec section 3.6. This module is the single home of the charging arithmetic; the
reproduction accounting engine in :mod:`prism.execution.target_weights` calls into it
rather than carrying its own copy.

**What exists at stage 2.** :func:`charge_weights` only -- the weight-space cost row of the
``historical_constant_weight`` reproduction engine, moved here from
``execution/target_weights.py::_cost_values`` without a change to its arithmetic, and the
three declared spread modes it is evaluated under:

``flat``
    one scalar half-spread for every name, ``ExecutionConfig.spread_bps``. Reached by
    passing ``spread_bps_per_name=None``, which must stay bit-for-bit the pre-bucket
    computation (frozen-v1 parity).
``bucket``
    a per-name vector priced from a ``(dollar-volume floor, one-way bps)`` schedule by
    :func:`bucket_spread_bps`.
``frame``
    a bar-by-name panel, one row per fill bar. The reproduction engine already accepts it
    (``backtest_target_weights(..., spread_bps_per_name=<DataFrame>)``) and hands
    :func:`charge_weights` that bar's row; the recorded trend configuration needs it,
    because it re-buckets from each fold's formation window and holds the vector across
    that fold's test bars.

**What stage 3 adds.** :class:`CostStack` -- the parsed form of a ``costs/*.toml`` file --
and :class:`CostInputs`, the per-name, per-day data it is evaluated against, and with them
the dollar-space methods that charge the canonical ``PortfolioState`` of
``execution/portfolio_state.py``:

``charge_trade``
    one-time cost, in dollars, at fill. Commission and half spread per fill on filled
    notional; the quadratic and participation terms on the **decided order notional**,
    allocated across that order's fills pro rata, so splitting an order does not shrink a
    superlinear charge.
``accrue_holding``
    per-bar cost, in dollars, on the marked state: borrow on short market value as marked,
    financing on negative cash, cash income on positive cash, with ``short_proceeds``
    deciding whether the collateral is excluded from that base.
``convex_terms``
    the same declared policy as the constructor's cost functions, in weight units over
    ``horizon_bars`` -- the object the stage 4 constructor optimizes against and the one
    the cost-consistency test compares with ``charge_trade`` plus ``accrue_holding``.

A ``CostModel`` built from an ``ExecutionConfig`` rather than a ``CostStack`` still refuses
all three, by name and with the reason: four stack keys have no field on
``ExecutionConfig``, and charging zero for a component nobody declared is exactly the
silent-zero defect this module exists to prevent.

Provenance of the two duplicated pieces (production never imports ``research``):

* :func:`charge_weights` is the body of ``src/prism/execution/target_weights.py::_cost_values``
  at commit 29a6c67, moved verbatim.
* :func:`bucket_spread_bps` reproduces ``research/arbitrage/residual_walk_forward.py:77-87``
  by value, including its fail-safe: unknown liquidity takes the widest step, never the
  cheapest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.config import ExecutionConfig

# The pre-registered conservative-upper schedule (docs/r2_design.md section 3, I-9),
# ``(dollar-volume floor, one-way bps)`` descending, ending at the 0.0 catch-all. Imported
# rather than re-declared: ``execution/edge.py`` already holds the production copy and the
# EDGE diagnostic brackets against it. The constant's home moves to the declared
# ``[spread].schedule`` of a cost stack when ``CostStack`` lands.
from prism.execution.edge import SPREAD_BUCKET_SCHEDULE_V1

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    from prism.execution.portfolio_state import Fill, OrderIntent, PortfolioState

__all__ = [
    "COST_TERMS",
    "DOLLAR_VOLUME_USES",
    "REQUIRED_STACK_KEYS",
    "SPREAD_BUCKET_SCHEDULE_V1",
    "ConvexCostTerms",
    "CostBreakdown",
    "CostInputs",
    "CostModel",
    "CostStack",
    "CostStackError",
    "DollarVolumeSpec",
    "ImpactSpec",
    "SpreadSpec",
    "bucket_spread_bps",
    "charge_weights",
]

Schedule = Sequence[Sequence[float]]


def _validated_schedule(schedule: Schedule) -> tuple[tuple[float, float], ...]:
    """Coerce and check a ``(floor, one-way bps)`` schedule.

    Steps descend by floor, the last floor is the 0.0 catch-all, and no step charges a
    credit. A malformed schedule raises rather than silently re-bucketing: the floors
    decide which names are charged what, and getting them wrong is a cost defect that
    reads as alpha.
    """
    steps: list[tuple[float, float]] = []
    for step in schedule:
        pair = tuple(step)
        if len(pair) != 2:
            raise ValueError(f"spread schedule steps must be (floor, one_way_bps) pairs, got {step!r}")
        floor, bps = (float(pair[0]), float(pair[1]))
        if not np.isfinite(floor) or not np.isfinite(bps):
            raise ValueError(f"spread schedule steps must be finite, got {step!r}")
        if bps < 0.0:
            raise ValueError(f"spread schedule charges a credit at floor {floor}: {bps} bps")
        steps.append((floor, bps))
    if not steps:
        raise ValueError("spread schedule is empty; declare at least the 0.0 catch-all step")
    floors = [floor for floor, _ in steps]
    if floors != sorted(floors, reverse=True) or len(set(floors)) != len(floors):
        raise ValueError(f"spread schedule floors must be strictly descending, got {floors}")
    if floors[-1] != 0.0:
        raise ValueError(f"spread schedule must end at the 0.0 catch-all floor, got {floors[-1]}")
    return tuple(steps)


def bucket_spread_bps(
    dollar_volume: pd.Series,
    *,
    schedule: Schedule = SPREAD_BUCKET_SCHEDULE_V1,
) -> pd.Series:
    """Map a per-name liquidity statistic through a spread bucket schedule.

    ``dollar_volume`` is the declared ``dollar_volume.spread_bucket`` statistic (the
    recorded configurations use a median dollar volume over a trailing or formation
    window). Each floor is an *inclusive lower bound* and the steps descend, so a name at
    exactly a floor takes that step.

    Unknown liquidity -- NaN, or any value below the last floor -- takes the widest step.
    That is the fail-safe direction and it is deliberate: a name whose liquidity we cannot
    establish is charged as if it were the least liquid, never as if it were the most.

    A value that is not a number at all (a string, an object) raises rather than being
    coerced to NaN and priced as unknown liquidity: a mis-typed liquidity column is a
    defect in the input, not a statement about a name's liquidity.

    Reproduces ``research/arbitrage/residual_walk_forward.py:77-87`` by value.
    """
    steps = _validated_schedule(schedule)
    raw = pd.Series(dollar_volume)
    values = pd.to_numeric(raw, errors="coerce")
    uncoercible = values.isna() & raw.notna()
    if uncoercible.any():
        offenders = list(raw.index[uncoercible])[:5]
        raise ValueError(
            f"liquidity statistic is not numeric for {offenders}; supply a numeric "
            "dollar-volume series (NaN means unknown liquidity, a string means a defect)"
        )
    mdv = values.to_numpy(dtype=float)
    floors = steps[:-1]
    with np.errstate(invalid="ignore"):
        out = np.select(
            [mdv >= floor for floor, _ in floors],
            [bps for _, bps in floors],
            default=steps[-1][1],
        )
    return pd.Series(out, index=raw.index, dtype=float)


@dataclass(frozen=True)
class SpreadSpec:
    """The declared ``[spread]`` block of a cost stack (spec section 3.6).

    ``mode`` is the one thing that decides where a name-day's half-spread comes from:

    * ``flat`` -- ``flat_bps`` for every name.
    * ``bucket`` -- ``schedule``, applied to the declared liquidity statistic.
    * ``frame`` -- a bar-by-name panel supplied by the caller (``CostInputs.spread_bps``
      once that type exists; the ``spread_bps_per_name`` DataFrame of the reproduction
      engine today). ``schedule`` and ``flat_bps`` may still be declared, as the fallback
      for a bar or name the panel does not cover.

    ``bound`` and ``unknown_liquidity`` are single-valued in the specification; they are
    carried as fields so a stack that declares something else is refused at load rather
    than silently reinterpreted.
    """

    mode: Literal["flat", "bucket", "frame"]
    statistic: str = "dollar_volume.spread_bucket"
    bound: Literal["inclusive_lower"] = "inclusive_lower"
    unknown_liquidity: Literal["widest_step"] = "widest_step"
    flat_bps: float | None = None
    schedule: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("flat", "bucket", "frame"):
            raise ValueError(f"spread mode must be flat, bucket or frame, got {self.mode!r}")
        if self.bound != "inclusive_lower":
            raise ValueError(f"spread bound must be inclusive_lower, got {self.bound!r}")
        if self.unknown_liquidity != "widest_step":
            raise ValueError(f"spread unknown_liquidity must be widest_step, got {self.unknown_liquidity!r}")
        if self.flat_bps is not None:
            flat = float(self.flat_bps)
            if not np.isfinite(flat) or flat < 0.0:
                raise ValueError(f"spread flat_bps must be finite and >= 0, got {self.flat_bps!r}")
            object.__setattr__(self, "flat_bps", flat)
        if self.schedule is not None:
            object.__setattr__(self, "schedule", _validated_schedule(self.schedule))
        if self.mode == "flat":
            if self.flat_bps is None:
                raise ValueError("spread mode 'flat' requires flat_bps")
            if self.schedule is not None:
                raise ValueError("spread mode 'flat' prices no schedule; declare mode 'bucket' instead")
        if self.mode == "bucket" and self.schedule is None:
            raise ValueError("spread mode 'bucket' requires a schedule")

    def per_name_bps(self, dollar_volume: pd.Series) -> pd.Series | None:
        """The per-name half-spread vector this spec prices, or ``None`` under ``flat``.

        ``None`` is the flat path's marker all the way down: :func:`charge_weights` takes
        ``spread_bps_per_name=None`` to mean the scalar ``ExecutionConfig.spread_bps``, and
        that path is bit-for-bit the pre-bucket computation.
        """
        if self.mode == "flat":
            return None
        if self.mode == "frame":
            raise ValueError(
                "spread mode 'frame' prices each bar from the supplied bar-by-name panel; "
                "pass that row to charge_weights instead of asking the spec for a vector"
            )
        assert self.schedule is not None  # guaranteed by __post_init__
        return bucket_spread_bps(dollar_volume, schedule=self.schedule)


def charge_weights(
    trade_w: np.ndarray,
    holding_w: np.ndarray,
    execution: ExecutionConfig,
    *,
    cost_multiplier: float = 1.0,
    dividend_return: float = 0.0,
    dollar_volume: np.ndarray | None = None,
    initial_capital: float = 1.0,
    spread_bps_per_name: np.ndarray | None = None,
) -> dict[str, float]:
    """One bar's charged cost in weight space, for the reproduction accounting engine.

    Parity: the body of ``src/prism/execution/target_weights.py::_cost_values`` at commit
    29a6c67, moved without change. ``trade_w`` is the signed weight change filled this bar
    and ``holding_w`` the weights held over it; both are fractions of current equity.
    Returns the eight-key reproduction cost row, in the recorded order --
    ``commission_spread``, ``impact``, ``borrow``, ``total``, ``turnover``, ``gross``,
    ``net``, ``dividend_return`` -- which is the column order of the ``costs`` frame the
    record carries.

    ``spread_bps_per_name`` selects the spread mode: ``None`` is ``flat`` (the scalar
    ``execution.spread_bps``), a per-name vector is ``bucket``, and one row of a
    bar-by-name panel is ``frame``. The participation term is priced against
    ``initial_capital``, which is what ``dollar_volume.participation.base =
    "initial_capital"`` means in reproduction mode.
    """
    trade_abs = np.abs(trade_w)
    if spread_bps_per_name is None:
        commission_spread = float(
            trade_abs.sum() * (execution.commission_bps + execution.spread_bps) / 10_000.0 * cost_multiplier
        )
    else:
        # Per-bucket spread (r2_design.md §3, I-9): commission stays scalar, the
        # spread is priced per name. None must reproduce the flat computation
        # bit-for-bit (frozen-v1 parity).
        commission_spread = float(
            (trade_abs * (execution.commission_bps + spread_bps_per_name)).sum() / 10_000.0 * cost_multiplier
        )
    impact = float(np.square(trade_abs).sum() * execution.slippage_coeff / 10_000.0 * cost_multiplier)
    if execution.adv_impact_coeff > 0.0 and dollar_volume is not None:
        valid = np.isfinite(dollar_volume)
        adv = np.where(valid, np.maximum(dollar_volume, execution.adv_floor_dollars), np.nan)
        participation = trade_abs * float(initial_capital) / adv
        if execution.adv_impact_model == "sqrt":
            participation = np.sqrt(participation)
        adv_bps = execution.adv_impact_coeff * participation
        impact += float(np.nansum(trade_abs * adv_bps / 10_000.0) * cost_multiplier)
    short_gross = float(np.abs(np.minimum(holding_w, 0.0)).sum())
    borrow = float(short_gross * execution.borrow_rate_bps_annual / 10_000.0 / execution.trading_days_per_year)
    total = commission_spread + impact + borrow
    return {
        "commission_spread": commission_spread,
        "impact": impact,
        "borrow": borrow,
        "total": total,
        "turnover": float(trade_abs.sum()),
        "gross": float(np.abs(holding_w).sum()),
        "net": float(holding_w.sum()),
        "dividend_return": float(dividend_return),
    }


# ---------------------------------------------------------------------------
# the declared cost stack, and the per-name inputs it is evaluated against
# ---------------------------------------------------------------------------

#: Column order of ``CostBreakdown.per_name`` -- the seven charged terms of §3.6.
COST_TERMS: tuple[str, ...] = (
    "commission",
    "spread",
    "impact_quadratic",
    "impact_participation",
    "borrow",
    "financing",
    "cash_income",
)

#: The three dollar-volume statistics a stack declares, each with its own estimator,
#: window, refresh cadence and floor (decision D10). Collapsing them into one series
#: changes charged cost, which is why the loader requires all three by name.
DOLLAR_VOLUME_USES: tuple[str, ...] = ("spread_bucket", "participation", "impact")

#: Top-level keys a declared stack must carry. An undeclared component is never defaulted
#: to zero: a zero that was never declared is indistinguishable in the record from a zero
#: that was (§3.6).
REQUIRED_STACK_KEYS: tuple[str, ...] = (
    "spread",
    "impact",
    "dollar_volume",
    "commission_bps",
    "borrow_bps_annual",
    "cash_income_bps_annual",
    "financing_bps_annual",
    "short_proceeds",
    "holding_cost_settlement",
    "trading_days_per_year",
)

SHORT_PROCEEDS = ("collateral_no_rebate", "proceeds_to_cash_with_rebate")
HOLDING_COST_SETTLEMENTS = ("per_bar", "monthly")
PARTICIPATION_MODELS = ("none", "linear", "sqrt")


class CostStackError(ValueError):
    """A declared cost stack that cannot be charged as written."""


def _content_id(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DollarVolumeSpec:
    """One declared liquidity statistic (§3.6, decision D10)."""

    estimator: Literal["median", "mean", "same_bar"]
    window: Literal["trailing", "formation"]
    window_bars: int
    refresh: Literal["bar", "decision", "fold", "run"]
    floor_dollars: float
    source: Literal["close_times_volume"] = "close_times_volume"
    base: Literal["current_equity", "initial_capital"] = "current_equity"

    def __post_init__(self) -> None:
        if self.estimator not in ("median", "mean", "same_bar"):
            raise CostStackError(f"dollar-volume estimator must be median, mean or same_bar, got {self.estimator!r}")
        if self.window not in ("trailing", "formation"):
            raise CostStackError(f"dollar-volume window must be trailing or formation, got {self.window!r}")
        if self.window == "formation" and self.refresh != "fold":
            raise CostStackError("a formation window is legal only with refresh = 'fold' (§3.6)")
        if self.refresh not in ("bar", "decision", "fold", "run"):
            raise CostStackError(f"dollar-volume refresh must be bar, decision, fold or run, got {self.refresh!r}")
        if int(self.window_bars) < 1:
            raise CostStackError(f"dollar-volume window_bars must be >= 1, got {self.window_bars!r}")
        object.__setattr__(self, "window_bars", int(self.window_bars))
        floor = float(self.floor_dollars)
        if not np.isfinite(floor) or floor < 0.0:
            raise CostStackError(f"dollar-volume floor_dollars must be finite and >= 0, got {self.floor_dollars!r}")
        object.__setattr__(self, "floor_dollars", floor)
        if self.source != "close_times_volume":
            raise CostStackError(f"dollar-volume source must be close_times_volume, got {self.source!r}")
        if self.base not in ("current_equity", "initial_capital"):
            raise CostStackError(f"dollar-volume base must be current_equity or initial_capital, got {self.base!r}")


@dataclass(frozen=True)
class ImpactSpec:
    """The declared ``[impact]`` block: one portfolio-relative term and one ADV-relative term."""

    quadratic_notional_coeff: float
    participation_model: Literal["none", "linear", "sqrt"]
    participation_coeff: float
    statistic: str = "dollar_volume.impact"

    def __post_init__(self) -> None:
        for name in ("quadratic_notional_coeff", "participation_coeff"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise CostStackError(f"impact {name} must be finite and >= 0, got {getattr(self, name)!r}")
            object.__setattr__(self, name, value)
        if self.participation_model not in PARTICIPATION_MODELS:
            raise CostStackError(
                f"impact participation_model must be one of {PARTICIPATION_MODELS}, got {self.participation_model!r}"
            )

    def participation_factor(self, participation: float) -> float:
        """``g(pi)``: identity for ``linear``, ``sqrt`` for ``sqrt``, zero for ``none``."""
        if self.participation_model == "none":
            return 0.0
        value = float(participation)
        if value < 0.0:
            raise CostStackError(f"participation must be >= 0, got {participation!r}")
        return math.sqrt(value) if self.participation_model == "sqrt" else value


@dataclass(frozen=True)
class CostStack:
    """The declared cost policy of a ``costs/*.toml`` file (§3.6).

    Everything charged anywhere in the canonical engine comes from here. Two rules the
    loader enforces rather than the caller remembering:

    * every key of :data:`REQUIRED_STACK_KEYS` is declared, because an undeclared
      component defaulted to zero is indistinguishable in the record from a declared zero;
    * ``cash_income_bps_annual <= financing_bps_annual``, because the financing/income pair
      is piecewise linear in ``L(w) - 1`` with slopes ``f`` above and ``c`` below, and is
      convex only when ``c <= f``. Without the check a declared rate pair makes the DCP
      check fail inside the solver with no statement of why.
    """

    spread: SpreadSpec
    impact: ImpactSpec
    dollar_volume: dict[str, DollarVolumeSpec]
    commission_bps: float
    borrow_bps_annual: float
    cash_income_bps_annual: float
    financing_bps_annual: float
    short_proceeds: Literal["collateral_no_rebate", "proceeds_to_cash_with_rebate"] = "collateral_no_rebate"
    holding_cost_settlement: Literal["per_bar", "monthly"] = "per_bar"
    trading_days_per_year: int = 252
    sources: dict[str, dict] = dataclasses.field(default_factory=dict)
    stack_id: str = ""

    def __post_init__(self) -> None:
        for name in ("commission_bps", "borrow_bps_annual", "cash_income_bps_annual", "financing_bps_annual"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise CostStackError(f"{name} must be finite and >= 0, got {getattr(self, name)!r}")
            object.__setattr__(self, name, value)
        if self.short_proceeds not in SHORT_PROCEEDS:
            raise CostStackError(f"short_proceeds must be one of {SHORT_PROCEEDS}, got {self.short_proceeds!r}")
        if self.holding_cost_settlement not in HOLDING_COST_SETTLEMENTS:
            raise CostStackError(
                f"holding_cost_settlement must be one of {HOLDING_COST_SETTLEMENTS}, "
                f"got {self.holding_cost_settlement!r}"
            )
        days = int(self.trading_days_per_year)
        if days <= 0:
            raise CostStackError(f"trading_days_per_year must be > 0, got {self.trading_days_per_year!r}")
        object.__setattr__(self, "trading_days_per_year", days)
        missing = [use for use in DOLLAR_VOLUME_USES if use not in self.dollar_volume]
        if missing:
            raise CostStackError(
                f"the cost stack declares no dollar-volume statistic for {missing}; the three uses are "
                "declared separately on purpose (decision D10) and none is defaulted from another"
            )
        for use, spec in self.dollar_volume.items():
            if not isinstance(spec, DollarVolumeSpec):
                raise CostStackError(f"dollar_volume[{use!r}] must be a DollarVolumeSpec, got {type(spec).__name__}")
        if self.dollar_volume["impact"].floor_dollars <= 0.0:
            raise CostStackError(
                "dollar_volume.impact.floor_dollars must be > 0, got "
                f"{self.dollar_volume['impact'].floor_dollars!r}: the impact term divides the traded notional "
                "by max(dollar_volume, floor_dollars), so a zero floor divides by zero on a name whose bar "
                "printed a price and no volume and charges an infinite impact cost. The floor is the whole "
                "defence, and 'unknown liquidity is never cheap' is the rule it implements (§3.6). The other "
                "two uses may declare 0.0: neither divides by the statistic."
            )
        if self.cash_income_bps_annual > self.financing_bps_annual:
            raise CostStackError(
                f"cash_income_bps_annual ({self.cash_income_bps_annual}) exceeds financing_bps_annual "
                f"({self.financing_bps_annual}): the financing/income pair is convex only when income <= "
                "financing, and a stack that inverts them makes the constructor's DCP check fail with no "
                "statement of why (§3.6)"
            )
        if not self.stack_id:
            object.__setattr__(self, "stack_id", _content_id(self.declaration()))

    # -- declared rates, per bar -------------------------------------------------

    @property
    def borrow_per_bar(self) -> float:
        """``borrow_bps_annual / 1e4 / trading_days_per_year`` -- charged on short market value."""
        return self.borrow_bps_annual / 1e4 / self.trading_days_per_year

    @property
    def financing_per_bar(self) -> float:
        return self.financing_bps_annual / 1e4 / self.trading_days_per_year

    @property
    def cash_income_per_bar(self) -> float:
        return self.cash_income_bps_annual / 1e4 / self.trading_days_per_year

    def declaration(self) -> dict:
        """The declared content, as the mapping a stack file would carry."""
        return {
            "spread": {
                "mode": self.spread.mode,
                "statistic": self.spread.statistic,
                "bound": self.spread.bound,
                "unknown_liquidity": self.spread.unknown_liquidity,
                "flat_bps": self.spread.flat_bps,
                "schedule": None if self.spread.schedule is None else [list(s) for s in self.spread.schedule],
            },
            "impact": dataclasses.asdict(self.impact),
            "dollar_volume": {use: dataclasses.asdict(spec) for use, spec in sorted(self.dollar_volume.items())},
            "commission_bps": self.commission_bps,
            "borrow_bps_annual": self.borrow_bps_annual,
            "cash_income_bps_annual": self.cash_income_bps_annual,
            "financing_bps_annual": self.financing_bps_annual,
            "short_proceeds": self.short_proceeds,
            "holding_cost_settlement": self.holding_cost_settlement,
            "trading_days_per_year": self.trading_days_per_year,
            "sources": self.sources,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "CostStack":
        """Parse a declared stack, refusing one that omits any required key (§3.6)."""
        missing = [key for key in REQUIRED_STACK_KEYS if key not in mapping]
        if missing:
            raise CostStackError(
                f"the cost stack omits {missing}; every component is declared explicitly, because a zero that "
                "was never declared is indistinguishable in the record from a zero that was (§3.6)"
            )
        spread_block = dict(mapping["spread"])  # type: ignore[arg-type]
        schedule = spread_block.get("schedule")
        spread = SpreadSpec(
            mode=spread_block["mode"],
            statistic=spread_block.get("statistic", "dollar_volume.spread_bucket"),
            bound=spread_block.get("bound", "inclusive_lower"),
            unknown_liquidity=spread_block.get("unknown_liquidity", "widest_step"),
            flat_bps=spread_block.get("flat_bps"),
            schedule=None if schedule is None else tuple(tuple(float(x) for x in step) for step in schedule),
        )
        impact = ImpactSpec(**dict(mapping["impact"]))  # type: ignore[arg-type]
        dollar_volume = {
            str(use): DollarVolumeSpec(**dict(spec))
            for use, spec in dict(mapping["dollar_volume"]).items()  # type: ignore[arg-type]
        }
        return cls(
            spread=spread,
            impact=impact,
            dollar_volume=dollar_volume,
            commission_bps=float(mapping["commission_bps"]),  # type: ignore[arg-type]
            borrow_bps_annual=float(mapping["borrow_bps_annual"]),  # type: ignore[arg-type]
            cash_income_bps_annual=float(mapping["cash_income_bps_annual"]),  # type: ignore[arg-type]
            financing_bps_annual=float(mapping["financing_bps_annual"]),  # type: ignore[arg-type]
            short_proceeds=str(mapping["short_proceeds"]),  # type: ignore[arg-type]
            holding_cost_settlement=str(mapping["holding_cost_settlement"]),  # type: ignore[arg-type]
            trading_days_per_year=int(mapping["trading_days_per_year"]),  # type: ignore[arg-type]
            sources=dict(mapping.get("sources", {})),  # type: ignore[arg-type]
        )

    def to_execution_config(self) -> ExecutionConfig:
        """The one adapter to the reproduction engine's parameter object (§3.6).

        A rename, not a reinterpretation. Four stack keys have no field on
        ``ExecutionConfig`` -- ``cash_income_bps_annual``, ``financing_bps_annual``,
        ``short_proceeds`` and ``holding_cost_settlement`` -- and they are dropped here,
        which is exactly why ``ExecutionConfig`` is not extended: they belong to the
        canonical engine and the reproduction engine has no term for them.
        """
        return ExecutionConfig(
            spread_bps=0.0 if self.spread.flat_bps is None else float(self.spread.flat_bps),
            slippage_coeff=self.impact.quadratic_notional_coeff,
            commission_bps=self.commission_bps,
            borrow_rate_bps_annual=self.borrow_bps_annual,
            trading_days_per_year=self.trading_days_per_year,
            adv_impact_coeff=self.impact.participation_coeff,
            adv_impact_model=self.impact.participation_model,
            adv_floor_dollars=self.dollar_volume["impact"].floor_dollars,
        )


@dataclass(frozen=True)
class CostInputs:
    """The per-name, per-day data a stack is evaluated against (§3.6).

    A separate object from the stack so that a refresh is a recorded event rather than an
    implicit rebind: the evaluator may rebuild these at a fold boundary and the record
    carries ``cost_inputs_id`` and ``cost_refresh = scope``.
    """

    as_of: str
    scope: Literal["run", "fold", "bar"] = "run"
    spread_bps: float | pd.Series | pd.DataFrame | None = None
    adv_bucket: pd.Series | None = None
    adv_participation: pd.Series | pd.DataFrame | None = None
    adv_impact: pd.Series | pd.DataFrame | None = None
    inputs_id: str = ""

    def __post_init__(self) -> None:
        if self.scope not in ("run", "fold", "bar"):
            raise CostStackError(f"cost inputs scope must be run, fold or bar, got {self.scope!r}")
        if not self.inputs_id:
            object.__setattr__(self, "inputs_id", _content_id(self._digest_payload()))

    def _digest_payload(self) -> dict:
        def digest(value: object) -> object:
            if value is None:
                return None
            if isinstance(value, (pd.Series, pd.DataFrame)):
                hashed = pd.util.hash_pandas_object(value.fillna(-1.0), index=True)
                return hashlib.sha256(np.ascontiguousarray(hashed.to_numpy())).hexdigest()
            return float(value)  # type: ignore[arg-type]

        return {
            "as_of": self.as_of,
            "scope": self.scope,
            "spread_bps": digest(self.spread_bps),
            "adv_bucket": digest(self.adv_bucket),
            "adv_participation": digest(self.adv_participation),
            "adv_impact": digest(self.adv_impact),
        }


def _lookup(source: object, sid: str, bar: str | None) -> float:
    """One value out of a scalar, a per-sid series, or a bar-by-sid frame.

    Returns NaN when the source says nothing about ``(sid, bar)``. The *caller* decides
    what silence means -- the widest spread step, or a refusal -- because the two answers
    differ per term and neither is ever a zero charge.
    """
    if source is None:
        return float("nan")
    if isinstance(source, pd.DataFrame):
        if bar is None:
            raise CostStackError("a bar-by-sid cost input needs the bar it is priced on")
        key = str(bar)[:10]
        if key not in source.index.astype(str) or sid not in source.columns:
            return float("nan")
        row = source.loc[source.index.astype(str) == key]
        if row.empty:
            return float("nan")
        value = row.iloc[0][sid]
    elif isinstance(source, pd.Series):
        if sid not in source.index:
            return float("nan")
        value = source.loc[sid]
    else:
        value = source
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


@dataclass(frozen=True)
class CostBreakdown:
    """Charged dollars, split into the seven declared terms (§3.6).

    ``cash_income`` is a signed credit and enters ``total`` with a minus sign, so ``total``
    is always the number debited from the book.
    """

    commission: float = 0.0
    spread: float = 0.0
    impact_quadratic: float = 0.0
    impact_participation: float = 0.0
    borrow: float = 0.0
    financing: float = 0.0
    cash_income: float = 0.0
    total: float = 0.0
    per_name: pd.DataFrame = dataclasses.field(default_factory=lambda: _empty_per_name())

    @classmethod
    def from_terms(cls, per_name: pd.DataFrame) -> "CostBreakdown":
        """Build a breakdown from a per-name term frame, summing the scalars from it."""
        frame = per_name.reindex(columns=list(COST_TERMS)).fillna(0.0)
        sums = {term: float(frame[term].sum()) for term in COST_TERMS}
        total = (
            sums["commission"]
            + sums["spread"]
            + sums["impact_quadratic"]
            + sums["impact_participation"]
            + sums["borrow"]
            + sums["financing"]
            - sums["cash_income"]
        )
        return cls(total=total, per_name=frame, **sums)

    @property
    def trade_total(self) -> float:
        """The one-time terms only, charged at fill."""
        return self.commission + self.spread + self.impact_quadratic + self.impact_participation

    @property
    def holding_total(self) -> float:
        """The per-bar terms only, accrued on the marked state."""
        return self.borrow + self.financing - self.cash_income

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        if not isinstance(other, CostBreakdown):  # pragma: no cover - defensive
            return NotImplemented
        combined = (
            pd.concat([self.per_name, other.per_name])
            .groupby(level=0)
            .sum()
            .reindex(columns=list(COST_TERMS))
            .fillna(0.0)
        )
        combined.index.name = "sid"
        return CostBreakdown.from_terms(combined)

    def to_reproduction_row(
        self, *, turnover: float, gross: float, net: float, dividend_return: float
    ) -> dict[str, float]:
        """The eight ``charge_weights`` keys, for records written in reproduction mode.

        Refuses to fold a financing or cash-income charge into the row: the reproduction
        engine has no such term, its recorded ``total`` is asserted to equal
        ``commission_spread + impact + borrow`` (``tests/test_ledger_conservation.py:43``),
        and hiding a fourth term inside one of the three would break that identity or
        misattribute the charge.
        """
        if self.financing or self.cash_income:
            raise CostStackError(
                "a reproduction cost row carries no financing or cash-income term; this breakdown charges "
                f"financing {self.financing} and income {self.cash_income}, which the reproduction record "
                "shape cannot represent (§3.6)"
            )
        return {
            "commission_spread": self.commission + self.spread,
            "impact": self.impact_quadratic + self.impact_participation,
            "borrow": self.borrow,
            "total": self.total,
            "turnover": float(turnover),
            "gross": float(gross),
            "net": float(net),
            "dividend_return": float(dividend_return),
        }


def _empty_per_name(sids: Sequence[str] = ()) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=pd.Index(list(sids), name="sid", dtype=object), columns=list(COST_TERMS))
    return frame


@dataclass(frozen=True)
class ConvexCostTerms:
    """The constructor's cost functions, in weight units over the horizon (§3.6, Q8).

    ``holding_costs(w) = b*h*sum_i max(-w_i, 0) + f*h*max(L(w)-1, 0) - c*h*max(1-L(w), 0)``
    under ``collateral_no_rebate``, with ``L(w)`` the gross long weight. Under
    ``proceeds_to_cash_with_rebate`` the financing and income base is
    ``1 - L(w) + sum_i max(-w_i, 0)`` instead, and :attr:`financing_base` says which.
    """

    sids: pd.Index
    linear_bps: np.ndarray
    quadratic_coeff: np.ndarray
    power_three_halves_coeff: np.ndarray
    borrow_per_horizon: np.ndarray
    financing_per_horizon: float
    cash_income_per_horizon: float
    equity: float
    horizon_bars: int
    financing_base: Literal["collateral_no_rebate", "proceeds_to_cash_with_rebate"] = "collateral_no_rebate"

    def trade_costs(self, trade_w: np.ndarray) -> float:
        """The one-time cost fraction of equity this trade vector budgets."""
        trade_abs = np.abs(np.asarray(trade_w, dtype=float))
        linear = float((self.linear_bps * trade_abs).sum()) / 1e4
        quadratic = float((self.quadratic_coeff * np.square(trade_abs)).sum()) / 1e4
        power = float((self.power_three_halves_coeff * np.power(trade_abs, 1.5)).sum()) / 1e4
        return linear + quadratic + power

    def holding_costs(self, holding_w: np.ndarray) -> float:
        """The holding cost fraction of equity over ``horizon_bars`` at these weights."""
        weights = np.asarray(holding_w, dtype=float)
        shorts = float(np.maximum(-weights, 0.0).sum())
        longs = float(np.maximum(weights, 0.0).sum())
        cash_fraction = 1.0 - longs
        if self.financing_base == "proceeds_to_cash_with_rebate":
            cash_fraction = 1.0 - longs + shorts
        borrow = float((self.borrow_per_horizon * np.maximum(-weights, 0.0)).sum())
        financing = self.financing_per_horizon * max(-cash_fraction, 0.0)
        income = self.cash_income_per_horizon * max(cash_fraction, 0.0)
        return borrow + financing - income

    def total(self, trade_w: np.ndarray, holding_w: np.ndarray) -> float:
        """The constructor's whole cost expression, as a fraction of equity."""
        return self.trade_costs(trade_w) + self.holding_costs(holding_w)

    def total_dollars(self, trade_w: np.ndarray, holding_w: np.ndarray) -> float:
        """The same expression in dollars, which is what ``charge_trade`` plus
        ``accrue_holding`` summed over the horizon is compared against (§6)."""
        return self.total(trade_w, holding_w) * self.equity


class CostModel:
    """The declared cost policy applied to one decision (§3.6).

    Two construction forms, and they are not interchangeable:

    * ``CostModel(stack, inputs, equity=..., horizon_bars=...)`` -- the canonical engine.
      ``charge_trade`` charges one-time cost in dollars at fill, ``accrue_holding``
      accrues per-bar cost in dollars on the marked state, and ``convex_terms`` states the
      same policy as the constructor's cost functions.
    * ``CostModel(execution_config)`` -- the ``historical_constant_weight`` reproduction
      engine, which has only :meth:`charge_weights`. The dollar-space methods refuse on
      such a model, by name: four of the stack's keys have no field on ``ExecutionConfig``
      (``cash_income_bps_annual``, ``financing_bps_annual``, ``short_proceeds``,
      ``holding_cost_settlement``), so charging dollars from one would either invent them
      or silently charge zero for components nobody declared.
    """

    def __init__(
        self,
        stack: CostStack | ExecutionConfig,
        inputs: CostInputs | None = None,
        *,
        spread: SpreadSpec | None = None,
        equity: float | None = None,
        horizon_bars: int | None = None,
    ) -> None:
        if isinstance(stack, CostStack):
            self.stack: CostStack | None = stack
            self.execution = stack.to_execution_config()
            self.spread = spread or stack.spread
        else:
            self.stack = None
            self.execution = stack
            self.spread = spread
        self.inputs = inputs
        self.equity = None if equity is None else float(equity)
        self.horizon_bars = None if horizon_bars is None else int(horizon_bars)
        if self.equity is not None and (not np.isfinite(self.equity) or self.equity <= 0.0):
            raise CostStackError(f"equity must be finite and > 0, got {equity!r}")
        if self.horizon_bars is not None and self.horizon_bars < 1:
            raise CostStackError(f"horizon_bars must be >= 1, got {horizon_bars!r}")

    def at_equity(self, equity: float, *, horizon_bars: int | None = None) -> "CostModel":
        """The same declared policy, bound to a different marked equity.

        The quadratic impact term is portfolio relative (``n^2 / E``), so the equity it
        divides by is the marked equity **at the decision close** the order was sized from.
        A model is therefore re-bound per decision rather than carrying one equity for a
        whole run, which would price every later order against a stale book.
        """
        return CostModel(
            self.stack if self.stack is not None else self.execution,
            self.inputs,
            spread=self.spread,
            equity=equity,
            horizon_bars=self.horizon_bars if horizon_bars is None else horizon_bars,
        )

    # ------------------------------------------------------------------ reproduction

    def spread_bps_per_name(self, dollar_volume: pd.Series) -> pd.Series | None:
        """The per-name half-spread vector the declared mode prices, or ``None`` for flat."""
        if self.spread is None:
            return None
        return self.spread.per_name_bps(dollar_volume)

    def charge_weights(
        self,
        trade_w: np.ndarray,
        holding_w: np.ndarray,
        *,
        cost_multiplier: float = 1.0,
        dividend_return: float = 0.0,
        dollar_volume: np.ndarray | None = None,
        initial_capital: float = 1.0,
        spread_bps_per_name: np.ndarray | None = None,
    ) -> dict[str, float]:
        """One bar's charged cost in weight space; see :func:`charge_weights`."""
        return charge_weights(
            trade_w,
            holding_w,
            self.execution,
            cost_multiplier=cost_multiplier,
            dividend_return=dividend_return,
            dollar_volume=dollar_volume,
            initial_capital=initial_capital,
            spread_bps_per_name=spread_bps_per_name,
        )

    # ------------------------------------------------------------------ canonical

    def _require_stack(self, method: str) -> CostStack:
        if self.stack is None:
            raise CostStackError(
                f"CostModel.{method} charges dollars against a declared CostStack; this model was built from an "
                "ExecutionConfig, which carries no cash_income_bps_annual, financing_bps_annual, short_proceeds "
                "or holding_cost_settlement. Build the model from the declared stack (§3.6); charging zero for "
                "components nobody declared is the silent-zero defect this refusal prevents"
            )
        return self.stack

    def _require_equity(self, method: str) -> float:
        if self.equity is None:
            raise CostStackError(
                f"CostModel.{method} needs the marked equity at the decision close: the quadratic impact term is "
                "portfolio relative (n^2 / E), not ADV relative (§3.6)"
            )
        return self.equity

    def half_spread_bps(self, sid: str, bar: str | None = None) -> float:
        """The one-way half spread charged on ``sid`` at ``bar``, in bps.

        The fallback chain is the declared one and never reaches zero: under ``flat`` a
        missing entry takes ``flat_bps``; under ``bucket`` and ``frame`` it takes the
        widest schedule step, which is the fail-safe direction (§3.6). A stack whose mode
        leaves no fallback declared raises rather than charging nothing.
        """
        stack = self._require_stack("half_spread_bps")
        spec = stack.spread
        if spec.mode == "frame":
            value = _lookup(None if self.inputs is None else self.inputs.spread_bps, sid, bar)
            if np.isfinite(value) and value >= 0.0:
                return float(value)
            return self._spread_fallback(spec, sid)
        if spec.mode == "flat":
            assert spec.flat_bps is not None  # guaranteed by SpreadSpec
            return float(spec.flat_bps)
        adv = _lookup(None if self.inputs is None else self.inputs.adv_bucket, sid, bar)
        assert spec.schedule is not None  # guaranteed by SpreadSpec
        priced = bucket_spread_bps(pd.Series({sid: adv}), schedule=spec.schedule)
        return float(priced.loc[sid])

    @staticmethod
    def _spread_fallback(spec: SpreadSpec, sid: str) -> float:
        if spec.schedule is not None:
            return float(spec.schedule[-1][1])
        if spec.flat_bps is not None:
            return float(spec.flat_bps)
        raise CostStackError(
            f"{sid}: the spread frame prices no half spread for this fill and the stack declares neither a "
            "schedule nor flat_bps to fall back to. A fill is never charged a zero spread by omission (§3.6)"
        )

    def _adv_impact(self, sid: str, bar: str | None) -> float:
        stack = self._require_stack("charge_trade")
        floor = stack.dollar_volume["impact"].floor_dollars
        raw = _lookup(None if self.inputs is None else self.inputs.adv_impact, sid, bar)
        if not np.isfinite(raw) or raw <= 0.0:
            if stack.impact.participation_coeff == 0.0:
                # The term is inert under this stack, so the missing statistic changes no
                # charged dollar. Refusing here would refuse a fill the declared policy
                # prices at zero for reasons that have nothing to do with liquidity.
                return float(floor) if floor > 0.0 else 1.0
            raise CostStackError(
                f"{sid}: the participation impact term is priced against dollar_volume.impact, which is "
                f"{raw!r} for this fill. The participation bound refuses a trade on a name with no liquidity "
                "statistic, so a fill on one is an accounting defect, not a free trade (§3.6)"
            )
        return max(float(raw), float(floor))

    def charge_trade(
        self,
        fills: Sequence["Fill"],
        *,
        decided_notional: Mapping[str, float] | None = None,
    ) -> CostBreakdown:
        """One-time cost in dollars, charged at fill (§3.6).

        The linear terms -- commission and half spread -- are charged per fill on filled
        notional at the fill price. The two superlinear terms are charged on the **decided
        order notional** and allocated across that order's fills pro rata by filled
        quantity, so splitting one decided order across ``k`` fills charges what the
        constructor budgeted rather than ``1/k`` of the quadratic term and ``k^-0.5`` of
        the participation term. That is why the method takes the fills of one
        ``OrderRecord`` at a time and why ``decided_notional`` exists: without it a
        partially filled order is charged as though the smaller order had been decided,
        which rewards order splitting.
        """
        stack = self._require_stack("charge_trade")
        equity = self._require_equity("charge_trade")
        rows: dict[str, dict[str, float]] = {}

        groups: dict[str, list["Fill"]] = {}
        for fill in fills:
            groups.setdefault(fill.client_order_id, []).append(fill)

        for client_order_id, group in groups.items():
            sid = group[0].sid
            reference = group[0].reference_price
            if any(f.sid != sid for f in group):
                raise CostStackError(f"{client_order_id}: one order's fills must all be on one sid")
            if any(abs(f.reference_price - reference) > 1e-12 for f in group):
                raise CostStackError(f"{client_order_id}: one order's fills must share one reference price")
            filled_reference_notional = sum(abs(f.qty) * reference for f in group)
            if decided_notional is not None and client_order_id in decided_notional:
                decided = abs(float(decided_notional[client_order_id]))
                if filled_reference_notional > decided * (1.0 + 1e-9):
                    raise CostStackError(
                        f"{client_order_id}: filled notional {filled_reference_notional} exceeds the decided "
                        f"notional {decided}; an over-fill is an accounting defect, not a cost input"
                    )
            elif len(group) > 1:
                raise CostStackError(
                    f"{client_order_id}: {len(group)} fills against one order need the decided order notional, "
                    "because the quadratic and participation terms are charged on the decided order and "
                    "allocated pro rata. Pass decided_notional; inferring it from the fills would charge the "
                    "split order less than the constructor budgeted (§3.6)"
                )
            else:
                decided = filled_reference_notional
            if decided <= 0.0:
                raise CostStackError(f"{client_order_id}: decided notional must be > 0, got {decided}")

            quadratic_total = stack.impact.quadratic_notional_coeff / 1e4 * decided * decided / equity
            participation_total = 0.0
            if stack.impact.participation_coeff > 0.0 and stack.impact.participation_model != "none":
                adv = self._adv_impact(sid, group[0].bar)
                participation = decided / adv
                participation_total = (
                    stack.impact.participation_coeff / 1e4 * decided * stack.impact.participation_factor(participation)
                )

            for fill in group:
                share = abs(fill.qty) * reference / decided
                filled_notional = abs(fill.notional)
                row = rows.setdefault(sid, {term: 0.0 for term in COST_TERMS})
                if fill.source == "broker":
                    # The venue's price already carries spread and impact; only the actual
                    # commission is debited (§3.6, "no double charge in operation").
                    row["commission"] += fill.commission
                    continue
                row["commission"] += stack.commission_bps / 1e4 * filled_notional
                row["spread"] += self.half_spread_bps(sid, fill.bar) / 1e4 * filled_notional
                row["impact_quadratic"] += quadratic_total * share
                row["impact_participation"] += participation_total * share

        frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(COST_TERMS)) if rows else _empty_per_name()
        frame.index.name = "sid"
        return CostBreakdown.from_terms(frame.sort_index())

    def accrue_holding(
        self,
        state: "PortfolioState",
        prices: pd.Series | Mapping[str, float] | None = None,
    ) -> CostBreakdown:
        """One bar's holding cost in dollars on the marked state (§3.6).

        Borrow is charged on short market value **as marked**, not on entry notional.
        Financing and cash income are charged on ``cash - short_collateral`` under
        ``collateral_no_rebate`` and on ``cash`` under ``proceeds_to_cash_with_rebate``:
        that difference is the entire observable content of the policy name, which is why
        it is a stack key and not a comment.
        """
        stack = self._require_stack("accrue_holding")
        rows: dict[str, dict[str, float]] = {}
        borrow_rate = stack.borrow_per_bar
        for sid in state.sids():
            position = state.positions[sid]
            if position.shares >= 0.0:
                continue
            short_value = -position.shares * position.mark(prices, sid)
            row = rows.setdefault(sid, {term: 0.0 for term in COST_TERMS})
            row["borrow"] += borrow_rate * short_value

        collateral = state.short_collateral(prices)
        base = state.cash if stack.short_proceeds == "proceeds_to_cash_with_rebate" else state.cash - collateral
        financing = stack.financing_per_bar * max(-base, 0.0)
        cash_income = stack.cash_income_per_bar * max(base, 0.0)

        frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(COST_TERMS)) if rows else _empty_per_name()
        frame.index.name = "sid"
        breakdown = CostBreakdown.from_terms(frame.sort_index())
        return CostBreakdown(
            commission=breakdown.commission,
            spread=breakdown.spread,
            impact_quadratic=breakdown.impact_quadratic,
            impact_participation=breakdown.impact_participation,
            borrow=breakdown.borrow,
            financing=financing,
            cash_income=cash_income,
            total=breakdown.total + financing - cash_income,
            per_name=breakdown.per_name,
        )

    def expected_for_intents(
        self,
        intents: Sequence["OrderIntent"],
        reference_prices: pd.Series | Mapping[str, float] | None = None,
    ) -> CostBreakdown:
        """The modeled one-time cost of a decided batch. Never charged (§3.6).

        Reconciliation compares this with the observed fill-versus-reference cost from
        ``prism.execution.spread.arrival_slippage_bps``, which is a measurement. Because a
        simulated fill prices at the raw bar price and its modeled cost is a separate cash
        line, that comparison is not circular.
        """
        from prism.execution.portfolio_state import Fill  # local: portfolio_state imports nothing from here

        pseudo: list[Fill] = []
        decided: dict[str, float] = {}
        for position, intent in enumerate(intents):
            price = _lookup(reference_prices, intent.sid, None)
            reference = float(price) if np.isfinite(price) and price > 0.0 else intent.reference_price
            client_order_id = f"expected:{intent.decision_bar}:{intent.sid}:{position}"
            decided[client_order_id] = abs(intent.qty * reference)
            pseudo.append(
                Fill(
                    execution_id=client_order_id,
                    client_order_id=client_order_id,
                    sid=intent.sid,
                    ts=intent.decision_bar,
                    qty=intent.qty,
                    fill_price=reference,
                    reference_price=reference,
                    source="simulated",
                )
            )
        return self.charge_trade(pseudo, decided_notional=decided)

    def convex_terms(
        self,
        sids: Sequence[str] | pd.Index | None = None,
        *,
        adv_dollars: pd.Series | None = None,
    ) -> ConvexCostTerms:
        """The constructor's cost functions, in weight units over the horizon (§3.6, Q8).

        Everything is expressed as a fraction of current marked equity: the one-time cost
        fraction is trade dollars over equity, and the holding cost fraction is the per-bar
        rate times ``horizon_bars`` times the relevant weight. The participation term maps
        to ``|dw|^1.5`` under ``participation_model = "sqrt"`` and into the quadratic
        coefficient under ``"linear"``, because trade dollars are ``|dw| * E``.
        """
        stack = self._require_stack("convex_terms")
        equity = self._require_equity("convex_terms")
        if self.horizon_bars is None:
            raise CostStackError("CostModel.convex_terms needs horizon_bars: holding costs accrue per bar (§3.6)")
        horizon = self.horizon_bars
        if sids is None:
            if self.inputs is None or self.inputs.adv_impact is None:
                raise CostStackError("convex_terms needs the constructor universe: pass sids")
            source = self.inputs.adv_impact
            sids = source.columns if isinstance(source, pd.DataFrame) else source.index
        index = pd.Index([str(s) for s in sids], name="sid")

        linear = np.array([stack.commission_bps + self.half_spread_bps(sid) for sid in index], dtype=float)
        quadratic = np.full(len(index), stack.impact.quadratic_notional_coeff, dtype=float)
        power = np.zeros(len(index), dtype=float)
        if stack.impact.participation_coeff > 0.0 and stack.impact.participation_model != "none":
            floor = stack.dollar_volume["impact"].floor_dollars
            for position, sid in enumerate(index):
                raw = _lookup(adv_dollars, sid, None)
                if not np.isfinite(raw) or raw <= 0.0:
                    raw = self._adv_impact(sid, None)
                adv = max(float(raw), float(floor))
                if stack.impact.participation_model == "sqrt":
                    power[position] = stack.impact.participation_coeff * math.sqrt(equity / adv)
                else:
                    quadratic[position] += stack.impact.participation_coeff * equity / adv
        return ConvexCostTerms(
            sids=index,
            linear_bps=linear,
            quadratic_coeff=quadratic,
            power_three_halves_coeff=power,
            borrow_per_horizon=np.full(len(index), stack.borrow_per_bar * horizon, dtype=float),
            financing_per_horizon=stack.financing_per_bar * horizon,
            cash_income_per_horizon=stack.cash_income_per_bar * horizon,
            equity=equity,
            horizon_bars=horizon,
            financing_base=stack.short_proceeds,
        )
