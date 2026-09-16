"""Batch rounding, repair, scaling, and the termination argument (spec §3.7).

The rule is stated rather than left to the phrase "at the increment", because parity
depends on it: **the trade delta is rounded, not the target**, to the per-sid increment,
with ties to even -- which is what ``float(round(delta))`` does today at
``src/prism/live/loop.py:142`` through :func:`prism.assembly.size_orders`. The
zero-crossing clamp and the dust rules move here from ``live/loop.py::targets_to_orders``
(Parity for operation; a declared option for evaluation, keyed by ``[rounding]
zero_crossing_clamp`` and ``apply_dust_rule``).

**Why a repair loop needs a proof.** Scaling a batch down and re-rounding to whole shares
can re-violate an increment or a minimum notional, and a naive "shrink until it fits" loop
oscillates: validity is *not* monotone in the scale factor. Run (§3.7, reproduced in
``tests/test_rounding.py``): a $10,000 book with four names at prices 320, 45, 12 and 180
and targets +0.24, -0.26, +0.25 and -0.23 has net exposure +0.0106, -0.0004, +0.0054,
-0.0068, -0.0235 and +0.0143 at scales 1.00, 0.90, 0.80, 0.70, 0.60 and 0.50. Against a
0.01 net band the batch is invalid at 1.00, valid from 0.90 through 0.70, and invalid
again at 0.60 and 0.50.

Two decreasing measures, and no iteration bound anywhere:

1. **Repair.** ``V(w) = sum over holding-state limits of v(w)``, plus the obligation
   shortfall in dollars over equity. Every ``v`` in the §3.7 table is non-negative, so
   ``V >= 0``. A single-increment move is accepted **only** if it reduces ``V`` by at
   least ``repair_min_decrement``, so at most ``ceil(V_0 / repair_min_decrement)`` moves
   are ever accepted; ``repair_max_moves`` caps the search independently. ``V`` is a
   real-valued measure decreasing by a fixed positive quantum: the loop terminates.
2. **Scaling.** A finite declared schedule, walked left to right, never revisited -- not a
   search for the largest valid scale, because validity is not monotone in the scale. At
   one scale a dust drop is a *removal*, so the batch is revalidated at that same scale
   after the drop and the schedule never advances for it. The measure is the lexicographic
   pair ``(remaining schedule entries, number of intents)``, which strictly decreases on
   every iteration of either loop and is bounded below.

Whole-batch scaling and whole-batch dropping are the only two repairs. **Dropping one side
of a balanced trade set is not a permitted repair**, because the surviving orders can
leave the portfolio outside its net, beta, cash, or gross limits; obligation legs are
excluded from scaling, because scaling them down would contradict R0's maximal progress.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Mapping, Sequence

import pandas as pd

from prism.portfolio import limits as limits_mod
from prism.portfolio.limits import Limit, LimitContext, is_holding_state, violation
from prism.portfolio.obligations import Executable, Obligation, ObligationPlan, plan_progress
from prism.portfolio.validate import (
    ValidationReport,
    ValidationSettings,
    capability_row,
    quantity_increment_for,
    validate_batch,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RoundingError",
    "RoundingSettings",
    "ProposedOrder",
    "QuantityRecord",
    "RepairMove",
    "RoundingOutput",
    "round_delta",
    "round_batch",
]


class RoundingError(ValueError):
    """A rounding input or declaration that must not be silently repaired."""


_OFF_INCREMENT_POLICIES = ("leave_residual", "close_fractional")
_ROUNDING_FAILURE_RESPONSES = ("scale_batch", "drop_batch")


# --------------------------------------------------------------------------
# Declared settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundingSettings:
    """The declared ``[rounding]`` block plus the profile keys sizing reads (§3.7, §3.9)."""

    quantity_increment_default: float = 1.0
    min_notional: float = 1.0
    default_order_type: str = "market_on_open"
    tie_rule: str = "half_to_even"
    zero_crossing_clamp: bool = True
    apply_dust_rule: bool = True
    fractional_increment: float = 0.001
    off_increment_holding: str = "leave_residual"
    scale_schedule: tuple[float, ...] = (0.75, 0.5, 0.25)
    repair_max_moves: int = 200
    repair_min_decrement: float = 1.0e-6
    min_implemented_gross_fraction: float = 0.80

    def __post_init__(self) -> None:
        if self.tie_rule != "half_to_even":
            raise RoundingError(
                f"tie_rule {self.tie_rule!r} is not implemented; the declared and only supported rule is "
                "'half_to_even', which is what float(round(delta)) does on the live path today"
            )
        if self.off_increment_holding not in _OFF_INCREMENT_POLICIES:
            raise RoundingError(
                f"off_increment_holding must be one of {_OFF_INCREMENT_POLICIES}, got {self.off_increment_holding!r}"
            )
        schedule = tuple(float(entry) for entry in self.scale_schedule)
        for entry in schedule:
            if not math.isfinite(entry) or not 0.0 < entry <= 1.0:
                raise RoundingError(f"scale_schedule entries are finite and in (0, 1], got {entry!r}")
        object.__setattr__(self, "scale_schedule", schedule)
        if int(self.repair_max_moves) < 0:
            raise RoundingError(f"repair_max_moves must be >= 0, got {self.repair_max_moves!r}")
        object.__setattr__(self, "repair_max_moves", int(self.repair_max_moves))
        decrement = float(self.repair_min_decrement)
        if not math.isfinite(decrement) or decrement <= 0.0:
            raise RoundingError(
                f"repair_min_decrement must be finite and > 0, got {decrement!r}: it is the quantum that "
                "makes the repair loop terminate, and zero would permit an unbounded sequence of moves"
            )
        object.__setattr__(self, "repair_min_decrement", decrement)

    @classmethod
    def from_profile(cls, execution_profile: Any) -> "RoundingSettings":
        """Read the ``[rounding]`` block and the sizing keys off an execution profile.

        Accepts the parsed ``ExecutionProfile`` of §3.9 or the plain mapping ``tomllib``
        produces. Unknown keys in the block are refused: a mis-spelled ``scale_schedule``
        that silently kept the default is a repair the operator did not declare.
        """
        block = dict(_block(execution_profile, "rounding") or {})
        known = {
            "tie_rule",
            "zero_crossing_clamp",
            "apply_dust_rule",
            "fractional_increment",
            "off_increment_holding",
            "scale_schedule",
            "repair_max_moves",
            "repair_min_decrement",
            "min_implemented_gross_fraction",
        }
        unknown = set(block) - known
        if unknown:
            raise RoundingError(f"unknown keys in the [rounding] block: {sorted(unknown)}")
        return cls(
            quantity_increment_default=float(_setting(execution_profile, "quantity_increment_default", 1.0)),
            min_notional=float(_setting(execution_profile, "min_notional", 1.0)),
            default_order_type=str(_setting(execution_profile, "default_order_type", "market_on_open")),
            tie_rule=str(block.get("tie_rule", "half_to_even")),
            zero_crossing_clamp=bool(block.get("zero_crossing_clamp", True)),
            apply_dust_rule=bool(block.get("apply_dust_rule", True)),
            fractional_increment=float(block.get("fractional_increment", 0.001)),
            off_increment_holding=str(block.get("off_increment_holding", "leave_residual")),
            scale_schedule=tuple(block.get("scale_schedule", (0.75, 0.5, 0.25))),
            repair_max_moves=int(block.get("repair_max_moves", 200)),
            repair_min_decrement=float(block.get("repair_min_decrement", 1.0e-6)),
            min_implemented_gross_fraction=float(block.get("min_implemented_gross_fraction", 0.80)),
        )

    def for_reproduction(self) -> "RoundingSettings":
        """The ``historical_constant_weight`` engine's forced values (§3.7).

        That engine reproduced ``backtest_target_weights``, which clamped nothing, dropped
        no dust, and held continuous quantities. Reproduction is its only use, so the
        values are forced rather than declared, and the resolved values are recorded.
        """
        return replace(
            self,
            zero_crossing_clamp=False,
            apply_dust_rule=False,
            quantity_increment_default=0.0,
            min_notional=0.0,
        )


def _block(profile: Any, name: str) -> Mapping[str, Any] | None:
    if profile is None:
        return None
    value = profile.get(name) if isinstance(profile, Mapping) else getattr(profile, name, None)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return {key: getattr(value, key) for key in getattr(value, "__dataclass_fields__", {})}


def _setting(profile: Any, name: str, default: Any) -> Any:
    if profile is None:
        return default
    if isinstance(profile, Mapping):
        return profile.get(name, default)
    return getattr(profile, name, default)


# --------------------------------------------------------------------------
# The intent type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposedOrder:
    """The fallback ``OrderIntent`` shape, field for field as §3.6 declares it.

    ``OrderIntent`` is declared once, in ``execution/portfolio_state.py``. Until that
    module exists this module emits this structurally identical record, and
    :func:`_intent_type` prefers the canonical class the moment it is importable, so a
    batch built here is the same object the rest of the engine consumes either way.
    """

    sid: str
    qty: float
    increment: float
    reference_price: float
    decision_bar: str
    order_type: str = "market_on_open"
    time_in_force: str = "opg"
    action_class: str = "C3"


_TIME_IN_FORCE_FOR_TYPE = {"market_on_open": "opg", "market_day": "day"}


def _intent_type(*, continuous_increment: bool = False) -> type:
    """The canonical ``OrderIntent`` when §3.6's module exists, else :class:`ProposedOrder`.

    Imported inside the function, never at module scope: ``portfolio`` must stay
    importable before ``execution/portfolio_state.py`` lands, and the import-hygiene probe
    imports every module under ``prism``.

    ``continuous_increment`` is the one case the canonical type cannot express: it refuses
    an increment of zero, and rightly so, because no venue accepts an order without a
    quantity grid. The continuous increment exists only for the
    ``historical_constant_weight`` reproduction engine, which submits nothing, so that
    batch is emitted as :class:`ProposedOrder` rather than given a fabricated grid.
    """
    if continuous_increment:
        return ProposedOrder
    try:
        from prism.execution.portfolio_state import OrderIntent  # type: ignore[attr-defined]
    except ImportError:
        return ProposedOrder
    return OrderIntent


# --------------------------------------------------------------------------
# Output records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantityRecord:
    """Per sid: the target quantity, the increment, the rounded quantity, the residual."""

    sid: str
    target_qty: float
    increment: float
    held_qty: float
    rounded_delta: float
    residual: float
    reference_price: float
    clamped: bool = False
    off_increment_holding: bool = False
    dropped_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The record shape."""
        return {
            "sid": self.sid,
            "target_qty": self.target_qty,
            "increment": self.increment,
            "held_qty": self.held_qty,
            "rounded_qty": self.rounded_delta,
            "residual": self.residual,
            "reference_price": self.reference_price,
            "clamped": self.clamped,
            "off_increment_holding": self.off_increment_holding,
            "dropped_by": self.dropped_by,
        }


@dataclass(frozen=True)
class RepairMove:
    """One accepted single-increment repair move, with ``V`` before and after."""

    sid: str
    increment_steps: int
    v_before: float
    v_after: float

    def to_dict(self) -> dict[str, Any]:
        """The record shape."""
        return {
            "sid": self.sid,
            "increment_steps": self.increment_steps,
            "v_before": self.v_before,
            "v_after": self.v_after,
        }


@dataclass(frozen=True)
class RoundingOutput:
    """The result of :func:`round_batch` (spec §3.7)."""

    intents: tuple[Any, ...] = ()
    quantities: dict[str, dict[str, Any]] = field(default_factory=dict)
    repair_moves: tuple[RepairMove, ...] = ()
    scale_applied: float = 1.0
    response: Literal["none", "scale_batch", "drop_batch"] = "none"
    binding_condition: str | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    dropped: tuple[dict[str, Any], ...] = ()
    obligations: dict[str, Any] = field(default_factory=dict)
    implemented_gross_fraction: float = 1.0
    not_implementable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """The JSON-safe record."""
        return {
            "intents": [
                {
                    "sid": intent.sid,
                    "qty": intent.qty,
                    "increment": intent.increment,
                    "reference_price": intent.reference_price,
                    "decision_bar": intent.decision_bar,
                    "order_type": intent.order_type,
                    "time_in_force": intent.time_in_force,
                    "action_class": intent.action_class,
                }
                for intent in self.intents
            ],
            "quantities": self.quantities,
            "repair_moves": [move.to_dict() for move in self.repair_moves],
            "scale_applied": self.scale_applied,
            "response": self.response,
            "binding_condition": self.binding_condition,
            "validation": self.validation,
            "dropped": list(self.dropped),
            "obligations": self.obligations,
            "implemented_gross_fraction": self.implemented_gross_fraction,
            "not_implementable": self.not_implementable,
        }


# --------------------------------------------------------------------------
# The rounding rule
# --------------------------------------------------------------------------


def round_delta(delta: float, increment: float) -> float:
    """Round a signed share delta to ``increment`` with ties to even.

    ``increment <= 0`` is the continuous case and returns the delta unchanged, which is
    what the reproduction engine needs. For ``increment = 1`` this is exactly
    ``float(round(delta))``, the live path's rule, ties included.
    """
    if not math.isfinite(delta):
        raise RoundingError(f"a share delta must be finite, got {delta!r}")
    if increment <= 0.0:
        return float(delta)
    steps = delta / increment
    # Snap a step count that is a tie only because of float division (0.1 arithmetic)
    # back onto the tie, so ties-to-even is decided on the intended value.
    nearest_half = round(steps * 2.0) / 2.0
    if abs(steps - nearest_half) < 1e-9:
        steps = nearest_half
    return float(round(steps)) * increment


# --------------------------------------------------------------------------
# State access
# --------------------------------------------------------------------------


def _read_positions(state: Any) -> dict[str, float]:
    positions = state.get("positions") if isinstance(state, Mapping) else getattr(state, "positions", None)
    if positions is None:
        return {}
    out: dict[str, float] = {}
    for sid, value in dict(positions).items():
        shares = getattr(value, "shares", value)
        out[str(sid)] = float(shares)
    return out


def _read_cash(state: Any) -> float:
    cash = state.get("cash", 0.0) if isinstance(state, Mapping) else getattr(state, "cash", 0.0)
    return float(cash)


def _read_committed_cash(state: Any) -> float:
    """Cash already promised to pending buy orders (§3.6 ``committed_cash``)."""
    if not isinstance(state, Mapping):
        method = getattr(state, "committed_cash", None)
        if callable(method):
            return float(method())
    pending = state.get("pending_orders") if isinstance(state, Mapping) else getattr(state, "pending_orders", None)
    if not pending:
        return 0.0
    total = 0.0
    for order in pending:
        intent = getattr(order, "intent", order)
        qty = float(getattr(intent, "qty", getattr(order, "qty", 0.0)))
        filled = float(getattr(order, "filled_qty", 0.0))
        unfilled = qty - math.copysign(min(abs(filled), abs(qty)), qty) if qty else 0.0
        if unfilled > 0.0:
            price = float(getattr(intent, "reference_price", getattr(order, "reference_price", 0.0)))
            total += unfilled * price
    return total


def _marked_equity(state: Any, positions: Mapping[str, float], prices: pd.Series) -> float:
    equity_fn = None if isinstance(state, Mapping) else getattr(state, "equity", None)
    if callable(equity_fn):
        return float(equity_fn(prices))
    total = _read_cash(state)
    for sid, shares in positions.items():
        if shares == 0.0:
            continue
        price = prices.get(sid, float("nan"))
        if price is None or not math.isfinite(float(price)) or float(price) <= 0.0:
            raise RoundingError(
                f"no usable mark for the held position in {sid!r} ({shares} shares, price {price!r}); a "
                "missing bar must never erase a holding or be marked at zero (SPEC N7)"
            )
        total += shares * float(price)
    return float(total)


# --------------------------------------------------------------------------
# round_batch
# --------------------------------------------------------------------------


def round_batch(
    targets: pd.Series,
    state: Any,
    prices: pd.Series,
    execution_profile: Any,
    limits: Iterable[Limit],
    obligations: Iterable[Obligation] = (),
    capabilities: Any = None,
    *,
    decision_bar: str,
    settings: RoundingSettings | None = None,
    validation: ValidationSettings | None = None,
    adv_dollars: pd.Series | None = None,
    risk: Any = None,
    executable: Mapping[str, Executable] | None = None,
    cost_model: Any = None,
    modeled_cost_dollars: float | None = None,
    achieved_bounds: Mapping[str, float] | None = None,
    on_rounding_failure: str = "scale_batch",
    on_blocked_obligation: str = "escalate",
    order_type: str | None = None,
) -> RoundingOutput:
    """Round a target book into an order batch, repair it, and revalidate the whole batch.

    Inputs include current positions and pending orders, not only targets and prices,
    because the validated object is the portfolio the batch implies together with
    everything already outstanding.

    On failure the whole batch is scaled down together or dropped -- never a silent
    partial batch, and never one side of a balanced trade set.
    """
    if on_rounding_failure not in _ROUNDING_FAILURE_RESPONSES:
        raise RoundingError(
            f"on_rounding_failure is one of {_ROUNDING_FAILURE_RESPONSES}, got {on_rounding_failure!r}"
        )
    limits = tuple(limits)
    obligations = tuple(obligations)
    prices = pd.Series(prices, dtype=float)
    settings = settings if settings is not None else RoundingSettings.from_profile(execution_profile)
    validation = validation if validation is not None else ValidationSettings.from_profile(execution_profile)
    positions = _read_positions(state)
    cash = _read_cash(state)
    committed = _read_committed_cash(state)
    equity = _marked_equity(state, positions, prices)
    if not math.isfinite(equity) or equity <= 0.0:
        raise RoundingError(f"equity must be finite and > 0 to size a batch, got {equity!r}")

    w0 = _marked_weights(positions, prices, equity)
    ctx = LimitContext(w0=w0, equity=equity, adv_dollars=adv_dollars, risk=risk)

    plan = plan_progress(
        obligations,
        positions,
        dict(executable or {}),
        reference_prices=prices,
        on_blocked_obligation=on_blocked_obligation,
    )
    exempt = set(plan.protected_sids)

    sized = _size_all(
        targets,
        positions=positions,
        prices=prices,
        equity=equity,
        settings=settings,
        execution_profile=execution_profile,
        capabilities=capabilities,
        exempt=exempt,
        scale=1.0,
        obligation_qty=plan.qty_by_sid(),
    )
    intended_gross = _intended_gross(sized, equity)

    def _validate(records: Sequence[QuantityRecord]) -> tuple[tuple[Any, ...], ValidationReport]:
        intents = _to_intents(
            records,
            positions=positions,
            decision_bar=decision_bar,
            settings=settings,
            order_type=order_type,
            obligation_sids=exempt,
        )
        report = validate_batch(
            intents,
            limits=limits,
            ctx=ctx,
            prices=prices,
            settings=validation,
            execution_profile=execution_profile,
            capabilities=capabilities,
            cash=cash,
            committed_cash=committed,
            positions=positions,
            cost_model=cost_model,
            modeled_cost_dollars=modeled_cost_dollars,
            achieved_bounds=achieved_bounds,
            exempt_sids=exempt,
            check_zero_crossing=settings.zero_crossing_clamp,
        )
        return intents, report

    live = [record for record in sized if record.dropped_by is None]
    dropped = [record for record in sized if record.dropped_by is not None]
    live, report, intents = _settle_dust(live, dropped, _validate)
    repair_moves: tuple[RepairMove, ...] = ()

    if report.ok:
        return _finish(
            intents,
            sized=sized,
            dropped=dropped,
            report=report,
            plan=plan,
            scale=1.0,
            response="none",
            repair_moves=repair_moves,
            settings=settings,
            equity=equity,
            intended_gross=intended_gross,
        )

    # -- repair: single-increment moves under the action restrictions -------
    live, repair_moves = _repair(
        live,
        ctx=ctx,
        limits=limits,
        plan=plan,
        prices=prices,
        settings=settings,
        positions=positions,
        capabilities=capabilities,
        exempt=exempt,
    )
    if repair_moves:
        live, report, intents = _settle_dust(live, dropped, _validate)
        if report.ok:
            return _finish(
                intents,
                sized=sized,
                dropped=dropped,
                report=report,
                plan=plan,
                scale=1.0,
                response="none",
                repair_moves=repair_moves,
                settings=settings,
                equity=equity,
                intended_gross=intended_gross,
            )

    # -- scaling: a finite declared schedule, walked once, never revisited --
    if on_rounding_failure == "scale_batch":
        for scale in settings.scale_schedule:
            scaled = _size_all(
                targets,
                positions=positions,
                prices=prices,
                equity=equity,
                settings=settings,
                execution_profile=execution_profile,
                capabilities=capabilities,
                exempt=exempt,
                scale=scale,
                obligation_qty=plan.qty_by_sid(),
            )
            scaled_live = [record for record in scaled if record.dropped_by is None]
            scaled_dropped = [record for record in scaled if record.dropped_by is not None]
            scaled_live, scaled_report, scaled_intents = _settle_dust(scaled_live, scaled_dropped, _validate)
            if scaled_report.ok:
                return _finish(
                    scaled_intents,
                    sized=scaled,
                    dropped=scaled_dropped,
                    report=scaled_report,
                    plan=plan,
                    scale=scale,
                    response="scale_batch",
                    repair_moves=repair_moves,
                    settings=settings,
                    equity=equity,
                    intended_gross=intended_gross,
                )

    # -- no entry validated: the whole batch is dropped ---------------------
    return RoundingOutput(
        intents=(),
        quantities={record.sid: record.to_dict() for record in sized},
        repair_moves=repair_moves,
        scale_applied=0.0,
        response="drop_batch",
        binding_condition=report.binding_condition,
        validation=report.to_dict(),
        dropped=tuple(
            {
                "sid": record.sid,
                "intended_notional": abs(record.rounded_delta) * record.reference_price,
                "rule": "drop_batch",
            }
            for record in sized
            if record.dropped_by is None
        )
        + tuple(
            {
                "sid": record.sid,
                "intended_notional": abs(record.target_qty - record.held_qty) * record.reference_price,
                "rule": record.dropped_by or "dust",
            }
            for record in dropped
        ),
        obligations=_obligation_record(plan),
        implemented_gross_fraction=0.0,
        not_implementable=True,
    )


def _settle_dust(
    live: list[QuantityRecord],
    dropped: list[QuantityRecord],
    validate: Any,
) -> tuple[list[QuantityRecord], ValidationReport, tuple[Any, ...]]:
    """Validate, dropping any order the validator rejects as sub-minimum, at this scale.

    A dust drop is a removal, so the batch is revalidated at the *same* scale after the
    drop; it never advances the schedule. The loop terminates because every iteration
    removes at least one intent from a finite set.
    """
    while True:
        intents, report = validate(live)
        if report.ok:
            return live, report, intents
        dust_sids = {
            finding.sid
            for finding in report.fatal_findings
            if finding.check == "min_notional" and finding.sid is not None
        }
        if not dust_sids:
            return live, report, intents
        for record in list(live):
            if record.sid in dust_sids:
                live.remove(record)
                dropped.append(replace(record, dropped_by="min_notional"))


def _finish(
    intents: Sequence[Any],
    *,
    sized: Sequence[QuantityRecord],
    dropped: Sequence[QuantityRecord],
    report: ValidationReport,
    plan: ObligationPlan,
    scale: float,
    response: str,
    repair_moves: Sequence[RepairMove],
    settings: RoundingSettings,
    equity: float,
    intended_gross: float,
) -> RoundingOutput:
    implemented = sum(abs(float(intent.qty)) * float(intent.reference_price) for intent in intents) / equity
    fraction = 1.0 if intended_gross <= 0.0 else min(1.0, implemented / intended_gross)
    return RoundingOutput(
        intents=tuple(intents),
        quantities={record.sid: record.to_dict() for record in sized},
        repair_moves=tuple(repair_moves),
        scale_applied=float(scale),
        response=response,  # type: ignore[arg-type]
        binding_condition=report.binding_condition,
        validation=report.to_dict(),
        dropped=tuple(
            {
                "sid": record.sid,
                "intended_notional": abs(record.target_qty - record.held_qty) * record.reference_price,
                "rule": record.dropped_by or "dust",
            }
            for record in dropped
        ),
        obligations=_obligation_record(plan),
        implemented_gross_fraction=fraction,
        not_implementable=fraction + 1e-12 < settings.min_implemented_gross_fraction,
    )


def _obligation_record(plan: ObligationPlan) -> dict[str, Any]:
    return {
        "legs": [
            {"sid": leg.sid, "qty": leg.qty, "progress": leg.progress, "partial": leg.partial, "reason": leg.reason}
            for leg in plan.legs
        ],
        "blocked": [{**blocked.obligation.to_dict(), "reason": blocked.reason} for blocked in plan.blocked],
        "escalations": list(plan.escalations),
    }


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------


def _marked_weights(positions: Mapping[str, float], prices: pd.Series, equity: float) -> pd.Series:
    if not positions:
        return pd.Series(dtype=float)
    out: dict[str, float] = {}
    for sid, shares in positions.items():
        if shares == 0.0:
            out[sid] = 0.0
            continue
        price = prices.get(sid, float("nan"))
        if price is None or not math.isfinite(float(price)):
            raise RoundingError(f"no usable mark for the held position in {sid!r}; a holding is never marked at zero")
        out[sid] = float(shares) * float(price) / equity
    return pd.Series(out, dtype=float)


def _size_all(
    targets: pd.Series,
    *,
    positions: Mapping[str, float],
    prices: pd.Series,
    equity: float,
    settings: RoundingSettings,
    execution_profile: Any,
    capabilities: Any,
    exempt: set[str],
    scale: float,
    obligation_qty: Mapping[str, float],
) -> list[QuantityRecord]:
    """Size every target into a rounded share delta at ``scale``, dust rules applied.

    Parity with :func:`prism.assembly.size_orders`: a NaN target is "no target" and leaves
    the position alone; a flat target on an unheld name needs no price; a non-flat target
    without a finite positive price raises; the zero-crossing clamp trades only to flat;
    the delta -- not the target -- is rounded; and an order below the minimum notional is
    dropped, loudly when its pre-rounding intent was economically real.

    Obligation legs are exempt from the dust rules and are never scaled: scaling them down
    would contradict R0's maximal progress.
    """
    records: list[QuantityRecord] = []
    dust_dropped: list[tuple[str, float]] = []
    series = pd.Series(targets, dtype=float)
    seen: set[str] = set()
    for raw_sid, weight in series.items():
        sid = str(raw_sid)
        seen.add(sid)
        if isinstance(weight, float) and math.isnan(weight):
            continue
        held = float(positions.get(sid, 0.0))
        if float(weight) == 0.0 and held == 0.0 and sid not in obligation_qty:
            continue
        price = _require_price(prices, sid)
        increment = _increment_for(sid, settings, execution_profile, capabilities)
        record = _size_one(
            sid,
            weight=float(weight),
            held=held,
            price=price,
            equity=equity,
            increment=increment,
            settings=settings,
            capabilities=capabilities,
            scale=scale,
            exempt=sid in exempt,
            obligation_qty=obligation_qty.get(sid),
            dust_dropped=dust_dropped,
        )
        if record is not None:
            records.append(record)
    # An obligation on a sid the target series does not mention is still executable.
    for sid, qty in obligation_qty.items():
        if sid in seen or qty == 0.0:
            continue
        price = _require_price(prices, sid)
        held = float(positions.get(sid, 0.0))
        increment = _increment_for(sid, settings, execution_profile, capabilities)
        records.append(
            QuantityRecord(
                sid=sid,
                target_qty=held + qty,
                increment=increment,
                held_qty=held,
                rounded_delta=qty,
                residual=0.0,
                reference_price=price,
            )
        )
    if dust_dropped:
        shown = sorted(dust_dropped, key=lambda item: -item[1])[:20]
        tail = "…" if len(dust_dropped) > len(shown) else ""
        logger.warning(
            "%d intended orders dropped by increment/notional rounding (book/target divergence, N7-loud): %s%s",
            len(dust_dropped),
            [(sid, round(notional, 2)) for sid, notional in shown],
            tail,
        )
    return records


def _increment_for(sid: str, settings: RoundingSettings, execution_profile: Any, capabilities: Any) -> float:
    """The per-sid increment, with the settings' continuous override honoured.

    ``RoundingSettings.for_reproduction`` forces a continuous increment, and that override
    must beat the profile's ``quantity_increment_default``: the reproduction engine held
    continuous quantities and reproduction is its only use.
    """
    if settings.quantity_increment_default <= 0.0:
        return 0.0
    return quantity_increment_for(sid, execution_profile=execution_profile, capabilities=capabilities)


def _require_price(prices: pd.Series, sid: str) -> float:
    price = prices.get(sid)
    if price is None or not math.isfinite(float(price)) or float(price) <= 0.0:
        raise RoundingError(f"no usable decision price for {sid!r} (got {price!r}); cannot size its target (SPEC N7)")
    return float(price)


def _size_one(
    sid: str,
    *,
    weight: float,
    held: float,
    price: float,
    equity: float,
    increment: float,
    settings: RoundingSettings,
    capabilities: Any,
    scale: float,
    exempt: bool,
    obligation_qty: float | None,
    dust_dropped: list[tuple[str, float]],
) -> QuantityRecord | None:
    target_shares = weight * equity / price
    clamped = False
    if settings.zero_crossing_clamp and held != 0.0 and target_shares != 0.0 and (held > 0.0) != (target_shares > 0.0):
        # A single venue order may not cross zero (Alpaca: "insufficient qty available").
        # Trade only to flat this bar; the opposite side opens next bar from flat.
        delta = -held
        clamped = True
    else:
        delta = target_shares - held
    if scale != 1.0:
        delta *= scale
    intended_notional = abs(delta) * price
    off_increment = increment > 0.0 and not _on_increment(held, increment)
    if off_increment and settings.off_increment_holding == "close_fractional":
        row = capability_row(capabilities, sid)
        if row is None or not bool(row.get("fractionable", False)):
            raise RoundingError(
                f"{sid}: [rounding] off_increment_holding = 'close_fractional' needs a capabilities row "
                "allowing a fractional order; none is available, so the residual cannot be closed"
            )
        rounded = delta
    else:
        rounded = round_delta(delta, increment)
    if obligation_qty is not None and obligation_qty != 0.0:
        rounded = _merge_obligation(rounded, obligation_qty)
    floor = 0.0 if (exempt or not settings.apply_dust_rule) else max(settings.min_notional, 1e-9)
    residual = target_shares - (held + rounded)
    if rounded == 0.0 or (floor > 0.0 and abs(rounded) * price < floor):
        if settings.apply_dust_rule and intended_notional >= max(settings.min_notional, 1.0):
            dust_dropped.append((sid, intended_notional))
        return QuantityRecord(
            sid=sid,
            target_qty=target_shares,
            increment=increment,
            held_qty=held,
            rounded_delta=0.0,
            residual=target_shares - held,
            reference_price=price,
            clamped=clamped,
            off_increment_holding=off_increment,
            dropped_by="dust" if rounded != 0.0 else "zero_delta",
        )
    return QuantityRecord(
        sid=sid,
        target_qty=target_shares,
        increment=increment,
        held_qty=held,
        rounded_delta=rounded,
        residual=residual,
        reference_price=price,
        clamped=clamped,
        off_increment_holding=off_increment,
    )


def _merge_obligation(rounded: float, obligation_qty: float) -> float:
    """Take at least the obligation's progress on this sid, in the obligation's direction.

    Obligation progress outranks every preference limit, so a target that would trade less
    than the obligation requires -- or in the other direction -- is overridden by the
    obligation leg, never averaged with it.
    """
    if obligation_qty > 0.0:
        return max(rounded, obligation_qty)
    return min(rounded, obligation_qty)


def _on_increment(qty: float, increment: float) -> bool:
    if increment <= 0.0:
        return True
    steps = qty / increment
    return abs(steps - round(steps)) <= 1e-9


def _intended_gross(records: Sequence[QuantityRecord], equity: float) -> float:
    return sum(abs(record.target_qty - record.held_qty) * record.reference_price for record in records) / equity


def _to_intents(
    records: Sequence[QuantityRecord],
    *,
    positions: Mapping[str, float],
    decision_bar: str,
    settings: RoundingSettings,
    order_type: str | None,
    obligation_sids: set[str],
) -> tuple[Any, ...]:
    continuous = any(record.increment <= 0.0 for record in records if record.rounded_delta != 0.0)
    intent_type = _intent_type(continuous_increment=continuous)
    resolved_type = order_type or settings.default_order_type
    tif = _TIME_IN_FORCE_FOR_TYPE.get(resolved_type)
    if tif is None:
        raise RoundingError(
            f"order type {resolved_type!r} has no declared time in force; the profile's vocabulary is "
            f"{sorted(_TIME_IN_FORCE_FOR_TYPE)}"
        )
    intents = []
    for record in sorted(records, key=lambda item: item.sid):
        if record.rounded_delta == 0.0:
            continue
        held = float(positions.get(record.sid, 0.0))
        action = "OB" if record.sid in obligation_sids else limits_mod.classify_action(held, record.rounded_delta)
        intents.append(
            intent_type(
                sid=record.sid,
                qty=record.rounded_delta,
                increment=record.increment,
                reference_price=record.reference_price,
                decision_bar=decision_bar,
                order_type=resolved_type,
                time_in_force=tif,
                action_class=action,
            )
        )
    return tuple(intents)


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------


def _repair_measure(
    records: Sequence[QuantityRecord],
    *,
    ctx: LimitContext,
    limits: Sequence[Limit],
    plan: ObligationPlan,
    prices: pd.Series,
) -> float:
    """``V(w)``: the summed holding-state violations plus the obligation shortfall.

    Non-negative by construction, because every ``v`` in the §3.7 table is a
    ``max(0, ...)`` and the shortfall is a magnitude. That is what makes the repair loop's
    decreasing measure bounded below, and so what makes it terminate.
    """
    implied = ctx.w0.copy()
    for record in records:
        if record.rounded_delta == 0.0:
            continue
        implied[record.sid] = float(implied.get(record.sid, 0.0)) + (
            record.rounded_delta * record.reference_price / ctx.equity
        )
    total = 0.0
    for limit in limits:
        if not is_holding_state(limit) or limit.name not in limits_mod.MEASURED_LIMITS:
            continue
        total += violation(limit, implied, ctx)
    return total + plan.shortfall_dollars(prices) / ctx.equity


def _repair(
    records: list[QuantityRecord],
    *,
    ctx: LimitContext,
    limits: Sequence[Limit],
    plan: ObligationPlan,
    prices: pd.Series,
    settings: RoundingSettings,
    positions: Mapping[str, float],
    capabilities: Any,
    exempt: set[str],
) -> tuple[list[QuantityRecord], tuple[RepairMove, ...]]:
    """Greedy single-increment repair with a decreasing measure, under action restrictions.

    A move is accepted only if it reduces ``V`` by at least ``repair_min_decrement``.
    Since ``V >= 0`` and every accepted move decreases it by that fixed quantum, at most
    ``ceil(V_0 / repair_min_decrement)`` moves are accepted; ``repair_max_moves`` caps the
    search independently. The loop can therefore not oscillate, whatever the increments
    and minimum notionals do.
    """
    moves: list[RepairMove] = []
    current = _repair_measure(records, ctx=ctx, limits=limits, plan=plan, prices=prices)
    budget = settings.repair_max_moves
    while budget > 0 and current > 0.0:
        best: tuple[float, int, int] | None = None  # (v_after, record index, steps)
        for index, record in enumerate(records):
            if record.sid in exempt:
                continue  # obligation progress is never traded away by a repair
            if record.increment <= 0.0:
                # A continuous increment has no single-increment move; it exists only for
                # the reproduction engine, which repairs nothing.
                continue
            increment = record.increment
            for steps in (-1, 1):
                candidate = _moved(record, steps * increment, positions=positions, settings=settings)
                if candidate is None:
                    continue
                trial = list(records)
                trial[index] = candidate
                value = _repair_measure(trial, ctx=ctx, limits=limits, plan=plan, prices=prices)
                if value <= current - settings.repair_min_decrement and (best is None or value < best[0]):
                    best = (value, index, steps)
        if best is None:
            break
        value, index, steps = best
        record = records[index]
        moved = _moved(record, steps * record.increment, positions=positions, settings=settings)
        assert moved is not None  # it was accepted above with the same arguments
        records[index] = moved
        moves.append(RepairMove(sid=record.sid, increment_steps=steps, v_before=current, v_after=value))
        current = value
        budget -= 1
    del capabilities  # the increment already came from the capabilities row at sizing time
    return records, tuple(moves)


def _moved(
    record: QuantityRecord,
    delta: float,
    *,
    positions: Mapping[str, float],
    settings: RoundingSettings,
) -> QuantityRecord | None:
    """One single-increment move, or ``None`` when an action restriction forbids it.

    A move may never cross zero and may never take an order below the minimum notional:
    both are action restrictions, and an action restriction is never waived, not even to
    reduce a preference violation.
    """
    qty = record.rounded_delta + delta
    if abs(qty) < 1e-12:
        return None
    held = float(positions.get(record.sid, 0.0))
    ends_at = held + qty
    if settings.zero_crossing_clamp and held != 0.0 and ends_at != 0.0 and (held > 0.0) != (ends_at > 0.0):
        return None
    if settings.apply_dust_rule and abs(qty) * record.reference_price < max(settings.min_notional, 1e-9):
        return None
    return replace(record, rounded_delta=qty, residual=record.target_qty - (held + qty))
