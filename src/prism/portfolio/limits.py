"""The two limit families, their four classes, and the violation measures (spec §3.7).

The families are kept apart because recovery treats them differently, and the
difference is the whole point of the split:

``ACTION`` restrictions -- :attr:`LimitClass.PHYSICAL` and
    :attr:`LimitClass.MANDATORY_ACTION` -- say what the engine may *do*. They are
    **never waived and never reduced**, on any path, so they carry no priority. A
    ``priority`` key on such a row is *recorded as ignored* rather than refused, so the
    reference profiles of §3.13 (``profiles/etf_small.toml`` writes ``priority = 0`` on
    its two physical rows) load unchanged.

``HOLDING_STATE`` limits -- :attr:`LimitClass.MANDATORY_STATE` and
    :attr:`LimitClass.PREFERENCE` -- say what the book may *hold*. They are reduced in a
    declared order when recovery cannot satisfy all of them: every mandatory-state limit
    before any preference limit, and inside each class by ascending ``priority`` (a
    **lower priority integer is reduced first**, the draft-1 rule draft 1.1 dropped).

The priority ordering between the families is therefore total and testable:
:func:`reduction_order` is the single implementation of it, and
:func:`is_action_restriction` is the single answer to "may this be reduced".

Two holding-state limits of the *same class* with equal priority are refused at load:
an ordering the file does not fix cannot be recovered deterministically. Equal priorities
*across* the two holding-state classes are accepted, because the class rank already fixes
that pair's order (see :func:`reduction_order`).

**Violation measures.** Every holding-state limit is a declared convex expression on the
proposed target ``w`` and the marked weights ``w0``, both in fractions of marked equity,
and :func:`violation` is the single evaluator of the §3.7 table. ``participation_per_name``
is declared ``physical`` by the reference profile yet carries a measure in that table: it
restricts the *trade*, so it is never reduced, but its excess is measurable and is
reported. That is why :func:`violation` keys off the limit's ``name``, and the class only
decides how the outcome is treated.

This module is the stage-3 limit-evaluation shim the heuristic constructor gains: it
evaluates every declared holding-state limit on ``w0`` and on the produced target and
fills ``violations`` and ``binding_condition`` **without changing the target**. It is not
a repair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    "ACTION_RESTRICTION_CLASSES",
    "HOLDING_STATE_CLASSES",
    "MEASURED_LIMITS",
    "BANDED_LIMITS",
    "LimitClass",
    "LimitError",
    "Limit",
    "LimitContext",
    "LimitEvaluation",
    "RiskLike",
    "is_action_restriction",
    "is_holding_state",
    "load_limits",
    "reduction_order",
    "violation",
    "evaluate_limits",
    "partition_violations",
    "binding_condition",
]


class LimitError(ValueError):
    """A limit declaration, or an evaluation input, that must not be silently repaired."""


class LimitClass(str, Enum):
    """The four classes. The value is the spelling a policy file uses in ``class``."""

    PHYSICAL = "physical"
    MANDATORY_ACTION = "mandatory_action"
    MANDATORY_STATE = "mandatory_state"
    PREFERENCE = "preference"


#: The action-restriction family: never waived, never reduced, no priority.
ACTION_RESTRICTION_CLASSES: frozenset[LimitClass] = frozenset({LimitClass.PHYSICAL, LimitClass.MANDATORY_ACTION})
#: The holding-state family: reduced in recovery, mandatory state first.
HOLDING_STATE_CLASSES: frozenset[LimitClass] = frozenset({LimitClass.MANDATORY_STATE, LimitClass.PREFERENCE})

#: The class rank :func:`reduction_order` sorts on before ``priority``. Mandatory-state
#: limits are reduced first; preferences after them.
_CLASS_RANK: dict[LimitClass, int] = {LimitClass.MANDATORY_STATE: 0, LimitClass.PREFERENCE: 1}

#: The limits of the §3.7 measure table, the only names :func:`violation` evaluates.
MEASURED_LIMITS: frozenset[str] = frozenset(
    {
        "max_gross",
        "max_name_weight",
        "net_band",
        "beta_band",
        "variance_budget",
        "turnover_cap",
        "participation_per_name",
    }
)

#: The two banded limits: ``value = x`` on the row reads as ``centre = 0, half_width = x``.
BANDED_LIMITS: frozenset[str] = frozenset({"net_band", "beta_band"})


def is_action_restriction(limit: "Limit") -> bool:
    """True when the limit is an action restriction, which is never waived or reduced."""
    return limit.cls in ACTION_RESTRICTION_CLASSES


def is_holding_state(limit: "Limit") -> bool:
    """True when the limit constrains holdings, and so takes part in the reduction order."""
    return limit.cls in HOLDING_STATE_CLASSES


@runtime_checkable
class RiskLike(Protocol):
    """The structural surface :func:`violation` needs from a §3.5 ``RiskSnapshot``.

    Declared structurally rather than imported: the snapshot type lands with the risk
    model, and the limit evaluator must not depend on the solver stack to answer what a
    variance budget or a beta band measures.
    """

    sids: Sequence[str]
    method: Mapping[str, Any]

    def variance(self, w: Any) -> float:  # pragma: no cover - protocol declaration
        ...


@dataclass(frozen=True)
class Limit:
    """One declared limit (spec §3.7).

    ``params`` is a table, not a scalar, so a banded limit is expressible: ``value`` for
    the one-sided limits, ``centre`` plus ``half_width`` for the bands. ``tolerance``
    overrides the profile's ``limit_tolerance`` for this limit alone. ``source`` is the
    profile path, ``"capabilities"``, ``"broker"``, or ``"notice"``.

    Construction is strict: this is the validated record, and :func:`load_limits` is the
    tolerant edge that turns a file's rows into it.
    """

    name: str
    cls: LimitClass
    priority: int | None
    params: Mapping[str, Any] = field(default_factory=dict)
    tolerance: float | None = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise LimitError("a limit needs a name")
        if not isinstance(self.cls, LimitClass):
            raise LimitError(f"{self.name}: cls must be a LimitClass, got {self.cls!r}")
        if is_action_restriction(self) and self.priority is not None:
            raise LimitError(
                f"{self.name}: an action restriction ({self.cls.value}) is never reduced and carries no "
                "priority; load_limits() records a declared priority as ignored, it is not a Limit field"
            )
        if is_holding_state(self) and self.priority is None:
            raise LimitError(
                f"{self.name}: a holding-state limit ({self.cls.value}) is reduced in declared priority "
                "order and must declare an integer priority"
            )
        if self.priority is not None and not isinstance(self.priority, int):
            raise LimitError(f"{self.name}: priority must be an integer, got {self.priority!r}")
        if self.tolerance is not None:
            tol = float(self.tolerance)
            if not math.isfinite(tol) or tol < 0.0:
                raise LimitError(f"{self.name}: tolerance must be finite and >= 0, got {self.tolerance!r}")
            object.__setattr__(self, "tolerance", tol)
        object.__setattr__(self, "params", dict(self.params))

    # -- parameter access ---------------------------------------------------

    def value(self) -> float:
        """The one-sided limit's ``value``, refused when the row did not declare one."""
        if "value" not in self.params:
            raise LimitError(f"{self.name}: no `value` declared in params {dict(self.params)!r}")
        val = float(self.params["value"])
        if not math.isfinite(val):
            raise LimitError(f"{self.name}: value must be finite, got {self.params['value']!r}")
        return val

    def band(self) -> tuple[float, float]:
        """A banded limit's ``(centre, half_width)``; ``value = x`` reads as ``(0.0, x)``."""
        params = self.params
        if "half_width" in params or "centre" in params:
            centre = float(params.get("centre", 0.0))
            if "half_width" not in params:
                raise LimitError(f"{self.name}: a band declaring `centre` must declare `half_width`")
            half = float(params["half_width"])
        else:
            centre, half = 0.0, self.value()
        if not math.isfinite(centre) or not math.isfinite(half):
            raise LimitError(f"{self.name}: band centre and half_width must be finite, got {centre}, {half}")
        if half < 0.0:
            raise LimitError(f"{self.name}: band half_width must be >= 0, got {half}")
        return centre, half

    def effective_tolerance(self, default: float) -> float:
        """This limit's tolerance: its own override, else the profile's ``limit_tolerance``."""
        return float(default) if self.tolerance is None else float(self.tolerance)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

_ROW_KEYS = frozenset({"name", "class", "priority", "params", "tolerance", "value", "centre", "half_width"})


def load_limits(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str,
) -> tuple[tuple[Limit, ...], tuple[str, ...]]:
    """Parse ``[[limits]]`` rows into validated limits, plus the notes of what was ignored.

    The loader accepts no spelling of ``class`` but the four enum values and no other
    case. Scalars written on the row (``value``, ``centre``, ``half_width``) fold into
    ``params``, which is how the reference profiles declare their limits; an explicit
    ``[limits.params]`` table is merged over them.

    A ``priority`` on an action restriction is dropped and reported in the second element
    of the return, never refused. Two holding-state limits of the same class with equal
    priority are refused.
    """
    parsed: list[Limit] = []
    ignored: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        where = f"{source} [[limits]] #{index + 1}"
        unknown = set(row) - _ROW_KEYS
        if unknown:
            raise LimitError(f"{where}: unknown keys {sorted(unknown)}; declared keys are {sorted(_ROW_KEYS)}")
        name = str(row.get("name", "")).strip()
        if not name:
            raise LimitError(f"{where}: every limit row declares a name")
        if name in seen:
            raise LimitError(f"{where}: {name!r} is declared twice; one row per limit")
        seen.add(name)
        raw_class = row.get("class")
        if not isinstance(raw_class, str):
            raise LimitError(f"{where}: {name} declares no `class`")
        try:
            cls = LimitClass(raw_class)
        except ValueError as exc:
            allowed = ", ".join(repr(member.value) for member in LimitClass)
            raise LimitError(f"{where}: {name} has class {raw_class!r}; allowed spellings are {allowed}") from exc
        params: dict[str, Any] = {}
        for scalar in ("value", "centre", "half_width"):
            if scalar in row:
                params[scalar] = row[scalar]
        table = row.get("params")
        if table is not None:
            if not isinstance(table, Mapping):
                raise LimitError(f"{where}: {name} params must be a table, got {type(table).__name__}")
            params.update(table)
        priority = row.get("priority")
        if cls in ACTION_RESTRICTION_CLASSES and priority is not None:
            ignored.append(
                f"{where}: priority {priority!r} on {name} ({cls.value}) is ignored; an action restriction "
                "is never reduced and carries no priority"
            )
            priority = None
        if priority is not None and not isinstance(priority, int):
            raise LimitError(f"{where}: {name} priority must be an integer, got {priority!r}")
        parsed.append(
            Limit(
                name=name,
                cls=cls,
                priority=priority,
                params=params,
                tolerance=None if row.get("tolerance") is None else float(row["tolerance"]),
                source=source,
            )
        )
    _refuse_equal_priorities(parsed, source=source)
    return tuple(parsed), tuple(ignored)


def _refuse_equal_priorities(limits: Sequence[Limit], *, source: str) -> None:
    by_class: dict[LimitClass, dict[int, str]] = {}
    for limit in limits:
        if not is_holding_state(limit):
            continue
        assert limit.priority is not None  # guaranteed by Limit.__post_init__
        bucket = by_class.setdefault(limit.cls, {})
        other = bucket.get(limit.priority)
        if other is not None:
            raise LimitError(
                f"{source}: {other!r} and {limit.name!r} are both {limit.cls.value} at priority "
                f"{limit.priority}; an ordering the file does not fix cannot be recovered "
                "deterministically, so equal priorities inside one holding-state class are refused"
            )
        bucket[limit.priority] = limit.name


def reduction_order(limits: Iterable[Limit]) -> tuple[Limit, ...]:
    """The holding-state limits in the order recovery reduces them.

    Mandatory-state limits first, then preferences; inside each class by ascending
    ``priority``, ties broken by name so the order is total even for a caller that built
    the limits in code rather than through :func:`load_limits`. Action restrictions are
    absent from the result: they are never reduced.
    """
    holding = [limit for limit in limits if is_holding_state(limit)]
    return tuple(sorted(holding, key=lambda lim: (_CLASS_RANK[lim.cls], int(lim.priority or 0), lim.name)))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LimitContext:
    """Everything the §3.7 measures read besides the proposed weights.

    ``w0`` is the marked state's weights, ``equity`` the marked equity in dollars, and
    ``adv_dollars`` the ``dollar_volume.participation`` statistic the cost stack declares
    (§3.6, D10) -- not the spread-bucketing or impact statistic. ``risk`` is a §3.5
    snapshot, required only by ``variance_budget`` and ``beta_band``.
    """

    w0: pd.Series
    equity: float = 1.0
    adv_dollars: pd.Series | None = None
    risk: RiskLike | None = None

    def __post_init__(self) -> None:
        w0 = pd.Series(self.w0, dtype=float)
        if not np.isfinite(w0.to_numpy(dtype=float)).all():
            offenders = [str(sid) for sid in w0.index[~np.isfinite(w0.to_numpy(dtype=float))]][:5]
            raise LimitError(f"marked weights must be finite; non-finite at {offenders}")
        object.__setattr__(self, "w0", w0)
        equity = float(self.equity)
        if not math.isfinite(equity) or equity <= 0.0:
            raise LimitError(f"equity must be finite and > 0 to measure a limit, got {self.equity!r}")
        object.__setattr__(self, "equity", equity)
        if self.adv_dollars is not None:
            object.__setattr__(self, "adv_dollars", pd.Series(self.adv_dollars, dtype=float))


def _aligned(w: pd.Series, ctx: LimitContext) -> tuple[pd.Series, pd.Series]:
    """``(w, w0)`` on the union index, a name absent from either read as flat there."""
    target = pd.Series(w, dtype=float)
    values = target.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        offenders = [str(sid) for sid in target.index[~np.isfinite(values)]][:5]
        raise LimitError(
            f"proposed weights must be finite to measure a limit; non-finite at {offenders}. A NaN target "
            "is 'no decision' and is resolved to the held weight before a limit is measured, never here"
        )
    index = target.index.union(ctx.w0.index)
    return target.reindex(index).fillna(0.0), ctx.w0.reindex(index).fillna(0.0)


def violation(limit: Limit, w: pd.Series, ctx: LimitContext) -> float:
    """The §3.7 violation measure ``v(w)`` for one limit; ``0.0`` when it is satisfied.

    Every measure is convex in ``w`` and non-negative, which is what makes the recovery
    bounds convex constraints and what makes the repair measure of ``rounding.py``
    monotone. A limit whose name is not in :data:`MEASURED_LIMITS` has no declared measure
    and raises rather than returning a silent zero.
    """
    name = limit.name
    if name not in MEASURED_LIMITS:
        raise LimitError(
            f"{name!r} declares no violation measure in the §3.7 table; measured limits are "
            f"{sorted(MEASURED_LIMITS)}. An action restriction without a measure is checked as a "
            "predicate by portfolio/validate.py, never scored here"
        )
    target, held = _aligned(w, ctx)
    if name == "max_gross":
        return max(0.0, float(target.abs().sum()) - limit.value())
    if name == "max_name_weight":
        cap = _per_name_cap(limit, target.index)
        return float(np.maximum(0.0, target.abs().to_numpy(dtype=float) - cap).sum())
    if name == "net_band":
        centre, half = limit.band()
        return max(0.0, abs(float(target.sum()) - centre) - half)
    if name == "beta_band":
        centre, half = limit.band()
        beta = _factor_column(limit, ctx, target.index)
        exposure = float(np.dot(beta, target.to_numpy(dtype=float)))
        return max(0.0, abs(exposure - centre) - half)
    if name == "variance_budget":
        return max(0.0, _ex_ante_sigma(ctx, target) - limit.value())
    if name == "turnover_cap":
        return max(0.0, float((target - held).abs().sum()) - limit.value())
    # participation_per_name
    adv = _adv_dollars(limit, ctx, target.index)
    traded = (target - held).abs().to_numpy(dtype=float) * ctx.equity
    excess = np.maximum(0.0, traded - limit.value() * adv)
    return float(excess.sum() / ctx.equity)


def _per_name_cap(limit: Limit, index: pd.Index) -> np.ndarray:
    """``value`` as a scalar cap, or ``value`` as a per-name table aligned to ``index``."""
    raw = limit.params.get("value")
    if isinstance(raw, Mapping) or isinstance(raw, pd.Series):
        caps = pd.Series(raw, dtype=float).reindex(index)
        if caps.isna().any():
            missing = [str(sid) for sid in index[caps.isna()]][:5]
            raise LimitError(
                f"{limit.name}: per-name cap table covers no value for {missing}; a name without a "
                "declared cap is not silently uncapped"
            )
        return caps.to_numpy(dtype=float)
    return np.full(len(index), limit.value(), dtype=float)


def _factor_column(limit: Limit, ctx: LimitContext, index: pd.Index) -> np.ndarray:
    """The loading column ``b`` of the factor ``params.factor_id`` names."""
    factor_id = limit.params.get("factor_id")
    if not isinstance(factor_id, str) or not factor_id:
        raise LimitError(f"{limit.name}: a beta band declares params.factor_id naming the factor it bounds")
    risk = _require_risk(limit, ctx)
    loadings = getattr(risk, "B", None)
    if loadings is None:
        raise LimitError(f"{limit.name}: the risk snapshot exposes no loadings B, so b'w is not evaluable")
    frame = pd.DataFrame(loadings)
    if factor_id not in frame.columns:
        raise LimitError(
            f"{limit.name}: factor {factor_id!r} is not a column of the snapshot's loadings "
            f"{[str(col) for col in frame.columns]}"
        )
    named = _method_factor_ids(risk)
    if named is not None and factor_id not in named:
        raise LimitError(
            f"{limit.name}: the snapshot's method block names factors {sorted(named)}, not {factor_id!r}; "
            "a beta band on a factor the snapshot does not declare is refused, never measured at zero"
        )
    column = frame[factor_id].reindex(index)
    if column.isna().any():
        missing = [str(sid) for sid in index[column.isna()]][:5]
        raise LimitError(
            f"{limit.name}: the snapshot carries no {factor_id!r} loading for {missing}; a missing loading "
            "is not a zero loading (D9: a name without an estimate is a declared substitution, recorded "
            "in the snapshot's method block, not an omission here)"
        )
    return column.to_numpy(dtype=float)


def _method_factor_ids(risk: RiskLike) -> set[str] | None:
    method = getattr(risk, "method", None)
    if not isinstance(method, Mapping):
        return None
    named = method.get("factor_ids")
    if named is None:
        return None
    return {str(item) for item in named}


def _ex_ante_sigma(ctx: LimitContext, target: pd.Series) -> float:
    """``sqrt(w' Sigma w)``, through the snapshot, never forming a dense covariance."""
    risk = ctx.risk
    if risk is None:
        raise LimitError(
            "variance_budget needs a risk snapshot; measuring it without one would report a satisfied "
            "budget for a book whose risk was never evaluated"
        )
    _require_coverage(risk, target.index)
    variance = float(risk.variance(target))
    if not math.isfinite(variance) or variance < 0.0:
        raise LimitError(f"the risk snapshot returned variance {variance!r}, which is not a variance (D9)")
    return math.sqrt(variance)


def _require_risk(limit: Limit, ctx: LimitContext) -> RiskLike:
    if ctx.risk is None:
        raise LimitError(f"{limit.name} needs a risk snapshot to measure; none was supplied")
    return ctx.risk


def _require_coverage(risk: RiskLike, index: pd.Index) -> None:
    """D9: a snapshot whose ``sids`` do not cover the universe is rejected before any solve."""
    covered = {str(sid) for sid in getattr(risk, "sids", ())}
    if not covered:
        return
    missing = [str(sid) for sid in index if str(sid) not in covered]
    if missing:
        raise LimitError(
            f"the risk snapshot covers no estimate for {missing[:5]}"
            f"{' and others' if len(missing) > 5 else ''}; D9 rejects a snapshot whose sids do not cover "
            "the constructor universe rather than treating an uncovered name as riskless"
        )


def _adv_dollars(limit: Limit, ctx: LimitContext, index: pd.Index) -> np.ndarray:
    if ctx.adv_dollars is None:
        raise LimitError(
            f"{limit.name} needs the declared dollar_volume.participation statistic; without it every "
            "trade would measure as unrestricted"
        )
    adv = ctx.adv_dollars.reindex(index)
    values = adv.to_numpy(dtype=float)
    unusable = ~np.isfinite(values) | (values <= 0.0)
    if unusable.any():
        # The fail-safe direction, as in the spread schedule's widest step: an
        # unknown dollar volume restricts the name to nothing, never to everything.
        values = np.where(unusable, 0.0, values)
    return values


@dataclass(frozen=True)
class LimitEvaluation:
    """One limit's measure on one book, with the stage the measure is attributed to."""

    limit_name: str
    cls: LimitClass
    priority: int | None
    v: float
    tolerance: float
    attributed_to: str

    @property
    def breached(self) -> bool:
        """True when the measure exceeds this limit's tolerance."""
        return self.v > self.tolerance

    def to_dict(self) -> dict[str, Any]:
        """The record shape §3.7 names: ``limit_name``, ``class``, ``priority``, ``v``, ``attributed_to``."""
        return {
            "limit_name": self.limit_name,
            "class": self.cls.value,
            "priority": self.priority,
            "v": self.v,
            "tolerance": self.tolerance,
            "attributed_to": self.attributed_to,
        }


def evaluate_limits(
    limits: Iterable[Limit],
    w: pd.Series,
    ctx: LimitContext,
    *,
    limit_tolerance: float = 1e-6,
    attributed_to: str = "marked",
) -> tuple[LimitEvaluation, ...]:
    """Measure every limit that has a declared measure, satisfied ones included.

    Satisfied limits are returned with ``v = 0.0`` rather than dropped: recovery walks
    "every holding-state limit in priority order, satisfied limits included with an
    achieved target of zero", and a record that omits the satisfied limits cannot show
    that they were evaluated.
    """
    out: list[LimitEvaluation] = []
    for limit in limits:
        if limit.name not in MEASURED_LIMITS:
            continue
        out.append(
            LimitEvaluation(
                limit_name=limit.name,
                cls=limit.cls,
                priority=limit.priority,
                v=violation(limit, w, ctx),
                tolerance=limit.effective_tolerance(limit_tolerance),
                attributed_to=attributed_to,
            )
        )
    return tuple(out)


def partition_violations(
    limits: Iterable[Limit],
    w_target: pd.Series,
    ctx: LimitContext,
    *,
    limit_tolerance: float = 1e-6,
    attributed_to: str = "rounding",
) -> dict[str, list[dict[str, Any]]]:
    """The ``{existing, remaining, introduced}`` partition of §3.7, on one target.

    ``existing`` is the marked state's measure for a limit the marked state breached;
    ``remaining`` is the target's measure for that same limit; ``introduced`` is a
    positive measure for a limit the marked state satisfied, attributed to the stage that
    first made it positive. The three sets are disjoint by construction: a limit appears
    in ``existing`` and ``remaining``, or in ``introduced``, or in neither.
    """
    limits = tuple(limits)
    marked = {ev.limit_name: ev for ev in evaluate_limits(limits, ctx.w0, ctx, limit_tolerance=limit_tolerance)}
    proposed = evaluate_limits(limits, w_target, ctx, limit_tolerance=limit_tolerance, attributed_to=attributed_to)
    partition: dict[str, list[dict[str, Any]]] = {"existing": [], "remaining": [], "introduced": []}
    for ev in proposed:
        before = marked[ev.limit_name]
        if before.breached:
            partition["existing"].append(before.to_dict())
            partition["remaining"].append(ev.to_dict())
        elif ev.breached:
            partition["introduced"].append(ev.to_dict())
    return partition


def binding_condition(partition: Mapping[str, Sequence[Mapping[str, Any]]]) -> str | None:
    """The limit that determined the outcome, or ``None`` when nothing bound.

    An introduced violation names the outcome before an inherited one does: an inherited
    breach is the state the decision started from, while an introduced breach is what this
    decision did. Inside each set the largest measure wins, ties broken by name.
    """
    for key in ("introduced", "remaining"):
        entries = list(partition.get(key, ()))
        if entries:
            worst = max(entries, key=lambda entry: (float(entry["v"]), str(entry["limit_name"])))
            return str(worst["limit_name"])
    return None


# --------------------------------------------------------------------------
# Action restrictions
# --------------------------------------------------------------------------

#: The restriction strings the ``capabilities`` row's ``restrictions`` column may carry
#: and this engine understands. Anything else is treated as blocking new exposure and is
#: recorded by name: an unrecognized broker restriction is a reason to do less, never a
#: reason to do more (the same fail-safe direction as the spread schedule's widest step).
KNOWN_RESTRICTIONS: frozenset[str] = frozenset({"halted", "no_trade", "no_new_exposure", "reduce_only"})


def restriction_blocks_all_trading(restrictions: Iterable[str]) -> bool:
    """True when the restriction set forbids trading the name at all this bar."""
    return any(str(item) in ("halted", "no_trade") for item in restrictions)


def restriction_blocks_new_exposure(restrictions: Iterable[str]) -> bool:
    """True when the restriction set permits reductions only (unknown strings included)."""
    for item in restrictions:
        text = str(item)
        if text in ("halted", "no_trade", "no_new_exposure", "reduce_only"):
            return True
        if text not in KNOWN_RESTRICTIONS:
            return True
    return False


def unknown_restrictions(restrictions: Iterable[str]) -> tuple[str, ...]:
    """The restriction strings this engine does not recognize, for the record."""
    return tuple(sorted({str(item) for item in restrictions if str(item) not in KNOWN_RESTRICTIONS}))


ActionClass = Literal["C0", "C1", "C2", "C3", "C4", "C5", "OB"]


def classify_action(held: float, delta: float) -> ActionClass:
    """The §3.10 action class of one share delta against one held quantity.

    ``C1`` reduces toward zero without crossing, ``C3`` opens or adds in either
    direction, and ``C2`` crosses through zero. Hedge classes (C4, C5) and obligation
    progress (OB) are assigned by the caller, which is the only party that knows whether
    the sid is a declared hedge instrument or carries an obligation.
    """
    if delta == 0.0:
        raise LimitError("a zero delta is not an action; it is the absence of one")
    if held == 0.0:
        return "C3"
    if (held > 0.0) == (delta > 0.0):
        return "C3"
    ends_at = held + delta
    if ends_at == 0.0 or (ends_at > 0.0) == (held > 0.0):
        return "C1"
    return "C2"
