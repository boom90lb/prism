"""Obligations: what the book *must* reduce, and how far the executable set lets it (spec §3.7).

An obligation is not a trade. It is the highest-priority *reduction* the executable set
allows, evaluated fresh every bar: a recall, a delisting notice, a broker requirement, or
a component's ``liquidate`` instruction states where the position must end up, and the
action restrictions of the day state how far toward that it is possible to move. A halt,
a participation restriction, or a missing borrow can leave progress partial, and the
status then says why. A blocked obligation escalates with its reason; it is never
silently carried as though it were satisfied.

Three consequences the types below enforce rather than describe:

* ``target_qty`` is a **signed position**, never a trade. For a short of -100 shares,
  ``reduce_to`` with ``target_qty = -30`` reduces the short to -30; ``target_qty = +30``
  is refused, because that crosses zero. A crossing is a C2 action, decomposed by §3.10
  into a C1 leg and a C3 leg each authorized on its own, and is never smuggled through an
  obligation.
* ``remaining_qty`` is a non-negative magnitude in **shares**, ``abs(current - target_qty)``
  at creation, decremented only by observed fills (:func:`record_fill`) -- never by a
  submission, and never by an intent that the venue has not printed.
* Quantities are share-denominated, so a split rebases them on its effective date by the
  same factor that rebases the position (§3.6): :func:`split_adjust` turns a
  ``remaining_qty`` of 50 into 100 across a 2-for-1.

The ordering in :func:`order_obligations` is total and deterministic, so two
implementations of this module sequence the same obligations the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Mapping, Sequence

import pandas as pd

__all__ = [
    "OBLIGATION_KINDS",
    "OBLIGATION_SOURCES",
    "OBLIGATION_STATUSES",
    "ObligationError",
    "Obligation",
    "Executable",
    "ObligationLeg",
    "BlockedObligation",
    "ObligationPlan",
    "obligation_from_current",
    "order_obligations",
    "split_adjust",
    "record_fill",
    "plan_progress",
]


class ObligationError(ValueError):
    """An obligation declaration or transition that must not be silently repaired."""


OBLIGATION_KINDS: tuple[str, ...] = ("cover", "liquidate", "reduce_to")
OBLIGATION_STATUSES: tuple[str, ...] = ("open", "partial", "done", "blocked")
OBLIGATION_SOURCES: tuple[str, ...] = (
    "recall",
    "delisting_notice",
    "broker_requirement",
    "component_liquidate",
)


@dataclass
class Obligation:
    """One obligation (spec §3.7).

    ``sid`` names the security; ``kind`` is ``cover`` (a recall on a short),
    ``liquidate`` (out entirely), or ``reduce_to`` (down to a stated signed position);
    ``deadline`` is an ISO date or ``None``; ``priority`` orders obligations against each
    other, lower first; ``status`` and ``reason`` carry why an obligation is where it is.
    """

    sid: str
    kind: Literal["cover", "liquidate", "reduce_to"]
    target_qty: float
    remaining_qty: float
    deadline: str | None = None
    priority: int = 0
    status: Literal["open", "partial", "done", "blocked"] = "open"
    reason: str = ""
    source: str = "broker_requirement"
    #: The signed position the obligation was created against, kept so that direction and
    #: the ``abs(target) <= abs(current)`` invariant remain checkable after creation.
    origin_qty: float | None = None

    def __post_init__(self) -> None:
        if not str(self.sid).strip():
            raise ObligationError("an obligation needs a sid")
        if self.kind not in OBLIGATION_KINDS:
            raise ObligationError(f"{self.sid}: kind must be one of {OBLIGATION_KINDS}, got {self.kind!r}")
        if self.status not in OBLIGATION_STATUSES:
            raise ObligationError(f"{self.sid}: status must be one of {OBLIGATION_STATUSES}, got {self.status!r}")
        if self.source not in OBLIGATION_SOURCES:
            raise ObligationError(f"{self.sid}: source must be one of {OBLIGATION_SOURCES}, got {self.source!r}")
        target = float(self.target_qty)
        remaining = float(self.remaining_qty)
        if not math.isfinite(target):
            raise ObligationError(f"{self.sid}: target_qty must be finite, got {self.target_qty!r}")
        if not math.isfinite(remaining) or remaining < 0.0:
            raise ObligationError(
                f"{self.sid}: remaining_qty is a non-negative magnitude in shares, got {self.remaining_qty!r}"
            )
        if self.kind in ("cover", "liquidate") and target != 0.0:
            raise ObligationError(
                f"{self.sid}: a {self.kind} obligation ends flat, so target_qty is 0.0, got {target}"
            )
        if not isinstance(self.priority, int):
            raise ObligationError(f"{self.sid}: priority must be an integer, got {self.priority!r}")
        if self.deadline is not None and not str(self.deadline).strip():
            raise ObligationError(f"{self.sid}: a deadline is an ISO date or None, not an empty string")
        self.target_qty = target
        self.remaining_qty = remaining
        if self.origin_qty is not None:
            self.origin_qty = float(self.origin_qty)
            _check_against_current(self.sid, self.kind, target, self.origin_qty)

    # -- derived ------------------------------------------------------------

    def direction(self, current_qty: float) -> float:
        """``+1`` when progress buys, ``-1`` when it sells, ``0.0`` when nothing is left.

        Progress always moves *toward* ``target_qty``, so the direction is the sign of
        ``target_qty - current_qty``. A current position already at or past the target
        leaves nothing to do.
        """
        gap = float(self.target_qty) - float(current_qty)
        if self.remaining_qty <= 0.0 or gap == 0.0:
            return 0.0
        return 1.0 if gap > 0.0 else -1.0

    def shortfall_qty(self, current_qty: float) -> float:
        """Shares still to trade: the smaller of ``remaining_qty`` and the live gap.

        The live gap matters because a broker-initiated fill -- a buy-in, a forced
        liquidation -- moves the position without this engine's order. The obligation is
        then already satisfied and must not generate a further order.
        """
        gap = abs(float(self.target_qty) - float(current_qty))
        return min(float(self.remaining_qty), gap)

    def to_dict(self) -> dict[str, Any]:
        """The record shape, JSON-safe."""
        return {
            "sid": self.sid,
            "kind": self.kind,
            "target_qty": self.target_qty,
            "remaining_qty": self.remaining_qty,
            "deadline": self.deadline,
            "priority": self.priority,
            "status": self.status,
            "reason": self.reason,
            "source": self.source,
        }


def _check_against_current(sid: str, kind: str, target: float, current: float) -> None:
    if target != 0.0 and current != 0.0 and (target > 0.0) != (current > 0.0):
        raise ObligationError(
            f"{sid}: target_qty {target} crosses zero against the held {current}; a crossing is a C2 "
            "action, decomposed by §3.10 into a C1 leg and a C3 leg each authorized on its own, and is "
            "never expressed as an obligation"
        )
    if abs(target) > abs(current) + 1e-12:
        raise ObligationError(
            f"{sid}: an obligation only reduces, so abs(target_qty) <= abs(current); got "
            f"target {target} against current {current}"
        )
    if target == 0.0 and current == 0.0 and kind != "liquidate":
        # A reduce_to-zero on a flat book is expressible but empty; a liquidate on a flat
        # book is the ordinary "already done" case a component instruction produces.
        return


def obligation_from_current(
    sid: str,
    kind: Literal["cover", "liquidate", "reduce_to"],
    *,
    current_qty: float,
    target_qty: float = 0.0,
    deadline: str | None = None,
    priority: int = 0,
    source: str = "broker_requirement",
    reason: str = "",
) -> Obligation:
    """Create an obligation against a held position, deriving ``remaining_qty`` from it.

    ``remaining_qty = abs(current_qty - target_qty)`` at creation, which is the only place
    that identity holds: from then on the field is decremented by observed fills alone.
    """
    current = float(current_qty)
    target = 0.0 if kind in ("cover", "liquidate") else float(target_qty)
    if kind == "cover" and current > 0.0:
        raise ObligationError(f"{sid}: a cover obligation applies to a short, but {current} is held long")
    _check_against_current(sid, kind, target, current)
    return Obligation(
        sid=sid,
        kind=kind,
        target_qty=target,
        remaining_qty=abs(current - target),
        deadline=deadline,
        priority=priority,
        status="done" if abs(current - target) == 0.0 else "open",
        reason=reason,
        source=source,
        origin_qty=current,
    )


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------

#: Sorts after every real deadline, so ``deadline = None`` is last.
_NULL_DEADLINE = "￿"


def order_obligations(
    obligations: Iterable[Obligation],
    reference_prices: Mapping[str, float] | pd.Series | None = None,
) -> tuple[Obligation, ...]:
    """The total order of §3.7: priority ascending, deadline ascending with null last,
    remaining notional at the reference price descending, sid ascending.

    A sid without a reference price contributes notional 0.0 to the ordering, which puts
    it after the priced obligations of the same ``(priority, deadline)`` group. The order
    stays total either way, because ``sid`` breaks every remaining tie.
    """
    prices = pd.Series(dtype=float) if reference_prices is None else pd.Series(reference_prices, dtype=float)

    def key(obligation: Obligation) -> tuple[int, str, float, str]:
        price = prices.get(obligation.sid, float("nan"))
        notional = (
            0.0 if price is None or not math.isfinite(float(price)) else abs(float(price)) * obligation.remaining_qty
        )
        deadline = _NULL_DEADLINE if obligation.deadline is None else str(obligation.deadline)
        return (int(obligation.priority), deadline, -notional, str(obligation.sid))

    return tuple(sorted(obligations, key=key))


def group_by_priority_deadline(
    obligations: Sequence[Obligation],
) -> tuple[tuple[tuple[int, str | None], tuple[Obligation, ...]], ...]:
    """The ordered obligations split into the ``(priority, deadline)`` groups R0 takes
    one at a time, in the order :func:`order_obligations` produced."""
    groups: list[tuple[tuple[int, str | None], list[Obligation]]] = []
    for obligation in obligations:
        key = (int(obligation.priority), obligation.deadline)
        if groups and groups[-1][0] == key:
            groups[-1][1].append(obligation)
        else:
            groups.append((key, [obligation]))
    return tuple((key, tuple(members)) for key, members in groups)


# --------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------


def split_adjust(obligation: Obligation, factor: float) -> Obligation:
    """Rebase an obligation across a split by the factor that rebases the position (§3.6).

    A 2-for-1 split is ``factor = 2.0``: a ``remaining_qty`` of 50 becomes 100 and a
    ``target_qty`` of -30 becomes -60. Share-denominated quantities that do not rebase
    would silently change the obligation's economic size on the effective date.
    """
    value = float(factor)
    if not math.isfinite(value) or value <= 0.0:
        raise ObligationError(f"{obligation.sid}: a split factor is finite and > 0, got {factor!r}")
    return replace(
        obligation,
        target_qty=obligation.target_qty * value,
        remaining_qty=obligation.remaining_qty * value,
        origin_qty=None if obligation.origin_qty is None else obligation.origin_qty * value,
    )


def record_fill(obligation: Obligation, filled_qty: float, *, tolerance: float = 1e-9) -> Obligation:
    """Decrement ``remaining_qty`` by an **observed** fill and restate the status.

    ``filled_qty`` is the signed quantity the venue printed. Only the component of it
    that moves toward ``target_qty`` counts as progress: a fill in the wrong direction is
    refused rather than credited, because crediting it would retire an obligation the book
    moved *away* from satisfying.
    """
    qty = float(filled_qty)
    if not math.isfinite(qty):
        raise ObligationError(f"{obligation.sid}: a fill quantity must be finite, got {filled_qty!r}")
    if qty == 0.0:
        return replace(obligation)
    if obligation.origin_qty is not None:
        toward = -1.0 if obligation.origin_qty > obligation.target_qty else 1.0
    else:
        toward = -1.0 if obligation.target_qty <= 0.0 else 1.0
    if (qty > 0.0) != (toward > 0.0):
        raise ObligationError(
            f"{obligation.sid}: fill of {qty} moves away from target_qty {obligation.target_qty}; an "
            "obligation is never credited with a fill in the opposite direction"
        )
    progress = min(abs(qty), obligation.remaining_qty)
    remaining = obligation.remaining_qty - progress
    if remaining <= tolerance:
        remaining, status = 0.0, "done"
    else:
        status = "partial"
    return replace(
        obligation,
        remaining_qty=remaining,
        status=status,
        origin_qty=None if obligation.origin_qty is None else obligation.origin_qty + qty,
    )


# --------------------------------------------------------------------------
# Progress against the executable set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Executable:
    """What the action restrictions allow on one sid this bar.

    ``max_reduce_qty`` is the magnitude of reduction the restrictions permit -- zero when
    the name is halted, when no borrow is available for the leg, or when a participation
    bound leaves no room. ``increment`` is the per-sid quantity increment (``0.0`` for a
    continuous increment), so that a plan never proposes a quantity the venue cannot
    accept. ``reason`` says why the allowance is what it is, and is what a blocked
    obligation escalates with.
    """

    sid: str
    max_reduce_qty: float
    increment: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        allowed = float(self.max_reduce_qty)
        if not math.isfinite(allowed) or allowed < 0.0:
            raise ObligationError(f"{self.sid}: max_reduce_qty is a non-negative magnitude, got {allowed!r}")
        increment = float(self.increment)
        if not math.isfinite(increment) or increment < 0.0:
            raise ObligationError(f"{self.sid}: increment must be finite and >= 0, got {increment!r}")
        object.__setattr__(self, "max_reduce_qty", allowed)
        object.__setattr__(self, "increment", increment)


@dataclass(frozen=True)
class ObligationLeg:
    """One obligation's executable progress this bar, as a signed share quantity."""

    sid: str
    qty: float
    progress: float
    obligation: Obligation
    partial: bool
    reason: str = ""


@dataclass(frozen=True)
class BlockedObligation:
    """An obligation the executable set allows no progress on, with the reason."""

    obligation: Obligation
    reason: str


@dataclass(frozen=True)
class ObligationPlan:
    """The result of :func:`plan_progress`."""

    legs: tuple[ObligationLeg, ...] = ()
    blocked: tuple[BlockedObligation, ...] = ()
    escalations: tuple[str, ...] = ()
    #: The sids an obligation leg touches, which batch scaling must never scale down.
    protected_sids: frozenset[str] = field(default_factory=frozenset)

    def qty_by_sid(self) -> dict[str, float]:
        """The signed obligation quantity per sid, summed over legs on the same name."""
        out: dict[str, float] = {}
        for leg in self.legs:
            out[leg.sid] = out.get(leg.sid, 0.0) + leg.qty
        return out

    def shortfall_dollars(self, reference_prices: Mapping[str, float] | pd.Series) -> float:
        """Unmet obligation quantity valued at the reference prices, in dollars.

        This is the obligation term of the repair measure ``V`` in ``rounding.py``: a
        batch that leaves obligation progress on the table measures worse than one that
        does not, so repair never trades obligation progress away for a preference.
        """
        prices = pd.Series(reference_prices, dtype=float)
        total = 0.0
        for blocked in self.blocked:
            price = prices.get(blocked.obligation.sid, float("nan"))
            if math.isfinite(float(price)):
                total += abs(float(price)) * blocked.obligation.remaining_qty
        for leg in self.legs:
            price = prices.get(leg.sid, float("nan"))
            if math.isfinite(float(price)):
                total += abs(float(price)) * max(0.0, leg.obligation.remaining_qty - leg.progress)
        return total


def plan_progress(
    obligations: Iterable[Obligation],
    positions: Mapping[str, float],
    executable: Mapping[str, Executable],
    *,
    reference_prices: Mapping[str, float] | pd.Series | None = None,
    on_blocked_obligation: str = "escalate",
) -> ObligationPlan:
    """The largest progress on obligations the action restrictions allow (R0 of §3.7).

    Obligations are taken in the total order of :func:`order_obligations`, one
    ``(priority, deadline)`` group at a time, and within a sid an earlier obligation's
    allowance is consumed before a later one sees it -- which is what makes "highest
    priority first" true per name and not only per group. Progress is the magnitude moved
    toward ``target_qty``, clipped to ``remaining_qty`` and to the sid's executable
    allowance, then floored to the sid's quantity increment.

    An obligation with an allowance of zero is **blocked**, not dropped:
    ``on_blocked_obligation`` decides whether that is recorded or escalated, and the
    escalation carries the venue's or the restriction's own reason. An obligation whose
    allowance is positive but smaller than what remains is **partial**, and the leg says
    so.
    """
    if on_blocked_obligation not in ("record", "escalate"):
        raise ObligationError(f"on_blocked_obligation is 'record' or 'escalate', got {on_blocked_obligation!r}")
    ordered = order_obligations(obligations, reference_prices)
    consumed: dict[str, float] = {}
    legs: list[ObligationLeg] = []
    blocked: list[BlockedObligation] = []
    escalations: list[str] = []
    for obligation in ordered:
        if obligation.status == "done" or obligation.remaining_qty <= 0.0:
            continue
        sid = obligation.sid
        current = float(positions.get(sid, 0.0))
        direction = obligation.direction(current)
        wanted = obligation.shortfall_qty(current)
        if direction == 0.0 or wanted <= 0.0:
            continue
        allowance = executable.get(sid)
        if allowance is None:
            reason = f"no executable allowance supplied for {sid}"
            blocked.append(BlockedObligation(obligation, reason))
            if on_blocked_obligation == "escalate":
                escalations.append(f"{sid}: {reason}")
            continue
        already = consumed.get(sid, 0.0)
        room = max(0.0, allowance.max_reduce_qty - already)
        progress = min(wanted, room)
        if allowance.increment > 0.0:
            progress = math.floor(progress / allowance.increment + 1e-9) * allowance.increment
        if progress <= 0.0:
            reason = allowance.reason or (
                f"the executable set allows no reduction in {sid} this bar "
                f"(allowance {allowance.max_reduce_qty}, already committed {already})"
            )
            blocked.append(BlockedObligation(obligation, reason))
            if on_blocked_obligation == "escalate":
                escalations.append(f"{sid}: {reason}")
            continue
        consumed[sid] = already + progress
        partial = progress + 1e-12 < wanted
        legs.append(
            ObligationLeg(
                sid=sid,
                qty=direction * progress,
                progress=progress,
                obligation=obligation,
                partial=partial,
                reason=allowance.reason if partial else "",
            )
        )
    return ObligationPlan(
        legs=tuple(legs),
        blocked=tuple(blocked),
        escalations=tuple(escalations),
        protected_sids=frozenset(leg.sid for leg in legs),
    )
