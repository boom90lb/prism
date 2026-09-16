"""The ordered transitions of the canonical accounting contract (spec §3.6).

One decision at the close of bar ``t`` moves the book through five transitions, in this
order and no other:

1. **Mark at close ``t``** -- prices, then the holding-cost accrual for bar ``t``.
2. **Decide** -- targets become intents in shares at the declared increment, sized from
   ``equity(close_t)`` and close-``t`` reference prices.
3. **Submit after the close** -- intents become ``OrderRecord``s with stable client order
   ids and ``status = "pending"``.
4. **Fill at the open of ``t+1`` or later** -- cash moves by
   ``-qty * fill_price - commission - spread_dollars - impact_dollars``, positions move by
   filled quantity, and the remainder expires or stays pending per the time in force.
5. **Mark at close ``t+1``** -- as step 1. Weights drift with prices; nothing rebalances
   implicitly between decisions.

**Where events are applied, and why it is not inside step 1.** §3.6 lists events as part of
the close mark. The canonical engine applies them at the **start of the bar, before that
bar's fills** (:func:`apply_events`), and the reason is that the two rules §3.6 states
elsewhere cannot both hold otherwise:

* *Entitlement is the signed share count at the close of the bar before the ex-date*, so a
  buyer filled at the ex-date open is not entitled and a seller filled at the ex-date open
  is. Applying the booking after that open's fill would credit the ex-date buyer, which is
  precisely the ``research/trading.py`` behaviour §3.6 corrects.
* *A split against a ``raw`` series multiplies shares by the factor on the effective date.*
  The open of the effective date already prints on the post-split basis, so shares bought
  at that open are already post-split. Rebasing the whole position after the fill would
  multiply that lot a second time, and rebasing before the open is the only order in which
  both the carried shares and the newly filled shares end on one basis.

Applying events first makes the state's share count *identically* the prior close's count,
so entitlement needs no shadow copy of yesterday's book. The transition is still a recorded
transition with its own sequence number; only its position in the bar differs from the
prose, and the deviation is reported as a finding rather than absorbed silently.

Every transition returns a **new** state. A transition that raises therefore leaves the
caller's state exactly as it was, which is what makes "fail loud" safe to act on: there is
no half-applied bar to unwind.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.execution.cost_model import CostBreakdown, CostModel
from prism.execution.portfolio_state import (
    BORROW_ACCRUAL_KEY,
    CASH_INCOME_ACCRUAL_KEY,
    FINANCING_ACCRUAL_KEY,
    AccountingError,
    Fill,
    OrderIntent,
    OrderRecord,
    PortfolioState,
    Position,
    bar_key,
    distribution_key,
    terminal_payout_key,
)
from prism.io.events import Adjustment, Event, applied_share_factor, require_price_basis

__all__ = [
    "EVENT_APPLICATION_ORDER",
    "AppliedEvent",
    "CloseMark",
    "DecisionBatch",
    "EventOutcome",
    "FillOutcome",
    "apply_events",
    "apply_fills",
    "decide",
    "expire_orders",
    "mark_at_close",
    "mark_prices",
    "round_to_increment",
    "submit",
]

#: The order events are applied in on their effective date (§3.6). Splits first, because
#: every later step reads a share count; halts last, because they only change how the
#: position is valued. A ``rebase`` is not in this list because it is not an event: it is a
#: row of the ``adjustments`` table, and :func:`apply_events` applies it ahead of every
#: event on the bar for the same reason splits come first here.
EVENT_APPLICATION_ORDER: tuple[str, ...] = (
    "split",
    "cash_distribution",
    "spinoff",
    "terminal_payout",
    "share_cancellation",
    "trading_cessation",
    "otc_continuation",
    "delisting_notice",
    "halt",
    "ticker_change",
    "membership_add",
    "membership_remove",
)

_ORDER_INDEX = {name: position for position, name in enumerate(EVENT_APPLICATION_ORDER)}


@dataclass(frozen=True)
class AppliedEvent:
    """One event, applied, with what it moved (§3.6; the adversarial-session record)."""

    sid: str
    event_type: str
    effective_date: str
    cash_delta: float
    equity_delta: float
    detail: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    revision: int | None = None

    def to_row(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row["detail"] = dict(self.detail)
        return row


@dataclass(frozen=True)
class EventOutcome:
    """The state after a bar's events, and the events that produced it."""

    state: PortfolioState
    applied: tuple[AppliedEvent, ...] = ()

    def frame(self) -> pd.DataFrame:
        columns = ["sid", "event_type", "effective_date", "cash_delta", "equity_delta", "detail", "source", "revision"]
        return pd.DataFrame([e.to_row() for e in self.applied], columns=columns)


@dataclass(frozen=True)
class CloseMark:
    """The state after a close mark, with the bar's accrual and what went stale."""

    state: PortfolioState
    accrual: CostBreakdown
    settled_cash: float = 0.0
    stale: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionBatch:
    """Sized intents and the arithmetic that produced them (§3.6 step 2).

    ``quantities`` carries one row per name considered, whether or not it produced an
    intent, so a name that was dropped for notional or rounded to nothing is visible in
    the record rather than absent from it.
    """

    intents: tuple[OrderIntent, ...]
    quantities: pd.DataFrame
    dropped: tuple[dict[str, Any], ...] = ()
    equity: float = 0.0
    decision_bar: str = ""


@dataclass(frozen=True)
class FillOutcome:
    """The state after applying fills, the fills applied, and the duplicates ignored."""

    state: PortfolioState
    applied: tuple[Fill, ...] = ()
    duplicates: tuple[str, ...] = ()
    completed: tuple[OrderRecord, ...] = ()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _events_of(events: Sequence[Event] | pd.DataFrame | None) -> list[Event]:
    if events is None:
        return []
    if isinstance(events, pd.DataFrame):
        if events.empty:
            return []
        return [Event.from_row(row) for row in events.to_dict("records")]
    return [e if isinstance(e, Event) else Event.from_row(e) for e in events]


def _adjustments_of(adjustments: Sequence[Adjustment] | pd.DataFrame | None) -> list[Adjustment]:
    if adjustments is None:
        return []
    if isinstance(adjustments, pd.DataFrame):
        if adjustments.empty:
            return []
        return [Adjustment.from_row(row) for row in adjustments.to_dict("records")]
    return [a if isinstance(a, Adjustment) else Adjustment.from_row(a) for a in adjustments]


def _rebases_on(adjustments: Sequence[Adjustment], bar: str) -> list[Adjustment]:
    """The ``rebase`` rows effective on ``bar``, in a deterministic order.

    Only ``rebase``. The ratio kinds (``split``, ``reverse_split``,
    ``stock_distribution``) reach the book through the ``events`` table, and the fixtures
    of §3.2 write both rows for one split; applying the adjustment as well would rebase the
    position twice. ``rebase`` has no event counterpart -- it *is* the vendor back-rewrite
    -- which is why it is the one kind read from this table.
    """
    return sorted(
        (a for a in adjustments if a.kind == "rebase" and a.effective_date == bar),
        key=lambda a: (a.sid, a.revision),
    )


def _sorted_for_application(events: Iterable[Event]) -> list[Event]:
    return sorted(
        events,
        key=lambda e: (_ORDER_INDEX.get(e.event_type, len(_ORDER_INDEX)), e.sid, e.revision),
    )


def _price_series(prices: Mapping[str, float] | pd.Series | None) -> pd.Series | None:
    if prices is None:
        return None
    if isinstance(prices, pd.Series):
        return prices
    return pd.Series(dict(prices), dtype=float)


def round_to_increment(quantity: float, increment: float) -> float:
    """Round a **trade delta** to the increment, ties to even (§3.7).

    The delta is rounded, not the target: that is what ``float(round(delta))`` does at
    ``src/prism/live/loop.py:142`` and parity with it is why the rule is stated rather than
    left to the phrase "at the increment".

    ``increment = 0.0`` declares a *continuous* quantity grid and returns the delta
    unrounded. No venue offers one; it is the setting
    ``portfolio/rounding.py::RoundingSettings.for_reproduction`` already forces
    (``quantity_increment_default = 0.0``), so that the canonical engine can be run
    against the ``historical_constant_weight`` book without lot rounding standing between
    the two. A negative increment is still a defect and still raises.
    """
    if increment < 0.0:
        raise AccountingError(f"quantity increment must be >= 0, got {increment}")
    if increment == 0.0:
        return float(quantity)
    steps = quantity / increment
    # np.round is half-to-even, which is the declared tie rule.
    return float(np.round(steps)) * increment


def _increment_for(increment: float | Mapping[str, float], sid: str) -> float:
    if isinstance(increment, Mapping):
        if sid not in increment:
            raise AccountingError(f"{sid}: no quantity increment declared for this name (§3.7)")
        return float(increment[sid])
    return float(increment)


def _first_bar_of_new_month(previous: str, bar: str) -> bool:
    if not previous:
        return False
    before = dt.date.fromisoformat(bar_key(previous))
    now = dt.date.fromisoformat(bar_key(bar))
    return (now.year, now.month) > (before.year, before.month)


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


def apply_events(
    state: PortfolioState,
    *,
    bar: str,
    events: Sequence[Event] | pd.DataFrame | None = (),
    adjustments: Sequence[Adjustment] | pd.DataFrame | None = (),
    price_basis: str | Mapping[str, str] = "split_adjusted_at_ingest",
    prices: Mapping[str, float] | pd.Series | None = None,
    settle_due: bool = True,
) -> EventOutcome:
    """Apply every event effective on ``bar``, and settle every claim due on it (§3.6).

    Called at the **start** of the bar, before that bar's fills; the module docstring says
    why. ``events`` is whatever the caller can see -- pass a
    :meth:`prism.io.dataset.DatasetView.events` frame and the application is as of that
    decision, so a payout announced later cannot resolve a claim the decision could not
    have known was resolved.

    ``adjustments`` is the §3.2 companion table, and the one kind read from it is
    ``rebase``: the vendor back-rewrite that restates a whole series. §3.6 states the rule
    -- "a vendor back-rewrite that arrives as a full-series revision is recorded as an
    adjustment event that rebases shares in the same transaction, so equity is continuous
    across the rewrite" -- and without this argument there is no path from that row to the
    share count, so a holding across a 4-for-1 rewrite silently loses three quarters of its
    value at the next mark and realises the loss in cash at the next fill. The ratio kinds
    are deliberately *not* read here; they arrive as ``split`` events.

    ``price_basis`` is read from the bar series and never inferred (decision D3): under
    ``split_adjusted_at_ingest`` a split applies **no** share change, because the vendor
    already restated the prices, and applying both would double-count the ratio. It is a
    per-series fact, so a mapping of sid to basis is accepted as well as one value for the
    whole book -- a dataset that mixes an as-printed source with an adjusted one is legal,
    and taking one series' basis for another is how a split gets counted twice.

    Called **once per bar**. It is not idempotent and does not try to be: an ex-date
    booking applied twice books the distribution twice. A process that can restart mid-bar
    records which bar it has applied in its durable state (§3.11) rather than relying on
    this function to notice.
    """
    if not isinstance(price_basis, Mapping):
        require_price_basis(price_basis)
    else:
        for basis in price_basis.values():
            require_price_basis(basis)
    key = bar_key(bar)
    working = state.copy()
    applied: list[AppliedEvent] = []
    rows = _events_of(events)

    if settle_due:
        if _first_bar_of_new_month(state.as_of, key):
            # Carried holding-cost accruals under ``holding_cost_settlement = "monthly"``
            # are due on the first bar of the next calendar month, the broker convention.
            # They settle here, at the start of the bar and before anything else moves
            # cash, for the same reason a pay date does: it is a claim whose date arrived.
            moved = _settle_accruals(working)
            if moved:
                applied.append(
                    AppliedEvent(
                        sid="",
                        event_type="holding_cost_settlement",
                        effective_date=key,
                        cash_delta=moved,
                        equity_delta=0.0,
                        detail={"settlement": "monthly"},
                    )
                )
        applied.extend(_settle_due_claims(working, key, rows))

    # Rebases first, before anything reads a share count: entitlement, a split factor and
    # a decided order quantity are all share-denominated, and the rewrite restated the
    # basis they are counted in.
    for adjustment in _rebases_on(_adjustments_of(adjustments), key):
        applied.append(_apply_rebase(working, adjustment, key, price_basis))

    for event in _sorted_for_application(e for e in rows if e.effective_date == key):
        record = _apply_one_event(working, event, key, price_basis, prices)
        if record is not None:
            applied.append(record)

    working.as_of = key
    working.sequence += 1
    return EventOutcome(state=working, applied=tuple(applied))


def _settle_due_claims(state: PortfolioState, bar: str, rows: Sequence[Event]) -> list[AppliedEvent]:
    """Convert receivables and liabilities whose pay date has arrived into cash.

    Equity-neutral by construction: the claim leaves one side of the balance and the same
    number arrives in cash. What it does change is cash, and therefore financing and
    buying power, on every bar between the ex-date and the pay date -- which is the whole
    point of booking on the ex-date and paying on the pay date.
    """
    out: list[AppliedEvent] = []
    for event in _sorted_for_application(rows):
        if event.event_type == "cash_distribution":
            pay_date = event.payload.get("pay_date")
            key = distribution_key(event.sid, event.effective_date)
        elif event.event_type == "terminal_payout":
            pay_date = event.payload.get("pay_date")
            key = terminal_payout_key(event.sid, event.effective_date)
        else:
            continue
        if pay_date is None or bar_key(pay_date) > bar:
            continue
        amount = state.receivables.get(key, 0.0)
        if amount:
            state.receivables.pop(key)
            state.cash += amount
            out.append(
                AppliedEvent(
                    sid=event.sid,
                    event_type=f"{event.event_type}_settlement",
                    effective_date=bar,
                    cash_delta=amount,
                    equity_delta=0.0,
                    detail={"leg": "receivable", "ex_date": event.effective_date, "pay_date": bar_key(pay_date)},
                    source=event.source,
                    revision=event.revision,
                )
            )
        owed = state.liabilities.get(key, 0.0)
        if owed:
            state.liabilities.pop(key)
            state.cash -= owed
            out.append(
                AppliedEvent(
                    sid=event.sid,
                    event_type=f"{event.event_type}_settlement",
                    effective_date=bar,
                    cash_delta=-owed,
                    equity_delta=0.0,
                    detail={"leg": "liability", "ex_date": event.effective_date, "pay_date": bar_key(pay_date)},
                    source=event.source,
                    revision=event.revision,
                )
            )
    return out


def _apply_rebase(
    state: PortfolioState,
    adjustment: Adjustment,
    bar: str,
    price_basis: str | Mapping[str, str],
) -> AppliedEvent:
    """Apply one ``rebase`` adjustment: shares up by the factor, the carried mark down by it.

    Equity is unchanged by construction (``_rebase_position`` divides the mark by the same
    factor it multiplies the shares by), which is the whole content of the rule: the
    restated price series and the restated share count are one transaction, and applying
    only the first is how a book loses ``1 - 1/factor`` of a holding with nothing in the
    tables saying it did.

    Unlike a split, the factor does **not** depend on the declared basis: a back-rewrite
    moved the prices themselves, so shares follow under either basis
    (:func:`prism.io.events.applied_share_factor`, decision D3).
    """
    sid = adjustment.sid
    basis = _basis_for(price_basis, sid)
    factor = applied_share_factor(adjustment.kind, basis, adjustment.factor)
    detail: dict[str, Any] = {
        "kind": adjustment.kind,
        "declared_factor": adjustment.factor,
        "share_factor": factor,
        "basis": basis,
        "applies_from": adjustment.applies_from,
        "canceled_orders": [],
    }
    if factor != 1.0:
        if sid in state.positions:
            _rebase_position(state, sid, factor)
            _rebase_obligations(state, sid, factor)
        # Cancelled whether or not the name is held: an order decided at the pre-rewrite
        # reference price is a quantity in a basis that no longer exists, and it would
        # fill at the restated price for the unrestated size.
        detail["canceled_orders"] = _cancel_open_orders(
            state, sid, bar, f"{adjustment.kind} {factor} restates the share basis"
        )
    return AppliedEvent(
        sid=sid,
        event_type="rebase",
        effective_date=bar,
        cash_delta=0.0,
        equity_delta=0.0,
        detail=detail,
        source=adjustment.source,
        revision=adjustment.revision,
    )


def _basis_for(price_basis: str | Mapping[str, str], sid: str) -> str:
    if isinstance(price_basis, Mapping):
        if sid not in price_basis:
            raise AccountingError(
                f"{sid}: no price_basis declared for this series. The basis decides whether a split moves shares "
                "at all (decision D3) and is never inferred from the data"
            )
        return require_price_basis(price_basis[sid])
    return price_basis


def _apply_one_event(
    state: PortfolioState,
    event: Event,
    bar: str,
    price_basis: str | Mapping[str, str],
    prices: Mapping[str, float] | pd.Series | None,
) -> AppliedEvent | None:
    sid = event.sid
    position = state.positions.get(sid)
    before = _equity_or_none(state, prices)

    def record(cash_delta: float, detail: dict[str, Any]) -> AppliedEvent:
        after = _equity_or_none(state, prices)
        equity_delta = 0.0 if before is None or after is None else after - before
        return AppliedEvent(
            sid=sid,
            event_type=event.event_type,
            effective_date=bar,
            cash_delta=cash_delta,
            equity_delta=equity_delta,
            detail=detail,
            source=event.source,
            revision=event.revision,
        )

    if event.event_type == "split":
        basis = _basis_for(price_basis, sid)
        factor = applied_share_factor("split", basis, float(event.payload["factor"]))
        detail: dict[str, Any] = {
            "declared_factor": float(event.payload["factor"]),
            "share_factor": factor,
            "basis": basis,
            "canceled_orders": [],
        }
        if factor == 1.0:
            return record(0.0, detail)
        if position is not None:
            _rebase_position(state, sid, factor)
            _rebase_obligations(state, sid, factor)
        # Outside the ``position is not None`` guard on purpose. A resting order decided at
        # the pre-split reference price carries a pre-split quantity; on a ``raw`` series
        # the next open prints post-split, so leaving it alive fills the old share count at
        # the new price and acquires ``1 / factor`` of the decided notional. That is a
        # silent mis-size, and it does not become correct because the book happened to be
        # flat in the name when the split landed.
        detail["canceled_orders"] = _cancel_open_orders(state, sid, bar, f"split {factor} rebases the decided size")
        return record(0.0, detail)

    if event.event_type == "cash_distribution":
        amount_per_share = float(event.payload["amount_per_share"])
        shares = 0.0 if position is None else position.shares
        amount = shares * amount_per_share
        pay_date = event.payload.get("pay_date")
        detail = {
            "amount_per_share": amount_per_share,
            "shares": shares,
            "amount": amount,
            "pay_date": None if pay_date is None else bar_key(pay_date),
            "pay_date_assumed": pay_date is None,
        }
        if amount == 0.0:
            detail["leg"] = "none"
            return record(0.0, detail)
        key = distribution_key(sid, bar)
        if pay_date is None:
            # No pay date is known, so the entry settles on the ex-date and the record
            # says ``pay_date_assumed``; it is never dropped for want of a date.
            state.cash += amount
            detail["leg"] = "cash"
            return record(amount, detail)
        if amount > 0.0:
            state.add_receivable(key, amount)
            detail["leg"] = "receivable"
        else:
            state.add_liability(key, -amount)
            detail["leg"] = "liability"
        return record(0.0, detail)

    if event.event_type == "terminal_payout":
        return _apply_terminal_payout(state, event, bar, record)

    if event.event_type == "share_cancellation":
        fraction = float(event.payload["fraction_cancelled"])
        if position is None:
            return record(0.0, {"fraction_cancelled": fraction, "shares": 0.0})
        cancelled = position.shares * fraction
        detail = {
            "fraction_cancelled": fraction,
            "shares_cancelled": cancelled,
            "proceeds": 0.0,
            "realized": -cancelled * _carried_mark(position, sid, prices),
            "canceled_orders": [],
        }
        state.set_shares(sid, position.shares - cancelled)
        if fraction >= 1.0:
            detail["canceled_orders"] = _cancel_open_orders(
                state, sid, bar, "the shares were cancelled; the order cannot fill"
            )
        return record(0.0, detail)

    if event.event_type == "trading_cessation":
        if position is not None:
            # The instrument stopped printing and no terminal value is known yet: a strict
            # claim of unknown value. It marks at its last bar and every record built on
            # the state is written ``incomplete`` (§3.2).
            position.valuation_status = "unresolved_claim"
        return record(
            0.0,
            {
                "last_session": event.payload.get("last_session"),
                "canceled_orders": _cancel_open_orders(state, sid, bar, "trading ceased; the order cannot fill"),
            },
        )

    if event.event_type == "halt":
        if position is not None and position.valuation_status == "priced":
            position.valuation_status = "stale"
        return record(0.0, {"reason": event.payload.get("reason"), "end": event.payload.get("end")})

    if event.event_type == "otc_continuation":
        # The series continues under another venue and vendor symbol; the claim is not
        # terminal. Nothing moves here, and treating it as terminal would write off a live
        # position.
        if position is not None and position.valuation_status == "unresolved_claim":
            position.valuation_status = "stale"
        return record(0.0, {"venue": event.payload.get("venue"), "vendor_symbol": event.payload.get("vendor_symbol")})

    # delisting_notice, ticker_change, spinoff, membership_*: recorded as seen, with no
    # ledger effect in this engine. A spinoff that distributes a child security is a
    # position-creating event and is not implemented; it is refused rather than silently
    # dropped, because dropping it would lose the child's value.
    if event.event_type == "spinoff":
        if position is None:
            return record(0.0, {"child_sid": event.payload.get("child_sid"), "note": "not held; no ledger effect"})
        raise AccountingError(
            f"{sid}: spinoff on {bar} distributes {event.payload.get('ratio')} shares of "
            f"{event.payload.get('child_sid')!r} per share held, and the canonical engine has no declared "
            "valuation for the child at the moment of distribution. Refusing rather than dropping it: the parent "
            "price marks down for the child on the ex-date, so an ignored spinoff loses the child's value "
            "silently. A spinoff on a name that is not held is a recorded no-op"
        )
    return record(0.0, {"note": "recorded, no ledger effect"})


def _apply_terminal_payout(state: PortfolioState, event: Event, bar: str, record) -> AppliedEvent:
    sid = event.sid
    position = state.positions.get(sid)
    if str(event.payload.get("status", "")) == "unknown":
        if position is not None:
            position.valuation_status = "unresolved_claim"
        return record(
            0.0,
            {
                "status": "unknown",
                "canceled_orders": _cancel_open_orders(
                    state, sid, bar, "the terminal claim is unresolved; the order cannot fill"
                ),
            },
        )
    amount_per_share = float(event.payload["amount_per_share"])
    pay_date = event.payload.get("pay_date")
    shares = 0.0 if position is None else position.shares
    proceeds = shares * amount_per_share
    detail = {
        "status": "resolved",
        "amount_per_share": amount_per_share,
        "shares": shares,
        "proceeds": proceeds,
        "pay_date": None if pay_date is None else bar_key(pay_date),
        "pay_date_assumed": pay_date is None,
    }
    detail["canceled_orders"] = _cancel_open_orders(
        state, sid, bar, "the terminal payout closed the instrument; the order cannot fill"
    )
    if position is None:
        return record(0.0, detail)
    state.positions.pop(sid, None)
    if pay_date is None or bar_key(pay_date) <= bar:
        state.cash += proceeds
        detail["leg"] = "cash"
        return record(proceeds, detail)
    key = terminal_payout_key(sid, bar)
    if proceeds >= 0.0:
        state.add_receivable(key, proceeds)
        detail["leg"] = "receivable"
    else:
        state.add_liability(key, -proceeds)
        detail["leg"] = "liability"
    return record(0.0, detail)


def _carried_mark(position: Position, sid: str, prices: Mapping[str, float] | pd.Series | None) -> float:
    """The mark to write a realized amount against, with no quotes needed.

    An event is applied at the start of a bar that has printed nothing yet, so the mark it
    realizes against is the one the position already carries. A holding with no mark at all
    still raises: there is no number to realize.
    """
    if prices is not None:
        return position.mark(prices, sid)
    if position.last_price is None:
        raise AccountingError(f"{sid}: position has never been marked, so no amount can be realized against it")
    return position.last_price


def _equity_or_none(state: PortfolioState, prices: Mapping[str, float] | pd.Series | None) -> float | None:
    """Equity, or ``None`` when the book cannot be valued at all.

    Used only to annotate an applied event with the equity it moved. With no quotes the
    book is valued at the marks it already carries, which is what "equity immediately
    before this event" means at the start of a bar that has printed nothing yet.
    """
    try:
        return state.carried_equity() if prices is None else state.equity(prices)
    except AccountingError:
        return None


def _rebase_position(state: PortfolioState, sid: str, factor: float) -> None:
    position = state.positions[sid]
    position.shares *= factor
    if position.last_price is not None:
        # share_factor * price_factor == 1 by construction: the rebase moves no equity.
        position.last_price /= factor


def _rebase_obligations(state: PortfolioState, sid: str, factor: float) -> None:
    """Split-adjust obligation quantities (§3.7): 50 shares to cover become 100 on a 2-for-1."""
    for obligation in state.obligations:
        if getattr(obligation, "sid", None) != sid:
            continue
        for name in ("target_qty", "remaining_qty"):
            if hasattr(obligation, name):
                try:
                    setattr(obligation, name, getattr(obligation, name) * factor)
                except (AttributeError, dataclasses.FrozenInstanceError):  # pragma: no cover - frozen obligation
                    raise AccountingError(
                        f"{sid}: obligation quantities are split-adjusted on the effective date (§3.7) and this "
                        "obligation object refuses the assignment"
                    ) from None


def _cancel_open_orders(state: PortfolioState, sid: str, bar: str, reason: str) -> list[str]:
    """Cancel open orders on a name whose share basis just changed.

    An order decided at a pre-split reference price is a stale quantity at a stale price,
    and rounding a rebased quantity back to the increment would change the decided
    notional silently. Cancelling is the honest response and matches the venue, which
    cancels resting orders across a corporate action; the next decision re-sizes on the
    new basis.
    """
    cancelled: list[str] = []
    for order in state.pending_orders:
        if order.sid != sid or not order.is_open:
            continue
        order.status = "canceled"
        order.reason = reason
        order.last_update = bar
        cancelled.append(order.client_order_id)
    state.pending_orders = [o for o in state.pending_orders if o.client_order_id not in set(cancelled)]
    return cancelled


# ---------------------------------------------------------------------------
# marking and accrual
# ---------------------------------------------------------------------------


def mark_prices(
    state: PortfolioState,
    *,
    bar: str,
    prices: Mapping[str, float] | pd.Series | None,
    stale_on_missing: bool = True,
) -> tuple[PortfolioState, tuple[str, ...]]:
    """Re-mark every position at ``bar``'s prices; return the state and what went stale.

    ``stale_on_missing`` is ``True`` at a close, where an absent print means the session
    ended without one, and ``False`` at an open, where the session is not over and an
    absent open says nothing yet. Either way the holding keeps its last mark: a missing bar
    never erases a holding and never simulates a sale (SPEC N7).

    An ``unresolved_claim`` is never downgraded to ``stale`` by a missing price: the claim
    is the stronger statement about why the name has no mark.
    """
    key = bar_key(bar)
    working = state.copy()
    series = _price_series(prices)
    stale: list[str] = []
    for sid, position in working.positions.items():
        price = None
        if series is not None and sid in series.index:
            value = float(series.loc[sid])
            if math.isfinite(value) and value > 0.0:
                price = value
        if price is not None:
            position.last_price = price
            position.last_price_bar = key
            if position.valuation_status == "stale":
                position.valuation_status = "priced"
        elif position.valuation_status == "priced" and stale_on_missing:
            if position.last_price is None:
                raise AccountingError(
                    f"{sid}: no price at {key} and no prior mark, so the holding cannot be carried. "
                    "It is never marked at zero (SPEC N7)"
                )
            position.valuation_status = "stale"
            stale.append(sid)
        elif position.valuation_status in ("stale", "unresolved_claim"):
            stale.append(sid)
    working.as_of = key
    working.sequence += 1
    return working, tuple(sorted(stale))


def mark_at_close(
    state: PortfolioState,
    *,
    bar: str,
    prices: Mapping[str, float] | pd.Series | None,
    cost_model: CostModel | None = None,
) -> CloseMark:
    """Transition 1: mark at the close of ``bar`` and accrue that bar's holding cost (§3.6).

    Holding costs accrue on the **marked** state, so the prices are applied first: borrow
    is charged on short market value as marked, not on entry notional.

    Settlement follows the stack's ``holding_cost_settlement``. ``per_bar`` debits cash at
    this same close and leaves no liability. ``monthly`` carries the accrual in
    ``liabilities`` (and cash income in ``receivables``); :func:`apply_events` settles that
    carry to cash on the first bar of the next calendar month, because by the time this
    transition runs the bar has begun and the month boundary has already been crossed.
    Equity is the same at every close under either value; ``cash``, and therefore
    ``buying_power``, is not.
    """
    working, stale = mark_prices(state, bar=bar, prices=prices, stale_on_missing=True)
    settled = 0.0
    settlement = "per_bar"
    if cost_model is not None and cost_model.stack is not None:
        settlement = cost_model.stack.holding_cost_settlement

    if cost_model is None or cost_model.stack is None:
        return CloseMark(state=working, accrual=CostBreakdown(), settled_cash=settled, stale=stale)

    accrual = cost_model.accrue_holding(working, prices)
    if settlement == "per_bar":
        working.cash -= accrual.holding_total
        settled += -accrual.holding_total
    else:
        if accrual.borrow:
            working.add_liability(BORROW_ACCRUAL_KEY, accrual.borrow)
        if accrual.financing:
            working.add_liability(FINANCING_ACCRUAL_KEY, accrual.financing)
        if accrual.cash_income:
            working.add_receivable(CASH_INCOME_ACCRUAL_KEY, accrual.cash_income)
    working.sequence += 1
    return CloseMark(state=working, accrual=accrual, settled_cash=settled, stale=stale)


def _settle_accruals(state: PortfolioState) -> float:
    """Debit carried borrow and financing, credit carried income; return the cash moved."""
    moved = 0.0
    for key in (BORROW_ACCRUAL_KEY, FINANCING_ACCRUAL_KEY):
        owed = state.liabilities.pop(key, 0.0)
        if owed:
            state.cash -= owed
            moved -= owed
    earned = state.receivables.pop(CASH_INCOME_ACCRUAL_KEY, 0.0)
    if earned:
        state.cash += earned
        moved += earned
    return moved


# ---------------------------------------------------------------------------
# decide and submit
# ---------------------------------------------------------------------------


def decide(
    state: PortfolioState,
    targets: pd.Series | Mapping[str, float],
    prices: pd.Series | Mapping[str, float],
    *,
    decision_bar: str,
    increment: float | Mapping[str, float] = 1.0,
    min_notional: float = 0.0,
    order_type: Literal["market_on_open", "market_day"] = "market_on_open",
    action_class: Literal["C0", "C1", "C2", "C3", "C4", "C5", "OB"] = "C1",
    zero_crossing_clamp: bool = False,
    equity: float | None = None,
) -> DecisionBatch:
    """Transition 2: targets become intents in shares, sized at the decision close (§3.6).

    ``targets`` are fractions of current marked equity; ``NaN`` means "no decision for that
    name", which holds the current position rather than selling it. The **trade delta** is
    rounded to the increment with ties to even (§3.7), never the target, and a rounded
    order below ``min_notional`` is dropped and recorded as a drop.

    ``zero_crossing_clamp`` clamps an order that would flip a held position through zero to
    flat for this bar, because a venue rejects a side-crossing single order; the opposite
    side opens next bar from a flat book. The ``historical_constant_weight`` reproduction
    engine forces it off; the canonical engine takes the profile's value.

    This is the sizing half of the decision. Whole-batch validation, repair, scaling, and
    the dust rule live in ``portfolio/validate.py`` and ``portfolio/rounding.py`` (§3.7)
    and run on the batch this returns.
    """
    key = bar_key(decision_bar)
    price_series = _price_series(prices)
    if price_series is None:
        raise AccountingError("a decision needs reference prices at the decision close")
    target_series = targets if isinstance(targets, pd.Series) else pd.Series(dict(targets), dtype=float)
    marked_equity = state.equity(price_series) if equity is None else float(equity)
    if not math.isfinite(marked_equity) or marked_equity <= 0.0:
        raise AccountingError(f"cannot size a decision at equity {marked_equity!r}")

    outstanding: dict[str, float] = {}
    for order in state.open_orders():
        outstanding[order.sid] = outstanding.get(order.sid, 0.0) + order.remaining_qty

    rows: list[dict[str, Any]] = []
    intents: list[OrderIntent] = []
    dropped: list[dict[str, Any]] = []
    for sid in [str(s) for s in target_series.index]:
        target = float(target_series.loc[sid])
        # The baseline is the position the book already implies: shares held plus the
        # unfilled quantity of every order still outstanding on the name. Sizing against
        # held shares alone would re-order a trade that is already at the venue, which is
        # why §3.7 validates "the portfolio the batch implies together with everything
        # already outstanding" rather than the held book.
        held = state.shares(sid) + outstanding.get(sid, 0.0)
        step = _increment_for(increment, sid)
        row: dict[str, Any] = {
            "sid": sid,
            "target_weight": target,
            "held_shares": state.shares(sid),
            "outstanding_shares": outstanding.get(sid, 0.0),
            "increment": step,
            "reference_price": np.nan,
            "target_shares": np.nan,
            "delta": np.nan,
            "rounded_delta": 0.0,
            "residual": 0.0,
            "intent": False,
            "reason": None,
        }
        if not math.isfinite(target):
            row["reason"] = "no decision (NaN target holds the position)"
            rows.append(row)
            continue
        price = float(price_series.loc[sid]) if sid in price_series.index else float("nan")
        if not math.isfinite(price) or price <= 0.0:
            # No reference price, no size: the participation restriction of §3.7 refuses a
            # trade on a name the decision cannot price, and it holds its prior position.
            row["reason"] = "no reference price at the decision close"
            rows.append(row)
            dropped.append({"sid": sid, "intended_notional": float("nan"), "rule": "no_reference_price"})
            continue
        target_shares = target * marked_equity / price
        delta = target_shares - held
        rounded = round_to_increment(delta, step)
        row.update(
            {
                "reference_price": price,
                "target_shares": target_shares,
                "delta": delta,
                "rounded_delta": rounded,
                "residual": delta - rounded,
            }
        )
        if zero_crossing_clamp and held != 0.0 and (held + rounded) * held < 0.0:
            clamped = round_to_increment(-held, step)
            dropped.append({"sid": sid, "intended_notional": abs(delta * price), "rule": "zero_crossing_clamp"})
            row["reason"] = "clamped to flat: a single order may not cross zero"
            rounded = clamped
            row["rounded_delta"] = rounded
        if rounded == 0.0:
            row["reason"] = row["reason"] or "rounds to no trade"
            rows.append(row)
            continue
        notional = abs(rounded * price)
        if notional < min_notional:
            dropped.append({"sid": sid, "intended_notional": abs(delta * price), "rule": "min_notional"})
            row["reason"] = f"below min_notional {min_notional}"
            row["rounded_delta"] = 0.0
            rows.append(row)
            continue
        intents.append(
            OrderIntent(
                sid=sid,
                qty=rounded,
                increment=step,
                reference_price=price,
                decision_bar=key,
                order_type=order_type,
                time_in_force="opg" if order_type == "market_on_open" else "day",
                action_class=action_class,
            )
        )
        row["intent"] = True
        rows.append(row)

    columns = [
        "sid",
        "target_weight",
        "held_shares",
        "outstanding_shares",
        "increment",
        "reference_price",
        "target_shares",
        "delta",
        "rounded_delta",
        "residual",
        "intent",
        "reason",
    ]
    return DecisionBatch(
        intents=tuple(intents),
        quantities=pd.DataFrame(rows, columns=columns),
        dropped=tuple(dropped),
        equity=marked_equity,
        decision_bar=key,
    )


def submit(
    state: PortfolioState,
    intents: Sequence[OrderIntent],
    *,
    submitted_at: str | None = None,
    prefix: str = "",
) -> PortfolioState:
    """Transition 3: intents become pending orders with stable client order ids (§3.6).

    The id is ``"{prefix}{decision_bar}:{sid}"``, the convention ``live/loop.py:18``
    already documents, so the same decision re-submitted after a crash collides with its
    own id at the venue rather than doubling the position. A collision inside one state is
    a defect and raises here.
    """
    working = state.copy()
    stamp = bar_key(submitted_at) if submitted_at else working.as_of
    existing = {order.client_order_id for order in working.pending_orders}
    for intent in intents:
        client_order_id = f"{prefix}{intent.decision_bar}:{intent.sid}"
        if client_order_id in existing:
            raise AccountingError(
                f"{client_order_id}: an order with this client id is already outstanding. Two live orders under "
                "one id cannot be told apart at reconciliation; give the replacement batch a suffixed id (§3.7)"
            )
        existing.add(client_order_id)
        working.pending_orders.append(
            OrderRecord(
                intent=intent,
                client_order_id=client_order_id,
                status="pending",
                filled_qty=0.0,
                fills=[],
                submitted_at=stamp,
                last_update=stamp,
            )
        )
    working.sequence += 1
    return working


# ---------------------------------------------------------------------------
# fills and expiry
# ---------------------------------------------------------------------------


def apply_fills(
    state: PortfolioState,
    fills: Sequence[Fill],
    *,
    known_executions: Iterable[str] | None = None,
) -> FillOutcome:
    """Transition 4: apply fills to cash, positions, and the orders that produced them.

    ``cash`` moves by ``-qty * fill_price - commission - spread_dollars - impact_dollars``
    and the position moves by the filled quantity, so equity changes by exactly the charged
    cost when the fill is marked at its own price. That is ``rebalance_conserves``
    (``formal/PrismFormal/Ledger.lean:40-43``) under the projection of §3.6.

    Three refusals, each of which is a real failure mode rather than a hypothetical:

    * a fill with no submitted order raises -- "no fill exists without a submitted order";
    * a fill that would take an order past its decided quantity raises;
    * a fill whose ``execution_id`` has already been applied is **ignored**, not applied
      twice.

    ``known_executions`` is the second half of that last rule and the reason it is a
    parameter rather than a field on the state. An order leaves ``pending_orders`` the
    moment it fills completely, so the state itself stops being able to recognize a replay
    of that order's own fill; the durable fills ledger (§3.11) is what remembers, and it
    passes its execution ids in here. Without it a re-read page of broker fills would raise
    "no such order" on exactly the fills that already settled correctly.
    """
    working = state.copy()
    applied: list[Fill] = []
    duplicates: list[str] = []
    completed: list[OrderRecord] = []
    settled = set() if known_executions is None else {str(e) for e in known_executions}

    for fill in fills:
        if fill.execution_id in settled:
            duplicates.append(fill.execution_id)
            continue
        order = working.order(fill.client_order_id)
        if order is None:
            raise AccountingError(
                f"{fill.client_order_id}: a fill arrived for an order this state does not carry. A fill without a "
                "submitted order is never applied (§3.6); reconcile the order set first"
            )
        if order.has_execution(fill.execution_id):
            duplicates.append(fill.execution_id)
            continue
        settled.add(fill.execution_id)
        if fill.sid != order.sid:
            raise AccountingError(f"{fill.client_order_id}: fill on {fill.sid} against an order on {order.sid}")
        remaining = order.remaining_qty
        if remaining == 0.0 or fill.qty * order.intent.qty < 0.0 or abs(fill.qty) > abs(remaining) * (1.0 + 1e-9):
            raise AccountingError(
                f"{fill.client_order_id}: fill of {fill.qty} against a remaining quantity of {remaining}. An "
                "over-fill is an accounting defect, never absorbed silently"
            )
        working.cash += fill.cash_delta
        working.set_shares(fill.sid, working.shares(fill.sid) + fill.qty, price=fill.fill_price, bar=fill.bar)
        position = working.positions.get(fill.sid)
        if position is not None:
            position.last_price = fill.fill_price
            position.last_price_bar = fill.bar
            if position.valuation_status == "stale":
                position.valuation_status = "priced"
        order.fills.append(fill)
        order.filled_qty += fill.qty
        order.last_update = fill.ts
        order.status = "filled" if order.remaining_qty == 0.0 else "partial"
        applied.append(fill)
        if order.status == "filled":
            completed.append(order)

    finished = {order.client_order_id for order in completed}
    working.pending_orders = [o for o in working.pending_orders if o.client_order_id not in finished]
    working.sequence += 1
    return FillOutcome(state=working, applied=tuple(applied), duplicates=tuple(duplicates), completed=tuple(completed))


def expire_orders(
    state: PortfolioState,
    *,
    bar: str,
    phase: Literal["open", "close"],
    reason: str | None = None,
    sids: Iterable[str] | None = None,
) -> tuple[PortfolioState, tuple[OrderRecord, ...]]:
    """Expire the remainders whose time in force has run out (§3.6, §3.9).

    ``opg`` can fill only at the next open, so its remainder expires at that open;
    ``day`` expires at that session's close. An expired remainder changes cash and
    positions not at all -- it only stops the order from filling later.

    ``sids`` restricts expiry to the names that actually traded on ``bar``. A name that did
    not trade held no auction and no session, so its order did not miss anything: "a bar
    with no price for a name leaves its order pending" (§3.6).
    """
    working = state.copy()
    horizon = bar_key(bar)
    wanted = "opg" if phase == "open" else "day"
    traded = None if sids is None else {str(s) for s in sids}
    expired: list[OrderRecord] = []
    for order in working.pending_orders:
        if not order.is_open or order.intent.time_in_force != wanted:
            continue
        if traded is not None and order.sid not in traded:
            continue
        if bar_key(order.submitted_at or order.intent.decision_bar) >= horizon:
            continue  # not yet eligible: submitted at or after this bar's decision
        order.status = "expired"
        order.reason = reason or f"{order.intent.time_in_force} remainder expired at the {phase} of {horizon}"
        order.last_update = horizon
        expired.append(order)
    gone = {order.client_order_id for order in expired}
    working.pending_orders = [o for o in working.pending_orders if o.client_order_id not in gone]
    working.sequence += 1
    return working, tuple(expired)
