"""Reconciliation: what the venue holds versus what the book says (spec §3.11).

The module's whole job is one distinction, because it decides what may gate a
submission:

* **Accounting comparisons gate.** Positions and cash, open orders against the
  pending set, and executions since the fills marker against the fills ledger.
  These are statements about the book, and a book that disagrees with the venue
  cannot be traded from.
* **Telemetry comparisons gate nothing.** Modeled against observed execution
  cost, realized against evaluated turnover, realized variance against the
  budget. A miss on any of them is recorded and changes no operating state. A
  cost model that was optimistic is a calibration finding, not a reason to stop
  trading, and wiring it to a halt would let an estimate veto a decision.

Within the accounting comparisons there is a second distinction the outcome
turns on. A difference in *positions or cash* is adopted: broker truth wins,
the difference is recorded with the reason, and the cycle continues (§3.11,
state S7 with the adopt-and-continue outcome). That is deliberate — a routine
partial fill must not stop a book from rebalancing. A difference in *orders* is
not adoptable: a pending order the venue reports at a different quantity or on
the other side is not explainable by a fill, it is two systems disagreeing about
what was submitted, and it holds the engine at reconcile-only until a human
looks. The same is true of an execution the fills ledger does not carry.

An order whose fate the venue will not state — not open, not terminal, no
executions — is neither clean nor a discrepancy: it is ``unknown_order_state``
(S3). "No open orders" and "cannot list open orders" are different facts, and
only one of them makes it safe to act.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from prism.execution.portfolio_state import PortfolioState, Position
from prism.live.broker import Broker, BrokerCapabilityUnsupported

logger = logging.getLogger(__name__)

__all__ = [
    "ACCOUNTING",
    "STATUS_CLEAN",
    "STATUS_DISCREPANCY",
    "STATUS_UNKNOWN_ORDER_STATE",
    "TELEMETRY",
    "Reconciliation",
    "ReconcileItem",
    "ReconcileSettings",
    "reconcile",
    "reconcile_fills",
    "reconcile_loop_state",
    "reconcile_orders",
    "reconcile_positions",
    "telemetry_reads",
]

ACCOUNTING = "accounting"
TELEMETRY = "telemetry"

STATUS_CLEAN = "clean"
STATUS_DISCREPANCY = "discrepancy"
STATUS_UNKNOWN_ORDER_STATE = "unknown_order_state"

#: Terminal venue statuses: an order in one of these has a fate, whatever it is.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day"}
)


@dataclass(frozen=True)
class ReconcileSettings:
    """The declared tolerances every comparison is made at.

    ``share_tolerance`` is float noise on a share count, not a position policy:
    anything above it is a real divergence and is reported even though broker
    truth is adopted either way. ``cost_tolerance_bps`` is measured on traded
    notional, which is the only denominator that makes a cost miss comparable
    across books.
    """

    share_tolerance: float = 1e-6
    cash_tolerance: float = 0.01
    cost_tolerance_bps: float = 25.0
    turnover_tolerance: float = 0.10
    variance_tolerance: float = 0.10


@dataclass(frozen=True)
class ReconcileItem:
    """One comparison that did not agree.

    ``cls`` is ``accounting`` or ``telemetry`` and is what a caller reads to
    decide whether the item may gate. ``adopted`` records that broker truth was
    taken for this item; ``blocking`` records that it was not adoptable.
    """

    check: str
    cls: str
    detail: str
    sid: str | None = None
    client_order_id: str | None = None
    engine: float | None = None
    observed: float | None = None
    difference: float | None = None
    adopted: bool = False
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "class": self.cls,
            "detail": self.detail,
            "sid": self.sid,
            "client_order_id": self.client_order_id,
            "engine": self.engine,
            "observed": self.observed,
            "difference": self.difference,
            "adopted": self.adopted,
            "blocking": self.blocking,
        }


@dataclass(frozen=True)
class Reconciliation:
    """The result of one reconciliation pass (§3.11)."""

    status: str
    items: tuple[ReconcileItem, ...] = ()
    state: PortfolioState | None = None
    settings: ReconcileSettings = field(default_factory=ReconcileSettings)

    @property
    def clean(self) -> bool:
        return self.status == STATUS_CLEAN

    @property
    def accounting_items(self) -> tuple[ReconcileItem, ...]:
        return tuple(item for item in self.items if item.cls == ACCOUNTING)

    @property
    def telemetry_items(self) -> tuple[ReconcileItem, ...]:
        return tuple(item for item in self.items if item.cls == TELEMETRY)

    @property
    def blocking_items(self) -> tuple[ReconcileItem, ...]:
        """The items that hold the engine at reconcile-only. Telemetry never appears here."""
        return tuple(item for item in self.items if item.blocking)

    @property
    def may_trade(self) -> bool:
        """Whether this pass leaves the engine able to decide and submit.

        An adopted position or cash difference does not remove that ability: it
        is the routine outcome of a partial fill, and refusing to rebalance
        because of one is how a book drifts for a month.
        """
        return self.status != STATUS_UNKNOWN_ORDER_STATE and not self.blocking_items

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "may_trade": self.may_trade,
            "items": [item.to_dict() for item in self.items],
            "tolerances": {
                "share_tolerance": self.settings.share_tolerance,
                "cash_tolerance": self.settings.cash_tolerance,
                "cost_tolerance_bps": self.settings.cost_tolerance_bps,
                "turnover_tolerance": self.settings.turnover_tolerance,
                "variance_tolerance": self.settings.variance_tolerance,
            },
        }


# ---------------------------------------------------------------------------
# views over whatever shape the caller has
# ---------------------------------------------------------------------------


def _field(row: Any, *names: str, default: Any = None) -> Any:
    """One field off a dataclass, an object, or a mapping — brokers hand back all three."""
    for name in names:
        if isinstance(row, Mapping):
            if name in row:
                return row[name]
        elif hasattr(row, name):
            return getattr(row, name)
    return default


def _order_view(row: Any) -> tuple[str, str, float, float, str]:
    """``(client_order_id, sid, qty, filled_qty, status)`` from a venue order report."""
    client_order_id = _field(row, "client_order_id", "id")
    if client_order_id is None:
        raise ValueError(f"venue order report {row!r} carries no client_order_id; it cannot be reconciled")
    sid = _field(row, "sid", "symbol", default="")
    qty = float(_field(row, "qty", "quantity", default=0.0) or 0.0)
    filled = float(_field(row, "filled_qty", "filled_quantity", default=0.0) or 0.0)
    status = str(_field(row, "status", default="") or "")
    return str(client_order_id), str(sid), qty, filled, status


def _fill_view(row: Any) -> tuple[str, str, str, float, float]:
    """``(execution_id, client_order_id, sid, qty, price)`` from a venue execution."""
    execution_id = _field(row, "execution_id", "id")
    if execution_id is None:
        raise ValueError(
            f"venue execution {row!r} carries no execution_id; the fills ledger is keyed by it (§3.11), "
            "and a row without one cannot be de-duplicated"
        )
    return (
        str(execution_id),
        str(_field(row, "client_order_id", default="")),
        str(_field(row, "sid", "symbol", default="")),
        float(_field(row, "qty", "quantity", default=0.0) or 0.0),
        float(_field(row, "price", "fill_price", default=0.0) or 0.0),
    )


# ---------------------------------------------------------------------------
# the accounting comparisons
# ---------------------------------------------------------------------------


def reconcile_positions(
    state: PortfolioState,
    *,
    broker_positions: Mapping[str, float],
    broker_cash: float,
    settings: ReconcileSettings = ReconcileSettings(),
    reason: str = "broker truth adopted",
) -> tuple[PortfolioState, list[ReconcileItem]]:
    """Adopt the venue's positions and cash, recording every difference absorbed.

    Every name is compared, in both directions: a name the venue holds and the
    book does not is as much a divergence as the reverse, and only comparing
    the book's own names would make an unexpected holding invisible. A name
    that leaves the book keeps nothing behind; a name that arrives is carried
    ``stale`` with no mark, never at a price nobody supplied — the next close
    prices it or the cycle refuses to trade and names it (§3.6).
    """
    adopted = state.copy()
    items: list[ReconcileItem] = []
    venue = {str(sid): float(shares) for sid, shares in dict(broker_positions).items() if float(shares) != 0.0}
    for sid in sorted(set(venue) | set(adopted.positions)):
        held = adopted.positions.get(sid)
        engine_shares = 0.0 if held is None else held.shares
        venue_shares = venue.get(sid, 0.0)
        difference = venue_shares - engine_shares
        if abs(difference) > settings.share_tolerance:
            items.append(
                ReconcileItem(
                    check="positions",
                    cls=ACCOUNTING,
                    detail=(
                        f"{sid}: the venue holds {venue_shares:g} shares and the book carried "
                        f"{engine_shares:g}; {reason}"
                    ),
                    sid=sid,
                    engine=engine_shares,
                    observed=venue_shares,
                    difference=difference,
                    adopted=True,
                )
            )
        if venue_shares == 0.0:
            adopted.positions.pop(sid, None)
        elif held is None:
            adopted.positions[sid] = Position(shares=venue_shares, valuation_status="stale")
        else:
            held.shares = venue_shares

    cash_difference = float(broker_cash) - adopted.cash
    if abs(cash_difference) > settings.cash_tolerance:
        items.append(
            ReconcileItem(
                check="cash",
                cls=ACCOUNTING,
                detail=(
                    f"the venue reports {float(broker_cash):.2f} cash and the book carried "
                    f"{adopted.cash:.2f}; {reason}"
                ),
                engine=adopted.cash,
                observed=float(broker_cash),
                difference=cash_difference,
                adopted=True,
            )
        )
    adopted.cash = float(broker_cash)
    if items:
        logger.warning(
            "reconcile: adopted broker truth over %d difference(s) (%s)",
            len(items),
            ", ".join(sorted({item.sid or "cash" for item in items})),
        )
    return adopted, items


def reconcile_orders(
    state: PortfolioState,
    *,
    open_orders: Sequence[Any] | None,
    terminal_order_ids: Iterable[str] = (),
    open_orders_available: bool = True,
) -> list[ReconcileItem]:
    """Compare the venue's live orders against the pending set, on id, quantity and side.

    A quantity or side difference is not adoptable and is marked blocking: an
    order the venue holds at a different size than the one that was decided is
    two systems disagreeing about what was submitted, which is also how one
    host's orders being substituted for another's surfaces (§3.10).

    An order the book still carries as open, that the venue neither reports live
    nor declares terminal and that has printed no execution, has no known fate:
    it is reported as ``unknown_order_state``. An adapter that cannot list open
    orders at all (``open_orders_available=False``) puts every open order in that
    class rather than being read as "the venue holds none".
    """
    items: list[ReconcileItem] = []
    engine_open = {record.client_order_id: record for record in state.open_orders()}
    terminal = {str(order_id) for order_id in terminal_order_ids}

    if not open_orders_available:
        for client_order_id, record in sorted(engine_open.items()):
            items.append(
                ReconcileItem(
                    check="unknown_order_state",
                    cls=ACCOUNTING,
                    detail=(
                        f"{client_order_id}: the adapter cannot list open orders, so this pending order's "
                        "fate is unknown; 'no open orders' and 'cannot list open orders' are different facts"
                    ),
                    sid=record.sid,
                    client_order_id=client_order_id,
                    engine=record.remaining_qty,
                    blocking=True,
                )
            )
        return items

    venue = {}
    for row in open_orders or ():
        client_order_id, sid, qty, filled, status = _order_view(row)
        venue[client_order_id] = (sid, qty, filled, status)

    for client_order_id, (sid, qty, _filled, status) in sorted(venue.items()):
        record = state.order(client_order_id)
        if record is None:
            items.append(
                ReconcileItem(
                    check="open_orders",
                    cls=ACCOUNTING,
                    detail=(
                        f"{client_order_id}: the venue holds a live order for {qty:g} share(s) of {sid!r} "
                        f"(status {status!r}) that the book has no record of submitting"
                    ),
                    sid=sid or None,
                    client_order_id=client_order_id,
                    observed=qty,
                    blocking=True,
                )
            )
            continue
        if sid and record.sid != sid:
            items.append(
                ReconcileItem(
                    check="open_orders",
                    cls=ACCOUNTING,
                    detail=(
                        f"{client_order_id}: the venue holds it against {sid!r} and the book decided it for "
                        f"{record.sid!r}; one client order id names two names"
                    ),
                    sid=sid,
                    client_order_id=client_order_id,
                    blocking=True,
                )
            )
            continue
        decided = record.intent.qty
        if qty * decided < 0.0:
            items.append(
                ReconcileItem(
                    check="open_orders",
                    cls=ACCOUNTING,
                    detail=(
                        f"{client_order_id}: the venue holds {qty:g} share(s) and the book decided "
                        f"{decided:g} — opposite sides, which no fill can explain"
                    ),
                    sid=record.sid,
                    client_order_id=client_order_id,
                    engine=decided,
                    observed=qty,
                    difference=qty - decided,
                    blocking=True,
                )
            )
        elif abs(qty - decided) > 1e-9 * max(1.0, abs(decided)):
            items.append(
                ReconcileItem(
                    check="open_orders",
                    cls=ACCOUNTING,
                    detail=(
                        f"{client_order_id}: the venue holds {qty:g} share(s) and the book decided "
                        f"{decided:g}; a submitted quantity is not adoptable"
                    ),
                    sid=record.sid,
                    client_order_id=client_order_id,
                    engine=decided,
                    observed=qty,
                    difference=qty - decided,
                    blocking=True,
                )
            )

    for client_order_id, record in sorted(engine_open.items()):
        if client_order_id in venue or client_order_id in terminal or record.fills:
            continue
        items.append(
            ReconcileItem(
                check="unknown_order_state",
                cls=ACCOUNTING,
                detail=(
                    f"{client_order_id}: submitted for {record.intent.qty:g} share(s) of {record.sid}, and the "
                    "venue reports it neither live nor terminal and has printed no execution against it"
                ),
                sid=record.sid,
                client_order_id=client_order_id,
                engine=record.intent.qty,
                blocking=True,
            )
        )
    return items


def reconcile_fills(
    fills: Sequence[Any] | None,
    *,
    ledger_execution_ids: Iterable[str],
) -> list[ReconcileItem]:
    """Compare the executions pulled since ``fills_marker`` against the fills ledger.

    An execution the ledger does not carry is the same case as an order
    mismatch: the book's record of what was traded is incomplete, so it is
    blocking. The protocol that makes this comparison meaningful is
    append-then-advance (§3.11) — rows are appended and flushed and only then is
    the marker saved, so a crash re-pulls executions the ledger already has
    (which is clean) and never advances past ones it does not.
    """
    known = {str(execution_id) for execution_id in ledger_execution_ids}
    items: list[ReconcileItem] = []
    for row in fills or ():
        execution_id, client_order_id, sid, qty, price = _fill_view(row)
        if execution_id in known:
            continue
        items.append(
            ReconcileItem(
                check="fills",
                cls=ACCOUNTING,
                detail=(
                    f"execution {execution_id} ({qty:g} share(s) of {sid} at {price:g} under order "
                    f"{client_order_id}) is not in the fills ledger; the book's record of what traded is "
                    "incomplete"
                ),
                sid=sid or None,
                client_order_id=client_order_id or None,
                observed=qty,
                blocking=True,
            )
        )
    return items


# ---------------------------------------------------------------------------
# the telemetry comparisons — recorded, never gating
# ---------------------------------------------------------------------------


def telemetry_reads(
    *,
    modeled_cost_dollars: float | None = None,
    observed_cost_dollars: float | None = None,
    traded_notional: float | None = None,
    realized_turnover: float | None = None,
    evaluated_turnover: float | None = None,
    realized_variance: float | None = None,
    variance_budget: float | None = None,
    settings: ReconcileSettings = ReconcileSettings(),
) -> list[ReconcileItem]:
    """The three recorded comparisons of §3.11. Every item is ``telemetry`` and never blocks.

    A comparison whose inputs are absent is not run and not reported: a missing
    number is not agreement, and inventing a zero for it would make an unmeasured
    book look like a calibrated one.
    """
    items: list[ReconcileItem] = []
    if modeled_cost_dollars is not None and observed_cost_dollars is not None:
        difference = float(observed_cost_dollars) - float(modeled_cost_dollars)
        denominator = abs(float(traded_notional)) if traded_notional else None
        if denominator:
            miss_bps = 1e4 * difference / denominator
            if abs(miss_bps) > settings.cost_tolerance_bps:
                items.append(
                    ReconcileItem(
                        check="execution_cost",
                        cls=TELEMETRY,
                        detail=(
                            f"observed execution cost {float(observed_cost_dollars):.2f} vs modeled "
                            f"{float(modeled_cost_dollars):.2f} on {denominator:.2f} traded "
                            f"({miss_bps:+.2f} bps, tolerance {settings.cost_tolerance_bps:.2f})"
                        ),
                        engine=float(modeled_cost_dollars),
                        observed=float(observed_cost_dollars),
                        difference=difference,
                    )
                )
        elif abs(difference) > 0.0:
            items.append(
                ReconcileItem(
                    check="execution_cost",
                    cls=TELEMETRY,
                    detail=(
                        f"observed execution cost {float(observed_cost_dollars):.2f} vs modeled "
                        f"{float(modeled_cost_dollars):.2f}; no traded notional was supplied, so the miss "
                        "is reported in dollars rather than in basis points"
                    ),
                    engine=float(modeled_cost_dollars),
                    observed=float(observed_cost_dollars),
                    difference=difference,
                )
            )
    if realized_turnover is not None and evaluated_turnover is not None:
        difference = float(realized_turnover) - float(evaluated_turnover)
        scale = max(abs(float(evaluated_turnover)), 1e-12)
        if abs(difference) / scale > settings.turnover_tolerance:
            items.append(
                ReconcileItem(
                    check="turnover",
                    cls=TELEMETRY,
                    detail=(
                        f"realized turnover {float(realized_turnover):.6f} vs evaluated "
                        f"{float(evaluated_turnover):.6f} ({difference / scale:+.2%}, tolerance "
                        f"{settings.turnover_tolerance:.2%})"
                    ),
                    engine=float(evaluated_turnover),
                    observed=float(realized_turnover),
                    difference=difference,
                )
            )
    if realized_variance is not None and variance_budget is not None:
        budget = float(variance_budget)
        difference = float(realized_variance) - budget
        if difference > settings.variance_tolerance * max(abs(budget), 1e-12):
            items.append(
                ReconcileItem(
                    check="variance",
                    cls=TELEMETRY,
                    detail=(
                        f"realized variance {float(realized_variance):.8f} is above the budget {budget:.8f} "
                        f"by {difference:.8f} (tolerance {settings.variance_tolerance:.2%} of budget). "
                        "Recorded only: a variance budget is a construction constraint, and a realized miss "
                        "is a finding about the risk model, not an operating fault"
                    ),
                    engine=budget,
                    observed=float(realized_variance),
                    difference=difference,
                )
            )
    return items


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------


def reconcile(
    state: PortfolioState,
    *,
    broker_positions: Mapping[str, float],
    broker_cash: float,
    open_orders: Sequence[Any] | None = None,
    open_orders_available: bool = True,
    terminal_order_ids: Iterable[str] = (),
    fills: Sequence[Any] | None = None,
    ledger_execution_ids: Iterable[str] = (),
    settings: ReconcileSettings = ReconcileSettings(),
    **telemetry: Any,
) -> Reconciliation:
    """One reconciliation pass over one book (§3.11).

    Returns the state with broker truth adopted for positions and cash, the
    items every comparison produced, and the status: ``clean``, ``discrepancy``
    with the list, or ``unknown_order_state``.
    """
    adopted, items = reconcile_positions(
        state,
        broker_positions=broker_positions,
        broker_cash=broker_cash,
        settings=settings,
    )
    items += reconcile_orders(
        state,
        open_orders=open_orders,
        terminal_order_ids=terminal_order_ids,
        open_orders_available=open_orders_available,
    )
    items += reconcile_fills(fills, ledger_execution_ids=ledger_execution_ids)
    items += telemetry_reads(settings=settings, **telemetry)

    if any(item.check == "unknown_order_state" for item in items):
        status = STATUS_UNKNOWN_ORDER_STATE
    elif items:
        status = STATUS_DISCREPANCY
    else:
        status = STATUS_CLEAN
    return Reconciliation(status=status, items=tuple(items), state=adopted, settings=settings)


def reconcile_loop_state(
    loop_state: Any,
    broker: Broker,
    *,
    ledger_execution_ids: Iterable[str] = (),
    terminal_order_ids: Iterable[str] = (),
    settings: ReconcileSettings = ReconcileSettings(),
    **telemetry: Any,
) -> Reconciliation:
    """Reconcile a :class:`~prism.live.state.LoopState` against a broker, in place.

    The adopted book is written back onto ``loop_state.portfolio``; the fills
    marker is **not** advanced here, because the protocol is append-then-advance
    and only the caller that appended the rows may save the new marker (§3.11).
    An adapter that does not implement ``open_orders`` or ``fills_since`` is not
    read as reporting nothing: the capability is recorded as unavailable and the
    orders it would have covered come back as ``unknown_order_state``.
    """
    try:
        open_orders: Sequence[Any] | None = broker.open_orders()
        open_orders_available = True
    except BrokerCapabilityUnsupported:
        open_orders, open_orders_available = None, False

    fills: Sequence[Any] | None = None
    try:
        fills, _marker = broker.fills_since(loop_state.fills_marker)
    except BrokerCapabilityUnsupported:
        logger.info(
            "reconcile: %s does not implement fills_since(); the executions comparison is not run this pass",
            type(broker).__name__,
        )

    report = reconcile(
        loop_state.portfolio,
        broker_positions=broker.positions(),
        broker_cash=broker.cash(),
        open_orders=open_orders,
        open_orders_available=open_orders_available,
        terminal_order_ids=terminal_order_ids,
        fills=fills,
        ledger_execution_ids=ledger_execution_ids,
        settings=settings,
        **telemetry,
    )
    if report.state is not None:
        loop_state.portfolio = report.state
    return report
