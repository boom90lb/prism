"""Broker contract for the live loop (SPEC.md §7.4 live extension; §3.9).

The contract is deliberately small: snapshot truth (positions, cash),
idempotent submission, and fill retrieval. Idempotency is the load-bearing
requirement — the loop's crash-safety protocol (``prism.live.loop``)
resubmits after a restart, so submitting the same ``client_order_id``
twice must never create two orders. Every real venue in scope supports a
client order id (Alpaca: ``client_order_id``), and adapters are expected
to map :class:`DuplicateOrder` onto the venue's duplicate-id rejection.

Section 3.9 of the implementation specification extends it with the six
capabilities the operating states of section 3.10 and the durable state of
section 3.11 need but cannot synthesise: account identity, dated per-security
capabilities, buying power, open orders with their quantities, an idempotent
cancel, executions since a marker, and the venue's clock. They are declared
here as concrete methods that *raise* rather than as abstract ones, for one
reason: an adapter that does not implement a capability must fail loudly at the
call that needs it, not become unconstructible. Every fake in the test suite,
and :class:`~prism.live.replay.ReplayBroker`, satisfies the original contract
and implements only the extensions it is exercised for.

Concrete network adapters are not in this module; they land when there
are credentials to run them against the paper venue. Tests exercise the
contract through a fake.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = [
    "CAPABILITY_COLUMNS",
    "AccountIdentity",
    "Broker",
    "BrokerCapabilityUnsupported",
    "CapabilityRow",
    "DuplicateOrder",
    "Fill",
    "FillRecord",
    "Order",
    "OrderRejected",
    "OrderStatus",
    "transmits_to_venue",
]


@dataclass(frozen=True)
class Order:
    """One next-open order decided at close ``decision_bar``.

    ``qty`` is signed shares (buy > 0, sell < 0). ``client_order_id`` is
    the idempotency key: deterministic in (decision_bar, symbol) so a
    restarted loop regenerates exactly the ids it may already have
    submitted. ``reference_price`` is the decision-close price the order
    was sized at — persisted so the fills ledger can record arrival
    slippage (fill vs reference), the quantity I-9 calibrates on.

    It carries no ``order_type`` or ``time_in_force``: this is the persisted
    live-loop order and its schema is the state file's. The canonical
    :class:`~prism.execution.portfolio_state.OrderIntent` carries both per order
    (§3.8), and the adapter reads them off whichever shape it is handed (§3.9).
    """

    client_order_id: str
    symbol: str
    qty: float
    decision_bar: str
    reference_price: float

    def __post_init__(self) -> None:
        if not self.client_order_id:
            raise ValueError("client_order_id must be non-empty")
        if self.qty == 0.0:
            raise ValueError(f"zero-qty order for {self.symbol!r} ({self.client_order_id})")
        if not self.reference_price > 0:
            raise ValueError(f"reference_price must be > 0, got {self.reference_price} " f"({self.client_order_id})")


@dataclass(frozen=True)
class Fill:
    """A completed execution, in the units the I-9 ledger calibrates on."""

    client_order_id: str
    symbol: str
    qty: float
    price: float
    filled_bar: str


@dataclass(frozen=True)
class AccountIdentity:
    """Which account, at which endpoint, this broker is actually talking to (§3.9).

    ``endpoint`` is the client's own resolved base URL, not a venue-reported
    field: the question it answers is "did this process point at the endpoint
    the operator intended", and a venue's self-description cannot answer that.
    ``is_paper`` is ``None`` when the venue does not report it — inferring it
    from the URL would turn a naming convention into an authorization fact.
    """

    account_id: str
    endpoint: str
    is_paper: bool | None


@dataclass(frozen=True)
class OrderStatus:
    """One order the venue currently knows, with the quantities ``open_order_ids`` omits.

    ``qty`` is signed shares as submitted; ``order_type`` and ``time_in_force``
    are the venue's own strings, because this is the venue's report of what it
    holds, not the engine's declaration of what it asked for. The execution
    profile's ``[order_type_map]`` is the translation in both directions.
    """

    client_order_id: str
    sid: str
    qty: float
    filled_qty: float
    order_type: str
    time_in_force: str
    status: str


@dataclass(frozen=True)
class FillRecord:
    """One venue *execution* — the fills-ledger key of section 3.11.

    Distinct from :class:`Fill`, which is one aggregate per client order id.
    ``execution_id`` is the venue's per-execution identity, which is what makes
    the append-then-advance protocol idempotent: a re-pull after a crash between
    the append and the marker save is dropped by key.
    """

    execution_id: str
    client_order_id: str
    sid: str
    qty: float
    price: float
    filled_at: str


#: The ``capabilities`` table of section 3.2, in declared column order. The three
#: list-valued columns are stored JSON-encoded, which is the form
#: ``prism.portfolio.validate.capability_row`` parses.
CAPABILITY_COLUMNS: tuple[str, ...] = (
    "sid",
    "valid_at",
    "fractionable",
    "quantity_increment",
    "min_notional",
    "shortable",
    "easy_to_borrow",
    "lot_size",
    "auction_eligible",
    "restrictions",
    "order_types",
    "sessions",
    "available_at",
    "ingested_at",
    "source",
    "revision",
)

_LIST_COLUMNS: tuple[str, ...] = ("restrictions", "order_types", "sessions")


@dataclass(frozen=True)
class CapabilityRow:
    """One dated row of the ``capabilities`` table (§3.2), as the broker reports it.

    Every field is a venue *classification*, never a prediction: ``auction_eligible``
    says the venue admits this name to the opening auction, not that an order will
    fill there, and ``lot_size`` is a round-lot classification, never the quantity
    increment. ``lot_size`` is ``None`` when the venue reports none — the alternative,
    defaulting it to 100, would state a fact no source supplied.

    ``valid_at`` is the row's time of record and, per section 3.2, also its derived
    availability. ``ingested_at`` is when this workspace stored it; ``revision`` is
    monotone within ``(key, source)`` and is assigned by the ingest path, so a row
    straight off a broker carries revision 1.
    """

    sid: str
    valid_at: str
    fractionable: bool
    quantity_increment: float
    min_notional: float
    shortable: bool
    easy_to_borrow: bool
    lot_size: int | None
    auction_eligible: bool
    restrictions: tuple[str, ...]
    order_types: tuple[str, ...]
    sessions: tuple[str, ...]
    available_at: str
    ingested_at: str
    source: str
    revision: int = 1

    def __post_init__(self) -> None:
        if not self.sid:
            raise ValueError("capability row needs a sid")
        if not self.valid_at:
            raise ValueError(f"{self.sid}: capability row needs a valid_at; an undated capability is not dated")
        increment = float(self.quantity_increment)
        if not math.isfinite(increment) or increment <= 0.0:
            raise ValueError(f"{self.sid}: quantity_increment must be finite and > 0, got {self.quantity_increment!r}")
        notional = float(self.min_notional)
        if not math.isfinite(notional) or notional < 0.0:
            raise ValueError(f"{self.sid}: min_notional must be finite and >= 0, got {self.min_notional!r}")
        if self.lot_size is not None and int(self.lot_size) < 1:
            raise ValueError(f"{self.sid}: lot_size must be >= 1 or None, got {self.lot_size!r}")
        if self.revision < 1:
            raise ValueError(f"{self.sid}: revision is monotone from 1, got {self.revision!r}")

    def to_row(self) -> dict[str, Any]:
        """The row in the table's storage form, with the list columns JSON-encoded."""
        row: dict[str, Any] = {}
        for column in CAPABILITY_COLUMNS:
            value = getattr(self, column)
            row[column] = json.dumps(list(value)) if column in _LIST_COLUMNS else value
        return row


class DuplicateOrder(Exception):
    """Raised by ``submit`` when the client_order_id was already accepted.

    The loop treats this as success (the order exists), which is what makes
    resubmission after a crash safe.
    """


class OrderRejected(Exception):
    """The venue definitively rejected this one order (a client-side 4xx:
    insufficient qty / a side-crossing order, non-shortable name, buying power).

    The venue is healthy — only this order is bad — so the loop skips it, logs
    loudly, and submits the rest; the order stays pending and settle tolerates
    its missing fill by re-anchoring to broker truth. This is distinct from a
    transport/connection failure (or a 5xx), which leaves the order's fate
    unknown and must propagate so the write-ahead protocol's crash-resume
    applies rather than silently dropping a possibly-live order.
    """


class BrokerCapabilityUnsupported(NotImplementedError):
    """This adapter does not implement a section 3.9 capability the caller needs.

    Raised instead of returning an empty or permissive answer: "no open orders"
    and "cannot list open orders" are different facts, and only one of them
    makes it safe to sweep.
    """


class Broker(ABC):
    """Minimal venue contract the live loop is written against."""

    #: Whether an order handed to :meth:`submit` actually reaches a trading
    #: venue. ``True`` for every adapter that transmits; ``False`` for the
    #: rehearsal double (``prism.live.no_submit.NoSubmitBroker``) and the
    #: replay venue, which record orders in-process. It exists so that the
    #: submission announcement in ``prism.live.loop`` — the one line in the log
    #: that says a run traded — cannot be printed by a run that did not.
    submits_to_venue: bool = True

    @abstractmethod
    def positions(self) -> dict[str, float]:
        """Current signed share positions (symbol -> shares), broker truth."""

    @abstractmethod
    def cash(self) -> float:
        """Settled cash in dollars, broker truth."""

    @abstractmethod
    def submit(self, order: Order) -> None:
        """Accept one order; raise :class:`DuplicateOrder` on a repeated id.

        Raise :class:`OrderRejected` when the venue rejects *this* order (a
        client-side 4xx) — the loop skips it and submits the rest. Any other
        exception (transport failure, 5xx) means the order may NOT have been
        accepted: it propagates, and the write-ahead protocol resumes the
        persisted decision idempotently on the next pass.

        An order carrying ``order_type`` and ``time_in_force`` (the canonical
        :class:`~prism.execution.portfolio_state.OrderIntent`) is submitted with
        those, mapped through the execution profile's ``[order_type_map]``; one
        that carries neither takes the adapter's constructed default (§3.9).
        """

    @abstractmethod
    def submitted_order_ids(self) -> set[str]:
        """Every client_order_id the venue knows (open or filled)."""

    @abstractmethod
    def fills_for(self, client_order_ids: set[str]) -> list[Fill]:
        """Fills for the given ids that have completed; absent = not filled yet."""

    # ------------------------------------------------------ section 3.9 extensions
    #
    # Concrete, not abstract, and every one of them raises: an adapter that cannot
    # answer must say so at the call, rather than being impossible to construct or
    # answering permissively.

    def account_identity(self) -> AccountIdentity:
        """Which account and endpoint this adapter is bound to (§3.9)."""
        raise BrokerCapabilityUnsupported(f"{type(self).__name__} does not implement account_identity()")

    def capabilities(self, sids: Sequence[str], as_of: str) -> list[CapabilityRow]:
        """Dated ``capabilities`` rows for ``sids``, valid at ``as_of`` (§3.2, §3.9)."""
        raise BrokerCapabilityUnsupported(f"{type(self).__name__} does not implement capabilities()")

    def buying_power(self) -> float:
        """Buying power in dollars, revalidated at submission (§3.9)."""
        raise BrokerCapabilityUnsupported(f"{type(self).__name__} does not implement buying_power()")

    def open_orders(self) -> list[OrderStatus]:
        """Every order live at the venue, with quantities (§3.9)."""
        raise BrokerCapabilityUnsupported(f"{type(self).__name__} does not implement open_orders()")

    def cancel(self, client_order_id: str) -> None:
        """Cancel one order by client id; idempotent on a terminal or unknown id (§3.9)."""
        raise BrokerCapabilityUnsupported(f"{type(self).__name__} does not implement cancel()")

    def fills_since(self, marker: str | None) -> tuple[list[FillRecord], str | None]:
        """Executions after ``marker``, and the next marker (§3.11 append-then-advance)."""
        raise BrokerCapabilityUnsupported(f"{type(self).__name__} does not implement fills_since()")

    def server_time(self) -> str:
        """The venue's clock, for the section 3.11 clock-health check (§3.9)."""
        raise BrokerCapabilityUnsupported(f"{type(self).__name__} does not implement server_time()")


def transmits_to_venue(broker: object) -> bool:
    """Whether orders handed to ``broker`` actually reach a trading venue.

    Read through :func:`getattr` with a ``True`` default rather than off the
    class, because the answer must be conservative: an adapter that does not
    declare itself is assumed to trade. Only a double that *knows* it transmits
    nothing may opt out, and opting out only ever suppresses a log line — never
    a rail.
    """
    return bool(getattr(broker, "submits_to_venue", True))
