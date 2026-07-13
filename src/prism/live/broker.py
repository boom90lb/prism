"""Broker contract for the live loop (SPEC.md §7.4 live extension).

The contract is deliberately small: snapshot truth (positions, cash),
idempotent submission, and fill retrieval. Idempotency is the load-bearing
requirement — the loop's crash-safety protocol (``prism.live.loop``)
resubmits after a restart, so submitting the same ``client_order_id``
twice must never create two orders. Every real venue in scope supports a
client order id (Alpaca: ``client_order_id``), and adapters are expected
to map :class:`DuplicateOrder` onto the venue's duplicate-id rejection.

Concrete network adapters are not in this module; they land when there
are credentials to run them against the paper venue. Tests exercise the
contract through a fake.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    """One next-open order decided at close ``decision_bar``.

    ``qty`` is signed shares (buy > 0, sell < 0). ``client_order_id`` is
    the idempotency key: deterministic in (decision_bar, symbol) so a
    restarted loop regenerates exactly the ids it may already have
    submitted. ``reference_price`` is the decision-close price the order
    was sized at — persisted so the fills ledger can record arrival
    slippage (fill vs reference), the quantity I-9 calibrates on.
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
            raise ValueError(
                f"reference_price must be > 0, got {self.reference_price} "
                f"({self.client_order_id})"
            )


@dataclass(frozen=True)
class Fill:
    """A completed execution, in the units the I-9 ledger calibrates on."""

    client_order_id: str
    symbol: str
    qty: float
    price: float
    filled_bar: str


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


class Broker(ABC):
    """Minimal venue contract the live loop is written against."""

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
        """

    @abstractmethod
    def submitted_order_ids(self) -> set[str]:
        """Every client_order_id the venue knows (open or filled)."""

    @abstractmethod
    def fills_for(self, client_order_ids: set[str]) -> list[Fill]:
        """Fills for the given ids that have completed; absent = not filled yet."""
