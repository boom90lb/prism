"""The canonical portfolio state: shares, cash, and the claims either side of them (spec §3.6).

The reproduction engine (``execution/target_weights.py``) holds *weights* and re-levers to
them every bar. The canonical engine holds **shares**, and weights are derived from the
marked state, never stored as truth. That difference is the whole reason this module
exists: a share-based book drifts with prices between decisions, its cash is a real
account that can earn or pay a declared rate, and a corporate action moves shares and
cash on dates that have nothing to do with a rebalance.

What the state is, and what it is not:

* ``positions`` are signed share counts with the price each was last marked at and a
  ``valuation_status`` saying how that mark was obtained. A position is never dropped
  because a bar is missing -- it goes ``stale`` and keeps its last mark. A silent zero is
  the defect this contract exists to prevent (SPEC N7).
* ``cash`` is a real balance. Short sale proceeds sit in it; ``short_collateral`` is the
  part of it the broker has already claimed, which is why it is subtracted from buying
  power and (under ``collateral_no_rebate``) from the cash-income base.
* ``receivables`` and ``liabilities`` are declared claims with a date: a distribution
  booked on its ex-date and paid later, an accrued borrow charge not yet settled, a
  terminal payout pending. They enter ``equity`` the moment they are booked, which is what
  keeps equity continuous across an ex-date price markdown.
* ``pending_orders`` are submitted intents that have not settled. They are not positions
  and they are not cash, but a pending buy has already promised cash, so
  ``committed_cash`` subtracts it from buying power before the fill.

Every quantity here is signed the same way: ``+`` is long / bought / owed to us, ``-`` is
short / sold / owed by us. ``liabilities`` and ``receivables`` are the exception and are
held as positive magnitudes on their respective side, because that is how a statement
reads and how ``equity`` uses them (``+ receivables - liabilities``).

The transitions that move any of this live in :mod:`prism.execution.accounting`; this
module is the state and its derived quantities only. It moves nothing on its own.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Mapping

import pandas as pd

__all__ = [
    "ACTION_CLASSES",
    "BORROW_ACCRUAL_KEY",
    "CASH_INCOME_ACCRUAL_KEY",
    "FILL_SOURCES",
    "FINANCING_ACCRUAL_KEY",
    "ORDER_STATUSES",
    "ORDER_TYPES",
    "TIME_IN_FORCE",
    "VALUATION_STATUSES",
    "AccountingError",
    "Fill",
    "OrderIntent",
    "OrderRecord",
    "PortfolioState",
    "Position",
    "UnpricedPosition",
    "bar_key",
    "distribution_key",
    "empty_state",
    "frame_of_positions",
    "state_from_shares",
    "terminal_payout_key",
]

#: ``Position.valuation_status`` (§3.6). ``priced`` marks at the bar, ``stale`` at
#: ``last_price`` because the bar did not print, ``unresolved_claim`` at ``last_price``
#: *and* flags every record built on the state as ``incomplete`` (§3.2 strict claims).
VALUATION_STATUSES: tuple[str, ...] = ("priced", "stale", "unresolved_claim")

#: ``OrderRecord.status`` (§3.6).
ORDER_STATUSES: tuple[str, ...] = ("pending", "partial", "filled", "expired", "rejected", "canceled")

#: The two order types the execution profile admits (§3.9).
ORDER_TYPES: tuple[str, ...] = ("market_on_open", "market_day")

#: ``_VALID_TIME_IN_FORCE`` of ``src/prism/live/alpaca.py:66``, by value.
TIME_IN_FORCE: tuple[str, ...] = ("opg", "day")

#: Action classes of §3.10, carried on the intent so authorization is recorded per order.
ACTION_CLASSES: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4", "C5", "OB")

#: Where a fill came from. A ``broker`` fill carries the venue's price and no modeled cost.
FILL_SOURCES: tuple[str, ...] = ("simulated", "broker")

#: Liability key for borrow accrued and not yet settled (``holding_cost_settlement``).
BORROW_ACCRUAL_KEY = "accrued_borrow"
#: Liability key for financing accrued and not yet settled.
FINANCING_ACCRUAL_KEY = "accrued_financing"
#: Receivable key for cash income accrued and not yet settled.
CASH_INCOME_ACCRUAL_KEY = "accrued_cash_income"

_ACCRUAL_KEYS = (BORROW_ACCRUAL_KEY, FINANCING_ACCRUAL_KEY, CASH_INCOME_ACCRUAL_KEY)


class AccountingError(ValueError):
    """A state or transition that would misstate the ledger."""


class UnpricedPosition(AccountingError):
    """A held position with no mark. Never valued at zero -- that is a silent erasure."""


def bar_key(value: str | dt.date | dt.datetime | pd.Timestamp) -> str:
    """The ISO ``YYYY-MM-DD`` bar key of a date, instant, or already-ISO string.

    Bars are the unit of every dated thing here: an effective date, a pay date, a fill
    timestamp's session. Comparing them as ISO dates is exact and order-preserving, and it
    is what the ``events`` and ``bars`` tables of §3.2 store.
    """
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    text = str(value)
    if len(text) < 10:
        raise AccountingError(f"expected an ISO date or instant, got {value!r}")
    head = text[:10]
    try:
        dt.date.fromisoformat(head)
    except ValueError as exc:
        raise AccountingError(f"expected an ISO date or instant, got {value!r}") from exc
    return head


def distribution_key(sid: str, ex_date: str) -> str:
    """The ``receivables`` / ``liabilities`` key of one distribution entitlement.

    Keyed by ``(sid, ex-date)`` rather than by pay date: the ex-date is what booked the
    entry, the pay date is read back off the event when it settles, and a second
    distribution on the same name with the same ex-date is the same entitlement.
    """
    return f"cash_distribution:{sid}:{bar_key(ex_date)}"


def terminal_payout_key(sid: str, effective_date: str) -> str:
    """The ``receivables`` key of a terminal payout awaiting its pay date."""
    return f"terminal_payout:{sid}:{bar_key(effective_date)}"


def _finite(value: Any, what: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AccountingError(f"{what} must be finite, got {value!r}")
    return number


def _price_of(prices: Mapping[str, float] | pd.Series | None, sid: str) -> float | None:
    """The usable price ``prices`` carries for ``sid``, or ``None``.

    ``None`` means "this quote says nothing", which is a different statement from a price
    of zero. NaN, a missing key, and a non-positive price all reduce to ``None``; only a
    finite positive number is a price.
    """
    if prices is None:
        return None
    try:
        if isinstance(prices, pd.Series):
            if sid not in prices.index:
                return None
            raw = prices.loc[sid]
            if isinstance(raw, pd.Series):  # duplicated index label
                raise AccountingError(f"price series carries {sid!r} more than once")
        else:
            if sid not in prices:
                return None
            raw = prices[sid]
    except KeyError:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise AccountingError(f"price for {sid!r} is not numeric: {raw!r}") from None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


@dataclass
class Position:
    """One name's signed share count and the mark it is carried at (§3.6)."""

    shares: float
    last_price: float | None = None
    last_price_bar: str | None = None
    valuation_status: Literal["priced", "stale", "unresolved_claim"] = "priced"

    def __post_init__(self) -> None:
        self.shares = _finite(self.shares, "position shares")
        if self.last_price is not None:
            price = _finite(self.last_price, "position last_price")
            if price <= 0.0:
                raise AccountingError(f"position last_price must be > 0, got {self.last_price!r}")
            self.last_price = price
        if self.last_price_bar is not None:
            self.last_price_bar = bar_key(self.last_price_bar)
        if self.valuation_status not in VALUATION_STATUSES:
            raise AccountingError(
                f"valuation_status must be one of {VALUATION_STATUSES}, got {self.valuation_status!r}"
            )

    def mark(self, prices: Mapping[str, float] | pd.Series | None, sid: str) -> float:
        """The price this position values at, given the bar's quotes.

        ``priced`` takes the bar's price; ``stale`` and ``unresolved_claim`` take
        ``last_price``, which is the last bar the name actually printed. A ``priced``
        position with no quote raises: the mark step (``accounting.mark_at_close``) is what
        turns an absent bar into ``stale``, and valuing it here as zero -- or quietly at a
        price nobody supplied -- is the erasure SPEC N7 forbids.
        """
        if self.valuation_status == "priced":
            price = _price_of(prices, sid)
            if price is not None:
                return price
            if self.last_price is not None:
                raise UnpricedPosition(
                    f"{sid}: position is marked 'priced' but the supplied quotes carry no price for it. "
                    "Mark the state for this bar first (a missing bar makes the holding stale at "
                    f"{self.last_price}); it is never valued at zero"
                )
            raise UnpricedPosition(f"{sid}: position has no price and no prior mark; it cannot be valued")
        if self.last_price is None:
            raise UnpricedPosition(
                f"{sid}: position is {self.valuation_status} with no prior mark, so it has no value to carry"
            )
        return self.last_price

    def value(self, prices: Mapping[str, float] | pd.Series | None, sid: str) -> float:
        """Signed marked value in dollars."""
        return self.shares * self.mark(prices, sid)

    def copy(self) -> "Position":
        return replace(self)


@dataclass(frozen=True)
class OrderIntent:
    """A decided order, in shares at the sid's increment, before submission (§3.6, §3.8)."""

    sid: str
    qty: float
    increment: float
    reference_price: float
    decision_bar: str
    order_type: Literal["market_on_open", "market_day"] = "market_on_open"
    time_in_force: Literal["opg", "day"] = "opg"
    action_class: Literal["C0", "C1", "C2", "C3", "C4", "C5", "OB"] = "C1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "sid", str(self.sid))
        qty = _finite(self.qty, f"{self.sid}: intent qty")
        if qty == 0.0:
            raise AccountingError(f"{self.sid}: an intent of zero shares is not an intent; drop it instead")
        object.__setattr__(self, "qty", qty)
        increment = _finite(self.increment, f"{self.sid}: intent increment")
        if increment < 0.0:
            raise AccountingError(f"{self.sid}: quantity increment must be >= 0, got {self.increment!r}")
        object.__setattr__(self, "increment", increment)
        reference = _finite(self.reference_price, f"{self.sid}: intent reference_price")
        if reference <= 0.0:
            raise AccountingError(f"{self.sid}: reference_price must be > 0, got {self.reference_price!r}")
        object.__setattr__(self, "reference_price", reference)
        object.__setattr__(self, "decision_bar", bar_key(self.decision_bar))
        if self.order_type not in ORDER_TYPES:
            raise AccountingError(f"{self.sid}: order_type must be one of {ORDER_TYPES}, got {self.order_type!r}")
        if self.time_in_force not in TIME_IN_FORCE:
            raise AccountingError(
                f"{self.sid}: time_in_force must be one of {TIME_IN_FORCE}, got {self.time_in_force!r}"
            )
        if self.action_class not in ACTION_CLASSES:
            raise AccountingError(
                f"{self.sid}: action_class must be one of {ACTION_CLASSES}, got {self.action_class!r}"
            )
        # market_on_open is an opening auction order and can only carry "opg"; market_day
        # is a day order. The pairing is the [order_type_map] block of §3.9, and letting
        # them cross would submit an order the venue cannot honour.
        expected = "opg" if self.order_type == "market_on_open" else "day"
        if self.time_in_force != expected:
            raise AccountingError(
                f"{self.sid}: {self.order_type} carries time_in_force {expected!r}, got {self.time_in_force!r} "
                "(§3.9 [order_type_map])"
            )
        if increment > 0.0:
            # increment == 0.0 is the declared continuous grid (§3.7,
            # RoundingSettings.for_reproduction): every quantity is on it.
            steps = qty / increment
            if abs(steps - round(steps)) > 1e-9 * max(1.0, abs(steps)):
                raise AccountingError(
                    f"{self.sid}: intent qty {qty} is not a whole multiple of the increment {increment}"
                )

    @property
    def notional(self) -> float:
        """Signed decided notional at the reference price."""
        return self.qty * self.reference_price


@dataclass(frozen=True)
class Fill:
    """One execution against one order (§3.6).

    The canonical fill keeps the executed price and the charged dollars **apart**: a
    simulated fill prices at the raw bar price under the profile's ``price_convention``
    and records the modeled spread and impact as separate debits, and a broker fill
    carries the venue's price with ``spread_dollars = impact_dollars = 0.0`` because the
    price already contains them. That separation is what makes the reconciliation of §3.6
    ("no double charge in operation") non-circular.

    This is a different type from :class:`research.execution_model.Fill`, which
    belongs to the retired research trader and embeds cost in the fill price.
    """

    execution_id: str
    client_order_id: str
    sid: str
    ts: str
    qty: float
    fill_price: float
    reference_price: float
    commission: float = 0.0
    spread_dollars: float = 0.0
    impact_dollars: float = 0.0
    source: Literal["simulated", "broker"] = "simulated"

    def __post_init__(self) -> None:
        for name in ("execution_id", "client_order_id", "sid", "ts"):
            object.__setattr__(self, name, str(getattr(self, name)))
        qty = _finite(self.qty, f"{self.sid}: fill qty")
        if qty == 0.0:
            raise AccountingError(f"{self.sid}: a fill of zero shares is not a fill")
        object.__setattr__(self, "qty", qty)
        price = _finite(self.fill_price, f"{self.sid}: fill_price")
        if price <= 0.0:
            raise AccountingError(f"{self.sid}: fill_price must be > 0, got {self.fill_price!r}")
        object.__setattr__(self, "fill_price", price)
        reference = _finite(self.reference_price, f"{self.sid}: fill reference_price")
        if reference <= 0.0:
            raise AccountingError(f"{self.sid}: reference_price must be > 0, got {self.reference_price!r}")
        object.__setattr__(self, "reference_price", reference)
        for name in ("commission", "spread_dollars", "impact_dollars"):
            value = _finite(getattr(self, name), f"{self.sid}: fill {name}")
            if value < 0.0:
                raise AccountingError(f"{self.sid}: fill {name} must be >= 0, got {value}")
            object.__setattr__(self, name, value)
        if self.source not in FILL_SOURCES:
            raise AccountingError(f"{self.sid}: fill source must be one of {FILL_SOURCES}, got {self.source!r}")
        if self.source == "broker" and (self.spread_dollars or self.impact_dollars):
            raise AccountingError(
                f"{self.sid}: a broker fill carries the venue's price, so spread_dollars and impact_dollars "
                "are 0.0 on it; charging them again is the double charge §3.6 forbids"
            )

    @property
    def bar(self) -> str:
        """The session this fill printed on."""
        return bar_key(self.ts)

    @property
    def notional(self) -> float:
        """Signed filled notional at the fill price."""
        return self.qty * self.fill_price

    @property
    def charged_cost(self) -> float:
        """Dollars debited on this fill over and above the traded notional."""
        return self.commission + self.spread_dollars + self.impact_dollars

    @property
    def cash_delta(self) -> float:
        """``-qty * fill_price - commission - spread_dollars - impact_dollars`` (§3.6 step 4)."""
        return -self.notional - self.charged_cost


@dataclass
class OrderRecord:
    """A submitted intent and everything that has happened to it (§3.6)."""

    intent: OrderIntent
    client_order_id: str
    status: Literal["pending", "partial", "filled", "expired", "rejected", "canceled"] = "pending"
    filled_qty: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    submitted_at: str = ""
    last_update: str = ""
    reason: str | None = None

    def __post_init__(self) -> None:
        self.client_order_id = str(self.client_order_id)
        if self.status not in ORDER_STATUSES:
            raise AccountingError(f"order status must be one of {ORDER_STATUSES}, got {self.status!r}")
        self.filled_qty = _finite(self.filled_qty, f"{self.client_order_id}: filled_qty")

    @property
    def sid(self) -> str:
        return self.intent.sid

    @property
    def is_open(self) -> bool:
        """Still capable of filling."""
        return self.status in ("pending", "partial")

    @property
    def remaining_qty(self) -> float:
        """Signed quantity still to fill; zero once complete."""
        remaining = self.intent.qty - self.filled_qty
        if abs(remaining) < 1e-12 * max(1.0, abs(self.intent.qty)):
            return 0.0
        return remaining

    def has_execution(self, execution_id: str) -> bool:
        """Whether this execution id has already been applied (idempotence key, §3.11)."""
        return any(f.execution_id == execution_id for f in self.fills)

    def copy(self) -> "OrderRecord":
        return OrderRecord(
            intent=self.intent,
            client_order_id=self.client_order_id,
            status=self.status,
            filled_qty=self.filled_qty,
            fills=list(self.fills),
            submitted_at=self.submitted_at,
            last_update=self.last_update,
            reason=self.reason,
        )


@dataclass
class PortfolioState:
    """Cash, shares, claims, and outstanding orders, as of one bar (§3.6).

    ``equity`` is ``cash + sum(shares * mark) + receivables - liabilities``. Everything
    else derived -- weights, collateral, buying power -- is computed from it and from the
    supplied quotes, never stored: a stored weight is a second source of truth that goes
    stale the moment a price moves.
    """

    as_of: str
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    liabilities: dict[str, float] = field(default_factory=dict)
    receivables: dict[str, float] = field(default_factory=dict)
    pending_orders: list[OrderRecord] = field(default_factory=list)
    # Obligation is declared in §3.7 and implemented in portfolio/obligations.py. It is
    # carried, not interpreted, here: the accounting kernel must not depend on the
    # constructor's module, and a split rebases obligation quantities through
    # accounting.apply_events by duck-typed field access.
    obligations: list[Any] = field(default_factory=list)
    sequence: int = 0

    def __post_init__(self) -> None:
        self.as_of = bar_key(self.as_of)
        self.cash = _finite(self.cash, "cash")
        for key, amount in list(self.liabilities.items()):
            self.liabilities[key] = _nonnegative_claim(amount, "liability", key)
        for key, amount in list(self.receivables.items()):
            self.receivables[key] = _nonnegative_claim(amount, "receivable", key)
        self.sequence = int(self.sequence)

    # ------------------------------------------------------------------ derived

    def sids(self) -> list[str]:
        """Held names, in a deterministic order."""
        return sorted(self.positions)

    def shares(self, sid: str) -> float:
        """Signed share count, zero for a name not held."""
        position = self.positions.get(sid)
        return 0.0 if position is None else position.shares

    def marks(self, prices: Mapping[str, float] | pd.Series | None = None) -> pd.Series:
        """Per-name mark price under the valuation rule of §3.6."""
        sids = self.sids()
        return pd.Series(
            [self.positions[sid].mark(prices, sid) for sid in sids],
            index=pd.Index(sids, name="sid"),
            dtype=float,
            name="mark",
        )

    def position_values(self, prices: Mapping[str, float] | pd.Series | None = None) -> pd.Series:
        """Per-name signed marked value in dollars."""
        sids = self.sids()
        return pd.Series(
            [self.positions[sid].value(prices, sid) for sid in sids],
            index=pd.Index(sids, name="sid"),
            dtype=float,
            name="value",
        )

    def positions_value(self, prices: Mapping[str, float] | pd.Series | None = None) -> float:
        """Total signed marked value of the book's positions."""
        return float(sum(position.value(prices, sid) for sid, position in self.positions.items()))

    def receivables_total(self) -> float:
        return float(sum(self.receivables.values()))

    def liabilities_total(self) -> float:
        return float(sum(self.liabilities.values()))

    def equity(self, prices: Mapping[str, float] | pd.Series | None = None) -> float:
        """``cash + marked positions + receivables - liabilities`` (§3.6)."""
        return float(self.cash + self.positions_value(prices) + self.receivables_total() - self.liabilities_total())

    def carried_equity(self) -> float:
        """Equity at the marks the state already carries, with no new quotes.

        The valuation *between* two bars: after an event has been applied at the start of a
        bar and before that bar has printed anything. It is not a substitute for
        :meth:`equity` -- it never consults a quote, so it cannot be stale by accident, and
        a holding with no mark at all raises rather than valuing at zero.
        """
        total = self.cash + self.receivables_total() - self.liabilities_total()
        for sid, position in self.positions.items():
            if position.last_price is None:
                raise UnpricedPosition(f"{sid}: position has never been marked, so it has no carried value")
            total += position.shares * position.last_price
        return float(total)

    def weights(self, prices: Mapping[str, float] | pd.Series | None = None) -> pd.Series:
        """Marked weights, derived from the state and never stored as truth."""
        values = self.position_values(prices)
        if values.empty:
            return values.rename("weight")
        equity = self.equity(prices)
        if not math.isfinite(equity) or equity <= 0.0:
            raise AccountingError(
                f"weights are undefined at equity {equity!r}: a book with no positive equity has no "
                "fraction-of-equity representation"
            )
        return (values / equity).rename("weight")

    def short_collateral(self, prices: Mapping[str, float] | pd.Series | None = None) -> float:
        """``sum_i max(-shares_i, 0) * price_i`` -- the proceeds the broker holds against the shorts."""
        total = 0.0
        for sid, position in self.positions.items():
            if position.shares < 0.0:
                total += -position.shares * position.mark(prices, sid)
        return float(total)

    def committed_cash(self) -> float:
        """Cash already promised to pending buy orders, at their reference prices (§3.6)."""
        total = 0.0
        for order in self.pending_orders:
            if not order.is_open:
                continue
            remaining = order.remaining_qty
            if remaining > 0.0:
                total += remaining * order.intent.reference_price
        return float(total)

    def buying_power(self, prices: Mapping[str, float] | pd.Series | None = None) -> float:
        """``cash - committed_cash() - short_collateral(prices)`` (§3.6, §3.7 whole-batch validation)."""
        return float(self.cash - self.committed_cash() - self.short_collateral(prices))

    def open_orders(self) -> list[OrderRecord]:
        """Pending or partially filled orders, in submission order."""
        return [order for order in self.pending_orders if order.is_open]

    def order(self, client_order_id: str) -> OrderRecord | None:
        for record in self.pending_orders:
            if record.client_order_id == client_order_id:
                return record
        return None

    def unresolved_claims(self) -> list[str]:
        """Names whose valuation rests on an unresolved terminal claim (§3.2 strict claims)."""
        return sorted(sid for sid, p in self.positions.items() if p.valuation_status == "unresolved_claim")

    def stale_marks(self) -> list[str]:
        """Names carried at a mark older than the current bar."""
        return sorted(sid for sid, p in self.positions.items() if p.valuation_status == "stale")

    @property
    def is_incomplete(self) -> bool:
        """True when any record built on this state must be written ``incomplete`` (§3.2)."""
        return bool(self.unresolved_claims())

    # ------------------------------------------------------------------ mutation helpers

    def set_shares(self, sid: str, shares: float, *, price: float | None = None, bar: str | None = None) -> None:
        """Set a signed share count, creating or removing the position as needed.

        A position that reaches exactly zero shares is removed: an empty position carries
        no claim and no valuation, and keeping it would put a name with no exposure into
        every weight vector and every validation set.
        """
        quantity = _finite(shares, f"{sid}: shares")
        position = self.positions.get(sid)
        if quantity == 0.0:
            self.positions.pop(sid, None)
            return
        if position is None:
            position = Position(shares=quantity, last_price=price, last_price_bar=bar)
            self.positions[sid] = position
            return
        position.shares = quantity
        if price is not None:
            position.last_price = _finite(price, f"{sid}: price")
            position.last_price_bar = None if bar is None else bar_key(bar)

    def add_liability(self, key: str, amount: float) -> None:
        """Book (or add to) a liability. Amounts are positive magnitudes owed."""
        self.liabilities[key] = _nonnegative_claim(self.liabilities.get(key, 0.0) + float(amount), "liability", key)
        if self.liabilities[key] == 0.0:
            self.liabilities.pop(key)

    def add_receivable(self, key: str, amount: float) -> None:
        """Book (or add to) a receivable. Amounts are positive magnitudes owed to us."""
        self.receivables[key] = _nonnegative_claim(self.receivables.get(key, 0.0) + float(amount), "receivable", key)
        if self.receivables[key] == 0.0:
            self.receivables.pop(key)

    def copy(self) -> "PortfolioState":
        """A deep copy. Transitions build one of these, so a failed transition leaves the
        caller's state untouched rather than half-applied."""
        return PortfolioState(
            as_of=self.as_of,
            cash=self.cash,
            positions={sid: position.copy() for sid, position in self.positions.items()},
            liabilities=dict(self.liabilities),
            receivables=dict(self.receivables),
            pending_orders=[order.copy() for order in self.pending_orders],
            obligations=list(self.obligations),
            sequence=self.sequence,
        )

    # ------------------------------------------------------------------ durable form

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, for ``LoopState`` schema version 3 (decision D4)."""
        return {
            "as_of": self.as_of,
            "cash": self.cash,
            "positions": {
                sid: {
                    "shares": p.shares,
                    "last_price": p.last_price,
                    "last_price_bar": p.last_price_bar,
                    "valuation_status": p.valuation_status,
                }
                for sid, p in self.positions.items()
            },
            "liabilities": dict(self.liabilities),
            "receivables": dict(self.receivables),
            "pending_orders": [
                {
                    "intent": {
                        "sid": o.intent.sid,
                        "qty": o.intent.qty,
                        "increment": o.intent.increment,
                        "reference_price": o.intent.reference_price,
                        "decision_bar": o.intent.decision_bar,
                        "order_type": o.intent.order_type,
                        "time_in_force": o.intent.time_in_force,
                        "action_class": o.intent.action_class,
                    },
                    "client_order_id": o.client_order_id,
                    "status": o.status,
                    "filled_qty": o.filled_qty,
                    "fills": [vars(f) for f in o.fills],
                    "submitted_at": o.submitted_at,
                    "last_update": o.last_update,
                    "reason": o.reason,
                }
                for o in self.pending_orders
            ],
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioState":
        """Rebuild a state written by :meth:`to_dict`. Obligations are not carried here;
        they are persisted by ``portfolio/obligations.py`` alongside the state."""
        positions = {
            str(sid): Position(
                shares=row["shares"],
                last_price=row.get("last_price"),
                last_price_bar=row.get("last_price_bar"),
                valuation_status=row.get("valuation_status", "priced"),
            )
            for sid, row in dict(payload.get("positions", {})).items()
        }
        orders = []
        for row in payload.get("pending_orders", []):
            intent = OrderIntent(**row["intent"])
            orders.append(
                OrderRecord(
                    intent=intent,
                    client_order_id=row["client_order_id"],
                    status=row.get("status", "pending"),
                    filled_qty=row.get("filled_qty", 0.0),
                    fills=[Fill(**f) for f in row.get("fills", [])],
                    submitted_at=row.get("submitted_at", ""),
                    last_update=row.get("last_update", ""),
                    reason=row.get("reason"),
                )
            )
        return cls(
            as_of=payload["as_of"],
            cash=payload.get("cash", 0.0),
            positions=positions,
            liabilities=dict(payload.get("liabilities", {})),
            receivables=dict(payload.get("receivables", {})),
            pending_orders=orders,
            sequence=int(payload.get("sequence", 0)),
        )


def _nonnegative_claim(amount: float, what: str, key: str) -> float:
    value = _finite(amount, f"{what} {key!r}")
    if value < 0.0:
        raise AccountingError(
            f"{what} {key!r} is {value}: claims are held as positive magnitudes on their own side, "
            "so a negative one is a sign error, not a credit"
        )
    return value


def empty_state(as_of: str, cash: float = 0.0) -> PortfolioState:
    """A flat book with ``cash`` and nothing else."""
    return PortfolioState(as_of=as_of, cash=cash)


def state_from_shares(
    as_of: str,
    shares: Mapping[str, float] | pd.Series,
    prices: Mapping[str, float] | pd.Series,
    *,
    cash: float = 0.0,
) -> PortfolioState:
    """Build a marked state from a share vector and the bar that marks it.

    Every name in ``shares`` must have a price: a starting book with an unpriced holding
    has no equity, and guessing one is the failure this module exists to prevent.
    """
    items: Iterable[tuple[str, float]] = [(str(sid), float(qty)) for sid, qty in dict(shares).items()]
    state = PortfolioState(as_of=as_of, cash=cash)
    for sid, quantity in items:
        if float(quantity) == 0.0:
            continue
        price = _price_of(prices, sid)
        if price is None:
            raise UnpricedPosition(f"{sid}: opening a position needs a price for the bar it is marked at")
        state.positions[sid] = Position(
            shares=float(quantity), last_price=price, last_price_bar=bar_key(as_of), valuation_status="priced"
        )
    return state


def frame_of_positions(state: PortfolioState, prices: Mapping[str, float] | pd.Series | None = None) -> pd.DataFrame:
    """One row per held name: shares, mark, value, and valuation status."""
    rows = []
    for sid in state.sids():
        position = state.positions[sid]
        rows.append(
            {
                "sid": sid,
                "shares": position.shares,
                "mark": position.mark(prices, sid),
                "value": position.value(prices, sid),
                "valuation_status": position.valuation_status,
                "last_price_bar": position.last_price_bar,
            }
        )
    columns = ["sid", "shares", "mark", "value", "valuation_status", "last_price_bar"]
    return pd.DataFrame(rows, columns=columns)
