"""Durable live-loop state at schema version 3 (SPEC.md §7.7; spec §3.11, decision D4).

One JSON document per loop, written atomically (temp file + ``os.replace``)
so a crash mid-write can never leave a torn state file. Loading is
fail-loud (N7): a missing file is a legitimate "fresh loop" (``None``), but
a corrupt or schema-incompatible file raises — silently starting flat over
a real book is the defect this module exists to prevent.

**What version 3 changes.** The book itself is no longer a bare
``{symbol: shares}`` dict beside a float: it is the canonical
:class:`~prism.execution.portfolio_state.PortfolioState` of §3.6, which carries
each position's mark and valuation status, the claims a distribution books, the
submitted orders with the executions already applied to them, and the sequence
number. Version 3, not a widened version 2, because version 2 is on the live
book: shipping a different layout under one number would defeat the fail-loud
version check this module exists to enforce, and the failure mode is starting
flat over a real book.

``pending_decision_bar``, ``last_settled_bar`` and ``last_refresh_bar`` keep
their v2 names and meanings at the top level, because the write-ahead protocol
and the cadence gate read them directly. ``fills_marker`` is the venue's
monotone activity cursor (§3.11 append-then-advance), ``components`` carries
per-component due-ness and the fit reference across a restart, and
``decision_id`` / ``config_id`` / ``engine_version`` say which decision, under
which configuration, this state belongs to.

**The v2 vocabulary still works.** ``positions``, ``cash`` and
``pending_orders`` remain readable and assignable, as *views* onto the
portfolio rather than as second copies of it: the live loop, the doctor and the
paper-state assessment are written against them, and one book with two
representations is the bug this whole stage is about. Every write through a
view lands in the portfolio, and the document on disk holds only the portfolio.

**What a restart may read back.** Version 3 keeps a *status* per persisted
order, so a fate the engine established before it died — a cancellation the
venue confirmed, a terminal status it reported — survives the restart instead
of being re-derived from the venue on every pass.
:func:`record_order_status` is the one way that status is written, and
:attr:`LoopState.open_pending_orders` is the view a resumed submission reads:
an order whose fate has been settled is not re-offered.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from prism import __version__ as ENGINE_VERSION
from prism.execution.portfolio_state import (
    ORDER_STATUSES,
    OrderIntent,
    OrderRecord,
    PortfolioState,
    Position,
    bar_key,
    empty_state,
)
from prism.live.broker import Order

logger = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = 3

#: ``as_of`` of a book that has never been marked. A fresh loop holds no bar,
#: and :class:`PortfolioState` requires an ISO date, so the sentinel is a real
#: date that sorts before every session the engine can trade: a comparison
#: against a real bar can never mistake it for one.
UNANCHORED_BAR = "0001-01-01"

#: The quantity increment a v2 pending order is expressed at. The v2 document
#: records no increment (the venue's increment is a §3.9 capability), and the
#: live books size whole shares, so a whole-share order migrates at 1.0. A
#: fractional one is carried as a single indivisible lot instead — see
#: :func:`_intent_of_order`.
LEGACY_QUANTITY_INCREMENT = 1.0

#: ``fit_scope`` of a component in operation (§3.8): a trailing window ending at
#: the decision bar, which is what the live books do today.
FIT_SCOPE_TRAILING = "trailing_window"


@dataclass
class ComponentState:
    """One component's durable schedule state and fit reference (§3.11, §3.3).

    ``last_output_bar`` is the last bar an output was produced *including* an
    all-NaN one: it carries per-component due-ness across a restart and anchors
    the validity clock, and a restart that forgot it would refresh a monthly
    component on the day it came back. ``fitted_state_ref`` names the artifact
    under ``workspace.state / "fits"`` (:mod:`prism.fits`), null for a component
    declaring itself stateless.

    Two optional fields belong to the tranched paper book
    (:mod:`prism.live.tranches`) and are written only when set, so a document
    that never carried them is byte-identical: ``phase_offset_bars`` is the
    session-grid offset the component refreshes on, and ``sizing`` is the
    volatility-scale decision its sub-targets were sized with (scale, status,
    reason, sigma and the last good scale that bounds a later failure).
    """

    component_id: str
    last_output_bar: str | None = None
    decision_bar: str | None = None
    expires_bar: str | None = None
    expired: bool = False
    sub_targets: dict[str, float] = field(default_factory=dict)
    fitted_state_ref: str | None = None
    fitted_through_bar: str | None = None
    fit_scope: str = FIT_SCOPE_TRAILING
    phase_offset_bars: int | None = None
    sizing: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "component_id": self.component_id,
            "last_output_bar": self.last_output_bar,
            "decision_bar": self.decision_bar,
            "expires_bar": self.expires_bar,
            "expired": bool(self.expired),
            "sub_targets": {str(sid): float(weight) for sid, weight in self.sub_targets.items()},
            "fitted_state_ref": self.fitted_state_ref,
            "fitted_through_bar": self.fitted_through_bar,
            "fit_scope": self.fit_scope,
        }
        if self.phase_offset_bars is not None:
            payload["phase_offset_bars"] = int(self.phase_offset_bars)
        if self.sizing is not None:
            payload["sizing"] = dict(self.sizing)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentState":
        offset = payload.get("phase_offset_bars")
        sizing = payload.get("sizing")
        return cls(
            component_id=str(payload["component_id"]),
            last_output_bar=_opt_str(payload.get("last_output_bar")),
            decision_bar=_opt_str(payload.get("decision_bar")),
            expires_bar=_opt_str(payload.get("expires_bar")),
            expired=bool(payload.get("expired", False)),
            sub_targets={str(sid): float(w) for sid, w in dict(payload.get("sub_targets") or {}).items()},
            fitted_state_ref=_opt_str(payload.get("fitted_state_ref")),
            fitted_through_bar=_opt_str(payload.get("fitted_through_bar")),
            fit_scope=str(payload.get("fit_scope", FIT_SCOPE_TRAILING)),
            phase_offset_bars=None if offset is None else int(offset),
            sizing=None if sizing is None else dict(sizing),
        )


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _intent_of_order(order: Order) -> OrderIntent:
    """The canonical intent of one persisted live-loop order.

    The increment is the one thing the v2 order shape does not carry. A whole
    number of shares is expressed at 1.0, which is what both live books size at
    (``paper_loop.py`` passes ``whole_shares=(tif == "opg")``). A fractional
    quantity is expressed as a single lot of its own size and said out loud,
    because inventing a venue increment here would let a later step re-round a
    live order, and rounding a submitted order is how a book trades a quantity
    nobody decided.
    """
    quantity = float(order.qty)
    increment = LEGACY_QUANTITY_INCREMENT
    if abs(quantity - round(quantity)) > 1e-9:
        increment = abs(quantity)
        logger.warning(
            "order %s carries a fractional quantity (%s) and the persisted order shape declares no "
            "quantity increment; carrying it as one indivisible lot rather than inventing the venue's "
            "increment (§3.9 capabilities are the source of that number)",
            order.client_order_id,
            quantity,
        )
    return OrderIntent(
        sid=str(order.symbol),
        qty=quantity,
        increment=increment,
        reference_price=float(order.reference_price),
        decision_bar=str(order.decision_bar),
    )


def _order_of_record(record: OrderRecord) -> Order:
    """The v2 order view of a canonical order record."""
    return Order(
        client_order_id=record.client_order_id,
        symbol=record.intent.sid,
        qty=record.intent.qty,
        decision_bar=record.intent.decision_bar,
        reference_price=record.intent.reference_price,
    )


def _same_order(record: OrderRecord, order: Order) -> bool:
    """Whether a record already carries exactly this order (so its fills survive a view write)."""
    return (
        record.client_order_id == order.client_order_id
        and record.intent.sid == str(order.symbol)
        and record.intent.qty == float(order.qty)
        and record.intent.reference_price == float(order.reference_price)
        and record.intent.decision_bar == bar_key(order.decision_bar)
    )


class LoopState:
    """Everything the daily loop must not lose between processes (§3.11).

    ``pending_orders`` is the write-ahead record: orders decided at close *t*
    that may or may not have reached the broker yet. While it is non-empty, the
    decision for ``pending_decision_bar`` is immutable — a restarted loop
    resumes submission from this record, it never re-decides (re-deciding after
    a partial submit would double-trade).

    Not a dataclass, because the v2 field names are views onto ``portfolio``
    rather than fields, and they are accepted at construction so that every
    caller written against version 2 keeps working against one book.
    """

    __slots__ = (
        "portfolio",
        "decision_id",
        "config_id",
        "engine_version",
        "components",
        "hysteresis",
        "fills_marker",
        "pending_decision_bar",
        "last_settled_bar",
        "last_refresh_bar",
        "schema_version",
    )

    def __init__(
        self,
        portfolio: PortfolioState | None = None,
        *,
        decision_id: str | None = None,
        config_id: str = "",
        engine_version: str = "",
        components: Sequence[ComponentState] | None = None,
        hysteresis: Mapping[str, Any] | None = None,
        fills_marker: str | None = None,
        pending_decision_bar: str | None = None,
        last_settled_bar: str | None = None,
        last_refresh_bar: str | None = None,
        schema_version: int = STATE_SCHEMA_VERSION,
        positions: Mapping[str, float] | None = None,
        cash: float | None = None,
        pending_orders: Sequence[Order] | None = None,
    ) -> None:
        self.portfolio = portfolio if portfolio is not None else empty_state(UNANCHORED_BAR)
        self.decision_id = decision_id
        self.config_id = str(config_id)
        self.engine_version = str(engine_version)
        self.components = list(components or [])
        self.hysteresis = dict(hysteresis or {})
        self.fills_marker = fills_marker
        self.pending_decision_bar = pending_decision_bar
        self.last_settled_bar = last_settled_bar
        self.last_refresh_bar = last_refresh_bar
        self.schema_version = int(schema_version)
        if positions is not None:
            self.positions = dict(positions)
        if cash is not None:
            self.cash = float(cash)
        if pending_orders is not None:
            self.pending_orders = list(pending_orders)

    # ------------------------------------------------------------------ v2 views

    @property
    def positions(self) -> dict[str, float]:
        """Signed shares per name — the v2 view of ``portfolio.positions``."""
        return {sid: position.shares for sid, position in self.portfolio.positions.items()}

    @positions.setter
    def positions(self, value: Mapping[str, float]) -> None:
        """Adopt a share vector, keeping the mark of every name that stays held.

        A name arriving without a price is carried ``stale`` with no mark rather
        than at a fabricated one: the next mark prices it from the decision
        bar's closes, or the cycle refuses to trade and names it (§3.6).
        """
        current = self.portfolio.positions
        rebuilt: dict[str, Position] = {}
        for sid, shares in dict(value).items():
            key = str(sid)
            quantity = float(shares)
            held = current.get(key)
            if held is None:
                rebuilt[key] = Position(shares=quantity, valuation_status="stale")
            else:
                rebuilt[key] = Position(
                    shares=quantity,
                    last_price=held.last_price,
                    last_price_bar=held.last_price_bar,
                    valuation_status=held.valuation_status,
                )
        self.portfolio.positions = rebuilt

    @property
    def cash(self) -> float:
        return self.portfolio.cash

    @cash.setter
    def cash(self, value: float) -> None:
        self.portfolio.cash = float(value)

    @property
    def pending_orders(self) -> list[Order]:
        """The persisted decision in the v2 order shape."""
        return [_order_of_record(record) for record in self.portfolio.pending_orders]

    @pending_orders.setter
    def pending_orders(self, value: Sequence[Order]) -> None:
        """Replace the pending set, keeping the record of any order that is unchanged.

        An order whose id, name, quantity, price and decision bar all match the
        record already held keeps that record — with its status and the
        executions already applied to it. Rebuilding it from the view would
        discard fills the book has already booked.
        """
        held = {record.client_order_id: record for record in self.portfolio.pending_orders}
        records: list[OrderRecord] = []
        for order in value:
            existing = held.get(order.client_order_id)
            if existing is not None and _same_order(existing, order):
                records.append(existing)
                continue
            records.append(OrderRecord(intent=_intent_of_order(order), client_order_id=order.client_order_id))
        self.portfolio.pending_orders = records

    @property
    def open_pending_orders(self) -> list[Order]:
        """The pending orders still capable of filling, in the v2 order shape.

        Every order a v2 document carried is open, and every order the loop
        persists starts open, so this is :attr:`pending_orders` until something
        records a terminal fate against a record — which is what restart
        recovery does when it resolves an ambiguous submission by cancelling it
        (:func:`record_order_status`). Submitting from this view rather than the
        unfiltered one is what stops a resumed decision from re-offering an
        order whose fate has just been settled.
        """
        return [_order_of_record(record) for record in self.portfolio.pending_orders if record.is_open]

    # ------------------------------------------------------------------ durable form

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "portfolio": self.portfolio.to_dict(),
            "decision_id": self.decision_id,
            "config_id": self.config_id,
            "engine_version": self.engine_version,
            "components": [component.to_dict() for component in self.components],
            "hysteresis": dict(self.hysteresis),
            "fills_marker": self.fills_marker,
            "pending_decision_bar": self.pending_decision_bar,
            "last_settled_bar": self.last_settled_bar,
            "last_refresh_bar": self.last_refresh_bar,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LoopState":
        return cls(
            portfolio=PortfolioState.from_dict(payload["portfolio"]),
            decision_id=_opt_str(payload.get("decision_id")),
            config_id=str(payload.get("config_id", "")),
            engine_version=str(payload.get("engine_version", "")),
            components=[ComponentState.from_dict(row) for row in payload.get("components", [])],
            hysteresis=dict(payload.get("hysteresis") or {}),
            fills_marker=_opt_str(payload.get("fills_marker")),
            pending_decision_bar=_opt_str(payload.get("pending_decision_bar")),
            last_settled_bar=_opt_str(payload.get("last_settled_bar")),
            last_refresh_bar=_opt_str(payload.get("last_refresh_bar")),
            schema_version=int(payload.get("schema_version", STATE_SCHEMA_VERSION)),
        )

    @classmethod
    def from_json(cls, raw: str, **migration: Any) -> "LoopState":
        """Parse a persisted document of any supported version, migrating it forward.

        ``migration`` is forwarded to :func:`migrate_state_payload`, which is
        how a caller supplies the sid resolver, the declared components and the
        run's identity to a v2 upgrade.
        """
        return cls.from_dict(migrate_state_payload(json.loads(raw), **migration))

    # ------------------------------------------------------------------ dunder

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LoopState):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return (
            f"LoopState(schema_version={self.schema_version}, positions={len(self.portfolio.positions)}, "
            f"cash={self.portfolio.cash!r}, pending_orders={len(self.portfolio.pending_orders)}, "
            f"pending_decision_bar={self.pending_decision_bar!r}, last_settled_bar={self.last_settled_bar!r}, "
            f"last_refresh_bar={self.last_refresh_bar!r}, fills_marker={self.fills_marker!r})"
        )


def record_order_status(
    state: LoopState,
    client_order_ids: Iterable[str],
    status: str,
    *,
    reason: str | None = None,
    last_update: str | None = None,
) -> tuple[str, ...]:
    """Record a fate against persisted orders, and return the ids it changed.

    The engine learns an order's fate from the venue, and a restart that has to
    ask again is a restart that may act on a book it does not know. So a fate
    the engine has *established* — a cancellation it issued and the venue
    confirmed, a terminal status the venue reported — is written onto the
    order record and saved with the rest of the state, rather than being
    re-derived on every pass.

    Idempotent: an id already carrying ``status`` is not reported as changed, so
    a caller that runs twice writes the same document. An id the state does not
    carry is ignored — the caller may be resolving a set the book has already
    settled — and an unsupported status is refused rather than stored, because
    a status outside :data:`ORDER_STATUSES` would fail the record's own
    validation on the next load.
    """
    if status not in ORDER_STATUSES:
        raise ValueError(f"order status must be one of {ORDER_STATUSES}, got {status!r}")
    wanted = {str(client_order_id) for client_order_id in client_order_ids}
    changed: list[str] = []
    for record in state.portfolio.pending_orders:
        if record.client_order_id not in wanted or record.status == status:
            continue
        record.status = status  # type: ignore[assignment]  # checked against ORDER_STATUSES above
        if reason is not None:
            record.reason = reason
        if last_update is not None:
            record.last_update = str(last_update)
        changed.append(record.client_order_id)
    return tuple(changed)


def _v1_to_v2(payload: dict) -> dict:
    """v1 -> v2 added ``last_refresh_bar`` (the decision-cadence anchor).

    A v1 book predates the cadence gate, so it loads unanchored (the next cycle
    refreshes) without disturbing positions, cash, or pending orders.
    """
    payload["last_refresh_bar"] = None
    payload["schema_version"] = 2
    return payload


def _v2_to_v3(
    payload: dict,
    *,
    resolve_sid: Callable[[str], str | None] | None,
    components: Iterable[str],
    config_id: str,
    engine_version: str,
) -> dict:
    """v2 -> v3, field by field, per the table in §3.11.

    Nothing is invented and nothing is dropped: every position becomes an
    unmarked (``stale``) :class:`Position`, cash is copied, every pending order
    keeps its client order id verbatim — it is the idempotency key, and
    regenerating one would resubmit a live order — the three bar markers are
    copied, ``fills_marker`` starts from ``last_settled_bar``, and the claims,
    obligations and sequence a v2 document never held start empty.

    No price is invented anywhere: the migrated positions carry no mark, so the
    first mark after migration prices them from the decision bar's closes or
    the cycle refuses to trade and names the unpriced sids (§3.6).
    """
    settled = _opt_str(payload.get("last_settled_bar"))
    refreshed = _opt_str(payload.get("last_refresh_bar"))
    positions: dict[str, Any] = {}
    for symbol, shares in dict(payload.get("positions") or {}).items():
        sid = str(symbol)
        if resolve_sid is not None:
            resolved = resolve_sid(sid)
            if not resolved:
                raise ValueError(
                    f"state migration v2 -> v3: symbol {sid!r} resolves to no sid in the listings valid on "
                    f"{settled!r}; the migration stops rather than carrying a position whose identity is "
                    "unknown (§3.11)"
                )
            sid = str(resolved)
        if sid in positions:
            raise ValueError(
                f"state migration v2 -> v3: two v2 symbols resolve to sid {sid!r}; merging them would "
                "silently net two books' positions into one"
            )
        positions[sid] = {
            "shares": float(shares),
            "last_price": None,
            "last_price_bar": None,
            "valuation_status": "stale",
        }

    orders: list[dict[str, Any]] = []
    for row in payload.get("pending_orders") or []:
        order = Order(**row) if isinstance(row, Mapping) else row
        intent = _intent_of_order(order)
        sid = intent.sid
        if resolve_sid is not None:
            resolved = resolve_sid(sid)
            if not resolved:
                raise ValueError(
                    f"state migration v2 -> v3: pending order {order.client_order_id!r} names symbol "
                    f"{sid!r}, which resolves to no sid; the migration stops rather than carrying an order "
                    "whose identity is unknown (§3.11)"
                )
            intent = OrderIntent(
                sid=str(resolved),
                qty=intent.qty,
                increment=intent.increment,
                reference_price=intent.reference_price,
                decision_bar=intent.decision_bar,
            )
        orders.append(
            {
                "intent": {
                    "sid": intent.sid,
                    "qty": intent.qty,
                    "increment": intent.increment,
                    "reference_price": intent.reference_price,
                    "decision_bar": intent.decision_bar,
                    "order_type": intent.order_type,
                    "time_in_force": intent.time_in_force,
                    "action_class": intent.action_class,
                },
                "client_order_id": order.client_order_id,
                "status": "pending",
                "filled_qty": 0.0,
                "fills": [],
                "submitted_at": "",
                "last_update": "",
                "reason": None,
            }
        )

    migrated = {
        "schema_version": 3,
        "portfolio": {
            "as_of": bar_key(settled) if settled else UNANCHORED_BAR,
            "cash": float(payload.get("cash", 0.0)),
            "positions": positions,
            "liabilities": {},
            "receivables": {},
            "pending_orders": orders,
            "sequence": 0,
        },
        "decision_id": None,
        "config_id": str(config_id),
        "engine_version": str(engine_version),
        "components": [
            ComponentState(component_id=str(component_id), last_output_bar=refreshed).to_dict()
            for component_id in components
        ],
        "hysteresis": {},
        "fills_marker": settled,
        "pending_decision_bar": _opt_str(payload.get("pending_decision_bar")),
        "last_settled_bar": settled,
        "last_refresh_bar": refreshed,
    }
    logger.warning(
        "migrated live state v2 -> v3: %d position(s), cash %.2f, %d pending order(s), fills_marker %r "
        "(initialized from last_settled_bar); positions carry no mark until the next close prices them",
        len(positions),
        migrated["portfolio"]["cash"],
        len(orders),
        settled,
    )
    return migrated


def migrate_state_payload(
    payload: dict,
    *,
    resolve_sid: Callable[[str], str | None] | None = None,
    components: Iterable[str] = (),
    config_id: str = "",
    engine_version: str = ENGINE_VERSION,
) -> dict:
    """Upgrade a persisted state payload to ``STATE_SCHEMA_VERSION``, or raise (N7).

    The v1 -> v2 step is kept ahead of the v2 -> v3 one, so a v1 document
    reaches version 3 in two declared hops. Any other version is a hard stop —
    migrate explicitly, never silently start flat over a real book.

    ``resolve_sid`` maps a v2 symbol onto the sid of §3.2, resolved through the
    listings valid on ``last_settled_bar``; a symbol it cannot resolve stops the
    migration and is named. With no resolver the v2 symbol *is* the key, which
    is what the live books use end to end today (their broker, ledgers and
    state are all symbol-keyed); the identity mapping is recorded here rather
    than assumed silently. ``config_id`` and ``engine_version`` come from the
    run performing the migration and are recorded as migrated rather than
    decided: ``decision_id`` stays null.
    """
    version = payload.get("schema_version")
    if version == 1:
        payload = _v1_to_v2(payload)
        version = 2
    if version == 2:
        if resolve_sid is None:
            logger.info(
                "state migration v2 -> v3: no sid resolver supplied, so each v2 symbol is carried as its own "
                "key (the identity mapping the live books already use end to end)"
            )
        payload = _v2_to_v3(
            payload,
            resolve_sid=resolve_sid,
            components=components,
            config_id=config_id,
            engine_version=engine_version,
        )
        version = 3
    if version != STATE_SCHEMA_VERSION:
        raise ValueError(
            f"live state schema_version {version!r} != supported "
            f"{STATE_SCHEMA_VERSION}; migrate explicitly, do not start flat (N7)"
        )
    return payload


# The pre-stage-3 private name, kept for callers that reached for it.
_migrate_state_payload = migrate_state_payload


class StateStore:
    """Atomic-write JSON persistence for :class:`LoopState`."""

    def __init__(self, path: Path, **migration: Any) -> None:
        self.path = Path(path)
        self._migration = dict(migration)

    def load(self) -> LoopState | None:
        """The persisted state, ``None`` for a genuinely fresh loop.

        Anything else — unreadable file, bad JSON, wrong schema — raises.
        """
        if not self.path.exists():
            return None
        raw = self.path.read_text(encoding="utf-8")
        try:
            return LoopState.from_json(raw, **self._migration)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError(f"corrupt live state at {self.path}: {exc}; refusing to start flat (N7)") from exc

    def save(self, state: LoopState) -> None:
        """Write-then-rename so the file on disk is always a whole document.

        The parent directory is fsynced after the rename: without it a power
        loss can drop the rename itself, silently reverting the state one
        version — after a submit, that is a re-decide on the next run, the
        exact defect the write-ahead protocol exists to prevent.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(state.to_json())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)
        dir_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
