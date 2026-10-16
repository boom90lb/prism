"""Alpaca Trading API adapter (SPEC.md §7.4 live extension; §4 execution home).

The concrete :class:`~prism.live.broker.Broker` for the R2 paper-loop
instrument: a thin REST shell over Alpaca's v2 trading API, written against
an injectable requests-compatible session so every venue mapping —
positions, cash, idempotent submit, duplicate-id detection, fill retrieval
— is tested offline against canned API payloads (``tests/test_live_alpaca.py``).
Only the transport itself is network-gated.

Venue mappings that carry the contract:

* **Idempotency.** Alpaca rejects a reused ``client_order_id`` with HTTP 422
  ("client_order_id must be unique"); that rejection maps to
  :class:`DuplicateOrder`, which the loop treats as success.
* **Next-open fills (N2).** Orders default to market-on-open
  (``time_in_force="opg"``): decided after close *t*, they fill in the
  opening auction of *t+1*. OPG orders must be whole shares — size with
  ``targets_to_orders(..., whole_shares=True)``; a fractional quantity is
  rejected here loudly (N7), never silently rounded.
* **Signed quantities.** The Broker contract speaks signed shares; Alpaca
  speaks (unsigned qty, side). The adapter converts both ways, including
  short positions reported with a negative ``qty`` string.

* **Per-order type and time in force (§3.9).** The venue order type and TIF
  come from the order when it carries them (the canonical ``OrderIntent``:
  ``order_type`` ∈ {``market_on_open``, ``market_day``}), translated by the
  execution profile's ``[order_type_map]``; an order that carries neither takes
  the adapter's constructed default, which is what every existing caller passes
  and why the paper loop and the morning sweep behave exactly as before.

Section 3.9 additions: :meth:`AlpacaBroker.account_identity`,
:meth:`~AlpacaBroker.capabilities`, :meth:`~AlpacaBroker.buying_power`,
:meth:`~AlpacaBroker.open_orders`, :meth:`~AlpacaBroker.cancel`,
:meth:`~AlpacaBroker.fills_since`, and :meth:`~AlpacaBroker.server_time`.
:meth:`~AlpacaBroker.account_margin` is read-only telemetry
(:mod:`prism.live.margin`): the margin report and ``prism-doctor`` call it, and
nothing on the order path does.
Three rules bind them. There is **no universal $25,000 pattern-day-trading
rule** here: buying power is the number the venue reports, and an account-level
restriction is a `mandatory_action` limit derived from the account, not a
threshold this adapter invents. The adapter **reads no deprecated venue field**
(:data:`DEPRECATED_ASSET_FIELDS`). And auction eligibility and lot size are
*classifications* on the capabilities row, never assumed fills: a name the venue
admits to the opening auction is a name whose OPG order the venue will accept,
which is a different claim from "the auction will print it".

Credentials come from the standard Alpaca environment variables via
:meth:`AlpacaBroker.from_env`; they travel only in request headers and are
never interpolated into exceptions or logs.

**Authorization (§3.10).** Two rules bind this adapter, both of them refusals
and neither of them a warning.

*The mode is an argument, never an inheritance.* ``mode`` is ``paper`` or
``live`` and comes from ``--account``; ``APCA_API_BASE_URL`` may set an
*endpoint* but never the mode. The resolved base URL is checked against that
mode's allowlist — exact string equality against
:data:`~prism.live.authorization.PAPER_BASE_URL` under ``paper`` and
:data:`~prism.live.authorization.LIVE_BASE_URL` under ``live`` — where the
client is constructed, so a live endpoint under paper mode (or the reverse) is
refused before any request exists to send.

*A live client with no authorization record submits nothing.* :meth:`submit`
verifies the bound :class:`~prism.live.authorization.AuthorizationGate` before
it builds the request body: account id, endpoint, ``config_id``,
``engine_version``, execution host and build identity, each compared exactly. It
is the last line — ``live/loop.py`` refuses the same session earlier, before its
first broker read — and it exists because a last line that runs after the
request is built is not a last line.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Iterator, Mapping, Sequence

import pandas as pd

from prism.live.alpaca_transport import (
    DEFAULT_BACKOFF_BASE,
    DEFAULT_MAX_RETRIES,
    AlpacaAPIError,
    AlpacaSession,
    resolve_env_credentials,
)
from prism.live.authorization import (
    LIVE_BASE_URL,
    PAPER_BASE_URL,
    AuthorizationGate,
    AuthorizationRefused,
    resolve_endpoint,
)
from prism.live.broker import (
    AccountIdentity,
    Broker,
    CapabilityRow,
    DuplicateOrder,
    Fill,
    FillRecord,
    Order,
    OrderRejected,
    OrderStatus,
)
from prism.live.margin import AccountMargin

__all__ = [
    "AlpacaAPIError",  # canonical home is prism.live.alpaca_transport; re-exported for callers
    "AlpacaBroker",
    "CAPABILITY_SOURCE",
    "DEFAULT_ORDER_TYPE_MAP",
    "DEPRECATED_ACCOUNT_FIELDS",
    "DEPRECATED_ASSET_FIELDS",
    "INFORMATIONAL_ASSET_ATTRIBUTES",
    "LIVE_BASE_URL",
    "PAPER_BASE_URL",
    "RESTRICTING_ASSET_ATTRIBUTES",
]

logger = logging.getLogger(__name__)

# Calendar the fills ledger's `filled_bar` is expressed in (matches BAR_TZ of
# the bar store: one daily bar label per NY session).
_FILL_BAR_TZ = "America/New_York"

# GET /v2/orders page size cap per the API; fewer rows than this means the
# last page was reached.
_ORDERS_PAGE_LIMIT = 500

_VALID_TIME_IN_FORCE = ("opg", "day")

#: GET /v2/account/activities page size; a shorter page is the last one.
_ACTIVITIES_PAGE_LIMIT = 100

#: The engine order-type vocabulary mapped onto the venue's, i.e. the
#: ``[order_type_map]`` block of a reference execution profile (§3.9). The
#: adapter accepts the block itself through ``order_type_map=`` so the profile,
#: not this module, stays the declaration.
DEFAULT_ORDER_TYPE_MAP: dict[str, tuple[str, str]] = {
    "market_on_open": ("market", "opg"),
    "market_day": ("market", "day"),
}

#: The row source recorded on every capability row this adapter produces.
CAPABILITY_SOURCE = "alpaca"

#: Venue asset fields this adapter must not read (§3.9). Alpaca documents
#: ``maintenance_margin_requirement`` as superseded by the long/short pair;
#: reading a deprecated field means acting on a number the venue has stopped
#: maintaining. ``tests/test_live_alpaca.py`` asserts the module names them
#: nowhere but here.
DEPRECATED_ASSET_FIELDS: frozenset[str] = frozenset({"maintenance_margin_requirement"})

#: Account fields this adapter must not read. Alpaca documents
#: ``portfolio_value`` as deprecated and equal to ``equity``, which is read instead.
DEPRECATED_ACCOUNT_FIELDS: frozenset[str] = frozenset({"portfolio_value"})

#: The ``GET /v2/account`` fields an :class:`~prism.live.margin.AccountMargin` needs.
#: A read that lacks one raises (N7) rather than reporting a cushion it cannot compute.
_MARGIN_REQUIRED_FIELDS: tuple[str, ...] = (
    "equity",
    "cash",
    "long_market_value",
    "short_market_value",
    "initial_margin",
    "maintenance_margin",
    "regt_buying_power",
    "buying_power",
    "multiplier",
)

#: Numeric ``GET /v2/account`` fields carried onto the margin read when present.
_MARGIN_OPTIONAL_FIELDS: tuple[str, ...] = (
    "last_equity",
    "last_maintenance_margin",
    "non_marginable_buying_power",
    "sma",
)

#: Asset attributes that classify a name without restricting how it may be
#: traded. Everything *not* listed here and not in
#: :data:`RESTRICTING_ASSET_ATTRIBUTES` is carried onto the capability row
#: verbatim, where ``portfolio/limits.py`` treats an unrecognized restriction as
#: blocking new exposure and records it by name: an unknown broker restriction
#: is a reason to do less.
INFORMATIONAL_ASSET_ATTRIBUTES: frozenset[str] = frozenset(
    {"has_options", "options_enabled", "options_late_close", "fractional_eod_enabled"}
)

#: Venue attributes that map onto the declared restriction vocabulary of
#: ``prism.portfolio.limits.KNOWN_RESTRICTIONS``.
RESTRICTING_ASSET_ATTRIBUTES: dict[str, str] = {
    "ptp_no_exception": "no_new_exposure",
    "ptp_with_exception": "no_new_exposure",
}

#: Venue order states in which a cancel is a no-op rather than a request.
_TERMINAL_ORDER_STATUSES = frozenset(
    {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day", "replaced"}
)

#: The exchange string Alpaca uses for over-the-counter names, which do not
#: participate in a primary opening auction.
_OTC_EXCHANGE = "OTC"


class AlpacaBroker(Broker):
    """:class:`Broker` over the Alpaca v2 REST trading API (paper by default).

    Transport rides :class:`~prism.live.alpaca_transport.AlpacaSession` — the
    same bounded 429/5xx retry the bar source carries, because a transient
    venue hiccup at 18:30 must not cost a session that "cannot be backfilled
    as live evidence" (docs/operations.md). Retrying ``submit`` is safe here
    precisely because every order carries a deterministic ``client_order_id``:
    a retry of an already-accepted POST is a duplicate-id rejection, which the
    protocol treats as success.
    """

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        *,
        base_url: str = PAPER_BASE_URL,
        mode: str = "paper",
        authorization: AuthorizationGate | None = None,
        session: Any | None = None,
        time_in_force: str = "opg",
        order_type_map: Mapping[str, Mapping[str, str]] | None = None,
        clock: Callable[[], str] | None = None,
        timeout: float = 30.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        # §3.10: the per-mode endpoint allowlist, enforced where the client is
        # constructed. A live endpoint under paper mode, or the reverse, never
        # gets as far as owning a session.
        base_url = resolve_endpoint(mode, base_url=base_url)
        if time_in_force not in _VALID_TIME_IN_FORCE:
            raise ValueError(
                f"time_in_force must be one of {_VALID_TIME_IN_FORCE}, got {time_in_force!r}; "
                "'opg' is the N2 next-open convention, 'day' admits fractional shares"
            )
        self._api = AlpacaSession(
            key_id,
            secret_key,
            base_url,
            session=session,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            sleep=sleep,
        )
        self._mode = mode
        self._authorization = authorization
        self._time_in_force = time_in_force
        self._order_type_map = _parse_order_type_map(order_type_map)
        self._clock = clock or _now_iso

    @property
    def endpoint(self) -> str:
        """The client's own resolved base URL — the endpoint §3.10 compares.

        A plain attribute rather than a call, because the authorization gate
        reads it on the path that must not reach the venue.
        """
        return self._api.base_url

    @property
    def mode(self) -> str:
        """``paper`` or ``live``, as ``--account`` selected it (§3.14)."""
        return self._mode

    @property
    def authorization(self) -> AuthorizationGate | None:
        """The gate this client submits under, or ``None`` when unbound."""
        return self._authorization

    def bind_authorization(self, gate: AuthorizationGate) -> None:
        """Bind an authorization gate to this client after construction.

        The gate needs the broker-reported account id, which needs a client to
        read it with; binding afterwards is what breaks that circle. The read
        (``account_identity``) is not an order request, and the gate refuses
        every order path until it is verified.
        """
        self._authorization = gate

    @classmethod
    def from_env(cls, *, mode: str = "paper", base_url: str | None = None, **kwargs: Any) -> "AlpacaBroker":
        """Construct from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``.

        ``APCA_API_BASE_URL`` (or the ``base_url`` argument) supplies the
        endpoint. Missing credentials raise (N7) — a broker that silently
        constructed unauthenticated would fail one request later with a less
        actionable error.

        **The mode is never settable from the environment** (§3.10). ``mode``
        arrives here from ``--account paper|live`` and the resolved endpoint is
        then checked against that mode's allowlist by exact string equality:
        :data:`LIVE_BASE_URL` under ``paper`` is refused, and so is
        :data:`PAPER_BASE_URL` under ``live``. That holds however the endpoint
        arrived — an exported ``APCA_API_BASE_URL`` and an explicit
        ``base_url=`` argument are refused alike, because a live endpoint under
        paper mode is a live endpoint under paper mode whoever typed it. Routing
        real money is ``mode="live"``, and a live client submits nothing until
        an authorization record is bound to it.
        """
        key_id, secret_key = resolve_env_credentials("starting the loop")
        from_environment = os.environ.get("APCA_API_BASE_URL")
        resolved = resolve_endpoint(mode, base_url=base_url, from_environment=from_environment)
        return cls(key_id, secret_key, base_url=resolved, mode=mode, **kwargs)

    # ------------------------------------------------------------ transport

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._api.request(method, path, **kwargs)

    @staticmethod
    def _json_or_raise(response: Any, context: str) -> Any:
        return AlpacaSession.json_or_raise(response, context)

    # ------------------------------------------------------------- contract

    def positions(self) -> dict[str, float]:
        payload = self._json_or_raise(self._request("GET", "/v2/positions"), "GET /v2/positions")
        out: dict[str, float] = {}
        for row in payload:
            qty = float(row["qty"])  # Alpaca signs qty: shorts are negative
            if qty != 0.0:
                out[str(row["symbol"])] = qty
        return out

    def cash(self) -> float:
        payload = self._json_or_raise(self._request("GET", "/v2/account"), "GET /v2/account")
        return float(payload["cash"])

    def submit(self, order: Order) -> None:
        self._verify_authorized()
        client_order_id, symbol, signed_qty, venue_type, venue_tif = self._venue_order(order)
        qty = abs(signed_qty)
        if venue_tif == "opg" and abs(qty - round(qty)) > 1e-9:
            # The last-line guard §3.9 keeps: it raises before any request is made, so
            # the order's fate is known and the loop treats it as a validation failure
            # (drop the order, revalidate the batch), not as a transport failure.
            raise ValueError(
                f"OPG (next-open) orders require whole shares, got qty {signed_qty} for "
                f"{symbol!r}; size the book with whole_shares=True or use "
                "time_in_force='day' (N7 — the adapter never rounds an order silently)"
            )
        if abs(qty - round(qty)) > 1e-9 and venue_tif != "day":
            raise ValueError(
                f"fractional orders take time in force 'day', not {venue_tif!r} "
                f"({qty} shares of {symbol!r}); the fractional-short prohibition and fractional "
                "eligibility are enforced in portfolio/validate.py, which sees the positions (§3.9)"
            )
        body = {
            "symbol": symbol,
            "qty": self._format_qty(qty),
            "side": "buy" if signed_qty > 0 else "sell",
            "type": venue_type,
            "time_in_force": venue_tif,
            "client_order_id": client_order_id,
        }
        response = self._request("POST", "/v2/orders", json=body)
        if response.status_code == 422 and "client" in self._error_message(response).lower():
            # "client_order_id must be unique" — the order already exists,
            # which the write-ahead protocol treats as success.
            raise DuplicateOrder(client_order_id)
        if 400 <= response.status_code < 500 and response.status_code != 429:
            # A client-side rejection of this one order (e.g. 403 "insufficient
            # qty available" on a side-crossing order, a non-shortable name):
            # the venue is healthy, so this is an answer about the order rather
            # than about the venue, and the loop stops the batch there and
            # revalidates what the accepted legs imply (§3.7,
            # `prism.live.loop.submit_pending`). A 429 is NOT a rejection — it is a transient the transport
            # already retried; exhausted, it falls through with the 5xx class to
            # AlpacaAPIError and propagates so the write-ahead crash-resume
            # applies (skipping a rate-limited order as "rejected" would drop it
            # for the session on a healthy venue).
            raise OrderRejected(
                f"order {client_order_id} rejected: HTTP {response.status_code}: " f"{self._error_message(response)}"
            )
        self._json_or_raise(response, f"POST /v2/orders ({client_order_id})")

    def _verify_authorized(self) -> None:
        """Refuse before the request body exists (§3.10, the adapter's last line).

        An unbound client is permitted only under ``paper`` at the paper
        endpoint, which is the pre-authorization paper instrument this stage
        keeps working. A live client with no record refuses every order: state
        S4b permits nothing, C0 included.
        """
        gate = self._authorization
        if gate is None:
            if self._mode != "paper" or self._api.base_url != PAPER_BASE_URL:
                raise AuthorizationRefused(
                    f"no authorization record is bound to this client (mode {self._mode!r}, endpoint "
                    f"{self._api.base_url!r}); refusing before the order request is built. State S4b "
                    "permits nothing and every broker write is refused, C0 included (§3.10)"
                )
            return
        gate.verify_binding(observed_endpoint=self._api.base_url)

    def _iter_orders(self, status: str, after: str | None) -> Iterator[dict]:
        """Every order row for ``status``, paginated on ``submitted_at``.

        `after` filters strictly on submitted_at; REST-sequential submissions
        get distinct nanosecond stamps, so no order is skipped at the page
        boundary in practice.
        """
        while True:
            params: dict[str, Any] = {
                "status": status,
                "limit": _ORDERS_PAGE_LIMIT,
                "direction": "asc",
            }
            if after is not None:
                params["after"] = after
            page = self._json_or_raise(self._request("GET", "/v2/orders", params=params), f"GET /v2/orders ({status})")
            yield from page
            if len(page) < _ORDERS_PAGE_LIMIT:
                return
            after = str(page[-1]["submitted_at"])

    def submitted_order_ids(self, since: str | None = None) -> set[str]:
        """Every client_order_id the venue knows, optionally submitted after ``since``.

        ``since`` (an ISO date or timestamp) is the loop's scan anchor
        (``prism.live.loop._known_order_ids``, duck-typed like
        ``open_order_ids``): a pending set is bounded to one decision bar, so
        paging the account's entire order history on every submit pass is
        latency and rate-limit exposure that grow with account age. ``None``
        keeps the bare-contract full-history behavior.
        """
        ids: set[str] = set()
        for row in self._iter_orders("all", since):
            client_order_id = row.get("client_order_id")
            if client_order_id:
                ids.add(str(client_order_id))
        return ids

    def open_order_ids(self, client_order_ids: set[str]) -> set[str]:
        """Subset of the given client ids still *live* at the venue.

        The completion sweep's safety guard (``prism.live.loop.sweep_pending``):
        an order that is still queued for its auction must never be swept — the
        residual would double-execute when the auction prints. Alpaca's
        ``status=open`` filter covers every non-terminal state (new, accepted,
        pending_new, partially_filled with the remainder working).
        """
        ids: set[str] = set()
        for row in self._iter_orders("open", None):
            client_order_id = row.get("client_order_id")
            if client_order_id and str(client_order_id) in client_order_ids:
                ids.add(str(client_order_id))
        return ids

    def fills_for(self, client_order_ids: set[str]) -> list[Fill]:
        fills: list[Fill] = []
        for client_order_id in sorted(client_order_ids):
            response = self._request(
                "GET",
                "/v2/orders:by_client_order_id",
                params={"client_order_id": client_order_id},
            )
            if response.status_code == 404:
                continue  # venue does not know the id: absent = not filled
            payload = self._json_or_raise(response, f"GET /v2/orders:by_client_order_id ({client_order_id})")
            # A fill is any executed quantity — whether the order is terminally
            # `filled` or a *partial* under an `expired`/`canceled` parent. OPG
            # (opening-auction) orders routinely fill less than the whole size,
            # and the unfilled remainder then expires; what the I-9 ledger needs
            # is the shares that actually traded and their price, not the
            # order's terminal label. ``filled_qty == 0`` (still open, or a clean
            # zero-fill expiry) is absent, not an error — the loop's settle
            # tolerates it and reconciles the book to broker truth.
            filled_qty = float(payload.get("filled_qty") or 0.0)
            avg_price = payload.get("filled_avg_price")
            if filled_qty <= 0.0 or avg_price is None:
                continue
            signed_qty = filled_qty if payload["side"] == "buy" else -filled_qty
            filled_bar = str(pd.Timestamp(payload["filled_at"]).tz_convert(_FILL_BAR_TZ).date())
            fills.append(
                Fill(
                    client_order_id=client_order_id,
                    symbol=str(payload["symbol"]),
                    qty=signed_qty,
                    price=float(avg_price),
                    filled_bar=filled_bar,
                )
            )
        return fills

    # ------------------------------------------------- section 3.9 extensions

    def account_identity(self) -> AccountIdentity:
        """Which Alpaca account, at which endpoint, this adapter is bound to.

        ``endpoint`` is the client's own resolved base URL, because the question
        is whether this process pointed where the operator intended. ``is_paper``
        stays ``None`` unless the venue itself reports the flag: the paper host
        name is a naming convention, and a convention is not an authorization fact.
        """
        payload = self._json_or_raise(self._request("GET", "/v2/account"), "GET /v2/account")
        account_id = str(payload.get("account_number") or payload.get("id") or "")
        if not account_id:
            raise AlpacaAPIError("GET /v2/account returned no account_number or id", 200)
        is_paper = bool(payload["is_paper"]) if "is_paper" in payload else None
        return AccountIdentity(account_id=account_id, endpoint=self._api.base_url, is_paper=is_paper)

    def buying_power(self) -> float:
        """The venue's own buying power, in dollars.

        No pattern-day-trading arithmetic happens here and no $25,000 threshold
        exists in this module: the engine holds no universal rule, and an
        account-level restriction is a `mandatory_action` limit derived from the
        account record, not a number this adapter invents (§3.7, §3.9).
        """
        payload = self._json_or_raise(self._request("GET", "/v2/account"), "GET /v2/account")
        if "buying_power" not in payload:
            raise AlpacaAPIError("GET /v2/account returned no buying_power", 200)
        return float(payload["buying_power"])

    def account_margin(self) -> AccountMargin:
        """The account's equity, sides and margin figures from one ``GET /v2/account``. Read-only.

        Telemetry for the margin report and ``prism-doctor``; no order path calls
        it, so a venue that changes these fields cannot change a decision. Every
        number is the venue's own: the maintenance requirement is read, never
        computed from a rule this adapter would have to assume, because a house
        requirement can exceed the regulatory minimum. A missing or non-numeric
        required field raises (N7). The short side is stored as its magnitude
        (:mod:`prism.live.margin`).
        """
        payload = self._json_or_raise(self._request("GET", "/v2/account"), "GET /v2/account")
        return _account_margin(payload)

    def capabilities(self, sids: Sequence[str], as_of: str) -> list[CapabilityRow]:
        """Dated ``capabilities`` rows for ``sids`` (§3.2 table, §3.9 refresh).

        One row per name the venue knows, stamped ``valid_at = as_of`` — the
        instant the caller is asking about, which is also the row's derived
        availability per section 3.2 — and ``ingested_at`` at this workspace's
        clock. A name the venue does not know yields **no row**, not a permissive
        one: a sid without a dated row fails submission validation under
        ``require_capabilities`` instead of defaulting to fully tradable.
        """
        stamp = str(as_of).strip()
        if not stamp:
            raise ValueError("capabilities(as_of=...) needs the instant the rows are valid at; §3.2 dates every row")
        ingested_at = self._clock()
        rows: list[CapabilityRow] = []
        for sid in sids:
            symbol = str(sid)
            response = self._request("GET", f"/v2/assets/{symbol}")
            if response.status_code == 404:
                logger.warning(
                    "Alpaca knows no asset %s; writing no capability row for it (a name with no dated "
                    "row is refused at submission, never assumed tradable)",
                    symbol,
                )
                continue
            payload = self._json_or_raise(response, f"GET /v2/assets/{symbol}")
            rows.append(_capability_row(payload, sid=symbol, as_of=stamp, ingested_at=ingested_at))
        return rows

    def open_orders(self) -> list[OrderStatus]:
        """Every order live at the venue, with the quantities ``open_order_ids`` omits.

        Section 3.10 permits C0 (cancel-only) in states where nothing else is
        permitted, and a cancel needs the order's side and size to be recorded
        before it is issued; membership alone cannot support that row.
        """
        return [_order_status(row) for row in self._iter_orders("open", None)]

    def cancel(self, client_order_id: str) -> None:
        """Cancel one order by client id. Idempotent on a terminal or unknown id.

        Alpaca cancels by *venue* order id, so the client id is resolved first.
        An id the venue does not know, an order already terminal, and a venue
        that answers 422 ("order is not cancelable") are all no-ops: the
        post-condition — this order is not working — already holds.
        """
        response = self._request("GET", "/v2/orders:by_client_order_id", params={"client_order_id": client_order_id})
        if response.status_code == 404:
            logger.info("cancel: venue does not know %s; nothing to cancel", client_order_id)
            return
        payload = self._json_or_raise(response, f"GET /v2/orders:by_client_order_id ({client_order_id})")
        status = str(payload.get("status") or "")
        if status in _TERMINAL_ORDER_STATUSES:
            logger.info("cancel: %s is already %s at the venue; no request issued", client_order_id, status)
            return
        order_id = str(payload.get("id") or "")
        if not order_id:
            raise AlpacaAPIError(f"venue order {client_order_id} carries no id to cancel", 200)
        deleted = self._request("DELETE", f"/v2/orders/{order_id}")
        if 200 <= deleted.status_code < 300:
            return
        if deleted.status_code in (404, 422):
            # Raced to a terminal state between the lookup and the cancel: the
            # order is not working, which is exactly what was asked for.
            logger.info("cancel: %s was already terminal at the venue (HTTP %s)", client_order_id, deleted.status_code)
            return
        self._json_or_raise(deleted, f"DELETE /v2/orders ({client_order_id})")

    def fills_since(self, marker: str | None = None) -> tuple[list[FillRecord], str | None]:
        """Executions after ``marker``, and the marker to store after appending them.

        The cursor is the venue's own activity id, opaque to the engine (§3.11):
        it is passed back as ``page_token`` and never parsed. Rows come back in
        ascending order, one :class:`FillRecord` per *execution* rather than one
        aggregate per order, because the fills ledger is keyed by execution id.
        The caller appends and flushes **before** saving the returned marker; a
        crash between the two re-pulls the same executions and the key drops them.

        A fill activity that does not carry its ``client_order_id`` is resolved
        once per venue order id, cached for the call.
        """
        records: list[FillRecord] = []
        cursor = marker
        next_marker = marker
        client_ids: dict[str, str] = {}
        while True:
            params: dict[str, Any] = {"page_size": _ACTIVITIES_PAGE_LIMIT, "direction": "asc"}
            if cursor:
                params["page_token"] = cursor
            page = self._json_or_raise(
                self._request("GET", "/v2/account/activities/FILL", params=params),
                "GET /v2/account/activities/FILL",
            )
            for row in page:
                records.append(self._fill_record(row, client_ids))
                next_marker = str(row["id"])
            if len(page) < _ACTIVITIES_PAGE_LIMIT:
                break
            cursor = next_marker
        return records, next_marker

    def server_time(self) -> str:
        """The venue's clock, for the section 3.11 clock-health check."""
        payload = self._json_or_raise(self._request("GET", "/v2/clock"), "GET /v2/clock")
        timestamp = str(payload.get("timestamp") or "")
        if not timestamp:
            raise AlpacaAPIError("GET /v2/clock returned no timestamp", 200)
        return timestamp

    def _fill_record(self, row: dict, client_ids: dict[str, str]) -> FillRecord:
        execution_id = str(row["id"])
        qty = float(row["qty"])
        if not qty > 0.0:
            raise AlpacaAPIError(f"fill activity {execution_id} reports qty {row['qty']!r}", 200)
        side = str(row["side"])
        client_order_id = str(row.get("client_order_id") or "")
        if not client_order_id:
            order_id = str(row["order_id"])
            if order_id not in client_ids:
                order = self._json_or_raise(
                    self._request("GET", f"/v2/orders/{order_id}"), f"GET /v2/orders/{order_id}"
                )
                client_ids[order_id] = str(order.get("client_order_id") or "")
            client_order_id = client_ids[order_id]
        if not client_order_id:
            raise AlpacaAPIError(f"fill activity {execution_id} resolves to no client_order_id", 200)
        return FillRecord(
            execution_id=execution_id,
            client_order_id=client_order_id,
            sid=str(row["symbol"]),
            qty=qty if side == "buy" else -qty,
            price=float(row["price"]),
            filled_at=str(row["transaction_time"]),
        )

    def _venue_order(self, order: Any) -> tuple[str, str, float, str, str]:
        """(client_order_id, symbol, signed qty, venue order type, venue TIF) for one order.

        The engine's ``order_type`` and ``time_in_force`` come off the order when
        it carries them and are translated through ``[order_type_map]``; an order
        carrying neither — :class:`~prism.live.broker.Order`, which every current
        caller submits — takes the adapter's constructed default, so the paper
        loop and the morning sweep are byte-identical to before (§3.9).
        """
        carrier = getattr(order, "intent", None) or order
        client_order_id = str(getattr(order, "client_order_id", "") or getattr(carrier, "client_order_id", "") or "")
        if not client_order_id:
            raise ValueError(f"{order!r} carries no client_order_id; the idempotency key is not optional")
        symbol = getattr(carrier, "symbol", None) or getattr(carrier, "sid", None)
        if not symbol:
            raise ValueError(f"order {client_order_id} names neither a symbol nor a sid")
        qty = float(getattr(carrier, "qty"))
        engine_type = getattr(carrier, "order_type", None)
        tif = getattr(carrier, "time_in_force", None)
        if engine_type is None and tif is None:
            return client_order_id, str(symbol), qty, "market", self._time_in_force
        if engine_type is not None:
            try:
                venue_type, mapped_tif = self._order_type_map[str(engine_type)]
            except KeyError:
                raise ValueError(
                    f"order {client_order_id} declares order_type {engine_type!r}, which the "
                    f"[order_type_map] does not name (declared: {sorted(self._order_type_map)})"
                ) from None
            if tif is not None and str(tif) != mapped_tif:
                raise ValueError(
                    f"order {client_order_id} pairs order_type {engine_type!r} with time_in_force "
                    f"{tif!r}; [order_type_map] maps it to {mapped_tif!r}, and the adapter does not "
                    "resolve a contradiction it was handed"
                )
            return client_order_id, str(symbol), qty, venue_type, mapped_tif
        if str(tif) not in _VALID_TIME_IN_FORCE:
            raise ValueError(
                f"order {client_order_id} declares time_in_force {tif!r}, not one of {_VALID_TIME_IN_FORCE}"
            )
        return client_order_id, str(symbol), qty, "market", str(tif)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _format_qty(qty: float) -> str:
        if abs(qty - round(qty)) <= 1e-9:
            return str(int(round(qty)))
        return repr(qty)  # full float precision for fractional 'day' orders

    @staticmethod
    def _error_message(response: Any) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text or ""
        if isinstance(payload, dict):
            return str(payload.get("message", ""))
        return response.text or ""


def _parse_order_type_map(block: Mapping[str, Mapping[str, str]] | None) -> dict[str, tuple[str, str]]:
    """The profile's ``[order_type_map]`` block as {engine name: (venue type, venue TIF)}.

    ``None`` keeps :data:`DEFAULT_ORDER_TYPE_MAP`, which is the reference profile's
    block written out; the declaration lives in the file, and this is its parse.
    """
    if block is None:
        return dict(DEFAULT_ORDER_TYPE_MAP)
    parsed: dict[str, tuple[str, str]] = {}
    for name, entry in dict(block).items():
        row = dict(entry)
        try:
            venue_type = str(row["venue_order_type"])
            venue_tif = str(row["venue_time_in_force"])
        except KeyError as exc:
            raise ValueError(
                f"[order_type_map.{name}] needs venue_order_type and venue_time_in_force, missing {exc}"
            ) from None
        if venue_tif not in _VALID_TIME_IN_FORCE:
            raise ValueError(
                f"[order_type_map.{name}] maps to time in force {venue_tif!r}; this adapter submits "
                f"{list(_VALID_TIME_IN_FORCE)}"
            )
        parsed[str(name)] = (venue_type, venue_tif)
    if not parsed:
        raise ValueError("[order_type_map] declares no order types; an empty vocabulary submits nothing")
    return parsed


def _account_margin(payload: Mapping[str, Any]) -> AccountMargin:
    """One ``GET /v2/account`` payload as an :class:`~prism.live.margin.AccountMargin`.

    Alpaca sends every number as a string. Equity is ``cash + long_market_value
    + short_market_value`` in its definition, so the short side arrives
    negative; its magnitude is what is stored.
    """
    missing = [name for name in _MARGIN_REQUIRED_FIELDS if payload.get(name) in (None, "")]
    if missing:
        raise AlpacaAPIError(f"GET /v2/account returned no {', '.join(missing)}; the margin read needs them", 200)
    numbers = {name: _account_number(payload, name) for name in _MARGIN_REQUIRED_FIELDS}
    numbers["short_market_value"] = abs(numbers["short_market_value"])
    optional = {
        name: _account_number(payload, name) if payload.get(name) not in (None, "") else None
        for name in _MARGIN_OPTIONAL_FIELDS
    }
    shorting = payload.get("shorting_enabled")
    status = payload.get("status")
    balance_asof = payload.get("balance_asof")
    return AccountMargin(
        equity=numbers["equity"],
        cash=numbers["cash"],
        long_market_value=numbers["long_market_value"],
        short_market_value=numbers["short_market_value"],
        initial_margin=numbers["initial_margin"],
        maintenance_margin=numbers["maintenance_margin"],
        regt_buying_power=numbers["regt_buying_power"],
        buying_power=numbers["buying_power"],
        multiplier=numbers["multiplier"],
        last_equity=optional["last_equity"],
        last_maintenance_margin=optional["last_maintenance_margin"],
        non_marginable_buying_power=optional["non_marginable_buying_power"],
        sma=optional["sma"],
        shorting_enabled=None if shorting is None else bool(shorting),
        status=None if status in (None, "") else str(status),
        balance_asof=None if balance_asof in (None, "") else str(balance_asof),
    )


def _account_number(payload: Mapping[str, Any], name: str) -> float:
    """A numeric account field (sent as a string), or raise naming it (N7)."""
    value = payload[name]
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise AlpacaAPIError(f"GET /v2/account field {name} is not a number: {value!r}", 200) from None
    if not math.isfinite(number):
        raise AlpacaAPIError(f"GET /v2/account field {name} is not finite: {value!r}", 200)
    return number


def _now_iso() -> str:
    """This workspace's ingestion stamp for a capability row (NY, ISO 8601)."""
    return pd.Timestamp.now(tz=_FILL_BAR_TZ).isoformat()


def _order_status(row: Mapping[str, Any]) -> OrderStatus:
    """One ``GET /v2/orders`` row as an :class:`OrderStatus`.

    ``qty`` and ``filled_qty`` are both signed by side, so ``qty - filled_qty`` is
    the working remainder in the same convention the Broker contract speaks.
    ``order_type`` and ``time_in_force`` stay the venue's own strings: this is the
    venue's report of what it holds, and translating it here would hide a venue
    order the engine's vocabulary cannot express.
    """
    client_order_id = str(row.get("client_order_id") or "")
    if not client_order_id:
        raise AlpacaAPIError(f"venue order {row.get('id')!r} carries no client_order_id", 200)
    sign = 1.0 if str(row.get("side")) == "buy" else -1.0
    return OrderStatus(
        client_order_id=client_order_id,
        sid=str(row["symbol"]),
        qty=sign * float(row.get("qty") or 0.0),
        filled_qty=sign * float(row.get("filled_qty") or 0.0),
        order_type=str(row.get("type") or ""),
        time_in_force=str(row.get("time_in_force") or ""),
        status=str(row.get("status") or ""),
    )


def _capability_row(payload: Mapping[str, Any], *, sid: str, as_of: str, ingested_at: str) -> CapabilityRow:
    """One ``GET /v2/assets/{symbol}`` payload as a dated ``capabilities`` row.

    Every value is read from the venue or left at the fail-safe reading; nothing
    is inferred in the permissive direction. Specifically:

    * ``shortable`` and ``easy_to_borrow`` default to **false** when the payload
      omits them, so a silent omission never opens a short.
    * ``quantity_increment`` is one share. Alpaca states no per-asset share
      increment for US equities; a *fractionable* name's increment is the
      execution profile's ``[rounding] fractional_increment``, which is where
      section 3.7 puts it, derived by ``portfolio/validate.py`` from this row's
      ``fractionable`` flag and the profile together.
    * ``min_notional`` is 0.0: the venue publishes no per-asset dollar minimum
      for equities, and the profile's ``min_notional`` is a *policy* dust floor,
      a different quantity that this row must not restate as a venue fact.
    * ``lot_size`` is ``None`` — the venue reports none, and 100 would be a claim
      no source made. It is a round-lot classification either way, never the
      increment.
    * ``auction_eligible`` says the venue admits the name to the primary opening
      auction (active, tradable, not OTC). It is a classification, never a
      predicted fill.
    """
    asset_class = str(payload.get("class") or "us_equity")
    if asset_class != "us_equity":
        raise ValueError(
            f"{sid}: asset class {asset_class!r}; the capabilities vocabulary (regular sessions, an "
            "opening auction, share increments) is declared for US equities only (§3.2)"
        )
    status = str(payload.get("status") or "")
    tradable = bool(payload.get("tradable", False))
    active = tradable and status == "active"
    exchange = str(payload.get("exchange") or "")

    restrictions: list[str] = []
    if not active:
        restrictions.append("no_trade")
    for attribute in (str(item) for item in payload.get("attributes") or ()):
        if attribute in INFORMATIONAL_ASSET_ATTRIBUTES:
            continue
        restrictions.append(RESTRICTING_ASSET_ATTRIBUTES.get(attribute, attribute))

    auction_eligible = active and exchange != _OTC_EXCHANGE
    if auction_eligible:
        order_types: tuple[str, ...] = ("market_on_open", "market_day")
    elif active:
        order_types = ("market_day",)
    else:
        order_types = ()

    return CapabilityRow(
        sid=sid,
        valid_at=as_of,
        fractionable=bool(payload.get("fractionable", False)),
        quantity_increment=1.0,
        min_notional=0.0,
        shortable=bool(payload.get("shortable", False)),
        easy_to_borrow=bool(payload.get("easy_to_borrow", False)),
        lot_size=None,
        auction_eligible=auction_eligible,
        restrictions=tuple(dict.fromkeys(restrictions)),
        order_types=order_types,
        sessions=("regular",),
        available_at=as_of,
        ingested_at=ingested_at,
        source=CAPABILITY_SOURCE,
        revision=1,
    )
