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

Credentials come from the standard Alpaca environment variables via
:meth:`AlpacaBroker.from_env`; they travel only in request headers and are
never interpolated into exceptions or logs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
import requests

from prism.live.broker import Broker, DuplicateOrder, Fill, Order

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"

# Calendar the fills ledger's `filled_bar` is expressed in (matches BAR_TZ of
# the bar store: one daily bar label per NY session).
_FILL_BAR_TZ = "America/New_York"

# GET /v2/orders page size cap per the API; fewer rows than this means the
# last page was reached.
_ORDERS_PAGE_LIMIT = 500

_VALID_TIME_IN_FORCE = ("opg", "day")


class AlpacaAPIError(RuntimeError):
    """A non-2xx venue response the adapter cannot map onto the contract.

    Carries ``status_code`` and the (truncated) response body. For ``submit``
    the loop's crash-safety semantics apply: the order may or may not have
    been accepted, and the write-ahead protocol retries it idempotently on
    the next pass.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class AlpacaBroker(Broker):
    """:class:`Broker` over the Alpaca v2 REST trading API (paper by default)."""

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        *,
        base_url: str = PAPER_BASE_URL,
        session: Any | None = None,
        time_in_force: str = "opg",
        timeout: float = 30.0,
    ) -> None:
        if not key_id or not secret_key:
            raise ValueError("Alpaca key_id and secret_key must be non-empty")
        if time_in_force not in _VALID_TIME_IN_FORCE:
            raise ValueError(
                f"time_in_force must be one of {_VALID_TIME_IN_FORCE}, got {time_in_force!r}; "
                "'opg' is the N2 next-open convention, 'day' admits fractional shares"
            )
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._base_url = base_url.rstrip("/")
        self._session = session if session is not None else requests.Session()
        self._time_in_force = time_in_force
        self._timeout = timeout

    @classmethod
    def from_env(cls, *, base_url: str | None = None, **kwargs: Any) -> "AlpacaBroker":
        """Construct from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``.

        ``APCA_API_BASE_URL`` (or the ``base_url`` argument) overrides the
        paper endpoint. Missing credentials raise (N7) — a broker that
        silently constructs unauthenticated would fail one request later
        with a less actionable error.
        """
        key_id = os.environ.get("APCA_API_KEY_ID", "")
        secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
        if not key_id or not secret_key:
            raise RuntimeError(
                "APCA_API_KEY_ID / APCA_API_SECRET_KEY are not set; "
                "export the paper-account credentials before starting the loop (N7)"
            )
        resolved = base_url or os.environ.get("APCA_API_BASE_URL") or PAPER_BASE_URL
        return cls(key_id, secret_key, base_url=resolved, **kwargs)

    # ------------------------------------------------------------ transport

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._session.request(
            method,
            self._base_url + path,
            headers=self._headers,
            timeout=self._timeout,
            **kwargs,
        )

    @staticmethod
    def _json_or_raise(response: Any, context: str) -> Any:
        if 200 <= response.status_code < 300:
            return response.json()
        body = (response.text or "")[:500]
        raise AlpacaAPIError(f"{context} -> HTTP {response.status_code}: {body}", response.status_code)

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
        qty = abs(order.qty)
        if self._time_in_force == "opg" and abs(qty - round(qty)) > 1e-9:
            raise ValueError(
                f"OPG (next-open) orders require whole shares, got qty {order.qty} for "
                f"{order.symbol!r}; size the book with whole_shares=True or use "
                "time_in_force='day' (N7 — the adapter never rounds an order silently)"
            )
        body = {
            "symbol": order.symbol,
            "qty": self._format_qty(qty),
            "side": "buy" if order.qty > 0 else "sell",
            "type": "market",
            "time_in_force": self._time_in_force,
            "client_order_id": order.client_order_id,
        }
        response = self._request("POST", "/v2/orders", json=body)
        if response.status_code == 422 and "client" in self._error_message(response).lower():
            # "client_order_id must be unique" — the order already exists,
            # which the write-ahead protocol treats as success.
            raise DuplicateOrder(order.client_order_id)
        self._json_or_raise(response, f"POST /v2/orders ({order.client_order_id})")

    def submitted_order_ids(self) -> set[str]:
        ids: set[str] = set()
        after: str | None = None
        while True:
            params: dict[str, Any] = {
                "status": "all",
                "limit": _ORDERS_PAGE_LIMIT,
                "direction": "asc",
            }
            if after is not None:
                params["after"] = after
            page = self._json_or_raise(
                self._request("GET", "/v2/orders", params=params), "GET /v2/orders"
            )
            for row in page:
                client_order_id = row.get("client_order_id")
                if client_order_id:
                    ids.add(str(client_order_id))
            if len(page) < _ORDERS_PAGE_LIMIT:
                return ids
            # `after` filters strictly on submitted_at; REST-sequential
            # submissions get distinct nanosecond stamps, so no order is
            # skipped at the page boundary in practice.
            after = str(page[-1]["submitted_at"])

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
            payload = self._json_or_raise(
                response, f"GET /v2/orders:by_client_order_id ({client_order_id})"
            )
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
            filled_bar = str(
                pd.Timestamp(payload["filled_at"]).tz_convert(_FILL_BAR_TZ).date()
            )
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
