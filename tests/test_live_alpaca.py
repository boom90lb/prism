"""Alpaca adapter venue mappings (SPEC §7.4), tested against canned payloads.

Every Broker-contract mapping — signed positions, cash, idempotent submit
with duplicate-id detection, order-id pagination, fill retrieval with NY
bar labels — runs offline through an injectable fake session; a stateful
mini-venue then proves the adapter satisfies the write-ahead protocol
end-to-end. Only the real HTTP transport is left network-gated.
"""

from __future__ import annotations

import json

import pytest

from prism.live import (
    AlpacaAPIError,
    AlpacaBroker,
    DuplicateOrder,
    LiveLoopContext,
    Order,
    OrderRejected,
    StateStore,
    decide_and_submit,
    read_fills_ledger,
    settle,
)
from prism.live.alpaca import PAPER_BASE_URL

KEY, SECRET = "test-key-id", "test-secret"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class FakeSession:
    """Routes (method, path) to a canned response or a handler(json, params)."""

    def __init__(self):
        self.routes = {}
        self.calls = []

    def route(self, method, path, response):
        self.routes[(method, path)] = response

    def request(self, method, url, headers=None, timeout=None, json=None, params=None):
        assert url.startswith(PAPER_BASE_URL)
        path = url[len(PAPER_BASE_URL) :]
        self.calls.append({"method": method, "path": path, "headers": headers, "json": json, "params": params})
        handler = self.routes[(method, path)]
        if callable(handler):
            return handler(json=json, params=params)
        return handler


@pytest.fixture()
def session():
    return FakeSession()


@pytest.fixture()
def broker(session):
    return AlpacaBroker(KEY, SECRET, session=session)


# ---------------------------------------------------------------------------
# Snapshot truth: positions and cash
# ---------------------------------------------------------------------------


def test_positions_signed_and_zero_dropped(broker, session) -> None:
    session.route(
        "GET",
        "/v2/positions",
        FakeResponse(
            payload=[
                {"symbol": "AAPL", "qty": "10"},
                {"symbol": "TSLA", "qty": "-2.5"},  # short: negative qty string
                {"symbol": "ZERO", "qty": "0"},
            ]
        ),
    )
    assert broker.positions() == {"AAPL": 10.0, "TSLA": -2.5}
    assert session.calls[0]["headers"]["APCA-API-KEY-ID"] == KEY
    assert session.calls[0]["headers"]["APCA-API-SECRET-KEY"] == SECRET


def test_cash_parsed_from_account(broker, session) -> None:
    session.route("GET", "/v2/account", FakeResponse(payload={"cash": "10000.50"}))
    assert broker.cash() == 10000.50


# ---------------------------------------------------------------------------
# Submit: body mapping, duplicate-id, errors, whole-share OPG rule
# ---------------------------------------------------------------------------


def _order(qty: float, symbol: str = "AAPL") -> Order:
    return Order(f"2026-07-06:{symbol}", symbol, qty, "2026-07-06", 100.0)


def test_submit_maps_signed_qty_to_side(broker, session) -> None:
    session.route("POST", "/v2/orders", FakeResponse(payload={"id": "x"}))
    broker.submit(_order(20.0))
    broker.submit(_order(-40.0, "MSFT"))
    buy, sell = session.calls[0]["json"], session.calls[1]["json"]
    assert buy == {
        "symbol": "AAPL",
        "qty": "20",
        "side": "buy",
        "type": "market",
        "time_in_force": "opg",
        "client_order_id": "2026-07-06:AAPL",
    }
    assert sell["side"] == "sell" and sell["qty"] == "40"


def test_duplicate_client_order_id_maps_to_DuplicateOrder(broker, session) -> None:
    session.route(
        "POST",
        "/v2/orders",
        FakeResponse(status_code=422, payload={"code": 40010001, "message": "client_order_id must be unique"}),
    )
    with pytest.raises(DuplicateOrder):
        broker.submit(_order(1.0))


def test_venue_error_raises_without_leaking_credentials(broker, session) -> None:
    session.route("POST", "/v2/orders", FakeResponse(status_code=500, payload={"message": "internal"}))
    with pytest.raises(AlpacaAPIError) as excinfo:
        broker.submit(_order(1.0))
    assert excinfo.value.status_code == 500
    assert SECRET not in str(excinfo.value) and KEY not in str(excinfo.value)


def test_client_side_rejection_maps_to_OrderRejected(broker, session) -> None:
    # 403 "insufficient qty available" — the 2026-07-08 ORCL side-crossing order.
    # A client-side rejection of one order maps to OrderRejected (the loop skips
    # it), unlike a 5xx/transport error which stays a fatal AlpacaAPIError.
    session.route(
        "POST",
        "/v2/orders",
        FakeResponse(status_code=403, payload={"message": "insufficient qty available for order"}),
    )
    with pytest.raises(OrderRejected) as excinfo:
        broker.submit(_order(1.0))
    assert SECRET not in str(excinfo.value) and KEY not in str(excinfo.value)


def test_opg_rejects_fractional_shares_loudly(broker) -> None:
    with pytest.raises(ValueError, match="whole shares"):
        broker.submit(_order(20.5))


def test_day_tif_admits_fractional_shares(session) -> None:
    broker = AlpacaBroker(KEY, SECRET, session=session, time_in_force="day")
    session.route("POST", "/v2/orders", FakeResponse(payload={"id": "x"}))
    broker.submit(_order(20.5))
    body = session.calls[0]["json"]
    assert body["qty"] == "20.5" and body["time_in_force"] == "day"


def test_invalid_time_in_force_rejected() -> None:
    with pytest.raises(ValueError, match="time_in_force"):
        AlpacaBroker(KEY, SECRET, time_in_force="gtc")


# ---------------------------------------------------------------------------
# Order listing (pagination) and fills
# ---------------------------------------------------------------------------


def test_submitted_order_ids_paginates(broker, session, monkeypatch) -> None:
    monkeypatch.setattr("prism.live.alpaca._ORDERS_PAGE_LIMIT", 2)
    pages = {
        None: [
            {"client_order_id": "d1:AAA", "submitted_at": "2026-07-06T20:01:00Z"},
            {"client_order_id": "d1:BBB", "submitted_at": "2026-07-06T20:02:00Z"},
        ],
        "2026-07-06T20:02:00Z": [{"client_order_id": "d1:CCC", "submitted_at": "2026-07-06T20:03:00Z"}],
    }

    def handler(json=None, params=None):
        assert params["status"] == "all" and params["direction"] == "asc"
        return FakeResponse(payload=pages[params.get("after")])

    session.route("GET", "/v2/orders", handler)
    assert broker.submitted_order_ids() == {"d1:AAA", "d1:BBB", "d1:CCC"}


def test_fills_for_maps_executed_quantity_including_partials(broker, session) -> None:
    payloads = {
        "d1:AAA": FakeResponse(
            payload={
                "status": "filled",
                "side": "buy",
                "symbol": "AAA",
                "filled_qty": "50",
                "filled_avg_price": "101.25",
                # 13:31Z = 09:31 America/New_York: same calendar day.
                "filled_at": "2026-07-07T13:31:00Z",
            }
        ),
        "d1:BBB": FakeResponse(
            payload={
                "status": "filled",
                "side": "sell",
                "symbol": "BBB",
                "filled_qty": "40",
                "filled_avg_price": "49.90",
                "filled_at": "2026-07-07T13:31:05Z",
            }
        ),
        # OPG remainder expired after a *partial* fill: the executed shares are a
        # real fill for the I-9 ledger even though the order's terminal label is
        # `expired`. This is exactly the ORCL case from the 2026-07-06 book.
        "d1:PART": FakeResponse(
            payload={
                "status": "expired",
                "side": "sell",
                "symbol": "D",
                "filled_qty": "1",
                "filled_avg_price": "138.93",
                "filled_at": "2026-07-08T13:30:26Z",
            }
        ),
        # OPG order that expired with nothing executed: absent, not a fill.
        "d1:ZERO": FakeResponse(
            payload={"status": "expired", "side": "buy", "symbol": "E", "filled_qty": "0"}
        ),
        "d1:OPEN": FakeResponse(payload={"status": "new", "side": "buy", "symbol": "C"}),
        "d1:GONE": FakeResponse(status_code=404, payload={"message": "order not found"}),
    }
    session.route(
        "GET",
        "/v2/orders:by_client_order_id",
        lambda json=None, params=None: payloads[params["client_order_id"]],
    )
    fills = broker.fills_for({"d1:AAA", "d1:BBB", "d1:PART", "d1:ZERO", "d1:OPEN", "d1:GONE"})
    by_id = {f.client_order_id: f for f in fills}
    # filled and partially-filled ids yield fills; open / zero-fill / unknown are
    # absent, not errors.
    assert set(by_id) == {"d1:AAA", "d1:BBB", "d1:PART"}
    assert by_id["d1:AAA"].qty == 50.0 and by_id["d1:AAA"].price == 101.25
    assert by_id["d1:BBB"].qty == -40.0  # sell side -> negative signed qty
    assert by_id["d1:PART"].qty == -1.0 and by_id["d1:PART"].price == 138.93
    assert by_id["d1:AAA"].filled_bar == "2026-07-07"
    assert by_id["d1:PART"].filled_bar == "2026-07-08"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_from_env_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="APCA_API_KEY_ID"):
        AlpacaBroker.from_env()


def test_from_env_reads_credentials_and_base_url(monkeypatch) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", SECRET)
    monkeypatch.setenv("APCA_API_BASE_URL", PAPER_BASE_URL)
    broker = AlpacaBroker.from_env()
    assert broker._base_url == PAPER_BASE_URL


# ---------------------------------------------------------------------------
# Contract compatibility: the adapter under the write-ahead protocol
# ---------------------------------------------------------------------------


class FakeAlpacaVenue(FakeSession):
    """A stateful mini-Alpaca: enough of the v2 API to run the loop against."""

    def __init__(self, cash: float = 10_000.0):
        super().__init__()
        self._cash = cash
        self._positions: dict[str, float] = {}
        self._orders: dict[str, dict] = {}

    def request(self, method, url, headers=None, timeout=None, json=None, params=None):
        path = url[len(PAPER_BASE_URL) :]
        self.calls.append({"method": method, "path": path, "json": json, "params": params})
        if (method, path) == ("GET", "/v2/account"):
            return FakeResponse(payload={"cash": str(self._cash)})
        if (method, path) == ("GET", "/v2/positions"):
            return FakeResponse(
                payload=[{"symbol": s, "qty": repr(q)} for s, q in self._positions.items() if q]
            )
        if (method, path) == ("POST", "/v2/orders"):
            if json["client_order_id"] in self._orders:
                return FakeResponse(status_code=422, payload={"message": "client_order_id must be unique"})
            self._orders[json["client_order_id"]] = dict(json, status="accepted")
            return FakeResponse(payload={"id": json["client_order_id"]})
        if (method, path) == ("GET", "/v2/orders"):
            return FakeResponse(
                payload=[{"client_order_id": cid, "submitted_at": "t"} for cid in self._orders]
            )
        if (method, path) == ("GET", "/v2/orders:by_client_order_id"):
            order = self._orders.get(params["client_order_id"])
            if order is None:
                return FakeResponse(status_code=404, payload={"message": "order not found"})
            return FakeResponse(payload=order)
        raise AssertionError(f"unrouted {method} {path}")

    def fill_all(self, prices: dict[str, float], filled_at: str) -> None:
        for order in self._orders.values():
            if order["status"] == "filled":
                continue
            qty = float(order["qty"])
            signed = qty if order["side"] == "buy" else -qty
            price = prices[order["symbol"]]
            self._positions[order["symbol"]] = self._positions.get(order["symbol"], 0.0) + signed
            self._cash -= signed * price
            order.update(
                status="filled",
                filled_qty=str(qty),
                filled_avg_price=str(price),
                filled_at=filled_at,
            )


def test_adapter_satisfies_write_ahead_protocol(tmp_path) -> None:
    import pandas as pd

    venue = FakeAlpacaVenue(cash=10_000.0)
    ctx = LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=AlpacaBroker(KEY, SECRET, session=venue),
        fills_ledger=tmp_path / "fills.jsonl",
    )
    prices = pd.Series({"AAA": 100.0})

    orders = decide_and_submit(
        ctx, "2026-07-06", pd.Series({"AAA": 0.5}), prices, whole_shares=True
    )
    assert [o.qty for o in orders] == [50.0]

    # A "restarted" decide for the same bar resumes the persisted decision;
    # the venue's own order list marks it known — exactly one order exists.
    decide_and_submit(ctx, "2026-07-06", pd.Series({"AAA": 0.9}), prices, whole_shares=True)
    assert len(venue._orders) == 1

    venue.fill_all({"AAA": 101.0}, filled_at="2026-07-07T13:31:00Z")
    fills = settle(ctx, "2026-07-06")
    assert len(fills) == 1 and fills[0].price == 101.0 and fills[0].filled_bar == "2026-07-07"

    ledger = read_fills_ledger(ctx.fills_ledger)
    assert ledger.iloc[0]["reference_price"] == 100.0  # arrival slippage recoverable (I-9)
    state = ctx.store.load()
    assert state.positions == {"AAA": 50.0}
    assert state.cash == pytest.approx(10_000.0 - 50.0 * 101.0)
