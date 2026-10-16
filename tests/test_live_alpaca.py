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
    # sleep is injected away so tests that exercise (or exhaust) the transient
    # retry never wait on real backoff.
    return AlpacaBroker(KEY, SECRET, session=session, sleep=lambda wait: None)


class SequencedResponses:
    """A route handler that plays canned responses in order (retry tests)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, json=None, params=None):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


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


def test_open_order_ids_filters_to_requested_and_uses_open_status(broker, session) -> None:
    # The sweep's double-execution guard: only orders still LIVE at the venue
    # come back, and only from the requested id set.
    def handler(json=None, params=None):
        assert params["status"] == "open" and params["direction"] == "asc"
        return FakeResponse(
            payload=[
                {"client_order_id": "d1:AAA", "submitted_at": "2026-07-06T20:01:00Z"},
                {"client_order_id": "other-book-id", "submitted_at": "2026-07-06T20:02:00Z"},
            ]
        )

    session.route("GET", "/v2/orders", handler)
    assert broker.open_order_ids({"d1:AAA", "d1:BBB"}) == {"d1:AAA"}


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
        "d1:ZERO": FakeResponse(payload={"status": "expired", "side": "buy", "symbol": "E", "filled_qty": "0"}),
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
    assert broker._api.base_url == PAPER_BASE_URL


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
            return FakeResponse(payload=[{"symbol": s, "qty": repr(q)} for s, q in self._positions.items() if q])
        if (method, path) == ("POST", "/v2/orders"):
            if json["client_order_id"] in self._orders:
                return FakeResponse(status_code=422, payload={"message": "client_order_id must be unique"})
            self._orders[json["client_order_id"]] = dict(json, status="accepted")
            return FakeResponse(payload={"id": json["client_order_id"]})
        if (method, path) == ("GET", "/v2/orders"):
            return FakeResponse(payload=[{"client_order_id": cid, "submitted_at": "t"} for cid in self._orders])
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

    orders = decide_and_submit(ctx, "2026-07-06", pd.Series({"AAA": 0.5}), prices, whole_shares=True)
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


# ---------------------------------------------------------------------------
# Transient-failure policy: the broker retries what the bar source retries
# ---------------------------------------------------------------------------


def test_submit_retries_transient_429_then_succeeds(session) -> None:
    # One rate-limit response must not cost the session (docs/operations.md:
    # a failed session cannot be backfilled). The transport retries and the
    # order lands.
    waits: list[float] = []
    broker = AlpacaBroker(KEY, SECRET, session=session, backoff_base=0.25, sleep=waits.append)
    session.route(
        "POST",
        "/v2/orders",
        SequencedResponses(
            [
                FakeResponse(status_code=429, payload={"message": "rate limit exceeded"}),
                FakeResponse(status_code=200, payload={"id": "abc"}),
            ]
        ),
    )
    broker.submit(_order(1.0))
    posts = [c for c in session.calls if c["method"] == "POST"]
    assert len(posts) == 2  # first attempt + one retry
    assert waits == [0.25]


def test_submit_rate_limit_exhaustion_is_transient_not_rejected(session) -> None:
    # 429 is a 4xx, but it is NOT a per-order rejection: mapping it to
    # OrderRejected would skip the order for the session on a healthy venue.
    # Exhausted retries surface as AlpacaAPIError so the write-ahead
    # crash-resume applies.
    broker = AlpacaBroker(KEY, SECRET, session=session, max_retries=1, sleep=lambda w: None)
    session.route(
        "POST",
        "/v2/orders",
        FakeResponse(status_code=429, payload={"message": "rate limit exceeded"}),
    )
    with pytest.raises(AlpacaAPIError) as excinfo:
        broker.submit(_order(1.0))
    assert excinfo.value.status_code == 429


def test_positions_read_retries_transient_5xx(session) -> None:
    broker = AlpacaBroker(KEY, SECRET, session=session, backoff_base=0.0, sleep=lambda w: None)
    session.route(
        "GET",
        "/v2/positions",
        SequencedResponses(
            [
                FakeResponse(status_code=503, payload={"message": "unavailable"}),
                FakeResponse(status_code=200, payload=[]),
            ]
        ),
    )
    assert broker.positions() == {}
    assert len(session.calls) == 2


def test_retry_honors_retry_after_header(session) -> None:
    waits: list[float] = []
    broker = AlpacaBroker(KEY, SECRET, session=session, backoff_base=0.1, sleep=waits.append)
    throttled = FakeResponse(status_code=429, payload={"message": "rate limit"})
    throttled.headers = {"Retry-After": "2.5"}
    session.route(
        "GET",
        "/v2/account",
        SequencedResponses([throttled, FakeResponse(status_code=200, payload={"cash": "100.0"})]),
    )
    assert broker.cash() == 100.0
    assert waits == [2.5]


def test_submitted_order_ids_since_maps_to_after_param(broker, session) -> None:
    seen_params: list[dict] = []

    def handler(json=None, params=None):
        seen_params.append(dict(params))
        return FakeResponse(payload=[{"client_order_id": "mom:2026-07-13:AAA", "submitted_at": "t1"}])

    session.route("GET", "/v2/orders", handler)
    assert broker.submitted_order_ids(since="2026-07-13") == {"mom:2026-07-13:AAA"}
    assert seen_params[0]["after"] == "2026-07-13"
    # Bare-contract call stays full-history: no `after` filter.
    seen_params.clear()
    assert broker.submitted_order_ids() == {"mom:2026-07-13:AAA"}
    assert "after" not in seen_params[0]


# ---------------------------------------------------------------------------
# §3.9: per-order type and time in force
# ---------------------------------------------------------------------------


class _Intent:
    """The shape the canonical OrderIntent presents to the adapter (sid + type + TIF)."""

    def __init__(self, client_order_id, sid, qty, order_type, time_in_force):
        self.client_order_id = client_order_id
        self.sid = sid
        self.qty = qty
        self.order_type = order_type
        self.time_in_force = time_in_force


def test_order_type_comes_from_the_order_not_the_construction_default(session) -> None:
    # The adapter is constructed for the auction; one order in the batch declares
    # market_day and is submitted as a day order. Before §3.9 this needed a second
    # broker instance (scripts/paper_sweep.py builds one with time_in_force="day").
    broker = AlpacaBroker(KEY, SECRET, session=session, sleep=lambda w: None)
    session.route("POST", "/v2/orders", FakeResponse(payload={"id": "x"}))
    broker.submit(_Intent("d1:AAA", "AAA", 20.0, "market_on_open", "opg"))
    broker.submit(_Intent("d1:BBB", "BBB", -5.5, "market_day", "day"))
    auction, day = session.calls[0]["json"], session.calls[1]["json"]
    assert auction["time_in_force"] == "opg" and auction["type"] == "market" and auction["qty"] == "20"
    assert day["time_in_force"] == "day" and day["side"] == "sell" and day["qty"] == "5.5"


def test_an_order_carrying_neither_keeps_the_constructed_default(broker, session) -> None:
    # The live path: prism.live.broker.Order has no order_type and no TIF, so the
    # paper loop's invocation is byte-identical to before the correction.
    session.route("POST", "/v2/orders", FakeResponse(payload={"id": "x"}))
    broker.submit(_order(3.0))
    assert session.calls[0]["json"]["time_in_force"] == "opg"


def test_undeclared_order_type_is_refused_rather_than_guessed(broker) -> None:
    with pytest.raises(ValueError, match="order_type_map"):
        broker.submit(_Intent("d1:AAA", "AAA", 1.0, "limit_close", "day"))


def test_contradictory_type_and_tif_are_refused(broker) -> None:
    with pytest.raises(ValueError, match="time_in_force"):
        broker.submit(_Intent("d1:AAA", "AAA", 1.0, "market_on_open", "day"))


def test_profile_order_type_map_is_the_declaration(session) -> None:
    from prism.live.alpaca import DEFAULT_ORDER_TYPE_MAP

    block = {
        "market_on_open": {"venue_order_type": "market", "venue_time_in_force": "opg"},
        "market_day": {"venue_order_type": "market", "venue_time_in_force": "day"},
    }
    broker = AlpacaBroker(KEY, SECRET, session=session, order_type_map=block)
    session.route("POST", "/v2/orders", FakeResponse(payload={"id": "x"}))
    broker.submit(_Intent("d1:AAA", "AAA", 1.0, "market_day", "day"))
    assert session.calls[0]["json"]["time_in_force"] == "day"
    # The shipped default is that same block, so the two agree by construction.
    assert DEFAULT_ORDER_TYPE_MAP["market_day"] == ("market", "day")


def test_fractional_never_leaves_the_process_without_day_tif(broker, session) -> None:
    # Both guards raise *before* any request, so the order's fate is known and the
    # loop treats it as a validation failure rather than a transport failure (§3.9).
    with pytest.raises(ValueError, match="whole shares"):
        broker.submit(_Intent("d1:AAA", "AAA", 20.5, "market_on_open", "opg"))
    assert session.calls == []


# ---------------------------------------------------------------------------
# §3.9: account identity, buying power, clock
# ---------------------------------------------------------------------------


def test_account_identity_reports_the_client_endpoint_and_no_inferred_paper_flag(broker, session) -> None:
    session.route(
        "GET",
        "/v2/account",
        FakeResponse(payload={"account_number": "PA123", "cash": "1.0", "buying_power": "2.0"}),
    )
    identity = broker.account_identity()
    assert identity.account_id == "PA123"
    assert identity.endpoint == PAPER_BASE_URL
    # The venue reports no paper flag, and the *hostname* is not evidence of one.
    assert identity.is_paper is None


def test_account_identity_reads_a_venue_reported_paper_flag(broker, session) -> None:
    session.route("GET", "/v2/account", FakeResponse(payload={"id": "abc", "is_paper": True}))
    assert broker.account_identity().is_paper is True


def test_buying_power_is_the_venue_number_with_no_pattern_day_trading_rule(broker, session) -> None:
    # An account below $25,000 that the venue flags as a pattern day trader still
    # gets the venue's own buying power: the engine holds no universal PDT rule
    # and this adapter invents no threshold (§3.9).
    session.route(
        "GET",
        "/v2/account",
        FakeResponse(
            payload={
                "account_number": "PA1",
                "cash": "9000.00",
                "equity": "9000.00",
                "buying_power": "18000.00",
                "pattern_day_trader": True,
                "daytrade_count": 7,
            }
        ),
    )
    assert broker.buying_power() == 18000.0


def test_no_pattern_day_trading_threshold_is_encoded_in_the_module() -> None:
    from pathlib import Path

    import prism.live.alpaca as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "25000" not in source and "25_000" not in source


def test_the_adapter_reads_no_deprecated_venue_field() -> None:
    from pathlib import Path

    import prism.live.alpaca as module

    lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    for field in module.DEPRECATED_ASSET_FIELDS:
        mentions = [line for line in lines if field in line]
        # It is named in its own declaration and in the comment above it, and nowhere
        # that reads a venue payload: acting on a field the venue has stopped
        # maintaining is acting on a number nobody maintains.
        assert mentions, field
        assert not [line for line in mentions if "payload" in line], field


#: A ``GET /v2/account`` body in Alpaca's encoding (numbers as strings), for a
#: long/short book at gross about 1. Shaped on the documented account fields;
#: not a capture from a real account. Equity = cash + long + short, so the short
#: side is negative; ``portfolio_value`` is deliberately different from equity.
MARGIN_ACCOUNT = {
    "account_number": "PA0TEST",
    "status": "ACTIVE",
    "cash": "950000.00",
    "equity": "944445.12",
    "portfolio_value": "1.00",
    "long_market_value": "472000.00",
    "short_market_value": "-477554.88",
    "initial_margin": "474777.44",
    "maintenance_margin": "261266.46",
    "last_maintenance_margin": "259000.00",
    "last_equity": "940000.00",
    "regt_buying_power": "939335.36",
    "buying_power": "939335.36",
    "non_marginable_buying_power": "469667.68",
    "multiplier": "2",
    "sma": "0",
    "shorting_enabled": True,
    "balance_asof": "2026-10-15",
}


def test_account_margin_maps_the_account_payload(broker, session) -> None:
    session.route("GET", "/v2/account", FakeResponse(payload=MARGIN_ACCOUNT))
    margin = broker.account_margin()
    assert [(call["method"], call["path"]) for call in session.calls] == [("GET", "/v2/account")]
    assert margin.equity == pytest.approx(944_445.12)  # equity, never the deprecated portfolio_value
    assert margin.cash == pytest.approx(950_000.0)
    assert margin.long_market_value == pytest.approx(472_000.0)
    assert margin.short_market_value == pytest.approx(477_554.88)  # the magnitude of the negative short side
    assert margin.initial_margin == pytest.approx(474_777.44)
    assert margin.maintenance_margin == pytest.approx(261_266.46)
    assert margin.regt_buying_power == pytest.approx(939_335.36) and margin.multiplier == 2.0
    assert margin.last_maintenance_margin == pytest.approx(259_000.0) and margin.sma == 0.0
    assert margin.shorting_enabled is True and margin.status == "ACTIVE" and margin.balance_asof == "2026-10-15"
    assert margin.cushion == pytest.approx((944_445.12 - 261_266.46) / 944_445.12)
    assert margin.gross_over_equity == pytest.approx((472_000.0 + 477_554.88) / 944_445.12)


def test_account_margin_leaves_optional_fields_none_when_the_venue_omits_them(broker, session) -> None:
    required = {
        key: MARGIN_ACCOUNT[key]
        for key in (
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
    }
    session.route("GET", "/v2/account", FakeResponse(payload=required))
    margin = broker.account_margin()
    assert margin.last_equity is None and margin.sma is None and margin.shorting_enabled is None
    assert margin.status is None and margin.balance_asof is None


@pytest.mark.parametrize("field", ["maintenance_margin", "regt_buying_power", "short_market_value", "equity"])
def test_account_margin_refuses_a_payload_without_a_required_field(broker, session, field) -> None:
    payload = {key: value for key, value in MARGIN_ACCOUNT.items() if key != field}
    session.route("GET", "/v2/account", FakeResponse(payload=payload))
    with pytest.raises(AlpacaAPIError, match=field):
        broker.account_margin()


def test_account_margin_refuses_a_field_that_is_not_a_number(broker, session) -> None:
    session.route("GET", "/v2/account", FakeResponse(payload={**MARGIN_ACCOUNT, "maintenance_margin": "n/a"}))
    with pytest.raises(AlpacaAPIError, match="maintenance_margin"):
        broker.account_margin()


def test_the_adapter_reads_no_deprecated_account_field() -> None:
    from pathlib import Path

    import prism.live.alpaca as module

    lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    for field in module.DEPRECATED_ACCOUNT_FIELDS:
        mentions = [line for line in lines if field in line]
        assert mentions, field
        assert not [line for line in mentions if "payload" in line], field


def test_only_the_margin_report_and_the_doctor_call_the_margin_read() -> None:
    # The margin read is telemetry. A call to it from any other module of the
    # shipped package is a decision path reading margin.
    import re
    from pathlib import Path

    import prism

    root = Path(prism.__file__).resolve().parent
    call = re.compile(r"\.account_margin\(")
    calling = sorted(
        str(path.relative_to(root)) for path in root.rglob("*.py") if call.search(path.read_text(encoding="utf-8"))
    )
    assert calling == ["scripts/doctor.py", "scripts/margin_report.py"]


class _MarginRefusingVenue(FakeAlpacaVenue):
    """The mini-venue with buying power reported, under an adapter whose margin read must never run."""

    def request(self, method, url, headers=None, timeout=None, json=None, params=None):
        if (method, url[len(PAPER_BASE_URL) :]) == ("GET", "/v2/account"):
            self.calls.append({"method": method, "path": "/v2/account", "json": json, "params": params})
            return FakeResponse(payload={"cash": str(self._cash), "buying_power": str(2 * self._cash)})
        return super().request(method, url, headers=headers, timeout=timeout, json=json, params=params)


def test_a_decision_settlement_and_buying_power_check_never_read_the_margin(tmp_path, monkeypatch) -> None:
    import pandas as pd

    def refuse(self):
        raise AssertionError("the order path read the account margin")

    monkeypatch.setattr(AlpacaBroker, "account_margin", refuse)
    venue = _MarginRefusingVenue(cash=10_000.0)
    ctx = LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=AlpacaBroker(KEY, SECRET, session=venue),
        fills_ledger=tmp_path / "fills.jsonl",
    )
    prices = pd.Series({"AAA": 100.0})
    orders = decide_and_submit(
        ctx, "2026-07-06", pd.Series({"AAA": 0.5}), prices, whole_shares=True, buying_power_check=True
    )
    assert [o.qty for o in orders] == [50.0]
    venue.fill_all({"AAA": 101.0}, filled_at="2026-07-07T13:31:00Z")
    assert len(settle(ctx, "2026-07-06")) == 1


def test_server_time_is_the_venue_clock(broker, session) -> None:
    session.route("GET", "/v2/clock", FakeResponse(payload={"timestamp": "2026-09-08T18:30:00-04:00"}))
    assert broker.server_time() == "2026-09-08T18:30:00-04:00"


def test_from_env_refuses_to_select_the_live_endpoint(monkeypatch) -> None:
    """§3.10 (stage 5 correction): the *mode* is the argument, not the endpoint.

    Draft 1.1's rule was "the environment may not name the live endpoint, but an
    explicit ``base_url=`` may". Stage 5 replaces it: a live endpoint under paper
    mode is refused however it arrived, because paper mode is what the operator
    selected and an argument is not a mode. Routing real money is
    ``mode="live"``, and that client still submits nothing without a record.
    """
    from prism.live.alpaca import LIVE_BASE_URL

    monkeypatch.setenv("APCA_API_KEY_ID", KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", SECRET)
    monkeypatch.setenv("APCA_API_BASE_URL", LIVE_BASE_URL)
    with pytest.raises(RuntimeError, match="live endpoint"):
        AlpacaBroker.from_env()
    # An explicit argument does not become a mode either.
    with pytest.raises(RuntimeError, match="live endpoint"):
        AlpacaBroker.from_env(base_url=LIVE_BASE_URL)
    # `--account live` is how real money is routed, and that still works.
    assert AlpacaBroker.from_env(mode="live", base_url=LIVE_BASE_URL)._api.base_url == LIVE_BASE_URL


# ---------------------------------------------------------------------------
# §3.2/§3.9: the dated capabilities table
# ---------------------------------------------------------------------------

AS_OF = "2026-09-08T16:00:00-04:00"


def _asset(symbol: str, **overrides):
    payload = {
        "symbol": symbol,
        "class": "us_equity",
        "exchange": "NASDAQ",
        "status": "active",
        "tradable": True,
        "shortable": True,
        "easy_to_borrow": True,
        "fractionable": True,
        "marginable": True,
        "attributes": [],
    }
    payload.update(overrides)
    return FakeResponse(payload=payload)


def _capabilities(broker, session, symbol, response):
    session.route("GET", f"/v2/assets/{symbol}", response)
    return broker.capabilities([symbol], AS_OF)


def test_capability_row_carries_the_section_32_fields(session) -> None:
    broker = AlpacaBroker(KEY, SECRET, session=session, clock=lambda: "2026-09-08T18:31:00-04:00")
    (row,) = _capabilities(broker, session, "AAPL", _asset("AAPL"))
    assert (row.sid, row.valid_at, row.available_at) == ("AAPL", AS_OF, AS_OF)
    assert row.ingested_at == "2026-09-08T18:31:00-04:00"
    assert (row.source, row.revision) == ("alpaca", 1)
    assert (row.fractionable, row.shortable, row.easy_to_borrow) == (True, True, True)
    assert row.order_types == ("market_on_open", "market_day") and row.sessions == ("regular",)
    assert row.auction_eligible is True
    assert row.restrictions == ()
    # Classifications, never fills or increments: the venue publishes no lot size,
    # and the fractional increment is the execution profile's, not the venue's.
    assert row.lot_size is None
    assert row.quantity_increment == 1.0


def test_an_omitted_shortable_flag_never_authorizes_a_short(broker, session) -> None:
    payload = _asset("XYZ")
    del payload._payload["shortable"]
    del payload._payload["easy_to_borrow"]
    (row,) = _capabilities(broker, session, "XYZ", payload)
    assert row.shortable is False and row.easy_to_borrow is False


def test_a_halted_name_carries_no_trade_and_no_order_types(broker, session) -> None:
    (row,) = _capabilities(broker, session, "HALT", _asset("HALT", status="inactive", tradable=False))
    assert row.restrictions == ("no_trade",)
    assert row.order_types == () and row.auction_eligible is False


def test_otc_is_classified_out_of_the_opening_auction(broker, session) -> None:
    (row,) = _capabilities(broker, session, "OTCX", _asset("OTCX", exchange="OTC"))
    assert row.auction_eligible is False
    assert row.order_types == ("market_day",)
    assert row.restrictions == ()  # not auction-eligible is not a restriction on trading


def test_an_unknown_venue_attribute_restricts_by_name(broker, session) -> None:
    # The fail-safe direction: an attribute this engine has never seen becomes a
    # restriction recorded by name, which portfolio/limits.py reads as blocking new
    # exposure. An informational attribute (options) does not.
    from prism.portfolio.limits import (
        restriction_blocks_new_exposure,
        unknown_restrictions,
    )

    (row,) = _capabilities(
        broker, session, "WEIRD", _asset("WEIRD", attributes=["has_options", "some_new_venue_flag"])
    )
    assert row.restrictions == ("some_new_venue_flag",)
    assert restriction_blocks_new_exposure(row.restrictions) is True
    assert unknown_restrictions(row.restrictions) == ("some_new_venue_flag",)


def test_a_ptp_attribute_maps_onto_the_declared_vocabulary(broker, session) -> None:
    from prism.portfolio.limits import KNOWN_RESTRICTIONS

    (row,) = _capabilities(broker, session, "PTP", _asset("PTP", attributes=["ptp_no_exception"]))
    assert row.restrictions == ("no_new_exposure",)
    assert set(row.restrictions) <= KNOWN_RESTRICTIONS


def test_an_unknown_symbol_yields_no_row_rather_than_a_permissive_one(broker, session) -> None:
    session.route("GET", "/v2/assets/GONE", FakeResponse(status_code=404, payload={"message": "not found"}))
    session.route("GET", "/v2/assets/AAPL", _asset("AAPL"))
    rows = broker.capabilities(["AAPL", "GONE"], AS_OF)
    assert [row.sid for row in rows] == ["AAPL"]


def test_a_non_equity_asset_class_is_refused(broker, session) -> None:
    with pytest.raises(ValueError, match="US equities"):
        _capabilities(broker, session, "BTCUSD", _asset("BTCUSD", **{"class": "crypto"}))


def test_capabilities_requires_the_instant_it_is_asked_about(broker) -> None:
    with pytest.raises(ValueError, match="as_of"):
        broker.capabilities(["AAPL"], "")


# ---------------------------------------------------------------------------
# §3.9: open orders and cancel
# ---------------------------------------------------------------------------


def test_open_orders_reports_signed_quantities(broker, session) -> None:
    def handler(json=None, params=None):
        assert params["status"] == "open"
        return FakeResponse(
            payload=[
                {
                    "client_order_id": "d1:AAA",
                    "symbol": "AAA",
                    "side": "buy",
                    "qty": "50",
                    "filled_qty": "10",
                    "type": "market",
                    "time_in_force": "opg",
                    "status": "partially_filled",
                    "submitted_at": "t1",
                },
                {
                    "client_order_id": "d1:BBB",
                    "symbol": "BBB",
                    "side": "sell",
                    "qty": "8",
                    "filled_qty": "0",
                    "type": "market",
                    "time_in_force": "day",
                    "status": "new",
                    "submitted_at": "t2",
                },
            ]
        )

    session.route("GET", "/v2/orders", handler)
    by_id = {row.client_order_id: row for row in broker.open_orders()}
    assert by_id["d1:AAA"].qty == 50.0 and by_id["d1:AAA"].filled_qty == 10.0
    assert by_id["d1:BBB"].qty == -8.0 and by_id["d1:BBB"].filled_qty == -0.0
    assert by_id["d1:BBB"].time_in_force == "day" and by_id["d1:BBB"].status == "new"


def test_cancel_is_idempotent_on_an_unknown_id(broker, session) -> None:
    session.route(
        "GET",
        "/v2/orders:by_client_order_id",
        FakeResponse(status_code=404, payload={"message": "order not found"}),
    )
    broker.cancel("d1:GONE")  # no exception, and no DELETE issued
    assert [c["method"] for c in session.calls] == ["GET"]


def test_cancel_is_idempotent_on_a_terminal_order(broker, session) -> None:
    session.route(
        "GET",
        "/v2/orders:by_client_order_id",
        FakeResponse(payload={"id": "venue-1", "status": "filled"}),
    )
    broker.cancel("d1:AAA")
    assert [c["method"] for c in session.calls] == ["GET"]


def test_cancel_deletes_a_working_order_and_tolerates_a_race(broker, session) -> None:
    session.route(
        "GET",
        "/v2/orders:by_client_order_id",
        FakeResponse(payload={"id": "venue-1", "status": "new"}),
    )
    session.route("DELETE", "/v2/orders/venue-1", FakeResponse(status_code=204))
    broker.cancel("d1:AAA")
    assert session.calls[-1]["method"] == "DELETE"
    # The order reached a terminal state between the lookup and the cancel: the
    # post-condition already holds, so 422 is not an error.
    session.route("DELETE", "/v2/orders/venue-1", FakeResponse(status_code=422, payload={"message": "not cancelable"}))
    broker.cancel("d1:AAA")


# ---------------------------------------------------------------------------
# §3.11: fills since a marker
# ---------------------------------------------------------------------------


def _activity(activity_id, symbol, side, qty, price, order_id="o1", **extra):
    row = {
        "id": activity_id,
        "activity_type": "FILL",
        "symbol": symbol,
        "side": side,
        "qty": str(qty),
        "price": str(price),
        "order_id": order_id,
        "transaction_time": "2026-09-08T13:31:00Z",
    }
    row.update(extra)
    return row


def test_fills_since_returns_executions_and_a_marker_that_resumes(broker, session, monkeypatch) -> None:
    monkeypatch.setattr("prism.live.alpaca._ACTIVITIES_PAGE_LIMIT", 2)
    pages = {
        None: [
            _activity("a1", "AAA", "buy", 30, 101.0, client_order_id="d1:AAA"),
            _activity("a2", "AAA", "buy", 20, 101.5, client_order_id="d1:AAA"),
        ],
        "a2": [_activity("a3", "BBB", "sell", 5, 49.0, client_order_id="d1:BBB")],
        "a3": [],
    }

    def handler(json=None, params=None):
        assert params["direction"] == "asc"
        return FakeResponse(payload=pages[params.get("page_token")])

    session.route("GET", "/v2/account/activities/FILL", handler)
    records, marker = broker.fills_since(None)
    # Two executions of one order stay two rows: the ledger is keyed by execution id.
    assert [r.execution_id for r in records] == ["a1", "a2", "a3"]
    assert [r.qty for r in records] == [30.0, 20.0, -5.0]
    assert marker == "a3"
    # Append-then-advance: resuming from the saved marker pulls nothing twice.
    again, next_marker = broker.fills_since(marker)
    assert again == [] and next_marker == "a3"


def test_fills_since_resolves_a_client_order_id_once_per_order(broker, session) -> None:
    session.route(
        "GET",
        "/v2/account/activities/FILL",
        FakeResponse(
            payload=[
                _activity("a1", "AAA", "buy", 1, 10.0, order_id="venue-1"),
                _activity("a2", "AAA", "buy", 2, 11.0, order_id="venue-1"),
            ]
        ),
    )
    session.route("GET", "/v2/orders/venue-1", FakeResponse(payload={"client_order_id": "d1:AAA"}))
    records, _ = broker.fills_since(None)
    assert [r.client_order_id for r in records] == ["d1:AAA", "d1:AAA"]
    lookups = [c for c in session.calls if c["path"] == "/v2/orders/venue-1"]
    assert len(lookups) == 1


def test_a_zero_quantity_fill_activity_is_a_defect_not_a_silent_skip(broker, session) -> None:
    session.route(
        "GET",
        "/v2/account/activities/FILL",
        FakeResponse(payload=[_activity("a1", "AAA", "buy", 0, 10.0, client_order_id="d1:AAA")]),
    )
    with pytest.raises(AlpacaAPIError, match="qty"):
        broker.fills_since(None)
