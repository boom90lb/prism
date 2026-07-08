"""AlpacaBarSource mappings (SPEC §7.0/§7.4), tested against canned payloads.

The live loop's Alpaca bar feed runs offline through an injectable fake session:
timestamp -> midnight-ET index, OHLCV column contract, pagination, split
adjustment, fail-loud on a non-2xx, and the DataLoader-compatible signature.
Only the real HTTP transport is left network-gated.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from prism.live import AlpacaAPIError, AlpacaBarSource
from prism.live.alpaca_data import DATA_BASE_URL

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
    """Routes (method, path) to a canned response or handler(json, params)."""

    def __init__(self):
        self.routes = {}
        self.calls = []

    def route(self, method, path, response):
        self.routes[(method, path)] = response

    def request(self, method, url, headers=None, timeout=None, json=None, params=None):
        assert url.startswith(DATA_BASE_URL)
        path = url[len(DATA_BASE_URL) :]
        self.calls.append({"method": method, "path": path, "headers": headers, "params": params})
        handler = self.routes[(method, path)]
        if callable(handler):
            return handler(json=json, params=params)
        return handler


def _bar(t, o, h, low, c, v):
    return {"t": t, "o": o, "h": h, "l": low, "c": c, "v": v, "n": 1, "vw": c}


@pytest.fixture()
def session():
    return FakeSession()


@pytest.fixture()
def source(session):
    return AlpacaBarSource(KEY, SECRET, session=session)


def test_fetch_incremental_parses_ohlcv_at_midnight_et(source, session) -> None:
    session.route(
        "GET",
        "/v2/stocks/AAPL/bars",
        FakeResponse(
            payload={
                "bars": [
                    # deliberately out of order to prove the frame sorts
                    _bar("2026-07-07T04:00:00Z", 315.29, 315.47, 310.18, 310.69, 1573402),
                    _bar("2026-07-06T04:00:00Z", 307.54, 314.19, 307.02, 312.73, 1434853),
                ],
                "next_page_token": None,
            }
        ),
    )
    df = source.fetch_incremental("AAPL", interval="1d")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert list(df.index.strftime("%Y-%m-%d")) == ["2026-07-06", "2026-07-07"]  # sorted
    assert str(df.index.tz) == "America/New_York"
    assert (df.index.hour == 0).all()  # 04:00Z EDT -> midnight ET, the store convention
    assert df["close"].iloc[-1] == 310.69 and df["volume"].iloc[0] == 1434853

    params = session.calls[-1]["params"]
    assert params["timeframe"] == "1Day" and params["adjustment"] == "split" and params["feed"] == "iex"


def test_fetch_incremental_paginates(source, session) -> None:
    pages = {
        None: FakeResponse(
            payload={"bars": [_bar("2026-07-06T04:00:00Z", 1, 1, 1, 100.0, 10)], "next_page_token": "P2"}
        ),
        "P2": FakeResponse(
            payload={"bars": [_bar("2026-07-07T04:00:00Z", 1, 1, 1, 101.0, 11)], "next_page_token": None}
        ),
    }
    session.route("GET", "/v2/stocks/AAPL/bars", lambda json=None, params=None: pages[params.get("page_token")])

    df = source.fetch_incremental("AAPL")
    assert len(df) == 2 and list(df["close"]) == [100.0, 101.0]
    assert session.calls[0]["params"].get("page_token") is None
    assert session.calls[1]["params"]["page_token"] == "P2"


def test_empty_bars_returns_empty_frame(source, session) -> None:
    session.route("GET", "/v2/stocks/ZZZ/bars", FakeResponse(payload={"bars": [], "next_page_token": None}))
    assert source.fetch_incremental("ZZZ").empty


def test_non_2xx_raises_without_leaking_credentials(source, session) -> None:
    session.route(
        "GET",
        "/v2/stocks/AAPL/bars",
        FakeResponse(status_code=403, payload={"message": "subscription does not permit querying recent SIP data"}),
    )
    with pytest.raises(AlpacaAPIError) as excinfo:
        source.fetch_incremental("AAPL")
    assert excinfo.value.status_code == 403
    assert SECRET not in str(excinfo.value)


def test_store_arg_accepted_and_ignored(source, session) -> None:
    # Duck-typed parity with DataLoader.fetch_incremental: a store kwarg must not
    # break the call (the Alpaca source refetches fresh and ignores it).
    session.route(
        "GET",
        "/v2/stocks/AAPL/bars",
        FakeResponse(payload={"bars": [_bar("2026-07-06T04:00:00Z", 1, 1, 1, 100.0, 10)], "next_page_token": None}),
    )
    df = source.fetch_incremental("AAPL", interval="1d", start_date="2026-01-01", store=object())
    assert len(df) == 1
    assert session.calls[-1]["params"]["start"] == "2026-01-01"


def test_from_env_reads_credentials_and_feed(monkeypatch) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.setenv("APCA_DATA_FEED", "sip")
    src = AlpacaBarSource.from_env(session=FakeSession())
    assert src._feed == "sip"  # env override honored


def test_from_env_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="APCA_API_KEY_ID"):
        AlpacaBarSource.from_env()
