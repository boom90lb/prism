"""Unit tests for capture-only observatory fetchers (mocked HTTP)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from research.scripts.observatory_capture import fetch_edgar_cadence, fetch_news_flow


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(payload)

    def json(self):
        return self._payload


def test_fetch_news_flow_counts():
    calls: list[dict] = []
    sleeps: list[float] = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params})
        return _Resp(
            200,
            {
                "status": "OK",
                "count": 2,
                "results": [
                    {"published_utc": "2026-07-22T10:00:00Z"},
                    {"published_utc": "2026-07-22T12:00:00Z"},
                ],
            },
        )

    out = fetch_news_flow(
        ("SPY", "AAPL"),
        api_key="test-key",
        lookback_days=1,
        pause_s=0.0,
        get=fake_get,
        sleep=sleeps.append,
    )
    assert out["n_articles_sum"] == 4
    assert out["per_ticker"]["SPY"]["n"] == 2
    assert out["per_ticker"]["SPY"]["latest_published_utc"] == "2026-07-22T12:00:00Z"
    assert len(calls) == 2
    assert sleeps == []


def test_fetch_news_missing_key_raises():
    with pytest.raises(RuntimeError, match="POLYGON_API_KEY"):
        fetch_news_flow(("SPY",), api_key="", get=lambda *a, **k: None)


def test_fetch_news_http_error_raises():
    def bad_get(*a, **k):
        return _Resp(500, text="boom")

    with pytest.raises(RuntimeError, match="HTTP 500"):
        fetch_news_flow(("SPY",), api_key="k", pause_s=0.0, get=bad_get, sleep=lambda s: None)


def test_fetch_news_429_retries_once():
    n = {"i": 0}

    def flaky_get(*a, **k):
        n["i"] += 1
        if n["i"] == 1:
            return _Resp(429, text="rate limited")
        return _Resp(200, {"status": "OK", "results": [{"published_utc": "2026-07-22T10:00:00Z"}]})

    sleeps: list[float] = []
    out = fetch_news_flow(
        ("SPY",),
        api_key="k",
        pause_s=0.0,
        get=flaky_get,
        sleep=sleeps.append,
    )
    assert out["n_articles_sum"] == 1
    assert sleeps and sleeps[0] >= 60.0


def test_fetch_edgar_cadence_parses_hits():
    def fake_get(url, params=None, headers=None, timeout=None):
        assert "efts.sec.gov" in url
        assert params["startdt"] == "2026-07-21"
        assert "User-Agent" in headers
        return _Resp(200, {"hits": {"total": {"value": 1234, "relation": "eq"}, "hits": []}})

    out = fetch_edgar_cadence(date(2026, 7, 21), user_agent="prism-test", get=fake_get)
    assert out["n_hits"] == 1234
    assert out["day"] == "2026-07-21"
    assert "10-K" in out["forms"]


def test_fetch_edgar_missing_ua_raises():
    with pytest.raises(RuntimeError, match="User-Agent"):
        fetch_edgar_cadence(date(2026, 7, 21), user_agent="  ", get=lambda *a, **k: None)
