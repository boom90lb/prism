"""Offline tests for the regime fetch adapters (canned payloads, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.regime.fetch import (
    FRED_CURVE_TENORS,
    DefiLlamaClient,
    FredClient,
    RegimeFetchError,
    fetch_curve_state,
    fetch_inflation_state,
    fetch_net_liquidity,
)
from prism.regime.liquidity import net_liquidity


class _Response:
    def __init__(self, payload: object, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self) -> object:
        return self._payload


class _FakeSession:
    """Requests-compatible fake: routes by FRED series_id / URL substring."""

    def __init__(self, by_series: dict[str, object] | None = None, by_url: dict[str, object] | None = None) -> None:
        self.by_series = by_series or {}
        self.by_url = by_url or {}
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> _Response:
        self.calls.append((url, params))
        if params and "series_id" in params:
            response = self.by_series.get(params["series_id"])
            if response is None:
                raise AssertionError(f"unexpected FRED series {params['series_id']!r}")
            return response if isinstance(response, _Response) else _Response(response)
        for fragment, response in self.by_url.items():
            if fragment in url:
                return response if isinstance(response, _Response) else _Response(response)
        raise AssertionError(f"unexpected URL {url!r}")


def _fred_payload(pairs: list[tuple[str, str]]) -> dict:
    return {"observations": [{"date": d, "value": v} for d, v in pairs]}


def _llama_payload(pairs: list[tuple[int, float]]) -> list[dict]:
    return [{"date": str(ts), "totalCirculatingUSD": {"peggedUSD": usd}} for ts, usd in pairs]


class TestFredClient:
    def test_series_parses_values_and_dot_marker(self):
        session = _FakeSession(
            by_series={"DGS10": _fred_payload([("2026-01-02", "4.57"), ("2026-01-05", "."), ("2026-01-06", "4.61")])}
        )
        series = FredClient("k3y", session=session).series("DGS10")
        assert series.name == "DGS10"
        assert series.dtype == float
        assert series.index.tz is None
        assert series.loc["2026-01-02"] == pytest.approx(4.57)
        assert np.isnan(series.loc["2026-01-05"])  # "." kept as NaN, row not dropped

    def test_date_window_params_forwarded(self):
        session = _FakeSession(by_series={"DTB3": _fred_payload([("2026-01-02", "3.71")])})
        FredClient("k3y", session=session).series("DTB3", start="2026-01-01", end="2026-02-01")
        _, params = session.calls[0]
        assert params["observation_start"] == "2026-01-01"
        assert params["observation_end"] == "2026-02-01"

    def test_http_error_raises_and_scrubs_key(self):
        session = _FakeSession(
            by_series={"DGS10": _Response({}, status_code=400, text="Bad Request: api_key k3y is not registered")}
        )
        with pytest.raises(RegimeFetchError) as excinfo:
            FredClient("k3y", session=session).series("DGS10")
        assert excinfo.value.status_code == 400
        assert "k3y" not in str(excinfo.value)

    def test_zero_observations_raise(self):
        session = _FakeSession(by_series={"DGS10": {"observations": []}})
        with pytest.raises(RegimeFetchError, match="zero observations"):
            FredClient("k3y", session=session).series("DGS10")

    def test_from_env_requires_key(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FRED_API_KEY"):
            FredClient.from_env()
        monkeypatch.setenv("FRED_API_KEY", "k3y")
        assert isinstance(FredClient.from_env(session=_FakeSession()), FredClient)

    def test_empty_key_rejected(self):
        with pytest.raises(ValueError, match="api_key"):
            FredClient("")


class TestDefiLlamaClient:
    def test_stablecoin_float_parses_and_rescales_to_millions(self):
        session = _FakeSession(
            by_url={"stablecoincharts/all": _llama_payload([(1735689600, 2.0e11), (1735776000, 2.1e11)])}
        )
        series = DefiLlamaClient(session=session).stablecoin_float()
        assert series.index.tz is None
        assert series.iloc[0] == pytest.approx(2.0e5)  # $200bn -> 200,000 $mm
        dollars = DefiLlamaClient(session=session).stablecoin_float(in_millions=False)
        assert dollars.iloc[1] == pytest.approx(2.1e11)

    def test_http_error_and_empty_raise(self):
        err = _FakeSession(by_url={"stablecoincharts/all": _Response([], status_code=500, text="boom")})
        with pytest.raises(RegimeFetchError, match="500"):
            DefiLlamaClient(session=err).stablecoin_float()
        empty = _FakeSession(by_url={"stablecoincharts/all": []})
        with pytest.raises(RegimeFetchError, match="empty"):
            DefiLlamaClient(session=empty).stablecoin_float()

    def test_malformed_payload_raises(self):
        session = _FakeSession(by_url={"stablecoincharts/all": [{"date": "1735689600", "wrong": {}}]})
        with pytest.raises(RegimeFetchError, match="payload shape"):
            DefiLlamaClient(session=session).stablecoin_float()


class TestAssemblers:
    def test_fetch_curve_state_feeds_the_tenor_panel(self):
        flat = _fred_payload([("2026-01-02", "4.00"), ("2026-01-05", "4.00")])
        session = _FakeSession(by_series={sid: flat for sid in FRED_CURVE_TENORS})
        state = fetch_curve_state(FredClient("k3y", session=session))
        assert list(state.columns) == ["level", "slope", "curvature"]
        assert len(state) == 2
        # A flat 4% curve: level 4, slope and curvature 0.
        assert state["level"].iloc[0] == pytest.approx(4.0)
        assert state["slope"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert state["curvature"].iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_fetch_inflation_state_columns(self):
        session = _FakeSession(
            by_series={
                "DFII10": _fred_payload([("2026-01-02", "1.5")]),
                "T10YIE": _fred_payload([("2026-01-02", "2.4")]),
            }
        )
        state = fetch_inflation_state(FredClient("k3y", session=session))
        assert list(state.columns) == ["real_yield", "breakeven", "breakeven_divergence"]
        assert state["breakeven_divergence"].iloc[0] == pytest.approx(0.4)

    def test_fetch_net_liquidity_matches_pure_identity(self):
        session = _FakeSession(
            by_series={
                "WALCL": _fred_payload([("2026-01-07", "6600000")]),
                "RRPONTSYD": _fred_payload([("2026-01-06", "400000"), ("2026-01-07", "380000")]),
                "WTREGEN": _fred_payload([("2026-01-07", "700000")]),
            },
            by_url={"stablecoincharts/all": _llama_payload([(1767744000, 2.5e11)])},
        )
        fred = FredClient("k3y", session=session)
        three_term = fetch_net_liquidity(fred)
        expected = net_liquidity(
            fred.series("WALCL"), fred.series("RRPONTSYD"), fred.series("WTREGEN")
        )
        pd.testing.assert_series_equal(three_term, expected)
        four_term = fetch_net_liquidity(fred, DefiLlamaClient(session=session))
        # 2026-01-07 (unix 1767744000 = 2026-01-07 UTC): +$250bn float in $mm.
        assert four_term.loc["2026-01-07"] == pytest.approx(6600000 - 380000 - 700000 + 250000)
