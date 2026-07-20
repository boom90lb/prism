"""The SPEC §7.7 regime-step provider: per-block values, named failures, JSON-safety.

Pins the provider contract the daily seam relies on: every configured block
appears in every read (values or a named failure — never silently absent),
the provider never raises, non-finite values become ``None`` (strict-JSON
safe for the regime ledger), the fetch window is causal (``end`` = the
decision bar), and any failure entry marks the session not clean for the
handoff §8 precondition-(b) 21-session clock.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from prism.live.regime_step import ABSENT_BLOCKS, REGIME_BLOCKS, RegimeTelemetry
from prism.regime.fetch import FRED_CURVE_TENORS, RegimeFetchError
from prism.scripts.paper_loop import _parse_args

_DATES = pd.date_range("2026-06-01", "2026-07-17", freq="B")
_BAR = "2026-07-17"


class FakeFred:
    """FredClient-shaped fake: canned series by id, window-sliced; ids in
    ``fail`` raise the adapter's own error class (transport/venue failure)."""

    def __init__(self, by_series: dict[str, pd.Series], fail: set[str] | None = None) -> None:
        self.by_series = by_series
        self.fail = fail or set()
        self.calls: list[tuple[str, str | None, str | None]] = []

    def series(self, series_id: str, *, start: str | None = None, end: str | None = None) -> pd.Series:
        self.calls.append((series_id, start, end))
        if series_id in self.fail:
            raise RegimeFetchError(f"FRED {series_id}: HTTP 500")
        series = self.by_series.get(series_id)
        if series is None:
            raise AssertionError(f"unexpected FRED series {series_id!r}")
        out = series
        if start:
            out = out.loc[out.index >= pd.Timestamp(start)]
        if end:
            out = out.loc[out.index <= pd.Timestamp(end)]
        if out.empty:
            raise RegimeFetchError(f"FRED {series_id}: zero observations returned")
        return out.rename(series_id)


class FakeLlama:
    """DefiLlamaClient-shaped fake: flat $250bn float."""

    def stablecoin_float(self, *, in_millions: bool = True) -> pd.Series:
        scale = 1.0 if in_millions else 1e6
        return pd.Series(250_000.0 * scale, index=_DATES, name="stablecoin_float")


def _flat(value: float) -> pd.Series:
    return pd.Series(float(value), index=_DATES)


def _fred(fail: set[str] | None = None, **overrides: pd.Series) -> FakeFred:
    by_series: dict[str, pd.Series] = {sid: _flat(4.0) for sid in FRED_CURVE_TENORS}
    by_series.update(
        {
            "WALCL": _flat(6_600_000.0),
            "RRPONTSYD": _flat(380_000.0),
            "WTREGEN": _flat(700_000.0),
            "DFII10": _flat(1.5),
            "T10YIE": _flat(2.4),
            "VIXCLS": _flat(18.0),
            "VXVCLS": _flat(20.0),
        }
    )
    by_series.update(overrides)
    return FakeFred(by_series, fail=fail)


def test_all_configured_blocks_present_and_valued() -> None:
    result = RegimeTelemetry(_fred())(_BAR)
    assert result["decision_bar"] == _BAR
    assert set(result["blocks"]) == set(REGIME_BLOCKS)
    assert result["failures"] == []
    assert result["absent"] == sorted(ABSENT_BLOCKS)

    curve = result["blocks"]["curve"]
    assert curve["level"] == pytest.approx(4.0)  # flat 4% curve
    assert curve["slope"] == pytest.approx(0.0, abs=1e-12)
    assert curve["curvature"] == pytest.approx(0.0, abs=1e-12)
    assert curve["asof"] == _BAR

    liquidity = result["blocks"]["liquidity"]
    assert liquidity["net_liquidity"] == pytest.approx(6_600_000 - 380_000 - 700_000)
    assert liquidity["net_liquidity_change"] == pytest.approx(0.0)  # flat series
    assert liquidity["stablecoin_term"] is False  # three-term identity by default

    inflation = result["blocks"]["inflation"]
    assert inflation["real_yield"] == pytest.approx(1.5)
    assert inflation["breakeven_divergence"] == pytest.approx(0.4)

    vol = result["blocks"]["vol"]
    assert vol["vix"] == pytest.approx(18.0)
    assert vol["term_ratio"] == pytest.approx(0.9)  # 18/20: contango


def test_injected_llama_adds_the_fourth_term() -> None:
    result = RegimeTelemetry(_fred(), FakeLlama())(_BAR)
    liquidity = result["blocks"]["liquidity"]
    assert liquidity["stablecoin_term"] is True
    assert liquidity["net_liquidity"] == pytest.approx(6_600_000 - 380_000 - 700_000 + 250_000)


def test_fetch_window_is_causal() -> None:
    fred = _fred()
    RegimeTelemetry(fred)(_BAR)
    assert fred.calls, "provider must fetch"
    assert all(end == _BAR for _, _, end in fred.calls)  # nothing after the decision bar
    assert all(start is not None and start < _BAR for _, start, _ in fred.calls)


def test_one_failing_block_is_named_loud_and_isolated(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="prism.live.regime_step"):
        result = RegimeTelemetry(_fred(fail={"DFII10"}))(_BAR)
    # The failing block carries the exception CLASS only — no message text
    # that could smuggle a URL or key into the ledger.
    assert result["blocks"]["inflation"] == {"error": "RegimeFetchError"}
    assert result["failures"] == [{"block": "inflation", "error": "RegimeFetchError"}]
    # The other blocks still valued: one dark source does not blind the read.
    for name in ("curve", "liquidity", "vol"):
        assert "error" not in result["blocks"][name]
    warnings = [r for r in caplog.records if "REGIME BLOCK FAILED" in r.getMessage()]
    assert len(warnings) == 1 and "inflation" in warnings[0].getMessage()
    assert "NOT clean" in warnings[0].getMessage()


def test_total_transport_failure_is_never_silently_empty() -> None:
    fred = _fred(fail=set(_fred().by_series))
    result = RegimeTelemetry(fred)(_BAR)
    assert set(result["blocks"]) == set(REGIME_BLOCKS)  # every block still appears
    assert {f["block"] for f in result["failures"]} == set(REGIME_BLOCKS)
    assert all(entry == {"error": "RegimeFetchError"} for entry in result["blocks"].values())


def test_all_nan_window_is_a_named_failure() -> None:
    nan = pd.Series(np.nan, index=_DATES)
    result = RegimeTelemetry(_fred(VIXCLS=nan, VXVCLS=nan))(_BAR)
    assert result["blocks"]["vol"] == {"error": "RegimeFetchError"}  # empty read booked loud, not as data
    assert {f["block"] for f in result["failures"]} == {"vol"}


def test_read_is_strict_json_safe_with_nan_tail() -> None:
    # VIX3M goes dark on the last session: the row still reads (VIX is finite)
    # and the non-finite fields become None, so the regime ledger row is
    # strict JSON (no NaN token).
    vix3m = _flat(20.0)
    vix3m.iloc[-1] = np.nan
    result = RegimeTelemetry(_fred(VXVCLS=vix3m))(_BAR)
    vol = result["blocks"]["vol"]
    assert vol["vix"] == pytest.approx(18.0)
    assert vol["vix3m"] is None and vol["term_ratio"] is None
    assert result["failures"] == []
    json.loads(json.dumps(result, allow_nan=False))  # round-trips under strict JSON


def test_provider_never_raises_even_on_a_malformed_bar() -> None:
    result = RegimeTelemetry(_fred())("not-a-date")
    assert {f["block"] for f in result["failures"]} == set(REGIME_BLOCKS)
    assert all("error" in entry for entry in result["blocks"].values())


def test_from_env_requires_fred_key(monkeypatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        RegimeTelemetry.from_env()
    monkeypatch.setenv("FRED_API_KEY", "k3y")
    assert isinstance(RegimeTelemetry.from_env(), RegimeTelemetry)


def test_lookback_days_validated() -> None:
    with pytest.raises(ValueError, match="lookback_days"):
        RegimeTelemetry(_fred(), lookback_days=0)


def test_paper_loop_flag_defaults_off() -> None:
    assert _parse_args([]).regime is False
    assert _parse_args(["--regime"]).regime is True
