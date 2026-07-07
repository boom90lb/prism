"""Live fetch adapters for the regime layer (SPEC.md §7.5; the §11 follow-on).

Thin ``requests`` shells over the free sources registered in
``regime.sources``, written against an injectable requests-compatible session
(the ``live.alpaca`` pattern) so every parse/assembly path is tested offline
on canned payloads; only the transport itself is network-gated.

Providers covered here: **FRED** (curve fallback, vol indices, net-liquidity
terms, inflation expectations, the DTB3 hurdle anchor) and **DefiLlama**
(aggregate stablecoin float, the fourth net-liquidity term). The Treasury
FiscalData XML and CBOE CSV sources in the registry remain unadapted — FRED
carries every series the shipped feature math consumes except VIX9D.

Conventions:

* Series come back float-valued on a tz-naive daily ``DatetimeIndex`` (FRED
  publishes dates; DefiLlama unix timestamps are normalized to UTC dates).
  Alignment with tz-aware price panels is the consumer's decision, not
  smuggled in here.
* FRED's ``"."`` marker (holiday/no-observation) becomes NaN and the row is
  *kept* — genuine absence stays visible (N7); nothing is forward-filled here
  (the regime feature math does its own causal within-series ffill).
* A fetch that yields zero observations raises — an empty regime series
  booked as data is the silent-degradation defect class (N7).
* The FRED API key travels only in query params and is scrubbed from any
  error text.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

from prism.regime.curve import curve_state_panel
from prism.regime.inflation import inflation_state
from prism.regime.liquidity import net_liquidity

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFILLAMA_BASE_URL = "https://stablecoins.llama.fi"

# FRED DGS series -> tenor in years (regime.sources CURVE_SOURCES fallback row).
FRED_CURVE_TENORS: dict[str, float] = {
    "DGS1MO": 1.0 / 12.0,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}


class RegimeFetchError(RuntimeError):
    """A regime-source response that cannot be booked as data (N7)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FredClient:
    """Minimal FRED ``series/observations`` client (free API key)."""

    def __init__(self, api_key: str, *, session: Any | None = None, timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError("FRED api_key must be non-empty")
        self._api_key = api_key
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout

    @classmethod
    def from_env(cls, **kwargs: Any) -> "FredClient":
        """Construct from ``FRED_API_KEY``; missing credentials raise (N7)."""
        api_key = os.environ.get("FRED_API_KEY", "")
        if not api_key:
            raise RuntimeError("FRED_API_KEY is not set; register a free key at fred.stlouisfed.org (N7)")
        return cls(api_key, **kwargs)

    def series(self, series_id: str, *, start: str | None = None, end: str | None = None) -> pd.Series:
        """One FRED series as a float Series (tz-naive daily index, NaN for ``"."``)."""
        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end
        response = self._session.get(FRED_BASE_URL, params=params, timeout=self._timeout)
        if response.status_code != 200:
            body = str(getattr(response, "text", ""))[:300].replace(self._api_key, "***")
            raise RegimeFetchError(f"FRED {series_id}: HTTP {response.status_code}: {body}", response.status_code)
        observations = response.json().get("observations", [])
        if not observations:
            raise RegimeFetchError(f"FRED {series_id}: zero observations returned (N7: absence is not data)")
        index = pd.to_datetime([obs["date"] for obs in observations])
        values = pd.to_numeric(pd.Series([obs["value"] for obs in observations]), errors="coerce")
        return pd.Series(values.to_numpy(), index=index, name=series_id, dtype=float)


class DefiLlamaClient:
    """Aggregate stablecoin float from DefiLlama (keyless)."""

    def __init__(self, *, session: Any | None = None, timeout: float = 30.0) -> None:
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout

    def stablecoin_float(self, *, in_millions: bool = True) -> pd.Series:
        """Daily aggregate USD-pegged circulating value.

        ``in_millions=True`` (default) rescales dollars to FRED's $millions so
        the series can be passed straight into ``net_liquidity`` beside WALCL/
        RRP/TGA (the units note on ``regime.sources``).
        """
        url = f"{DEFILLAMA_BASE_URL}/stablecoincharts/all"
        response = self._session.get(url, timeout=self._timeout)
        if response.status_code != 200:
            body = str(getattr(response, "text", ""))[:300]
            raise RegimeFetchError(f"DefiLlama stablecoincharts: HTTP {response.status_code}: {body}", response.status_code)
        rows = response.json()
        if not rows:
            raise RegimeFetchError("DefiLlama stablecoincharts: empty response (N7: absence is not data)")
        try:
            index = pd.to_datetime([int(row["date"]) for row in rows], unit="s", utc=True).tz_localize(None).normalize()
            values = [float(row["totalCirculatingUSD"]["peggedUSD"]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise RegimeFetchError(f"DefiLlama stablecoincharts: unexpected payload shape: {exc}") from exc
        scale = 1e-6 if in_millions else 1.0
        return pd.Series([v * scale for v in values], index=index, name="stablecoin_float", dtype=float)


# ------------------------------------------------------------------ assemblers
# One call per regime block: fetch the registered series and hand them to the
# (already-tested) pure feature math. These are the live counterparts of the
# synthetic-series unit tests.


def fetch_curve_state(fred: FredClient, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """FRED DGS curve -> ``curve_state_panel`` (level/slope/curvature per date)."""
    panel = pd.DataFrame(
        {tenor: fred.series(series_id, start=start, end=end) for series_id, tenor in FRED_CURVE_TENORS.items()}
    )
    return curve_state_panel(panel)


def fetch_inflation_state(fred: FredClient, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """FRED DFII10 + T10YIE -> ``inflation_state`` (real yield, breakeven, divergence)."""
    return inflation_state(
        real_yield=fred.series("DFII10", start=start, end=end),
        breakeven=fred.series("T10YIE", start=start, end=end),
    )


def fetch_net_liquidity(
    fred: FredClient,
    llama: DefiLlamaClient | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    tga_series_id: str = "WTREGEN",
) -> pd.Series:
    """FRED WALCL − RRPONTSYD − TGA [+ DefiLlama stablecoin float], in $millions.

    ``llama=None`` reproduces the three-term identity; passing a client adds
    the §7.5 R4 fourth term (already rescaled to $millions here).
    """
    stable = llama.stablecoin_float(in_millions=True) if llama is not None else None
    if stable is not None and start:
        stable = stable.loc[stable.index >= pd.Timestamp(start)]
    return net_liquidity(
        balance_sheet=fred.series("WALCL", start=start, end=end),
        reverse_repo=fred.series("RRPONTSYD", start=start, end=end),
        treasury_general_account=fred.series(tga_series_id, start=start, end=end),
        stablecoin_float=stable,
    )
