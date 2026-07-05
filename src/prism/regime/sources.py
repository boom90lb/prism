"""Free ($0) data-source map for the regime layer (SPEC.md §4).

This module is documentation-as-code: the canonical registry of the zero-budget
sources that feed ``regime.curve`` / ``regime.vol`` / ``regime.liquidity``. The
actual live fetch adapter (a thin `requests` shell against these URLs) is a
follow-on (SPEC.md §11) — it needs network access that the test sandbox does not
grant, so it is intentionally not implemented here. The pure feature math in the
sibling modules is tested on synthetic series instead.

Every entry is a free tier with no paid dependency. Rate limits and lags are the
binding constraints, not access.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeSource:
    """One free regime data source."""

    key: str
    provider: str
    series_or_endpoint: str
    cadence: str
    auth: str
    note: str


# Yield curve (SPEC.md §4: Treasury Fiscal Data primary, FRED fallback).
CURVE_SOURCES: tuple[RegimeSource, ...] = (
    RegimeSource(
        key="treasury_par_yield",
        provider="U.S. Treasury Fiscal Data",
        series_or_endpoint="v2/accounting/od/avg_interest_rates | Daily Treasury Par Yield Curve XML",
        cadence="daily (EOD, ~3:30pm ET)",
        auth="none",
        note="Authoritative CMT 1M-30Y; no API key. Preferred over any mirror.",
    ),
    RegimeSource(
        key="fred_dgs",
        provider="FRED",
        series_or_endpoint="DGS1MO,DGS3MO,DGS6MO,DGS1,DGS2,DGS3,DGS5,DGS7,DGS10,DGS20,DGS30",
        cadence="daily (~1 business-day lag)",
        auth="free API key",
        note="Fallback for the curve; also T10Y2Y/T10Y3M published slopes.",
    ),
)

# Volatility (SPEC.md §4: FRED VIX family primary, CBOE fallback; VIX9D is CBOE-only).
VOL_SOURCES: tuple[RegimeSource, ...] = (
    RegimeSource(
        key="fred_vix",
        provider="FRED",
        series_or_endpoint="VIXCLS (VIX), VXVCLS (VIX3M)",
        cadence="daily close (~1 session lag)",
        auth="free API key",
        note="EOD only; adequate for daily-horizon regime gating.",
    ),
    RegimeSource(
        key="cboe_vix9d",
        provider="CBOE",
        series_or_endpoint="VIX9D / VIX / VIX3M dashboard CSVs",
        cadence="daily EOD CSV; intraday delayed ~15-20 min",
        auth="none",
        note="VIX9D is NOT on FRED — pull the very-front slope from CBOE directly.",
    ),
)

# Macro / liquidity (SPEC.md §4: all FRED).
LIQUIDITY_SOURCES: tuple[RegimeSource, ...] = (
    RegimeSource(
        key="fred_net_liquidity",
        provider="FRED",
        series_or_endpoint="WALCL (balance sheet), RRPONTSYD (ON RRP), WTREGEN/WDTGAL (TGA)",
        cadence="weekly (WALCL) / daily (RRP, TGA)",
        auth="free API key",
        note="net_liquidity = WALCL - RRP - TGA; monitored diagnostic, IC-gated (I-8).",
    ),
)
