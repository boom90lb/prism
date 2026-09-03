"""Regime / context layer (SPEC.md §7.5).

Free ($0) EOD regime state that conditions sizing and gating for the core book —
it never becomes a tradable sleeve. Four blocks — the honest residue of the
lecture's laws 1/2/4, plus the R4 inflation-expectations follow-on:

* ``curve``     — yield-curve level/slope/curvature (law 2), fixed contrasts.
* ``vol``       — variance risk premium + VIX term slope (law 4).
* ``liquidity`` — Fed net liquidity WALCL - RRP - TGA (law 1), IC-gated
  diagnostic, with the optional stablecoin-float fourth term (§7.5 R4).
* ``inflation`` — real yield + breakeven-vs-target divergence (§7.5 R4), the
  financial-repression observables the nominal curve cannot see.

``sources`` is the documented free-data-source registry these consume;
``fetch`` is the thin live adapter over it (FRED + DefiLlama, injectable
transport). The feature math stays pure, causal, dependency-light
(numpy/pandas only) — production-import-path safe (N8).
"""

from __future__ import annotations

from prism.regime.curve import curve_state, curve_state_panel
from prism.regime.fetch import (
    DefiLlamaClient,
    FredClient,
    RegimeFetchError,
    fetch_curve_state,
    fetch_inflation_state,
    fetch_net_liquidity,
)
from prism.regime.inflation import breakeven_divergence, inflation_state
from prism.regime.liquidity import net_liquidity, net_liquidity_change
from prism.regime.vol import realized_volatility, variance_risk_premium, vix_term_slope

__all__ = [
    "DefiLlamaClient",
    "FredClient",
    "RegimeFetchError",
    "breakeven_divergence",
    "curve_state",
    "curve_state_panel",
    "fetch_curve_state",
    "fetch_inflation_state",
    "fetch_net_liquidity",
    "inflation_state",
    "net_liquidity",
    "net_liquidity_change",
    "realized_volatility",
    "variance_risk_premium",
    "vix_term_slope",
]
