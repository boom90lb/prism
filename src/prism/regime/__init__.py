"""Regime / context layer (SPEC.md §7.5).

Free ($0) EOD regime state that conditions sizing and gating for the core book —
it never becomes a tradable sleeve. Three blocks, the honest residue of the
lecture's laws 1/2/4:

* ``curve``     — yield-curve level/slope/curvature (law 2), fixed contrasts.
* ``vol``       — variance risk premium + VIX term slope (law 4).
* ``liquidity`` — Fed net liquidity WALCL - RRP - TGA (law 1), IC-gated diagnostic.

``sources`` is the documented free-data-source registry these consume. Pure,
causal, dependency-light (numpy/pandas only) — production-import-path safe (N8).
"""

from __future__ import annotations

from prism.regime.curve import curve_state, curve_state_panel
from prism.regime.liquidity import net_liquidity, net_liquidity_change
from prism.regime.vol import realized_volatility, variance_risk_premium, vix_term_slope

__all__ = [
    "curve_state",
    "curve_state_panel",
    "net_liquidity",
    "net_liquidity_change",
    "realized_volatility",
    "variance_risk_premium",
    "vix_term_slope",
]
