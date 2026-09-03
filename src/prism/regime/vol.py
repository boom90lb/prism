"""Volatility regime state (SPEC.md §3 law 4, §7.5).

The only piece of the diffusion / Black-Scholes law a zero-budget daily equity
bot can touch: the variance risk premium and the VIX term-structure slope, as
conditioning/gating features. Trading the surface (options, vol-selling, delta
hedging) is an explicit non-goal (SPEC.md §8) — nothing here prices or trades an
option; it consumes free published index levels (FRED VIXCLS/VXVCLS, CBOE VIX9D)
and free underlying OHLC.

All inputs are annualized volatilities in percent (e.g. 20.0 for 20%), matching
how VIX is quoted. Everything is causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUALIZATION = 252.0


def realized_volatility(returns: pd.Series, window: int, *, annualized_pct: bool = True) -> pd.Series:
    """Trailing realized volatility from daily returns.

    Rolling sample stdev of ``returns`` over ``window`` bars, annualized by
    ``sqrt(252)`` and expressed in percent when ``annualized_pct`` (to match VIX
    units). Causal: the value at bar ``t`` uses returns through ``t``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    r = pd.to_numeric(returns, errors="coerce")
    vol = r.rolling(window, min_periods=window).std(ddof=1)
    if annualized_pct:
        vol = vol * np.sqrt(_ANNUALIZATION) * 100.0
    return vol


def _as_aligned(a: pd.Series | float, b: pd.Series | float) -> tuple[pd.Series, pd.Series] | tuple[float, float]:
    """Coerce a scalar/Series pair: both scalar -> floats; else broadcast the
    scalar over the Series index and inner-join two Series."""
    a_is, b_is = isinstance(a, pd.Series), isinstance(b, pd.Series)
    if not a_is and not b_is:
        return float(a), float(b)
    if a_is and not b_is:
        a_s = a.astype(float)
        return a_s, pd.Series(float(b), index=a_s.index)
    if b_is and not a_is:
        b_s = b.astype(float)
        return pd.Series(float(a), index=b_s.index), b_s
    return a.astype(float).align(b.astype(float), join="inner")  # type: ignore[union-attr]


def variance_risk_premium(implied_vol: pd.Series | float, realized_vol: pd.Series | float) -> pd.Series | float:
    """VRP = implied variance minus realized variance (Bollerslev-Tauchen-Zhou).

    Both inputs are annualized vols in percent; the premium is returned in
    variance points (``implied^2 - realized^2``), positive when the market prices
    more variance than has recently realized (the usual risk-on state). Accepts
    two scalars, two Series (inner-joined), or a scalar broadcast over a Series.
    """
    iv, rv = _as_aligned(implied_vol, realized_vol)
    return iv**2 - rv**2


def vix_term_slope(
    short_vix: pd.Series | float,
    long_vix: pd.Series | float,
    *,
    as_ratio: bool = True,
) -> pd.Series | float:
    """VIX term-structure slope (contango vs backwardation).

    With ``as_ratio`` (default) returns ``short / long`` — a ratio > 1 is
    backwardation (near-term stress, risk-off), < 1 is contango (calm). With
    ``as_ratio=False`` returns ``long - short`` in vol points. Use e.g.
    (VIX9D, VIX) for the very-front slope or (VIX, VIX3M) for the 1-3 month slope.
    Accepts scalars or aligned Series; guards divide-by-zero to NaN.
    """
    s, long_series = _as_aligned(short_vix, long_vix)
    if isinstance(s, pd.Series):
        if as_ratio:
            return (s / long_series.where(long_series != 0.0)).replace([np.inf, -np.inf], np.nan)
        return long_series - s
    if as_ratio:
        return s / long_series if long_series != 0.0 else float("nan")
    return long_series - s
