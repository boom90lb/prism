"""Net-liquidity regime diagnostic (SPEC.md §3 law 1, §7.5).

The one defensible tradable-adjacent residue of the "capital conservation" law
for a retail actor: Fed net liquidity = balance sheet minus the two large cash
sinks. It is a **monitored diagnostic gated on IC (I-8)**, not a signal that
touches sizing until it earns its place — the net-liquidity/SPX link is a
contested QE-era artifact and, at weekly release cadence, gives only a handful of
independent observations per year (negligible breadth).

    net_liquidity = WALCL - RRPONTSYD - TGA

All inputs are FRED daily/weekly series in the same currency units (USD; FRED
publishes WALCL in $millions, RRP/TGA likewise — pass them in consistent units).
Pure and causal.
"""

from __future__ import annotations

import pandas as pd


def net_liquidity(
    balance_sheet: pd.Series,
    reverse_repo: pd.Series,
    treasury_general_account: pd.Series,
    stablecoin_float: pd.Series | None = None,
) -> pd.Series:
    """Fed net liquidity = WALCL - RRP - TGA [+ stablecoin float], union calendar.

    Series are aligned on the union of their dates and forward-filled *within*
    each series only (weekly WALCL vs daily RRP/TGA cadence mismatch), which is
    causal — a value known on its release date is carried forward, never backward.
    Dates before a series' first observation stay NaN (no fabricated history).

    ``stablecoin_float`` (optional, SPEC.md §7.5 R4 follow-on) is aggregate
    stablecoin market cap — structural T-bill demand outside the Fed's balance
    sheet, the fourth term of the identity for the post-GENIUS-Act issuance
    regime. It is *added*: growing float is liquidity entering the reachable
    risk system. Caller supplies consistent units across all four series (FRED
    publishes in $millions; DefiLlama in $ — rescale before passing). ``None``
    reproduces the three-term identity exactly.
    """
    cols = {
        "walcl": pd.to_numeric(balance_sheet, errors="coerce"),
        "rrp": pd.to_numeric(reverse_repo, errors="coerce"),
        "tga": pd.to_numeric(treasury_general_account, errors="coerce"),
    }
    if stablecoin_float is not None:
        # Must join at construction so the union calendar includes the
        # stablecoin series' own release dates.
        cols["stable"] = pd.to_numeric(stablecoin_float, errors="coerce")
    frame = pd.DataFrame(cols).sort_index()
    frame = frame.ffill()  # carry each last-known release forward (causal)
    net = frame["walcl"] - frame["rrp"] - frame["tga"]
    if stablecoin_float is not None:
        net = net + frame["stable"]
    return net.rename("net_liquidity")


def net_liquidity_change(net_liq: pd.Series, periods: int = 5) -> pd.Series:
    """Change in net liquidity over ``periods`` bars (default ~1 week of dailies).

    The level is near-non-stationary; its *change* is the regime-relevant
    quantity (liquidity expanding vs draining). Causal difference.
    """
    if periods < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")
    return pd.to_numeric(net_liq, errors="coerce").diff(periods).rename("net_liquidity_change")
