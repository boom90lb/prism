"""Joint crash diagnostic for multi-sleeve stress receipts (W1 / G4a).

Uncounted engineering surface for ``docs/v040_program.md`` W1 and the
aim-portfolio G4a gate: given two (or more) daily return streams and named
stress windows, report sleeve-alone and joint max drawdown, crash-window
return, and a simple fixed-weight blend sensitivity. Searches nothing,
appends to no trial ledger, moves no ratified statistic.

The preferred product narrative wants B1 alone vs B1+trend over 2020-03 and
2022 stress windows — this module is the pure instrument; data availability
and window labels are the caller's problem (local caches may not yet cover
those eras; synthetic tests pin the arithmetic).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    """Peak-to-trough drawdown on a cumulative wealth path (negative or zero)."""
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def window_return(returns: pd.Series, start: str, end: str) -> dict:
    """Total simple return and session count inside ``[start, end]`` (inclusive).

    Dates are compared on normalized session dates (tz stripped). Empty window
    → ``n=0``, ``total_return=None`` (unmeasured, not zero — N7).
    """
    r = _session_series(returns)
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    mask = (r.index >= lo) & (r.index <= hi)
    sub = r.loc[mask]
    n = int(len(sub))
    if n == 0:
        return {"n": 0, "total_return": None, "note": "empty window"}
    total = float((1.0 + sub).prod() - 1.0)
    return {"n": n, "total_return": total}


def blend_returns(
    sleeves: Mapping[str, pd.Series],
    weights: Mapping[str, float],
) -> pd.Series:
    """Fixed-weight sum of sleeve returns on the joint session index.

    Weights must be non-negative and sum to a positive finite total; they are
    renormalized to 1. Missing a sleeve on a session contributes 0 for that
    sleeve that day (cash), not forward-fill. This is a *sensitivity*
    instrument — not optimized aim-portfolio weights (G4b owns those).
    """
    if not sleeves:
        raise ValueError("sleeves must be non-empty")
    w = {k: float(weights.get(k, 0.0)) for k in sleeves}
    if any(v < 0.0 or not np.isfinite(v) for v in w.values()):
        raise ValueError(f"weights must be finite and >= 0, got {w}")
    total_w = sum(w.values())
    if total_w <= 0.0:
        raise ValueError(f"weights must sum to a positive total, got {w}")
    w = {k: v / total_w for k, v in w.items()}
    frame = pd.concat(
        {k: _session_series(s) for k, s in sleeves.items()},
        axis=1,
        join="outer",
    ).sort_index()
    frame = frame.fillna(0.0)
    blended = sum(frame[k] * w[k] for k in sleeves)
    return blended.rename("blend")


def joint_crash_report(
    sleeves: Mapping[str, pd.Series],
    windows: Mapping[str, tuple[str, str]],
    *,
    blend_weights: Mapping[str, float] | None = None,
) -> dict:
    """Sleeve-alone and joint stress receipts.

    Parameters
    ----------
    sleeves:
        Name → daily simple return series (any tz; normalized internally).
    windows:
        Name → (start, end) inclusive date strings for stress intervals.
    blend_weights:
        Optional fixed capital weights for a joint blend series. Default equal
        weight across sleeves.

    Returns a JSON-serializable dict with per-sleeve full-sample max DD,
    per-window total returns, and the same for the blend. Uncounted.
    """
    if not sleeves:
        raise ValueError("sleeves must be non-empty")
    if blend_weights is None:
        blend_weights = {k: 1.0 for k in sleeves}
    blend = blend_returns(sleeves, blend_weights)

    def _sleeve_block(series: pd.Series) -> dict:
        block: dict = {
            "n_sessions": int(series.dropna().shape[0]),
            "max_drawdown": max_drawdown(series),
            "windows": {},
        }
        for wname, (start, end) in windows.items():
            block["windows"][wname] = window_return(series, start, end)
        return block

    report: dict = {
        "sleeves": {name: _sleeve_block(_session_series(s)) for name, s in sleeves.items()},
        "blend": {
            "weights": {k: float(blend_weights.get(k, 0.0)) for k in sleeves},
            **_sleeve_block(blend),
        },
        "windows_defined": {k: {"start": a, "end": b} for k, (a, b) in windows.items()},
    }
    return report


def _session_series(returns: pd.Series) -> pd.Series:
    s = pd.to_numeric(returns, errors="coerce")
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    s = pd.Series(s.to_numpy(), index=idx.normalize(), dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s
