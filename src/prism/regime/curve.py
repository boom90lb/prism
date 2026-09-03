"""Yield-curve regime state (SPEC.md §3 law 2, §7.5).

Level / slope / curvature via **fixed** Litterman-Scheinkman-shaped contrasts,
NOT rolling PCA. Fixed contrasts are chosen deliberately: rolling PCA on yields
introduces eigenvector sign/rotation instability and subtle in-sample lookahead
if fit on the evaluation window, whereas the three fixed contrasts are the stable,
leak-free, standard shapes (Litterman & Scheinkman 1991) and are all a regime
state variable needs. These are conditioning features, gated on incremental IC
(I-8) before they touch sizing — never a tradable sleeve.

Inputs are constant-maturity Treasury (CMT) par yields from free sources
(Treasury.gov Daily Par Yield Curve, or FRED DGS-series); see ``regime.sources``.
Everything here is pure and causal: each row's state uses only that day's yields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Tenors are in years; 3-month = 0.25.
DEFAULT_LEVEL_TENORS = (2.0, 5.0, 10.0)
DEFAULT_SLOPE = (0.25, 10.0)  # long minus short: 10y - 3m
DEFAULT_CURVATURE = (2.0, 5.0, 10.0)  # butterfly: 2*belly - short_wing - long_wing


def _nearest(available: np.ndarray, tenor: float) -> int:
    return int(np.argmin(np.abs(available - tenor)))


def curve_state(
    yields: pd.Series,
    *,
    level_tenors: tuple[float, ...] = DEFAULT_LEVEL_TENORS,
    slope: tuple[float, float] = DEFAULT_SLOPE,
    curvature: tuple[float, float, float] = DEFAULT_CURVATURE,
) -> dict[str, float]:
    """Level / slope / curvature from one day's CMT yields.

    ``yields`` is indexed by tenor in years (e.g. ``{0.25, 2, 5, 10, 30}``) with
    yields in percent. Tenors are matched to the nearest available point, so a
    partial curve still yields a state (with a mild approximation). Contracts:

    * ``level`` — mean yield over ``level_tenors`` (parallel-shift proxy).
    * ``slope`` — ``y(long) - y(short)`` (steepening > 0, inversion < 0).
    * ``curvature`` — ``2*y(belly) - y(short_wing) - y(long_wing)`` (butterfly;
      positive = humped belly).

    Returns NaN fields if fewer than two finite tenor points exist.
    """
    y = pd.to_numeric(yields, errors="coerce")
    y = y[np.isfinite(y.to_numpy(dtype=float))]
    if y.shape[0] < 2:
        return {"level": float("nan"), "slope": float("nan"), "curvature": float("nan")}
    tenors = np.asarray([float(t) for t in y.index], dtype=float)
    vals = y.to_numpy(dtype=float)

    level = float(np.mean([vals[_nearest(tenors, t)] for t in level_tenors]))
    slope_val = float(vals[_nearest(tenors, slope[1])] - vals[_nearest(tenors, slope[0])])
    short_w, belly, long_w = curvature
    curv = float(
        2.0 * vals[_nearest(tenors, belly)]
        - vals[_nearest(tenors, short_w)]
        - vals[_nearest(tenors, long_w)]
    )
    return {"level": level, "slope": slope_val, "curvature": curv}


def curve_state_panel(
    yields_panel: pd.DataFrame,
    *,
    level_tenors: tuple[float, ...] = DEFAULT_LEVEL_TENORS,
    slope: tuple[float, float] = DEFAULT_SLOPE,
    curvature: tuple[float, float, float] = DEFAULT_CURVATURE,
) -> pd.DataFrame:
    """Apply :func:`curve_state` per date over a (dates x tenor) yield panel.

    Columns are tenors in years. Returns a (dates x {level, slope, curvature})
    frame; each row is computed only from that date's yields (causal by
    construction).
    """
    # No pre-cast/copy needed: curve_state coerces each row's tenor index itself.
    rows = {
        idx: curve_state(row, level_tenors=level_tenors, slope=slope, curvature=curvature)
        for idx, row in yields_panel.iterrows()
    }
    return pd.DataFrame.from_dict(rows, orient="index")[["level", "slope", "curvature"]]
