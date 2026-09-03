"""EDGE effective-spread estimator from OHLC bars (SPEC.md §6 I-9, the R2 cost harness).

The second cost-measurement instrument beside the fills ledger
(:mod:`prism.execution.spread`) — and the one that needs no fills. The estimator
of Ardia, Guidotti & Kroencke (2024, *Journal of Financial Economics*, "Efficient
estimation of bid-ask spreads from open, high, low, and close prices",
https://doi.org/10.1016/j.jfineco.2024.103916; reference implementation
github.com/eguidotti/bidask) recovers the *effective* bid-ask spread from daily
OHLC bars alone — exactly the panel already on disk, per name, over the whole
history, at $0.

**Evidentiary status — pre-registered (docs/edge_preregistration.md).** EDGE is a
*bracketing diagnostic*: :func:`edge_bracketing_diagnostic` reports whether the
pre-registered conservative-upper bucket schedule (``SPREAD_BUCKET_SCHEDULE_V1``,
docs/r2_design.md §3) sits above, inside, or below the EDGE per-bucket
distribution. It is **not** a calibration authority — the paper/live fills ledger
remains the sole trigger of the §10 historical-verdict recompute (docs/handoff.md).
EDGE moves no ratified statistic and introduces no counted trial.

**What the number is.** :func:`edge_spread` returns the *full effective spread* as
a fraction of price (``0.01`` = a 1% spread). The bucket schedule is quoted in
*one-way* bps (a half-spread), so the panel and the diagnostic convert
``one_way_bps = edge_fraction / 2 * 1e4``. EDGE measures the effective spread of
*all* trades; marketable retail flow often price-improves inside it, so for this
book EDGE reads as an upper-ish bound — the conservative direction for a cost
harness, and the reason it can only bracket, not calibrate.
"""

from __future__ import annotations

import math
import warnings
from typing import Union

import numpy as np
import pandas as pd

from prism.execution.spread import DEFAULT_BUCKET_FLOORS

ArrayLike = Union[np.ndarray, pd.Series]

# A name needs enough bars for the moment conditions to be estimable; below this
# floor the per-name EDGE is NaN (absent from its bucket distribution, counted,
# never a silent zero — N7). AGK report stability from a few dozen observations;
# 60 (~a quarter of trading days) is a deliberately conservative floor.
DEFAULT_MIN_OBS = 60

# The pre-registered conservative-upper schedule EDGE brackets against, duplicated
# by value from the research WFO (docs/r2_design.md §3; production never imports
# research, SPEC §9). ``(dollar-volume floor, one-way bps)``, descending, ending at
# the 0.0 catch-all — the same floors as ``DEFAULT_BUCKET_FLOORS``.
SPREAD_BUCKET_SCHEDULE_V1: tuple[tuple[float, float], ...] = (
    (500e6, 1.0),
    (100e6, 2.0),
    (25e6, 5.0),
    (0.0, 10.0),
)


def edge_spread(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    *,
    sign: bool = False,
) -> float:
    """Effective bid-ask spread from OHLC bars (Ardia-Guidotti-Kroencke 2024).

    A faithful port of the reference estimator (github.com/eguidotti/bidask): the
    asymptotically unbiased, variance-optimal GMM combination of the two moment
    conditions built from log open/high/low/close and the geometric midrange
    ``m = (log-high + log-low) / 2``. Returns the effective spread as a fraction of
    price (``0.01`` = 1%); ``sign=True`` returns the signed root of the
    variance-optimal estimate (a negative value is a small-sample artifact, not a
    negative spread).

    Returns NaN when the window is unestimable — fewer than 3 bars, fewer than 2
    non-degenerate bars, or no open/close price variation to identify the
    indicators — reported honestly (N7), never zeroed. Inputs are aligned,
    ascending-by-time price vectors of equal length; NaNs are tolerated (the
    estimator uses nan-aware moments internally).
    """
    o = np.asarray(open, dtype=float)
    h = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)

    nobs = o.shape[0]
    if h.shape[0] != nobs or low_arr.shape[0] != nobs or c.shape[0] != nobs:
        raise ValueError("open, high, low, close must have equal length (N7)")
    if nobs < 3:
        return float("nan")

    o = np.log(o)
    h = np.log(h)
    l = np.log(low_arr)
    c = np.log(c)
    m = (h + l) / 2.0

    h1, l1, c1, m1 = h[:-1], l[:-1], c[:-1], m[:-1]
    o, h, l, c, m = o[1:], h[1:], l[1:], c[1:], m[1:]

    r1 = m - o
    r2 = o - m1
    r3 = m - c1
    r4 = c1 - m1
    r5 = o - c1

    tau = np.where(np.isnan(h) | np.isnan(l) | np.isnan(c1), np.nan, (h != l) | (l != c1))
    po1 = tau * np.where(np.isnan(o) | np.isnan(h), np.nan, o != h)
    po2 = tau * np.where(np.isnan(o) | np.isnan(l), np.nan, o != l)
    pc1 = tau * np.where(np.isnan(c1) | np.isnan(h1), np.nan, c1 != h1)
    pc2 = tau * np.where(np.isnan(c1) | np.isnan(l1), np.nan, c1 != l1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pt = np.nanmean(tau)
        po = np.nanmean(po1) + np.nanmean(po2)
        pc = np.nanmean(pc1) + np.nanmean(pc2)

        if np.nansum(tau) < 2 or po == 0 or pc == 0:
            return float("nan")

        d1 = r1 - np.nanmean(r1) / pt * tau
        d3 = r3 - np.nanmean(r3) / pt * tau
        d5 = r5 - np.nanmean(r5) / pt * tau

        x1 = -4.0 / po * d1 * r2 + -4.0 / pc * d3 * r4
        x2 = -4.0 / po * d1 * r5 + -4.0 / pc * d5 * r4

        e1 = np.nanmean(x1)
        e2 = np.nanmean(x2)

        v1 = np.nanmean(x1**2) - e1**2
        v2 = np.nanmean(x2**2) - e2**2

    vt = v1 + v2
    s2 = (v2 * e1 + v1 * e2) / vt if vt > 0 else (e1 + e2) / 2.0

    s = math.sqrt(abs(s2))
    if sign:
        s *= float(np.sign(s2))
    return float(s)


def edge_spread_by_symbol(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    min_obs: int = DEFAULT_MIN_OBS,
) -> pd.Series:
    """Per-symbol EDGE effective spread, in one-way bps, over aligned OHLC panels.

    ``open``/``high``/``low``/``close`` are wide panels (dates × symbols) sharing
    an index and columns — a misaligned set is a data defect and raises (N7). For
    each symbol the full column is passed to :func:`edge_spread` (NaNs tolerated);
    a symbol with fewer than ``min_obs`` finite closes yields NaN (absent from the
    bracket, counted, never a silent zero). The result is one-way bps
    (``edge_fraction / 2 * 1e4``) — the convention the bucket schedule quotes, so
    the two are directly comparable.
    """
    if min_obs < 3:
        raise ValueError(f"min_obs must be >= 3 (EDGE needs 3 bars), got {min_obs}")
    panels = {"open": open, "high": high, "low": low, "close": close}
    ref_index, ref_columns = close.index, close.columns
    for name, panel in panels.items():
        if not panel.index.equals(ref_index) or not panel.columns.equals(ref_columns):
            raise ValueError(f"{name} panel is not aligned with close (index and columns must match) (N7)")

    out: dict[str, float] = {}
    for symbol in ref_columns:
        o = pd.to_numeric(open[symbol], errors="coerce").to_numpy(dtype=float)
        h = pd.to_numeric(high[symbol], errors="coerce").to_numpy(dtype=float)
        low_col = pd.to_numeric(low[symbol], errors="coerce").to_numpy(dtype=float)
        c = pd.to_numeric(close[symbol], errors="coerce").to_numpy(dtype=float)
        if int(np.isfinite(c).sum()) < min_obs:
            out[str(symbol)] = float("nan")
            continue
        fraction = edge_spread(o, h, low_col, c)
        out[str(symbol)] = float("nan") if not math.isfinite(fraction) else fraction / 2.0 * 1e4
    return pd.Series(out, name="edge_oneway_bps")


def edge_bracketing_diagnostic(
    edge_oneway_bps: pd.Series,
    median_dollar_volume: pd.Series,
    schedule: tuple[tuple[float, float], ...],
    *,
    floors: tuple[float, ...] = DEFAULT_BUCKET_FLOORS,
) -> pd.DataFrame:
    """Bracket the conservative-upper bucket schedule against the EDGE distribution.

    For each liquidity bucket — partitioned on ``median_dollar_volume`` by the same
    descending ``floors`` as the fills-ledger calibration, ending at the 0.0
    catch-all — report the count of names with a finite EDGE estimate, the EDGE
    one-way-bps quartiles (``edge_p25``/``edge_p50``/``edge_p75``), the schedule's
    one-way bps for that bucket, and where the schedule sits relative to the EDGE
    inter-quartile band: ``above`` (schedule > p75 — conservative vs the EDGE
    bulk), ``inside`` ([p25, p75]), or ``below`` (schedule < p25 — the schedule
    under-charges vs EDGE, read against the price-improvement caveat in the module
    docstring). This is a *descriptive* read, never a gate: fills remain the
    calibration authority.

    ``schedule`` carries the ``((floor, one_way_bps), …)`` shape of
    ``SPREAD_BUCKET_SCHEDULE_V1`` / the fills-calibrated schedule
    (:func:`prism.execution.spread.calibrated_bucket_schedule`) and must partition
    liquidity on the same floors, or the bracket would compare mismatched buckets
    (raises).
    """
    if not floors or sorted(floors, reverse=True) != list(floors) or floors[-1] != 0.0:
        raise ValueError(f"floors must be descending and end at the 0.0 catch-all, got {floors}")
    schedule_floors = [float(f) for f, _ in schedule]
    if schedule_floors != [float(f) for f in floors]:
        raise ValueError(
            f"schedule floors {schedule_floors} != bucket floors {[float(f) for f in floors]}; "
            "the schedule must partition liquidity on the diagnostic's floors"
        )

    edge = pd.to_numeric(edge_oneway_bps, errors="coerce")
    edge = edge[np.isfinite(edge)]
    mdv = pd.to_numeric(median_dollar_volume, errors="coerce")
    unmapped = sorted(str(s) for s in set(edge.index) - set(mdv[np.isfinite(mdv)].index))
    if unmapped:
        raise ValueError(
            f"no finite median_dollar_volume for names with an EDGE estimate {unmapped[:5]}…; "
            "supply the liquidity statistic for every bracketed name (N7)"
        )
    edge_vals = edge.to_numpy(dtype=float)
    name_mdv = edge.index.map(mdv).to_numpy(dtype=float)

    rows = []
    ceiling = np.inf
    for floor, sched_bps in schedule:
        in_bucket = (name_mdv >= floor) & (name_mdv < ceiling)
        ceiling = floor
        bucket_edge = edge_vals[in_bucket]
        n = int(bucket_edge.size)
        if n:
            p25, p50, p75 = (float(np.percentile(bucket_edge, q)) for q in (25, 50, 75))
            if float(sched_bps) > p75:
                position = "above"
            elif float(sched_bps) < p25:
                position = "below"
            else:
                position = "inside"
        else:
            p25 = p50 = p75 = float("nan")
            position = "empty"
        rows.append(
            {
                "floor": float(floor),
                "schedule_bps": float(sched_bps),
                "n_names": n,
                "edge_p25": p25,
                "edge_p50": p50,
                "edge_p75": p75,
                "position": position,
            }
        )
    return pd.DataFrame(rows)
