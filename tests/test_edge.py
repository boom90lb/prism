"""EDGE effective-spread estimator (Ardia-Guidotti-Kroencke 2024).

Correctness guards:
  1. Golden anchor — the shipped port reproduces the CANONICAL upstream estimator
     (github.com/eguidotti/bidask) bit-for-bit on fixed literal inputs. The golden
     values below were produced by running the verbatim reference ``edge()`` offline,
     independently of this port, so a shared transcription error (e.g. the
     variance-optimal GMM weighting) fails the rel=1e-9 assertion rather than passing
     a copy-vs-copy check.
  2. Recovery — the estimator recovers a known injected spread from a simulated
     bid-ask bounce.
  3. The bracketing diagnostic (docs/edge_preregistration.md) places the schedule
     above/inside/below a genuine EDGE inter-quartile band correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.execution.edge import (
    SPREAD_BUCKET_SCHEDULE_V1,
    edge_bracketing_diagnostic,
    edge_spread,
    edge_spread_by_symbol,
)


def _arr(space_separated: str) -> list[float]:
    return [float(x) for x in space_separated.split()]


# Golden fixtures: fixed OHLC inputs and the CANONICAL bidask/edge.py output on them
# (computed offline by the verbatim upstream reference, not this port).
CASE_A_OPEN = _arr(
    "100.2825 99.4175 98.1639 96.5576 94.1431 94.1856 92.3683 90.9341 90.7679 88.7796 89.1026 89.3432 90.6089 90.8048"
)
CASE_A_HIGH = _arr(
    "101.7524 101.0253 100.1984 97.1805 95.6874 95.1563 93.2564 93.5796 92.1944 90.7132 91.288 91.7768 92.3837 93.4569"
)
CASE_A_LOW = _arr(
    "98.5207 97.2412 96.8291 94.2781 93.9309 91.8139 90.6154 90.9341 90.1297 88.0856 88.5628 89.3432 89.7454 90.8048"
)
CASE_A_CLOSE = _arr(
    "100.0728 98.4827 96.8291 94.7755 95.3297 92.6822 92.1841 91.8239 90.2376 88.8313 90.2512 89.9085 91.8253 90.86"
)
CASE_A_EDGE = 0.006302338730321759

CASE_B_OPEN = _arr(
    "100.8466 100.5305 100.6926 99.4655 98.2509 97.9051 96.4156 95.1672 96.7493 95.2702 93.8776 92.979 95.7334 91.8848 93.1024 91.6473"
)
CASE_B_HIGH = _arr(
    "101.1698 101.7594 100.6926 99.8324 98.5588 98.0721 97.6174 96.8169 97.6809 96.482 94.398 95.4859 95.7334 92.9969 93.3641 92.4158"
)
CASE_B_LOW = _arr(
    "99.2352 100.1279 98.8191 97.3674 97.18 96.5273 95.0465 95.1672 95.3787 93.3899 91.6995 92.979 91.973 90.993 91.6015 89.7286"
)
CASE_B_CLOSE = _arr(
    "100.6807 101.497 99.4885 97.3674 97.18 96.5273 95.1034 95.8897 95.8918 93.7622 93.2457 95.4859 92.5255 92.9969 91.6015 90.6905"
)
CASE_B_EDGE = 0.00578089671637587

# A short, zero-spread noisy series whose variance-optimal estimate is negative —
# the small-sample artifact that sign=True is documented to expose.
CASE_NEG_OPEN = _arr("98.4756 90.2074 85.2615 80.7446 83.7324 90.7333")
CASE_NEG_HIGH = _arr("98.4756 96.2682 89.8026 85.2904 91.6407 104.7031")
CASE_NEG_LOW = _arr("88.714 84.3401 77.9038 79.7617 83.5912 89.551")
CASE_NEG_CLOSE = _arr("90.9104 86.6169 78.4886 83.1488 90.9691 97.4186")
CASE_NEG_EDGE = -0.024708448362586018


def _simulate_ohlc(n_days, trades_per_day, spread, sigma, seed):
    """OHLC bars from an efficient random walk observed with a ``spread`` bid-ask
    bounce (Roll model): each intraday trade prints at the efficient price times
    ``(1 ± spread/2)``. EDGE should recover ``spread`` (the full effective spread)."""
    rng = np.random.default_rng(seed)
    o = np.empty(n_days)
    h = np.empty(n_days)
    low = np.empty(n_days)
    c = np.empty(n_days)
    p = 100.0
    for d in range(n_days):
        efficient = p * np.exp(sigma * rng.standard_normal(trades_per_day)).cumprod()
        q = rng.choice([-1.0, 1.0], size=trades_per_day)
        observed = efficient * (1.0 + spread / 2.0 * q)
        p = float(efficient[-1])
        o[d], c[d], h[d], low[d] = observed[0], observed[-1], observed.max(), observed.min()
    return o, h, low, c


@pytest.mark.parametrize(
    "o,h,low,c,sign,golden",
    [
        (CASE_A_OPEN, CASE_A_HIGH, CASE_A_LOW, CASE_A_CLOSE, False, CASE_A_EDGE),
        (CASE_B_OPEN, CASE_B_HIGH, CASE_B_LOW, CASE_B_CLOSE, False, CASE_B_EDGE),
        (CASE_NEG_OPEN, CASE_NEG_HIGH, CASE_NEG_LOW, CASE_NEG_CLOSE, True, CASE_NEG_EDGE),
    ],
)
def test_edge_spread_matches_canonical_golden(o, h, low, c, sign, golden):
    # rel=1e-9 is tight enough that any deviation from the canonical GMM combination
    # (e.g. a naive same-variance weighting, or an r2/r5 swap) fails here.
    assert edge_spread(o, h, low, c, sign=sign) == pytest.approx(golden, rel=1e-9)


@pytest.mark.parametrize("spread", [0.005, 0.02])
def test_edge_spread_recovers_known_spread(spread):
    o, h, low, c = _simulate_ohlc(n_days=1500, trades_per_day=100, spread=spread, sigma=0.001, seed=7)
    est = edge_spread(o, h, low, c)
    assert est == pytest.approx(spread, rel=0.35)


def test_edge_spread_sign_exposes_negative_small_sample_estimate():
    signed = edge_spread(CASE_NEG_OPEN, CASE_NEG_HIGH, CASE_NEG_LOW, CASE_NEG_CLOSE, sign=True)
    unsigned = edge_spread(CASE_NEG_OPEN, CASE_NEG_HIGH, CASE_NEG_LOW, CASE_NEG_CLOSE)
    assert signed < 0.0  # sign=True exposes the small-sample negative artifact
    assert unsigned == pytest.approx(abs(signed), rel=1e-12)  # sign=False returns the magnitude


def test_edge_spread_nan_for_too_few_bars():
    assert np.isnan(edge_spread([100.0, 101.0], [101.0, 102.0], [99.0, 100.0], [100.0, 101.0]))


def test_edge_spread_nan_for_degenerate_constant_bars():
    flat = [100.0] * 50
    assert np.isnan(edge_spread(flat, flat, flat, flat))


def test_edge_spread_length_mismatch_raises():
    with pytest.raises(ValueError):
        edge_spread([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0])


def test_edge_spread_tolerates_nans_in_series():
    o, h, low, c = _simulate_ohlc(400, 60, spread=0.01, sigma=0.001, seed=3)
    o[5] = np.nan  # a single missing print must not blow up the estimator
    assert np.isfinite(edge_spread(o, h, low, c))


def _panel(symbol_specs, n_days, seed0):
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    cols = {"open": {}, "high": {}, "low": {}, "close": {}}
    for i, (sym, spread) in enumerate(symbol_specs):
        o, h, low, c = _simulate_ohlc(n_days, 60, spread=spread, sigma=0.001, seed=seed0 + i)
        cols["open"][sym], cols["high"][sym], cols["low"][sym], cols["close"][sym] = o, h, low, c
    return (pd.DataFrame(cols[k], index=idx) for k in ("open", "high", "low", "close"))


def test_edge_spread_by_symbol_orders_by_spread():
    open_p, high, low, close = _panel([("TIGHT", 0.004), ("WIDE", 0.02)], n_days=500, seed0=11)
    bps = edge_spread_by_symbol(open_p, high, low, close, min_obs=60)
    assert bps.name == "edge_oneway_bps"
    # one-way bps = spread/2 * 1e4: TIGHT ~ 20, WIDE ~ 100
    assert bps["TIGHT"] < bps["WIDE"]
    assert bps["TIGHT"] == pytest.approx(0.004 / 2 * 1e4, rel=0.4)
    assert bps["WIDE"] == pytest.approx(0.02 / 2 * 1e4, rel=0.4)


def test_edge_spread_by_symbol_below_min_obs_is_nan():
    open_p, high, low, close = _panel([("AAA", 0.01)], n_days=40, seed0=1)
    bps = edge_spread_by_symbol(open_p, high, low, close, min_obs=60)
    assert np.isnan(bps["AAA"])


def test_edge_spread_by_symbol_misaligned_raises():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    df = pd.DataFrame({"AAA": np.linspace(100.0, 110.0, 100)}, index=idx)
    mislabelled = df.rename(columns={"AAA": "BBB"})
    with pytest.raises(ValueError):
        edge_spread_by_symbol(df, df, df, mislabelled)


def test_edge_spread_by_symbol_rejects_min_obs_below_3():
    open_p, high, low, close = _panel([("AAA", 0.01)], n_days=100, seed0=5)
    with pytest.raises(ValueError):
        edge_spread_by_symbol(open_p, high, low, close, min_obs=2)


def test_edge_bracketing_diagnostic_positions_single_name_buckets():
    # One name per bucket; per-bucket quartiles collapse to the single value — the
    # degenerate/empty-bucket guard. The genuine IQR band is exercised below.
    edge_bps = pd.Series({"MEGA": 0.5, "BIG": 2.0, "SMALL": 40.0})
    mdv = pd.Series({"MEGA": 600e6, "BIG": 200e6, "SMALL": 5e6})
    table = edge_bracketing_diagnostic(edge_bps, mdv, SPREAD_BUCKET_SCHEDULE_V1)
    by_floor = {row["floor"]: row for _, row in table.iterrows()}
    assert by_floor[500e6]["position"] == "above"  # schedule 1.0 > edge 0.5
    assert by_floor[100e6]["position"] == "inside"  # schedule 2.0 == edge 2.0
    assert by_floor[0.0]["position"] == "below"  # schedule 10.0 < edge 40.0
    assert by_floor[25e6]["n_names"] == 0 and by_floor[25e6]["position"] == "empty"
    assert by_floor[500e6]["schedule_bps"] == 1.0


def test_edge_bracketing_diagnostic_iqr_band_multi_name():
    # Five names per populated bucket with EDGE one-way bps [1,2,3,4,5] -> a genuine
    # IQR (np.percentile gives p25=2, p50=3, p75=4). A custom schedule places one
    # bucket inside the band, one above p75, one below p25 — so the percentile
    # columns and the p25-vs-p75 band operands are exercised, not a collapsed value.
    floors = (500e6, 100e6, 25e6, 0.0)
    schedule = ((500e6, 3.0), (100e6, 6.0), (25e6, 0.5), (0.0, 10.0))
    edge_bps, mdv = {}, {}
    for bucket, mdv_val in [("mega", 600e6), ("big", 200e6), ("mid", 50e6)]:
        for j, bps in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            edge_bps[f"{bucket}{j}"] = bps
            mdv[f"{bucket}{j}"] = mdv_val
    table = edge_bracketing_diagnostic(pd.Series(edge_bps), pd.Series(mdv), schedule, floors=floors)
    by_floor = {row["floor"]: row for _, row in table.iterrows()}
    mega = by_floor[500e6]
    assert (mega["edge_p25"], mega["edge_p50"], mega["edge_p75"]) == (2.0, 3.0, 4.0)
    assert mega["n_names"] == 5
    assert mega["position"] == "inside"  # schedule 3.0 strictly inside [2.0, 4.0]
    assert by_floor[100e6]["position"] == "above"  # schedule 6.0 > p75 4.0
    assert by_floor[25e6]["position"] == "below"  # schedule 0.5 < p25 2.0
    assert by_floor[0.0]["position"] == "empty"


def test_edge_bracketing_diagnostic_schedule_floor_mismatch_raises():
    edge_bps = pd.Series({"AAA": 3.0})
    mdv = pd.Series({"AAA": 300e6})
    bad_schedule = ((500e6, 1.0), (100e6, 2.0), (0.0, 10.0))  # missing the 25e6 floor
    with pytest.raises(ValueError):
        edge_bracketing_diagnostic(edge_bps, mdv, bad_schedule)


def test_edge_bracketing_diagnostic_bad_floors_raises():
    edge_bps = pd.Series({"AAA": 3.0})
    mdv = pd.Series({"AAA": 300e6})
    # Non-descending floors (with a matching schedule) must be rejected before bucketing.
    bad_floors = (100e6, 500e6, 0.0)
    bad_schedule = ((100e6, 2.0), (500e6, 1.0), (0.0, 10.0))
    with pytest.raises(ValueError):
        edge_bracketing_diagnostic(edge_bps, mdv, bad_schedule, floors=bad_floors)


def test_edge_bracketing_diagnostic_unmapped_mdv_raises():
    edge_bps = pd.Series({"AAA": 3.0, "ZZZ": 4.0})
    mdv = pd.Series({"AAA": 300e6})  # ZZZ has an estimate but no liquidity screen
    with pytest.raises(ValueError):
        edge_bracketing_diagnostic(edge_bps, mdv, SPREAD_BUCKET_SCHEDULE_V1)
