"""Tests for the regime feature blocks (SPEC.md §7.5, laws 1/2/4).

Pure feature math only; the live FRED/Treasury/CBOE fetch adapter is out of scope
(network-gated, SPEC.md §11). These pin the contracts and causality.
"""

import numpy as np
import pandas as pd
import pytest

from prism.regime import (
    breakeven_divergence,
    curve_state,
    curve_state_panel,
    inflation_state,
    net_liquidity,
    net_liquidity_change,
    realized_volatility,
    variance_risk_premium,
    vix_term_slope,
)


class TestCurveState:
    def test_level_slope_curvature_contracts(self):
        y = pd.Series({0.25: 4.5, 2.0: 4.2, 5.0: 4.0, 10.0: 4.3, 30.0: 4.7})
        s = curve_state(y)
        assert s["level"] == pytest.approx((4.2 + 4.0 + 4.3) / 3)
        assert s["slope"] == pytest.approx(4.3 - 4.5)  # 10y - 3m (inverted here)
        assert s["curvature"] == pytest.approx(2 * 4.0 - 4.2 - 4.3)  # butterfly

    def test_inversion_has_negative_slope(self):
        y = pd.Series({0.25: 5.0, 10.0: 4.0})
        assert curve_state(y)["slope"] < 0

    def test_nearest_tenor_matching_partial_curve(self):
        # Missing exact tenors still yields a state via nearest match.
        y = pd.Series({0.5: 4.6, 7.0: 4.1, 20.0: 4.4})
        s = curve_state(y)
        assert np.isfinite(s["level"]) and np.isfinite(s["slope"])

    def test_degenerate_single_point(self):
        s = curve_state(pd.Series({10.0: 4.0}))
        assert np.isnan(s["level"])

    def test_panel_is_row_causal(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        panel = pd.DataFrame(
            {0.25: [4.5, 4.6, 4.7], 2.0: [4.2, 4.3, 4.4], 10.0: [4.0, 4.1, 4.2]}, index=idx
        )
        out = curve_state_panel(panel)
        assert list(out.columns) == ["level", "slope", "curvature"]
        # Row 0 depends only on row 0's yields.
        assert out.iloc[0]["slope"] == pytest.approx(4.0 - 4.5)


class TestVol:
    def test_realized_vol_annualized_pct(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0, 0.01, 300))
        rv = realized_volatility(r, 20)
        # ~1% daily -> ~16% annualized
        assert 10.0 < rv.dropna().mean() < 22.0

    def test_realized_vol_is_causal(self):
        r = pd.Series([0.0] * 10 + [0.05] * 10)
        rv = realized_volatility(r, 5)
        # First non-NaN appears only after `window` observations.
        assert rv.iloc[:4].isna().all()

    def test_vrp_variance_points(self):
        assert variance_risk_premium(20.0, 15.0) == pytest.approx(400 - 225)

    def test_vrp_series_broadcast_scalar(self):
        iv = pd.Series([20.0, 22.0], index=["a", "b"])
        out = variance_risk_premium(iv, 15.0)
        assert out["a"] == pytest.approx(400 - 225)

    def test_term_slope_backwardation_and_contango(self):
        assert vix_term_slope(25.0, 20.0) == pytest.approx(1.25)  # backwardation > 1
        assert vix_term_slope(15.0, 20.0) == pytest.approx(0.75)  # contango < 1

    def test_term_slope_difference_mode(self):
        assert vix_term_slope(15.0, 20.0, as_ratio=False) == pytest.approx(5.0)


class TestLiquidity:
    def test_net_liquidity_formula(self):
        idx = pd.to_datetime(["2026-01-05"])
        nl = net_liquidity(
            pd.Series([8e6], index=idx), pd.Series([1e6], index=idx), pd.Series([5e5], index=idx)
        )
        assert nl.iloc[0] == pytest.approx(8e6 - 1e6 - 5e5)

    def test_forward_fill_is_causal_not_backfilled(self):
        # Weekly WALCL carried forward to daily RRP/TGA dates; pre-history stays NaN.
        wal = pd.Series([8e6], index=pd.to_datetime(["2026-01-08"]))
        rrp = pd.Series([1e6, 1e6], index=pd.to_datetime(["2026-01-05", "2026-01-09"]))
        tga = pd.Series([5e5, 5e5], index=pd.to_datetime(["2026-01-05", "2026-01-09"]))
        nl = net_liquidity(wal, rrp, tga)
        assert np.isnan(nl.loc["2026-01-05"])  # WALCL not yet known -> no backfill
        assert nl.loc["2026-01-09"] == pytest.approx(8e6 - 1e6 - 5e5)

    def test_change_is_diff(self):
        nl = pd.Series([1.0, 2.0, 4.0, 7.0])
        assert net_liquidity_change(nl, 1).tolist()[1:] == [1.0, 2.0, 3.0]

    def test_stablecoin_float_is_additive_fourth_term(self):
        idx = pd.to_datetime(["2026-01-05"])
        base = net_liquidity(
            pd.Series([8e6], index=idx), pd.Series([1e6], index=idx), pd.Series([5e5], index=idx)
        )
        with_stable = net_liquidity(
            pd.Series([8e6], index=idx),
            pd.Series([1e6], index=idx),
            pd.Series([5e5], index=idx),
            stablecoin_float=pd.Series([2.5e5], index=idx),
        )
        assert with_stable.iloc[0] == pytest.approx(base.iloc[0] + 2.5e5)

    def test_stablecoin_float_ffills_causally(self):
        # Stablecoin mcap known 01-06 carries to 01-09; pre-history stays NaN.
        idx = pd.to_datetime(["2026-01-05", "2026-01-09"])
        stable = pd.Series([2.5e5], index=pd.to_datetime(["2026-01-06"]))
        nl = net_liquidity(
            pd.Series([8e6, 8e6], index=idx),
            pd.Series([1e6, 1e6], index=idx),
            pd.Series([5e5, 5e5], index=idx),
            stablecoin_float=stable,
        )
        assert np.isnan(nl.loc["2026-01-05"])  # float not yet known -> no backfill
        assert nl.loc["2026-01-09"] == pytest.approx(8e6 - 1e6 - 5e5 + 2.5e5)


class TestInflation:
    def test_breakeven_divergence_vs_target(self):
        be = pd.Series([2.6, 1.8], index=pd.to_datetime(["2026-01-05", "2026-01-06"]))
        div = breakeven_divergence(be)
        assert div.tolist() == pytest.approx([0.6, -0.2])
        assert breakeven_divergence(be, target_pct=3.0).tolist() == pytest.approx([-0.4, -1.2])

    def test_inflation_state_aligns_and_ffills_causally(self):
        real = pd.Series([1.1], index=pd.to_datetime(["2026-01-06"]))
        be = pd.Series([2.4, 2.5], index=pd.to_datetime(["2026-01-05", "2026-01-07"]))
        state = inflation_state(real, be)
        # Union calendar; real yield unknown on 01-05 stays NaN (no backfill).
        assert np.isnan(state.loc["2026-01-05", "real_yield"])
        # Carried forward on 01-07 (causal ffill within each series).
        assert state.loc["2026-01-07", "real_yield"] == pytest.approx(1.1)
        assert state.loc["2026-01-07", "breakeven_divergence"] == pytest.approx(0.5)
