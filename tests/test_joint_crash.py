"""Unit tests for the joint crash diagnostic (uncounted W1/G4a instrument)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.validation.joint_crash import (
    blend_returns,
    capital_allocation_sensitivity,
    joint_crash_report,
    max_drawdown,
    window_return,
)


def _series(vals: list[float], start: str = "2020-01-02") -> pd.Series:
    idx = pd.date_range(start, periods=len(vals), freq="B", tz="America/New_York")
    return pd.Series(vals, index=idx, dtype=float)


def test_max_drawdown_on_known_path():
    # wealth path: 1 → 1.1 → 0.55 → peak-to-trough = 0.55/1.1 - 1 = -0.5
    r = _series([0.10, 0.55 / 1.1 - 1.0])
    assert max_drawdown(r) == pytest.approx(-0.5, rel=1e-9)


def test_window_return_empty_is_unmeasured():
    r = _series([0.01, 0.02], start="2021-01-04")
    out = window_return(r, "2020-03-01", "2020-03-31")
    assert out["n"] == 0
    assert out["total_return"] is None


def test_window_return_product():
    r = _series([0.10, -0.10], start="2020-03-02")
    out = window_return(r, "2020-03-01", "2020-03-10")
    assert out["n"] == 2
    assert out["total_return"] == pytest.approx(1.1 * 0.9 - 1.0)


def test_blend_equal_weight():
    a = _series([0.02, 0.00])
    b = _series([0.00, 0.02])
    blend = blend_returns({"a": a, "b": b}, {"a": 1.0, "b": 1.0})
    assert blend.iloc[0] == pytest.approx(0.01)
    assert blend.iloc[1] == pytest.approx(0.01)


def test_blend_rejects_negative_weights():
    a = _series([0.01])
    with pytest.raises(ValueError, match="weights"):
        blend_returns({"a": a}, {"a": -1.0})


def test_joint_report_records_convexity_shape():
    # B1 crashes hard in the stress window; trend is flat-to-positive (CTA smile toy).
    idx = pd.date_range("2020-02-03", periods=40, freq="B", tz="America/New_York")
    b1 = pd.Series(0.001, index=idx)
    trend = pd.Series(0.0005, index=idx)
    # crash window mid-sample: B1 -5%/day for 5 days, trend +2%/day
    crash = idx[15:20]
    b1.loc[crash] = -0.05
    trend.loc[crash] = 0.02
    report = joint_crash_report(
        {"b1": b1, "trend": trend},
        {"crash": (str(crash[0].date()), str(crash[-1].date()))},
        blend_weights={"b1": 0.7, "trend": 0.3},
    )
    b1_win = report["sleeves"]["b1"]["windows"]["crash"]["total_return"]
    tr_win = report["sleeves"]["trend"]["windows"]["crash"]["total_return"]
    bl_win = report["blend"]["windows"]["crash"]["total_return"]
    assert b1_win is not None and b1_win < -0.2
    assert tr_win is not None and tr_win > 0.0
    assert bl_win is not None and b1_win < bl_win < tr_win
    assert report["blend"]["max_drawdown"] > report["sleeves"]["b1"]["max_drawdown"]  # less bad


def test_capital_allocation_sensitivity_grid_is_monotone_in_stress():
    """Pure G4a arithmetic: more trend weight improves crash-window total return."""
    idx = pd.date_range("2020-02-03", periods=40, freq="B", tz="America/New_York")
    b1 = pd.Series(0.001, index=idx)
    trend = pd.Series(0.0005, index=idx)
    crash = idx[15:20]
    b1.loc[crash] = -0.05
    trend.loc[crash] = 0.02
    windows = {"crash": (str(crash[0].date()), str(crash[-1].date()))}
    sens = capital_allocation_sensitivity(
        {"b1": b1, "trend": trend},
        windows,
        primary="b1",
        primary_weights=[0.0, 0.5, 1.0],
    )
    assert sens["gate"] == "G4a"
    assert sens["status"] == "uncounted_diagnostic"
    assert len(sens["rows"]) == 3
    rets = [row["windows"]["crash"]["total_return"] for row in sens["rows"]]
    # primary=b1 weight 0 → all trend (best crash), 1 → all b1 (worst crash)
    assert rets[0] > rets[1] > rets[2]


def test_capital_allocation_sensitivity_requires_two_sleeves():
    a = _series([0.01])
    with pytest.raises(ValueError, match="exactly two"):
        capital_allocation_sensitivity({"a": a}, {"w": ("2020-01-01", "2020-01-02")})
