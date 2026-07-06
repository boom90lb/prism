"""R2 mechanics: closed-form band, per-bucket spread (I-9), participation gate.

Everything lands default-off; the frozen-v1 parity claims are pinned here
(explicit-off == default) and by tests/test_residual_statarb.py passing
unchanged. Pre-registered constants (gamma_risk, bucket schedule) come from
docs/dev/R2_DESIGN.md and are asserted, not re-derived.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from prism.config import ExecutionConfig
from prism.execution.participation import max_participation_of
from prism.execution.target_weights import backtest_target_weights
from prism.portfolio.construct import closed_form_band
from research.arbitrage.residual_walk_forward import (
    GAMMA_RISK,
    SPREAD_BUCKET_SCHEDULE_V1,
    _online_banded_targets,
    bucket_spread_bps,
    run_residual_stat_arb_walk_forward,
)
from research.arbitrage.walk_forward import StatArbWalkForwardConfig
from tests.test_residual_statarb import _small_config, _synthetic_panel

_FREE = dict(spread_bps=0, commission_bps=0, slippage_coeff=0, borrow_rate_bps_annual=0)


def _run_wfo(closes, volumes, execution=None, initial_capital=1.0, **walk_overrides):
    walk: dict = dict(formation_bars=80, test_bars=40, min_test_bars=20)
    walk.update(walk_overrides)
    return run_residual_stat_arb_walk_forward(
        closes,
        closes.copy(),  # open == close keeps the test panel simple
        volumes,
        signal_config=_small_config(),
        walk_config=StatArbWalkForwardConfig(**walk),
        execution=execution or ExecutionConfig(**_FREE),
        initial_capital=initial_capital,
    )


# ---------------------------------------------------------------------------
# closed_form_band
# ---------------------------------------------------------------------------


def test_closed_form_band_cube_root_in_cost() -> None:
    var = np.array([0.01, 0.0004])
    base = closed_form_band(var, 1e-4)
    assert (base > 0).all()
    # band(8c) = 2 * band(c): the cube-root law.
    assert np.allclose(closed_form_band(var, 8e-4), 2.0 * base)


def test_closed_form_band_matches_formula() -> None:
    band = closed_form_band(np.array([0.01]), 4e-4, gamma_risk=1.0)[0]
    assert band == pytest.approx((1.5 * 4e-4 * 0.01) ** (1.0 / 3.0))


def test_closed_form_band_degenerate_inputs_disable() -> None:
    bands = closed_form_band(np.array([np.nan, 0.0, -1.0, np.inf, 0.01]), 4e-4)
    assert np.array_equal(bands[:4], np.zeros(4))
    assert bands[4] > 0
    # Degenerate per-name costs disable per name, not per cross-section.
    bands = closed_form_band(np.array([0.01, 0.01, 0.01]), np.array([0.0, np.nan, 4e-4]))
    assert np.array_equal(bands[:2], np.zeros(2))
    assert bands[2] > 0
    assert np.isfinite(closed_form_band(np.array([0.01]), 4e-4)).all()


def test_closed_form_band_gamma_risk_scaling() -> None:
    var = np.array([0.01])
    base = closed_form_band(var, 4e-4, gamma_risk=1.0)
    # band(8*gamma) = band(gamma) / 2; non-positive gamma disables the band.
    assert closed_form_band(var, 4e-4, gamma_risk=8.0)[0] == pytest.approx(base[0] / 2.0)
    assert closed_form_band(var, 4e-4, gamma_risk=0.0)[0] == 0.0
    assert closed_form_band(var, 4e-4, gamma_risk=-1.0)[0] == 0.0


def test_gamma_risk_constant_is_preregistered() -> None:
    assert GAMMA_RISK == 1.0  # docs/dev/R2_DESIGN.md §1: never fitted, never swept


# ---------------------------------------------------------------------------
# Per-name spread in the backtest engine (I-9)
# ---------------------------------------------------------------------------


def _spread_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=8, freq="B")
    open_prices = pd.DataFrame(
        {"A": 100.0 + np.arange(8), "B": 50.0 - 0.5 * np.arange(8)}, index=idx
    )
    weights = pd.DataFrame(
        {"A": [0.0, 0.3, 0.3, -0.2, -0.2, 0.1, 0.0, 0.0], "B": [0.0, -0.1, 0.2, 0.2, 0.0, -0.3, 0.0, 0.0]},
        index=idx,
    )
    return open_prices, weights


def test_flat_spread_parity_none_vs_filled_vector() -> None:
    open_prices, weights = _spread_fixture()
    execution = ExecutionConfig()  # all cost legs live
    base = backtest_target_weights(open_prices, weights, execution=execution)
    vector = backtest_target_weights(
        open_prices,
        weights,
        execution=execution,
        spread_bps_per_name=pd.Series(execution.spread_bps, index=["A", "B"]),
    )
    np.testing.assert_allclose(vector.returns.to_numpy(), base.returns.to_numpy(), rtol=1e-12, atol=0)
    np.testing.assert_allclose(vector.equity.to_numpy(), base.equity.to_numpy(), rtol=1e-12, atol=0)
    np.testing.assert_allclose(vector.costs.to_numpy(), base.costs.to_numpy(), rtol=1e-12, atol=0)


def test_missing_spread_entries_fall_back_to_flat() -> None:
    open_prices, weights = _spread_fixture()
    execution = ExecutionConfig()
    base = backtest_target_weights(open_prices, weights, execution=execution)
    partial = backtest_target_weights(
        open_prices,
        weights,
        execution=execution,
        spread_bps_per_name=pd.Series({"A": np.nan}),  # A NaN, B absent -> both flat
    )
    np.testing.assert_allclose(partial.costs.to_numpy(), base.costs.to_numpy(), rtol=1e-12, atol=0)


def test_bucket_cost_ordering_higher_spread_costs_more() -> None:
    open_prices, weights = _spread_fixture()
    execution = ExecutionConfig(commission_bps=0.0, slippage_coeff=0, borrow_rate_bps_annual=0)
    tight = backtest_target_weights(
        open_prices, weights, execution=execution,
        spread_bps_per_name=pd.Series(1.0, index=["A", "B"]),
    )
    wide = backtest_target_weights(
        open_prices, weights, execution=execution,
        spread_bps_per_name=pd.Series(10.0, index=["A", "B"]),
    )
    assert tight.costs["turnover"].sum() > 0
    # Identical trades, wider per-name spread -> strictly more cost, exactly 10x here.
    np.testing.assert_allclose(tight.costs["turnover"], wide.costs["turnover"], rtol=0, atol=0)
    assert wide.costs["total"].sum() > tight.costs["total"].sum()
    assert wide.costs["commission_spread"].sum() == pytest.approx(
        10.0 * tight.costs["commission_spread"].sum(), rel=1e-12
    )


def test_bucket_spread_schedule_mapping() -> None:
    mdv = pd.Series({"MEGA": 6e8, "LARGE": 2e8, "MID": 5e7, "SMALL": 1e7, "UNKNOWN": np.nan})
    out = bucket_spread_bps(mdv)
    assert out.tolist() == [1.0, 2.0, 5.0, 10.0, 10.0]  # NaN -> widest bucket, fail-safe
    # Schedule constant is the pre-registered one (R2_DESIGN §3).
    assert SPREAD_BUCKET_SCHEDULE_V1 == ((500e6, 1.0), (100e6, 2.0), (25e6, 5.0), (0.0, 10.0))


# ---------------------------------------------------------------------------
# Turnover monotone in band (R2 exit criterion)
# ---------------------------------------------------------------------------


def test_online_band_turnover_monotone_in_band() -> None:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    walk = rng.normal(0.0, 0.02, size=(120, 3)).cumsum(axis=0)
    targets = pd.DataFrame(np.clip(walk, -0.3, 0.3), index=idx, columns=["A", "B", "C"])
    cfg = StatArbWalkForwardConfig()
    turnovers = []
    for band in [0.0, 0.002, 0.01, 0.05]:
        out = _online_banded_targets(targets, band, cfg)
        turnovers.append(float((out - out.shift(1).fillna(0.0)).abs().to_numpy().sum()))
    assert turnovers[0] > 0
    assert all(a >= b for a, b in zip(turnovers, turnovers[1:]))


# ---------------------------------------------------------------------------
# Participation gate wiring
# ---------------------------------------------------------------------------


def test_online_gate_caps_daily_participation() -> None:
    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    targets = pd.DataFrame(rng.uniform(-0.3, 0.3, size=(40, 3)), index=idx, columns=["A", "B", "C"])
    dvol = pd.DataFrame(1e6, index=idx, columns=targets.columns)
    aum, cap = 1e6, 0.02
    cfg = StatArbWalkForwardConfig(max_participation=cap)
    out = _online_banded_targets(targets, 0.0, cfg, dollar_volume=dvol, aum=aum)
    prev = pd.Series(0.0, index=targets.columns)
    for day in out.index:
        realized = max_participation_of(prev, out.loc[day], dvol.loc[day], aum)
        assert realized <= cap + 1e-9
        prev = out.loc[day]
    # The gate genuinely bound somewhere (the raw targets want more than 0.02/day).
    assert (targets.diff().abs() > cap).to_numpy().any()


def test_online_gate_requires_dollar_volume() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    targets = pd.DataFrame(0.1, index=idx, columns=["A"])
    cfg = StatArbWalkForwardConfig(max_participation=0.05)
    with pytest.raises(ValueError, match="dollar_volume"):
        _online_banded_targets(targets, 0.0, cfg)


def test_wfo_participation_gate_caps_fills_end_to_end() -> None:
    closes, volumes = _synthetic_panel()
    aum, cap = 2e8, 0.01  # allowed daily weight delta ~ cap * ADV / aum ~ 0.005 < position_unit
    base = _run_wfo(closes, volumes, initial_capital=aum)
    gated = _run_wfo(closes, volumes, initial_capital=aum, max_participation=cap)
    # Trailing ADV recomputed exactly as the WFO does (screening convention).
    window = _small_config().dollar_volume_window
    adv = (closes * volumes).rolling(window, min_periods=window).median()

    def worst_participation(result) -> float:
        tw = result.portfolio.target_weights
        # _force_fold_flat zeroes the last two rows of each fold AFTER the
        # gate: fold flatten is a hard risk rule and legitimately bypasses the
        # participation cap, so those liquidation rows are excluded.
        forced: set = set()
        for fold in result.folds:
            forced.update(tw.loc[fold.test_start : fold.test_end].index[-2:])
        prev = tw.shift(1).fillna(0.0)
        worst = 0.0
        for day in tw.index:
            if day in forced or (tw.loc[day] - prev.loc[day]).abs().sum() == 0.0:
                continue
            worst = max(worst, max_participation_of(prev.loc[day], tw.loc[day], adv.loc[day], aum))
        return worst

    assert worst_participation(base) > cap  # ungated book violates the cap somewhere
    assert worst_participation(gated) <= cap + 1e-9


# ---------------------------------------------------------------------------
# WFO wiring: closed_form band, bucket spread, frozen-v1 parity
# ---------------------------------------------------------------------------


def test_wfo_defaults_equal_explicit_off() -> None:
    closes, volumes = _synthetic_panel()
    default = _run_wfo(closes, volumes)
    explicit = _run_wfo(
        closes, volumes, band_mode="fixed", no_trade_band=0.0, spread_mode="flat", max_participation=0.0
    )
    assert default.summary == explicit.summary


def test_wfo_closed_form_band_runs_and_cuts_turnover() -> None:
    closes, volumes = _synthetic_panel()
    execution = ExecutionConfig(spread_bps=2.0, commission_bps=1.0, slippage_coeff=0, borrow_rate_bps_annual=0)
    base = _run_wfo(closes, volumes, execution=execution)
    closed = _run_wfo(closes, volumes, execution=execution, band_mode="closed_form")
    assert math.isfinite(closed.summary["oos_periodic_sharpe"])
    assert closed.summary["avg_turnover"] <= base.summary["avg_turnover"]
    # Bucket spreads feed the band's c_i without breaking the path.
    combo = _run_wfo(closes, volumes, execution=execution, band_mode="closed_form", spread_mode="bucket")
    assert math.isfinite(combo.summary["oos_periodic_sharpe"])


def test_wfo_bucket_spread_charges_bucket_rate() -> None:
    closes, volumes = _synthetic_panel()
    volumes = volumes * 0.0 + 4e5  # dollar volume ~ 4e7 -> the 5.0 bps bucket
    execution = ExecutionConfig(spread_bps=1.0, commission_bps=0.0, slippage_coeff=0, borrow_rate_bps_annual=0)
    flat = _run_wfo(closes, volumes, execution=execution)
    bucket = _run_wfo(closes, volumes, execution=execution, spread_mode="bucket")
    # Same targets (no band, no gate): only the spread pricing moved.
    pd.testing.assert_frame_equal(flat.portfolio.target_weights, bucket.portfolio.target_weights)
    assert flat.portfolio.costs["commission_spread"].sum() > 0
    assert bucket.portfolio.costs["commission_spread"].sum() == pytest.approx(
        5.0 * flat.portfolio.costs["commission_spread"].sum(), rel=1e-9
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_walk_config_validates_r2_knobs() -> None:
    assert StatArbWalkForwardConfig(band_mode="closed_form").band_mode == "closed_form"
    with pytest.raises(ValueError, match="spread_mode"):
        StatArbWalkForwardConfig(spread_mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_participation"):
        StatArbWalkForwardConfig(max_participation=-0.01)
    with pytest.raises(ValueError, match="band_mode"):
        StatArbWalkForwardConfig(band_mode="cube")  # type: ignore[arg-type]
