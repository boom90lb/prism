"""R2 mechanics that survived the stage 3b driver retirement: the closed-form
band, the per-name spread in the backtest engine (I-9), and the pre-registered
bucket schedule.

All three are ``prism`` properties -- ``portfolio/construct.py::closed_form_band``,
``execution/target_weights.py::backtest_target_weights`` and
``execution/cost_model.py::bucket_spread_bps`` -- and are asserted against
``docs/r2_design.md``'s pre-registered constants rather than re-derived. The
eight tests that drove ``run_residual_stat_arb_walk_forward`` and
``_online_banded_targets`` were deleted with
``research/arbitrage/residual_walk_forward.py`` at stage 3b; the fold-level
behaviour they covered is now asserted against a recorded record in
``tests/test_trend_t0_reproduction.py`` and ``tests/test_demotion_b1_reproduction.py``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from prism.config import ExecutionConfig
from prism.execution.cost_model import SPREAD_BUCKET_SCHEDULE_V1, bucket_spread_bps
from prism.execution.target_weights import backtest_target_weights
from prism.portfolio.construct import closed_form_band

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
    # docs/r2_design.md §1: never fitted, never swept. The constant used to live
    # in the retired research driver; prism's own declared default is now the
    # pre-registration, and examples/trend/trend_t0.toml declares the same 1.0.
    assert inspect.signature(closed_form_band).parameters["gamma_risk"].default == 1.0


# ---------------------------------------------------------------------------
# Per-name spread in the backtest engine (I-9)
# ---------------------------------------------------------------------------


def _spread_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=8, freq="B")
    open_prices = pd.DataFrame({"A": 100.0 + np.arange(8), "B": 50.0 - 0.5 * np.arange(8)}, index=idx)
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
        open_prices,
        weights,
        execution=execution,
        spread_bps_per_name=pd.Series(1.0, index=["A", "B"]),
    )
    wide = backtest_target_weights(
        open_prices,
        weights,
        execution=execution,
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
    # Schedule constant is the pre-registered one (r2_design.md §3).
    assert SPREAD_BUCKET_SCHEDULE_V1 == ((500e6, 1.0), (100e6, 2.0), (25e6, 5.0), (0.0, 10.0))
