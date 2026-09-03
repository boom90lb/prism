"""Unit tests for the carry-flatten counterfactual arithmetic (research tier)."""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.research

from research.scripts.carry_flatten_diagnostic import (  # noqa: E402
    assert_fold_flattened,
    boundary_decomposition,
    bucket_spreads,
    carry_targets,
    fold_bounds,
    run_metrics,
)


def two_fold_weights() -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Two 4-row folds; each fold holds a book for two rows then flattens for two."""
    idx = pd.date_range("2024-03-04", periods=8, freq="B")
    data = {
        "AAA": [0.5, 0.5, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0],
        "BBB": [-0.5, -0.5, 0.0, 0.0, -0.7, -0.7, 0.0, 0.0],
    }
    return pd.DataFrame(data, index=idx), [(0, 3), (4, 7)]


def test_fold_bounds_align_and_reject_mismatch() -> None:
    weights, _ = two_fold_weights()
    folds = [
        {"fold": 0, "test_rows": 4, "test_start": weights.index[0].isoformat(), "test_end": weights.index[3].isoformat()},
        {"fold": 1, "test_rows": 4, "test_start": weights.index[4].isoformat(), "test_end": weights.index[7].isoformat()},
    ]
    assert fold_bounds(folds, weights.index) == [(0, 3), (4, 7)]

    shifted = [dict(folds[0]), dict(folds[1])]
    shifted[1]["test_start"] = weights.index[5].isoformat()
    with pytest.raises(AssertionError, match="test_start mismatch"):
        fold_bounds(shifted, weights.index)


def test_assert_fold_flattened_detects_unflattened_fold() -> None:
    weights, bounds = two_fold_weights()
    assert_fold_flattened(weights.to_numpy(dtype=float), bounds)

    unflattened = weights.copy()
    unflattened.iloc[3, 0] = 0.1
    with pytest.raises(AssertionError, match="not flattened"):
        assert_fold_flattened(unflattened.to_numpy(dtype=float), bounds)


def test_boundary_decomposition_arithmetic() -> None:
    weights, bounds = two_fold_weights()
    rows, aggregate = boundary_decomposition(weights.to_numpy(dtype=float), bounds)
    assert len(rows) == 1 and aggregate["n_interior_boundaries"] == 1
    # Book at fold-0 end: (0.5, -0.5); fold-1 opening book: (0.3, -0.7).
    assert rows[0]["gross_B"] == pytest.approx(1.0)
    assert rows[0]["gross_Bnext"] == pytest.approx(1.0)
    assert rows[0]["turnover_flatten"] == pytest.approx(2.0)
    assert rows[0]["turnover_carry"] == pytest.approx(abs(0.3 - 0.5) + abs(-0.7 + 0.5))
    assert aggregate["carry_over_flatten_ratio"] == pytest.approx(0.4 / 2.0)
    assert aggregate["total_turnover_saved"] == pytest.approx(1.6)


def test_carry_targets_replaces_flatten_rows() -> None:
    weights, bounds = two_fold_weights()

    interior = carry_targets(weights, bounds, carry_terminal=False)
    # Interior boundary rows 2-3 carry fold 0's last held book (row 1).
    for row in (2, 3):
        assert interior.iloc[row].tolist() == weights.iloc[1].tolist()
    # Terminal flatten rows 6-7 are kept.
    assert (interior.iloc[6] == 0.0).all() and (interior.iloc[7] == 0.0).all()
    # Non-boundary rows untouched.
    assert interior.iloc[0].equals(weights.iloc[0]) and interior.iloc[4].equals(weights.iloc[4])

    everything = carry_targets(weights, bounds, carry_terminal=True)
    for row in (6, 7):
        assert everything.iloc[row].tolist() == weights.iloc[5].tolist()


def test_run_metrics_on_constant_stream() -> None:
    idx = pd.date_range("2024-03-04", periods=504, freq="B")
    returns = pd.Series(0.001, index=idx)
    costs = pd.DataFrame(
        {"gross": 1.0, "turnover": 0.05, "total": 0.0001, "borrow": 0.00002}, index=idx
    )
    metrics = run_metrics(returns, costs)
    assert metrics["years"] == pytest.approx(2.0)
    assert metrics["total_return"] == pytest.approx(1.001**504 - 1.0)
    assert metrics["annualized_vol"] == pytest.approx(0.0)
    assert metrics["avg_gross"] == pytest.approx(1.0)
    assert metrics["avg_turnover"] == pytest.approx(0.05)
    assert metrics["total_cost"] == pytest.approx(0.0001 * 504)
    assert metrics["total_cost_bps_per_year"] == pytest.approx(0.0001 * 504 / 2.0 * 1e4)
    assert metrics["worst_day"] == pytest.approx(0.001)


def test_bucket_spreads_selects_by_floor() -> None:
    buckets = [[500e6, 1.0], [100e6, 2.0], [25e6, 5.0], [0.0, 10.0]]
    mdv = pd.Series({"MEGA": 600e6, "LARGE": 150e6, "MID": 30e6, "SMALL": 1e6, "GONE": np.nan})
    spreads = bucket_spreads(mdv, buckets)
    assert spreads["MEGA"] == 1.0
    assert spreads["LARGE"] == 2.0
    assert spreads["MID"] == 5.0
    assert spreads["SMALL"] == 10.0
    # NaN median dollar volume falls to the widest bucket, never to zero spread.
    assert spreads["GONE"] == 10.0
