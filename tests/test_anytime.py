"""Tests for the anytime-valid live monitor (prism.validation.anytime).

The load-bearing test is `test_coverage_is_time_uniform`: it empirically checks
that the confidence sequence covers the true mean uniformly over stopping times
at the stated rate. If the mixture boundary constant were wrong (too narrow),
miscoverage would exceed alpha and that test would fail — it is the specification
the closed form is validated against, not the derivation alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism.validation.anytime import (
    DEFAULT_RETURN_BOUND,
    anytime_confidence_sequence,
    anytime_monitor_read,
)


def _half_width(returns, **kw) -> float:
    lo, hi = anytime_confidence_sequence(returns, **kw)
    return (hi - lo) / 2.0


def test_coverage_is_time_uniform():
    """P(true mean ever leaves the sequence up to T) <= alpha, over many streams."""
    rng = np.random.default_rng(20260706)
    alpha = 0.05
    true_mu = 0.001
    sigma = 0.01
    bound = 0.5  # loose enough that no draw is clipped
    n_sims = 1200
    horizon = 250
    check_times = range(20, horizon + 1, 20)

    ever_excluded = 0
    for _ in range(n_sims):
        stream = rng.normal(true_mu, sigma, size=horizon)
        excluded = False
        for t in check_times:
            lo, hi = anytime_confidence_sequence(
                stream[:t], alpha=alpha, bound=bound, opt_horizon=horizon
            )
            if not (lo <= true_mu <= hi):
                excluded = True
                break
        ever_excluded += int(excluded)

    miscoverage = ever_excluded / n_sims
    # Time-uniform guarantee: miscoverage <= alpha. The mixture is conservative,
    # so empirically this sits well under alpha; the assertion has wide margin.
    assert miscoverage <= alpha, f"time-uniform miscoverage {miscoverage:.3f} exceeds alpha {alpha}"


def test_interval_shrinks_with_n():
    rng = np.random.default_rng(1)
    stream = rng.normal(0.001, 0.01, size=1000)
    hw_small = _half_width(stream[:100], opt_horizon=1000)
    hw_large = _half_width(stream[:1000], opt_horizon=1000)
    assert np.isfinite(hw_small) and np.isfinite(hw_large)
    assert hw_large < hw_small


def test_confirmed_on_strong_positive():
    rng = np.random.default_rng(7)
    # Tight bound + strong positive mean so the sequence clears zero at moderate n.
    stream = rng.normal(0.02, 0.005, size=400)
    read = anytime_monitor_read(stream, bound=0.05, opt_horizon=252, hurdle=0.0)
    assert read["verdict"] == "confirmed"
    assert read["ci_lower"] > 0.0
    assert read["edge_confirmed"] is True and read["edge_refuted"] is False


def test_refuted_on_strong_negative():
    rng = np.random.default_rng(8)
    stream = rng.normal(-0.02, 0.005, size=400)
    read = anytime_monitor_read(stream, bound=0.05, opt_horizon=252, hurdle=0.0)
    assert read["verdict"] == "refuted"
    assert read["ci_upper"] < 0.0
    assert read["edge_refuted"] is True and read["edge_confirmed"] is False


def test_zero_mean_stays_inconclusive():
    rng = np.random.default_rng(9)
    stream = rng.normal(0.0, 0.01, size=200)
    read = anytime_monitor_read(stream, bound=DEFAULT_RETURN_BOUND, hurdle=0.0)
    assert read["verdict"] == "inconclusive"
    assert read["ci_lower"] < 0.0 < read["ci_upper"]


def test_hurdle_shifts_verdict():
    rng = np.random.default_rng(7)
    stream = rng.normal(0.02, 0.005, size=400)
    below = anytime_monitor_read(stream, bound=0.05, hurdle=0.0)
    above = anytime_monitor_read(stream, bound=0.05, hurdle=0.015)
    assert below["verdict"] == "confirmed"
    # Raising the hurdle between the sequence's lower bound and its mean makes the
    # same evidence inconclusive — the read is against the hurdle, not zero.
    assert below["ci_lower"] < 0.015 < below["ci_upper"]
    assert above["verdict"] == "inconclusive"


def test_degenerate_inputs():
    lo, hi = anytime_confidence_sequence(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)

    lo, hi = anytime_confidence_sequence(np.array([np.nan, np.inf, -np.inf]))
    assert np.isnan(lo) and np.isnan(hi)

    read = anytime_monitor_read(np.array([]))
    assert read["n"] == 0 and read["verdict"] == "inconclusive"

    # n == 1: finite but very wide interval, inconclusive.
    lo, hi = anytime_confidence_sequence(np.array([0.01]))
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi


def test_bad_params_raise():
    good = np.array([0.001, 0.002, -0.001])
    for bad_alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            anytime_confidence_sequence(good, alpha=bad_alpha)
    for bad_bound in (0.0, -1.0):
        with pytest.raises(ValueError):
            anytime_confidence_sequence(good, bound=bad_bound)
    with pytest.raises(ValueError):
        anytime_confidence_sequence(good, opt_horizon=0)


def test_winsorisation_bounds_and_warns(caplog):
    import logging

    stream = np.array([0.01, -0.01, 0.9, 0.005, -0.005])  # 0.9 exceeds bound
    with caplog.at_level(logging.WARNING):
        lo, hi = anytime_confidence_sequence(stream, bound=0.1)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert any("winsoris" in rec.message.lower() for rec in caplog.records)
