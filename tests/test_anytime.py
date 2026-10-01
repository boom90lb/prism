"""Tests for the anytime-valid live monitor (prism.validation.anytime).

The load-bearing test is `test_coverage_is_time_uniform`: it empirically checks
that the confidence sequence covers the true mean uniformly over stopping times
at the stated rate. If the mixture boundary constant were wrong (too narrow),
miscoverage would exceed alpha and that test would fail — it is the specification
the closed form is validated against, not the derivation alone.
"""

from __future__ import annotations

import math

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


# ---------------------------------------------------------------------------
# Empirical-Bernstein (variance-adaptive) confidence sequence
# ---------------------------------------------------------------------------

from prism.validation.anytime import (  # noqa: E402
    empirical_bernstein_confidence_sequence,
    empirical_bernstein_monitor_read,
    empirical_bernstein_running_intervals,
)


def test_eb_coverage_is_time_uniform():
    """P(true mean ever leaves the EB sequence over 400 steps) <= alpha, over 400+ paths.

    Vectorised over paths so the whole check runs in well under a second. The
    draws are bounded (truncated normal inside the support) so no winsorisation
    moves the estimand; the guarantee is nonasymptotic, so a miscoverage above
    alpha plus Monte Carlo slack means the construction is wrong. This is a
    one-sided validity check that an over-wide sequence passes trivially
    (miscoverage here is ~0 even with alpha mis-wired to 0.5); the width test
    below carries the power.
    """
    rng = np.random.default_rng(20260930)
    alpha = 0.05
    n_paths, horizon = 600, 400
    true_mu, sigma, bound = 0.0005, 0.01, 0.1
    paths = rng.normal(true_mu, sigma, size=(n_paths, horizon))
    assert np.all(np.abs(paths) < bound)  # nothing clipped: the mean is the true mean
    lower, upper = empirical_bernstein_running_intervals(paths, alpha=alpha, bound=bound)
    ever_excluded = np.any((lower > true_mu) | (upper < true_mu), axis=1)
    miscoverage = float(ever_excluded.mean())
    slack = 3.0 * np.sqrt(alpha * (1 - alpha) / n_paths)  # three Monte Carlo standard errors
    assert miscoverage <= alpha + slack, f"time-uniform miscoverage {miscoverage:.3f} exceeds {alpha} + {slack:.3f}"


def test_eb_width_tracks_alpha_and_pins_to_the_fixed_n_radius():
    """Power companion to the coverage test. The half-width must widen as alpha falls, and at
    the coverage test's operating point (sigma 1%, support 0.1, n 400) the mean final half-width
    sits at a documented multiple of the fixed-n sub-Gaussian radius sigma*sqrt(2 log(2/alpha)/n).
    Measured 2.13x (stable to 3 d.p. across seeds). An alpha mis-wired to 0.5 gives 1.00x, to 0.2
    gives 1.45x, a dropped log(2/alpha) term 0.32x, lambda_max mis-set to 0.05 gives 27x. The band
    is a regression pin at this operating point (the ratio moves with sigma/bound), not a theorem."""
    rng = np.random.default_rng(20260930)
    n_paths, horizon, sigma, bound = 600, 400, 0.01, 0.1
    paths = rng.normal(0.0005, sigma, size=(n_paths, horizon))
    lo5, up5 = empirical_bernstein_running_intervals(paths, alpha=0.05, bound=bound)
    lo50, up50 = empirical_bernstein_running_intervals(paths, alpha=0.5, bound=bound)
    hw5, hw50 = (up5 - lo5) / 2.0, (up50 - lo50) / 2.0
    # The first steps are clipped to the full support at both alphas, so weak everywhere ...
    assert np.all(hw5 >= hw50)
    # ... and strict wherever the alpha=0.05 interval is narrower than the support.
    inside = hw5 < bound
    assert inside.any() and np.all(hw5[inside] > hw50[inside])
    fixed_n_radius = sigma * math.sqrt(2.0 * math.log(2.0 / 0.05) / horizon)
    ratio = float(hw5[:, -1].mean() / fixed_n_radius)
    assert 1.5 <= ratio <= 3.0, f"final half-width is {ratio:.2f}x the fixed-n radius; expected ~2.1x"


def test_eb_cs_center_is_the_bet_weighted_centre_even_when_the_interval_is_clipped():
    """At n = 5 and support 0.1 the width floor exceeds the support, so both endpoints clip and the
    interval is [-0.1, 0.1]; the midpoint would be 0 whatever the data. Constant observations make the
    bet-weighted centre equal the mean regardless of the lambdas, so cs_center must read 0.003."""
    read = empirical_bernstein_monitor_read(np.full(5, 0.003), bound=0.1)
    assert read["ci_lower"] == -0.1 and read["ci_upper"] == 0.1
    assert read["cs_center"] == pytest.approx(0.003)
    lo, hi, centre = empirical_bernstein_running_intervals(np.full((1, 5), 0.003), bound=0.1, return_center=True)
    assert lo.shape == hi.shape == centre.shape == (1, 5)
    np.testing.assert_allclose(centre, 0.003)


def test_eb_is_at_least_four_times_narrower_at_the_live_books_scale():
    """At n = 252, daily SD 0.7% and support 0.1, the variance-adaptive sequence beats the
    support-bound normal mixture by at least 4x: that is the whole reason for the switch."""
    rng = np.random.default_rng(3)
    stream = rng.normal(0.0, 0.007, size=252)
    lo_eb, hi_eb = empirical_bernstein_confidence_sequence(stream, bound=0.1)
    lo_nm, hi_nm = anytime_confidence_sequence(stream, bound=0.1, opt_horizon=252)
    assert np.isfinite([lo_eb, hi_eb, lo_nm, hi_nm]).all()
    assert (hi_nm - lo_nm) >= 4.0 * (hi_eb - lo_eb)


def test_eb_degenerate_inputs_match_the_normal_mixture():
    lo, hi = empirical_bernstein_confidence_sequence(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)
    lo, hi = empirical_bernstein_confidence_sequence(np.array([np.nan, np.inf, -np.inf]))
    assert np.isnan(lo) and np.isnan(hi)
    # n == 1: finite, the whole support, inconclusive.
    lo, hi = empirical_bernstein_confidence_sequence(np.array([0.01]), bound=0.1)
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
    read = empirical_bernstein_monitor_read(np.array([]))
    assert read["n"] == 0 and read["verdict"] == "inconclusive" and read["method"] == "empirical_bernstein"
    for bad_alpha in (0.0, 1.0):
        with pytest.raises(ValueError):
            empirical_bernstein_confidence_sequence(np.array([0.01, 0.02]), alpha=bad_alpha)
    with pytest.raises(ValueError):
        empirical_bernstein_confidence_sequence(np.array([0.01, 0.02]), bound=0.0)
    with pytest.raises(ValueError):
        empirical_bernstein_confidence_sequence(np.array([0.01, 0.02]), lambda_max=1.0)
    with pytest.raises(ValueError):
        empirical_bernstein_running_intervals(np.array([[0.01, np.nan]]))


def test_eb_verdicts_and_winsorisation_count():
    rng = np.random.default_rng(11)
    up = empirical_bernstein_monitor_read(rng.normal(0.02, 0.005, size=400), bound=0.05, hurdle=0.0)
    down = empirical_bernstein_monitor_read(rng.normal(-0.02, 0.005, size=400), bound=0.05, hurdle=0.0)
    flat = empirical_bernstein_monitor_read(rng.normal(0.0, 0.01, size=200), bound=0.1, hurdle=0.0)
    assert up["verdict"] == "confirmed" and up["ci_lower"] > 0.0
    assert down["verdict"] == "refuted" and down["ci_upper"] < 0.0
    assert flat["verdict"] == "inconclusive" and flat["ci_lower"] < 0.0 < flat["ci_upper"]
    clipped = empirical_bernstein_monitor_read(np.array([0.01, -0.01, 0.9, 0.005]), bound=0.1)
    assert clipped["n_winsorised"] == 1
