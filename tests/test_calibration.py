"""Tests for the shared-slope calibration (engine specification §3.4, decision D6).

The failures these are written to catch, in the order of what they would cost:

1. **A units error.** ``theta`` is dimensionless; expected return is recovered only by
   multiplying the standardized score back by the name's horizon volatility. An objective
   built on ``z`` alone is wrong by a factor of order ``sigma_h`` and silently reweights
   the whole cross-section by inverse volatility. Nothing about the run would look broken.
2. **A fold excluded for one arm and kept for another.** That breaks the same-sample
   comparison the whole experiment rests on (D6), and the resulting difference between
   arms would be a difference in samples wearing a difference in method's clothes.
3. **A silently inverted or silently zeroed sign.** A wrong-sign slope must disable new
   directional exposure, be recorded, and let the evaluation continue -- never be flipped
   into an edge, and never vanish into a zero nobody recorded.
4. **A label nobody could have earned.** The regressand is an open-to-open return with
   distributions credited, not a close-to-close return, and not one that starts at the
   decision bar's own close.
5. **An interval that is narrower than the data supports.** The block bootstrap exists to
   carry the dependence overlapping labels and persistent scores induce; ``test_coverage``
   measures what it achieves against nominal rather than asserting that it plumbs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from prism.calibration import (
    CalibrationError,
    CalibrationSpec,
    SigmaHSpec,
    apply_calibration,
    ewma_bar_sigma,
    executable_returns,
    fit_calibration,
    fold_exclusion,
    horizon_sigma,
    load_calibration,
    objective_endpoints,
    require_same_exclusion_geometry,
    require_sigma_agreement,
    standardize_cross_section,
    standardize_panel,
    validate_standardize,
)
from prism.validation.trials import stable_json_dumps
from prism.validation.uncertainty import BlockBootstrapSpec, block_bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
POINT_FILE = REPO_ROOT / "calibrations" / "shared_slope_point.toml"
P80_FILE = REPO_ROOT / "calibrations" / "shared_slope_p80.toml"

#: The reference ``[sigma_h]`` block of §3.4.
SIGMA = SigmaHSpec(estimator="ewma_daily", window_bars=63, scaling="per_bar_times_h")


def make_spec(**overrides: object) -> CalibrationSpec:
    """The reference calibration of §3.4, with named departures."""
    settings: dict[str, object] = {
        "kind": "shared_slope",
        "standardize": "rank_normal",
        "weighting": "per_date_equal",
        "shrink_pseudo_periods": 12.0,
        "expected_sign": "positive",
        "uncertainty": "time_block_bootstrap",
        "block_scheme": "circular_moving",
        "block_periods": 3,
        "n_resamples": 200,
        "seed": 20260906,
        "min_training_periods": 12,
        "min_blocks": 4,
        "inner_oos_required": False,
        "executable_return": "open_to_open",
        "label_distributions": "credited",
        "sigma_h": SIGMA,
        "confidence": None,
    }
    settings.update(overrides)
    return CalibrationSpec(**settings)  # type: ignore[arg-type]


def make_panels(
    n_periods: int,
    n_names: int,
    theta: float,
    *,
    seed: int = 7,
    sigma_level: float = 0.08,
    noise: float = 1.0,
    rho: float = 0.0,
    overlap: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Panels whose *standardized* score carries a known slope ``theta``.

    ``y = theta * z + noise * e`` and ``returns = y * sigma_h``, so the estimand
    ``E[r] / sigma_h = theta * z`` holds by construction. ``rho`` makes the raw scores
    persistent across decision periods and ``overlap`` gives the errors an MA structure:
    together they are what overlapping labels and a slow-moving ranking do to the
    estimating equation, and they are the case the block bootstrap exists for.
    """
    rng = np.random.default_rng(seed)
    raw = np.empty((n_periods, n_names), dtype=float)
    raw[0] = rng.normal(size=n_names)
    for period in range(1, n_periods):
        raw[period] = rho * raw[period - 1] + math.sqrt(1.0 - rho * rho) * rng.normal(size=n_names)
    index = pd.RangeIndex(n_periods, name="period")
    columns = pd.Index([f"S{i:03d}" for i in range(n_names)], name="sid")
    scores = pd.DataFrame(raw, index=index, columns=columns)
    z = standardize_panel(scores, "rank_normal").to_numpy(dtype=float)
    shocks = rng.normal(size=(n_periods + overlap, n_names))
    errors = sum(shocks[k : k + n_periods] for k in range(overlap + 1)) / math.sqrt(overlap + 1)
    sigma = np.full((n_periods, n_names), float(sigma_level), dtype=float)
    returns = (theta * z + noise * errors) * sigma
    return (
        scores,
        pd.DataFrame(returns, index=index, columns=columns),
        pd.DataFrame(sigma, index=index, columns=columns),
    )


# ---------------------------------------------------------------------------------
# 1. The shipped files, and what makes the three arms comparable (D6)
# ---------------------------------------------------------------------------------


def test_shipped_files_parse_and_declare_the_two_arms() -> None:
    point = load_calibration(POINT_FILE)
    p80 = load_calibration(P80_FILE)
    assert point.confidence is None, "the point arm declares no confidence; absence IS the declaration"
    assert p80.confidence == pytest.approx(0.80), "the brief fixes the uncertainty arm at 0.80"
    assert point.kind == p80.kind == "shared_slope"
    assert point.standardize == "rank_normal"
    assert point.weighting == "per_date_equal"
    assert point.shrink_pseudo_periods == pytest.approx(12.0)
    assert (point.min_training_periods, point.min_blocks) == (12, 4)
    assert point.sigma_h.to_dict() == {"estimator": "ewma_daily", "window_bars": 63, "scaling": "per_bar_times_h"}


def test_the_two_shipped_files_differ_in_confidence_and_nothing_else() -> None:
    """D6's same-sample requirement, as an executable check on the shipped files."""
    point = load_calibration(POINT_FILE)
    p80 = load_calibration(P80_FILE)
    assert point.exclusion_geometry() == p80.exclusion_geometry()
    shared = require_same_exclusion_geometry({"convex_point": point, "convex_p80_one_sided": p80})
    assert "confidence" not in shared


def test_an_arm_that_moves_an_exclusion_key_is_refused() -> None:
    point = load_calibration(POINT_FILE)
    tampered = make_spec(min_training_periods=6)
    with pytest.raises(CalibrationError, match="min_training_periods"):
        require_same_exclusion_geometry({"convex_point": point, "tampered": tampered})


def test_reference_geometry_shrink_factor_is_declared_not_accidental() -> None:
    """§3.4: 13 labelled periods and 12 pseudo-periods give 0.52, recorded per fold."""
    assert make_spec().shrink_factor(13) == pytest.approx(13.0 / 25.0)
    assert fold_exclusion(13, make_spec()).blocks_drawn == 5  # ceil(13 / 3)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("kind", "ridge", "kind"),
        ("standardize", "demean", "standardize"),
        ("weighting", "cap_weighted", "weighting"),
        ("confidence", 0.5, "confidence"),
        ("confidence", 1.0, "confidence"),
        ("expected_sign", "either", "expected_sign"),
        ("block_periods", 0, "block_periods"),
        ("min_blocks", 0, "min_blocks"),
        ("executable_return", "close_to_close", "executable_return"),
    ],
)
def test_a_declaration_outside_the_schema_is_refused(key: str, value: object, message: str) -> None:
    with pytest.raises(CalibrationError, match=message):
        make_spec(**{key: value})


def test_unknown_and_missing_keys_are_refused(tmp_path: Path) -> None:
    table = {
        "kind": "shared_slope",
        "standardize": "rank_normal",
        "weighting": "per_date_equal",
        "shrink_pseudo_periods": 12.0,
        "expected_sign": "positive",
        "uncertainty": "time_block_bootstrap",
        "block_scheme": "circular_moving",
        "block_periods": 3,
        "n_resamples": 200,
        "seed": 1,
        "min_training_periods": 12,
        "min_blocks": 4,
        "inner_oos_required": False,
        "executable_return": "open_to_open",
        "label_distributions": "credited",
        "sigma_h": {"estimator": "ewma_daily", "window_bars": 63, "scaling": "per_bar_times_h"},
    }
    with pytest.raises(CalibrationError, match="unknown keys"):
        CalibrationSpec.from_mapping({**table, "tuned_after_the_fact": 3})
    with pytest.raises(CalibrationError, match="missing required keys"):
        CalibrationSpec.from_mapping({key: value for key, value in table.items() if key != "min_blocks"})


def test_an_ewma_estimator_at_horizon_scaling_is_refused() -> None:
    """§3.4 defines ``ewma_daily`` per bar; the pair has no declared estimator."""
    with pytest.raises(CalibrationError, match="per bar"):
        SigmaHSpec(estimator="ewma_daily", window_bars=63, scaling="estimated_at_horizon")


def test_a_risk_snapshot_that_scales_differently_is_refused() -> None:
    spec = make_spec()
    require_sigma_agreement(spec, scaling="per_bar_times_h", horizon_bars=21)
    with pytest.raises(CalibrationError, match="commensurable"):
        require_sigma_agreement(spec, scaling="estimated_at_horizon")


# ---------------------------------------------------------------------------------
# 2. Standardization (§3.4)
# ---------------------------------------------------------------------------------


def test_rank_normal_is_the_declared_formula_with_average_ties() -> None:
    scores = pd.Series([3.0, 1.0, 2.0, 2.0, np.nan, np.inf], index=list("abcdef"))
    z = standardize_cross_section(scores, "rank_normal")
    # Four finite scores; 2.0 is tied at ranks 2 and 3, so both take 2.5.
    assert z["b"] == pytest.approx(ndtri(0.5 / 4))
    assert z["c"] == pytest.approx(ndtri(2.0 / 4))
    assert z["d"] == pytest.approx(ndtri(2.0 / 4))
    assert z["a"] == pytest.approx(ndtri(3.5 / 4))
    assert math.isnan(z["e"]) and math.isnan(z["f"]), "a non-finite score is not imputed"


def test_the_panel_path_equals_the_decision_time_path_cell_for_cell() -> None:
    """The fit standardizes with the vectorized path; a decision standardizes one row."""
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(rng.normal(size=(6, 9)), columns=[f"S{i}" for i in range(9)])
    panel.iloc[0, 4] = np.nan
    panel.iloc[2, 7] = np.inf
    for rule in ("rank_normal", "zscore", "none"):
        vectorized = standardize_panel(panel, rule).to_numpy(dtype=float)
        row_by_row = np.vstack(
            [standardize_cross_section(panel.loc[label], rule).to_numpy(dtype=float) for label in panel.index]
        )
        assert np.allclose(vectorized, row_by_row, equal_nan=True), rule


def test_zscore_refuses_a_cross_section_with_no_dispersion() -> None:
    flat = pd.Series([2.0, 2.0, 2.0])
    with pytest.raises(CalibrationError, match="not positive"):
        standardize_cross_section(flat, "zscore")
    panel = pd.DataFrame([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    with pytest.raises(CalibrationError, match="not positive"):
        standardize_panel(panel, "zscore")


def test_standardization_admissibility_follows_output_kind() -> None:
    validate_standardize("rank_normal", "rank_score")
    validate_standardize("zscore", "rank_score")
    validate_standardize("none", "standardized_forecast")
    validate_standardize("zscore", "standardized_forecast")
    with pytest.raises(CalibrationError, match="admits standardize"):
        validate_standardize("none", "rank_score")
    with pytest.raises(CalibrationError, match="admits standardize"):
        validate_standardize("rank_normal", "standardized_forecast")


# ---------------------------------------------------------------------------------
# 3. The label: what a fill could actually have earned (§3.4)
# ---------------------------------------------------------------------------------


def _open_panel() -> pd.DataFrame:
    bars = pd.RangeIndex(10, name="bar")
    return pd.DataFrame(
        {
            "AAA": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            "BBB": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        },
        index=bars,
    )


def test_the_label_is_open_to_open_from_the_bar_after_the_decision() -> None:
    opens = _open_panel()
    zero = pd.DataFrame(0.0, index=opens.index, columns=opens.columns)
    labels = executable_returns(opens, [0, 3], horizon_bars=2, distributions=zero)
    # Decision at bar 0: enter at open[1] = 11, exit at open[0 + 2 + 1] = open[3] = 13.
    assert labels.loc[0, "AAA"] == pytest.approx(13.0 / 11.0 - 1.0)
    # Decision at bar 3: enter at open[4] = 14, exit at open[6] = 16.
    assert labels.loc[3, "AAA"] == pytest.approx(16.0 / 14.0 - 1.0)
    assert list(labels.index) == [0, 3]


def test_a_period_without_a_complete_forward_label_is_left_unlabelled() -> None:
    opens = _open_panel()
    zero = pd.DataFrame(0.0, index=opens.index, columns=opens.columns)
    labels = executable_returns(opens, [0, 7, 8], horizon_bars=2, distributions=zero)
    assert np.isfinite(labels.loc[0]).all()
    assert labels.loc[7].isna().all(), "the exit bar lies beyond the panel"
    assert labels.loc[8].isna().all()


def test_distributions_are_credited_on_ex_dates_inside_the_horizon() -> None:
    opens = _open_panel()
    cash = pd.DataFrame(0.0, index=opens.index, columns=opens.columns)
    cash.loc[1, "AAA"] = 0.50  # ex-date inside (0, 2]
    cash.loc[3, "AAA"] = 0.25  # ex-date outside (0, 2]
    labels = executable_returns(opens, [0], horizon_bars=2, distributions=cash)
    assert labels.loc[0, "AAA"] == pytest.approx((13.0 + 0.50) / 11.0 - 1.0)
    assert labels.loc[0, "BBB"] == pytest.approx(103.0 / 101.0 - 1.0)


def test_a_credited_label_without_a_distributions_panel_is_refused() -> None:
    opens = _open_panel()
    with pytest.raises(CalibrationError, match="uncredited"):
        executable_returns(opens, [0], horizon_bars=2)
    with pytest.raises(CalibrationError, match="dropped silently"):
        executable_returns(
            opens,
            [0],
            horizon_bars=2,
            distributions=pd.DataFrame(0.0, index=opens.index, columns=opens.columns),
            label_distributions="ignored",
        )


def test_the_label_is_not_a_close_to_close_return() -> None:
    """A close-to-close reading of the same bars is a different number; §3.4 fixes which."""
    opens = _open_panel()
    zero = pd.DataFrame(0.0, index=opens.index, columns=opens.columns)
    executable = executable_returns(opens, [0], horizon_bars=2, distributions=zero).loc[0, "AAA"]
    closes = opens + 0.5  # any close panel that is not the open panel
    close_to_close = closes.iloc[0 + 2]["AAA"] / closes.iloc[0]["AAA"] - 1.0
    assert executable != pytest.approx(close_to_close)


# ---------------------------------------------------------------------------------
# 4. Horizon volatility (§3.4, "[sigma_h]")
# ---------------------------------------------------------------------------------


def test_horizon_sigma_is_the_per_bar_ewma_scaled_by_sqrt_h() -> None:
    rng = np.random.default_rng(5)
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rng.normal(scale=0.01, size=(80, 3)), axis=0)))
    sigma_h = SigmaHSpec(estimator="ewma_daily", window_bars=10, scaling="per_bar_times_h")
    per_bar = ewma_bar_sigma(prices, 10)
    expected = prices.pct_change().ewm(span=10, min_periods=10, adjust=False).std()
    assert np.allclose(per_bar.to_numpy(), expected.to_numpy(), equal_nan=True)
    scaled = horizon_sigma(prices, sigma_h, 21)
    assert np.allclose(scaled.to_numpy(), per_bar.to_numpy() * math.sqrt(21), equal_nan=True)
    assert scaled.iloc[:9].isna().all().all(), "the warm-up is non-finite, not a volatility from three bars"


def test_horizon_sigma_selects_decision_bars_without_reindexing() -> None:
    rng = np.random.default_rng(6)
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rng.normal(scale=0.01, size=(40, 2)), axis=0)))
    sigma_h = SigmaHSpec(estimator="ewma_daily", window_bars=5, scaling="per_bar_times_h")
    full = horizon_sigma(prices, sigma_h, 21)
    picked = horizon_sigma(prices, sigma_h, 21, decision_positions=[10, 20, 30])
    assert list(picked.index) == [10, 20, 30]
    assert np.allclose(picked.to_numpy(), full.iloc[[10, 20, 30]].to_numpy())


# ---------------------------------------------------------------------------------
# 5. The slope, its weighting, and its shrinkage (§3.4)
# ---------------------------------------------------------------------------------


def test_the_estimator_recovers_a_known_slope() -> None:
    """Averaged over ten independent samples, because one sample cannot show unbiasedness.

    Run (seeds 11-20): mean theta_hat 0.06487 against a true 0.06, per-sample standard
    deviation 0.00685, so the mean's standard error is 0.00217 and the ten-sample mean sits
    2.2 standard errors high. Widening the same check to 60 samples (seeds 11-70, Run)
    gives mean 0.06013, sd 0.00822, t = 0.13 against the true value: the estimator is
    unbiased and the ten-sample mean is sampling noise. Ten samples are what the core suite
    pays for; the tolerance below is set from the measured standard error, not from the
    number that happened to come out.
    """
    theta = 0.06
    spec = make_spec(shrink_pseudo_periods=0.0)
    estimates = []
    for seed in range(11, 21):
        scores, returns, sigma = make_panels(120, 40, theta, seed=seed, noise=0.5)
        fit = fit_calibration(scores, returns, sigma, spec)
        assert fit.theta_hat is not None
        assert fit.theta_hat == pytest.approx(theta, abs=0.03)
        estimates.append(fit.theta_hat)
    assert float(np.mean(estimates)) == pytest.approx(theta, abs=0.006)

    scores, returns, sigma = make_panels(120, 40, theta, seed=11, noise=0.5)
    fit = fit_calibration(scores, returns, sigma, spec)
    assert fit.shrink_factor == pytest.approx(1.0)
    assert fit.theta_shrunk == pytest.approx(fit.theta_hat)
    assert fit.n_periods == 120
    assert fit.n_observations == 120 * 40
    assert fit.fit_cutoff == str(scores.index[-1]), "the record carries the training cutoff"
    assert fit.sign_ok and fit.disabled_reason is None


def test_shrinkage_is_exactly_the_declared_pseudo_period_rule() -> None:
    scores, returns, sigma = make_panels(20, 30, 0.05, seed=12)
    spec = make_spec()
    fit = fit_calibration(scores, returns, sigma, spec)
    assert fit.shrink_factor == pytest.approx(20.0 / 32.0)
    assert fit.theta_shrunk == pytest.approx(fit.theta_hat * 20.0 / 32.0)


def test_per_date_equal_weighting_is_computed_as_declared() -> None:
    """Hand-computed on an unbalanced panel, where the two weightings must differ."""
    index = pd.RangeIndex(2)
    columns = pd.Index(["A", "B", "C"])
    scores = pd.DataFrame([[1.0, 2.0, 3.0], [1.0, 2.0, np.nan]], index=index, columns=columns)
    sigma = pd.DataFrame(0.10, index=index, columns=columns)
    returns = pd.DataFrame([[0.01, 0.02, 0.03], [-0.01, 0.04, np.nan]], index=index, columns=columns)
    z = standardize_panel(scores, "rank_normal").to_numpy(dtype=float)
    y = (returns / sigma).to_numpy(dtype=float)
    per_date = np.array([1.0 / 3.0, 1.0 / 2.0])
    numerator = float(np.nansum(per_date[:, None] * z * y))
    denominator = float(np.nansum(per_date[:, None] * z * z))
    spec = make_spec(shrink_pseudo_periods=0.0, min_training_periods=1, min_blocks=1, block_periods=1)
    fit = fit_calibration(scores, returns, sigma, spec)
    assert fit.theta_hat == pytest.approx(numerator / denominator)

    pooled = fit_calibration(
        scores,
        returns,
        sigma,
        make_spec(
            shrink_pseudo_periods=0.0,
            min_training_periods=1,
            min_blocks=1,
            block_periods=1,
            weighting="per_observation_equal",
        ),
    )
    unweighted = float(np.nansum(z * y)) / float(np.nansum(z * z))
    assert pooled.theta_hat == pytest.approx(unweighted)
    assert pooled.theta_hat != pytest.approx(fit.theta_hat), "the two declared weightings must differ here"


def test_scores_are_standardized_before_unlabelled_names_are_dropped() -> None:
    """The order matters, and the decision-time path fixes it.

    At a decision bar there are no labels at all, so the cross-section is standardized over
    every eligible finite score. The fit must standardize the same vector and only then
    drop the names whose forward label is missing; standardizing over the labelled subset
    would give a different ``z`` for every remaining name and calibrate a mapping the
    deployment never applies.
    """
    index = pd.RangeIndex(14)
    columns = pd.Index(["A", "B", "C", "D"])
    raw = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (14, 1))
    scores = pd.DataFrame(raw, index=index, columns=columns)
    sigma = pd.DataFrame(0.10, index=index, columns=columns)
    returns = pd.DataFrame(np.tile(np.array([-0.02, -0.01, 0.01, np.nan]), (14, 1)), index=index, columns=columns)
    spec = make_spec(shrink_pseudo_periods=0.0, min_training_periods=1, min_blocks=1, block_periods=1)
    fit = fit_calibration(scores, returns, sigma, spec)

    z_all = ndtri((np.array([1.0, 2.0, 3.0]) - 0.5) / 4.0)  # standardized over all four names
    z_labelled = ndtri((np.array([1.0, 2.0, 3.0]) - 0.5) / 3.0)  # if D had been dropped first
    y = np.array([-0.02, -0.01, 0.01]) / 0.10
    assert fit.theta_hat == pytest.approx(float(z_all @ y) / float(z_all @ z_all))
    assert fit.theta_hat != pytest.approx(float(z_labelled @ y) / float(z_labelled @ z_labelled))


def test_names_without_a_complete_observation_are_dropped_and_counted() -> None:
    scores, returns, sigma = make_panels(15, 10, 0.05, seed=13)
    returns.iloc[0, 0] = np.nan
    sigma.iloc[1, 1] = 0.0
    sigma.iloc[2, 2] = np.nan
    scores.iloc[3, 3] = np.nan
    fit = fit_calibration(scores, returns, sigma, make_spec())
    sample = fit.method["sample"]
    assert sample["n_complete"] == 15 * 10 - 4
    assert sample["n_missing_return"] == 1
    assert sample["n_invalid_sigma"] == 2, "a zero or missing volatility is recorded, never silently kept"
    assert sample["n_missing_score"] == 1


# ---------------------------------------------------------------------------------
# 6. Units: the multiplication the constructor depends on (§3.4, "Estimand and units")
# ---------------------------------------------------------------------------------


def test_apply_calibration_multiplies_back_by_horizon_volatility() -> None:
    scores, returns, sigma = make_panels(30, 20, 0.05, seed=14)
    fit = fit_calibration(scores, returns, sigma, make_spec())
    z_t = standardize_cross_section(scores.iloc[-1], "rank_normal")
    sigma_t = sigma.iloc[-1]
    a_t, mu_t = apply_calibration(fit, z_t, sigma_t)
    assert np.allclose(a_t.to_numpy(), z_t.to_numpy() * sigma_t.to_numpy())
    assert np.allclose(mu_t.to_numpy(), float(fit.theta_shrunk) * a_t.to_numpy())
    assert list(a_t.index) == list(z_t.index)


def test_the_volatility_factor_is_not_of_order_one() -> None:
    """§3.4: 0.25 annual at a 21-bar horizon is 0.0722, so ``z`` alone is not ``a``."""
    sigma_bar = 0.25 / math.sqrt(252.0)
    sigma_h = sigma_bar * math.sqrt(21.0)
    assert sigma_bar == pytest.approx(0.015749, abs=5e-7)
    assert sigma_h == pytest.approx(0.0722, abs=5e-5)
    fit = fit_calibration(*make_panels(20, 12, 0.05, seed=15), make_spec())
    z_t = pd.Series([1.0, -1.0], index=["A", "B"])
    a_t, mu_t = apply_calibration(fit, z_t, pd.Series([sigma_h, sigma_h], index=["A", "B"]))
    assert a_t["A"] == pytest.approx(sigma_h)
    assert abs(a_t["A"] - z_t["A"]) > 0.9, "using z alone would be wrong by a factor of order sigma_h"
    assert mu_t["A"] == pytest.approx(float(fit.theta_shrunk) * sigma_h)


def test_apply_calibration_refuses_a_missing_score_or_volatility() -> None:
    fit = fit_calibration(*make_panels(20, 12, 0.05, seed=16), make_spec())
    index = pd.Index(["A", "B"])
    with pytest.raises(CalibrationError, match="non-finite entries"):
        apply_calibration(fit, pd.Series([1.0, np.nan], index=index), pd.Series([0.07, 0.07], index=index))
    with pytest.raises(CalibrationError, match="non-positive volatility"):
        apply_calibration(fit, pd.Series([1.0, 1.0], index=index), pd.Series([0.07, 0.0], index=index))
    with pytest.raises(CalibrationError, match="share one index"):
        apply_calibration(
            fit,
            pd.Series([1.0, 1.0], index=index),
            pd.Series([0.07, 0.07], index=pd.Index(["B", "A"])),
        )


# ---------------------------------------------------------------------------------
# 7. The bootstrap and the two arms' endpoints (§3.4, "Uncertainty")
# ---------------------------------------------------------------------------------


def test_the_point_arm_endpoints_are_the_shrunk_slope_itself() -> None:
    scores, returns, sigma = make_panels(30, 20, 0.05, seed=17)
    fit = fit_calibration(scores, returns, sigma, make_spec(confidence=None))
    assert fit.lower == fit.upper == fit.theta_shrunk
    assert fit.confidence is None
    assert not fit.straddles_zero
    assert objective_endpoints(fit) == (fit.theta_shrunk, fit.theta_shrunk)


def test_the_uncertainty_arm_endpoints_are_the_declared_quantiles() -> None:
    scores, returns, sigma = make_panels(30, 20, 0.05, seed=17)
    spec = make_spec(confidence=0.80)
    fit = fit_calibration(scores, returns, sigma, spec)
    assert fit.confidence == pytest.approx(0.80)
    assert fit.lower < fit.theta_shrunk < fit.upper
    summary = fit.method["bootstrap"]
    assert summary["blocks_drawn"] == 10  # ceil(30 / 3)
    assert summary["n_values"] == spec.n_resamples


def test_the_resampled_slope_matches_a_period_by_period_refit() -> None:
    """The vectorized refit is the same number the declared resampler's callback gives."""
    scores, returns, sigma = make_panels(24, 15, 0.05, seed=18)
    spec = make_spec(confidence=0.80, n_resamples=64)
    fit = fit_calibration(scores, returns, sigma, spec)

    z = standardize_panel(scores, "rank_normal").to_numpy(dtype=float)
    y = (returns / sigma).to_numpy(dtype=float)
    weights = np.full(z.shape, 1.0 / z.shape[1])
    units = [(float(np.sum(weights[p] * z[p] * y[p])), float(np.sum(weights[p] * z[p] * z[p]))) for p in range(24)]
    shrink = spec.shrink_factor(24)

    def slope(drawn: list[tuple[float, float]]) -> float:
        numerator = sum(pair[0] for pair in drawn)
        denominator = sum(pair[1] for pair in drawn)
        return shrink * numerator / denominator if denominator else float("nan")

    reference = block_bootstrap(
        units,
        slope,
        BlockBootstrapSpec(block_periods=3, n_resamples=64, seed=spec.seed, min_blocks=4),
    )
    lower, upper = reference.one_sided_endpoints(0.80)
    assert fit.lower == pytest.approx(lower)
    assert fit.upper == pytest.approx(upper)


def test_a_refit_reproduces_the_same_endpoints() -> None:
    panels = make_panels(30, 20, 0.05, seed=19)
    spec = make_spec(confidence=0.80)
    first = fit_calibration(*panels, spec)
    second = fit_calibration(*panels, spec)
    assert (first.lower, first.upper, first.theta_hat) == (second.lower, second.upper, second.theta_hat)


def test_the_two_arms_agree_on_everything_except_the_endpoints() -> None:
    panels = make_panels(30, 20, 0.05, seed=20)
    point = fit_calibration(*panels, load_calibration(POINT_FILE))
    p80 = fit_calibration(*panels, load_calibration(P80_FILE))
    assert point.theta_hat == p80.theta_hat
    assert point.theta_shrunk == p80.theta_shrunk
    assert point.shrink_factor == p80.shrink_factor
    assert (point.n_periods, point.n_observations, point.blocks_drawn) == (
        p80.n_periods,
        p80.n_observations,
        p80.blocks_drawn,
    )
    assert point.excluded == p80.excluded
    assert point.sign_ok == p80.sign_ok
    assert (p80.lower, p80.upper) != (point.lower, point.upper)


# ---------------------------------------------------------------------------------
# 8. Exclusion: the same folds for every arm (§3.4, D6)
# ---------------------------------------------------------------------------------


def test_a_short_training_pool_is_excluded_not_failed() -> None:
    scores, returns, sigma = make_panels(11, 20, 0.05, seed=21)
    fit = fit_calibration(scores, returns, sigma, make_spec())
    assert fit.excluded
    assert fit.exclusion_reason is not None and "min_training_periods" in fit.exclusion_reason
    assert fit.theta_hat is None and fit.lower is None and fit.upper is None
    assert fit.disabled_reason == "fold_excluded"
    assert not fit.sign_ok


def test_too_few_blocks_is_an_exclusion_with_its_reason() -> None:
    spec = make_spec(min_training_periods=1, min_blocks=4, block_periods=3)
    scores, returns, sigma = make_panels(9, 20, 0.05, seed=22)
    fit = fit_calibration(scores, returns, sigma, spec)
    assert fit.excluded and fit.blocks_drawn == 3
    assert fit.exclusion_reason is not None and "min_blocks" in fit.exclusion_reason


def test_a_pool_with_no_labelled_period_is_excluded_not_a_silent_zero() -> None:
    scores, returns, sigma = make_panels(15, 20, 0.05, seed=23)
    returns.loc[:, :] = np.nan
    fit = fit_calibration(scores, returns, sigma, make_spec())
    assert fit.excluded and fit.n_periods == 0
    assert fit.theta_hat is None


def test_an_excluded_fold_is_excluded_for_all_three_arms_alike() -> None:
    """D6: a fold is never dropped for one arm and kept for another."""
    point = load_calibration(POINT_FILE)
    p80 = load_calibration(P80_FILE)
    for n_periods in range(0, 20):
        control = fold_exclusion(n_periods, point)  # the decile_control arm fits no slope
        assert control.to_dict() == fold_exclusion(n_periods, p80).to_dict()
        if n_periods == 0:
            continue
        scores, returns, sigma = make_panels(n_periods, 12, 0.05, seed=100 + n_periods)
        point_fit = fit_calibration(scores, returns, sigma, point)
        p80_fit = fit_calibration(scores, returns, sigma, p80)
        assert point_fit.excluded == p80_fit.excluded == control.excluded, n_periods
        assert point_fit.exclusion_reason == p80_fit.exclusion_reason == control.reason
    assert fold_exclusion(11, point).excluded and not fold_exclusion(12, point).excluded


def test_an_excluded_fit_refuses_to_be_used() -> None:
    scores, returns, sigma = make_panels(11, 20, 0.05, seed=24)
    fit = fit_calibration(scores, returns, sigma, make_spec())
    with pytest.raises(CalibrationError, match="excluded"):
        objective_endpoints(fit)
    with pytest.raises(CalibrationError, match="excluded"):
        apply_calibration(fit, pd.Series([1.0]), pd.Series([0.07]))


def test_every_fit_record_is_canonicalizable_json() -> None:
    """§3.12 canonicalizes with ``stable_json_dumps``, which refuses NaN."""
    panels = make_panels(30, 20, 0.05, seed=25)
    for spec in (load_calibration(POINT_FILE), load_calibration(P80_FILE)):
        record = fit_calibration(*panels, spec).to_record()
        assert stable_json_dumps(record)
        assert set(record) >= {
            "theta_hat",
            "theta_shrunk",
            "shrink_factor",
            "lower",
            "upper",
            "straddles_zero",
            "n_periods",
            "n_observations",
            "blocks_drawn",
            "fit_cutoff",
            "sign_ok",
            "excluded",
            "exclusion_reason",
            "method",
        }
        assert "coverage_statement" in record["method"]
    excluded = fit_calibration(*make_panels(11, 20, 0.05, seed=26), load_calibration(P80_FILE))
    assert stable_json_dumps(excluded.to_record())


# ---------------------------------------------------------------------------------
# 9. The sign policy (§3.4 [Q9])
# ---------------------------------------------------------------------------------


def test_a_wrong_sign_slope_disables_new_exposure_and_is_recorded() -> None:
    scores, returns, sigma = make_panels(30, 20, -0.08, seed=27, noise=0.2)
    fit = fit_calibration(scores, returns, sigma, make_spec(expected_sign="positive", confidence=0.80))
    assert fit.theta_shrunk is not None and fit.theta_shrunk < 0.0
    assert not fit.sign_ok
    assert fit.disabled_reason == "sign_contradiction"
    assert not fit.excluded, "a sign contradiction is a research outcome, not an exclusion"
    assert objective_endpoints(fit) == (0.0, 0.0), "no new directional exposure"
    assert fit.lower is not None and fit.lower < 0.0, "the estimate is recorded with its own sign"
    assert fit.to_record()["theta_shrunk"] < 0.0, "the slope is never inverted"


def test_a_correct_sign_slope_survives_a_negative_lower_endpoint() -> None:
    """§3.4: a negative lower endpoint under a correct-sign point estimate is uncertainty."""
    scores, returns, sigma = make_panels(30, 20, 0.02, seed=29, noise=4.0)
    fit = fit_calibration(scores, returns, sigma, make_spec(confidence=0.80))
    assert fit.theta_shrunk is not None and fit.theta_shrunk > 0.0
    assert fit.lower is not None and fit.lower < 0.0 < fit.upper
    assert fit.sign_ok and fit.disabled_reason is None
    assert fit.straddles_zero
    assert objective_endpoints(fit) == (fit.lower, fit.upper), "the endpoints pass through unchanged"


def test_the_expected_sign_can_be_negative() -> None:
    scores, returns, sigma = make_panels(30, 20, -0.08, seed=33, noise=0.2)
    fit = fit_calibration(scores, returns, sigma, make_spec(expected_sign="negative"))
    assert fit.sign_ok and fit.disabled_reason is None


# ---------------------------------------------------------------------------------
# 10. Estimation failures, which raise (§3.4)
# ---------------------------------------------------------------------------------


def test_a_degenerate_denominator_raises_rather_than_returning_zero() -> None:
    index = pd.RangeIndex(14)
    columns = pd.Index(["A"])
    scores = pd.DataFrame(1.0, index=index, columns=columns)  # one name: rank_normal gives z = 0
    returns = pd.DataFrame(0.01, index=index, columns=columns)
    sigma = pd.DataFrame(0.08, index=index, columns=columns)
    with pytest.raises(CalibrationError, match="no cross-sectional variation"):
        fit_calibration(scores, returns, sigma, make_spec(min_training_periods=1, min_blocks=1))


def test_an_empty_panel_is_a_caller_defect() -> None:
    empty = pd.DataFrame(index=pd.RangeIndex(0), columns=pd.Index(["A"]), dtype=float)
    with pytest.raises(CalibrationError, match="empty"):
        fit_calibration(empty, empty, empty, make_spec())


def test_misaligned_or_unordered_panels_are_refused() -> None:
    scores, returns, sigma = make_panels(14, 6, 0.05, seed=30)
    with pytest.raises(CalibrationError, match="index does not equal"):
        fit_calibration(scores, returns.iloc[:-1], sigma.iloc[:-1], make_spec())
    with pytest.raises(CalibrationError, match="columns do not equal"):
        fit_calibration(scores, returns.rename(columns={"S000": "ZZZ"}), sigma, make_spec())
    shuffled = scores.iloc[::-1]
    with pytest.raises(CalibrationError, match="not ascending"):
        fit_calibration(shuffled, returns.iloc[::-1], sigma.iloc[::-1], make_spec())


def test_a_block_shorter_than_a_label_is_refused() -> None:
    scores, returns, sigma = make_panels(20, 6, 0.05, seed=31)
    with pytest.raises(ValueError, match="shorter than horizon_bars"):
        fit_calibration(scores, returns, sigma, make_spec(block_periods=1), horizon_bars=63, update_every_bars=21)


def test_inner_out_of_sample_periods_are_required_when_declared() -> None:
    scores, returns, sigma = make_panels(20, 6, 0.05, seed=32)
    spec = make_spec(inner_oos_required=True)
    with pytest.raises(CalibrationError, match="inner out-of-sample"):
        fit_calibration(scores, returns, sigma, spec)
    inner = list(scores.index[4:])
    fit = fit_calibration(scores, returns, sigma, spec, inner)
    assert fit.n_periods == 16
    assert fit.method["sample"]["inner_oos_periods"] == 16
    with pytest.raises(CalibrationError, match="absent from the panels"):
        fit_calibration(scores, returns, sigma, spec, [999])


# ---------------------------------------------------------------------------------
# 11. What the bootstrap actually achieves (§3.4, "Uncertainty")
# ---------------------------------------------------------------------------------


def _coverage(
    *,
    reps: int,
    n_periods: int,
    n_names: int,
    theta: float,
    rho: float,
    overlap: int,
    block_periods: int,
    n_resamples: int = 250,
) -> tuple[float, float, float]:
    """(lower coverage, upper coverage, mean endpoint gap) of the shrunk estimand."""
    spec = make_spec(confidence=0.80, block_periods=block_periods, n_resamples=n_resamples)
    target = theta * spec.shrink_factor(n_periods)
    below = above = 0
    gaps = []
    for rep in range(reps):
        panels = make_panels(n_periods, n_names, theta, seed=1000 + rep, rho=rho, overlap=overlap)
        fit = fit_calibration(*panels, spec)
        assert fit.lower is not None and fit.upper is not None
        below += fit.lower <= target
        above += target <= fit.upper
        gaps.append(fit.upper - fit.lower)
    return below / reps, above / reps, float(np.mean(gaps))


def test_coverage_is_near_nominal_on_independent_periods() -> None:
    """Measured, not assumed: 120 replications at nominal 0.80 per side.

    Run (seeds 1000-1119, 40 periods of 40 names, 250 resamples): lower endpoint coverage
    0.783, upper 0.808, mean endpoint gap 0.0311. The Monte Carlo standard error at
    p = 0.8 on 120 draws is 0.037, so the band below is roughly +/- 3 standard errors: it
    passes a calibrated bootstrap and fails a broken one.
    """
    lower, upper, _ = _coverage(reps=120, n_periods=40, n_names=40, theta=0.05, rho=0.0, overlap=0, block_periods=3)
    assert 0.68 <= lower <= 0.92, lower
    assert 0.68 <= upper <= 0.92, upper


def test_the_block_scheme_carries_dependence_an_iid_resample_misses() -> None:
    """Persistent scores plus overlapping labels: the case ``block_periods`` exists for.

    Run (seeds 1000-1119, 40 periods of 40 names, AR(1) scores at rho = 0.9 and MA(2)
    label errors, 250 resamples, nominal 0.80 per side): with ``block_periods = 3``
    coverage is lower 0.725 / upper 0.758 at a mean endpoint gap of 0.0419; with
    ``block_periods = 1`` -- an iid resample of whole periods -- it falls to lower 0.675 /
    upper 0.692 at a gap of 0.0313. On the frozen reference geometry (13 periods of 96
    names, 5 blocks, Run) it is lower 0.708 / upper 0.700 at a gap of 0.0267.

    So the declared scheme widens the interval by a third under dependence and recovers
    part of the shortfall, and it **still undercovers**: at nominal 0.80 the achieved
    one-sided coverage is about 0.71 to 0.76 on dependent data, and about 0.70 at the
    reference fold geometry. The endpoints are optimistic there. That is a property of the
    declared method measured before the experiment runs, to be reported with the result --
    not a defect to patch and not a reason to widen an arm after seeing an outcome.
    """
    blocked_lower, blocked_upper, blocked_gap = _coverage(
        reps=120, n_periods=40, n_names=40, theta=0.05, rho=0.9, overlap=2, block_periods=3
    )
    iid_lower, iid_upper, iid_gap = _coverage(
        reps=120, n_periods=40, n_names=40, theta=0.05, rho=0.9, overlap=2, block_periods=1
    )
    assert blocked_gap > iid_gap, "the block scheme must widen the interval under dependence"
    assert blocked_lower > iid_lower and blocked_upper > iid_upper
    assert blocked_upper < 0.80, "and it still undercovers; that is reported, not repaired"
