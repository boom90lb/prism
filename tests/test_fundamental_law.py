"""Tests for the breadth diagnostics (SPEC.md N6; engine specification §3.12 [Q19]).

``breadth_diagnostic`` is the current function: breadth ratio, realized value, and whether
the realized exceeds the diagnostic — no hurdle, no viability verdict.
``fundamental_law_diagnostic`` is the legacy alias kept only to read recorded artifacts, and
the classes below that exercise its hurdle fields are testing that legacy shape, not a
supported way to decide anything.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.validation.metrics import (
    after_cost_hurdle_periodic,
    breadth_diagnostic,
    effective_breadth,
    effective_breadth_from_cov,
    fundamental_law_diagnostic,
    information_ratio_ceiling,
    rank_information_coefficient,
)

RECORDED_DIAGNOSTIC = Path(__file__).resolve().parents[1] / "results" / "demotion_b1" / "breadth_diagnostic.json"


def _legacy(*args, **kwargs):
    """Call the deprecated alias without the warning failing an unrelated assertion."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return fundamental_law_diagnostic(*args, **kwargs)


class TestAfterCostHurdle:
    def test_tbill_anchor_conversion(self):
        # 5.04% annual over 252 days at 1% daily vol -> 0.02 daily Sharpe.
        assert after_cost_hurdle_periodic(0.0504, 0.01) == pytest.approx(0.02)

    def test_zero_hurdle_is_explicit_zero(self):
        assert after_cost_hurdle_periodic(0.0, 0.01) == 0.0

    def test_degenerate_vol_is_nan_not_zero(self):
        assert np.isnan(after_cost_hurdle_periodic(0.05, 0.0))
        assert np.isnan(after_cost_hurdle_periodic(float("nan"), 0.01))

    def test_rejects_bad_periods(self):
        with pytest.raises(ValueError, match="periods_per_year"):
            after_cost_hurdle_periodic(0.05, 0.01, periods_per_year=0)

    def test_legacy_alias_reads_viability_off_the_hurdle(self):
        # The legacy shape only: a thin ceiling under a real-terms hurdle flips its
        # `viable` field off. The current diagnostic carries no such field, and the
        # after-cost read is a comparison of return series (prism.validation.compare).
        hurdle = after_cost_hurdle_periodic(0.0504, 0.01)
        diag = _legacy(
            0.01,
            information_coefficient=0.002,
            effective_breadth_value=25.0,
            after_cost_hurdle=hurdle,
        )
        assert diag["viable"] == 0.0  # ceiling 0.01 < hurdle 0.02
        diag_zero = _legacy(
            0.01,
            information_coefficient=0.002,
            effective_breadth_value=25.0,
        )
        assert diag_zero["viable"] == 1.0  # the implicit-zero overstatement


class TestEffectiveBreadth:
    def test_zero_correlation_gives_full_breadth(self):
        assert effective_breadth(0.0, 100) == pytest.approx(100.0)

    def test_unit_correlation_collapses_to_one(self):
        assert effective_breadth(1.0, 100) == pytest.approx(1.0, abs=1e-6)

    def test_equicorrelation_formula(self):
        # N/(1+(N-1)rho) = 50/(1+49*0.2)
        assert effective_breadth(0.2, 50) == pytest.approx(50 / (1 + 49 * 0.2))

    def test_limit_is_inverse_rho(self):
        # As N grows, N_eff -> 1/rho (the IC/sqrt(rho) ceiling basis).
        assert effective_breadth(0.25, 100000) == pytest.approx(1 / 0.25, rel=1e-2)

    def test_negative_correlation_clamped_not_exceeding_n(self):
        # A single average negative rho is unphysical for a large PSD book; clamp.
        assert effective_breadth(-0.5, 20) == pytest.approx(20.0)

    def test_single_name_and_degenerate(self):
        assert effective_breadth(0.3, 1) == 1.0
        assert np.isnan(effective_breadth(0.3, 0))
        assert np.isnan(effective_breadth(float("nan"), 10))


class TestEffectiveBreadthFromCov:
    def test_rank_one_is_one_bet(self):
        rank1 = np.outer(np.ones(10), np.ones(10))
        assert effective_breadth_from_cov(rank1) == pytest.approx(1.0)

    def test_identity_is_n_bets(self):
        assert effective_breadth_from_cov(np.eye(8)) == pytest.approx(8.0)

    def test_one_dominant_factor_is_near_one(self):
        # Big common factor + tiny idiosyncratic -> ~1 effective bet.
        cov = np.outer(np.ones(20), np.ones(20)) * 100.0 + np.eye(20) * 0.01
        assert effective_breadth_from_cov(cov) < 1.05

    def test_dataframe_and_degenerate(self):
        assert effective_breadth_from_cov(pd.DataFrame(np.eye(4))) == pytest.approx(4.0)
        assert np.isnan(effective_breadth_from_cov(np.zeros((3, 3))))
        assert np.isnan(effective_breadth_from_cov(np.array([1.0, 2.0])))  # not 2D


class TestInformationRatioCeiling:
    def test_ceiling_is_ic_sqrt_breadth(self):
        assert information_ratio_ceiling(0.03, 50) == pytest.approx(0.03 * np.sqrt(50))

    def test_uses_absolute_ic(self):
        assert information_ratio_ceiling(-0.03, 50) == pytest.approx(0.03 * np.sqrt(50))

    def test_degenerate(self):
        assert np.isnan(information_ratio_ceiling(0.03, -1))
        assert np.isnan(information_ratio_ceiling(float("nan"), 50))


class TestRankIC:
    def test_perfect_monotone_is_one(self):
        s = np.array([1.0, 2, 3, 4, 5])
        assert rank_information_coefficient(s, s * 10 + 1) == pytest.approx(1.0)

    def test_anti_monotone_is_minus_one(self):
        s = np.array([1.0, 2, 3, 4, 5])
        assert rank_information_coefficient(s, -s) == pytest.approx(-1.0)

    def test_nan_pairs_dropped(self):
        s = np.array([1.0, 2, np.nan, 4, 5])
        r = np.array([1.0, 2, 3, 4, np.nan])
        assert rank_information_coefficient(s, r) == pytest.approx(1.0)

    def test_too_few_pairs_is_nan(self):
        assert np.isnan(rank_information_coefficient(np.array([1.0, 2]), np.array([1.0, 2])))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            rank_information_coefficient(np.array([1.0, 2, 3]), np.array([1.0, 2]))


class TestBreadthDiagnostic:
    """The current diagnostic: a ratio and a prompt to investigate, not a verdict."""

    def test_fields_are_exactly_the_declared_six(self):
        d = breadth_diagnostic(0.15, 0.03, 50.0)
        assert set(d) == {
            "n_eff",
            "information_coefficient",
            "ir_ceiling",
            "realized",
            "breadth_ratio",
            "exceeds_diagnostic",
        }

    def test_carries_no_hurdle_and_no_viability_verdict(self):
        d = breadth_diagnostic(0.15, 0.03, 50.0)
        for dropped in ("after_cost_hurdle", "viable", "viability_margin", "falsification"):
            assert dropped not in d, f"{dropped} is dropped from the diagnostic (§3.12 [Q19])"

    def test_breadth_ratio_is_realized_over_the_implied_ceiling(self):
        d = breadth_diagnostic(0.15, 0.03, 50.0)
        assert d["ir_ceiling"] == pytest.approx(0.03 * np.sqrt(50.0))
        assert d["breadth_ratio"] == pytest.approx(0.15 / (0.03 * np.sqrt(50.0)))

    def test_exceeding_the_diagnostic_is_flagged(self):
        assert breadth_diagnostic(0.9, 0.03, 50.0)["exceeds_diagnostic"] == 1.0
        assert breadth_diagnostic(0.15, 0.03, 50.0)["exceeds_diagnostic"] == 0.0
        assert breadth_diagnostic(0.15, 0.03, 50.0)["breadth_ratio"] < 1.0

    def test_ratio_above_one_is_exactly_the_exceeding_case(self):
        d = breadth_diagnostic(0.9, 0.03, 50.0)
        assert d["breadth_ratio"] > 1.0
        assert d["exceeds_diagnostic"] == 1.0

    def test_uses_absolute_ic_like_the_ceiling(self):
        assert breadth_diagnostic(0.15, -0.03, 50.0)["ir_ceiling"] == pytest.approx(0.03 * np.sqrt(50.0))

    def test_nan_and_negative_breadth_propagate(self):
        for d in (
            breadth_diagnostic(float("nan"), 0.03, 50.0),
            breadth_diagnostic(0.15, float("nan"), 50.0),
            breadth_diagnostic(0.15, 0.03, -1.0),
        ):
            assert np.isnan(d["exceeds_diagnostic"])
            assert np.isnan(d["breadth_ratio"])

    def test_zero_ceiling_gives_no_ratio_rather_than_infinity(self):
        d = breadth_diagnostic(0.15, 0.0, 50.0)
        assert d["ir_ceiling"] == 0.0
        assert np.isnan(d["breadth_ratio"])
        assert d["exceeds_diagnostic"] == 1.0

    def test_reproduces_the_recorded_demotion_b1_numbers(self):
        # The recorded artifact is immutable evidence; the new diagnostic must reproduce
        # its ceiling and its net capture ratio exactly, or the rename changed a number.
        recorded = json.loads(RECORDED_DIAGNOSTIC.read_text())
        net = recorded["diagnostic_horizon_net"]
        d = breadth_diagnostic(net["realized_periodic_sharpe"], net["information_coefficient"], net["n_eff"])
        assert d["ir_ceiling"] == pytest.approx(net["ir_ceiling"], rel=0, abs=0)
        assert d["realized"] == pytest.approx(net["realized_periodic_sharpe"], rel=0, abs=0)
        assert d["breadth_ratio"] == pytest.approx(recorded["capture"]["net_over_ceiling"], rel=1e-15)
        assert d["exceeds_diagnostic"] == recorded["diagnostic_horizon_net"]["falsification"]


class TestLegacyAlias:
    """The alias stays readable for recorded artifacts and says it is deprecated."""

    def test_emits_a_deprecation_warning_naming_the_replacement(self):
        with pytest.warns(DeprecationWarning, match="breadth_diagnostic"):
            fundamental_law_diagnostic(0.15, 0.03, 50.0)

    def test_still_returns_the_recorded_field_set(self):
        d = _legacy(0.15, 0.03, 50.0, after_cost_hurdle=0.05)
        recorded = json.loads(RECORDED_DIAGNOSTIC.read_text())["diagnostic_horizon_net"]
        assert set(d) == set(recorded)

    def test_reproduces_the_recorded_block_field_for_field(self):
        recorded = json.loads(RECORDED_DIAGNOSTIC.read_text())["diagnostic_horizon_net"]
        d = _legacy(
            recorded["realized_periodic_sharpe"],
            recorded["information_coefficient"],
            recorded["n_eff"],
            after_cost_hurdle=recorded["after_cost_hurdle"],
        )
        for key, value in recorded.items():
            assert d[key] == pytest.approx(value, rel=1e-15), key


class TestFundamentalLawDiagnosticLegacy:
    def test_falsification_fires_when_realized_exceeds_ceiling(self):
        # ceiling = 0.03*sqrt(50) ~ 0.212; realized 0.9 is impossibly above it.
        d = _legacy(0.9, 0.03, 50.0)
        assert d["falsification"] == 1.0

    def test_no_falsification_below_ceiling(self):
        d = _legacy(0.15, 0.03, 50.0)
        assert d["falsification"] == 0.0

    def test_viability_margin_and_flag(self):
        # ceiling ~0.212; hurdle 0.05 -> viable, margin ~0.162.
        d = _legacy(0.1, 0.03, 50.0, after_cost_hurdle=0.05)
        assert d["viable"] == 1.0
        assert d["viability_margin"] == pytest.approx(0.03 * np.sqrt(50) - 0.05)

    def test_not_viable_when_ceiling_below_hurdle(self):
        # Thin IC and low breadth -> ceiling below a demanding hurdle.
        d = _legacy(0.0, 0.01, 4.0, after_cost_hurdle=0.10)
        assert d["viable"] == 0.0

    def test_nan_inputs_propagate_not_false(self):
        d = _legacy(float("nan"), 0.03, 50.0)
        assert np.isnan(d["falsification"])
