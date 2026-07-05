"""Tests for the fundamental-law breadth diagnostics (SPEC.md N6, §10, law 3a)."""

import numpy as np
import pandas as pd
import pytest

from prism.validation.metrics import (
    effective_breadth,
    effective_breadth_from_cov,
    fundamental_law_diagnostic,
    information_ratio_ceiling,
    rank_information_coefficient,
)


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


class TestFundamentalLawDiagnostic:
    def test_falsification_fires_when_realized_exceeds_ceiling(self):
        # ceiling = 0.03*sqrt(50) ~ 0.212; realized 0.9 is impossibly above it.
        d = fundamental_law_diagnostic(0.9, 0.03, 50.0)
        assert d["falsification"] == 1.0

    def test_no_falsification_below_ceiling(self):
        d = fundamental_law_diagnostic(0.15, 0.03, 50.0)
        assert d["falsification"] == 0.0

    def test_viability_margin_and_flag(self):
        # ceiling ~0.212; hurdle 0.05 -> viable, margin ~0.162.
        d = fundamental_law_diagnostic(0.1, 0.03, 50.0, after_cost_hurdle=0.05)
        assert d["viable"] == 1.0
        assert d["viability_margin"] == pytest.approx(0.03 * np.sqrt(50) - 0.05)

    def test_not_viable_when_ceiling_below_hurdle(self):
        # Thin IC and low breadth -> ceiling below a demanding hurdle.
        d = fundamental_law_diagnostic(0.0, 0.01, 4.0, after_cost_hurdle=0.10)
        assert d["viable"] == 0.0

    def test_nan_inputs_propagate_not_false(self):
        d = fundamental_law_diagnostic(float("nan"), 0.03, 50.0)
        assert np.isnan(d["falsification"])
