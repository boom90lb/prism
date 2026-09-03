"""Pure-helper tests for the run-directory breadth diagnostic (SPEC N6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.research

from research.scripts.breadth_diagnostic import (  # noqa: E402
    contribution_panel,
    horizon_rank_ics,
    spectrum_shares,
)


def _nyse_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=n, freq="B", tz="America/New_York")


class TestContributionPanel:
    def test_contribution_is_prior_weight_times_todays_return(self):
        idx = _nyse_index(4)
        closes = pd.DataFrame({"A": [100.0, 110.0, 121.0, 121.0], "B": [50.0, 50.0, 45.0, 45.0]}, index=idx)
        weights = pd.DataFrame({"A": [0.5, 0.5, 0.0, 0.0], "B": [-0.5, -0.5, 0.0, 0.0]}, index=idx)
        contrib = contribution_panel(weights, closes)
        # Day 0 has no prior book; days 3+ have a zero prior book -> only days 1-2 are active.
        assert list(contrib.index) == list(idx[1:3])
        assert contrib.loc[idx[1], "A"] == pytest.approx(0.5 * 0.10)
        assert contrib.loc[idx[2], "B"] == pytest.approx(-0.5 * -0.10)

    def test_never_held_names_are_dropped(self):
        idx = _nyse_index(3)
        closes = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [1.0, 1.0, 1.0]}, index=idx)
        weights = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [0.0, 0.0, 0.0]}, index=idx)
        assert list(contribution_panel(weights, closes).columns) == ["A"]

    def test_missing_return_under_held_weight_stays_nan(self):
        idx = _nyse_index(3)
        closes = pd.DataFrame({"A": [1.0, np.nan, np.nan]}, index=idx)
        weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
        contrib = contribution_panel(weights, closes)
        assert contrib["A"].isna().any()


class TestSpectrumShares:
    def test_rank_one_covariance_concentrates(self):
        v = np.array([1.0, 1.0, 1.0])
        cov = pd.DataFrame(np.outer(v, v))
        shares = spectrum_shares(cov)
        assert shares[0] == pytest.approx(1.0)
        assert shares[1] == pytest.approx(0.0, abs=1e-12)

    def test_identity_covariance_is_flat(self):
        shares = spectrum_shares(pd.DataFrame(np.eye(4)))
        assert shares == pytest.approx([0.25, 0.25, 0.25])

    def test_zero_covariance_is_nan(self):
        assert all(np.isnan(s) for s in spectrum_shares(pd.DataFrame(np.zeros((3, 3)))))


class TestHorizonRankICs:
    def test_perfect_monotone_signal_scores_ic_one(self):
        idx = _nyse_index(6)
        h = 2
        rng = np.random.default_rng(0)
        closes = pd.DataFrame(
            np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(6, 5)), axis=0) * 100.0,
            index=idx,
            columns=list("ABCDE"),
        )
        fwd = closes.shift(-h) / closes - 1.0
        ics = horizon_rank_ics(fwd, closes, idx, h)  # score := the forward return itself
        assert len(ics) == 4  # last h bars have no forward window
        assert np.allclose(ics.to_numpy(), 1.0)

    def test_bars_outside_panel_are_skipped(self):
        idx = _nyse_index(4)
        closes = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 3.0, 2.0, 1.0], "C": [1.0, 1.5, 2.0, 2.5]}, index=idx)
        scores = closes.copy()
        foreign = pd.DatetimeIndex([pd.Timestamp("2030-01-02", tz="America/New_York")])
        assert horizon_rank_ics(scores, closes, foreign, 1).empty
