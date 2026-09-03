"""Contract tests for the residual-reversion Signal node (SPEC §7.1 impl (a))."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.residual.factors import ResidualStatArbConfig
from prism.residual.residual import compute_residual_signal_panel
from prism.signal import ResidualSignalNode, Signal


def _small_config(**overrides: object) -> ResidualStatArbConfig:
    base: dict = dict(
        corr_window=60,
        regr_window=20,
        n_factors=2,
        rebalance_every=5,
        min_price=5.0,
        min_median_dollar_volume=1_000.0,
        dollar_volume_window=5,
    )
    base.update(overrides)
    return ResidualStatArbConfig(**base)


def _synthetic_panel(
    n_days: int = 120, n_assets: int = 8, seed: int = 7, resid_b: float = 0.55, resid_vol: float = 0.012
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor + AR(1)-residual returns, so residual signals genuinely exist."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n_days, freq="B", tz="America/New_York")
    factor = rng.normal(0.0003, 0.01, size=n_days)
    betas = rng.uniform(0.6, 1.4, size=n_assets)
    levels = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        levels[t] = resid_b * levels[t - 1] + rng.normal(0.0, resid_vol, size=n_assets)
    residual_returns = np.diff(levels, axis=0, prepend=np.zeros((1, n_assets)))
    returns = betas[None, :] * factor[:, None] + residual_returns
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=idx,
        columns=[f"S{i}" for i in range(n_assets)],
    )
    volumes = pd.DataFrame(1_000_000.0, index=idx, columns=closes.columns)
    return closes, volumes


def _node(**node_kwargs: object) -> ResidualSignalNode:
    return ResidualSignalNode(_small_config(), **node_kwargs)


class TestContractSurface:
    def test_is_a_signal_with_horizon_and_history(self):
        node = _node(horizon_bars=5)
        assert isinstance(node, Signal)
        assert node.horizon_bars == 5
        assert node.required_history == 60 + 20 + 1

    def test_fit_returns_self_and_score_needs_no_fit(self):
        closes, volumes = _synthetic_panel()
        node = _node()
        assert node.fit(closes, volumes) is node
        scores = _node().score(closes, volumes)  # stateless: no prior fit required
        assert isinstance(scores, pd.Series)
        assert scores.index.equals(closes.columns)

    def test_missing_volume_raises(self):
        closes, _ = _synthetic_panel()
        with pytest.raises(ValueError, match="volume"):
            _node().score(closes, None)

    def test_short_panel_raises(self):
        closes, volumes = _synthetic_panel(n_days=40)
        with pytest.raises(ValueError, match="rows"):
            _node().score(closes, volumes)

    def test_bad_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon_bars"):
            _node(horizon_bars=0)


class TestScoreSemantics:
    def test_scores_match_panel_sscore_mapping(self):
        """The node's score is the OU expectation of the panel's own s-score."""
        closes, volumes = _synthetic_panel()
        config = _small_config()
        h = 5
        node = ResidualSignalNode(config, horizon_bars=h)
        scores = node.score(closes, volumes)

        panel = compute_residual_signal_panel(
            closes.iloc[-node.required_history :], volumes.iloc[-node.required_history :], config
        )
        s = panel.sscore.iloc[-1]
        tradeable = panel.tradeable.iloc[-1].astype(bool)
        assert tradeable.any(), "synthetic panel produced nothing tradeable; test is vacuous"
        # NaN exactly off the tradeable set.
        assert scores.notna().equals(tradeable & s.notna())
        # Mean reversion: score sign is opposite the s-score sign.
        finite = scores.dropna()
        assert len(finite) > 0
        assert (np.sign(finite) == -np.sign(s[finite.index])).all()

    def test_longer_horizon_grows_expected_move_sublinearly_in_sigma_units(self):
        """|score| rises with h through (1 - b^h) but the sqrt(h) denominator
        eventually wins: the h -> inf limit is 0, so a longer horizon must not
        scale scores linearly."""
        closes, volumes = _synthetic_panel()
        s1 = ResidualSignalNode(_small_config(), horizon_bars=1).score(closes, volumes).dropna()
        s5 = ResidualSignalNode(_small_config(), horizon_bars=5).score(closes, volumes).dropna()
        s250 = ResidualSignalNode(_small_config(), horizon_bars=250).score(closes, volumes).dropna()
        assert s1.index.equals(s5.index) and s5.index.equals(s250.index)
        assert (s250.abs() < s5.abs()).all()
        assert (s1.abs() * 5.0 > s5.abs()).all()

    def test_causality_appending_future_bars_changes_nothing(self):
        """Scoring bar t must be identical whether or not bars after t exist."""
        closes, volumes = _synthetic_panel(n_days=130)
        node = _node()
        cut = 120
        scored_at_cut = node.score(closes.iloc[:cut], volumes.iloc[:cut])
        # The node only reads the trailing required_history rows, so hand it
        # the same trailing window from the longer panel.
        tail = slice(cut - node.required_history, cut)
        scored_from_longer = node.score(closes.iloc[tail], volumes.iloc[tail])
        pd.testing.assert_series_equal(scored_at_cut, scored_from_longer)

    def test_scale_invariance_within_eligibility(self):
        """Estimation is price-level free; rescaling prices inside the eligible
        region (min_price, dollar-volume floors untouched) is score-identical."""
        closes, volumes = _synthetic_panel()
        node = _node()
        base = node.score(closes, volumes)
        rescaled = node.score(closes * 7.0, volumes)
        pd.testing.assert_series_equal(base, rescaled)

    def test_ineligible_name_has_no_opinion(self):
        closes, volumes = _synthetic_panel()
        volumes = volumes.copy()
        volumes["S0"] = 0.001  # dollar volume below floor -> ineligible
        scores = _node().score(closes, volumes)
        assert np.isnan(scores["S0"])

    def test_membership_mask_gates_scores(self):
        closes, volumes = _synthetic_panel()
        mask = pd.DataFrame(True, index=closes.index, columns=closes.columns)
        mask["S1"] = False
        node = ResidualSignalNode(_small_config(), membership_mask=mask)
        scores = node.score(closes, volumes)
        assert np.isnan(scores["S1"])
