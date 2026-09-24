"""Contract and property tests for the characteristics-composite nodes
(``ResidualMomentumSignalNode``, ``IdiosyncraticVolSignalNode``) and the
trailing-window residualization they share (docs/characteristics_design.md).

What these guard, in order of cost: a residual that is not causal at the
decision bar (a future bar changing a past score); a score that moves under a
global price rescale (the estimation is on returns and must not see levels);
a name outside the eligibility screen receiving an opinion; and the sign
convention of the volatility node (top decile = low volatility) drifting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.residual.factors import ResidualStatArbConfig, trailing_factor_residuals
from prism.signal import (
    IdiosyncraticVolSignalNode,
    ResidualMomentumSignalNode,
    Signal,
    check_declarations,
)

LOOKBACK, SKIP = 120, 10


def _small_config(**overrides: object) -> ResidualStatArbConfig:
    base: dict = dict(
        corr_window=60,
        regr_window=20,
        n_factors=2,
        min_price=5.0,
        min_median_dollar_volume=1_000.0,
        dollar_volume_window=5,
    )
    base.update(overrides)
    return ResidualStatArbConfig(**base)


def _panel(n_days: int = 340, n_assets: int = 12, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One common factor plus per-name idiosyncratic drift and noise of distinct scale."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n_days, freq="B", tz="America/New_York")
    market = rng.normal(0.0, 0.01, size=(n_days, 1))
    betas = np.linspace(0.6, 1.4, n_assets)[None, :]
    drifts = np.linspace(-0.0008, 0.0008, n_assets)[None, :]
    scales = np.linspace(0.003, 0.012, n_assets)[None, :]
    noise = rng.normal(0.0, 1.0, size=(n_days, n_assets)) * scales
    returns = market * betas + drifts + noise
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=idx,
        columns=[f"S{i}" for i in range(n_assets)],
    )
    volumes = pd.DataFrame(2_000_000.0, index=idx, columns=closes.columns)
    return closes, volumes


def _resmom(**kwargs: object) -> ResidualMomentumSignalNode:
    kwargs.setdefault("lookback_bars", LOOKBACK)
    kwargs.setdefault("skip_bars", SKIP)
    return ResidualMomentumSignalNode(_small_config(), **kwargs)


def _ivol(**kwargs: object) -> IdiosyncraticVolSignalNode:
    kwargs.setdefault("estimation_bars", LOOKBACK)
    kwargs.setdefault("vol_bars", 30)
    return IdiosyncraticVolSignalNode(_small_config(), **kwargs)


class TestResidualization:
    def test_residuals_are_orthogonal_to_the_fitted_factors_and_indexed_by_the_window(self):
        closes, volumes = _panel()
        config = _small_config()
        residuals = trailing_factor_residuals(closes, volumes, config, window=LOOKBACK)
        assert len(residuals) == LOOKBACK
        assert residuals.index.equals(closes.index[-LOOKBACK:])
        assert set(residuals.columns) <= set(closes.columns)
        # The intercept is retained: the residual mean is the fitted alpha, not
        # zero, and the factor part is what was removed. Check the removal by
        # regressing the residuals back on the factor returns the window implies.
        from prism.residual.factors import (
            batched_factor_ols,
            compute_returns,
            estimate_eigenportfolios,
            factor_returns_from_window,
        )

        R = compute_returns(closes).iloc[-LOOKBACK:][residuals.columns].to_numpy()
        Q, _ = estimate_eigenportfolios(R, config.n_factors)
        F = factor_returns_from_window(R, Q)
        _, beta_back, _ = batched_factor_ols(residuals.to_numpy(), F)
        assert np.allclose(beta_back, 0.0, atol=1e-9)
        assert not np.allclose(residuals.mean(axis=0).to_numpy(), 0.0, atol=1e-6)

    def test_a_thin_cross_section_returns_no_names_rather_than_a_degenerate_fit(self):
        closes, volumes = _panel(n_assets=3)
        residuals = trailing_factor_residuals(closes, volumes, _small_config(n_factors=2), window=LOOKBACK)
        assert residuals.shape[1] == 0

    def test_window_shorter_than_the_factor_count_is_refused(self):
        closes, volumes = _panel()
        with pytest.raises(ValueError, match="window"):
            trailing_factor_residuals(closes, volumes, _small_config(n_factors=2), window=5)


@pytest.mark.parametrize("make", [_resmom, _ivol], ids=["residual_momentum", "idiosyncratic_vol"])
class TestContractSurface:
    def test_is_a_signal_that_declares_itself(self, make):
        node = make(horizon_bars=21)
        assert isinstance(node, Signal)
        assert node.horizon_bars == 21
        assert node.required_history == LOOKBACK + 1
        assert node.output_kind == "rank_score" and node.units == "rank"
        check_declarations(node)

    def test_fit_returns_self_and_score_needs_no_fit(self, make):
        closes, volumes = _panel()
        node = make()
        assert node.fit(closes, volumes) is node
        scores = make().score(closes, volumes)
        assert isinstance(scores, pd.Series)
        assert scores.index.equals(closes.columns)
        assert scores.notna().sum() >= 8

    def test_missing_volume_raises(self, make):
        closes, _ = _panel()
        with pytest.raises(ValueError, match="volume"):
            make().score(closes, None)

    def test_short_panel_raises(self, make):
        closes, volumes = _panel(n_days=40)
        with pytest.raises(ValueError, match="rows"):
            make().score(closes, volumes)

    def test_scale_invariance(self, make):
        closes, volumes = _panel()
        node = make()
        pd.testing.assert_series_equal(node.score(closes, volumes), node.score(closes * 13.0, volumes))

    def test_causality_appending_future_bars_changes_nothing(self, make):
        closes, volumes = _panel(n_days=360)
        node = make()
        cut = 300
        at_cut = node.score(closes.iloc[:cut], volumes.iloc[:cut])
        tail = slice(cut - node.required_history, cut)
        from_longer = node.score(closes.iloc[tail], volumes.iloc[tail])
        pd.testing.assert_series_equal(at_cut, from_longer)

    def test_an_ineligible_name_gets_no_opinion(self, make):
        closes, volumes = _panel()
        closes = closes.copy()
        closes["S3"] = closes["S3"] / 100.0  # below min_price throughout
        scores = make().score(closes, volumes)
        assert np.isnan(scores["S3"])


class TestScoreSemantics:
    def test_residual_momentum_ranks_by_idiosyncratic_drift_not_by_beta(self):
        """Names differ in beta and in drift; the score must follow the drift."""
        closes, volumes = _panel(n_days=600, seed=3)
        scores = _resmom(lookback_bars=250, skip_bars=10).score(closes, volumes).dropna()
        drifts = pd.Series(np.linspace(-0.0008, 0.0008, closes.shape[1]), index=closes.columns)
        rho = scores.rank().corr(drifts.loc[scores.index].rank())
        assert rho > 0.6, rho

    def test_residual_momentum_skips_the_most_recent_bars(self):
        """The score is the formation-window residual sum over its own sd, skip excluded."""
        closes, volumes = _panel(n_days=360)
        config = _small_config()
        node = ResidualMomentumSignalNode(config, lookback_bars=LOOKBACK, skip_bars=SKIP)
        scores = node.score(closes, volumes)
        residuals = trailing_factor_residuals(closes, volumes, config, window=LOOKBACK)
        formation = residuals.iloc[:-SKIP]
        expected = formation.sum(axis=0) / formation.std(axis=0, ddof=1)
        pd.testing.assert_series_equal(scores.loc[expected.index], expected, check_names=False)
        assert len(formation) == LOOKBACK - SKIP

    def test_idiosyncratic_vol_top_decile_is_the_low_volatility_leg(self):
        closes, volumes = _panel(n_days=400, seed=5)
        scores = _ivol(vol_bars=60).score(closes, volumes).dropna()
        scales = pd.Series(np.linspace(0.003, 0.012, closes.shape[1]), index=closes.columns)
        rho = scores.rank().corr(scales.loc[scores.index].rank())
        assert rho < -0.6, rho
        assert (scores < 0.0).all()

    @pytest.mark.parametrize(
        "make, kwargs, match",
        [
            (ResidualMomentumSignalNode, dict(lookback_bars=3), "lookback_bars"),
            (ResidualMomentumSignalNode, dict(lookback_bars=60, skip_bars=59), "residuals"),
            (ResidualMomentumSignalNode, dict(skip_bars=-1), "skip_bars"),
            (ResidualMomentumSignalNode, dict(horizon_bars=0), "horizon_bars"),
            (IdiosyncraticVolSignalNode, dict(estimation_bars=3), "estimation_bars"),
            (IdiosyncraticVolSignalNode, dict(vol_bars=2), "vol_bars"),
            (IdiosyncraticVolSignalNode, dict(estimation_bars=60, vol_bars=61), "vol_bars"),
            (IdiosyncraticVolSignalNode, dict(horizon_bars=0), "horizon_bars"),
        ],
    )
    def test_bad_params_raise(self, make, kwargs, match):
        with pytest.raises(ValueError, match=match):
            make(_small_config(), **kwargs)
