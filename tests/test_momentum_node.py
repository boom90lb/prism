"""Contract, property, and research-parity tests for the momentum book
(``MomentumSignalNode`` + ``construct_decile_neutral``) — the ratified B1
candidate's node and its decile long/short construction (SPEC §7.1;
docs/demotion_design.md §2b)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.portfolio import construct_decile_neutral
from prism.residual.factors import ResidualStatArbConfig
from prism.signal import MomentumSignalNode, Signal

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
    """Distinct per-name drifts, so 12-1 momentum genuinely separates names."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n_days, freq="B", tz="America/New_York")
    drifts = np.linspace(-0.0008, 0.0008, n_assets)
    returns = drifts[None, :] + rng.normal(0.0, 0.007, size=(n_days, n_assets))
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=idx,
        columns=[f"S{i}" for i in range(n_assets)],
    )
    volumes = pd.DataFrame(2_000_000.0, index=idx, columns=closes.columns)
    return closes, volumes


def _node(**kwargs: object) -> MomentumSignalNode:
    kwargs.setdefault("lookback_bars", LOOKBACK)
    kwargs.setdefault("skip_bars", SKIP)
    return MomentumSignalNode(_small_config(), **kwargs)


class TestContractSurface:
    def test_is_a_signal_with_horizon_and_history(self):
        node = _node(horizon_bars=21)
        assert isinstance(node, Signal)
        assert node.horizon_bars == 21
        assert node.required_history == max(LOOKBACK, 60) + 1  # lookback binds here

    def test_required_history_takes_the_larger_of_lookback_and_corr_window(self):
        short_lb = MomentumSignalNode(_small_config(corr_window=200), lookback_bars=40, skip_bars=5)
        assert short_lb.required_history == 200 + 1  # corr_window binds

    def test_fit_returns_self_and_score_needs_no_fit(self):
        closes, volumes = _panel()
        node = _node()
        assert node.fit(closes, volumes) is node
        scores = _node().score(closes, volumes)  # stateless: no prior fit
        assert isinstance(scores, pd.Series)
        assert scores.index.equals(closes.columns)

    def test_missing_volume_raises(self):
        closes, _ = _panel()
        with pytest.raises(ValueError, match="volume"):
            _node().score(closes, None)

    def test_short_panel_raises(self):
        closes, volumes = _panel(n_days=40)
        with pytest.raises(ValueError, match="rows"):
            _node().score(closes, volumes)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(lookback_bars=1), "lookback_bars"),
            (dict(skip_bars=-1), "skip_bars"),
            (dict(lookback_bars=10, skip_bars=10), "must be <"),
            (dict(horizon_bars=0), "horizon_bars"),
        ],
    )
    def test_bad_params_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MomentumSignalNode(_small_config(), **kwargs)


class TestScoreSemantics:
    def test_scale_invariance(self):
        """A global price rescale leaves a return ratio unchanged (contract)."""
        closes, volumes = _panel()
        node = _node()
        base = node.score(closes, volumes)
        rescaled = node.score(closes * 13.0, volumes)
        pd.testing.assert_series_equal(base, rescaled)

    def test_causality_appending_future_bars_changes_nothing(self):
        closes, volumes = _panel(n_days=360)
        node = _node()
        cut = 300
        at_cut = node.score(closes.iloc[:cut], volumes.iloc[:cut])
        tail = slice(cut - node.required_history, cut)
        from_longer = node.score(closes.iloc[tail], volumes.iloc[tail])
        pd.testing.assert_series_equal(at_cut, from_longer)

    def test_rank_increases_with_trailing_return(self):
        """Deterministic monotone panel: higher per-bar growth => higher score."""
        n_days = 200
        rates = np.array([0.9990, 0.9995, 1.0000, 1.0005, 1.0010])
        idx = pd.date_range("2020-01-02", periods=n_days, freq="B", tz="America/New_York")
        closes = pd.DataFrame(
            100.0 * rates[None, :] ** np.arange(n_days)[:, None],
            index=idx,
            columns=[f"S{i}" for i in range(len(rates))],
        )
        volumes = pd.DataFrame(2_000_000.0, index=idx, columns=closes.columns)
        scores = _node().score(closes, volumes).dropna()
        assert list(scores.sort_values().index) == list(closes.columns)  # ascending in rate

    def test_ineligible_name_has_no_opinion(self):
        closes, volumes = _panel()
        volumes = volumes.copy()
        volumes["S0"] = 0.001  # dollar volume below the floor -> ineligible
        scores = _node().score(closes, volumes)
        assert np.isnan(scores["S0"])

    def test_membership_mask_gates_scores(self):
        closes, volumes = _panel()
        mask = pd.DataFrame(True, index=closes.index, columns=closes.columns)
        mask["S1"] = False
        node = MomentumSignalNode(_small_config(), lookback_bars=LOOKBACK, skip_bars=SKIP, membership_mask=mask)
        assert np.isnan(node.score(closes, volumes)["S1"])

    def test_non_positive_base_price_is_nan(self):
        closes, volumes = _panel()
        closes = closes.copy()
        closes.iloc[-1 - LOOKBACK, closes.columns.get_loc("S2")] = -1.0  # bad base price
        assert np.isnan(_node().score(closes, volumes)["S2"])


class TestDecileNeutralConstruct:
    def _score_frame(self, seed: int = 3, n: int = 40, n_nan: int = 3) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        row = rng.normal(size=n)
        row[rng.choice(n, n_nan, replace=False)] = np.nan
        return pd.DataFrame([row], columns=[f"S{i}" for i in range(n)])

    def test_gross_one_and_balanced(self):
        book = construct_decile_neutral(self._score_frame(), decile=0.10).iloc[0]
        assert book.abs().sum() == pytest.approx(1.0)
        assert book.sum() == pytest.approx(0.0)  # market-neutral by balanced legs
        assert int((book > 0).sum()) == int((book < 0).sum())  # equal names per side

    def test_thin_cross_section_is_flat(self):
        thin = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], columns=list("ABCDE"))
        book = construct_decile_neutral(thin, decile=0.10).iloc[0]  # floor(5*0.1)=0
        assert (book == 0.0).all()

    def test_rank_based_not_magnitude(self):
        frame = self._score_frame()
        base = construct_decile_neutral(frame, decile=0.10)
        monotone = construct_decile_neutral(frame * 3.0 + 7.0, decile=0.10)  # order-preserving
        pd.testing.assert_frame_equal(base, monotone)

    def test_non_decile_names_are_flat_not_nan(self):
        book = construct_decile_neutral(self._score_frame(), decile=0.10).iloc[0]
        assert book.notna().all()
        assert (book == 0.0).any()  # the mid names are explicit flat

    def test_bad_decile_raises(self):
        for bad in (0.0, -0.1, 0.6):
            with pytest.raises(ValueError, match="decile"):
                construct_decile_neutral(self._score_frame(), decile=bad)


@pytest.mark.research
class TestResearchParity:
    """The production node/construct must reproduce the research WFO reference
    bit-for-bit (promoted, never imported by the live path — N8)."""

    def test_momentum_score_matches_research(self):
        from research.arbitrage.residual_walk_forward import _momentum_scores
        from research.arbitrage.walk_forward import StatArbWalkForwardConfig

        closes, volumes = _panel()
        config = _small_config()
        node = MomentumSignalNode(config, lookback_bars=LOOKBACK, skip_bars=SKIP)
        got = node.score(closes, volumes)

        wf = StatArbWalkForwardConfig(mom_lookback_bars=LOOKBACK, mom_skip_bars=SKIP)
        ref = _momentum_scores(closes, volumes, config, wf, None)
        ref_last = pd.Series(ref[-1], index=closes.columns, dtype=float)
        pd.testing.assert_series_equal(got, ref_last, check_names=False)

    def test_decile_construct_matches_research(self):
        from research.arbitrage.residual_walk_forward import _momentum_row

        rng = np.random.default_rng(5)
        row = rng.normal(size=50)
        row[[4, 11, 23]] = np.nan
        frame = pd.DataFrame([row], columns=[f"S{i}" for i in range(50)])
        got = construct_decile_neutral(frame, decile=0.10, max_symbol_abs_weight=1.0).iloc[0].to_numpy()
        ref = _momentum_row(np.where(np.isfinite(row), row, np.nan), 0.10)
        np.testing.assert_allclose(got, ref)
