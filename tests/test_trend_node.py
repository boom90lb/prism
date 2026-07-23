"""Contract and property tests for the trend sleeve node + inverse-vol
construct (``TrendSignalNode`` + ``construct_inverse_vol_targets``) —
docs/trend_design.md §2; SPEC §7.1. Uncounted mechanics only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.portfolio import construct_inverse_vol_targets
from prism.signal import TREND_V1_UNIVERSE, Signal, TrendSignalNode

LOOKBACK, SKIP = 120, 10
VOL_BARS = 30


def _panel(
    n_days: int = 340,
    n_assets: int = 8,
    seed: int = 11,
    *,
    drifts: np.ndarray | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n_days, freq="B", tz="America/New_York")
    if drifts is None:
        drifts = np.linspace(-0.0008, 0.0008, n_assets)
    vols = np.linspace(0.005, 0.02, n_assets)
    returns = drifts[None, :] + rng.normal(0.0, 1.0, size=(n_days, n_assets)) * vols[None, :]
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=idx,
        columns=[f"S{i}" for i in range(n_assets)],
    )


def _node(**kwargs: object) -> TrendSignalNode:
    kwargs.setdefault("lookback_bars", LOOKBACK)
    kwargs.setdefault("skip_bars", SKIP)
    return TrendSignalNode(**kwargs)


def _score_frame(close: pd.DataFrame, node: TrendSignalNode | None = None) -> pd.DataFrame:
    node = node or _node()
    # Build a short score history aligned to close for construct tests: score
    # only the last row (node contract); tile for multi-row construct checks.
    last = node.score(close)
    return pd.DataFrame([last.to_numpy()], index=close.index[-1:], columns=close.columns)


class TestContractSurface:
    def test_is_a_signal_with_horizon_and_history(self):
        node = _node(horizon_bars=21)
        assert isinstance(node, Signal)
        assert node.horizon_bars == 21
        assert node.required_history == LOOKBACK + 1

    def test_universe_constant_matches_design_ten(self):
        assert len(TREND_V1_UNIVERSE) == 10
        assert "SPY" in TREND_V1_UNIVERSE and "UUP" in TREND_V1_UNIVERSE

    def test_fit_returns_self_and_score_needs_no_fit(self):
        close = _panel()
        node = _node()
        assert node.fit(close) is node
        scores = _node().score(close)
        assert isinstance(scores, pd.Series)
        assert scores.index.equals(close.columns)

    def test_volume_optional(self):
        close = _panel()
        a = _node().score(close, None)
        b = _node().score(close, pd.DataFrame(1.0, index=close.index, columns=close.columns))
        pd.testing.assert_series_equal(a, b)

    def test_short_panel_raises(self):
        close = _panel(n_days=40)
        with pytest.raises(ValueError, match="rows"):
            _node().score(close)

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
            TrendSignalNode(**kwargs)


class TestScoreSemantics:
    def test_scale_invariance(self):
        close = _panel()
        node = _node()
        base = node.score(close)
        rescaled = node.score(close * 13.0)
        pd.testing.assert_series_equal(base, rescaled)

    def test_causality_appending_future_bars_changes_nothing(self):
        close = _panel(n_days=360)
        node = _node()
        cut = 300
        at_cut = node.score(close.iloc[:cut])
        tail = slice(cut - node.required_history, cut)
        from_longer = node.score(close.iloc[tail])
        pd.testing.assert_series_equal(at_cut, from_longer)

    def test_sign_tracks_trailing_return(self):
        """Monotone growth rates → ordered scores (sign of 12−1)."""
        n_days = 200
        rates = np.array([0.9990, 0.9995, 1.0000, 1.0005, 1.0010])
        idx = pd.date_range("2020-01-02", periods=n_days, freq="B", tz="America/New_York")
        close = pd.DataFrame(
            100.0 * rates[None, :] ** np.arange(n_days)[:, None],
            index=idx,
            columns=[f"S{i}" for i in range(len(rates))],
        )
        scores = _node().score(close).dropna()
        assert list(scores.sort_values().index) == list(close.columns)

    def test_non_positive_base_price_is_nan(self):
        close = _panel()
        close = close.copy()
        close.iloc[-1 - LOOKBACK, close.columns.get_loc("S2")] = -1.0
        assert np.isnan(_node().score(close)["S2"])

    def test_formula_matches_hand_calculation(self):
        close = _panel(n_days=200, n_assets=3, seed=0)
        node = _node(lookback_bars=50, skip_bars=5)
        got = node.score(close)
        base = close.iloc[-1 - 50]
        recent = close.iloc[-1 - 5]
        expected = recent / base - 1.0
        pd.testing.assert_series_equal(got, expected.astype(float), check_names=False)


class TestInverseVolConstruct:
    def test_gross_at_cap_when_all_signed(self):
        close = _panel(n_days=200)
        # Force all names to a non-zero score on the last bar.
        scores = pd.DataFrame(
            np.where(np.arange(close.shape[1]) % 2 == 0, 0.1, -0.1)[None, :],
            index=close.index[-1:],
            columns=close.columns,
        )
        close_tail = close.iloc[-(VOL_BARS + 5) :]
        # Align scores index to last row of the vol window panel.
        scores = scores.reindex(close_tail.index)
        scores.iloc[:-1] = np.nan
        scores.iloc[-1] = np.where(np.arange(close.shape[1]) % 2 == 0, 0.1, -0.1)
        book = construct_inverse_vol_targets(
            scores, close_tail, vol_ewma_bars=VOL_BARS, max_gross=1.0
        ).iloc[-1]
        assert book.abs().sum() == pytest.approx(1.0, rel=1e-6)

    def test_higher_vol_gets_smaller_abs_weight(self):
        n_days = 200
        idx = pd.date_range("2020-01-02", periods=n_days, freq="B", tz="America/New_York")
        rng = np.random.default_rng(0)
        low = 0.001 * rng.normal(size=n_days)
        high = 0.05 * rng.normal(size=n_days)
        close = pd.DataFrame(
            {
                "LOW": 100.0 * np.exp(np.cumsum(low)),
                "HIGH": 100.0 * np.exp(np.cumsum(high)),
            },
            index=idx,
        )
        scores = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        scores.iloc[-1] = 1.0  # both long
        book = construct_inverse_vol_targets(
            scores, close, vol_ewma_bars=VOL_BARS, max_gross=1.0
        ).iloc[-1]
        assert book["LOW"] > book["HIGH"] > 0.0

    def test_nan_score_is_flat_not_nan(self):
        close = _panel(n_days=200, n_assets=4)
        scores = pd.DataFrame(0.2, index=close.index, columns=close.columns)
        scores["S1"] = np.nan
        book = construct_inverse_vol_targets(
            scores, close, vol_ewma_bars=VOL_BARS, max_gross=1.0
        ).iloc[-1]
        assert book["S1"] == 0.0
        assert book.notna().all()

    def test_zero_score_is_flat(self):
        close = _panel(n_days=200, n_assets=3)
        scores = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        scores.iloc[-1, 0] = 0.1
        scores.iloc[-1, 1] = 0.0
        scores.iloc[-1, 2] = -0.1
        book = construct_inverse_vol_targets(
            scores, close, vol_ewma_bars=VOL_BARS, max_gross=1.0
        ).iloc[-1]
        assert book.iloc[1] == 0.0

    def test_column_mismatch_raises(self):
        close = _panel(n_days=100, n_assets=3)
        scores = pd.DataFrame(0.1, index=close.index, columns=["A", "B", "C"])
        with pytest.raises(ValueError, match="columns"):
            construct_inverse_vol_targets(scores, close, vol_ewma_bars=VOL_BARS)

    def test_bad_vol_window_raises(self):
        close = _panel(n_days=100, n_assets=3)
        scores = pd.DataFrame(0.1, index=close.index, columns=close.columns)
        with pytest.raises(ValueError, match="vol_ewma_bars"):
            construct_inverse_vol_targets(scores, close, vol_ewma_bars=1)

    def test_end_to_end_node_then_construct(self):
        close = _panel(n_days=300, n_assets=6, seed=2)
        node = _node(lookback_bars=LOOKBACK, skip_bars=SKIP)
        # Score every bar with expanding history (causal last-row semantics).
        rows = []
        for t in range(node.required_history, len(close)):
            rows.append(node.score(close.iloc[: t + 1]).to_numpy())
        scores = pd.DataFrame(
            rows,
            index=close.index[node.required_history :],
            columns=close.columns,
        )
        close_aligned = close.loc[scores.index]
        # Need vol warmup: drop first VOL_BARS of the aligned window.
        book = construct_inverse_vol_targets(
            scores.iloc[VOL_BARS:],
            close_aligned.iloc[VOL_BARS:],
            vol_ewma_bars=VOL_BARS,
            max_gross=1.0,
        )
        last = book.iloc[-1]
        assert last.notna().all()
        assert last.abs().sum() == pytest.approx(1.0, rel=1e-5) or last.abs().sum() == 0.0
        # Signs of non-zero weights match score signs.
        s_last = scores.iloc[-1]
        for name in close.columns:
            if last[name] != 0.0 and np.isfinite(s_last[name]) and s_last[name] != 0.0:
                assert np.sign(last[name]) == np.sign(s_last[name])
