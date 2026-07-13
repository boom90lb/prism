"""Signal contract + ensemble node (SPEC §7.1, R1).

Pins the contract obligations: standardized sqrt(h) score units (I-3),
scale invariance under a global price-level rescale, NaN-= -no-opinion for
short-history names, fail-loud configuration and lifecycle (N7), and the
per-bar-causal ARIMA scoring that replaces the legacy
forecast-from-train-end defect.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from prism.signal import EnsembleNodeConfig, EnsembleSignalNode, build_features

_SYMBOLS = ("AAA", "BBB", "CCC", "DDD")


def _panel(
    n: int = 240, symbols: tuple[str, ...] = _SYMBOLS, seed: int = 7, scale: float = 1.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="America/New_York")
    shocks = rng.normal(0.0002, 0.012, size=(n, len(symbols)))
    rets = shocks.copy()
    rets[1:] += 0.12 * shocks[:-1]  # mild AR(1) so the members have something to find
    close = pd.DataFrame(
        100.0 * scale * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=list(symbols)
    )
    volume = pd.DataFrame(
        rng.uniform(1e5, 4e5, size=(n, len(symbols))), index=idx, columns=list(symbols)
    )
    return close, volume


def _fast_config(**overrides) -> EnsembleNodeConfig:
    base: dict = dict(horizon_bars=3, min_train_rows=60, xgb_estimators=25, oof_splits=3)
    base.update(overrides)
    return EnsembleNodeConfig(**base)


@pytest.fixture(scope="module")
def fitted_node() -> tuple[EnsembleSignalNode, pd.DataFrame, pd.DataFrame]:
    close, volume = _panel()
    node = EnsembleSignalNode(_fast_config())
    node.fit(close.iloc[:200], volume.iloc[:200])
    return node, close, volume


# ---------------------------------------------------------------------------
# Contract basics
# ---------------------------------------------------------------------------


def test_scores_indexed_by_all_columns_and_finite(fitted_node) -> None:
    node, close, volume = fitted_node
    scores = node.score(close.iloc[:220], volume.iloc[:220])
    assert list(scores.index) == list(close.columns)
    assert scores.notna().all()  # every name has 200+ bars of clean history
    assert node.horizon_bars == 3
    assert node.required_history >= 22


def test_scores_use_sqrt_h_sigma_units(fitted_node) -> None:
    node, close, volume = fitted_node
    window = close.iloc[:220]
    scores = node.score(window, volume.iloc[:220])
    symbol = "AAA"
    cfg = node._config
    log_ret = np.log(window[symbol]).diff()
    sigma = max(float(log_ret.iloc[-cfg.vol_window :].std()), cfg.vol_floor)
    state = node._states[symbol]
    xgb_pred = float(state.xgb.predict(build_features(window[symbol], volume.iloc[:220][symbol]).iloc[[-1]])[0])
    arima_pred = state.arima.expected_h(log_ret)
    blend = node._weights["xgboost"] * xgb_pred + node._weights["arima"] * arima_pred
    expected = blend / (sigma * np.sqrt(cfg.horizon_bars))
    assert scores[symbol] == pytest.approx(expected, rel=1e-12)


def test_weights_are_normalized_inverse_oof_mae(fitted_node) -> None:
    node, _, _ = fitted_node
    assert node.weight_basis_ == "inverse_oof_mae"
    assert set(node._weights) == {"xgboost", "arima"}
    assert all(w > 0 for w in node._weights.values())
    assert sum(node._weights.values()) == pytest.approx(1.0)


def test_score_is_deterministic(fitted_node) -> None:
    node, close, volume = fitted_node
    a = node.score(close.iloc[:215], volume.iloc[:215])
    b = node.score(close.iloc[:215], volume.iloc[:215])
    pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------------------
# Scale invariance (the SPEC §7.1 property test)
# ---------------------------------------------------------------------------


def test_scale_invariance_under_price_level_shift() -> None:
    close, volume = _panel(seed=11)
    scaled = close * 1000.0

    node = EnsembleSignalNode(_fast_config()).fit(close.iloc[:200], volume.iloc[:200])
    node_scaled = EnsembleSignalNode(_fast_config()).fit(scaled.iloc[:200], volume.iloc[:200])

    scores = node.score(close.iloc[:225], volume.iloc[:225])
    scores_scaled = node_scaled.score(scaled.iloc[:225], volume.iloc[:225])

    assert scores.notna().equals(scores_scaled.notna())
    np.testing.assert_allclose(
        scores.to_numpy(), scores_scaled.to_numpy(), rtol=1e-4, atol=1e-7
    )


def test_feature_block_is_scale_invariant_and_causal() -> None:
    close, volume = _panel(n=80, seed=3)
    series = close["AAA"]
    feats = build_features(series, volume["AAA"])
    feats_scaled = build_features(series * 1000.0, volume["AAA"])
    np.testing.assert_allclose(
        feats.to_numpy(), feats_scaled.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )
    # Causal: features at bar t are unchanged by anything after t.
    t = 60
    feats_truncated = build_features(series.iloc[: t + 1], volume["AAA"].iloc[: t + 1])
    pd.testing.assert_frame_equal(feats.iloc[: t + 1], feats_truncated)
    # No backfill: leading rows stay NaN until each lookback fills (I-2).
    assert feats["mom_slow"].iloc[:21].isna().all()


# ---------------------------------------------------------------------------
# NaN = no opinion; fail-loud lifecycle (N7)
# ---------------------------------------------------------------------------


def test_short_history_symbol_gets_nan_not_zero(caplog) -> None:
    close, volume = _panel(seed=5)
    short = close.copy()
    short["NEWIPO"] = np.nan
    short.loc[short.index[-45:], "NEWIPO"] = 50.0 + np.arange(45.0)
    vol2 = volume.copy()
    vol2["NEWIPO"] = 1e5

    node = EnsembleSignalNode(_fast_config())
    with caplog.at_level(logging.WARNING):
        node.fit(short.iloc[:200], vol2.iloc[:200])
    assert node.skipped_symbols_ == ["NEWIPO"]
    assert "skipped" in caplog.text

    scores = node.score(short, vol2)
    assert np.isnan(scores["NEWIPO"])
    assert scores.drop("NEWIPO").notna().all()


def test_symbol_unseen_at_fit_scores_nan(fitted_node) -> None:
    node, close, volume = fitted_node
    wider = close.copy()
    wider["ZZZ"] = close["AAA"].to_numpy() * 2.0
    wider_volume = volume.copy()
    wider_volume["ZZZ"] = volume["AAA"].to_numpy()
    scores = node.score(wider, wider_volume)
    assert np.isnan(scores["ZZZ"])


def test_missing_volume_at_score_fails_loud(fitted_node) -> None:
    node, close, _ = fitted_node
    with pytest.raises(ValueError, match="volume panel"):
        node.score(close)  # node was fit with volume features


def test_config_rejects_prophet_unknown_and_duplicates() -> None:
    with pytest.raises(ValueError, match="N8"):
        EnsembleNodeConfig(members=("xgboost", "prophet"))
    with pytest.raises(ValueError, match="Unknown signal member"):
        EnsembleNodeConfig(members=("xgboost", "lightgbm"))
    with pytest.raises(ValueError, match="Duplicate"):
        EnsembleNodeConfig(members=("arima", "arima"))
    with pytest.raises(ValueError, match="members"):
        EnsembleNodeConfig(members=())


def test_score_before_fit_raises() -> None:
    close, _ = _panel(n=60)
    with pytest.raises(RuntimeError, match="before fit"):
        EnsembleSignalNode(_fast_config()).score(close)


def test_fit_with_no_usable_symbol_raises() -> None:
    close, volume = _panel(n=40)  # every name below min_train_rows
    with pytest.raises(ValueError, match="usable training rows"):
        EnsembleSignalNode(_fast_config()).fit(close, volume)


# ---------------------------------------------------------------------------
# Per-bar-causal ARIMA conditioning (the legacy staleness fix)
# ---------------------------------------------------------------------------


def test_score_conditions_on_post_train_data(fitted_node) -> None:
    node, close, volume = fitted_node
    calm = close.iloc[:225].copy()
    crashed = calm.copy()
    # Same decision bar, different recent path for one name: -2%/bar for 10 bars.
    factors = np.exp(-0.02 * np.arange(1, 11))
    crashed.loc[crashed.index[-10:], "BBB"] = calm["BBB"].iloc[-11] * factors

    s_calm = node.score(calm, volume.iloc[:225])
    s_crashed = node.score(crashed, volume.iloc[:225])
    # The node reacts to data after the training window (the legacy member
    # extrapolated from train-end and could not).
    assert abs(s_crashed["BBB"] - s_calm["BBB"]) > 1e-3
    # Untouched names are bit-identical.
    pd.testing.assert_series_equal(s_calm.drop("BBB"), s_crashed.drop("BBB"))


# ---------------------------------------------------------------------------
# Conformal score band
# ---------------------------------------------------------------------------


def test_score_band_brackets_point(fitted_node) -> None:
    node, close, volume = fitted_node
    assert node.conformal_ is not None and node.conformal_.is_fitted
    lower, point, upper = node.score_band(close.iloc[:220], volume.iloc[:220])
    finite = point.notna()
    assert finite.any()
    q = node.conformal_.quantile(node._config.conformal_alpha)
    assert q > 0
    np.testing.assert_allclose((upper - lower)[finite].to_numpy(), 2.0 * q)
    assert (lower[finite] < point[finite]).all()
    assert (point[finite] < upper[finite]).all()


def test_panel_validation_rejects_garbage(fitted_node) -> None:
    node, close, _ = fitted_node
    with pytest.raises(TypeError, match="wide DataFrame"):
        node.score(close["AAA"])  # a Series, not a panel
    shuffled = close.iloc[::-1]
    with pytest.raises(ValueError, match="sorted ascending"):
        node.score(shuffled)
    duplicated = pd.concat([close, close[["AAA"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate"):
        node.score(duplicated)
