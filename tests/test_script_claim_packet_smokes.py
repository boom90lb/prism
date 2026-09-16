"""Offline smoke tests for script-level research claim packets."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prism.execution.target_weights import backtest_target_weights
from prism.validation.trials import validate_claim_packet_dir
from research.features import forward_return_column
from research.models.base import BaseModel

# The smokes drive research/scripts CLIs (mlflow at module scope) in-process;
# mlflow is a reliable sentinel for the whole [research] extra here.
pytest.importorskip("mlflow", reason="needs the [research] extra")
pytestmark = pytest.mark.research

# The per-symbol fetch→features loop lives in _cli_common, so that is the
# monkeypatch seam for build_features.
import research.scripts._cli_common as cli_common  # noqa: E402


class _DummyRun:
    def __enter__(self):
        return SimpleNamespace(info=SimpleNamespace(run_id="dummy"))

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_mlflow(monkeypatch, module) -> None:
    monkeypatch.setattr(module, "init_mlflow", lambda *a, **k: "0", raising=False)
    monkeypatch.setattr(module.mlflow, "start_run", lambda *a, **k: _DummyRun(), raising=False)
    monkeypatch.setattr(module, "log_metrics_safe", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(module, "log_params_safe", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(module, "log_artifact_dir", lambda *a, **k: None, raising=False)


def _bars(n: int = 64) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-02", periods=n, tz="America/New_York")
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def _feature_frame(raw_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = raw_df.copy()
    target_col = f"target_{horizon}"
    out[target_col] = out["close"].shift(-horizon)
    out[forward_return_column(horizon)] = out[target_col] / out["close"] - 1.0
    return out


class _ConstModel(BaseModel):
    def __init__(self, value: float = 0.6):
        super().__init__(name="const", target_column="close", horizon=1)
        self.value = value
        self.is_fitted = True

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)

    def save(self, directory: Path) -> Path:
        return Path(directory)

    def load(self, model_path: Path):
        return self

    def evaluate(self, X_test, y_test):
        return {}


# `test_stat_arb_wfo_main_emits_valid_claim_packet` was removed at stage 3b with
# `research/scripts/stat_arb_wfo.py`, which retired together with the fold
# geometry it drove (`research/arbitrage/walk_forward.py`).
#
# `test_stat_arb_residual_wfo_main_emits_valid_claim_packet` was removed at
# stage 2 with `research/scripts/stat_arb_residual_wfo.py`, whose recorded case
# now reproduces through `prism.evaluate.walk_forward_evaluate`
# (`tests/test_demotion_b1_reproduction.py`). The evaluator writes a
# `ResultRecord` and a hash-chained trial row rather than a claim packet, so
# there is no claim-packet smoke left to run for that path.


def test_backtest_main_target_weights_emits_root_claim_packet(monkeypatch, tmp_path: Path) -> None:
    import research.scripts.backtest as mod

    _patch_mlflow(monkeypatch, mod)
    training_run = tmp_path / "training_run"
    fold_dir = training_run / "AAA" / "fold_0"
    fold_dir.mkdir(parents=True)
    raw = _bars()
    config = {
        "prediction_horizon": 1,
        "timeframe": "1d",
        "start_date": "2026-01-02",
        "end_date": None,
        "use_sentiment": False,
        "models": ["xgboost"],
        "n_splits": 1,
        "purge_horizon": 1,
        "embargo_pct": 0.0,
        "expanding": True,
    }

    class _Loader:
        def fetch_historical_data(self, *args, **kwargs):
            return raw

        def fetch_dividends(self, *args, **kwargs):
            return pd.Series([1.0], index=[raw.index[5]], name="amount")

    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(mod, "DataLoader", _Loader)
    monkeypatch.setattr(mod, "load_training_run", lambda _path: (config, {"AAA": [fold_dir]}))
    monkeypatch.setattr(
        cli_common,
        "build_features",
        lambda raw_df, symbol, feature_engineer, sentiment_analyzer, horizon: _feature_frame(raw_df, horizon),
    )
    monkeypatch.setattr(mod, "load_fold_ensemble", lambda *a, **k: _ConstModel(0.7))
    monkeypatch.setattr(
        mod,
        "_fold_date_range",
        lambda _fold, usable, **kwargs: (usable.index[0], usable.index[-1]),
    )
    monkeypatch.setattr(mod, "_fold_model_data", lambda _fold, symbol, full_df: full_df)
    captured = {}
    real_backtest_target_weights = mod.backtest_target_weights

    def _capture_backtest_target_weights(*args, **kwargs):
        captured["dollar_volume"] = kwargs.get("dollar_volume")
        captured["execution"] = kwargs.get("execution")
        return real_backtest_target_weights(*args, **kwargs)

    monkeypatch.setattr(mod, "backtest_target_weights", _capture_backtest_target_weights)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest.py",
            "--training_run",
            str(training_run),
            "--symbols",
            "AAA",
            "--adv_impact_coeff",
            "25",
            "--adv_floor_dollars",
            "1000",
        ],
    )

    mod.main()

    out_dirs = sorted(tmp_path.glob("wfo_backtest_*"))
    assert out_dirs
    packet = validate_claim_packet_dir(out_dirs[-1])
    assert packet["strategy"] == "ensemble_wfo_target_weights"
    assert (out_dirs[-1] / "target_weights.csv").exists()
    assert (out_dirs[-1] / "fill_weights.csv").exists()
    assert captured["execution"].adv_impact_coeff == 25
    assert captured["execution"].adv_floor_dollars == 1000
    assert captured["dollar_volume"] is not None
    pd.testing.assert_series_equal(
        captured["dollar_volume"]["AAA"],
        (raw["close"] * raw["volume"]).reindex(captured["dollar_volume"].index).rename("AAA"),
    )


def test_backtest_main_shared_construct_opt_in_emits_claim_packet(monkeypatch, tmp_path: Path) -> None:
    import research.scripts.backtest as mod

    _patch_mlflow(monkeypatch, mod)
    training_run = tmp_path / "training_run"
    fold_dir = training_run / "AAA" / "fold_0"
    fold_dir.mkdir(parents=True)
    raw = _bars()
    config = {
        "prediction_horizon": 1,
        "timeframe": "1d",
        "start_date": "2026-01-02",
        "end_date": None,
        "use_sentiment": False,
        "models": ["xgboost"],
        "n_splits": 1,
        "purge_horizon": 1,
        "embargo_pct": 0.0,
        "expanding": True,
    }

    class _Loader:
        def fetch_historical_data(self, *args, **kwargs):
            return raw

        def fetch_dividends(self, *args, **kwargs):
            return pd.Series(dtype=float)

    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(mod, "DataLoader", _Loader)
    monkeypatch.setattr(mod, "load_training_run", lambda _path: (config, {"AAA": [fold_dir]}))
    monkeypatch.setattr(
        cli_common,
        "build_features",
        lambda raw_df, symbol, feature_engineer, sentiment_analyzer, horizon: _feature_frame(raw_df, horizon),
    )
    monkeypatch.setattr(mod, "load_fold_ensemble", lambda *a, **k: _ConstModel(0.7))
    monkeypatch.setattr(
        mod,
        "_fold_date_range",
        lambda _fold, usable, **kwargs: (usable.index[0], usable.index[-1]),
    )
    monkeypatch.setattr(mod, "_fold_model_data", lambda _fold, symbol, full_df: full_df)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest.py",
            "--training_run",
            str(training_run),
            "--symbols",
            "AAA",
            "--construct_style",
            "shared",
        ],
    )

    mod.main()

    out_dirs = sorted(tmp_path.glob("wfo_backtest_*"))
    assert out_dirs
    packet = validate_claim_packet_dir(out_dirs[-1])
    assert packet["strategy"] == "ensemble_wfo_target_weights"
    assert packet["config"]["construct_style"] == "shared"
    assert packet["claim_tier"] == "no_claim"
    assert (out_dirs[-1] / "target_weights.csv").exists()


def test_rl_seed_eval_main_emits_member_claim_packet(monkeypatch, tmp_path: Path) -> None:
    import research.scripts.rl_seed_eval as mod

    _patch_mlflow(monkeypatch, mod)
    training_run = tmp_path / "training_run"
    fold_dir = training_run / "AAA" / "fold_0"
    fold_dir.mkdir(parents=True)
    raw = _bars()
    config = {
        "prediction_horizon": 1,
        "timeframe": "1d",
        "start_date": "2026-01-02",
        "end_date": None,
        "use_sentiment": False,
    }

    class _Loader:
        def fetch_historical_data(self, *args, **kwargs):
            return raw

        def fetch_dividends(self, *args, **kwargs):
            return pd.Series(dtype=float)

    def _seed_returns(member, seed, **kwargs):
        idx = pd.bdate_range("2026-03-02", periods=30)
        values = 0.001 + seed * 0.0001 + np.linspace(-0.0005, 0.0005, len(idx))
        return pd.Series(values, index=idx, name=member)

    monkeypatch.setattr(mod, "DataLoader", _Loader)
    monkeypatch.setattr(mod, "load_training_run", lambda _path: (config, {"AAA": [fold_dir]}))
    monkeypatch.setattr(
        cli_common,
        "build_features",
        lambda raw_df, symbol, feature_engineer, sentiment_analyzer, horizon: _feature_frame(raw_df, horizon),
    )
    monkeypatch.setattr(mod, "eval_member_seed", _seed_returns)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rl_seed_eval.py",
            "--training_run",
            str(training_run),
            "--members",
            "lstm_ppo",
            "--seeds",
            "0,1",
            "--output_dir",
            str(out_dir),
        ],
    )

    mod.main()

    packet = validate_claim_packet_dir(out_dir / "lstm_ppo")
    assert packet["strategy"] == "rl_member_seed_robustness"


def test_rl_seed_eval_rejects_sentiment_before_data_loading(monkeypatch, tmp_path: Path) -> None:
    import research.scripts.rl_seed_eval as mod

    training_run = tmp_path / "training_run"
    fold_dir = training_run / "AAA" / "fold_0"
    fold_dir.mkdir(parents=True)
    config = {
        "prediction_horizon": 1,
        "timeframe": "1d",
        "start_date": "2026-01-02",
        "end_date": None,
        "use_sentiment": True,
    }

    class _Loader:
        def __init__(self, *args, **kwargs):
            raise AssertionError("DataLoader should not be constructed")

    monkeypatch.setattr(mod, "DataLoader", _Loader)
    monkeypatch.setattr(mod, "load_training_run", lambda _path: (config, {"AAA": [fold_dir]}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rl_seed_eval.py",
            "--training_run",
            str(training_run),
            "--members",
            "lstm_ppo",
            "--seeds",
            "0",
            "--output_dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(ValueError, match="sentiment is disabled for rl_seed_eval"):
        mod.main()
