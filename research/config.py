"""Configuration for the quarantined research stack (SPEC.md §9).

The ensemble-side half of the old ``prism.config``: member/ensemble/training
dataclasses and the research artifact directories. Production config (data
directories, API keys for the $0 spine, execution/trading cost dataclasses)
stays in ``prism.config``; importing it here also runs its ``load_dotenv()``,
so keys resolve the same way in both halves.

Dropped in the split as dead public surface (zero importers anywhere):
``DEFAULT_MODELS``, ``DEFAULT_ENSEMBLE_CONFIG``, ``DEFAULT_TRADING_CONFIG``.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from prism.config import PROJECT_DIR

# Research artifact directories. runs/ fold artifacts reference MODELS_DIR
# siblings; mlruns/ is the MLflow file-store default.
MODELS_DIR = PROJECT_DIR / "models"
MLRUNS_DIR = PROJECT_DIR / "mlruns"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# API key for the news fetch in research/sentiment_analysis.py.
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# MLflow tracking URI — defaults to a local file store under the project root.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{MLRUNS_DIR}")


@dataclass
class ModelConfig:
    """Configuration for an individual model in the ensemble."""

    name: str
    enabled: bool = True
    weight: float = 1.0


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble model."""

    models: List[ModelConfig]
    weighting_strategy: str = "static"  # "static" or "dynamic"
    # Vol-targeting knobs for the forecast→position mapping. Exposed here
    # so a hyperparameter sweep can vary them as first-class, persisted
    # config rather than poking post-construction attrs.
    target_vol: float = 1.0
    position_cap: float = 1.0

    def __post_init__(self):
        assert self.target_vol > 0, f"target_vol must be > 0; got {self.target_vol}"
        assert self.position_cap > 0, (
            f"position_cap must be > 0; got {self.position_cap}"
        )


@dataclass
class TrainingConfig:
    """Configuration for model training.

    ``train_test_split`` and ``cv_folds`` are retired — train/test
    folding is driven entirely by the WFO knobs (``n_splits``,
    ``purge_horizon``, ``embargo_pct``, ``expanding``) consumed by
    ``PurgedWalkForward`` in research/scripts/training.py and research/scripts/backtest.py.
    """

    symbols: List[str]
    timeframe: str = "1d"
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    prediction_horizon: int = 5
    use_sentiment: bool = False
    # Outer-WFO knobs. purge_horizon=None -> use prediction_horizon.
    n_splits: int = 5
    purge_horizon: Optional[int] = None
    embargo_pct: float = 0.01
    expanding: bool = True

    def __post_init__(self):
        assert self.prediction_horizon > 0, (
            f"prediction_horizon must be > 0; got {self.prediction_horizon}"
        )
        assert self.n_splits >= 2, f"n_splits must be >= 2; got {self.n_splits}"
        assert 0.0 <= self.embargo_pct < 1.0, (
            f"embargo_pct must be in [0, 1); got {self.embargo_pct}"
        )
        if self.purge_horizon is not None:
            assert self.purge_horizon >= 0, (
                f"purge_horizon must be >= 0; got {self.purge_horizon}"
            )

    @property
    def effective_purge_horizon(self) -> int:
        """Purge horizon for the WFO splitter, defaulting to prediction_horizon
        when the explicit override is None — the forward-return label window
        is exactly ``prediction_horizon`` bars wide so that's the right
        default for AFML §7.4-style purging."""
        return self.purge_horizon if self.purge_horizon is not None else self.prediction_horizon


# Single source of truth for ensemble member weights. Scripts must read from
# here rather than hardcoding per-model overrides. The RL policy members
# (lstm_ppo, xlstm_ppo, xlstm_grpo) are not default members; research entry
# points that opt into them fall back to weight 1.0 via `.get(name, 1.0)`.
DEFAULT_MODEL_WEIGHTS = {
    "arima": 1.0,
    "prophet": 1.0,
    "xgboost": 1.0,
}

DEFAULT_TRAINING_CONFIG = TrainingConfig(
    symbols=["AAPL", "MSFT", "GOOG", "AMZN"],
    timeframe="1d",
    start_date="2020-01-01",
    end_date=None,
    prediction_horizon=5,
    use_sentiment=False,
)
