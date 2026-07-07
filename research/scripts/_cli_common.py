"""Shared CLI plumbing for the research batch scripts.

backtest.py, sweep.py, and rl_seed_eval.py used to re-implement the same
execution/trading cost flags, the args→config builders, and the per-symbol
fetch → build_features → usable-frame loop. Extracted here verbatim so the
three CLIs stay flag- and behavior-identical; nothing in this module changes
a default or a choice list.

``build_features`` and ``clean_data_for_training`` live here (not in
training.py) so the shared plumbing never imports from an entry-point
script; training.py consumes them like every other CLI.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from prism.config import ExecutionConfig, TradingConfig, TrainingConfig
from prism.features import FeatureEngineer, forward_return_column, is_label_column
from prism.io.loader import DataLoader
from prism.sentiment_analysis import SentimentAnalyzer

logger = logging.getLogger(__name__)


def clean_data_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative cleanup applied causally on the full enhanced frame.

    * Inf -> NaN (row-wise, causal).
    * Drop columns with >30% NaN (column-set decision; barely informative).
    * Forward-fill remaining feature NaNs (past -> future, causal).
    * Final feature fill-with-0 to handle warmup-period leading NaNs.

    Label columns keep their NaNs: the final ``horizon`` rows have no observed
    forward outcome and must be excluded by the downstream ``dropna(target)``.

    Outlier clipping is intentionally NOT done here — that lives in
    ``FeatureEngineer.transform_features`` which uses train-only bounds
    (audit b). Doing 5×IQR clipping on the full frame here
    would re-introduce a row-level future-leak via the IQR computation.
    """
    # The whole pipeline (point-in-time sentiment join, date
    # filtering) assumes an ET-localized index. Fail loud if feature building
    # dropped the tz rather than letting naive timestamps leak downstream.
    if isinstance(df.index, pd.DatetimeIndex):
        assert df.index.tz is not None, "training frame index lost its timezone"
    result = df.replace([np.inf, -np.inf], np.nan)
    label_cols = [col for col in result.columns if is_label_column(col)]
    labels = result[label_cols].copy()
    feature_cols = [col for col in result.columns if col not in label_cols]

    nan_pct = result[feature_cols].isna().mean()
    high_nan_cols = nan_pct[nan_pct > 0.3].index.tolist()
    if high_nan_cols:
        logger.warning(f"Dropping {len(high_nan_cols)} cols with >30% NaN")
        result = result.drop(columns=high_nan_cols)
    result = result.ffill().fillna(0)
    for col in label_cols:
        result[col] = labels[col]
    return result


def build_features(
    raw_df: pd.DataFrame,
    symbol: str,
    feature_engineer: FeatureEngineer,
    sentiment_analyzer: Optional[SentimentAnalyzer],
    horizon: int,
) -> pd.DataFrame:
    """Causally build the full enhanced frame for one symbol.

    Technical indicators (rolling/ewm/pct_change) are causal so they may
    be computed once on the full series. Sentiment join is point-in-time
    (the B9 fix). FE scaling + outlier clipping happens per fold inside
    ``train_symbol_wfo`` so this frame is intentionally UNSCALED.
    """
    df = feature_engineer.create_features(raw_df)
    df = feature_engineer.create_lagged_features(df, [1, 2, 5, 10])
    df = feature_engineer.create_target_variable(df, "close", horizon)

    if sentiment_analyzer is not None:
        sentiment = sentiment_analyzer.create_sentiment_features(
            symbol, pd.DatetimeIndex(df.index)  # type: ignore
        )
        if not sentiment.empty:
            df = df.join(sentiment)

    df = clean_data_for_training(df)
    return df


def fetch_training_frames(data_loader: DataLoader, config: TrainingConfig) -> dict[str, pd.DataFrame]:
    """Fetch full per-symbol OHLCV frames for WFO training.

    Train/test splitting is handled by the WFO outer loop via
    ``PurgedWalkForward``, so this returns one frame per symbol with no
    preemptive split. Symbols that come back empty are omitted. Lived on
    ``DataLoader`` until the io/ fold; it is research-CLI plumbing, so it
    lives here now.
    """
    all_data: dict[str, pd.DataFrame] = {}
    for symbol in config.symbols:
        logger.info(f"Fetching data for {symbol}")
        df = data_loader.fetch_historical_data(
            symbol=symbol,
            interval=config.timeframe,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        if df.empty:
            logger.warning(f"No data fetched for {symbol}, skipping")
            continue
        all_data[symbol] = df
    return all_data


def add_execution_args(
    parser: argparse.ArgumentParser,
    *,
    include_adv: bool = False,
    include_signal_threshold: bool = False,
) -> None:
    """Register the execution/trading cost flags shared by the three CLIs.

    ``include_adv`` adds the ADV-participation impact trio (backtest only);
    ``include_signal_threshold`` appends ``--signal_threshold`` here
    (rl_seed_eval; backtest registers it in its own strategy-flag group).
    """
    parser.add_argument("--initial_capital", type=float, default=10000.0)
    parser.add_argument("--position_size", type=float, default=0.1)
    parser.add_argument("--commission_bps", type=float, default=1.0)
    parser.add_argument("--spread_bps", type=float, default=1.0)
    parser.add_argument("--slippage_coeff", type=float, default=10.0)
    if include_adv:
        parser.add_argument("--adv_impact_coeff", type=float, default=0.0)
        parser.add_argument("--adv_impact_model", type=str, default="linear", choices=["linear", "sqrt"])
        parser.add_argument("--adv_floor_dollars", type=float, default=100000.0)
    parser.add_argument("--borrow_bps_annual", type=float, default=50.0)
    parser.add_argument(
        "--order_type", type=str, default="MOO", choices=["MOO", "MOC"],
    )
    if include_signal_threshold:
        parser.add_argument("--signal_threshold", type=float, default=0.1)


def build_execution_and_trading_configs(args) -> Tuple[ExecutionConfig, TradingConfig]:
    """Build the (ExecutionConfig, TradingConfig) pair from a parsed-args or
    SimpleNamespace object. getattr-defensive so callers that never register a
    flag (e.g. the sweep's trial namespaces) fall back to library defaults."""
    execution_config = ExecutionConfig(
        spread_bps=args.spread_bps,
        slippage_coeff=args.slippage_coeff,
        commission_bps=args.commission_bps,
        borrow_rate_bps_annual=args.borrow_bps_annual,
        adv_impact_coeff=getattr(args, "adv_impact_coeff", 0.0),
        adv_impact_model=getattr(args, "adv_impact_model", "linear"),
        adv_floor_dollars=getattr(args, "adv_floor_dollars", 100000.0),
        default_order_type=args.order_type,
    )
    trading_config = TradingConfig(
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        # The sweep varies this per trial; absent on a plain backtest → 0.1.
        signal_threshold=getattr(args, "signal_threshold", 0.1),
        execution_style=getattr(args, "execution_style", "legacy_orders"),
        rebalance_band_weight=getattr(args, "rebalance_band_weight", 0.0),
        rebalance_cost_multiplier=getattr(args, "rebalance_cost_multiplier", 1.0),
        max_gross_exposure=getattr(args, "max_gross_exposure", 1.0),
        execution=execution_config,
    )
    return execution_config, trading_config


def build_usable_symbol_frame(
    *,
    data_loader: DataLoader,
    feature_engineer: FeatureEngineer,
    symbol: str,
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    horizon: int,
    sentiment_analyzer=None,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """The per-symbol fetch → build_features → usable-frame core.

    Returns ``(raw_df, full_df, usable)`` where ``usable`` drops rows without
    the horizon's forward-return target, or ``None`` (logged) when the symbol
    has no data or the feature frame lacks the target column — callers
    skip/drop the symbol on None. ``raw_df`` is returned because the
    target-weight path prices fills off the raw open/volume columns.
    """
    raw_df = data_loader.fetch_historical_data(
        symbol=symbol,
        interval=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    if raw_df.empty:
        logger.warning(f"No data for {symbol}; skipping")
        return None
    full_df = build_features(
        raw_df=raw_df,
        symbol=symbol,
        feature_engineer=feature_engineer,
        sentiment_analyzer=sentiment_analyzer,
        horizon=horizon,
    )
    target_col = forward_return_column(horizon)
    if target_col not in full_df.columns:
        logger.error(
            f"{target_col} missing from features for {symbol}; "
            "training config mismatch"
        )
        return None
    return raw_df, full_df, full_df.dropna(subset=[target_col])
