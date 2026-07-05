"""Shared CLI plumbing for the research batch scripts.

backtest.py, sweep.py, and rl_seed_eval.py used to re-implement the same
execution/trading cost flags, the args→config builders, and the per-symbol
fetch → build_features → usable-frame loop. Extracted here verbatim so the
three CLIs stay flag- and behavior-identical; nothing in this module changes
a default or a choice list.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple

import pandas as pd

from prism.config import ExecutionConfig, TradingConfig
from prism.data_loader import DataLoader
from prism.features import FeatureEngineer, forward_return_column
from research.scripts.training import build_features

logger = logging.getLogger(__name__)


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
