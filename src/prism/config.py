"""Production configuration: directories, spine API keys, cost dataclasses.

The ensemble-side half of the pre-R1 config (member/ensemble/training
dataclasses, research artifact dirs, the MLflow URI, the Polygon news key)
lives in ``research/config.py`` with its only consumers (SPEC §9).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import dotenv

dotenv.load_dotenv()

# Project directories. This file lives at src/prism/config.py under src-layout,
# so the repo root is three levels up.
PROJECT_DIR = Path(__file__).parents[2]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"

for directory in [DATA_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# API key for the Twelve Data spine (prism.io.loader).
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")


@dataclass
class ExecutionConfig:
    """Execution model parameters: fills, costs, borrow.

    bps = basis points (1 bp = 0.01%). All cost figures one-way unless noted.
    """

    # Half of the quoted bid-ask spread. One-way cost on every fill.
    spread_bps: float = 1.0
    # Linear price-impact coefficient: extra slip_bps = slippage_coeff * |notional| / portfolio_value.
    # Coeff of 10 means a 100%-of-portfolio trade incurs +10 bps on top of half-spread.
    slippage_coeff: float = 10.0
    # Commission charged as bps of traded notional (one-way).
    commission_bps: float = 1.0
    # Annualized borrow rate charged on |short notional|, accrued daily over a 252-day year.
    borrow_rate_bps_annual: float = 50.0
    # "MOO" (market-on-open, fills at next bar's open) or "MOC" (next bar's close).
    # MOO is the rigorous default; close-of-bar t signal -> fill at open of t+1.
    default_order_type: str = "MOO"
    # Trading days per year used for borrow accrual.
    trading_days_per_year: int = 252
    # Capacity-aware impact (opt-in). When adv_impact_coeff > 0 AND a dollar_volume
    # panel is supplied to backtest_target_weights, add impact in bps =
    #   adv_impact_coeff * participation
    # or, for adv_impact_model="sqrt":
    #   adv_impact_coeff * sqrt(participation)
    # where participation = |trade_dollars_i| / max(adv_dollars_i, adv_floor_dollars)
    # per name, on top of the portfolio-relative slippage_coeff term. Default 0 ->
    # exact current behavior.
    adv_impact_coeff: float = 0.0
    adv_impact_model: str = "linear"
    adv_floor_dollars: float = 1.0e5

    def __post_init__(self):
        assert self.spread_bps >= 0, f"spread_bps must be >= 0; got {self.spread_bps}"
        assert self.slippage_coeff >= 0, (
            f"slippage_coeff must be >= 0; got {self.slippage_coeff}"
        )
        assert self.commission_bps >= 0, (
            f"commission_bps must be >= 0; got {self.commission_bps}"
        )
        assert self.borrow_rate_bps_annual >= 0, (
            f"borrow_rate_bps_annual must be >= 0; got {self.borrow_rate_bps_annual}"
        )
        assert self.default_order_type in ("MOO", "MOC"), (
            f"default_order_type must be MOO or MOC; got {self.default_order_type}"
        )
        assert self.trading_days_per_year > 0, (
            f"trading_days_per_year must be > 0; got {self.trading_days_per_year}"
        )
        assert self.adv_impact_coeff >= 0, (
            f"adv_impact_coeff must be >= 0; got {self.adv_impact_coeff}"
        )
        assert self.adv_impact_model in ("linear", "sqrt"), (
            "adv_impact_model must be 'linear' or 'sqrt'; "
            f"got {self.adv_impact_model!r}"
        )
        assert self.adv_floor_dollars > 0, (
            f"adv_floor_dollars must be > 0; got {self.adv_floor_dollars}"
        )


@dataclass
class TradingConfig:
    """Configuration for trading strategy."""

    initial_capital: float = 10000.0
    position_size: float = 0.1
    risk_free_rate: float = 0.02
    # Minimum |position_target| to emit a LONG/SHORT signal (else FLAT).
    # Exposed as config so a sweep can vary the trade threshold.
    signal_threshold: float = 0.1
    # Library default remains the original ternary order path; research/scripts/backtest.py
    # opts into target weights by default at the CLI layer.
    execution_style: str = "legacy_orders"
    # Absolute weight band around the currently filled target. In target-weight
    # mode, desired changes inside this band are suppressed rather than traded.
    rebalance_band_weight: float = 0.0
    # Multiplier on target-weight rebalance costs for stress tests.
    rebalance_cost_multiplier: float = 1.0
    # Portfolio gross cap for continuous target weights.
    max_gross_exposure: float = 1.0
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def __post_init__(self):
        assert 0 < self.position_size <= 1, (
            f"position_size must be in (0, 1]; got {self.position_size}"
        )
        assert self.initial_capital > 0, (
            f"initial_capital must be > 0; got {self.initial_capital}"
        )
        assert 0 <= self.signal_threshold < 1, (
            f"signal_threshold must be in [0, 1); got {self.signal_threshold}"
        )
        assert self.execution_style in ("legacy_orders", "target_weights"), (
            "execution_style must be 'legacy_orders' or 'target_weights'; "
            f"got {self.execution_style}"
        )
        assert self.rebalance_band_weight >= 0, (
            f"rebalance_band_weight must be >= 0; got {self.rebalance_band_weight}"
        )
        assert self.rebalance_cost_multiplier >= 0, (
            "rebalance_cost_multiplier must be >= 0; "
            f"got {self.rebalance_cost_multiplier}"
        )
        assert self.max_gross_exposure > 0, (
            f"max_gross_exposure must be > 0; got {self.max_gross_exposure}"
        )


