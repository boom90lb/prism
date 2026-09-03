"""IO / data-access layer (SPEC.md §7.0): loader, universe, rate limit, store.

The single data-access home: ``loader`` (the Twelve Data client with the
range-keyed cache and the opt-in incremental delta fetch), ``universe_sp500``
(point-in-time S&P 500 membership with counted survivorship coverage),
``rate_limit`` (the token bucket for the keyed $0 spine), and ``store`` (the
incremental parquet bar store).
Production-import-path safe (N8): stdlib + prism-internal only.
"""

from __future__ import annotations

from prism.io.loader import (
    BAR_TZ,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DataLoader,
)
from prism.io.rate_limit import (
    TWELVEDATA_PER_DAY,
    TWELVEDATA_PER_MINUTE,
    DataBudgetExhausted,
    TokenBucket,
)
from prism.io.store import (
    DEFAULT_OVERLAP_BARS,
    IncrementalBarStore,
    SplitRewriteDetected,
)

__all__ = [
    "BAR_TZ",
    "DEFAULT_OVERLAP_BARS",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DataBudgetExhausted",
    "DataLoader",
    "IncrementalBarStore",
    "SplitRewriteDetected",
    "TWELVEDATA_PER_DAY",
    "TWELVEDATA_PER_MINUTE",
    "TokenBucket",
]
