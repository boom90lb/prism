"""IO / data-access layer (SPEC.md §7.0) — being assembled under the R4 ADAPT.

Currently holds the rate limiter for the keyed $0 data spine; the incremental
store lands here next, and ``prism.data_loader`` folds in as it is reworked
(rename-with-rework — the module moves when its behavior does, not before).
Production-import-path safe (N8): stdlib + prism-internal only.
"""

from __future__ import annotations

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
    "DEFAULT_OVERLAP_BARS",
    "DataBudgetExhausted",
    "IncrementalBarStore",
    "SplitRewriteDetected",
    "TWELVEDATA_PER_DAY",
    "TWELVEDATA_PER_MINUTE",
    "TokenBucket",
]
