"""IO / data-access layer (SPEC.md §7.0): loader, universe, rate limit, store.

The single data-access home: ``loader`` (the Twelve Data client with the
range-keyed cache and the opt-in incremental delta fetch), ``universe_sp500``
(point-in-time S&P 500 membership with counted survivorship coverage),
``rate_limit`` (the token bucket for the keyed $0 spine), ``store`` (the
incremental parquet bar store), ``sessions`` (the bundled exchange calendar),
``manifest`` and ``dataset`` (the content-hashed dataset identity and the
as-of read path of engine spec §3.2), and ``synthetic`` (the deterministic
fixture writer behind ``prism data synthetic``).
Production-import-path safe (N8): stdlib + prism-internal only.
"""

from __future__ import annotations

from prism.io.dataset import (
    REFERENCE_DECISION_TIME,
    Dataset,
    DatasetError,
    DatasetView,
    StageNotAvailableError,
    read_decided_at,
)
from prism.io.loader import (
    BAR_TZ,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DataLoader,
)
from prism.io.manifest import (
    Manifest,
    ManifestError,
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
    "Dataset",
    "DatasetError",
    "DatasetView",
    "DEFAULT_OVERLAP_BARS",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DataBudgetExhausted",
    "DataLoader",
    "IncrementalBarStore",
    "Manifest",
    "ManifestError",
    "SplitRewriteDetected",
    "StageNotAvailableError",
    "REFERENCE_DECISION_TIME",
    "TWELVEDATA_PER_DAY",
    "TWELVEDATA_PER_MINUTE",
    "TokenBucket",
    "read_decided_at",
]
