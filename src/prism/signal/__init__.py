"""Signal layer (SPEC.md §7.1) — pluggable cross-sectional alpha nodes.

A node consumes wide panels and emits standardized scores
(``E[r_h] / (sigma_daily * sqrt(h))``, I-3): NaN = no opinion, failures
raise (N7), sizing belongs to construction (I-4). The forecast-ensemble
node here is implementation (b) of the contract; the residual-reversion
node (implementation (a)) lives with the residual layer and converges to
this contract with the R2 construction rework.

Production-import-path safe (N8): no JAX/torch/prophet in the closure.
"""

from __future__ import annotations

from prism.signal.base import Signal
from prism.signal.ensemble_node import (
    EnsembleNodeConfig,
    EnsembleSignalNode,
    build_features,
    forward_log_return,
)

__all__ = [
    "EnsembleNodeConfig",
    "EnsembleSignalNode",
    "Signal",
    "build_features",
    "forward_log_return",
]
