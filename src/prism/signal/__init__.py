"""Signal layer (SPEC.md §7.1) — pluggable cross-sectional alpha nodes.

A node consumes wide panels and emits standardized scores
(``E[r_h] / (sigma_daily * sqrt(h))``, I-3): NaN = no opinion, failures
raise (N7), sizing belongs to construction (I-4). ``ResidualSignalNode``
is implementation (a) of the contract (the Avellaneda-Lee residual core
under the standardized-score mapping); ``EnsembleSignalNode`` is
implementation (b), the salvaged forecast blend.

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
from prism.signal.residual_node import ResidualSignalNode

__all__ = [
    "EnsembleNodeConfig",
    "EnsembleSignalNode",
    "ResidualSignalNode",
    "Signal",
    "build_features",
    "forward_log_return",
]
