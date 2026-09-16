"""Signal layer (SPEC.md §7.1) — pluggable cross-sectional alpha nodes.

A node consumes wide panels and emits standardized scores
(``E[r_h] / (sigma_daily * sqrt(h))``, I-3): NaN = no opinion, failures
raise (N7), sizing belongs to construction (I-4). ``ResidualSignalNode``
is implementation (a) of the contract (the Avellaneda-Lee residual core
under the standardized-score mapping); implementation (b), the salvaged
forecast blend, left this package at stage 6 and is now the optional
``prism-ensemble`` distribution (``prism_ensemble.EnsembleSignalNode``),
which is what took xgboost and statsmodels out of the engine's dependency
set; ``MomentumSignalNode`` is the ratified B1 candidate's node — 12-1
cross-sectional momentum in rank space, consumed by the decile long/short
construct (its score is deliberately not sigma-normalized because the decile
book uses only the cross-sectional rank; see the node docstring);
``TrendSignalNode`` is the trend sleeve's
per-name 12−1 TSMOM node (default-off; docs/trend_design.md), consumed by
``construct_inverse_vol_targets``.

Every node declares what it emits (``output_kind``, ``units``), which close
panel it wants (``scoring_basis``), whether the rescale property holds, and
whether it carries fitted state; ``check_declarations`` refuses a component
that omits one (engine specification §3.3).

Production-import-path safe (N8): no JAX/torch/prophet in the closure.
"""

from __future__ import annotations

from prism.signal.base import (
    DECLARED_FIELDS,
    FITTED_STATES,
    OUTPUT_KINDS,
    SCORING_BASES,
    UNITS,
    DeclarationError,
    Signal,
    check_declarations,
)
from prism.signal.momentum_node import MomentumSignalNode
from prism.signal.residual_node import ResidualSignalNode
from prism.signal.trend_node import TREND_V1_UNIVERSE, TrendSignalNode

__all__ = [
    "DECLARED_FIELDS",
    "DeclarationError",
    "FITTED_STATES",
    "MomentumSignalNode",
    "OUTPUT_KINDS",
    "ResidualSignalNode",
    "SCORING_BASES",
    "Signal",
    "TREND_V1_UNIVERSE",
    "TrendSignalNode",
    "UNITS",
    "check_declarations",
]
