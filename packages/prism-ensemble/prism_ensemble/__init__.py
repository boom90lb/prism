"""The learned forecast-ensemble signal node — the engine's optional strategy package.

This is where ``prism.signal.ensemble_node`` went at stage 6 of the engine
overhaul (engine specification section 2, `signal/ensemble_node.py`, Removal:
"the learned model and its calibrators move to an optional strategy package, and
`xgboost` and `statsmodels` leave the dependency set entirely"). The module is
unchanged by the move except for the name of the thing an operator installs;
its pre-move source is at the tag ``pre-legacy-retirement-stage6``.

It is a separate distribution, ``prism-ensemble``, rather than an extra of
``prism-trading``: an extra of a distribution that no longer carries the code
would install xgboost and statsmodels and nothing that uses them. The engine
imports this package nowhere at module scope. ``prism.scripts.paper_books`` and
``prism.scripts.replay_loop`` import it inside the ``ensemble`` book's signal
factory, so an engine install without this package runs every other book and
refuses only that one, naming what to install.

It depends on ``prism``: the ``Signal`` contract (``prism.signal.base``), the
EnbPI calibrator (``prism.conformal``) and the purged splitter
(``prism.validation.walk_forward``). ``prism`` depends on nothing here.
"""

from __future__ import annotations

from prism_ensemble.ensemble_node import (
    ENSEMBLE_DISTRIBUTION,
    EnsembleNodeConfig,
    EnsembleSignalNode,
    build_features,
    forward_log_return,
)

__all__ = [
    "ENSEMBLE_DISTRIBUTION",
    "EnsembleNodeConfig",
    "EnsembleSignalNode",
    "build_features",
    "forward_log_return",
]
