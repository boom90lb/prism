"""Conformal prediction primitives (SPEC §9 KEEP+adapt).

Provides EnbPI-style (block-cross-conformal) calibration over out-of-fold
predictions plus an online ACI quantile adapter. Production consumer:
``prism.signal.ensemble_node`` (score band). The legacy research stack
(``research.models.ensemble``, ``research.trading``) attaches the same
primitives at its position output.
"""

from prism.conformal.aci import ACIState
from prism.conformal.enbpi import EnbPICalibrator

__all__ = ["EnbPICalibrator", "ACIState"]
