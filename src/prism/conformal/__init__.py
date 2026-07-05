"""Conformal prediction primitives for the ensemble.

Phase 2.5: provides EnbPI-style (block-cross-conformal) calibration over
the meta-learner's OOF positions plus an online ACI quantile adapter.
Attached at the ensemble position output; the trading strategy modulates
its confidence factor by the band width.
"""

from prism.conformal.aci import ACIState
from prism.conformal.enbpi import EnbPICalibrator

__all__ = ["EnbPICalibrator", "ACIState"]
