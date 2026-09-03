"""Time-series momentum Signal node for the trend sleeve (SPEC.md §7.1;
docs/trend_design.md §2) — per-name 12−1 TSMOM on the pinned ETF universe.

Score is the trailing total-return ratio ``close[t - skip] / close[t - lookback]
- 1`` (strictly trailing; appending future bars never changes a past score).
Construction consumes only the **sign** of this score with inverse-volatility
sizing (`construct_inverse_vol_targets`); magnitude is retained so diagnostics
and fragility probes (e.g. T2 no-skip) see the same raw series the design
describes, not a pre-thresholded ±1.

Unlike ``MomentumSignalNode``, this node is **not** residual-eligibility
screened: the trend universe is a fixed 10-ETF list
(``docs/trend_design.md`` §1), and a name without lookback history simply
carries NaN (empty cell → cash; no proxy splicing, N7). Volume is optional and
unused — accepted for Signal-contract call-site uniformity.

Stateless: ``fit`` validates and returns self; ``score`` needs no prior fit.
Default-off sleeve: landing mechanics is not a counted trial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.signal.base import Signal

# Pinned at trend_v1 ratification (docs/trend_design.md §1). Recorded here as
# the node's default membership hint; callers may pass a different column set
# for tests. Construction never invents names outside the close panel.
TREND_V1_UNIVERSE: tuple[str, ...] = (
    "SPY",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "GLD",
    "PDBC",
    "UUP",
)


class TrendSignalNode(Signal):
    """Per-name 12−1 time-series momentum (sign consumed downstream).

    ``lookback_bars`` / ``skip_bars`` default to the pinned 252 / 21 window.
    ``horizon_bars`` tags the forward holding period (does not enter the score).
    """

    def __init__(
        self,
        *,
        lookback_bars: int = 252,
        skip_bars: int = 21,
        horizon_bars: int = 21,
    ) -> None:
        if lookback_bars < 2:
            raise ValueError(f"lookback_bars must be >= 2, got {lookback_bars}")
        if skip_bars < 0:
            raise ValueError(f"skip_bars must be >= 0, got {skip_bars}")
        if skip_bars >= lookback_bars:
            raise ValueError(
                f"skip_bars ({skip_bars}) must be < lookback_bars ({lookback_bars})"
            )
        if horizon_bars < 1:
            raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
        self._lookback = int(lookback_bars)
        self._skip = int(skip_bars)
        self._horizon = int(horizon_bars)

    @property
    def horizon_bars(self) -> int:
        return self._horizon

    @property
    def required_history(self) -> int:
        # lookback + 1 prices for the trailing ratio (base at t-lookback, recent
        # at t-skip, decision at t).
        return self._lookback + 1

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "TrendSignalNode":
        """Validate the panel; the node keeps no cross-boundary fitted state."""
        _validate_close(close, required=self.required_history)
        return self

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        """Raw 12−1 TSMOM for the panel's last row (NaN = no opinion)."""
        _validate_close(close, required=self.required_history)
        base = pd.to_numeric(close.iloc[-1 - self._lookback], errors="coerce")
        recent = pd.to_numeric(close.iloc[-1 - self._skip], errors="coerce")
        with np.errstate(invalid="ignore", divide="ignore"):
            momentum = recent / base.where(base > 0.0) - 1.0
        scores = momentum.where(np.isfinite(momentum)).astype(float)
        scores.index = close.columns
        return scores


def _validate_close(close: pd.DataFrame, *, required: int) -> None:
    if not isinstance(close, pd.DataFrame):
        raise TypeError(f"expected a wide DataFrame close panel, got {type(close).__name__}")
    if close.columns.has_duplicates:
        raise ValueError("panel has duplicate symbol columns")
    if not close.index.is_monotonic_increasing:
        raise ValueError("panel index must be sorted ascending")
    if len(close) < required:
        raise ValueError(
            f"panel has {len(close)} rows; the trend node needs >= {required} "
            "(lookback_bars + 1)"
        )
