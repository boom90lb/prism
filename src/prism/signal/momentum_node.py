"""Cross-sectional momentum Signal node (SPEC.md §7.1; docs/momentum_design.md,
docs/demotion_design.md §2b) — the ratified B1 candidate's alpha node.

12-1 cross-sectional momentum: ``score[t] = close[t - skip] / close[t - lookback]
- 1`` (strictly trailing, so appending future bars never changes a past score),
masked by the SAME residual-eligibility screen the residual sleeve trades under
(``prism.residual.factors.compute_eligibility``). NaN = no opinion: a name
without ``lookback`` bars of history, an ineligible name, or a non-positive base
price.

Rank-consumed, not sigma-normalized (deliberate). The book this node feeds is the
decile long/short construct (``prism.portfolio.construct.construct_decile_neutral``),
which depends only on the cross-sectional *rank* of the scores, never their
magnitude. So the node emits the raw momentum ratio rather than the contract's
``E[r_h] / (sigma_daily * sqrt(h))`` form: dividing by a per-name sigma would
reorder names and produce a book that is not the ratified B1 (docs/momentum_design.md
§2, "adopting a different cell would be a new discovery event"). The ratio is still
price-level scale-invariant (contract-required), and ``horizon_bars`` tags the
forward holding period. This is the one node whose score lives in rank space, not
sigma space; faithfulness to demotion_design.md §2b outweighs a normalization the
decile construct discards.

Statelessness: momentum is a trailing price ratio plus a causal eligibility mask,
so like the residual node it carries no state across the train/score boundary —
``fit`` validates and returns self; ``score`` needs no prior ``fit``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.residual.factors import ResidualStatArbConfig, compute_eligibility
from prism.signal.base import Signal


class MomentumSignalNode(Signal):
    """12-1 cross-sectional momentum scores, residual-eligibility screened.

    ``config`` is the residual core's config, used ONLY for its eligibility
    screen (``min_price``, dollar-volume floor, history/membership) so the
    momentum book trades exactly the residual sleeve's universe. ``lookback_bars``
    / ``skip_bars`` are B1's momentum window (252 / 21 = "12 months, skipping the
    most recent"). ``horizon_bars`` tags the forward holding period the score is
    about (it never enters the score, which is rank-consumed). ``membership_mask``
    (day × symbol bool) threads the point-in-time index gate to the eligibility
    screen.
    """

    def __init__(
        self,
        config: ResidualStatArbConfig | None = None,
        *,
        lookback_bars: int = 252,
        skip_bars: int = 21,
        horizon_bars: int = 21,
        membership_mask: pd.DataFrame | None = None,
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
        self._config = config or ResidualStatArbConfig()
        self._lookback = int(lookback_bars)
        self._skip = int(skip_bars)
        self._horizon = int(horizon_bars)
        self._membership_mask = membership_mask

    @property
    def horizon_bars(self) -> int:
        return self._horizon

    @property
    def required_history(self) -> int:
        # lookback + 1 prices for the trailing ratio; corr_window + 1 for the
        # eligibility screen's full-history gate. The larger binds.
        return max(self._lookback, self._config.corr_window) + 1

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "MomentumSignalNode":
        """Validate the panel; the node keeps no cross-boundary fitted state."""
        _validate_panels(close, volume, required=self.required_history)
        return self

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        """Raw 12-1 momentum for the panel's last row, eligibility-masked."""
        volume = _validate_panels(close, volume, required=self.required_history)
        base = pd.to_numeric(close.iloc[-1 - self._lookback], errors="coerce")
        recent = pd.to_numeric(close.iloc[-1 - self._skip], errors="coerce")
        with np.errstate(invalid="ignore", divide="ignore"):
            momentum = recent / base.where(base > 0.0) - 1.0
        eligible = (
            compute_eligibility(close, volume, self._config, self._membership_mask)
            .iloc[-1]
            .astype(bool)
        )
        scores = momentum.where(eligible.to_numpy() & np.isfinite(momentum)).astype(float)
        scores.index = close.columns
        return scores


def _validate_panels(
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    *,
    required: int,
) -> pd.DataFrame:
    """Shared fit/score panel checks; returns the volume panel (never None)."""
    if not isinstance(close, pd.DataFrame):
        raise TypeError(f"expected a wide DataFrame close panel, got {type(close).__name__}")
    if volume is None:
        raise ValueError(
            "MomentumSignalNode requires the volume panel: the eligibility screen "
            "gates on dollar volume (N7, no silent pass)"
        )
    if not close.columns.equals(volume.columns) or not close.index.equals(volume.index):
        raise ValueError("close and volume panels must share index and columns")
    if close.columns.has_duplicates:
        raise ValueError("panel has duplicate symbol columns")
    if not close.index.is_monotonic_increasing:
        raise ValueError("panel index must be sorted ascending")
    if len(close) < required:
        raise ValueError(
            f"panel has {len(close)} rows; the momentum node needs >= {required} "
            "(max(lookback, corr_window) + 1)"
        )
    return volume
