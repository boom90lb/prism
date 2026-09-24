"""Idiosyncratic volatility Signal node (SPEC.md §5.3; docs/characteristics_design.md).

Ang, Hodrick, Xing and Zhang (2006): names with high recent idiosyncratic
volatility earn low subsequent returns. At the decision bar the node regresses
each eligible name's trailing ``estimation_bars`` daily returns on the top
``config.n_factors`` eigenportfolio returns struck on that window
(``prism.residual.factors.trailing_factor_residuals``) and emits

    score = -std(residual[t - vol_bars : t])

so that the *top* decile of the score is the low-volatility leg and the decile
long/short construct is long low idiosyncratic volatility, short high. The sign
is inside the score deliberately: the registration declares the component's
expected sign as positive, the rank-to-weight rule is the control's own, and a
reader sees one convention across every component.

Computed on returns, so price-level invariant within the eligible region (the
screen gates on absolute price and dollar volume). Stateless: a trailing refit
at every decision bar, ``fit`` validates and returns self.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.residual.factors import ResidualStatArbConfig, trailing_factor_residuals
from prism.signal.base import FittedState, OutputKind, ScoringBasis, Signal, Units


class IdiosyncraticVolSignalNode(Signal):
    """Negative trailing idiosyncratic volatility, residual-eligibility screened.

    ``estimation_bars`` is the factor-regression window (252 = the same twelve
    months the momentum book forms on); ``vol_bars`` is the trailing window
    the residual standard deviation is taken over (63 = one quarter).
    ``horizon_bars`` tags the forward holding period and never enters the score.
    """

    output_kind: OutputKind = "rank_score"
    units: Units = "rank"
    scoring_basis: ScoringBasis = "price"
    price_scale_invariant: bool = True
    fitted_state: FittedState = "none"
    estimated_decay_bars: int | None = None

    def __init__(
        self,
        config: ResidualStatArbConfig | None = None,
        *,
        estimation_bars: int = 252,
        vol_bars: int = 63,
        horizon_bars: int = 21,
        membership_mask: pd.DataFrame | None = None,
    ) -> None:
        self._config = config or ResidualStatArbConfig()
        if estimation_bars < self._config.n_factors + 5:
            raise ValueError(
                f"estimation_bars must be >= n_factors + 5 = {self._config.n_factors + 5}, got {estimation_bars}"
            )
        if vol_bars < 5:
            raise ValueError(f"vol_bars must be >= 5, got {vol_bars}")
        if vol_bars > estimation_bars:
            raise ValueError(f"vol_bars ({vol_bars}) must be <= estimation_bars ({estimation_bars})")
        if horizon_bars < 1:
            raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
        self._estimation = int(estimation_bars)
        self._vol_bars = int(vol_bars)
        self._horizon = int(horizon_bars)
        self._membership_mask = membership_mask

    @property
    def horizon_bars(self) -> int:
        return self._horizon

    @property
    def required_history(self) -> int:
        return max(self._estimation, self._config.corr_window) + 1

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "IdiosyncraticVolSignalNode":
        """Validate the panel; the node keeps no cross-boundary fitted state."""
        _validate_panels(close, volume, required=self.required_history)
        return self

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        """Negative idiosyncratic volatility for the panel's last row, eligibility-masked."""
        volume = _validate_panels(close, volume, required=self.required_history)
        residuals = trailing_factor_residuals(
            close, volume, self._config, self._membership_mask, window=self._estimation
        )
        recent = residuals.iloc[-self._vol_bars :]
        with np.errstate(invalid="ignore", divide="ignore"):
            ivol = recent.std(axis=0, ddof=1)
            raw = -ivol.where(ivol > 0.0)
        scores = pd.Series(np.nan, index=close.columns, dtype=float)
        scores.loc[raw.index] = raw.to_numpy(dtype=float)
        return scores.where(np.isfinite(scores)).astype(float)


def _validate_panels(close: pd.DataFrame, volume: pd.DataFrame | None, *, required: int) -> pd.DataFrame:
    """Shared fit/score panel checks; returns the volume panel (never None)."""
    if not isinstance(close, pd.DataFrame):
        raise TypeError(f"expected a wide DataFrame close panel, got {type(close).__name__}")
    if volume is None:
        raise ValueError(
            "the idiosyncratic volatility node requires the volume panel: the eligibility screen gates on "
            "dollar volume (N7, no silent pass)"
        )
    if not close.columns.equals(volume.columns) or not close.index.equals(volume.index):
        raise ValueError("close and volume panels must share index and columns")
    if close.columns.has_duplicates:
        raise ValueError("panel has duplicate symbol columns")
    if not close.index.is_monotonic_increasing:
        raise ValueError("panel index must be sorted ascending")
    if len(close) < required:
        raise ValueError(f"panel has {len(close)} rows; the idiosyncratic volatility node needs >= {required}")
    return volume
