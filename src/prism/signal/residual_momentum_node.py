"""Residual momentum Signal node (SPEC.md §5.3, "residualized versions of each";
docs/characteristics_design.md).

Blitz, Huij and Martens (2011) residual momentum: the 12-1 momentum of a
name's *factor-residual* return rather than its raw return, scaled by the
residual's own volatility over the same window. At the decision bar the node
regresses each eligible name's trailing ``lookback_bars`` daily returns on the
top ``config.n_factors`` eigenportfolio returns struck on that same window
(``prism.residual.factors.trailing_factor_residuals``), drops the most recent
``skip_bars`` residuals, and emits

    score = sum(residual[t - lookback : t - skip]) / std(residual[t - lookback : t - skip])

Rank-consumed like the raw momentum node: the book is the decile long/short
construct, which reads only cross-sectional order. The scaling by residual
volatility is part of the characteristic as published, not a sigma-normalization
of the emission, and it changes the order (a name with a large residual drift
and a large residual variance ranks below one with the same drift and less
variance), so it stays.

Everything is computed on returns, so the score is price-level invariant; the
eligibility screen deliberately is not (it gates on absolute price and dollar
volume), so the property holds within the eligible region. Stateless: a
trailing refit at every decision bar, ``fit`` validates and returns self.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.residual.factors import ResidualStatArbConfig, trailing_factor_residuals
from prism.signal.base import FittedState, OutputKind, ScoringBasis, Signal, Units


class ResidualMomentumSignalNode(Signal):
    """12-1 residual momentum scores, residual-eligibility screened.

    ``config`` supplies the eligibility screen and the factor count; the
    residualization window is ``lookback_bars`` (the characteristic's own
    formation window, 252 = twelve months), never ``config.regr_window``.
    ``skip_bars`` drops the most recent residuals (21 = the skipped month).
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
        lookback_bars: int = 252,
        skip_bars: int = 21,
        horizon_bars: int = 21,
        membership_mask: pd.DataFrame | None = None,
    ) -> None:
        self._config = config or ResidualStatArbConfig()
        if lookback_bars < self._config.n_factors + 5:
            raise ValueError(
                f"lookback_bars must be >= n_factors + 5 = {self._config.n_factors + 5}, got {lookback_bars}"
            )
        if skip_bars < 0:
            raise ValueError(f"skip_bars must be >= 0, got {skip_bars}")
        if lookback_bars - skip_bars < 3:
            raise ValueError(
                f"lookback_bars - skip_bars must leave >= 3 residuals, got {lookback_bars} - {skip_bars}"
            )
        if horizon_bars < 1:
            raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
        self._lookback = int(lookback_bars)
        self._skip = int(skip_bars)
        self._horizon = int(horizon_bars)
        self._membership_mask = membership_mask

    @property
    def horizon_bars(self) -> int:
        return self._horizon

    @property
    def required_history(self) -> int:
        # lookback returns need lookback + 1 prices; the eligibility screen's
        # full-history gate needs corr_window + 1. The larger binds.
        return max(self._lookback, self._config.corr_window) + 1

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "ResidualMomentumSignalNode":
        """Validate the panel; the node keeps no cross-boundary fitted state."""
        _validate_panels(close, volume, required=self.required_history, who="residual momentum")
        return self

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        """Residual 12-1 momentum for the panel's last row, eligibility-masked."""
        volume = _validate_panels(close, volume, required=self.required_history, who="residual momentum")
        residuals = trailing_factor_residuals(
            close, volume, self._config, self._membership_mask, window=self._lookback
        )
        formation = residuals.iloc[: len(residuals) - self._skip] if self._skip else residuals
        with np.errstate(invalid="ignore", divide="ignore"):
            spread = formation.std(axis=0, ddof=1)
            raw = formation.sum(axis=0) / spread.where(spread > 0.0)
        scores = pd.Series(np.nan, index=close.columns, dtype=float)
        scores.loc[raw.index] = raw.to_numpy(dtype=float)
        return scores.where(np.isfinite(scores)).astype(float)


def _validate_panels(
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    *,
    required: int,
    who: str,
) -> pd.DataFrame:
    """Shared fit/score panel checks; returns the volume panel (never None)."""
    if not isinstance(close, pd.DataFrame):
        raise TypeError(f"expected a wide DataFrame close panel, got {type(close).__name__}")
    if volume is None:
        raise ValueError(
            f"the {who} node requires the volume panel: the eligibility screen gates on dollar volume (N7, no silent pass)"
        )
    if not close.columns.equals(volume.columns) or not close.index.equals(volume.index):
        raise ValueError("close and volume panels must share index and columns")
    if close.columns.has_duplicates:
        raise ValueError("panel has duplicate symbol columns")
    if not close.index.is_monotonic_increasing:
        raise ValueError("panel index must be sorted ascending")
    if len(close) < required:
        raise ValueError(f"panel has {len(close)} rows; the {who} node needs >= {required}")
    return volume
