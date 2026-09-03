"""Residual-reversion signal node (SPEC.md §7.1 implementation (a)).

Wraps the Avellaneda-Lee residual core (`prism.residual`) in the ``Signal``
contract: at the decision bar it re-estimates eigenportfolios, factor betas,
and per-name OU fits on trailing windows only, then converts the resulting
s-score into the contract's standardized units.

The unit conversion is the OU conditional expectation. With the cumulative
residual ``X`` following ``X_{n+1} = a + b X_n + zeta`` and
``s = (X_t - equilibrium) / sigma_eq``, the expected residual move over ``h``
bars is ``E[X_{t+h} - X_t] = -s * sigma_eq * (1 - b^h)``. The book trades the
stock *hedged* (construction nets the factor legs), so the residual move is
the tradable expectation and the factor contribution has no place in the
score. Standardizing per I-3:

    score = -s * sigma_eq * (1 - b^h) / (sigma_daily * sqrt(h))

where ``sigma_daily`` is the name's trailing raw-return volatility over the
same regression window the OLS used.

Statelessness: unlike the forecast-ensemble node, the residual machinery
carries *no* fitted state across the train/score boundary — every estimate is
a trailing-window refit at the decision bar (causal by construction, N1/I-2).
``fit`` therefore only validates the panel and returns ``self``; ``score``
does not require a prior ``fit`` call.

Scale note: the *estimation* is exactly price-level invariant (everything is
computed on returns), but the eligibility screen is deliberately not — it
gates on absolute price (``min_price``) and dollar volume
(``min_median_dollar_volume``). The property test rescales within the
eligible region, where scores are bit-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.residual.factors import ResidualStatArbConfig, compute_returns
from prism.residual.residual import compute_residual_signal_panel
from prism.signal.base import Signal


class ResidualSignalNode(Signal):
    """Cross-sectional residual-reversion scores from the A-L signal core.

    ``config`` is the residual core's own config (windows, factor count,
    eligibility screens); ``horizon_bars`` is the forward horizon the emitted
    scores are about — it enters only the s-score → expected-return mapping,
    never the estimation. ``membership_mask`` (optional, day × symbol bool)
    is the point-in-time universe gate, threaded to the eligibility screen.
    """

    def __init__(
        self,
        config: ResidualStatArbConfig | None = None,
        *,
        horizon_bars: int = 5,
        membership_mask: pd.DataFrame | None = None,
    ) -> None:
        if horizon_bars < 1:
            raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
        self._config = config or ResidualStatArbConfig()
        self._horizon_bars = int(horizon_bars)
        self._membership_mask = membership_mask

    @property
    def horizon_bars(self) -> int:
        return self._horizon_bars

    @property
    def required_history(self) -> int:
        # warmup (corr + regr windows) of returns, plus the price row that
        # seeds the first return.
        return self._config.warmup_bars + 1

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "ResidualSignalNode":
        """Validate the panel; the node keeps no cross-boundary fitted state."""
        _validate_panels(close, volume, self._config, required=self.required_history)
        return self

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        """Standardized residual-reversion scores for the panel's last row."""
        volume = _validate_panels(close, volume, self._config, required=self.required_history)
        tail = close.iloc[-self.required_history :]
        vol_tail = volume.iloc[-self.required_history :]
        panel = compute_residual_signal_panel(tail, vol_tail, self._config, self._membership_mask)

        s = panel.sscore.iloc[-1]
        sigma_eq = panel.sigma_eq.iloc[-1]
        half_life = panel.half_life_bars.iloc[-1]
        tradeable = panel.tradeable.iloc[-1].astype(bool)

        h = float(self._horizon_bars)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = np.power(0.5, 1.0 / half_life)  # hl -> inf gives b -> 1 (score 0); masked below anyway
            expected_move = -s * sigma_eq * (1.0 - np.power(b, h))
            sigma_daily = compute_returns(tail).iloc[-self._config.regr_window :].std(ddof=1)
            scores = expected_move / (sigma_daily * np.sqrt(h))
        scores = scores.where(tradeable & np.isfinite(scores)).astype(float)
        scores.index = close.columns
        return scores


def _validate_panels(
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    config: ResidualStatArbConfig,
    *,
    required: int,
) -> pd.DataFrame:
    """Shared fit/score panel checks; returns the volume panel (never None)."""
    if not isinstance(close, pd.DataFrame):
        raise TypeError(f"expected a wide DataFrame close panel, got {type(close).__name__}")
    if volume is None:
        raise ValueError(
            "ResidualSignalNode requires the volume panel: eligibility screens on dollar volume (N7, no silent pass)"
        )
    if not close.columns.equals(volume.columns) or not close.index.equals(volume.index):
        raise ValueError("close and volume panels must share index and columns")
    if close.columns.has_duplicates:
        raise ValueError("panel has duplicate symbol columns")
    if not close.index.is_monotonic_increasing:
        raise ValueError("panel index must be sorted ascending")
    if len(close) < required:
        raise ValueError(
            f"panel has {len(close)} rows; the residual node needs >= {required} "
            f"(corr_window {config.corr_window} + regr_window {config.regr_window} + 1)"
        )
    return volume
