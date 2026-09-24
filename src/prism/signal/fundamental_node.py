"""Fundamental characteristic Signal node (SPEC.md §5.3; docs/fundamentals_design.md).

Emits one published accounting characteristic per security at the decision bar, read
from a frozen, hashed point-in-time panel (``prism.scripts.build_fundamental_panel``;
definitions in ``prism.io.fundamental_panel``). The panel already answers the causal
question: the value for session *t* uses only filings accepted by *t*'s close, and one
security per company carries it. The node adds only the published sign, so a high emitted
score is always the long leg: low asset growth, low accruals and low net issuance score
high.

Rank-consumed by the decile long/short construct. The node reads nothing from the price
panel it is handed except its last timestamp, so its scores are exactly invariant to a
price rescale; market equity for book-to-market comes from the frozen panel, which was
built from the same frozen dataset. Stateless.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from prism.io.fundamental_panel import CHARACTERISTICS, expected_signs
from prism.signal.base import FittedState, OutputKind, ScoringBasis, Signal, Units

NEW_YORK = "America/New_York"


class FundamentalCharacteristicNode(Signal):
    """One characteristic from a frozen point-in-time panel, signed to its published direction."""

    output_kind: OutputKind = "rank_score"
    units: Units = "rank"
    scoring_basis: ScoringBasis = "price"
    price_scale_invariant: bool = True
    fitted_state: FittedState = "none"
    estimated_decay_bars: int | None = None

    def __init__(self, characteristic: str, panel: str, panel_sha256: str, *, horizon_bars: int = 21) -> None:
        if characteristic not in CHARACTERISTICS:
            raise ValueError(f"characteristic must be one of {CHARACTERISTICS}, got {characteristic!r}")
        if horizon_bars < 1:
            raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
        path = Path(panel)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != panel_sha256:
            raise ValueError(f"panel {path} has sha256 {digest}, registered {panel_sha256}; refusing (N7)")
        frame = pd.read_parquet(path, columns=["date", "sid", characteristic])
        self._wide = frame.pivot(index="date", columns="sid", values=characteristic).sort_index()
        self._characteristic = characteristic
        self._sign = expected_signs()[characteristic]
        self._horizon = int(horizon_bars)

    @property
    def horizon_bars(self) -> int:
        return self._horizon

    @property
    def required_history(self) -> int:
        return 1

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "FundamentalCharacteristicNode":
        if not isinstance(close, pd.DataFrame) or close.empty:
            raise TypeError("expected a non-empty wide close panel")
        return self

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        """The characteristic for the panel's last row, signed; NaN where the frozen panel has no value."""
        if not isinstance(close, pd.DataFrame) or close.empty:
            raise TypeError("expected a non-empty wide close panel")
        stamp = pd.Timestamp(close.index[-1])
        day = (stamp.tz_convert(NEW_YORK) if stamp.tzinfo else stamp).tz_localize(None).normalize()
        if day not in self._wide.index:
            raise ValueError(f"{self._characteristic}: decision date {day.date()} is outside the frozen panel "
                             f"({self._wide.index[0].date()} to {self._wide.index[-1].date()}) (N7)")
        values = self._wide.loc[day].reindex(close.columns).astype(float)
        return (values * self._sign).rename(None)
