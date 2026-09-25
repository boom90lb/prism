"""Earnings-announcement Signal nodes (SPEC.md §5.3, §5.5): the announcement return and proximity.

``EarningsAnnouncementNode`` emits each security's latest announcement return (Chan,
Jegadeesh and Lakonishok 1996) while it is at most ``max_age_sessions`` old;
``AnnouncementProximityNode`` emits minus the days to its next expected announcement.
Both read a frozen, hashed point-in-time panel (``prism.scripts.build_earnings_panel``;
definitions in ``prism.io.earnings_events``). The panel already answers the causal
question: the value for session *t* uses only releases accepted, and returns realized,
by *t*'s close, and one security per company carries it. A high score is the long leg.

Rank-consumed by the decile long/short construct. The nodes read nothing from the price
panel they are handed except its last timestamp, so their scores are exactly invariant
to a price rescale. Stateless.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from prism.signal.base import FittedState, OutputKind, ScoringBasis, Signal, Units

NEW_YORK = "America/New_York"


class EarningsAnnouncementNode(Signal):
    """The latest announcement return per security, while it is no older than ``max_age_sessions``."""

    output_kind: OutputKind = "rank_score"
    units: Units = "rank"
    scoring_basis: ScoringBasis = "price"
    price_scale_invariant: bool = True
    fitted_state: FittedState = "none"
    estimated_decay_bars: int | None = None

    def __init__(self, panel: str, panel_sha256: str, *, max_age_sessions: int, horizon_bars: int = 21) -> None:
        if max_age_sessions < 0:
            raise ValueError(f"max_age_sessions must be >= 0, got {max_age_sessions}")
        if horizon_bars < 1:
            raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
        path = Path(panel)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != panel_sha256:
            raise ValueError(f"panel {path} has sha256 {digest}, registered {panel_sha256}; refusing (N7)")
        frame = pd.read_parquet(path, columns=["date", "sid", "ear", "age_sessions"])
        live = frame[frame["age_sessions"] <= int(max_age_sessions)]
        self._wide = live.pivot(index="date", columns="sid", values="ear").reindex(
            pd.DatetimeIndex(sorted(frame["date"].unique()))).sort_index()
        self._max_age = int(max_age_sessions)
        self._horizon = int(horizon_bars)

    @property
    def horizon_bars(self) -> int:
        return self._horizon

    @property
    def required_history(self) -> int:
        return 1

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "EarningsAnnouncementNode":
        if not isinstance(close, pd.DataFrame) or close.empty:
            raise TypeError("expected a non-empty wide close panel")
        return self

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        """The announcement return for the panel's last row; NaN where none is live."""
        if not isinstance(close, pd.DataFrame) or close.empty:
            raise TypeError("expected a non-empty wide close panel")
        stamp = pd.Timestamp(close.index[-1])
        day = (stamp.tz_convert(NEW_YORK) if stamp.tzinfo else stamp).tz_localize(None).normalize()
        if day not in self._wide.index:
            raise ValueError(f"announcement return: decision date {day.date()} is outside the frozen panel "
                             f"({self._wide.index[0].date()} to {self._wide.index[-1].date()}) (N7)")
        return self._wide.reindex(index=[day], columns=close.columns).iloc[0].astype(float).rename(None)


class AnnouncementProximityNode(Signal):
    """Minus the calendar days to each security's next expected announcement: the imminent announcers rank high.

    The earnings-announcement premium (Beaver 1968; Frazzini and Lamont 2007): stocks earn
    more around their announcements. The expected date comes from the same quarter a year
    earlier and releases accepted by the decision's close only (``prism.io.earnings_events``).
    A name up to five days overdue ranks highest; one with no year-earlier announcement has
    no opinion.
    """

    output_kind: OutputKind = "rank_score"
    units: Units = "rank"
    scoring_basis: ScoringBasis = "price"
    price_scale_invariant: bool = True
    fitted_state: FittedState = "none"
    estimated_decay_bars: int | None = None

    def __init__(self, panel: str, panel_sha256: str, *, horizon_bars: int = 5) -> None:
        if horizon_bars < 1:
            raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
        path = Path(panel)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != panel_sha256:
            raise ValueError(f"panel {path} has sha256 {digest}, registered {panel_sha256}; refusing (N7)")
        frame = pd.read_parquet(path, columns=["date", "sid", "days_to_expected"])
        self._wide = -frame.pivot(index="date", columns="sid", values="days_to_expected").sort_index()
        self._horizon = int(horizon_bars)

    @property
    def horizon_bars(self) -> int:
        return self._horizon

    @property
    def required_history(self) -> int:
        return 1

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "AnnouncementProximityNode":
        if not isinstance(close, pd.DataFrame) or close.empty:
            raise TypeError("expected a non-empty wide close panel")
        return self

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        """Minus the days to the expected announcement for the panel's last row; NaN where none is expected."""
        if not isinstance(close, pd.DataFrame) or close.empty:
            raise TypeError("expected a non-empty wide close panel")
        stamp = pd.Timestamp(close.index[-1])
        day = (stamp.tz_convert(NEW_YORK) if stamp.tzinfo else stamp).tz_localize(None).normalize()
        if day not in self._wide.index:
            raise ValueError(f"announcement proximity: decision date {day.date()} is outside the frozen panel "
                             f"({self._wide.index[0].date()} to {self._wide.index[-1].date()}) (N7)")
        return self._wide.reindex(index=[day], columns=close.columns).iloc[0].astype(float).rename(None)
