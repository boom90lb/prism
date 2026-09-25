"""Point-in-time earnings announcements and announcement returns (SPEC.md §5.3, N1, N7).

An announcement is an original 8-K furnishing item 2.02, results of operations and
financial condition (``edgar.earnings_releases``); amendments (8-K/A) are never events.
Its day 0 is the New York calendar date of the filing's acceptance, or the next session
when that date is not one. That is the date Compustat's ``rdq`` records for a release
after the close, so the window below covers the reaction session either way. A second
item 2.02 8-K from the same company within ``MIN_GAP_SESSIONS`` sessions of the last one
kept belongs to it (a supplemental or corrected release) and is dropped.

The announcement return (Chan, Jegadeesh and Lakonishok 1996; Chen and Zimmermann's
``AnnouncementReturn``) sums the security's close-to-close return less the market's over
sessions d0-2 .. d0+1. The market return is the equal-weighted mean return of that
session's index members, which is known at the close; French's market series is
published with a lag and could not be used live. The value is known at the close of
d0+1, its signal date, which is the first decision it may enter (N1). It trades at the
next open (N2). A window with a missing return gives no value.

For the announcement premium (Beaver 1968; Frazzini and Lamont 2007), each session also
carries the calendar days to the company's next expected announcement, from releases
accepted by that session's close only. Each known announcement's anniversary (its day
plus ``ANNIVERSARY_DAYS``) is a candidate unless an actual announcement has already
realized it (it falls within ``REALIZED_WITHIN_DAYS`` after the latest known one) or it
is more than ``OVERDUE_DAYS`` past; the expected date is the earliest candidate. This
history includes releases before the dataset's first session, dated by the next weekday
(no session calendar exists there) and de-duplicated within ``DUPLICATE_DAYS``; they
set anniversaries and are never events.

One security per company per session, by the rule of ``prism.io.fundamental_panel``:
index members first, then the more liquid security, then the security id.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from prism.io import edgar
from prism.io.fundamental_panel import dedupe_companies, dollar_volume_rank_key

MIN_GAP_SESSIONS = 5
WINDOW = (-2, 1)  # sessions relative to day 0, inclusive
MIN_MARKET_NAMES = 100
ANNIVERSARY_DAYS = 364
REALIZED_WITHIN_DAYS = 30
OVERDUE_DAYS = 5
DUPLICATE_DAYS = 7  # the calendar-day form of MIN_GAP_SESSIONS, for history without a session calendar

EVENT_COLUMNS = ("sid", "cik", "accn", "acceptance", "d0", "signal_date", "ear")
PANEL_COLUMNS = ("date", "sid", "cik", "member", "liquidity", "accn", "d0", "signal_date", "age_sessions", "ear",
                 "days_to_expected")


def naive_days(stamps: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(pd.to_datetime(stamps))
    if values.tz is not None:
        values = values.tz_convert(edgar.NEW_YORK).tz_localize(None)
    return values.normalize()


def _ns(days: pd.DatetimeIndex) -> np.ndarray:
    """Naive dates as int64 nanoseconds, for positional search."""
    return np.asarray(days, dtype="datetime64[ns]").view("int64")


def announcement_dates(releases: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Day 0 per original item 2.02 8-K, as a session position; near-duplicates dropped per company.

    ``releases`` carries ``cik``, ``accn``, ``form`` and a New York ``acceptance``;
    ``sessions`` is the dataset's naive session dates, sorted. Releases accepted before the
    first session or whose day 0 falls after the last session are dropped.
    """
    if not sessions.is_monotonic_increasing:
        raise ValueError("sessions must be sorted")
    frame = releases.loc[releases["form"] == "8-K", ["cik", "accn", "acceptance"]].copy()
    days = naive_days(frame["acceptance"])
    frame["d0_pos"] = np.searchsorted(_ns(sessions), _ns(days), side="left")
    inside = (days >= sessions[0]) & (frame["d0_pos"].to_numpy() < len(sessions))
    frame = frame[inside].sort_values(["cik", "acceptance", "accn"]).reset_index(drop=True)
    keep = np.zeros(len(frame), dtype=bool)
    ciks, positions = frame["cik"].to_numpy(), frame["d0_pos"].to_numpy(int)
    last_cik, last_pos = None, 0
    for i in range(len(frame)):
        if ciks[i] != last_cik or positions[i] - last_pos >= MIN_GAP_SESSIONS:
            keep[i] = True
            last_cik, last_pos = ciks[i], int(positions[i])
    out = frame.loc[keep, ["cik", "accn", "acceptance", "d0_pos"]]
    out["d0"] = sessions[out["d0_pos"].to_numpy(int)] if len(out) else pd.DatetimeIndex([])
    return out.reset_index(drop=True)


def market_returns(returns: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Equal-weighted mean return of each session's index members; NaN where fewer than MIN_MARKET_NAMES report."""
    held = returns.where(members.reindex(index=returns.index, columns=returns.columns, fill_value=False).astype(bool))
    counts = held.notna().sum(axis=1)
    return held.mean(axis=1).where(counts >= MIN_MARKET_NAMES)


def announcement_returns(d0_pos: np.ndarray, stock: np.ndarray, market: np.ndarray) -> np.ndarray:
    """Sum of (stock - market) return over d0-2 .. d0+1 per event; NaN if the window leaves the sample or has a gap."""
    low, high = WINDOW
    excess = np.asarray(stock, float) - np.asarray(market, float)
    out = np.full(len(d0_pos), np.nan)
    for i, p in enumerate(np.asarray(d0_pos, int)):
        if p + low < 0 or p + high >= len(excess):
            continue
        window = excess[p + low:p + high + 1]
        if not np.isnan(window).any():
            out[i] = float(window.sum())
    return out


def security_events(sid: str, intervals: pd.DataFrame, dates_by_cik: Mapping[int, pd.DataFrame],
                    sessions: pd.DatetimeIndex, stock: np.ndarray, market: np.ndarray) -> pd.DataFrame:
    """Announcements of the companies a security mapped to, each inside its mapping interval, with returns."""
    pieces = []
    for cik, start, end in zip(intervals["cik"].astype(int), pd.to_datetime(intervals["start"]),
                               pd.to_datetime(intervals["end"])):
        dates = dates_by_cik.get(int(cik))
        if dates is None or dates.empty:
            continue
        inside = dates[(dates["d0"] >= start) & (dates["d0"] <= end)].copy()
        if inside.empty:
            continue
        inside["ear"] = announcement_returns(inside["d0_pos"].to_numpy(int), stock, market)
        signal_pos = inside["d0_pos"].to_numpy(int) + WINDOW[1]
        valid = signal_pos < len(sessions)
        inside = inside[valid]
        inside["signal_pos"] = signal_pos[valid]
        inside["signal_date"] = sessions[inside["signal_pos"].to_numpy(int)]
        pieces.append(inside)
    if not pieces:
        return pd.DataFrame(columns=[*EVENT_COLUMNS, "signal_pos"])
    out = pd.concat(pieces, ignore_index=True)
    out["sid"] = sid
    return out[[*EVENT_COLUMNS, "signal_pos"]].sort_values("signal_pos").reset_index(drop=True)


def company_history(releases: pd.DataFrame) -> pd.DataFrame:
    """Every original item 2.02 8-K per company, over the whole store, for expected dates.

    ``day`` is the acceptance date rolled forward to a weekday; a release within
    ``DUPLICATE_DAYS`` of the last one kept for the company is dropped.
    """
    frame = releases.loc[releases["form"] == "8-K", ["cik", "acceptance"]]
    frame = frame.sort_values(["cik", "acceptance"]).reset_index(drop=True)
    rolled = np.busday_offset(naive_days(frame["acceptance"]).to_numpy().astype("datetime64[D]"), 0, roll="forward")
    frame["day"] = pd.DatetimeIndex(rolled.astype("datetime64[ns]"))
    keep = np.zeros(len(frame), dtype=bool)
    ciks, stamps = frame["cik"].to_numpy(), _ns(pd.DatetimeIndex(frame["day"]))
    gap = pd.Timedelta(days=DUPLICATE_DAYS).value
    last_cik, last_day = None, 0
    for i in range(len(frame)):
        if ciks[i] != last_cik or stamps[i] - last_day >= gap:
            keep[i] = True
            last_cik, last_day = ciks[i], int(stamps[i])
    return frame[keep].reset_index(drop=True)


def days_to_expected(history: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """Calendar days from each date to one company's next expected announcement, known at that date's close."""
    out = np.full(len(dates), np.nan)
    if history.empty:
        return out
    history = history.sort_values("acceptance")
    accepted = _ns(pd.DatetimeIndex(pd.to_datetime(history["acceptance"], utc=True)).tz_convert(None))
    closes = np.array([edgar.session_close(d).value for d in dates], dtype="int64")
    known = np.searchsorted(accepted, closes, side="right")
    day = _ns(pd.DatetimeIndex(history["day"]))
    anniversary = day + pd.Timedelta(days=ANNIVERSARY_DAYS).value
    realized, overdue, one_day = (pd.Timedelta(days=d).value for d in (REALIZED_WITHIN_DAYS, OVERDUE_DAYS, 1))
    stamps = _ns(pd.DatetimeIndex(dates))
    for i, n in enumerate(known):
        if n == 0:
            continue
        candidates = anniversary[:n]
        ok = (candidates > day[:n].max() + realized) & (candidates >= stamps[i] - overdue)
        if ok.any():
            out[i] = (candidates[ok].min() - stamps[i]) / one_day
    return out


def security_panel(sid: str, bars: pd.DataFrame, intervals: pd.DataFrame, events: pd.DataFrame,
                   history_by_cik: Mapping[int, pd.DataFrame], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per session the security traded while mapped to a company.

    Each row carries the latest announcement return known at that close with its age
    (``age_sessions`` counts sessions from the signal date, 0 on the signal date itself)
    and the days to the next expected announcement. Rows with neither carry nothing.
    """
    days = naive_days(bars["ts"])
    close = bars["close"].astype(float).reset_index(drop=True)
    frame = pd.DataFrame({"date": days, "liquidity": dollar_volume_rank_key(
        close, bars["volume"].astype(float).reset_index(drop=True)).to_numpy()})
    frame["pos"] = np.searchsorted(_ns(sessions), _ns(days))
    off = (frame["pos"].to_numpy() >= len(sessions)) | (
        _ns(sessions)[np.minimum(frame["pos"].to_numpy(), len(sessions) - 1)] != _ns(days))
    if off.any():
        raise edgar.EdgarError(f"{sid}: {int(off.sum())} bar dates are not sessions of the dataset (N7)")
    frame["cik"] = pd.array([pd.NA] * len(frame), dtype="Int64")
    for cik, start, end in zip(intervals["cik"].astype(int), pd.to_datetime(intervals["start"]),
                               pd.to_datetime(intervals["end"])):
        frame.loc[(frame["date"] >= start) & (frame["date"] <= end), "cik"] = int(cik)
    frame = frame.dropna(subset=["cik"])
    carried = ["accn", "acceptance", "d0", "signal_date", "ear", "signal_pos"]
    pieces = []
    for _, part in frame.groupby("cik"):
        cik = int(part["cik"].iloc[0])
        part = part.sort_values("pos").reset_index(drop=True)
        known = events[events["cik"] == cik].dropna(subset=["ear"]) if len(events) else events
        if len(known):
            part = pd.merge_asof(part, known.sort_values("signal_pos")[carried], left_on="pos",
                                 right_on="signal_pos", direction="backward")
        else:
            for column in carried:
                part[column] = np.nan
        part["age_sessions"] = (part["pos"] - part["signal_pos"]).astype(float)
        history = history_by_cik.get(cik)
        part["days_to_expected"] = (days_to_expected(history, pd.DatetimeIndex(part["date"]))
                                    if history is not None else np.nan)
        pieces.append(part)
    if not pieces:
        return pd.DataFrame(columns=list(PANEL_COLUMNS))
    out = pd.concat(pieces, ignore_index=True)
    out = out[out["ear"].notna() | out["days_to_expected"].notna()].reset_index(drop=True)
    out["sid"] = sid
    return out


def finalize(panel: pd.DataFrame, members: Mapping[str, tuple[pd.Timestamp | None, pd.Timestamp]]) -> pd.DataFrame:
    """One security per company per session, the fixed column set, typed."""
    if panel.empty:
        raise edgar.EdgarError("announcement panel is empty (N7)")
    out = dedupe_companies(panel, members)
    missing = [c for c in PANEL_COLUMNS if c not in out.columns]
    if missing:
        raise edgar.EdgarError(f"announcement panel lacks columns {missing}")
    out = out[list(PANEL_COLUMNS)].copy()
    out["cik"] = out["cik"].astype("int64")
    out["age_sessions"] = out["age_sessions"].astype(float)
    return out.sort_values(["date", "sid"]).reset_index(drop=True)
