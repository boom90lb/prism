"""Point-in-time characteristic panel from the EDGAR fundamentals store (SPEC.md §5.3, N1, N7).

Six published characteristics, defined once here and computed per company from the
facts it had filed by each moment (``prism.io.edgar``), then joined onto a security's
sessions so the value for session *t* uses only filings accepted by *t*'s close.

| id | characteristic (source) | definition, per company |
|---|---|---|
| ``be_me`` | book-to-market (Fama and French 1992) | latest stockholders' equity / market equity at *t*; NaN if equity <= 0 |
| ``gp_at`` | gross profitability (Novy-Marx 2013) | fiscal-year gross profit / total assets at that fiscal year end |
| ``at_gr1`` | asset growth (Cooper, Gulen and Schill 2008) | assets at the latest fiscal year end / assets a year earlier - 1 |
| ``oaccruals_at`` | operating accruals (Sloan 1996; cash-flow form) | (fiscal-year net income - operating cash flow) / assets at that year end |
| ``niq_su`` | earnings surprise (Foster, Olsen and Shevlin 1984) | (NI_q - NI_q-4) / std of that difference over the last 8 quarters (>= 6 required) |
| ``chcsho_12m`` | net share issuance (Pontiff and Woodgate 2008) | log of shares outstanding over shares ~12 months earlier, split-adjusted |

Gross profit is reported gross profit, or revenue less cost of revenue when a company
does not report it. The fiscal year is the latest annual net-income period; balance-sheet
values are the instants at its end (±7 days, for 52/53-week years). Quarters come from
``edgar.quarterly_values`` (reported, or differenced from year-to-date and annual values).

Market equity at session *t* is the latest reported share count carried onto the price
basis by every split between the count's date and the dataset's adjustment date, times
*t*'s split-adjusted close: equivalent to the count carried to *t* times *t*'s raw
price. Net share issuance carries the earlier count onto the later one's basis the same
way. Splits are fetched by ``prism.live.split_history`` and applied by :func:`split_factor`.

One security per company per session: when several of the dataset's securities map to
the same company (share classes; a renamed company listed under two tickers), index
members are preferred, then the higher trailing-20-session median dollar volume, then
the security id. The others carry no value for that session, so a company is counted
once in any cross-section.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from prism.io import edgar

CHARACTERISTICS = ("be_me", "gp_at", "at_gr1", "oaccruals_at", "niq_su", "chcsho_12m")
MATCH_DAYS = 7
YEAR_WINDOW = (320, 410)  # days between a period end and the "year earlier" one
SUE_QUARTERS = 8
SUE_MIN = 6
LIQUIDITY_BARS = 20


def split_factor(splits: pd.DataFrame, symbol: str, after: pd.Timestamp, through: pd.Timestamp) -> float:
    """Product of split ratios for ``symbol`` with ex-date in ``(after, through]``; 1.0 when there are none.

    ``splits`` has ``symbol``, ``ex_date`` and ``ratio`` (new shares per old share), as
    ``prism.live.split_history.fetch_splits`` returns it. A share count reported at
    ``after`` times this factor is on the share basis of ``through``; with ``after`` later
    than ``through`` the factor inverts, carrying a later count back.
    """
    if after == through:
        return 1.0
    lo, hi, invert = (after, through, False) if after < through else (through, after, True)
    rows = splits[(splits["symbol"] == symbol) & (splits["ex_date"] > lo) & (splits["ex_date"] <= hi)]
    factor = float(rows["ratio"].prod()) if len(rows) else 1.0
    return 1.0 / factor if invert else factor


def _by_end(frame: pd.DataFrame, kind: str) -> pd.Series:
    rows = frame[frame["kind"] == kind]
    return pd.Series(rows["val"].to_numpy(float), index=pd.DatetimeIndex(rows["end"])).sort_index()


def _at(series: pd.Series, when: pd.Timestamp, days: int = MATCH_DAYS) -> float:
    """The value whose date is within ``days`` of ``when`` (closest), else NaN."""
    if series.empty:
        return math.nan
    gaps = np.abs((series.index - when).days)
    i = int(np.argmin(gaps))
    return float(series.iloc[i]) if gaps[i] <= days else math.nan


def _year_before(series: pd.Series, when: pd.Timestamp) -> tuple[pd.Timestamp | None, float]:
    """The entry dated a year before ``when`` (within ``YEAR_WINDOW`` days, closest to 365)."""
    gaps = (when - series.index).days
    ok = (gaps >= YEAR_WINDOW[0]) & (gaps <= YEAR_WINDOW[1])
    if not ok.any():
        return None, math.nan
    candidates = series[ok]
    i = int(np.argmin(np.abs((when - candidates.index).days - 365)))
    return candidates.index[i], float(candidates.iloc[i])


def surprise(quarters: pd.Series) -> float:
    """Standardized unexpected earnings from a quarter-end-indexed net income series."""
    diffs = []
    for end, value in quarters.items():
        before, prior = _year_before(quarters, end)
        if before is not None and np.isfinite(prior):
            diffs.append((end, value - prior))
    if not diffs or diffs[-1][0] != quarters.index[-1]:
        return math.nan  # the latest quarter has no year-earlier comparison
    recent = np.array([d for _, d in diffs[-SUE_QUARTERS:]], dtype=float)
    if len(recent) < SUE_MIN:
        return math.nan
    spread = float(np.std(recent, ddof=1))
    return float(recent[-1] / spread) if spread > 0 else math.nan


def company_snapshot(pit: pd.DataFrame) -> dict[str, Any]:
    """Everything the characteristics need from one company's point-in-time facts."""
    out: dict[str, Any] = {k: math.nan for k in ("be", "shares", "shares_prior", "gp_at", "at_gr1", "oaccruals_at",
                                                 "niq_su")}
    out.update(shares_end=pd.NaT, shares_prior_end=pd.NaT, fiscal_year_end=pd.NaT)
    equity = edgar.latest_instant(pit, "equity")
    if len(equity):
        out["be"] = float(equity["value"].iloc[0])
    shares = _by_end(edgar.field_facts(pit, "shares_outstanding"), "instant")
    shares = shares[shares > 0]
    if len(shares):
        out["shares"], out["shares_end"] = float(shares.iloc[-1]), shares.index[-1]
        before, prior = _year_before(shares, shares.index[-1])
        if before is not None:
            out["shares_prior"], out["shares_prior_end"] = prior, before

    net_income = edgar.field_facts(pit, "net_income")
    annual_ni = _by_end(net_income, "annual")
    if len(annual_ni):
        fye = annual_ni.index[-1]
        out["fiscal_year_end"] = fye
        assets = _by_end(edgar.field_facts(pit, "assets"), "instant")
        at = _at(assets, fye)
        gp = _at(_by_end(edgar.field_facts(pit, "gross_profit"), "annual"), fye)
        if not np.isfinite(gp):
            gp = _at(_by_end(edgar.field_facts(pit, "revenue"), "annual"), fye) - _at(
                _by_end(edgar.field_facts(pit, "cost_of_revenue"), "annual"), fye)
        cfo = _at(_by_end(edgar.field_facts(pit, "cfo"), "annual"), fye)
        if np.isfinite(at) and at > 0:
            out["gp_at"] = gp / at
            out["oaccruals_at"] = (float(annual_ni.iloc[-1]) - cfo) / at
            _, prior_at = _year_before(assets, fye)  # every 10-K reports the prior year's balance sheet
            if np.isfinite(prior_at) and prior_at > 0:
                out["at_gr1"] = at / prior_at - 1.0
    quarters = edgar.quarterly_values(net_income)
    if len(quarters):
        series = pd.Series(quarters["val"].to_numpy(float), index=pd.DatetimeIndex(quarters["end"])).sort_index()
        out["niq_su"] = surprise(series)
    return out


def company_events(facts: pd.DataFrame, since: pd.Timestamp) -> pd.DataFrame:
    """One snapshot per moment the company's knowledge changed, from the last one before ``since`` on.

    ``facts`` is one company's facts with ``available_at``. Rows are indexed by the
    acceptance moment from which the snapshot holds.
    """
    moments = pd.DatetimeIndex(sorted(facts["available_at"].unique()))
    earlier = moments[moments < since]
    moments = moments[moments >= since]
    if len(earlier):
        moments = moments.insert(0, earlier[-1])
    rows = []
    for moment in moments:
        snapshot = company_snapshot(edgar.point_in_time(facts, moment))
        rows.append({"available_at": moment, **snapshot})
    return pd.DataFrame(rows)


def dollar_volume_rank_key(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Trailing 20-session median dollar volume through each session (point in time)."""
    return (close * volume).rolling(LIQUIDITY_BARS, min_periods=1).median()


def security_panel(sid: str, bars: pd.DataFrame, intervals: pd.DataFrame, events_by_cik: Mapping[int, pd.DataFrame],
                   splits: pd.DataFrame, basis_date: pd.Timestamp) -> pd.DataFrame:
    """Daily characteristics for one security: its sessions, the company it mapped to on each, the latest snapshot."""
    days = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True).dt.tz_convert(edgar.NEW_YORK).dt.normalize()
                            .dt.tz_localize(None))
    frame = pd.DataFrame({"date": days, "close": bars["close"].to_numpy(float),
                          "liquidity": dollar_volume_rank_key(bars["close"].astype(float).reset_index(drop=True),
                                                              bars["volume"].astype(float).reset_index(drop=True))
                          .to_numpy()})
    frame["cik"] = pd.array([pd.NA] * len(frame), dtype="Int64")
    for row in intervals.itertuples():
        inside = (frame["date"] >= row.start) & (frame["date"] <= row.end)
        frame.loc[inside, "cik"] = int(row.cik)
    frame = frame.dropna(subset=["cik"])
    pieces = []
    for cik, part in frame.groupby("cik"):
        events = events_by_cik.get(int(cik))
        if events is None or events.empty:
            continue
        part = part.copy()
        part["decision_at"] = [edgar.session_close(d) for d in part["date"]]
        joined = pd.merge_asof(part.sort_values("decision_at"), events.sort_values("available_at"),
                               left_on="decision_at", right_on="available_at", direction="backward")
        pieces.append(joined)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    out["sid"] = sid
    shares_on_basis = [s * split_factor(splits, sid, e, basis_date) if pd.notna(e) else math.nan
                       for s, e in zip(out["shares"], out["shares_end"])]
    out["me"] = np.asarray(shares_on_basis, dtype=float) * out["close"]
    out["be_me"] = np.where((out["be"] > 0) & (out["me"] > 0), out["be"] / out["me"], np.nan)
    issuance = []
    for s1, e1, s0, e0 in zip(out["shares"], out["shares_end"], out["shares_prior"], out["shares_prior_end"]):
        if pd.isna(e0) or not (s0 > 0 and s1 > 0):
            issuance.append(math.nan)
        else:
            issuance.append(math.log(s1 / (s0 * split_factor(splits, sid, e0, e1))))
    out["chcsho_12m"] = issuance
    return out


def dedupe_companies(panel: pd.DataFrame, members: Mapping[str, tuple[pd.Timestamp | None, pd.Timestamp]]
                     ) -> pd.DataFrame:
    """Keep one security per company per date: index members first, then liquidity, then sid."""
    def is_member(row_sid: str, day: pd.Timestamp) -> bool:
        span = members.get(row_sid)
        return span is not None and (span[0] is None or span[0] <= day) and day <= span[1]

    panel = panel.copy()
    panel["member"] = [is_member(s, d) for s, d in zip(panel["sid"], panel["date"])]
    panel = panel.sort_values(["date", "cik", "member", "liquidity", "sid"], ascending=[True, True, False, False, True])
    return panel.drop_duplicates(["date", "cik"], keep="first").sort_values(["date", "sid"]).reset_index(drop=True)


PANEL_COLUMNS = ("date", "sid", "cik", "available_at", "member", "liquidity", "close", "be", "me", "shares",
                 "shares_end", "fiscal_year_end", *CHARACTERISTICS)


def finalize(panel: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PANEL_COLUMNS if c not in panel.columns]
    if missing:
        raise edgar.EdgarError(f"characteristic panel lacks columns {missing}")
    out = panel[list(PANEL_COLUMNS)].copy()
    out["cik"] = out["cik"].astype("int64")
    return out


def expected_signs() -> dict[str, int]:
    """Sign that makes a high emitted score the long leg, per the published direction."""
    return {"be_me": 1, "gp_at": 1, "at_gr1": -1, "oaccruals_at": -1, "niq_su": 1, "chcsho_12m": -1}


def iter_companies(parquet_file: Any, wanted: Iterable[int]) -> Iterable[tuple[int, pd.DataFrame]]:
    """Stream one company's facts at a time from the builder's one-row-group-per-company file."""
    wanted = set(int(c) for c in wanted)
    for i in range(parquet_file.num_row_groups):
        group = parquet_file.read_row_group(i).to_pandas()
        for cik, part in group.groupby("cik"):
            if int(cik) in wanted:
                yield int(cik), part.reset_index(drop=True)
