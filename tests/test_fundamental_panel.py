"""Offline tests for the point-in-time characteristic panel (prism.io.fundamental_panel).

They pin each characteristic's definition on hand-built filings, the no-look-ahead
rule (a filing counts only from its acceptance), split handling for market equity and
net issuance (a 4-for-1 split between the share count's date and the price basis must
not change market equity or show up as issuance), and one security per company.
"""

import math

import numpy as np
import pandas as pd
import pytest

from prism.io import edgar
from prism.io import fundamental_panel as fp

NY = "America/New_York"
T = pd.Timestamp


def _fact(concept, end, val, accn, accepted, start=None, taxonomy="us-gaap", unit="USD"):
    return {"cik": 7, "taxonomy": taxonomy, "concept": concept, "unit": unit,
            "start": T(start) if start else pd.NaT, "end": T(end), "val": float(val), "accn": accn, "fy": 2023,
            "fp": "FY", "form": "10-K", "filed": T(accepted[:10]), "frame": None,
            "available_at": T(accepted, tz=NY), "availability_basis": "acceptance"}


def _company():
    fy22, fy23 = ("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31")
    k22, k23 = "2023-02-10 08:00", "2024-02-09 08:00"
    rows = [
        _fact("Assets", "2021-12-31", 800, "a22", k22), _fact("Assets", "2022-12-31", 1000, "a22", k22),
        _fact("Assets", "2023-12-31", 1200, "a23", k23), _fact("Assets", "2022-12-31", 1000, "a23", k23),
        _fact("StockholdersEquity", "2023-12-31", 500, "a23", k23),
        _fact("StockholdersEquity", "2022-12-31", 400, "a22", k22),
        _fact("NetIncomeLoss", fy22[1], 90, "a22", k22, start=fy22[0]),
        _fact("NetIncomeLoss", fy23[1], 120, "a23", k23, start=fy23[0]),
        _fact("NetCashProvidedByUsedInOperatingActivities", fy23[1], 150, "a23", k23, start=fy23[0]),
        _fact("Revenues", fy23[1], 1000, "a23", k23, start=fy23[0]),
        _fact("CostOfRevenue", fy23[1], 640, "a23", k23, start=fy23[0]),
        _fact("EntityCommonStockSharesOutstanding", "2023-01-31", 100, "a22", k22, taxonomy="dei", unit="shares"),
        _fact("EntityCommonStockSharesOutstanding", "2024-01-31", 420, "a23", k23, taxonomy="dei", unit="shares"),
    ]
    return pd.DataFrame(rows)


def test_split_factor_carries_counts_between_dates():
    splits = pd.DataFrame({"symbol": ["NVDA", "NVDA"], "ex_date": pd.to_datetime(["2021-07-20", "2024-06-10"]),
                           "ratio": [4.0, 10.0]})
    assert fp.split_factor(splits, "NVDA", T("2021-01-01"), T("2026-06-01")) == 40.0
    assert fp.split_factor(splits, "NVDA", T("2021-07-20"), T("2024-06-09")) == 1.0  # ex-date itself excluded
    assert fp.split_factor(splits, "NVDA", T("2024-07-01"), T("2021-01-01")) == pytest.approx(1 / 40)
    assert fp.split_factor(splits, "AAPL", T("2020-01-01"), T("2026-01-01")) == 1.0


def test_company_snapshot_definitions():
    facts = _company()
    snap = fp.company_snapshot(edgar.point_in_time(facts, T("2024-03-01 16:00", tz=NY)))
    assert snap["be"] == 500
    assert snap["gp_at"] == pytest.approx((1000 - 640) / 1200)  # gross profit from revenue less cost of revenue
    assert snap["oaccruals_at"] == pytest.approx((120 - 150) / 1200)
    assert snap["at_gr1"] == pytest.approx(1200 / 1000 - 1)
    assert (snap["shares"], snap["shares_prior"]) == (420, 100)
    assert snap["fiscal_year_end"] == T("2023-12-31")


def test_events_respect_acceptance():
    events = fp.company_events(_company(), since=T("2023-06-01", tz=NY))
    assert list(events["available_at"]) == [T("2023-02-10 08:00", tz=NY), T("2024-02-09 08:00", tz=NY)]
    before, after = events.iloc[0], events.iloc[1]
    assert before["be"] == 400 and after["be"] == 500
    assert before["at_gr1"] == pytest.approx(1000 / 800 - 1)


def test_surprise_standardizes_the_latest_year_on_year_change():
    ends = pd.date_range("2019-03-31", periods=16, freq="QE")
    values = pd.Series(np.arange(16, dtype=float) * 10 + np.tile([0, 5, 0, -5], 4), index=ends)
    diffs = values.to_numpy()[4:] - values.to_numpy()[:-4]
    expected = diffs[-1] / np.std(diffs[-8:], ddof=1) if np.std(diffs[-8:], ddof=1) > 0 else math.nan
    got = fp.surprise(values)
    assert (math.isnan(got) and math.isnan(expected)) or got == pytest.approx(expected)
    assert math.isnan(fp.surprise(values.iloc[:9]))  # only 5 year-on-year differences: below the minimum of 6


def _bars(days, close):
    ts = pd.DatetimeIndex(days).tz_localize(NY)
    return pd.DataFrame({"ts": ts, "close": close, "volume": 1_000.0})


def test_security_panel_splits_and_lookahead():
    events = fp.company_events(_company(), since=T("2023-06-01", tz=NY))
    days = ["2024-02-08", "2024-02-09", "2024-02-12", "2025-06-02"]
    bars = _bars(days, [10.0, 10.0, 10.0, 12.0])
    intervals = pd.DataFrame({"start": [T("2020-01-01")], "end": [T("2026-06-16")], "cik": [7]})
    # 4-for-1 split on 2024-01-15: between the 2023-01 count and the 2024-01 count, before the price basis
    splits = pd.DataFrame({"symbol": ["XYZ"], "ex_date": [T("2024-01-15")], "ratio": [4.0]})
    panel = fp.security_panel("XYZ", bars, intervals, {7: events}, splits, basis_date=T("2026-06-15"))
    panel = panel.set_index("date")
    # the 10-K accepted 2024-02-09 08:00 is visible at that day's close, not the day before
    assert panel.loc[T("2024-02-08"), "be"] == 400 and panel.loc[T("2024-02-09"), "be"] == 500
    # before it: 100 shares counted 2023-01-31, carried by the 4-for-1 split onto the basis: 400 x 10
    assert panel.loc[T("2024-02-08"), "me"] == pytest.approx(400 * 10.0)
    assert panel.loc[T("2024-02-09"), "me"] == pytest.approx(420 * 10.0)
    assert panel.loc[T("2024-02-09"), "be_me"] == pytest.approx(500 / 4200)
    # 100 -> 420 across a 4-for-1 split is 5% issuance, not 320%
    assert panel.loc[T("2024-02-09"), "chcsho_12m"] == pytest.approx(math.log(420 / 400))


def test_dedupe_keeps_one_security_per_company():
    day = T("2024-03-01")
    panel = pd.DataFrame({"date": [day, day, day], "cik": [5, 5, 9], "sid": ["GOOG", "GOOGL", "X"],
                          "liquidity": [2.0, 1.0, 1.0]})
    members = {"GOOGL": (None, T("2026-06-16")), "X": (None, T("2026-06-16"))}
    kept = fp.dedupe_companies(panel, members)
    assert list(kept["sid"]) == ["GOOGL", "X"]  # the index member wins over the more liquid non-member
    members["GOOG"] = (None, T("2026-06-16"))
    assert list(fp.dedupe_companies(panel, members)["sid"]) == ["GOOG", "X"]  # both members: liquidity decides
