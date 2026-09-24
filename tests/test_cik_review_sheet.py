"""Offline tests for the CIK review sheet's evidence and suggestion rules.

Each rule is pinned on the real case that motivated it: ARRS (successor issuer), TCF
(name handover), ARNC (a spin-off's pre-listing name must not split the parent), CONE
(co-filing operating partnership), VIAB (share-class ticker), ACE (a rename days
before removal is not the ticker's name).
"""

import pandas as pd
import pytest

from prism.scripts import cik_review_sheet as rs

T = pd.Timestamp


def test_name_periods_ignore_brief_names_and_clip_to_active_span():
    payload = {"name": "Chubb Ltd", "formerNames": [{"name": "ACE LTD", "from": "2009-12-17", "to": "2016-01-15"}]}
    names = [n for n, _, _ in rs.name_periods(payload, T("2009-06-15"), T("2016-01-19"), None)]
    assert names == ["ACE LTD"]  # "Chubb Ltd" was held 4 days inside the interval
    quirk = {"name": "X", "formerNames": [{"name": "OLD X", "from": "2012-07-02", "to": "2010-05-23"}]}
    assert [n for n, _, _ in rs.name_periods(quirk, T("2010-01-01"), T("2014-01-01"), None)] == ["OLD X", "X"]
    clipped = rs.name_periods({"name": "Gone Co", "formerNames": []}, T("2010-01-01"), T("2020-01-01"),
                              (T("2009-01-01"), T("2012-01-01")))
    assert clipped[0][2] == T("2012-01-01")


@pytest.mark.parametrize("ticker,name,fits", [
    ("ACE", "ACE LTD", True),
    ("AGN", "ALLERGAN INC", True),
    ("AGN", "Actavis plc", False),
    ("VIAB", "Viacom Inc.", True),  # share-class letter dropped
    ("CBS", "Viacom Inc.", False),
    ("BRK.B", "Berkshire Hathaway Inc", True),
])
def test_ticker_fits(ticker, name, fits):
    assert rs.ticker_fits(ticker, name) is fits


def _row(cik, names, fit=None, first=None, last=None, last_overall=None, llc=False, wiki=False,
         covers=(True, True)):
    return {"cik": cik, "name_now": names[-1] if names else None, "names_during_interval": names,
            "subsidiary_form_only": llc, "ticker_fit": fit is not None,
            "ticker_fit_from": None if fit is None else fit[0], "ticker_fit_to": None if fit is None else fit[1],
            "wikipedia_name_at_time": wiki, "filings_in_interval": 4 if first else 0, "first_in_interval": first,
            "last_in_interval": last, "last_periodic_overall": last_overall or last,
            "covers_start": covers[0], "covers_end": covers[1]}


START, END = T("2014-06-20"), T("2019-04-11")


def test_successor_issuer_split():
    rows = [_row(1141107, ["ARRIS GROUP INC"], first="2014-08-01", last="2015-11-06"),
            _row(1645494, ["ARRIS International plc"], first="2015-11-16", last="2019-03-29")]
    s = rs.suggest("ARRS", START, END, rows)
    assert s["kind"] == "split"
    assert [(g["cik"], g["from"], g["to"]) for g in s["segments"]] == [
        (1141107, "2014-06-20", "2015-11-15"), (1645494, "2015-11-16", "2019-04-11")]


def test_name_handover_split_and_the_arnc_guard():
    tcf = [_row(814184, ["TCF FINANCIAL CORP"], fit=("2009-06-15", "2019-11-28"), first="2009-07-30",
                last="2019-07-31", last_overall="2019-07-31"),
           _row(19612, ["CHEMICAL FINANCIAL CORP", "TCF FINANCIAL CORP"], fit=("2019-08-01", "2021-06-09"),
                first="2009-08-05", last="2021-05-07")]
    s = rs.suggest("TCF", T("1990-01-01"), T("2021-06-09"), tcf)
    assert s["kind"] == "split" and s["segments"][1]["from"] == "2019-08-01"
    # ARNC: the spin-off's name overlaps the parent's last months and outlasts it by days only
    arnc = [_row(4281, ["Arconic Inc."], fit=("2016-11-01", "2020-03-26"), first="2017-02-01", last="2020-02-21",
                 last_overall="2026-08-06", wiki=True),
            _row(1790982, ["Arconic Rolled Products Corp"], fit=("2019-12-01", "2020-04-01"), first="2020-03-30",
                 last="2020-03-30", covers=(False, True))]
    s = rs.suggest("ARNC", T("2016-11-01"), T("2020-04-01"), arnc)
    assert s == {**s, "kind": "cik", "cik": 4281}


def test_cofiling_partnership_is_dropped_and_unique_fit_wins():
    rows = [_row(1553023, ["CyrusOne Inc."], fit=("2013-01-01", "2022-03-30"), first="2013-03-29", last="2022-03-23"),
            _row(1575810, ["CyrusOne LP"], fit=("2013-08-14", "2015-11-06"), first="2013-08-14", last="2015-11-06",
                 llc=True)]
    s = rs.suggest("CONE", T("2013-01-01"), T("2022-03-30"), rows)
    assert s["kind"] == "cik" and s["cik"] == 1553023 and "1575810" in s["basis"]


def test_weak_coverage_and_none():
    both_fit = [_row(1, ["A CORP"], fit=("2010-01-01", "2015-01-01"), first="2010-02-01", last="2014-11-01"),
                _row(2, ["A HOLDINGS INC"], fit=("2010-01-01", "2015-01-01"), first="2012-02-01", last="2014-11-01",
                     covers=(False, True))]
    s = rs.suggest("A", T("2010-01-01"), T("2015-01-01"), both_fit)
    assert s["kind"] == "cik" and s["cik"] == 1 and s["basis"].startswith("WEAK")
    tie = [dict(both_fit[0]), dict(both_fit[1], covers_start=True)]
    assert rs.suggest("A", T("2010-01-01"), T("2015-01-01"), tie)["kind"] == "none"
