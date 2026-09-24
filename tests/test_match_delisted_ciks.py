"""Offline tests for name-based CIK matching (prism.scripts.match_delisted_ciks).

They pin the normalization, the SEC index parser, the Wikipedia name windows, the
accept-or-review decision (exactly one candidate that filed during the interval is
accepted; anything else goes to review), and the builder's per-interval override. The
end-to-end case is the one that motivated the rule: Wikipedia's "Alcoa Inc" matches
both the old Alcoa (now Howmet, CIK 4281) and today's Alcoa Corp, and only the old one
filed while the AA interval ran.
"""

import json

import pandas as pd
import pytest

from prism.io import edgar
from prism.scripts import build_fundamentals as bf
from prism.scripts import match_delisted_ciks as md

UA = "Prism research test@example.com"


@pytest.mark.parametrize("raw,expected", [
    ("Cabot Oil & Gas", "CABOT OIL AND GAS"),
    ("CABOT OIL & GAS CORP", "CABOT OIL AND GAS"),
    ("TESORO CORP /NEW/", "TESORO"),
    ("Michael Kors Holdings Ltd", "MICHAEL KORS"),
    ("The Walt Disney Company", "WALT DISNEY"),
    ("WALT DISNEY CO/", "WALT DISNEY"),
    ("Alphabet Inc. (Class A)", "ALPHABET"),
    ("News Corp Class B", "NEWS"),
    ("Abiomed[12]", "ABIOMED"),
    ("Holding", "HOLDING"),  # a name is never reduced to nothing
])
def test_normalize_name(raw, expected):
    assert md.normalize_name(raw) == expected


def test_iter_name_index_parses_names_with_colons_and_skips_nameless_rows():
    raw = b":0001003197:\nALCOA INC:0000004281:\nWEIRD: NAME CO:0000000007:\nbroken line\n"
    assert list(md.iter_name_index(raw)) == [("", 1003197), ("ALCOA INC", 4281), ("WEIRD: NAME CO", 7)]


def _changes_tables():
    return [[
        ["Date", "Added", "", "Removed", "", "Reason"],
        ["", "Ticker", "Security", "Ticker", "Security", ""],
        ["November 1, 2016", "ARNC", "Arconic Inc", "AA", "Alcoa Inc", "Separation"],
        ["June 3, 2015", "AA", "Alcoa Inc", "XYZ", "Xyz Corp", "Replaced"],
        ["not a date", "Q", "Q", "Q", "Q", ""],
    ]]


def test_change_rows_and_interval_names_windows():
    changes = md.change_rows(_changes_tables(), "sp500")
    assert list(changes["removed_name"]) == ["Alcoa Inc", "Xyz Corp"]
    names = md.interval_names(changes, "AA", pd.Timestamp("2015-06-03"), pd.Timestamp("2016-11-01"))
    assert {(n["name"], n["role"]) for n in names} == {("Alcoa Inc", "removed"), ("Alcoa Inc", "added")}
    # outside both windows: falls back to rows inside the interval
    late = md.interval_names(changes, "AA", pd.Timestamp("2014-01-01"), pd.Timestamp("2018-01-01"))
    assert {n["role"] for n in late} == {"removed_in_interval", "added_in_interval"}
    assert md.interval_names(changes, "NOPE", pd.Timestamp("2014-01-01"), pd.Timestamp("2018-01-01")) == []


def test_decide_accepts_only_a_unique_validated_candidate():
    names = [{"name": "Alcoa Inc", "role": "removed"}]
    start, end = pd.Timestamp("2010-01-01"), pd.Timestamp("2016-11-01")
    ok = {"status": "accepted", "entity_name": "Howmet"}
    no = {"status": "no_periodic_filing_in_interval", "entity_name": "Alcoa Corp"}
    assert md.decide("AA", start, end, names, {4281: ok, 1675149: no})["cik"] == 4281
    assert md.decide("AA", start, end, names, {4281: ok, 5: ok})["reason"] == "several_candidates_filed_in_interval"
    assert md.decide("AA", start, end, names, {1675149: no})["reason"] == "no_candidate_filed_in_interval"
    assert md.decide("AA", start, end, names, {})["reason"] == "no_sec_name_match"
    assert md.decide("AA", start, end, [], {})["reason"] == "no_wikipedia_name"


def _submissions(cik, name, dates, former=()):
    return {"cik": cik, "name": name, "tickers": [], "formerNames": [{"name": f} for f in former],
            "filings": {"recent": {
                "accessionNumber": [f"{cik}-{i}" for i in range(len(dates))], "form": ["10-Q"] * len(dates),
                "filingDate": dates, "acceptanceDateTime": [f"{d}T12:00:00.000Z" for d in dates],
                "reportDate": dates, "items": [""] * len(dates)}, "files": []}}


SUBS = {
    4281: _submissions(4281, "Howmet Aerospace Inc.", ["2012-05-01", "2016-08-01"], former=["ALCOA INC"]),
    1675149: _submissions(1675149, "Alcoa Corp", ["2017-02-28", "2020-05-01", "2022-05-01"]),
    11: _submissions(11, "Twin A", ["2013-05-01"]),
    12: _submissions(12, "Twin B", ["2013-06-01"]),
}


def _transport(url, headers, timeout):
    for cik, payload in SUBS.items():
        if url == edgar.SUBMISSIONS_URL.format(cik=cik):
            return json.dumps(payload).encode()
    raise AssertionError(f"unexpected url {url}")


def test_match_end_to_end(tmp_path):
    client = edgar.SecClient(UA, tmp_path, transport=_transport, clock=lambda: 0.0, sleep=lambda s: None)
    coverage = {"not_accepted": [
        {"ticker": "AA", "start": "1990-01-01", "end": "2016-11-01", "status": "no_periodic_filing_in_interval"},
        {"ticker": "TWN", "start": "2012-01-01", "end": "2014-01-01", "status": "unresolved"},
        {"ticker": "GONE", "start": "2012-01-01", "end": "2014-01-01", "status": "unresolved"},
        {"ticker": "OLD", "start": "1995-01-01", "end": "2005-01-01", "status": "unresolved"},  # pre-XBRL: skipped
    ]}
    changes = pd.DataFrame([
        {"index": "sp500", "date": pd.Timestamp("2016-11-01"), "added": "ARNC", "added_name": "Arconic",
         "removed": "AA", "removed_name": "Alcoa Inc"},
        {"index": "sp400", "date": pd.Timestamp("2014-01-01"), "added": "X", "added_name": "X",
         "removed": "TWN", "removed_name": "Twin Co"},
    ])
    index = (b"ALCOA INC:0000004281:\nALCOA CORP:0001675149:\nALCOA FUJIKURA LTD:0000009999:\n"
             b"TWIN CO:0000000011:\nTWIN CORP:0000000012:\n")
    results = {r["ticker"]: r for r in md.match(coverage, changes, index, client)}
    assert set(results) == {"AA", "TWN", "GONE"}
    assert results["AA"]["decision"] == "accepted" and results["AA"]["cik"] == 4281
    assert {c["cik"] for c in results["AA"]["candidates"]} == {4281, 1675149}  # Fujikura never matched
    assert results["TWN"]["reason"] == "several_candidates_filed_in_interval"
    assert results["GONE"]["reason"] == "no_wikipedia_name"


def test_builder_applies_name_match_to_one_interval(tmp_path):
    tickers = {"0": {"cik_str": 1675149, "ticker": "AA", "title": "Alcoa Corp"}}
    subs = {4281: SUBS[4281], 1675149: SUBS[1675149]}
    facts = {cik: {"cik": cik, "entityName": "x", "facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"end": "2015-12-31", "val": 1.0, "accn": f"{cik}-0", "form": "10-K", "filed": "2016-02-01"}]}}}}}
        for cik in subs}

    def transport(url, headers, timeout):
        if url == edgar.TICKERS_URL:
            return json.dumps(tickers).encode()
        for cik in subs:
            if url == edgar.SUBMISSIONS_URL.format(cik=cik):
                return json.dumps(subs[cik]).encode()
            if url == edgar.COMPANYFACTS_URL.format(cik=cik):
                return json.dumps(facts[cik]).encode()
        raise AssertionError(url)

    client = edgar.SecClient(UA, tmp_path / "raw", transport=transport, clock=lambda: 0.0, sleep=lambda s: None)
    membership = pd.DataFrame({"ticker": ["AA", "AA"], "start": pd.to_datetime(["1990-01-01", "2021-12-20"]),
                               "end": pd.to_datetime(["2016-11-01", "2026-09-23"])})
    matches_path = tmp_path / "matches.json"
    matches_path.write_text(json.dumps({"accepted": [
        {"ticker": "AA", "start": "1990-01-01", "cik": 4281, "title": "Howmet Aerospace Inc."}]}))
    bf.build(membership, client, tmp_path / "out", tmp_path / "ledger.json", "2026-09-24",
             name_matches=bf.load_name_matches(matches_path))
    accepted = pd.read_parquet(tmp_path / "out" / "ticker_cik.parquet")
    rows = {str(r.start.date()): (int(r.cik), r.source) for r in accepted.itertuples()}
    assert rows == {"1990-01-01": (4281, "name_match"), "2021-12-20": (1675149, "sec_current_ticker")}


def test_revision_before_reads_one_revision_and_fails_loud_otherwise():
    payload = {"query": {"pages": {"1": {"revisions": [{"revid": 1364118843, "timestamp": "2026-07-14T16:34:24Z"}]}}}}
    assert md.revision_before(payload) == (1364118843, "2026-07-14T16:34:24Z")
    with pytest.raises(edgar.EdgarError):
        md.revision_before({"query": {"pages": {"1": {"missing": ""}}}})


def test_constituent_rows_and_page_cik_candidate(tmp_path):
    tables = [[["Symbol", "Security", "GICS Sector", "CIK"], ["EA", "Electronic Arts", "Comm", "0000712515"],
               ["X", "No Cik Co", "Ind", ""]]]
    members = md.constituent_rows(tables, "sp500")
    assert members.set_index("ticker").loc["EA", "cik"] == 712515
    assert pd.isna(members.set_index("ticker").loc["X", "cik"])

    subs = {712515: _submissions(712515, "ELECTRONIC ARTS INC.", ["2024-05-01"])}

    def transport(url, headers, timeout):
        for cik, payload in subs.items():
            if url == edgar.SUBMISSIONS_URL.format(cik=cik):
                return json.dumps(payload).encode()
        raise AssertionError(url)

    client = edgar.SecClient(UA, tmp_path, transport=transport, clock=lambda: 0.0, sleep=lambda s: None)
    coverage = {"not_accepted": [{"ticker": "EA", "start": "1990-01-01", "end": "2026-06-16", "status": "unresolved"}]}
    empty_changes = pd.DataFrame({"index": [], "date": pd.to_datetime([]), "added": [], "added_name": [],
                                  "removed": [], "removed_name": []})
    (result,) = md.match(coverage, empty_changes, b"", client, constituents=members)
    assert result["decision"] == "accepted" and result["cik"] == 712515
