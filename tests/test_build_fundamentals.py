"""Offline tests for the EDGAR fundamentals builder (prism.scripts.build_fundamentals).

A fake SEC transport serves a ticker map, filing histories and company facts, so
the tests pin the identity rules: a current ticker is accepted for a membership
interval only if its CIK filed periodic reports inside it (ticker reuse is
refused), reviewed renames resolve through their successor, share-class
spellings match, unmapped names are ledgered rather than guessed, and missing
company facts are recorded.
"""

import email.message
import json
import urllib.error

import pandas as pd
import pytest

from prism.io import edgar
from prism.scripts import build_fundamentals as bf

UA = "Prism research test@example.com"

TICKERS = {
    "0": {"cik_str": 1, "ticker": "AAA", "title": "Alpha Corp"},
    "1": {"cik_str": 2, "ticker": "BBB", "title": "Beta Newco"},  # the ticker's current holder, filing since 2020
    "2": {"cik_str": 3, "ticker": "META", "title": "Meta Platforms"},
    "3": {"cik_str": 4, "ticker": "BRK-B", "title": "Berkshire Hathaway"},
}


def _submissions(cik, dates):
    return {"cik": cik, "filings": {"recent": {
        "accessionNumber": [f"{cik}-{i}" for i in range(len(dates))],
        "form": ["10-Q"] * len(dates),
        "filingDate": dates,
        "acceptanceDateTime": [f"{d}T12:00:00.000Z" for d in dates],
        "reportDate": dates,
        "items": [""] * len(dates),
    }, "files": []}}


def _facts(cik):
    return {"cik": cik, "entityName": "x", "facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"end": "2016-12-31", "val": 10.0, "accn": f"{cik}-0", "fy": 2016, "fp": "FY", "form": "10-K",
         "filed": "2017-02-01"}]}}}}}


def _transport(url, headers, timeout):
    assert headers["User-Agent"] == UA
    if url == edgar.TICKERS_URL:
        return json.dumps(TICKERS).encode()
    subs = {1: ["2017-02-01", "2020-05-01"], 2: ["2021-05-01"], 3: ["2014-05-01", "2023-05-01"],
            4: ["2012-05-01"]}
    for cik, dates in subs.items():
        if url == edgar.SUBMISSIONS_URL.format(cik=cik):
            return json.dumps(_submissions(cik, dates)).encode()
        if url == edgar.COMPANYFACTS_URL.format(cik=cik):
            if cik == 4:
                raise urllib.error.HTTPError(url, 404, "nf", email.message.Message(), None)
            return json.dumps(_facts(cik)).encode()
    raise AssertionError(f"unexpected url {url}")


MEMBERSHIP = pd.DataFrame({
    "ticker": ["AAA", "BBB", "FB", "OLD", "GONE", "BRK.B"],
    "start": pd.to_datetime(["2015-01-01", "2010-01-01", "2013-12-23", "1995-01-01", "2012-01-01", "2010-02-16"]),
    "end": pd.to_datetime(["2026-09-23", "2014-06-01", "2022-06-09", "2005-01-01", "2018-01-01", "2026-09-23"]),
})


@pytest.fixture
def built(tmp_path):
    client = edgar.SecClient(UA, tmp_path / "raw", transport=_transport, clock=lambda: 0.0, sleep=lambda s: None)
    ledger = bf.build(MEMBERSHIP, client, tmp_path / "out", tmp_path / "ledger.json", "2026-09-24")
    return ledger, tmp_path


def test_identity_rules(built):
    ledger, tmp_path = built
    accepted = pd.read_parquet(tmp_path / "out" / "ticker_cik.parquet").set_index("ticker")
    assert accepted.loc["AAA", "cik"] == 1
    assert accepted.loc["FB", "cik"] == 3 and accepted.loc["FB", "source"] == "rename_successor:META"
    assert accepted.loc["BRK.B", "cik"] == 4  # dot spelling matches SEC's hyphen
    assert "BBB" not in accepted.index  # reused ticker: CIK 2 filed nothing while BBB was a member
    statuses = {row["ticker"]: row["status"] for row in ledger["not_accepted"]}
    assert statuses == {"BBB": "no_periodic_filing_in_interval", "GONE": "unresolved"}
    assert ledger["status_counts"]["pre_xbrl"] == 1  # OLD left before XBRL; not a coverage gap
    assert ledger["xbrl_era_tickers"] == 5 and ledger["accepted_tickers"] == 3
    assert ledger["coverage_xbrl_era"] == pytest.approx(0.6)
    assert ledger["ciks_without_companyfacts"] == [4]


def test_facts_carry_acceptance_availability(built):
    _, tmp_path = built
    facts = pd.read_parquet(tmp_path / "out" / "facts.parquet")
    assert set(facts["cik"]) == {1, 3}
    row = facts[facts["cik"] == 1].iloc[0]
    assert row["availability_basis"] == "acceptance"
    assert row["available_at"] == pd.Timestamp("2017-02-01 07:00", tz="America/New_York")
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert set(manifest["companyfacts"]) == {"1", "3"}


def test_recent_joiner_accepted_by_prior_quarter_filing():
    filings = pd.DataFrame({"form": ["10-Q"], "filing_date": pd.to_datetime(["2026-07-25"])})
    assert bf.interval_status(filings, pd.Timestamp("2026-08-20"), pd.Timestamp("2026-09-23")) == "accepted"
    # a reused ticker's new holder that began filing after the old interval ended stays refused
    later = pd.DataFrame({"form": ["10-K"], "filing_date": pd.to_datetime(["2017-02-28"])})
    assert bf.interval_status(later, pd.Timestamp("1990-01-01"), pd.Timestamp("2016-11-01")) == \
        "no_periodic_filing_in_interval"
