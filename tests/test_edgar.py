"""Offline tests for the EDGAR fundamentals layer (prism.io.edgar).

They pin the point-in-time contract (a fact is usable at a decision close only
once its filing was accepted, and a restatement replaces the original only from
its own acceptance), the quarter and trailing-twelve-month arithmetic, and the
client's fail-loud behaviour. No network.
"""

import email.message
import gzip
import json
import urllib.error

import pandas as pd
import pytest

from prism.io import edgar

UA = "Prism research test@example.com"


def _fact(end, val, accn, filed, start=None, fy=2024, fp="FY", form="10-K"):
    entry = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start:
        entry["start"] = start
    return entry


def _payload(concepts, cik=1234):
    return {"cik": cik, "entityName": "Test Co", "facts": concepts}


# --------------------------------------------------------------------- client


def test_user_agent_requires_contact_email():
    with pytest.raises(edgar.EdgarError):
        edgar.user_agent_from_env({})
    with pytest.raises(edgar.EdgarError):
        edgar.user_agent_from_env({edgar.USER_AGENT_ENV: "no email here"})
    assert edgar.user_agent_from_env({edgar.USER_AGENT_ENV: UA}) == UA


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _client(tmp_path, transport, clock=None):
    clock = clock or _Clock()
    return edgar.SecClient(UA, tmp_path, transport=transport, clock=clock.time, sleep=clock.sleep), clock


def test_client_spaces_requests_and_caches(tmp_path):
    calls = []

    def transport(url, headers, timeout):
        calls.append((url, headers["User-Agent"]))
        return json.dumps({"cik": 1, "n": len(calls)}).encode()

    client, clock = _client(tmp_path, transport)
    first, meta = client.get_json("https://x/a", "a")
    client.get_json("https://x/b", "b")
    assert calls == [("https://x/a", UA), ("https://x/b", UA)]
    assert clock.sleeps and clock.sleeps[0] == pytest.approx(1 / edgar.MAX_REQUESTS_PER_SECOND)
    again, _ = client.get_json("https://x/a", "a")  # served from cache
    assert again == first and len(calls) == 2
    assert meta["sha256"] and meta["url"] == "https://x/a"


def test_client_refuses_tampered_cache(tmp_path):
    client, _ = _client(tmp_path, lambda url, headers, timeout: b'{"cik": 1}')
    client.get_json("https://x/a", "a")
    (tmp_path / "a.json.gz").write_bytes(gzip.compress(b'{"cik": 2}'))
    with pytest.raises(edgar.EdgarError):
        client.get_json("https://x/a", "a")


def _http_error(code):
    return urllib.error.HTTPError("https://x", code, "err", email.message.Message(), None)


def test_client_retries_rate_limits_then_succeeds(tmp_path):
    outcomes = [_http_error(429), _http_error(503), b'{"cik": 1}']

    def transport(url, headers, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client, clock = _client(tmp_path, transport)
    payload, _ = client.get_json("https://x/a", "a")
    assert payload == {"cik": 1}
    assert 1.0 in clock.sleeps and 2.0 in clock.sleeps  # exponential backoff


@pytest.mark.parametrize("code,error", [(403, edgar.EdgarError), (404, edgar.EdgarNotFound)])
def test_client_fails_loud_on_refusal_and_absence(tmp_path, code, error):
    def transport(url, headers, timeout):
        raise _http_error(code)

    client, _ = _client(tmp_path, transport)
    with pytest.raises(error):
        client.get_json("https://x/a", "a")


# --------------------------------------------------------------------- normalize


def test_companyfacts_frame_keeps_every_filing_and_filters_concepts():
    payload = _payload({
        "us-gaap": {
            "Assets": {"units": {"USD": [
                _fact("2023-12-31", 100.0, "A-1", "2024-02-10"),
                _fact("2023-12-31", 90.0, "A-2", "2024-05-01", fp="Q1", form="10-Q"),  # restated
            ]}},
            "Unwanted": {"units": {"USD": [_fact("2023-12-31", 1.0, "A-1", "2024-02-10")]}},
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            _fact("2024-02-01", 10.0, "A-1", "2024-02-10")]}}},
        "ifrs-full": {"Assets": {"units": {"USD": [_fact("2023-12-31", 5.0, "A-9", "2024-02-10")]}}},
    })
    frame = edgar.companyfacts_frame(payload, concepts={"Assets", "EntityCommonStockSharesOutstanding"})
    assert set(frame["concept"]) == {"Assets", "EntityCommonStockSharesOutstanding"}
    assert len(frame[frame["concept"] == "Assets"]) == 2
    assert set(frame["taxonomy"]) == {"us-gaap", "dei"}
    assert frame["filed"].dtype.kind == "M"


def test_companyfacts_frame_fails_loud_on_malformed_fact():
    payload = _payload({"us-gaap": {"Assets": {"units": {"USD": [{"end": "2023-12-31", "val": 1.0}]}}}})
    with pytest.raises(edgar.EdgarError):
        edgar.companyfacts_frame(payload)


def _submissions():
    return {
        "cik": 1234,
        "filings": {"recent": {
            "accessionNumber": ["A-3", "A-2", "E-1"],
            "form": ["10-K", "10-Q", "8-K"],
            "filingDate": ["2025-02-10", "2024-05-01", "2024-07-30"],
            "acceptanceDateTime": ["2025-02-10T11:00:00.000Z", "2024-05-01T20:30:00.000Z",
                                   "2024-07-30T20:30:28.000Z"],
            "reportDate": ["2024-12-31", "2024-03-31", "2024-07-30"],
            "items": ["", "", "2.02,9.01"],
        }},
    }


def test_submissions_acceptance_is_utc_converted_to_new_york():
    filings = edgar.submissions_filings(_submissions())
    eightk = filings[filings["accn"] == "E-1"].iloc[0]
    assert eightk["acceptance"] == pd.Timestamp("2024-07-30 16:30:28", tz="America/New_York")
    releases = edgar.earnings_releases(filings)
    assert list(releases["accn"]) == ["E-1"]


def test_earnings_releases_matches_item_exactly():
    filings = pd.DataFrame({"cik": [1, 1], "accn": ["x", "y"], "form": ["8-K", "8-K"],
                            "filing_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                            "acceptance": pd.to_datetime(["2024-01-01", "2024-01-02"]).tz_localize("UTC"),
                            "items": ["12.02", "2.02"]})
    assert list(edgar.earnings_releases(filings)["accn"]) == ["y"]


# --------------------------------------------------------------------- point in time


def _facts_with_availability():
    payload = _payload({"us-gaap": {"Assets": {"units": {"USD": [
        _fact("2023-12-31", 100.0, "A-1", "2024-02-10"),  # accn unknown to submissions: filed-date fallback
        _fact("2023-12-31", 90.0, "A-2", "2024-05-01", fp="Q1", form="10-Q"),  # accepted 16:30 ET
    ]}}}})
    facts = edgar.companyfacts_frame(payload)
    return edgar.attach_availability(facts, edgar.submissions_filings(_submissions()))


def test_availability_uses_acceptance_or_end_of_filing_day():
    facts = _facts_with_availability().set_index("accn")
    assert facts.loc["A-2", "available_at"] == pd.Timestamp("2024-05-01 16:30", tz="America/New_York")
    assert facts.loc["A-2", "availability_basis"] == "acceptance"
    assert facts.loc["A-1", "available_at"] == pd.Timestamp("2024-02-10 23:59:59", tz="America/New_York")
    assert facts.loc["A-1", "availability_basis"] == "filed_date_end"


def test_point_in_time_respects_acceptance_and_restatement():
    facts = _facts_with_availability()
    assert edgar.point_in_time(facts, edgar.session_close("2024-02-10")).empty  # filed that day: not at its close
    assert edgar.point_in_time(facts, edgar.session_close("2024-02-12"))["val"].tolist() == [100.0]
    # the restating 10-Q was accepted at 16:30, after the close of its filing day
    assert edgar.point_in_time(facts, edgar.session_close("2024-05-01"))["val"].tolist() == [100.0]
    assert edgar.point_in_time(facts, edgar.session_close("2024-05-02"))["val"].tolist() == [90.0]


def test_point_in_time_requires_aware_timestamp():
    with pytest.raises(edgar.EdgarError):
        edgar.point_in_time(_facts_with_availability(), pd.Timestamp("2024-06-01"))


# --------------------------------------------------------------------- fields and periods


def _pit(entries_by_concept):
    concepts = {c: {"units": {"USD": entries}} for c, entries in entries_by_concept.items()}
    facts = edgar.companyfacts_frame(_payload({"us-gaap": concepts}))
    facts = edgar.attach_availability(facts, edgar.submissions_filings(_submissions()))
    return edgar.point_in_time(facts, edgar.session_close("2030-01-02"))


def test_alias_resolution_per_period():
    pit = _pit({
        "SalesRevenueNet": [_fact("2017-12-31", 10.0, "B-1", "2018-02-01", start="2017-01-01")],
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            _fact("2018-12-31", 12.0, "B-2", "2019-02-01", start="2018-01-01")],
        "Revenues": [_fact("2018-12-31", 99.0, "B-2", "2019-02-01", start="2018-01-01")],
    })
    revenue = edgar.field_facts(pit, "revenue").set_index("end")["val"]
    assert revenue[pd.Timestamp("2017-12-31")] == 10.0
    assert revenue[pd.Timestamp("2018-12-31")] == 12.0  # the preferred alias wins where both report


def test_quarters_derive_q4_from_annual_and_cash_flow_from_year_to_date():
    pit = _pit({
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            _fact("2024-03-31", 10.0, "C-1", "2024-04-30", start="2024-01-01", fp="Q1", form="10-Q"),
            _fact("2024-06-30", 11.0, "C-2", "2024-07-30", start="2024-04-01", fp="Q2", form="10-Q"),
            _fact("2024-09-30", 12.0, "C-3", "2024-10-30", start="2024-07-01", fp="Q3", form="10-Q"),
            _fact("2024-09-30", 33.0, "C-3", "2024-10-30", start="2024-01-01", fp="Q3", form="10-Q"),
            _fact("2024-12-31", 46.0, "C-4", "2025-02-10", start="2024-01-01"),
        ],
        "NetCashProvidedByUsedInOperatingActivities": [
            _fact("2024-03-31", 5.0, "C-1", "2024-04-30", start="2024-01-01", fp="Q1", form="10-Q"),
            _fact("2024-06-30", 9.0, "C-2", "2024-07-30", start="2024-01-01", fp="Q2", form="10-Q"),
            _fact("2024-09-30", 15.0, "C-3", "2024-10-30", start="2024-01-01", fp="Q3", form="10-Q"),
            _fact("2024-12-31", 22.0, "C-4", "2025-02-10", start="2024-01-01"),
        ],
    })
    revenue_q = edgar.quarterly_values(edgar.field_facts(pit, "revenue")).set_index("end")
    assert revenue_q.loc[pd.Timestamp("2024-12-31"), "val"] == pytest.approx(13.0)  # 46 − 33
    assert revenue_q.loc[pd.Timestamp("2024-12-31"), "basis"] == "derived"
    assert revenue_q.loc[pd.Timestamp("2024-06-30"), "basis"] == "reported"
    cfo_q = edgar.quarterly_values(edgar.field_facts(pit, "cfo")).set_index("end")["val"]
    assert cfo_q.tolist() == pytest.approx([5.0, 4.0, 6.0, 7.0])

    ttm = edgar.trailing_twelve_months(pit, "revenue").iloc[0]
    assert ttm["ttm"] == pytest.approx(46.0)
    assert ttm["period_end"] == pd.Timestamp("2024-12-31")
    assert ttm["derived_quarters"] == 1


def test_ttm_absent_when_quarters_are_not_contiguous():
    pit = _pit({"RevenueFromContractWithCustomerExcludingAssessedTax": [
        _fact("2023-03-31", 1.0, "D-1", "2023-04-30", start="2023-01-01"),
        _fact("2023-06-30", 1.0, "D-2", "2023-07-30", start="2023-04-01"),
        _fact("2023-09-30", 1.0, "D-3", "2023-10-30", start="2023-07-01"),
        _fact("2024-06-30", 1.0, "D-4", "2024-07-30", start="2024-04-01"),  # a year missing
    ]})
    assert edgar.trailing_twelve_months(pit, "revenue").empty


def test_latest_instant_takes_latest_period():
    pit = _pit({"Assets": [_fact("2023-12-31", 100.0, "E-1", "2024-02-10"),
                           _fact("2024-03-31", 110.0, "E-2", "2024-05-01")]})
    row = edgar.latest_instant(pit, "assets").iloc[0]
    assert row["value"] == 110.0 and row["period_end"] == pd.Timestamp("2024-03-31")
