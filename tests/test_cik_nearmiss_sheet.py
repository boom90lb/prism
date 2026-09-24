"""Offline tests for near-miss CIK proposals (prism.scripts.cik_nearmiss_sheet).

Each rule is pinned on the case that motivated it: SEC's inverted names (CR Bard,
JCPenney), run-together names (Sotheby's → SOTHEBYS), abbreviations (U.S. Steel),
truncated names (Taylor Morrison), a former name outside the interval (iStar, once
Starwood Financial), and the ticker-as-name rule that must stay WEAK (TSS, Inc. is not
TSYS).
"""

import pandas as pd
import pytest

from prism.scripts import cik_nearmiss_sheet as nm


@pytest.mark.parametrize("name,expected", [
    ("CR Bard", ["CR", "BARD"]),
    ("BARD C R INC /NJ/", ["BARD", "CR"]),
    ("U.S. Steel", ["UNITED", "STATES", "STEEL"]),
    ("Victoria's Secret & Co.", ["VICTORIA", "S", "SECRET"]),
    ("HEINZ H J CO", ["HEINZ", "HJ"]),
])
def test_words(name, expected):
    assert nm.words(name) == expected


@pytest.mark.parametrize("wiki,sec,ticker,rule", [
    ("CR Bard", "BARD C R INC", "BCR", "same_letters"),
    ("JCPenney", "J C PENNEY CO INC", "JCP", "same_letters"),
    ("Sotheby's", "SOTHEBYS", "BID", "same_letters"),
    ("U.S. Steel", "UNITED STATES STEEL CORP", "X", "same_letters"),
    ("Taylor Morrison", "TAYLOR MORRISON HOME CORP", "TMHC", "all_words"),
    ("Webster Bank", "WEBSTER FINANCIAL CORP", "WBS", "first_word"),
    ("Suntory Global Spirits", "BEAM INC", "BEAM", "ticker_name"),
    ("Heinz", "KRAFT FOODS", "HNZ", None),
])
def test_closeness_rules(wiki, sec, ticker, rule):
    result = nm.closeness(nm.words(wiki), nm.words(sec), ticker)
    assert (result[1] if result else None) == rule


def test_close_candidates_reach_run_together_and_inverted_names():
    index = b"SOTHEBYS:0000823094:\nSOTHEBY'S INTERNATIONAL REALTY, INC.:0001420935:\nBARD C R INC /NJ/:0000009892:\n"
    found = nm.close_candidates(index, {("BID", "1990-01-01", "2019-10-03"): ["Sotheby's"],
                                        ("BCR", "1990-01-01", "2018-01-03"): ["CR Bard"]})
    bid = found[("BID", "1990-01-01", "2019-10-03")]
    assert bid[0]["cik"] == 823094 and bid[0]["rule"] == "same_letters"
    assert {c["cik"] for c in bid} == {823094, 1420935}
    assert [c["cik"] for c in found[("BCR", "1990-01-01", "2018-01-03")]] == [9892]


def test_strength_requires_a_name_held_during_the_interval():
    assert nm.strength_during(["ISTAR INC.", "ISTAR FINANCIAL INC"], ["Starwood"], "HOT") is None
    assert nm.strength_during(["STARWOOD HOTEL & RESORTS WORLDWIDE INC"], ["Starwood"], "HOT") == (3, "all_words")


def _row(cik, rank, rule, fit=False):
    return {"cik": cik, "name_now": f"co {cik}", "names_during_interval": [f"co {cik}"], "strength_rank": rank,
            "strength_rule": rule, "subsidiary_form_only": False, "ticker_fit": fit,
            "ticker_fit_from": "2010-01-01" if fit else None, "ticker_fit_to": "2020-01-01" if fit else None,
            "wikipedia_name_at_time": False, "first_in_interval": "2010-02-01", "last_in_interval": "2019-11-01",
            "last_periodic_overall": "2025-01-01", "covers_start": True, "covers_end": True}


def test_decisions_order_and_weak_label():
    start, end = pd.Timestamp("2009-06-15"), pd.Timestamp("2020-01-01")
    single_ticker_name = nm.decide_near_miss("TSS", start, end, [_row(1320760, 1, "ticker_name", fit=True)])
    assert single_ticker_name["basis"].startswith("WEAK")
    strongest = nm.decide_near_miss("X", start, end, [_row(1163302, 4, "same_letters"), _row(27093, 2, "first_word")])
    assert strongest["cik"] == 1163302
    # the review rules' ticker fit picks one candidate and the name rule the other: judgment
    clash = nm.decide_near_miss("X", start, end, [_row(1, 4, "same_letters"), _row(2, 2, "first_word", fit=True)])
    assert clash["kind"] == "none" and "disagree" in clash["basis"]
    assert nm.decide_near_miss("X", start, end, [])["kind"] == "none"
