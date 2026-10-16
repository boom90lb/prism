"""Point-in-time S&P index membership reconstruction (Koker method).

The headline invariant under test: names removed from the index still appear in
the ever-member union, and if their prices are unavailable they are *counted* in
the coverage skip-list -- survivorship is measured, never silently dropped.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prism.io.universe_file import load_universe_symbols
from prism.io.universe_sp500 import (
    CHANGES_PATCH_TABLE,
    HISTORY_FLOOR,
    QUARANTINE_TABLE,
    RENAME_TABLE,
    CoverageReport,
    apply_changes_patches,
    build_membership_mask,
    compute_coverage,
    ever_members,
    extract_tables,
    members_active_between,
    parse_changes_table,
    parse_constituents_table,
    reconstruct_membership,
    write_universe_file,
)

END = pd.Timestamp("2026-01-01")


def _changes(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "added", "removed"])


def _interval(df: pd.DataFrame, ticker: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    g = df[df["ticker"] == ticker].sort_values("start")
    return [(pd.Timestamp(r.start), pd.Timestamp(r.end)) for r in g.itertuples(index=False)]


def test_member_never_in_changes_spans_full_history() -> None:
    m = reconstruct_membership(["A"], _changes([]), end_date=END)
    assert _interval(m, "A") == [(HISTORY_FLOOR, END)]


def test_added_member_and_displaced_member_intervals() -> None:
    # B replaced X on 2018-06-01: X member until then, B member after.
    changes = _changes([("2018-06-01", "B", "X")])
    m = reconstruct_membership(["A", "B"], changes, end_date=END)
    assert _interval(m, "A") == [(HISTORY_FLOOR, END)]
    assert _interval(m, "B") == [(pd.Timestamp("2018-06-01"), END)]
    assert _interval(m, "X") == [(HISTORY_FLOOR, pd.Timestamp("2018-06-01"))]


def test_readdition_produces_two_disjoint_intervals() -> None:
    # A: added 2010, removed 2015, added back 2020; current member.
    changes = _changes(
        [("2010-01-04", "A", ""), ("2015-03-02", "", "A"), ("2020-09-01", "A", "")]
    )
    m = reconstruct_membership(["A"], changes, end_date=END)
    assert _interval(m, "A") == [
        (pd.Timestamp("2010-01-04"), pd.Timestamp("2015-03-02")),
        (pd.Timestamp("2020-09-01"), END),
    ]


def test_ever_members_includes_removed_names_survivorship() -> None:
    changes = _changes([("2018-06-01", "B", "X")])
    m = reconstruct_membership(["A", "B"], changes, end_date=END)
    # X left the index in 2018 but MUST remain in the universe.
    assert ever_members(m) == ["A", "B", "X"]


def test_members_active_between_scopes_to_window() -> None:
    # X left 2018, B joined 2018, A always. A 2020+ window excludes the
    # pre-window casualty X and keeps every name that was a member during it.
    changes = _changes([("2018-06-01", "B", "X")])
    m = reconstruct_membership(["A", "B"], changes, end_date=END)
    assert members_active_between(m, "2020-01-01", "2026-01-01") == ["A", "B"]
    # A window that straddles the 2018 change still includes X (member then).
    assert "X" in members_active_between(m, "2017-01-01", "2018-12-31")


def test_inconsistent_events_are_skipped_not_fatal() -> None:
    # Two genuinely inconsistent events that must be skipped (not crash, not
    # spawn phantom intervals): a duplicate add of an already-active member (C
    # at 2016), and a second removal of an already-removed name (Q at 2019).
    changes = _changes(
        [
            ("2015-01-05", "C", ""),
            ("2016-06-01", "C", ""),  # C already active -> duplicate add skipped
            ("2018-03-01", "", "Q"),
            ("2019-04-01", "", "Q"),  # Q already removed -> double removal skipped
        ]
    )
    m = reconstruct_membership(["A", "C"], changes, end_date=END)
    assert _interval(m, "A") == [(HISTORY_FLOOR, END)]
    assert _interval(m, "C") == [(pd.Timestamp("2015-01-05"), END)]  # single, not doubled
    assert _interval(m, "Q") == [(HISTORY_FLOOR, pd.Timestamp("2018-03-01"))]  # closed once


def test_future_dated_addition_excluded_as_of_horizon() -> None:
    # A change announced for after the horizon (END = 2026): the incoming name is
    # not yet a member (no inverted interval); the outgoing name still is.
    changes = _changes([("2030-01-01", "NEW", "OLD")])
    m = reconstruct_membership(["A"], changes, end_date=END)
    assert "NEW" not in set(m["ticker"])
    assert _interval(m, "OLD") == [(HISTORY_FLOOR, pd.Timestamp("2030-01-01"))]
    assert "NEW" not in set(ever_members(m))


def test_reconstruct_accepts_tz_aware_end_date() -> None:
    # Regression: a tz-aware horizon (pd.Timestamp.utcnow()) must not clash with
    # tz-naive Wikipedia change dates.
    changes = _changes([("2018-06-01", "B", "X")])
    m = reconstruct_membership(["A", "B"], changes, end_date=pd.Timestamp("2026-01-01", tz="UTC"))
    assert set(ever_members(m)) == {"A", "B", "X"}
    assert (m["start"] <= m["end"]).all()


def test_membership_mask_is_time_varying_and_tz_safe() -> None:
    changes = _changes([("2018-06-01", "B", "X")])
    m = reconstruct_membership(["A", "B"], changes, end_date=END)
    idx = pd.date_range("2018-01-01", "2018-12-31", freq="B", tz="America/New_York")
    mask = build_membership_mask(m, idx, ["A", "B", "X"])

    early = pd.Timestamp("2018-03-01", tz="America/New_York")
    late = pd.Timestamp("2018-09-03", tz="America/New_York")
    assert mask.loc[early, "A"] and mask.loc[early, "X"] and not mask.loc[early, "B"]
    assert mask.loc[late, "A"] and mask.loc[late, "B"] and not mask.loc[late, "X"]
    # A name absent from the universe altogether is never a member.
    mask2 = build_membership_mask(m, idx, ["A", "ZZZZ"])
    assert not mask2["ZZZZ"].any()


def test_coverage_counts_unavailable_delisted_names() -> None:
    ever = ["A", "B", "X"]
    # X is delisted; the price vendor returned no bars for it.
    report = compute_coverage(ever, available=["A", "B"], asof="2026-01-01")
    assert isinstance(report, CoverageReport)
    assert report.skipped == ["X"]
    assert report.resolved == ["A", "B"]
    assert report.n_ever_members == 3 and report.n_resolved == 2
    assert report.coverage_fraction == pytest.approx(2 / 3)


def test_coverage_applies_rename_table() -> None:
    report = compute_coverage(
        ["BRK.B"], available=["BRK/B"], asof="2026-01-01", rename_table={"BRK.B": "BRK/B"}
    )
    assert report.resolved == ["BRK.B"] and report.skipped == []


def test_coverage_quarantine_never_resolves_even_when_vendor_answers() -> None:
    # The vendor "answers" for B, but the answer is a known wrong instrument:
    # quarantine wins, and the exclusion is counted (skip-list) and auditable.
    report = compute_coverage(
        ["A", "B"], available=["A", "B"], asof="2026-01-01", quarantine_table={"B": "wrong instrument"}
    )
    assert report.resolved == ["A"]
    assert report.skipped == ["B"]
    assert report.quarantined == {"B": "wrong instrument"}
    assert report.n_ever_members == 2 and report.n_resolved == 1
    assert report.to_dict()["quarantined"] == {"B": "wrong instrument"}


def test_seeded_tables_carry_the_reviewed_2026_07_entries() -> None:
    # Pins the reviewed seed content (docs/data_integrity_diagnostic.md §6; ECHO
    # moved quarantine -> rename after the §8 identity verification, §9 executed)
    # so accidental edits are loud; any legitimate change re-reviews and updates here.
    assert RENAME_TABLE == {"FB": "META", "PCLN": "BKNG", "WLTW": "WTW", "SATS": "ECHO"}
    assert set(QUARANTINE_TABLE) == {"ACT", "ADS", "INFO", "SBNY", "SOLS"}
    assert all(reason for reason in QUARANTINE_TABLE.values())
    # §10: the one reviewed changes-table patch -- the ACT -> AGN rename event.
    assert [(p["date"], p["added"], p["removed"]) for p in CHANGES_PATCH_TABLE] == [
        ("2015-06-15", "AGN", "ACT")
    ]


def test_apply_changes_patches_appends_without_mutating_input() -> None:
    changes = _changes([("1999-04-12", "ACT", "")])
    out = apply_changes_patches(changes)
    assert len(out) == len(changes) + len(CHANGES_PATCH_TABLE)
    assert len(changes) == 1  # input untouched
    assert apply_changes_patches(changes, patches=()) is changes  # empty table = identity
    patched_row = out.iloc[-1]
    assert patched_row["added"] == "AGN" and patched_row["removed"] == "ACT"


def test_act_agn_rename_patch_closes_the_chain_seat() -> None:
    # The §10 defect in miniature. Wikipedia records the Watson/Actavis/Allergan
    # seat's 1999-04-12 addition under ACT (backfilled chain ticker) and its
    # 2020-05-12 removal under AGN (post-rename ticker); legacy Allergan Inc.
    # closes separately on 2015-03-23. Unpatched, the 2020 removal is dropped as
    # inconsistent and ACT runs open forever.
    changes = _changes(
        [
            ("1999-04-12", "ACT", ""),
            ("2015-03-23", "AAL", "AGN"),
            ("2020-05-12", "DXCM", "AGN"),
        ]
    )
    current = ["AAL", "DXCM"]

    unpatched = reconstruct_membership(current, changes, end_date=END)
    assert _interval(unpatched, "ACT") == [(pd.Timestamp("1999-04-12"), END)]  # the defect
    assert _interval(unpatched, "AGN") == [(HISTORY_FLOOR, pd.Timestamp("2015-03-23"))]

    patched = reconstruct_membership(current, apply_changes_patches(changes), end_date=END)
    assert _interval(patched, "ACT") == [
        (pd.Timestamp("1999-04-12"), pd.Timestamp("2015-06-15"))
    ]
    assert _interval(patched, "AGN") == [
        (HISTORY_FLOOR, pd.Timestamp("2015-03-23")),  # legacy Allergan Inc., untouched
        (pd.Timestamp("2015-06-15"), pd.Timestamp("2020-05-12")),  # Allergan plc
    ]
    # Bystanders unaffected by the patch.
    assert _interval(patched, "AAL") == _interval(unpatched, "AAL")
    assert _interval(patched, "DXCM") == _interval(unpatched, "DXCM")


CONSTITUENTS_HTML = """
<table class="wikitable" id="constituents">
  <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
  <tr><td>MMM</td><td>3M</td><td>Industrials</td></tr>
  <tr><td>AOS</td><td>A. O. Smith</td><td>Industrials</td></tr>
  <tr><td>NVDA</td><td>Nvidia<sup>[1]</sup></td><td>Information Technology</td></tr>
</table>
<table class="wikitable" id="changes">
  <tr><th>Date</th><th>Added</th><th>Removed</th><th>Reason</th></tr>
  <tr><th></th><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th><th></th></tr>
  <tr><td>June 20, 2018</td><td>B</td><td>Beta Co</td><td>X</td><td>Xenon Inc</td><td>M&amp;A</td></tr>
  <tr><td>March 2, 2015</td><td></td><td></td><td>A</td><td>Alpha</td><td>Bankruptcy[2]</td></tr>
</table>
"""


def test_extract_and_parse_wikipedia_tables() -> None:
    tables = extract_tables(CONSTITUENTS_HTML)
    assert parse_constituents_table(tables) == ["MMM", "AOS", "NVDA"]
    changes = parse_changes_table(tables)
    # Two parseable change rows; sub-header ('Ticker'/'Security') row dropped.
    assert list(changes.columns) == ["date", "added", "removed"]
    assert len(changes) == 2
    first = changes.iloc[0]
    assert first["date"] == pd.Timestamp("2018-06-20")
    assert first["added"] == "B" and first["removed"] == "X"


def test_end_to_end_parse_then_reconstruct() -> None:
    tables = extract_tables(CONSTITUENTS_HTML)
    current = parse_constituents_table(tables)
    changes = parse_changes_table(tables)
    m = reconstruct_membership(current, changes, end_date=END)
    union = set(ever_members(m))
    # Current names plus both removed names survive into the universe.
    assert {"MMM", "AOS", "NVDA", "X", "A"} <= union


def test_write_universe_file_matches_load_universe_contract(tmp_path) -> None:
    path = tmp_path / "sp500_pit_2026-01-01.txt"
    write_universe_file(["B", "A", "x", "A"], path, index="sp500", asof="2026-01-01")
    # Mirror research.scripts.training.load_universe: strip '#' comments, dedupe in order.
    loaded: list[str] = []
    for line in path.read_text().splitlines():
        token = line.split("#", 1)[0].strip()
        if token and token not in loaded:
            loaded.append(token)
    assert loaded == ["A", "B", "X"]
    assert load_universe_symbols(path) == ["A", "B", "X"]


def test_the_universe_header_names_the_index_it_was_built_from(tmp_path) -> None:
    """An S&P 400 build is labelled S&P 400, and no header claims to be survivorship-free."""
    from prism.io.universe_sp500 import CONSTITUENTS_ARTICLE, INDEX_NAME

    assert set(INDEX_NAME) == set(CONSTITUENTS_ARTICLE)
    for index, name in INDEX_NAME.items():
        path = tmp_path / f"{index}.txt"
        write_universe_file(["A"], path, index=index, asof="2026-01-01", source="wikipedia")
        header = [line for line in path.read_text().splitlines() if line.startswith("#")]
        assert header[0].startswith(f"# {name} universe as of 2026-01-01")
        assert f"'{CONSTITUENTS_ARTICLE[index]}'" in "\n".join(header)
        assert "Survivorship-bias-free" not in "\n".join(header)
        assert "# Not survivorship-free: the changes tables are selected rather than complete, and price" in header
        others = [other for other in INDEX_NAME.values() if other != name]
        assert not any(other in "\n".join(header) for other in others)

    with pytest.raises(ValueError, match="unknown index"):
        write_universe_file(["A"], tmp_path / "x.txt", index="sp600", asof="2026-01-01")
    assert not (tmp_path / "x.txt").exists()


def test_unrecorded_exits_close_at_the_earliest_evidence_and_unresolved_phantoms_drop():
    import pandas as pd

    from prism.io.universe_sp500 import resolve_unrecorded_exits

    membership = pd.DataFrame(
        [
            ("KEEP", "2015-01-01", "2026-09-23"),  # a current member: untouched
            ("PROM", "2016-01-01", "2026-09-23"),  # promoted, then demoted evidence later
            ("DEAD", "2018-01-01", "2026-09-23"),  # acquired: vendor series ends
            ("GONE", "2014-01-01", "2026-09-23"),  # no evidence at all
            ("OLD", "2012-01-01", "2019-06-30"),   # a recorded exit: untouched
            ("EARLY", "2017-01-01", "2026-09-23"),  # evidence predates the interval: ignored
        ],
        columns=["ticker", "start", "end"],
    )
    evidence = {
        "PROM": [(pd.Timestamp("2021-03-22"), "promoted_to_sp500"), (pd.Timestamp("2024-06-24"), "added_to_sp600")],
        "DEAD": [(pd.Timestamp("2022-10-05"), "vendor_last_bar")],
        "EARLY": [(pd.Timestamp("2016-05-01"), "added_to_sp600")],
    }
    resolved, ledger = resolve_unrecorded_exits(membership, ["KEEP"], asof="2026-09-23", evidence=evidence)
    ends = {row.ticker: row.end.date().isoformat() for row in resolved.itertuples()}
    assert ends == {"KEEP": "2026-09-23", "PROM": "2021-03-21", "DEAD": "2022-10-04", "OLD": "2019-06-30"}
    by = {entry["ticker"]: entry for entry in ledger}
    assert by["PROM"]["evidence"] == "promoted_to_sp500"
    assert by["GONE"]["evidence"] == "unresolved" and by["GONE"]["closed_at"] is None
    assert by["EARLY"]["evidence"] == "unresolved"
    assert set(by) == {"PROM", "DEAD", "GONE", "EARLY"}


def test_subtract_and_union_membership_respect_index_exclusivity():
    import pandas as pd

    from prism.io.universe_sp500 import subtract_membership, union_membership

    mid = pd.DataFrame(
        [("GAP", "1990-01-01", "2026-09-23"), ("ZZZ", "2020-01-01", "2020-12-31")], columns=["ticker", "start", "end"]
    )
    large = pd.DataFrame(
        [("GAP", "2000-01-01", "2022-02-02"), ("GAP", "2024-01-01", "2024-06-30")], columns=["ticker", "start", "end"]
    )
    clean = subtract_membership(mid, large)
    spans = [(r.ticker, r.start.date().isoformat(), r.end.date().isoformat()) for r in clean.itertuples()]
    assert spans == [
        ("GAP", "1990-01-01", "1999-12-31"),
        ("GAP", "2022-02-03", "2023-12-31"),
        ("GAP", "2024-07-01", "2026-09-23"),
        ("ZZZ", "2020-01-01", "2020-12-31"),
    ]
    union = union_membership(clean, large)
    spans = [(r.ticker, r.start.date().isoformat(), r.end.date().isoformat()) for r in union.itertuples()]
    assert spans == [("GAP", "1990-01-01", "2026-09-23"), ("ZZZ", "2020-01-01", "2020-12-31")]


# --------------------------------------------------------------------------- #
# Revision-pinned reads and upstream-deleted rows (2026-09-25)
# --------------------------------------------------------------------------- #

HISTORY_HTML = """
<table class="wikitable">
  <tr><th>Effective Date</th><th>Added</th><th>Removed</th><th>Reason</th><th>Refs</th></tr>
  <tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
  <tr><td>September 21, 2026</td><td>BE</td><td>Bloom Energy</td><td>TAP</td><td>Molson Coors</td><td>Cap</td><td></td></tr>
  <tr><td>December 2, 2013</td><td>ALLE |</td><td>Allegion</td><td>JCP |</td><td>JCPenney</td><td>Cap</td><td></td></tr>
</table>
"""


class _Response:
    def __init__(self, payload=None, text=""):
        self._payload, self.text = payload, text

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


def _wiki_get(revisions: dict[str, tuple[int, str]], pages: dict[int, str], calls: list[str]):
    """A stand-in for ``requests.get``: title -> (revid, html) through the API and index endpoints."""
    from prism.io.universe_sp500 import WIKI_API

    def get(url, params=None, timeout=None, headers=None):
        calls.append(url if params is None else f"{url}?{params['titles']}@{params['rvstart']}")
        if url == WIKI_API:
            revid, stamp = revisions[params["titles"]]
            return _Response({"query": {"pages": {"1": {"revisions": [{"revid": revid, "timestamp": stamp}]}}}})
        return _Response(text=pages[int(url.rsplit("oldid=", 1)[1])])

    return get


def test_normalize_ticker_drops_stray_table_pipes() -> None:
    from prism.io.universe_sp500 import normalize_ticker

    assert normalize_ticker("ALLE |") == "ALLE"
    assert normalize_ticker(" jcp| ") == "JCP"
    changes = parse_changes_table(extract_tables(HISTORY_HTML))
    assert set(changes["added"]) == {"BE", "ALLE"} and set(changes["removed"]) == {"TAP", "JCP"}


def test_changes_article_switches_on_the_move_date() -> None:
    from prism.io.universe_sp500 import changes_article

    assert changes_article("sp500", "2026-08-10") == "List of S&P 500 companies"
    assert changes_article("sp500", "2026-08-11") == "Historical components of the S&P 500"
    assert changes_article("sp500", "2027-03-24") == "Historical components of the S&P 500"
    assert changes_article("sp400", "2027-03-24") == "List of S&P 400 companies"


def test_fetch_index_tables_reads_both_articles_at_the_pinned_revision(tmp_path) -> None:
    from prism.io.universe_sp500 import fetch_index_tables

    calls: list[str] = []
    get = _wiki_get(
        {"List of S&P 500 companies": (11, "2027-03-20T00:00:00Z"),
         "Historical components of the S&P 500": (22, "2027-03-01T00:00:00Z")},
        {11: CONSTITUENTS_HTML, 22: HISTORY_HTML},
        calls,
    )
    current, changes, sources = fetch_index_tables("sp500", "2027-03-24", tmp_path, get=get)
    assert current == ["MMM", "AOS", "NVDA"]
    assert len(changes) == 2
    assert sources["constituents"]["revid"] == 11 and sources["changes"]["revid"] == 22
    # Pinned to the end of the as-of day, never the live page.
    assert all("@2027-03-24T23:59:59Z" in c for c in calls if "@" in c)
    assert (tmp_path / "wikipedia_Historical_components_of_the_S_P_500_rev22.html").exists()
    # A second read at the same as-of reuses the cached revisions: lookups only.
    calls.clear()
    fetch_index_tables("sp500", "2027-03-24", tmp_path, get=get)
    assert all("@" in c for c in calls)


def test_fetch_index_tables_before_the_move_reads_one_revision(tmp_path) -> None:
    from prism.io.universe_sp500 import fetch_index_tables

    calls: list[str] = []
    get = _wiki_get({"List of S&P 500 companies": (7, "2026-07-14T16:34:24Z")}, {7: CONSTITUENTS_HTML}, calls)
    current, changes, sources = fetch_index_tables("sp500", "2026-07-17", tmp_path, get=get)
    assert sources["changes"] is sources["constituents"]
    assert len(changes) == 2 and len([c for c in calls if "@" in c]) == 1


def test_revision_lookup_fails_loud_without_a_revision() -> None:
    from prism.io.universe_sp500 import revision_before

    with pytest.raises(ValueError):
        revision_before({"query": {"pages": {"1": {"missing": ""}}}})


def test_supplement_rows_close_the_renamed_seat_and_respect_the_asof() -> None:
    from prism.io.universe_sp500 import CHANGES_SUPPLEMENT_TABLE, supplement_changes

    assert [(r["date"], r["added"], r["removed"]) for r in CHANGES_SUPPLEMENT_TABLE] == [
        ("2023-07-10", "EG", "RE"), ("2024-03-25", "CPAY", "FLT"), ("2026-06-24", "ECHO", "SATS"),
        ("2026-08-18", "VMRK", "EQR")]
    # The deleted-row defect in miniature: RE joined in 2017 and was renamed EG in 2023.
    changes = _changes([("2017-06-19", "RE", "")])
    unrestored = reconstruct_membership(["EG"], changes, end_date=END)
    assert _interval(unrestored, "RE") == [(pd.Timestamp("2017-06-19"), END)]  # never closes
    assert _interval(unrestored, "EG") == [(HISTORY_FLOOR, END)]  # the same seat, counted twice
    fixed = reconstruct_membership(["EG"], supplement_changes(changes, END), end_date=END)
    assert _interval(fixed, "RE") == [(pd.Timestamp("2017-06-19"), pd.Timestamp("2023-07-10"))]
    assert _interval(fixed, "EG") == [(pd.Timestamp("2023-07-10"), END)]

    # The never-entered rename: without its row EQR vanishes from history and VMRK is backdated.
    asof = pd.Timestamp("2026-09-24")
    erased = reconstruct_membership(["VMRK"], _changes([]), end_date=asof)
    assert set(erased["ticker"]) == {"VMRK"}
    kept = reconstruct_membership(["VMRK"], supplement_changes(_changes([]), asof), end_date=asof)
    assert _interval(kept, "EQR") == [(HISTORY_FLOOR, pd.Timestamp("2026-08-18"))]
    assert _interval(kept, "VMRK") == [(pd.Timestamp("2026-08-18"), asof)]

    # No row dated after the as-of is added, and a table carrying a row gets no duplicate.
    early = supplement_changes(_changes([]), "2024-01-01")
    assert [(pd.Timestamp(d).strftime("%Y-%m-%d"), a) for d, a in zip(early["date"], early["added"])] == [
        ("2023-07-10", "EG")]
    full = _changes([(r["date"], r["added"], r["removed"]) for r in CHANGES_SUPPLEMENT_TABLE])
    assert supplement_changes(full, asof) is full


def test_history_differences_find_a_rewrite_through_the_baseline_only() -> None:
    from prism.io.universe_sp500 import history_differences

    baseline = pd.DataFrame([("EQR", "1990-01-01", "2026-07-17"), ("A", "2010-01-04", "2026-07-17")],
                            columns=["ticker", "start", "end"])
    # A later rebuild that extends open intervals and adds post-baseline names agrees.
    later = pd.DataFrame([("EQR", "1990-01-01", "2026-08-18"), ("VMRK", "2026-08-18", "2026-09-24"),
                          ("A", "2010-01-04", "2026-09-24")], columns=["ticker", "start", "end"])
    assert history_differences(later, baseline, "2026-07-17").empty
    # The unsupplemented rebuild rewrites EQR's seat as VMRK's.
    rewritten = pd.DataFrame([("VMRK", "1990-01-01", "2026-09-24"), ("A", "2010-01-04", "2026-09-24")],
                             columns=["ticker", "start", "end"])
    found = history_differences(rewritten, baseline, "2026-07-17")
    assert found.to_dict("records") == [
        {"ticker": "EQR", "start": "1990-01-01", "end": "2026-07-17", "side": "baseline"},
        {"ticker": "VMRK", "start": "1990-01-01", "end": "2026-07-17", "side": "rebuild"},
    ]
