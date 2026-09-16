"""The security master: vendor-neutral identifiers, listings, membership (engine spec §3.2, D2).

The failure this file is written to catch is the one that costs money rather
than a test run: two instruments collapsing into one identifier, or one
instrument splitting into two. The first sells a position in a company the book
never held — a ticker reused after a delisting inherits the dead name's
holding; the second orphans a real position, because the broker reports it
under a symbol the research side no longer joins to. Both are silent. Neither
is visible in a return series.

So the assertions are structural, not numerical: one instrument reached
through two vendors is one sid, a reused ticker is two sids and two holdings,
``resolve`` is single-valued or refuses, and a series that matched nothing is
counted as provisional rather than assumed. The gap flag covers the case
matching cannot see at all — reuse inside one vendor series with no event to
split on — and is asserted on both sides of the declared threshold, because a
flag that fires on everything is the same as no flag.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from prism.io import sessions
from prism.io.dataset import Dataset
from prism.io.manifest import Manifest, hash_tree
from prism.io.security_master import (
    ATTESTATION_SOURCE,
    DEFAULT_GAP_FLAG_SESSIONS,
    OVERRIDES_FILENAME,
    IdentityEvent,
    SecurityMaster,
    SecurityMasterError,
    SecurityOverrides,
    missing_sessions_between,
    segment_series,
    sid_of,
)

RESEARCH_VENDOR = "twelvedata"
BROKER_VENDOR = "alpaca"
INDEX_ID = "sp500"

FIRST = dt.date(2026, 1, 2)
LAST = dt.date(2026, 12, 31)


@pytest.fixture(scope="module")
def grid() -> list[dt.date]:
    return sessions.sessions_between(FIRST, LAST)


def _instant(day: dt.date, hour: int = 20) -> str:
    return str(pd.Timestamp(dt.datetime.combine(day, dt.time(hour, 0))).tz_localize(sessions.SESSION_TZ).isoformat())


# ---- one instrument, two vendors, one sid ----------------------------------


def test_two_vendors_covering_one_instrument_resolve_to_one_sid(grid: list[dt.date]) -> None:
    """The join D2 exists for: a broker position and a research bar series meet on one sid."""
    master = SecurityMaster()
    research = master.ingest_series(
        RESEARCH_VENDOR, "AAPL", ingested_at=_instant(grid[40]), bar_dates=grid[:40], name="Apple Inc."
    )
    broker = master.ingest_series(BROKER_VENDOR, "AAPL", ingested_at=_instant(grid[41]), bar_dates=grid[10:41])

    assert len(research.sids) == 1
    assert research.sids == broker.sids
    sid = research.sids[0]

    listings = master.frame("listings")
    assert set(listings["sid"]) == {sid}
    assert set(listings["vendor"]) == {RESEARCH_VENDOR, BROKER_VENDOR}
    assert len(listings) == 2

    # The join itself: a position the broker reports under its own symbol finds
    # the bars the research vendor supplied, because both name one sid.
    broker_positions = {"AAPL": 100.0}
    research_bars = {sid: "a close series"}
    joined = {
        master.sid_for_vendor_symbol(BROKER_VENDOR, symbol, grid[35]): qty for symbol, qty in broker_positions.items()
    }
    assert set(joined) <= set(research_bars)
    assert joined[sid] == 100.0


def test_a_second_vendor_verifies_the_identity_the_first_could_only_propose(grid: list[dt.date]) -> None:
    master = SecurityMaster()
    resolution = master.ingest_series(RESEARCH_VENDOR, "AAPL", ingested_at=_instant(grid[40]), bar_dates=grid[:40])
    sid = resolution.sids[0]
    assert master.status_of(sid) == "provisional"
    assert master.report().n_provisional_sids == 1

    master.ingest_series(BROKER_VENDOR, "AAPL", ingested_at=_instant(grid[41]), bar_dates=grid[10:41])
    assert master.status_of(sid) == "verified"
    report = master.report()
    assert report.n_provisional_sids == 0
    assert report.n_verified_sids == 1
    # Verification is a new row, not an edit: the provisional statement stays
    # readable, so a reproduction as of the earlier date still sees it.
    rows = master.frame("securities")
    assert list(rows.loc[rows["sid"] == sid, "entity_status"]) == ["provisional", "verified"]


def test_resolve_is_single_valued_where_two_vendor_mappings_overlap(grid: list[dt.date]) -> None:
    master = SecurityMaster()
    master.ingest_series(RESEARCH_VENDOR, "AAPL", ingested_at=_instant(grid[40]), bar_dates=grid[:40])
    master.ingest_series(BROKER_VENDOR, "AAPL", ingested_at=_instant(grid[41]), bar_dates=grid[10:41])

    day = grid[20]
    listings = master.frame("listings")
    overlapping = [
        row
        for row in listings.to_dict("records")
        if row["valid_from"] <= day.isoformat() and (row["valid_to"] or "9999") >= day.isoformat()
    ]
    assert len(overlapping) == 2, "the fixture must exercise two mappings live on the same date"
    assert master.resolve("AAPL", day) == overlapping[0]["sid"]


def test_a_segment_matching_two_sids_mints_a_third_and_makes_resolve_refuse(grid: list[dt.date]) -> None:
    """The ambiguous case of §3.2: a third sid, flagged, and a query that refuses to pick a winner.

    Two vendors have already produced two identities under one ticker over
    disjoint intervals — one of them is a stray, but nothing in the data says
    which. A third series spanning both matches them both, so it mints its own
    sid and waits for an attestation rather than silently joining either.
    """
    master = SecurityMaster()
    first = master.ingest_series(RESEARCH_VENDOR, "DUP", ingested_at=_instant(grid[40]), bar_dates=grid[:40]).sids[0]
    second = master.ingest_series(BROKER_VENDOR, "DUP", ingested_at=_instant(grid[101]), bar_dates=grid[60:101]).sids[
        0
    ]
    assert second != first, "disjoint intervals under one ticker are two instruments until attested otherwise"

    spanning = master.ingest_series("thirdvendor", "DUP", ingested_at=_instant(grid[101]), bar_dates=grid[:101])
    third = spanning.sids[0]
    assert (third, spanning.segments[0].matched_by) == (sid_of(2), "minted")
    assert spanning.segments[0].candidates == (first, second)
    flag = {item.sid: item for item in master.report().attestation_flags}[third]
    assert (flag.reason, flag.candidates) == ("ambiguous_match", (first, second))

    assert master.unresolvable_tickers() == (("DUP", (first, second, third)),)
    with pytest.raises(SecurityMasterError, match="single-valued"):
        master.resolve("DUP", grid[10])


# ---- ticker reuse ----------------------------------------------------------


def test_ticker_reuse_after_a_delisting_yields_two_sids_and_two_holdings(grid: list[dt.date]) -> None:
    """The failure this prevents: a reused ticker inheriting the dead name's position."""
    cease = grid[100]
    master = SecurityMaster()
    resolution = master.ingest_series(
        RESEARCH_VENDOR,
        "RUSE",
        ingested_at=_instant(grid[-1]),
        bar_dates=grid[:101] + grid[130:],
        events=[IdentityEvent.cessation("RUSE", cease)],
    )

    assert len(resolution.sids) == 2
    first, second = resolution.sids
    assert first != second

    # Non-overlapping intervals are what make the two identities separable.
    listings = master.frame("listings").to_dict("records")
    assert [row["valid_to"] for row in listings if row["sid"] == first] == [cease.isoformat()]
    assert [row["valid_from"] for row in listings if row["sid"] == second] == [
        sessions.next_session(cease).isoformat()
    ]

    assert master.resolve("RUSE", grid[50]) == first
    assert master.resolve("RUSE", grid[140]) == second

    # A position book keyed by sid keeps both, and neither inherits the other.
    holdings: dict[str, float] = {}
    holdings[master.resolve("RUSE", grid[50])] = 100.0
    holdings[master.resolve("RUSE", grid[140])] = 40.0
    assert holdings == {first: 100.0, second: 40.0}

    # The reused segment matched nothing, so it is provisional and flagged.
    assert master.status_of(second) == "provisional"
    flagged = {flag.sid: flag.reason for flag in master.report().attestation_flags}
    assert flagged[second] == "no_match"


def test_a_rename_splits_one_series_into_two_tickers_on_one_sid(grid: list[dt.date]) -> None:
    """A ticker change is a new listing interval, not a new instrument (RENAME_TABLE, §3.2)."""
    effective = grid[60]
    master = SecurityMaster()
    resolution = master.ingest_series(
        RESEARCH_VENDOR,
        "FB",
        ingested_at=_instant(grid[-1]),
        bar_dates=grid,
        events=[IdentityEvent.ticker_change("FB", "META", effective, announce_date=grid[55])],
    )
    assert len(resolution.sids) == 1
    sid = resolution.sids[0]
    assert master.resolve("FB", grid[10]) == sid
    assert master.resolve("META", grid[100]) == sid
    with pytest.raises(SecurityMasterError):
        master.resolve("FB", grid[100])

    rows = {row["ticker"]: row for row in master.frame("listings").to_dict("records")}
    assert rows["FB"]["valid_to"] == sessions.previous_session(effective).isoformat()
    assert rows["META"]["valid_from"] == effective.isoformat()
    assert rows["META"]["announce_date"] == grid[55].isoformat()


def test_segmentation_ignores_an_event_outside_the_series_span(grid: list[dt.date]) -> None:
    segments = segment_series("AAA", grid[10], grid[20], [IdentityEvent.cessation("AAA", grid[40])])
    assert len(segments) == 1
    assert segments[0].valid_to == grid[20]


# ---- the gap flag ----------------------------------------------------------


def _bars_with_gap(grid: list[dt.date], missing: int) -> list[dt.date]:
    """Sessions with exactly ``missing`` consecutive sessions removed from the middle."""
    return grid[:50] + grid[50 + missing :]


def test_the_gap_flag_fires_above_the_declared_threshold_and_not_at_it(grid: list[dt.date]) -> None:
    at_threshold = _bars_with_gap(grid, DEFAULT_GAP_FLAG_SESSIONS)
    over_threshold = _bars_with_gap(grid, DEFAULT_GAP_FLAG_SESSIONS + 1)
    assert missing_sessions_between(at_threshold[49], at_threshold[50]) == DEFAULT_GAP_FLAG_SESSIONS
    assert missing_sessions_between(over_threshold[49], over_threshold[50]) == DEFAULT_GAP_FLAG_SESSIONS + 1

    quiet = SecurityMaster()
    assert (
        quiet.ingest_series(RESEARCH_VENDOR, "HALT", ingested_at=_instant(grid[-1]), bar_dates=at_threshold).gap_flags
        == ()
    )
    assert quiet.report().gap_flags == ()

    loud = SecurityMaster()
    flags = loud.ingest_series(
        RESEARCH_VENDOR, "STRAY", ingested_at=_instant(grid[-1]), bar_dates=over_threshold
    ).gap_flags
    assert len(flags) == 1
    flag = flags[0]
    assert flag.missing_sessions == DEFAULT_GAP_FLAG_SESSIONS + 1
    assert flag.gap_flag_sessions == DEFAULT_GAP_FLAG_SESSIONS
    assert flag.ticker == "STRAY"
    entry = flag.to_coverage_entry()
    assert entry["sid"] == flags[0].sid and entry["flagged"] is True
    assert loud.report().coverage_entries()["gap_flags"] == [entry]


def test_the_gap_flag_honours_a_declared_threshold_and_stays_inside_the_segment(grid: list[dt.date]) -> None:
    cease = grid[100]
    master = SecurityMaster(gap_flag_sessions=3)
    # The 29-session hole between the two segments is not a gap inside either
    # of them: it separates two instruments, which is what the events said.
    resolution = master.ingest_series(
        RESEARCH_VENDOR,
        "RUSE",
        ingested_at=_instant(grid[-1]),
        bar_dates=grid[:101] + grid[130:],
        events=[IdentityEvent.cessation("RUSE", cease)],
    )
    assert resolution.gap_flags == ()

    unsplit = SecurityMaster(gap_flag_sessions=3)
    flags = unsplit.ingest_series(
        RESEARCH_VENDOR, "RUSE", ingested_at=_instant(grid[-1]), bar_dates=grid[:101] + grid[130:]
    ).gap_flags
    assert [flag.missing_sessions for flag in flags] == [29]


def test_counting_a_gap_outside_the_session_table_refuses(grid: list[dt.date]) -> None:
    with pytest.raises(SecurityMasterError, match="bundled table covers"):
        missing_sessions_between(dt.date(2005, 1, 3), grid[0])


# ---- provisional identities are counted ------------------------------------


def test_provisional_sids_are_counted_and_reported(grid: list[dt.date]) -> None:
    master = SecurityMaster()
    for symbol in ("AAA", "BBB", "CCC"):
        master.ingest_series(RESEARCH_VENDOR, symbol, ingested_at=_instant(grid[40]), bar_dates=grid[:40])
    report = master.report()
    assert report.n_sids == 3
    assert report.n_provisional_sids == 3
    assert report.provisional_sids == tuple(sid_of(n) for n in range(3))
    assert [flag.reason for flag in report.attestation_flags] == ["no_match"] * 3
    coverage = report.coverage_entries()
    assert coverage["n_provisional_sids"] == 3
    assert coverage["n_quarantined_sids"] == 0
    assert [flag["vendor_symbol"] for flag in coverage["attestation_flags"]] == ["AAA", "BBB", "CCC"]
    assert report.to_dict()["n_provisional_sids"] == 3


# ---- the attestation file --------------------------------------------------


SPEC_EXAMPLE = """
schema = 1

[[attestation]]
sid = "sec_000412"
action = "verify"
attested_identity = "Meta Platforms, Inc. Class A"
attested_by = "owner"
attested_at = "2026-07-14"
evidence = "docs/data_integrity_diagnostic.md §6"

[[attestation]]
sid = "sec_000913"
action = "quarantine"
attested_identity = "unknown"
reason = "vendor resolves to Enact Holdings (IPO 2021-09-16), never Watson/Actavis"
attested_by = "owner"
attested_at = "2026-07-14"
evidence = "docs/data_integrity_diagnostic.md §3"

[[merge]]
keep = "sec_000412"
absorb = "sec_001877"
attested_by = "owner"
attested_at = "2026-07-14"
evidence = "docs/data_integrity_diagnostic.md §6"

[[alias]]
sid = "sec_000412"
vendor = "alpaca"
vendor_symbol = "FB"
valid_from = "2012-05-18"
valid_to = "2022-06-08"
attested_by = "owner"
attested_at = "2026-07-14"
evidence = "docs/data_integrity_diagnostic.md §6"
"""


def test_the_worked_attestation_file_of_the_specification_parses(tmp_path: Path) -> None:
    path = tmp_path / OVERRIDES_FILENAME
    path.write_text(SPEC_EXAMPLE, encoding="utf-8")
    overrides = SecurityOverrides.read(path)
    assert [a.sid for a in overrides.attestations] == ["sec_000412", "sec_000913"]
    assert [a.action for a in overrides.attestations] == ["verify", "quarantine"]
    assert overrides.attestations[1].reason.startswith("vendor resolves to Enact")
    assert overrides.merges[0].keep == "sec_000412" and overrides.merges[0].absorb == "sec_001877"
    alias = overrides.aliases[0]
    assert (alias.vendor, alias.vendor_symbol, alias.valid_to) == ("alpaca", "FB", dt.date(2022, 6, 8))
    assert SecurityOverrides.read_if_present(tmp_path / "absent.toml").attestations == ()


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("schema = 2\n", "accepts 1"),
        (
            'schema = 1\n[[attestation]]\nsid = "s"\naction = "delete"\nattested_by = "o"\nattested_at = 2026-07-14\n',
            "verify",
        ),
        (
            'schema = 1\n[[attestation]]\nsid = "s"\naction = "quarantine"\nattested_by = "o"\nattested_at = 2026-07-14\n',
            "reason",
        ),
        (
            'schema = 1\n[[attestation]]\nsid = "s"\naction = "verify"\nattested_by = "o"\nattested_at = 2026-07-14\nnote = "x"\n',
            "unknown keys",
        ),
        ('schema = 1\n[[attestations]]\nsid = "s"\n', "unknown top-level"),
        (
            'schema = 1\n[[merge]]\nkeep = "a"\nabsorb = "a"\nattested_by = "o"\nattested_at = 2026-07-14\n',
            "into itself",
        ),
    ],
)
def test_a_malformed_attestation_file_refuses(tmp_path: Path, body: str, match: str) -> None:
    path = tmp_path / OVERRIDES_FILENAME
    path.write_text(body, encoding="utf-8")
    with pytest.raises(SecurityMasterError, match=match):
        SecurityOverrides.read(path)


def test_an_attestation_verifies_or_quarantines_without_changing_the_sid(tmp_path: Path, grid: list[dt.date]) -> None:
    master = SecurityMaster()
    kept = master.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[40]), bar_dates=grid[:40]).sids[0]
    dropped = master.ingest_series(RESEARCH_VENDOR, "BBB", ingested_at=_instant(grid[40]), bar_dates=grid[:40]).sids[0]
    path = tmp_path / OVERRIDES_FILENAME
    path.write_text(
        f"""
schema = 1

[[attestation]]
sid = "{kept}"
action = "verify"
attested_identity = "Attested Inc."
attested_by = "owner"
attested_at = "{grid[50].isoformat()}"

[[attestation]]
sid = "{dropped}"
action = "quarantine"
reason = "vendor resolves to a different instrument"
attested_by = "owner"
attested_at = "{grid[50].isoformat()}"
""",
        encoding="utf-8",
    )
    master.overrides = SecurityOverrides.read(path)
    master.apply_overrides()

    assert master.status_of(kept) == "verified"
    assert master.status_of(dropped) == "quarantined"
    report = master.report()
    assert report.n_quarantined_sids == 1
    assert report.quarantined_sids == (dropped,)
    # The sid is unchanged and every earlier row stays readable.
    rows = master.frame("securities")
    assert set(rows["sid"]) == {kept, dropped}
    assert set(rows.loc[rows["source"] == ATTESTATION_SOURCE, "sid"]) == {kept, dropped}
    assert master.resolve("AAA", grid[10]) == kept


def test_an_attestation_naming_an_unknown_sid_refuses(tmp_path: Path, grid: list[dt.date]) -> None:
    master = SecurityMaster()
    master.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[40]), bar_dates=grid[:40])
    path = tmp_path / OVERRIDES_FILENAME
    path.write_text(
        f'schema = 1\n\n[[attestation]]\nsid = "sec_999999"\naction = "verify"\nattested_by = "o"\n'
        f'attested_at = "{grid[50].isoformat()}"\n',
        encoding="utf-8",
    )
    master.overrides = SecurityOverrides.read(path)
    with pytest.raises(SecurityMasterError, match="mints no sid"):
        master.apply_overrides()


def test_an_alias_binds_a_vendor_symbol_that_overlap_matching_could_not(tmp_path: Path, grid: list[dt.date]) -> None:
    """The FB/META case: the broker's dead symbol maps onto the sid the research vendor minted."""
    master = SecurityMaster()
    sid = master.ingest_series(RESEARCH_VENDOR, "META", ingested_at=_instant(grid[40]), bar_dates=grid[:40]).sids[0]
    path = tmp_path / OVERRIDES_FILENAME
    path.write_text(
        f"""
schema = 1

[[alias]]
sid = "{sid}"
vendor = "{BROKER_VENDOR}"
vendor_symbol = "FB"
valid_from = "{grid[0].isoformat()}"
valid_to = "{grid[39].isoformat()}"
attested_by = "owner"
attested_at = "{grid[40].isoformat()}"
""",
        encoding="utf-8",
    )
    master.overrides = SecurityOverrides.read(path)
    resolution = master.ingest_series(BROKER_VENDOR, "FB", ingested_at=_instant(grid[41]), bar_dates=grid[:40])
    assert resolution.sids == (sid,)
    assert resolution.segments[0].matched_by == "alias"
    assert master.sid_for_vendor_symbol(BROKER_VENDOR, "FB", grid[10]) == sid
    assert master.report().n_sids == 1


def test_a_merge_moves_the_mapping_and_keeps_resolve_single_valued(tmp_path: Path, grid: list[dt.date]) -> None:
    master = SecurityMaster()
    keep = master.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[40]), bar_dates=grid[:40]).sids[0]
    absorb = master.ingest_series(BROKER_VENDOR, "AAA2", ingested_at=_instant(grid[40]), bar_dates=grid[:40]).sids[0]
    path = tmp_path / OVERRIDES_FILENAME
    path.write_text(
        f'schema = 1\n\n[[merge]]\nkeep = "{keep}"\nabsorb = "{absorb}"\nattested_by = "owner"\n'
        f'attested_at = "{grid[50].isoformat()}"\nevidence = "diagnostic §6"\n',
        encoding="utf-8",
    )
    master.overrides = SecurityOverrides.read(path)
    master.apply_overrides()

    assert master.status_of(absorb) == "quarantined"
    assert "merged into" in master.frame("securities").set_index("sid").loc[absorb, "quarantine_reason"].iloc[-1]
    # The absorbed mapping now answers under the kept sid, and only once.
    assert master.sid_for_vendor_symbol(BROKER_VENDOR, "AAA2", grid[10]) == keep
    assert master.resolve("AAA2", grid[10]) == keep
    assert master.unresolvable_tickers() == ()


# ---- membership ------------------------------------------------------------


def test_membership_intervals_are_translated_not_inherited(grid: list[dt.date]) -> None:
    master = SecurityMaster()
    master.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[-1]), bar_dates=grid)
    master.ingest_series(RESEARCH_VENDOR, "BBB", ingested_at=_instant(grid[-1]), bar_dates=grid)
    as_of = grid[-1]
    result = master.import_membership_intervals(
        INDEX_ID,
        [
            ("AAA", dt.date(1990, 1, 1), as_of),
            ("BBB", grid[10], grid[80]),
            ("ZZZ", grid[10], grid[80]),
        ],
        as_of=as_of,
        source="wikipedia@2026-06-16",
        ingested_at=_instant(as_of),
        on_unresolved="skip",
    )
    assert (result.n_intervals, result.n_rows) == (3, 2)
    assert [ticker for ticker, _ in result.unresolved] == ["ZZZ"]
    rows = {row["sid"]: row for row in master.frame("membership").to_dict("records")}
    open_row = rows[master.resolve("AAA", grid[5])]
    assert open_row["valid_from"] is None and open_row["floor_unknown"] is True
    assert open_row["valid_to"] is None, "an end equal to the file's as-of is an open interval, not a closure"
    closed_row = rows[master.resolve("BBB", grid[5])]
    assert (closed_row["valid_from"], closed_row["valid_to"]) == (grid[10].isoformat(), grid[80].isoformat())
    assert closed_row["floor_unknown"] is False
    assert result.to_dict()["n_unresolved"] == 1


def test_an_unresolvable_membership_ticker_refuses_by_default(grid: list[dt.date]) -> None:
    master = SecurityMaster()
    master.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[-1]), bar_dates=grid)
    with pytest.raises(SecurityMasterError, match="names one instrument"):
        master.import_membership_intervals(
            INDEX_ID,
            [("ZZZ", grid[10], grid[80])],
            as_of=grid[-1],
            source="wikipedia",
            ingested_at=_instant(grid[-1]),
        )


def test_a_membership_interval_spanning_a_ticker_reuse_refuses(grid: list[dt.date]) -> None:
    """Two instruments under one ticker inside one index interval is a question the data cannot answer."""
    master = SecurityMaster()
    master.ingest_series(
        RESEARCH_VENDOR,
        "RUSE",
        ingested_at=_instant(grid[-1]),
        bar_dates=grid[:101] + grid[130:],
        events=[IdentityEvent.cessation("RUSE", grid[100])],
    )
    with pytest.raises(SecurityMasterError, match="names one instrument"):
        master.import_membership_intervals(
            INDEX_ID,
            [("RUSE", grid[10], grid[200])],
            as_of=grid[-1],
            source="wikipedia",
            ingested_at=_instant(grid[-1]),
        )


# ---- persistence -----------------------------------------------------------


def test_a_sid_minted_once_survives_a_reopen_and_a_re_ingest(tmp_path: Path, grid: list[dt.date]) -> None:
    """Stability is the whole point of a surrogate: a position keyed by it stays the same instrument."""
    root = tmp_path / "data"
    first = SecurityMaster()
    minted = first.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[40]), bar_dates=grid[:40]).sids[0]
    first.ingest_series(RESEARCH_VENDOR, "BBB", ingested_at=_instant(grid[40]), bar_dates=grid[:40])
    first.write(root)

    reopened = SecurityMaster.open(root)
    assert reopened.sids() == first.sids()
    # The same series, extended by a week of new bars: the sid is reused and
    # the listing interval is a new revision, not a second identity.
    again = reopened.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[50]), bar_dates=grid[:45])
    assert again.sids == (minted,)
    listings = reopened.frame("listings")
    rows = listings.loc[listings["sid"] == minted]
    assert list(rows["revision"]) == [1, 2]
    assert list(rows["valid_to"]) == [grid[39].isoformat(), grid[44].isoformat()]
    # A third sid is minted from the next unused number, never from a reused one.
    fresh = reopened.ingest_series(RESEARCH_VENDOR, "CCC", ingested_at=_instant(grid[50]), bar_dates=grid[:40]).sids[0]
    assert fresh == sid_of(2)


def test_an_identical_re_ingest_writes_no_revision(tmp_path: Path, grid: list[dt.date]) -> None:
    master = SecurityMaster()
    master.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[40]), bar_dates=grid[:40])
    master.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[41]), bar_dates=grid[:40])
    listings = master.frame("listings")
    assert len(listings) == 1
    assert listings["revision"].iloc[0] == 1
    assert listings["ingested_at"].iloc[0] == _instant(
        grid[40]
    ), "the workspace has held the fact since the first read"


def test_write_produces_the_three_tables_in_the_declared_schema(tmp_path: Path, grid: list[dt.date]) -> None:
    from prism.io.security_master import LISTINGS_COLUMNS, MEMBERSHIP_COLUMNS, SECURITIES_COLUMNS

    root = tmp_path / "data"
    master = SecurityMaster()
    master.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[40]), bar_dates=grid[:40])
    master.import_membership_intervals(
        INDEX_ID,
        [("AAA", grid[0], grid[39])],
        as_of=grid[39],
        source="wikipedia",
        ingested_at=_instant(grid[40]),
    )
    written = master.write(root)
    assert [path.name for path in written] == ["securities.parquet", "listings.parquet", "membership.parquet"]
    assert list(pd.read_parquet(root / "securities.parquet").columns) == list(SECURITIES_COLUMNS)
    assert list(pd.read_parquet(root / "listings.parquet").columns) == list(LISTINGS_COLUMNS)
    assert list(pd.read_parquet(root / "membership.parquet").columns) == list(MEMBERSHIP_COLUMNS)


def test_a_duplicate_key_source_revision_is_refused_before_it_is_written(grid: list[dt.date]) -> None:
    master = SecurityMaster()
    master.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[40]), bar_dates=grid[:40])
    master._rows["listings"].append(dict(master._rows["listings"][0]))
    with pytest.raises(SecurityMasterError, match="ingestion defect"):
        master.validate()


# ---- the as-of reader sees what the master wrote ---------------------------


def _manifest(root: Path, written: tuple[Path, ...]) -> Manifest:
    source = {
        "lag_bars": 0,
        "revision_policy": "dedupe_identical",
        "price_basis": "split_adjusted_at_ingest",
        "url": None,
        "sha256": None,
        "redistributable": False,
    }
    sources = {vendor: dict(source) for vendor in (RESEARCH_VENDOR, BROKER_VENDOR, ATTESTATION_SOURCE)}
    order = [RESEARCH_VENDOR, BROKER_VENDOR, ATTESTATION_SOURCE]
    return Manifest(
        provenance="market",
        as_of=LAST.isoformat(),
        files=tuple(hash_tree(root, list(written))),
        sources=sources,
        source_priority={"securities": order, "listings": order, "membership": order},
        universe="none",
        vintage="security_master_test",
    )


def test_the_as_of_view_reads_the_tables_and_agrees_with_the_master(tmp_path: Path, grid: list[dt.date]) -> None:
    """The writer's output is the reader's input: one schema, one resolve, one identity."""
    root = tmp_path / "data"
    master = SecurityMaster()
    sid = master.ingest_series(RESEARCH_VENDOR, "AAA", ingested_at=_instant(grid[40]), bar_dates=grid[:40]).sids[0]
    master.ingest_series(BROKER_VENDOR, "AAA", ingested_at=_instant(grid[41]), bar_dates=grid[10:41])
    master.import_membership_intervals(
        INDEX_ID, [("AAA", grid[0], grid[39])], as_of=grid[-1], source=RESEARCH_VENDOR, ingested_at=_instant(grid[40])
    )
    written = master.write(root)
    dataset = Dataset(root, _manifest(root, written), verify_files=True)

    view = dataset.view("research_vintage", grid[41].isoformat(), vintage="security_master_test")
    assert view.resolve("AAA", grid[20].isoformat()) == sid == master.resolve("AAA", grid[20])
    # The membership interval closed at grid[39], two sessions before this view.
    assert view.membership(INDEX_ID) == set()
    assert dataset.view("research_vintage", grid[20].isoformat(), vintage="security_master_test").membership(
        INDEX_ID
    ) == {sid}

    # Point in time: the verification the second vendor's 20:00 ingest wrote is
    # not visible at that day's 16:00 close, and is visible at the next one.
    same_day = dataset.view("research_vintage", grid[41].isoformat(), vintage="security_master_test")
    assert same_day.securities([sid])["entity_status"].iloc[0] == "provisional"
    next_day = dataset.view("research_vintage", grid[42].isoformat(), vintage="security_master_test")
    assert next_day.securities([sid])["entity_status"].iloc[0] == "verified"


# ---- the real point-in-time membership file --------------------------------

MEMBERSHIP_FILE = Path(__file__).resolve().parents[1] / "data" / "universe" / "sp500_membership_2026-06-16.parquet"
MEMBERSHIP_AS_OF = dt.date(2026, 6, 16)


def test_the_tracked_membership_file_translates_under_the_declared_conventions() -> None:
    """The conventions of §3.2 pinned on the file the B1 reproduction pins (Run counts).

    891 intervals over 869 tickers; 519 rows end on the file's own as-of and
    become open intervals; 517 open at or before the 1990-01-01 history floor
    and become ``floor_unknown`` — 515 exactly at the floor plus BUD and DIS at
    1976-07-01. The two rows ending 2026-06-22 (CPB, POOL) are scheduled
    removals the scrape already knew and keep their closure date.
    """
    frame = pd.read_parquet(MEMBERSHIP_FILE)
    rows = [
        (str(record.ticker), pd.Timestamp(record.start).date(), pd.Timestamp(record.end).date())
        for record in frame.itertuples(index=False)
    ]
    spans: dict[str, list[dt.date]] = {}
    for ticker, start, end in rows:
        span = spans.setdefault(ticker, [start, end])
        spans[ticker] = [min(span[0], start), max(span[1], end)]

    master = SecurityMaster()
    for ticker, (start, end) in sorted(spans.items()):
        master.ingest_series(
            "legacy_cache", ticker, ingested_at=_instant(MEMBERSHIP_AS_OF), first_bar=start, last_bar=end
        )
    result = master.import_membership_intervals(
        INDEX_ID,
        rows,
        as_of=MEMBERSHIP_AS_OF,
        source="wikipedia@2026-06-16",
        ingested_at=_instant(MEMBERSHIP_AS_OF),
    )
    assert (len(spans), result.n_intervals, result.n_rows) == (869, 891, 891)

    written = master.frame("membership")
    assert int(written["valid_to"].isna().sum()) == 519
    assert int(written["floor_unknown"].sum()) == 517
    scheduled = written.loc[written["sid"].isin([master.resolve("CPB", dt.date(2026, 6, 15))])]
    assert list(scheduled["valid_to"]) == [
        "2026-06-22"
    ], "a removal dated after the as-of is a closure, not an open end"
    # Every minted identity is provisional and counted: nothing about a scraped
    # constituent list verifies which instrument a ticker denoted.
    report = master.report()
    assert report.n_sids == report.n_provisional_sids == 869
