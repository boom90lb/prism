"""``prism data ingest``: a declared mapping, a declared lag, a declared revision policy (§3.2).

The three properties the ingest path exists to hold, each asserted here against
a fixture whose expected values are written out independently of the code under
test:

*One declaration, two formats.* A CSV and a Parquet carrying the same content
ingest to the same rows under one mapping. The column names differ between the
two files on purpose, because the mapping is what resolves them.

*A lag is a source's property.* ``lag_bars`` reaches the manifest and never a
row, and a bar of session ``d`` from a source with ``lag_bars = 1`` is invisible
to a decision taken at the close of ``d`` and visible at the close of the next
session.

*A revision is a whole row.* A re-ingest that changes one field writes the next
revision carrying every field of the incoming row — a field the new file does
not carry is null in the new revision rather than inherited — and the previous
revision is left exactly where it was.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from prism.io import sessions
from prism.io.dataset import Dataset, DatasetError
from prism.io.local_files import (
    BAR_COLUMNS,
    BarBatch,
    IngestError,
    IngestMapping,
    ingest_local_files,
    merge_bar_revisions,
    read_mapped_file,
)

INGESTED_AT = "2026-09-08T20:00:00-04:00"
LATER_INGEST = "2026-09-09T20:00:00-04:00"

MAPPING_TOML = """
schema = 1

[source]
id = "vendor_a"
kind = "vendor_api"
lag_bars = 0
revision_policy = "dedupe_identical"
price_basis = "split_adjusted_at_ingest"
timezone = "America/New_York"
gap_flag_sessions = 15
redistributable = false

[columns]
ts = "datetime"
open = "o"
high = "h"
low = "l"
close = "c"
volume = "v"

[identity]
vendor_symbol = "symbol"
match = "ticker_interval_overlap"
"""


def _sessions(first: str, count: int) -> list[dt.date]:
    day = dt.date.fromisoformat(first)
    out: list[dt.date] = []
    while len(out) < count:
        if sessions.is_session(day):
            out.append(day)
        day = day + dt.timedelta(days=1)
    return out


def _frame(days: list[dt.date], symbol: str = "AAA", base: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [day.isoformat() for day in days],
            "symbol": [symbol] * len(days),
            "o": [base + i for i in range(len(days))],
            "h": [base + 1 + i for i in range(len(days))],
            "l": [base - 1 + i for i in range(len(days))],
            "c": [base + 0.5 + i for i in range(len(days))],
            "v": [1000.0 + i for i in range(len(days))],
        }
    )


def _mapping(tmp_path: Path, **overrides: str) -> IngestMapping:
    text = MAPPING_TOML
    for key, value in overrides.items():
        old = f"{key} = "
        lines = []
        for line in text.splitlines():
            lines.append(f"{key} = {value}" if line.startswith(old) else line)
        text = "\n".join(lines)
    path = tmp_path / "mapping.toml"
    path.write_text(text, encoding="utf-8")
    return IngestMapping.read(path)


def _bars(root: Path, sid: str) -> pd.DataFrame:
    return pd.read_parquet(root / "bars" / f"{sid}_1d.parquet")


# ---------------------------------------------------------------------------
# the mapping is the declaration
# ---------------------------------------------------------------------------


def test_a_csv_and_a_parquet_of_one_series_ingest_identically(tmp_path: Path) -> None:
    days = _sessions("2024-01-03", 8)
    frame = _frame(days)
    csv = tmp_path / "vendor.csv"
    frame.to_csv(csv, index=False)
    parquet = tmp_path / "vendor.parquet"
    frame.to_parquet(parquet, index=False)
    mapping = _mapping(tmp_path)

    from_csv = ingest_local_files(
        tmp_path / "a", [csv], mapping=mapping, fmt="csv", vintage="v1", ingested_at=INGESTED_AT
    )
    from_parquet = ingest_local_files(
        tmp_path / "b", [parquet], mapping=mapping, fmt="parquet", vintage="v1", ingested_at=INGESTED_AT
    )

    assert from_csv.symbols == from_parquet.symbols == {"AAA": "sec_000000"}
    left = _bars(tmp_path / "a", "sec_000000")
    right = _bars(tmp_path / "b", "sec_000000")
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert list(left.columns) == list(BAR_COLUMNS)
    assert len(left) == len(days)
    assert from_csv.manifest.id == from_parquet.manifest.id, "identical content under one mapping is one dataset"


def test_the_mapping_refuses_what_it_cannot_declare(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="schema"):
        IngestMapping.from_dict({"schema": 2, "source": {}, "columns": {}})
    with pytest.raises(IngestError, match="unknown key"):
        IngestMapping.from_dict(
            {
                "schema": 1,
                "source": {
                    "id": "a",
                    "lag_bars": 0,
                    "revision_policy": "dedupe_identical",
                    "price_basis": "raw",
                    "lag_days": 1,
                },
                "columns": {name: name for name in ("ts", "open", "high", "low", "close", "volume")},
            }
        )
    with pytest.raises(IngestError, match="revision_policy"):
        IngestMapping.from_dict(
            {
                "schema": 1,
                "source": {"id": "a", "lag_bars": 0, "revision_policy": "last_wins", "price_basis": "raw"},
                "columns": {name: name for name in ("ts", "open", "high", "low", "close", "volume")},
            }
        )
    with pytest.raises(IngestError, match=r"\['volume'\]"):
        IngestMapping.from_dict(
            {
                "schema": 1,
                "source": {"id": "a", "lag_bars": 0, "revision_policy": "dedupe_identical", "price_basis": "raw"},
                "columns": {name: name for name in ("ts", "open", "high", "low", "close")},
            }
        )
    with pytest.raises(IngestError, match="security_overrides.toml"):
        IngestMapping.from_dict(
            {
                "schema": 1,
                "source": {"id": "a", "lag_bars": 0, "revision_policy": "dedupe_identical", "price_basis": "raw"},
                "columns": {name: name for name in ("ts", "open", "high", "low", "close", "volume")},
                "identity": {"match": "cusip"},
            }
        )


def test_a_file_missing_a_declared_column_is_refused_by_name(tmp_path: Path) -> None:
    days = _sessions("2024-01-03", 3)
    frame = _frame(days).drop(columns=["v"])
    path = tmp_path / "vendor.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(IngestError, match=r"\['v'\]"):
        read_mapped_file(path, fmt="csv", mapping=_mapping(tmp_path))


def test_two_rows_of_one_file_for_one_timestamp_are_refused(tmp_path: Path) -> None:
    days = _sessions("2024-01-03", 3)
    frame = pd.concat([_frame(days), _frame(days[:1])], ignore_index=True)
    path = tmp_path / "vendor.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(IngestError, match="ingestion defect"):
        ingest_local_files(
            tmp_path / "data", [path], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=INGESTED_AT
        )


def test_a_row_with_no_vendor_symbol_is_refused(tmp_path: Path) -> None:
    days = _sessions("2024-01-03", 3)
    frame = _frame(days)
    frame.loc[1, "symbol"] = ""
    path = tmp_path / "vendor.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(IngestError, match="no vendor symbol"):
        ingest_local_files(
            tmp_path / "data", [path], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=INGESTED_AT
        )


# ---------------------------------------------------------------------------
# the declared lag
# ---------------------------------------------------------------------------


def test_a_declared_lag_reaches_the_manifest_and_never_a_row(tmp_path: Path) -> None:
    days = _sessions("2024-01-03", 5)
    path = tmp_path / "vendor.csv"
    _frame(days).to_csv(path, index=False)
    report = ingest_local_files(
        tmp_path / "data",
        [path],
        mapping=_mapping(tmp_path, lag_bars="2"),
        fmt="csv",
        vintage="v1",
        ingested_at=INGESTED_AT,
    )
    assert report.manifest.sources["vendor_a"]["lag_bars"] == 2
    assert report.manifest.lag_bars("vendor_a") == 2
    bars = _bars(tmp_path / "data", "sec_000000")
    assert "lag_bars" not in bars.columns, "a lag is a property of the source, never of a row"
    assert bars["available_at"].isna().all(), "the source declares no publication instant, so the field is empty"


@pytest.mark.parametrize("lag,visible_on_own_close", [(0, True), (1, False)])
def test_a_declared_lag_produces_the_availability_the_visibility_rule_needs(
    tmp_path: Path, lag: int, visible_on_own_close: bool
) -> None:
    days = _sessions("2024-01-03", 5)
    path = tmp_path / "vendor.csv"
    _frame(days).to_csv(path, index=False)
    root = tmp_path / f"data{lag}"
    ingest_local_files(
        root,
        [path],
        mapping=_mapping(tmp_path, lag_bars=str(lag)),
        fmt="csv",
        vintage="v1",
        ingested_at=INGESTED_AT,
    )
    dataset = Dataset.open(root)
    last = days[-1]
    view = dataset.view("research_vintage", last.isoformat(), vintage="v1")
    panel = view.panel(["sec_000000"], last.isoformat())
    assert (pd.Timestamp(last, tz="America/New_York") in panel.index) is visible_on_own_close
    later = dataset.view("research_vintage", sessions.next_session(last).isoformat(), vintage="v1")
    assert (
        pd.Timestamp(last, tz="America/New_York")
        in later.panel(["sec_000000"], sessions.next_session(last).isoformat()).index
    ), "one further session always makes a lag-1 bar visible"


# ---------------------------------------------------------------------------
# the declared revision policy
# ---------------------------------------------------------------------------


def _reingest(tmp_path: Path, root: Path, frame: pd.DataFrame, *, policy: str, name: str = "second.csv") -> None:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    ingest_local_files(
        root,
        [path],
        mapping=_mapping(tmp_path, revision_policy=f'"{policy}"'),
        fmt="csv",
        vintage="v1",
        ingested_at=LATER_INGEST,
    )


def _first_ingest(tmp_path: Path, *, policy: str = "dedupe_identical") -> tuple[Path, list[dt.date], pd.DataFrame]:
    days = _sessions("2024-01-03", 5)
    frame = _frame(days)
    path = tmp_path / "first.csv"
    frame.to_csv(path, index=False)
    root = tmp_path / "data"
    ingest_local_files(
        root,
        [path],
        mapping=_mapping(tmp_path, revision_policy=f'"{policy}"'),
        fmt="csv",
        vintage="v1",
        ingested_at=INGESTED_AT,
    )
    return root, days, frame


def test_a_revision_replaces_the_whole_row_rather_than_patching_a_field(tmp_path: Path) -> None:
    root, days, frame = _first_ingest(tmp_path)
    corrected = frame.copy()
    corrected.loc[2, "c"] = 999.0
    corrected.loc[2, "v"] = None  # the corrected file simply does not carry a volume
    _reingest(tmp_path, root, corrected, policy="dedupe_identical")

    bars = _bars(root, "sec_000000")
    target = pd.Timestamp(days[2], tz="America/New_York")
    rows = bars.loc[bars["ts"] == target].sort_values("revision")
    assert list(rows["revision"]) == [1, 2], "one changed field writes one new revision"
    first, second = rows.iloc[0], rows.iloc[1]
    assert first["close"] == frame.loc[2, "c"] and first["volume"] == frame.loc[2, "v"]
    assert second["close"] == 999.0
    assert pd.isna(second["volume"]), (
        "a revision is a complete replacement: a field the new file does not carry is null in the new revision, "
        "never inherited from the previous one"
    )
    assert second["ingested_at"] == LATER_INGEST and first["ingested_at"] == INGESTED_AT
    assert len(bars) == len(days) + 1, "only the changed row is revised"


def test_dedupe_identical_writes_nothing_when_the_reingested_row_is_unchanged(tmp_path: Path) -> None:
    root, days, frame = _first_ingest(tmp_path)
    _reingest(tmp_path, root, frame, policy="dedupe_identical")
    bars = _bars(root, "sec_000000")
    assert len(bars) == len(days)
    assert set(bars["revision"]) == {1}
    assert set(bars["ingested_at"]) == {INGESTED_AT}, "an unchanged fact keeps the date this workspace first held it"


def test_always_new_revision_writes_a_revision_even_when_nothing_changed(tmp_path: Path) -> None:
    root, days, frame = _first_ingest(tmp_path, policy="always_new_revision")
    _reingest(tmp_path, root, frame, policy="always_new_revision")
    bars = _bars(root, "sec_000000")
    assert len(bars) == 2 * len(days)
    assert sorted(set(bars["revision"])) == [1, 2]


def test_conflict_raise_refuses_a_differing_row_and_names_the_key(tmp_path: Path) -> None:
    root, days, frame = _first_ingest(tmp_path, policy="conflict_raise")
    corrected = frame.copy()
    corrected.loc[1, "c"] = 12.5
    with pytest.raises(IngestError, match="conflict_raise"):
        _reingest(tmp_path, root, corrected, policy="conflict_raise")
    assert len(_bars(root, "sec_000000")) == len(days), "a refused ingest writes nothing"


def test_a_key_absent_from_a_later_ingest_is_not_a_deletion(tmp_path: Path) -> None:
    root, days, frame = _first_ingest(tmp_path)
    _reingest(tmp_path, root, frame.drop(index=[3]).reset_index(drop=True), policy="dedupe_identical")
    bars = _bars(root, "sec_000000")
    assert len(bars) == len(days)
    dropped = bars.loc[bars["ts"] == pd.Timestamp(days[3], tz="America/New_York")]
    assert (
        len(dropped) == 1 and dropped.iloc[0]["revision"] == 1
    ), "a key absent from a later ingest is never a deletion; deletion is an explicit null-valued full revision"


def test_merge_bar_revisions_refuses_an_undeclared_policy() -> None:
    with pytest.raises(IngestError, match="revision_policy"):
        merge_bar_revisions(None, [], policy="last_writer_wins")


def test_a_second_batch_of_the_same_ingest_revises_within_the_call() -> None:
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-03", tz="America/New_York")])
    columns = {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10.0]}
    first = pd.DataFrame(columns, index=index)
    second = pd.DataFrame({**columns, "close": [1.75]}, index=index)
    frame, outcome = merge_bar_revisions(
        None,
        [
            BarBatch("sec_000000", first, INGESTED_AT, "vendor_a", "raw", origin="one.parquet"),
            BarBatch("sec_000000", second, LATER_INGEST, "vendor_a", "raw", origin="two.parquet"),
        ],
        policy="dedupe_identical",
    )
    assert list(frame["revision"]) == [1, 2]
    assert list(frame["close"]) == [1.5, 1.75]
    assert (outcome.written, outcome.revised, outcome.deduped) == (1, 1, 0)


# ---------------------------------------------------------------------------
# identity and coverage
# ---------------------------------------------------------------------------


def test_every_ingested_series_mints_a_provisional_sid_and_a_listing(tmp_path: Path) -> None:
    days = _sessions("2024-01-03", 4)
    path = tmp_path / "vendor.csv"
    pd.concat([_frame(days, "AAA"), _frame(days, "BBB", base=20.0)], ignore_index=True).to_csv(path, index=False)
    root = tmp_path / "data"
    report = ingest_local_files(
        root, [path], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=INGESTED_AT
    )
    assert report.symbols == {"AAA": "sec_000000", "BBB": "sec_000001"}
    securities = pd.read_parquet(root / "securities.parquet")
    assert set(securities["entity_status"]) == {"provisional"}
    listings = pd.read_parquet(root / "listings.parquet")
    assert set(zip(listings["vendor_symbol"], listings["sid"])) == {("AAA", "sec_000000"), ("BBB", "sec_000001")}
    assert set(listings["vendor"]) == {"vendor_a"}


def test_a_gap_longer_than_the_declared_threshold_is_flagged_and_a_shorter_one_is_not(tmp_path: Path) -> None:
    days = _sessions("2024-01-03", 60)
    kept = days[:10] + days[26:]  # sixteen missing sessions
    path = tmp_path / "gappy.csv"
    _frame(kept).to_csv(path, index=False)
    report = ingest_local_files(
        tmp_path / "flagged", [path], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=INGESTED_AT
    )
    assert len(report.gap_flags) == 1 and report.gap_flags[0]["missing_sessions"] == 16

    narrow = days[:10] + days[25:]  # fifteen missing sessions
    path = tmp_path / "narrow.csv"
    _frame(narrow).to_csv(path, index=False)
    unflagged = ingest_local_files(
        tmp_path / "clean", [path], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=INGESTED_AT
    )
    assert unflagged.gap_flags == ()


def test_a_series_outside_the_bundled_session_table_is_ingested_without_a_gap_scan(tmp_path: Path) -> None:
    days = [dt.date(2005, 12, 5) + dt.timedelta(days=7 * i) for i in range(4)]
    path = tmp_path / "old.csv"
    _frame(days).to_csv(path, index=False)
    report = ingest_local_files(
        tmp_path / "data", [path], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=INGESTED_AT
    )
    assert report.gap_flags == ()
    assert len(report.gap_scan_skipped) == 1
    assert sessions.TABLE_VERSION in report.gap_scan_skipped[0]["reason"]
    assert report.manifest.coverage["gap_scan_skipped"], "the skipped scan is recorded, never silent"


def test_the_manifest_records_the_earliest_session_table_release_covering_every_bar(tmp_path: Path) -> None:
    """The workspace's latest bar decides, so a later 2024 ingest does not undo a 2028 one."""
    root = tmp_path / "data"
    mapping = _mapping(tmp_path)
    early = tmp_path / "early.csv"
    _frame(_sessions("2024-01-03", 8)).to_csv(early, index=False)
    first = ingest_local_files(root, [early], mapping=mapping, fmt="csv", vintage="v1", ingested_at=INGESTED_AT)
    assert first.manifest.assumptions["session_table_version"] == "2006.2027-1"

    late = tmp_path / "late.csv"
    _frame(_sessions("2028-03-01", 8), symbol="BBB").to_csv(late, index=False)
    second = ingest_local_files(root, [late], mapping=mapping, fmt="csv", vintage="v1", ingested_at=LATER_INGEST)
    assert second.manifest.assumptions["session_table_version"] == "2006.2028-1"

    third = ingest_local_files(root, [early], mapping=mapping, fmt="csv", vintage="v1", ingested_at=LATER_INGEST)
    assert third.manifest.assumptions["session_table_version"] == "2006.2028-1"


def test_the_ingest_writes_the_event_tables_a_total_return_view_needs(tmp_path: Path) -> None:
    days = _sessions("2024-01-03", 4)
    path = tmp_path / "vendor.csv"
    _frame(days).to_csv(path, index=False)
    root = tmp_path / "data"
    ingest_local_files(root, [path], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=INGESTED_AT)
    dataset = Dataset.open(root)
    assert dataset.has_table("events") and dataset.has_table("adjustments")
    view = dataset.view("research_vintage", days[-1].isoformat(), vintage="v1")
    total = view.total_return(["sec_000000"], days[-1].isoformat())
    close = view.panel(["sec_000000"], days[-1].isoformat())
    # With no distributions the reinvested series is the close series; the two
    # differ only by the reinvestment factor's own float association, which is
    # why this is a tolerance and the revision tests above are exact.
    pd.testing.assert_frame_equal(total, close, check_exact=False, rtol=1e-12)


def test_an_overlapping_tail_reuses_the_sid_and_a_disjoint_one_mints_a_second(tmp_path: Path) -> None:
    """The identity rule is overlap, which is why a delta fetch re-requests overlap bars.

    A tail that shares a session with the stored listing interval is the same
    instrument and keeps its sid. A tail that begins after the stored interval
    ends overlaps nothing, so it matches nothing and mints a provisional sid —
    the same rule that makes a ticker reused after a delisting two instruments
    (§3.2). Both halves are asserted so the constraint on the fetch path is
    recorded rather than discovered later.
    """
    days = _sessions("2024-01-03", 6)
    root = tmp_path / "data"
    first = tmp_path / "first.csv"
    _frame(days[:3]).to_csv(first, index=False)
    ingest_local_files(root, [first], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=INGESTED_AT)

    overlapping = tmp_path / "overlapping.csv"
    _frame(days[2:], base=102.0).to_csv(overlapping, index=False)
    report = ingest_local_files(
        root, [overlapping], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=LATER_INGEST
    )
    assert report.symbols == {"AAA": "sec_000000"}
    assert report.minted_sids == (), "a sid is minted once and reused, which is the point of a surrogate key"
    assert _bars(root, "sec_000000")["ts"].nunique() == len(days), "one series, one bar file, one row per session"

    disjoint_root = tmp_path / "disjoint"
    ingest_local_files(
        disjoint_root, [first], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=INGESTED_AT
    )
    gap = tmp_path / "gap.csv"
    _frame(days[4:], base=104.0).to_csv(gap, index=False)
    second = ingest_local_files(
        disjoint_root, [gap], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=LATER_INGEST
    )
    assert second.symbols == {"AAA": "sec_000001"}
    assert second.minted_sids == ("sec_000001",)


def test_the_mapping_the_template_ships_is_the_one_this_reader_parses(tmp_path: Path) -> None:
    """``prism init`` writes ``mappings/twelvedata_daily.toml``; §3.2 prints it verbatim."""
    from prism.template import TWELVEDATA_MAPPING_TOML

    path = tmp_path / "twelvedata_daily.toml"
    path.write_text(TWELVEDATA_MAPPING_TOML, encoding="utf-8")
    mapping = IngestMapping.read(path)
    assert mapping.source.id == "twelvedata"
    assert mapping.source.lag_bars == 0
    assert mapping.source.revision_policy == "dedupe_identical"
    assert mapping.source.price_basis == "split_adjusted_at_ingest"
    assert mapping.source.gap_flag_sessions == 15
    assert mapping.source.redistributable is False
    assert mapping.columns["ts"] == "datetime"
    assert mapping.vendor_symbol == "symbol"
    assert mapping.match == "ticker_interval_overlap"


def test_the_manifest_names_every_source_it_orders(tmp_path: Path) -> None:
    days = _sessions("2024-01-03", 4)
    path = tmp_path / "vendor.csv"
    _frame(days).to_csv(path, index=False)
    root = tmp_path / "data"
    report = ingest_local_files(
        root, [path], mapping=_mapping(tmp_path), fmt="csv", vintage="v1", ingested_at=INGESTED_AT
    )
    for table in ("bars", "securities", "listings", "membership", "events", "adjustments"):
        assert report.manifest.priority_for(table) == ["vendor_a"]
    with pytest.raises(DatasetError):
        Dataset.open(root).view("research_vintage", days[-1].isoformat(), vintage="another_label")
