"""``prism data import-legacy``: the repository's own caches as the tables of §3.2.

The hard requirement of this stage is the round trip: what the import writes for
a symbol must be, cell for cell, what the loader's own file selection returns
for it. The loader picks "the first covering file in sorted-name order"
(``prism.io.loader:258-266``), so for the 579 single-file groups of the real
cache directory the comparison is exact by construction; for a group with more
than one file §3.2 requires every overlapping disagreement to be *reported*
rather than tolerated, because a disagreement means the two caches carry
different adjustment bases. Both halves are asserted here on a built cache
directory whose disagreement is planted deliberately.

The rest of this module pins the mappings §3.2 states in words: ingestion time
is the source file's modification time, availability is empty, every minted sid
is provisional, every bar is ``split_adjusted_at_ingest`` (decision D3), the
three reviewed tables of :mod:`prism.io.universe_sp500` become data rather than
code branches, and the un-ranged cache files are not imported.

Nothing here reads the real cache directory: the vendor bars are not
redistributable (§3.2, "Non-redistributable inputs") and are not in the
repository, so every fixture is built in ``tmp_path``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from prism.io import sessions
from prism.io.dataset import Dataset
from prism.io.legacy_import import (
    LEGACY_VENDOR,
    REVIEW_DATE,
    LegacyImportError,
    build_legacy_dataset,
    discover_legacy_inputs,
    import_legacy,
    round_trip_panel,
    sid_by_symbol,
)
from prism.io.security_master import OVERRIDES_FILENAME, SecurityOverrides
from prism.io.universe_sp500 import QUARANTINE_TABLE, RENAME_TABLE

#: Fixed modification times, so "ingested_at is the file's mtime" is a fact the
#: test states rather than reads back off whatever the filesystem happened to
#: record.
FIRST_FETCH = dt.datetime(2026, 6, 16, 18, 56, 55, tzinfo=dt.timezone(dt.timedelta(hours=-7)))
SECOND_FETCH = dt.datetime(2026, 7, 7, 18, 30, 2, tzinfo=dt.timezone(dt.timedelta(hours=-7)))
MEMBERSHIP_FETCH = dt.datetime(2026, 6, 21, 23, 10, 56, tzinfo=dt.timezone(dt.timedelta(hours=-7)))

SAMPLE_START = "2024-01-03"
SAMPLE_END = "2024-06-28"


def _sessions(first: str, last: str) -> list[dt.date]:
    return sessions.sessions_between(dt.date.fromisoformat(first), dt.date.fromisoformat(last))


def _bar_frame(days: list[dt.date], base: float) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day in days], name="datetime")
    return pd.DataFrame(
        {
            "open": [base + i for i in range(len(days))],
            "high": [base + 1.0 + i for i in range(len(days))],
            "low": [base - 1.0 + i for i in range(len(days))],
            "close": [base + 0.5 + i for i in range(len(days))],
            "volume": [1_000_000 + i for i in range(len(days))],
        },
        index=index,
    )


def _touch(path: Path, when: dt.datetime) -> Path:
    stamp = when.timestamp()
    os.utime(path, (stamp, stamp))
    return path


@pytest.fixture()
def cache(tmp_path: Path) -> Path:
    """A cache directory in the repository's own shape, with one planted disagreement.

    ``AAA`` has one range cache and one incremental-store file that agrees with
    it on the overlap. ``BBB`` has two range caches whose *names* both cover the
    sample — so the loader takes the first in sorted-name order — and whose
    contents disagree on one session, with the disagreeing file written later.
    ``ADS`` is a quarantined ticker of ``QUARANTINE_TABLE`` and its cache sits
    under ``data/quarantine``, as the real one does.
    """
    root = tmp_path / "cache"
    for name in ("store", "quarantine", "universe", "distributions"):
        (root / name).mkdir(parents=True, exist_ok=True)
    days = _sessions(SAMPLE_START, SAMPLE_END)

    _touch(_write(_bar_frame(days, 100.0), root / f"AAA_1d_{SAMPLE_START}_{SAMPLE_END}.parquet"), FIRST_FETCH)
    # The incremental store's tail is a slice of the same series: it agrees with
    # the range cache on the overlap, which is what an append that passed the
    # store's own rewrite check looks like.
    _touch(_write(_bar_frame(days, 100.0).iloc[-10:], root / "store" / "AAA_1d_bars.parquet"), SECOND_FETCH)

    _touch(_write(_bar_frame(days, 50.0), root / f"BBB_1d_2024-01-01_{SAMPLE_END}.parquet"), FIRST_FETCH)
    rewritten = _bar_frame(days, 50.0)
    rewritten.iloc[40, rewritten.columns.get_loc("close")] += 7.5
    _touch(_write(rewritten, root / f"BBB_1d_2024-01-02_{SAMPLE_END}.parquet"), SECOND_FETCH)

    _touch(
        _write(_bar_frame(days, 20.0), root / "quarantine" / f"ADS_1d_{SAMPLE_START}_{SAMPLE_END}.parquet"),
        FIRST_FETCH,
    )

    # The un-ranged legacy files, which the loader skips and the import must not read.
    _write(_bar_frame(days, 1.0), root / "AAA_1day.parquet")
    _write(_bar_frame(days, 1.0), root / "AAA_1h.parquet")

    membership = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "ADS", "FB", "ZZZ"],
            "start": pd.to_datetime(["1990-01-01", "2024-02-01", "1990-01-01", "1990-01-01", "1990-01-01"]),
            "end": pd.to_datetime(["2024-06-28", "2024-06-28", "2024-03-01", "2024-06-28", "2024-06-28"]),
        }
    )
    _touch(_write(membership, root / "universe" / "sp500_membership_2024-06-28.parquet"), MEMBERSHIP_FETCH)

    (root / "distributions" / "AAA.csv").write_text(
        "ex_date,pay_date,amount\n2024-03-15,2024-03-28,0.25\n1999-03-15,1999-03-28,0.10\n", encoding="utf-8"
    )
    (root / "distributions" / "AAA.provenance.json").write_text(
        json.dumps(
            {
                "captured_at": "2026-09-03T12:00:00-04:00",
                "sha256": "a" * 64,
                "source_url": "https://issuer.invalid/aaa",
            }
        ),
        encoding="utf-8",
    )
    return root


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


@pytest.fixture()
def imported(tmp_path: Path, cache: Path):
    return import_legacy(tmp_path / "data", cache_dir=cache, vintage="legacy_v1", index_id="sp500")


# ---------------------------------------------------------------------------
# the round trip
# ---------------------------------------------------------------------------


def test_the_import_round_trips_a_single_file_symbol_with_zero_difference(
    tmp_path: Path, cache: Path, imported
) -> None:
    trip = round_trip_panel(tmp_path / "data", cache_dir=cache, symbols=["AAA"], start=SAMPLE_START, end=SAMPLE_END)
    assert trip.exact, trip.to_dict()
    assert trip.max_abs_difference == 0.0
    assert trip.n_cells == len(_sessions(SAMPLE_START, SAMPLE_END))


def test_every_field_of_the_panel_round_trips_not_only_the_close(tmp_path: Path, cache: Path, imported) -> None:
    for field in ("open", "high", "low", "close", "volume"):
        trip = round_trip_panel(
            tmp_path / "data", cache_dir=cache, symbols=["AAA"], start=SAMPLE_START, end=SAMPLE_END, field=field
        )
        assert trip.exact, (field, trip.to_dict())


def test_two_caches_that_disagree_become_two_revisions_and_the_disagreement_is_reported(
    tmp_path: Path, cache: Path, imported
) -> None:
    """§3.2: a disagreement between two caches is a finding, never a tolerance."""
    sid = sid_by_symbol(tmp_path / "data")["BBB"]
    bars = pd.read_parquet(tmp_path / "data" / "bars" / f"{sid}_1d.parquet")
    revised = bars.loc[bars["revision"] == 2]
    assert len(revised) == 1, "exactly the one disagreeing row is a second revision"
    assert imported.rows_revised == 1

    trip = round_trip_panel(tmp_path / "data", cache_dir=cache, symbols=["BBB"], start=SAMPLE_START, end=SAMPLE_END)
    assert not trip.exact
    assert len(trip.disagreements) == 1
    finding = trip.disagreements[0]
    assert finding["symbol"] == "BBB"
    assert pd.Timestamp(finding["ts"]) == pd.Timestamp(revised.iloc[0]["ts"])
    assert finding["imported"] - finding["loader"] == pytest.approx(7.5)


def test_the_import_records_the_ticker_to_sid_mapping_it_assigned(tmp_path: Path, imported) -> None:
    mapping = sid_by_symbol(tmp_path / "data")
    assert mapping == imported.symbol_to_sid
    assert set(mapping) == {"AAA", "ADS", "BBB"}
    assert all(sid.startswith("sec_") for sid in mapping.values()), "a ticker is never a sid (decision D2)"


# ---------------------------------------------------------------------------
# the declared mappings of §3.2
# ---------------------------------------------------------------------------


def test_ingestion_time_is_the_files_modification_time_and_availability_is_empty(tmp_path: Path, imported) -> None:
    sid = imported.symbol_to_sid["AAA"]
    bars = pd.read_parquet(tmp_path / "data" / "bars" / f"{sid}_1d.parquet")
    stamps = {pd.Timestamp(value) for value in bars["ingested_at"]}
    assert stamps == {pd.Timestamp(FIRST_FETCH), pd.Timestamp(SECOND_FETCH)} or stamps == {pd.Timestamp(FIRST_FETCH)}
    assert bars["available_at"].isna().all()
    assert imported.manifest.assumptions["ingestion_time_quality"] == "file_mtime"


def test_every_bar_is_split_adjusted_at_ingest_and_every_sid_is_provisional(tmp_path: Path, imported) -> None:
    for sid in set(imported.symbol_to_sid.values()):
        bars = pd.read_parquet(tmp_path / "data" / "bars" / f"{sid}_1d.parquet")
        assert set(bars["price_basis"]) == {"split_adjusted_at_ingest"}, "decision D3"
        assert set(bars["source"]) == {LEGACY_VENDOR}
    securities = pd.read_parquet(tmp_path / "data" / "securities.parquet")
    minted = securities.loc[securities["source"] == LEGACY_VENDOR]
    assert set(minted["entity_status"]) == {"provisional"}
    assert imported.manifest.sources[LEGACY_VENDOR]["lag_bars"] == 0
    assert imported.manifest.sources[LEGACY_VENDOR]["price_basis"] == "split_adjusted_at_ingest"
    assert imported.manifest.sources[LEGACY_VENDOR]["redistributable"] is False


def test_the_unranged_files_are_not_imported_and_are_counted(tmp_path: Path, cache: Path, imported) -> None:
    inputs = discover_legacy_inputs(cache)
    assert sorted(p.name for p in inputs.unranged) == ["AAA_1day.parquet", "AAA_1h.parquet"]
    coverage = imported.manifest.coverage["inputs"]
    assert coverage["unranged_files_not_imported"] == ["AAA_1day.parquet", "AAA_1h.parquet"]
    sid = imported.symbol_to_sid["AAA"]
    bars = pd.read_parquet(tmp_path / "data" / "bars" / f"{sid}_1d.parquet")
    assert bars["close"].min() > 50.0, "the un-ranged file's prices never entered the table"


def test_a_membership_end_at_the_files_own_asof_becomes_an_open_interval(tmp_path: Path, imported) -> None:
    membership = pd.read_parquet(tmp_path / "data" / "membership.parquet")
    rows = {row["sid"]: row for row in membership.to_dict("records")}
    aaa = rows[imported.symbol_to_sid["AAA"]]
    assert aaa["valid_to"] is None, "an end equal to the file's as-of is an open interval, not a closure"
    assert aaa["valid_from"] is None and bool(aaa["floor_unknown"]) is True
    bbb = rows[imported.symbol_to_sid["BBB"]]
    assert bbb["valid_from"] == "2024-02-01" and bool(bbb["floor_unknown"]) is False
    assert set(membership["source"]) == {"wikipedia@2024-06-28"}


def test_a_quarantined_ticker_is_attested_and_holds_no_index_seat(tmp_path: Path, imported) -> None:
    """``QUARANTINE_TABLE`` becomes an attestation, and its seat is a counted skip.

    The vendor series under a quarantined ticker is a known *different*
    instrument, so attributing the index seat to its sid would be the very
    misattribution the table exists to prevent. The seat joins the counted
    skip-list instead — the same measured-survivorship-leak treatment
    ``prism.io.universe_sp500.resolve_coverage`` already applies — while the
    bars stay readable so an earlier record that consumed them still reproduces.
    """
    sid = imported.symbol_to_sid["ADS"]
    securities = pd.read_parquet(tmp_path / "data" / "securities.parquet")
    quarantined = securities.loc[(securities["sid"] == sid) & (securities["entity_status"] == "quarantined")]
    assert len(quarantined) == 1
    assert quarantined.iloc[0]["quarantine_reason"] == QUARANTINE_TABLE["ADS"]
    assert quarantined.iloc[0]["valid_at"].startswith(REVIEW_DATE.isoformat())

    membership = pd.read_parquet(tmp_path / "data" / "membership.parquet")
    assert sid not in set(membership["sid"]), "a quarantined identity holds no index seat"
    skipped = imported.membership[0]["quarantined_skipped"]
    assert [entry["ticker"] for entry in skipped] == ["ADS"]

    assert (tmp_path / "data" / OVERRIDES_FILENAME).is_file()
    overrides = SecurityOverrides.read(tmp_path / "data" / OVERRIDES_FILENAME)
    assert [(a.sid, a.action) for a in overrides.attestations] == [(sid, "quarantine")]

    bars = pd.read_parquet(tmp_path / "data" / "bars" / f"{sid}_1d.parquet")
    assert len(bars) > 0, "the quarantined cache is imported, not pretended absent (§3.2)"


def test_a_renamed_ticker_is_recorded_as_a_remap_and_never_dated_from_nothing(imported) -> None:
    remapped = {entry["ticker"]: entry for entry in imported.remapped_tickers}
    assert "FB" in remapped and remapped["FB"]["resolves_to"] == RENAME_TABLE["FB"]
    assert "carries no date" in remapped["FB"]["note"]
    unresolved = {entry["ticker"] for entry in imported.membership[0]["unresolved"]}
    assert "META" in unresolved, "the seat follows the successor's ticker, which this fixture has no cache for"


def test_distributions_become_events_and_an_undatable_one_is_recorded_not_dropped_silently(
    tmp_path: Path, imported
) -> None:
    events = pd.read_parquet(tmp_path / "data" / "events.parquet")
    assert len(events) == 1
    row = events.iloc[0]
    assert row["sid"] == imported.symbol_to_sid["AAA"]
    assert row["event_type"] == "cash_distribution"
    assert row["effective_date"] == "2024-03-15"
    assert row["source"] == "issuer_csv"
    assert pd.Timestamp(row["ingested_at"]) == pd.Timestamp("2026-09-03T12:00:00-04:00")
    payload = json.loads(row["payload"])
    assert payload["amount_per_share"] == 0.25 and payload["pay_date"] == "2024-03-28"
    assert payload["provenance_sha256"] == "a" * 64

    outside = {entry["symbol"]: entry for entry in imported.events_outside_session_table}
    assert outside["AAA"]["n_dropped"] == 1 and outside["AAA"]["first"] == "1999-03-15"
    assert sessions.TABLE_VERSION in outside["AAA"]["reason"]
    assert imported.manifest.coverage["events_outside_session_table"], "the exclusion is in the manifest, not silent"


def test_the_imported_dataset_reads_back_through_the_as_of_view(tmp_path: Path, imported) -> None:
    dataset = Dataset.open(tmp_path / "data")
    assert dataset.manifest.id == imported.manifest.id
    decision = "2024-03-20"
    view = dataset.view("research_vintage", decision, vintage="legacy_v1")
    sids = sorted(set(imported.symbol_to_sid.values()))
    panel = view.panel(sids, decision)
    assert list(panel.index[-1:]) == [pd.Timestamp(decision, tz="America/New_York")]
    assert set(view.membership("sp500")) == {imported.symbol_to_sid["AAA"], imported.symbol_to_sid["BBB"]}
    events = view.events(sids, "2024-01-01", decision)
    assert len(events) == 1
    total = view.total_return([imported.symbol_to_sid["AAA"]], decision)
    assert (
        total.iloc[-1, 0] > panel.loc[:, imported.symbol_to_sid["AAA"]].iloc[-1]
    ), "the reinvested series exceeds the as-traded close once a distribution has been paid"


def test_the_import_refuses_a_root_that_already_holds_tables(tmp_path: Path, cache: Path, imported) -> None:
    with pytest.raises(LegacyImportError, match="already holds"):
        import_legacy(tmp_path / "data", cache_dir=cache, vintage="legacy_v2")


def test_a_series_wholly_outside_the_session_table_is_skipped_and_recorded(tmp_path: Path, cache: Path) -> None:
    days = [dt.date(2005, 6, 6) + dt.timedelta(days=7 * i) for i in range(6)]
    _touch(_write(_bar_frame(days, 5.0), cache / "OLD_1d_2005-06-01_2005-07-15.parquet"), FIRST_FETCH)
    report = import_legacy(tmp_path / "data", cache_dir=cache, vintage="legacy_v1")
    assert "OLD" not in report.symbol_to_sid
    assert [entry["vendor_symbol"] for entry in report.skipped_series] == ["OLD"]
    assert sessions.TABLE_VERSION in report.skipped_series[0]["reason"]
    assert report.manifest.coverage["skipped_series"], "a skipped series is recorded in the manifest"


def test_a_series_that_starts_before_the_session_table_is_truncated_not_dropped(tmp_path: Path, cache: Path) -> None:
    """A bar the session table cannot date is not importable; the rest of the series still is.

    Availability is the close of a bar's own session plus the source's lag, so
    a bar outside the table has no date the importer may assign it. Dropping
    the whole series over that would lose every instrument whose history
    predates the table, so only the outside rows go, counted per series in the
    manifest ``coverage``. The property is the table's edge, not any
    particular year: these rows sit before 2006 because decision D12 moved the
    edge there.
    """
    old_days = [dt.date(2005, 6, 6) + dt.timedelta(days=7 * i) for i in range(6)]
    days = old_days + _sessions(SAMPLE_START, SAMPLE_END)
    _touch(_write(_bar_frame(days, 5.0), cache / f"LONG_1d_2005-06-01_{SAMPLE_END}.parquet"), FIRST_FETCH)
    report = import_legacy(tmp_path / "data", cache_dir=cache, vintage="legacy_v1")
    assert "LONG" in report.symbol_to_sid
    assert report.skipped_series == ()
    entry = {row["vendor_symbol"]: row for row in report.truncated_series}["LONG"]
    assert entry["n_rows_outside"] == len(old_days)
    assert entry["last_outside"] == max(old_days).isoformat()
    bars = pd.read_parquet(tmp_path / "data" / "bars" / f"{report.symbol_to_sid['LONG']}_1d.parquet")
    assert len(bars) == len(_sessions(SAMPLE_START, SAMPLE_END))
    assert report.manifest.coverage["truncated_series"], "a truncation is recorded in the manifest"


def test_the_stage_two_slice_still_writes_its_own_ticker_keyed_dataset(tmp_path: Path, cache: Path) -> None:
    """The two import paths are separate on purpose, and this pins the separation.

    §3.2 says the stage 2 reproduction "reads the caches directly through the
    existing loader and is unaffected by" the legacy import. The reproduction of
    ``results/demotion_b1`` runs through :func:`build_legacy_dataset`, whose sid
    is the ticker; ``import_legacy`` mints surrogate sids. A change to one must
    not move the other, so both are asserted here on one cache directory.
    """
    stage_two = build_legacy_dataset(
        tmp_path / "stage2",
        symbols=["AAA"],
        cache_dir=cache,
        start=SAMPLE_START,
        end=SAMPLE_END,
        vintage="stage2",
        index_id="sp500",
        membership_file=cache / "universe" / "sp500_membership_2024-06-28.parquet",
    )
    assert stage_two.sids == ("AAA",), "the stage 2 slice keys by ticker and mints no identifier"
    assert (tmp_path / "stage2" / "bars" / "AAA_1d.parquet").is_file()
    securities = pd.read_parquet(tmp_path / "stage2" / "securities.parquet")
    assert list(securities["source"]) == ["legacy_cache"]

    stage_three = import_legacy(tmp_path / "stage3", cache_dir=cache, vintage="legacy_v1")
    assert stage_three.symbol_to_sid["AAA"].startswith("sec_")
    assert stage_three.manifest.id != stage_two.manifest.id


def test_extra_indexes_share_one_dataset_and_each_answers_its_own_membership(tmp_path: Path, cache: Path) -> None:
    """One set of bars, several index ids: arms that differ only in universe read identical bars."""
    from prism.io.dataset import Dataset

    mid = pd.DataFrame(
        {"ticker": ["BBB"], "start": pd.to_datetime(["2024-01-01"]), "end": pd.to_datetime(["2024-06-28"])}
    )
    mid_path = _write(mid, cache / "universe" / "sp400_membership_2024-06-28.parquet")
    report = build_legacy_dataset(
        tmp_path / "multi",
        symbols=["AAA", "BBB"],
        cache_dir=cache,
        start=SAMPLE_START,
        end=SAMPLE_END,
        vintage="multi",
        index_id="sp500",
        membership_file=cache / "universe" / "sp500_membership_2024-06-28.parquet",
        extra_indexes={"sp400": mid_path},
    )
    ds = Dataset.open(tmp_path / "multi")
    assert ds.manifest.coverage["index_membership"] == {"sp500": 2, "sp400": 1}
    assert set(ds.manifest.assumptions["membership_files_sha256"]) == {"sp500", "sp400"}
    view = ds.view("research_vintage", "2024-03-01", vintage="multi")
    assert view.membership("sp400") == {"BBB"}
    assert view.membership("sp500") == {"AAA", "BBB"}
    single = build_legacy_dataset(
        tmp_path / "single",
        symbols=["AAA", "BBB"],
        cache_dir=cache,
        start=SAMPLE_START,
        end=SAMPLE_END,
        vintage="multi",
        index_id="sp500",
        membership_file=cache / "universe" / "sp500_membership_2024-06-28.parquet",
    )
    assert "membership_files_sha256" not in single.manifest.assumptions
    assert report.manifest.id != single.manifest.id
    with pytest.raises(Exception, match="declared twice"):
        build_legacy_dataset(
            tmp_path / "dup",
            symbols=["AAA"],
            cache_dir=cache,
            start=SAMPLE_START,
            end=SAMPLE_END,
            vintage="multi",
            index_id="sp500",
            membership_file=cache / "universe" / "sp500_membership_2024-06-28.parquet",
            extra_indexes={"sp500": mid_path},
        )
