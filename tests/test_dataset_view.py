"""The stage 1 as-of read path over a synthetic dataset (engine spec §3.2).

The property the whole engine rests on is negative: a view opened for a
decision date must never hand back anything that was not knowable at that
date's close (SPEC.md N2). The bar tests below therefore sweep decision dates
across the sample and assert both halves of it — nothing later than the
decision date, and the decision date's own bar *is* returned, because a
zero-lag bar becomes available at the close the book decides on. A test that
asserted only the first half would pass on a view that returned nothing.

The second group covers what stage 3a lifted: ``replay_recorded`` and the
revision selection rule both answer here now. Their own semantics — the two
worked examples of §3.2 row for row in both modes, the source-priority tie, the
pre-ingestion refusal — are pinned in ``tests/test_dataset_asof.py``; what
belongs in this file is that the synthetic fixture's stage 3 scenarios read
correctly through the same view the stage 1 accessors use.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from prism.io import sessions
from prism.io.dataset import Dataset, DatasetError
from prism.io.manifest import Manifest
from prism.io.synthetic import EQUITY_INDEX, ETF_INDEX, sid_of, write_synthetic_dataset

START = dt.date(2026, 1, 1)
END = dt.date(2026, 12, 31)
VINTAGE = "synthetic_v1"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    root = tmp_path_factory.mktemp("view") / "data"
    write_synthetic_dataset(root, seed=7, scenarios="all", n_equities=14, n_etfs=3, start=START, end=END)
    return Dataset.open(root, verify_files=True)


@pytest.fixture(scope="module")
def grid() -> list[dt.date]:
    return sessions.sessions_between(START, END)


def _view(dataset: Dataset, day: dt.date | str, **kwargs):
    day = day if isinstance(day, str) else day.isoformat()
    return dataset.view("research_vintage", day, vintage=VINTAGE, **kwargs)


# ---- causality -------------------------------------------------------------


def test_no_bar_is_ever_returned_later_than_the_decision_date(dataset: Dataset, grid: list[dt.date]) -> None:
    sids = [sid_of(n) for n in (12, 13, 14)]
    for day in grid[60::37]:
        view = _view(dataset, day)
        frame = view.bars(sids, day.isoformat())
        assert not frame.empty
        assert frame["ts"].dt.date.max() <= day
        assert frame["ts"].dt.date.max() == day, "a zero-lag bar is available at the close that decides on it"


def test_the_panel_obeys_the_same_horizon(dataset: Dataset, grid: list[dt.date]) -> None:
    sids = [sid_of(n) for n in (12, 13)]
    day = grid[120]
    panel = _view(dataset, day).panel(sids, day.isoformat())
    assert panel.index.max().date() == day
    assert list(panel.columns) == sids
    assert panel.notna().all().all()


def test_asking_beyond_the_decision_date_is_refused(dataset: Dataset, grid: list[dt.date]) -> None:
    view = _view(dataset, grid[100])
    later = grid[101].isoformat()
    with pytest.raises(DatasetError, match="later than decision date"):
        view.bars([sid_of(12)], later)
    with pytest.raises(DatasetError, match="later than decision date"):
        view.events([sid_of(12)], grid[0].isoformat(), later)
    with pytest.raises(DatasetError, match="later than decision date"):
        view.resolve("SYN012", later)


def test_a_view_one_session_earlier_drops_exactly_that_session(dataset: Dataset, grid: list[dt.date]) -> None:
    sids = [sid_of(12)]
    day, previous = grid[150], grid[149]
    at_day = _view(dataset, day).bars(sids, day.isoformat())
    at_previous = _view(dataset, previous).bars(sids, previous.isoformat())
    assert len(at_day) == len(at_previous) + 1
    pd.testing.assert_frame_equal(at_day.iloc[:-1], at_previous)


def test_the_cutoff_is_the_session_close_and_honours_an_early_close(dataset: Dataset) -> None:
    regular = _view(dataset, "2026-11-25")
    early = _view(dataset, "2026-11-27")
    assert regular.cutoff == "2026-11-25T16:00:00-05:00"
    assert early.cutoff == "2026-11-27T13:00:00-05:00"
    assert sessions.is_early_close(dt.date(2026, 11, 27))


def test_the_sample_decision_offset_moves_the_cutoff(dataset: Dataset) -> None:
    offset = _view(dataset, "2026-11-25", decision_offset_minutes=30)
    assert offset.cutoff == "2026-11-25T16:30:00-05:00"
    assert offset.describe()["decision_offset_minutes"] == 30


def test_a_non_session_decision_date_is_refused(dataset: Dataset) -> None:
    with pytest.raises(DatasetError, match="is not a session"):
        _view(dataset, "2026-07-04")
    with pytest.raises(DatasetError, match="is not a session"):
        _view(dataset, "2026-12-25")


# ---- both modes over the synthetic fixture ---------------------------------


def test_replay_recorded_needs_a_clock_and_takes_the_declared_one(dataset: Dataset) -> None:
    with pytest.raises(DatasetError, match="neither a recorded decided_at nor a declared decision_time"):
        dataset.view("replay_recorded", "2026-06-15")
    view = dataset.view("replay_recorded", "2026-06-15", decision_time="21:30 America/New_York")
    assert view.cutoff == "2026-06-15T21:30:00-04:00"
    assert view.cutoff_basis == "declared_decision_time"


def test_replay_recorded_refuses_a_vintage(dataset: Dataset) -> None:
    with pytest.raises(DatasetError, match="vintage is refused in replay_recorded"):
        dataset.view("replay_recorded", "2026-06-15", vintage=VINTAGE, decision_time="21:30 America/New_York")


def test_more_than_one_visible_revision_selects_the_highest(dataset: Dataset) -> None:
    outside = Manifest.read(dataset.root / "manifest.json").coverage["sids_outside_every_index"]
    revised = [sid for sid in outside if len(pd.read_parquet(dataset.root / f"bars/{sid}_1d.parquet")) > 251]
    assert revised, "the revision and rebase scenarios must have written extra revisions"
    view = _view(dataset, "2026-12-30")
    for sid in revised:
        stored = pd.read_parquet(dataset.root / f"bars/{sid}_1d.parquet")
        stored = stored.loc[stored["ts"].dt.date <= dt.date(2026, 12, 30)]
        assert stored["ts"].duplicated().any(), f"{sid} must carry more than one revision of some bar"
        frame = view.bars([sid], "2026-12-30")
        assert not frame["ts"].duplicated().any(), "the selection rule leaves one row per (sid, ts)"
        assert len(frame) == stored["ts"].nunique()
        for stamp, group in stored.groupby("ts"):
            selected = frame.loc[frame["ts"] == stamp]
            assert int(selected.iloc[0]["revision"]) == int(group["revision"].max())


def test_total_return_answers_from_the_events_table(dataset: Dataset) -> None:
    # Stage 3a lifted this refusal: the reinvested series is derived from bars
    # plus cash_distribution events. Its own behaviour is pinned in
    # tests/test_price_basis.py; what belongs here is that the view answers on
    # the calendar and columns the price panel uses.
    view = _view(dataset, "2026-06-15")
    close = view.panel([sid_of(3)], "2026-06-15", "close")
    total = view.total_return([sid_of(3)], "2026-06-15")
    assert list(total.columns) == [sid_of(3)]
    assert total.index.equals(close.index)
    assert (total.dropna() > 0).all().all()


def test_an_unknown_mode_is_refused_before_anything_is_read(dataset: Dataset) -> None:
    with pytest.raises(DatasetError, match="mode must be one of"):
        dataset.view("as_of", "2026-06-15")  # type: ignore[arg-type]


# ---- the vintage contract --------------------------------------------------


def test_research_vintage_requires_its_vintage_and_refuses_another(dataset: Dataset) -> None:
    with pytest.raises(DatasetError, match="vintage is required"):
        dataset.view("research_vintage", "2026-06-15")
    with pytest.raises(DatasetError, match="does not name this manifest"):
        dataset.view("research_vintage", "2026-06-15", vintage="some_other_label")


def test_research_vintage_refuses_a_replay_decision_time(dataset: Dataset) -> None:
    with pytest.raises(DatasetError, match="decision_time belongs to replay_recorded"):
        dataset.view("research_vintage", "2026-06-15", vintage=VINTAGE, decision_time="21:30")


def test_only_files_the_manifest_lists_are_read(dataset: Dataset) -> None:
    """A vintage is the file list; a file that appeared beside it is not part of the dataset."""
    manifest = dataset.manifest
    trimmed = Manifest(
        provenance=manifest.provenance,
        as_of=manifest.as_of,
        files=tuple(pair for pair in manifest.files if pair[0] != "events.parquet"),
        sources=manifest.sources,
        source_priority=manifest.source_priority,
        universe=manifest.universe,
        vintage=manifest.vintage,
    )
    view = Dataset(dataset.root, trimmed).view("research_vintage", "2026-06-15", vintage=VINTAGE)
    with pytest.raises(DatasetError, match="not listed in manifest"):
        view.events([sid_of(3)], "2026-01-02", "2026-06-15")


def test_a_row_whose_source_the_manifest_does_not_order_is_refused(dataset: Dataset, tmp_path: Path) -> None:
    manifest = dataset.manifest
    narrowed = Manifest(
        provenance=manifest.provenance,
        as_of=manifest.as_of,
        files=manifest.files,
        sources={"synthetic": manifest.sources["synthetic"]},
        source_priority={table: ["synthetic"] for table in manifest.source_priority},
        universe=manifest.universe,
        vintage=manifest.vintage,
    )
    view = Dataset(dataset.root, narrowed).view("research_vintage", "2026-12-30", vintage=VINTAGE)
    with pytest.raises(DatasetError, match="does not order"):
        view.bars([sid_of(1)], "2026-12-30")


# ---- as-of semantics per table --------------------------------------------


def test_membership_opens_on_the_effective_date_not_the_announcement(dataset: Dataset) -> None:
    membership = pd.read_parquet(dataset.root / "membership.parquet")
    equities = membership.loc[membership["index_id"] == EQUITY_INDEX]
    added = equities.loc[equities["valid_from"] > equities["valid_from"].min()].iloc[0]
    effective = dt.date.fromisoformat(added["valid_from"])
    announced = dt.date.fromisoformat(added["announce_date"])
    assert announced < effective
    assert added["sid"] not in _view(dataset, announced).membership(EQUITY_INDEX)
    assert added["sid"] not in _view(dataset, sessions.previous_session(effective)).membership(EQUITY_INDEX)
    assert added["sid"] in _view(dataset, effective).membership(EQUITY_INDEX)


def test_membership_closes_after_the_last_day_of_the_interval(dataset: Dataset) -> None:
    membership = pd.read_parquet(dataset.root / "membership.parquet")
    removed = membership.loc[membership["valid_to"].notna()].iloc[0]
    last_day = dt.date.fromisoformat(removed["valid_to"])
    assert removed["sid"] in _view(dataset, last_day).membership(EQUITY_INDEX)
    assert removed["sid"] not in _view(dataset, sessions.next_session(last_day)).membership(EQUITY_INDEX)


def test_both_index_memberships_resolve_from_one_dataset(dataset: Dataset) -> None:
    view = _view(dataset, "2026-06-15")
    equities = view.membership(EQUITY_INDEX)
    etfs = view.membership(ETF_INDEX)
    assert etfs == {sid_of(15), sid_of(16), sid_of(17)}
    assert equities.isdisjoint(etfs)
    assert view.securities(sorted(etfs))["asset_class"].eq("etf").all()


def test_a_capability_change_is_invisible_before_its_valid_at(dataset: Dataset) -> None:
    capabilities = pd.read_parquet(dataset.root / "capabilities.parquet")
    rows = capabilities.loc[capabilities["sid"] == sid_of(9)].sort_values("valid_at")
    change = pd.Timestamp(rows.iloc[1]["valid_at"])
    before = sessions.previous_session(change.date())
    assert float(_view(dataset, before).capabilities([sid_of(9)]).iloc[0]["quantity_increment"]) == 1.0
    assert float(_view(dataset, change.date()).capabilities([sid_of(9)]).iloc[0]["quantity_increment"]) == 5.0


def test_capabilities_refuses_a_sid_with_no_visible_row(dataset: Dataset) -> None:
    with pytest.raises(DatasetError, match="refuses rather than returning a partial frame"):
        _view(dataset, "2026-06-15").capabilities([sid_of(9), "sec_999999"])


def test_an_event_is_invisible_before_its_effective_date(dataset: Dataset) -> None:
    events = pd.read_parquet(dataset.root / "events.parquet")
    row = events.loc[events["event_type"] == "cash_distribution"].iloc[0]
    ex_date = dt.date.fromisoformat(row["effective_date"])
    before = sessions.previous_session(ex_date)
    assert _view(dataset, before).events([row["sid"]], START.isoformat(), before.isoformat()).empty
    visible = _view(dataset, ex_date).events([row["sid"]], START.isoformat(), ex_date.isoformat())
    assert len(visible) == 1
    assert json.loads(visible.iloc[0]["payload"])["pay_date"] > row["effective_date"]


def test_a_ticker_resolves_to_one_sid_per_date_across_a_reuse(dataset: Dataset) -> None:
    listings = pd.read_parquet(dataset.root / "listings.parquet")
    reuse = listings.loc[listings["ticker"] == "RUSE"].sort_values("valid_from")
    first, second = reuse.iloc[0], reuse.iloc[1]
    view = _view(dataset, "2026-12-30")
    assert view.resolve("RUSE", first["valid_to"]) == first["sid"]
    assert view.resolve("RUSE", second["valid_from"]) == second["sid"]
    assert first["sid"] != second["sid"]


def test_resolving_an_unknown_ticker_is_refused(dataset: Dataset) -> None:
    with pytest.raises(DatasetError, match="must be single-valued"):
        _view(dataset, "2026-06-15").resolve("NOSUCH", "2026-06-15")


def test_a_dataset_with_no_series_table_says_so(dataset: Dataset) -> None:
    with pytest.raises(DatasetError, match="lists no series table"):
        _view(dataset, "2026-06-15").series("cash_3m", "2026-01-02", "2026-06-15")


def test_bars_for_a_sid_the_manifest_does_not_carry_are_refused(dataset: Dataset) -> None:
    with pytest.raises(DatasetError, match="lists no bars for sid"):
        _view(dataset, "2026-06-15").bars(["sec_999999"], "2026-06-15")


def test_describe_carries_what_the_result_record_copies(dataset: Dataset) -> None:
    described = _view(dataset, "2026-06-15").describe()
    assert described["mode"] == "research_vintage"
    assert described["vintage"] == VINTAGE
    assert described["cutoff"] == "2026-06-15T16:00:00-04:00"
    assert described["data_id"] == dataset.manifest.id


def test_the_halted_sessions_are_absent_rather_than_filled(dataset: Dataset) -> None:
    events = pd.read_parquet(dataset.root / "events.parquet")
    halt = events.loc[events["event_type"] == "halt"].iloc[0]
    payload = json.loads(halt["payload"])
    end = dt.date.fromisoformat(payload["end"])
    frame = _view(dataset, end).bars([halt["sid"]], end.isoformat())
    present = set(frame["ts"].dt.date)
    halted = sessions.sessions_between(dt.date.fromisoformat(payload["start"]), sessions.previous_session(end))
    assert present.isdisjoint(halted)
    assert end in present
