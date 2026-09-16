"""The two worked as-of examples of §3.2, row for row, in both visibility modes.

Section 3.2 works two examples by hand and states the answer to every query in
both modes. They are the acceptance test for the as-of read path, and they are
built here as fixtures rather than paraphrased, because the value of a worked
example is lost the moment the code and the document drift.

What each example is for:

Example A is a historical dataset first downloaded long after the bars it
covers. It separates the two clocks at their widest: under ``research_vintage``
a 2024 decision sees the 2024 bars, because the world had published them; under
``replay_recorded`` the same decision sees nothing at all and is refused, because
this workspace held nothing in 2024. A view that answered the second question
with the first answer would let a backtest read a file that did not exist.

Example B is a correction after an earlier decision and then a withdrawal. It is
the case where the two clocks return *different revisions of the same row* — a
correction published at 09:00 and stored here at 20:00 is inside a 16:00
research cutoff and outside a 16:00 replay cutoff — and the case where a
null-valued full revision must return null rather than fall back to the value it
replaced. Both halves are asserted, in both modes, and the cutoff sensitivity of
the last query is asserted on both sides of the clock.

The fixtures are written by hand and not through ``io/synthetic.py``: the point
is that these exact rows produce these exact answers, so the rows are literals
here, in the order and with the values §3.2 prints.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from prism.io.dataset import (
    REFERENCE_DECISION_TIME,
    Dataset,
    DatasetError,
    read_decided_at,
)
from prism.io.manifest import Manifest, hash_tree

SID = "sec_000001"
VINTAGE = "asof_examples"
VENDOR = "vendor_a"
VENDOR_B = "vendor_b"

BAR_COLUMNS = [
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_basis",
    "available_at",
    "ingested_at",
    "source",
    "revision",
]

# The declared replay clock for every query §3.2 leaves undated: "a declared
# decision_time of 21:30 in replay_recorded".
NIGHTLY = REFERENCE_DECISION_TIME
# The clock the last query of example B states explicitly.
AT_CLOSE = "16:00 America/New_York"


# ---- fixture construction --------------------------------------------------


def _bar(
    ts: str,
    close: float | None,
    *,
    revision: int = 1,
    available_at: str | None = None,
    ingested_at: str,
    source: str = VENDOR,
) -> dict[str, Any]:
    """One bar row in the §3.2 schema. ``close = None`` is an explicit null."""
    value = float("nan") if close is None else float(close)
    return {
        "ts": pd.Timestamp(ts, tz="America/New_York"),
        "open": value,
        "high": value,
        "low": value,
        "close": value,
        "volume": float("nan") if close is None else 1_000_000.0,
        "price_basis": "split_adjusted_at_ingest",
        "available_at": available_at,
        "ingested_at": ingested_at,
        "source": source,
        "revision": revision,
    }


def _source(lag_bars: int = 0) -> dict[str, Any]:
    return {
        "lag_bars": lag_bars,
        "revision_policy": "always_new_revision",
        "price_basis": "split_adjusted_at_ingest",
        "url": None,
        "sha256": None,
        "redistributable": True,
    }


def _write(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    sources: dict[str, dict[str, Any]] | None = None,
    priority: list[str] | None = None,
    sid: str = SID,
) -> Dataset:
    """Write one bar file plus the manifest that freezes it, and open the dataset."""
    sources = sources or {VENDOR: _source()}
    priority = priority or list(sources)
    root.mkdir(parents=True, exist_ok=True)
    (root / "bars").mkdir(exist_ok=True)
    path = root / "bars" / f"{sid}_1d.parquet"
    pd.DataFrame(rows, columns=BAR_COLUMNS).to_parquet(path, index=False)
    manifest = Manifest(
        provenance="market",
        as_of=str(max(row["ts"] for row in rows).isoformat()),
        files=tuple(hash_tree(root, [path])),
        sources=sources,
        source_priority={"bars": priority},
        universe=sid,
        vintage=VINTAGE,
        assumptions={"availability": "session close of ts plus lag_bars sessions; available_at empty"},
    )
    manifest.write(root / "manifest.json")
    return Dataset.open(root, verify_files=True)


# ---- example A -------------------------------------------------------------

# §3.2, "Worked example A: a historical dataset first downloaded on 2026-09-07".
# One sid, one source with lag_bars = 0, all rows revision 1, available_at
# empty, ingested_at 2026-09-07T20:00 (2026-09-07 is not a session; ingestion is
# not a market event).
INGESTED_A = "2026-09-07T20:00:00-04:00"
EXAMPLE_A = [
    _bar("2024-03-14", 50.0, ingested_at=INGESTED_A),
    _bar("2024-03-15", 51.0, ingested_at=INGESTED_A),
    _bar("2024-03-18", 52.0, ingested_at=INGESTED_A),
]


@pytest.fixture(scope="module")
def example_a(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return _write(tmp_path_factory.mktemp("example_a") / "data", EXAMPLE_A)


def test_example_a_the_ingestion_day_is_not_a_session(example_a: Dataset) -> None:
    """The example's own aside: ingestion is not a market event."""
    from prism.io import sessions

    assert not sessions.is_session(dt.date(2026, 9, 7))
    assert sessions.is_session(dt.date(2026, 9, 8))


def test_example_a_research_vintage_at_2024_03_15_sees_two_rows(example_a: Dataset) -> None:
    view = example_a.view("research_vintage", "2024-03-15", vintage=VINTAGE)
    assert view.cutoff == "2024-03-15T16:00:00-04:00"
    frame = view.bars([SID], "2024-03-15")
    assert [stamp.date().isoformat() for stamp in frame["ts"]] == ["2024-03-14", "2024-03-15"]
    assert list(frame["close"]) == [50.0, 51.0]


def test_example_a_replay_recorded_at_2024_03_15_is_refused(example_a: Dataset) -> None:
    """Refused, because the first ingestion of the bars table is 2026-09-07."""
    view = example_a.view("replay_recorded", "2024-03-15", decision_time=NIGHTLY)
    with pytest.raises(DatasetError) as excinfo:
        view.bars([SID], "2024-03-15")
    message = str(excinfo.value)
    assert "replay_recorded refuses decision date 2024-03-15" in message
    assert "bars" in message
    assert "2026-09-07T20:00:00-04:00" in message


def test_example_a_the_refusal_holds_whatever_the_declared_clock(example_a: Dataset) -> None:
    for clock in (AT_CLOSE, NIGHTLY, "23:59 America/New_York"):
        view = example_a.view("replay_recorded", "2024-03-15", decision_time=clock)
        with pytest.raises(DatasetError, match="first ingestion"):
            view.bars([SID], "2024-03-15")


def test_example_a_research_vintage_at_2026_09_08_sees_all_three(example_a: Dataset) -> None:
    frame = example_a.view("research_vintage", "2026-09-08", vintage=VINTAGE).bars([SID], "2026-09-08")
    assert [stamp.date().isoformat() for stamp in frame["ts"]] == ["2024-03-14", "2024-03-15", "2024-03-18"]
    assert list(frame["close"]) == [50.0, 51.0, 52.0]


def test_example_a_replay_recorded_at_2026_09_08_sees_all_three(example_a: Dataset) -> None:
    """2026-09-07T20:00 precedes a 21:30 cutoff on 2026-09-08."""
    view = example_a.view("replay_recorded", "2026-09-08", decision_time=NIGHTLY)
    assert view.cutoff == "2026-09-08T21:30:00-04:00"
    frame = view.bars([SID], "2026-09-08")
    assert [stamp.date().isoformat() for stamp in frame["ts"]] == ["2024-03-14", "2024-03-15", "2024-03-18"]
    assert list(frame["close"]) == [50.0, 51.0, 52.0]


def test_example_a_the_two_modes_agree_once_the_workspace_holds_the_rows(example_a: Dataset) -> None:
    research = example_a.view("research_vintage", "2026-09-08", vintage=VINTAGE).bars([SID], "2026-09-08")
    replay = example_a.view("replay_recorded", "2026-09-08", decision_time=NIGHTLY).bars([SID], "2026-09-08")
    pd.testing.assert_frame_equal(research, replay)


# ---- example B -------------------------------------------------------------

# §3.2, "Worked example B: a correction after an earlier decision, then a
# withdrawal". One sid, one source, price_basis split_adjusted_at_ingest,
# lag_bars = 0. Three revisions of the same bar.
EXAMPLE_B = [
    _bar(
        "2024-03-14",
        50.0,
        revision=1,
        available_at="2024-03-14T16:05:00-04:00",
        ingested_at="2024-03-14T20:00:00-04:00",
    ),
    _bar(
        "2024-03-14",
        55.0,
        revision=2,
        available_at="2024-03-20T09:00:00-04:00",
        ingested_at="2024-03-20T20:00:00-04:00",
    ),
    _bar(
        "2024-03-14",
        None,
        revision=3,
        available_at="2024-03-25T09:00:00-04:00",
        ingested_at="2024-03-25T20:00:00-04:00",
    ),
]


@pytest.fixture(scope="module")
def example_b(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return _write(tmp_path_factory.mktemp("example_b") / "data", EXAMPLE_B)


def _b_views(dataset: Dataset, day: str, clock: str = NIGHTLY):
    return (
        dataset.view("research_vintage", day, vintage=VINTAGE),
        dataset.view("replay_recorded", day, decision_time=clock),
    )


@pytest.mark.parametrize(
    ("day", "revision", "close"),
    [("2024-03-15", 1, 50.0), ("2024-03-22", 2, 55.0)],
)
def test_example_b_both_modes_agree_on_the_dated_queries(
    example_b: Dataset, day: str, revision: int, close: float
) -> None:
    for view in _b_views(example_b, day):
        frame = view.bars([SID], day)
        assert len(frame) == 1, view.mode
        assert int(frame.iloc[0]["revision"]) == revision, view.mode
        assert float(frame.iloc[0]["close"]) == close, view.mode


def test_example_b_the_null_revision_returns_null_in_both_modes(example_b: Dataset) -> None:
    """Decision 2024-03-26: revision 3, close null; 55.0 is not returned."""
    for view in _b_views(example_b, "2024-03-26"):
        frame = view.bars([SID], "2024-03-26")
        assert len(frame) == 1, view.mode
        assert int(frame.iloc[0]["revision"]) == 3, view.mode
        assert pd.isna(frame.iloc[0]["close"]), view.mode
        assert 55.0 not in set(frame["close"].dropna()), view.mode


def test_example_b_the_null_revision_reaches_the_panel_as_a_missing_bar(example_b: Dataset) -> None:
    for view in _b_views(example_b, "2024-03-26"):
        panel = view.panel([SID], "2024-03-26")
        assert list(panel.columns) == [SID], view.mode
        assert len(panel) == 1, view.mode
        assert pd.isna(panel.iloc[0][SID]), view.mode


def test_example_b_the_two_clocks_disagree_at_a_16_00_cutoff(example_b: Dataset) -> None:
    """Decision 2024-03-20 with a 16:00 cutoff in both modes.

    ``research_vintage`` returns revision 2, because its availability 09:00
    precedes the cutoff; ``replay_recorded`` returns revision 1, because
    revision 2 was ingested at 20:00, after the cutoff.
    """
    research, replay = _b_views(example_b, "2024-03-20", clock=AT_CLOSE)
    assert research.cutoff == "2024-03-20T16:00:00-04:00"
    assert replay.cutoff == "2024-03-20T16:00:00-04:00"
    research_row = research.bars([SID], "2024-03-20").iloc[0]
    replay_row = replay.bars([SID], "2024-03-20").iloc[0]
    assert (int(research_row["revision"]), float(research_row["close"])) == (2, 55.0)
    assert (int(replay_row["revision"]), float(replay_row["close"])) == (1, 50.0)


def test_example_b_the_same_day_nightly_replay_clock_returns_revision_2(example_b: Dataset) -> None:
    """Under the 21:30 replay clock the same query returns revision 2."""
    view = example_b.view("replay_recorded", "2024-03-20", decision_time=NIGHTLY)
    assert view.cutoff == "2024-03-20T21:30:00-04:00"
    row = view.bars([SID], "2024-03-20").iloc[0]
    assert (int(row["revision"]), float(row["close"])) == (2, 55.0)


def test_example_b_the_cutoff_and_its_basis_travel_with_the_answer(example_b: Dataset) -> None:
    """Which is why the cutoff is recorded on the view rather than assumed."""
    at_close = example_b.view("replay_recorded", "2024-03-20", decision_time=AT_CLOSE).describe()
    nightly = example_b.view("replay_recorded", "2024-03-20", decision_time=NIGHTLY).describe()
    assert at_close["cutoff"] != nightly["cutoff"]
    assert at_close["cutoff_basis"] == nightly["cutoff_basis"] == "declared_decision_time"
    assert at_close["decision_time"] == AT_CLOSE
    assert at_close["mode"] == "replay_recorded"
    assert at_close["vintage"] is None


def test_example_b_the_undated_queries_are_invariant_to_the_replay_clock(example_b: Dataset) -> None:
    """§3.2: "every one of them is invariant to that choice except the last"."""
    for day, revision in (("2024-03-15", 1), ("2024-03-22", 2), ("2024-03-26", 3)):
        for clock in (AT_CLOSE, NIGHTLY):
            frame = example_b.view("replay_recorded", day, decision_time=clock).bars([SID], day)
            assert int(frame.iloc[0]["revision"]) == revision, (day, clock)


# ---- the record the modes carry -------------------------------------------


def test_the_research_view_records_the_vintage_the_lags_and_the_assumption(example_a: Dataset) -> None:
    described = example_a.view("research_vintage", "2026-09-08", vintage=VINTAGE).describe()
    assert described["mode"] == "research_vintage"
    assert described["vintage"] == VINTAGE
    assert described["cutoff_basis"] == "session_close"
    assert described["lag_bars"] == {VENDOR: 0}
    assert "corrections the source made silently before this vintage" in described["availability_assumption"]
    assert "taken as known at the row's own time of record" in described["availability_assumption"]
    assert "highest revision" in described["selection_rule"]


def test_the_replay_view_records_the_ingestion_statement(example_a: Dataset) -> None:
    described = example_a.view("replay_recorded", "2026-09-08", decision_time=NIGHTLY).describe()
    assert described["vintage"] is None
    assert "stored it at or before the decision's wall clock" in described["availability_assumption"]


# ---- the clock ------------------------------------------------------------


def test_replay_recorded_refuses_a_date_with_neither_clock(example_a: Dataset) -> None:
    with pytest.raises(DatasetError) as excinfo:
        example_a.view("replay_recorded", "2026-09-08")
    message = str(excinfo.value)
    assert "neither a recorded decided_at nor a declared decision_time" in message
    assert REFERENCE_DECISION_TIME in message


def test_a_recorded_decided_at_beats_the_declared_clock(tmp_path: Path) -> None:
    dataset = _write(tmp_path / "data", EXAMPLE_B)
    ledger = tmp_path / "targets.jsonl"
    ledger.write_text(
        '{"decision_bar": "2024-03-20", "decided_at": "2024-03-20T16:00:00-04:00", "targets": {}}\n'
        '{"decision_bar": "2024-03-22", "equity": 1.0, "targets": {}}\n',
        encoding="utf-8",
    )
    recorded = read_decided_at(ledger)
    assert recorded == {"2024-03-20": "2024-03-20T16:00:00-04:00"}
    with_ledger = Dataset.open(dataset.root, decided_at=recorded)
    # The recorded wall clock wins over the declared nightly clock, so the
    # answer is revision 1 even though 21:30 was asked for.
    view = with_ledger.view("replay_recorded", "2024-03-20", decision_time=NIGHTLY)
    assert view.cutoff_basis == "recorded_decided_at"
    assert view.cutoff == "2024-03-20T16:00:00-04:00"
    assert int(view.bars([SID], "2024-03-20").iloc[0]["revision"]) == 1
    # A session the ledger recorded no wall clock for falls back to the
    # declared clock rather than to a guess.
    fallback = with_ledger.view("replay_recorded", "2024-03-22", decision_time=NIGHTLY)
    assert fallback.cutoff_basis == "declared_decision_time"
    with pytest.raises(DatasetError, match="neither a recorded decided_at"):
        with_ledger.view("replay_recorded", "2024-03-22")


def test_a_clock_without_a_zone_is_refused(example_a: Dataset) -> None:
    for bad in ("21:30", "21:30 Mars/Olympus", "not a time America/New_York"):
        with pytest.raises(DatasetError):
            example_a.view("replay_recorded", "2026-09-08", decision_time=bad)


# ---- the tie rule across sources and revisions -----------------------------

# The two-source fixture §3.2 names: source A revision 2 and source B revision
# 1 are both visible for one key and priority decides. Revisions are never
# compared across sources, so B's lower revision wins when B outranks A.
TWO_SOURCE = [
    _bar("2024-03-14", 50.0, revision=1, ingested_at="2024-03-14T20:00:00-04:00", source=VENDOR),
    _bar("2024-03-14", 51.0, revision=2, ingested_at="2024-03-15T20:00:00-04:00", source=VENDOR),
    _bar("2024-03-14", 60.0, revision=1, ingested_at="2024-03-15T20:00:00-04:00", source=VENDOR_B),
]


@pytest.mark.parametrize(
    ("priority", "close"),
    [([VENDOR, VENDOR_B], 51.0), ([VENDOR_B, VENDOR], 60.0)],
)
def test_source_priority_decides_and_revisions_are_not_compared_across_sources(
    tmp_path: Path, priority: list[str], close: float
) -> None:
    dataset = _write(
        tmp_path / f"data_{priority[0]}",
        TWO_SOURCE,
        sources={VENDOR: _source(), VENDOR_B: _source()},
        priority=priority,
    )
    for view in _b_views(dataset, "2024-03-22"):
        frame = view.bars([SID], "2024-03-22")
        assert len(frame) == 1, view.mode
        assert float(frame.iloc[0]["close"]) == close, view.mode
        assert frame.iloc[0]["source"] == priority[0], view.mode


def test_a_source_the_manifest_does_not_order_is_refused(tmp_path: Path) -> None:
    dataset = _write(
        tmp_path / "data",
        TWO_SOURCE,
        sources={VENDOR: _source(), VENDOR_B: _source()},
        priority=[VENDOR, VENDOR_B],
    )
    narrowed = Manifest(
        provenance=dataset.manifest.provenance,
        as_of=dataset.manifest.as_of,
        files=dataset.manifest.files,
        sources={VENDOR: _source()},
        source_priority={"bars": [VENDOR]},
        universe=dataset.manifest.universe,
        vintage=VINTAGE,
    )
    view = Dataset(dataset.root, narrowed).view("research_vintage", "2024-03-22", vintage=VINTAGE)
    with pytest.raises(DatasetError, match="does not order"):
        view.bars([SID], "2024-03-22")


def test_two_rows_with_the_same_key_source_and_revision_are_an_ingestion_defect(tmp_path: Path) -> None:
    rows = [
        _bar("2024-03-14", 50.0, revision=1, ingested_at="2024-03-14T20:00:00-04:00"),
        _bar("2024-03-14", 51.0, revision=1, ingested_at="2024-03-15T20:00:00-04:00"),
    ]
    dataset = _write(tmp_path / "data", rows)
    for view in _b_views(dataset, "2024-03-22"):
        with pytest.raises(DatasetError, match="ingestion defect"):
            view.bars([SID], "2024-03-22")


# ---- causality, in both modes ---------------------------------------------


def test_no_query_returns_a_bar_later_than_its_decision_date(example_a: Dataset) -> None:
    """The negative property the whole engine rests on (SPEC.md N2)."""
    for view in _b_views(example_a, "2024-03-15"):
        if view.mode == "replay_recorded":
            continue  # refused for this dataset; covered above
        frame = view.bars([SID], "2024-03-15")
        assert frame["ts"].dt.date.max() <= dt.date(2024, 3, 15)
    for day in ("2024-03-14", "2024-03-15", "2024-03-18", "2026-09-08"):
        research = example_a.view("research_vintage", day, vintage=VINTAGE)
        replay = example_a.view("replay_recorded", day, decision_time=NIGHTLY)
        assert research.bars([SID], day)["ts"].dt.date.max() <= dt.date.fromisoformat(day)
        with pytest.raises(DatasetError, match="later than decision date"):
            research.bars([SID], "2026-09-09")
        with pytest.raises(DatasetError, match="later than decision date"):
            replay.bars([SID], "2026-09-09")


def test_asking_beyond_the_decision_date_is_refused_in_both_modes(example_b: Dataset) -> None:
    for view in _b_views(example_b, "2024-03-22"):
        with pytest.raises(DatasetError, match="later than decision date"):
            view.bars([SID], "2024-03-25")
        with pytest.raises(DatasetError, match="later than decision date"):
            view.panel([SID], "2024-03-25")


# ---- a missing bar never erases a holding ---------------------------------


def test_the_withdrawn_bar_carries_the_holding_stale_rather_than_erasing_it(tmp_path: Path) -> None:
    """§3.2: a missing bar never erases a holding or simulates a sale.

    The series prices 03-13 at 49.0 and then withdraws 03-14 entirely. At a
    decision on 03-26 the mark is the 03-13 close, flagged stale over the nine
    sessions since — not zero, not 55.0 (the revision the withdrawal replaced),
    and not forward-filled into the panel, which still reports the bar as
    missing.
    """
    rows = [
        _bar(
            "2024-03-13",
            49.0,
            revision=1,
            available_at="2024-03-13T16:05:00-04:00",
            ingested_at="2024-03-13T20:00:00-04:00",
        ),
        *EXAMPLE_B,
    ]
    dataset = _write(tmp_path / "data", rows)
    for view in _b_views(dataset, "2024-03-26"):
        marks = view.marks([SID], "2024-03-26")
        assert float(marks.loc[SID, "last_price"]) == 49.0, view.mode
        assert marks.loc[SID, "last_price_bar"] == "2024-03-13", view.mode
        assert marks.loc[SID, "valuation_status"] == "stale", view.mode
        assert int(marks.loc[SID, "sessions_stale"]) == 9, view.mode
        panel = view.panel([SID], "2024-03-26")
        assert pd.isna(panel.loc[panel.index[-1], SID]), view.mode


def test_a_priced_bar_on_the_decision_date_is_not_stale(example_a: Dataset) -> None:
    marks = example_a.view("research_vintage", "2024-03-18", vintage=VINTAGE).marks([SID], "2024-03-18")
    assert marks.loc[SID, "valuation_status"] == "priced"
    assert int(marks.loc[SID, "sessions_stale"]) == 0
    assert float(marks.loc[SID, "last_price"]) == 52.0


def test_a_sid_with_no_priced_bar_is_refused_rather_than_marked_at_nothing(tmp_path: Path) -> None:
    dataset = _write(tmp_path / "data", EXAMPLE_B)
    view = dataset.view("research_vintage", "2024-03-26", vintage=VINTAGE)
    with pytest.raises(DatasetError, match="refuses rather than valuing a holding at nothing"):
        view.marks([SID], "2024-03-26")
