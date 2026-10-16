"""Property tests for the walk-forward splitters and the frozen fold table.

For :class:`PurgedWalkForward`, that the two leakage controls do what they
claim: purge keeps test-overlapping training labels out of train, and embargo
keeps a buffer of bars after each test out of subsequent train sets.

For :class:`RollingWalkForward` and :class:`FoldTable` (engine stage 2), that
the recorded fold geometry is *reproduced*, not merely resembled — every
boundary of ``results/demotion_b1/folds.json`` (21 folds) and
``results/trend_t0/folds.json`` (74 folds) compared date for date against the
session index each run actually split, and the table's content hash pinned.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.validation.walk_forward import (
    INCLUSIVE_SESSION_DATES,
    FoldGeometryError,
    FoldTable,
    PurgedWalkForward,
    RollingWalkForward,
    resolve_fold_geometry,
)


@pytest.fixture
def series_1000() -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(1000)})


def test_n_splits_yielded(series_1000: pd.DataFrame) -> None:
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=5, embargo_pct=0.0)
    folds = list(wfo.split(series_1000))
    assert len(folds) == 5


def test_purge_horizon_enforced(series_1000: pd.DataFrame) -> None:
    """No training row may sit within `purge_horizon` bars before test_start."""
    h = 10
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=h, embargo_pct=0.0)
    for train_idx, test_idx in wfo.split(series_1000):
        test_start = int(test_idx.min())
        assert (
            int(train_idx.max()) + h <= test_start
        ), f"Purge violated: train_max={train_idx.max()} + h={h} > test_start={test_start}"


def test_train_and_test_disjoint(series_1000: pd.DataFrame) -> None:
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=5, embargo_pct=0.01)
    for train_idx, test_idx in wfo.split(series_1000):
        assert np.intersect1d(train_idx, test_idx).size == 0


def test_test_folds_non_overlapping(series_1000: pd.DataFrame) -> None:
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=5, embargo_pct=0.0)
    seen: set[int] = set()
    for _, test_idx in wfo.split(series_1000):
        as_set = set(int(i) for i in test_idx)
        assert seen.isdisjoint(as_set), "Test folds overlap"
        seen.update(as_set)


def test_embargo_excludes_prior_test_zones(series_1000: pd.DataFrame) -> None:
    """Embargo after fold k's test must be excluded from every later fold's
    train. (Effect appears at fold k+2+ because fold k+1's train naturally
    ends at fold k+1's test_start, which is where fold k's embargo begins
    when test slices are adjacent.)
    """
    n = 1000
    embargo_pct = 0.02
    embargo = int(embargo_pct * n)
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=5, embargo_pct=embargo_pct)
    folds = list(wfo.split(series_1000))
    for i, (_, test_i) in enumerate(folds):
        embargo_zone = set(
            range(
                int(test_i.max()) + 1,
                int(test_i.max()) + 1 + embargo,
            )
        )
        for later_train, _ in folds[i + 2 :]:
            assert embargo_zone.isdisjoint(
                set(int(j) for j in later_train)
            ), f"Embargo zone from fold {i} leaked into a later fold's train"


def test_expanding_window_grows(series_1000: pd.DataFrame) -> None:
    wfo = PurgedWalkForward(n_splits=4, purge_horizon=5, embargo_pct=0.0, expanding=True)
    train_sizes = [len(tr) for tr, _ in wfo.split(series_1000)]
    assert train_sizes == sorted(train_sizes), f"Expanding train should grow: {train_sizes}"


def test_rolling_window_bounded(series_1000: pd.DataFrame) -> None:
    """Rolling window train length is bounded by 2 × test_size (less purge)."""
    wfo = PurgedWalkForward(n_splits=4, purge_horizon=5, embargo_pct=0.0, expanding=False)
    test_size = len(series_1000) // (wfo.n_splits + 1)
    for train_idx, _ in wfo.split(series_1000):
        assert len(train_idx) <= test_size * 2


def test_fold_counts_recorded_when_all_yield(series_1000: pd.DataFrame) -> None:
    """After a full drain, the realized counters match the requested count."""
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=5, embargo_pct=0.0)
    folds = list(wfo.split(series_1000))
    assert len(folds) == 5
    assert wfo.n_yielded_ == 5
    assert wfo.n_skipped_ == 0


def test_skipped_fold_warns_and_counts() -> None:
    """purge_horizon >= test_size empties fold 0's training pool: the skip
    must emit a RuntimeWarning and be counted, never pass silently.
    """
    df = pd.DataFrame({"x": np.arange(40)})  # test_size = 10
    wfo = PurgedWalkForward(n_splits=3, purge_horizon=10, embargo_pct=0.0)
    with pytest.warns(RuntimeWarning, match="fold 0 of 3 skipped"):
        folds = list(wfo.split(df))
    assert len(folds) == 2
    assert wfo.n_yielded_ == 2
    assert wfo.n_skipped_ == 1


def test_strict_raises_on_skipped_fold() -> None:
    """strict=True turns the fold-count discrepancy into a hard error."""
    df = pd.DataFrame({"x": np.arange(40)})
    wfo = PurgedWalkForward(n_splits=3, purge_horizon=10, embargo_pct=0.0, strict=True)
    with pytest.raises(ValueError, match="fold 0 of 3 skipped"):
        list(wfo.split(df))


def test_strict_noop_when_all_folds_yield(series_1000: pd.DataFrame) -> None:
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=5, embargo_pct=0.01, strict=True)
    assert len(list(wfo.split(series_1000))) == 5


def test_rejects_bad_args(series_1000: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_splits"):
        PurgedWalkForward(n_splits=0, purge_horizon=5)
    with pytest.raises(ValueError, match="purge_horizon"):
        PurgedWalkForward(n_splits=5, purge_horizon=-1)
    with pytest.raises(ValueError, match="embargo_pct"):
        PurgedWalkForward(n_splits=5, purge_horizon=5, embargo_pct=1.5)


def test_rejects_too_short_input() -> None:
    wfo = PurgedWalkForward(n_splits=5, purge_horizon=5)
    with pytest.raises(ValueError, match="at least"):
        list(wfo.split(pd.DataFrame({"x": np.arange(8)})))


def test_zero_embargo_is_noop(series_1000: pd.DataFrame) -> None:
    """embargo=0 → fold k+2's train includes the bar right after fold k's
    test (which would otherwise be embargoed).
    """
    wfo = PurgedWalkForward(n_splits=3, purge_horizon=0, embargo_pct=0.0)
    folds = list(wfo.split(series_1000))
    _, test_0 = folds[0]
    later_train, _ = folds[2]
    assert int(test_0.max()) + 1 in set(int(i) for i in later_train)


def test_embargo_actually_excludes(series_1000: pd.DataFrame) -> None:
    """Positive control for the noop test: embargo>0 → fold k+2's train
    excludes the bar right after fold k's test.
    """
    wfo = PurgedWalkForward(n_splits=3, purge_horizon=0, embargo_pct=0.02)
    folds = list(wfo.split(series_1000))
    _, test_0 = folds[0]
    later_train, _ = folds[2]
    assert int(test_0.max()) + 1 not in set(int(i) for i in later_train)


# --------------------------------------------------------------------------
# RollingWalkForward and FoldTable (engine stage 2)
#
# The deliverable these tests guard is a reproduction: the rolling splitter has
# to cut the *recorded* fold boundaries, date for date, from the session index
# the recorded runs actually split. Counting 21 folds is not the claim; landing
# on 2021-03-29 / 2021-03-30 / 2021-06-28 is.
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLD_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "folds"

# The recorded evidence. Read only, never written: results/ is the research
# record, and a reproduction that differs from it is a finding, not a nudge.
RECORDED_FOLDS = {
    "demotion_b1": REPO_ROOT / "results" / "demotion_b1" / "folds.json",
    "trend_t0": REPO_ROOT / "results" / "trend_t0" / "folds.json",
}

# The geometry both recorded configurations declare (walk block of
# results/demotion_b1/config.json and results/trend_t0/config.json).
RECORDED_GEOMETRY = dict(formation_bars=312, test_bars=63, step_bars=None, min_test_bars=20)

# Pinned so "stable across runs" is checked against a constant rather than
# against another live computation. Changing one of these means the geometry,
# the canonical form, or the recorded evidence moved; all three are findings.
RECORDED_TABLE_SHA256 = {
    "demotion_b1": "fec02ca011e8c66309ab9e414403d1a6412a10569ff47bac775fa754ef3384f8",
    "trend_t0": "c17447526a06363f8edb33f9921b81e5d707769bad3ce84230d3acf276b0884c",
}

RECORDED_CASES = [
    pytest.param("demotion_b1", "demotion_b1_sessions.json", 21, id="momentum"),
    pytest.param("trend_t0", "trend_t0_sessions.json", 74, id="trend"),
]


def _sessions(fixture_name: str) -> pd.DatetimeIndex:
    """The session index a recorded run split, from its tracked fixture."""
    payload = json.loads((FOLD_FIXTURES / fixture_name).read_text(encoding="utf-8"))
    index = pd.DatetimeIndex(pd.to_datetime(payload["sessions"])).tz_localize(payload["timezone"])
    assert len(index) == payload["n_sessions"], f"{fixture_name} header disagrees with its own rows"
    return index


def _recorded_rows(case: str) -> list[dict]:
    return json.loads(RECORDED_FOLDS[case].read_text(encoding="utf-8"))


def _recorded_table(case: str) -> FoldTable:
    return FoldTable.from_json(RECORDED_FOLDS[case], table_id=f"{case}_folds", timezone="America/New_York")


@pytest.mark.parametrize("case,fixture,n_folds", RECORDED_CASES)
def test_rolling_reproduces_recorded_boundaries_date_for_date(case, fixture, n_folds) -> None:
    """Every recorded fold's four boundary dates and two row counts, exactly."""
    sessions = _sessions(fixture)
    recorded = _recorded_rows(case)
    assert len(recorded) == n_folds

    folds = list(RollingWalkForward(**RECORDED_GEOMETRY).iter_folds(len(sessions)))
    assert len(folds) == len(recorded), f"{case}: fold count moved"

    for row, fold in zip(recorded, folds):
        produced = {
            "fold": fold.fold,
            "formation_start": sessions[fold.formation_start].isoformat(),
            "formation_end": sessions[fold.formation_stop - 1].isoformat(),
            "test_start": sessions[fold.test_start].isoformat(),
            "test_end": sessions[fold.test_stop - 1].isoformat(),
            "formation_rows": fold.formation_rows,
            "test_rows": fold.test_rows,
        }
        assert produced == {key: row[key] for key in produced}, f"{case}: fold {row['fold']} moved"


def test_recorded_momentum_geometry_is_the_one_the_spec_names() -> None:
    """Anchor the parametrized reproduction on the literal recorded dates."""
    recorded = _recorded_rows("demotion_b1")
    first, last = recorded[0], recorded[-1]
    assert first["formation_start"] == "2020-01-02T00:00:00-05:00"
    assert first["formation_end"] == "2021-03-29T00:00:00-04:00"
    assert first["test_start"] == "2021-03-30T00:00:00-04:00"
    assert first["test_end"] == "2021-06-28T00:00:00-04:00"
    assert (first["formation_rows"], first["test_rows"]) == (312, 63)
    assert last["test_end"] == "2026-06-12T00:00:00-04:00"
    assert last["test_rows"] == 48


@pytest.mark.parametrize("case,fixture,n_folds", RECORDED_CASES)
def test_generated_fold_table_hashes_equal_to_the_recorded_one(case, fixture, n_folds) -> None:
    """The reproduction proof: same geometry, same content hash, no differences."""
    generated = RollingWalkForward(**RECORDED_GEOMETRY).fold_table(_sessions(fixture), table_id=f"{case}_folds")
    recorded = _recorded_table(case)
    assert generated.differences(recorded) == []
    generated.assert_matches(recorded)
    assert generated.content_hash == recorded.content_hash


@pytest.mark.parametrize("case,fixture,n_folds", RECORDED_CASES)
def test_recorded_table_hash_is_pinned(case, fixture, n_folds) -> None:
    """Stable across runs: the digest matches a constant, not a re-computation."""
    table = _recorded_table(case)
    assert len(table) == n_folds
    assert table.content_hash == RECORDED_TABLE_SHA256[case]
    assert _recorded_table(case).content_hash == table.content_hash


def test_fold_table_hash_changes_when_a_boundary_moves() -> None:
    """One session of drift in one boundary must not hash the same."""
    rows = _recorded_rows("demotion_b1")
    baseline = FoldTable.from_records(rows, table_id="baseline", timezone="America/New_York")
    # Each shift is chosen to stay inside its fold's ordering, so what the test
    # observes is the hash reacting to the move and not the validator refusing it.
    for key, days in (("formation_start", -1), ("formation_end", -1), ("test_start", 1), ("test_end", -1)):
        nudged = [dict(row) for row in rows]
        nudged[7][key] = (pd.Timestamp(nudged[7][key]) + pd.Timedelta(days=days)).isoformat()
        moved = FoldTable.from_records(nudged, table_id="baseline", timezone="America/New_York")
        assert moved.content_hash != baseline.content_hash, f"moving {key} left the hash unchanged"


def test_fold_table_hash_ignores_provenance_but_not_geometry() -> None:
    """Renaming a table or its source must not revoke the geometry's identity."""
    rows = _recorded_rows("demotion_b1")
    a = FoldTable.from_records(rows, table_id="one", source="a.json", timezone="America/New_York")
    b = FoldTable.from_records(rows, table_id="two", source="b.json", timezone="America/New_York")
    assert a.content_hash == b.content_hash
    assert a.fold_geometry()["table_id"] != b.fold_geometry()["table_id"]
    assert a.fold_geometry()["sha256"] == a.content_hash


def test_fold_table_hash_is_invariant_under_equivalent_timezone_forms() -> None:
    """A fixed -05:00 offset and America/New_York are one geometry, not two."""
    offsets = FoldTable.from_json(RECORDED_FOLDS["demotion_b1"], table_id="offsets")
    zoned = FoldTable.from_json(RECORDED_FOLDS["demotion_b1"], table_id="zoned", timezone="America/New_York")
    assert offsets.content_hash == zoned.content_hash


def test_fold_table_toml_round_trip_preserves_the_hash(tmp_path) -> None:
    """The registration form is lossless for the geometry it declares."""
    table = _recorded_table("demotion_b1")
    written = table.write_toml(tmp_path / "folds.toml")
    text = written.read_text(encoding="utf-8")
    assert 'timezone = "America/New_York"' in text
    assert "formation_start = 2020-01-02" in text
    reread = FoldTable.from_toml(written)
    assert reread.content_hash == table.content_hash
    assert reread.differences(table) == []


def test_fold_table_toml_refuses_to_drop_an_undeclared_timezone(tmp_path) -> None:
    table = FoldTable.from_json(RECORDED_FOLDS["demotion_b1"], table_id="offsets")
    assert table.timezone is None
    with pytest.raises(FoldGeometryError, match="declares no timezone"):
        table.to_toml()


def test_fold_table_toml_refuses_a_boundary_with_a_time_of_day() -> None:
    rows = _recorded_rows("demotion_b1")
    rows = [dict(row) for row in rows]
    rows[0]["test_end"] = "2021-06-28T09:30:00-04:00"
    table = FoldTable.from_records(rows, table_id="intraday", timezone="America/New_York")
    with pytest.raises(FoldGeometryError, match="not local midnight"):
        table.to_toml()


def test_last_fold_is_truncated_not_dropped() -> None:
    """The recorded 21st fold holds 48 test bars because the sample ran out."""
    sessions = _sessions("demotion_b1_sessions.json")
    folds = list(RollingWalkForward(**RECORDED_GEOMETRY).iter_folds(len(sessions)))
    assert [fold.test_rows for fold in folds[:-1]] == [63] * 20
    assert folds[-1].test_rows == 48
    assert folds[-1].test_stop == len(sessions) - 1


def test_a_tail_shorter_than_min_test_bars_is_dropped_not_scored() -> None:
    splitter = RollingWalkForward(formation_bars=10, test_bars=5, min_test_bars=3)
    # One full fold needs 10 + 5 bars plus the reserved next-open bar.
    assert [f.test_rows for f in splitter.iter_folds(16)] == [5]
    # 5 more bars would start a second fold, but only 2 would be testable.
    assert [f.test_rows for f in splitter.iter_folds(18)] == [5]
    # A third bar of tail clears min_test_bars, so the stub becomes a fold.
    assert [f.test_rows for f in splitter.iter_folds(19)] == [5, 3]


def test_a_bar_is_reserved_past_every_fold_for_the_next_open_fill() -> None:
    """Positions closed at fold end fill at the next open, so that bar must exist."""
    for n_obs in range(16, 60):
        folds = list(RollingWalkForward(formation_bars=10, test_bars=5, min_test_bars=3).iter_folds(n_obs))
        for fold in folds:
            assert fold.next_open_position == fold.test_stop
            assert fold.next_open_position <= n_obs - 1
            assert fold.next_open_position not in set(fold.test_positions.tolist())


def test_formation_window_never_touches_its_test_window() -> None:
    """Point-in-time causality, at the geometry level: no shared bar, no gap."""
    sessions = _sessions("demotion_b1_sessions.json")
    for fold in RollingWalkForward(**RECORDED_GEOMETRY).iter_folds(len(sessions)):
        assert fold.formation_stop == fold.test_start
        assert int(fold.formation_positions.max()) < int(fold.test_positions.min())


def test_test_windows_tile_without_overlap_or_gap_when_step_is_null() -> None:
    sessions = _sessions("demotion_b1_sessions.json")
    folds = list(RollingWalkForward(**RECORDED_GEOMETRY).iter_folds(len(sessions)))
    for earlier, later in zip(folds, folds[1:]):
        assert later.test_start == earlier.test_stop
    covered = np.concatenate([fold.test_positions for fold in folds])
    assert covered.tolist() == list(range(folds[0].test_start, folds[-1].test_stop))


def test_explicit_step_larger_than_the_test_window_leaves_gaps() -> None:
    splitter = RollingWalkForward(formation_bars=10, test_bars=5, step_bars=8, min_test_bars=5)
    assert splitter.effective_step_bars == 8
    folds = list(splitter.iter_folds(40))
    assert [f.test_start for f in folds] == [10, 18, 26, 34]
    assert [f.test_rows for f in folds] == [5, 5, 5, 5]
    assert folds[1].test_start > folds[0].test_stop  # bars 15..17 are traded by no fold


def test_split_yields_position_arrays_and_get_n_splits_agrees() -> None:
    frame = pd.DataFrame({"x": np.arange(400)})
    splitter = RollingWalkForward(formation_bars=100, test_bars=50, min_test_bars=20)
    folds = list(splitter.split(frame))
    assert splitter.get_n_splits(frame) == len(folds)
    for formation_idx, test_idx in folds:
        assert formation_idx.dtype.kind == "i" and test_idx.dtype.kind == "i"
        assert formation_idx[-1] + 1 == test_idx[0]
        assert frame.iloc[test_idx].shape[0] == len(test_idx)


def test_get_n_splits_refuses_to_guess_without_the_sample() -> None:
    with pytest.raises(ValueError, match="depends on its length"):
        RollingWalkForward(formation_bars=10, test_bars=5, min_test_bars=5).get_n_splits()


@pytest.mark.parametrize(
    "make_splitter",
    [
        lambda: PurgedWalkForward(n_splits=4, purge_horizon=5, embargo_pct=0.01),
        lambda: RollingWalkForward(formation_bars=100, test_bars=50, min_test_bars=20),
    ],
    ids=["purged", "rolling"],
)
def test_splitters_accept_the_scikit_learn_call_signature(make_splitter) -> None:
    """scikit-learn's model-selection functions call ``split(X, y, groups)`` and
    ``get_n_splits(X, y, groups)``. The extra arguments are ignored: the folds are the ones
    ``split(X)`` yields, so a splitter passed as ``cv=`` evaluates the same geometry."""
    frame = pd.DataFrame({"x": np.arange(400)})
    target = np.zeros(400)
    plain = list(make_splitter().split(frame))
    splitter = make_splitter()
    called = list(splitter.split(frame, target, groups=np.zeros(400)))
    assert len(plain) == len(called) == splitter.get_n_splits(frame, target, groups=None) > 0
    for (train_a, test_a), (train_b, test_b) in zip(plain, called):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


def test_purged_splitter_cannot_express_the_recorded_geometry() -> None:
    """Why walk_forward.py is a Correction and not a no-op at stage 2."""
    sessions = _sessions("demotion_b1_sessions.json")
    frame = pd.DataFrame({"x": np.arange(len(sessions))}, index=sessions)
    purged = list(PurgedWalkForward(n_splits=21, purge_horizon=0, embargo_pct=0.0).split(frame))
    rolling = list(RollingWalkForward(**RECORDED_GEOMETRY).split(frame))
    assert len(purged) == len(rolling) == 21
    purged_test_sizes = {len(test_idx) for _, test_idx in purged}
    assert purged_test_sizes != {63, 48}
    assert any(len(train_idx) != 312 for train_idx, _ in purged)
    assert {len(train_idx) for train_idx, _ in rolling} == {312}


@pytest.mark.parametrize("case,fixture,n_folds", RECORDED_CASES)
def test_recorded_boundaries_are_sessions_in_the_fixture_index(case, fixture, n_folds) -> None:
    """Guards the fixture: every recorded date must be a real bar of that index."""
    sessions = _sessions(fixture)
    positions = _recorded_table(case).positions(sessions)
    assert len(positions) == n_folds
    assert positions[0].formation_start == 0
    # The reserved next-open bar exists; it need not be the sample's last bar
    # (trend_t0 records seven unscored trailing sessions past its last fold).
    assert positions[-1].next_open_position < len(sessions)
    assert [fold.formation_rows for fold in positions] == [312] * n_folds


def test_positions_refuse_a_boundary_that_is_not_a_session() -> None:
    """A boundary must never snap to a neighbour: that is a different fold."""
    sessions = _sessions("demotion_b1_sessions.json")
    rows = [dict(row) for row in _recorded_rows("demotion_b1")]
    rows[3]["test_start"] = "2022-01-01T00:00:00-05:00"  # a Saturday, not a session
    table = FoldTable.from_records(rows, table_id="nudged", timezone="America/New_York")
    with pytest.raises(FoldGeometryError, match="is not a session"):
        table.positions(sessions)


def test_positions_require_a_bar_after_the_last_test_end() -> None:
    sessions = _sessions("demotion_b1_sessions.json")
    table = _recorded_table("demotion_b1")
    truncated = sessions[: int(np.flatnonzero(sessions == table.last_decision)[0]) + 1]
    with pytest.raises(FoldGeometryError, match="fold-end flatten"):
        table.positions(truncated)


def test_fold_table_refuses_overlapping_test_windows() -> None:
    rows = [dict(row) for row in _recorded_rows("demotion_b1")]
    rows[1]["test_start"] = rows[0]["test_end"]
    rows[1]["formation_end"] = "2021-06-25T00:00:00-04:00"
    with pytest.raises(FoldGeometryError, match="overlapping test windows"):
        FoldTable.from_records(rows, table_id="overlapping", timezone="America/New_York")


def test_fold_table_refuses_a_dropped_fold() -> None:
    rows = [dict(row) for row in _recorded_rows("demotion_b1")]
    del rows[5]
    with pytest.raises(FoldGeometryError, match="contiguous"):
        FoldTable.from_records(rows, table_id="gappy", timezone="America/New_York")


def test_fold_table_refuses_a_test_bar_inside_the_formation_window() -> None:
    rows = [dict(row) for row in _recorded_rows("demotion_b1")]
    rows[0]["test_start"] = rows[0]["formation_end"]
    with pytest.raises(FoldGeometryError, match="leakage"):
        FoldTable.from_records(rows, table_id="leaky", timezone="America/New_York")


def test_fold_table_refuses_an_empty_table() -> None:
    with pytest.raises(FoldGeometryError, match="holds no folds"):
        FoldTable(table_id="empty", folds=())


def test_fold_geometry_block_carries_id_hash_and_every_range() -> None:
    table = _recorded_table("demotion_b1")
    block = table.fold_geometry()
    assert block["sha256"] == RECORDED_TABLE_SHA256["demotion_b1"]
    assert block["n_folds"] == 21
    assert block["convention"] == INCLUSIVE_SESSION_DATES
    assert len(block["folds"]) == 21
    assert set(block["folds"][0]) == {"index", "train", "test"}
    assert block["first_decision"] == "2021-03-30T04:00:00+00:00"
    assert block["last_decision"] == "2026-06-12T04:00:00+00:00"
    json.dumps(block)  # the record is serializable as written


def test_resolve_fold_geometry_refuses_a_splitter_beside_a_recorded_table() -> None:
    """Splitter keyword arguments are never silently substituted for a table."""
    table = _recorded_table("demotion_b1")
    with pytest.raises(FoldGeometryError, match="never both"):
        resolve_fold_geometry(fold_table=table, splitter=RollingWalkForward(**RECORDED_GEOMETRY))


def test_resolve_fold_geometry_refuses_an_undeclared_geometry() -> None:
    with pytest.raises(FoldGeometryError, match="no fold geometry declared"):
        resolve_fold_geometry()


def test_resolve_fold_geometry_returns_the_declared_table_and_checks_it() -> None:
    sessions = _sessions("demotion_b1_sessions.json")
    table = _recorded_table("demotion_b1")
    assert resolve_fold_geometry(fold_table=table, index=sessions) is table


def test_resolve_fold_geometry_generates_from_a_splitter_only_when_told_how() -> None:
    sessions = _sessions("demotion_b1_sessions.json")
    splitter = RollingWalkForward(**RECORDED_GEOMETRY)
    with pytest.raises(FoldGeometryError, match="session index"):
        resolve_fold_geometry(splitter=splitter)
    with pytest.raises(FoldGeometryError, match="table_id"):
        resolve_fold_geometry(splitter=splitter, index=sessions)
    table = resolve_fold_geometry(splitter=splitter, index=sessions, table_id="generated")
    assert table.content_hash == RECORDED_TABLE_SHA256["demotion_b1"]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(formation_bars=0, test_bars=5), "formation_bars must be >= 1"),
        (dict(formation_bars=10, test_bars=1), "test_bars must be >= 2"),
        (dict(formation_bars=10, test_bars=5, min_test_bars=6), "min_test_bars must be in"),
        (dict(formation_bars=10, test_bars=5, min_test_bars=0), "min_test_bars must be in"),
        (dict(formation_bars=10, test_bars=5, min_test_bars=5, step_bars=4), "step_bars must be >="),
        (
            dict(formation_bars=10, test_bars=5, min_test_bars=5, close_positions_at_fold_end=False),
            "carry across folds",
        ),
    ],
)
def test_rolling_geometry_is_validated_at_construction(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        RollingWalkForward(**kwargs)


def test_rolling_refuses_a_sample_too_short_for_one_fold() -> None:
    splitter = RollingWalkForward(formation_bars=10, test_bars=5, min_test_bars=3)
    assert splitter.min_obs() == 14
    with pytest.raises(ValueError, match="need at least"):
        list(splitter.iter_folds(13))
    assert len(list(splitter.iter_folds(14))) == 1


def test_session_index_must_be_strictly_increasing() -> None:
    repeated = pd.DatetimeIndex(["2020-01-02", "2020-01-02", "2020-01-03"])
    with pytest.raises(FoldGeometryError, match="strictly increasing"):
        RollingWalkForward(formation_bars=1, test_bars=2, min_test_bars=1).fold_table(repeated, table_id="repeated")


def test_fold_table_reads_a_json_object_wrapping_its_folds(tmp_path) -> None:
    """Both recorded shapes import: a bare list, and ``{"folds": [...]}``."""
    rows = _recorded_rows("demotion_b1")
    wrapped = tmp_path / "folds.json"
    wrapped.write_text(json.dumps({"folds": rows, "note": "ignored"}), encoding="utf-8")
    table = FoldTable.from_json(wrapped, table_id="wrapped", timezone="America/New_York")
    assert table.content_hash == RECORDED_TABLE_SHA256["demotion_b1"]
    assert table.source == str(wrapped)


def test_fold_table_refuses_a_json_object_without_folds(tmp_path) -> None:
    path = tmp_path / "folds.json"
    path.write_text(json.dumps({"note": "no folds here"}), encoding="utf-8")
    with pytest.raises(FoldGeometryError, match="no 'folds' key"):
        FoldTable.from_json(path, table_id="empty")


def test_positions_resolve_across_equivalent_timezone_representations() -> None:
    """A boundary is an instant, not a string: UTC sessions must still resolve."""
    sessions = _sessions("demotion_b1_sessions.json").tz_convert("UTC")
    positions = _recorded_table("demotion_b1").positions(sessions)
    assert len(positions) == 21
    assert positions[0].formation_start == 0


def test_fold_table_refuses_a_row_missing_a_boundary() -> None:
    rows = [dict(row) for row in _recorded_rows("demotion_b1")]
    del rows[2]["test_end"]
    with pytest.raises(FoldGeometryError, match=r"missing \['test_end'\]"):
        FoldTable.from_records(rows, table_id="short", timezone="America/New_York")


def test_fold_table_refuses_a_row_without_a_fold_number() -> None:
    rows = [dict(row) for row in _recorded_rows("demotion_b1")]
    rows[0].pop("fold")
    with pytest.raises(FoldGeometryError, match="neither 'index' nor 'fold'"):
        FoldTable.from_records(rows, table_id="anonymous", timezone="America/New_York")
