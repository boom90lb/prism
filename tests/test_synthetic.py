"""Deterministic synthetic fixtures in the §3.2 schema (engine spec §3.2, stage 1).

Two properties carry the fixture. It must be *reproducible*: the same seed and
scenario set rewrite the same bytes, so a recorded ``data_id`` still names the
data it named. And it must be *labelled*: every row says it is synthetic, and
the manifest says which scenarios were written without being applied, so no
reader can mistake a generated result for a market one.

The scenario assertions attack the generator where a fixture usually rots —
scenarios that quietly write nothing. Each stage 1 scenario is checked for the
row it claims to write, and the later-stage scenarios are checked to be both
present as data and absent from every index.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from prism.io.manifest import Manifest
from prism.io.synthetic import (
    ALL_SCENARIOS,
    EQUITY_INDEX,
    ETF_INDEX,
    GAP_FLAG_SESSIONS,
    SCENARIOS,
    STAGE1_SCENARIOS,
    SyntheticError,
    resolve_scenarios,
    sid_of,
    write_synthetic_dataset,
)

START = dt.date(2026, 1, 1)
END = dt.date(2026, 12, 31)
SEED = 7

# The reference spread schedule of section 4, descending, inclusive lower bound.
SPREAD_SCHEDULE = ((5.0e8, 1.0), (1.0e8, 2.0), (2.5e7, 5.0), (0.0, 10.0))


def _write(root: Path, **overrides) -> Manifest:
    kwargs = dict(seed=SEED, scenarios="all", n_equities=14, n_etfs=3, start=START, end=END)
    kwargs.update(overrides)
    return write_synthetic_dataset(root, **kwargs)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Manifest]:
    root = tmp_path_factory.mktemp("synthetic") / "data"
    return root, _write(root)


def _table(root: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(root / f"{name}.parquet")


def _bars(root: Path, sid: str) -> pd.DataFrame:
    return pd.read_parquet(root / "bars" / f"{sid}_1d.parquet")


# ---- the scenario registry ------------------------------------------------


def test_the_twelve_scenario_ids_and_their_stages_match_the_specification() -> None:
    assert ALL_SCENARIOS == (
        "membership",
        "split",
        "distribution",
        "halt",
        "costs",
        "rounding",
        "spinoff",
        "ticker_reuse",
        "revision",
        "rebase",
        "delisting",
        "recovery",
    )
    assert {name: SCENARIOS[name].stage for name in ALL_SCENARIOS} == {
        "membership": 1,
        "split": 1,
        "distribution": 1,
        "halt": 1,
        "costs": 1,
        "rounding": 1,
        "spinoff": 3,
        "ticker_reuse": 3,
        "revision": 3,
        "rebase": 3,
        "delisting": 3,
        "recovery": 4,
    }
    assert set(STAGE1_SCENARIOS) == {name for name, s in SCENARIOS.items() if s.stage == 1}


def test_resolve_scenarios_accepts_all_and_a_subset_and_refuses_the_rest() -> None:
    assert resolve_scenarios("all") == ALL_SCENARIOS
    assert resolve_scenarios("costs,membership") == ("costs", "membership")
    assert resolve_scenarios(["costs"]) == ("costs",)
    with pytest.raises(SyntheticError, match="unknown scenarios"):
        resolve_scenarios("costs,teleport")
    with pytest.raises(SyntheticError, match="cannot be combined"):
        resolve_scenarios("all,costs")
    with pytest.raises(SyntheticError, match="at least one scenario"):
        resolve_scenarios("")


# ---- determinism ----------------------------------------------------------


def test_the_same_seed_rewrites_the_same_bytes(tmp_path: Path) -> None:
    first = _write(tmp_path / "a")
    second = _write(tmp_path / "b")
    assert first.id == second.id
    assert first.files == second.files
    for relative, _ in first.files:
        assert (tmp_path / "a" / relative).read_bytes() == (tmp_path / "b" / relative).read_bytes()


def test_the_manifest_records_the_earliest_session_table_release_covering_its_sessions(tmp_path: Path) -> None:
    """A fixture that ends in 2027 records the release it had before 2028 was added, so its id did not move."""
    before = _write(tmp_path / "2027", start=dt.date(2027, 1, 4), end=dt.date(2027, 12, 31))
    after = _write(tmp_path / "2028", start=dt.date(2028, 1, 3), end=dt.date(2028, 12, 29))
    assert before.assumptions["session_table_version"] == "2006.2027-1"
    assert after.assumptions["session_table_version"] == "2006.2028-1"


def test_a_different_seed_gives_different_prices_and_a_different_id(tmp_path: Path) -> None:
    first = _write(tmp_path / "a")
    other = _write(tmp_path / "b", seed=SEED + 1)
    assert other.id != first.id
    left = _bars(tmp_path / "a", sid_of(12))["close"].to_numpy()
    right = _bars(tmp_path / "b", sid_of(12))["close"].to_numpy()
    assert left.shape == right.shape
    assert not (left == right).any()


def test_turning_a_scenario_off_moves_no_other_name(tmp_path: Path) -> None:
    """Per-name streams: dropping the halt scenario must not reprice the rest of the universe."""
    everything = _write(tmp_path / "a")
    without_halt = _write(tmp_path / "b", scenarios=[s for s in ALL_SCENARIOS if s != "halt"])
    assert without_halt.id != everything.id
    untouched = sid_of(13)
    pd.testing.assert_frame_equal(_bars(tmp_path / "a", untouched), _bars(tmp_path / "b", untouched))


def test_a_scenario_subset_writes_no_later_stage_sid(tmp_path: Path) -> None:
    root = tmp_path / "subset"
    manifest = _write(root, scenarios="membership,costs")
    assert manifest.coverage["sids_outside_every_index"] == []
    assert manifest.assumptions["scenarios_written"] == ["costs", "membership"]
    assert manifest.assumptions["scenarios_written_not_applied"] == []
    assert not (root / "adjustments.parquet").exists()


# ---- labelling ------------------------------------------------------------


def test_every_row_of_every_table_is_labelled_synthetic(fixture: tuple[Path, Manifest]) -> None:
    root, manifest = fixture
    assert manifest.provenance == "synthetic"
    checked = 0
    for relative, _ in manifest.files:
        frame = pd.read_parquet(root / relative)
        assert not frame.empty, relative
        assert frame["synthetic"].all(), relative
        checked += 1
    assert checked == len(manifest.files)


def test_the_manifest_names_the_scenarios_it_wrote_without_applying_them(fixture: tuple[Path, Manifest]) -> None:
    _, manifest = fixture
    assumptions = manifest.assumptions
    assert assumptions["scenarios_written"] == sorted(ALL_SCENARIOS)
    assert assumptions["scenarios_written_not_applied"] == sorted(name for name, s in SCENARIOS.items() if s.stage > 1)
    assert assumptions["scenario_applied_at_stage"]["spinoff"] == 3
    assert assumptions["scenario_applied_at_stage"]["recovery"] == 4
    # The smoke summary prints both notes, so they say what applies each scenario rather than
    # naming an overhaul stage.
    assert "canonical simulator" in assumptions["scenarios_written_not_applied_note"]
    assert "canonical simulator" in assumptions["stage1_unapplied_in_universe"]
    for key in ("scenarios_written_not_applied_note", "stage1_unapplied_in_universe", "scenario_cases"):
        assert not re.search(r"\bstage \d|§\s?3\.|\[Q\d", assumptions[key]), key
    # The other half of section 4's "written but not exercised" table: these
    # two sids are inside the equity index, so an evaluation selects them.
    assert assumptions["scenarios_unapplied_in_universe"] == ["halt", "split"]
    assert set(assumptions["scenarios_unapplied_in_universe"]).isdisjoint(assumptions["scenarios_written_not_applied"])


def test_the_later_stage_scenario_sids_belong_to_no_index(fixture: tuple[Path, Manifest]) -> None:
    """This is what makes "written but not applied" mechanical rather than aspirational."""
    root, manifest = fixture
    outside = set(manifest.coverage["sids_outside_every_index"])
    assert len(outside) == 9  # spinoff 2, ticker_reuse 2, revision 1, rebase 1, delisting 2, recovery 1
    membership = _table(root, "membership")
    assert outside.isdisjoint(set(membership["sid"]))
    securities = _table(root, "securities")
    assert outside <= set(securities["sid"]), "an unindexed scenario sid is still a security"


def test_the_manifest_id_is_recomputable_from_the_files_on_disk(fixture: tuple[Path, Manifest]) -> None:
    root, manifest = fixture
    manifest.verify_files(root)
    assert Manifest.read(root / "manifest.json").id == manifest.id


# ---- the stage 1 scenarios -------------------------------------------------


def test_membership_writes_one_add_and_one_remove_inside_the_sample(fixture: tuple[Path, Manifest]) -> None:
    root, _ = fixture
    membership = _table(root, "membership")
    equities = membership.loc[membership["index_id"] == EQUITY_INDEX]
    first_session = equities["valid_from"].min()
    added = equities.loc[equities["valid_from"] > first_session]
    removed = equities.loc[equities["valid_to"].notna()]
    assert len(added) == 1 and len(removed) == 1
    assert added.iloc[0]["announce_date"] < added.iloc[0]["valid_from"]
    events = _table(root, "events")
    kinds = set(events["event_type"])
    assert {"membership_add", "membership_remove"} <= kinds
    remove_event = events.loc[events["event_type"] == "membership_remove"].iloc[0]
    assert remove_event["effective_date"] > removed.iloc[0]["valid_to"]
    assert remove_event["announce_date"] < remove_event["effective_date"]
    assert set(membership.loc[membership["index_id"] == ETF_INDEX, "sid"]) == {sid_of(15), sid_of(16), sid_of(17)}


def test_split_halves_the_raw_print_and_leaves_the_adjusted_series_continuous(fixture: tuple[Path, Manifest]) -> None:
    root, _ = fixture
    events = _table(root, "events")
    splits = events.loc[events["event_type"] == "split"]
    assert len(splits) == 2
    effective = dt.date.fromisoformat(splits.iloc[0]["effective_date"])
    assert {json.loads(p)["factor"] for p in splits["payload"]} == {2.0}
    adjustments = _table(root, "adjustments")
    assert set(adjustments.loc[adjustments["kind"] == "split", "sid"]) == set(splits["sid"])

    raw = _bars(root, sid_of(1))
    assert set(raw["price_basis"]) == {"raw"}
    dates = raw["ts"].dt.date
    before = float(raw.loc[dates < effective, "close"].iloc[-1])
    after = float(raw.loc[dates >= effective, "close"].iloc[0])
    assert 0.35 < after / before < 0.65, "the raw print did not halve across the split"

    adjusted = _bars(root, sid_of(2))
    assert set(adjusted["price_basis"]) == {"split_adjusted_at_ingest"}
    dates = adjusted["ts"].dt.date
    before = float(adjusted.loc[dates < effective, "close"].iloc[-1])
    after = float(adjusted.loc[dates >= effective, "close"].iloc[0])
    assert 0.8 < after / before < 1.25, "the already-adjusted series must not step across the split"


def test_distribution_carries_an_ex_date_and_a_later_pay_date(fixture: tuple[Path, Manifest]) -> None:
    root, _ = fixture
    events = _table(root, "events")
    row = events.loc[events["event_type"] == "cash_distribution"].iloc[0]
    payload = json.loads(row["payload"])
    assert payload["currency"] == "USD"
    assert payload["distribution_kind"] == "dividend"
    assert payload["amount_per_share"] > 0
    assert payload["pay_date"] > row["effective_date"]
    assert len(payload["provenance_sha256"]) == 64


def test_halt_removes_a_multi_session_run_that_stays_below_the_gap_flag(fixture: tuple[Path, Manifest]) -> None:
    root, _ = fixture
    events = _table(root, "events")
    row = events.loc[events["event_type"] == "halt"].iloc[0]
    payload = json.loads(row["payload"])
    frame = _bars(root, row["sid"])
    present = set(frame["ts"].dt.date)
    from prism.io import sessions

    halted = [
        day
        for day in sessions.sessions_between(
            dt.date.fromisoformat(payload["start"]), dt.date.fromisoformat(payload["end"])
        )
        if day not in present
    ]
    assert len(halted) == 3
    assert len(halted) < GAP_FLAG_SESSIONS


def test_costs_places_names_across_every_step_of_the_reference_spread_schedule(fixture: tuple[Path, Manifest]) -> None:
    root, _ = fixture

    def bucket(sid: str) -> float:
        frame = _bars(root, sid)
        median = float((frame["close"] * frame["volume"]).tail(20).median())
        for floor, bps in SPREAD_SCHEDULE:
            if median >= floor:
                return bps
        return SPREAD_SCHEDULE[-1][1]

    assert [bucket(sid_of(n)) for n in (5, 6, 7, 8)] == [1.0, 2.0, 5.0, 10.0]


def test_rounding_writes_a_dated_per_sid_quantity_increment(fixture: tuple[Path, Manifest]) -> None:
    root, _ = fixture
    capabilities = _table(root, "capabilities")
    rows = capabilities.loc[capabilities["sid"] == sid_of(9)].sort_values("valid_at")
    assert len(rows) == 2
    assert list(rows["quantity_increment"]) == [1.0, 5.0]
    assert rows.iloc[0]["valid_at"] < rows.iloc[1]["valid_at"]
    assert not capabilities["fractionable"].any(), "the synthetic profile is whole-share"


# ---- the later-stage scenarios, written as data ----------------------------


def test_spinoff_names_a_tracked_child_with_cash_in_lieu(fixture: tuple[Path, Manifest]) -> None:
    root, _ = fixture
    events = _table(root, "events")
    payload = json.loads(events.loc[events["event_type"] == "spinoff"].iloc[0]["payload"])
    assert payload["cash_in_lieu"] is True
    assert payload["child_sid"] in set(_table(root, "securities")["sid"])


def test_ticker_reuse_yields_two_sids_with_non_overlapping_listings(fixture: tuple[Path, Manifest]) -> None:
    root, _ = fixture
    listings = _table(root, "listings")
    reuse = listings.loc[listings["ticker"] == "RUSE"].sort_values("valid_from")
    assert len(reuse) == 2
    assert reuse.iloc[0]["sid"] != reuse.iloc[1]["sid"]
    assert reuse.iloc[0]["valid_to"] < reuse.iloc[1]["valid_from"]


def test_revision_writes_a_late_correction_and_a_null_full_revision(fixture: tuple[Path, Manifest]) -> None:
    root, manifest = fixture
    outside = manifest.coverage["sids_outside_every_index"]
    revised = [sid for sid in outside if len(_bars(root, sid)["revision"].unique()) == 3]
    assert len(revised) == 1
    frame = _bars(root, revised[0])
    duplicated = frame.loc[frame["ts"].duplicated(keep=False)].sort_values("revision")
    assert list(duplicated["revision"]) == [1, 2, 3]
    assert duplicated.iloc[1]["close"] == pytest.approx(duplicated.iloc[0]["close"] * 1.10, rel=1e-3)
    assert pd.isna(duplicated.iloc[2]["close"]), "the withdrawal is an explicit null, not a missing row"
    assert duplicated.iloc[1]["available_at"] < duplicated.iloc[2]["available_at"]


def test_rebase_writes_a_full_series_revision_with_a_matching_adjustments_row(fixture: tuple[Path, Manifest]) -> None:
    root, manifest = fixture
    adjustments = _table(root, "adjustments")
    rebase = adjustments.loc[adjustments["kind"] == "rebase"]
    assert len(rebase) == 1
    assert float(rebase.iloc[0]["factor"]) == 4.0
    frame = _bars(root, rebase.iloc[0]["sid"])
    first = frame.loc[frame["revision"] == 1]
    second = frame.loc[frame["revision"] == 2]
    assert len(first) == len(second), "a rebase revises the whole series, not a tail"
    ratio = first["close"].to_numpy() / second["close"].to_numpy()
    assert ratio.min() > 3.99 and ratio.max() < 4.01


def test_delisting_writes_one_known_terminal_payout_and_one_unknown(fixture: tuple[Path, Manifest]) -> None:
    root, _ = fixture
    events = _table(root, "events")
    payouts = [json.loads(p) for p in events.loc[events["event_type"] == "terminal_payout", "payload"]]
    assert len(payouts) == 2
    assert sum("status" in p and p["status"] == "unknown" for p in payouts) == 1
    assert sum("amount_per_share" in p for p in payouts) == 1


def test_recovery_writes_a_gap_the_coverage_sweep_flags(fixture: tuple[Path, Manifest]) -> None:
    root, manifest = fixture
    flags = manifest.coverage["gap_flags"]
    assert len(flags) == 1
    flag = flags[0]
    assert flag["missing_sessions"] > GAP_FLAG_SESSIONS
    assert flag["flagged"] is True
    capabilities = _table(root, "capabilities")
    restricted = capabilities.loc[capabilities["sid"] == flag["sid"]].sort_values("valid_at").iloc[-1]
    assert json.loads(restricted["restrictions"]) == ["no_new_exposure"]


# ---- refusals -------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"n_equities": 5}, "n_equities must be >="),
        ({"n_etfs": 1}, "n_etfs must be >="),
        ({"start": dt.date(2026, 1, 1), "end": dt.date(2026, 2, 20)}, "at least 60 sessions"),
        (
            {"scenarios": "recovery", "start": dt.date(2026, 1, 1), "end": dt.date(2026, 4, 15)},
            "needs at least 80 sessions",
        ),
        ({"scenarios": "teleport"}, "unknown scenarios"),
    ],
)
def test_the_generator_refuses_a_specification_it_cannot_honour(tmp_path: Path, overrides: dict, match: str) -> None:
    with pytest.raises(SyntheticError, match=match):
        _write(tmp_path / "refused", **overrides)
