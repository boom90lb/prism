"""``upgrade-diff``: what an upgrade would decide differently (§3.11, stage 5).

The mechanism the brief asks for is a way to review an upgrade *before*
adopting it, without keeping a fork of the engine alive beside the live one.
This file pins what that report says and — as importantly — what it refuses to
say. Two properties do the work:

* it compares **decisions**, the constructed book and the orders, never fill
  prices or realised equity, which belong to the venue and the day rather than
  to the engine version;
* a session that could not be run under one of the two versions is reported as
  *not compared*, with the reason, and never folded into "no differences
  found". ``identical`` over nothing is not agreement, which is why
  ``complete`` is a separate property and both are asserted together here.

The stage-0 recorded fixture (``tests/fixtures/paper_replay``) is the case that
makes the second property concrete: its manifest records that **no local bar
source covers the five recorded sessions**, so the recorded side is readable in
full and the replayed side cannot be produced at all. The report over it names
five uncompared sessions and carries the manifest's own caveats.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.live.daily import DailyBookConfig
from prism.live.replay import (
    EngineDecisions,
    SessionDecision,
    recorded_decisions,
    replayed_decisions,
    upgrade_diff,
    upgrade_diff_report,
)
from tests.test_live_daily import ConstSignal

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "paper_replay"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"

DECISION_BARS = ["2026-07-13", "2026-08-11"]
HOLD_BARS = ["2026-08-12", "2026-08-13", "2026-08-14"]

fixture_only = pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason=f"no recorded session capture at {MANIFEST_PATH}",
)


def _panels(n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2026-05-01", periods=n, freq="B", tz="America/New_York")
    close = pd.DataFrame({"AAA": [100.0] * n, "BBB": [50.0] * n}, index=idx)
    volume = pd.DataFrame(1e6, index=idx, columns=close.columns)
    return close, volume, close * 0.999


def _replay_side(tmp_path: Path, name: str, config: DailyBookConfig, bars, close, volume, open_):
    return replayed_decisions(
        lambda: ConstSignal({"AAA": 1.0, "BBB": -1.0}),
        close,
        volume,
        open_,
        config,
        tmp_path / name,
        bars=bars,
        engine_version=name,
        label=name,
    )


# ---------------------------------------------------------------------------
# two engine versions over the same sessions
# ---------------------------------------------------------------------------


def test_two_versions_that_size_differently_are_reported_name_by_name(tmp_path) -> None:
    close, volume, open_ = _panels()
    bars = [str(stamp.date()) for stamp in close.index[24:27]]
    # 0.05 against 0.10, both inside the 0.10 per-name cap, so the difference
    # reported is the sizing change itself and not the cap binding on one side.
    baseline = _replay_side(
        tmp_path, "v-old", DailyBookConfig(position_size=0.05, min_order_notional=1.0), bars, close, volume, open_
    )
    candidate = _replay_side(
        tmp_path, "v-new", DailyBookConfig(position_size=0.10, min_order_notional=1.0), bars, close, volume, open_
    )
    report = upgrade_diff(baseline, candidate)

    assert report.complete and not report.identical
    assert [s.bar for s in report.comparable] == bars
    first = report.sessions[0]
    assert first.comparable and first.differs
    moved = {change.sid: change for change in first.moved}
    assert moved["AAA"].baseline == pytest.approx(0.05) and moved["AAA"].candidate == pytest.approx(0.10)
    assert moved["BBB"].baseline == pytest.approx(-0.05) and moved["BBB"].candidate == pytest.approx(-0.10)
    assert first.max_weight_difference == pytest.approx(0.05)
    assert first.gross_weight_difference == pytest.approx(0.10)
    # The orders differ too, and both sides recorded an order set, so they were
    # actually compared rather than skipped.
    assert first.orders_compared and {c.sid for c in first.order_changes} == {"AAA", "BBB"}
    assert "0 with a different decision" not in report.summary()


def test_the_same_engine_twice_reports_no_difference_over_every_session(tmp_path) -> None:
    close, volume, open_ = _panels()
    bars = [str(stamp.date()) for stamp in close.index[24:28]]
    config = DailyBookConfig(position_size=0.1, min_order_notional=1.0)
    report = upgrade_diff(
        _replay_side(tmp_path, "left", config, bars, close, volume, open_),
        _replay_side(tmp_path, "right", config, bars, close, volume, open_),
    )
    assert report.complete and report.identical and report.differing == ()


def test_a_cadence_change_shows_up_as_a_session_one_version_re_decided(tmp_path) -> None:
    close, volume, open_ = _panels()
    bars = [str(stamp.date()) for stamp in close.index[24:29]]
    daily = _replay_side(
        tmp_path,
        "daily",
        DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=1),
        bars,
        close,
        volume,
        open_,
    )
    monthly = _replay_side(
        tmp_path,
        "monthly",
        DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=3),
        bars,
        close,
        volume,
        open_,
    )
    report = upgrade_diff(daily, monthly)
    assert report.complete
    changed = [session.bar for session in report.sessions if session.refresh_changed]
    assert changed, "a cadence difference must be visible as a session one version re-decided"
    assert all(session.differs for session in report.sessions if session.refresh_changed)


def test_a_session_outside_the_panel_is_named_as_not_compared_never_as_no_difference(tmp_path) -> None:
    close, volume, open_ = _panels()
    inside = [str(stamp.date()) for stamp in close.index[24:26]]
    outside = ["2027-01-04"]
    baseline = _replay_side(
        tmp_path, "v-old", DailyBookConfig(position_size=0.1, min_order_notional=1.0), inside, close, volume, open_
    )
    candidate = replayed_decisions(
        lambda: ConstSignal({"AAA": 1.0, "BBB": -1.0}),
        close,
        volume,
        open_,
        DailyBookConfig(position_size=0.1, min_order_notional=1.0),
        tmp_path / "v-new",
        bars=inside + outside,
        engine_version="v-new",
        label="v-new",
    )
    assert candidate.unavailable == {"2027-01-04": "no bar panel covers this session"}
    report = upgrade_diff(baseline, candidate, bars=inside + outside)
    assert not report.complete and report.identical
    (uncompared,) = report.incomparable
    assert uncompared.bar == "2027-01-04"
    assert "no decision recorded" in uncompared.reason or "no bar panel" in uncompared.reason
    assert "does not say the upgrade changes nothing" in report.summary()


def test_a_side_that_records_no_orders_is_not_read_as_every_order_changing() -> None:
    left = EngineDecisions(
        engine_version="a",
        label="a",
        decisions={"2026-07-13": SessionDecision("2026-07-13", True, {"AAA": 0.1}, orders=None)},
    )
    right = EngineDecisions(
        engine_version="b",
        label="b",
        decisions={"2026-07-13": SessionDecision("2026-07-13", True, {"AAA": 0.1}, orders={"AAA": 5.0})},
    )
    (session,) = upgrade_diff(left, right).sessions
    assert session.comparable and not session.orders_compared and session.order_changes == ()
    assert not session.differs


def test_a_declared_unavailable_side_names_the_reason_on_every_session() -> None:
    side = EngineDecisions.unavailable_for("v-new", DECISION_BARS, "the panel is gone", label="candidate")
    assert side.decisions == {}
    assert set(side.unavailable) == set(DECISION_BARS)
    report = upgrade_diff(
        EngineDecisions(
            engine_version="v-old",
            label="baseline",
            decisions={bar: SessionDecision(bar, True, {"AAA": 0.1}) for bar in DECISION_BARS},
        ),
        side,
    )
    assert not report.complete and len(report.incomparable) == len(DECISION_BARS)
    assert all("the panel is gone" in session.reason for session in report.incomparable)


# ---------------------------------------------------------------------------
# the stage-0 recorded fixture: what it can and cannot support
# ---------------------------------------------------------------------------


@fixture_only
def test_the_recorded_side_reads_the_five_sessions_with_their_books_and_orders() -> None:
    recorded = recorded_decisions(FIXTURE_DIR)
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert recorded.engine_version == "git:" + manifest["source"]["traded_commit"]
    assert sorted(recorded.decisions) == sorted(DECISION_BARS + HOLD_BARS)
    assert recorded.unavailable == {}

    counts = {bar: len(recorded.decisions[bar].targets) for bar in DECISION_BARS}
    assert counts == {
        session["bar"]: session["n_targets"] for session in manifest["sessions"] if session["kind"] == "decision"
    }
    # A hold is a fact, not a gap: the engine held, and the report says so.
    for bar in HOLD_BARS:
        held = recorded.decisions[bar]
        assert not held.refreshed and held.targets == {}
    # The submitted order set is reconstructable from the ledgers, sweep
    # children excluded (a completion order is a residual, not a decision).
    assert len(recorded.decisions["2026-07-13"].orders) == 98
    assert len(recorded.decisions["2026-08-11"].orders) == 111


@fixture_only
def test_the_fixture_has_no_bar_panel_so_the_replayed_side_cannot_be_produced() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["panels"]["status"] == "unavailable"
    assert all(row["symbols_with_a_bar"] == 0 for row in manifest["panels"]["per_session_coverage"])


@fixture_only
def test_the_report_over_the_fixture_names_five_uncompared_sessions_and_claims_nothing() -> None:
    report = upgrade_diff_report(FIXTURE_DIR)
    assert len(report.sessions) == 5
    assert report.comparable == ()
    assert len(report.incomparable) == 5
    # identical over nothing is not agreement, and the report says both.
    assert report.identical and not report.complete
    summary = report.summary()
    assert "0/5 session(s) compared" in summary
    assert "does not say the upgrade changes nothing" in summary
    # The capture's own caveats travel onto the report rather than being dropped
    # by the tool that read them.
    assert any("NO PANELS" in note for note in report.notes)
    json.dumps(report.to_dict())


@fixture_only
def test_the_fixture_compared_against_itself_is_a_full_agreement(tmp_path) -> None:
    """The one comparison the fixture *can* support: it reproduces its own record."""
    report = upgrade_diff_report(FIXTURE_DIR, baseline=FIXTURE_DIR, candidate=FIXTURE_DIR)
    assert report.complete and report.identical
    assert len(report.comparable) == 5
    assert all(session.orders_compared for session in report.comparable if session.baseline_refreshed)


@fixture_only
def test_a_candidate_that_changed_one_name_is_reported_against_the_record(tmp_path) -> None:
    """A synthetic candidate over the recorded sessions, to prove the diff bites."""
    recorded = recorded_decisions(FIXTURE_DIR, label="baseline")
    bar = DECISION_BARS[1]
    book = dict(recorded.decisions[bar].targets)
    victim = sorted(book)[0]
    book[victim] = book[victim] + 0.01
    candidate = EngineDecisions(
        engine_version="v-next",
        label="candidate",
        decisions={
            **recorded.decisions,
            bar: SessionDecision(bar, True, book, orders=recorded.decisions[bar].orders),
        },
    )
    report = upgrade_diff(recorded, candidate)
    assert report.complete and not report.identical
    (differing,) = report.differing
    assert differing.bar == bar
    assert [change.sid for change in differing.moved] == [victim]
    assert differing.max_weight_difference == pytest.approx(0.01)
