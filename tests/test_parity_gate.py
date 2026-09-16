"""The stage 5 replay parity gate (``prism.live.parity_gate``).

Two things are under test here and they pull in opposite directions. The gate
must report what it actually established, which is the checks that pass on the
stage-0 capture; and it must refuse to report what it did not establish, which
is the end-to-end decision comparison nobody can run without bar panels. A gate
that turned "we could not run it" into a pass would be worse than no gate, so
the uncomparable-is-not-a-pass case is tested directly, and so is the case where
the gate has to bite.

Skipped, not failed, when the capture is absent: it is built from a live run
directory only the operator box holds.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from prism.live.parity_gate import (
    FAIL,
    NOT_COMPARED,
    PASS,
    SHARE_TOLERANCE,
    WEIGHT_TOLERANCE,
    positions_before,
    probe_bar_coverage,
    run_gate,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "paper_replay"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason=(
        f"no baseline replay capture at {MANIFEST_PATH} — regenerate it with "
        "tests/fixtures/paper_replay/capture.py (its manifest records the command)"
    ),
)

DECISION_BARS = ("2026-07-13", "2026-08-11")
HOLD_BARS = ("2026-08-12", "2026-08-13", "2026-08-14")


@pytest.fixture(scope="module")
def report():
    return run_gate(FIXTURE_DIR)


def _check(report, name):
    found = [check for check in report.checks if check.name == name]
    assert found, f"no {name} check in the report"
    return found[0]


# ------------------------------------------------- what the gate cannot prove


def test_the_decision_comparison_is_reported_as_not_compared_and_never_as_a_pass(report) -> None:
    """The whole point. No bar panels, so no candidate side, so no comparison."""
    decision = _check(report, "decision-parity")
    assert decision.verdict == NOT_COMPARED
    assert decision.evidence["sessions_compared"] == 0
    assert decision.evidence["sessions_requested"] == 5
    assert "0 of 5 session(s) can be compared end to end" in decision.detail
    # And the run as a whole does not pass on the strength of the checks that did.
    assert report.verdict == "INCOMPLETE"
    assert report.exit_code == 2
    assert "does not say the new path decides the same thing" in report.summary()


def test_every_session_carries_the_uncompared_verdict_rather_than_being_dropped(report) -> None:
    bars = [row.bar for row in report.sessions]
    assert bars == [*DECISION_BARS, *HOLD_BARS]
    assert {row.decision_parity for row in report.sessions} == {NOT_COMPARED}


def test_the_report_names_what_would_close_the_decision_comparison(report) -> None:
    """A blocked gate that does not say how to unblock it is a dead end."""
    closing = report.to_dict()["what_would_close_decision_parity"]
    assert "503" in closing["symbols"]  # the scored universe, not the trimmed 119
    assert "2025-06-09" in closing["dates"] and "2026-08-14" in closing["dates"]
    assert "Alpaca" in closing["bar_source"]
    assert "split-adjustment" in closing["adjustment_basis"]


def test_the_captures_own_caveats_travel_onto_the_report(report) -> None:
    """A limitation recorded in a capture and dropped by the tool reading it is invisible."""
    assert any("NO PANELS" in note for note in report.notes)
    assert any("not contiguous" in note for note in report.notes)


# --------------------------------------------------- what the gate does prove


def test_the_capture_still_hashes_to_what_it_recorded(report) -> None:
    integrity = _check(report, "capture-integrity")
    assert integrity.verdict == PASS
    assert integrity.evidence["files_checked"] == 8


def test_the_reader_recovers_every_session_with_the_counts_the_capture_recorded(report) -> None:
    extraction = _check(report, "recorded-extraction")
    assert extraction.verdict == PASS
    manifest = json.loads(MANIFEST_PATH.read_text())
    expected = {str(s["bar"]): int(s["n_targets"]) for s in manifest["sessions"]}
    assert extraction.evidence["target_counts"] == expected
    # A hold is a fact the reader keeps, not a gap it invents.
    assert all(expected[bar] == 0 for bar in HOLD_BARS)


def test_the_recovered_books_compare_identical_to_themselves_at_the_declared_tolerance(report) -> None:
    parity = _check(report, "extraction-parity")
    assert parity.verdict == PASS
    assert parity.evidence["sessions_compared"] == 5
    assert parity.evidence["sessions_differing"] == 0
    assert parity.evidence["order_sets_compared"] == list(DECISION_BARS)


def test_every_recorded_order_is_rebuilt_share_for_share_from_the_recorded_inputs(report) -> None:
    """The substantive parity: the new sizing path reproduces the submitted orders.

    209 parent orders over the two decision sessions, rebuilt from the recorded
    book, equity and reference prices against a position book carried forward
    from the recorded fills, at a zero share tolerance.
    """
    sizing = _check(report, "order-sizing-parity")
    assert sizing.verdict == PASS
    assert sizing.evidence["orders_compared"] == 209
    assert sizing.evidence["orders_differing"] == 0
    per_session = sizing.evidence["per_session"]
    assert per_session["2026-07-13"]["orders_recorded"] == 98
    assert per_session["2026-08-11"]["orders_recorded"] == 111
    # Session one opens flat, and matching 98/98 orders from an empty book is
    # what establishes that rather than an assumption anyone had to make.
    assert per_session["2026-07-13"]["opening_positions"] == 0
    assert per_session["2026-07-13"]["names_priced_from_the_order_ledger"] == []
    # Session two carries session one's fills, and its twenty exits take their
    # decision price from the order ledger because the targets ledger prices the
    # book and not the exits. That weaker input is named, not hidden.
    assert per_session["2026-08-11"]["opening_positions"] == 98
    assert len(per_session["2026-08-11"]["names_priced_from_the_order_ledger"]) == 20


def test_the_artifact_carries_the_tolerance_it_was_judged_against(report) -> None:
    """The tolerance is in the artifact, not in a document beside it."""
    declared = report.to_dict()["declared_tolerance"]
    assert declared["weight_absolute"] == WEIGHT_TOLERANCE
    assert declared["share_absolute"] == SHARE_TOLERANCE
    # Restated in the capture's own units, so the exponent is readable.
    assert declared["weight_tolerance_in_dollars_at_smallest_recorded_equity"] < 0.001
    assert declared["weight_tolerance_in_shares_of_the_cheapest_name"] < 1e-3
    assert "whole shares" in declared["share_rationale"]
    json.dumps(report.to_dict())  # the artifact is serializable as written


# ------------------------------------------------------------- the gate bites


def test_the_gate_reports_a_perturbation_it_must_report(report) -> None:
    """The refutation check: a comparator that never differs passes everything."""
    refutation = _check(report, "refutation")
    assert refutation.verdict == PASS
    assert refutation.evidence["perturbed_session"] == "2026-08-11"
    assert refutation.evidence["sessions_reported_differing"] == ["2026-08-11"]


def test_a_corrupted_ledger_fails_the_gate(tmp_path) -> None:
    capture = tmp_path / "capture"
    shutil.copytree(FIXTURE_DIR, capture)
    targets = capture / "targets.jsonl"
    rows = [json.loads(line) for line in targets.read_text().splitlines() if line.strip()]
    victim = sorted(rows[0]["targets"])[0]
    rows[0]["targets"][victim] = rows[0]["targets"][victim] * 2
    targets.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    report = run_gate(capture)
    assert _check(report, "capture-integrity").verdict == FAIL
    assert report.verdict == "FAILED"
    assert report.exit_code == 1


def test_a_wrong_order_quantity_fails_the_sizing_check(tmp_path) -> None:
    """The check that would catch a sizing regression, shown catching one."""
    capture = tmp_path / "capture"
    shutil.copytree(FIXTURE_DIR, capture)
    unfilled = capture / "unfilled.jsonl"
    rows = [json.loads(line) for line in unfilled.read_text().splitlines() if line.strip()]
    rows[0]["order_qty"] = float(rows[0]["order_qty"]) + 1.0
    unfilled.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    report = run_gate(capture)
    sizing = _check(report, "order-sizing-parity")
    assert sizing.verdict == FAIL
    assert sizing.evidence["per_session"]["2026-07-13"]["orders_differing"] == 1
    assert report.exit_code == 1


# ---------------------------------------------------------- the panel probe


def test_the_probe_reports_an_absent_store_rather_than_raising(tmp_path) -> None:
    probe = probe_bar_coverage(["AAPL", "MSFT"], ["2026-07-13"], tmp_path / "nothing-here")
    assert probe["exists"] is False
    assert probe["symbols_with_a_cache"] == 0
    assert probe["per_session"] == {"2026-07-13": 0}


def test_the_probe_counts_a_store_that_covers_some_of_the_sessions(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    store = tmp_path / "store"
    store.mkdir()
    frame = pd.DataFrame(
        {"open": [1.0, 2.0], "close": [1.5, 2.5]},
        index=pd.to_datetime(["2026-07-13", "2026-07-14"]),
    )
    frame.to_parquet(store / "AAPL_1d_bars.parquet")
    probe = probe_bar_coverage(["AAPL", "MSFT"], ["2026-07-13", "2026-08-11"], store)
    assert probe["symbols_with_a_cache"] == 1
    assert probe["per_session"] == {"2026-07-13": 1, "2026-08-11": 0}
    assert probe["latest_cached_bar"] == "2026-07-14"


def test_a_probe_that_covers_nothing_keeps_the_decision_check_uncompared(tmp_path) -> None:
    report = run_gate(FIXTURE_DIR, store_dir=tmp_path / "empty")
    decision = _check(report, "decision-parity")
    assert decision.verdict == NOT_COMPARED
    assert decision.evidence["coverage"]["probed"] is True
    assert decision.evidence["shortfall"]["2026-07-13"] == "0/119 symbols have a bar"


# ------------------------------------------------------- the position carry


def test_a_fill_settles_into_the_book_on_its_filled_bar_and_not_before() -> None:
    """The carry rule is settle, not decision: a fill lands when it fills."""
    fills = [
        {"symbol": "AAA", "qty": 10.0, "decision_bar": "2026-07-13", "filled_bar": "2026-07-14"},
        {"symbol": "AAA", "qty": -4.0, "decision_bar": "2026-08-11", "filled_bar": "2026-08-12"},
    ]
    assert positions_before(fills, "2026-07-13") == {}
    assert positions_before(fills, "2026-07-14") == {"AAA": 10.0}
    assert positions_before(fills, "2026-08-11") == {"AAA": 10.0}
    assert positions_before(fills, "2026-08-12") == {"AAA": 6.0}


def test_a_position_that_nets_to_flat_leaves_the_book() -> None:
    fills = [
        {"symbol": "AAA", "qty": 10.0, "filled_bar": "2026-07-14"},
        {"symbol": "AAA", "qty": -10.0, "filled_bar": "2026-07-15"},
    ]
    assert positions_before(fills, "2026-07-16") == {}
