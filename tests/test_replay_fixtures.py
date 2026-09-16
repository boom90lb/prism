"""Integrity of the stage-0 baseline replay fixture (tests/fixtures/paper_replay).

Stage 1 owns the parity test that replays these sessions. This file owns the
fixture itself, and asks only questions the fixture must answer on its own: is
it the run directory's own arithmetic, does the manifest describe what was
actually captured, and does it claim a panel it does not have. A parity gate
built on a fixture nobody checked is a gate that passes for the wrong reason.

Skipped, not failed, when the fixture data is absent: the capture reads a live
run directory that only the operator box holds.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "paper_replay"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason=(
        f"no baseline replay fixture at {MANIFEST_PATH} — regenerate it against the recorded run "
        "directory with tests/fixtures/paper_replay/capture.py (its manifest records the command)"
    ),
)

DECISION_BARS = ["2026-07-13", "2026-08-11"]
HOLD_BARS = ["2026-08-12", "2026-08-13", "2026-08-14"]


def _jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURE_DIR / name).read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def targets() -> dict[str, dict]:
    return {row["decision_bar"]: row for row in _jsonl("targets.jsonl")}


@pytest.fixture(scope="module")
def fills() -> list[dict]:
    return _jsonl("fills.jsonl")


@pytest.fixture(scope="module")
def unfilled() -> list[dict]:
    return _jsonl("unfilled.jsonl")


# ---------------------------------------------------------------------------
# The manifest describes what was captured
# ---------------------------------------------------------------------------


def test_the_fixture_set_is_the_five_recorded_sessions(manifest):
    bars = [s["bar"] for s in manifest["sessions"]]
    assert bars == sorted(DECISION_BARS + HOLD_BARS)
    kinds = {s["bar"]: s["kind"] for s in manifest["sessions"]}
    assert [b for b in bars if kinds[b] == "decision"] == DECISION_BARS
    assert [b for b in bars if kinds[b] == "hold"] == HOLD_BARS
    assert manifest["decision_bearing_bars"] == DECISION_BARS
    assert manifest["hold_bars"] == HOLD_BARS
    # A hold session decides nothing: it may carry a mark and a concordance read
    # and must carry no target, no order and no fill.
    for session in manifest["sessions"]:
        assert session["has_equity_mark"], session["bar"]
        if session["kind"] == "hold":
            assert (session["n_targets"], session["n_fills"], session["n_unfilled"]) == (0, 0, 0), session["bar"]


def test_manifest_digests_match_the_captured_files(manifest):
    for name, meta in manifest["files"].items():
        path = FIXTURE_DIR / name
        assert path.exists(), f"manifest lists {name} but it is not in the fixture"
        blob = path.read_bytes()
        assert len(blob) == meta["bytes"], name
        assert hashlib.sha256(blob).hexdigest() == meta["sha256"], name
    # The ledgers are byte copies of the run directory's own files, so the
    # source digests must agree too — that is what ties the fixture to the record.
    for name, meta in manifest["source"]["files"].items():
        captured = FIXTURE_DIR / ("state_final.json" if name == "state.json" else name)
        assert hashlib.sha256(captured.read_bytes()).hexdigest() == meta["sha256"], name


def test_manifest_session_counts_match_the_ledgers(manifest, targets, fills, unfilled):
    concordance = _jsonl("concordance.jsonl")
    for session in manifest["sessions"]:
        bar = session["bar"]
        assert session["n_fills"] == sum(1 for f in fills if f["decision_bar"] == bar), bar
        assert session["n_unfilled"] == sum(1 for u in unfilled if u["decision_bar"] == bar), bar
        assert session["n_concordance"] == sum(1 for c in concordance if c["decision_bar"] == bar), bar
        assert session["n_targets"] == len(targets[bar]["targets"]) if bar in targets else session["n_targets"] == 0


# ---------------------------------------------------------------------------
# Self-consistency of the recorded ledgers
# ---------------------------------------------------------------------------


def _exit_context(targets, fills):
    """Fills split into decided-book trades and exits of the previous book.

    A refresh writes only the names it decided a weight for, so a name the
    previous book held and this one dropped is traded out with no row in this
    bar's `targets`/`reference_prices`. Both halves must still trace to a
    recorded decision, and mixing them up is how a parity test comes to accept
    an order the book never authorised.
    """
    order = sorted(targets)
    decided, exits = [], []
    for fill in fills:
        bar = fill["decision_bar"]
        (decided if fill["symbol"] in targets[bar]["reference_prices"] else exits).append(fill)
    return order, decided, exits


def test_every_fill_references_a_recorded_target(manifest, targets, fills):
    assert sorted(targets) == DECISION_BARS
    order, decided, exits = _exit_context(targets, fills)
    prefixes = set()
    for fill in fills:
        parts = fill["client_order_id"].split(":")
        assert parts[1:3] == [fill["decision_bar"], fill["symbol"]], fill["client_order_id"]
        assert len(parts) == 3 or parts[3] == "S1", fill["client_order_id"]
        prefixes.add(parts[0])
    assert len(prefixes) == 1, f"mixed order-id namespaces in one book: {sorted(prefixes)}"
    assert len({f["client_order_id"] for f in fills}) == len(fills), "duplicate fill rows"

    for fill in decided:
        book = targets[fill["decision_bar"]]["reference_prices"]
        assert fill["reference_price"] == book[fill["symbol"]], fill["client_order_id"]

    state = json.loads((FIXTURE_DIR / "state_final.json").read_text())
    for fill in exits:
        bar, symbol = fill["decision_bar"], fill["symbol"]
        prior = order[order.index(bar) - 1] if order.index(bar) else None
        assert prior is not None, f"{symbol} was traded at the first refresh {bar} without a target"
        weight = targets[prior]["targets"].get(symbol)
        assert weight is not None, f"{symbol} filled at {bar} was in no decided book"
        assert fill["qty"] * weight < 0, f"{symbol} at {bar} trades further into a dropped position"
        assert state["positions"].get(symbol, 0.0) == 0.0, f"{symbol} was exited at {bar} but is still held"

    counts = {s["bar"]: s.get("n_exit_symbols", 0) for s in manifest["sessions"]}
    for bar in DECISION_BARS:
        assert counts[bar] == len({f["symbol"] for f in exits if f["decision_bar"] == bar}), bar


def test_unfilled_residuals_reconcile_against_the_fills(targets, fills, unfilled):
    original: dict[str, float] = defaultdict(float)
    sweep: dict[str, float] = defaultdict(float)
    for fill in fills:
        parts = fill["client_order_id"].split(":")
        bucket = original if len(parts) == 3 else sweep
        bucket[":".join(parts[:3])] += fill["qty"]

    rows = {row["client_order_id"]: row for row in unfilled}
    assert len(rows) == len(unfilled), "duplicate unfilled rows"
    _, _, exits = _exit_context(targets, fills)
    exit_symbols = {(f["decision_bar"], f["symbol"]) for f in exits}
    for oid, row in rows.items():
        key = (row["decision_bar"], row["symbol"])
        assert row["symbol"] in targets[row["decision_bar"]]["reference_prices"] or key in exit_symbols, oid
        assert row["order_qty"] == pytest.approx(row["executed_qty"] + row["residual_qty"])
        assert row["executed_qty"] == pytest.approx(original.get(oid, 0.0)), oid
        assert row["residual_qty"] != 0.0, f"{oid} is a residual row with no residual"
    for oid, qty in sweep.items():
        assert oid in rows, f"sweep fill {oid}:S1 completes no recorded residual"
        residual = rows[oid]["residual_qty"]
        assert qty * residual > 0, f"{oid}:S1 fills against the residual's sign"
        assert abs(qty) <= abs(residual) + 1e-9, f"{oid}:S1 overfills its residual"


def test_equity_marks_are_ordered_and_land_on_the_reconciled_state(manifest):
    equity = _jsonl("equity.jsonl")
    bars = [row["decision_bar"] for row in equity]
    assert bars == sorted(bars) and len(set(bars)) == len(bars), "equity marks are not one ordered row per bar"
    assert set(DECISION_BARS + HOLD_BARS) <= set(bars)
    assert bars[-1] == HOLD_BARS[-1]
    state = json.loads((FIXTURE_DIR / "state_final.json").read_text())
    assert equity[-1]["cash"] == pytest.approx(state["cash"])
    for row in equity:
        assert row["equity"] > 0.0 and row["cash"] > 0.0, row


def test_concordance_reads_reference_a_decision_bearing_refresh(manifest):
    concordance = _jsonl("concordance.jsonl")
    bars = [row["decision_bar"] for row in concordance]
    assert bars == sorted(bars) and len(set(bars)) == len(bars)
    for row in concordance:
        assert row["refresh_bar"] in DECISION_BARS, row
        assert row["refresh_bar"] < row["decision_bar"] or row["decision_bar"] == row["refresh_bar"]
        assert row["n_held"] > 0 and row["n_target"] > 0


def test_universe_is_the_trimmed_set_the_sessions_touched(manifest, targets, fills, unfilled):
    listed = (FIXTURE_DIR / "universe.txt").read_text().split()
    assert listed == sorted(set(listed)), "universe.txt is unsorted or holds duplicates"
    assert len(listed) == manifest["universe"]["fixture_size"]
    state = json.loads((FIXTURE_DIR / "state_final.json").read_text())
    involved = set(state["positions"])
    for row in targets.values():
        involved |= set(row["targets"]) | set(row["reference_prices"])
    involved |= {row["symbol"] for row in fills} | {row["symbol"] for row in unfilled}
    assert set(listed) == involved
    # Trimmed, and honest that it is: the sessions ranked the whole index.
    assert len(listed) < manifest["universe"]["recorded_scored_universe_size"]


# ---------------------------------------------------------------------------
# The panel claim
# ---------------------------------------------------------------------------


def test_panels_exist_exactly_when_the_manifest_claims_them(manifest):
    panels = manifest["panels"]
    files = sorted((FIXTURE_DIR / "panels").glob("*.csv")) if (FIXTURE_DIR / "panels").is_dir() else []
    if panels["status"] == "reconstructed":
        assert [p.name for p in files] == ["close.csv", "open.csv", "volume.csv"]
        for row in panels["per_session_coverage"]:
            assert row["symbols_missing_a_bar"] == 0, row
    else:
        assert panels["status"] == "unavailable"
        assert not files, "the manifest says the panels could not be rebuilt, but panel files exist"
        assert panels["reason"].strip(), "an unavailable panel must say why"
        # A reader of the caveats alone must not come away thinking this fixture
        # is replayable, so the missing panel leads the list, not the footnotes.
        assert manifest["caveats"][0].startswith("NO PANELS.")


def test_the_manifest_carries_the_stated_limitations(manifest):
    text = " ".join(manifest["caveats"])
    assert manifest["reconstruction_date"]
    assert manifest["source"]["run_dir"] and manifest["source"]["traded_commit"]
    # The three limitations stage 1 must not replay without: the revision risk
    # the store's own split path exists for, the two-session ceiling, and the
    # modeled-vs-auction fill gap.
    assert "SplitRewriteDetected" in text
    assert "Two decision-bearing sessions" in text
    assert "ReplayBroker" in text
