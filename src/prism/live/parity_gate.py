"""The stage 5 replay parity gate (engine specification stage 5, "The transition is gated").

The specification gates the switch: recorded sessions replay under the new path
and must match the recorded targets and orders within a declared tolerance for
extraction parity, every intentional difference is approved, and only then is
the live checkout advanced and the wrappers repointed.

This module is the runnable half of that gate. It reads a session capture, runs
six checks over it, and writes one artifact carrying the verdict, the declared
tolerance, and the per-session differences. The tolerance lives in the artifact,
not in prose, so a report and the numbers it was judged against never drift
apart.

**What this gate can and cannot prove on the stage-0 capture.**
``tests/fixtures/paper_replay`` records ``panels.status = "unavailable"``: 119 of
119 symbols have no bar at any of the five captured sessions, because the live
loop fetched its bars from Alpaca over the network and persisted none. Without
those bars the candidate side of a decision comparison cannot be produced at
all, so the end-to-end decision parity is reported as NOT COMPARED with the
reason, and the gate exits nonzero. It is never reported as a pass. Identical
over nothing is not agreement.

What *is* demonstrable without the panels, and what this gate therefore
demonstrates:

* **Capture integrity.** Every ledger the manifest lists still hashes to the
  digest recorded at capture.
* **Extraction parity.** The new engine's reader recovers every captured
  session, with the same per-session target counts the capture recorded through
  a different code path, and the recovered books compare identical to
  themselves at the declared tolerance.
* **Order sizing parity.** This is the substantive claim. For each decision
  session the gate rebuilds the parent order set from the recorded decision
  inputs (the targets ledger's book, its equity, its reference prices, and a
  position book carried forward from the recorded fills) through
  :func:`prism.assembly.size_orders`, which is the *single* sizing
  implementation the live protocol and the evaluator now share, and compares it
  share for share against the order set the run actually submitted. That
  exercises settle, mark and order arithmetic against the recorded ledgers,
  which is exactly what the capture's own manifest says the trimmed universe is
  good for.
* **Refutability.** A deliberately perturbed candidate must be reported as
  differing. A gate that cannot fail proves nothing, so if the perturbation is
  not caught the whole run fails.

The order-sizing claim is not new here. ``tests/test_assembly_replay_parity.py``
established it at stage 1, in a stronger form that also runs composition, the
band and participation and reproduces the recorded equity mark. This gate
aggregates it into one artifact with a declared tolerance, a verdict and an
exit code, so the transition can be judged from a single file.

What is *not* demonstrable here, and what would close it, is stated on the
report: re-deriving the cross-sectional decision itself needs daily open and
close bars for the roughly 503 symbols these sessions actually scored, over the
274 trailing sessions of history before the first decision bar through the last
captured session, on the split-adjustment basis the vendor served on those
dates. See :data:`WHAT_WOULD_CLOSE_DECISION_PARITY`.

Run it::

    python -m prism.live.parity_gate
        --capture tests/fixtures/paper_replay
        --store-dir data/store
        --out docs/engine_overhaul_parity_gate.json

Exit codes: 0 every check passed, 1 a check failed, 2 a check could not be
evaluated (the stage-0 capture's state).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------- the tolerance

#: Absolute per-name portfolio-weight tolerance. Two books agree when no name's
#: weight differs by more than this. The artifact restates it in dollars at the
#: capture's own smallest recorded equity and in shares at the capture's own
#: cheapest recorded reference price, so the reader can see what the number buys
#: rather than taking the exponent on faith.
WEIGHT_TOLERANCE = 1e-9

WEIGHT_TOLERANCE_RATIONALE = (
    "Absolute, per name, on the decided book. Set far below one share of the cheapest name in "
    "the capture at the capture's own equity, so no weight difference inside the tolerance can "
    "change a single share of any order. It is a float-arithmetic band, not an economic "
    "allowance: a difference this gate tolerates is one that cannot reach the venue."
)

#: Absolute per-name share tolerance on the order set. Zero: orders are whole
#: shares, and there is no band in which two order sets are "the same order".
SHARE_TOLERANCE = 0.0

SHARE_TOLERANCE_RATIONALE = (
    "Zero. Orders leave this engine as whole shares (paper_books.BookParams.whole_shares is "
    "True for every shipped book), so one share of difference is a different order and there is "
    "no rounding band that makes two order sets the same. A nonzero share tolerance would hide "
    "exactly the class of difference this gate exists to find."
)

#: The perturbation the refutation check injects, as a multiple of the weight
#: tolerance. Large enough that it must be caught, small enough that catching it
#: says the tolerance is where it is declared to be.
REFUTATION_WEIGHT_MULTIPLE = 10.0

WHAT_WOULD_CLOSE_DECISION_PARITY: dict[str, Any] = {
    "missing_input": "daily OHLCV bars for the sessions the capture covers",
    "bar_source": (
        "any source that can serve daily open and close bars for these dates offline. The live "
        "loop read them from Alpaca and persisted none, and the local store "
        "(/home/b90/prism/data/store) holds 15 symbols with a latest bar of 2026-07-06, which "
        "reaches neither the universe nor the dates. Restoring them means a vendor pull into the "
        "bar store, not a code change."
    ),
    "symbols": (
        "the roughly 503 symbols the sessions actually scored, not the 119 the capture trimmed "
        "to. The trimmed set is the names these sessions priced, traded or held; the cross-"
        "sectional decile ranking cannot be re-derived from it, because a decile boundary is a "
        "property of the whole scored panel. data/universe/sp500_pit_resolved_2026-07-17.txt is "
        "the closest point-in-time membership list in the tree."
    ),
    "dates": (
        "2025-06-09 through 2026-08-14 inclusive: the 274 trailing sessions of history the "
        "signal requires before the first decision bar 2026-07-13 (252 lookback + 21 skip + 1), "
        "plus the 25 sessions from 2026-07-13 to the last captured session 2026-08-14."
    ),
    "adjustment_basis": (
        "the split-adjustment basis the vendor served on those dates. A panel rebuilt today may "
        "sit on a different basis after a vendor back-rewrite (io/store.py carries "
        "SplitRewriteDetected for exactly this), so a rebuilt panel makes replay parity a bounded "
        "claim about arithmetic rather than an unqualified reproduction of the session."
    ),
    "third_decision_session": (
        "the specification asks for at least three decision-bearing sessions and this capture has "
        "two: targets.jsonl holds exactly two rows. A third refresh must come from a session the "
        "loop actually runs, which the observation window supplies."
    ),
}

PASS = "PASS"
FAIL = "FAIL"
NOT_COMPARED = "NOT COMPARED"


# ------------------------------------------------------------------- the report


@dataclass(frozen=True)
class ParityCheck:
    """One check, its verdict, and the evidence it stands on."""

    name: str
    verdict: str
    detail: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "verdict": self.verdict,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class SessionRow:
    """One captured session, and what the gate could say about it."""

    bar: str
    kind: str
    target_parity: str
    order_parity: str
    decision_parity: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bar": self.bar,
            "kind": self.kind,
            "target_parity": self.target_parity,
            "order_parity": self.order_parity,
            "decision_parity": self.decision_parity,
            **dict(self.detail),
        }


@dataclass(frozen=True)
class GateReport:
    """The artifact. It carries its own tolerance, so it can be judged alone."""

    capture: Mapping[str, Any]
    checks: tuple[ParityCheck, ...]
    sessions: tuple[SessionRow, ...]
    tolerance: Mapping[str, Any]
    notes: tuple[str, ...] = ()
    generated_at: str = ""

    @property
    def failed(self) -> tuple[ParityCheck, ...]:
        return tuple(check for check in self.checks if check.verdict == FAIL)

    @property
    def uncompared(self) -> tuple[ParityCheck, ...]:
        return tuple(check for check in self.checks if check.verdict == NOT_COMPARED)

    @property
    def verdict(self) -> str:
        if self.failed:
            return "FAILED"
        if self.uncompared:
            return "INCOMPLETE"
        return "PASSED"

    @property
    def exit_code(self) -> int:
        return {"FAILED": 1, "INCOMPLETE": 2, "PASSED": 0}[self.verdict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": "engine overhaul stage 5 replay parity gate",
            "gate_module": "prism.live.parity_gate",
            "generated_at": self.generated_at,
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "declared_tolerance": dict(self.tolerance),
            "capture": dict(self.capture),
            "checks": [check.to_dict() for check in self.checks],
            "sessions": [row.to_dict() for row in self.sessions],
            "what_would_close_decision_parity": dict(WHAT_WOULD_CLOSE_DECISION_PARITY),
            "capture_caveats": list(self.notes),
        }

    def summary(self) -> str:
        lines = [
            f"replay parity gate: {self.verdict} (exit {self.exit_code})",
            f"capture {self.capture.get('dir')} traded by {self.capture.get('traded_commit')}, "
            f"{self.capture.get('sessions')} session(s)",
            f"declared tolerance: weight {self.tolerance['weight_absolute']:g} absolute per name, "
            f"share {self.tolerance['share_absolute']:g} absolute per name",
        ]
        for check in self.checks:
            lines.append(f"  [{check.verdict}] {check.name}: {check.detail}")
        lines.append("per session:")
        for row in self.sessions:
            lines.append(
                f"  {row.bar} ({row.kind}): targets {row.target_parity}, orders "
                f"{row.order_parity}, decision {row.decision_parity}"
            )
        if self.uncompared:
            lines.append(
                "This report does not say the new path decides the same thing. It says the "
                "checks above passed and the decision comparison was never run."
            )
        return "\n".join(lines)


# ------------------------------------------------------------------ the checks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_capture_integrity(capture_dir: Path, manifest: Mapping[str, Any]) -> ParityCheck:
    """Every ledger the manifest lists still hashes to its recorded digest."""
    recorded = manifest.get("files") or {}
    checked: dict[str, str] = {}
    broken: list[str] = []
    for name, entry in sorted(recorded.items()):
        path = capture_dir / name
        if not path.is_file():
            broken.append(f"{name}: absent")
            continue
        digest = _sha256(path)
        size = path.stat().st_size
        if digest != entry.get("sha256"):
            broken.append(f"{name}: sha256 {digest[:12]} against recorded {str(entry.get('sha256'))[:12]}")
        elif size != entry.get("bytes"):
            broken.append(f"{name}: {size} bytes against recorded {entry.get('bytes')}")
        else:
            checked[name] = digest
    if broken:
        return ParityCheck(
            "capture-integrity",
            FAIL,
            f"{len(broken)} of {len(recorded)} recorded file(s) do not match the manifest: {broken}",
            {"files_checked": len(recorded), "mismatches": broken},
        )
    return ParityCheck(
        "capture-integrity",
        PASS,
        f"{len(checked)} recorded ledger(s) hash to the digests the capture wrote",
        {"files_checked": len(checked), "digests": checked},
    )


def check_recorded_extraction(capture_dir: Path, manifest: Mapping[str, Any]) -> tuple[ParityCheck, Any]:
    """The new engine's reader recovers every captured session, with the recorded counts.

    This is a real cross-check and not a tautology: ``capture.py`` counted the
    targets of each session as it wrote the capture, and
    :func:`prism.live.replay.recorded_decisions` recounts them now by a different
    path. A reader that dropped a name, or read a hold as a gap, disagrees here.
    """
    from prism.live.replay import recorded_decisions

    recorded = recorded_decisions(capture_dir, label="recorded")
    sessions = manifest.get("sessions") or []
    expected_bars = [str(session["bar"]) for session in sessions]
    problems: list[str] = []
    if recorded.unavailable:
        problems.append(f"reader could not recover {sorted(recorded.unavailable)}")
    if sorted(recorded.decisions) != sorted(expected_bars):
        problems.append(f"recovered {sorted(recorded.decisions)} against manifest {sorted(expected_bars)}")
    counts: dict[str, int] = {}
    for session in sessions:
        bar = str(session["bar"])
        decision = recorded.decisions.get(bar)
        if decision is None:
            continue
        counts[bar] = len(decision.targets)
        if str(session.get("kind")) == "decision":
            if len(decision.targets) != int(session["n_targets"]):
                problems.append(
                    f"{bar}: reader found {len(decision.targets)} targets, capture recorded " f"{session['n_targets']}"
                )
            if not decision.refreshed:
                problems.append(f"{bar}: capture calls this a decision, reader read a hold")
        else:
            if decision.refreshed or decision.targets:
                problems.append(f"{bar}: capture calls this a hold, reader read a decision")
    verdict = FAIL if problems else PASS
    detail = (
        "; ".join(problems)
        if problems
        else (
            f"{len(recorded.decisions)}/{len(expected_bars)} session(s) recovered, per-session target "
            f"counts equal to the counts the capture recorded independently"
        )
    )
    return (
        ParityCheck(
            "recorded-extraction",
            verdict,
            detail,
            {"target_counts": counts, "engine_version_of_record": recorded.engine_version},
        ),
        recorded,
    )


def check_extraction_parity(recorded: Any, *, weight_tolerance: float, share_tolerance: float) -> ParityCheck:
    """The recovered books compare identical to themselves at the declared tolerance.

    The comparator's own zero point. A comparator that reported spurious
    differences here would make every later difference unreadable.
    """
    from prism.live.replay import upgrade_diff

    report = upgrade_diff(
        recorded,
        recorded,
        weight_tolerance=weight_tolerance,
        share_tolerance=share_tolerance,
    )
    if not report.complete or not report.identical:
        return ParityCheck(
            "extraction-parity",
            FAIL,
            f"{len(report.differing)} session(s) differ from themselves, " f"{len(report.incomparable)} not compared",
            {"report": report.to_dict()},
        )
    ordered = [s.bar for s in report.comparable if s.baseline_refreshed]
    return ParityCheck(
        "extraction-parity",
        PASS,
        f"{len(report.comparable)}/{len(report.sessions)} session(s) compared, 0 differing at the "
        f"declared tolerance, orders compared on {len(ordered)} decision session(s)",
        {
            "sessions_compared": len(report.comparable),
            "sessions_differing": len(report.differing),
            "order_sets_compared": ordered,
        },
    )


def positions_before(fills: Sequence[Mapping[str, Any]], bar: str) -> dict[str, float]:
    """The share book a session opened with, carried forward from recorded fills.

    Settle rule, not decision rule: a fill counts once its ``filled_bar`` is on
    or before ``bar``. The opening book at the first captured decision is flat,
    which is an assumption the gate does not have to take on faith: if it were
    wrong, that session's recomputed order quantities would not match the
    ledger, because an order is a delta against what is held.
    """
    book: dict[str, float] = {}
    for row in fills:
        filled = str(row.get("filled_bar") or "")
        if not filled or filled > bar:
            continue
        book[str(row["symbol"])] = book.get(str(row["symbol"]), 0.0) + float(row["qty"])
    return {sid: qty for sid, qty in book.items() if qty != 0.0}


def check_order_sizing_parity(
    capture_dir: Path,
    manifest: Mapping[str, Any],
    *,
    share_tolerance: float,
) -> tuple[ParityCheck, dict[str, dict[str, Any]]]:
    """Rebuild each decision session's order set and compare it with the record.

    The substantive parity this capture supports. The inputs are all recorded:
    the decided book, the marked equity and the decision reference prices come
    from ``targets.jsonl``, and the position book is carried forward from
    ``fills.jsonl``. The output is recomputed by
    :func:`prism.assembly.size_orders`, which is the one sizing implementation
    the live protocol and the evaluator share (``live/loop.py::targets_to_orders``
    delegates to it), and it is compared against the parent orders reconstructed
    from ``unfilled.jsonl`` and ``fills.jsonl``.

    Two inputs are named rather than assumed. The sizing knobs are the shipped
    momentum book's (``min_order_notional`` 1.0, ``whole_shares`` True,
    ``order_id_prefix`` "mom:" on the recorded sessions; since 2026-09-30 live
    ids carry a run-directory digest after the book prefix,
    ``paper_books.scoped_order_id_prefix``, and this comparison keys on symbol,
    not on the prefix, ``paper_books.BookParams``), which the capture's own
    client order ids confirm. And a held name the session exited carries no
    price in the targets ledger, because that ledger prices the book and not the
    exits, so those names take their decision price from the order ledger's own
    ``reference_price``; the count is reported per session and it is a weaker
    input than the rest.
    """
    from prism.assembly import size_orders
    from prism.live.replay import _recorded_orders

    targets_rows = {str(row["decision_bar"]): row for row in _read_jsonl(capture_dir / "targets.jsonl")}
    fills = _read_jsonl(capture_dir / "fills.jsonl")
    unfilled = _read_jsonl(capture_dir / "unfilled.jsonl")
    decision_bars = [
        str(session["bar"]) for session in (manifest.get("sessions") or []) if str(session.get("kind")) == "decision"
    ]
    per_session: dict[str, dict[str, Any]] = {}
    total_recorded = 0
    total_differing = 0
    problems: list[str] = []
    for bar in sorted(decision_bars):
        row = targets_rows.get(bar)
        if row is None:
            problems.append(f"{bar}: no targets row to size from")
            per_session[bar] = {"verdict": NOT_COMPARED, "reason": "no targets row"}
            continue
        session_fills = [f for f in fills if str(f.get("decision_bar")) == bar]
        session_unfilled = [u for u in unfilled if str(u.get("decision_bar")) == bar]
        recorded_orders = _recorded_orders(session_fills, session_unfilled)
        book = positions_before(fills, bar)
        targets = {str(sid): float(w) for sid, w in (row.get("targets") or {}).items()}
        for sid in book:
            targets.setdefault(sid, 0.0)
        prices = {str(sid): float(p) for sid, p in (row.get("reference_prices") or {}).items()}
        ledger_prices: dict[str, float] = {}
        for ledger_row in (*session_unfilled, *session_fills):
            price = ledger_row.get("reference_price")
            if price is not None:
                ledger_prices.setdefault(str(ledger_row["symbol"]), float(price))
        borrowed = sorted(sid for sid in targets if sid not in prices and sid in ledger_prices)
        for sid in borrowed:
            prices[sid] = ledger_prices[sid]
        unpriceable = sorted(sid for sid in targets if sid not in prices and targets[sid] != 0.0)
        if unpriceable:
            problems.append(f"{bar}: no recorded decision price for {unpriceable}")
            per_session[bar] = {"verdict": NOT_COMPARED, "reason": f"unpriceable names {unpriceable}"}
            continue
        rebuilt = {
            order.symbol: float(order.qty)
            for order in size_orders(
                pd.Series(targets, dtype=float),
                book,
                pd.Series(prices, dtype=float),
                float(row["equity"]),
                bar,
                min_order_notional=1.0,
                whole_shares=True,
                order_id_prefix="mom:",
            )
        }
        differing = {
            sid: {"recorded": recorded_orders.get(sid), "rebuilt": rebuilt.get(sid)}
            for sid in sorted(set(recorded_orders) | set(rebuilt))
            if abs(float(rebuilt.get(sid, 0.0)) - float(recorded_orders.get(sid, 0.0))) > share_tolerance
        }
        total_recorded += len(recorded_orders)
        total_differing += len(differing)
        per_session[bar] = {
            "verdict": PASS if not differing else FAIL,
            "orders_recorded": len(recorded_orders),
            "orders_rebuilt": len(rebuilt),
            "orders_differing": len(differing),
            "differences": differing,
            "opening_positions": len(book),
            "names_priced_from_the_order_ledger": borrowed,
        }
        if differing:
            problems.append(f"{bar}: {len(differing)} order(s) differ")
    if problems:
        return (
            ParityCheck(
                "order-sizing-parity",
                FAIL,
                "; ".join(problems),
                {"per_session": per_session},
            ),
            per_session,
        )
    if not per_session:
        return (
            ParityCheck(
                "order-sizing-parity",
                NOT_COMPARED,
                "the capture holds no decision-bearing session to size",
                {"per_session": per_session},
            ),
            per_session,
        )
    return (
        ParityCheck(
            "order-sizing-parity",
            PASS,
            f"{total_recorded} recorded parent order(s) over {len(per_session)} decision session(s) "
            f"rebuilt from the recorded decision inputs, {total_differing} differing at the declared "
            f"share tolerance",
            {
                "orders_compared": total_recorded,
                "orders_differing": total_differing,
                "per_session": per_session,
            },
        ),
        per_session,
    )


def check_gate_bites(recorded: Any, *, weight_tolerance: float, share_tolerance: float) -> ParityCheck:
    """A perturbed candidate must be reported as differing.

    The refutation. Everything above is a check that finds no difference, and a
    comparator wired to a constant "no difference" would pass all of them. So
    the gate perturbs one weight by ten tolerances and one order by one share
    and requires both to come back named. If they do not, the gate is broken and
    this run means nothing.
    """
    from prism.live.replay import EngineDecisions, SessionDecision, upgrade_diff

    refreshed = sorted(bar for bar, decision in recorded.decisions.items() if decision.refreshed)
    if not refreshed:
        return ParityCheck(
            "refutation",
            NOT_COMPARED,
            "no decision-bearing session in the capture to perturb",
            {},
        )
    bar = refreshed[-1]
    original = recorded.decisions[bar]
    book = dict(original.targets)
    victim_weight = sorted(book)[0]
    book[victim_weight] = book[victim_weight] + REFUTATION_WEIGHT_MULTIPLE * max(weight_tolerance, 1e-12)
    orders = dict(original.orders or {})
    victim_order = sorted(orders)[0] if orders else None
    if victim_order is not None:
        orders[victim_order] = orders[victim_order] + 1.0
    perturbed = EngineDecisions(
        engine_version="perturbed",
        label="refutation",
        decisions={
            **recorded.decisions,
            bar: SessionDecision(bar, True, book, orders=orders, equity=original.equity),
        },
    )
    report = upgrade_diff(
        recorded,
        perturbed,
        weight_tolerance=weight_tolerance,
        share_tolerance=share_tolerance,
    )
    caught_weight = any(
        change.sid == victim_weight for session in report.differing for change in session.weight_changes
    )
    caught_order = victim_order is None or any(
        change.sid == victim_order for session in report.differing for change in session.order_changes
    )
    if not (caught_weight and caught_order):
        return ParityCheck(
            "refutation",
            FAIL,
            f"the comparator did not report a perturbation it must report on {bar}: "
            f"weight {victim_weight} caught={caught_weight}, order {victim_order} caught={caught_order}",
            {"report": report.to_dict()},
        )
    return ParityCheck(
        "refutation",
        PASS,
        f"a {REFUTATION_WEIGHT_MULTIPLE:g}x-tolerance weight move on {victim_weight} and a one-share "
        f"order change on {victim_order} at {bar} are both reported as differences",
        {
            "perturbed_session": bar,
            "perturbed_weight_name": victim_weight,
            "perturbed_order_name": victim_order,
            "sessions_reported_differing": [s.bar for s in report.differing],
        },
    )


def probe_bar_coverage(
    symbols: Sequence[str],
    bars: Sequence[str],
    store_dir: Path,
    *,
    interval: str = "1d",
) -> dict[str, Any]:
    """How many of ``symbols`` have a bar on each of ``bars`` under ``store_dir``.

    Deliberately not :func:`prism.live.replay.load_local_bar_panels`, which
    raises when the caches are missing. The question here is "how much is
    there", and the answer has to be reportable rather than fatal.
    """
    if not store_dir.is_dir():
        return {
            "store_dir": str(store_dir),
            "exists": False,
            "symbols_requested": len(symbols),
            "symbols_with_a_cache": 0,
            "per_session": {str(bar): 0 for bar in bars},
            "latest_cached_bar": None,
        }
    wanted = {str(bar) for bar in bars}
    per_session = dict.fromkeys(wanted, 0)
    cached = 0
    latest: str | None = None
    for symbol in symbols:
        paths = sorted(store_dir.glob(f"{symbol}_{interval}_*.parquet"))
        if not paths:
            continue
        cached += 1
        dates: set[str] = set()
        for path in paths:
            frame = pd.read_parquet(path)
            dates |= {str(pd.Timestamp(stamp).date()) for stamp in frame.index}
        if dates:
            newest = max(dates)
            latest = newest if latest is None else max(latest, newest)
        for bar in wanted & dates:
            per_session[bar] += 1
    return {
        "store_dir": str(store_dir),
        "exists": True,
        "symbols_requested": len(symbols),
        "symbols_with_a_cache": cached,
        "per_session": {bar: per_session[bar] for bar in sorted(per_session)},
        "latest_cached_bar": latest,
    }


def check_decision_parity(
    capture_dir: Path,
    manifest: Mapping[str, Any],
    *,
    store_dir: Path | None,
) -> ParityCheck:
    """The end-to-end comparison: what the new path would decide on these sessions.

    It runs only when a bar source covers every captured session for the scored
    universe. On the stage-0 capture nothing does, so this is NOT COMPARED with
    the reason and the requirement to close it, and never a pass.
    """
    universe_path = capture_dir / "universe.txt"
    symbols = universe_path.read_text(encoding="utf-8").split() if universe_path.is_file() else []
    bars = [str(session["bar"]) for session in (manifest.get("sessions") or [])]
    manifest_panels = dict(manifest.get("panels") or {})
    if store_dir is None:
        coverage: dict[str, Any] = {
            "probed": False,
            "note": (
                "no --store-dir given, so the gate did not probe for bars. The capture's own probe "
                "is carried instead."
            ),
            "capture_probe": manifest_panels,
        }
    else:
        probed = probe_bar_coverage(symbols, bars, store_dir)
        coverage = {"probed": True, **probed, "capture_probe": manifest_panels}
    covered = coverage.get("per_session") or {}
    complete = bool(symbols) and bool(bars) and all(covered.get(bar, 0) == len(symbols) for bar in bars)
    if not complete:
        shortfall = {
            bar: f"{covered.get(bar, 0)}/{len(symbols)} symbols have a bar"
            for bar in bars
            if covered.get(bar, 0) < len(symbols)
        }
        return ParityCheck(
            "decision-parity",
            NOT_COMPARED,
            (
                f"the candidate side cannot be produced: 0 of {len(bars)} session(s) can be compared "
                f"end to end. {manifest_panels.get('reason', 'no bar panel covers these sessions')}. "
                "The new path was never run over these sessions, so this report says nothing about "
                "whether it would decide the same thing."
            ),
            {
                "sessions_requested": len(bars),
                "sessions_compared": 0,
                "coverage": coverage,
                "shortfall": shortfall,
                "what_would_close_it": dict(WHAT_WOULD_CLOSE_DECISION_PARITY),
            },
        )
    from prism.live.replay import (
        load_local_bar_panels,
        recorded_decisions,
        replayed_decisions,
        upgrade_diff,
    )
    from prism.scripts.paper_books import BOOKS, BookParams

    book = BOOKS["momentum"]
    params = BookParams()
    decision_every = book.decision_every(params)
    close, volume, open_ = load_local_bar_panels(symbols, store_dir or Path("data"))
    baseline = recorded_decisions(capture_dir, label="recorded")
    candidate = replayed_decisions(
        lambda: book.build_signal(params, decision_every, None),
        close,
        volume,
        open_,
        book.build_config(params, decision_every),
        capture_dir / "_replay_run",
        bars=bars,
        label="candidate",
    )
    report = upgrade_diff(
        baseline,
        candidate,
        bars=bars,
        weight_tolerance=WEIGHT_TOLERANCE,
        share_tolerance=SHARE_TOLERANCE,
    )
    if not report.complete:
        return ParityCheck(
            "decision-parity",
            NOT_COMPARED,
            f"{len(report.comparable)}/{len(report.sessions)} session(s) compared: "
            + "; ".join(f"{s.bar} {s.reason}" for s in report.incomparable),
            {"report": report.to_dict(), "coverage": coverage},
        )
    verdict = PASS if report.identical else FAIL
    return ParityCheck(
        "decision-parity",
        verdict,
        f"{len(report.comparable)}/{len(report.sessions)} session(s) compared, "
        f"{len(report.differing)} with a different decision at the declared tolerance",
        {"report": report.to_dict(), "coverage": coverage},
    )


# --------------------------------------------------------------------- the gate


def _cheapest_recorded_price(capture_dir: Path) -> float | None:
    prices = [
        float(price)
        for row in _read_jsonl(capture_dir / "targets.jsonl")
        for price in (row.get("reference_prices") or {}).values()
        if price is not None and float(price) > 0
    ]
    return min(prices) if prices else None


def run_gate(
    capture_dir: Path | str,
    *,
    store_dir: Path | str | None = None,
    weight_tolerance: float = WEIGHT_TOLERANCE,
    share_tolerance: float = SHARE_TOLERANCE,
) -> GateReport:
    """Run every check over one capture and return the artifact."""
    capture_dir = Path(capture_dir)
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{capture_dir} holds no manifest.json; a capture without its manifest cannot say which "
            "sessions it covers or which engine build traded them (N7)"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    store = Path(store_dir) if store_dir is not None else None

    checks: list[ParityCheck] = [check_capture_integrity(capture_dir, manifest)]
    extraction, recorded = check_recorded_extraction(capture_dir, manifest)
    checks.append(extraction)
    checks.append(
        check_extraction_parity(recorded, weight_tolerance=weight_tolerance, share_tolerance=share_tolerance)
    )
    sizing, per_session_orders = check_order_sizing_parity(capture_dir, manifest, share_tolerance=share_tolerance)
    checks.append(sizing)
    checks.append(check_gate_bites(recorded, weight_tolerance=weight_tolerance, share_tolerance=share_tolerance))
    decision = check_decision_parity(capture_dir, manifest, store_dir=store)
    checks.append(decision)

    equities = [
        float(session["recorded_equity"])
        for session in (manifest.get("sessions") or [])
        if session.get("recorded_equity") is not None
    ]
    cheapest = _cheapest_recorded_price(capture_dir)
    smallest_equity = min(equities) if equities else None
    tolerance: dict[str, Any] = {
        "weight_absolute": weight_tolerance,
        "weight_rationale": WEIGHT_TOLERANCE_RATIONALE,
        "share_absolute": share_tolerance,
        "share_rationale": SHARE_TOLERANCE_RATIONALE,
        "smallest_recorded_equity": smallest_equity,
        "cheapest_recorded_reference_price": cheapest,
        "weight_tolerance_in_dollars_at_smallest_recorded_equity": (
            None if smallest_equity is None else weight_tolerance * smallest_equity
        ),
        "weight_tolerance_in_shares_of_the_cheapest_name": (
            None
            if smallest_equity is None or cheapest is None or cheapest <= 0
            else weight_tolerance * smallest_equity / cheapest
        ),
    }

    rows: list[SessionRow] = []
    for session in manifest.get("sessions") or []:
        bar = str(session["bar"])
        kind = str(session.get("kind", ""))
        order_row = per_session_orders.get(bar)
        rows.append(
            SessionRow(
                bar=bar,
                kind=kind,
                target_parity=extraction.verdict,
                order_parity=(order_row["verdict"] if order_row else "no orders recorded"),
                decision_parity=decision.verdict,
                detail={
                    "targets_recorded": session.get("n_targets"),
                    "orders_recorded": (order_row or {}).get("orders_recorded"),
                    "orders_rebuilt": (order_row or {}).get("orders_rebuilt"),
                    "orders_differing": (order_row or {}).get("orders_differing"),
                    "max_weight_difference": 0.0 if extraction.verdict == PASS else None,
                    "decision_parity_reason": None if decision.verdict == PASS else decision.detail,
                },
            )
        )

    return GateReport(
        capture={
            "dir": str(capture_dir),
            "traded_commit": (manifest.get("source") or {}).get("traded_commit"),
            "sessions": len(manifest.get("sessions") or []),
            "decision_bearing_sessions": list(manifest.get("decision_bearing_bars") or []),
            "universe_size": (manifest.get("universe") or {}).get("fixture_size"),
            "recorded_scored_universe_size": (manifest.get("universe") or {}).get("recorded_scored_universe_size"),
            "panels_status": (manifest.get("panels") or {}).get("status"),
        },
        checks=tuple(checks),
        sessions=tuple(rows),
        tolerance=tolerance,
        notes=tuple(str(note) for note in (manifest.get("caveats") or ())),
        generated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m prism.live.parity_gate",
        description="The stage 5 replay parity gate over a recorded session capture.",
    )
    parser.add_argument(
        "--capture",
        default="tests/fixtures/paper_replay",
        help="the session capture directory (needs a manifest.json)",
    )
    parser.add_argument(
        "--store-dir",
        default=None,
        help=("a local bar store to probe for the candidate side; omitted, the capture's own probe " "is reported"),
    )
    parser.add_argument("--out", default=None, help="write the JSON artifact here")
    parser.add_argument("--weight-tolerance", type=float, default=WEIGHT_TOLERANCE)
    parser.add_argument("--share-tolerance", type=float, default=SHARE_TOLERANCE)
    args = parser.parse_args(argv)

    report = run_gate(
        args.capture,
        store_dir=args.store_dir,
        weight_tolerance=args.weight_tolerance,
        share_tolerance=args.share_tolerance,
    )
    print(report.summary())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"artifact written to {out}")
    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
