"""Obligations: the total order, the crossing refusal, split rebasing, and blocked progress.

The failures this file exists to catch are the three ways an obligation can quietly stop
being an obligation: a crossing smuggled through it, a fill credited that moved the book
the wrong way, and a blocked obligation carried as though it had been satisfied.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prism.portfolio.obligations import (
    Executable,
    Obligation,
    ObligationError,
    group_by_priority_deadline,
    obligation_from_current,
    order_obligations,
    plan_progress,
    record_fill,
    split_adjust,
)


def _cover(sid: str, current: float, **kwargs) -> Obligation:
    return obligation_from_current(sid, "cover", current_qty=current, source="recall", **kwargs)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_target_qty_is_a_signed_position_and_a_crossing_is_refused() -> None:
    # A short of -100 reduced to -30 is legal; "reduce to +30" crosses zero.
    reduced = obligation_from_current("AAA", "reduce_to", current_qty=-100.0, target_qty=-30.0)
    assert reduced.remaining_qty == pytest.approx(70.0)
    with pytest.raises(ObligationError, match="crosses zero"):
        obligation_from_current("AAA", "reduce_to", current_qty=-100.0, target_qty=30.0)


def test_an_obligation_only_reduces() -> None:
    with pytest.raises(ObligationError, match="only reduces"):
        obligation_from_current("AAA", "reduce_to", current_qty=-30.0, target_qty=-100.0)


def test_a_cover_applies_to_a_short_and_ends_flat() -> None:
    assert _cover("AAA", -100.0).target_qty == 0.0
    with pytest.raises(ObligationError, match="applies to a short"):
        _cover("AAA", 100.0)
    with pytest.raises(ObligationError, match="ends flat"):
        Obligation(sid="AAA", kind="cover", target_qty=-5.0, remaining_qty=5.0)


def test_remaining_qty_is_a_non_negative_magnitude() -> None:
    with pytest.raises(ObligationError, match="non-negative magnitude"):
        Obligation(sid="AAA", kind="liquidate", target_qty=0.0, remaining_qty=-1.0)


# ---------------------------------------------------------------------------
# The total order
# ---------------------------------------------------------------------------


def test_the_order_is_priority_deadline_notional_sid_and_is_total() -> None:
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 10.0, "DDD": 10.0})
    obligations = [
        Obligation(sid="DDD", kind="liquidate", target_qty=0.0, remaining_qty=5.0, priority=2, deadline=None),
        Obligation(sid="CCC", kind="liquidate", target_qty=0.0, remaining_qty=5.0, priority=1, deadline=None),
        Obligation(sid="BBB", kind="liquidate", target_qty=0.0, remaining_qty=5.0, priority=1, deadline="2026-09-10"),
        Obligation(sid="AAA", kind="liquidate", target_qty=0.0, remaining_qty=9.0, priority=1, deadline="2026-09-10"),
    ]
    ordered = order_obligations(obligations, prices)
    # priority 1 before 2; a real deadline before a null one; inside the (1, 2026-09-10)
    # group the larger remaining notional first (AAA 900 before BBB 500).
    assert [ob.sid for ob in ordered] == ["AAA", "BBB", "CCC", "DDD"]
    assert [key for key, _ in group_by_priority_deadline(ordered)] == [
        (1, "2026-09-10"),
        (1, None),
        (2, None),
    ]


def test_a_sid_without_a_reference_price_still_orders_deterministically() -> None:
    obligations = [
        Obligation(sid="ZZZ", kind="liquidate", target_qty=0.0, remaining_qty=1.0, priority=1),
        Obligation(sid="AAA", kind="liquidate", target_qty=0.0, remaining_qty=1.0, priority=1),
    ]
    assert [ob.sid for ob in order_obligations(obligations, pd.Series(dtype=float))] == ["AAA", "ZZZ"]


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def test_a_split_rebases_the_obligation_by_the_factor_that_rebases_the_position() -> None:
    obligation = obligation_from_current("AAA", "reduce_to", current_qty=-100.0, target_qty=-30.0)
    assert obligation.remaining_qty == pytest.approx(70.0)
    after = split_adjust(obligation, 2.0)
    assert after.remaining_qty == pytest.approx(140.0)
    assert after.target_qty == pytest.approx(-60.0)
    with pytest.raises(ObligationError, match="finite and > 0"):
        split_adjust(obligation, 0.0)


def test_remaining_qty_moves_only_on_an_observed_fill_and_never_the_wrong_way() -> None:
    obligation = _cover("AAA", -100.0)
    partial = record_fill(obligation, 40.0)  # a cover buys
    assert partial.remaining_qty == pytest.approx(60.0) and partial.status == "partial"
    done = record_fill(partial, 60.0)
    assert done.remaining_qty == 0.0 and done.status == "done"
    with pytest.raises(ObligationError, match="moves away from target_qty"):
        record_fill(obligation, -10.0)


def test_a_broker_initiated_fill_leaves_nothing_further_to_do() -> None:
    # A buy-in flattens the short outside this engine's orders; the live gap, not the
    # stale remaining_qty, decides what is still owed.
    obligation = _cover("AAA", -100.0)
    assert obligation.shortfall_qty(current_qty=0.0) == 0.0
    assert obligation.direction(current_qty=0.0) == 0.0


# ---------------------------------------------------------------------------
# Progress against the executable set
# ---------------------------------------------------------------------------


def test_an_obligation_is_the_progress_the_executable_set_allows_not_a_fixed_trade() -> None:
    obligation = _cover("AAA", -100.0)
    plan = plan_progress(
        [obligation],
        positions={"AAA": -100.0},
        executable={"AAA": Executable(sid="AAA", max_reduce_qty=30.0, increment=1.0, reason="5% of ADV")},
        reference_prices=pd.Series({"AAA": 10.0}),
    )
    assert len(plan.legs) == 1
    leg = plan.legs[0]
    assert leg.qty == pytest.approx(30.0)  # a cover buys back toward flat
    assert leg.partial and leg.reason == "5% of ADV"
    assert plan.protected_sids == {"AAA"}
    # The unmet 70 shares at 10 dollars is what the repair measure sees.
    assert plan.shortfall_dollars(pd.Series({"AAA": 10.0})) == pytest.approx(700.0)


def test_a_blocked_obligation_escalates_with_its_reason() -> None:
    obligation = _cover("AAA", -100.0)
    plan = plan_progress(
        [obligation],
        positions={"AAA": -100.0},
        executable={"AAA": Executable(sid="AAA", max_reduce_qty=0.0, reason="halted")},
        reference_prices=pd.Series({"AAA": 10.0}),
    )
    assert not plan.legs
    assert [blocked.reason for blocked in plan.blocked] == ["halted"]
    assert plan.escalations == ("AAA: halted",)
    assert plan.shortfall_dollars(pd.Series({"AAA": 10.0})) == pytest.approx(1_000.0)


def test_on_blocked_obligation_record_does_not_escalate() -> None:
    plan = plan_progress(
        [_cover("AAA", -100.0)],
        positions={"AAA": -100.0},
        executable={"AAA": Executable(sid="AAA", max_reduce_qty=0.0, reason="halted")},
        on_blocked_obligation="record",
    )
    assert plan.blocked and plan.escalations == ()


def test_a_missing_executable_row_blocks_rather_than_permitting_everything() -> None:
    plan = plan_progress([_cover("AAA", -100.0)], positions={"AAA": -100.0}, executable={})
    assert not plan.legs and "no executable allowance supplied" in plan.blocked[0].reason


def test_the_higher_priority_obligation_consumes_the_allowance_first() -> None:
    # Two obligations on one name and only 30 shares of room: priority 1 takes it all.
    urgent = Obligation(sid="AAA", kind="reduce_to", target_qty=-50.0, remaining_qty=50.0, priority=1, source="recall")
    later = Obligation(
        sid="AAA", kind="liquidate", target_qty=0.0, remaining_qty=100.0, priority=5, source="delisting_notice"
    )
    plan = plan_progress(
        [later, urgent],
        positions={"AAA": -100.0},
        executable={"AAA": Executable(sid="AAA", max_reduce_qty=30.0, increment=1.0)},
    )
    assert len(plan.legs) == 1
    assert plan.legs[0].obligation.priority == 1
    assert plan.legs[0].qty == pytest.approx(30.0)
    assert [blocked.obligation.priority for blocked in plan.blocked] == [5]


def test_progress_is_floored_to_the_quantity_increment() -> None:
    plan = plan_progress(
        [_cover("AAA", -100.0)],
        positions={"AAA": -100.0},
        executable={"AAA": Executable(sid="AAA", max_reduce_qty=30.7, increment=1.0)},
    )
    assert plan.legs[0].qty == pytest.approx(30.0)
