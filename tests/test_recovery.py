"""The recovery policy, the three stages, and the partition the record must report.

The acceptance case §3.7 names is here: **gap-induced limit conflict**. An obligation the
action restrictions *can* execute makes a preference violation worse, R1 cannot undo it
without undoing the obligation progress, and the three sets ``existing``, ``remaining`` and
``introduced`` must therefore say three different things about three different limits. A
recovery that reported one merged list of "violations" would hide exactly the distinction an
operator needs: what the book arrived with, what the decision did, and what neither could
cure.

The second thing defended here is the R2 guarantee, in the form it can actually be honored:
**R2 cannot increase any violation beyond the value R1 achieved**. Not "R2 introduces none"
-- R0 and R1 can each introduce one, and a path that promised otherwise would be lying.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.portfolio.convex import ConstructionInputs, ConvexConstructor, ConvexProgram
from prism.portfolio.limits import (
    Limit,
    LimitClass,
    LimitContext,
    reduction_order,
    violation,
)
from prism.portfolio.obligations import Obligation, obligation_from_current
from prism.portfolio.recovery import (
    RecoveryError,
    RecoveryPolicy,
    finalize_batch,
    no_action,
    recovery_entry,
)
from prism.portfolio.validate import ValidationSettings
from tests.test_convex import (
    EQUITY,
    SETTINGS,
    ComponentForecast,
    cost_terms,
    fit,
    inputs,
    limit,
    profile_limits,
    scores,
    snapshot,
    solve,
    universe,
)

REFERENCE_POLICY = Path(__file__).resolve().parents[1] / "examples/reference/policies/recovery.toml"


# ---------------------------------------------------------------------------
# The policy file
# ---------------------------------------------------------------------------


def test_the_reference_recovery_policy_loads_and_declares_every_branch() -> None:
    policy = RecoveryPolicy.from_toml(REFERENCE_POLICY)
    assert policy.on_infeasible == "cancel"
    assert policy.on_sign_contradiction == "hold"
    assert policy.on_blocked_obligation == "escalate"
    assert policy.bound_slack == pytest.approx(1.0e-6)
    assert policy.recovery_entry_margin == pytest.approx(0.002)
    assert policy.escalation.exit_code == 3
    assert policy.escalation.require_acknowledgement is True
    declaration = policy.declaration()
    assert declaration["escalation"]["write_alert"] is True
    assert set(declaration) >= {"on_infeasible", "bound_slack", "recovery_entry_margin"}


def test_an_unknown_key_or_an_unknown_spelling_is_refused_never_defaulted() -> None:
    with pytest.raises(RecoveryError, match="unknown keys"):
        RecoveryPolicy.from_mapping({"on_infeasable": "hold"}, source="typo.toml")
    with pytest.raises(RecoveryError, match="on_infeasible is one of"):
        RecoveryPolicy.from_mapping({"on_infeasible": "halt"}, source="bad.toml")
    with pytest.raises(RecoveryError, match="unknown \\[escalation\\] keys"):
        RecoveryPolicy.from_mapping({"escalation": {"exit_codes": 3}}, source="bad.toml")


def test_the_no_action_record_carries_the_declared_response_and_no_orders() -> None:
    for response in ("cancel", "hold", "escalate"):
        record = no_action(RecoveryPolicy(on_infeasible=response), "nothing is executable")
        assert record["response"] == response
        assert record["orders_submitted"] is False
        assert record["assumes_no_trade_executable"] is False
        assert ("escalation" in record) == (response == "escalate")


# ---------------------------------------------------------------------------
# Which path the marked state selects
# ---------------------------------------------------------------------------


def _ctx(w0: pd.Series, **kwargs) -> LimitContext:
    return LimitContext(w0=w0, equity=EQUITY, **kwargs)


def test_a_marginal_drift_over_a_preference_limit_does_not_flip_the_path() -> None:
    """The entry margin exists so a profile whose ``max_gross`` equals its target gross does
    not enter recovery on roughly half of decision bars."""
    index = universe(2)
    rows = [limit("max_gross", "preference", 1, value=1.0)]
    inside = pd.Series([0.5005, -0.5005], index=index)  # gross 1.001, inside the 0.002 margin
    outside = pd.Series([0.503, -0.503], index=index)  # gross 1.006, outside it
    assert recovery_entry(rows, _ctx(inside), limit_tolerance=1e-6, recovery_entry_margin=0.002)[0] is False
    entered, reason = recovery_entry(rows, _ctx(outside), limit_tolerance=1e-6, recovery_entry_margin=0.002)
    assert entered is True and "max_gross" in reason


def test_an_open_obligation_enters_recovery_whatever_the_limits_say() -> None:
    index = universe(2)
    flat = pd.Series([0.0, 0.0], index=index)
    obligation = obligation_from_current("S00", "liquidate", current_qty=100.0, priority=0)
    entered, reason = recovery_entry(
        [limit("max_gross", "preference", 1, value=1.0)],
        _ctx(flat),
        [obligation],
        limit_tolerance=1e-6,
        recovery_entry_margin=0.002,
    )
    assert entered is True
    assert "S00" in reason


# ---------------------------------------------------------------------------
# The three stages
# ---------------------------------------------------------------------------


def _four() -> pd.Index:
    return pd.Index(["A", "B", "C", "D"], name="sid")


def _inputs_for(index: pd.Index, w0: pd.Series, **kwargs) -> ConstructionInputs:
    z = pd.Series(np.linspace(1.5, -1.5, len(index)), index=index)
    return inputs(
        index,
        w0=w0,
        z=z,
        sigma=pd.Series(0.0722, index=index),
        risk=snapshot(index),
        costs=cost_terms(index),
        **kwargs,
    )


def test_an_inherited_gross_breach_is_reduced_and_reported_as_remaining_not_introduced() -> None:
    index = _four()
    held = pd.Series([0.40, 0.35, -0.35, -0.30], index=index)  # gross 1.40, net 0.10
    rows = profile_limits(gross=1.0, cap=0.5, net=0.02, budget=None, participation=0.05)
    out = solve(_inputs_for(index, held, limits=rows))
    assert out.path == "recovery"

    names = {entry["limit_name"] for entry in out.violations["existing"]}
    assert names == {"max_gross", "net_band"}
    assert out.violations["introduced"] == []
    before = {entry["limit_name"]: entry["v"] for entry in out.violations["existing"]}
    after = {entry["limit_name"]: entry["v"] for entry in out.violations["remaining"]}
    for name in names:
        assert after[name] <= before[name] + 1e-9, name
    assert out.diagnostics["achieved_bounds"]["max_gross"] <= 1e-6


def test_the_gap_induced_limit_conflict_partitions_into_three_different_answers() -> None:
    """The acceptance case: an obligation's progress introduces a breach nothing can cure.

    A, B and C are halted; D carries a recall. Covering D is the highest-priority action the
    restrictions allow, and it moves the book's net exposure from zero to 0.30 against a
    0.02 band that no other name may be traded to restore. The net band is therefore
    ``introduced``, attributed to ``R0``, with the obligation named as the binding
    condition -- and it is not ``existing`` (the marked book satisfied it) and not
    ``remaining`` (that set is for limits the marked book already breached).
    """
    index = _four()
    held = pd.Series([0.30, 0.0, 0.0, -0.30], index=index)  # net 0.00, gross 0.60
    positions = held * EQUITY / 100.0
    recall = obligation_from_current(
        "D", "cover", current_qty=float(positions["D"]), priority=0, source="recall", reason="borrow recalled"
    )
    rows = profile_limits(gross=1.0, cap=0.5, net=0.02, budget=None, participation=0.40) + [
        limit("halted_no_trade", "physical", None, sids=["A", "B", "C"]),
    ]
    out = solve(
        _inputs_for(index, held, limits=rows, obligations=[recall], positions=positions),
    )
    assert out.path == "recovery"
    assert out.w_target["D"] == pytest.approx(0.0, abs=1e-6)
    for sid in ("A", "B", "C"):
        assert out.w_target[sid] == pytest.approx(held[sid], abs=1e-6)

    assert out.violations["existing"] == []
    assert out.violations["remaining"] == []
    introduced = {entry["limit_name"]: entry for entry in out.violations["introduced"]}
    assert set(introduced) == {"net_band"}
    assert introduced["net_band"]["v"] == pytest.approx(0.28, abs=1e-4)
    assert introduced["net_band"]["attributed_to"] == "R0"
    assert out.binding_condition == "obligation:D"

    progress = out.diagnostics["recovery"]["obligations"]
    assert len(progress) == 1
    assert progress[0]["achieved_qty"] == pytest.approx(progress[0]["remaining_qty"], rel=1e-6)
    assert progress[0]["blocked"] is False
    assert out.diagnostics["attribution"]["net_band"] == "R0"


def test_obligation_progress_is_never_traded_away_for_a_preference_limit() -> None:
    """R1 minimizes the violation R0 introduced; it may not undo the progress to do it."""
    index = _four()
    held = pd.Series([0.30, 0.0, 0.0, -0.30], index=index)
    positions = held * EQUITY / 100.0
    recall = obligation_from_current("D", "cover", current_qty=float(positions["D"]), priority=0, source="recall")
    rows = profile_limits(gross=1.0, cap=0.5, net=0.02, budget=None, participation=0.40) + [
        limit("halted_no_trade", "physical", None, sids=["A", "B", "C"]),
    ]
    out = solve(_inputs_for(index, held, limits=rows, obligations=[recall], positions=positions))
    # The cover is complete and stays complete through R1 and R2.
    assert out.w_target["D"] == pytest.approx(0.0, abs=1e-6)
    assert out.diagnostics["achieved_bounds"]["net_band"] == pytest.approx(0.28, abs=1e-4)


def test_the_preference_response_decides_whether_that_batch_may_be_submitted() -> None:
    """``record`` reports the introduced preference breach; ``block`` refuses the batch.

    Both are faithful: §3.7 makes obligation progress outrank every preference limit *and*
    fails an introduced preference violation under ``block``. Under ``block`` an obligation
    that can only be executed by breaking a preference therefore stops the batch, which is
    a policy consequence to see rather than a case to special-case away.
    """
    index = _four()
    held = pd.Series([0.30, 0.0, 0.0, -0.30], index=index)
    positions = held * EQUITY / 100.0
    recall = obligation_from_current("D", "cover", current_qty=float(positions["D"]), priority=0, source="recall")
    rows = profile_limits(gross=1.0, cap=0.5, net=0.02, budget=None, participation=0.40) + [
        limit("halted_no_trade", "physical", None, sids=["A", "B", "C"]),
    ]
    common = dict(limits=rows, obligations=[recall], positions=positions)
    recorded = solve(_inputs_for(index, held, validation=ValidationSettings(preference_response="record"), **common))
    blocked = solve(_inputs_for(index, held, validation=ValidationSettings(preference_response="block"), **common))
    assert recorded.path == "recovery"
    assert blocked.path == "no_action"
    assert blocked.diagnostics["no_action"]["response"] == "cancel"


def test_r1_walks_every_holding_state_limit_in_reduction_order_satisfied_ones_included() -> None:
    index = _four()
    held = pd.Series([0.40, 0.35, -0.35, -0.30], index=index)
    rows = profile_limits(gross=1.0, cap=0.5, net=0.02, budget=None, participation=0.05)
    out = solve(_inputs_for(index, held, limits=rows))
    walked = [stage["subject"] for stage in out.diagnostics["stages"] if stage["stage"] == "R1"]
    assert walked == [row.name for row in reduction_order(rows)]
    assert set(out.diagnostics["achieved_bounds"]) == set(walked)
    assert out.diagnostics["stages"][-1]["stage"] == "R2"


def test_r2_cannot_increase_any_violation_beyond_the_value_r1_achieved() -> None:
    index = _four()
    held = pd.Series([0.40, 0.35, -0.35, -0.30], index=index)
    rows = profile_limits(gross=1.0, cap=0.5, net=0.02, budget=None, participation=0.05)
    x = _inputs_for(index, held, limits=rows)
    out = solve(x)
    ctx = LimitContext(w0=held, equity=EQUITY, adv_dollars=x.adv_dollars, risk=x.risk)
    for name, achieved in out.diagnostics["achieved_bounds"].items():
        row = next(item for item in rows if item.name == name)
        assert violation(row, out.w_target, ctx) <= achieved + 1e-6, name
    assert out.diagnostics["validation"]["ok"] is True


def test_a_blocked_obligation_escalates_with_the_reason_and_the_declared_exit_code() -> None:
    index = _four()
    held = pd.Series([0.0, 0.0, 0.0, -0.30], index=index)
    positions = held * EQUITY / 100.0
    recall = obligation_from_current("D", "cover", current_qty=float(positions["D"]), priority=0, source="recall")
    rows = profile_limits(gross=1.0, cap=0.5, net=0.5, budget=None, participation=0.05) + [
        limit("halted_no_trade", "physical", None, sids=["D"]),  # the recall cannot be executed today
    ]
    out = solve(
        _inputs_for(
            index,
            held,
            limits=rows,
            obligations=[recall],
            positions=positions,
            policy=RecoveryPolicy(on_blocked_obligation="escalate"),
        )
    )
    progress = out.diagnostics["recovery"]["obligations"]
    assert progress[0]["blocked"] is True
    escalations = out.diagnostics["escalations"]
    assert escalations and escalations[0]["kind"] == "blocked_obligation"
    assert escalations[0]["exit_code"] == 3
    assert "D" in escalations[0]["reason"]
    # The halted name is not traded, and the rest of the decision still proceeds.
    assert out.w_target["D"] == pytest.approx(held["D"], abs=1e-6)


def test_an_obligation_a_broker_fill_already_satisfied_is_not_blocked_and_does_not_escalate() -> None:
    """A buy-in or a forced liquidation moves the position without this engine's order.

    The obligation is then already met, and recording it as *blocked* would escalate a
    decision on which nothing is wrong.
    """
    index = _four()
    held = pd.Series([0.30, 0.0, 0.0, 0.0], index=index)  # the recalled short is already flat
    positions = held * EQUITY / 100.0
    recall = Obligation(
        sid="D", kind="cover", target_qty=0.0, remaining_qty=3_000.0, priority=0, source="recall", status="open"
    )
    rows = profile_limits(gross=1.0, cap=0.5, net=0.5, budget=None, participation=0.40)
    out = solve(
        _inputs_for(
            index,
            held,
            limits=rows,
            obligations=[recall],
            positions=positions,
            policy=RecoveryPolicy(on_blocked_obligation="escalate"),
        )
    )
    progress = out.diagnostics["recovery"]["obligations"]
    assert progress[0]["blocked"] is False
    assert out.diagnostics["escalations"] == []


def test_obligation_groups_are_solved_one_priority_at_a_time_in_ascending_order() -> None:
    index = _four()
    held = pd.Series([0.0, 0.0, -0.20, -0.30], index=index)
    positions = held * EQUITY / 100.0
    first = obligation_from_current("D", "cover", current_qty=float(positions["D"]), priority=0, source="recall")
    second = obligation_from_current(
        "C", "cover", current_qty=float(positions["C"]), priority=1, source="broker_requirement"
    )
    rows = profile_limits(gross=1.0, cap=0.5, net=0.5, budget=None, participation=0.40)
    out = solve(_inputs_for(index, held, limits=rows, obligations=[second, first], positions=positions))
    groups = [stage["subject"] for stage in out.diagnostics["stages"] if stage["stage"] == "R0"]
    assert groups == ["priority 0, deadline None", "priority 1, deadline None"]
    achieved = {row["sid"]: row["achieved_qty"] for row in out.diagnostics["recovery"]["obligations"]}
    assert achieved["D"] == pytest.approx(3_000.0, rel=1e-6)
    assert achieved["C"] == pytest.approx(2_000.0, rel=1e-6)


def test_a_recovery_stage_that_does_not_solve_puts_the_decision_on_the_no_action_path() -> None:
    """Halted *and* required to unwind is a contradiction between two action restrictions."""
    index = _four()
    held = pd.Series([0.30, 0.0, 0.0, -0.30], index=index)
    positions = held * EQUITY / 100.0
    recall = obligation_from_current("D", "cover", current_qty=float(positions["D"]), priority=0, source="recall")
    rows = profile_limits(gross=1.0, cap=0.5, net=0.5, budget=None, participation=0.40) + [
        limit("halted_no_trade", "physical", None, sids=["A"]),
    ]
    out = solve(
        _inputs_for(
            index,
            held,
            limits=rows,
            obligations=[recall],
            positions=positions,
            fit_=fit(sign_ok=False),  # every scored name must unwind ...
            policy=RecoveryPolicy(on_sign_contradiction="unwind", on_infeasible="hold"),
        )
    )
    assert out.path == "no_action"  # ... but A is halted, so the two restrictions have no common point
    assert out.diagnostics["recovery"]["failed_stage"] == "R0"
    assert out.diagnostics["no_action"]["response"] == "hold"
    assert out.w_target.isna().all()


# ---------------------------------------------------------------------------
# The second validation: after rounding
# ---------------------------------------------------------------------------


ROUNDING_PROFILE = {
    "quantity_increment_default": 1.0,
    "min_notional": 1.0,
    "default_order_type": "market_on_open",
    "order_types": ["market_on_open"],
    "fractional": {"allowed": False, "time_in_force": ["day"], "short_allowed": False},
    "validation": {
        "limit_tolerance": 1.0e-6,
        "cash_tolerance": 0.01,
        "cost_tolerance_bps": 0.1,
        "mode": "blocking",
    },
    "rounding": {
        "tie_rule": "half_to_even",
        "zero_crossing_clamp": True,
        "apply_dust_rule": True,
        "fractional_increment": 0.001,
        "off_increment_holding": "leave_residual",
        "scale_schedule": [0.75, 0.5, 0.25],
        "repair_max_moves": 200,
        "repair_min_decrement": 1.0e-6,
        "min_implemented_gross_fraction": 0.80,
    },
}


def test_the_batch_is_validated_again_after_rounding() -> None:
    """Two independent validations, §3.7: one on the solved target, one on the batch."""
    index = _four()
    held = pd.Series(0.0, index=index)
    rows = profile_limits(gross=1.0, cap=0.5, net=0.02, budget=None, participation=None)
    x = _inputs_for(index, held, limits=rows)
    out = solve(x)
    assert out.diagnostics["validation"]["ok"] is True
    batch = finalize_batch(
        out,
        state={"cash": EQUITY, "positions": {}, "pending_orders": []},
        prices=x.reference_prices,
        execution_profile=ROUNDING_PROFILE,
        limits=rows,
        policy=RecoveryPolicy(),
        decision_bar="2026-06-01",
        adv_dollars=x.adv_dollars,
        risk=x.risk,
    )
    assert batch.validation["stage"] == "post_rounding"
    assert batch.intents, "the batch must contain the orders the target implies"
    assert batch.response in ("none", "scale_batch", "drop_batch")


def test_the_declared_rounding_and_obligation_responses_reach_the_batch(monkeypatch) -> None:
    """``on_rounding_failure`` and ``on_blocked_obligation`` are policy values, and the
    second validation must be run under the same ones the solve was."""
    from prism.portfolio import rounding as rounding_mod

    captured: dict = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        return "batch"

    monkeypatch.setattr(rounding_mod, "round_batch", fake)
    index = _four()
    held = pd.Series(0.0, index=index)
    x = _inputs_for(index, held, limits=profile_limits(budget=None, participation=None))
    out = solve(x)
    policy = RecoveryPolicy(on_rounding_failure="drop_batch", on_blocked_obligation="record")
    assert (
        finalize_batch(
            out,
            state={"cash": EQUITY, "positions": {}, "pending_orders": []},
            prices=x.reference_prices,
            execution_profile=ROUNDING_PROFILE,
            limits=list(x.limits),
            policy=policy,
            decision_bar="2026-06-01",
        )
        == "batch"
    )
    assert captured["on_rounding_failure"] == "drop_batch"
    assert captured["on_blocked_obligation"] == "record"
    assert captured["achieved_bounds"] is None or isinstance(captured["achieved_bounds"], dict)


def test_a_no_action_decision_is_never_rounded_into_a_batch() -> None:
    index = _four()
    held = pd.Series(0.0, index=index)
    x = _inputs_for(index, held, limits=profile_limits(budget=None, participation=None))
    out = solve(x)
    out.path = "no_action"
    out.diagnostics["no_action"] = no_action(RecoveryPolicy(), "nothing executable")
    with pytest.raises(RecoveryError, match="no-action path"):
        finalize_batch(
            out,
            state={"cash": EQUITY, "positions": {}, "pending_orders": []},
            prices=x.reference_prices,
            execution_profile=ROUNDING_PROFILE,
            limits=list(x.limits),
            policy=RecoveryPolicy(),
            decision_bar="2026-06-01",
        )


def test_a_target_that_failed_post_solve_validation_is_never_rounded() -> None:
    index = _four()
    held = pd.Series(0.0, index=index)
    x = _inputs_for(index, held, limits=profile_limits(budget=None, participation=None))
    out = solve(x)
    out.diagnostics["validation"] = dict(out.diagnostics["validation"], ok=False, binding_condition="max_gross")
    with pytest.raises(RecoveryError, match="never rounded"):
        finalize_batch(
            out,
            state={"cash": EQUITY, "positions": {}, "pending_orders": []},
            prices=x.reference_prices,
            execution_profile=ROUNDING_PROFILE,
            limits=list(x.limits),
            policy=RecoveryPolicy(),
            decision_bar="2026-06-01",
        )
