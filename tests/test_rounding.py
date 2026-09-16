"""Batch rounding: parity with the live sizer, repair, scaling, and termination.

Two things are being defended here.

**Parity.** The zero-crossing clamp and the dust rules moved out of
``live/loop.py::targets_to_orders`` (which is :func:`prism.assembly.size_orders`) into
``portfolio/rounding.py``. Every parity case below runs both implementations on the same
inputs and asserts the same quantities, because a relocation that changes a quantity is a
silent change to what the instrument trades.

**Termination.** The adversarial case is the one the spec names: validity is not monotone
in the scale factor, so a loop that shrinks until it fits can oscillate forever. The
schedule is walked once and the repair measure decreases by a fixed quantum, and both of
those are asserted rather than assumed -- including a run whose *only* outcome is a
dropped batch, which is the case an unbounded loop would never reach.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from prism.assembly import size_orders
from prism.portfolio import rounding as rounding_mod
from prism.portfolio.limits import Limit, LimitClass
from prism.portfolio.obligations import Executable, obligation_from_current
from prism.portfolio.rounding import RoundingError, RoundingSettings, round_batch, round_delta
from prism.portfolio.validate import ValidationSettings

PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0})

PROFILE = {
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


def _profile(**overrides) -> dict:
    profile = {key: (dict(value) if isinstance(value, dict) else value) for key, value in PROFILE.items()}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(profile.get(key), dict):
            profile[key].update(value)
        else:
            profile[key] = value
    return profile


def _state(cash: float, positions: dict[str, float] | None = None) -> dict:
    return {"cash": cash, "positions": dict(positions or {}), "pending_orders": []}


def _preference(name: str, priority: int, **params) -> Limit:
    return Limit(name=name, cls=LimitClass.PREFERENCE, priority=priority, params=params, source="test")


def _by_sid(output) -> dict[str, float]:
    return {intent.sid: intent.qty for intent in output.intents}


# ---------------------------------------------------------------------------
# The rounding rule itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [7.5, -7.5, 6.5, -6.5, 0.5, -0.5, 1.4999, -1.4999, 12.0])
def test_whole_share_rounding_is_float_round_ties_included(delta: float) -> None:
    # The live path is `float(round(delta))` (loop.py:142 via assembly.size_orders):
    # ties to even, so 7.5 -> 8 and 6.5 -> 6. A tie_rule that rounded half away from zero
    # would move a share on every half-share target.
    assert round_delta(delta, 1.0) == float(round(delta))


def test_a_continuous_increment_leaves_the_delta_alone() -> None:
    assert round_delta(7.5, 0.0) == 7.5


def test_a_fractional_increment_rounds_to_that_grid() -> None:
    assert round_delta(1.23456, 0.001) == pytest.approx(1.235)
    assert round_delta(-1.2345, 0.001) == pytest.approx(-1.234)  # tie to even


def test_an_undeclared_tie_rule_is_refused_rather_than_approximated() -> None:
    with pytest.raises(RoundingError, match="is not implemented"):
        RoundingSettings(tie_rule="half_away_from_zero")


def test_a_zero_repair_decrement_is_refused_because_it_breaks_termination() -> None:
    with pytest.raises(RoundingError, match="makes the repair loop terminate"):
        RoundingSettings(repair_min_decrement=0.0)


# ---------------------------------------------------------------------------
# Parity with the live sizer
# ---------------------------------------------------------------------------


def test_parity_targets_diff_against_held_shares() -> None:
    targets = pd.Series({"AAA": 0.5, "BBB": -0.2})
    positions = {"AAA": 30.0}
    output = round_batch(
        targets,
        _state(cash=7_000.0, positions=positions),
        PRICES,
        PROFILE,
        limits=(),
        decision_bar="2026-07-06",
    )
    legacy = size_orders(targets, positions, PRICES, 10_000.0, "2026-07-06", whole_shares=True)
    assert _by_sid(output) == {order.symbol: order.qty for order in legacy}
    assert _by_sid(output) == {"AAA": pytest.approx(20.0), "BBB": pytest.approx(-40.0)}
    assert output.response == "none" and output.scale_applied == 1.0


def test_parity_nan_target_holds_and_dust_is_dropped() -> None:
    targets = pd.Series({"AAA": float("nan"), "BBB": 0.0001})
    positions = {"AAA": 30.0}
    profile = _profile(min_notional=5.0)
    output = round_batch(
        targets,
        _state(cash=7_000.0, positions=positions),
        PRICES,
        profile,
        limits=(),
        decision_bar="2026-07-06",
    )
    legacy = size_orders(targets, positions, PRICES, 10_000.0, "2026-07-06", min_order_notional=5.0, whole_shares=True)
    assert legacy == []
    assert output.intents == ()
    # The NaN name is left alone entirely -- no record, no order, no residual.
    assert "AAA" not in output.quantities
    assert output.quantities["BBB"]["dropped_by"] == "zero_delta"


def test_parity_a_missing_price_for_a_non_flat_target_raises() -> None:
    targets = pd.Series({"ZZZ": 0.1})
    with pytest.raises(RoundingError, match="no usable decision price"):
        round_batch(targets, _state(cash=1_000.0), PRICES, PROFILE, limits=(), decision_bar="2026-07-06")
    with pytest.raises(ValueError, match="no usable decision price"):
        size_orders(targets, {}, PRICES, 1_000.0, "2026-07-06")


def test_parity_a_flat_target_on_an_unheld_name_needs_no_price() -> None:
    targets = pd.Series({"AAA": 0.1, "ZZZ": 0.0})
    output = round_batch(targets, _state(cash=10_000.0), PRICES, PROFILE, limits=(), decision_bar="2026-07-06")
    assert _by_sid(output) == {"AAA": pytest.approx(10.0)}


def test_parity_zero_crossing_clamp() -> None:
    # Held short 1, target flips long: a single order may only cover to flat. A venue
    # rejects a side-crossing order ("insufficient qty available", the 2026-07-08 ORCL
    # crash), so the opposite side opens next bar from a flat book.
    targets = pd.Series({"AAA": 0.5})
    positions = {"AAA": -1.0}
    output = round_batch(
        targets, _state(cash=10_100.0, positions=positions), PRICES, PROFILE, limits=(), decision_bar="2026-07-06"
    )
    legacy = size_orders(targets, positions, PRICES, 10_000.0, "2026-07-06", whole_shares=True)
    assert _by_sid(output) == {order.symbol: order.qty for order in legacy}
    assert _by_sid(output) == {"AAA": pytest.approx(1.0)}  # buy 1 to flat, not buy 51
    assert output.quantities["AAA"]["clamped"] is True


def test_parity_no_clamp_when_the_order_does_not_cross_zero() -> None:
    targets = pd.Series({"AAA": -0.5})
    positions = {"AAA": -1.0}
    output = round_batch(
        targets, _state(cash=10_100.0, positions=positions), PRICES, PROFILE, limits=(), decision_bar="2026-07-06"
    )
    legacy = size_orders(targets, positions, PRICES, 10_000.0, "2026-07-06", whole_shares=True)
    assert _by_sid(output) == {order.symbol: order.qty for order in legacy}
    assert _by_sid(output) == {"AAA": pytest.approx(-49.0)}
    assert output.quantities["AAA"]["clamped"] is False


def test_the_clamp_and_the_dust_rule_are_declared_options_not_conventions() -> None:
    targets = pd.Series({"AAA": 0.5})
    positions = {"AAA": -1.0}
    reproduction = RoundingSettings.from_profile(PROFILE).for_reproduction()
    output = round_batch(
        targets,
        _state(cash=10_100.0, positions=positions),
        PRICES,
        PROFILE,
        limits=(),
        decision_bar="2026-07-06",
        settings=reproduction,
    )
    # historical_constant_weight: no clamp, no dust rule, a continuous increment -- which
    # is what backtest_target_weights did, and reproduction is its only use.
    assert _by_sid(output)["AAA"] == pytest.approx(51.0)
    assert reproduction.zero_crossing_clamp is False and reproduction.apply_dust_rule is False


def test_a_dust_drop_whose_intent_was_economically_real_is_recorded() -> None:
    # A $2,000 name can never enter a $1,000-per-name book: the censored order is a
    # reproducible book/target divergence, so it is recorded rather than swallowed.
    prices = pd.Series({"EXP": 2_000.0})
    output = round_batch(
        pd.Series({"EXP": 0.10}),
        _state(cash=10_000.0),
        prices,
        _profile(min_notional=1.0),
        limits=(),
        decision_bar="2026-07-06",
    )
    assert output.intents == ()
    assert output.dropped[0]["sid"] == "EXP"
    assert output.dropped[0]["intended_notional"] == pytest.approx(1_000.0)


# ---------------------------------------------------------------------------
# Increments, fractional handling, and off-increment holdings
# ---------------------------------------------------------------------------


def test_the_increment_comes_from_the_capabilities_row_only_when_the_profile_allows_fractional() -> None:
    capabilities = pd.DataFrame(
        [
            {
                "sid": "AAA",
                "fractionable": True,
                "min_notional": 1.0,
                "shortable": True,
                "easy_to_borrow": True,
                "restrictions": "[]",
                "order_types": '["market_on_open", "market_day"]',
                "sessions": '["regular"]',
            }
        ]
    ).set_index("sid")
    targets = pd.Series({"AAA": 0.1234})
    whole = round_batch(
        targets,
        _state(cash=10_000.0),
        PRICES,
        PROFILE,
        limits=(),
        decision_bar="2026-07-06",
        capabilities=capabilities,
    )
    assert _by_sid(whole)["AAA"] == pytest.approx(12.0)  # [fractional] allowed = false wins
    fractional_profile = _profile(
        fractional={"allowed": True, "time_in_force": ["day"], "short_allowed": False},
        order_types=["market_on_open", "market_day"],
    )
    fractional = round_batch(
        targets,
        _state(cash=10_000.0),
        PRICES,
        fractional_profile,
        limits=(),
        decision_bar="2026-07-06",
        capabilities=capabilities,
        order_type="market_day",
    )
    assert _by_sid(fractional)["AAA"] == pytest.approx(12.34)


def test_an_off_increment_holding_leaves_a_recorded_residual() -> None:
    # The fractional short the Alpaca fixture carries (tests/test_live_alpaca.py:102) is
    # the real case: under [fractional] allowed = false no executable action closes it, so
    # leave_residual records it as an operator item rather than inventing an obligation.
    output = round_batch(
        pd.Series({"AAA": 0.0}),
        _state(cash=10_000.0, positions={"AAA": 10.5}),
        PRICES,
        PROFILE,
        limits=(),
        decision_bar="2026-07-06",
    )
    record = output.quantities["AAA"]
    assert record["off_increment_holding"] is True
    assert record["rounded_qty"] == pytest.approx(-10.0)  # -10.5 ties to even
    assert record["residual"] == pytest.approx(-0.5)
    assert _by_sid(output) == {"AAA": pytest.approx(-10.0)}


def test_close_fractional_without_a_capability_that_allows_it_raises() -> None:
    profile = _profile(rounding={"off_increment_holding": "close_fractional"})
    with pytest.raises(RoundingError, match="needs a capabilities row"):
        round_batch(
            pd.Series({"AAA": 0.0}),
            _state(cash=10_000.0, positions={"AAA": 10.5}),
            PRICES,
            profile,
            limits=(),
            decision_bar="2026-07-06",
        )


# ---------------------------------------------------------------------------
# The adversarial scaling case: validity is not monotone in the scale
# ---------------------------------------------------------------------------

ADVERSARIAL_PRICES = pd.Series({"A": 320.0, "B": 45.0, "C": 12.0, "D": 180.0})
ADVERSARIAL_TARGETS = pd.Series({"A": 0.24, "B": -0.26, "C": 0.25, "D": -0.23})


def _net_at(scale: float) -> float:
    """Net exposure of the whole-share batch sized at ``scale`` from a flat $10,000 book."""
    total = 0.0
    for sid, weight in ADVERSARIAL_TARGETS.items():
        price = float(ADVERSARIAL_PRICES[sid])
        total += round_delta(weight * scale * 10_000.0 / price, 1.0) * price / 10_000.0
    return total


def test_validity_is_not_monotone_in_the_scale_factor() -> None:
    # Spec §3.7, Run: the reason scaling walks a declared schedule instead of searching
    # for the largest valid scale. Against a 0.01 net band the batch is invalid at 1.00,
    # valid at 0.90 through 0.70, and invalid again at 0.60 and 0.50.
    observed = {scale: _net_at(scale) for scale in (1.00, 0.90, 0.80, 0.70, 0.60, 0.50)}
    assert observed[1.00] == pytest.approx(0.0106)
    assert observed[0.90] == pytest.approx(-0.0004)
    assert observed[0.80] == pytest.approx(0.0054)
    assert observed[0.70] == pytest.approx(-0.0068)
    assert observed[0.60] == pytest.approx(-0.0235)
    assert observed[0.50] == pytest.approx(0.0143)
    valid = {scale: abs(net) <= 0.01 for scale, net in observed.items()}
    assert list(valid.values()) == [False, True, True, True, False, False]


def _adversarial_output(**kwargs):
    profile = _profile(
        rounding={"scale_schedule": [0.90, 0.80, 0.70, 0.60, 0.50], "repair_max_moves": 0},
        validation={"mode": "blocking"},
    )
    settings = ValidationSettings.from_profile(profile, recovery_policy={"preference_response": "block"})
    return round_batch(
        ADVERSARIAL_TARGETS,
        _state(cash=10_000.0),
        ADVERSARIAL_PRICES,
        profile,
        limits=[_preference("net_band", 1, value=0.01)],
        decision_bar="2026-07-06",
        validation=settings,
        **kwargs,
    )


def test_the_schedule_takes_the_first_entry_that_validates_the_whole_batch() -> None:
    output = _adversarial_output()
    assert output.response == "scale_batch"
    assert output.scale_applied == pytest.approx(0.90)
    assert _by_sid(output) == {
        "A": pytest.approx(7.0),
        "B": pytest.approx(-52.0),
        "C": pytest.approx(188.0),
        "D": pytest.approx(-12.0),
    }
    assert output.validation["ok"] is True


def test_the_schedule_is_walked_once_and_never_revisited() -> None:
    # The termination measure for scaling is the lexicographic pair (remaining schedule
    # entries, number of intents). Bounding the *validations* is how that measure is
    # observed from outside: one pass per schedule entry, plus one per dust removal.
    calls: list[int] = []
    real = rounding_mod.validate_batch

    def counting(intents, **kwargs):
        calls.append(len(intents))
        return real(intents, **kwargs)

    rounding_mod.validate_batch = counting  # type: ignore[assignment]
    try:
        output = _adversarial_output()
    finally:
        rounding_mod.validate_batch = real  # type: ignore[assignment]
    # scale 1.00 fails, 0.90 validates: two passes, no dust removals, no re-entry.
    assert calls == [4, 4]
    assert output.scale_applied == pytest.approx(0.90)


def test_a_batch_no_schedule_entry_validates_is_dropped_whole_never_partially() -> None:
    # A net band no scale can satisfy: the balanced trade set survives or dies together.
    profile = _profile(rounding={"scale_schedule": [0.90, 0.80, 0.70, 0.60, 0.50], "repair_max_moves": 0})
    settings = ValidationSettings.from_profile(profile, recovery_policy={"preference_response": "block"})
    output = round_batch(
        ADVERSARIAL_TARGETS,
        _state(cash=10_000.0),
        ADVERSARIAL_PRICES,
        profile,
        limits=[_preference("net_band", 1, value=0.0001)],
        decision_bar="2026-07-06",
        validation=settings,
    )
    assert output.response == "drop_batch"
    assert output.intents == ()  # not one surviving leg of the balanced set
    assert output.not_implementable is True
    assert output.binding_condition == "net_band"


def test_drop_batch_skips_the_schedule_entirely() -> None:
    output = _adversarial_output(on_rounding_failure="drop_batch")
    assert output.response == "drop_batch" and output.intents == ()


# ---------------------------------------------------------------------------
# Repair: the decreasing measure
# ---------------------------------------------------------------------------


def test_single_increment_repair_cures_the_band_and_records_each_move() -> None:
    profile = _profile(rounding={"scale_schedule": [0.90, 0.80], "repair_max_moves": 200})
    settings = ValidationSettings.from_profile(profile, recovery_policy={"preference_response": "block"})
    output = round_batch(
        ADVERSARIAL_TARGETS,
        _state(cash=10_000.0),
        ADVERSARIAL_PRICES,
        profile,
        limits=[_preference("net_band", 1, value=0.01)],
        decision_bar="2026-07-06",
        validation=settings,
    )
    assert output.response == "none" and output.scale_applied == 1.0
    assert output.repair_moves, "the batch was invalid at scale 1.0, so a repair move must be recorded"
    # V is a real measure that strictly decreases by at least repair_min_decrement per
    # accepted move; that, and V >= 0, is the whole termination argument.
    decrement = RoundingSettings.from_profile(profile).repair_min_decrement
    for move in output.repair_moves:
        assert move.v_after <= move.v_before - decrement
        assert move.v_after >= 0.0
    v0 = output.repair_moves[0].v_before
    assert len(output.repair_moves) <= math.ceil(v0 / decrement)
    assert abs(sum(intent.qty * ADVERSARIAL_PRICES[intent.sid] for intent in output.intents) / 10_000.0) <= 0.01


def test_repair_never_crosses_zero_and_never_creates_dust() -> None:
    # Both are action restrictions, and an action restriction is never waived -- not even
    # to reduce a preference violation.
    profile = _profile(min_notional=50.0, rounding={"scale_schedule": [0.5], "repair_max_moves": 200})
    settings = ValidationSettings.from_profile(profile, recovery_policy={"preference_response": "block"})
    output = round_batch(
        pd.Series({"AAA": 0.02, "BBB": -0.5}),
        _state(cash=10_000.0, positions={"AAA": 1.0}),
        PRICES,
        profile,
        limits=[_preference("net_band", 1, value=0.001)],
        decision_bar="2026-07-06",
        validation=settings,
    )
    for intent in output.intents:
        held = 1.0 if intent.sid == "AAA" else 0.0
        ends_at = held + intent.qty
        assert held == 0.0 or ends_at == 0.0 or (held > 0.0) == (ends_at > 0.0)
        assert abs(intent.qty) * intent.reference_price >= 50.0


def test_repair_can_be_disarmed_and_the_schedule_still_terminates() -> None:
    output = _adversarial_output()
    assert output.repair_moves == ()


# ---------------------------------------------------------------------------
# Obligations inside a batch
# ---------------------------------------------------------------------------


def test_an_obligation_leg_is_exempt_from_the_dust_rule_and_from_scaling() -> None:
    # A recall on a two-share short: the leg is smaller than the minimum notional, and it
    # is still submitted, because an obligation is not a preference.
    prices = pd.Series({"AAA": 100.0, "REC": 3.0})
    obligation = obligation_from_current("REC", "cover", current_qty=-2.0, source="recall", priority=1)
    profile = _profile(min_notional=50.0)
    output = round_batch(
        pd.Series({"AAA": 0.5, "REC": 0.0}),
        _state(cash=10_006.0, positions={"REC": -2.0}),
        prices,
        profile,
        limits=(),
        obligations=[obligation],
        decision_bar="2026-07-06",
        executable={"REC": Executable(sid="REC", max_reduce_qty=2.0, increment=1.0)},
    )
    assert _by_sid(output)["REC"] == pytest.approx(2.0)
    assert output.obligations["legs"][0]["progress"] == pytest.approx(2.0)
    intent = next(item for item in output.intents if item.sid == "REC")
    assert intent.action_class == "OB"


def test_a_blocked_obligation_is_recorded_with_its_reason_and_escalated() -> None:
    obligation = obligation_from_current("AAA", "cover", current_qty=-10.0, source="recall", priority=1)
    output = round_batch(
        pd.Series({"AAA": 0.0}),
        _state(cash=11_000.0, positions={"AAA": -10.0}),
        PRICES,
        PROFILE,
        limits=(),
        obligations=[obligation],
        decision_bar="2026-07-06",
        executable={"AAA": Executable(sid="AAA", max_reduce_qty=0.0, reason="halted")},
    )
    assert output.obligations["blocked"][0]["reason"] == "halted"
    assert output.obligations["escalations"] == ["AAA: halted"]


# ---------------------------------------------------------------------------
# Whole-batch validation of the implied portfolio
# ---------------------------------------------------------------------------


def test_the_batch_is_validated_against_the_portfolio_it_implies_including_pending_orders() -> None:
    # A pending buy has already promised the cash; a batch validated against cash it has
    # already committed would be submitted and then rejected for funds.
    state = {
        "cash": 10_000.0,
        "positions": {},
        "pending_orders": [
            {"qty": 90.0, "reference_price": 100.0, "filled_qty": 0.0},
        ],
    }

    class _Pending:
        def __init__(self, qty, price):
            self.qty = qty
            self.reference_price = price
            self.filled_qty = 0.0

    state["pending_orders"] = [_Pending(90.0, 100.0)]
    output = round_batch(
        pd.Series({"AAA": 0.5}),
        state,
        PRICES,
        PROFILE,
        limits=(),
        decision_bar="2026-07-06",
    )
    assert output.response == "drop_batch"
    assert output.validation["cash"]["committed_cash"] == pytest.approx(9_000.0)


def test_an_unmarked_holding_refuses_rather_than_being_valued_at_zero() -> None:
    with pytest.raises(RoundingError, match="never erase a holding"):
        round_batch(
            pd.Series({"AAA": 0.5}),
            _state(cash=1_000.0, positions={"GONE": 10.0}),
            PRICES,
            PROFILE,
            limits=(),
            decision_bar="2026-07-06",
        )


# ---------------------------------------------------------------------------
# The adversarial termination case: dust removal interacting with the schedule
# ---------------------------------------------------------------------------

TERMINATION_PRICES = pd.Series({s: 100.0 for s in ("A", "B", "C", "D", "E", "F")})
TERMINATION_TARGETS = pd.Series({"A": 0.06, "B": -0.06, "C": 0.06, "D": -0.06, "E": 0.06, "F": 0.30})


def _venue_minimum(min_notional: float) -> pd.DataFrame:
    rows = [
        {
            "sid": sid,
            "fractionable": False,
            "min_notional": min_notional,
            "shortable": True,
            "easy_to_borrow": True,
            "restrictions": "[]",
            "order_types": '["market_on_open"]',
            "sessions": '["regular"]',
        }
        for sid in TERMINATION_PRICES.index
    ]
    return pd.DataFrame(rows).set_index("sid")


def test_dust_removal_at_a_scale_never_advances_or_restarts_the_schedule() -> None:
    # The oscillation this case is built to provoke: scaling down pushes five of the six
    # orders under the venue's minimum notional, so each scale removes orders, and a loop
    # that re-entered the schedule after a removal -- or restarted it -- would never stop.
    # Here the removal is a *removal*: the batch is revalidated at the same scale, the
    # intent count strictly decreases, and the schedule advances only on a whole-batch
    # verdict. The net band is unsatisfiable at every scale, so the walk runs to the end
    # and the batch is dropped, which is the outcome a non-terminating loop never reaches.
    profile = _profile(
        min_notional=1.0,
        rounding={"scale_schedule": [0.75, 0.50, 0.25], "repair_max_moves": 0},
    )
    settings = ValidationSettings.from_profile(profile, recovery_policy={"preference_response": "block"})
    calls: list[int] = []
    real = rounding_mod.validate_batch

    def counting(intents, **kwargs):
        calls.append(len(intents))
        return real(intents, **kwargs)

    rounding_mod.validate_batch = counting  # type: ignore[assignment]
    try:
        output = round_batch(
            TERMINATION_TARGETS,
            _state(cash=10_000.0),
            TERMINATION_PRICES,
            profile,
            limits=[_preference("net_band", 1, value=0.001)],
            decision_bar="2026-07-06",
            validation=settings,
            capabilities=_venue_minimum(500.0),
        )
    finally:
        rounding_mod.validate_batch = real  # type: ignore[assignment]

    assert output.response == "drop_batch" and output.intents == ()
    # scale 1.00: six orders, all above the venue minimum, the band fails -> advance.
    # each later scale: six orders, five fall under the minimum and are removed, the
    # surviving one is revalidated at that same scale, the band still fails -> advance.
    assert calls == [6, 6, 1, 6, 1, 6, 1]
    # The declared bound the measure gives: one pass per schedule entry (plus the initial
    # one), and at most one further pass per removable intent inside each.
    schedule_entries = 1 + len(profile["rounding"]["scale_schedule"])
    assert len(calls) <= schedule_entries * (len(TERMINATION_TARGETS) + 1)
    # Inside a scale, a repeat pass is strictly smaller than the one that provoked it --
    # that is the decreasing half of the measure. A pass never grows the batch back.
    assert max(calls) <= len(TERMINATION_TARGETS)
    assert [calls[i + 1] for i in (1, 3, 5)] == [1, 1, 1]


def test_repair_stops_at_the_declared_move_cap_without_oscillating() -> None:
    profile = _profile(rounding={"scale_schedule": [0.75], "repair_max_moves": 3, "repair_min_decrement": 1.0e-6})
    settings = ValidationSettings.from_profile(profile, recovery_policy={"preference_response": "block"})
    output = round_batch(
        TERMINATION_TARGETS,
        _state(cash=10_000.0),
        TERMINATION_PRICES,
        profile,
        limits=[_preference("net_band", 1, value=0.001)],
        decision_bar="2026-07-06",
        validation=settings,
        capabilities=_venue_minimum(1.0),
    )
    assert len(output.repair_moves) <= 3
    values = [move.v_before for move in output.repair_moves] + [
        output.repair_moves[-1].v_after if output.repair_moves else 0.0
    ]
    # Strictly decreasing by at least the declared quantum: the measure that terminates.
    assert all(later <= earlier - 1.0e-6 for earlier, later in zip(values, values[1:]))
    assert all(value >= 0.0 for value in values)


# ---------------------------------------------------------------------------
# The canonical portfolio state
# ---------------------------------------------------------------------------


def test_round_batch_reads_the_canonical_portfolio_state() -> None:
    # The §3.6 state is the real input: shares with their marks, cash as a real balance,
    # and equity derived from them rather than passed in beside them.
    from prism.execution.portfolio_state import state_from_shares

    state = state_from_shares("2026-07-06", {"AAA": 30.0}, PRICES, cash=7_000.0)
    assert state.equity(PRICES) == pytest.approx(10_000.0)
    output = round_batch(
        pd.Series({"AAA": 0.5, "BBB": -0.2}),
        state,
        PRICES,
        PROFILE,
        limits=(),
        decision_bar="2026-07-06",
    )
    assert _by_sid(output) == {"AAA": pytest.approx(20.0), "BBB": pytest.approx(-40.0)}
    # And the emitted intents are the canonical OrderIntent, not the local fallback.
    from prism.execution.portfolio_state import OrderIntent

    assert all(isinstance(intent, OrderIntent) for intent in output.intents)


# ---------------------------------------------------------------------------
# The shipped reference files
# ---------------------------------------------------------------------------


def test_the_shipped_execution_profile_and_limit_rows_load_as_declared() -> None:
    # profiles/execution_synthetic.toml and the two profile templates are what `prism init`
    # writes. A [rounding] or [[limits]] row this module cannot read is a broken project.
    import tomllib

    from prism.portfolio.limits import load_limits
    from prism.template import ETF_SMALL_PROFILE_TOML, EXECUTION_SYNTHETIC_TOML, PAPER_PROFILE_TOML

    profile = tomllib.loads(EXECUTION_SYNTHETIC_TOML)
    settings = RoundingSettings.from_profile(profile)
    assert settings.scale_schedule == (0.75, 0.5, 0.25)
    assert settings.tie_rule == "half_to_even" and settings.zero_crossing_clamp is True
    assert settings.quantity_increment_default == 1.0 and settings.min_notional == 1.0
    validation = ValidationSettings.from_profile(profile)
    assert validation.mode == "blocking" and validation.limit_tolerance == 1e-6

    paper, ignored = load_limits(tomllib.loads(PAPER_PROFILE_TOML)["limits"], source="profiles/paper.toml")
    assert [limit.name for limit in paper] == ["max_gross", "max_name_weight"]
    assert ignored == ()
    etf, etf_ignored = load_limits(tomllib.loads(ETF_SMALL_PROFILE_TOML)["limits"], source="profiles/etf_small.toml")
    assert [limit.name for limit in etf] == [
        "max_gross",
        "max_name_weight",
        "participation_per_name",
        "quantity_increment",
    ]
    # The two physical rows write `priority = 0`; both are recorded as ignored, not refused.
    assert len(etf_ignored) == 2
