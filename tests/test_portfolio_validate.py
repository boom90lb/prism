"""Independent validation: the accept condition, and the things a solver may not excuse.

The defect this file is written against is the one that costs money quietly: a batch that
was authorized because something upstream reported success. :func:`validate_batch` takes
no solver argument at all, and the cases below check the accept condition term by term --
including the two that are *not* failures, because a validator that rejected every
recovery output would be turned off within a week.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prism.portfolio.limits import Limit, LimitClass, LimitContext
from prism.portfolio.validate import (
    ValidationError,
    ValidationSettings,
    capability_row,
    quantity_increment_for,
    validate_batch,
    validate_targets,
)

PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0})

PROFILE = {
    "quantity_increment_default": 1.0,
    "min_notional": 1.0,
    "order_types": ["market_on_open", "market_day"],
    "fractional": {"allowed": False, "time_in_force": ["day"], "short_allowed": False},
    "validation": {"limit_tolerance": 1e-6, "cash_tolerance": 0.01, "cost_tolerance_bps": 0.1, "mode": "blocking"},
    "rounding": {"fractional_increment": 0.001},
}


class Intent:
    """A structural stand-in for ``OrderIntent``; validation reads it by attribute."""

    def __init__(self, sid, qty, price=100.0, increment=1.0, order_type="market_on_open", tif="opg"):
        self.sid = sid
        self.qty = qty
        self.reference_price = price
        self.increment = increment
        self.decision_bar = "2026-09-08"
        self.order_type = order_type
        self.time_in_force = tif
        self.action_class = "C3"


def _limit(name, cls, priority=None, **params) -> Limit:
    return Limit(name=name, cls=LimitClass(cls), priority=priority, params=params, source="test")


def _ctx(w0=None, equity=10_000.0, **kwargs) -> LimitContext:
    return LimitContext(w0=pd.Series(w0 or {}, dtype=float), equity=equity, **kwargs)


def _caps(**overrides) -> pd.DataFrame:
    row = {
        "sid": "AAA",
        "fractionable": False,
        "quantity_increment": 1.0,
        "min_notional": 1.0,
        "shortable": True,
        "easy_to_borrow": True,
        "lot_size": 100,
        "auction_eligible": True,
        "restrictions": "[]",
        "order_types": '["market_on_open", "market_day"]',
        "sessions": '["regular"]',
    }
    row.update(overrides)
    return pd.DataFrame([row]).set_index("sid")


# ---------------------------------------------------------------------------
# Solver status never authorizes an order
# ---------------------------------------------------------------------------


def test_validate_batch_accepts_no_solver_status_at_all() -> None:
    import inspect

    parameters = set(inspect.signature(validate_batch).parameters)
    assert not parameters & {"solver", "solver_status", "status", "construction"}


def test_a_batch_from_an_optimal_solve_still_fails_on_the_numbers() -> None:
    # The "solve" reported optimal; the batch ends the account short a name the
    # capabilities row marks not shortable. The report is a refusal.
    report = validate_batch(
        [Intent("AAA", -10.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        capabilities=_caps(shortable=False),
        cash=10_000.0,
    )
    assert report.blocked
    assert {finding.check for finding in report.fatal_findings} == {"shortable"}


# ---------------------------------------------------------------------------
# The accept condition, term by term
# ---------------------------------------------------------------------------


def test_an_introduced_action_restriction_violation_always_fails() -> None:
    ctx = _ctx({"AAA": 0.0}, equity=1_000_000.0, adv_dollars=pd.Series({"AAA": 100_000.0}))
    limit = _limit("participation_per_name", "physical", None, value=0.05)
    for response in ("record", "block"):
        report = validate_batch(
            [Intent("AAA", 1_000.0)],
            limits=[limit],
            ctx=ctx,
            prices=PRICES,
            settings=ValidationSettings(preference_response=response),
            execution_profile=PROFILE,
            cash=1_000_000.0,
        )
        assert report.blocked, f"a physical restriction must fail under preference_response={response!r}"


def test_an_introduced_preference_violation_is_recorded_or_blocked_by_the_declared_response() -> None:
    ctx = _ctx({"AAA": 0.0})
    limit = _limit("max_gross", "preference", 1, value=0.05)
    intents = [Intent("AAA", 10.0)]  # 1000 dollars of a 10,000 book: gross 0.10
    recorded = validate_batch(
        intents,
        limits=[limit],
        ctx=ctx,
        prices=PRICES,
        settings=ValidationSettings(preference_response="record"),
        execution_profile=PROFILE,
        cash=10_000.0,
    )
    assert recorded.ok
    assert [entry["limit_name"] for entry in recorded.violations["introduced"]] == ["max_gross"]
    blocked = validate_batch(
        intents,
        limits=[limit],
        ctx=ctx,
        prices=PRICES,
        settings=ValidationSettings(preference_response="block"),
        execution_profile=PROFILE,
        cash=10_000.0,
    )
    assert blocked.blocked and blocked.binding_condition == "max_gross"


def test_an_inherited_violation_not_made_worse_never_fails() -> None:
    # The held book already breaches max_gross; the batch reduces it and still breaches.
    # Neither repair can cure that, so rejecting it would reject the normal recovery case.
    ctx = _ctx({"AAA": 1.5})
    limit = _limit("max_gross", "preference", 1, value=1.0)
    report = validate_batch(
        [Intent("AAA", -20.0)],
        limits=[limit],
        ctx=ctx,
        prices=PRICES,
        settings=ValidationSettings(preference_response="block"),
        execution_profile=PROFILE,
        cash=10_000.0,
        positions={"AAA": 150.0},
    )
    assert report.ok
    assert report.violations["existing"][0]["v"] == pytest.approx(0.5)
    assert report.violations["remaining"][0]["v"] == pytest.approx(0.3)
    assert not report.violations["introduced"]


def test_an_inherited_violation_made_worse_fails_under_block() -> None:
    ctx = _ctx({"AAA": 1.5})
    limit = _limit("max_gross", "preference", 1, value=1.0)
    report = validate_batch(
        [Intent("AAA", 20.0)],
        limits=[limit],
        ctx=ctx,
        prices=PRICES,
        settings=ValidationSettings(preference_response="block"),
        execution_profile=PROFILE,
        cash=10_000.0,
        positions={"AAA": 150.0},
    )
    assert report.blocked
    assert {finding.check for finding in report.fatal_findings} == {"inherited_violation_worsened"}


def test_a_violation_beyond_a_bound_r1_achieved_fails() -> None:
    ctx = _ctx({"AAA": 1.5})
    limit = _limit("max_gross", "preference", 1, value=1.0)
    # R1 achieved 0.30 on max_gross; a target measuring 0.50 exceeds it by more than the slack.
    report = validate_targets(
        pd.Series({"AAA": 1.5}),
        [limit],
        ctx,
        settings=ValidationSettings(),
        achieved_bounds={"max_gross": 0.30},
    )
    assert report.blocked
    assert report.fatal_findings[0].check == "achieved_bound"
    within = validate_targets(
        pd.Series({"AAA": 1.3}), [limit], ctx, settings=ValidationSettings(), achieved_bounds={"max_gross": 0.30}
    )
    assert within.ok


def test_report_only_softens_the_verdict_and_records_every_finding() -> None:
    ctx = _ctx()
    report = validate_batch(
        [Intent("AAA", -10.0)],
        limits=(),
        ctx=ctx,
        prices=PRICES,
        settings=ValidationSettings(mode="report_only"),
        execution_profile=PROFILE,
        capabilities=_caps(shortable=False),
        cash=10_000.0,
    )
    assert report.ok  # the parity gate submits; the finding is still on the record
    assert [finding.check for finding in report.findings] == ["shortable"]
    assert report.to_dict()["findings"][0]["fatal"] is True


# ---------------------------------------------------------------------------
# Finiteness
# ---------------------------------------------------------------------------


def test_a_non_finite_target_fails_before_any_limit_is_measured() -> None:
    report = validate_targets(pd.Series({"AAA": float("nan")}), [], _ctx(), settings=ValidationSettings())
    assert report.blocked and report.fatal_findings[0].check == "finiteness"


@pytest.mark.parametrize("intent", [Intent("AAA", float("nan")), Intent("AAA", 0.0), Intent("AAA", 10.0, price=0.0)])
def test_a_non_finite_or_empty_order_fails(intent) -> None:
    report = validate_batch(
        [intent],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_000.0,
    )
    assert report.blocked and report.fatal_findings[0].check == "finiteness"


def test_an_order_for_a_name_with_no_price_fails_rather_than_marking_it_at_zero() -> None:
    report = validate_batch(
        [Intent("ZZZ", 10.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_000.0,
    )
    assert report.blocked and "no usable price" in report.fatal_findings[0].detail


# ---------------------------------------------------------------------------
# Per-order action restrictions
# ---------------------------------------------------------------------------


def test_a_quantity_off_the_increment_fails() -> None:
    report = validate_batch(
        [Intent("AAA", 10.5)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        capabilities=_caps(),
        cash=10_000.0,
    )
    assert report.blocked and report.fatal_findings[0].check == "quantity_increment"


def test_the_increment_is_derived_and_lot_size_is_never_it() -> None:
    # lot_size 100 is a round-lot classification; the increment stays 1.0.
    assert quantity_increment_for("AAA", execution_profile=PROFILE, capabilities=_caps(lot_size=100)) == 1.0
    fractional = dict(PROFILE, fractional={"allowed": True, "time_in_force": ["day"], "short_allowed": False})
    assert quantity_increment_for("AAA", execution_profile=fractional, capabilities=_caps(fractionable=True)) == 0.001
    # Not fractionable at the venue: the profile's permission alone is not enough.
    assert quantity_increment_for("AAA", execution_profile=fractional, capabilities=_caps()) == 1.0


def test_an_order_below_the_minimum_notional_fails() -> None:
    report = validate_batch(
        [Intent("AAA", 1.0, price=0.5)],
        limits=(),
        ctx=_ctx(),
        prices=pd.Series({"AAA": 0.5}),
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        capabilities=_caps(),
        cash=10_000.0,
    )
    assert report.blocked and any(finding.check == "min_notional" for finding in report.fatal_findings)


def test_an_obligation_leg_is_exempt_from_the_policy_floor_but_not_the_venues() -> None:
    profile = dict(PROFILE, min_notional=50.0)
    exempt = validate_batch(
        [Intent("AAA", 1.0, price=6.0)],
        limits=(),
        ctx=_ctx(),
        prices=pd.Series({"AAA": 6.0}),
        settings=ValidationSettings(),
        execution_profile=profile,
        cash=10_000.0,
        exempt_sids=["AAA"],
    )
    assert exempt.ok
    venue = validate_batch(
        [Intent("AAA", 1.0, price=6.0)],
        limits=(),
        ctx=_ctx(),
        prices=pd.Series({"AAA": 6.0}),
        settings=ValidationSettings(),
        execution_profile=profile,
        capabilities=_caps(min_notional=10.0),
        cash=10_000.0,
        exempt_sids=["AAA"],
    )
    assert venue.blocked and any(finding.check == "min_notional" for finding in venue.fatal_findings)


def test_an_order_type_outside_the_profile_or_the_capabilities_row_fails() -> None:
    outside_profile = validate_batch(
        [Intent("AAA", 10.0, order_type="limit", tif="day")],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_000.0,
    )
    assert any(finding.check == "order_type" for finding in outside_profile.fatal_findings)
    outside_row = validate_batch(
        [Intent("AAA", 10.0, order_type="market_day", tif="day")],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        capabilities=_caps(order_types='["market_on_open"]'),
        cash=10_000.0,
    )
    assert any(finding.check == "order_type" for finding in outside_row.fatal_findings)


def test_a_fractional_order_may_never_open_a_short() -> None:
    profile = dict(PROFILE, fractional={"allowed": True, "time_in_force": ["day"], "short_allowed": False})
    report = validate_batch(
        [Intent("AAA", -10.5, increment=0.001, order_type="market_day", tif="day")],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=profile,
        capabilities=_caps(fractionable=True),
        cash=10_000.0,
    )
    assert any(finding.check == "fractional_short" for finding in report.fatal_findings)


def test_no_new_short_without_borrow() -> None:
    report = validate_batch(
        [Intent("AAA", -10.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        capabilities=_caps(easy_to_borrow=False),
        cash=10_000.0,
    )
    assert any(finding.check == "borrow" for finding in report.fatal_findings)
    # Covering an existing short needs no borrow.
    covering = validate_batch(
        [Intent("AAA", 5.0)],
        limits=(),
        ctx=_ctx({"AAA": -0.1}),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        capabilities=_caps(easy_to_borrow=False),
        cash=10_000.0,
        positions={"AAA": -10.0},
    )
    assert covering.ok


def test_a_halted_name_may_not_be_traded_and_a_restricted_one_only_reduced() -> None:
    halted = validate_batch(
        [Intent("AAA", -5.0)],
        limits=(),
        ctx=_ctx({"AAA": 0.1}),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        capabilities=_caps(restrictions='["halted"]'),
        cash=10_000.0,
        positions={"AAA": 10.0},
    )
    assert any(finding.check == "halted" for finding in halted.fatal_findings)
    adding = validate_batch(
        [Intent("AAA", 5.0)],
        limits=(),
        ctx=_ctx({"AAA": 0.1}),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        capabilities=_caps(restrictions='["no_new_exposure"]'),
        cash=10_000.0,
        positions={"AAA": 10.0},
    )
    assert any(finding.check == "reduce_only" for finding in adding.fatal_findings)
    reducing = validate_batch(
        [Intent("AAA", -5.0)],
        limits=(),
        ctx=_ctx({"AAA": 0.1}),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        capabilities=_caps(restrictions='["no_new_exposure"]'),
        cash=10_000.0,
        positions={"AAA": 10.0},
    )
    assert reducing.ok


def test_a_missing_capabilities_row_is_an_unchecked_note_or_a_failure_by_declaration() -> None:
    note = validate_batch(
        [Intent("AAA", 5.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_000.0,
    )
    assert note.ok and any("no dated capabilities row" in text for text in note.unchecked)
    required = validate_batch(
        [Intent("AAA", 5.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(require_capabilities=True),
        execution_profile=PROFILE,
        cash=10_000.0,
    )
    assert required.blocked and required.fatal_findings[0].check == "capabilities"


def test_two_orders_for_one_sid_in_one_batch_fail() -> None:
    report = validate_batch(
        [Intent("AAA", 5.0), Intent("AAA", 5.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_000.0,
    )
    assert any(finding.check == "duplicate_order" for finding in report.fatal_findings)


def test_a_side_crossing_order_fails_and_the_check_is_switchable_for_reproduction() -> None:
    crossing = validate_batch(
        [Intent("AAA", 51.0)],
        limits=(),
        ctx=_ctx({"AAA": -0.01}),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_100.0,
        positions={"AAA": -1.0},
    )
    assert any(finding.check == "zero_crossing" for finding in crossing.fatal_findings)
    reproduction = validate_batch(
        [Intent("AAA", 51.0)],
        limits=(),
        ctx=_ctx({"AAA": -0.01}),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_100.0,
        positions={"AAA": -1.0},
        check_zero_crossing=False,
    )
    assert reproduction.ok


# ---------------------------------------------------------------------------
# Cash
# ---------------------------------------------------------------------------


def test_a_pending_buy_reduces_available_funds_before_its_fill() -> None:
    report = validate_batch(
        [Intent("AAA", 20.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_000.0,
        committed_cash=9_000.0,
    )
    assert report.blocked and report.fatal_findings[0].check == "cash"
    assert report.cash["buying_power"] == pytest.approx(-1_000.0)


def test_short_proceeds_are_held_as_collateral_not_as_buying_power() -> None:
    # collateral_no_rebate: shorting 5,000 dollars of stock buys nothing.
    report = validate_batch(
        [Intent("AAA", -50.0), Intent("BBB", 120.0, price=50.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=5_000.0,
    )
    assert report.blocked and report.fatal_findings[0].check == "cash"


def test_the_charged_cost_comes_out_of_cash() -> None:
    report = validate_batch(
        [Intent("AAA", 100.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(cash_tolerance=0.01),
        execution_profile=PROFILE,
        cash=10_000.0,
        cost_model=_Model(5.0),
        modeled_cost_dollars=5.0,
    )
    assert report.cash["cash_after"] == pytest.approx(-5.0)
    assert report.blocked and report.fatal_findings[0].check == "cash"


# ---------------------------------------------------------------------------
# The cost expression against the cost model
# ---------------------------------------------------------------------------


class _Model:
    def __init__(self, total: float) -> None:
        self._total = total

    def expected_for_intents(self, intents, reference_prices):
        return {"total": self._total}


class _StageTwoModel:
    def expected_for_intents(self, intents, reference_prices):
        raise NotImplementedError("deferred to stage 3")


def test_no_declared_cost_expression_means_nothing_is_asserted() -> None:
    report = validate_batch(
        [Intent("AAA", 10.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_000.0,
    )
    assert report.cost["status"] == "not_supplied" and report.ok


def test_a_cost_expression_within_tolerance_passes_and_one_outside_it_fails() -> None:
    close = validate_batch(
        [Intent("AAA", 10.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_000.0,
        cost_model=_Model(1.0),
        modeled_cost_dollars=1.000009,
    )
    assert close.cost["status"] == "checked" and close.ok
    far = validate_batch(
        [Intent("AAA", 10.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_000.0,
        cost_model=_Model(1.0),
        modeled_cost_dollars=2.0,
    )
    # 1 dollar on 1,000 dollars traded is 10 bps against a 0.1 bp tolerance.
    assert far.cost["difference_bps"] == pytest.approx(10.0)
    assert far.blocked and far.fatal_findings[0].check == "cost"


def test_an_unverifiable_cost_expression_is_not_a_verified_one() -> None:
    report = validate_batch(
        [Intent("AAA", 10.0)],
        limits=(),
        ctx=_ctx(),
        prices=PRICES,
        settings=ValidationSettings(),
        execution_profile=PROFILE,
        cash=10_000.0,
        cost_model=_StageTwoModel(),
        modeled_cost_dollars=1.0,
    )
    assert report.cost["status"] == "unavailable" and report.blocked


# ---------------------------------------------------------------------------
# Settings and capability parsing
# ---------------------------------------------------------------------------


def test_the_settings_block_refuses_an_unknown_key() -> None:
    with pytest.raises(ValidationError, match="unknown keys"):
        ValidationSettings.from_profile({"validation": {"limit_tolerence": 1e-6}})


def test_the_settings_come_from_the_declared_blocks() -> None:
    settings = ValidationSettings.from_profile(
        PROFILE, recovery_policy={"preference_response": "block", "bound_slack": 1e-9}
    )
    assert settings.limit_tolerance == 1e-6 and settings.mode == "blocking"
    assert settings.preference_response == "block" and settings.bound_slack == 1e-9


def test_capability_list_columns_parse_from_json_and_from_lists() -> None:
    assert capability_row(_caps(), "AAA")["restrictions"] == ()
    assert capability_row(_caps(restrictions='["halted"]'), "AAA")["restrictions"] == ("halted",)
    assert capability_row({"AAA": {"restrictions": ["halted"]}}, "AAA")["restrictions"] == ("halted",)
    assert capability_row(_caps(), "ZZZ") is None
