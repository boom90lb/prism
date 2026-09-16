"""Flow F1, position-space combination (engine specification §3.3).

The state machine is the whole point of the flow, so the cases pinned here are
the ones whose confusion is expensive: a component that is not due must not be
scored at all, "no opinion" must hold the prior sub-target while an explicit
0.0 exits and an *expired* output exits, a masked name must leave the ranking
before the rule runs and carry no decision after it, and opposing sub-targets
must net rather than the last component winning.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from prism.composition import (
    ComponentPlan,
    ComponentState,
    CompositionError,
    compose,
)
from prism.signal.base import Signal

SESSIONS = [f"2026-01-{day:02d}" for day in range(1, 32)]


class FakeComponent(Signal):
    """Scores from a dict, optional instructions, and it counts its own calls."""

    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    price_scale_invariant = True
    fitted_state = "none"
    estimated_decay_bars = None

    def __init__(self, scores: dict[str, float], marks: dict[str, str] | None = None) -> None:
        self.scores = scores
        self.marks = marks
        self.score_calls = 0

    @property
    def horizon_bars(self) -> int:
        return 5

    @property
    def required_history(self) -> int:
        return 1

    def fit(self, close, volume=None) -> "FakeComponent":
        return self

    def score(self, close, volume=None) -> pd.Series:
        self.score_calls += 1
        return pd.Series(self.scores, dtype=float).reindex(close.columns)

    def instructions(self, close, volume=None) -> pd.Series | None:
        if self.marks is None:
            return None
        return pd.Series(self.marks, dtype=object)


def _panel(symbols, bars=6) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day in SESSIONS[:bars]])
    return pd.DataFrame(
        {symbol: [100.0 + i] * bars for i, symbol in enumerate(symbols)},
        index=index,
    )


def _plan(component, **kwargs) -> ComponentPlan:
    kwargs.setdefault("component_id", "c1")
    kwargs.setdefault("to_weights", "top_k_equal_weight")
    kwargs.setdefault("to_weights_params", {"k": 2, "max_gross": 1.0})
    return ComponentPlan(component=component, **kwargs)


def _compose(plans, decision_bar, panel, **kwargs):
    return compose(
        plans,
        decision_bar=decision_bar,
        scoring_panels={"price": panel},
        sessions=SESSIONS,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Due-ness: a component that is not due is not scored
# ---------------------------------------------------------------------------


def test_a_component_that_is_not_due_carries_its_sub_targets_and_is_not_scored() -> None:
    symbols = ["AAA", "BBB", "CCC"]
    panel = _panel(symbols)
    component = FakeComponent({"AAA": 3.0, "BBB": 2.0, "CCC": 1.0})
    plan = _plan(component, update_every_bars=21, valid_for_bars=26)
    state = ComponentState(
        component_id="c1",
        last_output_bar="2026-01-02",
        decision_bar="2026-01-02",
        sub_targets={"AAA": 0.5, "BBB": 0.5},
    )

    result = _compose([plan], "2026-01-05", panel, states={"c1": state})

    assert component.score_calls == 0  # not due -> not scored at all
    output = result.component_outputs[0]
    assert output.due is False and output.raw is None
    assert result.targets["AAA"] == 0.5 and result.targets["BBB"] == 0.5
    assert result.states["c1"].last_output_bar == "2026-01-02"  # anchor unmoved


def test_the_component_is_due_once_update_every_bars_have_elapsed() -> None:
    symbols = ["AAA", "BBB", "CCC"]
    panel = _panel(symbols)
    component = FakeComponent({"AAA": 3.0, "BBB": 2.0, "CCC": 1.0})
    plan = _plan(component, update_every_bars=3, valid_for_bars=10)
    state = ComponentState(component_id="c1", last_output_bar="2026-01-02", sub_targets={"AAA": 1.0})

    result = _compose([plan], "2026-01-05", panel, states={"c1": state})

    assert component.score_calls == 1
    assert result.component_outputs[0].due is True
    assert result.states["c1"].last_output_bar == "2026-01-05"


# ---------------------------------------------------------------------------
# The three "due" outcomes: finite replaces, NaN holds, 0.0 exits
# ---------------------------------------------------------------------------


def test_nan_holds_the_prior_sub_target_while_zero_sets_it_flat() -> None:
    # The three "due" outcomes in one cross-section, read straight off a
    # target_weights component so the rule cannot be blamed for any of them:
    # AAA finite (replaced), BBB NaN (no opinion -> prior held), CCC 0.0
    # (explicit flat, which is an exit and not an absence).
    symbols = ["AAA", "BBB", "CCC"]
    panel = _panel(symbols)
    component = FakeComponent({"AAA": 0.6, "BBB": float("nan"), "CCC": 0.0})
    plan = _plan(component, to_weights="none", output_kind="target_weights", to_weights_params={}, valid_for_bars=5)
    state = ComponentState(
        component_id="c1",
        last_output_bar="2026-01-02",
        sub_targets={"AAA": 0.25, "BBB": 0.25, "CCC": 0.25},
    )

    result = _compose([plan], "2026-01-03", panel, states={"c1": state})

    assert result.targets["AAA"] == 0.6  # replaced by the rule value
    assert result.targets["BBB"] == 0.25  # no opinion: the prior sub-target stands
    assert result.targets["CCC"] == 0.0  # explicit flat: a full exit
    assert result.states["c1"].sub_targets["BBB"] == 0.25


def test_a_sid_with_no_prior_sub_target_and_no_opinion_carries_no_decision() -> None:
    symbols = ["AAA", "BBB"]
    panel = _panel(symbols)
    component = FakeComponent({"AAA": 1.0, "BBB": float("nan")})
    plan = _plan(component, to_weights="none", output_kind="target_weights", to_weights_params={})

    result = _compose([plan], "2026-01-03", panel)

    assert result.targets["AAA"] == 1.0
    assert math.isnan(result.targets["BBB"])  # NaN, never 0.0: no decision at all
    assert "BBB" not in result.states["c1"].sub_targets  # NaN is not stored


# ---------------------------------------------------------------------------
# Expiry is an explicit zero, and a late refresh inside the grace is not expiry
# ---------------------------------------------------------------------------


def test_an_expired_output_flattens_every_sub_target_and_is_recorded() -> None:
    symbols = ["AAA", "BBB"]
    panel = _panel(symbols)
    component = FakeComponent({"AAA": 0.3, "BBB": float("nan")})
    plan = _plan(
        component,
        to_weights="none",
        output_kind="target_weights",
        to_weights_params={},
        update_every_bars=21,
        valid_for_bars=26,
    )
    state = ComponentState(
        component_id="c1",
        last_output_bar="2026-01-01",
        sub_targets={"AAA": 0.5, "BBB": -0.5},
    )

    # 2026-01-28 is 27 sessions past the output: beyond valid_for_bars = 26.
    result = _compose([plan], "2026-01-28", panel, states={"c1": state})

    assert component.score_calls == 1  # 27 elapsed >= update_every_bars, so it also refreshes
    output = result.component_outputs[0]
    assert output.expired is True
    assert result.targets["AAA"] == 0.3  # the fresh output replaces the expired one
    # BBB has no fresh opinion, and the sub-target it would have held EXPIRED:
    # an expired output is an explicit zero, never a stale -0.5.
    assert result.targets["BBB"] == 0.0


def test_a_late_refresh_inside_the_grace_holds_and_beyond_it_flattens() -> None:
    # valid_for_bars is the grace, not a second cadence: the reference book
    # refreshes every 21 sessions and holds for 26, so a refresh up to five
    # sessions late holds its book rather than liquidating it. One session
    # further and the book flattens — that is the whole point of the field.
    symbols = ["AAA", "BBB"]
    panel = _panel(symbols)
    component = FakeComponent({"AAA": float("nan"), "BBB": float("nan")})
    plan = _plan(
        component,
        to_weights="none",
        output_kind="target_weights",
        to_weights_params={},
        update_every_bars=21,
        valid_for_bars=26,
    )
    state = ComponentState(
        component_id="c1",
        last_output_bar="2026-01-01",
        sub_targets={"AAA": 0.5, "BBB": -0.5},
    )

    inside = _compose([plan], "2026-01-25", panel, states={"c1": state})  # 24 sessions late
    beyond = _compose([plan], "2026-01-28", panel, states={"c1": state})  # 27 sessions late

    assert inside.component_outputs[0].expired is False
    assert inside.targets["AAA"] == 0.5 and inside.targets["BBB"] == -0.5

    assert beyond.component_outputs[0].expired is True
    assert beyond.component_outputs[0].disabled_reason == "expired"
    assert beyond.targets["AAA"] == 0.0 and beyond.targets["BBB"] == 0.0
    assert beyond.states["c1"].sub_targets == {"AAA": 0.0, "BBB": 0.0}


def test_an_all_nan_output_still_advances_the_anchor_and_restarts_the_clock() -> None:
    symbols = ["AAA", "BBB"]
    panel = _panel(symbols)
    component = FakeComponent({"AAA": float("nan"), "BBB": float("nan")})
    plan = _plan(
        component,
        to_weights="none",
        output_kind="target_weights",
        to_weights_params={},
        update_every_bars=2,
        valid_for_bars=4,
    )
    state = ComponentState(component_id="c1", last_output_bar="2026-01-01", sub_targets={"AAA": 0.3})

    result = _compose([plan], "2026-01-03", panel, states={"c1": state})

    assert result.states["c1"].last_output_bar == "2026-01-03"
    assert result.states["c1"].sub_targets == {"AAA": 0.3}  # NaNs hold the priors
    assert any("entirely NaN output" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


def test_a_liquidate_instruction_sets_the_sid_flat_and_reports_it() -> None:
    symbols = ["AAA", "BBB"]
    panel = _panel(symbols)
    component = FakeComponent({"AAA": 2.0, "BBB": 1.0}, marks={"AAA": "liquidate"})
    plan = _plan(component, to_weights_params={"k": 2, "max_gross": 1.0})

    result = _compose([plan], "2026-01-03", panel)

    assert result.targets["AAA"] == 0.0  # flat despite ranking first
    assert result.liquidate == frozenset({"AAA"})
    assert result.states["c1"].sub_targets["AAA"] == 0.0


def test_a_no_decision_mark_holds_the_prior_rather_than_exiting() -> None:
    symbols = ["AAA", "BBB"]
    panel = _panel(symbols)
    component = FakeComponent({"AAA": 2.0, "BBB": 1.0}, marks={"BBB": "no_decision"})
    plan = _plan(component, to_weights_params={"k": 1, "max_gross": 1.0}, valid_for_bars=5)
    state = ComponentState(component_id="c1", last_output_bar="2026-01-01", sub_targets={"BBB": 0.4})

    result = _compose([plan], "2026-01-03", panel, states={"c1": state})

    assert result.targets["AAA"] == 1.0
    assert result.targets["BBB"] == 0.4  # held: the rule would have pinned it to 0.0
    assert result.liquidate == frozenset()


def test_an_unknown_instruction_is_refused() -> None:
    panel = _panel(["AAA"])
    component = FakeComponent({"AAA": 1.0}, marks={"AAA": "hedge"})
    with pytest.raises(CompositionError, match="unknown instruction"):
        _compose([_plan(component, to_weights_params={"k": 1, "max_gross": 1.0})], "2026-01-03", panel)


# ---------------------------------------------------------------------------
# The unrankable mask: out of the ranking first, no decision after
# ---------------------------------------------------------------------------


def test_a_masked_name_leaves_the_ranking_and_then_carries_no_decision() -> None:
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    panel = _panel(symbols)
    component = FakeComponent({"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0})
    plan = _plan(component, to_weights_params={"k": 2, "max_gross": 1.0})

    unmasked = _compose([plan], "2026-01-03", panel)
    masked = _compose([plan], "2026-01-03", panel, unrankable={"AAA"})

    assert unmasked.targets["AAA"] == 0.5 and unmasked.targets["BBB"] == 0.5
    # AAA left the cross-section before the rule ran, so CCC takes its slot ...
    assert masked.targets["BBB"] == 0.5 and masked.targets["CCC"] == 0.5
    # ... and AAA carries no decision, rather than the explicit 0.0 the rule
    # would have pinned it to: a held position is held until the event clears.
    assert math.isnan(masked.targets["AAA"])
    assert masked.component_outputs[0].masked == ("AAA",)


def test_a_masked_sid_outside_the_scored_cross_section_is_ignored() -> None:
    panel = _panel(["AAA", "BBB"])
    component = FakeComponent({"AAA": 2.0, "BBB": 1.0})
    result = _compose(
        [_plan(component, to_weights_params={"k": 1, "max_gross": 1.0})],
        "2026-01-03",
        panel,
        unrankable={"ZZZ"},
    )
    assert result.component_outputs[0].masked == ()


# ---------------------------------------------------------------------------
# The account target: weighted sum, netting, renormalization
# ---------------------------------------------------------------------------


def test_opposing_sub_targets_net_at_the_declared_weights() -> None:
    symbols = ["AAA", "BBB"]
    panel = _panel(symbols)
    long_book = FakeComponent({"AAA": 1.0, "BBB": float("nan")})
    short_book = FakeComponent({"AAA": 1.0, "BBB": float("nan")})

    plans = [
        ComponentPlan(
            component=long_book,
            component_id="long",
            weight=0.75,
            to_weights="top_k_equal_weight",
            to_weights_params={"k": 1, "max_gross": 1.0},
        ),
        ComponentPlan(
            component=short_book,
            component_id="short",
            weight=0.25,
            to_weights="none",
            output_kind="target_weights",
        ),
    ]
    short_book.scores = {"AAA": -1.0, "BBB": float("nan")}

    result = _compose(plans, "2026-01-03", panel)

    # 0.75 * (+1.0) + 0.25 * (-1.0)
    assert result.targets["AAA"] == pytest.approx(0.5)
    # The long book pinned BBB flat and the short book holds no sub-target for
    # it: the netted target is that explicit flat, not "no decision".
    assert result.targets["BBB"] == 0.0


def test_without_renormalization_a_dropped_component_reduces_the_gross() -> None:
    symbols = ["AAA", "BBB"]
    panel = _panel(symbols)
    live = FakeComponent({"AAA": 1.0, "BBB": float("nan")})
    dropped = FakeComponent({"AAA": 0.0, "BBB": float("nan")})
    plans = [
        ComponentPlan(
            component=live,
            component_id="live",
            weight=0.5,
            to_weights="none",
            output_kind="target_weights",
        ),
        ComponentPlan(
            component=dropped,
            component_id="dropped",
            weight=0.5,
            to_weights="none",
            output_kind="target_weights",
        ),
    ]

    plain = _compose(plans, "2026-01-03", panel)
    scaled = _compose(plans, "2026-01-03", panel, renormalize=True)

    assert plain.targets["AAA"] == pytest.approx(0.5)  # cash absorbs the difference
    assert plain.renormalization == 1.0
    assert scaled.targets["AAA"] == pytest.approx(1.0)  # the survivor is restored to full gross
    assert scaled.renormalization == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_output_that_expires_before_its_own_refresh_is_refused() -> None:
    component = FakeComponent({"AAA": 1.0})
    with pytest.raises(CompositionError, match="valid_for_bars"):
        _plan(component, update_every_bars=21, valid_for_bars=5)


def test_a_standardized_forecast_component_is_refused_by_flow_f1() -> None:
    component = FakeComponent({"AAA": 1.0})
    with pytest.raises(CompositionError, match="calibrated_convex"):
        _plan(component, output_kind="standardized_forecast")


def test_a_rank_score_component_must_declare_a_rule() -> None:
    component = FakeComponent({"AAA": 1.0})
    with pytest.raises(CompositionError, match="rank-to-weight rule"):
        _plan(component, to_weights="none", to_weights_params={})


def test_duplicate_component_ids_are_refused() -> None:
    panel = _panel(["AAA"])
    plans = [
        _plan(FakeComponent({"AAA": 1.0}), to_weights_params={"k": 1, "max_gross": 1.0}),
        _plan(FakeComponent({"AAA": 1.0}), to_weights_params={"k": 1, "max_gross": 1.0}),
    ]
    with pytest.raises(CompositionError, match="duplicate component id"):
        _compose(plans, "2026-01-03", panel)


def test_a_missing_scoring_basis_panel_is_refused() -> None:
    panel = _panel(["AAA"])
    plan = _plan(
        FakeComponent({"AAA": 1.0}),
        scoring_basis="total_return",
        to_weights_params={"k": 1, "max_gross": 1.0},
    )
    with pytest.raises(CompositionError, match="scoring_basis='total_return'"):
        _compose([plan], "2026-01-03", panel)


def test_composing_no_components_is_refused() -> None:
    with pytest.raises(CompositionError, match="nothing to combine"):
        compose([], decision_bar="2026-01-03", scoring_panels={"price": _panel(["AAA"])})


def test_a_weights_fn_bypasses_the_rule_vocabulary_but_not_the_no_decision_rule() -> None:
    # The seam the live books use: the construct is bound by the caller, and a
    # masked sid still comes back NaN rather than the construct's explicit 0.0.
    panel = _panel(["AAA", "BBB"])
    component = FakeComponent({"AAA": 1.0, "BBB": 1.0})
    plan = ComponentPlan(
        component=component,
        component_id="book",
        to_weights="decile_neutral",
        weights_fn=lambda scores: pd.Series(0.25, index=scores.index, dtype=float),
    )

    result = _compose([plan], "2026-01-03", panel, unrankable={"BBB"})

    assert result.targets["AAA"] == 0.25
    assert math.isnan(result.targets["BBB"])


def test_the_account_target_keeps_the_scored_cross_section_in_order() -> None:
    symbols = ["DDD", "AAA", "CCC"]
    panel = _panel(symbols)
    component = FakeComponent({s: float(i) for i, s in enumerate(symbols)})
    plan = _plan(component, to_weights_params={"k": 1, "max_gross": 1.0})

    result = _compose([plan], "2026-01-03", panel)

    assert list(result.targets.index) == symbols
    assert result.targets.to_numpy().dtype == np.dtype("float64")
