"""The two limit families, the four classes, and the §3.7 violation measures.

The failure this file exists to catch is a limit that measures zero because something it
needed was missing -- a risk snapshot, an ADV series, a per-name cap, a factor column.
A limit that silently reports "satisfied" for a book it never evaluated is worse than no
limit at all, so every one of those paths raises here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.portfolio.limits import (
    ACTION_RESTRICTION_CLASSES,
    HOLDING_STATE_CLASSES,
    Limit,
    LimitClass,
    LimitContext,
    LimitError,
    binding_condition,
    classify_action,
    evaluate_limits,
    is_action_restriction,
    load_limits,
    partition_violations,
    reduction_order,
    restriction_blocks_all_trading,
    restriction_blocks_new_exposure,
    unknown_restrictions,
    violation,
)


def _ctx(w0: dict[str, float] | None = None, **kwargs) -> LimitContext:
    return LimitContext(w0=pd.Series(w0 or {}, dtype=float), **kwargs)


def _limit(name: str, cls: str, priority: int | None = None, **params) -> Limit:
    return Limit(name=name, cls=LimitClass(cls), priority=priority, params=params, source="test")


# ---------------------------------------------------------------------------
# Families, classes, and the ordering between them
# ---------------------------------------------------------------------------


def test_the_four_classes_split_into_exactly_two_families() -> None:
    assert ACTION_RESTRICTION_CLASSES | HOLDING_STATE_CLASSES == set(LimitClass)
    assert not (ACTION_RESTRICTION_CLASSES & HOLDING_STATE_CLASSES)


def test_action_restrictions_carry_no_priority_and_holding_state_limits_require_one() -> None:
    with pytest.raises(LimitError, match="never reduced and carries no priority"):
        _limit("quantity_increment", "physical", priority=0, value=1.0)
    with pytest.raises(LimitError, match="must declare an integer priority"):
        _limit("max_gross", "preference", priority=None, value=1.0)


def test_reduction_order_is_mandatory_state_first_then_preference_by_ascending_priority() -> None:
    order = reduction_order(
        [
            _limit("max_gross", "preference", 2, value=1.0),
            _limit("quantity_increment", "physical", None, value=1.0),
            _limit("broker_requirement", "mandatory_state", 7, value=0.5),
            _limit("net_band", "preference", 1, value=0.02),
        ]
    )
    # The mandatory-state limit is reduced first even at a *higher* priority integer:
    # the class rank dominates, and only then does the lower integer go first.
    assert [limit.name for limit in order] == ["broker_requirement", "net_band", "max_gross"]
    # No action restriction appears in the reduction order at all.
    assert all(not is_action_restriction(limit) for limit in order)


def test_equal_priority_inside_one_holding_state_class_is_refused_at_load() -> None:
    with pytest.raises(LimitError, match="equal priorities inside one holding-state class are refused"):
        load_limits(
            [
                {"name": "max_gross", "class": "preference", "priority": 1, "value": 1.0},
                {"name": "net_band", "class": "preference", "priority": 1, "value": 0.02},
            ],
            source="profiles/x.toml",
        )


def test_equal_priority_across_the_two_holding_state_classes_is_accepted() -> None:
    # The class rank already fixes this pair's order, so nothing is undetermined.
    parsed, _ = load_limits(
        [
            {"name": "max_gross", "class": "preference", "priority": 1, "value": 1.0},
            {"name": "recall_state", "class": "mandatory_state", "priority": 1, "value": 0.5},
        ],
        source="profiles/x.toml",
    )
    assert [limit.name for limit in reduction_order(parsed)] == ["recall_state", "max_gross"]


def test_a_priority_on_an_action_restriction_is_recorded_as_ignored_not_refused() -> None:
    # profiles/etf_small.toml writes `priority = 0` on both of its physical rows; it must
    # keep loading, and the fact must be recorded rather than swallowed.
    parsed, ignored = load_limits(
        [
            {"name": "participation_per_name", "class": "physical", "priority": 0, "value": 0.05},
            {"name": "quantity_increment", "class": "physical", "priority": 0, "value": 1.0},
        ],
        source="profiles/etf_small.toml",
    )
    assert [limit.priority for limit in parsed] == [None, None]
    assert len(ignored) == 2
    assert "is ignored" in ignored[0]


def test_the_loader_accepts_no_other_spelling_and_no_other_case() -> None:
    with pytest.raises(LimitError, match="allowed spellings are"):
        load_limits([{"name": "max_gross", "class": "PREFERENCE", "priority": 1, "value": 1.0}], source="p")
    with pytest.raises(LimitError, match="allowed spellings are"):
        load_limits([{"name": "max_gross", "class": "soft", "priority": 1, "value": 1.0}], source="p")


def test_the_loader_folds_a_row_scalar_into_params_and_merges_an_explicit_table() -> None:
    parsed, _ = load_limits(
        [
            {"name": "max_gross", "class": "preference", "priority": 1, "value": 1.0},
            {
                "name": "variance_budget",
                "class": "preference",
                "priority": 4,
                "params": {"mode": "absolute", "value": 0.02},
            },
        ],
        source="p",
    )
    assert parsed[0].value() == 1.0
    assert parsed[1].params["mode"] == "absolute"


def test_a_band_written_as_a_scalar_value_reads_as_centre_zero() -> None:
    assert _limit("net_band", "preference", 1, value=0.02).band() == (0.0, 0.02)
    assert _limit("net_band", "preference", 1, centre=0.1, half_width=0.02).band() == (0.1, 0.02)


# ---------------------------------------------------------------------------
# The violation measures
# ---------------------------------------------------------------------------


def test_max_gross_and_max_name_weight_measures() -> None:
    ctx = _ctx({"A": 0.0, "B": 0.0})
    w = pd.Series({"A": 0.7, "B": -0.5})
    assert violation(_limit("max_gross", "preference", 1, value=1.0), w, ctx) == pytest.approx(0.2)
    assert violation(_limit("max_name_weight", "preference", 2, value=0.6), w, ctx) == pytest.approx(0.1)
    assert violation(_limit("max_gross", "preference", 1, value=2.0), w, ctx) == 0.0


def test_net_and_turnover_measures_read_the_marked_weights() -> None:
    ctx = _ctx({"A": 0.3, "B": -0.3})
    w = pd.Series({"A": 0.5, "B": -0.1})
    assert violation(_limit("net_band", "preference", 1, value=0.05), w, ctx) == pytest.approx(0.35)
    # turnover = |0.5-0.3| + |-0.1+0.3| = 0.4, cap 0.25
    assert violation(_limit("turnover_cap", "preference", 2, value=0.25), w, ctx) == pytest.approx(0.15)


def test_participation_measure_is_a_bound_on_the_trade_not_the_holding() -> None:
    ctx = _ctx({"A": 0.0}, equity=1_000_000.0, adv_dollars=pd.Series({"A": 1_000_000.0}))
    limit = _limit("participation_per_name", "physical", None, value=0.05)
    # traded 0.10 * 1e6 = 100,000 dollars against 0.05 * 1e6 = 50,000 permitted.
    assert violation(limit, pd.Series({"A": 0.10}), ctx) == pytest.approx(0.05)
    assert violation(limit, pd.Series({"A": 0.05}), ctx) == 0.0


def test_an_unknown_dollar_volume_restricts_the_name_to_nothing_never_to_everything() -> None:
    ctx = _ctx({"A": 0.0}, equity=1_000.0, adv_dollars=pd.Series({"A": float("nan")}))
    limit = _limit("participation_per_name", "physical", None, value=0.05)
    assert violation(limit, pd.Series({"A": 0.10}), ctx) == pytest.approx(0.10)


def test_participation_without_a_declared_adv_series_raises_rather_than_measuring_zero() -> None:
    ctx = _ctx({"A": 0.0}, equity=1_000.0)
    with pytest.raises(LimitError, match="dollar_volume.participation"):
        violation(_limit("participation_per_name", "physical", None, value=0.05), pd.Series({"A": 0.9}), ctx)


def test_variance_budget_without_a_snapshot_raises_rather_than_reporting_a_satisfied_budget() -> None:
    with pytest.raises(LimitError, match="needs a risk snapshot"):
        violation(_limit("variance_budget", "preference", 4, value=0.02), pd.Series({"A": 0.5}), _ctx({"A": 0.0}))


class _Snapshot:
    """A two-factor snapshot at correlation 0.9, the §3.5 worked check."""

    def __init__(self, sids=("A", "B"), factor_ids=("f1", "f2")) -> None:
        self.sids = list(sids)
        self.method = {"factor_ids": list(factor_ids)}
        self.B = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=list(sids), columns=list(factor_ids))
        self.omega = np.array([[1.0, 0.9], [0.9, 1.0]])

    def variance(self, w) -> float:
        aligned = pd.Series(w, dtype=float).reindex(self.sids).fillna(0.0).to_numpy(dtype=float)
        exposure = self.B.to_numpy(dtype=float).T @ aligned
        return float(exposure @ self.omega @ exposure)


def test_variance_budget_uses_the_full_factor_covariance() -> None:
    ctx = _ctx({"A": 0.0, "B": 0.0}, risk=_Snapshot())
    w = pd.Series({"A": 0.5, "B": 0.5})
    # 0.25 + 0.25 + 2*0.25*0.9 = 0.95; a diagonal-only representation would give 0.50.
    assert violation(_limit("variance_budget", "preference", 4, value=0.0), w, ctx) == pytest.approx(0.95**0.5)


def test_a_snapshot_that_does_not_cover_the_universe_is_rejected_before_any_measure() -> None:
    ctx = _ctx({"A": 0.0, "C": 0.0}, risk=_Snapshot())
    with pytest.raises(LimitError, match="covers no estimate"):
        violation(_limit("variance_budget", "preference", 4, value=0.02), pd.Series({"A": 0.5, "C": 0.5}), ctx)


def test_beta_band_reads_the_named_factor_column_and_refuses_an_undeclared_factor() -> None:
    ctx = _ctx({"A": 0.0, "B": 0.0}, risk=_Snapshot())
    w = pd.Series({"A": 0.6, "B": -0.2})
    limit = _limit("beta_band", "preference", 3, value=0.1, factor_id="f1")
    assert violation(limit, w, ctx) == pytest.approx(0.5)
    with pytest.raises(LimitError, match="not a column of the snapshot"):
        violation(_limit("beta_band", "preference", 3, value=0.1, factor_id="f9"), w, ctx)
    with pytest.raises(LimitError, match="declares params.factor_id"):
        violation(_limit("beta_band", "preference", 3, value=0.1), w, ctx)


def test_a_limit_without_a_declared_measure_raises_rather_than_returning_zero() -> None:
    with pytest.raises(LimitError, match="declares no violation measure"):
        violation(_limit("quantity_increment", "physical", None, value=1.0), pd.Series({"A": 0.1}), _ctx())


def test_a_non_finite_proposed_weight_is_refused_not_coerced() -> None:
    with pytest.raises(LimitError, match="must be finite to measure a limit"):
        violation(_limit("max_gross", "preference", 1, value=1.0), pd.Series({"A": float("nan")}), _ctx())


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


def test_the_partition_separates_inherited_from_introduced_breaches() -> None:
    # The marked book already breaches max_gross; the target introduces a name-weight breach.
    ctx = _ctx({"A": 0.9, "B": 0.9})
    limits = [
        _limit("max_gross", "preference", 1, value=1.0),
        _limit("max_name_weight", "preference", 2, value=0.95),
    ]
    partition = partition_violations(limits, pd.Series({"A": 1.2, "B": 0.4}), ctx)
    assert [entry["limit_name"] for entry in partition["existing"]] == ["max_gross"]
    assert [entry["limit_name"] for entry in partition["remaining"]] == ["max_gross"]
    assert [entry["limit_name"] for entry in partition["introduced"]] == ["max_name_weight"]
    # existing / remaining carry the marked and the target measures respectively.
    assert partition["existing"][0]["v"] == pytest.approx(0.8)
    assert partition["remaining"][0]["v"] == pytest.approx(0.6)
    # The three sets are disjoint on limit name.
    names = [entry["limit_name"] for key in partition for entry in partition[key]]
    assert len(set(names)) == 2
    assert binding_condition(partition) == "max_name_weight"


def test_evaluate_limits_returns_satisfied_limits_with_measure_zero() -> None:
    ctx = _ctx({"A": 0.1})
    evaluations = evaluate_limits([_limit("max_gross", "preference", 1, value=1.0)], pd.Series({"A": 0.1}), ctx)
    assert len(evaluations) == 1 and evaluations[0].v == 0.0 and not evaluations[0].breached


def test_a_per_limit_tolerance_overrides_the_profile_default() -> None:
    ctx = _ctx({"A": 0.0})
    limit = Limit(
        name="max_gross", cls=LimitClass.PREFERENCE, priority=1, params={"value": 1.0}, tolerance=0.05, source="t"
    )
    evaluation = evaluate_limits([limit], pd.Series({"A": 1.02}), ctx, limit_tolerance=1e-6)[0]
    assert evaluation.v == pytest.approx(0.02) and not evaluation.breached


# ---------------------------------------------------------------------------
# Action classification and broker restrictions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("held", "delta", "expected"),
    [
        (0.0, 5.0, "C3"),
        (0.0, -5.0, "C3"),
        (10.0, 5.0, "C3"),
        (-10.0, -5.0, "C3"),
        (10.0, -4.0, "C1"),
        (10.0, -10.0, "C1"),
        (-10.0, 10.0, "C1"),
        (10.0, -15.0, "C2"),
        (-1.0, 51.0, "C2"),
    ],
)
def test_action_classes(held: float, delta: float, expected: str) -> None:
    assert classify_action(held, delta) == expected


def test_an_unrecognized_broker_restriction_blocks_new_exposure() -> None:
    # The fail-safe direction: a restriction string this engine does not know is a reason
    # to do less, never a reason to do more.
    assert restriction_blocks_new_exposure(["some_new_regulatory_flag"])
    assert not restriction_blocks_all_trading(["some_new_regulatory_flag"])
    assert unknown_restrictions(["some_new_regulatory_flag", "halted"]) == ("some_new_regulatory_flag",)
    assert restriction_blocks_all_trading(["halted"])
    assert not restriction_blocks_new_exposure([])
