"""The state-coupled decision path (spec §3.7, §3.8): the seams that could forge a record.

Three failures are worth a test more than the rest of this module put together, and each
one is a way the record could look right while the account did something else.

1. The account trades a quantity no validation approved. Batch rounding sizes and
   revalidates the intents; the simulator then sizes *again* from the weight row the hook
   returns. If the two disagree by one share, the record's rounding block describes a batch
   that was never submitted.
2. A fold is dropped for one arm and kept for another. Admission is the shared calibration
   family's, and the control arm computes it without fitting a slope. If the count it reads
   diverges from the count the arms' fits report, decision D6 is silently broken and every
   per-fold denominator moves.
3. A bar on which the constructor declines to trade liquidates the book. "No decision" and
   "go flat" are different instructions and the difference is the whole position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.calibration import CalibrationSpec, fit_calibration
from prism.decisions import _row_from_intents, measurable_limits, usable_periods
from prism.execution.accounting import decide
from prism.execution.cost_model import CostModel, CostStack
from prism.execution.portfolio_state import (
    AccountingError,
    OrderIntent,
    state_from_shares,
)
from prism.execution.simulate import simulate
from prism.portfolio.limits import load_limits

BARS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


def _stack() -> CostStack:
    return CostStack.from_mapping(
        {
            "spread": {"mode": "flat", "flat_bps": 2.0},
            "impact": {"quadratic_notional_coeff": 0.0, "participation_model": "sqrt", "participation_coeff": 0.0},
            "dollar_volume": {
                "spread_bucket": {
                    "estimator": "median",
                    "window": "trailing",
                    "window_bars": 20,
                    "refresh": "fold",
                    "floor_dollars": 0.0,
                },
                "participation": {
                    "estimator": "median",
                    "window": "trailing",
                    "window_bars": 20,
                    "refresh": "fold",
                    "floor_dollars": 0.0,
                },
                "impact": {
                    "estimator": "same_bar",
                    "window": "trailing",
                    "window_bars": 1,
                    "refresh": "bar",
                    "floor_dollars": 100000.0,
                },
            },
            "commission_bps": 1.0,
            "borrow_bps_annual": 0.0,
            "cash_income_bps_annual": 0.0,
            "financing_bps_annual": 0.0,
            "short_proceeds": "collateral_no_rebate",
            "holding_cost_settlement": "per_bar",
            "trading_days_per_year": 252,
        }
    )


# --------------------------------------------------------------------------
# 1. the rounded batch and the simulator's own sizing are the same batch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("equity", [10_000.0, 137_431.17, 1_000_000.0])
def test_the_intent_row_reproduces_the_quantities_batch_rounding_validated(equity: float) -> None:
    """§3.7: the weight row the simulator sizes against must imply the validated shares.

    The hook returns weights, not shares, because that is the simulator's decision
    interface. The row is built as ``(held + qty) * price / equity`` at this bar's own
    equity and prices, so ``decide`` recovers ``held + qty`` exactly and trades ``qty``.
    A one-share divergence here would mean the account traded a batch the second
    validation never saw.
    """
    prices = pd.Series({"AAA": 401.37, "BBB": 17.05, "CCC": 1_913.44, "DDD": 3.01})
    positions = pd.Series({"AAA": 12.0, "BBB": -300.0, "CCC": 0.0, "DDD": 0.0})
    intents = [
        OrderIntent(sid="AAA", qty=-5.0, increment=1.0, reference_price=prices["AAA"], decision_bar=BARS[0]),
        OrderIntent(sid="BBB", qty=41.0, increment=1.0, reference_price=prices["BBB"], decision_bar=BARS[0]),
        OrderIntent(sid="CCC", qty=3.0, increment=1.0, reference_price=prices["CCC"], decision_bar=BARS[0]),
    ]
    row = _row_from_intents(intents, positions=positions, prices=prices, equity=equity)
    assert row is not None
    # DDD carries no intent, so it is absent from the row: absent is "hold", never "flat".
    assert "DDD" not in row.index

    state = state_from_shares(BARS[0], positions, prices, cash=equity - float((positions * prices).sum()))
    batch = decide(state, row, prices, decision_bar=BARS[0], increment=1.0, min_notional=0.0, equity=equity)
    traded = {intent.sid: intent.qty for intent in batch.intents}
    assert traded == {intent.sid: intent.qty for intent in intents}


def test_a_name_the_batch_declined_is_held_rather_than_sold() -> None:
    """A name with no intent is absent from the row, and the simulator holds it (§3.6)."""
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0})
    positions = pd.Series({"AAA": 10.0, "BBB": 20.0})
    intents = [OrderIntent(sid="AAA", qty=2.0, increment=1.0, reference_price=100.0, decision_bar=BARS[0])]
    row = _row_from_intents(intents, positions=positions, prices=prices, equity=10_000.0)
    state = state_from_shares(BARS[0], positions, prices, cash=8_000.0)
    batch = decide(state, row, prices, decision_bar=BARS[0], increment=1.0, equity=10_000.0)
    assert [(i.sid, i.qty) for i in batch.intents] == [("AAA", 2.0)]


def test_no_intents_is_no_row_at_all() -> None:
    assert _row_from_intents((), positions=pd.Series(dtype=float), prices=pd.Series(dtype=float), equity=1.0) is None


# --------------------------------------------------------------------------
# 2. fold admission is one decision, shared by every arm (D6)
# --------------------------------------------------------------------------


def _panels(n_periods: int, *, holes: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(11)
    index = pd.date_range("2023-01-31", periods=n_periods, freq="ME")
    names = [f"S{i}" for i in range(8)]
    scores = pd.DataFrame(rng.normal(size=(n_periods, 8)), index=index, columns=names)
    returns = pd.DataFrame(rng.normal(0.0, 0.05, size=(n_periods, 8)), index=index, columns=names)
    sigma = pd.DataFrame(np.full((n_periods, 8), 0.09), index=index, columns=names)
    for row in range(min(holes, n_periods)):
        returns.iloc[row] = np.nan
    return scores, returns, sigma


@pytest.mark.parametrize("n_periods,holes", [(20, 0), (14, 1), (13, 0), (12, 0), (12, 3), (6, 0)])
def test_usable_periods_is_the_count_the_arms_fits_report(n_periods: int, holes: int) -> None:
    """D6: the control's admission and a convex arm's fit must count the same periods.

    The control fits no slope, so it cannot learn the count from a fit; it reads it from
    the shared family's panels. If the two counts diverged, the same fold would be admitted
    for one arm and excluded for another and every per-fold denominator would move.
    """
    spec = CalibrationSpec.from_mapping(
        {
            "kind": "shared_slope",
            "standardize": "rank_normal",
            "weighting": "per_date_equal",
            "shrink_pseudo_periods": 12.0,
            "expected_sign": "positive",
            "uncertainty": "time_block_bootstrap",
            "block_scheme": "circular_moving",
            "block_periods": 3,
            "n_resamples": 64,
            "seed": 1,
            "min_training_periods": 12,
            "min_blocks": 4,
            "inner_oos_required": False,
            "executable_return": "open_to_open",
            "label_distributions": "ignored",
            "sigma_h": {"estimator": "ewma_daily", "window_bars": 63, "scaling": "per_bar_times_h"},
        },
        source="test",
    )
    scores, returns, sigma = _panels(n_periods, holes=holes)
    assert usable_periods(scores, returns, sigma, spec) == fit_calibration(scores, returns, sigma, spec).n_periods


# --------------------------------------------------------------------------
# 3. no decision is not a liquidation
# --------------------------------------------------------------------------


def _panel(values: float) -> pd.DataFrame:
    return pd.DataFrame({"AAA": [values] * len(BARS), "BBB": [values / 2] * len(BARS)}, index=BARS)


def test_a_hook_that_declines_holds_the_book() -> None:
    """§3.6: ``None`` from the hook is the no-decision marker, which holds every position."""
    calls: list[str] = []

    def hook(call):  # type: ignore[no-untyped-def]
        calls.append(call.bar)
        if call.bar != BARS[0]:
            return None
        return pd.Series({"AAA": 0.5})

    result = simulate(
        _panel(100.0),
        _panel(100.0),
        cost_model=CostModel(_stack(), equity=10_000.0, horizon_bars=21),
        initial_cash=10_000.0,
        increment=1.0,
        bars=BARS,
        decide_hook=hook,
    )
    assert calls == BARS
    shares = result.shares["AAA"].fillna(0.0)
    assert shares.loc[BARS[0]] == 0.0  # decided at the close of bar 0, filled at the open of bar 1
    assert shares.loc[BARS[1]] == 50.0
    assert (shares.loc[BARS[1:]] == 50.0).all()  # never sold by a bar that decided nothing


def test_the_hook_sees_the_equity_the_decision_is_sized_against() -> None:
    """The hook's ``equity`` is the marked equity ``decide`` uses, on the same bar."""
    seen: list[tuple[str, float]] = []

    def hook(call):  # type: ignore[no-untyped-def]
        seen.append((call.bar, call.equity))
        return None

    simulate(
        _panel(100.0),
        _panel(100.0),
        cost_model=CostModel(_stack(), equity=10_000.0, horizon_bars=21),
        initial_cash=10_000.0,
        bars=BARS,
        decide_hook=hook,
    )
    assert [equity for _, equity in seen] == [10_000.0] * len(BARS)


def test_simulate_refuses_a_decided_panel_and_a_hook_together() -> None:
    """Two decisions on one bar, with nothing in the record saying which traded."""
    with pytest.raises(AccountingError, match="either a decided target panel or a decide_hook"):
        simulate(
            _panel(100.0),
            _panel(100.0),
            _panel(0.0),
            cost_model=CostModel(_stack(), equity=1.0, horizon_bars=21),
            bars=BARS,
            decide_hook=lambda call: None,
        )


# --------------------------------------------------------------------------
# the limits a path cannot measure are recorded, never reported satisfied
# --------------------------------------------------------------------------


def test_a_path_without_a_snapshot_records_the_budget_as_unchecked() -> None:
    """§3.7: on the heuristic path the variance budget is not enforced and not measured.

    Reporting it satisfied would be worse than not reporting it: the book's risk was never
    evaluated, and a record that says "no violation" claims it was.
    """
    limits, _ = load_limits(
        [
            {"name": "max_gross", "class": "preference", "priority": 1, "value": 1.0},
            {"name": "variance_budget", "class": "preference", "priority": 2, "params": {"value": 0.0332}},
        ],
        source="test",
    )
    kept, unchecked = measurable_limits(limits, None)
    assert [limit.name for limit in kept] == ["max_gross"]
    assert unchecked == ("variance_budget",)
    kept, unchecked = measurable_limits(limits, object())
    assert len(kept) == 2 and unchecked == ()


# ---------------------------------------------------------------------------------
# 4. A holding nobody could compute is read as flat
# ---------------------------------------------------------------------------------
#
# ``decide_bar`` aligns the caller's ``w0`` and ``positions`` onto the constructor
# universe. Two cases arrive as a missing cell and they are not the same thing: a name the
# caller never mentioned is not held, and 0.0 is its weight; a name the caller mentioned as
# NaN is a holding that could not be computed. Filling both with zero would tell the
# constructor the account is flat in a name it may be short, and the program would then
# "open" a position that is really a doubling. The evaluator's own call site refuses an
# unpriced held name upstream (``evaluate.py::_decision_marks``), so this is unreachable
# from it today -- which is exactly why it needs a test of its own: nothing else in the
# suite would notice if a future caller stopped refusing first.


def test_a_name_the_caller_never_mentions_is_not_held_and_carries_zero() -> None:
    from prism.decisions import _aligned_holding

    index = pd.Index(["AAA", "BBB", "CCC"])
    aligned = _aligned_holding(pd.Series({"AAA": 0.4, "CCC": -0.2}), index, name="w0")
    assert list(aligned.index) == ["AAA", "BBB", "CCC"]
    assert aligned.to_dict() == {"AAA": 0.4, "BBB": 0.0, "CCC": -0.2}


def test_a_holding_stated_as_nan_is_refused_rather_than_read_as_flat() -> None:
    from prism.decisions import DecisionError, _aligned_holding

    index = pd.Index(["AAA", "BBB"])
    with pytest.raises(DecisionError, match="non-finite"):
        _aligned_holding(pd.Series({"AAA": 0.4, "BBB": np.nan}), index, name="w0")
    with pytest.raises(DecisionError, match="non-finite"):
        _aligned_holding(pd.Series({"AAA": 1.0, "BBB": np.inf}), index, name="positions")


def test_a_nan_outside_the_constructor_universe_is_not_the_constructors_problem() -> None:
    from prism.decisions import _aligned_holding

    # A name the universe does not carry cannot be traded on this bar, so its unusable
    # holding is not this decision's defect and must not stop the bar.
    aligned = _aligned_holding(pd.Series({"AAA": 0.4, "ZZZ": np.nan}), pd.Index(["AAA", "BBB"]), name="w0")
    assert aligned.to_dict() == {"AAA": 0.4, "BBB": 0.0}


def test_a_duplicated_sid_in_a_holding_vector_is_refused() -> None:
    from prism.decisions import DecisionError, _aligned_holding

    doubled = pd.Series([0.3, 0.5], index=pd.Index(["AAA", "AAA"]))
    with pytest.raises(DecisionError, match="more than once"):
        _aligned_holding(doubled, pd.Index(["AAA", "BBB"]), name="w0")
