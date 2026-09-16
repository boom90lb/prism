"""The two holding processes, on the panel where they must differ (spec §3.8, §3.6).

Ten thousand of cash, two names, one decision of half the book into each, and then
each name doubles in turn. Nothing else happens: no cost is charged, no event is
applied, every fill prints at the reference price. The two engines still end four
figures apart, and the gap is the whole reason the canonical engine exists.

* ``canonical`` holds **shares**. The decision buys 50 of each at 100; A doubling
  makes the book 15,000 and B doubling makes it 20,000. Nothing rebalanced, so
  nothing was sold into the first move.
* ``historical_constant_weight`` holds **weights**. After A doubles the book is
  still half and half, which means half of A's gain was rotated into B for free,
  and B's later double is earned on 7,500 rather than 5,000: 22,500.

The 2,500 difference is not a rounding tolerance to widen. It is a rebalancing
premium the reproduction engine collects without trading, and the specification's
own expected figures for this case are the two numbers asserted below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.config import ExecutionConfig
from prism.execution.accounting import round_to_increment
from prism.execution.cost_model import CostModel, CostStack
from prism.execution.portfolio_state import AccountingError, OrderIntent
from prism.execution.simulate import simulate
from prism.execution.target_weights import backtest_target_weights

BARS = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]

#: Opens and closes are equal on this panel, so "the fill printed at the reference
#: price" is a fact about the fixture rather than a policy of the simulator.
PRICES = pd.DataFrame(
    [[100.0, 100.0], [100.0, 100.0], [200.0, 100.0], [200.0, 200.0]],
    index=BARS,
    columns=["A", "B"],
)

#: A decision at the close of the first bar and never again. Every later row is
#: all-NaN, which both engines read as "no new pending order".
TARGETS = pd.DataFrame(np.nan, index=BARS, columns=["A", "B"])
TARGETS.loc[BARS[0]] = [0.5, 0.5]


def zero_cost_stack() -> CostStack:
    """A stack that charges nothing, declared term by term.

    Every component is written out because §3.6 refuses a stack that omits one: a
    zero nobody declared is indistinguishable in the record from a zero that was.
    """
    volume = {
        "estimator": "median",
        "window": "trailing",
        "window_bars": 20,
        "refresh": "decision",
        "floor_dollars": 100_000.0,
    }
    return CostStack.from_mapping(
        {
            "spread": {"mode": "flat", "flat_bps": 0.0},
            "impact": {
                "quadratic_notional_coeff": 0.0,
                "participation_model": "linear",
                "participation_coeff": 0.0,
            },
            "dollar_volume": {"spread_bucket": volume, "participation": volume, "impact": volume},
            "commission_bps": 0.0,
            "borrow_bps_annual": 0.0,
            "cash_income_bps_annual": 0.0,
            "financing_bps_annual": 0.0,
            "short_proceeds": "collateral_no_rebate",
            "holding_cost_settlement": "per_bar",
            "trading_days_per_year": 252,
        }
    )


def test_fixed_shares_end_at_twenty_thousand() -> None:
    result = simulate(
        PRICES,
        PRICES,
        TARGETS,
        cost_model=CostModel(zero_cost_stack(), equity=10_000.0),
        initial_cash=10_000.0,
    )

    assert result.equity.loc[BARS[0]] == pytest.approx(10_000.0)
    assert result.equity.loc[BARS[1]] == pytest.approx(10_000.0), "filled at 100, marked at 100"
    assert result.equity.loc[BARS[2]] == pytest.approx(15_000.0), "A doubled; B did not"
    assert result.equity.loc[BARS[3]] == pytest.approx(20_000.0)

    assert result.shares.loc[BARS[3]].to_dict() == {"A": 50.0, "B": 50.0}, "shares never moved after the fill"
    assert len(result.fills) == 2, "one decision, one fill per name, and no rebalance between decisions"
    assert float(result.costs["total"].sum()) == 0.0
    assert result.cash.loc[BARS[3]] == pytest.approx(0.0)


def test_constant_weights_report_twenty_two_thousand_five_hundred() -> None:
    reproduction = backtest_target_weights(
        PRICES,
        TARGETS,
        ExecutionConfig(spread_bps=0.0, commission_bps=0.0, slippage_coeff=0.0, borrow_rate_bps_annual=0.0),
        initial_capital=10_000.0,
    )

    assert float(reproduction.equity.iloc[-1]) == pytest.approx(22_500.0)
    # It never traded, and that is the point: the weights were re-levered by
    # assumption, so the rotation out of A and into B cost nothing and is
    # invisible in the turnover column.
    assert float(reproduction.costs["turnover"].sum()) == pytest.approx(1.0), "the entry trade only"
    assert float(reproduction.costs["total"].sum()) == 0.0


def test_the_gap_between_the_engines_is_the_uncharged_rebalance() -> None:
    """One statement of the finding, so a regression in either engine fails here."""
    canonical = simulate(
        PRICES,
        PRICES,
        TARGETS,
        cost_model=CostModel(zero_cost_stack(), equity=10_000.0),
        initial_cash=10_000.0,
    )
    reproduction = backtest_target_weights(
        PRICES,
        TARGETS,
        ExecutionConfig(spread_bps=0.0, commission_bps=0.0, slippage_coeff=0.0, borrow_rate_bps_annual=0.0),
        initial_capital=10_000.0,
    )
    canonical_end = float(canonical.equity.iloc[-1])
    reproduction_end = float(reproduction.equity.iloc[-1])

    assert reproduction_end - canonical_end == pytest.approx(2_500.0)
    assert reproduction_end / canonical_end - 1.0 == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# the continuous quantity grid
# ---------------------------------------------------------------------------


def test_a_continuous_increment_rounds_nothing() -> None:
    """``increment = 0.0`` is the declared continuous grid of §3.7, not an error."""
    assert round_to_increment(0.0024137, 0.0) == 0.0024137
    assert round_to_increment(0.0024137, 1.0) == 0.0
    with pytest.raises(AccountingError, match=">= 0"):
        round_to_increment(1.0, -1.0)

    intent = OrderIntent(sid="A", qty=0.0024137, increment=0.0, reference_price=100.0, decision_bar=BARS[0])
    assert intent.increment == 0.0
    with pytest.raises(AccountingError, match="whole multiple"):
        OrderIntent(sid="A", qty=0.5, increment=1.0, reference_price=100.0, decision_bar=BARS[0])


def test_a_continuous_book_holds_the_same_shares_at_any_scale() -> None:
    """The comparison run's setting: capital 1.0, fractional shares, same answer.

    The evaluator compares the two engines at the recorded configuration's own
    ``initial_capital = 1.0``, because changing the capital changes the
    participation bound and therefore the targets. That is only honest if the
    canonical engine is scale-free there, which it is exactly when the quantity
    grid is continuous.
    """
    small = simulate(
        PRICES,
        PRICES,
        TARGETS,
        cost_model=CostModel(zero_cost_stack(), equity=1.0),
        initial_cash=1.0,
        increment=0.0,
    )
    large = simulate(
        PRICES,
        PRICES,
        TARGETS,
        cost_model=CostModel(zero_cost_stack(), equity=10_000.0),
        initial_cash=10_000.0,
        increment=0.0,
    )

    assert float(small.equity.iloc[-1]) == pytest.approx(2.0)
    assert float(large.equity.iloc[-1]) == pytest.approx(20_000.0)
    scaled = (small.equity / small.equity.iloc[0] - large.equity / large.equity.iloc[0]).abs().max()
    assert scaled == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# the spread the canonical engine is handed
# ---------------------------------------------------------------------------


def test_the_two_cost_cadences_are_addressed_differently() -> None:
    """``once`` is a per-name vector; ``fold`` is a bar-by-name panel (§3.8).

    Both come out of the same ``_charged_spread`` the reproduction engine
    charges, which is what makes the two engines charge one spread on one
    name-day. They are not the same *shape*, and re-keying the per-name vector
    by bar turns every sid into a date — which is exactly what happened, and is
    why this is a test rather than a comment.
    """
    from types import SimpleNamespace

    from prism.evaluate import CanonicalSettings, _canonical_cost_model

    resolved = SimpleNamespace(session_labels=("2026-01-02", "2026-01-05"))
    stack = CostStack.from_mapping(
        dict(
            zero_cost_stack().declaration(),
            spread={"mode": "bucket", "flat_bps": 1.0, "schedule": [[1.0e9, 1.0], [0.0, 10.0]]},
        )
    )
    settings = CanonicalSettings(cost_stack=stack, share_increment=0.0)

    by_name = _canonical_cost_model(resolved, settings, pd.Series({"A": 3.0, "B": 7.0}), capital=1.0)
    assert by_name.half_spread_bps("A", BARS[1]) == pytest.approx(3.0)
    assert by_name.half_spread_bps("B", BARS[1]) == pytest.approx(7.0)
    # A name the vector does not price falls back where the *stack* says: the
    # widest schedule step, §3.6's "unknown liquidity is never cheap". That is
    # stricter than the reproduction engine's flat fallback, and unreachable on
    # a bucket vector, which already puts an unpriced name in that same step.
    assert by_name.half_spread_bps("Z", BARS[1]) == pytest.approx(10.0)

    panel = pd.DataFrame([[3.0, 7.0], [4.0, 8.0]], index=pd.DatetimeIndex(BARS[:2]), columns=["A", "B"])
    by_bar = _canonical_cost_model(resolved, settings, panel, capital=1.0)
    assert by_bar.half_spread_bps("A", BARS[0]) == pytest.approx(3.0)
    assert by_bar.half_spread_bps("A", BARS[1]) == pytest.approx(4.0)
    assert by_bar.half_spread_bps("Z", BARS[1]) == pytest.approx(10.0)
    # A bar the panel does not cover falls back the same way, rather than to a
    # neighbouring bar's row or to nothing.
    assert by_bar.half_spread_bps("A", BARS[3]) == pytest.approx(10.0)
