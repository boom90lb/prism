"""The worked timeline of spec §3.6, reproduced to the cent (stage 3 acceptance).

Two tables, both recomputed independently before they were coded against:

*The fixed-share timeline.* 10,000 cash, two names, a 2% cash buffer so the targets are
0.49 each, and a zero cost stack -- ``commission_bps``, ``spread``, both impact
coefficients, ``borrow_bps_annual``, ``financing_bps_annual`` and
``cash_income_bps_annual`` all ``0.0``, so no cost figure appears in the cash column. Fill
prices are the bar opens under ``price_convention = "next_open"``.

*The short-leg addendum.* The same zero stack except ``borrow_bps_annual = 50.0``,
``trading_days_per_year = 252``, ``short_proceeds = "collateral_no_rebate"`` and
``holding_cost_settlement = "per_bar"``. Ten shares of C sold short at 50, a 1.00 per share
distribution on the ex-date, and its pay date.

Dollars are compared to the cent. Weights are compared at an absolute tolerance of ``1e-9``
and never on a formatted string: the open-``t1`` B weight is exactly 0.49245 and a
four-decimal comparison of that cell is rounding-rule dependent. Per-bar accruals are
compared at ``1e-9`` too, because one bar of borrow on a 500 dollar short is 0.0099
dollars, which is below a cent and would round away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.config import ExecutionConfig
from prism.execution.accounting import apply_fills, decide, mark_at_close, mark_prices, submit
from prism.execution.portfolio_state import empty_state
from prism.execution.simulate import simulate
from prism.execution.spread import arrival_slippage_bps
from prism.execution.target_weights import backtest_target_weights
from prism.io.events import Event
from tests.test_accounting_conservation import cost_model, cost_stack

T0, T1, T2, T3 = "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"
BARS = [T0, T1, T2, T3]

#: A = [100, 101, 200, 200], B = [100, 100.5, 100, 200] -- the four-row panel §3.6 names.
OPENS = pd.DataFrame({"A": [100.0, 101.0, 200.0, 200.0], "B": [100.0, 100.5, 100.0, 200.0]}, index=BARS)
#: The closes the fixed-share book marks at. Only t1 differs from the opens.
CLOSES = pd.DataFrame({"A": [100.0, 100.0, 200.0, 200.0], "B": [100.0, 100.0, 100.0, 200.0]}, index=BARS)


def zero_stack(**overrides):
    """The zero cost stack of the worked timeline: every component declared, at zero."""
    declared = dict(
        spread={"mode": "flat", "flat_bps": 0.0},
        impact={"quadratic_notional_coeff": 0.0, "participation_model": "sqrt", "participation_coeff": 0.0},
        commission_bps=0.0,
        borrow_bps_annual=0.0,
        cash_income_bps_annual=0.0,
        financing_bps_annual=0.0,
    )
    declared.update(overrides)
    return cost_stack(**declared)


def _targets() -> pd.DataFrame:
    frame = pd.DataFrame(np.nan, index=BARS, columns=["A", "B"])
    frame.loc[T0] = [0.49, 0.49]
    return frame


def _run():
    return simulate(
        OPENS,
        CLOSES,
        _targets(),
        cost_model=cost_model(zero_stack(), equity=10_000.0, horizon_bars=1),
        initial_cash=10_000.0,
    )


# ---------------------------------------------------------------------------
# the fixed-share timeline
# ---------------------------------------------------------------------------


def test_the_worked_timeline_reproduces_to_the_cent() -> None:
    result = _run()
    ledger = result.ledger

    # close t0: decide at equity 10,000 against reference prices of 100; nothing held yet.
    assert round(ledger.loc[T0, "cash"], 2) == 10_000.00
    assert round(ledger.loc[T0, "equity_close"], 2) == 10_000.00
    assert result.shares.loc[T0].dropna().empty
    intents = {intent.sid: intent for intent in result.decisions[0].intents}
    assert sorted(intents) == ["A", "B"]
    assert intents["A"].qty == 49.0 and intents["B"].qty == 49.0
    assert intents["A"].reference_price == 100.0 and intents["B"].reference_price == 100.0

    # open t1: 49 A at 101 and 49 B at 100.5. 10,000 - 4,949 - 4,924.50 = 126.50.
    assert round(ledger.loc[T1, "cash"], 2) == 126.50
    assert round(ledger.loc[T1, "equity_open"], 2) == 10_000.00
    assert result.shares.loc[T1, "A"] == 49.0 and result.shares.loc[T1, "B"] == 49.0
    fills = result.fills.set_index("sid")
    assert fills.loc["A", "fill_price"] == 101.0 and fills.loc["B", "fill_price"] == 100.5

    # close t1 through t3: marked, never rebalanced.
    for bar, prices, equity in (
        (T1, (100.0, 100.0), 9_926.50),
        (T2, (200.0, 100.0), 14_826.50),
        (T3, (200.0, 200.0), 19_726.50),
    ):
        assert result.marks.loc[bar, "A"] == prices[0]
        assert result.marks.loc[bar, "B"] == prices[1]
        assert round(ledger.loc[bar, "equity_close"], 2) == equity
        assert round(ledger.loc[bar, "cash"], 2) == 126.50
        assert result.shares.loc[bar, "A"] == 49.0 and result.shares.loc[bar, "B"] == 49.0

    # not one cost figure appears in the cash column: the stack declares zero everywhere.
    assert result.costs["total"].abs().max() == 0.0
    assert float(result.fills[["commission", "spread_dollars", "impact_dollars"]].to_numpy().sum()) == 0.0


def test_the_worked_timeline_weights_match_at_1e_9() -> None:
    """Weights are derived, never state, and are compared at ``1e-9`` and not as strings."""
    result = _run()
    expected = {
        T1: {"A": 4_900.0 / 9_926.50, "B": 4_900.0 / 9_926.50},
        T2: {"A": 9_800.0 / 14_826.50, "B": 4_900.0 / 14_826.50},
        T3: {"A": 9_800.0 / 19_726.50, "B": 9_800.0 / 19_726.50},
    }
    for bar, weights in expected.items():
        for sid, value in weights.items():
            assert result.weights.loc[bar, sid] == pytest.approx(value, abs=1e-9)
    # the five-decimal cells the specification prints
    assert round(float(result.weights.loc[T1, "A"]), 5) == 0.49363
    assert round(float(result.weights.loc[T2, "A"]), 5) == 0.66098
    assert round(float(result.weights.loc[T2, "B"]), 5) == 0.33049
    assert round(float(result.weights.loc[T3, "A"]), 5) == 0.49679


def test_the_open_t1_row_is_a_mark_at_the_open_and_charges_nothing_for_the_deviation() -> None:
    """The open-``t1`` row of the table, driven transition by transition.

    It is a mark at the open, not a decision: the fill of step 4 is followed by a mark for
    the record, and no decision is taken at an open. The 73.50 fill-versus-reference
    deviation is overnight drift under a zero cost stack, not a charged cost -- nothing is
    debited for it, and ``arrival_slippage_bps`` still measures it.
    """
    model = cost_model(zero_stack(), equity=10_000.0, horizon_bars=1)
    state = empty_state(T0, cash=10_000.0)
    state = mark_at_close(state, bar=T0, prices=CLOSES.loc[T0], cost_model=model).state
    batch = decide(state, pd.Series({"A": 0.49, "B": 0.49}), CLOSES.loc[T0], decision_bar=T0)
    assert [(i.sid, i.qty) for i in batch.intents] == [("A", 49.0), ("B", 49.0)]
    state = submit(state, batch.intents, submitted_at=T0)

    fills = []
    for intent in batch.intents:
        price = float(OPENS.loc[T1, intent.sid])
        charged = model.at_equity(batch.equity).charge_trade(
            [
                _provisional(intent, price, T1),
            ],
            decided_notional={f"{T0}:{intent.sid}": abs(intent.notional)},
        )
        assert charged.total == 0.0
        fills.append(_provisional(intent, price, T1))
    state = apply_fills(state, fills).state
    state, _ = mark_prices(state, bar=T1, prices=OPENS.loc[T1], stale_on_missing=False)

    assert round(state.cash, 2) == 126.50
    assert round(state.equity(OPENS.loc[T1]), 2) == 10_000.00
    weights = state.weights(OPENS.loc[T1])
    assert weights["A"] == pytest.approx(0.49490, abs=1e-9)
    assert weights["B"] == pytest.approx(0.49245, abs=1e-9)

    ledger = pd.DataFrame(
        [
            {"symbol": f.sid, "qty": f.qty, "fill_price": f.fill_price, "reference_price": f.reference_price}
            for f in fills
        ]
    )
    deviation = float(sum(f.qty * (f.fill_price - f.reference_price) for f in fills))
    assert round(deviation, 2) == 73.50
    measured = arrival_slippage_bps(ledger)
    assert measured.tolist() == [100.0, 50.0]
    # measured, never charged: the cash column carries no debit for those 73.50 dollars.
    assert round(10_000.00 - state.cash, 2) == round(49 * 101.0 + 49 * 100.5, 2)


def _provisional(intent, price: float, bar: str):
    from prism.execution.portfolio_state import Fill

    return Fill(
        execution_id=f"sim:{intent.decision_bar}:{intent.sid}:{bar}:0",
        client_order_id=f"{intent.decision_bar}:{intent.sid}",
        sid=intent.sid,
        ts=bar,
        qty=intent.qty,
        fill_price=price,
        reference_price=intent.reference_price,
        source="simulated",
    )


# ---------------------------------------------------------------------------
# the reproduction engine on the same prices
# ---------------------------------------------------------------------------


def test_the_reproduction_engine_on_the_same_prices() -> None:
    """``backtest_target_weights`` on the same panel, and the gap that is the holding process.

    The constant-weight engine ends 2,293.60 above the fixed-share book on the same prices
    and the same single decision, with zero recorded cost and zero recorded trade after the
    first row. The gap is the implicit rebalancing its trade and cost records do not
    represent: to hold 0.49 / 0.49 through A doubling it must sell A and buy B, and it
    charges nothing for doing so.
    """
    targets = pd.DataFrame(np.nan, index=pd.to_datetime(BARS), columns=["A", "B"])
    targets.iloc[0] = [0.49, 0.49]
    execution = ExecutionConfig(
        spread_bps=0.0, slippage_coeff=0.0, commission_bps=0.0, borrow_rate_bps_annual=0.0, adv_impact_coeff=0.0
    )
    result = backtest_target_weights(
        OPENS.set_axis(pd.to_datetime(BARS)), targets, execution=execution, initial_capital=10_000.0
    )
    assert result.equity.to_list() == pytest.approx([10_000.000000, 14_778.592188, 22_020.102359], abs=1e-6)
    assert result.costs["total"].abs().max() == pytest.approx(0.0, abs=0.0)
    fixed_share = float(_run().ledger.loc[T3, "equity_close"])
    assert round(float(result.equity.iloc[-1]) - fixed_share, 2) == 2_293.60


def test_a_single_row_target_frame_yields_an_empty_reproduction_result() -> None:
    """The target frame must span the price index; §3.6 states this because it bites.

    ``backtest_target_weights`` intersects the two indices, so a one-row target frame gives
    a one-row sample and an empty result -- not an error, which is why it is worth pinning.
    """
    single = pd.DataFrame([[0.49, 0.49]], index=pd.to_datetime([T0]), columns=["A", "B"])
    result = backtest_target_weights(OPENS.set_axis(pd.to_datetime(BARS)), single, initial_capital=10_000.0)
    assert result.equity.empty


# ---------------------------------------------------------------------------
# the short-leg addendum
# ---------------------------------------------------------------------------

U0, U1, U2, U3 = "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05"
U_BARS = [U0, U1, U2, U3]
BORROW_PER_BAR = 500.0 * 0.005 / 252.0  # 0.00992063... dollars on a 500 dollar short


def _short_leg():
    opens = pd.DataFrame({"C": [50.0, 50.0, 49.0, 49.0]}, index=U_BARS)
    closes = pd.DataFrame({"C": [50.0, 50.0, 49.0, 49.0]}, index=U_BARS)
    targets = pd.DataFrame(np.nan, index=U_BARS, columns=["C"])
    targets.loc[U0] = [-0.05]  # -500 of a 10,000 book: ten shares at 50
    distribution = Event(
        sid="C",
        event_type="cash_distribution",
        effective_date=U2,
        payload={"amount_per_share": 1.0, "currency": "USD", "pay_date": U3, "distribution_kind": "dividend"},
        ingested_at="2026-01-01T00:00:00Z",
        source="test",
    )
    stack = zero_stack(borrow_bps_annual=50.0)
    return simulate(
        opens,
        closes,
        targets,
        cost_model=cost_model(stack, equity=10_000.0, horizon_bars=1),
        events=[distribution],
        initial_cash=10_000.0,
    )


def test_the_short_leg_addendum_reproduces() -> None:
    result = _short_leg()
    ledger = result.ledger

    # the sale: cash to 10,500.00 and the position to -10 shares.
    assert result.shares.loc[U1, "C"] == -10.0
    assert round(ledger.loc[U1, "cash"] + BORROW_PER_BAR, 2) == 10_500.00
    # the short is marked at 50, so the collateral behind it is 500.00
    assert result.ledger.loc[U1, "positions_value"] == pytest.approx(-500.0, abs=1e-9)
    # equity before the bar's accrual is exactly 10,500.00 - 500.00; the accrual is the
    # only thing between that and the recorded close.
    assert ledger.loc[U1, "equity_close"] == pytest.approx(10_000.00 - BORROW_PER_BAR, abs=1e-9)
    assert ledger.loc[U1, "holding_cost"] == pytest.approx(BORROW_PER_BAR, abs=1e-9)

    # the ex-date: a 10.00 liability, no cash movement, and equity neutral against the
    # markdown apart from that bar's borrow.
    assert ledger.loc[U2, "liabilities"] == pytest.approx(10.0, abs=1e-9)
    assert ledger.loc[U2, "cash"] == pytest.approx(ledger.loc[U1, "cash"] - ledger.loc[U2, "holding_cost"], abs=1e-9)
    assert ledger.loc[U2, "equity_close"] - ledger.loc[U1, "equity_close"] == pytest.approx(
        -ledger.loc[U2, "holding_cost"], abs=1e-9
    )
    # borrow is charged on the marked short value, not on entry notional: 490, not 500.
    assert ledger.loc[U2, "holding_cost"] == pytest.approx(490.0 * 0.005 / 252.0, abs=1e-12)

    # the pay date: the liability settles and cash falls by exactly 10.00 plus that bar's
    # settled borrow.
    assert ledger.loc[U3, "liabilities"] == pytest.approx(0.0, abs=1e-12)
    assert ledger.loc[U3, "cash"] == pytest.approx(
        ledger.loc[U2, "cash"] - 10.0 - ledger.loc[U3, "holding_cost"], abs=1e-9
    )


def test_collateral_no_rebate_earns_nothing_on_the_short_proceeds() -> None:
    """The whole observable content of the policy name (§3.6), stated as a test.

    Under ``collateral_no_rebate`` the income base is ``cash - short_collateral``, which is
    the unlevered 10,000; under ``proceeds_to_cash_with_rebate`` it is ``cash``, the full
    10,500. A declared 100 bps of cash income therefore pays on 10,000 in the first and on
    10,500 in the second, and the difference is exactly the 500 of proceeds.
    """
    opens = pd.DataFrame({"C": [50.0] * 3}, index=U_BARS[:3])
    closes = pd.DataFrame({"C": [50.0] * 3}, index=U_BARS[:3])
    targets = pd.DataFrame(np.nan, index=U_BARS[:3], columns=["C"])
    targets.loc[U0] = [-0.05]
    per_bar_rate = 100.0 / 1e4 / 252.0

    def run(policy: str):
        stack = zero_stack(cash_income_bps_annual=100.0, financing_bps_annual=200.0, short_proceeds=policy)
        return simulate(opens, closes, targets, cost_model=cost_model(stack, equity=10_000.0), initial_cash=10_000.0)

    no_rebate = run("collateral_no_rebate")
    rebate = run("proceeds_to_cash_with_rebate")
    # u0 has no position, so both books hold the same cash into u1: 10,000 plus that bar's
    # own income. The sale then adds 500 of proceeds, and the two policies differ only in
    # whether those 500 are in the income base.
    unlevered = 10_000.0 * (1.0 + per_bar_rate)
    assert no_rebate.costs.loc[U1, "cash_income"] == pytest.approx(unlevered * per_bar_rate, rel=1e-12)
    assert rebate.costs.loc[U1, "cash_income"] == pytest.approx((unlevered + 500.0) * per_bar_rate, rel=1e-12)
    assert rebate.costs.loc[U1, "cash_income"] - no_rebate.costs.loc[U1, "cash_income"] == pytest.approx(
        500.0 * per_bar_rate, rel=1e-9
    )


def test_short_collateral_and_buying_power_at_the_sale() -> None:
    """``short_collateral(50) = 500``; ``buying_power = 10,500 - 0 - 500 = 10,000``."""
    result = _short_leg()
    state = result.state
    prices = pd.Series({"C": 49.0})
    assert state.short_collateral(prices) == pytest.approx(490.0, abs=1e-12)
    assert state.buying_power(prices) == pytest.approx(state.cash - 490.0, abs=1e-12)
    assert state.committed_cash() == 0.0
