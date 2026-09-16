"""The share-based ledger bridge for ``execution/accounting.py`` (spec §3.6, SPEC §1 N4).

``tests/test_ledger_conservation.py`` stays exactly as it is: it is the **weight-space**
bridge for the ``historical_constant_weight`` reproduction engine, which still writes
records and would otherwise have no conservation check at all. This file is the share-space
bridge for the canonical engine, and the two are not interchangeable.

What Lean proves, and over what. ``formal/PrismFormal/Ledger.lean`` is stated over ``Book``
-- ``cash : Int`` with ``qty : List Int`` -- and ``wealth b prices = b.cash + dot b.qty
prices`` (``Ledger.lean:24-30``). Three theorems:

* ``rebalance_conserves`` (``Ledger.lean:40-43``): a rebalance at fixed prices changes
  wealth by exactly the charged costs.
* ``mark_to_market_pnl`` (``Ledger.lean:47-50``): between rebalances wealth moves only by
  held quantities times the price change.
* ``run_conserves`` (``Ledger.lean:80-89``): over any sequence of periods, final wealth is
  initial wealth plus accrued PnL minus charged costs.

The model contains no receivables, no liabilities, no per-bar accrual, and no event leg,
and it needs none: what it covers, it covers exactly, over integers. The projection ``Pi``
of §3.6 maps ``PortfolioState`` to ``Book`` -- cash and each position's shares and price to
integers on the declared share increment and price tick, in a fixed sid order. ``Pi`` is
exact on quantities, because the canonical engine holds every position at a whole
increment; it is exact on prices only when the price is a whole tick, so the price leg
carries a declared tolerance and :func:`test_projection_price_leg_carries_its_tolerance`
pins what that tolerance buys.

What moving to shares preserves, and what it does not (§3.6, "what a share-based bridge
test preserves"):

* preserved -- cost components sum to the recorded total;
* preserved -- equity moves only by PnL, declared cash flows, and charged costs,
  reconstructed independently from fills, prices, events, and the recorded cost rows;
* preserved -- with every rate zero, equity moves by price PnL alone;
* preserved -- a flat book has flat equity;
* **not** preserved -- equity as the cumulative product of a return series. Once shares are
  the state and cash is an account that can earn or pay a declared rate, there is no single
  portfolio return whose cumprod rebuilds the equity path; that identity belongs to a book
  re-levered to constant weights every bar, which is the reproduction engine's holding
  process and not this one. :func:`test_equity_is_not_a_cumprod_of_returns` states that as
  a fact rather than leaving it as an absence.
* **not** preserved -- the weight-space reconstruction of
  ``tests/test_ledger_conservation.py:30-39``, which has no counterpart when the state is
  shares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.execution.accounting import apply_fills, mark_at_close, mark_prices
from prism.execution.cost_model import COST_TERMS, CostInputs, CostModel, CostStack
from prism.execution.portfolio_state import Fill, OrderIntent, OrderRecord, PortfolioState, state_from_shares
from prism.execution.simulate import FillPolicy, simulate
from prism.io.events import Event

BARS = [
    "2026-03-02",
    "2026-03-03",
    "2026-03-04",
    "2026-03-05",
    "2026-03-06",
    "2026-03-09",
    "2026-03-10",
    "2026-03-11",
]


def cost_stack(**overrides) -> CostStack:
    """A declared stack whose every component is named, including its zeros (§3.6).

    The reference values of ``costs/retail_v1.toml`` where the specification gives them:
    ``commission_bps = 1.0``, ``quadratic_notional_coeff = 10.0``,
    ``participation_model = "sqrt"``, ``participation_coeff = 0.0``,
    ``impact.floor_dollars = 100000.0``, ``trading_days_per_year = 252``.
    """
    dollar_volume = {
        "spread_bucket": {
            "estimator": "median",
            "window": "trailing",
            "window_bars": 20,
            "refresh": "decision",
            "floor_dollars": 0.0,
        },
        "participation": {
            "estimator": "median",
            "window": "trailing",
            "window_bars": 20,
            "refresh": "decision",
            "floor_dollars": 0.0,
            "base": "current_equity",
        },
        "impact": {
            "estimator": "same_bar",
            "window": "trailing",
            "window_bars": 1,
            "refresh": "bar",
            "floor_dollars": 100000.0,
        },
    }
    declared = {
        "spread": {"mode": "flat", "flat_bps": 1.0},
        "impact": {
            "quadratic_notional_coeff": 10.0,
            "participation_model": "sqrt",
            "participation_coeff": 0.0,
        },
        "dollar_volume": dollar_volume,
        "commission_bps": 1.0,
        "borrow_bps_annual": 50.0,
        "cash_income_bps_annual": 0.0,
        "financing_bps_annual": 0.0,
        "short_proceeds": "collateral_no_rebate",
        "holding_cost_settlement": "per_bar",
        "trading_days_per_year": 252,
    }
    declared.update(overrides)
    return CostStack.from_mapping(declared)


def cost_model(stack: CostStack | None = None, *, equity: float = 100_000.0, horizon_bars: int = 5) -> CostModel:
    return CostModel(stack or cost_stack(), CostInputs(as_of=BARS[0]), equity=equity, horizon_bars=horizon_bars)


def _panels(bars=BARS, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sids = ["AAA", "BBB", "CCC"]
    closes = pd.DataFrame(
        100.0 * np.cumprod(1.0 + 0.01 * rng.standard_normal((len(bars), len(sids))), axis=0),
        index=list(bars),
        columns=sids,
    )
    opens = closes.shift(1).fillna(100.0) * (1.0 + 0.002 * rng.standard_normal((len(bars), len(sids))))
    volumes = pd.DataFrame(5.0e6, index=list(bars), columns=sids)
    return opens, closes, volumes


def _targets(bars=BARS, sids=("AAA", "BBB", "CCC")) -> pd.DataFrame:
    frame = pd.DataFrame(np.nan, index=list(bars), columns=list(sids))
    frame.loc[bars[0]] = [0.40, 0.30, -0.20]
    frame.loc[bars[3]] = [0.25, 0.35, -0.25]
    frame.loc[bars[5]] = [0.10, 0.20, -0.05]
    return frame


def _run(**kwargs):
    opens, closes, volumes = _panels()
    defaults = dict(
        targets=_targets(),
        cost_model=cost_model(),
        volumes=volumes,
        initial_cash=100_000.0,
        fills=FillPolicy(),
    )
    defaults.update(kwargs)
    return simulate(opens, closes, **defaults)


# ---------------------------------------------------------------------------
# preserved property 1: the components sum to the total
# ---------------------------------------------------------------------------


def test_cost_components_sum_to_the_recorded_total() -> None:
    result = _run()
    costs = result.costs
    recomposed = (
        costs["commission"]
        + costs["spread"]
        + costs["impact_quadratic"]
        + costs["impact_participation"]
        + costs["borrow"]
        + costs["financing"]
        - costs["cash_income"]
    )
    pd.testing.assert_series_equal(costs["total"], recomposed, check_names=False, rtol=0.0, atol=1e-12)
    assert costs["total"].sum() > 0.0, "the fixture must actually charge something"


def test_per_fill_charges_equal_the_recorded_cost_row() -> None:
    """The dollars on the fills are the dollars in the cost row: one charge, recorded once."""
    result = _run()
    charged = result.fills[["commission", "spread_dollars", "impact_dollars"]].sum().sum()
    recorded = result.costs[["commission", "spread", "impact_quadratic", "impact_participation"]].sum().sum()
    assert charged == pytest.approx(recorded, abs=1e-9)


# ---------------------------------------------------------------------------
# preserved property 2: equity moves only by PnL, declared cash flows and charged costs
# ---------------------------------------------------------------------------


def _reconstruct_equity_path(result) -> pd.Series:
    """Rebuild the equity path from fills, marks, events and the recorded cost rows.

    Independent of the ledger's own equity column: it uses the share and mark panels, the
    fills ledger, the applied-event rows and the charged-cost rows, and nothing else.
    """
    ledger = result.ledger
    shares = result.shares.fillna(0.0)
    marks = result.marks
    bars = list(ledger.index)
    rebuilt = [ledger["equity_close"].iloc[0]]
    for previous, bar in zip(bars[:-1], bars[1:]):
        carried = 0.0
        for sid in shares.columns:
            held = float(shares.loc[previous, sid])
            if held == 0.0:
                continue
            before, after = marks.loc[previous, sid], marks.loc[bar, sid]
            assert np.isfinite(before) and np.isfinite(after), f"{sid} lost its mark at {bar}"
            carried += held * (after - before)
        traded = 0.0
        for fill in result.fills[result.fills["bar"] == bar].itertuples():
            traded += fill.qty * (float(marks.loc[bar, fill.sid]) - fill.fill_price)
        events = result.events
        flows = 0.0
        if not events.empty:
            flows = float(events.loc[events["effective_date"] == bar, "equity_delta"].sum())
        costs = float(ledger.loc[bar, "trade_cost"] + ledger.loc[bar, "holding_cost"])
        rebuilt.append(rebuilt[-1] + carried + traded + flows - costs)
    return pd.Series(rebuilt, index=pd.Index(bars, name="bar"), name="equity_close")


def test_equity_moves_only_by_pnl_flows_and_charged_costs() -> None:
    result = _run()
    rebuilt = _reconstruct_equity_path(result)
    pd.testing.assert_series_equal(result.ledger["equity_close"], rebuilt, rtol=0.0, atol=1e-9)


def test_cash_is_the_sum_of_its_recorded_movements() -> None:
    """No unexplained cash term: the final balance is the declared movements, added up."""
    result = _run()
    reconstructed = (
        100_000.0
        + float(result.fills["cash_delta"].sum())
        + (0.0 if result.events.empty else float(result.events["cash_delta"].sum()))
        - float(result.ledger["holding_cost"].sum())
    )
    assert result.state.cash == pytest.approx(reconstructed, abs=1e-9)
    assert result.ledger["cash"].iloc[-1] == pytest.approx(reconstructed, abs=1e-9)


def test_the_balance_sheet_closes_on_every_bar() -> None:
    """``equity == cash + marked positions + receivables - liabilities``, every row."""
    ledger = _run().ledger
    identity = ledger["cash"] + ledger["positions_value"] + ledger["receivables"] - ledger["liabilities"]
    pd.testing.assert_series_equal(ledger["equity_close"], identity, check_names=False, rtol=0.0, atol=1e-9)


def test_a_distribution_and_its_pay_date_are_each_equity_neutral() -> None:
    """The ex-date books a claim and the pay date converts it; neither moves equity.

    The ex-date is equity-neutral *against the markdown*: the price falls by the amount per
    share and the receivable replaces it. The pay date is equity-neutral outright, and
    moves exactly the settled amount in cash.
    """
    bars = BARS[:4]
    opens = pd.DataFrame({"AAA": [100.0, 100.0, 99.0, 99.0]}, index=bars)
    closes = pd.DataFrame({"AAA": [100.0, 100.0, 99.0, 99.0]}, index=bars)
    targets = pd.DataFrame(np.nan, index=bars, columns=["AAA"])
    targets.loc[bars[0]] = [0.50]
    event = Event(
        sid="AAA",
        event_type="cash_distribution",
        effective_date=bars[2],
        payload={"amount_per_share": 1.0, "currency": "USD", "pay_date": bars[3], "distribution_kind": "dividend"},
        ingested_at="2026-01-01T00:00:00Z",
        source="test",
    )
    zero = cost_stack(
        spread={"mode": "flat", "flat_bps": 0.0},
        impact={"quadratic_notional_coeff": 0.0, "participation_model": "sqrt", "participation_coeff": 0.0},
        commission_bps=0.0,
        borrow_bps_annual=0.0,
    )
    result = simulate(
        opens, closes, targets, cost_model=cost_model(zero, equity=10_000.0), initial_cash=10_000.0, events=[event]
    )
    equity = result.ledger["equity_close"]
    shares = float(result.shares.loc[bars[1], "AAA"])
    assert shares == 50.0
    # ex-date: the markdown and the booked receivable cancel to the cent.
    assert equity.loc[bars[2]] == pytest.approx(equity.loc[bars[1]], abs=1e-9)
    assert result.ledger.loc[bars[2], "receivables"] == pytest.approx(shares * 1.0, abs=1e-9)
    assert result.ledger.loc[bars[2], "cash"] == pytest.approx(result.ledger.loc[bars[1], "cash"], abs=1e-9)
    # pay date: equity unchanged, cash up by exactly the settled amount, claim gone.
    assert equity.loc[bars[3]] == pytest.approx(equity.loc[bars[2]], abs=1e-9)
    assert result.ledger.loc[bars[3], "cash"] - result.ledger.loc[bars[2], "cash"] == pytest.approx(50.0, abs=1e-9)
    assert result.ledger.loc[bars[3], "receivables"] == 0.0


# ---------------------------------------------------------------------------
# preserved property 3: zero rates leave price PnL alone
# ---------------------------------------------------------------------------


def test_zero_rates_leave_price_pnl_alone() -> None:
    zero = cost_stack(
        spread={"mode": "flat", "flat_bps": 0.0},
        impact={"quadratic_notional_coeff": 0.0, "participation_model": "sqrt", "participation_coeff": 0.0},
        commission_bps=0.0,
        borrow_bps_annual=0.0,
    )
    result = _run(cost_model=cost_model(zero))
    assert result.costs["total"].abs().max() == pytest.approx(0.0, abs=0.0)
    ledger = result.ledger
    shares, marks = result.shares.fillna(0.0), result.marks
    bars = list(ledger.index)
    for previous, bar in zip(bars[:-1], bars[1:]):
        moved = ledger.loc[bar, "equity_close"] - ledger.loc[previous, "equity_close"]
        carried = sum(
            float(shares.loc[previous, sid]) * (float(marks.loc[bar, sid]) - float(marks.loc[previous, sid]))
            for sid in shares.columns
            if float(shares.loc[previous, sid]) != 0.0
        )
        traded = sum(
            fill.qty * (float(marks.loc[bar, fill.sid]) - fill.fill_price)
            for fill in result.fills[result.fills["bar"] == bar].itertuples()
        )
        assert moved == pytest.approx(carried + traded, abs=1e-9)


# ---------------------------------------------------------------------------
# preserved property 4: a flat book stays flat
# ---------------------------------------------------------------------------


def test_a_flat_book_has_flat_equity() -> None:
    opens, closes, volumes = _panels()
    flat = pd.DataFrame(np.nan, index=list(BARS), columns=list(closes.columns))
    result = simulate(opens, closes, flat, cost_model=cost_model(), volumes=volumes, initial_cash=1234.0)
    assert result.ledger["equity_close"].nunique() == 1
    assert result.ledger["equity_close"].iloc[-1] == pytest.approx(1234.0, abs=0.0)
    assert result.fills.empty
    assert result.costs["total"].abs().max() == 0.0


def test_a_flat_book_with_cash_income_earns_the_declared_rate_and_nothing_else() -> None:
    """The one thing a flat share book *can* do: earn the declared rate on its cash."""
    opens, closes, volumes = _panels()
    flat = pd.DataFrame(np.nan, index=list(BARS), columns=list(closes.columns))
    stack = cost_stack(cash_income_bps_annual=100.0, financing_bps_annual=200.0)
    result = simulate(opens, closes, flat, cost_model=cost_model(stack), volumes=volumes, initial_cash=1000.0)
    per_bar = 100.0 / 1e4 / 252
    expected = 1000.0
    for _ in range(len(BARS)):
        expected += expected * per_bar
    assert result.ledger["equity_close"].iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert result.costs["cash_income"].sum() == pytest.approx(expected - 1000.0, rel=1e-12)


# ---------------------------------------------------------------------------
# not preserved: the cumprod identity
# ---------------------------------------------------------------------------


def test_equity_is_not_a_cumprod_of_returns() -> None:
    """Stated, not assumed: the weight-space identity does not survive share accounting.

    ``tests/test_ledger_conservation.py::test_equity_is_cumprod_of_returns`` holds for the
    reproduction engine because it re-levers to constant weights every bar. A fixed-share
    book with a cash account that pays a declared rate has no such single return series,
    and pretending otherwise would reintroduce the implicit rebalancing §3.6 rejects.
    """
    stack = cost_stack(cash_income_bps_annual=100.0, financing_bps_annual=200.0)
    result = _run(cost_model=cost_model(stack))
    equity = result.ledger["equity_close"]
    returns = equity.pct_change().fillna(0.0)
    positions_return = (result.weights.fillna(0.0).shift(1).fillna(0.0) * result.marks.pct_change().fillna(0.0)).sum(
        axis=1
    )
    assert not np.allclose(returns.to_numpy(), positions_return.to_numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# the Lean bridge: the projection Pi, and the three theorems under it
# ---------------------------------------------------------------------------

TICK = 0.01
INCREMENT = 1.0


def _project(state: PortfolioState, prices: pd.Series, sids: list[str]) -> tuple[int, list[int], list[int]]:
    """``Pi``: PortfolioState -> Book, on the declared share increment and price tick.

    Cash and prices go to integer ticks, shares to integer increments, in a fixed sid
    order -- the ``dot`` of ``Ledger.lean:19-22`` is order-sensitive and the fixed order is
    part of the projection.
    """
    cash = int(round(state.cash / TICK))
    qty = [int(round(state.shares(sid) / INCREMENT)) for sid in sids]
    ticks = [int(round(float(prices[sid]) / TICK)) for sid in sids]
    return cash, qty, ticks


def _wealth(cash: int, qty: list[int], prices: list[int]) -> int:
    """``wealth`` of ``Ledger.lean:29-30``, over Python ints (which are exact)."""
    return cash + sum(q * p for q, p in zip(qty, prices))


def _fixture_state() -> tuple[PortfolioState, pd.Series, list[str]]:
    prices = pd.Series({"AAA": 100.0, "BBB": 25.0, "CCC": 12.5})
    state = state_from_shares("2026-03-02", {"AAA": 10.0, "BBB": -8.0, "CCC": 4.0}, prices, cash=5_000.0)
    return state, prices, ["AAA", "BBB", "CCC"]


def test_projection_is_exact_on_a_whole_tick_book() -> None:
    state, prices, sids = _fixture_state()
    cash, qty, ticks = _project(state, prices, sids)
    assert _wealth(cash, qty, ticks) * TICK == pytest.approx(state.equity(prices), abs=1e-9)


def test_rebalance_conserves_under_the_projection() -> None:
    """``rebalance_conserves``: a fill at fixed prices changes wealth by exactly the costs.

    ``costs`` under ``Pi`` is ``commission + spread_dollars + impact_dollars``, which is
    what the fill records and what cash was debited.
    """
    state, prices, sids = _fixture_state()
    intent = OrderIntent(sid="AAA", qty=5.0, increment=INCREMENT, reference_price=100.0, decision_bar="2026-03-02")
    state.pending_orders.append(
        OrderRecord(intent=intent, client_order_id="2026-03-02:AAA", submitted_at="2026-03-02")
    )
    fill = Fill(
        execution_id="e1",
        client_order_id="2026-03-02:AAA",
        sid="AAA",
        ts="2026-03-03",
        qty=5.0,
        fill_price=100.0,
        reference_price=100.0,
        commission=0.05,
        spread_dollars=0.05,
        impact_dollars=0.10,
    )
    before = _wealth(*_project(state, prices, sids))
    after_state = apply_fills(state, [fill]).state
    after = _wealth(*_project(after_state, prices, sids))
    assert (before - after) * TICK == pytest.approx(fill.charged_cost, abs=1e-9)
    assert after_state.equity(prices) == pytest.approx(state.equity(prices) - fill.charged_cost, abs=1e-9)


def test_mark_to_market_pnl_under_the_projection() -> None:
    """``mark_to_market_pnl``: between marks, wealth moves by held shares times the move."""
    state, prices, sids = _fixture_state()
    moved = pd.Series({"AAA": 101.0, "BBB": 24.5, "CCC": 12.5})
    marked, _ = mark_prices(state, bar="2026-03-03", prices=moved)
    cash, qty, before_ticks = _project(state, prices, sids)
    _, _, after_ticks = _project(marked, moved, sids)
    lean = _wealth(cash, qty, after_ticks) - _wealth(cash, qty, before_ticks)
    assert (marked.equity(moved) - state.equity(prices)) == pytest.approx(lean * TICK, abs=1e-9)
    assert lean * TICK == pytest.approx(10.0 * 1.0 + (-8.0) * (-0.5) + 4.0 * 0.0, abs=1e-9)


def test_run_conserves_over_a_simulated_sequence() -> None:
    """``run_conserves``: final equity is initial equity plus PnL minus charged costs.

    Driven over the simulator rather than over hand-built periods, so the theorem is
    checked against the code that will run in production. PnL is accumulated the way
    ``totalPnl`` does -- per period, on the quantities held into that period.
    """
    result = _run()
    ledger = result.ledger
    shares, marks = result.shares.fillna(0.0), result.marks
    bars = list(ledger.index)
    total_pnl = 0.0
    for previous, bar in zip(bars[:-1], bars[1:]):
        for sid in shares.columns:
            held = float(shares.loc[previous, sid])
            if held != 0.0:
                total_pnl += held * (float(marks.loc[bar, sid]) - float(marks.loc[previous, sid]))
        for fill in result.fills[result.fills["bar"] == bar].itertuples():
            total_pnl += fill.qty * (float(marks.loc[bar, fill.sid]) - fill.fill_price)
    total_costs = float(ledger["trade_cost"].sum() + ledger["holding_cost"].sum())
    final = float(ledger["equity_close"].iloc[-1])
    initial = float(ledger["equity_close"].iloc[0])
    assert final == pytest.approx(initial + total_pnl - total_costs, abs=1e-9)


def test_projection_price_leg_carries_its_tolerance() -> None:
    """``Pi`` is exact on quantities and only tick-exact on prices; this is that gap.

    A price that is not a whole tick projects to the nearest one, so the projected wealth
    differs from the float wealth by at most half a tick per share held. The bridge test
    therefore declares a tolerance on the price leg -- it does not pretend the projection
    is exact, and it does not widen the tolerance on the quantity leg, which is exact.
    """
    prices = pd.Series({"AAA": 100.333333, "BBB": 25.666666})
    state = state_from_shares("2026-03-02", {"AAA": 10.0, "BBB": -8.0}, prices, cash=1_000.0)
    sids = ["AAA", "BBB"]
    projected = _wealth(*_project(state, prices, sids)) * TICK
    gap = abs(projected - state.equity(prices))
    assert gap > 0.0, "the fixture must actually sit off the tick"
    assert gap <= 0.5 * TICK * sum(abs(state.shares(sid)) for sid in sids) + 1e-9


# ---------------------------------------------------------------------------
# the failures the accounting must not absorb
# ---------------------------------------------------------------------------


def test_a_duplicate_fill_changes_the_ledger_not_at_all() -> None:
    """Idempotence on the venue's execution id (§3.11): a replayed page is safe."""
    state, prices, sids = _fixture_state()
    intent = OrderIntent(sid="AAA", qty=5.0, increment=1.0, reference_price=100.0, decision_bar="2026-03-02")
    state.pending_orders.append(
        OrderRecord(intent=intent, client_order_id="2026-03-02:AAA", submitted_at="2026-03-02")
    )
    fill = Fill(
        execution_id="venue-1",
        client_order_id="2026-03-02:AAA",
        sid="AAA",
        ts="2026-03-03",
        qty=5.0,
        fill_price=100.0,
        reference_price=100.0,
        commission=0.05,
    )
    once = apply_fills(state, [fill]).state
    # The order filled completely and left ``pending_orders``; the durable fills ledger is
    # what recognizes the replay, and it passes its execution ids in (§3.11).
    twice = apply_fills(once, [fill], known_executions={"venue-1"})
    assert twice.duplicates == ("venue-1",)
    assert twice.applied == ()
    assert twice.state.cash == pytest.approx(once.cash, abs=0.0)
    assert twice.state.shares("AAA") == pytest.approx(once.shares("AAA"), abs=0.0)


def test_an_expired_remainder_moves_neither_cash_nor_positions() -> None:
    opens, closes, volumes = _panels()
    targets = _targets()
    # A participation cap so tight that only a sliver of each order fills at the open; the
    # opg remainder expires there and must leave the book untouched.
    policy = FillPolicy(model="participation_capped", max_participation_of_bar_volume=1e-6)
    result = simulate(
        opens,
        closes,
        targets,
        cost_model=cost_model(),
        volumes=volumes,
        initial_cash=100_000.0,
        fills=policy,
    )
    expired = [order for order in result.orders if order.status == "expired"]
    assert expired, "the fixture must actually expire a remainder"
    for order in expired:
        assert abs(order.filled_qty) < abs(order.intent.qty)
    rebuilt = _reconstruct_equity_path(result)
    pd.testing.assert_series_equal(result.ledger["equity_close"], rebuilt, rtol=0.0, atol=1e-9)


def test_a_missing_bar_leaves_the_holding_stale_and_never_at_zero() -> None:
    opens, closes, volumes = _panels()
    targets = _targets()
    closes.loc[BARS[4], "BBB"] = np.nan
    opens.loc[BARS[4], "BBB"] = np.nan
    volumes.loc[BARS[4], "BBB"] = np.nan
    result = simulate(opens, closes, targets, cost_model=cost_model(), volumes=volumes, initial_cash=100_000.0)
    assert result.ledger.loc[BARS[4], "stale"] == 1
    held = float(result.shares.loc[BARS[4], "BBB"])
    assert held != 0.0
    carried = float(result.marks.loc[BARS[4], "BBB"])
    assert carried == pytest.approx(float(result.marks.loc[BARS[3], "BBB"]), abs=0.0)
    rebuilt = _reconstruct_equity_path(result)
    pd.testing.assert_series_equal(result.ledger["equity_close"], rebuilt, rtol=0.0, atol=1e-9)


def test_a_fill_without_a_submitted_order_is_refused() -> None:
    state, _, _ = _fixture_state()
    orphan = Fill(
        execution_id="e9",
        client_order_id="never-submitted",
        sid="AAA",
        ts="2026-03-03",
        qty=1.0,
        fill_price=100.0,
        reference_price=100.0,
    )
    with pytest.raises(Exception, match="does not carry"):
        apply_fills(state, [orphan])


def test_an_over_fill_is_refused() -> None:
    state, _, _ = _fixture_state()
    intent = OrderIntent(sid="AAA", qty=5.0, increment=1.0, reference_price=100.0, decision_bar="2026-03-02")
    state.pending_orders.append(
        OrderRecord(intent=intent, client_order_id="2026-03-02:AAA", submitted_at="2026-03-02")
    )
    too_big = Fill(
        execution_id="e1",
        client_order_id="2026-03-02:AAA",
        sid="AAA",
        ts="2026-03-03",
        qty=6.0,
        fill_price=100.0,
        reference_price=100.0,
    )
    with pytest.raises(Exception, match="over-fill"):
        apply_fills(state, [too_big])


def test_holding_cost_settlement_moves_cash_but_never_equity() -> None:
    """``per_bar`` and ``monthly`` agree on equity at every close and differ in cash (§3.6)."""
    bars = ["2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02"]
    opens = pd.DataFrame({"AAA": [100.0] * 4}, index=bars)
    closes = pd.DataFrame({"AAA": [100.0] * 4}, index=bars)
    targets = pd.DataFrame(np.nan, index=bars, columns=["AAA"])
    targets.loc[bars[0]] = [-0.50]
    per_bar = simulate(
        opens,
        closes,
        targets,
        cost_model=cost_model(cost_stack(holding_cost_settlement="per_bar")),
        initial_cash=10_000.0,
    )
    monthly = simulate(
        opens,
        closes,
        targets,
        cost_model=cost_model(cost_stack(holding_cost_settlement="monthly")),
        initial_cash=10_000.0,
    )
    pd.testing.assert_series_equal(per_bar.ledger["equity_close"], monthly.ledger["equity_close"], rtol=0.0, atol=1e-9)
    accrual = 5_000.0 * 50.0 / 1e4 / 252.0  # 50 bps annual on a 5,000 short, one bar
    assert monthly.ledger.loc[bars[1], "cash"] > per_bar.ledger.loc[bars[1], "cash"]
    # March 31 carries one bar of accrual; April 1 settles that carry and then accrues its
    # own bar, so the liability is one bar again and cash has fallen by exactly the carry.
    assert monthly.ledger.loc[bars[1], "liabilities"] == pytest.approx(accrual, rel=1e-12)
    assert monthly.ledger.loc[bars[2], "liabilities"] == pytest.approx(accrual, rel=1e-12)
    assert monthly.ledger.loc[bars[3], "liabilities"] == pytest.approx(2.0 * accrual, rel=1e-12)
    assert monthly.ledger.loc[bars[2], "cash"] == pytest.approx(
        monthly.ledger.loc[bars[1], "cash"] - accrual, rel=1e-12
    )
    # per_bar has debited every bar; monthly has debited only the one month boundary.
    assert per_bar.ledger.loc[bars[3], "cash"] == pytest.approx(
        monthly.ledger.loc[bars[3], "cash"] - 2.0 * accrual, rel=1e-12
    )


def test_the_cost_terms_the_breakdown_declares_are_the_seven_the_spec_names() -> None:
    assert COST_TERMS == (
        "commission",
        "spread",
        "impact_quadratic",
        "impact_participation",
        "borrow",
        "financing",
        "cash_income",
    )


def test_a_decision_sizes_against_the_position_the_book_already_implies() -> None:
    """A pending order is part of the implied position, so it is not ordered twice.

    Sizing the delta against held shares alone would submit the same trade again while the
    first is still at the venue -- the double-position failure that is invisible until the
    second fill arrives.
    """
    from prism.execution.accounting import decide, submit

    prices = pd.Series({"AAA": 100.0})
    state = state_from_shares("2026-03-02", {"AAA": 20.0}, prices, cash=8_000.0)
    first = decide(state, pd.Series({"AAA": 0.50}), prices, decision_bar="2026-03-02")
    assert [(i.sid, i.qty) for i in first.intents] == [("AAA", 30.0)]
    state = submit(state, first.intents, submitted_at="2026-03-02")
    # the same target, decided again with the first order still outstanding
    second = decide(state, pd.Series({"AAA": 0.50}), prices, decision_bar="2026-03-03")
    assert second.intents == ()
    assert float(second.quantities.loc[0, "outstanding_shares"]) == 30.0


# ---------------------------------------------------------------------------
# the declared [fills] block: partial fills, rejections, and a missing open
# ---------------------------------------------------------------------------


def test_a_partial_fill_fraction_leaves_a_remainder_and_still_conserves() -> None:
    opens, closes, volumes = _panels()
    result = simulate(
        opens,
        closes,
        _targets(),
        cost_model=cost_model(),
        volumes=volumes,
        initial_cash=100_000.0,
        fills=FillPolicy(partial_fill_fraction=0.5),
    )
    filled = result.fills.set_index("client_order_id")["qty"]
    decided = {order.client_order_id: order.intent.qty for order in result.orders}
    for client_order_id, quantity in filled.items():
        assert abs(quantity) < abs(decided[client_order_id])
        assert quantity == pytest.approx(np.round(decided[client_order_id] * 0.5), abs=0.0)
    pd.testing.assert_series_equal(
        result.ledger["equity_close"], _reconstruct_equity_path(result), rtol=0.0, atol=1e-9
    )


def test_a_simulated_rejection_is_deterministic_and_creates_no_fill() -> None:
    """The rejection path is exercised against fixtures, so its draw is a hash, not a stream."""
    opens, closes, volumes = _panels()
    runs = [
        simulate(
            opens,
            closes,
            _targets(),
            cost_model=cost_model(),
            volumes=volumes,
            initial_cash=100_000.0,
            fills=FillPolicy(reject_probability=0.5, seed=7),
        )
        for _ in range(2)
    ]
    rejected = [{o.client_order_id for o in run.orders if o.status == "rejected"} for run in runs]
    assert rejected[0] == rejected[1] and rejected[0], "the fixture must reject something, repeatably"
    filled = set(runs[0].fills["client_order_id"])
    assert not (filled & rejected[0]), "a rejected order never fills"
    pd.testing.assert_series_equal(
        runs[0].ledger["equity_close"], _reconstruct_equity_path(runs[0]), rtol=0.0, atol=1e-9
    )


def test_a_bar_that_traded_without_an_open_fills_at_the_declared_reference_price() -> None:
    """``missing_open_policy = "reference_price"`` -- the one supported value (§3.6)."""
    bars = BARS[:3]
    opens = pd.DataFrame({"AAA": [100.0, np.nan, 100.0]}, index=bars)
    closes = pd.DataFrame({"AAA": [100.0, 101.0, 100.0]}, index=bars)
    volumes = pd.DataFrame({"AAA": [1e6, 1e6, 1e6]}, index=bars)
    targets = pd.DataFrame(np.nan, index=bars, columns=["AAA"])
    targets.loc[bars[0]] = [0.50]
    result = simulate(opens, closes, targets, cost_model=cost_model(), volumes=volumes, initial_cash=10_000.0)
    assert len(result.fills) == 1
    fill = result.fills.iloc[0]
    assert fill["bar"] == bars[1]
    assert fill["fill_price"] == pytest.approx(fill["reference_price"], abs=0.0) == 100.0


def test_the_state_round_trips_through_its_durable_form() -> None:
    """``LoopState`` schema version 3 persists this state (decision D4), so it must survive
    a JSON round trip byte for byte -- positions, claims, and pending orders with their
    client order ids and the fills already applied to them."""
    import json

    state = state_from_shares(
        "2026-03-02", {"AAA": 10.0, "BBB": -5.0}, pd.Series({"AAA": 100.0, "BBB": 50.0}), cash=1_000.0
    )
    state.add_receivable("cash_distribution:AAA:2026-03-02", 12.5)
    state.add_liability("accrued_borrow", 0.25)
    intent = OrderIntent(sid="AAA", qty=5.0, increment=1.0, reference_price=100.0, decision_bar="2026-03-02")
    record = OrderRecord(intent=intent, client_order_id="2026-03-02:AAA", submitted_at="2026-03-02")
    record.fills.append(
        Fill(
            execution_id="e1",
            client_order_id="2026-03-02:AAA",
            sid="AAA",
            ts="2026-03-03",
            qty=2.0,
            fill_price=100.5,
            reference_price=100.0,
            commission=0.01,
        )
    )
    record.filled_qty = 2.0
    record.status = "partial"
    state.pending_orders.append(record)

    blob = json.dumps(state.to_dict(), sort_keys=True)
    restored = PortfolioState.from_dict(json.loads(blob))
    assert json.dumps(restored.to_dict(), sort_keys=True) == blob
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0})
    assert restored.equity(prices) == state.equity(prices)
    assert restored.committed_cash() == state.committed_cash() == 300.0
