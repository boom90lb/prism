"""Reconciliation, and the line between what gates and what is only recorded (spec §3.11).

Two failures are worth more than the rest of this file put together, and both
are asymmetric:

* A routine partial fill leaves the book one share short. Refusing to trade over
  it would freeze a real account for a month. So a position or cash difference
  is adopted and the cycle continues.
* An order the venue holds at a different quantity, or on the other side, is two
  systems disagreeing about what was submitted. Adopting it would mean trading a
  quantity nobody decided. So it blocks, and no tolerance widens it.

The telemetry comparisons are tested for the property that they never gate, which
is the one thing that could quietly turn a cost model into a kill switch.
"""

from __future__ import annotations

import pytest

from prism.execution.portfolio_state import (
    Fill,
    OrderIntent,
    OrderRecord,
    PortfolioState,
    state_from_shares,
)
from prism.live.broker import Broker, BrokerCapabilityUnsupported, FillRecord, OrderStatus
from prism.live.reconcile import (
    ACCOUNTING,
    STATUS_CLEAN,
    STATUS_DISCREPANCY,
    STATUS_UNKNOWN_ORDER_STATE,
    TELEMETRY,
    ReconcileSettings,
    reconcile,
    reconcile_loop_state,
    telemetry_reads,
)
from prism.live.state import LoopState

BAR = "2026-07-06"
PRICES = {"AAA": 100.0, "BBB": 50.0}


def _book(shares=None, cash: float = 1_000.0) -> PortfolioState:
    return state_from_shares(BAR, shares if shares is not None else {"AAA": 10.0}, PRICES, cash=cash)


def _record(client_order_id: str = "mom:2026-07-06:AAA", qty: float = 5.0, sid: str = "AAA") -> OrderRecord:
    return OrderRecord(
        intent=OrderIntent(sid=sid, qty=qty, increment=1.0, reference_price=PRICES[sid], decision_bar=BAR),
        client_order_id=client_order_id,
    )


def _venue_order(client_order_id: str, sid: str = "AAA", qty: float = 5.0, filled: float = 0.0) -> OrderStatus:
    return OrderStatus(
        client_order_id=client_order_id,
        sid=sid,
        qty=qty,
        filled_qty=filled,
        order_type="market",
        time_in_force="opg",
        status="accepted",
    )


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def test_an_agreeing_book_is_clean() -> None:
    report = reconcile(_book(), broker_positions={"AAA": 10.0}, broker_cash=1_000.0)

    assert report.status == STATUS_CLEAN
    assert report.items == () and report.may_trade
    assert report.to_dict()["status"] == "clean"


def test_float_noise_below_the_declared_tolerance_is_not_a_divergence() -> None:
    report = reconcile(_book(), broker_positions={"AAA": 10.0 + 1e-9}, broker_cash=1_000.004)

    assert report.status == STATUS_CLEAN


# ---------------------------------------------------------------------------
# positions and cash: adopt broker truth, record what was absorbed
# ---------------------------------------------------------------------------


def test_a_position_difference_is_adopted_recorded_and_does_not_stop_trading() -> None:
    report = reconcile(_book(), broker_positions={"AAA": 9.0}, broker_cash=1_100.0)

    assert report.status == STATUS_DISCREPANCY
    assert report.may_trade, "a routine partial fill must not stop a book from rebalancing"
    assert report.blocking_items == ()
    positions = [item for item in report.items if item.check == "positions"]
    assert len(positions) == 1
    item = positions[0]
    assert (item.sid, item.engine, item.observed, item.difference) == ("AAA", 10.0, 9.0, -1.0)
    assert item.cls == ACCOUNTING and item.adopted
    assert report.state is not None and report.state.shares("AAA") == 9.0
    assert report.state.cash == 1_100.0


def test_a_holding_the_book_does_not_carry_is_adopted_unmarked_not_at_zero() -> None:
    report = reconcile(_book(), broker_positions={"AAA": 10.0, "CCC": 4.0}, broker_cash=1_000.0)

    assert report.state is not None
    carried = report.state.positions["CCC"]
    assert carried.shares == 4.0
    assert carried.last_price is None and carried.valuation_status == "stale"
    assert report.state.stale_marks() == ["CCC"]


def test_a_holding_the_venue_no_longer_carries_leaves_nothing_behind() -> None:
    report = reconcile(_book({"AAA": 10.0, "BBB": -2.0}), broker_positions={"AAA": 10.0}, broker_cash=1_000.0)

    assert report.state is not None and list(report.state.positions) == ["AAA"]
    assert [item.sid for item in report.items] == ["BBB"]
    assert report.items[0].observed == 0.0 and report.items[0].engine == -2.0


def test_the_original_state_is_not_mutated_by_a_pass() -> None:
    book = _book()

    report = reconcile(book, broker_positions={"AAA": 3.0}, broker_cash=7.0)

    assert book.shares("AAA") == 10.0 and book.cash == 1_000.0
    assert report.state is not None and report.state.shares("AAA") == 3.0


# ---------------------------------------------------------------------------
# orders: not adoptable
# ---------------------------------------------------------------------------


def test_an_open_order_at_a_different_quantity_blocks() -> None:
    book = _book()
    book.pending_orders.append(_record(qty=5.0))

    report = reconcile(
        book,
        broker_positions={"AAA": 10.0},
        broker_cash=1_000.0,
        open_orders=[_venue_order("mom:2026-07-06:AAA", qty=8.0)],
    )

    assert report.status == STATUS_DISCREPANCY
    assert not report.may_trade, "a submitted quantity is not adoptable (S7, reconcile only)"
    item = report.blocking_items[0]
    assert item.check == "open_orders" and item.cls == ACCOUNTING
    assert (item.engine, item.observed, item.difference) == (5.0, 8.0, 3.0)


def test_an_open_order_on_the_other_side_blocks_and_says_so() -> None:
    book = _book()
    book.pending_orders.append(_record(qty=5.0))

    report = reconcile(
        book,
        broker_positions={"AAA": 10.0},
        broker_cash=1_000.0,
        open_orders=[_venue_order("mom:2026-07-06:AAA", qty=-5.0)],
    )

    assert not report.may_trade
    assert "opposite sides" in report.blocking_items[0].detail


def test_an_order_at_the_venue_the_book_never_submitted_blocks() -> None:
    """This is how one host's orders being substituted for another's surfaces (§3.10)."""
    report = reconcile(
        _book(),
        broker_positions={"AAA": 10.0},
        broker_cash=1_000.0,
        open_orders=[_venue_order("other-host:2026-07-06:AAA")],
    )

    assert not report.may_trade
    assert "no record of submitting" in report.blocking_items[0].detail


def test_one_client_order_id_naming_two_names_blocks() -> None:
    book = _book()
    book.pending_orders.append(_record(sid="AAA"))

    report = reconcile(
        book,
        broker_positions={"AAA": 10.0},
        broker_cash=1_000.0,
        open_orders=[_venue_order("mom:2026-07-06:AAA", sid="BBB")],
    )

    assert not report.may_trade
    assert "names two names" in report.blocking_items[0].detail


def test_an_order_with_no_reported_fate_is_unknown_order_state() -> None:
    book = _book()
    book.pending_orders.append(_record())

    report = reconcile(book, broker_positions={"AAA": 10.0}, broker_cash=1_000.0, open_orders=[])

    assert report.status == STATUS_UNKNOWN_ORDER_STATE
    assert not report.may_trade
    assert report.items[0].client_order_id == "mom:2026-07-06:AAA"


def test_an_order_the_venue_declares_terminal_is_not_unknown() -> None:
    book = _book()
    book.pending_orders.append(_record())

    report = reconcile(
        book,
        broker_positions={"AAA": 10.0},
        broker_cash=1_000.0,
        open_orders=[],
        terminal_order_ids=["mom:2026-07-06:AAA"],
    )

    assert report.status == STATUS_CLEAN


def test_an_order_that_already_printed_an_execution_is_not_unknown() -> None:
    book = _book()
    record = _record()
    record.status = "partial"
    record.filled_qty = 2.0
    record.fills.append(
        Fill(
            execution_id="exec-1",
            client_order_id=record.client_order_id,
            sid="AAA",
            ts="2026-07-07T09:30:00-04:00",
            qty=2.0,
            fill_price=100.5,
            reference_price=100.0,
            source="broker",
        )
    )
    book.pending_orders.append(record)

    report = reconcile(book, broker_positions={"AAA": 10.0}, broker_cash=1_000.0, open_orders=[])

    assert report.status == STATUS_CLEAN


def test_an_adapter_that_cannot_list_open_orders_is_not_read_as_reporting_none() -> None:
    book = _book()
    book.pending_orders.append(_record())

    report = reconcile(
        book,
        broker_positions={"AAA": 10.0},
        broker_cash=1_000.0,
        open_orders=None,
        open_orders_available=False,
    )

    assert report.status == STATUS_UNKNOWN_ORDER_STATE
    assert "different facts" in report.items[0].detail


def test_no_pending_orders_and_no_capability_is_still_clean() -> None:
    report = reconcile(
        _book(),
        broker_positions={"AAA": 10.0},
        broker_cash=1_000.0,
        open_orders=None,
        open_orders_available=False,
    )

    assert report.status == STATUS_CLEAN


# ---------------------------------------------------------------------------
# fills against the ledger
# ---------------------------------------------------------------------------


def _execution(execution_id: str = "exec-1") -> FillRecord:
    return FillRecord(
        execution_id=execution_id,
        client_order_id="mom:2026-07-06:AAA",
        sid="AAA",
        qty=5.0,
        price=100.5,
        filled_at="2026-07-07T09:30:00-04:00",
    )


def test_an_execution_the_ledger_does_not_carry_blocks() -> None:
    report = reconcile(
        _book(),
        broker_positions={"AAA": 10.0},
        broker_cash=1_000.0,
        open_orders=[],
        fills=[_execution()],
        ledger_execution_ids=[],
    )

    assert not report.may_trade
    assert report.blocking_items[0].check == "fills"


def test_a_re_pulled_execution_the_ledger_already_holds_is_clean() -> None:
    """Append-then-advance re-pulls after a crash; the ledger key drops the duplicate (§3.11)."""
    report = reconcile(
        _book(),
        broker_positions={"AAA": 10.0},
        broker_cash=1_000.0,
        open_orders=[],
        fills=[_execution()],
        ledger_execution_ids=["exec-1"],
    )

    assert report.status == STATUS_CLEAN


def test_an_execution_without_an_execution_id_cannot_be_reconciled() -> None:
    with pytest.raises(ValueError, match="execution_id"):
        reconcile(
            _book(),
            broker_positions={"AAA": 10.0},
            broker_cash=1_000.0,
            fills=[{"client_order_id": "x", "qty": 1.0, "price": 2.0}],
            ledger_execution_ids=[],
        )


# ---------------------------------------------------------------------------
# telemetry gates nothing
# ---------------------------------------------------------------------------


def test_every_telemetry_miss_is_recorded_and_none_of_them_gate() -> None:
    report = reconcile(
        _book(),
        broker_positions={"AAA": 10.0},
        broker_cash=1_000.0,
        open_orders=[],
        modeled_cost_dollars=10.0,
        observed_cost_dollars=300.0,  # 58 bps of the traded notional, past the 25 bps tolerance
        traded_notional=50_000.0,
        realized_turnover=0.40,
        evaluated_turnover=0.10,
        realized_variance=0.05,
        variance_budget=0.02,
    )

    checks = {item.check for item in report.telemetry_items}
    assert checks == {"execution_cost", "turnover", "variance"}
    assert all(item.cls == TELEMETRY and not item.blocking for item in report.telemetry_items)
    assert report.status == STATUS_DISCREPANCY
    assert report.may_trade, "a cost, turnover or variance miss never changes what the engine will do"


def test_a_telemetry_comparison_with_a_missing_input_is_not_run() -> None:
    """A missing number is not agreement, and it is not a zero either."""
    assert telemetry_reads(modeled_cost_dollars=10.0) == []
    assert telemetry_reads(realized_turnover=0.4) == []
    assert telemetry_reads(realized_variance=0.5) == []


def test_a_cost_miss_inside_the_declared_tolerance_is_not_reported() -> None:
    settings = ReconcileSettings(cost_tolerance_bps=25.0)

    inside = telemetry_reads(
        modeled_cost_dollars=100.0,
        observed_cost_dollars=200.0,
        traded_notional=1_000_000.0,
        settings=settings,
    )
    outside = telemetry_reads(
        modeled_cost_dollars=100.0,
        observed_cost_dollars=3_000.0,
        traded_notional=1_000_000.0,
        settings=settings,
    )

    assert inside == []
    assert len(outside) == 1 and "bps" in outside[0].detail


def test_realized_variance_under_the_budget_is_not_a_finding() -> None:
    assert telemetry_reads(realized_variance=0.01, variance_budget=0.02) == []
    assert len(telemetry_reads(realized_variance=0.03, variance_budget=0.02)) == 1


# ---------------------------------------------------------------------------
# against a broker, in place
# ---------------------------------------------------------------------------


class FakeVenue(Broker):
    def __init__(self, positions, cash, open_orders=None, fills=None, supports=("open_orders", "fills_since")):
        self._positions = dict(positions)
        self._cash = float(cash)
        self._open = list(open_orders or [])
        self._fills = list(fills or [])
        self._supports = set(supports)

    def positions(self):
        return dict(self._positions)

    def cash(self):
        return self._cash

    def submit(self, order):  # pragma: no cover - not exercised here
        raise NotImplementedError

    def submitted_order_ids(self):  # pragma: no cover - not exercised here
        return set()

    def fills_for(self, client_order_ids):  # pragma: no cover - not exercised here
        return []

    def open_orders(self):
        if "open_orders" not in self._supports:
            raise BrokerCapabilityUnsupported("no open_orders")
        return list(self._open)

    def fills_since(self, marker):
        if "fills_since" not in self._supports:
            raise BrokerCapabilityUnsupported("no fills_since")
        return list(self._fills), "cursor-2"


def test_reconciling_a_loop_state_adopts_into_the_portfolio_and_leaves_the_marker_alone() -> None:
    state = LoopState(_book(), fills_marker="cursor-1")
    venue = FakeVenue({"AAA": 9.0}, 1_050.0, open_orders=[], fills=[_execution()])

    report = reconcile_loop_state(state, venue, ledger_execution_ids=["exec-1"])

    assert report.status == STATUS_DISCREPANCY and report.may_trade
    assert state.positions == {"AAA": 9.0} and state.cash == 1_050.0
    # append-then-advance: only the caller that appended the rows advances it.
    assert state.fills_marker == "cursor-1"


def test_a_broker_without_the_open_orders_capability_leaves_a_pending_order_unknown() -> None:
    book = _book()
    book.pending_orders.append(_record())
    state = LoopState(book)
    venue = FakeVenue({"AAA": 10.0}, 1_000.0, supports=("fills_since",))

    report = reconcile_loop_state(state, venue)

    assert report.status == STATUS_UNKNOWN_ORDER_STATE


def test_a_broker_without_fills_since_still_reconciles_the_book() -> None:
    state = LoopState(_book())
    venue = FakeVenue({"AAA": 10.0}, 1_000.0, open_orders=[], supports=("open_orders",))

    report = reconcile_loop_state(state, venue)

    assert report.status == STATUS_CLEAN
