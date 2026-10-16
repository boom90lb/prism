"""The buying-power pre-check (``decide_and_submit(buying_power_check=True)``, ``prism.live.buying_power``).

Pinned here:

* what counts as opening: a buy beyond a held short, a sale beyond a held long;
* a batch that fits is returned as the same list, untouched;
* a batch that does not fit has every opening quantity scaled by one factor,
  rounded down so it fits, with closing orders unchanged and dust dropped;
* zero, negative and non-finite buying power;
* in ``decide_and_submit``: off reads nothing and decides as before; on and not
  binding, the persisted decision is the unchecked one; on and binding, the
  persisted and submitted orders are the fitted ones, logged at WARNING; a
  resumed decision is not re-checked; ``run_daily_cycle`` passes the flag.

Nothing here opens a socket.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
import pytest

from prism.live import (
    DailyBookConfig,
    LiveLoopContext,
    StateStore,
    decide_and_submit,
    run_daily_cycle,
)
from prism.live.broker import BrokerCapabilityUnsupported, Order
from prism.live.buying_power import (
    BuyingPowerError,
    fit_opening_orders,
    opening_notional,
    opening_shares,
)
from tests.test_live_daily import ConstSignal
from tests.test_live_loop import FakeBroker

BAR = "2026-03-04"


def _order(symbol: str, qty: float, price: float) -> Order:
    return Order(f"{BAR}:{symbol}", symbol, qty, BAR, price)


class MarginBroker(FakeBroker):
    """A fake venue that reports a settable buying power and counts the reads."""

    def __init__(self, cash: float = 10_000.0, buying_power: float = 1e12) -> None:
        super().__init__(cash=cash)
        self.reported = buying_power
        self.reads = 0

    def buying_power(self) -> float:
        self.reads += 1
        return self.reported


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_what_counts_as_opening() -> None:
    assert opening_shares(_order("A", 10, 1.0), 0.0) == 10  # a new long
    assert opening_shares(_order("A", 10, 1.0), 5.0) == 10  # adds to a long
    assert opening_shares(_order("A", 10, 1.0), -4.0) == 6  # covers 4 of a short, then opens 6 (not sized so)
    assert opening_shares(_order("A", 10, 1.0), -10.0) == 0  # a cover only
    assert opening_shares(_order("A", -10, 1.0), 0.0) == 10  # a new short
    assert opening_shares(_order("A", -10, 1.0), 10.0) == 0  # sells a long
    assert opening_shares(_order("A", -10, 1.0), -3.0) == 10  # extends a short
    orders = [_order("A", 10, 2.0), _order("B", -5, 4.0)]
    assert opening_notional(orders, {"B": 5.0}) == pytest.approx(20.0)


ORDERS = [
    _order("NEWLONG", 100, 10.0),  # opens 1,000
    _order("SOLD", -10, 10.0),  # closes a long
    _order("COVER", 5, 10.0),  # closes a short
    _order("NEWSHORT", -50, 20.0),  # opens 1,000
]
POSITIONS = {"SOLD": 10.0, "COVER": -5.0}


def test_a_batch_that_fits_is_the_same_list() -> None:
    fitted, fit = fit_opening_orders(ORDERS, POSITIONS, 2_000.0, whole_shares=True)
    assert fitted is ORDERS
    assert not fit.binding and fit.factor == 1.0 and fit.required == pytest.approx(2_000.0)
    assert fit.scaled == () and fit.dropped == ()


def test_a_batch_that_does_not_fit_scales_every_opening_order_by_one_factor() -> None:
    fitted, fit = fit_opening_orders(ORDERS, POSITIONS, 1_000.0, whole_shares=True)
    assert fit.binding and fit.factor == pytest.approx(0.5)
    by_symbol = {order.symbol: order for order in fitted}
    assert by_symbol["NEWLONG"].qty == 50 and by_symbol["NEWSHORT"].qty == -25
    # Closing orders are the same objects, and nothing else about an order changes.
    assert by_symbol["SOLD"] is ORDERS[1] and by_symbol["COVER"] is ORDERS[2]
    assert by_symbol["NEWLONG"].client_order_id == ORDERS[0].client_order_id
    assert by_symbol["NEWLONG"].reference_price == ORDERS[0].reference_price
    assert [order.symbol for order in fitted] == [order.symbol for order in ORDERS]
    assert fit.fitted == pytest.approx(1_000.0) and fit.fitted <= fit.buying_power
    assert set(fit.scaled) == {ORDERS[0].client_order_id, ORDERS[3].client_order_id}


def test_whole_shares_round_down_so_the_fitted_batch_fits() -> None:
    fitted, fit = fit_opening_orders(ORDERS, POSITIONS, 700.0, whole_shares=True)
    assert fit.factor == pytest.approx(0.35)
    by_symbol = {order.symbol: order for order in fitted}
    assert by_symbol["NEWLONG"].qty == 35 and by_symbol["NEWSHORT"].qty == -17  # floor(17.5)
    assert fit.fitted == pytest.approx(35 * 10.0 + 17 * 20.0) and fit.fitted <= 700.0


def test_fractional_quantities_round_down_to_nine_decimals() -> None:
    orders = [_order("A", 3.0, 7.0), _order("B", -1.0, 11.0)]
    fitted, fit = fit_opening_orders(orders, {}, 10.0, whole_shares=False)
    assert fit.binding and fit.fitted <= 10.0
    for before, after in zip(orders, fitted):
        assert abs(after.qty) == pytest.approx(abs(before.qty) * fit.factor, abs=1e-8)
        assert round(abs(after.qty), 9) == abs(after.qty)


def test_no_buying_power_leaves_only_the_closing_orders_and_dust_is_dropped() -> None:
    for available in (0.0, -50.0):
        fitted, fit = fit_opening_orders(ORDERS, POSITIONS, available, whole_shares=True)
        assert fitted == [ORDERS[1], ORDERS[2]] and fit.buying_power == 0.0 and fit.factor == 0.0
        assert set(fit.dropped) == {ORDERS[0].client_order_id, ORDERS[3].client_order_id}
    # A scaled order below the notional floor is dropped as sizing drops dust.
    fitted, fit = fit_opening_orders([_order("A", 10, 10.0)], {}, 15.0, whole_shares=True, min_order_notional=20.0)
    assert fitted == [] and fit.dropped == (f"{BAR}:A",)
    with pytest.raises(BuyingPowerError):
        fit_opening_orders(ORDERS, POSITIONS, float("nan"), whole_shares=True)


# ---------------------------------------------------------------------------
# In the write-ahead step
# ---------------------------------------------------------------------------


PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0})
TARGETS = pd.Series({"AAA": 0.5, "BBB": -0.5})


def _ctx(root: Path, broker) -> LiveLoopContext:
    root.mkdir(parents=True, exist_ok=True)
    return LiveLoopContext(store=StateStore(root / "state.json"), broker=broker, fills_ledger=root / "fills.jsonl")


def test_off_reads_nothing(tmp_path: Path) -> None:
    class Unsupported(FakeBroker):
        def buying_power(self) -> float:  # the base class raises too; this makes the read itself the failure
            raise BrokerCapabilityUnsupported("read")

    plain = decide_and_submit(_ctx(tmp_path / "a", FakeBroker()), BAR, TARGETS, PRICES, whole_shares=True)
    off = decide_and_submit(_ctx(tmp_path / "b", Unsupported()), BAR, TARGETS, PRICES, whole_shares=True)
    assert plain == off and len(plain) == 2


def test_on_and_not_binding_decides_exactly_the_unchecked_batch(tmp_path: Path) -> None:
    broker = MarginBroker(buying_power=20_000.0)  # the batch opens 10,000
    checked = decide_and_submit(_ctx(tmp_path / "on", broker), BAR, TARGETS, PRICES, whole_shares=True,
                                buying_power_check=True)
    unchecked = decide_and_submit(_ctx(tmp_path / "off", FakeBroker()), BAR, TARGETS, PRICES, whole_shares=True)
    assert broker.reads == 1 and checked == unchecked
    assert StateStore(tmp_path / "on" / "state.json").load() == StateStore(tmp_path / "off" / "state.json").load()


def test_on_and_binding_persists_and_submits_the_fitted_batch(tmp_path: Path, caplog) -> None:
    broker = MarginBroker(buying_power=4_000.0)  # 40% of the 10,000 the batch opens
    caplog.set_level(logging.WARNING, logger="prism.live.loop")
    orders = decide_and_submit(_ctx(tmp_path, broker), BAR, TARGETS, PRICES, whole_shares=True,
                               buying_power_check=True)
    assert {o.symbol: o.qty for o in orders} == {"AAA": 20.0, "BBB": -40.0}  # 50 and -100 shares, times 0.4
    assert {o.client_order_id: o.qty for o in broker.orders.values()} == {o.client_order_id: o.qty for o in orders}
    persisted = StateStore(tmp_path / "state.json").load()
    assert [(o.symbol, o.qty) for o in persisted.pending_orders] == [(o.symbol, o.qty) for o in orders]
    warnings = [r.getMessage() for r in caplog.records if "BUYING POWER" in r.getMessage()]
    assert len(warnings) == 1 and "0.4000" in warnings[0]
    # A same-bar resume submits the persisted decision and does not read buying power again.
    again = decide_and_submit(_ctx(tmp_path, broker), BAR, TARGETS, PRICES, whole_shares=True,
                              buying_power_check=True)
    assert again == orders and broker.reads == 1


def test_the_daily_cycle_passes_the_flag(tmp_path: Path) -> None:
    index = pd.date_range("2026-05-01", periods=5, freq="B", tz="America/New_York")
    close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=index)
    volume = pd.DataFrame(1e6, index=index, columns=close.columns)
    config = DailyBookConfig(position_size=0.5, max_symbol_abs_weight=0.5, min_order_notional=1.0)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    results = {}
    for name, flag in (("off", False), ("on", True)):
        broker = MarginBroker(buying_power=2_500.0)
        results[name] = (run_daily_cycle(_ctx(tmp_path / name, broker), signal, close, volume, config,
                                         buying_power_check=flag), broker)
    (off, off_broker), (on, on_broker) = results["off"], results["on"]
    assert off_broker.reads == 0 and on_broker.reads == 1
    assert {o.symbol: o.qty for o in off.submitted_orders} == {"AAA": 50.0, "BBB": -100.0}
    assert {o.symbol: o.qty for o in on.submitted_orders} == {"AAA": 12.0, "BBB": -25.0}  # factor 0.25
    assert math.isclose(sum(abs(o.qty) * o.reference_price for o in on.submitted_orders), 2_450.0)
