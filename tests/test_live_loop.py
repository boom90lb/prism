"""live/ durable-state protocol (SPEC §7.7, the R2 paper-instrument core).

Pins the crash window the module exists for: orders decided at close *t*
survive a process restart between decision and fill — resumed, never lost,
never double-submitted, never re-decided — and every fill lands in the
append-only I-9 ledger beside its decision-close reference price.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prism.live import (
    Broker,
    DuplicateOrder,
    Fill,
    LiveLoopContext,
    LoopState,
    Order,
    OrderRejected,
    StateStore,
    decide_and_submit,
    read_fills_ledger,
    read_targets_ledger,
    settle,
    sweep_pending,
    targets_to_orders,
)


class FakeBroker(Broker):
    """In-memory venue: idempotent ids, fills at a settable open price."""

    def __init__(self, cash: float = 10_000.0) -> None:
        self._cash = cash
        self._positions: dict[str, float] = {}
        self.orders: dict[str, Order] = {}
        self.submit_calls = 0
        self.fail_after_n_submits: int | None = None
        self.open_prices: dict[str, float] = {}
        # Venue outcomes the tests exercise for the tolerant settle: ids that
        # never fill (e.g. an OPG auction that did not print), and ids that
        # fill only partially (id -> executed qty, less than the order qty).
        self.no_fill: set[str] = set()
        self.partial_fills: dict[str, float] = {}
        # Ids the venue rejects at submit with a non-Duplicate error (e.g. a
        # zero-crossing order): the loop must skip and continue, not crash.
        self.reject_ids: set[str] = set()
        # Ids still LIVE at the venue (auction not yet run / order working):
        # the sweep must never touch these. Executed fills are cached so a
        # repeated fills_for query (sweep + settle both ask) is idempotent,
        # matching the real venue where querying an order does not re-execute it.
        self.open_ids: set[str] = set()
        self._fills: dict[str, Fill] = {}

    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def cash(self) -> float:
        return self._cash

    def submit(self, order: Order) -> None:
        if order.client_order_id in self.reject_ids:
            raise OrderRejected(f"venue rejected {order.client_order_id}")
        if self.fail_after_n_submits is not None and self.submit_calls >= self.fail_after_n_submits:
            raise ConnectionError("simulated venue outage")
        if order.client_order_id in self.orders:
            raise DuplicateOrder(order.client_order_id)
        self.submit_calls += 1
        self.orders[order.client_order_id] = order

    def submitted_order_ids(self) -> set[str]:
        return set(self.orders)

    def open_order_ids(self, client_order_ids: set[str]) -> set[str]:
        return set(client_order_ids) & self.open_ids

    def fills_for(self, client_order_ids: set[str]) -> list[Fill]:
        fills = []
        for oid in sorted(client_order_ids & set(self.orders)):
            if oid in self.no_fill:
                continue  # e.g. an OPG order whose auction did not print
            if oid not in self._fills:
                order = self.orders[oid]
                qty = self.partial_fills.get(oid, order.qty)
                if qty == 0.0:
                    continue
                price = self.open_prices.get(order.symbol, order.reference_price)
                self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + qty
                self._cash -= qty * price
                self._fills[oid] = Fill(
                    client_order_id=oid,
                    symbol=order.symbol,
                    qty=qty,
                    price=price,
                    filled_bar="fill-bar",
                )
            fills.append(self._fills[oid])
        return fills


@pytest.fixture()
def ctx(tmp_path):
    return LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=FakeBroker(),
        fills_ledger=tmp_path / "fills.jsonl",
    )


def _targets(**weights: float) -> pd.Series:
    return pd.Series(weights, dtype=float)


_PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0})


# ---------------------------------------------------------------------------
# StateStore
# ---------------------------------------------------------------------------


def test_state_round_trip_atomic(tmp_path) -> None:
    store = StateStore(tmp_path / "s.json")
    assert store.load() is None  # fresh loop
    state = LoopState(
        positions={"AAA": 10.0},
        cash=123.45,
        pending_orders=[Order("d1:AAA", "AAA", 5.0, "d1", 100.0)],
        pending_decision_bar="d1",
        last_settled_bar="d0",
    )
    store.save(state)
    assert not (tmp_path / "s.json.tmp").exists()  # atomic rename left no temp
    loaded = store.load()
    assert loaded == state
    assert loaded.pending_orders[0].reference_price == 100.0


def test_corrupt_state_refuses_to_start_flat(tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to start flat"):
        StateStore(path).load()


def test_wrong_schema_version_raises(tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        StateStore(path).load()


# ---------------------------------------------------------------------------
# targets_to_orders
# ---------------------------------------------------------------------------


def test_targets_diff_against_held_shares() -> None:
    orders = targets_to_orders(
        _targets(AAA=0.5, BBB=-0.2),
        positions={"AAA": 30.0},
        prices=_PRICES,
        equity=10_000.0,
        decision_bar="2026-07-06",
    )
    by_symbol = {o.symbol: o for o in orders}
    # AAA: target 0.5*10000/100 = 50 shares, held 30 -> buy 20.
    assert by_symbol["AAA"].qty == pytest.approx(20.0)
    # BBB: target -0.2*10000/50 = -40 shares, held 0 -> sell 40.
    assert by_symbol["BBB"].qty == pytest.approx(-40.0)
    assert by_symbol["AAA"].client_order_id == "2026-07-06:AAA"
    assert by_symbol["AAA"].reference_price == 100.0


def test_nan_target_holds_and_dust_dropped() -> None:
    orders = targets_to_orders(
        _targets(AAA=np.nan, BBB=0.0001),
        positions={"AAA": 30.0},
        prices=_PRICES,
        equity=10_000.0,
        decision_bar="d",
        min_order_notional=5.0,
    )
    assert orders == []  # NaN = hold; BBB order is $1 of stock, under the floor


def test_missing_price_for_target_raises() -> None:
    with pytest.raises(ValueError, match="no usable decision price"):
        targets_to_orders(
            _targets(ZZZ=0.1), positions={}, prices=_PRICES, equity=1_000.0, decision_bar="d"
        )


# ---------------------------------------------------------------------------
# decide_and_submit: write-ahead + idempotent resume
# ---------------------------------------------------------------------------


def test_decision_persisted_before_submit_and_submitted(ctx) -> None:
    submitted = decide_and_submit(ctx, "2026-07-06", _targets(AAA=0.5), _PRICES)
    assert len(submitted) == 1
    state = ctx.store.load()
    assert state.pending_decision_bar == "2026-07-06"
    assert {o.client_order_id for o in state.pending_orders} == {"2026-07-06:AAA"}
    assert ctx.broker.submitted_order_ids() == {"2026-07-06:AAA"}


def test_crash_between_persist_and_submit_resumes_without_redeciding(ctx) -> None:
    ctx.broker.fail_after_n_submits = 1  # accept AAA, die before BBB
    with pytest.raises(ConnectionError):
        decide_and_submit(ctx, "d1", _targets(AAA=0.4, BBB=0.4), _PRICES)
    assert ctx.broker.submitted_order_ids() == {"d1:AAA"}  # partial submit happened

    # "Restart": same bar, DIFFERENT targets — the persisted decision wins.
    ctx.broker.fail_after_n_submits = None
    resumed = decide_and_submit(ctx, "d1", _targets(AAA=0.9), _PRICES)
    assert {o.client_order_id for o in resumed} == {"d1:AAA", "d1:BBB"}
    assert ctx.broker.submitted_order_ids() == {"d1:AAA", "d1:BBB"}
    # AAA was not re-submitted (still exactly one accepted order, original qty).
    assert ctx.broker.orders["d1:AAA"].qty == pytest.approx(0.4 * 10_000 / 100.0)
    assert ctx.broker.submit_calls == 2


def test_new_bar_with_unsettled_pending_raises(ctx) -> None:
    decide_and_submit(ctx, "d1", _targets(AAA=0.5), _PRICES)
    with pytest.raises(RuntimeError, match="unsettled"):
        decide_and_submit(ctx, "d2", _targets(AAA=0.5), _PRICES)


def test_redeciding_a_settled_bar_raises(ctx) -> None:
    decide_and_submit(ctx, "d1", _targets(AAA=0.5), _PRICES)
    settle(ctx, "d2")
    with pytest.raises(RuntimeError, match="not after last settled"):
        decide_and_submit(ctx, "d1", _targets(AAA=0.5), _PRICES)


# ---------------------------------------------------------------------------
# order sizing: a single order may not cross zero; a rejection must not wedge
# ---------------------------------------------------------------------------


def test_targets_to_orders_clamps_a_zero_crossing_order_to_flat() -> None:
    # Held short 1; the target flips to long. A single order may only cover to
    # flat — Alpaca rejects a side-flipping order ("insufficient qty available",
    # the 2026-07-08 ORCL crash). The opposite side opens next bar from flat.
    orders = targets_to_orders(
        _targets(AAA=0.5),  # long target
        positions={"AAA": -1.0},  # currently short 1
        prices=_PRICES,
        equity=10_000.0,
        decision_bar="d1",
        whole_shares=True,
    )
    assert len(orders) == 1
    assert orders[0].qty == 1.0  # buy 1 to flat, NOT buy 51 (a short->long flip)


def test_targets_to_orders_does_not_clamp_when_not_crossing_zero() -> None:
    # Held short 1, target deeper short: the order increases the short normally.
    orders = targets_to_orders(
        _targets(AAA=-0.5),
        positions={"AAA": -1.0},
        prices=_PRICES,
        equity=10_000.0,
        decision_bar="d1",
        whole_shares=True,
    )
    assert len(orders) == 1
    assert orders[0].qty == -49.0  # target -50, held -1 -> sell 49 more


def test_decide_and_submit_skips_a_rejected_order_without_crashing(ctx, caplog) -> None:
    ctx.broker.reject_ids.add("d1:BBB")  # the venue rejects one order (non-Duplicate)
    with caplog.at_level("ERROR"):
        orders = decide_and_submit(ctx, "d1", _targets(AAA=0.4, BBB=0.3), _PRICES)
    # the good order reached the venue, the rejected one is logged not raised
    assert "d1:AAA" in ctx.broker.submitted_order_ids()
    assert "d1:BBB" not in ctx.broker.submitted_order_ids()
    assert "rejected by the venue" in caplog.text and "d1:BBB" in caplog.text
    # both remain in the persisted decision — settle tolerates the missing fill
    assert {o.client_order_id for o in orders} == {"d1:AAA", "d1:BBB"}


# ---------------------------------------------------------------------------
# settle: fills ledgered with reference price, state re-anchored
# ---------------------------------------------------------------------------


def test_settle_ledgers_fills_and_reanchors_state(ctx) -> None:
    decide_and_submit(ctx, "d1", _targets(AAA=0.5), _PRICES)
    ctx.broker.open_prices["AAA"] = 101.0  # next-open fill, 1.0 above reference

    fills = settle(ctx, "d2")
    assert len(fills) == 1 and fills[0].price == 101.0

    ledger = read_fills_ledger(ctx.fills_ledger)
    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert row["reference_price"] == 100.0
    assert row["fill_price"] == 101.0  # arrival slippage is recoverable (I-9)
    assert row["decision_bar"] == "d1"

    state = ctx.store.load()
    assert state.pending_orders == [] and state.pending_decision_bar is None
    assert state.last_settled_bar == "d2"
    assert state.positions["AAA"] == pytest.approx(50.0)
    assert state.cash == pytest.approx(10_000.0 - 50.0 * 101.0)

    # The loop continues: next decision diffs against the filled book.
    # Marked at the fill price the book shows no MTM move: equity is
    # cash (10000 - 50*101) + 50*101 = exactly 10000.
    orders = decide_and_submit(ctx, "d3", _targets(AAA=0.5), pd.Series({"AAA": 101.0}))
    expected_delta = 0.5 * 10_000.0 / 101.0 - 50.0
    assert len(orders) == 1
    assert orders[0].qty == pytest.approx(expected_delta)


def test_settle_tolerates_unfilled_orders_and_reanchors(ctx, caplog) -> None:
    # Two orders decided; the venue fills AAA and never fills BBB (its OPG
    # auction did not print). Settle must ledger AAA, log BBB loudly, and let
    # the loop advance — not raise and wedge on the unfilled leg.
    decide_and_submit(ctx, "d1", _targets(AAA=0.4, BBB=0.3), _PRICES)
    ctx.broker.no_fill.add("d1:BBB")
    with caplog.at_level("WARNING"):
        fills = settle(ctx, "d2")

    assert {f.client_order_id for f in fills} == {"d1:AAA"}
    assert "did not fill" in caplog.text and "d1:BBB" in caplog.text

    ledger = read_fills_ledger(ctx.fills_ledger)
    assert list(ledger["client_order_id"]) == ["d1:AAA"]

    state = ctx.store.load()
    assert state.pending_orders == [] and state.pending_decision_bar is None
    assert state.last_settled_bar == "d2"
    assert "BBB" not in state.positions  # never filled -> not held (broker truth)
    # The loop keeps going: the next bar decides against the reconciled book.
    decide_and_submit(ctx, "d3", _targets(AAA=0.4), pd.Series({"AAA": 100.0}))


def test_settle_ledgers_partial_fill(ctx) -> None:
    decide_and_submit(ctx, "d1", _targets(AAA=0.5), _PRICES)  # 50 shares intended
    ctx.broker.partial_fills["d1:AAA"] = 20.0  # only 20 execute; remainder expires
    fills = settle(ctx, "d2")

    assert len(fills) == 1 and fills[0].qty == 20.0
    row = read_fills_ledger(ctx.fills_ledger).iloc[0]
    assert row["qty"] == 20.0 and row["reference_price"] == 100.0

    state = ctx.store.load()
    assert state.positions["AAA"] == pytest.approx(20.0)  # broker truth = the partial


def test_settle_with_zero_fills_logs_error_but_advances(ctx, caplog) -> None:
    decide_and_submit(ctx, "d1", _targets(AAA=0.5), _PRICES)
    ctx.broker.no_fill.add("d1:AAA")  # nothing filled at all
    with caplog.at_level("ERROR"):
        fills = settle(ctx, "d2")
    assert fills == []
    assert any(r.levelname == "ERROR" for r in caplog.records)
    assert read_fills_ledger(ctx.fills_ledger).empty  # nothing to ledger
    state = ctx.store.load()
    assert state.pending_orders == [] and state.last_settled_bar == "d2"


def test_settle_without_pending_is_noop(ctx) -> None:
    assert settle(ctx, "d1") == []
    assert read_fills_ledger(ctx.fills_ledger).empty


def test_ledger_appends_across_days(ctx) -> None:
    for bar, nxt in [("d1", "d2"), ("d3", "d4")]:
        decide_and_submit(ctx, bar, _targets(AAA=0.5 if bar == "d1" else 0.2), _PRICES)
        settle(ctx, nxt)
    ledger = read_fills_ledger(ctx.fills_ledger)
    assert list(ledger["decision_bar"]) == ["d1", "d3"]


# ---------------------------------------------------------------------------
# reconcile: broker truth wins, loudly
# ---------------------------------------------------------------------------


def test_reconcile_adopts_broker_truth_with_warning(ctx, caplog) -> None:
    import logging

    ctx.store.save(LoopState(positions={"AAA": 999.0}, cash=1.0))
    ctx.broker._positions = {"AAA": 10.0}
    with caplog.at_level(logging.WARNING):
        decide_and_submit(ctx, "d1", _targets(AAA=0.0), pd.Series({"AAA": 100.0}))
    assert "reconcile" in caplog.text
    state = ctx.store.load()
    # The decision was sized off broker truth (10 shares), not the stale 999.
    assert [o.qty for o in state.pending_orders] == [pytest.approx(-10.0)]


# ---------------------------------------------------------------------------
# order-id namespacing: two books on one venue account must never collide
# ---------------------------------------------------------------------------


def test_order_id_prefix_namespaces_client_ids(ctx) -> None:
    ctx.order_id_prefix = "mom:"
    decide_and_submit(ctx, "d1", _targets(AAA=0.5), _PRICES)
    assert ctx.broker.submitted_order_ids() == {"mom:d1:AAA"}
    state = ctx.store.load()
    assert state.pending_orders[0].client_order_id == "mom:d1:AAA"


# ---------------------------------------------------------------------------
# dust censoring is loud: a real intended order dropped by rounding is named
# ---------------------------------------------------------------------------


def test_round_to_zero_drop_is_logged_with_symbol_and_size(caplog) -> None:
    # $10k book, EXP priced $6,000: a 1% target is a $100 intent that rounds to
    # 0 whole shares — the NVR/AZO censoring class. It must be named, loudly.
    with caplog.at_level("WARNING"):
        orders = targets_to_orders(
            _targets(EXP=0.01, AAA=0.5),
            positions={},
            prices=pd.Series({"EXP": 6000.0, "AAA": 100.0}),
            equity=10_000.0,
            decision_bar="d",
            whole_shares=True,
        )
    assert {o.symbol for o in orders} == {"AAA"}
    assert "dropped by whole-share" in caplog.text and "EXP" in caplog.text


def test_sub_dollar_dust_stays_silent(caplog) -> None:
    # A sub-notional intent ($1 under a $5 floor) is genuine dust, not a
    # censored order — no warning noise.
    with caplog.at_level("WARNING"):
        targets_to_orders(
            _targets(BBB=0.0001),
            positions={},
            prices=_PRICES,
            equity=10_000.0,
            decision_bar="d",
            min_order_notional=5.0,
        )
    assert "dropped by whole-share" not in caplog.text


# ---------------------------------------------------------------------------
# targets ledger: the refresh book persists inside the write-ahead step
# ---------------------------------------------------------------------------


def test_targets_ledger_persists_refresh_book_idempotently(ctx, tmp_path) -> None:
    ctx.targets_ledger = tmp_path / "targets.jsonl"
    decide_and_submit(ctx, "d1", _targets(AAA=0.5, BBB=0.0), _PRICES, refresh_bar="d1")
    rows = read_targets_ledger(ctx.targets_ledger)
    assert len(rows) == 1
    assert rows[0]["refresh_bar"] == "d1" and rows[0]["decision_bar"] == "d1"
    assert rows[0]["targets"] == {"AAA": 0.5}  # explicit zeros are not persisted
    assert rows[0]["reference_prices"] == {"AAA": 100.0}
    assert rows[0]["equity"] == pytest.approx(10_000.0)
    # A same-bar resume never duplicates the row (and never re-decides).
    decide_and_submit(ctx, "d1", _targets(AAA=0.9), _PRICES, refresh_bar="d1")
    assert len(read_targets_ledger(ctx.targets_ledger)) == 1
    # An off-refresh decision writes no row.
    settle(ctx, "d1s")
    decide_and_submit(ctx, "d2", _targets(AAA=0.5), pd.Series({"AAA": 100.0}))
    assert len(read_targets_ledger(ctx.targets_ledger)) == 1


# ---------------------------------------------------------------------------
# unfilled ledger: every unexecuted residual is durable, not a truncated log
# ---------------------------------------------------------------------------


def test_settle_writes_unfilled_and_partial_residuals(ctx, tmp_path) -> None:
    ctx.unfilled_ledger = tmp_path / "unfilled.jsonl"
    decide_and_submit(ctx, "d1", _targets(AAA=0.4, BBB=0.3), _PRICES)
    ctx.broker.partial_fills["d1:AAA"] = 10.0  # 40 intended, 10 execute
    ctx.broker.no_fill.add("d1:BBB")  # 60 intended, auction never prints
    settle(ctx, "d2")
    rows = [json.loads(ln) for ln in ctx.unfilled_ledger.read_text().splitlines() if ln]
    by_symbol = {r["symbol"]: r for r in rows}
    assert by_symbol["AAA"]["order_qty"] == pytest.approx(40.0)
    assert by_symbol["AAA"]["executed_qty"] == pytest.approx(10.0)
    assert by_symbol["AAA"]["residual_qty"] == pytest.approx(30.0)
    assert by_symbol["BBB"]["executed_qty"] == 0.0
    assert by_symbol["BBB"]["residual_qty"] == pytest.approx(60.0)
    assert all(r["decision_bar"] == "d1" and r["settle_bar"] == "d2" for r in rows)


def test_settle_writes_no_unfilled_rows_when_everything_fills(ctx, tmp_path) -> None:
    ctx.unfilled_ledger = tmp_path / "unfilled.jsonl"
    decide_and_submit(ctx, "d1", _targets(AAA=0.4), _PRICES)
    settle(ctx, "d2")
    assert not ctx.unfilled_ledger.exists()


# ---------------------------------------------------------------------------
# sweep: unexecuted residuals complete the decision as suffixed DAY orders
# ---------------------------------------------------------------------------


def test_sweep_resubmits_residuals_write_ahead_first(ctx) -> None:
    decide_and_submit(ctx, "d1", _targets(AAA=0.4, BBB=0.3), _PRICES)
    ctx.broker.partial_fills["d1:AAA"] = 10.0  # residual 30
    ctx.broker.no_fill.add("d1:BBB")  # residual 60
    swept = sweep_pending(ctx)
    by_symbol = {o.symbol: o for o in swept}
    assert {o.client_order_id for o in swept} == {"d1:AAA:S1", "d1:BBB:S1"}
    assert by_symbol["AAA"].qty == pytest.approx(30.0)
    assert by_symbol["BBB"].qty == pytest.approx(60.0)
    # The sweep order carries the ORIGINAL decision-close reference: the fills
    # ledger then records the total arrival cost of executing the decision.
    assert by_symbol["AAA"].reference_price == 100.0
    assert by_symbol["AAA"].decision_bar == "d1"
    # Write-ahead: the sweep orders are persisted in the pending set…
    state = ctx.store.load()
    pending_ids = {o.client_order_id for o in state.pending_orders}
    assert {"d1:AAA:S1", "d1:BBB:S1"} <= pending_ids
    # …and reached the venue.
    assert {"d1:AAA:S1", "d1:BBB:S1"} <= ctx.broker.submitted_order_ids()


def test_sweep_skips_fully_filled_orders(ctx) -> None:
    decide_and_submit(ctx, "d1", _targets(AAA=0.4), _PRICES)
    assert sweep_pending(ctx) == []  # full fill: nothing to complete
    assert "d1:AAA:S1" not in ctx.broker.submitted_order_ids()


def test_sweep_never_touches_a_live_order(ctx, caplog) -> None:
    decide_and_submit(ctx, "d1", _targets(AAA=0.4), _PRICES)
    ctx.broker.no_fill.add("d1:AAA")
    ctx.broker.open_ids.add("d1:AAA")  # auction has not run yet
    with caplog.at_level("WARNING"):
        assert sweep_pending(ctx) == []
    assert "still live" in caplog.text
    assert "d1:AAA:S1" not in ctx.broker.submitted_order_ids()


def test_sweep_rerun_is_idempotent(ctx) -> None:
    decide_and_submit(ctx, "d1", _targets(AAA=0.4), _PRICES)
    ctx.broker.no_fill.add("d1:AAA")
    first = sweep_pending(ctx)
    assert [o.client_order_id for o in first] == ["d1:AAA:S1"]
    # Rerun: the child now exists (and by now has filled) — nothing new is
    # created and the pending set holds exactly one child.
    assert sweep_pending(ctx) == []
    state = ctx.store.load()
    ids = [o.client_order_id for o in state.pending_orders]
    assert ids.count("d1:AAA:S1") == 1


def test_sweep_requires_the_open_order_guard(ctx) -> None:
    class BlindBroker(FakeBroker):
        open_order_ids = None  # a broker that cannot distinguish live orders

    blind = BlindBroker()
    ctx.broker = blind
    decide_and_submit(ctx, "d1", _targets(AAA=0.4), _PRICES)
    ctx.broker.no_fill.add("d1:AAA")
    with pytest.raises(TypeError, match="open_order_ids"):
        sweep_pending(ctx)


def test_sweep_never_amplifies_a_venue_overfill(ctx, caplog) -> None:
    decide_and_submit(ctx, "d1", _targets(AAA=0.4), _PRICES)  # 40 intended
    ctx.broker.partial_fills["d1:AAA"] = 50.0  # venue anomaly: 50 executed
    with caplog.at_level("WARNING"):
        assert sweep_pending(ctx) == []
    assert "never amplifying" in caplog.text


def test_settle_after_sweep_ledgers_both_fill_populations(ctx) -> None:
    decide_and_submit(ctx, "d1", _targets(AAA=0.4), _PRICES)
    ctx.broker.partial_fills["d1:AAA"] = 10.0
    sweep_pending(ctx)  # completes the 30-share residual as d1:AAA:S1
    fills = settle(ctx, "d2")
    assert {f.client_order_id for f in fills} == {"d1:AAA", "d1:AAA:S1"}
    ledger = read_fills_ledger(ctx.fills_ledger)
    assert set(ledger["client_order_id"]) == {"d1:AAA", "d1:AAA:S1"}
    # Both populations share the decision-close reference (total arrival cost),
    # segmentable by the :S1 id suffix.
    assert set(ledger["reference_price"]) == {100.0}
    state = ctx.store.load()
    assert state.positions["AAA"] == pytest.approx(40.0)  # decision completed
    assert state.pending_orders == []


# ---------------------------------------------------------------------------
# Order validation
# ---------------------------------------------------------------------------


def test_order_validation() -> None:
    with pytest.raises(ValueError, match="zero-qty"):
        Order("id", "AAA", 0.0, "d", 100.0)
    with pytest.raises(ValueError, match="reference_price"):
        Order("id", "AAA", 1.0, "d", 0.0)
    with pytest.raises(ValueError, match="client_order_id"):
        Order("", "AAA", 1.0, "d", 100.0)
