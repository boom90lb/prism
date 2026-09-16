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
from prism.live.loop import PartialBatch, PartialBatchEscalation, RemainderCheck


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
        # Ids cancelled through the C0 path, in the order they were asked for.
        self.cancelled: list[str] = []

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

    def cancel(self, client_order_id: str) -> None:
        # Idempotent on a terminal or unknown id, as the §3.9 contract requires:
        # the venue no longer works the order, whether or not it ever did.
        self.cancelled.append(client_order_id)
        self.open_ids.discard(client_order_id)
        self.no_fill.add(client_order_id)

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
_PRICES3 = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 25.0})


# ---------------------------------------------------------------------------
# StateStore
# ---------------------------------------------------------------------------


def test_state_round_trip_atomic(tmp_path) -> None:
    store = StateStore(tmp_path / "s.json")
    assert store.load() is None  # fresh loop
    state = LoopState(
        positions={"AAA": 10.0},
        cash=123.45,
        pending_orders=[Order("2026-03-04:AAA", "AAA", 5.0, "2026-03-04", 100.0)],
        pending_decision_bar="2026-03-04",
        last_settled_bar="2026-03-02",
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
        targets_to_orders(_targets(ZZZ=0.1), positions={}, prices=_PRICES, equity=1_000.0, decision_bar="d")


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
        decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4, BBB=0.4), _PRICES)
    assert ctx.broker.submitted_order_ids() == {"2026-03-04:AAA"}  # partial submit happened

    # "Restart": same bar, DIFFERENT targets — the persisted decision wins.
    ctx.broker.fail_after_n_submits = None
    resumed = decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.9), _PRICES)
    assert {o.client_order_id for o in resumed} == {"2026-03-04:AAA", "2026-03-04:BBB"}
    assert ctx.broker.submitted_order_ids() == {"2026-03-04:AAA", "2026-03-04:BBB"}
    # AAA was not re-submitted (still exactly one accepted order, original qty).
    assert ctx.broker.orders["2026-03-04:AAA"].qty == pytest.approx(0.4 * 10_000 / 100.0)
    assert ctx.broker.submit_calls == 2


def test_new_bar_with_unsettled_pending_raises(ctx) -> None:
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.5), _PRICES)
    with pytest.raises(RuntimeError, match="unsettled"):
        decide_and_submit(ctx, "2026-03-06", _targets(AAA=0.5), _PRICES)


def test_redeciding_a_settled_bar_raises(ctx) -> None:
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.5), _PRICES)
    settle(ctx, "2026-03-06")
    with pytest.raises(RuntimeError, match="not after last settled"):
        decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.5), _PRICES)


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
        decision_bar="2026-03-04",
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
        decision_bar="2026-03-04",
        whole_shares=True,
    )
    assert len(orders) == 1
    assert orders[0].qty == -49.0  # target -50, held -1 -> sell 49 more


# ---------------------------------------------------------------------------
# a venue rejection mid-batch (§3.7 post-submit; acceptance case "Broker
# rejection of one order")
#
# `test_decide_and_submit_skips_a_rejected_order_without_crashing` was retired
# here rather than adapted. It asserted the behaviour the correction inverts —
# "skipping it, submitting the rest", with both ids left offerable in the
# persisted set — and it passed only because its rejected order happened to be
# the last one in the batch, so it never exercised the remainder at all.
# ---------------------------------------------------------------------------


def test_a_rejection_stops_the_batch_and_records_the_blocked_order(ctx, caplog) -> None:
    """The remainder is not offered, and the rejected leg is recorded with the reason."""
    ctx.broker.reject_ids.add("2026-03-04:BBB")  # the venue rejects the middle order
    with caplog.at_level("ERROR"):
        decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4, BBB=0.3, CCC=0.2), _PRICES3)

    # AAA reached the venue; BBB was refused; CCC was never offered.
    assert ctx.broker.submitted_order_ids() == {"2026-03-04:AAA"}
    assert "submission stops here" in caplog.text and "2026-03-04:BBB" in caplog.text

    state = ctx.store.load()
    fates = {record.client_order_id: record.status for record in state.portfolio.pending_orders}
    assert fates == {"2026-03-04:AAA": "pending", "2026-03-04:BBB": "rejected", "2026-03-04:CCC": "pending"}
    blocked = next(r for r in state.portfolio.pending_orders if r.client_order_id == "2026-03-04:BBB")
    assert "venue rejected 2026-03-04:BBB" in (blocked.reason or "")
    # The rejected leg is no longer offerable; the unoffered remainder still is.
    assert {o.client_order_id for o in state.open_pending_orders} == {"2026-03-04:AAA", "2026-03-04:CCC"}


def test_a_rejected_order_is_not_re_offered_by_a_resume_or_by_the_sweep(ctx) -> None:
    """The retry §3.7 removed from submission does not come back through the sweep."""
    ctx.broker.reject_ids.add("2026-03-04:BBB")
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4, BBB=0.3, CCC=0.2), _PRICES3)

    # A same-bar resume offers the remainder, and never the rejected leg —
    # even with the venue's refusal lifted, which is the point: the fate is
    # established durably, not re-derived from the venue's current mood.
    ctx.broker.reject_ids.clear()
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4, BBB=0.3, CCC=0.2), _PRICES3)
    assert ctx.broker.submitted_order_ids() == {"2026-03-04:AAA", "2026-03-04:CCC"}

    swept = sweep_pending(ctx)
    assert [o.client_order_id for o in swept] == []  # AAA and CCC filled; BBB is terminal
    assert not any(oid.startswith("2026-03-04:BBB") for oid in ctx.broker.submitted_order_ids())


def test_the_remainder_is_revalidated_against_the_accepted_legs(ctx) -> None:
    """What the check is shown is the portfolio the accepted legs imply, not the batch."""
    seen: list[PartialBatch] = []

    def watch(partial: PartialBatch) -> list[str]:
        seen.append(partial)
        return []

    ctx.remainder_check = RemainderCheck(validate=watch)
    ctx.broker.reject_ids.add("2026-03-04:BBB")
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4, BBB=0.3, CCC=0.2), _PRICES3)

    assert len(seen) == 1
    partial = seen[0]
    assert [o.client_order_id for o in partial.accepted] == ["2026-03-04:AAA"]
    assert [o.client_order_id for o in partial.rejected] == ["2026-03-04:BBB"]
    assert [o.client_order_id for o in partial.unsubmitted] == ["2026-03-04:CCC"]
    assert partial.cash == 10_000.0 and partial.positions == {}
    assert "venue rejected 2026-03-04:BBB" in partial.reasons["2026-03-04:BBB"]


def test_cancel_accepted_cancels_the_committed_legs_when_the_remainder_no_longer_validates(ctx) -> None:
    """The declared response is taken, and the cancellation is recorded durably."""
    ctx.remainder_check = RemainderCheck(validate=lambda partial: ["net_exposure"])
    ctx.broker.reject_ids.add("2026-03-04:BBB")
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4, BBB=0.3, CCC=0.2), _PRICES3)

    assert ctx.broker.cancelled == ["2026-03-04:AAA"]
    state = ctx.store.load()
    fates = {record.client_order_id: record.status for record in state.portfolio.pending_orders}
    assert fates["2026-03-04:AAA"] == "canceled" and fates["2026-03-04:BBB"] == "rejected"
    assert state.open_pending_orders == [] or {o.client_order_id for o in state.open_pending_orders} == {
        "2026-03-04:CCC"
    }


def test_hold_and_escalate_leaves_the_accepted_legs_and_raises(ctx) -> None:
    ctx.remainder_check = RemainderCheck(validate=lambda partial: ["max_gross"], on_partial_batch="hold_and_escalate")
    ctx.broker.reject_ids.add("2026-03-04:BBB")
    with pytest.raises(PartialBatchEscalation, match="max_gross"):
        decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4, BBB=0.3, CCC=0.2), _PRICES3)
    assert ctx.broker.cancelled == []
    assert ctx.broker.submitted_order_ids() == {"2026-03-04:AAA"}


def test_an_unknown_partial_batch_response_is_refused_at_declaration() -> None:
    with pytest.raises(ValueError, match="on_partial_batch"):
        RemainderCheck(validate=lambda partial: [], on_partial_batch="carry_on")


# ---------------------------------------------------------------------------
# settle: fills ledgered with reference price, state re-anchored
# ---------------------------------------------------------------------------


def test_settle_ledgers_fills_and_reanchors_state(ctx) -> None:
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.5), _PRICES)
    ctx.broker.open_prices["AAA"] = 101.0  # next-open fill, 1.0 above reference

    fills = settle(ctx, "2026-03-06")
    assert len(fills) == 1 and fills[0].price == 101.0

    ledger = read_fills_ledger(ctx.fills_ledger)
    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert row["reference_price"] == 100.0
    assert row["fill_price"] == 101.0  # arrival slippage is recoverable (I-9)
    assert row["decision_bar"] == "2026-03-04"

    state = ctx.store.load()
    assert state.pending_orders == [] and state.pending_decision_bar is None
    assert state.last_settled_bar == "2026-03-06"
    assert state.positions["AAA"] == pytest.approx(50.0)
    assert state.cash == pytest.approx(10_000.0 - 50.0 * 101.0)

    # The loop continues: next decision diffs against the filled book.
    # Marked at the fill price the book shows no MTM move: equity is
    # cash (10000 - 50*101) + 50*101 = exactly 10000.
    orders = decide_and_submit(ctx, "2026-03-10", _targets(AAA=0.5), pd.Series({"AAA": 101.0}))
    expected_delta = 0.5 * 10_000.0 / 101.0 - 50.0
    assert len(orders) == 1
    assert orders[0].qty == pytest.approx(expected_delta)


def test_settle_tolerates_unfilled_orders_and_reanchors(ctx, caplog) -> None:
    # Two orders decided; the venue fills AAA and never fills BBB (its OPG
    # auction did not print). Settle must ledger AAA, log BBB loudly, and let
    # the loop advance — not raise and wedge on the unfilled leg.
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4, BBB=0.3), _PRICES)
    ctx.broker.no_fill.add("2026-03-04:BBB")
    with caplog.at_level("WARNING"):
        fills = settle(ctx, "2026-03-06")

    assert {f.client_order_id for f in fills} == {"2026-03-04:AAA"}
    assert "did not fill" in caplog.text and "2026-03-04:BBB" in caplog.text

    ledger = read_fills_ledger(ctx.fills_ledger)
    assert list(ledger["client_order_id"]) == ["2026-03-04:AAA"]

    state = ctx.store.load()
    assert state.pending_orders == [] and state.pending_decision_bar is None
    assert state.last_settled_bar == "2026-03-06"
    assert "BBB" not in state.positions  # never filled -> not held (broker truth)
    # The loop keeps going: the next bar decides against the reconciled book.
    decide_and_submit(ctx, "2026-03-10", _targets(AAA=0.4), pd.Series({"AAA": 100.0}))


def test_settle_ledgers_partial_fill(ctx) -> None:
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.5), _PRICES)  # 50 shares intended
    ctx.broker.partial_fills["2026-03-04:AAA"] = 20.0  # only 20 execute; remainder expires
    fills = settle(ctx, "2026-03-06")

    assert len(fills) == 1 and fills[0].qty == 20.0
    row = read_fills_ledger(ctx.fills_ledger).iloc[0]
    assert row["qty"] == 20.0 and row["reference_price"] == 100.0

    state = ctx.store.load()
    assert state.positions["AAA"] == pytest.approx(20.0)  # broker truth = the partial


def test_settle_with_zero_fills_logs_error_but_advances(ctx, caplog) -> None:
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.5), _PRICES)
    ctx.broker.no_fill.add("2026-03-04:AAA")  # nothing filled at all
    with caplog.at_level("ERROR"):
        fills = settle(ctx, "2026-03-06")
    assert fills == []
    assert any(r.levelname == "ERROR" for r in caplog.records)
    assert read_fills_ledger(ctx.fills_ledger).empty  # nothing to ledger
    state = ctx.store.load()
    assert state.pending_orders == [] and state.last_settled_bar == "2026-03-06"


def test_settle_without_pending_is_noop(ctx) -> None:
    assert settle(ctx, "2026-03-04") == []
    assert read_fills_ledger(ctx.fills_ledger).empty


def test_ledger_appends_across_days(ctx) -> None:
    for bar, nxt in [("2026-03-04", "2026-03-06"), ("2026-03-10", "2026-03-12")]:
        decide_and_submit(ctx, bar, _targets(AAA=0.5 if bar == "2026-03-04" else 0.2), _PRICES)
        settle(ctx, nxt)
    ledger = read_fills_ledger(ctx.fills_ledger)
    assert list(ledger["decision_bar"]) == ["2026-03-04", "2026-03-10"]


# ---------------------------------------------------------------------------
# reconcile: broker truth wins, loudly
# ---------------------------------------------------------------------------


def test_reconcile_adopts_broker_truth_with_warning(ctx, caplog) -> None:
    import logging

    ctx.store.save(LoopState(positions={"AAA": 999.0}, cash=1.0))
    ctx.broker._positions = {"AAA": 10.0}
    with caplog.at_level(logging.WARNING):
        decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.0), pd.Series({"AAA": 100.0}))
    assert "reconcile" in caplog.text
    state = ctx.store.load()
    # The decision was sized off broker truth (10 shares), not the stale 999.
    assert [o.qty for o in state.pending_orders] == [pytest.approx(-10.0)]


# ---------------------------------------------------------------------------
# order-id namespacing: two books on one venue account must never collide
# ---------------------------------------------------------------------------


def test_order_id_prefix_namespaces_client_ids(ctx) -> None:
    ctx.order_id_prefix = "mom:"
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.5), _PRICES)
    assert ctx.broker.submitted_order_ids() == {"mom:2026-03-04:AAA"}
    state = ctx.store.load()
    assert state.pending_orders[0].client_order_id == "mom:2026-03-04:AAA"


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
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.5, BBB=0.0), _PRICES, refresh_bar="2026-03-04")
    rows = read_targets_ledger(ctx.targets_ledger)
    assert len(rows) == 1
    assert rows[0]["refresh_bar"] == "2026-03-04" and rows[0]["decision_bar"] == "2026-03-04"
    assert rows[0]["targets"] == {"AAA": 0.5}  # explicit zeros are not persisted
    assert rows[0]["reference_prices"] == {"AAA": 100.0}
    assert rows[0]["equity"] == pytest.approx(10_000.0)
    # A same-bar resume never duplicates the row (and never re-decides).
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.9), _PRICES, refresh_bar="2026-03-04")
    assert len(read_targets_ledger(ctx.targets_ledger)) == 1
    # An off-refresh decision writes no row.
    settle(ctx, "2026-03-05")
    decide_and_submit(ctx, "2026-03-06", _targets(AAA=0.5), pd.Series({"AAA": 100.0}))
    assert len(read_targets_ledger(ctx.targets_ledger)) == 1


# ---------------------------------------------------------------------------
# unfilled ledger: every unexecuted residual is durable, not a truncated log
# ---------------------------------------------------------------------------


def test_settle_writes_unfilled_and_partial_residuals(ctx, tmp_path) -> None:
    ctx.unfilled_ledger = tmp_path / "unfilled.jsonl"
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4, BBB=0.3), _PRICES)
    ctx.broker.partial_fills["2026-03-04:AAA"] = 10.0  # 40 intended, 10 execute
    ctx.broker.no_fill.add("2026-03-04:BBB")  # 60 intended, auction never prints
    settle(ctx, "2026-03-06")
    rows = [json.loads(ln) for ln in ctx.unfilled_ledger.read_text().splitlines() if ln]
    by_symbol = {r["symbol"]: r for r in rows}
    assert by_symbol["AAA"]["order_qty"] == pytest.approx(40.0)
    assert by_symbol["AAA"]["executed_qty"] == pytest.approx(10.0)
    assert by_symbol["AAA"]["residual_qty"] == pytest.approx(30.0)
    assert by_symbol["BBB"]["executed_qty"] == 0.0
    assert by_symbol["BBB"]["residual_qty"] == pytest.approx(60.0)
    assert all(r["decision_bar"] == "2026-03-04" and r["settle_bar"] == "2026-03-06" for r in rows)


def test_settle_writes_no_unfilled_rows_when_everything_fills(ctx, tmp_path) -> None:
    ctx.unfilled_ledger = tmp_path / "unfilled.jsonl"
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
    settle(ctx, "2026-03-06")
    assert not ctx.unfilled_ledger.exists()


# ---------------------------------------------------------------------------
# sweep: unexecuted residuals complete the decision as suffixed DAY orders
# ---------------------------------------------------------------------------


def test_sweep_resubmits_residuals_write_ahead_first(ctx) -> None:
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4, BBB=0.3), _PRICES)
    ctx.broker.partial_fills["2026-03-04:AAA"] = 10.0  # residual 30
    ctx.broker.no_fill.add("2026-03-04:BBB")  # residual 60
    swept = sweep_pending(ctx)
    by_symbol = {o.symbol: o for o in swept}
    assert {o.client_order_id for o in swept} == {"2026-03-04:AAA:S1", "2026-03-04:BBB:S1"}
    assert by_symbol["AAA"].qty == pytest.approx(30.0)
    assert by_symbol["BBB"].qty == pytest.approx(60.0)
    # The sweep order carries the ORIGINAL decision-close reference: the fills
    # ledger then records the total arrival cost of executing the decision.
    assert by_symbol["AAA"].reference_price == 100.0
    assert by_symbol["AAA"].decision_bar == "2026-03-04"
    # Write-ahead: the sweep orders are persisted in the pending set…
    state = ctx.store.load()
    pending_ids = {o.client_order_id for o in state.pending_orders}
    assert {"2026-03-04:AAA:S1", "2026-03-04:BBB:S1"} <= pending_ids
    # …and reached the venue.
    assert {"2026-03-04:AAA:S1", "2026-03-04:BBB:S1"} <= ctx.broker.submitted_order_ids()


def test_sweep_skips_fully_filled_orders(ctx) -> None:
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
    assert sweep_pending(ctx) == []  # full fill: nothing to complete
    assert "2026-03-04:AAA:S1" not in ctx.broker.submitted_order_ids()


def test_sweep_never_touches_a_live_order(ctx, caplog) -> None:
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
    ctx.broker.no_fill.add("2026-03-04:AAA")
    ctx.broker.open_ids.add("2026-03-04:AAA")  # auction has not run yet
    with caplog.at_level("WARNING"):
        assert sweep_pending(ctx) == []
    assert "still live" in caplog.text
    assert "2026-03-04:AAA:S1" not in ctx.broker.submitted_order_ids()


def test_sweep_rerun_is_idempotent(ctx) -> None:
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
    ctx.broker.no_fill.add("2026-03-04:AAA")
    first = sweep_pending(ctx)
    assert [o.client_order_id for o in first] == ["2026-03-04:AAA:S1"]
    # Rerun: the child now exists (and by now has filled) — nothing new is
    # created and the pending set holds exactly one child.
    assert sweep_pending(ctx) == []
    state = ctx.store.load()
    ids = [o.client_order_id for o in state.pending_orders]
    assert ids.count("2026-03-04:AAA:S1") == 1


def test_sweep_requires_the_open_order_guard(ctx) -> None:
    class BlindBroker(FakeBroker):
        open_order_ids = None  # a broker that cannot distinguish live orders

    blind = BlindBroker()
    ctx.broker = blind
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
    ctx.broker.no_fill.add("2026-03-04:AAA")
    with pytest.raises(TypeError, match="open_order_ids"):
        sweep_pending(ctx)


def test_sweep_never_amplifies_a_venue_overfill(ctx, caplog) -> None:
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)  # 40 intended
    ctx.broker.partial_fills["2026-03-04:AAA"] = 50.0  # venue anomaly: 50 executed
    with caplog.at_level("WARNING"):
        assert sweep_pending(ctx) == []
    assert "never amplifying" in caplog.text


def test_settle_after_sweep_ledgers_both_fill_populations(ctx) -> None:
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
    ctx.broker.partial_fills["2026-03-04:AAA"] = 10.0
    sweep_pending(ctx)  # completes the 30-share residual as 2026-03-04:AAA:S1
    fills = settle(ctx, "2026-03-06")
    assert {f.client_order_id for f in fills} == {"2026-03-04:AAA", "2026-03-04:AAA:S1"}
    ledger = read_fills_ledger(ctx.fills_ledger)
    assert set(ledger["client_order_id"]) == {"2026-03-04:AAA", "2026-03-04:AAA:S1"}
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


# ---------------------------------------------------------------------------
# Crash-idempotent settle ledgering and the scoped submit scan
# ---------------------------------------------------------------------------


def test_settle_crash_resume_does_not_duplicate_fill_rows(ctx) -> None:
    # A crash BETWEEN the fills-ledger append and the state save resumes with
    # the pending set intact; re-settling must not double-append the I-9 rows.
    decide_and_submit(ctx, "2026-01-02", _targets(AAA=0.5), _PRICES)
    pre_settle_state = ctx.store.path.read_text(encoding="utf-8")
    settle(ctx, "2026-01-02")
    rows_after_first = len(read_fills_ledger(ctx.fills_ledger))
    assert rows_after_first >= 1
    ctx.store.path.write_text(pre_settle_state, encoding="utf-8")  # the simulated crash
    settle(ctx, "2026-01-02")
    assert len(read_fills_ledger(ctx.fills_ledger)) == rows_after_first


def test_settle_crash_resume_does_not_duplicate_unfilled_rows(ctx, tmp_path) -> None:
    ctx = LiveLoopContext(
        store=ctx.store,
        broker=ctx.broker,
        fills_ledger=ctx.fills_ledger,
        unfilled_ledger=tmp_path / "unfilled.jsonl",
    )
    orders = decide_and_submit(ctx, "2026-01-02", _targets(AAA=0.5, BBB=0.2), _PRICES)
    ctx.broker.no_fill.add(orders[0].client_order_id)
    pre_settle_state = ctx.store.path.read_text(encoding="utf-8")
    settle(ctx, "2026-01-02")
    rows = [json.loads(line) for line in ctx.unfilled_ledger.read_text().splitlines() if line]
    ctx.store.path.write_text(pre_settle_state, encoding="utf-8")
    settle(ctx, "2026-01-02")
    rows_again = [json.loads(line) for line in ctx.unfilled_ledger.read_text().splitlines() if line]
    assert rows_again == rows


def test_known_order_ids_passes_since_only_when_supported() -> None:
    from prism.live.loop import _known_order_ids

    class SinceBroker(FakeBroker):
        def __init__(self) -> None:
            super().__init__()
            self.since_seen: str | None = "unset"

        def submitted_order_ids(self, since: str | None = None) -> set[str]:
            self.since_seen = since
            return set(self.orders)

    orders = [
        Order("mom:2026-07-14:BBB", "BBB", 1.0, "2026-07-14", 50.0),
        Order("mom:2026-07-13:AAA", "AAA", 1.0, "2026-07-13", 100.0),
    ]
    scoped = SinceBroker()
    assert _known_order_ids(scoped, orders) == set()
    assert scoped.since_seen == "2026-07-13"  # earliest pending decision bar
    # A bare-contract broker (no `since` param) still works, full-history.
    assert _known_order_ids(FakeBroker(), orders) == set()
