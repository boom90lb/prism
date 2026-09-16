"""Restart recovery on state version 3, and the ambiguous submission (§3.10, §3.11).

The write-ahead record says what was *decided*. It does not say what reached
the venue, and the gap between those two is the whole subject here. A restarted
process therefore asks the venue for each persisted order's fate before it
retries anything, and a fate the venue will not state is operating state S3:
cancellation and reconciliation only, until the fate is known.

Three things are pinned:

* the fate read itself — working, terminal, never submitted, or unknown — and
  that "the venue holds none of these" and "this adapter cannot list open
  orders" are never read as the same fact;
* that an unknown fate blocks new exposure on every retry path (the resumed
  decision and the completion sweep) while cancellation stays available, and
  that resolving it is durable, so a second restart does not ask again;
* that the account execution lock is held across the whole mutating path —
  before the first state load, after the last state save, settle inside.

The acceptance case runs against a **copy** of the live paper loop's own
version-2 ``state.json``: the live document is never written to, and the copy's
bytes are compared before and after to prove the recovery read it without
touching it.
"""

from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from prism.live.authorization import ActionClass, OperatingState, permitted_classes
from prism.live.broker import BrokerCapabilityUnsupported, Fill, Order, OrderStatus
from prism.live.lockfile import LockRefused, account_execution_lock
from prism.live.loop import (
    FATE_TERMINAL,
    FATE_UNKNOWN,
    FATE_UNSUBMITTED,
    FATE_WORKING,
    ExecutionLockSpec,
    LiveLoopContext,
    UnresolvedSubmission,
    cancel_orders,
    decide_and_submit,
    pending_order_recovery,
    read_order_fates,
    recover_on_restart,
    resolve_unresolved_submission,
    settle,
    sweep_pending,
)
from prism.live.state import LoopState, StateStore
from tests.test_live_loop import FakeBroker
from tests.test_live_state_v3 import _live_run_dir

HOST = socket.gethostname()
ACCOUNT = "PA-RESTART-0001"
_PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0})


def _targets(**weights: float) -> pd.Series:
    return pd.Series(weights, dtype=float)


class AnsweringVenue(FakeBroker):
    """A venue that can state whether an order is live, with quantity and side."""

    def open_orders(self) -> list[OrderStatus]:
        rows = []
        for client_order_id in sorted(self.open_ids & set(self.orders)):
            order = self.orders[client_order_id]
            rows.append(
                OrderStatus(
                    client_order_id=client_order_id,
                    sid=order.symbol,
                    qty=order.qty,
                    filled_qty=0.0,
                    order_type="market",
                    time_in_force="opg",
                    status="new",
                )
            )
        return rows

    def cancel(self, client_order_id: str) -> None:
        self.cancelled = getattr(self, "cancelled", [])
        self.cancelled.append(client_order_id)
        self.open_ids.discard(client_order_id)


class MuteVenue(AnsweringVenue):
    """A venue that knows the id and will not say whether it is still working.

    Not a contrived fake: it is the adapter that cannot list open orders, which
    §3.11 names explicitly, and the venue that answers a listing with an error
    on the one morning it matters.
    """

    open_order_ids = None  # the duck-typed membership read is absent entirely

    def open_orders(self) -> list[OrderStatus]:
        raise BrokerCapabilityUnsupported("MuteVenue does not implement open_orders()")


def _ctx(tmp_path: Path, broker, *, lock: bool = True) -> LiveLoopContext:
    return LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=broker,
        fills_ledger=tmp_path / "fills.jsonl",
        unfilled_ledger=tmp_path / "unfilled.jsonl",
        execution_lock=(
            ExecutionLockSpec(account_id=ACCOUNT, execution_host=HOST, lock_dir=tmp_path / "locks") if lock else None
        ),
    )


# ---------------------------------------------------------------------------
# the fate read
# ---------------------------------------------------------------------------


def _order(oid: str = "2026-07-06:AAA", symbol: str = "AAA", qty: float = 10.0) -> Order:
    return Order(client_order_id=oid, symbol=symbol, qty=qty, decision_bar="2026-07-06", reference_price=100.0)


def test_a_live_order_reads_as_working_and_an_unknown_id_as_never_submitted() -> None:
    broker = AnsweringVenue()
    working, never = _order("2026-07-06:AAA"), _order("2026-07-06:BBB", symbol="BBB")
    broker.submit(working)
    broker.open_ids.add(working.client_order_id)
    fates = {fate.client_order_id: fate for fate in read_order_fates(broker, [working, never])}
    assert fates["2026-07-06:AAA"].fate == FATE_WORKING
    assert fates["2026-07-06:BBB"].fate == FATE_UNSUBMITTED
    assert all(fate.known for fate in fates.values())


def test_an_id_the_venue_knows_and_does_not_report_live_is_terminal() -> None:
    broker = AnsweringVenue()
    order = _order()
    broker.submit(order)  # accepted, and not in open_ids: the venue is done with it
    (fate,) = read_order_fates(broker, [order])
    assert fate.fate == FATE_TERMINAL and fate.known


def test_an_adapter_that_cannot_list_open_orders_is_not_read_as_holding_none() -> None:
    broker = MuteVenue()
    order = _order()
    broker.submit(order)
    broker.no_fill.add(order.client_order_id)
    (fate,) = read_order_fates(broker, [order])
    assert fate.fate == FATE_UNKNOWN and not fate.known
    assert "cannot list open orders" in fate.detail


def test_an_execution_settles_the_fate_even_when_the_venue_cannot_be_listed() -> None:
    broker = MuteVenue()
    order = _order()
    broker.submit(order)
    broker.open_prices["AAA"] = 100.0
    (fate,) = read_order_fates(broker, [order])
    assert fate.known and fate.fate == "executed"


# ---------------------------------------------------------------------------
# restart mid-session, against a state file constructed here
# ---------------------------------------------------------------------------


def test_a_restart_mid_session_resumes_the_persisted_decision_and_does_not_re_decide(tmp_path) -> None:
    broker = AnsweringVenue()
    ctx = _ctx(tmp_path, broker)
    broker.fail_after_n_submits = 1  # accept AAA, die before BBB
    with pytest.raises(ConnectionError):
        decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4, BBB=0.4), _PRICES)
    broker.open_ids.add("2026-03-04:AAA")  # the accepted one is working at the venue

    # A restarted process: a brand new context over the same durable state.
    broker.fail_after_n_submits = None
    restarted = _ctx(tmp_path, broker)
    recovery = recover_on_restart(restarted, reconcile=False)
    assert recovery.decision_bar == "2026-03-04"
    assert recovery.resolved and recovery.may_submit
    assert {f.client_order_id: f.fate for f in recovery.fates} == {
        "2026-03-04:AAA": FATE_WORKING,
        "2026-03-04:BBB": FATE_UNSUBMITTED,
    }
    resumed = decide_and_submit(restarted, "2026-03-04", _targets(AAA=0.9), _PRICES)
    assert {o.client_order_id for o in resumed} == {"2026-03-04:AAA", "2026-03-04:BBB"}
    assert broker.submit_calls == 2  # AAA was not re-submitted


def test_restart_recovery_over_a_state_file_with_no_pending_orders_is_a_clean_resolved_read(tmp_path) -> None:
    state = LoopState(positions={"AAA": 10.0}, cash=1_000.0, last_settled_bar="2026-03-03")
    store = StateStore(tmp_path / "state.json")
    store.save(state)
    broker = AnsweringVenue()
    broker._positions = {"AAA": 10.0}
    broker._cash = 1_000.0
    recovery = recover_on_restart(_ctx(tmp_path, broker), reconcile=True)
    assert recovery.resolved and recovery.may_submit
    assert recovery.operating_states == frozenset()
    assert recovery.reconciliation is not None and recovery.reconciliation.clean


# ---------------------------------------------------------------------------
# the acceptance case: a copy of the live paper loop's version-2 state
# ---------------------------------------------------------------------------


@pytest.fixture()
def live_state_copy(tmp_path: Path) -> Iterator[tuple[Path, dict]]:
    """A **copy** of the live run's ``state.json``, never the file itself."""
    run_dir = _live_run_dir()
    if run_dir is None:
        pytest.skip("no live run directory holding runs/paper_loop_momentum2/state.json is visible")
    source = run_dir / "state.json"
    before = source.read_bytes()
    copy = tmp_path / "state.json"
    shutil.copy2(source, copy)
    yield copy, json.loads(before.decode("utf-8"))
    assert source.read_bytes() == before, "the live state file was modified by a test"


def test_a_restart_over_the_live_v2_state_migrates_it_and_recovers_clean(live_state_copy, tmp_path) -> None:
    copy, payload = live_state_copy
    if payload.get("schema_version") != 2:
        pytest.skip(f"the live document is at schema_version {payload.get('schema_version')!r}, not 2")
    broker = AnsweringVenue()
    broker._positions = dict(payload["positions"])
    broker._cash = float(payload["cash"])
    ctx = LiveLoopContext(
        store=StateStore(copy),
        broker=broker,
        fills_ledger=tmp_path / "fills.jsonl",
        execution_lock=ExecutionLockSpec(account_id=ACCOUNT, execution_host=HOST, lock_dir=tmp_path / "locks"),
    )
    recovery = recover_on_restart(ctx, reconcile=True)

    # Nothing ambiguous: the live book carries no pending order, so there is no
    # fate to establish and the account is in no restricting state at all.
    assert recovery.fates == ()
    assert recovery.resolved and recovery.may_submit
    assert recovery.operating_states == frozenset()
    assert recovery.reconciliation is not None and recovery.reconciliation.clean

    # The copy was migrated forward and saved at version 3 with the book intact.
    migrated = json.loads(copy.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 3
    assert migrated["portfolio"]["cash"] == pytest.approx(payload["cash"])
    assert len(migrated["portfolio"]["positions"]) == len(payload["positions"])
    assert migrated["last_settled_bar"] == payload["last_settled_bar"]
    assert migrated["last_refresh_bar"] == payload["last_refresh_bar"]
    assert migrated["pending_decision_bar"] == payload["pending_decision_bar"]


# ---------------------------------------------------------------------------
# the ambiguous submission: S3, C0 only, resolved before any retry
# ---------------------------------------------------------------------------


def _wedge_an_ambiguous_order(tmp_path: Path) -> tuple[LiveLoopContext, MuteVenue]:
    """Decide a bar, then restart against a venue that will not state a fate."""
    answering = AnsweringVenue()
    ctx = _ctx(tmp_path, answering)
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
    mute = MuteVenue()
    mute.orders = dict(answering.orders)  # the venue knows the id; it will not say more
    mute.no_fill = set(answering.orders)
    return _ctx(tmp_path, mute), mute


def test_an_unknown_fate_blocks_new_exposure(tmp_path) -> None:
    ctx, _mute = _wedge_an_ambiguous_order(tmp_path)
    recovery = pending_order_recovery(ctx, ctx.store.load())
    assert not recovery.resolved and not recovery.may_submit
    assert recovery.unknown_order_ids == ("2026-03-04:AAA",)
    assert recovery.operating_states == frozenset({OperatingState.S3_UNRESOLVED_SUBMISSION})

    with pytest.raises(UnresolvedSubmission, match="state S3"):
        decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.9), _PRICES)


def test_the_unknown_fate_state_permits_cancellation_and_nothing_else(tmp_path) -> None:
    ctx, mute = _wedge_an_ambiguous_order(tmp_path)
    recovery = pending_order_recovery(ctx, ctx.store.load())
    permitted = permitted_classes(recovery.operating_states)
    assert permitted == frozenset({ActionClass.C0})

    cancelled = cancel_orders(ctx, states=recovery.operating_states)
    assert cancelled == ["2026-03-04:AAA"]
    assert mute.cancelled == ["2026-03-04:AAA"]


class RegistrylessVenue(AnsweringVenue):
    """A venue that lists its open orders and cannot be asked what it ever took.

    The sweep's own guard does not catch this one: it has the open-order read
    the sweep demands. What it cannot answer is whether a parent order the
    engine believes it submitted ever reached the venue, and the residual the
    sweep would re-offer is the parent's quantity minus what printed.
    """

    def submitted_order_ids(self, since: str | None = None) -> set[str]:
        raise BrokerCapabilityUnsupported("RegistrylessVenue cannot list the ids it knows")


def test_the_completion_sweep_is_a_retry_and_refuses_while_a_fate_is_unknown(tmp_path) -> None:
    answering = AnsweringVenue()
    ctx = _ctx(tmp_path, answering)
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
    registryless = RegistrylessVenue()
    registryless.orders = dict(answering.orders)
    registryless.no_fill = set(answering.orders)
    with pytest.raises(UnresolvedSubmission, match="state S3"):
        sweep_pending(_ctx(tmp_path, registryless))


def test_resolving_the_ambiguity_cancels_it_records_it_and_lets_the_retry_proceed(tmp_path) -> None:
    ctx, mute = _wedge_an_ambiguous_order(tmp_path)
    resolved = resolve_unresolved_submission(ctx)
    assert resolved.resolved and resolved.may_submit
    assert mute.cancelled == ["2026-03-04:AAA"]

    # Durable: a second restart reads the settled fate instead of asking again,
    # and it does so even though the venue is still mute.
    reread = _ctx(tmp_path, mute)
    again = pending_order_recovery(reread, reread.store.load())
    assert again.resolved and again.fates == ()
    document = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    (record,) = document["portfolio"]["pending_orders"]
    assert record["status"] == "canceled"
    assert "S3" in str(record["reason"])

    # And the retry now runs rather than refusing.
    resumed = decide_and_submit(reread, "2026-03-04", _targets(AAA=0.9), _PRICES)
    assert [o.client_order_id for o in resumed] == ["2026-03-04:AAA"]


def test_a_cancelled_ambiguous_order_that_had_executed_still_settles(tmp_path) -> None:
    """Exactly-once delivery is not promised; the accounting is idempotent (§3.11)."""
    ctx, mute = _wedge_an_ambiguous_order(tmp_path)
    resolve_unresolved_submission(ctx)
    # It turns out the venue had it all along and it printed.
    mute.no_fill.clear()
    mute.open_prices["AAA"] = 101.0
    fills = settle(ctx, "2026-03-05")
    assert [f.client_order_id for f in fills] == ["2026-03-04:AAA"]
    ledger = [json.loads(line) for line in (tmp_path / "fills.jsonl").read_text().splitlines() if line.strip()]
    assert [row["client_order_id"] for row in ledger] == ["2026-03-04:AAA"]
    state = ctx.store.load()
    assert state.pending_orders == [] and state.last_settled_bar == "2026-03-05"


# ---------------------------------------------------------------------------
# the account execution lock: where it is held
# ---------------------------------------------------------------------------


def test_the_decision_path_holds_the_account_lock_across_the_write_ahead(tmp_path) -> None:
    ctx = _ctx(tmp_path, AnsweringVenue())
    with account_execution_lock(ACCOUNT, HOST, lock_dir=tmp_path / "locks"):
        with pytest.raises(LockRefused, match="second concurrent writer"):
            decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
    assert not (tmp_path / "state.json").exists()  # nothing was written under the refusal


@pytest.mark.parametrize("call", ["settle", "sweep", "cancel"])
def test_every_mutating_entry_point_holds_the_account_lock(tmp_path, call) -> None:
    ctx = _ctx(tmp_path, AnsweringVenue())
    decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
    with account_execution_lock(ACCOUNT, HOST, lock_dir=tmp_path / "locks"):
        with pytest.raises(LockRefused):
            if call == "settle":
                settle(ctx, "2026-03-05")
            elif call == "sweep":
                sweep_pending(ctx)
            else:
                cancel_orders(ctx)


def test_the_lock_is_re_entrant_within_one_invocation(tmp_path) -> None:
    """Settle inside the lock, then a decision inside the same held lock (§3.10)."""
    from prism.live.loop import held_execution_lock

    ctx = _ctx(tmp_path, AnsweringVenue())
    with held_execution_lock(ctx) as handle:
        assert handle is not None
        decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
        assert settle(ctx, "2026-03-05") != []
        assert decide_and_submit(ctx, "2026-03-06", _targets(AAA=0.5), _PRICES) is not None
    assert ctx._lock_depth == 0


def test_a_context_declaring_no_account_takes_no_account_lock(tmp_path) -> None:
    """The pre-authorization paper instrument keeps working exactly as today."""
    ctx = _ctx(tmp_path, AnsweringVenue(), lock=False)
    with account_execution_lock(ACCOUNT, HOST, lock_dir=tmp_path / "locks"):
        orders = decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)
    assert [o.client_order_id for o in orders] == ["2026-03-04:AAA"]


def test_the_lock_refuses_a_host_the_record_does_not_name(tmp_path) -> None:
    ctx = LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=AnsweringVenue(),
        fills_ledger=tmp_path / "fills.jsonl",
        execution_lock=ExecutionLockSpec(
            account_id=ACCOUNT, execution_host="some-other-host", lock_dir=tmp_path / "locks"
        ),
    )
    with pytest.raises(LockRefused, match="undeclared host"):
        decide_and_submit(ctx, "2026-03-04", _targets(AAA=0.4), _PRICES)


def test_a_fresh_loop_recovers_as_a_fresh_loop_and_not_as_an_empty_book(tmp_path) -> None:
    recovery = recover_on_restart(_ctx(tmp_path, AnsweringVenue()), reconcile=False)
    assert recovery.decision_bar is None and recovery.fates == ()
    assert recovery.resolved and recovery.may_submit
    assert not (tmp_path / "state.json").exists()


def test_the_recovery_serialises_to_something_a_record_can_carry(tmp_path) -> None:
    ctx, _mute = _wedge_an_ambiguous_order(tmp_path)
    payload = pending_order_recovery(ctx, ctx.store.load()).to_dict()
    assert payload["states"] == ["S3"]
    assert payload["unknown_order_ids"] == ["2026-03-04:AAA"]
    assert payload["fates"][0]["fate"] == FATE_UNKNOWN
    json.dumps(payload)  # a record has to be able to hold it
