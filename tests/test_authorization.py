"""§3.10 authorization: the record, the action table, and where the refusal lands.

Every refusal in this file is proved the same way, and the way matters more than
the exception type. The broker double :class:`RefusingBroker` records every call
and raises :class:`AssertionError` on each one, so a test that reaches the venue
fails *as a broken control*, not as an unexpected error. Asserting
``broker.calls == []`` afterwards then says the thing the specification actually
requires: the order request was never constructed, and nothing reached the
venue — as opposed to the venue having rejected it, which is a different and
much weaker fact.

The seven refusals §3.10 and the stage 5 exit require are the first group: no
record, a mismatched account id, a mismatched ``config_id``, a mismatched
``engine_version``, a mismatched execution host, an expired record, and a live
endpoint under paper mode. Each one is exercised through the real decision path
(``prism.live.loop.decide_and_submit``) and, where the adapter is the last line,
through ``AlpacaBroker.submit`` against a transport that fails on any request.

The second group is the scope separation the brief requires: a missing signal
and an expired research approval remove ``add_exposure`` and nothing else, so
cancellation and reduction stay available exactly where §3.10 says they do.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from prism.live.authorization import (
    AUTHORIZATION_FILENAME,
    LIVE_BASE_URL,
    PAPER_BASE_URL,
    ActionClass,
    ActionContext,
    AuthorizationGate,
    AuthorizationRecord,
    AuthorizationRefused,
    InputFreshness,
    OperatingState,
    ProcessIdentity,
    Scope,
    SessionIntent,
    classify_order,
    distribution_id,
    load_authorization,
    parse_authorization,
    permitted_classes,
    resolve_endpoint,
)
from prism.live.broker import Broker, Fill, Order
from prism.live.lockfile import LockRefused, account_execution_lock, account_lock_path
from prism.live.loop import (
    ExecutionLockSpec,
    LiveLoopContext,
    cancel_orders,
    decide_and_submit,
    execution_lock_spec,
)
from prism.live.state import LoopState, StateStore
from prism.spec import engine_version

HOST = "trader-1"
ACCOUNT_ID = "PA3RTY0001"
CONFIG_ID = "cfg-" + "0" * 8
ENGINE_VERSION = "0.4.0"
BUILD_ID = "git:" + "a" * 40
NOW = datetime(2026, 9, 9, 21, 30, tzinfo=timezone.utc)
FAR = (NOW + timedelta(days=30)).isoformat()
POLICY_IDS = {
    "profile": "p1",
    "costs": "c1",
    "execution": "e1",
    "monitoring": "m1",
    "recovery": "r1",
}


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class RefusingBroker(Broker):
    """A venue that must never be reached: every call records and fails the test.

    ``endpoint`` is a plain attribute, not a method, because the engine reads it
    on the path that must not touch the venue — asking an adapter where it
    points may not itself be a request.
    """

    def __init__(self, endpoint: str = LIVE_BASE_URL) -> None:
        self.endpoint = endpoint
        self.calls: list[tuple] = []

    def _reached(self, name: str, *args: object) -> None:
        self.calls.append((name, *args))
        raise AssertionError(
            f"the venue was reached: {name}{args!r}. The refusal must happen before any request is "
            "constructed (§3.10)"
        )

    def positions(self) -> dict[str, float]:
        self._reached("positions")
        return {}

    def cash(self) -> float:
        self._reached("cash")
        return 0.0

    def submit(self, order: Order) -> None:
        self._reached("submit", order.client_order_id)

    def submitted_order_ids(self, since: str | None = None) -> set[str]:
        self._reached("submitted_order_ids", since)
        return set()

    def fills_for(self, client_order_ids: set[str]) -> list[Fill]:
        self._reached("fills_for", sorted(client_order_ids))
        return []

    def cancel(self, client_order_id: str) -> None:
        self._reached("cancel", client_order_id)

    def open_order_ids(self, client_order_ids: set[str]) -> set[str]:
        self._reached("open_order_ids", sorted(client_order_ids))
        return set()


class WorkingBroker(Broker):
    """A venue that answers, and records every call it was asked to make.

    Used where the point is that an action *did* stay available: the recorded
    call list is then the evidence that the permitted write happened and that
    nothing else did.
    """

    def __init__(
        self,
        positions: dict[str, float] | None = None,
        cash: float = 100_000.0,
        endpoint: str = LIVE_BASE_URL,
    ) -> None:
        self.endpoint = endpoint
        self.calls: list[tuple] = []
        self._positions = dict(positions or {})
        self._cash = cash
        self.submitted: list[Order] = []
        self.cancelled: list[str] = []

    def positions(self) -> dict[str, float]:
        self.calls.append(("positions",))
        return dict(self._positions)

    def cash(self) -> float:
        self.calls.append(("cash",))
        return self._cash

    def submit(self, order: Order) -> None:
        self.calls.append(("submit", order.client_order_id))
        self.submitted.append(order)

    def submitted_order_ids(self, since: str | None = None) -> set[str]:
        self.calls.append(("submitted_order_ids", since))
        return {o.client_order_id for o in self.submitted}

    def fills_for(self, client_order_ids: set[str]) -> list[Fill]:
        self.calls.append(("fills_for", tuple(sorted(client_order_ids))))
        return []

    def cancel(self, client_order_id: str) -> None:
        self.calls.append(("cancel", client_order_id))
        self.cancelled.append(client_order_id)

    @property
    def writes(self) -> list[tuple]:
        """Only the calls that change something at the venue."""
        return [call for call in self.calls if call[0] in ("submit", "cancel")]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_record(**overrides: object) -> AuthorizationRecord:
    payload: dict[str, object] = {
        "schema": 1,
        "mode": "live",
        "account_name": "reference",
        "account_id": ACCOUNT_ID,
        "endpoint": LIVE_BASE_URL,
        "execution_host": HOST,
        "config_id": CONFIG_ID,
        "engine_version": ENGINE_VERSION,
        "distribution_id": BUILD_ID,
        "policy_ids": dict(POLICY_IDS),
        "scope": {"add_exposure": True, "manage_existing": True, "hedge": False},
        "submission_window_sessions": 1,
        "granted_by": "owner",
        "granted_at": NOW.isoformat(),
        "expires_at": FAR,
    }
    payload.update(overrides)
    return parse_authorization(payload)


def make_identity(**overrides: object) -> ProcessIdentity:
    values: dict[str, object] = {
        "mode": "live",
        "endpoint": LIVE_BASE_URL,
        "account_id": ACCOUNT_ID,
        "config_id": CONFIG_ID,
        "engine_version": ENGINE_VERSION,
        "distribution_id": BUILD_ID,
        "execution_host": HOST,
    }
    values.update(overrides)
    return ProcessIdentity(**values)  # type: ignore[arg-type]


def make_gate(record: AuthorizationRecord | None = None, **identity: object) -> AuthorizationGate:
    return AuthorizationGate(
        make_record() if record is None else record,
        make_identity(**identity),
        account="reference",
    )


def make_ctx(tmp_path: Path, broker: Broker, gate: AuthorizationGate | None, **state: object) -> LiveLoopContext:
    store = StateStore(tmp_path / "state.json")
    if state:
        store.save(LoopState(**state))  # type: ignore[arg-type]
    return LiveLoopContext(
        store=store,
        broker=broker,
        fills_ledger=tmp_path / "fills.jsonl",
        authorization=gate,
    )


PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0})


def targets(**weights: float) -> pd.Series:
    return pd.Series(weights, dtype=float)


def decide(ctx: LiveLoopContext, weights: pd.Series, **kwargs: object) -> list[Order]:
    return decide_and_submit(
        ctx,
        "2026-09-09",
        weights,
        PRICES,
        whole_shares=True,
        now=NOW,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# The seven refusals, each proved before any request exists
# ---------------------------------------------------------------------------


def test_live_order_refused_with_no_authorization_record(tmp_path) -> None:
    """S4b: no record for the named account, and every broker write is refused."""
    broker = RefusingBroker(endpoint=LIVE_BASE_URL)
    ctx = make_ctx(tmp_path, broker, None)
    with pytest.raises(AuthorizationRefused, match="no authorization record"):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []


def test_live_order_refused_on_a_mismatched_account_id(tmp_path) -> None:
    broker = RefusingBroker()
    ctx = make_ctx(tmp_path, broker, make_gate(account_id="PA-SOME-OTHER-BOOK"))
    with pytest.raises(AuthorizationRefused, match="account_id="):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []


def test_live_order_refused_on_a_mismatched_config_id(tmp_path) -> None:
    """Merging code never authorizes a live configuration (§3.10, §6 stage 6)."""
    broker = RefusingBroker()
    ctx = make_ctx(tmp_path, broker, make_gate(config_id="cfg-" + "9" * 8))
    with pytest.raises(AuthorizationRefused, match="config_id="):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []


def test_live_order_refused_on_a_mismatched_engine_version(tmp_path) -> None:
    broker = RefusingBroker()
    ctx = make_ctx(tmp_path, broker, make_gate(engine_version="0.4.1"))
    with pytest.raises(AuthorizationRefused, match="engine_version="):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []


def test_live_order_refused_on_a_mismatched_execution_host(tmp_path) -> None:
    """One execution host per account, checked in both modes (§3.10)."""
    broker = RefusingBroker()
    ctx = make_ctx(tmp_path, broker, make_gate(execution_host="trader-2"))
    with pytest.raises(AuthorizationRefused, match="execution_host="):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []


def test_live_order_refused_on_a_mismatched_build_identity(tmp_path) -> None:
    """A dirty checkout carrying the granted version string is still not it."""
    broker = RefusingBroker()
    ctx = make_ctx(tmp_path, broker, make_gate(distribution_id=BUILD_ID + "+dirty"))
    with pytest.raises(AuthorizationRefused, match="distribution_id="):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []


def test_live_order_refused_on_an_expired_record(tmp_path) -> None:
    """An expired grant cannot cover the submission window its batch is completed in."""
    broker = RefusingBroker()
    expired = make_record(expires_at=(NOW - timedelta(seconds=1)).isoformat())
    ctx = make_ctx(tmp_path, broker, make_gate(expired))
    with pytest.raises(AuthorizationRefused, match="expired at"):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []


def test_live_order_refused_when_the_grant_expires_inside_its_submission_window(tmp_path) -> None:
    """Not yet expired, but not covering the sweep that may complete the batch."""
    broker = RefusingBroker()
    soon = make_record(expires_at=(NOW + timedelta(hours=2)).isoformat())
    ctx = make_ctx(tmp_path, broker, make_gate(soon))
    with pytest.raises(AuthorizationRefused, match="inside the submission window"):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []


def test_live_order_refused_with_a_live_endpoint_under_paper_mode(tmp_path) -> None:
    """The process runs paper mode; the record and the client name the live venue."""
    broker = RefusingBroker(endpoint=LIVE_BASE_URL)
    ctx = make_ctx(tmp_path, broker, make_gate(mode="paper"))
    with pytest.raises(AuthorizationRefused, match="mode="):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []


def test_live_order_refused_when_the_client_points_off_the_record(tmp_path) -> None:
    """A paper record in front of a client that resolved the live endpoint."""
    broker = RefusingBroker(endpoint=LIVE_BASE_URL)
    paper = make_record(mode="paper", endpoint=PAPER_BASE_URL, expires_at=None)
    ctx = make_ctx(tmp_path, broker, make_gate(paper, mode="paper", endpoint=PAPER_BASE_URL))
    with pytest.raises(AuthorizationRefused, match="not the record's endpoint"):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []


def test_the_batch_is_refused_before_the_write_ahead_persist(tmp_path) -> None:
    """A refused batch leaves no pending state: nothing to resume, nothing to submit."""
    broker = WorkingBroker(positions={"AAA": 10.0})
    narrowed = make_record(scope={"add_exposure": False, "manage_existing": True, "hedge": False})
    ctx = make_ctx(tmp_path, broker, make_gate(narrowed))
    with pytest.raises(AuthorizationRefused, match="C3 not in"):
        decide(ctx, targets(AAA=0.9))
    assert broker.writes == []
    assert ctx.store.load() is None or ctx.store.load().pending_orders == []


# ---------------------------------------------------------------------------
# The adapter's last line: refused before the request body exists
# ---------------------------------------------------------------------------


class ExplodingSession:
    """Any HTTP request at all is a test failure."""

    def __init__(self) -> None:
        self.requests: list[tuple] = []

    def request(self, method: str, url: str, **kwargs: object):  # pragma: no cover - must not run
        self.requests.append((method, url))
        raise AssertionError(f"a request was constructed and sent: {method} {url}")


def test_the_adapter_refuses_a_live_client_with_no_record_before_building_the_body() -> None:
    from prism.live.alpaca import AlpacaBroker

    session = ExplodingSession()
    broker = AlpacaBroker("k", "s", base_url=LIVE_BASE_URL, mode="live", session=session)
    order = Order("2026-09-09:AAA", "AAA", 10.0, "2026-09-09", 100.0)
    with pytest.raises(AuthorizationRefused, match="no authorization record"):
        broker.submit(order)
    assert session.requests == []


def test_the_adapter_refuses_a_mismatched_record_before_building_the_body() -> None:
    from prism.live.alpaca import AlpacaBroker

    session = ExplodingSession()
    broker = AlpacaBroker(
        "k",
        "s",
        base_url=LIVE_BASE_URL,
        mode="live",
        session=session,
        authorization=make_gate(account_id="PA-OTHER"),
    )
    with pytest.raises(AuthorizationRefused, match="account_id="):
        broker.submit(Order("2026-09-09:AAA", "AAA", 10.0, "2026-09-09", 100.0))
    assert session.requests == []


def test_the_adapter_refuses_a_live_endpoint_under_paper_mode_at_construction() -> None:
    from prism.live.alpaca import AlpacaBroker

    with pytest.raises(AuthorizationRefused, match="live endpoint under mode 'paper'"):
        AlpacaBroker("k", "s", base_url=LIVE_BASE_URL, mode="paper", session=ExplodingSession())
    with pytest.raises(AuthorizationRefused, match="paper endpoint under mode 'live'"):
        AlpacaBroker("k", "s", base_url=PAPER_BASE_URL, mode="live", session=ExplodingSession())


def test_the_environment_may_set_an_endpoint_but_never_the_mode(monkeypatch) -> None:
    from prism.live.alpaca import AlpacaBroker

    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.setenv("APCA_API_BASE_URL", LIVE_BASE_URL)
    monkeypatch.setenv("PRISM_MODE", "live")  # no such thing: the mode is an argument
    with pytest.raises(AuthorizationRefused, match="live endpoint under mode 'paper'"):
        AlpacaBroker.from_env(session=ExplodingSession())
    monkeypatch.setenv("APCA_API_BASE_URL", PAPER_BASE_URL)
    assert AlpacaBroker.from_env(session=ExplodingSession()).endpoint == PAPER_BASE_URL


# ---------------------------------------------------------------------------
# Scope separation: what an expired grant and a missing signal do NOT disable
# ---------------------------------------------------------------------------


def test_cancellation_survives_an_expired_grant(tmp_path) -> None:
    """S4a keeps C0. An expired research approval is not a reason to strand orders.

    The proof is positive here: the cancel calls the venue received are exactly
    the ids asked for, and no other write happened.
    """
    broker = WorkingBroker()
    expired = make_record(expires_at=(NOW - timedelta(days=1)).isoformat())
    ctx = make_ctx(
        tmp_path,
        broker,
        make_gate(expired),
        pending_orders=[Order("2026-09-08:AAA", "AAA", 5.0, "2026-09-08", 100.0)],
        pending_decision_bar="2026-09-08",
    )
    assert cancel_orders(ctx, now=NOW) == ["2026-09-08:AAA"]
    assert broker.cancelled == ["2026-09-08:AAA"]
    assert [call for call in broker.writes if call[0] == "submit"] == []


def test_reduction_survives_an_expired_grant_and_a_rebalance_does_not(tmp_path) -> None:
    """The brief's rule, both halves, on one grant and one held book.

    A ``MANAGE_EXISTING`` session reduces the position and reaches the venue; the
    same grant refuses a rebalance that would add exposure, before any call.
    """
    expired = make_record(expires_at=(NOW - timedelta(days=1)).isoformat())

    working = WorkingBroker(positions={"AAA": 100.0}, cash=10_000.0)
    ctx = make_ctx(tmp_path, working, make_gate(expired))
    orders = decide(ctx, targets(AAA=0.10), intent=SessionIntent.MANAGE_EXISTING)
    assert orders and all(o.qty < 0 for o in orders)  # a reduction, and it was submitted
    assert [call for call in working.writes if call[0] == "submit"]

    refusing = RefusingBroker()
    (tmp_path / "second").mkdir()
    ctx2 = make_ctx(tmp_path / "second", refusing, make_gate(expired))
    with pytest.raises(AuthorizationRefused, match="expired at"):
        decide(ctx2, targets(AAA=0.9))
    assert refusing.calls == []


def test_a_stale_signal_keeps_reduction_and_cancellation(tmp_path) -> None:
    """S1: the signal is stale, the mark and halt status are not. C0, C1, OB stay."""
    broker = WorkingBroker(positions={"AAA": 100.0}, cash=10_000.0)
    ctx = make_ctx(tmp_path, broker, make_gate())
    states = [OperatingState.S1_STALE_ALPHA]
    orders = decide(
        ctx,
        targets(AAA=0.10),
        intent=SessionIntent.MANAGE_EXISTING,
        states=states,
    )
    assert orders and all(o.qty < 0 for o in orders)
    assert cancel_orders(ctx, ["2026-09-08:AAA"], states=states, now=NOW) == ["2026-09-08:AAA"]
    assert broker.cancelled == ["2026-09-08:AAA"]


def test_a_stale_signal_still_refuses_new_exposure(tmp_path) -> None:
    """The other half of S1: C3 is out, and the refusal names the sid."""
    broker = WorkingBroker(positions={"AAA": 100.0}, cash=10_000.0)
    ctx = make_ctx(tmp_path, broker, make_gate())
    with pytest.raises(AuthorizationRefused, match="BBB: C3 not in"):
        decide(ctx, targets(AAA=0.5, BBB=0.3), states=[OperatingState.S1_STALE_ALPHA])
    assert broker.writes == []


def test_no_record_refuses_even_a_cancel(tmp_path) -> None:
    """S4b is the one row that removes C0 as well: nothing, C0 included."""
    broker = RefusingBroker()
    ctx = make_ctx(tmp_path, broker, AuthorizationGate(None, make_identity(), account="reference"))
    with pytest.raises(AuthorizationRefused, match="no authorization record"):
        cancel_orders(ctx, ["2026-09-08:AAA"], now=NOW)
    assert broker.calls == []


def test_a_halt_refuses_the_engines_own_cancel_but_not_the_operators(tmp_path) -> None:
    """S6: "C0 only, and the engine performs no cancel of its own"."""
    broker = WorkingBroker()
    ctx = make_ctx(tmp_path, broker, make_gate())
    with pytest.raises(AuthorizationRefused, match="C0 on 2026-09-08:AAA is not permitted"):
        cancel_orders(ctx, ["2026-09-08:AAA"], states=[OperatingState.S6_HALT], now=NOW)
    assert broker.calls == []
    assert cancel_orders(
        ctx,
        ["2026-09-08:AAA"],
        states=[OperatingState.S6_HALT],
        action_context=ActionContext(operator_initiated=True),
        now=NOW,
    ) == ["2026-09-08:AAA"]
    assert broker.cancelled == ["2026-09-08:AAA"]


def test_a_manage_existing_session_may_not_smuggle_new_exposure(tmp_path) -> None:
    """The intent is a declaration, and a batch outside it is refused as a unit."""
    broker = WorkingBroker(positions={"AAA": 100.0}, cash=10_000.0)
    ctx = make_ctx(tmp_path, broker, make_gate())
    with pytest.raises(AuthorizationRefused, match="outside its declared classes"):
        decide(ctx, targets(AAA=0.10, BBB=0.20), intent=SessionIntent.MANAGE_EXISTING)
    assert broker.writes == []


def test_an_unauthorized_context_in_front_of_a_live_client_refuses(tmp_path) -> None:
    """No gate at all plus a live endpoint is S4b, not a legacy pass-through."""
    broker = RefusingBroker(endpoint=LIVE_BASE_URL)
    ctx = make_ctx(tmp_path, broker, None)
    with pytest.raises(AuthorizationRefused, match="carries no authorization record"):
        cancel_orders(ctx, ["x"], now=NOW)
    assert broker.calls == []


def test_the_pre_authorization_paper_instrument_still_runs(tmp_path) -> None:
    """Backward compatibility: no gate, no live endpoint, no behaviour change."""
    broker = WorkingBroker(positions={}, cash=10_000.0, endpoint=PAPER_BASE_URL)
    ctx = make_ctx(tmp_path, broker, None)
    orders = decide(ctx, targets(AAA=0.5))
    assert orders and [call for call in broker.writes if call[0] == "submit"]


# ---------------------------------------------------------------------------
# The table: how the rows combine
# ---------------------------------------------------------------------------

FULL = Scope(add_exposure=True, manage_existing=True, hedge=True)


def permitted(*states: OperatingState, scope: Scope = FULL, **context: object) -> set[str]:
    allowed = permitted_classes(states, scope, ActionContext(**context))  # type: ignore[arg-type]
    return {action.value for action in allowed}


def test_rows_combine_by_intersection_not_by_precedence() -> None:
    """S1 and S4a permit disjoint-ish sets; "most restrictive row" is undefined there.

    S1 permits {C0, C1, C4, OB} (C4 with fresh risk inputs) and S4a permits
    {C0, C1, OB} plus C4 only with ``hedge``. Neither is a subset of the other,
    so the combination is the intersection and nothing else.
    """
    s1 = permitted(OperatingState.S1_STALE_ALPHA, risk_inputs_fresh=True, hedge_risk_measure_falls=True)
    s4a = permitted(OperatingState.S4A_AUTHORIZATION_NARROWED, hedge_risk_measure_falls=True)
    both = permitted(
        OperatingState.S1_STALE_ALPHA,
        OperatingState.S4A_AUTHORIZATION_NARROWED,
        risk_inputs_fresh=True,
        hedge_risk_measure_falls=True,
    )
    assert "C4" in s1 and "C4" in s4a
    assert both == s1 & s4a
    assert "C3" not in both and {"C0", "C1", "OB"} <= both


def test_s4b_empties_every_combination() -> None:
    assert permitted(OperatingState.S4B_AUTHORIZATION_ABSENT) == set()
    assert permitted(OperatingState.S0_CLEAN, OperatingState.S4B_AUTHORIZATION_ABSENT) == set()


def test_c2_is_permitted_exactly_when_both_legs_are() -> None:
    """C2 is a C1 leg and a C3 leg, each authorized on its own."""
    assert "C2" in permitted()
    assert "C2" not in permitted(scope=Scope(add_exposure=False, manage_existing=True, hedge=True))
    assert "C2" not in permitted(scope=Scope(add_exposure=True, manage_existing=False, hedge=True))
    assert "C2" not in permitted(OperatingState.S1_STALE_ALPHA)  # C3 is gone, so C2 is


def test_c5_is_c1_in_a_full_liquidation_and_c3_otherwise() -> None:
    """That is what makes a full liquidation under a C3-forbidding state defined."""
    narrowed = Scope(add_exposure=False, manage_existing=True, hedge=True)
    assert "C5" not in permitted(scope=narrowed)  # tracks C3, which is gone
    assert "C5" in permitted(scope=narrowed, full_liquidation=True)  # tracks C1, which is not
    # And the alternate scope row: manage_existing alone carries C5 in a liquidation.
    assert "C5" in permitted(
        scope=Scope(add_exposure=True, manage_existing=True, hedge=False),
        full_liquidation=True,
    )


def test_c4_needs_the_validator_to_have_shown_the_risk_measure_falling() -> None:
    assert "C4" not in permitted()
    assert "C4" in permitted(hedge_risk_measure_falls=True)
    assert "C4" not in permitted(
        scope=Scope(add_exposure=True, manage_existing=True, hedge=False),
        hedge_risk_measure_falls=True,
    )


def test_per_sid_freshness_degrades_that_sid_only() -> None:
    """A sid failing every class is C0 only; the account is not the unit."""
    everything_stale = InputFreshness(mark=False, halt=False, covariance=False, signal=False)
    assert permitted(freshness=everything_stale) == {"C0"}
    assert "C3" in permitted(freshness=InputFreshness())


def test_a_reduction_to_flat_survives_a_stale_mark_and_a_partial_one_does_not() -> None:
    stale_mark = InputFreshness(mark=False)
    assert "C1" not in permitted(freshness=stale_mark)
    assert "C1" in permitted(freshness=stale_mark, reduction_to_flat=True)
    assert "C3" not in permitted(freshness=stale_mark, reduction_to_flat=True)


def test_a_stale_covariance_removes_the_hedge_and_nothing_else() -> None:
    stale_cov = InputFreshness(covariance=False)
    allowed = permitted(freshness=stale_cov, hedge_risk_measure_falls=True)
    assert "C4" not in allowed
    assert {"C0", "C1", "C3", "OB"} <= allowed


def test_s2_permits_c0_only_after_a_fresh_open_order_read() -> None:
    assert permitted(OperatingState.S2_STALE_BROKER_STATE) == set()
    assert permitted(OperatingState.S2_STALE_BROKER_STATE, open_orders_read_fresh=True) == {"C0"}


def test_s5_admits_c3_only_within_achieved_limits() -> None:
    assert "C3" not in permitted(OperatingState.S5_BREACHED_LIMITS)
    assert "C3" in permitted(OperatingState.S5_BREACHED_LIMITS, within_achieved_limits=True)


def test_s3_s7_s8_are_c0_and_reconcile_only() -> None:
    for state in (
        OperatingState.S3_UNRESOLVED_SUBMISSION,
        OperatingState.S7_ACCOUNTING_DISCREPANCY,
        OperatingState.S8_CLOCK_FAULT,
    ):
        assert permitted(state) == {"C0"}, state


def test_s9_is_a_monitor_row_that_writes_nothing() -> None:
    assert permitted(OperatingState.S9_SCHEDULER_FAULT) == set()


def test_scope_flags_apply_in_every_state_not_only_the_rows_that_name_them() -> None:
    manage_only = Scope(add_exposure=False, manage_existing=True, hedge=False)
    for state in (OperatingState.S0_CLEAN, OperatingState.S5_BREACHED_LIMITS):
        allowed = permitted(state, scope=manage_only, within_achieved_limits=True)
        assert "C3" not in allowed and "C0" in allowed


def test_classify_order_covers_the_five_position_cases() -> None:
    held = {"AAA": 10.0, "SSS": -10.0, "SPY": 5.0}
    assert classify_order("BBB", 3.0, held) is ActionClass.C3  # open
    assert classify_order("AAA", 3.0, held) is ActionClass.C3  # add
    assert classify_order("AAA", -3.0, held) is ActionClass.C1  # reduce
    assert classify_order("AAA", -10.0, held) is ActionClass.C1  # to flat
    assert classify_order("AAA", -15.0, held) is ActionClass.C2  # cross
    assert classify_order("SSS", 5.0, held) is ActionClass.C1  # cover part of a short
    assert classify_order("SPY", 2.0, held, hedge_sids=["SPY"]) is ActionClass.C4
    assert classify_order("SPY", -2.0, held, hedge_sids=["SPY"]) is ActionClass.C5


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_record_round_trips_through_the_declared_shape(tmp_path) -> None:
    record = make_record()
    path = tmp_path / AUTHORIZATION_FILENAME
    path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    loaded = load_authorization(tmp_path)
    assert loaded is not None
    assert loaded.to_dict() == record.to_dict()
    assert loaded.scope == Scope(add_exposure=True, manage_existing=True, hedge=False)


def test_an_absent_record_is_absence_and_a_corrupt_one_is_an_error(tmp_path) -> None:
    assert load_authorization(tmp_path) is None
    (tmp_path / AUTHORIZATION_FILENAME).write_text("{ not json", encoding="utf-8")
    with pytest.raises(AuthorizationRefused, match="cannot be read"):
        load_authorization(tmp_path)


def test_an_unknown_key_or_a_missing_field_authorizes_nothing() -> None:
    payload = make_record().to_dict()
    with pytest.raises(AuthorizationRefused, match="unknown key"):
        parse_authorization({**payload, "notes": "looks harmless"})
    del payload["execution_host"]
    with pytest.raises(AuthorizationRefused, match="missing required field"):
        parse_authorization(payload)


def test_live_requires_an_expiry_and_paper_does_not() -> None:
    with pytest.raises(AuthorizationRefused, match="expires_at is required"):
        make_record(expires_at=None)
    assert make_record(mode="paper", endpoint=PAPER_BASE_URL, expires_at=None).expiry is None


def test_a_naive_expiry_is_refused() -> None:
    with pytest.raises(AuthorizationRefused, match="carries no UTC offset"):
        make_record(expires_at="2026-12-31T00:00:00")


def test_another_schema_is_a_hard_stop() -> None:
    with pytest.raises(AuthorizationRefused, match="schema"):
        make_record(schema=2)


def test_policy_ids_mirror_every_key_of_the_policy_block() -> None:
    with pytest.raises(AuthorizationRefused, match="policy_ids is missing"):
        make_record(policy_ids={key: "x" for key in ("profile", "costs", "execution")})


def test_resolved_policy_ids_are_refused_rather_than_reconciled() -> None:
    gate = make_gate()
    gate.verify_policies(POLICY_IDS)
    with pytest.raises(AuthorizationRefused, match="disagree with the authorization record"):
        gate.verify_policies({**POLICY_IDS, "monitoring": "m2"})


def test_resolve_endpoint_is_an_exact_per_mode_allowlist() -> None:
    assert resolve_endpoint("paper") == PAPER_BASE_URL
    assert resolve_endpoint("live", base_url=LIVE_BASE_URL) == LIVE_BASE_URL
    assert resolve_endpoint("paper", from_environment=PAPER_BASE_URL + "/") == PAPER_BASE_URL
    with pytest.raises(AuthorizationRefused, match="unknown endpoint under mode 'paper'"):
        resolve_endpoint("paper", from_environment="https://paper-api.alpaca.markets.evil.test")
    with pytest.raises(AuthorizationRefused, match="there is no default"):
        resolve_endpoint("PAPER")


def test_an_expired_grant_narrows_the_scope_and_does_not_void_the_record() -> None:
    """S4a, stated as the gate sees it: add_exposure goes, the other two stay."""
    expired = make_gate(make_record(expires_at=(NOW - timedelta(days=1)).isoformat()))
    assert expired.expired(now=NOW)
    assert expired.scope(now=NOW) == Scope(add_exposure=False, manage_existing=True, hedge=False)
    assert expired.authorization_states(now=NOW) == frozenset({OperatingState.S4A_AUTHORIZATION_NARROWED})
    assert {a.value for a in expired.permitted(now=NOW)} == {"C0", "C1", "OB"}


def test_a_gate_with_no_record_is_s4b() -> None:
    gate = AuthorizationGate(None, make_identity(), account="reference")
    assert gate.authorization_states(now=NOW) == frozenset({OperatingState.S4B_AUTHORIZATION_ABSENT})
    assert gate.permitted(now=NOW) == frozenset()
    with pytest.raises(AuthorizationRefused, match="C0 is not permitted"):
        gate.authorize(ActionClass.C0, now=NOW)


# ---------------------------------------------------------------------------
# The resume and sweep paths: binding is re-verified, scope is not re-evaluated
# ---------------------------------------------------------------------------


def test_a_resumed_decision_re_verifies_the_binding_before_submitting(tmp_path) -> None:
    """The persisted batch is not re-scoped, but it is not submitted by any host either."""
    pending = [Order("2026-09-09:AAA", "AAA", 5.0, "2026-09-09", 100.0)]
    broker = RefusingBroker()
    ctx = make_ctx(
        tmp_path,
        broker,
        make_gate(execution_host="trader-2"),
        pending_orders=pending,
        pending_decision_bar="2026-09-09",
    )
    with pytest.raises(AuthorizationRefused, match="execution_host="):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []


def test_a_resumed_decision_is_not_re_scoped_by_a_later_expiry(tmp_path) -> None:
    """§3.10: a later expiry does not split a batch that was authorized as a unit."""
    from prism.live.loop import submit_pending

    broker = WorkingBroker()
    expired = make_gate(make_record(expires_at=(NOW - timedelta(days=1)).isoformat()))
    orders = [Order("2026-09-09:AAA", "AAA", 5.0, "2026-09-09", 100.0)]
    ctx = make_ctx(tmp_path, broker, expired, pending_orders=orders, pending_decision_bar="2026-09-09")
    submit_pending(ctx, ctx.store.load(), orders, gate=expired)
    assert [o.client_order_id for o in broker.submitted] == ["2026-09-09:AAA"]


def test_the_sweep_refuses_a_mis_bound_gate_before_reading_the_venue(tmp_path) -> None:
    from prism.live.loop import sweep_pending

    broker = RefusingBroker()
    ctx = make_ctx(
        tmp_path,
        broker,
        make_gate(config_id="cfg-" + "9" * 8),
        pending_orders=[Order("2026-09-09:AAA", "AAA", 5.0, "2026-09-09", 100.0)],
        pending_decision_bar="2026-09-09",
    )
    with pytest.raises(AuthorizationRefused, match="config_id="):
        sweep_pending(ctx)
    assert broker.calls == []


def test_a_stale_broker_state_permits_the_cancel_only_after_a_fresh_open_order_read(tmp_path) -> None:
    """S2, through the real cancel path: the read is the precondition, not a formality."""
    broker = WorkingBroker()
    ctx = make_ctx(tmp_path, broker, make_gate())
    with pytest.raises(AuthorizationRefused, match="C0 on 2026-09-08:AAA is not permitted"):
        cancel_orders(ctx, ["2026-09-08:AAA"], states=[OperatingState.S2_STALE_BROKER_STATE], now=NOW)
    assert broker.calls == []
    assert cancel_orders(
        ctx,
        ["2026-09-08:AAA"],
        states=[OperatingState.S2_STALE_BROKER_STATE],
        action_context=ActionContext(open_orders_read_fresh=True),
        now=NOW,
    ) == ["2026-09-08:AAA"]
    assert broker.cancelled == ["2026-09-08:AAA"]


def test_a_coverage_shortfall_blocks_classes_per_sid_not_per_account() -> None:
    """The batch is the unit of authorization; freshness is still per sid (§3.10, §3.11).

    One sid whose inputs are stale refuses the batch and the refusal names *that*
    sid; the fresh sid's own classes are untouched, which is what "blocks classes
    per sid, not the whole account" has to mean for a mechanism rather than a
    slogan.
    """
    gate = make_gate()
    stale = {"BBB": InputFreshness(mark=False, signal=False)}
    gate.authorize_batch(
        {"AAA": ActionClass.C3, "BBB": ActionClass.C1},
        freshness_by_sid={"BBB": InputFreshness(mark=False, signal=False)},
        context=ActionContext(reduction_to_flat=True),
        now=NOW,
    )
    with pytest.raises(AuthorizationRefused, match=r"BBB: C3 not in"):
        gate.authorize_batch(
            {"AAA": ActionClass.C3, "BBB": ActionClass.C3},
            freshness_by_sid=stale,
            now=NOW,
        )


class RecordingSession:
    """Accepts anything and records it — the counterpart to :class:`ExplodingSession`."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, object]] = []

    def request(self, method: str, url: str, headers=None, timeout=None, json=None, params=None):
        self.requests.append((method, url, json))

        class _Response:
            status_code = 200
            text = "{}"

            @staticmethod
            def json() -> dict:
                return {}

        return _Response()


def test_binding_a_record_is_what_lets_a_live_client_submit() -> None:
    """The binding order §3.10 implies: read the account, then bind, then submit.

    The gate needs the broker-reported account id and the broker needs a client
    to read it with, so the record is bound after construction. Before the bind
    the adapter refuses and the session sees nothing; after it, and only after
    it, the order request is built.
    """
    from prism.live.alpaca import AlpacaBroker

    session = RecordingSession()
    broker = AlpacaBroker("k", "s", base_url=LIVE_BASE_URL, mode="live", session=session)
    order = Order("2026-09-09:AAA", "AAA", 10.0, "2026-09-09", 100.0)
    with pytest.raises(AuthorizationRefused, match="no authorization record"):
        broker.submit(order)
    assert session.requests == []

    broker.bind_authorization(make_gate())
    broker.submit(order)
    assert [(method, url) for method, url, _ in session.requests] == [("POST", LIVE_BASE_URL + "/v2/orders")]


# ---------------------------------------------------------------------------
# The lock the record declares is the lock the session takes
# ---------------------------------------------------------------------------


class LockProbingBroker(WorkingBroker):
    """Asks, from inside the decision, whether a second writer could take this lock.

    ``positions()`` is the decision's first broker read and it happens after the
    state load, so it runs inside whatever lock the invocation took. Trying to
    acquire the same account lock from there is the only way to assert that the
    lock is *held* rather than merely constructible: a recorded ``"refused"``
    means a second process on this host would have been turned away at the exact
    moment this one was between its write-ahead persist and its submit.
    """

    def __init__(self, account_id: str, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.account_id = account_id
        self.second_writer: str | None = None

    def positions(self) -> dict[str, float]:
        try:
            with account_execution_lock(self.account_id, socket.gethostname()):
                self.second_writer = "granted"
        except LockRefused:
            self.second_writer = "refused"
        return super().positions()


def test_an_authorized_session_takes_its_account_lock_without_a_driver_arming_it(tmp_path, monkeypatch) -> None:
    """§3.10: the record declares the account and the host, so it declares the lock.

    Nothing in this test passes ``execution_lock=``. The record does it, because
    a control that arms only when a driver remembers to arm it is not a control.
    The proof is not that a spec object exists: it is that a second acquisition
    from inside the decision is refused, and that the lock file the record names
    is the one that was taken.
    """
    lock_dir = tmp_path / "locks"
    monkeypatch.setattr("prism.live.lockfile.default_lock_dir", lambda: lock_dir)
    host = socket.gethostname()
    gate = make_gate(make_record(execution_host=host), execution_host=host)
    broker = LockProbingBroker(ACCOUNT_ID, positions={"AAA": 10.0}, cash=100_000.0)
    ctx = make_ctx(tmp_path, broker, gate)

    assert ctx.execution_lock is None, "the driver armed nothing; the record is the declaration"
    assert execution_lock_spec(ctx) == ExecutionLockSpec(account_id=ACCOUNT_ID, execution_host=host)

    orders = decide(ctx, targets(AAA=0.9))
    assert orders and [call for call in broker.writes if call[0] == "submit"]
    assert broker.second_writer == "refused"
    assert account_lock_path(ACCOUNT_ID, lock_dir).exists()


def test_the_derived_lock_is_the_account_the_record_names_and_not_the_run_directory(tmp_path, monkeypatch) -> None:
    """Two run directories, one account: the key is the account, and it is keyed once."""
    lock_dir = tmp_path / "locks"
    monkeypatch.setattr("prism.live.lockfile.default_lock_dir", lambda: lock_dir)
    host = socket.gethostname()
    gate = make_gate(make_record(execution_host=host), execution_host=host)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = make_ctx(tmp_path / "a", WorkingBroker(), gate)
    second = make_ctx(tmp_path / "b", WorkingBroker(), gate)
    derived = execution_lock_spec(first)
    assert derived == execution_lock_spec(second)
    assert derived == ExecutionLockSpec(account_id=ACCOUNT_ID, execution_host=host)
    assert account_lock_path(ACCOUNT_ID, lock_dir).name.startswith(ACCOUNT_ID)


def test_a_record_naming_another_host_refuses_at_the_binding_and_takes_no_lock(tmp_path, monkeypatch) -> None:
    """The refusal that names the cause is the one that fires.

    Deriving the lock unconditionally would replace ``execution_host=`` — a
    refusal that says which host the owner approved — with a lock refusal
    underneath it. The session is refused either way; this pins *which* control
    refuses, and that no lock file is created on the way.
    """
    lock_dir = tmp_path / "locks"
    monkeypatch.setattr("prism.live.lockfile.default_lock_dir", lambda: lock_dir)
    broker = RefusingBroker()
    ctx = make_ctx(
        tmp_path, broker, make_gate(make_record(execution_host="trader-2"), execution_host=socket.gethostname())
    )

    assert execution_lock_spec(ctx) is None
    with pytest.raises(AuthorizationRefused, match="execution_host="):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []
    assert not lock_dir.exists()


def test_the_pre_authorization_paper_instrument_takes_no_account_lock(tmp_path, monkeypatch) -> None:
    """No record, nothing to derive from, and the run-directory lock still guards it."""
    lock_dir = tmp_path / "locks"
    monkeypatch.setattr("prism.live.lockfile.default_lock_dir", lambda: lock_dir)
    ctx = make_ctx(tmp_path, WorkingBroker(endpoint=PAPER_BASE_URL), None)
    assert execution_lock_spec(ctx) is None
    assert decide(ctx, targets(AAA=0.5))
    assert not lock_dir.exists()


def test_an_explicit_spec_still_overrides_the_derivation(tmp_path, monkeypatch) -> None:
    """The override the operator keeps: a lock directory somewhere else."""
    monkeypatch.setattr("prism.live.lockfile.default_lock_dir", lambda: tmp_path / "unused")
    host = socket.gethostname()
    gate = make_gate(make_record(execution_host=host), execution_host=host)
    ctx = make_ctx(tmp_path, WorkingBroker(), gate)
    ctx.execution_lock = ExecutionLockSpec(ACCOUNT_ID, host, lock_dir=tmp_path / "elsewhere")
    assert execution_lock_spec(ctx).lock_dir == tmp_path / "elsewhere"
    assert decide(ctx, targets(AAA=0.5))
    assert account_lock_path(ACCOUNT_ID, tmp_path / "elsewhere").exists()
    assert not (tmp_path / "unused").exists()


# ---------------------------------------------------------------------------
# The build identity a running process actually computes (§3.10 [Q18])
#
# Every refusal above is proved against a `ProcessIdentity` a test wrote down.
# That is the right way to test the *table*, and the wrong way to test the
# *stamp*: a version string alone cannot distinguish the pinned release a grant
# was made against from a modified checkout carrying the same string, and the
# thing that closes that gap is what `distribution_id()` computes on the host,
# not what a test can assert about a literal. So these cases run the real
# function, against a real git checkout and against the installed-distribution
# path, and then close the loop by refusing a record that names the same commit
# in its *clean* form while this process runs a dirty one.
# ---------------------------------------------------------------------------

#: ``git:<40 hex>`` with an optional ``+dirty``, or ``sha256:<64 hex>``.
STAMP = re.compile(r"^(git:[0-9a-f]{40}(\+dirty)?|sha256:[0-9a-f]{64})$")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(repo), "GIT_CONFIG_NOSYSTEM": "1"},
    )
    return result.stdout.strip()


@pytest.fixture()
def checkout(tmp_path) -> Path:
    """A real git checkout with one commit, and one tracked file to dirty."""
    repo = tmp_path / "checkout"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "pkg/module.py")
    _git(repo, "commit", "-q", "-m", "one")
    return repo


def test_a_checkout_stamps_its_commit_and_a_dirty_tree_says_so(checkout: Path) -> None:
    """The editable case: the stamp is the commit, and a modified tree is not it."""
    head = _git(checkout, "rev-parse", "HEAD")
    clean = distribution_id(package_root=checkout / "pkg")
    assert clean == f"git:{head}" and STAMP.match(clean)

    (checkout / "pkg" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    dirty = distribution_id(package_root=checkout / "pkg")
    assert dirty == f"git:{head}+dirty"

    # The two are different identities of the same version string, which is the
    # whole point of binding the build rather than the version (§3.10 [Q18]).
    assert clean != dirty


def test_an_installed_distribution_stamps_the_hash_of_its_RECORD(tmp_path, monkeypatch) -> None:
    """Outside a checkout the stamp is the loaded distribution's own manifest."""
    outside = tmp_path / "not-a-checkout" / "prism"
    outside.mkdir(parents=True)

    class InstalledDistribution:
        def locate_file(self, name: str) -> Path:
            return outside.parent / name

        def read_text(self, name: str) -> str | None:
            if name == "direct_url.json":
                return None
            assert name == "RECORD"
            return "prism/__init__.py,sha256=abc,10\n"

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: InstalledDistribution(), raising=True)
    stamp = distribution_id(package_root=outside)
    expected = hashlib.sha256("prism/__init__.py,sha256=abc,10\n".encode("utf-8")).hexdigest()
    assert stamp == f"sha256:{expected}" and STAMP.match(stamp)


def test_a_build_that_can_identify_itself_neither_way_is_refused(tmp_path, monkeypatch) -> None:
    """An unidentifiable build can match no record, so it is not run against one."""

    def missing(name: str):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing, raising=True)
    outside = tmp_path / "not-a-checkout"
    outside.mkdir()
    with pytest.raises(AuthorizationRefused, match="cannot identify this build"):
        distribution_id(package_root=outside)


def test_the_process_identity_carries_the_stamp_this_engine_computes() -> None:
    """`ProcessIdentity.current` reads the host, not its arguments, for these three."""
    identity = ProcessIdentity.current(
        mode="paper",
        endpoint=PAPER_BASE_URL,
        account_id=ACCOUNT_ID,
        config_id=CONFIG_ID,
    )
    assert identity.distribution_id == distribution_id()
    assert STAMP.match(identity.distribution_id), identity.distribution_id
    assert identity.execution_host == socket.gethostname()
    assert identity.engine_version == engine_version()


def test_a_record_naming_another_build_of_the_same_version_is_refused(tmp_path) -> None:
    """The refusal, proved against the stamp this process actually computes.

    The record names the *other* identity of this same tree — the clean commit
    where this process is dirty, or the dirty one where it is clean — with the
    engine version and every other field identical. Nothing but the build
    differs, and the session is refused before any request.
    """
    running = ProcessIdentity.current(
        mode="live",
        endpoint=LIVE_BASE_URL,
        account_id=ACCOUNT_ID,
        config_id=CONFIG_ID,
    )
    other = (
        running.distribution_id[: -len("+dirty")]
        if running.distribution_id.endswith("+dirty")
        else running.distribution_id + "+dirty"
    )
    broker = RefusingBroker()
    granted_elsewhere = make_record(
        engine_version=running.engine_version,
        execution_host=running.execution_host,
        distribution_id=other,
    )
    ctx = make_ctx(tmp_path, broker, AuthorizationGate(granted_elsewhere, running, account="reference"))
    with pytest.raises(AuthorizationRefused, match="distribution_id="):
        decide(ctx, targets(AAA=0.5))
    assert broker.calls == []

    # The same record, naming the identity this process really has, binds.
    granted_here = make_record(
        engine_version=running.engine_version,
        execution_host=running.execution_host,
        distribution_id=running.distribution_id,
    )
    AuthorizationGate(granted_here, running, account="reference").verify_binding()
