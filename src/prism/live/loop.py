"""The write-ahead decision protocol (SPEC.md §7.7).

The crash window that matters is between the decision (close *t*) and the
fill (open *t+1*): a restarted process must neither lose the queued orders
nor submit them twice. The protocol:

1. **Reconcile.** Broker truth wins for positions and cash; a divergence
   from the persisted state is logged loudly and adopted, never papered
   over.
2. **Decide once.** Target weights (already constructed upstream against
   held weights — I-4) are diffed into share orders at decision-close
   prices.
3. **Write ahead.** The pending orders are persisted *before* the first
   submit. From that moment the decision for the bar is immutable: a
   restart resumes submission from the persisted record and never
   re-decides (re-deciding after a partial submit double-trades).
4. **Submit idempotently.** Every order carries a deterministic
   ``client_order_id`` (``"{prefix}{decision_bar}:{symbol}"`` — the prefix
   namespaces books sharing one venue account); ids the broker already
   knows are skipped, and a venue duplicate rejection counts as success.
   A venue *rejection* is not skipped: submission stops there, the rejected
   order is recorded as blocked and is not re-offered, and the portfolio the
   accepted legs imply is revalidated before anything further is submitted
   (§3.7; :func:`submit_pending`).
5. **Settle.** Next session, fills are pulled, appended to the I-9 fills
   ledger (fill price beside the decision-close reference price — the
   arrival-slippage record the paper instrument exists to collect), and
   the state is re-anchored to broker truth with pending cleared.

**Authorization sits in front of all of it** (§3.10). ``decide_and_submit``
verifies the account's authorization record *before* its reconcile read: a
process whose record names another account, endpoint, configuration, engine
version, build or host makes no broker call whatever, and neither does one whose
grant has expired. The order batch is then classified into the action classes of
§3.10 and authorized as a unit at write-ahead time, before the persist and
before the first submit, because dropping one side of a balanced batch can leave
the portfolio outside its limits (§3.7).

Management of existing positions is a separate authorization from opening new
exposure, and the split is visible in the entry points: :func:`cancel_orders`
and a ``SessionIntent.MANAGE_EXISTING`` decision need only ``manage_existing``
and survive a stale signal or an expired grant (S1, S4a), while a rebalance
needs ``add_exposure`` and a grant that covers its whole submission window.

**The account execution lock wraps each of them** (§3.10). Every entry point
that may write to the broker or to state takes it before its first state load
and releases it after its last state save, settle included, so two processes
cannot tear one account's state between the write-ahead persist and the submit.
The lock a context takes is the one its authorization record declares, and it
is armed *from* that record whenever the record names this host, so an
authorized session on its declared execution host always holds its account lock
without a driver having to remember to ask for it. A context with no record —
the pre-authorization paper instrument — takes none and is still guarded by the
run-directory lock its CLI holds, and so does a record naming another host,
which :meth:`AuthorizationGate.verify_binding` refuses before any broker call.

**Restart recovery decides whether a resumed decision may be retried at all**
(§3.11). The write-ahead record says what was decided; it does not say what
reached the venue. So a resumed decision first asks the venue for each
persisted order's fate — working, terminal, or never received — and a fate the
venue will not state is state S3: cancellation and reconciliation only, no
retry, until :func:`resolve_unresolved_submission` cancels the ambiguous orders
and records the outcome durably. Exactly-once delivery is not promised anywhere
here; the accounting is idempotent, which is why cancelling an order that
turned out to have executed still settles correctly.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Mapping, Sequence

import pandas as pd

from prism.assembly import size_orders
from prism.live import ledgers
from prism.live.authorization import (
    LIVE_BASE_URL,
    ActionClass,
    ActionContext,
    AuthorizationGate,
    AuthorizationRecord,
    AuthorizationRefused,
    InputFreshness,
    OperatingState,
    SessionIntent,
    classify_order,
    execution_host,
)
from prism.live.broker import (
    Broker,
    BrokerCapabilityUnsupported,
    DuplicateOrder,
    Fill,
    Order,
    OrderRejected,
    transmits_to_venue,
)
from prism.live.lockfile import (
    DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    LockHandle,
    account_execution_lock,
)
from prism.live.reconcile import ReconcileSettings, Reconciliation, reconcile_loop_state
from prism.live.state import ComponentState, LoopState, StateStore, record_order_status

if TYPE_CHECKING:  # the §3.11 report type, for the signature only
    from prism.live.freshness import FreshnessReport

logger = logging.getLogger(__name__)

# Positions within this many shares of each other reconcile silently;
# anything larger is logged as a real divergence before broker truth wins.
_RECONCILE_SHARE_TOLERANCE = 1e-6

#: The token the submission announcement carries. It is emitted exactly once per
#: submitting pass, immediately before the first order is offered to a venue, and
#: never by a pass that offers none — so grepping a log for it answers "did this
#: run trade, and against what?" without reading the run that produced it. See
#: :func:`announce_submission`.
SUBMISSION_ANNOUNCEMENT = "SUBMITTING ORDERS TO THE VENUE"


@dataclass(frozen=True)
class ExecutionLockSpec:
    """Which account's execution lock this context takes, and where (§3.10).

    ``account_id`` and ``execution_host`` are the authorization record's own
    fields; a context carrying a record does not need this object at all, since
    :func:`execution_lock_spec` derives one from the record. It exists for the
    invocation that has no record — the pre-authorization paper instrument, and
    the tests that pin the lock's placement — and for an operator who has to
    point the lock directory somewhere other than the host default.
    """

    account_id: str
    execution_host: str
    lock_dir: Path | None = None
    max_heartbeat_age_seconds: float = DEFAULT_MAX_HEARTBEAT_AGE_SECONDS

    @classmethod
    def from_record(cls, record: AuthorizationRecord, *, lock_dir: Path | None = None) -> "ExecutionLockSpec":
        """The lock the account owner's record declares (§3.10).

        The record is the single declaration of which account runs from which
        host, so it is also the single declaration of which lock this process
        takes and which host is entitled to take it.
        """
        return cls(
            account_id=record.account_id,
            execution_host=record.execution_host,
            lock_dir=lock_dir,
        )


@dataclass(frozen=True)
class PartialBatch:
    """The portfolio a partly submitted batch implies (§3.7, post-submit).

    An order the venue has accepted cannot be scaled, so the two halves of a
    partly submitted batch are not symmetric: ``accepted`` are committed
    inputs, ``rejected`` is the leg the venue refused with its reason, and
    ``unsubmitted`` are the legs that were never offered because submission
    stopped at the rejection. The object §3.7 revalidates is the portfolio
    ``positions`` **plus** ``accepted`` implies — not the batch as decided, and
    not the remainder on its own.
    """

    positions: Mapping[str, float]
    cash: float
    accepted: tuple[Order, ...]
    rejected: tuple[Order, ...]
    unsubmitted: tuple[Order, ...]
    reasons: Mapping[str, str]


#: The two declared responses to a batch the venue split (§3.7 ``on_partial_batch``).
PARTIAL_BATCH_RESPONSES: tuple[str, ...] = ("cancel_accepted", "hold_and_escalate")


class PartialBatchEscalation(RuntimeError):
    """``hold_and_escalate``: the accepted legs stand and a human is called.

    Raised rather than returned because escalation is the declared response to
    a portfolio that no longer validates: the accepted legs are left at the
    venue deliberately, and the process must not continue as though the batch
    had completed.
    """


@dataclass(frozen=True)
class RemainderCheck:
    """What revalidates the remainder after a rejection, and the declared response.

    ``validate`` takes the :class:`PartialBatch` and returns the violations the
    implied portfolio *introduces* — an empty sequence when it introduces none.
    It is a declared seam rather than a fixed rule because §3.7 measures
    violations against **declared** holding-state limits, and a loop context
    that was given none has nothing to measure. Nothing here invents a bound: a
    context with no check still stops submitting at the rejection, still
    records the rejected order as blocked, and still records the unsubmitted
    remainder, which is exactly §3.7's "the revalidation introduces no
    violation" branch.

    ``on_partial_batch`` is the recovery policy's key of the same name, carried
    by value so the live loop reads no policy file: ``cancel_accepted`` issues
    C0 cancellations against the accepted legs, which every operating state
    permits, and ``hold_and_escalate`` leaves them and raises
    :class:`PartialBatchEscalation`.
    """

    validate: Callable[[PartialBatch], Sequence[str]] | None = None
    on_partial_batch: str = "cancel_accepted"

    def __post_init__(self) -> None:
        if self.on_partial_batch not in PARTIAL_BATCH_RESPONSES:
            raise ValueError(
                f"on_partial_batch is one of {PARTIAL_BATCH_RESPONSES}, got {self.on_partial_batch!r}; "
                "a mis-spelled response that silently kept the default would decide, unrecorded, what "
                "happens to legs the venue already accepted (§3.7)"
            )


@dataclass(frozen=True)
class SubmissionOutcome:
    """What one pass of :func:`submit_pending` actually did at the venue.

    Returned rather than only logged, because the caller — a stage driver, a
    CLI, a test — has to be able to state the fate of every leg: which reached
    the venue, which the venue refused and why, which were never offered, and
    what the revalidation of the remainder concluded.
    """

    submitted: tuple[str, ...] = ()
    accepted: tuple[str, ...] = ()
    rejected: tuple[tuple[str, str], ...] = ()
    unsubmitted: tuple[str, ...] = ()
    introduced: tuple[str, ...] = ()
    revalidated: bool = False
    response: str | None = None
    cancelled: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """The record shape (JSON-safe)."""
        return {
            "submitted": list(self.submitted),
            "accepted": list(self.accepted),
            "rejected": [{"client_order_id": oid, "reason": why} for oid, why in self.rejected],
            "unsubmitted": list(self.unsubmitted),
            "introduced": list(self.introduced),
            "revalidated": self.revalidated,
            "response": self.response,
            "cancelled": list(self.cancelled),
        }


@dataclass
class LiveLoopContext:
    """The durable pieces a loop step operates on.

    The optional ledgers are additive telemetry (SPEC §10: no ratified
    statistic moves): ``targets_ledger`` records the constructed book at each
    refresh (what the instrument *should* hold — the concordance baseline and
    the sweep's audit record), ``unfilled_ledger`` records every unexecuted
    residual at settle (the orders the venue did not print — previously only a
    truncated log line), ``concordance_ledger`` records how faithfully the
    held book tracks the last refresh targets, and ``regime_ledger`` records
    the per-cycle SPEC §7.5 regime telemetry read — the session record the
    handoff §8 precondition-(b) 21-session clock consults
    (docs/regime_step.md). ``order_id_prefix`` namespaces
    client order ids per book and run directory: the live driver passes the
    book prefix followed by a digest of the run directory's name, built by
    ``paper_books.scoped_order_id_prefix`` (``"mom:faf4b45b:"`` for the momentum
    book in ``runs/paper_loop_momentum2``; orders recorded before 2026-09-30
    carry the bare ``"mom:"``):
    two books sharing one venue account with the bare ``{bar}:{symbol}`` scheme
    silently substitute each other's same-bar orders, because a venue duplicate
    id is treated as submission success.

    ``authorization`` is the account's gate (§3.10). ``None`` is the
    pre-authorization paper instrument this stage keeps working, and it is
    permitted only while the broker declares no live endpoint: an unauthorized
    context in front of a live client refuses every write path, before the
    first broker call.

    ``execution_lock`` names the account-keyed execution lock every mutating
    entry point takes before its first state load and releases after its last
    state save, settle included (§3.10). Leaving it ``None`` is the normal
    case: :func:`execution_lock_spec` derives the lock from ``authorization``'s
    own record whenever that record names this host, so an authorized session
    is never unlocked because a driver forgot a line. Set it explicitly only to
    override the derivation — to point the lock directory somewhere other than
    the host default, or to lock a pre-authorization context that carries no
    record at all.

    ``remainder_check`` is §3.7's post-submit revalidation: what judges the
    portfolio left when the venue rejects one order mid-batch, and the declared
    response when that portfolio no longer validates. ``None`` declares no
    holding-state limits to this loop, which is the pre-authorization paper
    instrument's position: submission still stops at the rejection and the
    remainder is still recorded, but there is no bound against which a
    violation could be found, and none is invented.
    """

    store: StateStore
    broker: Broker
    fills_ledger: Path
    equity_ledger: Path | None = None
    targets_ledger: Path | None = None
    unfilled_ledger: Path | None = None
    concordance_ledger: Path | None = None
    regime_ledger: Path | None = None
    #: One row per session the daily cycle could shadow: the decided book's
    #: return on the same closes beside the paper book's (docs/live_monitor_design.md).
    shadow_ledger: Path | None = None
    order_id_prefix: str = ""
    authorization: AuthorizationGate | None = None
    execution_lock: ExecutionLockSpec | None = None
    remainder_check: RemainderCheck | None = None
    #: Re-entrancy for :func:`held_execution_lock`. One invocation takes the
    #: account lock once, whatever nesting of loop entry points it goes through:
    #: ``flock`` is held per open file description, so a second ``open`` of the
    #: same lock from this very process would be refused as if it were a second
    #: writer.
    _lock_depth: int = field(default=0, init=False, repr=False, compare=False)
    _lock_handle: LockHandle | None = field(default=None, init=False, repr=False, compare=False)


def targets_to_orders(
    target_weights: pd.Series,
    positions: dict[str, float],
    prices: pd.Series,
    equity: float,
    decision_bar: str,
    min_order_notional: float = 0.0,
    whole_shares: bool = False,
    order_id_prefix: str = "",
) -> list[Order]:
    """Diff constructed target weights against held shares into orders.

    The sizing itself — the zero-crossing clamp, whole-share rounding and the
    dust rules — is :func:`prism.assembly.size_orders`, so the live protocol
    and the evaluator size from one implementation (engine specification §3.8).
    This name and signature stay because the write-ahead protocol, the safety
    rails and the venue adapter are written against them.
    """
    return size_orders(
        target_weights,
        positions,
        prices,
        equity,
        decision_bar,
        min_order_notional=min_order_notional,
        whole_shares=whole_shares,
        order_id_prefix=order_id_prefix,
    )


def declared_endpoint(broker: Broker) -> str | None:
    """The base URL a broker adapter declares, without calling it.

    An *attribute*, never a method: this is read on the path that must not
    reach the venue, so asking the adapter where it points may not itself be a
    request. A broker that declares nothing (every in-memory fake, the replay
    venue) returns ``None``, which is not an endpoint and cannot be a live one.
    """
    value = getattr(broker, "endpoint", None)
    return str(value).rstrip("/") if isinstance(value, str) else None


def venue_identity(broker: Broker) -> tuple[str, str]:
    """``(endpoint, account)`` for the log, best effort and never fatal.

    The endpoint comes from the adapter's own declared base URL — the question
    is where *this process* pointed, which a venue's self-description cannot
    answer. The account id needs a read, and an adapter that cannot make it
    (every in-memory fake, the replay venue) reports ``unknown`` rather than
    taking the session down: this is a log line, not a rail, and a rail that is
    really a log line is worse than either.
    """
    endpoint = declared_endpoint(broker) or "unknown"
    account = "unknown"
    try:
        identity = broker.account_identity()
    except BrokerCapabilityUnsupported:
        pass  # the adapter cannot name its account; the endpoint still names the venue
    except Exception as exc:  # noqa: BLE001 - a log line must never abort a session
        logger.warning("could not read the account identity for the submission announcement: %s", exc)
    else:
        account = identity.account_id or account
        endpoint = identity.endpoint or endpoint
    return endpoint, account


def announce_submission(broker: Broker, orders: Sequence[Order]) -> None:
    """Say, once and unmistakably, that this run is about to trade.

    Called immediately before the first order of a pass is offered to the
    adapter, and only for an adapter that actually transmits
    (``prism.live.broker.transmits_to_venue``): a rehearsal under ``--no-submit``
    and a replay run reach this point too, and a log line claiming they traded
    would defeat the purpose of having one. The count is the number of orders
    this pass will offer — orders already at the venue are not re-offered and
    are not counted.
    """
    if not orders or not transmits_to_venue(broker):
        return
    endpoint, account = venue_identity(broker)
    bars = sorted({order.decision_bar for order in orders})
    logger.warning(
        "*** %s *** endpoint=%s account=%s decision_bar=%s orders=%d — this run is trading: those "
        "orders are being sent to the account named above, now.",
        SUBMISSION_ANNOUNCEMENT,
        endpoint,
        account,
        bars[0] if len(bars) == 1 else ",".join(bars),
        len(orders),
    )


def _refuse_unauthorized_live(ctx: LiveLoopContext) -> None:
    """Refuse an unauthorized context in front of a live client (S4b)."""
    endpoint = declared_endpoint(ctx.broker)
    if endpoint is not None and endpoint == LIVE_BASE_URL:
        raise AuthorizationRefused(
            f"the broker is bound to the live endpoint {endpoint!r} and this loop context carries no "
            "authorization record: state S4b permits nothing and every broker write is refused, C0 "
            "included. Refusing before any broker call (§3.10)"
        )


def verify_authorized_binding(ctx: LiveLoopContext) -> AuthorizationGate | None:
    """Verify that this context may talk to this venue at all, before it does.

    Identity only — account id, endpoint, ``config_id``, ``engine_version``,
    execution host and build — never scope and never expiry: a persisted batch
    is submitted as a unit and a later expiry does not split it (§3.10).
    """
    gate = ctx.authorization
    if gate is None:
        _refuse_unauthorized_live(ctx)
        return None
    gate.verify_binding(observed_endpoint=declared_endpoint(ctx.broker))
    return gate


def authorize_session(
    ctx: LiveLoopContext,
    intent: SessionIntent,
    *,
    states: Iterable[OperatingState] = (),
    action_context: ActionContext = ActionContext(),
    now: datetime | None = None,
) -> AuthorizationGate | None:
    """The whole pre-flight for an invocation that may write to the venue.

    Binding, then "is anything this intent covers permitted at all", then — for
    a rebalance, which may add exposure — that the grant covers the submission
    window its batch will be completed in. Every one of them raises
    :class:`~prism.live.authorization.AuthorizationRefused` before the caller's
    first broker call, which is what makes the refusal provable: nothing
    reached the venue because nothing was asked of it.
    """
    gate = verify_authorized_binding(ctx)
    if gate is None:
        return None
    gate.verify_intent(intent, states=states, context=action_context, now=now)
    if intent is SessionIntent.REBALANCE:
        gate.verify_submission_window(now=now)
    return gate


def apply_freshness(
    report: "FreshnessReport | None",
    states: Iterable[OperatingState] = (),
    action_context: ActionContext = ActionContext(),
) -> tuple[tuple[OperatingState, ...], ActionContext, dict[str, InputFreshness]]:
    """Fold a §3.11 freshness report into the three things §3.10 evaluates.

    The report is the only place the four checks are run, and this is the only
    place their answer is translated for the gate, in the three shapes the gate
    takes it in:

    * the **account-wide** operating rows the checks establish (S2, S8 — never
      S1, which is per sid, and never S9, which is a monitor row);
    * the two ``ActionContext`` conditions they establish, ``open_orders_read_fresh``
      for S2's "only after a fresh open-order read" and ``risk_inputs_fresh``
      for S1's "C4 only when the risk snapshot's declared inputs are fresh";
    * the **per-sid** verdicts, which is how "a coverage shortfall blocks
      classes per sid, not the whole account" becomes a mechanism rather than a
      sentence: a name the feed did not cover loses C3 while the name beside it
      keeps it.

    ``states`` and ``action_context`` the caller established separately are
    kept: the reconciliation pass establishes S7, the restart recovery S3, and
    the kill file S6, and none of those is a freshness read.
    """
    if report is None:
        return tuple(states), action_context, {}
    return (
        tuple(frozenset(states) | report.states),
        report.action_context(action_context),
        report.freshness_by_sid(),
    )


def _authorize_batch(
    gate: AuthorizationGate,
    orders: list[Order],
    positions: Mapping[str, float],
    intent: SessionIntent,
    *,
    states: Iterable[OperatingState] = (),
    action_context: ActionContext = ActionContext(),
    freshness_by_sid: Mapping[str, InputFreshness] | None = None,
    now: datetime | None = None,
) -> None:
    """Classify a sized batch into action classes and authorize it as a unit.

    ``freshness_by_sid`` is the per-security half of §3.10's freshness rule and
    is passed through unchanged: the batch is refused as a unit, and the
    refusal names the securities whose own inputs were not fresh enough for the
    class their order falls in.
    """
    if not orders:
        return
    actions: dict[str, ActionClass] = {
        order.symbol: classify_order(order.symbol, order.qty, positions, hedge_sids=gate.hedge_sids)
        for order in orders
    }
    outside = sorted(sid for sid, action in actions.items() if action not in intent.classes)
    if outside:
        raise AuthorizationRefused(
            f"a {intent.value} session produced order(s) outside its declared classes "
            f"{sorted(a.value for a in intent.classes)}: "
            f"{[(sid, actions[sid].value) for sid in outside[:8]]}; refusing before the write-ahead "
            "persist (§3.10)"
        )
    gate.authorize_batch(
        actions,
        states=states,
        context=action_context,
        freshness_by_sid=freshness_by_sid,
        now=now,
    )


# ---------------------------------------------------------------------------
# the account-keyed execution lock (§3.10)
# ---------------------------------------------------------------------------


def execution_lock_spec(ctx: LiveLoopContext) -> ExecutionLockSpec | None:
    """Which account lock this context takes, or ``None`` when there is none.

    An explicit ``ctx.execution_lock`` wins — that is the override, for an
    operator who must point the lock directory somewhere other than the host
    default and for a pre-authorization context that carries no record.

    Otherwise the lock is derived from the authorization record, because the
    record is already the single declaration of which account runs from which
    host (§3.10), and a control that only arms when a driver remembers to arm
    it protects nothing. The derivation is conditioned on the record naming
    *this* host, and that condition is what keeps it honest rather than
    convenient: :func:`~prism.live.lockfile.account_execution_lock` refuses
    outright from an undeclared host, so deriving unconditionally would replace
    the binding check's refusal — which happens before any broker call and says
    ``execution_host=`` — with a lock refusal underneath it. On a host the
    record does not name, the session is refused either way; the difference is
    only which control says so, and the binding check is the one that names the
    cause. Where the record *does* name this host, the derived lock can refuse
    for exactly one reason, which is the reason it exists: another writer on
    this host already holds this account.
    """
    if ctx.execution_lock is not None:
        return ctx.execution_lock
    gate = ctx.authorization
    record = None if gate is None else gate.record
    if record is None or record.execution_host != execution_host():
        return None
    return ExecutionLockSpec.from_record(record)


@contextlib.contextmanager
def held_execution_lock(ctx: LiveLoopContext) -> Generator[LockHandle | None, None, None]:
    """Hold this account's execution lock for the duration of one invocation.

    Placement is the whole point (§3.10): acquired **before the first state
    load** of any invocation that may write to the broker or to state, released
    **after that invocation's last state save**. Settle writes state and
    ledgers, so settle is inside the lock, not outside it.

    Re-entrant within one process, because ``flock`` is held per open file
    description: a second acquisition from this very process would be refused
    as though it were a second writer. So a driver may wrap a whole cycle in
    this and the entry points inside it nest rather than deadlock.

    Yields ``None`` for a context that declares no account — the
    pre-authorization paper instrument, whose exclusion is the run-directory
    lock its CLI takes. It never yields quietly on a *declared* account whose
    lock could not be taken: :func:`~prism.live.lockfile.account_execution_lock`
    raises there, and this is not the layer that downgrades it.
    """
    spec = execution_lock_spec(ctx)
    if spec is None:
        yield None
        return
    if ctx._lock_depth > 0:
        ctx._lock_depth += 1
        try:
            yield ctx._lock_handle
        finally:
            ctx._lock_depth -= 1
        return
    with account_execution_lock(
        spec.account_id,
        spec.execution_host,
        lock_dir=spec.lock_dir,
        max_heartbeat_age_seconds=spec.max_heartbeat_age_seconds,
    ) as handle:
        ctx._lock_depth = 1
        ctx._lock_handle = handle
        try:
            yield handle
        finally:
            ctx._lock_depth = 0
            ctx._lock_handle = None


# ---------------------------------------------------------------------------
# restart recovery, and the ambiguous submission (§3.11 item 4, state S3)
# ---------------------------------------------------------------------------

#: An execution printed against the order: it reached the venue and traded.
FATE_EXECUTED = "executed"
#: The venue reports the order live: it reached the venue and is working.
FATE_WORKING = "working"
#: The venue knows the id and does not report it live: it reached the venue and
#: is done (filled, expired, cancelled or rejected — settle books whichever).
FATE_TERMINAL = "terminal"
#: The venue's own registry does not know the id: it never reached the venue.
#: Re-offering it is safe, because the client order id is the idempotency key.
FATE_UNSUBMITTED = "unsubmitted"
#: The venue could not be asked. Not "no orders" — "no answer" (state S3).
FATE_UNKNOWN = "unknown"

#: Every fate except :data:`FATE_UNKNOWN`. A known fate is one the venue stated.
KNOWN_FATES: frozenset[str] = frozenset({FATE_EXECUTED, FATE_WORKING, FATE_TERMINAL, FATE_UNSUBMITTED})


class UnresolvedSubmission(RuntimeError):
    """An order's fate is unknown, so this account may only cancel and reconcile.

    Operating state S3 (§3.10): C0 and reconcile, with obligation progress
    escalating through the recovery policy rather than writing to the venue.
    Raised in place of a submission and never caught and downgraded — the whole
    point is that the engine does not retry into an outcome it cannot see.
    :func:`resolve_unresolved_submission` is how the state is left.
    """

    def __init__(self, message: str, recovery: "RestartRecovery | None" = None) -> None:
        super().__init__(message)
        self.recovery = recovery


@dataclass(frozen=True)
class OrderFate:
    """What the venue says about one persisted order, and how it was asked."""

    client_order_id: str
    symbol: str
    qty: float
    fate: str
    detail: str

    @property
    def known(self) -> bool:
        """Whether the venue stated this order's fate."""
        return self.fate in KNOWN_FATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "qty": self.qty,
            "fate": self.fate,
            "known": self.known,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RestartRecovery:
    """What a restarted process established before it acted (§3.11).

    ``fates`` covers the orders the persisted state still carries as open.
    ``reconciliation`` is the §3.11 accounting pass when one was run, and
    ``None`` when the caller asked for the fate read alone.
    """

    decision_bar: str | None = None
    fates: tuple[OrderFate, ...] = ()
    reconciliation: Reconciliation | None = None

    @property
    def unknown_order_ids(self) -> tuple[str, ...]:
        """The orders whose fate the venue would not state, in id order."""
        return tuple(sorted(fate.client_order_id for fate in self.fates if not fate.known))

    @property
    def resolved(self) -> bool:
        """Whether every persisted order's fate is known. S3 holds until it is."""
        return not self.unknown_order_ids

    @property
    def may_submit(self) -> bool:
        """Whether this account may write a new order at all.

        S3 removes every class but C0, and a blocking accounting item (S7's
        non-adoptable outcome: an open order differing in quantity or side, or
        an execution the ledger does not carry) does the same. Positions and
        cash adopted from broker truth are not blocking — refusing to rebalance
        over a routine partial fill is how a book drifts for a month (§3.11).
        """
        if not self.resolved:
            return False
        return self.reconciliation is None or self.reconciliation.may_trade

    @property
    def operating_states(self) -> frozenset[OperatingState]:
        """The §3.10 rows this recovery puts the account in."""
        states: set[OperatingState] = set()
        if not self.resolved:
            states.add(OperatingState.S3_UNRESOLVED_SUBMISSION)
        report = self.reconciliation
        if report is not None and report.blocking_items:
            states.add(OperatingState.S7_ACCOUNTING_DISCREPANCY)
        return frozenset(states)

    def refusal(self) -> str:
        """The message an unresolved recovery refuses a submission with."""
        shown = list(self.unknown_order_ids[:8])
        tail = "…" if len(self.unknown_order_ids) > len(shown) else ""
        return (
            f"the fate of {len(self.unknown_order_ids)} persisted order(s) from decision bar "
            f"{self.decision_bar!r} is unknown: {shown}{tail}. The venue reports them neither live nor "
            "terminal and no execution has printed against them, so this account is in state S3: "
            "cancellation and reconciliation only, and no retry until the fate is known (§3.10, §3.11). "
            "resolve_unresolved_submission() leaves it by cancelling them and recording the outcome."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_bar": self.decision_bar,
            "resolved": self.resolved,
            "may_submit": self.may_submit,
            "states": sorted(state.value for state in self.operating_states),
            "unknown_order_ids": list(self.unknown_order_ids),
            "fates": [fate.to_dict() for fate in self.fates],
            "reconciliation": None if self.reconciliation is None else self.reconciliation.to_dict(),
        }


def venue_live_order_ids(broker: Broker, client_order_ids: Iterable[str]) -> set[str] | None:
    """Which of these ids the venue reports LIVE, or ``None`` if it cannot say.

    This is a *membership* question — is this order still working — so the
    membership read answers it: ``open_order_ids()``, the duck-typed one the
    completion sweep already relies on. ``open_orders()`` is the richer §3.11
    read, carrying the quantity and side that reconciliation compares and that
    a recorded cancel needs, and it is the fallback here rather than the first
    choice, because asking for a whole order book to answer a set-membership
    question is a cost paid on every resume.

    A broker offering neither returns ``None``, which is *not* an empty set:
    "the venue holds none of these" and "this adapter cannot list open orders"
    are different facts, and only one of them makes it safe to act.
    """
    wanted = {str(client_order_id) for client_order_id in client_order_ids}
    membership = getattr(broker, "open_order_ids", None)
    if membership is not None:
        try:
            return {str(client_order_id) for client_order_id in membership(set(wanted))} & wanted
        except BrokerCapabilityUnsupported:
            pass
    try:
        rows = broker.open_orders()
    except BrokerCapabilityUnsupported:
        return None
    return {str(getattr(row, "client_order_id", "")) for row in rows} & wanted


def venue_known_order_ids(broker: Broker, orders: Sequence[Order]) -> set[str] | None:
    """Every id the venue's own registry knows, or ``None`` if it cannot be read.

    This is the read the idempotent submission already makes
    (:func:`_known_order_ids`), scoped to the pending set's bars where the
    adapter supports it.
    """
    if not orders:
        return set()
    try:
        return {str(client_order_id) for client_order_id in _known_order_ids(broker, list(orders))}
    except BrokerCapabilityUnsupported:
        return None


def read_order_fates(
    broker: Broker,
    orders: Sequence[Order],
    *,
    live_ids: set[str] | None = None,
    live_read_available: bool | None = None,
) -> tuple[OrderFate, ...]:
    """Ask the venue what happened to each persisted order (§3.11 item 4).

    Two reads answer it for the whole set: the registry of ids the venue knows,
    and the set it reports live. With both, every order has a fate — working,
    terminal, or never submitted — and none is ambiguous, because the client
    order id is the idempotency key and an id the venue does not know can be
    re-offered safely.

    With either read missing, the executions are pulled as a last resort: an
    order that printed a fill has a fate whatever else is unknown, and the rest
    are ``unknown``. That is state S3, and it is deliberately *not* the same as
    "nothing came back": an adapter that cannot list open orders has told us
    nothing, and acting on nothing is the failure this path exists to prevent.

    ``live_ids`` lets a caller that has already read the venue's open orders —
    the completion sweep does — pass them in rather than paying for the read
    twice; ``live_read_available=False`` declares that read impossible.
    """
    orders = list(orders)
    if not orders:
        return ()
    ids = [order.client_order_id for order in orders]
    if live_read_available is None:
        live = live_ids if live_ids is not None else venue_live_order_ids(broker, ids)
    else:
        live = set(live_ids or ()) if live_read_available else None
    known = venue_known_order_ids(broker, orders)

    executed: set[str] = set()
    if live is None or known is None:
        # The last resort, and only then: on a live venue this is one request
        # per id, which is why it is not paid for on the path that has answers.
        executed = {fill.client_order_id for fill in broker.fills_for(set(ids))}

    fates: list[OrderFate] = []
    for order in orders:
        client_order_id = order.client_order_id
        if client_order_id in executed:
            fate, detail = (
                FATE_EXECUTED,
                "an execution has printed against it, so it reached the venue and traded; settle books it",
            )
        elif known is None:
            fate, detail = (
                FATE_UNKNOWN,
                "the venue's order registry could not be read, so whether this order reached it is unknown",
            )
        elif client_order_id not in known:
            fate, detail = (
                FATE_UNSUBMITTED,
                "the venue's registry does not know this client order id, so it never reached the venue; "
                "re-offering it is safe because the id is the idempotency key",
            )
        elif live is None:
            fate, detail = (
                FATE_UNKNOWN,
                "the venue knows this client order id but cannot list open orders, so whether it is still "
                "working or already done is unknown",
            )
        elif client_order_id in live:
            fate, detail = (FATE_WORKING, "the venue reports it live")
        else:
            fate, detail = (
                FATE_TERMINAL,
                "the venue knows the id and does not report it live, so it is done; settle books whatever "
                "it executed",
            )
        fates.append(
            OrderFate(
                client_order_id=client_order_id,
                symbol=order.symbol,
                qty=float(order.qty),
                fate=fate,
                detail=detail,
            )
        )
    return tuple(fates)


def pending_order_recovery(
    ctx: LiveLoopContext,
    state: LoopState,
    *,
    live_ids: set[str] | None = None,
    live_read_available: bool | None = None,
) -> RestartRecovery:
    """The fate of every order this state still carries as open.

    The cheap half of :func:`recover_on_restart`: no reconciliation pass and no
    lock of its own (the caller holds it), so the submission paths can consult
    it on every resume without a second venue sweep.
    """
    fates = read_order_fates(
        ctx.broker,
        state.open_pending_orders,
        live_ids=live_ids,
        live_read_available=live_read_available,
    )
    return RestartRecovery(decision_bar=state.pending_decision_bar, fates=fates)


def recover_on_restart(
    ctx: LiveLoopContext,
    *,
    reconcile: bool = True,
    ledger_execution_ids: Iterable[str] = (),
    reconcile_settings: ReconcileSettings = ReconcileSettings(),
) -> RestartRecovery:
    """Everything a restarted process must establish before it acts (§3.11).

    In order: the authorization binding (a mis-bound process makes no broker
    call at all), the account execution lock, the durable state at schema
    version 3 — a version-2 document is migrated forward by the store, never
    read as a fresh loop — the fate of every order the state still carries as
    open, and, unless the caller declines it, the §3.11 accounting
    reconciliation.

    Returns the recovery rather than acting on it. Two callers act on it: the
    submission paths, which refuse while it is unresolved, and
    :func:`resolve_unresolved_submission`, which leaves S3 by cancelling what
    the venue would not account for.
    """
    verify_authorized_binding(ctx)
    with held_execution_lock(ctx):
        state = ctx.store.load()
        if state is None:
            logger.info("restart recovery: no durable state — this is a genuinely fresh loop")
            return RestartRecovery()
        recovery = pending_order_recovery(ctx, state)
        report: Reconciliation | None = None
        if reconcile:
            report = reconcile_loop_state(
                state,
                ctx.broker,
                ledger_execution_ids=ledger_execution_ids,
                settings=reconcile_settings,
            )
            # The pass adopts broker truth for positions and cash on the
            # in-memory state; persist it, so the next process reads the book
            # the venue reports rather than re-deriving the adoption. The fills
            # marker is deliberately not advanced here (append-then-advance).
            ctx.store.save(state)
        recovery = RestartRecovery(
            decision_bar=recovery.decision_bar,
            fates=recovery.fates,
            reconciliation=report,
        )
        if not recovery.resolved:
            logger.error("restart recovery: %s", recovery.refusal())
        elif recovery.fates:
            logger.info(
                "restart recovery: %d persisted order(s) from %s, every fate known (%s)",
                len(recovery.fates),
                recovery.decision_bar,
                ", ".join(sorted({fate.fate for fate in recovery.fates})),
            )
        return recovery


def resolve_unresolved_submission(
    ctx: LiveLoopContext,
    recovery: RestartRecovery | None = None,
    *,
    states: Iterable[OperatingState] = (),
    action_context: ActionContext = ActionContext(),
    now: datetime | None = None,
) -> RestartRecovery:
    """Leave S3 the only way it may be left: cancel what the venue will not account for.

    S3 permits C0 and reconciliation, so the resolution is a cancel. Cancelling
    an id the venue never received is a no-op, and cancelling one it did receive
    establishes the post-condition that matters — *this order is not working* —
    whichever of the two was true. The outcome is recorded against the order
    (:func:`~prism.live.state.record_order_status`) and saved, so a further
    restart reads the resolved fate instead of asking again.

    Accounting is untouched by this. A cancelled order that had already executed
    still settles: ``settle`` pulls fills for every persisted id whatever status
    is recorded here, and the fills ledger drops duplicates by key.
    Exactly-once delivery is not promised; the accounting is idempotent (§3.11).
    """
    with held_execution_lock(ctx):
        state = ctx.store.load()
        if state is None:
            return RestartRecovery()
        if recovery is None or recovery.decision_bar != state.pending_decision_bar:
            recovery = pending_order_recovery(ctx, state)
        if recovery.resolved:
            return recovery
        unknown = list(recovery.unknown_order_ids)
        logger.warning(
            "S3 resolution: cancelling %d order(s) of unknown fate from decision bar %s (%s)",
            len(unknown),
            recovery.decision_bar,
            unknown[:8],
        )
        cancel_orders(
            ctx,
            unknown,
            states=frozenset(states) | {OperatingState.S3_UNRESOLVED_SUBMISSION},
            action_context=action_context,
            now=now,
        )
        changed = record_order_status(
            state,
            unknown,
            "canceled",
            reason="fate unknown at restart; cancelled to resolve state S3 (§3.10, §3.11)",
            last_update=_utc_stamp(now),
        )
        ctx.store.save(state)
        logger.info("S3 resolution: recorded %d cancellation(s) durably: %s", len(changed), list(changed[:8]))
        resolved = pending_order_recovery(ctx, state)
        if not resolved.resolved:
            logger.error("S3 resolution did not clear: %s", resolved.refusal())
        return resolved


def _utc_stamp(now: datetime | None) -> str:
    """An ISO timestamp for a durable record; the caller's clock when it has one."""
    moment = now if now is not None else datetime.now(timezone.utc)
    return moment.isoformat()


def decide_and_submit(
    ctx: LiveLoopContext,
    decision_bar: str,
    target_weights: pd.Series,
    prices: pd.Series,
    min_order_notional: float = 0.0,
    whole_shares: bool = False,
    refresh_bar: str | None = None,
    order_guard: Callable[[list[Order]], None] | None = None,
    intent: SessionIntent = SessionIntent.REBALANCE,
    states: Iterable[OperatingState] = (),
    action_context: ActionContext = ActionContext(),
    freshness: "FreshnessReport | None" = None,
    now: datetime | None = None,
    panel: Mapping[str, Any] | None = None,
    components: Sequence[ComponentState] | None = None,
) -> list[Order]:
    """One decision step: reconcile → decide once → write ahead → submit.

    Re-running for the same ``decision_bar`` after a crash resumes the
    persisted decision idempotently. Calling for a new bar while a prior
    bar's orders are still pending raises — settle first, the book's true
    holdings are unknown until then. ``refresh_bar``, when provided, advances
    the persisted decision-cadence anchor (``last_refresh_bar``) inside the same
    write-ahead save; the caller passes it only on a refresh session.
    ``order_guard`` is the safety seam (``prism.live.safety``): it sees the
    freshly diffed order list *before* the write-ahead persist, so a raised
    :class:`~prism.live.safety.SafetyViolation` leaves no pending state and
    nothing at the venue. A resumed (already-persisted) decision is not
    re-guarded — it was vetted when decided, and re-deciding is the defect
    the protocol exists to prevent.

    Authorization runs first, above the state load and above the reconcile read
    (§3.10): a context whose record does not bind this process, or whose grant
    has expired, raises
    :class:`~prism.live.authorization.AuthorizationRefused` having made no
    broker call at all. ``intent`` declares what this session may emit —
    ``MANAGE_EXISTING`` is the cancellation-and-reduction path, which stays
    available under a stale signal (S1) and a narrowed or expired grant (S4a).
    ``states`` and ``action_context`` are what the reconciliation and recovery
    reads establish; an empty ``states`` is S0 and asserts nothing. ``freshness``
    is the §3.11 report, and passing it is what makes the per-security half of
    §3.10 real on this path: its account-wide rows and conditions join ``states``
    and ``action_context``, and its per-sid verdicts reach
    :meth:`~prism.live.authorization.AuthorizationGate.authorize_batch`, so a
    coverage shortfall blocks new exposure on the securities the feed missed and
    leaves the rest of the batch to be judged on its own inputs. ``panel`` is
    the provenance of the bar panel the decision was made on (the digest and
    file name ``prism.live.panel_log`` recorded); its fields are added to the
    refresh row of the targets ledger so the decision cites its inputs.
    ``components``, when provided, replaces the persisted per-component states
    (``LoopState.components``) inside the same write-ahead save, so a tranched
    book's tranche schedule and sub-targets advance atomically with the decision
    they produced (:mod:`prism.live.tranches`). ``None`` leaves them as loaded;
    a resumed decision never rewrites them.
    """
    states, action_context, freshness_by_sid = apply_freshness(freshness, states, action_context)
    gate = authorize_session(ctx, intent, states=states, action_context=action_context, now=now)
    with held_execution_lock(ctx):

        state = ctx.store.load() or LoopState()

        if state.pending_orders:
            if state.pending_decision_bar != decision_bar:
                raise RuntimeError(
                    f"pending orders from {state.pending_decision_bar} are unsettled; "
                    f"settle before deciding {decision_bar} (N2 ordering)"
                )
            recovery = pending_order_recovery(ctx, state)
            if not recovery.resolved:
                # State S3. A retry into an outcome the venue will not state is
                # how a book trades a quantity nobody decided; the batch waits
                # for a cancellation and a reconciliation, not for a retry.
                raise UnresolvedSubmission(recovery.refusal(), recovery)
            logger.warning(
                "resuming persisted decision for %s (%d orders, %d still open) — not re-deciding",
                decision_bar,
                len(state.pending_orders),
                len(state.open_pending_orders),
            )
        else:
            if state.last_settled_bar is not None and decision_bar <= state.last_settled_bar:
                raise RuntimeError(
                    f"decision bar {decision_bar} is not after last settled bar "
                    f"{state.last_settled_bar}; refusing to re-decide a processed bar"
                )
            _reconcile(state, ctx.broker)
            equity = state.cash + sum(
                shares * require_price(prices, symbol) for symbol, shares in state.positions.items()
            )
            orders = targets_to_orders(
                target_weights,
                state.positions,
                prices,
                equity,
                decision_bar,
                min_order_notional=min_order_notional,
                whole_shares=whole_shares,
                order_id_prefix=ctx.order_id_prefix,
            )
            if gate is not None:
                # The batch is authorized as a unit, at write-ahead time, before the
                # persist and before any submission (§3.10).
                _authorize_batch(
                    gate,
                    orders,
                    state.positions,
                    intent,
                    states=states,
                    action_context=action_context,
                    freshness_by_sid=freshness_by_sid,
                    now=now,
                )
            if order_guard is not None:
                order_guard(orders)  # SafetyViolation propagates BEFORE the write-ahead save
            state.pending_orders = orders
            state.pending_decision_bar = decision_bar
            if refresh_bar is not None:
                state.last_refresh_bar = refresh_bar  # advance the cadence anchor atomically
            if components is not None:
                state.components = list(components)  # the tranche states, in the same save
            ctx.store.save(state)  # the write-ahead: persisted before any submit
            if refresh_bar is not None and ctx.targets_ledger is not None:
                # Persist the constructed book at each refresh — the concordance
                # baseline (what the instrument SHOULD hold until the next refresh)
                # and the audit record behind the completion sweep. Written inside
                # the write-ahead step, idempotent per refresh bar.
                append_targets_ledger(
                    ctx.targets_ledger, refresh_bar, decision_bar, target_weights, prices, equity, extra=panel
                )

        # Orders whose fate has been settled — a cancellation recorded while
        # leaving S3 — are not re-offered; every order a v2 or v3 document
        # carries is open until something records otherwise, so this is the
        # persisted set unchanged on every ordinary pass.
        submit_pending(ctx, state, state.open_pending_orders, gate=gate, now=now)
        return list(state.pending_orders)


def submit_pending(
    ctx: LiveLoopContext,
    state: LoopState,
    orders: list[Order],
    *,
    gate: AuthorizationGate | None = None,
    now: datetime | None = None,
) -> SubmissionOutcome:
    """Idempotent submission of a persisted order set (decide and sweep share it).

    The gate is re-verified for *binding* here and is nowhere re-scoped: a
    resumed decision was authorized as a unit when it was decided, and
    re-evaluating scope at submission could drop one side of a balanced batch
    (§3.7, §3.10).

    **A rejection stops the batch** (§3.7, Correction at stage 5). Until this
    change the venue's rejection of one order was logged and the rest were
    submitted anyway, leaving the rejected order in the persisted pending set
    for the sweep to retry: the surviving orders were then a set nothing had
    validated, and one side of a balanced batch could stand alone. The
    corrected protocol has two phases, because an order the venue has accepted
    cannot be scaled:

    * pre-submit — whole-batch validation and authorization happen before the
      write-ahead persist, which is :func:`decide_and_submit`'s job;
    * post-submit — on the rejection of order *k*, submission of the remainder
      **stops**, the rejected order is recorded as blocked with the venue's
      reason and is not re-offered by a resume or by the sweep, and the
      portfolio implied by the held state plus the accepted orders is
      revalidated by ``ctx.remainder_check`` before anything further is
      offered. Introducing no violation ends the batch with the remainder
      recorded; introducing one takes the declared ``on_partial_batch``
      response.

    A transport failure or 5xx is still not caught here — it propagates, so the
    write-ahead crash-resume applies (N7 loud, never a silent drop).
    """
    if not orders:
        return SubmissionOutcome()
    if gate is not None:
        gate.verify_binding()
    known = _known_order_ids(ctx.broker, orders)
    to_offer = [order for order in orders if order.client_order_id not in known]
    announced = False
    accepted: list[Order] = []
    submitted: list[str] = []
    rejected: Order | None = None
    reason = ""
    unsubmitted: list[Order] = []
    for order in orders:
        if rejected is not None:
            # Submission stopped at the rejection: every later leg is offered
            # only by a new write-ahead generation, never by this pass.
            unsubmitted.append(order)
            continue
        if order.client_order_id in known:
            accepted.append(order)  # already at the venue: a committed input
            continue
        if not announced:
            # Immediately before the first order of this pass leaves the process,
            # and nowhere else: a pass whose orders are all already at the venue
            # offers nothing and says nothing.
            announce_submission(ctx.broker, to_offer)
            announced = True
        try:
            ctx.broker.submit(order)
        except DuplicateOrder:
            logger.info("order %s already at venue; treating as submitted", order.client_order_id)
            accepted.append(order)
        except OrderRejected as exc:
            rejected = order
            reason = str(exc)
        else:
            accepted.append(order)
            submitted.append(order.client_order_id)
    if rejected is None:
        return SubmissionOutcome(
            submitted=tuple(submitted),
            accepted=tuple(order.client_order_id for order in accepted),
        )
    return _stop_on_rejection(
        ctx,
        state,
        accepted=accepted,
        rejected=rejected,
        reason=reason,
        unsubmitted=unsubmitted,
        submitted=submitted,
        now=now,
    )


def _stop_on_rejection(
    ctx: LiveLoopContext,
    state: LoopState,
    *,
    accepted: list[Order],
    rejected: Order,
    reason: str,
    unsubmitted: list[Order],
    submitted: list[str],
    now: datetime | None = None,
) -> SubmissionOutcome:
    """The post-submit phase of §3.7: record, revalidate, then respond.

    Order matters and is the whole point. The rejection is written onto the
    durable record and saved *first*, because a crash between here and the
    revalidation must not leave an order that the venue refused looking open —
    a resumed submission reads :attr:`LoopState.open_pending_orders` and the
    sweep skips what is not open, so recording the fate is what stops the
    retry. Only then is the implied portfolio judged, and only then is a
    response taken.
    """
    logger.error(
        "order %s rejected by the venue: %s; submission stops here with %d order(s) unoffered — the "
        "portfolio the accepted legs imply is revalidated before anything further is submitted (§3.7)",
        rejected.client_order_id,
        reason,
        len(unsubmitted),
    )
    record_order_status(
        state,
        [rejected.client_order_id],
        "rejected",
        reason=f"venue rejection at submission: {reason}",
        last_update=_utc_stamp(now),
    )
    ctx.store.save(state)

    outcome = SubmissionOutcome(
        submitted=tuple(submitted),
        accepted=tuple(order.client_order_id for order in accepted),
        rejected=((rejected.client_order_id, reason),),
        unsubmitted=tuple(order.client_order_id for order in unsubmitted),
    )
    check = ctx.remainder_check
    if check is None or check.validate is None:
        logger.error(
            "no remainder check is declared to this loop context, so the portfolio the accepted legs "
            "imply is judged against no holding-state limit; the batch ends here and the %d unoffered "
            "order(s) are recorded: %s",
            len(unsubmitted),
            outcome.unsubmitted[:8],
        )
        return outcome

    partial = PartialBatch(
        positions=dict(state.positions),
        cash=float(state.cash),
        accepted=tuple(accepted),
        rejected=(rejected,),
        unsubmitted=tuple(unsubmitted),
        reasons={rejected.client_order_id: reason},
    )
    introduced = tuple(str(violation) for violation in check.validate(partial))
    outcome = replace(outcome, introduced=introduced, revalidated=True)
    if not introduced:
        logger.warning(
            "the portfolio the %d accepted leg(s) imply introduces no violation; the batch ends here "
            "and the %d unoffered order(s) are recorded (§3.7)",
            len(accepted),
            len(unsubmitted),
        )
        return outcome

    logger.error(
        "the portfolio the %d accepted leg(s) imply introduces %d violation(s) %s; taking the declared "
        "on_partial_batch response %r (§3.7)",
        len(accepted),
        len(introduced),
        list(introduced[:8]),
        check.on_partial_batch,
    )
    if check.on_partial_batch == "hold_and_escalate":
        raise PartialBatchEscalation(
            f"order {rejected.client_order_id} was rejected ({reason}); the portfolio the "
            f"{len(accepted)} accepted leg(s) imply introduces {list(introduced[:8])}, and "
            "on_partial_batch = 'hold_and_escalate' leaves them at the venue for a human (§3.7)"
        )
    # cancel_accepted: C0 against the accepted legs, which every operating
    # state permits. Cancelling one that has already filled is a no-op at the
    # venue and settles identically here — the accounting is idempotent (§3.11).
    ids = [order.client_order_id for order in accepted]
    cancelled = cancel_orders(ctx, ids, now=now) if ids else []
    if cancelled:
        record_order_status(
            state,
            cancelled,
            "canceled",
            reason=f"cancel_accepted after the venue rejected {rejected.client_order_id} (§3.7)",
            last_update=_utc_stamp(now),
        )
        ctx.store.save(state)
    return replace(outcome, response=check.on_partial_batch, cancelled=tuple(cancelled))


def _known_order_ids(broker: Broker, orders: list[Order]) -> set[str]:
    """Ids the venue already knows, scoped to the pending set's bars when possible.

    A pending set is always bounded to one decision bar, but the bare
    contract call pages the account's *entire* order history — latency and
    rate-limit exposure that grow with account age. An adapter that accepts
    ``since`` (duck-typed, the ``open_order_ids`` precedent) is asked only
    from the earliest pending decision bar; ids for bar *t* are submitted
    the evening of *t* at the earliest, so midnight-of-*t* strictly precedes
    every submission the pending set could have made.
    """
    submitted: Callable[..., set[str]] = broker.submitted_order_ids
    try:
        supports_since = "since" in inspect.signature(submitted).parameters
    except (TypeError, ValueError):  # builtins/partials without introspectable signatures
        supports_since = False
    if supports_since:
        return submitted(since=min(order.decision_bar for order in orders))
    return submitted()


def settle(ctx: LiveLoopContext, settle_bar: str) -> list[Fill]:
    """Pull fills for the pending decision, ledger them, re-anchor state.

    Ledgers every executed quantity (whole fills *and* partials under an
    expired/canceled parent) beside its decision-close reference price, then
    re-anchors positions and cash to **broker truth** and clears pending.

    An order that did not fill is *not* an emergency to halt on: a
    market-on-open order whose auction did not print, or whose remainder
    expired, is a routine venue outcome for the R2 paper instrument, and the
    book is still known exactly because we adopt broker truth. So settle does
    not raise on unfilled orders — it fails **loud, not silent** (N7): the
    unfilled ids are logged (WARNING, or ERROR when *nothing* filled, which
    can flag a systemic problem — auth, market closed, venue down), while the
    loop stays alive to decide the next bar. The book cannot silently drift
    because ``positions``/``cash`` come from the broker, not from assuming the
    decision executed. (The prior all-or-nothing raise made a single expired
    OPG leg wedge the whole loop until a manual state edit — the exact stall
    this instrument is meant to survive.)
    """
    with held_execution_lock(ctx):
        state = ctx.store.load()
        if state is None or not state.pending_orders:
            return []

        pending_ids = {o.client_order_id for o in state.pending_orders}
        fills = ctx.broker.fills_for(pending_ids)
        filled_ids = {f.client_order_id for f in fills}
        unfilled = pending_ids - filled_ids
        if unfilled:
            detail = sorted(unfilled)
            shown = detail[:8]
            tail = "…" if len(detail) > len(shown) else ""
            level = logging.ERROR if not fills else logging.WARNING
            logger.log(
                level,
                "settle %s: %d/%d pending orders did not fill (%s%s); ledgering the %d that did "
                "and re-anchoring the book to broker truth (N7: loud, not silent-zero)",
                state.pending_decision_bar,
                len(unfilled),
                len(pending_ids),
                shown,
                tail,
                len(fills),
            )

        # Record every unexecuted residual durably (unfilled and partially filled
        # orders alike). The truncated WARNING above is for the operator's eye; the
        # ledger is for calibration/repair — the 79-name unfilled set of 2026-07-08
        # was unrecoverable locally because only the first 8 ids were ever logged.
        if ctx.unfilled_ledger is not None:
            executed = {f.client_order_id: f.qty for f in fills}
            # Crash-idempotent: a resume between this append and the state save
            # below must not duplicate rows (the same discipline the monotone
            # per-session ledgers already carry).
            already_unfilled = {
                (row.get("client_order_id"), row.get("settle_bar")) for row in ledgers.read_rows(ctx.unfilled_ledger)
            }
            residual_rows = []
            for order in state.pending_orders:
                residual = order.qty - executed.get(order.client_order_id, 0.0)
                if residual == 0.0 or (order.client_order_id, settle_bar) in already_unfilled:
                    continue
                residual_rows.append(
                    {
                        "client_order_id": order.client_order_id,
                        "symbol": order.symbol,
                        "order_qty": order.qty,
                        "executed_qty": executed.get(order.client_order_id, 0.0),
                        "residual_qty": residual,
                        "reference_price": order.reference_price,
                        "decision_bar": state.pending_decision_bar,
                        "settle_bar": settle_bar,
                    }
                )
            if residual_rows:
                ledgers.append_rows(ctx.unfilled_ledger, residual_rows)

        reference = {o.client_order_id: o.reference_price for o in state.pending_orders}
        if fills:
            # Same crash-idempotency for the I-9 record: a settle resumed after a
            # crash between this append and the state save re-pulls the same
            # fills; ledgering them twice would corrupt the calibration stream.
            already_ledgered = {
                row.get("client_order_id")
                for row in ledgers.read_rows(ctx.fills_ledger)
                if row.get("decision_bar") == state.pending_decision_bar
            }
            fill_rows = [
                {
                    "client_order_id": f.client_order_id,
                    "symbol": f.symbol,
                    "qty": f.qty,
                    "fill_price": f.price,
                    "reference_price": reference[f.client_order_id],
                    "decision_bar": state.pending_decision_bar,
                    "filled_bar": f.filled_bar,
                }
                for f in fills
                if f.client_order_id not in already_ledgered
            ]
            if len(fill_rows) < len(fills):
                logger.warning(
                    "settle %s: %d fill row(s) already ledgered (crash-resume) — appending only "
                    "the %d new one(s), never duplicating the I-9 record",
                    state.pending_decision_bar,
                    len(fills) - len(fill_rows),
                    len(fill_rows),
                )
            if fill_rows:
                ledgers.append_rows(ctx.fills_ledger, fill_rows)

        state.positions = {s: q for s, q in ctx.broker.positions().items() if q != 0.0}
        state.cash = ctx.broker.cash()
        state.last_settled_bar = settle_bar
        state.pending_orders = []
        state.pending_decision_bar = None
        ctx.store.save(state)
        return fills


def sweep_pending(ctx: LiveLoopContext, *, sweep_suffix: str = "S1") -> list[Order]:
    """Morning completion sweep: re-submit unexecuted residuals as new orders.

    The Alpaca *paper* venue's opening-auction simulation prints only ~20-25%
    of OPG orders (observed across both instrument books), so a monthly
    refresh left standing would hold a fraction of the decided book for a
    whole cadence period — the paper stream then measures a different
    portfolio than the one the decision constructed. The sweep runs after the
    open, while the decision is still pending (settle happens at the next
    evening cycle): for every pending order that is *terminal* at the venue
    (filled, partially filled, expired, rejected, or never submitted) it
    computes the unexecuted residual and submits it as a fresh order under a
    deterministic suffixed id (``"{original_id}:{sweep_suffix}"``), appended
    to the persisted pending set *before* submission (the same write-ahead
    discipline as the decision itself). The evening settle then treats sweep
    fills exactly like auction fills — same decision-close ``reference_price``,
    so the fills ledger records the *total* arrival cost of executing the
    decision at this venue; the id suffix keeps the two fill populations
    segmentable for I-9 calibration.

    Safety properties:

    * An order still *live* at the venue is never swept (double-execution);
      the broker must expose ``open_order_ids`` (duck-typed) or the sweep
      refuses to run.
    * Idempotent: rerunning regenerates the same suffixed ids — existing ones
      are resumed/skipped through the venue's duplicate-id rejection, never
      duplicated in the persisted pending set.
    * One completion generation per decision: a sweep order's own unfilled
      residual is logged at settle (unfilled ledger) but not re-swept.
    * A residual whose sign disagrees with its parent order (venue over-fill,
      which should not happen) is skipped loudly, never amplified.

    The venue order type is whatever the context's broker was constructed
    with — the sweep CLI wires ``time_in_force="day"`` (a plain market order
    after the open) precisely because the auction the OPG order was for has
    already happened.
    """
    gate = verify_authorized_binding(ctx)
    with held_execution_lock(ctx):
        state = ctx.store.load()
        if state is None or not state.pending_orders:
            return []
        open_ids_fn = getattr(ctx.broker, "open_order_ids", None)
        if open_ids_fn is None:
            raise TypeError(
                "broker does not expose open_order_ids(client_order_ids); refusing to sweep "
                "blind — an order still queued for its auction would double-execute (N7)"
            )
        pending_ids = {o.client_order_id for o in state.pending_orders}
        open_ids = set(open_ids_fn(pending_ids))
        # A completion sweep is a retry: the residual it re-offers is the parent
        # order's quantity minus what the venue printed, and neither term is
        # known while the parent's fate is not (state S3, §3.11).
        recovery = pending_order_recovery(ctx, state, live_ids=open_ids)
        if not recovery.resolved:
            raise UnresolvedSubmission(recovery.refusal(), recovery)
        executed: dict[str, float] = {}
        for fill in ctx.broker.fills_for(pending_ids):
            executed[fill.client_order_id] = executed.get(fill.client_order_id, 0.0) + fill.qty

        existing_ids = set(pending_ids)
        # Only a parent whose own fate is still open is completed. An order the
        # venue rejected at submission, or one cancelled to leave state S3, has
        # a recorded terminal fate; re-offering its residual under a suffixed id
        # is the retry §3.7 removed from the submission path, and doing it from
        # here would put it straight back (§3.7, §3.11).
        persisted_open = {o.client_order_id for o in state.open_pending_orders}
        parents = [
            o
            for o in state.pending_orders
            if not o.client_order_id.endswith(f":{sweep_suffix}") and o.client_order_id in persisted_open
        ]
        new_orders: list[Order] = []
        skipped_open = 0
        for order in parents:
            child_id = f"{order.client_order_id}:{sweep_suffix}"
            if order.client_order_id in open_ids or child_id in open_ids:
                skipped_open += 1
                continue
            done = executed.get(order.client_order_id, 0.0) + executed.get(child_id, 0.0)
            residual = order.qty - done
            if residual == 0.0 or child_id in existing_ids:
                continue
            if residual * order.qty < 0:
                logger.warning(
                    "sweep: residual %s for %s exceeds the order (%s executed vs %s ordered); "
                    "skipping, never amplifying a venue over-fill",
                    residual,
                    order.client_order_id,
                    done,
                    order.qty,
                )
                continue
            new_orders.append(
                Order(
                    client_order_id=child_id,
                    symbol=order.symbol,
                    qty=residual,
                    decision_bar=order.decision_bar,
                    reference_price=order.reference_price,
                )
            )
        if skipped_open:
            logger.warning(
                "sweep %s: %d orders still live at the venue — not swept this pass "
                "(auction pending or open order); rerun after they turn terminal",
                state.pending_decision_bar,
                skipped_open,
            )
        if new_orders:
            state.pending_orders = list(state.pending_orders) + new_orders
            ctx.store.save(state)  # write-ahead: the sweep decision persists before any submit
            logger.info(
                "sweep %s: re-submitting %d unexecuted residuals as %s orders",
                state.pending_decision_bar,
                len(new_orders),
                sweep_suffix,
            )
        # Resume/submit every sweep order not yet at the venue (crash-safe rerun).
        sweep_orders = [o for o in state.pending_orders if o.client_order_id.endswith(f":{sweep_suffix}")]
        submit_pending(ctx, state, sweep_orders, gate=gate)
        return new_orders


def cancel_orders(
    ctx: LiveLoopContext,
    client_order_ids: Iterable[str] | None = None,
    *,
    states: Iterable[OperatingState] = (),
    action_context: ActionContext = ActionContext(),
    now: datetime | None = None,
) -> list[str]:
    """C0: cancel open orders. The narrowest thing the engine can ask a venue for.

    This is the action §3.10 permits in states where nothing else is — S2 (after
    a fresh open-order read), S3, S7, S8 — and the one an expired or narrowed
    grant keeps, because ``manage_existing`` and ``add_exposure`` are separate
    flags. It is authorized per order id and refuses the whole call on the first
    id it may not cancel, before any cancel request is made.

    Under S6 the engine performs no cancel of its own: pass
    ``action_context=ActionContext(operator_initiated=True)`` for the operator's
    own cancel, and an engine-initiated one is refused. ``client_order_ids``
    defaults to the persisted pending set.
    """
    gate = verify_authorized_binding(ctx)
    with held_execution_lock(ctx):
        if client_order_ids is None:
            state = ctx.store.load()
            client_order_ids = [] if state is None else [o.client_order_id for o in state.pending_orders]
        ids = list(dict.fromkeys(client_order_ids))
        if gate is not None:
            for client_order_id in ids:
                gate.authorize(
                    ActionClass.C0,
                    sid=client_order_id,
                    states=states,
                    context=action_context,
                    now=now,
                )
        for client_order_id in ids:
            ctx.broker.cancel(client_order_id)
        if ids:
            logger.info("cancelled %d order(s) at the venue: %s", len(ids), ids[:8])
        return ids


def read_fills_ledger(path: Path) -> pd.DataFrame:
    """The append-only fills ledger as a frame (the I-9 calibration input)."""
    return ledgers.read_frame(path)


def read_equity_ledger(path: Path) -> pd.DataFrame:
    """The append-only equity ledger as a frame — one mark-to-market NAV row per
    decision bar (``decision_bar``, ``equity``, ``cash``). This is the
    return-series source for the anytime-valid monitor
    (:mod:`prism.validation.anytime` via :mod:`prism.live.monitor`)."""
    return ledgers.read_frame(path)


def read_targets_ledger(path: Path) -> list[dict]:
    """The refresh-target ledger rows, oldest first (concordance baseline)."""
    return ledgers.read_rows(path)


def read_concordance_ledger(path: Path) -> pd.DataFrame:
    """The book-concordance telemetry ledger as a frame."""
    return ledgers.read_frame(path)


def read_shadow_ledger(path: Path) -> pd.DataFrame:
    """The shadow-return ledger as a frame — one row per shadowed session
    (``decision_bar``, ``prev_bar``, ``shadow_return``, ``paper_return``,
    ``transition``, ``refresh_bar``, ``n_names_used``, ``n_names_missing``).
    The tracking monitor (:func:`prism.live.monitor.shadow_tracking_read`)
    joins it with the equity ledger by decision bar."""
    return ledgers.read_frame(path)


def read_regime_ledger(path: Path) -> pd.DataFrame:
    """The per-cycle regime telemetry ledger as a frame — one row per decision
    bar (``decision_bar``, ``clean``, ``gross_scale``, ``blocks``,
    ``failures``, …). The handoff §8 precondition-(b) 21-session clock reads
    ``clean`` (docs/regime_step.md)."""
    return ledgers.read_frame(path)


def _reconcile(state: LoopState, broker: Broker) -> None:
    """Adopt broker truth, loudly when it disagrees with the persisted book."""
    broker_positions = {s: q for s, q in broker.positions().items() if q != 0.0}
    symbols = set(broker_positions) | set(state.positions)
    diverged = {
        s
        for s in symbols
        if abs(broker_positions.get(s, 0.0) - state.positions.get(s, 0.0)) > _RECONCILE_SHARE_TOLERANCE
    }
    if diverged:
        logger.warning(
            "reconcile: broker truth differs from persisted state on %s; adopting broker",
            sorted(diverged),
        )
    state.positions = broker_positions
    state.cash = broker.cash()


def require_price(prices: pd.Series, symbol: str) -> float:
    """A finite positive price for ``symbol``, or raise (N7) — the mark/size
    precondition shared by the protocol and the daily driver."""
    price = prices.get(symbol)
    if price is None or not math.isfinite(float(price)) or float(price) <= 0:
        raise ValueError(f"cannot value held position {symbol!r}: price {price!r} (N7)")
    return float(price)


def append_targets_ledger(
    path: Path,
    refresh_bar: str,
    decision_bar: str,
    target_weights: pd.Series,
    prices: pd.Series,
    equity: float,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Persist the constructed refresh book (nonzero weights + their decision
    closes). Idempotent per ``refresh_bar`` so a same-bar restart never
    duplicates the row; absent names read as an explicit 0.0 downstream.
    ``extra`` adds provenance fields (the panel digest and file) to the row;
    it may not override the row's own fields."""
    book = {
        str(symbol): float(weight)
        for symbol, weight in target_weights.items()
        if isinstance(weight, (int, float)) and math.isfinite(float(weight)) and float(weight) != 0.0
    }
    reference_prices = {symbol: float(prices[symbol]) for symbol in book}
    row: dict[str, Any] = {
        "refresh_bar": refresh_bar,
        "decision_bar": decision_bar,
        "equity": float(equity),
        "targets": book,
        "reference_prices": reference_prices,
    }
    if extra:
        clash = sorted(set(extra) & set(row))
        if clash:
            raise ValueError(f"targets ledger extra fields may not override the row's own: {clash}")
        row.update({str(key): value for key, value in extra.items()})
    ledgers.append_monotone(path, "refresh_bar", row)


def append_concordance_ledger(path: Path, decision_bar: str, row: dict) -> None:
    """Append one concordance read per decision bar (idempotent, monotone —
    the same discipline as the equity ledger)."""
    ledgers.append_monotone(path, "decision_bar", {"decision_bar": decision_bar, **row})


def append_shadow_ledger(path: Path, decision_bar: str, row: dict) -> bool:
    """Append one shadow read per decision bar (idempotent, monotone — the
    concordance ledger's discipline). Returns ``True`` when written."""
    return ledgers.append_monotone(path, "decision_bar", {"decision_bar": decision_bar, **row})


def append_regime_ledger(path: Path, decision_bar: str, state: dict, gross_scale: float | None) -> None:
    """Append one regime telemetry read per decision bar (idempotent, monotone
    — the equity-ledger discipline, so a same-bar restart never duplicates a
    session on the precondition-(b) clock). ``clean`` is the session verdict:
    True iff the read carries zero failure entries. ``gross_scale`` is the
    clamped multiplier the §7.7 action hook applied to this refresh's
    construction — null while the hook is unarmed
    (docs/sizing_preregistration.md unratified) or the session did not
    qualify (not a refresh, or a not-clean read)."""
    row = {
        "decision_bar": decision_bar,
        "clean": not state.get("failures"),
        "gross_scale": gross_scale,
        **{key: value for key, value in state.items() if key != "decision_bar"},
    }
    ledgers.append_monotone(path, "decision_bar", row)


def append_equity_ledger(path: Path, decision_bar: str, equity: float, cash: float) -> None:
    """Append one mark-to-market NAV snapshot for ``decision_bar``.

    Idempotent and monotone: a same-bar rerun (the write-ahead protocol's
    restart) or an out-of-order bar is skipped, so the ledger holds exactly one
    row per bar and re-running the loop never double-counts a daily return.
    ``decision_bar`` strings are ISO dates, so lexical ``<=`` is chronological.
    """
    ledgers.append_monotone(
        path,
        "decision_bar",
        {"decision_bar": decision_bar, "equity": float(equity), "cash": float(cash)},
    )


# Pre-2026-07-29 private names, kept so downstream callers migrate on their
# own clock (tests and the daily driver now use the public names above).
_require_price = require_price
_append_targets_ledger = append_targets_ledger
_append_concordance_ledger = append_concordance_ledger
_append_regime_ledger = append_regime_ledger
_append_equity_ledger = append_equity_ledger
