"""Authorization: the record, the action classes, the operating states (§3.10).

What this module is. An *authorization record* is what the account owner
approved: which account, at which endpoint, from which host, running which
configuration and which engine build, with which scope, until when. It lives at
``workspace.run_dir(account) / "authorization.json"``, it is a local attestation
whose integrity rests on filesystem permissions, and it is **not** a credential
and **not** a signature. It certifies nothing about profitability and does not
sandbox private code (§3.10).

What it is not. It is not a rail against a broken engine and it is not a risk
limit — those are ``live/safety.py`` and ``portfolio/limits.py``. It answers one
question: *did the owner approve this process, on this host, against this
account, for this action?* Every answer is a refusal or a permission, never a
warning: a control that cannot verify its precondition must refuse.

**Where the refusal happens.** Before any request. The gate is verified at the
top of ``loop.decide_and_submit`` — ahead of the reconcile read, so a mis-bound
process never touches the venue at all — and again in
``alpaca.AlpacaBroker.submit`` before the request body is built, which is the
last line. Nothing between the two constructs an order request.

**How the rows combine.** The permitted set is the *intersection* of the
permitted sets of every applicable row, further intersected with the gates of
the scope flags and with the per-sid freshness of the four input classes. That
replaces "the most restrictive applicable row governs", which is undefined when
rows permit disjoint sets: S1 permits {C0, C1, C4, OB} and S4a permits
{C0, C1, OB} plus C4 only with ``hedge``, and neither is a subset of the other.
:func:`permitted_classes` is that intersection, written once, and every caller
goes through it.

**C2 and C5 are derived, not tabulated.** C2 is a C1 leg and a C3 leg "each
authorized on its own", so it is permitted exactly when both legs are; C5 is
"treated as C3, except inside a full liquidation of the account, where it is
treated as C1". Deriving them is what makes a full liquidation under a state
that forbids C3 *defined* rather than undecidable, so the state table below
carries only the primitive classes C0, C1, C3, C4 and OB.

**Scope separation.** A missing signal (S1) and an expired or narrowed grant
(S4a) both keep C0, C1 and OB: management of existing positions is never
disabled by the absence of a reason to open a new one. What they remove is C3.
S4b — no record — is the only row that removes C0 as well, because a process
that cannot name what it was authorized to do may not write to the venue at
all.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import socket
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "AUTHORIZATION_FILENAME",
    "AUTHORIZATION_SCHEMA",
    "ENDPOINTS_BY_MODE",
    "LIVE_BASE_URL",
    "MODES",
    "PAPER_BASE_URL",
    "POLICY_KEYS",
    "ActionClass",
    "ActionContext",
    "AuthorizationGate",
    "AuthorizationRecord",
    "AuthorizationRefused",
    "InputFreshness",
    "OperatingState",
    "ProcessIdentity",
    "Scope",
    "SessionIntent",
    "classify_order",
    "distribution_id",
    "execution_host",
    "load_authorization",
    "parse_authorization",
    "permitted_classes",
    "refuse_an_unbindable_record",
    "resolve_endpoint",
]

#: The record schema this module reads. A different number is a hard stop, never
#: a migration guess: an authorization record is the one file whose
#: misinterpretation routes real money.
AUTHORIZATION_SCHEMA = 1

#: The record's name inside ``workspace.run_dir(account)``.
AUTHORIZATION_FILENAME = "authorization.json"

#: The two operating modes. ``--account paper|live`` selects one and there is no
#: default (§3.14); an environment variable may set an *endpoint* but never the
#: mode, which is why no code in this module reads the environment for it.
MODES: tuple[str, ...] = ("paper", "live")

#: Every key of a specification's ``[policy]`` block, mirrored by the record's
#: ``policy_ids`` so a reader of the record alone sees every policy file the
#: grant covered (§3.10).
POLICY_KEYS: tuple[str, ...] = ("profile", "costs", "execution", "monitoring", "recovery")

#: The venue endpoints, per mode. They live here rather than in the adapter
#: because they are an *authorization* allowlist: the comparison of §3.10 is
#: exact string equality against the per-mode entry, and the adapter re-exports
#: these names so existing callers are unaffected.
PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"

ENDPOINTS_BY_MODE: Mapping[str, tuple[str, ...]] = {
    "paper": (PAPER_BASE_URL,),
    "live": (LIVE_BASE_URL,),
}

#: The engine distribution whose installed version is ``engine_version`` (§3.10).
ENGINE_DISTRIBUTION = "prism-trading"

#: The calendar span used to bound a submission window when the caller supplies
#: no session-table window end. One session gap never exceeds a four-day holiday
#: weekend, so a week per session is an over-estimate — which is the direction a
#: refusal must err in. A caller holding the session table passes ``window_end``
#: and this bound is not consulted.
SESSION_CALENDAR_UPPER_BOUND_DAYS = 7


class AuthorizationRefused(RuntimeError):
    """This process may not perform this action against this account.

    Raised *before* any request is constructed. It is never caught and
    downgraded: the loop's ``OrderRejected`` path is for a venue's verdict on a
    healthy order, and this is the engine's verdict on its own authority.
    """


class ActionClass(str, Enum):
    """The action classes of §3.10, each gated by a scope flag.

    ==  ==========================================================  =================
    C0  cancel an open order                                        manage_existing
    C1  reduce a position toward zero without crossing              manage_existing
    C2  cross through zero, decomposed into a C1 leg and a C3 leg   manage_existing
                                                                    and add_exposure
    C3  open a new position or add to one, in either direction      add_exposure
    C4  add a declared hedge instrument when the validator shows    hedge
        the declared risk measure falls
    C5  reduce a declared hedge instrument; treated as C3, except   hedge, or
        inside a full liquidation, where it is treated as C1        manage_existing
                                                                    in a liquidation
    OB  obligation progress (cover, liquidate, reduce_to)           manage_existing
    ==  ==========================================================  =================
    """

    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"
    OB = "OB"


#: The classes the state table tabulates. C2 and C5 are *derived* from these by
#: :func:`permitted_classes`, per the decomposition rules of §3.10.
PRIMITIVE_CLASSES: frozenset[ActionClass] = frozenset(
    {ActionClass.C0, ActionClass.C1, ActionClass.C3, ActionClass.C4, ActionClass.OB}
)

#: Scope flags each class requires. C5 in a full liquidation is the one
#: context-dependent row (§3.10) and is handled in :func:`_scope_permitted`.
SCOPE_REQUIREMENTS: Mapping[ActionClass, tuple[str, ...]] = {
    ActionClass.C0: ("manage_existing",),
    ActionClass.C1: ("manage_existing",),
    ActionClass.C2: ("manage_existing", "add_exposure"),
    ActionClass.C3: ("add_exposure",),
    ActionClass.C4: ("hedge",),
    ActionClass.C5: ("hedge",),
    ActionClass.OB: ("manage_existing",),
}


class OperatingState(str, Enum):
    """The operating states of §3.10. Every one of them *narrows*; none widens.

    S0 is the identity of the intersection, so a caller that has established
    nothing passes no states at all and gets the scope-and-freshness answer.
    S9 is a monitor row: the monitor performs no broker action under it, and the
    next engine invocation is governed by its own data and broker-state
    freshness rather than by this row, so an engine path never passes S9.
    """

    S0_CLEAN = "S0"
    S1_STALE_ALPHA = "S1"
    S2_STALE_BROKER_STATE = "S2"
    S3_UNRESOLVED_SUBMISSION = "S3"
    S4A_AUTHORIZATION_NARROWED = "S4a"
    S4B_AUTHORIZATION_ABSENT = "S4b"
    S5_BREACHED_LIMITS = "S5"
    S6_HALT = "S6"
    S7_ACCOUNTING_DISCREPANCY = "S7"
    S8_CLOCK_FAULT = "S8"
    S9_SCHEDULER_FAULT = "S9"


class SessionIntent(str, Enum):
    """What an invocation declares it may do, before it knows its own batch.

    The decision path may open or add to a position by construction: it diffs a
    constructed book against the held one and the diff's sign is not known until
    after the broker has been read. Declaring the intent up front is what lets
    the gate refuse a mis-bound or expired grant *before* the first broker call
    rather than after, and it is what keeps ``MANAGE_EXISTING`` — cancellation,
    reduction, obligation progress — available under exactly the states §3.10
    says it stays available under.
    """

    REBALANCE = "rebalance"
    MANAGE_EXISTING = "manage_existing"

    @property
    def classes(self) -> frozenset[ActionClass]:
        """The classes an invocation with this intent may emit."""
        if self is SessionIntent.MANAGE_EXISTING:
            return frozenset({ActionClass.C0, ActionClass.C1, ActionClass.OB})
        return frozenset(ActionClass)


@dataclass(frozen=True)
class Scope:
    """The record's ``scope`` block: three independent flags (§3.10)."""

    add_exposure: bool = False
    manage_existing: bool = False
    hedge: bool = False

    def to_dict(self) -> dict[str, bool]:
        """The block as it is written in the record."""
        return {
            "add_exposure": self.add_exposure,
            "manage_existing": self.manage_existing,
            "hedge": self.hedge,
        }


@dataclass(frozen=True)
class InputFreshness:
    """The four input classes of §3.11, per sid, each checked separately.

    ``mark`` is the marking price, ``halt`` the halt status, ``covariance`` the
    risk snapshot, ``signal`` the signal inputs. They are independent: a sid may
    have a fresh mark and a stale signal (that is S1), and a sid failing every
    class degrades to C0 only *for that sid*, never for the account.
    """

    mark: bool = True
    halt: bool = True
    covariance: bool = True
    signal: bool = True


@dataclass(frozen=True)
class ActionContext:
    """The conditions §3.10's "only when" qualifiers are evaluated against.

    Each one names a fact the *caller* establishes; none is guessed here. The
    defaults are the conservative reading — a condition nobody established is
    false, so the action it qualifies is not in the row's set.
    """

    #: S1: C4 is in the row's set only when the risk snapshot's declared inputs
    #: are fresh.
    risk_inputs_fresh: bool = False
    #: S2: C0 is permitted "only after a fresh open-order read; then refresh".
    open_orders_read_fresh: bool = False
    #: S5: C3 is permitted "only within achieved limits" (and with add_exposure).
    within_achieved_limits: bool = False
    #: S6: the engine performs no cancel of its own; an operator-initiated
    #: cancel is the C0 that row permits.
    operator_initiated: bool = False
    #: C5's reclassification, and C5's alternate scope flag.
    full_liquidation: bool = False
    #: C1 all the way to flat needs no price, so it survives a stale mark; a
    #: partial reduction does not.
    reduction_to_flat: bool = False
    #: C4: "when the validator shows the declared risk measure falls". A hedge
    #: nobody demonstrated is not a hedge.
    hedge_risk_measure_falls: bool = False
    #: The per-sid freshness of the four input classes.
    freshness: InputFreshness = InputFreshness()


def _state_permitted(state: OperatingState, context: ActionContext) -> frozenset[ActionClass]:
    """The primitive classes one row permits, with its "only when" applied."""
    if state is OperatingState.S0_CLEAN:
        return PRIMITIVE_CLASSES
    if state is OperatingState.S1_STALE_ALPHA:
        allowed = {ActionClass.C0, ActionClass.C1, ActionClass.OB}
        if context.risk_inputs_fresh:
            allowed.add(ActionClass.C4)
        return frozenset(allowed)
    if state is OperatingState.S2_STALE_BROKER_STATE:
        # "C0 only, and only after a fresh open-order read; then refresh."
        return frozenset({ActionClass.C0}) if context.open_orders_read_fresh else frozenset()
    if state is OperatingState.S3_UNRESOLVED_SUBMISSION:
        # OB under S3 means escalation through the recovery policy, never a new
        # broker write, so it is not in this set of permitted broker actions.
        return frozenset({ActionClass.C0})
    if state is OperatingState.S4A_AUTHORIZATION_NARROWED:
        # The row's "C4 only with hedge" is the scope gate, which applies to C4
        # in every state; naming C4 here and letting the scope gate remove it is
        # the same set, stated once.
        return frozenset({ActionClass.C0, ActionClass.C1, ActionClass.C4, ActionClass.OB})
    if state is OperatingState.S4B_AUTHORIZATION_ABSENT:
        return frozenset()
    if state is OperatingState.S5_BREACHED_LIMITS:
        allowed = {ActionClass.C0, ActionClass.C1, ActionClass.C4, ActionClass.OB}
        if context.within_achieved_limits:
            allowed.add(ActionClass.C3)
        return frozenset(allowed)
    if state is OperatingState.S6_HALT:
        # "C0 only, and the engine performs no cancel of its own." OB escalates
        # per policy and is not a broker write here.
        return frozenset({ActionClass.C0}) if context.operator_initiated else frozenset()
    if state is OperatingState.S7_ACCOUNTING_DISCREPANCY:
        # The positions/cash disagreement is adopt-and-continue and does not
        # remove the engine from S0 (§3.11); this row is the other outcome —
        # open orders differing in quantity or side, or a fill absent from the
        # ledger — which is C0 and reconcile only.
        return frozenset({ActionClass.C0})
    if state is OperatingState.S8_CLOCK_FAULT:
        return frozenset({ActionClass.C0})
    if state is OperatingState.S9_SCHEDULER_FAULT:
        return frozenset()
    raise ValueError(f"unhandled operating state {state!r}")  # pragma: no cover - the enum is closed


def _freshness_permitted(context: ActionContext) -> frozenset[ActionClass]:
    """The primitive classes this sid's input freshness leaves available (§3.11).

    A sid failing every class degrades to C0 only. A reduction all the way to
    flat needs no price and survives a stale mark; a partial one does not,
    because its size depends on the price. A hedge sized on a stale covariance
    is not a demonstrated risk reduction.
    """
    fresh = context.freshness
    if not fresh.halt:
        # Without a halt status, nothing about the sid is known well enough to
        # send anything but a cancel.
        return frozenset({ActionClass.C0})
    allowed = {ActionClass.C0, ActionClass.OB}
    if fresh.mark or context.reduction_to_flat:
        allowed.add(ActionClass.C1)
    if fresh.mark and fresh.signal:
        allowed.add(ActionClass.C3)
    if fresh.mark and fresh.covariance:
        allowed.add(ActionClass.C4)
    return frozenset(allowed)


def _scope_permitted(scope: Scope, context: ActionContext) -> frozenset[ActionClass]:
    """Every class the scope flags admit, C2's and C5's compound rows included."""
    flags = scope.to_dict()
    allowed: set[ActionClass] = set()
    for action, required in SCOPE_REQUIREMENTS.items():
        if action is ActionClass.C5 and context.full_liquidation:
            # "hedge, or manage_existing in a full liquidation."
            if scope.hedge or scope.manage_existing:
                allowed.add(action)
            continue
        if all(flags[flag] for flag in required):
            allowed.add(action)
    return frozenset(allowed)


def permitted_classes(
    states: Iterable[OperatingState] = (),
    scope: Scope = Scope(add_exposure=True, manage_existing=True, hedge=True),
    context: ActionContext = ActionContext(),
) -> frozenset[ActionClass]:
    """Every action class permitted here: the intersection of every row (§3.10).

    The order of the intersection is fixed and total — states, then the per-sid
    freshness of the four input classes, then the scope flags — and C2 and C5
    are derived last from the primitives that survive it:

    * C2 is permitted exactly when C1 and C3 both are, since it is a C1 leg and
      a C3 leg each authorized on its own.
    * C5 is permitted exactly when C1 is inside a full liquidation of the
      account, and exactly when C3 is otherwise.
    * C4 additionally requires that the validator showed the declared risk
      measure falling; a hedge nobody demonstrated is not a hedge.

    Scope flags apply in every state, not only the rows that name them, and a
    state set containing S4b permits nothing, whatever the other rows say.
    """
    allowed = set(PRIMITIVE_CLASSES)
    for state in states:
        allowed &= _state_permitted(state, context)
    allowed &= _freshness_permitted(context)
    scoped = _scope_permitted(scope, context)
    allowed &= scoped
    if not context.hedge_risk_measure_falls:
        allowed.discard(ActionClass.C4)

    if ActionClass.C1 in allowed and ActionClass.C3 in allowed and ActionClass.C2 in scoped:
        allowed.add(ActionClass.C2)
    reduce_leg = ActionClass.C1 if context.full_liquidation else ActionClass.C3
    if reduce_leg in allowed and ActionClass.C5 in scoped:
        allowed.add(ActionClass.C5)
    return frozenset(allowed)


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthorizationRecord:
    """``workspace.run_dir(account) / "authorization.json"``, parsed (§3.10).

    Every field is required; the file is read strictly, because a record with a
    field the reader silently defaulted is a record nobody approved.
    ``granted_by`` and ``granted_at`` are attestations recorded for the audit
    trail and nothing checks them. ``expires_at`` is required and non-null under
    ``live`` and may be null under ``paper``.
    """

    schema: int
    mode: str
    account_name: str
    account_id: str
    endpoint: str
    execution_host: str
    config_id: str
    engine_version: str
    distribution_id: str
    policy_ids: Mapping[str, str]
    scope: Scope
    submission_window_sessions: int
    granted_by: str
    granted_at: str
    expires_at: str | None
    source: Path | None = None

    def __post_init__(self) -> None:
        if self.schema != AUTHORIZATION_SCHEMA:
            raise AuthorizationRefused(
                f"authorization record schema {self.schema!r} is not {AUTHORIZATION_SCHEMA}; refusing to "
                "interpret a record written for another reader (§3.10)"
            )
        if self.mode not in MODES:
            raise AuthorizationRefused(f"authorization record mode {self.mode!r} is not one of {MODES}")
        for name in ("account_name", "account_id", "endpoint", "execution_host", "config_id", "engine_version"):
            if not str(getattr(self, name)):
                raise AuthorizationRefused(f"authorization record {name} is empty; it binds nothing")
        if not str(self.distribution_id):
            raise AuthorizationRefused(
                "authorization record distribution_id is empty; a version string alone cannot distinguish "
                "the pinned release the grant was made against from a modified checkout carrying the same "
                "string (§3.10)"
            )
        missing = [key for key in POLICY_KEYS if key not in self.policy_ids]
        if missing:
            raise AuthorizationRefused(
                f"authorization record policy_ids is missing {missing}; it mirrors every key of the "
                "[policy] block so a reader of the record alone sees every policy file the grant covered"
            )
        if int(self.submission_window_sessions) < 1:
            raise AuthorizationRefused(
                f"submission_window_sessions must be >= 1, got {self.submission_window_sessions!r}"
            )
        if self.mode == "live" and self.expires_at is None:
            raise AuthorizationRefused("expires_at is required and non-null under mode 'live' (§3.10)")
        if self.expires_at is not None:
            _parse_timestamp(self.expires_at, field="expires_at")

    @property
    def expiry(self) -> datetime | None:
        """``expires_at`` as an aware datetime, or ``None`` when unbounded."""
        if self.expires_at is None:
            return None
        return _parse_timestamp(self.expires_at, field="expires_at")

    def to_dict(self) -> dict[str, Any]:
        """The record in the shape §3.10 declares, for writing it back out."""
        return {
            "schema": self.schema,
            "mode": self.mode,
            "account_name": self.account_name,
            "account_id": self.account_id,
            "endpoint": self.endpoint,
            "execution_host": self.execution_host,
            "config_id": self.config_id,
            "engine_version": self.engine_version,
            "distribution_id": self.distribution_id,
            "policy_ids": dict(self.policy_ids),
            "scope": self.scope.to_dict(),
            "submission_window_sessions": int(self.submission_window_sessions),
            "granted_by": self.granted_by,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
        }


_RECORD_FIELDS: tuple[str, ...] = (
    "schema",
    "mode",
    "account_name",
    "account_id",
    "endpoint",
    "execution_host",
    "config_id",
    "engine_version",
    "distribution_id",
    "policy_ids",
    "scope",
    "submission_window_sessions",
    "granted_by",
    "granted_at",
    "expires_at",
)


def _parse_timestamp(value: str, *, field: str) -> datetime:
    """An ISO 8601 instant with an explicit offset, or a refusal.

    A naive timestamp is refused rather than assumed local or UTC: an expiry
    whose meaning depends on the reader's host is not an expiry.
    """
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise AuthorizationRefused(
            f"authorization record {field}={value!r} is not an ISO 8601 instant: {exc}"
        ) from exc
    if parsed.tzinfo is None:
        raise AuthorizationRefused(
            f"authorization record {field}={value!r} carries no UTC offset; an expiry whose meaning depends "
            "on the reader's host is not an expiry"
        )
    return parsed


def parse_authorization(payload: Mapping[str, Any], *, source: Path | None = None) -> AuthorizationRecord:
    """Parse one record mapping strictly: no unknown keys, no defaulted fields."""
    where = f" in {source}" if source is not None else ""
    unknown = sorted(set(payload) - set(_RECORD_FIELDS))
    if unknown:
        raise AuthorizationRefused(
            f"authorization record{where} carries unknown key(s) {unknown}; a record the reader does not "
            "fully understand authorizes nothing (§3.10)"
        )
    missing = [name for name in _RECORD_FIELDS if name not in payload]
    if missing:
        raise AuthorizationRefused(f"authorization record{where} is missing required field(s) {missing}")

    scope_block = payload["scope"]
    if not isinstance(scope_block, Mapping):
        raise AuthorizationRefused(f"authorization record{where} scope must be a table of the three flags")
    unknown_flags = sorted(set(scope_block) - {"add_exposure", "manage_existing", "hedge"})
    if unknown_flags:
        raise AuthorizationRefused(f"authorization record{where} scope carries unknown flag(s) {unknown_flags}")
    policy_ids = payload["policy_ids"]
    if not isinstance(policy_ids, Mapping):
        raise AuthorizationRefused(f"authorization record{where} policy_ids must be a table")

    return AuthorizationRecord(
        schema=int(payload["schema"]),
        mode=str(payload["mode"]),
        account_name=str(payload["account_name"]),
        account_id=str(payload["account_id"]),
        endpoint=str(payload["endpoint"]),
        execution_host=str(payload["execution_host"]),
        config_id=str(payload["config_id"]),
        engine_version=str(payload["engine_version"]),
        distribution_id=str(payload["distribution_id"]),
        policy_ids={str(key): str(value) for key, value in policy_ids.items()},
        scope=Scope(
            add_exposure=bool(scope_block.get("add_exposure", False)),
            manage_existing=bool(scope_block.get("manage_existing", False)),
            hedge=bool(scope_block.get("hedge", False)),
        ),
        submission_window_sessions=int(payload["submission_window_sessions"]),
        granted_by=str(payload["granted_by"]),
        granted_at=str(payload["granted_at"]),
        expires_at=None if payload["expires_at"] is None else str(payload["expires_at"]),
        source=source,
    )


def load_authorization(run_dir: Path | str) -> AuthorizationRecord | None:
    """The record in ``run_dir``, or ``None`` when the file is absent (S4b).

    Absence is not an error here — it is a *fact* the gate turns into S4b, which
    permits nothing. A record that exists but cannot be parsed is an error: a
    half-read authorization is worse than none, because it looks like one.
    """
    path = Path(run_dir) / AUTHORIZATION_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorizationRefused(f"authorization record {path} cannot be read: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AuthorizationRefused(f"authorization record {path} is not a JSON object")
    return parse_authorization(payload, source=path)


def refuse_an_unbindable_record(run_dir: Path | str, *, entry_point: str, instead: str) -> None:
    """Refuse when ``run_dir`` holds a record ``entry_point`` cannot bind against.

    An entry point that resolves no specification computes no ``config_id``,
    and ``config_id`` is the record's authoritative binding (§3.10). Such an
    entry point can therefore run *beside* a record but never *under* one: every
    comparison it could make would either be missing or be the record against
    itself. Running anyway would give an operator the appearance of the controls
    with none of their content, which is worse than plainly having none — so the
    session is refused and told which entry point does honor the record.

    Absence is not an error here. The pre-authorization paper instrument is a
    declared, still-supported path (§3.14), and it is what runs today.
    """
    path = Path(run_dir) / AUTHORIZATION_FILENAME
    if not path.is_file():
        return
    raise AuthorizationRefused(
        f"{path} holds an authorization record and {entry_point} cannot bind against it: the record's "
        f"authoritative binding is config_id, and this entry point resolves no specification, so it "
        f"computes none. Run the authorized session with {instead}; a run directory whose record is set "
        "aside runs the pre-authorization paper instrument as before (§3.10, §3.14)"
    )


# --------------------------------------------------------------------------
# The running process's own identity
# --------------------------------------------------------------------------


def execution_host() -> str:
    """This host's name, ``socket.gethostname()`` — what ``execution_host`` binds."""
    return socket.gethostname()


def distribution_id(package_root: Path | None = None) -> str:
    """Identify the loaded engine, not a private project's enclosing checkout.

    A noneditable installed package is identified by its own distribution's
    ``RECORD`` even when its virtual environment lives inside a git repository.
    Source and editable imports use ``git:<commit>[+dirty]``. Metadata for a
    different installed copy cannot identify an untracked source import.
    """
    root = (Path(package_root) if package_root is not None else Path(__file__).parent).resolve()
    try:
        distribution = importlib.metadata.distribution(ENGINE_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        installed_package = Path(distribution.locate_file("prism")).resolve()
        if root.is_relative_to(installed_package):
            direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
            if not direct_url.get("dir_info", {}).get("editable", False):
                record = distribution.read_text("RECORD")
                if not record:
                    raise AuthorizationRefused(
                        f"cannot identify this build: the installed {ENGINE_DISTRIBUTION} containing {root} "
                        "has no RECORD; an enclosing git repository cannot identify an installed wheel (§3.10)"
                    )
                return "sha256:" + hashlib.sha256(record.encode("utf-8")).hexdigest()
    stamp = _git_stamp(root)
    if stamp is not None:
        return stamp
    raise AuthorizationRefused(
        f"cannot identify this build: {root} is not inside a git checkout and no matching installed RECORD "
        f"identifies its {ENGINE_DISTRIBUTION} package; an unidentifiable build can match no record (§3.10)"
    )


def _git_stamp(root: Path) -> str | None:
    """``git:<commit>[+dirty]`` for the checkout containing ``root``, else None."""
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):  # no git on the host
        return None
    commit = head.stdout.strip()
    if head.returncode != 0 or not commit:
        return None
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "-uno"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = status.returncode == 0 and bool(status.stdout.strip())
    return f"git:{commit}" + ("+dirty" if dirty else "")


def resolve_endpoint(mode: str, *, base_url: str | None = None, from_environment: str | None = None) -> str:
    """The endpoint for ``mode``, refusing anything off that mode's allowlist.

    An environment variable may set an endpoint but never the mode (§3.10), so
    ``mode`` arrives as an argument and ``from_environment`` is only ever an
    endpoint. The comparison is exact string equality against the per-mode
    allowlist, and a live endpoint under ``paper`` — or the reverse — is refused
    here, where the client is constructed, before any order request.
    """
    if mode not in MODES:
        raise AuthorizationRefused(f"mode must be one of {MODES}, got {mode!r}; there is no default (§3.14)")
    allowed = ENDPOINTS_BY_MODE[mode]
    resolved = str(base_url or from_environment or allowed[0]).rstrip("/")
    if resolved not in allowed:
        named = "a live" if resolved == LIVE_BASE_URL else ("a paper" if resolved == PAPER_BASE_URL else "an unknown")
        raise AuthorizationRefused(
            f"endpoint {resolved!r} is {named} endpoint under mode {mode!r}; the allowlist for {mode!r} is "
            f"{list(allowed)} and the comparison is exact (§3.10). An environment variable may set an "
            "endpoint but never the mode: select the mode with --account paper|live."
        )
    return resolved


@dataclass(frozen=True)
class ProcessIdentity:
    """What this process *is*, as the record's fields are compared against it.

    ``account_id`` is the broker-reported account id, read once when the session
    binds and carried here afterwards: the gate must be answerable without a
    venue round trip, because it is consulted on the path that must not reach
    the venue. ``endpoint`` is the client's own resolved base URL — there is no
    broker-reported endpoint, and a venue's self-description could not answer
    "did this process point where the operator intended" anyway.
    """

    mode: str
    endpoint: str
    account_id: str
    config_id: str
    engine_version: str
    distribution_id: str
    execution_host: str

    @classmethod
    def current(
        cls,
        *,
        mode: str,
        endpoint: str,
        account_id: str,
        config_id: str,
        engine_version: str | None = None,
        build_id: str | None = None,
        host: str | None = None,
    ) -> "ProcessIdentity":
        """This process's identity, filling version, build and host from the host."""
        if engine_version is None:
            from prism.spec import engine_version as installed_engine_version

            engine_version = installed_engine_version()
        return cls(
            mode=mode,
            endpoint=str(endpoint).rstrip("/"),
            account_id=account_id,
            config_id=config_id,
            engine_version=engine_version,
            distribution_id=distribution_id() if build_id is None else build_id,
            execution_host=execution_host() if host is None else host,
        )


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def classify_order(
    symbol: str,
    qty: float,
    positions: Mapping[str, float],
    *,
    hedge_sids: Sequence[str] = (),
) -> ActionClass:
    """Which action class one signed-share order falls in (§3.10).

    Classification is over the *held* position and the position the order would
    leave: opening or adding in either direction is C3, reducing toward zero
    without crossing is C1, crossing zero is C2, and an order in a declared
    hedge instrument is C4 when it increases the hedge and C5 when it reduces
    one. A declared hedge instrument is one that appears in the ``[[hedges]]``
    table of the execution profile (§3.9); the caller passes that list, because
    this module parses no profile.
    """
    held = float(positions.get(symbol, 0.0))
    target = held + float(qty)
    if symbol in set(hedge_sids):
        return ActionClass.C4 if abs(target) > abs(held) else ActionClass.C5
    if held == 0.0:
        return ActionClass.C3
    if held * target < 0.0:
        return ActionClass.C2
    if abs(target) < abs(held):
        return ActionClass.C1
    return ActionClass.C3


class AuthorizationGate:
    """One account's record, bound to one process, answering one question at a time.

    Two verifications, at two points, with two different jobs:

    * :meth:`verify_binding` asks whether the record applies to *this* process at
      all — same mode, endpoint, account, configuration, engine build and host.
      It runs before the first broker read of any invocation that may write, so
      a mis-bound process never reaches the venue, and again in the adapter's
      ``submit`` before the request body is built.
    * :meth:`authorize` and :meth:`authorize_batch` ask whether an action is
      permitted, given the states the caller established. They run at
      write-ahead time, for the decision's whole order batch as a unit.

    A gate constructed with ``record=None`` is state S4b: every broker write is
    refused, C0 included.
    """

    def __init__(
        self,
        record: AuthorizationRecord | None,
        identity: ProcessIdentity,
        *,
        account: str | None = None,
        hedge_sids: Sequence[str] = (),
    ) -> None:
        self.record = record
        self.identity = identity
        self.account = account or (record.account_name if record is not None else identity.account_id)
        self.hedge_sids = tuple(hedge_sids)

    @classmethod
    def for_run_dir(
        cls,
        run_dir: Path | str,
        identity: ProcessIdentity,
        *,
        account: str | None = None,
        hedge_sids: Sequence[str] = (),
    ) -> "AuthorizationGate":
        """Load ``run_dir``'s record (absent is S4b, never an implicit grant)."""
        return cls(load_authorization(run_dir), identity, account=account, hedge_sids=hedge_sids)

    # -------------------------------------------------------------- binding

    def verify_binding(self, *, observed_endpoint: str | None = None) -> None:
        """Refuse unless this record was written for exactly this process.

        Every comparison is exact string equality and every failure names the
        field, the record's value and the process's value. A mismatch is not a
        narrowing — it means the record describes some *other* process — so it
        refuses outright rather than degrading to S4a: S4a is a grant that has
        run out of scope, and this is no grant at all.
        """
        record = self.record
        if record is None:
            raise AuthorizationRefused(
                f"no authorization record for account {self.account!r} ({AUTHORIZATION_FILENAME} absent): "
                "state S4b permits nothing and every broker write is refused, C0 included (§3.10)"
            )
        checks = (
            ("mode", record.mode, self.identity.mode),
            ("endpoint", record.endpoint.rstrip("/"), self.identity.endpoint.rstrip("/")),
            ("account_id", record.account_id, self.identity.account_id),
            ("config_id", record.config_id, self.identity.config_id),
            ("engine_version", record.engine_version, self.identity.engine_version),
            ("execution_host", record.execution_host, self.identity.execution_host),
            ("distribution_id", record.distribution_id, self.identity.distribution_id),
        )
        for name, granted, running in checks:
            if granted != running:
                raise AuthorizationRefused(
                    f"authorization record for account {self.account!r} was granted for {name}={granted!r}; "
                    f"this process runs {name}={running!r}. Refusing before any request: the owner approved "
                    "a different process (§3.10)"
                )
        # The record's own endpoint must be on its mode's allowlist, and the
        # client's resolved base URL must be the endpoint the record names.
        resolve_endpoint(record.mode, base_url=record.endpoint)
        if observed_endpoint is not None and str(observed_endpoint).rstrip("/") != record.endpoint.rstrip("/"):
            raise AuthorizationRefused(
                f"the client resolved base URL {observed_endpoint!r}, which is not the record's endpoint "
                f"{record.endpoint!r}; refusing before any request (§3.10)"
            )

    def verify_policies(self, resolved: Mapping[str, str]) -> None:
        """Refuse when the resolved policy content ids disagree with the record.

        ``config_id`` is the authoritative binding and ``policy_ids`` is a
        display of the resolved files, but a run whose resolved policy content
        ids disagree with the record is refused rather than reconciled (§3.10).
        """
        record = self.record
        if record is None:
            raise AuthorizationRefused(f"no authorization record for account {self.account!r}")
        differing = {
            key: (record.policy_ids.get(key), resolved.get(key))
            for key in sorted(set(record.policy_ids) | set(resolved))
            if record.policy_ids.get(key) != resolved.get(key)
        }
        if differing:
            raise AuthorizationRefused(
                f"resolved policy ids disagree with the authorization record for account {self.account!r}: "
                f"{differing}; refused rather than reconciled (§3.10)"
            )

    # ---------------------------------------------------------------- states

    def expired(self, *, now: datetime | None = None) -> bool:
        """Whether the grant's ``expires_at`` has passed (never under a null one)."""
        record = self.record
        if record is None:
            return False
        expiry = record.expiry
        if expiry is None:
            return False
        return _utcnow(now) >= expiry

    def scope(self, *, now: datetime | None = None) -> Scope:
        """The effective scope: an expired grant keeps management, not exposure.

        S4a is "record present; ``add_exposure`` false or expired;
        ``manage_existing`` present", so expiry *narrows* the scope rather than
        voiding the record. That row is what keeps a missing signal or an
        expired research approval from disabling management of existing
        positions.
        """
        record = self.record
        if record is None:
            return Scope()
        if self.expired(now=now):
            return replace(record.scope, add_exposure=False)
        return record.scope

    def authorization_states(self, *, now: datetime | None = None) -> frozenset[OperatingState]:
        """S4b, S4a, or neither — the states this *record* puts the engine in."""
        record = self.record
        if record is None:
            return frozenset({OperatingState.S4B_AUTHORIZATION_ABSENT})
        if self.expired(now=now) or not record.scope.add_exposure:
            return frozenset({OperatingState.S4A_AUTHORIZATION_NARROWED})
        return frozenset()

    def permitted(
        self,
        *,
        states: Iterable[OperatingState] = (),
        context: ActionContext = ActionContext(),
        now: datetime | None = None,
    ) -> frozenset[ActionClass]:
        """Every class permitted here, the record-derived states included."""
        combined = frozenset(states) | self.authorization_states(now=now)
        return permitted_classes(combined, self.scope(now=now), context)

    # ------------------------------------------------------------- decisions

    def verify_submission_window(
        self,
        *,
        window_end: datetime | None = None,
        now: datetime | None = None,
    ) -> None:
        """Refuse a decision whose grant expires inside its own submission window.

        The batch is authorized as a unit at write-ahead time and submitted as a
        unit, possibly completed by a later sweep: dropping one side of a
        balanced batch can leave the portfolio outside its net, beta, cash or
        gross limits (§3.7). For that to be sound ``expires_at`` must be no
        earlier than the end of the declared submission window, so a decision
        whose authorization expires inside it is refused before it is persisted
        (§3.10).

        ``window_end`` comes from the session table when the caller holds one.
        Without it the window is bounded by
        :data:`SESSION_CALENDAR_UPPER_BOUND_DAYS` calendar days per session,
        which over-estimates the span and so errs toward refusing.
        """
        record = self.record
        if record is None:
            raise AuthorizationRefused(f"no authorization record for account {self.account!r}")
        expiry = record.expiry
        if expiry is None:
            return  # unbounded, and legal only under paper (checked at parse)
        moment = _utcnow(now)
        if moment >= expiry:
            raise AuthorizationRefused(
                f"the authorization for account {self.account!r} expired at {record.expires_at} and it is "
                f"now {moment.isoformat()}; a decision that may add exposure is refused before it is "
                "persisted (§3.10). Cancellation, reduction and obligation progress stay available under "
                "S4a; a new decision does not."
            )
        end = window_end
        if end is None:
            end = moment + timedelta(days=SESSION_CALENDAR_UPPER_BOUND_DAYS * int(record.submission_window_sessions))
        if expiry < end:
            raise AuthorizationRefused(
                f"the authorization for account {self.account!r} expires at {record.expires_at}, inside the "
                f"submission window ending {end.isoformat()} ({record.submission_window_sessions} session(s) "
                "after the decision); refusing before the write-ahead persist, because a batch cannot be "
                "split (§3.7, §3.10)"
            )

    def verify_intent(
        self,
        intent: SessionIntent,
        *,
        states: Iterable[OperatingState] = (),
        context: ActionContext = ActionContext(),
        now: datetime | None = None,
    ) -> frozenset[ActionClass]:
        """Refuse an invocation that could do nothing it declared it might do.

        The check is deliberately weak — it refuses only when *no* class the
        intent covers is permitted — because the batch itself is authorized
        class by class at write-ahead time. What it buys is that a session with
        no authority at all (S4b, a halt, a clock fault) stops before its first
        broker read rather than after it.
        """
        allowed = self.permitted(states=states, context=context, now=now)
        usable = allowed & intent.classes
        if not usable:
            combined = frozenset(states) | self.authorization_states(now=now)
            raise AuthorizationRefused(
                f"account {self.account!r}: nothing a {intent.value} session may do is permitted here. "
                f"States {sorted(s.value for s in combined) or ['S0']}, scope {self.scope(now=now).to_dict()}, "
                f"permitted {sorted(a.value for a in allowed)}; refusing before any broker request (§3.10)"
            )
        return usable

    def authorize(
        self,
        action: ActionClass,
        *,
        sid: str | None = None,
        states: Iterable[OperatingState] = (),
        context: ActionContext = ActionContext(),
        now: datetime | None = None,
    ) -> None:
        """Refuse unless ``action`` is in the intersection of every applicable row."""
        allowed = self.permitted(states=states, context=context, now=now)
        if action in allowed:
            return
        combined = frozenset(states) | self.authorization_states(now=now)
        where = f" on {sid}" if sid else ""
        raise AuthorizationRefused(
            f"{action.value}{where} is not permitted for account {self.account!r}: states "
            f"{sorted(s.value for s in combined) or ['S0']}, scope {self.scope(now=now).to_dict()}, "
            f"permitted {sorted(a.value for a in allowed)}. Refusing before any request (§3.10)"
        )

    def authorize_batch(
        self,
        actions: Mapping[str, ActionClass],
        *,
        states: Iterable[OperatingState] = (),
        context: ActionContext = ActionContext(),
        freshness_by_sid: Mapping[str, InputFreshness] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Authorize a whole decision batch as a unit, naming every refusal.

        The batch is the unit because dropping one side of a balanced batch can
        leave the portfolio outside its net, beta, cash or gross limits (§3.7).
        Per-sid freshness still applies per sid — a coverage shortfall blocks
        classes per sid, not the whole account — so one sid whose inputs are
        stale refuses the *batch*, and the refusal names that sid.
        """
        refusals: list[str] = []
        for sid in sorted(actions):
            action = actions[sid]
            sid_context = context
            if freshness_by_sid is not None and sid in freshness_by_sid:
                sid_context = replace(context, freshness=freshness_by_sid[sid])
            allowed = self.permitted(states=states, context=sid_context, now=now)
            if action not in allowed:
                refusals.append(f"{sid}: {action.value} not in {sorted(a.value for a in allowed)}")
        if refusals:
            combined = frozenset(states) | self.authorization_states(now=now)
            shown = refusals[:8]
            raise AuthorizationRefused(
                f"account {self.account!r}: {len(refusals)} of {len(actions)} order(s) are not permitted "
                f"(states {sorted(s.value for s in combined) or ['S0']}, scope "
                f"{self.scope(now=now).to_dict()}): {shown}{'…' if len(refusals) > len(shown) else ''}. The "
                "batch is authorized as a unit and is refused as one, before the write-ahead persist (§3.10)"
            )


def _utcnow(now: datetime | None) -> datetime:
    """The clock the gate reads, refusing a naive one."""
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise AuthorizationRefused("the clock passed to the authorization gate carries no UTC offset")
    return now
