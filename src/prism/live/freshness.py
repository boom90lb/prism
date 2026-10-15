"""Freshness: four separate checks, each with its own threshold and consequence (§3.11).

Why four and not one. The tempting implementation is a single maximum
timestamp — "the newest thing I have is N minutes old" — and it is wrong in
four different directions at once, because the four questions have four
different answers and four different operating consequences:

* **Data freshness** asks whether each *source* published, by its declared
  deadline, for enough of the *eligible* securities. A single maximum
  timestamp is satisfied by one symbol arriving on time out of five hundred.
  Consequence: S1 for the affected sids, per input class and per sid.
* **Broker-state freshness** asks how long ago positions and cash were read.
  A stale book with perfectly fresh prices is the worse of the two faults, and
  a maximum over both hides it behind the fresher one. Consequence: S2.
* **Clock health** asks whether *this host's* clock agrees with the venue's.
  Every age above is computed against the host clock, so a host clock that has
  drifted makes all of them agree with each other and none of them true. No
  server time available is a fault, not an exemption. Consequence: S8.
* **Scheduler liveness** asks whether the job fired at all. It is the one
  check no timestamp can answer, because the failure mode *is* the absence of
  a timestamp: nothing was written, so nothing is stale. The recorded
  2026-07-23..29 outage produced six consecutive nonzero nightly exits while
  the morning sweep exited zero and the doctor reported all checks passing.
  Consequence: S9, alert only, from a trigger independent of the trading
  scheduler — a check that scheduler runs cannot detect its own failure.

**A holiday is not a missed run.** This is the case the check exists to get
right and the case weekday arithmetic gets wrong. The engine carries
``prism.io.sessions`` — a rule-derived exchange calendar covering 2006-2028 —
precisely so that "the job did not fire" and "the market was shut" are
different answers. Scheduler liveness counts *sessions* between the last
recorded decision bar and the last expected session, and a date the table
calls a closure is reported as a closure and is never counted as missed. A
date the table cannot answer for raises: a calendar that guesses about a year
it does not hold would be silently wrong exactly when nobody is watching.

**No wall clock lives in this module.** ``now`` and ``as_of`` are required
keyword arguments on every entry point and nothing here calls
``datetime.now`` or ``date.today``. That is not fastidiousness: §3.11 requires
a replayed session to be evaluated against its *recorded* decision bar, and
the wall-clock read in ``daily.warn_if_stale_panel`` is exactly the one a
replay must not inherit, or every historical session fails the check for being
historical. A caller in operation passes the wall clock; a caller in replay
passes the recorded bar; the checks cannot tell them apart and do not need to.

**Policy lives in ``live/authorization.py``, not here.** This module produces
*verdicts* — which sources were late, which sids are uncovered, which classes
are stale — and converts them into the vocabulary §3.10 already speaks:
:class:`~prism.live.authorization.OperatingState` rows and per-sid
:class:`~prism.live.authorization.InputFreshness`. What those imply for what
may be sent is computed once, in
:func:`~prism.live.authorization.permitted_classes`, and is not restated here.
That is what makes "missing alpha data blocks new exposure while cancellation
and reduction stay available" a consequence of one rule rather than of two
rules that have to be kept in agreement.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from prism.io import sessions
from prism.live.authorization import ActionContext, InputFreshness, OperatingState
from prism.live.broker import Broker, BrokerCapabilityUnsupported

logger = logging.getLogger(__name__)

__all__ = [
    "INPUT_CLASSES",
    "SESSION_TABLE_WARNING_DAYS",
    "BrokerStateFreshness",
    "ClockHealth",
    "DataFreshness",
    "FreshnessError",
    "FreshnessReport",
    "FreshnessThresholds",
    "PublicationDeadline",
    "SchedulerLiveness",
    "SessionTableHorizon",
    "SourceObservation",
    "SourceVerdict",
    "check_broker_state",
    "check_clock",
    "check_data_freshness",
    "check_scheduler",
    "halted_sids",
    "read_server_time",
    "session_table_horizon",
]

#: The four input classes of §3.10, checked separately and per sid. The names
#: are the field names of :class:`~prism.live.authorization.InputFreshness`,
#: which is what makes a verdict here assignable there without a translation
#: table that could drift.
INPUT_CLASSES: tuple[str, ...] = ("mark", "halt", "covariance", "signal")

#: §3.11: "The health check warns when the table ends within ninety days."
SESSION_TABLE_WARNING_DAYS = 90

_ET = ZoneInfo(sessions.SESSION_TZ)


class FreshnessError(ValueError):
    """A freshness check cannot be evaluated as asked, so it refuses to answer.

    Raised rather than defaulted. Every other outcome in this module is a
    verdict — "stale", "late", "uncovered" — which a caller may act on. This is
    the case where the *question* is malformed: a class nobody declared a
    deadline for, a naive timestamp, a threshold that will not parse. A control
    that cannot verify its precondition must refuse, and a check that silently
    answered "fresh" here would be the failure it exists to prevent.
    """


def _require_aware(value: dt.datetime, *, field: str) -> dt.datetime:
    """Refuse a naive timestamp: an age computed across two unknown zones is not an age."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise FreshnessError(f"{field} carries no UTC offset; a freshness age needs an unambiguous instant")
    return value


def _as_date(value: str | dt.date | dt.datetime, *, field: str) -> dt.date:
    """A calendar date from a bar string, a date, or a datetime."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise FreshnessError(f"{field} {value!r} is not an ISO date") from exc


# ---------------------------------------------------------------------------
# Declared thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreshnessThresholds:
    """The ``[freshness]`` block of the execution profile (§3.9), parsed.

    The defaults are the profile's declared values, restated here so that a
    caller holding no profile still runs against a declared number rather than
    against a number invented at the call site.
    """

    max_broker_state_age_seconds: float = 900.0
    max_clock_skew_seconds: float = 30.0
    required_coverage: float = 1.0

    @classmethod
    def from_profile(cls, execution_profile: Any) -> "FreshnessThresholds":
        """Read ``[freshness]`` off a parsed profile or a raw mapping.

        A key the file does not declare keeps this class's default; a key it
        declares at an unusable value raises, because a threshold that cannot
        be read is not a threshold.
        """
        if execution_profile is None:
            block: Mapping[str, Any] = {}
        elif isinstance(execution_profile, Mapping):
            block = dict(execution_profile.get("freshness") or {})
        else:
            value = getattr(execution_profile, "freshness", None)
            block = dict(value) if isinstance(value, Mapping) else {}
        defaults = cls()
        out: dict[str, float] = {}
        for name, default in (
            ("max_broker_state_age_seconds", defaults.max_broker_state_age_seconds),
            ("max_clock_skew_seconds", defaults.max_clock_skew_seconds),
            ("required_coverage", defaults.required_coverage),
        ):
            raw = block.get(name, default)
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise FreshnessError(f"[freshness] {name} must be a number, got {raw!r}") from exc
            if number < 0.0:
                raise FreshnessError(f"[freshness] {name} must not be negative, got {number}")
            out[name] = number
        if out["required_coverage"] > 1.0:
            raise FreshnessError(
                f"[freshness] required_coverage is a fraction in [0, 1], got {out['required_coverage']}"
            )
        return cls(**out)


@dataclass(frozen=True)
class PublicationDeadline:
    """When one source's data for one input class is due (§3.11).

    The deadline is a wall-clock time in the exchange's zone on a session
    relative to the decision bar, not an elapsed age: "the close panel is due
    by 20:00 ET on the bar" is a statement a vendor can be held to, while "no
    older than four hours" is a statement about when somebody happened to look.
    ``sessions_after`` is how a T+1 source is declared — a fundamentals feed due
    the morning after the bar sets it to 1 — and it counts *sessions*, so a
    Friday bar's T+1 deadline lands on Monday rather than on Saturday.
    """

    source: str
    input_class: str
    at: dt.time
    sessions_after: int = 0
    required_coverage: float = 1.0

    def __post_init__(self) -> None:
        if self.input_class not in INPUT_CLASSES:
            raise FreshnessError(
                f"{self.source!r} declares input class {self.input_class!r}; the four classes of §3.10 are "
                f"{list(INPUT_CLASSES)}"
            )
        if self.sessions_after < 0:
            raise FreshnessError(
                f"{self.source!r} declares sessions_after {self.sessions_after}; a deadline is not retroactive"
            )
        if not 0.0 <= self.required_coverage <= 1.0:
            raise FreshnessError(
                f"{self.source!r} declares required_coverage {self.required_coverage}; it is a fraction in [0, 1]"
            )

    def deadline(self, decision_bar: str | dt.date) -> dt.datetime:
        """The instant this source is late after, for ``decision_bar``.

        Raises through :mod:`prism.io.sessions` when the bar, or the session
        ``sessions_after`` steps to, falls outside the bundled table.
        """
        day = _as_date(decision_bar, field="decision_bar")
        for _ in range(self.sessions_after):
            day = sessions.next_session(day)
        return dt.datetime.combine(day, self.at, tzinfo=_ET)


# ---------------------------------------------------------------------------
# Check 1: data freshness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceObservation:
    """What a source actually delivered for one decision bar.

    ``published_at`` is when the data became available to this process, not
    when the vendor stamped it; ``covered_sids`` is the securities it carries.
    An observation the caller does not supply at all is not the same fact as
    one reporting zero coverage, and both are stale — the difference is only in
    the diagnosis the verdict carries.
    """

    source: str
    published_at: dt.datetime | None = None
    covered_sids: frozenset[str] = frozenset()

    @classmethod
    def of(cls, source: str, sids: Iterable[str], *, published_at: dt.datetime | None) -> "SourceObservation":
        """An observation from any iterable of sids."""
        return cls(source=source, published_at=published_at, covered_sids=frozenset(str(s) for s in sids))


@dataclass(frozen=True)
class SourceVerdict:
    """One source's answer for one input class on one decision bar."""

    source: str
    input_class: str
    deadline: dt.datetime
    published_at: dt.datetime | None
    coverage: float
    required_coverage: float
    missing_sids: tuple[str, ...]
    now: dt.datetime

    @property
    def published(self) -> bool:
        """Whether anything arrived from this source for this bar."""
        return self.published_at is not None

    @property
    def late(self) -> bool:
        """Published after its deadline, or unpublished with the deadline already past."""
        if self.published_at is None:
            return self.now > self.deadline
        return self.published_at > self.deadline

    @property
    def pending(self) -> bool:
        """Unpublished, but not yet due — the run is early rather than the vendor late."""
        return self.published_at is None and not self.late

    @property
    def shortfall(self) -> bool:
        """Published on time, but for too few of the eligible securities."""
        return self.coverage < self.required_coverage

    @property
    def ok(self) -> bool:
        return self.published and not self.late and not self.shortfall

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "input_class": self.input_class,
            "deadline": self.deadline.isoformat(),
            "published_at": None if self.published_at is None else self.published_at.isoformat(),
            "coverage": self.coverage,
            "required_coverage": self.required_coverage,
            "missing_sids": list(self.missing_sids),
            "late": self.late,
            "pending": self.pending,
            "shortfall": self.shortfall,
            "ok": self.ok,
        }

    def detail(self) -> str:
        """One line naming the fault, or the pass."""
        head = f"{self.source} [{self.input_class}]"
        if self.pending:
            return f"{head}: not published; due {self.deadline.isoformat()}, which is not yet past"
        if not self.published:
            return f"{head}: not published and past its {self.deadline.isoformat()} deadline"
        if self.late:
            return (
                f"{head}: published {self.published_at.isoformat() if self.published_at else '?'}, after its "
                f"{self.deadline.isoformat()} deadline"
            )
        if self.shortfall:
            shown = list(self.missing_sids[:8])
            return (
                f"{head}: covers {self.coverage:.1%} of eligible securities, below the declared "
                f"{self.required_coverage:.1%}; missing {shown}" + ("…" if len(self.missing_sids) > len(shown) else "")
            )
        stamp = self.published_at.isoformat() if self.published_at else "?"
        return f"{head}: on time at {stamp}, {self.coverage:.1%} coverage"


@dataclass(frozen=True)
class DataFreshness:
    """Check 1 (§3.11): every declared source against its deadline and its coverage.

    The per-class answer is a *union* over the sources declared for that class:
    two vendors of marking prices are complementary, and a sid one of them
    published on time is a sid with a fresh mark whatever the other did. The
    per-source shortfall, by contrast, stays per source, because a source that
    covers half the book has failed its own declaration even when another
    source covers the rest.
    """

    decision_bar: str
    eligible_sids: tuple[str, ...]
    classes: tuple[str, ...]
    sources: tuple[SourceVerdict, ...]
    stale_by_class: Mapping[str, tuple[str, ...]]
    halted: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Every declared source on time and at coverage, and nothing halted."""
        return all(verdict.ok for verdict in self.sources) and not self.halted

    @property
    def stale_sids(self) -> tuple[str, ...]:
        """Every sid stale in at least one checked class."""
        out: set[str] = set()
        for stale in self.stale_by_class.values():
            out.update(stale)
        return tuple(sorted(out))

    @property
    def states(self) -> frozenset[OperatingState]:
        """The rows this check imposes **account-wide**: none, by construction.

        S1 is a per-*sid* row. §3.10 defines it as "signal inputs fail freshness
        **for a sid**, while that sid's mark and halt status are fresh", and
        §3.11 says in as many words that a coverage shortfall blocks classes per
        sid, not the whole account. Returning S1 here would intersect it into
        every name's permitted set, so one vendor missing one symbol would stop
        the other four hundred and ninety-nine from trading — which is the
        behaviour those two sentences exist to forbid.

        The row is not lost: it is carried by :meth:`freshness_by_sid`, and
        :func:`~prism.live.authorization.permitted_classes` reconstructs exactly
        S1's set from ``InputFreshness(signal=False, mark=True, halt=True)`` —
        C0, C1 and OB, with C4 surviving only on a fresh covariance. See
        :meth:`states_for` for the same answer stated per sid, which is what a
        report shows an operator.
        """
        return frozenset()

    def states_for(self, sid: str) -> frozenset[OperatingState]:
        """S1 for one security, when its signal failed while its mark and halt held."""
        stale = {name: sid in set(self.stale_by_class.get(name, ())) for name in INPUT_CLASSES}
        halted = sid in set(self.halted)
        if stale["signal"] and not stale["mark"] and not stale["halt"] and not halted:
            return frozenset({OperatingState.S1_STALE_ALPHA})
        return frozenset()

    @property
    def affected_states(self) -> frozenset[OperatingState]:
        """Every row some eligible security is in — the reporting view, not the gating one."""
        out: set[OperatingState] = set()
        for sid in self.eligible_sids:
            out |= self.states_for(sid)
        return frozenset(out)

    def freshness_by_sid(self) -> dict[str, InputFreshness]:
        """The per-sid, per-class verdict in the vocabulary §3.10 consumes.

        A class this report did not check keeps :class:`InputFreshness`'s
        default, which is fresh: the report makes no claim about a class the
        caller did not name, and naming fewer than four classes is the caller
        asserting the others are established elsewhere.

        A security under an open halt is carried with a **stale mark**. That is
        not an extra policy invented here: a halted name has no trade to mark
        against, so its last close is not a current price, which is §3.6's
        stale-valuation rule. The consequence then falls out of
        :func:`permitted_classes` unchanged — a cancel and a reduction all the
        way to flat survive it, a partial reduction and new exposure do not.
        """
        halted = set(self.halted)
        out: dict[str, InputFreshness] = {}
        for sid in self.eligible_sids:
            fields = {name: True for name in INPUT_CLASSES}
            for input_class in self.classes:
                fields[input_class] = sid not in set(self.stale_by_class.get(input_class, ()))
            if sid in halted:
                fields["mark"] = False
            out[sid] = InputFreshness(**fields)
        return out

    def detail(self) -> str:
        faults = [verdict.detail() for verdict in self.sources if not verdict.ok]
        if self.halted:
            faults.append(f"halted at {self.decision_bar}: {list(self.halted[:8])}")
        if faults:
            return "; ".join(faults)
        return (
            f"{len(self.sources)} source(s) on time for {len(self.eligible_sids)} eligible security(ies) "
            f"across {list(self.classes)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": "data_freshness",
            "decision_bar": self.decision_bar,
            "eligible": len(self.eligible_sids),
            "classes": list(self.classes),
            "sources": [verdict.to_dict() for verdict in self.sources],
            "stale_by_class": {name: list(sids) for name, sids in self.stale_by_class.items()},
            "halted": list(self.halted),
            "ok": self.ok,
            "states": sorted(state.value for state in self.affected_states),
        }


def check_data_freshness(
    *,
    decision_bar: str | dt.date,
    eligible_sids: Iterable[str],
    deadlines: Sequence[PublicationDeadline],
    observations: Iterable[SourceObservation] = (),
    now: dt.datetime,
    classes: Sequence[str] = INPUT_CLASSES,
    required_coverage: float | None = None,
    halted: Iterable[str] = (),
) -> DataFreshness:
    """Check 1: declared publication deadlines per source, with required coverage.

    Every class in ``classes`` must have at least one declared deadline. A class
    named with no source behind it raises rather than passing: "nobody told me
    when this was due" and "it arrived on time" are different facts, and only
    one of them may authorize an order.

    ``required_coverage`` overrides each deadline's own declaration when given,
    which is how the execution profile's ``[freshness]`` block applies one
    number across sources that declare none of their own.
    """
    _require_aware(now, field="now")
    bar = _as_date(decision_bar, field="decision_bar")
    eligible = tuple(sorted({str(sid) for sid in eligible_sids}))
    named = tuple(classes)
    unknown = [name for name in named if name not in INPUT_CLASSES]
    if unknown:
        raise FreshnessError(f"unknown input class(es) {unknown}; the four of §3.10 are {list(INPUT_CLASSES)}")
    undeclared = [name for name in named if not any(d.input_class == name for d in deadlines)]
    if undeclared:
        raise FreshnessError(
            f"input class(es) {undeclared} were named for checking but no source declares a publication "
            "deadline for them; a class with no declared deadline cannot be verified, and is refused (§3.11)"
        )

    by_source = {observation.source: observation for observation in observations}
    verdicts: list[SourceVerdict] = []
    covered_by_class: dict[str, set[str]] = {name: set() for name in named}
    for declared in deadlines:
        if declared.input_class not in covered_by_class:
            continue
        observation = by_source.get(declared.source)
        covered = frozenset() if observation is None else observation.covered_sids
        published_at = None if observation is None else observation.published_at
        if published_at is not None:
            published_at = _require_aware(published_at, field=f"{declared.source} published_at")
        present = {sid for sid in eligible if sid in covered}
        coverage = 1.0 if not eligible else len(present) / len(eligible)
        required = declared.required_coverage if required_coverage is None else float(required_coverage)
        verdict = SourceVerdict(
            source=declared.source,
            input_class=declared.input_class,
            deadline=declared.deadline(bar),
            published_at=published_at,
            coverage=coverage,
            required_coverage=required,
            missing_sids=tuple(sid for sid in eligible if sid not in present),
            now=now,
        )
        verdicts.append(verdict)
        if verdict.published and not verdict.late:
            covered_by_class[declared.input_class].update(present)

    stale_by_class = {name: tuple(sid for sid in eligible if sid not in covered_by_class[name]) for name in named}
    report = DataFreshness(
        decision_bar=bar.isoformat(),
        eligible_sids=eligible,
        classes=named,
        sources=tuple(verdicts),
        stale_by_class=stale_by_class,
        halted=tuple(sorted({str(sid) for sid in halted})),
    )
    if not report.ok:
        logger.warning("data freshness at %s: %s", bar.isoformat(), report.detail())
    return report


def halted_sids(events: Any, sids: Iterable[str], *, as_of: str | dt.date) -> tuple[str, ...]:
    """Securities under an open halt at ``as_of``, read from the §3.2 events table.

    The session table deliberately does not describe intraday halts — those are
    events, and they reach freshness here (§3.11). A halt row is open when its
    ``start`` is at or before ``as_of`` and its ``end`` is null or not before
    it; a row that has ended is history and is not reported.

    ``events`` is a ``pandas`` frame of the events table or any iterable of row
    mappings, so a caller holding a ``DatasetView.events(...)`` frame and a
    caller holding recorded rows use the same function.
    """
    from prism.io.events import decode_payload

    day = _as_date(as_of, field="as_of").isoformat()
    wanted = {str(sid) for sid in sids}
    open_halts: set[str] = set()
    for row in _rows(events):
        if str(row.get("event_type", "")) != "halt":
            continue
        sid = str(row.get("sid", ""))
        if sid not in wanted:
            continue
        payload = decode_payload(row.get("payload"))
        start = str(payload.get("start") or row.get("effective_date") or "")[:10]
        end_raw = payload.get("end")
        end = None if end_raw in (None, "") else str(end_raw)[:10]
        if start and start > day:
            continue
        if end is not None and end < day:
            continue
        open_halts.add(sid)
    return tuple(sorted(open_halts))


def _rows(table: Any) -> list[Mapping[str, Any]]:
    """Row mappings from a frame or from an iterable of mappings."""
    if table is None:
        return []
    to_dict = getattr(table, "to_dict", None)
    if callable(to_dict) and hasattr(table, "columns"):
        return list(to_dict(orient="records"))
    return list(table)


# ---------------------------------------------------------------------------
# Check 2: broker-state freshness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrokerStateFreshness:
    """Check 2 (§3.11): how long ago positions and cash were read.

    Separate from check 1 because it is a different fault with a different
    consequence: S2 permits C0 only, and only after a fresh open-order read.
    A book whose prices are current and whose *positions* are an hour old is a
    book that can compute a beautifully sized order against holdings it no
    longer has, and no maximum over data timestamps ever notices.
    """

    refreshed_at: dt.datetime | None
    now: dt.datetime
    max_age_seconds: float
    open_orders_read_fresh: bool = False

    @property
    def age_seconds(self) -> float | None:
        """Seconds since the last broker read, or ``None`` when there has been none."""
        if self.refreshed_at is None:
            return None
        return (self.now - self.refreshed_at).total_seconds()

    @property
    def ok(self) -> bool:
        """An unknown age is stale, never fresh: unknown freshness is the fail-safe direction."""
        age = self.age_seconds
        return age is not None and age <= self.max_age_seconds

    @property
    def states(self) -> frozenset[OperatingState]:
        return frozenset() if self.ok else frozenset({OperatingState.S2_STALE_BROKER_STATE})

    def detail(self) -> str:
        if self.refreshed_at is None:
            return "broker state has never been read here; an unknown age is stale, not fresh (S2)"
        age = self.age_seconds or 0.0
        verdict = "within" if self.ok else "beyond"
        return (
            f"broker state read {self.refreshed_at.isoformat()}, {age:.0f}s ago, {verdict} the declared "
            f"max_broker_state_age_seconds {self.max_age_seconds:.0f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": "broker_state_freshness",
            "refreshed_at": None if self.refreshed_at is None else self.refreshed_at.isoformat(),
            "age_seconds": self.age_seconds,
            "max_age_seconds": self.max_age_seconds,
            "open_orders_read_fresh": self.open_orders_read_fresh,
            "ok": self.ok,
            "states": sorted(state.value for state in self.states),
        }


def check_broker_state(
    *,
    refreshed_at: dt.datetime | None,
    now: dt.datetime,
    max_age_seconds: float,
    open_orders_read_fresh: bool = False,
) -> BrokerStateFreshness:
    """Check 2: the age of the last positions-and-cash read against its declared bound."""
    _require_aware(now, field="now")
    if refreshed_at is not None:
        _require_aware(refreshed_at, field="refreshed_at")
    if max_age_seconds < 0.0:
        raise FreshnessError(f"max_broker_state_age_seconds must not be negative, got {max_age_seconds}")
    verdict = BrokerStateFreshness(
        refreshed_at=refreshed_at,
        now=now,
        max_age_seconds=float(max_age_seconds),
        open_orders_read_fresh=open_orders_read_fresh,
    )
    if not verdict.ok:
        logger.warning("broker-state freshness: %s", verdict.detail())
    return verdict


# ---------------------------------------------------------------------------
# Check 3: clock health
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockHealth:
    """Check 3 (§3.11): this host's clock against the venue's server time.

    It is separate because it is the check the other three *depend on*. Ages in
    checks 1 and 2 are differences against the host clock, so a host twenty
    minutes fast reports twenty minutes of extra age everywhere, and a host
    twenty minutes slow reports twenty minutes less — uniformly, consistently,
    and wrongly. No server time available is a fault (S8), not a pass: an
    unverifiable clock is an unverified clock.
    """

    server_time: dt.datetime | None
    host_time: dt.datetime
    max_skew_seconds: float
    unavailable_reason: str | None = None

    @property
    def skew_seconds(self) -> float | None:
        """Host clock minus venue clock, in seconds; positive means the host is ahead."""
        if self.server_time is None:
            return None
        return (self.host_time - self.server_time).total_seconds()

    @property
    def ok(self) -> bool:
        skew = self.skew_seconds
        return skew is not None and abs(skew) <= self.max_skew_seconds

    @property
    def states(self) -> frozenset[OperatingState]:
        return frozenset() if self.ok else frozenset({OperatingState.S8_CLOCK_FAULT})

    def detail(self) -> str:
        if self.server_time is None:
            reason = self.unavailable_reason or "the adapter reported none"
            return f"no venue server time available ({reason}); an unverifiable clock is a fault (S8)"
        skew = self.skew_seconds or 0.0
        verdict = "within" if self.ok else "beyond"
        return (
            f"host clock {skew:+.1f}s against venue time {self.server_time.isoformat()}, {verdict} the declared "
            f"max_clock_skew_seconds {self.max_skew_seconds:.0f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": "clock_health",
            "server_time": None if self.server_time is None else self.server_time.isoformat(),
            "host_time": self.host_time.isoformat(),
            "skew_seconds": self.skew_seconds,
            "max_skew_seconds": self.max_skew_seconds,
            "unavailable_reason": self.unavailable_reason,
            "ok": self.ok,
            "states": sorted(state.value for state in self.states),
        }


def read_server_time(broker: Broker) -> tuple[dt.datetime | None, str | None]:
    """``(server_time, why_it_is_missing)`` from an adapter that may not implement it.

    An adapter without the capability, a transport that failed, and a value that
    will not parse are all "no server time", which is state S8. They are
    distinguished only in the reason string, because the consequence is one.
    """
    try:
        raw = broker.server_time()
    except BrokerCapabilityUnsupported as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - any transport failure is "no server time", not a crash
        return None, f"{type(exc).__name__}: {exc}"
    if raw is None:
        return None, "the adapter returned no server time"
    return _parse_server_time(str(raw))


def _parse_server_time(raw: str) -> tuple[dt.datetime | None, str | None]:
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None, f"unparseable server time {raw!r}"
    if parsed.tzinfo is None:
        return None, f"server time {raw!r} carries no UTC offset"
    return parsed, None


def check_clock(
    *,
    host_time: dt.datetime,
    max_skew_seconds: float,
    server_time: dt.datetime | str | None = None,
    broker: Broker | None = None,
) -> ClockHealth:
    """Check 3: the host clock against ``Broker.server_time``.

    Pass ``server_time`` when it has already been read (a replay holds the
    recorded one), or ``broker`` to read it here. Neither is not an error: it is
    the "no server time available" fault the row declares, and it is S8.
    """
    _require_aware(host_time, field="host_time")
    if max_skew_seconds < 0.0:
        raise FreshnessError(f"max_clock_skew_seconds must not be negative, got {max_skew_seconds}")
    resolved: dt.datetime | None = None
    reason: str | None = None
    if isinstance(server_time, str):
        resolved, reason = _parse_server_time(server_time)
    elif server_time is not None:
        resolved = _require_aware(server_time, field="server_time")
    elif broker is not None:
        resolved, reason = read_server_time(broker)
    else:
        reason = "no server time was read and no broker was supplied"
    verdict = ClockHealth(
        server_time=resolved,
        host_time=host_time,
        max_skew_seconds=float(max_skew_seconds),
        unavailable_reason=reason,
    )
    if not verdict.ok:
        logger.warning("clock health: %s", verdict.detail())
    return verdict


# ---------------------------------------------------------------------------
# Check 4: scheduler liveness — the holiday question
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerLiveness:
    """Check 4 (§3.11): did the job fire at all?

    The other three checks read timestamps. This one reads their *absence*,
    which is why it cannot be folded into them: a scheduler that never fired
    wrote no timestamp to be stale, and a maximum over an empty set is not a
    fault, it is an empty set.

    ``missed`` counts sessions, never weekdays. ``closures`` is the other half
    of the same answer and is reported rather than dropped, so an operator
    reading an alert sees why a four-day gap was three missed sessions — or why
    a Monday with no run was Labor Day and no fault at all.
    """

    as_of: dt.date
    last_decision_bar: dt.date | None
    horizon_end: dt.date | None
    missed: tuple[dt.date, ...] = ()
    closures: tuple[tuple[dt.date, str], ...] = ()
    source: str = ""

    @property
    def ok(self) -> bool:
        """No expected session went unrun. An unknown last bar is not ok — nothing ran here."""
        return self.last_decision_bar is not None and not self.missed

    @property
    def known(self) -> bool:
        """Whether a last decision bar was found at all."""
        return self.last_decision_bar is not None

    @property
    def states(self) -> frozenset[OperatingState]:
        """S9 when a session was expected and did not run.

        S9 is a *monitor* row: it authorizes no broker action and removes none.
        The next engine invocation is governed by its own data and broker-state
        freshness, which is why :attr:`FreshnessReport.states` — the set an
        engine invocation passes to the authorization gate — does not carry it.
        """
        return frozenset({OperatingState.S9_SCHEDULER_FAULT}) if self.missed else frozenset()

    def closure_detail(self) -> str:
        return ", ".join(f"{day.isoformat()} {name}" for day, name in self.closures)

    def detail(self) -> str:
        if self.last_decision_bar is None:
            where = f" in {self.source}" if self.source else ""
            return f"no decision bar recorded{where}; nothing has run here"
        closed = self.closure_detail()
        tail = f"; closures in the gap (not missed runs): {closed}" if closed else ""
        horizon = self.horizon_end.isoformat() if self.horizon_end else "?"
        if not self.missed:
            return f"last decision bar {self.last_decision_bar.isoformat()}, last expected session {horizon}{tail}"
        shown = [day.isoformat() for day in self.missed[:8]]
        return (
            f"{len(self.missed)} expected session(s) never ran between {self.last_decision_bar.isoformat()} and "
            f"{horizon}: {shown}" + ("…" if len(self.missed) > len(shown) else "") + tail
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": "scheduler_liveness",
            "as_of": self.as_of.isoformat(),
            "last_decision_bar": None if self.last_decision_bar is None else self.last_decision_bar.isoformat(),
            "horizon_end": None if self.horizon_end is None else self.horizon_end.isoformat(),
            "missed": [day.isoformat() for day in self.missed],
            "closures": [[day.isoformat(), name] for day, name in self.closures],
            "source": self.source,
            "ok": self.ok,
            "states": sorted(state.value for state in self.states),
        }


def check_scheduler(
    *,
    last_decision_bar: str | dt.date | None,
    as_of: str | dt.date,
    source: str = "",
) -> SchedulerLiveness:
    """Check 4: the last decision bar against the last expected session in the table.

    A session is expected only once its close is past, so the horizon ends at
    the last session strictly *before* ``as_of``: at any hour of ``as_of``
    itself the close-of-day decision does not yet exist, and counting it would
    make every check run during the trading day report a missed session.

    A holiday is not a missed session. The sessions between the last recorded
    bar and the horizon come from :mod:`prism.io.sessions`, so a closure in the
    gap is reported as a closure and never counted. A date the table does not
    cover raises :class:`~prism.io.sessions.SessionTableRangeError` rather than
    falling back to weekday arithmetic, which is wrong about every holiday.
    """
    today = _as_date(as_of, field="as_of")
    if last_decision_bar is None:
        return SchedulerLiveness(as_of=today, last_decision_bar=None, horizon_end=None, source=source)
    last = _as_date(last_decision_bar, field="last_decision_bar")
    horizon_end = sessions.previous_session(today)
    closures = tuple(
        (day, sessions.session_row(day).name)
        for day in _weekdays(last + dt.timedelta(days=1), today)
        if not sessions.is_session(day)
    )
    if last >= horizon_end:
        return SchedulerLiveness(
            as_of=today,
            last_decision_bar=last,
            horizon_end=horizon_end,
            closures=closures,
            source=source,
        )
    missed = tuple(sessions.sessions_between(sessions.next_session(last), horizon_end))
    verdict = SchedulerLiveness(
        as_of=today,
        last_decision_bar=last,
        horizon_end=horizon_end,
        missed=missed,
        closures=closures,
        source=source,
    )
    if verdict.missed:
        logger.error("scheduler liveness: %s", verdict.detail())
    return verdict


def _weekdays(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    """Every weekday in ``[start, end]``; weekends carry no calendar row at all."""
    day = start
    while day <= end:
        if day.weekday() < sessions.SATURDAY:
            yield day
        day += dt.timedelta(days=1)


# ---------------------------------------------------------------------------
# The session table's own horizon
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionTableHorizon:
    """How much calendar the bundled table has left (§3.11).

    The table is the authority every session question is answered from, so its
    running out is a fault of the *engine*, not of the book: past the coverage
    end no missed-run check can be evaluated at all, because the table refuses
    rather than guessing. Ninety days is the declared warning distance.
    """

    as_of: dt.date
    coverage_end: dt.date
    version: str
    warning_days: int = SESSION_TABLE_WARNING_DAYS

    @property
    def days_remaining(self) -> int:
        return (self.coverage_end - self.as_of).days

    @property
    def exhausted(self) -> bool:
        """The table can no longer answer for today."""
        return self.days_remaining < 0

    @property
    def warn(self) -> bool:
        return not self.exhausted and self.days_remaining <= self.warning_days

    @property
    def ok(self) -> bool:
        return not self.exhausted and not self.warn

    def detail(self) -> str:
        if self.exhausted:
            return (
                f"the bundled session table (version {self.version}) ended {self.coverage_end.isoformat()}, before "
                f"{self.as_of.isoformat()}; no session question can be answered — extend prism/io/sessions.py"
            )
        if self.warn:
            return (
                f"the bundled session table (version {self.version}) ends {self.coverage_end.isoformat()}, in "
                f"{self.days_remaining} day(s), within {self.warning_days}; extend prism/io/sessions.py"
            )
        return (
            f"the bundled session table (version {self.version}) covers through {self.coverage_end.isoformat()}, "
            f"{self.days_remaining} day(s) out"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": "session_table",
            "as_of": self.as_of.isoformat(),
            "coverage_end": self.coverage_end.isoformat(),
            "version": self.version,
            "days_remaining": self.days_remaining,
            "warn": self.warn,
            "exhausted": self.exhausted,
        }


def session_table_horizon(
    *, as_of: str | dt.date, warning_days: int = SESSION_TABLE_WARNING_DAYS
) -> SessionTableHorizon:
    """The bundled table's remaining coverage at ``as_of``."""
    return SessionTableHorizon(
        as_of=_as_date(as_of, field="as_of"),
        coverage_end=sessions.coverage_end(),
        version=sessions.TABLE_VERSION,
        warning_days=int(warning_days),
    )


# ---------------------------------------------------------------------------
# The four verdicts together
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreshnessReport:
    """The four checks as one record, and their translation into §3.10's vocabulary.

    The four stay separately addressable — ``report.data``, ``report.broker_state``,
    ``report.clock``, ``report.scheduler`` — because an operator reading an alert
    needs to know *which* of the four fired, and because each carries its own
    threshold and its own consequence. What this class adds is only the
    translation, done once: the operating states an engine invocation passes to
    the authorization gate, the per-sid input freshness it passes alongside, and
    the two ``ActionContext`` conditions §3.10's "only when" qualifiers read.
    """

    data: DataFreshness | None = None
    broker_state: BrokerStateFreshness | None = None
    clock: ClockHealth | None = None
    scheduler: SchedulerLiveness | None = None
    session_table: SessionTableHorizon | None = None

    @property
    def checks(self) -> tuple[Any, ...]:
        """The checks that were actually run, in the order §3.11 tabulates them."""
        return tuple(c for c in (self.data, self.broker_state, self.clock, self.scheduler) if c is not None)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def states(self) -> frozenset[OperatingState]:
        """The rows an *engine* invocation passes to the authorization gate.

        Two rows are deliberately not among them. S9, because it is a monitor
        row: it authorizes no broker action and removes none, and §3.10 says the
        next engine invocation is governed by its own data and broker-state
        freshness — an engine that inherited S9 would refuse to trade because a
        *previous* session had not, which is the opposite of recovery. And S1,
        because it is per sid: it reaches the gate through
        :meth:`freshness_by_sid`, which is the only way "a coverage shortfall
        blocks classes per sid, not the whole account" can be true.
        """
        out: set[OperatingState] = set()
        for check in (self.data, self.broker_state, self.clock):
            if check is not None:
                out |= check.states
        return frozenset(out)

    @property
    def monitor_states(self) -> frozenset[OperatingState]:
        """Every row the four checks establish — what an operator is shown.

        Wider than :attr:`states` in two directions, and neither belongs in the
        set handed to the authorization gate: S1, which is per sid and reaches
        the gate through :meth:`freshness_by_sid`, and S9, which is a monitor
        row that authorizes and removes nothing.
        """
        out = set(self.states)
        if self.data is not None:
            out |= self.data.affected_states
        if self.scheduler is not None:
            out |= self.scheduler.states
        return frozenset(out)

    def freshness_by_sid(self) -> dict[str, InputFreshness]:
        """Per-sid input freshness for ``AuthorizationGate.authorize_batch``."""
        return {} if self.data is None else self.data.freshness_by_sid()

    def action_context(self, base: ActionContext = ActionContext()) -> ActionContext:
        """``base`` with the two conditions these checks establish filled in.

        ``open_orders_read_fresh`` is S2's "only after a fresh open-order read".
        ``risk_inputs_fresh`` is S1's "C4 only when the risk snapshot's declared
        inputs are fresh", which is exactly "no eligible security's covariance
        class is stale". Nothing else is inferred: a condition nobody
        established keeps ``base``'s value, which defaults to false.
        """
        updates: dict[str, Any] = {}
        if self.broker_state is not None:
            updates["open_orders_read_fresh"] = self.broker_state.open_orders_read_fresh
        if self.data is not None and "covariance" in self.data.classes:
            updates["risk_inputs_fresh"] = not self.data.stale_by_class.get("covariance")
        return replace(base, **updates) if updates else base

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "states": sorted(state.value for state in self.states),
            "monitor_states": sorted(state.value for state in self.monitor_states),
            "checks": [check.to_dict() for check in self.checks],
            "session_table": None if self.session_table is None else self.session_table.to_dict(),
        }
