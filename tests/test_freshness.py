"""§3.11 freshness: four separate checks, and the holiday that is not a missed run.

Two claims are under attack in this file, and they are the two the
specification is explicit about.

**One timestamp is not four checks.** ``test_each_check_fires_alone`` builds
four scenarios in which exactly one of data freshness, broker-state freshness,
clock health and scheduler liveness fails while the other three pass. A single
"newest thing I have" age is satisfied in three of the four and would report
health; each scenario names what that age would have missed and what state the
separate check reaches instead.

**A holiday is not a missed run.** The live paper book's run directory
(``runs/paper_loop_momentum2``) last completed 2026-08-14 and the loop has been
dark since. The fifteen sessions from 2026-08-17 to 2026-09-04 are missed runs
and are written out literally below rather than derived, so a bug in the
session table cannot agree with a bug in the check. 2026-09-07 is Labor Day: it
is reported as a closure and is never counted. Both directions are tested, and
so is the weekday count that gets it wrong — the point of carrying a session
table is that ``previous weekday`` and ``previous session`` are different
functions.

No test here writes to the live run directory. The one test that reads it at
all reads two ledger lines and skips when the directory is absent, so the file
runs on a fresh clone.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from prism.io import sessions
from prism.live.authorization import (
    LIVE_BASE_URL,
    ActionClass,
    ActionContext,
    AuthorizationGate,
    AuthorizationRefused,
    InputFreshness,
    OperatingState,
    ProcessIdentity,
    Scope,
    parse_authorization,
    permitted_classes,
)
from prism.live.broker import Broker, BrokerCapabilityUnsupported, Fill, Order
from prism.live.freshness import (
    INPUT_CLASSES,
    BrokerStateFreshness,
    DataFreshness,
    FreshnessError,
    FreshnessReport,
    FreshnessThresholds,
    PublicationDeadline,
    SourceObservation,
    check_broker_state,
    check_clock,
    check_data_freshness,
    check_scheduler,
    halted_sids,
    read_server_time,
    session_table_horizon,
)
from prism.live.loop import LiveLoopContext, decide_and_submit
from prism.live.state import StateStore

UTC = dt.timezone.utc

#: A frozen historical scenario, not live state: the paper book's last completed
#: session before the 2026-08-15 .. 2026-09-25 scheduler outage. The synthetic
#: tests below replay that outage; nothing here may be read from the live run
#: directory, which moves every session.
LAST_COMPLETED = "2026-08-14"

#: The fifteen sessions the exchange held and the loop did not decide, written
#: out rather than derived. 2026-09-07 is deliberately absent: it is Labor Day.
MISSED = (
    "2026-08-17",
    "2026-08-18",
    "2026-08-19",
    "2026-08-20",
    "2026-08-21",
    "2026-08-24",
    "2026-08-25",
    "2026-08-26",
    "2026-08-27",
    "2026-08-28",
    "2026-08-31",
    "2026-09-01",
    "2026-09-02",
    "2026-09-03",
    "2026-09-04",
)

LABOR_DAY = dt.date(2026, 9, 7)

BAR = "2026-09-04"
NOW = dt.datetime(2026, 9, 4, 21, 30, tzinfo=UTC)  # 17:30 ET, after the close
BOOK = ("AAPL", "MSFT", "NVDA", "SPY")

ACCOUNT_ID = "PA3RTY0001"
CONFIG_ID = "cfg-" + "0" * 8
HOST = "trader-1"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def marks(at: dt.time = dt.time(20, 0), **kwargs: object) -> PublicationDeadline:
    return PublicationDeadline(source="bars", input_class="mark", at=at, **kwargs)  # type: ignore[arg-type]


def signals(at: dt.time = dt.time(20, 0), **kwargs: object) -> PublicationDeadline:
    return PublicationDeadline(source="scores", input_class="signal", at=at, **kwargs)  # type: ignore[arg-type]


def on_time(source: str, sids=BOOK, *, when: dt.datetime | None = None) -> SourceObservation:
    """A source that published every eligible security, before its deadline."""
    return SourceObservation.of(source, sids, published_at=when or dt.datetime(2026, 9, 4, 20, 5, tzinfo=UTC))


def healthy_data(**overrides) -> DataFreshness:
    kwargs = dict(
        decision_bar=BAR,
        eligible_sids=BOOK,
        deadlines=[marks(), signals()],
        observations=[on_time("bars"), on_time("scores")],
        now=NOW,
        classes=("mark", "signal"),
    )
    kwargs.update(overrides)
    return check_data_freshness(**kwargs)  # type: ignore[arg-type]


def make_gate(**scope_overrides: bool) -> AuthorizationGate:
    scope = {"add_exposure": True, "manage_existing": True, "hedge": False}
    scope.update(scope_overrides)
    record = parse_authorization(
        {
            "schema": 1,
            "mode": "live",
            "account_name": "reference",
            "account_id": ACCOUNT_ID,
            "endpoint": LIVE_BASE_URL,
            "execution_host": HOST,
            "config_id": CONFIG_ID,
            "engine_version": "0.4.0",
            "distribution_id": "git:" + "a" * 40,
            "policy_ids": {k: k for k in ("profile", "costs", "execution", "monitoring", "recovery")},
            "scope": scope,
            "submission_window_sessions": 1,
            "granted_by": "owner",
            "granted_at": NOW.isoformat(),
            "expires_at": (NOW + dt.timedelta(days=30)).isoformat(),
        }
    )
    identity = ProcessIdentity(
        mode="live",
        endpoint=LIVE_BASE_URL,
        account_id=ACCOUNT_ID,
        config_id=CONFIG_ID,
        engine_version="0.4.0",
        distribution_id="git:" + "a" * 40,
        execution_host=HOST,
    )
    return AuthorizationGate(record, identity)


class SilentBroker(Broker):
    """An adapter that implements nothing it is not asked to: ``server_time`` raises."""

    def positions(self) -> dict[str, float]:
        return {}

    def cash(self) -> float:
        return 0.0

    def submit(self, order: Order) -> None:  # pragma: no cover - never called here
        raise AssertionError("no order is submitted by a freshness check")

    def submitted_order_ids(self) -> set[str]:
        return set()

    def fills_for(self, client_order_ids: set[str]) -> list[Fill]:
        return []


class RecordingBroker(SilentBroker):
    """A venue that answers and records every order it was actually given.

    The counterpart to :class:`SilentBroker`: where a refusal is the claim, the
    empty ``submitted`` list is the evidence; where *availability* is the claim,
    the recorded order is.
    """

    def __init__(self, positions: dict[str, float] | None = None, cash: float = 90_000.0) -> None:
        self._positions = dict(positions or {})
        self._cash = cash
        self.submitted: list[Order] = []

    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def cash(self) -> float:
        return self._cash

    def submit(self, order: Order) -> None:
        self.submitted.append(order)

    def submitted_order_ids(self) -> set[str]:
        return {order.client_order_id for order in self.submitted}


class TimedBroker(SilentBroker):
    def __init__(self, stamp: str | None) -> None:
        self.stamp = stamp

    def server_time(self) -> str:
        if self.stamp is None:
            raise TimeoutError("the venue did not answer")
        return self.stamp


# ---------------------------------------------------------------------------
# The four checks are four
# ---------------------------------------------------------------------------


def test_each_check_fires_alone_so_one_timestamp_cannot_stand_for_four() -> None:
    """Four scenarios, one fault each, three healthy neighbours in every one.

    This is the specification's claim that a single maximum timestamp is
    insufficient, made falsifiable: in each row below the "newest thing I have"
    is fresh and the engine is nonetheless in a named fault state.
    """
    fresh_data = healthy_data()
    fresh_broker = check_broker_state(
        refreshed_at=NOW - dt.timedelta(seconds=60), now=NOW, max_age_seconds=900.0, open_orders_read_fresh=True
    )
    fresh_clock = check_clock(host_time=NOW, max_skew_seconds=30.0, server_time=NOW - dt.timedelta(seconds=2))
    fresh_scheduler = check_scheduler(last_decision_bar=BAR, as_of="2026-09-08")

    # Baseline: all four healthy, no state at all.
    assert FreshnessReport(fresh_data, fresh_broker, fresh_clock, fresh_scheduler).ok
    assert FreshnessReport(fresh_data, fresh_broker, fresh_clock, fresh_scheduler).monitor_states == frozenset()

    # 1. Data freshness alone: the vendor missed its deadline. Broker state,
    #    clock and scheduler are all current, so no age over them notices.
    late = healthy_data(observations=[on_time("bars"), SourceObservation.of("scores", BOOK, published_at=None)])
    only_data = FreshnessReport(late, fresh_broker, fresh_clock, fresh_scheduler)
    assert only_data.monitor_states == frozenset({OperatingState.S1_STALE_ALPHA})

    # 2. Broker-state freshness alone: prices current, positions an hour old.
    stale_book = check_broker_state(refreshed_at=NOW - dt.timedelta(hours=1), now=NOW, max_age_seconds=900.0)
    only_broker = FreshnessReport(fresh_data, stale_book, fresh_clock, fresh_scheduler)
    assert only_broker.monitor_states == frozenset({OperatingState.S2_STALE_BROKER_STATE})

    # 3. Clock health alone: every age above was computed against this clock, so
    #    a drifted host makes all of them agree with each other and none true.
    skewed = check_clock(host_time=NOW, max_skew_seconds=30.0, server_time=NOW - dt.timedelta(minutes=5))
    only_clock = FreshnessReport(fresh_data, fresh_broker, skewed, fresh_scheduler)
    assert only_clock.monitor_states == frozenset({OperatingState.S8_CLOCK_FAULT})

    # 4. Scheduler liveness alone: nothing is stale, because nothing was written.
    dark = check_scheduler(last_decision_bar=LAST_COMPLETED, as_of="2026-09-08")
    only_scheduler = FreshnessReport(fresh_data, fresh_broker, fresh_clock, dark)
    assert only_scheduler.monitor_states == frozenset({OperatingState.S9_SCHEDULER_FAULT})
    assert only_scheduler.states == frozenset(), "S9 is a monitor row and never governs an engine invocation"


def test_a_report_names_which_of_the_four_ran() -> None:
    report = FreshnessReport(scheduler=check_scheduler(last_decision_bar=BAR, as_of="2026-09-08"))
    payload = report.to_dict()
    assert [check["check"] for check in payload["checks"]] == ["scheduler_liveness"]
    assert report.ok and report.freshness_by_sid() == {}


# ---------------------------------------------------------------------------
# Check 1: data freshness, coverage, and the partial feed outage
# ---------------------------------------------------------------------------


def test_a_coverage_shortfall_blocks_new_exposure_and_leaves_cancellation_and_reduction() -> None:
    """The stage 5 exit condition, proved through the §3.10 action table itself.

    The signal feed publishes on time for two of four names. The two it covers
    keep every class; the two it does not lose C3 — new exposure — while C0,
    C1 and OB survive, because a cancel needs no signal and a reduction toward
    zero is not new exposure. The verdict is not restated here: it is computed
    by ``permitted_classes``, so the two modules cannot drift apart.
    """
    partial = check_data_freshness(
        decision_bar=BAR,
        eligible_sids=BOOK,
        deadlines=[marks(), signals(required_coverage=1.0)],
        observations=[on_time("bars"), on_time("scores", ("AAPL", "MSFT"))],
        now=NOW,
        classes=("mark", "signal"),
    )
    assert not partial.ok
    shortfall = next(v for v in partial.sources if v.source == "scores")
    assert shortfall.shortfall and shortfall.coverage == 0.5
    assert shortfall.missing_sids == ("NVDA", "SPY")

    # S1 is a per-sid row, and this is the sentence that makes it one: "a
    # coverage shortfall blocks classes per sid, not the whole account" (§3.10).
    # An account-wide S1 would stop AAPL trading because NVDA's vendor was late.
    assert partial.states == frozenset(), "no row is imposed account-wide by a coverage shortfall"
    assert partial.states_for("NVDA") == frozenset({OperatingState.S1_STALE_ALPHA})
    assert partial.states_for("AAPL") == frozenset()
    assert partial.affected_states == frozenset({OperatingState.S1_STALE_ALPHA})

    by_sid = partial.freshness_by_sid()
    covered = permitted_classes(partial.states, Scope(True, True, True), ActionContext(freshness=by_sid["AAPL"]))
    uncovered = permitted_classes(partial.states, Scope(True, True, True), ActionContext(freshness=by_sid["NVDA"]))
    assert by_sid["NVDA"] == InputFreshness(mark=True, halt=True, covariance=True, signal=False)

    assert ActionClass.C3 in covered
    assert ActionClass.C3 not in uncovered, "a name with no signal may not take on new exposure"
    for stays in (ActionClass.C0, ActionClass.C1, ActionClass.OB):
        assert stays in uncovered, f"{stays.value} must stay available on an uncovered name"


def test_the_batch_refusal_names_the_uncovered_security() -> None:
    """A coverage shortfall blocks classes per sid, and refuses the batch as a unit."""
    partial = check_data_freshness(
        decision_bar=BAR,
        eligible_sids=BOOK,
        deadlines=[marks(), signals()],
        observations=[on_time("bars"), on_time("scores", ("AAPL", "MSFT"))],
        now=NOW,
        classes=("mark", "signal"),
    )
    gate = make_gate()
    report = FreshnessReport(data=partial)

    # Reductions on the uncovered names are authorized; the report's own states
    # and per-sid freshness are what the gate is given.
    gate.authorize_batch(
        {"AAPL": ActionClass.C3, "NVDA": ActionClass.C1, "SPY": ActionClass.C0},
        states=report.states,
        context=report.action_context(ActionContext(reduction_to_flat=False)),
        freshness_by_sid=report.freshness_by_sid(),
    )
    with pytest.raises(AuthorizationRefused) as refused:
        gate.authorize_batch(
            {"AAPL": ActionClass.C3, "NVDA": ActionClass.C3},
            states=report.states,
            context=report.action_context(),
            freshness_by_sid=report.freshness_by_sid(),
        )
    assert "NVDA" in str(refused.value)
    assert "AAPL" not in str(refused.value), "the covered name is not what refused the batch"


def test_a_late_source_is_stale_even_though_it_arrived() -> None:
    late = check_data_freshness(
        decision_bar=BAR,
        eligible_sids=BOOK,
        deadlines=[marks(), signals(at=dt.time(18, 0))],
        observations=[on_time("bars"), on_time("scores", when=dt.datetime(2026, 9, 4, 23, 0, tzinfo=UTC))],
        now=NOW,
        classes=("mark", "signal"),
    )
    verdict = next(v for v in late.sources if v.source == "scores")
    assert verdict.published and verdict.late and not verdict.pending
    assert late.stale_by_class["signal"] == tuple(sorted(BOOK))
    assert late.stale_by_class["mark"] == ()


def test_an_unpublished_source_before_its_deadline_is_pending_not_late() -> None:
    early = check_data_freshness(
        decision_bar=BAR,
        eligible_sids=BOOK,
        deadlines=[marks(at=dt.time(23, 59))],
        observations=[],
        now=dt.datetime(2026, 9, 4, 18, 0, tzinfo=UTC),
        classes=("mark",),
    )
    verdict = early.sources[0]
    assert verdict.pending and not verdict.late and not verdict.published
    assert early.stale_by_class["mark"] == tuple(sorted(BOOK)), "pending data is still absent data"


def test_two_sources_of_one_class_are_complementary() -> None:
    """A sid one vendor published on time is fresh whatever the other did."""
    split = check_data_freshness(
        decision_bar=BAR,
        eligible_sids=BOOK,
        deadlines=[
            PublicationDeadline("primary", "mark", dt.time(20, 0), required_coverage=0.5),
            PublicationDeadline("backup", "mark", dt.time(20, 0), required_coverage=0.5),
        ],
        observations=[on_time("primary", ("AAPL", "MSFT")), on_time("backup", ("NVDA", "SPY"))],
        now=NOW,
        classes=("mark",),
    )
    assert split.ok
    assert split.stale_by_class["mark"] == ()


def test_a_class_with_no_declared_deadline_is_refused_not_passed() -> None:
    with pytest.raises(FreshnessError, match="no source declares a publication deadline"):
        check_data_freshness(
            decision_bar=BAR,
            eligible_sids=BOOK,
            deadlines=[marks()],
            observations=[on_time("bars")],
            now=NOW,
            classes=("mark", "covariance"),
        )


def test_an_unchecked_class_makes_no_claim() -> None:
    """Naming two classes asserts nothing about the other two, and says so."""
    report = healthy_data()
    assert report.classes == ("mark", "signal")
    freshness = report.freshness_by_sid()["AAPL"]
    assert freshness == InputFreshness(mark=True, halt=True, covariance=True, signal=True)


def test_the_input_class_names_are_the_authorization_modules_field_names() -> None:
    """The two modules share one vocabulary, or a verdict silently lands nowhere.

    ``freshness_by_sid`` builds ``InputFreshness(**fields)`` keyed by
    ``INPUT_CLASSES``. A rename on either side has to break here rather than in
    a live session, where the symptom would be a class quietly reading fresh.
    """
    assert tuple(f.name for f in dataclasses.fields(InputFreshness)) == INPUT_CLASSES


def test_a_deadline_counts_sessions_not_days() -> None:
    """A Friday bar's T+1 deadline lands on Monday, and skips Labor Day weekend."""
    friday = PublicationDeadline("fundamentals", "signal", dt.time(9, 0), sessions_after=1)
    assert friday.deadline("2026-09-04").date() == dt.date(2026, 9, 8), "Monday 09-07 is Labor Day"
    assert friday.deadline("2026-09-08").date() == dt.date(2026, 9, 9)


def test_a_naive_timestamp_is_refused() -> None:
    with pytest.raises(FreshnessError, match="no UTC offset"):
        check_data_freshness(
            decision_bar=BAR,
            eligible_sids=BOOK,
            deadlines=[marks()],
            observations=[on_time("bars")],
            now=dt.datetime(2026, 9, 4, 21, 30),
            classes=("mark",),
        )


def test_an_unknown_input_class_is_refused() -> None:
    with pytest.raises(FreshnessError, match="input class"):
        PublicationDeadline("bars", "liquidity", dt.time(20, 0))


# ---------------------------------------------------------------------------
# Halts reach freshness through halt status, not through the session table
# ---------------------------------------------------------------------------


def test_an_open_halt_carries_a_stale_mark_and_keeps_the_flattening_reduction() -> None:
    """A halted name has no trade to mark against, so its last close is not a price.

    The consequence is not restated here either: the stale mark runs through
    ``permitted_classes``, which keeps C0 and a reduction all the way to flat
    and removes the partial reduction and new exposure.
    """
    events = [
        {
            "sid": "NVDA",
            "event_type": "halt",
            "effective_date": "2026-09-04",
            "payload": json.dumps({"start": "2026-09-04", "end": None, "reason": "news pending"}),
        },
        {
            "sid": "MSFT",
            "event_type": "halt",
            "effective_date": "2026-08-03",
            "payload": json.dumps({"start": "2026-08-03", "end": "2026-08-03", "reason": "volatility"}),
        },
    ]
    open_now = halted_sids(events, BOOK, as_of=BAR)
    assert open_now == ("NVDA",), "a halt that has ended is history, not a current halt"

    report = healthy_data(halted=open_now)
    assert not report.ok
    freshness = report.freshness_by_sid()["NVDA"]
    assert freshness.mark is False and freshness.halt is True

    flat = permitted_classes((), Scope(True, True, True), ActionContext(freshness=freshness, reduction_to_flat=True))
    partial = permitted_classes((), Scope(True, True, True), ActionContext(freshness=freshness))
    assert {ActionClass.C0, ActionClass.C1, ActionClass.OB} <= flat
    assert ActionClass.C1 not in partial, "a partial reduction is sized on a price the halt removed"
    assert ActionClass.C3 not in flat and ActionClass.C3 not in partial


def test_an_unknown_halt_status_degrades_that_sid_to_cancel_only() -> None:
    """§3.10: a sid failing every class degrades to C0 only, for that sid."""
    allowed = permitted_classes(
        (), Scope(True, True, True), ActionContext(freshness=InputFreshness(mark=False, halt=False))
    )
    assert allowed == frozenset({ActionClass.C0})


def test_halt_rows_for_other_securities_are_ignored() -> None:
    events = [
        {
            "sid": "TSLA",
            "event_type": "halt",
            "effective_date": "2026-09-04",
            "payload": {"start": "2026-09-04", "end": None, "reason": "news pending"},
        },
        {"sid": "AAPL", "event_type": "split", "effective_date": "2026-09-04", "payload": {}},
    ]
    assert halted_sids(events, BOOK, as_of=BAR) == ()


# ---------------------------------------------------------------------------
# Check 2: broker-state freshness
# ---------------------------------------------------------------------------


def test_an_unknown_broker_state_age_is_stale_not_fresh() -> None:
    verdict = check_broker_state(refreshed_at=None, now=NOW, max_age_seconds=900.0)
    assert not verdict.ok and verdict.age_seconds is None
    assert verdict.states == frozenset({OperatingState.S2_STALE_BROKER_STATE})
    assert "unknown age is stale" in verdict.detail()


def test_broker_state_within_its_bound_is_fresh_and_at_the_bound_is_still_fresh() -> None:
    assert check_broker_state(refreshed_at=NOW - dt.timedelta(seconds=899), now=NOW, max_age_seconds=900.0).ok
    assert check_broker_state(refreshed_at=NOW - dt.timedelta(seconds=900), now=NOW, max_age_seconds=900.0).ok
    assert not check_broker_state(refreshed_at=NOW - dt.timedelta(seconds=901), now=NOW, max_age_seconds=900.0).ok


def test_s2_permits_a_cancel_only_after_a_fresh_open_order_read() -> None:
    """The row's "only when" is a condition, not a footnote (§3.10)."""
    without = check_broker_state(refreshed_at=NOW - dt.timedelta(hours=2), now=NOW, max_age_seconds=900.0)
    with_read = check_broker_state(
        refreshed_at=NOW - dt.timedelta(hours=2), now=NOW, max_age_seconds=900.0, open_orders_read_fresh=True
    )
    blind = FreshnessReport(broker_state=without)
    seeing = FreshnessReport(broker_state=with_read)
    assert permitted_classes(blind.states, Scope(True, True, True), blind.action_context()) == frozenset()
    assert ActionClass.C0 in permitted_classes(seeing.states, Scope(True, True, True), seeing.action_context())


# ---------------------------------------------------------------------------
# Check 3: clock health
# ---------------------------------------------------------------------------


def test_no_server_time_available_is_itself_a_fault() -> None:
    unsupported = check_clock(host_time=NOW, max_skew_seconds=30.0, broker=SilentBroker())
    assert not unsupported.ok and unsupported.skew_seconds is None
    assert unsupported.states == frozenset({OperatingState.S8_CLOCK_FAULT})
    assert "server_time" in (unsupported.unavailable_reason or "")

    failed = check_clock(host_time=NOW, max_skew_seconds=30.0, broker=TimedBroker(None))
    assert not failed.ok and "TimeoutError" in (failed.unavailable_reason or "")

    nothing = check_clock(host_time=NOW, max_skew_seconds=30.0)
    assert not nothing.ok and "no broker" in (nothing.unavailable_reason or "")


def test_an_unparseable_or_naive_server_time_is_no_server_time() -> None:
    assert read_server_time(TimedBroker("not a time"))[0] is None
    assert read_server_time(TimedBroker("2026-09-04T17:30:00"))[0] is None
    parsed, reason = read_server_time(TimedBroker("2026-09-04T21:30:00Z"))
    assert reason is None and parsed == NOW


def test_clock_skew_is_signed_and_bounded_in_both_directions() -> None:
    ahead = check_clock(host_time=NOW, max_skew_seconds=30.0, server_time=NOW - dt.timedelta(seconds=45))
    behind = check_clock(host_time=NOW, max_skew_seconds=30.0, server_time=NOW + dt.timedelta(seconds=45))
    assert ahead.skew_seconds == 45.0 and behind.skew_seconds == -45.0
    assert not ahead.ok and not behind.ok
    assert check_clock(host_time=NOW, max_skew_seconds=30.0, server_time=NOW + dt.timedelta(seconds=29)).ok


def test_a_clock_fault_leaves_a_cancel_and_nothing_else() -> None:
    report = FreshnessReport(clock=check_clock(host_time=NOW, max_skew_seconds=30.0, broker=SilentBroker()))
    assert permitted_classes(report.states, Scope(True, True, True), ActionContext()) == frozenset({ActionClass.C0})


# ---------------------------------------------------------------------------
# Check 4: scheduler liveness — a holiday is not a missed session
# ---------------------------------------------------------------------------


def test_the_fifteen_dark_sessions_are_missed_runs() -> None:
    verdict = check_scheduler(last_decision_bar=LAST_COMPLETED, as_of="2026-09-08", source="equity.jsonl")
    assert tuple(day.isoformat() for day in verdict.missed) == MISSED
    assert len(verdict.missed) == 15
    assert verdict.horizon_end == dt.date(2026, 9, 4)
    assert not verdict.ok
    assert verdict.states == frozenset({OperatingState.S9_SCHEDULER_FAULT})


def test_labor_day_is_a_closure_and_is_never_counted_as_missed() -> None:
    verdict = check_scheduler(last_decision_bar=LAST_COMPLETED, as_of="2026-09-08")
    assert LABOR_DAY not in verdict.missed
    assert (LABOR_DAY, "Labor Day") in verdict.closures
    assert "Labor Day" in verdict.detail() and "not missed runs" in verdict.detail()
    assert sessions.status(LABOR_DAY) == "holiday"


def test_a_gap_that_is_only_a_holiday_is_no_fault_at_all() -> None:
    """The other direction: the job fired on every session there was.

    Friday 2026-09-04 decided; Monday 2026-09-07 is Labor Day; the check runs
    Tuesday 2026-09-08 before that day's close. Nothing was missed, and the
    closure is still reported so the operator can see why the gap was four days.
    """
    verdict = check_scheduler(last_decision_bar="2026-09-04", as_of="2026-09-08")
    assert verdict.ok and verdict.missed == ()
    assert verdict.closures == ((LABOR_DAY, "Labor Day"),)


def test_weekday_arithmetic_gets_the_holiday_wrong() -> None:
    """Why the table is carried at all: the two functions disagree by exactly one."""
    last, today = dt.date(2026, 9, 4), dt.date(2026, 9, 8)
    weekdays = sum(1 for i in range(1, (today - last).days + 1) if (last + dt.timedelta(days=i)).weekday() < 5)
    verdict = check_scheduler(last_decision_bar=last, as_of=today)
    assert weekdays == 2, "Monday and Tuesday are weekdays"
    assert len(verdict.missed) == 0, "Monday was Labor Day and Tuesday has not closed"


def test_a_session_is_expected_only_once_its_close_is_past() -> None:
    """At any hour of ``as_of`` the close-of-day decision does not yet exist."""
    same_day = check_scheduler(last_decision_bar="2026-09-03", as_of="2026-09-04")
    assert same_day.ok and same_day.horizon_end == dt.date(2026, 9, 3)


def test_a_run_directory_that_has_decided_nothing_is_not_ok() -> None:
    verdict = check_scheduler(last_decision_bar=None, as_of="2026-09-08", source="equity.jsonl")
    assert not verdict.ok and not verdict.known and verdict.missed == ()
    assert "nothing has run here" in verdict.detail()


def test_a_date_the_table_cannot_cover_raises_rather_than_counting_weekdays() -> None:
    with pytest.raises(sessions.SessionTableRangeError):
        check_scheduler(last_decision_bar="2028-01-03", as_of="2028-01-10")


def test_the_verdict_is_the_same_whenever_it_is_evaluated() -> None:
    """Replay determinism: the inputs are the recorded clock, never wall time."""
    first = check_scheduler(last_decision_bar=LAST_COMPLETED, as_of="2026-09-08").to_dict()
    second = check_scheduler(last_decision_bar=LAST_COMPLETED, as_of="2026-09-08").to_dict()
    assert first == second


def test_the_module_reads_no_wall_clock() -> None:
    """§3.11: a replayed session is evaluated against its recorded decision bar.

    ``warn_if_stale_panel``'s ``pd.Timestamp.now`` is the read a replay must not
    inherit, or every historical session fails for being historical. The
    guarantee here is structural — the module has no clock to read — so it is
    checked structurally.
    """
    source = Path(__import__("prism.live.freshness", fromlist=["__file__"]).__file__).read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    for banned in (r"datetime\.now\(", r"date\.today\(", r"Timestamp\.now\(", r"time\.time\("):
        assert not re.search(banned, body), f"{banned} would make a replayed session fail for being historical"


def test_now_and_as_of_are_required() -> None:
    with pytest.raises(TypeError):
        check_broker_state(refreshed_at=NOW, max_age_seconds=900.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        check_scheduler(last_decision_bar=BAR)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# The session table's own horizon
# ---------------------------------------------------------------------------


def test_the_health_check_warns_within_ninety_days_of_the_table_ending() -> None:
    end = sessions.coverage_end()
    outside = session_table_horizon(as_of=end - dt.timedelta(days=91))
    inside = session_table_horizon(as_of=end - dt.timedelta(days=90))
    boundary = session_table_horizon(as_of=end)
    past = session_table_horizon(as_of=end + dt.timedelta(days=1))

    assert not outside.warn and outside.ok
    assert inside.warn and not inside.exhausted
    assert boundary.warn and boundary.days_remaining == 0
    assert past.exhausted and not past.warn
    assert "extend prism/io/sessions.py" in past.detail()
    assert sessions.TABLE_VERSION in inside.detail()


# ---------------------------------------------------------------------------
# Thresholds come from the declared profile
# ---------------------------------------------------------------------------


def test_thresholds_are_read_from_the_execution_profile() -> None:
    parsed = FreshnessThresholds.from_profile(
        {"freshness": {"max_broker_state_age_seconds": 300, "max_clock_skew_seconds": 5}}
    )
    assert parsed.max_broker_state_age_seconds == 300.0
    assert parsed.max_clock_skew_seconds == 5.0
    assert parsed.required_coverage == 1.0, "an undeclared key keeps the declared default"
    assert FreshnessThresholds.from_profile(None) == FreshnessThresholds()


def test_an_unreadable_threshold_is_refused() -> None:
    with pytest.raises(FreshnessError, match="must be a number"):
        FreshnessThresholds.from_profile({"freshness": {"max_clock_skew_seconds": "soon"}})
    with pytest.raises(FreshnessError, match="must not be negative"):
        FreshnessThresholds.from_profile({"freshness": {"max_broker_state_age_seconds": -1}})
    with pytest.raises(FreshnessError, match="fraction"):
        FreshnessThresholds.from_profile({"freshness": {"required_coverage": 1.5}})


# ---------------------------------------------------------------------------
# The live run directory, read-only
# ---------------------------------------------------------------------------

LIVE_RUN_DIR = Path("/home/b90/prism/runs/paper_loop_momentum2")


@pytest.mark.skipif(not LIVE_RUN_DIR.is_dir(), reason="the live paper run directory is not on this machine")
def test_the_live_run_directorys_ledger_agrees_with_the_session_table() -> None:
    """Read-only, and independent of which session the live book last completed.

    An earlier version pinned the ledger's last decision bar to a date, so the
    test went red the first night the loop ran again (2026-09-25). What can be
    asserted about live state without freezing it: the last decision bar is an
    exchange session no later than today, and as of the session after it the
    scheduler check reports nothing missed. The file mtimes are asserted
    unchanged so a future edit cannot quietly make this test a writer.
    """
    from prism.io import sessions as session_table

    ledger = LIVE_RUN_DIR / "equity.jsonl"
    before = {path.name: path.stat().st_mtime_ns for path in sorted(LIVE_RUN_DIR.iterdir())}
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    last = dt.date.fromisoformat(str(rows[-1]["decision_bar"]))
    assert session_table.is_session(last), f"{last} is not an exchange session"
    assert last <= dt.date.today(), f"the live ledger's last decision bar {last} is in the future"

    verdict = check_scheduler(
        last_decision_bar=last, as_of=session_table.next_session(last), source=str(ledger)
    )
    assert tuple(verdict.missed) == (), f"sessions missed between {last} and its successor: {verdict.missed}"

    after = {path.name: path.stat().st_mtime_ns for path in sorted(LIVE_RUN_DIR.iterdir())}
    assert before == after, "the live run directory must never be written by a test"


# ---------------------------------------------------------------------------
# The doctor's two session-table checks
# ---------------------------------------------------------------------------


def write_equity(run_dir: Path, bars) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = run_dir / "equity.jsonl"
    ledger.write_text(
        "\n".join(json.dumps({"decision_bar": bar, "cash": 1.0, "equity": 1.0}) for bar in bars) + "\n",
        encoding="utf-8",
    )
    return ledger


def test_the_doctor_reports_a_dark_loop_and_names_the_holiday(tmp_path: Path) -> None:
    from prism.scripts.doctor import check_missed_sessions

    run_dir = tmp_path / "run"
    write_equity(run_dir, ["2026-08-13", LAST_COMPLETED])
    result = check_missed_sessions(run_dir, today=dt.date(2026, 9, 8))
    assert result.status == "FAIL"
    assert "15 expected session(s)" in result.detail
    assert "Labor Day" in result.detail and "not missed runs" in result.detail


def test_the_doctor_does_not_call_a_holiday_a_missed_session(tmp_path: Path) -> None:
    from prism.scripts.doctor import check_missed_sessions

    run_dir = tmp_path / "run"
    write_equity(run_dir, ["2026-09-03", "2026-09-04"])
    result = check_missed_sessions(run_dir, today=dt.date(2026, 9, 8))
    assert result.status == "PASS", result.detail
    assert "Labor Day" in result.detail


def test_the_doctors_weekday_age_and_the_session_check_disagree_by_the_holiday(tmp_path: Path) -> None:
    """Both rows are kept: one needs no table, the other knows what a session is.

    Across Labor Day weekend the holiday-blind weekday count reaches the
    one-missed-session WARN threshold while the session table reports nothing
    missed. §3.11 keeps the weekday age precisely as this cross-check, and makes
    ``check`` the authority.
    """
    from prism.scripts.doctor import (
        LEDGER_STALE_WARN_WEEKDAYS,
        check_equity_ledger,
        check_missed_sessions,
        weekdays_since,
    )

    run_dir = tmp_path / "run"
    write_equity(run_dir, ["2026-09-04"])
    today = dt.date(2026, 9, 8)
    assert weekdays_since(dt.date(2026, 9, 4), today) >= LEDGER_STALE_WARN_WEEKDAYS
    assert check_equity_ledger(run_dir, today=today).status == "WARN"
    assert check_missed_sessions(run_dir, today=today).status == "PASS"


def test_the_doctor_warns_rather_than_failing_on_a_run_directory_that_never_ran(tmp_path: Path) -> None:
    from prism.scripts.doctor import check_missed_sessions

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = check_missed_sessions(run_dir, today=dt.date(2026, 9, 8))
    assert result.status == "WARN" and "no NAV rows" in result.detail


def test_the_doctor_fails_loud_outside_the_session_table(tmp_path: Path) -> None:
    from prism.scripts.doctor import check_missed_sessions

    run_dir = tmp_path / "run"
    write_equity(run_dir, ["2028-01-03"])
    result = check_missed_sessions(run_dir, today=dt.date(2028, 1, 10))
    assert result.status == "FAIL" and "bundled session table" in result.detail


def test_the_doctors_session_table_check_warns_within_ninety_days() -> None:
    from prism.scripts.doctor import check_session_table

    end = sessions.coverage_end()
    assert check_session_table(today=end - dt.timedelta(days=120)).status == "PASS"
    assert check_session_table(today=end - dt.timedelta(days=30)).status == "WARN"
    assert check_session_table(today=end + dt.timedelta(days=1)).status == "FAIL"


def test_the_doctors_network_flag_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m prism.scripts.doctor --network` keeps its contract (backward compatibility).

    The two probes are replaced, because the point is the wiring and not the
    venue: the flag must still parse, still run the offline surface, and still
    append the probe results.
    """
    from prism.scripts import doctor

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APCA_API_KEY_ID", "x")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "y")
    monkeypatch.delenv("APCA_API_BASE_URL", raising=False)
    called: list[str] = []
    monkeypatch.setattr(
        doctor,
        "check_alpaca_account",
        lambda *a, **k: (called.append("account"), [doctor.CheckResult("alpaca-account", "PASS", "stub")])[1],
    )
    monkeypatch.setattr(
        doctor,
        "check_twelvedata_quote",
        lambda *a, **k: (called.append("quote"), doctor.CheckResult("twelvedata-quote", "PASS", "stub"))[1],
    )
    run_dir = tmp_path / "runs" / "paper"
    write_equity(run_dir, ["2026-09-04"])
    universe = tmp_path / "u.txt"
    universe.write_text("\n".join(f"SYM{i}" for i in range(150)), encoding="utf-8")
    code = doctor.main(["--network", "--run-dir", str(run_dir), "--universe-file", str(universe)])
    assert called == ["account", "quote"]
    assert code in (0, 1)


# ---------------------------------------------------------------------------
# `prism check --expect-session`
# ---------------------------------------------------------------------------


def prepare_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APCA_API_KEY_ID", "x")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "y")
    monkeypatch.delenv("APCA_API_BASE_URL", raising=False)
    run_dir = tmp_path / "runs" / "paper"
    run_dir.mkdir(parents=True)
    return run_dir


def test_check_expect_session_exits_nonzero_on_a_missed_session(tmp_path, monkeypatch, capsys) -> None:
    from prism.cli import main

    run_dir = prepare_cli(tmp_path, monkeypatch)
    write_equity(run_dir, [LAST_COMPLETED])
    code = main(["check", str(run_dir), "--expect-session", "--asof", "2026-09-08"])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL  missed-sessions" in out
    assert "Labor Day" in out
    # All four §3.11 rows are named, so a report of one evaluated check cannot
    # read as four passes.
    for row in ("data-freshness", "broker-state-freshness", "clock-health", "missed-sessions"):
        assert row in out
    assert out.count("missed-sessions") == 1, "the scheduler row is printed once, not beside a duplicate"


def test_check_expect_session_passes_over_a_holiday_only_gap(tmp_path, monkeypatch, capsys) -> None:
    from prism.cli import main

    run_dir = prepare_cli(tmp_path, monkeypatch)
    write_equity(run_dir, ["2026-09-04"])
    code = main(["check", str(run_dir), "--expect-session", "--asof", "2026-09-08"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "PASS  missed-sessions" in out
    assert "Labor Day" in out


def test_check_expect_session_refuses_a_directory_that_decided_nothing(tmp_path, monkeypatch, capsys) -> None:
    """The flag is the caller asserting a session should have been decided."""
    from prism.cli import main

    run_dir = prepare_cli(tmp_path, monkeypatch)
    assert main(["check", str(run_dir), "--asof", "2026-09-08"]) == 0
    code = main(["check", str(run_dir), "--expect-session", "--asof", "2026-09-08"])
    out = capsys.readouterr().out
    assert code == 1
    assert "--expect-session asserts one should have been" in out


def test_check_asof_pins_the_clock_and_refuses_a_non_date(tmp_path, monkeypatch, capsys) -> None:
    from prism.cli import main

    run_dir = prepare_cli(tmp_path, monkeypatch)
    write_equity(run_dir, ["2026-09-04"])
    assert main(["check", str(run_dir), "--asof", "2026-09-08"]) == 0
    assert "as-of    2026-09-08" in capsys.readouterr().out
    assert main(["check", str(run_dir), "--asof", "the 8th"]) == 1
    assert "not an ISO date" in capsys.readouterr().err


def test_check_refuses_a_path_that_is_not_a_run_directory(tmp_path, monkeypatch, capsys) -> None:
    from prism.cli import main

    prepare_cli(tmp_path, monkeypatch)
    assert main(["check", "runs/nowhere"]) == 1
    assert "is not a run directory" in capsys.readouterr().err


def test_check_monitor_names_the_stage_it_waits_on(tmp_path, monkeypatch, capsys) -> None:
    from prism.cli import main

    run_dir = prepare_cli(tmp_path, monkeypatch)
    assert main(["check", str(run_dir), "--monitor"]) == 2
    assert "stage 5" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The report reaches the gate through the loop, not only in a unit test
# ---------------------------------------------------------------------------

#: Marks for the whole book. A price is needed for every held name, so the
#: reconcile read and the sizing both resolve.
PRICES = pd.Series({"AAPL": 200.0, "MSFT": 400.0, "NVDA": 100.0, "SPY": 500.0})


def partial_outage() -> FreshnessReport:
    """The signal feed publishes for two of four names, on time, at full marks."""
    return FreshnessReport(
        data=check_data_freshness(
            decision_bar=BAR,
            eligible_sids=BOOK,
            deadlines=[marks(), signals(required_coverage=1.0)],
            observations=[on_time("bars"), on_time("scores", ("AAPL", "MSFT"))],
            now=NOW,
            classes=("mark", "signal"),
        )
    )


def outage_ctx(tmp_path: Path, broker: Broker, gate=None) -> LiveLoopContext:
    return LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=broker,
        fills_ledger=tmp_path / "fills.jsonl",
        authorization=make_gate() if gate is None else gate,
    )


def test_a_partial_feed_outage_blocks_the_uncovered_name_through_the_whole_loop(tmp_path) -> None:
    """The stage 5 exit condition end to end, and not only at the gate.

    ``permitted_classes`` getting the per-sid answer right is worth nothing if
    the decision path never hands it one. Here the report goes into
    ``decide_and_submit`` and the assertions are made at the venue: the batch
    that would open new exposure on the name the feed missed is refused, the
    refusal names *that* name and not the covered one, nothing was submitted,
    and no pending state was persisted — so there is nothing for a restart to
    resume either.
    """
    report = partial_outage()
    broker = RecordingBroker(positions={"NVDA": 100.0})
    ctx = outage_ctx(tmp_path, broker)

    with pytest.raises(AuthorizationRefused) as refused:
        decide_and_submit(
            ctx,
            BAR,
            pd.Series({"AAPL": 0.30, "NVDA": 0.50}),
            PRICES,
            whole_shares=True,
            freshness=report,
            now=NOW,
        )
    assert "NVDA: C3" in str(refused.value)
    assert "AAPL" not in str(refused.value), "the covered name is not what refused the batch"
    assert broker.submitted == [], "the refusal must precede the request, not follow it"
    persisted = ctx.store.load()
    assert persisted is None or persisted.pending_orders == []


def test_the_covered_name_stays_tradable_in_the_same_outage(tmp_path) -> None:
    """The other half: "blocks classes per sid, not the whole account".

    Same outage, same report, same session. New exposure on the covered name
    goes to the venue, and so does a partial reduction on the uncovered one —
    the missing input is the signal, and a reduction is not new exposure.
    """
    broker = RecordingBroker(positions={"NVDA": 100.0})
    ctx = outage_ctx(tmp_path, broker)

    orders = decide_and_submit(
        ctx,
        BAR,
        pd.Series({"AAPL": 0.30, "NVDA": 0.05}),
        PRICES,
        whole_shares=True,
        freshness=partial_outage(),
        now=NOW,
    )
    by_sid = {order.symbol: order.qty for order in orders}
    assert by_sid == {"AAPL": 150.0, "NVDA": -50.0}
    assert {order.symbol for order in broker.submitted} == {"AAPL", "NVDA"}


def test_without_the_report_the_same_batch_is_authorized(tmp_path) -> None:
    """The control that makes the two tests above evidence rather than decoration.

    Identical account, identical book, identical order batch — and it is
    permitted, because nothing told the gate the feed had missed NVDA. The
    outage does its work through the ``freshness=`` argument and nowhere else,
    which is the wiring under test.
    """
    broker = RecordingBroker(positions={"NVDA": 100.0})
    ctx = outage_ctx(tmp_path, broker)

    orders = decide_and_submit(
        ctx,
        BAR,
        pd.Series({"AAPL": 0.30, "NVDA": 0.50}),
        PRICES,
        whole_shares=True,
        now=NOW,
    )
    assert {order.symbol for order in orders} == {"AAPL", "NVDA"}
    assert len(broker.submitted) == 2


def test_the_reports_account_wide_rows_reach_the_loop_as_well(tmp_path) -> None:
    """S2 is not per sid, and it stops the session before its first broker read."""
    broker = RecordingBroker(positions={"NVDA": 100.0})
    ctx = outage_ctx(tmp_path, broker)
    report = FreshnessReport(
        data=healthy_data(),
        broker_state=check_broker_state(refreshed_at=NOW - dt.timedelta(hours=1), now=NOW, max_age_seconds=900.0),
    )

    with pytest.raises(AuthorizationRefused, match=r"States \['S2'\]"):
        decide_and_submit(
            ctx,
            BAR,
            pd.Series({"AAPL": 0.30}),
            PRICES,
            whole_shares=True,
            freshness=report,
            now=NOW,
        )
    assert broker.submitted == []
    assert not (tmp_path / "state.json").exists()
