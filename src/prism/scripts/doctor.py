"""Preflight + operational-health doctor for the paper loop (SPEC §7.7).

Two jobs, one exit code.

**Preflight** turns the first-run failure classes into a report instead of a
mid-loop crash: missing credentials (N7 raises at ``from_env``), a missing or
degenerate universe file, an unwritable data dir, corrupt or schema-stale loop
state, and a forgotten kill switch (the book silently not trading is exactly
the silence N7 bans).

**Health** answers the question a green scheduler cannot: *is the book actually
alive?* The 2026-07-23..29 outage is the specimen — every nightly cycle died
unable to value a held position while the morning sweep exited 0 four times
(it had nothing pending, so it had nothing to fail on) and this doctor
reported "8 pass, 0 fail". A no-op success is not health. So the health checks
read the *durable record of work done*, never the exit status of the last
thing that ran:

* ``equity-ledger`` — how many sessions since the last NAV mark. A dark loop
  cannot hide from this; it is the one check that stays true when the log,
  the scheduler, and the sweep all look fine.
* ``missed-sessions`` — which sessions the exchange held that this book did
  not decide, read off the bundled session table (§3.11). A holiday is not a
  missed run, and the weekday age above cannot tell the two apart; it stays as
  the cross-check that needs no table.
* ``session-table`` — warns when that table ends within ninety days, because
  past its coverage end no session question can be answered at all.
* ``panel-record`` — did the last completed session record the panel it
  decided on (``prism.live.panel_log``)? A record that lags the equity ledger
  means decisions nobody can re-derive.
* ``shadow-record`` — did the last completed session write its shadow-return
  row (``shadow.jsonl``, the decided book marked on the paper book's closes)?
  The cycle shields that step from the trading path, so a shadow computation
  that fails every session leaves only a log line; this row is where that
  shows, and the tracking read the registration leans on is empty without it.
* ``regime-clock`` — consecutive clean sessions on the handoff §8
  precondition-(b) clock (docs/regime_step.md §4).
* ``nightly-log`` — the last wrapper verdict, and whether a *successful* one
  is stale (the "green but old" failure).
* ``wrapper-provenance`` — did the scheduler execute the repo's ops/ wrappers,
  or a drifted copy? The wrappers stamp ``wrapper=<physical path>`` onto their
  RUN/SWEEP lines; an unstamped or foreign-path session FAILs. This is the
  "what runs is what was reviewed" invariant: the 2026-07-23 outage's alert
  layer was committed but a stale ~/bin copy kept executing, invisibly.
* ``artifact-sync`` — did the off-box backup land? Read from the status file
  ``ops/artifacts_sync.sh`` writes after every attempt. The predecessor script
  logged PUSH-FAILED and exited 0 every night from 2026-09-25, so the last
  off-box copy of the ledgers was 2026-08-14 and nothing reported it.
* ``alert-transport`` — can any alert leave this machine? The marker, the log
  line and the exit code are all on-box; without ``PRISM_ALERT_CMD`` or a
  heartbeat URL a dead scheduler is reported to nobody (that was 2026-08-15 →
  2026-09-25).
* ``margin-cushion``, ``vol-scale-fallback``, ``cash-holding-drift``,
  ``tranche-state``, ``deployment-profile`` — the opt-in book features (a
  recorded margin read, the volatility scale, the T-bill holding, the tranched
  book and the deployment seed). Each PASSes when its feature has left no
  record in the run directory and otherwise only ever WARNs: the cushion below
  25% or a stale read, two or more consecutive scale fallbacks, the holding
  more than its band from target, tranche states the cycle could not have
  written, a seed that is missing, unreadable or readable by others. Promoting
  any of them to a failure is the owner's decision.
* ``alpaca-margin-cushion`` (network) — the account's cushion now, from one
  read-only margin read; a WARN below 25% or when the read fails.
* ``alpaca-account-book`` (network) — does the venue's book belong to this run
  directory? A run directory with no persisted book over a live account is the
  fatal condition that produced the outage.
* ``alpaca-holdings-priceable`` (network) — can every held position be marked?
  This is the mark step's precondition, probed before the loop pays for it.

Offline by default — no vendor credit is spent and no venue is touched.
``--network`` adds the Alpaca probes (free, unmetered: one account read, one
batch bar read of the held book) and one Twelve Data quote (1 credit of the
daily 800).

Exit code 0 when every check passes or warns, 1 when anything FAILs.
WARN means "runnable, but look": a missing Twelve Data key with the Alpaca
bar source configured is a WARN, not a FAIL.

    python -m prism.scripts.doctor                       # offline preflight + health
    python -m prism.scripts.doctor --network             # + venue reconciliation
    prism-doctor --run-dir "$(cat runs/ACTIVE_RUN_DIR)"  # installed entry point

Without ``--run-dir`` the run directory is the one ``runs/ACTIVE_RUN_DIR``
names, relative to the current directory; a missing or empty switch is refused
(exit 1) rather than checking a retired or current-directory book.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

# The switch file's name, by value: ``prism.live.safety.ACTIVE_RUN_DIR_FILENAME``.
# Not imported, because any ``prism.live`` import runs that package's __init__,
# which loads the venue adapter, and the offline paper-state assessment keeps
# this module's import closure venue-free (tests/test_paper_state_assessment.py
# asserts it). Every ``prism.live`` import in this module is therefore local.
ACTIVE_RUN_DIR_FILENAME = "ACTIVE_RUN_DIR"
from prism.scripts import announce_alias
from prism.workspace import load_env, resolve_workspace

logger = logging.getLogger(__name__)

# The decile construct needs a non-degenerate cross-section: below ~100 names
# a 0.10 decile book holds <10 per leg and concentration risk dwarfs signal.
MIN_MOMENTUM_UNIVERSE = 100

# Session-age thresholds in weekdays, holiday-blind (a holiday-only gap never
# reaches 2 weekdays, and Fri->Mon is 1). One missed session is a WARN because
# a single vendor stall is recoverable; four is a dead loop.
LEDGER_STALE_WARN_WEEKDAYS = 2
LEDGER_STALE_FAIL_WEEKDAYS = 4

# handoff §8 precondition (b): >= 21 consecutive clean regime sessions.
CLEAN_SESSIONS_REQUIRED = 21

# A successful backup older than this many weekdays means at least one nightly
# ran without one landing (the sync runs every nightly, after the doctor).
SYNC_STALE_WARN_WEEKDAYS = 1

# The live universe file is a snapshot of index membership; S&P changes land
# several times a quarter, and the 2026-07-17 build traded 2026-09-25 holding
# shorts in two names that had left the index (docs/operations.md).
UNIVERSE_STALE_DAYS = 45

# `# Current sp500 members as of 2026-09-25 with local price data …` — the
# header prism-build-universe writes (prism.scripts.build_sp500_universe).
_UNIVERSE_AS_OF = re.compile(r"\bas of (?P<date>\d{4}-\d{2}-\d{2})\b")

# `2026-07-29T18:30:16-0700 EXIT 1` — the wrapper's verdict line (ops/).
_LOG_VERDICT = re.compile(r"^(?P<ts>\S+)\s+(?P<verdict>EXIT|SKIP)\b\D*(?P<code>\d+)?")

# `… RUN: … wrapper=/abs/path commit=abc1234` — the wrappers' provenance stamp.
_WRAPPER_STAMP = re.compile(r"\bwrapper=(?P<path>\S+)")


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str


def _result(name: str, ok: bool, detail: str, *, warn: bool = False) -> CheckResult:
    return CheckResult(name, "PASS" if ok else ("WARN" if warn else "FAIL"), detail)


def check_python() -> CheckResult:
    ok = sys.version_info >= (3, 12)
    return _result("python", ok, f"{sys.version.split()[0]} (need >= 3.12)")


def check_env_credentials(env: dict[str, str], *, mode: str = "paper") -> list[CheckResult]:
    """Presence checks only — values are never printed (docs/security.md §2).

    ``mode`` is the ``--account paper|live`` the operator named. The endpoint
    check is a **refusal** under the wrong mode, not a warning (§3.10): an
    environment variable may set an endpoint but never the mode, so an endpoint
    off the mode's allowlist means this shell would point the loop somewhere the
    operator did not select, and a doctor that reports that as a warning has
    already failed at its one job.
    """
    results = []
    alpaca = bool(env.get("APCA_API_KEY_ID")) and bool(env.get("APCA_API_SECRET_KEY"))
    results.append(
        _result(
            "alpaca-credentials",
            alpaca,
            (
                "APCA_API_KEY_ID/APCA_API_SECRET_KEY present"
                if alpaca
                else "APCA_API_KEY_ID/APCA_API_SECRET_KEY missing — the paper loop cannot trade; "
                "put the *paper* keys in .env (docs/quickstart.md)"
            ),
        )
    )
    twelve = bool(env.get("TWELVEDATA_API_KEY"))
    results.append(
        _result(
            "twelvedata-key",
            twelve,
            (
                "TWELVEDATA_API_KEY present"
                if twelve
                else "TWELVEDATA_API_KEY missing — fine for the Alpaca bar source (the loop "
                "default), required for the research spine"
            ),
            warn=True,
        )
    )
    results.append(check_endpoint_mode(env, mode=mode))
    return results


def check_endpoint_mode(env: Mapping[str, str], *, mode: str = "paper") -> CheckResult:
    """The resolved endpoint against ``mode``'s allowlist — a FAIL, not a WARN.

    The comparison is the one ``AlpacaBroker`` makes when it is constructed
    (``prism.live.authorization.resolve_endpoint``), so the doctor reports
    exactly the verdict the loop will reach rather than a second opinion.
    """
    from prism.live.authorization import AuthorizationRefused, resolve_endpoint

    base = env.get("APCA_API_BASE_URL", "")
    try:
        resolved = resolve_endpoint(mode, from_environment=base or None)
    except AuthorizationRefused as exc:
        return _result("alpaca-endpoint", False, f"{exc}")
    where = f"({resolved})" if base else f"({resolved}, the default)"
    return _result("alpaca-endpoint", True, f"{mode} endpoint {where}")


def check_authorization(run_dir: Path, *, mode: str = "paper", host: str | None = None) -> CheckResult:
    """The §3.10 authorization record for this run directory, read as the loop reads it.

    Offline: the account id is the venue's answer and is checked when the client
    binds, so what is checkable here is the record's own consistency — schema,
    mode, endpoint allowlist, execution host, expiry. Under ``paper`` an absent
    record is the pre-authorization paper instrument and passes with a note;
    under ``live`` it is state S4b, where every broker write is refused.
    """
    from prism.live.authorization import (
        AuthorizationRefused,
        execution_host,
        load_authorization,
    )

    try:
        record = load_authorization(run_dir)
    except AuthorizationRefused as exc:
        return _result("authorization", False, f"{exc}")
    if record is None:
        if mode == "live":
            return _result(
                "authorization",
                False,
                f"no authorization record in {run_dir} under --account live: state S4b permits nothing "
                "and every broker write is refused, C0 included (§3.10)",
            )
        return _result(
            "authorization",
            True,
            f"no authorization record in {run_dir} — the pre-authorization paper instrument; a live "
            "account requires one",
        )
    problems = []
    if record.mode != mode:
        problems.append(f"record mode {record.mode!r} != --account {mode!r}")
    this_host = execution_host() if host is None else host
    if record.execution_host != this_host:
        problems.append(f"execution_host {record.execution_host!r} != this host {this_host!r}")
    if problems:
        return _result("authorization", False, "; ".join(problems) + " (§3.10)")
    expiry = f", expires {record.expires_at}" if record.expires_at else ", no expiry (paper)"
    return _result(
        "authorization",
        True,
        f"{record.mode} grant for account {record.account_name!r} at {record.endpoint}, scope "
        f"{record.scope.to_dict()}{expiry}",
    )


def _read_universe(path: Path | None) -> list[str]:
    """The universe file's symbols, or ``[]`` — the loop's parse, minus the raise."""
    from prism.io.universe_file import load_universe_symbols

    if path is None or not path.exists():
        return []
    return load_universe_symbols(path, require_nonempty=False)


def universe_as_of(path: Path) -> date | None:
    """The ``as of YYYY-MM-DD`` date in the file's comment header, or None.

    Only the leading comment block is read: a symbol line never carries a date,
    and a hand-written list has no header at all, which is legitimate and is
    reported as "no as-of", not as stale.
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        match = _UNIVERSE_AS_OF.search(stripped)
        if match:
            try:
                return date.fromisoformat(match.group("date"))
            except ValueError:
                return None
    return None


def check_universe_file(path: Path | None, *, today: date | None = None) -> CheckResult:
    """Existence, breadth and age of the universe file the loop will trade.

    Age is read off the header's ``as of`` date. Membership drift is silent in
    every other row: a stale file still parses, still has ~500 names, and still
    trades — it traded shorts in two index leavers on 2026-09-25 — so the only
    place the drift shows is the date the file itself claims.
    """
    today = today or date.today()
    if path is None:
        return _result(
            "universe-file", False, "no --universe-file given — pass the file the loop will trade", warn=True
        )
    if not path.exists():
        return _result("universe-file", False, f"{path} does not exist")
    symbols = _read_universe(path)
    if not symbols:
        return _result("universe-file", False, f"{path} parses to zero symbols")
    as_of = universe_as_of(path)
    if as_of is None:
        age_note, stale = ", no 'as of' date in the header", False
    else:
        age_days = (today - as_of).days
        age_note, stale = f", as of {as_of.isoformat()} ({age_days} days ago)", age_days > UNIVERSE_STALE_DAYS
    if len(symbols) < MIN_MOMENTUM_UNIVERSE:
        return _result(
            "universe-file",
            False,
            f"{path}: {len(symbols)} symbols < {MIN_MOMENTUM_UNIVERSE} — a decile book "
            f"degenerates on a thin cross-section (fine for the ensemble cost instrument){age_note}",
            warn=True,
        )
    if stale:
        return _result(
            "universe-file",
            False,
            f"{path}: {len(symbols)} symbols{age_note} — older than {UNIVERSE_STALE_DAYS} days; index "
            "membership has moved since this build, so the book may hold names that left the index. "
            "Rebuild it (prism-build-universe)",
            warn=True,
        )
    return _result("universe-file", True, f"{path}: {len(symbols)} symbols{age_note}")


def check_data_dir(data_dir: Path) -> CheckResult:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return _result("data-dir", False, f"{data_dir} not writable: {exc}")
    return _result("data-dir", True, f"{data_dir} writable")


def check_loop_state(run_dir: Path) -> CheckResult:
    """Load the persisted state exactly the way the loop will (fail-loud, N7)."""
    from prism.live.state import StateStore

    state_path = run_dir / "state.json"
    if not state_path.exists():
        return _result("loop-state", True, f"{state_path} absent — fresh loop")
    try:
        state = StateStore(state_path).load()
    except ValueError as exc:
        return _result("loop-state", False, f"{exc}")
    assert state is not None
    pending = len(state.pending_orders)
    return _result(
        "loop-state",
        True,
        f"{len(state.positions)} positions, cash {state.cash:.2f}, "
        f"{pending} pending orders"
        + (f" for {state.pending_decision_bar} (will settle next cycle)" if pending else "")
        + (f", last refresh {state.last_refresh_bar}" if state.last_refresh_bar else ""),
    )


def check_kill_switch(run_dir: Path) -> CheckResult:
    kill = run_dir / "KILL_SWITCH"
    if kill.exists():
        return _result(
            "kill-switch",
            False,
            f"{kill} PRESENT — the book is halted and will not trade; delete the file to resume",
            warn=True,
        )
    return _result("kill-switch", True, f"{kill} absent — trading enabled")


# ---------------------------------------------------------------------------
# Operational health — the durable record of work done, not a green exit code
# ---------------------------------------------------------------------------


def weekdays_since(last: date, today: date) -> int:
    """Weekdays in ``(last, today]`` — a holiday-blind session age.

    Deliberately not a trading calendar: the loop already owns the exchange
    calendar through its panel index, and a health check that needs a holiday
    table to answer "is the book dark?" is a health check that breaks in
    January. Holiday-blindness only ever *over*-counts, by at most one or two,
    which is why the FAIL threshold sits at four.
    """
    if today <= last:
        return 0
    days = (today - last).days
    return sum(1 for i in range(1, days + 1) if (last + timedelta(days=i)).weekday() < 5)


def _last_ledger_bar(path: Path) -> str | None:
    """The last ``decision_bar`` in an append-only per-session ledger."""
    import json

    if not path.exists():
        return None
    for line in reversed([ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]):
        try:
            return str(json.loads(line)["decision_bar"])
        except (ValueError, KeyError, TypeError):
            continue
    return None


def check_equity_ledger(run_dir: Path, *, today: date | None = None) -> CheckResult:
    """Session age of the last NAV mark — the load-bearing liveness signal.

    The equity ledger gains exactly one row per completed cycle, so its tail is
    the last time the loop *finished*. Nothing else in the operational surface
    has that property: the scheduler reports that it launched something, the
    sweep log reports that a no-op no-opped, and a crash loop leaves both green.
    """
    today = today or date.today()
    ledger = run_dir / "equity.jsonl"
    last = _last_ledger_bar(ledger)
    if last is None:
        # Not a FAIL on its own — a genuinely fresh run directory has no NAV
        # history and that is legitimate. It becomes fatal only in company with
        # a live venue book, which is what check_account_book decides.
        return _result(
            "equity-ledger",
            False,
            f"{ledger} has no NAV rows — no cycle has completed in this run directory; "
            "fatal if the account already holds a book (see alpaca-account-book)",
            warn=True,
        )
    try:
        age = weekdays_since(date.fromisoformat(last), today)
    except ValueError:
        return _result("equity-ledger", False, f"{ledger}: unparseable last decision_bar {last!r}")
    detail = f"last NAV mark {last} ({age} weekday(s) ago), {today.isoformat()} today"
    if age >= LEDGER_STALE_FAIL_WEEKDAYS:
        return _result(
            "equity-ledger",
            False,
            detail + " — the loop is DARK: no cycle has completed for a full trading week. "
            "Read the nightly log; failed sessions cannot be backfilled as live evidence",
        )
    if age >= LEDGER_STALE_WARN_WEEKDAYS:
        return _result("equity-ledger", False, detail + " — a session was missed; check the nightly log", warn=True)
    return _result("equity-ledger", True, detail)


def check_missed_sessions(run_dir: Path, *, today: date | None = None) -> CheckResult:
    """Scheduler liveness against the bundled session table — a holiday is not a miss.

    This is the §3.11 missed-run check, and it is a *different* question from
    ``equity-ledger`` above. That one asks "how old is the newest NAV mark?" in
    holiday-blind weekdays, which only ever over-counts and therefore cannot say
    which sessions were expected. This one asks "which sessions did the exchange
    hold that this book did not decide?" and answers with dates. The weekday age
    stays as a cross-check, because it needs no table and so cannot be wrong
    about a year the table does not cover; ``prism check --expect-session`` is
    the freshness authority (§3.11).

    Fails loud rather than counting weekdays when the table cannot answer: a
    calendar that guesses about a year it does not hold is silently wrong exactly
    when nobody is watching.
    """
    from prism.live.freshness import check_scheduler

    today = today or date.today()
    ledger = run_dir / "equity.jsonl"
    last = _last_ledger_bar(ledger)
    if last is None:
        # Mirrors check_equity_ledger: a genuinely fresh run directory has no
        # NAV history and that is legitimate, not a scheduler fault.
        return _result(
            "missed-sessions",
            False,
            f"{ledger} has no NAV rows — there is no last decision bar to expect sessions after",
            warn=True,
        )
    try:
        verdict = check_scheduler(last_decision_bar=last, as_of=today, source=str(ledger))
    except ValueError as exc:
        # SessionTableRangeError (the table does not cover this year) and
        # FreshnessError (the ledger's bar is not a date) are both ValueErrors,
        # and both mean the same thing here: this check has no answer, and says
        # so instead of falling back to weekday arithmetic.
        return _result("missed-sessions", False, f"{exc}")
    if verdict.missed:
        return _result("missed-sessions", False, verdict.detail())
    return _result("missed-sessions", True, verdict.detail())


def check_session_table(*, today: date | None = None) -> CheckResult:
    """The bundled session table's remaining coverage (§3.11: warn within ninety days).

    The table is what every session question is answered from, so its running
    out is an engine fault rather than a book fault: past its coverage end the
    missed-run check cannot be evaluated at all, because the table refuses
    instead of guessing. That is a FAIL; ninety days out is the WARN.
    """
    from prism.live.freshness import session_table_horizon

    horizon = session_table_horizon(as_of=today or date.today())
    if horizon.exhausted:
        return _result("session-table", False, horizon.detail())
    return _result("session-table", not horizon.warn, horizon.detail(), warn=True)


@dataclass(frozen=True)
class RegimeStreak:
    """The precondition-(b) clock read off ``regime.jsonl`` under docs/regime_step.md §4.

    ``run`` is the trailing run of clean rows on consecutive sessions ending at
    the last row; ``missed`` lists the sessions after the last row whose close has
    passed without a row. ``streak`` is what stands now: ``run``, or 0 once a
    session has gone unrecorded, because absence of the record is absence of the
    session.
    """

    streak: int
    run: int
    rows: int
    last_bar: str | None
    missed: tuple[date, ...]
    run_break: str | None


def regime_streak(rows: Sequence[Mapping[str, object]], *, as_of: date) -> RegimeStreak:
    """Consecutive clean sessions standing at ``as_of`` (docs/regime_step.md §4; owner decision 2026-09-25).

    A session counts iff it appended a row with ``clean: true``. The run breaks
    on an unclean row and on a session with no row between two rows; a rerun's
    later row for the same bar replaces the earlier one. A session is expected
    once its close has passed, so the horizon is the last session strictly
    before ``as_of`` — the nightly's own row for ``as_of`` is never counted as
    missed while the day is open. Raises ``ValueError`` when the session table
    cannot date a bar; the caller reports that rather than guessing.
    """
    from prism.io import sessions

    by_bar: dict[date, bool] = {}
    for row in rows:
        by_bar[date.fromisoformat(str(row["decision_bar"])[:10])] = bool(row.get("clean"))
    bars = sorted(by_bar)
    if not bars:
        return RegimeStreak(0, 0, 0, None, (), None)
    run, run_break, later = 0, None, None
    for bar in reversed(bars):
        if not by_bar[bar]:
            run_break = f"row {bar} is not clean"
            break
        if later is not None and sessions.previous_session(later) != bar:
            run_break = f"no row for the session(s) between {bar} and {later}"
            break
        run += 1
        later = bar
    last = bars[-1]
    horizon = sessions.previous_session(as_of)
    missed = tuple(sessions.sessions_between(sessions.next_session(last), horizon)) if horizon > last else ()
    return RegimeStreak(0 if missed else run, run, len(by_bar), last.isoformat(), missed, run_break)


def check_regime_clock(run_dir: Path, *, today: date | None = None) -> CheckResult:
    """Consecutive clean sessions on the handoff §8 precondition-(b) clock (:func:`regime_streak`).

    Reported, never FAILed — an incomplete clock is the normal state of an
    honest program, and this check exists so the count is read off the ledger
    instead of remembered.
    """
    from prism.live import read_regime_ledger

    today = today or date.today()
    ledger = run_dir / "regime.jsonl"
    frame = read_regime_ledger(ledger)
    if frame.empty:
        return _result(
            "regime-clock",
            False,
            f"{ledger} empty or absent — precondition (b) clock at 0/{CLEAN_SESSIONS_REQUIRED} "
            "(run the loop with --regime and FRED_API_KEY to start it)",
            warn=True,
        )
    try:
        clock = regime_streak(frame.to_dict("records"), as_of=today)
    except (KeyError, ValueError) as exc:
        return _result("regime-clock", False, f"{ledger}: cannot date the clock ({exc})", warn=True)
    if clock.missed:
        detail = (
            f"0/{CLEAN_SESSIONS_REQUIRED} consecutive clean sessions as of {today.isoformat()}: "
            f"{len(clock.missed)} session(s) since {clock.last_bar} have no regime row, and absence of the "
            f"record is absence of the session (docs/regime_step.md §4); the run through {clock.last_bar} "
            f"was {clock.run} ({clock.rows} row(s) total)"
        )
    else:
        detail = (
            f"{clock.streak}/{CLEAN_SESSIONS_REQUIRED} consecutive clean sessions through {clock.last_bar} "
            f"({clock.rows} row(s) total)"
        )
        if clock.run_break:
            detail += f"; the run starts after: {clock.run_break}"
    if clock.streak >= CLEAN_SESSIONS_REQUIRED:
        return _result("regime-clock", True, detail + " — precondition (b) satisfied")
    return _result("regime-clock", False, detail, warn=True)


def check_panel_record(run_dir: Path, *, today: date | None = None) -> CheckResult:
    """Did the last completed session record the panel it decided on?

    Since 2026-09-30 every cycle writes the bar panel it scored under
    ``<run_dir>/panels/`` before deciding (``prism.live.panel_log``), so each
    decision can be re-derived from its inputs. Nothing else checked that the
    record kept up with the ledger. PASS when the newest index row is the equity
    ledger's last decision bar; WARN when no panel has ever been recorded (the
    log starts with the first session after the change) or when the panel is
    *ahead* of the NAV mark (a cycle recorded its inputs and then did not
    complete, which ``equity-ledger`` dates); FAIL when the record lags the
    ledger, because a session then decided on bars nobody can see again.
    """
    from prism.io import sessions

    today = today or date.today()
    index_path = run_dir / "panels" / "index.jsonl"
    ledger = run_dir / "equity.jsonl"
    last_panel = _last_ledger_bar(index_path)
    if last_panel is None:
        return _result(
            "panel-record",
            False,
            f"{index_path} absent or empty — no session has recorded its panel since 2026-09-30 "
            "(prism.live.panel_log writes it before each decision)",
            warn=True,
        )
    last_nav = _last_ledger_bar(ledger)
    if last_nav is None:
        return _result(
            "panel-record",
            False,
            f"panel recorded for {last_panel} but {ledger} has no NAV rows — no cycle has completed",
            warn=True,
        )
    if last_panel == last_nav:
        return _result("panel-record", True, f"panel recorded for {last_panel}, the ledger's last session")
    try:
        panel_date, nav_date = date.fromisoformat(last_panel), date.fromisoformat(last_nav)
    except ValueError as exc:
        return _result("panel-record", False, f"{index_path} / {ledger}: unparseable decision bar ({exc})")
    if panel_date > nav_date:
        return _result(
            "panel-record",
            False,
            f"panel recorded for {last_panel} but the last NAV mark is {last_nav} ({today.isoformat()} today): "
            "that cycle recorded its inputs and did not complete; see equity-ledger and the nightly log",
            warn=True,
        )
    try:
        lag = sum(1 for d in sessions.sessions_between(panel_date, nav_date) if d > panel_date)
        lag_text = f"{lag} session(s)"
    except ValueError:
        lag_text = "an uncountable interval (outside the session table)"
    return _result(
        "panel-record",
        False,
        f"panel record ends at {last_panel} but the ledger's last session is {last_nav}: the record lags by "
        f"{lag_text}, so those decisions cannot be re-derived from their inputs (docs/live_monitor_design.md)",
    )


def check_shadow_record(run_dir: Path, *, today: date | None = None) -> CheckResult:
    """Did the last completed session write its shadow-return row (``<run_dir>/shadow.jsonl``)?

    The cycle records, after each NAV mark, the decided book's return on the
    same closes as the paper book's (``prism.live.daily._shadow_record``), and
    the tracking read of paper minus shadow is one conjunct of "the live
    monitor is not in contradiction" (docs/live_monitor_design.md §4). That
    step is shielded from the trading path on purpose, so a shadow computation
    that fails every session leaves nothing but a log line while the tracking
    read stays empty. This row makes that durable.

    A row is owed for the ledger's last bar only when the cycle could have
    written one: the previous ledger bar is the preceding exchange session (a
    gap is not a day) and a refresh book in ``targets.jsonl`` predates that
    bar. PASS when the newest shadow row is the ledger's last bar; WARN when
    no row has ever been written or none was owed (with the reason); FAIL when
    one was owed and is missing, counting the lag as ``panel-record`` does.
    """
    from prism.io import sessions
    from prism.live.ledgers import read_rows

    today = today or date.today()
    shadow_path = run_dir / "shadow.jsonl"
    ledger = run_dir / "equity.jsonl"
    shadow_rows = read_rows(shadow_path)
    last_shadow = _last_ledger_bar(shadow_path)
    if last_shadow is None:
        return _result(
            "shadow-record",
            False,
            f"{shadow_path} absent or empty — no shadow row yet; the first can land only once a refresh "
            "precedes the previous session (docs/live_monitor_design.md)",
            warn=True,
        )
    nav_bars = sorted({str(row["decision_bar"]) for row in read_rows(ledger) if "decision_bar" in row})
    if not nav_bars:
        return _result(
            "shadow-record",
            False,
            f"shadow row for {last_shadow} but {ledger} has no NAV rows — no cycle has completed",
            warn=True,
        )
    last_nav = nav_bars[-1]
    if last_shadow == last_nav:
        newest = next(row for row in reversed(shadow_rows) if str(row.get("decision_bar")) == last_shadow)
        try:
            diff = float(newest["paper_return"]) - float(newest["shadow_return"])
            diff_text = f"; paper minus shadow {diff:+.6f}"
        except (KeyError, TypeError, ValueError):
            diff_text = ""
        return _result(
            "shadow-record", True, f"shadow recorded for {last_shadow}, the ledger's last session{diff_text}"
        )
    try:
        shadow_date, nav_date = date.fromisoformat(last_shadow[:10]), date.fromisoformat(last_nav[:10])
    except ValueError as exc:
        return _result("shadow-record", False, f"{shadow_path} / {ledger}: unparseable decision bar ({exc})")
    if shadow_date > nav_date:
        return _result(
            "shadow-record",
            False,
            f"shadow row for {last_shadow} but the last NAV mark is {last_nav} ({today.isoformat()} today): the "
            "shadow ledger is ahead of the equity ledger, which the cycle never writes in that order; see "
            "equity-ledger and the nightly log",
            warn=True,
        )
    # Was a row owed for the last bar? Mirror the cycle's own three preconditions.
    if len(nav_bars) < 2:
        return _result(
            "shadow-record",
            False,
            f"shadow record ends at {last_shadow}; the ledger's last session {last_nav} has no earlier ledger "
            "session to shadow from, so no row was owed",
            warn=True,
        )
    prev_bar = nav_bars[-2]
    try:
        prev_date = date.fromisoformat(prev_bar[:10])
        consecutive = sessions.is_session(prev_date) and sessions.next_session(prev_date) == nav_date
    except ValueError as exc:
        return _result("shadow-record", False, f"{ledger}: cannot date {prev_bar!r} ({exc})", warn=True)
    if not consecutive:
        return _result(
            "shadow-record",
            False,
            f"shadow record ends at {last_shadow}; no row was owed for {last_nav} because the previous ledger "
            f"bar {prev_bar} is not the preceding exchange session (a gap is not a day; see missed-sessions)",
            warn=True,
        )
    refreshes = read_rows(run_dir / "targets.jsonl")
    if not any(str(row.get("refresh_bar", "")) < prev_bar for row in refreshes if "refresh_bar" in row):
        return _result(
            "shadow-record",
            False,
            f"shadow record ends at {last_shadow}; no row was owed for {last_nav} because no refresh book in "
            f"{run_dir / 'targets.jsonl'} predates {prev_bar}",
            warn=True,
        )
    try:
        lag = sum(1 for d in sessions.sessions_between(shadow_date, nav_date) if d > shadow_date)
        lag_text = f"{lag} session(s)"
    except ValueError:
        lag_text = "an uncountable interval (outside the session table)"
    return _result(
        "shadow-record",
        False,
        f"shadow record ends at {last_shadow} but the ledger's last session is {last_nav}: the record lags by "
        f"{lag_text} although a row was owed, so the shadow step failed and the tracking read is missing "
        "those sessions; read LIVE MONITOR lines in the nightly log (docs/live_monitor_design.md)",
    )


# ---------------------------------------------------------------------------
# Opt-in book features: margin, volatility scale, T-bill holding, tranches,
# deployment seed. Each PASSes with "not in use" when its feature has left no
# record, and none of them FAILs: they warn first, and promoting any of them to
# a failure is the owner's decision (docs/operations.md, 2026-10-16).
# ---------------------------------------------------------------------------


def _cushion_result(name: str, *, cushion: float | None, equity: float, maintenance: float, where: str) -> CheckResult:
    """The cushion verdict shared by the recorded and the live margin checks."""
    from prism.live.margin import CUSHION_WARN_FLOOR

    if cushion is None:
        return _result(
            name,
            False,
            f"equity {equity:,.2f} is not positive ({where}): the account is in deficit and has no cushion",
            warn=True,
        )
    detail = (
        f"cushion {cushion:.1%} = (equity {equity:,.2f} - maintenance margin {maintenance:,.2f}) / equity ({where})"
    )
    if cushion < CUSHION_WARN_FLOOR:
        return _result(
            name,
            False,
            detail + f" — below the {CUSHION_WARN_FLOOR:.0%} warning floor. Nothing acts on it: selling the T-bill "
            "holding first is not implemented, so lowering gross is a manual decision",
            warn=True,
        )
    return _result(name, True, detail)


def _levered_book_evidence(run_dir: Path) -> list[str]:
    """What in the run directory says the book uses margin beyond an unscaled long/short book."""
    from prism.live.ledgers import read_rows

    evidence = []
    sizing = read_rows(run_dir / "sizing.jsonl")
    if sizing:
        evidence.append(f"the book is volatility-scaled up to 2x gross ({len(sizing)} sizing.jsonl row(s))")
    if read_rows(run_dir / "cash_holding.jsonl"):
        evidence.append("the book holds a T-bill ETF beside its shorts (cash_holding.jsonl records it)")
    elif any(isinstance(row.get("cash_holding"), dict) for row in read_rows(run_dir / "targets.jsonl")):
        evidence.append("the book holds a T-bill ETF beside its shorts (targets.jsonl records cash_holding)")
    return evidence


def check_margin_cushion(run_dir: Path, *, today: date | None = None) -> CheckResult:
    """The newest recorded margin read (``margin.jsonl``): cushion and age.

    ``python -m prism.scripts.margin_report --run-dir … --record`` writes the
    record; nothing schedules it. Absent, the check passes while the book is an
    unscaled long/short book without a T-bill holding, whose cushion stays near
    70% at the regulatory minimums, and warns when the run directory shows a
    volatility-scaled book or a T-bill holding. Present, it warns when the
    cushion is below 25% or the read is two or more weekdays old.
    """
    from prism.live.margin import (
        MARGIN_LEDGER_FILENAME,
        MarginError,
        read_margin_rows,
        row_cushion,
    )

    today = today or date.today()
    path = run_dir / MARGIN_LEDGER_FILENAME
    rows = read_margin_rows(path)
    if not rows:
        evidence = _levered_book_evidence(run_dir)
        if evidence:
            return _result(
                "margin-cushion",
                False,
                f"{path} absent or empty, but {'; '.join(evidence)}: no margin read is recorded for a book that "
                f"draws on margin. Record one (python -m prism.scripts.margin_report --run-dir {run_dir} --record)",
                warn=True,
            )
        return _result(
            "margin-cushion",
            True,
            f"{path} absent — no margin read recorded here; not needed while the book is unscaled and holds no "
            "T-bill ETF (prism.scripts.margin_report --record writes one)",
        )
    row = rows[-1]
    try:
        cushion = row_cushion(row)
        as_of = date.fromisoformat(str(row["as_of"]))
        equity, maintenance = float(row["equity"]), float(row["maintenance_margin"])
    except (KeyError, TypeError, ValueError, MarginError) as exc:
        return _result("margin-cushion", False, f"{path}: newest row unreadable ({exc})", warn=True)
    age = weekdays_since(as_of, today)
    gross = row.get("gross_over_equity")
    where = f"read {as_of.isoformat()}, {age} weekday(s) ago" + (
        f", gross {float(gross):.2f}x equity" if isinstance(gross, (int, float)) else ""
    )
    result = _cushion_result("margin-cushion", cushion=cushion, equity=equity, maintenance=maintenance, where=where)
    if result.status == "PASS" and age >= LEDGER_STALE_WARN_WEEKDAYS:
        return _result(
            "margin-cushion",
            False,
            result.detail + " — stale: the margin can have moved since; record a new read",
            warn=True,
        )
    return result


def check_live_margin(broker: object) -> CheckResult:
    """The account's cushion now, from one read-only ``account_margin()`` (network). Warns, never fails."""
    try:
        margin = broker.account_margin()  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 — a probe failure is a WARN, not a book failure
        return _result("alpaca-margin-cushion", False, f"account margin read failed: {exc}", warn=True)
    gross = margin.gross_over_equity
    where = "venue, now" + ("" if gross is None else f", gross {gross:.2f}x equity")
    return _cushion_result(
        "alpaca-margin-cushion",
        cushion=margin.cushion,
        equity=margin.equity,
        maintenance=margin.maintenance_margin,
        where=where,
    )


#: Consecutive volatility-scale fallbacks at which the doctor warns.
VOL_SCALE_FALLBACK_STREAK_WARN = 2


def check_vol_scale_fallback(run_dir: Path) -> CheckResult:
    """Consecutive refresh rows of ``sizing.jsonl`` whose scale decision fell back.

    A fallback sizes the book at ``min(1, last good scale)`` because the
    volatility window could not be trusted (``prism.portfolio.vol_scale``). One
    is a stale bar or a short window; two in a row means the book is being held
    below its rule by a data problem. A tranched row counts when any of its due
    tranches fell back. A ``warmup`` decision (no good scale yet) is not a
    fallback.
    """
    from prism.live.ledgers import read_rows
    from prism.live.vol_sizing import SIZING_LEDGER_FILENAME, row_decisions

    path = run_dir / SIZING_LEDGER_FILENAME
    rows = read_rows(path)
    if not rows:
        return _result("vol-scale-fallback", True, f"{path} absent — the volatility scale is not in use")
    streak: list[str] = []
    reasons: list[str] = []
    for row in reversed(rows):
        fallbacks = [reason for status, reason in row_decisions(row) if status == "fallback"]
        if not fallbacks:
            break
        streak.append(str(row.get("decision_bar")))
        reasons.extend(str(reason) for reason in fallbacks)
    last = rows[-1]
    scale = last.get("account_scale", last.get("scale"))
    scale_text = f", scale {float(scale):.3f}" if isinstance(scale, (int, float)) else ""
    if len(streak) >= VOL_SCALE_FALLBACK_STREAK_WARN:
        return _result(
            "vol-scale-fallback",
            False,
            f"{len(streak)} consecutive refresh row(s) fell back, {streak[-1]} through {streak[0]} (reasons "
            f"{sorted(set(reasons))}){scale_text}: the book is held at min(1, last good scale), below its rule. "
            "Check the bar panel for stale or flat bars and the window length (prism.live.vol_sizing)",
            warn=True,
        )
    statuses = sorted({status for status, _ in row_decisions(last)})
    note = f"; the last refresh fell back ({sorted(set(reasons))}), which is not yet a streak" if streak else ""
    return _result(
        "vol-scale-fallback",
        True,
        f"{len(rows)} refresh row(s); last {last.get('decision_bar')} status {statuses}{scale_text}{note}",
    )


def check_cash_holding_drift(run_dir: Path) -> CheckResult:
    """The T-bill holding against its target, from the newest record in ``cash_holding.jsonl``.

    The cycle records the holding's decision on every session it decides,
    refresh or not (``prism.live.cash_holding``), and moves the holding to its
    target when it drifts more than the band, by at most the per-order bound
    per session. Two things warn. After the decision, the holding still sits
    more than the band from its target: it is being built or sold over several
    sessions, which is expected for a few sessions after a start or a large
    change in the long side, and means its orders are not filling if it
    persists. And, when the previous record is from the previous session, the
    holding marked now differs by more than the band from what that session
    decided.
    """
    from prism.io import sessions
    from prism.live.cash_holding import LEDGER_FILENAME
    from prism.live.ledgers import read_rows

    path = run_dir / LEDGER_FILENAME
    rows = [row for row in read_rows(path) if "decision_bar" in row]
    if not rows:
        return _result("cash-holding-drift", True, f"no record in {path} — the T-bill holding is not in use")
    bar = str(rows[-1]["decision_bar"])
    record = rows[-1]
    try:
        symbol = str(record["symbol"])
        held, target, weight, band = (float(record[key]) for key in ("held_weight", "target_weight", "weight", "band"))
        action = str(record.get("action"))
    except (KeyError, TypeError, ValueError) as exc:
        return _result("cash-holding-drift", False, f"{path}: {bar} record unreadable ({exc})", warn=True)
    residual = abs(weight - target)
    detail = (
        f"{symbol} at {bar}: held {held:.4f}, target {target:.4f}, decided {weight:.4f} of equity "
        f"({action}, band {band:.4f})"
    )
    problems = []
    if residual > band + 1e-12:
        problems.append(
            f"after the decision the holding sits {residual:.4f} from its target, more than the band: it is moving "
            "toward it within the per-order bound, expected for a few sessions after a start or a large change in the "
            "long side, and a sign its orders are not filling if it persists"
        )
    if len(rows) > 1:
        previous_bar = str(rows[-2]["decision_bar"])
        try:
            consecutive = sessions.next_session(date.fromisoformat(previous_bar[:10])) == date.fromisoformat(bar[:10])
            decided_before = float(rows[-2]["weight"])
        except (KeyError, TypeError, ValueError):
            consecutive, decided_before = False, float("nan")
        if consecutive and abs(held - decided_before) > band:
            problems.append(
                f"the holding marked at {bar} ({held:.4f}) differs by more than the band from the "
                f"{decided_before:.4f} decided at {previous_bar}: that order did not fill as decided, or equity "
                "moved by more than the band"
            )
    if problems:
        return _result("cash-holding-drift", False, detail + " — " + "; ".join(problems), warn=True)
    return _result("cash-holding-drift", True, detail)


def check_tranche_state(run_dir: Path) -> CheckResult:
    """The integrity of a tranched book's persisted tranche states (``prism.live.tranches.inspect_tranche_state``).

    Also compares the count and base offset with the newest tranche record in
    ``targets.jsonl``. A state that cannot be loaded is ``loop-state``'s FAIL;
    here it is a WARN that defers to it.
    """
    from prism.live.ledgers import read_rows
    from prism.live.state import StateStore
    from prism.live.tranches import TARGETS_TRANCHES_KEY, inspect_tranche_state

    state_path = run_dir / "state.json"
    if not state_path.exists():
        return _result("tranche-state", True, f"{state_path} absent — no tranche state to check")
    try:
        state = StateStore(state_path).load()
    except ValueError as exc:
        return _result("tranche-state", False, f"cannot read {state_path} (see loop-state): {exc}", warn=True)
    report = inspect_tranche_state(state)
    if report is None:
        return _result("tranche-state", True, "no tranche state — the single book")
    problems = list(report.problems)
    records = [
        row[TARGETS_TRANCHES_KEY]
        for row in read_rows(run_dir / "targets.jsonl")
        if isinstance(row.get(TARGETS_TRANCHES_KEY), dict)
    ]
    if records:
        newest = records[-1]
        if newest.get("count") != report.count:
            problems.append(
                f"targets.jsonl's newest tranche record counts {newest.get('count')}, the state {report.count}"
            )
        if report.base_offset is not None and newest.get("base_offset") != report.base_offset:
            problems.append(
                f"targets.jsonl's newest tranche record has base offset {newest.get('base_offset')}, the state "
                f"{report.base_offset}"
            )
    gross = "n/a" if report.gross is None else f"{report.gross:.3f}"
    last_refresh = None if state is None else state.last_refresh_bar
    detail = (
        f"{report.count} tranche(s) of {report.book}, {report.built} built, base offset {report.base_offset}, cycle "
        f"{report.cycle} sessions, summed sub-target gross {gross}, last refresh {last_refresh}"
    )
    if problems:
        return _result("tranche-state", False, detail + " — " + "; ".join(problems), warn=True)
    return _result("tranche-state", True, detail)


def check_deployment_profile(run_dir: Path) -> CheckResult:
    """The deployment seed (``deployment.json``): readable and private when present, warned when a mode needs it.

    A tranched book is recognised from its tranche states, a seeded
    volatility-scaled book from ``sizing.jsonl`` rows on the seeded grid
    (``schedule = "grid"``). A seeded single book without the volatility scale
    leaves no record of its mode beside the profile itself, so its missing
    profile is not seen here; the loop refuses that run before any venue read
    (``prism.live.deployment``). The seed is never printed.
    """
    import stat

    from prism.live.deployment import (
        DEPLOYMENT_FILENAME,
        DeploymentProfileError,
        read_deployment,
    )
    from prism.live.ledgers import read_rows
    from prism.live.state import StateStore
    from prism.live.tranches import inspect_tranche_state

    path = run_dir / DEPLOYMENT_FILENAME
    if path.exists():
        try:
            profile = read_deployment(run_dir)
        except DeploymentProfileError as exc:
            return _result(
                "deployment-profile",
                False,
                f"{exc} — a seeded session will refuse this run directory",
                warn=True,
            )
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            return _result(
                "deployment-profile",
                False,
                f"{path} has mode {mode:o}, not 600: the seed determines every draw this deployment makes (the "
                f"phase offset itself is recorded in plain text in the log and ledgers); chmod 600 {path}",
                warn=True,
            )
        return _result("deployment-profile", True, f"{path}: created {profile.created_bar}, mode 600 (seed not shown)")

    evidence: list[str] = []
    seeded_offset = False
    try:
        state = StateStore(run_dir / "state.json").load() if (run_dir / "state.json").exists() else None
    except ValueError:
        state = None  # loop-state reports an unreadable state
    report = inspect_tranche_state(state)
    if report is not None:
        evidence.append(f"state.json holds {report.count} tranche state(s) at base offset {report.base_offset}")
        seeded_offset = bool(report.base_offset)
    grid = [row for row in read_rows(run_dir / "sizing.jsonl") if row.get("schedule") == "grid"]
    if grid:
        evidence.append(f"sizing.jsonl records {len(grid)} refresh(es) on the seeded grid")
        seeded_offset = True
    if not evidence:
        return _result(
            "deployment-profile",
            True,
            f"{path} absent — no seeded or tranched mode is recorded in this run directory",
        )
    if seeded_offset:
        consequence = (
            "only a seeded phase puts the book there, so the next session with the same flags will be refused "
            "(DeploymentProfileMissing); restore the profile from the backup of this run directory"
        )
    else:
        consequence = (
            "at base offset 0 the book is either unseeded, on the epoch grid every unseeded tranched deployment "
            "shares (with as many tranches as sessions in the cycle the base only relabels the tranches), or "
            "seeded at offset 0 with its profile lost"
        )
    return _result(
        "deployment-profile",
        False,
        f"{path} absent, but {'; '.join(evidence)}: {consequence}",
        warn=True,
    )


def check_nightly_log(run_dir: Path, *, today: date | None = None) -> CheckResult:
    """The wrapper's last verdict — including a *stale success*.

    Parses the `… EXIT <rc>` / `… SKIP` lines the ops wrappers append. Any
    nonzero verdict FAILs (item 6: alert on any nonzero nightly result). A zero
    verdict that is days old also FAILs, because "the last thing that ran
    succeeded" and "the loop is running" are different claims and the outage
    lived in the gap between them.
    """
    today = today or date.today()
    log = run_dir / "nightly.log"
    if not log.exists():
        return _result(
            "nightly-log",
            False,
            f"{log} absent — the loop has never run here, or the wrapper writes elsewhere "
            "(ops/paper_loop_nightly.sh owns this log)",
            warn=True,
        )
    verdicts = [
        m for m in (_LOG_VERDICT.match(ln.strip()) for ln in log.read_text(encoding="utf-8").splitlines()) if m
    ]
    if not verdicts:
        return _result(
            "nightly-log", False, f"{log}: no EXIT/SKIP verdict lines — a truncated or foreign log", warn=True
        )
    last = verdicts[-1]
    kind = last.group("verdict")
    code = int(last.group("code") or 0)
    try:
        stamp = datetime.strptime(last.group("ts"), "%Y-%m-%dT%H:%M:%S%z").date()
        age: int | None = weekdays_since(stamp, today)
    except ValueError:
        stamp, age = None, None
    when = f"{stamp.isoformat()} ({age} weekday(s) ago)" if stamp else "unparseable timestamp"
    consecutive = 0
    for match in reversed(verdicts):
        if match.group("verdict") == "EXIT" and int(match.group("code") or 0) == 0:
            break
        consecutive += 1
    if kind == "SKIP":
        return _result("nightly-log", False, f"{log}: last run SKIPped at {when} — credentials not armed", warn=True)
    if code != 0:
        return _result(
            "nightly-log",
            False,
            f"{log}: last run EXIT {code} at {when}; {consecutive} consecutive failed session(s) — "
            "read the traceback at the tail of the log",
        )
    if age is not None and age >= LEDGER_STALE_FAIL_WEEKDAYS:
        return _result(
            "nightly-log",
            False,
            f"{log}: last run succeeded, but at {when} — a STALE success. The scheduler is not "
            "firing, or it is firing something that writes elsewhere",
        )
    if age is not None and age >= LEDGER_STALE_WARN_WEEKDAYS:
        return _result("nightly-log", False, f"{log}: last run EXIT 0 at {when} — a session was missed", warn=True)
    return _result("nightly-log", True, f"{log}: last run EXIT 0 at {when}")


def _repo_ops_file(name: str) -> Path | None:
    """Locate the repo's ops/ wrapper from this installation, or None.

    Editable install: three parents up from this file is the repo root. Slim
    or wheel install: the wrappers cd into the repo before invoking python,
    so the cwd candidate covers it.
    """
    for base in (Path.cwd(), Path(__file__).resolve().parents[3]):
        candidate = base / "ops" / name
        if candidate.exists():
            return candidate.resolve()
    return None


def check_wrapper_provenance(run_dir: Path) -> CheckResult:
    """Was the last session driven by the repo's ops/ wrapper, byte-for-byte?

    Path identity, not content hashing: the stamp records the physical path
    (`readlink -f`) of the executing file, so a symlinked deployment resolves
    to the repo file itself and *cannot* drift, while a copy — however
    faithful today — stamps its own path and FAILs. An unstamped session
    means a pre-provenance wrapper ran: exactly the condition under which the
    committed alert layer silently never executed (found 2026-07-31).
    """
    problems: list[str] = []
    oks: list[str] = []
    unverifiable: list[str] = []
    for log_name, marker, ops_name in (
        ("nightly.log", "RUN", "paper_loop_nightly.sh"),
        ("sweep.log", "SWEEP", "paper_sweep_morning.sh"),
    ):
        log = run_dir / log_name
        required = marker == "RUN"  # the trading path; a fresh dir may have no sweep yet
        if not log.exists():
            if required:
                unverifiable.append(f"{log_name} absent — no session to verify")
            continue
        entry = re.compile(rf"^\S+\s+{marker}:")
        lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if entry.match(ln)]
        if not lines:
            if required:
                unverifiable.append(f"{log_name}: no {marker} lines — no session to verify")
            continue
        stamp = _WRAPPER_STAMP.search(lines[-1])
        if stamp is None:
            problems.append(
                f"{log_name}: last {marker} is unstamped — a pre-provenance copy ran, "
                f"not ops/{ops_name}; repoint the scheduler (or symlink) at the repo file"
            )
            continue
        expected = _repo_ops_file(ops_name)
        if expected is None:
            unverifiable.append(f"ops/{ops_name} not locatable from this install")
            continue
        stamped = Path(stamp.group("path"))
        try:
            stamped = stamped.resolve()
        except OSError:
            pass
        if stamped != expected:
            problems.append(
                f"{log_name}: ran {stamp.group('path')} but the repo wrapper is {expected} — "
                "a drifted copy is scheduled"
            )
        else:
            oks.append(f"{log_name} ran ops/{ops_name}")
    if problems:
        return _result("wrapper-provenance", False, "; ".join(problems))
    if unverifiable and not oks:
        return _result("wrapper-provenance", False, "; ".join(unverifiable), warn=True)
    return _result("wrapper-provenance", True, "; ".join(oks))


def check_artifact_sync(run_dir: Path, *, today: date | None = None) -> CheckResult:
    """Did the last off-box backup land? Read from ``<run_dir>/sync_status.json``.

    ``ops/artifacts_sync.sh`` writes that file after every attempt, success or
    failure, with the stage it reached. The row exists because its predecessor
    reported failure only by a log line and an exit code nobody read: from
    2026-09-25 every push was refused by the remote's file-size limit, the
    script exited 0 regardless, and the ledgers' last off-box copy stayed at
    2026-08-14. A failed report FAILs; an absent one warns, because the file
    only exists once the versioned script has run; a stale success warns,
    because the sync runs every nightly and a day without a report is a day
    the nightly did not get that far.
    """
    today = today or date.today()
    status_path = run_dir / "sync_status.json"
    if not status_path.exists():
        return _result(
            "artifact-sync",
            False,
            f"{status_path} absent — no backup has reported since this check was added "
            "(ops/artifacts_sync.sh writes it; the nightly runs it after the doctor)",
            warn=True,
        )
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        at = datetime.fromisoformat(str(status["at"]))
        ok = bool(status["ok"])
        stage = str(status.get("stage", "?"))
        detail = str(status.get("detail", ""))
        head = status.get("head")
    except (ValueError, KeyError, TypeError) as exc:
        return _result("artifact-sync", False, f"{status_path}: unreadable status ({exc})")
    when = at.astimezone().date() if at.tzinfo else at.date()
    age = weekdays_since(when, today)
    stamp = f"{str(status['at'])} ({age} weekday(s) ago)"
    if not ok:
        return _result(
            "artifact-sync",
            False,
            f"backup FAILED at stage {stage!r}, reported {stamp}: {detail} — the ledgers have no off-box "
            "copy since the last successful push (docs/operations.md, 2026-09-30)",
        )
    if age > SYNC_STALE_WARN_WEEKDAYS:
        return _result(
            "artifact-sync",
            False,
            f"last successful backup {stamp}, head {head} — stale: the sync runs every nightly, so a "
            "nightly since then did not reach it",
            warn=True,
        )
    return _result("artifact-sync", True, f"backup at {head} reported {stamp}: {detail}")


def check_alert_transport(env: Mapping[str, str]) -> CheckResult:
    """Can any alert leave this machine? Presence only — values are never printed.

    The three notifications ``ops/_alert.sh`` fires are all on-box: a marker
    file, a log line, an exit code. Each needs someone to look. ``PRISM_ALERT_CMD``
    is the push transport for the alerts the wrappers raise; the heartbeat URLs
    are the dead-man's monitor that reports a scheduler which never fires at
    all, which no in-session check can. With neither set, the 2026-08-15 →
    2026-09-25 silence repeats on the next disabled task.
    """
    names = ("PRISM_ALERT_CMD", "PRISM_HEARTBEAT_URL", "PRISM_HEARTBEAT_URL_SWEEP")
    present = [name for name in names if env.get(name)]
    if not env.get("PRISM_ALERT_CMD") and not env.get("PRISM_HEARTBEAT_URL"):
        return _result(
            "alert-transport",
            False,
            "neither PRISM_ALERT_CMD nor PRISM_HEARTBEAT_URL is set — no alert can leave this machine; the "
            "ALERT marker, the log line and the scheduler's exit code are all on-box. Set them in .env "
            "(ops/alert_curl.sh, ops/_alert.sh; docs/operations.md, 2026-09-30)",
            warn=True,
        )
    return _result("alert-transport", True, f"{', '.join(present)} set")


def check_account_book(
    held: Mapping[str, float],
    universe: Sequence[str],
    state: object | None,
    *,
    run_dir: Path,
) -> CheckResult:
    """Does the venue's book belong to this run directory? (pure; caller reads)

    Three distinct disagreements, in descending severity:

    1. **Unattributed book** — the venue holds positions and this run directory
       persists none. The loop can now trade through it (broker truth drives the
       fetch universe), but the NAV and session ledgers of the book being marked
       live in some *other* directory, so this directory's evidence stream is
       not the account's history. This is the 2026-07-23 condition, and it is a
       FAIL: reattach the run directory deliberately, do not let a new evidence
       stream silently adopt an old book.
    2. **Divergence** — both books exist and disagree. The loop's reconcile step
       adopts broker truth loudly; worth eyes, not a stop.
    3. **Off-universe holdings** — held names the configured universe no longer
       contains (an index leaver). Valuation/exit-only and handled, but reported
       so an unexpected symbol — a position no book of this program decided —
       cannot hide among them.
    """
    held_names = {str(s) for s, q in held.items() if float(q) != 0.0}
    persisted = (
        {str(s) for s, q in getattr(state, "positions", {}).items() if float(q) != 0.0} if state is not None else set()
    )
    off_universe = sorted(held_names - {str(s).upper() for s in universe})
    trailer = f"; {len(off_universe)} off-universe (valuation/exit only): {off_universe[:10]}" if off_universe else ""

    if held_names and not persisted:
        return _result(
            "alpaca-account-book",
            False,
            f"the venue holds {len(held_names)} position(s) but {run_dir}/state.json persists "
            f"none — this run directory has no record of the account's book. The cycle can run "
            f"(broker truth drives valuation), but its ledgers are NOT this account's history: "
            f"reattach the run directory that decided the book, or retire it deliberately{trailer}",
        )
    if not held_names:
        return _result("alpaca-account-book", True, "venue holds no positions — flat account" + trailer)
    if held_names != persisted:
        only_venue = sorted(held_names - persisted)[:10]
        only_state = sorted(persisted - held_names)[:10]
        return _result(
            "alpaca-account-book",
            False,
            f"venue {len(held_names)} position(s) vs persisted {len(persisted)}: "
            f"venue-only {only_venue}, state-only {only_state} — the loop adopts broker truth "
            f"at reconcile, but this much drift means sessions were lost{trailer}",
            warn=True,
        )
    return _result(
        "alpaca-account-book", True, f"{len(held_names)} position(s), reconciled against state.json" + trailer
    )


def check_holdings_priceable(held: Mapping[str, float], prices: Mapping[str, float | None]) -> CheckResult:
    """Can every held position be marked? — the mark step's precondition.

    ``run_daily_cycle`` values *every* held position at today's close and
    refuses loudly otherwise (``_require_price``, N7). That refusal is correct
    and it is also fatal to the whole cycle, so the question is worth asking
    before the loop spends minutes fetching a 500-name panel to discover it.
    """
    held_names = sorted(str(s) for s, q in held.items() if float(q) != 0.0)
    if not held_names:
        return _result("alpaca-holdings-priceable", True, "no positions to mark")
    unpriceable = [
        s
        for s in held_names
        if prices.get(s) is None or not (isinstance(prices.get(s), (int, float)) and float(prices[s] or 0) > 0)
    ]
    if unpriceable:
        return _result(
            "alpaca-holdings-priceable",
            False,
            f"{len(unpriceable)}/{len(held_names)} held position(s) have no usable bar: "
            f"{unpriceable[:10]} — the mark step will refuse the cycle (N7). Restore the bar "
            "source for these names or close the positions at the venue",
        )
    return _result("alpaca-holdings-priceable", True, f"all {len(held_names)} held position(s) priceable")


# ---------------------------------------------------------------------------
# Network probes
# ---------------------------------------------------------------------------


def check_alpaca_account(universe_file: Path | None = None, run_dir: Path | None = None) -> list[CheckResult]:
    """Account reads → four answers: keys, margin cushion, book attribution, priceability.

    The account read and the held-book bar probe are both free and unmetered at
    Alpaca, so the reconciliation costs nothing beyond the credential probe it
    replaces — the reason "8 pass, 0 fail" was reported over a fatal account is
    that nobody asked, not that asking was expensive.
    """
    from prism.live import AlpacaBarSource, AlpacaBroker
    from prism.live.state import StateStore

    try:
        broker = AlpacaBroker.from_env()
        cash = broker.cash()
        positions = broker.positions()
    except Exception as exc:  # any layer: env, auth, transport
        return [_result("alpaca-account", False, f"account read failed: {exc}")]
    results = [_result("alpaca-account", True, f"cash {cash:.2f}, {len(positions)} open positions")]
    # Read-only margin telemetry: one more account read, a WARN at most.
    results.append(check_live_margin(broker))
    if run_dir is None:
        return results

    universe = _read_universe(universe_file)
    state = None
    try:
        state = StateStore(run_dir / "state.json").load()
    except ValueError:
        pass  # check_loop_state already FAILs on a corrupt state file
    results.append(check_account_book(positions, universe, state, run_dir=run_dir))

    held = sorted(positions)
    if not held:
        results.append(check_holdings_priceable(positions, {}))
        return results
    try:
        source = AlpacaBarSource.from_env()
        start = (date.today() - timedelta(days=14)).isoformat()
        frames = source.fetch_batch(held, interval="1d", start_date=start)
        prices: dict[str, float | None] = {}
        for symbol in held:
            frame = frames.get(symbol)
            close = None
            if frame is not None and not frame.empty and "close" in frame.columns:
                close = float(frame["close"].iloc[-1])
            prices[symbol] = close
    except Exception as exc:  # noqa: BLE001 — a probe failure is not a book failure
        results.append(_result("alpaca-holdings-priceable", False, f"bar probe failed: {exc}", warn=True))
        return results
    results.append(check_holdings_priceable(positions, prices))
    return results


def check_twelvedata_quote(api_key: str | None, cache_dir: Path) -> CheckResult:
    """Network probe: one bar for SPY (1 credit of the daily 800).

    WARN, never FAIL. The live loop reads Alpaca's own feed by default, so a
    dead research-spine key does not stop a paper session — and this doctor's
    exit code now drives the nightly alert (ops/paper_loop_nightly.sh). An alert
    that fires on something the live path does not use is an alert the operator
    learns to ignore, which is the failure mode P0 exists to remove.
    """
    from prism.io.loader import DataLoader

    loader = DataLoader(api_key=api_key, cache_dir=cache_dir)
    if not loader.api_key:
        return _result("twelvedata-quote", False, "no key to probe", warn=True)
    df = loader.fetch_historical_data("SPY", "1d", force_refresh=True)
    if df.empty:
        return _result(
            "twelvedata-quote",
            False,
            "SPY fetch returned empty — key invalid, plan exhausted, or vendor down. Research "
            "spine only; the live loop's default bar source is Alpaca",
            warn=True,
        )
    return _result("twelvedata-quote", True, f"SPY bars through {df.index[-1].date()}")


def run_checks(
    *,
    run_dir: Path,
    universe_file: Path | None,
    data_dir: Path | None = None,
    workspace: str | None = None,
    env: dict[str, str] | None = None,
    network: bool = False,
    today: date | None = None,
    mode: str = "paper",
) -> list[CheckResult]:
    """Run the offline checks, plus the venue/vendor probes when ``network``.

    ``data_dir`` defaults to the resolved workspace's ``data/`` -- the same
    bar cache the loop writes. The Twelve Data key comes from ``env`` (the
    process environment by default, after the command loaded ``.env``), so
    the probe and the ``twelvedata-key`` check always report the same key.
    """
    data_dir = data_dir or resolve_workspace(workspace).data
    env = dict(os.environ) if env is None else env
    results = [check_python()]
    results.extend(check_env_credentials(env, mode=mode))
    results.append(check_alert_transport(env))
    results.append(check_authorization(run_dir, mode=mode))
    results.append(check_universe_file(universe_file, today=today))
    results.append(check_data_dir(data_dir))
    results.append(check_loop_state(run_dir))
    results.append(check_kill_switch(run_dir))
    # Health: the durable record of work done. Offline, because a dark loop must
    # be detectable without credentials — the alerting path cannot depend on the
    # same venue reachability whose absence it has to report.
    results.append(check_equity_ledger(run_dir, today=today))
    results.append(check_missed_sessions(run_dir, today=today))
    results.append(check_session_table(today=today))
    results.append(check_nightly_log(run_dir, today=today))
    results.append(check_wrapper_provenance(run_dir))
    results.append(check_regime_clock(run_dir, today=today))
    results.append(check_panel_record(run_dir, today=today))
    results.append(check_shadow_record(run_dir, today=today))
    results.append(check_artifact_sync(run_dir, today=today))
    # Opt-in book features (off by default): each PASSes when not in use and
    # only ever WARNs, so the nightly verdict of a book without them is unchanged.
    results.append(check_margin_cushion(run_dir, today=today))
    results.append(check_vol_scale_fallback(run_dir))
    results.append(check_cash_holding_drift(run_dir))
    results.append(check_tranche_state(run_dir))
    results.append(check_deployment_profile(run_dir))
    if network:
        results.extend(check_alpaca_account(universe_file, run_dir))
        results.append(check_twelvedata_quote(env.get("TWELVEDATA_API_KEY"), data_dir))
    return results


def main(argv: list[str] | None = None) -> int:
    announce_alias("prism-doctor", "prism check")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Preflight checks for the prism paper loop.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Loop run directory. Default: the directory runs/ACTIVE_RUN_DIR names, relative to the current "
        "directory; refused with exit 1 when the switch is missing or empty.",
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=Path("data/universe/sp500_current.txt"),
        help="Universe file the loop will trade (default: the nightly's).",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Also probe Alpaca (free account read) and Twelve Data (1 credit).",
    )
    parser.add_argument(
        "--account",
        choices=("paper", "live"),
        default="paper",
        help="The mode the checks are run for (§3.10). The endpoint check is a refusal under the "
        "wrong mode; `prism run --account` has no default, and this legacy alias defaults to paper "
        "because it is the paper loop's doctor.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Writable root whose data/ is the bar cache to check "
        "(default: $PRISM_WORKSPACE, else the current directory).",
    )
    args = parser.parse_args(argv)

    # Credentials come from the project's .env, loaded here rather than at
    # import (spec §3.1): the credential checks below read os.environ and must
    # see what the loop will see.
    load_env()

    # The same rule as the wrappers (ops/, 2026-09-30): a missing or empty switch
    # is refused, never resolved to a retired book or to the current directory,
    # because a green report about the wrong directory is worse than no report.
    run_dir = args.run_dir
    if run_dir is None:
        active = Path("runs") / ACTIVE_RUN_DIR_FILENAME
        content = active.read_text(encoding="utf-8").strip() if active.is_file() else ""
        if not content:
            print(
                f"run-dir  refused: {active} is missing or empty, so nothing names the live book. "
                "Pass --run-dir, or write the run directory's path (relative to the current directory "
                "or absolute) as the file's single line (docs/operations.md, 2026-09-30)."
            )
            return 1
        run_dir = Path(content)

    results = run_checks(
        run_dir=run_dir,
        universe_file=args.universe_file,
        workspace=args.workspace,
        network=args.network,
        mode=args.account,
    )
    # Name the run directory: "which book did you just certify as healthy?" was
    # the unasked question behind the 2026-07-23 outage (ACTIVE_RUN_DIR had been
    # pointed at a directory that had never completed a cycle).
    print(f"run-dir  {run_dir}")
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"{r.status:4}  {r.name:<{width}}  {r.detail}")
    failed = [r for r in results if r.status == "FAIL"]
    warned = [r for r in results if r.status == "WARN"]
    print(
        f"\n{len(results) - len(failed) - len(warned)} pass, {len(warned)} warn, {len(failed)} fail"
        + ("" if args.network else "  (offline checks only; --network reconciles the venue book)")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
