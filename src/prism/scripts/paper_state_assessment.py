"""Assessment of the paused paper deployment's recorded state (engine spec §5, stage 0).

The paper loop went dark when the scheduler stopped firing; the code did not
fail. Before anyone restarts it, the run directory has to be read for what it
actually records rather than what it is remembered to say, and the interruption
has to be written down as a gap. That is this script's whole job.

It is a reader. The run directory arrives as an absolute path argument and is
never derived from a package-relative constant — a run directory pointed at the
wrong book is the 2026-07-23 outage, and a default is how that happens. Nothing
is written into the run directory; the report goes to stdout, or to ``--out``.
It runs fully offline: with no credentials and no ``--network`` it completes and
says the broker comparison was not performed.

Three things it refuses to do, because each would be a fabrication:

* **Invent a session.** The sessions since the last completed one come from the
  bundled table (``prism.io.sessions``) and are reported as a gap, each one "no
  session ran" — never a clean session, never a fill.
* **Count a session that never ran.** The regime clock is the doctor's
  (``prism.scripts.doctor.regime_streak``): a session with no ``regime.jsonl``
  row is not clean, so a gap restarts the count (docs/regime_step.md §4; owner
  decision 2026-09-25).
* **Guess a schema.** An unrecognized ``state.json`` ``schema_version`` stops the
  assessment; a best-effort read of an unknown book is the silent wrong answer.

    python -m prism.scripts.paper_state_assessment /abs/path/runs/paper_loop_momentum2
    python -m prism.scripts.paper_state_assessment /abs/run-dir --out assessment.txt
    python -m prism.scripts.paper_state_assessment /abs/run-dir --network   # + venue book

Exit 0 when the assessment completes, 2 when it cannot read what it was asked to
read (a missing or malformed ``state.json``, an unknown schema, a run directory
whose dates the session table does not cover).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from prism.io import sessions
from prism.scripts.doctor import CLEAN_SESSIONS_REQUIRED, regime_streak
from prism.workspace import load_env

EXIT_OK = 0
EXIT_UNREADABLE = 2

# What ``prism.live.state.StateStore.load`` will accept for this file: v1 is
# migrated to v2 on load, anything else is a hard stop
# (``prism/live/state.py:57-75``). This reader accepts exactly that set rather
# than importing it, because importing ``prism.live.state`` executes
# ``prism/live/__init__.py`` and drags the Alpaca broker chain into an offline
# reader. (Since the workspace correction of spec §3.1, no import anywhere in
# the package loads ``.env`` or creates a directory, so the offline path here
# holds no credential at all until ``--network`` asks for one.)
# tests/test_paper_state_assessment.py pins the two schema sets together so a
# bump there fails here loudly instead of drifting.
SUPPORTED_STATE_SCHEMA_VERSIONS = (1, 2, 3)
CURRENT_STATE_SCHEMA_VERSION = 3

# Every ledger the loop appends, in the order the report reads them. All six key
# their rows on ``decision_bar``.
LEDGER_FILES = (
    "equity.jsonl",
    "targets.jsonl",
    "fills.jsonl",
    "unfilled.jsonl",
    "concordance.jsonl",
    "regime.jsonl",
)

# The wrapper's provenance stamp and verdict lines, as ops/*.sh writes them.
# The verdict pattern deliberately anchors on the *second* field so the doctor's
# own trailing `DOCTOR EXIT 0` is not mistaken for the loop's verdict.
_RUN_LINE = re.compile(r"^(?P<ts>\S+)\s+RUN:\s+(?P<fields>.*)$")
_VERDICT_LINE = re.compile(r"^(?P<ts>\S+)\s+(?P<verdict>EXIT|SKIP)\b\D*(?P<code>\d+)?")
_DOCTOR_HEADER = re.compile(r"^(?P<ts>\S+)\s+DOCTOR:\s*$")
_DOCTOR_SUMMARY = re.compile(r"^(?P<pass>\d+) pass, (?P<warn>\d+) warn, (?P<fail>\d+) fail")
_DOCTOR_CHECK = re.compile(r"^(?P<status>PASS|WARN|FAIL)\s+(?P<name>\S+)\s+(?P<detail>.*)$")
_COMMIT_FIELD = re.compile(r"\bcommit=(?P<commit>\S+)")


class AssessmentError(RuntimeError):
    """The assessment could not read something it was asked to read.

    Distinct from a finding: a finding is written into the report, this stops
    the run. Raised for a missing or malformed ``state.json``, an unknown state
    schema, and dates the session table does not cover — every case where
    continuing would mean reporting a number nobody can stand behind.
    """


@dataclass(frozen=True)
class StateFacts:
    """What ``state.json`` records about the book at the moment it went dark."""

    schema_version: int
    cash: float
    positions: dict[str, float]
    pending_order_ids: tuple[str, ...]
    pending_order_count: int
    last_refresh_bar: str | None
    last_settled_bar: str | None
    pending_decision_bar: str | None

    @property
    def zero_positions(self) -> tuple[str, ...]:
        """Symbols carried at exactly zero shares — a book entry that should have been dropped."""
        return tuple(sorted(symbol for symbol, qty in self.positions.items() if qty == 0))


@dataclass(frozen=True)
class LedgerFacts:
    """One JSONL ledger's shape: how many rows, how damaged, and what it spans."""

    name: str
    present: bool
    rows: int
    malformed: int
    first_bar: str | None
    last_bar: str | None
    extra: str = ""


@dataclass(frozen=True)
class NightlyFacts:
    """The last wrapper invocation and the last doctor block in ``nightly.log``."""

    present: bool
    last_run_ts: str | None = None
    last_run_fields: str | None = None
    last_run_commit: str | None = None
    commits_seen: tuple[str, ...] = ()
    last_verdict: str | None = None
    last_verdict_code: str | None = None
    last_verdict_ts: str | None = None
    doctor_ts: str | None = None
    doctor_summary: str | None = None
    doctor_nonpass: tuple[str, ...] = ()


@dataclass(frozen=True)
class GapFacts:
    """The sessions that were expected after the last completed one and did not run."""

    asof: dt.date
    last_completed: dt.date | None
    last_completed_source: str
    horizon_end: dt.date | None
    missed: tuple[dt.date, ...] = ()
    closures: tuple[tuple[dt.date, str], ...] = ()
    disagreements: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegimeClock:
    """The doctor's precondition-(b) clock, read the way the doctor reads it."""

    rows: int
    streak: int
    last_bar: str | None
    trailing_unclean: bool
    run: int = 0
    missed: int = 0


@dataclass(frozen=True)
class GitFacts:
    """The distance between the commit the last session traded and the checkout that would restart it."""

    available: bool
    reason: str | None = None
    repo: Path | None = None
    head: str | None = None
    traded_commit: str | None = None
    ancestor: bool | None = None
    commits: tuple[str, ...] = ()
    commits_only_on_traded: int = 0
    diffstats: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# readers


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AssessmentError(f"{path} is absent; the assessment cannot describe a book it cannot read")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AssessmentError(f"{path} is unreadable: {exc}") from exc
    except ValueError as exc:
        raise AssessmentError(f"{path} is not parseable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssessmentError(f"{path} holds {type(payload).__name__}, not a JSON object")
    return payload


def read_state(run_dir: Path) -> StateFacts:
    """The durable loop state, or raise if its schema is not one the loop would load."""
    payload = _read_json_object(run_dir / "state.json")
    version = payload.get("schema_version")
    if version not in SUPPORTED_STATE_SCHEMA_VERSIONS:
        raise AssessmentError(
            f"{run_dir / 'state.json'} has schema_version {version!r}, not one of "
            f"{list(SUPPORTED_STATE_SCHEMA_VERSIONS)} (prism/live/state.py:57-75). The assessment "
            "refuses to read a book whose layout it does not know."
        )
    # Version 3 moved the book inside ``portfolio`` (§3.11): positions carry a
    # mark and a valuation status, cash and the pending orders live there too.
    # Versions 1 and 2 keep them at the top level. The reader reads whichever
    # layout the document declares rather than guessing from the keys present.
    book = payload.get("portfolio") if int(version) >= 3 else payload
    if not isinstance(book, dict):
        raise AssessmentError(
            f"{run_dir / 'state.json'} declares schema_version {version!r} and carries "
            f"portfolio={type(book).__name__}, not an object; the book cannot be read"
        )
    positions = book.get("positions") or {}
    if not isinstance(positions, dict):
        raise AssessmentError(f"{run_dir / 'state.json'} positions is {type(positions).__name__}, not an object")
    pending = book.get("pending_orders") or []
    if not isinstance(pending, list):
        raise AssessmentError(f"{run_dir / 'state.json'} pending_orders is {type(pending).__name__}, not a list")
    order_ids = tuple(
        str(order.get("client_order_id", "<no client_order_id>")) for order in pending if isinstance(order, dict)
    )
    return StateFacts(
        schema_version=int(version),
        cash=float(book.get("cash", 0.0)),
        positions={str(k): _shares_of(v) for k, v in positions.items()},
        pending_order_ids=order_ids,
        pending_order_count=len(pending),
        last_refresh_bar=_opt_str(payload.get("last_refresh_bar")),
        last_settled_bar=_opt_str(payload.get("last_settled_bar")),
        pending_decision_bar=_opt_str(payload.get("pending_decision_bar")),
    )


def _shares_of(value: Any) -> float:
    """Signed shares from either layout: a bare number (v1, v2) or a v3 position object."""
    if isinstance(value, Mapping):
        return float(value["shares"])
    return float(value)


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def read_ledger(path: Path) -> tuple[LedgerFacts, list[dict[str, Any]]]:
    """A JSONL ledger's rows and its shape. Malformed lines are counted, never dropped in silence."""
    name = path.name
    if not path.exists():
        return LedgerFacts(name=name, present=False, rows=0, malformed=0, first_bar=None, last_bar=None), []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AssessmentError(f"{path} is unreadable: {exc}") from exc
    rows: list[dict[str, Any]] = []
    malformed = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            malformed += 1
    bars = [str(row["decision_bar"]) for row in rows if "decision_bar" in row]
    facts = LedgerFacts(
        name=name,
        present=True,
        rows=len(rows),
        malformed=malformed,
        first_bar=min(bars) if bars else None,
        last_bar=max(bars) if bars else None,
    )
    return facts, rows


def read_nightly_log(path: Path) -> NightlyFacts:
    """The last RUN stamp, the last wrapper verdict, and the last doctor block."""
    if not path.exists():
        return NightlyFacts(present=False)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise AssessmentError(f"{path} is unreadable: {exc}") from exc

    facts: dict[str, Any] = {"present": True}
    commits: list[str] = []
    doctor_ts: str | None = None
    doctor_summary: str | None = None
    doctor_nonpass: list[str] = []
    in_doctor = False
    for line in lines:
        run = _RUN_LINE.match(line)
        if run:
            facts["last_run_ts"] = run.group("ts")
            facts["last_run_fields"] = run.group("fields")
            stamp = _COMMIT_FIELD.search(run.group("fields"))
            facts["last_run_commit"] = stamp.group("commit") if stamp else None
            if stamp and stamp.group("commit") not in commits:
                commits.append(stamp.group("commit"))
            in_doctor = False
            continue
        verdict = _VERDICT_LINE.match(line)
        if verdict:
            facts["last_verdict"] = verdict.group("verdict")
            facts["last_verdict_code"] = verdict.group("code")
            facts["last_verdict_ts"] = verdict.group("ts")
            in_doctor = False
            continue
        header = _DOCTOR_HEADER.match(line)
        if header:
            doctor_ts = header.group("ts")
            doctor_summary = None
            doctor_nonpass = []
            in_doctor = True
            continue
        if not in_doctor:
            continue
        check = _DOCTOR_CHECK.match(line)
        if check and check.group("status") != "PASS":
            doctor_nonpass.append(f"{check.group('status')}  {check.group('name')}  {check.group('detail')}")
        summary = _DOCTOR_SUMMARY.match(line)
        if summary:
            doctor_summary = summary.group(0)
            in_doctor = False
    facts["commits_seen"] = tuple(commits)
    facts["doctor_ts"] = doctor_ts
    facts["doctor_summary"] = doctor_summary
    facts["doctor_nonpass"] = tuple(doctor_nonpass)
    return NightlyFacts(**facts)


# ---------------------------------------------------------------------------
# derived readings


def regime_clock(rows: Sequence[dict[str, Any]], asof: dt.date) -> RegimeClock:
    """The doctor's clock (``prism.scripts.doctor.regime_streak``) at ``asof``."""
    found = regime_streak(rows, as_of=asof)
    last_clean = bool(rows[-1].get("clean")) if rows else False
    return RegimeClock(rows=found.rows, streak=found.streak, last_bar=found.last_bar,
                       trailing_unclean=bool(rows) and not last_clean, run=found.run, missed=len(found.missed))


def _parse_bar(value: str | None) -> dt.date | None:
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def session_gap(*, state: StateFacts, equity: LedgerFacts, nightly: NightlyFacts, asof: dt.date) -> GapFacts:
    """Expected-but-unrun sessions between the last completed one and ``asof``.

    The last completed session is read off the equity ledger, because a NAV mark
    is the durable record of work *done* — the same reason doctor.py's health
    checks read ledgers instead of exit statuses. The state file's
    ``pending_decision_bar`` and the log's last RUN date are cross-checked
    against it and any disagreement is reported rather than resolved.

    A session is "expected" only once its close is past, so the horizon ends at
    the last session strictly before ``asof``: at any hour of ``asof`` itself the
    close-of-day decision does not yet exist.
    """
    last_completed = _parse_bar(equity.last_bar)
    source = f"equity.jsonl last decision_bar {equity.last_bar}"
    if last_completed is None:
        return GapFacts(
            asof=asof,
            last_completed=None,
            last_completed_source="none recorded — equity.jsonl carries no decision_bar",
            horizon_end=None,
        )

    disagreements: list[str] = []
    state_bar = _parse_bar(state.pending_decision_bar)
    if state_bar is not None and state_bar != last_completed:
        disagreements.append(
            f"state.json pending_decision_bar {state_bar.isoformat()} != equity.jsonl last mark "
            f"{last_completed.isoformat()}"
        )
    log_bar = _parse_bar(nightly.last_run_ts)
    if log_bar is not None and log_bar != last_completed:
        disagreements.append(
            f"nightly.log last RUN {log_bar.isoformat()} != equity.jsonl last mark {last_completed.isoformat()}"
        )

    try:
        horizon_end = sessions.previous_session(asof)
        first_expected = sessions.next_session(last_completed)
        missed = tuple(sessions.sessions_between(first_expected, horizon_end)) if first_expected <= horizon_end else ()
        closures = tuple(
            (day, sessions.session_row(day).name)
            for day in _weekdays(last_completed + dt.timedelta(days=1), asof)
            if not sessions.is_session(day)
        )
    except sessions.SessionTableRangeError as exc:
        raise AssessmentError(
            f"the bundled session table cannot cover this run directory: {exc}. Extend "
            "prism/io/sessions.py rather than counting weekdays."
        ) from exc
    return GapFacts(
        asof=asof,
        last_completed=last_completed,
        last_completed_source=source,
        horizon_end=horizon_end,
        missed=missed,
        closures=closures,
        disagreements=tuple(disagreements),
    )


def _weekdays(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    day = start
    while day <= end:
        if day.weekday() < sessions.SATURDAY:
            yield day
        day += dt.timedelta(days=1)


# ---------------------------------------------------------------------------
# the commit question


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """One read-only git invocation.

    Only ``rev-parse``, ``merge-base``, ``rev-list``, ``log`` and ``diff`` are
    ever passed here: the live checkout is another writer's, and this script
    reads it without touching its index, its worktree, or its refs.
    """
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _discover_repo(run_dir: Path) -> Path | None:
    """The checkout that owns the run directory — the one whose HEAD would restart the loop."""
    for candidate in [run_dir, *run_dir.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def commit_difference(*, run_dir: Path, traded_commit: str | None, repo: Path | None) -> GitFacts:
    """Which commits separate the commit the last session traded from the checkout that would restart it."""
    repo = repo or _discover_repo(run_dir)
    if repo is None:
        return GitFacts(available=False, reason=f"no git checkout contains {run_dir}; pass --repo to name one")
    if traded_commit is None:
        return GitFacts(
            available=False,
            repo=repo,
            reason="nightly.log records no commit= stamp, so the traded commit is unknown",
        )
    head = _git(repo, "rev-parse", "HEAD")
    if head.returncode != 0:
        return GitFacts(available=False, repo=repo, reason=f"git rev-parse HEAD failed: {head.stderr.strip()}")
    resolved = _git(repo, "rev-parse", "--verify", f"{traded_commit}^{{commit}}")
    if resolved.returncode != 0:
        return GitFacts(
            available=False,
            repo=repo,
            head=head.stdout.strip(),
            traded_commit=traded_commit,
            reason=f"{traded_commit} is not a commit in {repo}: {resolved.stderr.strip()}",
        )
    head_sha = head.stdout.strip()
    ancestor = _git(repo, "merge-base", "--is-ancestor", traded_commit, head_sha).returncode == 0
    log = _git(repo, "log", "--oneline", "--no-decorate", f"{traded_commit}..{head_sha}")
    if log.returncode != 0:
        return GitFacts(
            available=False,
            repo=repo,
            head=head_sha,
            traded_commit=traded_commit,
            reason=f"git log failed: {log.stderr.strip()}",
        )
    behind = _git(repo, "rev-list", "--count", f"{head_sha}..{traded_commit}")
    diffstats: dict[str, str] = {}
    for target in ("src/prism", "src/prism/live", "ops"):
        stat = _git(repo, "diff", "--stat", f"{traded_commit}..{head_sha}", "--", target)
        diffstats[target] = stat.stdout.strip() if stat.returncode == 0 else f"git diff failed: {stat.stderr.strip()}"
    return GitFacts(
        available=True,
        repo=repo,
        head=head_sha,
        traded_commit=traded_commit,
        ancestor=ancestor,
        commits=tuple(line for line in log.stdout.splitlines() if line.strip()),
        commits_only_on_traded=int(behind.stdout.strip() or 0) if behind.returncode == 0 else 0,
        diffstats=diffstats,
    )


# ---------------------------------------------------------------------------
# optional venue read


def broker_reconciliation(state: StateFacts, *, network: bool, since: str | None) -> list[str]:
    """Compare the venue's book against ``state.json``, or say plainly that it was not done."""
    if not network:
        return [
            "NOT PERFORMED — the assessment ran offline. No credential was used and no venue was",
            "contacted. (The offline path loads no .env at all: importing the package reads no",
            "credential, and only --network loads one.) Re-run with --network (and Alpaca paper",
            "credentials in the environment or the project's .env) to reconcile positions and open",
            "orders before the scheduler is restarted.",
        ]
    # The venue read is the one path that needs a credential; load the
    # project's .env here rather than at import (spec §3.1).
    load_env()
    from prism.live.alpaca import AlpacaBroker

    broker = AlpacaBroker.from_env()
    venue_positions = broker.positions()
    venue_cash = broker.cash()
    known_ids = broker.submitted_order_ids(since)
    open_ids = broker.open_order_ids(known_ids)

    lines = [
        f"venue cash        {venue_cash:,.2f}   state.json cash {state.cash:,.2f}"
        + ("" if abs(venue_cash - state.cash) < 0.01 else "   MISMATCH"),
        f"venue positions   {len(venue_positions)}   state.json positions {len(state.positions)}",
    ]
    only_venue = sorted(set(venue_positions) - set(state.positions))
    only_state = sorted(set(state.positions) - set(venue_positions))
    differing = sorted(
        symbol
        for symbol in set(venue_positions) & set(state.positions)
        if venue_positions[symbol] != state.positions[symbol]
    )
    lines.append(f"at venue only     {', '.join(only_venue) if only_venue else 'none'}")
    lines.append(f"in state only     {', '.join(only_state) if only_state else 'none'}")
    lines.append(
        "quantity differs  "
        + (
            ", ".join(f"{s} venue {venue_positions[s]:g} vs state {state.positions[s]:g}" for s in differing)
            if differing
            else "none"
        )
    )
    lines.append(
        f"open orders       {len(open_ids)} live of {len(known_ids)} client id(s) the venue knows"
        + (f" since {since}" if since else " (full history)")
        + f"; state.json records {state.pending_order_count} pending"
    )
    if open_ids:
        lines.append(f"                  {', '.join(sorted(open_ids))}")
    return lines


# ---------------------------------------------------------------------------
# report


def _money(value: float) -> str:
    return f"{value:,.2f}"


def build_report(
    *,
    run_dir: Path,
    asof: dt.date,
    network: bool,
    repo: Path | None,
    notes: Sequence[str],
) -> list[str]:
    """Read the run directory and render the assessment. Raises ``AssessmentError`` if it cannot."""
    state = read_state(run_dir)
    ledgers: dict[str, LedgerFacts] = {}
    rows_by_ledger: dict[str, list[dict[str, Any]]] = {}
    for name in LEDGER_FILES:
        facts, rows = read_ledger(run_dir / name)
        ledgers[name] = facts
        rows_by_ledger[name] = rows
    nightly = read_nightly_log(run_dir / "nightly.log")
    gap = session_gap(state=state, equity=ledgers["equity.jsonl"], nightly=nightly, asof=asof)
    clock = regime_clock(rows_by_ledger["regime.jsonl"], asof)
    git = commit_difference(run_dir=run_dir, traded_commit=nightly.last_run_commit, repo=repo)

    coverage_start, coverage_end = sessions.coverage()
    out: list[str] = []
    out.append("PRISM PAPER-STATE ASSESSMENT (engine spec stage 0)")
    out.append("=" * 78)
    out.append(f"run directory   {run_dir}")
    out.append(f"as of           {asof.isoformat()}")
    out.append(
        f"session table   version {sessions.TABLE_VERSION}, covers "
        f"{coverage_start.isoformat()}..{coverage_end.isoformat()}"
    )
    out.append("read-only       nothing was written into the run directory")
    out.append("")

    out.append("STATE (state.json)")
    out.append("-" * 78)
    out.append(
        f"schema_version        {state.schema_version} (loader accepts {list(SUPPORTED_STATE_SCHEMA_VERSIONS)})"
    )
    out.append(f"cash                  {_money(state.cash)}")
    longs = sum(1 for q in state.positions.values() if q > 0)
    shorts = sum(1 for q in state.positions.values() if q < 0)
    out.append(f"positions             {len(state.positions)} ({longs} long, {shorts} short)")
    if state.zero_positions:
        out.append(f"  zero-share entries  {', '.join(state.zero_positions)} — a book entry that should be absent")
    out.append(f"pending_orders        {state.pending_order_count}")
    if state.pending_order_ids:
        out.append(f"  client_order_ids    {', '.join(state.pending_order_ids)}")
    out.append(f"last_refresh_bar      {state.last_refresh_bar}")
    out.append(f"last_settled_bar      {state.last_settled_bar}")
    out.append(f"pending_decision_bar  {state.pending_decision_bar}")
    out.append("")

    out.append("LEDGERS")
    out.append("-" * 78)
    width = max(len(name) for name in LEDGER_FILES)
    for name in LEDGER_FILES:
        facts = ledgers[name]
        if not facts.present:
            out.append(f"{name:<{width}}  ABSENT")
            continue
        span = f"{facts.first_bar}..{facts.last_bar}" if facts.first_bar else "no decision_bar in any row"
        damage = f"   {facts.malformed} MALFORMED LINE(S)" if facts.malformed else ""
        out.append(f"{name:<{width}}  {facts.rows:>4} row(s)   {span}{damage}")
    equity_rows = rows_by_ledger["equity.jsonl"]
    if equity_rows:
        last = equity_rows[-1]
        out.append(
            f"last NAV mark     decision_bar {last.get('decision_bar')}"
            f", equity {_money(float(last.get('equity', 0.0)))}"
            f", cash {_money(float(last.get('cash', 0.0)))}"
        )
    out.append("")

    out.append("NIGHTLY LOG (tail)")
    out.append("-" * 78)
    if not nightly.present:
        out.append("nightly.log ABSENT — no wrapper verdict and no doctor summary to report")
    else:
        out.append(f"last RUN          {nightly.last_run_ts}  {nightly.last_run_fields}")
        out.append(
            f"last wrapper exit {nightly.last_verdict} {nightly.last_verdict_code} at {nightly.last_verdict_ts}"
        )
        out.append(f"traded commit     {nightly.last_run_commit}")
        if len(nightly.commits_seen) > 1:
            out.append(f"  commits in log    {', '.join(nightly.commits_seen)} (the last one traded)")
        out.append(f"last doctor       {nightly.doctor_ts}: {nightly.doctor_summary}")
        for line in nightly.doctor_nonpass:
            out.append(f"  {line}")
    out.append("")

    out.append("SESSION GAP")
    out.append("-" * 78)
    if gap.last_completed is None:
        out.append(f"last completed session  UNKNOWN ({gap.last_completed_source})")
        out.append("No gap is computed. An empty record is reported as empty; no session is invented.")
    else:
        out.append(f"last completed session  {gap.last_completed.isoformat()} ({gap.last_completed_source})")
        for line in gap.disagreements:
            out.append(f"  DISAGREEMENT          {line}")
        end = gap.horizon_end.isoformat() if gap.horizon_end else "n/a"
        out.append(f"expected through        {end} (the last session whose close precedes {asof.isoformat()})")
        out.append(f"missed sessions         {len(gap.missed)}")
        for day in gap.missed:
            marker = " (early close)" if sessions.is_early_close(day) else ""
            out.append(f"  {day.isoformat()}{marker}  no session ran")
        if gap.closures:
            out.append("exchange closures in the window (not missed sessions):")
            for day, name in gap.closures:
                out.append(f"  {day.isoformat()}  {name}")
        if gap.missed:
            out.append("")
            out.append("These sessions produced no decision, no order, and no fill. They are a gap in the")
            out.append("record and are never counted as clean sessions.")
    out.append("")

    out.append("REGIME CLEAN-SESSION CLOCK")
    out.append("-" * 78)
    out.append(
        f"clock                   {clock.streak}/{CLEAN_SESSIONS_REQUIRED} consecutive clean sessions as of "
        f"{asof.isoformat()} ({clock.rows} row(s) total, last {clock.last_bar})"
    )
    out.append("rule                    a session counts iff it appended a clean regime.jsonl row; a session")
    out.append("                        that never ran appended none and is not clean, so it restarts the")
    out.append("                        count (docs/regime_step.md §4; owner decision 2026-09-25;")
    out.append("                        prism.scripts.doctor.regime_streak).")
    if clock.missed:
        out.append(f"                        The run through {clock.last_bar} was {clock.run}; the {clock.missed}")
        out.append("                        unrecorded session(s) since restarted it.")
    if clock.trailing_unclean:
        out.append("                        The newest row is not clean; the streak was already 0 before the gap.")
    out.append("")

    out.append("COMMIT DIFFERENCE")
    out.append("-" * 78)
    if not git.available:
        out.append(f"NOT DETERMINED — {git.reason}")
    else:
        out.append(f"repository        {git.repo}")
        out.append(f"last session traded  {git.traded_commit}")
        out.append(f"current checkout     {git.head}")
        if git.ancestor is False:
            out.append(
                f"  HISTORY DIVERGED — the traded commit is not an ancestor of HEAD, and {git.commits_only_on_traded}"
            )
            out.append("  commit(s) reachable from it are not reachable from HEAD (a rewritten or re-founded")
            out.append("  history). The commit list below is one-sided: it is what HEAD adds. The diffstat")
            out.append("  that follows compares the two trees directly and is complete either way.")
        out.append(f"commits HEAD adds    {len(git.commits)}")
        for line in git.commits:
            out.append(f"  {line}")
        for target, stat in git.diffstats.items():
            out.append(f"tree difference under {target}:")
            if stat:
                for line in stat.splitlines():
                    out.append(f"  {line}")
            else:
                out.append("  none")
        out.append("A restart runs the current checkout, not the commit the last session traded.")
    out.append("")

    out.append("BROKER RECONCILIATION")
    out.append("-" * 78)
    out.extend(
        broker_reconciliation(
            state,
            network=network,
            since=ledgers["equity.jsonl"].first_bar,
        )
    )
    out.append("")

    out.append("OPEN ITEMS FOR THE OWNER")
    out.append("-" * 78)
    out.append("1. Restarting the scheduler is an owner act under the existing operating rules; this")
    out.append("   assessment does not perform it and does not authorize it.")
    if not network:
        out.append("2. The venue reconciliation has not been performed (see above); do it before restart.")
    for note in notes:
        out.append(f"NOTE: {note}")
    if sessions.coverage_end() - asof <= dt.timedelta(days=90):
        out.append(
            f"WARNING: the bundled session table ends {sessions.coverage_end().isoformat()}, within ninety days "
            f"of {asof.isoformat()}. Extend prism/io/sessions.py."
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read a paper-loop run directory and report its state, its session gap, and the commit "
        "difference. Read-only; offline unless --network is given.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Absolute path to the run directory to assess. Required, and never defaulted.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the report here instead of stdout. Never inside the run directory.",
    )
    parser.add_argument(
        "--asof",
        type=dt.date.fromisoformat,
        default=None,
        help="Assess as of this date (YYYY-MM-DD); default today. Sessions are expected only once closed.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Git checkout to compare the traded commit against; default the one containing the run directory.",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Also read the broker's positions and open orders (needs Alpaca credentials). Off by default.",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="TEXT",
        help="Record an operator note verbatim under OPEN ITEMS (e.g. a deferred restart and its reason).",
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    if not run_dir.is_absolute():
        parser.error(f"run_dir must be an absolute path, got {run_dir}")
    if not run_dir.is_dir():
        parser.error(f"{run_dir} is not a directory")
    if args.out is not None and args.out.resolve().is_relative_to(run_dir.resolve()):
        parser.error(f"--out {args.out} is inside the run directory; the assessment never writes there")

    try:
        report = build_report(
            run_dir=run_dir,
            asof=args.asof or dt.date.today(),
            network=args.network,
            repo=args.repo,
            notes=args.note,
        )
    except AssessmentError as exc:
        print(f"assessment failed: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE

    text = "\n".join(report) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
