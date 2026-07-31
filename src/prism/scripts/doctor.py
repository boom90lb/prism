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
* ``regime-clock`` — consecutive clean sessions on the handoff §8
  precondition-(b) clock (docs/regime_step.md §4).
* ``nightly-log`` — the last wrapper verdict, and whether a *successful* one
  is stale (the "green but old" failure).
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
    prism-doctor --run-dir runs/paper_loop_momentum      # installed entry point
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

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

# `2026-07-29T18:30:16-0700 EXIT 1` — the wrapper's verdict line (ops/).
_LOG_VERDICT = re.compile(r"^(?P<ts>\S+)\s+(?P<verdict>EXIT|SKIP)\b\D*(?P<code>\d+)?")


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


def check_env_credentials(env: dict[str, str]) -> list[CheckResult]:
    """Presence checks only — values are never printed (docs/security.md §2)."""
    results = []
    alpaca = bool(env.get("APCA_API_KEY_ID")) and bool(env.get("APCA_API_SECRET_KEY"))
    results.append(
        _result(
            "alpaca-credentials",
            alpaca,
            "APCA_API_KEY_ID/APCA_API_SECRET_KEY present"
            if alpaca
            else "APCA_API_KEY_ID/APCA_API_SECRET_KEY missing — the paper loop cannot trade; "
            "put the *paper* keys in .env (docs/quickstart.md)",
        )
    )
    twelve = bool(env.get("TWELVEDATA_API_KEY"))
    results.append(
        _result(
            "twelvedata-key",
            twelve,
            "TWELVEDATA_API_KEY present"
            if twelve
            else "TWELVEDATA_API_KEY missing — fine for the Alpaca bar source (the loop "
            "default), required for the research spine",
            warn=True,
        )
    )
    base = env.get("APCA_API_BASE_URL", "")
    live = "paper" not in base and bool(base)
    results.append(
        _result(
            "alpaca-endpoint",
            not live,
            f"LIVE endpoint configured ({base}) — real money; unset APCA_API_BASE_URL for paper"
            if live
            else (f"paper endpoint ({base})" if base else "paper endpoint (default)"),
            warn=True,
        )
    )
    return results


def _read_universe(path: Path | None) -> list[str]:
    """The universe file's symbols, or ``[]`` — the loop's parse, minus the raise."""
    from prism.io.universe_file import load_universe_symbols

    if path is None or not path.exists():
        return []
    return load_universe_symbols(path, require_nonempty=False)


def check_universe_file(path: Path | None) -> CheckResult:
    if path is None:
        return _result(
            "universe-file", False, "no --universe-file given — pass the file the loop will trade", warn=True
        )
    if not path.exists():
        return _result("universe-file", False, f"{path} does not exist")
    symbols = _read_universe(path)
    if not symbols:
        return _result("universe-file", False, f"{path} parses to zero symbols")
    if len(symbols) < MIN_MOMENTUM_UNIVERSE:
        return _result(
            "universe-file",
            False,
            f"{path}: {len(symbols)} symbols < {MIN_MOMENTUM_UNIVERSE} — a decile book "
            "degenerates on a thin cross-section (fine for the ensemble cost instrument)",
            warn=True,
        )
    return _result("universe-file", True, f"{path}: {len(symbols)} symbols")


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


def check_regime_clock(run_dir: Path) -> CheckResult:
    """Consecutive clean sessions on the handoff §8 precondition-(b) clock.

    A session counts iff it appended a regime row with ``clean: true``; a gap in
    the ledger is not clean either, because absence of the record is absence of
    the session (docs/regime_step.md §4). Reported, never FAILed — an
    incomplete clock is the normal state of an honest program, and this check
    exists so the count is read off the ledger instead of remembered.
    """
    from prism.live import read_regime_ledger

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
    clean = [bool(v) for v in frame.get("clean", [])]
    streak = 0
    for value in reversed(clean):
        if not value:
            break
        streak += 1
    last_bar = str(frame["decision_bar"].iloc[-1]) if "decision_bar" in frame else "?"
    detail = (
        f"{streak}/{CLEAN_SESSIONS_REQUIRED} consecutive clean sessions through {last_bar} "
        f"({len(clean)} row(s) total)"
    )
    if streak >= CLEAN_SESSIONS_REQUIRED:
        return _result("regime-clock", True, detail + " — precondition (b) satisfied")
    return _result("regime-clock", False, detail, warn=True)


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
    verdicts = [m for m in (_LOG_VERDICT.match(ln.strip()) for ln in log.read_text(encoding="utf-8").splitlines()) if m]
    if not verdicts:
        return _result("nightly-log", False, f"{log}: no EXIT/SKIP verdict lines — a truncated or foreign log", warn=True)
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
    persisted = {
        str(s) for s, q in getattr(state, "positions", {}).items() if float(q) != 0.0
    } if state is not None else set()
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
    """One account read → three answers: keys, book attribution, priceability.

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


def check_twelvedata_quote() -> CheckResult:
    """Network probe: one bar for SPY (1 credit of the daily 800).

    WARN, never FAIL. The live loop reads Alpaca's own feed by default, so a
    dead research-spine key does not stop a paper session — and this doctor's
    exit code now drives the nightly alert (ops/paper_loop_nightly.sh). An alert
    that fires on something the live path does not use is an alert the operator
    learns to ignore, which is the failure mode P0 exists to remove.
    """
    from prism.io.loader import DataLoader

    loader = DataLoader()
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
    env: dict[str, str] | None = None,
    network: bool = False,
    today: date | None = None,
) -> list[CheckResult]:
    from prism.config import DATA_DIR

    env = dict(os.environ) if env is None else env
    results = [check_python()]
    results.extend(check_env_credentials(env))
    results.append(check_universe_file(universe_file))
    results.append(check_data_dir(data_dir or DATA_DIR))
    results.append(check_loop_state(run_dir))
    results.append(check_kill_switch(run_dir))
    # Health: the durable record of work done. Offline, because a dark loop must
    # be detectable without credentials — the alerting path cannot depend on the
    # same venue reachability whose absence it has to report.
    results.append(check_equity_ledger(run_dir, today=today))
    results.append(check_nightly_log(run_dir, today=today))
    results.append(check_regime_clock(run_dir))
    if network:
        results.extend(check_alpaca_account(universe_file, run_dir))
        results.append(check_twelvedata_quote())
    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Preflight checks for the prism paper loop.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Loop run directory; default honors runs/ACTIVE_RUN_DIR, else runs/paper_loop_momentum.",
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
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    if run_dir is None:
        active = Path("runs/ACTIVE_RUN_DIR")
        run_dir = Path(
            active.read_text(encoding="utf-8").strip() if active.exists() else "runs/paper_loop_momentum"
        )

    results = run_checks(run_dir=run_dir, universe_file=args.universe_file, network=args.network)
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
