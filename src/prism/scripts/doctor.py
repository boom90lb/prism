"""Preflight doctor — verify a checkout can run the paper loop (SPEC §7.7).

Turns the first-run failure classes into a pre-flight report instead of a
mid-loop crash: missing credentials (N7 raises at ``from_env``), a missing or
degenerate universe file (the 2026-07-15 POOL class: universe-file changes
bite at *valuation*, immediately), an unwritable data dir, corrupt or
schema-stale loop state, and a forgotten kill switch (the book silently not
trading is exactly the silence N7 bans).

Offline by default — no vendor credit is spent and no venue is touched.
``--network`` adds two live probes: the Alpaca account read (free,
unmetered) and one Twelve Data quote (1 credit of the daily 800), each
answering "do these keys actually work?", the question an offline check
cannot.

Exit code 0 when every check passes or warns, 1 when anything FAILs.
WARN means "runnable, but look": a missing Twelve Data key with the Alpaca
bar source configured is a WARN, not a FAIL.

    python -m prism.scripts.doctor                       # offline preflight
    python -m prism.scripts.doctor --network             # + credential probes
    prism-doctor --run-dir runs/paper_loop_momentum      # installed entry point
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# The decile construct needs a non-degenerate cross-section: below ~100 names
# a 0.10 decile book holds <10 per leg and concentration risk dwarfs signal.
MIN_MOMENTUM_UNIVERSE = 100


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


def check_universe_file(path: Path | None) -> CheckResult:
    if path is None:
        return _result(
            "universe-file", False, "no --universe-file given — pass the file the loop will trade", warn=True
        )
    if not path.exists():
        return _result("universe-file", False, f"{path} does not exist")
    symbols = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
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


def check_alpaca_account() -> CheckResult:
    """Network probe: the free, unmetered account read — do the keys work?"""
    from prism.live import AlpacaBroker

    try:
        broker = AlpacaBroker.from_env()
        cash = broker.cash()
        positions = broker.positions()
    except Exception as exc:  # any layer: env, auth, transport
        return _result("alpaca-account", False, f"account read failed: {exc}")
    return _result("alpaca-account", True, f"cash {cash:.2f}, {len(positions)} open positions")


def check_twelvedata_quote() -> CheckResult:
    """Network probe: one bar for SPY (1 credit of the daily 800)."""
    from prism.io.loader import DataLoader

    loader = DataLoader()
    if not loader.api_key:
        return _result("twelvedata-quote", False, "no key to probe", warn=True)
    df = loader.fetch_historical_data("SPY", "1d", force_refresh=True)
    if df.empty:
        return _result(
            "twelvedata-quote", False, "SPY fetch returned empty — key invalid, plan exhausted, or vendor down"
        )
    return _result("twelvedata-quote", True, f"SPY bars through {df.index[-1].date()}")


def run_checks(
    *,
    run_dir: Path,
    universe_file: Path | None,
    data_dir: Path | None = None,
    env: dict[str, str] | None = None,
    network: bool = False,
) -> list[CheckResult]:
    from prism.config import DATA_DIR

    env = dict(os.environ) if env is None else env
    results = [check_python()]
    results.extend(check_env_credentials(env))
    results.append(check_universe_file(universe_file))
    results.append(check_data_dir(data_dir or DATA_DIR))
    results.append(check_loop_state(run_dir))
    results.append(check_kill_switch(run_dir))
    if network:
        results.append(check_alpaca_account())
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
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"{r.status:4}  {r.name:<{width}}  {r.detail}")
    failed = [r for r in results if r.status == "FAIL"]
    warned = [r for r in results if r.status == "WARN"]
    print(
        f"\n{len(results) - len(failed) - len(warned)} pass, {len(warned)} warn, {len(failed)} fail"
        + ("" if args.network else "  (offline checks only; --network probes credentials)")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
