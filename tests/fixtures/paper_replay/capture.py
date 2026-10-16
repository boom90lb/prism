"""Capture the baseline replay fixture from the recorded paper run (engine spec stage 0).

The stage-5 replay parity gate needs the live book's own recorded sessions to
compare against, and the run directory alone is not enough to replay them: the
loop never persisted the bar panels it decided on (it read them from Alpaca over
the network), while ``prism.live.replay.replay_daily_cycles`` takes ``close``,
``volume`` and ``open_`` as arguments. This script assembles what the record
does hold — targets, fills, unfilled residuals, concordance reads, equity marks,
the final reconciled state, and the spin-off mask the second refresh consumed —
and reconstructs the panels from the local incremental bar store, trimmed to the
names those sessions actually touched.

It never fabricates a panel. A session whose universe the store cannot cover at
that session's bar is recorded as unreconstructable, with the counts that say so,
and no panel file is written. A parity test that silently replays a panel the
session never saw is worse than no parity test.

Read-only against the live checkout: the run directory, the bar store and the
legacy cache are opened for reading and nothing under them is created or
modified. Deterministic given the same inputs and ``--reconstruction-date``.

Run it from the repository root::

    PYTHONPATH=src python tests/fixtures/paper_replay/capture.py \
        --run-dir runs/paper_loop_momentum2 \
        --store-dir data/store \
        --legacy-cache-dir data
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from prism.io import sessions
from prism.io.store import BAR_TZ, IncrementalBarStore

logger = logging.getLogger("capture_paper_replay")

SCHEMA_VERSION = 1

# The fixture set is stated by the specification rather than discovered, because
# the point of a baseline is that it does not drift: the record holds exactly two
# decision-bearing sessions and three hold sessions after the second of them. The
# capture asserts the record still matches, so a changed run directory fails here
# instead of quietly producing a different baseline.
EXPECTED_DECISION_BARS = ("2026-07-13", "2026-08-11")
EXPECTED_HOLD_BARS = ("2026-08-12", "2026-08-13", "2026-08-14")

# Momentum book B1: 252-bar lookback with a 21-bar skip (the RUN lines in
# nightly.log), so a decision bar needs 273 trailing closes plus itself.
DEFAULT_MIN_HISTORY = 274

LEDGERS = ("targets.jsonl", "fills.jsonl", "unfilled.jsonl", "concordance.jsonl", "equity.jsonl")
STATE_FILE = "state.json"
SPINOFF_MASK_FILE = "spinoff_mask_2026-08-11.json"

CAVEATS = [
    "Two decision-bearing sessions, not three. The engine specification's 'at least three "
    "decision-bearing sessions' is not cuttable from this record: targets.jsonl holds exactly "
    "two rows. The fixture set is the two refreshes plus the three hold sessions that follow "
    "the second, and nothing here supplies a third refresh.",
    "Reconstructed panels are not the panels the sessions saw. IncrementalBarStore carries a "
    "SplitRewriteDetected path (src/prism/io/store.py:50) precisely because vendors rewrite "
    "split-adjusted history wholesale; any panel rebuilt after the fact may sit on a different "
    "adjustment basis than the session did. Replay parity against this fixture is a bounded "
    "claim about the engine's arithmetic, not an unqualified reproduction of the live session.",
    "The trimmed universe cannot reproduce the ranking. These sessions scored the full S&P 500 "
    "panel (the RUN lines record 502-503 symbols); the fixture universe is only the names the "
    "sessions traded, marked or held. That is enough to check settle, mark and order arithmetic "
    "against the recorded ledgers, and not enough to re-derive the cross-sectional decile book. "
    "Reproducing the decision itself needs the full recorded universe panel.",
    "Recorded fills are live auction prints; a replay's are modeled. The ledger's fill prices "
    "came from Alpaca OPG auctions and the morning sweep; ReplayBroker fills at the next open "
    "print exactly (src/prism/live/replay.py:49-58). Parity must therefore be defined on the "
    "decided book and the orders, never on fill price or realised equity.",
    "The recorded sessions are not contiguous. The run directory was deleted and restored from "
    "backup history on 2026-07-29 and the sessions 2026-07-18 through 2026-07-29 did not run "
    "(runs/paper_loop_momentum2/REATTACHED.md). The equity and concordance ledgers skip that "
    "interval; a replay over a continuous calendar will produce marks the record does not have.",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — the run directory is not the recorded paper book")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty — an empty ledger is a defect, not a baseline")
    return rows


def fixture_universe(
    targets: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    unfilled: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[str]:
    """Every symbol the fixture sessions priced, traded or still hold.

    The union of the decided books and their reference prices, the fill and
    residual ledgers, and the final reconciled positions — the names a replay of
    these sessions must be able to price, as opposed to the ~500-name panel the
    sessions ranked over.
    """
    names: set[str] = set(state.get("positions", {}))
    for row in targets:
        names |= set(row["targets"]) | set(row["reference_prices"])
    names |= {row["symbol"] for row in fills}
    names |= {row["symbol"] for row in unfilled}
    return sorted(names)


def session_index(
    decision_bars: tuple[str, ...],
    hold_bars: tuple[str, ...],
    targets: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    unfilled: list[dict[str, Any]],
    concordance: list[dict[str, Any]],
    equity: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per fixture session: its kind and what the record attributes to it.

    ``n_exit_symbols`` counts the names a refresh traded out of but did not
    decide a weight for — the previous book's holdings the new book dropped.
    """
    by_bar = {row["decision_bar"]: row for row in targets}
    equity_bars = {row["decision_bar"] for row in equity}
    index: list[dict[str, Any]] = []
    for bar in sorted(decision_bars + hold_bars):
        decision_bearing = bar in decision_bars
        row: dict[str, Any] = {
            "bar": bar,
            "kind": "decision" if decision_bearing else "hold",
            "is_trading_session": sessions.is_session(dt.date.fromisoformat(bar)),
            "n_targets": len(by_bar[bar]["targets"]) if decision_bearing else 0,
            "n_priced": len(by_bar[bar]["reference_prices"]) if decision_bearing else 0,
            "n_fills": sum(1 for f in fills if f["decision_bar"] == bar),
            "n_unfilled": sum(1 for u in unfilled if u["decision_bar"] == bar),
            "n_concordance": sum(1 for c in concordance if c["decision_bar"] == bar),
            "has_equity_mark": bar in equity_bars,
        }
        if decision_bearing:
            priced = set(by_bar[bar]["reference_prices"])
            # A refresh exits every name the previous book held that this book
            # does not: those fills are priced at the bar but carry no target
            # weight, so they sit outside `reference_prices` entirely.
            row["recorded_equity"] = by_bar[bar]["equity"]
            row["n_exit_symbols"] = len({f["symbol"] for f in fills if f["decision_bar"] == bar} - priced)
        index.append(row)
    return index


def reconstruct_panels(
    store: IncrementalBarStore,
    symbols: list[str],
    session_bars: list[str],
    *,
    interval: str = "1d",
    min_history: int = DEFAULT_MIN_HISTORY,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None, dict[str, Any]]:
    """Rebuild (close, volume, open) through the last session bar, or refuse.

    Reads the incremental bar store directly — the same series
    ``prism.live.daily.fetch_universe_panels`` serves from, without a network
    fetch — and truncates at the last fixture session. Panels are returned only
    when every fixture symbol carries a bar at *every* fixture session and at
    least ``min_history`` rows through the first one — the trailing history
    ``replay_daily_cycles`` demands of its opening bar; otherwise the caller
    gets ``None`` and a report saying exactly how far short the store fell. A
    partial panel is not a baseline, so none is written.
    """
    through = pd.Timestamp(session_bars[-1], tz=BAR_TZ)
    frames: dict[str, pd.DataFrame] = {}
    dates: dict[str, pd.DatetimeIndex] = {}
    absent: list[str] = []
    for symbol in symbols:
        bars = store.read(symbol, interval)
        if bars.empty or not {"open", "close"} <= set(bars.columns):
            absent.append(symbol)
            continue
        index = bars.index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(f"{symbol} {interval}: the bar store returned a frame with no timestamp index")
        keep = index.normalize() <= through
        frames[symbol] = bars[keep]
        dates[symbol] = index.normalize()[keep]

    latest = max((d[-1] for d in dates.values() if len(d)), default=None)
    per_session: list[dict[str, Any]] = []
    for bar in session_bars:
        stamp = pd.Timestamp(bar, tz=BAR_TZ)
        covered = [s for s, d in dates.items() if stamp in d]
        per_session.append(
            {
                "bar": bar,
                "universe_size": len(symbols),
                "symbols_with_a_bar": len(covered),
                "symbols_missing_a_bar": len(symbols) - len(covered),
            }
        )
    # replay_daily_cycles refuses a first bar with fewer trailing rows than the
    # signal's required_history, so the binding count is at the FIRST session.
    first = pd.Timestamp(session_bars[0], tz=BAR_TZ)
    history = min((int((d <= first).sum()) for d in dates.values()), default=0)
    report: dict[str, Any] = {
        "store_interval": interval,
        "min_history_required": min_history,
        "min_trailing_rows_among_stored_symbols": history,
        "symbols_absent_from_store": len(absent),
        "latest_stored_bar": None if latest is None else str(latest.date()),
        "per_session_coverage": per_session,
    }

    shortfalls = [row for row in per_session if row["symbols_missing_a_bar"]]
    if shortfalls or history < min_history:
        stored = (
            f"the store covers {len(frames)}/{len(symbols)} fixture symbols, latest bar {report['latest_stored_bar']}"
        )
        if shortfalls:
            worst = shortfalls[0]
            detail = f"{worst['symbols_missing_a_bar']}/{len(symbols)} have no bar at session {worst['bar']}"
        else:
            detail = f"only {history} trailing rows against a {min_history}-row requirement"
        report["status"] = "unavailable"
        report["reason"] = f"{stored}; {detail}"
        return None, report

    close = pd.DataFrame({s: pd.to_numeric(f["close"], errors="coerce") for s, f in frames.items()}).sort_index()
    open_ = pd.DataFrame({s: pd.to_numeric(f["open"], errors="coerce") for s, f in frames.items()}).sort_index()
    volume = (
        pd.DataFrame(
            {s: pd.to_numeric(f["volume"], errors="coerce") for s, f in frames.items() if "volume" in f.columns}
        )
        .reindex(columns=close.columns)
        .sort_index()
    )
    report["status"] = "reconstructed"
    report["rows"] = int(close.shape[0])
    report["columns"] = int(close.shape[1])
    return (close, volume, open_), report


def probe_legacy_cache(cache_dir: Path, symbols: list[str]) -> dict[str, Any]:
    """Latest cached daily bar per fixture symbol in the legacy range-keyed cache.

    Recorded so that "the panels cannot be reconstructed" is a claim about every
    local bar source, not only about the incremental store.
    """
    pattern = re.compile(r"^(?P<symbol>.+)_1d_\d{4}-\d{2}-\d{2}_(?P<end>\d{4}-\d{2}-\d{2})\.parquet$")
    latest: dict[str, str] = {}
    for path in sorted(cache_dir.glob("*_1d_*.parquet")):
        match = pattern.match(path.name)
        if match is None or match["symbol"] not in set(symbols):
            continue
        symbol, end = match["symbol"], match["end"]
        latest[symbol] = max(latest.get(symbol, end), end)
    return {
        "cache_dir": str(cache_dir),
        "symbols_with_a_cache": len(latest),
        "latest_cached_bar": max(latest.values(), default=None),
        "symbols_without_a_cache": len(set(symbols) - set(latest)),
    }


def _traded_commit(nightly_log: Path) -> str | None:
    """The commit of the last RUN line — the code the recorded sessions ran."""
    if not nightly_log.exists():
        return None
    hits = re.findall(r"^.*RUN:.*\bcommit=(\S+)", nightly_log.read_text(encoding="utf-8", errors="replace"), re.M)
    return hits[-1] if hits else None


def capture(
    run_dir: Path,
    store_dir: Path,
    out_dir: Path,
    *,
    reconstruction_date: str,
    legacy_cache_dir: Path | None,
    min_history: int,
) -> dict[str, Any]:
    """Write the fixture and its manifest; return the manifest."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} is not a directory — pass the recorded run directory")
    if not store_dir.is_dir():
        # IncrementalBarStore creates its directory on construction, and this
        # script must never write into the live checkout.
        raise FileNotFoundError(f"{store_dir} is not a directory — refusing to create a bar store here")

    targets = _read_jsonl(run_dir / "targets.jsonl")
    fills = _read_jsonl(run_dir / "fills.jsonl")
    unfilled = _read_jsonl(run_dir / "unfilled.jsonl")
    concordance = _read_jsonl(run_dir / "concordance.jsonl")
    equity = _read_jsonl(run_dir / "equity.jsonl")
    state = json.loads((run_dir / STATE_FILE).read_text(encoding="utf-8"))

    recorded_decision_bars = tuple(sorted({row["decision_bar"] for row in targets}))
    if recorded_decision_bars != EXPECTED_DECISION_BARS:
        raise ValueError(
            f"{run_dir}/targets.jsonl holds decision bars {recorded_decision_bars}, not the "
            f"{EXPECTED_DECISION_BARS} the engine specification's stage-0 fixture paragraph states — "
            "the record changed; re-read the specification before recapturing the baseline"
        )
    last_mark = equity[-1]["decision_bar"]
    if last_mark != EXPECTED_HOLD_BARS[-1]:
        raise ValueError(
            f"{run_dir}/equity.jsonl ends at {last_mark}, not {EXPECTED_HOLD_BARS[-1]}; the recorded "
            "run advanced past the stated fixture set"
        )

    universe = fixture_universe(targets, fills, unfilled, state)
    session_bars = sorted(EXPECTED_DECISION_BARS + EXPECTED_HOLD_BARS)
    panels, panel_report = reconstruct_panels(
        IncrementalBarStore(store_dir), universe, session_bars, min_history=min_history
    )
    if legacy_cache_dir is not None:
        panel_report["legacy_cache_probe"] = probe_legacy_cache(legacy_cache_dir, universe)

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / "panels"
    if panel_dir.exists():
        shutil.rmtree(panel_dir)

    written: list[str] = []
    for name in LEDGERS:
        shutil.copyfile(run_dir / name, out_dir / name)
        written.append(name)
    shutil.copyfile(run_dir / STATE_FILE, out_dir / "state_final.json")
    written.append("state_final.json")
    if (run_dir / SPINOFF_MASK_FILE).exists():
        shutil.copyfile(run_dir / SPINOFF_MASK_FILE, out_dir / SPINOFF_MASK_FILE)
        written.append(SPINOFF_MASK_FILE)
    (out_dir / "universe.txt").write_text("\n".join(universe) + "\n", encoding="utf-8")
    written.append("universe.txt")

    if panels is not None:
        panel_dir.mkdir()
        for label, frame in zip(("close", "volume", "open"), panels):
            frame.to_csv(panel_dir / f"{label}.csv", index_label="datetime")
            written.append(f"panels/{label}.csv")
        logger.info("reconstructed panels: %d bars x %d symbols", panels[0].shape[0], panels[0].shape[1])
    else:
        logger.warning("panels NOT reconstructed: %s", panel_report["reason"])

    mask_path = run_dir / SPINOFF_MASK_FILE
    scored = json.loads(mask_path.read_text(encoding="utf-8")).get("symbols_checked", []) if mask_path.exists() else []

    caveats = list(CAVEATS)
    if panels is None:
        caveats.insert(
            0,
            "NO PANELS. The bar panels could not be reconstructed for any fixture session: "
            f"{panel_report['reason']}. The live loop read its bars from Alpaca over the network and "
            "persisted none, and no local source reaches these bars. Nothing in this directory can be "
            "driven through replay_daily_cycles until a bar source covering "
            f"{session_bars[0]}..{session_bars[-1]} for the fixture universe is restored; a parity test "
            "must skip on that, never pass.",
        )
    manifest = {
        "fixture": "paper_replay",
        "schema_version": SCHEMA_VERSION,
        "reconstruction_date": reconstruction_date,
        "purpose": (
            "Baseline for the engine overhaul's replay parity gate (engine spec stage 0). Recorded "
            "sessions only — nothing here is a modeled-fill fixture."
        ),
        "source": {
            "run_dir": str(run_dir),
            "store_dir": str(store_dir),
            "traded_commit": _traded_commit(run_dir / "nightly.log"),
            "bar_source": "alpaca (the live loop fetched panels over the network and persisted none)",
            "files": {
                name: {"sha256": _sha256(run_dir / name), "bytes": (run_dir / name).stat().st_size}
                for name in (*LEDGERS, STATE_FILE)
            },
        },
        "sessions": session_index(
            EXPECTED_DECISION_BARS, EXPECTED_HOLD_BARS, targets, fills, unfilled, concordance, equity
        ),
        "decision_bearing_bars": list(EXPECTED_DECISION_BARS),
        "hold_bars": list(EXPECTED_HOLD_BARS),
        "universe": {
            "fixture_size": len(universe),
            "recorded_scored_universe_size": len(scored),
            "note": (
                "universe.txt is trimmed to the names these sessions priced, traded or hold. The "
                "sessions themselves ranked the full S&P 500 panel."
            ),
        },
        "panels": panel_report,
        "caveats": caveats,
        "files": {
            name: {"sha256": _sha256(out_dir / name), "bytes": (out_dir / name).stat().st_size}
            for name in sorted(written)
        },
        "capture_command": (
            "PYTHONPATH=src python tests/fixtures/paper_replay/capture.py "
            f"--run-dir {run_dir} --store-dir {store_dir} "
            + (f"--legacy-cache-dir {legacy_cache_dir} " if legacy_cache_dir is not None else "")
            + f"--reconstruction-date {reconstruction_date} --min-history {min_history}"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Recorded paper run directory (read-only).")
    parser.add_argument("--store-dir", type=Path, required=True, help="Incremental bar store directory (read-only).")
    parser.add_argument(
        "--legacy-cache-dir",
        type=Path,
        default=None,
        help="Legacy range-keyed parquet cache, probed for the record (read-only).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Fixture output directory (default: this script's directory).",
    )
    parser.add_argument(
        "--reconstruction-date",
        default=dt.datetime.now(dt.UTC).date().isoformat(),
        help="Date recorded in the manifest; pass it explicitly to reproduce a capture byte for byte.",
    )
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = capture(
        args.run_dir.resolve(),
        args.store_dir.resolve(),
        args.out,
        reconstruction_date=args.reconstruction_date,
        legacy_cache_dir=None if args.legacy_cache_dir is None else args.legacy_cache_dir.resolve(),
        min_history=args.min_history,
    )
    logger.info(
        "wrote %s: %d sessions (%d decision-bearing), %d symbols, panels=%s",
        args.out / "manifest.json",
        len(manifest["sessions"]),
        len(manifest["decision_bearing_bars"]),
        manifest["universe"]["fixture_size"],
        manifest["panels"]["status"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
