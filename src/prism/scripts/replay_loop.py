"""Replay CLI — drive the live daily cycle over local bars, faster than realtime.

A thin shell over :mod:`prism.live.replay` (which owns every piece of logic and
its tests): loads the local parquet caches offline, aligns the ragged union to
a consensus calendar, and replays the *real* ``run_daily_cycle`` — write-ahead
protocol, tolerant settle, cadence gate, monitor — with modeled next-open
fills. A diagnostic instrument only: its fills never feed I-9 cost calibration
and its equity stream never joins the live paper stream the promotion gate's
concordance read is defined on (prism/live/replay.py module docstring).

    # pre-flight the B1 momentum book over the post-discovery window
    python -m prism.scripts.replay_loop --book momentum \\
        --universe-file data/universe/sp500_current.txt \\
        --start 2026-06-17 --run-dir runs/replay_momentum_oos

    # rehearse a cutover from an inherited live book
    python -m prism.scripts.replay_loop --book momentum \\
        --universe-file data/universe/sp500_current.txt \\
        --positions '{"COST": 1, "MMM": -3, "ORCL": -1}' --cash 99649.62 \\
        --start 2026-06-17 --run-dir runs/replay_cutover

Fit policy matches the live loop: a fresh signal is fitted per cycle on the
trailing panel (causal; free for the stateless momentum node, ~half a minute
per cycle for the ensemble forecaster).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from prism.io.universe_sp500 import build_membership_mask
from prism.live.daily import DailyBookConfig
from prism.live.replay import (
    align_replay_panels,
    load_local_bar_panels,
    replay_daily_cycles,
)
from prism.residual.factors import ResidualStatArbConfig
from prism.signal.base import Signal
from prism.signal.ensemble_node import EnsembleNodeConfig, EnsembleSignalNode
from prism.signal.momentum_node import MomentumSignalNode
from prism.signal.trend_node import TREND_V1_UNIVERSE, TrendSignalNode

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the live daily cycle over local bars with modeled next-open fills "
        "(diagnostic only — see prism/live/replay.py for what a replay may and may not claim)."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated universe; or use --universe-file.")
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=None,
        help="File with one symbol per line (blank lines and #comments skipped); overrides --symbols.",
    )
    parser.add_argument(
        "--book",
        choices=("ensemble", "momentum", "trend"),
        default="ensemble",
        help="Which book to replay (same construction paths as prism.scripts.paper_loop).",
    )
    parser.add_argument("--mom-lookback", type=int, default=252, help="Momentum/trend lookback bars (B1/T0: 252).")
    parser.add_argument("--mom-skip", type=int, default=21, help="Momentum/trend skip bars (B1/T0: 21).")
    parser.add_argument("--decile", type=float, default=0.10, help="Decile fraction per leg (B1: 0.10).")
    parser.add_argument(
        "--vol-ewma-bars",
        type=int,
        default=63,
        help="Trend inverse-vol EWMA window (docs/trend_design.md §2: 63).",
    )
    parser.add_argument(
        "--decision-every",
        type=int,
        default=None,
        help="Refresh cadence in trading sessions; default 1 (ensemble) or 21 (momentum/trend).",
    )
    parser.add_argument("--start", default=None, help="First decision bar (YYYY-MM-DD); default earliest feasible.")
    parser.add_argument("--end", default=None, help="Last decision bar (YYYY-MM-DD); default last local bar.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/replay"),
        help="Fresh directory for the replay's state/fills/equity artifacts. Refuses a dir "
        "that already holds loop state — replay streams stay separate from live ledgers.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Local parquet cache directory.")
    parser.add_argument(
        "--membership-file",
        type=Path,
        default=None,
        help="Point-in-time membership intervals parquet ([ticker, start, end], e.g. "
        "data/universe/sp500_membership_2026-06-16.parquet). Momentum book only: ANDs the "
        "PIT index gate into eligibility (prism.io.universe_sp500.build_membership_mask), "
        "matching the research backtest's universe handling. Without it the replay trades "
        "a membership-blind screen — expect book differences on names whose membership "
        "changed mid-sample.",
    )
    parser.add_argument(
        "--max-missing",
        type=float,
        default=None,
        help="Fraction of the universe allowed to have no local bars; default 0.0 (ensemble) "
        "or 0.10 (momentum/trend), mirroring paper_loop.",
    )
    parser.add_argument(
        "--consensus",
        type=float,
        default=0.95,
        help="Keep calendar rows where at least this fraction of names have a close.",
    )
    parser.add_argument(
        "--clean-window",
        type=int,
        default=400,
        help="Names must have complete closes over this many trailing rows to be replayed.",
    )
    parser.add_argument("--cash", type=float, default=100_000.0, help="Starting cash.")
    parser.add_argument(
        "--positions",
        default=None,
        help='Starting positions as JSON, e.g. \'{"COST": 1, "MMM": -3}\' — rehearse a cutover '
        "from an inherited book. Names must survive the clean screen (they are passed as keep=).",
    )
    parser.add_argument("--position-size", type=float, default=0.05)
    parser.add_argument("--max-gross", type=float, default=1.0)
    parser.add_argument("--max-symbol-weight", type=float, default=0.10)
    parser.add_argument("--band", type=float, default=0.0, help="Online no-trade half-width (0 disables).")
    parser.add_argument("--min-notional", type=float, default=1.0)
    parser.add_argument(
        "--tif",
        choices=("opg", "day"),
        default="opg",
        help="Sizing semantics to model: opg = whole shares (the live OPG book, "
        "default = parity with every prior replay), day = fractional shares "
        "(the live --tif day path). Fills are modeled either way.",
    )
    parser.add_argument("--horizon", type=int, default=5, help="Ensemble signal forward horizon in bars.")
    return parser.parse_args(argv)


def _load_universe_file(path: Path) -> list[str]:
    """One symbol per line; blank lines and ``#`` comments skipped, upper-cased."""
    symbols = [
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not symbols:
        raise ValueError(f"universe file {path} has no symbols")
    return symbols


def _book_config(args: argparse.Namespace, decision_every: int) -> DailyBookConfig:
    """Construction config for the parsed args — sizing semantics wired once.

    Both book paths thread ``whole_shares=(--tif == "opg")``: opg models the
    live whole-share OPG book (parity with every prior scripted replay), day
    the fractional-share sizing of paper_loop's ``--tif day`` path. Extracted
    from the two inline constructions so the wiring is testable without a
    replay run (tests/test_live_replay.py).
    """
    whole_shares = args.tif == "opg"
    if args.book == "momentum":
        return DailyBookConfig(
            book="decile_neutral",
            decile=args.decile,
            decision_every=decision_every,
            max_gross=args.max_gross,
            max_symbol_abs_weight=args.max_symbol_weight,
            no_trade_band=args.band,
            min_order_notional=args.min_notional,
            whole_shares=whole_shares,
        )
    if args.book == "trend":
        return DailyBookConfig(
            book="inverse_vol",
            vol_ewma_bars=args.vol_ewma_bars,
            decision_every=decision_every,
            max_gross=args.max_gross,
            max_symbol_abs_weight=args.max_symbol_weight,
            no_trade_band=args.band,
            min_order_notional=args.min_notional,
            whole_shares=whole_shares,
        )
    return DailyBookConfig(
        position_size=args.position_size,
        max_gross=args.max_gross,
        max_symbol_abs_weight=args.max_symbol_weight,
        no_trade_band=args.band,
        min_order_notional=args.min_notional,
        whole_shares=whole_shares,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    if args.universe_file is not None:
        symbols = _load_universe_file(args.universe_file)
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.book == "trend":
        symbols = list(TREND_V1_UNIVERSE)
    else:
        raise SystemExit("provide --symbols or --universe-file")
    positions = {k.upper(): float(v) for k, v in json.loads(args.positions).items()} if args.positions else {}

    max_missing = (
        args.max_missing
        if args.max_missing is not None
        else (0.10 if args.book in ("momentum", "trend") else 0.0)
    )
    close, volume, open_ = load_local_bar_panels(symbols, args.data_dir, max_missing=max_missing)
    close, volume, open_ = align_replay_panels(
        close,
        volume,
        open_,
        consensus=args.consensus,
        clean_window=args.clean_window,
        keep=tuple(positions),
    )
    logger.info(
        "panels: %d bars x %d names, %s .. %s (local caches, offline)",
        len(close),
        close.shape[1],
        close.index[0].date(),
        close.index[-1].date(),
    )

    decision_every = args.decision_every if args.decision_every is not None else (
        21 if args.book in ("momentum", "trend") else 1
    )
    config = _book_config(args, decision_every)
    if args.book == "momentum":
        membership_mask = None
        if args.membership_file is not None:
            membership = pd.read_parquet(args.membership_file)
            # Built over the ALIGNED panel; compute_eligibility reindexes it to
            # each cycle's truncated sub-panel, so one full-panel mask serves
            # every cycle.
            membership_mask = build_membership_mask(membership, close.index, close.columns)
            logger.info(
                "membership mask: %s — %d member-days over %d names",
                args.membership_file,
                int(membership_mask.to_numpy().sum()),
                membership_mask.shape[1],
            )

        def signal_factory() -> Signal:
            return MomentumSignalNode(
                ResidualStatArbConfig(),
                lookback_bars=args.mom_lookback,
                skip_bars=args.mom_skip,
                horizon_bars=decision_every,
                membership_mask=membership_mask,
            )

    elif args.book == "trend":

        def signal_factory() -> Signal:
            return TrendSignalNode(
                lookback_bars=args.mom_lookback,
                skip_bars=args.mom_skip,
                horizon_bars=decision_every,
            )

    else:

        def signal_factory() -> Signal:
            return EnsembleSignalNode(EnsembleNodeConfig(horizon_bars=args.horizon))

    results, broker = replay_daily_cycles(
        signal_factory,
        close,
        volume,
        open_,
        config,
        args.run_dir,
        start_bar=args.start,
        end_bar=args.end,
        initial_cash=args.cash,
        initial_positions=positions,
    )

    pattern = "".join("R" if r.submitted_orders else "." for r in results)
    first, last = results[0], results[-1]
    monitor = last.monitor_read or {}
    logger.info("order pattern (R=traded, .=hold): %s", pattern)
    logger.info(
        "replay done: %d cycles %s..%s, equity %.2f -> %.2f (%+.4f%%), final book %d names, "
        "monitor n=%s verdict=%s — MODELED FILLS: diagnostic stream, not calibration or concordance",
        len(results),
        first.decision_bar,
        last.decision_bar,
        first.equity,
        last.equity,
        100.0 * (last.equity / first.equity - 1.0),
        sum(1 for q in broker.positions().values() if q != 0.0),
        monitor.get("n"),
        monitor.get("verdict"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
