"""Alpaca paper-loop CLI — the I-9 cost-measurement instrument (SPEC §13, R2).

A thin, network-gated shell: it parses arguments and connects credentials to
pieces that are each tested offline — ``DataLoader.fetch_incremental``
(tests/test_incremental_store.py), ``EnsembleSignalNode``
(tests/test_signal_node.py), the online construction and write-ahead
protocol (tests/test_live_daily.py, tests/test_live_loop.py), and the
Alpaca venue mappings (tests/test_live_alpaca.py). The shell itself holds
no logic worth testing; anything that grows logic must move down into
``prism.live.daily`` where it can be exercised without credentials.

Run once per session, after the close (OPG next-open orders must reach
Alpaca before ~09:28 ET the next morning):

    python -m prism.scripts.paper_loop --symbols AAPL,MSFT,GOOG \\
        --run-dir runs/paper_loop --position-size 0.05

Credentials: ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` (paper endpoint
by default; ``APCA_API_BASE_URL`` overrides). Bars come from the Twelve
Data key the loader already uses.

Fit policy (explicit, §7.7 model staleness): the signal is refit every run
on the full trailing panel up to the decision bar. Causal for live use —
fitting on bars ≤ *t* to decide at *t* sees no future — and at paper-book
size the refit cost is minutes. A drift-gated retrain cadence replaces this
when the loop graduates from cost instrument to unattended operation (R4).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from prism.io.loader import DataLoader
from prism.live import (
    AlpacaBarSource,
    AlpacaBroker,
    DailyBookConfig,
    LiveLoopContext,
    StateStore,
    fetch_universe_panels,
    run_daily_cycle,
)
from prism.signal.ensemble_node import EnsembleNodeConfig, EnsembleSignalNode

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One daily paper-loop cycle: settle -> fetch -> score -> construct -> submit."
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated universe (e.g. AAPL,MSFT,GOOG).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/paper_loop"),
        help="Directory for durable state (state.json) and the fills ledger (fills.jsonl).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Panel start (YYYY-MM-DD); default lets the loader/store decide.",
    )
    parser.add_argument("--position-size", type=float, default=0.05)
    parser.add_argument("--max-gross", type=float, default=1.0)
    parser.add_argument("--max-symbol-weight", type=float, default=0.10)
    parser.add_argument(
        "--band",
        type=float,
        default=0.0,
        help="Online no-trade half-width in weight units (0 disables).",
    )
    parser.add_argument(
        "--max-participation",
        type=float,
        default=None,
        help="Per-name %%ADV trade cap (e.g. 0.01); default off — it does not bind at paper scale.",
    )
    parser.add_argument("--min-notional", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=5, help="Signal forward horizon in bars.")
    parser.add_argument(
        "--bar-source",
        choices=("alpaca", "twelvedata"),
        default="alpaca",
        help="Daily-bar source. 'alpaca' (default) reads the broker's own IEX feed so "
        "the decision and the fill share one venue and one clock — no cross-vendor EOD "
        "lag (the 2026-07-07 stall). 'twelvedata' uses the research spine's incremental store.",
    )
    parser.add_argument(
        "--tif",
        choices=("opg", "day"),
        default="opg",
        help="Order time-in-force: opg = next-open auction (N2, whole shares), "
        "day = market at next session (admits fractional shares).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Bars from Alpaca (the broker's own feed) by default so decision and fill
    # share one venue and clock; the Twelve Data spine stays available as a
    # fallback. Both satisfy the duck-typed fetch_incremental read path.
    loader: Any = AlpacaBarSource.from_env() if args.bar_source == "alpaca" else DataLoader()
    close, volume = fetch_universe_panels(loader, symbols, start_date=args.start_date)
    logger.info(
        "panels: %d bars x %d symbols through %s (source=%s)",
        len(close),
        close.shape[1],
        close.index[-1],
        args.bar_source,
    )

    # The instrument decides on the freshest bar the vendor has published. If
    # that bar is many calendar days old the feed is lagging (or the loop has
    # been dark), the decision is being made on stale prices, and any resulting
    # fill carries extra arrival lag versus its close-t reference — flag it
    # loudly (N7) rather than trade on it silently. 4 days clears a normal
    # Fri->Tue holiday weekend; more than that is anomalous.
    last_bar = close.index[-1]
    stale_days = (pd.Timestamp.now(tz=last_bar.tz) - last_bar).days
    if stale_days > 4:
        logger.warning(
            "latest available bar %s is %d calendar days old — the vendor feed is lagging "
            "or the loop has been dark; deciding on a stale panel, and fills will carry "
            "extra arrival lag beyond the close-t reference (verify before trusting them)",
            last_bar.date(),
            stale_days,
        )

    signal = EnsembleSignalNode(EnsembleNodeConfig(horizon_bars=args.horizon))
    signal.fit(close, volume)
    logger.info(
        "signal fit: %d symbols, weights %s", len(signal.fitted_symbols_), signal.weight_basis_
    )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    ctx = LiveLoopContext(
        store=StateStore(args.run_dir / "state.json"),
        broker=AlpacaBroker.from_env(time_in_force=args.tif),
        fills_ledger=args.run_dir / "fills.jsonl",
        equity_ledger=args.run_dir / "equity.jsonl",
    )
    config = DailyBookConfig(
        position_size=args.position_size,
        max_gross=args.max_gross,
        max_symbol_abs_weight=args.max_symbol_weight,
        no_trade_band=args.band,
        max_participation=args.max_participation,
        min_order_notional=args.min_notional,
        whole_shares=(args.tif == "opg"),
    )
    result = run_daily_cycle(ctx, signal, close, volume, config)

    held = result.target_weights[result.target_weights.abs() > 1e-9]
    logger.info(
        "cycle %s: settled %d fills, submitted %d orders, equity %.2f, book %s",
        result.decision_bar,
        len(result.settled_fills),
        len(result.submitted_orders),
        result.equity,
        {k: round(v, 4) for k, v in held.sort_values().items()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
