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

    # ensemble cost instrument (default)
    python -m prism.scripts.paper_loop --symbols AAPL,MSFT,GOOG \\
        --run-dir runs/paper_loop --position-size 0.05

    # the ratified B1 momentum book (12-1 decile L/S, monthly) on a PIT universe
    python -m prism.scripts.paper_loop --book momentum \\
        --universe-file data/universe/sp500_current.txt --run-dir runs/paper_loop

Credentials: ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` (paper endpoint
by default; ``APCA_API_BASE_URL`` overrides). Bars come from Alpaca's own IEX
feed by default (``--bar-source``); the Twelve Data spine stays a fallback.

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
    SafetyConfig,
    StateStore,
    fetch_universe_panels,
    run_daily_cycle,
)
from prism.residual.factors import ResidualStatArbConfig
from prism.signal.ensemble_node import EnsembleNodeConfig, EnsembleSignalNode
from prism.signal.momentum_node import MomentumSignalNode

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One daily paper-loop cycle: settle -> fetch -> score -> construct -> submit."
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated universe (e.g. AAPL,MSFT,GOOG). Ignored when --universe-file is given.",
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=None,
        help="File with one symbol per line (blank lines and #comments skipped); overrides "
        "--symbols. Use it for the ~500-name momentum book — a 15-name --symbols list gives a "
        "degenerate one-per-leg decile.",
    )
    parser.add_argument(
        "--book",
        choices=("ensemble", "momentum"),
        default="ensemble",
        help="Which book to trade. 'ensemble' (default) is the XGBoost+ARIMA directional cost "
        "instrument; 'momentum' is the ratified B1 candidate (12-1 cross-sectional momentum, "
        "decile long/short, neutral by balanced legs, monthly cadence — docs/momentum_design.md). "
        "The live-monitor read is a B1 concordance read only under 'momentum'.",
    )
    parser.add_argument("--mom-lookback", type=int, default=252, help="Momentum lookback bars (B1: 252).")
    parser.add_argument("--mom-skip", type=int, default=21, help="Momentum skip bars (B1: 21).")
    parser.add_argument("--decile", type=float, default=0.10, help="Decile fraction per leg (B1: 0.10).")
    parser.add_argument(
        "--decision-every",
        type=int,
        default=None,
        help="Refresh cadence in trading sessions; default 1 (daily) for ensemble, 21 "
        "(≈monthly) for momentum — B1's cadence.",
    )
    parser.add_argument(
        "--max-missing",
        type=float,
        default=None,
        help="Fraction of the universe allowed to return no bars before failing loud; "
        "default 0.0 (ensemble, strict) or 0.10 (momentum, tolerates a few stale/renamed "
        "tickers the venue no longer serves — they are dropped with a warning naming them).",
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
    parser.add_argument(
        "--kill-switch",
        type=Path,
        default=None,
        help="Halt-file path; its PRESENCE stops the book (settle + NAV mark only, no orders, "
        "exit 2). Default {run-dir}/KILL_SWITCH; 'off' disables the rail.",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.5,
        help="Peak-to-current drawdown fraction on the equity ledger beyond which the book "
        "halts (default 0.5 — catastrophic-only for the paper instrument; tighten for real "
        "money). 0 disables.",
    )
    parser.add_argument(
        "--max-order-fraction",
        type=float,
        default=None,
        help="Per-order notional bound as a fraction of equity, enforced before the write-ahead "
        "persist. Default 2x --max-symbol-weight (no legitimate single order can exceed 1x "
        "under the down-only caps and the flip-to-flat clamp). 0 disables.",
    )
    parser.add_argument(
        "--max-orders",
        type=int,
        default=None,
        help="Per-decision order-count bound. Default 2x universe size + 10 (an order per "
        "name entering plus one per name exiting is the geometric ceiling). 0 disables.",
    )
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


def _with_held_names(symbols: list[str], state: Any) -> tuple[list[str], list[str]]:
    """Fetch universe = configured universe ∪ persisted held book.

    A held name must stay fetchable until the book exits it: the mark step
    values every position and refuses loudly otherwise (N7), and only a priced
    name can be rebalanced to flat. Index leavers drop out of a regenerated
    universe file while the book still holds them (POOL, 2026-07-15), so the
    extras ride along for valuation/exit — the caller keeps them OUT of the
    scoring universe so they can leave the book but never re-enter it.
    """
    held = getattr(state, "positions", None) if state is not None else None
    extras = sorted(set(held) - set(symbols)) if held else []
    return symbols + extras, extras


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    if args.universe_file is not None:
        symbols = _load_universe_file(args.universe_file)
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        raise SystemExit("provide --symbols or --universe-file")

    score_universe = list(symbols)
    symbols, valuation_extras = _with_held_names(symbols, StateStore(args.run_dir / "state.json").load())
    if valuation_extras:
        logger.warning(
            "universe file lacks %d held name(s) %s; fetching them for valuation/exit only "
            "(they are masked out of scoring and exit at the next refresh)",
            len(valuation_extras),
            valuation_extras,
        )

    # Bars from Alpaca (the broker's own feed) by default so decision and fill
    # share one venue and clock; the Twelve Data spine stays available as a
    # fallback. Both satisfy the duck-typed fetch_incremental read path.
    loader: Any = AlpacaBarSource.from_env() if args.bar_source == "alpaca" else DataLoader()
    start_date = args.start_date
    if start_date is None and args.book == "momentum":
        # ~3y is ample for the 252-bar lookback + eligibility window and keeps the
        # universe-scale batch fetch to a handful of pages.
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    max_missing = args.max_missing if args.max_missing is not None else (0.10 if args.book == "momentum" else 0.0)
    close, volume = fetch_universe_panels(loader, symbols, start_date=start_date, max_missing=max_missing)
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

    decision_every = args.decision_every if args.decision_every is not None else (
        21 if args.book == "momentum" else 1
    )
    if args.book == "momentum":
        # Valuation-only extras are masked ineligible: NaN-scored names get an
        # explicit 0.0 from the decile construct (its explicit-flat pin), so a
        # departed-but-held name exits at the next refresh and cannot re-enter.
        score_mask = None
        if valuation_extras:
            score_mask = pd.DataFrame(False, index=close.index, columns=close.columns)
            score_mask.loc[:, [s for s in score_universe if s in close.columns]] = True
        signal = MomentumSignalNode(
            ResidualStatArbConfig(),
            lookback_bars=args.mom_lookback,
            skip_bars=args.mom_skip,
            horizon_bars=decision_every,
            membership_mask=score_mask,
        )
        signal.fit(close, volume)
        config = DailyBookConfig(
            book="decile_neutral",
            decile=args.decile,
            decision_every=decision_every,
            max_gross=args.max_gross,
            max_symbol_abs_weight=args.max_symbol_weight,
            no_trade_band=args.band,
            max_participation=args.max_participation,
            min_order_notional=args.min_notional,
            whole_shares=(args.tif == "opg"),
        )
        logger.info(
            "book=momentum: 12-1 decile L/S, lookback=%d skip=%d decile=%.2f cadence=%d, %d symbols",
            args.mom_lookback,
            args.mom_skip,
            args.decile,
            decision_every,
            len(symbols),
        )
    else:
        signal = EnsembleSignalNode(EnsembleNodeConfig(horizon_bars=args.horizon))
        signal.fit(close, volume)
        config = DailyBookConfig(
            position_size=args.position_size,
            max_gross=args.max_gross,
            max_symbol_abs_weight=args.max_symbol_weight,
            no_trade_band=args.band,
            max_participation=args.max_participation,
            min_order_notional=args.min_notional,
            whole_shares=(args.tif == "opg"),
        )
        logger.info(
            "book=ensemble: %d symbols, weights %s",
            len(signal.fitted_symbols_),
            signal.weight_basis_,
        )

    # Safety rails (prism.live.safety): inert at this book's normal state, loud
    # at pathology. The notional bound is derived from the construction cap
    # (2x is slack for whole-share rounding), the order-count bound from the
    # universe's geometric ceiling; both catch order-of-magnitude corruption,
    # not strategy behavior. 'off'/0 disables a rail explicitly.
    kill_switch: Path | None
    if args.kill_switch is None:
        kill_switch = args.run_dir / "KILL_SWITCH"
    elif str(args.kill_switch) == "off":
        kill_switch = None
    else:
        kill_switch = args.kill_switch
    max_order_fraction = (
        args.max_order_fraction if args.max_order_fraction is not None else 2.0 * args.max_symbol_weight
    )
    max_orders = args.max_orders if args.max_orders is not None else 2 * len(symbols) + 10
    safety = SafetyConfig(
        kill_switch=kill_switch,
        max_drawdown=args.max_drawdown if args.max_drawdown > 0 else None,
        max_order_fraction=max_order_fraction if max_order_fraction > 0 else None,
        max_orders=max_orders if max_orders > 0 else None,
    )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    ctx = LiveLoopContext(
        store=StateStore(args.run_dir / "state.json"),
        broker=AlpacaBroker.from_env(time_in_force=args.tif),
        fills_ledger=args.run_dir / "fills.jsonl",
        equity_ledger=args.run_dir / "equity.jsonl",
        targets_ledger=args.run_dir / "targets.jsonl",
        unfilled_ledger=args.run_dir / "unfilled.jsonl",
        concordance_ledger=args.run_dir / "concordance.jsonl",
        # Namespaced client ids: two books sharing one venue account with the
        # bare {bar}:{symbol} scheme silently substitute each other's same-bar
        # orders (duplicate-id == success). Persisted pending orders keep the
        # ids they were decided with, so flipping the prefix is resume-safe.
        order_id_prefix="mom:" if args.book == "momentum" else "",
    )
    result = run_daily_cycle(ctx, signal, close, volume, config, safety=safety)

    held = result.target_weights[result.target_weights.abs() > 1e-9]
    logger.info(
        "cycle %s: settled %d fills, submitted %d orders, equity %.2f, book %s",
        result.decision_bar,
        len(result.settled_fills),
        len(result.submitted_orders),
        result.equity,
        {k: round(v, 4) for k, v in held.sort_values().items()},
    )
    if result.halted is not None:
        # Exit non-zero so the nightly wrapper's failure path fires: a halted
        # book — even a deliberately halted one — is a state that demands eyes.
        logger.error("cycle %s HALTED: %s", result.decision_bar, result.halted)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
