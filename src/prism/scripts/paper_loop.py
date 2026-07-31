"""Alpaca paper-loop CLI — the I-9 cost-measurement instrument (SPEC §13, R2).

A thin, network-gated shell: it parses arguments and connects credentials to
pieces that are each tested offline — ``DataLoader.fetch_incremental``
(tests/test_incremental_store.py), ``EnsembleSignalNode``
(tests/test_signal_node.py), the broker-truth universe reconciliation
(``resolve_fetch_universe``; tests/test_paper_loop_universe.py), the online
construction and write-ahead protocol (tests/test_live_daily.py,
tests/test_live_loop.py), the Alpaca venue mappings
(tests/test_live_alpaca.py), and the per-book assembly registry
(``prism.scripts.paper_books``; tests/test_paper_books.py). The shell itself
holds no logic worth testing; anything that grows logic must move down into
``prism.live.daily`` (cycle logic) or ``prism.scripts.paper_books`` (CLI
assembly) where it can be exercised without credentials. The whole mutating
path runs under the run directory's writer lock
(``prism.live.lockfile.run_dir_lock``): the write-ahead protocol survives
crashes, not two concurrent writers.

Run once per session, after the close (OPG next-open orders must reach
Alpaca before ~09:28 ET the next morning):

    # ensemble cost instrument (default)
    python -m prism.scripts.paper_loop --symbols AAPL,MSFT,GOOG \\
        --run-dir runs/paper_loop --position-size 0.05

    # the ratified B1 momentum book (12-1 decile L/S, monthly) on a PIT universe
    python -m prism.scripts.paper_loop --book momentum \\
        --universe-file data/universe/sp500_current.txt --run-dir runs/paper_loop

    # trend sleeve (TSMOM ETF book, inverse-vol; uncounted paper instrument)
    python -m prism.scripts.paper_loop --book trend \\
        --run-dir runs/paper_loop_trend

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
import json
import logging
import sys
from pathlib import Path
from typing import Any

from prism.io.loader import DataLoader
from prism.live import (
    PROFILE_IDS,
    AlpacaBarSource,
    AlpacaBroker,
    LiveLoopContext,
    RegimeTelemetry,
    StateStore,
    assert_research_paper_bit_identity,
    fetch_universe_panels,
    resolve_fetch_universe,
    resolve_risk_profile,
    run_daily_cycle,
    spinoff_unrankable_provider,
)
from prism.live.daily import warn_if_stale_panel
from prism.live.lockfile import run_dir_lock
from prism.scripts.paper_books import (
    BOOKS,
    BookParams,
    apply_certified_paper_pin,
    build_safety_config,
    default_panel_start,
    resolve_cli_universe,
    valuation_masked_membership,
)

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
        choices=("ensemble", "momentum", "trend"),
        default="ensemble",
        help="Which book to trade. 'ensemble' (default) is the XGBoost+ARIMA directional cost "
        "instrument; 'momentum' is the ratified B1 candidate (12-1 cross-sectional momentum, "
        "decile long/short, neutral by balanced legs, monthly cadence — docs/momentum_design.md); "
        "'trend' is the trend sleeve (per-name 12−1 TSMOM on the pinned 10-ETF universe, "
        "inverse-vol construct, monthly cadence — docs/trend_design.md; uncounted paper "
        "instrument — not a counted trial). The live-monitor concordance read is B1-shaped "
        "only under 'momentum'.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_IDS),
        default=None,
        help="Optional W6 risk profile id (docs/risk_profile_schema.md, FROZEN). "
        "research_paper forces the certified B1 paper path (book=momentum + pinned "
        "DailyBookConfig) and refuses construction flag overrides that would fork it. "
        "Other profiles write run-dir metadata + tighten-only construction; they are "
        "not a GO authorization (handoff §8 still binds).",
    )
    parser.add_argument("--mom-lookback", type=int, default=252, help="Momentum/trend lookback bars (B1/T0: 252).")
    parser.add_argument("--mom-skip", type=int, default=21, help="Momentum/trend skip bars (B1/T0: 21).")
    parser.add_argument("--decile", type=float, default=0.10, help="Decile fraction per leg (B1: 0.10).")
    parser.add_argument(
        "--vol-ewma-bars",
        type=int,
        default=63,
        help="Trend inverse-vol EWMA window in bars (docs/trend_design.md §2: 63).",
    )
    parser.add_argument(
        "--decision-every",
        type=int,
        default=None,
        help="Refresh cadence in trading sessions; default 1 (daily) for ensemble, 21 "
        "(≈monthly) for momentum/trend — B1/T0 cadence.",
    )
    parser.add_argument(
        "--max-missing",
        type=float,
        default=None,
        help="Fraction of the universe allowed to return no bars before failing loud; "
        "default 0.0 (ensemble, strict) or 0.10 (momentum/trend, tolerates a few stale "
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
        "--spinoff-mask",
        action="store_true",
        help="Mask names with an Alpaca-reported spin-off inside the momentum lookback as "
        "unrankable this refresh (docs/bar_vendor_divergence.md §5): no new position may "
        "open on a divergent rank; a held name is held until the window clears the event. "
        "Momentum book only; default off. Detection is cached per decision bar in --run-dir; "
        "a detection failure warns loudly and the refresh proceeds unmasked.",
    )
    parser.add_argument(
        "--regime",
        action="store_true",
        help="Read SPEC §7.5 regime telemetry every cycle (FRED curve / net liquidity / "
        "inflation + VIX term structure; requires FRED_API_KEY) into {run-dir}/regime.jsonl — "
        "the handoff §8 precondition-(b) session record (docs/regime_step.md). Telemetry only: "
        "the de-gross action hook has no CLI path and stays unarmed until "
        "docs/sizing_preregistration.md ratifies. Default off.",
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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    # W6 frozen: optional named profile. research_paper is the G6 instrument —
    # pin construction to CERTIFIED_B1_PAPER_CONFIG and refuse silent forks.
    # Non-paper profiles: metadata + tighten-only book config; multi-sleeve
    # live routing and GO still require handoff §8 + sleeve admission.
    params = BookParams(
        lookback_bars=args.mom_lookback,
        skip_bars=args.mom_skip,
        decile=args.decile,
        vol_ewma_bars=args.vol_ewma_bars,
        horizon_bars=args.horizon,
        decision_every=args.decision_every,
        position_size=args.position_size,
        max_gross=args.max_gross,
        max_symbol_abs_weight=args.max_symbol_weight,
        no_trade_band=args.band,
        max_participation=args.max_participation,
        min_order_notional=args.min_notional,
        whole_shares=(args.tif == "opg"),
    )
    resolved_profile = None
    if args.profile is not None:
        resolved_profile = resolve_risk_profile(args.profile)
        if args.profile == "research_paper":
            if args.book not in ("ensemble", "momentum"):
                raise SystemExit(
                    "--profile research_paper requires --book momentum "
                    f"(got {args.book!r}); the certified paper path is B1 only"
                )
            args.book = "momentum"
            if args.tif != "opg":
                raise SystemExit(
                    "--profile research_paper requires --tif opg (whole-share "
                    "certified paper path)"
                )
            params = apply_certified_paper_pin(params, resolved_profile.book_config)
            logger.info(
                "profile=research_paper: pinned to certified B1 paper DailyBookConfig "
                "(G6; docs/risk_profile_schema.md FROZEN)"
            )
        else:
            logger.info(
                "profile=%s resolved (schema FROZEN); book=%s max_gross=%.3f "
                "de_gross_armed=%s — not a GO authorization",
                args.profile,
                args.book,
                resolved_profile.book_config.max_gross,
                resolved_profile.hedge.de_gross_armed,
            )

    spec = BOOKS[args.book]
    symbols = resolve_cli_universe(args.universe_file, args.symbols, spec)
    if args.spinoff_mask and args.book != "momentum":
        raise SystemExit(
            "--spinoff-mask applies to --book momentum only (the lookback-window mechanic, "
            "docs/bar_vendor_divergence.md §5)"
        )

    # One writer per run directory: state.json and the ledgers are shared with
    # the morning sweep and any manual invocation, and the write-ahead protocol
    # protects against crashes, not concurrent writers. Held through the whole
    # mutating path (reconcile reads state; the cycle writes it).
    with run_dir_lock(args.run_dir):
        # Broker truth decides the fetch universe, BEFORE any bar is fetched. The
        # mark step values what the venue holds, and this run directory's persisted
        # state is not that: a fresh run directory persists nothing (every cycle
        # 2026-07-23..29 died on `cannot value held position 'POOL'`), and a
        # populated one is only a cache of the last reconcile. One broker instance
        # serves the reconciliation and the cycle, so the account is read on the
        # same credentials the orders will use.
        store = StateStore(args.run_dir / "state.json")
        broker = AlpacaBroker.from_env(time_in_force=args.tif)
        universe = resolve_fetch_universe(symbols, broker, state=store.load())
        score_universe = list(universe.score)
        symbols = list(universe.fetch)
        valuation_extras = list(universe.valuation_only)

        # Bars from Alpaca (the broker's own feed) by default so decision and fill
        # share one venue and clock; the Twelve Data spine stays available as a
        # fallback. Both satisfy the duck-typed fetch_incremental read path.
        loader: Any = AlpacaBarSource.from_env() if args.bar_source == "alpaca" else DataLoader()
        start_date = args.start_date if args.start_date is not None else default_panel_start(spec)
        max_missing = args.max_missing if args.max_missing is not None else spec.default_max_missing
        # ``required``: a held name may not be silently dropped by the max_missing
        # tolerance — an unpriceable position cannot be marked or exited (N7).
        close, volume = fetch_universe_panels(
            loader,
            symbols,
            start_date=start_date,
            max_missing=max_missing,
            required=sorted(universe.held),
        )
        logger.info(
            "panels: %d bars x %d symbols through %s (source=%s)",
            len(close),
            close.shape[1],
            close.index[-1],
            args.bar_source,
        )
        warn_if_stale_panel(close)

        decision_every = spec.decision_every(params)
        # Valuation-only extras are masked ineligible (momentum only): NaN-scored
        # names get an explicit 0.0 from the decile construct (its explicit-flat
        # pin), so a departed-but-held name exits at the next refresh and cannot
        # re-enter.
        score_mask = (
            valuation_masked_membership(close, score_universe, bool(valuation_extras))
            if args.book == "momentum"
            else None
        )
        signal = spec.build_signal(params, decision_every, score_mask)
        signal.fit(close, volume)
        config = spec.build_config(params, decision_every)
        logger.info("%s", spec.describe(params, decision_every, len(symbols), signal))

        # Spin-off eligibility mask (docs/bar_vendor_divergence.md §5): a name with
        # a spin-off inside the momentum lookback is unrankable this refresh.
        # The window/universe/intersection decisions live in the tested factory
        # (prism.live.spinoff_mask.spinoff_unrankable_provider) — this shell only
        # connects it. Consulted by run_daily_cycle on refresh sessions only.
        unrankable = (
            spinoff_unrankable_provider(close, params.lookback_bars, score_universe, args.run_dir)
            if args.spinoff_mask
            else None
        )

        # SPEC §7.7 regime step (docs/regime_step.md): §7.5 telemetry every cycle
        # when armed, with the real FRED client from the environment. A missing
        # FRED_API_KEY fails loud here, before any venue call (N7) — the operator
        # asked for the regime record, so a cycle without one must not run
        # quietly. Telemetry only: the gross-scale ACTION hook is deliberately
        # not constructible from the CLI; it arms in code only after
        # docs/sizing_preregistration.md ratifies.
        regime_provider = RegimeTelemetry.from_env() if args.regime else None

        safety = build_safety_config(
            run_dir=args.run_dir,
            kill_switch_arg=args.kill_switch,
            max_drawdown=args.max_drawdown,
            max_order_fraction=args.max_order_fraction,
            max_symbol_abs_weight=params.max_symbol_abs_weight,
            max_orders=args.max_orders,
            n_symbols=len(symbols),
        )

        args.run_dir.mkdir(parents=True, exist_ok=True)
        if resolved_profile is not None:
            if args.profile == "research_paper":
                # G6: the constructed config must match the certified paper pin.
                assert_research_paper_bit_identity(config)
            profile_path = args.run_dir / "profile.json"
            profile_path.write_text(
                json.dumps(resolved_profile.to_public_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            logger.info("wrote %s", profile_path)
        ctx = LiveLoopContext(
            store=store,
            broker=broker,
            fills_ledger=args.run_dir / "fills.jsonl",
            equity_ledger=args.run_dir / "equity.jsonl",
            targets_ledger=args.run_dir / "targets.jsonl",
            unfilled_ledger=args.run_dir / "unfilled.jsonl",
            concordance_ledger=args.run_dir / "concordance.jsonl",
            regime_ledger=args.run_dir / "regime.jsonl",
            # Namespaced client ids: two books sharing one venue account with the
            # bare {bar}:{symbol} scheme silently substitute each other's same-bar
            # orders (duplicate-id == success). Persisted pending orders keep the
            # ids they were decided with, so flipping the prefix is resume-safe.
            order_id_prefix=spec.order_id_prefix,
        )
        result = run_daily_cycle(
            ctx, signal, close, volume, config, safety=safety, unrankable=unrankable, regime=regime_provider
        )

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
