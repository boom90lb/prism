"""One daily cycle of the paper loop (SPEC.md §7.7) — the spine, wired.

This is the driver the live package existed to host: it connects the delta
fetch (``io``/loader), a fitted :class:`~prism.signal.base.Signal`, online
construction (down-only caps → the stateful no-trade band against *held*
weights → the participation gate), and the write-ahead decision protocol
(``prism.live.loop``), with last session's fills settled into the I-9 ledger
first. The residualize stage (§7.2) is not wired yet — the residual node is
sequenced behind the §10 demotion runs — so the driver takes any Signal and
constructs an unhedged directional book; the paper instrument's purpose is
cost measurement (SPEC §13 R2), which needs turnover, not edge.

Cadence and ordering (one call per session, after the close):

1. **Settle.** If a prior decision is pending for an *earlier* bar, its
   fills (which occurred at an open strictly before today's close) are
   pulled and ledgered. A pending decision for *today's* bar means this is
   a same-session restart: skip settle and let the write-ahead protocol
   resume submission idempotently.
2. **Mark.** Broker-truth positions and cash are marked at today's closes
   into held weights and equity — the same marks ``decide_and_submit``
   sizes with.
3. **Score → construct.** The signal scores the panel's last row; scores
   map linearly to capped weights (I-4, one sizing function), the online
   band holds names whose target moved less than the band from what is
   held, and the participation gate shrinks trades that exceed the %ADV
   cap (§7.4).
4. **Decide/submit.** ``decide_and_submit`` persists the decision before
   the first submit and submits idempotently (N2 crash window).

Fit cadence is deliberately *not* owned here: the caller passes a fitted
Signal, and which model serves "today" is the operator's explicit staleness
policy (§7.7), not a hidden side effect of the driver.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from prism.execution.participation import participation_capped_targets
from prism.live.broker import Fill, Order
from prism.live.loop import LiveLoopContext, _require_price, decide_and_submit, settle
from prism.portfolio.construct import construct_directional_targets, step_no_trade_band
from prism.signal.base import Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailyBookConfig:
    """Construction policy for the daily book — sizing folds in once (I-4).

    ``position_size`` maps a full-conviction score (|score| ≥ 1) to a
    per-name weight; caps are down-only (a low-gross book stays low-gross).
    ``no_trade_band`` is the *online* half-width applied against held
    weights (the batch band inside ``construct_directional_targets`` stays
    off — replaying hysteresis from flat every day is the defect the online
    step exists to fix). ``max_participation`` of None disables the %ADV
    gate (at paper scale it never binds; enable it before any AUM claim).
    ``whole_shares`` matches Alpaca OPG (next-open) order requirements.
    """

    position_size: float
    max_gross: float = 1.0
    max_symbol_abs_weight: float = 0.10
    no_trade_band: float = 0.0
    max_participation: float | None = None
    adv_window_bars: int = 20
    min_order_notional: float = 1.0
    whole_shares: bool = True


@dataclass(frozen=True)
class DailyCycleResult:
    """What one cycle did, for the operator's log and the session record."""

    decision_bar: str
    settled_fills: list[Fill]
    submitted_orders: list[Order]
    equity: float
    target_weights: pd.Series


def fetch_universe_panels(
    loader: Any,
    symbols: Sequence[str],
    *,
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    store: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Wide (close, volume) panels via the incremental store's delta fetch.

    ``loader`` is duck-typed to ``DataLoader.fetch_incremental`` (SPEC §7.0
    read path: seed once, then one tail request per symbol per day against
    the vendor budget). A symbol that yields no bars raises (N7): the book
    cannot silently shrink because a fetch failed. Returns ``volume=None``
    when no symbol carries a volume column.
    """
    if not symbols:
        raise ValueError("empty symbol list — nothing to fetch")
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for symbol in symbols:
        bars = loader.fetch_incremental(
            symbol, interval=interval, start_date=start_date, end_date=end_date, store=store
        )
        if bars.empty or "close" not in bars.columns:
            raise RuntimeError(
                f"no bars for {symbol!r} ({interval}) from the incremental fetch; "
                "fix the source or drop the symbol explicitly (N7)"
            )
        closes[symbol] = pd.to_numeric(bars["close"], errors="coerce")
        if "volume" in bars.columns:
            volumes[symbol] = pd.to_numeric(bars["volume"], errors="coerce")
    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex(columns=close.columns).sort_index() if volumes else None
    return close, volume


def run_daily_cycle(
    ctx: LiveLoopContext,
    signal: Signal,
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    config: DailyBookConfig,
    decision_bar: str | None = None,
) -> DailyCycleResult:
    """Run one settle → mark → score → construct → decide/submit cycle.

    ``close``/``volume`` are the full trailing panels whose last row is the
    decision bar *t* (N1); ``decision_bar`` defaults to that row's date.
    Re-running within the same bar resumes the persisted decision without
    settling or re-deciding (the write-ahead protocol's restart semantics).
    """
    if close.empty:
        raise ValueError("empty close panel — nothing to decide on")
    if len(close) < signal.required_history:
        raise ValueError(
            f"panel has {len(close)} rows < signal required_history "
            f"{signal.required_history}; refusing to score a truncated panel (N7)"
        )
    decision_bar = decision_bar or str(close.index[-1].date())

    # 1. Settle the previous bar's decision. Its settle marker is the bar the
    # decision was made on ("that decision bar is fully processed"), keeping
    # today's decision_bar strictly greater for the re-decide guard.
    settled: list[Fill] = []
    state = ctx.store.load()
    if state is not None and state.pending_orders:
        if state.pending_decision_bar == decision_bar:
            logger.info(
                "pending decision for %s already persisted — resuming submission, not settling",
                decision_bar,
            )
        else:
            settled = settle(ctx, state.pending_decision_bar)
            logger.info(
                "settled %d fills from decision bar %s", len(settled), state.pending_decision_bar
            )

    # 2. Mark broker truth at today's closes.
    prices = close.iloc[-1]
    positions = {s: q for s, q in ctx.broker.positions().items() if q != 0.0}
    cash = ctx.broker.cash()
    equity = cash + sum(shares * _require_price(prices, s) for s, shares in positions.items())
    if equity <= 0:
        raise RuntimeError(f"non-positive equity {equity}; the loop cannot size a book (N7)")
    prev_weights = pd.Series(
        {s: shares * _require_price(prices, s) / equity for s, shares in positions.items()},
        dtype=float,
    )

    # 3. Score and construct against held weights.
    scores = signal.score(close, volume)
    raw = construct_directional_targets(
        pd.DataFrame([scores], index=close.index[-1:]),
        position_size=config.position_size,
        max_gross=config.max_gross,
        max_symbol_abs_weight=config.max_symbol_abs_weight,
        no_trade_band=0.0,  # the online step below owns hysteresis (never the flat replay)
    ).iloc[0]
    targets = step_no_trade_band(prev_weights, raw, config.no_trade_band)
    if config.max_participation is not None:
        if volume is None:
            raise ValueError(
                "max_participation is set but there is no volume panel to compute ADV (N7)"
            )
        dollar_volume = (
            (close * volume).rolling(config.adv_window_bars, min_periods=1).mean().iloc[-1]
        )
        targets = participation_capped_targets(
            prev_weights,
            targets,
            dollar_volume,
            aum=equity,
            max_participation=config.max_participation,
        )

    # 4. Write-ahead decide and idempotent submit.
    orders = decide_and_submit(
        ctx,
        decision_bar,
        targets,
        prices,
        min_order_notional=config.min_order_notional,
        whole_shares=config.whole_shares,
    )
    logger.info(
        "decision %s: %d orders submitted, equity %.2f, gross %.4f",
        decision_bar,
        len(orders),
        equity,
        float(targets.abs().sum()),
    )
    return DailyCycleResult(
        decision_bar=decision_bar,
        settled_fills=settled,
        submitted_orders=orders,
        equity=equity,
        target_weights=targets,
    )
