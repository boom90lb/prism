"""One daily cycle of the paper loop (SPEC.md §7.7) — the spine, wired.

This is the driver the live package existed to host: it connects the delta
fetch (``io``/loader), a fitted :class:`~prism.signal.base.Signal`, online
construction (down-only caps → the stateful no-trade band against *held*
weights → the participation gate), and the write-ahead decision protocol
(``prism.live.loop``), with last session's fills settled into the I-9 ledger
first. The driver constructs one of three books per ``DailyBookConfig.book``:
an unhedged directional book (linear score → weight), the market-neutral
decile long/short book the ratified B1 momentum candidate trades, or the
trend sleeve's inverse-vol construct (``inverse_vol``; docs/trend_design.md
§2, default-off paper instrument). The §7.2 factor-residualize stage is
unwired for all three — the directional book is a cost-measurement
instrument (SPEC §13 R2, which needs turnover, not edge), B1 is
eligibility-screened momentum that is market-neutral by balanced decile
legs, not by factor neutralization (adding a residualize stage would change
its ratified config — a discovery event, docs/momentum_design.md §2), and
trend is per-name TSMOM on a fixed ETF list with no residualize stage.

Cadence and ordering (one call per session, after the close):

1. **Settle.** If a prior decision is pending for an *earlier* bar, its
   fills (which occurred at an open strictly before today's close) are
   pulled and ledgered. A pending decision for *today's* bar means this is
   a same-session restart: skip settle and let the write-ahead protocol
   resume submission idempotently.
2. **Mark.** Broker-truth positions and cash are marked at today's closes
   into held weights and equity — the same marks ``decide_and_submit``
   sizes with.
3. **Regime.** When a ``regime`` provider is wired (SPEC §7.5 through the
   §7.7 step, ``prism.live.regime_step``), its telemetry is read for the
   decision bar on every session — refresh or not, halted or not, because
   the handoff §8 precondition-(b) clock counts sessions — and appended to
   the regime ledger. Telemetry only: the de-gross ACTION hook
   (``regime_gross_scale``) stays unarmed until
   docs/sizing_preregistration.md ratifies.
4. **Cadence → score → construct.** On a refresh session (the decision
   cadence has elapsed — daily by default, ≈monthly for B1) the signal
   scores the panel's last row and construction maps scores to capped
   weights (linear for the directional book, top/bottom-decile long/short
   for the neutral book; I-4, one sizing point). The online band holds names
   whose target moved less than the band from what is held, and the
   participation gate shrinks trades over the %ADV cap (§7.4). Off a refresh
   session the book holds its filled weights while NAV still marks daily.
5. **Decide/submit.** ``decide_and_submit`` persists the decision before
   the first submit and submits idempotently (N2 crash window).
6. **Monitor.** After the mark-to-market NAV is appended to the equity
   ledger, the anytime-valid confidence sequence reads the accruing stream
   (:mod:`prism.live.monitor`) as *additive telemetry* beside the ratified
   rolling-PSR promotion read (docs/momentum_design.md). Time-uniform
   coverage makes a per-cycle read safe; it moves no ratified statistic and
   starts no counted trial (SPEC §10).

Fit cadence is deliberately *not* owned here: the caller passes a fitted
Signal, and which model serves "today" is the operator's explicit staleness
policy (§7.7), not a hidden side effect of the driver.

Two pre-cycle helpers live here for the same reason the driver does — the CLI
shell must hold no logic worth testing. :func:`resolve_fetch_universe` splits
the configured universe from the broker's actual book *before* any fetch (the
mark step values venue truth, so the fetch must cover it), and
:func:`fetch_universe_panels` assembles the panels, treating the held names as
must-be-priced rather than droppable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.execution.participation import participation_capped_targets
from prism.live.broker import Broker, Fill, Order
from prism.live.loop import (
    LiveLoopContext,
    append_concordance_ledger,
    append_equity_ledger,
    append_regime_ledger,
    decide_and_submit,
    read_targets_ledger,
    require_price,
    settle,
)
from prism.live.monitor import book_concordance, paper_monitor_read
from prism.live.safety import SafetyConfig, check_orders, halt_reason
from prism.live.state import LoopState
from prism.portfolio.construct import (
    construct_decile_neutral,
    construct_directional_targets,
    construct_inverse_vol_targets,
    step_no_trade_band,
)
from prism.signal.base import Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailyBookConfig:
    """Construction policy for the daily book — sizing folds in once (I-4).

    ``book`` selects the construction: ``"directional"`` maps each score
    linearly to a weight (``position_size`` is the |score|≥1 → weight slope),
    ``"decile_neutral"`` builds the ratified B1 momentum book — equal-weight
    top/bottom ``decile`` long/short, market-neutral by balanced legs, using
    only the cross-sectional *rank* of the scores (``position_size`` is ignored),
    ``"inverse_vol"`` is the trend sleeve construct (docs/trend_design.md §2):
    sign(score) × inverse EWMA-vol, L1-scaled to ``max_gross``
    (``position_size`` / ``decile`` ignored; ``vol_ewma_bars`` pins the vol
    window, default 63). ``decision_every`` is the refresh cadence in trading
    sessions (1 = daily; 21 ≈ monthly for B1/trend): off a refresh session the
    book holds its filled weights and only NAV marks. Caps are down-only (a
    low-gross book stays low-gross). ``no_trade_band`` is the *online*
    half-width applied against held weights (the batch band stays off —
    replaying hysteresis from flat every day is the defect the online step
    fixes). ``max_participation`` of None disables the %ADV gate (at paper
    scale it never binds; enable it before any AUM claim). ``whole_shares``
    matches Alpaca OPG (next-open) order requirements.
    """

    position_size: float = 0.0
    max_gross: float = 1.0
    max_symbol_abs_weight: float = 0.10
    no_trade_band: float = 0.0
    max_participation: float | None = None
    adv_window_bars: int = 20
    min_order_notional: float = 1.0
    whole_shares: bool = True
    book: str = "directional"
    decile: float = 0.10
    vol_ewma_bars: int = 63
    decision_every: int = 1

    def __post_init__(self) -> None:
        if self.book not in ("directional", "decile_neutral", "inverse_vol"):
            raise ValueError(
                f"unknown book {self.book!r}: expected 'directional', "
                f"'decile_neutral', or 'inverse_vol'"
            )
        if self.book == "directional" and not self.position_size > 0:
            raise ValueError("the directional book requires position_size > 0")
        if self.book == "decile_neutral" and not 0.0 < self.decile <= 0.5:
            raise ValueError(f"the decile book requires decile in (0, 0.5], got {self.decile}")
        if self.book == "inverse_vol" and self.vol_ewma_bars < 2:
            raise ValueError(
                f"the inverse_vol book requires vol_ewma_bars >= 2, got {self.vol_ewma_bars}"
            )
        if self.decision_every < 1:
            raise ValueError(f"decision_every must be >= 1, got {self.decision_every}")


@dataclass(frozen=True)
class DailyCycleResult:
    """What one cycle did, for the operator's log and the session record.

    ``monitor_read`` is the anytime-valid confidence-sequence verdict over the
    equity ledger after this cycle's NAV mark (additive telemetry, SPEC §10) —
    ``None`` when no equity ledger is configured.
    """

    decision_bar: str
    settled_fills: list[Fill]
    submitted_orders: list[Order]
    equity: float
    target_weights: pd.Series
    monitor_read: dict | None = None
    concordance: dict | None = None
    halted: str | None = None
    # Symbols the ``unrankable`` provider masked at this refresh (empty = the
    # provider ran and flagged nothing; None = no provider, or not a refresh
    # session so nothing was scored).
    masked: list[str] | None = None
    # The SPEC §7.5 regime telemetry read this cycle (None = no provider
    # wired), and the clamped gross multiplier the §7.7 action hook applied
    # to this refresh's construction (None = hook unarmed, not a refresh
    # session, or the regime read carried failure entries).
    regime: dict | None = None
    regime_scale: float | None = None


@dataclass(frozen=True)
class HeldUniverse:
    """The fetch universe reconciled against **broker truth** (N7).

    The mark step values what the *venue* holds (``run_daily_cycle`` step 2
    reads ``ctx.broker.positions()``), so the fetch universe must cover the
    venue's book or the cycle dies unable to price a position. Splitting the
    two roles is what makes an index leaver safe:

    * ``score`` — the configured universe, the only names that may be *ranked*.
      A departed name must be able to leave the book but never re-enter it.
    * ``valuation_only`` — held names outside ``score``. Fetched so they can be
      marked and exited; masked out of scoring by the caller, where the decile
      construct's explicit-flat pin turns them into exit orders at the next
      refresh.
    * ``fetch`` — ``score`` + ``valuation_only``, in that order.
    * ``held`` — the broker-truth position book (nonzero shares) the split was
      derived from. Every one of these names MUST come back priced, which is
      what ``fetch_universe_panels(..., required=...)`` enforces.
    """

    fetch: tuple[str, ...]
    score: tuple[str, ...]
    valuation_only: tuple[str, ...]
    held: Mapping[str, float]


def resolve_fetch_universe(
    configured: Sequence[str],
    broker: Broker,
    *,
    state: LoopState | None = None,
) -> HeldUniverse:
    """Reconcile the configured universe against the broker's actual book.

    Called BEFORE the fetch, because a held name that was never fetched cannot
    be valued and the cycle then dies at the mark step (N7). The authority for
    "what is held" is the **broker**, never the persisted state:

    * A fresh run directory persists nothing, so a state-derived universe is
      empty over a live account — every cycle from 2026-07-23 through 2026-07-29
      died on ``cannot value held position 'POOL'`` for exactly this reason.
    * A *populated* state is only a cache of the last reconcile, so it can be
      stale too: the retired ``runs/paper_loop_momentum`` records 28 positions
      against the same venue account's 98.

    ``state`` is therefore advisory only. Its positions are still unioned into
    the fetch list — a superset costs one extra symbol per stale name and
    protects the reconcile/divergence path — but they never define ``held`` and
    never make a name *required* to price, because a position the venue has
    already closed must not be able to fail the fetch.

    Raises only on an empty configured universe; a broker read that fails
    raises from the broker (loudly, before any vendor call).
    """
    score = tuple(dict.fromkeys(str(s).strip().upper() for s in configured if str(s).strip()))
    if not score:
        raise ValueError("empty configured universe — nothing to score (N7)")
    held = {str(s): float(q) for s, q in broker.positions().items() if float(q) != 0.0}
    persisted = {str(s) for s, q in (state.positions if state is not None else {}).items() if float(q) != 0.0}

    valuation_only = tuple(sorted((set(held) | persisted) - set(score)))
    if valuation_only:
        logger.warning(
            "%d held name(s) are outside the configured universe %s; fetching them for "
            "valuation/exit only (masked out of scoring, so they exit at the next refresh "
            "and cannot re-enter)",
            len(valuation_only),
            list(valuation_only),
        )
    if held and not persisted:
        # The 2026-07-23 class: a run directory with no memory of the account's
        # book. Recoverable now (broker truth drives the fetch), but the session
        # record and the NAV ledger live in whichever directory decided the book
        # — prism-doctor FAILs on this so the operator reattaches deliberately.
        logger.warning(
            "this run directory has NO persisted book while the venue holds %d position(s): "
            "the cycle proceeds on broker truth, but the equity/regime ledgers of the book "
            "being marked are somewhere else — reattach the run directory (prism-doctor)",
            len(held),
        )
    elif persisted - set(held) or set(held) - persisted:
        logger.warning(
            "broker book (%d position(s)) differs from the persisted book (%d): adopting "
            "broker truth for valuation; the loop's reconcile step will adopt it for state",
            len(held),
            len(persisted),
        )
    return HeldUniverse(fetch=score + valuation_only, score=score, valuation_only=valuation_only, held=held)


def fetch_universe_panels(
    loader: Any,
    symbols: Sequence[str],
    *,
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    store: Any | None = None,
    max_missing: float = 0.0,
    required: Collection[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Wide (close, volume) panels via the incremental store's delta fetch.

    ``loader`` is duck-typed to ``DataLoader.fetch_incremental``; a loader that
    also exposes ``fetch_batch`` (Alpaca) pulls the whole universe in a few
    multi-symbol requests instead of one per name (SPEC §7.0 read path). Returns
    ``volume=None`` when no symbol carries a volume column.

    ``max_missing`` is the fraction of symbols allowed to return no bars before
    the fetch fails loud (N7). The default 0.0 keeps a hand-picked universe
    strict — any missing name is an error. A large point-in-time universe sets it
    above zero to tolerate a few stale/renamed tickers the venue no longer serves
    under that symbol: those are dropped with a loud warning naming them (so the
    universe file can be curated), but a miss rate above the bound still raises,
    because that signals a systemic feed/auth failure, not a few dead tickers.

    ``required`` names are exempt from that tolerance: they are the broker's held
    book (``HeldUniverse.held``), and a held position with no bars cannot be
    marked or exited, so it raises here — naming the position and the remedy —
    instead of being silently dropped into the same unpriceable-position death
    at the mark step the reconciliation exists to prevent (N7).
    """
    if not symbols:
        raise ValueError("empty symbol list — nothing to fetch")
    symbols = list(symbols)
    if hasattr(loader, "fetch_batch"):
        # A batch-capable source (Alpaca) pulls the whole universe in a few
        # paginated multi-symbol requests; one request per name over hundreds of
        # names exhausts the vendor's ~200 req/min budget (a 429).
        frames = loader.fetch_batch(symbols, interval=interval, start_date=start_date, end_date=end_date)
        bars_by_symbol = {s: frames.get(s, pd.DataFrame()) for s in symbols}
    else:
        bars_by_symbol = {
            s: loader.fetch_incremental(
                s, interval=interval, start_date=start_date, end_date=end_date, store=store
            )
            for s in symbols
        }
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    missing: list[str] = []
    for symbol in symbols:
        bars = bars_by_symbol[symbol]
        if bars.empty or "close" not in bars.columns:
            missing.append(symbol)
            continue
        closes[symbol] = pd.to_numeric(bars["close"], errors="coerce")
        if "volume" in bars.columns:
            volumes[symbol] = pd.to_numeric(bars["volume"], errors="coerce")
    if missing:
        shown = missing[:20]
        tail = "…" if len(missing) > len(shown) else ""
        unpriceable = sorted(set(missing) & {str(s) for s in (required or ())})
        if unpriceable:
            raise RuntimeError(
                f"no bars for held position(s) {unpriceable}: the book cannot be marked or "
                "exited without them, so the max_missing tolerance does not apply — restore "
                "the bar source for these names or close the positions at the venue (N7)"
            )
        if len(missing) > max_missing * len(symbols):
            raise RuntimeError(
                f"no bars for {len(missing)}/{len(symbols)} symbols "
                f"({len(missing) / len(symbols):.1%} > max_missing {max_missing:.1%}): "
                f"{shown}{tail}; fix the source, prune the universe, or raise max_missing (N7)"
            )
        logger.warning(
            "no bars for %d/%d symbols (%.1f%% <= max_missing %.1f%%); dropping them and trading "
            "the rest — prune these from the universe if they are stale/renamed: %s%s",
            len(missing),
            len(symbols),
            100 * len(missing) / len(symbols),
            100 * max_missing,
            shown,
            tail,
        )
    if not closes:
        raise RuntimeError(f"no bars for any of the {len(symbols)} requested symbols (N7)")
    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex(columns=close.columns).sort_index() if volumes else None
    return close, volume


def warn_if_stale_panel(close: pd.DataFrame, *, max_calendar_days: int = 4) -> int:
    """Warn loudly (N7) when the panel's freshest bar is anomalously old.

    The instrument decides on the freshest bar the vendor has published. If
    that bar is many calendar days old the feed is lagging (or the loop has
    been dark), the decision is being made on stale prices, and any resulting
    fill carries extra arrival lag versus its close-t reference — flag it
    rather than trade on it silently. The default 4 clears a normal Fri->Tue
    holiday weekend; more than that is anomalous. Returns the age in days.
    """
    last_bar = close.index[-1]
    stale_days = int((pd.Timestamp.now(tz=last_bar.tz) - last_bar).days)
    if stale_days > max_calendar_days:
        logger.warning(
            "latest available bar %s is %d calendar days old — the vendor feed is lagging "
            "or the loop has been dark; deciding on a stale panel, and fills will carry "
            "extra arrival lag beyond the close-t reference (verify before trusting them)",
            last_bar.date(),
            stale_days,
        )
    return stale_days


def run_daily_cycle(
    ctx: LiveLoopContext,
    signal: Signal,
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    config: DailyBookConfig,
    decision_bar: str | None = None,
    safety: SafetyConfig | None = None,
    unrankable: Callable[[str], Collection[str]] | None = None,
    regime: Callable[[str], dict] | None = None,
    regime_gross_scale: Callable[[dict], float] | None = None,
) -> DailyCycleResult:
    """Run one settle → mark → score → construct → decide/submit cycle.

    ``close``/``volume`` are the full trailing panels whose last row is the
    decision bar *t* (N1); ``decision_bar`` defaults to that row's date.
    Re-running within the same bar resumes the persisted decision without
    settling or re-deciding (the write-ahead protocol's restart semantics).

    ``safety`` (``prism.live.safety``, ``None`` = no rails) adds two vetoes:
    a *halt* (kill-switch file, drawdown bound) checked after the mark step —
    the cycle still settles and marks NAV so the ledger stays continuous, but
    skips score → construct → submit and reports ``halted`` — and an *order
    guard* (per-order notional, order count) enforced inside the write-ahead
    protocol before anything is persisted or submitted.

    ``unrankable`` (``None`` = off, bit-identical to the unmasked loop) is
    consulted only on refresh sessions with the decision bar and returns
    symbols that may not be ranked this refresh — e.g. names with a spin-off
    inside the momentum lookback, where the live vendor's split-only bars
    diverge from the spine's back-adjusted basis
    (docs/bar_vendor_divergence.md §5, ``prism.live.spinoff_mask``). A masked
    name is scored NaN (it cannot distort decile membership) and its target
    carries NO decision (NaN): no new position opens on a divergent rank, a
    held position is held until the window clears the event. The provider owns
    its failure policy (fetch failure = loud warning, empty mask).

    ``regime`` (``None`` = off, bit-identical to the unwired loop) is the
    SPEC §7.5 telemetry provider (``prism.live.regime_step``), consulted with
    the decision bar on EVERY cycle — non-refresh and halted sessions
    included, because the handoff §8 precondition-(b) clock counts sessions
    and a halted cycle must still record regime state. The provider owns its
    failure policy (named per-block failure entries, never a raise, never a
    silently-empty dict); a provider that violates that contract anyway is
    caught here and recorded as a named ``provider`` failure entry with a
    loud warning, so telemetry can never take down the certified book's
    cycle. The read lands in ``DailyCycleResult.regime`` and the regime
    ledger; ANY failure entry marks the session not clean for the
    precondition-(b) 21-session clock (docs/regime_step.md).

    ``regime_gross_scale`` is the §7.5 de-gross ACTION hook and stays
    unarmed (``None``) until docs/sizing_preregistration.md is
    ratified — it has deliberately no CLI path. When armed it fires only on
    a refresh session whose regime read carries ZERO failure entries: its
    return value multiplies ``config.max_gross`` for this refresh's
    construction, clamped to ``[0, config.max_gross]``, and the applied
    multiplier is recorded (``regime_scale``; regime ledger). On a dirty
    read the refresh constructs at the configured gross with a loud
    warning — a telemetry failure de-arms the action, it never silently
    de-grosses. Requires ``regime``; a non-finite scale raises (N7).
    """
    if regime_gross_scale is not None and regime is None:
        raise ValueError(
            "regime_gross_scale requires the regime telemetry provider: the hook fires only on "
            "a clean regime read, so without one it could never legally apply (a dead switch, N7)"
        )
    if close.empty:
        raise ValueError("empty close panel — nothing to decide on")
    if len(close) < signal.required_history:
        raise ValueError(
            f"panel has {len(close)} rows < signal required_history "
            f"{signal.required_history}; refusing to score a truncated panel (N7)"
        )
    decision_bar = decision_bar or str(close.index[-1].date())

    # 1. Settle the previous bar's decision.
    state = ctx.store.load()
    settled = _settle_prior(ctx, state, decision_bar)

    # 2. Mark broker truth at today's closes.
    prices = close.iloc[-1]
    equity, cash, prev_weights = _mark_book(ctx.broker, prices)

    # 2b. Book-concordance telemetry (pure; gates nothing).
    concordance = _concordance_read(ctx, prev_weights, decision_bar)

    # 2c. Halt rails (kill switch, drawdown) — decided on the marked equity
    # before any decision is constructed or persisted.
    halted = _halt_check(safety, equity, ctx)

    # 2d. Regime telemetry (SPEC §7.5) — read EVERY cycle, halted or not.
    regime_state = _regime_read(regime, decision_bar)

    # 3. Cadence gate, then score → construct against held weights. Off a refresh
    # session (the decision cadence has not elapsed) the book holds its filled
    # weights — no re-score, no trades — so a monthly book rebalances monthly
    # while NAV still marks daily.
    refresh = halted is None and _is_refresh_session(
        state, decision_bar, close.index, config.decision_every
    )
    effective_max_gross, regime_scale = _effective_gross(
        config, refresh, regime_gross_scale, regime_state, decision_bar
    )

    masked: list[str] | None = None
    if refresh:
        targets, masked = _construct_refresh_targets(
            signal,
            close,
            volume,
            config,
            prev_weights,
            equity,
            effective_max_gross,
            unrankable,
            decision_bar,
        )
    else:
        targets = prev_weights  # hold the filled book: no decision this session

    # 4. Write-ahead decide and idempotent submit. Skipped entirely under a
    # halt — a persisted-but-unsubmitted decision would resume trading on the
    # next non-halted run, which is not what a kill switch means.
    orders: list[Order] = []
    if halted is None:
        orders = decide_and_submit(
            ctx,
            decision_bar,
            targets,
            prices,
            min_order_notional=config.min_order_notional,
            whole_shares=config.whole_shares,
            refresh_bar=decision_bar if refresh else None,
            order_guard=(
                None if safety is None else (lambda order_list: check_orders(order_list, equity, safety))
            ),
        )
        logger.info(
            "decision %s: %d orders submitted, equity %.2f, gross %.4f",
            decision_bar,
            len(orders),
            equity,
            float(targets.abs().sum()),
        )
    # Record the regime read durably beside the NAV mark — on halted and
    # non-refresh cycles too, because the precondition-(b) clock reads this
    # ledger. Idempotent per decision bar (the equity-ledger discipline).
    if ctx.regime_ledger is not None and regime_state is not None:
        append_regime_ledger(ctx.regime_ledger, decision_bar, regime_state, regime_scale)

    # Record the post-settle mark-to-market NAV for this bar (the anytime-valid
    # monitor's return-series source). Idempotent per bar, so same-bar restarts
    # never double-count. Written last so a crash mid-cycle logs no equity for
    # an incomplete bar.
    monitor_read: dict | None = None
    if ctx.equity_ledger is not None:
        append_equity_ledger(ctx.equity_ledger, decision_bar, equity, cash)
        # Arm the anytime-valid monitor as additive telemetry beside the ratified
        # rolling-PSR promotion read (docs/momentum_design.md): time-uniform
        # coverage makes a per-cycle read safe to log, it moves no ratified
        # statistic and starts no counted trial. Becoming a binding promotion/kill
        # read is deferred to a future program's pre-registration.
        monitor_read = paper_monitor_read(ctx.equity_ledger)
        logger.info(
            "monitor %s: verdict=%s n=%s mean=%s ci=[%s, %s]",
            decision_bar,
            monitor_read.get("verdict"),
            monitor_read.get("n"),
            monitor_read.get("mean"),
            monitor_read.get("ci_lower"),
            monitor_read.get("ci_upper"),
        )
    return DailyCycleResult(
        decision_bar=decision_bar,
        settled_fills=settled,
        submitted_orders=orders,
        equity=equity,
        target_weights=targets,
        monitor_read=monitor_read,
        concordance=concordance,
        halted=halted,
        masked=masked,
        regime=regime_state,
        regime_scale=regime_scale,
    )


def _settle_prior(ctx: LiveLoopContext, state: LoopState | None, decision_bar: str) -> list[Fill]:
    """Step 1: settle a pending decision from an *earlier* bar.

    The settle marker is the bar the decision was made on ("that decision bar
    is fully processed"), keeping today's decision_bar strictly greater for
    the re-decide guard. A pending decision for *today's* bar is a
    same-session restart: skip settle, the write-ahead protocol resumes
    submission idempotently.
    """
    if state is None or not state.pending_orders:
        return []
    if state.pending_decision_bar == decision_bar:
        # WARNING, not INFO: a decision bar that has not advanced since the
        # last cycle means the loop is stuck resuming the same bar (a dark
        # or lagging data feed is the usual cause) and is not accruing
        # fills/equity — that should be visible without reading every line.
        logger.warning(
            "pending decision for %s already persisted — resuming submission, not "
            "settling (decision bar has not advanced since last cycle)",
            decision_bar,
        )
        return []
    settled = settle(ctx, state.pending_decision_bar)
    logger.info("settled %d fills from decision bar %s", len(settled), state.pending_decision_bar)
    return settled


def _mark_book(broker: Broker, prices: pd.Series) -> tuple[float, float, pd.Series]:
    """Step 2: broker-truth positions and cash marked at today's closes.

    Returns ``(equity, cash, prev_weights)`` — the same marks
    ``decide_and_submit`` sizes with. Every held position must price
    (``require_price``, N7) and equity must be positive.
    """
    positions = {s: q for s, q in broker.positions().items() if q != 0.0}
    cash = broker.cash()
    equity = cash + sum(shares * require_price(prices, s) for s, shares in positions.items())
    if equity <= 0:
        raise RuntimeError(f"non-positive equity {equity}; the loop cannot size a book (N7)")
    prev_weights = pd.Series(
        {s: shares * require_price(prices, s) / equity for s, shares in positions.items()},
        dtype=float,
    )
    return equity, cash, prev_weights


def _concordance_read(ctx: LiveLoopContext, prev_weights: pd.Series, decision_bar: str) -> dict | None:
    """Step 2b: how faithfully does the held book track the last refresh's book?

    Computed against the latest persisted refresh targets STRICTLY before
    today (the book the instrument should currently embody), so a refresh day
    reads its predecessor, never its own about-to-be-decided targets. Pure
    telemetry: a 0.19-gross partial-fill book and a faithful book are
    indistinguishable in the equity monitor; they are not indistinguishable
    here.
    """
    if ctx.targets_ledger is None or ctx.concordance_ledger is None:
        return None
    prior = [
        row for row in read_targets_ledger(ctx.targets_ledger) if row["refresh_bar"] < decision_bar
    ]
    if not prior:
        return None
    baseline = prior[-1]
    concordance = book_concordance(prev_weights, pd.Series(baseline["targets"], dtype=float))
    concordance["refresh_bar"] = baseline["refresh_bar"]
    append_concordance_ledger(ctx.concordance_ledger, decision_bar, concordance)
    logger.info(
        "concordance %s vs refresh %s: active_share=%.4f gross %.4f/%.4f corr=%s",
        decision_bar,
        baseline["refresh_bar"],
        concordance["active_share"],
        concordance["gross_held"],
        concordance["gross_target"],
        None if concordance["weight_corr"] is None else round(concordance["weight_corr"], 4),
    )
    return concordance


def _halt_check(safety: SafetyConfig | None, equity: float, ctx: LiveLoopContext) -> str | None:
    """Step 2c: the halt rails' verdict on the marked equity.

    A halted cycle still settles and marks NAV (the ledger stays continuous
    through a halt) but initiates nothing, and it does NOT resume a pending
    submission either: the kill switch must gate every path to the venue.
    """
    if safety is None:
        return None
    halted = halt_reason(safety, equity, ctx.equity_ledger)
    if halted is not None:
        logger.error(
            "SAFETY HALT (%s): settling and marking NAV only — no decision, no orders "
            "this cycle",
            halted,
        )
    return halted


def _regime_read(regime: Callable[[str], dict] | None, decision_bar: str) -> dict | None:
    """Step 2d: the §7.5 telemetry read, shielded so it can never kill a cycle.

    The provider owns its failure policy; a raise or a silently-empty result
    here is a contract violation, converted to a named ``provider`` failure
    entry so telemetry can never take down the certified book's cycle (N7:
    loud and named — not fatal, not silent).
    """
    if regime is None:
        return None
    try:
        regime_state = regime(decision_bar)
        if not isinstance(regime_state, dict) or not (
            regime_state.get("blocks") or regime_state.get("failures")
        ):
            raise RuntimeError("regime provider returned a silently-empty result")
        return regime_state
    except Exception as exc:  # noqa: BLE001 — telemetry, never a cycle precondition
        logger.warning(
            "REGIME PROVIDER FAILED for %s (%s) — recording a named provider failure; this "
            "session is NOT clean for the handoff §8 precondition-(b) clock (N7)",
            decision_bar,
            type(exc).__name__,
        )
        return {
            "decision_bar": decision_bar,
            "blocks": {},
            "failures": [{"block": "provider", "error": type(exc).__name__}],
        }


def _effective_gross(
    config: DailyBookConfig,
    refresh: bool,
    regime_gross_scale: Callable[[dict], float] | None,
    regime_state: dict | None,
    decision_bar: str,
) -> tuple[float, float | None]:
    """The §7.5 de-gross ACTION hook: only on a refresh, only on a regime read
    with zero failure entries, clamped to [0, config.max_gross]. A dirty read
    de-arms the action loudly — construction proceeds at the configured
    gross, never a silent de-gross on broken telemetry. Returns
    ``(effective_max_gross, regime_scale)``.
    """
    if not refresh or regime_gross_scale is None:
        return config.max_gross, None
    if regime_state is None or regime_state.get("failures"):
        logger.warning(
            "regime gross-scale hook NOT applied at refresh %s: the regime read is not clean; "
            "constructing at the configured max_gross %.4f (a telemetry failure de-arms the "
            "action, it never silently de-grosses)",
            decision_bar,
            config.max_gross,
        )
        return config.max_gross, None
    scale = float(regime_gross_scale(regime_state))
    if not math.isfinite(scale):
        raise ValueError(f"regime_gross_scale returned {scale}; refusing to size a book on it (N7)")
    regime_scale = min(max(scale, 0.0), 1.0)
    effective_max_gross = regime_scale * config.max_gross
    logger.info(
        "regime gross scale %.4f applied at refresh %s: effective max_gross %.4f (configured %.4f)",
        regime_scale,
        decision_bar,
        effective_max_gross,
        config.max_gross,
    )
    return effective_max_gross, regime_scale


def _construct_refresh_targets(
    signal: Signal,
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    config: DailyBookConfig,
    prev_weights: pd.Series,
    equity: float,
    effective_max_gross: float,
    unrankable: Callable[[str], Collection[str]] | None,
    decision_bar: str,
) -> tuple[pd.Series, list[str] | None]:
    """Step 3 on a refresh session: score → construct → band → participation.

    Returns ``(targets, masked)``. Sizing folds in exactly once (I-4): the
    book construct owns weights, the online band owns hysteresis against
    *held* weights, the participation gate shrinks trades over the %ADV cap.
    """
    score_row = pd.DataFrame([signal.score(close, volume)], index=close.index[-1:])
    masked: list[str] | None = None
    if unrankable is not None:
        # Unrankable names (spin-off inside the lookback — see the
        # run_daily_cycle docstring) leave the scored cross-section entirely,
        # so decile membership is decided over rankable names only.
        masked = sorted(set(unrankable(decision_bar)) & set(score_row.columns))
        if masked:
            score_row.loc[:, masked] = float("nan")
            logger.info(
                "unrankable mask: %d name(s) held out of ranking this refresh: %s",
                len(masked),
                masked,
            )
    raw = _construct_raw_book(score_row, close, config, effective_max_gross)
    if masked:
        # Hold, don't flatten: the decile construct pins a NaN score to an
        # explicit 0.0 (exit), but an unrankable name carries NO decision.
        # A NaN target resolves to the held weight in step_no_trade_band
        # (0.0 for an unheld name), so no new position opens on a divergent
        # rank and a held name is held until the event clears the lookback.
        raw.loc[masked] = float("nan")
    targets = step_no_trade_band(prev_weights, raw, config.no_trade_band)
    if config.max_participation is not None:
        if volume is None:
            raise ValueError("max_participation is set but there is no volume panel to compute ADV (N7)")
        dollar_volume = (close * volume).rolling(config.adv_window_bars, min_periods=1).mean().iloc[-1]
        targets = participation_capped_targets(
            prev_weights,
            targets,
            dollar_volume,
            aum=equity,
            max_participation=config.max_participation,
        )
    return targets, masked


def _construct_raw_book(
    score_row: pd.DataFrame,
    close: pd.DataFrame,
    config: DailyBookConfig,
    effective_max_gross: float,
) -> pd.Series:
    """One scored row → the book's raw target weights (pre-band, pre-gate)."""
    if effective_max_gross <= 0.0:
        # A zero effective cap — regime_scale clamped to 0.0, the full
        # de-gross — admits no gross, and cap_book refuses a 0 cap, so
        # emit each book's own flat form directly: the decile /
        # inverse_vol books' explicit 0.0 for every name (NaN-score pin
        # included), the directional book's flat-where-decided (a NaN
        # score keeps carrying no decision, exactly as the construct
        # would emit). The band step then exits every held, decidable name.
        if config.book in ("decile_neutral", "inverse_vol"):
            return pd.Series(0.0, index=score_row.columns, dtype=float)
        return score_row.iloc[0] * 0.0
    if config.book == "decile_neutral":
        # The decile construct emits an explicit weight (0.0 outside the
        # decile) for every scored name, so a held name that falls out of the
        # decile — including a book inherited at cutover, as long as it is in
        # the fetched universe — is rebalanced to flat by the online band. A
        # held name absent from the universe cannot be priced and is refused
        # loudly at the mark step (N7): trade only what you can value, so a
        # cutover universe must contain the names it inherits.
        return construct_decile_neutral(
            score_row,
            decile=config.decile,
            max_gross=effective_max_gross,
            max_symbol_abs_weight=config.max_symbol_abs_weight,
        ).iloc[0]
    if config.book == "inverse_vol":
        # Trend sleeve (docs/trend_design.md §2). Vol needs trailing
        # close history; align the one-row score onto the full panel so
        # only the decision bar carries a sign and earlier bars stay
        # NaN (unused by the last-row construct read).
        scores_aligned = pd.DataFrame(
            np.nan, index=close.index, columns=score_row.columns, dtype=float
        )
        scores_aligned.iloc[-1] = score_row.iloc[0].reindex(score_row.columns)
        return construct_inverse_vol_targets(
            scores_aligned,
            close.reindex(columns=score_row.columns),
            vol_ewma_bars=config.vol_ewma_bars,
            max_gross=effective_max_gross,
            max_symbol_abs_weight=config.max_symbol_abs_weight,
        ).iloc[-1]
    return construct_directional_targets(
        score_row,
        position_size=config.position_size,
        max_gross=effective_max_gross,
        max_symbol_abs_weight=config.max_symbol_abs_weight,
        no_trade_band=0.0,  # the online band owns hysteresis (never the flat replay)
    ).iloc[0]


def _is_refresh_session(
    state: LoopState | None,
    decision_bar: str,
    index: pd.DatetimeIndex,
    decision_every: int,
) -> bool:
    """Whether this session refreshes the book (SPEC §7.7 decision cadence).

    ``decision_every`` trading sessions must elapse between refreshes (1 = every
    session, the frozen daily default). The first cycle, or a state with no
    cadence anchor, always refreshes. The elapsed count is read off the panel's
    own index — the trading calendar the loop already holds — so it matches the
    backtest's bar-count cadence, not wall-clock months.
    """
    if decision_every <= 1:
        return True
    last_refresh = state.last_refresh_bar if state is not None else None
    if last_refresh is None:
        return True
    dates = [str(d.date()) for d in index]
    elapsed = sum(1 for dt in dates if last_refresh < dt <= decision_bar)
    return elapsed >= decision_every
