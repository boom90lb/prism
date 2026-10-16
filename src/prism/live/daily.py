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
   cadence has elapsed — daily by default, ≈monthly for B1; under a seeded
   phase, the book's session-grid date has come) the signal
   scores the panel's last row and construction maps scores to capped
   weights (linear for the directional book, top/bottom-decile long/short
   for the neutral book; I-4, one sizing point). The online band holds names
   whose target moved less than the band from what is held, and the
   participation gate shrinks trades over the %ADV cap (§7.4). Off a refresh
   session the book holds its filled weights while NAV still marks daily.
   With the volatility scale on (``vol_scale``, default off;
   :mod:`prism.live.vol_sizing`) the refresh first decides the decile book's
   scale from its base book rebuilt on the panel, records it in
   ``<run_dir>/sizing.jsonl``, and constructs the scaled book. A tranched
   book (``tranches > 1``, default off; :mod:`prism.live.tranches`) decides on
   every session one of its tranches is due, re-scoring only the due tranches
   and composing the account target from every tranche's sub-targets. With a
   T-bill holding (``cash_holding``, default off; :mod:`prism.live.cash_holding`)
   the idle equity above a cash buffer is then set in an ETF on every session
   that is not halted, within a band and a per-order bound.
5. **Decide/submit.** ``decide_and_submit`` persists the decision before
   the first submit and submits idempotently (N2 crash window).
6. **Monitor.** After the mark-to-market NAV is appended to the equity
   ledger, the cycle records the shadow return (the decided book marked on
   the same closes, ``<run_dir>/shadow.jsonl``) and reads the live monitor
   (:mod:`prism.live.monitor`, registered in docs/live_monitor_design.md): the
   return monitor's confidence sequence over consecutive-session NAV returns
   and the tracking read of paper minus shadow; with ``tbill_monitor`` (default
   off) also a report-only read of the same returns against the T-bill rate.
   Shielded like the regime provider: a failure is logged loudly and recorded,
   never raised into the cycle. It gates no order.

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

import datetime as dt
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from prism.assembly import (
    AccountState,
    DecisionContext,
    assemble,
    heuristic_book_rule,
    raw_book_weights,
)
from prism.composition import ComponentPlan, due_on_schedule
from prism.io import sessions
from prism.live.broker import Broker, Fill, Order, transmits_to_venue
from prism.live.cash_holding import TARGETS_LEDGER_KEY as CASH_HOLDING_KEY
from prism.live.cash_holding import (
    CashHolding,
    apply_cash_holding,
    log_decision,
    require_cash_holding_inputs,
)
from prism.live.loop import (
    LiveLoopContext,
    append_concordance_ledger,
    append_equity_ledger,
    append_regime_ledger,
    append_shadow_ledger,
    decide_and_submit,
    read_equity_ledger,
    read_targets_ledger,
    require_price,
    settle,
)
from prism.live.monitor import (
    TbillReference,
    bar_date,
    book_concordance,
    paper_monitor_read,
    shadow_session_return,
    tbill_monitor_read,
)
from prism.live.panel_log import (
    PanelLogConflict,
    PanelRecord,
    latest_record,
    record_panel,
)
from prism.live.safety import SafetyConfig, SafetyViolation, check_orders, halt_reason
from prism.live.state import LoopState
from prism.live.tranches import (
    TrancheDecision,
    TrancheSession,
    decide_tranches,
    open_tranche_session,
    refuse_tranche_state_under_single_book,
    require_tranche_inputs,
)
from prism.live.vol_sizing import (
    SIZING_LEDGER_FILENAME,
    TARGETS_LEDGER_KEY,
    LiveScale,
    append_sizing_decision,
    decide_live_scale,
    last_good_scale,
    require_vol_scale_inputs,
    scaled_book_rule,
    sizing_row,
    targets_ledger_record,
)
from prism.portfolio.vol_scale import VolScaleRule
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
    ``None`` when no equity ledger is configured. With ``tbill_monitor`` it also
    carries the report-only T-bill-referenced read under ``tbill``.
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
    # The index row of the bar panel this cycle decided on
    # (``prism.live.panel_log``): None when no run directory was available to
    # record it in.
    panel: dict | None = None
    # The volatility-scale decision this refresh sized the book with, as its
    # row in ``{run-dir}/sizing.jsonl`` (``prism.live.vol_sizing``): None when
    # the scale is off or this session did not refresh.
    vol_scale: dict | None = None
    # The tranched book's session record (``prism.live.tranches``): the tranche
    # count, base grid offset, wind-down flag, the tranches due this session,
    # how many have been built and any withheld entries. None for the single book.
    tranches: dict | None = None
    # The T-bill holding's decision this session (``prism.live.cash_holding``):
    # None when the holding is off or the session halted.
    cash_holding: dict | None = None


@dataclass(frozen=True)
class HeldUniverse:
    """The fetch universe reconciled against **broker truth** (N7).

    The mark step values what the *venue* holds (``run_daily_cycle`` step 2
    reads ``ctx.broker.positions()``), so the fetch universe must cover the
    venue's book or the cycle dies unable to price a position. Splitting the
    two roles is what makes an index leaver safe:

    * ``score`` — the configured universe, the only names that may be *ranked*.
      A departed name must be able to leave the book but never re-enter it.
    * ``valuation_only`` — held names outside ``score``, and the ``unscored``
      names the caller asked to price (the T-bill holding). Fetched so they can
      be marked, exited or sized; masked out of scoring by the caller, where the
      decile construct's explicit-flat pin turns a held name outside ``score``
      into an exit order at the next refresh unless a later step (the cash
      holding) sets its weight.
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
    unscored: Sequence[str] = (),
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

    ``unscored`` (default empty, unchanged behaviour) names symbols to fetch and
    price that are never ranked, whether held or not: the T-bill holding of
    :mod:`prism.live.cash_holding`. They join ``valuation_only`` and are not
    reported as outside names; a symbol that is also in the configured universe
    is refused, since it cannot be both ranked and not.

    Raises on an empty configured universe and on that overlap; a broker read
    that fails raises from the broker (loudly, before any vendor call).
    """
    score = tuple(dict.fromkeys(str(s).strip().upper() for s in configured if str(s).strip()))
    if not score:
        raise ValueError("empty configured universe — nothing to score (N7)")
    priced = {str(s).strip().upper() for s in unscored if str(s).strip()}
    overlap = sorted(priced & set(score))
    if overlap:
        raise ValueError(
            f"{overlap} are both in the configured universe and named unscored; a name is either ranked or "
            "priced only, never both (N7)"
        )
    held = {str(s): float(q) for s, q in broker.positions().items() if float(q) != 0.0}
    persisted = {str(s) for s, q in (state.positions if state is not None else {}).items() if float(q) != 0.0}

    outside = tuple(sorted((set(held) | persisted) - set(score) - priced))
    valuation_only = tuple(sorted(set(outside) | priced))
    if outside:
        logger.warning(
            "%d held name(s) are outside the configured universe %s; fetching them for "
            "valuation/exit only (masked out of scoring, so they exit at the next refresh "
            "and cannot re-enter)",
            len(outside),
            list(outside),
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


#: The exchange's wall clock. Session dates and close times are stated in it.
EXCHANGE_TZ = "America/New_York"


class PartialBarError(SafetyViolation):
    """The panel's last bar is not a completed, published session bar.

    A :class:`~prism.live.safety.SafetyViolation` because the remedy is the
    same as for every other pre-trade rail: nothing was decided, nothing was
    sent, and the operator re-runs after the close or passes ``--no-submit``.
    """


def require_published_panel(
    close: pd.DataFrame,
    *,
    now: dt.datetime | pd.Timestamp | None = None,
    allow_partial: bool = False,
    publication_delay_minutes: int = 30,
) -> dict[str, Any]:
    """Refuse to decide on a daily bar that is still being traded.

    The bar fetch asks for no end date, so a run during market hours receives
    today's row with the price as of the last print, not the close. A decision
    taken on it is not the close-to-next-open decision the whole evaluation
    assumes (every backtest number was produced at the close), and the
    2026-09-25 catch-up rehearsal did exactly that. The stale-panel warning
    beside this cannot catch it: a bar from this morning is zero days old.

    The rule: the last bar's date must be an exchange session, and when ``now``
    falls on that date it must be at least ``publication_delay_minutes`` past
    that session's close (``prism.io.sessions.session_close`` knows the 13:00
    early closes). A last bar from an earlier date is complete at any hour.
    A bar dated after ``now`` is refused outright: no clock can make it
    complete. Under ``allow_partial`` (the rehearsal path) an in-progress bar
    is logged as a WARNING instead of raised, so a daytime ``--no-submit`` still
    exercises the command but says plainly what it decided on.

    ``now`` must be timezone-aware; it defaults to the current time in
    ``EXCHANGE_TZ``. Returns ``{last_bar, published_at, now, partial}`` for the
    session log.
    """
    zone = ZoneInfo(EXCHANGE_TZ)
    last = pd.Timestamp(close.index[-1])
    last_date = (last.tz_convert(zone) if last.tzinfo is not None else last).date()
    if now is None:
        now_ts = pd.Timestamp.now(tz=zone)
    else:
        now_ts = pd.Timestamp(now)
        if now_ts.tzinfo is None:
            raise ValueError("require_published_panel: `now` must be timezone-aware; a naive clock has no session")
        now_ts = now_ts.tz_convert(zone)
    if not sessions.is_session(last_date):
        raise PartialBarError(
            f"the panel's last bar is dated {last_date.isoformat()}, which is not an exchange session "
            f"({sessions.status(last_date)}); a decision needs a completed session bar (N1)"
        )
    close_at = dt.datetime.combine(last_date, sessions.session_close(last_date), tzinfo=zone)
    published_at = pd.Timestamp(close_at) + pd.Timedelta(minutes=publication_delay_minutes)
    partial = now_ts < published_at
    report = {
        "last_bar": last_date.isoformat(),
        "published_at": published_at.isoformat(),
        "now": now_ts.isoformat(),
        "partial": bool(partial),
    }
    if now_ts.date() < last_date:
        raise PartialBarError(
            f"the panel's last bar is dated {last_date.isoformat()}, after the clock ({now_ts.isoformat()}); "
            "a bar from the future is a feed or clock defect, not a decision input (N7)"
        )
    if partial:
        if not allow_partial:
            raise PartialBarError(
                f"the panel's last bar {last_date.isoformat()} is still in progress at {now_ts.isoformat()}: "
                f"that session closes at {close_at.time().isoformat(timespec='minutes')} ET and its bar counts "
                f"as published from {published_at.isoformat()}. Deciding now would trade a close-to-open "
                "decision on an intraday price. Re-run after that time, or pass --no-submit to rehearse."
            )
        logger.warning(
            "PARTIAL BAR: this rehearsal decided on %s before its session closed (now %s, published from %s); "
            "the orders logged below are not the ones the close would produce",
            last_date.isoformat(),
            now_ts.isoformat(),
            published_at.isoformat(),
        )
    return report


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
    run_dir: Path | None = None,
    phase_offset: int | None = None,
    vol_scale: VolScaleRule | None = None,
    tranches: int = 1,
    wind_down: bool = False,
    cash_holding: CashHolding | None = None,
    buying_power_check: bool = False,
    tbill_monitor: TbillReference | None = None,
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
    de-grosses. Requires ``regime``; a non-finite scale raises (N7). It is
    refused with ``config.max_gross > 1``: the hook's multiplier is clamped to
    ``[0, 1]`` of the configured gross, so above 1 a clean read could size a
    book above the unscaled one, and it is refused beside ``vol_scale``, which
    sizes the same gross. Both refusals raise before any state moves.

    ``run_dir`` is where the panel this cycle decides on is recorded
    (``prism.live.panel_log``, under ``<run_dir>/panels/``) before anything is
    settled, marked or decided, so every decision in the run directory can be
    re-derived from the exact bars it saw. It defaults to the directory that
    holds ``ctx.targets_ledger``, which is the run directory for every shipped
    driver (and the scratch mirror under a rehearsal); with neither, nothing is
    recorded and ``DailyCycleResult.panel`` is None. A failure to record raises
    before any state changes: the cycle decides only on a recorded panel.

    ``phase_offset`` (``None`` = off, bit-identical: the first refresh sets the
    phase and ``config.decision_every`` sessions separate refreshes) puts the
    refresh cadence on the fixed session grid of ``prism.io.sessions``: the book
    refreshes where ``(grid_position - phase_offset) % decision_every == 0``.
    A book with no refresh yet waits for its first grid date instead of
    refreshing on its first session, and a missed grid date (an outage) is
    caught up once, on the first session after more than ``decision_every`` have
    elapsed, after which the book is back on its grid. The schedule is
    :func:`prism.composition.due_on_schedule`, the predicate composition and the
    evaluator use. The paper loop's ``--rebalance-phase seeded`` draws the
    offset from the run directory's deployment seed
    (:mod:`prism.live.deployment`), so deployments that start on the same day
    do not refresh on the same day. It must be an integer in
    ``[0, decision_every)`` with ``decision_every > 1`` (a book that refreshes
    every session has no phase), and the decision bar must be an exchange
    session in the bundled table; anything else raises before any state moves.

    ``vol_scale`` (``None`` = off, bit-identical) sizes the decile book by the
    Barroso and Santa-Clara constant-volatility rule
    (:mod:`prism.live.vol_sizing`). On each refresh the scale is decided from
    the unscaled base book rebuilt point in time on this panel, the account
    target is that scale times the unscaled book capped at
    ``min(config.max_gross, vol_scale.max_gross)``, and the decision is recorded
    in ``{run-dir}/sizing.jsonl`` before the write-ahead step and in the refresh
    row of the targets ledger. A window it cannot trust falls back to
    ``min(1, last good scale)``, so a failure never sizes the book above the
    unscaled one. It is independent of the ``regime`` telemetry, which still
    reads every session. It refuses, before any state moves, a book other than
    ``decile_neutral``, a regime gross-scale hook, ``config.max_gross < 1``, a
    signal that does not declare ``fitted_state = "none"``, a context without a
    targets ledger, and a panel shorter than
    :func:`prism.live.vol_sizing.required_panel_bars`.

    ``tranches`` (``1`` = off, bit-identical) splits the decile book into that
    many overlapping tranches on the session grid (:mod:`prism.live.tranches`),
    each holding ``1/tranches`` of capital and refreshing every
    ``config.decision_every`` sessions on its own grid date, offset from
    ``phase_offset`` (0 when it is None). The cycle decides on every session a
    tranche is due, re-scores only the due tranches, and sizes orders against the
    sum of every tranche's sub-targets. Tranches start empty and are built one
    per grid date (staged entry); their states persist in
    ``LoopState.components`` in the write-ahead save, and each refresh session
    appends one targets-ledger row. With ``vol_scale`` each due tranche's scale
    is decided from the tranched base book rebuilt point in time from the panel.
    The concordance read runs only for the single book. ``wind_down`` (tranched
    books only) sets each tranche flat on its own due date, so the book exits over
    one cycle. Refused before any state moves: a book other than
    ``decile_neutral``, a count that does not divide ``decision_every``, the
    regime gross-scale hook, a run directory whose tranche layout differs or that
    holds an untranched book, and a first tranched decision on an account that
    holds positions. With ``tranches=1`` the cycle refuses a run directory that
    holds tranche states.

    ``cash_holding`` (``None`` = off, bit-identical) holds the idle own equity in
    a T-bill ETF (:mod:`prism.live.cash_holding`): on every session that is not
    halted, after the book's target is decided, the ETF's weight is set to
    ``max(0, 1 - buffer - L)``, ``L`` the stock long side of that target, when the
    held weight is more than ``band`` away from it, in steps no larger than the
    per-order bound of ``safety`` less one share. Under ``wind_down`` its target
    is zero. A refresh row of the targets ledger carries the decision under
    ``cash_holding``. The caller must fetch the ETF as a priced, unscored name
    (``resolve_fetch_universe(..., unscored=...)``) and keep it out of the
    signal's cross-section; the cycle refuses, before any state moves, a book
    other than ``decile_neutral`` and a panel without a usable close for it.

    ``buying_power_check`` (``False`` = off, bit-identical) has
    :func:`prism.live.loop.decide_and_submit` fit the opening orders inside the
    venue's buying power (:mod:`prism.live.buying_power`), scaling them down
    together, loudly, when they do not fit.

    ``tbill_monitor`` (``None`` = off, bit-identical) adds a report-only read to
    the live monitor: the same return sequence against the per-session T-bill
    hurdle instead of zero, with its inputs and a detection horizon
    (:func:`prism.live.monitor.tbill_monitor_read`). It lands under
    ``monitor_read["tbill"]``, beside the registered hurdle-0 read, which is
    computed exactly as without it. It gates nothing, and a failure of it is
    logged and recorded as ``unavailable`` without touching the cycle.
    """
    if regime_gross_scale is not None and regime is None:
        raise ValueError(
            "regime_gross_scale requires the regime telemetry provider: the hook fires only on "
            "a clean regime read, so without one it could never legally apply (a dead switch, N7)"
        )
    _require_bounded_regime_hook(config, regime_gross_scale)
    if close.empty:
        raise ValueError("empty close panel — nothing to decide on")
    if len(close) < signal.required_history:
        raise ValueError(
            f"panel has {len(close)} rows < signal required_history "
            f"{signal.required_history}; refusing to score a truncated panel (N7)"
        )
    if vol_scale is not None:
        require_vol_scale_inputs(
            vol_scale,
            config=config,
            signal=signal,
            close=close,
            regime_gross_scale=regime_gross_scale,
            targets_ledger=ctx.targets_ledger,
        )
    if cash_holding is not None:
        require_cash_holding_inputs(cash_holding, book=config.book, close=close)
    decision_bar = decision_bar or str(close.index[-1].date())
    if phase_offset is not None:
        _require_grid_schedule(phase_offset, config.decision_every, decision_bar)
    base_offset = require_tranche_inputs(
        tranches,
        config,
        phase_offset=phase_offset,
        decision_bar=decision_bar,
        regime_gross_scale=regime_gross_scale,
        wind_down=wind_down,
    )

    # 0. Record the panel this cycle decides on, before any state moves. The
    # refresh row in targets.jsonl cites its digest (step 4). The persisted
    # state is loaded first because the recorder needs to know whether this
    # bar's write-ahead step already ran: a re-run of such a bar resumes or
    # holds and decides nothing from the panel fetched now.
    state = ctx.store.load()
    tranche_session: TrancheSession | None = None
    if base_offset is not None:
        # The tranched book plans its session (layout, flat start, due tranches)
        # before anything is recorded, so each refusal leaves the run directory as it was.
        tranche_session = open_tranche_session(
            signal,
            config,
            tranches,
            base_offset,
            state,
            decision_bar,
            [str(bar.date()) for bar in close.index],
            broker=ctx.broker,
            wind_down=wind_down,
            exclude=() if cash_holding is None else (cash_holding.symbol,),
        )
    else:
        refuse_tranche_state_under_single_book(state)
    decided_already = state is not None and state.pending_decision_bar == decision_bar
    panel_record = _record_decision_panel(
        ctx, run_dir, decision_bar, close, volume, decided_already=decided_already
    )

    # 1. Settle the previous bar's decision.
    settled = _settle_prior(ctx, state, decision_bar)

    # 2. Mark broker truth at today's closes.
    prices = close.iloc[-1]
    equity, cash, prev_weights, positions = _mark_book(ctx.broker, prices)

    # 2b. Book-concordance telemetry (pure; gates nothing). The single book only:
    # a tranched book refreshes on most sessions, so its baseline would be the
    # previous session's target and the read a fill-rate read, not the drift read
    # the ledger records.
    concordance = _concordance_read(ctx, prev_weights, decision_bar) if tranche_session is None else None

    # 2c. Halt rails (kill switch, drawdown) — decided on the marked equity
    # before any decision is constructed or persisted.
    halted = _halt_check(safety, equity, ctx)

    # 2d. Regime telemetry (SPEC §7.5) — read EVERY cycle, halted or not.
    regime_state = _regime_read(regime, decision_bar)

    # 3. Cadence gate, then score → construct against held weights. Off a refresh
    # session (the decision cadence has not elapsed) the book holds its filled
    # weights — no re-score, no trades — so a monthly book rebalances monthly
    # while NAV still marks daily.
    if tranche_session is not None:
        refresh = halted is None and bool(tranche_session.due)
    else:
        refresh = halted is None and _is_refresh_session(
            state, decision_bar, close.index, config.decision_every, phase_offset
        )
    effective_max_gross, regime_scale = _effective_gross(
        config, refresh, regime_gross_scale, regime_state, decision_bar
    )

    masked: list[str] | None = None
    sizing: dict | None = None
    live_scale: LiveScale | None = None
    tranche_decision: TrancheDecision | None = None
    if refresh and tranche_session is not None:
        # 3t. The tranched book (default off): every tranche composed, the due
        # ones re-scored (or set flat under a wind-down), sized per tranche when
        # the volatility scale is on.
        tranche_decision = decide_tranches(
            tranche_session,
            signal,
            close,
            volume,
            config,
            prev_weights=prev_weights,
            positions=positions,
            equity=equity,
            unrankable=None if unrankable is None or wind_down else unrankable(decision_bar),
            decision_bar=decision_bar,
            vol_scale=vol_scale,
            exclude=() if cash_holding is None else (cash_holding.symbol,),
        )
        targets, masked = tranche_decision.targets, tranche_decision.masked
        sizing = tranche_decision.sizing
        if sizing is not None:
            _record_tranched_sizing(ctx, run_dir, sizing)
    elif refresh and vol_scale is not None:
        # 3a. The volatility scale (default off): decided from the base book
        # rebuilt on this panel, then applied inside the construct, before the
        # band and the participation gate see the target.
        live_scale = decide_live_scale(
            signal,
            close,
            volume,
            config,
            vol_scale,
            decision_bar=decision_bar,
            phase_offset=phase_offset,
            last_good=last_good_scale(ctx.targets_ledger, decision_bar),
        )
        weights_fn, applied = scaled_book_rule(config, close, effective_max_gross, live_scale)
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
            weights_fn=weights_fn,
        )
        sizing = sizing_row(live_scale, targets, applied)
        _record_sizing(ctx, run_dir, sizing)
    elif refresh:
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

    # 3c. The T-bill holding (default off): idle own equity above the buffer, set
    # after the book's target on every session that is not halted.
    cash_record: dict | None = None
    if cash_holding is not None and halted is None:
        targets, cash_decision = apply_cash_holding(
            cash_holding,
            targets,
            prev_weights,
            prices=prices,
            equity=equity,
            max_order_fraction=None if safety is None else safety.max_order_fraction,
            wind_down=wind_down,
        )
        cash_record = cash_decision.to_row()
        log_decision(cash_decision, decision_bar)

    # 4. Write-ahead decide and idempotent submit. Skipped entirely under a
    # halt — a persisted-but-unsubmitted decision would resume trading on the
    # next non-halted run, which is not what a kill switch means.
    orders: list[Order] = []
    if halted is None:
        panel_fields: dict[str, Any] | None
        if tranche_decision is not None:
            panel_fields = {**(_panel_provenance(panel_record) or {}), **tranche_decision.targets_extra}
        elif live_scale is None:
            panel_fields = _panel_provenance(panel_record)
        else:
            panel_fields = {
                **(_panel_provenance(panel_record) or {}),
                TARGETS_LEDGER_KEY: targets_ledger_record(live_scale),
            }
        if cash_record is not None and refresh:
            panel_fields = {**(panel_fields or {}), CASH_HOLDING_KEY: cash_record}
        # A tranched book's targets-ledger row is the account target its tranches'
        # holdings make, not the held weights its undecided names were sized at.
        ledger_targets = None if tranche_decision is None else tranche_decision.recorded_targets(targets)
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
            panel=panel_fields,
            components=None if tranche_decision is None else tranche_decision.components,
            buying_power_check=buying_power_check,
            ledger_targets=ledger_targets,
        )
        logger.info(
            "decision %s: %d orders %s, equity %.2f, gross %.4f",
            decision_bar,
            len(orders),
            "submitted" if transmits_to_venue(ctx.broker) else "decided and NOT submitted",
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
        # The live monitor (docs/live_monitor_design.md): today's shadow row, then
        # the return and tracking reads. Telemetry, shielded like the regime
        # provider: it can never take down a cycle whose orders are already
        # persisted and whose NAV is already marked.
        monitor_read = _monitor_read(ctx, decision_bar, close, tbill_monitor)
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
        # The index row minus its wall-clock stamp: two cycles over identical
        # inputs must produce identical results (the bit-identity tests compare
        # them field by field), and recorded_at is the one field that cannot.
        panel=None if panel_record is None else {k: v for k, v in panel_record.to_row().items() if k != "recorded_at"},
        vol_scale=sizing,
        tranches=(
            None
            if tranche_session is None
            else tranche_session.record() if tranche_decision is None else tranche_decision.record
        ),
        cash_holding=cash_record,
    )


def _record_decision_panel(
    ctx: LiveLoopContext,
    run_dir: Path | None,
    decision_bar: str,
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    *,
    decided_already: bool = False,
) -> PanelRecord | None:
    """Step 0: persist the bars this cycle decides on (``prism.live.panel_log``).

    The run directory is ``run_dir``, else the directory holding the targets
    ledger (every shipped driver puts both in the run directory). With neither
    there is nowhere to record and None is returned. Recording raises on any
    write failure, so a cycle that reaches the decision has its inputs on disk.

    ``decided_already`` says the persisted state's write-ahead step already ran
    for this bar (``pending_decision_bar == decision_bar``). Such a re-run
    resumes the persisted orders or holds the filled book; it makes no decision
    from the panel fetched now. So when that panel differs from the recorded one
    (a vendor revision, a fetch window that moved, a held name that changed in
    the fetch universe) the conflict is logged at ERROR with both digests and
    the recorded panel is returned: it is the one the decision was made on and
    stays its provenance, and the refetched panel is not recorded (a second
    index row for the bar would break the delta chain). Without a persisted
    decision for the bar a fresh decision would be made on the new content, so
    the conflict raises as before (N7).
    """
    # Imported here rather than beside ``record_panel`` above so the import block
    # stays untouched while another change lands in it; hoist when convenient.

    directory = run_dir
    if directory is None and ctx.targets_ledger is not None:
        directory = Path(ctx.targets_ledger).parent
    if directory is None:
        return None
    panel: dict[str, pd.DataFrame] = {"close": close}
    if volume is not None:
        panel["volume"] = volume
    try:
        return record_panel(directory, decision_bar, panel)
    except PanelLogConflict as exc:
        if not decided_already:
            raise
        recorded = latest_record(directory)
        if recorded is None or recorded.decision_bar != decision_bar:
            raise  # the conflict names this bar, so its record must be the newest; anything else is a log defect
        logger.error(
            "panel for %s differs from the one recorded for its persisted decision: recorded sha256 %s "
            "(%d bars from %s), refetched now %d bars from %s (%s). This cycle resumes or holds that decision "
            "and decides nothing from the refetched panel, so the recorded panel stays the decision's provenance "
            "and the refetched one is not recorded. A vendor revision or a fetch window that moved explains "
            "a small difference; anything larger is a feed defect (N7)",
            decision_bar,
            recorded.sha256,
            recorded.n_bars,
            recorded.first_bar,
            len(close),
            str(pd.Timestamp(close.index[0]).date()),
            exc,
        )
        return recorded


def _record_sizing(ctx: LiveLoopContext, run_dir: Path | None, row: dict) -> None:
    """Step 3a: write this refresh's scale decision ahead of the write-ahead step.

    The ledger is ``<run_dir>/sizing.jsonl``, where ``run_dir`` defaults to the
    directory that holds the targets ledger (the volatility scale refuses a
    context without one). A same-bar re-run finds its own row and writes
    nothing; a row that disagrees raises before anything is persisted.
    """
    if run_dir is not None:
        directory = Path(run_dir)
    elif ctx.targets_ledger is not None:
        directory = Path(ctx.targets_ledger).parent
    else:  # refused up front by require_vol_scale_inputs; kept so the record can never go nowhere
        raise ValueError("the volatility scale has no run directory to record its decision in (N7)")
    written = append_sizing_decision(directory / SIZING_LEDGER_FILENAME, row)
    message = (
        "volatility scale %.4f at refresh %s (%s%s): sigma %s over %d finite returns, zero fraction %s, last good %s; "
        "target gross %.4f, long side %.4f of equity%s"
    )
    arguments = (
        row["scale"],
        row["decision_bar"],
        row["status"],
        "" if row["reason"] is None else f", {row['reason']}",
        "n/a" if row["sigma"] is None else f"{row['sigma']:.4f}",
        row["n_valid"],
        "n/a" if row["zero_fraction"] is None else f"{row['zero_fraction']:.3f}",
        row["last_good"],
        row["target_gross"],
        row["long_over_equity"],
        "" if written else " (already recorded for this bar)",
    )
    if row["status"] == "fallback":
        # N7: loud. The book is sized at min(1, last good scale), never above the unscaled book.
        logger.warning("FALLBACK " + message, *arguments)
    else:
        logger.info(message, *arguments)


def _record_tranched_sizing(ctx: LiveLoopContext, run_dir: Path | None, row: dict) -> None:
    """Step 3t: write a tranched refresh's scale decisions ahead of the write-ahead step.

    The same ledger and directory rule as :func:`_record_sizing`, one row per
    session (``schedule = "tranches"``) carrying every due tranche's decision.
    A fallback is logged at WARNING (N7).
    """
    if run_dir is not None:
        directory = Path(run_dir)
    elif ctx.targets_ledger is not None:
        directory = Path(ctx.targets_ledger).parent
    else:  # refused up front by require_vol_scale_inputs; kept so the record can never go nowhere
        raise ValueError("the volatility scale has no run directory to record its decision in (N7)")
    written = append_sizing_decision(directory / SIZING_LEDGER_FILENAME, row)
    decisions = row["tranches"]
    described = ", ".join(
        f"{tid} {entry['scale']:.4f} {entry['status']}" + ("" if entry["reason"] is None else f" ({entry['reason']})")
        for tid, entry in decisions.items()
    )
    message = (
        "volatility scale at tranched refresh %s: %s; sigma %s over %d finite returns, zero fraction %s; account "
        "scale %s, target gross %.4f, long side %.4f of equity%s"
    )
    arguments = (
        row["decision_bar"],
        described,
        "n/a" if row["sigma"] is None else f"{row['sigma']:.4f}",
        row["n_valid"],
        "n/a" if row["zero_fraction"] is None else f"{row['zero_fraction']:.3f}",
        "n/a" if row["account_scale"] is None else f"{row['account_scale']:.4f}",
        row["target_gross"],
        row["long_over_equity"],
        "" if written else " (already recorded for this bar)",
    )
    if any(entry["status"] == "fallback" for entry in decisions.values()):
        logger.warning("FALLBACK " + message, *arguments)
    else:
        logger.info(message, *arguments)


def _panel_provenance(record: PanelRecord | None) -> dict[str, Any] | None:
    """The fields the refresh row in ``targets.jsonl`` carries for its panel."""
    if record is None:
        return None
    return {"panel_sha256": record.sha256, "panel_file": record.file}


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


def _mark_book(broker: Broker, prices: pd.Series) -> tuple[float, float, pd.Series, dict[str, float]]:
    """Step 2: broker-truth positions and cash marked at today's closes.

    Returns ``(equity, cash, prev_weights, positions)`` — the same marks
    ``decide_and_submit`` sizes with, and the non-zero share counts they were
    marked from. Every held position must price (``require_price``, N7) and
    equity must be positive.
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
    return equity, cash, prev_weights, positions


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


def _shadow_record(ctx: LiveLoopContext, decision_bar: str, close: pd.DataFrame) -> dict | None:
    """Record today's shadow return: the decided book's return on the same closes as the paper book's.

    The shadow for the session ending at ``decision_bar`` holds the latest
    refresh book decided STRICTLY before the previous bar's close, from that
    close to today's (``prism.live.monitor.shadow_session_return``), so it is the
    book the paper account should have carried through the whole session. The
    session is a *transition* when a refresh was decided at the previous bar:
    its next-open fills switched the held book mid-session, and the tracking
    read sets such sessions aside as the execution wedge. Nothing is recorded,
    and the reason is logged, when there is no previous session in the equity
    ledger, when it is not the exchange session right before today (a gap is not
    a day), or when no refresh book predates it. The returned row is what was
    appended to ``ctx.shadow_ledger`` (idempotent per decision bar).
    """
    if ctx.equity_ledger is None or ctx.targets_ledger is None or ctx.shadow_ledger is None:
        return None
    equity = read_equity_ledger(ctx.equity_ledger)
    if equity.empty:
        return None
    earlier = equity[equity["decision_bar"].astype(str) < decision_bar].sort_values("decision_bar")
    if earlier.empty:
        logger.info("shadow %s: no earlier session in the equity ledger; nothing to shadow yet", decision_bar)
        return None
    prev_bar = str(earlier["decision_bar"].iloc[-1])
    prev_date, today_date = bar_date(prev_bar), bar_date(decision_bar)
    if not (sessions.is_session(prev_date) and sessions.next_session(prev_date) == today_date):
        logger.info(
            "shadow %s: previous ledger bar %s is not the preceding exchange session; a gap is not a day, "
            "no shadow row",
            decision_bar,
            prev_bar,
        )
        return None
    refreshes = read_targets_ledger(ctx.targets_ledger)
    prior = [row for row in refreshes if str(row["refresh_bar"]) < prev_bar]
    if not prior:
        logger.info("shadow %s: no refresh book decided before %s; no shadow row", decision_bar, prev_bar)
        return None
    baseline = prior[-1]
    transition = any(str(row["refresh_bar"]) == prev_bar for row in refreshes)
    nav_now = float(equity.loc[equity["decision_bar"].astype(str) == decision_bar, "equity"].iloc[-1])
    nav_prev = float(earlier["equity"].iloc[-1])
    shadow = shadow_session_return(close, baseline["targets"], decision_bar, prev_bar)
    row = {
        "prev_bar": prev_bar,
        "shadow_return": shadow["shadow_return"],
        "paper_return": nav_now / nav_prev - 1.0,
        "transition": transition,
        "refresh_bar": str(baseline["refresh_bar"]),
        "n_names_used": shadow["n_names_used"],
        "n_names_missing": shadow["n_names_missing"],
        "gross_missing": shadow["gross_missing"],
    }
    append_shadow_ledger(ctx.shadow_ledger, decision_bar, row)
    logger.info(
        "shadow %s (from %s, book of %s%s): shadow=%.6f paper=%.6f diff=%.6f names %d used / %d missing",
        decision_bar,
        prev_bar,
        row["refresh_bar"],
        ", transition" if transition else "",
        row["shadow_return"],
        row["paper_return"],
        row["paper_return"] - row["shadow_return"],
        row["n_names_used"],
        row["n_names_missing"],
    )
    return row


def _monitor_read(
    ctx: LiveLoopContext, decision_bar: str, close: pd.DataFrame, tbill: TbillReference | None = None
) -> dict:
    """The live monitor after this cycle's NAV mark, shielded so it can never kill a cycle.

    Records the shadow row, then reads the return monitor and, when a shadow
    ledger is wired, the tracking monitor (``paper_monitor_read`` attaches it
    under ``tracking``). Either part raising is a defect in telemetry, not in the
    book: it is logged at WARNING with its type and the read carries
    ``verdict = "unavailable"`` and the error, so a broken monitor is visible in
    the log and the session record and never aborts a legitimate session whose
    orders are persisted and whose NAV is marked (N7: loud and named, not fatal).

    With ``tbill`` the report-only T-bill-referenced read is attached under
    ``tbill`` after the registered read is complete (:func:`_tbill_read`); the
    registered read is the same dict, key for key, as without it.
    """
    assert ctx.equity_ledger is not None
    try:
        _shadow_record(ctx, decision_bar, close)
    except Exception as exc:  # noqa: BLE001 — telemetry, never a cycle precondition
        logger.warning(
            "LIVE MONITOR: SHADOW RECORD FAILED for %s (%s: %s) — no shadow row this session; the "
            "tracking read will count it as unmatched (N7)",
            decision_bar,
            type(exc).__name__,
            exc,
        )
    try:
        read = paper_monitor_read(ctx.equity_ledger, shadow_ledger_path=ctx.shadow_ledger)
    except Exception as exc:  # noqa: BLE001 — telemetry, never a cycle precondition
        logger.warning(
            "LIVE MONITOR FAILED for %s (%s: %s) — the cycle's orders and NAV mark stand; the monitor "
            "read is unavailable this session (N7)",
            decision_bar,
            type(exc).__name__,
            exc,
        )
        return {"verdict": "unavailable", "n": 0, "error": type(exc).__name__, "detail": str(exc)}
    logger.info(
        "monitor %s: verdict=%s n=%s mean=%s ci=[%s, %s] method=%s gaps=%s",
        decision_bar,
        read.get("verdict"),
        read.get("n"),
        read.get("mean"),
        read.get("ci_lower"),
        read.get("ci_upper"),
        read.get("method"),
        read.get("n_gaps"),
    )
    tracking = read.get("tracking")
    if tracking is not None:
        logger.info(
            "tracking %s: verdict=%s n=%s mean_diff=%s ci=[%s, %s] corr=%s slope=%s transitions=%s unmatched=%s",
            decision_bar,
            tracking.get("verdict"),
            tracking.get("n"),
            tracking.get("mean_difference"),
            tracking.get("ci_lower"),
            tracking.get("ci_upper"),
            None if tracking.get("correlation") is None else round(tracking["correlation"], 4),
            None if tracking.get("slope") is None else round(tracking["slope"], 4),
            tracking.get("n_transition"),
            tracking.get("n_unmatched"),
        )
    if tbill is not None:
        read["tbill"] = _tbill_read(ctx, decision_bar, tbill)
    return read


def _tbill_read(ctx: LiveLoopContext, decision_bar: str, tbill: TbillReference) -> dict:
    """The report-only T-bill-referenced return read, shielded like the registered one.

    A failure is logged at WARNING and recorded as ``unavailable`` with the
    error; it never changes the registered read or the cycle.
    """
    assert ctx.equity_ledger is not None
    try:
        read = tbill_monitor_read(ctx.equity_ledger, tbill, targets_ledger_path=ctx.targets_ledger)
    except Exception as exc:  # noqa: BLE001 — report-only telemetry, never a cycle precondition
        logger.warning(
            "LIVE MONITOR: T-BILL READ FAILED for %s (%s: %s) — report only; the registered read stands (N7)",
            decision_bar,
            type(exc).__name__,
            exc,
        )
        return {"verdict": "unavailable", "n": 0, "report_only": True, "error": type(exc).__name__,
                "detail": str(exc)}
    logger.info(
        "monitor vs T-bills (report only) %s: verdict=%s n=%s mean=%s hurdle=%s ci=[%s, %s] long=%s etf=%s "
        "detection_years=%s",
        decision_bar,
        read.get("verdict"),
        read.get("n"),
        read.get("mean"),
        read.get("hurdle"),
        read.get("ci_lower"),
        read.get("ci_upper"),
        round(read["exposure"]["long_over_equity_mean"], 4),
        round(read["exposure"]["sleeve_fraction_mean"], 4),
        None if read["detection_horizon"]["years"] is None else round(read["detection_horizon"]["years"], 1),
    )
    return read


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


def _require_bounded_regime_hook(
    config: DailyBookConfig, regime_gross_scale: Callable[[dict], float] | None
) -> None:
    """Refuse the §7.5 de-gross hook on a book configured above 1x gross (N7).

    The hook multiplies ``config.max_gross`` by a factor clamped to ``[0, 1]``,
    which de-grosses a book whose ceiling is its unscaled gross. With a ceiling
    above 1 the same factor admits a book above the unscaled one -- the
    inverse-vol book is scaled to its ceiling outright -- so a de-gross action
    could lever up. It has no CLI path, so the live loop never reaches this; the
    refusal keeps a future caller from arming it on a levered configuration.
    """
    if regime_gross_scale is not None and config.max_gross > 1.0:
        raise ValueError(
            f"regime_gross_scale is refused with max_gross={config.max_gross} > 1.0: the hook scales the "
            "configured gross ceiling, so above 1 a de-gross action could size a book above the unscaled one (N7)"
        )


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
    *,
    weights_fn: Callable[[pd.Series], pd.Series] | None = None,
) -> tuple[pd.Series, list[str] | None]:
    """Step 3 on a refresh session: the shared decision assembly (§3.8).

    The score → construct → band → participation segment is
    :func:`prism.assembly.assemble`, so this book and any evaluation of it
    run the same code. What stays here is what the mask contract makes the
    caller's: the driver consults the ``unrankable`` provider and hands the
    result in — assembly never fetches it — and reports back which names it
    actually applied to (the provider's list intersected with the scored
    cross-section).

    ``weights_fn`` replaces the book's construct as the component's rule; the
    volatility scale passes the construct times its scale
    (:func:`prism.live.vol_sizing.scaled_book_rule`). ``None`` is the construct.

    Returns ``(targets, masked)``. No order intents are sized here: the
    write-ahead protocol reconciles against broker truth and sizes inside
    ``decide_and_submit``, which is why ``AccountState.positions`` is None.
    """
    supplied = None if unrankable is None else frozenset(unrankable(decision_bar))
    plan = ComponentPlan(
        component=signal,
        component_id=config.book,
        to_weights=config.book,
        weights_fn=heuristic_book_rule(config, close, effective_max_gross) if weights_fn is None else weights_fn,
    )
    output = assemble(
        DecisionContext(
            decision_bar=decision_bar,
            components=(plan,),
            scoring_panels={"price": close},
            volume=volume,
            construction=config,
            state=AccountState(equity=equity, held_weights=prev_weights, positions=None),
            sessions=[str(bar.date()) for bar in close.index],
            unrankable=() if supplied is None else supplied,
            max_gross=effective_max_gross,
        )
    )
    masked = None if supplied is None else sorted(output.diagnostics["masked"])
    if masked:
        # Unrankable names (spin-off inside the lookback — see the
        # run_daily_cycle docstring) leave the scored cross-section entirely,
        # so decile membership is decided over rankable names only, and they
        # then carry NO decision: a held position is held until the window
        # clears the event, and no new position opens on a divergent rank.
        logger.info(
            "unrankable mask: %d name(s) held out of ranking this refresh: %s",
            len(masked),
            masked,
        )
    return output.targets, masked


#: The book construct moved to :mod:`prism.assembly` (engine specification §3.8);
#: the name stays because the parity tests are written against it.
_construct_raw_book = raw_book_weights


def _is_refresh_session(
    state: LoopState | None,
    decision_bar: str,
    index: pd.DatetimeIndex,
    decision_every: int,
    phase_offset: int | None = None,
) -> bool:
    """Whether this session refreshes the book (SPEC §7.7 decision cadence).

    ``decision_every`` trading sessions must elapse between refreshes (1 = every
    session, the frozen daily default). Without a ``phase_offset`` the first
    cycle, or a state with no cadence anchor, always refreshes. The elapsed
    count is read off the panel's own index — the trading calendar the loop
    already holds — so it matches the backtest's bar-count cadence, not
    wall-clock months.

    With a ``phase_offset`` the book is on the session grid
    (:func:`prism.composition.due_on_schedule`, the predicate composition uses):
    due on its grid dates, waiting for the first one when it has never
    refreshed, and catching up once after a missed one.
    """
    if decision_every <= 1:
        return True
    last_refresh = state.last_refresh_bar if state is not None else None
    elapsed = 0
    if last_refresh is not None:
        dates = [str(d.date()) for d in index]
        elapsed = sum(1 for dt in dates if last_refresh < dt <= decision_bar)
    return due_on_schedule(
        decision_every, phase_offset, last_refresh, decision_bar, elapsed=elapsed, owner="the live book"
    )


def _require_grid_schedule(phase_offset: int, decision_every: int, decision_bar: str) -> None:
    """Refuse a grid schedule the cycle cannot honour, before any state moves (N7).

    The offset is a residue of the refresh cycle, so it must be an integer in
    ``[0, decision_every)``; a daily book has no phase to set; and the decision
    bar must have a grid position, which a closed day or a date outside the
    bundled session table does not.
    """
    if decision_every <= 1:
        raise ValueError(
            "phase_offset requires decision_every > 1: a book that refreshes every session has no phase to set"
        )
    if isinstance(phase_offset, bool) or not isinstance(phase_offset, int) or not 0 <= phase_offset < decision_every:
        raise ValueError(
            f"phase_offset must be an integer in [0, decision_every={decision_every}), got {phase_offset!r}"
        )
    try:
        sessions.grid_position(dt.date.fromisoformat(decision_bar[:10]))
    except ValueError as exc:
        raise ValueError(
            f"decision bar {decision_bar!r} has no session-grid position, so the seeded phase cannot place it: "
            f"{exc}; refusing to decide (N7)"
        ) from exc
