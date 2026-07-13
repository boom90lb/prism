"""Historical replay of the live daily cycle (SPEC.md §7.7) — a diagnostic
instrument, not an evidence source.

Drives the *real* ``run_daily_cycle`` — write-ahead protocol, tolerant settle,
cadence gate, online band, monitor — over local historical bars with a
simulated venue, so many cycles run in minutes instead of one per trading day.
Promoted from the scratch harnesses that validated the live spine and the
momentum-book cutover before their first live cycles (2026-07-08).

What a replay is for: mechanics validation and pre-flight — does the spine run
clean and causally over N cycles, does a cutover state flatten and rebalance,
does the cadence hold, does the monitor accrue. What a replay is **not**
(machinery ≠ claim):

* **Never I-9 cost calibration.** :class:`ReplayBroker` fills at the next
  bar's open print exactly — zero spread, zero impact — because any richer
  fill needs a fill *model*, and calibrating the spread schedule against
  one's own fill model is the cost-calibration circularity docs/handoff.md
  names. The ``fills.jsonl`` a replay writes is a replay artifact; it must
  never feed ``prism.execution.spread`` calibration.
* **Never the promotion gate's live-monitor conjunct.** The concordance read
  (docs/momentum_design.md §3) is defined on the *live paper stream*; a
  replayed ``equity.jsonl`` is a modeled-fill diagnostic stream. The two must
  stay in separate run-dirs — :func:`replay_daily_cycles` refuses to write
  into a run-dir that already holds loop state, so a replay cannot land in
  (or resume) a live ledger by accident.

Production-import-path safe (N8): numpy/pandas only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from prism.live.broker import Broker, DuplicateOrder, Fill, Order
from prism.live.daily import DailyBookConfig, DailyCycleResult, run_daily_cycle
from prism.live.loop import LiveLoopContext
from prism.live.state import StateStore
from prism.signal.base import Signal

logger = logging.getLogger(__name__)


class ReplayBroker(Broker):
    """In-memory venue: each pending order fills once, at the armed next open.

    The driver arms each cycle's fill prices with :meth:`set_fill` *before*
    running the cycle, so orders decided at close *t−1* settle at bar *t*'s
    open — the N2 next-open semantics the live OPG orders carry. A name whose
    open is missing/non-finite falls back to the order's decision-close
    ``reference_price`` with a loud warning (a replay data gap, not a venue
    outcome). Positions and cash are broker truth for the loop's reconcile.
    """

    def __init__(self, cash: float, positions: dict[str, float] | None = None) -> None:
        self._cash = float(cash)
        self._pos: dict[str, float] = dict(positions or {})
        self._orders: dict[str, Order] = {}
        self._filled: set[str] = set()
        self._open_row: pd.Series | None = None
        self._fill_bar: str | None = None

    def set_fill(self, open_row: pd.Series, bar: str) -> None:
        """Arm the open prices (and bar label) the next settle fills at."""
        self._open_row = open_row
        self._fill_bar = bar

    def positions(self) -> dict[str, float]:
        return {s: q for s, q in self._pos.items() if q != 0.0}

    def cash(self) -> float:
        return self._cash

    def submit(self, order: Order) -> None:
        if order.client_order_id in self._orders:
            raise DuplicateOrder(order.client_order_id)
        self._orders[order.client_order_id] = order

    def submitted_order_ids(self) -> set[str]:
        return set(self._orders)

    def fills_for(self, client_order_ids: set[str]) -> list[Fill]:
        if self._open_row is None or self._fill_bar is None:
            raise RuntimeError("ReplayBroker has no armed fill bar — call set_fill first (N7)")
        fills: list[Fill] = []
        for oid in sorted(client_order_ids & set(self._orders)):
            if oid in self._filled:
                continue
            order = self._orders[oid]
            raw = self._open_row.get(order.symbol)
            price = float(raw) if raw is not None and np.isfinite(raw) and raw > 0 else None
            if price is None:
                logger.warning(
                    "no open print for %s at %s — filling at the decision-close reference "
                    "%.4f (replay data gap; a live venue would have expired the order)",
                    order.symbol,
                    self._fill_bar,
                    order.reference_price,
                )
                price = float(order.reference_price)
            self._pos[order.symbol] = self._pos.get(order.symbol, 0.0) + order.qty
            self._cash -= order.qty * price
            self._filled.add(oid)
            fills.append(
                Fill(
                    client_order_id=oid,
                    symbol=order.symbol,
                    qty=order.qty,
                    price=price,
                    filled_bar=self._fill_bar,
                )
            )
        return fills


def load_local_bar_panels(
    symbols: Sequence[str],
    data_dir: str | Path = "data",
    *,
    interval: str = "1d",
    max_missing: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wide (close, volume, open) panels from the local parquet caches, offline.

    Concatenates every ``{symbol}_{interval}_*.parquet`` under ``data_dir``
    (the main range cache plus any delta caches), de-duplicated keep-last and
    sorted — the same union the incremental store would serve, without a
    network fetch. Replays need the *open* column for fills, which is why this
    does not reuse ``fetch_universe_panels`` (close/volume only).

    ``max_missing`` mirrors ``fetch_universe_panels``: the fraction of symbols
    allowed to have no local cache before failing loud (N7); tolerated misses
    are dropped with a warning naming them.
    """
    if not symbols:
        raise ValueError("empty symbol list — nothing to load")
    data_dir = Path(data_dir)
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    missing: list[str] = []
    for symbol in symbols:
        paths = sorted(data_dir.glob(f"{symbol}_{interval}_*.parquet"))
        if not paths:
            missing.append(symbol)
            continue
        bars = pd.concat([pd.read_parquet(p) for p in paths])
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        if bars.empty or "close" not in bars.columns or "open" not in bars.columns:
            missing.append(symbol)
            continue
        closes[symbol] = pd.to_numeric(bars["close"], errors="coerce")
        opens[symbol] = pd.to_numeric(bars["open"], errors="coerce")
        if "volume" in bars.columns:
            volumes[symbol] = pd.to_numeric(bars["volume"], errors="coerce")
    if missing:
        shown = missing[:20]
        tail = "…" if len(missing) > len(shown) else ""
        if len(missing) > max_missing * len(symbols):
            raise RuntimeError(
                f"no local bars for {len(missing)}/{len(symbols)} symbols "
                f"({len(missing) / len(symbols):.1%} > max_missing {max_missing:.1%}) "
                f"under {data_dir}: {shown}{tail}; fetch them first or raise max_missing (N7)"
            )
        logger.warning(
            "no local bars for %d/%d symbols (<= max_missing %.1f%%); replaying without them: %s%s",
            len(missing),
            len(symbols),
            100 * max_missing,
            shown,
            tail,
        )
    if not closes:
        raise RuntimeError(f"no local bars for any of the {len(symbols)} requested symbols (N7)")
    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex(columns=close.columns).sort_index()
    open_ = pd.DataFrame(opens).reindex(columns=close.columns).sort_index()
    return close, volume, open_


def align_replay_panels(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    open_: pd.DataFrame,
    *,
    consensus: float = 0.95,
    clean_window: int = 400,
    keep: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restrict ragged per-name panels to a consensus calendar and clean names.

    A naive union of per-name vendor caches is ragged (vendor-holiday drift),
    and a strict full-history eligibility screen NaNs the whole book on the
    stray rows. The live venue serves ONE calendar for every name, so a replay
    mirrors that: keep rows where at least ``consensus`` of names have a close,
    then keep names with a complete close history over the trailing
    ``clean_window`` rows. ``keep`` names (e.g. an inherited cutover book that
    the mark step must price, N7) must survive the screen: one that does not
    raises loudly rather than silently vanishing from the replayed book.
    """
    rows = close.index[close.notna().sum(axis=1) >= consensus * close.shape[1]]
    close, volume, open_ = close.loc[rows], volume.loc[rows], open_.loc[rows]
    recent = close.tail(clean_window)
    clean = [s for s in close.columns if bool(recent[s].notna().all())]
    dropped_keep = [s for s in keep if s not in clean]
    if dropped_keep:
        raise ValueError(
            f"keep names {dropped_keep} lack a clean {clean_window}-bar close history — "
            f"the replay could not price them at the mark step (N7)"
        )
    return close[clean], volume[clean], open_[clean]


def replay_daily_cycles(
    signal_factory: Callable[[], Signal],
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    open_: pd.DataFrame,
    config: DailyBookConfig,
    run_dir: str | Path,
    *,
    start_bar: str | None = None,
    end_bar: str | None = None,
    initial_cash: float = 100_000.0,
    initial_positions: dict[str, float] | None = None,
) -> tuple[list[DailyCycleResult], ReplayBroker]:
    """Run the real daily cycle over every bar in ``[start_bar, end_bar]``.

    Each cycle *t*: arm bar *t*'s opens on the broker (the prior decision's
    N2 next-open fills), fit a fresh ``signal_factory()`` on the trailing
    panels through *t* (the live loop's own refit-per-run staleness policy —
    causal by construction, and free for a stateless node whose fit only
    validates), then ``run_daily_cycle`` on the truncated panels. The final
    bar's orders remain pending, exactly as a live loop ends its day.

    ``run_dir`` receives ``state.json`` / ``fills.jsonl`` / ``equity.jsonl``.
    It must not already hold loop state: a replay stream is a modeled-fill
    diagnostic and may never resume — or be mistaken for — a live ledger
    (see the module docstring), so an existing ``state.json`` refuses loudly
    rather than being adopted.

    Returns the per-cycle results and the broker (final positions/cash).
    """
    run_dir = Path(run_dir)
    if (run_dir / "state.json").exists():
        raise RuntimeError(
            f"{run_dir} already holds loop state — refusing to replay into an existing "
            f"run-dir (a replay stream must stay separate from any live ledger); "
            f"point --run-dir at a fresh directory or delete the old replay first"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    required = signal_factory().required_history
    dates = [str(d.date()) for d in close.index]
    bars = [
        (i, t)
        for i, t in enumerate(close.index)
        if (start_bar is None or dates[i] >= start_bar) and (end_bar is None or dates[i] <= end_bar)
    ]
    if not bars:
        raise ValueError(f"no bars in [{start_bar}, {end_bar}] — nothing to replay")
    first_pos = bars[0][0]
    if first_pos + 1 < required:
        raise ValueError(
            f"first replay bar {dates[first_pos]} has only {first_pos + 1} trailing rows "
            f"< required_history {required}; start the window later or extend the panel (N7)"
        )

    broker = ReplayBroker(initial_cash, initial_positions)
    ctx = LiveLoopContext(
        store=StateStore(run_dir / "state.json"),
        broker=broker,
        fills_ledger=run_dir / "fills.jsonl",
        equity_ledger=run_dir / "equity.jsonl",
        # The per-refresh decided book, same artifact the live loop persists —
        # the concordance-diagnostic object (replay book vs the research
        # backtest's target rows). Still a replay artifact: never calibration,
        # never the live concordance stream.
        targets_ledger=run_dir / "targets.jsonl",
    )
    logger.info(
        "replay: %d cycles %s -> %s over %d names (modeled next-open fills — diagnostic stream)",
        len(bars),
        dates[bars[0][0]],
        dates[bars[-1][0]],
        close.shape[1],
    )
    results: list[DailyCycleResult] = []
    for i, t in bars:
        sub_close = close.iloc[: i + 1]
        sub_volume = volume.iloc[: i + 1] if volume is not None else None
        broker.set_fill(open_.loc[t], dates[i])
        signal = signal_factory().fit(sub_close, sub_volume)
        results.append(run_daily_cycle(ctx, signal, sub_close, sub_volume, config, decision_bar=dates[i]))
    return results, broker
