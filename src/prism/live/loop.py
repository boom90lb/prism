"""The write-ahead decision protocol (SPEC.md §7.7).

The crash window that matters is between the decision (close *t*) and the
fill (open *t+1*): a restarted process must neither lose the queued orders
nor submit them twice. The protocol:

1. **Reconcile.** Broker truth wins for positions and cash; a divergence
   from the persisted state is logged loudly and adopted, never papered
   over.
2. **Decide once.** Target weights (already constructed upstream against
   held weights — I-4) are diffed into share orders at decision-close
   prices.
3. **Write ahead.** The pending orders are persisted *before* the first
   submit. From that moment the decision for the bar is immutable: a
   restart resumes submission from the persisted record and never
   re-decides (re-deciding after a partial submit double-trades).
4. **Submit idempotently.** Every order carries a deterministic
   ``client_order_id`` (``"{decision_bar}:{symbol}"``); ids the broker
   already knows are skipped, and a venue duplicate rejection counts as
   success.
5. **Settle.** Next session, fills are pulled, appended to the I-9 fills
   ledger (fill price beside the decision-close reference price — the
   arrival-slippage record the paper instrument exists to collect), and
   the state is re-anchored to broker truth with pending cleared.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from prism.live.broker import Broker, DuplicateOrder, Fill, Order
from prism.live.state import LoopState, StateStore

logger = logging.getLogger(__name__)

# Positions within this many shares of each other reconcile silently;
# anything larger is logged as a real divergence before broker truth wins.
_RECONCILE_SHARE_TOLERANCE = 1e-6


@dataclass
class LiveLoopContext:
    """The durable pieces a loop step operates on."""

    store: StateStore
    broker: Broker
    fills_ledger: Path
    equity_ledger: Path | None = None


def targets_to_orders(
    target_weights: pd.Series,
    positions: dict[str, float],
    prices: pd.Series,
    equity: float,
    decision_bar: str,
    min_order_notional: float = 0.0,
    whole_shares: bool = False,
) -> list[Order]:
    """Diff constructed target weights against held shares into orders.

    A NaN target is "no target" — the position is left alone (the signal
    layer's no-opinion never forces a liquidation; construction emits an
    explicit 0.0 when it wants flat). A symbol with a target but no finite
    positive price cannot be sized and raises (N7). Orders below
    ``min_order_notional`` dollars are dropped as dust. ``whole_shares``
    rounds each share *delta* to the nearest integer (required by
    market-on-open venue order types, e.g. Alpaca OPG); a delta that rounds
    to zero is dropped like dust.
    """
    if equity <= 0:
        raise ValueError(f"equity must be > 0 to size a book, got {equity}")
    orders: list[Order] = []
    for symbol, weight in target_weights.items():
        if isinstance(weight, float) and math.isnan(weight):
            continue
        price = prices.get(symbol)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            raise ValueError(
                f"no usable decision price for {symbol!r} (got {price!r}); "
                "cannot size its target (N7)"
            )
        price = float(price)
        target_shares = float(weight) * equity / price
        delta = target_shares - positions.get(symbol, 0.0)
        if whole_shares:
            delta = float(round(delta))
        if abs(delta) * price < max(min_order_notional, 1e-9):
            continue
        orders.append(
            Order(
                client_order_id=f"{decision_bar}:{symbol}",
                symbol=str(symbol),
                qty=delta,
                decision_bar=decision_bar,
                reference_price=price,
            )
        )
    return orders


def decide_and_submit(
    ctx: LiveLoopContext,
    decision_bar: str,
    target_weights: pd.Series,
    prices: pd.Series,
    min_order_notional: float = 0.0,
    whole_shares: bool = False,
) -> list[Order]:
    """One decision step: reconcile → decide once → write ahead → submit.

    Re-running for the same ``decision_bar`` after a crash resumes the
    persisted decision idempotently. Calling for a new bar while a prior
    bar's orders are still pending raises — settle first, the book's true
    holdings are unknown until then.
    """
    state = ctx.store.load() or LoopState()

    if state.pending_orders:
        if state.pending_decision_bar != decision_bar:
            raise RuntimeError(
                f"pending orders from {state.pending_decision_bar} are unsettled; "
                f"settle before deciding {decision_bar} (N2 ordering)"
            )
        logger.warning(
            "resuming persisted decision for %s (%d orders) — not re-deciding",
            decision_bar,
            len(state.pending_orders),
        )
    else:
        if state.last_settled_bar is not None and decision_bar <= state.last_settled_bar:
            raise RuntimeError(
                f"decision bar {decision_bar} is not after last settled bar "
                f"{state.last_settled_bar}; refusing to re-decide a processed bar"
            )
        _reconcile(state, ctx.broker)
        equity = state.cash + sum(
            shares * _require_price(prices, symbol) for symbol, shares in state.positions.items()
        )
        state.pending_orders = targets_to_orders(
            target_weights,
            state.positions,
            prices,
            equity,
            decision_bar,
            min_order_notional=min_order_notional,
            whole_shares=whole_shares,
        )
        state.pending_decision_bar = decision_bar
        ctx.store.save(state)  # the write-ahead: persisted before any submit

    known = ctx.broker.submitted_order_ids()
    for order in state.pending_orders:
        if order.client_order_id in known:
            continue
        try:
            ctx.broker.submit(order)
        except DuplicateOrder:
            logger.info("order %s already at venue; treating as submitted", order.client_order_id)
    return list(state.pending_orders)


def settle(ctx: LiveLoopContext, settle_bar: str) -> list[Fill]:
    """Pull fills for the pending decision, ledger them, re-anchor state.

    Every pending order must have a completed fill; a missing fill means
    the book's holdings are not what the decision assumed, and that is an
    operator problem, not something to average over (N7).
    """
    state = ctx.store.load()
    if state is None or not state.pending_orders:
        return []

    pending_ids = {o.client_order_id for o in state.pending_orders}
    fills = ctx.broker.fills_for(pending_ids)
    filled_ids = {f.client_order_id for f in fills}
    missing = pending_ids - filled_ids
    if missing:
        raise RuntimeError(
            f"{len(missing)} pending orders have no fill at settle "
            f"({sorted(missing)[:5]}…); reconcile manually before continuing (N7)"
        )

    reference = {o.client_order_id: o.reference_price for o in state.pending_orders}
    _append_fills_ledger(
        ctx.fills_ledger,
        [
            {
                "client_order_id": f.client_order_id,
                "symbol": f.symbol,
                "qty": f.qty,
                "fill_price": f.price,
                "reference_price": reference[f.client_order_id],
                "decision_bar": state.pending_decision_bar,
                "filled_bar": f.filled_bar,
            }
            for f in fills
        ],
    )

    state.positions = {s: q for s, q in ctx.broker.positions().items() if q != 0.0}
    state.cash = ctx.broker.cash()
    state.last_settled_bar = settle_bar
    state.pending_orders = []
    state.pending_decision_bar = None
    ctx.store.save(state)
    return fills


def read_fills_ledger(path: Path) -> pd.DataFrame:
    """The append-only fills ledger as a frame (the I-9 calibration input)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)


def read_equity_ledger(path: Path) -> pd.DataFrame:
    """The append-only equity ledger as a frame — one mark-to-market NAV row per
    decision bar (``decision_bar``, ``equity``, ``cash``). This is the
    return-series source for the anytime-valid monitor
    (:mod:`prism.validation.anytime` via :mod:`prism.live.monitor`)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)


def _reconcile(state: LoopState, broker: Broker) -> None:
    """Adopt broker truth, loudly when it disagrees with the persisted book."""
    broker_positions = {s: q for s, q in broker.positions().items() if q != 0.0}
    symbols = set(broker_positions) | set(state.positions)
    diverged = {
        s
        for s in symbols
        if abs(broker_positions.get(s, 0.0) - state.positions.get(s, 0.0))
        > _RECONCILE_SHARE_TOLERANCE
    }
    if diverged:
        logger.warning(
            "reconcile: broker truth differs from persisted state on %s; adopting broker",
            sorted(diverged),
        )
    state.positions = broker_positions
    state.cash = broker.cash()


def _require_price(prices: pd.Series, symbol: str) -> float:
    price = prices.get(symbol)
    if price is None or not math.isfinite(float(price)) or float(price) <= 0:
        raise ValueError(f"cannot value held position {symbol!r}: price {price!r} (N7)")
    return float(price)


def _append_fills_ledger(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _append_equity_ledger(path: Path, decision_bar: str, equity: float, cash: float) -> None:
    """Append one mark-to-market NAV snapshot for ``decision_bar``.

    Idempotent and monotone: a same-bar rerun (the write-ahead protocol's
    restart) or an out-of-order bar is skipped, so the ledger holds exactly one
    row per bar and re-running the loop never double-counts a daily return.
    ``decision_bar`` strings are ISO dates, so lexical ``<=`` is chronological.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines and decision_bar <= json.loads(lines[-1])["decision_bar"]:
            return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"decision_bar": decision_bar, "equity": float(equity), "cash": float(cash)},
                sort_keys=True,
            )
            + "\n"
        )
