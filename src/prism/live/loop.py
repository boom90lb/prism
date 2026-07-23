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
   ``client_order_id`` (``"{prefix}{decision_bar}:{symbol}"`` — the prefix
   namespaces books sharing one venue account); ids the broker already
   knows are skipped, and a venue duplicate rejection counts as success.
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
from typing import Callable

import pandas as pd

from prism.live.broker import Broker, DuplicateOrder, Fill, Order, OrderRejected
from prism.live.state import LoopState, StateStore

logger = logging.getLogger(__name__)

# Positions within this many shares of each other reconcile silently;
# anything larger is logged as a real divergence before broker truth wins.
_RECONCILE_SHARE_TOLERANCE = 1e-6


@dataclass
class LiveLoopContext:
    """The durable pieces a loop step operates on.

    The optional ledgers are additive telemetry (SPEC §10: no ratified
    statistic moves): ``targets_ledger`` records the constructed book at each
    refresh (what the instrument *should* hold — the concordance baseline and
    the sweep's audit record), ``unfilled_ledger`` records every unexecuted
    residual at settle (the orders the venue did not print — previously only a
    truncated log line), ``concordance_ledger`` records how faithfully the
    held book tracks the last refresh targets, and ``regime_ledger`` records
    the per-cycle SPEC §7.5 regime telemetry read — the session record the
    handoff §8 precondition-(b) 21-session clock consults
    (docs/regime_step.md). ``order_id_prefix`` namespaces
    client order ids per book/run-dir (``"mom:"`` for the momentum instrument):
    two books sharing one venue account with the bare ``{bar}:{symbol}`` scheme
    silently substitute each other's same-bar orders, because a venue duplicate
    id is treated as submission success.
    """

    store: StateStore
    broker: Broker
    fills_ledger: Path
    equity_ledger: Path | None = None
    targets_ledger: Path | None = None
    unfilled_ledger: Path | None = None
    concordance_ledger: Path | None = None
    regime_ledger: Path | None = None
    order_id_prefix: str = ""


def targets_to_orders(
    target_weights: pd.Series,
    positions: dict[str, float],
    prices: pd.Series,
    equity: float,
    decision_bar: str,
    min_order_notional: float = 0.0,
    whole_shares: bool = False,
    order_id_prefix: str = "",
) -> list[Order]:
    """Diff constructed target weights against held shares into orders.

    A NaN target is "no target" — the position is left alone (the signal
    layer's no-opinion never forces a liquidation; construction emits an
    explicit 0.0 when it wants flat). A symbol with a *non-flat* target but no
    finite positive price cannot be sized and raises (N7); a flat (0.0) target on
    an unheld name needs no price and is skipped. Orders below
    ``min_order_notional`` dollars are dropped as dust. ``whole_shares``
    rounds each share *delta* to the nearest integer (required by
    market-on-open venue order types, e.g. Alpaca OPG); a delta that rounds
    to zero is dropped like dust — and because a censored order is a silent
    book/target divergence (a $2,000+ name can never enter a $1,000-per-name
    book), every drop whose *intended* notional was economically real is
    logged loudly with its size (N7: loud, not silent). An order that would
    flip a held position through zero (short→long or long→short) is clamped
    to flat this bar — a venue rejects a side-crossing single order
    ("insufficient qty available"), so the opposite side opens next bar from
    a flat book. ``order_id_prefix`` namespaces the deterministic client id
    (``"{prefix}{decision_bar}:{symbol}"``) so two books on one venue account
    can never collide on the same bar and symbol.
    """
    if equity <= 0:
        raise ValueError(f"equity must be > 0 to size a book, got {equity}")
    orders: list[Order] = []
    dust_dropped: list[tuple[str, float]] = []  # (symbol, intended $ notional)
    for symbol, weight in target_weights.items():
        if isinstance(weight, float) and math.isnan(weight):
            continue
        held = positions.get(symbol, 0.0)
        if float(weight) == 0.0 and held == 0.0:
            # Flat target with nothing held: no trade to size, so no decision
            # price is needed. A large universe carries names the construct left
            # at 0.0 (ineligible, or a gapped latest bar) whose price is NaN — a
            # no-op here, not an N7 error.
            continue
        price = prices.get(symbol)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            raise ValueError(
                f"no usable decision price for {symbol!r} (got {price!r}); "
                "cannot size its target (N7)"
            )
        price = float(price)
        target_shares = float(weight) * equity / price
        if held != 0.0 and target_shares != 0.0 and (held > 0.0) != (target_shares > 0.0):
            # A single venue order may not cross zero — Alpaca rejects flipping a
            # position's side in one order ("insufficient qty available"). Trade
            # only to flat this bar; the opposite side opens next bar from flat.
            delta = -held
        else:
            delta = target_shares - held
        intended_notional = abs(delta) * price
        if whole_shares:
            delta = float(round(delta))
        if abs(delta) * price < max(min_order_notional, 1e-9):
            if intended_notional >= max(min_order_notional, 1.0):
                # The pre-round intent was a real order; rounding/dust censored
                # it. Silent censoring is a reproducible book/target divergence.
                dust_dropped.append((str(symbol), intended_notional))
            continue
        orders.append(
            Order(
                client_order_id=f"{order_id_prefix}{decision_bar}:{symbol}",
                symbol=str(symbol),
                qty=delta,
                decision_bar=decision_bar,
                reference_price=price,
            )
        )
    if dust_dropped:
        shown = sorted(dust_dropped, key=lambda item: -item[1])[:20]
        tail = "…" if len(dust_dropped) > len(shown) else ""
        logger.warning(
            "%s: %d intended orders dropped by whole-share/notional rounding "
            "(book/target divergence, N7-loud): %s%s",
            decision_bar,
            len(dust_dropped),
            [(sym, round(notional, 2)) for sym, notional in shown],
            tail,
        )
    return orders


def decide_and_submit(
    ctx: LiveLoopContext,
    decision_bar: str,
    target_weights: pd.Series,
    prices: pd.Series,
    min_order_notional: float = 0.0,
    whole_shares: bool = False,
    refresh_bar: str | None = None,
    order_guard: Callable[[list[Order]], None] | None = None,
) -> list[Order]:
    """One decision step: reconcile → decide once → write ahead → submit.

    Re-running for the same ``decision_bar`` after a crash resumes the
    persisted decision idempotently. Calling for a new bar while a prior
    bar's orders are still pending raises — settle first, the book's true
    holdings are unknown until then. ``refresh_bar``, when provided, advances
    the persisted decision-cadence anchor (``last_refresh_bar``) inside the same
    write-ahead save; the caller passes it only on a refresh session.
    ``order_guard`` is the safety seam (``prism.live.safety``): it sees the
    freshly diffed order list *before* the write-ahead persist, so a raised
    :class:`~prism.live.safety.SafetyViolation` leaves no pending state and
    nothing at the venue. A resumed (already-persisted) decision is not
    re-guarded — it was vetted when decided, and re-deciding is the defect
    the protocol exists to prevent.
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
        orders = targets_to_orders(
            target_weights,
            state.positions,
            prices,
            equity,
            decision_bar,
            min_order_notional=min_order_notional,
            whole_shares=whole_shares,
            order_id_prefix=ctx.order_id_prefix,
        )
        if order_guard is not None:
            order_guard(orders)  # SafetyViolation propagates BEFORE the write-ahead save
        state.pending_orders = orders
        state.pending_decision_bar = decision_bar
        if refresh_bar is not None:
            state.last_refresh_bar = refresh_bar  # advance the cadence anchor atomically
        ctx.store.save(state)  # the write-ahead: persisted before any submit
        if refresh_bar is not None and ctx.targets_ledger is not None:
            # Persist the constructed book at each refresh — the concordance
            # baseline (what the instrument SHOULD hold until the next refresh)
            # and the audit record behind the completion sweep. Written inside
            # the write-ahead step, idempotent per refresh bar.
            _append_targets_ledger(
                ctx.targets_ledger, refresh_bar, decision_bar, target_weights, prices, equity
            )

    _submit_pending(ctx.broker, state.pending_orders)
    return list(state.pending_orders)


def _submit_pending(broker: Broker, orders: list[Order]) -> None:
    """Idempotent submission of a persisted order set (decide and sweep share it)."""
    known = broker.submitted_order_ids()
    for order in orders:
        if order.client_order_id in known:
            continue
        try:
            broker.submit(order)
        except DuplicateOrder:
            logger.info("order %s already at venue; treating as submitted", order.client_order_id)
        except OrderRejected as exc:
            # The venue rejected this one order (a side-crossing residual, a
            # non-shortable name, buying power). It must not crash the cycle and
            # wedge the loop: log loudly and submit the rest. The order stays in
            # the persisted pending set, so a same-bar resume retries it and
            # settle tolerates a missing fill by re-anchoring to broker truth. A
            # transport failure / 5xx is NOT caught here — it propagates so the
            # write-ahead crash-resume applies (N7 loud, not silent-drop).
            logger.error(
                "order %s rejected by the venue: %s; skipping it, submitting the rest",
                order.client_order_id,
                exc,
            )


def settle(ctx: LiveLoopContext, settle_bar: str) -> list[Fill]:
    """Pull fills for the pending decision, ledger them, re-anchor state.

    Ledgers every executed quantity (whole fills *and* partials under an
    expired/canceled parent) beside its decision-close reference price, then
    re-anchors positions and cash to **broker truth** and clears pending.

    An order that did not fill is *not* an emergency to halt on: a
    market-on-open order whose auction did not print, or whose remainder
    expired, is a routine venue outcome for the R2 paper instrument, and the
    book is still known exactly because we adopt broker truth. So settle does
    not raise on unfilled orders — it fails **loud, not silent** (N7): the
    unfilled ids are logged (WARNING, or ERROR when *nothing* filled, which
    can flag a systemic problem — auth, market closed, venue down), while the
    loop stays alive to decide the next bar. The book cannot silently drift
    because ``positions``/``cash`` come from the broker, not from assuming the
    decision executed. (The prior all-or-nothing raise made a single expired
    OPG leg wedge the whole loop until a manual state edit — the exact stall
    this instrument is meant to survive.)
    """
    state = ctx.store.load()
    if state is None or not state.pending_orders:
        return []

    pending_ids = {o.client_order_id for o in state.pending_orders}
    fills = ctx.broker.fills_for(pending_ids)
    filled_ids = {f.client_order_id for f in fills}
    unfilled = pending_ids - filled_ids
    if unfilled:
        detail = sorted(unfilled)
        shown = detail[:8]
        tail = "…" if len(detail) > len(shown) else ""
        level = logging.ERROR if not fills else logging.WARNING
        logger.log(
            level,
            "settle %s: %d/%d pending orders did not fill (%s%s); ledgering the %d that did "
            "and re-anchoring the book to broker truth (N7: loud, not silent-zero)",
            state.pending_decision_bar,
            len(unfilled),
            len(pending_ids),
            shown,
            tail,
            len(fills),
        )

    # Record every unexecuted residual durably (unfilled and partially filled
    # orders alike). The truncated WARNING above is for the operator's eye; the
    # ledger is for calibration/repair — the 79-name unfilled set of 2026-07-08
    # was unrecoverable locally because only the first 8 ids were ever logged.
    if ctx.unfilled_ledger is not None:
        executed = {f.client_order_id: f.qty for f in fills}
        residual_rows = []
        for order in state.pending_orders:
            residual = order.qty - executed.get(order.client_order_id, 0.0)
            if residual == 0.0:
                continue
            residual_rows.append(
                {
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "order_qty": order.qty,
                    "executed_qty": executed.get(order.client_order_id, 0.0),
                    "residual_qty": residual,
                    "reference_price": order.reference_price,
                    "decision_bar": state.pending_decision_bar,
                    "settle_bar": settle_bar,
                }
            )
        if residual_rows:
            _append_jsonl(ctx.unfilled_ledger, residual_rows)

    reference = {o.client_order_id: o.reference_price for o in state.pending_orders}
    if fills:
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


def sweep_pending(ctx: LiveLoopContext, *, sweep_suffix: str = "S1") -> list[Order]:
    """Morning completion sweep: re-submit unexecuted residuals as new orders.

    The Alpaca *paper* venue's opening-auction simulation prints only ~20-25%
    of OPG orders (observed across both instrument books), so a monthly
    refresh left standing would hold a fraction of the decided book for a
    whole cadence period — the paper stream then measures a different
    portfolio than the one the decision constructed. The sweep runs after the
    open, while the decision is still pending (settle happens at the next
    evening cycle): for every pending order that is *terminal* at the venue
    (filled, partially filled, expired, rejected, or never submitted) it
    computes the unexecuted residual and submits it as a fresh order under a
    deterministic suffixed id (``"{original_id}:{sweep_suffix}"``), appended
    to the persisted pending set *before* submission (the same write-ahead
    discipline as the decision itself). The evening settle then treats sweep
    fills exactly like auction fills — same decision-close ``reference_price``,
    so the fills ledger records the *total* arrival cost of executing the
    decision at this venue; the id suffix keeps the two fill populations
    segmentable for I-9 calibration.

    Safety properties:

    * An order still *live* at the venue is never swept (double-execution);
      the broker must expose ``open_order_ids`` (duck-typed) or the sweep
      refuses to run.
    * Idempotent: rerunning regenerates the same suffixed ids — existing ones
      are resumed/skipped through the venue's duplicate-id rejection, never
      duplicated in the persisted pending set.
    * One completion generation per decision: a sweep order's own unfilled
      residual is logged at settle (unfilled ledger) but not re-swept.
    * A residual whose sign disagrees with its parent order (venue over-fill,
      which should not happen) is skipped loudly, never amplified.

    The venue order type is whatever the context's broker was constructed
    with — the sweep CLI wires ``time_in_force="day"`` (a plain market order
    after the open) precisely because the auction the OPG order was for has
    already happened.
    """
    state = ctx.store.load()
    if state is None or not state.pending_orders:
        return []
    open_ids_fn = getattr(ctx.broker, "open_order_ids", None)
    if open_ids_fn is None:
        raise TypeError(
            "broker does not expose open_order_ids(client_order_ids); refusing to sweep "
            "blind — an order still queued for its auction would double-execute (N7)"
        )
    pending_ids = {o.client_order_id for o in state.pending_orders}
    open_ids = set(open_ids_fn(pending_ids))
    executed: dict[str, float] = {}
    for fill in ctx.broker.fills_for(pending_ids):
        executed[fill.client_order_id] = executed.get(fill.client_order_id, 0.0) + fill.qty

    existing_ids = set(pending_ids)
    parents = [o for o in state.pending_orders if not o.client_order_id.endswith(f":{sweep_suffix}")]
    new_orders: list[Order] = []
    skipped_open = 0
    for order in parents:
        child_id = f"{order.client_order_id}:{sweep_suffix}"
        if order.client_order_id in open_ids or child_id in open_ids:
            skipped_open += 1
            continue
        done = executed.get(order.client_order_id, 0.0) + executed.get(child_id, 0.0)
        residual = order.qty - done
        if residual == 0.0 or child_id in existing_ids:
            continue
        if residual * order.qty < 0:
            logger.warning(
                "sweep: residual %s for %s exceeds the order (%s executed vs %s ordered); "
                "skipping, never amplifying a venue over-fill",
                residual,
                order.client_order_id,
                done,
                order.qty,
            )
            continue
        new_orders.append(
            Order(
                client_order_id=child_id,
                symbol=order.symbol,
                qty=residual,
                decision_bar=order.decision_bar,
                reference_price=order.reference_price,
            )
        )
    if skipped_open:
        logger.warning(
            "sweep %s: %d orders still live at the venue — not swept this pass "
            "(auction pending or open order); rerun after they turn terminal",
            state.pending_decision_bar,
            skipped_open,
        )
    if new_orders:
        state.pending_orders = list(state.pending_orders) + new_orders
        ctx.store.save(state)  # write-ahead: the sweep decision persists before any submit
        logger.info(
            "sweep %s: re-submitting %d unexecuted residuals as %s orders",
            state.pending_decision_bar,
            len(new_orders),
            sweep_suffix,
        )
    # Resume/submit every sweep order not yet at the venue (crash-safe rerun).
    sweep_orders = [
        o for o in state.pending_orders if o.client_order_id.endswith(f":{sweep_suffix}")
    ]
    _submit_pending(ctx.broker, sweep_orders)
    return new_orders


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


def read_targets_ledger(path: Path) -> list[dict]:
    """The refresh-target ledger rows, oldest first (concordance baseline)."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_concordance_ledger(path: Path) -> pd.DataFrame:
    """The book-concordance telemetry ledger as a frame."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)


def read_regime_ledger(path: Path) -> pd.DataFrame:
    """The per-cycle regime telemetry ledger as a frame — one row per decision
    bar (``decision_bar``, ``clean``, ``gross_scale``, ``blocks``,
    ``failures``, …). The handoff §8 precondition-(b) 21-session clock reads
    ``clean`` (docs/regime_step.md)."""
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


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _append_fills_ledger(path: Path, rows: list[dict]) -> None:
    _append_jsonl(path, rows)


def _append_targets_ledger(
    path: Path,
    refresh_bar: str,
    decision_bar: str,
    target_weights: pd.Series,
    prices: pd.Series,
    equity: float,
) -> None:
    """Persist the constructed refresh book (nonzero weights + their decision
    closes). Idempotent per ``refresh_bar`` so a same-bar restart never
    duplicates the row; absent names read as an explicit 0.0 downstream."""
    path = Path(path)
    if path.exists():
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines and refresh_bar <= json.loads(lines[-1])["refresh_bar"]:
            return
    book = {
        str(symbol): float(weight)
        for symbol, weight in target_weights.items()
        if isinstance(weight, (int, float))
        and math.isfinite(float(weight))
        and float(weight) != 0.0
    }
    reference_prices = {symbol: float(prices[symbol]) for symbol in book}
    _append_jsonl(
        path,
        [
            {
                "refresh_bar": refresh_bar,
                "decision_bar": decision_bar,
                "equity": float(equity),
                "targets": book,
                "reference_prices": reference_prices,
            }
        ],
    )


def _append_concordance_ledger(path: Path, decision_bar: str, row: dict) -> None:
    """Append one concordance read per decision bar (idempotent, monotone —
    the same discipline as the equity ledger)."""
    path = Path(path)
    if path.exists():
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines and decision_bar <= json.loads(lines[-1])["decision_bar"]:
            return
    _append_jsonl(path, [{"decision_bar": decision_bar, **row}])


def _append_regime_ledger(path: Path, decision_bar: str, state: dict, gross_scale: float | None) -> None:
    """Append one regime telemetry read per decision bar (idempotent, monotone
    — the equity-ledger discipline, so a same-bar restart never duplicates a
    session on the precondition-(b) clock). ``clean`` is the session verdict:
    True iff the read carries zero failure entries. ``gross_scale`` is the
    clamped multiplier the §7.7 action hook applied to this refresh's
    construction — null while the hook is unarmed
    (docs/sizing_preregistration.md unratified) or the session did not
    qualify (not a refresh, or a not-clean read)."""
    path = Path(path)
    if path.exists():
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines and decision_bar <= json.loads(lines[-1])["decision_bar"]:
            return
    row = {
        "decision_bar": decision_bar,
        "clean": not state.get("failures"),
        "gross_scale": gross_scale,
        **{key: value for key, value in state.items() if key != "decision_bar"},
    }
    _append_jsonl(path, [row])


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
