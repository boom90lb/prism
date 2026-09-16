"""The fixed-share simulator: the canonical accounting contract driven over bars (spec §3.6).

What distinguishes this from ``execution/target_weights.py`` is the holding process, not
the arithmetic. The reproduction engine holds constant *weights* and silently re-levers to
them every bar; this engine holds **shares** and lets weights drift with prices between
decisions. On the four-bar panel of §3.6's worked timeline the two differ by 2,293.60 on
the same prices and the same single decision -- not because either mis-adds, but because
the constant-weight book traded to stay at its target and charged nothing for doing so.

One bar, in order:

1. events effective on the bar, and every claim whose pay date has arrived
   (:func:`prism.execution.accounting.apply_events`, applied before the open for the
   reason its module docstring gives);
2. fills at the open, for orders submitted at an earlier close, under the declared
   ``[fills]`` model;
3. remainders that cannot fill again expire per their time in force;
4. a mark at the open, for the record;
5. the close mark and that bar's holding-cost accrual;
6. the decision at the close, sized in shares, and its submission.

The ``[fills]`` block is declared, never assumed, and two rules in it are absolute: a
simulated fill is never created for a bar that did not trade, and never at a price the bar
did not print. Everything else -- participation caps, partial fractions, rejections -- is a
declared value whose reference setting is the trivial one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.execution.accounting import (
    AppliedEvent,
    DecisionBatch,
    apply_events,
    apply_fills,
    decide,
    expire_orders,
    mark_at_close,
    mark_prices,
    round_to_increment,
)
from prism.execution.cost_model import COST_TERMS, CostBreakdown, CostModel
from prism.execution.portfolio_state import (
    AccountingError,
    Fill,
    OrderRecord,
    PortfolioState,
    bar_key,
    empty_state,
)
from prism.io.events import Adjustment, Event

__all__ = [
    "DecisionCall",
    "FillPolicy",
    "SimulationResult",
    "simulate",
]

FILL_MODELS = ("full_next_open", "participation_capped")


@dataclass(frozen=True)
class FillPolicy:
    """The declared ``[fills]`` block of a simulating execution profile (§3.6, §3.9).

    A live broker profile carries no ``[fills]`` block at all, because the venue supplies
    the fills; this object exists only for simulation. ``partial_fill_fraction`` and
    ``reject_probability`` are there so the partial-fill and rejection paths are exercised
    against deterministic fixtures rather than only in production, which is why the
    simulated rejection draw is a hash of the order id and not a random stream: the same
    run rejects the same orders every time, on any machine.
    """

    model: Literal["full_next_open", "participation_capped"] = "full_next_open"
    max_participation_of_bar_volume: float = 0.0
    partial_fill_fraction: float = 1.0
    reject_probability: float = 0.0
    missing_open_policy: Literal["reference_price"] = "reference_price"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.model not in FILL_MODELS:
            raise AccountingError(f"fill model must be one of {FILL_MODELS}, got {self.model!r}")
        if self.model == "participation_capped" and not self.max_participation_of_bar_volume > 0.0:
            raise AccountingError(
                "fills.model = 'participation_capped' needs max_participation_of_bar_volume > 0; a cap of zero "
                "fills nothing, ever, which is a silent no-trade rather than a declared policy"
            )
        if not 0.0 < float(self.partial_fill_fraction) <= 1.0:
            raise AccountingError(f"partial_fill_fraction must be in (0, 1], got {self.partial_fill_fraction!r}")
        if not 0.0 <= float(self.reject_probability) < 1.0:
            raise AccountingError(f"reject_probability must be in [0, 1), got {self.reject_probability!r}")
        if self.missing_open_policy != "reference_price":
            raise AccountingError(
                f"missing_open_policy has one supported value, 'reference_price', got {self.missing_open_policy!r}"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FillPolicy":
        """Parse the ``[fills]`` table of a profile file."""
        known = {
            "model",
            "max_participation_of_bar_volume",
            "partial_fill_fraction",
            "reject_probability",
            "missing_open_policy",
            "seed",
        }
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise AccountingError(f"the [fills] block declares unknown keys {unknown}")
        return cls(**{k: v for k, v in mapping.items()})

    def rejects(self, client_order_id: str, bar: str) -> bool:
        """Whether this order is rejected at this bar, deterministically."""
        if self.reject_probability <= 0.0:
            return False
        digest = hashlib.sha256(f"{self.seed}:{bar}:{client_order_id}".encode("utf-8")).hexdigest()[:16]
        return int(digest, 16) / float(1 << 64) < self.reject_probability


@dataclass(frozen=True)
class DecisionCall:
    """What the simulator hands a state-coupled decision at step 6 (§3.8).

    The hook is called at the close of every bar, after that bar's events, fills,
    expiries, marks and holding-cost accrual, and before anything is submitted. It sees
    the account as it then stands -- ``equity`` is the marked equity the decision would be
    sized against, and ``state`` is a copy, so a hook that mutates what it is given
    cannot reach the accounting.

    Returning ``None`` is "no decision on this bar", which holds every position, exactly
    as an all-``NaN`` row of ``targets`` does. Returning a series of target weights in
    fractions of marked equity is the decision; a ``NaN`` cell holds that name.

    This is the seam a constructor whose inputs depend on the account needs: the
    participation bound is trade dollars over dollar volume *from current equity*, batch
    rounding is at the per-sid increment against current equity, and neither can be
    evaluated by a pass that produces every target before the first fill (§3.7, §3.8).
    """

    bar: str
    state: PortfolioState
    equity: float
    prices: pd.Series
    traded: frozenset[str]


@dataclass(frozen=True)
class SimulationResult:
    """Everything one run recorded. Every frame is indexed by bar, in bar order."""

    state: PortfolioState
    ledger: pd.DataFrame
    fills: pd.DataFrame
    costs: pd.DataFrame
    events: pd.DataFrame
    shares: pd.DataFrame
    marks: pd.DataFrame
    weights: pd.DataFrame
    orders: tuple[OrderRecord, ...] = ()
    decisions: tuple[DecisionBatch, ...] = ()

    @property
    def equity(self) -> pd.Series:
        """Marked equity at each close."""
        return self.ledger["equity_close"].rename("equity")

    @property
    def cash(self) -> pd.Series:
        return self.ledger["cash"].rename("cash")


def _row(frame: pd.DataFrame | None, bar: str) -> pd.Series | None:
    if frame is None or bar not in frame.index:
        return None
    row = frame.loc[bar]
    if isinstance(row, pd.DataFrame):
        raise AccountingError(f"the bar panel carries {bar} more than once")
    return pd.to_numeric(row, errors="coerce").astype(float)


def _traded_sids(open_row: pd.Series | None, close_row: pd.Series | None, volume_row: pd.Series | None) -> set[str]:
    """The names that printed on this bar.

    A name trades if it printed a price or a volume. Everything else -- a halt, a bar the
    vendor never delivered, a name withdrawn by a null revision -- did not trade, and the
    simulator creates no fill on it at any price.
    """
    traded: set[str] = set()
    for row in (open_row, close_row):
        if row is None:
            continue
        finite = row[np.isfinite(row.to_numpy(dtype=float)) & (row.to_numpy(dtype=float) > 0.0)]
        traded.update(str(sid) for sid in finite.index)
    if volume_row is not None:
        values = volume_row.to_numpy(dtype=float)
        traded.update(str(sid) for sid in volume_row.index[np.isfinite(values) & (values > 0.0)])
    return traded


def simulate(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    targets: pd.DataFrame | None = None,
    *,
    cost_model: CostModel,
    volumes: pd.DataFrame | None = None,
    events: Sequence[Event] | pd.DataFrame | None = (),
    adjustments: Sequence[Adjustment] | pd.DataFrame | None = (),
    price_basis: str | Mapping[str, str] = "split_adjusted_at_ingest",
    initial_cash: float = 0.0,
    initial_state: PortfolioState | None = None,
    fills: FillPolicy | None = None,
    increment: float | Mapping[str, float] = 1.0,
    min_notional: float = 0.0,
    order_type: Literal["market_on_open", "market_day"] = "market_on_open",
    zero_crossing_clamp: bool = False,
    bars: Sequence[str] | None = None,
    decide_hook: "Callable[[DecisionCall], pd.Series | None] | None" = None,
) -> SimulationResult:
    """Run the canonical fixed-share accounting over a bar panel.

    ``opens`` and ``closes`` are ``bar x sid`` frames of as-traded prices -- the shape
    ``prism.io.dataset.BarPanels.fields`` delivers. ``targets`` is a ``bar x sid`` frame of
    close-time decisions in fractions of marked equity; an all-``NaN`` row is the declared
    no-decision marker, and a ``NaN`` cell holds that name's position. A decision at the
    close of bar ``t`` can fill no earlier than the open of ``t+1``, in backtest and live
    alike.

    ``adjustments`` is the §3.2 companion table; the simulator reads its ``rebase`` rows so
    that a vendor back-rewrite restates the share count on the same bar it restates the
    prices, and equity is continuous across the rewrite (§3.6). Omitting it on a panel that
    carries a rewrite is how a holding loses ``1 - 1/factor`` of its value silently.

    ``cost_model`` must be built from a declared :class:`~prism.execution.cost_model.CostStack`.
    There is deliberately no "no costs" default: a zero charge is only ever the result of a
    declared zero, because a zero nobody declared is indistinguishable in the record from
    one that was.

    ``decide_hook`` is the state-coupled alternative to ``targets``: instead of reading a
    row that was decided before any fill, the simulator calls the hook at the close of
    each bar with the account as it then stands (:class:`DecisionCall`) and takes the row
    it returns. Exactly one of the two may be given -- a run that carried both would have
    two answers on the same bar with nothing in the record saying which was traded.
    """
    if targets is not None and decide_hook is not None:
        raise AccountingError(
            "simulate takes either a decided target panel or a decide_hook, never both: a bar that carried "
            "both would have two decisions and nothing in the record would say which one traded"
        )
    if cost_model.stack is None:
        raise AccountingError(
            "the canonical simulator charges dollars, so it needs a CostModel built from a declared CostStack "
            "(§3.6). Declare the stack, including its zeros, rather than passing the reproduction engine's "
            "ExecutionConfig"
        )
    policy = fills or FillPolicy()
    opens = opens.copy()
    closes = closes.copy()
    opens.index = [bar_key(b) for b in opens.index]
    closes.index = [bar_key(b) for b in closes.index]
    if volumes is not None:
        volumes = volumes.copy()
        volumes.index = [bar_key(b) for b in volumes.index]
    if targets is not None:
        targets = targets.copy()
        targets.index = [bar_key(b) for b in targets.index]

    index = [bar_key(b) for b in (bars if bars is not None else closes.index)]
    if len(set(index)) != len(index):
        raise AccountingError("the bar index carries a duplicate bar")
    if list(index) != sorted(index):
        raise AccountingError("bars must be in ascending order; the accounting is sequential")

    event_rows = list(events) if not isinstance(events, pd.DataFrame) else events
    adjustment_rows = list(adjustments) if not isinstance(adjustments, pd.DataFrame) else adjustments
    state = initial_state.copy() if initial_state is not None else empty_state(index[0], cash=initial_cash)

    ledger_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    event_records: list[AppliedEvent] = []
    share_rows: dict[str, dict[str, float]] = {}
    mark_rows: dict[str, dict[str, float]] = {}
    weight_rows: dict[str, dict[str, float]] = {}
    orders: list[OrderRecord] = []
    decisions: list[DecisionBatch] = []
    decision_equity: dict[str, float] = {}

    for bar in index:
        open_row = _row(opens, bar)
        close_row = _row(closes, bar)
        volume_row = _row(volumes, bar)
        traded = _traded_sids(open_row, close_row, volume_row)

        # 1. events effective on this bar, before any fill on it.
        outcome = apply_events(
            state,
            bar=bar,
            events=event_rows,
            adjustments=adjustment_rows,
            price_basis=price_basis,
            prices=None,
        )
        state = outcome.state
        event_records.extend(outcome.applied)
        event_cash = float(sum(e.cash_delta for e in outcome.applied))
        event_equity = float(sum(e.equity_delta for e in outcome.applied))

        # 2. fills at the open.
        bar_fills, rejected, trade_cost = _fills_at_open(
            state,
            bar=bar,
            open_row=open_row,
            volume_row=volume_row,
            traded=traded,
            policy=policy,
            cost_model=cost_model,
            decision_equity=decision_equity,
        )
        if rejected:
            state, recorded = _reject(state, rejected, bar)
            orders.extend(recorded)
        if bar_fills:
            applied = apply_fills(state, bar_fills)
            state = applied.state
            orders.extend(applied.completed)
            for fill in applied.applied:
                fill_rows.append(_fill_row(fill))

        # 3. opg remainders expire at the open of a bar that traded.
        state, expired = expire_orders(state, bar=bar, phase="open", sids=traded)
        orders.extend(expired)

        # 4. a mark at the open, for the record, valued at the marks the state then
        #    carries: the open where the name printed one, the prior close where it did
        #    not. An absent open is not yet a missing bar -- the session has not ended --
        #    so nothing goes stale here and nothing is refused for want of a quote.
        state, _ = mark_prices(state, bar=bar, prices=open_row, stale_on_missing=False)
        equity_open = state.carried_equity()

        # 5. the close mark and this bar's holding-cost accrual.
        close_mark = mark_at_close(state, bar=bar, prices=close_row, cost_model=cost_model)
        state = close_mark.state
        equity_close = state.equity(close_row)

        # 6. day remainders expire at the close; then the decision.
        state, expired_day = expire_orders(state, bar=bar, phase="close", sids=traded)
        orders.extend(expired_day)

        row: pd.Series | None = None
        if decide_hook is not None:
            proposed = decide_hook(
                DecisionCall(
                    bar=bar,
                    state=state.copy(),
                    equity=equity_close,
                    prices=close_row if close_row is not None else pd.Series(dtype=float),
                    traded=frozenset(traded),
                )
            )
            if proposed is not None:
                row = pd.to_numeric(pd.Series(proposed, dtype=float), errors="coerce").astype(float)
        elif targets is not None and bar in targets.index:
            row = pd.to_numeric(targets.loc[bar], errors="coerce").astype(float)
        if row is not None:
            if not row.isna().all():
                batch = decide(
                    state,
                    row.dropna(),
                    close_row if close_row is not None else pd.Series(dtype=float),
                    decision_bar=bar,
                    increment=increment,
                    min_notional=min_notional,
                    order_type=order_type,
                    zero_crossing_clamp=zero_crossing_clamp,
                    equity=equity_close,
                )
                decisions.append(batch)
                state = _submit(state, batch, bar)
                for intent in batch.intents:
                    decision_equity[f"{bar}:{intent.sid}"] = equity_close

        share_rows[bar] = {sid: position.shares for sid, position in state.positions.items()}
        mark_rows[bar] = {sid: position.mark(close_row, sid) for sid, position in state.positions.items()}
        weight_rows[bar] = _safe_weights(state, close_row)
        accrual = close_mark.accrual
        cost_rows.append(
            {
                "bar": bar,
                **{term: getattr(trade_cost, term) + getattr(accrual, term) for term in COST_TERMS},
                "trade_total": trade_cost.trade_total,
                "holding_total": accrual.holding_total,
                "total": trade_cost.total + accrual.total,
            }
        )
        ledger_rows.append(
            {
                "bar": bar,
                "equity_open": equity_open,
                "equity_close": equity_close,
                "cash": state.cash,
                "receivables": state.receivables_total(),
                "liabilities": state.liabilities_total(),
                "positions_value": state.positions_value(close_row),
                "trade_cost": trade_cost.total,
                "holding_cost": accrual.holding_total,
                "settled_cash": close_mark.settled_cash,
                "event_cash": event_cash,
                "event_equity": event_equity,
                "n_fills": sum(1 for f in bar_fills),
                "n_rejected": len(rejected),
                "n_expired": len(expired) + len(expired_day),
                "stale": len(close_mark.stale),
                "incomplete": state.is_incomplete,
                "pending_orders": len(state.pending_orders),
            }
        )

    ledger = pd.DataFrame(ledger_rows).set_index("bar")
    costs = pd.DataFrame(cost_rows).set_index("bar")
    fills_frame = pd.DataFrame(
        fill_rows,
        columns=[
            "execution_id",
            "client_order_id",
            "sid",
            "ts",
            "bar",
            "qty",
            "fill_price",
            "reference_price",
            "commission",
            "spread_dollars",
            "impact_dollars",
            "source",
            "cash_delta",
        ],
    )
    events_frame = pd.DataFrame(
        [e.to_row() for e in event_records],
        columns=["sid", "event_type", "effective_date", "cash_delta", "equity_delta", "detail", "source", "revision"],
    )
    return SimulationResult(
        state=state,
        ledger=ledger,
        fills=fills_frame,
        costs=costs,
        events=events_frame,
        shares=_panel(share_rows, index),
        marks=_panel(mark_rows, index),
        weights=_panel(weight_rows, index),
        orders=tuple(orders) + tuple(state.pending_orders),
        decisions=tuple(decisions),
    )


def _panel(rows: Mapping[str, Mapping[str, float]], index: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict({bar: dict(rows.get(bar, {})) for bar in index}, orient="index")
    frame.index.name = "bar"
    return frame.reindex(index=list(index)).sort_index(axis=1)


def _fill_row(fill: Fill) -> dict[str, Any]:
    return {
        "execution_id": fill.execution_id,
        "client_order_id": fill.client_order_id,
        "sid": fill.sid,
        "ts": fill.ts,
        "bar": fill.bar,
        "qty": fill.qty,
        "fill_price": fill.fill_price,
        "reference_price": fill.reference_price,
        "commission": fill.commission,
        "spread_dollars": fill.spread_dollars,
        "impact_dollars": fill.impact_dollars,
        "source": fill.source,
        "cash_delta": fill.cash_delta,
    }


def _safe_weights(state: PortfolioState, prices: pd.Series | None) -> dict[str, float]:
    if not state.positions:
        return {}
    try:
        return {str(sid): float(w) for sid, w in state.weights(prices).items()}
    except AccountingError:
        # A book with no positive equity has no fraction-of-equity representation. The
        # weights row is a report; the ledger row above it still states the dollars.
        return {}


def _submit(state: PortfolioState, batch: DecisionBatch, bar: str) -> PortfolioState:
    from prism.execution.accounting import submit

    if not batch.intents:
        return state
    return submit(state, batch.intents, submitted_at=bar)


def _reject(
    state: PortfolioState, rejected: Sequence[OrderRecord], bar: str
) -> tuple[PortfolioState, tuple[OrderRecord, ...]]:
    """Mark the simulated venue rejections and take them out of the pending set.

    Returns the records **from the new state**, not the ones handed in: transitions copy,
    so mutating the caller's objects would leave the recorded order set describing a state
    that no longer exists.
    """
    working = state.copy()
    gone = {order.client_order_id for order in rejected}
    out: list[OrderRecord] = []
    for order in working.pending_orders:
        if order.client_order_id in gone:
            order.status = "rejected"
            order.reason = f"simulated venue rejection at {bar}"
            order.last_update = bar
            out.append(order)
    working.pending_orders = [o for o in working.pending_orders if o.client_order_id not in gone]
    working.sequence += 1
    return working, tuple(out)


def _fills_at_open(
    state: PortfolioState,
    *,
    bar: str,
    open_row: pd.Series | None,
    volume_row: pd.Series | None,
    traded: set[str],
    policy: FillPolicy,
    cost_model: CostModel,
    decision_equity: Mapping[str, float],
) -> tuple[list[Fill], list[OrderRecord], CostBreakdown]:
    """Build this bar's simulated fills, and the orders the venue rejected instead.

    The two absolutes of §3.6 are enforced here and nowhere else: no fill is created for a
    name that did not trade on this bar, and no fill is priced at anything but a price the
    bar printed -- with the single declared exception of ``missing_open_policy =
    "reference_price"``, which prices a bar that traded but printed no open at the intent's
    own reference price rather than inventing one.
    """
    out: list[Fill] = []
    rejected: list[OrderRecord] = []
    charged = CostBreakdown()
    for order in state.open_orders():
        if bar_key(order.submitted_at or order.intent.decision_bar) >= bar:
            continue  # decided at this close or later: it cannot have filled at this open
        sid = order.sid
        if sid not in traded:
            continue  # the bar did not trade: the order stays pending, untouched
        if policy.rejects(order.client_order_id, bar):
            rejected.append(order)
            continue
        price = None
        if open_row is not None and sid in open_row.index:
            value = float(open_row.loc[sid])
            if np.isfinite(value) and value > 0.0:
                price = value
        if price is None:
            price = order.intent.reference_price  # missing_open_policy = "reference_price"
        remaining = order.remaining_qty
        eligible = abs(remaining)
        if policy.model == "participation_capped":
            if volume_row is None or sid not in volume_row.index:
                raise AccountingError(
                    f"{sid}: fills.model = 'participation_capped' prices the cap against bar volume, and no "
                    f"volume is supplied for {bar}. A missing volume never becomes an unbounded fill"
                )
            volume = float(volume_row.loc[sid])
            if not np.isfinite(volume) or volume <= 0.0:
                continue
            eligible = min(eligible, policy.max_participation_of_bar_volume * volume)
        eligible *= policy.partial_fill_fraction
        quantity = round_to_increment(eligible, order.intent.increment)
        quantity = min(quantity, abs(remaining))
        if quantity <= 0.0:
            continue
        signed = float(np.sign(remaining) * quantity)
        provisional = Fill(
            execution_id=_execution_id(order, bar),
            client_order_id=order.client_order_id,
            sid=sid,
            ts=bar,
            qty=signed,
            fill_price=price,
            reference_price=order.intent.reference_price,
            source="simulated",
        )
        # The quadratic term is portfolio relative, so it divides by the marked equity the
        # order was sized from. An order carried in from an earlier session -- a resumed
        # run, a state restored from disk -- has no recorded decision equity here, and the
        # book's current marked equity is the only honest stand-in; it is never the model's
        # construction-time equity, which would be a number from another run.
        equity = decision_equity.get(order.client_order_id)
        if equity is None:
            equity = state.carried_equity()
        model = cost_model.at_equity(equity)
        breakdown = model.charge_trade(
            [provisional], decided_notional={order.client_order_id: abs(order.intent.notional)}
        )
        charged = charged + breakdown
        out.append(
            Fill(
                execution_id=provisional.execution_id,
                client_order_id=order.client_order_id,
                sid=sid,
                ts=bar,
                qty=signed,
                fill_price=price,
                reference_price=order.intent.reference_price,
                commission=breakdown.commission,
                spread_dollars=breakdown.spread,
                impact_dollars=breakdown.impact_quadratic + breakdown.impact_participation,
                source="simulated",
            )
        )
    return out, rejected, charged


def _execution_id(order: OrderRecord, bar: str) -> str:
    """A deterministic simulated execution id: the fills-ledger key, and the idempotence key."""
    return f"sim:{order.client_order_id}:{bar}:{len(order.fills)}"
