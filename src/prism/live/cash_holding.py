"""Idle equity held in a T-bill ETF beside the long/short book, off by default.

``prism.live.daily.run_daily_cycle(cash_holding=CashHolding("SGOV"))`` holds the
equity the stock long side does not use, less a cash buffer, in a T-bill ETF:

    target = max(0, 1 - buffer - L)

where ``L`` is the stock long side over equity in the account target this cycle
decided, after the caps, the band and the participation gate. Short-sale
proceeds are collateral and are not invested: only own equity the long side
leaves unused is idle (SPEC §3). The ETF is a long position, so its total
return is in the account's return, and only the buffer is idle cash.

**When it trades.** On every session the cycle decides (a refresh or not; not
under a halt), the holding is compared with its target. It is left alone while
it is within ``band`` of the target, and otherwise moved to the target, so it
trades on drift, not on every small change in the long side. A target of zero
is the exception: a holding above zero is then sold even inside the band,
because with no idle equity any ETF held is financed by a margin debit. Under
``wind_down`` the target is zero, so the holding is sold along with the book.

**Order size.** One session's change is bounded by the per-order notional bound
(``prism.live.safety.SafetyConfig.max_order_fraction``, 2 × the per-name cap =
20% of equity by default) less one share, so the order it produces passes that
guard after whole-share rounding, and a holding of 44% of equity is built over
three sessions. Without a bound the change is not limited.

**Not scored.** The ETF is priced, never ranked: the caller fetches it as an
unscored name (``resolve_fetch_universe(..., unscored=...)``) and keeps it out
of the scored cross-section (the momentum book's membership mask). The cycle
refuses a holding whose symbol it cannot price at the decision close.

Production-import-path safe (N8): pandas only.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

logger = logging.getLogger(__name__)

#: The default cash buffer: equity kept as cash, beside the ETF, for fills that
#: differ from their decision prices and for the debits of a session.
DEFAULT_BUFFER = 0.02

#: The default rebalancing band, in weight units (fractions of equity).
DEFAULT_BAND = 0.02

#: The book the holding is implemented for: the decile long/short book, whose
#: CLI path keeps the ETF out of the scored cross-section.
CASH_HOLDING_BOOK = "decile_neutral"

#: The key the refresh row of ``targets.jsonl`` carries the holding's decision under.
TARGETS_LEDGER_KEY = "cash_holding"


class CashHoldingError(ValueError):
    """The T-bill holding cannot be decided on these inputs (N7)."""


@dataclass(frozen=True)
class CashHolding:
    """Which ETF holds idle equity, the cash buffer beside it, and its rebalancing band."""

    symbol: str
    buffer: float = DEFAULT_BUFFER
    band: float = DEFAULT_BAND

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise CashHoldingError("the cash holding needs a symbol")
        object.__setattr__(self, "symbol", symbol)
        for name in ("buffer", "band"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise CashHoldingError(f"cash holding {name} must be a finite number, got {value!r}")
            object.__setattr__(self, name, float(value))
        if not 0.0 <= self.buffer <= 1.0:
            raise CashHoldingError(f"cash holding buffer must be in [0, 1], got {self.buffer}")
        if not 0.0 <= self.band < 1.0:
            raise CashHoldingError(f"cash holding band must be in [0, 1), got {self.band}")


@dataclass(frozen=True)
class CashHoldingDecision:
    """One session's decision for the holding, in fractions of equity.

    ``long_weight`` is the stock long side the target was computed from,
    ``target_weight`` the formula's value (zero under a wind-down), ``weight``
    what the account target carries, ``step_bound`` the largest change one
    session may make (None without a per-order bound). ``action`` is ``hold``
    (within the band), ``rebalance`` (moved to the target) or ``bounded``
    (moved toward it by ``step_bound``).
    """

    symbol: str
    long_weight: float
    held_weight: float
    target_weight: float
    weight: float
    step_bound: float | None
    action: str
    wind_down: bool
    buffer: float
    band: float

    def to_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "long_weight": self.long_weight,
            "held_weight": self.held_weight,
            "target_weight": self.target_weight,
            "weight": self.weight,
            "step_bound": self.step_bound,
            "action": self.action,
            "wind_down": self.wind_down,
            "buffer": self.buffer,
            "band": self.band,
        }


def holding_target(long_weight: float, buffer: float) -> float:
    """``max(0, 1 - buffer - long_weight)``: the idle own equity above the buffer."""
    if not math.isfinite(long_weight) or long_weight < 0.0:
        raise CashHoldingError(f"the stock long side must be finite and >= 0, got {long_weight!r}")
    return max(0.0, 1.0 - float(buffer) - float(long_weight))


def stock_long_weight(targets: pd.Series, held_weights: pd.Series, *, exclude: str) -> float:
    """The long side the account target leaves the stocks, as a fraction of equity.

    A finite target is the name's weight. A NaN target leaves the position as
    held (``prism.assembly.size_orders``), and so does a held name the target
    does not list, so both count at their held weight. ``exclude`` is the
    holding's own symbol. Short weights do not count: short proceeds are
    collateral, not idle equity.
    """
    total = 0.0
    names = dict.fromkeys([*(str(s) for s in targets.index), *(str(s) for s in held_weights.index)])
    for name in names:
        if name == exclude:
            continue
        weight = float("nan")
        if name in targets.index:
            weight = float(targets[name])
        if not math.isfinite(weight):
            weight = float(held_weights.get(name, 0.0))
        if not math.isfinite(weight):
            raise CashHoldingError(f"{name}: no finite target or held weight to count in the long side (N7)")
        total += max(0.0, weight)
    return total


def apply_cash_holding(
    holding: CashHolding,
    targets: pd.Series,
    held_weights: pd.Series,
    *,
    prices: Mapping[str, float] | pd.Series,
    equity: float,
    max_order_fraction: float | None,
    wind_down: bool = False,
) -> tuple[pd.Series, CashHoldingDecision]:
    """The account target with the holding's weight set, and the decision that set it.

    ``targets`` is the stock book this cycle decided (on a session that does
    not refresh, the held weights); ``held_weights`` the marked book, both as
    fractions of ``equity``. ``prices`` are the decision closes. The input
    series is not modified.
    """
    symbol = holding.symbol
    if not (math.isfinite(equity) and equity > 0.0):
        raise CashHoldingError(f"equity must be finite and > 0 to size the cash holding, got {equity!r}")
    price = prices.get(symbol)
    if price is None or not math.isfinite(float(price)) or float(price) <= 0.0:
        raise CashHoldingError(
            f"no usable decision close for the cash holding {symbol!r} (got {price!r}); fetch it as a priced, "
            "unscored name (N7)"
        )
    held = float(held_weights.get(symbol, 0.0))
    if not math.isfinite(held):
        raise CashHoldingError(f"the held weight of {symbol!r} is not finite ({held!r}) (N7)")
    long_weight = stock_long_weight(targets, held_weights, exclude=symbol)
    target = 0.0 if wind_down else holding_target(long_weight, holding.buffer)

    step_bound: float | None = None
    if max_order_fraction is not None:
        # One share of slack: sizing rounds the share delta to the nearest whole
        # share, so the order can carry up to half a share more than the weight.
        step_bound = float(max_order_fraction) - float(price) / float(equity)
        if not step_bound > 0.0:
            raise CashHoldingError(
                f"one share of {symbol} ({float(price):.2f}) is more than the per-order bound "
                f"{float(max_order_fraction):.2%} of equity {float(equity):.2f}: the holding cannot be traded "
                "inside it (N7)"
            )

    change = target - held
    if abs(change) <= holding.band and not (target == 0.0 and held > 0.0):
        weight, action = held, "hold"
    elif step_bound is not None and abs(change) > step_bound:
        weight, action = held + math.copysign(step_bound, change), "bounded"
    else:
        weight, action = target, "rebalance"

    decided = targets.copy()
    decided.loc[symbol] = weight
    decision = CashHoldingDecision(
        symbol=symbol,
        long_weight=long_weight,
        held_weight=held,
        target_weight=target,
        weight=weight,
        step_bound=step_bound,
        action=action,
        wind_down=bool(wind_down),
        buffer=holding.buffer,
        band=holding.band,
    )
    return decided, decision


def require_cash_holding_inputs(holding: CashHolding, *, book: str, close: pd.DataFrame) -> None:
    """Refuse a holding the cycle cannot run, before any state moves (N7).

    The decile book only, and the ETF must be a column of the panel with a
    usable close on the decision bar.
    """
    if book != CASH_HOLDING_BOOK:
        raise CashHoldingError(
            f"the cash holding is implemented for the {CASH_HOLDING_BOOK!r} book only (got {book!r}): that book's "
            "CLI path keeps the ETF out of the scored cross-section"
        )
    if holding.symbol not in close.columns:
        raise CashHoldingError(
            f"the cash holding {holding.symbol!r} is not in the bar panel: fetch it as a priced, unscored name (N7)"
        )
    last = close[holding.symbol].iloc[-1]
    if not (math.isfinite(float(last)) and float(last) > 0.0):
        raise CashHoldingError(
            f"the cash holding {holding.symbol!r} has no usable close on the decision bar (got {last!r}) (N7)"
        )


def log_decision(decision: CashHoldingDecision, decision_bar: str) -> None:
    """One line per session; a bounded step is INFO too, since building over sessions is the design."""
    logger.info(
        "cash holding %s at %s: %s, held %.4f -> %.4f of equity (target %.4f = max(0, 1 - %.2f buffer - %.4f long "
        "side)%s%s)",
        decision.symbol,
        decision_bar,
        decision.action,
        decision.held_weight,
        decision.weight,
        decision.target_weight,
        decision.buffer,
        decision.long_weight,
        "" if decision.step_bound is None else f", step bound {decision.step_bound:.4f}",
        ", WIND-DOWN: target zero" if decision.wind_down else "",
    )


__all__ = [
    "CASH_HOLDING_BOOK",
    "CashHolding",
    "CashHoldingDecision",
    "CashHoldingError",
    "DEFAULT_BAND",
    "DEFAULT_BUFFER",
    "TARGETS_LEDGER_KEY",
    "apply_cash_holding",
    "holding_target",
    "log_decision",
    "require_cash_holding_inputs",
    "stock_long_weight",
]
