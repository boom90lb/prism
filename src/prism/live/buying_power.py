"""Fit a decision's opening orders inside the venue's buying power, off by default.

``prism.live.loop.decide_and_submit(buying_power_check=True)`` reads the
venue's buying power once per new decision, after the orders are sized and
before they are authorized, guarded and persisted. When the orders that open or
extend a position need more than that, every opening order is scaled down by
the same factor, loudly, so the batch fits instead of having the venue reject
some of it at submission (where the rest of the batch stops, §3.7). Orders that
only close or reduce a position are never scaled, and their proceeds are not
credited: they are not available until the orders fill, which for next-open
orders is after the venue has checked the batch (``prism.live.capabilities``
credits no sale proceeds either). Unlike that finding, a buy that covers a held
short is not counted as using buying power, since it lowers the margin
requirement; whether the venue's order-entry check agrees is not verified.

**Units.** The venue states buying power in dollars of position it will let the
account open: for a 2x Reg T margin account, Alpaca's ``buying_power`` is twice
the excess of equity over the initial margin requirement. The 50% initial
requirement of the opening orders, half their notional, therefore fits exactly
when their notional fits within ``buying_power``, and that is the comparison
made here. For a 4x (pattern day trader) account Alpaca reports its intraday
buying power in the same field, which can exceed what the account may carry
overnight; that case is not detected here and needs the Reg T figure.

**Bit-identical when it does not bind.** A batch that fits is returned as the
same list, untouched. With the check off nothing is read.

Production-import-path safe (N8): standard library only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from prism.live.broker import Order

#: Fractional quantities are rounded down to this many decimals of a share
#: (Alpaca accepts up to nine), so a scaled fractional batch still fits.
FRACTIONAL_DECIMALS = 9


class BuyingPowerError(ValueError):
    """The buying-power check cannot run on these inputs (N7)."""


@dataclass(frozen=True)
class BuyingPowerFit:
    """What the check found and did.

    ``required`` is the notional of the opening orders before the fit,
    ``fitted`` after it, both at the orders' reference prices; ``factor`` the
    scale applied to every opening quantity (1.0 when the batch fits);
    ``scaled`` and ``dropped`` the client order ids it changed or removed.
    """

    buying_power: float
    required: float
    fitted: float
    factor: float
    binding: bool
    scaled: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()

    def to_row(self) -> dict[str, Any]:
        return {
            "buying_power": self.buying_power,
            "required": self.required,
            "fitted": self.fitted,
            "factor": self.factor,
            "binding": self.binding,
            "scaled": list(self.scaled),
            "dropped": list(self.dropped),
        }


def opening_shares(order: Order, held: float) -> float:
    """The part of ``order`` that opens or extends a position, in shares (>= 0).

    A buy opens what it buys beyond covering a held short; a sale opens what it
    sells beyond a held long.
    """
    qty = float(order.qty)
    if qty > 0.0:
        return max(0.0, qty - max(0.0, -float(held)))
    return max(0.0, -qty - max(0.0, float(held)))


def opening_notional(orders: Sequence[Order], positions: Mapping[str, float]) -> float:
    """Dollars of buying power the opening part of ``orders`` needs at their reference prices."""
    return sum(
        opening_shares(order, float(positions.get(order.symbol, 0.0))) * float(order.reference_price)
        for order in orders
    )


def fit_opening_orders(
    orders: list[Order],
    positions: Mapping[str, float],
    buying_power: float,
    *,
    whole_shares: bool,
    min_order_notional: float = 0.0,
) -> tuple[list[Order], BuyingPowerFit]:
    """``orders`` with every opening quantity scaled by one factor so the batch fits ``buying_power``.

    Scaled quantities are rounded down (whole shares, or nine decimals), so the
    fitted opening notional never exceeds ``buying_power``. An order whose scaled
    notional falls below ``min_order_notional`` (and any that scales to zero) is
    dropped, as sizing drops dust. A negative ``buying_power`` is read as zero:
    nothing may open. A non-finite one raises (N7).
    """
    available = float(buying_power)
    if not math.isfinite(available):
        raise BuyingPowerError(f"the venue reported a non-finite buying power ({buying_power!r}) (N7)")
    available = max(0.0, available)
    required = opening_notional(orders, positions)
    if required <= available:
        return orders, BuyingPowerFit(
            buying_power=available, required=required, fitted=required, factor=1.0, binding=False
        )
    factor = available / required
    floor_notional = max(float(min_order_notional), 1e-9)
    fitted: list[Order] = []
    scaled: list[str] = []
    dropped: list[str] = []
    for order in orders:
        held = float(positions.get(order.symbol, 0.0))
        opening = opening_shares(order, held)
        if opening == 0.0:
            fitted.append(order)
            continue
        closing = abs(float(order.qty)) - opening
        kept = opening * factor
        if whole_shares:
            kept = float(math.floor(kept))
        else:
            unit = 10**FRACTIONAL_DECIMALS
            kept = math.floor(kept * unit) / unit
        size = closing + kept
        if size <= 0.0 or size * float(order.reference_price) < floor_notional:
            dropped.append(order.client_order_id)
            continue
        fitted.append(replace(order, qty=math.copysign(size, float(order.qty))))
        scaled.append(order.client_order_id)
    return fitted, BuyingPowerFit(
        buying_power=available,
        required=required,
        fitted=opening_notional(fitted, positions),
        factor=factor,
        binding=True,
        scaled=tuple(scaled),
        dropped=tuple(dropped),
    )


__all__ = [
    "BuyingPowerError",
    "BuyingPowerFit",
    "FRACTIONAL_DECIMALS",
    "fit_opening_orders",
    "opening_notional",
    "opening_shares",
]
