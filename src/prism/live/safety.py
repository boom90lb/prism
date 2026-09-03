"""Live-loop safety rails (SPEC.md §7.7) — pre-commit vetoes on the daily cycle.

Two distinct rails, at two distinct points in the cycle, with two distinct
failure modes:

* **Halt rails** (kill switch, drawdown) answer "should this book trade at
  all today?" — checked after the mark step, *before* any decision is
  constructed or persisted. A halted cycle still settles yesterday's fills,
  still marks NAV into the equity ledger (history stays continuous through a
  halt), but skips score → construct → submit entirely and reports the
  reason. Halting is a state, not an exception: the operator asked for it
  (kill switch) or pre-registered it (drawdown bound), and the loop's job is
  to obey loudly, not crash.

* **Order guards** (per-order notional, order count) answer "does the order
  list the decision produced look like it came from this configuration?" —
  checked inside the write-ahead protocol *before* the pending set is
  persisted or anything reaches the venue. A violation here means equity,
  prices, or construction are corrupted (a fat-fingered config, a 100×-wrong
  equity mark), and the only safe response is :class:`SafetyViolation` —
  loud, pre-commit, nothing half-submitted. The bound worth using is derived,
  not free: under down-only caps and the flip-to-flat clamp
  (``targets_to_orders``), no legitimate single order can exceed
  ``max_symbol_abs_weight × equity`` in notional (entering at the cap or
  exiting a capped position are the extremes), so a guard at twice that is
  slack against rounding yet catches every order-of-magnitude corruption.

The rails protect the *account*, not the strategy: they never liquidate, never
re-decide, and never move a ratified statistic. Auto-liquidation is a policy
decision that belongs to an owner, not a rail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from prism.live.broker import Order
from prism.live.loop import read_equity_ledger

logger = logging.getLogger(__name__)


class SafetyViolation(RuntimeError):
    """An order set violated a pre-submission guard; nothing was persisted."""


@dataclass(frozen=True)
class SafetyConfig:
    """Rails for one book's daily cycle. ``None`` disables a rail.

    ``kill_switch`` is a file path: its *presence* halts trading (create it to
    stop the book, delete it to resume — no code, no state edits, works from
    any shell while the loop is dark). ``max_drawdown`` is the peak-to-current
    fraction of the equity ledger's NAV history beyond which the book stops
    initiating (0.25 = halt below 75% of peak). ``max_order_fraction`` bounds
    any single order's notional as a fraction of marked equity;
    ``max_orders`` bounds the count of orders one decision may produce.
    """

    kill_switch: Path | None = None
    max_drawdown: float | None = None
    max_order_fraction: float | None = None
    max_orders: int | None = None

    def __post_init__(self) -> None:
        if self.max_drawdown is not None and not 0.0 < self.max_drawdown < 1.0:
            raise ValueError(f"max_drawdown must be in (0, 1), got {self.max_drawdown}")
        if self.max_order_fraction is not None and not self.max_order_fraction > 0.0:
            raise ValueError(f"max_order_fraction must be > 0, got {self.max_order_fraction}")
        if self.max_orders is not None and self.max_orders < 1:
            raise ValueError(f"max_orders must be >= 1, got {self.max_orders}")


def halt_reason(config: SafetyConfig, equity: float, equity_ledger: Path | None) -> str | None:
    """Why this cycle must not trade, or ``None`` when it may.

    Kill switch wins over drawdown (the explicit instruction over the derived
    one). Drawdown is measured against the running NAV peak of the equity
    ledger including today's mark, so a halt is decided on exactly the series
    the monitor reads — no second bookkeeping. With no ledger configured the
    drawdown rail cannot fire (one mark is not a history).
    """
    if config.kill_switch is not None and Path(config.kill_switch).exists():
        return f"kill switch present at {config.kill_switch}"
    if config.max_drawdown is not None and equity_ledger is not None:
        history = read_equity_ledger(equity_ledger)
        peak = float(equity)
        if not history.empty:
            peak = max(peak, float(history["equity"].max()))
        drawdown = 1.0 - float(equity) / peak
        if drawdown > config.max_drawdown:
            return (
                f"drawdown {drawdown:.1%} exceeds max_drawdown {config.max_drawdown:.1%} "
                f"(equity {equity:.2f} vs peak {peak:.2f})"
            )
    return None


def check_orders(orders: list[Order], equity: float, config: SafetyConfig) -> None:
    """Raise :class:`SafetyViolation` when the order set breaks a guard.

    Runs before the write-ahead persist (``decide_and_submit`` order_guard
    seam), so a violation leaves no pending state and nothing at the venue.
    """
    if config.max_orders is not None and len(orders) > config.max_orders:
        raise SafetyViolation(
            f"decision produced {len(orders)} orders > max_orders {config.max_orders}; "
            "refusing before the write-ahead persist — check universe/config corruption"
        )
    if config.max_order_fraction is not None:
        bound = config.max_order_fraction * equity
        oversized = [
            (o.symbol, abs(o.qty) * o.reference_price)
            for o in orders
            if abs(o.qty) * o.reference_price > bound
        ]
        if oversized:
            worst = sorted(oversized, key=lambda item: -item[1])[:5]
            raise SafetyViolation(
                f"{len(oversized)} orders exceed max_order_fraction "
                f"{config.max_order_fraction:.2%} of equity {equity:.2f} "
                f"(bound {bound:.2f}); worst {[(s, round(n, 2)) for s, n in worst]}; "
                "refusing before the write-ahead persist — check equity/price corruption"
            )
