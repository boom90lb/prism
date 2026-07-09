"""Live loop (SPEC.md §7.7) — durable state, reconcile, decide, order.

This package is the engineering core of the R2 paper-loop instrument (the
I-9 cost-calibration instrument SPEC §13 pulls forward from R4): it exists
to force durable order state into existence and to record every fill as
spread-calibration data. The binding invariant is N2 crash-safety — a daily
loop restarts between the decision (close *t*) and the fill (open *t+1*),
and nothing important may live only in memory:

* ``state``  — the durable loop state (positions, cash, pending orders,
  last processed bar), atomic-write JSON, schema-versioned, fail-loud.
* ``broker`` — the Broker contract (idempotent submit keyed by
  ``client_order_id``) plus the Order/Fill types.
* ``alpaca`` — the concrete Alpaca REST adapter, written against an
  injectable session so every venue mapping is tested offline; only the
  transport is network-gated.
* ``loop``   — the write-ahead decision protocol: reconcile broker truth,
  turn constructed target weights into orders, persist pending orders
  BEFORE submitting, resume idempotently after a crash, settle fills into
  the append-only fills ledger (the I-9 calibration record).
* ``daily``  — the one-cycle driver wiring delta-fetched panels → Signal
  → online construction (caps, stateful band, participation gate) → the
  decision protocol (``prism.scripts.paper_loop`` is its CLI shell).

Production-import-path safe (N8): numpy/pandas/requests only, no research
heavyweights, no prophet/matplotlib.
"""

from __future__ import annotations

from prism.live.alpaca import (
    LIVE_BASE_URL,
    PAPER_BASE_URL,
    AlpacaAPIError,
    AlpacaBroker,
)
from prism.live.alpaca_data import DATA_BASE_URL, DEFAULT_FEED, AlpacaBarSource
from prism.live.broker import Broker, DuplicateOrder, Fill, Order, OrderRejected
from prism.live.daily import (
    DailyBookConfig,
    DailyCycleResult,
    fetch_universe_panels,
    run_daily_cycle,
)
from prism.live.loop import (
    LiveLoopContext,
    decide_and_submit,
    read_equity_ledger,
    read_fills_ledger,
    settle,
    targets_to_orders,
)
from prism.live.state import LoopState, StateStore

__all__ = [
    "DATA_BASE_URL",
    "DEFAULT_FEED",
    "LIVE_BASE_URL",
    "PAPER_BASE_URL",
    "AlpacaAPIError",
    "AlpacaBarSource",
    "AlpacaBroker",
    "Broker",
    "DailyBookConfig",
    "DailyCycleResult",
    "DuplicateOrder",
    "Fill",
    "LiveLoopContext",
    "OrderRejected",
    "LoopState",
    "Order",
    "StateStore",
    "decide_and_submit",
    "fetch_universe_panels",
    "read_equity_ledger",
    "read_fills_ledger",
    "run_daily_cycle",
    "settle",
    "targets_to_orders",
]
