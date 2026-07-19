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
* ``replay`` — the diagnostic replay instrument: the same daily cycle over
  local historical bars with a simulated next-open venue, faster than
  realtime (``prism.scripts.replay_loop`` is its CLI shell). Modeled fills:
  never cost calibration, never the live-monitor concordance stream.

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
    read_concordance_ledger,
    read_equity_ledger,
    read_fills_ledger,
    read_targets_ledger,
    settle,
    sweep_pending,
    targets_to_orders,
)
from prism.live.monitor import book_concordance, paper_monitor_read
from prism.live.replay import (
    ReplayBroker,
    align_replay_panels,
    load_local_bar_panels,
    replay_daily_cycles,
)
from prism.live.safety import SafetyConfig, SafetyViolation, check_orders, halt_reason
from prism.live.spinoff_mask import (
    CORPORATE_ACTIONS_URL,
    fetch_spinoffs,
    spinoff_flags,
    spinoff_unrankable,
)
from prism.live.state import LoopState, StateStore

__all__ = [
    "CORPORATE_ACTIONS_URL",
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
    "ReplayBroker",
    "SafetyConfig",
    "SafetyViolation",
    "StateStore",
    "align_replay_panels",
    "book_concordance",
    "check_orders",
    "decide_and_submit",
    "fetch_spinoffs",
    "halt_reason",
    "fetch_universe_panels",
    "load_local_bar_panels",
    "paper_monitor_read",
    "read_concordance_ledger",
    "read_equity_ledger",
    "read_fills_ledger",
    "read_targets_ledger",
    "replay_daily_cycles",
    "run_daily_cycle",
    "settle",
    "spinoff_flags",
    "spinoff_unrankable",
    "sweep_pending",
    "targets_to_orders",
]
