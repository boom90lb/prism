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
  ``client_order_id``) plus the Order/Fill types. Concrete network
  adapters (Alpaca) land separately, once there are credentials to
  exercise them against the paper API — an untested REST shell would be
  drafted, not run.
* ``loop``   — the write-ahead decision protocol: reconcile broker truth,
  turn constructed target weights into orders, persist pending orders
  BEFORE submitting, resume idempotently after a crash, settle fills into
  the append-only fills ledger (the I-9 calibration record).

Production-import-path safe (N8): numpy/pandas only, no research
heavyweights, no prophet/matplotlib.
"""

from __future__ import annotations

from prism.live.broker import Broker, DuplicateOrder, Fill, Order
from prism.live.loop import (
    LiveLoopContext,
    decide_and_submit,
    read_fills_ledger,
    settle,
    targets_to_orders,
)
from prism.live.state import LoopState, StateStore

__all__ = [
    "Broker",
    "DuplicateOrder",
    "Fill",
    "LiveLoopContext",
    "LoopState",
    "Order",
    "StateStore",
    "decide_and_submit",
    "read_fills_ledger",
    "settle",
    "targets_to_orders",
]
