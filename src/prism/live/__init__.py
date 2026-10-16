"""Live loop (SPEC.md §7.7) — durable state, reconcile, decide, order.

This package is the engineering core of the R2 paper-loop instrument (the
I-9 cost-calibration instrument SPEC §13 pulls forward from R4): it exists
to force durable order state into existence and to record every fill as
spread-calibration data. The binding invariant is N2 crash-safety — a daily
loop restarts between the decision (close *t*) and the fill (open *t+1*),
and nothing important may live only in memory:

* ``state``  — the durable loop state at schema version 3 (the canonical
  ``PortfolioState``, the fills marker, the per-component schedule and fit
  references, the bar markers), atomic-write JSON, schema-versioned,
  fail-loud, with the v1 and v2 documents migrated forward explicitly.
* ``reconcile`` — the venue against the book: positions and cash adopted,
  orders and executions compared and never adopted, and the cost, turnover
  and variance reads that are recorded and gate nothing (§3.11).
* ``broker`` — the Broker contract (idempotent submit keyed by
  ``client_order_id``) plus the Order/Fill types.
* ``ledgers`` — the one JSONL append/read discipline every run-dir ledger
  rides: fsynced appends, torn-tail tolerance, monotone idempotency.
* ``authorization`` — the §3.10 authorization record, the action classes and
  the operating states: what the account owner approved, and the intersection
  rule that turns the states, the scope flags and the per-input-class freshness
  into the set of actions this process may take. Every answer is a refusal or a
  permission; a control that cannot verify its precondition never warns and
  proceeds.
* ``lockfile`` — the run directory's advisory writer lock (the write-ahead
  protocol survives crashes, not two concurrent writers).
* ``alpaca_transport`` — the shared Alpaca REST session: one auth/timeout/
  transient-retry (429/5xx) policy for the broker, the bar source, and the
  corporate-actions fetch.
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
* ``deployment`` — the per-deployment seed (``{run-dir}/deployment.json``,
  mode 0600) and the draws taken from it, such as the seeded rebalance phase
  that keeps deployments started on the same day from trading on the same
  days. Off unless a seeded mode is requested.
* ``vol_sizing`` — the volatility scale of the decile book (Barroso and
  Santa-Clara, ``prism.portfolio.vol_scale``): the base book rebuilt point in
  time from the panel, the scale decided at each refresh with its failure
  path, and the per-refresh record ``{run-dir}/sizing.jsonl``. Off unless
  ``run_daily_cycle(vol_scale=...)`` is given.
* ``tranches`` — the tranched decile book: overlapping tranches on the session
  grid, built one per grid date (staged entry) and exited one per grid date
  (wind-down), their states persisted in ``LoopState.components``. Off unless
  ``run_daily_cycle(tranches=...)`` is given a count above one.
* ``cash_holding`` — idle own equity above a cash buffer held in a T-bill ETF
  beside the decile book, set every session within a band and the per-order
  bound. Off unless ``run_daily_cycle(cash_holding=...)`` is given.
* ``buying_power`` — the pre-check that scales a decision's opening orders
  down together when they need more than the venue's buying power. Off unless
  ``decide_and_submit(buying_power_check=True)``.
* ``margin`` — the account's margin position read from the venue
  (``AlpacaBroker.account_margin``) and its record ``{run-dir}/margin.jsonl``.
  Read-only telemetry for ``prism.scripts.margin_report`` and ``prism-doctor``;
  nothing on the order path reads it.
* ``regime_step`` — the SPEC §7.7 regime step: per-cycle §7.5 telemetry
  through the ``prism.regime.fetch`` adapters, read by ``daily`` every
  session into the regime ledger (docs/regime_step.md); the de-gross
  action hook stays unarmed until docs/sizing_preregistration.md
  ratifies.
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
from prism.live.authorization import (
    ActionClass,
    ActionContext,
    AuthorizationGate,
    AuthorizationRecord,
    AuthorizationRefused,
    InputFreshness,
    OperatingState,
    ProcessIdentity,
    Scope,
    SessionIntent,
    classify_order,
    load_authorization,
    permitted_classes,
    resolve_endpoint,
)
from prism.live.broker import Broker, DuplicateOrder, Fill, Order, OrderRejected, transmits_to_venue
from prism.live.daily import (
    DailyBookConfig,
    DailyCycleResult,
    HeldUniverse,
    PartialBarError,
    fetch_universe_panels,
    require_published_panel,
    resolve_fetch_universe,
    run_daily_cycle,
)
from prism.live.loop import (
    SUBMISSION_ANNOUNCEMENT,
    LiveLoopContext,
    announce_submission,
    cancel_orders,
    decide_and_submit,
    read_concordance_ledger,
    read_equity_ledger,
    read_fills_ledger,
    read_regime_ledger,
    read_targets_ledger,
    settle,
    sweep_pending,
    targets_to_orders,
)
from prism.live.monitor import book_concordance, paper_monitor_read
from prism.live.no_submit import (
    NO_SUBMIT,
    WOULD_SUBMIT,
    NoSubmitBroker,
    NoSubmitSession,
    VenueWriteRefused,
    mirror_run_dir,
    no_submit_session,
)
from prism.live.reconcile import (
    ReconcileItem,
    ReconcileSettings,
    Reconciliation,
    reconcile,
    reconcile_loop_state,
)
from prism.live.regime_step import ABSENT_BLOCKS, REGIME_BLOCKS, RegimeTelemetry
from prism.live.replay import (
    ReplayBroker,
    align_replay_panels,
    load_local_bar_panels,
    replay_daily_cycles,
)
from prism.live.risk_profile import (
    CERTIFIED_B1_PAPER_CONFIG,
    PROFILE_IDS,
    HedgePolicy,
    RiskProfile,
    SleeveBand,
    assert_research_paper_bit_identity,
    book_config_matches_certified_paper,
    resolve_risk_profile,
    validate_profile_payload,
)
from prism.live.safety import (
    SESSION_HISTORY_FILES,
    SUBMISSION_MODE_EXIT_CODE,
    RunDirHistory,
    SafetyConfig,
    SafetyViolation,
    active_run_dir_pointer,
    check_orders,
    halt_reason,
    require_active_run_dir,
    require_established_run_dir,
    sample_run_dir_history,
    submission_mode_refusal,
)
from prism.live.spinoff_mask import (
    CORPORATE_ACTIONS_URL,
    fetch_spinoffs,
    spinoff_flags,
    spinoff_unrankable,
    spinoff_unrankable_provider,
)
from prism.live.state import ComponentState, LoopState, StateStore

__all__ = [
    "ABSENT_BLOCKS",
    "CORPORATE_ACTIONS_URL",
    "DATA_BASE_URL",
    "DEFAULT_FEED",
    "LIVE_BASE_URL",
    "NO_SUBMIT",
    "NoSubmitBroker",
    "NoSubmitSession",
    "PAPER_BASE_URL",
    "REGIME_BLOCKS",
    "ActionClass",
    "ActionContext",
    "AlpacaAPIError",
    "AlpacaBarSource",
    "AlpacaBroker",
    "AuthorizationGate",
    "AuthorizationRecord",
    "AuthorizationRefused",
    "Broker",
    "CERTIFIED_B1_PAPER_CONFIG",
    "ComponentState",
    "DailyBookConfig",
    "DailyCycleResult",
    "DuplicateOrder",
    "Fill",
    "HedgePolicy",
    "HeldUniverse",
    "InputFreshness",
    "LiveLoopContext",
    "OperatingState",
    "OrderRejected",
    "LoopState",
    "Order",
    "PROFILE_IDS",
    "ProcessIdentity",
    "ReconcileItem",
    "ReconcileSettings",
    "Reconciliation",
    "RegimeTelemetry",
    "ReplayBroker",
    "RiskProfile",
    "PartialBarError",
    "RunDirHistory",
    "SESSION_HISTORY_FILES",
    "SUBMISSION_ANNOUNCEMENT",
    "SUBMISSION_MODE_EXIT_CODE",
    "SafetyConfig",
    "SafetyViolation",
    "Scope",
    "SessionIntent",
    "SleeveBand",
    "StateStore",
    "VenueWriteRefused",
    "WOULD_SUBMIT",
    "align_replay_panels",
    "announce_submission",
    "assert_research_paper_bit_identity",
    "book_concordance",
    "book_config_matches_certified_paper",
    "cancel_orders",
    "check_orders",
    "classify_order",
    "decide_and_submit",
    "fetch_spinoffs",
    "halt_reason",
    "fetch_universe_panels",
    "load_authorization",
    "load_local_bar_panels",
    "mirror_run_dir",
    "no_submit_session",
    "paper_monitor_read",
    "permitted_classes",
    "read_concordance_ledger",
    "reconcile",
    "reconcile_loop_state",
    "read_equity_ledger",
    "read_fills_ledger",
    "active_run_dir_pointer",
    "require_active_run_dir",
    "require_established_run_dir",
    "require_published_panel",
    "resolve_endpoint",
    "resolve_fetch_universe",
    "resolve_risk_profile",
    "sample_run_dir_history",
    "submission_mode_refusal",
    "transmits_to_venue",
    "validate_profile_payload",
    "read_regime_ledger",
    "read_targets_ledger",
    "replay_daily_cycles",
    "run_daily_cycle",
    "settle",
    "spinoff_flags",
    "spinoff_unrankable",
    "spinoff_unrankable_provider",
    "sweep_pending",
    "targets_to_orders",
]
