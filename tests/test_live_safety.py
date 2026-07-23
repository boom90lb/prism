"""Safety rails (prism.live.safety): halt semantics and pre-persist guards.

Pins the two contracts: a *halt* (kill switch, drawdown) settles and marks NAV
but constructs nothing, persists nothing, submits nothing — including the
same-bar resume path — while an *order guard* violation raises before the
write-ahead persist, leaving no pending state and nothing at the venue.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from prism.live import (
    DailyBookConfig,
    LiveLoopContext,
    Order,
    SafetyConfig,
    SafetyViolation,
    StateStore,
    check_orders,
    halt_reason,
    run_daily_cycle,
)
from tests.test_live_daily import ConstSignal, _panels
from tests.test_live_loop import FakeBroker


@pytest.fixture()
def ctx(tmp_path):
    return LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=FakeBroker(),
        fills_ledger=tmp_path / "fills.jsonl",
        equity_ledger=tmp_path / "equity.jsonl",
    )


_CONFIG = DailyBookConfig(position_size=0.1, min_order_notional=1.0)
_SIGNAL = ConstSignal({"AAA": 1.0, "BBB": -1.0})


# ---------------------------------------------------------------------------
# SafetyConfig validation
# ---------------------------------------------------------------------------


def test_safety_config_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError):
        SafetyConfig(max_drawdown=1.5)
    with pytest.raises(ValueError):
        SafetyConfig(max_order_fraction=0.0)
    with pytest.raises(ValueError):
        SafetyConfig(max_orders=0)


# ---------------------------------------------------------------------------
# Halt rails: kill switch and drawdown
# ---------------------------------------------------------------------------


def test_kill_switch_halts_but_still_marks_nav(ctx, tmp_path) -> None:
    kill = tmp_path / "KILL_SWITCH"
    kill.touch()
    close, volume = _panels()
    result = run_daily_cycle(
        ctx, _SIGNAL, close, volume, _CONFIG, safety=SafetyConfig(kill_switch=kill)
    )
    assert result.halted is not None and "kill switch" in result.halted
    assert result.submitted_orders == []
    assert ctx.broker.submit_calls == 0  # nothing reached the venue
    state = ctx.store.load()
    assert state is None or not state.pending_orders  # nothing persisted either
    # The ledger stays continuous through a halt: today's NAV row exists.
    rows = [json.loads(ln) for ln in ctx.equity_ledger.read_text().splitlines() if ln]
    assert len(rows) == 1 and rows[0]["equity"] == pytest.approx(10_000.0)


def test_kill_switch_blocks_the_same_bar_resume_path(ctx, tmp_path) -> None:
    close, volume = _panels()
    # Bar decided and submitted normally…
    first = run_daily_cycle(ctx, _SIGNAL, close, volume, _CONFIG)
    assert first.halted is None and ctx.broker.submit_calls > 0
    submitted_before = ctx.broker.submit_calls
    # …then the operator drops the kill switch and the same bar re-runs (the
    # write-ahead restart). The resume path must be gated too.
    kill = tmp_path / "KILL_SWITCH"
    kill.touch()
    resumed = run_daily_cycle(
        ctx, _SIGNAL, close, volume, _CONFIG, safety=SafetyConfig(kill_switch=kill)
    )
    assert resumed.halted is not None
    assert ctx.broker.submit_calls == submitted_before  # no new venue traffic


def test_drawdown_halt_reads_the_equity_ledger_peak(ctx) -> None:
    # A prior NAV peak far above today's mark: 10k equity vs 100k peak = 90% dd.
    ctx.equity_ledger.write_text(
        json.dumps({"decision_bar": "2026-04-01", "equity": 100_000.0, "cash": 0.0}) + "\n"
    )
    close, volume = _panels()
    result = run_daily_cycle(
        ctx, _SIGNAL, close, volume, _CONFIG, safety=SafetyConfig(max_drawdown=0.25)
    )
    assert result.halted is not None and "drawdown" in result.halted
    assert ctx.broker.submit_calls == 0


def test_drawdown_within_bound_trades_normally(ctx) -> None:
    ctx.equity_ledger.write_text(
        json.dumps({"decision_bar": "2026-04-01", "equity": 10_500.0, "cash": 0.0}) + "\n"
    )
    close, volume = _panels()
    result = run_daily_cycle(
        ctx, _SIGNAL, close, volume, _CONFIG, safety=SafetyConfig(max_drawdown=0.25)
    )
    assert result.halted is None
    assert ctx.broker.submit_calls > 0


def test_halt_reason_pure_paths(tmp_path) -> None:
    assert halt_reason(SafetyConfig(), 100.0, None) is None
    # No ledger configured -> the drawdown rail cannot fire on one mark.
    assert halt_reason(SafetyConfig(max_drawdown=0.1), 100.0, None) is None
    missing = tmp_path / "not_there"
    assert halt_reason(SafetyConfig(kill_switch=missing), 100.0, None) is None


# ---------------------------------------------------------------------------
# Order guards: pre-persist, derived-bound semantics
# ---------------------------------------------------------------------------


def _order(symbol: str, qty: float, price: float) -> Order:
    return Order(
        client_order_id=f"d:{symbol}", symbol=symbol, qty=qty, decision_bar="d", reference_price=price
    )


def test_check_orders_notional_bound() -> None:
    config = SafetyConfig(max_order_fraction=0.05)
    check_orders([_order("AAA", 4.0, 100.0)], 10_000.0, config)  # $400 < $500: fine
    with pytest.raises(SafetyViolation, match="max_order_fraction"):
        check_orders([_order("AAA", 6.0, 100.0)], 10_000.0, config)  # $600 > $500


def test_check_orders_count_bound() -> None:
    config = SafetyConfig(max_orders=2)
    orders = [_order(s, 1.0, 10.0) for s in ("AAA", "BBB", "CCC")]
    with pytest.raises(SafetyViolation, match="max_orders"):
        check_orders(orders, 10_000.0, config)


def test_order_guard_violation_leaves_no_pending_state(ctx) -> None:
    close, volume = _panels()
    # A bound tighter than any real order forces the guard to fire.
    with pytest.raises(SafetyViolation):
        run_daily_cycle(
            ctx, _SIGNAL, close, volume, _CONFIG, safety=SafetyConfig(max_order_fraction=1e-6)
        )
    assert ctx.broker.submit_calls == 0  # nothing reached the venue
    state = ctx.store.load()
    assert state is None or not state.pending_orders  # and nothing was persisted
