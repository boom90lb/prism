"""The paper-loop equity ledger + anytime-valid monitor feed (SPEC §7.7, §10).

Pins the seam that turns the accruing loop into a live monitor: one durable NAV
row per decision bar (idempotent against the same-bar restarts the write-ahead
protocol performs), and the bridge that reads it into the time-uniform
confidence sequence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prism.live import (
    DailyBookConfig,
    LiveLoopContext,
    StateStore,
    read_equity_ledger,
    run_daily_cycle,
)
from prism.live.loop import _append_equity_ledger
from prism.live.monitor import paper_monitor_read
from tests.test_live_daily import ConstSignal, _panels
from tests.test_live_loop import FakeBroker

_CONFIG = DailyBookConfig(position_size=0.1, min_order_notional=1.0)


def _ctx(tmp_path):
    return LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=FakeBroker(),
        fills_ledger=tmp_path / "fills.jsonl",
        equity_ledger=tmp_path / "equity.jsonl",
    )


# ---------------------------------------------------------------------------
# The ledger primitive
# ---------------------------------------------------------------------------


def test_append_equity_ledger_idempotent_and_monotone(tmp_path):
    p = tmp_path / "equity.jsonl"
    _append_equity_ledger(p, "2026-05-01", 100_000.0, 100_000.0)
    _append_equity_ledger(p, "2026-05-01", 111_111.0, 0.0)  # same bar (restart) — skip
    _append_equity_ledger(p, "2026-05-04", 100_500.0, 90_000.0)  # newer bar — append
    _append_equity_ledger(p, "2026-05-02", 999.0, 999.0)  # out-of-order older bar — skip
    df = read_equity_ledger(p)
    assert list(df["decision_bar"]) == ["2026-05-01", "2026-05-04"]
    assert list(df["equity"]) == [100_000.0, 100_500.0]


def test_read_equity_ledger_missing_is_empty(tmp_path):
    assert read_equity_ledger(tmp_path / "nope.jsonl").empty


# ---------------------------------------------------------------------------
# Wiring into the daily cycle
# ---------------------------------------------------------------------------


def test_daily_cycle_writes_one_row_per_bar_idempotently(tmp_path):
    ctx = _ctx(tmp_path)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    close, volume = _panels(n=30)
    run_daily_cycle(ctx, signal, close, volume, _CONFIG)
    run_daily_cycle(ctx, signal, close, volume, _CONFIG)  # same-bar resume — no second row
    df = read_equity_ledger(ctx.equity_ledger)
    assert len(df) == 1
    assert df["equity"].iloc[0] == pytest.approx(10_000.0)

    close2, volume2 = _panels(n=31)  # a new bar
    run_daily_cycle(ctx, signal, close2, volume2, _CONFIG)
    assert len(read_equity_ledger(ctx.equity_ledger)) == 2


def test_daily_cycle_without_equity_ledger_is_noop(tmp_path):
    ctx = LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=FakeBroker(),
        fills_ledger=tmp_path / "fills.jsonl",
    )  # equity_ledger defaults None
    close, volume = _panels()
    result = run_daily_cycle(ctx, ConstSignal({"AAA": 1.0}), close, volume, _CONFIG)
    assert not (tmp_path / "equity.jsonl").exists()
    assert result.monitor_read is None  # no ledger -> no in-loop monitor read


def test_daily_cycle_arms_anytime_monitor(tmp_path):
    ctx = _ctx(tmp_path)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    close, volume = _panels(n=30)
    result = run_daily_cycle(ctx, signal, close, volume, _CONFIG)
    # One NAV point after the first cycle -> no return yet -> inconclusive, but the
    # read is armed in-loop (additive telemetry), not left to the CLI.
    assert result.monitor_read is not None
    assert result.monitor_read["verdict"] == "inconclusive"

    close2, volume2 = _panels(n=31)  # a newer bar -> a second NAV point -> one return
    result2 = run_daily_cycle(ctx, signal, close2, volume2, _CONFIG)
    assert result2.monitor_read is not None
    assert "verdict" in result2.monitor_read
    assert result2.monitor_read["n"] == 1


# ---------------------------------------------------------------------------
# The monitor bridge
# ---------------------------------------------------------------------------


def test_monitor_read_empty_and_singleton_are_inconclusive(tmp_path):
    p = tmp_path / "equity.jsonl"
    r0 = paper_monitor_read(p)
    assert r0["n"] == 0 and r0["verdict"] == "inconclusive"

    _append_equity_ledger(p, "2026-05-01", 100_000.0, 100_000.0)
    r1 = paper_monitor_read(p)
    assert r1["n"] == 0 and r1["n_equity_points"] == 1 and r1["verdict"] == "inconclusive"


def test_monitor_read_flat_equity_is_inconclusive(tmp_path):
    p = tmp_path / "equity.jsonl"
    for bar in pd.date_range("2026-05-01", periods=40, freq="B"):
        _append_equity_ledger(p, str(bar.date()), 100_000.0, 100_000.0)
    r = paper_monitor_read(p, hurdle=0.0)
    assert r["n"] == 39  # 40 NAV points -> 39 returns
    assert r["verdict"] == "inconclusive"  # zero returns straddle the hurdle
    assert r["latest_equity"] == 100_000.0


def test_monitor_read_confirms_a_strong_steady_climb(tmp_path):
    p = tmp_path / "equity.jsonl"
    equity = 100_000.0
    for bar in pd.date_range("2026-05-01", periods=400, freq="B"):
        _append_equity_ledger(p, str(bar.date()), equity, equity)
        equity *= 1.02  # +2%/bar: mean return 0.02 >> the CS half-width at bound 0.05
    r = paper_monitor_read(p, hurdle=0.0, bound=0.05)
    assert r["verdict"] == "confirmed"
    assert r["ci_lower"] > 0.0
