"""The daily driver (SPEC §7.7): fetch → score → construct → decide, wired.

Pins the seams the driver owns: panel assembly from the incremental fetch
(fail-loud, N7), the settle-then-decide ordering across bars, same-bar
restart resumption, held-weight hysteresis through the *online* band, the
participation gate on ADV, and whole-share order sizing for OPG venues.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from prism.live import (
    DailyBookConfig,
    LiveLoopContext,
    StateStore,
    fetch_universe_panels,
    read_fills_ledger,
    run_daily_cycle,
    targets_to_orders,
)
from prism.live.state import STATE_SCHEMA_VERSION
from prism.signal.base import Signal
from tests.test_live_loop import FakeBroker


class ConstSignal(Signal):
    """Fixed scores per symbol; NaN elsewhere. Enough to drive the wiring."""

    def __init__(self, scores: dict[str, float], required: int = 3) -> None:
        self._scores = scores
        self._required = required

    @property
    def horizon_bars(self) -> int:
        return 1

    @property
    def required_history(self) -> int:
        return self._required

    def fit(self, close, volume=None) -> "ConstSignal":
        return self

    def score(self, close, volume=None) -> pd.Series:
        return pd.Series(self._scores, dtype=float).reindex(close.columns)


def _panels(n: int = 30, prices: dict[str, float] | None = None, volume: float = 1e6):
    prices = prices or {"AAA": 100.0, "BBB": 50.0}
    idx = pd.date_range("2026-05-01", periods=n, freq="B", tz="America/New_York")
    close = pd.DataFrame({s: [p] * n for s, p in prices.items()}, index=idx)
    vol = pd.DataFrame(volume, index=idx, columns=close.columns)
    return close, vol


@pytest.fixture()
def ctx(tmp_path):
    return LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=FakeBroker(),
        fills_ledger=tmp_path / "fills.jsonl",
    )


_CONFIG = DailyBookConfig(position_size=0.1, min_order_notional=1.0)


# ---------------------------------------------------------------------------
# Whole-share sizing (OPG venue requirement)
# ---------------------------------------------------------------------------


def test_targets_to_orders_skips_flat_unheld_name_without_price() -> None:
    # A large universe carries names the construct leaves at 0.0 that also lack a
    # decision price (an ineligible / gapped latest bar). A flat, unheld name is a
    # no-op — skipped, not an N7 error that crashes the whole book.
    orders = targets_to_orders(
        pd.Series({"AAA": 0.5, "GAP": 0.0}),
        positions={},
        prices=pd.Series({"AAA": 100.0, "GAP": float("nan")}),
        equity=10_000.0,
        decision_bar="d",
        whole_shares=True,
    )
    assert {o.symbol for o in orders} == {"AAA"}  # GAP skipped despite its NaN price


def test_whole_shares_rounds_deltas_and_drops_zero() -> None:
    orders = targets_to_orders(
        pd.Series({"AAA": 0.5, "BBB": 0.001}),
        positions={},
        prices=pd.Series({"AAA": 98.0, "BBB": 50.0}),
        equity=10_000.0,
        decision_bar="d",
        whole_shares=True,
    )
    by_symbol = {o.symbol: o for o in orders}
    # AAA: 0.5*10000/98 = 51.02 shares -> 51 whole; BBB: 0.2 shares -> 0, dropped.
    assert by_symbol["AAA"].qty == 51.0
    assert "BBB" not in by_symbol


# ---------------------------------------------------------------------------
# Panel assembly from the incremental fetch
# ---------------------------------------------------------------------------


class FakeLoader:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.requests: list[str] = []

    def fetch_incremental(self, symbol, interval="1d", start_date=None, end_date=None, store=None):
        self.requests.append(symbol)
        return self.frames.get(symbol, pd.DataFrame())


def _bars(n: int, price: float, with_volume: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2026-05-01", periods=n, freq="B", tz="America/New_York")
    data = {"close": [price] * n}
    if with_volume:
        data["volume"] = [1e6] * n
    return pd.DataFrame(data, index=idx)


def test_fetch_universe_panels_assembles_wide_frames() -> None:
    loader = FakeLoader({"AAA": _bars(10, 100.0), "BBB": _bars(10, 50.0)})
    close, volume = fetch_universe_panels(loader, ["AAA", "BBB"])
    assert list(close.columns) == ["AAA", "BBB"] and len(close) == 10
    assert volume is not None and float(volume.iloc[-1, 0]) == 1e6
    assert str(close.index.tz) == "America/New_York"
    assert loader.requests == ["AAA", "BBB"]


def test_fetch_universe_panels_fails_loud_on_missing_symbol() -> None:
    loader = FakeLoader({"AAA": _bars(10, 100.0)})
    with pytest.raises(RuntimeError, match="GONE"):  # default max_missing=0.0 -> strict
        fetch_universe_panels(loader, ["AAA", "GONE"])


def test_fetch_universe_panels_tolerates_bounded_missing() -> None:
    loader = FakeLoader({"AAA": _bars(10, 100.0), "BBB": _bars(10, 50.0)})  # CCC absent
    close, _ = fetch_universe_panels(loader, ["AAA", "BBB", "CCC"], max_missing=0.5)
    assert list(close.columns) == ["AAA", "BBB"]  # CCC dropped and traded around, not raised


def test_fetch_universe_panels_raises_when_missing_exceeds_bound() -> None:
    loader = FakeLoader({"AAA": _bars(10, 100.0)})  # BBB, CCC absent -> 2/3 miss
    with pytest.raises(RuntimeError, match="max_missing"):
        fetch_universe_panels(loader, ["AAA", "BBB", "CCC"], max_missing=0.1)


def test_fetch_universe_panels_without_volume_returns_none() -> None:
    loader = FakeLoader({"AAA": _bars(10, 100.0, with_volume=False)})
    close, volume = fetch_universe_panels(loader, ["AAA"])
    assert volume is None and len(close) == 10


class FakeBatchLoader:
    """A batch-capable loader (like AlpacaBarSource); fetch_universe_panels must
    prefer its one-shot fetch_batch over a per-symbol loop."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.batch_calls = 0

    def fetch_batch(self, symbols, interval="1d", start_date=None, end_date=None):
        self.batch_calls += 1
        return {s: self.frames.get(s, pd.DataFrame()) for s in symbols}


def test_fetch_universe_panels_uses_batch_when_available() -> None:
    loader = FakeBatchLoader({"AAA": _bars(10, 100.0), "BBB": _bars(10, 50.0)})
    close, volume = fetch_universe_panels(loader, ["AAA", "BBB"])
    assert list(close.columns) == ["AAA", "BBB"] and len(close) == 10
    assert volume is not None
    assert loader.batch_calls == 1  # one batched call, not two per-symbol fetches


# ---------------------------------------------------------------------------
# The daily cycle
# ---------------------------------------------------------------------------


def test_first_cycle_decides_and_submits(ctx) -> None:
    close, volume = _panels()
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    result = run_daily_cycle(ctx, signal, close, volume, _CONFIG)

    assert result.settled_fills == []
    assert result.equity == pytest.approx(10_000.0)
    by_symbol = {o.symbol: o for o in result.submitted_orders}
    # 0.1 weight at equity 10k: AAA +10 shares at 100, BBB -20 shares at 50.
    assert by_symbol["AAA"].qty == 10.0
    assert by_symbol["BBB"].qty == -20.0
    assert result.decision_bar == str(close.index[-1].date())
    state = ctx.store.load()
    assert state.pending_decision_bar == result.decision_bar


def test_nan_score_produces_no_order(ctx) -> None:
    close, volume = _panels(prices={"AAA": 100.0, "BBB": 50.0, "CCC": 10.0})
    signal = ConstSignal({"AAA": 1.0})  # BBB/CCC: no opinion
    result = run_daily_cycle(ctx, signal, close, volume, _CONFIG)
    assert {o.symbol for o in result.submitted_orders} == {"AAA"}


def test_next_bar_settles_then_rediffs_against_held(ctx) -> None:
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    close, volume = _panels(n=30)
    first = run_daily_cycle(ctx, signal, close, volume, _CONFIG)

    close2, volume2 = _panels(n=31)  # one new bar; fills happened at its open
    second = run_daily_cycle(ctx, signal, close2, volume2, _CONFIG)

    assert len(second.settled_fills) == 2
    ledger = read_fills_ledger(ctx.fills_ledger)
    assert set(ledger["decision_bar"]) == {first.decision_bar}
    state = ctx.store.load()
    assert state.last_settled_bar == first.decision_bar
    # Same scores, unchanged prices: the held book already matches the
    # target, so the re-diff is all dust and nothing trades.
    assert second.submitted_orders == []
    assert state.positions == {"AAA": 10.0, "BBB": -20.0}


def test_same_bar_rerun_resumes_without_settling_or_redeciding(ctx) -> None:
    close, volume = _panels()
    ctx.broker.fail_after_n_submits = 1  # crash mid-submit
    with pytest.raises(ConnectionError):
        run_daily_cycle(ctx, ConstSignal({"AAA": 1.0, "BBB": -1.0}), close, volume, _CONFIG)
    assert len(ctx.broker.submitted_order_ids()) == 1

    ctx.broker.fail_after_n_submits = None
    # Restart with DIFFERENT scores: the persisted decision must win.
    result = run_daily_cycle(ctx, ConstSignal({"AAA": -1.0, "BBB": 1.0}), close, volume, _CONFIG)
    assert result.settled_fills == []  # no settle attempted within the same bar
    assert len(ctx.broker.submitted_order_ids()) == 2
    by_symbol = {o.symbol: o for o in result.submitted_orders}
    assert by_symbol["AAA"].qty == 10.0  # original decision, not the new scores


def test_online_band_holds_within_band(tmp_path) -> None:
    ctx = LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=FakeBroker(),
        fills_ledger=tmp_path / "fills.jsonl",
    )
    config = DailyBookConfig(position_size=0.1, no_trade_band=0.05, min_order_notional=1.0)
    close, volume = _panels(n=30, prices={"AAA": 100.0})
    run_daily_cycle(ctx, ConstSignal({"AAA": 1.0}), close, volume, config)

    # Next bar: fills settle to a held weight of 0.1; the fresh target 0.09
    # is within the 0.05 band of held, so the book holds and nothing trades.
    close2, volume2 = _panels(n=31, prices={"AAA": 100.0})
    result = run_daily_cycle(ctx, ConstSignal({"AAA": 0.9}), close2, volume2, config)
    assert result.submitted_orders == []
    assert result.target_weights["AAA"] == pytest.approx(0.1)


def test_participation_gate_shrinks_the_trade(ctx) -> None:
    config = DailyBookConfig(position_size=0.1, max_participation=0.01, min_order_notional=1.0)
    # ADV = 100 shares * $100 = $10k; allowed weight delta = 0.01*10k/10k = 0.01.
    close, volume = _panels(n=30, prices={"AAA": 100.0}, volume=100.0)
    result = run_daily_cycle(ctx, ConstSignal({"AAA": 1.0}), close, volume, config)
    assert result.target_weights["AAA"] == pytest.approx(0.01)
    assert [o.qty for o in result.submitted_orders] == [1.0]


def test_participation_without_volume_raises(ctx) -> None:
    config = DailyBookConfig(position_size=0.1, max_participation=0.01)
    close, _ = _panels()
    with pytest.raises(ValueError, match="no volume panel"):
        run_daily_cycle(ctx, ConstSignal({"AAA": 1.0}), close, None, config)


def test_truncated_panel_refused(ctx) -> None:
    close, volume = _panels(n=5)
    signal = ConstSignal({"AAA": 1.0}, required=10)
    with pytest.raises(ValueError, match="required_history"):
        run_daily_cycle(ctx, signal, close, volume, _CONFIG)


# ---------------------------------------------------------------------------
# The decile-neutral (B1 momentum) book + monthly decision cadence (S3/S4)
# ---------------------------------------------------------------------------


def _decile_panels(n: int = 30, n_sym: int = 10):
    return _panels(n=n, prices={f"S{i}": 100.0 for i in range(n_sym)})


def _ranked_signal(n_sym: int = 10) -> ConstSignal:
    return ConstSignal({f"S{i}": float(i) for i in range(n_sym)})


def test_decile_neutral_book_is_balanced_long_short(ctx) -> None:
    config = DailyBookConfig(
        book="decile_neutral", decile=0.2, max_symbol_abs_weight=1.0, min_order_notional=1.0
    )
    result = run_daily_cycle(ctx, _ranked_signal(), *_decile_panels(), config)
    tw = result.target_weights
    assert tw.abs().sum() == pytest.approx(1.0)  # raw gross 1.0
    assert tw.sum() == pytest.approx(0.0)  # market-neutral by balanced legs
    longs = {o.symbol for o in result.submitted_orders if o.qty > 0}
    shorts = {o.symbol for o in result.submitted_orders if o.qty < 0}
    assert longs == {"S8", "S9"}  # top decile by score
    assert shorts == {"S0", "S1"}  # bottom decile


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(book="bogus"), "unknown book"),
        (dict(book="decile_neutral", decile=0.7), "decile"),
        (dict(book="directional", position_size=0.0), "position_size"),
        (dict(book="directional", position_size=0.1, decision_every=0), "decision_every"),
    ],
)
def test_book_config_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        DailyBookConfig(**kwargs)


def test_monthly_cadence_refreshes_then_holds(ctx) -> None:
    signal = _ranked_signal()
    config = DailyBookConfig(
        book="decile_neutral",
        decile=0.2,
        decision_every=5,
        max_symbol_abs_weight=1.0,
        min_order_notional=1.0,
    )
    # Session 1 (bar 26): first ever -> refresh, book established.
    r1 = run_daily_cycle(ctx, signal, *_decile_panels(n=26), config)
    assert len(r1.submitted_orders) == 4
    assert ctx.store.load().last_refresh_bar == r1.decision_bar

    # Sessions 2-5 (bars 27-30): cadence has not elapsed -> hold, anchor frozen.
    for n in range(27, 31):
        rn = run_daily_cycle(ctx, signal, *_decile_panels(n=n), config)
        assert rn.submitted_orders == [], f"expected a hold at bar index {n}"
        assert ctx.store.load().last_refresh_bar == r1.decision_bar

    # Session 6 (bar 31): 5 sessions elapsed -> refresh, anchor advances.
    r6 = run_daily_cycle(ctx, signal, *_decile_panels(n=31), config)
    assert ctx.store.load().last_refresh_bar == r6.decision_bar
    assert r6.decision_bar != r1.decision_bar


def test_decile_book_exits_a_name_that_leaves_the_decile(ctx) -> None:
    config = DailyBookConfig(
        book="decile_neutral", decile=0.2, max_symbol_abs_weight=1.0, min_order_notional=1.0
    )
    # Cycle 1 on 10 names: S8,S9 long; S0,S1 short.
    run_daily_cycle(ctx, _ranked_signal(10), *_decile_panels(n=26, n_sym=10), config)
    # Cycle 2: S9's momentum reshuffles to mid-pack — it drops out of the top
    # decile, so the full-rebalance book must exit the held long, not carry it.
    new_scores = {f"S{i}": float(i) for i in range(10)}
    new_scores["S9"] = 4.5  # was the top score; now mid -> out of the decile
    r2 = run_daily_cycle(ctx, ConstSignal(new_scores), *_decile_panels(n=27, n_sym=10), config)
    closed = {o.symbol: o.qty for o in r2.submitted_orders}
    assert "S9" in closed and closed["S9"] < 0  # sold to flat (exited the long)
    assert r2.target_weights["S9"] == pytest.approx(0.0)


def test_state_v1_migrates_to_v2(tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "positions": {"AAA": 3.0},
                "cash": 500.0,
                "pending_orders": [],
                "pending_decision_bar": None,
                "last_settled_bar": "2026-07-06",
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    state = StateStore(path).load()
    assert state.schema_version == STATE_SCHEMA_VERSION == 2
    assert state.last_refresh_bar is None  # a v1 book loads unanchored
    assert state.positions == {"AAA": 3.0}  # positions preserved, not reset
    assert state.last_settled_bar == "2026-07-06"
