"""The daily driver (SPEC §7.7): fetch → score → construct → decide, wired.

Pins the seams the driver owns: panel assembly from the incremental fetch
(fail-loud, N7), the settle-then-decide ordering across bars, same-bar
restart resumption, held-weight hysteresis through the *online* band, the
participation gate on ADV, and whole-share order sizing for OPG venues.
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields as dataclass_fields
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from prism.live import (
    DailyBookConfig,
    DailyCycleResult,
    LiveLoopContext,
    SafetyConfig,
    StateStore,
    book_concordance,
    fetch_universe_panels,
    read_concordance_ledger,
    read_fills_ledger,
    read_regime_ledger,
    run_daily_cycle,
    targets_to_orders,
)
from prism.live.state import STATE_SCHEMA_VERSION, LoopState
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
    config = DailyBookConfig(book="decile_neutral", decile=0.2, max_symbol_abs_weight=1.0, min_order_notional=1.0)
    result = run_daily_cycle(ctx, _ranked_signal(), *_decile_panels(), config)
    tw = result.target_weights
    assert tw.abs().sum() == pytest.approx(1.0)  # raw gross 1.0
    assert tw.sum() == pytest.approx(0.0)  # market-neutral by balanced legs
    longs = {o.symbol for o in result.submitted_orders if o.qty > 0}
    shorts = {o.symbol for o in result.submitted_orders if o.qty < 0}
    assert longs == {"S8", "S9"}  # top decile by score
    assert shorts == {"S0", "S1"}  # bottom decile


def test_inverse_vol_book_honors_score_signs(ctx) -> None:
    """Trend construct path in the daily driver (docs/trend_design.md §2)."""
    import numpy as np

    n, vol_bars = 80, 20
    idx = pd.date_range("2026-01-02", periods=n, freq="B", tz="America/New_York")
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        {
            "LOW": 100.0 * np.exp(np.cumsum(0.001 + 0.005 * rng.normal(size=n))),
            "HIGH": 100.0 * np.exp(np.cumsum(-0.001 + 0.03 * rng.normal(size=n))),
        },
        index=idx,
    )
    volume = pd.DataFrame(1e6, index=idx, columns=close.columns)
    config = DailyBookConfig(
        book="inverse_vol",
        vol_ewma_bars=vol_bars,
        max_gross=1.0,
        max_symbol_abs_weight=1.0,
        min_order_notional=1.0,
    )
    # LOW long, HIGH short — signs from scores; LOW should get larger |w|.
    result = run_daily_cycle(ctx, ConstSignal({"LOW": 0.2, "HIGH": -0.2}, required=5), close, volume, config)
    tw = result.target_weights
    assert tw["LOW"] > 0.0
    assert tw["HIGH"] < 0.0
    assert tw.abs().sum() == pytest.approx(1.0, rel=1e-5)
    assert tw["LOW"] > abs(tw["HIGH"])  # lower vol → larger absolute weight


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(book="bogus"), "unknown book"),
        (dict(book="decile_neutral", decile=0.7), "decile"),
        (dict(book="directional", position_size=0.0), "position_size"),
        (dict(book="directional", position_size=0.1, decision_every=0), "decision_every"),
        (dict(book="inverse_vol", vol_ewma_bars=1), "vol_ewma_bars"),
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
    config = DailyBookConfig(book="decile_neutral", decile=0.2, max_symbol_abs_weight=1.0, min_order_notional=1.0)
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
    # v1 -> v2 (the cadence anchor) then v2 -> v3 (the canonical book): two
    # declared hops, no silent reset of a real position.
    assert state.schema_version == STATE_SCHEMA_VERSION == 3
    assert state.last_refresh_bar is None  # a v1 book loads unanchored
    assert state.positions == {"AAA": 3.0}  # positions preserved, not reset
    assert state.last_settled_bar == "2026-07-06"
    assert state.fills_marker == "2026-07-06"  # initialized from last_settled_bar (D4)


# ---------------------------------------------------------------------------
# Book-concordance telemetry: the held book vs the last refresh targets
# ---------------------------------------------------------------------------


def test_concordance_telemetry_tracks_prior_refresh(ctx, tmp_path) -> None:
    ctx.targets_ledger = tmp_path / "targets.jsonl"
    ctx.concordance_ledger = tmp_path / "concordance.jsonl"
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    close, vol = _panels(30)
    r1 = run_daily_cycle(ctx, signal, close, vol, _CONFIG)
    assert r1.concordance is None  # first refresh: no prior baseline to track

    close2, vol2 = _panels(31)
    r2 = run_daily_cycle(ctx, signal, close2, vol2, _CONFIG)
    assert r2.concordance is not None
    assert r2.concordance["refresh_bar"] == r1.decision_bar
    # Fills executed exactly at reference on flat prices: the held book IS the
    # decided book, and the telemetry must say so.
    assert r2.concordance["active_share"] == pytest.approx(0.0, abs=1e-9)
    assert r2.concordance["gross_ratio"] == pytest.approx(1.0)
    rows = read_concordance_ledger(ctx.concordance_ledger)
    assert list(rows["decision_bar"]) == [r2.decision_bar]

    # A same-bar rerun resumes the persisted decision and never duplicates the
    # concordance row.
    run_daily_cycle(ctx, signal, close2, vol2, _CONFIG)
    assert len(read_concordance_ledger(ctx.concordance_ledger)) == 1


def test_concordance_sees_a_partial_fill_book(ctx, tmp_path) -> None:
    # One of two decided orders never fills: the held book is HALF the decided
    # book and the telemetry must surface it (the 2026-07-08 22/101 case in
    # miniature) — this is invisible to the equity monitor.
    ctx.targets_ledger = tmp_path / "targets.jsonl"
    ctx.concordance_ledger = tmp_path / "concordance.jsonl"
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    close, vol = _panels(30)
    r1 = run_daily_cycle(ctx, signal, close, vol, _CONFIG)
    bar1 = r1.decision_bar
    ctx.broker.no_fill.add(f"{bar1}:BBB")  # the short leg never prints

    close2, vol2 = _panels(31)
    r2 = run_daily_cycle(ctx, signal, close2, vol2, _CONFIG)
    assert r2.concordance is not None
    # Decided gross 0.2 (±0.1); only the long leg filled -> gross_held 0.1.
    assert r2.concordance["gross_held"] == pytest.approx(0.1, rel=1e-6)
    assert r2.concordance["gross_target"] == pytest.approx(0.2, rel=1e-6)
    assert r2.concordance["active_share"] == pytest.approx(0.05, rel=1e-6)


# ---------------------------------------------------------------------------
# book_concordance (pure)
# ---------------------------------------------------------------------------


def test_book_concordance_identical_books() -> None:
    w = pd.Series({"AAA": 0.1, "BBB": -0.1})
    read = book_concordance(w, w.copy())
    assert read["active_share"] == pytest.approx(0.0)
    assert read["weight_corr"] == pytest.approx(1.0)
    assert read["gross_ratio"] == pytest.approx(1.0)
    assert read["n_held"] == read["n_target"] == 2


def test_book_concordance_disjoint_books() -> None:
    held = pd.Series({"AAA": 0.1})
    target = pd.Series({"BBB": 0.1})
    read = book_concordance(held, target)
    assert read["active_share"] == pytest.approx(0.1)  # 0.5 * (0.1 + 0.1)
    assert read["weight_corr"] == pytest.approx(-1.0)
    assert read["gross_ratio"] == pytest.approx(1.0)


def test_book_concordance_degenerate_sides() -> None:
    held = pd.Series(dtype=float)
    target = pd.Series({"AAA": 0.1})
    read = book_concordance(held, target)
    assert read["weight_corr"] is None  # a flat side has no correlation
    assert read["gross_held"] == 0.0
    assert read["gross_ratio"] == 0.0
    assert read["active_share"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# The SPEC §7.7 regime step: every-cycle telemetry + the unarmed gross hook
# ---------------------------------------------------------------------------


class RecordingRegime:
    """Deterministic provider: records the bars consulted, returns a canned
    read (clean by default; pass ``failures`` for a dirty one)."""

    def __init__(self, failures: list[dict] | None = None) -> None:
        self.failures = failures or []
        self.calls: list[str] = []

    def __call__(self, decision_bar: str) -> dict:
        self.calls.append(decision_bar)
        return {
            "decision_bar": decision_bar,
            "blocks": {"curve": {"level": 4.0, "slope": 0.1, "curvature": -0.2, "asof": decision_bar}},
            "failures": list(self.failures),
        }


class RecordingScale:
    """Gross-scale hook that records when it fires."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[str] = []

    def __call__(self, state: dict) -> float:
        self.calls.append(state["decision_bar"])
        return self.value


_DECILE_CONFIG = DailyBookConfig(book="decile_neutral", decile=0.2, max_symbol_abs_weight=1.0, min_order_notional=1.0)


def test_regime_read_every_cycle_including_non_refresh(ctx, tmp_path) -> None:
    ctx.regime_ledger = tmp_path / "regime.jsonl"
    provider = RecordingRegime()
    config = DailyBookConfig(position_size=0.1, decision_every=5, min_order_notional=1.0)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    r1 = run_daily_cycle(ctx, signal, *_panels(30), config, regime=provider)
    r2 = run_daily_cycle(ctx, signal, *_panels(31), config, regime=provider)  # cadence: a hold

    assert provider.calls == [r1.decision_bar, r2.decision_bar]  # non-refresh sessions read too
    assert r2.submitted_orders == [] and r2.regime is not None
    assert r2.regime["blocks"]["curve"]["level"] == 4.0
    rows = read_regime_ledger(ctx.regime_ledger)
    assert list(rows["decision_bar"]) == [r1.decision_bar, r2.decision_bar]
    assert list(rows["clean"]) == [True, True]
    assert set(rows.columns) >= {"decision_bar", "clean", "gross_scale", "blocks", "failures"}
    assert rows["gross_scale"].isna().all()  # the action hook is unarmed


def test_regime_read_and_ledgered_on_halted_cycle(ctx, tmp_path) -> None:
    ctx.regime_ledger = tmp_path / "regime.jsonl"
    kill = tmp_path / "KILL_SWITCH"
    kill.touch()
    provider = RecordingRegime()
    result = run_daily_cycle(
        ctx,
        ConstSignal({"AAA": 1.0, "BBB": -1.0}),
        *_panels(),
        _CONFIG,
        safety=SafetyConfig(kill_switch=kill),
        regime=provider,
    )
    assert result.halted is not None and result.submitted_orders == []
    assert result.regime is not None and provider.calls == [result.decision_bar]
    rows = read_regime_ledger(ctx.regime_ledger)  # the halted session still counts on the clock
    assert list(rows["decision_bar"]) == [result.decision_bar]


def test_raising_provider_is_contained_named_and_loud(ctx, tmp_path, caplog) -> None:
    def broken(decision_bar: str) -> dict:
        raise RuntimeError("transport down")

    baseline_ctx = LiveLoopContext(
        store=StateStore(tmp_path / "b" / "state.json"),
        broker=FakeBroker(),
        fills_ledger=tmp_path / "b" / "fills.jsonl",
    )
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    baseline = run_daily_cycle(baseline_ctx, signal, *_panels(), _CONFIG)
    with caplog.at_level(logging.WARNING, logger="prism.live.daily"):
        result = run_daily_cycle(ctx, signal, *_panels(), _CONFIG, regime=broken)

    # Telemetry failure never touches the book: the cycle completes with the
    # exact orders the unwired loop decides.
    assert [(o.symbol, o.qty) for o in result.submitted_orders] == [
        (o.symbol, o.qty) for o in baseline.submitted_orders
    ]
    assert result.regime["failures"] == [{"block": "provider", "error": "RuntimeError"}]
    assert any("REGIME PROVIDER FAILED" in r.getMessage() for r in caplog.records)


def test_empty_provider_result_is_a_named_provider_failure(ctx) -> None:
    result = run_daily_cycle(ctx, ConstSignal({"AAA": 1.0, "BBB": -1.0}), *_panels(), _CONFIG, regime=lambda bar: {})
    assert result.regime["failures"] == [{"block": "provider", "error": "RuntimeError"}]
    assert len(result.submitted_orders) == 2  # the cycle still decided


# ---------------------------------------------------------------------------
# The panel the cycle decides on is recorded and cited by the refresh row
# ---------------------------------------------------------------------------


def test_cycle_records_its_panel_and_the_targets_row_cites_it(tmp_path) -> None:
    from prism.live.panel_log import read_index, reconstruct_panel

    ctx = _full_ctx(tmp_path)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    close, vol = _panels(30)
    result = run_daily_cycle(ctx, signal, close, vol, _CONFIG)
    records = read_index(tmp_path)
    assert len(records) == 1 and records[0].kind == "base"
    # The result omits the wall-clock stamp so identical cycles compare equal.
    assert result.panel == {k: v for k, v in records[0].to_row().items() if k != "recorded_at"}
    row = json.loads((tmp_path / "targets.jsonl").read_text().splitlines()[0])
    assert row["panel_sha256"] == records[0].sha256 and row["panel_file"] == records[0].file
    assert set(row) >= {"refresh_bar", "decision_bar", "equity", "targets", "reference_prices"}
    panel = reconstruct_panel(tmp_path, result.decision_bar)
    pd.testing.assert_frame_equal(panel["close"], close, check_freq=False, check_names=False)
    pd.testing.assert_frame_equal(panel["volume"], vol, check_freq=False, check_names=False)
    # The next session is a delta against the first, and a same-bar rerun adds nothing.
    close2, vol2 = _panels(31)
    run_daily_cycle(ctx, signal, close2, vol2, _CONFIG)
    run_daily_cycle(ctx, signal, close2, vol2, _CONFIG)
    records = read_index(tmp_path)
    assert [r.kind for r in records] == ["base", "delta"]
    second = reconstruct_panel(tmp_path, records[1].decision_bar)["close"]
    pd.testing.assert_frame_equal(second, close2, check_freq=False, check_names=False)


def test_cycle_records_nothing_without_a_run_directory(ctx, tmp_path) -> None:
    result = run_daily_cycle(ctx, ConstSignal({"AAA": 1.0, "BBB": -1.0}), *_panels(), _CONFIG)
    assert result.panel is None
    assert not (tmp_path / "panels").exists()


def test_explicit_run_dir_wins_over_the_targets_ledger_location(ctx, tmp_path) -> None:
    from prism.live.panel_log import read_index

    ctx.targets_ledger = tmp_path / "ledgers" / "targets.jsonl"
    elsewhere = tmp_path / "elsewhere"
    result = run_daily_cycle(ctx, ConstSignal({"AAA": 1.0, "BBB": -1.0}), *_panels(), _CONFIG, run_dir=elsewhere)
    assert result.panel is not None
    assert len(read_index(elsewhere)) == 1
    assert not (tmp_path / "ledgers" / "panels").exists()


def test_a_same_bar_rerun_that_sees_different_bars_keeps_the_recorded_panel_and_resumes(tmp_path, caplog) -> None:
    """The write-ahead step already ran for this bar, so a re-run resumes the persisted orders and decides
    nothing from the panel it fetched now. The recorded panel stays the decision's provenance, the index gets
    no second row for the bar, and the mismatch is logged at ERROR with both digests."""
    from prism.live.panel_log import read_index

    ctx = _full_ctx(tmp_path)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    close, vol = _panels(30)
    first = run_daily_cycle(ctx, signal, close, vol, _CONFIG)
    at_venue = ctx.broker.submitted_order_ids()
    with caplog.at_level(logging.ERROR, logger="prism.live.daily"):
        second = run_daily_cycle(ctx, signal, close * 1.01, vol, _CONFIG)  # same bar, different bars seen
    records = read_index(tmp_path)
    assert len(records) == 1
    assert first.panel is not None and second.panel is not None
    assert second.panel["sha256"] == first.panel["sha256"] == records[0].sha256
    state = ctx.store.load()
    assert state is not None and len(state.pending_orders) == 2  # the first decision stands untouched
    assert ctx.broker.submitted_order_ids() == at_venue  # nothing new reached the venue
    assert [o.client_order_id for o in second.submitted_orders] == [o.client_order_id for o in first.submitted_orders]
    errors = [
        r for r in caplog.records if r.levelno == logging.ERROR and "differs from the one recorded" in r.getMessage()
    ]
    assert len(errors) == 1 and records[0].sha256 in errors[0].getMessage()


def test_a_pre_open_rerun_with_a_shifted_window_submits_the_persisted_decision(tmp_path, caplog) -> None:
    """A nightly that died before its first submit is re-run next morning on a window one bar shorter (the
    fetch window moved) and otherwise identical bars. The persisted decision is submitted in full and the index
    keeps the panel it was decided on."""
    from prism.live.panel_log import read_index

    ctx = _full_ctx(tmp_path)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    close, vol = _panels(30)
    ctx.broker.fail_after_n_submits = 0  # the venue is dark: the write-ahead persists, nothing is sent
    with pytest.raises(ConnectionError):
        run_daily_cycle(ctx, signal, close, vol, _CONFIG)
    persisted = ctx.store.load()
    assert persisted is not None and len(persisted.pending_orders) == 2
    assert ctx.broker.submitted_order_ids() == set()
    ctx.broker.fail_after_n_submits = None
    with caplog.at_level(logging.ERROR, logger="prism.live.daily"):
        result = run_daily_cycle(ctx, signal, close.iloc[1:], vol.iloc[1:], _CONFIG)
    assert ctx.broker.submitted_order_ids() == {o.client_order_id for o in persisted.pending_orders}
    assert {o.client_order_id for o in result.submitted_orders} == ctx.broker.submitted_order_ids()
    records = read_index(tmp_path)
    assert len(records) == 1 and records[0].n_bars == 30  # the recorded panel, not the 29-bar refetch
    assert any("differs from the one recorded" in r.getMessage() for r in caplog.records)


def test_a_rerun_of_a_hold_session_with_a_revised_panel_holds_again(tmp_path, caplog) -> None:
    """Off the refresh cadence the cycle holds the filled book and the write-ahead step persists that (no
    orders) for the bar. A re-run with a revised panel decides nothing new either, so it is tolerated too."""
    from prism.live.panel_log import read_index

    config = DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=5)
    ctx = _full_ctx(tmp_path)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    run_daily_cycle(ctx, signal, *_panels(30), config)  # the refresh
    close, vol = _panels(31)
    run_daily_cycle(ctx, signal, close, vol, config)  # settles, then holds
    state = ctx.store.load()
    assert state is not None and state.pending_orders == []
    assert state.pending_decision_bar == str(close.index[-1].date())
    with caplog.at_level(logging.ERROR, logger="prism.live.daily"):
        result = run_daily_cycle(ctx, signal, close * 1.01, vol, config)
    assert result.submitted_orders == []
    assert [r.kind for r in read_index(tmp_path)] == ["base", "delta"]
    assert any("differs from the one recorded" in r.getMessage() for r in caplog.records)


def test_a_conflicting_panel_with_no_pending_decision_still_refuses(tmp_path) -> None:
    """Without a persisted decision for the bar a fresh decision would be made on the new content: the
    conflict raises before any state moves, whether the store is empty or holds an earlier bar's decision."""
    from prism.live.panel_log import PanelLogConflict, record_panel

    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    close, vol = _panels(30)
    bar = str(close.index[-1].date())
    ctx = _full_ctx(tmp_path)
    record_panel(tmp_path, bar, {"close": close * 1.01, "volume": vol})
    with pytest.raises(PanelLogConflict):
        run_daily_cycle(ctx, signal, close, vol, _CONFIG)
    assert ctx.store.load() is None and ctx.broker.submitted_order_ids() == set()

    other = tmp_path / "other"
    other.mkdir()
    ctx = _full_ctx(other)
    ctx.store.save(LoopState(pending_decision_bar=str(close.index[-2].date())))  # an earlier bar's hold
    record_panel(other, bar, {"close": close * 1.01, "volume": vol})
    with pytest.raises(PanelLogConflict):
        run_daily_cycle(ctx, signal, close, vol, _CONFIG)
    assert ctx.broker.submitted_order_ids() == set()


def _full_ctx(root) -> LiveLoopContext:
    return LiveLoopContext(
        store=StateStore(root / "state.json"),
        broker=FakeBroker(),
        fills_ledger=root / "fills.jsonl",
        equity_ledger=root / "equity.jsonl",
        targets_ledger=root / "targets.jsonl",
        unfilled_ledger=root / "unfilled.jsonl",
        concordance_ledger=root / "concordance.jsonl",
        regime_ledger=root / "regime.jsonl",
    )


def test_bit_identity_with_both_regime_params_none(tmp_path) -> None:
    # The certified B1 book's behavior must be bit-identical when the feature
    # is off: every DailyCycleResult field and every ledger byte must match a
    # run that never heard of the seam.
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    results: dict[str, list[DailyCycleResult]] = {}
    for name, kwargs in (("bare", {}), ("wired_off", {"regime": None, "regime_gross_scale": None})):
        ctx = _full_ctx(tmp_path / name)
        results[name] = [
            run_daily_cycle(ctx, signal, *_panels(30), _CONFIG, **kwargs),
            run_daily_cycle(ctx, signal, *_panels(31), _CONFIG, **kwargs),
        ]
    for bare, off in zip(results["bare"], results["wired_off"]):
        for field in dataclass_fields(DailyCycleResult):
            left, right = getattr(bare, field.name), getattr(off, field.name)
            if isinstance(left, pd.Series):
                pd.testing.assert_series_equal(left, right)
            else:
                assert left == right, field.name
    for ledger in ("state.json", "fills.jsonl", "equity.jsonl", "targets.jsonl", "concordance.jsonl"):
        assert (tmp_path / "bare" / ledger).read_bytes() == (tmp_path / "wired_off" / ledger).read_bytes()
    for absent in ("regime.jsonl", "unfilled.jsonl"):  # nothing regime-shaped may exist when off
        assert not (tmp_path / "bare" / absent).exists()
        assert not (tmp_path / "wired_off" / absent).exists()


def test_gross_scale_applied_on_clean_refresh_and_recorded(ctx, tmp_path) -> None:
    ctx.regime_ledger = tmp_path / "regime.jsonl"
    hook = RecordingScale(0.5)
    result = run_daily_cycle(
        ctx,
        _ranked_signal(),
        *_decile_panels(),
        _DECILE_CONFIG,
        regime=RecordingRegime(),
        regime_gross_scale=hook,
    )
    assert hook.calls == [result.decision_bar]
    assert result.regime_scale == pytest.approx(0.5)
    assert result.target_weights.abs().sum() == pytest.approx(0.5)  # max_gross 1.0 scaled to 0.5
    rows = read_regime_ledger(ctx.regime_ledger)
    assert rows["gross_scale"].iloc[0] == pytest.approx(0.5)
    assert bool(rows["clean"].iloc[0]) is True


@pytest.mark.parametrize("raw_scale, applied", [(1.7, 1.0), (-0.3, 0.0)])
def test_gross_scale_clamped_to_the_configured_gross(ctx, raw_scale, applied) -> None:
    result = run_daily_cycle(
        ctx,
        _ranked_signal(),
        *_decile_panels(),
        _DECILE_CONFIG,
        regime=RecordingRegime(),
        regime_gross_scale=RecordingScale(raw_scale),
    )
    assert result.regime_scale == pytest.approx(applied)
    assert result.target_weights.abs().sum() == pytest.approx(applied * 1.0)
    if applied == 0.0:
        assert result.submitted_orders == []  # a fully de-grossed fresh book opens nothing


def test_gross_scale_not_applied_on_dirty_telemetry(ctx, caplog) -> None:
    hook = RecordingScale(0.5)
    dirty = RecordingRegime(failures=[{"block": "curve", "error": "RegimeFetchError"}])
    with caplog.at_level(logging.WARNING, logger="prism.live.daily"):
        result = run_daily_cycle(
            ctx, _ranked_signal(), *_decile_panels(), _DECILE_CONFIG, regime=dirty, regime_gross_scale=hook
        )
    assert hook.calls == []  # the action de-arms on a not-clean read
    assert result.regime_scale is None
    assert result.target_weights.abs().sum() == pytest.approx(1.0)  # configured gross, no silent de-gross
    assert any("NOT applied" in r.getMessage() for r in caplog.records)


def test_gross_scale_fires_only_on_refresh_sessions(ctx) -> None:
    hook = RecordingScale(0.5)
    provider = RecordingRegime()
    config = DailyBookConfig(
        book="decile_neutral",
        decile=0.2,
        decision_every=5,
        max_symbol_abs_weight=1.0,
        min_order_notional=1.0,
    )
    r1 = run_daily_cycle(
        ctx, _ranked_signal(), *_decile_panels(n=26), config, regime=provider, regime_gross_scale=hook
    )
    r2 = run_daily_cycle(
        ctx, _ranked_signal(), *_decile_panels(n=27), config, regime=provider, regime_gross_scale=hook
    )
    assert r1.regime_scale == pytest.approx(0.5)
    assert hook.calls == [r1.decision_bar]  # the hold session reads telemetry but never re-sizes
    assert provider.calls == [r1.decision_bar, r2.decision_bar]
    assert r2.regime_scale is None and r2.submitted_orders == []


def test_gross_scale_requires_the_regime_provider(ctx) -> None:
    with pytest.raises(ValueError, match="regime_gross_scale requires"):
        run_daily_cycle(
            ctx,
            ConstSignal({"AAA": 1.0, "BBB": -1.0}),
            *_panels(),
            _CONFIG,
            regime_gross_scale=RecordingScale(0.5),
        )


@pytest.mark.parametrize("book", ["decile_neutral", "inverse_vol"])
def test_gross_scale_hook_is_refused_above_one_times_gross_before_any_state_moves(tmp_path, book) -> None:
    # The hook multiplies max_gross by a factor in [0, 1]; above 1x a "de-gross" could size a book above the
    # unscaled one (the inverse-vol book is scaled to its ceiling outright), so it is refused outright.
    ctx = _full_ctx(tmp_path)
    hook = RecordingScale(1.0)
    config = DailyBookConfig(book=book, decile=0.2, max_gross=1.5, max_symbol_abs_weight=1.0, min_order_notional=1.0)
    with pytest.raises(ValueError, match=r"refused with max_gross=1\.5 > 1\.0"):
        run_daily_cycle(
            ctx, _ranked_signal(), *_decile_panels(), config, regime=RecordingRegime(), regime_gross_scale=hook
        )
    assert hook.calls == []
    assert ctx.store.load() is None and not (tmp_path / "panels").exists()
    assert ctx.broker.submitted_order_ids() == set()
    # At 1x the hook stays available, unchanged.
    at_one = replace(config, max_gross=1.0)
    result = run_daily_cycle(
        ctx, _ranked_signal(), *_decile_panels(), at_one, regime=RecordingRegime(), regime_gross_scale=hook
    )
    assert result.regime_scale == pytest.approx(1.0)


def test_non_finite_gross_scale_raises(ctx) -> None:
    with pytest.raises(ValueError, match="regime_gross_scale returned"):
        run_daily_cycle(
            ctx,
            _ranked_signal(),
            *_decile_panels(),
            _DECILE_CONFIG,
            regime=RecordingRegime(),
            regime_gross_scale=RecordingScale(float("nan")),
        )


def test_regime_ledger_idempotent_within_a_bar(ctx, tmp_path) -> None:
    ctx.regime_ledger = tmp_path / "regime.jsonl"
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    close, volume = _panels()
    run_daily_cycle(ctx, signal, close, volume, _CONFIG, regime=RecordingRegime())
    run_daily_cycle(ctx, signal, close, volume, _CONFIG, regime=RecordingRegime())  # same-bar resume
    assert len(read_regime_ledger(ctx.regime_ledger)) == 1  # one session, one row on the clock


# ---------------------------------------------------------------------------
# The last bar must be a completed session bar (require_published_panel)
# ---------------------------------------------------------------------------


def _panel_ending(date: str, *, tz: str | None = None) -> pd.DataFrame:
    from prism.live.daily import EXCHANGE_TZ

    index = pd.DatetimeIndex(pd.bdate_range(end=date, periods=5))
    if tz is not None:
        index = index.tz_localize(EXCHANGE_TZ)  # the Alpaca source's midnight-ET contract
    return pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=index)


def _et(when: str) -> pd.Timestamp:
    return pd.Timestamp(when, tz="America/New_York")


def test_the_nightly_hour_passes_on_the_day_it_decides() -> None:
    """18:30 PT is 21:30 ET: the regular 16:00 close plus the 30-minute delay is long past."""
    from prism.live.daily import PartialBarError, require_published_panel

    report = require_published_panel(_panel_ending("2026-09-30"), now=_et("2026-09-30 21:30"))
    assert report == {
        "last_bar": "2026-09-30",
        "published_at": "2026-09-30T16:30:00-04:00",
        "now": "2026-09-30T21:30:00-04:00",
        "partial": False,
    }
    # Exactly at publication counts as published; a minute before does not.
    assert require_published_panel(_panel_ending("2026-09-30"), now=_et("2026-09-30 16:30"))["partial"] is False
    with pytest.raises(PartialBarError, match="still in progress"):
        require_published_panel(_panel_ending("2026-09-30"), now=_et("2026-09-30 16:29"))


def test_a_market_hours_catch_up_is_refused_and_a_rehearsal_warns(caplog) -> None:
    """The 2026-09-25 rehearsal's shape: a 10:00 ET run holding today's in-progress bar."""
    from prism.live.daily import PartialBarError, require_published_panel
    from prism.live.safety import SafetyViolation

    with pytest.raises(PartialBarError, match="2026-09-30 is still in progress") as refusal:
        require_published_panel(_panel_ending("2026-09-30"), now=_et("2026-09-30 10:00"))
    assert isinstance(refusal.value, SafetyViolation) and "--no-submit" in str(refusal.value)
    with caplog.at_level(logging.WARNING, logger="prism.live.daily"):
        report = require_published_panel(_panel_ending("2026-09-30"), now=_et("2026-09-30 10:00"), allow_partial=True)
    assert report["partial"] is True
    assert any("PARTIAL BAR" in r.getMessage() and "rehearsal" in r.getMessage() for r in caplog.records)


def test_an_early_close_publishes_at_13_30_et() -> None:
    """The day after Thanksgiving closes at 13:00; the table knows, so the rail does."""
    from prism.live.daily import PartialBarError, require_published_panel

    with pytest.raises(PartialBarError, match="closes at 13:00 ET"):
        require_published_panel(_panel_ending("2026-11-27"), now=_et("2026-11-27 13:15"))
    report = require_published_panel(_panel_ending("2026-11-27"), now=_et("2026-11-27 13:45"))
    assert report["partial"] is False and report["published_at"] == "2026-11-27T13:30:00-05:00"
    # A regular day at the same hour is still in progress.
    with pytest.raises(PartialBarError):
        require_published_panel(_panel_ending("2026-11-30"), now=_et("2026-11-30 13:45"))


def test_a_last_bar_from_a_previous_session_passes_at_any_hour() -> None:
    """A holiday or pre-open run holds yesterday's completed bar; same-bar resume handles the rest."""
    from prism.live.daily import require_published_panel

    for when in ("2026-09-30 06:00", "2026-09-30 10:00", "2026-09-30 15:59", "2026-10-03 12:00"):
        report = require_published_panel(_panel_ending("2026-09-29"), now=_et(when))
        assert report["partial"] is False and report["last_bar"] == "2026-09-29"


def test_a_non_session_last_bar_is_refused_even_on_a_rehearsal() -> None:
    from prism.live.daily import PartialBarError, require_published_panel

    saturday = pd.DataFrame({"AAA": 1.0}, index=pd.DatetimeIndex(["2026-09-25", "2026-09-26"]))
    with pytest.raises(PartialBarError, match="not an exchange session"):
        require_published_panel(saturday, now=_et("2026-09-30 21:30"), allow_partial=True)
    holiday = pd.DataFrame({"AAA": 1.0}, index=pd.DatetimeIndex(["2026-11-25", "2026-11-26"]))
    with pytest.raises(PartialBarError, match="holiday"):
        require_published_panel(holiday, now=_et("2026-11-27 21:30"))


def test_a_bar_dated_after_the_clock_is_refused_outright() -> None:
    from prism.live.daily import PartialBarError, require_published_panel

    with pytest.raises(PartialBarError, match="from the future"):
        require_published_panel(_panel_ending("2026-10-01"), now=_et("2026-09-30 21:30"), allow_partial=True)


def test_tz_aware_midnight_et_bars_and_other_clocks_are_read_in_exchange_time() -> None:
    """The live bar source indexes at midnight ET; the clock may arrive in any zone."""
    from prism.live.daily import require_published_panel

    panel = _panel_ending("2026-09-30", tz="America/New_York")
    assert panel.index.tz is not None
    # 01:30 UTC on 10-01 is 21:30 ET on 09-30: published.
    report = require_published_panel(panel, now=pd.Timestamp("2026-10-01 01:30", tz="UTC"))
    assert report["partial"] is False and report["now"] == "2026-09-30T21:30:00-04:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        require_published_panel(panel, now=pd.Timestamp("2026-09-30 21:30"))
    # The default clock is the wall clock; a bar from 2020 is published on any sane clock.
    assert require_published_panel(_panel_ending("2020-01-03", tz="America/New_York"))["partial"] is False


# ---------------------------------------------------------------------------
# The live monitor in the cycle: shadow rows, tracking, and containment
# ---------------------------------------------------------------------------


def _session_panels(n: int, moves: dict[str, float]):
    """Flat-then-drifting close panel on consecutive exchange sessions (no holiday pairs),
    so every adjacent cycle is a consecutive-session pair for the monitor."""
    import datetime as dt

    from prism.io import sessions

    days = sessions.sessions_between(dt.date(2026, 7, 6), dt.date(2026, 12, 31))[:n]
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York") for d in days])
    close = pd.DataFrame(
        {s: [100.0 * (1.0 + m) ** k for k in range(n)] for s, m in moves.items()}, index=idx
    )
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    return close, vol


def _monitor_ctx(root) -> LiveLoopContext:
    ctx = _full_ctx(root)
    ctx.shadow_ledger = root / "shadow.jsonl"
    return ctx


def test_cycle_records_shadow_rows_and_attaches_the_tracking_read(tmp_path) -> None:
    from prism.live.loop import read_shadow_ledger

    ctx = _monitor_ctx(tmp_path)
    ctx.broker._cash = 1_000_000.0  # whole-share rounding at 10k would distort the held weights
    config = DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=21)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    moves = {"AAA": 0.001, "BBB": -0.0005}
    results = [run_daily_cycle(ctx, signal, *_session_panels(n, moves), config) for n in range(30, 36)]
    rows = read_shadow_ledger(ctx.shadow_ledger)
    # Cycle 1 refreshes (no earlier session); cycle 2 is the transition after that refresh
    # but no book predates its previous bar; cycles 3..6 shadow the cycle-1 book.
    assert list(rows["decision_bar"]) == [r.decision_bar for r in results[2:]]
    assert set(rows["refresh_bar"]) == {results[0].decision_bar}
    assert not rows["transition"].any()
    assert (rows["n_names_used"] == 2).all() and (rows["n_names_missing"] == 0).all()
    # Fills were at reference on the held book, so paper reproduces the shadow to rounding.
    assert np.allclose(rows["paper_return"], rows["shadow_return"], atol=2e-5)
    expected = 0.1 * 0.001 - 0.1 * (-0.0005)
    assert rows["shadow_return"].iloc[-1] == pytest.approx(expected, rel=1e-6)
    last = results[-1].monitor_read
    assert last is not None and last["method"] == "empirical_bernstein" and last["n_gaps"] == 0
    assert last["n"] == 5 and last["tracking"]["n"] == 4 and last["tracking"]["verdict"] == "inconclusive"
    # A same-bar rerun never duplicates a shadow row.
    run_daily_cycle(ctx, signal, *_session_panels(35, moves), config)
    assert len(read_shadow_ledger(ctx.shadow_ledger)) == 4


def test_transition_session_is_flagged_and_set_aside(tmp_path) -> None:
    from prism.live.loop import read_shadow_ledger

    ctx = _monitor_ctx(tmp_path)
    ctx.broker._cash = 1_000_000.0
    config = DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=3)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    moves = {"AAA": 0.001, "BBB": -0.0005}
    results = [run_daily_cycle(ctx, signal, *_session_panels(n, moves), config) for n in range(30, 38)]
    refreshes = [r.decision_bar for r in results if r.submitted_orders or r is results[0]]
    rows = read_shadow_ledger(ctx.shadow_ledger)
    # Session after the second refresh (cycle 4 refreshes: bars 30, 33, 36) is a transition
    # against the FIRST refresh's book, which is the book held through most of it.
    assert results[3].decision_bar in refreshes
    flagged = rows[rows["transition"]]
    assert list(flagged["decision_bar"]) == [results[4].decision_bar, results[7].decision_bar]
    assert flagged["refresh_bar"].iloc[0] == results[0].decision_bar
    tracking = results[-1].monitor_read["tracking"]
    assert tracking["n_transition"] == 2 and tracking["n"] == len(rows) - 2
    # Keyed by the refresh whose next-open fills caused each transition (the transition's previous
    # bar), not by the baseline the row records: results[4] follows results[3], results[7] results[6].
    assert set(tracking["wedge_by_refresh"]) == {results[3].decision_bar, results[6].decision_bar}


def test_gap_session_records_no_shadow_row(tmp_path, caplog) -> None:
    from prism.live.loop import read_shadow_ledger

    ctx = _monitor_ctx(tmp_path)
    config = DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=21)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    moves = {"AAA": 0.001, "BBB": -0.0005}
    run_daily_cycle(ctx, signal, *_session_panels(30, moves), config)
    run_daily_cycle(ctx, signal, *_session_panels(31, moves), config)
    close, vol = _session_panels(40, moves)  # skips eight sessions: the outage shape
    with caplog.at_level(logging.INFO, logger="prism.live.daily"):
        result = run_daily_cycle(ctx, signal, close, vol, config)
    assert not ctx.shadow_ledger.exists() or len(read_shadow_ledger(ctx.shadow_ledger)) == 0
    assert any("a gap is not a day" in r.getMessage() for r in caplog.records)
    assert result.monitor_read["n_gaps"] == 1 and result.monitor_read["n"] == 1


def test_monitor_failure_is_contained_and_loud(tmp_path, caplog, monkeypatch) -> None:
    import prism.live.daily as daily_mod

    def broken(*args, **kwargs):
        raise RuntimeError("ledger unreadable")

    monkeypatch.setattr(daily_mod, "paper_monitor_read", broken)
    ctx = _monitor_ctx(tmp_path)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    with caplog.at_level(logging.WARNING, logger="prism.live.daily"):
        result = run_daily_cycle(ctx, signal, *_panels(), _CONFIG)
    assert len(result.submitted_orders) == 2  # the cycle decided and persisted as usual
    assert result.monitor_read == {
        "verdict": "unavailable",
        "n": 0,
        "error": "RuntimeError",
        "detail": "ledger unreadable",
    }
    assert any("LIVE MONITOR FAILED" in r.getMessage() for r in caplog.records)
    assert len(read_fills_ledger(ctx.fills_ledger)) == 0 and (tmp_path / "equity.jsonl").exists()


def test_shadow_failure_is_contained_and_the_return_read_still_lands(tmp_path, caplog, monkeypatch) -> None:
    import prism.live.daily as daily_mod

    def broken(*args, **kwargs):
        raise ValueError("unpriced names")

    monkeypatch.setattr(daily_mod, "shadow_session_return", broken)
    ctx = _monitor_ctx(tmp_path)
    config = DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=21)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    moves = {"AAA": 0.001, "BBB": -0.0005}
    with caplog.at_level(logging.WARNING, logger="prism.live.daily"):
        results = [run_daily_cycle(ctx, signal, *_session_panels(n, moves), config) for n in range(30, 33)]
    assert any("SHADOW RECORD FAILED" in r.getMessage() for r in caplog.records)
    assert results[-1].monitor_read["verdict"] == "inconclusive" and results[-1].monitor_read["n"] == 2
    assert results[-1].monitor_read["tracking"]["n"] == 0
