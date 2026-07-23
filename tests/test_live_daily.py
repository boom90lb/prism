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
    result = run_daily_cycle(
        ctx, ConstSignal({"LOW": 0.2, "HIGH": -0.2}, required=5), close, volume, config
    )
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


_DECILE_CONFIG = DailyBookConfig(
    book="decile_neutral", decile=0.2, max_symbol_abs_weight=1.0, min_order_notional=1.0
)


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
    result = run_daily_cycle(
        ctx, ConstSignal({"AAA": 1.0, "BBB": -1.0}), *_panels(), _CONFIG, regime=lambda bar: {}
    )
    assert result.regime["failures"] == [{"block": "provider", "error": "RuntimeError"}]
    assert len(result.submitted_orders) == 2  # the cycle still decided


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
