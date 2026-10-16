"""The paper book's volatility scale (``run_daily_cycle(vol_scale=...)``, ``prism.live.vol_sizing``).

Pinned here:

* off is bit-identical: every result field and every ledger byte matches a run
  that never heard of the seam, and no sizing ledger appears;
* on, the scale is the rule applied to the base book rebuilt point in time from
  the panel, checked against an independent reconstruction; the target is the
  scale times the unscaled book, capped at the account ceiling; a rule pinned at
  scale 1 reproduces the unscaled book bit for bit;
* the failure path never sizes above 1 or above the last good scale, which is
  read from the persisted refresh rows only;
* the base book is point in time: a later decision reproduces an earlier one's
  returns on the bars they share, and the decision bar's own close moves only
  the window's last return;
* the record: one sizing row per refresh, written ahead of the write-ahead step,
  idempotent on a same-bar re-run, and a conflicting row raises before anything
  is persisted;
* refusals that keep the scale from running wrongly, all before any state moves;
* the CLI: ``--vol-scale`` defaults off, its refusals come before any venue read,
  and an armed session writes its record into the run directory.

Nothing here opens a socket.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prism.io import sessions
from prism.live import (
    DailyBookConfig,
    DailyCycleResult,
    LiveLoopContext,
    SafetyConfig,
    StateStore,
    run_daily_cycle,
)
from prism.live.risk_profile import resolve_risk_profile
from prism.live.safety import SafetyViolation
from prism.live.vol_sizing import (
    SIZING_LEDGER_FILENAME,
    TARGETS_LEDGER_KEY,
    SizingLedgerConflict,
    VolSizingError,
    base_book_window,
    last_good_scale,
    read_sizing_ledger,
    required_panel_bars,
)
from prism.portfolio.vol_scale import VolScaleRule
from prism.scripts import paper_loop
from prism.scripts.paper_books import VOL_SCALE_PRESETS, resolve_vol_scale
from prism.signal.base import Signal
from tests.test_live_daily import ConstSignal, RecordingRegime
from tests.test_live_loop import FakeBroker
from tests.test_paper_no_submit import (
    BarSourceDouble,
    VenueDouble,
    _established,
    _loop_argv,
    _panels,
    _point_switch,
)

N_SYM = 10
MOMENTUM_BARS = 5
EVERY = 5
RULE = VolScaleRule(target_vol=0.12, lookback_bars=20, max_scale=2.0)
CONFIG = DailyBookConfig(
    book="decile_neutral",
    decile=0.2,
    decision_every=EVERY,
    max_gross=2.0,
    max_symbol_abs_weight=1.0,
    min_order_notional=1.0,
)
#: Real exchange sessions, so the seeded (grid) schedule can place every bar.
DAYS = sessions.sessions_between(dt.date(2025, 1, 2), dt.date(2025, 12, 31))


class TrailingReturn(Signal):
    """The ``MOMENTUM_BARS``-bar trailing return: a stateless rank score, like 12-1 momentum."""

    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    price_scale_invariant = True
    fitted_state = "none"
    estimated_decay_bars = None

    @property
    def horizon_bars(self) -> int:
        return EVERY

    @property
    def required_history(self) -> int:
        return MOMENTUM_BARS + 1

    def fit(self, close, volume=None) -> "TrailingReturn":
        return self

    def score(self, close, volume=None) -> pd.Series:
        scores = close.iloc[-1] / close.iloc[-1 - MOMENTUM_BARS] - 1.0
        scores.index = close.columns
        return scores.astype(float)


SIGNAL = TrailingReturn()


def _walk(n_bars: int = 80, *, seed: int = 7, flat_from: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random-walk closes for ``N_SYM`` names on real sessions; flat from bar ``flat_from`` on."""
    rng = np.random.default_rng(seed)
    moves = rng.normal(0.0, 0.015, size=(n_bars, N_SYM))
    moves[0] = 0.0
    levels = 100.0 * np.exp(np.cumsum(moves, axis=0))
    if flat_from is not None:
        levels[flat_from:] = levels[flat_from - 1]
    index = pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day in DAYS[:n_bars]])
    close = pd.DataFrame(levels, index=index, columns=[f"S{i}" for i in range(N_SYM)])
    return close, pd.DataFrame(1e6, index=index, columns=close.columns)


def _ctx(root: Path, *, targets: bool = True) -> LiveLoopContext:
    root.mkdir(parents=True, exist_ok=True)
    return LiveLoopContext(
        store=StateStore(root / "state.json"),
        broker=FakeBroker(cash=1_000_000.0),
        fills_ledger=root / "fills.jsonl",
        equity_ledger=root / "equity.jsonl",
        targets_ledger=(root / "targets.jsonl") if targets else None,
        unfilled_ledger=root / "unfilled.jsonl",
        concordance_ledger=root / "concordance.jsonl",
    )


def _cycle(ctx, close, volume, through: int, **kwargs) -> DailyCycleResult:
    """One cycle on the panel through bar ``through`` (inclusive)."""
    return run_daily_cycle(
        ctx, SIGNAL, close.iloc[: through + 1], volume.iloc[: through + 1], kwargs.pop("config", CONFIG), **kwargs
    )


def _reference_scale(close: pd.DataFrame, rule: VolScaleRule, every: int) -> tuple[float, float]:
    """The rule on the base book, rebuilt here with plain loops and nothing from the package.

    The book refreshes every ``every`` bars counting back from the last bar;
    each refresh goes long the top ``floor(0.2 * n)`` trailing returns at
    ``+0.5 / k`` and short the bottom ``k`` at ``-0.5 / k``; the return at bar
    *b* is the book held at the close of *b - 1* times the bar's price moves.
    """
    prices = close.to_numpy(dtype=float)
    last = len(prices) - 1
    refreshes = [p for p in range(last + 1) if (last - p) % every == 0]
    k = int(N_SYM * 0.2)

    def book(position: int) -> np.ndarray:
        momentum = prices[position] / prices[position - MOMENTUM_BARS] - 1.0
        order = np.argsort(momentum, kind="stable")
        weights = np.zeros(N_SYM)
        weights[order[:k]] = -0.5 / k
        weights[order[-k:]] = 0.5 / k
        return weights

    returns = []
    for bar in range(last - rule.lookback_bars + 1, last + 1):
        held = book(max(p for p in refreshes if p <= bar - 1))
        returns.append(float(np.sum(held * (prices[bar] / prices[bar - 1] - 1.0))))
    sigma = float(np.std(returns, ddof=1) * math.sqrt(252))
    return min(rule.max_scale, rule.target_vol / sigma), sigma


# ---------------------------------------------------------------------------
# Off is bit-identical
# ---------------------------------------------------------------------------


def test_off_is_bit_identical(tmp_path: Path) -> None:
    close, volume = _walk()
    results: dict[str, list[DailyCycleResult]] = {}
    for name, kwargs in (("bare", {}), ("off", {"vol_scale": None})):
        ctx = _ctx(tmp_path / name)
        results[name] = [_cycle(ctx, close, volume, t, **kwargs) for t in range(29, 41)]
    for bare, off in zip(results["bare"], results["off"]):
        for field in dataclass_fields(DailyCycleResult):
            left, right = getattr(bare, field.name), getattr(off, field.name)
            if isinstance(left, pd.Series):
                pd.testing.assert_series_equal(left, right)
            else:
                assert left == right, field.name
        assert bare.vol_scale is None
    for ledger in ("state.json", "fills.jsonl", "equity.jsonl", "targets.jsonl", "concordance.jsonl"):
        assert (tmp_path / "bare" / ledger).read_bytes() == (tmp_path / "off" / ledger).read_bytes(), ledger
    for name in ("bare", "off"):
        assert not (tmp_path / name / SIZING_LEDGER_FILENAME).exists()
        assert TARGETS_LEDGER_KEY not in (tmp_path / name / "targets.jsonl").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# On: the scale, the target, the ceiling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_vol", [0.06, 0.15])
def test_the_scale_is_the_rule_on_an_independently_rebuilt_base_book(tmp_path: Path, target_vol: float) -> None:
    close, volume = _walk()
    rule = replace(RULE, target_vol=target_vol)
    t = required_panel_bars(SIGNAL.required_history, rule, EVERY) - 1
    assert t == 29  # 6 bars of history + a 20-return window + a 5-bar cadence - 1, as a panel position
    expected_scale, expected_sigma = _reference_scale(close.iloc[: t + 1], rule, EVERY)
    assert 0.0 < expected_scale < rule.max_scale  # neither cap binds, so the test sees the ratio itself

    scaled = _cycle(_ctx(tmp_path / "on"), close, volume, t, vol_scale=rule)
    unscaled = _cycle(_ctx(tmp_path / "off"), close, volume, t)
    record = scaled.vol_scale
    assert record is not None and record["status"] == "ok" and record["reason"] is None
    assert record["scale"] == pytest.approx(expected_scale, rel=1e-12)
    assert record["sigma"] == pytest.approx(expected_sigma, rel=1e-12)
    assert record["n_valid"] == rule.lookback_bars and record["zero_fraction"] == 0.0
    assert record["schedule"] == "panel"
    assert record["base_refresh_bars"] == [str(close.index[p].date()) for p in (9, 14, 19, 24)]
    # The scaled target is the scale times the unscaled book, float for float (no band, nothing held).
    pd.testing.assert_series_equal(scaled.target_weights, unscaled.target_weights * record["scale"])
    assert record["unscaled_gross"] == pytest.approx(1.0)
    assert record["target_gross"] == pytest.approx(record["scale"])
    assert record["long_over_equity"] == pytest.approx(record["scale"] / 2.0)
    assert record["sleeve_fraction"] == pytest.approx(max(0.0, 0.98 - record["scale"] / 2.0))
    assert record["ceiling"] == 2.0 and record["ceiling_factor"] == 1.0


def test_the_account_ceiling_caps_the_scaled_book(tmp_path: Path) -> None:
    close, volume = _walk()
    rule = replace(RULE, target_vol=1.0)  # far above the book's volatility: the scale sits at its cap of 2
    result = _cycle(_ctx(tmp_path), close, volume, 29, vol_scale=rule, config=replace(CONFIG, max_gross=1.5))
    record = result.vol_scale
    assert record["scale"] == 2.0 and record["ceiling"] == 1.5
    assert record["ceiling_factor"] == pytest.approx(0.75)
    assert result.target_weights.abs().sum() == pytest.approx(1.5)


def test_a_rule_pinned_at_scale_one_reproduces_the_unscaled_book_bit_for_bit(tmp_path: Path) -> None:
    close, volume = _walk()
    pinned = VolScaleRule(target_vol=1e6, lookback_bars=20, max_scale=1.0)
    runs: dict[str, list[DailyCycleResult]] = {}
    for name, kwargs in (("off", {}), ("pinned", {"vol_scale": pinned})):
        ctx = _ctx(tmp_path / name)
        runs[name] = [_cycle(ctx, close, volume, t, **kwargs) for t in range(29, 41)]
    for off, pinned_result in zip(runs["off"], runs["pinned"]):
        for field in dataclass_fields(DailyCycleResult):
            if field.name == "vol_scale":
                continue
            left, right = getattr(off, field.name), getattr(pinned_result, field.name)
            if isinstance(left, pd.Series):
                pd.testing.assert_series_equal(left, right)
            else:
                assert left == right, field.name
    assert [r.vol_scale["scale"] for r in runs["pinned"] if r.vol_scale is not None] == [1.0, 1.0, 1.0]
    for ledger in ("state.json", "fills.jsonl", "equity.jsonl", "concordance.jsonl"):
        assert (tmp_path / "off" / ledger).read_bytes() == (tmp_path / "pinned" / ledger).read_bytes(), ledger
    rows = {name: _rows(tmp_path / name / "targets.jsonl") for name in ("off", "pinned")}
    assert [row.pop(TARGETS_LEDGER_KEY) for row in rows["pinned"]] == [
        {"scale": 1.0, "status": "ok", "reason": None, "sigma": pytest.approx(r.vol_scale["sigma"])}
        for r in runs["pinned"]
        if r.vol_scale is not None
    ]
    assert rows["off"] == rows["pinned"]


def test_the_seeded_schedule_rebuilds_the_base_book_on_its_grid_dates(tmp_path: Path) -> None:
    close, volume = _walk()
    t = 31
    offset = sessions.grid_position(close.index[t].date()) % EVERY
    result = _cycle(_ctx(tmp_path), close, volume, t, vol_scale=RULE, phase_offset=offset)
    record = result.vol_scale
    assert record is not None and record["schedule"] == "grid"
    on_grid = [
        str(close.index[p].date())
        for p in range(t - RULE.lookback_bars - EVERY + 1, t)
        if (sessions.grid_position(close.index[p].date()) - offset) % EVERY == 0
    ]
    assert record["base_refresh_bars"] == on_grid
    # On a gap-free panel the grid dates are every fifth bar back from the decision, so the scale agrees with
    # the reference rebuilt on the panel schedule.
    assert record["scale"] == pytest.approx(_reference_scale(close.iloc[: t + 1], RULE, EVERY)[0], rel=1e-12)


def test_regime_telemetry_still_reads_beside_the_scale(tmp_path: Path) -> None:
    close, volume = _walk()
    ctx = _ctx(tmp_path)
    ctx.regime_ledger = tmp_path / "regime.jsonl"
    provider = RecordingRegime()
    result = _cycle(ctx, close, volume, 29, vol_scale=RULE, regime=provider)
    assert provider.calls == [result.decision_bar]
    assert result.regime is not None and result.vol_scale is not None
    assert (tmp_path / "regime.jsonl").exists()


# ---------------------------------------------------------------------------
# The failure path
# ---------------------------------------------------------------------------


def _session_bars(root: Path, close, volume, rule: VolScaleRule, through: range) -> list[DailyCycleResult]:
    ctx = _ctx(root)
    return [_cycle(ctx, close, volume, t, vol_scale=rule) for t in through]


@pytest.mark.parametrize("target_vol", [0.06, 0.2])
def test_a_stale_panel_falls_back_to_at_most_one_and_at_most_the_last_good_scale(
    tmp_path: Path, target_vol: float
) -> None:
    # Flat from bar 40: refreshes at 29, 34 and 39 see full windows; 44 and 49 see 5 and 10 zero returns of 20.
    close, volume = _walk(flat_from=40)
    rule = replace(RULE, target_vol=target_vol)
    results = _session_bars(tmp_path, close, volume, rule, range(29, 50))
    decisions = [r.vol_scale for r in results if r.vol_scale is not None]
    assert [d["decision_bar"] for d in decisions] == [str(close.index[t].date()) for t in (29, 34, 39, 44, 49)]
    assert [d["status"] for d in decisions] == ["ok", "ok", "ok", "fallback", "fallback"]
    assert [d["reason"] for d in decisions[3:]] == ["zero_returns", "zero_returns"]
    last_good = decisions[2]["scale"]
    for fallback in decisions[3:]:
        assert fallback["last_good"] == last_good
        assert fallback["scale"] == min(1.0, last_good)
    assert (last_good < 1.0) is (target_vol == 0.06)  # one case is bounded by the last good scale, one by 1


def test_a_decision_bar_that_is_not_the_panels_last_bar_falls_back(tmp_path: Path) -> None:
    close, volume = _walk()
    window = close.iloc[:31], volume.iloc[:31]
    result = run_daily_cycle(
        _ctx(tmp_path), SIGNAL, *window, CONFIG, decision_bar=str(close.index[29].date()), vol_scale=RULE
    )
    assert result.vol_scale["status"] == "fallback"
    assert result.vol_scale["reason"] == "decision_bar_not_last_bar"
    assert result.vol_scale["scale"] == 1.0


@pytest.mark.parametrize("last_good", [None, 0.4, 1.0, 1.7])
@pytest.mark.parametrize("flat_from", [16, 26])
def test_no_failure_sizes_above_one_or_above_the_last_good_scale(tmp_path: Path, last_good, flat_from) -> None:
    close, volume = _walk(flat_from=flat_from)
    ctx = _ctx(tmp_path)
    if last_good is not None:
        _seed_refresh_row(ctx.targets_ledger, str(close.index[24].date()), last_good)
    result = _cycle(ctx, close, volume, 29, vol_scale=RULE)
    record = result.vol_scale
    assert record["status"] == "fallback" and record["reason"] == "zero_returns"
    bound = 1.0 if last_good is None else min(1.0, last_good)
    assert record["scale"] == bound
    assert result.target_weights.abs().sum() <= 1.0 + 1e-12  # never above the unscaled book's gross


def test_the_last_good_scale_comes_from_persisted_refreshes_only(tmp_path: Path) -> None:
    ledger = tmp_path / "targets.jsonl"
    assert last_good_scale(ledger, "2025-03-01") is None  # no ledger yet
    _seed_refresh_row(ledger, "2025-01-10", 0.7)
    _seed_refresh_row(ledger, "2025-01-17", 1.0, status="fallback")
    assert last_good_scale(ledger, "2025-03-01") == 0.7  # a fallback is passed over
    assert last_good_scale(ledger, "2025-01-10") is None  # a row on or after the bar is not before it
    _append(ledger, {"refresh_bar": "2025-01-24", "decision_bar": "2025-01-24", "targets": {}})
    assert last_good_scale(ledger, "2025-03-01") is None  # an unscaled refresh ends the search


# ---------------------------------------------------------------------------
# Point in time
# ---------------------------------------------------------------------------


def test_a_later_decision_reproduces_an_earlier_ones_returns_on_shared_bars() -> None:
    close, volume = _walk()
    offset = sessions.grid_position(close.index[34].date()) % EVERY
    early, _, _ = base_book_window(
        SIGNAL, close.iloc[:35], volume.iloc[:35], CONFIG, max_gross=2.0, lookback_bars=20, phase_offset=offset
    )
    late, _, _ = base_book_window(
        SIGNAL, close.iloc[:45], volume.iloc[:45], CONFIG, max_gross=2.0, lookback_bars=20, phase_offset=offset
    )
    # early covers bars 15..34, late bars 25..44; they share 25..34.
    np.testing.assert_array_equal(early[10:], late[:10])


def test_the_decision_bars_close_moves_only_the_last_return() -> None:
    close, volume = _walk()
    window = close.iloc[:30].copy()
    bumped = window.copy()
    bumped.iloc[-1] = bumped.iloc[-1] * np.linspace(0.9, 1.1, N_SYM)
    kwargs = dict(max_gross=2.0, lookback_bars=20, phase_offset=None)
    base, bars, _ = base_book_window(SIGNAL, window, volume.iloc[:30], CONFIG, **kwargs)
    moved, moved_bars, _ = base_book_window(SIGNAL, bumped, volume.iloc[:30], CONFIG, **kwargs)
    np.testing.assert_array_equal(base[:-1], moved[:-1])
    assert base[-1] != moved[-1]
    assert bars == moved_bars and str(window.index[-1].date()) not in bars


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_one_sizing_row_per_refresh_and_none_on_hold_sessions(tmp_path: Path) -> None:
    close, volume = _walk()
    results = _session_bars(tmp_path, close, volume, RULE, range(29, 41))
    refreshes = [r for r in results if r.vol_scale is not None]
    assert [r.decision_bar for r in refreshes] == [str(close.index[t].date()) for t in (29, 34, 39)]
    frame = read_sizing_ledger(tmp_path / SIZING_LEDGER_FILENAME)
    assert list(frame["decision_bar"]) == [r.decision_bar for r in refreshes]
    assert _rows(tmp_path / SIZING_LEDGER_FILENAME) == [_json_round_trip(r.vol_scale) for r in refreshes]
    targets = _rows(tmp_path / "targets.jsonl")
    assert [row["refresh_bar"] for row in targets] == [r.decision_bar for r in refreshes]
    assert [row[TARGETS_LEDGER_KEY]["scale"] for row in targets] == [r.vol_scale["scale"] for r in refreshes]
    assert all("panel_sha256" in row for row in targets)  # the panel provenance is kept beside the scale


def test_a_decision_refused_before_the_write_ahead_is_recorded_once_and_bounds_nothing(tmp_path: Path) -> None:
    close, volume = _walk()
    ctx = _ctx(tmp_path)
    with pytest.raises(SafetyViolation):
        _cycle(ctx, close, volume, 29, vol_scale=RULE, safety=SafetyConfig(max_orders=1))
    assert _persisted_decision(ctx) is None  # nothing persisted, nothing sent
    assert ctx.broker.submitted_order_ids() == set()
    assert len(_rows(tmp_path / SIZING_LEDGER_FILENAME)) == 1
    assert last_good_scale(ctx.targets_ledger, "2099-01-01") is None  # the refused decision is no bound

    result = _cycle(ctx, close, volume, 29, vol_scale=RULE)  # the same bar, the guard relaxed
    rows = _rows(tmp_path / SIZING_LEDGER_FILENAME)
    assert len(rows) == 1 and rows[0] == _json_round_trip(result.vol_scale)
    assert ctx.broker.submitted_order_ids()
    assert last_good_scale(ctx.targets_ledger, "2099-01-01") == result.vol_scale["scale"]


def test_a_same_bar_rerun_after_the_write_ahead_decides_nothing_new(tmp_path: Path) -> None:
    close, volume = _walk()
    ctx = _ctx(tmp_path)
    first = _cycle(ctx, close, volume, 29, vol_scale=RULE)
    second = _cycle(ctx, close, volume, 29, vol_scale=RULE)
    assert first.vol_scale is not None and second.vol_scale is None  # resumed, not re-decided
    assert len(_rows(tmp_path / SIZING_LEDGER_FILENAME)) == 1


def test_a_conflicting_sizing_row_raises_before_anything_is_persisted(tmp_path: Path) -> None:
    close, volume = _walk()
    ctx = _ctx(tmp_path)
    honest = _cycle(_ctx(tmp_path / "reference"), close, volume, 29, vol_scale=RULE).vol_scale
    _append(tmp_path / SIZING_LEDGER_FILENAME, {**_json_round_trip(honest), "scale": 0.123})
    with pytest.raises(SizingLedgerConflict, match="different scale decision"):
        _cycle(ctx, close, volume, 29, vol_scale=RULE)
    assert _persisted_decision(ctx) is None
    assert ctx.broker.submitted_order_ids() == set()
    assert not (tmp_path / "targets.jsonl").exists()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case, match",
    [
        ("directional_book", "decile_neutral"),
        ("regime_hook", "regime hook is refused"),
        ("low_gross", "max_gross >= 1.0"),
        ("fitted_signal", "fitted_state"),
        ("no_targets_ledger", "targets ledger"),
        ("short_panel", "needs 30"),
    ],
)
def test_inputs_the_scale_cannot_honour_raise_before_any_state_moves(tmp_path: Path, case: str, match: str) -> None:
    close, volume = _walk()
    ctx = _ctx(tmp_path, targets=case != "no_targets_ledger")
    signal: Signal = ConstSignal({f"S{i}": float(i) for i in range(N_SYM)}) if case == "fitted_signal" else SIGNAL
    config = CONFIG
    kwargs: dict = {}
    through = 29
    if case == "directional_book":
        config = DailyBookConfig(position_size=0.1, max_gross=2.0, decision_every=EVERY)
    elif case == "regime_hook":
        config = replace(CONFIG, max_gross=1.0)
        kwargs = {"regime": RecordingRegime(), "regime_gross_scale": lambda state: 1.0}
    elif case == "low_gross":
        config = replace(CONFIG, max_gross=0.5)
    elif case == "short_panel":
        through = 28
    with pytest.raises(VolSizingError, match=match):
        run_daily_cycle(
            ctx, signal, close.iloc[: through + 1], volume.iloc[: through + 1], config, vol_scale=RULE, **kwargs
        )
    assert ctx.store.load() is None
    assert not (tmp_path / "panels").exists()
    assert ctx.broker.submitted_order_ids() == set()


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def test_the_preset_is_the_registered_rule() -> None:
    rule = VOL_SCALE_PRESETS["bsc-12-126-2"]
    assert (rule.target_vol, rule.lookback_bars, rule.max_scale, rule.max_gross) == (0.12, 126, 2.0, 2.0)
    assert (rule.annualization_bars, rule.max_zero_return_fraction) == (252, 0.05)
    # 12-1 momentum's history, a 126-return window and a 21-session cadence.
    assert required_panel_bars(253, rule, 21) == 399


def test_resolve_vol_scale() -> None:
    assert resolve_vol_scale("off", book="trend", max_gross=1.0) is None
    assert resolve_vol_scale("bsc-12-126-2", book="momentum", max_gross=2.0) is VOL_SCALE_PRESETS["bsc-12-126-2"]
    with pytest.raises(SystemExit, match="--book momentum only"):
        resolve_vol_scale("bsc-12-126-2", book="trend", max_gross=2.0)
    with pytest.raises(SystemExit, match="requires --max-gross 2"):
        resolve_vol_scale("bsc-12-126-2", book="momentum", max_gross=1.0)
    for profile_id in ("research_paper", "conservative"):
        with pytest.raises(SystemExit, match=f"--profile {profile_id}"):
            resolve_vol_scale(
                "bsc-12-126-2", book="momentum", max_gross=2.0, profile=resolve_risk_profile(profile_id)
            )


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch) -> VenueDouble:
    """``paper_loop`` wired to in-memory doubles, the way ``tests/test_paper_no_submit.py`` wires it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret-key")
    monkeypatch.delenv("APCA_API_BASE_URL", raising=False)
    broker = VenueDouble()
    close, volume = _panels()
    monkeypatch.setattr(paper_loop, "AlpacaBroker", SimpleNamespace(from_env=lambda **kw: broker), raising=True)
    monkeypatch.setattr(
        paper_loop,
        "AlpacaBarSource",
        SimpleNamespace(from_env=lambda **kw: BarSourceDouble(close, volume)),
        raising=True,
    )
    return broker


@pytest.fixture()
def cycles(monkeypatch) -> list[dict]:
    """The keyword arguments of every ``run_daily_cycle`` call ``paper_loop.main`` makes."""
    calls: list[dict] = []
    real = paper_loop.run_daily_cycle

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(paper_loop, "run_daily_cycle", spy, raising=True)
    return calls


ARMED = ("--vol-scale", "bsc-12-126-2", "--max-gross", "2", "--decision-every", "21")


def test_the_default_passes_no_scale_and_writes_no_record(wired: VenueDouble, tmp_path: Path, cycles) -> None:
    run_dir = tmp_path / "runs" / "plain"
    _point_switch(run_dir)
    assert paper_loop.main(_loop_argv(run_dir, "--submit", "--allow-new-run-dir", "--decision-every", "21")) == 0
    assert cycles[-1]["vol_scale"] is None
    assert not (run_dir / SIZING_LEDGER_FILENAME).exists()


def test_an_armed_session_records_its_decision_in_the_run_directory(
    wired: VenueDouble, tmp_path: Path, cycles
) -> None:
    run_dir = tmp_path / "runs" / "scaled"
    _point_switch(run_dir)
    assert paper_loop.main(_loop_argv(run_dir, "--submit", "--allow-new-run-dir", *ARMED)) == 0
    assert cycles[-1]["vol_scale"] is VOL_SCALE_PRESETS["bsc-12-126-2"]
    rows = _rows(run_dir / SIZING_LEDGER_FILENAME)
    assert len(rows) == 1
    # The fixture's names grow at constant rates, so the base book's returns are constant up to rounding: the
    # estimate is degenerate and the book is sized at the fallback, 1, rather than levered to the cap.
    assert rows[0]["status"] == "fallback" and rows[0]["reason"] == "degenerate_sigma"
    assert rows[0]["scale"] == 1.0 and rows[0]["target_gross"] <= 1.0 + 1e-12
    assert _rows(run_dir / "targets.jsonl")[0][TARGETS_LEDGER_KEY]["status"] == "fallback"
    assert wired.submitted


@pytest.mark.parametrize(
    "extra, match",
    [
        (("--vol-scale", "bsc-12-126-2", "--decision-every", "21"), "requires --max-gross 2"),
        (("--vol-scale", "bsc-12-126-2", "--max-gross", "2", "--profile", "research_paper"), "research_paper"),
    ],
)
def test_the_cli_refuses_before_any_venue_read(wired: VenueDouble, tmp_path: Path, cycles, extra, match) -> None:
    run_dir = _established(tmp_path)
    _point_switch(run_dir)
    with pytest.raises(SystemExit, match=match):
        paper_loop.main(_loop_argv(run_dir, "--submit", *extra))
    assert wired.reads == [] and wired.submitted == [] and cycles == []


def test_the_cli_refuses_a_book_it_cannot_scale(wired: VenueDouble, tmp_path: Path, cycles) -> None:
    run_dir = _established(tmp_path)
    _point_switch(run_dir)
    argv = ["--book", "trend", "--run-dir", str(run_dir), "--submit", *ARMED]
    with pytest.raises(SystemExit, match="--book momentum only"):
        paper_loop.main(argv)
    assert wired.reads == [] and cycles == []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _seed_refresh_row(path: Path, bar: str, scale: float, *, status: str = "ok") -> None:
    _append(
        path,
        {
            "refresh_bar": bar,
            "decision_bar": bar,
            "equity": 1_000_000.0,
            "targets": {},
            "reference_prices": {},
            TARGETS_LEDGER_KEY: {"scale": scale, "status": status, "reason": None, "sigma": 0.1},
        },
    )


def _persisted_decision(ctx: LiveLoopContext) -> str | None:
    state = ctx.store.load()
    return None if state is None else state.pending_decision_bar


def _json_round_trip(row: dict) -> dict:
    return json.loads(json.dumps(row, sort_keys=True))
