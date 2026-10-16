"""The T-bill holding beside the decile book (``run_daily_cycle(cash_holding=...)``, ``prism.live.cash_holding``).

Pinned here:

* the target formula ``max(0, 1 - buffer - L)`` and how the stock long side
  ``L`` is counted (finite targets, held weights for NaN or unlisted names,
  shorts and the ETF itself excluded);
* the trading rule: hold inside the band, move to the target outside it, by at
  most the per-order bound less one share, sell a residual when the target is
  zero, target zero under a wind-down;
* off is bit-identical: every result field and every ledger byte matches a run
  that never heard of the holding;
* on, the holding is built over several sessions inside the per-order guard,
  on sessions the book does not refresh too, then held inside its band; the
  stock book is unchanged; a halted session does not touch it; a tranched book
  carries it;
* the documented exits: ``buffer = 1`` sells it in bounded steps, and dropping
  the holding while the ETF is held trips the per-order guard;
* refusals before any state moves, and ``resolve_fetch_universe(unscored=...)``;
* the CLI: the flags default off, their refusals come before any venue read,
  and an armed session fetches the ETF, keeps it out of the ranking and buys it
  inside the bound.

Nothing here opens a socket.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
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
    resolve_fetch_universe,
    run_daily_cycle,
)
from prism.live.cash_holding import (
    TARGETS_LEDGER_KEY,
    CashHolding,
    CashHoldingError,
    apply_cash_holding,
    holding_target,
    stock_long_weight,
)
from prism.live.safety import SafetyViolation
from prism.live.state import LoopState
from prism.scripts import paper_loop
from prism.scripts.paper_books import resolve_cash_holding
from tests.test_live_daily import ConstSignal
from tests.test_live_loop import FakeBroker
from tests.test_paper_no_submit import (
    SYMBOLS,
    BarSourceDouble,
    VenueDouble,
    _established,
    _loop_argv,
    _panels,
    _point_switch,
)

N_SYM = 20
NAMES = [f"S{i:02d}" for i in range(N_SYM)]
ETF = "BIL"
PRICE = 100.0
CASH = 100_000.0
#: Real exchange sessions, so a tranched book can place every bar on the grid.
DAYS = sessions.sessions_between(dt.date(2025, 1, 2), dt.date(2025, 6, 30))
#: Twenty names, top and bottom 20%: four a leg at 0.125 each, so the long side is 0.5.
CONFIG = DailyBookConfig(
    book="decile_neutral",
    decile=0.2,
    decision_every=1,
    max_gross=1.0,
    max_symbol_abs_weight=1.0,
    min_order_notional=1.0,
)
SIGNAL = ConstSignal({name: float(i) for i, name in enumerate(NAMES)})
SAFETY = SafetyConfig(max_order_fraction=0.20)
HOLDING = CashHolding(ETF)
#: 0.20 of equity less one share at 100.00 on 100,000.
STEP = 0.20 - PRICE / CASH


def _panel(n_bars: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day in DAYS[:n_bars]])
    close = pd.DataFrame(PRICE, index=index, columns=[*NAMES, ETF])
    return close, pd.DataFrame(1e6, index=index, columns=close.columns)


def _ctx(root: Path, broker: FakeBroker | None = None) -> LiveLoopContext:
    root.mkdir(parents=True, exist_ok=True)
    return LiveLoopContext(
        store=StateStore(root / "state.json"),
        broker=broker or FakeBroker(cash=CASH),
        fills_ledger=root / "fills.jsonl",
        equity_ledger=root / "equity.jsonl",
        targets_ledger=root / "targets.jsonl",
        unfilled_ledger=root / "unfilled.jsonl",
        concordance_ledger=root / "concordance.jsonl",
    )


def _cycle(ctx, close, volume, through: int, **kwargs) -> DailyCycleResult:
    kwargs.setdefault("safety", SAFETY)
    return run_daily_cycle(
        ctx, SIGNAL, close.iloc[: through + 1], volume.iloc[: through + 1], kwargs.pop("config", CONFIG), **kwargs
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _etf_orders(results: list[DailyCycleResult]) -> list[float]:
    return [sum(o.qty for o in r.submitted_orders if o.symbol == ETF) for r in results]


# ---------------------------------------------------------------------------
# The formula and the rule
# ---------------------------------------------------------------------------


def test_the_target_is_the_idle_equity_above_the_buffer() -> None:
    assert holding_target(0.5, 0.02) == pytest.approx(0.48)
    assert holding_target(0.0, 0.02) == pytest.approx(0.98)
    assert holding_target(0.98, 0.02) == 0.0
    assert holding_target(1.4, 0.02) == 0.0  # a long side above equity leaves nothing idle
    assert holding_target(0.3, 1.0) == 0.0
    with pytest.raises(CashHoldingError):
        holding_target(float("nan"), 0.02)


def test_the_long_side_counts_held_weights_where_the_target_decides_nothing() -> None:
    targets = pd.Series({"A": 0.3, "B": -0.2, "C": float("nan"), ETF: 0.4})
    held = pd.Series({"C": 0.1, "D": 0.05, "E": -0.3, ETF: 0.4})
    # A at its target, C (NaN: left as held) and D (unlisted: left as held) at their held weights; the shorts
    # B and E and the ETF itself do not count.
    assert stock_long_weight(targets, held, exclude=ETF) == pytest.approx(0.45)


def _apply(held: float, long_side: float, **kwargs):
    targets = pd.Series({"A": long_side, "B": -long_side, ETF: 0.0})
    held_weights = pd.Series({ETF: held}) if held else pd.Series(dtype=float)
    kwargs.setdefault("max_order_fraction", 0.20)
    return apply_cash_holding(
        kwargs.pop("holding", HOLDING), targets, held_weights, prices={ETF: PRICE}, equity=CASH, **kwargs
    )


def test_inside_the_band_the_holding_is_held_and_outside_it_moves_to_the_target() -> None:
    decided, decision = _apply(0.47, 0.5)  # target 0.48, 0.01 away
    assert decision.action == "hold" and decided[ETF] == 0.47 and decision.target_weight == pytest.approx(0.48)
    decided, decision = _apply(0.40, 0.5)  # 0.08 away
    assert decision.action == "rebalance" and decided[ETF] == pytest.approx(0.48)
    decided, decision = _apply(0.60, 0.5)  # above the target: sold down to it
    assert decision.action == "rebalance" and decided[ETF] == pytest.approx(0.48)


def test_one_session_moves_at_most_the_per_order_bound_less_one_share() -> None:
    decided, decision = _apply(0.0, 0.5)
    assert decision.action == "bounded" and decision.step_bound == pytest.approx(STEP)
    assert decided[ETF] == pytest.approx(STEP)
    decided, decision = _apply(0.9, 0.5)  # a sale is bounded too
    assert decision.action == "bounded" and decided[ETF] == pytest.approx(0.9 - STEP)
    decided, decision = _apply(0.0, 0.5, max_order_fraction=None)  # no bound, no limit
    assert decision.action == "rebalance" and decision.step_bound is None and decided[ETF] == pytest.approx(0.48)


def test_a_zero_target_sells_a_residual_inside_the_band_and_a_wind_down_targets_zero() -> None:
    decided, decision = _apply(0.01, 0.99)  # long side uses the equity: target zero, 0.01 held
    assert decision.target_weight == 0.0 and decision.action == "rebalance" and decided[ETF] == 0.0
    decided, decision = _apply(0.3, 0.2, wind_down=True)
    assert decision.wind_down and decision.target_weight == 0.0
    assert decision.action == "bounded" and decided[ETF] == pytest.approx(0.3 - STEP)
    decided, decision = _apply(0.0, 0.2, wind_down=True)
    assert decision.action == "hold" and decided[ETF] == 0.0


def test_the_input_series_are_not_modified_and_every_other_weight_is_kept() -> None:
    targets = pd.Series({"A": 0.5, "B": -0.5, "C": float("nan")})
    held = pd.Series({"A": 0.5, "B": -0.5})
    before = targets.copy()
    decided, _ = apply_cash_holding(HOLDING, targets, held, prices={ETF: PRICE}, equity=CASH, max_order_fraction=0.2)
    pd.testing.assert_series_equal(targets, before)
    pd.testing.assert_series_equal(decided.drop(ETF), before)


def test_the_rule_refuses_what_it_cannot_size() -> None:
    with pytest.raises(CashHoldingError, match="no usable decision close"):
        apply_cash_holding(HOLDING, pd.Series(dtype=float), pd.Series(dtype=float), prices={}, equity=CASH,
                           max_order_fraction=0.2)
    with pytest.raises(CashHoldingError, match="per-order bound"):
        # One 100.00 share is 25% of 400.00 of equity, above a 20% bound.
        apply_cash_holding(HOLDING, pd.Series(dtype=float), pd.Series(dtype=float), prices={ETF: PRICE},
                           equity=400.0, max_order_fraction=0.2)
    for kwargs in ({"symbol": " "}, {"symbol": ETF, "buffer": -0.1}, {"symbol": ETF, "buffer": 1.5},
                   {"symbol": ETF, "band": 1.0}, {"symbol": ETF, "band": float("nan")}):
        with pytest.raises(CashHoldingError):
            CashHolding(**kwargs)
    assert CashHolding(" sgov ").symbol == "SGOV"


# ---------------------------------------------------------------------------
# Off is bit-identical
# ---------------------------------------------------------------------------


def test_off_is_bit_identical(tmp_path: Path) -> None:
    close, volume = _panel()
    results: dict[str, list[DailyCycleResult]] = {}
    for name, kwargs in (("bare", {}), ("off", {"cash_holding": None, "buying_power_check": False})):
        ctx = _ctx(tmp_path / name)
        results[name] = [_cycle(ctx, close, volume, t, **kwargs) for t in range(3, 9)]
    for bare, off in zip(results["bare"], results["off"]):
        for field in dataclass_fields(DailyCycleResult):
            left, right = getattr(bare, field.name), getattr(off, field.name)
            if isinstance(left, pd.Series):
                pd.testing.assert_series_equal(left, right)
            else:
                assert left == right, field.name
        assert bare.cash_holding is None
    for ledger in ("state.json", "fills.jsonl", "equity.jsonl", "targets.jsonl", "concordance.jsonl"):
        assert (tmp_path / "bare" / ledger).read_bytes() == (tmp_path / "off" / ledger).read_bytes(), ledger
    assert TARGETS_LEDGER_KEY not in (tmp_path / "bare" / "targets.jsonl").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# On: built over sessions, inside the guard, then held
# ---------------------------------------------------------------------------


def test_the_holding_is_built_over_sessions_inside_the_order_bound_then_held(tmp_path: Path) -> None:
    close, volume = _panel()
    ctx = _ctx(tmp_path / "on")
    results = [_cycle(ctx, close, volume, t, cash_holding=HOLDING) for t in range(3, 9)]
    # Target 1 - 0.02 - 0.5 = 0.48 of 100,000 at 100.00: 480 shares, at most 199 (19,900 <= 20,000) a session.
    assert _etf_orders(results) == [199.0, 199.0, 82.0, 0.0, 0.0, 0.0]
    assert [r.cash_holding["action"] for r in results] == ["bounded", "bounded", "rebalance", "hold", "hold", "hold"]
    for result in results:
        record = result.cash_holding
        assert record["symbol"] == ETF and record["long_weight"] == pytest.approx(0.5)
        assert record["target_weight"] == pytest.approx(0.48)
        assert result.target_weights[ETF] == pytest.approx(record["weight"])
        for order in result.submitted_orders:  # every order passed the guard it was sized for
            assert abs(order.qty) * order.reference_price <= SAFETY.max_order_fraction * result.equity
    assert ctx.broker.positions()[ETF] == 480.0
    # Each refresh row of the targets ledger carries the decision.
    rows = _rows(tmp_path / "on" / "targets.jsonl")
    assert [row[TARGETS_LEDGER_KEY]["action"] for row in rows] == [r.cash_holding["action"] for r in results]


def test_the_stock_book_is_the_book_without_the_holding(tmp_path: Path) -> None:
    close, volume = _panel()
    plain_ctx, held_ctx = _ctx(tmp_path / "plain"), _ctx(tmp_path / "on")
    plain = [_cycle(plain_ctx, close, volume, t) for t in range(3, 6)]
    held = [_cycle(held_ctx, close, volume, t, cash_holding=HOLDING) for t in range(3, 6)]
    for left, right in zip(plain, held):
        pd.testing.assert_series_equal(left.target_weights.drop(ETF), right.target_weights.drop(ETF))
        assert left.target_weights[ETF] == 0.0  # NaN-scored: the construct's explicit flat, untouched when off
        assert sorted(o.symbol for o in left.submitted_orders) == sorted(
            o.symbol for o in right.submitted_orders if o.symbol != ETF
        )


def test_the_holding_trades_on_sessions_the_book_does_not_refresh(tmp_path: Path) -> None:
    close, volume = _panel()
    ctx = _ctx(tmp_path)
    monthly = replace(CONFIG, decision_every=5)
    results = [_cycle(ctx, close, volume, t, cash_holding=HOLDING, config=monthly) for t in range(3, 9)]
    stock_orders = [sum(1 for o in r.submitted_orders if o.symbol != ETF) for r in results]
    assert stock_orders[0] == 8 and stock_orders[1:5] == [0, 0, 0, 0]  # refresh on the first session only
    assert _etf_orders(results)[:4] == [199.0, 199.0, 82.0, 0.0]
    # Only refresh sessions write a targets row.
    assert len(_rows(tmp_path / "targets.jsonl")) == 2


def test_a_halted_session_does_not_touch_the_holding(tmp_path: Path) -> None:
    close, volume = _panel()
    ctx = _ctx(tmp_path)
    _cycle(ctx, close, volume, 3, cash_holding=HOLDING)
    kill = tmp_path / "KILL_SWITCH"
    kill.write_text("")
    halted = _cycle(ctx, close, volume, 4, cash_holding=HOLDING, safety=replace(SAFETY, kill_switch=kill))
    assert halted.halted is not None and halted.cash_holding is None and halted.submitted_orders == []


def test_buffer_one_sells_the_holding_in_bounded_steps(tmp_path: Path) -> None:
    close, volume = _panel()
    ctx = _ctx(tmp_path)
    for t in range(3, 7):  # the fourth session holds and settles the third's order
        _cycle(ctx, close, volume, t, cash_holding=HOLDING)
    assert ctx.broker.positions()[ETF] == 480.0
    exit_holding = CashHolding(ETF, buffer=1.0)
    results = [_cycle(ctx, close, volume, t, cash_holding=exit_holding) for t in range(7, 11)]
    assert _etf_orders(results) == [-199.0, -199.0, -82.0, 0.0]
    assert ctx.broker.positions().get(ETF, 0.0) == 0.0


def test_dropping_the_holding_while_the_etf_is_held_trips_the_order_guard(tmp_path: Path) -> None:
    # The documented reason to exit through buffer = 1 or a wind-down instead: the construct sets the
    # unscored ETF flat, and one 48%-of-equity sale is refused before anything is persisted.
    close, volume = _panel()
    ctx = _ctx(tmp_path)
    for t in range(3, 7):
        _cycle(ctx, close, volume, t, cash_holding=HOLDING)
    assert ctx.broker.positions()[ETF] == 480.0
    with pytest.raises(SafetyViolation, match="max_order_fraction"):
        _cycle(ctx, close, volume, 7)


def test_a_tranched_book_carries_the_holding(tmp_path: Path) -> None:
    close, volume = _panel(14)
    ctx = _ctx(tmp_path)
    tranched = replace(CONFIG, decision_every=3)
    results = [_cycle(ctx, close, volume, t, cash_holding=HOLDING, config=tranched, tranches=3) for t in range(3, 9)]
    # Staged entry: one tranche (1/3 of the book) built a session, so the long side grows 1/6 a session.
    longs = [r.cash_holding["long_weight"] for r in results]
    assert longs[:3] == pytest.approx([0.5 / 3, 1.0 / 3, 0.5])
    assert all(abs(order) * PRICE <= 0.2 * CASH for order in _etf_orders(results))
    assert ctx.broker.positions()[ETF] == pytest.approx(480.0)
    assert results[-1].tranches is not None and results[-1].tranches["built"] == 3


def test_a_tranched_book_started_off_its_grid_buys_the_holding_and_then_builds(tmp_path: Path) -> None:
    """The holding trades before the first tranche is due; the flat-account check does not count it."""
    close, volume = _panel(30)
    ctx = _ctx(tmp_path)
    tranched = replace(CONFIG, decision_every=21)
    start = 5  # 2025-01-10, grid position 19 mod 21: tranches 0, 7 and 14 are not due for two sessions
    assert sessions.grid_position(close.index[start].date()) % 21 == 19
    results = [
        _cycle(ctx, close, volume, t, cash_holding=HOLDING, config=tranched, tranches=3) for t in range(start, 23)
    ]
    assert [r.tranches["due"] for r in results[:2]] == [[], []]
    assert _etf_orders(results)[:2] == [199.0, 199.0]
    assert all(o.symbol == ETF for r in results[:2] for o in r.submitted_orders)
    assert results[2].tranches["due"] == ["decile_neutral/t00"] and results[2].tranches["built"] == 1
    assert results[2].tranches["catch_up"] == []  # the ETF is not the tranches' to flatten
    assert results[-1].tranches["built"] == 3
    assert all(abs(order) * PRICE <= 0.2 * CASH for order in _etf_orders(results))


def test_the_cycle_refuses_a_holding_it_cannot_run_before_any_state_moves(tmp_path: Path) -> None:
    close, volume = _panel()
    cases = [
        ({"config": replace(CONFIG, book="directional", position_size=0.1)}, close, "decile_neutral"),
        ({}, close.drop(columns=[ETF]), "not in the bar panel"),
    ]
    gapped = close.copy()
    gapped.iloc[-1, gapped.columns.get_loc(ETF)] = float("nan")
    cases.append(({}, gapped, "no usable close"))
    for index, (kwargs, panel, match) in enumerate(cases):
        ctx = _ctx(tmp_path / f"case{index}")
        with pytest.raises(CashHoldingError, match=match):
            run_daily_cycle(ctx, SIGNAL, panel, volume.reindex(columns=panel.columns), kwargs.pop("config", CONFIG),
                            safety=SAFETY, cash_holding=HOLDING)
        assert not (tmp_path / f"case{index}" / "state.json").exists()
        assert not (tmp_path / f"case{index}" / "panels").exists()


# ---------------------------------------------------------------------------
# The fetch universe
# ---------------------------------------------------------------------------


def test_an_unscored_name_is_fetched_and_priced_and_never_reported_as_outside(caplog) -> None:
    broker = FakeBroker()
    caplog.set_level(logging.WARNING, logger="prism.live.daily")
    universe = resolve_fetch_universe(["AAA", "BBB"], broker, unscored=[" sgov "])
    assert universe.score == ("AAA", "BBB") and universe.valuation_only == ("SGOV",)
    assert universe.fetch == ("AAA", "BBB", "SGOV") and universe.held == {}
    assert not [r for r in caplog.records if "outside the configured universe" in r.getMessage()]

    broker._positions = {"SGOV": 10.0, "ZZZ": 5.0}
    state = LoopState(positions={"SGOV": 10.0, "ZZZ": 5.0})
    universe = resolve_fetch_universe(["AAA"], broker, state=state, unscored=["SGOV"])
    assert universe.valuation_only == ("SGOV", "ZZZ") and universe.held == {"SGOV": 10.0, "ZZZ": 5.0}
    outside = [r.getMessage() for r in caplog.records if "outside the configured universe" in r.getMessage()]
    assert outside and "ZZZ" in outside[-1] and "SGOV" not in outside[-1]

    with pytest.raises(ValueError, match="both in the configured universe"):
        resolve_fetch_universe(["AAA", "SGOV"], broker, unscored=["SGOV"])
    # No unscored names: exactly the previous split.
    assert resolve_fetch_universe(["AAA"], broker, state=state) == resolve_fetch_universe(
        ["AAA"], broker, state=state, unscored=()
    )


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


SLEEVE = "SGOV"


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch) -> VenueDouble:
    """``paper_loop`` wired to in-memory doubles, with a T-bill ETF in the bar source.

    The ETF trends up faster than any stock, so it would take a slot in the long
    leg if it were ranked.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret-key")
    monkeypatch.delenv("APCA_API_BASE_URL", raising=False)
    broker = VenueDouble()
    close, volume = _panels()
    close[SLEEVE] = 100.0 * np.exp(np.arange(len(close)) * 0.002)
    volume[SLEEVE] = 1_000_000.0
    monkeypatch.setattr(paper_loop, "AlpacaBroker", SimpleNamespace(from_env=lambda **kw: broker), raising=True)
    monkeypatch.setattr(
        paper_loop,
        "AlpacaBarSource",
        SimpleNamespace(from_env=lambda **kw: BarSourceDouble(close, volume)),
        raising=True,
    )
    return broker


@pytest.fixture()
def cycles(monkeypatch) -> list[tuple[dict, DailyCycleResult]]:
    """The keyword arguments and the result of every ``run_daily_cycle`` call ``paper_loop.main`` makes."""
    calls: list[tuple[dict, DailyCycleResult]] = []
    real = paper_loop.run_daily_cycle

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append((kwargs, result))
        return result

    monkeypatch.setattr(paper_loop, "run_daily_cycle", spy, raising=True)
    return calls


def test_the_flags_default_off() -> None:
    args = paper_loop._parse_args([])
    assert args.cash_sleeve is None and args.cash_sleeve_buffer is None and args.cash_sleeve_band is None
    assert args.buying_power_check is False


def test_the_default_passes_no_holding(wired: VenueDouble, tmp_path: Path, cycles) -> None:
    run_dir = tmp_path / "runs" / "plain"
    _point_switch(run_dir)
    assert paper_loop.main(_loop_argv(run_dir, "--submit", "--allow-new-run-dir")) == 0
    kwargs, result = cycles[-1]
    assert kwargs["cash_holding"] is None and kwargs["buying_power_check"] is False
    assert SLEEVE not in result.target_weights.index and result.cash_holding is None


def test_an_armed_session_buys_the_etf_inside_the_bound_and_never_ranks_it(
    wired: VenueDouble, tmp_path: Path, cycles
) -> None:
    plain_dir = tmp_path / "runs" / "plain"
    _point_switch(plain_dir)
    assert paper_loop.main(_loop_argv(plain_dir, "--submit", "--allow-new-run-dir")) == 0
    _, plain = cycles[-1]
    wired.submitted.clear()

    run_dir = tmp_path / "runs" / "sleeve"
    _point_switch(run_dir)
    argv = _loop_argv(run_dir, "--submit", "--allow-new-run-dir", "--cash-sleeve", SLEEVE.lower())
    assert paper_loop.main(argv) == 0
    kwargs, result = cycles[-1]
    assert kwargs["cash_holding"] == CashHolding(SLEEVE)
    # The stock book is the one decided without the ETF in the panel: the ETF took no slot in a leg.
    pd.testing.assert_series_equal(result.target_weights.drop(SLEEVE), plain.target_weights)
    record = result.cash_holding
    assert record["action"] == "bounded"
    assert record["long_weight"] == pytest.approx(float(plain.target_weights.clip(lower=0.0).sum()))
    assert record["target_weight"] == pytest.approx(0.98 - record["long_weight"])
    etf = [o for o in wired.submitted if o.symbol == SLEEVE]
    assert len(etf) == 1 and etf[0].qty > 0
    assert etf[0].qty * etf[0].reference_price <= 0.2 * result.equity


@pytest.mark.parametrize(
    "extra, match",
    [
        (("--cash-sleeve", SLEEVE, "--book", "trend"), "--book momentum only"),
        (("--cash-sleeve", SLEEVE, "--profile", "research_paper"), "research_paper"),
        (("--cash-sleeve", "aaa"), "in the configured universe"),
        (("--cash-sleeve-buffer", "0.05"), "need --cash-sleeve"),
        (("--cash-sleeve", SLEEVE, "--cash-sleeve-band", "1.5"), "band"),
    ],
)
def test_the_cli_refuses_before_any_venue_read(wired: VenueDouble, tmp_path: Path, cycles, extra, match) -> None:
    run_dir = _established(tmp_path)
    _point_switch(run_dir)
    with pytest.raises(SystemExit, match=match):
        paper_loop.main(_loop_argv(run_dir, "--submit", *extra))
    assert wired.reads == [] and wired.submitted == [] and cycles == []


def test_resolve_cash_holding() -> None:
    assert resolve_cash_holding(None, buffer=None, band=None, book="trend", universe=SYMBOLS) is None
    holding = resolve_cash_holding("bil", buffer=0.05, band=0.01, book="momentum", universe=SYMBOLS)
    assert holding == CashHolding("BIL", buffer=0.05, band=0.01)
    assert resolve_cash_holding("SGOV", buffer=None, band=None, book="momentum", universe=SYMBOLS) == CashHolding(
        "SGOV", buffer=0.02, band=0.02
    )
    assert math.isclose(holding.buffer, 0.05)
