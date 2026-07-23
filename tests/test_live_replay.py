"""The replay instrument (prism/live/replay.py): the real daily cycle over
local bars with a simulated next-open venue.

Pins the seams replay owns: the ReplayBroker fill semantics (next-open print,
once per order, reference-price fallback on a data gap), the ragged-union
panel alignment (consensus calendar + clean-name screen, keep names fail
loud), the run-dir separation guard (a replay may never adopt or pollute an
existing loop ledger), and the multi-cycle driver against the real cadence
gate and write-ahead protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.live import (
    DailyBookConfig,
    DuplicateOrder,
    Order,
    ReplayBroker,
    align_replay_panels,
    load_local_bar_panels,
    read_equity_ledger,
    read_fills_ledger,
    read_targets_ledger,
    replay_daily_cycles,
)
from prism.signal.base import Signal
from tests.test_live_daily import ConstSignal


def _order(oid: str = "2026-07-01:AAA", symbol: str = "AAA", qty: float = 10.0) -> Order:
    return Order(client_order_id=oid, symbol=symbol, qty=qty, decision_bar="2026-07-01", reference_price=100.0)


# ---------------------------------------------------------------------------
# ReplayBroker — the simulated next-open venue
# ---------------------------------------------------------------------------


def test_replay_broker_duplicate_submit_raises() -> None:
    broker = ReplayBroker(cash=1_000.0)
    broker.submit(_order())
    with pytest.raises(DuplicateOrder):
        broker.submit(_order())
    assert broker.submitted_order_ids() == {"2026-07-01:AAA"}


def test_replay_broker_requires_an_armed_fill_bar() -> None:
    broker = ReplayBroker(cash=1_000.0)
    broker.submit(_order())
    with pytest.raises(RuntimeError, match="no armed fill bar"):
        broker.fills_for({"2026-07-01:AAA"})


def test_replay_broker_fills_once_at_the_armed_open() -> None:
    broker = ReplayBroker(cash=10_000.0, positions={"AAA": 5.0})
    broker.submit(_order(qty=10.0))
    broker.set_fill(pd.Series({"AAA": 101.5}), "2026-07-02")
    fills = broker.fills_for({"2026-07-01:AAA"})
    assert len(fills) == 1
    assert fills[0].price == 101.5
    assert fills[0].filled_bar == "2026-07-02"
    assert broker.positions() == {"AAA": 15.0}
    assert broker.cash() == pytest.approx(10_000.0 - 10.0 * 101.5)
    # A second settle pass sees nothing new — the order filled once.
    assert broker.fills_for({"2026-07-01:AAA"}) == []


def test_replay_broker_falls_back_to_reference_on_missing_open(caplog) -> None:
    broker = ReplayBroker(cash=10_000.0)
    broker.submit(_order(qty=-4.0))
    broker.set_fill(pd.Series({"AAA": np.nan}), "2026-07-02")
    with caplog.at_level("WARNING"):
        fills = broker.fills_for({"2026-07-01:AAA"})
    assert fills[0].price == 100.0  # the decision-close reference
    assert "no open print" in caplog.text
    assert broker.positions() == {"AAA": -4.0}


# ---------------------------------------------------------------------------
# Panel alignment — consensus calendar + clean-name screen
# ---------------------------------------------------------------------------


def _ragged_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2026-01-01", periods=10, freq="B", tz="America/New_York")
    close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0, "CCC": 20.0}, index=idx)
    close.loc[idx[3], ["BBB", "CCC"]] = np.nan  # a vendor-holiday row: 1/3 present
    close.loc[idx[7:], "CCC"] = np.nan  # CCC's recent history is dirty
    volume = pd.DataFrame(1e6, index=idx, columns=close.columns)
    open_ = close * 0.999
    return close, volume, open_


def test_align_drops_non_consensus_rows_and_dirty_names() -> None:
    close, volume, open_ = _ragged_panels()
    # consensus 0.6: the 1/3-present holiday row falls, the 2/3-present rows
    # (only CCC missing) stay — so the two screens separate: the calendar drops
    # one row, the clean-name screen drops CCC.
    a_close, a_volume, a_open = align_replay_panels(close, volume, open_, consensus=0.6, clean_window=5)
    assert len(a_close) == 9
    assert list(a_close.columns) == ["AAA", "BBB"]  # CCC's dirty tail dropped it
    assert list(a_volume.columns) == ["AAA", "BBB"]
    assert list(a_open.columns) == ["AAA", "BBB"]


def test_align_keep_name_that_fails_the_screen_raises() -> None:
    close, volume, open_ = _ragged_panels()
    with pytest.raises(ValueError, match="keep names \\['CCC'\\]"):
        align_replay_panels(close, volume, open_, consensus=0.6, clean_window=5, keep=("CCC",))


# ---------------------------------------------------------------------------
# Local parquet loader — offline union of main + delta caches
# ---------------------------------------------------------------------------


def test_load_local_bar_panels_unions_caches_and_bounds_missing(tmp_path) -> None:
    idx1 = pd.date_range("2026-01-01", periods=5, freq="B")
    idx2 = pd.date_range(idx1[-1], periods=3, freq="B")  # overlaps the main cache's last bar
    bars = pd.DataFrame({"open": 99.0, "close": 100.0, "volume": 1e6}, index=idx1)
    delta = pd.DataFrame({"open": 101.0, "close": 102.0, "volume": 2e6}, index=idx2)
    bars.to_parquet(tmp_path / "AAA_1d_2026-01-01_2026-01-07.parquet")
    delta.to_parquet(tmp_path / "AAA_1d_2026-01-07_2026-01-09.parquet")

    with pytest.raises(RuntimeError, match="no local bars for 1/2"):
        load_local_bar_panels(["AAA", "GONE"], tmp_path)

    close, volume, open_ = load_local_bar_panels(["AAA", "GONE"], tmp_path, max_missing=0.5)
    assert list(close.columns) == ["AAA"]
    assert len(close) == 7  # 5 + 3 with the overlapping bar de-duplicated keep-last
    assert close["AAA"].iloc[-1] == 102.0 and close["AAA"].loc[idx1[-1]] == 102.0
    assert open_["AAA"].iloc[0] == 99.0 and volume["AAA"].iloc[-1] == 2e6


# ---------------------------------------------------------------------------
# The replay driver — real cycles, cadence, and the separation guard
# ---------------------------------------------------------------------------


class FlipSignal(Signal):
    """Directional scores that invert once the panel reaches ``flip_at`` rows —
    a deterministic mid-replay signal change to pin the cadence gate."""

    def __init__(self, flip_at: int, required: int = 3) -> None:
        self._flip_at = flip_at
        self._required = required

    @property
    def horizon_bars(self) -> int:
        return 1

    @property
    def required_history(self) -> int:
        return self._required

    def fit(self, close, volume=None) -> "FlipSignal":
        return self

    def score(self, close, volume=None) -> pd.Series:
        sign = -1.0 if len(close) >= self._flip_at else 1.0
        return pd.Series({"AAA": sign, "BBB": -sign}, dtype=float).reindex(close.columns)


def _flat_panels(n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2026-05-01", periods=n, freq="B", tz="America/New_York")
    close = pd.DataFrame({"AAA": [100.0] * n, "BBB": [50.0] * n}, index=idx)
    volume = pd.DataFrame(1e6, index=idx, columns=close.columns)
    open_ = close * 0.999
    return close, volume, open_


def test_replay_runs_cycles_settles_fills_and_accrues_ledgers(tmp_path) -> None:
    close, volume, open_ = _flat_panels(30)
    results, broker = replay_daily_cycles(
        lambda: ConstSignal({"AAA": 1.0, "BBB": -1.0}),
        close,
        volume,
        open_,
        DailyBookConfig(position_size=0.1, min_order_notional=1.0),
        tmp_path / "replay",
        start_bar=str(close.index[24].date()),
    )
    assert len(results) == 6  # panel rows 24..29
    assert len(results[0].submitted_orders) == 2  # cycle 0 opens the book
    assert len(results[1].settled_fills) == 2  # ...which fills at cycle 1's open
    assert all(not r.submitted_orders for r in results[1:])  # steady book, no churn
    assert broker.positions() == {"AAA": 100.0, "BBB": -200.0}
    equity = read_equity_ledger(tmp_path / "replay" / "equity.jsonl")
    assert len(equity) == 6  # one NAV mark per cycle
    fills = read_fills_ledger(tmp_path / "replay" / "fills.jsonl")
    assert len(fills) == 2
    assert results[-1].monitor_read is not None and results[-1].monitor_read["n"] == 5
    # The per-refresh decided book persists exactly as in the live loop — the
    # replay-vs-backtest concordance object. Daily cadence with a steady book:
    # cycle 0 refreshes and later refreshes re-emit the same held targets.
    targets = read_targets_ledger(tmp_path / "replay" / "targets.jsonl")
    assert targets[0]["refresh_bar"] == results[0].decision_bar
    assert targets[0]["targets"] == {"AAA": 0.1, "BBB": -0.1}
    assert targets[0]["reference_prices"] == {"AAA": 100.0, "BBB": 50.0}


def test_replay_holds_between_refreshes_and_trades_the_flip_on_cadence(tmp_path) -> None:
    close, volume, open_ = _flat_panels(30)
    # Replay the last 7 bars (panel rows 24..30). The signal flips from row 25 —
    # i.e. from cycle 1 — but decision_every=3 means the flip may only execute at
    # the cycle-3 refresh; the cadence gate must hold cycles 1-2 despite it.
    results, broker = replay_daily_cycles(
        lambda: FlipSignal(flip_at=25),
        close,
        volume,
        open_,
        DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=3),
        tmp_path / "replay",
        start_bar=str(close.index[23].date()),
    )
    pattern = ["R" if r.submitted_orders else "." for r in results]
    assert pattern[0] == "R"  # first cycle always refreshes (no cadence anchor yet)
    assert pattern[1:3] == [".", "."]  # the flip is visible but the cadence holds it
    assert pattern[3] == "R"  # the refresh trades the flip...
    assert pattern[4:6] == [".", "."]
    # ...but only to FLAT: the flip-through-zero clamp (targets_to_orders) trades
    # a side-crossing target to zero this refresh and opens the opposite side at
    # the NEXT refresh (cycle 6) — whose orders are still pending when the window
    # ends. The honest live-vs-backtest wrinkle, visible in a replay.
    assert pattern[6] == "R"
    assert broker.positions() == {}  # flat: the opposite side never got its fill bar


def test_replay_refuses_an_existing_run_dir(tmp_path) -> None:
    close, volume, open_ = _flat_panels(10)
    run_dir = tmp_path / "live_lookalike"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already holds loop state"):
        replay_daily_cycles(
            lambda: ConstSignal({"AAA": 1.0}),
            close,
            volume,
            open_,
            DailyBookConfig(position_size=0.1),
            run_dir,
        )


def test_replay_refuses_a_start_bar_without_required_history(tmp_path) -> None:
    close, volume, open_ = _flat_panels(10)
    with pytest.raises(ValueError, match="required_history"):
        replay_daily_cycles(
            lambda: ConstSignal({"AAA": 1.0}, required=8),
            close,
            volume,
            open_,
            DailyBookConfig(position_size=0.1),
            tmp_path / "replay",
            start_bar=str(close.index[2].date()),
        )


def test_replay_cli_tif_defaults_to_whole_share_parity() -> None:
    """--tif defaults to opg (whole shares, parity with every prior scripted
    replay); day is the fractional-share sizing path mirroring paper_loop."""
    from prism.scripts.replay_loop import _parse_args

    assert _parse_args([]).tif == "opg"
    assert _parse_args(["--tif", "day"]).tif == "day"
    with pytest.raises(SystemExit):
        _parse_args(["--tif", "ioc"])


def test_replay_tif_wires_whole_shares_at_both_book_call_sites() -> None:
    """--tif drives DailyBookConfig.whole_shares at BOTH construction paths —
    the momentum decile book and the ensemble directional book — not just the
    flag parse: opg = whole shares, day = fractional."""
    from prism.scripts.replay_loop import _book_config, _parse_args

    for book, expected_book in (
        ("momentum", "decile_neutral"),
        ("ensemble", "directional"),
        ("trend", "inverse_vol"),
    ):
        for tif, whole in (("opg", True), ("day", False)):
            args = _parse_args(["--book", book, "--tif", tif])
            config = _book_config(
                args, decision_every=21 if book in ("momentum", "trend") else 1
            )
            assert config.book == expected_book  # the branch under test is the branch taken
            assert config.whole_shares is whole
    # And the momentum path threads the cadence it was handed.
    assert _book_config(_parse_args(["--book", "momentum"]), decision_every=21).decision_every == 21
    assert _book_config(_parse_args(["--book", "trend"]), decision_every=21).vol_ewma_bars == 63
