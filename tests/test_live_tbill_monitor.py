"""The report-only T-bill-referenced live-monitor read (``prism.live.monitor.tbill_monitor_read``).

It sits beside the registered hurdle-0 read (docs/live_monitor_design.md) and must
never change it: off, every cycle output and ledger byte is what it was; on, the
registered read is the same dict key for key and the new read lands under ``tbill``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.live import DailyBookConfig, DailyCycleResult, run_daily_cycle
from prism.live.loop import append_equity_ledger, append_targets_ledger
from prism.live.monitor import (
    DEFAULT_TBILL_ETF_FEE,
    TbillReference,
    detection_horizon_years,
    held_book_exposure,
    load_tbill_reference,
    paper_monitor_read,
    tbill_monitor_read,
)
from prism.validation.hurdle import tbill_hurdle
from tests.test_live_daily import ConstSignal, _full_ctx, _session_panels
from tests.test_live_monitor import _session_bars, _write_equity

LANE = Path(__file__).resolve().parents[1]
EXAMPLE = LANE / "examples" / "reference" / "monitor" / "tbill_reference.toml"
RF = 0.0421
REFERENCE = TbillReference(rf_annual=RF, rf_source="test rate")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def test_the_shipped_example_loads_and_states_its_sources() -> None:
    reference = load_tbill_reference(EXAMPLE)
    assert reference.rf_annual == pytest.approx(0.0421) and "French RF" in reference.rf_source
    assert reference.cash_yield == 0.0 and reference.etf_fee_annual == DEFAULT_TBILL_ETF_FEE
    assert reference.planning_sharpe is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('rf_annual = 0.04\nrf_source = "x"\nrf_anual = 0.05\n', "unknown key"),
        ("rf_annual = 0.04\n", "missing"),
        ('rf_annual = 0.04\nrf_source = "  "\n', "rf_source must name"),
        ('rf_annual = 0.04\nrf_source = "x"\ncash_yield = -0.01\n', "cash_yield must be >= 0"),
        ('rf_annual = 0.04\nrf_source = "x"\nplanning_sharpe = 0.3\n', "planning_source must name"),
        ('rf_annual = 0.04\nrf_source = "x"\nplanning_sharpe = 0.0\nplanning_source = "y"\n', "planning_sharpe"),
        ('rf_annual = true\nrf_source = "x"\n', "real number"),
        ("rf_annual = [", "cannot be read"),
    ],
)
def test_the_loader_refuses_what_it_cannot_read(tmp_path, text, message) -> None:
    path = tmp_path / "reference.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_tbill_reference(path)


def test_detection_horizon_is_the_fixed_sample_years_and_none_without_an_edge() -> None:
    assert detection_horizon_years(0.30) == pytest.approx(30.0616, abs=1e-3)
    assert detection_horizon_years(0.105) == pytest.approx(245.40, abs=1e-2)
    assert detection_horizon_years(0.30, alpha=0.025) == pytest.approx((1.959964 / 0.30) ** 2, rel=1e-6)
    assert detection_horizon_years(0.0) is None and detection_horizon_years(-0.2) is None
    with pytest.raises(ValueError):
        detection_horizon_years(0.3, alpha=1.0)


# ---------------------------------------------------------------------------
# The book held through each session
# ---------------------------------------------------------------------------


def test_held_exposure_takes_the_latest_book_decided_by_the_previous_close() -> None:
    rows = [
        {"refresh_bar": "2026-07-06", "targets": {"AAA": 0.3, "BBB": -0.3}},
        {"refresh_bar": "2026-07-08", "targets": {"AAA": 0.5, "BBB": -0.5, "SGOV": 0.48},
         "cash_holding": {"symbol": "SGOV", "weight": 0.48}},
    ]
    prev = ["2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"]
    found = held_book_exposure(rows, prev)
    # 07-02: no book yet; 07-06 and 07-07: the first book; 07-08 and 07-09: the second, ETF excluded from L.
    assert found["n_sessions"] == 5 and found["n_sessions_without_book"] == 1
    assert found["long_over_equity_mean"] == pytest.approx((0.0 + 0.3 + 0.3 + 0.5 + 0.5) / 5)
    assert found["sleeve_fraction_mean"] == pytest.approx(2 * 0.48 / 5)
    assert found["etf_symbols"] == ["SGOV"] and found["source"] == "targets ledger"
    none = held_book_exposure(None, prev)
    assert none["source"] is None and none["long_over_equity_mean"] == 0.0 and none["n_sessions_without_book"] == 5


def test_held_exposure_takes_the_etf_share_from_the_per_session_record_when_there_is_one() -> None:
    # A monthly book: one targets row, whose holding record is the first, bounded step of the build.
    rows = [
        {"refresh_bar": "2026-07-06", "targets": {"AAA": 0.5, "BBB": -0.5, "SGOV": 0.199},
         "cash_holding": {"symbol": "SGOV", "weight": 0.199}},
    ]
    holding = [
        {"decision_bar": "2026-07-02", "symbol": "SGOV", "weight": 0.199},  # bought before the first book
        {"decision_bar": "2026-07-06", "symbol": "SGOV", "weight": 0.199},
        {"decision_bar": "2026-07-07", "symbol": "SGOV", "weight": 0.398},
        {"decision_bar": "2026-07-08", "symbol": "SGOV", "weight": 0.48},
    ]
    prev = ["2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"]
    found = held_book_exposure(rows, prev, holding_rows=holding)
    assert found["n_sessions_without_book"] == 1
    assert found["long_over_equity_mean"] == pytest.approx(4 * 0.5 / 5)  # the ETF is not stock long side
    assert found["sleeve_fraction_mean"] == pytest.approx((0.199 + 0.199 + 0.398 + 0.48 + 0.48) / 5)
    assert found["source"] == "targets ledger and cash-holding ledger" and found["etf_symbols"] == ["SGOV"]
    # Without the record, the refresh row's weight stands for every later session.
    assert held_book_exposure(rows, prev)["sleeve_fraction_mean"] == pytest.approx(4 * 0.199 / 5)


def test_the_cycle_records_the_holding_every_session_and_the_tbill_read_uses_it(tmp_path) -> None:
    from dataclasses import replace

    from prism.live.cash_holding import LEDGER_FILENAME
    from tests.test_live_cash_holding import CONFIG, HOLDING, _ctx, _cycle, _panel

    close, volume = _panel()
    ctx = _ctx(tmp_path)
    monthly = replace(CONFIG, decision_every=21)
    results = [
        _cycle(ctx, close, volume, t, cash_holding=HOLDING, config=monthly, tbill_monitor=REFERENCE)
        for t in range(3, 9)
    ]
    recorded = [json.loads(line) for line in (tmp_path / LEDGER_FILENAME).read_text().splitlines()]
    assert [row["decision_bar"] for row in recorded] == [r.decision_bar for r in results]
    assert [row["weight"] for row in recorded] == [r.cash_holding["weight"] for r in results]
    exposure = results[-1].monitor_read["tbill"]["exposure"]
    assert exposure["source"] == "targets ledger and cash-holding ledger"
    # Five return sessions held the ETF at the weights decided at each previous close.
    expected = sum(r.cash_holding["weight"] for r in results[:-1]) / 5
    assert exposure["sleeve_fraction_mean"] == pytest.approx(expected)
    # A same-bar re-run resumes and records nothing new.
    _cycle(ctx, close, volume, 8, cash_holding=HOLDING, config=monthly)
    assert len((tmp_path / LEDGER_FILENAME).read_text().splitlines()) == len(results)


# ---------------------------------------------------------------------------
# The read
# ---------------------------------------------------------------------------


def _equity(tmp_path, n=120, drift=0.0004, seed=3) -> Path:
    rng = np.random.default_rng(seed)
    bars = _session_bars("2026-01-02", n)
    navs = 100_000.0 * np.cumprod(1.0 + rng.normal(drift, 0.006, size=n))
    path = tmp_path / "equity.jsonl"
    _write_equity(path, bars, navs)
    return path


def test_without_an_etf_or_cash_yield_the_hurdle_is_the_bill_rate(tmp_path) -> None:
    path = _equity(tmp_path)
    read = tbill_monitor_read(path, REFERENCE)
    assert read["hurdle_annual"] == RF and read["hurdle"] == pytest.approx(RF / 252)
    registered_at_that_hurdle = paper_monitor_read(path, hurdle=RF / 252)
    for key, value in registered_at_that_hurdle.items():
        assert read[key] == value, key
    assert read["reference"] == "tbill" and read["report_only"] is True
    assert read["hurdle_inputs"]["vol_basis"] == "realized_annualized" and read["hurdle_inputs"]["rf_source"] == "test rate"
    assert read["detection_horizon"]["years"] is None
    # The mean is the same series the hurdle-0 read sees; only the bar moved.
    assert read["mean"] == paper_monitor_read(path)["mean"]


def test_an_etf_holding_is_credited_its_distributions_through_the_shared_hurdle(tmp_path) -> None:
    path = _equity(tmp_path)
    bars = _session_bars("2026-01-02", 120)
    targets = tmp_path / "targets.jsonl"
    for bar in bars[:: 21]:
        weights = pd.Series({"AAA": 0.5, "BBB": -0.5, "SGOV": 0.48})
        append_targets_ledger(targets, bar, bar, weights, pd.Series(100.0, index=weights.index), 1e5,
                              extra={"cash_holding": {"symbol": "SGOV", "weight": 0.48}})
    reference = TbillReference(rf_annual=RF, rf_source="test rate", cash_yield=0.033, cash_yield_source="listed",
                               planning_sharpe=0.105, planning_source="planning")
    read = tbill_monitor_read(path, reference, targets_ledger_path=targets)
    exposure = read["exposure"]
    # Every return session's previous bar is on or after the first refresh, so the book is always held.
    assert exposure["n_sessions_without_book"] == 0
    assert exposure["long_over_equity_mean"] == pytest.approx(0.5) and exposure["sleeve_fraction_mean"] == 0.48
    expected = tbill_hurdle(RF, 0.033, 0.5, 0.1, sleeve_fraction=0.48, sleeve_yield=RF - DEFAULT_TBILL_ETF_FEE)
    assert read["hurdle_annual"] == pytest.approx(expected.return_hurdle_annual, abs=1e-15)
    assert read["hurdle_annual"] == pytest.approx(RF - (RF - 0.001) * 0.48 - 0.033 * 0.02)
    assert read["hurdle_inputs"]["sleeve_fraction"] == 0.48
    assert read["detection_horizon"]["years"] == pytest.approx(detection_horizon_years(0.105))


def test_the_read_refuses_inputs_it_cannot_honor(tmp_path) -> None:
    path = _equity(tmp_path)
    with pytest.raises(ValueError, match="needs the targets ledger"):
        tbill_monitor_read(path, TbillReference(rf_annual=RF, rf_source="x", cash_yield=0.01))
    targets = tmp_path / "targets.jsonl"
    weights = pd.Series({"AAA": 0.9, "SGOV": 0.5})
    append_targets_ledger(targets, "2026-01-02", "2026-01-02", weights, pd.Series(100.0, index=weights.index), 1e5,
                          extra={"cash_holding": {"symbol": "SGOV", "weight": 0.5}})
    with pytest.raises(ValueError, match="exceeds the idle equity"):
        tbill_monitor_read(path, REFERENCE, targets_ledger_path=targets)


def test_before_two_returns_the_sharpe_form_is_omitted_and_the_return_hurdle_stands(tmp_path) -> None:
    path = tmp_path / "equity.jsonl"
    append_equity_ledger(path, "2026-01-02", 1e5, 1e5)
    read = tbill_monitor_read(path, REFERENCE)
    assert read["n"] == 0 and read["verdict"] == "inconclusive" and read["hurdle_annual"] == RF
    assert read["hurdle_inputs"]["sharpe_annual"] is None and read["hurdle_inputs"]["vol_annual"] is None


# ---------------------------------------------------------------------------
# In the cycle: off is bit-identical, on leaves the registered read alone
# ---------------------------------------------------------------------------


def _cycles(root, **kwargs) -> list[DailyCycleResult]:
    ctx = _full_ctx(root)
    ctx.shadow_ledger = root / "shadow.jsonl"
    ctx.broker._cash = 1_000_000.0
    config = DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=3)
    signal = ConstSignal({"AAA": 1.0, "BBB": -1.0})
    moves = {"AAA": 0.001, "BBB": -0.0005}
    return [run_daily_cycle(ctx, signal, *_session_panels(n, moves), config, **kwargs) for n in range(30, 37)]


def _ledger_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(root.glob("*.jsonl"))}


def test_off_is_bit_identical_and_on_changes_only_the_new_key(tmp_path) -> None:
    bare = _cycles(tmp_path / "bare")
    off = _cycles(tmp_path / "off", tbill_monitor=None)
    on = _cycles(tmp_path / "on", tbill_monitor=REFERENCE)
    for a, b, c in zip(bare, off, on):
        for field in dataclass_fields(DailyCycleResult):
            left, middle, right = (getattr(r, field.name) for r in (a, b, c))
            if isinstance(left, pd.Series):
                pd.testing.assert_series_equal(left, middle)
                pd.testing.assert_series_equal(left, right)
            elif field.name == "monitor_read":
                assert left == middle and "tbill" not in left
                assert {k: v for k, v in right.items() if k != "tbill"} == left
            else:
                assert left == middle == right, field.name
    assert _ledger_bytes(tmp_path / "bare") == _ledger_bytes(tmp_path / "off") == _ledger_bytes(tmp_path / "on")
    last = on[-1].monitor_read["tbill"]
    assert last["report_only"] is True and last["n"] == on[-1].monitor_read["n"]
    # The first cycle refreshes on the ledger's first bar, so every return session holds a book.
    assert last["hurdle"] == pytest.approx(RF / 252) and last["exposure"]["n_sessions_without_book"] == 0
    assert last["exposure"]["long_over_equity_mean"] == pytest.approx(0.1, rel=1e-3)


def test_a_failed_tbill_read_is_contained_and_leaves_the_registered_read(tmp_path, caplog, monkeypatch) -> None:
    import prism.live.daily as daily_mod

    def broken(*args, **kwargs):
        raise RuntimeError("targets ledger unreadable")

    monkeypatch.setattr(daily_mod, "tbill_monitor_read", broken)
    with caplog.at_level(logging.WARNING, logger="prism.live.daily"):
        on = _cycles(tmp_path / "on", tbill_monitor=REFERENCE)
    bare = _cycles(tmp_path / "bare")
    assert on[-1].monitor_read["tbill"] == {"verdict": "unavailable", "n": 0, "report_only": True,
                                            "error": "RuntimeError", "detail": "targets ledger unreadable"}
    assert {k: v for k, v in on[-1].monitor_read.items() if k != "tbill"} == bare[-1].monitor_read
    assert any("T-BILL READ FAILED" in record.getMessage() for record in caplog.records)


def test_the_paper_loop_flag_parses_and_defaults_off() -> None:
    from prism.scripts.paper_loop import _parse_args

    assert _parse_args(["--no-submit"]).tbill_monitor is None
    assert _parse_args(["--no-submit", "--tbill-monitor", str(EXAMPLE)]).tbill_monitor == EXAMPLE
