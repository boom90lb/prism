"""Arithmetic and read-only guarantees of the trend_v1 adjudication helper (spec §5, tests 11-13)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.arbitrage.trend_walk_forward import TREND_TZ
from research.scripts import trend_adjudicate as adj
from research.scripts import trend_wfo

TRIAL_ARGV = (
    ["--lookback", "126"],
    ["--skip", "0"],
    ["--decision_every", "63"],
    ["--sizing", "equal_notional"],
)


def _write_cell(run_dir: Path, config: dict, sharpe: float | None, total_return: float | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str))
    (run_dir / "summary.json").write_text(
        json.dumps({"sharpe": sharpe, "total_return": total_return, "config_hash": "abc"}, allow_nan=True)
    )


def _registered_set(root: Path, sharpes: list[float | None], returns: list[float | None]) -> tuple[Path, list[Path], Path]:
    t0 = root / "trend_t0"
    _write_cell(t0, trend_wfo._config_payload(trend_wfo.parse_args([])), sharpes[0], returns[0])
    trials = []
    for k, argv in enumerate(TRIAL_ARGV, start=1):
        d = root / f"trend_t{k}"
        _write_cell(d, trend_wfo._config_payload(trend_wfo.parse_args(argv)), sharpes[k], returns[k])
        trials.append(d)
    ledger = root / "trend_v1_trials.jsonl"
    ledger.write_text("".join(json.dumps({"output_dir": str(d)}) + "\n" for d in (t0, *trials)))
    return t0, trials, ledger


# --------------------------------------------------------------------------------------
# 11. fragility
# --------------------------------------------------------------------------------------


def test_fragility_arithmetic(tmp_path: Path) -> None:
    nan = float("nan")
    # median rule with a NaN probe mapped to -inf: sorted [-inf, -0.2, 0.1, 0.3] -> median -0.05 < 0
    read = adj.fragility_arithmetic(0.5, [0.3, nan, -0.2, 0.1], 0.2, [0.1, nan, -0.05, 0.05])
    assert read["median_sharpe_t1_t4"] == pytest.approx(-0.05)
    assert read["kill_a_median_negative"] is True
    assert read["kill_b_sign_flip_sharpe"] is False  # -0.2 flips sign but |.| < 0.5; NaN excluded
    assert read["n_nan_sharpes"] == 1
    assert read["verdict"] == "KILL (fragility)"
    # flip rule: one probe flips at magnitude above T0
    read = adj.fragility_arithmetic(0.5, [0.6, 0.4, -0.7, 0.3], 0.2, [0.3, 0.2, 0.1, 0.05])
    assert read["kill_a_median_negative"] is False
    assert read["kill_b_sign_flip_sharpe"] is True and read["kill_b_flipping_trials_sharpe"] == [3]
    assert read["secondary_sign_flip_total_return"] == []
    assert read["verdict"] == "KILL (fragility)"
    # secondary lens only (total return flips, Sharpe does not): survives, lens reported
    read = adj.fragility_arithmetic(0.5, [0.6, 0.4, 0.2, 0.3], 0.2, [0.3, -0.5, 0.1, 0.05])
    assert read["kill_b_sign_flip_sharpe"] is False and read["secondary_sign_flip_total_return"] == [2]
    assert read["verdict"] == "SURVIVES fragility read"
    # a NaN T0 Sharpe never counts as a flip
    read = adj.fragility_arithmetic(nan, [0.6, 0.4, 0.2, 0.3], nan, [0.3, 0.1, 0.1, 0.05])
    assert read["kill_b_sign_flip_sharpe"] is False and math.isnan(read["t0_sharpe"])

    # Directory read: a valid registered set is readable and reproduces the arithmetic.
    t0, trials, ledger = _registered_set(tmp_path / "ok", [0.5, 0.3, None, -0.2, 0.1], [0.2, 0.1, None, -0.05, 0.05])
    read = adj.fragility_read(t0, trials, ledger)
    assert read["readable"] is True and read["kill_a_median_negative"] is True
    assert read["median_sharpe_t1_t4"] == pytest.approx(-0.05)

    # A fallback dir is not a registered cell.
    t0, trials, ledger = _registered_set(tmp_path / "fb", [0.5, 0.3, 0.2, 0.1, 0.1], [0.1] * 5)
    cfg = json.loads((trials[1] / "config.json").read_text())
    cfg["price_return_fallback"] = True
    (trials[1] / "config.json").write_text(json.dumps(cfg))
    read = adj.fragility_read(t0, trials, ledger)
    assert read["readable"] is False and "price_return_fallback" in read["reason"]

    # A sub-sample dir (end_date 2021-12-31) is not a registered cell.
    t0, trials, ledger = _registered_set(tmp_path / "sub", [0.5, 0.3, 0.2, 0.1, 0.1], [0.1] * 5)
    cfg = json.loads((t0 / "config.json").read_text())
    cfg["end_date"] = "2021-12-31T00:00:00-05:00"
    (t0 / "config.json").write_text(json.dumps(cfg))
    read = adj.fragility_read(t0, trials, ledger)
    assert read["readable"] is False and "end_date" in read["reason"]

    # A wrong knob moved (two keys) is refused; so is a mis-ordered ledger.
    t0, trials, ledger = _registered_set(tmp_path / "knob", [0.5, 0.3, 0.2, 0.1, 0.1], [0.1] * 5)
    cfg = json.loads((trials[0] / "config.json").read_text())
    cfg["signal"]["skip_bars"] = 0
    (trials[0] / "config.json").write_text(json.dumps(cfg))
    read = adj.fragility_read(t0, trials, ledger)
    assert read["readable"] is False and "config keys moved" in read["reason"]
    t0, trials, ledger = _registered_set(tmp_path / "order", [0.5, 0.3, 0.2, 0.1, 0.1], [0.1] * 5)
    ledger.write_text("".join(json.dumps({"output_dir": str(d)}) + "\n" for d in (*trials, t0)))
    read = adj.fragility_read(t0, trials, ledger)
    assert read["readable"] is False and "ledger output_dirs" in read["reason"]
    # Fewer than four trial dirs is unreadable.
    assert adj.fragility_read(t0, trials[:3], ledger)["readable"] is False


# --------------------------------------------------------------------------------------
# 12. convexity
# --------------------------------------------------------------------------------------


def _window_returns(index: pd.DatetimeIndex, daily: dict[int, float], default: float, window: int = 21) -> pd.Series:
    """Constant daily return per window k (rows k*21+1 .. k*21+21)."""
    values = np.full(len(index), default)
    for k, r in daily.items():
        values[k * window + 1 : k * window + window + 1] = r
    return pd.Series(values, index=index, name="daily_return")


def test_convexity_arithmetic() -> None:
    window = 21
    n_windows = 20
    n_rows = n_windows * window + 1
    index = pd.bdate_range("2019-01-02", periods=n_rows, tz=TREND_TZ)
    decision_bars = [index[k * window] for k in range(n_windows + 1)]  # the last one has no complete window
    # Sleeve: convex on the SPY leg (does well in windows 3 and 7), poor on the B1 leg's worst windows (10, 12).
    sleeve = _window_returns(index, {3: 0.002, 7: 0.003, 10: -0.005, 12: -0.004}, 0.0005)
    # SPY: piecewise-constant level on each window block; worst two window returns at k=3 and k=7.
    levels = [100.0]
    for k in range(n_windows):
        step = {3: -0.10, 7: -0.08}.get(k, 0.01)
        levels.append(levels[-1] * (1.0 + step))
    spy = pd.Series(np.nan, index=index)
    for k in range(n_windows + 1):
        spy.iloc[k * window] = levels[k]
    spy = spy.ffill()
    # B1 stream: starts mid-window 4 (so window 4 is excluded), worst windows k=10 and k=12.
    b1_start = 4 * window + 5
    b1_index = index[b1_start:]
    b1_vals = np.full(len(b1_index), 0.001)
    for k, r in {10: -0.01, 12: -0.008}.items():
        lo = k * window + 1 - b1_start
        b1_vals[lo : lo + window] = r
    b1 = pd.Series(b1_vals, index=b1_index, name="daily_return")

    oos_start = index[15 * window]
    out = adj.convexity_read(sleeve, index, spy, decision_bars, b1, decile=0.10, oos_start=oos_start)
    full = out["full_sample"]
    assert full["n_windows"] == n_windows
    spy_leg = full["spy"]
    assert spy_leg["n"] == 20 and spy_leg["m"] == 2
    assert spy_leg["worst_decision_bars"] == [decision_bars[3].isoformat(), decision_bars[7].isoformat()]
    expected_cond = np.mean([(1.002) ** window - 1.0, (1.003) ** window - 1.0])
    assert spy_leg["mu_cond"] == pytest.approx(expected_cond)
    all_windows = [(1.0005) ** window - 1.0] * 20
    for k, r in {3: 0.002, 7: 0.003, 10: -0.005, 12: -0.004}.items():
        all_windows[k] = (1.0 + r) ** window - 1.0
    assert spy_leg["mu_unc"] == pytest.approx(float(np.mean(all_windows)))
    assert spy_leg["pass"] is True and spy_leg["threshold"] == pytest.approx(-0.08)
    b1_leg = full["b1_overlap"]
    assert b1_leg["n"] == 15 and b1_leg["m"] == 1  # windows 5..19 only; floor(1.5) = 1
    assert b1_leg["worst_decision_bars"] == [decision_bars[10].isoformat()]
    assert b1_leg["mu_cond"] == pytest.approx((0.995) ** window - 1.0)
    assert b1_leg["mu_unc"] == pytest.approx(float(np.mean(all_windows[5:])))
    assert b1_leg["pass"] is False and "power_note" in b1_leg
    assert full["pass"] is False
    assert adj.convexity_verdict(full["pass"], 0.3) == adj.VERDICT_REFUSED
    assert adj.convexity_verdict(full["pass"], -0.3) == adj.VERDICT_NONPOSITIVE
    assert adj.convexity_verdict(True, 0.3) == adj.VERDICT_CONVEX
    oos = out["oos_segment"]
    assert oos["n_windows"] == 5 and oos["spy"]["n"] == 5 and oos["spy"]["m"] == 1 and oos["b1_overlap"]["n"] == 5
    assert oos["spy"]["span"][0] == oos_start.isoformat()
    # Without B1 the b1 leg is null and the sample is unreadable rather than silently passing.
    no_b1 = adj.convexity_read(sleeve, index, spy, decision_bars, None, decile=0.10, oos_start=None)
    assert no_b1["full_sample"]["b1_overlap"] is None and no_b1["full_sample"]["pass"] is None
    assert no_b1["oos_segment"]["pass"] is None
    # An empty OOS segment (all d before oos_start) is null with a reason.
    late = adj.convexity_read(sleeve, index, spy, decision_bars, b1, decile=0.10, oos_start=index[-1])
    assert late["oos_segment"]["n_windows"] == 0 and late["oos_segment"]["pass"] is None
    # Incomplete windows (returns missing) are dropped, not padded.
    truncated = sleeve.iloc[:-5]
    fewer = adj.convexity_read(truncated, index, spy, decision_bars, b1, decile=0.10, oos_start=None)
    assert fewer["full_sample"]["n_windows"] == n_windows - 1


# --------------------------------------------------------------------------------------
# 12b. readability of the convexity --t0 dir: T0 or T5 shape, never a sub-sample
# --------------------------------------------------------------------------------------


def _write_convexity_artifacts(t0: Path, n_windows: int = 6, window: int = 21) -> Path:
    """Minimal folds.json / returns.csv / score_close.csv so convexity_read_dir can run; returns a B1 path."""
    n_rows = n_windows * window + 1
    index = pd.bdate_range("2019-01-02", periods=n_rows, tz=TREND_TZ)
    _window_returns(index, {1: -0.001}, 0.0005).to_csv(t0 / "returns.csv")
    pd.DataFrame({"SPY": np.linspace(100.0, 110.0, n_rows), "TLT": 50.0}, index=index).to_csv(t0 / "score_close.csv")
    folds = [{"fold": 0, "decision_bars": [index[k * window].isoformat() for k in range(n_windows)]}]
    (t0 / "folds.json").write_text(json.dumps(folds))
    b1_path = t0.parent / "b1_returns.csv"
    pd.Series(0.0004, index=index, name="daily_return").to_csv(b1_path)
    return b1_path


def _set_end_date(run_dir: Path, text: str) -> None:
    cfg = json.loads((run_dir / "config.json").read_text())
    cfg["end_date"] = text
    (run_dir / "config.json").write_text(json.dumps(cfg))


def test_t5_dir_readable_for_convexity_not_fragility(tmp_path: Path) -> None:
    t5_text = "2027-07-20T00:00:00-04:00"
    sub_text = "2021-12-31T00:00:00-05:00"
    t0_text = "2026-07-17T00:00:00-04:00"

    # registered_cell_problems: strict vs allow_t5.
    t0, trials, ledger = _registered_set(tmp_path / "cells", [0.4, 0.3, 0.2, 0.1, 0.1], [0.1] * 5)
    assert json.loads((t0 / "config.json").read_text())["end_date"] == t0_text
    assert adj.registered_cell_problems(t0) == [] and adj.registered_cell_problems(t0, allow_t5=True) == []
    _set_end_date(t0, t5_text)
    assert adj.registered_cell_problems(t0, allow_t5=True) == []
    strict = adj.registered_cell_problems(t0)
    assert len(strict) == 1 and "end_date" in strict[0]
    _set_end_date(t0, sub_text)
    assert any("end_date" in p for p in adj.registered_cell_problems(t0))
    assert any("end_date" in p for p in adj.registered_cell_problems(t0, allow_t5=True))
    # A fallback dir is refused under allow_t5 too (the convention checks do not relax).
    _set_end_date(t0, t5_text)
    cfg = json.loads((t0 / "config.json").read_text())
    cfg["price_return_fallback"] = True
    (t0 / "config.json").write_text(json.dumps(cfg))
    assert any("price_return_fallback" in p for p in adj.registered_cell_problems(t0, allow_t5=True))

    # fragility_read stays strict: a T5-shaped T0 dir is unreadable for fragility.
    t0, trials, ledger = _registered_set(tmp_path / "frag", [0.4, 0.3, 0.2, 0.1, 0.1], [0.1] * 5)
    _set_end_date(t0, t5_text)
    read = adj.fragility_read(t0, trials, ledger)
    assert read["readable"] is False and "end_date" in read["reason"]

    # main(): a T5 dir is readable for convexity, and inputs.t0_end_date shows it is a T5 read.
    t5_dir = tmp_path / "t5" / "trend_t5"
    _write_cell(t5_dir, trend_wfo._config_payload(trend_wfo.parse_args([])), 0.4, 0.2)
    _set_end_date(t5_dir, t5_text)
    b1_path = _write_convexity_artifacts(t5_dir)
    report = adj.main(["--t0", str(t5_dir), "--b1_returns", str(b1_path)])
    assert report["convexity"]["readable"] is True
    assert report["inputs"]["t0_end_date"] == t5_text
    assert report["fragility"]["readable"] is False  # no --trials
    assert report["verdict"] != "unreadable: T0 dir is not a registered cell"

    # main(): a sub-sample dir stays unreadable for convexity.
    sub_dir = tmp_path / "sub" / "trend_t0"
    _write_cell(sub_dir, trend_wfo._config_payload(trend_wfo.parse_args([])), 0.4, 0.2)
    _set_end_date(sub_dir, sub_text)
    b1_path = _write_convexity_artifacts(sub_dir)
    report = adj.main(["--t0", str(sub_dir), "--b1_returns", str(b1_path)])
    assert report["convexity"]["readable"] is False and "end_date" in report["convexity"]["reason"]
    assert report["inputs"]["t0_end_date"] == sub_text
    assert report["verdict"] == "unreadable: T0 dir is not a registered cell"


# --------------------------------------------------------------------------------------
# 13. read-only
# --------------------------------------------------------------------------------------


def test_adjudicate_writes_nothing_but_out(tmp_path: Path) -> None:
    window = 21
    n_rows = 6 * window + 1
    index = pd.bdate_range("2019-01-02", periods=n_rows, tz=TREND_TZ)
    t0, trials, ledger = _registered_set(tmp_path, [0.4, 0.3, 0.2, 0.1, 0.1], [0.2, 0.1, 0.1, 0.05, 0.05])
    returns = _window_returns(index, {1: -0.001}, 0.0005)
    returns.to_csv(t0 / "returns.csv")
    score_close = pd.DataFrame({"SPY": np.linspace(100.0, 110.0, n_rows), "TLT": 50.0}, index=index)
    score_close.to_csv(t0 / "score_close.csv")
    folds = [{"fold": 0, "decision_bars": [index[k * window].isoformat() for k in range(6)]}]
    (t0 / "folds.json").write_text(json.dumps(folds))
    b1_path = tmp_path / "b1_returns.csv"
    pd.Series(0.0004, index=index, name="daily_return").to_csv(b1_path)

    before = ledger.read_bytes()
    listing_before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    out_path = tmp_path / "read" / "adjudication.json"
    report = adj.main(
        [
            "--t0",
            str(t0),
            "--trials",
            ",".join(str(d) for d in trials),
            "--ledger",
            str(ledger),
            "--b1_returns",
            str(b1_path),
            "--out",
            str(out_path),
        ]
    )
    assert ledger.read_bytes() == before
    listing_after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    new = set(listing_after) - set(listing_before)
    assert new == {Path("read"), Path("read") / "adjudication.json"}
    written = json.loads(out_path.read_text())
    assert written["fragility"]["readable"] is True
    assert written["convexity"]["readable"] is True
    assert written["convexity"]["full_sample"]["n_windows"] == 6  # d = 0..105 each has 21 complete rows
    assert written["verdict"] == report["verdict"]
    assert written["convexity"]["oos_segment"]["pass"] is None  # no bar at or after 2026-07-20 here
