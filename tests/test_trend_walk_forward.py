"""Offline synthetic tests for the trend_v1 walk-forward core and its counted CLI driver.

Numbering follows the implementation spec §5. Test 6b reads the repo's
``data/distributions`` read-only (skip-guarded); everything else is synthetic.
No test appends to the family ledger under ``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.config import PROJECT_DIR, ExecutionConfig
from prism.execution.target_weights import backtest_target_weights
from prism.portfolio.construct import construct_inverse_vol_targets
from prism.signal.trend_node import TREND_V1_UNIVERSE, TrendSignalNode
from prism.validation.trials import validate_claim_packet_dir
from research.arbitrage.trend_walk_forward import (
    PINNED_CACHE_FILES,
    TREND_TZ,
    CacheOverlapDisagreement,
    TrendSleeveConfig,
    construct_targets,
    listing_entry_mask,
    load_distributions,
    load_symbol_bars,
    resolve_end_date,
    run_trend_walk_forward,
    score_panel_tsmom,
    total_return_close,
)
from research.arbitrage.walk_forward import StatArbWalkForwardConfig, iter_walk_forward_slices
from research.scripts import trend_wfo
from research.scripts.joint_crash_receipt import decision_grid_mask
from research.scripts.joint_crash_receipt import score_panel_tsmom as receipt_score_panel
from research.scripts.joint_crash_receipt import trend_sleeve_returns

LEDGER_KEYS = {
    "ts",
    "strategy",
    "config_hash",
    "oos_periodic_sharpe",
    "oos_annualized_sharpe",
    "n_obs",
    "n_folds",
    "n_symbols",
    "design_trials",
    "output_dir",
    "config",
}


# --------------------------------------------------------------------------------------
# Synthetic helpers
# --------------------------------------------------------------------------------------


def _calendar(n: int, start: str = "2015-01-05") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, tz=TREND_TZ)


def _random_walk(index: pd.DatetimeIndex, symbols: list[str], *, seed: int, drift: float | np.ndarray = 0.0002):
    rng = np.random.default_rng(seed)
    n, k = len(index), len(symbols)
    drifts = np.broadcast_to(np.asarray(drift, dtype=float), (k,))
    rets = drifts[None, :] + rng.normal(0.0, 0.01, size=(n, k))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=index, columns=symbols)
    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=(n, k)))
    volume = pd.DataFrame(rng.integers(500_000, 2_000_000, size=(n, k)).astype(float), index=index, columns=symbols)
    return close, open_, volume


def _flat_walk() -> StatArbWalkForwardConfig:
    return StatArbWalkForwardConfig(
        formation_bars=315,
        test_bars=63,
        min_test_bars=20,
        max_gross=1.0,
        max_symbol_abs_weight=1.0,
        no_trade_band=0.0,
        band_mode="fixed",
        spread_mode="flat",
        max_participation=0.0,
    )


def _pinned_walk() -> StatArbWalkForwardConfig:
    return trend_wfo._walk_config()


def _write_parquet(path: Path, index: pd.DatetimeIndex, close: np.ndarray, open_: np.ndarray, volume: np.ndarray) -> None:
    frame = pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close), "close": close, "volume": volume},
        index=index,
    )
    frame.to_parquet(path)


@pytest.fixture
def synthetic_data_tree(tmp_path: Path) -> dict[str, object]:
    """Ten-name parquet caches under the exact PINNED_CACHE_FILES names, ending 2021-12-31.

    SPY's two files overlap 10 sessions exactly; every other pair is disjoint. 2021-12-30
    is a synthetic holiday (guard-6 negative case). PDBC lists late (listing rule).
    Dividend-paying names carry positive drift so the book is long at their ex-dates.
    """
    full = pd.bdate_range(end="2021-12-31", periods=901, tz=TREND_TZ)
    holiday = pd.Timestamp("2021-12-30", tz=TREND_TZ)
    calendar = full[full != holiday]
    assert calendar[-1] == pd.Timestamp("2021-12-31", tz=TREND_TZ)
    symbols = list(TREND_V1_UNIVERSE)
    drift = np.array([0.0012 if s in ("SPY", "TLT", "IEF", "LQD", "HYG", "PDBC") else -0.0004 for s in symbols])
    close, open_, volume = _random_walk(calendar, symbols, seed=2026, drift=drift)
    # PDBC lists late: no bars before session 200.
    late_start = 200
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n = len(calendar)
    split = 450
    for sym in symbols:
        first_file, second_file = PINNED_CACHE_FILES[sym]
        lo = late_start if sym == "PDBC" else 0
        a_end = split + 10 if sym == "SPY" else split
        rows_a = slice(lo, a_end)
        rows_b = slice(split, n)
        for path, rows in ((data_dir / first_file, rows_a), (data_dir / second_file, rows_b)):
            _write_parquet(
                path,
                calendar[rows],
                close[sym].to_numpy()[rows],
                open_[sym].to_numpy()[rows],
                volume[sym].to_numpy()[rows],
            )
    dist_dir = tmp_path / "distributions"
    dist_dir.mkdir()
    records: dict[str, list[tuple[str, float]]] = {s: [] for s in symbols}
    for sym in ("SPY", "TLT", "IEF", "LQD", "HYG"):
        for pos in range(300, n, 63):
            records[sym].append((calendar[pos].date().isoformat(), 0.5))
    records["PDBC"].append((calendar[700].date().isoformat(), 1.0))
    records["EFA"].append((calendar[500].date().isoformat(), 0.3))
    records["EEM"].append((calendar[520].date().isoformat(), 0.2))
    records["UUP"].append((calendar[540].date().isoformat(), 0.1))
    for sym in symbols:
        lines = ["ex_date,pay_date,amount"]
        for ex, amount in records[sym]:
            lines.append(f"{ex},{ex},{amount}")
        (dist_dir / f"{sym}.csv").write_text("\n".join(lines) + "\n")
        (dist_dir / f"{sym}.provenance.json").write_text(json.dumps({"symbol": sym, "source": "synthetic"}))
    return {
        "data_dir": data_dir,
        "dist_dir": dist_dir,
        "calendar": calendar,
        "n_sess": int(n),
        "names_with_records": [s for s in symbols if records[s]],
        "holiday": "2021-12-30",
    }


def _scratch_args(tree: dict[str, object], tmp_path: Path, tag: str) -> tuple[list[str], Path, Path]:
    out = tmp_path / f"out_{tag}"
    ledger = tmp_path / f"ledger_{tag}.jsonl"
    argv = [
        "--data_dir",
        str(tree["data_dir"]),
        "--distributions_dir",
        str(tree["dist_dir"]),
        "--end_date",
        "2021-12-31",
        "--output_dir",
        str(out),
        "--trial_ledger",
        str(ledger),
    ]
    return argv, out, ledger


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    results = project / "results"
    results.mkdir(parents=True)
    monkeypatch.setattr(trend_wfo, "PROJECT_DIR", project)
    monkeypatch.setattr(trend_wfo, "RESULTS_DIR", results)
    monkeypatch.setattr(trend_wfo, "TREND_LEDGER_DEFAULT", results / "trend_v1_trials.jsonl")
    return results, results / "trend_v1_trials.jsonl"


# --------------------------------------------------------------------------------------
# 1. signal parity
# --------------------------------------------------------------------------------------


def test_score_panel_equals_receipt_and_node() -> None:
    close, _, _ = _random_walk(_calendar(400), ["A", "B", "C"], seed=1)
    close.iloc[:30, 1] = np.nan  # a late lister keeps NaN through the ratio
    ours = score_panel_tsmom(close, lookback_bars=252, skip_bars=21)
    theirs = receipt_score_panel(close, lookback_bars=252, skip_bars=21)
    assert np.array_equal(ours.to_numpy(), theirs.to_numpy(), equal_nan=True)
    node = TrendSignalNode(lookback_bars=252, skip_bars=21, horizon_bars=21).score(close)
    assert np.allclose(ours.iloc[-1].to_numpy(), node.to_numpy(), atol=1e-12, rtol=0.0, equal_nan=True)


# --------------------------------------------------------------------------------------
# 2. causality
# --------------------------------------------------------------------------------------


def test_targets_causal_under_appended_future_bars() -> None:
    symbols = [f"S{i}" for i in range(6)]
    close, open_, volume = _random_walk(_calendar(740), symbols, seed=7)
    sleeve = TrendSleeveConfig(price_convention="price_return")
    walk = _pinned_walk()
    n = 700
    scores_full = score_panel_tsmom(close, lookback_bars=252, skip_bars=21).where(listing_entry_mask(close, 252))
    scores_short = score_panel_tsmom(close.iloc[:n], lookback_bars=252, skip_bars=21).where(
        listing_entry_mask(close.iloc[:n], 252)
    )
    t_full = construct_targets(scores_full, close, sleeve, walk)
    t_short = construct_targets(scores_short, close.iloc[:n], sleeve, walk)
    assert np.array_equal(t_full.iloc[:n].to_numpy(), t_short.to_numpy())

    kwargs = dict(sleeve=sleeve, walk=walk, execution=ExecutionConfig(), dividends=None)
    res_short = run_trend_walk_forward(
        close.iloc[:n], open_.iloc[:n], volume.iloc[:n], score_close=close.iloc[:n], **kwargs
    )
    res_full = run_trend_walk_forward(close, open_, volume, score_close=close, **kwargs)
    fold0 = next(iter_walk_forward_slices(n, walk))
    test_index = close.index[fold0.test]
    assert np.array_equal(
        res_short.portfolio.target_weights.loc[test_index].to_numpy(),
        res_full.portfolio.target_weights.loc[test_index].to_numpy(),
    )
    assert np.array_equal(
        res_short.portfolio.returns.loc[test_index].to_numpy(), res_full.portfolio.returns.loc[test_index].to_numpy()
    )
    assert res_short.folds[0].band == res_full.folds[0].band
    assert res_short.folds[0].spread_bps == res_full.folds[0].spread_bps


# --------------------------------------------------------------------------------------
# 3. listing rule
# --------------------------------------------------------------------------------------


def test_listing_rule_and_empty_cell_cash() -> None:
    symbols = ["A", "B", "C", "D"]
    close, open_, volume = _random_walk(_calendar(760), symbols, seed=3)
    late = 100
    for frame in (close, open_, volume):
        frame.iloc[:late, 2] = np.nan  # "C" lists 100 rows late
    sleeve = TrendSleeveConfig(price_convention="price_return", lookback_bars=126, skip_bars=21)
    walk = _pinned_walk()
    scores = score_panel_tsmom(close, lookback_bars=126, skip_bars=21).where(listing_entry_mask(close, sleeve.entry_bars))
    targets = construct_targets(scores, close, sleeve, walk)
    assert not targets.isna().any().any()
    entry = late + 252
    assert (targets["C"].iloc[:entry] == 0.0).all()
    assert (targets["C"].iloc[entry:].abs() > 0).any()
    # With lookback 126 the score for C is finite well before entry — the entry rule, not the lookback, gates it.
    raw_scores = score_panel_tsmom(close, lookback_bars=126, skip_bars=21)
    assert np.isfinite(raw_scores["C"].iloc[late + 126 : entry]).all()
    # Names listed at row 0 enter at 252 regardless of the shorter lookback.
    assert (targets["A"].iloc[:252] == 0.0).all()
    assert (targets["A"].iloc[252:].abs() > 0).any()

    res = run_trend_walk_forward(
        close, open_, volume, score_close=close, dividends=None, sleeve=sleeve, walk=walk, execution=ExecutionConfig()
    )
    assert not res.portfolio.target_weights.isna().any().any()
    assert res.entry_sessions["C"] == close.index[entry].isoformat()
    assert res.entry_sessions["A"] == close.index[252].isoformat()


# --------------------------------------------------------------------------------------
# 4. decision grid + all-zero exit
# --------------------------------------------------------------------------------------


def test_decision_grid_hold_and_all_zero_exit() -> None:
    symbols = ["A", "B", "C"]
    close, open_, volume = _random_walk(_calendar(300), symbols, seed=4)
    walk = StatArbWalkForwardConfig(
        formation_bars=84,  # > the 63-bar EWMA warm-up, so the first decision bar already has a finite sigma
        test_bars=63,
        min_test_bars=20,
        max_gross=1.0,
        max_symbol_abs_weight=1.0,
        no_trade_band=0.0,
        band_mode="fixed",
        spread_mode="flat",
        max_participation=0.0,
    )
    sleeve = TrendSleeveConfig(price_convention="price_return", lookback_bars=30, skip_bars=5, decision_every=21, entry_bars=30)
    fold0 = next(iter_walk_forward_slices(len(close), walk))
    ts = fold0.test.start  # 84; decision bars 84, 105, 126
    zero_bar = ts + 21
    score_close = close.copy()
    score_close.iloc[zero_bar - sleeve.skip_bars] = np.nan  # every name's score is NaN on that decision bar
    res = run_trend_walk_forward(
        close, open_, volume, score_close=score_close, dividends=None, sleeve=sleeve, walk=walk, execution=ExecutionConfig()
    )
    tw = res.portfolio.target_weights
    idx = close.index
    first_row = tw.loc[idx[ts]].to_numpy()
    assert (np.abs(first_row) > 0).any()
    for pos in range(ts + 1, zero_bar):
        assert np.array_equal(tw.loc[idx[pos]].to_numpy(), first_row)
    for pos in range(zero_bar, zero_bar + 21):
        assert (tw.loc[idx[pos]].to_numpy() == 0.0).all()
    fills = res.portfolio.fill_weights
    assert (np.abs(fills.loc[idx[zero_bar]].to_numpy()) > 0).any()  # still held on the decision row itself
    for pos in range(zero_bar + 1, zero_bar + 21):
        assert (fills.loc[idx[pos]].to_numpy() == 0.0).all()  # the zero target fills at the next open
    assert (np.abs(tw.loc[idx[zero_bar + 21]].to_numpy()) > 0).any()  # re-enters on the next decision bar


# --------------------------------------------------------------------------------------
# 5. explicit-file splice, N7 overlap check, tz handling
# --------------------------------------------------------------------------------------


def test_overlap_disagreement_fails_loud(tmp_path: Path) -> None:
    index = _calendar(120)
    close, open_, volume = _random_walk(index, ["Z"], seed=5)
    c = close["Z"].to_numpy()
    o = open_["Z"].to_numpy()
    v = volume["Z"].to_numpy()
    a_path = tmp_path / "Z_a.parquet"
    b_path = tmp_path / "Z_b.parquet"
    _write_parquet(a_path, index[:60], c[:60], o[:60], v[:60])
    last = index[-1]
    end = resolve_end_date(last.date().isoformat())

    _write_parquet(b_path, index[50:], c[50:] * (1.0 + 1e-4), o[50:], v[50:])
    with pytest.raises(CacheOverlapDisagreement, match="Z: 10 overlapping sessions disagree"):
        load_symbol_bars([a_path, b_path], end_date=end, symbol="Z")

    _write_parquet(b_path, index[50:], c[50:], o[50:], v[50:])
    frame, meta = load_symbol_bars([a_path, b_path], end_date=end, symbol="Z")
    assert len(frame) == 120 and frame.index.is_unique
    assert meta["overlap_sessions_checked"] == 10
    assert np.allclose(frame["close"].to_numpy(), c)

    # Within-file duplicate stamps: keep-last, counted.
    dup_index = index[:60].append(index[59:60])
    c_dup = np.concatenate([c[:60], [c[59] * 1.5]])
    _write_parquet(a_path, dup_index, c_dup, np.concatenate([o[:60], [o[59]]]), np.concatenate([v[:60], [v[59]]]))
    frame, meta = load_symbol_bars([a_path], end_date=end, symbol="Z")
    files = meta["files"]
    assert isinstance(files, list) and files[0]["dup_rows_dropped"] == 1 and files[0]["rows_unique"] == 60
    assert frame["close"].iloc[-1] == pytest.approx(c[59] * 1.5)

    # tz-naive end_date is refused: localisation is the driver's job.
    with pytest.raises(ValueError, match="tz-aware"):
        load_symbol_bars([b_path], end_date=pd.Timestamp(last.date().isoformat()), symbol="Z")
    # NY-midnight retains the last bar; a UTC midnight of the same date silently drops it.
    kept, _ = load_symbol_bars([b_path], end_date=end, symbol="Z")
    assert kept.index[-1] == last
    dropped, _ = load_symbol_bars([b_path], end_date=pd.Timestamp(last.date().isoformat(), tz="UTC"), symbol="Z")
    assert dropped.index[-1] == index[-2]


# --------------------------------------------------------------------------------------
# 6. total-return panel and dividend cash credit
# --------------------------------------------------------------------------------------


def test_total_return_panel_and_dividend_cash_credit(tmp_path: Path) -> None:
    index = _calendar(40)
    close = pd.DataFrame(100.0, index=index, columns=["A", "B"])
    opens = close.copy()
    e_pos = 10
    ex_date = index[e_pos]
    divs = pd.DataFrame(0.0, index=index, columns=["A", "B"])
    divs.loc[ex_date, "A"] = 1.0
    tr = total_return_close(close, divs)
    ratio = tr["A"] / close["A"]
    assert np.allclose(ratio.iloc[:e_pos], 1.0)
    assert np.allclose(ratio.iloc[e_pos:], 1.01)
    assert np.allclose(tr["B"], 100.0)

    zero_cost = ExecutionConfig(spread_bps=0.0, slippage_coeff=0.0, commission_bps=0.0, borrow_rate_bps_annual=0.0)
    targets = pd.DataFrame({"A": 1.0, "B": 0.0}, index=index)
    result = backtest_target_weights(opens, targets, execution=zero_cost, dividends=divs)
    credited_row = index[e_pos - 1]
    assert result.returns.loc[credited_row] == pytest.approx(1.0 / opens.loc[credited_row, "A"])
    assert result.costs["dividend_return"].sum() == pytest.approx(result.returns.loc[credited_row])
    assert (result.returns.drop(credited_row) == 0.0).all()

    # Loader semantics: weekend ex_date shifts to the next session; same-day rows summed; header-only OK.
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    saturday = (index[5] + pd.Timedelta(days=5 - index[5].weekday())).date().isoformat()
    assert pd.Timestamp(saturday).weekday() == 5
    same_day = index[20].date().isoformat()
    (dist_dir / "A.csv").write_text(
        "ex_date,pay_date,amount\n" f"{saturday},{saturday},0.25\n" f"{same_day},{same_day},0.5\n" f"{same_day},{same_day},0.75\n"
    )
    (dist_dir / "B.csv").write_text("ex_date,pay_date,amount\n")
    for sym in ("A", "B"):
        (dist_dir / f"{sym}.provenance.json").write_text("{}")
    frame, meta = load_distributions(
        dist_dir, ["A", "B"], index, first_session={"A": index[0], "B": index[0]}, end_date=resolve_end_date(index[-1].date().isoformat())
    )
    monday = index[index > pd.Timestamp(saturday, tz=TREND_TZ)][0]
    assert frame.loc[monday, "A"] == pytest.approx(0.25)
    assert frame.loc[index[20], "A"] == pytest.approx(1.25)
    assert frame["A"].sum() == pytest.approx(1.5)
    a_meta = meta["A"]
    assert isinstance(a_meta, dict)
    assert a_meta["n_shifted"] == 1 and a_meta["n_records_in_sample"] == 3 and a_meta["n_sessions_credited"] == 2
    b_meta = meta["B"]
    assert isinstance(b_meta, dict)
    assert (frame["B"] == 0.0).all() and b_meta["n_records_in_sample"] == 0 and b_meta["first_ex_date"] is None
    # Negative amounts are refused.
    (dist_dir / "B.csv").write_text(f"ex_date,pay_date,amount\n{same_day},{same_day},-0.1\n")
    with pytest.raises(SystemExit, match="negative or non-finite"):
        load_distributions(
            dist_dir, ["A", "B"], index, first_session={"A": index[0], "B": index[0]}, end_date=resolve_end_date(index[-1].date().isoformat())
        )


# --------------------------------------------------------------------------------------
# 6b. the real distribution files (read-only rehearsal)
# --------------------------------------------------------------------------------------


_REPO_DIST = PROJECT_DIR / "data" / "distributions"
_REPO_SPY_MIN = PROJECT_DIR / "data" / PINNED_CACHE_FILES["SPY"][0]


@pytest.mark.skipif(
    not ((_REPO_DIST / "PDBC.csv").exists() and _REPO_SPY_MIN.exists()),
    reason="repo distribution files / SPY min cache not present",
)
def test_load_distributions_on_repo_files() -> None:
    end = resolve_end_date("2026-07-17")
    bars = pd.read_parquet(_REPO_SPY_MIN)
    index = pd.DatetimeIndex(bars.index[~bars.index.duplicated(keep="last")]).sort_values()
    index = index[index <= end]
    first = {sym: index[0] for sym in TREND_V1_UNIVERSE}
    divs, meta = load_distributions(_REPO_DIST, TREND_V1_UNIVERSE, index, first_session=first, end_date=end)
    assert divs.at[pd.Timestamp("2021-12-03", tz=TREND_TZ), "PDBC"] == pytest.approx(5.39)
    gld = meta["GLD"]
    assert isinstance(gld, dict)
    assert (divs["GLD"] == 0.0).all() and gld["n_records_in_sample"] == 0
    for sym in TREND_V1_UNIVERSE:
        m = meta[sym]
        assert isinstance(m, dict)
        assert isinstance(m["n_shifted"], int)
        assert m["n_sessions_credited"] == int((divs[sym] != 0.0).sum())
        credited = divs.index[divs[sym] != 0.0]
        assert credited.isin(index).all()
    assert "verification" in meta


# --------------------------------------------------------------------------------------
# 7. flat-stack parity with the joint-crash receipt
# --------------------------------------------------------------------------------------


def test_flat_stack_parity_with_joint_crash_receipt() -> None:
    symbols = [f"S{i}" for i in range(6)]
    close, open_, volume = _random_walk(_calendar(700), symbols, seed=11)
    walk = _flat_walk()
    sleeve = TrendSleeveConfig(price_convention="price_return")
    execution = ExecutionConfig()
    scores = score_panel_tsmom(close, lookback_bars=252, skip_bars=21)
    grid = np.flatnonzero(decision_grid_mask(close.index, 21))
    for pos in grid[grid >= 252]:
        row = scores.iloc[pos].to_numpy()
        assert np.isfinite(row).all() and (row != 0.0).all()

    res = run_trend_walk_forward(
        close, open_, volume, score_close=close, dividends=None, sleeve=sleeve, walk=walk, execution=execution
    )
    receipt, _ = trend_sleeve_returns(open_, close, execution=execution)
    # Receipt replica for gross parity (the receipt returns net only): joint_crash_receipt.py:171-190.
    targets_full = construct_inverse_vol_targets(scores, close, vol_ewma_bars=63, max_gross=1.0, max_symbol_abs_weight=1.0)
    mask = decision_grid_mask(targets_full.index, 21)
    masked = targets_full.where(pd.Series(mask, index=targets_full.index), other=np.nan)
    all_zero = (masked.fillna(0.0).abs().sum(axis=1) == 0.0) & mask
    masked = masked.mask(all_zero, other=np.nan)
    replica = backtest_target_weights(open_, masked, execution=execution, initial_capital=1.0)
    assert np.allclose(replica.returns.to_numpy(), receipt.to_numpy(), atol=1e-15, rtol=0.0)

    fold0 = next(iter_walk_forward_slices(len(close), walk))
    ts, te = fold0.test.start, fold0.test.stop
    assert ts == 315 and ts % 21 == 0
    idx = close.index
    net_rows = idx[ts + 2 : te - 1]  # ts+2 .. te-2 inclusive
    gross_rows = idx[ts + 1 : te - 1]  # ts+1 .. te-2 inclusive
    ours_net = res.portfolio.returns.loc[net_rows].to_numpy()
    theirs_net = receipt.loc[net_rows].to_numpy()
    assert np.allclose(ours_net, theirs_net, atol=1e-12, rtol=0.0)
    ours_gross = (res.portfolio.returns + res.portfolio.costs["total"]).loc[gross_rows].to_numpy()
    theirs_gross = (replica.returns + replica.costs["total"]).loc[gross_rows].to_numpy()
    assert np.allclose(ours_gross, theirs_gross, atol=1e-12, rtol=0.0)
    # Documented exclusions: row ts+1 shares the fill but not the prior book (cost differs), row ts is flat here.
    first = idx[ts + 1]
    assert abs(res.portfolio.returns.loc[first] - receipt.loc[first]) > 1e-12
    assert res.portfolio.returns.loc[idx[ts]] == pytest.approx(0.0)
    assert receipt.loc[idx[ts]] != 0.0


# --------------------------------------------------------------------------------------
# 8. CLI packet + ledger row (uncounted shape, no monkeypatching)
# --------------------------------------------------------------------------------------


def test_cli_emits_valid_packet_and_ledger_row(synthetic_data_tree: dict[str, object], tmp_path: Path) -> None:
    argv, out, ledger = _scratch_args(synthetic_data_tree, tmp_path, "t8")
    trend_wfo.main(argv)
    packet = validate_claim_packet_dir(out)
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    assert len(rows) == 1 and set(rows[0]) == LEDGER_KEYS
    assert rows[0]["strategy"] == "trend_v1" and rows[0]["design_trials"] == 6
    summary = json.loads((out / "summary.json").read_text())
    assert "after_cost_hurdle" in summary and "dsr" in summary
    assert summary["fallback_reason"] is None and summary["counted"] is False
    assert summary["end_date"] == "2021-12-31T00:00:00-05:00"
    config = json.loads((out / "config.json").read_text())
    assert config["data_convention"] == "split_adjusted_total_return_dividends_as_cash"
    assert config["end_date"] == "2021-12-31T00:00:00-05:00"
    assert config["price_return_fallback"] is False
    assert packet["data"]["n_sessions"] == synthetic_data_tree["n_sess"]
    last = pd.Timestamp(packet["data"]["sample_cutoff"]["last_session"])
    assert last == pd.Timestamp("2021-12-31", tz=TREND_TZ)
    assert packet["data"]["fallback_reason"] is None
    assert packet["strategy"] == "trend_v1"
    dividends = pd.read_csv(out / "dividends.csv", index_col=0)
    for sym in synthetic_data_tree["names_with_records"]:  # type: ignore[union-attr]
        assert (dividends[sym] != 0.0).any()
    assert (dividends["GLD"] == 0.0).all()
    costs = pd.read_csv(out / "costs.csv", index_col=0)
    assert costs["dividend_return"].sum() > 0
    assert summary["total_dividend_return"] == pytest.approx(costs["dividend_return"].sum())
    folds = json.loads((out / "folds.json").read_text())
    assert summary["n_folds"] == len(folds)
    assert all(len(f["decision_bars"]) == -(-f["test_rows"] // 21) for f in folds)  # ceil(test_rows / 21)
    assert all(f["test_rows"] == 63 for f in folds[:-1]) and folds[0]["formation_rows"] == 312
    # The scoring panel ratchets up only at ex-dates: TR / price is non-decreasing per name.
    score_close = pd.read_csv(out / "score_close.csv", index_col=0)
    closes = pd.read_csv(out / "closes.csv", index_col=0)
    for sym in TREND_V1_UNIVERSE:
        ratio = (score_close[sym] / closes[sym]).dropna().to_numpy()
        assert (np.diff(ratio) >= -1e-12).all()
        assert ratio[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# 8b. price_return fallback gating
# --------------------------------------------------------------------------------------


def test_price_return_fallback_gating(
    synthetic_data_tree: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # (i) price_return without the flag
    argv, out, ledger = _scratch_args(synthetic_data_tree, tmp_path, "i")
    with pytest.raises(SystemExit, match="price_return is a fallback, not a cell"):
        trend_wfo.main(argv + ["--price_convention", "price_return"])
    assert not out.exists() and not ledger.exists()
    # (ii) flag + complete distributions
    argv, out, ledger = _scratch_args(synthetic_data_tree, tmp_path, "ii")
    with pytest.raises(SystemExit, match="distributions are complete"):
        trend_wfo.main(argv + ["--price_convention", "price_return", "--allow_price_return_fallback"])
    assert not out.exists() and not ledger.exists()
    # (iii) flag + empty distributions dir -> runs as a scratch diagnostic
    empty = tmp_path / "empty_dist"
    empty.mkdir()
    argv, out, ledger = _scratch_args(synthetic_data_tree, tmp_path, "iii")
    argv[argv.index("--distributions_dir") + 1] = str(empty)
    trend_wfo.main(argv + ["--price_convention", "price_return", "--allow_price_return_fallback"])
    packet = validate_claim_packet_dir(out)
    config = json.loads((out / "config.json").read_text())
    assert config["price_return_fallback"] is True
    assert config["data_convention"] == "split_adjusted_open_close_price_return_no_dividends"
    assert packet["data"]["fallback_reason"]["missing_distribution_files"] == sorted(TREND_V1_UNIVERSE)
    dividends = pd.read_csv(out / "dividends.csv", index_col=0)
    assert (dividends.to_numpy() == 0.0).all()
    costs = pd.read_csv(out / "costs.csv", index_col=0)
    assert costs["dividend_return"].sum() == 0.0
    # (iv) flag with total_return
    argv, out, ledger = _scratch_args(synthetic_data_tree, tmp_path, "iv")
    with pytest.raises(SystemExit, match="no silent no-op flags"):
        trend_wfo.main(argv + ["--allow_price_return_fallback"])
    assert not out.exists() and not ledger.exists()
    # (v) counted-path refusal: default ledger + registered end_date, refused by guard 4 before any read/write.
    _, default_ledger = _patch_paths(monkeypatch, tmp_path)
    argv, out, _ = _scratch_args(synthetic_data_tree, tmp_path, "v")
    argv[argv.index("--trial_ledger") + 1] = str(default_ledger)
    argv[argv.index("--end_date") + 1] = "2026-07-17"
    argv[argv.index("--distributions_dir") + 1] = str(empty)
    with pytest.raises(SystemExit, match="not a registered cell"):
        trend_wfo.main(argv + ["--price_convention", "price_return", "--allow_price_return_fallback"])
    assert not default_ledger.exists() and not out.exists()


# --------------------------------------------------------------------------------------
# 8c. end_date guards on both paths
# --------------------------------------------------------------------------------------


def test_end_date_guards(synthetic_data_tree: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir, default_ledger = _patch_paths(monkeypatch, tmp_path)

    # (i) default ledger + 2021-12-31 -> guard 2 counted branch
    argv, out, _ = _scratch_args(synthetic_data_tree, tmp_path, "c1")
    argv[argv.index("--trial_ledger") + 1] = str(default_ledger)
    with pytest.raises(SystemExit, match="not a registered cell on the counted path"):
        trend_wfo.main(argv)
    assert not default_ledger.exists() and not out.exists()

    # (ii) results-dir output + scratch ledger + 2021-12-31 -> counted -> guard 2
    argv, _, ledger = _scratch_args(synthetic_data_tree, tmp_path, "c2")
    out = results_dir / "trend_t0"
    argv[argv.index("--output_dir") + 1] = str(out)
    with pytest.raises(SystemExit, match="not a registered cell on the counted path"):
        trend_wfo.main(argv)
    assert not out.exists() and not ledger.exists()

    # (iii) scratch + scratch + 2021-12-31 -> runs
    argv, out, ledger = _scratch_args(synthetic_data_tree, tmp_path, "c3")
    trend_wfo.main(argv)
    packet = validate_claim_packet_dir(out)
    assert packet["data"]["n_sessions"] == synthetic_data_tree["n_sess"]
    assert packet["data"]["sample_cutoff"]["rule"].startswith("uncounted sub-sample")

    # (iv) default ledger + 2026-07-17 -> passes guard 2, refused by guard 6 (cache ends 2021-12-31)
    argv, out, _ = _scratch_args(synthetic_data_tree, tmp_path, "c4")
    argv[argv.index("--trial_ledger") + 1] = str(default_ledger)
    argv[argv.index("--end_date") + 1] = "2026-07-17"
    with pytest.raises(SystemExit) as exc:
        trend_wfo.main(argv)
    assert "cache ends 2021-12-31T00:00:00-05:00 but end_date is 2026-07-17T00:00:00-04:00" in str(exc.value)
    assert not default_ledger.exists() and not out.exists()

    # (v) --allow_post_ratification with 2026-07-17 -> guard 2 (T5 needs a later date); with 2027-07-19 -> guard 6
    argv, out, _ = _scratch_args(synthetic_data_tree, tmp_path, "c5")
    argv[argv.index("--trial_ledger") + 1] = str(default_ledger)
    argv[argv.index("--end_date") + 1] = "2026-07-17"
    with pytest.raises(SystemExit, match="not a T5 shape"):
        trend_wfo.main(argv + ["--allow_post_ratification"])
    argv[argv.index("--end_date") + 1] = "2027-07-19"
    with pytest.raises(SystemExit) as exc:
        trend_wfo.main(argv + ["--allow_post_ratification"])
    assert "cache ends 2021-12-31T00:00:00-05:00 but end_date is 2027-07-19T00:00:00-04:00" in str(exc.value)
    assert not default_ledger.exists() and not out.exists()

    # (vi) scratch + scratch + the default end_date -> the uncounted cap
    argv, out, ledger = _scratch_args(synthetic_data_tree, tmp_path, "c6")
    argv[argv.index("--end_date") + 1] = "2026-07-17"
    with pytest.raises(SystemExit) as exc:
        trend_wfo.main(argv)
    assert "TREND_UNCOUNTED_END_MAX" in str(exc.value) and "docs/trend_design.md:73-74" in str(exc.value)
    assert not out.exists() and not ledger.exists()
    # ... and --allow_post_ratification is refused on the uncounted path too.
    with pytest.raises(SystemExit, match="refused on the uncounted path"):
        trend_wfo.main(argv + ["--allow_post_ratification"])
    assert not out.exists() and not ledger.exists()

    # (vii) the boundary of the cap runs
    argv, out, ledger = _scratch_args(synthetic_data_tree, tmp_path, "c7")
    trend_wfo.main(argv)
    validate_claim_packet_dir(out)
    assert len(ledger.read_text().splitlines()) == 1

    # (viii) a non-session end_date on the uncounted path -> guard 6 uncounted branch
    argv, out, ledger = _scratch_args(synthetic_data_tree, tmp_path, "c8")
    argv[argv.index("--end_date") + 1] = str(synthetic_data_tree["holiday"])
    with pytest.raises(SystemExit, match="is not a session of the loaded panel"):
        trend_wfo.main(argv)
    assert not out.exists() and not ledger.exists()

    # (ix) an offset-carrying end_date is refused by resolve_end_date
    argv, out, ledger = _scratch_args(synthetic_data_tree, tmp_path, "c9")
    argv[argv.index("--end_date") + 1] = "2021-12-31T00:00:00+00:00"
    with pytest.raises(SystemExit, match="bare YYYY-MM-DD"):
        trend_wfo.main(argv)
    assert not out.exists() and not ledger.exists()


# --------------------------------------------------------------------------------------
# 9. single-knob deltas
# --------------------------------------------------------------------------------------


def _flat(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = json.dumps(v, sort_keys=True) if isinstance(v, list) else v
    return out


def test_trial_flags_move_exactly_one_config_field() -> None:
    t0 = _flat(trend_wfo._config_payload(trend_wfo.parse_args([])))
    trials = {
        "signal.lookback_bars": ["--lookback", "126"],
        "signal.skip_bars": ["--skip", "0"],
        "signal.decision_every": ["--decision_every", "63"],
        "construction.sizing": ["--sizing", "equal_notional"],
    }
    payloads = [t0]
    for key, argv in trials.items():
        tk = _flat(trend_wfo._config_payload(trend_wfo.parse_args(argv)))
        moved = {k for k in set(t0) | set(tk) if t0.get(k) != tk.get(k)}
        assert moved == {key}
        payloads.append(tk)
    for payload in payloads:
        assert payload["price_return_fallback"] is False
        assert payload["price_convention"] == "total_return"
        assert payload["end_date"] == "2026-07-17T00:00:00-04:00"
        assert payload["design_trials"] == 6
        text = json.dumps(payload, sort_keys=True)
        assert "TREND_UNCOUNTED_END_MAX" not in text and "uncounted" not in text
    assert "horizon_bars" not in json.dumps(t0)
    assert t0["walk.max_symbol_abs_weight"] == 1.0 and t0["walk.band_mode"] == "closed_form"
    assert t0["spread_accounting"] == "per_fold_bucket"


# --------------------------------------------------------------------------------------
# 10. budget guard
# --------------------------------------------------------------------------------------


def test_budget_guard_refuses_seventh_row(synthetic_data_tree: dict[str, object], tmp_path: Path) -> None:
    argv, out, ledger = _scratch_args(synthetic_data_tree, tmp_path, "budget")
    rows = [json.dumps({"strategy": "trend_v1", "oos_periodic_sharpe": 0.01 * k, "output_dir": f"x{k}"}) for k in range(6)]
    ledger.write_text("\n".join(rows) + "\n")
    before = ledger.read_bytes()
    with pytest.raises(SystemExit, match="budget of 6 is exhausted"):
        trend_wfo.main(argv)
    assert not out.exists()
    assert ledger.read_bytes() == before
