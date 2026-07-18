"""Carry-mode fold-flatten counterfactual for a finished walk-forward run (program review finding 3).

The research backtest force-flattens the book on the last two bars of every
fold (``research/arbitrage/walk_forward.py``), charging full round-trip
costs; the live loop never flattens. A carry-mode run is impossible in-tree
(the walk-forward CLI rejects it), so this diagnostic quantifies the
artifact *without* touching counted machinery: the recorded
``target_weights.csv`` already encodes every signal/band/cap/cadence
decision, and the counterfactual replays the frozen weight-space accounting
engine (``prism.execution.target_weights.backtest_target_weights``) with
fold-boundary targets replaced by carry semantics. Reads ONLY committed
artifacts (the run directory and the bar caches, quarantine included for
names quarantined after the run); never appends to the trials ledger, never
invokes walk-forward code.

Validation gate, asserted before any counterfactual is read: replaying the
RECORDED targets must reproduce the run's recorded ``returns.csv`` to
``--tolerance`` (bit-noise scale). If the gate fails, the replay does not
implement the run's accounting and no number below it means anything.

Counterfactuals (both reported; A is the headline):

- **A — carry interior boundaries, keep terminal flatten:** at each interior
  fold boundary the two zeroed target rows are replaced by the fold's last
  held book, so the next fold's opening targets trade *from* that book
  instead of from flat. The terminal flatten is a one-off window-edge
  artifact, not a recurring live divergence, so A isolates the systematic
  component a live-vs-certified comparison consumes.
- **B — carry every boundary including the terminal one:** pure
  mark-to-market comparison, reported for completeness.

Known approximation: the recorded fold-opening targets were themselves
produced by a band process starting from flat, so a true carry regime would
additionally hold small legacy deviations the recorded targets trade away —
the measured cost saving is a lower bound on the artifact.

Uncounted diagnostic: searches nothing, changes no trial code path, writes
one JSON. Frame and results: ``docs/carry_flatten_diagnostic.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from prism.config import ExecutionConfig
from prism.execution.target_weights import backtest_target_weights
from prism.residual.factors import consensus_trading_days

DELTA_KEYS = (
    "sharpe",
    "periodic_sharpe",
    "annualized_vol",
    "annualized_return",
    "max_drawdown",
    "skew_daily",
    "avg_gross",
    "avg_turnover",
    "total_cost",
    "total_cost_bps_per_year",
    "total_return",
)


def load_run_frame(run_dir: Path, name: str) -> pd.DataFrame:
    """A run CSV on its tz-aware exchange calendar (the index the caches use)."""
    frame = pd.read_csv(run_dir / name, index_col=0)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_convert(
        "America/New_York"
    )
    return frame


def fold_bounds(folds: list[dict], test_index: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """Inclusive (start, end) row positions of each fold's test window.

    Cross-checked row-by-row against the recorded fold dates so a
    misaligned ``folds.json`` fails loudly instead of silently shifting
    every boundary.
    """
    bounds: list[tuple[int, int]] = []
    pos = 0
    for f in folds:
        rows = int(f["test_rows"])
        assert rows >= 3, f"fold {f['fold']} has {rows} test rows; need >= 3 to carry a book"
        start, end = pos, pos + rows - 1
        assert test_index[start].isoformat() == f["test_start"], (
            f"fold {f['fold']} test_start mismatch at row {start}"
        )
        assert test_index[end].isoformat() == f["test_end"], (
            f"fold {f['fold']} test_end mismatch at row {end}"
        )
        bounds.append((start, end))
        pos += rows
    assert pos == len(test_index), f"folds cover {pos} rows, run has {len(test_index)}"
    return bounds


def assert_fold_flattened(weights_arr: np.ndarray, bounds: list[tuple[int, int]]) -> None:
    """Every fold must end with the recorded two-bar flatten and a non-empty book before it."""
    for k, (_, end) in enumerate(bounds):
        assert np.all(weights_arr[end] == 0.0) and np.all(weights_arr[end - 1] == 0.0), (
            f"fold {k} last two target rows are not flattened; nothing to counterfact"
        )
        assert np.abs(weights_arr[end - 2]).sum() > 0.0, f"fold {k} has no book to carry"


def boundary_decomposition(
    weights_arr: np.ndarray, bounds: list[tuple[int, int]]
) -> tuple[list[dict], dict]:
    """Per-interior-boundary turnover under flatten vs carry semantics, plus aggregates."""
    rows: list[dict] = []
    for k in range(len(bounds) - 1):
        _, end = bounds[k]
        next_start, _ = bounds[k + 1]
        book = weights_arr[end - 2]
        next_book = weights_arr[next_start]
        rows.append(
            {
                "boundary": k,
                "gross_B": float(np.abs(book).sum()),
                "gross_Bnext": float(np.abs(next_book).sum()),
                "turnover_flatten": float(np.abs(book).sum() + np.abs(next_book).sum()),
                "turnover_carry": float(np.abs(next_book - book).sum()),
            }
        )
    frame = pd.DataFrame(rows)
    saved = frame["turnover_flatten"] - frame["turnover_carry"]
    aggregate = {
        "n_interior_boundaries": int(len(frame)),
        "avg_gross_at_fold_end": float(frame["gross_B"].mean()),
        "avg_gross_next_open_book": float(frame["gross_Bnext"].mean()),
        "avg_turnover_flatten_per_boundary": float(frame["turnover_flatten"].mean()),
        "avg_turnover_carry_per_boundary": float(frame["turnover_carry"].mean()),
        "avg_turnover_saved_per_boundary": float(saved.mean()),
        "total_turnover_saved": float(saved.sum()),
        "carry_over_flatten_ratio": float(
            frame["turnover_carry"].sum() / frame["turnover_flatten"].sum()
        ),
    }
    return rows, aggregate


def carry_targets(
    weights: pd.DataFrame, bounds: list[tuple[int, int]], *, carry_terminal: bool
) -> pd.DataFrame:
    """Recorded targets with fold-boundary flatten rows replaced by the fold's last held book."""
    out = weights.copy()
    carried = bounds if carry_terminal else bounds[:-1]
    for _, end in carried:
        out.iloc[end - 1] = weights.iloc[end - 2].to_numpy()
        out.iloc[end] = weights.iloc[end - 2].to_numpy()
    return out


def run_metrics(returns: pd.Series, costs: pd.DataFrame) -> dict:
    """Summary statistics on a replayed daily-return stream (252-day annualization)."""
    equity = (1.0 + returns).cumprod()
    drawdown = float((1.0 - equity / equity.cummax()).max())
    n = len(returns)
    years = n / 252.0
    total = float((1.0 + returns).prod() - 1.0)
    return {
        "total_return": total,
        "annualized_return": float((1.0 + total) ** (252.0 / n) - 1.0),
        "annualized_vol": float(returns.std(ddof=1) * np.sqrt(252.0)),
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)),
        "periodic_sharpe": float(returns.mean() / returns.std(ddof=1)),
        "max_drawdown": drawdown,
        "skew_daily": float(returns.skew()),
        "kurtosis_daily": float(returns.kurt()),
        "worst_day": float(returns.min()),
        "avg_gross": float(costs["gross"].mean()),
        "avg_turnover": float(costs["turnover"].mean()),
        "total_cost": float(costs["total"].sum()),
        "total_cost_bps_per_year": float(costs["total"].sum() / years * 1e4),
        "total_borrow": float(costs["borrow"].sum()),
        "years": years,
    }


def bucket_spreads(median_dollar_volume: pd.Series, buckets: list[list[float]]) -> pd.Series:
    """Per-name spread bps from the run's dollar-volume bucket schedule."""
    schedule = [(float(floor), float(bps)) for floor, bps in buckets]
    values = median_dollar_volume.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        spread = np.select(
            [values >= floor for floor, _ in schedule[:-1]],
            [bps for _, bps in schedule[:-1]],
            default=schedule[-1][1],
        )
    return pd.Series(spread, index=median_dollar_volume.index, dtype=float)


def load_price_panel(
    symbols: list[str], data_dir: Path, quarantine_dir: Path, cache_suffix: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Open/close/volume panels from the run's own bar caches, consensus-calendar filtered.

    Quarantined caches are read from ``quarantine_dir`` because the run
    consumed them before quarantine; reproducing its accounting requires the
    same bars, wrong instrument or not.
    """
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for sym in symbols:
        path = data_dir / f"{sym}{cache_suffix}"
        if not path.exists():
            path = quarantine_dir / f"{sym}{cache_suffix}"
        if not path.exists():
            missing.append(sym)
            continue
        bars = pd.read_parquet(path)
        if not bars.index.is_unique:
            bars = bars[~bars.index.duplicated(keep="last")]
        frames[sym] = bars.sort_index()

    closes = pd.DataFrame({s: f["close"].astype(float) for s, f in frames.items()}).sort_index()
    closes = closes.dropna(how="all")
    opens = pd.DataFrame({s: f["open"].astype(float) for s, f in frames.items()}).reindex(
        closes.index
    )
    volumes = pd.DataFrame({s: f["volume"].astype(float) for s, f in frames.items()}).reindex(
        closes.index
    )
    trading_days = consensus_trading_days(closes)
    return closes.loc[trading_days], opens.loc[trading_days], volumes.loc[trading_days], missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", default="results/demotion_b1", help="Finished run directory")
    parser.add_argument("--data_dir", default="data", help="Bar-cache directory")
    parser.add_argument(
        "--quarantine_dir",
        default="data/quarantine",
        help="Fallback cache directory for names quarantined after the run",
    )
    parser.add_argument(
        "--cache_suffix",
        default=None,
        help="Cache filename suffix (default: _1d_<start_date>_<end_date>.parquet from config)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="Validation-gate ceiling on |replayed - recorded| daily returns",
    )
    parser.add_argument(
        "--out", default=None, help="Output JSON (default: results/carry_flatten_diagnostic_<today>.json)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    folds = json.loads((run_dir / "folds.json").read_text())
    cache_suffix = args.cache_suffix or f"_1d_{config['start_date']}_{config['end_date']}.parquet"

    assert config["execution"]["adv_impact_coeff"] == 0.0, (
        "run used the dollar-volume impact path; this replay does not feed dollar_volume"
    )

    weights = load_run_frame(run_dir, "target_weights.csv")
    recorded_returns = load_run_frame(run_dir, "returns.csv")["daily_return"]
    assert not weights.isna().any().any(), "recorded targets contain NaN (drop markers?)"
    n_test = len(weights)

    closes, opens, volumes, missing = load_price_panel(
        config["symbols"], Path(args.data_dir), Path(args.quarantine_dir), cache_suffix
    )
    for sym in missing:
        if sym in weights.columns:
            assert not (weights[sym] != 0.0).any(), f"missing cache for held name {sym}"

    full_index = closes.index
    start_pos = int(full_index.get_indexer([weights.index[0]])[0])
    assert start_pos > 0, "no formation rows before the test window"
    assert full_index[start_pos : start_pos + n_test].equals(weights.index), (
        "test window is not a contiguous slice of the cache calendar"
    )
    assert start_pos == int(folds[0]["formation_rows"]), (
        f"panel formation rows {start_pos} != recorded fold-0 formation_rows"
    )

    # First-formation-window bucket spreads: the backtest prices each name's
    # spread from its median dollar volume over the initial formation window.
    median_dollar_volume = (closes * volumes).iloc[0:start_pos].median(axis=0, skipna=True)
    spread_bps = bucket_spreads(median_dollar_volume, config["spread_schedule"]["buckets"])
    execution = ExecutionConfig(**config["execution"])

    engine_symbols = [s for s in weights.columns if s in closes.columns]
    dropped_columns = [s for s in weights.columns if s not in closes.columns]

    def run_engine(targets_test: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        full_targets = pd.DataFrame(0.0, index=full_index, columns=engine_symbols)
        full_targets.loc[targets_test.index, engine_symbols] = targets_test[
            engine_symbols
        ].to_numpy()
        result = backtest_target_weights(
            opens[engine_symbols],
            full_targets,
            execution=execution,
            initial_capital=1.0,
            spread_bps_per_name=spread_bps.reindex(engine_symbols),
        )
        return result.returns.loc[weights.index], result.costs.loc[weights.index]

    # ---- validation gate: recorded targets must reproduce recorded returns ----
    base_returns, base_costs = run_engine(weights)
    diff = (base_returns - recorded_returns).abs()
    max_diff = float(diff.max())
    assert max_diff <= args.tolerance, (
        f"validation gate FAILED: replay does not reproduce recorded returns "
        f"(max abs diff {max_diff:.3e} > {args.tolerance:.0e}); counterfactual not run"
    )
    base_metrics = run_metrics(base_returns, base_costs)
    validation = {
        "max_abs_return_diff": max_diff,
        "mean_abs_return_diff": float(diff.mean()),
        "tolerance": args.tolerance,
        "recorded_sharpe": summary["sharpe"],
        "replayed_sharpe": base_metrics["sharpe"],
        "recorded_total_cost": summary["total_cost"],
        "replayed_total_cost": base_metrics["total_cost"],
        "dropped_all_zero_columns": dropped_columns,
        "missing_caches": missing,
    }

    bounds = fold_bounds(folds, weights.index)
    weights_arr = weights[engine_symbols].to_numpy(dtype=float)
    assert_fold_flattened(weights_arr, bounds)
    per_boundary, decomposition = boundary_decomposition(weights_arr, bounds)

    carry_a_returns, carry_a_costs = run_engine(
        carry_targets(weights, bounds, carry_terminal=False)
    )
    carry_b_returns, carry_b_costs = run_engine(carry_targets(weights, bounds, carry_terminal=True))
    carry_a = run_metrics(carry_a_returns, carry_a_costs)
    carry_b = run_metrics(carry_b_returns, carry_b_costs)

    certified = {
        "sharpe": summary["sharpe"],
        "annualized_vol": summary["annualized_vol"],
        "annualized_return": summary["annualized_return"],
        "max_drawdown": summary["max_drawdown"],
        "avg_gross": summary["avg_gross"],
        "avg_turnover": summary["avg_turnover"],
        "total_cost": summary["total_cost"],
        "total_return": summary["total_return"],
        "skew_daily": float(recorded_returns.skew()),
        "kurtosis_daily": float(recorded_returns.kurt()),
        "worst_day": float(recorded_returns.min()),
        "oos_periodic_sharpe": summary["oos_periodic_sharpe"],
    }

    payload = {
        "meta": {
            "generated": date.today().isoformat(),
            "run_dir": str(run_dir),
            "cache_suffix": cache_suffix,
            "n_test_rows": n_test,
            "n_folds": len(folds),
            "note": (
                "fold-opening targets were band-stepped from flat in the recorded run; "
                "the measured cost saving is a lower bound on the flatten artifact"
            ),
        },
        "validation": validation,
        "boundary_decomposition": decomposition,
        "baseline_replayed": base_metrics,
        "carry_interior_terminal_flatten": carry_a,
        "carry_all_boundaries": carry_b,
        "certified_recorded": certified,
        "deltas_carryA_minus_base": {k: carry_a[k] - base_metrics[k] for k in DELTA_KEYS},
        "deltas_carryB_minus_base": {k: carry_b[k] - base_metrics[k] for k in DELTA_KEYS},
        "per_boundary": per_boundary,
    }
    out_path = (
        Path(args.out)
        if args.out
        else Path("results") / f"carry_flatten_diagnostic_{date.today().isoformat()}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "per_boundary"}, indent=2))


if __name__ == "__main__":
    main()
