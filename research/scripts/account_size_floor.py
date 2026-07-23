"""Account-size floor: whole-share discretization of the decile book vs capital.

The live loop sizes whole-share OPG orders (venue requirement), so a small
account cannot hold the constructed ~100-name decile book: per-name target
notionals fall below one share and the dust filter censors them
(``prism.live.loop.targets_to_orders`` logs each censoring loudly, N7). This
diagnostic measures that discretization as a function of starting capital by
comparing replay runs that differ ONLY in ``--cash``: for each refresh it
reconstructs the *achieved* post-fill book from the replay's own ledgers
(``targets.jsonl``, ``fills.jsonl``) and scores it against the constructed
targets with the same ``book_concordance`` metric the live loop logs
(active share, gross held vs gross target), plus the count of names censored
out of the book entirely.

Reads ONLY replay run directories (``prism.scripts.replay_loop`` output —
modeled fills, diagnostic stream); never touches live ledgers, never appends
to the trials ledger, searches nothing. The cross-size *relative* comparison
is the measurement; the absolute window return is one sample of a
membership-blind replay and claims nothing (prism/live/replay.py).

Uncounted diagnostic: one JSON out. Frame and results:
``docs/account_size_floor.md``.

    python -m research.scripts.account_size_floor \\
        --runs 1000000=runs/replay_floor_1000000 --runs 10000=runs/replay_floor_10000 \\
        --output results/account_size_floor_2026-07-18.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from prism.live import (
    book_concordance,
    read_equity_ledger,
    read_fills_ledger,
    read_targets_ledger,
)
from prism.live.state import StateStore


def achieved_positions_by_refresh(
    targets_rows: list[dict], fills: pd.DataFrame
) -> list[dict[str, float]]:
    """Post-fill share positions after each refresh row, in ledger order.

    The replay trades only on refresh bars (hold cadence between), so the book
    after refresh *r* is the cumulative sum of every fill whose
    ``decision_bar`` is at or before that refresh bar.
    """
    books: list[dict[str, float]] = []
    positions: dict[str, float] = {}
    remaining = fills.sort_values("decision_bar") if not fills.empty else fills
    cursor = 0
    for row in targets_rows:
        bar = row["refresh_bar"]
        if not remaining.empty:
            while cursor < len(remaining) and remaining.iloc[cursor]["decision_bar"] <= bar:
                fill = remaining.iloc[cursor]
                symbol = str(fill["symbol"])
                positions[symbol] = positions.get(symbol, 0.0) + float(fill["qty"])
                cursor += 1
        books.append({s: q for s, q in positions.items() if q != 0.0})
    return books


def refresh_concordance(row: dict, achieved: dict[str, float]) -> dict:
    """Score one achieved post-fill book against its refresh targets.

    Weights are formed on the refresh row's own sizing basis (its equity and
    decision-close reference prices — exactly what ``targets_to_orders``
    sized with), so the divergence measured here is discretization, not
    price drift. A held name absent from the row's reference prices is
    reported, never silently dropped (N7).
    """
    targets = pd.Series(row["targets"], dtype=float)
    prices = row["reference_prices"]
    equity = float(row["equity"])
    priced = {s: q for s, q in achieved.items() if s in prices}
    unpriced = sorted(set(achieved) - set(priced))
    held = pd.Series(
        {s: q * float(prices[s]) / equity for s, q in priced.items()}, dtype=float
    )
    result = book_concordance(held, targets)
    censored = sorted(s for s, w in row["targets"].items() if w != 0.0 and s not in achieved)
    return {
        "refresh_bar": row["refresh_bar"],
        "active_share": result["active_share"],
        "gross_held": result["gross_held"],
        "gross_target": result["gross_target"],
        "weight_corr": result["weight_corr"],
        "target_names": sum(1 for w in row["targets"].values() if w != 0.0),
        "achieved_names": len(achieved),
        "censored_names": len(censored),
        "censored": censored,
        "unpriced_names": unpriced,
    }


def summarize_run(run_dir: Path) -> dict:
    """One replay run directory -> per-refresh concordance + terminal stats."""
    targets_rows = read_targets_ledger(run_dir / "targets.jsonl")
    if not targets_rows:
        raise ValueError(f"{run_dir}: no targets.jsonl rows — not a completed replay run")
    fills = read_fills_ledger(run_dir / "fills.jsonl")
    books = achieved_positions_by_refresh(targets_rows, fills)
    refreshes = [refresh_concordance(row, book) for row, book in zip(targets_rows, books)]

    equity = read_equity_ledger(run_dir / "equity.jsonl")
    state = StateStore(run_dir / "state.json").load()
    active_shares = [r["active_share"] for r in refreshes]
    gross_ratios = [
        r["gross_held"] / r["gross_target"] for r in refreshes if r["gross_target"] > 0
    ]
    return {
        "refreshes": refreshes,
        "active_share_mean": float(pd.Series(active_shares).mean()),
        "active_share_max": float(pd.Series(active_shares).max()),
        "gross_ratio_mean": float(pd.Series(gross_ratios).mean()),
        "censored_names_mean": float(pd.Series([r["censored_names"] for r in refreshes]).mean()),
        "first_bar": str(equity.iloc[0]["decision_bar"]),
        "last_bar": str(equity.iloc[-1]["decision_bar"]),
        "sessions": int(len(equity)),
        "initial_equity": float(equity.iloc[0]["equity"]),
        "terminal_equity": float(equity.iloc[-1]["equity"]),
        "total_return": float(equity.iloc[-1]["equity"] / equity.iloc[0]["equity"] - 1.0),
        "final_names": len(state.positions) if state is not None else None,
    }


def summarize(runs: dict[int, Path]) -> dict:
    """All runs, plus each run's terminal-return gap to the largest-cash baseline."""
    per_run = {cash: summarize_run(path) for cash, path in runs.items()}
    baseline_cash = max(per_run)
    baseline_return = per_run[baseline_cash]["total_return"]
    for cash, summary in per_run.items():
        summary["return_gap_vs_baseline"] = float(summary["total_return"] - baseline_return)
    return {
        "baseline_cash": baseline_cash,
        "runs": {str(cash): per_run[cash] for cash in sorted(per_run)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize whole-share discretization across replay runs differing only in cash."
    )
    parser.add_argument(
        "--runs",
        action="append",
        required=True,
        metavar="CASH=RUN_DIR",
        help="Starting cash and its replay run directory; repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path for the summary JSON.")
    args = parser.parse_args(argv)

    runs: dict[int, Path] = {}
    for spec in args.runs:
        cash_token, _, dir_token = spec.partition("=")
        if not dir_token:
            raise SystemExit(f"--runs expects CASH=RUN_DIR, got {spec!r}")
        runs[int(cash_token)] = Path(dir_token)

    result = summarize(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for cash in sorted(runs):
        run = result["runs"][str(cash)]
        print(
            f"cash {cash:>9,}: AS mean {run['active_share_mean']:.4f} max {run['active_share_max']:.4f}  "
            f"gross ratio {run['gross_ratio_mean']:.3f}  censored/refresh {run['censored_names_mean']:.1f}  "
            f"return {run['total_return']:+.4f} (gap {run['return_gap_vs_baseline']:+.4f})"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
