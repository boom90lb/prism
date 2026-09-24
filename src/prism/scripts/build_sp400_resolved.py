"""Build a point-in-time S&P MidCap 400 universe with its unrecorded exits resolved.

SPEC §5.2: breadth beyond the S&P 500 with the survivorship leak counted, not
hidden. The S&P 500 builder (`prism.scripts.build_sp500_universe`) is the
pattern; this script adds the one step the S&P 400 page needs that the S&P 500
page mostly does not. Wikipedia's S&P 400 "Selected past changes" table omits
many exits (promotions to the 500, demotions to the 600, acquisitions,
renames), and a forward sweep keeps every such interval open to the as-of
date. Those phantoms are closed on dated evidence, in this fixed order of
precedence-by-date (the earliest wins), decided before any return is read:

1. ``promoted_to_sp500`` -- the start of the ticker's next S&P 500 interval
   (`data/universe/sp500_membership_<asof>.parquet`);
2. ``added_to_sp600`` -- the ticker's addition date in the S&P 600 page's
   changes table;
3. ``vendor_last_bar`` -- the last bar the price vendor holds, when it ends at
   least ten sessions before the pull's end (an acquisition or delisting).

An interval with none of these is dropped and recorded ``unresolved``.
Prices are read only for existence and last date; no return enters any rule.

Artifacts under ``--out_dir`` (default ``data/universe``), all prefixed
``sp400_`` so nothing the live loop reads is touched:
``sp400_membership_<asof>.parquet`` (resolved), ``sp400_exits_<asof>.json``
(the exit ledger), ``sp400_coverage_<asof>.json`` (the price coverage ledger),
``sp400_pit_resolved_<asof>.txt`` (the price-resolved window universe).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from prism.io.loader import API_KEY_ENV_VAR, DataLoader
from prism.io.universe_sp500 import (
    QUARANTINE_TABLE,
    RENAME_TABLE,
    compute_coverage,
    extract_tables,
    members_active_between,
    normalize_ticker,
    parse_changes_table,
    parse_constituents_table,
    reconstruct_membership,
    resolve_unrecorded_exits,
    write_universe_file,
)
from prism.logging_utils import configure_logging
from prism.scripts.build_sp500_universe import _pull_prices
from prism.workspace import Workspace, load_env, resolve_workspace

logger = logging.getLogger(__name__)

#: A vendor series ending this many calendar days before the pull's end is
#: read as an exit; closer than that it is a live name with a stale last bar.
VENDOR_EXIT_SLACK_DAYS = 14


def _vendor(ticker: str) -> str:
    t = normalize_ticker(ticker)
    return normalize_ticker(RENAME_TABLE.get(t, t))


def public_evidence(sp500_membership: pd.DataFrame, sp600_changes: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """Dated exit evidence from the S&P 500 membership and the S&P 600 changes table."""
    evidence: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for row in sp500_membership.itertuples(index=False):
        evidence.setdefault(normalize_ticker(row.ticker), []).append((pd.Timestamp(row.start), "promoted_to_sp500"))
    for row in sp600_changes.itertuples(index=False):
        if row.added:
            evidence.setdefault(normalize_ticker(row.added), []).append((pd.Timestamp(row.date), "added_to_sp600"))
    return evidence


def vendor_evidence(
    tickers: list[str], data_dir: Path, *, start: str, end: str
) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """``vendor_last_bar`` evidence from the local bar caches of the pull window."""
    cutoff = pd.Timestamp(end) - pd.Timedelta(days=VENDOR_EXIT_SLACK_DAYS)
    evidence: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for ticker in tickers:
        path = data_dir / f"{_vendor(ticker)}_1d_{start}_{end}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        stamps = pd.to_datetime(frame.index if "timestamp" not in frame.columns else frame["timestamp"])
        last = pd.Timestamp(stamps.max())
        last = last.tz_localize(None) if last.tz is not None else last
        if last < cutoff:
            evidence.setdefault(ticker, []).append((last.normalize() + pd.Timedelta(days=1), "vendor_last_bar"))
    return evidence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asof", required=True, help="The reconstruction horizon (the page's as-of date).")
    p.add_argument("--sp400_html", required=True, help="Saved Wikipedia 'List of S&P 400 companies' page.")
    p.add_argument("--sp600_html", required=True, help="Saved Wikipedia 'List of S&P 600 companies' page.")
    p.add_argument("--sp500_membership", required=True, help="The resolved S&P 500 membership parquet.")
    p.add_argument("--start_date", default="2020-01-01")
    p.add_argument("--end_date", default="2026-06-16", help="The pull's end; the frozen S&P 500 caches end here.")
    p.add_argument("--workspace", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--skip_prices", action="store_true", help="Membership and public evidence only; no vendor calls.")
    p.add_argument("--rate_per_min", type=float, default=7.0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace: Workspace = resolve_workspace(args.workspace).ensure()
    load_env()
    configure_logging(workspace.logs, verbose=args.verbose, run_name="build_sp400_resolved")
    out_dir = Path(args.out_dir) if args.out_dir else workspace.data / "universe"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables400 = extract_tables(Path(args.sp400_html).read_text())
    current = parse_constituents_table(tables400)
    changes400 = parse_changes_table(tables400)
    sp600_changes = parse_changes_table(extract_tables(Path(args.sp600_html).read_text()))
    sp500 = pd.read_parquet(args.sp500_membership)

    raw = reconstruct_membership(current, changes400, end_date=args.asof)
    raw = raw.assign(ticker=raw["ticker"].map(normalize_ticker))
    evidence = public_evidence(sp500, sp600_changes)
    staged, _ = resolve_unrecorded_exits(raw, current, asof=args.asof, evidence=evidence)
    # The pull list is every name active in the window after public evidence,
    # plus every still-unresolved phantom (its vendor series is the remaining
    # evidence). Unresolved phantoms were dropped by the staged pass, so they
    # are added back from the raw reconstruction for the pull only.
    phantoms = sorted(
        {normalize_ticker(r.ticker) for r in raw.itertuples(index=False)
         if pd.Timestamp(r.end) >= pd.Timestamp(args.asof) and normalize_ticker(r.ticker) not in set(current)}
    )
    window = members_active_between(staged, args.start_date, args.end_date)
    pull = sorted(set(window) | set(phantoms))
    result: dict[str, object] = {
        "asof": args.asof,
        "n_current_members": len(current),
        "n_changes_sp400": int(len(changes400)),
        "n_changes_sp600": int(len(sp600_changes)),
        "n_raw_intervals": int(len(raw)),
        "n_phantom_tickers": len(phantoms),
        "n_pull": len(pull),
    }

    vendor: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    available: list[str] = []
    if not args.skip_prices:
        fetchable = [s for s in pull if s not in QUARANTINE_TABLE]
        loader = DataLoader(api_key=os.environ.get(API_KEY_ENV_VAR), cache_dir=workspace.data)
        vendor_symbols = [_vendor(s) for s in fetchable]
        fetched = set(
            _pull_prices(vendor_symbols, args.start_date, args.end_date, loader=loader, rate_per_min=args.rate_per_min)
        )
        available = [s for s in fetchable if _vendor(s) in fetched]
        vendor = vendor_evidence(phantoms, workspace.data, start=args.start_date, end=args.end_date)

    combined: dict[str, list[tuple[pd.Timestamp, str]]] = {k: list(v) for k, v in evidence.items()}
    for ticker, items in vendor.items():
        combined.setdefault(ticker, []).extend(items)
    resolved, ledger = resolve_unrecorded_exits(raw, current, asof=args.asof, evidence=combined)
    membership_path = out_dir / f"sp400_membership_{args.asof}.parquet"
    resolved.to_parquet(membership_path)
    counts: dict[str, int] = {}
    for entry in ledger:
        counts[str(entry["evidence"])] = counts.get(str(entry["evidence"]), 0) + 1
    exits_path = out_dir / f"sp400_exits_{args.asof}.json"
    exits_path.write_text(json.dumps({"asof": args.asof, "counts": counts, "rules": __doc__.split("\n\n")[1:4],
                                      "vendor_exit_slack_days": VENDOR_EXIT_SLACK_DAYS, "ledger": ledger}, indent=2))
    result.update({"membership_file": str(membership_path), "exits_file": str(exits_path), "exit_counts": counts})

    final_window = members_active_between(resolved, args.start_date, args.end_date)
    if not args.skip_prices:
        coverage = compute_coverage(final_window, available, asof=args.asof)
        coverage_path = out_dir / f"sp400_coverage_{args.asof}.json"
        coverage_path.write_text(json.dumps(coverage.to_dict(), indent=2))
        resolved_path = out_dir / f"sp400_pit_resolved_{args.asof}.txt"
        write_universe_file(coverage.resolved, resolved_path, asof=args.asof, source="wikipedia+exits+twelvedata_resolved")
        result.update({
            "n_window_members": len(final_window),
            "n_resolved": coverage.n_resolved,
            "coverage_fraction": coverage.coverage_fraction,
            "coverage_file": str(coverage_path),
            "resolved_file": str(resolved_path),
        })
    else:
        result["n_window_members"] = len(final_window)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
