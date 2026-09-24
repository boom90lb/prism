"""Build the `breadth_v1` frozen dataset: one set of bars, three index ids (docs/breadth_design.md §2).

Usage::

    python -m research.scripts.build_breadth_dataset --asof 2026-09-23

Writes, all from tracked universe artifacts and the local bar caches:

* ``data/universe/sp400_exclusive_membership_<asof>.parquet`` -- the resolved
  S&P 400 membership in vendor symbology, minus every day the ticker holds in
  the S&P 500 (the indices are mutually exclusive);
* ``data/universe/sp900_membership_<asof>.parquet`` -- the per-ticker union of
  the S&P 500 and exclusive S&P 400 memberships;
* ``data/frozen/breadth_v1/`` -- the dataset, primary index ``sp900`` with
  ``sp500`` and ``sp400`` as extra index ids, over the union of both indices'
  price-resolved names, 2020-01-01 to 2026-06-16 (the certified window).

It refuses to write into a dataset directory that already holds files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prism.io.legacy_import import build_legacy_dataset
from prism.io.universe_sp500 import (
    RENAME_TABLE,
    normalize_ticker,
    subtract_membership,
    union_membership,
)
from prism.workspace import resolve_workspace


def _vendor(ticker: str) -> str:
    t = normalize_ticker(ticker)
    return normalize_ticker(RENAME_TABLE.get(t, t))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asof", required=True, help="The S&P 400 artifacts' as-of date")
    p.add_argument("--sp500_asof", default="2026-06-16")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2026-06-16")
    p.add_argument("--out", default="data/frozen/breadth_v1")
    p.add_argument("--workspace", default=None)
    args = p.parse_args(argv)
    workspace = resolve_workspace(args.workspace)
    universe = workspace.data / "universe"

    sp500_path = universe / f"sp500_membership_{args.sp500_asof}.parquet"
    sp500 = pd.read_parquet(sp500_path)
    sp500_resolved = json.loads((universe / f"sp500_coverage_{args.sp500_asof}.json").read_text())["resolved"]
    sp400 = pd.read_parquet(universe / f"sp400_membership_{args.asof}.parquet")
    sp400 = sp400.assign(ticker=sp400["ticker"].map(_vendor))
    sp400_resolved = json.loads((universe / f"sp400_coverage_{args.asof}.json").read_text())["resolved"]

    exclusive = subtract_membership(sp400, sp500)
    union = union_membership(sp500, exclusive)
    exclusive_path = universe / f"sp400_exclusive_membership_{args.asof}.parquet"
    union_path = universe / f"sp900_membership_{args.asof}.parquet"
    exclusive.to_parquet(exclusive_path)
    union.to_parquet(union_path)

    symbols = sorted({normalize_ticker(s) for s in sp500_resolved} | {_vendor(s) for s in sp400_resolved})
    report = build_legacy_dataset(
        Path(args.out).resolve(),
        symbols=symbols,
        cache_dir=workspace.data,
        start=args.start,
        end=args.end,
        vintage=f"breadth_v1_{args.end}",
        index_id="sp900",
        membership_file=union_path,
        universe_note=(
            "sp900 = S&P 500 membership (certified) union the S&P 400 membership reconstructed from Wikipedia with "
            "unrecorded exits resolved on dated evidence and every S&P 500 day removed; docs/breadth_design.md"
        ),
        coverage={
            "sp500_resolved": len(sp500_resolved),
            "sp400_resolved": len(sp400_resolved),
            "sp400_exits_file": f"data/universe/sp400_exits_{args.asof}.json",
        },
        extra_indexes={"sp500": sp500_path, "sp400": exclusive_path},
    )
    print(json.dumps({
        "dataset": str(report.root),
        "data_id": report.manifest.id,
        "n_sids": len(report.sids),
        "n_sessions": len(report.sessions),
        "first_session": report.sessions[0],
        "last_session": report.sessions[-1],
        "without_bars": list(report.without_bars),
        "index_membership": report.manifest.coverage["index_membership"],
        "exclusive_file": str(exclusive_path),
        "union_file": str(union_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
