"""CLI: EDGE bracketing diagnostic over cached OHLC panels (SPEC §6 I-9, R2 cost harness).

Reads wide (dates × symbols) OHLC + volume panels from parquet and prints, per
liquidity bucket, whether the pre-registered conservative-upper spread schedule
(``SPREAD_BUCKET_SCHEDULE_V1``) sits above / inside / below the EDGE
effective-spread distribution. A bracketing diagnostic only — fills remain the
calibration authority (docs/edge_preregistration.md); this reads nothing but bars
already on disk and prints. No counted trial, no pre-registration to spend.

    python -m prism.scripts.edge_diagnostic \\
        --open open.parquet --high high.parquet --low low.parquet \\
        --close close.parquet --volume volume.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from prism.execution.edge import (
    DEFAULT_MIN_OBS,
    SPREAD_BUCKET_SCHEDULE_V1,
    edge_bracketing_diagnostic,
    edge_spread_by_symbol,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EDGE bracketing diagnostic: bucket schedule vs the OHLC effective-spread distribution."
    )
    parser.add_argument("--open", type=Path, required=True, help="Wide (dates × symbols) open-price parquet.")
    parser.add_argument("--high", type=Path, required=True, help="Wide high-price parquet.")
    parser.add_argument("--low", type=Path, required=True, help="Wide low-price parquet.")
    parser.add_argument("--close", type=Path, required=True, help="Wide close-price parquet (defines the alignment).")
    parser.add_argument(
        "--volume",
        type=Path,
        required=True,
        help="Wide volume parquet; median(close × volume) per name is the liquidity screen.",
    )
    parser.add_argument(
        "--min-obs", type=int, default=DEFAULT_MIN_OBS, help="Min finite bars for a per-name estimate."
    )
    args = parser.parse_args(argv)

    open_panel = pd.read_parquet(args.open)
    high = pd.read_parquet(args.high)
    low = pd.read_parquet(args.low)
    close = pd.read_parquet(args.close)
    volume = pd.read_parquet(args.volume).reindex(index=close.index, columns=close.columns)

    edge_bps = edge_spread_by_symbol(open_panel, high, low, close, min_obs=args.min_obs)
    median_dollar_volume = (close * volume).median()
    table = edge_bracketing_diagnostic(edge_bps, median_dollar_volume, SPREAD_BUCKET_SCHEDULE_V1)

    print(
        json.dumps(
            {
                "n_names_total": int(edge_bps.size),
                "n_names_estimated": int(edge_bps.notna().sum()),
                "buckets": table.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
