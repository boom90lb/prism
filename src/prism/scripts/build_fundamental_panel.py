"""Freeze the point-in-time characteristic panel for a frozen price dataset.

Joins the EDGAR fundamentals store (``prism.scripts.build_fundamentals``) onto a frozen
dataset's securities and sessions, with the definitions in ``prism.io.fundamental_panel``,
and writes the panel a fundamentals signal node reads::

    python -m prism.scripts.build_fundamental_panel --dataset data/frozen/demotion_b1 \\
        --out data/frozen/fundamentals_v1 --fetch-splits

``--fetch-splits`` reads split history for the dataset's tickers from Alpaca's
corporate-actions endpoint with the paper account's keys (loaded from the checkout's
``.env``); without it, ``{out}/splits.parquet`` must already exist. Companies are
streamed one at a time. Outputs are written to ``.partial`` names and renamed into
place at the end, and ``{out}/freeze.json`` records the sha256 of every input and output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq

from prism.io import fundamental_panel as fp
from prism.io.universe_sp500 import normalize_ticker

logger = logging.getLogger(__name__)

SPLITS_START = "2015-01-01"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(dataset: Path) -> tuple[dict[str, Any], dict[str, tuple[pd.Timestamp | None, pd.Timestamp]],
                                          dict[str, str]]:
    manifest = json.loads((dataset / "manifest.json").read_text())
    membership = pd.read_parquet(dataset / "membership.parquet")
    members = {}
    for row in membership.itertuples():
        start = None if row.valid_from is None or pd.isna(row.valid_from) else pd.Timestamp(row.valid_from)
        members[str(row.sid)] = (start, pd.Timestamp(row.valid_to))
    listings = pd.read_parquet(dataset / "listings.parquet")
    tickers = {str(r.sid): normalize_ticker(str(r.ticker)) for r in listings.itertuples()}
    return manifest, members, tickers


def fetch_and_freeze_splits(tickers: Sequence[str], end: str, path: Path) -> None:
    from prism.live.alpaca_transport import resolve_env_credentials
    from prism.live.split_history import fetch_splits
    from prism.workspace import load_env

    load_env()
    key_id, secret_key = resolve_env_credentials("fetching split history")
    splits = fetch_splits(sorted(set(tickers)), SPLITS_START, end, key_id=key_id, secret_key=secret_key)
    partial = path.with_name(path.name + ".partial")
    splits.to_parquet(partial)
    os.replace(partial, path)


def build(dataset: Path, edgar_dir: Path, out: Path, since: pd.Timestamp, fetch: bool) -> dict[str, Any]:
    manifest, members, tickers = load_dataset(dataset)
    basis_date = pd.Timestamp(manifest["as_of"]).tz_localize(None).normalize()
    out.mkdir(parents=True, exist_ok=True)
    splits_path = out / "splits.parquet"
    if fetch:
        fetch_and_freeze_splits(list(tickers.values()), str(basis_date.date()), splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"{splits_path} is missing; run with --fetch-splits once (N7)")
    splits = pd.read_parquet(splits_path)

    ticker_cik = pd.read_parquet(edgar_dir / "ticker_cik.parquet")
    wanted = ticker_cik[ticker_cik["ticker"].isin(set(tickers.values()))]
    ciks = sorted({int(c) for c in wanted["cik"]})
    events: dict[int, pd.DataFrame] = {}
    for cik, facts in fp.iter_companies(pq.ParquetFile(edgar_dir / "facts.parquet"), ciks):
        events[cik] = fp.company_events(facts, since.tz_localize(fp.edgar.NEW_YORK))
    logger.info("snapshots for %d of %d companies", len(events), len(ciks))

    pieces = []
    for sid, ticker in sorted(tickers.items()):
        bar_path = dataset / "bars" / f"{sid}_1d.parquet"
        if not bar_path.exists():
            continue
        intervals = wanted[wanted["ticker"] == ticker][["start", "end", "cik"]]
        if intervals.empty:
            continue
        bars = pd.read_parquet(bar_path, columns=["ts", "close", "volume"])
        piece = fp.security_panel(sid, bars, intervals, events, splits, basis_date)
        if len(piece):
            pieces.append(piece)
    panel = fp.finalize(fp.dedupe_companies(pd.concat(pieces, ignore_index=True), members))

    panel_path = out / "panel.parquet"
    partial = panel_path.with_name(panel_path.name + ".partial")
    panel.to_parquet(partial)
    os.replace(partial, panel_path)
    coverage = {c: float(panel.groupby("date")[c].apply(lambda s: s.notna().mean()).median())
                for c in fp.CHARACTERISTICS}
    freeze = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(dataset), "id": manifest["id"], "as_of": manifest["as_of"]},
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in (
            ("facts", edgar_dir / "facts.parquet"), ("ticker_cik", edgar_dir / "ticker_cik.parquet"),
            ("edgar_manifest", edgar_dir / "manifest.json"), ("splits", splits_path))},
        "panel": {"path": str(panel_path), "sha256": sha256_file(panel_path), "rows": int(len(panel)),
                  "securities": int(panel["sid"].nunique()), "companies": int(panel["cik"].nunique()),
                  "first": str(panel["date"].min().date()), "last": str(panel["date"].max().date())},
        "splits": {"rows": int(len(splits)), "symbols": int(splits["symbol"].nunique()) if len(splits) else 0},
        "median_daily_coverage": coverage,
        "since": str(since.date()),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
    }
    freeze_path = out / "freeze.json"
    partial = freeze_path.with_name(freeze_path.name + ".partial")
    partial.write_text(json.dumps(freeze, indent=2, sort_keys=True))
    os.replace(partial, freeze_path)
    return freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the point-in-time characteristic panel.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--edgar-dir", type=Path, default=Path("data/edgar"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--since", default="2019-01-01", help="first snapshot moment kept (plus the one before)")
    parser.add_argument("--fetch-splits", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    freeze = build(args.dataset, args.edgar_dir, args.out, pd.Timestamp(args.since), args.fetch_splits)
    print(json.dumps({k: freeze[k] for k in ("panel", "splits", "median_daily_coverage", "peak_rss_mb")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
