"""Freeze the point-in-time earnings-announcement panel for a frozen price dataset.

Joins the item 2.02 8-Ks of the EDGAR store (``prism.scripts.build_fundamentals``) onto
a frozen dataset's securities and sessions, with the definitions in
``prism.io.earnings_events``, and writes the event list and the daily panel an
announcement signal node reads::

    python -m prism.scripts.build_earnings_panel --dataset data/frozen/demotion_b1 \\
        --out data/frozen/earnings_v1

Outputs are written to ``.partial`` names and renamed into place at the end, and
``{out}/freeze.json`` records the sha256 of every input and output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from prism.io import earnings_events as ee
from prism.io import edgar
from prism.scripts.build_fundamental_panel import load_dataset, sha256_file

logger = logging.getLogger(__name__)


def load_releases(filings_path: Path) -> pd.DataFrame:
    """Original and amended 8-Ks naming item 2.02, read without loading the whole filings table."""
    table = pq.read_table(filings_path, columns=["cik", "accn", "form", "filing_date", "acceptance", "items"],
                          filters=[("form", "in", ["8-K", "8-K/A"])])
    table = table.filter(pc.match_substring(pc.fill_null(table["items"], ""), "2.02"))  # type: ignore[attr-defined]
    return edgar.earnings_releases(table.to_pandas())


def _write(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    frame.to_parquet(partial)
    os.replace(partial, path)


def build(dataset: Path, edgar_dir: Path, out: Path) -> dict[str, Any]:
    manifest, members, tickers = load_dataset(dataset)
    out.mkdir(parents=True, exist_ok=True)
    bars: dict[str, pd.DataFrame] = {}
    for sid in sorted(tickers):
        path = dataset / "bars" / f"{sid}_1d.parquet"
        if path.exists():
            bars[sid] = pd.read_parquet(path, columns=["ts", "close", "volume"])
    if not bars:
        raise edgar.EdgarError(f"{dataset} has no bars (N7)")
    close = pd.DataFrame({sid: b.set_index(ee.naive_days(b["ts"]))["close"].astype(float) for sid, b in bars.items()})
    close = close.sort_index()
    sessions = pd.DatetimeIndex(close.index)
    returns = close.pct_change(fill_method=None)
    member_mask = pd.DataFrame(False, index=sessions, columns=close.columns)
    for sid in close.columns:
        span = members.get(sid)
        if span is not None:
            start = span[0] if span[0] is not None else sessions[0]
            member_mask.loc[(sessions >= start) & (sessions <= span[1]), sid] = True
    market = ee.market_returns(returns, member_mask).to_numpy(float)

    ticker_cik = pd.read_parquet(edgar_dir / "ticker_cik.parquet")
    wanted = ticker_cik[ticker_cik["ticker"].isin(set(tickers.values()))]
    releases = load_releases(edgar_dir / "filings.parquet")
    releases = releases[releases["cik"].isin(set(wanted["cik"].astype(int).tolist()))]
    dates = ee.announcement_dates(releases, sessions)
    dates_by_cik = {int(part["cik"].iloc[0]): part for _, part in dates.groupby("cik")}
    logger.info("%d announcements for %d companies", len(dates), len(dates_by_cik))

    event_pieces, panel_pieces = [], []
    for sid, ticker in sorted(tickers.items()):
        if sid not in bars:
            continue
        intervals = wanted[wanted["ticker"] == ticker][["start", "end", "cik"]]
        if intervals.empty:
            continue
        events = ee.security_events(sid, intervals, dates_by_cik, sessions, returns[sid].to_numpy(float), market)
        if events.empty:
            continue
        event_pieces.append(events)
        piece = ee.security_panel(sid, bars[sid], events, sessions)
        if len(piece):
            panel_pieces.append(piece)
    events_all = pd.concat(event_pieces, ignore_index=True)[list(ee.EVENT_COLUMNS)]
    panel = ee.finalize(pd.concat(panel_pieces, ignore_index=True), members)

    events_path, panel_path = out / "events.parquet", out / "panel.parquet"
    _write(events_all, events_path)
    _write(panel, panel_path)
    member_rows = panel[panel["member"]]
    per_day_members = member_mask.sum(axis=1)
    coverage = {}
    for max_age in (60, 126):
        live = member_rows[member_rows["age_sessions"] <= max_age].groupby("date").size()
        coverage[f"age_le_{max_age}"] = float((live / per_day_members.reindex(live.index)).median())
    years = (sessions[-1] - sessions[0]).days / 365.25
    freeze = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(dataset), "id": manifest["id"], "as_of": manifest["as_of"]},
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in (
            ("filings", edgar_dir / "filings.parquet"), ("ticker_cik", edgar_dir / "ticker_cik.parquet"),
            ("edgar_manifest", edgar_dir / "manifest.json"))},
        "definitions": {"window": list(ee.WINDOW), "min_gap_sessions": ee.MIN_GAP_SESSIONS,
                        "min_market_names": ee.MIN_MARKET_NAMES},
        "events": {"path": str(events_path), "sha256": sha256_file(events_path), "rows": int(len(events_all)),
                   "securities": int(events_all["sid"].nunique()),
                   "without_return": int(events_all["ear"].isna().sum()),
                   "per_security_year": round(len(events_all) / max(events_all["sid"].nunique(), 1) / years, 2)},
        "panel": {"path": str(panel_path), "sha256": sha256_file(panel_path), "rows": int(len(panel)),
                  "securities": int(panel["sid"].nunique()), "companies": int(panel["cik"].nunique()),
                  "first": str(panel["date"].min().date()), "last": str(panel["date"].max().date())},
        "median_daily_member_coverage": coverage,
        "market_sessions_without_value": int(np.isnan(market[1:]).sum()),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
    }
    freeze_path = out / "freeze.json"
    partial = freeze_path.with_name(freeze_path.name + ".partial")
    partial.write_text(json.dumps(freeze, indent=2, sort_keys=True))
    os.replace(partial, freeze_path)
    return freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the point-in-time earnings-announcement panel.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--edgar-dir", type=Path, default=Path("data/edgar"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    freeze = build(args.dataset, args.edgar_dir, args.out)
    print(json.dumps({k: freeze[k] for k in ("events", "panel", "median_daily_member_coverage",
                                            "market_sessions_without_value", "peak_rss_mb")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
