"""Dividend wedge between a run's price-return ledger and a total-return book (SPEC §5 / I-7).

The B1 momentum evidence ledger is price-return-only — tagged
``split_adjusted_open_close_price_return_no_dividends`` per I-7 — while the
live paper equity accrues dividend cash through the broker. The two equity
streams therefore differ by the book's net dividend flow, and any M6-time
comparison of paper equity against the certified numbers consumes that wedge
whether or not it has been measured. Twelve Data ``/dividends`` is 403 on the
basic tier (``docs/operations.md``), so this script measures the wedge from
the Alpaca corporate-actions endpoint instead — the same admitted $0 vendor
and key the live loop already reads bars from, a data surface the repo has
not touched before this diagnostic.

Accrual convention matches the ledger machinery (``execution/target_weights``
and the breadth diagnostic's contribution panel): the book decided at close
``t-1`` earns ex-date ``t``'s per-share amount, expressed as a return on the
prior close — ``wedge[t] = Σ_i w_i[t-1] · amount_i[t] / close_i[t-1]``. A
positive total means a total-return ledger would have shown MORE than the
certified price-return one (the certified number is conservative); a negative
total means the price-return ledger flatters the strategy by the same amount.

Uncounted diagnostic: searches nothing, changes no trial code path, writes
one JSON (plus the raw dividend records beside it, as evidence and refetch
cache). Frame and results: ``docs/program_review_2026-07.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from research.scripts.data_integrity_sweep import bar_dates

CORPORATE_ACTIONS_URL = "https://data.alpaca.markets/v1/corporate-actions"
SYMBOL_BATCH = 50
PAGE_LIMIT = 1000


def fetch_cash_dividends(
    symbols: list[str],
    start: str,
    end: str,
    *,
    key_id: str,
    secret_key: str,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> list[dict]:
    """All cash-dividend records for ``symbols`` in ``[start, end]`` (ex-date basis), paginated."""
    sess = session if session is not None else requests.Session()
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}
    records: list[dict] = []
    for i in range(0, len(symbols), SYMBOL_BATCH):
        batch = symbols[i : i + SYMBOL_BATCH]
        page_token: str | None = None
        while True:
            params: dict = {
                "symbols": ",".join(batch),
                "types": "cash_dividend",
                "start": start,
                "end": end,
                "limit": PAGE_LIMIT,
            }
            if page_token:
                params["page_token"] = page_token
            resp = sess.get(CORPORATE_ACTIONS_URL, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            records.extend(payload.get("corporate_actions", {}).get("cash_dividends", []) or [])
            page_token = payload.get("next_page_token")
            if not page_token:
                break
    return records


def dividend_panel(records: list[dict], index: pd.DatetimeIndex, columns: pd.Index) -> tuple[pd.DataFrame, dict]:
    """Records → ex-date × symbol per-share amounts on the run calendar, plus accounting meta.

    Ex-dates that are not run trading days (holiday mismatches, out-of-window
    records) are dropped and counted rather than silently shifted — the run
    calendar is the ledger's calendar, and a wedge estimate should say what it
    excluded.
    """
    rows = [
        {"ex_date": pd.Timestamp(r["ex_date"]), "symbol": r["symbol"], "amount": float(r["rate"])}
        for r in records
        if r.get("symbol") in set(columns)
    ]
    meta = {
        "n_records": len(records),
        "n_records_in_universe": len(rows),
        "n_special": sum(1 for r in records if r.get("special")),
        "n_foreign": sum(1 for r in records if r.get("foreign")),
    }
    panel = pd.DataFrame(0.0, index=index, columns=columns)
    dropped = 0
    for row in rows:
        if row["ex_date"] in panel.index:
            panel.loc[row["ex_date"], row["symbol"]] += row["amount"]
        else:
            dropped += 1
    meta["n_ex_dates_off_calendar"] = dropped
    return panel, meta


def wedge_report(weights: pd.DataFrame, closes: pd.DataFrame, dividends: pd.DataFrame) -> dict:
    """Book dividend flow under the ``w[t-1] · amount[t] / close[t-1]`` accrual, split by leg."""
    prev_w = weights.shift(1)
    div_return = dividends / closes.shift(1)
    flow = (prev_w * div_return).fillna(0.0)
    per_day = flow.sum(axis=1)
    years = (weights.index[-1] - weights.index[0]).days / 365.25
    long_flow = flow.where(prev_w > 0.0, 0.0).sum(axis=1)
    short_flow = flow.where(prev_w < 0.0, 0.0).sum(axis=1)
    long_gross = prev_w.clip(lower=0.0).sum(axis=1)
    short_gross = (-prev_w.clip(upper=0.0)).sum(axis=1)
    active = long_gross + short_gross > 0.0
    return {
        "window_years": years,
        "total_wedge_return": float(per_day.sum()),
        "annualized_wedge_bps": float(per_day.sum() / years * 1e4),
        "long_leg": {
            "flow_annualized_bps": float(long_flow.sum() / years * 1e4),
            "avg_gross": float(long_gross[active].mean()),
            "portfolio_yield_pct": float(long_flow.sum() / years / long_gross[active].mean() * 100.0),
        },
        "short_leg": {
            "flow_annualized_bps": float(short_flow.sum() / years * 1e4),
            "avg_gross": float(short_gross[active].mean()),
            "portfolio_yield_pct": float(-short_flow.sum() / years / short_gross[active].mean() * 100.0),
        },
        "n_ex_date_days_hit": int((per_day != 0.0).sum()),
    }


def _load_closes(symbols: list[str], data_dir: Path, pattern_suffix: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-name closes from the same bar caches the run's panel was built from, on the run calendar."""
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        path = data_dir / f"{sym}{pattern_suffix}"
        if not path.exists():
            continue
        bars = pd.read_parquet(path)
        close = pd.to_numeric(bars["close"], errors="coerce").groupby(bar_dates(bars.index)).last()
        out[sym] = close.reindex(index)
    return pd.DataFrame(out, index=index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", default="results/demotion_b1", help="Finished run directory")
    parser.add_argument("--data_dir", default="data", help="Bar-cache directory")
    parser.add_argument("--cache_suffix", default="_1d_2020-01-01_2026-06-16.parquet", help="Cache filename suffix")
    parser.add_argument(
        "--records_json", default=None, help="Reuse previously fetched dividend records instead of hitting the API"
    )
    parser.add_argument("--out", default=None, help="Output JSON (default: results/dividend_wedge_<today>.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    weights = pd.read_csv(run_dir / "target_weights.csv", index_col=0)
    widx = pd.DatetimeIndex(pd.to_datetime(weights.index, utc=True)).tz_convert("America/New_York")
    weights.index = widx.tz_localize(None).normalize()
    ever_held = sorted(weights.columns[(weights != 0.0).any(axis=0)])
    weights = weights[ever_held]
    start, end = str(weights.index.min().date()), str(weights.index.max().date())

    if args.records_json:
        records = json.loads(Path(args.records_json).read_text())
    else:
        key_id, secret_key = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not key_id or not secret_key:
            raise SystemExit("APCA_API_KEY_ID / APCA_API_SECRET_KEY not in environment (source .env first)")
        records = fetch_cash_dividends(ever_held, start, end, key_id=key_id, secret_key=secret_key)
        raw_path = Path("results") / f"alpaca_cash_dividends_{start}_{end}.json"
        raw_path.write_text(json.dumps(records, indent=1))

    closes = _load_closes(ever_held, Path(args.data_dir), args.cache_suffix, weights.index)
    dividends, div_meta = dividend_panel(records, weights.index, weights.columns)
    report = wedge_report(weights, closes, dividends)

    payload = {
        "meta": {
            "generated": date.today().isoformat(),
            "run_dir": str(run_dir),
            "window": [start, end],
            "n_ever_held": len(ever_held),
            "source": CORPORATE_ACTIONS_URL,
            "records_reused": bool(args.records_json),
            **div_meta,
        },
        "wedge": report,
    }
    out_path = Path(args.out) if args.out else Path("results") / f"dividend_wedge_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
