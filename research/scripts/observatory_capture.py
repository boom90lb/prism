"""Capture-only observatory fetchers (W5) — news flow + EDGAR cadence.

Writes point-in-time expectation-state payloads into the append-only store
(``prism.io.observatory``). No modeling, no scores, no trial ledger. Capture
is time-irreversible: an uncaptured day is gone.

Lanes:

* ``news`` — Polygon reference news counts for a small equity ticker set
  (count + earliest/latest published_utc in the lookback). Requires
  ``POLYGON_API_KEY``.
* ``edgar`` — SEC EDGAR full-text search hit count for a calendar day over
  a pinned form set (10-K / 10-Q / 8-K). Free; requires a descriptive
  ``User-Agent`` (``SEC_USER_AGENT`` env or the script default).

Fail loud on missing credentials, non-200 HTTP, or unparseable payloads (N7).
Uncounted; factory modeling is deferred until ``docs/factory_amendment.md``
ratifies.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from prism.io.observatory import append_capture

# Pinned form set for cadence (not a search over free text).
EDGAR_FORMS = ("10-K", "10-Q", "8-K")
# Small default equity set for news-flow capture (not a universe claim).
# Keep short: free-tier Polygon is ~5 req/min on many plans.
DEFAULT_NEWS_TICKERS = ("SPY", "QQQ", "AAPL", "MSFT", "JPM")
DEFAULT_SEC_UA = "prism-observatory/0.4 (research capture; local operator)"
POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
# Free-tier pacing between Polygon news calls.
DEFAULT_NEWS_PAUSE_S = 12.5


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_news_flow(
    tickers: tuple[str, ...] | list[str],
    *,
    api_key: str,
    lookback_days: int = 1,
    limit_per_ticker: int = 50,
    timeout: float = 30.0,
    pause_s: float = DEFAULT_NEWS_PAUSE_S,
    get: Callable[..., Any] = requests.get,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Polygon news-flow snapshot (counts only; no sentiment)."""
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is required for the news lane (N7)")
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    start_s = start.strftime("%Y-%m-%dT00:00:00Z")
    end_s = end.strftime("%Y-%m-%dT23:59:59Z")
    headers = {"Authorization": f"Bearer {api_key}"}
    per_ticker: dict[str, Any] = {}
    total = 0
    for i, sym in enumerate(tickers):
        if i > 0 and pause_s > 0:
            sleep(pause_s)
        params = {
            "ticker": sym,
            "published_utc.gte": start_s,
            "published_utc.lte": end_s,
            "limit": int(limit_per_ticker),
            "sort": "published_utc",
            "order": "desc",
        }
        resp = get(POLYGON_NEWS_URL, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 429:
            # One paced retry after a full free-tier minute window.
            sleep(max(pause_s, 60.0))
            resp = get(POLYGON_NEWS_URL, params=params, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Polygon news HTTP {resp.status_code} for {sym}: {resp.text[:200]!r}"
            )
        payload = resp.json()
        status = payload.get("status")
        if status not in (None, "OK", "DELAYED"):
            raise RuntimeError(f"Polygon news status={status!r} for {sym}: {payload!r}")
        results = payload.get("results") or []
        published = [r.get("published_utc") for r in results if r.get("published_utc")]
        n = len(results)
        total += n
        per_ticker[sym] = {
            "n": n,
            "earliest_published_utc": min(published) if published else None,
            "latest_published_utc": max(published) if published else None,
            "result_count_field": payload.get("count"),
        }
    return {
        "lookback_days": lookback_days,
        "window": {"start": start_s, "end": end_s},
        "tickers": list(tickers),
        "n_articles_sum": total,
        "per_ticker": per_ticker,
    }


def fetch_edgar_cadence(
    day: date,
    *,
    forms: tuple[str, ...] | list[str] = EDGAR_FORMS,
    user_agent: str,
    timeout: float = 30.0,
    get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    """SEC EDGAR full-text search hit count for one calendar day + form set."""
    if not user_agent or not user_agent.strip():
        raise RuntimeError("SEC User-Agent is required for the edgar lane (N7)")
    day_s = day.isoformat()
    # Omit q entirely: bare q="" 500s and q="*" forces zero hits under forms filter.
    # forms + custom dateRange is the cadence surface (verified live 2026-07).
    params = {
        "dateRange": "custom",
        "startdt": day_s,
        "enddt": day_s,
        "forms": ",".join(forms),
    }
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    resp = get(EDGAR_SEARCH_URL, params=params, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"EDGAR search HTTP {resp.status_code} for {day_s}: {resp.text[:200]!r}"
        )
    payload = resp.json()
    # Shape: {"hits": {"total": {"value": N, ...}, "hits": [...]}, ...}
    hits = payload.get("hits") or {}
    total = hits.get("total")
    if isinstance(total, dict):
        n = total.get("value")
    else:
        n = total
    if n is None:
        raise RuntimeError(f"EDGAR search response missing hits.total for {day_s}: keys={list(payload)[:12]}")
    return {
        "day": day_s,
        "forms": list(forms),
        "n_hits": int(n),
        "source": "efts.sec.gov/LATEST/search-index",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--lane",
        choices=("news", "edgar", "both"),
        default="both",
        help="Which capture lane(s) to run",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/observatory"),
        help="Directory for lane jsonl files (created if missing)",
    )
    p.add_argument(
        "--news-tickers",
        default=",".join(DEFAULT_NEWS_TICKERS),
        help="Comma-separated tickers for the news lane",
    )
    p.add_argument("--news-lookback-days", type=int, default=1)
    p.add_argument(
        "--edgar-day",
        default=None,
        help="YYYY-MM-DD for EDGAR cadence (default: UTC today)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and print payloads; do not append to the store",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    captured_at = utc_now_iso()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ran: list[str] = []

    if args.lane in ("news", "both"):
        api_key = os.getenv("POLYGON_API_KEY") or ""
        tickers = tuple(t.strip().upper() for t in args.news_tickers.split(",") if t.strip())
        news_payload = fetch_news_flow(
            tickers,
            api_key=api_key,
            lookback_days=args.news_lookback_days,
        )
        record = {
            "captured_at": captured_at,
            "lane": "news",
            "payload": news_payload,
        }
        if args.dry_run:
            print(record)
        else:
            path = out_dir / "news.jsonl"
            append_capture(path, record)
            print(f"appended news → {path} n_articles_sum={news_payload['n_articles_sum']}")
        ran.append("news")

    if args.lane in ("edgar", "both"):
        day = date.fromisoformat(args.edgar_day) if args.edgar_day else datetime.now(timezone.utc).date()
        ua = os.getenv("SEC_USER_AGENT") or DEFAULT_SEC_UA
        edgar_payload = fetch_edgar_cadence(day, user_agent=ua)
        record = {
            "captured_at": captured_at,
            "lane": "edgar",
            "payload": edgar_payload,
        }
        if args.dry_run:
            print(record)
        else:
            path = out_dir / "edgar.jsonl"
            append_capture(path, record)
            print(f"appended edgar → {path} day={edgar_payload['day']} n_hits={edgar_payload['n_hits']}")
        ran.append("edgar")

    if not ran:
        print("no lanes ran", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
