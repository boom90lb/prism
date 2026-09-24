"""Build the point-in-time fundamentals store from SEC EDGAR for a membership universe.

Run from the repo root, with ``PRISM_SEC_USER_AGENT`` set to a name and contact
email (SEC's fair-access policy; ``prism.io.edgar``)::

    python -m prism.scripts.build_fundamentals \\
        --membership data/universe/sp900_membership_2026-09-23.parquet --asof 2026-09-24

Steps, each recorded:

1. **Ticker → CIK.** Every ever-member ticker is looked up in SEC's current
   ticker map, directly or through its reviewed rename successor
   (``prism.io.universe_sp500.RENAME_TABLE``). A current ticker can name a
   different company than the one that held it during a membership interval
   (ticker reuse), so a mapping is **accepted for an interval only if that CIK
   filed a 10-K or 10-Q inside it, or in the quarter before it** (intervals ending before the XBRL era carry
   no XBRL facts and are marked ``pre_xbrl``). Names SEC's current map does not
   carry (delisted, acquired, renamed without a reviewed successor) are ledgered
   as unresolved, never guessed; resolving them by name is a separate, reviewed
   step.
2. **Fetch.** Filing histories (all pages) and company facts for every accepted
   CIK, cached with hashes under ``--cache-dir``.
3. **Normalize.** Facts for the concepts behind ``prism.io.edgar.FIELDS``, every
   filing's report kept, each stamped with ``available_at`` from its filing's
   acceptance time; filings (forms, acceptance times, 8-K items).

Outputs: ``{out}/facts.parquet``, ``{out}/filings.parquet``,
``{out}/ticker_cik.parquet`` (accepted ticker-interval → CIK rows),
``{out}/manifest.json`` (every payload's hash), and the tracked coverage ledger
``{ledger_dir}/edgar_coverage_{asof}.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from prism.io import edgar
from prism.io.universe_sp500 import RENAME_TABLE, normalize_ticker

logger = logging.getLogger(__name__)

XBRL_ERA_START = pd.Timestamp("2009-06-15")
LOOKBACK = pd.Timedelta(days=100)
PERIODIC_FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT")


def sec_key(ticker: str) -> str:
    """SEC spells share classes with a hyphen (BRK-B); universe files may use a dot."""
    return normalize_ticker(ticker).replace(".", "-").replace("/", "-")


def sec_ticker_map(payload: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    """``company_tickers.json`` as {ticker: (cik, title)}."""
    out: dict[str, tuple[int, str]] = {}
    for entry in payload.values():
        try:
            out[sec_key(entry["ticker"])] = (int(entry["cik_str"]), str(entry["title"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise edgar.EdgarError(f"malformed company_tickers entry: {entry!r}") from exc
    return out


def candidate_ciks(tickers: Sequence[str], sec_map: Mapping[str, tuple[int, str]]) -> dict[str, dict[str, Any]]:
    """Per ticker: the candidate CIK and how it was found, or why none was."""
    out: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        key = sec_key(ticker)
        successor = RENAME_TABLE.get(normalize_ticker(ticker))
        if key in sec_map:
            cik, title = sec_map[key]
            out[ticker] = {"cik": cik, "title": title, "source": "sec_current_ticker"}
        elif successor is not None and sec_key(successor) in sec_map:
            cik, title = sec_map[sec_key(successor)]
            out[ticker] = {"cik": cik, "title": title, "source": f"rename_successor:{successor}"}
        else:
            out[ticker] = {"cik": None, "title": None, "source": None, "reason": "not_in_sec_current_ticker_map"}
    return out


def interval_status(filings: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """``accepted`` if the CIK filed a periodic report in the interval's XBRL-era part, or in the quarter before it.

    The look-back is one quarter and only backward: a member that joined days ago has
    filed nothing since joining, while a reused ticker's new holder has not filed before
    the old holder's interval ended.
    """
    if end < XBRL_ERA_START:
        return "pre_xbrl"
    lo = max(start - LOOKBACK, XBRL_ERA_START)
    periodic = filings[filings["form"].isin(PERIODIC_FORMS)]
    inside = periodic[(periodic["filing_date"] >= lo) & (periodic["filing_date"] <= end)]
    return "accepted" if len(inside) else "no_periodic_filing_in_interval"


def all_filings(client: edgar.SecClient, cik: int, refresh: bool) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    payload, meta = client.submissions(cik, refresh)
    metas = [meta]
    pages = []
    for page in payload.get("filings", {}).get("files", []):
        page_payload, page_meta = client.submissions_page(page["name"], refresh)
        pages.append(page_payload)
        metas.append(page_meta)
    return edgar.submissions_filings(payload, pages), metas


def build(membership: pd.DataFrame, client: edgar.SecClient, out_dir: Path, ledger_path: Path, asof: str,
          refresh: bool = False, limit: int | None = None) -> dict[str, Any]:
    tickers_payload, tickers_meta = client.company_tickers(refresh)
    sec_map = sec_ticker_map(tickers_payload)
    tickers = sorted(membership["ticker"].map(normalize_ticker).unique())
    if limit is not None:
        tickers = tickers[:limit]
    candidates = candidate_ciks(tickers, sec_map)

    manifest: dict[str, Any] = {"company_tickers": tickers_meta, "submissions": {}, "companyfacts": {}}
    filings_by_cik: dict[int, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        info = candidates[ticker]
        spans = membership[membership["ticker"].map(normalize_ticker) == ticker]
        cik = info["cik"]
        if cik is not None and cik not in filings_by_cik:
            try:
                filings_by_cik[cik], metas = all_filings(client, cik, refresh)
                manifest["submissions"][str(cik)] = metas
            except edgar.EdgarNotFound:
                filings_by_cik[cik] = pd.DataFrame(columns=list(edgar.FILING_COLUMNS))
        for span in spans.itertuples():
            if span.end < XBRL_ERA_START:
                status = "pre_xbrl"
            elif cik is None:
                status = "unresolved"
            else:
                status = interval_status(filings_by_cik[cik], span.start, span.end)
            rows.append({"ticker": ticker, "start": span.start, "end": span.end, "cik": cik, "title": info["title"],
                         "source": info["source"], "status": status, "reason": info.get("reason")})
    intervals = pd.DataFrame(rows)
    intervals["cik"] = intervals["cik"].astype("Int64")

    accepted_ciks = sorted({int(c) for c in intervals.loc[intervals["status"] == "accepted", "cik"]})
    fact_frames, missing_facts = [], []
    concepts = edgar.concept_names()
    for cik in accepted_ciks:
        try:
            payload, meta = client.companyfacts(cik, refresh)
        except edgar.EdgarNotFound:
            missing_facts.append(cik)
            continue
        manifest["companyfacts"][str(cik)] = meta
        facts = edgar.companyfacts_frame(payload, concepts)
        fact_frames.append(edgar.attach_availability(facts, filings_by_cik[cik]))

    out_dir.mkdir(parents=True, exist_ok=True)
    facts_all = pd.concat(fact_frames, ignore_index=True) if fact_frames else pd.DataFrame()
    filings_all = pd.concat([filings_by_cik[c] for c in accepted_ciks], ignore_index=True) if accepted_ciks else \
        pd.DataFrame(columns=list(edgar.FILING_COLUMNS))
    facts_all.to_parquet(out_dir / "facts.parquet")
    filings_all.to_parquet(out_dir / "filings.parquet")
    intervals[intervals["status"] == "accepted"].reset_index(drop=True).to_parquet(out_dir / "ticker_cik.parquet")
    manifest["built_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    xbrl_era = intervals[intervals["status"] != "pre_xbrl"]
    xbrl_tickers = sorted(set(xbrl_era["ticker"]))
    accepted_tickers = sorted(set(xbrl_era.loc[xbrl_era["status"] == "accepted", "ticker"]))
    ledger = {
        "asof": asof,
        "membership_rows": int(len(membership)),
        "tickers": len(tickers),
        "xbrl_era_tickers": len(xbrl_tickers),
        "accepted_tickers": len(accepted_tickers),
        "coverage_xbrl_era": (len(accepted_tickers) / len(xbrl_tickers)) if xbrl_tickers else None,
        "accepted_ciks": len(accepted_ciks),
        "ciks_without_companyfacts": missing_facts,
        "facts_rows": int(len(facts_all)),
        "availability_basis": facts_all["availability_basis"].value_counts().to_dict() if len(facts_all) else {},
        "status_counts": intervals["status"].value_counts().to_dict(),
        "not_accepted": [
            {"ticker": r.ticker, "start": str(r.start.date()), "end": str(r.end.date()),
             "cik": None if pd.isna(r.cik) else int(r.cik), "status": r.status, "reason": r.reason}
            for r in xbrl_era[xbrl_era["status"] != "accepted"].itertuples()
        ],
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))
    return ledger


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the point-in-time EDGAR fundamentals store.")
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--asof", required=True, help="label for the coverage ledger, YYYY-MM-DD")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/edgar/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/edgar"))
    parser.add_argument("--ledger-dir", type=Path, default=Path("data/universe"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="first N tickers only (smoke runs)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = edgar.SecClient(edgar.user_agent_from_env(), args.cache_dir)
    membership = pd.read_parquet(args.membership)
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    ledger = build(membership, client, args.out_dir, args.ledger_dir / f"edgar_coverage_{args.asof}{suffix}.json",
                   args.asof, refresh=args.refresh, limit=args.limit)
    summary = {k: ledger[k] for k in ("tickers", "xbrl_era_tickers", "accepted_tickers", "coverage_xbrl_era",
                                      "accepted_ciks", "facts_rows", "status_counts")}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
