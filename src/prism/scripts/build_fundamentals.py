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

Companies are processed one at a time and appended to disk, so peak memory is one
company's payloads, not the universe's (the ledger records ``peak_rss_mb``). Every
output is written to ``<name>.partial`` and renamed into place only when the whole
build succeeds, so an interrupted build leaves the previous outputs intact.

Outputs: ``{out}/facts.parquet``, ``{out}/filings.parquet``,
``{out}/ticker_cik.parquet`` (accepted ticker-interval → CIK rows),
``{out}/manifest.json`` (every payload's hash), and the tracked coverage ledger
``{ledger_dir}/edgar_coverage_{asof}.json``.

The default directories are inside the repository, where the maintainer's store lives.
``docs/edgar_point_in_time.md`` documents every output's schema and shows a run that passes
``--cache-dir``, ``--out-dir`` and ``--ledger-dir`` and writes nothing into the checkout.
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
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prism.io import edgar
from prism.io.universe_sp500 import RENAME_TABLE, normalize_ticker

logger = logging.getLogger(__name__)

XBRL_ERA_START = pd.Timestamp("2009-06-15")
LOOKBACK = pd.Timedelta(days=100)
PERIODIC_FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT")

_NY = pa.timestamp("ns", tz=edgar.NEW_YORK)
#: Output schemas, fixed so every company's rows append to one file whatever columns it happens to leave empty.
FACTS_SCHEMA = pa.schema([
    ("cik", pa.int64()), ("taxonomy", pa.string()), ("concept", pa.string()), ("unit", pa.string()),
    ("start", pa.timestamp("ns")), ("end", pa.timestamp("ns")), ("val", pa.float64()), ("accn", pa.string()),
    ("fy", pa.int64()), ("fp", pa.string()), ("form", pa.string()), ("filed", pa.timestamp("ns")),
    ("frame", pa.string()), ("available_at", _NY), ("availability_basis", pa.string()),
])
FILINGS_SCHEMA = pa.schema([
    ("cik", pa.int64()), ("accn", pa.string()), ("form", pa.string()), ("filing_date", pa.timestamp("ns")),
    ("acceptance", _NY), ("report_date", pa.timestamp("ns")), ("items", pa.string()),
])


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


class _TableSink:
    """Appends frames to one parquet file under a fixed schema, one company at a time.

    Writes go to ``<name>.partial``; :meth:`commit` renames it over the final path, so a
    build that dies part-way leaves the previous complete file in place.
    """

    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path = path
        self.partial = path.with_name(path.name + ".partial")
        self.schema = schema
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame[self.schema.names], schema=self.schema, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.partial, table.schema)
        self.writer.write_table(table)  # type: ignore[union-attr]
        self.rows += table.num_rows

    def commit(self) -> None:
        if self.writer is None:
            self.write(pd.DataFrame({name: pd.Series(dtype=object) for name in self.schema.names}))
        assert self.writer is not None
        self.writer.close()
        os.replace(self.partial, self.path)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.partial.unlink(missing_ok=True)


def _peak_rss_mb() -> float:
    """Peak resident memory of this process so far (Linux reports kilobytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_name_matches(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Accepted per-interval CIKs from ``prism.scripts.match_delisted_ciks``, keyed by (ticker, start)."""
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    out = {}
    for row in payload["accepted"]:
        key = (normalize_ticker(row["ticker"]), str(pd.Timestamp(row["start"]).date()))
        out[key] = {"cik": int(row["cik"]), "title": row.get("title"), "source": "name_match"}
    return out


def load_review_decisions(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Owner-approved decisions from the CIK review (``prism.scripts.cik_review_sheet``), keyed by (ticker, start).

    Each decision becomes a list of segments that must tile its interval exactly: one
    segment for a single CIK, several for a split. A decision that does not tile is refused
    (N7), and so is one whose interval no longer matches the membership (checked in ``build``).
    """
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    out = {}
    for row in payload["decisions"]:
        start, end = pd.Timestamp(row["start"]), pd.Timestamp(row["end"])
        if row["kind"] == "split":
            segments = [{"start": pd.Timestamp(s["from"]), "end": pd.Timestamp(s["to"]), "cik": int(s["cik"]),
                         "title": s.get("name")} for s in row["segments"]]
        elif row["kind"] == "cik":
            segments = [{"start": start, "end": end, "cik": int(row["cik"]), "title": row.get("name")}]
        else:
            raise edgar.EdgarError(f"review decision for {row['ticker']} {row['start']} has kind {row['kind']!r}")
        segments.sort(key=lambda s: s["start"])
        tiles = segments[0]["start"] == start and segments[-1]["end"] == end and all(
            b["start"] == a["end"] + pd.Timedelta(days=1) for a, b in zip(segments, segments[1:]))
        if not tiles:
            raise edgar.EdgarError(f"review decision for {row['ticker']} {row['start']} does not tile its interval")
        key = (normalize_ticker(row["ticker"]), str(start.date()))
        out[key] = {"segments": segments, "end": end, "source": "review_decision", "reason": row.get("reason")}
    return out


def merge_review_decisions(paths: Sequence[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """Several decision files as one mapping; an interval decided in two files is refused (N7)."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for key, decision in load_review_decisions(path).items():
            if key in merged:
                raise edgar.EdgarError(f"{key[0]} {key[1]} is decided in more than one decisions file ({path})")
            merged[key] = decision
    return merged


def _segments_for(span: Any, override: Mapping[str, Any] | None,
                  fallback: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The (start, end, cik, title, source) pieces a membership interval resolves to."""
    if override is None:
        override = fallback
    if "segments" not in override:
        return [{"start": span.start, "end": span.end, "cik": override["cik"], "title": override["title"],
                 "source": override["source"], "reason": override.get("reason")}]
    if override["end"] != span.end:
        raise edgar.EdgarError(f"review decision for {span.ticker} {span.start.date()} ends {override['end'].date()}, "
                               f"but the membership interval ends {span.end.date()}; re-review it")
    return [{**segment, "source": override["source"], "reason": override.get("reason")}
            for segment in override["segments"]]


def build(membership: pd.DataFrame, client: edgar.SecClient, out_dir: Path, ledger_path: Path, asof: str,
          refresh: bool = False, limit: int | None = None,
          name_matches: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """One company at a time: filings, interval checks, facts, appended to disk, then released.

    Peak memory is one company's payloads plus the small identity tables, not the whole
    universe's facts (the 2026-09-24 build held ~1,100 companies in memory at once).
    A ``name_matches`` entry assigns its CIK to that one membership interval, ahead of the
    ticker's current-map candidate, so a reused ticker's earlier holder resolves separately;
    an entry with ``segments`` (a review decision) splits the interval between CIKs.
    """
    tickers_payload, tickers_meta = client.company_tickers(refresh)
    sec_map = sec_ticker_map(tickers_payload)
    tickers = sorted(membership["ticker"].map(normalize_ticker).unique())
    if limit is not None:
        tickers = tickers[:limit]
    candidates = candidate_ciks(tickers, sec_map)
    overrides = dict(name_matches or {})
    spans = membership.assign(ticker=membership["ticker"].map(normalize_ticker))
    spans = spans[spans["ticker"].isin(tickers)].sort_values(["ticker", "start"], kind="mergesort")
    assigned: list[dict[str, Any]] = []
    for span in spans.itertuples():
        override = overrides.get((span.ticker, str(span.start.date())))
        for piece in _segments_for(span, override, candidates[span.ticker]):
            assigned.append({"ticker": span.ticker, **piece})
    intervals_by_cik: dict[int, list[dict[str, Any]]] = {}
    for item in assigned:
        if item["cik"] is not None and item["end"] >= XBRL_ERA_START:
            intervals_by_cik.setdefault(int(item["cik"]), []).append(item)

    def interval_rows(items: list[dict[str, Any]], filings: pd.DataFrame | None) -> list[dict[str, Any]]:
        out = []
        for item in items:
            if item["end"] < XBRL_ERA_START:
                status = "pre_xbrl"
            elif filings is None:
                status = "unresolved"
            else:
                status = interval_status(filings, item["start"], item["end"])
            out.append({**item, "status": status})
        return out

    out_dir.mkdir(parents=True, exist_ok=True)
    facts_sink = _TableSink(out_dir / "facts.parquet", FACTS_SCHEMA)
    filings_sink = _TableSink(out_dir / "filings.parquet", FILINGS_SCHEMA)
    manifest: dict[str, Any] = {"company_tickers": tickers_meta, "submissions": {}, "companyfacts": {}}
    rows: list[dict[str, Any]] = []
    missing_facts: list[int] = []
    basis_counts: dict[str, int] = {}
    concepts = edgar.concept_names()
    try:
        rows.extend(interval_rows([a for a in assigned if a["cik"] is None or a["end"] < XBRL_ERA_START], None))
        for cik in sorted(intervals_by_cik):
            try:
                filings, metas = all_filings(client, cik, refresh)
                manifest["submissions"][str(cik)] = metas
            except edgar.EdgarNotFound:
                filings = pd.DataFrame(columns=list(edgar.FILING_COLUMNS))
            cik_rows = interval_rows(intervals_by_cik[cik], filings)
            rows.extend(cik_rows)
            if not any(row["status"] == "accepted" for row in cik_rows):
                continue
            filings_sink.write(filings)
            try:
                payload, meta = client.companyfacts(cik, refresh)
            except edgar.EdgarNotFound:
                missing_facts.append(cik)
                continue
            manifest["companyfacts"][str(cik)] = meta
            facts = edgar.attach_availability(edgar.companyfacts_frame(payload, concepts), filings)
            del payload
            for basis, count in facts["availability_basis"].value_counts().items():
                basis_counts[str(basis)] = basis_counts.get(str(basis), 0) + int(count)
            facts_sink.write(facts)
    except BaseException:
        facts_sink.abort()
        filings_sink.abort()
        raise

    intervals = pd.DataFrame(rows).sort_values(["ticker", "start"], kind="mergesort").reset_index(drop=True)
    intervals["cik"] = intervals["cik"].astype("Int64")
    accepted_ciks = sorted({int(c) for c in intervals.loc[intervals["status"] == "accepted", "cik"]})
    facts_sink.commit()
    filings_sink.commit()
    _replace_parquet(intervals[intervals["status"] == "accepted"].reset_index(drop=True),
                     out_dir / "ticker_cik.parquet")
    manifest["built_at"] = datetime.now(timezone.utc).isoformat()
    _replace_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))

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
        "facts_rows": facts_sink.rows,
        "filings_rows": filings_sink.rows,
        "availability_basis": basis_counts,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "status_counts": intervals["status"].value_counts().to_dict(),
        "not_accepted": [
            {"ticker": r.ticker, "start": str(r.start.date()), "end": str(r.end.date()),
             "cik": None if pd.isna(r.cik) else int(r.cik), "status": r.status, "reason": r.reason}
            for r in xbrl_era[xbrl_era["status"] != "accepted"].itertuples()
        ],
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    _replace_text(ledger_path, json.dumps(ledger, indent=2, sort_keys=True))
    return ledger


def _replace_text(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def _replace_parquet(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    frame.to_parquet(partial)
    os.replace(partial, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the point-in-time EDGAR fundamentals store.")
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--asof", required=True, help="label for the coverage ledger, YYYY-MM-DD")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/edgar/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/edgar"))
    parser.add_argument("--ledger-dir", type=Path, default=Path("data/universe"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="first N tickers only (smoke runs)")
    parser.add_argument("--name-matches", type=Path, default=None,
                        help="cik_name_matches ledger from prism.scripts.match_delisted_ciks (per-interval overrides)")
    parser.add_argument("--review-decisions", type=Path, nargs="+", default=None,
                        help="owner-approved decision files (cik_review_decisions, cik_nearmiss_decisions); "
                             "they take precedence over name matches, and two files deciding one interval is refused")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = edgar.SecClient(edgar.user_agent_from_env(), args.cache_dir)
    membership = pd.read_parquet(args.membership)
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    ledger = build(membership, client, args.out_dir, args.ledger_dir / f"edgar_coverage_{args.asof}{suffix}.json",
                   args.asof, refresh=args.refresh, limit=args.limit,
                   name_matches={**load_name_matches(args.name_matches),
                                 **merge_review_decisions(args.review_decisions or [])})
    summary = {k: ledger[k] for k in ("tickers", "xbrl_era_tickers", "accepted_tickers", "coverage_xbrl_era",
                                      "accepted_ciks", "facts_rows", "filings_rows", "peak_rss_mb", "status_counts")}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
