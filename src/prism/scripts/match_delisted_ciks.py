"""Map unresolved index tickers to SEC CIKs by company name, validated by filing activity.

The fundamentals builder (``prism.scripts.build_fundamentals``) maps a ticker through
SEC's *current* ticker map, so names that were delisted, acquired or renamed arrive
unresolved, and a reused ticker's earlier holder arrives rejected. This step recovers
them from the company name Wikipedia's index change tables record beside the ticker::

    PRISM_SEC_USER_AGENT="Prism research you@example.com" python -m prism.scripts.match_delisted_ciks \\
        --coverage data/universe/edgar_coverage_2026-09-24.json --asof 2026-09-24

Per unresolved ticker interval (XBRL era only):

1. **Names.** The securities Wikipedia lists for that ticker's removal near the
   interval's end and its addition near its start, on the S&P 500 and S&P 400 pages
   (±45 days; failing both, any change row for the ticker inside the interval), plus
   the name the constituents table gives a still-listed ticker; where that table
   carries a CIK (the S&P 500 page does), the CIK is a candidate too. Pages
   are read at revisions pinned by date (``WIKI_REVISION_ASOF``), never live: the S&P
   500 article dropped its changes table in 2026-07/08.
2. **Candidates.** Every SEC entity whose name, normalized the same way (case,
   punctuation, ``&``→AND, SEC's ``/NEW/``-style tags and trailing legal-form words
   such as Inc, Corp, Holdings, Ltd removed), equals one of those names exactly.
3. **Validation.** A candidate survives only if it filed a 10-K or 10-Q during the
   interval (``build_fundamentals.interval_status``): same name, same era, reporting.
4. **Decision.** Exactly one survivor is accepted, with every piece of evidence
   recorded. Several survivors, or none, go to the review list with the reason. Nothing
   is guessed and nothing fuzzy is accepted.

Output: the tracked ``{ledger_dir}/cik_name_matches_{asof}.json``, which the builder
reads with ``--name-matches`` as per-interval overrides.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd

from prism.io import edgar
from prism.io.universe_sp500 import extract_tables, normalize_ticker
from prism.scripts.build_fundamentals import (
    XBRL_ERA_START,
    all_filings,
    interval_status,
)

logger = logging.getLogger(__name__)

#: Wikipedia article titles. The S&P 500 article's "Selected changes" table was removed
#: between revision 1364118843 (2026-07-14, 220 KB) and mid-August 2026 (~87 KB), so
#: names are always read from a revision pinned by date, never from the live page.
WIKI_TITLES = {"sp500": "List_of_S%26P_500_companies", "sp400": "List_of_S%26P_400_companies"}
#: Revision dates: the build dates of the membership files the coverage ledger came from
#: (``sp500_membership_2026-07-17``, ``sp400_membership_2026-09-23``).
WIKI_REVISION_ASOF = {"sp500": "2026-07-17", "sp400": "2026-09-23"}
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_INDEX = "https://en.wikipedia.org/w/index.php"
NAME_WINDOW = pd.Timedelta(days=45)
PERIODIC = ("10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT")

#: Legal-form and share-class words dropped from the end of a name (repeatedly).
TRAILING_WORDS = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "LLC", "LP", "NV",
    "SA", "AG", "SE", "HOLDINGS", "HOLDING", "GROUP", "CLASS", "COMMON", "STOCK", "SHARES",
}
_SEC_TAG = re.compile(r"/[A-Z0-9 .]{1,12}/?")
_PAREN = re.compile(r"\([^)]*\)")
_FOOTNOTE = re.compile(r"\[[^\]]*\]")


def normalize_name(name: str) -> str:
    """Upper-case company name with punctuation, SEC state tags and trailing legal forms removed."""
    text = _FOOTNOTE.sub(" ", str(name)).upper()
    text = _PAREN.sub(" ", text)
    text = _SEC_TAG.sub(" ", text)
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    words = text.split()
    if words and words[0] == "THE":
        words = words[1:]
    while len(words) > 1 and (words[-1] in TRAILING_WORDS or (len(words[-1]) == 1 and words[-2] == "CLASS")):
        words = words[:-1]
    return " ".join(words)


# --------------------------------------------------------------------- names from Wikipedia


def change_rows(tables: list[list[list[str]]], index: str) -> pd.DataFrame:
    """Rows of a 'Selected changes' table with the security names the universe builder discards.

    Same layout assumption as ``prism.io.universe_sp500.parse_changes_table``:
    ``[date, added_ticker, added_security, removed_ticker, removed_security, reason]``.
    """
    for table in tables:
        header = next((i for i, row in enumerate(table)
                       if any("added" in c.lower() for c in row) and any("removed" in c.lower() for c in row)), None)
        if header is None:
            continue
        records = []
        for row in table[header + 1:]:
            date = pd.to_datetime(row[0], errors="coerce") if row else pd.NaT
            if pd.isna(date) or len(row) < 5:
                continue
            records.append({"index": index, "date": date, "added": normalize_ticker(row[1]), "added_name": row[2],
                            "removed": normalize_ticker(row[3]), "removed_name": row[4]})
        if records:
            return pd.DataFrame.from_records(records)
    raise edgar.EdgarError(f"no changes table with Added/Removed columns on the {index} page")


def constituent_rows(tables: list[list[list[str]]], index: str) -> pd.DataFrame:
    """The constituents table at the pinned revision: ticker, security name, and CIK where the page lists one."""
    for table in tables:
        if not table or not any("symbol" in c.lower() for c in table[0]):
            continue
        header = [c.strip().lower() for c in table[0]]
        sym, sec = header.index("symbol"), header.index("security")
        cik_col = header.index("cik") if "cik" in header else None
        records = []
        for row in table[1:]:
            if len(row) <= max(sym, sec):
                continue
            cik = row[cik_col].strip() if cik_col is not None and cik_col < len(row) else ""
            records.append({"index": index, "ticker": normalize_ticker(row[sym]), "security": row[sec],
                            "cik": int(cik) if cik.isdigit() else None})
        if records:
            return pd.DataFrame.from_records(records)
    raise edgar.EdgarError(f"no constituents table with a Symbol column on the {index} page")


def interval_names(changes: pd.DataFrame, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Names Wikipedia gives this ticker at the interval's removal and addition dates."""
    removed = changes[(changes["removed"] == ticker) & ((changes["date"] - end).abs() <= NAME_WINDOW)]
    added = changes[(changes["added"] == ticker) & ((changes["date"] - start).abs() <= NAME_WINDOW)]
    out = [{"name": r.removed_name, "role": "removed", "date": str(r.date.date()), "index": r.index}
           for r in removed.itertuples(index=False)]
    out += [{"name": r.added_name, "role": "added", "date": str(r.date.date()), "index": r.index}
            for r in added.itertuples(index=False)]
    if not out:
        inside = changes[(changes["date"] >= start - NAME_WINDOW) & (changes["date"] <= end + NAME_WINDOW)]
        out += [{"name": r.removed_name, "role": "removed_in_interval", "date": str(r.date.date()), "index": r.index}
                for r in inside[inside["removed"] == ticker].itertuples(index=False)]
        out += [{"name": r.added_name, "role": "added_in_interval", "date": str(r.date.date()), "index": r.index}
                for r in inside[inside["added"] == ticker].itertuples(index=False)]
    return [n for n in out if normalize_name(n["name"])]


def constituent_names(constituents: pd.DataFrame, ticker: str) -> list[dict]:
    """Names (and page CIKs) the pinned constituents tables give this ticker."""
    rows = constituents[constituents["ticker"] == ticker]
    return [{"name": r.security, "role": "constituent", "index": r.index,
             "page_cik": None if pd.isna(r.cik) else int(r.cik)} for r in rows.itertuples(index=False)
            if normalize_name(r.security)]


# --------------------------------------------------------------------- names at SEC


def iter_name_index(raw: bytes) -> Iterator[tuple[str, int]]:
    """``(name, cik)`` per line of SEC's ``NAME:CIK:`` index, streamed."""
    for line in io.TextIOWrapper(io.BytesIO(raw), encoding="latin-1"):
        line = line.rstrip("\n")
        if not line.endswith(":"):
            continue
        name, _, rest = line[:-1].rpartition(":")
        if rest.isdigit():
            yield name, int(rest)


def sec_candidates(raw_index: bytes, wanted: Iterable[str]) -> dict[str, list[tuple[int, str]]]:
    """For each wanted normalized name, every (cik, SEC name) whose normalized name equals it."""
    targets = set(wanted)
    out: dict[str, list[tuple[int, str]]] = {}
    for name, cik in iter_name_index(raw_index):
        key = normalize_name(name)
        if key in targets:
            out.setdefault(key, []).append((cik, name))
    return out


# --------------------------------------------------------------------- decision


def decide(ticker: str, start: pd.Timestamp, end: pd.Timestamp, names: list[dict],
           candidates: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Accept exactly one validated candidate; otherwise send the interval to review."""
    record = {"ticker": ticker, "start": str(start.date()), "end": str(end.date()), "wikipedia_names": names,
              "candidates": [{"cik": cik, **info} for cik, info in sorted(candidates.items())]}
    valid = [cik for cik, info in candidates.items() if info["status"] == "accepted"]
    if not names:
        return {**record, "decision": "review", "reason": "no_wikipedia_name"}
    if not candidates:
        return {**record, "decision": "review", "reason": "no_sec_name_match"}
    if len(valid) == 1:
        return {**record, "decision": "accepted", "cik": valid[0], "title": candidates[valid[0]]["entity_name"]}
    reason = "no_candidate_filed_in_interval" if not valid else "several_candidates_filed_in_interval"
    return {**record, "decision": "review", "reason": reason}


def candidate_evidence(client: edgar.SecClient, cik: int, sec_names: list[str], start: pd.Timestamp,
                       end: pd.Timestamp, refresh: bool) -> dict[str, Any]:
    try:
        payload, _ = client.submissions(cik, refresh)
        filings, _ = all_filings(client, cik, refresh)
    except edgar.EdgarNotFound:
        return {"sec_names": sec_names, "entity_name": None, "status": "no_submissions"}
    periodic = filings[filings["form"].isin(PERIODIC)]
    inside = periodic[(periodic["filing_date"] >= max(start, XBRL_ERA_START)) & (periodic["filing_date"] <= end)]
    return {
        "sec_names": sec_names,
        "entity_name": payload.get("name"),
        "former_names": [f.get("name") for f in payload.get("formerNames", [])],
        "current_tickers": payload.get("tickers", []),
        "periodic_filings_in_interval": int(len(inside)),
        "first_periodic": None if periodic.empty else str(periodic["filing_date"].min().date()),
        "last_periodic": None if periodic.empty else str(periodic["filing_date"].max().date()),
        "status": interval_status(filings, start, end),
    }


def revision_before(payload: Mapping[str, Any]) -> tuple[int, str]:
    """The (revid, timestamp) of the single revision a ``prop=revisions`` API query returned."""
    try:
        (page,) = payload["query"]["pages"].values()
        (revision,) = page["revisions"]
        return int(revision["revid"]), str(revision["timestamp"])
    except (KeyError, ValueError, TypeError) as exc:
        raise edgar.EdgarError(f"unexpected Wikipedia revisions payload: {payload!r:.300}") from exc


def fetch_wikipedia(page: str, cache_dir: Path, revision_asof: str, refresh: bool) -> tuple[str, dict[str, Any]]:
    """The article as of the last revision at or before ``revision_asof`` (end of day UTC), cached by revid."""
    import requests  # local import, as in prism.io.universe_sp500

    headers = {"User-Agent": "prism/universe-builder"}
    params = {"action": "query", "prop": "revisions", "titles": WIKI_TITLES[page].replace("%26", "&"),
              "rvlimit": 1, "rvdir": "older", "rvstart": f"{revision_asof}T23:59:59Z",
              "rvprop": "ids|timestamp", "format": "json"}
    response = requests.get(WIKI_API, params=params, timeout=30, headers=headers)
    response.raise_for_status()
    revid, timestamp = revision_before(response.json())
    url = f"{WIKI_INDEX}?title={WIKI_TITLES[page]}&oldid={revid}"
    path = cache_dir / f"wikipedia_{page}_rev{revid}.html"
    if refresh or not path.exists():
        body = requests.get(url, timeout=30, headers=headers)
        body.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body.text)
    html = path.read_text()
    return html, {"url": url, "revid": revid, "revision_timestamp": timestamp, "revision_asof": revision_asof,
                  "path": str(path), "sha256": hashlib.sha256(html.encode()).hexdigest()}


def match(coverage: Mapping[str, Any], changes: pd.DataFrame, raw_index: bytes, client: edgar.SecClient,
          refresh: bool = False, constituents: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    targets = [(normalize_ticker(r["ticker"]), pd.Timestamp(r["start"]), pd.Timestamp(r["end"]))
               for r in coverage["not_accepted"]
               if r["status"] in ("unresolved", "no_periodic_filing_in_interval")
               and pd.Timestamp(r["end"]) >= XBRL_ERA_START]
    if constituents is None:
        constituents = pd.DataFrame(columns=["index", "ticker", "security", "cik"])
    names = {t: interval_names(changes, *t) + constituent_names(constituents, t[0]) for t in targets}
    wanted = {normalize_name(n["name"]) for found in names.values() for n in found}
    by_name = sec_candidates(raw_index, wanted)
    results = []
    for ticker, start, end in targets:
        keys = {normalize_name(n["name"]) for n in names[(ticker, start, end)]}
        pool: dict[int, list[str]] = {}
        for key in keys:
            for cik, sec_name in by_name.get(key, []):
                pool.setdefault(cik, []).append(sec_name)
        for n in names[(ticker, start, end)]:
            if n.get("page_cik") is not None:
                pool.setdefault(int(n["page_cik"]), []).append(f"wikipedia {n['index']} constituents CIK")
        candidates = {cik: candidate_evidence(client, cik, sec_names, start, end, refresh)
                      for cik, sec_names in pool.items()}
        results.append(decide(ticker, start, end, names[(ticker, start, end)], candidates))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Map unresolved index tickers to SEC CIKs by company name.")
    parser.add_argument("--coverage", type=Path, required=True, help="edgar_coverage ledger from build_fundamentals")
    parser.add_argument("--asof", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/edgar/raw"))
    parser.add_argument("--ledger-dir", type=Path, default=Path("data/universe"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = edgar.SecClient(edgar.user_agent_from_env(), args.cache_dir)
    coverage = json.loads(args.coverage.read_text())
    sources: dict[str, Any] = {}
    frames, members = [], []
    for page in WIKI_TITLES:
        html, meta = fetch_wikipedia(page, args.cache_dir, WIKI_REVISION_ASOF[page], args.refresh)
        sources[f"wikipedia_{page}"] = meta
        tables = extract_tables(html)
        frames.append(change_rows(tables, page))
        members.append(constituent_rows(tables, page))
    raw_index, index_meta = client.name_index(args.refresh)
    sources["sec_name_index"] = {k: index_meta[k] for k in ("url", "fetched_at", "sha256")}
    results = match(coverage, pd.concat(frames, ignore_index=True), raw_index, client, args.refresh,
                    constituents=pd.concat(members, ignore_index=True))
    del raw_index

    accepted = [r for r in results if r["decision"] == "accepted"]
    review = [r for r in results if r["decision"] != "accepted"]
    reasons = pd.Series([r["reason"] for r in review], dtype=object).value_counts().to_dict()
    out = {
        "asof": args.asof,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "coverage_input": str(args.coverage),
        "sources": sources,
        "counts": {"intervals": len(results), "accepted": len(accepted), "review": len(review),
                   "review_reasons": reasons},
        "accepted": accepted,
        "review": review,
    }
    path = args.ledger_dir / f"cik_name_matches_{args.asof}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    print(json.dumps(out["counts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
