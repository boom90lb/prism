"""Review sheet for ticker intervals that name matching left ambiguous.

``prism.scripts.match_delisted_ciks`` refuses an interval when more than one SEC filer
whose name matches filed during it. Those are corporate-lineage cases (mergers that
kept the target's name, spin-offs, redomiciles, successor issuers), and they need a
decision per case. This script lays out the evidence and a suggestion for each; the
owner's decisions are recorded in the same file and nothing is applied without them::

    python -m prism.scripts.cik_review_sheet \\
        --matches data/universe/cik_name_matches_2026-09-24.json --asof 2026-09-24

Evidence per candidate, from cached SEC filing histories only:

* **Name during the interval**: the names SEC records the filer holding over the
  interval's XBRL-era part (former names carry from/to dates).
* **Ticker fit**: whether the ticker's letters appear, in order and starting with its
  first letter, in one of those names (ACE in "ACE Ltd", AGN in "Allergan Inc", not in
  "Actavis plc").
* **Coverage**: the first and last 10-K/10-Q inside the interval, and whether they
  reach within a quarter of its start and end.

Suggestion, in order (first rule that decides wins):

0. Names count only if held for 90 days inside the interval and inside the filer's
   active filing span. Candidates whose names are all LLC/LP forms (co-filing
   subsidiaries, operating partnerships) are dropped when a corporate candidate exists.
1. **Successor split.** Two candidates; the first stops filing altogether and the
   second's filings begin after it and span a quarter: split at the successor's first
   periodic filing in the interval.
2. **Name-handover split.** Exactly two candidates fit the ticker by name, in sequence,
   and the second keeps the fitting name for a quarter after the first loses it: split
   where the second's fitting name begins (TCF: old TCF, then Chemical Financial
   renamed TCF in 2019).
3. **Ticker fit.** Exactly one candidate fits the ticker (a trailing share-class
   letter may be dropped: VIAB → VIA).
4. **Name at the time.** Exactly one candidate held Wikipedia's name for the ticker
   during the interval.
5. **Only corporate filer** left after step 0.
6. **Coverage (weak).** Exactly one candidate files across the whole interval; marked
   WEAK for the reviewer to confirm.
7. Otherwise **none**: the case is marked for judgment, not guessed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from prism.io import edgar
from prism.scripts.build_fundamentals import XBRL_ERA_START, all_filings
from prism.scripts.match_delisted_ciks import PERIODIC, normalize_name

QUARTER = pd.Timedelta(days=120)


MIN_HOLD = pd.Timedelta(days=90)
SUBSIDIARY_FORMS = re.compile(r"\b(LLC|L\.?L\.?C|LP|L\.?P|LIMITED PARTNERSHIP|PARTNERSHIP)\b\.?\s*$", re.I)


def name_periods(payload: Mapping[str, Any], start: pd.Timestamp, end: pd.Timestamp,
                 active: tuple[pd.Timestamp, pd.Timestamp] | None) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """(name, from, to) the filer held within [start, end] and within its active filing span.

    SEC records former names with from/to dates; the current name runs from the last of
    them. A name counts only if held for ``MIN_HOLD`` (or the whole window, if shorter):
    a rename days before a removal does not make that name the ticker's name.
    """
    lo, hi = start, end
    if active is not None:
        lo, hi = max(lo, active[0]), min(hi, active[1])
    if lo > hi:
        return []
    spans, latest = [], None
    for former in payload.get("formerNames", []):
        begin = pd.Timestamp(str(former.get("from", ""))[:10] or "1900-01-01")
        until = pd.Timestamp(str(former.get("to", ""))[:10] or "2100-01-01")
        a, b = min(begin, until), max(begin, until)  # SEC occasionally records from > to
        latest = b if latest is None else max(latest, b)
        spans.append((str(former["name"]), a, b))
    spans.append((str(payload.get("name", "")), latest or pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")))
    need = min(MIN_HOLD, hi - lo)
    out = []
    for name, a, b in spans:
        a, b = max(a, lo), min(b, hi)
        if b - a >= need and name:
            out.append((name, a, b))
    return out


def ticker_variants(ticker: str) -> list[str]:
    """The ticker, and without a trailing share-class letter (VIAB → VIA, BRK.B → BRK)."""
    base = re.split(r"[./-]", ticker.upper())[0]
    out = [ticker.upper(), base]
    if len(base) >= 4 and base[-1] in "ABCK":
        out.append(base[:-1])
    return list(dict.fromkeys(out))


def ticker_fits(ticker: str, name: str) -> bool:
    """A variant of the ticker has its letters in order in the name, the first letters coinciding."""
    text = normalize_name(name).replace(" ", "")
    for variant in ticker_variants(ticker):
        letters = [c for c in variant if c.isalpha()]
        if not letters or not text or text[0] != letters[0]:
            continue
        position = 0
        for letter in letters:
            position = text.find(letter, position)
            if position < 0:
                break
            position += 1
        else:
            return True
    return False


def candidate_row(ticker: str, cik: int, payload: Mapping[str, Any], filings: pd.DataFrame,
                  start: pd.Timestamp, end: pd.Timestamp, wiki_names: Sequence[str]) -> dict[str, Any]:
    lo = max(start, XBRL_ERA_START)
    periodic = filings[filings["form"].isin(PERIODIC)]
    inside = periodic[(periodic["filing_date"] >= lo) & (periodic["filing_date"] <= end)]
    first = inside["filing_date"].min() if len(inside) else None
    last = inside["filing_date"].max() if len(inside) else None
    active = None if periodic.empty else (periodic["filing_date"].min() - QUARTER,
                                          periodic["filing_date"].max() + QUARTER)
    periods = name_periods(payload, lo, end, active)
    fitting = [(a, b) for name, a, b in periods if ticker_fits(ticker, name)]
    wiki = {normalize_name(n) for n in wiki_names}
    return {
        "cik": cik,
        "name_now": payload.get("name"),
        "names_during_interval": [name for name, _, _ in periods],
        "subsidiary_form_only": bool(periods) and all(SUBSIDIARY_FORMS.search(n) for n, _, _ in periods),
        "ticker_fit": bool(fitting),
        "ticker_fit_from": None if not fitting else str(min(a for a, _ in fitting).date()),
        "ticker_fit_to": None if not fitting else str(max(b for _, b in fitting).date()),
        "wikipedia_name_at_time": any(normalize_name(n) in wiki for n, _, _ in periods),
        "filings_in_interval": int(len(inside)),
        "first_in_interval": None if first is None else str(first.date()),
        "last_in_interval": None if last is None else str(last.date()),
        "last_periodic_overall": None if periodic.empty else str(periodic["filing_date"].max().date()),
        "covers_start": bool(first is not None and first - lo <= QUARTER),
        "covers_end": bool(last is not None and end - last <= QUARTER),
    }


def _split(start: pd.Timestamp, end: pd.Timestamp, first: Mapping[str, Any], second: Mapping[str, Any],
           cut: pd.Timestamp, basis: str) -> dict[str, Any]:
    return {"kind": "split", "basis": basis,
            "segments": [{"from": str(start.date()), "to": str((cut - pd.Timedelta(days=1)).date()),
                          "cik": first["cik"], "name": first["name_now"]},
                         {"from": str(cut.date()), "to": str(end.date()), "cik": second["cik"],
                          "name": second["name_now"]}]}


def suggest(ticker: str, start: pd.Timestamp, end: pd.Timestamp, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The layered rules in the module docstring, first that decides wins."""
    pool = [r for r in rows if r["names_during_interval"]]
    corporate = [r for r in pool if not r["subsidiary_form_only"]]
    dropped = []
    if corporate and len(corporate) < len(pool):
        dropped = [r["cik"] for r in pool if r["subsidiary_form_only"]]
        pool = corporate
    note = f"; co-filing LLC/LP dropped: {dropped}" if dropped else ""

    filed = sorted((r for r in pool if r["first_in_interval"]), key=lambda r: r["first_in_interval"])
    if len(filed) == 2:
        a, b = filed
        stops = pd.Timestamp(a["last_periodic_overall"]) <= pd.Timestamp(b["first_in_interval"]) + QUARTER
        span = pd.Timestamp(b["last_in_interval"]) - pd.Timestamp(b["first_in_interval"])
        if stops and span >= MIN_HOLD and a["last_in_interval"] < b["first_in_interval"]:
            return _split(start, end, a, b, pd.Timestamp(b["first_in_interval"]),
                          "successor issuer: the predecessor stopped filing and the successor took over" + note)
    fits = sorted((r for r in pool if r["ticker_fit"]), key=lambda r: r["ticker_fit_from"])
    if len(fits) == 2:
        a, b = fits
        overlap = pd.Timestamp(a["ticker_fit_to"]) - pd.Timestamp(b["ticker_fit_from"])
        outlasts = pd.Timestamp(b["ticker_fit_to"]) - pd.Timestamp(a["ticker_fit_to"])
        if pd.Timestamp(b["ticker_fit_from"]) > pd.Timestamp(a["ticker_fit_from"]) + MIN_HOLD \
                and overlap <= pd.Timedelta(days=150) and outlasts >= MIN_HOLD:
            return _split(start, end, a, b, pd.Timestamp(b["ticker_fit_from"]),
                          f"name handover: {ticker} fits {a['cik']} then {b['cik']}" + note)
    if len(fits) == 1:
        return {"kind": "cik", "cik": fits[0]["cik"], "name": fits[0]["name_now"],
                "basis": f"ticker fit: {ticker} in {fits[0]['names_during_interval']}" + note}
    named = [r for r in (fits or pool) if r["wikipedia_name_at_time"]]
    if len(named) == 1:
        return {"kind": "cik", "cik": named[0]["cik"], "name": named[0]["name_now"],
                "basis": "Wikipedia's name was this filer's name during the interval" + note}
    if len(pool) == 1:
        return {"kind": "cik", "cik": pool[0]["cik"], "name": pool[0]["name_now"],
                "basis": "the only corporate filer" + note}
    full = [r for r in pool if r["covers_start"] and r["covers_end"]]
    if len(full) == 1:
        return {"kind": "cik", "cik": full[0]["cik"], "name": full[0]["name_now"],
                "basis": "WEAK: the only candidate filing across the whole interval; confirm" + note}
    return {"kind": "none", "basis": "signals do not separate the candidates; needs judgment" + note}


def build_sheet(matches: Mapping[str, Any], client: edgar.SecClient) -> list[dict[str, Any]]:
    cases = []
    for record in matches["review"]:
        if record["reason"] != "several_candidates_filed_in_interval":
            continue
        ticker, start, end = record["ticker"], pd.Timestamp(record["start"]), pd.Timestamp(record["end"])
        rows = []
        for candidate in record["candidates"]:
            if candidate["status"] != "accepted":
                continue
            payload, _ = client.submissions(candidate["cik"])
            filings, _ = all_filings(client, candidate["cik"], refresh=False)
            rows.append(candidate_row(ticker, candidate["cik"], payload, filings, start, end,
                                      [n["name"] for n in record["wikipedia_names"]]))
        cases.append({"ticker": ticker, "start": record["start"], "end": record["end"],
                      "wikipedia_names": sorted({n["name"] for n in record["wikipedia_names"]}),
                      "candidates": rows, "suggestion": suggest(ticker, start, end, rows), "decision": None})
    return cases


def render(cases: Sequence[Mapping[str, Any]], asof: str) -> str:
    lines = [f"# CIK review sheet — {asof}", "",
             "Ticker intervals where more than one SEC filer with a matching name filed during the interval. "
             "Each row shows the suggestion and its basis; `decision` in the JSON twin stays empty until the "
             "owner approves or overrides it.", "",
             "| # | ticker | interval | Wikipedia name | suggestion | basis |", "|---|---|---|---|---|---|"]
    for i, case in enumerate(cases, 1):
        s = case["suggestion"]
        if s["kind"] == "split":
            pick = "; ".join(f"{g['from']}→{g['to']}: {g['cik']} {g['name']}" for g in s["segments"])
            pick = f"split — {pick}"
        elif s["kind"] == "cik":
            pick = f"{s['cik']} {s['name']}"
        else:
            pick = "**needs judgment**"
        lines.append(f"| {i} | {case['ticker']} | {case['start']} → {case['end']} | "
                     f"{', '.join(case['wikipedia_names'])} | {pick} | {s['basis']} |")
    lines += ["", "## Candidates", ""]
    for i, case in enumerate(cases, 1):
        lines.append(f"**{i}. {case['ticker']}** ({case['start']} → {case['end']})")
        for r in case["candidates"]:
            lines.append(f"- {r['cik']} *{r['name_now']}*: names during interval {r['names_during_interval']}; "
                         f"ticker fit {r['ticker_fit']} ({r['ticker_fit_from']} → {r['ticker_fit_to']}); "
                         f"LLC/LP only {r['subsidiary_form_only']}; {r['filings_in_interval']} filings "
                         f"{r['first_in_interval']} → {r['last_in_interval']} (last overall "
                         f"{r['last_periodic_overall']})")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review sheet for ambiguous name matches.")
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/edgar/raw"))
    parser.add_argument("--ledger-dir", type=Path, default=Path("data/universe"))
    args = parser.parse_args(argv)

    def cache_only(url: str, headers: Mapping[str, str], timeout: float) -> bytes:
        raise edgar.EdgarError(f"review sheet reads the cache only; {url} is not cached")

    client = edgar.SecClient("Prism review sheet cache-only@localhost", args.cache_dir, transport=cache_only)
    cases = build_sheet(json.loads(args.matches.read_text()), client)
    stem = args.ledger_dir / f"cik_review_{args.asof}"
    stem.with_suffix(".json").write_text(json.dumps({"asof": args.asof, "matches_input": str(args.matches),
                                                     "cases": cases}, indent=2, sort_keys=True))
    stem.with_suffix(".md").write_text(render(cases, args.asof))
    kinds = pd.Series([c["suggestion"]["kind"] for c in cases]).value_counts().to_dict()
    print(json.dumps({"cases": len(cases), "suggestions": kinds}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
