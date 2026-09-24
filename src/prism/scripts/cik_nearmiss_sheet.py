"""Review sheet for ticker intervals whose Wikipedia name has no exact SEC name match.

``prism.scripts.match_delisted_ciks`` accepts only exact normalized-name matches, so
"Taylor Morrison" does not reach "TAYLOR MORRISON HOME CORP" and "CR Bard" does not
reach SEC's inverted "BARD C R INC". This script proposes close matches for the owner
to approve; nothing here is applied without a recorded decision::

    PRISM_SEC_USER_AGENT="Prism research you@example.com" python -m prism.scripts.cik_nearmiss_sheet \\
        --matches data/universe/cik_name_matches_2026-09-24.json --asof 2026-09-24

Candidates, each rule named in the output (words are the normalized name's words, with
runs of single letters merged: "C R" → "CR", and a trailing AND/OF dropped):

Common abbreviations are expanded first (``EXPANSIONS``: US → UNITED STATES, INTL, ...).

* ``same_letters``: the same letters once spaces are removed, in either word order
  ("SOTHEBY S" / "SOTHEBYS", "CR BARD" / "BARD CR").
* ``all_words``: every Wikipedia word appears in the SEC name ("TAYLOR MORRISON" in
  "TAYLOR MORRISON HOME"); scored down by each extra SEC word.
* ``first_word``: the same first word ("WEBSTER BANK" / "WEBSTER FINANCIAL"), scored by
  word overlap.
* ``ticker_name``: the SEC name is the ticker itself ("BEAM", "CA").

The ten best per interval are checked the same way as every other match: the filer must
have filed a 10-K or 10-Q during the interval, and must have matched under a name it held
during the interval (iStar, once "Starwood Financial", does not match Starwood for 2009-16).
Survivors are decided in order: a structural split from the review sheet's rules
(``prism.scripts.cik_review_sheet.suggest``); else the one candidate matching by the
strongest rule (``RULE_RANK``), unless the review rules pick a different one, in which
case the case goes to judgment; else the review rules' pick. A pick that rests on
``ticker_name`` alone is always marked WEAK.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from prism.io import edgar
from prism.scripts.build_fundamentals import all_filings, interval_status
from prism.scripts.cik_review_sheet import candidate_row, suggest
from prism.scripts.match_delisted_ciks import iter_name_index, normalize_name

TOP_N = 10
MIN_SCORE = 0.3


#: Abbreviations expanded on both sides before comparing (kept short and explicit).
EXPANSIONS = {"US": ("UNITED", "STATES"), "USA": ("UNITED", "STATES"), "INTL": ("INTERNATIONAL",),
              "NATL": ("NATIONAL",), "MFG": ("MANUFACTURING",)}


def words(name: str) -> list[str]:
    """Normalized words: runs of single letters merged ("C R" → "CR"), abbreviations expanded,
    a trailing AND/OF dropped."""
    merged: list[str] = []
    in_run = False
    for word in normalize_name(name).split():
        if len(word) == 1 and word.isalpha():
            if in_run:
                merged[-1] += word
            else:
                merged.append(word)
                in_run = True
        else:
            merged.append(word)
            in_run = False
    expanded = [part for word in merged for part in EXPANSIONS.get(word, (word,))]
    while len(expanded) > 1 and expanded[-1] in {"AND", "OF"}:
        expanded.pop()
    return expanded


def closeness(target: Sequence[str], sec: Sequence[str], ticker: str) -> tuple[float, str] | None:
    """(score, rule) for an SEC name against a Wikipedia name, or None if no rule applies."""
    if not target or not sec:
        return None
    if "".join(target) == "".join(sec) or "".join(sorted(target)) == "".join(sorted(sec)):
        return 1.0, "same_letters"
    if set(target) <= set(sec):
        return max(0.9 - 0.05 * (len(set(sec)) - len(set(target))), 0.5), "all_words"
    if target[0] == sec[0]:
        overlap = len(set(target) & set(sec)) / len(set(target) | set(sec))
        return 0.3 + 0.4 * overlap, "first_word"
    if "".join(sec) == ticker.upper().replace(".", ""):
        return 0.6, "ticker_name"
    return None


def close_candidates(raw_index: bytes, targets: Mapping[tuple[str, str, str], list[str]]
                     ) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """For each (ticker, start, end) target with its Wikipedia names, the best SEC names by rule."""
    wanted = {key: [words(n) for n in names] for key, names in targets.items()}
    by_word: dict[str, set[tuple[str, str, str]]] = {}
    for key, name_words in wanted.items():
        for w in name_words:
            for token in [*w, "".join(w), "".join(sorted(w))]:  # run-together forms reach "SOTHEBYS", "CABELAS"
                by_word.setdefault(token, set()).add(key)
        by_word.setdefault(key[0].upper().replace(".", ""), set()).add(key)
    found: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {key: {} for key in wanted}
    for name, cik in iter_name_index(raw_index):
        sec = words(name)
        if not sec:
            continue
        keys = set()
        for token in {sec[0], "".join(sec)}:
            keys |= by_word.get(token, set())
        for key in keys:
            best = None
            for target in wanted[key]:
                result = closeness(target, sec, key[0])
                if result and (best is None or result[0] > best[0]):
                    best = result
            if best and best[0] >= MIN_SCORE:
                current = found[key].get(cik)
                if current is None or best[0] > current["score"]:
                    found[key][cik] = {"cik": cik, "sec_name": name, "score": round(best[0], 3), "rule": best[1]}
    return {key: sorted(c.values(), key=lambda c: (-c["score"], c["cik"]))[:TOP_N] for key, c in found.items()}


RULE_RANK = {"same_letters": 4, "all_words": 3, "first_word": 2, "ticker_name": 1}


def strength_during(names_during: Sequence[str], wiki_names: Sequence[str], ticker: str) -> tuple[int, str] | None:
    """The strongest rule by which a name the filer held during the interval matches Wikipedia's name."""
    best = None
    for held in names_during:
        for wiki in wiki_names:
            result = closeness(words(wiki), words(held), ticker)
            if result and result[0] >= MIN_SCORE and (best is None or RULE_RANK[result[1]] > best[0]):
                best = (RULE_RANK[result[1]], result[1])
    return best


def decide_near_miss(ticker: str, start: pd.Timestamp, end: pd.Timestamp, rows: list[dict[str, Any]]
                     ) -> dict[str, Any]:
    """Structural splits first, then the one candidate with the strongest name rule, then the review rules."""
    if not rows:
        return {"kind": "none", "basis": "no close match that filed during the interval under a matching name; "
                                         "needs judgment"}
    if len(rows) == 1:
        r = rows[0]
        weak = "WEAK (ticker-as-name only); confirm: " if r["strength_rule"] == "ticker_name" else ""
        return {"kind": "cik", "cik": r["cik"], "name": r["name_now"],
                "basis": f"{weak}only close match that filed during the interval ({r['strength_rule']}: "
                         f"{r['names_during_interval']})"}
    structural = suggest(ticker, start, end, rows)
    if structural["kind"] == "split":
        return structural
    top = max(r["strength_rank"] for r in rows)
    strongest = [r for r in rows if r["strength_rank"] == top]
    if len(strongest) == 1:
        pick = strongest[0]
        if structural["kind"] == "cik" and structural["cik"] != pick["cik"] and not structural["basis"].startswith("WEAK"):
            return {"kind": "none", "basis": f"signals disagree: strongest name rule picks {pick['cik']}, "
                                             f"{structural['basis']} picks {structural['cik']}; needs judgment"}
        weak = "WEAK (ticker-as-name only); confirm: " if pick["strength_rule"] == "ticker_name" else ""
        return {"kind": "cik", "cik": pick["cik"], "name": pick["name_now"],
                "basis": f"{weak}the only candidate matching by {pick['strength_rule']} under a name held during the "
                         f"interval ({pick['names_during_interval']})"}
    return structural


def build_sheet(matches: Mapping[str, Any], raw_index: bytes, client: edgar.SecClient) -> list[dict[str, Any]]:
    cases = [r for r in matches["review"] if r["reason"] == "no_sec_name_match"]
    targets = {(r["ticker"], r["start"], r["end"]): sorted({n["name"] for n in r["wikipedia_names"]}) for r in cases}
    proposals = close_candidates(raw_index, targets)
    sheet = []
    for (ticker, start_s, end_s), names in targets.items():
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        rows, checked = [], []
        for proposal in proposals[(ticker, start_s, end_s)]:
            try:
                payload, _ = client.submissions(proposal["cik"])
                filings, _ = all_filings(client, proposal["cik"], refresh=False)
            except edgar.EdgarNotFound:
                checked.append({**proposal, "status": "no_submissions"})
                continue
            status = interval_status(filings, start, end)
            row = None
            if status == "accepted":
                row = candidate_row(ticker, proposal["cik"], payload, filings, start, end, names)
                strength = strength_during(row["names_during_interval"], names, ticker)
                if strength is None:
                    status = "name_not_held_during_interval"
                else:
                    rows.append({**row, "score": proposal["score"], "rule": proposal["rule"],
                                 "sec_name": proposal["sec_name"], "strength_rank": strength[0],
                                 "strength_rule": strength[1]})
            checked.append({**proposal, "status": status})
        sheet.append({"ticker": ticker, "start": start_s, "end": end_s, "wikipedia_names": names,
                      "proposals_checked": checked, "candidates": rows,
                      "suggestion": decide_near_miss(ticker, start, end, rows)})
    return sheet


def render(cases: Sequence[Mapping[str, Any]], asof: str) -> str:
    lines = [f"# CIK near-miss review sheet — {asof}", "",
             "Ticker intervals whose Wikipedia name has no exact SEC name match. Close matches were proposed by "
             "the named rules and kept only if the filer filed a 10-K or 10-Q during the interval. Decisions are "
             f"recorded separately, in `cik_nearmiss_decisions_{asof}.json`, and only those are applied.", "",
             "| # | ticker | interval | Wikipedia name | suggestion | basis |", "|---|---|---|---|---|---|"]
    for i, case in enumerate(cases, 1):
        s = case["suggestion"]
        if s["kind"] == "split":
            pick = "split — " + "; ".join(f"{g['from']}→{g['to']}: {g['cik']} {g['name']}" for g in s["segments"])
        elif s["kind"] == "cik":
            pick = f"{s['cik']} {s['name']}"
        else:
            pick = "**needs judgment**"
        lines.append(f"| {i} | {case['ticker']} | {case['start']} → {case['end']} | "
                     f"{', '.join(case['wikipedia_names'])} | {pick} | {s['basis']} |")
    lines += ["", "## Proposals checked", ""]
    for i, case in enumerate(cases, 1):
        lines.append(f"**{i}. {case['ticker']}** ({case['start']} → {case['end']}; Wikipedia: "
                     f"{', '.join(case['wikipedia_names'])})")
        for p in case["proposals_checked"]:
            lines.append(f"- {p['cik']} '{p['sec_name']}' — {p['rule']} {p['score']}; {p['status']}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review sheet for near-miss CIK name matches.")
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/edgar/raw"))
    parser.add_argument("--ledger-dir", type=Path, default=Path("data/universe"))
    args = parser.parse_args(argv)

    client = edgar.SecClient(edgar.user_agent_from_env(), args.cache_dir)
    raw_index, _ = client.name_index()
    cases = build_sheet(json.loads(args.matches.read_text()), raw_index, client)
    del raw_index
    stem = args.ledger_dir / f"cik_nearmiss_review_{args.asof}"
    stem.with_suffix(".json").write_text(json.dumps({"asof": args.asof, "matches_input": str(args.matches),
                                                     "cases": cases}, indent=2, sort_keys=True, default=str))
    stem.with_suffix(".md").write_text(render(cases, args.asof))
    kinds = pd.Series([c["suggestion"]["kind"] for c in cases]).value_counts().to_dict()
    print(json.dumps({"cases": len(cases), "suggestions": kinds}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
