"""Survivorship-bias-free S&P 500 universe reconstruction (Koker method).

Point-in-time index membership is reconstructed from the Wikipedia "List of
S&P 500 companies" page: the current-constituents table plus the "Selected
changes to the list" table (date, added ticker, removed ticker). Walking the
changes backward from the scrape date yields the member set on any historical
date; the union over all dates is the survivorship-bias-free universe -- it
*includes* names since removed by delisting, acquisition, or relegation, which
a current-constituents snapshot would silently drop.

Price coverage is a separate, measured concern. Free vendors do not always
retain prices for delisted tickers, so the builder emits an explicit coverage
ledger (resolved vs skipped) rather than quietly dropping the dead names that
make the universe survivorship-free in the first place. Re-introducing
survivorship through missing delisted prices is only acceptable if it is
*counted* -- see ``compute_coverage`` and ``src/prism/scripts/build_sp500_universe.py``.

Dependency note: the project env has neither lxml nor bs4, so ``pandas.read_html``
is unavailable. The two tables are extracted with a small stdlib ``html.parser``
subclass, and every reconstruction function operates on plain DataFrames so the
logic is unit-testable without network.

Pages are read at the last revision on or before the build's as-of date
(``fetch_index_tables``), never live. The S&P 500 changes table moved to the
"Historical components of the S&P 500" article on 2026-08-11, and editors have
since deleted rename rows from it; ``CHANGES_SUPPLEMENT_TABLE`` puts those
back, with renames it never carried, and ``history_differences`` finds any new
rewrite against the reconstruction of a pinned baseline revision.

Reference: Teddy Koker, "Creating a Survivorship-Bias-Free S&P 500 Dataset with
Python" (2019). Koker used iShares IVV monthly holdings as the membership proxy;
this module uses the Wikipedia changes table (longer history) and keeps the
same explicit rename-table / skip-set discipline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Interval floor for names that were already members before the earliest row in
# the changes table (membership "since before records"). Chosen well before any
# backtest sample so masking treats it as -inf.
HISTORY_FLOOR = pd.Timestamp("1990-01-01")

# Manual, *reviewed* ticker remaps from Wikipedia spelling to the price vendor's
# spelling (Koker's `rename_table`). Entries are added only after the coverage
# ledger or an integrity sweep shows a name is wrong/missing *and* a human
# confirms the vendor symbol. Never auto-populate -- a wrong remap silently
# swaps one company's price history for another's.
# Reviewed 2026-07-14 (docs/data_integrity_diagnostic.md §6): pure index
# renames are invisible to the Wikipedia changes table, the old symbols'
# vendor resolutions are wrong instruments, and the successor caches carry the
# full genuine history.
RENAME_TABLE: dict[str, str] = {
    "FB": "META",  # Facebook -> Meta Platforms (2022-06); vendor "FB" is a 2025+ stray
    "PCLN": "BKNG",  # Priceline -> Booking Holdings (2018); vendor "PCLN" is a 2025+ stray
    "WLTW": "WTW",  # Willis Towers Watson rename (2022-01); vendor "WLTW" is a sparse remnant
    # EchoStar ticker change (2026-06-24, CUSIP unchanged); the vendor backfills the
    # full genuine chain under ECHO (verified against Alpaca SIP bars, diagnostic §8).
    # PIT guard (§9): before 2021-11-23 the ticker ECHO denoted Echo Global Logistics
    # (buyout completed that date; never an S&P 500 member; no genuine series
    # retrievable) -- never attribute pre-rename ECHO-ticker references to EchoStar.
    "SATS": "ECHO",
}

# Manual, *reviewed* vendor-collision quarantine: symbols whose vendor
# resolution is known to return a DIFFERENT instrument (or corrupted data),
# with no genuine series available and no rename successor. Quarantined names
# are excluded from the price pull and the resolved/tradeable universe and are
# counted in the coverage skip-list -- the same measured-survivorship-leak
# treatment as any other unretrievable delisted name. Membership intervals are
# NOT touched (history is evidence; the mask needs it). Evidence per name:
# docs/data_integrity_diagnostic.md §3.
QUARANTINE_TABLE: dict[str, str] = {
    "ACT": "vendor resolves to Enact Holdings (IPO 2021-09-16), never Watson/Actavis; chain membership ended 2020-05-12 under AGN",
    "ADS": "vendor resolves to Adidas AG (Xetra), never Alliance Data; membership ended 2020-06-22",
    "INFO": "dual-feed merge while listed; wrong instrument after the 2022-02 SPGI merger",
    "SBNY": "no genuine bars (failed 2023-03-15); vendor rows from 2024-08 are a different instrument",
    "SOLS": "corrupted splice (459,999x single-day return); genuine segment unrecoverable",
}

# Manual, *reviewed* corrections to the Wikipedia "Selected changes" table:
# real-world membership-relevant events the upstream table omits, applied to
# the parsed changes frame before reconstruction (``apply_changes_patches``)
# so the interval sweep sees them as ordinary add/remove events. Every entry
# is primary-sourced in docs/data_integrity_diagnostic.md (§10) -- never
# patch from memory; a wrong event silently rewrites membership history.
CHANGES_PATCH_TABLE: tuple[dict[str, str], ...] = (
    # Actavis plc -> Allergan plc rename, ticker ACT -> AGN at the 2015-06-15
    # open (MIAX corporate-action alert 2015-06-12; Allergan 8-K). Invisible
    # to the Wikipedia table, which records the seat's 1999-04-12 addition
    # under ACT (backfilled chain ticker) and its 2020-05-12 removal under
    # AGN -- without this event the removal is dropped as inconsistent and
    # the ACT interval never closes (§10). PIT guards, recorded not
    # mechanized: 1999-04-12 -> 2013-01-24 the seat traded as WPI (Watson
    # Pharmaceuticals); vendor ACT from 2021-09-16 is Enact Holdings, never
    # a member (see QUARANTINE_TABLE). Distinct from the legacy Allergan
    # Inc. seat (AGN, HISTORY_FLOOR -> 2015-03-23), which closes upstream.
    {"date": "2015-06-15", "added": "AGN", "removed": "ACT"},
)


# Rows the builder adds to the upstream "Selected changes" table, each applied
# only when the table being read lacks it and only on builds whose as-of is on
# or after its date (``supplement_changes``). Two kinds, both ticker renames:
# rows the baseline revision carried and editors later deleted ("renamings are
# not index changes"), and renames the table never carried. The reconstruction
# needs them either way: without the row, the old ticker's interval never
# closes (or the old ticker vanishes from history) and the new ticker is read
# as a member since HISTORY_FLOOR. Kept apart from ``CHANGES_PATCH_TABLE``
# because ``prism.io.legacy_import`` turns every patch into membership events.
# Never add a row from memory: each cites the revision or filing it comes from.
CHANGES_SUPPLEMENT_TABLE: tuple[dict[str, str], ...] = (
    # Everest Re Group -> Everest Group; in revision 1364118843, deleted 2026-08-23 (revision 1370890563).
    {"date": "2023-07-10", "added": "EG", "removed": "RE"},
    # FleetCor -> Corpay; in revision 1364118843, deleted 2026-09-21 (revision 1376064088).
    {"date": "2024-03-25", "added": "CPAY", "removed": "FLT"},
    # EchoStar SATS -> ECHO; in revision 1364118843, deleted 2026-07-18 (revision 1364845917). See RENAME_TABLE.
    {"date": "2026-06-24", "added": "ECHO", "removed": "SATS"},
    # Equity Residential -> Vivmark Residential after the AvalonBay merger, CIK 906107 unchanged: "beginning on
    # August 18, 2026, the Company Common Shares will trade on the NYSE under the ticker symbol 'VMRK'" (8-K,
    # accession 0001140361-26-033377, filed 2026-08-17). Never in the table; the list article switched the
    # constituent row in revision 1369954750.
    {"date": "2026-08-18", "added": "VMRK", "removed": "EQR"},
)

# MediaWiki endpoints. Articles are read at the last revision on or before the
# build's as-of date (end of day UTC), never live, so a rebuild at a past as-of
# reads what that date's build would have read.
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_INDEX = "https://en.wikipedia.org/w/index.php"

#: The article holding each index's constituents table.
CONSTITUENTS_ARTICLE: dict[str, str] = {
    "sp500": "List of S&P 500 companies",
    "sp400": "List of S&P 400 companies",
}

#: Each index's name as a written universe file's header gives it.
INDEX_NAME: dict[str, str] = {
    "sp500": "S&P 500",
    "sp400": "S&P 400",
}

#: The article holding each index's "Selected changes" table, by the first date
#: it held it. The S&P 500 table moved to its own article on 2026-08-11
#: (revision 1368903080 created it; list-article revision 1368903137 removed it).
CHANGES_ARTICLE: dict[str, tuple[tuple[str, str], ...]] = {
    "sp500": (
        ("1990-01-01", "List of S&P 500 companies"),
        ("2026-08-11", "Historical components of the S&P 500"),
    ),
    "sp400": (("1990-01-01", "List of S&P 400 companies"),),
}

#: The revision every S&P 500 rebuild is checked against: the list article on
#: 2026-07-14, the revision behind ``sp500_membership_2026-07-17``. A rebuild at
#: a later as-of must reconstruct the same history through that date
#: (``history_differences``).
CHANGES_BASELINE_REVID = 1364118843


def normalize_ticker(ticker: str) -> str:
    """Uppercase + strip; drop internal whitespace and stray table pipes. No vendor remap here.

    Wiki markup leaks a ``|`` into a few changes-table cells ("ALLE |", "JCP |"
    in the 2026-08 article); no ticker contains one.
    """
    return re.sub(r"[\s|]+", "", str(ticker).strip().upper())


# --------------------------------------------------------------------------- #
# HTML table extraction (stdlib only -- no lxml/bs4 in the env)
# --------------------------------------------------------------------------- #

_FOOTNOTE = re.compile(r"\[[^\]]*\]")


def _clean_cell(text: str) -> str:
    """Strip footnote markers like ``[1]`` / ``[note 2]`` and collapse spaces."""
    return re.sub(r"\s+", " ", _FOOTNOTE.sub("", text)).strip()


class _TableExtractor(HTMLParser):
    """Collect every ``<table>`` as a list of rows, each a list of cell strings.

    Deliberately flat: it does not model nested tables (the S&P 500 constituents
    and changes tables do not nest tables inside the cells we read). Header and
    data cells (``th``/``td``) are both captured in row order.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(_clean_cell("".join(self._cell)))  # type: ignore[union-attr]
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self._table.append(self._row)  # type: ignore[union-attr]
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def extract_tables(html: str) -> list[list[list[str]]]:
    """Parse raw HTML into a list of tables (rows of cleaned cell strings)."""
    parser = _TableExtractor()
    parser.feed(html)
    return parser.tables


def _header_index(table: list[list[str]], *needles: str) -> int | None:
    """First row index whose cells contain *all* needles (case-insensitive)."""
    wants = [n.lower() for n in needles]
    for i, row in enumerate(table):
        cells = [c.lower() for c in row]
        if all(any(w in c for c in cells) for w in wants):
            return i
    return None


def parse_constituents_table(tables: list[list[list[str]]]) -> list[str]:
    """Tickers from the constituents table (header contains 'Symbol')."""
    for table in tables:
        hi = _header_index(table, "symbol")
        if hi is None:
            continue
        header = [c.lower() for c in table[hi]]
        col = next((j for j, c in enumerate(header) if "symbol" in c), 0)
        out: list[str] = []
        for row in table[hi + 1 :]:
            if col < len(row) and row[col]:
                tok = normalize_ticker(row[col])
                if tok and tok not in out:
                    out.append(tok)
        if out:
            return out
    raise ValueError("No constituents table with a 'Symbol' column was found")


def parse_changes_table(tables: list[list[list[str]]]) -> pd.DataFrame:
    """Parse the 'Selected changes' table into ``[date, added, removed]``.

    Assumes the current Wikipedia layout: a two-row header (``Date | Added |
    Removed | Reason`` over ``Ticker | Security`` sub-columns), and data rows of
    ``[date, added_ticker, added_security, removed_ticker, removed_security,
    reason]``. Rows with an unparseable date are skipped. Column layout is
    pinned here on purpose -- if Wikipedia restructures the table this raises
    rather than silently mis-mapping tickers.
    """
    for table in tables:
        hi = _header_index(table, "added", "removed")
        if hi is None:
            continue
        # Data starts after the (one- or two-row) header block. The sub-header
        # row repeats 'Ticker'/'Security'; skip any leading rows that look like
        # headers (no parseable date in column 0).
        records: list[dict[str, object]] = []
        for row in table[hi + 1 :]:
            if not row:
                continue
            date = pd.to_datetime(row[0], errors="coerce")
            if pd.isna(date):
                continue
            added = normalize_ticker(row[1]) if len(row) > 1 else ""
            removed = normalize_ticker(row[3]) if len(row) > 3 else ""
            records.append({"date": date, "added": added, "removed": removed})
        if records:
            return pd.DataFrame.from_records(records, columns=["date", "added", "removed"])
    raise ValueError("No 'Selected changes' table with Added/Removed columns was found")


def changes_article(index: str, asof: str | pd.Timestamp) -> str:
    """The article that held ``index``'s changes table on ``asof`` (``CHANGES_ARTICLE``)."""
    day = pd.Timestamp(asof).tz_localize(None) if pd.Timestamp(asof).tz is not None else pd.Timestamp(asof)
    held = [title for start, title in CHANGES_ARTICLE[index] if pd.Timestamp(start) <= day]
    if not held:
        raise ValueError(f"no changes article is recorded for {index} on {asof}")
    return held[-1]


def revision_before(payload: dict) -> tuple[int, str]:
    """The (revid, timestamp) of the single revision a ``prop=revisions`` API query returned.

    Same contract as ``prism.scripts.match_delisted_ciks.revision_before``; an
    article with no revision at or before the requested time fails loud (N7).
    """
    try:
        (page,) = payload["query"]["pages"].values()
        (revision,) = page["revisions"]
        return int(revision["revid"]), str(revision["timestamp"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"unexpected Wikipedia revisions payload: {payload!r:.300}") from exc


def fetch_pinned_article(
    title: str, asof: str, cache_dir: Path, *, refresh: bool = False, get=None, timeout: float = 30.0
) -> tuple[str, dict[str, object]]:
    """The article as of its last revision at or before ``asof`` (end of day UTC), cached by revid.

    Two network reads: the revision lookup, then the revision's HTML unless it is
    cached. A revision's content never changes, so a cached file is reused
    across builds; the returned provenance (url, revid, revision timestamp,
    sha256) is what the builder records. ``get`` is ``requests.get`` unless a
    test supplies one.
    """
    import hashlib

    if get is None:
        import requests  # local import: keep module importable without network stack

        get = requests.get
    headers = {"User-Agent": "prism/universe-builder"}
    params = {"action": "query", "prop": "revisions", "titles": title, "rvlimit": 1, "rvdir": "older",
              "rvstart": f"{pd.Timestamp(asof).strftime('%Y-%m-%d')}T23:59:59Z", "rvprop": "ids|timestamp",
              "format": "json"}
    response = get(WIKI_API, params=params, timeout=timeout, headers=headers)
    response.raise_for_status()
    revid, timestamp = revision_before(response.json())
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    path = Path(cache_dir) / f"wikipedia_{slug}_rev{revid}.html"
    url = f"{WIKI_INDEX}?oldid={revid}"
    if refresh or not path.exists():
        body = get(url, timeout=timeout, headers=headers)
        body.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body.text, encoding="utf-8")
    html = path.read_text(encoding="utf-8")
    return html, {"title": title, "url": url, "revid": revid, "revision_timestamp": timestamp,
                  "asof": str(asof), "path": str(path), "sha256": hashlib.sha256(html.encode()).hexdigest()}


def fetch_index_tables(
    index: str, asof: str, cache_dir: Path, *, refresh: bool = False, get=None
) -> tuple[list[str], pd.DataFrame, dict[str, dict[str, object]]]:
    """``(current_members, changes, sources)`` for ``index`` from revisions pinned at ``asof``.

    The constituents come from ``CONSTITUENTS_ARTICLE``; the changes from the
    article ``CHANGES_ARTICLE`` names for that date, which is the same revision
    when both tables share an article.
    """
    html, constituents_meta = fetch_pinned_article(CONSTITUENTS_ARTICLE[index], asof, cache_dir, refresh=refresh, get=get)
    current = parse_constituents_table(extract_tables(html))
    title = changes_article(index, asof)
    if title != CONSTITUENTS_ARTICLE[index]:
        html, changes_meta = fetch_pinned_article(title, asof, cache_dir, refresh=refresh, get=get)
    else:
        changes_meta = constituents_meta
    changes = parse_changes_table(extract_tables(html))
    return current, changes, {"constituents": constituents_meta, "changes": changes_meta}


def _event_keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    added = frame["added"].fillna("").map(normalize_ticker)
    removed = frame["removed"].fillna("").map(normalize_ticker)
    return set(zip(dates, added, removed))


def supplement_changes(
    changes: pd.DataFrame, asof: str | pd.Timestamp, supplement: tuple[dict[str, str], ...] = CHANGES_SUPPLEMENT_TABLE
) -> pd.DataFrame:
    """Append the ``CHANGES_SUPPLEMENT_TABLE`` rows dated on or before ``asof`` that ``changes`` lacks.

    A revision that still carries a row gets no duplicate, and a row dated after
    ``asof`` is never added, so an older build reconstructs exactly as it did.
    Returns a new frame, or the input unchanged when nothing is missing.
    """
    day = pd.Timestamp(asof)
    day = day.tz_localize(None) if day.tz is not None else day
    present = _event_keys(changes)
    missing = [row for row in supplement
               if pd.Timestamp(row["date"]) <= day
               and (row["date"], normalize_ticker(row["added"]), normalize_ticker(row["removed"])) not in present]
    if not missing:
        return changes
    return apply_changes_patches(changes, patches=tuple(missing))


def history_differences(
    membership: pd.DataFrame, baseline: pd.DataFrame, through: str | pd.Timestamp
) -> pd.DataFrame:
    """Intervals on which two reconstructions disagree on or before ``through``.

    Both are clipped at ``through``; ``side`` says which one holds the interval.
    A rebuild must not rewrite history a published build already fixed: a
    non-empty answer means the upstream table changed under it (a deleted row,
    a rename entered only in the constituents table) and is reviewed before the
    build is used.
    """
    cut = pd.Timestamp(through)

    def clipped(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
        starts, ends = pd.to_datetime(frame["start"]), pd.to_datetime(frame["end"])
        keep = starts <= cut
        return {(normalize_ticker(t), s.strftime("%Y-%m-%d"), min(e, cut).strftime("%Y-%m-%d"))
                for t, s, e in zip(frame["ticker"][keep], starts[keep], ends[keep])}

    ours, theirs = clipped(membership), clipped(baseline)
    rows = [(*key, "rebuild") for key in ours - theirs] + [(*key, "baseline") for key in theirs - ours]
    return pd.DataFrame(sorted(rows), columns=["ticker", "start", "end", "side"])


# --------------------------------------------------------------------------- #
# Point-in-time membership reconstruction
# --------------------------------------------------------------------------- #


def apply_changes_patches(
    changes: pd.DataFrame, patches: tuple[dict[str, str], ...] = CHANGES_PATCH_TABLE
) -> pd.DataFrame:
    """Append the reviewed ``CHANGES_PATCH_TABLE`` events to a parsed changes frame.

    Returns a new frame; the input is not mutated. Ordering is free because
    ``reconstruct_membership`` sorts by date (stable, so upstream rows keep
    precedence within a shared date). An empty patch table is the identity.
    """
    if not patches:
        return changes
    patch = pd.DataFrame.from_records(
        [
            {
                "date": pd.Timestamp(p["date"]),
                "added": normalize_ticker(p.get("added", "")),
                "removed": normalize_ticker(p.get("removed", "")),
            }
            for p in patches
        ],
        columns=["date", "added", "removed"],
    )
    return pd.concat([changes, patch], ignore_index=True)


def reconstruct_membership(
    current_members: Iterable[str],
    changes: pd.DataFrame,
    *,
    end_date: str | pd.Timestamp,
    history_floor: pd.Timestamp = HISTORY_FLOOR,
) -> pd.DataFrame:
    """Reconstruct per-ticker membership intervals ``[start, end]`` (inclusive).

    Two passes:
      1. *Reverse* over the changes (newest first) from the current member set to
         find membership just before the earliest recorded change.
      2. *Forward* sweep from that initial set, opening an interval on each
         ``added`` event and closing one on each ``removed`` event; intervals
         still open at ``end_date`` are closed there.

    Re-additions produce multiple disjoint intervals for a ticker, which is
    correct. Inconsistent events (remove a non-member, re-add an active member)
    are skipped rather than corrupting the sweep.

    Returns a tidy frame ``[ticker, start, end]`` sorted by (ticker, start).
    """
    end_date = pd.Timestamp(end_date)
    if end_date.tz is not None:  # align with tz-naive Wikipedia change dates / HISTORY_FLOOR
        end_date = end_date.tz_localize(None)
    current = {normalize_ticker(t) for t in current_members}

    ch = changes.copy()
    ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
    ch = ch.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)
    ch["added"] = ch.get("added", "").fillna("").map(normalize_ticker)
    ch["removed"] = ch.get("removed", "").fillna("").map(normalize_ticker)

    # Pass 1: reverse to the pre-history membership set.
    members = set(current)
    for row in ch.iloc[::-1].itertuples(index=False):
        if row.added:
            members.discard(row.added)
        if row.removed:
            members.add(row.removed)
    initial = set(members)

    # Pass 2: forward sweep into intervals.
    active = set(initial)
    open_start: dict[str, pd.Timestamp] = {t: history_floor for t in initial}
    rows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for row in ch.itertuples(index=False):
        if row.removed and row.removed in active:
            rows.append((row.removed, open_start.pop(row.removed), row.date))
            active.discard(row.removed)
        if row.added and row.added not in active:
            active.add(row.added)
            open_start[row.added] = row.date
    for ticker in active:
        rows.append((ticker, open_start[ticker], end_date))

    out = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    # Drop inverted intervals (start > end): these come from changes dated after
    # ``end_date`` -- announced-but-not-yet-effective additions, which are not
    # members as of the reconstruction horizon.
    out = out[out["start"] <= out["end"]]
    return out.sort_values(["ticker", "start"], kind="stable").reset_index(drop=True)


def ever_members(membership: pd.DataFrame) -> list[str]:
    """Sorted union of every ticker that was ever a member (survivorship-free)."""
    return sorted(set(membership["ticker"].map(normalize_ticker)))


def members_active_between(
    membership: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> list[str]:
    """Ever-members whose membership overlaps ``[start, end]``.

    Survivorship-free *within the window*: includes names removed during it,
    excludes only names that left before it (irrelevant to a backtest over the
    window). This is the right scoping for a price pull -- fetching names delisted
    years before ``start`` adds cost without adding in-sample breadth.
    """
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    lo = lo.tz_localize(None) if lo.tz is not None else lo
    hi = hi.tz_localize(None) if hi.tz is not None else hi
    starts = pd.to_datetime(membership["start"])
    ends = pd.to_datetime(membership["end"])
    if starts.dt.tz is not None:
        starts = starts.dt.tz_localize(None)
    if ends.dt.tz is not None:
        ends = ends.dt.tz_localize(None)
    overlap = (starts <= hi) & (ends >= lo)
    return sorted(set(membership.loc[overlap, "ticker"].map(normalize_ticker)))


def build_membership_mask(
    membership: pd.DataFrame, index: pd.Index, symbols: Iterable[str]
) -> pd.DataFrame:
    """Day x symbol boolean: was ``symbol`` an index member on that day?

    Comparison is on calendar dates (tz dropped) because membership is daily and
    the price index is tz-aware (ET); this sidesteps tz-mismatch errors while
    staying correct to the day.
    """
    idx = pd.DatetimeIndex(index)
    idx_dates = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    cols = [normalize_ticker(s) for s in symbols]
    mask = pd.DataFrame(False, index=index, columns=cols)

    by_ticker = {t: g for t, g in membership.assign(ticker=membership["ticker"].map(normalize_ticker)).groupby("ticker")}
    for ticker in cols:
        grp = by_ticker.get(ticker)
        if grp is None:
            continue
        col = np.zeros(len(idx_dates), dtype=bool)
        for r in grp.itertuples(index=False):
            start = pd.Timestamp(r.start).normalize()
            end = pd.Timestamp(r.end).normalize()
            col |= (idx_dates >= start) & (idx_dates <= end)
        mask[ticker] = col
    return mask


# --------------------------------------------------------------------------- #
# Coverage ledger (the integrity crux of free survivorship-free data)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CoverageReport:
    """Honest accounting of which ever-members have usable price data."""

    asof: str
    n_ever_members: int
    n_resolved: int
    resolved: list[str]
    skipped: list[str]
    rename_table: dict[str, str]
    quarantined: dict[str, str] = field(default_factory=dict)

    @property
    def coverage_fraction(self) -> float:
        return self.n_resolved / self.n_ever_members if self.n_ever_members else float("nan")

    def to_dict(self) -> dict[str, object]:
        return {
            "asof": self.asof,
            "n_ever_members": self.n_ever_members,
            "n_resolved": self.n_resolved,
            "n_skipped": len(self.skipped),
            "coverage_fraction": self.coverage_fraction,
            "resolved": self.resolved,
            "skipped": self.skipped,
            "rename_table": dict(self.rename_table),
            "quarantined": dict(self.quarantined),
        }


def compute_coverage(
    ever: Iterable[str],
    available: Iterable[str],
    *,
    asof: str,
    rename_table: dict[str, str] | None = None,
    quarantine_table: dict[str, str] | None = None,
) -> CoverageReport:
    """Split the ever-member union into price-resolved vs skipped (delisted gaps).

    ``available`` is the set of symbols the price vendor actually returned bars
    for; a ticker resolves if its (renamed) symbol is in that set. The skipped
    list is the *measured* survivorship leak -- names that belong in the
    universe but have no price history. It must be surfaced, never hidden.

    Quarantined names (``QUARANTINE_TABLE``) can never resolve, even when the
    vendor answers: the answer is a known wrong instrument, which is strictly
    worse than no answer. They are counted in ``skipped`` and echoed with
    their reasons in ``quarantined`` so the exclusion is auditable.
    """
    rename_table = rename_table or RENAME_TABLE
    quarantine_table = QUARANTINE_TABLE if quarantine_table is None else quarantine_table
    avail = {normalize_ticker(s) for s in available}
    resolved, skipped, quarantined = [], [], {}
    for raw in ever:
        t = normalize_ticker(raw)
        if t in quarantine_table:
            skipped.append(t)
            quarantined[t] = quarantine_table[t]
            continue
        vendor = normalize_ticker(rename_table.get(t, t))
        (resolved if vendor in avail else skipped).append(t)
    return CoverageReport(
        asof=asof,
        n_ever_members=len(resolved) + len(skipped),
        n_resolved=len(resolved),
        resolved=sorted(resolved),
        skipped=sorted(skipped),
        rename_table=dict(rename_table),
        quarantined=quarantined,
    )


def write_universe_file(
    symbols: Iterable[str], path: str | Path, *, index: str, asof: str, source: str = "wikipedia"
) -> None:
    """Write a window's index members in the ``load_universe`` format.

    A comment header naming ``index`` (a key of ``CONSTITUENTS_ARTICLE``) and
    its caveats, then one symbol per line, sorted and de-duplicated. The list
    is not survivorship-free: the changes tables it is built from are selected
    rather than complete, and a price-resolved list also drops the removed
    names the vendor has no bars for.
    """
    if index not in INDEX_NAME:
        raise ValueError(f"unknown index {index!r}; the universe header names one of {sorted(INDEX_NAME)}")
    syms = sorted({normalize_ticker(s) for s in symbols})
    header = (
        f"# {INDEX_NAME[index]} universe as of {asof}: tickers that were index members at some time in the "
        "build window.\n"
        f"# Source: {source}.\n"
        f"# Membership: Wikipedia '{CONSTITUENTS_ARTICLE[index]}' constituents and the index's changes table.\n"
        "# One symbol per line; '#' starts a comment; blank lines ignored.\n"
        "# Includes names since removed by delisting, acquisition or a change of index.\n"
        "# Not survivorship-free: the changes tables are selected rather than complete, and price\n"
        "# coverage for removed names is partial; the companion coverage JSON records how much.\n"
    )
    Path(path).write_text(header + "\n".join(syms) + "\n")


# --------------------------------------------------------------------------- #
# Unrecorded exits (the "Selected changes" tables are selected, not complete)
# --------------------------------------------------------------------------- #

#: Evidence kinds, in the order the ledger reports them. Each is a dated fact
#: that the ticker was no longer in the reconstructed index from that date.
EXIT_EVIDENCE_KINDS: tuple[str, ...] = ("promoted_to_sp500", "added_to_sp600", "vendor_last_bar")


def resolve_unrecorded_exits(
    membership: pd.DataFrame,
    current_members: Iterable[str],
    *,
    asof: str | pd.Timestamp,
    evidence: dict[str, list[tuple[pd.Timestamp, str]]],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Close the intervals a forward sweep left open for tickers that are not current members.

    ``reconstruct_membership`` closes an interval only on a recorded removal,
    so a ticker whose exit the changes table omits stays a member until the
    reconstruction horizon: a phantom. The S&P 400 page omits many exits
    (promotions to the 500, demotions to the 600, acquisitions, renames). An
    interval open at ``asof`` whose ticker is not in ``current_members`` is
    closed on the day before the earliest dated exit evidence that falls after
    the interval's start; an interval with no such evidence is dropped and
    reported as ``unresolved``. Dropping is a choice with a bias of unknown sign
    (the name was a member for part of that span), which is why every dropped
    interval is in the ledger rather than silently absent.

    ``evidence`` maps a normalized ticker to ``(date, kind)`` pairs, each a fact
    that the ticker had left the index by ``date``; ``kind`` is one of
    ``EXIT_EVIDENCE_KINDS``. Returns ``(resolved_membership, ledger)``.
    """
    horizon = pd.Timestamp(asof)
    horizon = horizon.tz_localize(None) if horizon.tz is not None else horizon
    current = {normalize_ticker(t) for t in current_members}
    frame = membership.copy()
    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    frame["start"] = pd.to_datetime(frame["start"])
    frame["end"] = pd.to_datetime(frame["end"])
    keep: list[bool] = []
    ends: list[pd.Timestamp] = []
    ledger: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        phantom = row.end >= horizon and row.ticker not in current
        if not phantom:
            keep.append(True)
            ends.append(row.end)
            continue
        dated = sorted(
            (pd.Timestamp(date), str(kind))
            for date, kind in evidence.get(row.ticker, [])
            if pd.Timestamp(date) > row.start and pd.Timestamp(date) <= horizon
        )
        if dated:
            exit_date, kind = dated[0]
            closed = exit_date - pd.Timedelta(days=1)
            keep.append(True)
            ends.append(closed)
            ledger.append(
                {"ticker": row.ticker, "start": row.start.date().isoformat(), "closed_at": closed.date().isoformat(),
                 "evidence": kind, "evidence_date": exit_date.date().isoformat()}
            )
        else:
            keep.append(False)
            ends.append(row.end)
            ledger.append(
                {"ticker": row.ticker, "start": row.start.date().isoformat(), "closed_at": None,
                 "evidence": "unresolved", "evidence_date": None}
            )
    frame["end"] = ends
    resolved = frame.loc[keep].reset_index(drop=True)
    resolved = resolved[resolved["start"] <= resolved["end"]]
    return resolved.sort_values(["ticker", "start"], kind="stable").reset_index(drop=True), ledger


# --------------------------------------------------------------------------- #
# Interval algebra across indices (S&P 500 / 400 / 600 are mutually exclusive)
# --------------------------------------------------------------------------- #


def _day_intervals(frame: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    out: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for row in frame.itertuples(index=False):
        start = pd.Timestamp(row.start).normalize()
        end = pd.Timestamp(row.end).normalize()
        start = start.tz_localize(None) if start.tz is not None else start
        end = end.tz_localize(None) if end.tz is not None else end
        if start <= end:
            out.setdefault(normalize_ticker(row.ticker), []).append((start, end))
    return out


def _merged(spans: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Union of closed day intervals; touching intervals (end + 1 day == start) merge."""
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + pd.Timedelta(days=1):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _frame(intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]) -> pd.DataFrame:
    rows = [(ticker, start, end) for ticker, spans in intervals.items() for start, end in spans]
    frame = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    return frame.sort_values(["ticker", "start"], kind="stable").reset_index(drop=True)


def subtract_membership(minuend: pd.DataFrame, subtrahend: pd.DataFrame) -> pd.DataFrame:
    """``minuend`` membership with every day the same ticker holds in ``subtrahend`` removed.

    The S&P 500, 400 and 600 are mutually exclusive by construction (the S&P
    Composite 1500), so a day on which a ticker is an S&P 500 member is not a
    day it is an S&P 400 member, whatever an incomplete changes table implies.
    """
    left = _day_intervals(minuend)
    right = {ticker: _merged(spans) for ticker, spans in _day_intervals(subtrahend).items()}
    out: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for ticker, spans in left.items():
        pieces = _merged(spans)
        for cut_start, cut_end in right.get(ticker, []):
            nxt: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            for start, end in pieces:
                if cut_end < start or cut_start > end:
                    nxt.append((start, end))
                    continue
                if start < cut_start:
                    nxt.append((start, cut_start - pd.Timedelta(days=1)))
                if end > cut_end:
                    nxt.append((cut_end + pd.Timedelta(days=1), end))
            pieces = nxt
        if pieces:
            out[ticker] = pieces
    return _frame(out)


def union_membership(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker union of several memberships, overlapping and touching intervals merged."""
    combined: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for frame in frames:
        for ticker, spans in _day_intervals(frame).items():
            combined.setdefault(ticker, []).extend(spans)
    return _frame({ticker: _merged(spans) for ticker, spans in combined.items()})
