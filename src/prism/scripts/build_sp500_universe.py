"""Build a survivorship-bias-free S&P 500 universe (Koker method).

Reconstructs point-in-time membership from Wikipedia (the "List of S&P 500
companies" constituents table and the "Selected changes" table, which moved to
"Historical components of the S&P 500" on 2026-08-11), read at the last
revision on or before ``--asof``, never the live page. Writes six artifacts
under ``data/universe/``:

* ``sp500_pit_<asof>.txt``        -- the ever-member union, in the ``load_universe``
                                     format (feeds ``--universe``).
* ``sp500_membership_<asof>.parquet`` -- per-ticker membership intervals
                                     ``[ticker, start, end]`` (feeds ``--membership``).
* ``sp500_coverage_<asof>.json``  -- the coverage ledger: which ever-members have
                                     usable price data vs the skipped (delisted,
                                     unretrievable, or quarantined) names. The
                                     skip-list is the *measured* survivorship
                                     leak -- never hidden.
* ``sp500_pit_resolved_<asof>.txt`` -- the price-resolved subset of the pit union
                                     (the backtest universe; feeds ``--universe_file``).
* ``sp500_current.txt``           -- current members with a local bar cache (the
                                     live loop's fetchable-today proxy; undated
                                     because the live wrappers reference it by name).
* ``sp500_sources_<asof>.json``   -- the Wikipedia revisions read (revid, timestamp,
                                     sha256), the supplement rows applied, and the history check.

Vendor symbol collisions are handled by three reviewed tables in
``prism.io.universe_sp500``: ``RENAME_TABLE`` (old symbol -> successor),
``QUARANTINE_TABLE`` (symbols whose vendor resolution is a known wrong
instrument; never fetched, counted as coverage skips), and
``CHANGES_PATCH_TABLE`` (membership-relevant events the Wikipedia changes
table omits -- e.g. index-invisible ticker renames -- applied before
reconstruction so intervals open and close under the right symbols). A fourth,
``CHANGES_SUPPLEMENT_TABLE``, adds the ticker renames the table lacks: rows
editors deleted after 2026-07-14 and renames never entered. An S&P 500 build at
or after that baseline refuses to write while its history through the
baseline differs from the baseline revision's own reconstruction
(``history_differences``).

Network: fetching Wikipedia and pulling Twelvedata prices needs egress; run with
the sandbox disabled, or pass ``--html-file`` for an offline membership-only
build (membership reconstruction needs no network) and ``--skip-prices`` to
defer the price pull and coverage ledger.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from prism.io.loader import API_KEY_ENV_VAR, DataLoader
from prism.io.universe_sp500 import (
    CHANGES_BASELINE_REVID,
    CONSTITUENTS_ARTICLE,
    QUARANTINE_TABLE,
    RENAME_TABLE,
    apply_changes_patches,
    compute_coverage,
    ever_members,
    extract_tables,
    fetch_index_tables,
    fetch_pinned_article,
    history_differences,
    members_active_between,
    normalize_ticker,
    parse_changes_table,
    parse_constituents_table,
    reconstruct_membership,
    supplement_changes,
    write_universe_file,
)
from prism.logging_utils import configure_logging, get_symbol_logger
from prism.scripts import announce_alias
from prism.workspace import Workspace, load_env, resolve_workspace

logger = logging.getLogger(__name__)

#: The indices this builder reconstructs (the articles each reads are
#: ``CONSTITUENTS_ARTICLE`` and ``CHANGES_ARTICLE``).
INDICES: tuple[str, ...] = tuple(CONSTITUENTS_ARTICLE)

#: The as-of date whose pinned list-article revision is ``CHANGES_BASELINE_REVID``.
#: S&P 500 builds at or after it are checked against that revision's changes table.
CHANGES_BASELINE_ASOF = "2026-07-17"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a survivorship-bias-free S&P 500 universe.")
    p.add_argument("--asof", type=str, default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
                   help="As-of date; names the artifacts and closes still-open membership intervals.")
    p.add_argument("--start_date", type=str, default="1990-01-01", help="Price-history start for the pull.")
    p.add_argument("--end_date", type=str, default=None)
    p.add_argument("--workspace", type=str, default=None,
                   help="Writable root (default: $PRISM_WORKSPACE, else the current directory). "
                        "The price cache is {workspace}/data and the logs are {workspace}/logs.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory for the universe artifacts (default: {workspace}/data/universe).")
    p.add_argument("--index", type=str, default="sp500", choices=sorted(INDICES),
                   help="Which index to reconstruct; every artifact is prefixed with it.")
    p.add_argument("--html_file", type=str, default=None,
                   help="Read the constituents (and, without --changes_html_file, the changes) table "
                        "from a local file instead of the pinned revision (offline membership build).")
    p.add_argument("--changes_html_file", type=str, default=None,
                   help="Read the changes table from this local file (with --html_file).")
    p.add_argument("--baseline_html_file", type=str, default=None,
                   help="The baseline revision's HTML for the history check, instead of fetching it.")
    p.add_argument("--refresh_wikipedia", action="store_true",
                   help="Re-download pinned revisions even when cached.")
    p.add_argument("--skip_prices", action="store_true",
                   help="Build membership + universe file only; skip the price pull and coverage ledger.")
    p.add_argument("--max_names", type=int, default=None, help="Cap the price pull to the first N names (testing).")
    p.add_argument("--rate_per_min", type=float, default=7.0,
                   help="Max price-fetch calls/min (Twelvedata free tier is 8/min). Staying under the "
                        "limit means an empty result is a genuine data gap, not a throttled call. Raise on a paid tier.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _to_vendor(ticker: str) -> str:
    """Canonical (Wikipedia) ticker -> price-vendor symbol via the reviewed rename table."""
    t = normalize_ticker(ticker)
    return normalize_ticker(RENAME_TABLE.get(t, t))


def _write_current_members(
    membership: pd.DataFrame, asof: str, path: Path, data_dir: Path, *, index: str = "sp500"
) -> None:
    """Current members (intervals open at ``asof``) that have a local bar cache.

    The live loop's universe file: a fetchable-today proxy, NOT survivorship-free
    (backtests use the pit files). Quarantined symbols are excluded even when a
    stale cache lingers on disk. ``data_dir`` is the workspace's bar cache --
    the same directory the pull below writes into.
    """
    current = members_active_between(membership, asof, asof)
    cached = {p.name.split("_1d_")[0] for p in data_dir.glob("*_1d_*.parquet")}
    names = [s for s in current if s in cached and s not in QUARANTINE_TABLE]
    header = (
        f"# Current {index} members as of {asof} with local price data (fetchable\n"
        f"# proxy for the live loop; NOT survivorship-free -- backtests use {index}_pit_*).\n"
        f"# Built from {index}_membership_{asof}.parquet (intervals open at as-of)\n"
        f"# INTERSECT data/*_1d_*.parquet, minus quarantined symbols. {len(names)} names.\n"
    )
    path.write_text(header + "\n".join(names) + "\n")


def _load_tables(args: argparse.Namespace, cache_dir: Path) -> tuple[list[str], pd.DataFrame, dict[str, object]]:
    """Constituents, changes and their provenance: local files when given, else revisions pinned at ``--asof``."""
    if args.html_file:
        tables = extract_tables(Path(args.html_file).read_text())
        changes_tables = extract_tables(Path(args.changes_html_file).read_text()) if args.changes_html_file else tables
        sources: dict[str, object] = {"constituents": {"path": args.html_file},
                                      "changes": {"path": args.changes_html_file or args.html_file}}
        return parse_constituents_table(tables), parse_changes_table(changes_tables), sources
    current, changes, pinned = fetch_index_tables(args.index, args.asof, cache_dir, refresh=args.refresh_wikipedia)
    return current, changes, dict(pinned)


def _baseline_membership(args: argparse.Namespace, cache_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Revision ``CHANGES_BASELINE_REVID`` reconstructed at its as-of; a different revision there fails loud."""
    if args.baseline_html_file:
        html = Path(args.baseline_html_file).read_text()
        meta: dict[str, object] = {"path": args.baseline_html_file}
    else:
        html, meta = fetch_pinned_article(CONSTITUENTS_ARTICLE["sp500"], CHANGES_BASELINE_ASOF, cache_dir,
                                          refresh=args.refresh_wikipedia)
        if meta["revid"] != CHANGES_BASELINE_REVID:
            raise SystemExit(f"baseline revision at {CHANGES_BASELINE_ASOF} is {meta['revid']}, "
                             f"not {CHANGES_BASELINE_REVID}: the article's history changed (N7)")
    tables = extract_tables(html)
    changes = apply_changes_patches(supplement_changes(parse_changes_table(tables), CHANGES_BASELINE_ASOF))
    membership = reconstruct_membership(parse_constituents_table(tables), changes, end_date=CHANGES_BASELINE_ASOF)
    return membership, meta


def _pull_prices(
    symbols: list[str],
    start: str,
    end: str | None,
    *,
    loader: DataLoader,
    rate_per_min: float = 7.0,
    consec_empty_warn: int = 15,
) -> list[str]:
    """Fetch each symbol (paced under the API rate limit); return those with bars.

    Pacing keeps calls under the vendor's per-minute cap so a rate-limited call
    never poses as a delisted-name gap in the coverage ledger. Cache hits make
    no API call and are exempt (``DataLoader.has_cached``, the same covering-
    range lookup the fetch itself performs) -- without the exemption a
    mostly-cached regeneration pays the full interval per name and runs ~90
    minutes instead of minutes. Delisting gaps are scattered, so a long run of
    *consecutive* empties signals throttling/quota exhaustion rather than
    genuine gaps -- that is flagged loudly so the coverage number is not
    silently undercounted.
    """
    available: list[str] = []
    min_interval = 60.0 / max(rate_per_min, 0.1)
    last = 0.0
    consec_empty = 0
    for i, symbol in enumerate(symbols):
        log = get_symbol_logger(logger, symbol)
        if not loader.has_cached(symbol, "1d", start, end):
            wait = min_interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
        df = loader.fetch_historical_data(symbol, "1d", start, end)
        if df.empty or not {"open", "close", "volume"} <= set(df.columns):
            consec_empty += 1
            log.warning("no usable bars; counted as a coverage skip (consecutive empties=%d)", consec_empty)
            if consec_empty == consec_empty_warn:
                logger.warning(
                    "%d consecutive empties -- likely rate-limit/quota, not delisting. Coverage may be "
                    "undercounted; lower --rate_per_min or resume later (cache makes resumption cheap).",
                    consec_empty,
                )
            continue
        consec_empty = 0
        available.append(symbol)
        if (i + 1) % 25 == 0:
            logger.info("pull progress: %d/%d fetched, %d resolved", i + 1, len(symbols), len(available))
    return available


def main() -> None:
    announce_alias("prism-build-universe", "prism data universe")
    args = parse_args()
    # Entry-point duties, in order: the workspace is created before the first
    # write, and ``.env`` is loaded before the vendor key is read (spec §3.1).
    workspace: Workspace = resolve_workspace(args.workspace).ensure()
    load_env()
    configure_logging(workspace.logs, verbose=args.verbose)
    out_dir = Path(args.out_dir) if args.out_dir else workspace.data / "universe"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = args.index
    wiki_cache = workspace.data / "wikipedia"
    current, changes, sources = _load_tables(args, wiki_cache)
    history_check: dict[str, object] = {"checked": False}
    n_upstream = len(changes)
    if index == "sp500":
        changes = supplement_changes(changes, args.asof)
    tail = changes.iloc[n_upstream:]
    supplemented = [{"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "added": a, "removed": r}
                    for d, a, r in zip(tail["date"], tail["added"], tail["removed"])]
    if supplemented:
        logger.info("added %d supplement row(s) the upstream table lacks: %s", len(supplemented), supplemented)
    n_raw = len(changes)
    # The reviewed changes patches are S&P 500 events (the ACT -> AGN seat);
    # applying them to another index would inject a membership event it never
    # had. The rename and quarantine tables are vendor-symbology facts and
    # apply to every index.
    changes = apply_changes_patches(changes) if index == "sp500" else apply_changes_patches(changes, patches=())
    if len(changes) != n_raw:
        logger.info("applied %d reviewed changes-table patch event(s)", len(changes) - n_raw)
    membership = reconstruct_membership(current, changes, end_date=args.asof)
    if index == "sp500" and pd.Timestamp(args.asof) >= pd.Timestamp(CHANGES_BASELINE_ASOF):
        baseline, baseline_meta = _baseline_membership(args, wiki_cache)
        differences = history_differences(membership, baseline, CHANGES_BASELINE_ASOF)
        history_check = {"checked": True, "through": CHANGES_BASELINE_ASOF, "baseline": baseline_meta,
                         "differences": differences.to_dict("records")}
        if not differences.empty:
            raise SystemExit(
                f"the rebuild's membership through {CHANGES_BASELINE_ASOF} differs from revision "
                f"{CHANGES_BASELINE_REVID}'s reconstruction on {len(differences)} interval(s):\n"
                f"{differences.to_string(index=False)}\nThe upstream table changed under the build (a deleted row, "
                "or a rename entered only in the constituents table). Find the event's primary source and add it "
                "to CHANGES_SUPPLEMENT_TABLE, or show the change is a correction (N7). Nothing was written."
            )
    # Remap to vendor symbols so the universe file, membership mask, and price
    # panel all share one symbology (rename table is empty until coverage gaps
    # are reviewed -- identity for now).
    membership = membership.assign(ticker=membership["ticker"].map(_to_vendor))
    ever = ever_members(membership)
    # Scope the tradeable universe to names active during the backtest window
    # (still survivorship-free within it -- includes names removed during the
    # window); the full membership parquet keeps every interval for the mask.
    window = members_active_between(membership, args.start_date, args.asof)

    universe_path = out_dir / f"{index}_pit_{args.asof}.txt"
    membership_path = out_dir / f"{index}_membership_{args.asof}.parquet"
    write_universe_file(window, universe_path, index=index, asof=args.asof, source="wikipedia")
    membership.to_parquet(membership_path)
    logger.info(
        "Wrote %d window-active members (of %d ever-members; %d intervals) -> %s",
        len(window), len(ever), len(membership), universe_path,
    )

    sources_path = out_dir / f"{index}_sources_{args.asof}.json"
    sources_path.write_text(json.dumps({"index": index, "asof": args.asof, "wikipedia": sources,
                                        "supplement_rows": supplemented, "history_check": history_check},
                                       indent=2, default=str) + "\n")

    result: dict[str, object] = {
        "index": index,
        "asof": args.asof,
        "sources_file": str(sources_path),
        "n_current_members": len(current),
        "n_changes": int(len(changes)),
        "n_ever_members": len(ever),
        "n_window_members": len(window),
        "n_intervals": int(len(membership)),
        "universe_file": str(universe_path),
        "membership_file": str(membership_path),
    }

    if not args.skip_prices:
        pull = window if args.max_names is None else window[: args.max_names]
        # Quarantined names are never fetched: the vendor's answer is a known
        # wrong instrument, and pulling it would re-create the bad cache the
        # quarantine exists to keep out (docs/data_integrity_diagnostic.md §6).
        fetchable = [s for s in pull if normalize_ticker(s) not in QUARANTINE_TABLE]
        loader = DataLoader(api_key=os.environ.get(API_KEY_ENV_VAR), cache_dir=workspace.data)
        available = _pull_prices(
            fetchable, args.start_date, args.end_date, loader=loader, rate_per_min=args.rate_per_min
        )
        coverage = compute_coverage(pull, available, asof=args.asof)
        coverage_path = out_dir / f"{index}_coverage_{args.asof}.json"
        coverage_path.write_text(json.dumps(coverage.to_dict(), indent=2))
        resolved_path = out_dir / f"{index}_pit_resolved_{args.asof}.txt"
        write_universe_file(
            coverage.resolved, resolved_path, index=index, asof=args.asof, source="wikipedia+twelvedata_resolved"
        )
        current_path = out_dir / f"{index}_current.txt"
        _write_current_members(membership, args.asof, current_path, workspace.data, index=index)
        result["coverage_file"] = str(coverage_path)
        result["coverage_fraction"] = coverage.coverage_fraction
        result["n_resolved"] = coverage.n_resolved
        result["n_skipped"] = len(coverage.skipped)
        result["n_quarantined"] = len(coverage.quarantined)
        result["resolved_file"] = str(resolved_path)
        result["current_file"] = str(current_path)
        logger.info(
            "Price coverage %.1f%% (%d/%d); %d delisted/unretrievable names recorded as skips (%d quarantined)",
            100 * coverage.coverage_fraction, coverage.n_resolved, len(pull), len(coverage.skipped),
            len(coverage.quarantined),
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
