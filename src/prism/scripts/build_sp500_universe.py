"""Build a survivorship-bias-free S&P 500 universe (Koker method).

Reconstructs point-in-time membership from the Wikipedia "List of S&P 500
companies" page (current constituents + 'Selected changes'), then writes five
artifacts under ``data/universe/``:

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

Vendor symbol collisions are handled by two reviewed tables in
``prism.io.universe_sp500``: ``RENAME_TABLE`` (old symbol -> successor) and
``QUARANTINE_TABLE`` (symbols whose vendor resolution is a known wrong
instrument; never fetched, counted as coverage skips).

Network: fetching Wikipedia and pulling Twelvedata prices needs egress; run with
the sandbox disabled, or pass ``--html-file`` for an offline membership-only
build (membership reconstruction needs no network) and ``--skip-prices`` to
defer the price pull and coverage ledger.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from prism.config import DATA_DIR
from prism.io.loader import DataLoader
from prism.io.universe_sp500 import (
    QUARANTINE_TABLE,
    RENAME_TABLE,
    WIKI_URL,
    compute_coverage,
    ever_members,
    extract_tables,
    fetch_sp500_wikipedia,
    members_active_between,
    normalize_ticker,
    parse_changes_table,
    parse_constituents_table,
    reconstruct_membership,
    write_universe_file,
)
from prism.logging_utils import configure_logging, get_symbol_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a survivorship-bias-free S&P 500 universe.")
    p.add_argument("--asof", type=str, default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
                   help="As-of date; names the artifacts and closes still-open membership intervals.")
    p.add_argument("--start_date", type=str, default="1990-01-01", help="Price-history start for the pull.")
    p.add_argument("--end_date", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=str(DATA_DIR / "universe"))
    p.add_argument("--wiki_url", type=str, default=WIKI_URL)
    p.add_argument("--html_file", type=str, default=None,
                   help="Read the Wikipedia HTML from a local file instead of the network "
                        "(reproducible / offline membership build).")
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


def _write_current_members(membership: pd.DataFrame, asof: str, path: Path) -> None:
    """Current members (intervals open at ``asof``) that have a local bar cache.

    The live loop's universe file: a fetchable-today proxy, NOT survivorship-free
    (backtests use the pit files). Quarantined symbols are excluded even when a
    stale cache lingers on disk.
    """
    current = members_active_between(membership, asof, asof)
    cached = {p.name.split("_1d_")[0] for p in DATA_DIR.glob("*_1d_*.parquet")}
    names = [s for s in current if s in cached and s not in QUARANTINE_TABLE]
    header = (
        f"# Current S&P 500 members as of {asof} with local price data (fetchable\n"
        "# proxy for the live loop; NOT survivorship-free -- backtests use sp500_pit_*).\n"
        f"# Built from sp500_membership_{asof}.parquet (intervals open at as-of)\n"
        f"# INTERSECT data/*_1d_*.parquet, minus quarantined symbols. {len(names)} names.\n"
    )
    path.write_text(header + "\n".join(names) + "\n")


def _load_tables(args: argparse.Namespace) -> tuple[list[str], pd.DataFrame]:
    if args.html_file:
        tables = extract_tables(Path(args.html_file).read_text())
        return parse_constituents_table(tables), parse_changes_table(tables)
    return fetch_sp500_wikipedia(args.wiki_url)


def _pull_prices(
    symbols: list[str], start: str, end: str | None, *, rate_per_min: float = 7.0, consec_empty_warn: int = 15
) -> list[str]:
    """Fetch each symbol (paced under the API rate limit); return those with bars.

    Pacing keeps calls under the vendor's per-minute cap so a rate-limited call
    never poses as a delisted-name gap in the coverage ledger. Delisting gaps are
    scattered, so a long run of *consecutive* empties signals throttling/quota
    exhaustion rather than genuine gaps -- that is flagged loudly so the coverage
    number is not silently undercounted.
    """
    loader = DataLoader()
    available: list[str] = []
    min_interval = 60.0 / max(rate_per_min, 0.1)
    last = 0.0
    consec_empty = 0
    for i, symbol in enumerate(symbols):
        log = get_symbol_logger(logger, symbol)
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
    args = parse_args()
    configure_logging(verbose=args.verbose)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current, changes = _load_tables(args)
    membership = reconstruct_membership(current, changes, end_date=args.asof)
    # Remap to vendor symbols so the universe file, membership mask, and price
    # panel all share one symbology (rename table is empty until coverage gaps
    # are reviewed -- identity for now).
    membership = membership.assign(ticker=membership["ticker"].map(_to_vendor))
    ever = ever_members(membership)
    # Scope the tradeable universe to names active during the backtest window
    # (still survivorship-free within it -- includes names removed during the
    # window); the full membership parquet keeps every interval for the mask.
    window = members_active_between(membership, args.start_date, args.asof)

    universe_path = out_dir / f"sp500_pit_{args.asof}.txt"
    membership_path = out_dir / f"sp500_membership_{args.asof}.parquet"
    write_universe_file(window, universe_path, asof=args.asof, source="wikipedia")
    membership.to_parquet(membership_path)
    logger.info(
        "Wrote %d window-active members (of %d ever-members; %d intervals) -> %s",
        len(window), len(ever), len(membership), universe_path,
    )

    result: dict[str, object] = {
        "asof": args.asof,
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
        available = _pull_prices(fetchable, args.start_date, args.end_date, rate_per_min=args.rate_per_min)
        coverage = compute_coverage(pull, available, asof=args.asof)
        coverage_path = out_dir / f"sp500_coverage_{args.asof}.json"
        coverage_path.write_text(json.dumps(coverage.to_dict(), indent=2))
        resolved_path = out_dir / f"sp500_pit_resolved_{args.asof}.txt"
        write_universe_file(coverage.resolved, resolved_path, asof=args.asof, source="wikipedia+twelvedata_resolved")
        current_path = out_dir / "sp500_current.txt"
        _write_current_members(membership, args.asof, current_path)
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
