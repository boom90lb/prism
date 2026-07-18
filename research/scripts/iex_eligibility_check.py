"""IEX-vs-consolidated eligibility wedge on the live universe (M6 pre-flight).

The live loop screens tradeability on Alpaca IEX-feed volume (``DEFAULT_FEED``,
``src/prism/live/alpaca_data.py``) while the certified backtest screened the
same rule on consolidated Twelve Data volume; the rule itself is
``compute_eligibility``'s trailing 20-bar median dollar volume >= $1M
(``src/prism/residual/factors.py``). IEX prints only a few percent of the tape,
so the identical $1M floor is a materially higher *effective* consolidated
floor live — measured 2026-07-17 at a ~5.1% median per-name share, i.e. an
effective consolidated-equivalent floor near $20M, excluding exactly one
S&P 500 name (ACT). Whether that wedge moves the decile book is an empirical
question this instrument answers on demand, before every M6-family read.

Offline on the consolidated side (reads the bar caches), online on the live
side (one ``fetch_batch`` over the fetch universe — file ∪ held book, the same
set the live loop fetches). Report-only: writes one JSON and prints a summary;
no universe file, cache, or state is touched. Credentials are the ``APCA_*``
env the loop already uses and travel only in request headers.

Per name it reports the screen's volume leg on both feeds — apples-to-apples
at the cache-end date and on the latest IEX bar (what tonight's run sees) —
plus the per-name median IEX/consolidated volume share on aligned sessions.
Frame and results: ``docs/program_review_2026-07.md`` §2 (corrections ledger)
and ``MARKETS.md``; evidence JSON beside the other diagnostics in
``results/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from research.scripts.data_integrity_sweep import (
    SCREEN_FLOOR_DOLLARS,
    SCREEN_WINDOW_BARS,
    bar_dates,
)

# Sessions of IEX/consolidated overlap required before a per-name volume-share
# median is reported (below this the ratio is noise, not a share estimate).
MIN_SHARE_OVERLAP_DAYS = 10

# Names above the floor but under this multiple of it are reported as flap
# risk: a routine volume dip can toggle their eligibility night to night.
NEAR_FLOOR_MULTIPLE = 2.0


def universe_symbols(text: str) -> list[str]:
    """Symbols from a universe file's text: one per line, ``#`` comments stripped."""
    out: list[str] = []
    for line in text.splitlines():
        sym = line.split("#", 1)[0].strip()
        if sym:
            out.append(sym)
    return out


def median_dollar_volume(bars: pd.DataFrame, window: int = SCREEN_WINDOW_BARS) -> pd.Series:
    """The eligibility screen's volume leg on one OHLCV frame, either feed.

    Exactly ``compute_eligibility``'s arithmetic — non-negative volume times
    close, ``rolling(window, min_periods=window).median()`` — on a naive
    normalized date index (duplicate dates deduped keep-last, the sweep's
    treatment). The NaN warm-up head is dropped, so ``.iloc[-1]`` is the value
    the screen compares to the floor on the series' last bar.
    """
    dates = bar_dates(bars.index)
    close = pd.to_numeric(bars["close"], errors="coerce")
    volume = pd.to_numeric(bars["volume"], errors="coerce").clip(lower=0.0)
    dollar = (close * volume).groupby(dates).last()
    return dollar.rolling(window, min_periods=window).median().dropna()


def volume_share(
    iex_bars: pd.DataFrame, cons_bars: pd.DataFrame, min_overlap: int = MIN_SHARE_OVERLAP_DAYS
) -> float:
    """Per-name median IEX/consolidated share volume ratio on aligned sessions.

    Sessions where the consolidated print is zero or either feed is missing are
    excluded; fewer than ``min_overlap`` usable sessions returns NaN rather
    than a ratio estimated from noise.
    """
    iex = pd.to_numeric(iex_bars["volume"], errors="coerce").groupby(bar_dates(iex_bars.index)).last()
    cons = pd.to_numeric(cons_bars["volume"], errors="coerce").groupby(bar_dates(cons_bars.index)).last()
    both = pd.concat([iex.rename("iex"), cons.rename("cons")], axis=1).dropna()
    both = both[both["cons"] > 0]
    if len(both) < min_overlap:
        return float("nan")
    return float((both["iex"] / both["cons"]).median())


def screen_comparison(
    cons_med: dict[str, pd.Series],
    iex_med: dict[str, pd.Series],
    floor: float = SCREEN_FLOOR_DOLLARS,
    near_multiple: float = NEAR_FLOOR_MULTIPLE,
) -> dict:
    """Both feeds' screen verdicts per name, plus the mask-difference summary.

    Apples-to-apples masks are evaluated at each name's last consolidated date
    (the IEX median is read at-or-before that date, tolerating a one-session
    feed lag); the live screen is the latest IEX bar. The symmetric difference
    of the as-of fail sets is the wedge's decile-book bite.
    """
    per_name: dict[str, dict] = {}
    for sym in sorted(set(cons_med) | set(iex_med)):
        entry: dict = {}
        cons = cons_med.get(sym)
        iex = iex_med.get(sym)
        if cons is not None and len(cons):
            asof = cons.index[-1]
            entry["asof"] = str(asof.date())
            entry["cons_med_asof"] = float(cons.iloc[-1])
            if iex is not None and len(iex):
                at = iex.loc[:asof]
                if len(at):
                    entry["iex_med_asof"] = float(at.iloc[-1])
        if iex is not None and len(iex):
            entry["iex_med_last"] = float(iex.iloc[-1])
            entry["iex_last_date"] = str(iex.index[-1].date())
        per_name[sym] = entry

    both = [s for s, e in per_name.items() if "cons_med_asof" in e and "iex_med_asof" in e]
    fail_cons_asof = sorted(s for s in both if per_name[s]["cons_med_asof"] < floor)
    fail_iex_asof = sorted(s for s in both if per_name[s]["iex_med_asof"] < floor)
    latest = {s: e["iex_med_last"] for s, e in per_name.items() if "iex_med_last" in e}
    summary = {
        "n_names": len(per_name),
        "n_with_both_feeds_asof": len(both),
        "fail_cons_asof": fail_cons_asof,
        "fail_iex_asof": fail_iex_asof,
        "mask_symmetric_difference_asof": sorted(set(fail_cons_asof) ^ set(fail_iex_asof)),
        "fail_iex_last": sorted(s for s, v in latest.items() if v < floor),
        "near_floor_iex_last": sorted(s for s, v in latest.items() if floor <= v < near_multiple * floor),
    }
    return {"per_name": per_name, "summary": summary}


def share_stats(shares: dict[str, float], floor: float = SCREEN_FLOOR_DOLLARS) -> dict:
    """Distribution of per-name IEX volume shares and the implied effective floor.

    ``implied_consolidated_floor_at_median`` is ``floor / median_share`` — the
    consolidated dollar volume a typical name needs so that its *IEX* slice
    still clears the live screen's floor.
    """
    ser = pd.Series(shares, dtype=float).dropna()
    if ser.empty:
        return {"n": 0}
    return {
        "n": int(len(ser)),
        "median": float(ser.median()),
        "p05": float(ser.quantile(0.05)),
        "p25": float(ser.quantile(0.25)),
        "p75": float(ser.quantile(0.75)),
        "p95": float(ser.quantile(0.95)),
        "min": float(ser.min()),
        "max": float(ser.max()),
        "implied_consolidated_floor_at_median": float(floor / ser.median()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", default="data/universe/sp500_current.txt", help="Live universe file")
    parser.add_argument(
        "--state",
        default="runs/paper_loop_momentum2/state.json",
        help="Live loop state.json; held positions are unioned into the fetch set (missing file = no extras)",
    )
    parser.add_argument("--data_dir", default="data", help="Consolidated bar-cache directory")
    parser.add_argument("--cache_suffix", default="_1d_2020-01-01_2026-06-16.parquet", help="Cache filename suffix")
    parser.add_argument(
        "--start",
        default=str(date.today() - timedelta(days=180)),
        help="IEX fetch start (enough history for the rolling median plus share overlap)",
    )
    parser.add_argument("--feed", default="iex", help="Alpaca feed to screen (the live loop's default feed)")
    parser.add_argument("--floor", type=float, default=SCREEN_FLOOR_DOLLARS)
    parser.add_argument("--window", type=int, default=SCREEN_WINDOW_BARS)
    parser.add_argument("--out", default=None, help="Output JSON (default: results/iex_eligibility_<today>.json)")
    return parser.parse_args()


def main() -> None:
    # Imported here so the pure helpers stay importable without the live extras.
    from prism.live.alpaca_data import AlpacaBarSource

    args = parse_args()
    file_syms = universe_symbols(Path(args.universe).read_text())
    state_path = Path(args.state)
    held = sorted(json.loads(state_path.read_text()).get("positions", {})) if state_path.exists() else []
    symbols = sorted(set(file_syms) | set(held))

    cons_bars: dict[str, pd.DataFrame] = {}
    missing_cache: list[str] = []
    for sym in symbols:
        path = Path(args.data_dir) / f"{sym}{args.cache_suffix}"
        if path.exists():
            cons_bars[sym] = pd.read_parquet(path)
        else:
            missing_cache.append(sym)

    source = AlpacaBarSource.from_env(feed=args.feed)
    frames = source.fetch_batch(symbols, "1d", start_date=args.start)
    iex_bars = {s: f for s, f in frames.items() if s in set(symbols) and not f.empty}
    empty_iex = sorted(set(symbols) - set(iex_bars))

    cons_med = {s: m for s, m in ((s, median_dollar_volume(b, args.window)) for s, b in cons_bars.items()) if len(m)}
    # A cache can exist yet be unscreenable (a fresh rename ticker with fewer
    # bars than the rolling window) — report that explicitly, not via absence.
    unscreenable = sorted(set(cons_bars) - set(cons_med))
    iex_med = {s: m for s, m in ((s, median_dollar_volume(b, args.window)) for s, b in iex_bars.items()) if len(m)}
    shares = {s: volume_share(iex_bars[s], cons_bars[s]) for s in iex_bars if s in cons_bars}

    comparison = screen_comparison(cons_med, iex_med, floor=args.floor)
    held_set = set(held)
    for sym, entry in comparison["per_name"].items():
        entry["held"] = sym in held_set
        share = shares.get(sym)
        if share is not None and share == share:  # NaN-safe
            entry["share_median"] = share
    comparison["summary"]["held_failing_iex_last"] = sorted(
        set(comparison["summary"]["fail_iex_last"]) & held_set
    )

    payload = {
        "meta": {
            "generated": date.today().isoformat(),
            "universe_file": args.universe,
            "n_universe_file": len(file_syms),
            "n_held": len(held),
            "n_fetch_set": len(symbols),
            "feed": args.feed,
            "floor_dollars": args.floor,
            "window_bars": args.window,
            "iex_start": args.start,
            "cache_suffix": args.cache_suffix,
            "missing_cache": missing_cache,
            "cache_without_screenable_series": unscreenable,
            "empty_iex": empty_iex,
        },
        "share_stats": share_stats(shares, floor=args.floor),
        "summary": comparison["summary"],
        "per_name": comparison["per_name"],
    }
    out_path = Path(args.out) if args.out else Path("results") / f"iex_eligibility_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    brief = {k: payload[k] for k in ("meta", "share_stats", "summary")}
    print(json.dumps(brief, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
