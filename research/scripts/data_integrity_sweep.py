"""Vendor symbol-collision and cache-hygiene sweep over the bar caches (SPEC §5 / I-7).

Two 2026-07-13 findings motivate this. ``data/ADS_1d_*.parquet`` holds Adidas
AG (Xetra) over its entire range — a vendor symbol collision, never the S&P
constituent Alliance Data — and ``data/INFO_1d_*.parquet`` carries 105
duplicated dates plus bars four years past IHS Markit's 2022 delisting. Both
symbols sit inside the PIT universe, so the contamination class has to be
enumerated once and cleared against every finished book, not patched
name-by-name as instances surface.

Offline and deterministic (uncounted diagnostic: searches nothing, changes no
trial code path). Every bar cache is scored on

* duplicated dates (vendor series-merge smell),
* bars printed on NYSE full-day closures — a US-listed series has none, while
  a foreign-venue collision like Xetra trades straight through them,
* session coverage against the expected NYSE calendar,
* volume scale (median share / dollar volume: an S&P constituent prints
  millions of shares, a foreign local line prints thousands), and
* whether the series could ever clear the trials' $1M / 20-bar-median
  dollar-volume screen, computed from the cache's own numbers — exactly what
  the backtest's eligibility screen saw.

Suspect caches are then cross-referenced against each run directory's
``target_weights.csv`` (was the name ever actually held, and what did it
contribute to the certified return stream?) and the PIT membership intervals
(was it ever allowed in?). Writes one JSON, reads everything else. Frame and
results: ``docs/data_integrity_diagnostic.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# NYSE full-day closures 2020-01-01 .. 2026-12-31, hardcoded because the core
# dependency set deliberately carries no exchange-calendar package. Weekend
# holidays appear as their observed weekday (or not at all, per NYSE rules);
# 2025-01-09 is the National Day of Mourning closure. The sweep self-audits
# this table: ``closure_bar_counts`` in the output counts how many caches
# print a bar on each date — a near-universal count would mean the table is
# wrong on that date, a handful means those caches follow a foreign calendar.
NYSE_FULL_CLOSURES: frozenset[str] = frozenset(
    {
        "2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25",
        "2020-07-03", "2020-09-07", "2020-11-26", "2020-12-25",
        "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31",
        "2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24",
        "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20",
        "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26",
        "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
        "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
        "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
        "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18",
        "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27",
        "2025-12-25",
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
        "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    }
)

# Flag thresholds, pinned here and restated in the diagnostic doc.
OFF_CALENDAR_MIN_BARS = 3  # >= 3 closure-date bars = foreign calendar; tolerates stray vendor rows
THIN_MEDIAN_SHARE_VOLUME = 50_000.0  # S&P constituents print millions of shares; foreign local lines, thousands
TRUNCATED_BEFORE = "2026-05-15"  # primary caches run to 2026-06-15; an early stop deserves eyes, not a verdict
GAPPY_MAX_MISSING = 15  # missing >15 NYSE sessions = sparse/spliced series (halts this long are themselves news)
SCREEN_FLOOR_DOLLARS = 1_000_000.0  # ResidualStatArbConfig.min_median_dollar_volume default (residual/factors.py)
SCREEN_WINDOW_BARS = 20  # ResidualStatArbConfig.dollar_volume_window default


def bar_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Cache index → naive normalized dates (caches store tz-aware ET midnight stamps)."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return idx.normalize()


@lru_cache(maxsize=None)
def expected_sessions(first: str, last: str) -> int:
    """NYSE sessions in ``[first, last]``: weekdays minus full-day closures."""
    bdays = pd.bdate_range(first, last)
    return int((~pd.Index(bdays.strftime("%Y-%m-%d")).isin(NYSE_FULL_CLOSURES)).sum())


def cache_metrics(bars: pd.DataFrame) -> dict:
    """Hygiene metrics for one bar cache (columns ``open/high/low/close/volume``)."""
    dates = bar_dates(bars.index)
    date_strs = pd.Index(dates.strftime("%Y-%m-%d"))
    unique = pd.DatetimeIndex(dates.unique())
    unique_strs = pd.Index(unique.strftime("%Y-%m-%d"))
    on_calendar = int((~unique_strs.isin(NYSE_FULL_CLOSURES) & (unique.dayofweek < 5)).sum())
    first, last = dates.min(), dates.max()
    close = pd.to_numeric(bars["close"], errors="coerce")
    volume = pd.to_numeric(bars["volume"], errors="coerce")
    dollar = close * volume
    # Duplicate dates are deduped keep-last before any return/median arithmetic,
    # matching the EDGE panel treatment of the INFO cache.
    dedup_close = close.groupby(dates).last()
    dv20 = dollar.groupby(dates).last().rolling(SCREEN_WINDOW_BARS, min_periods=SCREEN_WINDOW_BARS).median()
    rets = dedup_close.pct_change()
    return {
        "n_rows": int(len(bars)),
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "duplicate_dates": int(dates.duplicated().sum()),
        "closure_bars": sorted(set(date_strs) & NYSE_FULL_CLOSURES),
        "n_closure_bars": len(set(date_strs) & NYSE_FULL_CLOSURES),
        "expected_sessions": expected_sessions(str(first.date()), str(last.date())),
        "missing_sessions": int(max(expected_sessions(str(first.date()), str(last.date())) - on_calendar, 0)),
        "median_share_volume": float(volume.median()),
        "median_dollar_volume": float(dollar.median()),
        "max_dv20_median": float(dv20.max()) if dv20.notna().any() else float("nan"),
        "max_abs_daily_return": float(rets.abs().max()) if rets.notna().any() else float("nan"),
    }


def classify(metrics: dict, min_last_date: str = TRUNCATED_BEFORE) -> list[str]:
    """Flag list for one cache; any flag makes the cache a suspect."""
    flags = []
    if metrics["duplicate_dates"] > 0:
        flags.append("duplicate_dates")
    if metrics["n_closure_bars"] >= OFF_CALENDAR_MIN_BARS:
        flags.append("off_calendar_bars")
    if metrics["median_share_volume"] < THIN_MEDIAN_SHARE_VOLUME:
        flags.append("thin_volume")
    if metrics["last_date"] < min_last_date:
        flags.append("truncated_series")
    if metrics["missing_sessions"] > GAPPY_MAX_MISSING:
        flags.append("gappy_series")
    return flags


def run_exposure(run_dir: Path, suspect_bars: dict[str, pd.DataFrame]) -> tuple[dict, pd.Timestamp, pd.Timestamp]:
    """Held-book exposure of suspect names in one finished run.

    ``target_weights.csv`` is the ground truth for what the run's book held.
    The contribution below uses the suspect's own cache closes — the same
    series the trial's panel was built from — so a nonzero read here is a
    direct contamination of the run's certified return stream, not a proxy.
    """
    weights = pd.read_csv(run_dir / "target_weights.csv", index_col=0)
    widx = pd.DatetimeIndex(pd.to_datetime(weights.index, utc=True)).tz_convert("America/New_York")
    weights.index = widx.tz_localize(None).normalize()
    out: dict[str, dict] = {}
    for sym, bars in suspect_bars.items():
        if sym not in weights.columns:
            out[sym] = {"in_universe": False}
            continue
        w = weights[sym].astype(float)
        nz = w[w != 0.0]
        entry: dict = {"in_universe": True, "days_held": int(len(nz)), "max_abs_weight": float(w.abs().max())}
        if len(nz):
            closes = pd.to_numeric(bars["close"], errors="coerce").groupby(bar_dates(bars.index)).last()
            rets = closes.pct_change().reindex(weights.index)
            contrib = (w.shift(1) * rets).dropna()
            entry.update(
                {
                    "first_held": str(nz.index.min().date()),
                    "last_held": str(nz.index.max().date()),
                    "return_contribution_total": float(contrib.sum()),
                    "return_contribution_days": int((contrib != 0.0).sum()),
                }
            )
        out[sym] = entry
    return out, weights.index.min(), weights.index.max()


def membership_intervals(membership: pd.DataFrame, symbols: list[str]) -> dict[str, list[dict]]:
    """PIT membership intervals ``[{start, end}]`` for ``symbols`` (empty list = never a member)."""
    out: dict[str, list[dict]] = {s: [] for s in symbols}
    for r in membership[membership["ticker"].isin(symbols)].itertuples():
        out[r.ticker].append({"start": str(pd.Timestamp(r.start).date()), "end": str(pd.Timestamp(r.end).date())})
    return out


def screen_max_within(
    bars: pd.DataFrame, intervals: list[dict], window_lo: pd.Timestamp, window_hi: pd.Timestamp
) -> float:
    """Max 20-bar-median dollar volume inside membership ∩ traded window.

    An upper bound on eligibility: if this never reaches the screen floor, the
    name could not have entered any book during the window even before the
    full-history and price screens are considered.
    """
    dates = bar_dates(bars.index)
    dollar = pd.to_numeric(bars["close"], errors="coerce") * pd.to_numeric(bars["volume"], errors="coerce")
    dv20 = dollar.groupby(dates).last().rolling(SCREEN_WINDOW_BARS, min_periods=SCREEN_WINDOW_BARS).median()
    best: list[float] = []
    for iv in intervals:
        lo = max(pd.Timestamp(iv["start"]), window_lo)
        hi = min(pd.Timestamp(iv["end"]), window_hi)
        if lo > hi:
            continue
        seg = dv20.loc[(dv20.index >= lo) & (dv20.index <= hi)].dropna()
        if len(seg):
            best.append(float(seg.max()))
    return max(best) if best else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data", help="Bar-cache directory")
    parser.add_argument("--pattern", default="*_1d_2020-01-01_2026-06-16.parquet", help="Cache filename glob")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run dirs to cross-reference (default: every results/* dir holding a target_weights.csv)",
    )
    parser.add_argument("--membership", default="data/universe/sp500_membership_2026-06-16.parquet")
    parser.add_argument("--out", default=None, help="Output JSON (default: results/data_integrity_sweep_<today>.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    caches = sorted(data_dir.glob(args.pattern))
    if not caches:
        raise SystemExit(f"No caches match {args.pattern!r} under {data_dir}")

    per_cache: dict[str, dict] = {}
    suspect_bars: dict[str, pd.DataFrame] = {}
    closure_counts: dict[str, int] = {}
    for path in caches:
        symbol = path.name.split("_1d_")[0]
        bars = pd.read_parquet(path)
        metrics = cache_metrics(bars)
        metrics["flags"] = classify(metrics)
        per_cache[symbol] = metrics
        for d in metrics["closure_bars"]:
            closure_counts[d] = closure_counts.get(d, 0) + 1
        if metrics["flags"]:
            suspect_bars[symbol] = bars

    run_dirs = (
        [Path(r) for r in args.runs]
        if args.runs
        else sorted(p.parent for p in Path("results").glob("*/target_weights.csv"))
    )
    exposure: dict[str, dict] = {}
    window_lo: pd.Timestamp | None = None
    window_hi: pd.Timestamp | None = None
    for run_dir in run_dirs:
        exp, lo, hi = run_exposure(run_dir, suspect_bars)
        exposure[str(run_dir)] = exp
        window_lo = lo if window_lo is None or lo < window_lo else window_lo
        window_hi = hi if window_hi is None or hi > window_hi else window_hi

    membership = pd.read_parquet(args.membership)
    intervals = membership_intervals(membership, sorted(suspect_bars))
    suspects: dict[str, dict] = {}
    for sym in sorted(suspect_bars):
        smax = (
            screen_max_within(suspect_bars[sym], intervals[sym], window_lo, window_hi)
            if intervals[sym] and window_lo is not None
            else float("nan")
        )
        suspects[sym] = {
            "flags": per_cache[sym]["flags"],
            "membership_intervals": intervals[sym],
            "held_in_any_run": any(exposure[r].get(sym, {}).get("days_held", 0) > 0 for r in exposure),
            "max_screen_dv20_in_membership_and_traded_window": smax,
            "screen_passable_in_window": bool(np.isfinite(smax) and smax >= SCREEN_FLOOR_DOLLARS),
        }

    payload = {
        "meta": {
            "generated": date.today().isoformat(),
            "data_dir": str(data_dir),
            "pattern": args.pattern,
            "n_caches": len(caches),
            "n_suspects": len(suspect_bars),
            "thresholds": {
                "off_calendar_min_bars": OFF_CALENDAR_MIN_BARS,
                "thin_median_share_volume": THIN_MEDIAN_SHARE_VOLUME,
                "truncated_before": TRUNCATED_BEFORE,
                "screen_floor_dollars": SCREEN_FLOOR_DOLLARS,
                "screen_window_bars": SCREEN_WINDOW_BARS,
            },
            "runs": [str(r) for r in run_dirs],
            "traded_window": (
                [str(window_lo.date()), str(window_hi.date())] if window_lo is not None else None
            ),
            "membership_file": args.membership,
        },
        "closure_bar_counts": dict(sorted(closure_counts.items())),
        "suspects": suspects,
        "exposure": exposure,
        "caches": per_cache,
    }
    out_path = (
        Path(args.out) if args.out else Path("results") / f"data_integrity_sweep_{date.today().isoformat()}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=True))
    brief = {k: payload[k] for k in ("meta", "closure_bar_counts", "suspects", "exposure")}
    print(json.dumps(brief, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
