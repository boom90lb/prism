"""Bar-vendor divergence between Alpaca IEX daily closes and the spine bar caches.

The live paper loop decides and marks on Alpaca IEX daily bars
(``prism.live.alpaca_data``, ``DEFAULT_FEED='iex'``), while certification,
backtests, and the replay concordance all price off the spine-vendor parquet
caches (``data/*_1d_*.parquet``). ``docs/replay_concordance_diagnostic.md``
established CONCORDANT with the *same* prices on both sides by construction;
its residual-wedge list names venue fill rates and the IEX-volume eligibility
screen but not bar-vendor divergence. Decile membership and NAV marks are
rank-sensitive, so a close-level disagreement between the two vendors is a
wedge that instrument is structurally blind to. This script measures it:

- **Close-diff panel** over the current live universe
  (``data/universe/sp500_current.txt``) on the common sessions of a requested
  window: per-name diff distribution in bps, panel tail fractions.
- **Rank impact**: at each month-end decision bar in the window, the 12-1
  momentum score (``close[t-skip]/close[t-lookback] - 1``, the
  ``MomentumSignalNode`` convention) is computed from both vendors' closes at
  the same two endpoint sessions, and top/bottom-decile membership
  (``floor(n * decile)`` per leg, stable sort — the ``_decile_row``
  convention) is compared. Names lacking an endpoint on either side are
  excluded loudly, never silently (N7).
- **Decile-boundary sensitivity**: spine endpoint closes are perturbed by
  draws from the measured empirical diff distribution and leg flips are
  recounted (seeded Monte Carlo) — the fallback read where direct IEX history
  is thin, and a robustness band around the point measurement.
- **NAV-mark impact**: the held live book (``runs/.../state.json``) marked on
  IEX vs spine closes session by session; per-book difference in bps of NAV.

Known hazard, handled explicitly: the spine caches are frozen at their fetch
date while Alpaca serves a *current* split-adjusted series. A split between
the cache freeze and today rebases the whole Alpaca history for that name, so
a large, level-stable ratio is an adjustment-basis artifact, not price
divergence. Such names are flagged (``|median diff| > threshold``), reported
separately, and excluded from the distribution stats — loudly.

Uncounted diagnostic: searches nothing, changes no trial code path, writes one
JSON. Requires ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` in the
environment (source ``.env`` first); the fetch fails loud, never degrades.
Frame and results: ``docs/bar_vendor_divergence.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.live.alpaca_data import AlpacaBarSource
from research.scripts.data_integrity_sweep import bar_dates

DEFAULT_UNIVERSE = "data/universe/sp500_current.txt"
DEFAULT_CACHE_SUFFIX = "_1d_2020-01-01_2026-06-16.parquet"
DEFAULT_STATE_JSON = "runs/paper_loop_momentum2/state.json"

# Fraction of universe names Alpaca may return zero bars for before the run
# aborts: a few dead/renamed tickers are a finding, a large fraction is a
# feed/auth failure (the fetch_universe_panels rationale, prism/live/daily.py).
MAX_MISSING_FRACTION = 0.2

# |median diff| above this is a level shift (split after the cache freeze,
# adjustment-basis mismatch), not per-session price divergence.
ADJUSTMENT_FLAG_BPS = 250.0

TAIL_THRESHOLDS_BPS = (5.0, 25.0)


class CountingSession:
    """Requests-compatible session wrapper that counts HTTP calls."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.n_requests = 0

    def request(self, *args: Any, **kwargs: Any) -> Any:
        self.n_requests += 1
        return self.inner.request(*args, **kwargs)


# ---------------------------------------------------------------- panel build


def parse_universe(text: str) -> list[str]:
    """Universe-file text -> symbols (comment and blank lines skipped)."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def load_spine_closes(
    symbols: list[str], data_dir: Path, cache_suffix: str
) -> tuple[pd.DataFrame, list[str]]:
    """Per-name spine closes on naive normalized dates, plus names with no cache."""
    out: dict[str, pd.Series] = {}
    missing: list[str] = []
    for sym in symbols:
        path = data_dir / f"{sym}{cache_suffix}"
        if not path.exists():
            missing.append(sym)
            continue
        bars = pd.read_parquet(path)
        close = pd.to_numeric(bars["close"], errors="coerce").groupby(bar_dates(bars.index)).last()
        out[sym] = close
    return pd.DataFrame(out).sort_index(), missing


def iex_close_panel(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """``fetch_batch`` output -> wide close panel on naive dates, plus empty names."""
    out: dict[str, pd.Series] = {}
    empty: list[str] = []
    for sym, bars in frames.items():
        if bars.empty:
            empty.append(sym)
            continue
        close = pd.to_numeric(bars["close"], errors="coerce").groupby(bar_dates(bars.index)).last()
        out[sym] = close
    return pd.DataFrame(out).sort_index(), sorted(empty)


def fetch_iex_closes(
    source: Any, symbols: list[str], start: str, end: str | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """IEX daily close panel for ``symbols`` via a batch-capable bar source."""
    frames = source.fetch_batch(symbols, "1d", start_date=start, end_date=end)
    return iex_close_panel(frames)


def align_panels(
    spine: pd.DataFrame, iex: pd.DataFrame, start: str, end: str | None = None
) -> tuple[pd.DataFrame, dict]:
    """Diff panel in bps ((iex/spine - 1) * 1e4) on common sessions x common names.

    Coverage anomalies — names or sessions present on one side only, and
    name-days NaN on one side of a common session — are counted and returned,
    never dropped silently (N7).
    """
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end else max(spine.index.max(), iex.index.max())
    spine_w = spine.loc[(spine.index >= lo) & (spine.index <= hi)].dropna(how="all")
    iex_w = iex.loc[(iex.index >= lo) & (iex.index <= hi)].dropna(how="all")

    common_names = sorted(set(spine_w.columns) & set(iex_w.columns))
    sessions = spine_w.index.intersection(iex_w.index)
    if len(sessions) == 0 or not common_names:
        raise SystemExit(
            f"no overlap between spine and IEX panels in [{start}, {end}]: "
            f"{len(sessions)} common sessions, {len(common_names)} common names"
        )
    s = spine_w.loc[sessions, common_names]
    x = iex_w.loc[sessions, common_names]
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (x / s.where(s > 0.0) - 1.0) * 1e4

    coverage = {
        "window_effective": [str(sessions.min().date()), str(sessions.max().date())],
        "n_common_sessions": int(len(sessions)),
        "n_common_names": len(common_names),
        "names_spine_only": sorted(set(spine_w.columns) - set(iex_w.columns)),
        "names_iex_only": sorted(set(iex_w.columns) - set(spine_w.columns)),
        "sessions_spine_only": [str(d.date()) for d in spine_w.index.difference(iex_w.index)],
        "sessions_iex_only": [str(d.date()) for d in iex_w.index.difference(spine_w.index)],
        "name_days_nan_spine": int(s.isna().to_numpy().sum()),
        "name_days_nan_iex": int((x.isna() & s.notna()).to_numpy().sum()),
        "names_partial_iex": {
            sym: int(n)
            for sym, n in (x.isna() & s.notna()).sum(axis=0).items()
            if int(n) > 0
        },
        # Common names whose diff column is entirely NaN (zero sessions where
        # BOTH vendors price them): they would otherwise vanish from every
        # distribution silently (N7).
        "names_no_valid_diff_days": [sym for sym in common_names if diff[sym].isna().all()],
    }
    return diff, coverage


# ------------------------------------------------------------------ diff stats


def per_name_diff_stats(diff: pd.DataFrame) -> dict[str, dict]:
    """Per-name signed mean/median and abs p95/max of the bps diff panel."""
    out: dict[str, dict] = {}
    for sym in diff.columns:
        vals = diff[sym].dropna()
        if vals.empty:
            continue
        absvals = vals.abs()
        out[sym] = {
            "mean_bps": float(vals.mean()),
            "median_bps": float(vals.median()),
            "p95_abs_bps": float(absvals.quantile(0.95)),
            "max_abs_bps": float(absvals.max()),
            "n_days": int(len(vals)),
        }
    return out


def flag_adjustment_mismatch(
    name_stats: dict[str, dict], level_bps: float = ADJUSTMENT_FLAG_BPS
) -> list[dict]:
    """Names whose |median diff| exceeds ``level_bps`` — adjustment-basis suspects."""
    return [
        {"symbol": sym, **stats}
        for sym, stats in sorted(name_stats.items())
        if abs(stats["median_bps"]) > level_bps
    ]


def panel_diff_stats(
    diff: pd.DataFrame,
    thresholds_bps: tuple[float, ...] = TAIL_THRESHOLDS_BPS,
    exclude: list[str] | None = None,
) -> dict:
    """Pooled name-day distribution of the bps diffs, tail fractions included."""
    kept = diff.drop(columns=list(exclude or []), errors="ignore")
    vals = kept.to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise SystemExit("diff panel has zero finite name-days; nothing to summarize")
    absvals = np.abs(vals)
    per_name_max = kept.abs().max(axis=0).dropna().sort_values(ascending=False)
    return {
        "n_name_days": int(vals.size),
        "n_names": int(kept.notna().any(axis=0).sum()),
        "mean_bps": float(vals.mean()),
        "mean_abs_bps": float(absvals.mean()),
        "median_abs_bps": float(np.median(absvals)),
        "p95_abs_bps": float(np.quantile(absvals, 0.95)),
        "p99_abs_bps": float(np.quantile(absvals, 0.99)),
        "max_abs_bps": float(absvals.max()),
        "frac_exact_equal": float((absvals == 0.0).mean()),
        **{
            f"frac_abs_gt_{t:g}bps": float((absvals > t).mean())
            for t in thresholds_bps
        },
        "worst_names_by_max_abs": [
            {"symbol": sym, "max_abs_bps": float(v)} for sym, v in per_name_max.head(10).items()
        ],
    }


# ----------------------------------------------------------------- rank impact


def month_end_decision_bars(
    calendar: pd.DatetimeIndex, start: str, end: str | None = None
) -> list[pd.Timestamp]:
    """Last session of each calendar month within ``[start, end]``.

    A trailing candidate that is merely the panel's final session mid-month
    (the cache freeze truncating the month, not a genuine month boundary) is
    dropped: it would masquerade as a decision bar that never happened.
    """
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end else calendar.max()
    window = calendar[(calendar >= lo) & (calendar <= hi)]
    bars = [group.max() for _, group in window.to_series().groupby([window.year, window.month])]
    bars = sorted(bars)
    if bars and bars[-1] == calendar.max():
        month_last_bday = pd.offsets.BMonthEnd().rollforward(bars[-1])
        if month_last_bday != bars[-1]:
            bars = bars[:-1]
    return bars


def score_endpoints(
    calendar: pd.DatetimeIndex, decision_bar: pd.Timestamp, skip: int = 21, lookback: int = 252
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The two sessions whose closes form the 12-1 score at ``decision_bar``."""
    pos = calendar.get_indexer([decision_bar])[0]
    if pos < 0:
        raise SystemExit(f"decision bar {decision_bar.date()} is not on the spine calendar")
    if pos - lookback < 0:
        raise SystemExit(
            f"decision bar {decision_bar.date()} has only {pos} prior sessions; "
            f"needs {lookback} for the lookback endpoint"
        )
    return calendar[pos - skip], calendar[pos - lookback]


def endpoint_scores(
    closes: pd.DataFrame, skip_date: pd.Timestamp, look_date: pd.Timestamp
) -> pd.Series:
    """``close[skip]/close[look] - 1`` per name; NaN where an endpoint is absent."""
    n = closes.shape[1]
    recent = (
        closes.loc[skip_date]
        if skip_date in closes.index
        else pd.Series(np.nan, index=closes.columns)
    )
    look = (
        closes.loc[look_date]
        if look_date in closes.index
        else pd.Series(np.nan, index=closes.columns)
    )
    assert len(recent) == len(look) == n
    with np.errstate(invalid="ignore", divide="ignore"):
        return (recent / look.where(look > 0.0) - 1.0).astype(float)


def decile_legs(scores: pd.Series, decile: float = 0.10) -> tuple[frozenset, frozenset, int]:
    """Top/bottom-decile membership under the ``_decile_row`` convention.

    ``floor(n_finite * decile)`` names per side, stable sort over the
    symbol-sorted cross-section so ties break identically on both vendors'
    score vectors.
    """
    ordered = scores.reindex(sorted(scores.index))
    finite = ordered.dropna()
    n_dec = int(len(finite) * decile)
    if n_dec < 1:
        return frozenset(), frozenset(), 0
    order = np.argsort(finite.to_numpy(), kind="stable")
    short = frozenset(finite.index[order[:n_dec]])
    long = frozenset(finite.index[order[-n_dec:]])
    return long, short, n_dec


def leg_flip_report(
    spine_scores: pd.Series, iex_scores: pd.Series, decile: float = 0.10
) -> dict:
    """Decile-membership flips when the same cross-section is ranked per vendor.

    Restricted to names finite on BOTH sides so leg sizes match; one-sided
    names are counted (and listed) as excluded rather than dropped silently.
    """
    both = sorted(set(spine_scores.dropna().index) & set(iex_scores.dropna().index))
    excluded_spine_only = sorted(set(spine_scores.dropna().index) - set(both))
    excluded_iex_only = sorted(set(iex_scores.dropna().index) - set(both))
    s = spine_scores.reindex(both)
    x = iex_scores.reindex(both)
    s_long, s_short, n_dec = decile_legs(s, decile)
    x_long, x_short, n_dec_x = decile_legs(x, decile)
    assert n_dec == n_dec_x, "leg sizes must match on a shared cross-section"
    score_diff_bps = ((x - s).abs() * 1e4).astype(float)
    return {
        "n_common": len(both),
        "n_dec_per_leg": n_dec,
        "long_flips": len(s_long ^ x_long) // 2,
        "short_flips": len(s_short ^ x_short) // 2,
        "long_out": sorted(s_long - x_long),
        "long_in": sorted(x_long - s_long),
        "short_out": sorted(s_short - x_short),
        "short_in": sorted(x_short - s_short),
        "spearman": float(s.corr(x, method="spearman")),
        "max_abs_score_diff_bps": float(score_diff_bps.max()),
        "median_abs_score_diff_bps": float(score_diff_bps.median()),
        "n_excluded_spine_score_only": len(excluded_spine_only),
        "n_excluded_iex_score_only": len(excluded_iex_only),
        "excluded_spine_score_only": excluded_spine_only[:20],
        "excluded_iex_score_only": excluded_iex_only[:20],
    }


def perturbed_flip_stats(
    skip_close: pd.Series,
    look_close: pd.Series,
    diff_sample_bps: np.ndarray,
    *,
    decile: float = 0.10,
    n_draws: int = 200,
    seed: int = 0,
) -> dict:
    """Decile-boundary sensitivity: flips under empirical close-noise resampling.

    Each draw perturbs both spine endpoint closes multiplicatively by bps
    values resampled (iid per name per endpoint) from the measured diff
    distribution, re-scores, and counts leg flips against the unperturbed
    legs. This is the honest fallback read where direct IEX endpoint history
    is missing, and a robustness band around the point comparison elsewhere.
    """
    base_scores = (skip_close / look_close.where(look_close > 0.0) - 1.0).dropna()
    names = sorted(base_scores.index)
    base_long, base_short, n_dec = decile_legs(base_scores.reindex(names), decile)
    if n_dec < 1 or diff_sample_bps.size == 0:
        raise SystemExit("perturbation needs a non-empty cross-section and diff sample")
    rng = np.random.default_rng(seed)
    skip_v = skip_close.reindex(names).to_numpy()
    look_v = look_close.reindex(names).to_numpy()
    flips = np.zeros(n_draws, dtype=int)
    for k in range(n_draws):
        noise = rng.choice(diff_sample_bps, size=(2, len(names)), replace=True) / 1e4
        scores = pd.Series(
            (skip_v * (1.0 + noise[0])) / (look_v * (1.0 + noise[1])) - 1.0, index=names
        )
        p_long, p_short, _ = decile_legs(scores, decile)
        flips[k] = len(base_long ^ p_long) // 2 + len(base_short ^ p_short) // 2
    return {
        "n_draws": int(n_draws),
        "seed": int(seed),
        "n_names": len(names),
        "n_dec_per_leg": int(n_dec),
        "diff_sample_size": int(diff_sample_bps.size),
        "mean_flips": float(flips.mean()),
        "median_flips": float(np.median(flips)),
        "p95_flips": float(np.quantile(flips, 0.95)),
        "max_flips": int(flips.max()),
    }


# -------------------------------------------------------------------- NAV mark


def nav_mark_report(
    positions: dict[str, float],
    cash: float,
    spine: pd.DataFrame,
    iex: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> dict:
    """Held book marked on IEX vs spine closes, session by session, in NAV bps.

    Names the book holds but a vendor cannot price on a session are excluded
    from BOTH marks that session (the diff stays a like-for-like partial-book
    read) and counted; a name never covered is reported by symbol (N7).
    """
    held = sorted(positions)
    shares = pd.Series({s: float(q) for s, q in positions.items()})
    rows: list[dict] = []
    never_covered = set(held)
    for day in sessions:
        s_close = spine.reindex(columns=held).loc[day] if day in spine.index else None
        x_close = iex.reindex(columns=held).loc[day] if day in iex.index else None
        if s_close is None or x_close is None:
            continue
        covered = s_close.notna() & x_close.notna()
        names = covered[covered].index
        never_covered -= set(names)
        nav_s = cash + float((shares[names] * s_close[names]).sum())
        nav_x = cash + float((shares[names] * x_close[names]).sum())
        rows.append(
            {
                "date": str(day.date()),
                "nav_diff_bps": (nav_x / nav_s - 1.0) * 1e4,
                "n_covered": int(covered.sum()),
                "n_excluded": int(len(held) - covered.sum()),
            }
        )
    if not rows:
        raise SystemExit("NAV mark: no session could be marked on both vendors")
    diffs = np.array([abs(r["nav_diff_bps"]) for r in rows])
    return {
        "n_names_held": len(held),
        "cash": float(cash),
        "n_sessions_marked": len(rows),
        "median_abs_nav_diff_bps": float(np.median(diffs)),
        "mean_abs_nav_diff_bps": float(diffs.mean()),
        "max_abs_nav_diff_bps": float(diffs.max()),
        "min_names_covered": int(min(r["n_covered"] for r in rows)),
        "names_never_covered": sorted(never_covered),
        "per_session": rows,
    }


# ------------------------------------------------------------------------ main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--universe_file", default=DEFAULT_UNIVERSE, help="Live universe file")
    parser.add_argument("--data_dir", default="data", help="Spine bar-cache directory")
    parser.add_argument("--cache_suffix", default=DEFAULT_CACHE_SUFFIX, help="Cache filename suffix")
    parser.add_argument("--start", default="2026-01-02", help="Diff-window start (inclusive)")
    parser.add_argument(
        "--end", default=None, help="Diff-window end (default: latest common session)"
    )
    parser.add_argument(
        "--iex_start",
        default="2025-01-02",
        help="IEX fetch start — early enough that the 252-bar lookback endpoint of "
        "every in-window decision bar lands inside the fetched history",
    )
    parser.add_argument("--skip", type=int, default=21, help="Momentum skip bars (B1: 21)")
    parser.add_argument("--lookback", type=int, default=252, help="Momentum lookback bars (B1: 252)")
    parser.add_argument("--decile", type=float, default=0.10, help="Per-leg decile fraction")
    parser.add_argument("--state_json", default=DEFAULT_STATE_JSON, help="Live book state.json")
    parser.add_argument("--n_draws", type=int, default=200, help="Perturbation Monte Carlo draws")
    parser.add_argument("--seed", type=int, default=20260719, help="Perturbation RNG seed")
    parser.add_argument(
        "--iex_cache",
        default=None,
        help="Optional parquet path: reuse a previously fetched IEX close panel "
        "if it exists, else fetch and write it (keeps reruns off the API)",
    )
    parser.add_argument(
        "--out", default=None, help="Output JSON (default: results/bar_vendor_divergence_<today>.json)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    universe = parse_universe(Path(args.universe_file).read_text())
    if not universe:
        raise SystemExit(f"{args.universe_file} yielded an empty universe")
    state = json.loads(Path(args.state_json).read_text())
    held = sorted(state["positions"])
    # Fetch/load the union of the universe file and the held book: the live
    # loop itself prices file ∪ book (an index-leaver stays held, exit-only),
    # so the NAV read needs bars for held names the universe file dropped.
    symbols = sorted(set(universe) | set(held))
    held_extra = sorted(set(held) - set(universe))
    if held_extra:
        print(f"LOUD: {len(held_extra)} held names outside the universe file: {held_extra}")

    spine_closes, missing_spine = load_spine_closes(symbols, Path(args.data_dir), args.cache_suffix)
    if missing_spine:
        print(f"LOUD: {len(missing_spine)} names have no spine cache: {missing_spine}")

    n_requests = 0
    fetch_set = [s for s in symbols if s not in missing_spine]
    if args.iex_cache and Path(args.iex_cache).exists():
        iex_closes = pd.read_parquet(args.iex_cache)
        iex_closes.index = pd.DatetimeIndex(iex_closes.index)
        empty_iex = sorted(set(fetch_set) - set(iex_closes.columns))
        iex_reused = True
    else:
        session = CountingSession(__import__("requests").Session())
        source = AlpacaBarSource.from_env(session=session)
        iex_closes, empty_iex = fetch_iex_closes(source, fetch_set, args.iex_start)
        n_requests = session.n_requests
        if args.iex_cache:
            iex_closes.to_parquet(args.iex_cache)
        iex_reused = False
    if empty_iex:
        print(f"LOUD: {len(empty_iex)} names returned zero IEX bars: {empty_iex}")
    if len(empty_iex) > MAX_MISSING_FRACTION * len(symbols):
        raise SystemExit(
            f"{len(empty_iex)}/{len(symbols)} names empty on IEX exceeds the "
            f"{MAX_MISSING_FRACTION:.0%} bound — feed/auth failure, not dead tickers (N7)"
        )

    # ---- close-diff panel -------------------------------------------------
    # Diff and flags over the full symbol set (held extras flaggable too);
    # the reported distribution is the universe cross-section per the spec.
    diff, coverage = align_panels(spine_closes, iex_closes, args.start, args.end)
    coverage["names_missing_spine_cache"] = missing_spine
    coverage["names_empty_iex"] = empty_iex
    coverage["held_names_outside_universe"] = held_extra
    if coverage["names_no_valid_diff_days"]:
        print(
            "LOUD: names with zero sessions priced by both vendors "
            f"(absent from every distribution): {coverage['names_no_valid_diff_days']}"
        )
    name_stats = per_name_diff_stats(diff)
    flagged = flag_adjustment_mismatch(name_stats)
    flagged_names = [f["symbol"] for f in flagged]
    if flagged:
        print(f"LOUD: adjustment-basis suspects excluded from distribution: {flagged_names}")
    universe_diff = diff[[s for s in diff.columns if s in set(universe)]]
    panel = panel_diff_stats(universe_diff, exclude=flagged_names)

    # ---- rank impact at month-end decision bars ---------------------------
    calendar = pd.DatetimeIndex(spine_closes.dropna(how="all").index)
    decision_bars = month_end_decision_bars(calendar, args.start, coverage["window_effective"][1])
    diff_sample = universe_diff.drop(columns=flagged_names, errors="ignore").to_numpy().ravel()
    diff_sample = diff_sample[np.isfinite(diff_sample)]

    # The decided cross-section is the universe file (held extras are
    # exit-only in the live loop and never re-ranked), minus flagged names.
    rank_names = [s for s in universe if s not in set(flagged_names)]
    per_refresh: list[dict] = []
    perturbation: list[dict] = []
    for bar in decision_bars:
        skip_date, look_date = score_endpoints(calendar, bar, args.skip, args.lookback)
        spine_cross = spine_closes[[s for s in rank_names if s in spine_closes.columns]]
        iex_cross = iex_closes[[s for s in rank_names if s in iex_closes.columns]]
        s_scores = endpoint_scores(spine_cross, skip_date, look_date)
        x_scores = endpoint_scores(iex_cross, skip_date, look_date)
        report = leg_flip_report(s_scores, x_scores, args.decile)
        report.update(
            date=str(bar.date()),
            skip_endpoint=str(skip_date.date()),
            lookback_endpoint=str(look_date.date()),
        )
        per_refresh.append(report)
        skip_close = spine_cross.loc[skip_date].dropna()
        look_close = spine_cross.loc[look_date].dropna()
        shared = skip_close.index.intersection(look_close.index)
        mc = perturbed_flip_stats(
            skip_close[shared],
            look_close[shared],
            diff_sample,
            decile=args.decile,
            n_draws=args.n_draws,
            seed=args.seed,
        )
        mc["date"] = str(bar.date())
        perturbation.append(mc)

    # ---- NAV mark on the held book ----------------------------------------
    # Two reads: raw (every held name, adjustment-basis wedge included — what
    # a naive cross-vendor mark would actually show) and ex-flagged (genuine
    # per-session vendor divergence, the headline). A flagged held name marks
    # hugely different on the two vendors, but that is the frozen cache's
    # adjustment basis, not price disagreement — attributed, not hidden.
    sessions = pd.DatetimeIndex(diff.index)
    held_flagged = sorted(set(held) & set(flagged_names))
    nav_raw = nav_mark_report(
        state["positions"], float(state["cash"]), spine_closes, iex_closes, sessions
    )
    clean_positions = {s: q for s, q in state["positions"].items() if s not in set(held_flagged)}
    nav_clean = nav_mark_report(
        clean_positions, float(state["cash"]), spine_closes, iex_closes, sessions
    )
    nav = {
        "book_source": args.state_json,
        "book_last_settled_bar": state.get("last_settled_bar"),
        "adjustment_flagged_held_names": held_flagged,
        "ex_adjustment_flagged": nav_clean,
        "raw_including_adjustment_flagged": {
            k: v for k, v in nav_raw.items() if k != "per_session"
        },
    }
    if held_flagged:
        print(
            f"LOUD: held names {held_flagged} are adjustment-basis flagged; "
            "the raw NAV read carries that artifact, the headline excludes it"
        )

    payload = {
        "meta": {
            "generated": date.today().isoformat(),
            "universe_file": args.universe_file,
            "n_universe": len(universe),
            "cache_suffix": args.cache_suffix,
            "window_requested": [args.start, args.end or "latest common session"],
            "iex_fetch_start": args.iex_start,
            "iex_feed": "iex",
            "iex_reused_from_cache": iex_reused,
            "n_http_requests": n_requests,
            "skip_bars": args.skip,
            "lookback_bars": args.lookback,
            "decile": args.decile,
            "adjustment_flag_bps": ADJUSTMENT_FLAG_BPS,
            "note": (
                "spine caches frozen at their fetch date vs a current Alpaca "
                "split-adjusted series: flagged names are adjustment-basis "
                "artifacts, not price divergence; the eligibility screen the "
                "live book applies is not reproduced here (rank read is over "
                "the current-universe cross-section)"
            ),
        },
        "coverage": coverage,
        "close_diff_panel": panel,
        "close_diff_per_name": name_stats,
        "adjustment_basis_flags": flagged,
        "rank_impact": {
            "decision_bars": [str(b.date()) for b in decision_bars],
            "total_long_flips": sum(r["long_flips"] for r in per_refresh),
            "total_short_flips": sum(r["short_flips"] for r in per_refresh),
            "per_refresh": per_refresh,
        },
        "decile_boundary_sensitivity": perturbation,
        "nav_mark": nav,
    }
    out_path = (
        Path(args.out)
        if args.out
        else Path("results") / f"bar_vendor_divergence_{date.today().isoformat()}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2))
    compact = {
        k: v
        for k, v in payload.items()
        if k not in ("close_diff_per_name", "decile_boundary_sensitivity")
    }
    compact["nav_mark"] = {
        **{k: v for k, v in nav.items() if k != "ex_adjustment_flagged"},
        "ex_adjustment_flagged": {
            k: v for k, v in nav["ex_adjustment_flagged"].items() if k != "per_session"
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
