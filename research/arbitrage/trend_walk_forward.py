"""trend_v1 walk-forward core — pure, synthetic-testable (docs/trend_design.md §§1-3).

No argparse, no ledger, no git. The counted CLI driver is
``research/scripts/trend_wfo.py``; the read-only adjudicator is
``research/scripts/trend_adjudicate.py``.

Pinned readings (all path:line pointers read at authoring time):

* **Fold geometry mirrors B1 exactly.** ``StatArbWalkForwardConfig(formation_bars=312,
  test_bars=63, step_bars=None, min_test_bars=20)`` -> ``iter_walk_forward_slices``
  (``research/arbitrage/walk_forward.py:155-180``); ``_force_fold_flat`` zeroes the last
  two target rows of each fold (``walk_forward.py:264-271``); one full-panel
  ``backtest_target_weights`` over zero-outside-test targets, sliced to the union of
  test indices (``residual_walk_forward.py:392, 478-500``; ``walk_forward.py:278-295``).
  The closed-form band is *defined* on a formation-window replay
  (``residual_walk_forward.py:444-462``), so "identical cost stack" has no meaning
  without formation windows; and the certified B1 stream is the convexity conditioning
  leg, so identical purge/flatten seams keep the two streams comparable. The fold
  flatten is a recorded turnover tax, never fitted.
* **Spread accounting = the frozen schedule, applied per fold.** Each fold's bucket
  Series (312-bar formation median of PRICE close x volume through
  ``SPREAD_BUCKET_SCHEDULE_V1``, NaN -> 10 bps; ``residual_walk_forward.py:77-87,
  425-427``) drives BOTH that fold's band cost and the accounting of every fill inside
  that fold's test window (a day x symbol frame handed to ``backtest_target_weights``).
  The sibling froze the FIRST fold's Series for the whole sample
  (``residual_walk_forward.py:428-433``: "one backtest prices the whole sample") — inert
  on B1's 2020+ large-cap panel, but on this 2007+ ETF panel fold 0 puts six of ten names
  at 10 bps that the schedule prices at 1 bps by the last fold. Per-fold buckets are
  equally causal (formation precedes test) and are the schedule as registered; hashed by
  the driver as ``spread_accounting="per_fold_bucket"``; per-fold buckets go to
  ``folds.json``. Pinned 2026-09-03 before any full-sample run.
* **Price convention.** ``total_return`` is the registered pin (``docs/trend_design.md``
  §2: trailing 252-bar *total* return); ``price_return`` is a scratch-only
  data-missing fallback whose gating lives in the driver. Both are hashed.
* **total_return panel.** ``TR_t = TR_{t-1} * (close_t + div_t) / close_{t-1}``, rebased to
  the first close; the backtest credits ``divs[ex] / open[ex-1]`` times the weights held
  at open ``ex-1`` into return row ``ex-1`` (``src/prism/execution/target_weights.py:337-357``),
  with the ex-date price drop inside ``open_ret`` of the same row; ``dividend_return`` is a
  ``costs`` column (``target_weights.py:199``).
* **Decision grid.** ``i % decision_every == 0`` relative to each fold's test-window start
  (``residual_walk_forward.py:232-233``); non-decision rows repeat the held book; an all-zero
  decision row is an explicit zero target through ``step_no_trade_band``
  (``src/prism/portfolio/construct.py:100-131``; band 0 => exact exit); the formation
  replay is frozen the same way before ``var(ddof=1)`` (``residual_walk_forward.py:304-306, 452``).
* **Construction.** ``construct_inverse_vol_targets`` (``construct.py:248-310``; NaN score /
  NaN sigma -> explicit 0.0); ``equal_notional`` (T4) uses the same support and
  ``w_i = sign_i * max_gross / |support|``. Full-panel construction is causal (EWMA with
  ``adjust=False``, trailing ratio), so slicing full-panel targets to a window equals
  constructing on the window.
* **Listing rule.** Each name enters at its own first cached session + ``entry_bars`` (252,
  ``docs/trend_design.md`` §1 honesty note (i)) regardless of ``lookback_bars``; before entry
  the weight is an explicit 0.0 (empty cell holds cash). ``entry_bars`` is a universe rule,
  so T1 moves exactly one hashed key.
* **Signal.** ``score_panel_tsmom`` is duplicated by value from
  ``research/scripts/joint_crash_receipt.py:115-136`` (SPEC §7: no imports from
  ``research/scripts`` into the arbitrage tier); the node parity guard mirrors
  ``joint_crash_receipt.py:151-170`` and ``tests/test_trend_walk_forward.py`` asserts exact
  equality with both.
* **Timezone.** The cache index dtype is ``datetime64[ns, America/New_York]``
  (``src/prism/io/loader.py:45-55`` localises to BAR_TZ before caching). A tz-naive
  ``end_date`` raises on comparison, ``tz="UTC"`` silently drops the cutoff bar, so the
  loaders REFUSE tz-naive input and never localise themselves; the driver owns
  localisation via ``resolve_end_date``. ``TREND_UNCOUNTED_END_MAX`` is a driver guard
  constant (``docs/trend_design.md`` §0, "none runs after it outside the §3 counted set"),
  deliberately absent from the hashed payload. This core has no such guard by design
  (pure, synthetic-testable) — that residual is a governance matter, not a driver defect.
* **Caches.** Explicit file lists only (``PINNED_CACHE_FILES``); never glob. Within-file
  duplicate stamps (EEM: a vendor artifact) are resolved keep-last and counted; sessions
  present in more than one listed file must agree on close to ``OVERLAP_REL_TOL`` (N7
  fail-loud), after which cross-file keep-last is inert. No consensus-day filter, no
  eligibility screen, no ffill: the union calendar is asserted gap-free per symbol.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from prism.config import ExecutionConfig
from prism.execution.target_weights import PortfolioBacktestResult, backtest_target_weights
from prism.portfolio.construct import cap_book, closed_form_band, construct_inverse_vol_targets
from prism.signal.trend_node import TREND_V1_UNIVERSE, TrendSignalNode
from prism.validation.metrics import periodic_sharpe
from research.arbitrage.residual_walk_forward import (
    GAMMA_RISK,
    SPREAD_BUCKET_SCHEDULE_V1,
    _fold_cost_share,
    _online_banded_targets,
    bucket_spread_bps,
)
from research.arbitrage.walk_forward import (
    StatArbWalkForwardConfig,
    _empty_targets,
    _fold_metrics_from_result,
    _FoldSlices,
    _force_fold_flat,
    _numeric_prices,
    _slice_portfolio_result,
    iter_walk_forward_slices,
)

__all__ = [
    "STRATEGY_TAG",
    "TREND_V1_UNIVERSE",
    "TREND_TZ",
    "TREND_END_DATE_DEFAULT",
    "TREND_UNCOUNTED_END_MAX",
    "TREND_T5_END_MIN",
    "ENTRY_BARS",
    "OVERLAP_REL_TOL",
    "DATA_CONVENTION",
    "PINNED_CACHE_FILES",
    "GAMMA_RISK",
    "SPREAD_BUCKET_SCHEDULE_V1",
    "CacheOverlapDisagreement",
    "TrendSleeveConfig",
    "TrendFoldResult",
    "TrendWalkForwardResult",
    "resolve_end_date",
    "load_symbol_bars",
    "load_pinned_panels",
    "missing_distribution_files",
    "distributions_missing_message",
    "load_distributions",
    "total_return_close",
    "score_panel_tsmom",
    "listing_entry_mask",
    "construct_equal_notional_targets",
    "construct_targets",
    "freeze_to_decision_bars",
    "run_trend_walk_forward",
    "trend_fold_to_dict",
]

STRATEGY_TAG = "trend_v1"
# Cache index tz (src/prism/io/loader.py:45-55 writer; index dtype datetime64[ns, America/New_York]).
TREND_TZ = "America/New_York"
# Last bar before family ratification 2026-07-18 (docs/trend_design.md status banner and §3).
TREND_END_DATE_DEFAULT = "2026-07-17"
# Hard cap for any run that is not a registered cell (docs/trend_design.md §0, last paragraph:
# "none runs after it outside the §3 counted set"); a guard, NOT a knob — never hashed.
TREND_UNCOUNTED_END_MAX = "2021-12-31"
# T5 promotion read: sample extended >= 1 year past ratification 2026-07-18 (docs/trend_design.md:167);
# a guard, NOT a knob — never hashed.
TREND_T5_END_MIN = "2027-07-17"
# §1 listing rule: a name enters its cell from listing + 252 bars (docs/trend_design.md §1, note (i)).
ENTRY_BARS = 252
OVERLAP_REL_TOL = 1e-6
DATA_CONVENTION: dict[str, str] = {
    "total_return": "split_adjusted_total_return_dividends_as_cash",
    # = research/scripts/stat_arb_residual_wfo.py:591 (the sibling's recorded convention).
    "price_return": "split_adjusted_open_close_price_return_no_dividends",
}
PINNED_CACHE_FILES: dict[str, tuple[str, ...]] = {
    "SPY": ("SPY_1d_min_2026-08-14.parquet", "SPY_1d_2018-01-01_2026-07-22.parquet"),
    **{
        s: (f"{s}_1d_2007-01-01_2017-12-31.parquet", f"{s}_1d_2018-01-01_2026-07-22.parquet")
        for s in ("EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "PDBC", "UUP")
    },
}
_REQUIRED_BAR_COLUMNS = ("open", "close", "volume")
_BARE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SIZING_MODES = ("inverse_vol", "equal_notional")
VERIFICATION_FILE = "verification_2026-09-03.json"


class CacheOverlapDisagreement(ValueError):
    """Two listed cache files disagree on a session's close beyond ``OVERLAP_REL_TOL``."""


def resolve_end_date(text: str) -> pd.Timestamp:
    """``pd.Timestamp(text, tz=TREND_TZ)`` for a bare ``YYYY-MM-DD`` (no time, no offset).

    NY midnight is the only localisation under which the cutoff bar of the cache
    is retained (``tz="UTC"`` drops it; tz-naive raises on comparison).
    """
    if not isinstance(text, str) or not _BARE_DATE.match(text):
        raise SystemExit(
            f"end_date must be a bare YYYY-MM-DD (no time, no offset), got {text!r}; "
            f"localisation is fixed to {TREND_TZ} midnight"
        )
    try:
        return pd.Timestamp(text, tz=TREND_TZ)
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
        raise SystemExit(f"end_date {text!r} is not a calendar date: {exc}") from exc


def _require_tz_aware(value: pd.Timestamp, name: str) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or value.tzinfo is None:
        raise ValueError(
            f"{name} must be a tz-aware pandas Timestamp (use resolve_end_date); loaders never localise, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class TrendSleeveConfig:
    """The four registered knobs plus the pinned constants they sit beside."""

    price_convention: str = "total_return"  # | "price_return" (scratch fallback only; driver-gated)
    lookback_bars: int = 252
    skip_bars: int = 21
    decision_every: int = 21
    entry_bars: int = ENTRY_BARS  # fixed; independent of lookback (T1 moves one key)
    sizing: str = "inverse_vol"  # | "equal_notional"
    vol_ewma_bars: int = 63
    annualization_bars: int = 252
    dollar_volume_window: int = 20  # gate ADV window (residual_walk_forward.py:412-414; B1 config.json:20)

    def __post_init__(self) -> None:
        if self.lookback_bars < 2:
            raise ValueError(f"lookback_bars must be >= 2, got {self.lookback_bars}")
        if not 0 <= self.skip_bars < self.lookback_bars:
            raise ValueError(f"skip_bars must satisfy 0 <= skip < lookback, got {self.skip_bars}/{self.lookback_bars}")
        if self.decision_every < 1:
            raise ValueError(f"decision_every must be >= 1, got {self.decision_every}")
        if self.entry_bars < 1:
            raise ValueError(f"entry_bars must be >= 1, got {self.entry_bars}")
        if self.sizing not in _SIZING_MODES:
            raise ValueError(f"sizing must be one of {_SIZING_MODES}, got {self.sizing!r}")
        if self.price_convention not in DATA_CONVENTION:
            raise ValueError(f"price_convention must be one of {tuple(DATA_CONVENTION)}, got {self.price_convention!r}")
        if self.vol_ewma_bars < 2:
            raise ValueError(f"vol_ewma_bars must be >= 2, got {self.vol_ewma_bars}")
        if self.annualization_bars < 1:
            raise ValueError(f"annualization_bars must be >= 1, got {self.annualization_bars}")
        if self.dollar_volume_window < 1:
            raise ValueError(f"dollar_volume_window must be >= 1, got {self.dollar_volume_window}")


# --------------------------------------------------------------------------------------
# Loading (explicit file lists, N7 fail-loud)
# --------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_symbol_bars(
    paths: Sequence[Path | str],
    *,
    end_date: pd.Timestamp,
    symbol: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Splice the listed parquet files (in order) into one open/close/volume frame.

    Per file: require ``open, close, volume`` and a tz-aware index; drop within-file
    duplicate stamps keep-last (counted); sort. Across files: sessions present in
    both must agree on close to ``OVERLAP_REL_TOL`` (else ``CacheOverlapDisagreement``),
    then concat with later-file-wins keep-last (inert given agreement). Finally
    truncate to ``index <= end_date`` (tz-aware vs tz-aware).
    """
    end_date = _require_tz_aware(end_date, "end_date")
    path_list = [Path(p) for p in paths]
    if not path_list:
        raise ValueError("load_symbol_bars needs at least one path")
    sym = symbol or path_list[0].name.split("_")[0]
    frames: list[pd.DataFrame] = []
    files_meta: list[dict[str, object]] = []
    for path in path_list:
        if not path.is_file():
            raise SystemExit(f"{sym}: pinned cache file missing: {path}")
        raw = pd.read_parquet(path)
        missing = [c for c in _REQUIRED_BAR_COLUMNS if c not in raw.columns]
        if missing:
            raise SystemExit(f"{sym}: {path.name} lacks required columns {missing}")
        if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.tz is None:
            raise SystemExit(f"{sym}: {path.name} index is not a tz-aware DatetimeIndex (loader.py writes {TREND_TZ})")
        rows_raw = int(len(raw))
        dup_mask = raw.index.duplicated(keep="last")
        frame = raw.loc[~dup_mask, list(_REQUIRED_BAR_COLUMNS)].sort_index()
        frame = frame.apply(pd.to_numeric, errors="coerce").astype(float)
        files_meta.append(
            {
                "file": path.name,
                "sha256": _sha256_file(path),
                "rows_raw": rows_raw,
                "rows_unique": int(len(frame)),
                "first": frame.index[0].isoformat() if len(frame) else None,
                "last": frame.index[-1].isoformat() if len(frame) else None,
                "dup_rows_dropped": int(dup_mask.sum()),
            }
        )
        frames.append(frame)

    combined = frames[0]
    n_overlap = 0
    max_rel = 0.0
    for frame in frames[1:]:
        common = combined.index.intersection(frame.index)
        if len(common):
            a = combined.loc[common, "close"].to_numpy(dtype=float)
            b = frame.loc[common, "close"].to_numpy(dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.abs(a / b - 1.0)
            finite = np.isfinite(rel)
            worst = float(np.max(rel[finite])) if finite.any() else 0.0
            n_overlap += int(len(common))
            max_rel = max(max_rel, worst)
            if worst > OVERLAP_REL_TOL:
                bad = np.flatnonzero(finite & (rel > OVERLAP_REL_TOL))
                first_bad = pd.Timestamp(common[bad[0]]).date().isoformat()
                raise CacheOverlapDisagreement(
                    f"{sym}: {len(bad)} overlapping sessions disagree, max rel {worst:.3e} ({first_bad})"
                )
        combined = pd.concat([combined, frame])
        combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()

    combined = combined.loc[combined.index <= end_date]
    meta: dict[str, object] = {
        "files": files_meta,
        "overlap_sessions_checked": int(n_overlap),
        "max_rel_close_diff": float(max_rel),
    }
    return combined, meta


def load_pinned_panels(
    data_dir: Path | str,
    symbols: Sequence[str],
    *,
    end_date: pd.Timestamp,
    cache_files: Mapping[str, Sequence[str]] = PINNED_CACHE_FILES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    """Outer-joined day x symbol ``(closes, opens, volumes)`` on the union calendar, no ffill.

    Asserts the union index is unique and monotonic, and that every symbol's sessions
    between its first and last valid close equal the union calendar over that span
    (else ``SystemExit`` naming the symbol and the first gap). Does NOT assert the
    cutoff itself — that is the driver's guard 6, so the core stays usable on
    synthetic panels of any span.
    """
    end_date = _require_tz_aware(end_date, "end_date")
    root = Path(data_dir)
    symbol_list = list(symbols)
    closes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    meta: dict[str, dict[str, object]] = {}
    for sym in symbol_list:
        if sym not in cache_files:
            raise SystemExit(f"{sym}: no pinned cache file list (PINNED_CACHE_FILES); never glob")
        paths = [root / name for name in cache_files[sym]]
        frame, sym_meta = load_symbol_bars(paths, end_date=end_date, symbol=sym)
        if frame.empty:
            raise SystemExit(f"{sym}: no bars at or before {end_date.isoformat()} in {[p.name for p in paths]}")
        closes[sym] = frame["close"]
        opens[sym] = frame["open"]
        volumes[sym] = frame["volume"]
        meta[sym] = sym_meta

    close_panel = pd.DataFrame(closes).sort_index().reindex(columns=symbol_list).astype(float)
    index = close_panel.index
    if not index.is_unique or not index.is_monotonic_increasing:
        raise SystemExit("union calendar is not unique+monotonic after splicing")
    open_panel = pd.DataFrame(opens).reindex(index=index, columns=symbol_list).astype(float)
    volume_panel = pd.DataFrame(volumes).reindex(index=index, columns=symbol_list).astype(float)

    for sym in symbol_list:
        col = close_panel[sym]
        valid = index[col.notna().to_numpy()]
        if len(valid) == 0:
            raise SystemExit(f"{sym}: no valid close at or before {end_date.isoformat()}")
        first = pd.Timestamp(valid[0])
        last = pd.Timestamp(valid[-1])
        span = col.loc[first:last]
        gaps = span.index[span.isna().to_numpy()]
        if len(gaps):
            raise SystemExit(
                f"{sym}: close gap on {pd.Timestamp(gaps[0]).isoformat()} ({len(gaps)} missing sessions between "
                f"{first.date()} and {last.date()}) — the union calendar is not "
                "gap-free for this name (no ffill, no proxy: N7)"
            )
        meta[sym].update(
            {
                "first_session": first.isoformat(),
                "last_session": last.isoformat(),
                "n_sessions": int(span.notna().sum()),
            }
        )
    return close_panel, open_panel, volume_panel, meta


def missing_distribution_files(dist_dir: Path | str, symbols: Sequence[str]) -> list[str]:
    """Names for which ``<SYM>.csv`` or ``<SYM>.provenance.json`` is absent; ``[]`` when complete."""
    root = Path(dist_dir)
    return [s for s in symbols if not ((root / f"{s}.csv").is_file() and (root / f"{s}.provenance.json").is_file())]


def distributions_missing_message(names: Sequence[str]) -> str:
    return (
        f"distributions missing for {sorted(names)}; the registered convention is total_return — fetch the "
        "missing files, or run a scratch diagnostic with --price_convention price_return "
        "--allow_price_return_fallback (non-results/ --output_dir and non-default --trial_ledger)"
    )


def load_distributions(
    dist_dir: Path | str,
    symbols: Sequence[str],
    index: pd.DatetimeIndex,
    *,
    first_session: Mapping[str, pd.Timestamp],
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Per-share cash frame (day x symbol, 0.0 elsewhere) from ``<SYM>.csv`` + provenance.

    ``ex_date`` is localised to ``TREND_TZ`` midnight (the same localisation as
    ``end_date`` and the cache index); rows outside ``[first_session[sym], end_date]``
    are dropped; an ex_date not on ``index`` moves to the next session (``n_shifted``);
    same-session rows are summed. Negative or non-finite amounts are ``SystemExit``.
    """
    end_date = _require_tz_aware(end_date, "end_date")
    root = Path(dist_dir)
    calendar = pd.DatetimeIndex(index)
    if calendar.tz is None:
        raise ValueError("index must be tz-aware (the cache calendar)")
    if not calendar.is_monotonic_increasing or not calendar.is_unique:
        raise ValueError("index must be sorted and unique")
    symbol_list = list(symbols)
    missing = missing_distribution_files(root, symbol_list)
    if missing:
        raise SystemExit(distributions_missing_message(missing))

    values = np.zeros((len(calendar), len(symbol_list)), dtype=float)
    meta: dict[str, object] = {}
    for col_pos, sym in enumerate(symbol_list):
        csv_path = root / f"{sym}.csv"
        prov_path = root / f"{sym}.provenance.json"
        table = pd.read_csv(csv_path, dtype=str)
        if not {"ex_date", "amount"} <= set(table.columns):
            raise SystemExit(f"{sym}: {csv_path} must carry columns ex_date, pay_date, amount; got {list(table.columns)}")
        first = _require_tz_aware(pd.Timestamp(first_session[sym]), f"first_session[{sym}]")
        n_shifted = 0
        n_beyond = 0
        ex_dates: list[pd.Timestamp] = []
        amounts: list[float] = []
        if len(table):
            ex_series = pd.to_datetime(table["ex_date"].str.strip(), format="%Y-%m-%d").dt.tz_localize(TREND_TZ)
            amount_series = pd.to_numeric(table["amount"], errors="coerce")
            amount_arr = amount_series.to_numpy(dtype=float)
            if not np.isfinite(amount_arr).all() or (amount_arr < 0.0).any():
                raise SystemExit(f"{sym}: negative or non-finite distribution amount in {csv_path}")
            keep = (ex_series >= first) & (ex_series <= end_date)
            for ex, amount in zip(ex_series[keep], amount_arr[keep.to_numpy()]):
                ex_dates.append(pd.Timestamp(ex))
                amounts.append(float(amount))
        for ex, amount in zip(ex_dates, amounts):
            pos = int(calendar.searchsorted(ex))
            if pos >= len(calendar):
                n_beyond += 1
                continue
            if calendar[pos] != ex:
                n_shifted += 1
            values[pos, col_pos] += amount
        meta[sym] = {
            "file": str(csv_path),
            "sha256": _sha256_file(csv_path),
            "provenance_file": str(prov_path),
            "provenance_sha256": _sha256_file(prov_path),
            "n_records_in_sample": int(len(ex_dates)),
            "n_sessions_credited": int((values[:, col_pos] != 0.0).sum()),
            "first_ex_date": min(ex_dates).date().isoformat() if ex_dates else None,
            "last_ex_date": max(ex_dates).date().isoformat() if ex_dates else None,
            "n_shifted": int(n_shifted),
            "n_beyond_index": int(n_beyond),
        }
    verification = root / VERIFICATION_FILE
    if verification.is_file():
        payload = json.loads(verification.read_text())
        meta["verification"] = {
            "file": str(verification),
            "sha256": _sha256_file(verification),
            "overall": payload.get("overall"),
            "per_symbol": payload.get("per_symbol"),
        }
    out = pd.DataFrame(values, index=calendar, columns=symbol_list)
    return out, meta


# --------------------------------------------------------------------------------------
# Total-return panel, signal, entry rule, construction
# --------------------------------------------------------------------------------------


def total_return_close(close: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Per column ``TR_t = TR_{t-1} * (close_t + div_t) / close_{t-1}``, ``TR_first = close_first``.

    NaN before listing. ``TR_t / close_t`` is non-decreasing and equals
    ``prod_{ex <= t} (1 + div_ex / close_ex)`` (reinvestment at the ex-date close).
    """
    close_num = close.apply(pd.to_numeric, errors="coerce").astype(float)
    div = dividends.reindex(index=close.index, columns=close.columns).apply(pd.to_numeric, errors="coerce")
    div = div.fillna(0.0).astype(float)
    ratio = (close_num + div) / close_num.shift(1)
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        first = close_num[col].first_valid_index()
        if first is None:
            continue
        r = ratio[col].loc[first:].copy()
        r.loc[first] = 1.0
        out.loc[first:, col] = float(close_num.at[first, col]) * r.cumprod()
    return out


def score_panel_tsmom(
    close: pd.DataFrame,
    *,
    lookback_bars: int = 252,
    skip_bars: int = 21,
) -> pd.DataFrame:
    """Full-panel 12-1 TSMOM matching ``TrendSignalNode.score`` at every bar.

    Duplicated by value from ``research/scripts/joint_crash_receipt.py:115-136`` (SPEC §7:
    the arbitrage tier does not import from ``research/scripts``); the test suite asserts
    exact equality with the receipt's function and with the node on the last row.
    ``close[t-skip] / close[t-lookback] - 1`` with non-positive bases -> NaN.
    """
    px = close.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n_days, n_symbols = px.shape
    scores = np.full((n_days, n_symbols), np.nan)
    if n_days > lookback_bars:
        with np.errstate(invalid="ignore", divide="ignore"):
            base = px[: n_days - lookback_bars]
            recent = px[lookback_bars - skip_bars : n_days - skip_bars]
            scores[lookback_bars:] = np.where(base > 0.0, recent / base - 1.0, np.nan)
    return pd.DataFrame(scores, index=close.index, columns=close.columns)


def listing_entry_mask(close: pd.DataFrame, entry_bars: int) -> pd.DataFrame:
    """True at position >= first_valid_position + entry_bars, per column (never before)."""
    if entry_bars < 1:
        raise ValueError(f"entry_bars must be >= 1, got {entry_bars}")
    px = close.apply(pd.to_numeric, errors="coerce")
    mask = np.zeros(px.shape, dtype=bool)
    for j, col in enumerate(px.columns):
        first = px[col].first_valid_index()
        if first is None:
            continue
        p0 = int(px.index.get_loc(first))
        mask[p0 + entry_bars :, j] = True
    return pd.DataFrame(mask, index=close.index, columns=close.columns)


def _ewma_sigma(close: pd.DataFrame, vol_ewma_bars: int, annualization_bars: int) -> pd.DataFrame:
    # Same estimator as construct_inverse_vol_targets (construct.py:280-286). ``fill_method=None``
    # is identical to the pad default when the only NaNs are leading (the gap check guarantees).
    close_num = close.apply(pd.to_numeric, errors="coerce")
    rets = close_num.pct_change(fill_method=None)
    return rets.ewm(span=vol_ewma_bars, min_periods=vol_ewma_bars, adjust=False).std() * np.sqrt(
        float(annualization_bars)
    )


def construct_equal_notional_targets(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    *,
    vol_ewma_bars: int = 63,
    max_gross: float = 1.0,
    max_symbol_abs_weight: float = 1.0,
    annualization_bars: int = 252,
) -> pd.DataFrame:
    """T4 sizing: ``w_i = sign_i * max_gross / |S|`` on the SAME support as inverse-vol.

    ``S = finite(score) & sign != 0 & finite(sigma) & sigma > 0`` with sigma the 63-bar EWMA
    vol used only to define the support, so T4 differs from T0 in magnitudes only.
    """
    if vol_ewma_bars < 2:
        raise ValueError(f"vol_ewma_bars must be >= 2, got {vol_ewma_bars}")
    if annualization_bars < 1:
        raise ValueError(f"annualization_bars must be >= 1, got {annualization_bars}")
    if not scores.columns.equals(close.columns):
        raise ValueError("scores and close must share columns")
    if not scores.index.equals(close.index):
        raise ValueError("scores and close must share index")
    sigma = _ewma_sigma(close, vol_ewma_bars, annualization_bars).to_numpy(dtype=float)
    score_arr = scores.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    signs = np.sign(score_arr)
    support = np.isfinite(score_arr) & (signs != 0.0) & np.isfinite(sigma) & (sigma > 0.0)
    n_support = support.sum(axis=1, keepdims=True)
    weights = np.where(support, signs * float(max_gross) / np.maximum(n_support, 1), 0.0)
    targets = pd.DataFrame(weights, index=scores.index, columns=scores.columns)
    return cap_book(targets, max_gross=max_gross, max_symbol_abs_weight=max_symbol_abs_weight)


def construct_targets(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    sleeve: TrendSleeveConfig,
    walk: StatArbWalkForwardConfig,
) -> pd.DataFrame:
    """Close-time targets for the whole panel (causal; NaN score / NaN sigma -> explicit 0.0)."""
    if sleeve.sizing == "inverse_vol":
        return construct_inverse_vol_targets(
            scores,
            close,
            vol_ewma_bars=sleeve.vol_ewma_bars,
            max_gross=walk.max_gross,
            max_symbol_abs_weight=walk.max_symbol_abs_weight,
            annualization_bars=sleeve.annualization_bars,
        )
    return construct_equal_notional_targets(
        scores,
        close,
        vol_ewma_bars=sleeve.vol_ewma_bars,
        max_gross=walk.max_gross,
        max_symbol_abs_weight=walk.max_symbol_abs_weight,
        annualization_bars=sleeve.annualization_bars,
    )


def freeze_to_decision_bars(frame: pd.DataFrame, decision_every: int) -> pd.DataFrame:
    """``rows[offset] = rows[offset-1]`` when ``offset % decision_every != 0`` (== residual_walk_forward.py:304-306)."""
    if decision_every < 1:
        raise ValueError(f"decision_every must be >= 1, got {decision_every}")
    values = frame.to_numpy(dtype=float).copy()
    for offset in range(1, len(values)):
        if offset % decision_every != 0:
            values[offset] = values[offset - 1]
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def _assert_node_parity(score_close: pd.DataFrame, scores: pd.DataFrame, sleeve: TrendSleeveConfig) -> None:
    """Panel last row must equal ``TrendSignalNode.score`` (joint_crash_receipt.py:151-170)."""
    node = TrendSignalNode(
        lookback_bars=sleeve.lookback_bars,
        skip_bars=sleeve.skip_bars,
        horizon_bars=sleeve.decision_every,
    )
    last_node = node.score(score_close).to_numpy(dtype=float)
    last_panel = scores.iloc[-1].to_numpy(dtype=float)
    if not np.allclose(last_node, last_panel, equal_nan=True, rtol=0.0, atol=1e-12):
        raise SystemExit("score_panel_tsmom disagrees with TrendSignalNode on the last bar — abort")


# --------------------------------------------------------------------------------------
# Walk-forward core
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrendFoldResult:
    """Diagnostics and metrics for one trend walk-forward fold."""

    fold: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    formation_rows: int
    test_rows: int
    names_traded: int
    decision_bars: tuple[str, ...]
    spread_bps: dict[str, float]
    band: dict[str, float]
    sigma2_target: dict[str, float]
    cost_to_gross_pnl: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class TrendWalkForwardResult:
    """End-to-end trend walk-forward result."""

    folds: tuple[TrendFoldResult, ...]
    portfolio: PortfolioBacktestResult
    summary: dict[str, float | str]
    spread_bps_frame: pd.DataFrame | None  # day x symbol bps actually charged (NaN outside test windows; None when flat)
    entry_sessions: dict[str, str | None]
    dividends: pd.DataFrame | None  # the per-share cash frame actually passed to the backtest


@dataclass(frozen=True)
class _FoldShell:
    slices: _FoldSlices
    names_traded: int
    decision_bars: tuple[str, ...]
    spread_bps: dict[str, float]
    band: dict[str, float]
    sigma2_target: dict[str, float]


PARTICIPATION_GATE_NOTE = (
    "inert by magnitude at initial_capital=1.0 (allowed Δw = 0.05·ADV$/1.0); "
    "only the NaN/zero-ADV hold-prior rule is live"
)


def run_trend_walk_forward(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    score_close: pd.DataFrame,
    dividends: pd.DataFrame | None,
    sleeve: TrendSleeveConfig,
    walk: StatArbWalkForwardConfig,
    execution: ExecutionConfig,
    initial_capital: float = 1.0,
) -> TrendWalkForwardResult:
    """Fold accounting (B1 geometry), frozen cost stack, one full-panel backtest.

    ``closes/opens/volumes`` are PRICE panels (fills, ADV, spread buckets);
    ``score_close`` is the scoring panel (total-return close under ``total_return``,
    the price close under ``price_return``) used for the signal and the vol estimate;
    ``dividends`` is the per-share cash frame at ex-date sessions (``None`` under
    ``price_return``).
    """
    if walk.band_mode not in ("fixed", "closed_form"):
        raise ValueError("trend_v1 supports band_mode 'fixed' or 'closed_form' (cost_aware is a residual construct)")
    closes = _numeric_prices(closes)
    symbols = [str(c) for c in closes.columns]
    if not symbols:
        raise ValueError("close panel has no columns")
    opens = _numeric_prices(opens).reindex(index=closes.index, columns=symbols)
    vols = _numeric_prices(volumes).reindex(index=closes.index, columns=symbols)
    score_close = _numeric_prices(score_close).reindex(index=closes.index, columns=symbols)
    n_obs = len(closes)
    if n_obs < walk.formation_bars + walk.min_test_bars + 1:
        raise ValueError("Not enough rows for the requested trend WFO")
    div_frame: pd.DataFrame | None = None
    if dividends is not None:
        div_frame = (
            dividends.reindex(index=closes.index, columns=symbols)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

    raw_dv = closes * vols  # PRICE close x volume (residual_walk_forward.py:404)
    adv_trailing = raw_dv.rolling(sleeve.dollar_volume_window, min_periods=sleeve.dollar_volume_window).median()
    scores = score_panel_tsmom(score_close, lookback_bars=sleeve.lookback_bars, skip_bars=sleeve.skip_bars)
    _assert_node_parity(score_close, scores, sleeve)
    entry_mask = listing_entry_mask(closes, sleeve.entry_bars)
    scores = scores.where(entry_mask)
    targets_all = construct_targets(scores, score_close, sleeve, walk)

    full_targets = _empty_targets(closes.index, symbols)
    all_test_index = pd.Index([])
    spread_frame = pd.DataFrame(np.nan, index=closes.index, columns=symbols, dtype=float)
    shells: list[_FoldShell] = []
    bucket = walk.spread_mode == "bucket"
    for slices in iter_walk_forward_slices(n_obs, walk):
        test_index = closes.index[slices.test]
        targets = freeze_to_decision_bars(targets_all.iloc[slices.test], sleeve.decision_every)
        fold_spread = bucket_spread_bps(raw_dv.iloc[slices.formation].median(axis=0, skipna=True))
        # This fold's buckets price every fill inside its test window (per-fold accounting).
        spread_frame.loc[test_index, symbols] = fold_spread.reindex(symbols).to_numpy(dtype=float)
        effective_spread = fold_spread if bucket else pd.Series(float(execution.spread_bps), index=symbols)
        sigma2 = pd.Series(np.nan, index=symbols, dtype=float)
        band: float | pd.Series
        if walk.band_mode == "closed_form":
            formation_targets = freeze_to_decision_bars(targets_all.iloc[slices.formation], sleeve.decision_every)
            sigma2 = formation_targets.diff().iloc[1:].var(ddof=1)
            spread_i: float | np.ndarray = (
                fold_spread.reindex(symbols).to_numpy(dtype=float) if bucket else float(execution.spread_bps)
            )
            round_trip = 2.0 * (execution.commission_bps + spread_i) / 10_000.0
            band = pd.Series(
                closed_form_band(sigma2.to_numpy(dtype=float), round_trip, gamma_risk=GAMMA_RISK),
                index=symbols,
            )
        else:
            band = float(walk.no_trade_band)  # 0.0 = pass-through
        targets = _online_banded_targets(
            targets,
            band,
            walk,
            dollar_volume=adv_trailing.loc[test_index],
            aum=initial_capital,
            decision_every=sleeve.decision_every,
        )
        targets = _force_fold_flat(targets)
        full_targets.loc[test_index, symbols] = targets
        all_test_index = all_test_index.union(test_index)
        band_series = band if isinstance(band, pd.Series) else pd.Series(band, index=symbols, dtype=float)
        shells.append(
            _FoldShell(
                slices=slices,
                names_traded=int((targets.abs() > 0).any(axis=0).sum()),
                decision_bars=tuple(pd.Timestamp(t).isoformat() for t in test_index[:: sleeve.decision_every]),
                spread_bps={s: float(effective_spread[s]) for s in symbols},
                band={s: float(band_series[s]) for s in symbols},
                sigma2_target={s: float(sigma2[s]) for s in symbols},
            )
        )
    if not shells:
        raise ValueError("No walk-forward folds were produced")

    full_portfolio = backtest_target_weights(
        opens,
        full_targets,
        execution=execution,
        dollar_volume=raw_dv.reindex(index=opens.index, columns=opens.columns),
        initial_capital=initial_capital,
        spread_bps_per_name=spread_frame if bucket else None,
        dividends=div_frame,
    )
    portfolio = _slice_portfolio_result(full_portfolio, all_test_index)

    folds: list[TrendFoldResult] = []
    for shell in shells:
        formation_index = closes.index[shell.slices.formation]
        test_index = closes.index[shell.slices.test]
        folds.append(
            TrendFoldResult(
                fold=int(shell.slices.fold),
                formation_start=pd.Timestamp(formation_index[0]),
                formation_end=pd.Timestamp(formation_index[-1]),
                test_start=pd.Timestamp(test_index[0]),
                test_end=pd.Timestamp(test_index[-1]),
                formation_rows=len(formation_index),
                test_rows=len(test_index),
                names_traded=shell.names_traded,
                decision_bars=shell.decision_bars,
                spread_bps=shell.spread_bps,
                band=shell.band,
                sigma2_target=shell.sigma2_target,
                cost_to_gross_pnl=_fold_cost_share(full_portfolio, test_index),
                metrics=_fold_metrics_from_result(full_portfolio, test_index),
            )
        )

    entry_sessions: dict[str, str | None] = {}
    for sym in symbols:
        hits = entry_mask.index[entry_mask[sym].to_numpy(dtype=bool)]
        entry_sessions[sym] = pd.Timestamp(hits[0]).isoformat() if len(hits) else None

    summary: dict[str, float | str] = dict(portfolio.metrics)
    summary.update(
        {
            "n_folds": float(len(folds)),
            "n_symbols": float(len(symbols)),
            "oos_periodic_sharpe": float(periodic_sharpe(portfolio.returns)),
            "avg_names_traded": float(np.mean([f.names_traded for f in folds])),
            "n_sessions": float(n_obs),
            "n_test_rows": float(len(portfolio.returns)),
            "participation_gate_note": PARTICIPATION_GATE_NOTE,
            "total_dividend_return": float(portfolio.costs["dividend_return"].sum()),
        }
    )
    return TrendWalkForwardResult(
        folds=tuple(folds),
        portfolio=portfolio,
        summary=summary,
        spread_bps_frame=spread_frame if bucket else None,
        entry_sessions=entry_sessions,
        dividends=div_frame,
    )


def trend_fold_to_dict(fold: TrendFoldResult) -> dict[str, object]:
    """Deterministic JSON-compatible fold record (dates ISO with offset)."""
    return {
        "fold": int(fold.fold),
        "formation_start": fold.formation_start.isoformat(),
        "formation_end": fold.formation_end.isoformat(),
        "test_start": fold.test_start.isoformat(),
        "test_end": fold.test_end.isoformat(),
        "formation_rows": int(fold.formation_rows),
        "test_rows": int(fold.test_rows),
        "names_traded": int(fold.names_traded),
        "decision_bars": list(fold.decision_bars),
        "spread_bps": {k: float(v) for k, v in sorted(fold.spread_bps.items())},
        "band": {k: float(v) for k, v in sorted(fold.band.items())},
        "sigma2_target": {k: float(v) for k, v in sorted(fold.sigma2_target.items())},
        "cost_to_gross_pnl": float(fold.cost_to_gross_pnl),
        "metrics": {k: float(v) for k, v in sorted(fold.metrics.items())},
    }

