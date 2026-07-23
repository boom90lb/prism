"""Joint crash receipt: B1 alone vs B1+trend over pinned stress windows (W1).

Uncounted diagnostic. Builds trend sleeve daily returns from local TREND_V1
ETF caches (``TrendSignalNode`` + ``construct_inverse_vol_targets`` + next-open
``backtest_target_weights`` on the ``decision_every=21`` grid), joins the
certified B1 stream (``results/demotion_b1/returns.csv``), and writes a
JSON receipt via ``prism.validation.joint_crash.joint_crash_report``.

Does not fetch, does not append to any trial ledger, does not move a
ratified statistic. Empty windows report ``n=0`` / ``total_return=null``
(N7). Frame: ``docs/joint_crash_diagnostic.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from prism.config import ExecutionConfig
from prism.execution.target_weights import backtest_target_weights
from prism.portfolio.construct import construct_inverse_vol_targets
from prism.signal.trend_node import TREND_V1_UNIVERSE, TrendSignalNode
from prism.validation.joint_crash import joint_crash_report

DEFAULT_WINDOWS = {
    "covid_2020_03": ("2020-03-01", "2020-03-31"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
}
DEFAULT_BLEND = {"b1": 0.7, "trend": 0.3}


def strip_sessions(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return idx.normalize()


def load_b1_returns(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    # demotion returns carry tz-aware NY midnight strings
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_convert("America/New_York")
    frame.index = strip_sessions(idx)
    col = "daily_return" if "daily_return" in frame.columns else frame.columns[0]
    return frame[col].astype(float).rename("b1")


def load_etf_panels(
    data_dir: Path,
    symbols: tuple[str, ...],
    *,
    prefer_prefix: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load open/close panels from the longest covering ``{sym}_1d_*.parquet``.

    Prefers files whose name contains ``prefer_prefix`` when set (e.g. the
    2018 deep-history pull); otherwise picks the file with the most rows.
    Duplicate session rows (vendor quirk) keep last.
    """
    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    sources: list[str] = []
    missing: list[str] = []
    for sym in symbols:
        paths = sorted(data_dir.glob(f"{sym}_1d_*.parquet"))
        if not paths:
            missing.append(sym)
            continue
        if prefer_prefix:
            preferred = [p for p in paths if prefer_prefix in p.name]
            paths = preferred or paths
        best = max(paths, key=lambda p: p.stat().st_size)
        bars = pd.read_parquet(best)
        if not isinstance(bars.index, pd.DatetimeIndex):
            bars.index = pd.to_datetime(bars.index)
        if bars.index.duplicated().any():
            bars = bars[~bars.index.duplicated(keep="last")]
        bars = bars.sort_index()
        for col in ("open", "close"):
            if col not in bars.columns:
                raise SystemExit(f"{best.name} missing required column {col!r}")
        idx = strip_sessions(bars.index)
        opens[sym] = pd.Series(bars["open"].to_numpy(dtype=float), index=idx, name=sym)
        closes[sym] = pd.Series(bars["close"].to_numpy(dtype=float), index=idx, name=sym)
        sources.append(best.name)
    if missing:
        raise SystemExit(
            f"missing trend ETF caches for {missing}; expected data/{{SYM}}_1d_*.parquet. "
            "Backfill via DataLoader before running this offline receipt."
        )
    open_panel = pd.DataFrame(opens).sort_index()
    close_panel = pd.DataFrame(closes).sort_index()
    # Intersection calendar: require every name present (fixed 10-name universe).
    both = open_panel.notna().all(axis=1) & close_panel.notna().all(axis=1)
    open_panel = open_panel.loc[both]
    close_panel = close_panel.loc[both]
    return open_panel, close_panel, sources


def decision_grid_mask(index: pd.DatetimeIndex, decision_every: int) -> np.ndarray:
    """Refresh on bar 0, decision_every, 2*decision_every, … of the panel."""
    if decision_every < 1:
        raise ValueError(f"decision_every must be >= 1, got {decision_every}")
    n = len(index)
    mask = np.zeros(n, dtype=bool)
    mask[::decision_every] = True
    return mask


def score_panel_tsmom(
    close: pd.DataFrame,
    *,
    lookback_bars: int = 252,
    skip_bars: int = 21,
) -> pd.DataFrame:
    """Full-panel 12−1 TSMOM matching ``TrendSignalNode.score`` at every bar.

    The node contract scores only the last row of a panel (live path). Offline
    receipts need the same formula on every session: ``close[t-skip] /
    close[t-lookback] - 1`` with non-positive bases → NaN. Spot-checked against
    the node on the panel tail.
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


def trend_sleeve_returns(
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    decision_every: int = 21,
    vol_ewma_bars: int = 63,
    lookback_bars: int = 252,
    skip_bars: int = 21,
    execution: ExecutionConfig | None = None,
) -> tuple[pd.Series, dict]:
    """Pinned T0 mechanics → next-open sleeve returns (uncounted)."""
    # Spot-check vectorized panel vs node on the last bar (contract surface).
    node = TrendSignalNode(
        lookback_bars=lookback_bars,
        skip_bars=skip_bars,
        horizon_bars=decision_every,
    )
    scores = score_panel_tsmom(
        close_panel, lookback_bars=lookback_bars, skip_bars=skip_bars
    )
    last_node = node.score(close_panel)
    last_panel = scores.iloc[-1]
    if not np.allclose(
        last_node.to_numpy(dtype=float),
        last_panel.to_numpy(dtype=float),
        equal_nan=True,
        rtol=0.0,
        atol=1e-12,
    ):
        raise SystemExit(
            "score_panel_tsmom disagrees with TrendSignalNode on last bar — abort"
        )
    targets_full = construct_inverse_vol_targets(
        scores,
        close_panel,
        vol_ewma_bars=vol_ewma_bars,
        max_gross=1.0,
        max_symbol_abs_weight=1.0,
    )
    # Hold between refresh sessions: NaN target row = no-op in backtest.
    mask = decision_grid_mask(targets_full.index, decision_every)
    targets = targets_full.where(pd.Series(mask, index=targets_full.index), other=np.nan)
    # Suppress all-zero decision rows (no tradable names yet) as no-ops too.
    all_zero = (targets.fillna(0.0).abs().sum(axis=1) == 0.0) & mask
    targets = targets.mask(all_zero, other=np.nan)

    result = backtest_target_weights(
        open_panel,
        targets,
        execution=execution or ExecutionConfig(),
        initial_capital=1.0,
    )
    rets = result.returns.rename("trend")
    meta = {
        "lookback_bars": lookback_bars,
        "skip_bars": skip_bars,
        "decision_every": decision_every,
        "vol_ewma_bars": vol_ewma_bars,
        "n_sessions": int(rets.dropna().shape[0]),
        "span": _span(rets.index),
        "n_refresh_rows": int(mask.sum()),
        "execution": {
            "spread_bps": float((execution or ExecutionConfig()).spread_bps),
            "commission_bps": float((execution or ExecutionConfig()).commission_bps),
            "borrow_rate_bps_annual": float(
                (execution or ExecutionConfig()).borrow_rate_bps_annual
            ),
        },
    }
    return rets, meta


def _span(index: pd.DatetimeIndex) -> list[str] | None:
    if len(index) == 0:
        return None
    return [str(pd.Timestamp(index[0]).date()), str(pd.Timestamp(index[-1]).date())]


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--b1_returns", type=Path, default=Path("results/demotion_b1/returns.csv"))
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument(
        "--prefer_cache_prefix",
        default="2018-01-01",
        help="Prefer caches whose filename contains this token (deep-history pull).",
    )
    p.add_argument("--decision_every", type=int, default=21)
    p.add_argument("--vol_ewma_bars", type=int, default=63)
    p.add_argument("--b1_weight", type=float, default=DEFAULT_BLEND["b1"])
    p.add_argument("--trend_weight", type=float, default=DEFAULT_BLEND["trend"])
    p.add_argument(
        "--sensitivity",
        action="store_true",
        help="Also write a fixed-weight capital allocation sensitivity grid "
        "(G4a; b1 weight 0..1 step 0.1). Not optimized weights (G4b).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default results/joint_crash_receipt_YYYY-MM-DD.json)",
    )
    return p.parse_args()


def main() -> None:
    from prism.validation.joint_crash import capital_allocation_sensitivity

    args = parse_args()
    b1 = load_b1_returns(args.b1_returns)
    open_panel, close_panel, sources = load_etf_panels(
        args.data_dir,
        TREND_V1_UNIVERSE,
        prefer_prefix=args.prefer_cache_prefix or None,
    )
    trend, trend_meta = trend_sleeve_returns(
        open_panel,
        close_panel,
        decision_every=args.decision_every,
        vol_ewma_bars=args.vol_ewma_bars,
    )
    sleeves = {"b1": b1, "trend": trend}
    blend_weights = {"b1": args.b1_weight, "trend": args.trend_weight}
    report = joint_crash_report(
        sleeves,
        DEFAULT_WINDOWS,
        blend_weights=blend_weights,
    )
    sensitivity = None
    if args.sensitivity:
        sensitivity = capital_allocation_sensitivity(
            sleeves, DEFAULT_WINDOWS, primary="b1"
        )
    out = args.out or Path(f"results/joint_crash_receipt_{date.today().isoformat()}.json")
    payload = {
        "instrument": "joint_crash",
        "status": "uncounted_diagnostic",
        "gate": "G4a",
        "asof": date.today().isoformat(),
        "b1_source": str(args.b1_returns),
        "b1_span": _span(b1.index),
        "b1_n_sessions": int(b1.dropna().shape[0]),
        "trend_sources": sources,
        "trend_meta": trend_meta,
        "panel_span": _span(close_panel.index),
        "panel_n_sessions": int(len(close_panel)),
        "universe": list(TREND_V1_UNIVERSE),
        "report": report,
        "capital_allocation_sensitivity": sensitivity,
        "notes": [
            "B1 certified demotion stream starts 2021-03-30 on this checkout — "
            "covid_2020_03 is empty for B1 by construction until a longer B1 "
            "stream is produced (not invented here).",
            "Trend stream is offline T0 mechanics on local deep-history ETF "
            "caches; next-open fills; decision_every grid; default ExecutionConfig "
            "costs. Not a promotion read; not G4b.",
            "capital_allocation_sensitivity (when present) is fixed-weight "
            "G4a arithmetic only — optimized multi-sleeve weights require G4b.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"wrote": str(out), "report": report}
    if sensitivity is not None:
        summary["sensitivity_n_rows"] = len(sensitivity["rows"])
    print(json.dumps(_clean(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
