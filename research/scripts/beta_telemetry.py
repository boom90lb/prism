"""Conditional-beta telemetry for the certified B1 momentum book (desk-review Q3, tier 1).

B1 is dollar-neutral by balanced decile legs, not factor-neutralized — the
SPEC §7.2 residualize stage is unwired by design (src/prism/live/daily.py).
Dollar-neutral is not beta-neutral, and the classic momentum-crash mechanism
is a *conditional* exposure: after a market fall the short leg fills with
crashed high-beta names, so the book's beta flips exactly when the market
whipsaws back. An unconditional beta averaging near zero is therefore the
number most likely to falsely reassure — the conditional cells are the point.

What this instrument may claim: realized sample betas of a recorded daily
return stream on a market series — full-sample OLS, rolling 63-session, and
three conditional cells (prior-session worst-decile trailing 21-bar market
windows; prior-session book drawdown state; the book's own worst-decile
months, an ex-post attribution cell) — with n reported for every cell. A
cell whose n falls below ``--min_obs`` reports beta null with its n, loud,
never a silently thin estimate (N7). What it may not claim: any regime event
outside the sample — a sample covariance cannot exhibit one, the identical
caveat docs/momentum_design.md §0 pins on N_eff — nor any change to a
ratified statistic. Output feeds the sizing pre-registration's
crash-conditional de-gross term (docs/handoff.md §8 GO preconditions) as
measured telemetry, nothing more.

Data (committed local artifacts only; this script performs no network I/O):

- book: ``results/demotion_b1/returns.csv`` — the certified B1 OOS stream;
- market, overlap window: SPY bar cache(s) ``data/SPY_1d_*.parquet`` —
  local SPY coverage begins 2025-06-01 (trend-program ADV fetch), so SPY
  cells cover only the final ~12.5 months of the book sample;
- market, full sample: cap-blind equal-weight proxy over the run's own bar
  caches (quarantine fallback included — the certified run consumed those
  bars), validated against SPY on the overlap window before being read;
- cross-check: ``runs/replay_floor_1000000/equity.jsonl`` (113-session
  replay stream through the live-loop mechanics).

Uncounted diagnostic: searches nothing, changes no trial code path, never
appends to the trials ledger, writes one JSON. Frame and results:
``docs/beta_telemetry.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from prism.residual.factors import consensus_trading_days

# ---------------------------------------------------------------------------
# pure computation helpers (unit-tested in tests/test_beta_telemetry.py)
# ---------------------------------------------------------------------------


def strip_sessions(index: pd.Index) -> pd.DatetimeIndex:
    """Normalize a bar index to tz-naive session dates.

    The book CSV (tz-aware NY midnight, mixed DST offsets), the parquet
    caches (tz-aware NY midnight) and the replay ledger (plain date strings)
    all carry the same NYSE sessions under different clothes; every join in
    this instrument is by session, so strip the clothes once, here.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return idx.normalize()


def beta_cell(
    book: pd.Series,
    market: pd.Series,
    mask: pd.Series | None = None,
    min_obs: int = 21,
) -> dict:
    """OLS beta of ``book`` on ``market`` over their joint sample, optionally masked.

    Returns ``{"n", "beta", "alpha_daily", "corr", "r2"}``; below ``min_obs``
    joint observations (or under a zero-variance market cell) ``beta`` is
    ``None`` with the ``n`` still reported and a ``note`` saying why — a thin
    conditional cell must read as *unmeasured*, never as a quiet number (N7).
    """
    frame = pd.concat([book.rename("book"), market.rename("mkt")], axis=1, join="inner").dropna()
    if mask is not None:
        aligned = mask.reindex(frame.index).fillna(False).astype(bool)
        frame = frame.loc[aligned]
    n = int(len(frame))
    cell: dict = {"n": n, "beta": None, "alpha_daily": None, "corr": None, "r2": None}
    if n < max(int(min_obs), 2):
        cell["note"] = f"n={n} below min_obs={min_obs}; beta reported null, not estimated"
        return cell
    market_var = float(frame["mkt"].var(ddof=1))
    # Absolute floor, not an exact-zero test: a constant series carries
    # float-noise variance (~1e-38) that would turn the ratio into garbage.
    # Real daily-return cells sit at var >= ~1e-8; 1e-12 only catches
    # degenerate ones.
    if market_var < 1e-12:
        cell["note"] = "market variance is (numerically) zero on this cell; beta undefined"
        return cell
    beta = float(frame["book"].cov(frame["mkt"]) / market_var)
    corr = float(frame["book"].corr(frame["mkt"]))
    cell.update(
        beta=beta,
        alpha_daily=float(frame["book"].mean() - beta * frame["mkt"].mean()),
        corr=corr,
        r2=corr * corr,
    )
    return cell


def rolling_beta(book: pd.Series, market: pd.Series, window: int) -> pd.Series:
    """Rolling ``window``-session OLS beta over the joint sample (full windows only)."""
    frame = pd.concat([book.rename("book"), market.rename("mkt")], axis=1, join="inner").dropna()
    cov = frame["book"].rolling(window).cov(frame["mkt"])
    var = frame["mkt"].rolling(window).var(ddof=1)
    return (cov / var).dropna()


def rolling_summary(betas: pd.Series, window: int) -> dict:
    """Mean/min/max/last (with dates) of a rolling-beta series; loud when empty."""
    if betas.empty:
        return {
            "window": window,
            "n_windows": 0,
            "mean": None,
            "min": None,
            "max": None,
            "last": None,
            "note": "no full rolling window fits in the joint sample",
        }
    return {
        "window": window,
        "n_windows": int(len(betas)),
        "mean": float(betas.mean()),
        "min": float(betas.min()),
        "min_date": str(betas.idxmin().date()),
        "max": float(betas.max()),
        "max_date": str(betas.idxmax().date()),
        "last": float(betas.iloc[-1]),
        "last_date": str(betas.index[-1].date()),
    }


def trailing_worst_decile_mask(
    market_returns: pd.Series, joint_index: pd.DatetimeIndex, bars: int, q: float
) -> tuple[pd.Series, float, int]:
    """Mask over ``joint_index``: prior-session trailing ``bars``-bar market return in the worst ``q``-quantile.

    Conditioning is strictly prior — the trailing window ends the session
    *before* the return being conditioned — so the conditioning variable
    shares no bar with the conditioned return (no mechanical overlap). The
    quantile threshold is computed over the same prior-session distribution
    the mask is applied to. Returns ``(mask, threshold, n_defined)``.
    """
    trailing = np.expm1(np.log1p(market_returns).rolling(bars).sum())
    prior = trailing.shift(1).reindex(joint_index)
    defined = prior.dropna()
    if defined.empty:
        return pd.Series(False, index=joint_index), float("nan"), 0
    threshold = float(defined.quantile(q))
    mask = (prior <= threshold).fillna(False)
    return mask, threshold, int(len(defined))


def drawdown_state_mask(book_returns: pd.Series, threshold: float) -> tuple[pd.Series, pd.Series]:
    """Prior-session book drawdown exceeds ``threshold`` (peak from sample start).

    The state is lagged one session (drawdown as of the *prior* close) so a
    day's own return never puts it into its own conditioning set. Returns
    ``(state_mask, drawdown_series)``.
    """
    equity = (1.0 + book_returns).cumprod()
    drawdown = 1.0 - equity / equity.cummax()
    state = drawdown.shift(1).fillna(0.0) > threshold
    return state, drawdown


def worst_month_mask(
    book_returns: pd.Series, joint_index: pd.DatetimeIndex, q: float
) -> tuple[pd.Series, float, list[str], int]:
    """Mask over ``joint_index`` covering the book's worst-``q``-quantile calendar months.

    Contemporaneous by construction (a month is selected on the same returns
    then conditioned on) — this is the ex-post attribution cell "what was the
    book's market exposure during its own worst months", not a tradeable
    state. Returns ``(mask, threshold, worst_months, n_months)``.
    """
    joint = book_returns.reindex(joint_index).dropna()
    if joint.empty:
        return pd.Series(False, index=joint_index), float("nan"), [], 0
    monthly = (1.0 + joint).groupby(joint.index.to_period("M")).prod() - 1.0
    threshold = float(monthly.quantile(q))
    worst = monthly.index[monthly <= threshold]
    mask = pd.Series(joint_index.to_period("M").isin(worst), index=joint_index)
    return mask, threshold, [str(p) for p in worst], int(len(monthly))


def equal_weight_returns(closes: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Cap-blind equal-weight daily return across names with a bar that day.

    Returns ``(proxy_returns, names_per_day)``; days with zero contributing
    names are dropped rather than read as zero return.
    """
    returns = closes.pct_change(fill_method=None)
    counts = returns.notna().sum(axis=1)
    proxy = returns.mean(axis=1)
    return proxy[counts > 0], counts


# ---------------------------------------------------------------------------
# artifact loaders (local files only — no DataLoader, no network path at all)
# ---------------------------------------------------------------------------


def load_book_returns(run_dir: Path) -> pd.Series:
    frame = pd.read_csv(run_dir / "returns.csv", index_col=0)
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_convert("America/New_York")
    frame.index = strip_sessions(idx)
    return frame["daily_return"].astype(float)


def load_spy_closes(data_dir: Path) -> tuple[pd.Series, list[str]]:
    files = sorted(data_dir.glob("SPY_1d_*.parquet"))
    if not files:
        raise SystemExit(
            f"no SPY_1d_*.parquet bar cache under {data_dir}; the market leg has no local source "
            "and a diagnostic never fetches"
        )
    pieces = [pd.read_parquet(f)["close"].astype(float) for f in files]
    closes = pd.concat(pieces).sort_index()
    closes = closes[~closes.index.duplicated(keep="last")]
    closes.index = strip_sessions(closes.index)
    return closes, [f.name for f in files]


def load_panel_closes(
    symbols: list[str], data_dir: Path, quarantine_dir: Path, suffix: str
) -> tuple[pd.DataFrame, list[str]]:
    """Close panel from the run's own bar caches, consensus-calendar filtered.

    Quarantined caches fall back to ``quarantine_dir`` because the certified
    run consumed those bars; the proxy should be the panel the book actually
    traded against, wrong instrument or not.
    """
    frames: dict[str, pd.Series] = {}
    missing: list[str] = []
    for sym in symbols:
        path = data_dir / f"{sym}{suffix}"
        if not path.exists():
            path = quarantine_dir / f"{sym}{suffix}"
        if not path.exists():
            missing.append(sym)
            continue
        bars = pd.read_parquet(path)
        if not bars.index.is_unique:
            bars = bars[~bars.index.duplicated(keep="last")]
        frames[sym] = bars["close"].astype(float).sort_index()
    closes = pd.DataFrame(frames).sort_index().dropna(how="all")
    closes = closes.loc[consensus_trading_days(closes)]
    closes.index = strip_sessions(closes.index)
    return closes, missing


def load_replay_returns(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    equity = pd.Series({r["decision_bar"]: float(r["equity"]) for r in rows}).sort_index()
    equity.index = strip_sessions(pd.DatetimeIndex(pd.to_datetime(equity.index)))
    return equity.pct_change().dropna()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _span(index: pd.DatetimeIndex) -> list[str] | None:
    if len(index) == 0:
        return None
    return [str(index[0].date()), str(index[-1].date())]


def _clean(obj):
    """NaN → null recursively so the JSON never carries a bare NaN token."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", default="results/demotion_b1", help="Finished run directory")
    parser.add_argument("--data_dir", default="data", help="Bar-cache directory")
    parser.add_argument(
        "--quarantine_dir",
        default="data/quarantine",
        help="Fallback cache directory for names quarantined after the run",
    )
    parser.add_argument(
        "--replay_dir",
        default="runs/replay_floor_1000000",
        help="Replay run directory for the cross-check stream (missing = cross-check null)",
    )
    parser.add_argument("--rolling_window", type=int, default=63, help="Rolling-beta window, sessions")
    parser.add_argument("--trailing_bars", type=int, default=21, help="Trailing market-return window, bars")
    parser.add_argument("--decile", type=float, default=0.10, help="Worst-tail quantile for conditional cells")
    parser.add_argument(
        "--drawdown_threshold", type=float, default=0.05, help="Book drawdown state threshold (fraction)"
    )
    parser.add_argument(
        "--min_obs",
        type=int,
        default=21,
        help="Minimum joint observations to report a beta; below it the cell is null with its n",
    )
    parser.add_argument(
        "--out", default=None, help="Output JSON (default: results/beta_telemetry_<today>.json)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())

    book = load_book_returns(run_dir)
    spy_closes, spy_files = load_spy_closes(data_dir)
    spy = spy_closes.pct_change(fill_method=None).dropna()

    suffix = f"_1d_{config['start_date']}_{config['end_date']}.parquet"
    panel, missing = load_panel_closes(
        [str(s) for s in config["symbols"]], data_dir, Path(args.quarantine_dir), suffix
    )
    proxy, names_per_day = equal_weight_returns(panel)

    markets = {"spy": spy, "ew_proxy": proxy}
    joint = {name: book.index.intersection(series.index) for name, series in markets.items()}

    coverage_note = None
    if len(spy) and spy.index[0] > book.index[0]:
        coverage_note = (
            f"local SPY coverage begins {spy.index[0].date()} but the certified book sample begins "
            f"{book.index[0].date()}; SPY cells measure only the overlap window — the earlier span "
            "(including the 2022 bear) is measured by the equal-weight panel proxy, never silently by SPY"
        )

    unconditional = {
        name: beta_cell(book, series, min_obs=args.min_obs) for name, series in markets.items()
    }
    rolling = {
        name: rolling_summary(rolling_beta(book, series, args.rolling_window), args.rolling_window)
        for name, series in markets.items()
    }

    cond_market: dict = {
        "definition": (
            f"beta over sessions whose prior-session trailing {args.trailing_bars}-bar market return "
            f"is in the worst {args.decile:.0%} of the joint sample (conditioning strictly prior)"
        )
    }
    for name, series in markets.items():
        mask, threshold, n_defined = trailing_worst_decile_mask(
            series, joint[name], args.trailing_bars, args.decile
        )
        cell = beta_cell(book, series, mask=mask, min_obs=args.min_obs)
        cell["threshold_trailing_return"] = threshold
        cell["n_conditioning_days_defined"] = n_defined
        cond_market[name] = cell

    dd_mask, drawdown = drawdown_state_mask(book, args.drawdown_threshold)
    cond_drawdown: dict = {
        "definition": (
            f"beta over sessions whose prior close had book equity more than "
            f"{args.drawdown_threshold:.0%} below its running peak"
        ),
        "n_state_days_full_book_sample": int(dd_mask.sum()),
        "max_drawdown_full_book_sample": float(drawdown.max()),
    }
    for name, series in markets.items():
        cond_drawdown[name] = beta_cell(book, series, mask=dd_mask, min_obs=args.min_obs)

    cond_months: dict = {
        "definition": (
            f"beta over sessions inside the book's worst-{args.decile:.0%} calendar months of the "
            "joint sample — ex-post attribution (month selected on the returns it conditions), "
            "not a tradeable state"
        )
    }
    for name, series in markets.items():
        mask, threshold, months, n_months = worst_month_mask(book, joint[name], args.decile)
        cell = beta_cell(book, series, mask=mask, min_obs=args.min_obs)
        cell["threshold_monthly_return"] = threshold
        cell["worst_months"] = months
        cell["n_months_in_joint_sample"] = n_months
        cond_months[name] = cell

    replay = load_replay_returns(Path(args.replay_dir) / "equity.jsonl")
    if replay is None:
        replay_block: dict = {
            "source": str(Path(args.replay_dir) / "equity.jsonl"),
            "note": "replay ledger not present; cross-check not run",
        }
    else:
        replay_block = {
            "source": str(Path(args.replay_dir) / "equity.jsonl"),
            "n_sessions": int(len(replay)),
            "span": _span(replay.index),
            "beta_vs_spy": beta_cell(replay, spy, min_obs=args.min_obs),
            "note": (
                "113-session replay stream through live-loop mechanics; too short to stand alone, "
                "read only as a cross-check against the certified-stream SPY cell"
            ),
        }

    payload = {
        "meta": {
            "generated": date.today().isoformat(),
            "script": "research/scripts/beta_telemetry.py",
            "status": (
                "uncounted diagnostic: searches nothing, changes no trial code path, appends "
                "nothing to the trials ledger, moves no ratified statistic"
            ),
            "run_dir": str(run_dir),
            "config_hash": summary.get("config_hash"),
            "certified_sharpe": summary.get("sharpe"),
            "parameters": {
                "rolling_window": args.rolling_window,
                "trailing_bars": args.trailing_bars,
                "decile": args.decile,
                "drawdown_threshold": args.drawdown_threshold,
                "min_obs": args.min_obs,
            },
        },
        "book": {
            "source": str(run_dir / "returns.csv"),
            "n_days": int(len(book)),
            "span": _span(book.index),
        },
        "market_series": {
            "spy": {
                "files": spy_files,
                "span": _span(spy.index),
                "n_days": int(len(spy)),
                "joint_span": _span(joint["spy"]),
                "n_joint": int(len(joint["spy"])),
                "coverage_note": coverage_note,
            },
            "ew_proxy": {
                "definition": (
                    "cap-blind equal-weight daily mean return over the certified run's own "
                    "bar-cache panel (quarantine fallback included) — a market proxy, not SPY"
                ),
                "n_names": int(panel.shape[1]),
                "missing_caches": missing,
                "names_per_day_median": float(names_per_day.median()),
                "names_per_day_min": int(names_per_day.min()),
                "joint_span": _span(joint["ew_proxy"]),
                "n_joint": int(len(joint["ew_proxy"])),
                "validation_vs_spy_overlap": beta_cell(proxy, spy, min_obs=args.min_obs),
            },
        },
        "unconditional": unconditional,
        "rolling": rolling,
        "conditional": {
            "worst_decile_market_21bar": cond_market,
            "book_drawdown_state": cond_drawdown,
            "worst_decile_book_months": cond_months,
        },
        "replay_cross_check": replay_block,
    }

    out_path = (
        Path(args.out)
        if args.out
        else Path("results") / f"beta_telemetry_{date.today().isoformat()}.json"
    )
    out_path.write_text(json.dumps(_clean(payload), indent=2))
    print(json.dumps(_clean(payload), indent=2))


if __name__ == "__main__":
    main()
