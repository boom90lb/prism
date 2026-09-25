"""Earnings-announcement reads beside the replication criteria (docs/earnings_design.md §6-§7).

Two parts, both fixed before any earnings arm ran:

1. **The discovery set's kill rule** (``registrations/2026-09-earnings-daily.toml``). Per
   arm, the studentized 95% interval for net periodic Sharpe (42-session circular
   blocks, daily decisions); an arm whose interval lies entirely below zero is dead.
   Beside it, each arm's daily correlation with the promoted vol-scaled momentum's
   in-house net return, since SPEC §5.5 judges a fast signal by its effect on the
   combined book.
2. **The horizon profile** SPEC §5.5 asks of every signal, for all three earnings
   scores: the cross-sectional rank IC of each session's score with the open-to-open
   return from the next open over the following 1, 2, 5, 10, 21 and 63 sessions, over
   that session's index members, averaged over sessions with a Newey-West standard
   error (lag = horizon - 1, for the overlap).

Reads the registered read of the daily registration, the frozen announcement panel and
the frozen price dataset; writes ``results/2026-09-earnings-daily/earnings_read.json``.
Refuses to start unless the registrations, books and design doc are committed unchanged.
Run from the repo root after both registered reads::

    python -m research.scripts.earnings_read
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.validation.compare import SharpeBootstrapSpec, sharpe_interval
from research.scripts.factor_lab import LabError, require_committed
from research.scripts.vol_momentum_replication import require_hash

DAILY_REGISTRATION = Path("registrations/2026-09-earnings-daily.toml")
MONTHLY_REGISTRATION = Path("registrations/2026-09-earnings.toml")
BOOKS = (Path("examples/earnings/daily.toml"), Path("examples/earnings/monthly.toml"))
DESIGN_DOC = Path("docs/earnings_design.md")
DAILY_READ = Path("results/2026-09-earnings-daily")
OUT = DAILY_READ / "earnings_read.json"
DATASET = Path("data/frozen/demotion_b1")
PANEL = Path("data/frozen/earnings_v1/panel.parquet")
PANEL_SHA256 = "d4703a6cc1374e862d398df1a5bc4069e4e04f0ec1fc1fe9c81614a0fc3d701e"
MOMENTUM_SERIES = Path("results/2026-09-vol-momentum/series.csv")
MOMENTUM_SHA256 = "ded96a7226e7ae70c71422741a46a3ec1620442dbbb0c2e38aae64f49db37395"
DAILY_ARMS = ("ear_event", "announcement_premium")
KILL_BOOTSTRAP = {"block_bars": 42, "n_resamples": 2000, "seed": 20260929, "confidence": 0.95}
HORIZONS = (1, 2, 5, 10, 21, 63)
MIN_NAMES = 30
FIRST_SESSION = pd.Timestamp("2021-03-30")  # the registrations' first decision
SIGNALS = {  # score name -> (panel column, maximum age in sessions or None, sign)
    "ear_monthly": ("ear", 126, 1.0),
    "ear_event": ("ear", 60, 1.0),
    "announcement_premium": ("days_to_expected", None, -1.0),
}


# --------------------------------------------------------------------- kill rule


def kill_read(net: pd.Series, boot: Mapping[str, Any] = KILL_BOOTSTRAP, per_year: int = 252) -> dict[str, Any]:
    """The studentized interval for one arm's net periodic Sharpe; dead when it lies entirely below zero."""
    spec = SharpeBootstrapSpec(block_bars=int(boot["block_bars"]), n_resamples=int(boot["n_resamples"]),
                               seed=int(boot["seed"]), confidence=float(boot["confidence"]), decision_interval_bars=1)
    interval = sharpe_interval(net.dropna().to_numpy(float), spec, label="net").to_dict()
    root = math.sqrt(per_year)
    return {"interval": interval, "annualized": {k: interval[k] * root for k in ("sharpe_a", "lower", "upper")},
            "killed": bool(interval["upper"] < 0.0)}


def momentum_correlation(net: pd.Series, momentum_net: pd.Series) -> dict[str, Any]:
    joined = pd.concat({"arm": net, "momentum": momentum_net}, axis=1).dropna()
    return {"daily_corr": float(joined["arm"].corr(joined["momentum"])), "n": int(len(joined)),
            "first": str(joined.index[0].date()), "last": str(joined.index[-1].date())}


# --------------------------------------------------------------------- horizon profile


def forward_open_returns(opens: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Row t: the open-to-open return from session t+1's open to session t+1+horizon's open (N2)."""
    return opens.shift(-(1 + horizon)) / opens.shift(-1) - 1.0


def rank_ic(scores: pd.DataFrame, forward: pd.DataFrame, min_names: int = MIN_NAMES) -> pd.Series:
    """Per-session Spearman correlation of score with forward return over names carrying both."""
    both = scores.notna() & forward.notna()
    s = scores.where(both).rank(axis=1)
    f = forward.where(both).rank(axis=1)
    s = s.sub(s.mean(axis=1), axis=0)
    f = f.sub(f.mean(axis=1), axis=0)
    ic: pd.Series = (s * f).sum(axis=1) / np.sqrt((s**2).sum(axis=1) * (f**2).sum(axis=1))
    return ic.where(both.sum(axis=1) >= min_names)


def newey_west_mean(series: pd.Series, lags: int) -> dict[str, float]:
    values = series.dropna().to_numpy(float)
    n = len(values)
    if n < 2:
        return {"mean": math.nan, "se": math.nan, "t": math.nan, "n": n}
    centered = values - values.mean()
    variance = float(centered @ centered) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * float(centered[lag:] @ centered[:-lag]) / n
    se = math.sqrt(max(variance, 0.0) / n)
    return {"mean": float(values.mean()), "se": se, "t": float(values.mean() / se) if se > 0 else math.nan, "n": n}


def signal_scores(panel: pd.DataFrame, column: str, max_age: int | None, sign: float) -> pd.DataFrame:
    frame = panel if max_age is None else panel[panel["age_sessions"] <= max_age]
    return sign * frame.pivot(index="date", columns="sid", values=column).sort_index()


def horizon_profile(panel: pd.DataFrame, opens: pd.DataFrame, members: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (column, max_age, sign) in SIGNALS.items():
        scores = signal_scores(panel, column, max_age, sign).reindex(index=opens.index, columns=opens.columns)
        scores = scores.where(members.reindex(index=scores.index, columns=scores.columns, fill_value=False))
        scores = scores.loc[scores.index >= FIRST_SESSION]
        by_horizon = {}
        for h in HORIZONS:
            ic = rank_ic(scores, forward_open_returns(opens, h).reindex(scores.index))
            by_horizon[str(h)] = newey_west_mean(ic, lags=max(h - 1, 0))
        out[name] = by_horizon
    return out


# --------------------------------------------------------------------- inputs


def load_opens_and_members(dataset: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    listings = pd.read_parquet(dataset / "listings.parquet")
    opens = {}
    for sid in listings["sid"].astype(str):
        path = dataset / "bars" / f"{sid}_1d.parquet"
        if path.exists():
            bars = pd.read_parquet(path, columns=["ts", "open"])
            index = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True)).tz_convert("America/New_York")
            opens[sid] = pd.Series(bars["open"].to_numpy(float), index=index.tz_localize(None).normalize())
    frame = pd.DataFrame(opens).sort_index()
    membership = pd.read_parquet(dataset / "membership.parquet")
    members = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    starts = pd.to_datetime(membership["valid_from"]).fillna(frame.index[0])
    for sid, start, end in zip(membership["sid"].astype(str), starts, pd.to_datetime(membership["valid_to"])):
        if sid in members.columns:
            members.loc[(frame.index >= start) & (frame.index <= end), sid] = True
    return frame, members


def run(code_commit: str) -> dict[str, Any]:
    if hashlib.sha256(PANEL.read_bytes()).hexdigest() != PANEL_SHA256:
        raise LabError(f"{PANEL} differs from the registered panel (N7)")
    require_hash(MOMENTUM_SERIES, MOMENTUM_SHA256, "momentum record")
    momentum = pd.read_csv(MOMENTUM_SERIES, index_col=0, parse_dates=True)["in_house|net"].dropna()
    arms = {}
    for arm in DAILY_ARMS:
        path = DAILY_READ / "returns" / f"{arm}.csv"
        if not path.exists():
            raise LabError(f"{path} is missing; run the daily registration's registered read first")
        returns = pd.read_csv(path, index_col=0)
        index = pd.DatetimeIndex(pd.to_datetime(returns.index, utc=True)).tz_convert("America/New_York")
        net = pd.Series(returns.iloc[:, 0].to_numpy(float), index=index.tz_localize(None).normalize())
        arms[arm] = {"kill": kill_read(net), "momentum": momentum_correlation(net, momentum)}
    panel = pd.read_parquet(PANEL, columns=["date", "sid", "ear", "age_sessions", "days_to_expected"])
    opens, members = load_opens_and_members(DATASET)
    summary = {"code_commit": code_commit, "panel_sha256": PANEL_SHA256, "arms": arms,
               "killed": sorted(a for a, r in arms.items() if r["kill"]["killed"]),
               "horizon_profile": horizon_profile(panel, opens, members),
               "horizons": list(HORIZONS), "min_names": MIN_NAMES, "first_session": str(FIRST_SESSION.date())}
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description="Earnings-announcement reads beside the replication criteria.").parse_args(argv)
    code_commit = require_committed([DAILY_REGISTRATION, MONTHLY_REGISTRATION, *BOOKS, DESIGN_DOC])
    summary = run(code_commit)
    print(json.dumps({"killed": summary["killed"],
                      "momentum_corr": {a: r["momentum"]["daily_corr"] for a, r in summary["arms"].items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
