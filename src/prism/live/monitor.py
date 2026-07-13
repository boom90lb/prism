"""Anytime-valid monitor over the paper-loop equity ledger (SPEC.md §10, I-9).

The daily loop appends one mark-to-market NAV per decision bar to an equity
ledger (``prism.live.loop._append_equity_ledger``). This module turns that
accruing ledger into a live read: it forms the per-bar net-return series and
passes it through the time-uniform confidence sequence in
:mod:`prism.validation.anytime`, so the paper stream can be inspected after
every cycle and stopped at any bar without inflating type-I error — the temporal
analogue of the deflated-Sharpe's cross-sectional deflation.

**Governance.** Additive telemetry beside the ratified rolling-PSR promotion
read (``docs/momentum_design.md``): it moves no ratified statistic and starts no
counted trial, so it needs no new pre-registration. It becomes a binding
promotion/kill read only if a future program's pre-registration adopts it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from prism.live.loop import read_equity_ledger
from prism.validation.anytime import anytime_monitor_read

# Daily NAV returns of a paper book never approach ±10%; a bound tighter than the
# generic 0.5 default in prism.validation.anytime yields a more informative
# interval while staying winsorisation-safe (coverage holds for any bound).
DEFAULT_NAV_RETURN_BOUND = 0.1


def paper_monitor_read(
    equity_ledger_path: Path | str,
    *,
    hurdle: float = 0.0,
    alpha: float = 0.05,
    bound: float = DEFAULT_NAV_RETURN_BOUND,
    opt_horizon: int = 252,
) -> dict:
    """Anytime-valid verdict on the paper book's mean daily net return vs ``hurdle``.

    Loads the equity ledger, forms the per-bar net-return series
    (``equity_t / equity_{t-1} - 1`` over bars in chronological order), and reads
    it through :func:`prism.validation.anytime.anytime_monitor_read`. Returns that
    verdict dict augmented with ``n_equity_points`` and ``latest_equity``.

    With fewer than two equity points there is no return yet, so the verdict is
    ``inconclusive`` with ``n = 0`` — accruing, not failing.
    """
    df = read_equity_ledger(equity_ledger_path)
    n_points = int(len(df))
    if n_points < 2:
        return {
            "n": 0,
            "n_equity_points": n_points,
            "latest_equity": float(df["equity"].iloc[-1]) if n_points else float("nan"),
            "hurdle": float(hurdle),
            "alpha": float(alpha),
            "verdict": "inconclusive",
        }
    equity = df.sort_values("decision_bar")["equity"].astype(float)
    returns = equity.pct_change().dropna().to_numpy()
    read = anytime_monitor_read(
        returns, hurdle=hurdle, alpha=alpha, bound=bound, opt_horizon=opt_horizon
    )
    read["n_equity_points"] = n_points
    read["latest_equity"] = float(equity.iloc[-1])
    return read


def book_concordance(held_weights: pd.Series, target_weights: pd.Series) -> dict:
    """How faithfully the held book tracks the last refresh's target book.

    The promotion read (docs/momentum_design.md §3) is defined on the paper
    equity stream, but the stream is only evidence about the *strategy* to the
    extent the instrument actually holds the strategy's book — a 0.19-gross
    partial-fill book and a full B1 book produce indistinguishable monitor
    output. This read makes the gap first-order visible at any sample size:

    * ``active_share``  — ``0.5 * Σ|w_held − w_target|`` over the union of
      names (absent = explicit 0.0); 0 is a perfect replication, 1 is a
      disjoint book.
    * ``weight_corr``   — Pearson correlation of the two weight vectors over
      the union (``None`` when either side is degenerate).
    * ``gross_held`` / ``gross_target`` / ``gross_ratio`` — how much of the
      decided gross is actually deployed.

    Pure telemetry: it gates nothing and moves no ratified statistic.
    """
    union = held_weights.index.union(target_weights.index)
    held = held_weights.reindex(union).fillna(0.0).astype(float)
    target = target_weights.reindex(union).fillna(0.0).astype(float)
    gross_held = float(held.abs().sum())
    gross_target = float(target.abs().sum())
    corr: float | None = None
    if len(union) >= 2 and float(held.std()) > 0.0 and float(target.std()) > 0.0:
        raw_corr = float(held.corr(target))
        corr = raw_corr if math.isfinite(raw_corr) else None
    return {
        "active_share": float(0.5 * (held - target).abs().sum()),
        "weight_corr": corr,
        "gross_held": gross_held,
        "gross_target": gross_target,
        "gross_ratio": gross_held / gross_target if gross_target > 0.0 else None,
        "n_held": int((held != 0.0).sum()),
        "n_target": int((target != 0.0).sum()),
    }
