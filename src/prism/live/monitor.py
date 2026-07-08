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

from pathlib import Path

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
