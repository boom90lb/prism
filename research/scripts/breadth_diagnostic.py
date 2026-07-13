"""FLAM breadth/ceiling diagnostic over a completed run's weight panel (SPEC N6/§10).

Reads a finished walk-forward run directory (``config.json`` + ``summary.json``
+ ``target_weights.csv`` + ``costs.csv``), rebuilds the price panel through the
same ``DataLoader`` path the trial used, and reports the fundamental-law
accounting the trial's claim packet does not carry:

* ``N_eff`` of the *traded book* — the participation ratio of the covariance of
  position-weighted daily return contributions (``w[t-1] * r[t]``), i.e. how
  many independent bets the book actually held, versus how many names it held.
* the top-eigenvalue shares of that covariance (a momentum decile book's crash
  tail is expected to concentrate here),
* rank IC of the signal at the decision horizon (non-overlapping bars, with a
  dense-bar reference), recomputed from the run's own config knobs, and
* ``fundamental_law_diagnostic`` at the decision-horizon frequency: ceiling
  ``|IC|·sqrt(N_eff)`` vs realized (net and gross) vs the run's recorded
  after-cost hurdle.

This is a *diagnostic*, not a counted trial: it searches nothing, changes no
trial code path, and writes only ``breadth_diagnostic.json`` into the run
directory. Signal reconstruction currently covers ``sleeve_mode ==
"momentum_only"`` runs (the B1 candidate); the breadth half is signal-agnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prism.residual.factors import ResidualStatArbConfig, consensus_trading_days
from prism.validation.metrics import (
    effective_breadth_from_cov,
    fundamental_law_diagnostic,
    rank_information_coefficient,
)
from research.arbitrage.residual_walk_forward import _momentum_scores
from research.arbitrage.walk_forward import StatArbWalkForwardConfig
from research.scripts.stat_arb_residual_wfo import _fetch_frames, _panel_matrices


def contribution_panel(weights: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Position-weighted daily return contributions, ``w[t-1] * r[t]``.

    ``weights`` are decided at close ``t`` and filled at open ``t+1`` (N2), so
    the weight row at ``t-1`` earns the close-to-close return into ``t`` — the
    same convention as the weight-space backtester. Rows are restricted to days
    where the prior book had any position (fold-formation gaps carry no bets
    and would only dilute the covariance); columns to names ever held. NaN
    contributions (a held name with no return, which the eligibility screen
    should preclude) stay NaN so a covariance over a broken panel fails loud
    rather than reading as a zero bet.
    """
    returns = closes.pct_change(fill_method=None).reindex(weights.index)
    prev = weights.shift(1)
    contrib = prev * returns[weights.columns]
    active = prev.abs().sum(axis=1) > 0.0
    ever_held = weights.columns[(weights != 0.0).any(axis=0)]
    return contrib.loc[active, ever_held]


def spectrum_shares(cov: pd.DataFrame, top: int = 3) -> list[float]:
    """Fractions of total variance carried by the ``top`` eigenvalues of ``cov``."""
    eig = np.linalg.eigvalsh(np.asarray(cov, dtype=float))
    eig = np.clip(eig[::-1], 0.0, None)
    total = float(eig.sum())
    if total <= 0.0:
        return [float("nan")] * top
    return [float(v / total) for v in eig[:top]]


def horizon_rank_ics(
    scores: pd.DataFrame,
    closes: pd.DataFrame,
    bars: pd.DatetimeIndex,
    horizon_bars: int,
) -> pd.Series:
    """Per-bar cross-sectional rank IC of ``scores`` vs ``horizon_bars``-forward returns.

    Evaluated only at ``bars`` (the caller picks dense or non-overlapping
    cadence). Bars whose forward window runs off the end of the panel, or with
    a degenerate cross-section, come back NaN and are dropped.
    """
    fwd = closes.shift(-horizon_bars) / closes - 1.0
    out: dict[pd.Timestamp, float] = {}
    for t in bars:
        if t not in scores.index or t not in fwd.index:
            continue
        out[t] = rank_information_coefficient(
            scores.loc[t].to_numpy(dtype=float), fwd.loc[t].to_numpy(dtype=float)
        )
    return pd.Series(out, dtype=float).dropna()


def _load_run(run_dir: Path) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    weights = pd.read_csv(run_dir / "target_weights.csv", index_col=0)
    weights.index = pd.DatetimeIndex(pd.to_datetime(weights.index, utc=True)).tz_convert("America/New_York")
    costs = pd.read_csv(run_dir / "costs.csv", index_col=0)
    costs.index = pd.DatetimeIndex(pd.to_datetime(costs.index, utc=True)).tz_convert("America/New_York")
    return config, summary, weights, costs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True, help="Finished run directory (config/summary/weights/costs)")
    parser.add_argument(
        "--horizon", type=int, default=None, help="IC forward horizon in bars (default: the run's decision cadence)"
    )
    parser.add_argument("--out", default=None, help="Output JSON path (default: <run_dir>/breadth_diagnostic.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config, summary, weights, costs = _load_run(run_dir)

    walk = config["walk"]
    if walk.get("sleeve_mode") != "momentum_only":
        raise SystemExit(
            f"Signal reconstruction implemented for sleeve_mode='momentum_only' runs; got {walk.get('sleeve_mode')!r}"
        )
    horizon = int(args.horizon or walk["mom_decision_every"])

    frames = _fetch_frames([str(s) for s in config["symbols"]], config["start_date"], config["end_date"])
    closes, _, volumes = _panel_matrices(frames)
    # The WFO evaluates on the consensus calendar (stray per-vendor holiday
    # bars would NaN the whole cross-section and bench every name for a full
    # corr_window under strict full-history eligibility) — reproduce it here or
    # the recomputed eligibility/scores are all-NaN.
    trading_days = consensus_trading_days(closes)
    closes = closes.loc[trading_days]
    volumes = volumes.loc[trading_days]
    missing = [c for c in weights.columns if c not in closes.columns]
    if missing:
        raise RuntimeError(f"Panel is missing {len(missing)} weight columns (cache drift?): {missing[:5]}")

    # --- breadth of the traded book -------------------------------------------------
    contrib = contribution_panel(weights, closes)
    cov = contrib.cov()
    n_eff_book = effective_breadth_from_cov(cov)
    held_per_day = (weights != 0.0).sum(axis=1)
    names_cov = closes[contrib.columns].pct_change(fill_method=None).reindex(contrib.index).cov()

    # --- IC at the decision horizon -------------------------------------------------
    signal_cfg = ResidualStatArbConfig(**{**config["signal"], "etf_symbols": tuple(config["signal"]["etf_symbols"])})
    walk_cfg = StatArbWalkForwardConfig(**walk)
    membership_file = (config.get("universe") or {}).get("membership_file")
    mask = None
    if membership_file:
        from prism.io.universe_sp500 import build_membership_mask

        mask = build_membership_mask(pd.read_parquet(membership_file), closes.index, list(closes.columns))
    scores = pd.DataFrame(
        _momentum_scores(closes, volumes, signal_cfg, walk_cfg, mask), index=closes.index, columns=closes.columns
    )
    traded_bars = weights.index[weights.index.isin(closes.index)]
    ic_nonoverlap = horizon_rank_ics(scores, closes, traded_bars[::horizon], horizon)
    ic_dense = horizon_rank_ics(scores, closes, traded_bars, horizon)
    ic = float(ic_nonoverlap.mean())
    ic_se = float(ic_nonoverlap.std(ddof=1) / np.sqrt(len(ic_nonoverlap))) if len(ic_nonoverlap) > 1 else float("nan")

    # --- fundamental-law accounting at the horizon frequency ------------------------
    # Daily Sharpes from the run scale to the decision horizon by sqrt(h) (iid
    # approximation, stated). Gross comes from the costs ledger's per-day gross
    # return column; net and the hurdle come from the summary.
    scale = float(np.sqrt(horizon))
    net_daily = float(summary["oos_periodic_sharpe"])
    # costs.csv's "gross"/"net" columns are exposures; the gross *return* is
    # the net daily return with the charged cost added back.
    net_ret = pd.read_csv(run_dir / "returns.csv", index_col=0)["daily_return"].to_numpy(dtype=float)
    gross_ret = net_ret + costs["total"].to_numpy(dtype=float)
    gross_daily = float(gross_ret.mean() / gross_ret.std(ddof=1))
    hurdle_daily = float(summary["after_cost_hurdle"]["periodic_sharpe_hurdle"])
    diag_net = fundamental_law_diagnostic(net_daily * scale, ic, n_eff_book, after_cost_hurdle=hurdle_daily * scale)
    diag_gross = fundamental_law_diagnostic(gross_daily * scale, ic, n_eff_book, after_cost_hurdle=hurdle_daily * scale)
    # SPEC §7.6: the viability gate reads the *lower* CI bound of IC, never the
    # point estimate. Normal-approx one-sided 95% lower bound on the mean of
    # the non-overlapping per-bar ICs (n is small; stated, not hidden).
    ic_lower = ic - 1.645 * ic_se if np.isfinite(ic_se) else float("nan")
    diag_ic_lower = fundamental_law_diagnostic(
        net_daily * scale, ic_lower, n_eff_book, after_cost_hurdle=hurdle_daily * scale
    )
    # Only the viability fields are meaningful at the lower-bound IC — the
    # falsification gate (realized > ceiling) compares against the *achievable*
    # ceiling and must be read at the point estimate, not a shrunk one.
    viability_at_ic_lower = {
        "information_coefficient": ic_lower,
        "ir_ceiling": diag_ic_lower["ir_ceiling"],
        "viable": diag_ic_lower["viable"],
        "viability_margin": diag_ic_lower["viability_margin"],
    }

    payload = {
        "run_dir": str(run_dir),
        "config_hash": summary.get("config_hash"),
        "horizon_bars": horizon,
        "n_eff": {
            "book_contribution_participation_ratio": n_eff_book,
            "held_names_return_participation_ratio": effective_breadth_from_cov(names_cov),
            "top_eigenvalue_shares": spectrum_shares(cov),
            "avg_names_held": float(held_per_day[held_per_day > 0].mean()),
            "n_ever_held": int(contrib.shape[1]),
            "n_active_days": int(contrib.shape[0]),
        },
        "ic": {
            "mean_rank_ic_nonoverlap": ic,
            "se_nonoverlap": ic_se,
            "n_nonoverlap": int(len(ic_nonoverlap)),
            "lower_95_one_sided": ic_lower,
            "mean_rank_ic_dense": float(ic_dense.mean()),
            "n_dense": int(len(ic_dense)),
        },
        "realized": {
            "net_daily_sharpe": net_daily,
            "gross_daily_sharpe": gross_daily,
            "hurdle_daily_sharpe": hurdle_daily,
            "sqrt_h_scaling": "daily Sharpes scaled by sqrt(horizon_bars) to the horizon frequency (iid approx)",
        },
        "diagnostic_horizon_net": diag_net,
        "diagnostic_horizon_gross": diag_gross,
        "viability_at_ic_lower_95": viability_at_ic_lower,
        "capture": {
            "net_over_ceiling": (net_daily * scale) / diag_net["ir_ceiling"] if diag_net["ir_ceiling"] else float("nan"),
            "gross_over_ceiling": (
                (gross_daily * scale) / diag_gross["ir_ceiling"] if diag_gross["ir_ceiling"] else float("nan")
            ),
        },
    }
    out_path = Path(args.out) if args.out else run_dir / "breadth_diagnostic.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
