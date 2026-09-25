"""Read a registered selection set: comparison, deflation, breadth, and spanning, from its own records.

Usage::

    python -m research.scripts.registered_read --registration registrations/2026-09-characteristics.toml \
        --dataset data/frozen/demotion_b1 --out results/2026-09-characteristics

What it reads. The selection set's trial log (``trials/<selection_set>.jsonl``):
the latest ``finished`` row of each declared arm under this registration id,
and the run record each names (``results/<run_id>/record.json``). Every number
below is recomputed from those records' own return and weight artifacts; the
selection is the registration's arm list, so the read cannot pick rows.

What it writes under ``--out``:

* ``comparison.json`` / ``comparison.txt`` -- the paired studentized Sharpe
  differences of every arm pair (``prism.scripts.construction_experiment
  .build_comparison``, the construction read's own code path);
* ``read.json`` -- per arm: the record's net Sharpe, gross Sharpe, return,
  volatility, drawdown, turnover and charged cost; the Deflated Sharpe Ratio
  against this set (every declared arm, ``deflated_sharpe_ratio_with_n``);
  ``N_eff`` as the participation ratio of the covariance of position-weighted
  daily return contributions, the retired diagnostic's definition
  (``git show c08e3d0^:research/scripts/breadth_diagnostic.py``); a held-book
  rank IC; and, when ``--spanning`` is given, the regression of an arm's daily
  net returns on a sibling's over their common dates with a Newey-West
  intercept t-statistic;
* ``returns/<arm>.csv`` -- each arm's daily net return series, so a later
  reader can re-deflate or re-span without the ignored run directories.

A diagnostic over finished records; it searches nothing and appends no trial row.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.io.dataset import Dataset
from prism.registration import load_registration
from prism.scripts.construction_experiment import build_comparison, report_text
from prism.validation.metrics import (
    deflated_sharpe_ratio_with_n,
    effective_breadth_from_cov,
    probabilistic_sharpe_ratio,
    rank_information_coefficient,
)
from prism.workspace import Workspace, resolve_workspace

HORIZON = 21


def finished_records(workspace: Workspace, registration: Any) -> dict[str, Path]:
    """The latest finished record of each declared arm under this registration."""
    path = workspace.trial_log(registration.selection_set)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("event") != "finished" or row.get("arm_id") not in registration.arm_ids:
            continue
        latest[str(row["arm_id"])] = row
    missing = [arm for arm in registration.arm_ids if arm not in latest]
    if missing:
        raise SystemExit(f"arms without a finished row in {path}: {missing}")
    out: dict[str, Path] = {}
    for arm, row in latest.items():
        record = workspace.root / "results" / str(row["run_id"]) / "record.json"
        if not record.exists():
            raise SystemExit(f"{arm}: finished row names {row['run_id']} but {record} is absent")
        out[arm] = record
    return {arm: out[arm] for arm in registration.arm_ids}


def _series(path: Path, column: str | None = None) -> pd.DataFrame | pd.Series:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_convert("America/New_York")
    return frame if column is None else frame[column]


def contribution_breadth(weights: pd.DataFrame, closes: pd.DataFrame) -> dict[str, Any]:
    """N_eff of the traded book: participation ratio of cov(w[t-1] * r[t]) over active days and held names.

    The retired diagnostic's definition exactly (``git show
    c08e3d0^:research/scripts/breadth_diagnostic.py``, ``contribution_panel``):
    NaN contributions stay NaN and the covariance is pairwise. On the certified
    B1 record this reproduces the recorded 52.74755156170981 to 1e-13 (Run,
    2026-09-23); filling NaN with zero instead gives 55.2, so the filling is
    part of the estimator and is not changed here.
    """
    returns = closes.pct_change(fill_method=None).reindex(weights.index)
    prev = weights.shift(1)
    held = weights.columns[(weights != 0.0).any(axis=0)]
    contrib = (prev * returns[weights.columns]).loc[prev.abs().sum(axis=1) > 0.0, held]
    cov = contrib.cov()
    eig = np.clip(np.linalg.eigvalsh(cov.to_numpy())[::-1], 0.0, None)
    names_held = (weights != 0.0).sum(axis=1)
    return {
        "n_eff_book_contribution": float(effective_breadth_from_cov(cov)),
        "top_eigenvalue_shares": [float(v / eig.sum()) for v in eig[:3]] if eig.sum() > 0 else None,
        "avg_names_held": float(names_held[names_held > 0].mean()),
        "n_ever_held": int(len(held)),
        "n_active_days": int(len(contrib)),
    }


def held_book_ic(decisions: pd.DataFrame, closes: pd.DataFrame) -> dict[str, Any]:
    """Rank IC of target weight vs the next 21-session close-to-close return, over held names, per decision.

    ``decisions`` holds one row per decision bar; ``closes`` is the daily
    session panel, so the forward return is 21 sessions, not 21 rows of the
    decision index.
    """
    forward = closes.shift(-HORIZON) / closes - 1.0
    ics: list[float] = []
    for t in decisions.index:
        if t not in forward.index:
            continue
        row = decisions.loc[t]
        names = row.index[row.fillna(0.0) != 0.0]
        value = rank_information_coefficient(row[names].to_numpy(float), forward.loc[t, names].to_numpy(float))
        if math.isfinite(value):
            ics.append(value)
    series = pd.Series(ics, dtype=float)
    se = float(series.std(ddof=1) / math.sqrt(len(series))) if len(series) > 1 else float("nan")
    return {"mean": float(series.mean()) if len(series) else float("nan"), "se": se, "n": int(len(series)),
            "note": "held names only (long leg vs short leg separation), one cross-section per decision bar"}


def daily_targets(decisions: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Decision-bar targets held on every session until the next decision, as the certified B1 record wrote them.

    The canonical engine records one target row per decision bar; the retired
    diagnostic read one row per session. Sessions before the first decision
    carry no target. The breadth estimator below is defined on the daily
    panel, so the expansion is part of reading it, not a modelling choice.
    """
    expanded = decisions.reindex(sessions.union(decisions.index)).ffill().reindex(sessions)
    return expanded.fillna(0.0)


def newey_west_intercept(y: pd.Series, x: pd.Series, lags: int) -> dict[str, float]:
    """OLS of y on [1, x] with a Newey-West (Bartlett) covariance; returns the intercept and slope stats."""
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    Y = frame["y"].to_numpy(float)
    X = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]
    S = scores.T @ scores
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = scores[lag:].T @ scores[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return {
        "n_obs": int(len(frame)),
        "alpha_daily": float(beta[0]),
        "alpha_annualized": float(beta[0] * 252.0),
        "alpha_t_newey_west": float(beta[0] / se[0]),
        "beta": float(beta[1]),
        "beta_t_newey_west": float(beta[1] / se[1]),
        "correlation": float(frame["y"].corr(frame["x"])),
        "lags": lags,
    }


def comparison_or_skip(workspace: Workspace, registration: Any, records: dict[str, Path]) -> tuple[dict[str, Any], str]:
    """The pairwise arm comparison, or a recorded skip for a registration with one arm (nothing to pair)."""
    if len(records) < 2:
        note = f"one arm ({', '.join(records)}): a single-arm registration has no pairwise comparison"
        return {"skipped": note, "arms": list(records)}, note
    comparison = build_comparison(workspace=workspace, registration=registration, records=list(records.values()))
    return comparison, report_text(comparison)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workspace", default=None)
    parser.add_argument(
        "--spanning",
        nargs=2,
        action="append",
        default=[],
        metavar=("ARM", "SIBLING_RETURNS_CSV"),
        help="regress ARM's net returns on a sibling's returns.csv (repeatable)",
    )
    parser.add_argument("--hac_lags", type=int, default=HORIZON)
    args = parser.parse_args(argv)

    workspace = resolve_workspace(args.workspace)
    registration = load_registration(args.registration, workspace=workspace)
    records = finished_records(workspace, registration)
    out = Path(args.out)
    (out / "returns").mkdir(parents=True, exist_ok=True)

    comparison, comparison_text = comparison_or_skip(workspace, registration, records)
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True, default=str) + "\n")
    (out / "comparison.txt").write_text(comparison_text + "\n")

    dataset = Dataset.open(Path(args.dataset))
    net: dict[str, pd.Series] = {}
    arms: dict[str, dict[str, Any]] = {}
    for arm, record_path in records.items():
        record = json.loads(record_path.read_text())
        run_dir = record_path.parent
        r = _series(run_dir / "returns.csv", "daily_return")
        costs = _series(run_dir / "costs.csv")
        decisions = _series(run_dir / "target_weights.csv")
        r.rename("daily_return").to_csv(out / "returns" / f"{arm}.csv")
        net[arm] = r
        gross = r + costs["total"].reindex(r.index).fillna(0.0)
        end = str(r.index[-1].date())
        view = dataset.view("research_vintage", end, vintage=dataset.manifest.vintage)
        panel = view.panel(list(decisions.columns), end, "close")
        closes = panel.reindex(index=r.index)
        weights = daily_targets(decisions, r.index) if len(decisions) < len(r) else decisions
        perf = record["performance"]
        years = len(r) / 252.0
        arms[arm] = {
            "run_id": record["ids"].get("run_id"),
            "config_id": record["ids"].get("config_id"),
            "evidence_tier": record["flags"].get("evidence_tier"),
            "n_obs": int(len(r)),
            "net_sharpe_annualized": perf["sharpe"]["annualized"],
            "net_sharpe_periodic": perf["sharpe"]["periodic"],
            "gross_sharpe_annualized": float(gross.mean() / gross.std(ddof=1) * math.sqrt(252.0)),
            "annualized_return": float((1.0 + r).prod() ** (1.0 / years) - 1.0) if years > 0 else float("nan"),
            "annualized_vol": float(r.std(ddof=1) * math.sqrt(252.0)),
            "max_drawdown": perf["drawdown"]["max_drawdown"],
            "turnover_total": perf["turnover"]["total"],
            "charged_cost_total": perf["cost"]["total"],
            "psr_vs_zero": float(probabilistic_sharpe_ratio(r.to_numpy(float), 0.0)),
            "breadth": contribution_breadth(weights, closes),
            "held_book_ic": held_book_ic(decisions, panel),
            "n_decision_rows": int(len(decisions)),
            "excluded_folds": [row.get("index") for row in record["evaluation"].get("excluded_folds", [])],
        }
    trial_sharpes = [float(net[arm].mean() / net[arm].std(ddof=1)) for arm in records]
    for arm in records:
        arms[arm]["dsr_vs_set"] = float(
            deflated_sharpe_ratio_with_n(net[arm].to_numpy(float), trial_sharpes, max(len(trial_sharpes), registration.trial_budget))
        )
    spanning: dict[str, Any] = {}
    for arm, sibling in args.spanning:
        sib = _series(Path(sibling), "daily_return")
        common = net[arm].index.intersection(sib.index)
        spanning[f"{arm}~{sibling}"] = newey_west_intercept(net[arm].loc[common], sib.loc[common], args.hac_lags)
    payload = {
        "registration": {"id": registration.id, "selection_set": registration.selection_set,
                          "trial_budget": registration.trial_budget, "arms": list(registration.arm_ids)},
        "dataset": {"root": str(args.dataset), "data_id": dataset.manifest.id},
        "records": {arm: str(path) for arm, path in records.items()},
        "deflation": {"method": "deflated_sharpe_ratio_with_n", "n_trials": max(len(trial_sharpes), registration.trial_budget),
                      "trial_periodic_sharpes": dict(zip(records, trial_sharpes))},
        "arms": arms,
        "spanning": spanning,
    }
    (out / "read.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({arm: {k: arms[arm][k] for k in ("net_sharpe_annualized", "dsr_vs_set", "annualized_return",
                                                       "max_drawdown")} | {"n_eff": arms[arm]["breadth"]["n_eff_book_contribution"]}
                      for arm in arms}, indent=2))
    print(comparison_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
