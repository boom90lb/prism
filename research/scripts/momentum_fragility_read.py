"""Momentum M-series fragility read (docs/momentum_design.md §3, first clause).

Read-only over finished trial directories; appends to no ledger and moves no
ratified value. It computes the pre-committed fragility kill exactly as
registered, and refuses to adjudicate anything other than the registered set:

* **kill (a)** — the *median* net annualized Sharpe of M1–M5 under bucket
  spreads is negative;
* **kill (b)** — any single knob move flips the sign of the net result at a
  magnitude greater than B1's own point estimate. "Net result" is read as the
  same statistic the median clause uses (net annualized Sharpe); the same test
  on net total return is reported alongside as a secondary lens.

Guards (each raises): every trial directory must be a registered M1–M5 cell;
its ``config.json`` must differ from B1's in exactly the registered knob
(``design_trials`` excepted; M4's cadence lives in two config keys); the
ledger's output_dirs must be exactly B1 followed by the trials, in order.

Also reported, never adjudicated: the N6 breadth diagnostic flags (net and
gross) from each run's ``breadth_diagnostic.json``; the arithmetic mean
annualized daily return against the hurdle's annual basis (the periodic
hurdle test is vol-invariant, so this is the plain form of pass/fail); and a
uniform DSR recompute of every row against the *finished* ledger at
``N = max(design_trials, rows)`` — the per-row ``residual_set_dsr`` the driver
wrote is sequence-dependent (deflated against the rows present at run time).

The promotion clause (M6 + paper) is not readable here and is not computed.

    python -m research.scripts.momentum_fragility_read \
        --b1 results/demotion_b1 --trials results/momentum_m1,...,results/momentum_m5 \
        --ledger results/momentum_v1_trials.jsonl --out results/momentum_m_series_read.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.validation.metrics import deflated_sharpe_ratio_with_n

TRIAL_DELTAS = {
    "momentum_m1": "mom_skip_bars=0",
    "momentum_m2": "mom_lookback_bars=126",
    "momentum_m3": "mom_decile=0.2",
    "momentum_m4": "decision_every=63",
    "momentum_m5": "mom_skip_bars=42",
}
# The config keys each registered delta is allowed to move relative to B1.
REGISTERED_KEYS = {
    "momentum_m1": {"walk.mom_skip_bars"},
    "momentum_m2": {"walk.mom_lookback_bars"},
    "momentum_m3": {"walk.mom_decile"},
    "momentum_m4": {"walk.mom_decision_every", "signal.decision_every"},
    "momentum_m5": {"walk.mom_skip_bars"},
}
ALWAYS_ALLOWED = {"design_trials"}
DESIGN_TRIALS = 8


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = json.dumps(v, sort_keys=True) if isinstance(v, list) else v
    return out


def _returns(run_dir: Path) -> pd.Series:
    frame = pd.read_csv(run_dir / "returns.csv", index_col=0)
    return frame.iloc[:, 0].astype(float)


def _row(run_dir: Path, *, is_b1: bool = False) -> dict[str, Any]:
    if not is_b1 and run_dir.name not in TRIAL_DELTAS:
        raise SystemExit(f"{run_dir} is not a registered M1-M5 cell; refusing to adjudicate it")
    s = json.loads((run_dir / "summary.json").read_text())
    packet = json.loads((run_dir / "claim_packet.json").read_text())
    cfg = json.loads((run_dir / "config.json").read_text())
    walk = cfg["walk"]
    r = _returns(run_dir)
    hurdle_annual_pct = float(s["after_cost_hurdle"]["annual_pct"])
    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "delta": "B1 (M0 import)" if is_b1 else TRIAL_DELTAS[run_dir.name],
        "config_hash": s["config_hash"],
        "sharpe": float(s["sharpe"]),
        "total_return": float(s["total_return"]),
        "total_cost": float(s["total_cost"]),
        "gross_total_return": float(packet["metrics"]["gross_total_return"]),
        "arith_mean_return_annual_pct": float(r.mean() * 252.0 * 100.0),
        "hurdle_annual_pct": hurdle_annual_pct,
        "avg_turnover": float(s["avg_turnover"]),
        "max_drawdown": float(s["max_drawdown"]),
        "oos_periodic_sharpe": float(s["oos_periodic_sharpe"]),
        "periodic_hurdle": float(s["after_cost_hurdle"]["periodic_sharpe_hurdle"]),
        "clears_hurdle": float(s["oos_periodic_sharpe"]) > float(s["after_cost_hurdle"]["periodic_sharpe_hurdle"]),
        "dsr_as_written_sequence_dependent": s.get("residual_set_dsr"),
        "deflation_trials": s.get("deflation_trials"),
        "claim_tier": s["claim_tier"],
        "n_folds": s["n_folds"],
        "n_symbols": s["n_symbols"],
        "code_commit": packet.get("code_commit"),
        "knobs": {
            "mom_lookback_bars": walk["mom_lookback_bars"],
            "mom_skip_bars": walk["mom_skip_bars"],
            "mom_decile": walk["mom_decile"],
            "mom_decision_every": walk["mom_decision_every"],
            "decision_every": cfg["signal"]["decision_every"],
            "spread_mode": walk["spread_mode"],
            "band_mode": walk["band_mode"],
            "max_participation": walk["max_participation"],
        },
    }
    bd = run_dir / "breadth_diagnostic.json"
    if bd.exists():
        b = json.loads(bd.read_text())
        row["n6"] = {
            "n_eff": b["n_eff"]["book_contribution_participation_ratio"],
            "rank_ic": b["ic"]["mean_rank_ic_nonoverlap"],
            "ic_se": b["ic"]["se_nonoverlap"],
            "ic_n": b["ic"]["n_nonoverlap"],
            "ic_lower_95_one_sided": b["ic"]["lower_95_one_sided"],
            "ceiling": b["diagnostic_horizon_net"]["ir_ceiling"],
            "realized_net": b["diagnostic_horizon_net"]["realized_periodic_sharpe"],
            "realized_gross": b["diagnostic_horizon_gross"]["realized_periodic_sharpe"],
            "hurdle": b["diagnostic_horizon_net"]["after_cost_hurdle"],
            "falsification_net": bool(b["diagnostic_horizon_net"]["falsification"]),
            "falsification_gross": bool(b["diagnostic_horizon_gross"]["falsification"]),
            "capture_net": b["capture"]["net_over_ceiling"],
            "capture_gross": b["capture"]["gross_over_ceiling"],
        }
    return row


def check_single_knob(b1_dir: Path, run_dir: Path) -> dict[str, Any]:
    a = _flat(json.loads((b1_dir / "config.json").read_text()))
    b = _flat(json.loads((run_dir / "config.json").read_text()))
    moved = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    allowed = REGISTERED_KEYS[run_dir.name] | ALWAYS_ALLOWED
    if moved - allowed:
        raise SystemExit(f"{run_dir}: unregistered config keys moved vs B1: {sorted(moved - allowed)}")
    if not (moved & REGISTERED_KEYS[run_dir.name]):
        raise SystemExit(f"{run_dir}: registered knob {sorted(REGISTERED_KEYS[run_dir.name])} did not move")
    return {k: [a.get(k), b.get(k)] for k in sorted(moved)}


def fragility_read(b1: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    sharpes = [float(t["sharpe"]) for t in trials]
    median_sharpe = statistics.median(sharpes)
    b1_sharpe = float(b1["sharpe"])
    b1_ret = float(b1["total_return"])
    flips_sharpe = [
        t["run_dir"]
        for t in trials
        if (float(t["sharpe"]) < 0) != (b1_sharpe < 0) and abs(float(t["sharpe"])) > abs(b1_sharpe)
    ]
    flips_return = [
        t["run_dir"]
        for t in trials
        if (float(t["total_return"]) < 0) != (b1_ret < 0) and abs(float(t["total_return"])) > abs(b1_ret)
    ]
    kill_a = median_sharpe < 0
    kill_b = bool(flips_sharpe)
    return {
        "rule": "docs/momentum_design.md §3 fragility kill: median(net annualized Sharpe of M1-M5) < 0, "
        "or any single knob move flips the sign of the net result at magnitude > B1's point estimate",
        "b1_sharpe": b1_sharpe,
        "b1_total_return": b1_ret,
        "median_sharpe_m1_m5": median_sharpe,
        "min_sharpe_m1_m5": min(sharpes),
        "n_above_b1": sum(1 for x in sharpes if x > b1_sharpe),
        "kill_a_median_negative": kill_a,
        "kill_b_sign_flip_sharpe": kill_b,
        "kill_b_flipping_trials_sharpe": flips_sharpe,
        "secondary_sign_flip_total_return": flips_return,
        "n_trials_read": len(trials),
        "verdict": "KILL (fragility)" if (kill_a or kill_b) else "SURVIVES fragility read",
        "note": "Survival is not promotion: promotion reads only at M6 (extension >= 2027-06) + paper (§3).",
    }


def uniform_dsr(rows: list[dict[str, Any]], ledger_rows: list[dict[str, Any]], design_trials: int) -> dict[str, float]:
    trial_sharpes = np.asarray([float(r["oos_periodic_sharpe"]) for r in ledger_rows], dtype=float)
    n = max(design_trials, len(ledger_rows))
    out: dict[str, float] = {}
    for r in rows:
        out[r["run_dir"]] = float(deflated_sharpe_ratio_with_n(_returns(Path(r["run_dir"])).to_numpy(), trial_sharpes, n))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--b1", default="results/demotion_b1")
    p.add_argument("--trials", default=",".join(f"results/{k}" for k in TRIAL_DELTAS))
    p.add_argument("--ledger", default="results/momentum_v1_trials.jsonl")
    p.add_argument("--design_trials", type=int, default=DESIGN_TRIALS)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    b1_dir = Path(args.b1)
    trial_dirs = [Path(t) for t in args.trials.split(",") if t]
    if len(trial_dirs) != 5:
        raise SystemExit(f"fragility read needs exactly M1-M5, got {len(trial_dirs)}")
    b1 = _row(b1_dir, is_b1=True)
    trials = [_row(d) for d in trial_dirs]
    moved = {d.name: check_single_knob(b1_dir, d) for d in trial_dirs}

    ledger_rows = [json.loads(line) for line in Path(args.ledger).read_text().splitlines() if line.strip()]
    expected = [str(b1_dir)] + [str(d) for d in trial_dirs]
    actual = [r["output_dir"] for r in ledger_rows]
    if actual != expected:
        raise SystemExit(f"ledger output_dirs {actual} != registered set {expected}")
    ledger = {
        "path": args.ledger,
        "rows": len(ledger_rows),
        "output_dirs": actual,
        "config_hashes": [r["config_hash"] for r in ledger_rows],
        "note": "ledger config_hash is sha256(config payload)[:12]; the packet/summary config_hash is "
        "stable_config_hash({strategy, config, data}) — two digests of one configuration",
    }
    report = {
        "b1": b1,
        "trials": trials,
        "config_keys_moved_vs_b1": moved,
        "ledger": ledger,
        "dsr_uniform_vs_final_ledger": {
            "n_deflation": max(args.design_trials, len(ledger_rows)),
            "values": uniform_dsr([b1] + trials, ledger_rows, args.design_trials),
        },
        "read": fragility_read(b1, trials),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
