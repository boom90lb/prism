"""Volatility-scaled momentum, replication route (registration ``registrations/2026-09-vol-momentum.toml``).

Design, criteria and limits: ``docs/vol_momentum_design.md``. One counted row in
``trials/vol_momentum_v1.jsonl``. The run refuses to start unless the
registration and design doc are committed unchanged at HEAD, and refuses any
input whose sha256 differs from the registered one.

Run from the repo root::

    python -m research.scripts.vol_momentum_replication
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.validation.compare import SharpeBootstrapSpec, sharpe_interval
from prism.validation.hurdle import tbill_hurdle
from prism.validation.trials import RunRecorder, TrialLog, stable_json_dumps
from research.scripts.factor_lab import (
    CACHE_DIR,
    LabError,
    annualized_stats,
    compound,
    fetch_sources,
    load_in_house,
    load_registration,
    parse_french,
    require_committed,
    scale_constant_vol,
)

REGISTRATION = Path("registrations/2026-09-vol-momentum.toml")
DESIGN_DOC = Path("docs/vol_momentum_design.md")
OUT_DIR = Path("results/2026-09-vol-momentum")
TRIAL_LOG = Path("trials/vol_momentum_v1.jsonl")


def ny_session_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Engine timestamps (``2022-03-29 00:00:00-04:00``) as naive New York session dates."""
    return pd.to_datetime(index, utc=True).tz_convert("America/New_York").normalize().tz_localize(None)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_hash(path: Path, expected: str, what: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise LabError(f"{what} at {path} has sha256 {actual}, registered {expected}; refusing (N7)")


# --------------------------------------------------------------------- series


def rule_scale(returns: np.ndarray, source: Mapping[str, Any]) -> np.ndarray:
    """The registered rule on a daily return array whose index 0 is a decision bar."""
    return scale_constant_vol(
        returns,
        refresh=int(source["refresh_sessions"]),
        target_vol=float(source["target_vol"]),
        lookback_sessions=int(source["lookback_sessions"]),
        max_scale=float(source["max_scale"]),
    )


def first_scored_index(source: Mapping[str, Any]) -> int:
    """The session after the first decision bar with a full lookback."""
    refresh, lookback = int(source["refresh_sessions"]), int(source["lookback_sessions"])
    first_decision = next(i for i in range(0, 10 * lookback, refresh) if i + 1 >= lookback)
    return first_decision + 1


def scaled_in_house(
    unscaled_net: pd.Series, gross: pd.Series, total_cost: pd.Series, source: Mapping[str, Any], change_cost_bps: float
) -> pd.DataFrame:
    """Overlay of the rule on the recorded book: return × scale, recorded cost × scale, plus scale-change cost."""
    if not (unscaled_net.index.equals(gross.index) and unscaled_net.index.equals(total_cost.index)):
        raise LabError("in-house returns and costs are not aligned session for session")
    values = unscaled_net.to_numpy(dtype=float)
    scale = rule_scale(values, source)
    previous = np.concatenate([[1.0], scale[:-1]])
    change_cost = np.abs(scale - previous) * gross.to_numpy(dtype=float) * change_cost_bps / 1e4
    net = scale * values - change_cost
    charged = scale * total_cost.to_numpy(dtype=float) + change_cost
    frame = pd.DataFrame({"scale": scale, "net": net, "charged_cost": charged, "unscaled_net": values},
                         index=unscaled_net.index)
    return frame.iloc[first_scored_index(source):]


def scaled_reference(momentum: pd.Series, source: Mapping[str, Any], warmup: int) -> pd.DataFrame:
    values = momentum.to_numpy(dtype=float)
    scale = rule_scale(values, source)
    frame = pd.DataFrame({"scale": scale, "gross": scale * values, "unscaled": values}, index=momentum.index)
    return frame.iloc[warmup:]


# --------------------------------------------------------------------- criteria


def publication_bar(reference: pd.Series, source: Mapping[str, Any], criteria: Mapping[str, Any]) -> dict[str, Any]:
    per_year = int(criteria["sessions_per_year"])
    full = annualized_stats(reference, per_year)
    post = annualized_stats(reference.loc[source["publication"] + "-01":], per_year)
    passes_bar = (
        full["t"] is not None
        and full["t"] >= float(criteria["publication_bar_t"])
        and full["years"] >= float(criteria["publication_bar_min_years"])
    )
    if post["sharpe"] is not None and post["years"] >= float(criteria["post_publication_min_years"]):
        prior, basis = post["sharpe"], "post_publication"
    else:
        prior, basis = float(criteria["pre_publication_haircut"]) * full["sharpe"], "haircut_full_sample"
    return {"full": full, "post_publication": post, "bar_pass": bool(passes_bar), "prior_sharpe": prior,
            "prior_basis": basis, "pass": bool(passes_bar and prior > 0)}


def fidelity(in_house: pd.Series, reference: pd.Series, floor: float) -> dict[str, Any]:
    joined = pd.concat([in_house.rename("in_house"), reference.rename("reference")], axis=1, join="inner").dropna()
    if len(joined) < 60:
        raise LabError(f"only {len(joined)} overlapping sessions; fidelity is undefined")
    daily = float(joined.corr().iloc[0, 1])
    weekly_frame = compound(joined, "W-FRI").dropna()
    return {
        "n": int(len(joined)),
        "first": str(joined.index[0].date()),
        "last": str(joined.index[-1].date()),
        "daily_corr": daily,
        "weekly_corr": float(weekly_frame.corr().iloc[0, 1]),
        "floor": floor,
        "pass": bool(daily >= floor),
    }


def economics(rho: float, prior: float, frame: pd.DataFrame, rf_annual: float, target_vol: float,
              per_year: int = 252, *, long_over_equity: float) -> dict[str, Any]:
    """Criterion 3 of the registration: ``rho * prior - cost drag`` against ``rf / target_vol``.

    The hurdle is the registered one: the book earns nothing on idle cash
    (``cash_yield=0``), so the idle fraction does not change the number and the
    shared :func:`prism.validation.hurdle.tbill_hurdle` returns
    ``rf_annual / target_vol`` exactly, reproducing the recorded 0.351 at rf
    4.21% and a 12% target. The book's measured long side over equity is still
    passed and recorded with it (``hurdle_inputs``), so the inputs travel with
    the hurdle rather than a placeholder; it has no default because a caller
    that does not know it should not be computing the hurdle. The volatility
    basis is the scaling rule's target, not the book's realized volatility; the
    two differed (12.0% vs 12.7%) in the recorded run.
    """
    vol = float(frame["net"].std(ddof=1) * math.sqrt(per_year))
    drag = float(frame["charged_cost"].mean() * per_year / vol)
    figure = rho * prior - drag
    result = tbill_hurdle(
        rf_annual, 0.0, long_over_equity, target_vol, periods_per_year=per_year,
        rf_source="French RF, mean of the last hurdle_window_sessions of the pinned 202607 file, annualized",
        cash_yield_source="registered convention: idle cash earns nothing (retail_v1 cash_income_bps_annual = 0)",
        vol_basis="target",
    )
    hurdle = result.sharpe_annual
    return {
        "hurdle_inputs": result.to_dict(),
        "rho_times_prior": rho * prior,
        "cost_drag_sharpe": drag,
        "charged_cost_annual": float(frame["charged_cost"].mean() * per_year),
        "scaled_vol_annual": vol,
        "figure_f3": figure,
        "rf_annual": rf_annual,
        "hurdle_sharpe": hurdle,
        "pass": bool(figure > hurdle),
        "reported_beside": {"hurdle_zero_if_cash_earns_bills": 0.0, "pass_at_zero": bool(figure > 0.0)},
    }


def no_contradiction(frame: pd.DataFrame, figure: float, boot: Mapping[str, Any], per_year: int = 252
                     ) -> dict[str, Any]:
    spec = SharpeBootstrapSpec(block_bars=int(boot["block_bars"]), n_resamples=int(boot["n_resamples"]),
                               seed=int(boot["seed"]), confidence=float(boot["confidence"]), decision_interval_bars=21)
    interval = sharpe_interval(frame["net"].to_numpy(), spec, label="scaled_in_house_net").to_dict()
    root = math.sqrt(per_year)
    annual = {key: interval[key] * root for key in ("sharpe_a", "lower", "upper")}
    return {"interval": interval, "annualized": annual, "figure_f3": figure,
            "pass": bool(annual["upper"] >= figure)}


# --------------------------------------------------------------------- driver


def run(reg: Mapping[str, Any], out: Path, log_path: Path, code_commit: str) -> dict[str, Any]:
    ref_cfg, impl = reg["reference"], reg["implementation"]
    fetched = fetch_sources({ref_cfg["momentum_source"]: {"url": ref_cfg["momentum_url"]},
                             ref_cfg["cash_source"]: {"url": ref_cfg["cash_url"]}}, CACHE_DIR)
    require_hash(Path(fetched[ref_cfg["momentum_source"]]["path"]), ref_cfg["momentum_sha256"], "French momentum")
    require_hash(Path(fetched[ref_cfg["cash_source"]]["path"]), ref_cfg["cash_sha256"], "French 3-factor")
    require_hash(Path(impl["returns"]), impl["returns_sha256"], "in-house returns")
    require_hash(Path(impl["costs"]), impl["costs_sha256"], "in-house costs")

    log = TrialLog(log_path)
    arm_id = "momentum_control:constant_vol_bsc"
    if log.exists() and any(row.get("arm_id") == arm_id for row in log.verify()):
        raise LabError(f"{arm_id} already has a row in {log_path}; the budget is one")
    config = {key: reg[key] for key in ("id", "selection_set", "route", "source", "reference", "criteria",
                                        "implementation")}
    recorder = RunRecorder(
        log=log,
        run_id=f"run-{uuid.uuid4().hex[:16]}",
        config_id=hashlib.sha256(stable_json_dumps(config).encode()).hexdigest(),
        data_id=hashlib.sha256("|".join([ref_cfg["momentum_sha256"], ref_cfg["cash_sha256"], impl["returns_sha256"],
                                         impl["costs_sha256"]]).encode()).hexdigest(),
        kind="new",
        selection_set_id=reg["selection_set"],
        arm_id=arm_id,
    )
    recorder.started()
    try:
        summary = evaluate(reg, fetched)
    except Exception as exc:
        recorder.failed(f"{type(exc).__name__}: {exc}")
        raise
    summary["code_commit"] = code_commit
    series = summary.pop("_series")
    recorder.finished({
        "promoted": summary["promoted"],
        "criteria": {name: summary[name]["pass"] for name in ("publication_bar", "fidelity", "economics",
                                                              "no_contradiction")},
        "scaled_in_house_sharpe": summary["no_contradiction"]["annualized"]["sharpe_a"],
        "code_commit": code_commit,
    })
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    series.to_csv(out / "series.csv")
    return summary


def evaluate(reg: Mapping[str, Any], fetched: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    source, ref_cfg, criteria, impl = reg["source"], reg["reference"], reg["criteria"], reg["implementation"]
    per_year = int(criteria["sessions_per_year"])

    momentum_table, momentum_vintage = parse_french(Path(fetched[ref_cfg["momentum_source"]]["path"]).read_bytes())
    cash_table, _ = parse_french(Path(fetched[ref_cfg["cash_source"]]["path"]).read_bytes())
    momentum = momentum_table[ref_cfg["momentum_column"]].dropna()
    reference = scaled_reference(momentum, source, int(ref_cfg["warmup_sessions"]))

    unscaled = load_in_house(Path(impl["returns"]))
    costs = pd.read_csv(impl["costs"], index_col=0)
    costs.index = ny_session_dates(costs.index)
    if str(unscaled.index[0].date()) != impl["refresh_phase_start"]:
        raise LabError(f"in-house series starts {unscaled.index[0].date()}, registered {impl['refresh_phase_start']}")
    in_house = scaled_in_house(unscaled, costs["gross"], costs["total"], source, float(impl["scale_change_cost_bps"]))

    bar = publication_bar(reference["gross"], source, criteria)
    fid = fidelity(in_house["net"], reference["gross"], float(criteria["fidelity_floor"]))
    rf_annual = float(cash_table[ref_cfg["cash_column"]].dropna().iloc[-int(criteria["hurdle_window_sessions"]):]
                      .mean() * per_year)
    # L/E of the scaled book: the unscaled book's long side is half its ~1.0x recorded gross (costs.csv gross
    # mean 1.0016), times the mean scale; the convention of docs/hurdle_restatement_2026-09-30.md.
    econ = economics(fid["daily_corr"], bar["prior_sharpe"], in_house, rf_annual, float(source["target_vol"]),
                     per_year, long_over_equity=0.5 * float(in_house["scale"].mean()))
    contra = no_contradiction(in_house, econ["figure_f3"], reg["bootstrap"], per_year)
    promoted = all(part["pass"] for part in (bar, fid, econ, contra))

    series = pd.concat({"reference": reference[["scale", "gross"]],
                        "in_house": in_house[["scale", "net", "charged_cost", "unscaled_net"]]}, axis=1)
    series.columns = [f"{a}|{b}" for a, b in series.columns]
    return {
        "registration": reg["id"],
        "selection_set": reg["selection_set"],
        "route": reg["route"],
        "momentum_vintage": momentum_vintage,
        "publication_bar": bar,
        "fidelity": fid,
        "economics": econ,
        "no_contradiction": contra,
        "in_house": {
            "first": str(in_house.index[0].date()),
            "last": str(in_house.index[-1].date()),
            "n": int(len(in_house)),
            "mean_scale": float(in_house["scale"].mean()),
            "scale_min": float(in_house["scale"].min()),
            "scale_max": float(in_house["scale"].max()),
            "scaled": annualized_stats(in_house["net"], per_year),
            "unscaled_same_window": annualized_stats(in_house["unscaled_net"], per_year),
            "n_eff": impl["n_eff_unscaled"],
        },
        "promoted": bool(promoted),
        "_series": series,
    }


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description="Volatility-scaled momentum, replication route.").parse_args(argv)
    reg = load_registration(REGISTRATION)
    code_commit = require_committed([REGISTRATION, DESIGN_DOC])
    summary = run(reg, OUT_DIR, TRIAL_LOG, code_commit)
    print(json.dumps({name: summary[name]["pass"] for name in ("publication_bar", "fidelity", "economics",
                                                                "no_contradiction")} | {"promoted": summary["promoted"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
