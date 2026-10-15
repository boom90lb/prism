"""Fundamentals, replication route (registrations ``2026-09-fundamentals*.toml``).

Design, criteria and limits: ``docs/fundamentals_design.md``. Reads the six engine arms'
records (``research.scripts.registered_read`` output plus each run's ``costs.csv``), the
pinned JKP reference series and French cash series, and decides:

* **per arm**, the four SPEC §4 replication criteria: publication bar and prior on the
  JKP series, daily tracking of the in-house arm against it, economics at our costs
  against the T-bill hurdle at the volatility the arm can run at, and no in-house
  contradiction;
* **the pinned combination** (SPEC §4, Combinations): the arms passing criteria 1 and 2
  plus the promoted volatility-scaled momentum, each scaled to 12% volatility, averaged
  at equal weights and scaled to 12% again under the 2x gross ceiling; its prior from the
  same combination of public series.

Refuses to start unless both registrations, the control spec and the design doc are
committed unchanged, and refuses any input whose sha256 differs from the registered one.
Appends one trial row, for the combination, to its own log. Run from the repo root::

    python -m research.scripts.fundamentals_replication
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

from prism.validation.trials import RunRecorder, TrialLog, stable_json_dumps
from research.scripts.factor_lab import (
    CACHE_DIR,
    LabError,
    _effective,
    annualized_stats,
    fetch_sources,
    load_registration,
    parse_french,
    parse_jkp,
    refresh_mask,
    require_committed,
)
from research.scripts.vol_momentum_replication import (
    economics,
    fidelity,
    first_scored_index,
    no_contradiction,
    publication_bar,
    require_hash,
    rule_scale,
    scaled_in_house,
)

REGISTRATION = Path("registrations/2026-09-fundamentals-replication.toml")
ENGINE_REGISTRATION = Path("registrations/2026-09-fundamentals.toml")
CONTROL_SPEC = Path("examples/fundamentals/control.toml")
DESIGN_DOC = Path("docs/fundamentals_design.md")
OUT_DIR = Path("results/2026-09-fundamentals/replication")


def ny_dates(index: pd.Index) -> pd.DatetimeIndex:
    return pd.to_datetime(index, utc=True).tz_convert("America/New_York").normalize().tz_localize(None)


# --------------------------------------------------------------------- inputs


def arm_records(engine_read: Path, trial_log: Path, arms: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Per arm: daily net return, charged cost and gross exposure, from its latest finished run."""
    finished: dict[str, str] = {}
    for row in TrialLog(trial_log).verify():
        if row.get("event") == "finished" and row.get("arm_id") in arms:
            finished[row["arm_id"]] = row["run_id"]
    out = {}
    for arm in arms:
        if arm not in finished:
            raise LabError(f"arm {arm} has no finished run in {trial_log}")
        returns = pd.read_csv(engine_read / "returns" / f"{arm}.csv", index_col=0)
        costs = pd.read_csv(Path("results") / finished[arm] / "costs.csv", index_col=0)
        returns.index, costs.index = ny_dates(returns.index), ny_dates(costs.index)
        frame = pd.DataFrame({"net": returns.iloc[:, 0].astype(float)}).join(costs[["total", "gross"]], how="inner")
        if len(frame) != len(returns):
            raise LabError(f"arm {arm}: returns and costs do not align session for session")
        out[arm] = frame.rename(columns={"total": "charged_cost"})
    return out


def jkp_series(path: Path, name: str) -> pd.Series:
    wide = parse_jkp(Path(path).read_bytes())
    if name not in wide.columns:
        raise LabError(f"{path} carries no {name!r} series")
    return wide[name].dropna().astype(float)


# --------------------------------------------------------------------- per arm


def arm_criteria(arm: str, frame: pd.DataFrame, public: pd.Series, publication: str, criteria: Mapping[str, Any],
                 boot: Mapping[str, Any], rf_annual: float) -> dict[str, Any]:
    per_year = int(criteria["sessions_per_year"])
    bar = publication_bar(public, {"publication": publication}, criteria)
    fid = fidelity(frame["net"], public, float(criteria["fidelity_floor"]))
    fid["direction_mismatch"] = bool(fid["daily_corr"] < 0)
    own_vol = float(frame["net"].std(ddof=1) * math.sqrt(per_year))
    runnable_vol = min(float(criteria["target_vol"]), float(criteria["max_scale"]) * own_vol)
    # L/E = scale / 2 for a 1x-gross equal long/short arm run at runnable / own volatility
    # (docs/hurdle_restatement_2026-09-30.md, replication-route records).
    econ = economics(fid["daily_corr"], bar["prior_sharpe"], frame, rf_annual, runnable_vol, per_year,
                     long_over_equity=0.5 * runnable_vol / own_vol)
    econ["own_vol_annual"] = own_vol
    econ["runnable_vol"] = runnable_vol
    contra = no_contradiction(frame, econ["figure_f3"], boot, per_year)
    passes = {"publication_bar": bar["pass"], "fidelity": fid["pass"], "economics": econ["pass"],
              "no_contradiction": contra["pass"]}
    return {"arm": arm, "publication": publication, "publication_bar": bar, "fidelity": fid, "economics": econ,
            "no_contradiction": contra, "in_house": annualized_stats(frame["net"], per_year), "passes": passes,
            "promoted": bool(all(passes.values()))}


# --------------------------------------------------------------------- combination


def capped_scale(returns: np.ndarray, caps: np.ndarray, target_vol: float, lookback: int, refresh: int,
                 per_year: int = 252) -> np.ndarray:
    """The constant-volatility rule with a per-decision cap (the gross ceiling), applied from the next session."""
    decisions = np.full(returns.size, np.nan)
    for i in np.flatnonzero(refresh_mask(returns.size, refresh)):
        if i + 1 < lookback:
            continue
        sigma = float(np.std(returns[i + 1 - lookback: i + 1], ddof=1) * math.sqrt(per_year))
        if not np.isfinite(sigma) or sigma <= 0:
            raise LabError(f"degenerate volatility estimate at session {i}")
        decisions[i] = min(float(caps[i]), target_vol / sigma)
    return _effective(decisions)


def combine(parts: Mapping[str, pd.DataFrame], cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Equal-weight average of scaled parts, scaled again to the target under the total-gross ceiling.

    Each part frame has ``net``, ``charged_cost``, ``scale`` and ``gross`` on a common
    daily index. Costs are summed with no netting credit.
    """
    names = sorted(parts)
    index = parts[names[0]].index
    for name in names[1:]:
        index = index.intersection(parts[name].index)
    aligned = {n: parts[n].loc[index] for n in names}
    net = pd.concat([aligned[n]["net"] for n in names], axis=1).mean(axis=1)  # equal weights
    cost = pd.concat([aligned[n]["charged_cost"] for n in names], axis=1).mean(axis=1)
    gross = pd.concat([aligned[n]["scale"] * aligned[n]["gross"] for n in names], axis=1).mean(axis=1)
    caps = float(cfg["max_total_gross"]) / gross.to_numpy(float)
    scale = capped_scale(net.to_numpy(float), caps, float(cfg["target_vol"]), int(cfg["lookback_sessions"]),
                         int(cfg["refresh_sessions"]))
    previous = np.concatenate([[1.0], scale[:-1]])
    change = np.abs(scale - previous) * gross.to_numpy(float) * float(cfg["scale_change_cost_bps"]) / 1e4
    frame = pd.DataFrame({"scale": scale, "net": scale * net.to_numpy(float) - change,
                          "charged_cost": scale * cost.to_numpy(float) + change, "gross": scale * gross.to_numpy(float)},
                         index=index)
    return frame.iloc[first_scored_index({"refresh_sessions": cfg["refresh_sessions"],
                                          "lookback_sessions": cfg["lookback_sessions"]}):]


def scale_part(frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """A fundamental arm scaled to the target by the momentum part's rule."""
    source = {"refresh_sessions": cfg["refresh_sessions"], "target_vol": cfg["target_vol"],
              "lookback_sessions": cfg["lookback_sessions"], "max_scale": cfg["part_max_scale"]}
    scaled = scaled_in_house(frame["net"], frame["gross"], frame["charged_cost"], source,
                             float(cfg["scale_change_cost_bps"]))
    return scaled.assign(gross=frame["gross"].reindex(scaled.index))[["net", "charged_cost", "scale", "gross"]]


def scale_public(series: pd.Series, cfg: Mapping[str, Any]) -> pd.DataFrame:
    source = {"refresh_sessions": cfg["refresh_sessions"], "target_vol": cfg["target_vol"],
              "lookback_sessions": cfg["lookback_sessions"], "max_scale": cfg["part_max_scale"]}
    values = series.to_numpy(float)
    scale = rule_scale(values, source)
    frame = pd.DataFrame({"net": scale * values, "charged_cost": 0.0, "scale": scale, "gross": 1.0},
                         index=series.index)
    return frame.iloc[first_scored_index(source):]


def combination_criteria(parts_in: Mapping[str, pd.DataFrame], parts_public: Mapping[str, pd.DataFrame],
                         publications: Mapping[str, str], cfg: Mapping[str, Any], criteria: Mapping[str, Any],
                         boot: Mapping[str, Any], rf_annual: float) -> dict[str, Any]:
    per_year = int(criteria["sessions_per_year"])
    inhouse = combine(parts_in, cfg)
    public = combine(parts_public, cfg)
    latest = max(pd.Timestamp(p + "-01") for p in publications.values())
    post = public.loc[latest:]
    post_years = len(post) / per_year
    if post_years >= float(criteria["post_publication_min_years"]):
        prior, basis = annualized_stats(post["net"], per_year)["sharpe"], "post_publication"
    else:
        prior = float(criteria["pre_publication_haircut"]) * annualized_stats(public["net"], per_year)["sharpe"]
        basis = "haircut_full_window"
    fid = fidelity(inhouse["net"], public["net"], float(criteria["fidelity_floor"]))
    # L/E = half the combined book's mean gross (equal long/short legs).
    econ = economics(fid["daily_corr"], prior, inhouse, rf_annual, float(cfg["target_vol"]), per_year,
                     long_over_equity=0.5 * float(inhouse["gross"].mean()))
    contra = no_contradiction(inhouse, econ["figure_f3"], boot, per_year)
    passes = {"prior_positive": bool(prior > 0), "economics": econ["pass"], "no_contradiction": contra["pass"]}
    return {
        "parts": sorted(parts_in), "prior": {"sharpe": prior, "basis": basis, "window_from": str(latest.date()),
                                             "window_years": post_years,
                                             "full_window": annualized_stats(public["net"], per_year)},
        "fidelity": fid, "economics": econ, "no_contradiction": contra,
        "in_house": annualized_stats(inhouse["net"], per_year), "mean_scale": float(inhouse["scale"].mean()),
        "passes": passes, "promoted": bool(all(passes.values())),
        "_series": pd.concat({"in_house": inhouse, "public": public}, axis=1),
    }


def momentum_part(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The promoted momentum configuration's in-house and public series from its committed record."""
    series = pd.read_csv(path, index_col=0, parse_dates=True)
    inhouse = series[["in_house|net", "in_house|charged_cost", "in_house|scale"]].dropna()
    inhouse.columns = ["net", "charged_cost", "scale"]
    public = series[["reference|gross", "reference|scale"]].dropna()
    public.columns = ["net", "scale"]
    return inhouse.assign(gross=1.0), public.assign(charged_cost=0.0, gross=1.0)


# --------------------------------------------------------------------- driver


def evaluate(reg: Mapping[str, Any], arms: Mapping[str, pd.DataFrame], jkp_paths: Mapping[str, Path],
             cash_path: Path) -> dict[str, Any]:
    criteria, boot, combo_cfg = reg["criteria"], reg["bootstrap"], reg["combination"]
    per_year = int(criteria["sessions_per_year"])
    cash_table, _ = parse_french(Path(cash_path).read_bytes())
    rf_annual = float(cash_table["RF"].dropna().iloc[-int(criteria["hurdle_window_sessions"]):].mean() * per_year)
    publics = {c["id"]: jkp_series(jkp_paths[c["id"]], c["id"]) for c in reg["characteristic"]}
    publications = {c["id"]: c["publication"] for c in reg["characteristic"]}
    per_arm = {arm: arm_criteria(arm, arms[arm], publics[arm], publications[arm], criteria, boot, rf_annual)
               for arm in publications}

    eligible = [a for a, r in per_arm.items() if r["passes"]["publication_bar"] and r["passes"]["fidelity"]]
    combination: dict[str, Any]
    if len(eligible) + 1 < int(combo_cfg["min_parts"]):
        combination = {"evaluated": False, "parts": eligible + [combo_cfg["momentum_part"]],
                       "reason": f"fewer than {combo_cfg['min_parts']} parts pass criteria 1 and 2", "promoted": False}
    else:
        momentum_in, momentum_public = momentum_part(Path(combo_cfg["momentum_series"]))
        parts_in = {a: scale_part(arms[a], combo_cfg) for a in eligible}
        parts_public = {a: scale_public(publics[a], combo_cfg) for a in eligible}
        parts_in[combo_cfg["momentum_part"]] = momentum_in
        parts_public[combo_cfg["momentum_part"]] = momentum_public
        pubs = {a: publications[a] for a in eligible} | {combo_cfg["momentum_part"]: combo_cfg["momentum_publication"]}
        combination = {"evaluated": True, **combination_criteria(parts_in, parts_public, pubs, combo_cfg, criteria,
                                                                 boot, rf_annual)}
    return {"registration": reg["id"], "selection_set": reg["selection_set"], "rf_annual": rf_annual,
            "arms": per_arm, "combination": combination,
            "promoted_arms": sorted(a for a, r in per_arm.items() if r["promoted"]),
            "combination_promoted": bool(combination.get("promoted"))}


def run(reg: Mapping[str, Any], out: Path, code_commit: str) -> dict[str, Any]:
    criteria, combo_cfg = reg["criteria"], reg["combination"]
    sources = {f"jkp_{c['id']}_daily": {"url": c["jkp_url"]} for c in reg["characteristic"]}
    sources["ff3_daily"] = {"url": criteria["cash_url"]}
    fetched = fetch_sources(sources, CACHE_DIR)
    for c in reg["characteristic"]:
        require_hash(Path(fetched[f"jkp_{c['id']}_daily"]["path"]), c["jkp_sha256"], f"JKP {c['id']}")
    require_hash(Path(fetched["ff3_daily"]["path"]), criteria["cash_sha256"], "French 3-factor")
    require_hash(Path(combo_cfg["momentum_series"]), combo_cfg["momentum_series_sha256"], "momentum record")

    engine = load_registration(Path(reg["engine_registration"]))
    arm_ids = [a["id"] for a in engine["arms"]]
    arms = arm_records(Path(reg["engine_read"]), Path("trials") / f"{engine['selection_set']}.jsonl", arm_ids)

    log = TrialLog(Path(reg["combination_trial_log"]))
    if log.exists() and any(row.get("arm_id") == "combination" for row in log.verify()):
        raise LabError("the combination already has a row; the budget is one")
    recorder = RunRecorder(
        log=log, run_id=f"run-{uuid.uuid4().hex[:16]}",
        config_id=hashlib.sha256(stable_json_dumps({k: reg[k] for k in ("id", "criteria", "combination")}).encode())
        .hexdigest(),
        data_id=hashlib.sha256("|".join(c["jkp_sha256"] for c in reg["characteristic"]).encode()).hexdigest(),
        kind="new", selection_set_id=reg["selection_set"], arm_id="combination",
    )
    recorder.started()
    try:
        summary = evaluate(reg, arms, {c["id"]: Path(fetched[f"jkp_{c['id']}_daily"]["path"])
                                       for c in reg["characteristic"]}, Path(fetched["ff3_daily"]["path"]))
    except Exception as exc:
        recorder.failed(f"{type(exc).__name__}: {exc}")
        raise
    series = summary["combination"].pop("_series", None)
    summary["code_commit"] = code_commit
    recorder.finished({"combination_promoted": summary["combination_promoted"],
                       "promoted_arms": summary["promoted_arms"], "code_commit": code_commit})
    out.mkdir(parents=True, exist_ok=True)
    (out / "replication.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    if series is not None:
        series.columns = [f"{a}|{b}" for a, b in series.columns]
        series.to_csv(out / "combination_series.csv")
    return summary


def promotion_outcome(path: Path = OUT_DIR / "replication.json") -> dict[str, Any]:
    """The registration's promotion callback: per-arm and combination verdicts from the written read."""
    summary = json.loads(Path(path).read_text())
    return {"promoted_arms": summary["promoted_arms"], "combination_promoted": summary["combination_promoted"],
            "arms": {a: r["passes"] for a, r in summary["arms"].items()},
            "combination": summary["combination"].get("passes")}


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description="Fundamentals, replication route.").parse_args(argv)
    reg = load_registration(REGISTRATION)
    code_commit = require_committed([REGISTRATION, ENGINE_REGISTRATION, CONTROL_SPEC, DESIGN_DOC])
    summary = run(reg, OUT_DIR, code_commit)
    print(json.dumps(promotion_outcome(OUT_DIR / "replication.json"), indent=2))
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
