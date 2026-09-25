"""Earnings-announcement return, replication route (registrations ``2026-09-earnings*.toml``).

Design, criteria and limits: ``docs/earnings_design.md``. Reads the one engine arm's
record (``research.scripts.registered_read`` output plus the run's ``costs.csv``), the
pinned Chen and Zimmermann daily value-weighted ``AnnouncementReturn`` long-short
series and the French cash series, and decides:

* **the arm**, on the four SPEC §4 replication criteria: publication bar and prior on the
  public series, daily tracking of the in-house arm against it, economics at our costs
  against the T-bill hurdle at the volatility the arm can run at, and no in-house
  contradiction;
* **the pinned combination** (SPEC §4, Combinations): the arm, if it passes criteria 1
  and 2, with the promoted volatility-scaled momentum, each scaled to 12% volatility,
  averaged at equal weights and scaled to 12% again under the 2x gross ceiling; its
  prior from the same combination of public series.

The per-arm and combination arithmetic is the fundamentals read's
(``research.scripts.fundamentals_replication``), imported unchanged. Refuses to start
unless the registrations, the control spec and the design doc are committed unchanged,
and refuses any input whose sha256 differs from the registered one. Appends one trial
row, for the combination, to its own log. Run from the repo root::

    python -m research.scripts.earnings_replication
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from prism.validation.trials import RunRecorder, TrialLog, stable_json_dumps
from research.scripts.factor_lab import (
    CACHE_DIR,
    LabError,
    fetch_sources,
    load_registration,
    parse_french,
    require_committed,
)
from research.scripts.fundamentals_replication import (
    arm_criteria,
    arm_records,
    combination_criteria,
    momentum_part,
    scale_part,
    scale_public,
)
from research.scripts.vol_momentum_replication import require_hash

REGISTRATION = Path("registrations/2026-09-earnings-replication.toml")
ENGINE_REGISTRATION = Path("registrations/2026-09-earnings.toml")
CONTROL_SPEC = Path("examples/earnings/monthly.toml")
DESIGN_DOC = Path("docs/earnings_design.md")
OUT_DIR = Path("results/2026-09-earnings/replication")
PUBLIC_SOURCE = "cz_daily_predictor_vw"


def parse_cz_daily(payload: bytes, member: str) -> pd.Series:
    """One Chen and Zimmermann daily portfolio file's long-short column, as decimal returns.

    The files carry ``date, port01 .. port05, portLS`` in percent, ``NA`` where a
    portfolio is empty.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if member not in archive.namelist():
            raise LabError(f"the archive carries no {member!r}")
        frame = pd.read_csv(archive.open(member), na_values=["NA"])
    if frame.columns[0] != "date" or "portLS" not in frame.columns:
        raise LabError(f"{member}: expected date and portLS columns, found {list(frame.columns)}")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    series = pd.Series(frame["portLS"].to_numpy(float) / 100.0, index=pd.DatetimeIndex(dates), name=None).dropna()
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise LabError(f"{member}: dates are not strictly increasing")
    if series.empty or float(series.abs().median()) > 0.05:
        raise LabError(f"{member}: empty, or not percent returns (median |r| {series.abs().median():.4f} after /100)")
    return series


def evaluate(reg: Mapping[str, Any], arms: Mapping[str, pd.DataFrame], public_path: Path,
             cash_path: Path) -> dict[str, Any]:
    criteria, boot, combo_cfg, char = reg["criteria"], reg["bootstrap"], reg["combination"], reg["characteristic"]
    per_year = int(criteria["sessions_per_year"])
    cash_table, _ = parse_french(Path(cash_path).read_bytes())
    rf_annual = float(cash_table["RF"].dropna().iloc[-int(criteria["hurdle_window_sessions"]):].mean() * per_year)
    public = parse_cz_daily(Path(public_path).read_bytes(), char["cz_member"])
    arm = str(char["id"])
    per_arm = {arm: arm_criteria(arm, arms[arm], public, char["publication"], criteria, boot, rf_annual)}

    eligible = [a for a, r in per_arm.items() if r["passes"]["publication_bar"] and r["passes"]["fidelity"]]
    combination: dict[str, Any]
    if len(eligible) + 1 < int(combo_cfg["min_parts"]):
        combination = {"evaluated": False, "parts": eligible + [combo_cfg["momentum_part"]],
                       "reason": f"fewer than {combo_cfg['min_parts']} parts pass criteria 1 and 2", "promoted": False}
    else:
        momentum_in, momentum_public = momentum_part(Path(combo_cfg["momentum_series"]))
        parts_in = {a: scale_part(arms[a], combo_cfg) for a in eligible}
        parts_public = {a: scale_public(public, combo_cfg) for a in eligible}
        parts_in[combo_cfg["momentum_part"]] = momentum_in
        parts_public[combo_cfg["momentum_part"]] = momentum_public
        pubs = {a: char["publication"] for a in eligible} | {combo_cfg["momentum_part"]: combo_cfg["momentum_publication"]}
        combination = {"evaluated": True, **combination_criteria(parts_in, parts_public, pubs, combo_cfg, criteria,
                                                                 boot, rf_annual)}
    return {"registration": reg["id"], "selection_set": reg["selection_set"], "rf_annual": rf_annual,
            "public_series": {"first": str(public.index[0].date()), "last": str(public.index[-1].date()),
                              "n": int(len(public))},
            "arms": per_arm, "combination": combination,
            "promoted_arms": sorted(a for a, r in per_arm.items() if r["promoted"]),
            "combination_promoted": bool(combination.get("promoted"))}


def run(reg: Mapping[str, Any], out: Path, code_commit: str) -> dict[str, Any]:
    criteria, combo_cfg, char = reg["criteria"], reg["combination"], reg["characteristic"]
    fetched = fetch_sources({PUBLIC_SOURCE: {"url": char["cz_url"]}, "ff3_daily": {"url": criteria["cash_url"]}},
                            CACHE_DIR)
    require_hash(Path(fetched[PUBLIC_SOURCE]["path"]), char["cz_sha256"], "Chen and Zimmermann daily VW portfolios")
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
        data_id=hashlib.sha256(str(char["cz_sha256"]).encode()).hexdigest(),
        kind="new", selection_set_id=reg["selection_set"], arm_id="combination",
    )
    recorder.started()
    try:
        summary = evaluate(reg, arms, Path(fetched[PUBLIC_SOURCE]["path"]), Path(fetched["ff3_daily"]["path"]))
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
    """The registration's promotion callback: the arm's and the combination's verdicts from the written read."""
    summary = json.loads(Path(path).read_text())
    return {"promoted_arms": summary["promoted_arms"], "combination_promoted": summary["combination_promoted"],
            "arms": {a: r["passes"] for a, r in summary["arms"].items()},
            "combination": summary["combination"].get("passes")}


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description="Earnings-announcement return, replication route.").parse_args(argv)
    reg = load_registration(REGISTRATION)
    code_commit = require_committed([REGISTRATION, ENGINE_REGISTRATION, CONTROL_SPEC, DESIGN_DOC])
    summary = run(reg, OUT_DIR, code_commit)
    print(json.dumps(promotion_outcome(OUT_DIR / "replication.json"), indent=2))
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
