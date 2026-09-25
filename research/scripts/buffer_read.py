"""The rank-buffer reads (docs/buffer_design.md §4), fixed before any buffered arm ran.

For each of the four buffered arms (``registrations/2026-09-buffer-{monthly,daily}.toml``),
against its unbuffered counterpart's committed record:

1. **Kill.** The studentized 95% interval for the arm's net periodic Sharpe (42-session
   circular blocks); an arm whose interval lies entirely below zero is dead.
2. **What the buffer did**, reported beside the kill:
   - turnover and charged cost a year, buffered against unbuffered;
   - the paired studentized interval for (buffered − unbuffered) net periodic Sharpe on
     the sessions both scored, resampled on the same blocks.

Reads the two registered reads of the buffer registrations and the committed records of
the counterparts; writes ``results/2026-09-buffer/buffer_read.json``. Refuses to start
unless the registrations, books and design doc are committed unchanged. Run from the repo
root after both registered reads::

    python -m research.scripts.buffer_read
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from prism.validation.compare import (
    SharpeBootstrapSpec,
    sharpe_difference_interval,
    sharpe_interval,
)
from research.scripts.factor_lab import LabError, require_committed

REGISTRATIONS = (Path("registrations/2026-09-buffer-monthly.toml"), Path("registrations/2026-09-buffer-daily.toml"))
BOOKS = (Path("examples/buffer/monthly.toml"), Path("examples/buffer/daily.toml"))
DESIGN_DOC = Path("docs/buffer_design.md")
OUT = Path("results/2026-09-buffer/buffer_read.json")
BOOTSTRAP = {"block_bars": 42, "n_resamples": 2000, "seed": 20261001, "confidence": 0.95}
PER_YEAR = 252
#: buffered arm -> (its registered read, its decision interval, the unbuffered counterpart's read and arm)
ARMS: dict[str, tuple[Path, int, Path, str]] = {
    "momentum_buffered": (Path("results/2026-09-buffer-monthly"), 21,
                          Path("results/2026-09-characteristics"), "momentum_control"),
    "ear_monthly_buffered": (Path("results/2026-09-buffer-monthly"), 21, Path("results/2026-09-earnings"), "ear_monthly"),
    "ear_event_buffered": (Path("results/2026-09-buffer-daily"), 1, Path("results/2026-09-earnings-daily"), "ear_event"),
    "premium_buffered": (Path("results/2026-09-buffer-daily"), 1, Path("results/2026-09-earnings-daily"),
                         "announcement_premium"),
}


def net_returns(read_dir: Path, arm: str) -> pd.Series:
    path = read_dir / "returns" / f"{arm}.csv"
    if not path.exists():
        raise LabError(f"{path} is missing; run that registration's registered read first")
    frame = pd.read_csv(path, index_col=0)
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_convert("America/New_York")
    return pd.Series(frame.iloc[:, 0].to_numpy(float), index=index.tz_localize(None).normalize())


def arm_facts(read_dir: Path, arm: str) -> dict[str, float]:
    record = json.loads((read_dir / "read.json").read_text())["arms"][arm]
    years = float(record["n_obs"]) / PER_YEAR
    return {"net_sharpe": float(record["net_sharpe_annualized"]), "gross_sharpe": float(record["gross_sharpe_annualized"]),
            "turnover_per_year": float(record["turnover_total"]) / years,
            "cost_per_year": float(record["charged_cost_total"]) / years, "n_obs": int(record["n_obs"])}


def spec_for(decision_interval: int) -> SharpeBootstrapSpec:
    return SharpeBootstrapSpec(block_bars=int(BOOTSTRAP["block_bars"]), n_resamples=int(BOOTSTRAP["n_resamples"]),
                               seed=int(BOOTSTRAP["seed"]), confidence=float(BOOTSTRAP["confidence"]),
                               decision_interval_bars=int(decision_interval))


def annualized(interval: dict[str, Any]) -> dict[str, float]:
    root = math.sqrt(PER_YEAR)
    return {key: float(interval[key]) * root for key in ("sharpe_a", "lower", "upper")}


def read_arm(arm: str, read_dir: Path, interval_bars: int, base_dir: Path, base_arm: str) -> dict[str, Any]:
    net = net_returns(read_dir, arm)
    base = net_returns(base_dir, base_arm)
    spec = spec_for(interval_bars)
    own = sharpe_interval(net.to_numpy(float), spec, label=arm).to_dict()
    common = net.index.intersection(base.index)
    paired = sharpe_difference_interval(net.loc[common].to_numpy(float), base.loc[common].to_numpy(float), spec,
                                        label=f"{arm}_minus_{base_arm}").to_dict()
    return {
        "kill": {"interval": own, "annualized": annualized(own), "killed": bool(own["upper"] < 0.0)},
        "buffered": arm_facts(read_dir, arm),
        "unbuffered": {"arm": base_arm, "read": str(base_dir), **arm_facts(base_dir, base_arm)},
        "paired_difference": {"interval": paired, "annualized": annualized(paired), "n_common": int(len(common)),
                              "first": str(common[0].date()), "last": str(common[-1].date())},
    }


def run(code_commit: str) -> dict[str, Any]:
    arms = {arm: read_arm(arm, *spec) for arm, spec in ARMS.items()}
    summary = {"code_commit": code_commit, "bootstrap": BOOTSTRAP, "arms": arms,
               "killed": sorted(a for a, r in arms.items() if r["kill"]["killed"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description="The rank-buffer reads.").parse_args(argv)
    code_commit = require_committed([*REGISTRATIONS, *BOOKS, DESIGN_DOC])
    summary = run(code_commit)
    print(json.dumps({"killed": summary["killed"],
                      "turnover_per_year": {a: (round(r["unbuffered"]["turnover_per_year"], 1),
                                                round(r["buffered"]["turnover_per_year"], 1))
                                            for a, r in summary["arms"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
