"""The same recorded configuration under both accounting engines (spec §3.8).

Stage 3b adds the canonical fixed-share simulator beside the
``historical_constant_weight`` reproduction engine, and it owes an empirical
answer to one question: on the recorded momentum configuration, how far apart
are they? The specification predicts the constant-weight engine *overstates*,
because it re-levers to its target weights as prices move between decisions and
charges nothing for the trade that would take. This module measures the size of
that, and reports whatever it is.

**What is held fixed.** One dataset, one frozen fold table, one universe, one
specification, one capital, and one cost policy whose every charged term is
identical between the engines by construction: the canonical stack's
commission, spread, quadratic impact and borrow are the recorded
``ExecutionConfig``'s, and the per-fold bucket spread the reproduction engine
charges is routed into the canonical model's ``CostInputs`` rather than
re-derived. The target sequence is identical too — the same
``walk_forward_evaluate`` produces it, and the canonical run drops only the
rows that restate the standing target, which the reproduction engine also
treats as no-ops. What is left to differ is the holding process, which is the
point.

**Measured, 2026-09-08, and the prediction did not hold (Run).** On
``results/demotion_b1`` — the certified B1 momentum arm, bucket spread,
``fold_portfolio = "reset"``, ``cost_refresh = "once"``, capital 1.0, 21 folds,
1,308 scored bars — the fixed-share book is *higher*, not lower::

    figure                     constant_weight         fixed_share      relative
    annualized_return                 0.048045            0.052801       +9.90%
    annualized_vol                    0.115079            0.118911       +3.33%
    sharpe (annualized)               0.465442            0.492285       +5.77%
    oos_periodic_sharpe               0.029320            0.031011       +5.77%
    max_drawdown                      0.139996            0.159789      +14.14%
    calmar                            0.343191            0.330440       -3.72%
    avg_gross                         0.968038            0.973067       +0.52%
    avg_turnover                      0.049977            0.052063       +4.17%
    total_cost                        0.036959            0.038006       +2.83%
    total_return                      0.275799            0.306131      +11.00%

The uncharged rebalance is real, but on a *momentum* book it is a drag, not a
premium: re-levering to a constant weight sells what has risen and buys what has
fallen inside the holding period, which is the opposite of the trade the sleeve
is expressing. So the reproduction engine understates this configuration's
return by 11 percent of itself while also understating its volatility, its
drawdown, its turnover and its charged cost — the drifted book is larger and
more concentrated when the next decision trades it, which is where the extra
turnover and the extra 2.8 percent of cost come from. The direction of the bias
is a property of the strategy, not of the engines: it would reverse on a
mean-reverting book. What is *not* configuration-dependent is that the two
engines disagree by two figures in the first significant place, which is why the
label on a record has to be checked rather than assumed.

**What is not claimed.** Neither arm here corrects the certified record. The
canonical arm is a different holding process and is recorded as a new result
with its own ``config_id``; both arms run unregistered, with
``kind = "exploratory"``, so they append to ``trials/exploratory.jsonl``, belong
to no selection set, enter no independent-evidence count, and may not be cited
for promotion (§3.12). The certified reproduction is
:mod:`prism.scripts.reproduce`, which is unchanged and still runs under
``historical_constant_weight``.

Usage::

    python -m prism.scripts.engine_comparison \\
        --recorded /home/b90/prism/results/demotion_b1 \\
        --cache-dir /home/b90/prism/data --out engine_comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from prism.evaluate import CanonicalSettings, EvaluationResult, walk_forward_evaluate
from prism.execution.cost_model import CostStack
from prism.io.dataset import Dataset
from prism.logging_utils import configure_logging
from prism.scripts.reproduce import ARMS, FOLD_TABLE, FOLD_TABLE_SHA256, ReproductionError, execution_from_recorded
from prism.scripts.reproduce import open_dataset as _open_dataset
from prism.spec import load_spec
from prism.validation.trials import CANONICAL_ENGINE, REPRODUCTION_ENGINE
from prism.validation.walk_forward import FoldTable
from prism.workspace import Workspace, resolve_workspace

logger = logging.getLogger(__name__)

#: The figures reported side by side. Every one comes out of the record the
#: evaluator wrote, never out of a recomputation in this module.
HEADLINE: tuple[str, ...] = (
    "total_return",
    "annualized_return",
    "annualized_vol",
    "sharpe",
    "oos_periodic_sharpe",
    "max_drawdown",
    "calmar",
    "avg_gross",
    "avg_net",
    "avg_turnover",
    "total_cost",
)

#: The canonical run's quantity grid. The recorded configuration declares
#: ``initial_capital = 1.0`` and prices its participation bound against it, so
#: the only comparison in which both engines see the same targets is the one at
#: that capital — where a whole share is unaffordable. §3.7 already declares a
#: continuous grid for exactly this weight-space comparison
#: (``RoundingSettings.for_reproduction``).
COMPARISON_INCREMENT = 0.0


def stack_from_recorded(recorded_config: Mapping[str, Any]) -> CostStack:
    """The recorded ``execution`` block as a declared cost stack (§3.6).

    A rename, term for term, plus the four keys ``ExecutionConfig`` has no field
    for. Those four are declared **zero** here and that is a statement about the
    recorded configuration, not a default: the recorded driver modelled no
    financing, no interest on uninvested cash, kept short proceeds as collateral
    with no rebate, and settled its borrow every bar. Charging anything else
    would make the comparison a comparison of cost policies.
    """
    execution = dict(recorded_config["execution"])
    schedule = [(float(lo), float(bps)) for lo, bps in recorded_config["spread_schedule"]["buckets"]]
    window = int(dict(recorded_config["signal"])["dollar_volume_window"])
    floor = float(execution["adv_floor_dollars"])
    volume = {
        "estimator": "median",
        "window": "trailing",
        "window_bars": window,
        "refresh": "fold",
        "floor_dollars": floor,
    }
    return CostStack.from_mapping(
        {
            # mode "bucket" is the recorded walk.spread_mode; flat_bps is the
            # fallback for a name the bucket vector does not price, which is what
            # the reproduction engine falls back to (`.fillna(execution.spread_bps)`).
            "spread": {
                "mode": "bucket",
                "flat_bps": float(execution["spread_bps"]),
                "schedule": schedule,
            },
            "impact": {
                "quadratic_notional_coeff": float(execution["slippage_coeff"]),
                "participation_model": str(execution["adv_impact_model"]),
                "participation_coeff": float(execution["adv_impact_coeff"]),
            },
            "dollar_volume": {
                "spread_bucket": dict(volume),
                "participation": dict(volume, base="initial_capital"),
                "impact": dict(volume),
            },
            "commission_bps": float(execution["commission_bps"]),
            "borrow_bps_annual": float(execution["borrow_rate_bps_annual"]),
            "cash_income_bps_annual": 0.0,
            "financing_bps_annual": 0.0,
            "short_proceeds": "collateral_no_rebate",
            "holding_cost_settlement": "per_bar",
            "trading_days_per_year": int(execution["trading_days_per_year"]),
        }
    )


@dataclass(frozen=True)
class EngineArm:
    """One engine's evaluation of the shared configuration."""

    engine: str
    result: EvaluationResult
    seconds: float

    def metrics(self) -> dict[str, float]:
        performance = self.result.record.performance
        metrics = {key: float(value) for key, value in self.result.metrics.items()}
        metrics["oos_periodic_sharpe"] = float(performance["net"]["periodic_sharpe"])
        return {key: metrics[key] for key in HEADLINE if key in metrics}


def run_comparison(
    *,
    workspace: Workspace,
    recorded_dir: Path,
    cache_dir: Path,
    case: str = "demotion_b1",
    arm: str = "bucket",
    dataset: Dataset | None = None,
    reconcile: bool = True,
) -> dict[str, EngineArm]:
    """Evaluate ``arm`` of ``case`` under both engines and return both arms."""
    recorded_config = json.loads((Path(recorded_dir) / "config.json").read_text(encoding="utf-8"))
    ds = (
        dataset
        if dataset is not None
        else _open_dataset(
            workspace, case=case, recorded_config=recorded_config, cache_dir=cache_dir, recorded_dir=recorded_dir
        )
    )
    table = FoldTable.from_toml(workspace.root / FOLD_TABLE)
    if table.content_hash != FOLD_TABLE_SHA256:
        raise ReproductionError(
            f"{FOLD_TABLE} hashes to {table.content_hash}, not the recorded {FOLD_TABLE_SHA256}; the two engines "
            "are compared on the recorded geometry or they are not compared"
        )
    spec = load_spec(workspace.root / ARMS[arm], workspace=workspace)
    execution = execution_from_recorded(recorded_config)
    capital = float(recorded_config.get("initial_capital", 1.0))
    settings = CanonicalSettings(
        cost_stack=stack_from_recorded(recorded_config),
        share_increment=COMPARISON_INCREMENT,
        min_notional=0.0,
        zero_crossing_clamp=False,
    )

    arms: dict[str, EngineArm] = {}
    for engine in (REPRODUCTION_ENGINE, CANONICAL_ENGINE):
        started = time.time()
        result = walk_forward_evaluate(
            spec,
            workspace=workspace,
            dataset=ds,
            fold_table=table,
            engine=engine,
            capital=capital,
            kind="exploratory",
            selection_set_id=None,
            arm_id=f"{case}_{arm}_spread_{engine}",
            run_channel="evidence",
            execution=execution,
            canonical=settings if engine == CANONICAL_ENGINE else None,
            fold_portfolio="reset",
            cost_refresh="once",
            reconcile=reconcile,
            untested_conclusions=(
                "The two engines are compared on one target sequence built from the emitted weight book; "
                "state-dependent construction (assemble, §3.8) would let the canonical arm's own drifted "
                "holdings feed the band and the participation bound, and is not exercised here.",
            ),
        )
        arms[engine] = EngineArm(engine=engine, result=result, seconds=time.time() - started)
        logger.info(
            "engine %s: run_id %s config_id %s in %.1fs", engine, result.run_id, result.config_id, arms[engine].seconds
        )
    return arms


def report(arms: Mapping[str, EngineArm]) -> dict[str, Any]:
    """The side-by-side table: every headline figure, both engines, and the difference."""
    reproduction = arms[REPRODUCTION_ENGINE].metrics()
    canonical = arms[CANONICAL_ENGINE].metrics()
    differences = {}
    for key in HEADLINE:
        if key not in reproduction or key not in canonical:
            continue
        left, right = reproduction[key], canonical[key]
        differences[key] = {
            REPRODUCTION_ENGINE: left,
            CANONICAL_ENGINE: right,
            "absolute": right - left,
            "relative": (right - left) / left if left else None,
        }
    return {
        "held_fixed": {
            "dataset": arms[REPRODUCTION_ENGINE].result.data_id,
            "fold_table_sha256": arms[REPRODUCTION_ENGINE].result.fold_table.content_hash,
            "capital": arms[REPRODUCTION_ENGINE].result.record.evaluation["capital"],
            "fold_portfolio": arms[REPRODUCTION_ENGINE].result.record.evaluation["fold_portfolio"],
            "cost_refresh": arms[REPRODUCTION_ENGINE].result.record.costs["cost_refresh"]["cadence"],
            "n_obs": int(len(arms[REPRODUCTION_ENGINE].result.scored.returns)),
        },
        "arms": {
            engine: {
                "run_id": arm.result.run_id,
                "config_id": arm.result.config_id,
                "holding_process": arm.result.record.evaluation["holding_process"],
                "executable_return": arm.result.record.conventions["executable_return"],
                "record": str(arm.result.record_path),
                "trial_log": str(arm.result.log_path),
                "n_obs": int(len(arm.result.scored.returns)),
                "seconds": round(arm.seconds, 1),
                "engine_detail": arm.result.record.evaluation["engine"],
                "metrics": arm.metrics(),
            }
            for engine, arm in arms.items()
        },
        "difference": differences,
    }


def text_table(payload: Mapping[str, Any]) -> str:
    """The report as the side-by-side table a reader can scan."""
    lines = [
        f"{'figure':<22}{'constant_weight':>20}{'fixed_share':>20}{'difference':>18}{'relative':>14}",
        "-" * 94,
    ]
    for key, row in payload["difference"].items():
        relative = row["relative"]
        lines.append(
            f"{key:<22}{row[REPRODUCTION_ENGINE]:>20.6f}{row[CANONICAL_ENGINE]:>20.6f}"
            f"{row['absolute']:>18.6f}" + (f"{relative:>14.4f}" if relative is not None else f"{'--':>14}")
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="demotion_b1")
    parser.add_argument("--arm", default="bucket", choices=sorted(ARMS))
    parser.add_argument("--recorded", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    workspace = resolve_workspace(args.workspace).ensure()
    configure_logging(workspace.logs, verbose=args.verbose, run_name="engine_comparison")
    arms = run_comparison(
        workspace=workspace,
        recorded_dir=Path(args.recorded).expanduser(),
        cache_dir=Path(args.cache_dir).expanduser(),
        case=args.case,
        arm=args.arm,
    )
    payload = report(arms)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    print(text_table(payload))
    return 0


__all__ = [
    "COMPARISON_INCREMENT",
    "HEADLINE",
    "EngineArm",
    "main",
    "report",
    "run_comparison",
    "stack_from_recorded",
    "text_table",
]


if __name__ == "__main__":  # pragma: no cover - console entry
    raise SystemExit(main())
