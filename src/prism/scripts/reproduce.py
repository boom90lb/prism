"""The reproduction of a recorded evidence directory (spec section 5).

Two cases are registered. ``demotion_b1`` is the certified B1 momentum record
in ``results/demotion_b1/``, which the stage 2 exit reproduces through the
generic evaluator "in ``historical_constant_weight`` mode, on the imported fold
table, under both the flat spread value and the bucket schedule as recorded".
``trend_t0`` is the registered T0 trend cell in ``results/trend_t0/``, which
section 5 schedules to stage 3 because it needs capabilities the data layer
does not deliver until then: a ``total_return`` scoring basis with cash
distributions applied to the scoring series while costs stay on the as-traded
close, a per-fold bucket spread frame keyed to the frozen fold table, a fixed
registered list universe, inverse-volatility sizing, and a session table that
reaches 2006.

What this module is, and what it is not. It is a *runner*, not a second engine:
it builds the dataset the recorded configuration consumed, resolves the frozen
fold table, and calls :func:`prism.evaluate.walk_forward_evaluate` once per arm.
Every number it reports comes out of a :class:`~prism.validation.trials.ResultRecord`
written by that evaluator and out of a hash-chained trial-log row, exactly as a
new evaluation's would. Nothing here computes a return, a cost, or a weight.

``demotion_b1`` runs two arms, because the recorded configuration's cost knob is
the one that moves the headline: ``bucket`` is the recorded ``walk.spread_mode``,
and ``flat`` charges the scalar ``execution.spread_bps`` instead. They are two
specifications and therefore two ``config_id``s and two records; the difference
between them is a property of the cost stack, not of the reproduction.
``trend_t0`` runs the one arm the cell registers; a second cost cadence on it
would be a new specification and therefore new evidence, which a reproduction
may not manufacture.

Every arm runs with ``kind = "reproduction"``, which
:func:`prism.validation.trials.attempt_counts` excludes from independent
evidence: reproducing a recorded result answers no new research question and
must not deflate anything.

Usage::

    python -m prism.scripts.reproduce --case demotion_b1 \\
        --recorded /home/b90/prism/results/demotion_b1 \\
        --cache-dir /home/b90/prism/data

    python -m prism.scripts.reproduce --case trend_t0 \\
        --recorded /home/b90/prism/results/trend_t0 \\
        --cache-dir /home/b90/prism/data

The dataset is built once under ``<workspace>/data/frozen/<case>`` and reused;
an import refuses rather than overwrites unless the directory is removed first,
because a dataset is written whole so its manifest describes exactly what is
there.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from prism.config import ExecutionConfig
from prism.evaluate import EvaluationResult, walk_forward_evaluate
from prism.io.dataset import Dataset
from prism.io.legacy_import import (
    build_legacy_dataset,
    import_legacy,
    legacy_summary_json,
    summary_json,
)
from prism.logging_utils import configure_logging
from prism.spec import load_spec
from prism.validation.walk_forward import FoldTable
from prism.workspace import Workspace, resolve_workspace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Case:
    """One registered reproduction: what it reads, what it runs, and what it must land on."""

    case_id: str
    #: The selection set the row joins. A reproduction row is excluded from
    #: that set's independent-evidence count by
    #: :func:`prism.validation.trials.attempt_counts`, so joining the history
    #: costs the set nothing.
    selection_set_id: str
    #: Arm name -> the specification it is declared in.
    arms: Mapping[str, str]
    #: The frozen fold table, imported from the recorded ``folds.json``, and
    #: the content hash it must carry. A table that hashes to anything else is
    #: not the recorded geometry and the run is refused rather than scored
    #: against a different set of windows.
    fold_table: str
    fold_table_sha256: str
    #: The cost-refresh cadence the recorded driver charged (§3.8). It is a
    #: property of the recorded run, never a knob this runner picks.
    cost_refresh: str
    #: Whether the evaluated span is taken from the record's ``start_date`` and
    #: ``end_date`` rather than from the dataset manifest's coverage. It is
    #: needed where the pinned caches run past the recorded cutoff, which the
    #: trend cell's do (its SPY file reaches 2026-08-13 and its cutoff is
    #: 2026-07-17) and the momentum cell's do not; the momentum arm keeps
    #: reading its span off the manifest so its record is unmoved.
    span_from_record: bool
    #: The arm the recorded headline belongs to. Any other arm is a different
    #: cost stack with no recorded counterpart, and is reported without a
    #: difference column rather than compared against a number that does not
    #: describe it.
    compared_arm: str
    #: The headline figures the recorded evidence states, and the record field
    #: each is read from. The reproduction reports the actual difference on
    #: every one of them, whatever it is.
    headline: Mapping[str, float]
    reproduces_reason: str
    untested_conclusions: tuple[str, ...]


#: What the reproduction engine leaves unmodelled, carried on every arm.
_ENGINE_CAVEAT = (
    "The reproduction engine holds weights constant between decision bars and charges nothing for "
    "the rebalance that implies; the canonical fixed-share simulator is what a forward claim rests "
    "on, and prism.scripts.engine_comparison measures how far apart the two are on this very "
    "configuration."
)

CASES: dict[str, Case] = {
    "demotion_b1": Case(
        case_id="demotion_b1",
        # The closed 17-trial residual set the live book's DSR deflates against
        # (``SPEC.md:30``).
        selection_set_id="stat_arb_residual",
        arms={
            "bucket": "examples/momentum/demotion_b1.toml",
            "flat": "examples/momentum/demotion_b1_flat_spread.toml",
        },
        fold_table="folds/demotion_b1.toml",
        fold_table_sha256="fec02ca011e8c66309ab9e414403d1a6412a10569ff47bac775fa754ef3384f8",
        # The recorded driver prices one simulation from the first formation
        # window's buckets; 'once' is that cadence, and 'fold' would be a
        # different charge on the same targets (§3.8).
        cost_refresh="once",
        span_from_record=False,
        compared_arm="bucket",
        headline={
            "total_return": 0.27579902407593426,
            "sharpe": 0.46544245642237375,
            "total_cost": 0.03695893383824346,
            "annualized_return": 0.048045335752954665,
            "annualized_vol": 0.11507946745850657,
            "max_drawdown": 0.13999581002068906,
            "avg_gross": 0.9680381326842661,
            "avg_turnover": 0.049976844189589156,
            "oos_periodic_sharpe": 0.029320118792967115,
        },
        reproduces_reason=(
            "stage 2 Parity reproduction of the certified record at {recorded}, under its own recorded "
            "configuration and its own frozen fold table"
        ),
        untested_conclusions=(_ENGINE_CAVEAT,),
    ),
    "trend_t0": Case(
        case_id="trend_t0",
        # The trend family's own six-trial budget
        # (``results/trend_v1_trials.jsonl``, ``docs/trend_design.md`` §3).
        selection_set_id="trend_v1",
        arms={"bucket": "examples/trend/trend_t0.toml"},
        fold_table="folds/trend_t0.toml",
        fold_table_sha256="c17447526a06363f8edb33f9921b81e5d707769bad3ce84230d3acf276b0884c",
        # ``config.spread_accounting = "per_fold_bucket"``: each fold's own
        # formation window prices every fill inside its test window. This is
        # the cadence the trend core deliberately does NOT share with the
        # momentum sibling, whose 'once' vector was inert on a 2020+ large-cap
        # panel and is not inert on a 2007+ ETF panel.
        cost_refresh="fold",
        span_from_record=True,
        compared_arm="bucket",
        headline={
            "total_return": 0.014239773174520298,
            "annualized_return": 0.0007645807869265386,
            "annualized_vol": 0.058672558952025396,
            "sharpe": 0.04240125941180066,
            "psr": 0.5722947668747193,
            "max_drawdown": 0.19762262438448686,
            "calmar": 0.003868892993946888,
            "avg_gross": 0.966832454909768,
            "avg_net": 0.3866149935103457,
            "avg_turnover": 0.04085462953988753,
            "total_cost": 0.13107768306134374,
            "oos_periodic_sharpe": 0.0026710282780847954,
        },
        reproduces_reason=(
            "stage 3b reproduction of the registered T0 trend cell at {recorded}, under its own recorded "
            "configuration, its own pinned cache files and its own frozen fold table"
        ),
        untested_conclusions=(
            _ENGINE_CAVEAT,
            "The recorded cell's own conclusion is negative: the sleeve carries no standalone edge at T0 "
            "(annualized Sharpe 0.042, below its own 3.78% t-bill hurdle) and its convexity admission was "
            "refused. Reproducing it faithfully reproduces that negative, which is what this row is evidence "
            "of and the only thing it may be cited for.",
        ),
    ),
}

#: Backward-compatible aliases for the stage 2 case. They are what
#: ``tests/test_demotion_b1_reproduction.py`` reads, and they name one case
#: rather than the registry, which is why they carry its id.
SELECTION_SET_ID = CASES["demotion_b1"].selection_set_id
ARMS: dict[str, str] = dict(CASES["demotion_b1"].arms)
FOLD_TABLE = CASES["demotion_b1"].fold_table
FOLD_TABLE_SHA256 = CASES["demotion_b1"].fold_table_sha256
RECORDED_HEADLINE: dict[str, float] = dict(CASES["demotion_b1"].headline)


class ReproductionError(RuntimeError):
    """The reproduction cannot be run as asked, and running it anyway would forge the comparison."""


@dataclass(frozen=True)
class ArmResult:
    """One arm's evaluation and its difference from the recorded headline."""

    arm: str
    spec_path: Path
    result: EvaluationResult
    seconds: float

    def metrics(self) -> dict[str, float]:
        """The headline figures, in the shape ``results/demotion_b1/summary.json`` reports them.

        They are the scored walk-forward's own metrics, and the record's
        ``performance`` block is checked against them here rather than
        trusted: a report that quoted one number while the record carried
        another would be a difference nobody could see.
        """
        performance = self.result.record.performance
        metrics = {key: float(value) for key, value in self.result.metrics.items()}
        metrics["oos_periodic_sharpe"] = float(performance["net"]["periodic_sharpe"])
        for key, recorded in (
            ("total_return", performance["net"]["total_return"]),
            ("sharpe", performance["sharpe"]["annualized"]),
            ("total_cost", performance["cost"]["total"]),
            ("avg_gross", performance["exposures"]["avg_gross"]),
            ("avg_net", performance["exposures"]["avg_net"]),
            ("avg_turnover", performance["turnover"]["average"]),
            ("max_drawdown", performance["drawdown"]["max_drawdown"]),
        ):
            if not math.isclose(metrics[key], float(recorded), rel_tol=1e-12, abs_tol=1e-15):
                raise ReproductionError(
                    f"the record's performance.{key} is {recorded!r} but the scored evaluation reports "
                    f"{metrics[key]!r}; the report quotes the record, so the two must be one number"
                )
        return metrics

    def differences(self, reference: Mapping[str, float]) -> dict[str, dict[str, float]]:
        """Reproduced minus recorded, per headline figure, absolute and relative."""
        out: dict[str, dict[str, float]] = {}
        mine = self.metrics()
        for key, recorded in reference.items():
            if key not in mine:
                continue
            actual = mine[key]
            absolute = actual - float(recorded)
            out[key] = {
                "recorded": float(recorded),
                "reproduced": float(actual),
                "absolute": absolute,
                "relative": absolute / float(recorded) if float(recorded) else float("nan"),
            }
        return out


def open_dataset(
    workspace: Workspace,
    *,
    case: str,
    recorded_config: Mapping[str, Any],
    cache_dir: Path,
    recorded_dir: Path,
) -> Dataset:
    """The reproduction dataset, built from the legacy caches on first use.

    The dataset is the recorded configuration's own inputs and nothing else.
    Rebuilding it is deterministic, so the ``data_id`` in the record is stable
    across runs, and an existing one is reused rather than rewritten because a
    dataset is written whole.

    The two cases need two importers, and the difference is a fact about what
    each record pins rather than a preference. ``demotion_b1`` names a resolved
    universe over the range-encoded caches plus a membership file, which
    :func:`build_legacy_dataset` reads through the loader's own file selection
    and keys by ticker. ``trend_t0`` names twenty specific cache files, two per
    symbol, and a directory of issuer distribution histories the total-return
    scoring basis is computed from; :func:`import_legacy` is the importer that
    can splice a pinned pair and write ``cash_distribution`` events, so it is
    the one used, and the ``cache_files`` pin keeps the dozen later
    ``SPY_1d_min_*.parquet`` files in the live cache out of the panel.
    """
    root = workspace.data / "frozen" / case
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        dataset = Dataset.open(root)
        logger.info("reusing the reproduction dataset at %s (data_id %s)", root, dataset.manifest.id)
        return dataset
    if case == "trend_t0":
        report = import_legacy(
            root,
            cache_dir=Path(cache_dir),
            vintage=f"{case}_{str(recorded_config['end_date'])[:10]}",
            symbols=list(recorded_config["symbols"]),
            distributions_dir=_recorded_input(
                recorded_dir, str(recorded_config["distributions_dir"]), directory=True
            ),
            cache_files=dict(recorded_config["cache_files"]),
        )
        logger.info("built the reproduction dataset\n%s", legacy_summary_json(report))
        return Dataset.open(root)
    universe = dict(recorded_config.get("universe") or {})
    membership = universe.get("membership_file")
    momentum = build_legacy_dataset(
        root,
        symbols=list(recorded_config["symbols"]),
        cache_dir=Path(cache_dir),
        start=str(recorded_config["start_date"]),
        end=str(recorded_config["end_date"]),
        vintage=f"{case}_{recorded_config['end_date']}",
        index_id="sp500",
        membership_file=None if membership is None else _recorded_input(recorded_dir, str(membership)),
        universe_note=universe.get("universe_file"),
        coverage=universe.get("coverage"),
    )
    logger.info("built the reproduction dataset\n%s", summary_json(momentum))
    return Dataset.open(root)


def execution_from_recorded(recorded_config: Mapping[str, Any]) -> ExecutionConfig:
    """The recorded ``execution`` block as the reproduction engine's parameter object."""
    block = dict(recorded_config["execution"])
    fields = {f.name for f in ExecutionConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = sorted(set(block) - fields)
    if unknown:
        raise ReproductionError(
            f"the recorded execution block carries {unknown}, which ExecutionConfig does not declare; the "
            "reproduction refuses rather than dropping a cost parameter it cannot honor"
        )
    return ExecutionConfig(**block)


def run_case(
    *,
    workspace: Workspace,
    case: str,
    recorded_dir: Path,
    cache_dir: Path,
    arms: Sequence[str],
    reconcile: bool = True,
) -> dict[str, ArmResult]:
    """Evaluate every arm of ``case`` through the evaluator and return their results."""
    registered = CASES.get(case)
    if registered is None:
        raise ReproductionError(f"no reproduction case named {case!r}; §5 registers {sorted(CASES)}")
    recorded_config = json.loads((Path(recorded_dir) / "config.json").read_text(encoding="utf-8"))
    dataset = open_dataset(
        workspace, case=case, recorded_config=recorded_config, cache_dir=cache_dir, recorded_dir=recorded_dir
    )
    table = FoldTable.from_toml(workspace.root / registered.fold_table)
    if table.content_hash != registered.fold_table_sha256:
        raise ReproductionError(
            f"{registered.fold_table} hashes to {table.content_hash}, not the recorded "
            f"{registered.fold_table_sha256}; the fold geometry is the recorded one or the run is not a "
            "reproduction"
        )
    execution = execution_from_recorded(recorded_config)
    capital = float(recorded_config.get("initial_capital", 1.0))
    # The recorded sample, where the record is the authority on it rather than
    # the files: the trend cell's pinned caches run past its cutoff, and a span
    # read off the files would score bars the recorded run never saw.
    span: dict[str, str] = (
        {
            "start": str(recorded_config["start_date"])[:10],
            "end": str(recorded_config["end_date"])[:10],
        }
        if registered.span_from_record
        else {}
    )
    results: dict[str, ArmResult] = {}
    for arm in arms:
        if arm not in registered.arms:
            raise ReproductionError(f"unknown arm {arm!r}; {case} declares {sorted(registered.arms)}")
        spec_path = workspace.root / registered.arms[arm]
        spec = load_spec(spec_path, workspace=workspace)
        started = time.time()
        result = walk_forward_evaluate(
            spec,
            workspace=workspace,
            dataset=dataset,
            fold_table=table,
            engine="historical_constant_weight",
            capital=capital,
            kind="reproduction",
            selection_set_id=registered.selection_set_id,
            arm_id=f"{case}_{arm}_spread",
            run_channel="evidence",
            execution=execution,
            fold_portfolio="reset",
            cost_refresh=registered.cost_refresh,
            **span,
            reconcile=reconcile,
            reproduces=(
                {
                    "run_id": f"legacy:{Path(recorded_dir).name}",
                    "reason": registered.reproduces_reason.format(recorded=recorded_dir),
                },
            ),
            untested_conclusions=registered.untested_conclusions,
        )
        results[arm] = ArmResult(arm=arm, spec_path=spec_path, result=result, seconds=time.time() - started)
        logger.info(
            "arm %s: run_id %s config_id %s in %.1fs",
            arm,
            result.run_id,
            result.config_id,
            results[arm].seconds,
        )
    return results


def report(
    results: Mapping[str, ArmResult],
    *,
    case: str = "demotion_b1",
    reference: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """The difference report: every headline figure, reproduced against recorded.

    Only the case's ``compared_arm`` carries a difference column. Another arm
    is a different cost stack with no recorded counterpart, and quoting the
    recorded headline beside it would invite a comparison the evidence does not
    support.
    """
    registered = CASES.get(case)
    if registered is None:
        raise ReproductionError(f"no reproduction case named {case!r}; §5 registers {sorted(CASES)}")
    headline = dict(registered.headline if reference is None else reference)
    return {
        "case": case,
        "compared_arm": registered.compared_arm,
        "reference": headline,
        "arms": {
            arm: {
                "arm_id": f"{arm}_spread",
                "spec": str(result.spec_path),
                "run_id": result.result.run_id,
                "config_id": result.result.config_id,
                "data_id": result.result.data_id,
                "record": str(result.result.record_path),
                "trial_log": str(result.result.log_path),
                "fold_table_sha256": result.result.fold_table.content_hash,
                "n_folds": len(result.result.folds),
                "n_obs": int(len(result.result.scored.returns)),
                "seconds": round(result.seconds, 1),
                "metrics": result.metrics(),
                "differences": result.differences(headline) if arm == registered.compared_arm else None,
            }
            for arm, result in results.items()
        },
    }


def _recorded_input(recorded_dir: Path, relative: str, *, directory: bool = False) -> Path:
    """A recorded input of the required kind, resolved against the run's repository."""
    root = Path(recorded_dir).resolve()
    for parent in (root, *root.parents):
        candidate = parent / relative
        matches_kind = candidate.is_dir() if directory else candidate.is_file()
        if matches_kind:
            return candidate
    expected = "directory" if directory else "file"
    raise ReproductionError(
        f"the recorded configuration names {relative!r}, which is not a {expected} above {root}; the reproduction "
        "reads the inputs the record names and mints nothing in their place"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="demotion_b1", choices=sorted(CASES))
    parser.add_argument("--recorded", required=True, help="the recorded evidence directory to reproduce")
    parser.add_argument("--cache-dir", required=True, help="the per-symbol bar caches the recorded run read")
    parser.add_argument("--workspace", default=None)
    parser.add_argument(
        "--arms",
        default=None,
        help="comma-separated arm names; defaults to every arm the case declares",
    )
    parser.add_argument("--out", default=None, help="where to write the difference report JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    workspace = resolve_workspace(args.workspace).ensure()
    configure_logging(workspace.logs, verbose=args.verbose, run_name="reproduce")
    declared = sorted(CASES[args.case].arms)
    results = run_case(
        workspace=workspace,
        case=args.case,
        recorded_dir=Path(args.recorded).expanduser(),
        cache_dir=Path(args.cache_dir).expanduser(),
        arms=declared if args.arms is None else [a.strip() for a in str(args.arms).split(",") if a.strip()],
    )
    payload = report(results, case=args.case)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    print(text)
    return 0


__all__ = [
    "ARMS",
    "CASES",
    "FOLD_TABLE",
    "FOLD_TABLE_SHA256",
    "RECORDED_HEADLINE",
    "SELECTION_SET_ID",
    "ArmResult",
    "Case",
    "ReproductionError",
    "main",
    "open_dataset",
    "report",
    "run_case",
]


if __name__ == "__main__":  # pragma: no cover - console entry
    raise SystemExit(main())
