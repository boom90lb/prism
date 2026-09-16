#!/usr/bin/env python3
"""Verify a private owner project against a local wheel, without broker access.

Build the wheel separately, then pass --source, --wheel and a NEW --out directory.
The existing frozen demotion_b1 dataset is required. The only evaluation is the
registered B1 bucket reproduction. Engineering run rows never join a trial set.
No original file is written; no credential file is read; publication is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def offline_guard(event: str, args: tuple) -> None:
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError(f"owner migration is offline: refused {event}")
    if event == "open" and isinstance(args[0], (str, bytes)):
        if Path(os.fsdecode(args[0])).name == ".env":
            raise RuntimeError("owner migration never reads a credential file")


def worker(project: Path) -> None:
    sys.addaudithook(offline_guard)
    import prism
    from prism.config import ExecutionConfig
    from prism.evaluate import walk_forward_evaluate
    from prism.io.dataset import Dataset
    from prism.live.authorization import distribution_id
    from prism.spec import load_spec
    from prism.validation.trials import TrialLog, attempt_counts
    from prism.validation.walk_forward import FoldTable
    from prism.workspace import Workspace

    project = project.resolve()
    installed = Path(prism.__file__).resolve()
    assert installed.is_relative_to(project / ".venv"), installed
    dist = importlib.metadata.distribution("prism-trading")
    assert dist.version == "0.4.0", dist.version
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    assert not direct.get("dir_info", {}).get("editable", False), direct
    assert ".whl" in direct.get("url", ""), direct
    pin = tomllib.loads((project / "pyproject.toml").read_text())["project"]["dependencies"]
    assert pin == ["prism-trading==0.4.0"], pin
    installed_record_sha256 = hashlib.sha256(dist.read_text("RECORD").encode("utf-8")).hexdigest()
    reported_distribution_id = distribution_id()
    assert reported_distribution_id == "sha256:" + installed_record_sha256, reported_distribution_id

    history = json.loads((project / "history_manifest.json").read_text())
    for name, value in history.items():
        assert digest(project / name) == value["sha256"], name
    counts = {}
    for path in sorted((project / "trials").glob("*.jsonl")):
        rows = TrialLog(path).verify()
        counts[str(path.relative_to(project))] = {
            "rows": len(rows), "attempt_counts": attempt_counts(rows, registered=path.stem != "exploratory")
        }
    save(project / "history_counts.json", counts)

    workspace = Workspace(project).ensure()
    config = json.loads((project / "results/demotion_b1/config.json").read_text())
    reference = json.loads((project / "results/demotion_b1/summary.json").read_text())
    fold_reference = json.loads((project / "results/demotion_b1/folds.json").read_text())
    table = FoldTable.from_toml(project / "folds/demotion_b1.toml")
    assert table.content_hash == "fec02ca011e8c66309ab9e414403d1a6412a10569ff47bac775fa754ef3384f8"
    assert len(table.folds) == 21
    spec = load_spec(project / "examples/momentum/demotion_b1.toml", workspace=workspace)
    dataset = Dataset.open(project / "data/frozen/demotion_b1")
    # Declared before the run; same tolerance as the recorded golden test.
    tolerance = 1e-9
    started = time.monotonic()
    result = walk_forward_evaluate(
        spec, workspace=workspace, dataset=dataset, fold_table=table,
        engine="historical_constant_weight", capital=float(config["initial_capital"]),
        kind="reproduction", run_channel="engineering", execution=ExecutionConfig(**config["execution"]),
        fold_portfolio="reset", cost_refresh="once", reconcile=False,
        reproduces=({"run_id": "legacy:demotion_b1", "reason": "Installed wheel owner migration, same registered bucket configuration and frozen fold table"},),
        untested_conclusions=("Engineering reproduction only; no new independent evidence, promotion or deployment authorization.",),
    )
    keys = ("total_return", "sharpe", "total_cost", "annualized_return", "annualized_vol",
            "max_drawdown", "avg_gross", "avg_turnover", "oos_periodic_sharpe")
    actuals = dict(result.metrics)
    actuals["oos_periodic_sharpe"] = float(result.record.performance["net"]["periodic_sharpe"])
    comparisons = {}
    for key in keys:
        actual, expected = float(actuals[key]), float(reference[key])
        assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (key, actual, expected)
        comparisons[key] = {"recorded": expected, "reproduced": actual, "absolute_difference": actual - expected}
    assert len(result.scored.returns) == sum(row["test_rows"] for row in fold_reference)
    assert result.log_path == project / "runs/engineering.jsonl", result.log_path
    rows = TrialLog(result.log_path).verify()
    assert len(rows) == 2 and {row["kind"] for row in rows} == {"reproduction"}
    assert {row["run_channel"] for row in rows} == {"engineering"}
    for name, value in history.items():
        assert digest(project / name) == value["sha256"], name
    for name, before in counts.items():
        rows = TrialLog(project / name).verify()
        assert attempt_counts(rows, registered=Path(name).stem != "exploratory") == before["attempt_counts"]
    save(project / "verification.json", {
        "status": "Tested", "distribution": dist.metadata["Name"], "version": dist.version,
        "import_file": str(installed), "distribution_id": reported_distribution_id,
        "installed_record_sha256": installed_record_sha256,
        "direct_url": direct,
        "run_id": result.run_id, "record": str(result.record_path), "data_id": result.data_id,
        "config_id": result.config_id, "fold_table_sha256": table.content_hash,
        "n_folds": len(result.folds), "n_observations": len(result.scored.returns),
        "relative_and_absolute_tolerance": tolerance, "metrics": comparisons,
        "history_files_unchanged": len(history), "native_ledger_chains_verified": len(counts),
        "trial_rows_appended": 0, "engineering_rows": 2, "seconds": time.monotonic() - started,
        "offline_guard": "socket connection, address lookup, bind and .env opens refused by Python audit hook",
        "scope": "Full historical B1 bucket reproduction from installed wheel; no new research claim or operating session",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.source is None or args.wheel is None or args.out is None:
        parser.error("--source, --wheel and --out are required")
    source, wheel, out = (p.resolve() for p in (args.source, args.wheel, args.out))
    dataset = (args.dataset or source / "data/frozen/demotion_b1").resolve()
    if out.exists():
        parser.error(f"refusing existing output {out}; use a new directory")
    if out.is_relative_to(source):
        parser.error("output must be outside the engine checkout")
    if out == dataset or out.is_relative_to(dataset):
        parser.error("output must be outside the frozen input dataset")
    if not (dataset / "manifest.json").is_file():
        parser.error(f"frozen B1 dataset missing: {dataset}")
    with zipfile.ZipFile(wheel) as archive:
        metadata = BytesParser().parsebytes(archive.read("prism_trading-0.4.0.dist-info/METADATA"))
        assert metadata["Name"] == "prism-trading" and metadata["Version"] == "0.4.0"
    # Do not pass broker/vendor credentials or Python import overrides to child processes.
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "UV_CACHE_DIR"}}
    env["PYTHONNOUSERSITE"] = "1"
    env["UV_PYTHON_DOWNLOADS"] = "never"
    project = out / "private-project"
    project.mkdir(parents=True)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    source_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source))
    source_diff_sha256 = hashlib.sha256(subprocess.check_output(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"], cwd=source
    )).hexdigest()
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "SPEC.md", "docs", "examples", "folds", "calibrations",
         "registrations", "trials", "results", "data/universe", "data/frozen"],
        cwd=source,
    ).decode().split("\0")
    copied = []
    history = {}
    for name in filter(None, tracked):
        candidate = source / name
        if candidate.is_symlink():
            raise RuntimeError(f"refusing symlink input {name}")
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate, target)
        copied.append(name)
        if name.endswith(".jsonl"):
            history[name] = {"sha256": digest(candidate), "bytes": candidate.stat().st_size,
                             "rows": len(candidate.read_bytes().splitlines())}
    shutil.copytree(dataset, project / "data/frozen/demotion_b1")
    (project / "vendor").mkdir()
    shutil.copyfile(wheel, project / "vendor" / wheel.name)
    shutil.copyfile(Path(__file__), project / "verify_migration.py")
    (project / "pyproject.toml").write_text(
        '[project]\nname = "prism-owner-research"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n'
        'dependencies = ["prism-trading==0.4.0"]\n\n[tool.uv]\npackage = false\n\n'
        f'[tool.uv.sources]\nprism-trading = {{ path = "vendor/{wheel.name}" }}\n'
    )
    save(project / "history_manifest.json", history)
    # Source hash and preservation checks cover every frozen file as well as the ledgers.
    data_files = {str(p.relative_to(dataset)): digest(p) for p in dataset.rglob("*") if p.is_file()}
    save(project / "dataset_copy_manifest.json", data_files)
    for name, value in data_files.items():
        assert digest(project / "data/frozen/demotion_b1" / name) == value
    log = out / "execution.log"
    commands = [
        ["uv", "venv", ".venv", "--python", sys.executable],
        ["uv", "lock"],
        ["uv", "sync", "--frozen", "--no-dev"],
        [str(project / ".venv/bin/python"), "verify_migration.py", "--worker", str(project)],
    ]
    with log.open("w") as handle:
        for command in commands:
            print("Running", " ".join(command), flush=True)
            handle.write("$ " + " ".join(command) + "\n")
            handle.flush()
            subprocess.run(command, cwd=project, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    for name, value in history.items():
        assert digest(source / name) == value["sha256"], f"source ledger changed: {name}"
        assert digest(project / name) == value["sha256"], f"private ledger changed: {name}"
    for name, value in data_files.items():
        assert digest(dataset / name) == value, f"source data changed: {name}"
        assert digest(project / "data/frozen/demotion_b1" / name) == value
    verification = json.loads((project / "verification.json").read_text())
    save(out / "receipt.json", {
        **verification, "source_commit": source_commit, "source_dirty": source_dirty,
        "source_tracked_diff_sha256": source_diff_sha256, "source": str(source), "project": str(project),
        "wheel": str(wheel), "wheel_sha256": digest(wheel), "private_lock_sha256": digest(project / "uv.lock"),
        "copied_tracked_input_files": len(copied), "copied_dataset_files": len(data_files),
        "history": history, "native_history": json.loads((project / "history_counts.json").read_text()),
    })
    print(f"Tested: {out / 'receipt.json'}", flush=True)


if __name__ == "__main__":
    main()
