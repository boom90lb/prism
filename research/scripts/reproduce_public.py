"""Uncounted reproduction of the published-factor evidence (``docs/public_reproduction_design.md``).

Reproduction, not an evaluation: no trial-ledger row, nothing written under
``results/``, ``trials/`` or ``data/public/``, and no recorded verdict changes.
Every output, downloads included, goes into one empty directory chosen by the
caller.

* Layer A (offline, always): recompute the four criteria of
  ``registrations/2026-09-vol-momentum.toml`` from the tracked
  ``results/2026-09-vol-momentum/series.csv`` and compare them with the recorded
  ``summary.json`` (A1); rebuild the scaled in-house series from its tracked,
  hash-checked inputs (A2); check that the recorded reference follows the
  registered rule (A3).
* Layer B (network, keyless): download the current French momentum and
  three-factor files, record their hashes and vintage, measure drift against
  the record, and recompute the criteria on the new vintage cut at the recorded
  last date (B1) and through its last date (B2).
* ``--with-factor-lab`` (network): rerun the factor lab's uncounted priors and
  fidelity parts on the current sources, into the output directory, and compare
  them with the recorded files.

Run from the repo root::

    python -m research.scripts.reproduce_public --out DIR [--offline] [--with-factor-lab]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.validation.trials import current_git_state
from research.scripts import factor_lab as fl
from research.scripts import vol_momentum_replication as vm

REPO = Path(__file__).resolve().parents[2]
VM_REGISTRATION = Path("registrations/2026-09-vol-momentum.toml")
FL_REGISTRATION = Path("registrations/2026-09-factor-lab.toml")
VM_RECORD = Path("results/2026-09-vol-momentum")
FL_RECORD = Path("results/2026-09-factor-lab")
# Recorded for this file in results/history_import/records.jsonl, line 1.
LIVE_BOOK_RETURNS = Path("results/demotion_b1/returns.csv")
LIVE_BOOK_RETURNS_SHA256 = "90956649b2fb795017713749f0a10d175383df0ca3a0f914a8cd9b29c06a83f4"
PROTECTED = (Path("results"), Path("trials"), Path("data/public"))
FIRST_LINE = "Reproduction, not an evaluation: uncounted, no ledger row, no recorded verdict changes."
TOLERANCE = 1e-9
USER_AGENT = "prism-reproduce-public/1.0 (research; keyless)"
SUMMARY_KEYS = ("publication_bar", "fidelity", "economics", "no_contradiction", "in_house", "promoted")
IN_HOUSE_COLUMNS = ("scale", "net", "charged_cost", "unscaled_net")


class ReproductionError(RuntimeError):
    """An input, a download or an output path failed a check; raised, never swallowed (SPEC N7)."""


# --------------------------------------------------------------------- paths and files


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_sha256(path: Path, expected: str, what: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ReproductionError(f"{what} at {path} has sha256 {actual}, recorded {expected}; this clone is not "
                                "the recorded state")
    return actual


def prepare_out(out: Path, repo: Path = REPO) -> Path:
    """Resolve ``out``, refuse a protected or non-empty directory, and create it."""
    resolved = Path(out).expanduser().resolve()
    for protected in PROTECTED:
        root = (repo / protected).resolve()
        if resolved == root or root in resolved.parents:
            raise ReproductionError(f"output directory {resolved} is under {protected}/; a reproduction writes "
                                    "nothing there")
    if resolved.exists():
        if not resolved.is_dir():
            raise ReproductionError(f"output path {resolved} exists and is not a directory")
        if any(resolved.iterdir()):
            raise ReproductionError(f"output directory {resolved} is not empty")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def download(url: str) -> tuple[bytes, dict[str, str]]:
    """GET ``url``; return the body and the response headers that identify the file version."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
        headers = {name: value for name in ("Last-Modified", "ETag", "Content-Length")
                   if (value := response.headers.get(name)) is not None}
    return payload, headers


def fetch(sources: Mapping[str, str], cache: Path) -> dict[str, dict[str, Any]]:
    """Download every ``source_id -> url`` into ``cache`` (always fresh); record what was served."""
    cache.mkdir(parents=True, exist_ok=True)
    fetched: dict[str, dict[str, Any]] = {}
    for source_id, url in sources.items():
        try:
            payload, headers = download(url)
        except Exception as exc:  # N7: a failed fetch raises with its source named
            raise ReproductionError(f"download failed for {source_id} ({url}): {exc}") from exc
        if not payload:
            raise ReproductionError(f"download returned no bytes for {source_id} ({url})")
        target = cache / f"{source_id}.zip"
        target.write_bytes(payload)
        fetched[source_id] = {
            "url": url,
            "path": str(target),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "last_modified": headers.get("Last-Modified"),
            "etag": headers.get("ETag"),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    return fetched


# --------------------------------------------------------------------- comparison


def plain(value: Any) -> Any:
    """``value`` with numpy scalars and containers turned into JSON types."""
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compare(recorded: Any, reproduced: Any, path: str = "", tol: float = TOLERANCE) -> list[dict[str, Any]]:
    """Leaf-by-leaf comparison over the keys of ``recorded``.

    Floats agree within ``tol`` absolute; integers, strings, booleans and
    ``None`` must be equal. Each leaf gives one row with its dotted path.
    """
    reproduced = plain(reproduced)
    if isinstance(recorded, Mapping):
        if not isinstance(reproduced, Mapping):
            return [{"path": path, "recorded": "<mapping>", "reproduced": reproduced, "match": False}]
        rows: list[dict[str, Any]] = []
        for key, item in recorded.items():
            child = f"{path}.{key}" if path else str(key)
            if key not in reproduced:
                rows.append({"path": child, "recorded": plain(item), "reproduced": "<missing>", "match": False})
            else:
                rows.extend(compare(item, reproduced[key], child, tol))
        return rows
    if isinstance(recorded, list):
        if not isinstance(reproduced, list) or len(recorded) != len(reproduced):
            return [{"path": path, "recorded": recorded, "reproduced": reproduced, "match": False}]
        return [row for i, item in enumerate(recorded) for row in compare(item, reproduced[i], f"{path}[{i}]", tol)]
    numeric = (int, float)
    if (isinstance(recorded, numeric) and not isinstance(recorded, bool)
            and isinstance(reproduced, numeric) and not isinstance(reproduced, bool)):
        difference = float(reproduced) - float(recorded)
        if isinstance(recorded, int) and isinstance(reproduced, int):
            match = recorded == reproduced
        else:
            match = abs(difference) <= tol
        return [{"path": path, "recorded": recorded, "reproduced": reproduced, "difference": difference,
                 "match": bool(match)}]
    return [{"path": path, "recorded": recorded, "reproduced": reproduced,
             "match": bool(type(recorded) is type(reproduced) and recorded == reproduced)}]


def comparison_block(recorded: Any, reproduced: Any) -> dict[str, Any]:
    rows = compare(recorded, reproduced)
    mismatches = [row for row in rows if not row["match"]]
    return {"leaves_compared": len(rows), "leaves_differing": len(mismatches), "match": not mismatches,
            "differing": mismatches}


def drift(recorded: pd.Series, current: pd.Series, tol: float = TOLERANCE) -> dict[str, Any]:
    """How a recorded daily series and its current counterpart differ, session by session."""
    common = recorded.index.intersection(current.index)
    difference = (current.loc[common] - recorded.loc[common]).abs()
    changed = difference[difference > tol]
    removed = recorded.index.difference(current.index)
    added = current.index.difference(recorded.index)
    first, last = recorded.index[0], recorded.index[-1]

    def dates(index: pd.Index) -> list[str]:
        return [str(day.date()) for day in index[:5]]

    return {
        "recorded_first": str(first.date()),
        "recorded_last": str(last.date()),
        "current_first": str(current.index[0].date()),
        "current_last": str(current.index[-1].date()),
        "common_sessions": int(len(common)),
        "changed_sessions": int(len(changed)),
        "tolerance": tol,
        "max_abs_difference": float(difference.max()) if len(difference) else None,
        "mean_abs_difference": float(difference.mean()) if len(difference) else None,
        "first_changed": str(changed.index[0].date()) if len(changed) else None,
        "last_changed": str(changed.index[-1].date()) if len(changed) else None,
        "correlation": float(np.corrcoef(recorded.loc[common], current.loc[common])[0, 1]) if len(common) > 2
        else None,
        "removed_sessions": int(len(removed)),
        "removed_first_dates": dates(removed),
        "added_before_recorded_first": int((added < first).sum()),
        "added_inside_recorded_window": int(((added >= first) & (added <= last)).sum()),
        "added_inside_first_dates": dates(added[(added >= first) & (added <= last)]),
        "added_after_recorded_last": int((added > last).sum()),
    }


# --------------------------------------------------------------------- volatility-scaled momentum


def recorded_series(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The recorded scaled reference (scale, gross) and scaled in-house frame from ``series.csv``."""
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    reference = frame[["reference|scale", "reference|gross"]].dropna(how="all")
    reference.columns = ["scale", "gross"]
    in_house = frame[[f"in_house|{name}" for name in IN_HOUSE_COLUMNS]].dropna(how="all")
    in_house.columns = list(IN_HOUSE_COLUMNS)
    for name, part in (("reference", reference), ("in_house", in_house)):
        if part.empty or part.isna().any().any():
            raise ReproductionError(f"{path}: the {name} columns are empty or have gaps")
    return reference, in_house


def in_house_summary(in_house: pd.DataFrame, reg: Mapping[str, Any], per_year: int) -> dict[str, Any]:
    """The ``in_house`` block of the recorded summary, recomputed (``vol_momentum_replication.evaluate``)."""
    return {
        "first": str(in_house.index[0].date()),
        "last": str(in_house.index[-1].date()),
        "n": int(len(in_house)),
        "mean_scale": float(in_house["scale"].mean()),
        "scale_min": float(in_house["scale"].min()),
        "scale_max": float(in_house["scale"].max()),
        "scaled": fl.annualized_stats(in_house["net"], per_year),
        "unscaled_same_window": fl.annualized_stats(in_house["unscaled_net"], per_year),
        "n_eff": reg["implementation"]["n_eff_unscaled"],
    }


def criteria(reference_gross: pd.Series, in_house: pd.DataFrame, reg: Mapping[str, Any], rf_annual: float,
             rf_source: str) -> dict[str, Any]:
    """The registered four criteria, with the registered functions, on the given series and rf."""
    source, settings = reg["source"], reg["criteria"]
    per_year = int(settings["sessions_per_year"])
    bar = vm.publication_bar(reference_gross, source, settings)
    fid = vm.fidelity(in_house["net"], reference_gross, float(settings["fidelity_floor"]))
    econ = vm.economics(fid["daily_corr"], bar["prior_sharpe"], in_house, rf_annual, float(source["target_vol"]),
                        per_year, long_over_equity=0.5 * float(in_house["scale"].mean()))
    # The registered function labels rf as coming from the pinned file; say where this rf came from instead.
    econ["hurdle_inputs"]["rf_source"] = rf_source
    contra = vm.no_contradiction(in_house, econ["figure_f3"], reg["bootstrap"], per_year)
    return plain({
        "publication_bar": bar,
        "fidelity": fid,
        "economics": econ,
        "no_contradiction": contra,
        "in_house": in_house_summary(in_house, reg, per_year),
        "promoted": all(part["pass"] for part in (bar, fid, econ, contra)),
    })


def rebuild_in_house(reg: Mapping[str, Any], repo: Path = REPO) -> tuple[pd.DataFrame, dict[str, str]]:
    """A2: the scaled in-house frame from the hash-checked returns and costs, by the registered overlay."""
    impl = reg["implementation"]
    returns_path, costs_path = repo / impl["returns"], repo / impl["costs"]
    hashes = {
        impl["returns"]: require_sha256(returns_path, impl["returns_sha256"], "in-house returns"),
        impl["costs"]: require_sha256(costs_path, impl["costs_sha256"], "in-house costs"),
    }
    unscaled = fl.load_in_house(returns_path)
    costs = pd.read_csv(costs_path, index_col=0)
    costs.index = vm.ny_session_dates(costs.index)
    if str(unscaled.index[0].date()) != impl["refresh_phase_start"]:
        raise ReproductionError(f"in-house series starts {unscaled.index[0].date()}, registered "
                                f"{impl['refresh_phase_start']}")
    frame = vm.scaled_in_house(unscaled, costs["gross"], costs["total"], reg["source"],
                               float(impl["scale_change_cost_bps"]))
    return frame, hashes


def frame_difference(recorded: pd.DataFrame, rebuilt: pd.DataFrame, tol: float = TOLERANCE) -> dict[str, Any]:
    same_index = bool(recorded.index.equals(rebuilt.index))
    columns: dict[str, Any] = {}
    if same_index:
        for name in recorded.columns:
            columns[name] = float((rebuilt[name] - recorded[name]).abs().max())
    match = same_index and all(value <= tol for value in columns.values())
    return {"same_sessions": same_index, "sessions": int(len(recorded)), "max_abs_difference": columns,
            "tolerance": tol, "match": bool(match)}


def implied_unscaled(reference: pd.DataFrame) -> pd.Series:
    """The unscaled reference return implied by the recorded scale and scaled return."""
    if (reference["scale"] <= 0).any():
        raise ReproductionError("recorded reference has a non-positive scale")
    return reference["gross"] / reference["scale"]


def reference_consistency(reference: pd.DataFrame, reg: Mapping[str, Any], tol: float = TOLERANCE
                          ) -> dict[str, Any]:
    """A3: re-apply the registered rule to the implied unscaled reference; compare the scales it can check.

    The recorded reference starts at the warm-up's end. When that is a refresh
    date, the refresh grid inside the window is the registered one, and every
    decision whose lookback lies wholly inside the window is checkable: those
    effective from ``first_scored_index`` on.
    """
    source = reg["source"]
    warmup, refresh = int(reg["reference"]["warmup_sessions"]), int(source["refresh_sessions"])
    if warmup % refresh:
        raise ReproductionError(f"warm-up {warmup} is not on the {refresh}-session refresh grid")
    recomputed = vm.rule_scale(implied_unscaled(reference).to_numpy(dtype=float), source)
    start = vm.first_scored_index(source)
    difference = np.abs(recomputed[start:] - reference["scale"].to_numpy(dtype=float)[start:])
    return {
        "checked_from": str(reference.index[start].date()),
        "checked_sessions": int(difference.size),
        "unchecked_sessions": int(start),
        "max_abs_scale_difference": float(difference.max()),
        "tolerance": tol,
        "match": bool(difference.max() <= tol),
    }


def recompute_rf(cash: pd.Series, last: pd.Timestamp, reg: Mapping[str, Any]) -> float:
    settings = reg["criteria"]
    window = cash.loc[:last].iloc[-int(settings["hurdle_window_sessions"]):]
    return float(window.mean() * int(settings["sessions_per_year"]))


def layer_a(reg: Mapping[str, Any], summary: Mapping[str, Any], repo: Path = REPO
            ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, str]]:
    reference, in_house = recorded_series(repo / VM_RECORD / "series.csv")
    rf = float(summary["economics"]["rf_annual"])
    a1 = criteria(reference["gross"], in_house, reg, rf, f"rf from record: {VM_RECORD / 'summary.json'} "
                  "economics.rf_annual")
    recorded = {key: summary[key] for key in SUMMARY_KEYS}
    rebuilt, hashes = rebuild_in_house(reg, repo)
    a2 = frame_difference(in_house, rebuilt[list(IN_HOUSE_COLUMNS)])
    a3 = reference_consistency(reference, reg)
    block = {
        "a1_criteria": {"rf_basis": "from record", "rf_annual": rf, "values": a1,
                        "comparison": comparison_block(recorded, a1)},
        "a2_in_house_rebuild": a2,
        "a3_reference_rule": a3,
        "not_compared": {key: summary.get(key) for key in summary if key not in SUMMARY_KEYS},
    }
    block["match"] = bool(block["a1_criteria"]["comparison"]["match"] and a2["match"] and a3["match"])
    return block, reference, rebuilt, hashes


def layer_b(reg: Mapping[str, Any], summary: Mapping[str, Any], reference: pd.DataFrame, in_house: pd.DataFrame,
            fetched: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ref_cfg, source = reg["reference"], reg["source"]
    momentum_table, momentum_vintage = fl.parse_french(Path(fetched[ref_cfg["momentum_source"]]["path"]).read_bytes())
    cash_table, cash_vintage = fl.parse_french(Path(fetched[ref_cfg["cash_source"]]["path"]).read_bytes())
    momentum = momentum_table[ref_cfg["momentum_column"]].dropna()
    cash = cash_table[ref_cfg["cash_column"]].dropna()
    recorded_last = reference.index[-1]
    rf_recorded = float(summary["economics"]["rf_annual"])
    latest = vm.scaled_reference(momentum, source, int(ref_cfg["warmup_sessions"]))
    in_house_frame = in_house[list(IN_HOUSE_COLUMNS)]
    windows: dict[str, Any] = {}
    for name, cut in (("b1_recorded_window", recorded_last), ("b2_latest", momentum.index[-1])):
        series = vm.scaled_reference(momentum.loc[:cut], source, int(ref_cfg["warmup_sessions"]))
        rf_new = recompute_rf(cash, cut, reg)
        label = f"vintage '{momentum_vintage}'"
        windows[name] = {
            "label": f"reproduction on {label}; not a registered result",
            "reference_last": str(series.index[-1].date()),
            "rf_from_record": {"rf_annual": rf_recorded, "values": criteria(
                series["gross"], in_house_frame, reg, rf_recorded,
                f"rf from record: {VM_RECORD / 'summary.json'} economics.rf_annual")},
            "rf_recomputed": {"rf_annual": rf_new, "values": criteria(
                series["gross"], in_house_frame, reg, rf_new,
                f"French RF of the downloaded file ({cash_vintage}), mean of the "
                f"{reg['criteria']['hurdle_window_sessions']} sessions ending {cut.date()}, annualized")},
        }
    return {
        "momentum_vintage": momentum_vintage,
        "cash_vintage": cash_vintage,
        "recorded_vintage": summary.get("momentum_vintage"),
        "raw_momentum_drift": drift(implied_unscaled(reference), momentum),
        "scaled_reference_drift": drift(reference["gross"], latest["gross"]),
        "scale_drift": drift(reference["scale"], latest["scale"]),
        "windows": windows,
    }


# --------------------------------------------------------------------- factor lab


def factor_lab_inputs(reg: Mapping[str, Any], vm_reg: Mapping[str, Any], repo: Path = REPO
                      ) -> tuple[dict[str, Any], dict[str, str]]:
    """The factor-lab registration with in-house paths made absolute, and their hashes, checked where recorded."""
    expected = {LIVE_BOOK_RETURNS.as_posix(): LIVE_BOOK_RETURNS_SHA256,
                vm_reg["implementation"]["returns"]: vm_reg["implementation"]["returns_sha256"]}
    resolved = copy.deepcopy(dict(reg))
    hashes: dict[str, str] = {}
    for entry in resolved["fidelity"]:
        relative = entry["in_house"]
        path = repo / relative
        if not path.exists():
            raise ReproductionError(f"factor-lab in-house input {relative} is missing from this checkout")
        hashes[relative] = (require_sha256(path, expected[relative], f"factor-lab input {relative}")
                            if relative in expected else sha256_file(path))
        entry["in_house"] = str(path)
    return resolved, hashes


def cut_to_recorded(data: fl.PublicData, recorded_priors: Mapping[str, Any]) -> fl.PublicData:
    """Each daily series cut at the last date the recorded priors give for it."""
    missing = sorted(set(data.series) - set(recorded_priors))
    if missing:
        raise ReproductionError(f"series without a recorded prior: {missing}")
    series = {sid: values.loc[: recorded_priors[sid]["full"]["last"]] for sid, values in data.series.items()}
    return fl.PublicData(series=series, market_total=data.market_total, jkp_large_monthly=data.jkp_large_monthly,
                         vintages=data.vintages)


def series_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return plain({"full_sharpe": record["full"]["sharpe"], "full_last": record["full"]["last"],
                  "prior_sharpe": record.get("prior_sharpe"), "prior_pass": record.get("prior_pass")})


def fidelity_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    stats = record.get("daily") or record.get("monthly") or {}
    return plain({"corr": stats.get("corr"), "n": stats.get("n"), "meets_floor": record.get("meets_floor")})


def fidelity_key(record: Mapping[str, Any]) -> str:
    return f"{record['book']}|{record['reference']}|{record['frequency']}"


def correlation_differences(recorded_dir: Path, reproduced_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for recorded_file in sorted(recorded_dir.glob("*.csv")):
        reproduced_file = reproduced_dir / recorded_file.name
        if not reproduced_file.exists():
            out[recorded_file.name] = {"reproduced": "<missing>"}
            continue
        recorded = pd.read_csv(recorded_file, index_col=0)
        reproduced = pd.read_csv(reproduced_file, index_col=0)
        same_labels = recorded.index.equals(reproduced.index) and recorded.columns.equals(reproduced.columns)
        out[recorded_file.name] = {
            "same_labels": bool(same_labels),
            "max_abs_difference": float((reproduced - recorded).abs().max().max()) if same_labels else None,
        }
    return out


def factor_lab_layer(reg: Mapping[str, Any], vm_reg: Mapping[str, Any], fetched: Mapping[str, Mapping[str, Any]],
                     out: Path, repo: Path = REPO) -> tuple[dict[str, Any], dict[str, str]]:
    resolved, hashes = factor_lab_inputs(reg, vm_reg, repo)
    recorded_priors = json.loads((repo / FL_RECORD / "priors.json").read_text())["priors"]
    recorded_fidelity = json.loads((repo / FL_RECORD / "fidelity.json").read_text())
    recorded_by_key = {fidelity_key(record): record for record in recorded_fidelity["records"]}
    data = fl.build_public_data(resolved, fetched)
    result: dict[str, Any] = {"vintages": data.vintages, "windows": {}}
    for label, window in (("recorded_window", cut_to_recorded(data, recorded_priors)), ("current", data)):
        target = out / "factor_lab" / label
        target.mkdir(parents=True, exist_ok=False)
        priors = fl.run_priors(resolved, window, target)
        fidelity = fl.run_fidelity(resolved, window, priors["priors"], target)
        reproduced_by_key = {fidelity_key(record): record for record in fidelity["records"]}
        result["windows"][label] = {
            "label": "reproduction on the downloaded vintages; not a registered result",
            "priors": {sid: comparison_block(recorded_priors[sid], priors["priors"].get(sid))
                       for sid in recorded_priors},
            "fidelity": {key: comparison_block(record, reproduced_by_key.get(key))
                         for key, record in recorded_by_key.items()},
            "correlations": correlation_differences(repo / FL_RECORD / "correlations", target / "correlations"),
            "series_summary": {sid: series_summary(priors["priors"][sid]) for sid in recorded_priors},
            "fidelity_summary": {key: fidelity_summary(record) for key, record in reproduced_by_key.items()},
        }
    result["recorded_series_summary"] = {sid: series_summary(record) for sid, record in recorded_priors.items()}
    result["recorded_fidelity_summary"] = {key: fidelity_summary(record) for key, record in recorded_by_key.items()}
    return result, hashes


# --------------------------------------------------------------------- downloads and manifest


def download_plan(vm_reg: Mapping[str, Any], fl_reg: Mapping[str, Any] | None) -> dict[str, str]:
    ref_cfg = vm_reg["reference"]
    plan = {ref_cfg["momentum_source"]: ref_cfg["momentum_url"], ref_cfg["cash_source"]: ref_cfg["cash_url"]}
    if fl_reg is not None:
        for source_id, spec in fl_reg["sources"].items():
            if source_id in plan and plan[source_id] != spec["url"]:
                raise ReproductionError(f"{source_id} has different URLs in the two registrations")
            plan[source_id] = spec["url"]
    return plan


def annotate_downloads(fetched: Mapping[str, Mapping[str, Any]], vm_reg: Mapping[str, Any],
                       recorded_sources: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each download with its vintage line, data dates and hash matches against the record."""
    ref_cfg = vm_reg["reference"]
    pins = {ref_cfg["momentum_source"]: ref_cfg["momentum_sha256"], ref_cfg["cash_source"]: ref_cfg["cash_sha256"]}
    out: dict[str, dict[str, Any]] = {}
    for source_id, entry in fetched.items():
        record = dict(entry)
        payload = Path(entry["path"]).read_bytes()
        if "ken.french" in entry["url"]:
            table, vintage = fl.parse_french(payload)
        else:
            table, vintage = fl.parse_jkp(payload), None
        record.update(vintage=vintage, first_date=str(table.index[0].date()), last_date=str(table.index[-1].date()))
        if source_id in pins:
            record["registration_sha256"] = pins[source_id]
            record["matches_registration"] = entry["sha256"] == pins[source_id]
        recorded = recorded_sources.get(source_id)
        if recorded is not None:
            record["factor_lab_sha256"] = recorded["sha256"]
            record["factor_lab_bytes"] = recorded["bytes"]
            record["matches_factor_lab_record"] = entry["sha256"] == recorded["sha256"]
        out[source_id] = record
    return out


# --------------------------------------------------------------------- report


def _num(value: float | None, spec: str) -> str:
    return "None" if value is None else format(value, spec)


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


REPORT_ROWS = (
    ("prior Sharpe (post-publication)", ("publication_bar", "prior_sharpe")),
    ("post-publication t", ("publication_bar", "post_publication", "t")),
    ("full-sample Sharpe", ("publication_bar", "full", "sharpe")),
    ("reference last date", ("publication_bar", "full", "last")),
    ("daily tracking correlation", ("fidelity", "daily_corr")),
    ("cost drag (Sharpe)", ("economics", "cost_drag_sharpe")),
    ("criterion-3 figure", ("economics", "figure_f3")),
    ("rf (annual)", ("economics", "rf_annual")),
    ("hurdle (Sharpe)", ("economics", "hurdle_sharpe")),
    ("economics pass", ("economics", "pass")),
    ("in-house interval upper", ("no_contradiction", "annualized", "upper")),
    ("all four pass", ("promoted",)),
)


def _get(values: Mapping[str, Any], keys: Sequence[str]) -> Any:
    item: Any = values
    for key in keys:
        if not isinstance(item, Mapping) or key not in item:
            return None
        item = item[key]
    return item


def report(manifest: Mapping[str, Any], vol: Mapping[str, Any], lab: Mapping[str, Any] | None) -> str:
    a = vol["layer_a"]
    lines = [FIRST_LINE, "", "# Public-factor reproduction", "",
             f"Code commit {manifest['code_commit']} (tree clean: {manifest['tree_clean']}). Design: "
             "`docs/public_reproduction_design.md`. Recorded numbers come from `results/2026-09-vol-momentum/` and "
             "`results/2026-09-factor-lab/`; the registered verdicts stand as recorded.", "",
             "## Layer A (offline)", "",
             f"- A1, four criteria from the tracked series: {a['a1_criteria']['comparison']['leaves_compared']} "
             f"recorded values compared, {a['a1_criteria']['comparison']['leaves_differing']} differ beyond "
             f"{TOLERANCE:g}.",
             f"- A2, in-house series rebuilt from tracked inputs: match {a['a2_in_house_rebuild']['match']}, "
             f"largest difference {max(a['a2_in_house_rebuild']['max_abs_difference'].values(), default=None)}.",
             f"- A3, recorded reference against the registered rule: match {a['a3_reference_rule']['match']} over "
             f"{a['a3_reference_rule']['checked_sessions']} sessions from {a['a3_reference_rule']['checked_from']}, "
             f"largest scale difference {_num(a['a3_reference_rule']['max_abs_scale_difference'], '.3g')}.", ""]
    for row in a["a1_criteria"]["comparison"]["differing"][:20]:
        lines.append(f"  - differs: `{row['path']}` recorded {row['recorded']!r}, reproduced {row['reproduced']!r}")
    downloads = manifest.get("downloads") or {}
    if downloads:
        lines += ["## Downloads", "", "| source | bytes | sha256 | Last-Modified | vintage | last date | matches "
                  "registration | matches factor-lab record |", "|---|---|---|---|---|---|---|---|"]
        for source_id, entry in downloads.items():
            lines.append(f"| {source_id} | {entry['bytes']} | `{entry['sha256'][:12]}…` | "
                         f"{entry.get('last_modified')} | {entry.get('vintage')} | {entry.get('last_date')} | "
                         f"{entry.get('matches_registration', '—')} | {entry.get('matches_factor_lab_record', '—')} |")
        lines.append("")
    b = vol.get("layer_b")
    if b:
        raw, scaled = b["raw_momentum_drift"], b["scaled_reference_drift"]
        lines += ["## Layer B (current French vintage)", "",
                  f"Recorded vintage: {b['recorded_vintage']!r}. Downloaded: {b['momentum_vintage']!r}.", "",
                  f"- Unscaled momentum, {raw['recorded_first']} to {raw['recorded_last']}: "
                  f"{raw['changed_sessions']} of {raw['common_sessions']} common sessions differ beyond "
                  f"{TOLERANCE:g} (largest {_num(raw['max_abs_difference'], '.2g')}, first {raw['first_changed']}, "
                  f"last {raw['last_changed']}); {raw['added_inside_recorded_window']} sessions added and "
                  f"{raw['removed_sessions']} removed inside that window, {raw['added_after_recorded_last']} added "
                  f"after it. The record keeps the reference only from the end of the warm-up, so the downloaded "
                  f"file's first {raw['added_before_recorded_first']} sessions have no recorded counterpart.",
                  f"- Scaled reference: {scaled['changed_sessions']} of {scaled['common_sessions']} common sessions "
                  f"differ (largest {_num(scaled['max_abs_difference'], '.2g')}, correlation "
                  f"{_num(scaled['correlation'], '.9f')}).",
                  "",
                  "B1 is the downloaded file cut at the recorded last date, so it differs from the record only by "
                  "history revisions. B2 runs through the downloaded file's last date, adding the extension. Numbers "
                  "and pass flags on the downloaded vintage describe what the registered tests show on that input. "
                  "They are not registered results, do not replace the recorded ones, and change no verdict: "
                  "`vol_momentum_v1` stands as recorded and its forward stop rule governs it.", "",
                  "| quantity | recorded | A1 | B1 rf record | B1 rf recomputed | B2 rf record | B2 rf recomputed |",
                  "|---|---|---|---|---|---|---|"]
        w = b["windows"]
        recorded_summary = vol["recorded"]
        for label, keys in REPORT_ROWS:
            cells = [_get(recorded_summary, keys), _get(a["a1_criteria"]["values"], keys),
                     _get(w["b1_recorded_window"]["rf_from_record"]["values"], keys),
                     _get(w["b1_recorded_window"]["rf_recomputed"]["values"], keys),
                     _get(w["b2_latest"]["rf_from_record"]["values"], keys),
                     _get(w["b2_latest"]["rf_recomputed"]["values"], keys)]
            lines.append(f"| {label} | " + " | ".join(_fmt(cell) for cell in cells) + " |")
        lines.append("")
        flips = [(window, rf, flag) for window in ("b1_recorded_window", "b2_latest")
                 for rf in ("rf_from_record", "rf_recomputed")
                 for flag, keys in (("economics pass", ("economics", "pass")), ("all four pass", ("promoted",)))
                 if _get(w[window][rf]["values"], keys) != _get(recorded_summary, keys)]
        for window, rf, flag in flips:
            lines.append(f"- On the downloaded vintage, `{flag}` differs from the record in {window} with "
                         f"{rf.replace('_', ' ')}.")
        if flips:
            lines.append("")
    if lab:
        recorded_window, current = lab["windows"]["recorded_window"], lab["windows"]["current"]
        recorded_series_summary = lab["recorded_series_summary"]
        lines += ["## Factor lab (uncounted measurements)", "",
                  "Full-sample Sharpe and prior Sharpe: recorded, recomputed on the downloaded files cut at each "
                  "series' recorded last date, and recomputed through the downloaded files' last date. Series "
                  "without a publication date have no prior (descriptive only).", "",
                  "| series | full Sharpe recorded | cut | current | prior recorded | cut | current |",
                  "|---|---|---|---|---|---|---|"]
        for sid, entry in recorded_series_summary.items():
            cut, now = recorded_window["series_summary"][sid], current["series_summary"][sid]
            lines.append(f"| {sid} | {_fmt(entry['full_sharpe'])} | {_fmt(cut['full_sharpe'])} | "
                         f"{_fmt(now['full_sharpe'])} | {_fmt(entry['prior_sharpe'])} | {_fmt(cut['prior_sharpe'])} | "
                         f"{_fmt(now['prior_sharpe'])} |")
        lines += ["", "| tracking pair | correlation recorded | cut | current |", "|---|---|---|---|"]
        for key, entry in lab["recorded_fidelity_summary"].items():
            lines.append(f"| {key.replace('|', ' / ')} | {_fmt(entry['corr'])} | "
                         f"{_fmt(_get(recorded_window['fidelity_summary'], (key, 'corr')))} | "
                         f"{_fmt(_get(current['fidelity_summary'], (key, 'corr')))} |")
        lines.append("")
        for label in ("recorded_window", "current"):
            entries = lab["windows"][label]["correlations"]
            unmatched = sorted(name for name, entry in entries.items() if entry.get("max_abs_difference") is None)
            largest = max((entry["max_abs_difference"] for name, entry in entries.items() if name not in unmatched),
                          default=float("nan"))
            lines.append(f"- {label}: largest difference across {len(entries) - len(unmatched)} correlation "
                         f"matrices {largest:.2g}" + (f"; not comparable (labels differ or file missing): "
                                                      f"{', '.join(unmatched)}." if unmatched else "."))
        for label in ("recorded_window", "current"):
            window = lab["windows"][label]
            differing_priors = sum(1 for block in window["priors"].values() if not block["match"])
            differing_fidelity = sum(1 for block in window["fidelity"].values() if not block["match"])
            lines.append(f"- {label}: {differing_priors} of {len(window['priors'])} series and "
                         f"{differing_fidelity} of {len(window['fidelity'])} tracking pairs differ from the record "
                         f"beyond {TOLERANCE:g}.")
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- driver


def run(out: Path, *, offline: bool, with_factor_lab: bool, argv: Sequence[str] = (), repo: Path = REPO
        ) -> dict[str, Any]:
    if offline and with_factor_lab:
        raise ReproductionError("--with-factor-lab downloads the factor-lab sources; it cannot run --offline")
    out = prepare_out(out, repo)
    started = datetime.now(timezone.utc).isoformat()
    git = current_git_state(repo)
    vm_reg = fl.load_registration(repo / VM_REGISTRATION)
    fl_reg = fl.load_registration(repo / FL_REGISTRATION) if with_factor_lab else None
    summary = json.loads((repo / VM_RECORD / "summary.json").read_text())
    recorded_sources = json.loads((repo / FL_RECORD / "sources.json").read_text())["sources"]

    a, reference, rebuilt, hashes = layer_a(vm_reg, summary, repo)
    vol: dict[str, Any] = {"recorded": {key: summary[key] for key in SUMMARY_KEYS}, "layer_a": a}
    downloads: dict[str, Any] = {}
    lab = None
    if not offline:
        fetched = fetch(download_plan(vm_reg, fl_reg), out / "cache")
        downloads = annotate_downloads(fetched, vm_reg, recorded_sources)
        vol["layer_b"] = layer_b(vm_reg, summary, reference, rebuilt, fetched)
        if fl_reg is not None:
            lab, lab_hashes = factor_lab_layer(fl_reg, vm_reg, fetched, out, repo)
            hashes.update(lab_hashes)
            (out / "priors_diff.json").write_text(json.dumps(plain(lab), indent=2, sort_keys=True))
    inputs = {str(path): sha256_file(repo / path) for path in (
        VM_REGISTRATION, FL_REGISTRATION, VM_RECORD / "summary.json", VM_RECORD / "series.csv",
        FL_RECORD / "sources.json", FL_RECORD / "priors.json", FL_RECORD / "fidelity.json")}
    inputs.update(hashes)
    manifest = {
        "statement": FIRST_LINE,
        "design": "docs/public_reproduction_design.md",
        "command": ["python", "-m", "research.scripts.reproduce_public", *argv],
        "code_commit": git.commit,
        "tree_clean": None if git.dirty is None else not git.dirty,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "layers": ["A"] + ([] if offline else ["B"]) + (["factor_lab"] if lab is not None else []),
        "layer_a_match": a["match"],
        "tolerance": TOLERANCE,
        "inputs_sha256": inputs,
        "downloads": downloads,
    }
    (out / "vol_momentum.json").write_text(json.dumps(plain(vol), indent=2, sort_keys=True))
    (out / "manifest.json").write_text(json.dumps(plain(manifest), indent=2, sort_keys=True))
    (out / "report.md").write_text(report(manifest, plain(vol), plain(lab) if lab is not None else None))
    return {"manifest": manifest, "vol_momentum": vol, "factor_lab": lab}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uncounted reproduction of the published-factor evidence "
                                                 "(docs/public_reproduction_design.md).")
    parser.add_argument("--out", type=Path, required=True, help="an empty directory outside results/, trials/ "
                                                                "and data/public/")
    parser.add_argument("--offline", action="store_true", help="layer A only; no download")
    parser.add_argument("--with-factor-lab", action="store_true",
                        help="also rerun the factor lab's priors and fidelity parts on the current sources")
    args = parser.parse_args(argv)
    result = run(args.out, offline=args.offline, with_factor_lab=args.with_factor_lab,
                 argv=list(sys.argv[1:] if argv is None else argv))
    manifest = result["manifest"]
    print(json.dumps({"out": str(Path(args.out).resolve()), "layers": manifest["layers"],
                      "layer_a_match": manifest["layer_a_match"],
                      "downloads": {key: {"sha256": entry["sha256"][:12],
                                          "matches_registration": entry.get("matches_registration"),
                                          "matches_factor_lab_record": entry.get("matches_factor_lab_record")}
                                    for key, entry in manifest["downloads"].items()}}, indent=2))
    return 0 if manifest["layer_a_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
