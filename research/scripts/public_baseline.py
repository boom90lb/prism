"""Public baseline: tranched, volatility-scaled 12-1 momentum (selection set ``public_baseline_v1``).

Registration ``registrations/2026-10-public-baseline.toml``; design, criteria and
limits ``docs/public_baseline_design.md``. This module is the set's only driver.
It runs every counted row through the canonical engine, one at a time, and reads
the criteria off what the rows recorded. It never chooses a configuration: the
rows, their order and every threshold come from the registration.

Run from the repository root (the workspace)::

    python -m research.scripts.public_baseline --part pin
    python -m research.scripts.public_baseline --part check --dataset <frozen dataset>
    python -m research.scripts.public_baseline --part rows --dataset <frozen dataset>
    python -m research.scripts.public_baseline --part capacity --dataset <frozen dataset>
    python -m research.scripts.public_baseline --part all --dataset <frozen dataset>

Parts
-----
``pin``
    Fetches the two Kenneth French files named by :data:`FRENCH_URLS` into the
    vintage cache and prints what a registration pins (sha256, bytes,
    ``Last-Modified``, the CRSP vintage line). Hashing is not an evaluation: it
    computes no statistic and needs no registration.
``check``
    Validates the registration and every input it names without evaluating
    anything: the TOML's keys and rows, the dataset's manifest id and vintage
    (the manifest only, no bars), the cost stack, the fold table (built on the
    session calendar), one specification per row, the parity series, and
    whether the pinned French files are in the cache. Run it before committing
    the registration: an edited registration is a new registration.
``rows``
    Refuses unless the registration, the design document and every input file
    they name (cost stack, policy files, parity series) are committed unchanged
    at HEAD. Verifies the French files against their pinned hashes
    (fetching them into the cache if absent), runs P00 first and applies the
    parity gate, then runs the remaining rows in registration order, then
    writes the summary once every row has finished.
``summary``
    Recomputes the summary from the rows' recorded outputs. Evaluates nothing.
``capacity``
    The uncounted crowding re-pricing of design section 9, from the recorded
    trades of the priced rows. Adds no ledger row and may not select.
``all``
    ``rows`` then ``capacity``.

Where it writes
---------------
Everything is inside the workspace root (``--workspace``, default the current
directory, which must be the repository checkout):

* ``results/2026-10-public-baseline/`` -- the set's outputs (design section 11);
* ``trials/public_baseline_v1.jsonl`` -- the ledger, appended by the engine (one
  ``started`` and one terminal row per evaluated row) and, for a row whose
  specification cannot be built, by this driver (one ``failed`` row);
* ``results/<run_id>/``, ``state/fits/``, ``runs/``, ``logs/`` -- what
  ``prism.evaluate.walk_forward_evaluate`` writes for every run it records;
* ``data/public/vintages/<sha256[:12]>/`` (``[reference] cache_dir``) -- the
  French files, written once per hash and never overwritten.

The frozen dataset (``--dataset``, default ``[implementation] dataset.path``) is
opened read-only and may live outside the workspace. A write target inside the
dataset, or outside the workspace, is refused.

Registration keys this driver reads
-----------------------------------
The tables of ``docs/public_baseline_design.md`` section 2b, read strictly: an
unknown key is refused, so a misspelling cannot pass as a default. The keys
section 2b leaves open are fixed here:

* ``[implementation] dataset = { path, manifest_id, vintage }``: the frozen
  dataset of ``characteristics_v1`` (``registrations/2026-09-characteristics.toml``
  freezes ``data_id`` = the manifest id).
* ``[implementation] folds = { test_sessions, formation_sessions }``: contiguous
  test windows of ``test_sessions`` sessions from ``p00_anchor`` to
  ``scored_last`` (the last may be shorter; ``0`` is one window spanning the
  whole sample), each with the ``formation_sessions`` sessions before it as its
  formation window. The canonical engine buckets each name's spread from its
  fold's formation window and its members at the formation end, so with one
  window a name that joins the index after the anchor is charged the widest
  spread step for the whole run; ``{ test_sessions = 63, formation_sessions =
  312 }`` reproduces folds 4-20 of ``folds/demotion_b1.toml``, the geometry
  ``vol_momentum_v1``'s in-house book was charged under.
* ``[implementation]`` optional: ``universe_index`` (``"sp500"``),
  ``eligibility`` (the ``characteristics_v1`` control's screen), ``policy``
  (that control's profile, execution, monitoring and recovery files) and
  ``tranche_valid_for_sessions`` (twice ``refresh_sessions``; a single-tranche
  row keeps the control's ``refresh_sessions``).
* ``[reference]`` optional: ``momentum_column`` (``"Mom"``), ``cash_column``
  (``"RF"``), ``cash_bytes``, ``cash_last_modified``, ``cash_vintage_line``.
* ``[criteria]`` optional: ``prior_cut`` (``"2025-12-31"``), the date the
  post-publication prior is also reported cut at.
* ``[capacity]`` optional: ``rows`` (T21, P00, D-split, D-liq), ``users_at``
  ($100k and $1M) and ``thin_dollar_volume`` ($100M).

Interpretations, stated so a reader can disagree with them
----------------------------------------------------------
* Criterion 5's "no session's gross exceeds 2.0" is read on the decided
  target: the rule's ceiling binds each decision's target, and between one
  tranche's refreshes the held book drifts with prices and can close above it.
  The held maximum and the number of sessions above the ceiling are reported
  beside the criterion.
* Criterion 6's bound is ``tranching_paired_lower_bound``: the criterion fails
  when the whole paired interval lies below it.
* The N4 check rebuilds every session's change in equity from held shares,
  marks, fills, events and charged costs (the identity of
  ``tests/test_accounting_conservation.py``) on the simulation the engine ran.
  The driver observes that simulation by wrapping
  ``prism.evaluate.simulate`` for the duration of one call; the wrapper passes
  arguments and result through unchanged and the driver refuses unless it saw
  exactly one simulation.

A diagnostic universe partition (rows ``D-part*``) is applied by filtering the
dataset's point-in-time membership through ``research.scripts.universe_partition``
and declaring the filter as a manifest transformation, so the partitioned
dataset has its own ``data_id``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from prism.evaluate import CanonicalSettings, resolve_log_path, walk_forward_evaluate
from prism.execution.cost_model import CostStack
from prism.execution.simulate import FILL_MODELS, FillPolicy
from prism.io import sessions
from prism.io.dataset import Dataset
from prism.portfolio.rounding import RoundingSettings
from prism.portfolio.vol_scale import base_book_returns
from prism.spec import load_spec
from prism.validation.capacity import (
    CrowdScenario,
    crowd_capacity_table,
    crowd_impact_cost,
)
from prism.validation.compare import SharpeBootstrapSpec, sharpe_difference_interval
from prism.validation.hurdle import cash_toll, tbill_hurdle
from prism.validation.metrics import (
    EULER_GAMMA,
    deflated_sharpe_ratio_with_n,
    effective_breadth_from_cov,
    periodic_sharpe,
    probabilistic_sharpe_ratio,
    rank_information_coefficient,
)
from prism.validation.trials import (
    FAILED,
    FINISHED,
    INCOMPLETE,
    STARTED,
    RunRecorder,
    TrialLog,
    current_git_state,
    stable_json_dumps,
)
from prism.validation.walk_forward import FoldTable
from prism.workspace import Workspace
from research.scripts import universe_partition
from research.scripts.factor_lab import (
    USER_AGENT,
    LabError,
    annualized_stats,
    parse_french,
)
from research.scripts.registered_read import contribution_breadth
from research.scripts.vol_momentum_replication import (
    fidelity,
    no_contradiction,
    publication_bar,
    scaled_reference,
)

REGISTRATION = Path("registrations/2026-10-public-baseline.toml")
DESIGN_DOC = Path("docs/public_baseline_design.md")
OUT_DIR = Path("results/2026-10-public-baseline")
SELECTION_SET = "public_baseline_v1"

#: The two public files a registration pins (``--part pin`` fetches these).
FRENCH_URLS: dict[str, str] = {
    "ff_mom_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
    "ff3_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
}

#: The ``characteristics_v1`` control's eligibility screen (``examples/characteristics/control.toml``).
DEFAULT_ELIGIBILITY: dict[str, Any] = {
    "min_price": 5.0,
    "min_median_dollar_volume": 1_000_000.0,
    "dollar_volume_window": 20,
    "return_history_bars": 252,
    "listing_seasoning_bars": 0,
    "membership_required": True,
    "exclude_asset_classes": [],
}

#: The ``characteristics_v1`` momentum_control arm's policy files, cost stack excepted.
DEFAULT_POLICY: dict[str, str] = {
    "profile": "examples/reference/profiles/experiment_v1.toml",
    "execution": "examples/reference/profiles/execution_alpaca.toml",
    "monitoring": "examples/reference/policies/monitoring.toml",
    "recovery": "examples/reference/policies/recovery.toml",
}

#: That arm's construction block with the band off: the account_state path never
#: applied the declared closed-form band and now refuses one
#: (``docs/construction_band_audit_2026-09-24.md``).
CONTROL_CONSTRUCTION: dict[str, Any] = {
    "kind": "heuristic",
    "step_order": "construct_band_cap_participation",
    "eligibility_exit": "reduce_only",
    "band": {"mode": "off"},
    "participation": {
        "max_participation": 0.05,
        "statistic": "median",
        "window_bars": 20,
        "floor_dollars": 0.0,
        "base": "initial_capital",
    },
    "spread": {"mode": "bucket"},
}

CONSTRUCTION_RULES: dict[str, str] = {"decile_neutral": "decile_long_short"}
LEG_WEIGHTS = re.compile(r"^sqrt_median_dollar_volume_(\d+)$")
ROW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DEFAULT_CAPACITY_ROWS = ("T21", "P00", "D-split", "D-liq")

#: What every row's record says it does not establish (``untested_conclusions``).
UNTESTED: tuple[str, ...] = (
    "In-house returns are price returns: the frozen dataset carries no cash distributions, while the public "
    "reference series is a total return; the dividend wedge is not charged.",
    "The frozen dataset's point-in-time membership is not survivorship-complete; missing names flatter "
    "tracking and cost.",
    "Fills are simulated (next open, or half at the open and half at the close for D-split); no record here "
    "is evidence about actual fills, and paper fills cannot measure impact.",
    "Idle own equity held in a T-bill ETF is not simulated; it enters only through the hurdle.",
)

#: Relative tolerance of the per-session N4 identity, and the absolute floor in dollars.
N4_RELATIVE_TOLERANCE = 1e-9
N4_ABSOLUTE_TOLERANCE = 1e-6
#: Slack on the gross ceiling and on the failure-path bound (relative rounding).
GROSS_TOLERANCE = 1e-9
#: Calendar days within which a buy and a sell of one name count as a wash-sale pair.
WASH_SALE_DAYS = 30
#: Horizons of the rank-IC profile (SPEC section 5).
IC_HORIZONS = (1, 2, 5, 10, 21, 63)


class BaselineError(LabError):
    """A registration, an input or an invariant of this set failed; raised, never swallowed (N7)."""


class ParityFailure(BaselineError):
    """P00 failed the parity gate: the run stops and no verdict is read."""


# ===================================================================== registration


@dataclass(frozen=True)
class Row:
    """One counted row of the set, resolved from ``[rows]``."""

    id: str
    group: str
    tranches: int
    offset: int
    capital: float
    fill_model: str
    to_weights: str = "decile_long_short"
    dollar_volume_window: int | None = None
    keep_fraction: float | None = None
    seed: int | None = None

    @property
    def partitioned(self) -> bool:
        return self.seed is not None


@dataclass(frozen=True)
class Baseline:
    """The registration, validated, with its rows resolved."""

    path: Path
    sha256: str
    raw: Mapping[str, Any]
    rows: tuple[Row, ...]
    base_offset: int

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    def table(self, name: str) -> Mapping[str, Any]:
        return self.raw[name]

    @property
    def refresh(self) -> int:
        return int(self.raw["source"]["refresh_sessions"])

    @property
    def per_year(self) -> int:
        return int(self.raw["criteria"]["sessions_per_year"])

    @property
    def candidate(self) -> str:
        return str(self.raw["candidate"])

    @property
    def phase_ids(self) -> list[str]:
        return [row.id for row in self.rows if row.group == "phases"]

    def row(self, row_id: str) -> Row:
        for row in self.rows:
            if row.id == row_id:
                return row
        raise BaselineError(f"row {row_id!r} is not declared in {self.path}")

    def run_order(self) -> list[Row]:
        """P00 first (the parity gate), then the candidate, then the rest in declaration order."""
        first = self.phase_ids[0]
        lead = [self.row(first), self.row(self.candidate)]
        return lead + [row for row in self.rows if row.id not in (first, self.candidate)]


_TABLE_KEYS: dict[str, tuple[set[str], set[str]]] = {
    "source": (
        {"citation_characteristic", "citation_sizing", "publication", "target_vol", "lookback_sessions",
         "refresh_sessions", "max_scale"},
        set(),
    ),
    "reference": (
        {"momentum_source", "momentum_url", "momentum_sha256", "momentum_bytes", "momentum_last_modified",
         "momentum_vintage_line", "cash_source", "cash_url", "cash_sha256", "cache_dir", "warmup_sessions"},
        {"momentum_column", "cash_column", "cash_bytes", "cash_last_modified", "cash_vintage_line"},
    ),
    "criteria": (
        {"publication_bar_t", "publication_bar_min_years", "post_publication_min_years", "pre_publication_haircut",
         "fidelity_floor", "hurdle_window_sessions", "sessions_per_year", "vol_basis_rule", "sleeve_buffer",
         "sleeve_fee_annual", "buffer_cash_yield", "reported_cash_yields", "parity_min_daily_corr",
         "parity_max_abs_sharpe_diff", "tranching_paired_lower_bound"},
        {"prior_cut"},
    ),
    "implementation": (
        {"dataset", "signal", "construction", "grid_epoch", "p00_anchor", "initial_capital", "share_increment",
         "fill_model", "cost_stack", "scored_last", "parity_series", "folds"},
        {"universe_index", "eligibility", "policy", "tranche_valid_for_sessions"},
    ),
    "guards": ({"min_valid_returns", "max_zero_return_fraction", "require_decision_bar_is_last_bar"}, set()),
    "bootstrap": ({"block_bars", "n_resamples", "seed", "confidence"}, set()),
    "kill": (
        {"first_read_min_sessions", "read_every_sessions", "max_drawdown", "min_forward_fidelity",
         "economics_consecutive_reads", "monitor"},
        set(),
    ),
    "capacity": (
        {"follower_capital", "window_share", "synchronization", "impact_y", "sigma", "selection_forbidden"},
        {"rows", "users_at", "thin_dollar_volume"},
    ),
}
_TOP_KEYS = {"id", "created", "selection_set", "route", "trial_budget", "candidate", "rows", *_TABLE_KEYS}
_ROW_KEYS = {"tranches", "fill_model", "leg_weights", "keep_fraction", "seeds", "initial_capital", "offsets"}
_WINDOW_SHARES = {"open_low", "open_high", "close", "full_day"}


def _keys(table: Any, required: set[str], optional: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(table, Mapping):
        raise BaselineError(f"{where} is {table!r}, not a table")
    missing = sorted(required - set(table))
    if missing:
        raise BaselineError(f"{where} lacks {missing}")
    unknown = sorted(set(table) - required - optional)
    if unknown:
        raise BaselineError(f"{where} declares unknown key(s) {unknown}; a misspelled key would pass as a default")
    return table


def _date(value: Any, where: str) -> dt.date:
    try:
        return value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise BaselineError(f"{where} = {value!r} is not an ISO date") from exc


def _int(value: Any, where: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BaselineError(f"{where} = {value!r} is not an integer")
    if minimum is not None and value < minimum:
        raise BaselineError(f"{where} = {value!r} is below {minimum}")
    return int(value)


def _float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BaselineError(f"{where} = {value!r} is not a finite number")
    return float(value)


def _expand_ids(spec: str) -> list[str]:
    """``"P00..P20"`` -> ``["P00", ..., "P20"]``: one prefix, one zero-padded width."""
    match = re.fullmatch(r"([A-Za-z_-]*)(\d+)\.\.([A-Za-z_-]*)(\d+)", str(spec).strip())
    if match is None or match.group(1) != match.group(3) or len(match.group(2)) != len(match.group(4)):
        raise BaselineError(f"[rows] phases ids = {spec!r} is not of the form 'P00..P20'")
    start, stop, width = int(match.group(2)), int(match.group(4)), len(match.group(2))
    if stop < start:
        raise BaselineError(f"[rows] phases ids = {spec!r} runs backwards")
    return [f"{match.group(1)}{number:0{width}d}" for number in range(start, stop + 1)]


def _expand_range(spec: str) -> list[int]:
    match = re.fullmatch(r"(\d+)\.\.(\d+)", str(spec).strip())
    if match is None or int(match.group(2)) < int(match.group(1)):
        raise BaselineError(f"[rows] phases offsets = {spec!r} is not of the form '0..20'")
    return list(range(int(match.group(1)), int(match.group(2)) + 1))


def _parse_offsets(spec: str, base: int, refresh: int, where: str) -> list[int]:
    """``"r, r+7, r+14"`` -> grid offsets, ``r`` being P00's offset."""
    out = []
    for term in str(spec).split(","):
        match = re.fullmatch(r"r(?:\s*\+\s*(\d+))?", term.strip())
        if match is None:
            raise BaselineError(f"{where} offsets = {spec!r}: term {term!r} is not 'r' or 'r+N'")
        out.append((base + int(match.group(1) or 0)) % refresh)
    return out


def _parse_rows(table: Any, *, refresh: int, base: int, capital: float, fill_model: str) -> tuple[Row, ...]:
    if not isinstance(table, Mapping) or not table:
        raise BaselineError("[rows] is missing or empty")
    rows: list[Row] = []
    for key, entry in table.items():
        where = f"[rows] {key}"
        if key == "phases":
            entry = _keys(entry, {"ids", "tranches", "offsets"}, set(), where)
            if _int(entry["tranches"], f"{where} tranches") != 1:
                raise BaselineError(f"{where}: a phase row is a single-tranche book (tranches = 1)")
            ids, offsets = _expand_ids(entry["ids"]), _expand_range(entry["offsets"])
            if len(ids) != len(offsets) or any(k >= refresh for k in offsets):
                raise BaselineError(f"{where}: {len(ids)} ids against offsets {offsets} on a {refresh}-session cycle")
            rows.extend(Row(row_id, "phases", 1, (base + k) % refresh, capital, fill_model)
                        for row_id, k in zip(ids, offsets))
            continue
        entry = _keys(entry, {"tranches"}, _ROW_KEYS - {"tranches"}, where)
        tranches = _int(entry["tranches"], f"{where} tranches", 1)
        if tranches > refresh:
            raise BaselineError(f"{where}: tranches = {tranches} exceeds the {refresh}-session cycle")
        model = str(entry.get("fill_model", fill_model))
        if model not in FILL_MODELS:
            raise BaselineError(f"{where}: fill_model {model!r} is not one of {FILL_MODELS}")
        to_weights, window = "decile_long_short", None
        if "leg_weights" in entry:
            match = LEG_WEIGHTS.fullmatch(str(entry["leg_weights"]))
            if match is None:
                raise BaselineError(
                    f"{where}: leg_weights {entry['leg_weights']!r} is not 'sqrt_median_dollar_volume_N'")
            to_weights, window = "decile_neutral_sqrt_dv", int(match.group(1))
        size = _float(entry.get("initial_capital", capital), f"{where} initial_capital")
        if size <= 0.0:
            raise BaselineError(f"{where}: initial_capital must be positive")
        if "offsets" in entry:
            declared = _parse_offsets(entry["offsets"], base, refresh, where)
            spacing = refresh // tranches
            expected = [(base + j * spacing) % refresh for j in range(tranches)]
            if declared != expected:
                raise BaselineError(
                    f"{where}: offsets {entry['offsets']!r} resolve to {declared}, but {tranches} tranches on a "
                    f"{refresh}-session cycle sit at {expected} (prism.composition.tranche_plans)"
                )
        common = dict(group=key, tranches=tranches, offset=base, capital=size, fill_model=model,
                      to_weights=to_weights, dollar_volume_window=window)
        if "seeds" in entry or "keep_fraction" in entry:
            if not ("seeds" in entry and "keep_fraction" in entry):
                raise BaselineError(f"{where}: a partition row declares both keep_fraction and seeds")
            keep = _float(entry["keep_fraction"], f"{where} keep_fraction")
            seeds = entry["seeds"]
            if not 0.0 < keep < 1.0 or not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
                raise BaselineError(f"{where}: keep_fraction in (0, 1) and a non-empty list of distinct seeds")
            for number, seed in enumerate(seeds, start=1):
                _int(seed, f"{where} seed", 0)
                if seed >= universe_partition.MAX_SEED:
                    raise BaselineError(f"{where}: seed {seed} is not below 2**63")
                rows.append(Row(f"{key}{number}", keep_fraction=keep, seed=int(seed), **common))
        else:
            rows.append(Row(key, **common))
    ids = [row.id for row in rows]
    duplicated = sorted({row_id for row_id in ids if ids.count(row_id) > 1})
    if duplicated:
        raise BaselineError(f"[rows] declares {duplicated} more than once")
    bad = [row_id for row_id in ids if not ROW_ID.fullmatch(row_id)]
    if bad:
        raise BaselineError(f"[rows] ids {bad} are not plain names (letters, digits, '_', '.', '-')")
    if not any(row.group == "phases" for row in rows):
        raise BaselineError("[rows] declares no phases; P00 is the parity row")
    return tuple(rows)


def parse_registration(raw: Mapping[str, Any], *, path: Path, sha256: str) -> Baseline:
    """Validate the registration strictly and resolve its rows."""
    _keys(raw, _TOP_KEYS - {"created"}, {"created"}, str(path))
    for name, (required, optional) in _TABLE_KEYS.items():
        _keys(raw[name], required, optional, f"{path} [{name}]")
    if raw["selection_set"] != SELECTION_SET:
        raise BaselineError(f"{path} names selection set {raw['selection_set']!r}; this driver runs {SELECTION_SET!r}")
    if raw["route"] != "replication":
        raise BaselineError(f"{path} route is {raw['route']!r}; this set is on the replication route")
    source, impl, criteria, guards = raw["source"], raw["implementation"], raw["criteria"], raw["guards"]
    refresh = _int(source["refresh_sessions"], "[source] refresh_sessions", 1)
    lookback = _int(source["lookback_sessions"], "[source] lookback_sessions", 2)
    for key in ("target_vol", "max_scale"):
        if _float(source[key], f"[source] {key}") <= 0.0:
            raise BaselineError(f"[source] {key} must be positive")
    if _date(impl["grid_epoch"], "[implementation] grid_epoch") != sessions.GRID_EPOCH:
        raise BaselineError(
            f"[implementation] grid_epoch {impl['grid_epoch']} is not prism.io.sessions.GRID_EPOCH "
            f"({sessions.GRID_EPOCH}); the grid positions would not be the ones the engine computes"
        )
    anchor = _date(impl["p00_anchor"], "[implementation] p00_anchor")
    last = _date(impl["scored_last"], "[implementation] scored_last")
    if last <= anchor:
        raise BaselineError("[implementation] scored_last does not follow p00_anchor")
    try:
        base = sessions.grid_position(anchor) % refresh
        sessions.grid_position(last)
    except Exception as exc:
        raise BaselineError(f"[implementation] p00_anchor and scored_last must be sessions: {exc}") from exc
    if _int(guards["min_valid_returns"], "[guards] min_valid_returns") != lookback:
        raise BaselineError(
            "[guards] min_valid_returns must equal [source] lookback_sessions: prism.portfolio.vol_scale.decide_scale "
            "falls back unless the whole window is finite"
        )
    if guards["require_decision_bar_is_last_bar"] is not True:
        raise BaselineError("[guards] require_decision_bar_is_last_bar = false is not a rule decide_scale implements")
    fraction = _float(guards["max_zero_return_fraction"], "[guards] max_zero_return_fraction")
    if not 0.0 <= fraction < 1.0:
        raise BaselineError("[guards] max_zero_return_fraction must lie in [0, 1)")
    dataset = _keys(impl["dataset"], {"path", "manifest_id", "vintage"}, set(), "[implementation] dataset")
    if not re.fullmatch(r"[0-9a-f]{64}", str(dataset["manifest_id"])):
        raise BaselineError("[implementation] dataset.manifest_id is not a sha256 hex digest")
    signal = _keys(impl["signal"], {"node", "lookback", "skip"}, set(), "[implementation] signal")
    _int(signal["lookback"], "[implementation] signal.lookback", 2)
    _int(signal["skip"], "[implementation] signal.skip", 0)
    construction = _keys(impl["construction"], {"rule", "decile", "max_symbol_abs_weight", "raw_gross"}, set(),
                         "[implementation] construction")
    if construction["rule"] not in CONSTRUCTION_RULES:
        raise BaselineError(f"[implementation] construction.rule {construction['rule']!r} is not one of "
                            f"{sorted(CONSTRUCTION_RULES)}")
    folds = _keys(impl["folds"], {"test_sessions", "formation_sessions"}, set(), "[implementation] folds")
    _int(folds["test_sessions"], "[implementation] folds.test_sessions", 0)
    _int(folds["formation_sessions"], "[implementation] folds.formation_sessions", 1)
    if impl["fill_model"] not in FILL_MODELS:
        raise BaselineError(f"[implementation] fill_model {impl['fill_model']!r} is not one of {FILL_MODELS}")
    if "policy" in impl:
        _keys(impl["policy"], set(DEFAULT_POLICY), set(), "[implementation] policy")
    if "eligibility" in impl:
        _keys(impl["eligibility"], set(), set(DEFAULT_ELIGIBILITY), "[implementation] eligibility")
    if criteria["vol_basis_rule"] != "larger_hurdle_of_target_and_realized":
        raise BaselineError("[criteria] vol_basis_rule: only 'larger_hurdle_of_target_and_realized' is implemented")
    shares = _keys(raw["capacity"]["window_share"], _WINDOW_SHARES, set(), "[capacity] window_share")
    for key in _WINDOW_SHARES:
        if not 0.0 < _float(shares[key], f"[capacity] window_share.{key}") <= 1.0:
            raise BaselineError(f"[capacity] window_share.{key} must lie in (0, 1]")
    if raw["capacity"]["selection_forbidden"] is not True:
        raise BaselineError("[capacity] selection_forbidden must be true: the scenarios may not choose a row")
    if raw["capacity"]["sigma"] != "trailing_20_session_daily":
        raise BaselineError("[capacity] sigma: only 'trailing_20_session_daily' is implemented")
    rows = _parse_rows(raw["rows"], refresh=refresh, base=base,
                       capital=_float(impl["initial_capital"], "[implementation] initial_capital"),
                       fill_model=str(impl["fill_model"]))
    budget = _int(raw["trial_budget"], "trial_budget", 1)
    if len(rows) != budget:
        raise BaselineError(f"[rows] resolves to {len(rows)} rows but trial_budget = {budget}")
    if raw["candidate"] not in {row.id for row in rows}:
        raise BaselineError(f"candidate {raw['candidate']!r} is not a declared row")
    for row_id in raw["capacity"].get("rows", DEFAULT_CAPACITY_ROWS):
        if row_id not in {row.id for row in rows}:
            raise BaselineError(f"[capacity] rows names {row_id!r}, which is not a declared row")
    for entry in raw["capacity"]["synchronization"]:
        _parse_sync(entry, {row.id for row in rows})
    return Baseline(path=path, sha256=sha256, raw=raw, rows=rows, base_offset=base)


def load_baseline(root: Path) -> Baseline:
    path = root / REGISTRATION
    if not path.is_file():
        raise BaselineError(f"no registration at {path}")
    payload = path.read_bytes()
    try:
        raw = tomllib.loads(payload.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise BaselineError(f"{path} is not valid TOML: {exc}") from exc
    return parse_registration(raw, path=path, sha256=hashlib.sha256(payload).hexdigest())


def _parse_sync(entry: Any, ids: set[str]) -> tuple[str, float]:
    match = re.fullmatch(r"([A-Za-z0-9_.-]+):(\d+)(?:/(\d+))?", str(entry))
    if match is None or match.group(1) not in ids:
        raise BaselineError(f"[capacity] synchronization entry {entry!r} is not 'ROW:phi' for a declared row")
    phi = int(match.group(2)) / int(match.group(3) or 1)
    if not 0.0 < phi <= 1.0:
        raise BaselineError(f"[capacity] synchronization entry {entry!r} has phi outside (0, 1]")
    return match.group(1), phi


def require_committed(root: Path, paths: Sequence[Path]) -> str:
    """Refuse unless every path is tracked and unchanged against HEAD in ``root``; return HEAD.

    ``research/scripts/factor_lab.py::require_committed`` with the repository named
    rather than taken from the current directory.
    """
    for path in paths:
        tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", str(path)],
                                 capture_output=True)
        if tracked.returncode != 0:
            raise BaselineError(f"{path} is not committed; the registration must precede the first run (SPEC §4)")
        clean = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", str(path)])
        if clean.returncode != 0:
            raise BaselineError(f"{path} differs from HEAD; an edited registration is a new registration (SPEC §4)")
    state = current_git_state(root)
    if not state.commit:
        raise BaselineError("git HEAD is unavailable; a run must carry its code commit")
    return str(state.commit)


# ===================================================================== public data


Fetcher = Callable[[str], tuple[bytes, Mapping[str, str]]]


def fetch_url(url: str) -> tuple[bytes, dict[str, str]]:
    """Download one file; raise with the URL named on any failure (N7)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read(), dict(response.headers.items())
    except Exception as exc:
        raise BaselineError(f"fetch failed for {url}: {exc}") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def vintage_path(cache_root: Path, sha256: str, url: str) -> Path:
    return cache_root / sha256[:12] / url.rsplit("/", 1)[-1]


def _store_once(target: Path, payload: bytes) -> None:
    """Write ``payload`` to ``target`` unless it exists; an existing file is never replaced."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, target)  # fails if the target exists: no overwrite, even under a race
    except FileExistsError:
        pass
    finally:
        temporary.unlink()


def cached_vintage(url: str, sha256: str, cache_root: Path, *, fetch: Fetcher | None) -> Path:
    """The pinned file, from ``cache_root/<sha256[:12]>/`` or fetched into it.

    A cached file whose bytes do not hash to the pin is refused and left as it is;
    a download whose bytes do not hash to the pin is refused and written nowhere,
    because a new vintage is a new registration. ``fetch=None`` never downloads.
    """
    target = vintage_path(cache_root, sha256, url)
    if target.is_file():
        actual = _sha256(target.read_bytes())
        if actual != sha256:
            raise BaselineError(f"cached {target} hashes to {actual}, pinned {sha256}; the cache is never overwritten")
        return target
    if fetch is None:
        raise BaselineError(f"{url} is not in the vintage cache ({target}) and fetching is off")
    payload, headers = fetch(url)
    actual = _sha256(payload)
    if actual != sha256:
        raise BaselineError(
            f"{url} now serves sha256 {actual} ({len(payload)} bytes), pinned {sha256}: the vintage has changed; "
            "a different vintage is a new registration (nothing was written)"
        )
    _store_once(target, payload)
    provenance = {"url": url, "sha256": actual, "bytes": len(payload), "fetched_at": dt.datetime.now(dt.timezone.utc)
                  .isoformat(), "last_modified": _header(headers, "Last-Modified")}
    _store_once(target.with_name(target.name + ".json"), json.dumps(provenance, indent=2, sort_keys=True).encode())
    return target


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def pin(cache_root: Path, *, fetch: Fetcher) -> dict[str, Any]:
    """Fetch :data:`FRENCH_URLS` into the cache and return what a registration pins. Computes no statistic."""
    out: dict[str, Any] = {}
    for source_id, url in FRENCH_URLS.items():
        payload, headers = fetch(url)
        digest = _sha256(payload)
        target = vintage_path(cache_root, digest, url)
        _store_once(target, payload)
        if _sha256(target.read_bytes()) != digest:
            raise BaselineError(f"{target} exists with other bytes than the file just fetched")
        _, vintage = parse_french(payload)
        out[source_id] = {"url": url, "sha256": digest, "bytes": len(payload),
                          "last_modified": _header(headers, "Last-Modified"), "vintage_line": vintage,
                          "path": str(target)}
    return out


def load_reference(baseline: Baseline, root: Path, *, fetch: Fetcher | None) -> dict[str, Any]:
    """The pinned French files, verified: hash, byte count and vintage line, then parsed."""
    ref = baseline.table("reference")
    cache_root = root / str(ref["cache_dir"])
    out: dict[str, Any] = {}
    for prefix in ("momentum", "cash"):
        path = cached_vintage(str(ref[f"{prefix}_url"]), str(ref[f"{prefix}_sha256"]), cache_root, fetch=fetch)
        payload = path.read_bytes()
        expected_bytes = ref.get(f"{prefix}_bytes")
        if expected_bytes is not None and int(expected_bytes) != len(payload):
            raise BaselineError(f"{path} holds {len(payload)} bytes, pinned {expected_bytes}")
        table, vintage = parse_french(payload)
        expected_line = ref.get(f"{prefix}_vintage_line")
        if expected_line is not None and expected_line != vintage:
            raise BaselineError(f"{path} states vintage {vintage!r}, pinned {expected_line!r}")
        column = str(ref.get(f"{prefix}_column", "Mom" if prefix == "momentum" else "RF"))
        if column not in table.columns:
            raise BaselineError(f"{path} has no column {column!r}: {list(table.columns)}")
        out[prefix] = {"path": str(path), "sha256": _sha256(payload), "bytes": len(payload), "vintage": vintage,
                       "series": table[column].dropna()}
    return out


# ===================================================================== specification, folds, data


def _node_path(node: str) -> str:
    """``prism.signal.momentum_node.MomentumSignalNode`` -> ``prism.signal.momentum_node:MomentumSignalNode``."""
    if ":" in node:
        return node
    module, _, name = node.rpartition(".")
    if not module:
        raise BaselineError(f"[implementation] signal.node {node!r} names no module")
    return f"{module}:{name}"


def row_specification(baseline: Baseline, row: Row) -> dict[str, Any]:
    """The row's specification file (as a table) and how it is evaluated: one function of the TOML and the row."""
    source, impl, guards = baseline.table("source"), baseline.table("implementation"), baseline.table("guards")
    construction = impl["construction"]
    refresh = baseline.refresh
    raw_gross = float(construction["raw_gross"])
    cap = float(construction["max_symbol_abs_weight"])
    ceiling = float(source["max_scale"]) * raw_gross
    to_weights_params: dict[str, Any] = {"decile": float(construction["decile"]), "max_gross": raw_gross,
                                         "max_symbol_abs_weight": cap}
    if row.dollar_volume_window is not None:
        to_weights_params["dollar_volume_window"] = int(row.dollar_volume_window)
    valid_for = refresh if row.tranches == 1 else int(impl.get("tranche_valid_for_sessions", 2 * refresh))
    policy = dict(impl.get("policy", DEFAULT_POLICY))
    eligibility = dict(DEFAULT_ELIGIBILITY, **dict(impl.get("eligibility", {})))
    spec = {
        "strategy": {
            "name": f"{SELECTION_SET}:{row.id}",
            "universe_index": str(impl.get("universe_index", "sp500")),
            "combination": "position_space",
            "renormalize": False,
            "eligibility": eligibility,
            "components": [
                {
                    "id": "momentum",
                    "path": _node_path(str(impl["signal"]["node"])),
                    "weight": 1.0,
                    "horizon_bars": refresh,
                    "update_every_bars": refresh,
                    "valid_for_bars": valid_for,
                    "to_weights": CONSTRUCTION_RULES[construction["rule"]] if row.to_weights == "decile_long_short"
                    else row.to_weights,
                    "tranches": row.tranches,
                    "phase_offset_bars": row.offset,
                    "params": {"lookback_bars": int(impl["signal"]["lookback"]),
                               "skip_bars": int(impl["signal"]["skip"]), "horizon_bars": refresh},
                    "to_weights_params": to_weights_params,
                }
            ],
        },
        "construction": dict(CONTROL_CONSTRUCTION, max_gross=ceiling, max_symbol_abs_weight=cap),
        "sizing": {
            "rule": "constant_vol",
            "target_vol": float(source["target_vol"]),
            "lookback_bars": int(source["lookback_sessions"]),
            "max_scale": float(source["max_scale"]),
            "max_gross": ceiling,
            "annualization_bars": baseline.per_year,
            "max_zero_return_fraction": float(guards["max_zero_return_fraction"]),
        },
        "policy": {
            "profile": policy["profile"],
            "costs": str(impl["cost_stack"]),
            "execution": policy["execution"],
            "monitoring": policy["monitoring"],
            "recovery": policy["recovery"],
        },
    }
    evaluation = {
        "engine": "canonical",
        "decision_coupling": "account_state",
        "fold_portfolio": "carry",
        "cost_refresh": "fold",
        "visibility_mode": "research_vintage",
        "capital": float(row.capital),
        "fill_model": row.fill_model,
        "share_increment": float(impl["share_increment"]),
        "scored_last": str(impl["scored_last"]),
        "partition": None if not row.partitioned else {
            "index_id": spec["strategy"]["universe_index"], "seed": int(row.seed), "keep": float(row.keep_fraction),
            "rule": "HMAC-SHA256(seed, sid) / 2**64 < keep (research/scripts/universe_partition.py)",
        },
    }
    return {"row": row.id, "specification": spec, "evaluation": evaluation}


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BaselineError(f"a specification value {value!r} has no TOML form")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise BaselineError(f"a specification value of type {type(value).__name__} has no TOML form here")


def _render_table(table: Mapping[str, Any], prefix: str, lines: list[str]) -> None:
    scalars = {k: v for k, v in table.items() if not isinstance(v, Mapping)
               and not (isinstance(v, list) and v and all(isinstance(e, Mapping) for e in v))}
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_scalar(value)}")
    for key, value in table.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            lines.extend(["", f"[{name}]"])
            _render_table(value, name, lines)
        elif isinstance(value, list) and value and all(isinstance(e, Mapping) for e in value):
            for element in value:
                lines.extend(["", f"[[{name}]]"])
                _render_table(element, name, lines)


def render_spec_toml(spec: Mapping[str, Any]) -> str:
    """The specification table as TOML; ``tomllib.loads`` of the result is ``spec``."""
    lines = [f"# {SELECTION_SET} row specification, generated by research/scripts/public_baseline.py; do not edit."]
    _render_table(spec, "", lines)
    text = "\n".join(lines).replace("\n\n\n", "\n\n") + "\n"
    if tomllib.loads(text) != json.loads(json.dumps(spec)):
        raise BaselineError("the rendered specification does not round-trip through tomllib")
    return text


def build_fold_table(baseline: Baseline) -> FoldTable:
    """Contiguous test windows from ``p00_anchor`` to ``scored_last``, each after its formation window.

    Built on the session calendar (``prism.io.sessions``), never on the dataset; the
    engine refuses a boundary its dataset does not trade.
    """
    impl = baseline.table("implementation")
    anchor, last = _date(impl["p00_anchor"], "p00_anchor"), _date(impl["scored_last"], "scored_last")
    test = int(impl["folds"]["test_sessions"])
    formation = int(impl["folds"]["formation_sessions"])
    scored = sessions.sessions_between(anchor, last)
    history = sessions.sessions_between(sessions.GRID_EPOCH, anchor)[:-1]
    width = len(scored) if test == 0 else test
    calendar = history + scored
    records = []
    for index, start in enumerate(range(0, len(scored), width)):
        stop = min(start + width, len(scored)) - 1
        at = len(history) + start
        if at - formation < 0:
            raise BaselineError(f"fold {index} needs {formation} formation sessions before {scored[start]}")
        records.append({"index": index, "formation_start": calendar[at - formation].isoformat(),
                        "formation_end": calendar[at - 1].isoformat(), "test_start": scored[start].isoformat(),
                        "test_end": scored[stop].isoformat()})
    # The registration form: bare dates localized to New York, as folds/*.toml are read.
    return FoldTable.from_records(records, table_id=SELECTION_SET, source=str(REGISTRATION),
                                  timezone="America/New_York")


def open_dataset(path: Path, baseline: Baseline) -> Dataset:
    declared = baseline.table("implementation")["dataset"]
    dataset = Dataset.open(Path(path))
    if dataset.manifest.id != declared["manifest_id"]:
        raise BaselineError(f"dataset at {path} has manifest id {dataset.manifest.id}, registered "
                            f"{declared['manifest_id']}")
    if dataset.manifest.vintage != declared["vintage"]:
        raise BaselineError(f"dataset at {path} has vintage {dataset.manifest.vintage!r}, registered "
                            f"{declared['vintage']!r}")
    return dataset


class PartitionedDataset(Dataset):
    """``base`` with one index's point-in-time membership filtered through a seeded partition.

    A name's fate depends only on (seed, sid) (``research/scripts/universe_partition.py``),
    so the filter applies to every membership interval of the name alike. The filter is
    declared as a manifest transformation, which gives the partitioned dataset its own
    ``data_id``. Nothing is written: the bars, tables and manifest on disk are the base's.
    """

    def __init__(self, base: Dataset, *, index_id: str, seed: int, keep: float) -> None:
        note = (f"public_baseline_v1 universe partition: {index_id} membership kept where "
                f"HMAC-SHA256(seed={seed}, sid) / 2**64 < {keep} (research/scripts/universe_partition.py)")
        manifest = dataclasses.replace(base.manifest, transformations=[*base.manifest.transformations, note])
        super().__init__(base.root, manifest, decided_at=base.decided_at)
        self.index_id, self.seed, self.keep = index_id, int(seed), float(keep)
        self._filtered: pd.DataFrame | None = None
        self.kept: tuple[str, ...] = ()
        self.dropped: tuple[str, ...] = ()

    def read_table(self, relative_path: str) -> pd.DataFrame:
        frame = super().read_table(relative_path)
        if relative_path != "membership.parquet":
            return frame
        if self._filtered is None:
            in_index = frame["index_id"].astype(str) == self.index_id
            sids = sorted(set(frame.loc[in_index, "sid"].astype(str)))
            self.kept = universe_partition.partition(sids, self.seed, self.keep)
            self.dropped = tuple(sorted(set(sids) - set(self.kept)))
            if not self.kept:
                raise BaselineError(f"partition seed {self.seed} keeps no member of {self.index_id!r}")
            keep = ~in_index | frame["sid"].astype(str).isin(set(self.kept))
            self._filtered = frame.loc[keep]
        return self._filtered


def canonical_settings(root: Path, spec_table: Mapping[str, Any], evaluation: Mapping[str, Any]) -> CanonicalSettings:
    """The charged policy: the row's cost stack and execution profile, and its fill model.

    ``prism.scripts.construction_experiment.canonical_settings`` with the files named
    by the row instead of a ``prism.registration`` object.
    """
    policy = spec_table["policy"]
    stack = CostStack.from_mapping(tomllib.loads((root / policy["costs"]).read_text(encoding="utf-8")))
    profile = tomllib.loads((root / policy["execution"]).read_text(encoding="utf-8"))
    increment = float(profile.get("quantity_increment_default", 1.0))
    if increment != float(evaluation["share_increment"]):
        raise BaselineError(f"{policy['execution']} trades in increments of {increment}, registered "
                            f"share_increment {evaluation['share_increment']}")
    rounding = RoundingSettings.from_profile(profile)
    return CanonicalSettings(cost_stack=stack, share_increment=increment,
                             min_notional=float(profile.get("min_notional", 0.0)),
                             zero_crossing_clamp=bool(rounding.zero_crossing_clamp),
                             fills=FillPolicy(model=evaluation["fill_model"]))


# ===================================================================== running a row


@contextlib.contextmanager
def observe_simulation() -> Iterator[list[Any]]:
    """Collect the ``SimulationResult`` of every ``prism.evaluate.simulate`` call made inside the block.

    The wrapper passes arguments and result through unchanged; it exists because the
    evaluation record stores no per-name fills, which the flow, wash-sale, capacity and
    N4 reads need exactly.
    """
    import prism.evaluate as engine  # noqa: PLC0415 - the module whose global is wrapped

    original = engine.simulate
    seen: list[Any] = []

    def observed(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        seen.append(result)
        return result

    engine.simulate = observed
    try:
        yield seen
    finally:
        engine.simulate = original


def _rss() -> dict[str, float | None]:
    """Resident and peak memory in MB (``/proc/self/status``; peak from getrusage elsewhere)."""
    current = peak = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 1024.0
    except OSError:
        pass
    if peak is None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {"rss_mb": current, "peak_mb": peak}


def _session_dates(values: Any) -> pd.DatetimeIndex:
    """Engine stamps (aware, New York midnight), their CSV text, or ISO bar labels, as naive session dates.

    An aware index is converted to New York first. Text is cut to its calendar date,
    which is the session: the engine writes New York midnight with that day's offset,
    and parsing a year of such text whole would mix two offsets.
    """
    if isinstance(values, pd.DatetimeIndex):
        stamps = values.tz_convert("America/New_York").tz_localize(None) if values.tz is not None else values
        return stamps.normalize()
    text = pd.Index(values).astype(str)
    if not text.str.fullmatch(r"\d{4}-\d{2}-\d{2}([ T]00:00:00([+-]\d{2}:\d{2})?)?").all():
        bad = text[~text.str.fullmatch(r"\d{4}-\d{2}-\d{2}([ T]00:00:00([+-]\d{2}:\d{2})?)?")][0]
        raise BaselineError(f"{bad!r} is not a session label (a date, or local midnight with its offset)")
    return pd.DatetimeIndex(pd.to_datetime(text.str.slice(0, 10), format="%Y-%m-%d"))


def n4_check(simulation: Any) -> dict[str, Any]:
    """Every session's change in equity, rebuilt from held shares, marks, fills, events and charged costs.

    The identity of ``tests/test_accounting_conservation.py::_reconstruct_equity_path``,
    evaluated per session rather than cumulatively so that float accumulation over a
    long run cannot mask or fake a break.
    """
    ledger = simulation.ledger
    bars = list(ledger.index)
    fills = simulation.fills
    names = shares_columns = list(simulation.shares.columns)
    if not fills.empty:
        names = sorted(set(shares_columns) | set(fills["sid"].astype(str)))
    shares = simulation.shares.reindex(index=bars, columns=names).fillna(0.0)
    marks = simulation.marks.reindex(index=bars, columns=names)
    # A position closed during the session has no close mark. Its mark cancels out of the
    # identity (held * m_t from the carried leg against qty * m_t from the closing fills,
    # qty = -held), so any finite value is exact; 0.0 is used.
    marks = marks.mask((shares == 0.0) & marks.isna(), 0.0)
    held = shares.shift(1).fillna(0.0).to_numpy(dtype=float)
    move = (marks - marks.shift(1)).to_numpy(dtype=float)
    unmarked = (held != 0.0) & ~np.isfinite(move)
    carried = np.where(held != 0.0, held * np.nan_to_num(move), 0.0).sum(axis=1)
    traded = pd.Series(0.0, index=bars)
    if not fills.empty:
        position = {sid: number for number, sid in enumerate(marks.columns)}
        row_of = {bar: number for number, bar in enumerate(bars)}
        values = marks.to_numpy(dtype=float)
        at = np.array([values[row_of[str(b)], position[str(s)]] for b, s in zip(fills["bar"], fills["sid"])])
        gain = fills["qty"].astype(float).to_numpy() * (at - fills["fill_price"].astype(float).to_numpy())
        traded = pd.Series(gain, index=fills["bar"].astype(str)).groupby(level=0).sum().reindex(bars, fill_value=0.0)
    events = simulation.events
    flows = pd.Series(0.0, index=bars)
    if not events.empty:
        by_date = events.groupby(events["effective_date"].astype(str))["equity_delta"].sum()
        flows = by_date.reindex(bars, fill_value=0.0)
    costs = (ledger["trade_cost"] + ledger["holding_cost"]).to_numpy(dtype=float)
    equity = ledger["equity_close"].to_numpy(dtype=float)
    explained = carried + traded.to_numpy(dtype=float) + flows.to_numpy(dtype=float) - costs
    residual = np.diff(equity) - explained[1:]
    tolerance = N4_ABSOLUTE_TOLERANCE + N4_RELATIVE_TOLERANCE * np.abs(equity[:-1])
    worst = int(np.argmax(np.abs(residual) / tolerance)) if residual.size else 0
    return {
        "identity": "E_t - E_t-1 = held_t-1 * (mark_t - mark_t-1) + fills * (mark_t - fill_price) + event equity "
                    "- trade cost - holding cost",
        "n_sessions": int(residual.size),
        "max_abs_residual": float(np.abs(residual).max()) if residual.size else 0.0,
        "worst_bar": bars[worst + 1] if residual.size else None,
        "n_unmarked_holdings": int(unmarked.sum()),
        "relative_tolerance": N4_RELATIVE_TOLERANCE,
        "absolute_tolerance": N4_ABSOLUTE_TOLERANCE,
        "pass": bool(residual.size and (np.abs(residual) <= tolerance).all() and not unmarked.any()),
    }


def _nonzero_columns(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.fillna(0.0)
    return frame.loc[:, (values != 0.0).any(axis=0)]


def write_row_outputs(row_dir: Path, *, result: Any, simulation: Any, max_gross: float) -> dict[str, Any]:
    """The row's per-session series, targets, holdings, fills and sizing decisions, and its mechanics."""
    scored = result.scored
    index = _session_dates(scored.returns.index)
    equity = simulation.ledger["equity_close"].astype(float)
    equity.index = _session_dates(equity.index)
    sizing_block = result.record.evaluation["sizing"]
    sizing = pd.read_csv(sizing_block["sessions_path"], index_col="ts")
    sizing.index = _session_dates(sizing.index)
    costs = scored.costs.copy()
    costs.index = index
    series = pd.DataFrame({"net": scored.returns.to_numpy(dtype=float)}, index=index)
    series["cost"] = costs["total"].to_numpy(dtype=float)
    series["gross_return"] = series["net"] + series["cost"]
    for column in ("commission_spread", "impact", "borrow", "financing", "cash_income", "turnover"):
        series[column] = costs[column].to_numpy(dtype=float)
    series["gross"] = costs["gross"].to_numpy(dtype=float)
    series["net_exposure"] = costs["net"].to_numpy(dtype=float)
    series["equity"] = equity.reindex(index).to_numpy(dtype=float)
    series["equity_prior"] = equity.shift(1).reindex(index).to_numpy(dtype=float)
    for column in ("long_over_equity", "sleeve_fraction", "scale", "target_gross", "ceiling_factor"):
        series[column] = sizing[column].reindex(index).to_numpy(dtype=float)
    series.index.name = "date"
    series.to_csv(row_dir / "series.csv")

    targets = _nonzero_columns(result.full.target_weights.dropna(how="all"))
    targets.index = _session_dates(targets.index)
    targets.to_parquet(row_dir / "targets.parquet", compression="zstd")
    holdings = _nonzero_columns(result.full.fill_weights)
    holdings.index = _session_dates(holdings.index)
    holdings.to_parquet(row_dir / "holdings.parquet", compression="zstd")
    fills = simulation.fills.loc[:, ["bar", "sid", "qty", "fill_price", "reference_price", "execution_id"]].copy()
    fills["leg"] = np.where(fills["execution_id"].astype(str).str.endswith(":close"), "close", "open")
    fills["notional"] = fills["qty"].astype(float) * fills["fill_price"].astype(float)
    fills = fills.drop(columns="execution_id")
    fills.to_parquet(row_dir / "fills.parquet", compression="zstd", index=False)
    decisions = pd.read_csv(sizing_block["decisions_path"])
    decisions.to_csv(row_dir / "sizing_decisions.csv", index=False)

    fallbacks = decisions.loc[decisions["status"] == "fallback"]
    bound = np.minimum(1.0, fallbacks["last_good"].fillna(1.0).to_numpy(dtype=float))
    raised = int((fallbacks["scale"].to_numpy(dtype=float) > bound * (1.0 + GROSS_TOLERANCE)).sum())
    target_gross = series["target_gross"].dropna()
    mechanics = {
        "n4": n4_check(simulation),
        "target_gross_max": float(target_gross.max()) if len(target_gross) else None,
        "target_gross_within_ceiling": bool((target_gross <= max_gross * (1.0 + GROSS_TOLERANCE)).all()),
        "held_gross_max": float(series["gross"].max()),
        "held_gross_sessions_above_ceiling": int((series["gross"] > max_gross * (1.0 + GROSS_TOLERANCE)).sum()),
        "scale_fallbacks": int(len(fallbacks)),
        "scale_fallback_reasons": {str(k): int(v) for k, v in fallbacks["reason"].value_counts().items()},
        "scale_fallbacks_above_bound": raised,
    }
    mechanics["pass"] = bool(mechanics["n4"]["pass"] and mechanics["target_gross_within_ceiling"] and raised == 0)
    return mechanics


def _ledger_rows(log_path: Path) -> list[dict[str, Any]]:
    log = TrialLog(log_path)
    return [row for row in log.verify() if row.get("selection_set_id") == SELECTION_SET] if log.exists() else []


def row_status(rows: Sequence[Mapping[str, Any]], row_id: str) -> str | None:
    """The last terminal event recorded for ``row_id``, ``started`` if it never ended, None if never run.

    ``research/scripts/factor_lab.py::_arm_status``.
    """
    runs = [entry for entry in rows if entry.get("arm_id") == row_id]
    if not runs:
        return None
    terminal = [entry["event"] for entry in runs if entry["event"] != STARTED]
    return terminal[-1] if terminal else STARTED


def _log_path(workspace: Workspace, dataset: Dataset) -> Path:
    channel = "engineering" if dataset.manifest.provenance == "synthetic" else "evidence"
    return resolve_log_path(workspace, selection_set_id=SELECTION_SET, run_channel=channel)


def evaluate_row(
    *,
    root: Path,
    out: Path,
    baseline: Baseline,
    row: Row,
    dataset: Dataset,
    folds: FoldTable,
    kind: str,
    code_commit: str,
) -> dict[str, Any]:
    """Run one row through the canonical engine and write ``rows/<id>/``; return its row record.

    The engine appends the row's ``started`` and terminal ledger rows. A row whose
    specification or dataset cannot be built never reaches the engine, so this
    function appends its ``failed`` row: an attempt that did not resolve is still an
    attempt.
    """
    workspace = Workspace(root)
    row_dir = out / "rows" / row.id
    if row_dir.exists():
        stamp = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        shutil.move(str(row_dir), str(out / "rows" / f"{row.id}.superseded-{stamp}"))
    row_dir.mkdir(parents=True)
    declared = row_specification(baseline, row)
    canonical_json = stable_json_dumps(declared)
    record: dict[str, Any] = {"row": row.id, "kind": kind, "code_commit": code_commit,
                              "specification": declared, "specification_sha256": _sha256(canonical_json.encode()),
                              "fold_table_sha256": folds.content_hash, "started_at": time.time(),
                              "memory_before": _rss()}
    data: Dataset = dataset
    try:
        spec_path = row_dir / "spec.toml"
        spec_path.write_text(render_spec_toml(declared["specification"]), encoding="utf-8")
        if row.partitioned:
            data = PartitionedDataset(dataset, index_id=declared["evaluation"]["partition"]["index_id"],
                                      seed=int(row.seed), keep=float(row.keep_fraction))
        spec = load_spec(spec_path, workspace=workspace)
        settings = canonical_settings(root, declared["specification"], declared["evaluation"])
    except Exception as exc:
        log = TrialLog(_log_path(workspace, dataset))
        synthetic = dataset.manifest.provenance == "synthetic"
        RunRecorder(log=log, run_id=f"run-{uuid.uuid4().hex[:16]}", config_id=None, data_id=data.manifest.id,
                    kind=kind, selection_set_id=SELECTION_SET, arm_id=row.id, capital=row.capital,
                    synthetic=synthetic, run_channel="engineering" if synthetic else "evidence").failed(
            f"specification did not build: {type(exc).__name__}: {exc}")
        raise BaselineError(f"row {row.id}: the specification did not build: {exc}") from exc
    started = time.time()
    with observe_simulation() as seen:
        result = walk_forward_evaluate(
            spec,
            workspace=workspace,
            dataset=data,
            fold_table=folds,
            engine="canonical",
            capital=float(row.capital),
            kind=kind,
            selection_set_id=SELECTION_SET,
            registration_id=baseline.id,
            registration_hash=baseline.sha256,
            arm_id=row.id,
            run_channel="evidence",
            canonical=settings,
            visibility_mode="research_vintage",
            fold_portfolio="carry",
            cost_refresh="fold",
            decision_coupling="account_state",
            start=str(folds.folds[0].formation_start.date()),
            # One session past the last scored bar: the engine fills the last decision there
            # and scores nothing on it (FoldTable.positions reserves it).
            end=sessions.next_session(_date(declared["evaluation"]["scored_last"], "scored_last")).isoformat(),
            untested_conclusions=UNTESTED,
        )
    if len(seen) != 1:
        raise BaselineError(f"row {row.id}: the engine ran {len(seen)} simulations; expected exactly one")
    mechanics = write_row_outputs(row_dir, result=result, simulation=seen[0],
                                  max_gross=float(declared["specification"]["sizing"]["max_gross"]))
    del seen
    record.update(
        run_id=result.run_id,
        config_id=result.config_id,
        data_id=result.data_id,
        record_path=str(result.record_path),
        record_sha256=_sha256(Path(result.record_path).read_bytes()),
        trial_log=str(result.log_path),
        seconds=time.time() - started,
        memory_after=_rss(),
        engine={key: result.record.evaluation["engine"].get(key) for key in
                ("n_decision_bars", "n_fills", "n_orders", "n_events_applied", "max_stale_marks",
                 "pending_orders_at_end", "final_cash")},
        sizing={key: value for key, value in result.record.evaluation["sizing"].items()
                if key not in ("sessions_path", "decisions_path")},
        warnings=list(result.record.warnings),
        mechanics=mechanics,
    )
    if row.partitioned:
        assert isinstance(data, PartitionedDataset)
        record["partition"] = {"kept": len(data.kept), "dropped": len(data.dropped),
                               "kept_sha256": _sha256("\n".join(data.kept).encode())}
        (row_dir / "partition_kept.txt").write_text("\n".join(data.kept) + "\n", encoding="utf-8")
    (row_dir / "row.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=_json_default) + "\n")
    with (out / "memory.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"row": row.id, "run_id": result.run_id, "seconds": record["seconds"],
                                 "before": record["memory_before"], "after": record["memory_after"]}) + "\n")
    return record


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, dt.date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=True) + "\n")


# ===================================================================== reading rows back


def load_series(out: Path, row_id: str) -> pd.DataFrame:
    path = out / "rows" / row_id / "series.csv"
    if not path.is_file():
        raise BaselineError(f"row {row_id} has no recorded series at {path}")
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return frame


def load_row_record(out: Path, row_id: str) -> dict[str, Any]:
    path = out / "rows" / row_id / "row.json"
    if not path.is_file():
        raise BaselineError(f"row {row_id} has no row record at {path}")
    return json.loads(path.read_text())


def _parquet(out: Path, row_id: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(out / "rows" / row_id / f"{name}.parquet")


def scaled_start(out: Path, row_id: str) -> pd.Timestamp:
    """The first session on which every plan of the row holds a scale from an ``ok`` decision.

    A scale decided at close *t* applies from *t + 1*, so this is the session after
    the latest of the plans' first ``ok`` decisions.
    """
    decisions = pd.read_csv(out / "rows" / row_id / "sizing_decisions.csv")
    plans = load_row_record(out, row_id)["sizing"]["plans"]
    firsts = []
    for plan in plans:
        ok = decisions.loc[(decisions["plan_id"] == plan) & (decisions["status"] == "ok"), "bar"]
        if ok.empty:
            raise BaselineError(f"row {row_id}: plan {plan} never had a full volatility window; it never scaled")
        firsts.append(pd.Timestamp(str(ok.iloc[0])[:10]))
    index = load_series(out, row_id).index
    later = index[index > max(firsts)]
    if later.empty:
        raise BaselineError(f"row {row_id}: no session follows its last first-ok decision")
    return later[0]


# ===================================================================== statistics


def _annual(values: pd.Series, per_year: int) -> dict[str, Any]:
    clean = values.dropna()
    stats = annualized_stats(clean, per_year)
    equity = (1.0 + clean).cumprod()
    stats["max_drawdown"] = float((1.0 - equity / equity.cummax()).max()) if len(clean) else None
    stats["annual_return"] = float(equity.iloc[-1] ** (per_year / len(clean)) - 1.0) if len(clean) else None
    return stats


def _boot(baseline: Baseline) -> SharpeBootstrapSpec:
    b = baseline.table("bootstrap")
    return SharpeBootstrapSpec(block_bars=int(b["block_bars"]), n_resamples=int(b["n_resamples"]), seed=int(b["seed"]),
                               confidence=float(b["confidence"]), decision_interval_bars=baseline.refresh)


def paired(a: pd.Series, b: pd.Series, baseline: Baseline, label: str) -> dict[str, Any]:
    """Paired studentized interval of the annualized net-Sharpe difference ``a - b`` on common dates."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    root = math.sqrt(baseline.per_year)
    if joined["a"].equals(joined["b"]):
        # Two rows can trade identically (rounding that never binds, say); the difference is then exactly
        # zero and a studentized interval has no standard error to divide by. Recorded as such, not drawn.
        sharpe = periodic_sharpe(joined["a"].to_numpy())
        return {"label": label, "identical": True, "n_periods": int(len(joined)), "difference": 0.0,
                "lower": 0.0, "upper": 0.0,
                "annualized": {"sharpe_a": sharpe * root, "sharpe_b": sharpe * root, "difference": 0.0,
                               "lower": 0.0, "upper": 0.0},
                "note": "the two series are identical on the window; no interval is drawn"}
    interval = sharpe_difference_interval(joined["a"].to_numpy(), joined["b"].to_numpy(), _boot(baseline),
                                          label=label).to_dict()
    interval["identical"] = False
    interval["annualized"] = {key: interval[key] * root for key in ("sharpe_a", "sharpe_b", "difference", "lower",
                                                                    "upper")}
    return interval


def expected_max_sharpe(trial_sharpes: Sequence[float], n_trials: float) -> float:
    """The false-strategy benchmark at a possibly fractional trial count.

    ``prism.validation.metrics.expected_max_sharpe_with_n`` by value, without its floor
    at the observed count, so that the effective number of independent rows (which is
    below the count of rows) can be the N.
    """
    values = np.asarray([v for v in trial_sharpes if math.isfinite(v)], dtype=float)
    if values.size < 2 or n_trials < 2:
        return float("nan")
    spread = float(values.std(ddof=1))
    if spread < 1e-15:
        return 0.0
    n = float(n_trials)
    z = (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n) + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(spread * z)


def hurdles(*, rf: float, vol_target: float, vol_realized: float, long_over_equity: float, sleeve: float,
            criteria: Mapping[str, Any], per_year: int) -> dict[str, Any]:
    """The governing hurdle (T-bill ETF holding) and the reported cash cases, each at both volatility bases.

    The governing case is ``tbill_hurdle`` with the ETF holding the idle equity above
    the buffer, earning ``rf - sleeve_fee_annual``, and the buffer earning
    ``buffer_cash_yield``. Each case is stated at the target and at the realized
    volatility, and governs at whichever gives the larger Sharpe hurdle.
    """
    fee, buffer_yield = float(criteria["sleeve_fee_annual"]), float(criteria["buffer_cash_yield"])
    rf_source = f"French RF, mean of the last {criteria['hurdle_window_sessions']} sessions of the pinned file"

    def case(**kwargs: Any) -> dict[str, Any]:
        bases = {}
        for basis, vol in (("target", vol_target), ("realized_annualized", vol_realized)):
            bases[basis] = tbill_hurdle(rf, long_over_equity=long_over_equity, vol_annual=vol,
                                        periods_per_year=per_year, rf_source=rf_source, vol_basis=basis,
                                        **kwargs).to_dict()
        governing = max(bases, key=lambda basis: bases[basis]["sharpe_annual"])
        return {"bases": bases, "governing_basis": governing, "sharpe_annual": bases[governing]["sharpe_annual"],
                "return_hurdle_annual": bases[governing]["return_hurdle_annual"],
                "vol_annual": bases[governing]["vol_annual"]}

    out = {"governing": case(cash_yield=buffer_yield, cash_yield_source="[criteria] buffer_cash_yield",
                             sleeve_fraction=sleeve, sleeve_yield=rf - fee,
                             sleeve_yield_source=f"T-bill ETF at rf less [criteria] sleeve_fee_annual ({fee})")}
    for y in criteria["reported_cash_yields"]:
        out[f"cash_yield_{float(y):g}"] = case(cash_yield=float(y),
                                               cash_yield_source="[criteria] reported_cash_yields, no ETF holding")
    return out


def economics_of(series: pd.Series, costs: pd.Series, exposure: pd.DataFrame, *, rho: float, prior: float, rf: float,
                 baseline: Baseline) -> dict[str, Any]:
    """Criterion 3 for one row: ``rho * prior - cost drag`` against every hurdle case."""
    per_year, criteria = baseline.per_year, baseline.table("criteria")
    vol = float(series.std(ddof=1) * math.sqrt(per_year))
    charged = float(costs.mean() * per_year)
    drag = charged / vol
    figure = rho * prior - drag
    long_side = float(exposure["long_over_equity"].mean())
    sleeve = float(np.maximum(0.0, 1.0 - float(criteria["sleeve_buffer"]) - exposure["long_over_equity"]).mean())
    cases = hurdles(rf=rf, vol_target=float(baseline.table("source")["target_vol"]), vol_realized=vol,
                    long_over_equity=long_side, sleeve=sleeve, criteria=criteria, per_year=per_year)
    for case in cases.values():
        case["pass"] = bool(figure > case["sharpe_annual"])
        case["excess_over_bills_annual"] = figure * case["vol_annual"] - case["return_hurdle_annual"]
        case["margin_annual"] = (figure - case["sharpe_annual"]) * case["vol_annual"]
    return {"rho": rho, "prior_sharpe": prior, "rho_times_prior": rho * prior, "charged_cost_annual": charged,
            "realized_vol_annual": vol, "cost_drag_sharpe": drag, "figure_f3": figure,
            "long_over_equity_mean": long_side, "sleeve_fraction_mean": sleeve, "rf_annual": rf, "cases": cases,
            "pass": cases["governing"]["pass"]}


def held_book_ic(decisions: pd.DataFrame, closes: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """Rank IC of target weight against the next ``horizon``-session return, over held names, per decision.

    ``research/scripts/registered_read.py::held_book_ic`` with the horizon a parameter.
    The mean is over every decision bar (overlapping windows when decisions are closer
    than ``horizon``); the standard error is from every ``horizon``-th of them.
    """
    forward = closes.shift(-horizon) / closes - 1.0
    values = {}
    for stamp in decisions.index:
        if stamp not in forward.index:
            continue
        row = decisions.loc[stamp].fillna(0.0)
        names = row.index[row != 0.0]
        value = rank_information_coefficient(row[names].to_numpy(float), forward.loc[stamp, names].to_numpy(float))
        if math.isfinite(value):
            values[stamp] = value
    series = pd.Series(values, dtype=float)
    spaced = series.iloc[::max(1, horizon)] if len(series) else series
    se = float(spaced.std(ddof=1) / math.sqrt(len(spaced))) if len(spaced) > 1 else float("nan")
    return {"horizon": horizon, "mean": float(series.mean()) if len(series) else float("nan"), "n": int(len(series)),
            "se_non_overlapping": se, "n_non_overlapping": int(len(spaced))}


def wash_sale_pairs(targets: pd.DataFrame, first: pd.Timestamp | None = None, last: pd.Timestamp | None = None,
                    days: int = WASH_SALE_DAYS) -> dict[str, Any]:
    """Opposite-direction target changes of one name within ``days`` calendar days, from the decision rows.

    Changes are taken along the whole target path (the first row is the book's
    first build); a pair counts when both of its changes fall in ``[first, last]``.
    """
    full = targets.fillna(0.0)
    change = full.diff()
    change.iloc[0] = full.iloc[0]
    lo = change.index[0] if first is None else first
    hi = change.index[-1] if last is None else last
    change = change.loc[(change.index >= lo) & (change.index <= hi)]
    pairs = sell_then_buy = 0
    for sid in change.columns:
        moves = change[sid]
        moves = moves[moves.abs() > 1e-12]
        signs, stamps = np.sign(moves.to_numpy()), moves.index
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1] and (stamps[i] - stamps[i - 1]).days <= days:
                pairs += 1
                sell_then_buy += int(signs[i - 1] < 0 and signs[i] > 0)
    years = (hi - lo).days / 365.25
    return {"pairs": pairs, "sell_then_buy": sell_then_buy,
            "pairs_per_year": pairs / years if years > 0 else float("nan"), "window_days": days}


def flow_shape(turnover: pd.Series, cycle: int) -> dict[str, Any]:
    """How evenly a flow series spreads over sessions."""
    values = turnover.fillna(0.0).to_numpy(dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        return {"mean": 0.0, "max": 0.0, "peak_to_mean": None, "busiest_cycle_share": None, "hhi_ratio": None}
    top = max(1, int(round(len(values) / cycle)))
    return {"mean": float(values.mean()), "max": float(values.max()),
            "peak_to_mean": float(values.max() / values.mean()),
            "busiest_cycle_share": float(np.sort(values)[::-1][:top].sum() / total),
            "hhi_ratio": float(len(values) * np.sum((values / total) ** 2))}


def same_trade_fraction(fills: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Of one deployment's traded notional, the share another trades in the same name, direction and session."""
    keyed = {}
    for row_id, frame in fills.items():
        size = (frame["notional"] / frame["equity_prior"]).abs()
        keyed[row_id] = size.groupby([frame["bar"].astype(str), frame["sid"].astype(str),
                                      np.sign(frame["qty"].astype(float))]).sum()
    fractions = []
    ids = list(keyed)
    for a in ids:
        for b in ids:
            if a == b:
                continue
            joined = pd.concat([keyed[a].rename("a"), keyed[b].rename("b")], axis=1, join="inner")
            fractions.append(float(np.minimum(joined["a"], joined["b"]).sum() / keyed[a].sum()))
    return {"pairs": len(fractions), "mean": float(np.mean(fractions)) if fractions else None,
            "max": float(np.max(fractions)) if fractions else None}


def _fills_with_equity(out: Path, row_id: str, series: pd.DataFrame) -> pd.DataFrame:
    frame = _parquet(out, row_id, "fills")
    frame["date"] = pd.to_datetime(frame["bar"].astype(str))
    frame["equity_prior"] = series["equity_prior"].reindex(frame["date"]).to_numpy(dtype=float)
    return frame


def _price_panels(dataset: Dataset, sids: Sequence[str], end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    view = dataset.view("research_vintage", end, vintage=dataset.manifest.vintage)
    close = view.panel(list(sids), end, "close")
    volume = view.panel(list(sids), end, "volume")
    close.index = _session_dates(close.index)
    volume.index = _session_dates(volume.index)
    return close, volume.reindex(index=close.index, columns=close.columns)


# ===================================================================== summary


@dataclass
class Context:
    root: Path
    out: Path
    baseline: Baseline
    reference: dict[str, Any]
    dataset: Dataset | None


def _window(series: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    return series.loc[(series.index >= first) & (series.index <= last)]


def parity_gate(ctx: Context) -> dict[str, Any]:
    """P00 against ``vol_momentum_v1``'s recorded scaled in-house series, on that series' own window."""
    impl, criteria = ctx.baseline.table("implementation"), ctx.baseline.table("criteria")
    path = ctx.root / str(impl["parity_series"])
    recorded = pd.read_csv(path, index_col=0, parse_dates=[0])
    if "in_house|net" not in recorded.columns:
        raise BaselineError(f"{path} has no in_house|net column")
    recorded = recorded["in_house|net"].dropna()
    p00 = load_series(ctx.out, ctx.baseline.phase_ids[0])["net"]
    missing = recorded.index.difference(p00.index)
    if len(missing):
        raise BaselineError(f"P00 has no return on {len(missing)} session(s) of the parity window, first {missing[0]}")
    ours = p00.reindex(recorded.index)
    corr = float(np.corrcoef(ours.to_numpy(), recorded.to_numpy())[0, 1])
    root = math.sqrt(ctx.baseline.per_year)
    sharpe_ours, sharpe_recorded = periodic_sharpe(ours.to_numpy()) * root, periodic_sharpe(recorded.to_numpy()) * root
    result = {
        "series": str(impl["parity_series"]),
        "series_sha256": _sha256(path.read_bytes()),
        "first": str(recorded.index[0].date()),
        "last": str(recorded.index[-1].date()),
        "n": int(len(recorded)),
        "daily_corr": corr,
        "sharpe_p00": sharpe_ours,
        "sharpe_recorded": sharpe_recorded,
        "abs_sharpe_diff": abs(sharpe_ours - sharpe_recorded),
        "min_daily_corr": float(criteria["parity_min_daily_corr"]),
        "max_abs_sharpe_diff": float(criteria["parity_max_abs_sharpe_diff"]),
    }
    result["pass"] = bool(corr >= result["min_daily_corr"]
                          and result["abs_sharpe_diff"] <= result["max_abs_sharpe_diff"])
    return result


def reference_block(ctx: Context) -> tuple[pd.DataFrame, dict[str, Any], float]:
    """The scaled public reference, criterion 1, and the hurdle's rf."""
    source, criteria = ctx.baseline.table("source"), ctx.baseline.table("criteria")
    ref = ctx.baseline.table("reference")
    momentum = ctx.reference["momentum"]["series"]
    scaled = scaled_reference(momentum, source, int(ref["warmup_sessions"]))
    bar = publication_bar(scaled["gross"], source, criteria)
    per_year = ctx.baseline.per_year
    cut = str(criteria.get("prior_cut", "2025-12-31"))
    bar["post_publication_cut"] = annualized_stats(scaled["gross"].loc[source["publication"] + "-01":cut], per_year)
    bar["post_publication_cut"]["cut"] = cut
    bar["haircut_full_sample_prior"] = float(criteria["pre_publication_haircut"]) * float(bar["full"]["sharpe"])
    cash = ctx.reference["cash"]["series"]
    rf = float(cash.iloc[-int(criteria["hurdle_window_sessions"]):].mean() * per_year)
    return scaled, bar, rf


def summarize(ctx: Context, *, parity: Mapping[str, Any]) -> dict[str, Any]:
    """Every criterion, case and diagnostic of design sections 5-8, from the recorded rows."""
    baseline, out = ctx.baseline, ctx.out
    per_year, criteria, impl = baseline.per_year, baseline.table("criteria"), baseline.table("implementation")
    candidate = baseline.candidate
    ids = [row.id for row in baseline.rows]
    first = max(scaled_start(out, row_id) for row_id in ids)
    last = pd.Timestamp(str(impl["scored_last"]))
    series = {row_id: _window(load_series(out, row_id), first, last) for row_id in ids}
    lengths = {len(frame) for frame in series.values()}
    if len(lengths) != 1:
        raise BaselineError(f"the rows' series differ in length on the common window: {sorted(lengths)}")
    records = {row_id: load_row_record(out, row_id) for row_id in ids}
    net = pd.DataFrame({row_id: frame["net"] for row_id, frame in series.items()})
    t21 = series[candidate]

    scaled, bar, rf = reference_block(ctx)
    reference = scaled["gross"]
    fid = fidelity(t21["net"], reference, float(criteria["fidelity_floor"]))
    econ = economics_of(t21["net"], t21["cost"], t21, rho=fid["daily_corr"], prior=float(bar["prior_sharpe"]), rf=rf,
                        baseline=baseline)
    contra = no_contradiction(t21, econ["figure_f3"], baseline.table("bootstrap"), per_year)
    mech = records[candidate]["mechanics"]
    mechanics = {"parity": dict(parity), "candidate": mech,
                 "pass": bool(parity.get("pass") and mech["pass"])}
    phases = baseline.phase_ids
    p_bar = net[phases].mean(axis=1)
    tranching = {"t21_minus_p_bar": paired(net[candidate], p_bar, baseline, f"{candidate}_minus_P_bar"),
                 "t21_minus_p00": paired(net[candidate], net[phases[0]], baseline, f"{candidate}_minus_{phases[0]}")}
    bound = float(criteria["tranching_paired_lower_bound"])
    tranching["bound"] = bound
    tranching["pass"] = bool(tranching["t21_minus_p_bar"]["annualized"]["upper"] >= bound)
    criteria_block = {"publication_bar": bar, "tracking": fid, "economics": econ, "no_contradiction": contra,
                      "mechanics": mechanics, "tranching": tranching}
    approved = all(block["pass"] for block in criteria_block.values())

    # deflation and breadth (section 7)
    sharpes = {row_id: periodic_sharpe(net[row_id].to_numpy()) for row_id in ids}
    participation = float(effective_breadth_from_cov(net.corr().to_numpy()))
    # Floor 2 (design section 7); a degenerate correlation matrix (a row with no variance) stays NaN, visibly.
    n_eff_rows = max(2.0, participation) if math.isfinite(participation) else float("nan")
    dsr = {"trial_periodic_sharpes": sharpes, "n_rows": len(ids), "n_effective": n_eff_rows,
           "participation_ratio": participation,
           "dsr_against_rows": deflated_sharpe_ratio_with_n(net[candidate].to_numpy(), list(sharpes.values()),
                                                            len(ids)),
           "dsr_against_effective": probabilistic_sharpe_ratio(
               net[candidate].to_numpy(), expected_max_sharpe(list(sharpes.values()), n_eff_rows)),
           "inherited_not_pooled": {"momentum_v1": 17, "sizing_lab_v1": 8, "vol_momentum_v1": 1}}
    breadth = breadth_block(ctx, candidate, series[candidate], first, last) if ctx.dataset is not None else {
        "skipped": "no dataset was opened"}

    summary = {
        "registration": {"id": baseline.id, "path": str(REGISTRATION), "sha256": baseline.sha256,
                         "selection_set": SELECTION_SET, "trial_budget": len(ids), "candidate": candidate},
        "inputs": {"momentum": {k: v for k, v in ctx.reference["momentum"].items() if k != "series"},
                   "cash": {k: v for k, v in ctx.reference["cash"].items() if k != "series"},
                   "dataset": dict(impl["dataset"]), "cost_stack": str(impl["cost_stack"]),
                   "rows": {row_id: {k: records[row_id].get(k) for k in ("run_id", "config_id", "data_id", "kind",
                                                                         "code_commit", "specification_sha256",
                                                                         "record_sha256", "fold_table_sha256")}
                            for row_id in ids}},
        "conventions": {"in_house": "price return (no cash distributions in the frozen dataset)",
                        "reference": "total return (French daily momentum factor, scaled by the rule, gross)",
                        "note": "the in-house and public series differ in return convention; the dividend wedge is "
                                "not charged"},
        "window": {"first": str(first.date()), "last": str(last.date()), "n": int(len(t21))},
        "criteria": criteria_block,
        "approved": bool(approved),
        "verdict": "approved" if approved else "not_approved",
        "deflation": dsr,
        "breadth": breadth,
        "rows": {row_id: row_statistics(series[row_id], records[row_id], per_year) for row_id in ids},
        "cash_toll": {row_id: cash_tolls(series[row_id], rf, criteria, per_year) for row_id in ids},
        "reference_cap": reference_cap(scaled, baseline),
    }
    summary["phase_dispersion"] = phase_dispersion(series, phases, candidate, p_bar, per_year)
    summary["mechanics_of_tranching"] = tranching_mechanics(ctx, series, phases, candidate)
    summary["rounding"] = rounding_block(ctx, series, first, last)
    summary["diagnostics"] = diagnostics_block(ctx, series, reference, bar, rf)
    summary["flow"] = flow_block(ctx, series, phases, candidate)
    return summary


def row_statistics(series: pd.DataFrame, record: Mapping[str, Any], per_year: int) -> dict[str, Any]:
    stats = _annual(series["net"], per_year)
    stats.update(turnover_annual=float(series["turnover"].mean() * per_year),
                 charged_cost_annual=float(series["cost"].mean() * per_year),
                 gross_mean=float(series["gross"].mean()),
                 long_over_equity_mean=float(series["long_over_equity"].mean()),
                 scale_mean=float(series["scale"].mean()), run_id=record.get("run_id"),
                 mechanics_pass=record["mechanics"]["pass"])
    return stats


def cash_tolls(series: pd.DataFrame, rf: float, criteria: Mapping[str, Any], per_year: int) -> dict[str, Any]:
    vol = float(series["net"].std(ddof=1) * math.sqrt(per_year))
    sharpe = periodic_sharpe(series["net"].to_numpy()) * math.sqrt(per_year)
    gross, long_side = float(series["gross"].mean()), float(series["long_over_equity"].mean())
    yields = {"t_bill_etf": rf - float(criteria["sleeve_fee_annual"]),
              **{f"cash_yield_{float(y):g}": float(y) for y in criteria["reported_cash_yields"]}}
    return {name: cash_toll(sharpe, vol / gross, long_side, y, 0.0).to_dict() for name, y in yields.items()}


def reference_cap(scaled: pd.DataFrame, baseline: Baseline) -> dict[str, Any]:
    cap = float(baseline.table("source")["max_scale"])
    post = scaled.loc[baseline.table("source")["publication"] + "-01":]

    def fractions(frame: pd.DataFrame) -> dict[str, float]:
        return {"at_or_above_0.9_cap": float((frame["scale"] >= 0.9 * cap).mean()),
                "at_cap": float((frame["scale"] >= cap - 1e-12).mean()), "n": int(len(frame))}

    return {"full_sample": fractions(scaled), "post_publication": fractions(post),
            "note": "describes the volatility series; reported, not a criterion"}


def phase_dispersion(series: Mapping[str, pd.DataFrame], phases: Sequence[str], candidate: str, p_bar: pd.Series,
                     per_year: int) -> dict[str, Any]:
    stats = {row_id: _annual(series[row_id]["net"], per_year) for row_id in phases}
    out: dict[str, Any] = {"per_phase": stats}
    for key in ("sharpe", "annual_return", "max_drawdown"):
        values = np.asarray([stats[row_id][key] for row_id in phases], dtype=float)
        out[key] = {"median": float(np.median(values)), "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)), "min": float(values.min()), "max": float(values.max())}
    returns = np.asarray([stats[row_id]["annual_return"] for row_id in phases], dtype=float)
    out["annual_return_sd_pct"] = float(returns.std(ddof=1) * 100.0)
    for label, values in ((candidate, series[candidate]["net"]), ("P_bar", p_bar)):
        own = _annual(values, per_year)
        sharpe_values = np.asarray([stats[row_id]["sharpe"] for row_id in phases], dtype=float)
        out[label] = {"stats": own, "sharpe_percentile_among_phases": float((sharpe_values < own["sharpe"]).mean())}
    out["notes"] = ["A single-phase user carries this dispersion; a tranched user carries almost none.",
                    "The in-house sample is short; the dispersion is estimated noisily."]
    return out


def tranching_mechanics(ctx: Context, series: Mapping[str, pd.DataFrame], phases: Sequence[str],
                        candidate: str) -> dict[str, Any]:
    out, per_year = ctx.out, ctx.baseline.per_year
    first, last = series[candidate].index[0], series[candidate].index[-1]

    def one(row_id: str) -> dict[str, Any]:
        frame = series[row_id]
        fills = _fills_with_equity(out, row_id, load_series(out, row_id))
        fills = fills.loc[(fills["date"] >= first) & (fills["date"] <= last)]
        orders = fills.groupby(["date", "sid"])["notional"].sum().abs()
        targets = _parquet(out, row_id, "targets")
        return {"turnover_annual": float(frame["turnover"].mean() * per_year),
                "cost_drag_sharpe": float(frame["cost"].mean() * per_year /
                                          (frame["net"].std(ddof=1) * math.sqrt(per_year))),
                "orders_per_session": float(len(orders) / len(frame)),
                "median_order_notional": float(orders.median()) if len(orders) else None,
                "wash_sale": wash_sale_pairs(targets, first, last),
                "scale": {"mean": float(frame["scale"].mean()), "min": float(frame["scale"].min()),
                          "max": float(frame["scale"].max()),
                          "at_cap_fraction": float((frame["scale"] >= float(ctx.baseline.table("source")["max_scale"])
                                                    - 1e-12).mean())},
                "long_over_equity_mean": float(frame["long_over_equity"].mean()),
                "sleeve_fraction_mean": float(frame["sleeve_fraction"].mean()),
                "reg_t": reg_t(frame)}

    candidate_block = one(candidate)
    per_phase = {row_id: one(row_id) for row_id in phases}
    mean = {key: float(np.mean([per_phase[p][key] for p in phases if per_phase[p][key] is not None]))
            for key in ("turnover_annual", "cost_drag_sharpe", "orders_per_session", "median_order_notional")}
    mean["wash_sale_pairs_per_year"] = float(np.mean([per_phase[p]["wash_sale"]["pairs_per_year"] for p in phases]))
    decisions = pd.read_csv(out / "rows" / candidate / "sizing_decisions.csv")
    candidate_block["scale_failures"] = {
        "fallbacks": int((decisions["status"] == "fallback").sum()),
        "reasons": {str(k): int(v) for k, v in decisions.loc[decisions["status"] == "fallback", "reason"]
                    .value_counts().items()}}
    return {candidate: candidate_block, "phases_mean": mean, "per_phase": per_phase,
            "notes": ["Reg T rates are 50% initial and 30% maintenance on stock and ETF, Alpaca's documented "
                      "minimums; the ETF's margin treatment is unverified."]}


def reg_t(frame: pd.DataFrame) -> dict[str, Any]:
    """Initial (50%) and maintenance (30%) excess over equity, the ETF holding counted as a long position."""
    exposure = frame["gross"] + frame["sleeve_fraction"]
    initial, maintenance = 1.0 - 0.5 * exposure, 1.0 - 0.3 * exposure
    return {"initial_excess_min": float(initial.min()), "initial_excess_mean": float(initial.mean()),
            "maintenance_excess_min": float(maintenance.min()), "maintenance_excess_mean": float(maintenance.mean())}


def account_size_rows(baseline: Baseline) -> list[Row]:
    """The rows that differ from the candidate only in capital or tranche count (design section 8.3)."""
    lead = baseline.row(baseline.candidate)
    return [row for row in baseline.rows
            if row.id != lead.id and row.group != "phases" and not row.partitioned
            and (row.fill_model, row.to_weights) == (lead.fill_model, lead.to_weights)
            and (row.capital, row.tranches) != (lead.capital, lead.tranches)]


def diagnostic_rows(baseline: Baseline) -> list[Row]:
    """The rows that differ from the candidate in execution, weighting or universe (design sections 8.4-8.6)."""
    lead = baseline.row(baseline.candidate)
    return [row for row in baseline.rows
            if row.id != lead.id and row.group != "phases" and (row.capital, row.tranches) == (lead.capital,
                                                                                               lead.tranches)]


def rounding_block(ctx: Context, series: Mapping[str, pd.DataFrame], first: pd.Timestamp,
                   last: pd.Timestamp) -> dict[str, Any]:
    """Names held against targeted, tracking error against the continuous target, and Sharpe against T21."""
    baseline, out = ctx.baseline, ctx.out
    candidate = baseline.candidate
    rows = [baseline.row(candidate)] + account_size_rows(baseline)
    result: dict[str, Any] = {}
    for row in rows:
        targets = _parquet(out, row.id, "targets")
        holdings = _parquet(out, row.id, "holdings")
        sessions_index = series[row.id].index
        in_effect = targets.reindex(sessions_index.union(targets.index)).ffill().reindex(sessions_index).fillna(0.0)
        held = holdings.reindex(index=sessions_index).fillna(0.0)
        targeted_n = (in_effect != 0.0).sum(axis=1)
        held_n = (held != 0.0).sum(axis=1)
        entry: dict[str, Any] = {"capital": row.capital, "tranches": row.tranches,
                                 "names_targeted_mean": float(targeted_n.mean()),
                                 "names_held_mean": float(held_n.mean()),
                                 "held_over_targeted_mean": float((held_n / targeted_n.where(targeted_n > 0)).mean())}
        if ctx.dataset is not None:
            close, _ = _price_panels(ctx.dataset, list(targets.columns), str(last.date()))
            full_index = load_series(out, row.id).index
            weights = targets.reindex(full_index.union(targets.index)).ffill().reindex(full_index).fillna(0.0).shift(1)
            target_return = base_book_returns(close.reindex(index=full_index, columns=weights.columns), weights)
            gap = (series[row.id]["gross_return"] - target_return.reindex(sessions_index)).dropna()
            entry["tracking_error_vs_target_annual"] = float(gap.std(ddof=1) * math.sqrt(baseline.per_year))
            entry["tracking_error_note"] = ("realized gross return against the return of the decided target held from "
                                            "the session after its fill; the open-to-close timing gap is common to "
                                            "every account size")
        if row.id != candidate:
            entry["net_sharpe_vs_candidate"] = paired(series[row.id]["net"], series[candidate]["net"], baseline,
                                                      f"{row.id}_minus_{candidate}")
        result[row.id] = entry
    return result


def diagnostics_block(ctx: Context, series: Mapping[str, pd.DataFrame], reference: pd.Series, bar: Mapping[str, Any],
                      rf: float) -> dict[str, Any]:
    """Venue (D-split), liquidity weighting (D-liq) and holdings diversity (D-part*) against the candidate."""
    baseline, out, per_year = ctx.baseline, ctx.out, ctx.baseline.per_year
    candidate = baseline.candidate
    floor = float(baseline.table("criteria")["fidelity_floor"])
    base_rho = fidelity(series[candidate]["net"], reference, floor)["daily_corr"]
    prior = float(bar["prior_sharpe"])
    out_block: dict[str, Any] = {}
    for row in diagnostic_rows(baseline):
        frame = series[row.id]
        fid = fidelity(frame["net"], reference, floor)
        entry = {"tracking": fid, "economics": economics_of(frame["net"], frame["cost"], frame, rho=fid["daily_corr"],
                                                           prior=prior, rf=rf, baseline=baseline),
                 "paired_vs_candidate": paired(frame["net"], series[candidate]["net"], baseline,
                                               f"{row.id}_minus_{candidate}"),
                 "tracking_loss": base_rho - fid["daily_corr"],
                 "tracking_loss_cost_in_figure": (base_rho - fid["daily_corr"]) * prior}
        if row.to_weights == "decile_neutral_sqrt_dv" and ctx.dataset is not None:
            entry["thin_name_participation"] = {row_id: thin_participation(ctx, row_id, series[row_id])
                                                for row_id in (candidate, row.id)}
        out_block[row.id] = entry
    parts = [row for row in baseline.rows if row.partitioned]
    if parts:
        kept = {row.id: set((out / "rows" / row.id / "partition_kept.txt").read_text().split()) for row in parts}
        jaccard = {f"{a}~{b}": len(kept[a] & kept[b]) / len(kept[a] | kept[b])
                   for i, a in enumerate(kept) for b in list(kept)[i + 1:]}
        held = {row.id: _parquet(out, row.id, "holdings").reindex(series[row.id].index).fillna(0.0) for row in parts}
        held_jaccard = {}
        for i, a in enumerate(held):
            for b in list(held)[i + 1:]:
                names_a, names_b = held[a] != 0.0, held[b].reindex(columns=held[a].columns, fill_value=0.0) != 0.0
                both = (names_a & names_b).sum(axis=1)
                either = names_a.sum(axis=1) + (held[b] != 0.0).sum(axis=1) - both
                held_jaccard[f"{a}~{b}"] = float((both / either.where(either > 0)).mean())
        out_block["holdings_diversity"] = {"universe_jaccard": jaccard, "held_names_jaccard_mean": held_jaccard,
                                           "aggregate_flow": partition_flow(ctx, [row.id for row in parts], candidate,
                                                                            series)}
    return out_block


def thin_participation(ctx: Context, row_id: str, series: pd.DataFrame) -> dict[str, Any]:
    """Order notional over trailing 20-session median dollar volume, in the least liquid tenth of traded names."""
    fills = _fills_with_equity(ctx.out, row_id, load_series(ctx.out, row_id))
    fills = fills.loc[fills["date"].isin(series.index)]
    orders = fills.groupby(["date", "sid"])["notional"].sum().abs()
    close, volume = _price_panels(ctx.dataset, sorted(set(fills["sid"].astype(str))), str(series.index[-1].date()))
    dollar = close * volume
    median_dv = dollar.where(dollar > 0.0).rolling(LIQUIDITY_WINDOW, min_periods=LIQUIDITY_MIN_VALID).median().shift(1)
    stacked = median_dv.stack()
    stacked.index.names = ["date", "sid"]
    dv = stacked.reindex(orders.index)
    participation = (orders / dv).dropna()
    cut = float(dv.quantile(0.10))
    thin = participation[dv.reindex(participation.index) <= cut]
    return {"dollar_volume_p10": cut, "median_participation": float(participation.median()),
            "thin_median_participation": float(thin.median()) if len(thin) else None,
            "thin_p95_participation": float(thin.quantile(0.95)) if len(thin) else None,
            "n_thin_orders": int(len(thin))}


def partition_flow(ctx: Context, parts: Sequence[str], candidate: str, series: Mapping[str, pd.DataFrame]
                   ) -> dict[str, Any]:
    """Aggregate per-name flow with deployments split evenly over the partitions, against all on the candidate."""

    def per_name(row_id: str) -> pd.Series:
        fills = _fills_with_equity(ctx.out, row_id, load_series(ctx.out, row_id))
        fills = fills.loc[fills["date"].isin(series[row_id].index)]
        return (fills["notional"] / fills["equity_prior"]).abs().groupby([fills["date"], fills["sid"]]).sum()

    whole = per_name(candidate)
    split = pd.concat([per_name(row_id) for row_id in parts], axis=1).fillna(0.0).mean(axis=1)
    return {"notional_ratio": float(split.sum() / whole.sum()),
            "impact_weighted_ratio": float((split ** 1.5).sum() / (whole ** 1.5).sum()),
            "note": "per unit of follower capital: the partitions' mean per-name flow against the candidate's; the "
                    "impact-weighted ratio weights each name-session by flow**1.5, as square-root impact does"}


def flow_block(ctx: Context, series: Mapping[str, pd.DataFrame], phases: Sequence[str], candidate: str
               ) -> dict[str, Any]:
    """Section 8.7: how follower flow lands on sessions under four schemes, and how often users trade together."""
    cycle = ctx.baseline.refresh
    turnover = pd.DataFrame({row_id: series[row_id]["turnover"] for row_id in [*phases, candidate]})
    full = load_series(ctx.out, candidate)
    entry = full["turnover"].iloc[:cycle]
    steady = float(turnover[candidate].mean())
    fills = {row_id: _fills_with_equity(ctx.out, row_id, load_series(ctx.out, row_id)) for row_id in phases}
    fills = {row_id: frame.loc[frame["date"].isin(series[row_id].index)] for row_id, frame in fills.items()}
    return {
        "single_phase_same_day": flow_shape(turnover[phases[0]], cycle),
        "single_phase_uniform_random": flow_shape(turnover[list(phases)].mean(axis=1), cycle),
        "tranched": flow_shape(turnover[candidate], cycle),
        "tranched_burst_staged_entry": {"first_sessions": int(len(entry)), "max": float(entry.max()),
                                        "max_over_steady_mean": float(entry.max() / steady) if steady > 0 else None},
        "same_trade_fraction": {
            "single_phase_same_day": 1.0,
            "tranched_same_grid": 1.0,
            "single_phase_distinct_phases": same_trade_fraction(fills),
            "note": "two deployments on one phase, or two tranched deployments on the one grid at one size, make "
                    "the same trades by construction; distinct phases are measured from the recorded fills",
        },
    }


def breadth_block(ctx: Context, candidate: str, series: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp
                  ) -> dict[str, Any]:
    """N6: N_eff and the IC * sqrt(N_eff) ceiling, and the rank-IC horizon profile, for the candidate."""
    per_year = ctx.baseline.per_year
    targets = _parquet(ctx.out, candidate, "targets")
    targets = targets.loc[(targets.index >= first) & (targets.index <= last)]
    close, _ = _price_panels(ctx.dataset, list(targets.columns), str(last.date()))
    close_full = close.loc[close.index >= first]
    daily = targets.reindex(series.index.union(targets.index)).ffill().reindex(series.index).fillna(0.0)
    breadth = contribution_breadth(daily, close_full.reindex(columns=daily.columns))
    profile = {h: held_book_ic(targets, close_full.reindex(columns=targets.columns), h) for h in IC_HORIZONS}
    horizon = ctx.baseline.refresh
    ic = profile.get(horizon, held_book_ic(targets, close_full.reindex(columns=targets.columns), horizon))["mean"]
    ceiling = abs(ic) * math.sqrt(breadth["n_eff_book_contribution"]) * math.sqrt(per_year / horizon)
    realized = periodic_sharpe(series["net"].to_numpy()) * math.sqrt(per_year)
    return {"n_eff": breadth, "ic_profile": {str(h): v for h, v in profile.items()}, "ic_horizon": horizon,
            "ceiling_annual": ceiling, "realized_net_sharpe_annual": realized,
            "exceeds_ceiling": bool(realized > ceiling),
            "note": "N_eff is the participation ratio of the covariance of position-weighted daily return "
                    "contributions (research/scripts/registered_read.py::contribution_breadth); a realized Sharpe "
                    "above the ceiling motivates a look for a leak and establishes nothing"}


def write_summary(ctx: Context, summary: Mapping[str, Any], scaled: pd.DataFrame | None = None) -> None:
    out = ctx.out
    _write_json(out / "summary.json", summary)
    if "phase_dispersion" in summary:
        _write_json(out / "phase_dispersion.json", summary["phase_dispersion"])
    if "flow" in summary:
        _write_json(out / "flow.json", summary["flow"])
    if scaled is not None:
        t21 = load_series(out, ctx.baseline.candidate)
        frame = pd.concat({"reference": scaled[["scale", "gross"]],
                           "in_house": t21[["net", "cost", "scale", "gross", "long_over_equity", "sleeve_fraction",
                                            "target_gross"]]}, axis=1)
        frame.columns = [f"{a}|{b}" for a, b in frame.columns]
        frame.to_csv(out / "series.csv")


# ===================================================================== capacity (section 9; uncounted)

#: The registered sigma is the trailing 20-session daily volatility; dollar volume is the
#: trailing 20-session median. Both end at the close before the trade.
LIQUIDITY_WINDOW = 20
#: A window counts with at least this many valid sessions, so one missing bar does not
#: leave a traded name unpriced.
LIQUIDITY_MIN_VALID = 10


def trailing_liquidity(close: pd.DataFrame, dollar: pd.DataFrame, trades: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Trailing median dollar volume and daily volatility at each traded cell, from the prior close back.

    A traded cell whose window holds fewer than :data:`LIQUIDITY_MIN_VALID` valid
    sessions takes the name's last earlier value (still causal), and the cells that did
    are counted. A traded cell with no earlier value at all is left for
    ``prism.validation.capacity`` to refuse (N7): there is nothing to price it with.
    """
    window, minimum = LIQUIDITY_WINDOW, LIQUIDITY_MIN_VALID
    median = dollar.where(dollar > 0.0).rolling(window, min_periods=minimum).median().shift(1)
    sigma = close.pct_change(fill_method=None).rolling(window, min_periods=minimum).std(ddof=1).shift(1)
    sigma = sigma.where(sigma > 0.0)
    traded = trades.reindex(index=close.index, columns=close.columns).fillna(0.0) > 0.0
    counts = {}
    out = []
    for name, frame in (("dollar_volume", median), ("sigma", sigma)):
        missing = traded & frame.isna()
        carried = frame.ffill()
        counts[name] = int((missing & carried.notna()).to_numpy().sum())
        out.append(frame.where(~missing, carried).reindex(index=trades.index, columns=trades.columns))
    return out[0], out[1], counts


def capacity(ctx: Context) -> dict[str, Any]:
    """Re-price the priced rows' recorded trades under the registered crowding grid. Selects nothing."""
    if ctx.dataset is None:
        raise BaselineError("the capacity re-pricing needs the dataset's prices and volumes")
    baseline, out = ctx.baseline, ctx.out
    cfg = baseline.table("capacity")
    summary_path = out / "summary.json"
    if not summary_path.is_file():
        raise BaselineError("the capacity re-pricing reads the summary's economics; run --part rows or summary first")
    summary = json.loads(summary_path.read_text())
    if summary.get("verdict") not in ("approved", "not_approved"):
        raise BaselineError(f"the summary's verdict is {summary.get('verdict')!r}; with no verdict read there is no "
                            "economics to re-price")
    first, last = pd.Timestamp(summary["window"]["first"]), pd.Timestamp(summary["window"]["last"])
    stack = tomllib.loads((ctx.root / str(baseline.table("implementation")["cost_stack"])).read_text())
    charged_coeff = float(stack["impact"]["participation_coeff"]) / 1e4
    floor = float(stack["dollar_volume"]["impact"]["floor_dollars"])
    shares = cfg["window_share"]
    sync: dict[str, list[float]] = {}
    for entry in cfg["synchronization"]:
        row_id, phi = _parse_sync(entry, {row.id for row in baseline.rows})
        sync.setdefault(row_id, []).append(phi)
    users_at = [float(v) for v in cfg.get("users_at", [100_000.0, 1_000_000.0])]
    thin = float(cfg.get("thin_dollar_volume", 100_000_000.0))
    _, bar, rf = reference_block(ctx)
    reference = scaled_reference(ctx.reference["momentum"]["series"], baseline.table("source"),
                                 int(baseline.table("reference")["warmup_sessions"]))["gross"]
    result: dict[str, Any] = {"labels": {"assumptions": "every coefficient and window share is an assumption: the "
                                                        "auction shares come from 2015-17 industry notes and are not "
                                                        "checked for 2026; paper fills cannot measure impact",
                                         "selection": "uncounted; may not choose a configuration"},
                              "rows": {}}
    (out / "capacity").mkdir(parents=True, exist_ok=True)
    for row_id in cfg.get("rows", DEFAULT_CAPACITY_ROWS):
        row = baseline.row(row_id)
        full = load_series(out, row_id)
        frame = _window(full, first, last)
        fills = _fills_with_equity(out, row_id, full)
        fills = fills.loc[fills["date"].isin(frame.index)]
        trades = fills.assign(abs_notional=fills["notional"].abs()).pivot_table(
            index="date", columns="sid", values="abs_notional", aggfunc="sum").reindex(frame.index).fillna(0.0)
        close, volume = _price_panels(ctx.dataset, list(trades.columns), str(last.date()))
        dollar = close * volume
        median_dv, sigma, filled = trailing_liquidity(close, dollar, trades)
        same_bar = dollar.reindex(index=trades.index, columns=trades.columns)
        equity = frame["equity_prior"]
        charged_annual = 0.0
        if charged_coeff > 0.0:
            # The stack's own participation term (Y * sigma = participation_coeff / 1e4, against same-bar dollar
            # volume with its floor), recomputed on the recorded trades so the scenarios replace it, not add to it.
            charged = crowd_impact_cost(trades, same_bar, 1.0, equity,
                                        CrowdScenario(impact_y=charged_coeff, windows=((1.0, 1.0),)),
                                        floor_dollars=floor)
            charged_annual = float((charged.sum(axis=1) / equity).mean() * baseline.per_year)
        fid = fidelity(frame["net"], reference, float(baseline.table("criteria")["fidelity_floor"]))
        econ = economics_of(frame["net"], frame["cost"], frame, rho=fid["daily_corr"],
                            prior=float(bar["prior_sharpe"]), rf=rf, baseline=baseline)
        margin = float(econ["cases"]["governing"]["margin_annual"])
        if row.fill_model == "split_open_close":
            windows = {f"open_{k}+close": ((0.5, float(shares[f"open_{k}"])), (0.5, float(shares["close"])))
                       for k in ("low", "high")}
        else:
            windows = {key: ((1.0, float(shares[key])),) for key in ("open_low", "open_high", "close", "full_day")}
        scenarios = [CrowdScenario(follower_capital=float(capital), synchronized_fraction=phi, windows=pairs,
                                   impact_y=float(y), label=f"{row_id}|{name}|phi={phi:g}|Y={float(y):g}")
                     for name, pairs in windows.items() for phi in sync.get(row_id, [1.0])
                     for y in cfg["impact_y"] for capital in cfg["follower_capital"]]
        table = crowd_capacity_table(trades, median_dv, sigma, equity, scenarios, margin_annual=margin,
                                     charged_annual=charged_annual, users_at=users_at, thin_dollar_volume=thin,
                                     periods_per_year=baseline.per_year, floor_dollars=floor)
        table.to_csv(out / "capacity" / f"{row_id}.csv", index=False)
        names_dv = median_dv.where(trades > 0).median(axis=0).dropna()
        result["rows"][row_id] = {
            "margin_annual": margin, "charged_participation_annual_recomputed": charged_annual,
            "statistics": {"window_sessions": LIQUIDITY_WINDOW, "min_valid_sessions": LIQUIDITY_MIN_VALID,
                           "traded_cells": int((trades > 0).to_numpy().sum()), "carried_forward": filled},
            "economics": econ, "n_scenarios": int(len(table)), "table": f"capacity/{row_id}.csv",
            "binding_names": {"n_traded": int(len(names_dv)), "n_below_thin": int((names_dv < thin).sum()),
                              "below_thin": sorted(names_dv.index[names_dv < thin].astype(str)),
                              "median_dollar_volume_p05": float(names_dv.quantile(0.05)),
                              "median_dollar_volume_p10": float(names_dv.quantile(0.10))},
            "own_capacity": table.loc[table["follower_capital"] == table["follower_capital"].min(),
                                      ["windows", "synchronized_fraction", "impact_y", "own_capacity_multiple",
                                       "own_capacity_initial_equity"]].to_dict(orient="records"),
        }
    _write_json(out / "capacity" / "summary.json", result)
    return result


# ===================================================================== orchestration


def _check_write_targets(root: Path, dataset_root: Path | None, baseline: Baseline) -> None:
    """Every write target lies inside the workspace and none inside the dataset."""
    targets = [root / OUT_DIR, root / "trials", root / str(baseline.table("reference")["cache_dir"]),
               root / "results", root / "state", root / "runs", root / "logs"]
    resolved_root = root.resolve()
    for target in targets:
        resolved = target.resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise BaselineError(f"write target {target} resolves outside the workspace {root}")
        if dataset_root is not None:
            data = dataset_root.resolve()
            if resolved == data or data in resolved.parents or resolved in data.parents:
                raise BaselineError(f"write target {target} overlaps the read-only dataset {dataset_root}")


def _dataset_path(args: argparse.Namespace, root: Path, baseline: Baseline) -> Path:
    path = Path(args.dataset) if args.dataset else Path(str(baseline.table("implementation")["dataset"]["path"]))
    return path if path.is_absolute() else root / path


def check(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Validate the registration and its inputs. Evaluates nothing and needs no commit."""
    baseline = load_baseline(root)
    report: dict[str, Any] = {"registration": str(baseline.path), "sha256": baseline.sha256,
                              "rows": [row.id for row in baseline.run_order()], "base_offset": baseline.base_offset}
    path = _dataset_path(args, root, baseline)
    _check_write_targets(root, path, baseline)
    dataset = open_dataset(path, baseline)
    report["dataset"] = {"path": str(path), "manifest_id": dataset.manifest.id, "vintage": dataset.manifest.vintage}
    folds = build_fold_table(baseline)
    report["folds"] = {"n": folds.n_folds, "sha256": folds.content_hash,
                       "first_test": str(folds.folds[0].test_start.date()),
                       "last_test": str(folds.folds[-1].test_end.date())}
    specs = {}
    for row in baseline.rows:
        declared = row_specification(baseline, row)
        with tempfile.TemporaryDirectory() as scratch:
            spec_path = Path(scratch) / "spec.toml"
            spec_path.write_text(render_spec_toml(declared["specification"]))
            load_spec(spec_path, workspace=Workspace(root))
        canonical_settings(root, declared["specification"], declared["evaluation"])
        specs[row.id] = _sha256(stable_json_dumps(declared).encode())
    report["specifications"] = specs
    parity = root / str(baseline.table("implementation")["parity_series"])
    frame = pd.read_csv(parity, index_col=0, nrows=1)
    if "in_house|net" not in frame.columns:
        raise BaselineError(f"{parity} has no in_house|net column")
    report["parity_series_sha256"] = _sha256(parity.read_bytes())
    ref = baseline.table("reference")
    report["reference_cached"] = {prefix: vintage_path(root / str(ref["cache_dir"]), str(ref[f"{prefix}_sha256"]),
                                                       str(ref[f"{prefix}_url"])).is_file()
                                  for prefix in ("momentum", "cash")}
    return report


def committed_inputs(baseline: Baseline) -> list[Path]:
    """Every file the registration names that decides a result: they must be committed unchanged at HEAD.

    The registration and design document, the cost stack, the four policy files and the
    parity series. A local edit to any of them would change an evaluation without
    changing the registration.
    """
    impl = baseline.table("implementation")
    policy = dict(impl.get("policy", DEFAULT_POLICY))
    named = [impl["cost_stack"], *(policy[key] for key in ("profile", "execution", "monitoring", "recovery")),
             impl["parity_series"]]
    return [REGISTRATION, DESIGN_DOC, *(Path(str(value)) for value in named)]


def run_rows(root: Path, args: argparse.Namespace, *, fetch: Fetcher | None) -> int:
    baseline = load_baseline(root)
    code_commit = require_committed(root, committed_inputs(baseline))
    path = _dataset_path(args, root, baseline)
    _check_write_targets(root, path, baseline)
    out = root / OUT_DIR
    reference = load_reference(baseline, root, fetch=fetch)
    dataset = open_dataset(path, baseline)
    folds = build_fold_table(baseline)
    out.mkdir(parents=True, exist_ok=True)
    fold_text = folds.to_toml()
    fold_path = out / "folds.toml"
    if fold_path.exists() and fold_path.read_text() != fold_text:
        raise BaselineError(f"{fold_path} records another fold geometry; it is never rewritten")
    fold_path.write_text(fold_text)
    with (out / "invocations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(), "part": args.part,
                                 "code_commit": code_commit, "registration_sha256": baseline.sha256,
                                 "design_sha256": _sha256((root / DESIGN_DOC).read_bytes()),
                                 "dataset": str(path), "retry": args.retry, "rows": args.rows}) + "\n")
    ctx = Context(root=root, out=out, baseline=baseline, reference=reference, dataset=dataset)
    log_path = _log_path(Workspace(root), dataset)
    ledger = _ledger_rows(log_path)
    retry = set() if not args.retry or args.retry == "failed" else set(args.retry.split(","))
    for row_id in retry:
        baseline.row(row_id)
    wanted = None if not args.rows else set(args.rows.split(","))
    plan: list[tuple[Row, str]] = []
    for row in baseline.run_order():
        status = row_status(ledger, row.id)
        if row.id in retry or (args.retry == "failed" and status in (FAILED, INCOMPLETE, STARTED)):
            if status is None:
                raise BaselineError(f"row {row.id} has no first attempt to retry")
            plan.append((row, "retry"))
        elif status is None:
            plan.append((row, "new"))
        elif status == FINISHED:
            if not (out / "rows" / row.id / "row.json").is_file():
                raise BaselineError(f"row {row.id} finished in {log_path} but {out}/rows/{row.id} is missing; "
                                    f"re-run it with --retry {row.id}")
        else:
            raise BaselineError(f"row {row.id}'s last attempt is {status!r}; re-run it with --retry failed")
    if wanted is not None:
        plan = [(row, kind) for row, kind in plan if row.id in wanted]
    p00 = baseline.phase_ids[0]
    for row, kind in plan:
        if row.id != p00:
            parity_path = out / "parity.json"
            gate = json.loads(parity_path.read_text()) if parity_path.is_file() else None
            if gate is None or not gate.get("pass"):
                raise ParityFailure(f"row {row.id} waits for P00's parity gate, which has "
                                    f"{'not been read' if gate is None else 'failed'}; nothing else runs")
        print(f"[public_baseline] {row.id} ({kind})", flush=True)
        evaluate_row(root=root, out=out, baseline=baseline, row=row, dataset=dataset, folds=folds, kind=kind,
                     code_commit=code_commit)
        if row.id == p00:
            gate = parity_gate(ctx)
            _write_json(out / "parity.json", gate)
            record = load_row_record(out, p00)
            record["mechanics"]["parity"] = gate
            record["mechanics"]["pass"] = bool(record["mechanics"]["pass"] and gate["pass"])
            _write_json(out / "rows" / p00 / "row.json", record)
            if not gate["pass"]:
                _write_json(out / "summary.json", {
                    "registration": {"id": baseline.id, "sha256": baseline.sha256},
                    "verdict": "mechanics_failure", "approved": False, "parity": gate,
                    "note": "P00 failed the parity gate: the run stopped, the row is recorded as a mechanics failure "
                            "and no verdict is read. A re-run after a fix is a retry and spends nothing."})
                raise ParityFailure(f"P00 failed the parity gate (corr {gate['daily_corr']:.4f}, "
                                    f"|dSharpe| {gate['abs_sharpe_diff']:.4f}); the run stops")
    ledger = _ledger_rows(log_path)
    unfinished = [row.id for row in baseline.rows if row_status(ledger, row.id) != FINISHED]
    if unfinished:
        print(f"[public_baseline] {len(unfinished)} row(s) not finished: {unfinished}; no summary yet", flush=True)
        return 0
    write_full_summary(ctx)
    return 0


def write_full_summary(ctx: Context) -> dict[str, Any]:
    parity_path = ctx.out / "parity.json"
    if not parity_path.is_file():
        raise BaselineError("no parity gate has been read; the summary needs P00's")
    parity = json.loads(parity_path.read_text())
    if not parity.get("pass"):
        raise ParityFailure("P00 failed the parity gate; no verdict is read")
    summary = summarize(ctx, parity=parity)
    scaled, _, _ = reference_block(ctx)
    write_summary(ctx, summary, scaled)
    print(json.dumps({"verdict": summary["verdict"],
                      "criteria": {name: block["pass"] for name, block in summary["criteria"].items()}}, indent=2))
    return summary


def _context(root: Path, args: argparse.Namespace, *, fetch: Fetcher | None) -> Context:
    baseline = load_baseline(root)
    require_committed(root, committed_inputs(baseline))
    path = _dataset_path(args, root, baseline)
    _check_write_targets(root, path, baseline)
    return Context(root=root, out=root / OUT_DIR, baseline=baseline,
                   reference=load_reference(baseline, root, fetch=fetch), dataset=open_dataset(path, baseline))


def main(argv: Sequence[str] | None = None, *, fetch: Fetcher | None = fetch_url) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="See the module docstring for parts, outputs and registration keys.")
    parser.add_argument("--part", choices=("pin", "check", "rows", "summary", "capacity", "all"), required=True)
    parser.add_argument("--workspace", default=None, help="repository checkout to run in (default: the current "
                                                          "directory); every output is written inside it")
    parser.add_argument("--dataset", default=None, help="frozen dataset root, opened read-only (default: "
                                                        "[implementation] dataset.path)")
    parser.add_argument("--rows", default=None, help="comma-separated row ids to run this invocation (default: all "
                                                     "not yet finished)")
    parser.add_argument("--retry", default=None, help="'failed' or comma-separated row ids to re-run as retries, "
                                                      "which spend nothing")
    parser.add_argument("--cache-dir", default="data/public/vintages", help="vintage cache for --part pin")
    args = parser.parse_args(argv)
    root = Path(args.workspace or Path.cwd()).resolve()
    try:
        if args.part == "pin":
            if fetch is None:
                raise BaselineError("--part pin needs network access")
            print(json.dumps(pin(root / args.cache_dir, fetch=fetch), indent=2))
            return 0
        if args.part == "check":
            print(json.dumps(check(root, args), indent=2, default=_json_default))
            return 0
        if args.part in ("rows", "all"):
            run_rows(root, args, fetch=fetch)
        if args.part == "summary":
            write_full_summary(_context(root, args, fetch=fetch))
        if args.part in ("capacity", "all"):
            if args.part == "all" and not (root / OUT_DIR / "summary.json").is_file():
                print("[public_baseline] no summary yet; capacity waits for every row", flush=True)
                return 0
            print(json.dumps({row_id: block["margin_annual"] for row_id, block in
                              capacity(_context(root, args, fetch=fetch))["rows"].items()}, indent=2))
        return 0
    except ParityFailure as exc:
        print(f"[public_baseline] stopped: {exc}", file=sys.stderr)
        return 3
    except LabError as exc:
        print(f"[public_baseline] refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
