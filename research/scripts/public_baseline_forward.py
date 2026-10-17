"""Forward stop rule of ``public_baseline_v1`` (``docs/public_baseline_design.md`` section 10).

The registration (``registrations/2026-10-public-baseline.toml``, ``[kill]``) fixes
the rule; this module applies it and changes none of it. It is the reader the
design document says "will be committed before the first read". It refuses any
read before that read's scheduled as-of, and it refuses to run unless the
registration, the design document, the inputs they name and this file are
committed unchanged at HEAD.

Run from the repository root (the workspace)::

    python -m research.scripts.public_baseline_forward status [--asof <date>]
    python -m research.scripts.public_baseline_forward freeze --asof <date>
    python -m research.scripts.public_baseline_forward run    --asof <date>
    python -m research.scripts.public_baseline_forward read   --asof <date> [--paper-run-dir <run dir>]

Run the network and engine steps one at a time, under ``nice``.

The rule
--------
**Forward stream.** T21 (the registration's candidate) re-run with the canonical
engine on data after the registration, by the ``forward_v1`` method
(``docs/forward_design.md``): S&P 500 point-in-time membership rebuilt at the
as-of date (``prism.scripts.build_sp500_universe``) and Twelve Data daily bars
from 2020-01-01 frozen into a dataset by the certified importer
(``prism.io.legacy_import.build_legacy_dataset``). Earlier bars enter only the
formation and volatility-estimate windows.

**Reads.** The first at the first as-of with ``[kill] first_read_min_sessions``
(126) forward sessions, then every ``read_every_sessions`` (63). Each of these
removes eligibility; reinstatement is a new registration:

* (a) the forward net Sharpe's 95% interval (the registration's studentized
  circular block bootstrap) lies entirely below zero;
* (b) the drawdown from the forward peak, the peak starting at the first
  forward session, exceeds ``max_drawdown`` (25%);
* (c) the daily correlation with the scaled reference on the latest French
  vintage is below ``min_forward_fidelity`` (0.5);
* (d) on ``economics_consecutive_reads`` (2) consecutive reads, the forward
  tracking correlation times the post-publication Sharpe on the latest vintage,
  less the forward cost drag, is below the governing hurdle.

Report only: (e) an empirical-Bernstein anytime-valid read (``[kill] monitor``:
bound 0.1, alpha 0.05) of the daily account excess over bills, against zero,
with the expected detection horizon beside it.

**Paper stream.** With ``--paper-run-dir`` the paper loop's run directory is read
too, once it holds the T21 book (21 tranches on the session grid, the
volatility scale, a T-bill holding). Paper accounts simulate neither dividends
nor borrow fees, so each session's NAV return gets the T-bill ETF's
distributions from ``data/distributions/<ETF>.csv`` added and 50 bp a year
charged on the short market value.

Interpretations, stated before any forward session exists
---------------------------------------------------------
* **First forward session.** The first exchange session after the registration's
  ``created`` date. The registration was committed on 2026-10-16 before that
  session opened; starting on the next session (2026-10-19) keeps every
  forward bar strictly after the registration's date.
* **Scheduled as-of.** Read *k* (k = 1, 2, ...) has threshold ``126 + 63 (k - 1)``
  forward sessions and its as-of is the session that many sessions after the
  first forward session: the session whose open fills the last forward
  decision, which the engine reserves and scores nothing on (``forward_read``'s
  geometry). A read is always taken at its scheduled as-of, however late this
  tool runs, and is refused before it. Only reads the bundled session table
  can date are scheduled (three at registration, through 2027-10-20); later
  ones wait for the table to be extended. Reads are taken in order: read *k*
  refuses until read *k - 1* is recorded, because (d) compares consecutive
  reads. A read is written once and never rewritten.
* **Warm-up.** The engine's volatility estimate uses the base book's returns
  from the run's own decisions, so it has none before the run's first test bar
  (``prism.evaluate.SIZING_BASE_BOOK_NOTE``). The run therefore starts the
  fewest whole test windows before the first forward session that cover
  ``lookback_sessions + refresh_sessions`` sessions (three 63-session windows,
  189 sessions, against the 147 the last tranche needs before its first
  full-window scale applies). Those sessions build the tranches and fill the
  estimate window; none of their returns enters a statistic. The fold geometry
  is the registration's (63-session test windows after 312-session formation
  windows), contiguous from the warm-up's first session.
* **Latest vintage.** (c) and (d) read the French momentum and three-factor
  files as served when the read is taken, cached under the read's directory by
  hash and recorded with their hash, byte count, ``Last-Modified`` and CRSP
  vintage line. The reference is ``Mom`` scaled by the registered rule, scored
  from its 1,261st row (``vol_momentum_replication.scaled_reference``). The
  post-publication Sharpe is its annualized Sharpe from 2015-04-01 to the
  file's end. The hurdle's rf is the mean of the last 252 rows of that file's
  ``RF``, annualized, the registration's section 6 rule applied to the latest
  vintage; the pinned file's rate is not reused because it is not refetchable
  once French publishes a newer vintage.
* **Governing hurdle and cost drag** (d) are the registration's section 6 and
  criterion 3 on the forward sessions (``public_baseline.economics_of``): the
  T-bill ETF case at the larger of the target and realized volatility, with
  L/E and the ETF share from the forward run's per-session record.
* **(e).** The account excess on session *t* is T21's net return plus the ETF
  share ``max(0, 1 - buffer - L_t/E_t)`` times ``(rf - fee) / 252``, less
  ``rf / 252``, with the read's rf. Its detection horizon is
  ``monitor.detection_horizon_years`` at the read's own economics figure (to
  reject a zero Sharpe) and at that figure less the governing hurdle (to detect
  the margin over bills); the registration's planning figures, about 29 and
  about 250 years, are quoted beside it.
* **Paper stream.** Positions are rebuilt from the run directory's fills ledger
  and must equal its persisted state's positions; closes come from the run
  directory's own panel record. A distribution is credited on its ex-date to
  the ETF shares held at the previous close; borrow accrues per session on the
  previous close's short market value. (a), (b), (c) and (e) are read on the
  corrected paper returns as on the forward stream; the paper NAV already
  holds the ETF, so (e) adds no ETF term. The paper NAV carries no separable
  cost record, so (d) for the paper stream is the paper tracking correlation
  times the same post-publication Sharpe, less the forward stream's cost drag,
  against the forward stream's governing hurdle. A condition on either stream
  removes eligibility.

Where it writes
---------------
Inside the workspace only: ``results/public-baseline-forward/<as-of>/`` (the
read's fold table, run record, row outputs, French files and ``read.json``),
``trials/public_baseline_v1_forward.jsonl`` (one ``started`` and one terminal
row per forward run, appended by the engine), what
``prism.evaluate.walk_forward_evaluate`` writes for a recorded run, and
``data/frozen/public_baseline_forward_<as-of>/`` (the frozen dataset).
``public_baseline_v1``'s own ledger and results are never written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.evaluate import resolve_log_path, walk_forward_evaluate
from prism.io import sessions
from prism.io.dataset import Dataset
from prism.live import ledgers
from prism.live.loop import read_equity_ledger, read_targets_ledger
from prism.live.monitor import consecutive_session_returns, detection_horizon_years
from prism.live.panel_log import read_index, reconstruct_panel
from prism.live.state import StateStore
from prism.live.tranches import inspect_tranche_state
from prism.live.vol_sizing import SIZING_LEDGER_FILENAME
from prism.spec import load_spec
from prism.validation.anytime import empirical_bernstein_monitor_read
from prism.validation.compare import sharpe_interval
from prism.validation.trials import RunRecorder, TrialLog, stable_json_dumps
from prism.validation.walk_forward import FoldTable
from prism.workspace import Workspace
from research.scripts import public_baseline as pb
from research.scripts.factor_lab import LabError, parse_french
from research.scripts.vol_momentum_replication import (
    fidelity,
    publication_bar,
    scaled_reference,
)

FORWARD_SET = "public_baseline_v1_forward"
READER = Path("research/scripts/public_baseline_forward.py")
READS_ROOT = Path("results/public-baseline-forward")
DATASET_ROOT = Path("data/frozen")
DATASET_PREFIX = "public_baseline_forward_"
#: The forward_v1 method's bar history (``research/scripts/forward_read.py`` HISTORY_START).
HISTORY_START = "2020-01-01"
#: Design section 10: paper accounts charge no borrow, so the paper stream is charged 50 bp a year on short value.
PAPER_BORROW_ANNUAL = 0.0050
#: Design section 1.1: SGOV, with BIL the alternate.
TBILL_ETFS = ("SGOV", "BIL")
DISTRIBUTIONS_DIR = Path("data/distributions")
#: Positions rebuilt from fills must match the persisted state to this many shares.
POSITION_TOLERANCE = 1e-6
#: The registration's planning statement for (e), quoted beside the computed horizons.
PLANNING_STATEMENT = (
    "docs/public_baseline_design.md section 10: about 29 years to reject a zero Sharpe at 0.30 and about 250 "
    "years to detect the expected margin over bills (planning arithmetic, derived from recorded figures, not "
    "measured)"
)
#: What a forward run's record says it does not establish, beside the set's own list.
FORWARD_UNTESTED: tuple[str, ...] = pb.UNTESTED + (
    "Forward run of public_baseline_v1's candidate for a scheduled stop-rule read (design section 10); the "
    "warm-up test windows before the first forward session build the tranches and the volatility window and "
    "enter no forward statistic.",
)


class ReadRefused(pb.BaselineError):
    """A read asked for before its scheduled as-of, out of order, or on inputs that fail a check (N7)."""


# ===================================================================== the clock


@dataclass(frozen=True)
class ScheduledRead:
    """One read of the stop rule: its order, threshold in forward sessions and scheduled as-of."""

    index: int
    name: str
    threshold: int
    asof: dt.date


@dataclass(frozen=True)
class Clock:
    """The schedule a registration implies: first forward session, warm-up and every datable read."""

    registered: dt.date
    first_forward: dt.date
    warmup_sessions: int
    test_sessions: int
    formation_sessions: int
    reads: tuple[ScheduledRead, ...]


def shift_sessions(day: dt.date, n: int) -> dt.date:
    """The session ``n`` sessions after (n > 0) or before (n < 0) session ``day``."""
    if not sessions.is_session(day):
        raise ValueError(f"{day} is not a session")
    out = day
    for _ in range(abs(n)):
        out = sessions.next_session(out) if n > 0 else sessions.previous_session(out)
    return out


def first_forward_session(registered: dt.date) -> dt.date:
    """The first exchange session after the registration's date."""
    return sessions.next_session(registered)


def build_clock(baseline: pb.Baseline) -> Clock:
    """The clock from the registration's ``created`` date, ``[kill]``, ``[source]`` and fold geometry."""
    created = baseline.raw.get("created")
    if not isinstance(created, dt.date) or isinstance(created, dt.datetime):
        raise ReadRefused(f"{baseline.path} declares no created date (got {created!r}); the clock has no start")
    kill, source = baseline.table("kill"), baseline.table("source")
    folds = baseline.table("implementation")["folds"]
    first_read = pb._int(kill["first_read_min_sessions"], "[kill] first_read_min_sessions", 1)
    every = pb._int(kill["read_every_sessions"], "[kill] read_every_sessions", 1)
    test = pb._int(folds["test_sessions"], "[implementation] folds.test_sessions", 1)
    formation = pb._int(folds["formation_sessions"], "[implementation] folds.formation_sessions", 1)
    if first_read % test or every % test:
        raise ReadRefused(f"read thresholds {first_read} + k * {every} are not whole {test}-session test windows")
    need = int(source["lookback_sessions"]) + int(source["refresh_sessions"])
    warmup = math.ceil(need / test) * test
    first = first_forward_session(created)
    reads: list[ScheduledRead] = []
    for k in range(10_000):
        threshold = first_read + k * every
        try:
            asof = shift_sessions(first, threshold)
        except sessions.SessionTableRangeError:
            break  # beyond the bundled session table; extend the table before this read
        reads.append(ScheduledRead(k, f"read_{k + 1}", threshold, asof))
    if not reads:
        raise ReadRefused("the session table dates no read; extend prism.io.sessions before the first read")
    return Clock(registered=created, first_forward=first, warmup_sessions=warmup, test_sessions=test,
                 formation_sessions=formation, reads=tuple(reads))


def due_read(clock: Clock, asof: dt.date) -> ScheduledRead:
    """The latest read whose scheduled as-of is on or before ``asof``; refused before the first."""
    due = [read for read in clock.reads if read.asof <= asof]
    if not due:
        first = clock.reads[0]
        elapsed = len(sessions.sessions_between(clock.first_forward, asof)) if asof >= clock.first_forward else 0
        raise ReadRefused(
            f"as-of {asof} is before the first scheduled read: {first.name} needs {first.threshold} forward "
            f"sessions from {clock.first_forward} and is first possible at as-of {first.asof} "
            f"({elapsed} session(s) dated so far). Nothing was read."
        )
    return due[-1]


def read_dir(root: Path, read: ScheduledRead) -> Path:
    return root / READS_ROOT / str(read.asof)


def dataset_dir(root: Path, read: ScheduledRead) -> Path:
    return root / DATASET_ROOT / f"{DATASET_PREFIX}{read.asof}"


def require_previous(root: Path, clock: Clock, read: ScheduledRead) -> dict[str, Any] | None:
    """The previous read's record, which must exist for every read after the first."""
    if read.index == 0:
        return None
    previous = clock.reads[read.index - 1]
    path = read_dir(root, previous) / "read.json"
    if not path.is_file():
        raise ReadRefused(
            f"{read.name} (as-of {read.asof}) needs {previous.name} (as-of {previous.asof}) recorded first: "
            f"condition (d) compares consecutive reads. Take it with --asof {previous.asof}."
        )
    return json.loads(path.read_text())


def forward_fold_table(clock: Clock, read: ScheduledRead) -> FoldTable:
    """Contiguous test windows from the warm-up's first session to the session before the as-of.

    ``public_baseline.build_fold_table``'s construction on the session calendar,
    anchored at the warm-up instead of ``p00_anchor``.
    """
    first_test = shift_sessions(clock.first_forward, -clock.warmup_sessions)
    scored = sessions.sessions_between(first_test, sessions.previous_session(read.asof))
    if len(scored) != clock.warmup_sessions + read.threshold:
        raise ReadRefused(f"{read.name}: {len(scored)} test sessions, expected warm-up {clock.warmup_sessions} + "
                          f"{read.threshold} forward (N7)")
    history = sessions.sessions_between(sessions.GRID_EPOCH, first_test)[:-1]
    calendar = history + scored
    records = []
    for index, start in enumerate(range(0, len(scored), clock.test_sessions)):
        stop = min(start + clock.test_sessions, len(scored)) - 1
        at = len(history) + start
        if at - clock.formation_sessions < 0:
            raise ReadRefused(f"fold {index} needs {clock.formation_sessions} formation sessions before {scored[start]}")
        records.append({"index": index, "formation_start": calendar[at - clock.formation_sessions].isoformat(),
                        "formation_end": calendar[at - 1].isoformat(), "test_start": scored[start].isoformat(),
                        "test_end": scored[stop].isoformat()})
    return FoldTable.from_records(records, table_id=f"{FORWARD_SET}_{read.name}_{read.asof}",
                                  source=f"{READER} {read.name}", timezone="America/New_York")


# ===================================================================== data steps


def freeze_commands(root: Path, read: ScheduledRead) -> list[list[str]]:
    """The pinned data steps at the scheduled as-of (``forward_read.freeze_commands``' first two)."""
    universe = read_dir(root, read) / "universe"
    py = sys.executable
    return [[py, "-m", "prism.scripts.build_sp500_universe", "--asof", str(read.asof), "--start_date", HISTORY_START,
             "--end_date", str(read.asof), "--out_dir", str(universe)],
            [py, "-m", "research.scripts.public_baseline_forward", "import", "--asof", str(read.asof),
             "--workspace", str(root)]]


def import_dataset(root: Path, read: ScheduledRead) -> dict[str, Any]:
    """Freeze Twelve Data bars from 2020-01-01 and the rebuilt membership (``forward_read.import_dataset``)."""
    from prism.io.legacy_import import build_legacy_dataset

    universe = read_dir(root, read) / "universe"
    resolved = universe / f"sp500_pit_resolved_{read.asof}.txt"
    symbols = [line.split("#", 1)[0].strip() for line in resolved.read_text().splitlines()]
    coverage = json.loads((universe / f"sp500_coverage_{read.asof}.json").read_text())
    report = build_legacy_dataset(dataset_dir(root, read).resolve(), symbols=[s for s in symbols if s],
                                  cache_dir=(root / "data").resolve(), start=HISTORY_START, end=str(read.asof),
                                  vintage=f"{DATASET_PREFIX}{read.asof}", index_id="sp500",
                                  membership_file=universe / f"sp500_membership_{read.asof}.parquet",
                                  universe_note=str(resolved), coverage=coverage)
    return {"root": str(dataset_dir(root, read)), "data_id": report.manifest.id}


# ===================================================================== the forward run


def committed_inputs(baseline: pb.Baseline) -> list[Path]:
    """The set's committed inputs and this reader: the rule and the code that applies it."""
    return [*pb.committed_inputs(baseline), READER]


def _forward_log(workspace: Workspace, dataset: Dataset) -> Path:
    channel = "engineering" if dataset.manifest.provenance == "synthetic" else "evidence"
    return resolve_log_path(workspace, selection_set_id=FORWARD_SET, run_channel=channel)


def run_forward(root: Path, baseline: pb.Baseline, clock: Clock, read: ScheduledRead, *, dataset_path: Path,
                kind: str, code_commit: str) -> dict[str, Any]:
    """Run the candidate on the read's dataset through the canonical engine; write ``rows/<id>/``.

    ``public_baseline.evaluate_row`` with the forward fold table, the forward
    ledger and the read's directory: the specification is the registered row's,
    unchanged.
    """
    out = read_dir(root, read)
    row = baseline.row(baseline.candidate)
    row_dir = out / "rows" / row.id
    if (out / "read.json").exists():
        raise ReadRefused(f"{read.name} is already recorded at {out / 'read.json'}; its run is never replaced")
    if row_dir.exists():
        if kind != "retry":
            raise ReadRefused(f"{read.name}: {row.id} already ran into {row_dir}; a re-run is --retry")
        stamp = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        shutil.move(str(row_dir), str(out / "rows" / f"{row.id}.superseded-{stamp}"))
    elif kind == "retry":
        raise ReadRefused(f"{read.name}: {row.id} has no run to retry in {out}")
    folds = forward_fold_table(clock, read)
    out.mkdir(parents=True, exist_ok=True)
    fold_text = folds.to_toml()
    fold_path = out / "folds.toml"
    if fold_path.exists() and fold_path.read_text() != fold_text:
        raise ReadRefused(f"{fold_path} records another fold geometry; it is never rewritten")
    fold_path.write_text(fold_text)
    row_dir.mkdir(parents=True)
    declared = pb.row_specification(baseline, row)
    declared["evaluation"]["scored_last"] = sessions.previous_session(read.asof).isoformat()
    declared["evaluation"]["forward"] = {"read": read.name, "scheduled_asof": str(read.asof),
                                         "first_forward_session": str(clock.first_forward),
                                         "warmup_sessions": clock.warmup_sessions}
    workspace = Workspace(root)
    dataset = Dataset.open(dataset_path)
    record: dict[str, Any] = {"row": row.id, "kind": kind, "code_commit": code_commit, "read": read.name,
                              "scheduled_asof": str(read.asof), "specification": declared,
                              "specification_sha256": pb._sha256(stable_json_dumps(declared).encode()),
                              "fold_table_sha256": folds.content_hash,
                              "dataset": {"path": str(dataset_path), "manifest_id": dataset.manifest.id,
                                          "vintage": dataset.manifest.vintage,
                                          "provenance": dataset.manifest.provenance},
                              "started_at": time.time(), "memory_before": pb._rss()}
    try:
        spec_path = row_dir / "spec.toml"
        spec_path.write_text(pb.render_spec_toml(declared["specification"]), encoding="utf-8")
        spec = load_spec(spec_path, workspace=workspace)
        settings = pb.canonical_settings(root, declared["specification"], declared["evaluation"])
    except Exception as exc:
        synthetic = dataset.manifest.provenance == "synthetic"
        RunRecorder(log=TrialLog(_forward_log(workspace, dataset)), run_id=f"run-{uuid.uuid4().hex[:16]}",
                    config_id=None, data_id=dataset.manifest.id, kind=kind, selection_set_id=FORWARD_SET,
                    arm_id=row.id, capital=row.capital, synthetic=synthetic,
                    run_channel="engineering" if synthetic else "evidence").failed(
            f"specification did not build: {type(exc).__name__}: {exc}")
        raise ReadRefused(f"{read.name}: the specification did not build: {exc}") from exc
    started = time.time()
    with pb.observe_simulation() as seen:
        result = walk_forward_evaluate(
            spec, workspace=workspace, dataset=dataset, fold_table=folds, engine="canonical",
            capital=float(row.capital), kind=kind, selection_set_id=FORWARD_SET, registration_id=baseline.id,
            registration_hash=baseline.sha256, arm_id=row.id, run_channel="evidence", canonical=settings,
            visibility_mode="research_vintage", fold_portfolio="carry", cost_refresh="fold",
            decision_coupling="account_state", start=str(folds.folds[0].formation_start.date()),
            end=read.asof.isoformat(), untested_conclusions=FORWARD_UNTESTED)
    if len(seen) != 1:
        raise ReadRefused(f"{read.name}: the engine ran {len(seen)} simulations; expected exactly one")
    mechanics = pb.write_row_outputs(row_dir, result=result, simulation=seen[0],
                                     max_gross=float(declared["specification"]["sizing"]["max_gross"]))
    del seen
    record.update(
        run_id=result.run_id, config_id=result.config_id, data_id=result.data_id,
        # Workspace-relative, so a published read names no machine path (docs/security.md rule 8).
        record_path=workspace.relative(result.record_path),
        record_sha256=pb._sha256(Path(result.record_path).read_bytes()),
        trial_log=workspace.relative(result.log_path), seconds=time.time() - started, memory_after=pb._rss(),
        executable_return=result.record.conventions["executable_return"],
        fill_model=result.record.evaluation["engine"]["settings"]["fills"]["model"],
        sizing={key: value for key, value in result.record.evaluation["sizing"].items()
                if key not in ("sessions_path", "decisions_path")},
        warnings=list(result.record.warnings), mechanics=mechanics,
    )
    (row_dir / "row.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=pb._json_default) + "\n")
    return record


# ===================================================================== the latest French vintage


def latest_vintage(url: str, cache_root: Path, *, fetch: pb.Fetcher) -> dict[str, Any]:
    """Fetch ``url`` as served now, keep it under ``cache_root/<sha256[:12]>/`` (never overwritten), and parse it."""
    payload, headers = fetch(url)
    digest = pb._sha256(payload)
    target = pb.vintage_path(cache_root, digest, url)
    pb._store_once(target, payload)
    if pb._sha256(target.read_bytes()) != digest:
        raise ReadRefused(f"{target} exists with other bytes than the file just fetched")
    table, vintage = parse_french(payload)
    return {"url": url, "sha256": digest, "bytes": len(payload), "last_modified": pb._header(headers, "Last-Modified"),
            "vintage_line": vintage, "path": str(target), "first": str(table.index[0].date()),
            "last": str(table.index[-1].date()), "table": table}


def reference_inputs(baseline: pb.Baseline, out: Path, *, fetch: pb.Fetcher) -> dict[str, Any]:
    """The scaled reference, its publication bar and the hurdle's rf, all on the latest vintage."""
    ref, source, criteria = baseline.table("reference"), baseline.table("source"), baseline.table("criteria")
    momentum = latest_vintage(str(ref["momentum_url"]), out / "french", fetch=fetch)
    cash = latest_vintage(str(ref["cash_url"]), out / "french", fetch=fetch)
    columns = {"momentum": str(ref.get("momentum_column", "Mom")), "cash": str(ref.get("cash_column", "RF"))}
    for name, entry in (("momentum", momentum), ("cash", cash)):
        if columns[name] not in entry["table"].columns:
            raise ReadRefused(f"{entry['url']} has no column {columns[name]!r}: {list(entry['table'].columns)}")
    scaled = scaled_reference(momentum["table"][columns["momentum"]].dropna(), source, int(ref["warmup_sessions"]))
    bar = publication_bar(scaled["gross"], source, criteria)
    window = int(criteria["hurdle_window_sessions"])
    rf = float(cash["table"][columns["cash"]].dropna().iloc[-window:].mean() * baseline.per_year)
    files = {name: {k: v for k, v in entry.items() if k != "table"} for name, entry in
             (("momentum", momentum), ("cash", cash))}
    return {"scaled": scaled["gross"], "publication_bar": bar, "rf_annual": rf, "files": files,
            "rf_source": f"French {columns['cash']}, mean of the last {window} rows of the latest file, x "
                         f"{baseline.per_year}"}


# ===================================================================== the conditions


def max_drawdown_from_start(returns: pd.Series) -> float:
    """Largest fall of compounded wealth from its running peak, the peak starting at 1 (the forward start)."""
    wealth = np.cumprod(1.0 + returns.to_numpy(dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float(np.max(1.0 - wealth / peak)) if len(wealth) else float("nan")


def tracking_of(net: pd.Series, reference: pd.Series, baseline: pb.Baseline) -> dict[str, Any]:
    """Daily (and weekly) correlation with the scaled reference over their overlap (criterion 2's statistic)."""
    try:
        return fidelity(net, reference, float(baseline.table("kill")["min_forward_fidelity"]))
    except LabError as exc:
        raise ReadRefused(f"condition (c) cannot be read on the latest French vintage: {exc}") from exc


def stream_conditions(net: pd.Series, *, reference: pd.Series, post_publication_sharpe: float, cost_drag: float,
                      governing_hurdle: float, previous_d: bool | None, excess: pd.Series,
                      baseline: pb.Baseline) -> dict[str, Any]:
    """Conditions (a)-(d) and the report-only (e) for one stream of daily net returns.

    ``previous_d`` is the previous read's (d) for the same stream (None when that
    stream was not read then). ``excess`` is the stream's daily account excess
    over bills, which (e) reads against zero.
    """
    kill = baseline.table("kill")
    per_year, root = baseline.per_year, math.sqrt(baseline.per_year)
    interval = sharpe_interval(net.to_numpy(dtype=float), pb._boot(baseline), label="forward_net").to_dict()
    annual = {key: float(interval[key]) * root for key in ("sharpe_a", "lower", "upper")}
    drawdown = max_drawdown_from_start(net)
    tracking = tracking_of(net, reference, baseline)
    figure = float(tracking["daily_corr"]) * float(post_publication_sharpe) - float(cost_drag)
    d_now = bool(figure < float(governing_hurdle))
    consecutive = int(kill["economics_consecutive_reads"])
    if consecutive != 2:
        raise ReadRefused(f"[kill] economics_consecutive_reads = {consecutive}; this reader implements 2")
    conditions = {
        "a_interval_below_zero": bool(annual["upper"] < 0.0),
        "b_drawdown_above_limit": bool(drawdown > float(kill["max_drawdown"])),
        "c_fidelity_below_floor": bool(float(tracking["daily_corr"]) < float(kill["min_forward_fidelity"])),
        "d_economics_below_hurdle": d_now,
        "d_two_consecutive_reads": bool(d_now and previous_d is True),
    }
    monitor = kill["monitor"]
    if monitor.get("method") != "empirical_bernstein" or monitor.get("report_only") is not True:
        raise ReadRefused(f"[kill] monitor = {dict(monitor)}; this reader implements the report-only "
                          "empirical-Bernstein read")
    alpha = float(monitor["alpha"])
    e_read = empirical_bernstein_monitor_read(excess.to_numpy(dtype=float), hurdle=0.0, alpha=alpha,
                                              bound=float(monitor["bound"]))
    margin = figure - float(governing_hurdle)
    return {
        "n": int(len(net)), "first": str(net.index[0].date()), "last": str(net.index[-1].date()),
        "net_sharpe_annualized": annual["sharpe_a"],
        "a_interval": {"interval": interval, "annualized": annual},
        "b_max_drawdown_from_forward_peak": drawdown,
        "c_tracking": tracking,
        "d_economics": {"rho": float(tracking["daily_corr"]), "post_publication_sharpe": float(post_publication_sharpe),
                        "cost_drag_sharpe": float(cost_drag), "figure": figure,
                        "governing_hurdle_sharpe": float(governing_hurdle), "margin": margin,
                        "previous_read_below": previous_d},
        "conditions": conditions,
        "removes_eligibility": bool(conditions["a_interval_below_zero"] or conditions["b_drawdown_above_limit"]
                                    or conditions["c_fidelity_below_floor"]
                                    or conditions["d_two_consecutive_reads"]),
        "e_report_only": {
            "statistic": "daily account excess over bills, against zero",
            "read": e_read,
            "detection_horizon": {
                "alpha": alpha, "method": "prism.live.monitor.DETECTION_HORIZON_METHOD",
                "years_to_reject_zero_sharpe_at_figure": detection_horizon_years(figure, alpha=alpha)
                if math.isfinite(figure) else None,
                "years_to_detect_margin_over_bills": detection_horizon_years(margin, alpha=alpha)
                if math.isfinite(margin) else None,
                "figure": figure, "margin": margin, "registered_planning_statement": PLANNING_STATEMENT,
            },
        },
    }


def forward_excess(net: pd.Series, long_over_equity: pd.Series, *, rf: float, baseline: pb.Baseline) -> pd.Series:
    """T21's net plus the T-bill ETF share at ``rf - fee``, less ``rf``, per session (design section 10 (e))."""
    criteria, per_year = baseline.table("criteria"), baseline.per_year
    share = np.maximum(0.0, 1.0 - float(criteria["sleeve_buffer"]) - long_over_equity.reindex(net.index))
    if share.isna().any():
        raise ReadRefused("the forward run's record has sessions without a long side over equity (N7)")
    fee = float(criteria["sleeve_fee_annual"])
    return net + share * (rf - fee) / per_year - rf / per_year


# ===================================================================== the paper stream


def load_distributions(root: Path, symbol: str) -> pd.DataFrame:
    """``data/distributions/<symbol>.csv`` (ex_date, pay_date, amount), which must carry its provenance file."""
    path = root / DISTRIBUTIONS_DIR / f"{symbol}.csv"
    provenance = root / DISTRIBUTIONS_DIR / f"{symbol}.provenance.json"
    if not path.is_file() or not provenance.is_file():
        raise ReadRefused(f"{path} and {provenance} must both exist before a paper-stream read (design section 10)")
    frame = pd.read_csv(path, parse_dates=["ex_date"])
    if frame["amount"].isna().any() or (frame["amount"] < 0).any():
        raise ReadRefused(f"{path} has a missing or negative amount (N7)")
    return frame


def positions_by_bar(fills: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Signed shares per symbol after each bar's fills, keyed by the fills' ``filled_bar`` (ISO date)."""
    by_bar: dict[str, list[tuple[str, float]]] = {}
    for fill in fills:
        by_bar.setdefault(str(fill["filled_bar"])[:10], []).append((str(fill["symbol"]), float(fill["qty"])))
    held: dict[str, float] = {}
    out: dict[str, dict[str, float]] = {}
    for bar in sorted(by_bar):
        for symbol, qty in by_bar[bar]:
            held[symbol] = held.get(symbol, 0.0) + qty
        out[bar] = {s: q for s, q in held.items() if abs(q) > POSITION_TOLERANCE}
    return out


def held_at(positions: Mapping[str, Mapping[str, float]], bar: str) -> dict[str, float]:
    """The shares held after the last bar of fills on or before ``bar``."""
    keys = [key for key in positions if key <= bar]
    return dict(positions[max(keys)]) if keys else {}


def correct_paper_returns(series: Mapping[str, Any], equity: Mapping[str, float],
                          positions: Mapping[str, Mapping[str, float]], close: pd.DataFrame,
                          distributions: pd.DataFrame, etf: str, *, borrow_annual: float,
                          per_year: int) -> pd.DataFrame:
    """Each consecutive-session NAV return with the ETF's distributions added and borrow charged.

    ``series`` is :func:`prism.live.monitor.consecutive_session_returns` of the
    equity ledger; ``equity`` the NAV by bar. A distribution whose ex-date maps
    to session *t* (the first session on or after it) is paid on the ETF shares
    held at the previous close; borrow accrues for the session on the previous
    close's short market value. Both are divided by the previous close's NAV.
    """
    closes = close.copy()
    closes.index = pd.DatetimeIndex([pd.Timestamp(str(stamp)[:10]) for stamp in closes.index])
    paid: dict[str, float] = {}
    for ex_date, amount in zip(distributions["ex_date"], distributions["amount"]):
        day = ex_date.date()
        session = day if sessions.is_session(day) else sessions.next_session(day)
        paid[session.isoformat()] = paid.get(session.isoformat(), 0.0) + float(amount)
    rows = []
    for ret, bar, prev in zip(series["returns"], series["bars"], series["prev_bars"]):
        bar, prev = str(bar)[:10], str(prev)[:10]
        held = held_at(positions, prev)
        nav = float(equity[prev])
        shorts = {s: -q for s, q in held.items() if q < 0.0}
        short_value = 0.0
        for symbol, shares in shorts.items():
            price = closes[symbol].get(pd.Timestamp(prev), np.nan) if symbol in closes.columns else np.nan
            if not math.isfinite(float(price)):
                raise ReadRefused(f"no close for the short position {symbol} at {prev}: borrow cannot be charged (N7)")
            short_value += shares * float(price)
        borrow = borrow_annual / per_year * short_value
        distribution = paid.get(bar, 0.0) * held.get(etf, 0.0)
        rows.append({"date": pd.Timestamp(bar), "prev": prev, "raw": float(ret), "distribution": distribution,
                     "borrow": borrow, "short_value": short_value, "nav_prev": nav,
                     "net": float(ret) + (distribution - borrow) / nav})
    return pd.DataFrame(rows).set_index("date")


def load_paper_stream(root: Path, run_dir: Path, baseline: pb.Baseline, *, first: dt.date,
                      last: dt.date) -> dict[str, Any]:
    """The paper stream of a run directory that holds the T21 book, corrected, on ``[first, last]``."""
    row = baseline.row(baseline.candidate)
    state = StateStore(run_dir / "state.json").load()
    report = inspect_tranche_state(state)
    if report is None or report.problems or report.count != row.tranches or report.cycle != baseline.refresh:
        found = None if report is None else {"count": report.count, "cycle": report.cycle,
                                             "problems": list(report.problems)}
        raise ReadRefused(f"{run_dir} does not hold the registered {row.tranches}-tranche book on a "
                          f"{baseline.refresh}-session cycle (found {found}); it is not a T21 paper stream")
    if not (run_dir / SIZING_LEDGER_FILENAME).is_file() or not ledgers.read_rows(run_dir / SIZING_LEDGER_FILENAME):
        raise ReadRefused(f"{run_dir} records no volatility-scale decision ({SIZING_LEDGER_FILENAME}); T21 is scaled")
    holdings = {str((r.get("cash_holding") or {}).get("symbol")) for r in read_targets_ledger(run_dir / "targets.jsonl")
                if r.get("cash_holding")}
    if len(holdings) != 1 or next(iter(holdings)) not in TBILL_ETFS:
        raise ReadRefused(f"{run_dir} holds T-bill ETF(s) {sorted(holdings)}; T21 holds one of {TBILL_ETFS}")
    etf = next(iter(holdings))
    distributions = load_distributions(root, etf)
    fills = ledgers.read_rows(run_dir / "fills.jsonl")
    positions = positions_by_bar(fills)
    final = held_at(positions, "9999-12-31")
    persisted = {s: float(q) for s, q in state.positions.items() if abs(float(q)) > POSITION_TOLERANCE}
    mismatch = {s: (final.get(s, 0.0), persisted.get(s, 0.0)) for s in set(final) | set(persisted)
                if abs(final.get(s, 0.0) - persisted.get(s, 0.0)) > POSITION_TOLERANCE}
    if mismatch:
        raise ReadRefused(f"{run_dir}: positions rebuilt from fills.jsonl differ from state.json on "
                          f"{sorted(mismatch)[:5]}; the corrections need every position's history (N7)")
    frame = read_equity_ledger(run_dir / "equity.jsonl")
    series = consecutive_session_returns(frame)
    keep = [i for i, bar in enumerate(series["bars"]) if first.isoformat() <= str(bar)[:10] <= last.isoformat()]
    if not keep:
        raise ReadRefused(f"{run_dir} has no consecutive-session return in {first}..{last}")
    window = {key: [series[key][i] for i in keep] for key in ("bars", "prev_bars")}
    window["returns"] = series["returns"][keep]
    equity = {str(b)[:10]: float(v) for b, v in zip(frame["decision_bar"], frame["equity"])}
    recorded = [r.decision_bar for r in read_index(run_dir) if r.decision_bar <= last.isoformat()]
    if not recorded:
        raise ReadRefused(f"{run_dir} records no panel on or before {last}; closes for borrow are unavailable")
    close = reconstruct_panel(run_dir, max(recorded))["close"]
    corrected = correct_paper_returns(window, equity, positions, close, distributions, etf,
                                      borrow_annual=PAPER_BORROW_ANNUAL, per_year=baseline.per_year)
    return {"frame": corrected, "etf": etf, "n_fills": len(fills), "n_gaps_in_window": sum(
        1 for skip in series["skipped"] if first.isoformat() <= str(skip["bar"])[:10] <= last.isoformat()),
        "distributions_total": float(corrected["distribution"].sum()),
        "borrow_total": float(corrected["borrow"].sum()), "panel_bar": max(recorded)}


# ===================================================================== the read


def take_read(root: Path, baseline: pb.Baseline, clock: Clock, read: ScheduledRead, *, fetch: pb.Fetcher,
              code_commit: str, paper_run_dir: Path | None = None) -> dict[str, Any]:
    """Every condition of design section 10 at one scheduled read; written once to ``read.json``."""
    out = read_dir(root, read)
    target = out / "read.json"
    if target.exists():
        raise ReadRefused(f"{read.name} is already recorded at {target}; a read is taken once")
    previous = require_previous(root, clock, read)
    candidate = baseline.candidate
    if not (out / "rows" / candidate / "row.json").is_file():
        raise ReadRefused(f"{read.name}: no forward run of {candidate} in {out}; run it first")
    first, last = clock.first_forward, sessions.previous_session(read.asof)
    series = pb.load_series(out, candidate)
    forward = series.loc[(series.index >= pd.Timestamp(first)) & (series.index <= pd.Timestamp(last))]
    if len(forward) != read.threshold:
        raise ReadRefused(f"{read.name}: the run records {len(forward)} forward sessions in {first}..{last}, "
                          f"the read needs {read.threshold} (N7)")
    scaled_from = pb.scaled_start(out, candidate)
    reference = reference_inputs(baseline, out, fetch=fetch)
    bar = reference["publication_bar"]
    post = bar["post_publication"]["sharpe"]
    if post is None or not math.isfinite(float(post)):
        raise ReadRefused("the latest vintage gives no post-publication Sharpe (N7)")
    rf = reference["rf_annual"]

    def previous_d(stream: str) -> bool | None:
        if previous is None or stream not in previous.get("streams", {}):
            return None
        return bool(previous["streams"][stream]["conditions"]["d_economics_below_hurdle"])

    # Criterion 3's arithmetic on the forward sessions: the cost drag and the governing hurdle (design section 6).
    rho = float(tracking_of(forward["net"], reference["scaled"], baseline)["daily_corr"])
    econ = pb.economics_of(forward["net"], forward["cost"], forward, rho=rho, prior=float(post), rf=rf,
                           baseline=baseline)
    drag, hurdle = float(econ["cost_drag_sharpe"]), float(econ["cases"]["governing"]["sharpe_annual"])
    streams: dict[str, Any] = {}
    streams["forward"] = stream_conditions(
        forward["net"], reference=reference["scaled"], post_publication_sharpe=float(post), cost_drag=drag,
        governing_hurdle=hurdle, previous_d=previous_d("forward"),
        excess=forward_excess(forward["net"], forward["long_over_equity"], rf=rf, baseline=baseline),
        baseline=baseline)
    streams["forward"]["d_economics"]["hurdle_cases"] = {
        name: {k: case[k] for k in ("governing_basis", "sharpe_annual", "return_hurdle_annual", "vol_annual")}
        for name, case in econ["cases"].items()}
    streams["forward"]["d_economics"].update(
        charged_cost_annual=econ["charged_cost_annual"], realized_vol_annual=econ["realized_vol_annual"],
        long_over_equity_mean=econ["long_over_equity_mean"], sleeve_fraction_mean=econ["sleeve_fraction_mean"],
        rf_annual=rf, figure_from_economics_of=econ["figure_f3"])
    run_record = pb.load_row_record(out, candidate)
    # Recorded beside the conditions, not one of them: section 10 lists no mechanics condition.
    streams["forward"].update(scaled_from=str(scaled_from.date()),
                              sessions_before_every_tranche_scaled=int((forward.index < scaled_from).sum()),
                              run_id=run_record.get("run_id"), data_id=run_record.get("data_id"),
                              mechanics={"pass": run_record["mechanics"]["pass"],
                                         "n4_pass": run_record["mechanics"]["n4"]["pass"],
                                         "target_gross_within_ceiling":
                                             run_record["mechanics"]["target_gross_within_ceiling"],
                                         "scale_fallbacks": run_record["mechanics"]["scale_fallbacks"]})
    if paper_run_dir is not None:
        paper = load_paper_stream(root, paper_run_dir, baseline, first=first, last=last)
        net = paper["frame"]["net"]
        per_year = baseline.per_year
        streams["paper"] = stream_conditions(
            net, reference=reference["scaled"], post_publication_sharpe=float(post), cost_drag=drag,
            governing_hurdle=hurdle, previous_d=previous_d("paper"), excess=net - rf / per_year, baseline=baseline)
        streams["paper"]["d_economics"]["note"] = (
            "the paper NAV carries no separable cost record: the cost drag and the governing hurdle are the forward "
            "stream's; the tracking correlation is the paper stream's")
        streams["paper"]["corrections"] = {k: v for k, v in paper.items() if k != "frame"}
        streams["paper"]["corrections"].update(borrow_annual=PAPER_BORROW_ANNUAL, run_dir=str(paper_run_dir))
        paper["frame"].to_csv(out / "paper_stream.csv")
    removed_now = any(stream["removes_eligibility"] for stream in streams.values())
    earlier = bool(previous and previous["eligibility"]["removed"])
    first_removed = previous["eligibility"]["first_removed_at"] if earlier else (read.name if removed_now else None)
    summary = {
        "registration": {"id": baseline.id, "path": str(pb.REGISTRATION), "sha256": baseline.sha256,
                         "selection_set": pb.SELECTION_SET, "candidate": candidate, "forward_set": FORWARD_SET},
        "read": {"name": read.name, "index": read.index, "threshold": read.threshold,
                 "scheduled_asof": str(read.asof), "first_forward_session": str(first),
                 "last_forward_session": str(last), "warmup_sessions": clock.warmup_sessions},
        "code_commit": code_commit,
        "taken_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference": {"files": reference["files"], "rf_annual": rf, "rf_source": reference["rf_source"],
                      "publication_bar": bar},
        "conventions": {"forward": "price return (the frozen dataset carries no cash distributions)",
                        "paper": None if paper_run_dir is None else
                        "paper NAV return plus the ETF's distributions, less 50 bp/yr borrow on short value",
                        "reference": "total return (French daily momentum, scaled by the rule, gross)"},
        "streams": streams,
        "eligibility": {"removed": bool(earlier or removed_now), "removed_at_this_read": bool(removed_now),
                        "first_removed_at": first_removed,
                        "by_stream": {name: stream["removes_eligibility"] for name, stream in streams.items()},
                        "note": "removal is final; reinstatement is a new registration (design section 10)"},
    }
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, sort_keys=True, default=pb._json_default, allow_nan=True) + "\n"
    with open(target, "x", encoding="utf-8") as handle:  # a read is written once
        handle.write(text)
    return summary


# ===================================================================== command line


def status(clock: Clock, asof: dt.date) -> dict[str, Any]:
    """The dry run: the schedule and what is due at ``asof``. Reads no data."""
    try:
        due: str | None = due_read(clock, asof).name
        refusal = None
    except ReadRefused as exc:
        due, refusal = None, str(exc)
    return {"registered": str(clock.registered), "first_forward_session": str(clock.first_forward),
            "warmup_sessions": clock.warmup_sessions, "due": due, "refusal": refusal,
            "scheduled": {read.name: {"threshold": read.threshold, "asof": str(read.asof)} for read in clock.reads},
            "unscheduled": f"reads after {clock.reads[-1].name} fall past the bundled session table (ends "
                           f"{sessions.COVERAGE_END}); extend prism.io.sessions before they can be dated"}


def main(argv: Sequence[str] | None = None, *, fetch: pb.Fetcher | None = pb.fetch_url,
         today: dt.date | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="See the module docstring for the rule and its interpretations.")
    parser.add_argument("command", choices=("status", "freeze", "import", "run", "read"))
    parser.add_argument("--asof", type=dt.date.fromisoformat, default=None,
                        help="the date the read is asked for (default today); the latest read scheduled on or "
                             "before it is the one taken, at its own scheduled as-of")
    parser.add_argument("--workspace", default=None, help="repository checkout to run in (default: the current "
                                                          "directory); every output is written inside it")
    parser.add_argument("--dataset", default=None, help="run: the frozen dataset (default "
                                                        "data/frozen/public_baseline_forward_<as-of>)")
    parser.add_argument("--retry", action="store_true", help="run: re-run a read's forward run (spends nothing)")
    parser.add_argument("--paper-run-dir", default=None, help="read: also read this paper-loop run directory, "
                                                              "which must hold the T21 book")
    args = parser.parse_args(argv)
    root = Path(args.workspace or Path.cwd()).resolve()
    today = today or dt.date.today()
    asof = args.asof or today
    try:
        baseline = pb.load_baseline(root)
        clock = build_clock(baseline)
        if args.command == "status":
            print(json.dumps(status(clock, asof), indent=2))
            return 0
        if asof > today:
            raise ReadRefused(f"as-of {asof} is after today ({today}); no data exists for it yet")
        read = due_read(clock, asof)
        if args.command == "freeze":
            for step in freeze_commands(root, read):
                print("+", " ".join(step), flush=True)
                subprocess.run(step, check=True, cwd=root)
            return 0
        if args.command == "import":
            print(json.dumps(import_dataset(root, read), indent=2))
            return 0
        code_commit = pb.require_committed(root, committed_inputs(baseline))
        if args.command == "run":
            if read.index > 0:
                require_previous(root, clock, read)
            dataset = Path(args.dataset) if args.dataset else dataset_dir(root, read)
            record = run_forward(root, baseline, clock, read, dataset_path=dataset,
                                 kind="retry" if args.retry else "new", code_commit=code_commit)
            print(json.dumps({"read": read.name, "run_id": record["run_id"], "mechanics_pass":
                              record["mechanics"]["pass"]}, indent=2))
            return 0
        if fetch is None:
            raise ReadRefused("the read fetches the latest French files and needs network access")
        summary = take_read(root, baseline, clock, read, fetch=fetch, code_commit=code_commit,
                            paper_run_dir=Path(args.paper_run_dir) if args.paper_run_dir else None)
        print(json.dumps({"read": read.name, "scheduled_asof": str(read.asof),
                          "eligibility": summary["eligibility"],
                          "conditions": {name: stream["conditions"] for name, stream in summary["streams"].items()}},
                         indent=2))
        return 0
    except LabError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
