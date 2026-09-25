"""Forward reads of every pending clock, as their registrations fix them.

One tool for the five forward obligations registered before any forward session
existed. It implements them; it does not redesign them:

* ``forward_v1`` (``docs/forward_design.md`` §2-4): F1, F2 against FB; interim at
  >= 126 forward sessions kills on the paired interval, promotion at >= 252;
  DSR against nine rows.
* ``earnings_daily_v1`` (``docs/earnings_design.md`` §6 and the 2026-09-25
  addendum): the two daily arms; DSR against three rows.
* ``buffer_monthly_v1`` / ``buffer_daily_v1`` (``docs/buffer_design.md`` §5): the
  four buffered arms; DSR against eight rows; N6 viability is a criterion.
* ``vol_momentum_v1``'s kill rule (``registrations/2026-09-vol-momentum.toml``,
  ``docs/vol_momentum_design.md`` §4): forward_v1's FB stream scaled by the
  Barroso-Santa-Clara rule; first read at >= 126 forward sessions, then every 63.

**Geometry.** Rolling folds of the certified shape (312 formation, 63 test, step
63), cut by the engine's own splitter (``RollingWalkForward``), the first test
window opening on the clock's first forward session. Every read threshold is a
multiple of 63, so each read has a *scheduled as-of*: the session whose open
closes the threshold's last forward return (the splitter reserves it for the
fold-end flatten). A read is always taken at its scheduled as-of, however late
the tool runs, and is refused before it. The fold table is written by this rule
and content-hashed (``FoldTable.content_hash`` of the table re-read from its
TOML, the form every registration uses).

**Data.** At each scheduled as-of: S&P 500 membership rebuilt by
``prism.scripts.build_sp500_universe`` into the read's own directory (never
``data/universe``, whose ``sp500_current.txt`` the live loop trades), Twelve Data
daily bars from 2020-01-01 frozen by ``build_legacy_dataset`` (the certified
dataset's importer), and for the earnings and buffer clocks the announcement
panel rebuilt by ``prism.scripts.build_earnings_panel`` after the EDGAR store is
refreshed. Bars before the first forward session enter only formation windows.

**Runs.** Each clock's source registration is re-emitted with only its id,
creation date, selection set, trial budget, sample, fold table and freeze record
changed (and, for books that pin an announcement panel, the control book with
only its panel path and hash changed). The derived files are written beside the
read and hashed; the arms then run through ``construction_experiment`` exactly as
in-house. The vol-momentum warm-up (FB over the 189 sessions before the first
forward session, three whole test windows, the fewest that let the rule's
126-session estimate scale the first forward session) enters only the
volatility estimate, as the registration says.

**Statistics.** Per arm on forward sessions only: net and gross Sharpe, the
studentized circular-block interval with each registration's bootstrap, the
paired interval against FB for forward_v1, DSR against the registered rows (the
in-house rows' recorded periodic Sharpes plus this set's forward Sharpes where
the registration counts them), N_eff (``registered_read.contribution_breadth``),
the held-book rank IC at the component's declared horizon (open-to-open from the
next open, N2; Newey-West lag ``ceil(h / decision interval) - 1``), N6 viability
(ceiling ``IC x sqrt(N_eff x 252 / h)`` at the one-sided lower-95 IC against cost
drag plus the hurdle, as the in-house reads computed it), and the SPEC §3 hurdle
``(rf - y * max(0, 1 - L/E)) / vol`` with rf the mean bond-equivalent yield of
FRED DTB3 over the forward sessions, y the cash yield of ``retail_v2`` and L/E
the book's recorded average long exposure. The live monitor contradicts when the
paper stream's anytime verdict is ``refuted``.

Subcommands (all refuse before the clock's first read; ``status`` is the dry run)::

    python -m research.scripts.forward_read status --asof 2026-09-25
    python -m research.scripts.forward_read plan   --clock forward_v1 --asof <date>
    python -m research.scripts.forward_read freeze --clock forward_v1 --asof <date>
    python -m research.scripts.forward_read run    --clock forward_v1 --asof <date>
    python -m research.scripts.forward_read read   --clock forward_v1 --asof <date>

Run the network and engine steps one at a time under ``nice -n 19 taskset -c 6-11``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.io import sessions
from prism.validation.compare import (
    SharpeBootstrapSpec,
    sharpe_difference_interval,
    sharpe_interval,
)
from prism.validation.metrics import (
    deflated_sharpe_ratio_with_n,
    probabilistic_sharpe_ratio,
)
from prism.validation.walk_forward import FoldTable, RollingWalkForward

FORMATION, TEST, STEP = 312, 63, 63
PER_YEAR = 252
ONE_SIDED_95 = 1.6448536269514722
TZ = "America/New_York"
#: Three 63-session windows: the first scale decision with a full 126-session
#: lookback falls at warm-up index 126, so forward session 0 (index 189) is scaled.
VOL_WARMUP_FOLDS = 3
HISTORY_START = "2020-01-01"
COST_STACK_CASH = Path("examples/reference/costs/retail_v2.toml")
READS_ROOT = Path("results/forward")
LIVE_RUN_DIR_POINTER = Path("runs/ACTIVE_RUN_DIR")
REGISTERED_PANEL = "data/frozen/earnings_v1/panel.parquet"
REGISTERED_PANEL_SHA256 = "d4703a6cc1374e862d398df1a5bc4069e4e04f0ec1fc1fe9c81614a0fc3d701e"


class ReadRefused(RuntimeError):
    """A read asked for before its scheduled as-of, or on inputs that fail a registered check (N7)."""


# --------------------------------------------------------------------- the clocks


@dataclass(frozen=True)
class ArmSpec:
    """One arm read forward: its id in the source registration, decision interval and declared horizon."""

    arm: str
    interval: int
    horizon: int


@dataclass(frozen=True)
class Clock:
    name: str
    route: str
    registration: Path
    design: Path
    arms: tuple[ArmSpec, ...]
    forward_start: dt.date
    bootstrap: Mapping[str, Any]
    kill: str  # "paired_vs_benchmark" | "own_interval" | "vol_scaled"
    thresholds: tuple[tuple[str, int], ...]
    deflation_rows: tuple[tuple[str, str], ...] = ()  # (read.json path or "forward", arm)
    n_deflation: int = 0
    benchmark_arm: str | None = None
    book: Path | None = None  # a control book that pins the announcement panel
    n6_criterion: bool = True


def _vol_thresholds(first: int = 126, every: int = 63, horizon: int = 2 * PER_YEAR) -> tuple[tuple[str, int], ...]:
    """Kill reads at 126, 189, 252, ... forward sessions; ``scheduled_reads`` drops those the session table cannot date."""
    return tuple((f"kill_{k + 1}", first + k * every) for k in range((horizon - first) // every + 1))


CHARACTERISTICS_READ = "results/2026-09-characteristics/read.json"
EARNINGS_READ = "results/2026-09-earnings/read.json"
EARNINGS_DAILY_READ = "results/2026-09-earnings-daily/read.json"
BUFFER_MONTHLY_READ = "results/2026-09-buffer-monthly/read.json"
BUFFER_DAILY_READ = "results/2026-09-buffer-daily/read.json"
EIGHT_BUFFER_ROWS = (
    (BUFFER_MONTHLY_READ, "momentum_buffered"), (BUFFER_MONTHLY_READ, "ear_monthly_buffered"),
    (BUFFER_DAILY_READ, "ear_event_buffered"), (BUFFER_DAILY_READ, "premium_buffered"),
    (CHARACTERISTICS_READ, "momentum_control"), (EARNINGS_READ, "ear_monthly"),
    (EARNINGS_DAILY_READ, "ear_event"), (EARNINGS_DAILY_READ, "announcement_premium"),
)

CLOCKS: dict[str, Clock] = {
    "forward_v1": Clock(
        name="forward_v1", route="discovery",
        registration=Path("registrations/2026-09-characteristics.toml"), design=Path("docs/forward_design.md"),
        arms=(ArmSpec("residual_momentum_only", 21, 21), ArmSpec("momentum_residual_momentum", 21, 21),
              ArmSpec("momentum_control", 21, 21)),
        forward_start=dt.date(2026, 9, 24),
        # The arms' own registration's [sharpe_bootstrap] (paired, as the kill reads it).
        bootstrap={"block_bars": 42, "n_resamples": 2000, "seed": 20260916, "confidence": 0.95},
        kill="paired_vs_benchmark", benchmark_arm="momentum_control",
        thresholds=(("interim", 126), ("promotion", 252)),
        deflation_rows=tuple((CHARACTERISTICS_READ, arm) for arm in (
            "momentum_control", "residual_momentum_only", "idiosyncratic_vol_only", "momentum_residual_momentum",
            "momentum_idiosyncratic_vol", "composite_three")) + tuple(
            ("forward", arm) for arm in ("residual_momentum_only", "momentum_residual_momentum", "momentum_control")),
        n_deflation=9,
    ),
    "earnings_daily_v1": Clock(
        name="earnings_daily_v1", route="discovery",
        registration=Path("registrations/2026-09-earnings-daily.toml"), design=Path("docs/earnings_design.md"),
        arms=(ArmSpec("ear_event", 1, 21), ArmSpec("announcement_premium", 1, 5)),
        forward_start=dt.date(2026, 9, 24),
        bootstrap={"block_bars": 42, "n_resamples": 2000, "seed": 20260929, "confidence": 0.95},
        kill="own_interval", thresholds=(("interim", 126), ("promotion", 252)),
        deflation_rows=((EARNINGS_DAILY_READ, "ear_event"), (EARNINGS_DAILY_READ, "announcement_premium"),
                        (EARNINGS_READ, "ear_monthly")),
        n_deflation=3, book=Path("examples/earnings/daily.toml"),
    ),
    "buffer_monthly_v1": Clock(
        name="buffer_monthly_v1", route="discovery",
        registration=Path("registrations/2026-09-buffer-monthly.toml"), design=Path("docs/buffer_design.md"),
        arms=(ArmSpec("momentum_buffered", 21, 21), ArmSpec("ear_monthly_buffered", 21, 21)),
        forward_start=dt.date(2026, 9, 25),
        bootstrap={"block_bars": 42, "n_resamples": 2000, "seed": 20261001, "confidence": 0.95},
        kill="own_interval", thresholds=(("interim", 126), ("promotion", 252)),
        deflation_rows=EIGHT_BUFFER_ROWS, n_deflation=8, book=Path("examples/buffer/monthly.toml"),
    ),
    "buffer_daily_v1": Clock(
        name="buffer_daily_v1", route="discovery",
        registration=Path("registrations/2026-09-buffer-daily.toml"), design=Path("docs/buffer_design.md"),
        arms=(ArmSpec("ear_event_buffered", 1, 21), ArmSpec("premium_buffered", 1, 5)),
        forward_start=dt.date(2026, 9, 25),
        bootstrap={"block_bars": 42, "n_resamples": 2000, "seed": 20261001, "confidence": 0.95},
        kill="own_interval", thresholds=(("interim", 126), ("promotion", 252)),
        deflation_rows=EIGHT_BUFFER_ROWS, n_deflation=8, book=Path("examples/buffer/daily.toml"),
    ),
    "vol_momentum_v1": Clock(
        name="vol_momentum_v1", route="replication",
        registration=Path("registrations/2026-09-characteristics.toml"), design=Path("docs/vol_momentum_design.md"),
        arms=(ArmSpec("momentum_control", 21, 21),),
        forward_start=dt.date(2026, 9, 24),
        bootstrap={"block_bars": 42, "n_resamples": 2000, "seed": 20260925, "confidence": 0.95},
        kill="vol_scaled", thresholds=_vol_thresholds(), n6_criterion=False,
    ),
}

#: The vol-momentum rule and kill thresholds, restated from registrations/2026-09-vol-momentum.toml
#: and checked against it at read time (``check_vol_registration``).
VOL_REGISTRATION = Path("registrations/2026-09-vol-momentum.toml")
VOL_SOURCE = {"target_vol": 0.12, "lookback_sessions": 126, "refresh_sessions": 21, "max_scale": 2.0}
VOL_KILL = {"max_drawdown": 0.25, "min_forward_fidelity": 0.5, "scale_change_cost_bps": 10.0}


# --------------------------------------------------------------------- calendar and geometry


def session_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    return sessions.sessions_between(start, end)


def shift_sessions(day: dt.date, n: int) -> dt.date:
    """The session ``n`` sessions after (n > 0) or before (n < 0) session ``day``."""
    if not sessions.is_session(day):
        raise ValueError(f"{day} is not a session")
    out = day
    for _ in range(abs(n)):
        out = sessions.next_session(out) if n > 0 else sessions.previous_session(out)
    return out


def first_forward_session(clock: Clock) -> dt.date:
    day = clock.forward_start
    return day if sessions.is_session(day) else sessions.next_session(day)


def build_fold_table(first_test: dt.date, asof: dt.date, *, table_id: str, source: str) -> FoldTable:
    """The certified rolling geometry cut from sessions ``[first_test - 312, asof]``; the as-of bar is reserved."""
    start = shift_sessions(first_test, -FORMATION)
    index = pd.DatetimeIndex(pd.to_datetime(session_dates(start, asof))).tz_localize(TZ)
    splitter = RollingWalkForward(formation_bars=FORMATION, test_bars=TEST, step_bars=STEP)
    return splitter.fold_table(index, table_id=table_id, source=source)


def test_sessions(table: FoldTable) -> list[dt.date]:
    out: list[dt.date] = []
    for fold in table.folds:
        out.extend(session_dates(fold.test_start.date(), fold.test_end.date()))
    return out


def forward_session_count(clock: Clock, asof: dt.date) -> int:
    """Forward sessions whose returns a dataset ending at ``asof`` holds, under the registered geometry."""
    first = first_forward_session(clock)
    if asof <= first:
        return 0
    try:
        table = build_fold_table(first, asof, table_id="count", source="count")
    except ValueError:  # fewer sessions than one fold needs
        return 0
    return sum(1 for day in test_sessions(table) if day >= first)


@dataclass(frozen=True)
class ScheduledRead:
    clock: str
    name: str
    threshold: int
    asof: dt.date


def scheduled_reads(clock: Clock) -> list[ScheduledRead]:
    """Every read with its scheduled as-of: the session ``threshold`` sessions after the first forward session."""
    first = first_forward_session(clock)
    out = []
    for name, threshold in clock.thresholds:
        try:
            asof = shift_sessions(first, threshold)
        except sessions.SessionTableRangeError:
            continue  # beyond the bundled session table; extend the table before this read
        out.append(ScheduledRead(clock.name, name, threshold, asof))
    return out


def due_read(clock: Clock, asof: dt.date) -> ScheduledRead:
    """The latest read whose scheduled as-of is on or before ``asof``; refuses before the first."""
    reads = scheduled_reads(clock)
    due = [read for read in reads if read.asof <= asof]
    if not due:
        n = forward_session_count(clock, asof)
        first = reads[0]
        raise ReadRefused(
            f"{clock.name}: {n} forward session(s) at as-of {asof} (first forward session "
            f"{first_forward_session(clock)}); its first read ({first.name}) needs {first.threshold}, "
            f"first possible at as-of {first.asof}. Nothing was read."
        )
    return due[-1]


def fold_table_for(clock: Clock, read: ScheduledRead, *, warmup: bool = False) -> FoldTable:
    """The read's fold table, written by the forward rule and checked to hold exactly the threshold."""
    first = first_forward_session(clock)
    if warmup:
        first_test = shift_sessions(first, -VOL_WARMUP_FOLDS * TEST)
        table = build_fold_table(first_test, first, table_id=f"{clock.name}_warmup_{read.asof}",
                                 source=f"research.scripts.forward_read warm-up for {read.name}")
        if len(table.folds) != VOL_WARMUP_FOLDS or table.folds[-1].test_end.date() != sessions.previous_session(first):
            raise ReadRefused(f"warm-up table does not end on the session before {first} (N7)")
        return table
    table = build_fold_table(first, read.asof, table_id=f"{clock.name}_{read.name}_{read.asof}",
                             source=f"research.scripts.forward_read {clock.name} {read.name}")
    held = test_sessions(table)
    if held[0] != first or len(held) != read.threshold:
        raise ReadRefused(f"{clock.name} {read.name}: the table holds {len(held)} sessions from {held[0]}, "
                          f"not {read.threshold} from {first} (N7)")
    return table


def write_fold_table(table: FoldTable, path: Path) -> tuple[FoldTable, str]:
    """Write the TOML form and return the table re-read from it with its content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table.to_toml(), encoding="utf-8")
    loaded = FoldTable.from_toml(path)
    if loaded.content_hash != table.content_hash:
        raise ReadRefused(f"{path} does not round-trip to the same geometry (N7)")
    return loaded, loaded.content_hash


# --------------------------------------------------------------------- derived registration and book


_HEADER = re.compile(r"\[\[?\s*([A-Za-z0-9_.]+)\s*\]\]?\s*(#.*)?")


def replace_lines(text: str, replacements: Sequence[tuple[str, str, int]], *, where: str,
                  section: str | None = None) -> str:
    """Replace whole lines matched by each regex, which must match exactly ``count`` lines.

    With ``section``, only lines inside that TOML table count ("" is the top level, before any header).
    """
    lines = text.splitlines()
    tables, current = [], ""
    for line in lines:
        header = _HEADER.fullmatch(line.strip()) if line.lstrip().startswith("[") else None
        if header:
            current = header.group(1)
        tables.append(current)
    for pattern, new, count in replacements:
        regex = re.compile(pattern)
        hits = [i for i, line in enumerate(lines)
                if regex.fullmatch(line) and (section is None or tables[i] == section)]
        if len(hits) != count:
            place = f" in [{section}]" if section else (" at the top level" if section == "" else "")
            raise ReadRefused(f"{where}: {pattern!r} matches {len(hits)} line(s){place}, expected {count} (N7)")
        for i in hits:
            lines[i] = new
    return "\n".join(lines) + "\n"


def keep_arms(text: str, arms: Sequence[str], *, where: str) -> str:
    """Drop every ``[[arms]]`` block (with its ``[arms.*]`` sub-tables) whose id is not in ``arms``.

    Comment and blank lines that end a dropped block stay, so the next block's comments are never lost.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "[[arms]]"]
    drop: set[int] = set()
    found: set[str] = set()
    for start in starts:
        end = start + 1
        while end < len(lines):
            header = _HEADER.fullmatch(lines[end].strip()) if lines[end].lstrip().startswith("[") else None
            if header and not (header.group(1).startswith("arms.") and not lines[end].strip().startswith("[[")):
                break
            end += 1
        while end - 1 > start and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
            end -= 1
        ids = [m.group(1) for line in lines[start:end] if (m := re.fullmatch(r'id = "(.*)"', line))]
        if len(ids) != 1:
            raise ReadRefused(f"{where}: an [[arms]] block at line {start + 1} declares {len(ids)} ids (N7)")
        if ids[0] in arms:
            found.add(ids[0])
        else:
            drop.update(range(start, end))
    if found != set(arms):
        raise ReadRefused(f"{where} does not declare arms {sorted(set(arms) - found)} (N7)")
    return "\n".join(line for i, line in enumerate(lines) if i not in drop) + "\n"


def forward_selection_set(clock: Clock) -> str:
    return "forward_v1" if clock.name == "forward_v1" else f"{clock.name}_forward"


def derive_registration(source_text: str, *, clock: Clock, read: ScheduledRead, table: FoldTable,
                        fold_path: Path, freeze_path: Path, book_path: Path | None, source_path: Path,
                        suffix: str = "", selection_set: str | None = None) -> str:
    """The source registration with only identity, sample, folds, freeze record (and book) changed."""
    source_sha = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    first, last = table.first_decision.date(), table.last_decision.date()
    history = table.folds[0].formation_start.date()
    n_arms = len(clock.arms)
    by_section: dict[str, list[tuple[str, str, int]]] = {
        "": [(r'id = ".*"', f'id = "{clock.name}-{read.name}{suffix}-{read.asof}"', 1),
             (r"created = \d{4}-\d{2}-\d{2}", f"created = {read.asof}", 1),
             (r'selection_set = ".*"', f'selection_set = "{selection_set or forward_selection_set(clock)}"', 1),
             (r"trial_budget = \d+", f"trial_budget = {n_arms}", 1)],
        "sample": [(r"history_start = \d{4}-\d{2}-\d{2}", f"history_start = {history}", 1),
                   (r"first_decision = \d{4}-\d{2}-\d{2}", f"first_decision = {first}", 1),
                   (r"last_decision = \d{4}-\d{2}-\d{2}", f"last_decision = {last}", 1)],
        "folds": [(r'table = ".*"', f'table = "{fold_path}"', 1)],
        "frozen": [(r'freeze_record = ".*"', f'freeze_record = "{freeze_path}"', 1)],
    }
    if book_path is not None:
        by_section["benchmark"] = [(r'control = ".*"', f'control = "{book_path}"', 1)]
    body = keep_arms(source_text, [spec.arm for spec in clock.arms], where=str(source_path))
    for section, replacements in by_section.items():
        body = replace_lines(body, replacements, where=str(source_path), section=section)
    header = (f"# Derived by research.scripts.forward_read for {clock.name} {read.name}{suffix} at as-of {read.asof}\n"
              f"# from {source_path} (sha256 {source_sha}). Only id, created, selection_set, trial_budget,\n"
              f"# [sample], [folds].table, [frozen].freeze_record{', [benchmark].control' if book_path else ''} "
              "differ from it, and only the arms this read runs are kept.\n")
    return header + body


def derive_book(source_text: str, *, panel_path: Path, panel_sha256: str, source_path: Path) -> str:
    """The control book with only the announcement panel's path and hash changed."""
    count = source_text.count(f'panel = "{REGISTERED_PANEL}"')
    if count < 1 or source_text.count(f'panel_sha256 = "{REGISTERED_PANEL_SHA256}"') != count:
        raise ReadRefused(f"{source_path} does not pin the registered panel {REGISTERED_PANEL} (N7)")
    header = (f"# Derived by research.scripts.forward_read from {source_path} "
              f"(sha256 {hashlib.sha256(source_text.encode()).hexdigest()}); only the panel path and hash differ.\n")
    return header + replace_lines(source_text, [
        (r'panel = ".*"', f'panel = "{panel_path}"', count),
        (r'panel_sha256 = ".*"', f'panel_sha256 = "{panel_sha256}"', count),
    ], where=str(source_path))


# --------------------------------------------------------------------- statistics


def spec_for(boot: Mapping[str, Any], interval: int) -> SharpeBootstrapSpec:
    return SharpeBootstrapSpec(block_bars=int(boot["block_bars"]), n_resamples=int(boot["n_resamples"]),
                               seed=int(boot["seed"]), confidence=float(boot["confidence"]),
                               decision_interval_bars=int(interval))


def annualized(interval: Mapping[str, Any]) -> dict[str, float]:
    root = math.sqrt(PER_YEAR)
    return {key: float(interval[key]) * root for key in ("sharpe_a", "lower", "upper")}


def own_interval(net: pd.Series, boot: Mapping[str, Any], interval: int) -> dict[str, Any]:
    found = sharpe_interval(net.to_numpy(float), spec_for(boot, interval), label="net").to_dict()
    return {"interval": found, "annualized": annualized(found), "entirely_below_zero": bool(found["upper"] < 0.0)}


def paired_interval(net: pd.Series, base: pd.Series, boot: Mapping[str, Any], interval: int) -> dict[str, Any]:
    common = net.index.intersection(base.index)
    found = sharpe_difference_interval(net.loc[common].to_numpy(float), base.loc[common].to_numpy(float),
                                       spec_for(boot, interval), label="difference").to_dict()
    return {"interval": found, "annualized": annualized(found), "n_common": int(len(common)),
            "entirely_below_zero": bool(found["upper"] < 0.0)}


def periodic_sharpe(returns: pd.Series) -> float:
    values = returns.dropna().to_numpy(float)
    sd = values.std(ddof=1) if len(values) > 1 else float("nan")
    return float(values.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    """Largest fall of compounded wealth from its running peak, the peak starting at 1 (forward start)."""
    wealth = np.cumprod(1.0 + returns.to_numpy(float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float(np.max(1.0 - wealth / peak)) if len(wealth) else float("nan")


def deflation(net: pd.Series, trial_sharpes: Mapping[str, float], n_rows: int) -> dict[str, Any]:
    """DSR against exactly the registered rows; fewer Sharpes than rows refuses (never deflate against fewer)."""
    if len(trial_sharpes) != n_rows or not all(np.isfinite(v) for v in trial_sharpes.values()):
        raise ReadRefused(f"deflation needs {n_rows} finite row Sharpes, got {dict(trial_sharpes)} (N7)")
    values = list(trial_sharpes.values())
    return {"dsr": float(deflated_sharpe_ratio_with_n(net.to_numpy(float), values, n_rows)), "n_rows": n_rows,
            "row_periodic_sharpes": dict(trial_sharpes)}


def tbill_hurdle(rf_annual: float, cash_yield: float, long_over_equity: float, vol_annual: float) -> dict[str, float]:
    """SPEC §3 as amended 2026-09-25: annualized Sharpe hurdle ``(rf - y * max(0, 1 - L/E)) / vol``."""
    idle = max(0.0, 1.0 - float(long_over_equity))
    excess = float(rf_annual) - float(cash_yield) * idle
    sharpe = excess / float(vol_annual) if vol_annual > 0 else float("nan")
    return {"rf_annual": float(rf_annual), "cash_yield": float(cash_yield), "long_over_equity": float(long_over_equity),
            "idle_fraction": idle, "return_hurdle_annual": excess, "vol_annual": float(vol_annual),
            "sharpe_annual": sharpe, "sharpe_periodic": sharpe / math.sqrt(PER_YEAR)}


def average_long_over_equity(weights: pd.DataFrame) -> float:
    """Mean over sessions of the long side's weight sum (target weights are fractions of equity)."""
    return float(weights.clip(lower=0.0).sum(axis=1).mean())


def bond_equivalent_yield(discount_pct: pd.Series, days: int = 91) -> pd.Series:
    """FRED DTB3 (discount basis, percent) to bond-equivalent yield, as a fraction."""
    d = discount_pct / 100.0
    return 365.0 * d / (360.0 - days * d)


def held_book_ic(decisions: pd.DataFrame, opens: pd.DataFrame, horizon: int, interval: int) -> dict[str, Any]:
    """Rank IC of each decision's target weights, over held names, with the open-to-open return over ``horizon``.

    The return runs from the open after the decision (N2) to ``horizon`` sessions later. One cross-section per
    decision; overlapping horizons (daily decisions, horizon > 1) get a Newey-West standard error.
    """
    from research.scripts.earnings_read import forward_open_returns, newey_west_mean

    forward = forward_open_returns(opens, horizon)
    values: dict[pd.Timestamp, float] = {}
    for t in decisions.index:
        if t not in forward.index:
            continue
        row = decisions.loc[t]
        names = row.index[row.fillna(0.0) != 0.0]
        pair = pd.concat([row[names], forward.loc[t, names.intersection(forward.columns)]], axis=1).dropna()
        if len(pair) >= 3 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
            values[t] = float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))
    lags = max(math.ceil(horizon / interval) - 1, 0)
    stats = newey_west_mean(pd.Series(values, dtype=float), lags=lags)
    return {**stats, "horizon": horizon, "decision_interval": interval, "nw_lags": lags}


def n6_viability(ic: Mapping[str, Any], n_eff: float, horizon: int, gross_sharpe: float, net_sharpe: float,
                 hurdle_sharpe: float) -> dict[str, Any]:
    """SPEC N6 on the discovery route: the ceiling at the one-sided lower-95 IC must exceed cost drag plus hurdle."""
    root = math.sqrt(n_eff * PER_YEAR / horizon) if n_eff > 0 else float("nan")
    lower_ic = float(ic["mean"]) - ONE_SIDED_95 * float(ic["se"])
    ceiling, ceiling_lower = float(ic["mean"]) * root, lower_ic * root
    drag = float(gross_sharpe) - float(net_sharpe)
    bar = drag + float(hurdle_sharpe)
    return {"ic": float(ic["mean"]), "ic_se": float(ic["se"]), "ic_lower_95": lower_ic, "n_eff": float(n_eff),
            "horizon": horizon, "ceiling": ceiling, "ceiling_at_lower_95": ceiling_lower, "cost_drag": drag,
            "hurdle": float(hurdle_sharpe), "bar": bar, "realized_gross_above_ceiling": bool(gross_sharpe > ceiling),
            "viable": bool(np.isfinite(ceiling_lower) and ceiling_lower > bar)}


# --------------------------------------------------------------------- decisions


@dataclass
class ArmInputs:
    """One arm's forward record, cut to forward sessions."""

    net: pd.Series
    cost: pd.Series
    weights: pd.DataFrame  # target weights held on every forward session
    decisions: pd.DataFrame  # target weights on decision bars
    n_eff: float
    gross_exposure: pd.Series | None = None  # the record's own gross exposure (costs.csv ``gross``)
    turnover: pd.Series | None = None  # costs.csv ``turnover``, forward sessions


@dataclass
class ReadContext:
    rf_annual: float
    cash_yield: float
    monitor_verdict: str
    trial_sharpes: dict[str, float] = field(default_factory=dict)  # "path:arm" -> periodic Sharpe (in-house rows)
    opens: pd.DataFrame | None = None


def arm_statistics(spec: ArmSpec, inputs: ArmInputs, context: ReadContext, clock: Clock) -> dict[str, Any]:
    net, gross = inputs.net, inputs.net + inputs.cost.reindex(inputs.net.index).fillna(0.0)
    vol = float(net.std(ddof=1) * math.sqrt(PER_YEAR))
    net_sharpe = periodic_sharpe(net) * math.sqrt(PER_YEAR)
    gross_sharpe = periodic_sharpe(gross) * math.sqrt(PER_YEAR)
    hurdle = tbill_hurdle(context.rf_annual, context.cash_yield, average_long_over_equity(inputs.weights), vol)
    out: dict[str, Any] = {
        "n_forward": int(len(net)), "first": str(net.index[0].date()), "last": str(net.index[-1].date()),
        "net_sharpe_annualized": net_sharpe, "gross_sharpe_annualized": gross_sharpe,
        "net_sharpe_periodic": periodic_sharpe(net), "annualized_vol": vol,
        "net_total_return": float(np.prod(1.0 + net.to_numpy(float)) - 1.0),
        "annualized_return": float(np.prod(1.0 + net.to_numpy(float)) ** (PER_YEAR / len(net)) - 1.0),
        "psr_vs_zero": float(probabilistic_sharpe_ratio(net.to_numpy(float), 0.0)),
        "turnover_annual": None if inputs.turnover is None else float(inputs.turnover.mean() * PER_YEAR),
        "max_drawdown": max_drawdown(net), "charged_cost_annual": float(inputs.cost.mean() * PER_YEAR),
        "interval": own_interval(net, clock.bootstrap, spec.interval), "hurdle": hurdle, "n_eff": inputs.n_eff,
    }
    if context.opens is not None:
        ic = held_book_ic(inputs.decisions, context.opens, spec.horizon, spec.interval)
        out["held_book_ic"] = ic
        out["n6"] = n6_viability(ic, inputs.n_eff, spec.horizon, gross_sharpe, net_sharpe, hurdle["sharpe_annual"])
    return out


def decide(clock: Clock, read: ScheduledRead, arms: Mapping[str, ArmInputs], context: ReadContext) -> dict[str, Any]:
    """Every registered number and verdict for one scheduled read of one discovery clock."""
    if clock.kill == "vol_scaled":
        raise ValueError("the vol-momentum kill is read by decide_vol")
    stats = {spec.arm: arm_statistics(spec, arms[spec.arm], context, clock) for spec in clock.arms}
    verdicts: dict[str, dict[str, Any]] = {}
    forward_sharpes = {f"forward:{arm}": stats[arm]["net_sharpe_periodic"] for arm in stats}
    for spec in clock.arms:
        arm, s = spec.arm, stats[spec.arm]
        if clock.kill == "paired_vs_benchmark":
            if arm == clock.benchmark_arm:
                verdicts[arm] = {"role": "benchmark"}
                continue
            paired = paired_interval(arms[arm].net, arms[clock.benchmark_arm].net, clock.bootstrap, spec.interval)
            s["paired_vs_benchmark"] = paired
            killed = paired["entirely_below_zero"]
        else:
            killed = s["interval"]["entirely_below_zero"]
        verdict: dict[str, Any] = {"killed": bool(killed)}
        if read.name == "promotion" and not killed:
            rows = {}
            for source, row_arm in clock.deflation_rows:
                key = f"{source}:{row_arm}"
                rows[key] = forward_sharpes[f"forward:{row_arm}"] if source == "forward" else context.trial_sharpes[key]
            s["deflation"] = deflation(arms[arm].net, rows, clock.n_deflation)
            criteria = {
                "net_edge": bool(s["net_total_return"] > 0.0 and s["net_sharpe_periodic"] > 0.0),
                "dsr_above_half": bool(s["deflation"]["dsr"] > 0.5),
                "above_tbill_hurdle": bool(s["net_sharpe_periodic"] > s["hurdle"]["sharpe_periodic"]),
                "monitor_not_in_contradiction": context.monitor_verdict != "refuted",
            }
            if clock.n6_criterion:
                criteria["n6_viable"] = bool(s.get("n6", {}).get("viable", False))
            verdict.update(criteria=criteria, promoted=all(criteria.values()))
        verdicts[arm] = verdict
    pairwise = {}
    if clock.kill == "paired_vs_benchmark":  # registered_read's comparison: every arm pair, paired on common sessions
        specs = list(clock.arms)
        for i, left in enumerate(specs):
            for right in specs[i + 1:]:
                pairwise[f"{left.arm}-{right.arm}"] = paired_interval(arms[left.arm].net, arms[right.arm].net,
                                                                      clock.bootstrap, left.interval)
    return {"clock": clock.name, "read": read.name, "scheduled_asof": str(read.asof), "threshold": read.threshold,
            "arms": stats, "verdicts": verdicts, "pairwise": pairwise,
            "killed": sorted(a for a, v in verdicts.items() if v.get("killed")),
            "promoted": sorted(a for a, v in verdicts.items() if v.get("promoted")),
            "monitor_verdict": context.monitor_verdict}


def decide_vol(read: ScheduledRead, warmup_net: pd.Series, forward: ArmInputs, gross_exposure: pd.Series,
               reference: pd.Series, boot: Mapping[str, Any] = CLOCKS["vol_momentum_v1"].bootstrap) -> dict[str, Any]:
    """vol_momentum_v1's kill: FB's forward stream scaled by the rule, pre-forward sessions in the estimate only.

    The overlay is the replication's (``vol_momentum_replication.scaled_in_house``): return x scale, less the
    registered one-way cost on |change in scale| x the record's gross exposure. ``reference`` is the latest French
    UMD vintage scaled by the same rule (``scaled_reference``); the forward correlation is read over its overlap
    with the forward sessions.
    """
    from research.scripts.vol_momentum_replication import rule_scale

    stream = pd.concat([warmup_net, forward.net])
    if not stream.index.is_monotonic_increasing or stream.index.has_duplicates:
        raise ReadRefused("warm-up and forward sessions overlap or are out of order (N7)")
    scale = pd.Series(rule_scale(stream.to_numpy(float), VOL_SOURCE), index=stream.index)
    previous = scale.shift(1).fillna(1.0)
    gross = gross_exposure.reindex(stream.index).fillna(0.0)
    change_cost = (scale - previous).abs() * gross * VOL_KILL["scale_change_cost_bps"] / 1e4
    scaled = (scale * stream - change_cost).loc[forward.net.index]
    interval = own_interval(scaled, boot, 21)
    drawdown = max_drawdown(scaled)
    joined = pd.concat({"book": scaled, "reference": reference}, axis=1).dropna()
    correlation = float(joined["book"].corr(joined["reference"])) if len(joined) > 2 else float("nan")
    conditions = {
        "a_interval_below_zero": interval["entirely_below_zero"],
        "b_drawdown_above_limit": bool(drawdown > VOL_KILL["max_drawdown"]),
        "c_fidelity_below_floor": bool(not np.isfinite(correlation) or correlation < VOL_KILL["min_forward_fidelity"]),
    }
    return {"clock": "vol_momentum_v1", "read": read.name, "scheduled_asof": str(read.asof),
            "threshold": read.threshold, "interval": interval, "max_drawdown_from_forward_peak": drawdown,
            "reference_correlation": {"daily_corr": correlation, "n_overlap": int(len(joined)),
                                      "first": str(joined.index[0].date()) if len(joined) else None,
                                      "last": str(joined.index[-1].date()) if len(joined) else None},
            "mean_scale": float(scale.loc[forward.net.index].mean()), "conditions": conditions,
            "killed": bool(any(conditions.values()))}


# --------------------------------------------------------------------- inputs from disk


def warmup_selection_set(clock: Clock) -> str:
    return f"{clock.name}_warmup"


def read_dir(clock: Clock, read: ScheduledRead) -> Path:
    return READS_ROOT / f"{read.asof}" / clock.name


def dataset_dir(read: ScheduledRead) -> Path:
    return Path("data/frozen") / f"forward_{read.asof}"


def panel_dir(read: ScheduledRead) -> Path:
    return Path("data/frozen") / f"earnings_forward_{read.asof}"


def in_house_sharpes(clock: Clock) -> dict[str, float]:
    out = {}
    for source, arm in clock.deflation_rows:
        if source == "forward":
            continue
        record = json.loads(Path(source).read_text())
        out[f"{source}:{arm}"] = float(record["deflation"]["trial_periodic_sharpes"][arm])
    return out


def cash_yield() -> tuple[float, dict[str, Any]]:
    stack = tomllib.loads(COST_STACK_CASH.read_text())
    bps = float(stack["cash_income_bps_annual"])
    return bps / 1e4, {"path": str(COST_STACK_CASH), "cash_income_bps_annual": bps,
                       "reference": stack.get("source", {}).get("cash_income", {}).get("reference")}


def monitor_verdict() -> tuple[str, dict[str, Any]]:
    from prism.live.monitor import paper_monitor_read

    run_dir = Path(LIVE_RUN_DIR_POINTER.read_text().strip())
    found = paper_monitor_read(run_dir / "equity.jsonl")
    return str(found["verdict"]), {"run_dir": str(run_dir), **{k: found[k] for k in ("n", "verdict") if k in found}}


def tbill_rate(first: dt.date, last: dt.date) -> tuple[float, dict[str, Any]]:
    from prism.regime.fetch import FredClient

    series = FredClient.from_env().series("DTB3", start=str(first), end=str(last)).dropna()
    if series.empty:
        raise ReadRefused(f"FRED DTB3 has no observation in {first}..{last} (N7)")
    bey = bond_equivalent_yield(series)
    payload = series.to_csv().encode()
    return float(bey.mean()), {"series": "FRED:DTB3", "n_obs": int(len(series)), "first": str(series.index[0].date()),
                               "last": str(series.index[-1].date()), "sha256": hashlib.sha256(payload).hexdigest(),
                               "conversion": "discount_to_bond_equivalent (91 days), mean over the forward sessions"}


def finished_record(selection_set: str, arm: str, data_id: str) -> Path:
    """The latest finished run of ``arm`` in ``selection_set`` on the read's dataset (its ``data_id``)."""
    path = Path("trials") / f"{selection_set}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    done = [r for r in rows if r.get("event") == "finished" and r.get("arm_id") == arm and r.get("data_id") == data_id]
    if not done:
        raise ReadRefused(f"no finished row for {arm} on data_id {data_id} in {path}; run the arms first")
    return Path("results") / str(done[-1]["run_id"]) / "record.json"


def load_arm(record_path: Path, dataset: Path, forward_first: dt.date) -> tuple[ArmInputs, pd.DataFrame]:
    """The arm's record from its run directory, cut to forward sessions, plus the dataset's opens."""
    from prism.io.dataset import Dataset
    from research.scripts.earnings_read import load_opens_and_members
    from research.scripts.registered_read import (
        _series,
        contribution_breadth,
        daily_targets,
    )

    run = record_path.parent
    returns = _series(run / "returns.csv", "daily_return")
    costs = _series(run / "costs.csv")
    decisions = _series(run / "target_weights.csv")
    weights = daily_targets(decisions, returns.index) if len(decisions) < len(returns) else decisions
    keep = returns.index.tz_localize(None).normalize() >= pd.Timestamp(forward_first)
    net, weights = returns[keep], weights.reindex(returns.index)[keep]
    ds = Dataset.open(dataset)
    end = str(net.index[-1].date())
    closes = ds.view("research_vintage", end, vintage=ds.manifest.vintage).panel(
        list(decisions.columns), end, "close").reindex(index=net.index)
    breadth = contribution_breadth(weights, closes)
    opens, _ = load_opens_and_members(dataset)
    naive = decisions.copy()
    naive.index = naive.index.tz_localize(None).normalize()
    inputs = ArmInputs(net=net, cost=costs["total"].reindex(net.index).fillna(0.0), weights=weights,
                       decisions=naive[naive.index >= pd.Timestamp(forward_first)],
                       n_eff=float(breadth["n_eff_book_contribution"]),
                       gross_exposure=costs["gross"].reindex(returns.index),
                       turnover=costs["turnover"].reindex(net.index).fillna(0.0))
    return inputs, opens


def dataset_id(read: ScheduledRead) -> str:
    return json.loads((dataset_dir(read) / "manifest.json").read_text())["id"]


# --------------------------------------------------------------------- subcommands


def status(asof: dt.date) -> dict[str, Any]:
    """The dry run: forward sessions per clock at ``asof`` and each scheduled read's as-of. Reads nothing."""
    out = {}
    for clock in CLOCKS.values():
        reads = scheduled_reads(clock)
        try:
            due: str | None = due_read(clock, asof).name
            refusal = None
        except ReadRefused as exc:
            due, refusal = None, str(exc)
        first = first_forward_session(clock)
        elapsed = len(session_dates(first, sessions.previous_session(asof))) if asof > first else 0
        out[clock.name] = {"first_forward_session": str(first), "elapsed_sessions_with_returns": elapsed,
                           "forward_sessions_in_folds": forward_session_count(clock, asof), "due": due,
                           "refusal": refusal,
                           "scheduled": {r.name: str(r.asof) for r in reads}}
    return out


def jobs(clock: Clock) -> list[tuple[str, bool, str]]:
    """(file suffix, warm-up?, selection set) of each derived registration a read runs."""
    out = [("", False, forward_selection_set(clock))]
    if clock.kill == "vol_scaled":
        out.insert(0, ("-warmup", True, warmup_selection_set(clock)))
    return out


def plan(clock: Clock, read: ScheduledRead, panel: tuple[Path, str] | None = None) -> dict[str, Any]:
    """Write the read's fold table(s) and derived registration(s) (and book); return their paths and hashes."""
    base = read_dir(clock, read)
    written: dict[str, Any] = {"read": {**read.__dict__, "asof": str(read.asof)}}
    book_path = None
    if clock.book is not None:
        if panel is None:
            raise ReadRefused(f"{clock.name} needs the rebuilt announcement panel; run freeze first")
        base.mkdir(parents=True, exist_ok=True)
        book_path = base / clock.book.name
        text = derive_book(clock.book.read_text(), panel_path=panel[0], panel_sha256=panel[1], source_path=clock.book)
        book_path.write_text(text, encoding="utf-8")
        written["book"] = {"path": str(book_path), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                           "panel": str(panel[0]), "panel_sha256": panel[1]}
    for suffix, warmup, selection_set in jobs(clock):
        table = fold_table_for(clock, read, warmup=warmup)
        fold_path = base / f"folds{suffix}.toml"
        table, fold_hash = write_fold_table(table, fold_path)
        registration_path = base / f"registration{suffix}.toml"
        text = derive_registration(clock.registration.read_text(), clock=clock, read=read, table=table,
                                   fold_path=fold_path, freeze_path=base / f"freeze{suffix}.json",
                                   book_path=book_path, source_path=clock.registration, suffix=suffix,
                                   selection_set=selection_set)
        registration_path.write_text(text, encoding="utf-8")
        written[f"fold_table{suffix}"] = {"path": str(fold_path), "sha256": fold_hash, "n_folds": len(table.folds),
                                          "first_test": str(table.first_decision.date()),
                                          "last_test": str(table.last_decision.date())}
        written[f"registration{suffix}"] = {"path": str(registration_path), "selection_set": selection_set,
                                            "sha256": hashlib.sha256(text.encode()).hexdigest()}
    (base / "plan.json").write_text(json.dumps(written, indent=2, default=str) + "\n")
    return written


def freeze_commands(clock: Clock, read: ScheduledRead) -> list[list[str]]:
    """The pinned data steps at the scheduled as-of, in order; each writes only under its own directories."""
    universe = read_dir(clock, read).parent / "universe"
    py = sys.executable
    steps = [[py, "-m", "prism.scripts.build_sp500_universe", "--asof", str(read.asof), "--start_date", HISTORY_START,
              "--end_date", str(read.asof), "--out_dir", str(universe)],
             [py, "-m", "research.scripts.forward_read", "import", "--asof", str(read.asof)]]
    if clock.book is not None:
        steps += [[py, "-m", "prism.scripts.build_fundamentals", "--membership",
                   str(universe / f"sp500_membership_{read.asof}.parquet"), "--asof", str(read.asof),
                   "--ledger-dir", str(universe), "--refresh",
                   "--name-matches", "data/universe/cik_name_matches_2026-09-24.json",
                   "--review-decisions", "data/universe/cik_review_decisions_2026-09-24.json",
                   "data/universe/cik_nearmiss_decisions_2026-09-24.json"],
                  [py, "-m", "prism.scripts.build_earnings_panel", "--dataset", str(dataset_dir(read)),
                   "--out", str(panel_dir(read))]]
    return steps


def import_dataset(asof: dt.date) -> dict[str, Any]:
    """Freeze Twelve Data bars from 2020-01-01 and the rebuilt membership with the certified importer."""
    from prism.io.legacy_import import build_legacy_dataset

    read = ScheduledRead("any", "import", 0, asof)
    universe = READS_ROOT / f"{asof}" / "universe"
    resolved = universe / f"sp500_pit_resolved_{asof}.txt"
    symbols = [line.split("#", 1)[0].strip() for line in resolved.read_text().splitlines()]
    coverage = json.loads((universe / f"sp500_coverage_{asof}.json").read_text())
    report = build_legacy_dataset(dataset_dir(read).resolve(), symbols=[s for s in symbols if s],
                                  cache_dir=Path("data").resolve(), start=HISTORY_START, end=str(asof),
                                  vintage=f"forward_{asof}", index_id="sp500",
                                  membership_file=universe / f"sp500_membership_{asof}.parquet",
                                  universe_note=str(resolved), coverage=coverage)
    return {"root": str(dataset_dir(read)), "data_id": report.manifest.id}


def run_arms(clock: Clock, read: ScheduledRead) -> list[str]:
    """Freeze the score panel and run every arm of each derived registration through construction_experiment."""
    from prism.io.dataset import Dataset
    from prism.registration import load_registration
    from prism.scripts.construction_experiment import freeze_scores, run_arm
    from prism.workspace import resolve_workspace

    workspace = resolve_workspace(None)
    dataset = Dataset.open(dataset_dir(read))
    base = read_dir(clock, read)
    done = []
    for suffix, _, _ in jobs(clock):
        registration = load_registration(base / f"registration{suffix}.toml", workspace=workspace)
        freeze_scores(workspace=workspace, registration=registration, dataset=dataset)
        for spec in clock.arms:
            run_arm(workspace=workspace, registration=registration, dataset=dataset, arm_id=spec.arm,
                    capital=float(registration.frozen.capital_scenarios[0]))
            done.append(f"{registration.id}:{spec.arm}")
    return done


def read_clock(clock: Clock, read: ScheduledRead, code_commit: str) -> dict[str, Any]:
    base = read_dir(clock, read)
    first = first_forward_session(clock)
    data_id = dataset_id(read)
    y, y_meta = cash_yield()
    verdict, monitor_meta = monitor_verdict()
    if clock.kill == "vol_scaled":
        check_vol_registration()
        from research.scripts.registered_read import _series

        (spec,) = clock.arms
        forward, _ = load_arm(finished_record(forward_selection_set(clock), spec.arm, data_id), dataset_dir(read), first)
        warm_path = finished_record(warmup_selection_set(clock), spec.arm, data_id)
        warm = _series(warm_path.parent / "returns.csv", "daily_return")
        warm_gross = _series(warm_path.parent / "costs.csv")["gross"]
        warm = warm[warm.index.tz_localize(None).normalize() < pd.Timestamp(first)]
        gross = pd.concat([warm_gross.reindex(warm.index), forward.gross_exposure.reindex(forward.net.index)])
        for series in (warm, forward.net, gross):
            series.index = series.index.tz_localize(None).normalize()
        reference, ref_meta = french_reference()
        summary = decide_vol(read, warm, forward, gross, reference)
        summary["reference"] = ref_meta
    else:
        opens = None
        arms: dict[str, ArmInputs] = {}
        for spec in clock.arms:
            record = finished_record(forward_selection_set(clock), spec.arm, data_id)
            arms[spec.arm], opens = load_arm(record, dataset_dir(read), first)
        any_net = next(iter(arms.values())).net
        rf, rf_meta = tbill_rate(any_net.index[0].date(), any_net.index[-1].date())
        context = ReadContext(rf_annual=rf, cash_yield=y, monitor_verdict=verdict,
                              trial_sharpes=in_house_sharpes(clock), opens=opens)
        summary = decide(clock, read, arms, context)
        summary["tbill"] = rf_meta
    summary.update(code_commit=code_commit, data_id=data_id, cash_yield=y_meta, monitor=monitor_meta,
                   plan=json.loads((base / "plan.json").read_text()))
    (base / "read.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    return summary


def check_vol_registration() -> None:
    reg = tomllib.loads(VOL_REGISTRATION.read_text())
    for key, value in VOL_SOURCE.items():
        if reg["source"][key] != value:
            raise ReadRefused(f"{VOL_REGISTRATION} [source].{key} = {reg['source'][key]}, restated {value} (N7)")
    kill = reg["kill"]
    if (kill["max_drawdown"], kill["min_forward_fidelity"], kill["first_read_min_sessions"],
            kill["read_every_sessions"]) != (VOL_KILL["max_drawdown"], VOL_KILL["min_forward_fidelity"], 126, 63):
        raise ReadRefused(f"{VOL_REGISTRATION} [kill] differs from the restated rule (N7)")
    if reg["implementation"]["scale_change_cost_bps"] != VOL_KILL["scale_change_cost_bps"]:
        raise ReadRefused(f"{VOL_REGISTRATION} scale_change_cost_bps differs from the restated rule (N7)")


def french_reference() -> tuple[pd.Series, dict[str, Any]]:
    """The latest French daily UMD vintage, scaled by the rule; its hash and vintage line are recorded."""
    from research.scripts.factor_lab import fetch_sources, parse_french
    from research.scripts.vol_momentum_replication import scaled_reference

    reg = tomllib.loads(VOL_REGISTRATION.read_text())
    ref = reg["reference"]
    fetched = fetch_sources({"ff_mom_daily_latest": {"url": ref["momentum_url"]}}, READS_ROOT / "_french", refresh=True)
    entry = fetched["ff_mom_daily_latest"]
    table, vintage = parse_french(Path(entry["path"]).read_bytes())
    scaled = scaled_reference(table[ref["momentum_column"]].dropna(), reg["source"], int(ref["warmup_sessions"]))
    return scaled["gross"], {"url": ref["momentum_url"], "sha256": entry["sha256"], "vintage": vintage,
                             "last": str(table.index[-1].date())}


def code_commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], capture_output=True, text=True).stdout.strip()
    return head + ("+dirty" if dirty else "")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("status", "plan", "freeze", "import", "run", "read"))
    parser.add_argument("--asof", type=dt.date.fromisoformat, default=dt.date.today())
    parser.add_argument("--clock", choices=sorted(CLOCKS))
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(status(args.asof), indent=2))
        return 0
    if args.command == "import":
        scheduled = {r.asof for c in CLOCKS.values() for r in scheduled_reads(c)}
        if args.asof not in scheduled or args.asof > dt.date.today():
            print(f"REFUSED: {args.asof} is not a scheduled read's as-of that has passed; nothing was imported",
                  file=sys.stderr)
            return 2
        print(json.dumps(import_dataset(args.asof), indent=2))
        return 0
    if args.clock is None:
        parser.error(f"{args.command} needs --clock")
    clock = CLOCKS[args.clock]
    try:
        read = due_read(clock, args.asof)
    except ReadRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if args.command == "freeze":
        for step in freeze_commands(clock, read):
            print("+", " ".join(step), flush=True)
            subprocess.run(step, check=True)
        return 0
    if args.command == "plan":
        panel = None
        if clock.book is not None:
            path = panel_dir(read) / "panel.parquet"
            panel = (path, hashlib.sha256(path.read_bytes()).hexdigest())
        print(json.dumps(plan(clock, read, panel), indent=2, default=str))
        return 0
    if args.command == "run":
        print(json.dumps(run_arms(clock, read), indent=2))
        return 0
    summary = read_clock(clock, read, code_commit())
    print(json.dumps({k: summary.get(k) for k in ("clock", "read", "scheduled_asof", "killed", "promoted")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
