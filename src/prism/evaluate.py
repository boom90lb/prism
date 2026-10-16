"""The evaluator (engine specification §3.8): walk-forward, train-only, recorded.

Two entry points, and the difference between them is what they may write.

:func:`walk_forward_evaluate` is the evaluation. It cuts the sample with a
frozen fold table (§3.8: a table of dates, content-hashed and recorded, never
splitter keyword arguments restated), fits every component on that fold's
formation window and nothing outside it, decides every test bar through its own
point-in-time view, accounts the stitched target sequence once, scores the test
rows only, writes a :class:`~prism.validation.trials.ResultRecord` with
``record_schema = 1``, and records the attempt in the workspace's hash-chained
trial log under the protocol of §3.12: a ``started`` row before any
computation, exactly one terminal row after it, and no invocation ever writing
a false ``incomplete`` for a run that is still executing.

:func:`smoke_evaluate` is the fixed engineering run of stage 1, kept working
unchanged. It writes **no** record and **no** trial row: a smoke run answers no
research question, and a smoke path that minted evidence rows would put an
engineering run into the selection set promotion is read from.

**Two accounting engines, and they are never silently substituted (§3.8).**
``historical_constant_weight`` is ``backtest_target_weights`` unchanged: it
holds *weights* between decision bars and re-levers to them as prices move,
charging nothing for the trade that would take. It is a reproduction mode, is
labeled one in every record it writes, and exists to reproduce recorded
evidence. ``canonical`` is the fixed-share simulator of §3.6
(``execution/simulate.py``) on ``PortfolioState``: shares are the state, fills
happen at the next open under the declared ``[fills]`` policy, holding costs
accrue per bar, and events are applied on their effective dates. It is what a
forward claim rests on.

The two are distinguishable in every record — ``evaluation.accounting_engine``,
``evaluation.holding_process``, ``conventions.executable_return`` and the
caveat in ``warnings`` all move together, and
``validation/trials.py::record_schema_problems`` refuses a record whose label
disagrees with its own contents. They are two configurations: ``config_id``
differs, so a canonical run is never read as a restatement of a reproduction.

From this stage the record carries a **real** ``visibility_mode`` and
``vintage``: the evaluator cuts every decision bar through a
:class:`~prism.io.dataset.DatasetView` under the declared mode, so the record
states the mode that was enforced rather than the placeholder stage 2 had to
write.

Timing is the engine's, not this module's, and it is the same in both: a target
decided at close *t* fills at the open of *t+1*. Between decision bars the
target rows are all-NaN, which the reproduction engine reads as "no new pending
order" and the canonical engine reads as "hold the shares".
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import platform
import socket
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prism.composition import (
    ComponentPlan,
    ComponentState,
    CompositionResult,
    compose,
    due_on_grid,
    tranche_plans,
)
from prism.config import ExecutionConfig
from prism.execution.cost_model import (
    SPREAD_BUCKET_SCHEDULE_V1,
    CostInputs,
    CostModel,
    CostStack,
    SpreadSpec,
    bucket_spread_bps,
)
from prism.execution.participation import participation_capped_targets
from prism.execution.simulate import (
    DecisionCall,
    FillPolicy,
    SimulationResult,
    simulate,
)
from prism.execution.target_weights import (
    PortfolioBacktestResult,
    backtest_target_weights,
    compute_portfolio_metrics,
)
from prism.io import sessions
from prism.io.dataset import (
    MODES,
    Dataset,
    DatasetError,
    DatasetView,
    StageNotAvailableError,
)
from prism.io.events import (
    distribution_amounts,
    require_price_basis,
    series_price_basis,
)
from prism.portfolio.construct import cap_book, closed_form_band, step_no_trade_band
from prism.spec import (
    Eligibility,
    LoadedComponent,
    Specification,
    apply_to_weights,
    engine_version,
)
from prism.validation.trials import (
    CANONICAL_ENGINE,
    DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    ENGINE_EXECUTABLE_RETURN,
    ENGINE_HOLDING_PROCESS,
    ENGINEERING_LOG,
    EXPLORATORY_LOG,
    FAILED,
    REPRODUCTION_ENGINE,
    RUN_CHANNELS,
    RUN_KINDS,
    STARTED,
    UNAVAILABLE,
    ResultRecord,
    RunRecorder,
    TrialLog,
    attempt_counts,
    cap_evidence_tier,
    classify_claim_tier,
    compute_trial_metrics,
    empty_repair_counts,
    reconcile_trial_log,
    utc_now,
)
from prism.validation.walk_forward import (
    FoldTable,
    RollingFold,
    RollingWalkForward,
    resolve_fold_geometry,
)
from prism.workspace import Workspace

#: The two accounting engines of §3.8. ``historical_constant_weight`` holds
#: weights and is a reproduction mode; ``canonical`` holds shares and is what a
#: new evaluation runs on. Neither is ever substituted for the other, and the
#: record says which one produced it.
ENGINES: tuple[str, ...] = (REPRODUCTION_ENGINE, CANONICAL_ENGINE)

#: The visibility modes of §3.2 an evaluation may declare. ``research_vintage``
#: cuts at the session close of a frozen vintage; ``replay_recorded``
#: reproduces what the workspace possessed at a recorded wall clock and reads
#: no vintage label at all.
VISIBILITY_MODES: tuple[str, ...] = MODES

#: Weight-space unit the reproduction driver already uses when no capital is
#: declared (§3.14).
DEFAULT_CAPITAL = 1.0

RECORDING_STATEMENT = (
    "No ResultRecord and no trial-log row were written. The recording protocol, the trial ledger, and the fold "
    "table are stage 2's; a smoke run answers no research question and enters no selection set."
)


class EvaluationError(ValueError):
    """The evaluation cannot run as configured."""


@dataclass(frozen=True)
class SmokeResult:
    """One smoke run: what it ran on, what it produced, and what it did not exercise."""

    spec_path: Path
    strategy: str
    engine: str
    capital: float
    dataset_root: Path
    manifest_id: str
    vintage: str
    universe: tuple[str, ...]
    decision_dates: tuple[str, ...]
    span: tuple[str, str]
    backtest: PortfolioBacktestResult
    unexercised_scenarios: tuple[str, ...]
    unexercised_note: str
    in_universe_unapplied_scenarios: tuple[str, ...] = ()
    in_universe_unapplied_note: str = ""

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.backtest.metrics)

    def text(self) -> str:
        """The printed summary, including the unexercised scenario ids (§3.2)."""
        lines = [
            "prism evaluate --smoke",
            f"specification   {self.spec_path}",
            f"strategy        {self.strategy}",
            f"engine          {self.engine}  ({ENGINE_HOLDING_PROCESS.get(self.engine, 'unknown')}: "
            + (
                "weights are held constant between decisions and re-levered without charge"
                if self.engine == REPRODUCTION_ENGINE
                else "shares are held between decisions and weights drift with prices"
            )
            + ")",
            f"capital         {self.capital!r}",
            f"dataset         {self.dataset_root}",
            f"manifest        {self.manifest_id}  vintage {self.vintage}",
            f"span            {self.span[0]} .. {self.span[1]}",
            f"universe        {len(self.universe)} sid(s)",
            f"decisions       {len(self.decision_dates)} bar(s): "
            f"{self.decision_dates[0]} .. {self.decision_dates[-1]}",
            f"return bars     {len(self.backtest.returns)}",
            "",
            "summary",
        ]
        for key, value in self.backtest.metrics.items():
            lines.append(f"  {key:<20} {value:.6g}")
        lines.append("")
        if self.unexercised_scenarios:
            lines.append(f"unexercised scenarios  {', '.join(self.unexercised_scenarios)}")
            lines.append(f"  {self.unexercised_note}")
        else:
            lines.append("unexercised scenarios  none: every scenario the fixture writes is applied at this stage")
        if self.in_universe_unapplied_scenarios:
            lines.append(f"  ... and in the universe but unapplied  {', '.join(self.in_universe_unapplied_scenarios)}")
            lines.append(f"  {self.in_universe_unapplied_note}")
        lines.append("")
        lines.append(RECORDING_STATEMENT)
        return "\n".join(lines)


def smoke_evaluate(
    spec: Specification,
    *,
    workspace: Workspace,
    dataset: Dataset | None = None,
    engine: str = REPRODUCTION_ENGINE,
    capital: float = DEFAULT_CAPITAL,
    execution: ExecutionConfig | None = None,
    canonical: "CanonicalSettings | None" = None,
    start: str | None = None,
    end: str | None = None,
) -> SmokeResult:
    """Run ``spec`` over the workspace's dataset and return the run's summary.

    Nothing is written: the caller prints the result. Every decision bar is read
    through its own ``research_vintage`` view, so the panel a component scores
    is the panel the as-of path would have handed it, rather than a slice of one
    long frame taken after the fact.
    """
    if engine not in ENGINES:
        raise EvaluationError(f"unknown engine {engine!r}; the engines of §3.8 are {', '.join(ENGINES)}")
    if engine == CANONICAL_ENGINE and canonical is None:
        raise EvaluationError(
            "the canonical engine charges dollars, so the smoke path needs CanonicalSettings carrying a declared "
            "CostStack (§3.6); there is deliberately no 'no costs' default"
        )
    gridded = [loaded.component_id for loaded in spec.components if loaded.spec.grid_schedule is not None]
    if gridded:
        raise EvaluationError(
            f"component(s) {gridded} declare tranches or phase_offset_bars, and the smoke path decides every "
            "update_every_bars sessions from the first decidable bar with no component state, so it would run a "
            "different schedule from the one declared; evaluate them with walk_forward_evaluate"
        )
    if engine == REPRODUCTION_ENGINE and canonical is not None:
        raise EvaluationError(
            "CanonicalSettings were declared for a historical_constant_weight run, which would ignore them (§3.8)"
        )
    if capital <= 0:
        raise EvaluationError(f"capital must be > 0, got {capital!r}")
    if spec.strategy.combination != "position_space":
        raise StageNotAvailableError(
            f"[strategy] combination={spec.strategy.combination!r} runs the calibrated convex flow (F2), whose "
            "estimator and solver arrive at stage 4; the smoke path runs combination='position_space'",
            stage=4,
        )
    if not spec.components:
        raise EvaluationError(f"{spec.path} declares no components")
    for loaded in spec.components:
        if loaded.declarations["fitted_state"] == "train_only":
            # The smoke path never fits: it scores unfitted components over one
            # span, so a train-only component would be scored before it was
            # fitted. Fitting has an owner, a window and a cadence
            # (`prepare_components`), and the evaluation that supplies all
            # three is `walk_forward_evaluate` — which is where a train-only
            # component belongs and where its fit artifact is written.
            raise EvaluationError(
                f"component {loaded.component_id!r} declares fitted_state='train_only' and the smoke path does "
                "not fit: it would be scored before it was fitted. Run it through walk_forward_evaluate, which "
                "fits once per fold on that fold's formation window and records the fitted_state_ref"
            )

    ds = dataset if dataset is not None else Dataset.open(workspace.data)
    vintage = ds.manifest.vintage
    if vintage is None:
        raise EvaluationError(f"the dataset at {ds.root} carries no vintage label; open it with a frozen manifest")
    span_start, span_end = _span(ds, start, end)
    grid = sessions.sessions_between(_as_date(span_start), _as_date(span_end))
    if len(grid) < 2:
        raise EvaluationError(f"the span {span_start}..{span_end} holds {len(grid)} session(s); a fill needs two")

    visibility = _Visibility(mode="research_vintage", vintage=vintage)
    final_view = visibility.view(ds, grid[-1].isoformat())
    universe = _universe(spec, ds, visibility, grid)
    required = max(int(loaded.declarations["required_history"]) for loaded in spec.components)
    stride = min(int(loaded.spec.update_every_bars) for loaded in spec.components)
    if len(grid) < required + 1:
        raise EvaluationError(
            f"the components need {required} bars of history and the span holds {len(grid)} session(s); the "
            "smoke fixture must cover at least one decision bar plus the fill bar that follows it"
        )

    opens = final_view.panel(list(universe), grid[-1].isoformat(), "open")
    stamp_of = {stamp.date(): stamp for stamp in opens.index}
    targets = pd.DataFrame(np.nan, index=opens.index, columns=list(universe), dtype=float)
    decided: list[str] = []
    for position in range(required - 1, len(grid) - 1, stride):
        day = grid[position]
        if day not in stamp_of:
            raise EvaluationError(
                f"decision session {day.isoformat()} has no row in the open panel; the session grid and the "
                "fixture's bars disagree, which is a dataset defect rather than a bar to skip (N7)"
            )
        view = visibility.view(ds, day.isoformat())
        row = _decide(spec, view, day.isoformat(), universe)
        if row is None:
            continue
        targets.loc[stamp_of[day]] = row.to_numpy(dtype=float)
        decided.append(day.isoformat())
    if not decided:
        raise EvaluationError(
            "no decision bar produced a target row; every component declined the whole cross-section, which is a "
            "degenerate run, not an empty result (N7)"
        )
    dividends = _dividends(final_view, universe, opens, span_start, grid[-1].isoformat())

    if engine == CANONICAL_ENGINE:
        assert canonical is not None  # checked above
        last = grid[-1].isoformat()
        smoke_events = final_view.events(list(universe), span_start, last)
        decisions, _ = _canonical_decisions(targets)
        simulation = simulate(
            _by_bar(opens),
            _by_bar(final_view.panel(list(universe), last, "close").reindex(index=opens.index)),
            _by_bar(decisions),
            cost_model=CostModel(canonical.cost_stack, CostInputs(as_of=last, scope="run"), equity=float(capital)),
            volumes=_by_bar(final_view.panel(list(universe), last, "volume").reindex(index=opens.index)),
            events=smoke_events,
            price_basis=_smoke_price_basis(ds, final_view, canonical.price_basis, smoke_events, last),
            initial_cash=float(capital),
            fills=canonical.fills,
            increment=float(canonical.share_increment),
            min_notional=float(canonical.min_notional),
            zero_crossing_clamp=bool(canonical.zero_crossing_clamp),
            bars=[_iso(stamp) for stamp in opens.index],
        )
        result = _canonical_backtest_result(simulation, index=opens.index, universe=universe, targets=targets)
    else:
        result = backtest_target_weights(
            opens,
            targets,
            execution or ExecutionConfig(),
            initial_capital=capital,
            dividends=dividends,
        )
    assumptions = dict(ds.manifest.assumptions or {})
    unexercised = tuple(str(s) for s in assumptions.get("scenarios_written_not_applied", ()))
    note = str(assumptions.get("scenarios_written_not_applied_note", ""))
    # Section 4's "written but not exercised" table has two halves. The list
    # above is the half whose sids belong to no index; this is the half the
    # evaluation does select and price while the scenario's own effect goes
    # unapplied. Both halves are printed, because an operator reading the
    # summary is owed the whole set of things this stage does not test.
    in_universe = tuple(str(s) for s in assumptions.get("scenarios_unapplied_in_universe", ()))
    in_universe_note = str(assumptions.get("stage1_unapplied_in_universe", ""))
    return SmokeResult(
        spec_path=spec.path,
        strategy=spec.strategy.name,
        engine=engine,
        capital=float(capital),
        dataset_root=ds.root,
        manifest_id=ds.manifest.id,
        vintage=vintage,
        universe=tuple(universe),
        decision_dates=tuple(decided),
        span=(span_start, span_end),
        backtest=result,
        unexercised_scenarios=unexercised,
        unexercised_note=note,
        in_universe_unapplied_scenarios=in_universe,
        in_universe_unapplied_note=in_universe_note,
    )


def _smoke_price_basis(
    dataset: Dataset, view: DatasetView, declared: str | None, events: pd.DataFrame, end: str
) -> str | dict[str, str]:
    """The smoke path's price basis, under the same rule the evaluator applies."""
    basis = _price_basis(dataset, declared)
    if basis != PER_SERIES_BASIS:
        return basis
    sids = [] if events.empty else dict.fromkeys(events["sid"].astype(str).tolist())
    return {sid: series_price_basis(view.bars([sid], end)) for sid in sids}


def _decide(spec: Specification, view: DatasetView, day: str, universe: Sequence[str]) -> pd.Series | None:
    """One decision bar: score every component, apply its rank-to-weight rule, compose (flow F1).

    A name that has left the point-in-time universe gets an explicit 0.0, not
    NaN: leaving the index is an exit, and NaN means "hold what you have" to the
    accounting engine.
    """
    members = _members(spec, view, universe)
    if not members:
        return None
    close = view.panel(members, day, "close")
    volume = view.panel(members, day, "volume")
    eligible = _eligibility_mask(spec.strategy.eligibility, view, close, volume, members)
    _bind_membership_loaded(spec, view, close)

    total: pd.Series | None = None
    holding_weight = 0.0
    for loaded in spec.components:
        basis = _basis_panel(loaded, view, members, day, close)
        scores = loaded.component.score(basis, volume)
        scores = pd.to_numeric(scores, errors="coerce").where(eligible)
        marks = loaded.component.instructions(basis, volume)
        no_decision = _marked(marks, "no_decision")
        sub = apply_to_weights(
            loaded.spec.to_weights, scores, loaded.spec.to_weights_params, close=basis, no_decision=no_decision
        )
        liquidate = [sid for sid in _marked(marks, "liquidate") if sid in sub.index]
        if liquidate:
            sub.loc[liquidate] = 0.0
        if bool(sub.isna().all()):
            continue
        holding_weight += float(loaded.spec.weight)
        weighted = sub * float(loaded.spec.weight)
        total = weighted if total is None else total.add(weighted)
    if total is None:
        return None
    if spec.strategy.renormalize and holding_weight > 0.0:
        declared = sum(float(loaded.spec.weight) for loaded in spec.components)
        total = total * (declared / holding_weight)
    # A sid outside the point-in-time universe is an explicit flat, never NaN:
    # NaN reaches the accounting engine as "hold what you have".
    row = pd.Series(0.0, index=pd.Index(list(universe)), dtype=float)
    row.loc[total.index] = total.to_numpy(dtype=float)
    return row


def _basis_panel(
    loaded: LoadedComponent, view: DatasetView, members: list[str], day: str, close: pd.DataFrame
) -> pd.DataFrame:
    """The panel the component's declared ``scoring_basis`` names (§3.3)."""
    if loaded.declarations["scoring_basis"] == "total_return":
        return view.total_return(members, day)
    return close


def _marked(marks: pd.Series | None, mark: str) -> list[str]:
    if marks is None or marks.empty:
        return []
    return [str(sid) for sid, value in marks.items() if str(value) == mark]


def _members(spec: Specification, view: DatasetView, universe: Sequence[str]) -> list[str]:
    if spec.strategy.universe_index is None:
        return list(universe)
    members = view.membership(spec.strategy.universe_index)
    return [sid for sid in universe if sid in members]


def _eligibility_mask(
    screen: Eligibility,
    view: DatasetView,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    members: Sequence[str],
) -> pd.Series:
    """The ``[strategy.eligibility]`` screen of §3.3 at the panel's last bar.

    Every gate is computed from information at or before the decision bar. A
    name failing any of them scores NaN — no opinion — rather than being dropped
    from the cross-section, so a name that has just become ineligible is exited
    by the rank-to-weight rule instead of being frozen at its held weight.
    """
    last = close.iloc[-1]
    mask = pd.Series(True, index=close.columns, dtype=bool)
    if screen.min_price > 0.0:
        mask &= pd.to_numeric(last, errors="coerce").fillna(-1.0) >= screen.min_price
    if screen.min_median_dollar_volume > 0.0:
        window = max(int(screen.dollar_volume_window), 1)
        dollar = (close * volume).tail(window)
        median = dollar.median(axis=0, skipna=False)
        mask &= median.fillna(-1.0) >= screen.min_median_dollar_volume
    if screen.return_history_bars > 0:
        finite = close.pct_change(fill_method=None).notna().tail(int(screen.return_history_bars)).sum(axis=0)
        mask &= finite >= int(screen.return_history_bars)
    if screen.listing_seasoning_bars > 0:
        mask &= _seasoned(view, list(members), int(screen.listing_seasoning_bars)).reindex(
            mask.index, fill_value=False
        )
    if screen.exclude_asset_classes:
        securities = view.securities(list(members))
        if not securities.empty and "asset_class" in securities.columns:
            excluded = set(screen.exclude_asset_classes)
            classes = securities.set_index("sid")["asset_class"].astype(str)
            mask &= ~classes.reindex(mask.index).isin(excluded).fillna(False)
    return mask


def _seasoned(view: DatasetView, members: list[str], bars: int) -> pd.Series:
    """Sids whose earliest listing began at least ``bars`` sessions before the decision date.

    A null ``valid_from`` is the ``floor_unknown`` case — the listing opened
    before the sample — and counts as seasoned.
    """
    listings = view.listings(members)
    out = pd.Series(False, index=pd.Index(members), dtype=bool)
    if listings.empty:
        return out
    for sid, rows in listings.groupby("sid"):
        starts = pd.to_datetime(rows["valid_from"], errors="coerce")
        if starts.isna().any():
            out.loc[str(sid)] = True
            continue
        first = starts.min().date()
        out.loc[str(sid)] = len(sessions.sessions_between(first, view.decision_day)) - 1 >= bars
    return out


def _universe(spec: Specification, dataset: Dataset, visibility: "_Visibility", grid: Sequence[dt.date]) -> list[str]:
    """Every sid the strategy could hold over the span, in a stable order.

    The union over *every* decision date, not the membership of one date and
    not a sample of three: a name removed mid-sample was tradeable before its
    removal and its exit has to be accounted, which a column that never existed
    cannot express, and a name that joined and left between two sampled dates
    would be missing from the universe altogether.
    """
    if spec.strategy.universe_index is None:
        listed = list(spec.strategy.universe_list or ())
        if not listed:
            raise EvaluationError(f"{spec.path} declares neither a universe index nor a universe list")
        return _listed_universe(spec, dataset, visibility, grid, listed)
    members: set[str] = set()
    for day in grid:
        view = visibility.view(dataset, day.isoformat())
        members |= view.membership(spec.strategy.universe_index)
    if not members:
        raise EvaluationError(
            f"index {spec.strategy.universe_index!r} has no members over {grid[0]}..{grid[-1]}; an empty universe "
            "is a defect in the dataset or the specification, not an empty result (N7)"
        )
    return sorted(members)


def _listed_universe(
    spec: Specification,
    dataset: Dataset,
    visibility: "_Visibility",
    grid: Sequence[dt.date],
    listed: Sequence[str],
) -> list[str]:
    """``universe_list`` -- a literal *symbol* list (§3.3) -- resolved onto this dataset's sids.

    Two datasets answer the same declaration. One keys its bars by ticker
    (``build_legacy_dataset``, whose documented assumption is ``sid = ticker``):
    the symbol *is* the sid and nothing is resolved, which is the stage 2
    behaviour and is kept bit-identical. The other mints identifiers (``prism
    data import-legacy``, §3.2, ``sec_%06d``): the symbol is resolved through
    the ``listings`` table by :meth:`DatasetView.resolve`, which refuses a
    ticker that is not single-valued on the date -- so ticker reuse is a
    refusal and never a silent choice between two instruments.

    The resolution date is the last session of the evaluated span, not the
    declaration's ``universe_as_of``: the as-of is provenance about when the
    list was registered, while the sid a symbol names has to be answered on a
    date the view can see. A symbol that does not resolve there is refused with
    its declaration named, because an evaluation that quietly dropped a
    declared name would score a smaller book than the one specified.
    """
    resolved: list[str] = []
    view: DatasetView | None = None
    day = grid[-1].isoformat()
    for symbol in listed:
        if dataset.manifest.lists_file(dataset.bars_path(symbol)):
            resolved.append(str(symbol))
            continue
        if view is None:
            view = visibility.view(dataset, day)
        try:
            resolved.append(str(view.resolve(str(symbol), day)))
        except DatasetError as exc:
            raise EvaluationError(
                f"{spec.path} [strategy] universe_list names {symbol!r}, which manifest {dataset.manifest.id} "
                f"carries neither as a sid nor as a ticker resolving to one sid on {day} "
                f"(universe_as_of {spec.strategy.universe_as_of!r}): {exc}"
            ) from exc
    seen = [sid for sid in resolved if resolved.count(sid) > 1]
    if seen:
        raise EvaluationError(
            f"{spec.path} [strategy] universe_list resolves {sorted(set(seen))} more than once; two declared "
            "symbols naming one instrument would double its weight in the cross-section"
        )
    return resolved


def _dividends(view: DatasetView, universe: Sequence[str], prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Cash distributions as an amount-per-share frame on the ex-date session (§3.2 ``events``).

    The placement is :func:`prism.io.events.distribution_amounts` and nothing
    else. There was a second implementation here, and it differed from that one
    in two ways that both lost money silently: an ex-date that is not a session
    on this panel's calendar was dropped rather than credited to the next
    session, and a ``cash_distribution`` payload carrying no ``amount_per_share``
    read as ``0.0`` instead of refusing. A silent zero on cash income is a
    defect (N7), so the rule now lives in one place and both engines share it.

    The accounting engine credits ``dividends[t+1] / open[t]`` to the weights it
    held into bar ``t+1``, so ``prices`` is the open panel that credit is taken
    against and the amount belongs on the bar the distribution goes ex.
    """
    panel = prices.reindex(columns=list(universe))
    events = view.events(list(universe), start, end)
    if events.empty:
        return pd.DataFrame(0.0, index=pd.DatetimeIndex(panel.index), columns=list(universe), dtype=float)
    return distribution_amounts(events, panel)


def _span(dataset: Dataset, start: str | None, end: str | None) -> tuple[str, str]:
    """The evaluated span: whatever the caller declared, and the manifest's coverage for the rest.

    The manifest is consulted only for a bound the caller left open. A caller
    that named both has already answered the question, and refusing it because
    the manifest happens not to carry a ``coverage.sessions`` entry -- which
    ``prism data import-legacy`` does not write -- would refuse a fully
    specified span for a fact about a different import path.
    """
    if start is not None and end is not None:
        return str(start), str(end)
    coverage = dataset.manifest.coverage or {}
    covered = coverage.get("sessions")
    if not covered:
        raise EvaluationError(
            f"manifest {dataset.manifest.id} records no coverage.sessions, so the run cannot tell what span the "
            "dataset covers; pass start and end explicitly"
        )
    return (start or str(covered[0])), (end or str(covered[-1]))


def _as_date(value: str) -> dt.date:
    return dt.date.fromisoformat(str(value)[:10])


# ==========================================================================
# The full evaluator (§3.8) — walk-forward, train-only, recorded
# ==========================================================================

#: The two fold-portfolio modes of §3.8. ``reset`` reproduces the recorded
#: driver: targets are forced to zero on the last two decision rows of every
#: fold, so the liquidation fills at the following open *inside* the fold.
FOLD_PORTFOLIOS: tuple[str, ...] = ("reset", "carry")

#: Cost-refresh cadences of §3.8. ``fold`` rebuilds the cost inputs from each
#: fold's formation window and charges each fold's test bars from its own
#: buckets; ``once`` prices the whole stitched simulation from the first
#: formation window, which is the recorded driver's cadence. Both engines charge
#: whichever is declared — the reproduction engine through
#: ``spread_bps_per_name`` and the canonical engine through the same vector
#: routed into ``CostInputs`` — so the cadence is a property of the run and not
#: of the engine, and the window that produced each refresh is recorded.
COST_REFRESH_CADENCES: tuple[str, ...] = ("fold", "once")

#: ``FittedComponent.fit_scope`` in evaluation. Operation uses
#: ``trailing_window``; the two are recorded so evidence produced under a
#: fold-train fit is never read as describing a trailing-window one (§3.8).
FIT_SCOPE_EVALUATION = "fold_train"

#: Why a ``replay_recorded`` record carries no vintage. The mode reproduces what
#: this workspace possessed at a wall clock, which is not a frozen file list, so
#: there is no vintage label to record and the field says so rather than
#: repeating the manifest's.
NO_VINTAGE_REASON = "replay_recorded cuts at the recorded decision wall clock and reads no vintage label (§3.2)"

#: What the reproduction engine does *not* model, carried on every record it
#: writes so no reader has to know the engine's history to read the number. Its
#: first field is :data:`prism.validation.trials.ENGINE_CAVEAT_MARKER`, which is
#: what the record validator looks for.
REPRODUCTION_CAVEAT = (
    "accounting_engine=historical_constant_weight: weights are held constant between decision bars and "
    "rebalanced implicitly by price drift; there are no share quantities, no partial fills and no per-bar "
    "holding costs. The rebalance it performs between decisions is never charged, so the reproduction number "
    "is not the number the canonical engine produces on the same targets (§3.8)."
)

#: What the canonical engine does model, and the one thing it does not yet: the
#: targets it accounts are constructed from the emitted weight book rather than
#: from the simulated fixed-share state. State-dependent construction is
#: ``assemble`` (§3.8) and arrives with the convex constructor; until it does,
#: this is a declared limitation on the record and not an unstated one.
CANONICAL_NOTE = (
    "accounting_engine=canonical: shares are the state, fills happen at the next open under the declared "
    "[fills] policy, holding costs accrue per bar and events are applied on their effective dates. The target "
    "sequence it accounts is constructed from the emitted weight book, not from the simulated state: the band "
    "and the participation bound read the last emitted target rather than the drifted holding, which is what "
    "keeps the two engines comparable on one set of decisions. State-dependent construction is assemble() (§3.8)."
)

#: The same engine under ``decision_coupling = "account_state"``, where the sentence above
#: about the emitted weight book is exactly what does *not* happen: the constructor is
#: called at the close of each bar with the account as it then stands.
COUPLED_NOTE = (
    "accounting_engine=canonical, decision_coupling=account_state: shares are the state, fills happen at the "
    "next open under the declared [fills] policy, holding costs accrue per bar and events are applied on their "
    "effective dates. Each decision is constructed inside the simulation against the marked equity and the "
    "marked position book of its own bar: the participation restriction is measured on trade dollars over "
    "dollar volume from that equity, batch rounding rounds at the per-sid increment against it, and the "
    "reduce-only box is written on the held position (§3.7, §3.8)."
)


@dataclass(frozen=True)
class _Visibility:
    """The declared visibility mode, and how a decision bar is cut under it (§3.2).

    One object so the mode is decided once, at identity resolution, rather than
    at each of the six places the evaluator opens a view. A view opened any
    other way would be a second, unrecorded, visibility policy.
    """

    mode: str
    vintage: str | None
    decision_time: str | None = None
    decision_offset_minutes: int = 0

    def view(self, dataset: Dataset, day: str) -> DatasetView:
        if self.mode == "research_vintage":
            return dataset.view(
                self.mode,
                day,
                vintage=self.vintage,
                decision_offset_minutes=self.decision_offset_minutes,
            )
        return dataset.view(self.mode, day, decision_time=self.decision_time)


@dataclass(frozen=True)
class CanonicalSettings:
    """Everything the canonical engine needs that the reproduction engine has no term for.

    It is a required argument of a canonical run, never a default. The stack
    declares every charged component including its zeros (§3.6); the quantity
    increment, the dust floor and the fill policy are the execution profile's
    (§3.7, §3.9). A canonical run with no declared stack is refused rather than
    charged nothing, because a zero nobody declared is indistinguishable in the
    record from a zero that was.

    ``share_increment = 0.0`` is the declared *continuous* grid
    (``portfolio/rounding.py::RoundingSettings.for_reproduction``). It is what
    the engine comparison runs on: the recorded configurations size their
    participation bound against ``initial_capital``, so comparing the two
    engines at one capital is the only comparison in which the targets are the
    same, and at ``initial_capital = 1.0`` a whole share is unaffordable.
    """

    cost_stack: CostStack
    share_increment: float = 1.0
    min_notional: float = 0.0
    zero_crossing_clamp: bool = False
    fills: FillPolicy = FillPolicy()
    price_basis: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cost_stack, CostStack):
            raise RunRefused(
                "the canonical engine charges dollars against a declared CostStack (§3.6); "
                f"got {type(self.cost_stack).__name__}"
            )
        if float(self.share_increment) < 0.0:
            raise RunRefused(
                f"share_increment must be >= 0 (0 declares a continuous grid), got {self.share_increment}"
            )
        if float(self.min_notional) < 0.0:
            raise RunRefused(f"min_notional must be >= 0, got {self.min_notional}")

    def identity(self) -> dict[str, object]:
        """What enters ``config_id``: the declared policy, not the objects."""
        return {
            "cost_stack_id": self.cost_stack.stack_id,
            "share_increment": float(self.share_increment),
            "min_notional": float(self.min_notional),
            "zero_crossing_clamp": bool(self.zero_crossing_clamp),
            "fills": dataclasses.asdict(self.fills),
            "price_basis": self.price_basis,
        }

    def to_record(self) -> dict[str, object]:
        return dict(self.identity(), cost_stack=self.cost_stack.declaration())


class RunRefused(EvaluationError):
    """The run was refused before any trial row was written (§3.12, "Ordering").

    A refusal is not an attempt: it writes no trial row. The refusal, its
    reason, and whatever identity resolved go to ``runs/refusals.jsonl``.
    """


@dataclass(frozen=True)
class FittedComponent:
    """One component fitted for one fold (§3.8).

    ``fit_window`` is the range the fit actually saw, so a reader can check the
    train-only boundary from the record alone rather than trusting the loop
    that produced it. ``fitted_state_ref`` is the content hash of the artifact
    under ``workspace.state / "fits"``, null for a component declaring
    ``fitted_state = "none"``.
    """

    loaded: LoadedComponent
    component: object
    fitted_state_ref: str | None
    fitted_through_bar: str | None
    fit_scope: str
    fit_window: tuple[str, str] | None
    fit_rows: int

    @property
    def component_id(self) -> str:
        return str(self.loaded.component_id)

    def plan(self, *, combination: str = "position_space") -> ComponentPlan:
        """The composition plan for this component, carrying the fitted object."""
        return dataclasses.replace(
            ComponentPlan.from_loaded(self.loaded, combination=combination), component=self.component
        )

    def plans(self, *, combination: str = "position_space") -> tuple[ComponentPlan, ...]:
        """The plans composition runs for this component: one, or one per declared tranche.

        Every tranche carries the same fitted object, so a tranched component is
        fitted once per fold and scored by whichever tranche is due.
        """
        plan = self.plan(combination=combination)
        grid = self.loaded.spec.grid_schedule
        if grid is None:
            return (plan,)
        tranches, offset = grid
        return tranche_plans(plan, tranches, offset)

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "fit_cutoff": self.fitted_through_bar,
            "fit_window": None if self.fit_window is None else list(self.fit_window),
            "fit_rows": int(self.fit_rows),
            "fitted_state_ref": self.fitted_state_ref,
            "fit_scope": self.fit_scope,
        }


@dataclass(frozen=True)
class CalibrationRequest:
    """What the evaluator hands a calibration hook, once per fold (§3.4, §3.8).

    Every panel in it ends at ``train_end`` and starts at ``train_start``: the
    hook is called inside the train-only boundary, and a hook that wants a
    later bar has to ask for a different fold, not a longer panel. The
    estimator itself (``calibration.py::fit_calibration``) arrives at stage 4;
    this is the seam it plugs into, which is what stage 2 owes it.
    """

    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    scoring_panels: Mapping[str, pd.DataFrame]
    volume: pd.DataFrame | None
    components: tuple[FittedComponent, ...]
    spec: Specification


@dataclass(frozen=True)
class FoldEvaluation:
    """One fold: what was fitted on it, what was decided in it, what it earned."""

    fold: int
    train: tuple[str, str]
    test: tuple[str, str]
    n_test_bars: int
    decisions: tuple[str, ...]
    boundary_decisions: tuple[str, ...]
    components: tuple[FittedComponent, ...]
    calibration: Mapping[str, object] | None
    excluded: bool
    exclusion_reason: str | None
    metrics: Mapping[str, object]
    turnover: Mapping[str, float]

    def to_record(self) -> dict[str, object]:
        return {
            "fold": int(self.fold),
            "train": {"start": self.train[0], "end": self.train[1]},
            "test": {"start": self.test[0], "end": self.test[1]},
            "n_test_bars": int(self.n_test_bars),
            "n_decisions": len(self.decisions),
            "boundary_decisions": list(self.boundary_decisions),
            "components": [component.to_record() for component in self.components],
            "calibration": None if self.calibration is None else dict(self.calibration),
            "excluded": bool(self.excluded),
            "exclusion_reason": self.exclusion_reason,
            "metrics": dict(self.metrics),
            "turnover": dict(self.turnover),
        }


@dataclass(frozen=True)
class EvaluationResult:
    """One recorded evaluation: the record, the paths, and the objects behind them."""

    record: ResultRecord
    record_path: Path
    log_path: Path
    run_id: str
    config_id: str
    data_id: str
    fold_table: FoldTable
    folds: tuple[FoldEvaluation, ...]
    scored: PortfolioBacktestResult
    full: PortfolioBacktestResult
    trial_rows: tuple[Mapping[str, object], ...]
    reconciled: Mapping[str, object]

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.scored.metrics)


def mint_run_id() -> str:
    """A fresh run identity. Unique per execution, never derived from the configuration.

    Two executions of one configuration are two runs and one attempt: the
    attempt count keys on ``arm_id`` inside a registration and on ``config_id``
    outside one (§3.12), so a deterministic re-run inflates no count even
    though it mints a new ``run_id``.
    """
    return "run-" + uuid.uuid4().hex[:16]


def resolve_log_path(
    workspace: Workspace,
    *,
    selection_set_id: str | None,
    run_channel: str,
) -> Path:
    """Where a run's rows go (§3.12, "Log routing").

    A registered run appends to ``trials/<selection_set_id>.jsonl``; an
    evaluation with no registration appends to ``trials/exploratory.jsonl``,
    which belongs to no selection set and cannot be cited for promotion; an
    engineering-channel run appends to ``runs/engineering.jsonl``, which enters
    no attempt count at all.
    """
    if run_channel not in RUN_CHANNELS:
        raise RunRefused(f"unknown run_channel {run_channel!r}; §3.12 declares {RUN_CHANNELS}")
    if run_channel == "engineering":
        return workspace.runs / f"{ENGINEERING_LOG}.jsonl"
    return workspace.trial_log(selection_set_id or EXPLORATORY_LOG)


def prepare_components(
    spec: Specification,
    view: DatasetView,
    decision_bar: str,
    *,
    fits_dir: Path,
    members: Sequence[str],
    window: tuple[str, str] | None = None,
    previous: Sequence[FittedComponent] = (),
    fit_scope: str = FIT_SCOPE_EVALUATION,
    elapsed_bars: int | None = None,
) -> tuple[FittedComponent, ...]:
    """Fit (or reuse) every component for one decision bar (§3.8).

    ``window`` is the training range. In evaluation it is the fold's formation
    window and nothing outside it may be read — that is the whole train-only
    invariant, and it is enforced here, once, rather than trusted to each
    component. ``training_window_bars = 0`` means "every bar the window
    exposes"; a positive value truncates the window to its last that many bars.

    A component declaring ``fitted_state = "none"`` is never fitted and carries
    a null ``fitted_state_ref``. One declaring ``train_only`` is fitted, its
    state is written under ``fits_dir`` and the artifact's content hash is
    returned, so a replay can reconstruct the fit rather than re-deriving it.
    """
    previous_by_id = {entry.component_id: entry for entry in previous}
    prepared: list[FittedComponent] = []
    for loaded in spec.components:
        component_id = loaded.component_id
        if str(loaded.declarations["fitted_state"]) == "none":
            prepared.append(
                FittedComponent(
                    loaded=loaded,
                    component=loaded.component,
                    fitted_state_ref=None,
                    fitted_through_bar=None,
                    fit_scope=fit_scope,
                    fit_window=None,
                    fit_rows=0,
                )
            )
            continue
        refit_every = max(int(loaded.declarations["refit_every_bars"]), 1)
        carried = previous_by_id.get(component_id)
        if carried is not None and elapsed_bars is not None and elapsed_bars < refit_every:
            prepared.append(carried)
            continue
        panel = _fit_panel(loaded, view, list(members), decision_bar, window)
        required = int(loaded.declarations["required_history"])
        if len(panel) < required:
            raise EvaluationError(
                f"component {component_id!r} needs {required} bars to fit and the training window "
                f"{window} holds {len(panel)}; a short fit is a defect in the fold geometry or the "
                "sample, not a fit to attempt anyway"
            )
        volume = view.panel(list(panel.columns), decision_bar, "volume").reindex(index=panel.index)
        fitted = loaded.component.fit(panel, volume)
        if fitted is None:
            raise EvaluationError(
                f"component {component_id!r}.fit returned None; the contract is 'fits on the training panel "
                "and returns self' (§3.3)"
            )
        prepared.append(
            FittedComponent(
                loaded=loaded,
                component=fitted,
                fitted_state_ref=_save_fit(fitted, component_id=component_id, fits_dir=fits_dir),
                fitted_through_bar=_iso(panel.index[-1]),
                fit_scope=fit_scope,
                fit_window=(_iso(panel.index[0]), _iso(panel.index[-1])),
                fit_rows=int(len(panel)),
            )
        )
    return tuple(prepared)


def _fit_panel(
    loaded: LoadedComponent,
    view: DatasetView,
    members: list[str],
    decision_bar: str,
    window: tuple[str, str] | None,
) -> pd.DataFrame:
    """The panel one component is fitted on: inside the window, never past the cutoff."""
    panel = _basis_panel(loaded, view, members, decision_bar, view.panel(members, decision_bar, "close"))
    if window is not None:
        start = pd.Timestamp(window[0], tz=panel.index.tz)
        end = pd.Timestamp(window[1], tz=panel.index.tz)
        panel = panel.loc[(panel.index >= start) & (panel.index <= end)]
    training_window_bars = int(loaded.declarations["training_window_bars"])
    if training_window_bars > 0:
        panel = panel.tail(training_window_bars)
    if panel.empty:
        raise EvaluationError(
            f"component {loaded.component_id!r}: the training window {window} selects no bar of the "
            f"panel ending {decision_bar}; a fit on nothing is a defect, not an empty result (N7)"
        )
    return panel


def _save_fit(component: object, *, component_id: str, fits_dir: Path) -> str:
    """Write the fit artifact and return its content hash (``fitted_state_ref``)."""
    directory = Path(fits_dir) / component_id
    directory.mkdir(parents=True, exist_ok=True)
    staging = directory / f".staging-{uuid.uuid4().hex}"
    try:
        component.save_state(staging)  # type: ignore[attr-defined]
    except NotImplementedError as exc:
        raise EvaluationError(
            f"component {component_id!r} declares fitted_state='train_only' and must implement save_state: "
            "a train-only fit that cannot be persisted cannot be replayed, and the record's "
            "fitted_state_ref would name nothing (§3.11)"
        ) from exc
    if not staging.is_file():
        raise EvaluationError(
            f"component {component_id!r}.save_state wrote no file at {staging}; the fit artifact is one file "
            "per content hash (§3.11)"
        )
    digest = hashlib.sha256(staging.read_bytes()).hexdigest()
    target = directory / f"{digest}.state"
    if target.exists():
        staging.unlink()
    else:
        staging.replace(target)
    return digest


@dataclass(frozen=True)
class _Resolved:
    """Everything identity resolution produces, before the first decision."""

    dataset: Dataset
    view: DatasetView
    visibility: _Visibility
    vintage: str
    universe: tuple[str, ...]
    session_index: pd.DatetimeIndex
    session_labels: tuple[str, ...]
    opens: pd.DataFrame
    fold_table: FoldTable
    folds: tuple[RollingFold, ...]
    config_id: str
    data_id: str
    evaluation_context: Mapping[str, object]
    synthetic: bool
    price_basis: str

    def view_at(self, day: str) -> DatasetView:
        """The view for one decision bar, under the run's declared visibility mode."""
        return self.visibility.view(self.dataset, day)


def walk_forward_evaluate(
    spec: Specification,
    *,
    workspace: Workspace,
    fold_table: FoldTable | None = None,
    splitter: RollingWalkForward | None = None,
    fold_table_id: str | None = None,
    dataset: Dataset | None = None,
    engine: str = "historical_constant_weight",
    capital: float = DEFAULT_CAPITAL,
    kind: str = "new",
    selection_set_id: str | None = None,
    registration_id: str | None = None,
    registration_hash: str | None = None,
    arm_id: str | None = None,
    run_channel: str = "evidence",
    synthetic: bool | None = None,
    execution: ExecutionConfig | None = None,
    canonical: CanonicalSettings | None = None,
    visibility_mode: str = "research_vintage",
    decision_time: str | None = None,
    price_basis: str | None = None,
    fold_portfolio: str = "reset",
    cost_refresh: str = "fold",
    calibration_hook: Callable[[CalibrationRequest], Mapping[str, object]] | None = None,
    decision_coupling: str = "targets_only",
    calibration_family: str | Path | None = None,
    registration: object | None = None,
    start: str | None = None,
    end: str | None = None,
    reconcile: bool = True,
    max_heartbeat_age_seconds: float = DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    run_id: str | None = None,
    reproduces: Sequence[Mapping[str, str]] = (),
    supersedes: Sequence[str] = (),
    untested_conclusions: Sequence[str] = (),
) -> EvaluationResult:
    """Evaluate ``spec`` walk-forward on a frozen fold table, and record it (§3.8, §3.12).

    The order is the protocol's and is not negotiable. Refusals come first and
    write no row. Identity resolution comes next, and a failure *there* writes a
    ``failed`` row, because a specification that never resolved is still an
    attempt. The ``started`` row precedes the first fit. Exactly one terminal
    row follows. A run that dies between the two leaves a ``started`` row with
    no terminal row, which only :func:`reconcile_trial_log` may close, and only
    for a process it can show is gone.

    Fitting is train-only: one fit per fold on that fold's formation window,
    recorded per fold with its cutoff and window, so the boundary is auditable
    from the record rather than trusted to this loop.

    Folds come from ``fold_table`` — the frozen table a registration declares —
    or, for a run that is generating one rather than reproducing one, from
    ``splitter``. Exactly one of the two, which :func:`resolve_fold_geometry`
    enforces: a recorded table is never silently replaced by parameters that
    happen to be nearby.

    Routing follows §3.12. A registered run appends to
    ``trials/<selection_set_id>.jsonl``, an unregistered one to
    ``trials/exploratory.jsonl``, and a run on a synthetic dataset to
    ``runs/engineering.jsonl`` whatever the caller declared — that last is a
    fact about the manifest, not a choice, and the record carries a warning
    naming the channel the caller asked for.
    """
    _refuse(
        spec,
        engine=engine,
        capital=capital,
        kind=kind,
        run_channel=run_channel,
        fold_portfolio=fold_portfolio,
        cost_refresh=cost_refresh,
        canonical=canonical,
        visibility_mode=visibility_mode,
        decision_time=decision_time,
        decision_coupling=decision_coupling,
    )
    workspace.ensure()
    declared_log = TrialLog(resolve_log_path(workspace, selection_set_id=selection_set_id, run_channel=run_channel))
    identity = run_id or mint_run_id()
    executed = execution or ExecutionConfig()
    try:
        resolved = _resolve(
            spec,
            workspace=workspace,
            dataset=dataset,
            fold_table=fold_table,
            splitter=splitter,
            fold_table_id=fold_table_id,
            engine=engine,
            capital=capital,
            fold_portfolio=fold_portfolio,
            cost_refresh=cost_refresh,
            synthetic=synthetic,
            start=start,
            end=end,
            visibility_mode=visibility_mode,
            decision_time=decision_time,
            canonical=canonical,
            price_basis=price_basis,
            decision_coupling=decision_coupling,
        )
    except Exception as exc:  # identity resolution failed: still an attempt (§3.12)
        declared_log.append(
            {
                "ts": utc_now(),
                "event": FAILED,
                "run_id": identity,
                "config_id": None,
                "data_id": UNAVAILABLE,
                "kind": kind,
                "arm_id": arm_id,
                "selection_set_id": selection_set_id,
                "capital": float(capital),
                "synthetic": bool(synthetic) if synthetic is not None else False,
                "run_channel": run_channel,
                "summary": None,
                "reason": f"identity resolution failed: {type(exc).__name__}: {exc}",
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "heartbeat_at": None,
            }
        )
        raise

    # Routing is settled *after* resolution, because whether the run is
    # synthetic is a fact about the dataset, not a flag the caller chooses: a
    # run on synthetic fixtures writes to the engineering channel, which is in
    # no selection set and enters no attempt count (§3.12, "Which commands
    # write rows"). The declared channel is kept in a warning rather than
    # silently overwritten.
    channel = "engineering" if resolved.synthetic else run_channel
    routing_warnings: list[str] = []
    if channel != run_channel:
        routing_warnings.append(
            f"run_channel {run_channel!r} was declared, but the dataset's manifest is synthetic "
            f"({resolved.dataset.manifest.id}); the run was routed to the engineering channel, which is in no "
            "selection set, enters no attempt count, and no promotion or comparison may cite (§3.12)"
        )
    log = TrialLog(resolve_log_path(workspace, selection_set_id=selection_set_id, run_channel=channel))
    reconciled: Mapping[str, object] = {"path": str(log.path), "n_open": 0, "closed": [], "in_flight": []}
    if reconcile and log.exists():
        reconciled = reconcile_trial_log(log.path, max_heartbeat_age_seconds=max_heartbeat_age_seconds).to_dict()

    if registration is not None:
        # §3.12, "Ordering": the stop-rule and budget checks run before the ``started``
        # row, and a refused run writes no trial row at all. The refusal, its reason and
        # the resolved config_id go to ``runs/refusals.jsonl`` instead, so a refusal is
        # visible without being counted as an attempt.
        _check_registration(
            registration,
            workspace=workspace,
            resolved=resolved,
            log=log,
            run_id=identity,
            arm_id=arm_id,
            kind=kind,
            capital=float(capital),
            run_channel=channel,
        )

    recorder = RunRecorder(
        log=log,
        run_id=identity,
        config_id=resolved.config_id,
        data_id=resolved.data_id,
        kind=kind,
        selection_set_id=selection_set_id,
        arm_id=arm_id,
        capital=float(capital),
        synthetic=resolved.synthetic,
        run_channel=channel,
    )
    started = recorder.started()
    construction_record: dict[str, object] | None = None
    stateful: tuple[PortfolioBacktestResult, dict[str, object]] | None = None
    try:
        if decision_coupling == "account_state":
            assert canonical is not None  # _refuse guarantees it
            (
                targets,
                folds,
                warnings,
                cost_inputs,
                stateful_result,
                stateful_detail,
                construction_record,
            ) = _evaluate_stateful(
                spec,
                resolved,
                workspace=workspace,
                fold_portfolio=fold_portfolio,
                execution=executed,
                capital=float(capital),
                canonical=canonical,
                calibration_family=calibration_family,
                cost_refresh=cost_refresh,
            )
            stateful = (stateful_result, stateful_detail)
        else:
            targets, folds, warnings, cost_inputs = _evaluate_folds(
                spec,
                resolved,
                workspace=workspace,
                fold_portfolio=fold_portfolio,
                calibration_hook=calibration_hook,
                execution=executed,
                capital=float(capital),
            )
        scored_index, excluded = _scored_index(resolved, folds)
        dividends = _dividends(
            resolved.view,
            resolved.universe,
            resolved.opens,
            resolved.session_labels[0],
            resolved.session_labels[-1],
        )
        spread_bps_per_name = _charged_spread(
            resolved, folds=folds, cost_inputs=cost_inputs, cost_refresh=cost_refresh
        )
        if stateful is not None:
            full, engine_detail = stateful
        elif engine == REPRODUCTION_ENGINE:
            full = backtest_target_weights(
                resolved.opens,
                targets,
                executed,
                initial_capital=float(capital),
                dividends=dividends,
                spread_bps_per_name=spread_bps_per_name,
            )
            engine_detail: dict[str, object] = {"note": REPRODUCTION_CAVEAT}
        else:
            assert canonical is not None  # _refuse guarantees it
            full, engine_detail = _simulate_canonical(
                resolved,
                targets,
                canonical=canonical,
                spread_bps_per_name=spread_bps_per_name,
                capital=float(capital),
                warnings=warnings,
            )
        scored = _slice_result(full, scored_index, capital=float(capital))
        folds = tuple(_attribute_turnover(fold, full, resolved) for fold in folds)
        record, record_path = _write_record(
            spec,
            resolved,
            workspace=workspace,
            run_id=identity,
            kind=kind,
            arm_id=arm_id,
            selection_set_id=selection_set_id,
            registration_id=registration_id,
            registration_hash=registration_hash,
            run_channel=channel,
            engine=engine,
            capital=float(capital),
            execution=executed,
            fold_portfolio=fold_portfolio,
            cost_refresh=cost_refresh,
            cost_inputs=cost_inputs,
            charged_spread=spread_bps_per_name,
            canonical=canonical,
            engine_detail=engine_detail,
            folds=folds,
            excluded=excluded,
            scored=scored,
            full=full,
            targets=targets,
            warnings=[*routing_warnings, *warnings],
            log=log,
            reproduces=reproduces,
            supersedes=supersedes,
            untested_conclusions=untested_conclusions,
            calibration_hook=calibration_hook,
            construction_record=construction_record,
        )
    except Exception as exc:
        recorder.failed(f"{type(exc).__name__}: {exc}")
        raise
    finished = recorder.finished(
        {
            "status": record.status["status"],
            "evidence_tier": record.flags["evidence_tier"],
            "n_obs": int(len(scored.returns)),
            "n_folds": len(folds),
            "periodic_sharpe": record.performance["sharpe"]["periodic"],
            "record": str(record_path),
        }
    )
    return EvaluationResult(
        record=record,
        record_path=record_path,
        log_path=log.path,
        run_id=identity,
        config_id=resolved.config_id,
        data_id=resolved.data_id,
        fold_table=resolved.fold_table,
        folds=folds,
        scored=scored,
        full=full,
        trial_rows=(started, finished),
        reconciled=reconciled,
    )


def _check_registration(
    registration: object,
    *,
    workspace: Workspace,
    resolved: _Resolved,
    log: TrialLog,
    run_id: str,
    arm_id: str | None,
    kind: str,
    capital: float,
    run_channel: str,
) -> str:
    """Strict mode and the trial budget, before the ``started`` row (§3.12, §3.13).

    Strict mode is defined against the registration's declaration and nothing else: the
    run's resolved ``config_id`` equals a declared arm's at a declared capital scenario, or
    the run is refused. The refusal is recorded and is not an attempt.
    """
    from prism.registration import RegistrationError

    try:
        matched = registration.check_strict(  # type: ignore[attr-defined]
            config_id=resolved.config_id,
            evaluation_context=resolved.evaluation_context,
            workspace=workspace,
        )
        if arm_id is not None and matched != arm_id:
            raise RegistrationError(
                f"the run declares arm_id={arm_id!r} and its config_id resolves to the declared arm "
                f"{matched!r}; an arm label that names a different configuration would make the record "
                "unreadable (§3.13)"
            )
        if kind not in ("retry", "replay", "reproduction"):
            registration.check_budget(_attempts_spent(log, arm_id=matched))  # type: ignore[attr-defined]
    except RegistrationError as exc:
        _record_refusal(
            workspace,
            {
                "ts": utc_now(),
                "run_id": run_id,
                "config_id": resolved.config_id,
                "data_id": resolved.data_id,
                "registration_id": getattr(registration, "id", None),
                "arm_id": arm_id,
                "kind": kind,
                "capital": float(capital),
                "run_channel": run_channel,
                "reason": f"{type(exc).__name__}: {exc}",
            },
        )
        raise RunRefused(str(exc)) from exc
    return matched


def _attempts_spent(log: TrialLog, *, arm_id: str | None) -> int:
    """Attempts already spent in this selection set, excluding this arm (§3.12).

    Within a registered selection set an attempt is one distinct ``arm_id``; repeating an
    identical execution is a row, not a new attempt, and ``retry``, ``replay`` and
    ``reproduction`` rows are flagged and excluded. So re-running an arm after a crash
    spends nothing, and the third arm cannot be blocked by the first two failing.
    """
    if not log.exists():
        return 0
    spent: set[str] = set()
    for row in log.rows():
        if str(row.get("event")) != STARTED:
            continue
        if str(row.get("kind")) in ("retry", "replay", "reproduction"):
            continue
        other = row.get("arm_id")
        if other is None or (arm_id is not None and str(other) == str(arm_id)):
            continue
        spent.add(str(other))
    return len(spent)


def _record_refusal(workspace: Workspace, row: Mapping[str, object]) -> Path:
    """Append one refusal to ``runs/refusals.jsonl``; a refusal writes no trial row."""
    path = workspace.runs / "refusals.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def _refuse(
    spec: Specification,
    *,
    engine: str,
    capital: float,
    kind: str,
    run_channel: str,
    fold_portfolio: str,
    cost_refresh: str,
    canonical: CanonicalSettings | None = None,
    visibility_mode: str = "research_vintage",
    decision_time: str | None = None,
    decision_coupling: str = "targets_only",
) -> None:
    """Every check that refuses the run before a row is written (§3.12, "Ordering")."""
    if engine not in ENGINES:
        raise RunRefused(f"unknown engine {engine!r}; the engines of §3.8 are {', '.join(ENGINES)}")
    if not float(capital) > 0.0:
        raise RunRefused(f"capital must be > 0, got {capital!r}")
    if kind not in RUN_KINDS:
        raise RunRefused(f"unknown run kind {kind!r}; §3.12 declares {RUN_KINDS}")
    if run_channel not in RUN_CHANNELS:
        raise RunRefused(f"unknown run_channel {run_channel!r}; §3.12 declares {RUN_CHANNELS}")
    if fold_portfolio not in FOLD_PORTFOLIOS:
        raise RunRefused(f"unknown fold_portfolio {fold_portfolio!r}; §3.8 declares {FOLD_PORTFOLIOS}")
    if cost_refresh not in COST_REFRESH_CADENCES:
        raise RunRefused(f"unknown cost_refresh {cost_refresh!r}; §3.8 declares {COST_REFRESH_CADENCES}")
    if visibility_mode not in VISIBILITY_MODES:
        raise RunRefused(f"unknown visibility_mode {visibility_mode!r}; §3.2 declares {VISIBILITY_MODES}")
    if decision_time is not None and visibility_mode != "replay_recorded":
        raise RunRefused("decision_time belongs to replay_recorded; research_vintage cuts at the session close (§3.2)")
    if engine == CANONICAL_ENGINE and canonical is None:
        raise RunRefused(
            "the canonical engine charges dollars, so it needs CanonicalSettings carrying a declared CostStack "
            "(§3.6): the quantity increment, the dust floor and the fill policy are the execution profile's and "
            "every charged component including its zeros is the stack's. There is deliberately no 'no costs' "
            "default, because a zero nobody declared is indistinguishable in the record from one that was"
        )
    if engine == REPRODUCTION_ENGINE and canonical is not None:
        raise RunRefused(
            "CanonicalSettings were declared for a historical_constant_weight run, which has no term for a cost "
            "stack, a quantity increment or a fill policy and would ignore all three. Declare engine='canonical' "
            "or drop the settings; a silently ignored cost policy is the defect this refusal prevents (§3.8)"
        )
    if decision_coupling not in DECISION_COUPLINGS:
        raise RunRefused(f"unknown decision_coupling {decision_coupling!r}; §3.8 declares {DECISION_COUPLINGS}")
    if decision_coupling == "account_state" and fold_portfolio != "carry":
        raise RunRefused(
            f"decision_coupling='account_state' carries one continuous account across the sample; "
            f"fold_portfolio={fold_portfolio!r} asks for a forced liquidation at each fold boundary, which the "
            "state-coupled loop does not impose. Declare fold_portfolio='carry' (§3.8)"
        )
    if decision_coupling == "account_state" and engine != CANONICAL_ENGINE:
        raise RunRefused(
            "decision_coupling='account_state' decides against marked equity and a marked position book, "
            f"which only the canonical engine carries; engine={engine!r} (§3.8)"
        )
    if decision_coupling == "account_state":
        declared_band = dict(dict(getattr(spec.construction, "options", None) or {}).get("band") or {})
        if str(declared_band.get("mode", "off")) != "off":
            raise RunRefused(
                f"{spec.path} [construction.band] declares mode={declared_band.get('mode')!r}, but the "
                "account_state path applies no no-trade band: its heuristic construction applies the "
                "participation cap only (prism.decisions._heuristic_construction). A declared band that does "
                "not act is a silent defect (N7; docs/construction_band_audit_2026-09-24.md). Declare "
                "mode = \"off\" on this path"
            )
    if spec.strategy.combination != "position_space" and decision_coupling != "account_state":
        raise StageNotAvailableError(
            f"[strategy] combination={spec.strategy.combination!r} runs the calibrated convex flow (F2), whose "
            "constructor sizes against current equity and therefore needs decision_coupling='account_state'; "
            "the targets-only loop decides every bar before the first fill and cannot evaluate the "
            "participation restriction or batch rounding at all (§3.7, §3.8)",
            stage=4,
        )
    if not spec.components:
        raise RunRefused(f"{spec.path} declares no components")


def _resolve(
    spec: Specification,
    *,
    workspace: Workspace,
    dataset: Dataset | None,
    fold_table: FoldTable | None,
    splitter: RollingWalkForward | None,
    fold_table_id: str | None,
    engine: str,
    capital: float,
    fold_portfolio: str,
    cost_refresh: str,
    synthetic: bool | None,
    start: str | None,
    end: str | None,
    visibility_mode: str,
    decision_time: str | None,
    canonical: "CanonicalSettings | None",
    price_basis: str | None,
    decision_coupling: str = "targets_only",
) -> _Resolved:
    """Resolve the dataset, the universe, the fold table and ``config_id``.

    Everything here is identity: a failure is recorded as a ``failed`` attempt
    rather than swallowed, and nothing here has written a row yet.
    """
    ds = dataset if dataset is not None else Dataset.open(workspace.data)
    vintage = ds.manifest.vintage
    if visibility_mode == "research_vintage" and vintage is None:
        raise EvaluationError(f"the dataset at {ds.root} carries no vintage label; open it with a frozen manifest")
    visibility = _Visibility(
        mode=visibility_mode,
        vintage=str(vintage) if visibility_mode == "research_vintage" else None,
        decision_time=decision_time,
    )
    span_start, span_end = _span(ds, start, end)
    grid = sessions.sessions_between(_as_date(span_start), _as_date(span_end))
    if len(grid) < 2:
        raise EvaluationError(f"the span {span_start}..{span_end} holds {len(grid)} session(s); a fill needs two")
    view = visibility.view(ds, grid[-1].isoformat())
    universe = tuple(_universe(spec, ds, visibility, grid))
    opens = view.panel(list(universe), grid[-1].isoformat(), "open")
    if len(opens) < 2:
        raise EvaluationError(
            f"the open panel over {span_start}..{span_end} holds {len(opens)} bar(s); a next-open fill needs two"
        )
    table = resolve_fold_geometry(
        fold_table=fold_table,
        splitter=splitter,
        index=opens.index,
        table_id=fold_table_id,
        source=str(spec.path),
    )
    folds = table.positions(opens.index)
    basis = _price_basis(
        ds, price_basis if price_basis is not None else (canonical.price_basis if canonical else None)
    )
    context = {
        # §3.12's evaluation_context. From stage 3b visibility_mode and vintage
        # are the mode and label every view in this run was cut under, not the
        # placeholder stage 2 had to write: the data layer enforces them.
        "visibility_mode": visibility.mode,
        "vintage": visibility.vintage if visibility.vintage is not None else UNAVAILABLE,
        "data_id": ds.manifest.id,
        "fold_table_sha256": table.content_hash,
        "fold_portfolio": fold_portfolio,
        "accounting_engine": engine,
        "holding_process": ENGINE_HOLDING_PROCESS[engine],
        "executable_return": ENGINE_EXECUTABLE_RETURN[engine],
        "capital": float(capital),
        "cost_refresh": cost_refresh,
        "price_basis": basis,
        # Whether the decision saw the account is part of what was run: the same
        # specification decided before the first fill and decided against marked equity
        # are two different evaluations, and §3.7's participation and rounding steps only
        # exist in the second (§3.8).
        "decision_coupling": decision_coupling,
        # A canonical run's charged policy is part of what it is. The
        # reproduction engine has no term for most of it and carries a null,
        # so the two engines never collide on one identity for a second reason.
        "canonical": canonical.identity() if canonical is not None else None,
    }
    declared_synthetic = (ds.manifest.provenance == "synthetic") if synthetic is None else bool(synthetic)
    return _Resolved(
        dataset=ds,
        view=view,
        visibility=visibility,
        vintage=str(vintage) if vintage is not None else UNAVAILABLE,
        universe=universe,
        session_index=opens.index,
        session_labels=tuple(_iso(stamp) for stamp in opens.index),
        opens=opens,
        fold_table=table,
        folds=folds,
        config_id=spec.config_id(evaluation_context=context),
        data_id=str(ds.manifest.id),
        evaluation_context=context,
        synthetic=declared_synthetic,
        price_basis=basis,
    )


#: What ``price_basis`` records when the manifest's sources do not agree on one:
#: the basis is a property of each bar series and is read from it, which is what
#: §3.2 requires and what ``io/events.py::series_price_basis`` implements. The
#: literal is stable, so it can enter ``config_id`` without naming sids.
PER_SERIES_BASIS = "per_series"


def _price_basis(dataset: Dataset, declared: str | None) -> str:
    """The price basis the run's bars are on (decision D3, §3.2).

    A split moves shares on a ``raw`` series and does not on a
    ``split_adjusted_at_ingest`` one, so this is an accounting input and never a
    guess. It is the caller's declaration when there is one, the manifest's when
    its sources agree, and :data:`PER_SERIES_BASIS` when they do not — in which
    case the canonical engine reads the basis off each series that carries an
    event, one series at a time, rather than taking a majority vote.
    """
    if declared is not None:
        return require_price_basis(str(declared))
    bases = {str(entry.get("price_basis")) for entry in (dataset.manifest.sources or {}).values()}
    bases.discard("None")
    if len(bases) == 1:
        return require_price_basis(bases.pop())
    return PER_SERIES_BASIS


def _price_basis_for(resolved: _Resolved, sids: Sequence[str]) -> str | dict[str, str]:
    """The basis the simulator is given: one string, or one per event-carrying series.

    Only a series an event lands on needs a basis, so only those are read. A sid
    whose series mixes two bases is refused by ``series_price_basis`` rather than
    resolved by majority, and a sid an event names but the mapping does not cover
    is refused by ``accounting.apply_events`` — neither is ever defaulted (D3).
    """
    if resolved.price_basis != PER_SERIES_BASIS:
        return resolved.price_basis
    end = resolved.session_labels[-1]
    out: dict[str, str] = {}
    for sid in dict.fromkeys(str(s) for s in sids):
        out[sid] = series_price_basis(resolved.view.bars([sid], end))
    return out


#: Event types that make a sid unrankable for a decision (§3.8). The window
#: rule is the operating provider's: effective after the earliest score
#: endpoint and on or before the decision bar.
SPINOFF_EVENT_TYPES: tuple[str, ...] = ("spinoff",)

#: What ``executable_return`` names on a ``historical_constant_weight`` record:
#: the decision at close *t* fills at open *t+1* and earns open-to-open. The
#: canonical engine measures a different thing on a different book and is keyed
#: by :data:`prism.validation.trials.ENGINE_EXECUTABLE_RETURN`; this name is
#: kept because the reproduction path and its tests are written against it.
EXECUTABLE_RETURN = ENGINE_EXECUTABLE_RETURN[REPRODUCTION_ENGINE]


# ==========================================================================
# The heuristic constructor in evaluation (§3.7)
# ==========================================================================

#: ``[construction.band] mode``. ``off`` is no hysteresis; ``fixed`` is the
#: declared scalar half-width; ``closed_form`` is the Martin (2012) cube-root
#: band, sized per name inside each fold's formation window and never on a test
#: bar (`prism/portfolio/construct.py:146`).
BAND_MODES: tuple[str, ...] = ("off", "fixed", "closed_form")

#: ``[construction] step_order``, the two orders of §3.7. The rank-to-weight
#: rule always caps first; the names say whether the account row is re-capped
#: after the band, which it must be when a per-name band blends a held and a
#: fresh weight and the blend can exceed ``max_gross``.
STEP_ORDERS: tuple[str, ...] = ("construct_cap_band_participation", "construct_band_cap_participation")

#: The trailing dollar-volume statistics ``[construction.participation]`` and
#: ``[construction.spread]`` may name. Both recorded configurations use the
#: median, which is also what the eligibility screen uses.
ADV_STATISTICS: tuple[str, ...] = ("median", "mean")

#: ``[construction.spread] mode``: the flat scalar of the execution stack, or
#: the liquidity-bucket schedule of §3.6 priced from a formation window.
SPREAD_MODES: tuple[str, ...] = ("flat", "bucket")

#: What the participation bound is a fraction of. The reproduction engine has
#: no marked equity, so ``initial_capital`` is the only base it can honor,
#: which is what ``dollar_volume.participation.base`` means in that mode (§3.6).
PARTICIPATION_BASES: tuple[str, ...] = ("initial_capital",)


class ConstructionError(EvaluationError):
    """The ``[construction]`` block asks for something §3.7 does not define."""


@dataclass(frozen=True)
class _Construction:
    """The ``[construction]`` block, resolved (§3.7).

    Every field is declared in the specification and enters ``config_id``
    through ``ConstructionSpec.options``; nothing here is defaulted silently
    into a behaviour the file did not ask for. With ``band.mode = "off"`` and
    no participation bound this is the identity, and the evaluator emits the
    rank-to-weight rule's row unchanged — the stage 1 behaviour.
    """

    step_order: str
    max_gross: float
    max_symbol_abs_weight: float
    band_mode: str
    band_width: float
    gamma_risk: float
    max_participation: float | None
    adv_window_bars: int
    adv_statistic: str
    adv_floor_dollars: float
    participation_base: str
    spread_mode: str

    @property
    def active(self) -> bool:
        """Whether the constructor changes the rank-to-weight rule's row at all."""
        return self.band_mode != "off" or self.max_participation is not None

    @property
    def needs_dollar_volume(self) -> bool:
        return self.max_participation is not None

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "heuristic",
            "step_order": self.step_order,
            "max_gross": float(self.max_gross),
            "max_symbol_abs_weight": float(self.max_symbol_abs_weight),
            "band": {
                "mode": self.band_mode,
                "width": float(self.band_width),
                "gamma_risk": float(self.gamma_risk),
                "estimated_on": "fold_formation_window" if self.band_mode == "closed_form" else "declared",
            },
            "participation": (
                None
                if self.max_participation is None
                else {
                    "max_participation": float(self.max_participation),
                    "statistic": self.adv_statistic,
                    "window_bars": int(self.adv_window_bars),
                    "floor_dollars": float(self.adv_floor_dollars),
                    "base": self.participation_base,
                }
            ),
            "spread": {"mode": self.spread_mode},
        }

    @classmethod
    def from_spec(cls, spec: Specification) -> "_Construction":
        options = dict(spec.construction.options)
        step_order = str(options.get("step_order", STEP_ORDERS[0]))
        if step_order not in STEP_ORDERS:
            raise ConstructionError(
                f"{spec.path} [construction]: unknown step_order {step_order!r}; §3.7 declares {STEP_ORDERS}"
            )
        band = dict(options.get("band") or {})
        band_mode = str(band.get("mode", "off"))
        if band_mode not in BAND_MODES:
            raise ConstructionError(
                f"{spec.path} [construction.band]: unknown mode {band_mode!r}; §3.7 declares {BAND_MODES}"
            )
        width = float(band.get("width", 0.0))
        if width < 0.0:
            raise ConstructionError(f"{spec.path} [construction.band]: width must be >= 0, got {width}")
        if band_mode == "fixed" and width == 0.0:
            raise ConstructionError(
                f"{spec.path} [construction.band]: mode='fixed' with width 0.0 is band='off' written twice; "
                "declare the mode the run actually uses"
            )
        gamma = float(band.get("gamma_risk", 1.0))
        if not gamma > 0.0:
            raise ConstructionError(f"{spec.path} [construction.band]: gamma_risk must be > 0, got {gamma}")

        gate = dict(options.get("participation") or {})
        max_participation = gate.get("max_participation")
        if max_participation is not None:
            max_participation = float(max_participation)
            if not max_participation > 0.0:
                raise ConstructionError(
                    f"{spec.path} [construction.participation]: max_participation must be > 0 when declared, got "
                    f"{max_participation}; omit the block to run without the bound"
                )
        statistic = str(gate.get("statistic", "median"))
        if statistic not in ADV_STATISTICS:
            raise ConstructionError(
                f"{spec.path} [construction.participation]: unknown statistic {statistic!r}; "
                f"§3.6 declares {ADV_STATISTICS}"
            )
        base = str(gate.get("base", PARTICIPATION_BASES[0]))
        if base not in PARTICIPATION_BASES:
            raise ConstructionError(
                f"{spec.path} [construction.participation]: base {base!r} needs the marked equity of the canonical "
                f"engine, which arrives at stage 3; the reproduction engine honors {PARTICIPATION_BASES}"
            )
        spread = dict(options.get("spread") or {})
        spread_mode = str(spread.get("mode", "flat"))
        if spread_mode not in SPREAD_MODES:
            raise ConstructionError(
                f"{spec.path} [construction.spread]: unknown mode {spread_mode!r}; §3.6 declares {SPREAD_MODES}"
            )
        caps = _rule_caps(spec)
        return cls(
            step_order=step_order,
            max_gross=float(options.get("max_gross", caps[0])),
            max_symbol_abs_weight=float(options.get("max_symbol_abs_weight", caps[1])),
            band_mode=band_mode,
            band_width=width,
            gamma_risk=gamma,
            max_participation=max_participation,
            adv_window_bars=int(gate.get("window_bars", spread.get("window_bars", 20))),
            adv_statistic=statistic,
            adv_floor_dollars=float(gate.get("floor_dollars", 0.0)),
            participation_base=base,
            spread_mode=spread_mode,
        )


def _rule_caps(spec: Specification) -> tuple[float, float]:
    """The account caps the rank-to-weight rules already declare, as the re-cap default.

    A ``[construction]`` block that re-caps after the band must re-cap at the
    same limits the rule capped at, or the second cap silently becomes a
    different risk limit from the first.
    """
    gross = 0.0
    per_name = 0.0
    for loaded in spec.components:
        params = dict(loaded.spec.to_weights_params or {})
        gross = max(gross, float(params.get("max_gross", 1.0)))
        per_name = max(per_name, float(params.get("max_symbol_abs_weight", 1.0)))
    return (gross or 1.0), (per_name or 1.0)


@dataclass(frozen=True)
class _FoldCostInputs:
    """The cost inputs one fold's formation window produced (§3.8, "Cost inputs are per fold")."""

    fold: int
    window: tuple[str, str]
    spread_bps: pd.Series | None
    band: float | pd.Series

    def to_record(self) -> dict[str, object]:
        summary: dict[str, object] = {
            "fold": int(self.fold),
            "start": self.window[0],
            "end": self.window[1],
        }
        if self.spread_bps is None:
            summary["spread_bps"] = None
        else:
            counts = self.spread_bps.value_counts().sort_index()
            summary["spread_bps"] = {
                "n_names": int(self.spread_bps.notna().sum()),
                "min": float(self.spread_bps.min()),
                "max": float(self.spread_bps.max()),
                "mean": float(self.spread_bps.mean()),
                "buckets": {f"{float(bps):g}": int(n) for bps, n in counts.items()},
            }
        if isinstance(self.band, pd.Series):
            positive = self.band[self.band > 0.0]
            summary["band"] = {
                "kind": "per_name",
                "n_positive": int(len(positive)),
                "median": float(positive.median()) if len(positive) else 0.0,
                "max": float(self.band.max()) if len(self.band) else 0.0,
            }
        else:
            summary["band"] = {"kind": "scalar", "width": float(self.band)}
        return summary


def _bucket_spread(dollar_volume: pd.DataFrame, statistic: str) -> pd.Series:
    """Per-name spread in bps from a formation window's dollar volume (§3.6).

    The window statistic is taken with ``skipna=True`` and an all-NaN name
    lands in the widest step, which is ``bucket_spread_bps``'s
    ``unknown_liquidity`` rule: unknown liquidity is never cheap.
    """
    per_name = dollar_volume.median(axis=0, skipna=True) if statistic == "median" else dollar_volume.mean(axis=0)
    return bucket_spread_bps(per_name)


def _trailing_dollar_volume(close: pd.DataFrame, volume: pd.DataFrame, *, window: int, statistic: str) -> pd.Series:
    """The trailing dollar-volume statistic at the panel's last bar, full-window only.

    ``min_periods = window`` deliberately: a name without a full window has no
    statistic, and the participation bound refuses an order on it rather than
    sizing one against a partial estimate (§3.6). The last bar's value depends only on
    the last ``window`` rows, so only those are rolled: it is called on the full history
    at every decision.
    """
    dollar = (close * volume).iloc[-int(window):]
    rolling = dollar.rolling(int(window), min_periods=int(window))
    frame = rolling.median() if statistic == "median" else rolling.mean()
    return frame.iloc[-1]


def _apply_construction(
    raw: pd.Series,
    held: pd.Series,
    construction: _Construction,
    *,
    band: float | pd.Series,
    dollar_volume: pd.Series | None,
    capital: float,
    liquidate: Sequence[str] = (),
) -> pd.Series:
    """One decision through the band, the re-cap and the participation bound (§3.7).

    The order is the recorded driver's
    (`research/arbitrage/residual_walk_forward.py:239-257`), duplicated with
    this citation rather than imported: the band steps the *held* book toward
    the fresh target, the caps are re-applied because a per-name blend can
    exceed them, the bound shrinks whatever trade survives, and the row that
    comes out is both the emitted target and tomorrow's held state — so
    unfilled demand has to clear the band again rather than accumulating
    invisibly. A sid under a ``liquidate`` instruction skips the band, because
    the flat is the decision and not a target to be damped (§3.3).
    """
    stepped = step_no_trade_band(held, raw, band)
    exempt = [sid for sid in dict.fromkeys(liquidate) if sid in raw.index]
    if exempt:
        stepped.loc[exempt] = raw.loc[exempt].to_numpy(dtype=float)
    if construction.step_order == "construct_band_cap_participation":
        stepped = cap_book(
            stepped.to_frame().T,
            max_gross=construction.max_gross,
            max_symbol_abs_weight=construction.max_symbol_abs_weight,
        ).iloc[0]
    if construction.max_participation is not None:
        if dollar_volume is None:
            raise ConstructionError(
                "[construction.participation] declares a bound but no dollar-volume statistic reached the "
                "decision; a bound that cannot be evaluated is refused, never waived (N7)"
            )
        stepped = participation_capped_targets(
            held,
            stepped,
            dollar_volume,
            aum=float(capital),
            max_participation=float(construction.max_participation),
            adv_floor=float(construction.adv_floor_dollars),
        )
    return stepped.reindex(raw.index).fillna(0.0)


def _evaluate_folds(
    spec: Specification,
    resolved: _Resolved,
    *,
    workspace: Workspace,
    fold_portfolio: str,
    calibration_hook: Callable[[CalibrationRequest], Mapping[str, object]] | None,
    execution: ExecutionConfig,
    capital: float,
) -> tuple[pd.DataFrame, tuple[FoldEvaluation, ...], list[str], tuple[_FoldCostInputs, ...]]:
    """Fit each fold on its formation window and decide each of its test bars.

    Three invariants live here and nowhere else. The fit never sees a bar
    outside the fold's formation window (:func:`prepare_components` enforces the
    window; this loop supplies it and the record carries what was used). The
    fold's cost inputs and its no-trade band are estimated on that same
    formation window and on nothing else, which is why they are built here and
    recorded per fold (§3.8, "Cost inputs are per fold"). And under
    ``fold_portfolio = "reset"`` the targets of the last two decision rows of
    every fold are zero, so the liquidation fills at the following open inside
    the fold and nothing fills on its own decision bar — the recorded driver's
    mechanic (`research/arbitrage/walk_forward.py:264`), duplicated with this
    citation rather than imported, because ``prism`` imports no research module.

    The decision loop only reads a panel on a bar where some component is due
    or expiring. On any other bar the emitted row is the held book, which is
    exactly what composition returns when nothing is due — the same row, at a
    fraction of the cost, on a walk-forward that decides monthly and accounts
    daily.
    """
    labels = resolved.session_labels
    universe = list(resolved.universe)
    targets = pd.DataFrame(np.nan, index=resolved.session_index, columns=universe, dtype=float)
    construction = _Construction.from_spec(spec)
    index_of = {label: number for number, label in enumerate(labels)}
    needs_total_return = any(str(l.declarations["scoring_basis"]) == "total_return" for l in spec.components)
    history = max(int(l.declarations["required_history"]) for l in spec.components)
    warnings: list[str] = []
    evaluations: list[FoldEvaluation] = []
    cost_inputs: list[_FoldCostInputs] = []
    states: dict[str, ComponentState] = {}
    held = pd.Series(0.0, index=pd.Index(universe), dtype=float)
    previous: tuple[FittedComponent, ...] = ()
    previous_cutoff: int | None = None

    for fold in resolved.folds:
        train_start, train_end = labels[fold.formation_start], labels[fold.formation_stop - 1]
        test_start, test_end = labels[fold.test_start], labels[fold.test_stop - 1]
        fit_view = resolved.view_at(train_end)
        members = _members(spec, fit_view, universe)
        if not members:
            raise EvaluationError(
                f"fold {fold.fold}: the point-in-time universe is empty at the formation end {train_end}; an "
                "empty cross-section is a dataset defect, not an empty result (N7)"
            )
        cutoff_position = fold.formation_stop - 1
        elapsed = None if previous_cutoff is None else cutoff_position - previous_cutoff
        fitted = prepare_components(
            spec,
            fit_view,
            train_end,
            fits_dir=workspace.fits(),
            members=members,
            window=(train_start, train_end),
            previous=previous,
            elapsed_bars=elapsed,
        )
        previous, previous_cutoff = fitted, cutoff_position

        calibration: dict[str, object] | None = None
        if calibration_hook is not None:
            calibration = dict(
                calibration_hook(
                    _calibration_request(
                        spec,
                        resolved,
                        fold=fold,
                        fitted=fitted,
                        members=members,
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        needs_total_return=needs_total_return,
                    )
                )
            )
        excluded = bool(calibration.get("excluded")) if calibration else False
        declared_reason = calibration.get("exclusion_reason") if calibration else None
        exclusion_reason = str(declared_reason) if excluded and declared_reason else None
        if excluded and not exclusion_reason:
            raise EvaluationError(
                f"fold {fold.fold}: the calibration hook excluded the fold without naming a reason; an "
                "unexplained exclusion moves every per-fold denominator invisibly (§3.4)"
            )

        if fold_portfolio == "reset" or excluded:
            states = {}
            held = pd.Series(0.0, index=pd.Index(universe), dtype=float)
        plans = tuple(plan for component in fitted for plan in component.plans())
        inputs = _fold_cost_inputs(
            spec,
            resolved,
            fold=fold,
            plans=plans,
            construction=construction,
            execution=execution,
            train_start=train_start,
            train_end=train_end,
            universe=universe,
            history=history,
            needs_total_return=needs_total_return,
            warnings=warnings,
        )
        cost_inputs.append(inputs)
        decisions: list[str] = []
        boundary: list[str] = []
        if excluded:
            # An excluded fold trades nothing: it is flat from its first bar,
            # so its exclusion cannot leak into another fold's return through a
            # carried position.
            targets.iloc[fold.test_start] = 0.0
            boundary.append(test_start)
        else:
            for position in range(fold.test_start, fold.test_stop):
                day = labels[position]
                if not _due(plans, states, index_of, position, decision_bar=day):
                    # Nothing is due and nothing expires: composition would
                    # return the carried sub-targets and the constructor would
                    # step the held book toward itself. The emitted row is the
                    # held book, and no panel is read.
                    targets.iloc[position] = held.to_numpy(dtype=float)
                    decisions.append(day)
                    continue
                row, states, gate_adv = _decide_bar(
                    spec,
                    resolved,
                    day,
                    position=position,
                    universe=universe,
                    plans=plans,
                    states=states,
                    needs_total_return=needs_total_return,
                    history=history,
                    warnings=warnings,
                    construction=construction,
                )
                if row is None:
                    continue
                held = _apply_construction(
                    row,
                    held,
                    construction,
                    band=inputs.band,
                    dollar_volume=gate_adv,
                    capital=capital,
                )
                targets.iloc[position] = held.to_numpy(dtype=float)
                decisions.append(day)
            if not decisions:
                raise EvaluationError(
                    f"fold {fold.fold} ({test_start}..{test_end}) produced no decision row; every component "
                    "declined the whole cross-section on every test bar, which is a degenerate fold, not an "
                    "empty result (N7)"
                )
            if fold_portfolio == "reset":
                flat = range(max(fold.test_start, fold.test_stop - 2), fold.test_stop)
                for position in flat:
                    targets.iloc[position] = 0.0
                held = pd.Series(0.0, index=pd.Index(universe), dtype=float)
                boundary = [labels[position] for position in flat]
                if decisions:
                    boundary = sorted({decisions[0], *boundary})
        evaluations.append(
            FoldEvaluation(
                fold=int(fold.fold),
                train=(train_start, train_end),
                test=(test_start, test_end),
                n_test_bars=int(fold.test_rows),
                decisions=tuple(decisions),
                boundary_decisions=tuple(boundary),
                components=fitted,
                calibration=calibration,
                excluded=excluded,
                exclusion_reason=exclusion_reason,
                metrics={},
                turnover={},
            )
        )
    return targets, tuple(evaluations), warnings, tuple(cost_inputs)


# ==========================================================================
# The state-coupled path: the constructor decides inside the simulation (§3.8)
# ==========================================================================


#: What the state-coupled path names its coupling in the record. ``targets_only`` is the
#: stage-2 loop, which decides every bar before the first fill; ``account_state`` decides
#: at the close of each bar against the account as it then stands.
DECISION_COUPLINGS: tuple[str, ...] = ("targets_only", "account_state")


@dataclass
class _FoldPlan:
    """One fold, prepared: its fit, its calibration, its cost inputs, its test range."""

    fold: RollingFold
    train: tuple[str, str]
    test: tuple[str, str]
    fitted: tuple[FittedComponent, ...]
    plans: tuple[ComponentPlan, ...]
    members: tuple[str, ...]
    calibration: object | None
    excluded: bool
    exclusion_reason: str | None
    cost_inputs: _FoldCostInputs


def _training_positions(fold: RollingFold, *, update_every_bars: int, required_history: int) -> list[int]:
    """The formation window's decision periods, anchored at the fit cutoff.

    Anchored at the cutoff rather than the window's first bar because the cutoff is where
    the fold's decisions resume: the last training period is the one immediately before
    the first test decision, at the same cadence. A period whose score needs bars the
    sample does not have is not supplied at all -- a component that refuses a short panel
    is right to refuse it, and inventing a score for it would put a number in the training
    pool that the decision path could never have produced.
    """
    step = int(update_every_bars)
    positions: list[int] = []
    position = int(fold.formation_stop) - 1
    while position >= int(fold.formation_start):
        if position + 1 >= int(required_history):
            positions.append(position)
        position -= step
    return sorted(positions)


def _training_panels(
    resolved: _Resolved,
    *,
    fold: RollingFold,
    plan: ComponentPlan,
    members: Sequence[str],
    close: pd.DataFrame,
    volume: pd.DataFrame,
    positions: Sequence[int],
    horizon_bars: int,
    label_distributions: str,
    sigma_h: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The fold's ``(scores, returns, sigma_h)`` training panels, train-only by slicing.

    The label reads ``opens`` truncated at the formation window's last bar, so a period
    whose exit bar lies beyond the window has a non-finite label instead of one computed
    from a test bar. That truncation is the train-only boundary, and it is why the last
    period or two of every fold carry no label (§3.4).
    """
    from prism.calibration import (  # local: §3.4's label and volatility helpers
        executable_returns,
        horizon_sigma,
    )

    columns = pd.Index([str(sid) for sid in members])
    stamps = resolved.session_index[list(positions)]
    scores = pd.DataFrame(np.nan, index=stamps, columns=columns, dtype=float)
    for row, position in enumerate(positions):
        raw = plan.component.score(close.iloc[: position + 1], volume.iloc[: position + 1])
        scores.iloc[row] = pd.to_numeric(pd.Series(raw), errors="coerce").reindex(columns).to_numpy(dtype=float)
    opens = resolved.opens.reindex(columns=columns).iloc[: int(fold.formation_stop)]
    distributions = None
    if label_distributions == "credited":
        # The frozen dataset carries the cash distributions it carries. Where it carries
        # none the credited label credits zero, which is what the accounting also credits;
        # the record says so rather than letting a silent zero look like a decision.
        distributions = pd.DataFrame(0.0, index=opens.index, columns=opens.columns, dtype=float)
    returns = executable_returns(
        opens,
        list(positions),
        int(horizon_bars),
        distributions=distributions,
        label_distributions=label_distributions,
    )
    sigma = horizon_sigma(
        close.reindex(columns=columns), sigma_h, int(horizon_bars), decision_positions=list(positions)
    )
    returns.index = stamps
    sigma.index = stamps
    return scores, returns.reindex(columns=columns), sigma.reindex(columns=columns)


def _stack_bucket_spread(
    resolved: _Resolved,
    *,
    stack: CostStack,
    members: Sequence[str],
    close: pd.DataFrame,
    volume: pd.DataFrame,
    train_start: str,
    train_end: str,
    universe: Sequence[str],
) -> pd.Series | None:
    """The fold's per-name spread, bucketed from its own formation window (§3.6, §3.8).

    Driven by the *cost stack's* declared spread mode rather than the strategy's
    ``[construction.spread]`` block: the stack is what charges the spread on the canonical
    engine, and §3.6 puts the dollar-volume statistic for spread bucketing in the stack.
    A stack that declares a flat spread gets no vector here and is charged its flat value.
    """
    if stack.spread.mode != "bucket":
        return None
    index = close.index
    window = (index >= pd.Timestamp(train_start, tz=index.tz)) & (index <= pd.Timestamp(train_end, tz=index.tz))
    dollar_volume = (close.loc[window] * volume.reindex(index=index).loc[window]).reindex(columns=list(members))
    statistic = stack.dollar_volume["spread_bucket"].estimator
    per_name = (
        dollar_volume.median(axis=0, skipna=True) if statistic == "median" else dollar_volume.mean(axis=0, skipna=True)
    )
    return bucket_spread_bps(per_name, schedule=stack.spread.schedule).reindex(pd.Index(list(universe)))


def _evaluate_stateful(
    spec: Specification,
    resolved: _Resolved,
    *,
    workspace: Workspace,
    fold_portfolio: str,
    execution: ExecutionConfig,
    capital: float,
    canonical: CanonicalSettings,
    calibration_family: str | Path | None,
    cost_refresh: str,
) -> tuple[
    pd.DataFrame,
    tuple[FoldEvaluation, ...],
    list[str],
    tuple[_FoldCostInputs, ...],
    PortfolioBacktestResult,
    dict[str, object],
    dict[str, object],
]:
    """Fit each fold, then decide every bar inside one continuous simulation (§3.7, §3.8).

    Two passes and one simulation. The first pass is per fold and is train-only: it fits
    the components on the formation window, builds the calibration panels from that window
    alone, admits or excludes the fold, and buckets the fold's spread. The second pass is
    the simulation itself, which calls back at the close of each bar with the account as
    it stands; the constructor sizes against *that* equity, the participation restriction
    is measured against it, and batch rounding rounds against it.

    That coupling is the whole reason this path exists. It is also why the capital sweep
    means something: the same frozen cost stack and the same charging rules produce
    different charged dollars at different account sizes, because participation is trade
    dollars over dollar volume from current equity and rounding at the per-sid increment
    changes filled notional (§5, stage 4 exit).
    """
    from prism.decisions import (  # local: the constructor stack is the construction extra's
        ArmPolicies,
        calibrate_fold,
        decide_bar,
        forecast_at,
        snapshot_at,
    )

    labels = resolved.session_labels
    universe = list(resolved.universe)
    index_of = {label: number for number, label in enumerate(labels)}
    construction = _Construction.from_spec(spec)
    policies = ArmPolicies.from_spec(spec)
    family = policies.calibration
    if calibration_family is not None:
        from prism.calibration import load_calibration

        family = load_calibration(calibration_family)
    needs_total_return = any(str(l.declarations["scoring_basis"]) == "total_return" for l in spec.components)
    history = max(int(l.declarations["required_history"]) for l in spec.components)
    declarations = {loaded.component_id: dict(loaded.declarations) for loaded in spec.components}
    warnings: list[str] = []
    if policies.kind == "convex" and family is None:
        raise EvaluationError(
            "a convex arm needs a calibration family to decide fold admission for every arm alike (D6); "
            "declare it on the registration's [shared] block"
        )

    # ---------------------------------------------------------------- pass 1
    plans_by_fold: list[_FoldPlan] = []
    previous: tuple[FittedComponent, ...] = ()
    previous_cutoff: int | None = None
    for fold in resolved.folds:
        train_start, train_end = labels[fold.formation_start], labels[fold.formation_stop - 1]
        test_start, test_end = labels[fold.test_start], labels[fold.test_stop - 1]
        fit_view = resolved.view_at(train_end)
        members = _members(spec, fit_view, universe)
        if not members:
            raise EvaluationError(
                f"fold {fold.fold}: the point-in-time universe is empty at the formation end {train_end}; an "
                "empty cross-section is a dataset defect, not an empty result (N7)"
            )
        cutoff_position = fold.formation_stop - 1
        elapsed = None if previous_cutoff is None else cutoff_position - previous_cutoff
        fitted = prepare_components(
            spec,
            fit_view,
            train_end,
            fits_dir=workspace.fits(),
            members=members,
            window=(train_start, train_end),
            previous=previous,
            elapsed_bars=elapsed,
        )
        previous, previous_cutoff = fitted, cutoff_position
        # Fold admission calibrates each declared component once; composition
        # runs its tranches, which share the component's fitted object.
        declared_plans = tuple(component.plan(combination=spec.strategy.combination) for component in fitted)
        plans = tuple(
            plan for component in fitted for plan in component.plans(combination=spec.strategy.combination)
        )

        close = fit_view.panel(list(members), train_end, "close").reindex(
            index=resolved.session_index[: fold.formation_stop]
        )
        volume = fit_view.panel(list(members), train_end, "volume").reindex(index=close.index, columns=close.columns)
        calibration = None
        excluded = False
        reason: str | None = None
        # Fold admission scores every component over the formation window
        # below; a component that trains on past cross-sections needs the
        # point-in-time membership mask here as much as at a decision bar. The
        # view is the fit cutoff's, so no row after the cutoff is bound.
        _bind_membership(spec, fit_view, plans, close)
        if family is not None:
            fits: dict[str, object] = {}
            records: dict[str, object] = {}
            for plan in declared_plans:
                declared = declarations[plan.component_id]
                positions = _training_positions(
                    fold,
                    update_every_bars=int(plan.update_every_bars),
                    required_history=int(declared["required_history"]),
                )
                if not positions:
                    raise EvaluationError(
                        f"fold {fold.fold}: no formation-window decision period carries the "
                        f"{declared['required_history']} bars component {plan.component_id!r} declares it needs; the "
                        "fold cannot be calibrated and the geometry, not this loop, is what says so"
                    )
                scores, returns, sigma = _training_panels(
                    resolved,
                    fold=fold,
                    plan=plan,
                    members=members,
                    close=close,
                    volume=volume,
                    positions=positions,
                    horizon_bars=int(declared["horizon_bars"]),
                    label_distributions=family.label_distributions,
                    sigma_h=family.sigma_h,
                )
                fold_calibration = calibrate_fold(
                    fold=int(fold.fold),
                    component_id=plan.component_id,
                    scores=scores,
                    returns=returns,
                    sigma_h=sigma,
                    family=family,
                    arm_spec=policies.calibration if policies.kind == "convex" else None,
                    fit_cutoff=train_end,
                    horizon_bars=int(declared["horizon_bars"]),
                    update_every_bars=int(plan.update_every_bars),
                )
                records[plan.component_id] = fold_calibration.to_record()
                if fold_calibration.excluded:
                    excluded = True
                    reason = fold_calibration.exclusion_reason
                for cid, fit in fold_calibration.fits.items():
                    fits[cid] = fit
            calibration = {"components": records, "fits": fits}

        spread = _stack_bucket_spread(
            resolved,
            stack=canonical.cost_stack,
            members=members,
            close=close,
            volume=volume,
            train_start=train_start,
            train_end=train_end,
            universe=universe,
        )
        plans_by_fold.append(
            _FoldPlan(
                fold=fold,
                train=(train_start, train_end),
                test=(test_start, test_end),
                fitted=fitted,
                plans=plans,
                members=tuple(members),
                calibration=calibration,
                excluded=excluded,
                exclusion_reason=reason,
                cost_inputs=_FoldCostInputs(
                    fold=int(fold.fold), window=(train_start, train_end), spread_bps=spread, band=0.0
                ),
            )
        )

    declared_refresh = canonical.cost_stack.dollar_volume["spread_bucket"].refresh
    if declared_refresh != cost_refresh:
        warnings.append(
            f"the cost stack declares dollar_volume.spread_bucket.refresh={declared_refresh!r} and this run was "
            f"asked for cost_refresh={cost_refresh!r}; the run's cadence is what was applied, and each fold's "
            "spread was bucketed from that fold's own formation window (§3.8)"
        )
    cost_inputs = tuple(plan.cost_inputs for plan in plans_by_fold)
    shells = tuple(
        FoldEvaluation(
            fold=int(plan.fold.fold),
            train=plan.train,
            test=plan.test,
            n_test_bars=int(plan.fold.test_rows),
            decisions=(),
            boundary_decisions=(),
            components=plan.fitted,
            calibration=None,
            excluded=plan.excluded,
            exclusion_reason=plan.exclusion_reason,
            metrics={},
            turnover={},
        )
        for plan in plans_by_fold
    )
    spread_bps_per_name = _charged_spread(resolved, folds=shells, cost_inputs=cost_inputs, cost_refresh=cost_refresh)

    # ---------------------------------------------------------------- pass 2
    end = labels[-1]
    closes = resolved.view.panel(universe, end, "close").reindex(index=resolved.session_index)
    volumes = resolved.view.panel(universe, end, "volume").reindex(index=resolved.session_index)
    events = resolved.view.events(universe, labels[0], end)
    charged = _canonical_cost_model(resolved, canonical, spread_bps_per_name, capital=capital)

    fold_at: dict[int, _FoldPlan] = {}
    for plan in plans_by_fold:
        for position in range(plan.fold.test_start, plan.fold.test_stop):
            fold_at[position] = plan

    targets = pd.DataFrame(np.nan, index=resolved.session_index, columns=universe, dtype=float)
    states: dict[str, ComponentState] = {}
    decisions_by_fold: dict[int, list[str]] = {plan.fold.fold: [] for plan in plans_by_fold}
    bar_records: list[dict[str, object]] = []
    flattened: set[int] = set()
    horizon = max(int(l.declarations["horizon_bars"]) for l in spec.components)

    def hook(call: DecisionCall) -> pd.Series | None:
        nonlocal states
        position = index_of.get(call.bar)
        if position is None:
            return None
        plan = fold_at.get(position)
        if plan is None:
            return None
        if plan.excluded:
            # An excluded fold trades nothing: it flattens on its first test bar so its
            # exclusion cannot leak into another fold's return through a carried position.
            if plan.fold.fold in flattened:
                return None
            flattened.add(plan.fold.fold)
            states = {}
            held = {sid: 0.0 for sid, held_position in call.state.positions.items() if held_position.shares}
            if not held:
                return None
            targets.iloc[position] = 0.0
            return pd.Series(held, dtype=float)
        if not _due(plan.plans, states, index_of, position, decision_bar=call.bar):
            return None
        scored = _score_bar(
            spec,
            resolved,
            call.bar,
            position=position,
            universe=universe,
            plans=plan.plans,
            states=states,
            needs_total_return=needs_total_return,
            history=history,
            warnings=warnings,
            construction=construction,
        )
        if scored is None:
            return None
        states = dict(scored.composition.states)
        outcome = _construct_at_bar(
            call,
            spec=spec,
            resolved=resolved,
            plan=plan,
            scored=scored,
            policies=policies,
            charged=charged,
            horizon_bars=horizon,
            universe=universe,
            forecast_at=forecast_at,
            snapshot_at=snapshot_at,
            decide_bar=decide_bar,
        )
        bar_records.append(outcome.record)
        if outcome.record.get("deferred_unpriced_entries"):
            warnings.append(
                f"{call.bar}: entries into {outcome.record['deferred_unpriced_entries']} deferred; a tranche carries "
                "a target for them, the account does not hold them, and they have no price on this bar"
            )
        decisions_by_fold[plan.fold.fold].append(call.bar)
        if outcome.construction is not None:
            row = pd.Series(outcome.construction.w_target, dtype=float).reindex(pd.Index(universe))
            targets.iloc[position] = row.to_numpy(dtype=float)
        return outcome.row

    simulation = simulate(
        _by_bar(resolved.opens),
        _by_bar(closes),
        cost_model=charged,
        volumes=_by_bar(volumes),
        events=events,
        price_basis=_price_basis_for(resolved, [] if events.empty else events["sid"].astype(str).tolist()),
        initial_cash=float(capital),
        fills=canonical.fills,
        increment=float(canonical.share_increment),
        min_notional=float(canonical.min_notional),
        order_type="market_on_open",
        zero_crossing_clamp=bool(canonical.zero_crossing_clamp),
        bars=[_iso(stamp) for stamp in resolved.session_index],
        decide_hook=hook,
    )
    if simulation.state.is_incomplete:
        warnings.append(
            "the canonical book ends with an unresolved claim on at least one name; the record is complete only "
            "when every terminal outcome is known (§3.2, §3.6)"
        )
    for plan in plans_by_fold:
        if not plan.excluded and not decisions_by_fold[plan.fold.fold]:
            raise EvaluationError(
                f"fold {plan.fold.fold} ({plan.test[0]}..{plan.test[1]}) produced no decision row; every "
                "component declined the whole cross-section on every test bar, which is a degenerate fold, "
                "not an empty result (N7)"
            )

    folds = tuple(
        FoldEvaluation(
            fold=int(plan.fold.fold),
            train=plan.train,
            test=plan.test,
            n_test_bars=int(plan.fold.test_rows),
            decisions=tuple(decisions_by_fold[plan.fold.fold]),
            boundary_decisions=(),
            components=plan.fitted,
            calibration=None if plan.calibration is None else dict(plan.calibration["components"]),
            excluded=plan.excluded,
            exclusion_reason=plan.exclusion_reason,
            metrics={},
            turnover={},
        )
        for plan in plans_by_fold
    )
    stale = int(simulation.ledger["stale"].max()) if len(simulation.ledger) else 0
    detail: dict[str, object] = {
        "note": COUPLED_NOTE,
        "decision_coupling": "account_state",
        "n_decision_bars": sum(len(rows) for rows in decisions_by_fold.values()),
        "n_target_rows": int(targets.notna().any(axis=1).sum()),
        "n_fills": int(len(simulation.fills)),
        "n_orders": int(len(simulation.orders)),
        "n_events_applied": int(len(simulation.events)),
        "max_stale_marks": stale,
        "pending_orders_at_end": int(len(simulation.state.pending_orders)),
        "final_cash": float(simulation.state.cash),
        "settings": canonical.to_record(),
    }
    result = _canonical_backtest_result(
        simulation, index=resolved.session_index, universe=resolved.universe, targets=targets
    )
    construction_record = _construction_record(bar_records, policies=policies, plans=plans_by_fold)
    return targets, folds, warnings, cost_inputs, result, detail, construction_record


def _construct_at_bar(
    call: DecisionCall,
    *,
    spec: Specification,
    resolved: _Resolved,
    plan: _FoldPlan,
    scored: "_BarScores",
    policies: object,
    charged: CostModel,
    horizon_bars: int,
    universe: Sequence[str],
    forecast_at: object,
    snapshot_at: object,
    decide_bar: object,
) -> object:
    """Assemble one decision bar's constructor inputs and construct it.

    The universe the constructor solves over is the scored cross-section plus every held
    name, never the scored cross-section alone: a held name that has left the tradeable
    set must stay in the program under the reduce-only restriction, because dropping it
    would leave the account holding a position no constraint covers (§3.7).
    """
    prices = scored.close.iloc[-1] if len(scored.close) else pd.Series(dtype=float)
    equity = float(call.equity)
    held = [str(sid) for sid, position in call.state.positions.items() if position.shares]
    stack = charged.stack
    participation = stack.dollar_volume["participation"] if stack is not None else None
    adv = _trailing_dollar_volume(
        scored.close,
        scored.volume,
        window=int(participation.window_bars) if participation is not None else 20,
        statistic=str(participation.estimator) if participation is not None else "median",
    )
    costs = _constructor_costs(charged, bar=call.bar, equity=equity, horizon_bars=int(horizon_bars))

    forecasts: list[object] = []
    heuristic_target: pd.Series | None = None
    component_outputs: dict[str, object] = {}
    if policies.kind == "convex":
        fits = dict((plan.calibration or {}).get("fits") or {})
        for output in scored.composition.component_outputs:
            fit = fits.get(output.component_id)
            if fit is None or output.raw is None:
                continue
            forecast = forecast_at(
                component_id=output.component_id,
                fit=fit,
                raw=output.raw,
                prices=scored.close,
                spec=policies.calibration,
                horizon_bars=int(horizon_bars),
            )
            forecasts.append(forecast)
            component_outputs[output.component_id] = {
                "n_scored": int(output.raw.notna().sum()),
                "n_forecast": int(len(forecast.z)),
                "due": bool(output.due),
                "expired": bool(output.expired),
                "disabled_reason": output.disabled_reason,
            }
        scored_sids = [str(sid) for forecast in forecasts for sid in forecast.z.index]
    else:
        heuristic_target = scored.row
        scored_sids = list(universe)
        for output in scored.composition.component_outputs:
            component_outputs[output.component_id] = {
                "n_scored": 0 if output.raw is None else int(output.raw.notna().sum()),
                "due": bool(output.due),
                "expired": bool(output.expired),
                "disabled_reason": output.disabled_reason,
            }

    sids = list(dict.fromkeys([*scored_sids, *held]))
    if not sids:
        raise EvaluationError(f"{call.bar}: the constructor universe is empty and the account holds nothing")
    index = pd.Index(sids)
    shares = pd.Series({sid: float(call.state.shares(sid)) for sid in sids}, dtype=float)
    marks = _decision_marks(prices, index, state=call.state, bar=call.bar)
    w0 = shares * marks / equity

    risk = None
    if policies.risk_policy is not None:
        window = int(policies.risk_policy.covariance_window_bars)
        returns = scored.close.reindex(columns=index).pct_change().iloc[1:].tail(window)
        risk = snapshot_at(
            returns=returns,
            policy=policies.risk_policy,
            decision_bar=call.bar,
            universe=sids,
            held=held,
            eligible={sid: bool(sid in set(scored_sids)) for sid in sids},
            valuation_status={
                sid: str(position.valuation_status)
                for sid, position in call.state.positions.items()
                if position.shares
            },
            halted=[sid for sid in sids if sid not in call.traded],
        )

    target = None if heuristic_target is None else heuristic_target.reindex(index)
    deferred: list[str] = []
    if target is not None and any(component.phase_offset_bars is not None for component in plan.plans):
        target, deferred = _defer_unpriced_entries(target, marks=marks, shares=shares)
    outcome = decide_bar(
        decision_bar=call.bar,
        policies=policies,
        state=call.state,
        equity=equity,
        prices=marks,
        sids=sids,
        w0=w0,
        positions=shares,
        adv_dollars=adv.reindex(index),
        costs=costs,
        heuristic_target=target,
        forecasts=tuple(forecasts),
        risk=risk,
        component_outputs=component_outputs,
        fold=int(plan.fold.fold),
    )
    if deferred:
        outcome.record["deferred_unpriced_entries"] = deferred
    return outcome


def _defer_unpriced_entries(
    target: pd.Series, *, marks: pd.Series, shares: pd.Series
) -> tuple[pd.Series, list[str]]:
    """Withhold the target of a name the account does not hold and cannot price on this bar.

    Only a book on the session grid reaches this. Its tranches carry their sub-targets
    between their own refreshes, and the account decides on every session one of them
    refreshes, so the account target can name a stock no tranche is refreshing that has
    no price on this bar: typically an entry ordered on the previous bar that did not
    fill because the stock stopped trading. Rounding refuses to size a non-flat target
    without a price (N7), and with nothing held there is nothing to keep or exit, so the
    name carries no decision on this bar -- the heuristic construction reads that as the
    held weight, zero -- and the entry waits for a priced bar. The deferred names are
    returned so the bar record and the run's warnings carry them.
    """
    values = target.to_numpy(dtype=float)
    unpriced = marks.reindex(target.index).to_numpy(dtype=float) <= 0.0
    unheld = shares.reindex(target.index).to_numpy(dtype=float) == 0.0
    mask = unpriced & unheld & np.isfinite(values) & (values != 0.0)
    if not mask.any():
        return target, []
    deferred = [str(sid) for sid in target.index[mask]]
    return target.where(~mask), deferred


def _decision_marks(prices: pd.Series, index: pd.Index, *, state: object, bar: str) -> pd.Series:
    """The price every name in the constructor universe is marked at, or a refusal.

    A scored name printed a close on this bar by construction. A *held* name that did not
    print takes the mark the account carries for it, which is the mark the simulator's own
    equity uses -- so the constructor's ``w0`` and the equity it is a fraction of are one
    valuation. A held name with no price at all is refused rather than imputed to flat:
    reading it as flat would let the program open a position in a name nobody could value,
    and §3.7 says a name with no marked weight is flat only if the caller says so.
    """
    marks = pd.Series(prices, dtype=float).reindex(index)
    values = marks.to_numpy(dtype=float)
    unpriced = ~np.isfinite(values) | (values <= 0.0)
    if unpriced.any():
        carried = {}
        for sid, position in getattr(state, "positions", {}).items():
            price = getattr(position, "last_price", None)
            if price is not None and math.isfinite(float(price)) and float(price) > 0.0:
                carried[str(sid)] = float(price)
        marks = marks.where(~unpriced, pd.Series(carried, dtype=float).reindex(index))
    values = marks.to_numpy(dtype=float)
    still = ~np.isfinite(values) | (values <= 0.0)
    held = (
        np.abs(
            pd.Series(
                {str(sid): float(getattr(pos, "shares", 0.0)) for sid, pos in getattr(state, "positions", {}).items()},
                dtype=float,
            )
            .reindex(index)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        > 0.0
    )
    offenders = [str(sid) for sid, bad, is_held in zip(index, still, held) if bad and is_held]
    if offenders:
        raise EvaluationError(
            f"{bar}: {offenders[:5]} are held and carry no usable mark, on the bar or in the account. A held "
            "name with no valuation is a refusal, never a zero weight: reading it as flat would let the "
            "constructor open a position in a name nobody could value (§3.7)"
        )
    return marks.where(~still, 0.0)


def _constructor_costs(charged: CostModel, *, bar: str, equity: float, horizon_bars: int) -> CostModel:
    """The charged policy as the constructor's cost functions, on this bar's spread.

    The accounting model carries a bar-by-name spread frame under ``cost_refresh = "fold"``
    and ``convex_terms`` states a cost function per name, not per name-day, so the row for
    this bar is bound here. The stack, the schedule and the fallback are the charged
    model's own: the constructor optimizes against the policy the account is charged
    under, which is the only way the modeled and charged costs are the same policy (§3.6).
    """
    spread = None if charged.inputs is None else charged.inputs.spread_bps
    if isinstance(spread, pd.DataFrame):
        spread = spread.loc[bar] if bar in spread.index else None
    return CostModel(
        charged.stack,
        CostInputs(as_of=bar, scope="run", spread_bps=spread),
        spread=charged.spread,
        equity=float(equity),
        horizon_bars=int(horizon_bars),
    )


def _construction_record(
    bar_records: Sequence[Mapping[str, object]],
    *,
    policies: object,
    plans: Sequence[_FoldPlan],
) -> dict[str, object]:
    """The construction block of the record: per bar, and the summaries §3.7 names.

    The medians are over decision dates, which is the unit the acceptance check of §3.7 is
    stated in: "the comparison records the median across decision dates for each arm".
    """
    at_cap = [
        float(row["diagnostics"]["at_cap_fraction"])
        for row in bar_records
        if isinstance(row.get("diagnostics"), Mapping) and row["diagnostics"].get("at_cap_fraction") is not None
    ]
    ex_ante = [
        float(row["diagnostics"]["ex_ante_sd"])
        for row in bar_records
        if isinstance(row.get("diagnostics"), Mapping) and row["diagnostics"].get("ex_ante_sd") is not None
    ]
    responses: dict[str, int] = {}
    for row in bar_records:
        repair = row.get("repair") or {}
        response = str(repair.get("response", "unavailable"))
        responses[response] = responses.get(response, 0) + 1
    paths: dict[str, int] = {}
    for row in bar_records:
        paths[str(row.get("path"))] = paths.get(str(row.get("path")), 0) + 1
    flat = sum(
        1
        for row in bar_records
        if isinstance(row.get("diagnostics"), Mapping)
        and bool(row["diagnostics"].get("flat_from_straddling_endpoints"))
    )
    straddling = sum(
        1
        for row in bar_records
        if isinstance(row.get("diagnostics"), Mapping) and row["diagnostics"].get("straddling_components")
    )
    return {
        "kind": policies.kind,
        "decision_coupling": "account_state",
        "policies": policies.declaration(),
        "n_decisions": len(bar_records),
        "paths": paths,
        "repair": responses,
        "at_cap_fraction": {
            "median": float(np.median(at_cap)) if at_cap else None,
            "max": float(np.max(at_cap)) if at_cap else None,
            "n": len(at_cap),
            "threshold": policies.settings.get("max_at_cap_fraction"),
            "concentration_dominated": (
                None
                if not at_cap or policies.settings.get("max_at_cap_fraction") is None
                else bool(float(np.median(at_cap)) > float(policies.settings["max_at_cap_fraction"]))
            ),
        },
        "ex_ante_sd": {
            "median": float(np.median(ex_ante)) if ex_ante else None,
            "max": float(np.max(ex_ante)) if ex_ante else None,
            "n": len(ex_ante),
        },
        "straddling_decisions": straddling,
        "flat_from_straddling_endpoints": flat,
        "excluded_folds": [
            {"index": int(plan.fold.fold), "reason": plan.exclusion_reason} for plan in plans if plan.excluded
        ],
        "calibration": {
            int(plan.fold.fold): (None if plan.calibration is None else plan.calibration["components"])
            for plan in plans
        },
        "decisions": list(bar_records),
    }


def _calibration_request(
    spec: Specification,
    resolved: _Resolved,
    *,
    fold: RollingFold,
    fitted: tuple[FittedComponent, ...],
    members: Sequence[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    needs_total_return: bool,
) -> CalibrationRequest:
    """Build the hook's request from train-only panels (§3.4)."""
    view = resolved.view_at(train_end)
    close = view.panel(list(members), train_end, "close")
    window = (close.index >= pd.Timestamp(train_start, tz=close.index.tz)) & (
        close.index <= pd.Timestamp(train_end, tz=close.index.tz)
    )
    panels: dict[str, pd.DataFrame] = {"price": close.loc[window]}
    if needs_total_return:
        total = view.total_return(list(members), train_end)
        panels["total_return"] = total.loc[
            (total.index >= panels["price"].index[0]) & (total.index <= panels["price"].index[-1])
        ]
    volume = view.panel(list(members), train_end, "volume").reindex(index=panels["price"].index)
    return CalibrationRequest(
        fold=int(fold.fold),
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        scoring_panels=panels,
        volume=volume,
        components=fitted,
        spec=spec,
    )


def _due(
    plans: Sequence[ComponentPlan],
    states: Mapping[str, ComponentState],
    positions: Mapping[str, int],
    position: int,
    *,
    decision_bar: str | None = None,
) -> bool:
    """Whether any component is due for a refresh, or expires, at this bar (§3.3).

    The same two predicates ``_compose_one`` evaluates — due-ness is
    :func:`prism.composition.due_on_grid` in both places — read here so a bar
    on which neither fires can skip the panel read entirely. When this is
    False, composition returns the carried sub-targets unchanged and the
    constructor steps the held book toward itself, so the emitted row is the
    held book — which is what the caller writes instead. ``decision_bar`` is
    the label at ``position``; a plan on the session grid needs it, and an
    elapsed-count plan never reads it.
    """
    for plan in plans:
        state = states.get(plan.component_id)
        last = None if state is None else state.last_output_bar
        elapsed = 0
        if last is not None:
            anchor = positions.get(last)
            if anchor is None:
                return True
            elapsed = position - anchor
        if plan.phase_offset_bars is not None and decision_bar is None:
            raise EvaluationError(
                f"component {plan.component_id!r} is scheduled on the session grid and the due-ness check was "
                "given no decision bar label to place on it"
            )
        if due_on_grid(plan, state, str(decision_bar), elapsed=elapsed):
            return True
        if last is not None and elapsed > int(plan.valid_for_bars):
            return True
    return False


def _decide_bar(
    spec: Specification,
    resolved: _Resolved,
    day: str,
    *,
    position: int,
    universe: Sequence[str],
    plans: Sequence[ComponentPlan],
    states: Mapping[str, ComponentState],
    needs_total_return: bool,
    history: int,
    warnings: list[str],
    construction: "_Construction | None" = None,
) -> tuple[pd.Series | None, dict[str, ComponentState], pd.Series | None]:
    """One decision bar, through its own point-in-time view (N1).

    A sid outside the point-in-time universe, or one that has just failed the
    eligibility screen, gets an explicit ``0.0``: leaving the tradeable set is
    an exit. A sid the spin-off mask withholds keeps its held weight, because
    the mask makes its score NaN — no opinion — which is not the same thing.

    Both are applied as masks over the whole universe and never as a column
    filter. Dropping a sid from the scored panel would leave a held name with
    no rule value at all, and §3.3's state machine holds a sub-target it is
    given no value for — so a name that had just left the index or failed the
    screen would be frozen at its last weight instead of exited, which is the
    opposite of what either gate means. It also has to be a mask rather than a
    filter for a second reason: a component's own screen does not know about
    index membership, so a name that has left the index would otherwise score
    finitely and re-enter the cross-section it was removed from.

    Returns the account target before construction, the state to persist, and
    the trailing dollar-volume statistic the participation bound needs (None
    when no bound is declared).
    """
    scored = _score_bar(
        spec,
        resolved,
        day,
        position=position,
        universe=universe,
        plans=plans,
        states=states,
        needs_total_return=needs_total_return,
        history=history,
        warnings=warnings,
        construction=construction,
    )
    if scored is None:
        return None, dict(states), None
    return scored.row, dict(scored.composition.states), scored.adv


@dataclass(frozen=True)
class _BarScores:
    """One decision bar's panels and composition output, kept for a state-coupled path.

    ``_decide_bar`` returns only what the stage-2 loop consumes. The constructor of §3.7
    needs the panels themselves -- the close window a horizon volatility and a risk
    snapshot are estimated on, the eligibility mask that decides which names carry a
    forecast, and the component's raw scores rather than a rank-to-weight rule's row -- so
    the body is here and both callers read the same one.
    """

    day: str
    members: tuple[str, ...]
    close: pd.DataFrame
    volume: pd.DataFrame
    eligible: pd.Series
    ineligible: tuple[str, ...]
    composition: CompositionResult
    row: pd.Series
    adv: pd.Series | None


def _score_bar(
    spec: Specification,
    resolved: _Resolved,
    day: str,
    *,
    position: int,
    universe: Sequence[str],
    plans: Sequence[ComponentPlan],
    states: Mapping[str, ComponentState],
    needs_total_return: bool,
    history: int,
    warnings: list[str],
    construction: "_Construction | None" = None,
) -> "_BarScores | None":
    """Score one decision bar through its own point-in-time view (N1). See :func:`_decide_bar`."""
    view = resolved.view_at(day)
    scored = list(universe)
    members = set(_members(spec, view, scored))
    close, volume = _close_and_volume(view, scored, day)
    eligible = _eligibility_mask(spec.strategy.eligibility, view, close, volume, scored)
    # Leaving the point-in-time universe and failing the screen are the same
    # instruction to the rank-to-weight rule: no score, therefore an explicit
    # flat. They are not the same as no opinion, which would hold the position.
    ineligible = [sid for sid in scored if sid not in members or not bool(eligible.get(sid, False))]
    if len(ineligible) == len(scored):
        warnings.append(
            f"{day}: no sid is both a point-in-time member and eligible, so the account target is an explicit "
            "flat rather than a held book"
        )
    panels: dict[str, pd.DataFrame] = {"price": close}
    if needs_total_return:
        panels["total_return"] = view.total_return(scored, day)
    _bind_membership(spec, view, plans, close)
    earliest = resolved.session_labels[max(0, position - history + 1)]
    result = compose(
        list(plans),
        decision_bar=day,
        scoring_panels=panels,
        sessions=list(resolved.session_labels[: position + 1]),
        volume=volume,
        states=dict(states),
        unrankable=_unrankable(view, scored, start=earliest, end=day),
        ineligible=ineligible,
        renormalize=bool(spec.strategy.renormalize),
    )
    row = pd.Series(0.0, index=pd.Index(list(universe)), dtype=float)
    row.loc[result.targets.index] = result.targets.to_numpy(dtype=float)
    adv: pd.Series | None = None
    if construction is not None and construction.needs_dollar_volume:
        adv = _trailing_dollar_volume(
            close,
            volume,
            window=construction.adv_window_bars,
            statistic=construction.adv_statistic,
        ).reindex(pd.Index(list(universe)))
    return _BarScores(
        day=day,
        members=tuple(sorted(members)),
        close=close,
        volume=volume,
        eligible=eligible,
        ineligible=tuple(ineligible),
        composition=result,
        row=row,
        adv=adv,
    )


def _fold_cost_inputs(
    spec: Specification,
    resolved: _Resolved,
    *,
    fold: RollingFold,
    plans: Sequence[ComponentPlan],
    construction: _Construction,
    execution: ExecutionConfig,
    train_start: str,
    train_end: str,
    universe: Sequence[str],
    history: int,
    needs_total_return: bool,
    warnings: list[str],
) -> _FoldCostInputs:
    """The fold's cost inputs and no-trade band, estimated on its formation window only.

    Both are train-only by construction: the spread buckets come from the
    formation window's dollar volume, and the closed-form band's variance comes
    from replaying the rank-to-weight rule over the same window and taking the
    variance of its day-over-day target changes. The replay is discarded after
    the estimate and feeds nothing forward — the same discipline the recorded
    driver states at `research/arbitrage/residual_walk_forward.py:444-462`.
    """
    view = resolved.view_at(train_end)
    columns = list(universe)
    spread: pd.Series | None = None
    if construction.spread_mode == "bucket" or construction.band_mode == "closed_form":
        close = view.panel(columns, train_end, "close")
        volume = view.panel(columns, train_end, "volume")
        window = (close.index >= pd.Timestamp(train_start, tz=close.index.tz)) & (
            close.index <= pd.Timestamp(train_end, tz=close.index.tz)
        )
        if construction.spread_mode == "bucket":
            spread = _bucket_spread(
                (close.loc[window] * volume.reindex(index=close.index).loc[window]),
                construction.adv_statistic,
            ).reindex(pd.Index(columns))

    band: float | pd.Series = 0.0
    if construction.band_mode == "fixed":
        band = float(construction.band_width)
    elif construction.band_mode == "closed_form":
        replay = _replay_targets(
            spec,
            resolved,
            plans=plans,
            positions=range(fold.formation_start, fold.formation_stop),
            universe=columns,
            history=history,
            needs_total_return=needs_total_return,
            warnings=warnings,
        )
        sigma2 = replay.diff().iloc[1:].var(ddof=1).reindex(pd.Index(columns))
        per_name_bps = spread.to_numpy(dtype=float) if spread is not None else float(execution.spread_bps)
        round_trip = 2.0 * (execution.commission_bps + per_name_bps) / 10_000.0
        band = pd.Series(
            closed_form_band(sigma2.to_numpy(dtype=float), round_trip, gamma_risk=float(construction.gamma_risk)),
            index=pd.Index(columns),
        )
    return _FoldCostInputs(fold=int(fold.fold), window=(train_start, train_end), spread_bps=spread, band=band)


def _replay_targets(
    spec: Specification,
    resolved: _Resolved,
    *,
    plans: Sequence[ComponentPlan],
    positions: Sequence[int] | range,
    universe: Sequence[str],
    history: int,
    needs_total_return: bool,
    warnings: list[str],
) -> pd.DataFrame:
    """The rank-to-weight rule's rows over a window, replayed from flat.

    Pre-band and pre-bound: the band's own variance estimate cannot depend on
    the band. State starts empty, so the window's first bar is a decision bar,
    which is the cadence convention the recorded driver replays under.
    """
    labels = resolved.session_labels
    index = {label: number for number, label in enumerate(labels)}
    states: dict[str, ComponentState] = {}
    rows: list[np.ndarray] = []
    last = np.zeros(len(universe), dtype=float)
    stamps: list[pd.Timestamp] = []
    for position in positions:
        day = labels[position]
        stamps.append(resolved.session_index[position])
        if position + 1 < history:
            # Fewer sessions than the components declare they need. A component
            # with no history has no opinion, which the rank-to-weight rule
            # reads as an explicit flat; its schedule starts at the first bar
            # it can score, not before.
            rows.append(np.zeros(len(universe), dtype=float))
            continue
        if not _due(plans, states, index, position, decision_bar=day):
            rows.append(last)
            continue
        row, states, _ = _decide_bar(
            spec,
            resolved,
            day,
            position=position,
            universe=universe,
            plans=plans,
            states=states,
            needs_total_return=needs_total_return,
            history=history,
            warnings=warnings,
        )
        last = last if row is None else row.to_numpy(dtype=float)
        rows.append(last)
    return pd.DataFrame(
        np.vstack(rows) if rows else np.zeros((0, len(universe))),
        index=pd.DatetimeIndex(stamps),
        columns=list(universe),
    )


def _bind_membership(spec: Specification, view: DatasetView, plans: Sequence[ComponentPlan], close: pd.DataFrame) -> None:
    """Bind the bar x sid point-in-time membership mask to every component that declares it needs one.

    A component whose fit reaches back over past cross-sections (the learned
    cross-section node) must train only on names that were index members on
    those bars; the panel it is handed spans every sid in the evaluated
    universe, including names that join the index after the decision bar. The
    mask's rows up to the decision bar depend only on whether each bar lies in
    a visible membership interval, so binding it adds nothing the decision bar
    could not know (N1). A spec with no universe index binds an all-True mask:
    its universe is the declared list, which has no membership history.
    """
    wanting = [plan.component for plan in plans if bool(getattr(plan.component, "uses_membership", False))]
    if not wanting:
        return
    if spec.strategy.universe_index is None:
        mask = pd.DataFrame(True, index=close.index, columns=close.columns, dtype=bool)
    else:
        mask = view.membership_mask(spec.strategy.universe_index, close.index, list(close.columns))
    for component in wanting:
        component.bind_membership(mask)


def _bind_membership_loaded(spec: Specification, view: DatasetView, close: pd.DataFrame) -> None:
    """:func:`_bind_membership` for the stateless path, which scores loaded components directly."""
    wanting = [loaded.component for loaded in spec.components if bool(getattr(loaded.component, "uses_membership", False))]
    if not wanting:
        return
    if spec.strategy.universe_index is None:
        mask = pd.DataFrame(True, index=close.index, columns=close.columns, dtype=bool)
    else:
        mask = view.membership_mask(spec.strategy.universe_index, close.index, list(close.columns))
    for component in wanting:
        component.bind_membership(mask)


def _close_and_volume(view: DatasetView, members: Sequence[str], day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The close and volume panels the decision bar sees.

    ``DatasetView.panel`` answers from the dataset's wide bar cache when that
    cache can answer exactly, so two calls are two slices of one read; when it
    cannot, each call reads the per-sid files and the second read is the price
    of a shape the cache must not approximate.
    """
    close = view.panel(list(members), day, "close")
    volume = view.panel(list(members), day, "volume")
    if close.empty:
        raise EvaluationError(
            f"{day}: the visible bars for {len(members)} member(s) are empty; a decision bar with no panel is a "
            "dataset defect, not an empty cross-section (N7)"
        )
    return close, volume.reindex(index=close.index, columns=close.columns)


def _unrankable(view: DatasetView, members: Sequence[str], *, start: str, end: str) -> frozenset[str]:
    """Sids withheld from ranking at this decision (§3.8, "The unrankable mask").

    Derived from spin-off rows in ``view.events`` under the same window rule the
    operating provider applies: effective after the earliest score endpoint and
    on or before the decision bar. The set is intersected with the scored
    universe by construction, since ``members`` is that universe.
    """
    events = view.events(list(members), start, end)
    if events.empty or "event_type" not in events.columns:
        return frozenset()
    spun = events.loc[events["event_type"].astype(str).isin(SPINOFF_EVENT_TYPES)]
    return frozenset(str(sid) for sid in spun["sid"].astype(str).unique())


def _charged_spread(
    resolved: _Resolved,
    *,
    folds: Sequence[FoldEvaluation],
    cost_inputs: Sequence[_FoldCostInputs],
    cost_refresh: str,
) -> pd.Series | pd.DataFrame | None:
    """What the accounting engine charges for the spread, per the refresh cadence (§3.8).

    ``once`` prices the whole sample from the first fold's formation window.
    One simulation runs over the stitched target sequence, and the first
    formation window precedes every test bar in it, so that vector is causal
    everywhere — which is the convention the recorded driver states and applies
    (`research/arbitrage/residual_walk_forward.py:428-433`).

    ``fold`` prices each fold's test bars from that fold's own formation
    window, as a day-by-name frame. It is the cadence §3.6 defines and it is
    *not* what the recorded run charged, so the two are never silently
    exchanged: a record says which it used and the two are different evidence.

    Returns ``None`` when no bucket schedule is declared, which is bit-identical
    to the flat scalar spread of the execution stack.
    """
    priced = [inputs for inputs in cost_inputs if inputs.spread_bps is not None]
    if not priced:
        return None
    if cost_refresh == "once":
        return priced[0].spread_bps
    frame = pd.DataFrame(np.nan, index=resolved.session_index, columns=list(resolved.universe), dtype=float)
    position_of = {label: number for number, label in enumerate(resolved.session_labels)}
    by_fold = {inputs.fold: inputs for inputs in priced}
    for evaluation in folds:
        inputs = by_fold.get(evaluation.fold)
        if inputs is None or inputs.spread_bps is None:
            continue
        first = position_of[evaluation.test[0]]
        # A decision at close t fills at open t+1, so the fold's own buckets
        # have to reach the bar its last decision fills on.
        last = min(position_of[evaluation.test[1]] + 1, len(resolved.session_index) - 1)
        frame.iloc[first : last + 1] = inputs.spread_bps.reindex(frame.columns).to_numpy(dtype=float)
    return frame


# ==========================================================================
# The canonical engine in evaluation (§3.8)
# ==========================================================================


def _canonical_decisions(targets: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """The decision rows, with every row that only restates the standing target dropped.

    The evaluator emits a target row on every test bar, including bars on which
    nothing was due: on those the row *is* the held book. Both engines treat
    such a row as a no-op — ``backtest_target_weights`` creates no pending order
    when the row equals the weights already filled
    (``execution/target_weights.py``: ``if not np.allclose(effective, current)``)
    — so dropping it here changes nothing about which decisions the canonical
    engine sees. Not dropping it would: ``decide`` would size a trade back to
    the target from the *drifted* share book on every bar, which is a daily
    rebalance, and a daily rebalance is a different holding process from the one
    under test.

    The predicate is the reproduction engine's own, ``np.allclose`` at its
    default tolerances, so the two engines act on exactly the same set of bars.
    Returns the masked frame and the bars that survived.
    """
    out = targets.copy()
    columns = out.shape[1]
    standing = np.zeros(columns, dtype=float)
    kept: list[str] = []
    for position in range(len(out)):
        row = out.iloc[position].to_numpy(dtype=float)
        if np.isnan(row).all():
            continue
        filled = np.where(np.isnan(row), standing, row)
        if np.allclose(filled, standing):
            out.iloc[position] = np.nan
            continue
        standing = filled
        kept.append(_iso(out.index[position]))
    return out, tuple(kept)


def _by_bar(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Re-key a **session-indexed** frame by ISO bar, which is what the accounting speaks.

    Only ever applied to something indexed by bar. A per-name vector — the
    ``once`` cadence's spread, indexed by sid — is left alone, and passing one
    here would silently rewrite sids as dates.
    """
    if frame is None:
        return None
    out = frame.copy()
    out.index = [_iso(stamp) for stamp in out.index]
    return out


def _canonical_cost_model(
    resolved: _Resolved,
    canonical: CanonicalSettings,
    spread_bps_per_name: pd.Series | pd.DataFrame | None,
    *,
    capital: float,
) -> CostModel:
    """The declared stack, bound to the spread the reproduction engine would have charged.

    The point of routing the per-fold bucket vector into ``CostInputs`` rather
    than letting the stack re-derive it is that the two engines then charge the
    *same* spread on the same name-day. A difference between the engines is then
    a difference in the holding process and in nothing else, which is the only
    way the comparison this stage owes means anything.
    """
    stack = canonical.cost_stack
    inputs = CostInputs(as_of=resolved.session_labels[-1], scope="run")
    if spread_bps_per_name is not None:
        # Only the *mode* changes: the schedule and flat_bps the stack declared
        # stay, and stay in that order, so a name the vector does not price
        # falls back where the stack says it falls back — the widest schedule
        # step, which is §3.6's "unknown liquidity is never cheap". That is a
        # stricter fallback than the reproduction engine's flat spread, and it
        # is reachable only for a name the bucket vector does not cover, which
        # `_bucket_spread` covers by construction (an all-NaN name is already
        # in the widest step).
        spread = SpreadSpec(
            mode="frame",
            statistic=stack.spread.statistic,
            flat_bps=stack.spread.flat_bps,
            schedule=stack.spread.schedule,
        )
        stack = dataclasses.replace(stack, spread=spread, stack_id=stack.stack_id)
        # A ``once`` spread is a per-name vector and is passed through as it is;
        # a ``fold`` spread is a bar-by-name panel and is re-keyed by bar, which
        # is how ``CostModel._lookup`` addresses a frame.
        per_bar = isinstance(spread_bps_per_name, pd.DataFrame)
        inputs = CostInputs(
            as_of=resolved.session_labels[-1],
            scope="fold" if per_bar else "run",
            spread_bps=_by_bar(spread_bps_per_name) if per_bar else spread_bps_per_name,
        )
    return CostModel(stack, inputs, equity=float(capital))


def _simulate_canonical(
    resolved: _Resolved,
    targets: pd.DataFrame,
    *,
    canonical: CanonicalSettings,
    spread_bps_per_name: pd.Series | pd.DataFrame | None,
    capital: float,
    warnings: list[str],
) -> tuple[PortfolioBacktestResult, dict[str, object]]:
    """Account the stitched target sequence with the fixed-share simulator (§3.6, §3.8).

    Shares are the state. Between decisions nothing rebalances, so the weights
    the book actually carries drift with prices; the equity path is the marked
    close-to-close path of a real account, not the compounding of a weight row.
    """
    columns = list(resolved.universe)
    end = resolved.session_labels[-1]
    closes = resolved.view.panel(columns, end, "close").reindex(index=resolved.session_index)
    volumes = resolved.view.panel(columns, end, "volume").reindex(index=resolved.session_index)
    events = resolved.view.events(columns, resolved.session_labels[0], end)
    decisions, decision_bars = _canonical_decisions(targets)

    simulation = simulate(
        _by_bar(resolved.opens),
        _by_bar(closes),
        _by_bar(decisions),
        cost_model=_canonical_cost_model(resolved, canonical, spread_bps_per_name, capital=capital),
        volumes=_by_bar(volumes),
        events=events,
        price_basis=_price_basis_for(resolved, [] if events.empty else events["sid"].astype(str).tolist()),
        initial_cash=float(capital),
        fills=canonical.fills,
        increment=float(canonical.share_increment),
        min_notional=float(canonical.min_notional),
        order_type="market_on_open",
        zero_crossing_clamp=bool(canonical.zero_crossing_clamp),
        bars=[_iso(stamp) for stamp in resolved.session_index],
    )
    if simulation.state.is_incomplete:
        warnings.append(
            "the canonical book ends with an unresolved claim on at least one name; the record is complete only "
            "when every terminal outcome is known (§3.2, §3.6)"
        )
    stale = int(simulation.ledger["stale"].max()) if len(simulation.ledger) else 0
    detail: dict[str, object] = {
        "note": CANONICAL_NOTE,
        "n_decision_bars": len(decision_bars),
        "n_target_rows": int(targets.notna().any(axis=1).sum()),
        "n_fills": int(len(simulation.fills)),
        "n_orders": int(len(simulation.orders)),
        "n_events_applied": int(len(simulation.events)),
        "max_stale_marks": stale,
        "pending_orders_at_end": int(len(simulation.state.pending_orders)),
        "final_cash": float(simulation.state.cash),
        "settings": canonical.to_record(),
    }
    result = _canonical_backtest_result(
        simulation, index=resolved.session_index, universe=resolved.universe, targets=targets
    )
    return result, detail


def _canonical_backtest_result(
    simulation: SimulationResult,
    *,
    index: pd.DatetimeIndex,
    universe: Sequence[str],
    targets: pd.DataFrame,
) -> PortfolioBacktestResult:
    """The simulation, in the shape every record writer downstream already reads.

    Two conversions, and both are stated rather than convenient. The return of
    bar *t* is the change in **marked equity** from the close of *t-1* to the
    close of *t*, so the first bar of the sample carries no return: there is no
    earlier close to measure from. Charged cost is converted to the same unit by
    dividing that bar's charged dollars by the equity it was charged against,
    which makes ``gross = net + cost`` an identity rather than an approximation
    (equity is additive: ``E_t = E_{t-1} + pnl_t - cost_t``).
    """
    stamps = list(index)
    equity = pd.Series(simulation.ledger["equity_close"].to_numpy(dtype=float), index=pd.DatetimeIndex(stamps))
    base = equity.shift(1)
    returns = (equity / base - 1.0).iloc[1:].rename("daily_return")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise EvaluationError(
            "the canonical equity path is non-finite or passes through zero; a book with no equity has no return "
            "and is a refusal rather than a NaN in the record (N7)"
        )

    costs = simulation.costs.copy()
    costs.index = pd.DatetimeIndex(stamps)
    charged = costs.div(base, axis=0).iloc[1:]
    fills = simulation.fills
    traded = pd.Series(0.0, index=pd.DatetimeIndex(stamps))
    if not fills.empty:
        notional = (fills["qty"].astype(float) * fills["fill_price"].astype(float)).abs()
        by_bar = notional.groupby(fills["bar"].astype(str)).sum()
        traded = pd.Series(
            {stamp: float(by_bar.get(_iso(stamp), 0.0)) for stamp in stamps}, index=pd.DatetimeIndex(stamps)
        )
    weights = simulation.weights.copy()
    weights.index = pd.DatetimeIndex(stamps)
    weights = weights.reindex(columns=list(universe)).fillna(0.0)
    dividends = pd.Series(simulation.ledger["event_equity"].to_numpy(dtype=float), index=pd.DatetimeIndex(stamps)).div(
        base
    )

    frame = pd.DataFrame(
        {
            "commission_spread": charged["commission"] + charged["spread"],
            "impact": charged["impact_quadratic"] + charged["impact_participation"],
            "borrow": charged["borrow"],
            "financing": charged["financing"],
            "cash_income": charged["cash_income"],
            "total": charged["total"],
            "turnover": (traded / base).iloc[1:],
            "gross": weights.abs().sum(axis=1).iloc[1:],
            "net": weights.sum(axis=1).iloc[1:],
            "dividend_return": dividends.iloc[1:].fillna(0.0),
        }
    )
    return PortfolioBacktestResult(
        returns=returns,
        equity=equity.iloc[1:],
        target_weights=targets,
        fill_weights=weights.iloc[1:],
        costs=frame,
        metrics=compute_portfolio_metrics(returns, equity.iloc[1:], frame),
    )


def _scored_index(
    resolved: _Resolved, folds: Sequence[FoldEvaluation]
) -> tuple[pd.DatetimeIndex, list[dict[str, object]]]:
    """The bars the record's metrics are taken on, and the folds left out.

    Metrics are taken on test rows only, as the recorded driver does
    (`research/arbitrage/walk_forward.py:414`). A fold the calibration hook
    excluded contributes no bar, and the exclusion with its reason is recorded,
    because a silently dropped fold changes what every per-fold number averages.
    """
    positions: list[int] = []
    excluded: list[dict[str, object]] = []
    for evaluation, fold in zip(folds, resolved.folds):
        if evaluation.excluded:
            excluded.append({"index": int(fold.fold), "reason": evaluation.exclusion_reason})
            continue
        positions.extend(range(fold.test_start, fold.test_stop))
    if not positions:
        raise EvaluationError(
            "every fold was excluded: there is no out-of-sample bar left to score, which is a refusal rather "
            "than a result with no observations (N7)"
        )
    return resolved.session_index[positions], excluded


def _slice_result(full: PortfolioBacktestResult, index: pd.Index, *, capital: float) -> PortfolioBacktestResult:
    """One continuous simulation, scored on the test rows only.

    The equity path is re-compounded from the sliced returns so it opens at the
    scenario's capital, exactly as ``_slice_portfolio_result``
    (`research/arbitrage/walk_forward.py:400`) does; the returns themselves come
    from the single stitched simulation and are never re-run per fold.
    """
    kept = full.returns.index.intersection(index)
    if kept.empty:
        raise EvaluationError(
            "the scored index selects no return bar of the simulation; the fold table and the traded session "
            "index disagree, which is a geometry defect rather than an empty result (N7)"
        )
    returns = full.returns.loc[kept]
    costs = full.costs.loc[kept]
    equity = ((1.0 + returns).cumprod() * float(capital)).rename("equity")
    return PortfolioBacktestResult(
        returns=returns,
        equity=equity,
        target_weights=full.target_weights.loc[full.target_weights.index.intersection(index)],
        fill_weights=full.fill_weights.loc[full.fill_weights.index.intersection(index)],
        costs=costs,
        metrics=compute_portfolio_metrics(returns, equity, costs),
    )


def _attribute_turnover(
    evaluation: FoldEvaluation, full: PortfolioBacktestResult, resolved: _Resolved
) -> FoldEvaluation:
    """Split the fold's turnover and charged cost into boundary and interior (§3.8).

    Under ``reset`` half of every fold's decisions are forced full-gross trades,
    so a cost difference between two arms can be an artifact of the boundary
    rather than a property of either. The two are therefore reported
    separately, per fold, under both modes.
    """
    position_of = {label: number for number, label in enumerate(resolved.session_labels)}
    first = position_of[evaluation.test[0]]
    last = position_of[evaluation.test[1]]
    test_positions = list(range(first, last + 1))
    stamps = resolved.session_index[test_positions]
    rows = full.costs.reindex(stamps).dropna(how="all")
    # A decision at close t fills at open t+1, so a boundary *decision* is
    # charged on the following bar's cost row.
    boundary_stamps = {
        resolved.session_index[position_of[label] + 1]
        for label in evaluation.boundary_decisions
        if position_of[label] + 1 < len(resolved.session_index)
    }
    boundary = rows.loc[rows.index.isin(boundary_stamps)]
    interior = rows.drop(index=boundary.index)
    metrics = compute_trial_metrics(
        full.returns.reindex(stamps).dropna(), full.costs.reindex(stamps).dropna(how="all")
    )
    return dataclasses.replace(
        evaluation,
        metrics=metrics,
        turnover={
            "total": float(rows["turnover"].sum()) if "turnover" in rows else 0.0,
            "boundary": float(boundary["turnover"].sum()) if "turnover" in boundary else 0.0,
            "interior": float(interior["turnover"].sum()) if "turnover" in interior else 0.0,
            "cost_total": float(rows["total"].sum()) if "total" in rows else 0.0,
            "cost_boundary": float(boundary["total"].sum()) if "total" in boundary else 0.0,
            "cost_interior": float(interior["total"].sum()) if "total" in interior else 0.0,
            "n_boundary_bars": int(len(boundary)),
            "n_interior_bars": int(len(interior)),
        },
    )


def _write_record(
    spec: Specification,
    resolved: _Resolved,
    *,
    workspace: Workspace,
    run_id: str,
    kind: str,
    arm_id: str | None,
    selection_set_id: str | None,
    registration_id: str | None,
    registration_hash: str | None,
    run_channel: str,
    engine: str,
    capital: float,
    execution: ExecutionConfig,
    fold_portfolio: str,
    cost_refresh: str,
    cost_inputs: Sequence[_FoldCostInputs],
    charged_spread: pd.Series | pd.DataFrame | None,
    canonical: CanonicalSettings | None,
    engine_detail: Mapping[str, object],
    folds: Sequence[FoldEvaluation],
    excluded: Sequence[Mapping[str, object]],
    scored: PortfolioBacktestResult,
    full: PortfolioBacktestResult,
    targets: pd.DataFrame,
    warnings: Sequence[str],
    log: TrialLog,
    reproduces: Sequence[Mapping[str, str]],
    supersedes: Sequence[str],
    untested_conclusions: Sequence[str],
    calibration_hook: object | None,
    construction_record: Mapping[str, object] | None = None,
) -> tuple[ResultRecord, Path]:
    """Build, validate and write the ``ResultRecord`` and its artifacts (§3.12)."""
    construction = _Construction.from_spec(spec)
    run_dir = workspace.results / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in (
        ("returns.csv", scored.returns),
        ("equity.csv", scored.equity),
        ("costs.csv", scored.costs),
        ("target_weights.csv", targets.dropna(how="all")),
    ):
        path = run_dir / name
        frame.to_csv(path)
        written.append(path)
    folds_path = run_dir / "folds.json"
    folds_path.write_text(json.dumps(resolved.fold_table.to_records(), indent=2) + "\n", encoding="utf-8")
    written.append(folds_path)

    metrics = compute_trial_metrics(scored.returns, scored.costs, targets.dropna(how="all"))
    tier = cap_evidence_tier(
        classify_claim_tier(metrics),
        synthetic=resolved.synthetic,
        run_channel=run_channel,
    )
    counts = attempt_counts(log.rows(), registered=selection_set_id is not None)
    coverage = _coverage(resolved, untested_conclusions=untested_conclusions)
    turnover = {
        "total": float(scored.costs["turnover"].sum()),
        "boundary": float(sum(fold.turnover.get("boundary", 0.0) for fold in folds)),
        "interior": float(sum(fold.turnover.get("interior", 0.0) for fold in folds)),
        "average": float(scored.costs["turnover"].mean()),
    }
    cost_total = {
        "total": float(scored.costs["total"].sum()),
        "boundary": float(sum(fold.turnover.get("cost_boundary", 0.0) for fold in folds)),
        "interior": float(sum(fold.turnover.get("cost_interior", 0.0) for fold in folds)),
    }
    record = ResultRecord(
        ids={
            "run_id": run_id,
            "config_id": resolved.config_id,
            "data_id": resolved.data_id,
            "kind": kind,
            "arm_id": arm_id,
            "registration_id": registration_id,
            "registration_hash": registration_hash,
            "selection_set_ids": [selection_set_id] if selection_set_id else [],
            "reproduces": [dict(entry) for entry in reproduces],
            "supersedes": list(supersedes),
        },
        status={"status": "finished", "incomplete_reasons": []},
        versions={
            "engine": engine_version(),
            "solver": UNAVAILABLE,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
        },
        conventions={
            "returns": _returns_convention(engine),
            "scoring_basis": {
                loaded.component_id: str(loaded.declarations["scoring_basis"]) for loaded in spec.components
            },
            "price_basis": "as_traded_close",
            "executable_return": ENGINE_EXECUTABLE_RETURN[engine],
            "excess_return_definition": UNAVAILABLE,
            "hurdle_series_id": UNAVAILABLE,
            "hurdle_conversion": UNAVAILABLE,
            "cash_and_financing": _cash_and_financing(engine, canonical),
            "execution_profile": _file_reference(spec.policy.execution),
        },
        data={
            "visibility_mode": resolved.visibility.mode,
            "vintage": (resolved.visibility.vintage if resolved.visibility.vintage is not None else UNAVAILABLE),
            "manifest_id": resolved.data_id,
            "lag_assumptions": {
                source: int(entry.get("lag_bars", 0)) for source, entry in resolved.dataset.manifest.sources.items()
            },
            "score_panel": None,
            "price_basis": resolved.price_basis,
            "unavailable_reason": None if resolved.visibility.vintage is not None else NO_VINTAGE_REASON,
            "observed": {
                "view_mode": resolved.visibility.mode,
                "decision_time": resolved.visibility.decision_time,
                "manifest_vintage_label": resolved.dataset.manifest.vintage,
                "provenance": str(resolved.dataset.manifest.provenance),
            },
        },
        coverage=coverage,
        costs={
            "stack": _execution_stack(execution),
            "calibration_source": UNAVAILABLE,
            "cost_refresh": {
                "cadence": cost_refresh,
                "windows": [{"fold": fold.fold, "start": fold.train[0], "end": fold.train[1]} for fold in folds],
                "charged": _charged_spread_record(charged_spread, execution),
                "per_fold": [inputs.to_record() for inputs in cost_inputs],
                "note": (
                    "each fold's cost inputs are estimated on that fold's formation window and recorded above; "
                    "the cadence says which of them the accounting engine charged. Under 'once' the first "
                    "window prices the whole stitched simulation, which is causal for every test bar in it; "
                    "under 'fold' each fold's test bars carry their own buckets"
                ),
            },
            "dollar_volume_statistics": {
                "spread": _spread_statistic(construction, execution),
                "participation": (
                    UNAVAILABLE
                    if construction.max_participation is None
                    else {
                        "statistic": construction.adv_statistic,
                        "window_bars": int(construction.adv_window_bars),
                        "floor_dollars": float(construction.adv_floor_dollars),
                        "base": construction.participation_base,
                        "max_participation": float(construction.max_participation),
                    }
                ),
                "impact": (
                    UNAVAILABLE
                    if execution.adv_impact_coeff <= 0.0
                    else {"model": execution.adv_impact_model, "coeff": float(execution.adv_impact_coeff)}
                ),
            },
        },
        evaluation={
            "fold_geometry": resolved.fold_table.fold_geometry(),
            "excluded_folds": [dict(entry) for entry in excluded],
            "fold_portfolio": fold_portfolio,
            "construction": (
                construction.to_record()
                if construction_record is None
                else dict(construction_record, heuristic_settings=construction.to_record())
            ),
            "train_only_transforms": {
                "fits": True,
                "calibration": calibration_hook is not None,
                "note": (
                    "every fit is taken on its fold's formation window and nothing outside it; each fold's "
                    "cutoff and window are recorded per component below"
                ),
            },
            "calibration": (
                _calibration_block(spec, folds, calibration_hook is not None)
                if construction_record is None
                else _stateful_calibration_block(spec, construction_record)
            ),
            "risk_model": _file_reference(spec.construction.risk_model),
            "accounting_engine": engine,
            # The engine's *label* and the process it names are recorded
            # together, and validation refuses a record where they disagree: the
            # two engines differ by the whole uncharged rebalance and must never
            # be read for one another (§3.8).
            "holding_process": ENGINE_HOLDING_PROCESS[engine],
            "engine": dict(engine_detail),
            "capital": float(capital),
            "components": [
                {
                    "component_id": loaded.component_id,
                    "fit_cutoff": _last_fit(folds, loaded.component_id, "fit_cutoff"),
                    "fit_window": _last_fit(folds, loaded.component_id, "fit_window"),
                    "fitted_state_ref": _last_fit(folds, loaded.component_id, "fitted_state_ref"),
                    "fits": [
                        dict(component.to_record(), fold=fold.fold)
                        for fold in folds
                        for component in fold.components
                        if component.component_id == loaded.component_id
                    ],
                }
                for loaded in spec.components
            ],
            "folds": [fold.to_record() for fold in folds],
        },
        performance={
            "periodic_returns_path": str(run_dir / "returns.csv"),
            "gross": {
                "total_return": metrics["gross_total_return"],
                "periodic_sharpe": metrics["gross_periodic_sharpe"],
            },
            "net": {"total_return": metrics["total_return"], "periodic_sharpe": metrics["periodic_sharpe"]},
            "sharpe": {
                "periodic": metrics["periodic_sharpe"],
                "annualized": metrics["annualized_sharpe"],
                "se_method": "none_reported",
                "se_note": (
                    "the studentized block-bootstrap interval is computed by 'prism compare' over the recorded "
                    "return path (validation/compare.py); this record reports the point estimate"
                ),
            },
            "drawdown": {"max_drawdown": float(scored.metrics["max_drawdown"])},
            "tails": _tails(scored.returns),
            "turnover": turnover,
            "exposures": {
                "avg_gross": float(scored.metrics["avg_gross"]),
                "avg_net": float(scored.metrics["avg_net"]),
            },
            "violations": _violation_counts(construction_record),
            "repair": (
                dict(empty_repair_counts(), note=_repair_note(engine))
                if construction_record is None
                else _repair_counts(construction_record)
            ),
            "binding_conditions": _binding_counts(construction_record),
            "cost": cost_total,
            "sample": {"n_obs": int(len(scored.returns)), "n_folds": len(folds), "n_excluded_folds": len(excluded)},
        },
        selection={
            "n_trials_observed": int(counts["n_trials_observed"]),
            "attempt_basis": counts["attempt_basis"],
            "dependence_adjustment": {"method": UNAVAILABLE, "value": UNAVAILABLE},
            "unobserved_search_note": (
                "the trial log records the runs that were appended; the hash chain detects change relative to a "
                "trusted copy only and does not prove the search was exhaustive (§3.12)"
            ),
            "legacy_independent_count": None,
            "n_independent": int(counts["n_independent"]),
            "excluded_rows": dict(counts["excluded_rows"]),
            "n_intervals": 0,
        },
        flags={
            "synthetic": bool(resolved.synthetic),
            "exploratory": kind == "exploratory",
            "prospective": None,
            "fill_source": "simulated",
            "run_channel": run_channel,
            "evidence_tier": tier,
        },
        legacy={
            "legacy_ledger_config_hash": None,
            "legacy_packet_config_hash": None,
            "legacy_code_commit": None,
            "legacy_run_key": None,
            "source_ledger": None,
        },
        artifacts=[{"path": str(path), "sha256": _sha256_path(path), "artifact_status": "present"} for path in written]
        + [
            {
                "path": str(spec.path),
                "sha256": _sha256_path(spec.path),
                "artifact_status": "present" if Path(spec.path).is_file() else "unavailable",
            }
        ],
        warnings=[_engine_note(engine, engine_detail), *warnings],
    ).validate()
    record_path = record.write(run_dir / "record.json")
    return record, record_path


def _returns_convention(engine: str) -> str:
    """How a distribution enters the return, per engine (§3.12 ``conventions.returns``)."""
    if engine == REPRODUCTION_ENGINE:
        return "price plus cash distributions, credited on the ex-date at the fill open"
    return (
        "price plus cash distributions, booked as a receivable (or a liability on a short) at the ex-date open "
        "and settled to cash on the pay date; equity is continuous across the ex-date markdown (§3.6)"
    )


def _cash_and_financing(engine: str, canonical: CanonicalSettings | None) -> str:
    """What the record says about cash, financing and borrow, per engine (§3.12)."""
    if engine == REPRODUCTION_ENGINE or canonical is None:
        return (
            "the reproduction engine models neither interest on uninvested cash nor financing; borrow is "
            "charged by the execution stack on short notional only"
        )
    stack = canonical.cost_stack
    return (
        f"declared cost stack {stack.stack_id}: borrow {stack.borrow_bps_annual} bps/yr on short market value as "
        f"marked, financing {stack.financing_bps_annual} bps/yr and cash income {stack.cash_income_bps_annual} "
        f"bps/yr on the {stack.short_proceeds} base, settled {stack.holding_cost_settlement}, "
        f"{stack.trading_days_per_year} trading days per year"
    )


def _engine_note(engine: str, engine_detail: Mapping[str, object]) -> str:
    """The caveat the record leads with, which must describe the run that was made.

    The canonical engine's own note says the target sequence was built from the emitted
    weight book rather than from the simulated state. Under ``account_state`` coupling that
    sentence is false, and a record whose leading warning contradicts what it recorded is
    worse than one with no warning at all.
    """
    if engine == REPRODUCTION_ENGINE:
        return REPRODUCTION_CAVEAT
    if str(engine_detail.get("decision_coupling")) == "account_state":
        return COUPLED_NOTE
    return CANONICAL_NOTE


def _repair_counts(construction: Mapping[str, object]) -> dict[str, object]:
    """The repair outcome counts §3.8 asks be recorded per rebalance and per scenario.

    ``attempted`` is every decision that reached batch rounding, ``succeeded`` those that
    rounded without a scale or a drop. They are counts of decisions, not a narrative:
    "how often repair fails" is read from here.
    """
    responses = dict(construction.get("repair") or {})
    scaled = int(responses.get("scale_batch", 0))
    dropped = int(responses.get("drop_batch", 0))
    none = int(responses.get("none", 0))
    no_action = int(responses.get("no_action", 0))
    return dict(
        empty_repair_counts(),
        attempted=scaled + dropped + none,
        succeeded=none,
        scaled_batch=scaled,
        dropped_batch=dropped,
        no_action=no_action,
        by_response=responses,
        note=(
            "counted over decision dates from portfolio/rounding.py::round_batch, the second of the two "
            "independent validations of §3.7"
        ),
    )


def _binding_counts(construction: Mapping[str, object] | None) -> dict[str, object]:
    """How often each limit determined the outcome, over decision dates (§3.12)."""
    if construction is None:
        return {}
    counts: dict[str, int] = {}
    for row in construction.get("decisions") or ():
        name = row.get("binding_condition")
        if name is None:
            continue
        counts[str(name)] = counts.get(str(name), 0) + 1
    return counts


def _violation_counts(construction: Mapping[str, object] | None) -> list[dict[str, object]]:
    """Every limit that was measured breached, with the count of decisions it bound on."""
    if construction is None:
        return []
    tally: dict[tuple[str, str, str], int] = {}
    for row in construction.get("decisions") or ():
        violations = row.get("violations") or {}
        for key in ("existing", "remaining", "introduced"):
            for entry in violations.get(key, ()):  # type: ignore[union-attr]
                name = (str(entry["limit_name"]), str(entry["class"]), key)
                tally[name] = tally.get(name, 0) + 1
    return [
        {"limit_name": name, "class": cls, "partition": key, "n_decisions": count}
        for (name, cls, key), count in sorted(tally.items())
    ]


def _repair_note(engine: str) -> str:
    """Why the repair counts are what they are, per engine (§3.7, §3.8)."""
    if engine == REPRODUCTION_ENGINE:
        return (
            "batch rounding repair belongs to portfolio/rounding.py::round_batch (§3.7); the "
            "historical_constant_weight engine holds weights, not shares, so no batch is rounded and every "
            "outcome count is zero rather than absent"
        )
    return (
        "the canonical engine sizes each decision through accounting.decide, which rounds the trade delta to the "
        "declared increment and drops an order below min_notional; whole-batch repair and the scaling schedule "
        "are portfolio/rounding.py::round_batch and reach the evaluator with assemble (§3.7, §3.8), so every "
        "outcome count is zero rather than absent"
    )


def _coverage(resolved: _Resolved, *, untested_conclusions: Sequence[str]) -> dict[str, object]:
    """The coverage disclosures, computed where they can be and named where they cannot."""
    priced = [sid for sid in resolved.universe if sid in resolved.opens.columns]
    missing = sorted(set(resolved.universe) - set(priced))
    assumptions = dict(resolved.dataset.manifest.assumptions or {})
    untested = list(untested_conclusions)
    unapplied = [str(item) for item in assumptions.get("scenarios_written_not_applied", ())]
    if unapplied:
        untested.append("scenarios written into the dataset but not applied by this stage: " + ", ".join(unapplied))
    return {
        "membership_coverage": float(len(priced)) / float(len(resolved.universe)) if resolved.universe else 0.0,
        "missing_securities": missing,
        "terminal_outcomes": UNAVAILABLE,
        "provisional_sid_count": UNAVAILABLE,
        "quarantined_sid_count": UNAVAILABLE,
        "untested_conclusions": untested,
        "bias_status": "unquantified",
        "unavailable_reason": (
            "terminal outcomes, the provisional and quarantined sid counts and a quantified survivorship bias "
            "come from the security master and the events table (io/security_master.py, io/events.py), which "
            "arrive at stage 3"
        ),
        "n_universe": len(resolved.universe),
    }


def _calibration_block(spec: Specification, folds: Sequence[FoldEvaluation], hooked: bool) -> dict[str, object]:
    """The record's calibration block (§3.12): the file, its digest, and the per-fold fit."""
    reference = _file_reference(spec.construction.calibration)
    return {
        "file_id": reference["path"],
        "sha256": reference["sha256"],
        "confidence": None,
        "fit_per_fold": [
            {"fold": fold.fold, "fit": None if fold.calibration is None else dict(fold.calibration)} for fold in folds
        ],
        "hook": "declared" if hooked else "none",
        "note": (
            "the shared-slope estimator (calibration.py::fit_calibration) arrives at stage 4; stage 2 calls the "
            "hook once per fold on that fold's training window and records what it returns (§3.4)"
        ),
    }


def _stateful_calibration_block(spec: Specification, construction: Mapping[str, object]) -> dict[str, object]:
    """The record's calibration block on the state-coupled path (§3.12).

    Everything here comes from the fits the run actually used: the file and its digest,
    the confidence the file declares (``null`` on the point arm, by declaration, which is
    not the same as "not recorded"), and the per-fold fit including the folds the shared
    admission excluded for every arm alike (D6).
    """
    reference = _file_reference(spec.construction.calibration)
    policies = construction.get("policies") or {}
    per_fold = construction.get("calibration") or {}
    confidence: object = None
    fits: list[dict[str, object]] = []
    for fold in sorted(per_fold, key=lambda key: int(key)):
        components = per_fold[fold]
        fits.append({"fold": int(fold), "fit": None if components is None else dict(components)})
        if components:
            for entry in components.values():
                for record in (entry.get("fits") or {}).values():
                    if record.get("confidence") is not None:
                        confidence = record["confidence"]
    return {
        "file_id": reference["path"],
        "sha256": reference["sha256"],
        "confidence": confidence,
        "fit_per_fold": fits,
        "hook": "none",
        "excluded_folds": list(construction.get("excluded_folds") or []),
        "family": (policies or {}).get("calibration"),
        "note": (
            "the shared-slope fit of §3.4, taken on each fold's formation window alone; fold admission is the "
            "shared family's and is the same decision for every arm of the registration (D6)"
        ),
    }


def _last_fit(folds: Sequence[FoldEvaluation], component_id: str, key: str) -> object:
    """The last fold's value of one fit field for one component."""
    for fold in reversed(list(folds)):
        for component in fold.components:
            if component.component_id == component_id:
                return component.to_record()[key]
    return None


def _tails(returns: pd.Series) -> dict[str, float | None]:
    """The tail description the record carries beside the headline Sharpe."""
    finite = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return {"skew": None, "excess_kurtosis": None, "min": None, "max": None, "p05": None, "p95": None}
    return {
        "skew": float(finite.skew()) if len(finite) > 2 else None,
        "excess_kurtosis": float(finite.kurtosis()) if len(finite) > 3 else None,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p05": float(finite.quantile(0.05)),
        "p95": float(finite.quantile(0.95)),
    }


def _file_reference(path: Path | None) -> dict[str, str | None]:
    """``{path, sha256}`` for a declared file, or nulls when none is declared."""
    if path is None:
        return {"path": None, "sha256": None}
    target = Path(path)
    return {"path": str(target), "sha256": _sha256_path(target) if target.is_file() else None}


def _charged_spread_record(charged: pd.Series | pd.DataFrame | None, execution: ExecutionConfig) -> dict[str, object]:
    """What the accounting engine actually charged for the spread, as a record block."""
    if charged is None:
        return {"mode": "flat", "value_bps": float(execution.spread_bps)}
    values = charged.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "mode": "bucket_frame" if isinstance(charged, pd.DataFrame) else "bucket",
        "n_names": int(charged.shape[-1] if isinstance(charged, pd.DataFrame) else len(charged)),
        "min_bps": float(finite.min()) if finite.size else None,
        "max_bps": float(finite.max()) if finite.size else None,
        "mean_bps": float(finite.mean()) if finite.size else None,
        "fallback_bps": float(execution.spread_bps),
    }


def _spread_statistic(construction: "_Construction", execution: ExecutionConfig) -> dict[str, object]:
    """The ``dollar_volume.spread_bucket`` declaration of §3.6, or the flat scalar."""
    if construction.spread_mode == "flat":
        return {"mode": "flat", "value_bps": float(execution.spread_bps)}
    declared = SpreadSpec(mode="bucket", schedule=SPREAD_BUCKET_SCHEDULE_V1)
    return {
        "mode": "bucket",
        "statistic": declared.statistic,
        "bound": declared.bound,
        "unknown_liquidity": declared.unknown_liquidity,
        "window": "fold_formation",
        "schedule": [[float(floor), float(bps)] for floor, bps in (declared.schedule or ())],
    }


def _execution_stack(execution: ExecutionConfig) -> dict[str, object]:
    """The resolved cost stack as recorded numbers, not a reference to an object."""
    if dataclasses.is_dataclass(execution):
        return {key: _plain(value) for key, value in dataclasses.asdict(execution).items()}
    return {key: _plain(value) for key, value in vars(execution).items()}


def _plain(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_path(path: Path | str) -> str | None:
    """Content hash of a file, or None when it is not there to hash."""
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(stamp: object) -> str:
    """A session label: the calendar date, which is what a decision bar is named by."""
    return pd.Timestamp(stamp).strftime("%Y-%m-%d")
