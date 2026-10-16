"""The tranched paper book: overlapping tranches, staged entry and wind-down, off by default.

``prism.live.daily.run_daily_cycle(tranches=K)`` with ``K > 1`` splits the
decile momentum book into ``K`` overlapping tranches, the way Jegadeesh and
Titman (1993) built overlapping momentum portfolios. Each tranche holds
``1/K`` of capital and refreshes every ``decision_every`` sessions on its own
date of the fixed session grid (``prism.io.sessions.grid_position``): tranche
*j* refreshes where ``(grid_position - offset_j) % decision_every == 0``, with
``offset_j = (base + j * decision_every // K) % decision_every``
(:func:`prism.composition.tranche_plans`). At ``K = decision_every`` exactly one
tranche refreshes on every session. ``base`` is the seeded phase offset when
one is given (``--rebalance-phase seeded``), else 0, the epoch grid; at ``K =
decision_every`` the base only relabels the tranches.

The account target is the sum of the tranches' sub-targets in position space
(:mod:`prism.composition`, :func:`prism.assembly.assemble`), and orders are the
difference between that target and the whole shares held, as for the single
book. Tranching changes when trades happen, not what is held: on a session where
one tranche refreshes, the account trades roughly ``1/K`` of the monthly change.
It does not net trades away in general; the only sessions where several
tranches refresh at once are catch-ups after an outage.

**Schedule and state.** The due-ness predicate is
:func:`prism.composition.due_on_grid`, the one the evaluator and composition
use: a tranche with no output yet waits for its first grid date (staged entry,
so a new book reaches full size over ``decision_every`` sessions instead of
buying it on day one), and a tranche that missed its grid date catches up once
on the first session after more than ``decision_every`` have elapsed. Each
tranche's output is valid for ``2 * decision_every`` sessions, so a missed date
makes a tranche late, not expired, unless the loop was dark for more than two
cycles, in which case the tranche goes flat and is rebuilt. The tranche states
(schedule anchor, sub-targets, grid offset and, with the volatility scale, the
scale decision) are ``LoopState.components``, persisted in the same write-ahead
save that advances ``last_refresh_bar``, so a restart in the middle of the build
resumes the same schedule and never rebuilds a tranche twice.

**A tranche's refresh is its whole book.** The decile construct gives an
explicit weight to every scored name; a name the tranche held that is no longer
in the scored cross-section (it left the universe and is not held, or returned
no bars) is set flat in that tranche as well, so stale sub-targets cannot
accumulate. Between its refreshes a tranche carries its sub-targets, so a name
the account no longer holds and cannot price at the decision close can remain
in the account target; its entry is withheld (target zero, its held weight)
and named in the log, as the evaluator does (``prism.evaluate``), rather than
failing the sizing step.

**Wind-down.** ``wind_down=True`` sets each tranche flat on its own due date
instead of re-scoring it, so the book exits over one cycle, a ``1/K`` slice per
session. The unrankable mask is not consulted (nothing is ranked) and the
volatility scale is not decided. The kill switch is unchanged and still stops
everything at once.

**Volatility scale.** With ``vol_scale``, each due tranche's scale is decided
at its refresh from the tranched base book rebuilt point in time from the panel
(:func:`prism.live.vol_sizing.tranched_base_book_window`): the mean of every
tranche's unscaled book as last refreshed, ``sum_j D_j / K`` at raw gross 1.0.
The tranche's sub-targets are its construct times that scale, capped at the
account gross ceiling; since the tranche weights sum to one, a ceiling on each
tranche is a ceiling on the account. A window that cannot be trusted gives
``min(1, the tranche's last good scale)``, or 1 with none. The last good scale
is read from the tranche's persisted state, so only decisions that reached the
write-ahead save bound a later one. Each refresh session appends one row to
``{run-dir}/sizing.jsonl`` (``schedule = "tranches"``).

**Refusals, all before any state moves (N7).** A book other than
``decile_neutral``; a tranche count that does not divide ``decision_every``; the
regime gross-scale hook; a run directory whose persisted tranche layout (count,
ids, offsets) differs from the one requested; a run directory that holds an
untranched book; and a first tranched decision on an account that already holds
positions, because staged entry would sell all but ``1/K`` of them on its first
session. The single book (``tranches=1``) refuses a run directory that holds
tranche states, so a session that forgets ``--tranches`` cannot silently treat a
tranched book as a monthly one.

Production-import-path safe (N8): numpy and pandas only.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.assembly import AccountState, DecisionContext, assemble, heuristic_book_rule
from prism.composition import ComponentPlan
from prism.composition import ComponentState as PlanState
from prism.composition import due_on_grid, tranche_plans
from prism.io import sessions
from prism.live.state import ComponentState, LoopState
from prism.live.vol_sizing import (
    TARGETS_LEDGER_KEY,
    LiveScale,
    scaled_book_rule,
    tranched_base_book_window,
)
from prism.portfolio.vol_scale import SLEEVE_BUFFER, VolScaleRule, decide_scale

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism.live.broker import Broker
    from prism.live.daily import DailyBookConfig
    from prism.signal.base import Signal

logger = logging.getLogger(__name__)

#: The book the tranched schedule is implemented for: the decile book, whose
#: construct is the same at every refresh and whose raw gross is 1.0.
TRANCHED_BOOK = "decile_neutral"

#: How a tranche's component id is spelled (``composition.tranche_plans``).
TRANCHE_ID = re.compile(r"^(?P<book>.+)/t(?P<index>\d{2})$")

#: The key the refresh row of ``targets.jsonl`` carries the tranche record under.
TARGETS_TRANCHES_KEY = "tranches"

#: ``schedule`` of a tranched row in ``sizing.jsonl``.
TRANCHED_SCHEDULE = "tranches"


class TrancheError(ValueError):
    """The tranched book cannot run on these inputs (N7)."""


@dataclass(frozen=True)
class TrancheSession:
    """What one session of the tranched book plans to do, decided before any state moves.

    ``plans`` are the tranches in index order, each with its grid offset;
    ``prior`` the persisted state of every tranche that has one; ``due`` the ids
    that refresh at this decision bar; ``first_decision`` is True when no tranche
    of this book has ever been persisted.
    """

    plans: tuple[ComponentPlan, ...]
    prior: Mapping[str, ComponentState]
    due: tuple[str, ...]
    base_offset: int
    wind_down: bool
    first_decision: bool

    @property
    def count(self) -> int:
        return len(self.plans)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(plan.component_id for plan in self.plans)

    def built(self) -> int:
        """Tranches that have produced an output, as persisted before this session."""
        return sum(1 for state in self.prior.values() if state.last_output_bar is not None)

    def record(self, *, built: int | None = None, deferred: Sequence[str] = ()) -> dict[str, Any]:
        """The session's tranche record (``DailyCycleResult.tranches`` and the targets-ledger row)."""
        return {
            "count": self.count,
            "base_offset": self.base_offset,
            "wind_down": self.wind_down,
            "due": list(self.due),
            "built": self.built() if built is None else int(built),
            "deferred": list(deferred),
        }


@dataclass(frozen=True)
class TrancheDecision:
    """A refresh session's account target and the state it persists.

    ``components`` are every tranche's new persisted state, in index order, for
    the write-ahead save. ``sizing`` is the ``sizing.jsonl`` row (None when the
    volatility scale is off or the book is winding down). ``targets_extra`` is
    what the refresh row of ``targets.jsonl`` carries beside the panel provenance.
    """

    targets: pd.Series
    masked: list[str] | None
    components: tuple[ComponentState, ...]
    deferred: tuple[str, ...]
    sizing: dict[str, Any] | None
    targets_extra: dict[str, Any]
    record: dict[str, Any]


def require_tranche_inputs(
    tranches: int,
    config: "DailyBookConfig",
    *,
    phase_offset: int | None,
    decision_bar: str,
    regime_gross_scale: object | None,
    wind_down: bool,
) -> int | None:
    """The base grid offset of a tranched book, or None for the single book; refuse what cannot run (N7).

    Called before any state moves. ``phase_offset`` has already been checked
    as a residue of the cadence by the cycle when it is given.
    """
    if isinstance(tranches, bool) or not isinstance(tranches, (int, np.integer)) or int(tranches) < 1:
        raise TrancheError(f"tranches must be an integer >= 1, got {tranches!r}")
    count = int(tranches)
    if count == 1:
        if wind_down:
            raise TrancheError(
                "wind_down sets each tranche flat on its own due date, so it needs a tranched book (tranches > 1); "
                "the single book stops through the kill switch"
            )
        return None
    if config.book != TRANCHED_BOOK:
        raise TrancheError(
            f"tranches > 1 is implemented for the {TRANCHED_BOOK!r} book only, whose construct and raw gross are "
            f"the same at every refresh; got book {config.book!r}"
        )
    every = int(config.decision_every)
    if every < 2 or count > every or every % count != 0:
        raise TrancheError(
            f"tranches={count} must divide decision_every={every} (and be at most it), so the tranches sit evenly "
            "on the cycle and one grid offset never carries two of them"
        )
    if regime_gross_scale is not None:
        raise TrancheError(
            "the regime gross-scale hook scales the construct of the session's refresh, which under tranches is one "
            "tranche, so its meaning is undefined here; it is refused with tranches > 1"
        )
    base = 0 if phase_offset is None else int(phase_offset)
    try:
        sessions.grid_position(dt.date.fromisoformat(str(decision_bar)[:10]))
    except ValueError as exc:
        raise TrancheError(
            f"decision bar {decision_bar!r} has no session-grid position, so the tranches cannot be placed: {exc}; "
            "refusing to decide (N7)"
        ) from exc
    return base


def refuse_tranche_state_under_single_book(state: LoopState | None) -> None:
    """The single book refuses a run directory whose state holds tranches (N7).

    The untranched cycle never writes component states, so a tranche state can
    only come from a tranched session; reading such a directory as a monthly
    book would hold the account for a cycle on a refresh anchor that moves
    every session.
    """
    if state is None:
        return
    tranched = sorted(c.component_id for c in state.components if TRANCHE_ID.match(c.component_id))
    if tranched:
        raise TrancheError(
            f"this run directory holds a tranched book ({len(tranched)} tranche state(s): {tranched[:3]}"
            f"{'…' if len(tranched) > 3 else ''}); run it with the tranche count it was built with (--tranches "
            f"{len(tranched)}), adding --wind-down to exit it, rather than as a single book (N7)"
        )


def open_tranche_session(
    signal: "Signal",
    config: "DailyBookConfig",
    tranches: int,
    base_offset: int,
    state: LoopState | None,
    decision_bar: str,
    calendar: Sequence[str],
    *,
    broker: "Broker",
    wind_down: bool,
) -> TrancheSession:
    """Plan the session: the layout, its compatibility with the run directory, and which tranches are due.

    Raises before any state moves when the persisted layout differs from the
    requested one, when the run directory holds an untranched book, or when the
    tranched book's first decision would start from an account that holds
    positions.
    """
    plans = _layout(signal, config, tranches, base_offset)
    ids = [plan.component_id for plan in plans]
    persisted = list(state.components) if state is not None else []
    if persisted:
        _require_same_layout(persisted, plans)
        prior = {component.component_id: component for component in persisted}
    else:
        prior = {}
        if state is not None and state.last_refresh_bar is not None:
            raise TrancheError(
                f"this run directory holds an untranched book (last refresh {state.last_refresh_bar}) and no tranche "
                "state: a tranched book starts empty and builds one tranche per refresh, so it would sell all but "
                f"1/{tranches} of the held book on its first session. Start the tranched book in its own run "
                "directory on a flat account (N7)"
            )
        held = {str(s): float(q) for s, q in broker.positions().items() if float(q) != 0.0}
        if held:
            raise TrancheError(
                f"the tranched book's first decision would start from an account holding {len(held)} position(s) "
                f"({sorted(held)[:5]}{'…' if len(held) > 5 else ''}) that no tranche holds: staged entry targets "
                f"1/{tranches} of the book on its first session and nothing else, so it would sell the rest at "
                "once. Run a tranched book on a flat account (N7)"
            )
    due = tuple(
        plan.component_id
        for plan in plans
        if due_on_grid(
            plan,
            _plan_state(prior.get(plan.component_id), plan.component_id),
            decision_bar,
            elapsed=_elapsed(calendar, prior.get(plan.component_id), decision_bar),
        )
    )
    session = TrancheSession(
        plans=tuple(plans),
        prior=prior,
        due=due,
        base_offset=int(base_offset),
        wind_down=bool(wind_down),
        first_decision=not persisted,
    )
    logger.info(
        "tranches %s: %d of %d due %s (%d built before this session, base offset %d%s)",
        decision_bar,
        len(due),
        len(ids),
        list(due),
        session.built(),
        base_offset,
        ", WIND-DOWN: due tranches go flat" if wind_down else "",
    )
    return session


def decide_tranches(
    session: TrancheSession,
    signal: "Signal",
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    config: "DailyBookConfig",
    *,
    prev_weights: pd.Series,
    equity: float,
    unrankable: Collection[str] | None,
    decision_bar: str,
    vol_scale: VolScaleRule | None,
) -> TrancheDecision:
    """The account target of a refresh session: every tranche composed, the due ones re-scored.

    ``unrankable`` is the mask the caller fetched for this bar (None = no
    provider, or a wind-down, which ranks nothing).
    """
    if not session.due:
        raise TrancheError(f"{decision_bar}: no tranche is due, so there is nothing to decide (N7)")
    max_gross = float(config.max_gross)
    base_rule = heuristic_book_rule(config, close, max_gross)
    decisions: dict[str, LiveScale] = {}
    applied: dict[str, dict[str, float]] = {}
    rules: dict[str, Callable[[pd.Series], pd.Series]] = {}
    refresh_bars: tuple[str, ...] = ()
    sized = vol_scale is not None and not session.wind_down
    if sized:
        assert vol_scale is not None
        returns, refresh_bars = tranched_base_book_window(
            signal,
            close,
            volume,
            config,
            max_gross=max_gross,
            lookback_bars=vol_scale.lookback_bars,
            offsets=[_offset(plan) for plan in session.plans],
        )
        ceiling = min(max_gross, float(vol_scale.max_scale if vol_scale.max_gross is None else vol_scale.max_gross))
        for tid in session.due:
            decision = decide_scale(
                returns,
                vol_scale,
                last_good=_last_good(session.prior.get(tid)),
                decision_bar=decision_bar,
                panel_last_bar=close.index[-1],
            )
            scale = LiveScale(
                rule=vol_scale,
                decision=decision,
                schedule=TRANCHED_SCHEDULE,
                base_refresh_bars=refresh_bars,
                ceiling=ceiling,
            )
            decisions[tid] = scale
            rules[tid], applied[tid] = scaled_book_rule(config, close, max_gross, scale)

    plans: list[ComponentPlan] = []
    for plan in session.plans:
        tid = plan.component_id
        held = tuple((session.prior[tid].sub_targets if tid in session.prior else {}).keys())
        if session.wind_down:
            rule = _flat_rule(held)
        else:
            rule = _whole_book_rule(rules.get(tid, base_rule), held)
        plans.append(replace(plan, weights_fn=rule))

    output = assemble(
        DecisionContext(
            decision_bar=decision_bar,
            components=tuple(plans),
            scoring_panels={"price": close},
            volume=volume,
            construction=config,
            state=AccountState(equity=equity, held_weights=prev_weights, positions=None),
            sessions=[str(bar.date()) for bar in close.index],
            component_state={tid: _plan_state(state, tid) for tid, state in session.prior.items()},
            unrankable=() if unrankable is None else frozenset(unrankable),
            max_gross=max_gross,
        )
    )
    ran = tuple(out.component_id for out in output.component_outputs if out.due)
    if ran != session.due:
        raise TrancheError(
            f"{decision_bar}: the session planned tranches {list(session.due)} and composition refreshed "
            f"{list(ran)}; the two read one predicate, so this is a defect, and nothing is decided on it (N7)"
        )
    targets, deferred = defer_unpriced_entries(output.targets, close.iloc[-1], prev_weights, decision_bar)

    new_states: Mapping[str, PlanState] = output.diagnostics["component_states"]
    components: list[ComponentState] = []
    for plan in session.plans:
        tid = plan.component_id
        prior = session.prior.get(tid)
        if tid in session.due:
            # A due tranche's sub-targets are new: they carry this session's scale
            # decision, or none (unscaled, or flat under a wind-down).
            sizing = _sizing_state(decisions[tid], _last_good(prior)) if tid in decisions else None
        else:
            sizing = None if prior is None else prior.sizing
        components.append(_persisted_state(new_states[tid], offset=_offset(plan), sizing=sizing))

    masked = None if unrankable is None else sorted(output.diagnostics["masked"])
    if masked:
        # As for the single book: a masked name is held out of the due tranches'
        # ranking and carries no decision in them, so each holds its prior sub-target.
        logger.info("unrankable mask: %d name(s) held out of ranking this refresh: %s", len(masked), masked)
    built = sum(1 for component in components if component.last_output_bar is not None)
    record = session.record(built=built, deferred=deferred)
    extra: dict[str, Any] = {
        TARGETS_TRANCHES_KEY: {key: record[key] for key in ("count", "base_offset", "wind_down", "due", "built")}
    }
    row: dict[str, Any] | None = None
    if sized:
        assert vol_scale is not None
        row = _sizing_row(
            vol_scale,
            decision_bar,
            decisions,
            applied,
            refresh_bars,
            targets,
            account_scale=_account_scale(session.plans, components),
        )
        extra[TARGETS_LEDGER_KEY] = {
            "sigma": row["sigma"],
            TARGETS_TRANCHES_KEY: {
                tid: {key: entry[key] for key in ("scale", "status", "reason")}
                for tid, entry in row["tranches"].items()
            },
        }
    return TrancheDecision(
        targets=targets,
        masked=masked,
        components=tuple(components),
        deferred=tuple(deferred),
        sizing=row,
        targets_extra=extra,
        record=record,
    )


def defer_unpriced_entries(
    targets: pd.Series, prices: pd.Series, held_weights: pd.Series, decision_bar: str
) -> tuple[pd.Series, list[str]]:
    """Withhold the target of a name the account does not hold and cannot price at the decision close.

    A tranche carries its sub-targets between its refreshes, so the account
    target can name a stock no tranche is refreshing that has no price on this
    bar: one that left the universe and was never held, or an entry ordered on
    the previous bar that did not fill because the stock stopped trading.
    Sizing refuses a non-flat target without a price (N7), and with nothing held
    there is nothing to keep or exit, so its target is its held weight, zero,
    and the entry waits for a priced bar. ``prism.evaluate`` does the same on a
    grid book. A held name always has a price here: the mark step refused the
    cycle otherwise.
    """
    held = {str(sid) for sid, weight in held_weights.items() if float(weight) != 0.0}
    deferred: list[str] = []
    for sid, weight in targets.items():
        value = float(weight)
        if not math.isfinite(value) or value == 0.0 or str(sid) in held:
            continue
        price = prices.get(sid)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0.0:
            deferred.append(str(sid))
    if not deferred:
        return targets, []
    withheld = targets.copy()
    withheld.loc[deferred] = 0.0
    logger.warning(
        "%s: entries into %d name(s) the account does not hold and cannot price at this close withheld (target "
        "zero, the held weight); a tranche carries a sub-target for each until its next refresh: %s",
        decision_bar,
        len(deferred),
        deferred[:20],
    )
    return withheld, deferred


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _offset(plan: ComponentPlan) -> int:
    """A tranche plan's grid offset; every plan :func:`_layout` builds carries one."""
    if plan.phase_offset_bars is None:
        raise TrancheError(f"tranche {plan.component_id!r} has no grid offset (defect, N7)")
    return int(plan.phase_offset_bars)


def _unbound_rule(scores: pd.Series) -> pd.Series:
    raise TrancheError("a tranche's weight rule is bound at the decision, not at planning (defect, N7)")


def _layout(signal: "Signal", config: "DailyBookConfig", tranches: int, base_offset: int) -> tuple[ComponentPlan, ...]:
    """The tranche plans: ``composition.tranche_plans`` of the book's one plan."""
    every = int(config.decision_every)
    plan = ComponentPlan(
        component=signal,
        component_id=config.book,
        to_weights=config.book,
        weights_fn=_unbound_rule,
        update_every_bars=every,
        valid_for_bars=2 * every,
        phase_offset_bars=int(base_offset),
    )
    return tranche_plans(plan, int(tranches), int(base_offset))


def _require_same_layout(persisted: Sequence[ComponentState], plans: Sequence[ComponentPlan]) -> None:
    """The run directory's tranche layout must be the one requested (N7)."""
    found = [(c.component_id, c.phase_offset_bars) for c in persisted]
    wanted = [(p.component_id, _offset(p)) for p in plans]
    if sorted(found, key=str) != sorted(wanted, key=str):
        raise TrancheError(
            f"this run directory's persisted tranche layout {found[:4]}{'…' if len(found) > 4 else ''} "
            f"({len(found)} state(s)) is not the requested {wanted[:4]}{'…' if len(wanted) > 4 else ''} "
            f"({len(wanted)} tranche(s)); a different tranche count, book or base offset is a different book, "
            "so it needs its own run directory (N7)"
        )


def _elapsed(calendar: Sequence[str], state: ComponentState | None, decision_bar: str) -> int:
    """Sessions after the tranche's last output through the decision bar, on the panel's calendar.

    The rule composition counts with (``composition._bars_elapsed``).
    """
    last = None if state is None else state.last_output_bar
    if last is None:
        return 0
    return sum(1 for bar in calendar if last < bar <= decision_bar)


def _plan_state(state: ComponentState | None, component_id: str) -> PlanState:
    """The composition state of a persisted tranche (an empty one when it has never decided)."""
    if state is None:
        return PlanState(component_id=component_id)
    return PlanState(
        component_id=state.component_id,
        last_output_bar=state.last_output_bar,
        decision_bar=state.decision_bar,
        expires_bar=state.expires_bar,
        expired=state.expired,
        sub_targets=dict(state.sub_targets),
        fitted_state_ref=state.fitted_state_ref,
        fitted_through_bar=state.fitted_through_bar,
        fit_scope=state.fit_scope,
    )


def _persisted_state(state: PlanState, *, offset: int, sizing: dict[str, Any] | None) -> ComponentState:
    return ComponentState(
        component_id=state.component_id,
        last_output_bar=state.last_output_bar,
        decision_bar=state.decision_bar,
        expires_bar=state.expires_bar,
        expired=bool(state.expired),
        sub_targets={str(sid): float(value) for sid, value in state.sub_targets.items()},
        fitted_state_ref=state.fitted_state_ref,
        fitted_through_bar=state.fitted_through_bar,
        fit_scope=state.fit_scope,
        phase_offset_bars=int(offset),
        sizing=None if sizing is None else dict(sizing),
    )


def _whole_book_rule(
    rule: Callable[[pd.Series], pd.Series], held: Collection[str]
) -> Callable[[pd.Series], pd.Series]:
    """``rule``, with an explicit flat for every name the tranche holds that the cross-section lacks."""

    def whole(scores: pd.Series) -> pd.Series:
        values = rule(scores)
        missing = [sid for sid in held if sid not in values.index]
        if not missing:
            return values
        return pd.concat([values.astype(float), pd.Series(0.0, index=pd.Index(missing, dtype=object), dtype=float)])

    return whole


def _flat_rule(held: Collection[str]) -> Callable[[pd.Series], pd.Series]:
    """The wind-down rule: every scored name and every name the tranche holds, flat."""

    def flat(scores: pd.Series) -> pd.Series:
        names = list(dict.fromkeys([*map(str, scores.index), *held]))
        return pd.Series(0.0, index=pd.Index(names, dtype=object), dtype=float)

    return flat


def _last_good(state: ComponentState | None) -> float | None:
    if state is None or state.sizing is None:
        return None
    value = state.sizing.get("last_good")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TrancheError(f"tranche {state.component_id} persists a last good scale of {value!r} (N7)")
    if float(value) <= 0.0:
        raise TrancheError(f"tranche {state.component_id} persists a non-positive last good scale {value!r} (N7)")
    return float(value)


def _sizing_state(scale: LiveScale, prior_last_good: float | None) -> dict[str, Any]:
    """What a due tranche persists about the scale its new sub-targets were sized with."""
    decision = scale.decision
    return {
        "scale": _plain(decision.scale),
        "status": decision.status,
        "reason": decision.reason,
        "sigma": _plain(decision.sigma),
        "last_good": _plain(decision.scale) if decision.status == "ok" else prior_last_good,
    }


def _account_scale(plans: Sequence[ComponentPlan], components: Sequence[ComponentState]) -> float | None:
    """The plan-weighted mean scale of the tranches holding a sub-target (an unscaled tranche at 1)."""
    weights = {plan.component_id: abs(float(plan.weight)) for plan in plans}
    total = 0.0
    weighted = 0.0
    for component in components:
        if not any(float(value) != 0.0 for value in component.sub_targets.values()):
            continue
        scale = 1.0 if component.sizing is None else float(component.sizing.get("scale") or 1.0)
        total += weights[component.component_id]
        weighted += weights[component.component_id] * scale
    return None if total <= 0.0 else weighted / total


def _sizing_row(
    rule: VolScaleRule,
    decision_bar: str,
    decisions: Mapping[str, LiveScale],
    applied: Mapping[str, Mapping[str, float]],
    refresh_bars: Sequence[str],
    targets: pd.Series,
    *,
    account_scale: float | None,
) -> dict[str, Any]:
    """The ``sizing.jsonl`` row of a tranched refresh session.

    The window statistics (sigma, finite returns, zero fraction) are the shared
    base book's and do not depend on the tranche; each due tranche's decision is
    under ``tranches`` and the target it describes under ``tranche_targets``.
    """
    first = next(iter(decisions.values()))
    values = pd.to_numeric(pd.Series(targets, dtype=float), errors="coerce").to_numpy(dtype=float)
    finite = np.where(np.isfinite(values), values, 0.0)
    long_side = float(np.clip(finite, 0.0, None).sum())
    return {
        "decision_bar": decision_bar,
        "rule": rule.to_dict(),
        "schedule": TRANCHED_SCHEDULE,
        "base_refresh_bars": list(refresh_bars),
        "sigma": _plain(first.decision.sigma),
        "n_valid": int(first.decision.n_valid),
        "zero_fraction": _plain(first.decision.zero_fraction),
        "tranches": {
            tid: {
                "scale": _plain(scale.decision.scale),
                "status": scale.decision.status,
                "reason": scale.decision.reason,
                "last_good": _plain(scale.decision.last_good),
            }
            for tid, scale in decisions.items()
        },
        "tranche_targets": {
            tid: {
                "unscaled_gross": _plain(applied[tid].get("unscaled_gross", math.nan)),
                "ceiling_factor": _plain(applied[tid].get("ceiling_factor", math.nan)),
            }
            for tid in decisions
        },
        "ceiling": first.ceiling,
        "account_scale": None if account_scale is None else float(account_scale),
        "target_gross": float(np.abs(finite).sum()),
        "long_over_equity": long_side,
        "sleeve_fraction": max(0.0, 1.0 - SLEEVE_BUFFER - long_side),
        "sleeve_buffer": SLEEVE_BUFFER,
    }


def _plain(value: Any) -> Any:
    """A JSON-safe scalar: NaN and infinities become None, numpy scalars plain Python."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value
