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

**Holdings in shares.** Each tranche is held in shares between its refreshes,
the way overlapping Jegadeesh-Titman cohorts are held and the way the evaluator
holds them (``prism.evaluate._TrancheHoldings``; the registered rule is
``docs/public_baseline_design.md`` §1.1). At tranche *j*'s refresh at close *t*
its holding is reset to ``(1/K) * w_j,i * E_t / P_t,i`` shares of each name
*i*, where ``w_j`` is its book (times its volatility scale, under ``vol_scale``),
``E_t`` the marked equity and ``P_t,i`` the decision close. Between refreshes the
shares are held and drift with prices. The account target at close *t* is
``sum_j shares_j * P_t / E_t``, but a session decides only the names the
refreshing tranche held or now holds, plus any name whose account shares differ
from the tranches' total by more than half a share (an unfilled order, a share
lost to rounding, or a position no tranche holds); every other name carries no
decision and is held. So a session's orders are the refreshing tranche's change
and those catch-ups, never a rebalance of the drifted book back to constant
weights. The cash-holding ETF (``exclude``) is not the tranches' to decide.

The holdings are persisted per tranche (``ComponentState.holdings``) with the
session and decision close each name was sized at. A split between that session
and the decision bar shows in the panel, whose bars are split-adjusted to their
last session (the live source's ``adjustment=split``): the close the panel now
shows for the sizing session differs from the one recorded, and the shares are
restated by the ratio, as the evaluator restates them by the account's split
factor. Bars adjusted for dividends as well would restate holdings for
dividends, which is wrong; the live source does not serve those.

With the volatility scale on, the account target the holdings make is capped at
the account gross ceiling; when the cap binds, every tranche's holding is scaled
by the same factor and every name is decided on that session, as in the
evaluator. Without it the account target is not capped between refreshes, as
the single book is not.

**Schedule and state.** The due-ness predicate is
:func:`prism.composition.due_on_grid`, the one the evaluator and composition
use: a tranche with no output yet waits for its first grid date (staged entry,
so a new book reaches full size over ``decision_every`` sessions instead of
buying it on day one), and a tranche that missed its grid date catches up once
on the first session after more than ``decision_every`` have elapsed. Each
tranche's output is valid for ``2 * decision_every`` sessions, so a missed date
makes a tranche late, not expired, unless the loop was dark for more than two
cycles, in which case the tranche goes flat and is rebuilt. The tranche states
(schedule anchor, sub-targets, holding in shares, grid offset and, with the
volatility scale, the scale decision) are ``LoopState.components``, persisted in
the same write-ahead
save that advances ``last_refresh_bar``, so a restart in the middle of the build
resumes the same schedule and never rebuilds a tranche twice.

**A tranche's refresh is its whole book.** The decile construct gives an
explicit weight to every scored name; a name the tranche held that is no longer
in the scored cross-section (it left the universe and is not held, or returned
no bars) is set flat in that tranche as well, so stale holdings cannot
accumulate. A name the refreshing tranche targets that has no price at the
decision close keeps the tranche's previous holding of it, or, if it held none,
is not entered and is named in the log and the session record (``deferred``),
as the evaluator does. A name masked as unrankable keeps the refreshing
tranche's previous holding and is never entered.

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
account gross ceiling, so each tranche is sized within the ceiling at its
refresh; the account target its held shares make is capped again as they drift
(above). A window that cannot be trusted gives
``min(1, the tranche's last good scale)``, or 1 with none. The last good scale
is read from the tranche's persisted state, so only decisions that reached the
write-ahead save bound a later one. Each refresh session appends one row to
``{run-dir}/sizing.jsonl`` (``schedule = "tranches"``).

**Refusals, all before any state moves (N7).** A book other than
``decile_neutral``; a tranche count that does not divide ``decision_every``; the
regime gross-scale hook; a run directory whose persisted tranche layout (count,
ids, offsets) differs from the one requested; a run directory that holds an
untranched book; a built tranche state that carries no share holdings (written
before tranches were held in shares, so its holdings are unknown); and a first
tranched decision on an account that already holds positions other than the
cash-holding ETF, because staged entry would sell all but ``1/K`` of them on its
first session. The single book (``tranches=1``) refuses a run directory that holds
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

from prism.assembly import construct_targets, heuristic_book_rule
from prism.composition import ComponentPlan
from prism.composition import ComponentState as PlanState
from prism.composition import compose, due_on_grid, tranche_plans
from prism.io import sessions
from prism.live.state import ComponentState, LoopState
from prism.live.vol_sizing import (
    TARGETS_LEDGER_KEY,
    LiveScale,
    scaled_book_rule,
    tranched_base_book_window,
)
from prism.portfolio.vol_scale import (
    SLEEVE_BUFFER,
    VolScaleRule,
    cap_gross,
    decide_scale,
)

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

    def record(
        self,
        *,
        built: int | None = None,
        deferred: Sequence[str] = (),
        decided: int = 0,
        catch_up: Sequence[str] = (),
        ceiling_factor: float = 1.0,
    ) -> dict[str, Any]:
        """The session's tranche record (``DailyCycleResult.tranches`` and the targets-ledger row).

        ``decided`` counts the names the session decided, ``catch_up`` names
        those decided only because the account's shares differ from the
        tranches' total, and ``ceiling_factor`` is below 1 when the account
        target was capped at the gross ceiling. A session with no tranche due
        decides nothing.
        """
        return {
            "count": self.count,
            "base_offset": self.base_offset,
            "wind_down": self.wind_down,
            "due": list(self.due),
            "built": self.built() if built is None else int(built),
            "deferred": list(deferred),
            "decided": int(decided),
            "catch_up": list(catch_up),
            "ceiling_factor": float(ceiling_factor),
        }


@dataclass(frozen=True)
class TrancheDecision:
    """A refresh session's decision, the account target the tranches' holdings make, and the state it persists.

    ``targets`` is what the session sizes orders from: the decided names at the
    account target, every other name at its held weight (no decision). ``full``
    is the account target at this close, ``sum_j shares_j * P / E`` for every
    name a tranche holds or the composition scored (NaN where a held name has
    no price), capped at the ceiling when the volatility scale is on.
    ``decided`` names what this session decides. ``components`` are every
    tranche's new persisted state, in index order, for the write-ahead save.
    ``sizing`` is the ``sizing.jsonl`` row (None when the volatility scale is off
    or the book is winding down). ``targets_extra`` is what the row of
    ``targets.jsonl`` carries beside the panel provenance.
    """

    targets: pd.Series
    full: pd.Series
    decided: tuple[str, ...]
    masked: list[str] | None
    components: tuple[ComponentState, ...]
    deferred: tuple[str, ...]
    sizing: dict[str, Any] | None
    targets_extra: dict[str, Any]
    record: dict[str, Any]

    def recorded_targets(self, targets: pd.Series) -> pd.Series:
        """The account target for ``targets.jsonl``: ``targets`` with each undecided name at its ``full`` value.

        ``targets`` is the series the session sized from (after the T-bill
        holding, when there is one). An undecided name is held, and the
        tranches' holdings put it at ``full``; recording that rather than the
        held weight makes the row the account target, before whole-share
        rounding, as the evaluator records it.
        """
        recorded = targets.astype(float).copy()
        decided = set(self.decided)
        held = [sid for sid in self.full.index if sid not in decided]
        for sid in held:
            recorded.loc[sid] = float(self.full[sid])
        return recorded


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


@dataclass(frozen=True)
class TrancheStateReport:
    """What a run directory's persisted tranche states say, and what is wrong with them.

    ``cycle`` is the cadence the offsets imply (``spacing * count``), ``built``
    the tranches that have produced an output, ``gross`` the account gross of
    the summed sub-targets at weight ``1/count``. ``problems`` is empty for a
    state the tranched cycle could have written.
    """

    count: int
    book: str | None
    base_offset: int | None
    cycle: int | None
    built: int
    gross: float | None
    problems: tuple[str, ...]


def inspect_tranche_state(state: LoopState | None) -> TrancheStateReport | None:
    """Check the tranche states of ``state`` against what the tranched cycle writes; None when it holds none.

    Read-only, for ``prism-doctor``; the cycle never calls it. The checks are
    the invariants the cycle relies on or establishes: ids ``<book>/tNN`` for
    one book with indices ``0..K-1``; grid offsets ``(base + j * s) % (s * K)``
    (:func:`prism.composition.tranche_plans`, where ``s`` is the spacing and
    ``s * K`` the cadence, inferred from the offsets because the cadence is not
    persisted); finite sub-targets; share holdings on every tranche that holds
    a sub-target (the cycle refuses a state without them), none sized after the
    state's last refresh; a scale record whose last good scale is finite and
    positive (the cycle refuses otherwise); output bars no later than the
    state's last refresh; expiry bars no earlier than the output; and no other
    component kind in the same state.
    """
    if state is None:
        return None
    tranched = [c for c in state.components if TRANCHE_ID.match(c.component_id)]
    if not tranched:
        return None
    problems: list[str] = []
    others = sorted(c.component_id for c in state.components if not TRANCHE_ID.match(c.component_id))
    if others:
        problems.append(f"the state mixes tranche states with other components {others[:3]}")

    parsed = [(TRANCHE_ID.match(c.component_id), c) for c in tranched]
    books = sorted({match.group("book") for match, _ in parsed if match is not None})
    book = books[0] if len(books) == 1 else None
    if len(books) != 1:
        problems.append(f"tranche ids name {len(books)} books {books}; one book is tranched at a time")
    elif book != TRANCHED_BOOK:
        problems.append(f"tranches of book {book!r}; the tranched cycle is implemented for {TRANCHED_BOOK!r}")
    by_index: dict[int, ComponentState] = {}
    for match, component in parsed:
        assert match is not None
        index = int(match.group("index"))
        if index in by_index:
            problems.append(f"tranche index {index:02d} appears more than once")
        by_index[index] = component
    count = len(tranched)
    if sorted(by_index) != list(range(count)):
        problems.append(f"tranche indices {sorted(by_index)} are not 00..{count - 1:02d}")
    if count < 2:
        problems.append("a single tranche state; tranche_plans gives one tranche the book's own id")

    base_offset: int | None = None
    cycle: int | None = None
    offsets = {index: component.phase_offset_bars for index, component in by_index.items()}
    unplaced = sorted(index for index, offset in offsets.items() if offset is None)
    if unplaced:
        problems.append(f"tranche(s) {[f't{i:02d}' for i in unplaced]} persist no grid offset")
    elif count >= 2 and sorted(by_index) == list(range(count)):
        base_offset = int(offsets[0])  # type: ignore[arg-type]
        ordered = sorted(int(offset) for offset in offsets.values())  # type: ignore[arg-type]
        gaps = {b - a for a, b in zip(ordered, ordered[1:])}
        spacing = gaps.pop() if len(gaps) == 1 else None
        if spacing is None or spacing < 1 or ordered[0] >= spacing:
            problems.append(f"grid offsets {ordered} are not evenly spaced on one cycle")
        else:
            cycle = spacing * count
            wrong = [
                f"t{j:02d}={offsets[j]} (expected {(base_offset + j * spacing) % cycle})"
                for j in range(count)
                if offsets[j] != (base_offset + j * spacing) % cycle
            ]
            if wrong:
                problems.append(
                    f"grid offsets do not follow base {base_offset} + j*{spacing} mod {cycle}: {wrong[:3]}"
                )

    built = 0
    gross = 0.0
    finite_gross = True
    for index in sorted(by_index):
        component = by_index[index]
        tid = component.component_id
        if component.last_output_bar is not None:
            built += 1
        bad = sorted(
            str(sid)
            for sid, value in component.sub_targets.items()
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        )
        if bad:
            problems.append(f"{tid} carries non-finite sub-targets for {bad[:3]}")
            finite_gross = False
        else:
            gross += sum(abs(float(value)) for value in component.sub_targets.values()) / count
        last_refresh = state.last_refresh_bar
        holds_target = any(
            not isinstance(value, bool) and isinstance(value, (int, float)) and float(value) != 0.0
            for value in component.sub_targets.values()
        )
        if component.holdings is None and holds_target:
            problems.append(f"{tid} carries sub-targets and persists no share holdings")
        elif component.holdings is not None and last_refresh is not None:
            late = sorted(sid for sid, entry in component.holdings.items() if str(entry["bar"]) > str(last_refresh))
            if late:
                problems.append(
                    f"{tid} holds {late[:3]} in shares sized after the state's last refresh {last_refresh}"
                )
        if component.sizing is not None:
            try:
                _last_good(component)
            except TrancheError as exc:
                problems.append(str(exc))
        if component.last_output_bar is not None and last_refresh is not None:
            if str(component.last_output_bar) > str(last_refresh):
                problems.append(
                    f"{tid} last output {component.last_output_bar} is after the state's last refresh {last_refresh}"
                )
        if component.last_output_bar is not None and component.expires_bar is not None:
            if str(component.expires_bar) < str(component.last_output_bar):
                problems.append(
                    f"{tid} expires {component.expires_bar}, before its output {component.last_output_bar}"
                )
    return TrancheStateReport(
        count=count,
        book=book,
        base_offset=base_offset,
        cycle=cycle,
        built=built,
        gross=gross if finite_gross else None,
        problems=tuple(problems),
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
    exclude: Collection[str] = (),
) -> TrancheSession:
    """Plan the session: the layout, its compatibility with the run directory, and which tranches are due.

    Raises before any state moves when the persisted layout differs from the
    requested one, when the run directory holds an untranched book, when a
    built tranche persists no share holdings, or when the tranched book's first
    decision would start from an account that holds positions. ``exclude``
    names positions that are not the tranches' (the cash-holding ETF), which
    the flat-account check ignores: the ETF is traded on every session,
    including those before the first tranche is due.
    """
    plans = _layout(signal, config, tranches, base_offset)
    ids = [plan.component_id for plan in plans]
    persisted = list(state.components) if state is not None else []
    if persisted:
        _require_same_layout(persisted, plans)
        _require_share_holdings(persisted)
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
        excluded = {str(sid) for sid in exclude}
        held = {
            str(s): float(q) for s, q in broker.positions().items() if float(q) != 0.0 and str(s) not in excluded
        }
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
    positions: Mapping[str, float],
    equity: float,
    unrankable: Collection[str] | None,
    decision_bar: str,
    vol_scale: VolScaleRule | None,
    exclude: Collection[str] = (),
) -> TrancheDecision:
    """The decision of a refresh session: the due tranches re-scored and re-held in shares, the rest held.

    ``positions`` are the account's shares at the decision close and
    ``prev_weights`` the same book as weights of ``equity``. ``unrankable`` is
    the mask the caller fetched for this bar (None = no provider, or a
    wind-down, which ranks nothing). ``exclude`` names symbols that are not the
    tranches' to decide (the cash-holding ETF): they are neither caught up nor
    flattened here.
    """
    if not session.due:
        raise TrancheError(f"{decision_bar}: no tranche is due, so there is nothing to decide (N7)")
    if not (math.isfinite(float(equity)) and float(equity) > 0.0):
        raise TrancheError(f"{decision_bar}: a tranched book needs a positive marked equity, got {equity!r} (N7)")
    max_gross = float(config.max_gross)
    base_rule = heuristic_book_rule(config, close, max_gross)
    decisions: dict[str, LiveScale] = {}
    applied: dict[str, dict[str, float]] = {}
    rules: dict[str, Callable[[pd.Series], pd.Series]] = {}
    refresh_bars: tuple[str, ...] = ()
    ceiling: float | None = None
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
        prior_state = session.prior.get(tid)
        held = () if prior_state is None else tuple(dict.fromkeys([*prior_state.sub_targets, *_entries(prior_state)]))
        if session.wind_down:
            rule = _flat_rule(held)
        else:
            rule = _whole_book_rule(rules.get(tid, base_rule), held)
        plans.append(replace(plan, weights_fn=rule))

    composition = compose(
        tuple(plans),
        decision_bar=decision_bar,
        scoring_panels={"price": close},
        sessions=[str(bar.date()) for bar in close.index],
        volume=volume,
        states={tid: _plan_state(state, tid) for tid, state in session.prior.items()},
        unrankable=() if unrankable is None else frozenset(unrankable),
    )
    ran = tuple(out.component_id for out in composition.component_outputs if out.due)
    if ran != session.due:
        raise TrancheError(
            f"{decision_bar}: the session planned tranches {list(session.due)} and composition refreshed "
            f"{list(ran)}; the two read one predicate, so this is a defect, and nothing is decided on it (N7)"
        )

    held_book = _hold_in_shares(
        session,
        composition,
        close,
        positions=positions,
        equity=float(equity),
        decision_bar=decision_bar,
        whole_shares=bool(config.whole_shares),
        cap=ceiling,
        exclude=exclude,
    )
    targets = construct_targets(
        held_book.row,
        prev_weights=prev_weights,
        close=close,
        volume=volume,
        config=config,
        equity=float(equity),
        liquidate=composition.liquidate,
        max_gross=max_gross,
    )

    new_states: Mapping[str, PlanState] = composition.states
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
        components.append(
            _persisted_state(
                new_states[tid], offset=_offset(plan), sizing=sizing, holdings=held_book.holdings.get(tid)
            )
        )

    masked: list[str] | None = None
    if unrankable is not None:
        masked = sorted({str(sid) for out in composition.component_outputs for sid in out.masked})
    if masked:
        # As for the single book: a masked name is held out of the due tranches'
        # ranking and carries no decision in them, so each keeps its prior holding.
        logger.info("unrankable mask: %d name(s) held out of ranking this refresh: %s", len(masked), masked)
    deferred = list(held_book.deferred)
    if deferred:
        logger.warning(
            "%s: entries into %d name(s) a refreshing tranche targets and that have no price at this close are "
            "deferred: that tranche does not hold them until its next refresh: %s",
            decision_bar,
            len(deferred),
            deferred[:20],
        )
    built = sum(1 for component in components if component.last_output_bar is not None)
    record = session.record(
        built=built,
        deferred=deferred,
        decided=len(held_book.decided),
        catch_up=held_book.catch_up,
        ceiling_factor=held_book.ceiling_factor,
    )
    logger.info(
        "tranches %s: %d name(s) decided (%d caught up%s), every other name held; account target gross %.4f",
        decision_bar,
        len(held_book.decided),
        len(held_book.catch_up),
        "" if held_book.ceiling_factor == 1.0 else f", ceiling binding at {held_book.ceiling_factor:.6f}",
        float(np.nansum(np.abs(held_book.full.to_numpy(dtype=float)))),
    )
    extra: dict[str, Any] = {
        TARGETS_TRANCHES_KEY: {
            key: record[key]
            for key in ("count", "base_offset", "wind_down", "due", "built", "decided", "catch_up", "ceiling_factor")
        }
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
            held_book.full,
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
        full=held_book.full,
        decided=held_book.decided,
        masked=masked,
        components=tuple(components),
        deferred=tuple(deferred),
        sizing=row,
        targets_extra=extra,
        record=record,
    )


@dataclass(frozen=True)
class _HeldBook:
    """One session's tranche holdings in shares and the decision row they make.

    ``holdings`` are every tranche's persisted holdings after this session (the
    refreshing ones reset, the rest unchanged, all scaled when the ceiling
    binds). ``full`` is the account target they make at this close; ``row`` is
    ``full`` on the decided names and NaN (no decision) elsewhere.
    """

    holdings: dict[str, dict[str, dict[str, Any]]]
    full: pd.Series
    row: pd.Series
    decided: tuple[str, ...]
    catch_up: tuple[str, ...]
    deferred: tuple[str, ...]
    ceiling_factor: float


def _hold_in_shares(
    session: TrancheSession,
    composition: Any,
    close: pd.DataFrame,
    *,
    positions: Mapping[str, float],
    equity: float,
    decision_bar: str,
    whole_shares: bool,
    cap: float | None,
    exclude: Collection[str],
) -> _HeldBook:
    """Reset the refreshing tranches' holdings, restate the rest, and say which names this session decides.

    The live counterpart of ``prism.evaluate._TrancheHoldings.target``: a
    refreshing tranche (due, or its output expired) holds ``weight * sub_target
    * E / P`` shares of each name it targets, a name without a price keeps its
    previous holding or is deferred, and a masked name keeps its previous
    holding. The decided names are those a refreshing tranche held or now holds
    and those whose account shares differ from the tranches' total by more than
    half a share (or, for fractional shares, by more than rounding); all names
    when the ceiling binds.
    """
    excluded = {str(sid) for sid in exclude}
    prices = close.iloc[-1]
    row_of = {str(ts.date()): number for number, ts in enumerate(close.index)}
    outputs = {str(out.component_id): out for out in composition.component_outputs}
    refreshed = {tid for tid, out in outputs.items() if out.due or out.expired}

    holdings: dict[str, dict[str, dict[str, Any]]] = {}
    touched: set[str] = set()
    deferred: list[str] = []
    for plan in session.plans:
        tid = str(plan.component_id)
        previous = _entries(session.prior.get(tid))
        if tid not in refreshed:
            holdings[tid] = {sid: dict(entry) for sid, entry in previous.items()}
            continue
        touched.update(previous)
        masked = {str(sid) for sid in outputs[tid].masked}
        fresh: dict[str, dict[str, Any]] = {}
        for sid, value in composition.states[tid].sub_targets.items():
            key = str(sid)
            weight = float(value)
            if key in excluded:
                if math.isfinite(weight) and weight != 0.0:
                    raise TrancheError(
                        f"{decision_bar}: tranche {tid} targets {key}, which is not the tranches' to hold (the cash "
                        "holding); the ETF must stay out of the scored cross-section (N7)"
                    )
                continue
            if key in masked or not math.isfinite(weight):
                # No decision on this name in this tranche: it keeps what it held, and is never entered.
                if key in previous:
                    fresh[key] = dict(previous[key])
                continue
            if weight == 0.0:
                continue
            price = _positive_price(prices.get(key))
            if price is not None:
                shares = float(plan.weight) * weight * equity / price
                fresh[key] = {"shares": shares, "bar": decision_bar, "close": price}
            elif key in previous:
                fresh[key] = dict(previous[key])
            else:
                deferred.append(key)
        for key in sorted(masked & set(previous) - set(fresh)):
            fresh[key] = dict(previous[key])
        holdings[tid] = fresh
        touched.update(fresh)

    restated: dict[str, float] = {}
    totals: dict[str, float] = {}
    for tid, entries in holdings.items():
        for sid, entry in entries.items():
            shares = _current_shares(sid, entry, close, row_of, decision_bar, prices, restated)
            totals[sid] = totals.get(sid, 0.0) + shares
    if restated:
        logger.warning(
            "%s: tranche holdings restated for a change in the panel's close at their sizing session (a split): %s",
            decision_bar,
            sorted(restated.items())[:10],
        )

    scored = composition.targets
    names = [
        sid
        for sid in dict.fromkeys([*(str(sid) for sid in scored.index), *totals, *touched, *deferred])
        if sid not in excluded
    ]
    values = np.zeros(len(names), dtype=float)
    for number, sid in enumerate(names):
        if sid in scored.index and not math.isfinite(float(scored[sid])):
            values[number] = np.nan
        total = totals.get(sid, 0.0)
        if total != 0.0:
            price = _positive_price(prices.get(sid))
            values[number] = np.nan if price is None else total * price / equity
        elif sid in deferred:
            values[number] = np.nan
    full = pd.Series(values, index=pd.Index(names, dtype=object), dtype=float)

    ceiling_factor = 1.0
    if cap is not None:
        full, ceiling_factor = cap_gross(full, float(cap))
        if ceiling_factor != 1.0:
            for entries in holdings.values():
                for entry in entries.values():
                    entry["shares"] = float(entry["shares"]) * ceiling_factor
            totals = {sid: total * ceiling_factor for sid, total in totals.items()}

    account = {str(sid): float(q) for sid, q in positions.items() if float(q) != 0.0 and str(sid) not in excluded}
    catch_up: list[str] = []
    slot = set(full.index)
    for sid in sorted(set(totals) | set(account)):
        if sid in touched or sid not in slot:
            continue
        planned, held = totals.get(sid, 0.0), account.get(sid, 0.0)
        gap = abs(planned - held)
        if (whole_shares and gap > 0.5) or (not whole_shares and gap > 1e-9 * max(1.0, abs(planned))):
            catch_up.append(sid)
    if ceiling_factor != 1.0:
        decided = tuple(full.index)
    else:
        decided = tuple(sid for sid in full.index if sid in touched or sid in catch_up)
    row = pd.Series(np.nan, index=full.index, dtype=float)
    if decided:
        row.loc[list(decided)] = full.loc[list(decided)].to_numpy(dtype=float)
    return _HeldBook(
        holdings=holdings,
        full=full,
        row=row,
        decided=decided,
        catch_up=tuple(catch_up),
        deferred=tuple(dict.fromkeys(deferred)),
        ceiling_factor=float(ceiling_factor),
    )


def _current_shares(
    sid: str,
    entry: Mapping[str, Any],
    close: pd.DataFrame,
    row_of: Mapping[str, int],
    decision_bar: str,
    prices: pd.Series,
    restated: dict[str, float],
) -> float:
    """A holding's shares at the decision close, restated for any split since it was sized.

    The panel's bars are split-adjusted to its last session, so the close it now
    shows for the sizing session is the recorded close divided by every split
    since; their ratio restates the shares. A name with no price at the decision
    close has no target and is not restated. A priced name whose sizing session
    is missing from the panel, or unpriced there, cannot be checked and is
    refused (N7): a missed split would size an order off by the split ratio.
    """
    shares = float(entry["shares"])
    bar = str(entry["bar"])
    if bar == decision_bar or _positive_price(prices.get(sid)) is None:
        return shares
    at = row_of.get(bar)
    then = None if at is None or sid not in close.columns else _positive_price(close[sid].iat[at])
    if then is None:
        raise TrancheError(
            f"{decision_bar}: a tranche holds {sid} in shares sized at {bar}, and this panel has no close for it "
            "there, so a split since then cannot be checked; refusing to size against it (N7)"
        )
    factor = float(entry["close"]) / then
    if factor != 1.0:
        restated[sid] = factor
    return shares * factor


def _positive_price(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _entries(state: ComponentState | None) -> dict[str, dict[str, Any]]:
    """A persisted tranche's holdings in shares (none for a tranche that has never held any)."""
    if state is None or state.holdings is None:
        return {}
    return state.holdings


def _require_share_holdings(persisted: Sequence[ComponentState]) -> None:
    """Every built tranche persists its holdings in shares (N7).

    A state written before tranches were held in shares carries sub-targets and
    no holdings; reading it as empty would sell the whole book as catch-ups.
    """
    missing = sorted(
        c.component_id
        for c in persisted
        if c.holdings is None and any(float(value) != 0.0 for value in c.sub_targets.values())
    )
    if missing:
        raise TrancheError(
            f"tranche state(s) {missing[:3]}{'…' if len(missing) > 3 else ''} carry sub-targets and no share "
            "holdings: they were written before the tranched book held its tranches in shares, so what each holds "
            "is unknown. Wind the book down or flatten the account and start a new run directory (N7)"
        )


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


def _persisted_state(
    state: PlanState,
    *,
    offset: int,
    sizing: dict[str, Any] | None,
    holdings: Mapping[str, Mapping[str, Any]] | None,
) -> ComponentState:
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
        holdings=None if holdings is None else {str(sid): dict(entry) for sid, entry in holdings.items()},
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
    ``targets`` is the account target the tranches' holdings make at this close.
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
