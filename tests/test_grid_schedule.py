"""The session-grid cadence and overlapping tranches (composition, evaluation, specification).

A component either refreshes by elapsed sessions, so its first output sets its
phase, or it sits on the session grid: due where ``(grid_position - offset) %
update_every_bars == 0``. The grid is opt-in. Every case that matters is pinned
here:

* the grid itself: a fixed epoch, one position per session, no position for a
  closed day;
* the one predicate (``prism.composition.due_on_grid``) on its own, against a
  calendar with an outage, and through ``compose`` and ``evaluate._due``, which
  must agree on every bar;
* tranches: exactly one of 21 refreshes on every session for two years, a new
  book is built one tranche per session, and a time-invariant signal's
  tranched book converges to the untranched one;
* the specification: the two fields enter ``config_id`` only when declared,
  so every shipped example keeps the identity it had before they existed, and
  tranches are refused under the calibrated flow and under renormalization;
* the engine: a single-phase book placed on the grid at its own phase is the
  same book, bit for bit, and a tranched book refreshes one tranche per bar;
* the refusals: the paths that cannot honour a grid schedule say so (N7).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prism.evaluate as evaluate_module
from prism.cli import EXIT_REFUSED, main
from prism.composition import (
    ComponentPlan,
    ComponentState,
    CompositionError,
    compose,
    due_on_grid,
    tranche_plans,
)
from prism.evaluate import (
    CanonicalSettings,
    EvaluationError,
    _due,
    smoke_evaluate,
    walk_forward_evaluate,
)
from prism.execution.cost_model import CostStack
from prism.io import sessions
from prism.io.synthetic import write_synthetic_dataset
from prism.signal.base import Signal
from prism.spec import SpecError, load_spec, load_spec_table
from prism.template import write_template
from prism.validation.trials import stable_json_dumps
from prism.validation.walk_forward import RollingWalkForward
from prism.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Two years of the exchange calendar, as decision-bar labels.
TWO_YEARS = [day.isoformat() for day in sessions.sessions_between(dt.date(2024, 1, 1), dt.date(2025, 12, 31))]


class ConstantRank(Signal):
    """A time-invariant score that counts its own calls."""

    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    price_scale_invariant = True
    fitted_state = "none"
    estimated_decay_bars = None

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.score_calls = 0

    @property
    def horizon_bars(self) -> int:
        return 5

    @property
    def required_history(self) -> int:
        return 1

    def fit(self, close, volume=None) -> "ConstantRank":
        return self

    def score(self, close, volume=None) -> pd.Series:
        self.score_calls += 1
        return pd.Series(self.scores, dtype=float).reindex(close.columns)


SCORES = {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0}


def _plan(component: Signal | None = None, **kwargs) -> ComponentPlan:
    kwargs.setdefault("component_id", "mom")
    kwargs.setdefault("to_weights", "top_k_equal_weight")
    kwargs.setdefault("to_weights_params", {"k": 2, "max_gross": 1.0})
    kwargs.setdefault("update_every_bars", 21)
    kwargs.setdefault("valid_for_bars", 42)
    return ComponentPlan(component=component or ConstantRank(SCORES), **kwargs)


def _panel(labels: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day in labels])
    return pd.DataFrame({sid: [100.0 + i] * len(labels) for i, sid in enumerate(SCORES)}, index=index)


def _step(plans, labels: list[str], *, upto: int, states=None):
    """Compose at ``labels[upto]`` against the calendar ``labels[: upto + 1]``."""
    return compose(
        plans,
        decision_bar=labels[upto],
        scoring_panels={"price": _panel(labels[max(0, upto - 2) : upto + 1])},
        sessions=labels[: upto + 1],
        states=states,
    )


def _position(label: str) -> int:
    return sessions.grid_position(dt.date.fromisoformat(label))


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def test_the_grid_epoch_is_the_first_session_of_2006_and_is_position_zero() -> None:
    assert sessions.GRID_EPOCH == dt.date(2006, 1, 3)
    assert sessions.is_session(sessions.GRID_EPOCH)
    assert not sessions.is_session(dt.date(2006, 1, 2)), "January 2, 2006 was the observed New Year's closure"
    assert sessions.grid_position(sessions.GRID_EPOCH) == 0


def test_grid_positions_are_pinned_for_the_dates_registrations_name() -> None:
    # Literals, not recomputations: a table change that moved a past session
    # would move every deployment's schedule, and has to fail here first.
    assert sessions.grid_position(dt.date(2007, 1, 3)) == 251  # 2006 held 251 sessions; 2007-01-02 was closed
    assert sessions.grid_position(dt.date(2022, 3, 29)) == 4087  # vol_momentum_v1's first decision bar
    assert sessions.grid_position(dt.date(2026, 6, 12)) == 5142


def test_every_covered_session_has_the_next_grid_position() -> None:
    covered = sessions.sessions_between(sessions.GRID_EPOCH, sessions.coverage_end())
    assert [sessions.grid_position(day) for day in covered] == list(range(len(covered)))


def test_a_closed_day_and_an_uncovered_day_have_no_grid_position() -> None:
    with pytest.raises(sessions.NotASessionError, match="2026-07-03"):
        sessions.grid_position(dt.date(2026, 7, 3))  # Independence Day observed
    with pytest.raises(sessions.NotASessionError, match="weekend"):
        sessions.grid_position(dt.date(2026, 10, 17))
    with pytest.raises(sessions.SessionTableRangeError):
        sessions.grid_position(dt.date(2005, 12, 30))


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("last", "elapsed", "expected"),
    [(None, 0, True), ("2026-01-02", 0, False), ("2026-01-02", 20, False), ("2026-01-02", 21, True), ("x", 30, True)],
)
def test_without_an_offset_the_predicate_is_the_elapsed_count_rule(last, elapsed, expected) -> None:
    plan = _plan(valid_for_bars=21)
    state = ComponentState(component_id="mom", last_output_bar=last)
    # Off-grid and on-grid bars alike: an elapsed-count plan never reads the grid.
    for bar in ("2026-10-14", "2026-10-17", "not a date"):
        assert due_on_grid(plan, state, bar, elapsed=elapsed) is expected


def test_on_the_grid_the_predicate_reads_the_grid_position() -> None:
    bar = "2026-10-14"
    offset = _position(bar) % 21
    on, off = _plan(phase_offset_bars=offset), _plan(phase_offset_bars=(offset + 1) % 21)
    fresh = ComponentState(component_id="mom")
    held = ComponentState(component_id="mom", last_output_bar="2026-09-15")

    # Staged entry: a plan with no output waits for its own grid date.
    assert due_on_grid(on, fresh, bar, elapsed=0) is True
    assert due_on_grid(off, fresh, bar, elapsed=0) is False
    assert due_on_grid(on, None, bar, elapsed=0) is True
    # On its grid it is due whatever elapsed says, as long as something did.
    assert due_on_grid(on, held, bar, elapsed=21) is True
    assert due_on_grid(on, held, bar, elapsed=20) is True, "back on the grid after an off-grid catch-up"
    assert due_on_grid(on, held, bar, elapsed=0) is False, "an output already produced on this bar"
    # Off its grid only a missed date makes it due: more than one cycle elapsed.
    assert due_on_grid(off, held, bar, elapsed=21) is False
    assert due_on_grid(off, held, bar, elapsed=22) is True


def test_a_plan_on_the_grid_refuses_a_bar_that_is_not_a_session() -> None:
    plan = _plan(phase_offset_bars=3)
    with pytest.raises(CompositionError, match="no grid position"):
        due_on_grid(plan, None, "2026-07-03", elapsed=0)
    with pytest.raises(CompositionError, match="no grid position"):
        due_on_grid(plan, None, "not a date", elapsed=0)


def _refreshes(plan: ComponentPlan, calendar: list[str], *, down: frozenset[str] = frozenset()) -> list[str]:
    """The bars ``plan`` refreshes on over ``calendar``, skipping the bars in ``down``.

    Elapsed sessions are counted on the whole calendar, as the live loop counts
    them on its price panel: a session the loop did not run on still elapsed.
    """
    last: int | None = None
    refreshed: list[str] = []
    for number, bar in enumerate(calendar):
        if bar in down:
            continue
        state = ComponentState(component_id=plan.component_id, last_output_bar=None if last is None else calendar[last])
        if due_on_grid(plan, state, bar, elapsed=0 if last is None else number - last):
            refreshed.append(bar)
            last = number
    return refreshed


def test_a_missed_grid_date_catches_up_once_and_returns_to_the_grid() -> None:
    offset = _position(TWO_YEARS[0]) % 21
    plan = _plan(phase_offset_bars=offset)
    on_grid = [bar for bar in TWO_YEARS if (_position(bar) - offset) % 21 == 0]
    assert _refreshes(plan, TWO_YEARS) == on_grid, "with no outage every refresh is a grid date"

    # A 30-session outage swallows the fourth and fifth grid dates.
    start = TWO_YEARS.index(on_grid[3]) - 5
    outage = TWO_YEARS[start : start + 30]
    assert on_grid[3] in outage and on_grid[4] in outage
    seen = _refreshes(plan, TWO_YEARS, down=frozenset(outage))

    resume = TWO_YEARS[start + 30]  # the first session back
    assert resume not in on_grid
    assert seen[:3] == on_grid[:3]
    assert seen[3] == resume, "one catch-up, on the first session back"
    assert seen[4:] == on_grid[5:], "and then the grid again, not a cycle counted from the catch-up"


def test_the_grid_sets_the_same_dates_wherever_a_deployment_starts() -> None:
    plan = _plan(phase_offset_bars=11)
    early, late = _refreshes(plan, TWO_YEARS), _refreshes(plan, TWO_YEARS[137:])
    assert late == [bar for bar in early if bar >= TWO_YEARS[137]]
    # Under the elapsed rule the start date is the phase, which is the point of the grid.
    elapsed = _plan()
    assert _refreshes(elapsed, TWO_YEARS[137:])[0] == TWO_YEARS[137]
    assert _refreshes(elapsed, TWO_YEARS[138:])[0] == TWO_YEARS[138]


# ---------------------------------------------------------------------------
# Tranches
# ---------------------------------------------------------------------------


def test_tranche_plans_spread_the_cycle_and_split_the_weight() -> None:
    plan = _plan(weight=0.6)
    tranches = tranche_plans(plan, 21, 13)
    assert [t.component_id for t in tranches] == [f"mom/t{j:02d}" for j in range(21)]
    assert sorted(t.phase_offset_bars for t in tranches) == list(range(21))
    assert tranches[0].phase_offset_bars == 13 and tranches[8].phase_offset_bars == 0
    assert all(t.weight == pytest.approx(0.6 / 21) for t in tranches)
    assert all((t.update_every_bars, t.valid_for_bars) == (21, 42) for t in tranches)
    assert all(t.component is plan.component for t in tranches), "one fitted object, many tranches"

    three = tranche_plans(plan, 3, 13)
    assert [t.phase_offset_bars for t in three] == [13, 20, 6]  # r, r+7, r+14 (mod 21)

    (single,) = tranche_plans(plan, 1, 5)
    assert single.component_id == "mom" and single.weight == 0.6 and single.phase_offset_bars == 5


@pytest.mark.parametrize(("k", "base"), [(0, 0), (22, 0), (True, 0), (21, 21), (21, -1), (2.0, 0)])
def test_tranche_plans_refuse_what_the_cycle_cannot_hold(k, base) -> None:
    with pytest.raises(CompositionError):
        tranche_plans(_plan(), k, base)


@pytest.mark.parametrize("offset", [-1, 21, True, 1.5])
def test_a_plan_refuses_an_offset_outside_its_cycle(offset) -> None:
    with pytest.raises(CompositionError, match="phase_offset_bars"):
        _plan(phase_offset_bars=offset)


def test_exactly_one_of_21_tranches_refreshes_on_every_session_for_two_years() -> None:
    component = ConstantRank(SCORES)
    plans = tranche_plans(_plan(component), 21, 0)
    states: dict[str, ComponentState] = {}
    last_refresh: dict[str, int] = {}
    for number, bar in enumerate(TWO_YEARS):
        due = [
            plan.component_id
            for plan in plans
            if due_on_grid(
                plan,
                states.get(plan.component_id),
                bar,
                elapsed=number - last_refresh.get(plan.component_id, number),
            )
        ]
        assert len(due) == 1, f"{bar}: {due}"
        (tranche,) = due
        if tranche in last_refresh:
            assert number - last_refresh[tranche] == 21
        last_refresh[tranche] = number
        states[tranche] = ComponentState(component_id=tranche, last_output_bar=bar)
    assert len(last_refresh) == 21


def test_compose_builds_a_tranched_book_one_tranche_per_session() -> None:
    labels = TWO_YEARS[:30]
    component = ConstantRank(SCORES)
    plans = tranche_plans(_plan(component, update_every_bars=3, valid_for_bars=6), 3, _position(labels[0]) % 3)
    untranched = _step([_plan(ConstantRank(SCORES), update_every_bars=3, valid_for_bars=6)], labels, upto=0)

    states = None
    gross = []
    for upto in range(6):
        result = _step(plans, labels, upto=upto, states=states)
        states = result.states
        gross.append(float(result.targets.abs().sum()))
        assert sum(output.due for output in result.component_outputs) == 1
    assert component.score_calls == 6, "one tranche scored per session, never all three"
    assert gross[:3] == pytest.approx([1 / 3, 2 / 3, 1.0]), "staged entry: one third of the book per session"
    # A time-invariant signal's tranches agree, so the full book is the untranched book.
    np.testing.assert_allclose(
        result.targets.reindex(untranched.targets.index).to_numpy(), untranched.targets.to_numpy(), atol=1e-15
    )


def test_compose_and_the_evaluators_filter_agree_on_every_bar() -> None:
    labels = TWO_YEARS[:60]
    positions = {label: number for number, label in enumerate(labels)}
    for plans in (
        tranche_plans(_plan(update_every_bars=5, valid_for_bars=10), 5, 2),
        tranche_plans(_plan(update_every_bars=5, valid_for_bars=10), 2, 4),
        (_plan(update_every_bars=5, valid_for_bars=10),),
    ):
        states: dict[str, ComponentState] = {}
        for upto, bar in enumerate(labels):
            result = _step(list(plans), labels, upto=upto, states=states)
            assert _due(plans, states, positions, upto, decision_bar=bar) is any(
                output.due or output.expired for output in result.component_outputs
            ), bar
            states = dict(result.states)


def test_the_evaluators_filter_needs_the_bar_for_a_plan_on_the_grid() -> None:
    plan = _plan(phase_offset_bars=0)
    with pytest.raises(EvaluationError, match="decision bar"):
        _due([plan], {}, {"2026-10-14": 0}, 0)
    # An elapsed-count plan never needed it, which is how every existing caller reads.
    assert _due([_plan()], {}, {"2026-10-14": 0}, 0) is True


# ---------------------------------------------------------------------------
# The specification
# ---------------------------------------------------------------------------


def _table(**component_fields) -> dict:
    component = {
        "id": "momentum",
        "path": "prism.signal.momentum_node:MomentumSignalNode",
        "weight": 1.0,
        "horizon_bars": 21,
        "update_every_bars": 21,
        "valid_for_bars": 42,
        "to_weights": "decile_long_short",
        "to_weights_params": {"decile": 0.1},
        "params": {"lookback_bars": 252, "skip_bars": 21, "horizon_bars": 21},
    }
    component.update(component_fields)
    return {
        "strategy": {
            "name": "grid_book",
            "universe_index": "sp500",
            "combination": "position_space",
            "components": [component],
        },
        "construction": {"kind": "heuristic"},
        "policy": {
            "profile": "examples/reference/profiles/paper.toml",
            "costs": "examples/reference/costs/retail_v1.toml",
            "execution": "examples/reference/profiles/execution_alpaca.toml",
            "monitoring": "examples/reference/policies/monitoring.toml",
            "recovery": "examples/reference/policies/recovery.toml",
        },
    }


def _resolved_component(table: dict) -> dict:
    spec = load_spec_table(table, path="grid.toml", workspace=Workspace(REPO_ROOT), instantiate=False)
    return spec.resolved()["strategy"]["components"][0]


def test_the_grid_fields_enter_the_resolved_specification_only_when_declared() -> None:
    assert "tranches" not in _resolved_component(_table())
    assert "phase_offset_bars" not in _resolved_component(_table())

    tranched = _resolved_component(_table(tranches=21))
    assert (tranched["tranches"], tranched["phase_offset_bars"]) == (21, 0), "the implicit offset is recorded"
    phased = _resolved_component(_table(phase_offset_bars=13))
    assert (phased["tranches"], phased["phase_offset_bars"]) == (1, 13), "the implicit tranche count is recorded"
    # Saying the default out loud changes nothing: one schedule, one identity.
    assert _resolved_component(_table(tranches=1, phase_offset_bars=13)) == phased


def test_a_grid_schedule_is_a_different_configuration() -> None:
    workspace = Workspace(REPO_ROOT)
    ids = {
        load_spec_table(table, path="grid.toml", workspace=workspace).config_id(engine_version="pinned")
        for table in (_table(), _table(tranches=21), _table(tranches=3), _table(phase_offset_bars=13))
    }
    assert len(ids) == 4


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"tranches": 22}, "above update_every_bars"),
        ({"tranches": 0}, "must be >= 1"),
        ({"tranches": 2.0}, "must be an integer"),
        ({"phase_offset_bars": 21}, r"\[0, 21\)"),
        ({"phase_offset_bars": -1}, "must be >= 0"),
        ({"phase_offset_bars": True}, "must be an integer"),
    ],
)
def test_the_specification_refuses_a_grid_the_cycle_cannot_hold(fields, match) -> None:
    with pytest.raises(SpecError, match=match):
        _resolved_component(_table(**fields))


def test_tranches_are_refused_under_the_calibrated_flow() -> None:
    table = _table(tranches=3, to_weights="none", to_weights_params={})
    table["strategy"]["combination"] = "calibrated_convex"
    table["construction"]["kind"] = "convex"
    with pytest.raises(SpecError, match="position_space"):
        _resolved_component(table)
    # One phase-offset plan keeps its declared id, so the calibrated flow can carry it.
    table["strategy"]["components"][0].pop("tranches")
    table["strategy"]["components"][0]["phase_offset_bars"] = 4
    assert _resolved_component(table)["phase_offset_bars"] == 4


def test_tranches_are_refused_under_renormalization() -> None:
    table = _table(tranches=3)
    table["strategy"]["renormalize"] = True
    with pytest.raises(SpecError, match="renormalize"):
        _resolved_component(table)
    # A single phase on the grid has nothing to scale up, so it may renormalize.
    table["strategy"]["components"][0].pop("tranches")
    table["strategy"]["components"][0]["phase_offset_bars"] = 4
    assert _resolved_component(table)["phase_offset_bars"] == 4


#: ``sha256(stable_json_dumps(spec.resolved()))`` of every shipped example and
#: of both template specifications, recorded before the grid fields existed.
#: ``resolved()`` is the only part of ``config_id`` a component-table field can
#: reach; the rest is file content and implementation identity. That identity
#: carries the git commit of each component's source, so a literal ``config_id``
#: would change with every commit; this digest changes only when the
#: specification a file declares does.
SHIPPED_RESOLVED = {
    "examples/breadth/control.toml": "b56c656d47dd2a7c3eb4df545ce4c4ad48bdaac9b0445e31d5b67e7f2c51ca3c",
    "examples/buffer/daily.toml": "709cf0838f1aa976ebc7f295d4c1dc159adfd7665f891715726541286fae73de",
    "examples/buffer/monthly.toml": "9e8fa55b040acdb4e2638a9bdf976f95a82b8bc19fb8d2f7e5562dfe943c3820",
    "examples/characteristics/control.toml": "aff73c31cff167dc887a6966317c2775889cb19cc0fdc6e071cc46b011285fba",
    "examples/earnings/daily.toml": "d92481b77d5ab166742122b28d18fb72ceec5f32e2708d7cf8953e49b5276abb",
    "examples/earnings/monthly.toml": "064eb74ccb797325cec206d2d15784b1101934607f357a54fc5cb5258253ac41",
    "examples/fundamentals/control.toml": "4b04b3ae0367d0657fb14d34ca5dcb5766de5097ad1b608e7a9d710570c7e000",
    "examples/learned_xsection/control.toml": "b9ed8263bf7e72d3abb7ee87873e1d9ab033a174a01d59c2dafd5bac62cce436",
    "examples/momentum/demotion_b1.toml": "64e35136712cfae38306fb9ec8d7de9c795b8898f4b37e121454b89e788e7c55",
    "examples/momentum/demotion_b1_flat_spread.toml": "bfa5b451686e462f153a6d3e038c3fa71cf99bee2ca197e8b7fd107e90b0a946",
    "examples/momentum/reference.toml": "b0fbc5b43a115653d7612af086a1e0629e24bcca77c3f5b5891f554b02d3d57f",
    "examples/trend/reference.toml": "7f1e9c790f06e571be1a7c915f679c07b371a16c5289315040216f6cd71416b9",
    "examples/trend/trend_t0.toml": "1a2aecf8771177b657feae01e9722ab2fc2eef1d3e1b5ef54b1cb85100539037",
    "template:specs/book.toml": "4745a5cb77abe2a55008fd143e6febcbcfb56abe85986ff1bee9d9765d05eeef",
    "template:specs/etf_core.toml": "8e5dae405f568178b80351cb4e0d1724de456cc82ef1a624adaa748589470f07",
}


def _resolved_digest(path: Path, workspace: Workspace) -> str:
    spec = load_spec(path, workspace=workspace, instantiate=False)
    return hashlib.sha256(stable_json_dumps(spec.resolved()).encode("utf-8")).hexdigest()


def test_every_shipped_specification_keeps_its_identity(tmp_path: Path) -> None:
    shipped = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "examples").rglob("*.toml")
        if "examples/reference/" not in path.relative_to(REPO_ROOT).as_posix()
    )
    assert shipped == sorted(key for key in SHIPPED_RESOLVED if not key.startswith("template:")), (
        "an example was added or removed; record its resolved digest here"
    )
    repo = Workspace(REPO_ROOT)
    digests = {relative: _resolved_digest(REPO_ROOT / relative, repo) for relative in shipped}
    template = tmp_path / "template"
    write_template(template)
    for name in ("book.toml", "etf_core.toml"):
        digests[f"template:specs/{name}"] = _resolved_digest(template / "specs" / name, Workspace(template))
    assert digests == SHIPPED_RESOLVED


# ---------------------------------------------------------------------------
# The paths that cannot honour a grid schedule refuse it (N7)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_imports():
    """Undo the template's ``strategies`` import and its sys.path entry after the test."""
    before_path, before_modules = list(sys.path), set(sys.modules)
    yield
    sys.path[:] = before_path
    for name in set(sys.modules) - before_modules:
        del sys.modules[name]


def _template_with_tranches(root: Path) -> Path:
    write_template(root)
    path = root / "specs" / "book.toml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("valid_for_bars = 21\n", "valid_for_bars = 21\ntranches = 3\n", 1), encoding="utf-8")
    return path


def test_prism_run_refuses_a_grid_schedule_it_would_not_trade(tmp_path: Path, capsys, isolated_imports) -> None:
    root = tmp_path / "proj"
    path = _template_with_tranches(root)
    argv = ["--workspace", str(root), "run", "--spec", str(path), "--account", "paper", "--once"]
    assert main([*argv, "--broker", "synthetic"]) == EXIT_REFUSED
    assert "tranches or phase_offset_bars" in capsys.readouterr().err


def test_the_score_freeze_refuses_a_grid_schedule_it_would_not_score() -> None:
    from prism.scripts.construction_experiment import ExperimentError, freeze_scores

    workspace = Workspace(REPO_ROOT)
    spec = load_spec_table(_table(phase_offset_bars=3), path="grid.toml", workspace=workspace)

    class Registration:
        arm_ids = ("arm",)

        def arm_specification(self, arm, *, workspace):
            return spec

    # Refused before the dataset is opened or anything is written.
    with pytest.raises(ExperimentError, match="tranches or phase_offset_bars"):
        freeze_scores(workspace=workspace, registration=Registration(), dataset=None)


def test_the_smoke_path_refuses_a_grid_schedule_it_would_not_run(tmp_path: Path, isolated_imports) -> None:
    root = tmp_path / "proj"
    path = _template_with_tranches(root)
    workspace = Workspace(root)
    spec = load_spec(path, workspace=workspace)
    with pytest.raises(EvaluationError, match="tranches or phase_offset_bars"):
        smoke_evaluate(spec, workspace=workspace)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

PACKAGE = "grid_strategies"
SPLITTER = RollingWalkForward(formation_bars=60, test_bars=40, min_test_bars=20)

TRAILING_SIGNAL = '''
from prism.signal.base import Signal


class TrailingReturn(Signal):
    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    price_scale_invariant = True
    fitted_state = "none"
    estimated_decay_bars = None
    refit_every_bars = 1
    training_window_bars = 0

    @property
    def horizon_bars(self):
        return 5

    @property
    def required_history(self):
        return 6

    def fit(self, close, volume=None):
        return self

    def score(self, close, volume=None):
        return close.iloc[-1] / close.iloc[-6] - 1.0
'''

BOOK_TOML = """
[strategy]
name = "trailing_book"
universe_index = "synthetic_100"
combination = "position_space"

[[strategy.components]]
id = "trailing"
path = "grid_strategies.trailing:TrailingReturn"
weight = 1.0
horizon_bars = 5
update_every_bars = 5
valid_for_bars = 10
to_weights = "decile_long_short"
{grid}
[strategy.components.to_weights_params]
decile = 0.20
max_gross = 1.0
max_symbol_abs_weight = 0.35

[construction]
kind = "heuristic"

[policy]
profile = "profiles/paper.toml"
costs = "costs/retail_v1.toml"
execution = "profiles/execution_synthetic.toml"
monitoring = "policies/monitoring.toml"
recovery = "policies/recovery.toml"
"""

POLICY_FILES = {
    "profiles/paper.toml": 'name = "paper"\n',
    "costs/retail_v1.toml": "commission_bps = 1.0\n",
    "profiles/execution_synthetic.toml": 'venue = "synthetic"\n',
    "policies/monitoring.toml": "review_bars = 21\n",
    "policies/recovery.toml": (
        'on_infeasible = "cancel"\n'
        'on_sign_contradiction = "hold"\n'
        'on_blocked_obligation = "escalate"\n'
        'on_rounding_failure = "scale_batch"\n'
        'preference_response = "record"\n'
    ),
}


@pytest.fixture(scope="module")
def engine_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One synthetic project, built once and shared through per-run symlinked workspaces."""
    root = tmp_path_factory.mktemp("grid") / "source"
    workspace = Workspace(root).ensure()
    write_synthetic_dataset(
        workspace.data,
        seed=11,
        scenarios="all",
        n_equities=12,
        n_etfs=2,
        start=dt.date(2026, 1, 1),
        end=dt.date(2026, 12, 31),
    )
    (root / PACKAGE).mkdir()
    (root / PACKAGE / "__init__.py").write_text("", encoding="utf-8")
    (root / PACKAGE / "trailing.py").write_text(textwrap.dedent(TRAILING_SIGNAL), encoding="utf-8")
    for relative, content in POLICY_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    before_path, before_modules = list(sys.path), set(sys.modules)
    yield root
    sys.path[:] = before_path
    for name in set(sys.modules) - before_modules:
        del sys.modules[name]


def _canonical_settings() -> CanonicalSettings:
    volume = {
        "estimator": "median",
        "window": "trailing",
        "window_bars": 20,
        "refresh": "decision",
        "floor_dollars": 100_000.0,
    }
    stack = CostStack.from_mapping(
        {
            "spread": {"mode": "flat", "flat_bps": 0.0},
            "impact": {
                "quadratic_notional_coeff": 0.0,
                "participation_model": "linear",
                "participation_coeff": 0.0,
            },
            "dollar_volume": {"spread_bucket": volume, "participation": volume, "impact": volume},
            "commission_bps": 0.0,
            "borrow_bps_annual": 0.0,
            "cash_income_bps_annual": 0.0,
            "financing_bps_annual": 0.0,
            "short_proceeds": "collateral_no_rebate",
            "holding_cost_settlement": "per_bar",
            "trading_days_per_year": 252,
        }
    )
    return CanonicalSettings(cost_stack=stack)


def _run(engine_root: Path, tmp_path: Path, name: str, grid: str, **options):
    """One account_state evaluation of the trailing book with ``grid`` in its component table."""
    root = tmp_path / name
    root.mkdir()
    for entry in sorted(engine_root.iterdir()):
        if entry.name not in {"results", "trials", "state", "runs", "logs", "specs"}:
            (root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    workspace = Workspace(root).ensure()
    (root / "specs").mkdir()
    (root / "specs" / "book.toml").write_text(BOOK_TOML.format(grid=grid), encoding="utf-8")
    spec = load_spec(root / "specs" / "book.toml", workspace=workspace)
    settings = {
        "engine": "canonical",
        "canonical": _canonical_settings(),
        "decision_coupling": "account_state",
        "fold_portfolio": "carry",
        # Whole shares need an account that can buy one: at the default capital
        # of 1.0 every order rounds to zero and every book is the same flat book.
        "capital": 100_000.0,
    }
    settings.update(options)
    return walk_forward_evaluate(
        spec,
        workspace=workspace,
        splitter=SPLITTER,
        fold_table_id="grid_unit",
        selection_set_id="unit_set",
        registration_id="unit_set",
        kind="new",
        **settings,
    )


def test_a_single_phase_book_on_the_grid_at_its_own_phase_is_the_same_book(engine_root, tmp_path) -> None:
    """The parity a phase-offset row rests on: the grid moves no decision when it agrees with the start."""
    elapsed = _run(engine_root, tmp_path, "elapsed", "")
    first = elapsed.folds[0].test[0]
    offset = _position(first) % 5
    gridded = _run(engine_root, tmp_path, "gridded", f"phase_offset_bars = {offset}\n")

    assert gridded.config_id != elapsed.config_id
    assert gridded.record.evaluation["engine"]["n_decision_bars"] == elapsed.record.evaluation["engine"][
        "n_decision_bars"
    ]
    assert elapsed.record.evaluation["engine"]["n_fills"] > 0 and elapsed.scored.returns.abs().sum() > 0.0
    pd.testing.assert_series_equal(gridded.scored.returns, elapsed.scored.returns, check_exact=True)

    # One session later on the grid is a different book with the same cadence.
    shifted = _run(engine_root, tmp_path, "shifted", f"phase_offset_bars = {(offset + 1) % 5}\n")
    assert not shifted.scored.returns.equals(elapsed.scored.returns)


def test_a_tranched_book_refreshes_one_tranche_per_bar_and_enters_in_stages(
    engine_root, tmp_path, monkeypatch
) -> None:
    calls: list[tuple[str, list[str], int]] = []
    real_compose = evaluate_module.compose

    def spy(plans, **kwargs):
        result = real_compose(plans, **kwargs)
        due = [output.component_id for output in result.component_outputs if output.due]
        holding = sum(1 for state in result.states.values() if any(v != 0.0 for v in state.sub_targets.values()))
        calls.append((kwargs["decision_bar"], due, holding))
        gross = float(result.targets.abs().sum())
        assert gross <= 1.0 + 1e-12, f"{kwargs['decision_bar']}: tranched gross {gross} above the rule's cap"
        return result

    monkeypatch.setattr(evaluate_module, "compose", spy)
    result = _run(engine_root, tmp_path, "tranched", "tranches = 5\n")

    tested = [day for fold in result.folds for day in fold.decisions]
    assert [bar for bar, _, _ in calls] == tested, "every test bar is a decision bar: some tranche is due"
    for bar, due, _ in calls:
        assert len(due) == 1, f"{bar}: {due}"
        (tranche,) = due
        assert (_position(bar) - int(tranche[-2:])) % 5 == 0, f"{bar}: {tranche} is off its grid offset"
    assert [holding for _, _, holding in calls[:6]] == [1, 2, 3, 4, 5, 5], "staged entry, one tranche a session"
    assert result.record.evaluation["engine"]["n_fills"] > 0
    # The synthetic panel stops pricing a name some tranche still targets; its entry
    # waits for a priced bar instead of failing to round, and the run says so.
    assert any("deferred" in warning for warning in result.record.warnings)


def test_an_unpriced_entry_is_deferred_and_nothing_else_is() -> None:
    sids = pd.Index(["held", "held_unmarked", "new_unpriced", "new_priced", "flat_unpriced", "no_target"])
    target = pd.Series([0.1, -0.1, 0.2, 0.3, 0.0, np.nan], index=sids)
    # A held name normally carries the account's mark; one that does not is the
    # mark check's refusal to make (``_decision_marks``), never a deferral.
    marks = pd.Series([50.0, 0.0, 0.0, 20.0, 0.0, 0.0], index=sids)
    shares = pd.Series([10.0, -5.0, 0.0, 0.0, 0.0, 0.0], index=sids)
    deferred_target, deferred = evaluate_module._defer_unpriced_entries(target, marks=marks, shares=shares)
    assert deferred == ["new_unpriced"]
    expected = target.copy()
    expected["new_unpriced"] = np.nan  # no decision: the heuristic construction holds it at w0 = 0
    pd.testing.assert_series_equal(deferred_target, expected)
    same, none = evaluate_module._defer_unpriced_entries(target.drop("new_unpriced"), marks=marks, shares=shares)
    assert none == [] and same.equals(target.drop("new_unpriced"))
