"""The stage 4 solver profiling instrument (research/scripts/solver_profile.py).

The failure this file exists to catch is a **fast number for the wrong program**. The whole
value of the DPP half of the profiling is the comparison against the shipped constructor, and
a parametrized twin that quietly relaxed a constraint, dropped the power-1.5 impact term, or
solved a different objective would report a large speedup that means nothing. So the twin is
held to the production program's own optimum on identical data, in both compilation modes and
in both cone modes, and it is held to the *structure* too: the same limits, the same cost
atoms, the same second-order cone count.

The second failure guarded here is a memory column that does not mean what it says. Peak
resident set size is monotonic within a process, so profiling several cross-section sizes in
one process would report the largest one's peak for every row. The orchestration forks per
cell, and the test asserts that the child actually is a separate process.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from research.scripts import solver_profile as sp

pytest.importorskip("cvxpy", reason="the convex constructor is the declared `construction` extra")

from prism.portfolio.convex import ConvexProgram  # noqa: E402

SMALL = 40


def _program(n: int = SMALL, *, seed: int = 7) -> tuple[ConvexProgram, sp.TwinData]:
    shape = sp.make_shape(n, seed=seed)
    snapshot = sp.make_snapshot(shape, seed=seed + 100)
    program = ConvexProgram(sp.make_inputs(shape, snapshot, sp.make_fit()))
    return program, sp.twin_data(program)


def _production_solution(program: ConvexProgram):
    return program.solve(
        program.economic_objective(),
        program.action_constraints() + program.holding_state_constraints(),
        sense="maximize",
    )


# ---------------------------------------------------------------------------
# The twin solves the production program
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dpp", "parametrize_cone"),
    [(False, True), (True, True), (True, False)],
)
def test_the_twin_reproduces_the_production_optimum(dpp: bool, parametrize_cone: bool) -> None:
    """A profiling twin that solved a different problem would report a meaningless speedup."""
    program, data = _program()
    production = _production_solution(program)
    assert production.optimal, production.status
    twin = sp.TwinProgram(data, dpp=dpp, parametrize_cone=parametrize_cone)
    result = twin.solve()
    assert result["status"] in ("optimal", "optimal_inaccurate"), result["status"]
    reference = production.w.to_numpy(dtype=float)
    solution = np.asarray(result["w"], dtype=float)
    # The two programs are the same program written twice, so they agree to solver tolerance,
    # not to machine precision: CLARABEL's declared tol_gap_rel is 1e-9 and the constructor's
    # own limit tolerance is 1e-6.
    assert np.max(np.abs(solution - reference)) < 1.0e-4
    assert abs(float(result["value"]) - float(production.value)) < 1.0e-8


def test_the_twin_honors_every_limit_the_profile_declares() -> None:
    """Checked on the solution, against the declared values, not against the twin's own bounds."""
    program, data = _program()
    twin = sp.TwinProgram(data, dpp=True)
    result = twin.solve()
    w = np.asarray(result["w"], dtype=float)
    tolerance = 1.0e-6
    assert np.abs(w).sum() <= sp.MAX_GROSS - sp.CASH_BUFFER + tolerance
    assert np.max(np.abs(w)) <= sp.MAX_NAME_WEIGHT + tolerance
    assert abs(w.sum()) <= sp.NET_BAND + tolerance
    assert math_sqrt_variance(program, w) <= sp.VARIANCE_BUDGET + tolerance
    trade = np.abs(w - data.w0)
    assert np.all(trade <= data.trade_hi + tolerance)


def math_sqrt_variance(program: ConvexProgram, w: np.ndarray) -> float:
    """``sqrt(w' Sigma w)`` through the snapshot itself, not through the twin's cone."""
    return float(np.sqrt(program.risk.variance(w)))


def test_the_twin_carries_the_same_cost_atoms_as_the_production_program() -> None:
    """A twin that dropped the power-1.5 impact term would be a different, easier program."""
    _, data = _program()
    assert np.all(data.linear_bps > 0.0), "the spread and commission terms must be present"
    assert np.any(data.quadratic != 0.0), "the quadratic notional impact term must be present"
    assert np.any(data.power != 0.0), "the sqrt participation term maps to |dw|^1.5 and must be present"
    assert np.any(data.borrow != 0.0), "the borrow term must be present"
    assert data.financing > 0.0 and data.income > 0.0
    assert data.income <= data.financing, "a cash rate above financing is not convex and must not be profiled"


def test_the_twin_and_the_production_program_canonicalize_to_the_same_cone_count() -> None:
    """Structure, not just the optimum: the same conic atoms, in the same numbers.

    The twin carries N more variables and N more equality rows -- that is the ``dw`` splitting
    the DPP formulation needs, and it is exactly what the profiling reports as its cost. It
    must not carry a different *cone* structure, which would mean a different program.
    """
    import cvxpy as cp

    program, data = _program()
    objective = program.economic_objective()
    constraints = program.action_constraints() + program.holding_state_constraints()
    production = cp.Problem(cp.Maximize(objective), constraints)
    production_record = sp._canonical_record(production.get_problem_data(cp.CLARABEL)[0], production)
    twin_record = sp.TwinProgram(data, dpp=True).canonical_shape()

    assert production_record["cone_dims"]["n_second_order_cones"] == twin_record["cone_dims"]["n_second_order_cones"]
    assert (
        production_record["cone_dims"]["second_order_cone_sizes"]
        == twin_record["cone_dims"]["second_order_cone_sizes"]
    )
    assert twin_record["n_scalar_variables"] == 2 * production_record["n_scalar_variables"]
    assert twin_record["n_scalar_eq_constraints"] - production_record["n_scalar_eq_constraints"] == SMALL


def test_the_dpp_twin_is_actually_dpp_and_the_constant_twin_is_not_parametrized() -> None:
    """If the parametrized twin fell back to non-DPP compilation the comparison would be empty.

    ``is_dcp(dpp=True)`` is true of a problem with no parameters at all, so the discriminating
    assertion is on the parameter count, not on the flag.
    """
    _, data = _program()
    dpp = sp.TwinProgram(data, dpp=True)
    fixed = sp.TwinProgram(data, dpp=True, parametrize_cone=False)
    constant = sp.TwinProgram(data, dpp=False)
    assert dpp.is_dpp() and fixed.is_dpp() and constant.is_dpp()
    assert len(dpp.problem.parameters()) > 0
    assert len(constant.problem.parameters()) == 0
    # The cone is the N x K parameter; holding it constant must drop exactly that many scalars.
    dpp_scalars = dpp.canonical_shape()["parameter_tensor"]["n_scalar_parameters"]
    fixed_scalars = fixed.canonical_shape()["parameter_tensor"]["n_scalar_parameters"]
    assert dpp_scalars - fixed_scalars == SMALL * (sp.N_FACTORS + 1)


def test_reassigning_a_dpp_twin_moves_the_solution_to_the_new_bar() -> None:
    """A parameter assignment that silently did nothing would make the DPP cell a re-solve of
    one bar, which is fast and meaningless."""
    program_a, data_a = _program(seed=7)
    _, data_b = _program(seed=11)
    twin = sp.TwinProgram(data_a, dpp=True)
    first = twin.solve()
    twin.assign(data_b)
    second = twin.solve()
    assert np.max(np.abs(np.asarray(first["w"]) - np.asarray(second["w"]))) > 1.0e-4
    fresh = sp.TwinProgram(data_b, dpp=False).solve()
    assert np.max(np.abs(np.asarray(second["w"], dtype=float) - np.asarray(fresh["w"], dtype=float))) < 1.0e-4
    assert twin.dpp and program_a.n == SMALL


def test_a_constant_twin_refuses_to_be_repointed() -> None:
    """Silently accepting an assignment on a constant twin would report a DPP-shaped timing
    for a program that never used the new bar's data."""
    _, data = _program()
    with pytest.raises(RuntimeError):
        sp.TwinProgram(data, dpp=False).assign(data)


# ---------------------------------------------------------------------------
# The declared shape is the profile's shape
# ---------------------------------------------------------------------------


def test_the_profiled_shape_is_the_experiment_profile_and_the_reference_risk_model() -> None:
    """Every structural constant is read from the shipped files, so a profiling number cannot
    describe limits nobody declared."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    profile = tomllib.loads((root / "examples/reference/profiles/experiment_v1.toml").read_text())
    values = {row["name"]: row for row in profile["limits"]}
    assert values["max_gross"]["value"] == sp.MAX_GROSS
    assert values["max_name_weight"]["value"] == sp.MAX_NAME_WEIGHT
    assert values["net_band"]["value"] == sp.NET_BAND
    assert values["variance_budget"]["params"]["value"] == sp.VARIANCE_BUDGET
    assert values["variance_budget"]["params"]["horizon_bars"] == sp.HORIZON_BARS
    assert values["participation_per_name"]["value"] == sp.PARTICIPATION
    assert profile["cash_buffer"] == sp.CASH_BUFFER

    risk = tomllib.loads((root / "examples/reference/risk/factor_diagonal.toml").read_text())
    assert risk["n_factors"] == sp.N_FACTORS
    assert risk["horizon_bars"] == sp.HORIZON_BARS
    assert risk["scaling"] == sp.SCALING
    assert risk["eigenvalue_floor_relative"] == sp.EIGENVALUE_FLOOR_RELATIVE

    registration = tomllib.loads((root / "registrations/2026-09-construction.toml").read_text())
    assert registration["shared"]["risk_aversion"] == sp.RISK_AVERSION
    assert registration["shared"]["eps"] == sp.EPS
    assert len(registration["frozen"]["capital_scenarios"]) == sp.RUN_CAPITAL_SCENARIOS
    convex_arms = [arm for arm in registration["arms"] if arm["construction"]["kind"] == "convex"]
    assert len(convex_arms) == sp.RUN_CONVEX_ARMS


def test_the_profiled_program_takes_the_normal_path_so_the_timing_is_a_solve_and_not_a_refusal() -> None:
    """A no-action cell would be fast and would measure nothing."""
    from prism.portfolio.convex import ConvexConstructor

    shape = sp.make_shape(SMALL, seed=3)
    snapshot = sp.make_snapshot(shape, seed=303)
    output = ConvexConstructor().construct(sp.make_inputs(shape, snapshot, sp.make_fit()))
    assert output.path == "normal"
    assert output.diagnostics["validation"]["ok"] is True
    assert np.isfinite(output.w_target.to_numpy(dtype=float)).all()


# ---------------------------------------------------------------------------
# The memory column means what it says
# ---------------------------------------------------------------------------


def test_each_cell_runs_in_its_own_process_so_peak_rss_is_that_cell_s_peak() -> None:
    """``ru_maxrss`` is a high-water mark that never falls, so two cross-section sizes profiled
    in one process would report the larger one's peak for both."""
    record = sp.run_cell_in_subprocess(
        n=SMALL,
        mode="twin_constant",
        bars=2,
        seed=5,
        equity=sp.DEFAULT_EQUITY,
        full_decision=False,
        memory_budget_mib=6_000.0,
    )
    assert not record.get("failed"), record
    assert record["memory"]["peak_rss_mib"] > 0.0
    assert record["memory"]["sampled_peak_rss_mib"] > 0.0
    assert record["canonical_shape"]["n_scalar_variables"] == 2 * SMALL


def test_a_cell_that_exceeds_the_memory_budget_reports_its_phase_and_its_figure() -> None:
    """The spec refuses a roadmap item that cites an unattributed out-of-memory result. A cell
    the kernel killed, carrying a signal number and nothing else, would be exactly that.

    So the budget is set below what a 700-name DPP compile needs, and the record is required to
    say the harness stopped it, which phase it was in, and how much it was holding. The sampler
    is what makes the figure real: a phase marker records the memory a phase *began* with, and
    the memory that stops a cell is spent inside the phase, after the marker.
    """
    budget = 900.0
    record = sp.run_cell_in_subprocess(
        n=700,
        mode="twin_dpp",
        bars=1,
        seed=5,
        equity=sp.DEFAULT_EQUITY,
        full_decision=False,
        memory_budget_mib=budget,
    )
    assert record.get("failed") is True
    assert record.get("stopped_at_memory_budget") is True
    assert record.get("killed_by_oom_killer") is False
    trace = record.get("trace")
    assert trace is not None
    assert "memory_budget_exceeded" in str(trace["last_phase"])
    assert trace["peak_rss_observed_mib"] > budget
    assert trace["n_rows"] > trace["n_phase_marks"], "the sampler, not the markers, saw the growth"


def test_the_cell_record_is_json_serializable() -> None:
    """The artifact is a tracked JSON file; a numpy scalar in the record would break the write
    after the expensive part of the run has already happened."""
    record = sp.run_cell(
        n=SMALL,
        mode="production",
        bars=2,
        seed=9,
        equity=sp.DEFAULT_EQUITY,
        full_decision=True,
    )
    json.dumps(record)
    # Every bar's status is counted, not just collected into a set: "optimal_inaccurate appeared"
    # and "every bar was optimal_inaccurate" are different facts and the record must tell them apart.
    assert sum(record["status_counts"].values()) == 2
    assert set(record["status_counts"]) == set(record["statuses"])
    assert record["complete_decision"]["construct_seconds"]["n"] == 2
    assert record["complete_decision"]["paths"] == ["normal"]
    assert record["canonical_shape"]["cone_dims"]["n_second_order_cones"] >= SMALL


def test_the_environment_block_names_the_hardware_and_the_solver_versions() -> None:
    """Every number in the artifact belongs to a machine and a solver build, or it belongs to
    nothing."""
    env = sp.environment()
    for key in ("cpu_count_logical", "memory_total_mib", "cvxpy", "clarabel", "platform", "python"):
        assert env[key] is not None, key
    assert env["solver"] == "CLARABEL"
    assert env["solver_tolerances"]["tol_feas"] > 0.0
    assert "sandbox" in env["sandbox_note"].lower()


def test_the_complete_evaluation_projection_uses_the_registration_s_own_arithmetic() -> None:
    cell = {
        "mode": "production",
        "n_names": 574,
        "complete_decision": {
            "construct_seconds": {"median": 1.0},
            "finalize_batch_seconds": {"median": 0.5},
        },
    }
    record = sp.projection([cell])
    assert record["decisions_per_arm_per_capital_scenario"] == 63
    assert record["total_convex_decisions"] == 63 * 2 * 6
    entry = record["by_cross_section_size"]["574"]
    assert entry["median_seconds_per_decision"] == pytest.approx(1.5)
    assert entry["seconds_for_both_convex_arms_all_capital_scenarios"] == pytest.approx(1.5 * 756)


def test_the_scaling_fit_recovers_a_known_exponent_and_flags_an_extrapolation() -> None:
    """A shape the harness stopped is described by the curve the completed shapes trace. If the
    fit were wrong, a roadmap item would be citing a fabricated number instead of a measured
    one, which is the failure this whole artifact exists to prevent."""
    cells = [
        {
            "mode": "production",
            "n_names": n,
            "per_bar_total_seconds": {"median": 1.0e-6 * n**2},
            "cold": {"compile_seconds": 1.0e-6 * n**2},
            "memory": {"rss_at_start_mib": 100.0, "peak_rss_mib": 100.0 + 1.0e-3 * n**2},
        }
        for n in (500, 1000, 1500)
    ]
    record = sp.scaling(cells, extrapolate_to=(500, 3000))
    entry = record["by_mode"]["production"]
    assert record["baseline_rss_mib"] == pytest.approx(100.0)
    assert entry["per_bar_seconds"]["exponent"] == pytest.approx(2.0, abs=1.0e-6)
    assert entry["marginal_peak_rss_mib"]["exponent"] == pytest.approx(2.0, abs=1.0e-6)
    assert "500" not in entry["extrapolated"], "a measured size is never reported as extrapolated"
    assert entry["extrapolated"]["3000"]["per_bar_seconds"] == pytest.approx(9.0, rel=1.0e-6)
    assert entry["extrapolated"]["3000"]["marginal_peak_rss_mib"] == pytest.approx(9000.0, rel=1.0e-6)


def test_the_scaling_fit_ignores_the_cells_that_did_not_complete() -> None:
    """A stopped cell holds a lower bound on its memory, never the figure it would have reached;
    fitting on it would drag the curve down and understate the cost."""
    completed = [
        {
            "mode": "twin_dpp",
            "n_names": n,
            "per_bar_total_seconds": {"median": 0.001 * n},
            "cold": {"compile_seconds": 0.001 * n},
            "memory": {"rss_at_start_mib": 100.0, "peak_rss_mib": 100.0 + 2.0 * n},
        }
        for n in (500, 1000)
    ]
    stopped = {"mode": "twin_dpp", "n_names": 3000, "failed": True, "stopped_at_memory_budget": True}
    record = sp.scaling(completed + [stopped], extrapolate_to=(3000,))
    entry = record["by_mode"]["twin_dpp"]
    assert entry["completed_at_n_names"] == [500, 1000]
    assert entry["marginal_peak_rss_mib"]["fitted_on_n_names"] == [500.0, 1000.0]
    assert entry["extrapolated"]["3000"]["marginal_peak_rss_mib"] == pytest.approx(6000.0, rel=1.0e-6)


def test_a_failed_cell_never_reaches_the_projection() -> None:
    """A projection built from a cell that crashed would be an invented number."""
    assert sp.projection([{"mode": "production", "n_names": 3000, "failed": True}])["by_cross_section_size"] == {}


def test_the_summary_reports_no_out_of_memory_on_a_path_that_completed() -> None:
    """The spec forbids a roadmap item citing an unattributed out-of-memory result. The summary
    must therefore separate "this mode completed" from "this mode was stopped, holding X"."""
    cells = [
        {
            "mode": "production",
            "n_names": n,
            "statuses": ["optimal"],
            "cold": {"compile_seconds": 0.1},
            "per_bar_total_seconds": {"median": 0.001 * n},
            "repeat": {"solver_seconds": {"median": 0.0005 * n}},
            "complete_decision": {"construct_seconds": {"median": 0.002 * n}, "finalize_batch_seconds": None},
            "memory": {"rss_at_start_mib": 180.0, "peak_rss_mib": 180.0 + 0.03 * n},
        }
        for n in (500, 3000)
    ] + [
        {
            "mode": "twin_dpp",
            "n_names": 3000,
            "failed": True,
            "stopped_at_memory_budget": True,
            "trace": {"peak_rss_observed_mib": 8200.0, "phase_at_peak_rss": "bar0:solve", "seconds_to_last_row": 20.0},
        }
    ]
    record = sp.summary(cells, sp.scaling(cells))
    headline = record["the_shipped_constructor_at_the_largest_shape_profiled"]
    assert headline["n_names"] == 3000 and headline["ran"] is True
    assert headline["peak_rss_above_interpreter_baseline_mib"] == pytest.approx(90.0)
    stopped = record["what_ran_out_of_memory"]["twin_dpp"]
    assert len(stopped) == 1
    assert stopped[0]["n_names"] == 3000 and stopped[0]["resident_mib_at_stop"] == 8200.0
    assert record["what_ran_out_of_memory"]["twin_dpp_fixed_cone"] == []
    json.dumps(record)


def test_a_child_cell_can_import_prism_without_an_ambient_pythonpath() -> None:
    """A child process inherits the environment, never the parent's ``sys.path``.

    In a checkout where ``prism`` is not installed -- this worktree is one -- a cell whose child
    could not import it would come back as a failure that says nothing about the solver, and the
    artifact would carry a hole where a measurement should be. The failure is invisible when the
    profiling is run by hand with PYTHONPATH already set, which is exactly why it is pinned here.
    """
    env = sp.child_environment()
    root = Path(__file__).resolve().parents[1]
    entries = env["PYTHONPATH"].split(":")
    assert str(root / "src") in entries and str(root) in entries

    import subprocess

    completed = subprocess.run(
        [sys.executable, "-c", "import prism, research.scripts.solver_profile as m; print(m.N_FACTORS)"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
        env={key: value for key, value in env.items() if key != "PYTHONPATH"} | {"PYTHONPATH": env["PYTHONPATH"]},
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == str(sp.N_FACTORS)


def test_the_driver_module_is_runnable_as_a_module() -> None:
    """The artifact records the command; a module that cannot be run as one is not reproducible."""
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-m", "research.scripts.solver_profile", "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=sp.child_environment(),
    )
    assert completed.returncode == 0
    assert "--memory-budget-mib" in completed.stdout
