"""The full evaluator (engine specification §3.8, §3.12, stage 2).

The property this harness exists to protect is the train-only boundary, so it
is tested directly and with a component that *would* leak if the boundary were
wrong: an oracle that memorizes the total return of every name over whatever
panel it is handed at fit time and holds that ranking through the fold. Fitted
on its fold's formation window it earns what such a ranking is worth
out-of-sample; fitted on the whole sample it is a look-ahead, and the same
evaluation turns a loss into a profit. Both numbers are asserted, so a
regression that widened the fit window would fail here rather than produce a
better-looking backtest.

Around that: the record satisfies ``record_schema = 1`` and carries the fields
§3.12 lists, the fold table's content hash is what the record's
``fold_geometry`` reports, the reset mechanic flattens inside the fold, and the
run-recording protocol's rows appear in the workspace's trial log — one
``started`` row before any computation and exactly one terminal row after it,
including when the evaluation raises.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from prism.evaluate import (
    CANONICAL_NOTE,
    REPRODUCTION_CAVEAT,
    CanonicalSettings,
    EvaluationResult,
    RunRefused,
    _canonical_decisions,
    prepare_components,
    resolve_log_path,
    walk_forward_evaluate,
)
from prism.execution.cost_model import CostStack
from prism.io.dataset import Dataset
from prism.io.synthetic import write_synthetic_dataset
from prism.spec import load_spec
from prism.validation.trials import (
    RecordSchemaError,
    ResultRecord,
    TrialLog,
    validate_result_record,
)
from prism.validation.walk_forward import (
    FoldBoundary,
    FoldGeometryError,
    FoldTable,
    RollingWalkForward,
)
from prism.workspace import Workspace

START = dt.date(2026, 1, 1)
END = dt.date(2026, 12, 31)
PACKAGE = "wfe_strategies"
SPLITTER = RollingWalkForward(formation_bars=60, test_bars=40, min_test_bars=20)

ORACLE_SIGNAL = '''
import json

import pandas as pd

from prism.signal.base import Signal


class OracleRank(Signal):
    """Memorizes each name's total return over the panel it is *fitted* on.

    Deliberately leak-prone: what it scores is a fact about its fit window. Fit
    on a formation window it is a momentum ranking; fit on the whole sample it
    is a look-ahead. Which one the evaluator hands it is the property under
    test, so every fit panel it sees is recorded in ``FIT_LOG``.
    """

    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    price_scale_invariant = True
    fitted_state = "train_only"
    estimated_decay_bars = None
    refit_every_bars = 1
    training_window_bars = 0

    FIT_LOG = []

    def __init__(self):
        self.memo = None

    @property
    def horizon_bars(self):
        return 21

    @property
    def required_history(self):
        return 3

    def fit(self, close, volume=None):
        type(self).FIT_LOG.append(
            {
                "start": str(close.index[0].date()),
                "end": str(close.index[-1].date()),
                "rows": int(len(close)),
            }
        )
        self.memo = (close.iloc[-1] / close.iloc[0] - 1.0).astype(float)
        return self

    def score(self, close, volume=None):
        if self.memo is None:
            raise ValueError("scored before it was fitted")
        return self.memo.reindex(close.columns)

    def save_state(self, path):
        path.write_text(json.dumps({k: float(v) for k, v in self.memo.items()}, sort_keys=True), encoding="utf-8")

    def load_state(self, path):
        self.memo = pd.Series(json.loads(path.read_text(encoding="utf-8")), dtype=float)
        return self
'''

BOOK_TOML = """
[strategy]
name = "oracle_book"
universe_index = "synthetic_100"
combination = "position_space"

[[strategy.components]]
id = "oracle"
path = "wfe_strategies.oracle:OracleRank"
weight = 1.0
horizon_bars = 21
update_every_bars = 5
valid_for_bars = 10
to_weights = "decile_long_short"

[strategy.components.to_weights_params]
decile = 0.20
max_gross = 1.0
max_symbol_abs_weight = 0.35

[strategy.eligibility]
min_price = 5.0

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
    # The stage 4 recovery loader parses this file rather than only resolving its path, so
    # the fixture declares the keys ``portfolio/recovery.py::RecoveryPolicy`` reads. Their
    # values are the reference ones; the fixture asserts nothing about them.
    "policies/recovery.toml": (
        'on_infeasible = "cancel"\n'
        'on_sign_contradiction = "hold"\n'
        'on_blocked_obligation = "escalate"\n'
        'on_rounding_failure = "scale_batch"\n'
        'preference_response = "record"\n'
    ),
}


@pytest.fixture(scope="module")
def source_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One private project with the synthetic fixture, built once."""
    root = tmp_path_factory.mktemp("wfe") / "source"
    workspace = Workspace(root).ensure()
    write_synthetic_dataset(workspace.data, seed=11, scenarios="all", n_equities=12, n_etfs=2, start=START, end=END)
    (root / PACKAGE).mkdir()
    (root / PACKAGE / "__init__.py").write_text("", encoding="utf-8")
    (root / PACKAGE / "oracle.py").write_text(textwrap.dedent(ORACLE_SIGNAL), encoding="utf-8")
    for relative, content in POLICY_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "specs").mkdir()
    (root / "specs" / "book.toml").write_text(BOOK_TOML, encoding="utf-8")

    before_path = list(sys.path)
    before_modules = set(sys.modules)
    yield root
    sys.path[:] = before_path
    for name in set(sys.modules) - before_modules:
        del sys.modules[name]


def _link(source_root: Path, root: Path) -> Workspace:
    """A workspace sharing the built fixture's data and code through symlinks.

    Its ``results/``, ``trials/``, ``state/`` and ``runs/`` are its own, so one
    test's rows are never another test's history.
    """
    root.mkdir(parents=True)
    for entry in sorted(source_root.iterdir()):
        if entry.name in {"results", "trials", "state", "runs", "logs"}:
            continue
        (root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    return Workspace(root).ensure()


@pytest.fixture
def project(source_root: Path, tmp_path: Path) -> Workspace:
    """A fresh workspace per test, for the tests that write rows or records."""
    return _link(source_root, tmp_path / "project")


@pytest.fixture(scope="module")
def shared_project(source_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    """One workspace for the read-only tests, so they share one evaluation."""
    return _link(source_root, tmp_path_factory.mktemp("wfe_shared") / "project")


@pytest.fixture(scope="module")
def baseline(shared_project: Workspace) -> tuple[EvaluationResult, list[dict]]:
    """One recorded evaluation and the fit panels the oracle saw producing it.

    Module-scoped on purpose: every test below that only *reads* the outcome
    reads the same one, so the file does not pay for a walk-forward run per
    assertion.
    """
    fits = _fit_log(shared_project)
    fits.clear()
    result = _evaluate(shared_project)
    return result, list(fits)


@pytest.fixture
def fit_log(project: Workspace):
    """Every fit panel the oracle saw during one test, in order."""
    fits = _fit_log(project)
    fits.clear()
    yield fits
    fits.clear()


def _fit_log(project: Workspace) -> list:
    load_spec(project.root / "specs" / "book.toml", workspace=project)
    return sys.modules[f"{PACKAGE}.oracle"].OracleRank.FIT_LOG


def _evaluate(project: Workspace, **overrides: object) -> EvaluationResult:
    spec = load_spec(project.root / "specs" / "book.toml", workspace=project)
    options: dict[str, object] = {
        "workspace": project,
        "splitter": SPLITTER,
        "fold_table_id": "wfe_smoke",
        "selection_set_id": "unit_set",
        "registration_id": "unit_set",
        "kind": "new",
    }
    options.update(overrides)
    return walk_forward_evaluate(spec, **options)  # type: ignore[arg-type]


# ---- the train-only boundary ----------------------------------------------


def test_every_fit_stops_at_its_folds_formation_end(baseline) -> None:
    result, fit_log = baseline

    assert len(fit_log) == len(result.folds) > 1, "one fit per fold, and more than one fold"
    for fold, seen in zip(result.folds, fit_log):
        assert (seen["start"], seen["end"]) == fold.train, "the fit panel is the fold's formation window"
        assert seen["end"] < fold.test[0], "no fit saw a bar on or after its fold's first test bar"
        assert seen["rows"] == SPLITTER.formation_bars
    # The same claim, read from the record rather than from the component: the
    # boundary has to be auditable by a reader who did not run the loop.
    for fold in result.record.evaluation["folds"]:
        for component in fold["components"]:
            assert component["fit_window"] == [fold["train"]["start"], fold["train"]["end"]]
            assert component["fit_cutoff"] == fold["train"]["end"]
            assert component["fit_scope"] == "fold_train"
            assert component["fitted_state_ref"], "a train-only fit writes an artifact and records its hash"


def test_a_leaking_fit_would_change_the_answer(project: Workspace, fit_log: list, monkeypatch) -> None:
    """The boundary test above is not decoration: breaking it flips the result.

    The leak simulated here is the one the harness exists to prevent — one fit
    over the whole sample before splitting — and it is applied through the same
    seam the evaluator uses, so nothing else about the run changes.
    """
    honest = _evaluate(project)
    honest_return = float(honest.scored.metrics["total_return"])
    honest_sharpe = float(honest.record.performance["sharpe"]["periodic"])

    dataset = Dataset.open(project.data)
    last_session = str(dataset.manifest.coverage["sessions"][-1])
    whole_sample = dataset.view("research_vintage", last_session, vintage=dataset.manifest.vintage)

    def leaking_prepare(spec, view, decision_bar, **kwargs):
        kwargs["window"] = None
        kwargs["previous"] = ()
        kwargs["elapsed_bars"] = None
        return prepare_components(spec, whole_sample, last_session, **kwargs)

    monkeypatch.setattr("prism.evaluate.prepare_components", leaking_prepare)
    fit_log.clear()
    leaked = _evaluate(project)

    assert all(seen["end"] == last_session for seen in fit_log), "the simulated leak fits past every fold"
    leaked_return = float(leaked.scored.metrics["total_return"])
    leaked_sharpe = float(leaked.record.performance["sharpe"]["periodic"])
    assert (
        honest_return < 0.0 < leaked_return
    ), f"the look-ahead turns a loss into a profit: honest {honest_return:.4f}, leaked {leaked_return:.4f}"
    assert leaked_return - honest_return > 0.4
    assert leaked_sharpe > honest_sharpe


def test_a_fit_window_shorter_than_the_component_needs_is_refused(project: Workspace, fit_log: list) -> None:
    """A short fit is a defect in the geometry, not a fit to attempt anyway."""
    spec = load_spec(project.root / "specs" / "book.toml", workspace=project)
    dataset = Dataset.open(project.data)
    last = str(dataset.manifest.coverage["sessions"][-1])
    view = dataset.view("research_vintage", last, vintage=dataset.manifest.vintage)
    members = sorted(view.membership("synthetic_100"))
    sessions = [str(stamp.date()) for stamp in view.panel(members, last, "close").index]

    with pytest.raises(ValueError, match="needs 3 bars to fit"):
        prepare_components(
            spec,
            view,
            last,
            fits_dir=project.fits(),
            members=members,
            window=(sessions[-2], sessions[-1]),
        )


# ---- the record ------------------------------------------------------------


def test_the_record_carries_the_fields_section_3_12_lists(baseline) -> None:
    result, _ = baseline

    payload = json.loads(result.record_path.read_text(encoding="utf-8"))
    validate_result_record(payload)
    assert ResultRecord.from_dict(payload).to_dict() == payload, "the record round-trips"

    assert payload["record_schema"] == 1
    assert payload["evaluation"]["accounting_engine"] == "historical_constant_weight"
    assert payload["evaluation"]["capital"] == 1.0
    assert payload["evaluation"]["fold_portfolio"] == "reset"
    assert payload["costs"]["cost_refresh"]["cadence"] == "fold"
    assert payload["conventions"]["scoring_basis"] == {"oracle": "price"}
    assert payload["conventions"]["executable_return"] == "next_open_to_open"
    assert payload["status"]["status"] == "finished"
    assert payload["ids"]["run_id"] == result.run_id
    assert payload["ids"]["config_id"] == result.config_id
    assert payload["performance"]["repair"]["attempted"] == 0
    assert any("historical_constant_weight" in warning for warning in payload["warnings"])


def test_the_record_states_the_mode_and_vintage_it_was_cut_under(baseline, shared_project: Workspace) -> None:
    """From stage 3b the record asserts the visibility the data layer enforced.

    Stage 2 wrote ``"unavailable"`` here because nothing enforced a mode. Every
    view this evaluation opened now goes through one declared
    ``_Visibility``, so the record states it — and it is inside ``config_id``,
    which means a run under the other mode is a different configuration rather
    than the same one relabelled.
    """
    result, _ = baseline
    vintage = Dataset.open(shared_project.data).manifest.vintage

    data = result.record.data
    assert data["visibility_mode"] == "research_vintage"
    assert data["vintage"] == vintage
    assert data["unavailable_reason"] is None
    assert data["observed"]["view_mode"] == "research_vintage"
    assert data["observed"]["manifest_vintage_label"] == vintage
    assert result.record.evaluation["holding_process"] == "constant_weight"


# ---- the two accounting engines -------------------------------------------


def _zero_cost_stack() -> CostStack:
    """A stack that charges nothing, every component of it declared (§3.6).

    The comparison this file makes is about the *holding process*, so the cost
    policy is held at zero on both sides — declared zero, not omitted zero,
    which the loader refuses.
    """
    volume = {
        "estimator": "median",
        "window": "trailing",
        "window_bars": 20,
        "refresh": "decision",
        "floor_dollars": 100_000.0,
    }
    return CostStack.from_mapping(
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


def _canonical_settings(**overrides) -> CanonicalSettings:
    # A continuous quantity grid, because the fixture's capital is 1.0 and a
    # whole share is unaffordable there; §3.7 already declares that grid for the
    # weight-space comparison (RoundingSettings.for_reproduction).
    return CanonicalSettings(cost_stack=_zero_cost_stack(), share_increment=0.0, **overrides)


@pytest.fixture(scope="module")
def canonical_workspace(source_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    """Its own workspace: the canonical run's protocol rows are its own history."""
    return _link(source_root, tmp_path_factory.mktemp("wfe_canonical") / "project")


@pytest.fixture(scope="module")
def canonical_baseline(canonical_workspace: Workspace) -> EvaluationResult:
    """The same specification, the same folds, the other accounting engine."""
    return _evaluate(canonical_workspace, engine="canonical", canonical=_canonical_settings())


def test_the_canonical_engine_runs_and_records_a_different_process(canonical_baseline, baseline) -> None:
    reproduction, _ = baseline
    canonical = canonical_baseline

    assert canonical.record.evaluation["accounting_engine"] == "canonical"
    assert canonical.record.evaluation["holding_process"] == "fixed_share"
    assert canonical.record.conventions["executable_return"] == "next_open_fill_marked_close_to_close"
    assert canonical.record.evaluation["engine"]["n_fills"] > 0, "shares actually moved"
    assert canonical.record.evaluation["engine"]["settings"]["share_increment"] == 0.0
    assert canonical.record.evaluation["engine"]["settings"]["cost_stack_id"]

    # Two engines, two identities. Nothing about the two records lets one be
    # read as a restatement of the other.
    assert canonical.config_id != reproduction.config_id
    assert canonical.record.ids["config_id"] == canonical.config_id
    assert canonical.fold_table.content_hash == reproduction.fold_table.content_hash

    # And two numbers. The same folds, the same decisions and a cost policy of
    # declared zeros on both sides still give different returns, because the
    # constant-weight book re-levers between decisions and the fixed-share book
    # does not. A test that only checked the labels would pass on an engine that
    # quietly ran the other one.
    assert float(canonical.scored.metrics["total_return"]) != float(reproduction.scored.metrics["total_return"])
    assert float(canonical.scored.costs["turnover"].sum()) > 0.0


def test_the_caveat_belongs_to_the_engine_that_earned_it(canonical_baseline, baseline) -> None:
    reproduction, _ = baseline

    assert any(REPRODUCTION_CAVEAT == warning for warning in reproduction.record.warnings)
    assert not any(CANONICAL_NOTE == warning for warning in reproduction.record.warnings)
    assert any(CANONICAL_NOTE == warning for warning in canonical_baseline.record.warnings)
    assert not any("historical_constant_weight" in warning for warning in canonical_baseline.record.warnings)


@pytest.mark.parametrize("claimed", ["historical_constant_weight", "canonical"])
def test_a_record_carrying_the_wrong_engine_label_fails_validation(canonical_baseline, baseline, claimed) -> None:
    """§3.8: the two engines are never silently substituted.

    Relabelling one engine's record with the other's name leaves every other
    field describing the process that actually ran, and the validator refuses
    the record rather than letting a 2,500-per-10,000 difference in holding
    process travel under the wrong name.
    """
    source = canonical_baseline if claimed == "historical_constant_weight" else baseline[0]
    payload = json.loads(source.record_path.read_text(encoding="utf-8"))
    validate_result_record(payload)  # honest to begin with

    payload["evaluation"]["accounting_engine"] = claimed
    with pytest.raises(RecordSchemaError) as excinfo:
        validate_result_record(payload)
    message = str(excinfo.value)
    assert "evaluation.holding_process" in message
    assert "conventions.executable_return" in message
    assert "warnings" in message


def test_a_record_that_drops_the_holding_process_is_refused(baseline) -> None:
    result, _ = baseline
    payload = json.loads(result.record_path.read_text(encoding="utf-8"))

    del payload["evaluation"]["holding_process"]
    with pytest.raises(RecordSchemaError, match="evaluation.holding_process: missing"):
        validate_result_record(payload)


def test_the_two_engines_act_on_the_same_decision_bars(canonical_baseline, baseline) -> None:
    """The masked target frame drops only rows the reproduction engine ignored too.

    ``_canonical_decisions`` is the seam where the two engines could quietly
    diverge — feed the canonical engine one more row and it rebalances daily,
    which is a different strategy. The rows it keeps are exactly the bars on
    which the reproduction engine's own filled weights changed.
    """
    reproduction, _ = baseline
    targets = reproduction.full.target_weights
    masked, kept = _canonical_decisions(targets)

    fills = reproduction.full.fill_weights
    changed = {
        str(stamp.date())
        for previous, stamp in zip(fills.index[:-1], fills.index[1:])
        if not fills.loc[stamp].equals(fills.loc[previous])
    }
    # A decision at close t moves the filled weights at t+1, so the bars the
    # mask keeps are the bars before the ones that moved.
    moved = {
        str(fills.index[position - 1].date())
        for position, stamp in enumerate(fills.index)
        if position and str(stamp.date()) in changed
    }
    assert moved <= set(kept)
    assert len(kept) < int(targets.notna().any(axis=1).sum()), "restated rows were dropped"
    assert canonical_baseline.record.evaluation["engine"]["n_decision_bars"] == len(kept)


def test_the_canonical_engine_needs_a_declared_cost_stack(project: Workspace) -> None:
    with pytest.raises(RunRefused, match="declared CostStack"):
        _evaluate(project, engine="canonical")

    assert not list(project.trials.glob("*.jsonl")), "a refusal is not an attempt"
    assert not list(project.runs.glob("*.jsonl"))


def test_a_cost_stack_declared_for_the_reproduction_engine_is_refused(project: Workspace) -> None:
    """It would be ignored, and an ignored cost policy is the silent defect."""
    with pytest.raises(RunRefused, match="would ignore all three"):
        _evaluate(project, engine="historical_constant_weight", canonical=_canonical_settings())

    assert not list(project.trials.glob("*.jsonl"))


def test_the_recorded_fold_geometry_is_the_tables_content_hash(baseline) -> None:
    result, _ = baseline

    geometry = result.record.evaluation["fold_geometry"]
    assert geometry["sha256"] == result.fold_table.content_hash
    assert geometry["n_folds"] == len(result.folds) == len(result.fold_table)
    for recorded, fold in zip(geometry["folds"], result.folds):
        assert recorded["train"]["start"][:10] == fold.train[0]
        assert recorded["test"]["end"][:10] == fold.test[1]
    # The table itself is the authority, so the record's identity moves with it.
    assert result.record.evaluation["fold_geometry"]["table_id"] == "wfe_smoke"


def test_a_record_missing_a_section_3_12_field_is_refused(baseline) -> None:
    result, _ = baseline
    payload = json.loads(result.record_path.read_text(encoding="utf-8"))

    del payload["evaluation"]["accounting_engine"]
    payload["flags"]["fill_source"] = "guessed"

    with pytest.raises(RecordSchemaError) as excinfo:
        validate_result_record(payload)
    message = str(excinfo.value)
    assert "evaluation.accounting_engine: missing" in message
    assert "flags.fill_source" in message


def test_a_synthetic_run_is_capped_at_pipeline_runs(baseline) -> None:
    result, _ = baseline

    assert result.record.flags["synthetic"] is True
    assert result.record.flags["evidence_tier"] in {"no_claim", "mechanics_clean"}


def test_a_run_on_a_synthetic_fixture_is_routed_to_the_engineering_channel(baseline, shared_project) -> None:
    """§3.12: a synthetic run is in no selection set and enters no attempt count.

    The caller declared ``run_channel="evidence"``; whether the dataset is
    synthetic is a fact about the manifest, not a flag the caller chooses, so
    the routing follows the manifest and says so in the record.
    """
    result, _ = baseline

    assert result.record.flags["run_channel"] == "engineering"
    assert result.log_path == shared_project.runs / "engineering.jsonl"
    assert not (shared_project.trials / "unit_set.jsonl").exists()
    assert result.record.selection["n_trials_observed"] == 0
    assert result.record.selection["excluded_rows"].get("engineering")
    assert any("routed to the engineering channel" in warning for warning in result.record.warnings)


def test_log_routing_follows_the_registration_and_the_channel(shared_project: Workspace) -> None:
    """The three routes of §3.12, "Log routing", as a rule rather than a run."""
    assert resolve_log_path(shared_project, selection_set_id="momentum_v1", run_channel="evidence") == (
        shared_project.trials / "momentum_v1.jsonl"
    )
    assert resolve_log_path(shared_project, selection_set_id=None, run_channel="evidence") == (
        shared_project.trials / "exploratory.jsonl"
    )
    assert resolve_log_path(shared_project, selection_set_id="momentum_v1", run_channel="engineering") == (
        shared_project.runs / "engineering.jsonl"
    )
    with pytest.raises(RunRefused, match="unknown run_channel"):
        resolve_log_path(shared_project, selection_set_id=None, run_channel="rumour")


# ---- the walk-forward mechanics -------------------------------------------


def test_metrics_are_taken_on_test_rows_only(baseline) -> None:
    result, _ = baseline

    expected = sum(len(pd.date_range(start=fold.test[0], end=fold.test[1], freq="D")) for fold in result.folds)
    assert len(result.scored.returns) < expected, "calendar days are not sessions; the scored index is sessions"
    scored = set(result.scored.returns.index)
    for fold in result.folds:
        in_fold = [stamp for stamp in scored if fold.test[0] <= str(stamp.date()) <= fold.test[1]]
        assert in_fold, f"fold {fold.fold} contributed no scored bar"
    outside = [stamp for stamp in scored if str(stamp.date()) < result.folds[0].test[0]]
    assert not outside, "no formation bar is scored"
    assert len(result.full.returns) > len(result.scored.returns), "one continuous simulation, scored on part of it"


def test_reset_flattens_inside_the_fold(baseline) -> None:
    """§3.8: the liquidation fills at the following open *inside* the fold."""
    result, _ = baseline
    targets = result.full.target_weights
    fills = result.full.fill_weights

    for fold in result.folds:
        rows = targets.loc[(targets.index >= fold.test[0]) & (targets.index <= fold.test[1])].dropna(how="all")
        assert float(rows.iloc[-1].abs().sum()) == 0.0
        assert float(rows.iloc[-2].abs().sum()) == 0.0
        last_fill = fills.loc[fills.index == pd.Timestamp(fold.test[1], tz=fills.index.tz)]
        assert float(last_fill.abs().sum().sum()) == 0.0, "the book is flat on the fold's last bar"
        assert fold.turnover["boundary"] > 0.0
        assert fold.turnover["cost_boundary"] >= 0.0


def test_the_carry_mode_does_not_force_a_flat(project: Workspace) -> None:
    result = _evaluate(project, fold_portfolio="carry")

    assert result.record.evaluation["fold_portfolio"] == "carry"
    flat_ends = [
        fold
        for fold in result.folds
        if float(
            result.full.fill_weights.loc[pd.Timestamp(fold.test[1], tz=result.full.fill_weights.index.tz)].abs().sum()
        )
        == 0.0
    ]
    assert not flat_ends, "carry holds the portfolio across the boundary"
    assert all(fold.turnover["boundary"] == 0.0 for fold in result.folds)


def test_the_calibration_hook_runs_once_per_fold_on_training_data_only(project: Workspace) -> None:
    seen: list[tuple[int, str, str, str]] = []

    def hook(request):
        panel = request.scoring_panels["price"]
        seen.append((request.fold, request.train_start, request.train_end, str(panel.index[-1].date())))
        return {"theta_hat": 0.5, "n_periods": int(len(panel))}

    result = _evaluate(project, calibration_hook=hook)

    assert [entry[0] for entry in seen] == [fold.fold for fold in result.folds]
    for (_, train_start, train_end, panel_end), fold in zip(seen, result.folds):
        assert (train_start, train_end) == fold.train
        assert panel_end == fold.train[1], "the hook never sees a bar past the formation end"
    block = result.record.evaluation["calibration"]
    assert block["hook"] == "declared"
    assert [entry["fit"]["theta_hat"] for entry in block["fit_per_fold"]] == [0.5] * len(result.folds)


def test_an_excluded_fold_is_recorded_and_scored_nowhere(project: Workspace) -> None:
    def hook(request):
        if request.fold == 1:
            return {"excluded": True, "exclusion_reason": "fewer than min_blocks blocks"}
        return {"excluded": False}

    result = _evaluate(project, calibration_hook=hook)

    excluded = result.record.evaluation["excluded_folds"]
    assert excluded == [{"index": 1, "reason": "fewer than min_blocks blocks"}]
    dropped = result.folds[1]
    scored = {str(stamp.date()) for stamp in result.scored.returns.index}
    assert not [day for day in scored if dropped.test[0] <= day <= dropped.test[1]]
    assert result.record.performance["sample"]["n_excluded_folds"] == 1


def test_an_exclusion_without_a_reason_is_refused(project: Workspace) -> None:
    with pytest.raises(ValueError, match="without naming a reason"):
        _evaluate(project, calibration_hook=lambda request: {"excluded": True})


# ---- the run-recording protocol, through the evaluator ---------------------


def test_the_run_writes_a_started_row_and_one_terminal_row(baseline, shared_project: Workspace) -> None:
    result, _ = baseline

    rows = TrialLog(result.log_path).verify()
    # The fixture is synthetic, so the run is on the engineering channel
    # whatever the caller declared — see the routing test below.
    assert result.log_path == shared_project.runs / "engineering.jsonl"
    assert [row["event"] for row in rows] == ["started", "finished"]
    assert {row["run_id"] for row in rows} == {result.run_id}
    assert rows[0]["config_id"] == result.config_id
    assert rows[0]["data_id"] == result.data_id
    assert rows[0]["heartbeat_at"] and rows[0]["pid"] and rows[0]["host"]
    assert rows[1]["summary"]["record"] == str(result.record_path)


def test_an_unregistered_evaluation_carries_no_selection_set(project: Workspace) -> None:
    result = _evaluate(project, selection_set_id=None, registration_id=None, kind="exploratory")

    assert result.record.flags["exploratory"] is True
    assert result.record.ids["selection_set_ids"] == []
    assert result.record.ids["registration_id"] is None


def test_an_engineering_run_lands_outside_every_selection_set(project: Workspace) -> None:
    result = _evaluate(project, run_channel="engineering", selection_set_id="unit_set")

    assert result.log_path == project.runs / "engineering.jsonl"
    assert result.record.flags["run_channel"] == "engineering"
    assert not (project.trials / "unit_set.jsonl").exists()
    assert result.record.selection["n_trials_observed"] == 0


def test_a_failure_during_evaluation_writes_a_failed_row(project: Workspace, monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("the accounting engine fell over")

    monkeypatch.setattr("prism.evaluate.backtest_target_weights", explode)

    with pytest.raises(RuntimeError, match="fell over"):
        _evaluate(project)

    rows = TrialLog(project.runs / "engineering.jsonl").verify()
    assert [row["event"] for row in rows] == ["started", "failed"]
    assert "fell over" in str(rows[1]["reason"])
    assert not list(project.results.glob("*/record.json")), "a failed run writes no record"


def test_a_failure_resolving_identity_is_still_an_attempt(project: Workspace) -> None:
    """§3.12: a specification that never resolved is still an attempt."""
    elsewhere = FoldTable(
        table_id="not_this_sample",
        folds=(
            FoldBoundary(
                index=0,
                formation_start=pd.Timestamp("1999-01-04"),
                formation_end=pd.Timestamp("1999-06-30"),
                test_start=pd.Timestamp("1999-07-01"),
                test_end=pd.Timestamp("1999-09-30"),
            ),
        ),
    )

    with pytest.raises(FoldGeometryError):
        _evaluate(project, fold_table=elsewhere, splitter=None)

    # Nothing resolved, so nothing is known about the dataset: the row goes to
    # the log the caller declared.
    rows = TrialLog(project.trials / "unit_set.jsonl").verify()
    assert [row["event"] for row in rows] == ["failed"]
    assert rows[0]["config_id"] is None, "no configuration identity resolved, and none is claimed"
    assert "identity resolution failed" in str(rows[0]["reason"])


def test_a_crashed_run_is_closed_by_the_next_invocation(project: Workspace) -> None:
    """The next invocation closes a *dead* run, and only through reconcile."""
    log = TrialLog(project.runs / "engineering.jsonl")
    log.append(
        {
            "ts": "2020-01-01T00:00:00.000000Z",
            "event": "started",
            "run_id": "run-crashed",
            "config_id": "0" * 64,
            "data_id": "1" * 64,
            "kind": "new",
            "arm_id": None,
            "selection_set_id": "unit_set",
            "capital": 1.0,
            "synthetic": False,
            "run_channel": "evidence",
            "summary": None,
            "reason": None,
            "pid": 1,
            "host": "a-host-that-is-not-this-one",
            "heartbeat_at": "2020-01-01T00:00:00.000000Z",
        }
    )

    result = _evaluate(project)

    events = [(row["event"], row["run_id"]) for row in TrialLog(result.log_path).verify()]
    assert events == [
        ("started", "run-crashed"),
        ("incomplete", "run-crashed"),
        ("started", result.run_id),
        ("finished", result.run_id),
    ]
    assert result.reconciled["closed"][0]["run_id"] == "run-crashed"


# ---- refusals --------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"capital": 0.0}, "capital must be > 0"),
        ({"engine": "made_up"}, "unknown engine"),
        ({"kind": "made_up"}, "unknown run kind"),
        ({"fold_portfolio": "hold_forever"}, "unknown fold_portfolio"),
        ({"cost_refresh": "hourly"}, "unknown cost_refresh"),
        ({"visibility_mode": "whenever"}, "unknown visibility_mode"),
        ({"decision_time": "15:45 America/New_York"}, "decision_time belongs to replay_recorded"),
    ],
)
def test_a_refused_run_writes_no_row(project: Workspace, overrides: dict, match: str) -> None:
    with pytest.raises(RunRefused, match=match):
        _evaluate(project, **overrides)

    assert not list(project.trials.glob("*.jsonl"))
    assert not list(project.runs.glob("*.jsonl"))


# ---- the state-coupled path (§3.7, §3.8) -----------------------------------


@pytest.fixture(scope="module")
def coupled_workspace(source_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    """Its own workspace: a state-coupled run's protocol rows are its own history."""
    return _link(source_root, tmp_path_factory.mktemp("wfe_coupled") / "project")


@pytest.fixture(scope="module")
def coupled_baseline(coupled_workspace: Workspace) -> EvaluationResult:
    return _evaluate(
        coupled_workspace,
        engine="canonical",
        canonical=_canonical_settings(),
        decision_coupling="account_state",
        fold_portfolio="carry",
    )


def test_the_state_coupled_path_rounds_and_revalidates_every_decision(coupled_baseline) -> None:
    """§3.7: every decision reaches batch rounding, and the record counts the outcome.

    "How often repair fails" is a recorded count, not a narrative, and this is the field it
    is read from: the repair block's outcomes must add up to the decisions that were made.
    """
    construction = coupled_baseline.record.evaluation["construction"]
    assert construction["decision_coupling"] == "account_state"
    assert construction["n_decisions"] > 0
    repair = coupled_baseline.record.performance["repair"]
    assert repair["attempted"] + repair["no_action"] == construction["n_decisions"]
    assert repair["succeeded"] + repair["scaled_batch"] + repair["dropped_batch"] == repair["attempted"]
    assert set(construction["paths"]) <= {"normal", "recovery", "no_action"}


def test_the_state_coupled_path_carries_one_account_and_records_its_coupling(coupled_baseline) -> None:
    """The coupling is part of the identity: two couplings are two evaluations (§3.12)."""
    assert coupled_baseline.record.evaluation["accounting_engine"] == "canonical"
    assert coupled_baseline.record.evaluation["fold_portfolio"] == "carry"
    assert coupled_baseline.record.evaluation["engine"]["decision_coupling"] == "account_state"
    assert len(coupled_baseline.scored.returns) > 0


def test_the_state_coupled_path_refuses_a_fold_reset(project: Workspace) -> None:
    """A forced fold-end liquidation is not what one carried account does (§3.8)."""
    with pytest.raises(RunRefused, match="fold_portfolio"):
        _evaluate(
            project,
            engine="canonical",
            canonical=_canonical_settings(),
            decision_coupling="account_state",
            fold_portfolio="reset",
        )


def test_the_state_coupled_path_refuses_the_reproduction_engine(project: Workspace) -> None:
    """The reproduction engine carries no marked position book to decide against (§3.8)."""
    with pytest.raises(RunRefused, match="account_state"):
        _evaluate(project, engine="historical_constant_weight", decision_coupling="account_state")


def test_the_state_coupled_path_refuses_a_declared_band_it_would_not_apply(project: Workspace) -> None:
    """The account_state path applies no no-trade band, so declaring one is refused (N7).

    docs/construction_band_audit_2026-09-24.md: records reported a ``closed_form`` band that
    never acted on this path. A declaration the path cannot honour is refused before a row.
    """
    variant = project.root / "specs_band"
    variant.mkdir()
    (variant / "book.toml").write_text(
        BOOK_TOML.replace('[construction]\nkind = "heuristic"\n',
                          '[construction]\nkind = "heuristic"\n\n[construction.band]\nmode = "closed_form"\n'),
        encoding="utf-8",
    )
    spec = load_spec(variant / "book.toml", workspace=project)
    with pytest.raises(RunRefused, match="no-trade band"):
        walk_forward_evaluate(spec, workspace=project, splitter=SPLITTER, fold_table_id="wfe_smoke",
                              selection_set_id="unit_set", registration_id="unit_set", kind="new",
                              engine="canonical", canonical=_canonical_settings(),
                              decision_coupling="account_state", fold_portfolio="carry")
