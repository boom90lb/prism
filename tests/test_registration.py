"""The stage 4 registration, its arms, and strict mode over them (spec §3.13, D5, D8).

The failures these tests are written against, in order of what they would cost:

* a run that is *not* a declared arm being accepted as one — a standardization
  or weighting alternative slipped in as a free diagnostic, or an arm tuned
  after a disappointing result — because that is the failure the whole
  registration exists to prevent and it is invisible in the record afterwards;
* the two convex arms silently ceasing to share their calibration family or
  their risk model, which turns "the uncertainty adjustment cost this much" into
  a comparison of two unrelated configurations;
* a convex arm running at ``risk_aversion = 0`` with no variance budget, where
  the factor covariance is inert and the answer is a maximal-concentration
  corner nobody declared (decision D8);
* a declared path that does not exist, or a fold table whose geometry does not
  match the sample the registration states, found on the day the run is
  launched rather than the day the file is written.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from prism.live.risk_profile import load_declared_profile
from prism.registration import Registration, RegistrationError, load_registration
from prism.spec import load_spec
from prism.validation.trials import ENGINE_EXECUTABLE_RETURN
from prism.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = "registrations/2026-09-construction.toml"
PROFILE = "examples/reference/profiles/experiment_v1.toml"

#: Everything a copy of the project needs to load the stage 4 registration.
_PROJECT_FILES: tuple[str, ...] = (
    REGISTRATION,
    PROFILE,
    "calibrations/shared_slope_point.toml",
    "calibrations/shared_slope_p80.toml",
    "examples/reference/risk/factor_diagonal.toml",
    "folds/demotion_b1.toml",
    "examples/momentum/reference.toml",
    "examples/reference/costs/retail_v1.toml",
    "examples/reference/policies/monitoring.toml",
    "examples/reference/policies/recovery.toml",
    "examples/reference/profiles/execution_alpaca.toml",
    "examples/reference/profiles/paper.toml",
)


#: An evaluation context that agrees with everything the registration freezes.
#: Only the run-time facts (the data id, the vintage) are fixtures.
#:
#: ``executable_return`` comes from the *engine's* mapping, not the
#: registration's ``[frozen]`` value: the two keys share a name and not a
#: vocabulary — the registration freezes the calibration's label convention
#: (``open_to_open``) and a canonical run records
#: ``next_open_fill_marked_close_to_close``. A context that echoed the
#: registration would hide a strict-mode check that refuses every real run.
def _context(registration: Registration, *, capital: float = 100000.0, **overrides: Any) -> dict[str, Any]:
    context: dict[str, Any] = {
        "visibility_mode": registration.frozen.visibility_mode,
        "vintage": "2026-06-16",
        "data_id": "fixture-data-id",
        "fold_table_sha256": registration.fold_table.content_hash,
        "fold_portfolio": registration.frozen.fold_portfolio,
        "accounting_engine": registration.frozen.accounting_engine,
        "executable_return": ENGINE_EXECUTABLE_RETURN[registration.frozen.accounting_engine],
        "capital": capital,
        "cost_refresh": registration.frozen.cost_refresh,
        "price_basis": "as_traded",
        "canonical": None,
    }
    context.update(overrides)
    return context


@pytest.fixture(scope="module")
def shipped() -> Registration:
    """The registration as it ships, loaded against the repository as the workspace."""
    return load_registration(REPO_ROOT / REGISTRATION, workspace=Workspace(REPO_ROOT))


def _project(tmp_path: Path) -> Workspace:
    """A writable copy of the project files the registration reads."""
    for relative in _PROJECT_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    return Workspace(tmp_path)


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{path} does not contain {old!r}; the fixture edit would be a no-op"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# --------------------------------------------------------------------------
# The file as it ships
# --------------------------------------------------------------------------


def test_registration_parses_and_declares_exactly_three_arms(shipped: Registration) -> None:
    payload = tomllib.loads((REPO_ROOT / REGISTRATION).read_text(encoding="utf-8"))
    assert [arm["id"] for arm in payload["arms"]] == [
        "decile_control",
        "convex_point",
        "convex_p80_one_sided",
    ]
    assert shipped.arm_ids == ("decile_control", "convex_point", "convex_p80_one_sided")
    assert shipped.strict is True
    # One attempt is one arm id, so three arms across six scenarios spend three.
    assert shipped.trial_budget == 3
    assert len(shipped.frozen.capital_scenarios) == 6

    kinds = {arm.id: arm.kind for arm in shipped.arms}
    assert kinds == {"decile_control": "heuristic", "convex_point": "convex", "convex_p80_one_sided": "convex"}
    confidences = {arm.id: arm.confidence for arm in shipped.arms}
    assert confidences == {"decile_control": None, "convex_point": None, "convex_p80_one_sided": 0.80}


def test_every_declared_input_path_exists(shipped: Registration) -> None:
    """Every path the registration names is a file, bar the panel the run itself freezes."""
    declared = [
        shipped.shared.risk_model,
        shipped.shared.calibration_family,
        shipped.fold_table_path,
        shipped.frozen.cost_stack,
        shipped.frozen.execution_profile,
        shipped.benchmark.control,
        *[arm.profile for arm in shipped.arms],
        *[arm.calibration for arm in shipped.arms if arm.calibration is not None],
    ]
    for path in declared:
        assert path.is_file(), f"{path} is declared and does not exist"
    # The freeze record is the one declared path the run's own opening step
    # writes (`prism evaluate --freeze-scores`), so it is required when a run is
    # authorized, not when the file is read.
    assert shipped.frozen.freeze_record.name.endswith(".freeze.json")


def test_component_import_path_is_the_control_book_s_own(shipped: Registration) -> None:
    control = tomllib.loads(shipped.benchmark.control.read_text(encoding="utf-8"))
    declared = {component.id: component.path for component in shipped.components}
    for entry in control["strategy"]["components"]:
        assert declared[entry["id"]] == entry["path"]


def test_a_registration_scoring_basis_that_contradicts_the_component_refuses(tmp_path: Path) -> None:
    """The basis fixes the panel the score and the label are computed on."""
    workspace = _project(tmp_path)
    _edit(tmp_path / REGISTRATION, 'scoring_basis = "price"', 'scoring_basis = "total_return"')
    registration = load_registration(tmp_path / REGISTRATION, workspace=workspace)
    with pytest.raises(RegistrationError, match="scoring_basis"):
        registration.arm_specification("decile_control", workspace=workspace)


def test_promotion_rule_is_none_and_cannot_promote(shipped: Registration) -> None:
    """No arm promotes out of this registration, and the rule says so structurally."""
    assert shipped.promotion_rule.kind == "none"
    assert shipped.promotion_rule.promotes() is False
    assert shipped.promotion_rule.thresholds == {}
    assert shipped.promotion_rule.callback is None
    assert "no arm promotes" in (shipped.promotion_rule.note or "").lower()
    assert shipped.stop_rule.kind == "structured"


def test_sample_dates_are_the_fold_table_s_own(shipped: Registration) -> None:
    assert shipped.sample.first_decision == shipped.fold_table.first_decision.date()
    assert shipped.sample.last_decision == shipped.fold_table.last_decision.date()
    assert shipped.fold_table.n_folds == 21


# --------------------------------------------------------------------------
# The experiment profile and the variance budget (decision D8)
# --------------------------------------------------------------------------


def test_experiment_profile_declares_a_shared_absolute_variance_budget(shipped: Registration) -> None:
    convex = [arm for arm in shipped.arms if arm.kind == "convex"]
    assert len({arm.profile for arm in convex}) == 1, "both convex arms name one profile"

    profile = load_declared_profile(convex[0].profile, tighten_only=False)
    budget = profile.limit("variance_budget")
    assert budget is not None
    assert budget.cls.value == "preference"
    # Fixed before the run: an absolute per-horizon standard deviation, not a
    # rule that reads a value off the run it is constraining.
    assert budget.params["mode"] == "absolute"
    assert budget.value() == pytest.approx(0.0332)
    # And stated over the horizon the risk snapshot is built for, so it bounds
    # the quantity it names.
    risk = tomllib.loads(shipped.shared.risk_model.read_text(encoding="utf-8"))
    assert int(budget.params["horizon_bars"]) == int(risk["horizon_bars"])
    # Zero risk aversion is what makes the budget the only channel the risk
    # model acts through.
    assert float(shipped.shared.convex_construction["risk_aversion"]) == 0.0


def test_budget_is_the_reference_book_s_own_realized_volatility(shipped: Registration) -> None:
    """The number is anchored to recorded evidence, not chosen to make an arm win.

    0.007249325 per session over 1308 recorded sessions, times sqrt(21) for the
    forecast horizon, is 0.03322058; the declared budget is that to three
    significant figures.
    """
    profile = load_declared_profile(REPO_ROOT / PROFILE, tighten_only=False)
    budget = profile.limit("variance_budget")
    assert budget is not None
    assert budget.value() == pytest.approx(0.007249325 * 21**0.5, rel=1e-3)


def test_convex_arm_without_a_variance_budget_is_refused(tmp_path: Path) -> None:
    """risk_aversion = 0 and no budget leaves the factor covariance inert (D8)."""
    workspace = _project(tmp_path)
    profile = tmp_path / PROFILE
    text = profile.read_text(encoding="utf-8")
    head, marker, rest = text.partition('[[limits]]\nname = "variance_budget"')
    assert marker, "the fixture profile no longer declares a variance budget"
    _, participation, tail = rest.partition('[[limits]]\nname = "participation_per_name"')
    profile.write_text(head + participation + tail, encoding="utf-8")

    with pytest.raises(RegistrationError, match="maximal-concentration corner"):
        load_registration(tmp_path / REGISTRATION, workspace=workspace)


def test_convex_arms_with_different_budgets_are_refused(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    loose = tmp_path / "examples/reference/profiles/experiment_v1_loose.toml"
    loose.write_text(
        (tmp_path / PROFILE).read_text(encoding="utf-8").replace("value = 0.0332", "value = 0.0500"),
        encoding="utf-8",
    )
    _edit(
        tmp_path / REGISTRATION,
        'id = "convex_p80_one_sided"\nprofile = "examples/reference/profiles/experiment_v1.toml"',
        'id = "convex_p80_one_sided"\nprofile = "examples/reference/profiles/experiment_v1_loose.toml"',
    )
    with pytest.raises(RegistrationError, match="shared by both convex arms"):
        load_registration(tmp_path / REGISTRATION, workspace=workspace)


# --------------------------------------------------------------------------
# The convex arms share their calibration and risk model by construction
# --------------------------------------------------------------------------


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        label = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{label}."))
        else:
            flat[label] = value
    return flat


def test_convex_arms_differ_in_the_calibration_and_in_nothing_else(shipped: Registration) -> None:
    """The identity of the two convex arms differs in one field: the calibration content.

    This is the check that makes "the arms share their mapping and their risk
    model" a property of the files rather than a promise: the risk-model digest,
    every policy digest, the component identity, and the resolved specification
    are compared field by field.
    """
    workspace = Workspace(REPO_ROOT)
    point = shipped.arm_specification("convex_point", workspace=workspace)
    p80 = shipped.arm_specification("convex_p80_one_sided", workspace=workspace)

    left = _flatten(point.identity_payload(evaluation_context=None)["common"])
    right = _flatten(p80.identity_payload(evaluation_context=None)["common"])
    assert set(left) == set(right)
    differing = {key for key in left if left[key] != right[key]}
    assert differing == {"construction.calibration_content"}

    assert point.construction.risk_model == p80.construction.risk_model == shipped.shared.risk_model
    assert point.config_id() != p80.config_id()


def test_the_two_calibration_files_differ_in_confidence_alone(shipped: Registration) -> None:
    family = tomllib.loads(shipped.shared.calibration_family.read_text(encoding="utf-8"))
    p80 = tomllib.loads(shipped.arm("convex_p80_one_sided").calibration.read_text(encoding="utf-8"))
    assert set(p80) - set(family) == {"confidence"}
    assert {key for key in family if family[key] != p80.get(key)} == set()
    assert p80["confidence"] == 0.80


def test_a_calibration_that_drifts_from_the_family_refuses_the_registration(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    _edit(
        tmp_path / "calibrations/shared_slope_p80.toml",
        "shrink_pseudo_periods = 12.0",
        "shrink_pseudo_periods = 6.0",
    )
    with pytest.raises(RegistrationError, match="shrink_pseudo_periods"):
        load_registration(tmp_path / REGISTRATION, workspace=workspace)


def test_a_shared_restatement_that_disagrees_with_the_file_refuses(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    _edit(tmp_path / REGISTRATION, 'standardize = "rank_normal"', 'standardize = "zscore"')
    with pytest.raises(RegistrationError, match="the run reads"):
        load_registration(tmp_path / REGISTRATION, workspace=workspace)


def test_an_arm_naming_another_risk_model_refuses(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    other = tmp_path / "examples/reference/risk/factor_diagonal_k5.toml"
    other.write_text(
        (tmp_path / "examples/reference/risk/factor_diagonal.toml")
        .read_text(encoding="utf-8")
        .replace("n_factors = 15", "n_factors = 5"),
        encoding="utf-8",
    )
    _edit(
        tmp_path / REGISTRATION,
        'calibration = "calibrations/shared_slope_p80.toml"',
        'calibration = "calibrations/shared_slope_p80.toml"\nrisk_model = "examples/reference/risk/factor_diagonal_k5.toml"',
    )
    with pytest.raises(RegistrationError, match="share one risk model"):
        load_registration(tmp_path / REGISTRATION, workspace=workspace)


def test_a_calibration_sign_conflicting_with_the_component_refuses(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    for name in ("shared_slope_point", "shared_slope_p80"):
        _edit(tmp_path / f"calibrations/{name}.toml", 'expected_sign = "positive"', 'expected_sign = "negative"')
    with pytest.raises(RegistrationError, match="expected_sign"):
        load_registration(tmp_path / REGISTRATION, workspace=workspace)


# --------------------------------------------------------------------------
# Strict mode
# --------------------------------------------------------------------------


# ``check_strict`` requires the frozen score panel, which the run's opening
# command materializes and which does not exist in the repository. The arm
# identity is therefore exercised with ``check_freeze=False`` below, and the
# freeze binding has tests of its own.


def test_each_declared_arm_is_accepted_under_the_declared_context(shipped: Registration) -> None:
    workspace = Workspace(REPO_ROOT)
    context = _context(shipped)
    for arm_id, config_id in shipped.arm_config_ids(evaluation_context=context, workspace=workspace).items():
        assert (
            shipped.check_strict(
                config_id=config_id, evaluation_context=context, workspace=workspace, check_freeze=False
            )
            == arm_id
        )


def test_a_weighting_alternative_is_refused_because_it_is_not_a_declared_arm(tmp_path: Path) -> None:
    """A standardization or weighting variant is a declared arm or it is not run."""
    workspace = _project(tmp_path)
    registration = load_registration(tmp_path / REGISTRATION, workspace=workspace)

    variant_calibration = tmp_path / "calibrations/shared_slope_point_per_observation.toml"
    variant_calibration.write_text(
        (tmp_path / "calibrations/shared_slope_point.toml")
        .read_text(encoding="utf-8")
        .replace('weighting = "per_date_equal"', 'weighting = "per_observation_equal"'),
        encoding="utf-8",
    )
    spec_path = tmp_path / "specs/weighting_variant.toml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(_variant_spec(tmp_path, calibration=variant_calibration.name), encoding="utf-8")

    context = _context(registration)
    variant = load_spec(spec_path, workspace=workspace).config_id(evaluation_context=context)
    with pytest.raises(RegistrationError) as excinfo:
        registration.check_strict(
            config_id=variant, evaluation_context=context, workspace=workspace, check_freeze=False
        )
    message = str(excinfo.value)
    for arm_id in registration.arm_ids:
        assert arm_id in message, "the refusal names the arm ids it accepts"


def test_strict_mode_keys_on_identity_not_on_provenance(tmp_path: Path) -> None:
    """A hand-written specification that *is* a declared arm is accepted as that arm.

    The dual of the test above, and the reason strict mode is defined on
    ``config_id``: an arm is a configuration, not a file path or a command line.
    """
    workspace = _project(tmp_path)
    registration = load_registration(tmp_path / REGISTRATION, workspace=workspace)

    spec_path = tmp_path / "specs/decile_by_hand.toml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    control = (tmp_path / "examples/momentum/reference.toml").read_text(encoding="utf-8")
    control = control.replace(
        'profile = "examples/reference/profiles/paper.toml"',
        'profile = "examples/reference/profiles/experiment_v1.toml"',
    ).replace(
        'step_order = "construct_cap_band_participation"',
        'step_order = "construct_cap_band_participation"\neligibility_exit = "reduce_only"',
    )
    spec_path.write_text(control, encoding="utf-8")

    context = _context(registration)
    by_hand = load_spec(spec_path, workspace=workspace).config_id(evaluation_context=context)
    assert (
        registration.check_strict(
            config_id=by_hand, evaluation_context=context, workspace=workspace, check_freeze=False
        )
        == "decile_control"
    )


def test_a_capital_scenario_outside_the_declaration_is_refused(shipped: Registration) -> None:
    workspace = Workspace(REPO_ROOT)
    context = _context(shipped, capital=100000.0)
    config_id = shipped.arm_config_ids(evaluation_context=context, workspace=workspace)["convex_point"]
    off_scenario = _context(shipped, capital=75000.0)
    with pytest.raises(RegistrationError, match="capital scenarios"):
        shipped.check_strict(
            config_id=config_id, evaluation_context=off_scenario, workspace=workspace, check_freeze=False
        )


def test_a_different_accounting_engine_is_refused(shipped: Registration) -> None:
    with pytest.raises(RegistrationError, match="accounting_engine"):
        shipped.check_frozen_inputs(_context(shipped, accounting_engine="historical_constant_weight"))


def test_the_frozen_label_convention_is_the_calibrations_own(shipped: Registration) -> None:
    """`[frozen].executable_return` is the label the mapping is fitted on, not the engine's."""
    assert shipped.frozen.executable_return == "open_to_open"
    for arm in shipped.arms:
        if arm.calibration is None:
            continue
        payload = tomllib.loads(arm.calibration.read_text(encoding="utf-8"))
        assert payload["executable_return"] == shipped.frozen.executable_return
    # The engine's own key of that name carries a different string for the
    # canonical engine, and a registered canonical run is still accepted.
    assert ENGINE_EXECUTABLE_RETURN["canonical"] != shipped.frozen.executable_return
    shipped.check_frozen_inputs(_context(shipped))


def test_a_calibration_fitted_on_another_label_refuses(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    for name in ("shared_slope_point", "shared_slope_p80"):
        _edit(
            tmp_path / f"calibrations/{name}.toml",
            'executable_return = "open_to_open"',
            'executable_return = "close_to_close"',
        )
    with pytest.raises(RegistrationError, match="one convention"):
        load_registration(tmp_path / REGISTRATION, workspace=workspace)


def test_a_moved_fold_table_is_refused(shipped: Registration) -> None:
    with pytest.raises(RegistrationError, match="fold geometry"):
        shipped.check_frozen_inputs(_context(shipped, fold_table_sha256="0" * 64))


def test_a_context_that_records_no_fold_hash_is_refused(shipped: Registration) -> None:
    context = _context(shipped)
    del context["fold_table_sha256"]
    with pytest.raises(RegistrationError, match="no fold_table_sha256"):
        shipped.check_frozen_inputs(context)


def test_a_run_before_the_panel_is_frozen_is_refused(shipped: Registration, tmp_path: Path) -> None:
    """Strict mode requires the frozen score panel the three arms share.

    The record is bound to a path that does not exist rather than to the shipped one,
    because the shipped one is written by the experiment's own opening step and is present
    in the tree from the moment stage 4 has been run: a test whose premise is the absence
    of a generated artifact stops testing anything the day the artifact is generated.
    """
    missing = replace(shipped, frozen=replace(shipped.frozen, freeze_record=tmp_path / "not-frozen-yet.json"))
    with pytest.raises(RegistrationError, match="--freeze-scores"):
        missing.check_freeze_record(_context(missing))


def test_a_run_on_another_dataset_than_the_frozen_panel_is_refused(shipped: Registration, tmp_path: Path) -> None:
    record = tmp_path / "momentum_construction_v1.freeze.json"
    record.write_text(
        json.dumps({"data_id": "frozen-data-id", "visibility_mode": "research_vintage", "panel_sha256": "abc"}),
        encoding="utf-8",
    )
    rebound = replace(shipped, frozen=replace(shipped.frozen, freeze_record=record))
    assert rebound.check_freeze_record(_context(rebound, data_id="frozen-data-id"))["data_id"] == "frozen-data-id"
    with pytest.raises(RegistrationError, match="data_id"):
        rebound.check_freeze_record(_context(rebound, data_id="some-other-dataset"))


def test_the_budget_admits_three_attempts_and_no_fourth(shipped: Registration) -> None:
    shipped.check_budget(2)
    with pytest.raises(RegistrationError, match="trial_budget=3"):
        shipped.check_budget(3)


def _variant_spec(root: Path, *, calibration: str) -> str:
    """A convex specification that is not a declared arm, written as a project would."""
    control = tomllib.loads((root / "examples/momentum/reference.toml").read_text(encoding="utf-8"))
    component = control["strategy"]["components"][0]
    params = "\n".join(f"{key} = {value!r}" for key, value in component["params"].items())
    return f"""
[strategy]
name = "weighting_variant"
universe_index = "{control['strategy']['universe_index']}"
combination = "calibrated_convex"
renormalize = false

[[strategy.components]]
id = "{component['id']}"
path = "{component['path']}"
weight = 1.0
horizon_bars = {component['horizon_bars']}
update_every_bars = {component['update_every_bars']}
valid_for_bars = {component['valid_for_bars']}
to_weights = "none"

[strategy.components.params]
{params}

[strategy.eligibility]
min_price = {control['strategy']['eligibility']['min_price']}
min_median_dollar_volume = {control['strategy']['eligibility']['min_median_dollar_volume']}
dollar_volume_window = {control['strategy']['eligibility']['dollar_volume_window']}
return_history_bars = {control['strategy']['eligibility']['return_history_bars']}
membership_required = true

[construction]
kind = "convex"
calibration = "calibrations/{calibration}"
risk_model = "examples/reference/risk/factor_diagonal.toml"
risk_aversion = 0.0
eps = 1.0e-8
max_at_cap_fraction = 0.35
eligibility_exit = "reduce_only"

[construction.band]
mode = "off"

[policy]
profile = "examples/reference/profiles/experiment_v1.toml"
costs = "examples/reference/costs/retail_v1.toml"
execution = "examples/reference/profiles/execution_alpaca.toml"
monitoring = "examples/reference/policies/monitoring.toml"
recovery = "examples/reference/policies/recovery.toml"
"""


# --------------------------------------------------------------------------
# Per-arm component weights (registrations/2026-09-characteristics.toml)
# --------------------------------------------------------------------------

CHARACTERISTICS = "registrations/2026-09-characteristics.toml"

_CHARACTERISTICS_FILES: tuple[str, ...] = (
    CHARACTERISTICS,
    "examples/characteristics/control.toml",
    PROFILE,
    "calibrations/shared_slope_point.toml",
    "examples/reference/risk/factor_diagonal.toml",
    "folds/demotion_b1.toml",
    "examples/reference/costs/retail_v1.toml",
    "examples/reference/policies/monitoring.toml",
    "examples/reference/policies/recovery.toml",
    "examples/reference/profiles/execution_alpaca.toml",
)


@pytest.fixture(scope="module")
def characteristics() -> Registration:
    return load_registration(REPO_ROOT / CHARACTERISTICS, workspace=Workspace(REPO_ROOT))


def _characteristics_project(tmp_path: Path) -> Workspace:
    for relative in _CHARACTERISTICS_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    return Workspace(tmp_path)


def test_characteristics_registration_declares_six_weighted_arms(characteristics: Registration) -> None:
    assert characteristics.arm_ids == (
        "momentum_control",
        "residual_momentum_only",
        "idiosyncratic_vol_only",
        "momentum_residual_momentum",
        "momentum_idiosyncratic_vol",
        "composite_three",
    )
    assert characteristics.trial_budget == 6
    assert {c.id for c in characteristics.components} == {"momentum", "residual_momentum", "idiosyncratic_vol"}
    for arm in characteristics.arms:
        assert arm.kind == "heuristic"
        assert arm.component_weights is not None
        assert set(arm.component_weights) == {"momentum", "residual_momentum", "idiosyncratic_vol"}
    assert characteristics.arm("momentum_control").component_weights == {
        "momentum": 1.0,
        "residual_momentum": 0.0,
        "idiosyncratic_vol": 0.0,
    }


def test_arm_weights_replace_the_control_s_and_every_component_is_still_run(characteristics: Registration) -> None:
    """The control spec weights momentum 1 and the others 0; an arm's table overrides all three."""
    spec = characteristics.arm_specification("composite_three", workspace=Workspace(REPO_ROOT))
    weights = {loaded.component_id: float(loaded.spec.weight) for loaded in spec.components}
    assert set(weights) == {"momentum", "residual_momentum", "idiosyncratic_vol"}
    assert all(abs(w - 1.0 / 3.0) < 1e-12 for w in weights.values())
    control = characteristics.arm_specification("momentum_control", workspace=Workspace(REPO_ROOT))
    assert {loaded.component_id: float(loaded.spec.weight) for loaded in control.components} == {
        "momentum": 1.0,
        "residual_momentum": 0.0,
        "idiosyncratic_vol": 0.0,
    }
    # Every declared component is loaded and scored even at weight zero; that
    # is what makes the freeze cover all three panels.
    assert len(control.components) == 3


def test_every_weighted_arm_has_its_own_identity(characteristics: Registration) -> None:
    ids = characteristics.arm_config_ids(
        evaluation_context=_context(characteristics), workspace=Workspace(REPO_ROOT)
    )
    assert len(set(ids.values())) == len(characteristics.arms)


def test_a_partial_weight_table_is_refused(tmp_path: Path) -> None:
    ws = _characteristics_project(tmp_path)
    _edit(
        tmp_path / CHARACTERISTICS,
        "[arms.components]\nmomentum = 1.0\nresidual_momentum = 0.0\nidiosyncratic_vol = 0.0\n",
        "[arms.components]\nmomentum = 1.0\n",
    )
    with pytest.raises(RegistrationError, match="every declared component is weighted explicitly"):
        load_registration(tmp_path / CHARACTERISTICS, workspace=ws)


def test_an_arm_weighting_every_component_at_zero_is_refused(tmp_path: Path) -> None:
    ws = _characteristics_project(tmp_path)
    _edit(
        tmp_path / CHARACTERISTICS,
        "[arms.components]\nmomentum = 1.0\nresidual_momentum = 0.0\nidiosyncratic_vol = 0.0\n",
        "[arms.components]\nmomentum = 0.0\nresidual_momentum = 0.0\nidiosyncratic_vol = 0.0\n",
    )
    with pytest.raises(RegistrationError, match="would hold nothing"):
        load_registration(tmp_path / CHARACTERISTICS, workspace=ws)


def test_a_negative_or_non_numeric_weight_is_refused(tmp_path: Path) -> None:
    ws = _characteristics_project(tmp_path)
    _edit(
        tmp_path / CHARACTERISTICS,
        "[arms.components]\nmomentum = 1.0\nresidual_momentum = 0.0\nidiosyncratic_vol = 0.0\n",
        "[arms.components]\nmomentum = -1.0\nresidual_momentum = 0.0\nidiosyncratic_vol = 0.0\n",
    )
    with pytest.raises(RegistrationError, match="non-negative"):
        load_registration(tmp_path / CHARACTERISTICS, workspace=ws)


def test_the_construction_registration_still_carries_no_weight_table(shipped: Registration) -> None:
    assert all(arm.component_weights is None for arm in shipped.arms)


# --------------------------------------------------------------------------
# The learned cross-section seam (registrations/2026-09-learned-xsection.toml)
# --------------------------------------------------------------------------

LEARNED = "registrations/2026-09-learned-xsection.toml"


@pytest.fixture(scope="module")
def learned() -> Registration:
    return load_registration(REPO_ROOT / LEARNED, workspace=Workspace(REPO_ROOT))


def test_learned_registration_runs_x0_to_x4_one_hot(learned: Registration) -> None:
    assert learned.arm_ids == ("X0", "X1", "X2", "X3", "X4")
    assert learned.trial_budget == 5
    for index, arm in enumerate(learned.arms):
        weights = arm.component_weights
        assert weights is not None
        assert [weights[f"x{i}"] for i in range(5)] == [1.0 if i == index else 0.0 for i in range(5)]


def test_each_learned_component_differs_from_x0_in_exactly_the_design_s_one_value(learned: Registration) -> None:
    """Design §3: X1 lambda 10, X2 drop rev_1, X3 18 cross-sections, X4 pinned GBM."""
    control = tomllib.loads((REPO_ROOT / "examples/learned_xsection/control.toml").read_text(encoding="utf-8"))
    params = {row["id"]: dict(row["params"]) for row in control["strategy"]["components"]}
    x0 = params["x0"]
    assert x0 == {"learner": "ridge", "ridge_lambda": 1.0, "n_cross_sections": 36, "spacing_bars": 21, "horizon_bars": 21}
    assert {k: v for k, v in params["x1"].items() if x0.get(k) != v} == {"ridge_lambda": 10.0}
    assert {k: v for k, v in params["x2"].items() if x0.get(k) != v} == {
        "features": ["mom_12_1", "mom_6_1", "vol_63", "maxret_21", "high_52w", "dv_63"]
    }
    assert {k: v for k, v in params["x3"].items() if x0.get(k) != v} == {"n_cross_sections": 18}
    assert params["x4"] == {
        "learner": "gbm",
        "n_cross_sections": 36,
        "spacing_bars": 21,
        "horizon_bars": 21,
        "gbm_trees": 200,
        "gbm_depth": 2,
        "gbm_learning_rate": 0.05,
        "gbm_min_leaf": 50,
        "gbm_seed": 20260718,
    }


def test_learned_fold_table_is_folds_12_to_20_of_the_recorded_geometry(learned: Registration) -> None:
    recorded = tomllib.loads((REPO_ROOT / "folds/demotion_b1.toml").read_text(encoding="utf-8"))["fold"]
    table = tomllib.loads((REPO_ROOT / "folds/learned_xsection_v1.toml").read_text(encoding="utf-8"))["fold"]
    keys = ("formation_start", "formation_end", "test_start", "test_end")
    assert [tuple(row[k] for k in keys) for row in table] == [
        tuple(row[k] for k in keys) for row in recorded if int(row["index"]) >= 12
    ]
    assert [int(row["index"]) for row in table] == list(range(9))
    assert str(learned.sample.first_decision) == "2024-04-02"
    assert learned.sample.warmup_bars == 252 + 36 * 21


def test_breadth_arms_differ_only_in_the_index_they_trade() -> None:
    reg = load_registration(REPO_ROOT / "registrations/2026-09-breadth.toml", workspace=Workspace(REPO_ROOT))
    assert reg.arm_ids == ("sp500_control", "sp400_only", "sp900_union")
    specs = {arm: reg.arm_specification(arm, workspace=Workspace(REPO_ROOT)) for arm in reg.arm_ids}
    assert {arm: spec.strategy.universe_index for arm, spec in specs.items()} == {
        "sp500_control": "sp500",
        "sp400_only": "sp400",
        "sp900_union": "sp900",
    }
    ids = reg.arm_config_ids(evaluation_context=_context(reg), workspace=Workspace(REPO_ROOT))
    assert len(set(ids.values())) == 3


def test_an_empty_universe_index_override_is_refused(tmp_path: Path) -> None:
    ws = _characteristics_project(tmp_path)
    _edit(
        tmp_path / CHARACTERISTICS,
        'id = "momentum_control"\nprofile = "examples/reference/profiles/experiment_v1.toml"\n',
        'id = "momentum_control"\nprofile = "examples/reference/profiles/experiment_v1.toml"\nuniverse_index = ""\n',
    )
    with pytest.raises(RegistrationError, match="universe_index"):
        load_registration(tmp_path / CHARACTERISTICS, workspace=ws)
