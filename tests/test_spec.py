"""The specification loader, configuration identity, and the rank-to-weight rules.

The failures these tests are written against, in order of what they would cost:

* a ``config_id`` that does not move when the run would behave differently — a
  policy file edited, a component's implementation changed, the engine upgraded —
  because that silently makes two different runs comparable (§3.12);
* a ``config_id`` that moves when nothing behaved differently (a comment, a key
  order), because that revokes an authorization for no reason;
* a rank-to-weight rule that does not reproduce the live book's weights, because
  the momentum and trend sleeves are Parity work and a one-ulp move is a new
  book (§3.3);
* a component path refused with somebody else's traceback instead of a message
  naming the path.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.live.daily import DailyBookConfig, _construct_raw_book
from prism.portfolio.construct import construct_decile_neutral, construct_inverse_vol_targets
from prism.signal.base import DeclarationError
from prism.spec import SpecError, apply_to_weights, load_spec, resolve_to_weights_params
from prism.workspace import Workspace

# The section 4 example component, verbatim in its declarations.
EXAMPLE_SIGNAL = """
import pandas as pd
from prism.signal.base import Signal


class TrailingReturnRank(Signal):
    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    fitted_state = "none"
    price_scale_invariant = True
    estimated_decay_bars = None
    horizon_bars = 21
    update_every_bars = 21
    valid_for_bars = 21
    refit_every_bars = 1
    training_window_bars = 0

    def __init__(self, lookback_bars: int = 126) -> None:
        self.lookback_bars = lookback_bars

    @property
    def required_history(self) -> int:
        return self.lookback_bars + 1

    def fit(self, close, volume=None):
        return self

    def score(self, close, volume=None):
        if len(close) < self.required_history:
            return pd.Series(float("nan"), index=close.columns)
        return close.iloc[-1] / close.iloc[-1 - self.lookback_bars] - 1.0
"""

BOOK_TOML = """
[strategy]
name = "trailing_rank"
universe_index = "synthetic_100"
combination = "position_space"

[[strategy.components]]
id = "trailing"
path = "{package}.my_signal:TrailingReturnRank"
weight = 1.0
horizon_bars = 21
update_every_bars = 21
valid_for_bars = 21
to_weights = "decile_long_short"

[strategy.components.params]
lookback_bars = 126

[strategy.components.to_weights_params]
decile = 0.10
max_gross = 1.0
max_symbol_abs_weight = 0.10

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
    "profiles/paper.toml": 'name = "paper"\ncash_buffer = 0.02\n',
    "costs/retail_v1.toml": "commission_bps = 1.0\n",
    "profiles/execution_synthetic.toml": 'venue = "synthetic"\n',
    "policies/monitoring.toml": "review_bars = 21\n",
    "policies/recovery.toml": 'mode = "reduce_only"\n',
}

_COUNTER = [0]


@pytest.fixture(autouse=True)
def _isolate_imports():
    """Keep each project's ``sys.path`` entry and modules out of the next test."""
    before_path = list(sys.path)
    before_modules = set(sys.modules)
    yield
    sys.path[:] = before_path
    for name in set(sys.modules) - before_modules:
        del sys.modules[name]


def _project(
    tmp_path: Path, *, signal_source: str = EXAMPLE_SIGNAL, book: str = BOOK_TOML
) -> tuple[Workspace, Path, str]:
    """Write a private project (strategies package, policies, spec) into a workspace."""
    _COUNTER[0] += 1
    package = f"strategies_{_COUNTER[0]}"
    root = tmp_path / f"project_{_COUNTER[0]}"
    workspace = Workspace(root).ensure()
    (root / package).mkdir()
    (root / package / "__init__.py").write_text("", encoding="utf-8")
    (root / package / "my_signal.py").write_text(textwrap.dedent(signal_source), encoding="utf-8")
    for relative, content in POLICY_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    spec_path = root / "specs" / "book.toml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(book.format(package=package), encoding="utf-8")
    return workspace, spec_path, package


# ---------------------------------------------------------------- loading


def test_loads_the_section_4_example(tmp_path):
    workspace, spec_path, package = _project(tmp_path)
    spec = load_spec(spec_path, workspace=workspace)

    assert spec.strategy.name == "trailing_rank"
    assert spec.strategy.universe_index == "synthetic_100"
    assert spec.strategy.combination == "position_space"
    assert spec.strategy.renormalize is False
    assert spec.construction.kind == "heuristic"
    assert spec.policy.profile == workspace.root / "profiles/paper.toml"

    loaded = spec.component("trailing")
    assert type(loaded.component).__name__ == "TrailingReturnRank"
    assert loaded.component.__class__.__module__ == f"{package}.my_signal"
    # [strategy.components.params] is forwarded to __init__ verbatim.
    assert loaded.component.lookback_bars == 126
    # The loader assigns the configured id and the table's schedule.
    assert loaded.component.component_id == "trailing"
    assert loaded.component.update_every_bars == 21
    assert loaded.component.valid_for_bars == 21


def test_resolved_specification_carries_defaults_not_paths(tmp_path):
    workspace, spec_path, _ = _project(tmp_path)
    resolved = load_spec(spec_path, workspace=workspace).resolved()

    # Eligibility was not declared at all; every default is written out (§3.3).
    assert resolved["strategy"]["eligibility"] == {
        "min_price": 0.0,
        "min_median_dollar_volume": 0.0,
        "dollar_volume_window": 20,
        "return_history_bars": 0,
        "listing_seasoning_bars": 0,
        "membership_required": False,
        "exclude_asset_classes": [],
    }
    # Policy paths never enter the resolved specification: their content does.
    assert "policy" not in resolved
    assert "profiles/paper.toml" not in str(resolved)


def test_instantiate_false_parses_without_importing_and_refuses_config_id(tmp_path):
    workspace, spec_path, package = _project(tmp_path)
    spec = load_spec(spec_path, workspace=workspace, instantiate=False)
    assert spec.components == ()
    assert f"{package}.my_signal" not in sys.modules
    with pytest.raises(SpecError, match="instantiate=False"):
        spec.config_id()


def test_workspace_root_is_placed_on_sys_path(tmp_path):
    workspace, spec_path, _ = _project(tmp_path)
    assert str(workspace.root) not in sys.path
    load_spec(spec_path, workspace=workspace)
    assert str(workspace.root) in sys.path


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("{package}.absent:TrailingReturnRank", "not importable"),
        ("{package}.my_signal:NoSuchClass", "is unknown"),
        ("{package}.my_signal:pd", "not a subclass"),
        ("{package}.my_signal.TrailingReturnRank", "not 'module:Class'"),
    ],
)
def test_unknown_component_path_is_refused_by_name(tmp_path, path, expected):
    workspace, spec_path, package = _project(
        tmp_path, book=BOOK_TOML.replace('path = "{package}.my_signal:TrailingReturnRank"', f'path = "{path}"')
    )
    declared = path.format(package=package)
    with pytest.raises(SpecError) as excinfo:
        load_spec(spec_path, workspace=workspace)
    message = str(excinfo.value)
    assert declared in message, message
    assert expected in message, message
    # The refusal is the loader's, not a bare ImportError escaping the module.
    assert excinfo.value.__cause__ is None
    assert not isinstance(excinfo.value, ImportError)


def test_a_component_module_that_raises_is_refused_with_its_error(tmp_path):
    workspace, spec_path, _ = _project(tmp_path, signal_source="raise RuntimeError('vendor sdk missing')\n")
    with pytest.raises(SpecError) as excinfo:
        load_spec(spec_path, workspace=workspace)
    assert "not importable" in str(excinfo.value)
    assert "RuntimeError: vendor sdk missing" in str(excinfo.value)


def test_bad_component_params_are_refused_naming_the_component(tmp_path):
    book = BOOK_TOML.replace("lookback_bars = 126", "lookback_bars = 126\nunknown_kwarg = 3")
    workspace, spec_path, _ = _project(tmp_path, book=book)
    with pytest.raises(SpecError) as excinfo:
        load_spec(spec_path, workspace=workspace)
    assert "'trailing'" in str(excinfo.value)
    assert "unknown_kwarg" in str(excinfo.value)


def test_a_class_declared_schedule_that_contradicts_the_table_is_refused(tmp_path):
    book = BOOK_TOML.replace(
        "update_every_bars = 21\nvalid_for_bars = 21", "update_every_bars = 5\nvalid_for_bars = 21"
    )
    workspace, spec_path, _ = _project(tmp_path, book=book)
    with pytest.raises(SpecError, match="update_every_bars"):
        load_spec(spec_path, workspace=workspace)


def test_expiry_before_the_next_refresh_is_refused(tmp_path):
    book = BOOK_TOML.replace("valid_for_bars = 21", "valid_for_bars = 3")
    workspace, spec_path, _ = _project(tmp_path, book=book)
    with pytest.raises(SpecError, match="valid_for_bars"):
        load_spec(spec_path, workspace=workspace)


def test_combination_and_construction_kind_must_pair(tmp_path):
    book = BOOK_TOML.replace('combination = "position_space"', 'combination = "calibrated_convex"')
    workspace, spec_path, _ = _project(tmp_path, book=book)
    with pytest.raises(SpecError, match="pairs with"):
        load_spec(spec_path, workspace=workspace)


def test_a_rank_score_component_needs_a_rank_to_weight_rule(tmp_path):
    book = BOOK_TOML.replace('to_weights = "decile_long_short"', 'to_weights = "none"').replace(
        "[strategy.components.to_weights_params]\ndecile = 0.10\nmax_gross = 1.0\nmax_symbol_abs_weight = 0.10\n",
        "",
    )
    workspace, spec_path, _ = _project(tmp_path, book=book)
    with pytest.raises(SpecError, match="rank_score"):
        load_spec(spec_path, workspace=workspace)


def test_exactly_one_universe_form(tmp_path):
    book = BOOK_TOML.replace(
        'universe_index = "synthetic_100"', 'universe_list = ["SPY"]\nuniverse_as_of = "2026-07-17"'
    )
    workspace, spec_path, _ = _project(tmp_path, book=book)
    with pytest.raises(SpecError, match="universe_registration"):
        load_spec(spec_path, workspace=workspace)

    book_both = BOOK_TOML.replace(
        'universe_index = "synthetic_100"', 'universe_index = "synthetic_100"\nuniverse_list = ["SPY"]'
    )
    workspace2, spec_path2, _ = _project(tmp_path, book=book_both)
    with pytest.raises(SpecError, match="exactly one"):
        load_spec(spec_path2, workspace=workspace2)


def test_a_missing_policy_file_is_refused_with_the_resolved_path(tmp_path):
    workspace, spec_path, _ = _project(tmp_path)
    (workspace.root / "policies/recovery.toml").unlink()
    with pytest.raises(SpecError) as excinfo:
        load_spec(spec_path, workspace=workspace)
    assert "policies/recovery.toml" in str(excinfo.value)
    assert str(workspace.root) in str(excinfo.value)


def test_an_undeclaring_component_is_refused_by_the_declaration_contract(tmp_path):
    source = EXAMPLE_SIGNAL.replace('    scoring_basis = "price"\n', "")
    workspace, spec_path, _ = _project(tmp_path, signal_source=source)
    with pytest.raises(DeclarationError, match="scoring_basis"):
        load_spec(spec_path, workspace=workspace)


def test_unknown_keys_are_refused(tmp_path):
    book = BOOK_TOML.replace('name = "trailing_rank"', 'name = "trailing_rank"\nrebalance = "monthly"')
    workspace, spec_path, _ = _project(tmp_path, book=book)
    with pytest.raises(SpecError, match="rebalance"):
        load_spec(spec_path, workspace=workspace)


# ---------------------------------------------------------------- identity


def test_config_id_is_stable_across_loads(tmp_path):
    workspace, spec_path, _ = _project(tmp_path)
    first = load_spec(spec_path, workspace=workspace).config_id()
    second = load_spec(spec_path, workspace=workspace).config_id()
    assert first == second
    assert len(first) == 64


def test_config_id_ignores_comments_and_key_order_in_the_toml(tmp_path):
    workspace, spec_path, _ = _project(tmp_path)
    baseline = load_spec(spec_path, workspace=workspace).config_id()

    text = spec_path.read_text(encoding="utf-8")
    text = "# a note for the reader\n" + text.replace(
        "weight = 1.0\nhorizon_bars = 21", "horizon_bars = 21\nweight = 1.0"
    )
    spec_path.write_text(text, encoding="utf-8")
    assert load_spec(spec_path, workspace=workspace).config_id() == baseline


@pytest.mark.parametrize("relative", sorted(POLICY_FILES))
def test_config_id_changes_when_a_referenced_policy_file_changes(tmp_path, relative):
    workspace, spec_path, _ = _project(tmp_path)
    baseline = load_spec(spec_path, workspace=workspace).config_id()
    target = workspace.root / relative
    target.write_text(target.read_text(encoding="utf-8") + "extra_knob = 7\n", encoding="utf-8")
    assert load_spec(spec_path, workspace=workspace).config_id() != baseline


def test_config_id_follows_policy_content_not_its_path(tmp_path):
    workspace, spec_path, _ = _project(tmp_path)
    baseline = load_spec(spec_path, workspace=workspace).config_id()

    moved = workspace.root / "policies" / "paper.toml"
    moved.write_text((workspace.root / "profiles/paper.toml").read_text(encoding="utf-8"), encoding="utf-8")
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            'profile = "profiles/paper.toml"', 'profile = "policies/paper.toml"'
        ),
        encoding="utf-8",
    )
    assert load_spec(spec_path, workspace=workspace).config_id() == baseline


def test_config_id_changes_when_the_component_implementation_changes(tmp_path):
    workspace, spec_path, package = _project(tmp_path)
    baseline = load_spec(spec_path, workspace=workspace).config_id()

    module = workspace.root / package / "my_signal.py"
    module.write_text(
        module.read_text(encoding="utf-8").replace(
            "return close.iloc[-1] / close.iloc[-1 - self.lookback_bars] - 1.0",
            "return close.iloc[-1] / close.iloc[-1 - self.lookback_bars] - 1.5",
        ),
        encoding="utf-8",
    )
    del sys.modules[f"{package}.my_signal"]
    assert load_spec(spec_path, workspace=workspace).config_id() != baseline


def test_config_id_changes_when_the_engine_version_changes(tmp_path):
    workspace, spec_path, _ = _project(tmp_path)
    spec = load_spec(spec_path, workspace=workspace)
    assert spec.config_id(engine_version="0.4.0") != spec.config_id(engine_version="0.4.1")


def test_config_id_changes_with_a_resolved_parameter(tmp_path):
    workspace, spec_path, _ = _project(tmp_path)
    baseline = load_spec(spec_path, workspace=workspace).config_id()
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace("max_symbol_abs_weight = 0.10", "max_symbol_abs_weight = 0.35"),
        encoding="utf-8",
    )
    assert load_spec(spec_path, workspace=workspace).config_id() != baseline


def test_each_capital_scenario_is_a_distinct_configuration(tmp_path):
    """§3.12 [Q22]: capital lives in evaluation_context, so six scenarios are six ids."""
    workspace, spec_path, _ = _project(tmp_path)
    spec = load_spec(spec_path, workspace=workspace)
    ids = {spec.config_id(evaluation_context={"capital": capital}) for capital in (25_000.0, 100_000.0, 1_000_000.0)}
    assert len(ids) == 3
    assert spec.config_id() not in ids  # an operating session carries no evaluation context


def test_component_identity_records_the_implementation(tmp_path):
    workspace, spec_path, package = _project(tmp_path)
    identity = dict(load_spec(spec_path, workspace=workspace).component("trailing").identity)
    assert identity["component_id"] == "trailing"
    assert identity["import_path"] == f"{package}.my_signal:TrailingReturnRank"
    assert identity["output_kind"] == "rank_score"
    assert identity["units"] == "rank"
    assert identity["scoring_basis"] == "price"
    assert (identity["horizon_bars"], identity["update_every_bars"], identity["valid_for_bars"]) == (21, 21, 21)
    # A project package is not an installed distribution; that is recorded, not nulled.
    assert identity["distribution"] == "unavailable"
    assert identity["version"] == "unavailable"
    assert len(identity["source_sha256"]) == 64


def test_a_shipped_component_identifies_its_distribution_and_source(tmp_path):
    book = BOOK_TOML.replace(
        'path = "{package}.my_signal:TrailingReturnRank"',
        'path = "prism.signal.trend_node:TrendSignalNode"',
    ).replace(
        "[strategy.components.params]\nlookback_bars = 126\n",
        "[strategy.components.params]\nlookback_bars = 252\nskip_bars = 21\nhorizon_bars = 21\n",
    )
    workspace, spec_path, _ = _project(tmp_path, book=book)
    identity = dict(load_spec(spec_path, workspace=workspace).component("trailing").identity)
    assert identity["scoring_basis"] == "total_return"
    source = Path(sys.modules["prism.signal.trend_node"].__file__)
    assert identity["source_sha256"] == __import__("hashlib").sha256(source.read_bytes()).hexdigest()


# ---------------------------------------------------------------- rank-to-weight rules


def _scores(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_decile_long_short_on_a_known_cross_section():
    scores = _scores({f"s{i:02d}": float(i) for i in range(10)} | {"nan": np.nan})
    weights = apply_to_weights(
        "decile_long_short", scores, {"decile": 0.20, "max_gross": 1.0, "max_symbol_abs_weight": 1.0}
    )
    # 10 finite scores, decile 0.20 -> two names per side at +/- 0.25.
    assert weights["s09"] == pytest.approx(0.25)
    assert weights["s08"] == pytest.approx(0.25)
    assert weights["s00"] == pytest.approx(-0.25)
    assert weights["s01"] == pytest.approx(-0.25)
    assert weights.drop(["s00", "s01", "s08", "s09"]).eq(0.0).all()
    # An ineligible NaN score is an explicit flat, never a held stale position.
    assert weights["nan"] == 0.0
    assert weights.sum() == pytest.approx(0.0)


def test_decile_long_short_emits_a_zero_row_when_the_cross_section_is_thin():
    scores = _scores({"a": 1.0, "b": 2.0, "c": 3.0})
    weights = apply_to_weights("decile_long_short", scores, {"decile": 0.10})
    assert weights.eq(0.0).all()


def test_top_k_equal_weight_on_a_known_cross_section():
    scores = _scores({"a": 5.0, "b": 4.0, "c": 3.0, "d": np.nan, "e": 1.0})
    weights = apply_to_weights(
        "top_k_equal_weight", scores, {"k": 2, "max_gross": 0.98, "max_symbol_abs_weight": 0.25}
    )
    assert weights["a"] == pytest.approx(0.25)  # 0.49 each, capped down to 0.25
    assert weights["b"] == pytest.approx(0.25)
    assert weights[["c", "d", "e"]].eq(0.0).all()
    assert (weights >= 0.0).all()  # no short leg is ever produced


def test_top_k_equal_weight_spreads_over_what_it_has(caplog):
    scores = _scores({"a": 5.0, "b": np.nan, "c": np.nan})
    with caplog.at_level("WARNING"):
        weights = apply_to_weights("top_k_equal_weight", scores, {"k": 3, "max_gross": 0.9})
    assert weights["a"] == pytest.approx(0.9)
    assert "top_k_equal_weight" in caplog.text


def test_top_k_equal_weight_breaks_ties_by_ascending_sid():
    scores = _scores({"zz": 1.0, "aa": 1.0, "mm": 1.0})
    weights = apply_to_weights("top_k_equal_weight", scores, {"k": 1, "max_gross": 1.0})
    assert weights["aa"] == pytest.approx(1.0)
    assert weights[["mm", "zz"]].eq(0.0).all()


def test_sign_inverse_vol_equalizes_risk_on_a_known_panel():
    # B's daily returns are exactly twice A's, so sigma_B = 2 sigma_A and B
    # carries half of A's weight whatever the EWMA span does.
    index = pd.date_range("2024-01-02", periods=40, freq="B", tz="America/New_York")
    base = 0.01 * np.sin(np.arange(40) / 3.0)
    close = pd.DataFrame(
        {
            "A": 100.0 * np.cumprod(1.0 + base),
            "B": 100.0 * np.cumprod(1.0 + 2.0 * base),
            "flat": 100.0 * np.cumprod(1.0 + base),
            "nan": 100.0 * np.cumprod(1.0 + base),
        },
        index=index,
    )
    scores = _scores({"A": 0.4, "B": -0.9, "flat": 0.0, "nan": np.nan})
    weights = apply_to_weights(
        "sign_inverse_vol",
        scores,
        {"vol_ewma_bars": 5, "annualization_bars": 252, "max_gross": 1.0, "max_symbol_abs_weight": 1.0},
        close=close,
    )
    assert weights["A"] == pytest.approx(2.0 / 3.0, rel=1e-9)
    assert weights["B"] == pytest.approx(-1.0 / 3.0, rel=1e-9)
    # A zero sign and a non-finite score are explicit flats, not held positions.
    assert weights["flat"] == 0.0
    assert weights["nan"] == 0.0
    assert weights.abs().sum() == pytest.approx(1.0)


def test_sign_inverse_vol_needs_the_close_panel():
    with pytest.raises(SpecError, match="close panel"):
        apply_to_weights("sign_inverse_vol", _scores({"A": 1.0}), {"vol_ewma_bars": 5})


def test_none_applies_no_rule():
    scores = _scores({"a": 3.0, "b": np.nan, "c": -7.0})
    weights = apply_to_weights("none", scores)
    pd.testing.assert_series_equal(weights, scores)


def test_no_decision_names_carry_no_decision():
    scores = _scores({f"s{i:02d}": float(i) for i in range(10)})
    weights = apply_to_weights("decile_long_short", scores, {"decile": 0.20}, no_decision=["s00", "s05", "absent"])
    assert np.isnan(weights["s00"])  # would have been -0.25
    assert np.isnan(weights["s05"])  # would have been 0.0
    assert weights["s09"] == pytest.approx(0.25)


def test_to_weights_params_are_validated_and_defaulted():
    assert resolve_to_weights_params("decile_long_short", {"decile": 0.1}) == {
        "decile": 0.1,
        "max_gross": 1.0,
        "max_symbol_abs_weight": 1.0,
    }
    with pytest.raises(SpecError, match="requires 'decile'"):
        resolve_to_weights_params("decile_long_short", {})
    with pytest.raises(SpecError, match="does not take"):
        resolve_to_weights_params("top_k_equal_weight", {"k": 3, "decile": 0.1})
    with pytest.raises(SpecError, match="unknown to_weights rule"):
        resolve_to_weights_params("softmax", {})
    with pytest.raises(SpecError, match="decile must be in"):
        resolve_to_weights_params("decile_long_short", {"decile": 0.9})


# ---------------------------------------------------------------- parity with the live books


def _momentum_shaped_scores(n_dates: int = 24, n_symbols: int = 480) -> pd.DataFrame:
    """A momentum-book-shaped score panel: wide, ranked, with ineligible holes."""
    rng = np.random.default_rng(20260907)
    symbols = [f"SYM{i:04d}" for i in range(n_symbols)]
    index = pd.date_range("2024-01-02", periods=n_dates, freq="B", tz="America/New_York")
    values = rng.normal(size=(n_dates, n_symbols))
    values[rng.random(values.shape) < 0.12] = np.nan  # eligibility screen holes
    values[3, :] = np.nan  # a wholly unscored bar
    values[4, :4] = values[4, 0]  # ties, to pin the stable-sort rule
    return pd.DataFrame(values, index=index, columns=symbols)


def test_decile_long_short_reproduces_the_live_momentum_book():
    """Parity: the rule is ``construct_decile_neutral`` at the momentum book's settings.

    Bitwise against the one-row call the live path makes, and to 1e-12 against
    the same construct run over the whole panel at once. The batch form is not
    bitwise identical to itself row by row: ``cap_book`` decides whether to
    scale a row from ``out.abs().sum(axis=1)``, and that reduction sums a wide
    frame's row in a different order than a one-row frame's, so a raw gross of
    exactly 1.0 reads as 1.0000000000000003 in the batch and the row is scaled
    down by two ulps. The decision-time path is the one-row form, which is what
    the engine will call and what this rule reproduces exactly.
    """
    panel = _momentum_shaped_scores()
    params = {"decile": 0.10, "max_gross": 1.0, "max_symbol_abs_weight": 0.10}
    batch = construct_decile_neutral(panel, decile=0.10, max_gross=1.0, max_symbol_abs_weight=0.10)
    for bar in panel.index:
        got = apply_to_weights("decile_long_short", panel.loc[bar], params)
        one_row = construct_decile_neutral(
            panel.loc[[bar]], decile=0.10, max_gross=1.0, max_symbol_abs_weight=0.10
        ).iloc[0]
        pd.testing.assert_series_equal(got, one_row, check_exact=True, check_names=False)
        pd.testing.assert_series_equal(got, batch.loc[bar], check_exact=False, rtol=1e-12, check_names=False)


def test_decile_long_short_reproduces_the_live_daily_construction_step():
    """Parity against the production path itself (``prism.live.daily``), not just the helper."""
    panel = _momentum_shaped_scores(n_dates=6, n_symbols=200)
    close = pd.DataFrame(100.0, index=panel.index, columns=panel.columns)
    config = DailyBookConfig(book="decile_neutral", decile=0.10, max_gross=1.0, max_symbol_abs_weight=0.10)
    for bar in panel.index:
        live = _construct_raw_book(panel.loc[[bar]], close.loc[:bar], config, config.max_gross)
        got = apply_to_weights(
            "decile_long_short", panel.loc[bar], {"decile": 0.10, "max_gross": 1.0, "max_symbol_abs_weight": 0.10}
        )
        pd.testing.assert_series_equal(got, live, check_exact=True, check_names=False)


def test_sign_inverse_vol_reproduces_the_live_trend_book():
    rng = np.random.default_rng(7)
    symbols = ["SPY", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "PDBC", "UUP"]
    index = pd.date_range("2023-01-02", periods=200, freq="B", tz="America/New_York")
    returns = rng.normal(scale=0.01, size=(len(index), len(symbols)))
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + returns, axis=0), index=index, columns=symbols)
    scores = pd.Series(rng.normal(size=len(symbols)), index=symbols)
    scores.iloc[2] = np.nan
    scores.iloc[5] = 0.0
    params = {"vol_ewma_bars": 63, "annualization_bars": 252, "max_gross": 1.0, "max_symbol_abs_weight": 0.35}

    config = DailyBookConfig(book="inverse_vol", vol_ewma_bars=63, max_gross=1.0, max_symbol_abs_weight=0.35)
    live = _construct_raw_book(scores.to_frame().T.set_index(index[-1:]), close, config, config.max_gross)
    got = apply_to_weights("sign_inverse_vol", scores, params, close=close)
    pd.testing.assert_series_equal(got, live, check_exact=True, check_names=False)

    aligned = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    aligned.iloc[-1] = scores.to_numpy(dtype=float)
    reference = construct_inverse_vol_targets(
        aligned, close, vol_ewma_bars=63, max_gross=1.0, max_symbol_abs_weight=0.35, annualization_bars=252
    ).iloc[-1]
    pd.testing.assert_series_equal(got, reference, check_exact=True, check_names=False)
