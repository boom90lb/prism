"""Component declarations (engine specification §3.3).

The engine reads behaviour off a component's declarations: which close panel to
supply (``scoring_basis``), whether the conformance rescale check runs
(``price_scale_invariant``), whether a fit artifact exists (``fitted_state``),
how an emission is consumed (``output_kind`` / ``units``), and when it stops
driving a target (``valid_for_bars``). A missing or wrong declaration is
therefore a defect the loader and the conformance suite must refuse, not a
field to default. These tests pin the refusal and the shipped components'
declared values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from prism.signal import (
    DECLARED_FIELDS,
    DeclarationError,
    MomentumSignalNode,
    ResidualSignalNode,
    Signal,
    TrendSignalNode,
    check_declarations,
)
from prism_ensemble import EnsembleNodeConfig, EnsembleSignalNode

_OMIT = object()

#: The declarations a component states for itself; the rest have base defaults.
_STATED = {
    "output_kind": "rank_score",
    "units": "rank",
    "scoring_basis": "price",
    "price_scale_invariant": True,
    "fitted_state": "none",
    "estimated_decay_bars": None,
}


class _StubBase(Signal):
    """A concrete component with no declarations, so variants can add them."""

    @property
    def horizon_bars(self) -> int:
        return 5

    @property
    def required_history(self) -> int:
        return 10

    def fit(self, close, volume=None) -> "_StubBase":
        return self

    def score(self, close, volume=None) -> pd.Series:
        return pd.Series(0.0, index=close.columns, dtype=float)


def _stub(**overrides: object) -> Signal:
    """A component declaring ``_STATED`` with ``overrides`` applied (``_OMIT`` drops one)."""
    namespace = {**_STATED, **overrides}
    namespace = {k: v for k, v in namespace.items() if v is not _OMIT}
    return type("StubComponent", (_StubBase,), namespace)()


def _shipped() -> dict[str, Signal]:
    return {
        "momentum": MomentumSignalNode(),
        "trend": TrendSignalNode(),
        "residual": ResidualSignalNode(),
        "ensemble": EnsembleSignalNode(EnsembleNodeConfig()),
    }


# ---------------------------------------------------------------------------
# Every shipped component declares every field, at the declared type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_shipped()))
def test_shipped_component_declares_every_field_at_the_declared_type(name: str) -> None:
    component = _shipped()[name]
    declared = check_declarations(component)
    assert set(declared) == set(DECLARED_FIELDS)
    assert isinstance(declared["component_id"], str)
    assert declared["output_kind"] in ("rank_score", "standardized_forecast", "target_weights")
    assert declared["units"] in ("rank", "sigma_h", "weight")
    assert declared["scoring_basis"] in ("price", "total_return")
    assert isinstance(declared["price_scale_invariant"], bool)
    assert declared["fitted_state"] in ("none", "train_only")
    assert declared["estimated_decay_bars"] is None or isinstance(declared["estimated_decay_bars"], int)
    for field in ("horizon_bars", "update_every_bars", "valid_for_bars", "refit_every_bars"):
        assert isinstance(declared[field], int) and declared[field] >= 1
    assert isinstance(declared["training_window_bars"], int) and declared["training_window_bars"] >= 0
    assert isinstance(declared["required_history"], int) and declared["required_history"] >= 1


def test_momentum_and_trend_declare_the_specified_kinds() -> None:
    """§3.3 fixes these four values per node; the books' Parity depends on them."""
    momentum = check_declarations(MomentumSignalNode())
    assert (momentum["output_kind"], momentum["units"]) == ("rank_score", "rank")
    assert momentum["scoring_basis"] == "price"
    assert momentum["price_scale_invariant"] is True

    trend = check_declarations(TrendSignalNode())
    assert (trend["output_kind"], trend["units"]) == ("rank_score", "rank")
    assert trend["scoring_basis"] == "total_return"  # scores reinvested, costs on price
    assert trend["price_scale_invariant"] is True


def test_forecast_nodes_declare_sigma_h_and_only_the_ensemble_is_stateful() -> None:
    residual = check_declarations(ResidualSignalNode())
    ensemble = check_declarations(EnsembleSignalNode(EnsembleNodeConfig()))
    assert (residual["output_kind"], residual["units"]) == ("standardized_forecast", "sigma_h")
    assert (ensemble["output_kind"], ensemble["units"]) == ("standardized_forecast", "sigma_h")
    assert residual["fitted_state"] == "none"  # every estimate is a trailing-window refit
    assert ensemble["fitted_state"] == "train_only"  # members cross the train/score boundary


def test_schedule_defaults_are_the_live_books_behaviour() -> None:
    """Score every decision, hold one bar, refit every decision, train on everything (§3.8)."""
    node = MomentumSignalNode()
    assert (node.update_every_bars, node.valid_for_bars) == (1, 1)
    assert (node.refit_every_bars, node.training_window_bars) == (1, 0)
    node.update_every_bars = 21  # the loader assigns from the component table
    node.valid_for_bars = 26
    assert check_declarations(node)["valid_for_bars"] == 26


# ---------------------------------------------------------------------------
# A missing or wrong declaration is refused, naming the field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", sorted(_STATED))
def test_missing_declaration_is_refused_naming_the_field(field: str) -> None:
    component = _stub(**{field: _OMIT})
    with pytest.raises(DeclarationError, match=field):
        check_declarations(component)


@pytest.mark.parametrize("field", ["horizon_bars", "required_history"])
def test_a_component_without_the_abstract_declarations_cannot_be_built(field: str) -> None:
    namespace = dict(_STATED)
    namespace["fit"] = lambda self, close, volume=None: self
    namespace["score"] = lambda self, close, volume=None: pd.Series(dtype=float)
    other = "required_history" if field == "horizon_bars" else "horizon_bars"
    namespace[other] = property(lambda self: 1)
    broken = type("Broken", (Signal,), namespace)
    with pytest.raises(TypeError, match=field):
        broken()


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"output_kind": "alpha"}, "output_kind"),
        ({"units": "bps"}, "units"),
        ({"scoring_basis": "adjusted"}, "scoring_basis"),
        ({"fitted_state": "online"}, "fitted_state"),
        ({"price_scale_invariant": 1}, "price_scale_invariant"),
        ({"estimated_decay_bars": 2.5}, "estimated_decay_bars"),
        ({"estimated_decay_bars": 0}, "estimated_decay_bars"),
        ({"component_id": 7}, "component_id"),
        ({"refit_every_bars": 0}, "refit_every_bars"),
        ({"training_window_bars": -1}, "training_window_bars"),
        ({"update_every_bars": True}, "update_every_bars"),
    ],
)
def test_wrong_declaration_is_refused_naming_the_field(overrides: dict, field: str) -> None:
    with pytest.raises(DeclarationError, match=field):
        check_declarations(_stub(**overrides))


def test_units_must_match_the_output_kind() -> None:
    """A rank score in weight units names no quantity the engine could act on."""
    with pytest.raises(DeclarationError, match="units"):
        check_declarations(_stub(output_kind="rank_score", units="weight"))
    check_declarations(_stub(output_kind="target_weights", units="weight"))


def test_output_expiring_before_its_own_refresh_is_refused() -> None:
    with pytest.raises(DeclarationError, match="valid_for_bars"):
        check_declarations(_stub(update_every_bars=21, valid_for_bars=5))
    check_declarations(_stub(update_every_bars=21, valid_for_bars=26))


def test_class_component_id_may_not_contradict_the_configured_id() -> None:
    assert check_declarations(_stub(), configured_id="mom_12_1")["component_id"] == ""
    assert check_declarations(_stub(component_id="mom_12_1"), configured_id="mom_12_1")
    with pytest.raises(DeclarationError, match="component_id"):
        check_declarations(_stub(component_id="something_else"), configured_id="mom_12_1")


# ---------------------------------------------------------------------------
# save_state / load_state and instructions
# ---------------------------------------------------------------------------


def test_a_stateless_component_refuses_to_persist_state(tmp_path: Path) -> None:
    node = MomentumSignalNode()
    with pytest.raises(NotImplementedError, match="fitted_state='none'"):
        node.save_state(tmp_path / "fit.bin")
    with pytest.raises(NotImplementedError, match="fitted_state='none'"):
        node.load_state(tmp_path / "fit.bin")


def test_the_stateful_component_has_no_artifact_and_says_so(tmp_path: Path) -> None:
    """The fit store exists (:mod:`prism.fits`); this node's member serialization does not.

    The refusal names that gap rather than a stage boundary: it must stay a loud
    NotImplementedError, because a stub artifact would be a fit reference that
    resolves to a file carrying no fit.
    """
    node = EnsembleSignalNode(EnsembleNodeConfig())
    with pytest.raises(NotImplementedError, match="does not implement save_state"):
        node.save_state(tmp_path / "fit.bin")
    with pytest.raises(NotImplementedError, match="does not implement load_state"):
        node.load_state(tmp_path / "fit.bin")


def test_a_component_declaring_no_instruction_returns_none() -> None:
    close = pd.DataFrame({"AAA": [1.0, 2.0], "BBB": [3.0, 4.0]})
    assert MomentumSignalNode().instructions(close) is None
    assert _stub().instructions(close) is None
