"""The component conformance suite (engine specification §3.3).

These tests are written against the failure that would cost the most: a
component that peeks at bars after its decision bar, passes conformance, and
therefore produces a backtest whose Sharpe is an artefact of the leak. The
central test is not "momentum conforms" but "a component that looks five bars
ahead is rejected" — a suite that only ever sees honest components proves
nothing about the one dishonest component it exists to catch.

The geometry that makes that catchable is asserted directly
(``test_the_withheld_future_is_what_catches_the_leak``): a forward-looking
feature is NaN at the end of any panel, so a check window sitting at the panel's
end would exonerate the leak. The suite therefore withholds a block of bars
after the check window and lets only the contaminating arm see them.

The remaining tests pin the refusal that stands — a missing declaration, because
a component that cannot be measured has no verdict to report — and the two
checks that read ``fitted_state``: a stateful component must raise when scored
before ``fit`` (check 1), and two independent fits of it must write the same
artifact (check 6). The stage-1 refusal of a ``train_only`` component is gone
with the stage that could not persist a fit (decision D4).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.cli import main
from prism.conformance import (
    CACHE_KEY_ELEMENTS,
    DEFAULT_FUTURE_BARS,
    TRUSTED_PLUGIN_STATEMENT,
    ComponentReport,
    ConformanceError,
    check_component,
    conform_specification,
)
from prism.io.dataset import Dataset
from prism.io.synthetic import EQUITY_INDEX, write_synthetic_dataset
from prism.signal.base import Signal
from prism.signal.momentum_node import MomentumSignalNode
from prism.spec import load_spec
from prism.workspace import Workspace

START = dt.date(2026, 1, 1)
END = dt.date(2027, 12, 31)
VINTAGE = "synthetic_v1"
DECISION_DATE = END.isoformat()


# ---- components under test -------------------------------------------------


class Trailing(Signal):
    """An honest component: a strictly trailing price ratio, NaN where it cannot score."""

    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    price_scale_invariant = True
    fitted_state = "none"
    estimated_decay_bars = None

    def __init__(self, lookback_bars: int = 30) -> None:
        self.lookback_bars = int(lookback_bars)

    @property
    def horizon_bars(self) -> int:
        return 21

    @property
    def required_history(self) -> int:
        return self.lookback_bars + 1

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "Trailing":
        self._require(close)
        return self

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        self._require(close)
        base = close.iloc[-1 - self.lookback_bars]
        return (close.iloc[-1] / base.where(base > 0) - 1.0).astype(float)

    def _require(self, close: pd.DataFrame) -> None:
        if len(close) < self.required_history:
            raise ValueError(f"panel has {len(close)} rows; {type(self).__name__} needs {self.required_history}")


class Leaky(Trailing):
    """The classic leak: one feature frame computed over the whole panel, then cached and sliced.

    Nothing here is exotic. It is what happens when a derived feature is
    computed once "for speed" over everything the component was handed and
    memoised by name, so bar *t* is served a value that used bar *t + forward*.
    """

    def __init__(self, lookback_bars: int = 30, forward_bars: int = 5) -> None:
        super().__init__(lookback_bars)
        self.forward_bars = int(forward_bars)
        self._features: pd.DataFrame | None = None

    def feature_cache_key(self) -> dict[str, object]:
        return {
            "data_id": "test",
            "implementation_identity": f"{type(self).__name__}",
            "params": (self.lookback_bars, self.forward_bars),
            "fit_cutoff": None,
            "scoring_basis": self.scoring_basis,
        }

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        self._require(close)
        if self._features is None:
            self._features = close.shift(-self.forward_bars) / close - 1.0
        stamp = close.index[-1]
        if stamp in self._features.index:
            row = self._features.loc[stamp]
            if bool(row.notna().any()):
                return row.astype(float)
        return super().score(close, volume)


class Nondeterministic(Trailing):
    """A component whose score moves between two calls on one fitted object."""

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        scores = super().score(close, volume)
        noise = np.random.default_rng().normal(0.0, 1e-9, len(scores))
        return scores + noise


class Undeclared(Trailing):
    """Declares everything but ``estimated_decay_bars``."""

    estimated_decay_bars = property(lambda self: (_ for _ in ()).throw(AttributeError("estimated_decay_bars")))


class TrainOnly(Trailing):
    """Declares fitted state and behaves as though it had none: scores unfitted, persists nothing."""

    fitted_state = "train_only"


class Stateful(Trailing):
    """An honest stateful component: refuses to score unfitted, and persists its fit.

    The fit is a mean of the training panel's last row. It deliberately does not
    enter the score, so the component stays price-scale invariant as it
    declares; what it exercises here is the boundary (check 1) and the artifact
    (check 6).
    """

    fitted_state = "train_only"

    def __init__(self, lookback_bars: int = 30) -> None:
        super().__init__(lookback_bars)
        self.level_: float | None = None

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "Stateful":
        self._require(close)
        self.level_ = float(close.iloc[-1].mean())
        return self

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        if self.level_ is None:
            raise RuntimeError("Stateful declares fitted_state='train_only' and must be fitted before it scores")
        return super().score(close, volume)

    def save_state(self, path: Path) -> None:
        path.write_text(json.dumps({"level": self.level_}, sort_keys=True), encoding="utf-8")

    def load_state(self, path: Path) -> "Stateful":
        self.level_ = json.loads(path.read_text(encoding="utf-8"))["level"]
        return self


class RandomizedFit(Stateful):
    """A stateful component whose fit is not reproducible: the artifact moves between fits."""

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "RandomizedFit":
        super().fit(close, volume)
        self.level_ = float(np.random.default_rng().normal())
        return self


class ThinCacheKey(Trailing):
    """Declares a feature cache whose key cannot separate two datasets or two fits."""

    def feature_cache_key(self) -> dict[str, object]:
        return {"data_id": "test", "params": (), "scoring_basis": "price"}


class ZeroForNoOpinion(Trailing):
    """Fills the unscoreable names with 0.0 — a position instruction dressed up as an opinion."""

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        return super().score(close, volume).fillna(0.0)


class SilentOnShortPanel(Trailing):
    """Returns NaN instead of raising when the panel is shorter than ``required_history``."""

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        if len(close) < self.required_history:
            return pd.Series(float("nan"), index=close.columns, dtype=float)
        return super().score(close, volume)


class ScaleDependent(Trailing):
    """Emits a price level while declaring ``price_scale_invariant = True``."""

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        self._require(close)
        return close.iloc[-1].astype(float)


# ---- fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    root = tmp_path_factory.mktemp("conform") / "data"
    write_synthetic_dataset(root, seed=11, scenarios="all", n_equities=14, n_etfs=3, start=START, end=END)
    return Dataset.open(root)


@pytest.fixture(scope="module")
def panels(dataset: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    view = dataset.view("research_vintage", DECISION_DATE, vintage=VINTAGE)
    sids = sorted(view.membership(EQUITY_INDEX))
    return view.panel(sids, DECISION_DATE, "close"), view.panel(sids, DECISION_DATE, "volume")


def _run(component: Signal, panels: tuple[pd.DataFrame, pd.DataFrame], **kwargs) -> ComponentReport:
    close, volume = panels
    factory = kwargs.pop("factory", lambda: type(component)())
    return check_component(component, close=close, volume=volume, component_id="under_test", factory=factory, **kwargs)


def _check(report: ComponentReport, number: int):
    return next(check for check in report.checks if check.number == number)


# ---- the honest cases ------------------------------------------------------


def test_the_shipped_momentum_component_conforms(panels) -> None:
    def make() -> MomentumSignalNode:
        return MomentumSignalNode(lookback_bars=252, skip_bars=21, horizon_bars=21)

    report = _run(make(), panels, factory=make)

    assert report.ok, [c for c in report.checks if c.failed]
    assert [check.number for check in report.checks] == [1, 2, 3, 4, 5, 6, 7]
    # The rescale check ran, because the node declares the property (§3.3 check 5).
    assert _check(report, 5).status == "PASS"
    assert report.cache_declared is False


def test_an_honest_component_passes_every_check(panels) -> None:
    report = _run(Trailing(), panels)
    assert report.ok, [c for c in report.checks if c.failed]
    assert report.declarations["required_history"] == 31


# ---- the leak --------------------------------------------------------------


def test_a_component_that_peeks_at_future_bars_is_rejected(panels) -> None:
    report = _run(Leaky(), panels)

    assert not report.ok
    causality = _check(report, 2)
    assert causality.status == "FAIL"
    assert "not causal" in causality.detail
    # The cold reference the leak is measured against is the honest value, so
    # the report names the bar and the sid that moved rather than "mismatch".
    assert "sec_" in causality.detail
    # Its declared cache is what serves the future-contaminated row, so the
    # cache-reuse arm fails too.
    assert _check(report, 3).status == "FAIL"


def test_the_withheld_future_is_what_catches_the_leak(panels) -> None:
    """The same leaky component passes when the sweep withholds less than it peeks at.

    This is the geometry the suite depends on: a forward-looking feature is NaN
    at the end of the panel, so a check window at the panel's end sees nothing.
    If a refactor moves the window back to the end, this test goes green in the
    wrong direction and ``test_a_component_that_peeks_at_future_bars_is_rejected``
    fails — which is the pair, and why both are here.
    """
    blind = _run(Leaky(forward_bars=12), panels, future_bars=2)
    assert _check(blind, 2).status == "PASS"

    seeing = _run(Leaky(forward_bars=12), panels)
    assert DEFAULT_FUTURE_BARS > 12
    assert _check(seeing, 2).status == "FAIL"


def test_a_nondeterministic_component_is_rejected(panels) -> None:
    report = _run(Nondeterministic(), panels)

    determinism = _check(report, 6)
    assert determinism.status == "FAIL"
    assert "reproducible" in determinism.detail
    assert not report.ok


def test_a_scale_dependent_component_declaring_invariance_is_rejected(panels) -> None:
    report = _run(ScaleDependent(), panels)

    rescale = _check(report, 5)
    assert rescale.status == "FAIL"
    assert "price_scale_invariant=True" in rescale.detail


def test_zero_where_the_component_has_no_opinion_is_rejected(panels) -> None:
    report = _run(ZeroForNoOpinion(), panels)

    history = _check(report, 4)
    assert history.status == "FAIL"
    assert "NaN" in history.detail


def test_a_silent_answer_on_a_short_panel_is_rejected(panels) -> None:
    report = _run(SilentOnShortPanel(), panels)

    history = _check(report, 4)
    assert history.status == "FAIL"
    assert "instead of raising" in history.detail


def test_a_feature_cache_key_missing_an_element_is_rejected(panels) -> None:
    report = _run(ThinCacheKey(), panels)

    cache = _check(report, 3)
    assert cache.status == "FAIL"
    assert "implementation_identity" in cache.detail
    assert report.cache_declared is True
    assert set(CACHE_KEY_ELEMENTS) == {
        "data_id",
        "implementation_identity",
        "params",
        "fit_cutoff",
        "scoring_basis",
    }


# ---- refusals --------------------------------------------------------------


def test_a_missing_declaration_is_refused_naming_the_field(panels) -> None:
    with pytest.raises(ConformanceError, match="estimated_decay_bars"):
        _run(Undeclared(), panels)


def test_a_stateful_component_is_measured_not_refused(panels) -> None:
    """The stage-1 refusal is lifted: a fit is a real artifact, so it can be checked (D4)."""
    report = _run(Stateful(), panels, factory=Stateful)

    assert report.ok, [c for c in report.checks if c.failed]
    assert [check.number for check in report.checks] == [1, 2, 3, 4, 5, 6, 7]
    assert _check(report, 1).status == "PASS"
    assert "raised RuntimeError" in _check(report, 1).detail
    # Check 6 compared two independently fitted artifacts by content hash.
    assert _check(report, 6).status == "PASS"
    assert "sha256:" in _check(report, 6).detail


def test_a_stateful_component_that_scores_unfitted_fails_the_boundary(panels) -> None:
    report = _run(TrainOnly(), panels, factory=TrainOnly)

    boundary = _check(report, 1)
    assert boundary.status == "FAIL"
    assert "unfitted model" in boundary.detail
    assert not report.ok


def test_a_stateful_component_that_cannot_persist_its_fit_fails_determinism(panels) -> None:
    """A declared fit that cannot be written has no fitted_state_ref, so it cannot be replayed (§3.11)."""
    report = _run(TrainOnly(), panels, factory=TrainOnly)

    determinism = _check(report, 6)
    assert determinism.status == "FAIL"
    assert "fitted_state_ref" in determinism.detail


def test_a_randomized_fit_is_caught_by_the_artifact_hash(panels) -> None:
    report = _run(RandomizedFit(), panels, factory=RandomizedFit)

    determinism = _check(report, 6)
    assert determinism.status == "FAIL"
    assert "two independent fits" in determinism.detail
    assert not report.ok


def test_a_panel_too_short_for_the_component_is_refused(panels) -> None:
    close, volume = panels
    component = Trailing(lookback_bars=30)
    short = component.required_history + 5  # no room for the withheld future

    with pytest.raises(ConformanceError, match="required_history=31"):
        check_component(
            component,
            close=close.iloc[:short],
            volume=volume.iloc[:short],
            component_id="under_test",
            factory=Trailing,
        )


# ---- the report ------------------------------------------------------------


def test_the_report_states_that_a_plugin_is_trusted_code(panels, tmp_path: Path) -> None:
    from prism.conformance import ConformanceReport

    report = ConformanceReport(spec_path=tmp_path / "book.toml", dataset={}, components=(_run(Trailing(), panels),))
    text = report.text()

    assert TRUSTED_PLUGIN_STATEMENT in text
    assert "not a sandbox" in text
    for number in range(1, 8):
        assert f" {number} " in text
    assert "declarations" in text


# ---- through the specification loader --------------------------------------


EXAMPLE_SIGNAL = """
import pandas as pd
from prism.signal.base import Signal


class TrailingReturnRank(Signal):
    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    price_scale_invariant = True
    fitted_state = "none"
    estimated_decay_bars = None
    refit_every_bars = 1
    training_window_bars = 0

    def __init__(self, lookback_bars: int = 60) -> None:
        self.lookback_bars = lookback_bars

    @property
    def horizon_bars(self) -> int:
        return 21

    @property
    def required_history(self) -> int:
        return self.lookback_bars + 1

    def fit(self, close, volume=None):
        return self

    def score(self, close, volume=None):
        if len(close) < self.required_history:
            raise ValueError("panel too short")
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
valid_for_bars = 26
to_weights = "decile_long_short"

[strategy.components.params]
lookback_bars = 60

[strategy.components.to_weights_params]
decile = 0.20
max_gross = 1.0
max_symbol_abs_weight = 0.35

[strategy.eligibility]
min_price = 5.0
min_median_dollar_volume = 1000000.0
listing_seasoning_bars = 20

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
    "policies/recovery.toml": 'mode = "reduce_only"\n',
}

_COUNTER = [0]


@pytest.fixture
def isolate_imports():
    before_path = list(sys.path)
    before_modules = set(sys.modules)
    yield
    sys.path[:] = before_path
    for name in set(sys.modules) - before_modules:
        del sys.modules[name]


@pytest.fixture
def restore_root_logger():
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if getattr(handler, "_te_managed", False):
                root.removeHandler(handler)
                handler.close()
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def project(tmp_path: Path, dataset: Dataset) -> tuple[Workspace, Path]:
    """A private project whose workspace ``data/`` is the module's synthetic fixture."""
    _COUNTER[0] += 1
    package = f"conform_strategies_{_COUNTER[0]}"
    root = tmp_path / f"project_{_COUNTER[0]}"
    workspace = Workspace(root).ensure()
    (root / package).mkdir()
    (root / package / "__init__.py").write_text("", encoding="utf-8")
    (root / package / "my_signal.py").write_text(textwrap.dedent(EXAMPLE_SIGNAL), encoding="utf-8")
    for relative, content in POLICY_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    spec_path = root / "specs" / "book.toml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(BOOK_TOML.format(package=package), encoding="utf-8")
    for source in sorted(dataset.root.rglob("*")):
        target = workspace.data / source.relative_to(dataset.root)
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    return workspace, spec_path


def test_conform_runs_a_specification_against_the_workspace_dataset(tmp_path, dataset, isolate_imports) -> None:
    workspace, spec_path = project(tmp_path, dataset)
    spec = load_spec(spec_path, workspace=workspace)

    report = conform_specification(spec, workspace=workspace)

    assert report.ok, [c for component in report.components for c in component.checks if c.failed]
    assert [component.component_id for component in report.components] == ["trailing"]
    assert report.dataset["vintage"] == VINTAGE
    assert report.dataset["decision date"] == DECISION_DATE
    # The declarations the loader read are the ones the report records.
    assert report.components[0].declarations["update_every_bars"] == 21


def test_the_prism_conform_command_returns_zero_and_prints_the_report(
    tmp_path, dataset, isolate_imports, restore_root_logger, capsys
) -> None:
    workspace, spec_path = project(tmp_path, dataset)

    code = main(["--workspace", str(workspace.root), "conform", str(spec_path)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "prism conform" in printed
    assert TRUSTED_PLUGIN_STATEMENT in printed
    assert "0 failed check(s)" in printed
