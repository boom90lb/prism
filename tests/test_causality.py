"""The function-level look-ahead checker, ``prism.causality``.

Four groups of tests:

* **Parity with the strict suite.** ``prism.conformance.check_component`` runs the same
  prefix comparisons on a ``Signal``. On the components that suite's own tests use (an
  honest trailing ratio, the shipped momentum node, a cached five-bar forward leak, the
  same leak against a withheld future too short to see it, a nondeterministic scorer, a
  stateful scorer), the two must reach the same verdict on causality, cache reuse and
  determinism, name the same first difference, and cut the panel at the same bar. Each
  case also pins the expected verdict, so the two cannot agree by both being wrong.
* **Behaviour on plain functions**: the leaks vectorized pandas code actually contains
  (a forward return, a full-sample z-score, a centred window, a backward fill, a cached
  feature) are caught, including when they are returned as 1-D numpy arrays with one value
  per bar; honest functions pass; and the documented blind spots (look-ahead longer than
  the withheld future, a code path the data never exercises, state without a factory) are
  pinned as blind spots so that a change to them is visible.
* **Output shapes**: a per-bar result is never read as a cross section without a refusal,
  ``.tail(1)`` is a cross section, and a panel output that holds only its last bars is
  refused rather than reported as look-ahead.
* **Hygiene**: the module imports numpy and pandas and nothing else from this package,
  and the example runs.
"""

from __future__ import annotations

import ast
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism import causality, conformance
from prism.causality import (
    CHECKS,
    CausalityCheckError,
    CausalityReport,
    LookaheadDetected,
    assert_causal,
    check_causality,
)
from prism.io.dataset import Dataset
from prism.io.synthetic import EQUITY_INDEX, write_synthetic_dataset
from prism.signal.momentum_node import MomentumSignalNode
from tests.test_conformance import Leaky, Nondeterministic, Stateful, Trailing

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "causality" / "lookahead_check.py"

START = dt.date(2026, 1, 1)
END = dt.date(2027, 12, 31)
VINTAGE = "synthetic_v1"
DECISION_DATE = END.isoformat()


# ---------------------------------------------------------------------------
# Parity with prism.conformance.check_component
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bars(tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same synthetic panel ``tests/test_conformance.py`` checks components on."""
    root = tmp_path_factory.mktemp("causality") / "data"
    write_synthetic_dataset(root, seed=11, scenarios="all", n_equities=14, n_etfs=3, start=START, end=END)
    view = Dataset.open(root).view("research_vintage", DECISION_DATE, vintage=VINTAGE)
    sids = sorted(view.membership(EQUITY_INDEX))
    return view.panel(sids, DECISION_DATE, "close"), view.panel(sids, DECISION_DATE, "volume")


def _momentum() -> MomentumSignalNode:
    return MomentumSignalNode(lookback_bars=252, skip_bars=21, horizon_bars=21)


def _forward_twelve() -> Leaky:
    return Leaky(forward_bars=12)


#: (case, factory, keyword arguments, expected causality, expected cache reuse, expected determinism)
PARITY_CASES = [
    ("honest trailing ratio", Trailing, {}, "PASS", "PASS", "PASS"),
    ("shipped momentum node", _momentum, {}, "PASS", "PASS", "PASS"),
    ("cached five-bar forward leak", Leaky, {}, "FAIL", "FAIL", "PASS"),
    ("twelve-bar leak, 21 bars withheld", _forward_twelve, {}, "FAIL", "FAIL", "PASS"),
    ("twelve-bar leak, 2 bars withheld", _forward_twelve, {"future_bars": 2}, "PASS", "PASS", "PASS"),
    ("nondeterministic scorer", Nondeterministic, {}, "FAIL", "FAIL", "FAIL"),
    ("stateful scorer", Stateful, {}, "PASS", "PASS", "PASS"),
]


@pytest.mark.parametrize(
    ("make", "kwargs", "causal", "cache", "deterministic"),
    [case[1:] for case in PARITY_CASES],
    ids=[case[0] for case in PARITY_CASES],
)
def test_the_function_checker_agrees_with_the_conformance_suite(bars, make, kwargs, causal, cache, deterministic):
    close, volume = bars
    strict = conformance.check_component(
        make(), close=close, volume=volume, component_id="under_test", factory=make, **kwargs
    )
    strict_checks = {check.number: check for check in strict.checks}

    window = kwargs.get("check_window_bars", conformance.DEFAULT_CHECK_WINDOW_BARS)
    future = kwargs.get("future_bars", conformance.DEFAULT_FUTURE_BARS)
    training_rows = len(close) - window - future
    train_close, train_volume = close.iloc[:training_rows], volume.iloc[:training_rows]
    component = make()
    component.fit(train_close, train_volume)
    report = check_causality(
        component.score,
        close,
        volume,
        factory=lambda: make().fit(train_close, train_volume).score,
        **kwargs,
    )

    # The pinned verdicts, so the two checkers cannot agree by both being wrong.
    assert report.check("causality").status == causal
    assert (report.check("causality").difference is not None) == (causal == "FAIL")
    assert report.check("cache_reuse").status == cache
    assert report.check("determinism").status == deterministic
    # The parity itself: checks 2, 3 and 6 of the strict suite.
    assert strict_checks[2].status == report.check("causality").status
    assert strict_checks[3].status == report.check("cache_reuse").status
    assert strict_checks[6].status == report.check("determinism").status
    if deterministic == "PASS":
        # Both name the same bar, the same sid and the same two values. (A nondeterministic
        # scorer gives different values on every run, so only its verdicts can agree.)
        for name, number in (("causality", 2), ("cache_reuse", 3)):
            difference = report.check(name).difference
            if difference is not None:
                detail = strict_checks[number].detail
                assert f"bar {difference.bar}: sid {difference.label!r} scored" in detail
                assert repr(difference.reference) in detail and repr(difference.observed) in detail
    # Both cut the panel at the same bar: the strict suite fits on the rows before the window.
    assert report.training_rows == training_rows
    assert f"checked at fit cutoff {close.index[training_rows - 1]}" in strict_checks[7].detail
    assert report.check_bars == tuple(close.index[training_rows : training_rows + window])
    assert report.output == "cross_section"
    assert report.isolated


def test_the_defaults_match_the_conformance_suite() -> None:
    assert causality.DEFAULT_CHECK_WINDOW_BARS == conformance.DEFAULT_CHECK_WINDOW_BARS
    assert causality.DEFAULT_FUTURE_BARS == conformance.DEFAULT_FUTURE_BARS


# ---------------------------------------------------------------------------
# Plain functions
# ---------------------------------------------------------------------------

N_BARS = 400
N_ASSETS = 8


@pytest.fixture(scope="module")
def close() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2020-01-01", periods=N_BARS)
    paths = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (N_BARS, N_ASSETS)), axis=0))
    return pd.DataFrame(paths, index=index, columns=[f"asset_{i}" for i in range(N_ASSETS)])


@pytest.fixture(scope="module")
def volume(close: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        rng.integers(1_000, 100_000, close.shape).astype(float), index=close.index, columns=close.columns
    )


def _statuses(report: CausalityReport) -> dict[str, str]:
    return {check.name: check.status for check in report.checks}


def test_an_honest_cross_sectional_function_passes(close) -> None:
    report = check_causality(lambda c: c.iloc[-1] / c.iloc[-253] - 1.0, close)

    assert report.ok, report.text()
    assert report.output == "cross_section"
    assert [check.name for check in report.checks] == list(CHECKS)
    assert report.training_rows == N_BARS - causality.DEFAULT_CHECK_WINDOW_BARS - causality.DEFAULT_FUTURE_BARS


def test_an_honest_vectorized_function_passes(close) -> None:
    report = check_causality(lambda c: c.shift(21) / c.shift(252) - 1.0, close)

    assert report.ok, report.text()
    assert report.output == "panel"


def test_a_forward_return_feature_is_caught(close) -> None:
    report = check_causality(lambda c: c.shift(-5) / c - 1.0, close)

    assert not report.ok
    causal = report.check("causality")
    assert causal.status == "FAIL"
    assert causal.difference is not None
    assert causal.difference.bar in report.check_bars
    assert causal.difference.label in close.columns
    # Computed on data ending at the bar, the forward return does not exist yet.
    assert np.isnan(causal.difference.reference) and np.isfinite(causal.difference.observed)
    # A purely forward feature has no value at decision time at all.
    assert report.check("coverage").status == "FAIL"


def test_a_full_sample_zscore_is_caught(close) -> None:
    report = check_causality(lambda c: (c - c.mean()) / c.std(), close)

    assert report.check("causality").status == "FAIL"
    assert report.check("coverage").status == "PASS"


def test_a_centred_rolling_window_is_caught(close) -> None:
    report = check_causality(lambda c: c.rolling(11, center=True).mean(), close)

    assert report.check("causality").status == "FAIL"


def test_a_backward_fill_is_caught_only_where_the_panel_has_a_gap(close) -> None:
    """A blind spot, pinned: a look-ahead on a path the data never exercises is not seen."""

    def filled(c: pd.DataFrame) -> pd.DataFrame:
        return c.bfill()

    assert check_causality(filled, close).ok

    gapped = close.copy()
    last_check_bar = N_BARS - causality.DEFAULT_FUTURE_BARS - 1
    gapped.iloc[last_check_bar - 2 : last_check_bar + 3, 0] = np.nan
    report = check_causality(filled, gapped)
    assert report.check("causality").status == "FAIL"
    assert report.check("causality").difference.label == "asset_0"


def test_a_look_ahead_longer_than_the_withheld_future_is_not_seen(close) -> None:
    """The geometry, pinned: a ten-bar look-ahead is caught with 21 bars withheld and missed with 5."""

    def ten_ahead(c: pd.DataFrame) -> pd.DataFrame:
        return (c.shift(-10) / c - 1.0).fillna(0.0)

    seeing = check_causality(ten_ahead, close, future_bars=21)
    blind = check_causality(ten_ahead, close, future_bars=5)

    assert seeing.check("causality").status == "FAIL"
    assert blind.ok, "with 5 bars withheld a 10-bar look-ahead has nothing to read; if this now fails, update the docs"


def _cached_forward_leak(forward_bars: int = 5, lookback_bars: int = 30):
    """A closure that computes a forward feature on the first panel it sees and caches it by date."""
    cache: dict[str, pd.DataFrame] = {}

    def score(c: pd.DataFrame) -> pd.Series:
        if "features" not in cache:
            cache["features"] = c.shift(-forward_bars) / c - 1.0
        features = cache["features"]
        stamp = c.index[-1]
        if stamp in features.index and bool(features.loc[stamp].notna().any()):
            return features.loc[stamp]
        return c.iloc[-1] / c.iloc[-1 - lookback_bars] - 1.0

    return score


def test_a_cached_feature_is_caught_when_a_factory_isolates_the_reference(close) -> None:
    report = check_causality(_cached_forward_leak(), close, factory=_cached_forward_leak)

    assert report.isolated
    assert report.check("causality").status == "FAIL"
    assert "something kept from the longer panel" in report.check("causality").detail
    assert report.check("cache_reuse").status == "FAIL"


def test_without_a_factory_a_cached_feature_can_hide(close) -> None:
    """A blind spot, pinned: the first, shortest prefix fills the cache, so the leak never shows."""
    report = check_causality(_cached_forward_leak(), close)

    assert not report.isolated
    assert report.ok
    assert "not isolated" in report.text()


def test_a_nondeterministic_function_fails_determinism(close) -> None:
    rng = np.random.default_rng(0)
    report = check_causality(lambda c: c.iloc[-1] + rng.normal(0.0, 1e-12, c.shape[1]), close)

    assert report.check("determinism").status == "FAIL"
    assert "fix it first" in report.check("determinism").detail


def test_a_function_that_is_never_finite_fails_coverage(close) -> None:
    report = check_causality(lambda c: c.iloc[-1] * np.nan, close)

    assert _statuses(report) == {"causality": "PASS", "cache_reuse": "PASS", "determinism": "PASS", "coverage": "FAIL"}
    assert not report.ok


def test_every_panel_is_cut(close, volume) -> None:
    def dollar_volume(c: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
        return (c * v).rolling(20).mean()

    def next_day_volume(c: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
        return c.pct_change(20) * v.shift(-1)

    assert check_causality(dollar_volume, close, volume).ok
    assert check_causality(next_day_volume, close, volume).check("causality").status == "FAIL"


def test_a_single_asset_series_and_a_scalar(close) -> None:
    prices = close["asset_0"]

    trailing = check_causality(lambda s: s.pct_change(20), prices)
    assert trailing.ok and trailing.output == "panel"

    leaking = check_causality(lambda s: s.shift(-1), prices)
    assert leaking.check("causality").status == "FAIL"
    assert leaking.check("causality").difference.label == causality.SCALAR_LABEL

    scalar = check_causality(lambda s: float(s.iloc[-1] / s.iloc[-21] - 1.0), prices)
    assert scalar.ok and scalar.output == "cross_section"


def test_numpy_outputs(close) -> None:
    flat = check_causality(lambda c: (c.iloc[-1] / c.iloc[-21]).to_numpy(), close)
    assert flat.ok and flat.output == "cross_section"

    # np.roll(-1) hands every row the next bar's values.
    rolled = check_causality(lambda c: np.roll(c.to_numpy(), -1, axis=0), close)
    assert rolled.output == "panel"
    assert rolled.check("causality").status == "FAIL"


#: Leaks written as a 1-D array (or, for one case, a frame turned into one), as numpy or numba
#: single-asset code returns them: one value per bar, no bar index.
PER_BAR_ARRAY_LEAKS = {
    "next bar's price": (lambda s: s.shift(-1).to_numpy(), "series"),
    "np.roll by one bar": (lambda s: np.roll(s.to_numpy(), -1), "series"),
    "full-sample z-score": (lambda s: ((s - s.mean()) / s.std()).to_numpy(), "series"),
    "next-day log return aligned to today": (lambda s: np.r_[np.diff(np.log(s.to_numpy())), np.nan], "series"),
    "cross-sectional mean five bars ahead": (lambda c: c.mean(axis=1).shift(-5).to_numpy(), "frame"),
}


@pytest.mark.parametrize("name", list(PER_BAR_ARRAY_LEAKS))
def test_a_leak_in_a_one_dimensional_array_with_one_value_per_bar_is_caught(close, name) -> None:
    fn, panel = PER_BAR_ARRAY_LEAKS[name]
    report = check_causality(fn, close["asset_0"] if panel == "series" else close)

    assert report.output == "panel"
    assert report.check("causality").status == "FAIL"
    assert not report.ok


def test_an_honest_one_dimensional_array_with_one_value_per_bar_passes(close) -> None:
    report = check_causality(lambda s: s.pct_change(20).to_numpy(), close["asset_0"])

    assert report.ok, report.text()
    assert report.output == "panel"


def test_a_series_numbered_by_position_with_one_value_per_bar_is_refused(close) -> None:
    with pytest.raises(CausalityCheckError, match="lost its index"):
        check_causality(lambda s: s.shift(-1).reset_index(drop=True), close["asset_0"])


#: Per-bar results whose labels change with the number of bars handed in, read as cross sections.
CHANGING_LABELS = {
    "an explicit cross section that is one value per bar": (lambda s: s.shift(-1).to_numpy(), "cross_section"),
    "an array one shorter than the bars": (lambda s: np.diff(s.to_numpy()), "auto"),
    "dates moved one bar back": (lambda s: s.shift(-1, freq="B"), "auto"),
    "a monthly resample": (lambda s: s.resample("ME").last(), "auto"),
}


@pytest.mark.parametrize("name", list(CHANGING_LABELS))
def test_a_cross_section_whose_labels_change_with_the_bars_handed_in_is_refused(close, name) -> None:
    """A per-bar result read as a cross section would never be compared with the reference."""
    fn, output = CHANGING_LABELS[name]
    with pytest.raises(CausalityCheckError, match="cannot depend on how many bars"):
        check_causality(fn, close["asset_0"], output=output)


def test_a_cross_section_of_assets_that_come_and_go_is_allowed(close) -> None:
    """Labels that are all columns of the panel may change: assets drop in and out."""
    above_median = check_causality(lambda c: c.iloc[-1].where(c.iloc[-1] > c.iloc[-1].median()).dropna(), close)

    assert above_median.ok, above_median.text()
    assert above_median.output == "cross_section"


def test_an_array_as_long_as_both_the_bars_handed_in_and_the_columns_is_refused(close) -> None:
    first_prefix = N_BARS - causality.DEFAULT_CHECK_WINDOW_BARS - causality.DEFAULT_FUTURE_BARS + 1
    wide = pd.DataFrame(
        np.repeat(close.to_numpy()[:, :1], first_prefix, axis=1),
        index=close.index,
        columns=[f"asset_{i}" for i in range(first_prefix)],
    )

    with pytest.raises(CausalityCheckError, match="unclear whether it holds one value per asset"):
        check_causality(lambda c: (c.iloc[-1] / c.iloc[-21]).to_numpy(), wide)
    assert check_causality(lambda c: (c.iloc[-1] / c.iloc[-21]).to_numpy(), wide, output="cross_section").ok


def test_an_output_holding_only_the_last_bar_is_that_bars_cross_section(close) -> None:
    """``.tail(1)`` is a cross section, not a panel missing every check bar."""
    frame = check_causality(lambda c: c.pct_change(20).tail(1), close)
    series = check_causality(lambda s: s.pct_change(20).tail(1), close["asset_0"])
    explicit = check_causality(lambda c: c.pct_change(20).tail(1), close, output="cross_section")

    for report in (frame, series, explicit):
        assert report.ok, report.text()
        assert report.output == "cross_section"


def test_a_panel_output_holding_only_its_last_bars_is_refused_not_reported_as_lookahead(close) -> None:
    for tail in (1, causality.DEFAULT_FUTURE_BARS + 3):
        with pytest.raises(CausalityCheckError, match="returns only the last bar"):
            check_causality(lambda c, k=tail: c.pct_change(20).tail(k), close, output="panel")


def test_a_row_missing_only_from_the_full_panels_output_fails_causality(close) -> None:
    def drops_the_row_25_bars_from_the_end(c: pd.DataFrame) -> pd.DataFrame:
        out = c.pct_change(20)
        return out.drop(out.index[-25])

    report = check_causality(drops_the_row_25_bars_from_the_end, close)

    assert report.output == "panel"
    causal = report.check("causality")
    assert causal.status == "FAIL"
    assert "which rows the function returns depends on later bars" in causal.detail
    assert causal.difference is not None and causal.difference.bar == close.index[N_BARS - 25]


def test_rtol_relaxes_every_comparison(close) -> None:
    def length_dependent(c: pd.DataFrame) -> pd.DataFrame:
        return c.pct_change(5) * (1.0 + 1e-13 * len(c))

    assert check_causality(length_dependent, close).check("causality").status == "FAIL"
    relaxed = check_causality(length_dependent, close, rtol=1e-9)
    assert relaxed.ok
    assert "relative tolerance 1e-09" in relaxed.text()


def test_a_function_that_mutates_its_input_cannot_alter_the_panel(close) -> None:
    before = close.copy()

    def mutating(c: pd.DataFrame) -> pd.Series:
        c.iloc[-1] = c.iloc[-1] * 2.0
        return c.iloc[-1] / c.iloc[-21] - 1.0

    assert check_causality(mutating, close).ok
    pd.testing.assert_frame_equal(close, before)


def test_assert_causal_returns_the_report_or_raises_an_assertion_error(close) -> None:
    report = assert_causal(lambda c: c.shift(1), close)
    assert report.ok

    with pytest.raises(LookaheadDetected) as caught:
        assert_causal(lambda c: c.shift(-1), close)
    assert isinstance(caught.value, AssertionError)
    assert not caught.value.report.ok
    assert str(caught.value).startswith("look-ahead check: FAIL")


def test_the_report_text_and_lookup(close) -> None:
    report = check_causality(lambda c: c.shift(-1), close)
    text = report.text()

    assert text.splitlines()[0] == "look-ahead check: FAIL"
    for name in CHECKS:
        assert name in text
    assert f"{causality.DEFAULT_FUTURE_BARS} withheld after them" in text
    with pytest.raises(KeyError, match="no check named"):
        report.check("nonsense")


# ---------------------------------------------------------------------------
# Refusals: no verdict is better than a fabricated one
# ---------------------------------------------------------------------------


def _honest(c: pd.DataFrame) -> pd.Series:
    return c.iloc[-1] / c.iloc[-21] - 1.0


def test_refusals_of_the_inputs(close, volume) -> None:
    with pytest.raises(CausalityCheckError, match="at least one panel"):
        check_causality(_honest)
    with pytest.raises(CausalityCheckError, match="DataFrame or Series"):
        check_causality(_honest, close.to_numpy())
    with pytest.raises(CausalityCheckError, match="sorted ascending"):
        check_causality(_honest, close.iloc[::-1])
    with pytest.raises(CausalityCheckError, match="sorted ascending"):
        check_causality(_honest, pd.concat([close, close.iloc[-1:]]))
    with pytest.raises(CausalityCheckError, match="does not share"):
        check_causality(_honest, close, volume.iloc[1:])
    with pytest.raises(CausalityCheckError, match="needs at least 27 rows"):
        check_causality(_honest, close.iloc[:26])
    with pytest.raises(CausalityCheckError, match="check_window_bars"):
        check_causality(_honest, close, check_window_bars=0)
    with pytest.raises(CausalityCheckError, match="future_bars"):
        check_causality(_honest, close, future_bars=True)
    with pytest.raises(CausalityCheckError, match="output must be one of"):
        check_causality(_honest, close, output="rows")
    with pytest.raises(CausalityCheckError, match="rtol"):
        check_causality(_honest, close, rtol=-1.0)
    with pytest.raises(CausalityCheckError, match="score must be callable"):
        check_causality("momentum", close)  # type: ignore[arg-type]
    with pytest.raises(CausalityCheckError, match="not a callable score function"):
        check_causality(_honest, close, factory=lambda: 1.0)


def test_a_function_that_raises_is_reported_with_the_call_that_failed(close) -> None:
    def needs_300(c: pd.DataFrame) -> pd.Series:
        if len(c) < 300:
            raise ValueError("needs 300 bars")
        return _honest(c)

    with pytest.raises(CausalityCheckError, match="raised ValueError on the causal reference") as caught:
        check_causality(needs_300, close.iloc[:200])
    assert isinstance(caught.value.__cause__, ValueError)


def test_refusals_of_the_outputs(close) -> None:
    with pytest.raises(CausalityCheckError, match="non-numeric"):
        check_causality(lambda c: pd.Series("buy", index=c.columns), close)
    with pytest.raises(CausalityCheckError, match="more than once"):
        check_causality(lambda c: pd.Series([1.0, 2.0], index=["a", "a"]), close)
    with pytest.raises(CausalityCheckError, match="returned a DataFrame"):
        check_causality(lambda c: c, close, output="cross_section")
    with pytest.raises(CausalityCheckError, match="indexed by bar"):
        check_causality(lambda c: c.iloc[-1].to_frame("score"), close)
    with pytest.raises(CausalityCheckError, match="expected a Series"):
        check_causality(lambda c: None, close)


def test_an_ambiguous_series_is_refused_and_an_explicit_output_settles_it(close) -> None:
    """With integer bars, a Series labelled 0..k-1 could be one value per asset or per bar."""
    numbered = close.reset_index(drop=True)

    def positional(c: pd.DataFrame) -> pd.Series:
        return pd.Series((c.iloc[-1] / c.iloc[-21] - 1.0).to_numpy())

    with pytest.raises(CausalityCheckError, match="unclear"):
        check_causality(positional, numbered)
    assert check_causality(positional, numbered, output="cross_section").ok


# ---------------------------------------------------------------------------
# Hygiene and the example
# ---------------------------------------------------------------------------


def test_the_module_imports_only_numpy_and_pandas() -> None:
    tree = ast.parse((REPO_ROOT / "src" / "prism" / "causality.py").read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    third_party = roots - set(sys.stdlib_module_names) - {"__future__"}
    assert third_party == {"numpy", "pandas"}

    probe = (
        "import json, sys; import prism.causality; "
        "print(json.dumps(sorted(m for m in sys.modules if m == 'prism' or m.startswith('prism.') or m == 'scipy')))"
    )
    proc = subprocess.run(
        [sys.executable, "-P", "-c", probe],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == ["prism", "prism.causality"]


def test_the_synthetic_example_runs_and_reaches_the_expected_verdicts(capsys) -> None:
    spec = importlib.util.spec_from_file_location("lookahead_check_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reports = module.main()
    out = capsys.readouterr().out

    assert out.splitlines()[0] == module.HEADER
    assert {name: report.ok for name, report in reports.items()} == module.EXPECTED
    for name in module.EXPECTED:
        assert name in out
