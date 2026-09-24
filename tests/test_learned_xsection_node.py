"""Mechanics gates of the learned cross-section family (docs/learned_xsection_design.md §5).

Each test is one of the design's landing gates or guards a property the design
states in words:

* parity — with the learner bypassed and the score pinned to ``mom_12_1``, the
  decile book equals the momentum node's (the family's central parity test);
* causality — appending future bars never changes a past score, and every
  training row satisfies ``s + 21 <= t`` (property-tested, not assumed);
* point-in-time membership — a name that is not an index member at a
  formation bar never enters that bar's training rows, and the node refuses to
  score with no membership bound;
* NaN discipline — a name with a NaN feature takes no position, NaN rows never
  enter the fit;
* determinism — identical inputs give identical scores and coefficients;
* the learners — ridge recovers a planted linear rule; the numpy GBM fits a
  planted step and respects ``min_leaf``;
* the membership mask the engine binds is the view's own interval rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.portfolio import construct_decile_neutral
from prism.residual.factors import ResidualStatArbConfig
from prism.signal import MomentumSignalNode, Signal, check_declarations
from prism.signal.learned_xsection_node import (
    FEATURES,
    LearnedCrossSectionError,
    LearnedCrossSectionNode,
    _fit_gbm,
    _predict_gbm,
    _rank_centered,
    _ridge,
)


def _config() -> ResidualStatArbConfig:
    return ResidualStatArbConfig(
        corr_window=60,
        regr_window=20,
        n_factors=2,
        min_price=5.0,
        min_median_dollar_volume=1_000.0,
        dollar_volume_window=5,
    )


def _panel(n_days: int = 1100, n_assets: int = 40, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n_days, freq="B", tz="America/New_York")
    market = rng.normal(0.0, 0.01, size=(n_days, 1))
    returns = market + rng.normal(0.0, 0.015, size=(n_days, n_assets))
    closes = pd.DataFrame(
        50.0 * np.exp(np.cumsum(returns, axis=0)), index=idx, columns=[f"S{i:02d}" for i in range(n_assets)]
    )
    volumes = pd.DataFrame(
        rng.uniform(1e6, 5e6, size=(n_days, n_assets)), index=idx, columns=closes.columns
    )
    return closes, volumes


def _all_members(closes: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=closes.index, columns=closes.columns)


def _node(**kwargs: object) -> LearnedCrossSectionNode:
    kwargs.setdefault("n_cross_sections", 36)
    node = LearnedCrossSectionNode(_config(), **kwargs)
    return node


class TestContract:
    def test_declares_itself_as_a_stateless_rank_score(self):
        node = _node()
        assert isinstance(node, Signal)
        check_declarations(node)
        assert node.output_kind == "rank_score" and node.fitted_state == "none"
        assert node.uses_membership is True
        assert node.required_history == 253

    def test_refuses_to_score_without_a_membership_binding(self):
        closes, volumes = _panel(n_days=400)
        with pytest.raises(LearnedCrossSectionError, match="membership"):
            _node().score(closes, volumes)

    def test_runway_short_of_36_cross_sections_is_all_nan_not_an_error(self):
        closes, volumes = _panel(n_days=900)  # 900 - 1 - 21*36 = 143 < 252
        node = _node()
        node.bind_membership(_all_members(closes))
        scores = node.score(closes, volumes)
        assert scores.isna().all()
        assert node.last_fit == {"learner": "ridge", "runway": node.last_fit["runway"], "needed": 36}

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(learner="lasso"),
            dict(features=("mom_12_1", "earnings_yield")),
            dict(features=()),
            dict(ridge_lambda=-1.0),
            dict(spacing_bars=10, horizon_bars=21),
            dict(gbm_trees=0),
        ],
    )
    def test_bad_params_raise(self, kwargs):
        with pytest.raises(ValueError):
            LearnedCrossSectionNode(_config(), **kwargs)


class TestParity:
    def test_bypass_decile_book_equals_the_momentum_node_s(self):
        """Design §5: learner bypassed, score = mom_12_1 -> the sibling book, bit for bit."""
        closes, volumes = _panel(n_days=600, n_assets=60)
        bypass = LearnedCrossSectionNode(_config(), learner="bypass")
        bypass.bind_membership(_all_members(closes))
        momentum = MomentumSignalNode(_config(), lookback_bars=252, skip_bars=21)
        for cut in (400, 470, 599):
            a = bypass.score(closes.iloc[: cut + 1], volumes.iloc[: cut + 1])
            b = momentum.score(closes.iloc[: cut + 1], volumes.iloc[: cut + 1])
            assert a.notna().equals(b.notna())
            book_a = construct_decile_neutral(a.to_frame().T, decile=0.10)
            book_b = construct_decile_neutral(b.to_frame().T, decile=0.10)
            pd.testing.assert_frame_equal(book_a, book_b)
            assert (book_a.abs().sum(axis=1) > 0.99).all()  # a real book, not two empty rows


class TestCausality:
    def test_appending_future_bars_changes_no_past_score(self):
        closes, volumes = _panel(n_days=1150)
        node = _node()
        node.bind_membership(_all_members(closes))
        cut = 1080
        at_cut = node.score(closes.iloc[: cut + 1], volumes.iloc[: cut + 1])
        full = _node()
        full.bind_membership(_all_members(closes))
        full.score(closes, volumes)  # score the longer panel first
        again = full.score(closes.iloc[: cut + 1], volumes.iloc[: cut + 1])
        pd.testing.assert_series_equal(at_cut, again)
        assert at_cut.notna().sum() > 10

    @pytest.mark.parametrize("n_days", [1050, 1100, 1163])
    def test_every_training_target_is_realized_at_the_decision_bar(self, n_days):
        """Property: 36 formation bars spaced 21 apart, and s + 21 <= t for every one."""
        closes, volumes = _panel(n_days=n_days)
        node = _node()
        node.bind_membership(_all_members(closes))
        node.score(closes, volumes)
        fit = node.last_fit
        assert fit is not None and "formation_positions" in fit
        t = fit["decision_position"]
        positions = fit["formation_positions"]
        assert t == n_days - 1
        assert len(positions) == 36
        assert np.all(np.diff(positions) == 21)
        assert fit["max_target_position"] == max(positions) + 21 <= t
        assert min(positions) >= 252  # every formation bar has full feature history

    def test_a_decision_bar_price_perturbation_moves_only_the_latest_target(self):
        """close[t] is the target bar of the most recent cross-section and of nothing earlier."""
        closes, volumes = _panel(n_days=1100)
        node = _node()
        node.bind_membership(_all_members(closes))
        node.score(closes, volumes)
        assert node.last_fit["max_target_position"] == len(closes) - 1  # type: ignore[index]


class TestMembership:
    def test_a_non_member_at_formation_bars_never_enters_training(self):
        """A name that joins the index only at t must not shape the fit that scores t."""
        closes, volumes = _panel(n_days=1100, n_assets=40, seed=3)
        # Plant a joiner: S00 has an enormous drift before it joins, which
        # would dominate a momentum coefficient if it were in the pool.
        boosted = closes.copy()
        boosted["S00"] = boosted["S00"] * np.exp(np.linspace(0.0, 4.0, len(boosted)))
        mask = _all_members(boosted)
        mask.loc[: boosted.index[-2], "S00"] = False  # member only on the decision bar
        node = _node()
        node.bind_membership(mask)
        with_joiner_blocked = node.score(boosted, volumes)
        beta_blocked = dict(node.last_fit["beta"])  # type: ignore[index]

        dropped = _node()
        dropped.bind_membership(mask.drop(columns=["S00"]))
        dropped.score(boosted.drop(columns=["S00"]), volumes.drop(columns=["S00"]))
        beta_dropped = dict(dropped.last_fit["beta"])  # type: ignore[index]
        # Training rows are identical, so the coefficients are identical.
        for name in FEATURES:
            assert beta_blocked[name] == pytest.approx(beta_dropped[name], abs=1e-12)
        assert np.isfinite(with_joiner_blocked["S00"])  # a member at t is scored at t

    def test_a_non_member_at_t_gets_no_opinion(self):
        closes, volumes = _panel(n_days=1100)
        mask = _all_members(closes)
        mask.iloc[-1, 5] = False
        node = _node()
        node.bind_membership(mask)
        assert np.isnan(node.score(closes, volumes).iloc[5])


class TestNanDiscipline:
    def test_an_eligible_name_with_a_nan_feature_is_liquidated_and_scores_nan(self):
        """A NaN volume 30 bars back leaves the 5-bar dollar-volume screen passing and dv_63 undefined."""
        closes, volumes = _panel(n_days=1100)
        volumes = volumes.copy()
        name = closes.columns[7]
        volumes.iloc[-30, 7] = np.nan
        node = _node()
        node.bind_membership(_all_members(closes))
        scores = node.score(closes, volumes)
        marks = node.instructions(closes, volumes)
        assert np.isnan(scores[name])
        assert marks is not None and marks.get(name) == "liquidate"
        # Every other name still scores: the NaN row left the fit, not the book.
        assert scores.drop(index=name).notna().sum() >= 35

    def test_no_marks_when_every_eligible_name_has_finite_features(self):
        closes, volumes = _panel(n_days=1100)
        node = _node()
        node.bind_membership(_all_members(closes))
        assert node.instructions(closes, volumes) is None


class TestDeterminismAndLearners:
    def test_identical_inputs_give_identical_scores(self):
        closes, volumes = _panel(n_days=1100)
        a, b = _node(), _node()
        a.bind_membership(_all_members(closes))
        b.bind_membership(_all_members(closes))
        pd.testing.assert_series_equal(a.score(closes, volumes), b.score(closes, volumes))
        g1, g2 = _node(learner="gbm", gbm_trees=20), _node(learner="gbm", gbm_trees=20)
        g1.bind_membership(_all_members(closes))
        g2.bind_membership(_all_members(closes))
        pd.testing.assert_series_equal(g1.score(closes, volumes), g2.score(closes, volumes))

    def test_rank_transform_is_centered_and_bounded(self):
        values = np.array([3.0, 1.0, 2.0, 2.0, 10.0])
        ranked = _rank_centered(values)
        assert ranked.mean() == pytest.approx(0.0, abs=1e-15)
        assert ranked.min() == -0.5 and ranked.max() == 0.5
        assert ranked[2] == ranked[3]

    def test_ridge_recovers_a_planted_linear_rule(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-0.5, 0.5, size=(5000, 3))
        y = X @ np.array([0.5, -0.2, 0.0]) + rng.normal(0, 0.01, 5000)
        beta = _ridge(X, y, 1.0)
        assert beta == pytest.approx([0.5, -0.2, 0.0], abs=0.01)

    def test_gbm_fits_a_planted_step_and_respects_min_leaf(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(-0.5, 0.5, size=(2000, 2))
        y = np.where(X[:, 0] > 0.1, 1.0, -1.0)
        model = _fit_gbm(X, y, n_trees=100, depth=2, learning_rate=0.1, min_leaf=50)
        prediction = _predict_gbm(model, X)
        assert np.corrcoef(prediction, y)[0, 1] > 0.99
        # min_leaf: every split's children in the first tree hold >= 50 rows.
        first = model["trees"][0]  # type: ignore[index]
        root = first[0]
        assert root[0] == "split" and root[1] == 0 and root[2] == pytest.approx(0.1, abs=0.02)


def test_learned_node_scores_a_realistic_cross_section_with_every_learner():
    closes, volumes = _panel(n_days=1100)
    for learner in ("ridge", "gbm", "bypass"):
        node = _node(learner=learner, gbm_trees=10)
        node.bind_membership(_all_members(closes))
        scores = node.score(closes, volumes)
        assert scores.notna().sum() >= 30, learner


FROZEN = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "frozen" / "demotion_b1"


@pytest.mark.skipif(not (FROZEN / "manifest.json").exists(), reason="frozen demotion_b1 dataset not present")
def test_membership_mask_agrees_with_the_view_s_single_date_rule_on_every_sampled_bar():
    """The mask the engine binds is the same interval rule as ``DatasetView.membership``, bar by bar."""
    from prism.io.dataset import Dataset

    ds = Dataset.open(FROZEN)
    sessions = ds.manifest.coverage["sessions"]
    end = str(sessions[-1])
    view = ds.view("research_vintage", end, vintage=ds.manifest.vintage)
    universe = sorted({sid for day in sessions[::200] for sid in ds.view("research_vintage", str(day), vintage=ds.manifest.vintage).membership("sp500")})
    close = view.panel(universe, end, "close")
    mask = view.membership_mask("sp500", close.index, universe)
    for position in range(0, len(close.index), 97):
        day = close.index[position].date().isoformat()
        expected = ds.view("research_vintage", day, vintage=ds.manifest.vintage).membership("sp500")
        got = set(mask.columns[mask.iloc[position].to_numpy()])
        assert got == (expected & set(universe)), day


def test_the_evaluator_binds_membership_only_to_components_that_declare_it():
    from types import SimpleNamespace

    from prism.evaluate import _bind_membership

    closes, _ = _panel(n_days=300, n_assets=5)
    learned = LearnedCrossSectionNode(_config())
    momentum = MomentumSignalNode(_config())
    calls: list[tuple[str, int]] = []

    class _View:
        def membership_mask(self, index_id, index, columns):
            calls.append((index_id, len(columns)))
            return pd.DataFrame(True, index=index, columns=columns)

    plans = [SimpleNamespace(component=learned), SimpleNamespace(component=momentum)]
    spec = SimpleNamespace(strategy=SimpleNamespace(universe_index="sp500"))
    _bind_membership(spec, _View(), plans, closes)
    assert calls == [("sp500", 5)]
    assert learned._membership is not None and learned._membership.shape == closes.shape
    assert not hasattr(momentum, "_membership")
