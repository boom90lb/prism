"""Constant-volatility sizing: the rule, its failure path, and the engine that applies it.

Pinned here:

* the rule's arithmetic is the public factor lab's
  (``research/scripts/factor_lab.py::scale_constant_vol``) at the same
  decision bars, to the bit;
* every guard fires on the series it exists for -- a panel that ends before
  the decision bar, a short or broken window, a stale or flat panel, a
  degenerate estimate -- and the failure path never exceeds 1 or the last
  good scale; a short window before any good decision is warm-up;
* nothing after the decision bar moves the decision (N1);
* the specification: ``[sizing]`` enters ``config_id`` only when declared and
  is refused where nothing would apply it;
* the engine: per-plan scales move only on the plan's own due bar, a tranched
  book's base book is not diluted by staged entry, the account target never
  exceeds the gross ceiling, a rule pinned at scale 1 is the unscaled book bit
  for bit, and the run records what it did.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prism.evaluate as evaluate_module
from prism.composition import ComponentPlan, compose, tranche_plans
from prism.evaluate import (
    CanonicalSettings,
    EvaluationError,
    RunRefused,
    smoke_evaluate,
    walk_forward_evaluate,
)
from prism.execution.cost_model import CostStack
from prism.io import sessions
from prism.io.synthetic import write_synthetic_dataset
from prism.portfolio.vol_scale import (
    FALLBACK_REASONS,
    VolScaleError,
    VolScaleRule,
    base_book_returns,
    cap_gross,
    decide_scale,
    session_exposure,
)
from prism.signal.base import Signal
from prism.spec import SpecError, load_spec, load_spec_table
from prism.validation.walk_forward import RollingWalkForward
from prism.workspace import Workspace
from research.scripts import factor_lab

REPO_ROOT = Path(__file__).resolve().parents[1]
DAY = "2026-03-02"
RULE = VolScaleRule()


def _decide(returns, *, last_good=None, rule=RULE, panel_last_bar=DAY):
    return decide_scale(returns, rule, last_good=last_good, decision_bar=DAY, panel_last_bar=panel_last_bar)


def _regime_returns(seed: int = 7, n: int = 900) -> np.ndarray:
    """Daily returns whose volatility moves between regimes, including one quiet enough to hit the cap."""
    rng = np.random.default_rng(seed)
    vol = np.repeat([0.004, 0.012, 0.025, 0.008, 0.002, 0.015], n // 6 + 1)[:n]
    return rng.normal(0.0002, 1.0, n) * vol


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("refresh", "lookback", "target", "cap"), [(21, 126, 0.12, 2.0), (5, 40, 0.10, 1.5)])
def test_the_scale_is_the_factor_labs_at_the_same_decision_bars(refresh, lookback, target, cap) -> None:
    returns = _regime_returns()
    rule = VolScaleRule(target_vol=target, lookback_bars=lookback, max_scale=cap)
    # The lab's effective series: the decision at close i applies from session i + 1.
    effective = factor_lab.scale_constant_vol(
        returns, refresh=refresh, target_vol=target, lookback_sessions=lookback, max_scale=cap
    )
    statuses = set()
    last_good = None
    for i in np.flatnonzero(factor_lab.refresh_mask(returns.size - 1, refresh)):
        decision = _decide(returns[: i + 1], last_good=last_good, rule=rule)
        statuses.add(decision.status)
        assert decision.scale == effective[i + 1], f"session {i}: {decision} vs {effective[i + 1]}"
        if decision.status == "ok":
            last_good = decision.scale
    assert statuses == {"warmup", "ok"}
    assert np.isclose(effective, cap).any() and (effective < 1.0).any(), "the series exercises the cap and a de-gross"


def test_a_short_window_before_any_good_decision_is_warm_up_at_scale_one() -> None:
    decision = _decide(_regime_returns()[:125])
    assert (decision.scale, decision.status, decision.reason, decision.n_valid) == (1.0, "warmup", None, 125)
    assert math.isnan(decision.sigma)


def _window(kind: str) -> np.ndarray:
    returns = _regime_returns(n=300)[-126:].copy()
    if kind == "nan":
        returns[40] = np.nan
    elif kind == "zeros":
        returns[:7] = 0.0  # 7 / 126 = 5.6% > 5%
    elif kind == "flat":
        returns[:] = 0.003  # finite, no zeros, no dispersion
    return returns


@pytest.mark.parametrize(
    ("kind", "panel_last_bar", "reason"),
    [
        ("clean", "2026-02-27", "decision_bar_not_last_bar"),
        ("nan", DAY, "insufficient_returns"),
        ("zeros", DAY, "zero_returns"),
        ("flat", DAY, "degenerate_sigma"),
    ],
)
@pytest.mark.parametrize(("last_good", "expected"), [(None, 1.0), (1.7, 1.0), (0.6, 0.6)])
def test_every_guard_fires_and_the_failure_never_exceeds_one_or_the_last_good_scale(
    kind, panel_last_bar, reason, last_good, expected
) -> None:
    window = _window(kind)
    decision = _decide(window, last_good=last_good, panel_last_bar=panel_last_bar)
    if reason == "insufficient_returns" and last_good is None:
        # A short window before any good decision is warm-up, not a failure; the scale is the same 1.
        assert (decision.status, decision.reason) == ("warmup", None)
    else:
        assert (decision.status, decision.reason) == ("fallback", reason)
    assert decision.scale == expected
    assert decision.scale <= 1.0 and (last_good is None or decision.scale <= last_good)
    assert reason in FALLBACK_REASONS


def test_the_clean_window_passes_and_six_zeros_are_not_a_stale_panel() -> None:
    clean = _window("clean")
    assert _decide(clean).status == "ok"
    six = clean.copy()
    six[:6] = 0.0  # 6 / 126 = 4.8%
    decision = _decide(six)
    assert decision.status == "ok" and decision.zero_fraction == pytest.approx(6 / 126)


def test_the_panel_end_is_compared_as_a_session_date_whatever_its_type() -> None:
    clean = _window("clean")
    stamp = pd.Timestamp("2026-03-02 16:00", tz="America/New_York")
    assert _decide(clean, panel_last_bar=stamp).status == "ok"
    assert _decide(clean, panel_last_bar=dt.date(2026, 3, 2)).status == "ok"
    with pytest.raises(VolScaleError, match="not a session date"):
        _decide(clean, panel_last_bar="yesterday")


def test_nothing_after_the_decision_bar_moves_the_decision() -> None:
    rng = np.random.default_rng(5)
    index = pd.date_range("2025-01-02", periods=200, freq="B", tz="America/New_York")
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (200, 6)), axis=0)), index=index)
    weights = pd.DataFrame(rng.normal(0, 0.2, (200, 6)), index=index)
    t = 150
    rule = VolScaleRule(lookback_bars=60)
    before = base_book_returns(close, weights)
    decision = decide_scale(before.iloc[: t + 1], rule, last_good=None, decision_bar=index[t], panel_last_bar=index[t])

    perturbed_close, perturbed_weights = close.copy(), weights.copy()
    perturbed_close.iloc[t + 1 :] *= 1.0 + rng.normal(0, 0.3, perturbed_close.iloc[t + 1 :].shape)
    perturbed_weights.iloc[t:] = rng.normal(0, 3.0, perturbed_weights.iloc[t:].shape)  # w[t] only moves r[t+1]
    after = base_book_returns(perturbed_close, perturbed_weights)
    pd.testing.assert_series_equal(after.iloc[: t + 1], before.iloc[: t + 1], check_exact=True)
    assert not np.allclose(after.iloc[t + 1 :], before.iloc[t + 1 :])
    assert decide_scale(after.iloc[: t + 1], rule, last_good=None, decision_bar=index[t], panel_last_bar=index[t]) == (
        decision
    )


def test_base_book_returns_hold_the_prior_close_weights_and_mark_a_missing_price_unchanged() -> None:
    index = pd.date_range("2025-01-02", periods=4, freq="B", tz="America/New_York")
    close = pd.DataFrame({"A": [100.0, 110.0, 121.0, np.nan], "B": [50.0, 45.0, 45.0, 40.0]}, index=index)
    weights = pd.DataFrame({"A": [np.nan, 0.5, 0.5, 9.0], "B": [np.nan, -0.5, -0.5, 9.0]}, index=index)
    returns = base_book_returns(close, weights)
    assert math.isnan(returns.iloc[0]) and math.isnan(returns.iloc[1]), "no book at the first close"
    assert returns.iloc[2] == pytest.approx(0.5 * 0.10 - 0.5 * (45.0 / 45.0 - 1.0))
    # A has no close on the last bar and contributes nothing; B is short 0.5 into a -11.1% move. The last
    # row's weights (9.0) are held from the last close on and enter no return here.
    assert returns.iloc[3] == pytest.approx(-0.5 * (40.0 / 45.0 - 1.0))
    # The same panel in Fortran order is the same returns, to the bit.
    fortran = base_book_returns(close.copy(), weights.copy())
    assert fortran.to_numpy().tobytes() == returns.to_numpy().tobytes()
    stale = pd.DataFrame(100.0, index=index, columns=["A", "B"])
    assert (base_book_returns(stale, weights.fillna(0.0)).iloc[1:] == 0.0).all(), "a stale panel returns zeros"
    with pytest.raises(VolScaleError, match="same index"):
        base_book_returns(close, weights.iloc[1:])
    with pytest.raises(VolScaleError, match="does not carry"):
        base_book_returns(close, weights.assign(C=0.1))


def test_the_rule_resolves_its_defaults_and_refuses_what_it_cannot_mean() -> None:
    rule = VolScaleRule.from_mapping({"rule": "constant_vol"})
    assert rule.to_dict() == {
        "rule": "constant_vol",
        "target_vol": 0.12,
        "lookback_bars": 126,
        "max_scale": 2.0,
        "max_gross": 2.0,
        "annualization_bars": 252,
        "max_zero_return_fraction": 0.05,
    }
    assert VolScaleRule(max_scale=3.0).max_gross == 3.0, "the ceiling defaults to the cap"
    for table, match in [
        ({}, "declares no rule"),
        ({"rule": "variance_managed"}, "not one of"),
        ({"rule": "constant_vol", "lookback": 126}, "unknown key"),
        ({"rule": "constant_vol", "target_vol": 0.0}, "must be > 0"),
        ({"rule": "constant_vol", "target_vol": True}, "finite number"),
        ({"rule": "constant_vol", "lookback_bars": 126.0}, "integer"),
        ({"rule": "constant_vol", "lookback_bars": 1}, ">= 2"),
        ({"rule": "constant_vol", "max_scale": math.inf}, "finite"),
        ({"rule": "constant_vol", "max_zero_return_fraction": 1.0}, r"\[0, 1\)"),
    ]:
        with pytest.raises(VolScaleError, match=match):
            VolScaleRule.from_mapping(table)


def test_the_gross_ceiling_scales_down_and_never_up() -> None:
    target = pd.Series([1.2, -0.9, np.nan, 0.3], index=list("abcd"))
    capped, factor = cap_gross(target, 2.0)
    assert factor == pytest.approx(2.0 / 2.4)
    assert float(np.nansum(np.abs(capped))) == pytest.approx(2.0) and math.isnan(capped["c"])
    same, one = cap_gross(target * 0.5, 2.0)
    assert one == 1.0 and same.equals(target * 0.5)


def test_session_exposure_reports_the_long_side_and_the_sleeve() -> None:
    weights = pd.DataFrame({"A": [0.6, 1.1], "B": [-0.5, -1.0], "C": [np.nan, 0.2]})
    exposure = session_exposure(weights)
    assert exposure["gross"].tolist() == pytest.approx([1.1, 2.3])
    assert exposure["long_over_equity"].tolist() == pytest.approx([0.6, 1.3])
    assert exposure["sleeve_fraction"].tolist() == pytest.approx([0.38, 0.0])


# ---------------------------------------------------------------------------
# The specification
# ---------------------------------------------------------------------------


def _table(**sizing) -> dict:
    table = {
        "strategy": {
            "name": "sized_book",
            "universe_index": "sp500",
            "combination": "position_space",
            "components": [
                {
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
            ],
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
    if sizing:
        table["sizing"] = dict(sizing)
    return table


def _load(table: dict, *, instantiate: bool = False):
    return load_spec_table(table, path="sized.toml", workspace=Workspace(REPO_ROOT), instantiate=instantiate)


def test_sizing_enters_the_resolved_specification_only_when_declared() -> None:
    assert "sizing" not in _load(_table()).resolved()
    assert _load(_table()).sizing is None
    declared = _load(_table(rule="constant_vol")).resolved()["sizing"]
    assert declared == VolScaleRule().to_dict(), "every default is resolved into the identity"
    # Saying a default out loud is the same configuration; changing one is not.
    explicit = _load(_table(rule="constant_vol", max_gross=2.0)).resolved()
    assert explicit == _load(_table(rule="constant_vol")).resolved()
    ids = {
        _load(table, instantiate=True).config_id(engine_version="pinned")
        for table in (_table(), _table(rule="constant_vol"), _table(rule="constant_vol", target_vol=0.10))
    }
    assert len(ids) == 3


def test_sizing_is_refused_where_nothing_would_scale_by_it() -> None:
    with pytest.raises(SpecError, match=r"\[sizing\].*unknown key"):
        _load(_table(rule="constant_vol", lookback=126))
    convex = _table(rule="constant_vol")
    convex["strategy"]["combination"] = "calibrated_convex"
    convex["construction"]["kind"] = "convex"
    convex["strategy"]["components"][0].update(to_weights="none", to_weights_params={})
    with pytest.raises(SpecError, match="position_space"):
        _load(convex)
    renormalized = _table(rule="constant_vol")
    renormalized["strategy"]["renormalize"] = True
    with pytest.raises(SpecError, match="renormalize"):
        _load(renormalized)


# ---------------------------------------------------------------------------
# The tracker, driven through composition
# ---------------------------------------------------------------------------


class TrailingRank(Signal):
    """A five-session trailing return: a book that changes as prices do."""

    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    price_scale_invariant = True
    fitted_state = "none"
    estimated_decay_bars = None

    @property
    def horizon_bars(self) -> int:
        return 5

    @property
    def required_history(self) -> int:
        return 6

    def fit(self, close, volume=None) -> "TrailingRank":
        return self

    def score(self, close, volume=None) -> pd.Series:
        if len(close) < 6:
            return pd.Series(np.nan, index=close.columns, dtype=float)
        return close.iloc[-1] / close.iloc[-6] - 1.0


LABELS = [day.isoformat() for day in sessions.sessions_between(dt.date(2025, 1, 2), dt.date(2025, 12, 31))]
STAMPS = pd.DatetimeIndex([pd.Timestamp(f"{day} 16:00", tz="America/New_York") for day in LABELS])
SIDS = [f"S{i:02d}" for i in range(10)]


def _closes(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    moves = rng.normal(0.0, 0.015, (len(STAMPS), len(SIDS)))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(moves, axis=0)), index=STAMPS, columns=SIDS)


def _tranches(k: int = 5) -> tuple[ComponentPlan, ...]:
    plan = ComponentPlan(
        component=TrailingRank(),
        component_id="trail",
        to_weights="decile_long_short",
        to_weights_params={"decile": 0.2, "max_gross": 1.0, "max_symbol_abs_weight": 1.0},
        update_every_bars=5,
        valid_for_bars=10,
    )
    return tranche_plans(plan, k, 0)


def _drive(close: pd.DataFrame, rule: VolScaleRule, *, k: int = 5, stop: int | None = None):
    """Run composition and the tracker over every bar, as the account_state hook does."""
    plans = _tranches(k)
    tracker = evaluate_module._VolSizing(
        rule, groups={"trail": [plan.component_id for plan in plans]}, session_index=STAMPS, universe=SIDS
    )
    states: dict = {}
    rows, results = [], []
    for position in range(stop if stop is not None else len(LABELS)):
        day = LABELS[position]
        panel = close.iloc[: position + 1]
        result = compose(plans, decision_bar=day, scoring_panels={"price": panel}, sessions=LABELS[: position + 1],
                         states=states)
        states = dict(result.states)
        row = pd.Series(0.0, index=pd.Index(SIDS), dtype=float)
        row.loc[result.targets.index] = result.targets.to_numpy(dtype=float)
        scored = evaluate_module._BarScores(
            day=day, members=tuple(SIDS), close=panel, volume=panel, eligible=pd.Series(True, index=SIDS),
            ineligible=(), composition=result, row=row, adv=None,
        )
        sized = tracker.size(scored, plans, position=position, day=day)
        rows.append(sized.row)
        results.append(result)
    return tracker, rows, results, plans


def test_a_plans_scale_moves_only_on_its_own_due_bar() -> None:
    tracker, _, results, plans = _drive(_closes(), VolScaleRule(target_vol=0.15, lookback_bars=20))
    decisions = pd.DataFrame(tracker.decisions)
    assert set(decisions["status"]) == {"warmup", "ok"}
    due = {(LABELS[p], o.component_id) for p, result in enumerate(results) for o in result.component_outputs if o.due}
    assert set(zip(decisions["bar"], decisions["plan_id"])) == due
    snapshots = pd.DataFrame.from_dict(tracker._snapshots, orient="index").sort_index()
    for plan in plans:
        column = snapshots[f"scale:{plan.component_id}"]
        moved = {LABELS[p] for p in column.index[1:] if column[p] != column[p - 1] and not np.isnan(column[p - 1])}
        assert moved <= {bar for bar, pid in due if pid == plan.component_id}, plan.component_id


def _holding(result) -> list[dict]:
    return [dict(s.sub_targets) for s in result.states.values() if any(v != 0.0 for v in s.sub_targets.values())]


def test_staged_entry_does_not_dilute_the_base_book() -> None:
    tracker, _, results, _ = _drive(_closes(), VolScaleRule(lookback_bars=20), stop=40)
    counts = [len(_holding(result)) for result in results]
    assert counts[:5] == [0] * 5, "no tranche can score before six bars exist"
    first = counts.index(1)
    assert counts[first : first + 5] == [1, 2, 3, 4, 5], "staged entry, one tranche a session"
    # The history keeps only what a later window can reach; a run this short keeps all of it.
    tracker, _, results, _ = _drive(_closes(), VolScaleRule(lookback_bars=20), stop=first + 3)
    history = dict(tracker._history["trail"])
    assert all(history[p] is None for p in range(first)), "no book before the first holding tranche"
    position = first + 2  # three of five tranches built
    built = _holding(results[position])
    assert len(built) == 3
    base = history[position]
    # The base book is the mean of the built tranches, at the rule's raw gross, not 3/5 of it.
    assert float(np.abs(base).sum()) == pytest.approx(1.0)
    mean = pd.DataFrame(built).fillna(0.0).mean().reindex(SIDS, fill_value=0.0).to_numpy()
    np.testing.assert_allclose(base, mean, rtol=0, atol=1e-15)


def test_the_tracker_decides_on_the_base_books_own_returns() -> None:
    close = _closes()
    rule = VolScaleRule(target_vol=0.15, lookback_bars=20)
    tracker, _, _, _ = _drive(close, rule)
    # Rebuild the base book from its recorded weights and check one late decision by hand.
    late = pd.DataFrame(tracker.decisions).iloc[-1]
    position = LABELS.index(late["bar"])
    window = tracker._base_returns("trail", close.iloc[: position + 1], position)
    assert window.size == rule.lookback_bars and np.isfinite(window).all()
    expected = min(rule.max_scale, rule.target_vol / (np.std(window, ddof=1) * math.sqrt(252)))
    assert late["status"] == "ok" and late["scale"] == pytest.approx(expected, rel=1e-12)


def test_nothing_after_a_bar_moves_the_trackers_decisions_through_it() -> None:
    close = _closes()
    rule = VolScaleRule(target_vol=0.15, lookback_bars=20)
    cut = 120
    shocked = close.copy()
    shocked.iloc[cut + 1 :] *= np.exp(np.random.default_rng(9).normal(0, 0.2, shocked.iloc[cut + 1 :].shape))
    base, base_rows, _, _ = _drive(close, rule, stop=len(LABELS))
    moved, moved_rows, _, _ = _drive(shocked, rule, stop=len(LABELS))
    through = [row for row in base.decisions if row["bar"] <= LABELS[cut]]
    assert through and through == [row for row in moved.decisions if row["bar"] <= LABELS[cut]]
    for position in range(cut + 1):
        pd.testing.assert_series_equal(base_rows[position], moved_rows[position], check_exact=True)
    assert base.decisions != moved.decisions, "the shock does move later decisions"


def test_at_scale_one_the_sized_target_is_compositions_bit_for_bit() -> None:
    pinned = VolScaleRule(target_vol=1e6, lookback_bars=20, max_scale=1.0)
    tracker, rows, results, _ = _drive(_closes(), pinned)
    assert set(pd.DataFrame(tracker.decisions)["scale"]) == {1.0}
    for row, result in zip(rows, results):
        expected = pd.Series(0.0, index=pd.Index(SIDS), dtype=float)
        expected.loc[result.targets.index] = result.targets.to_numpy(dtype=float)
        pd.testing.assert_series_equal(row, expected, check_exact=True)


# ---------------------------------------------------------------------------
# The engine (the scaffolding of tests/test_grid_schedule.py, duplicated)
# ---------------------------------------------------------------------------

PACKAGE = "sizing_strategies"
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
name = "sized_trailing_book"
universe_index = "synthetic_100"
combination = "position_space"

[[strategy.components]]
id = "trailing"
path = "sizing_strategies.trailing:TrailingReturn"
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
{sizing}
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
    root = tmp_path_factory.mktemp("sizing") / "source"
    Workspace(root).ensure()
    write_synthetic_dataset(
        Workspace(root).data,
        seed=11,
        scenarios="all",
        n_equities=12,
        n_etfs=2,
        start=dt.date(2026, 1, 1),
        end=dt.date(2026, 7, 31),
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
    volume = {"estimator": "median", "window": "trailing", "window_bars": 20, "refresh": "decision",
              "floor_dollars": 100_000.0}
    stack = CostStack.from_mapping(
        {
            "spread": {"mode": "flat", "flat_bps": 0.0},
            "impact": {"quadratic_notional_coeff": 0.0, "participation_model": "linear", "participation_coeff": 0.0},
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


def _spec(engine_root: Path, tmp_path: Path, name: str, sizing: str, grid: str = "tranches = 5\n"):
    root = tmp_path / name
    root.mkdir()
    for entry in sorted(engine_root.iterdir()):
        if entry.name not in {"results", "trials", "state", "runs", "logs", "specs"}:
            (root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    workspace = Workspace(root).ensure()
    (root / "specs").mkdir()
    (root / "specs" / "book.toml").write_text(BOOK_TOML.format(sizing=sizing, grid=grid), encoding="utf-8")
    return load_spec(root / "specs" / "book.toml", workspace=workspace), workspace


def _run(engine_root: Path, tmp_path: Path, name: str, sizing: str, **options):
    spec, workspace = _spec(engine_root, tmp_path, name, sizing)
    settings = {
        "engine": "canonical",
        "canonical": _canonical_settings(),
        "decision_coupling": "account_state",
        "fold_portfolio": "carry",
        "capital": 100_000.0,
    }
    settings.update(options)
    return walk_forward_evaluate(
        spec,
        workspace=workspace,
        splitter=SPLITTER,
        fold_table_id="sizing_unit",
        selection_set_id="unit_set",
        registration_id="unit_set",
        kind="new",
        **settings,
    )


SIZED = '[sizing]\nrule = "constant_vol"\ntarget_vol = {target}\nlookback_bars = 10\nmax_scale = {cap}\n{extra}'


@pytest.fixture(scope="module")
def plain(engine_root, tmp_path_factory):
    """The tranched book without ``[sizing]``, run once for every comparison below."""
    return _run(engine_root, tmp_path_factory.mktemp("plain"), "plain", "")


def test_an_engine_run_sizes_every_tranche_and_stays_under_the_ceiling(engine_root, plain, tmp_path) -> None:
    sized = _run(engine_root, tmp_path, "sized", SIZED.format(target=0.6, cap=2.0, extra=""))
    assert sized.config_id != plain.config_id
    assert "sizing" not in plain.record.evaluation
    assert not (plain.record_path.parent / "sizing.csv").exists()

    block = sized.record.evaluation["sizing"]
    assert block["rule"]["lookback_bars"] == 10 and block["plans"] == [f"trailing/t{j:02d}" for j in range(5)]
    sessions_frame = pd.read_csv(block["sessions_path"], index_col="ts")
    decisions = pd.read_csv(block["decisions_path"])
    assert block["n_decisions"] == len(decisions) > 0
    assert block["status_counts"]["ok"] > 0 and block["status_counts"]["warmup"] > 0
    # Every scale is the rule's: capped, and never below zero.
    assert (decisions["scale"] > 0.0).all() and (decisions["scale"] <= 2.0).all()
    assert (decisions["scale"] > 1.0).any(), "the rule levers the book up somewhere, so the cap is exercised"
    assert sessions_frame["target_gross"].max() <= 2.0 + 1e-12
    assert block["scored_sessions"]["target_gross_max"] <= 2.0 + 1e-12
    # The ceiling bounds each decision's target; the held book drifts between refreshes and is recorded as
    # it is (on this synthetic panel it closes above 2.0 on some sessions).
    assert sessions_frame["gross"].notna().all() and sessions_frame["gross"].max() > 1.0
    assert {"gross", "long_over_equity", "sleeve_fraction", "scale"} <= set(sessions_frame.columns)
    assert not sized.scored.returns.equals(plain.scored.returns)
    json.dumps(block)  # plain JSON all the way down


def test_the_ceiling_binds_when_the_cap_would_pass_it(engine_root, tmp_path) -> None:
    sized = _run(engine_root, tmp_path, "ceiling", SIZED.format(target=0.9, cap=4.0, extra="max_gross = 1.5\n"))
    block = sized.record.evaluation["sizing"]
    sessions_frame = pd.read_csv(block["sessions_path"], index_col="ts")
    assert (sessions_frame["ceiling_factor"] < 1.0).any()
    assert sessions_frame["target_gross"].max() <= 1.5 + 1e-12
    assert any("ceiling" in warning for warning in sized.record.warnings)


def test_a_rule_pinned_at_scale_one_is_the_unscaled_book_bit_for_bit(engine_root, plain, tmp_path) -> None:
    pinned = _run(engine_root, tmp_path, "pinned", SIZED.format(target=1e6, cap=1.0, extra=""))
    pd.testing.assert_series_equal(pinned.scored.returns, plain.scored.returns, check_exact=True)
    assert pinned.record.evaluation["engine"]["n_fills"] == plain.record.evaluation["engine"]["n_fills"]


def test_the_paths_that_would_ignore_sizing_refuse_it(engine_root, tmp_path) -> None:
    spec, workspace = _spec(engine_root, tmp_path, "refused", SIZED.format(target=0.6, cap=2.0, extra=""))
    with pytest.raises(RunRefused, match=r"\[sizing\]"):
        walk_forward_evaluate(
            spec,
            workspace=workspace,
            splitter=SPLITTER,
            fold_table_id="sizing_unit",
            engine="canonical",
            canonical=_canonical_settings(),
            decision_coupling="targets_only",
            capital=100_000.0,
        )
    assert not (workspace.root / "trials").exists() or not any((workspace.root / "trials").iterdir())
    untranched, workspace = _spec(engine_root, tmp_path, "smoke", SIZED.format(target=0.6, cap=2.0, extra=""), grid="")
    with pytest.raises(EvaluationError, match=r"\[sizing\]"):
        smoke_evaluate(untranched, workspace=workspace)
