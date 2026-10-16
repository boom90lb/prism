"""Three diagnostic variants: a split open/close fill model, a liquidity-weighted decile rule, seeded partitions.

Pinned here:

* ``fills.model = "split_open_close"`` fills half of each order at the next open and the rest
  at that session's close (N2), so a continuous-grid order's average fill price is the mean of
  the open and the close; whole shares round the open leg half to even; both legs pay the
  bucket half spread and together pay the decided order's impact once; equity still moves only
  by PnL and charged cost (N4); a bar that traded without a close expires the remainder; the
  declared default policy and its identity are unchanged;
* ``to_weights = "decile_neutral_sqrt_dv"`` holds the decile rule's names, weights each leg
  by the square root of trailing median dollar volume, sums each leg to 0.5 under the
  per-name cap, refuses a rankable name with no dollar volume, and leaves every other rule
  untouched;
* ``research.scripts.universe_partition.partition`` is a deterministic per-name hash that
  keeps about ``keep`` of any universe;
* both engine variants run end to end through ``walk_forward_evaluate``.

Synthetic data only.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import hmac
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prism.evaluate as evaluate_module
from prism.composition import ComponentPlan, compose
from prism.evaluate import CanonicalSettings, walk_forward_evaluate
from prism.execution.cost_model import CostInputs, CostModel, CostStack
from prism.execution.portfolio_state import AccountingError
from prism.execution.simulate import SPLIT_OPEN_SHARE, FillPolicy, simulate
from prism.io.synthetic import write_synthetic_dataset
from prism.portfolio.construct import (
    _decile_row,
    capped_proportional,
    liquidity_weighted_decile_row,
)
from prism.signal.base import Signal
from prism.spec import (
    DOLLAR_VOLUME_RULES,
    TO_WEIGHTS_RULES,
    SpecError,
    apply_to_weights,
    load_spec,
    resolve_to_weights_params,
    rule_dollar_volume,
)
from prism.validation.trials import RecordSchemaError, validate_result_record
from prism.validation.walk_forward import RollingWalkForward
from prism.workspace import Workspace
from research.scripts.universe_partition import MAX_SEED, draw, partition
from tests.test_accounting_conservation import (
    BARS,
    _panels,
    _reconstruct_equity_path,
    _targets,
    cost_model,
    cost_stack,
)

SPLIT = FillPolicy(model="split_open_close")
BUCKETS = [[500_000_000.0, 1.0], [100_000_000.0, 2.0], [25_000_000.0, 5.0], [0.0, 10.0]]


# ---------------------------------------------------------------------------
# split_open_close: the fill model
# ---------------------------------------------------------------------------


def _one_name(opens: list[float], closes: list[float], volumes: list[float], target: float, cash: float = 10_000.0):
    bars = BARS[: len(opens)]
    frame = lambda values: pd.DataFrame({"AAA": values}, index=bars)  # noqa: E731
    targets = pd.DataFrame(np.nan, index=bars, columns=["AAA"])
    targets.loc[bars[0]] = [target]
    return simulate(
        frame(opens),
        frame(closes),
        targets,
        cost_model=cost_model(equity=cash),
        volumes=frame(volumes),
        initial_cash=cash,
        fills=SPLIT,
    )


def test_an_order_fills_half_at_the_next_open_and_the_rest_at_that_close() -> None:
    result = _one_name([100.0, 101.0, 102.0], [100.0, 103.0, 102.0], [1e6] * 3, target=0.40)
    fills = result.fills
    assert list(fills["bar"]) == [BARS[1], BARS[1]], "both legs on t+1, none on the decision bar (N2)"
    assert list(fills["qty"]) == [20.0, 20.0]
    assert list(fills["fill_price"]) == [101.0, 103.0], "the open, then the close the bar printed"
    assert fills["execution_id"].iloc[1].endswith(":close") and not fills["execution_id"].iloc[0].endswith(":close")
    average = float((fills["qty"] * fills["fill_price"]).sum() / fills["qty"].sum())
    assert average == pytest.approx((101.0 + 103.0) / 2.0, abs=0.0)
    assert [order.status for order in result.orders] == ["filled"]
    assert result.ledger.loc[BARS[1], "n_fills"] == 2
    assert result.shares.loc[BARS[1], "AAA"] == 40.0


def test_the_average_fill_is_the_mean_of_open_and_close_on_a_continuous_grid_and_equity_conserves() -> None:
    opens, closes, volumes = _panels()
    result = simulate(
        opens,
        closes,
        _targets(),
        cost_model=cost_model(),
        volumes=volumes,
        initial_cash=100_000.0,
        fills=SPLIT,
        increment=0.0,
    )
    fills = result.fills
    assert len(fills) == 2 * len({order.client_order_id for order in result.orders}), "two legs per order"
    for (client_order_id, bar), legs in fills.groupby(["client_order_id", "bar"]):
        sid = legs["sid"].iloc[0]
        average = float((legs["qty"] * legs["fill_price"]).sum() / legs["qty"].sum())
        assert average == pytest.approx((opens.loc[bar, sid] + closes.loc[bar, sid]) / 2.0, rel=1e-12)
        assert legs["qty"].iloc[0] == pytest.approx(SPLIT_OPEN_SHARE * legs["qty"].sum(), rel=1e-12)
    assert all(order.status == "filled" for order in result.orders)
    pd.testing.assert_series_equal(
        result.ledger["equity_close"], _reconstruct_equity_path(result), rtol=0.0, atol=1e-9
    )


def test_whole_shares_round_the_open_leg_half_to_even_and_the_close_takes_the_rest() -> None:
    three = _one_name([100.0] * 3, [100.0] * 3, [1e6] * 3, target=0.03)
    assert list(three.fills["qty"]) == [2.0, 1.0], "round(1.5) = 2 at the open, 1 at the close"
    one = _one_name([100.0] * 3, [100.0] * 3, [1e6] * 3, target=0.01)
    assert list(one.fills["qty"]) == [1.0], "round(0.5) = 0 at the open, so the one share fills at the close"
    assert one.fills["execution_id"].iloc[0].endswith(":close")


def test_both_legs_pay_the_bucket_half_spread_and_the_order_pays_its_impact_once() -> None:
    opens, closes, volumes = _panels()
    stack = cost_stack(spread={"mode": "bucket", "schedule": BUCKETS})
    liquidity = pd.Series({"AAA": 2.0e8, "BBB": 3.0e7, "CCC": 1.0e6})
    bps = {"AAA": 2.0, "BBB": 5.0, "CCC": 10.0}

    def run(policy: FillPolicy):
        model = CostModel(stack, CostInputs(as_of=BARS[0], adv_bucket=liquidity), equity=100_000.0, horizon_bars=5)
        return simulate(
            opens, closes, _targets(), cost_model=model, volumes=volumes, initial_cash=100_000.0, fills=policy
        )

    split, whole = run(SPLIT), run(FillPolicy())
    for fill in split.fills.itertuples():
        assert fill.spread_dollars == pytest.approx(bps[fill.sid] / 1e4 * abs(fill.qty) * fill.fill_price, rel=1e-12)
    assert split.fills["spread_dollars"].gt(0.0).all(), "the close leg is charged, not free"
    # The first decision is the same in both runs, so its orders' impact must match across the legs.
    first = BARS[1]
    split_impact = split.fills[split.fills["bar"] == first].groupby("client_order_id")["impact_dollars"].sum()
    whole_impact = whole.fills[whole.fills["bar"] == first].set_index("client_order_id")["impact_dollars"]
    pd.testing.assert_series_equal(split_impact.sort_index(), whole_impact.sort_index(), check_names=False, rtol=1e-12)
    assert float(split_impact.sum()) > 0.0
    pd.testing.assert_series_equal(split.ledger["equity_close"], _reconstruct_equity_path(split), rtol=0.0, atol=1e-9)


def test_a_bar_that_traded_without_a_close_expires_the_remainder_on_that_bar() -> None:
    result = _one_name([100.0, 101.0, 102.0], [100.0, np.nan, 102.0], [1e6] * 3, target=0.40)
    assert list(result.fills["qty"]) == [20.0] and list(result.fills["fill_price"]) == [101.0]
    (order,) = result.orders
    assert order.status == "expired" and order.last_update == BARS[1]
    assert "printed no close" in order.reason
    assert result.ledger.loc[BARS[2], "pending_orders"] == 0, "nothing carries to the next bar"


def test_a_bar_that_did_not_trade_leaves_the_order_for_the_next_one() -> None:
    result = _one_name([100.0, np.nan, 102.0, 102.0], [100.0, np.nan, 104.0, 104.0], [1e6, np.nan, 1e6, 1e6], 0.40)
    assert list(result.fills["bar"]) == [BARS[2], BARS[2]]
    assert list(result.fills["fill_price"]) == [102.0, 104.0]


def test_the_split_model_refuses_settings_it_gives_no_meaning() -> None:
    with pytest.raises(AccountingError, match="split_open_close"):
        FillPolicy(model="split_open_close", partial_fill_fraction=0.5)
    with pytest.raises(AccountingError, match="split_open_close"):
        FillPolicy(model="split_open_close", max_participation_of_bar_volume=0.01)
    assert FillPolicy.from_mapping({"model": "split_open_close"}) == SPLIT


def test_the_default_policy_and_its_identity_are_unchanged() -> None:
    assert dataclasses.asdict(FillPolicy()) == {
        "model": "full_next_open",
        "max_participation_of_bar_volume": 0.0,
        "partial_fill_fraction": 1.0,
        "reject_probability": 0.0,
        "missing_open_policy": "reference_price",
        "seed": 0,
    }
    stack = cost_stack()
    default = CanonicalSettings(cost_stack=stack).identity()
    split = CanonicalSettings(cost_stack=stack, fills=SPLIT).identity()
    assert default["fills"]["model"] == "full_next_open" and split["fills"]["model"] == "split_open_close"
    assert {k: v for k, v in default.items() if k != "fills"} == {k: v for k, v in split.items() if k != "fills"}


def test_the_engine_note_states_the_split_model_only_when_it_was_declared() -> None:
    plain = {"decision_coupling": "account_state", "settings": {"fills": dataclasses.asdict(FillPolicy())}}
    split = {"decision_coupling": "account_state", "settings": {"fills": dataclasses.asdict(SPLIT)}}
    assert evaluate_module._engine_note("canonical", plain) == evaluate_module.COUPLED_NOTE
    assert (
        evaluate_module._engine_note("canonical", split)
        == evaluate_module.COUPLED_NOTE + evaluate_module.SPLIT_FILLS_NOTE
    )
    assert evaluate_module._fills_note("x", FillPolicy()) == "x"


# ---------------------------------------------------------------------------
# decile_neutral_sqrt_dv: the rule
# ---------------------------------------------------------------------------


def test_capped_proportional_sums_to_the_total_under_the_cap_and_keeps_proportions() -> None:
    raw = np.array([100.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    weights = capped_proportional(raw, 0.5, 0.10)
    assert weights.sum() == pytest.approx(0.5, abs=1e-15)
    assert weights.max() <= 0.10 + 1e-15 and weights[0] == 0.10
    free = weights < 0.10
    np.testing.assert_allclose(weights[free] / weights[free].sum(), raw[free] / raw[free].sum(), rtol=1e-14)
    np.testing.assert_array_equal(capped_proportional(np.ones(4), 0.5, 0.10), np.full(4, 0.10))
    assert capped_proportional(np.zeros(0), 0.5, 0.10).shape == (0,)
    with pytest.raises(ValueError):
        capped_proportional(np.array([1.0, 0.0]), 0.5, 0.10)


def _cross_section(n: int = 120, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n)
    liquidity = np.exp(rng.normal(19.0, 1.2, size=n))
    return scores, liquidity


def test_the_row_holds_the_decile_rules_names_and_weights_each_leg_by_root_liquidity() -> None:
    scores, liquidity = _cross_section()
    scores[[5, 9]] = np.nan
    liquidity[int(np.nanargmax(scores))] = 1e15  # one leg member large enough to bind the cap
    row = liquidity_weighted_decile_row(scores, liquidity, 0.10, leg_gross=0.5, max_symbol_abs_weight=0.10)
    np.testing.assert_array_equal(np.sign(row), np.sign(_decile_row(scores, 0.10)))
    assert row[row > 0].sum() == pytest.approx(0.5, abs=1e-15) and row[row < 0].sum() == pytest.approx(-0.5, abs=1e-15)
    assert np.abs(row).max() == pytest.approx(0.10, abs=1e-15), "the cap binds and holds"
    for leg in (row > 0, row < 0):
        free = leg & (np.abs(row) < 0.10)
        ratio = np.abs(row[free]) / np.sqrt(liquidity[free])
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
    liquidity[np.flatnonzero(row < 0)[0]] = np.nan
    with pytest.raises(ValueError, match="liquidity"):
        liquidity_weighted_decile_row(scores, liquidity, 0.10)


def _dollar_volume(n_bars: int, sids: list[str], seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(19.0, 1.0, size=len(sids)))
    noise = np.exp(0.2 * rng.standard_normal((n_bars, len(sids))))
    index = pd.bdate_range("2026-01-02", periods=n_bars)
    return pd.DataFrame(base * noise, index=index, columns=sids)


def test_the_spec_rule_uses_the_trailing_median_and_matches_the_construct_row() -> None:
    sids = [f"S{i:03d}" for i in range(100)]
    scores = pd.Series(np.random.default_rng(2).normal(size=100), index=sids, name="2026-02-27")
    panel = _dollar_volume(30, sids)
    params = {"decile": 0.10, "max_symbol_abs_weight": 0.10}
    weights = apply_to_weights("decile_neutral_sqrt_dv", scores, params, dollar_volume=panel)
    median = panel.tail(20).median(axis=0, skipna=False)
    expected = liquidity_weighted_decile_row(
        scores.to_numpy(), median.to_numpy(), 0.10, leg_gross=0.5, max_symbol_abs_weight=0.10
    )
    np.testing.assert_array_equal(weights.to_numpy(), expected)
    assert weights.name == scores.name and list(weights.index) == sids
    assert weights[weights > 0].sum() == pytest.approx(0.5) and weights[weights < 0].sum() == pytest.approx(-0.5)
    assert weights.abs().max() <= 0.10 + 1e-15
    plain = apply_to_weights("decile_long_short", scores, {"decile": 0.10})
    assert set(weights.index[weights != 0]) == set(plain.index[plain != 0]), "same names as the decile rule"
    # Only the last 20 bars count: a bar before them changes nothing, a bar inside them does.
    early = panel.copy()
    early.iloc[0] *= 50.0
    pd.testing.assert_series_equal(
        apply_to_weights("decile_neutral_sqrt_dv", scores, params, dollar_volume=early), weights
    )
    late = panel.copy()
    late.iloc[-10:, :50] *= 50.0
    assert not apply_to_weights("decile_neutral_sqrt_dv", scores, params, dollar_volume=late).equals(weights)
    capped_gross = apply_to_weights(
        "decile_neutral_sqrt_dv", scores, {**params, "max_gross": 0.6}, dollar_volume=panel
    )
    assert capped_gross[capped_gross > 0].sum() == pytest.approx(0.3)


def test_the_spec_rule_refuses_a_rankable_name_without_dollar_volume_and_a_missing_panel() -> None:
    sids = [f"S{i:03d}" for i in range(40)]
    scores = pd.Series(np.arange(40, dtype=float), index=sids, name="2026-02-27")
    panel = _dollar_volume(25, sids)
    with pytest.raises(SpecError, match="dollar-volume panel"):
        apply_to_weights("decile_neutral_sqrt_dv", scores, {"decile": 0.10})
    with pytest.raises(SpecError, match="fewer than dollar_volume_window"):
        apply_to_weights("decile_neutral_sqrt_dv", scores, {"decile": 0.10}, dollar_volume=panel.tail(5))
    gappy = panel.copy()
    gappy.iloc[-3, 7] = np.nan
    with pytest.raises(SpecError, match="S007"):
        apply_to_weights("decile_neutral_sqrt_dv", scores, {"decile": 0.10}, dollar_volume=gappy)
    unranked = scores.copy()
    unranked["S007"] = np.nan  # a screened-out name: no score, so its missing dollar volume is never needed
    weights = apply_to_weights("decile_neutral_sqrt_dv", unranked, {"decile": 0.10}, dollar_volume=gappy)
    assert weights["S007"] == 0.0


def test_the_rules_parameters_resolve_with_their_defaults_and_bounds() -> None:
    assert "decile_neutral_sqrt_dv" in TO_WEIGHTS_RULES and DOLLAR_VOLUME_RULES == {"decile_neutral_sqrt_dv"}
    assert resolve_to_weights_params("decile_neutral_sqrt_dv", {"decile": 0.1}) == {
        "decile": 0.1,
        "dollar_volume_window": 20,
        "max_gross": 1.0,
        "max_symbol_abs_weight": 1.0,
    }
    assert (
        resolve_to_weights_params("decile_neutral_sqrt_dv", {"decile": 0.1, "dollar_volume_window": 1})[
            "dollar_volume_window"
        ]
        == 1
    )
    for bad in (
        {},
        {"decile": 0.6},
        {"decile": 0.1, "dollar_volume_window": 0},
        {"decile": 0.1, "hold_quantile": 0.2},
    ):
        with pytest.raises(SpecError):
            resolve_to_weights_params("decile_neutral_sqrt_dv", bad)
    # The existing rules' integer bounds did not move.
    with pytest.raises(SpecError):
        resolve_to_weights_params("sign_inverse_vol", {"vol_ewma_bars": 1})
    assert resolve_to_weights_params("top_k_equal_weight", {"k": 1})["k"] == 1


def test_rule_dollar_volume_reads_nothing_for_other_rules_and_the_screens_product_for_this_one() -> None:
    for rule in TO_WEIGHTS_RULES:
        if rule not in DOLLAR_VOLUME_RULES:
            assert rule_dollar_volume(rule, {}, None, None) is None
    sids = ["A", "B", "C"]
    index = pd.bdate_range("2026-01-02", periods=30)
    price = pd.DataFrame(np.linspace(10.0, 20.0, 90).reshape(30, 3), index=index, columns=sids)
    volume = pd.DataFrame(1e5, index=index, columns=sids)
    got = rule_dollar_volume("decile_neutral_sqrt_dv", {"decile": 0.2, "dollar_volume_window": 7}, price, volume)
    pd.testing.assert_frame_equal(got, (price * volume).tail(7))
    with pytest.raises(SpecError, match="price and volume"):
        rule_dollar_volume("decile_neutral_sqrt_dv", {"decile": 0.2}, price, None)


class _Momentum(Signal):
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


def test_composition_feeds_the_rule_the_price_panels_dollar_volume() -> None:
    sids = [f"S{i:02d}" for i in range(30)]
    rng = np.random.default_rng(5)
    index = pd.bdate_range("2026-01-02", periods=40)
    price = pd.DataFrame(
        50.0 * np.cumprod(1 + 0.01 * rng.standard_normal((40, 30)), axis=0), index=index, columns=sids
    )
    volume = pd.DataFrame(np.exp(rng.normal(13.0, 1.0, size=(40, 30))), index=index, columns=sids)
    params = {"decile": 0.2, "max_symbol_abs_weight": 0.2}
    plan = ComponentPlan(
        component=_Momentum(),
        component_id="m",
        to_weights="decile_neutral_sqrt_dv",
        to_weights_params=params,
        update_every_bars=1,
        valid_for_bars=1,
    )
    decision = str(index[-1].date())
    result = compose([plan], decision_bar=decision, scoring_panels={"price": price}, volume=volume)
    expected = apply_to_weights(
        "decile_neutral_sqrt_dv",
        _Momentum().score(price),
        params,
        dollar_volume=rule_dollar_volume("decile_neutral_sqrt_dv", params, price, volume),
    )
    pd.testing.assert_series_equal(result.targets.reindex(sids), expected.reindex(sids), check_names=False)
    with pytest.raises(SpecError, match="price and volume"):
        compose([plan], decision_bar=decision, scoring_panels={"price": price})


# ---------------------------------------------------------------------------
# research.scripts.universe_partition
# ---------------------------------------------------------------------------

UNIVERSE = [f"SID{i:05d}" for i in range(10_000)]


def test_a_partition_is_a_keyed_hash_of_the_security_id() -> None:
    digest = hmac.new((12345).to_bytes(8, "big"), b"SID00042", hashlib.sha256).digest()
    assert draw("SID00042", 12345) == int.from_bytes(digest[:8], "big") / 2.0**64
    assert draw("AAPL", 1) == 0.503195952010735, "the definition is pinned: key, message and byte order"


@pytest.mark.parametrize("seed", [1, 2**40 + 7, MAX_SEED - 1])
def test_a_partition_is_deterministic_and_keeps_about_the_declared_share(seed: int) -> None:
    kept = partition(UNIVERSE, seed)
    assert kept == partition(UNIVERSE, seed) == partition(list(UNIVERSE), seed, keep=0.8)
    assert abs(len(kept) / len(UNIVERSE) - 0.8) < 0.02  # binomial sd 0.004 at n = 10,000
    assert list(kept) == [sid for sid in UNIVERSE if sid in set(kept)], "input order is kept"
    assert partition(UNIVERSE, seed, keep=1.0) == tuple(UNIVERSE)


def test_a_names_fate_does_not_depend_on_the_rest_of_the_universe_and_seeds_differ() -> None:
    half = UNIVERSE[::2]
    assert partition(half, 99) == tuple(sid for sid in partition(UNIVERSE, 99) if sid in set(half))
    a, b = set(partition(UNIVERSE, 1)), set(partition(UNIVERSE, 2))
    jaccard = len(a & b) / len(a | b)
    assert 0.62 < jaccard < 0.71, "independent 80% draws overlap at about 0.64 / 0.96"


@pytest.mark.parametrize(
    "call",
    [
        lambda: partition(UNIVERSE[:5], -1),
        lambda: partition(UNIVERSE[:5], MAX_SEED),
        lambda: partition(UNIVERSE[:5], True),
        lambda: partition(UNIVERSE[:5], 1.0),
        lambda: partition(UNIVERSE[:5], 1, keep=0.0),
        lambda: partition(UNIVERSE[:5], 1, keep=1.5),
        lambda: partition(["A", "A"], 1),
        lambda: partition(["A", ""], 1),
        lambda: partition([], -5),
    ],
)
def test_a_partition_refuses_bad_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()


# ---------------------------------------------------------------------------
# Both engine variants, end to end (the scaffolding of tests/test_vol_scale.py)
# ---------------------------------------------------------------------------

PACKAGE = "variant_strategies"
SPLITTER = RollingWalkForward(formation_bars=60, test_bars=40, min_test_bars=20)

TRAILING_SIGNAL = """
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
"""

BOOK_TOML = """
[strategy]
name = "variant_trailing_book"
universe_index = "synthetic_100"
combination = "position_space"

[strategy.eligibility]
min_median_dollar_volume = 1.0
dollar_volume_window = 20

[[strategy.components]]
id = "trailing"
path = "variant_strategies.trailing:TrailingReturn"
weight = 1.0
horizon_bars = 5
update_every_bars = 5
valid_for_bars = 10
to_weights = "{rule}"
tranches = 5

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
    root = tmp_path_factory.mktemp("variants") / "source"
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


def _canonical_settings(fills: FillPolicy = FillPolicy()) -> CanonicalSettings:
    volume = {
        "estimator": "median",
        "window": "trailing",
        "window_bars": 20,
        "refresh": "decision",
        "floor_dollars": 100_000.0,
    }
    stack = CostStack.from_mapping(
        {
            "spread": {"mode": "flat", "flat_bps": 2.0},
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
    return CanonicalSettings(cost_stack=stack, fills=fills)


def _run(engine_root: Path, tmp_path: Path, name: str, *, rule: str = "decile_long_short", fills=FillPolicy()):
    root = tmp_path / name
    root.mkdir()
    for entry in sorted(engine_root.iterdir()):
        if entry.name not in {"results", "trials", "state", "runs", "logs", "specs"}:
            (root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    workspace = Workspace(root).ensure()
    (root / "specs").mkdir()
    (root / "specs" / "book.toml").write_text(BOOK_TOML.format(rule=rule), encoding="utf-8")
    spec = load_spec(root / "specs" / "book.toml", workspace=workspace)
    return walk_forward_evaluate(
        spec,
        workspace=workspace,
        splitter=SPLITTER,
        fold_table_id="variants_unit",
        selection_set_id="unit_set",
        registration_id="unit_set",
        kind="new",
        engine="canonical",
        canonical=_canonical_settings(fills),
        decision_coupling="account_state",
        fold_portfolio="carry",
        capital=100_000.0,
    )


@pytest.fixture(scope="module")
def plain(engine_root, tmp_path_factory):
    """The tranched decile book with next-open fills, run once for both comparisons."""
    return _run(engine_root, tmp_path_factory.mktemp("plain"), "plain")


def test_an_engine_run_under_split_fills_trades_both_auctions_and_says_so(engine_root, plain, tmp_path) -> None:
    split = _run(engine_root, tmp_path, "split", fills=SPLIT)
    assert split.config_id != plain.config_id
    engine = split.record.evaluation["engine"]
    assert engine["settings"]["fills"]["model"] == "split_open_close"
    assert engine["note"].endswith(evaluate_module.SPLIT_FILLS_NOTE)
    assert evaluate_module.SPLIT_FILLS_NOTE.strip() in " ".join(split.record.warnings)
    assert engine["n_fills"] > plain.record.evaluation["engine"]["n_fills"], "most orders now fill in two legs"
    assert not split.scored.returns.equals(plain.scored.returns)
    assert evaluate_module.SPLIT_FILLS_NOTE.strip() not in " ".join(plain.record.warnings)

    # The record's return convention names the split model, and the validator holds the label to the model.
    assert split.record.conventions["executable_return"] == "split_open_close_fill_marked_close_to_close"
    assert plain.record.conventions["executable_return"] == "next_open_fill_marked_close_to_close"
    for result, other in ((split, plain), (plain, split)):
        payload = json.loads(result.record_path.read_text(encoding="utf-8"))
        validate_result_record(payload)
        payload["conventions"]["executable_return"] = other.record.conventions["executable_return"]
        with pytest.raises(RecordSchemaError, match="conventions.executable_return"):
            validate_result_record(payload)


def test_an_engine_run_under_the_liquidity_weighted_rule_reweights_the_book(engine_root, plain, tmp_path) -> None:
    liquid = _run(engine_root, tmp_path, "liquid", rule="decile_neutral_sqrt_dv")
    assert liquid.config_id != plain.config_id
    assert not liquid.full.target_weights.equals(plain.full.target_weights)
    assert not liquid.scored.returns.equals(plain.scored.returns)
    assert liquid.full.costs["gross"].max() > 0.0
