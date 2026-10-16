"""Tests for capacity + cost-toll diagnostics (SPEC.md §3 law 5, §10) and crowding re-pricing."""

import math
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.config import ExecutionConfig
from prism.execution.cost_model import CostInputs, CostModel, CostStack
from prism.execution.portfolio_state import Fill
from prism.execution.target_weights import backtest_target_weights
from prism.validation.capacity import (
    CrowdScenario,
    breakeven_follower_capital,
    capacity_curve,
    cost_toll,
    crowd_capacity_table,
    crowd_impact_cost,
)


def _panel(n=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    syms = ["A", "B", "C"]
    prices = pd.DataFrame(
        100 * np.cumprod(1 + 0.001 * rng.standard_normal((n, 3)), axis=0), index=dates, columns=syms
    )
    tw = pd.DataFrame(0.2 * np.sign(rng.standard_normal((n, 3))), index=dates, columns=syms)
    dvol = pd.DataFrame(1e6, index=dates, columns=syms)
    return prices, tw, dvol


class TestCapacityCurve:
    def test_cost_monotone_increasing_in_aum(self):
        prices, tw, dvol = _panel()
        exe = ExecutionConfig(adv_impact_coeff=10.0, adv_impact_model="sqrt", adv_floor_dollars=1.0)
        cc = capacity_curve(prices, tw, dvol, [1e3, 1e5, 1e7, 1e9], execution=exe)
        costs = cc["total_cost"].to_numpy()
        assert np.all(np.diff(costs) >= -1e-9)  # non-decreasing in AUM

    def test_gross_sharpe_invariant_to_aum(self):
        # AUM scales only the sqrt-ADV impact; gross (pre-cost) is unchanged up to
        # float reconstruction round-off (gross = net + cost re-adds the AUM term).
        prices, tw, dvol = _panel()
        exe = ExecutionConfig(adv_impact_coeff=10.0, adv_impact_model="sqrt", adv_floor_dollars=1.0)
        cc = capacity_curve(prices, tw, dvol, [1e3, 1e9], execution=exe)
        g = cc["gross_sharpe_ann"].to_numpy()
        assert g[0] == pytest.approx(g[1], rel=1e-9)

    def test_flat_when_impact_off(self):
        # adv_impact_coeff=0 -> AUM does not enter cost -> net flat.
        prices, tw, dvol = _panel()
        exe = ExecutionConfig(adv_impact_coeff=0.0)
        cc = capacity_curve(prices, tw, dvol, [1e3, 1e9], execution=exe)
        assert cc["net_sharpe_ann"].nunique() == 1

    def test_index_sorted_and_named(self):
        prices, tw, dvol = _panel()
        cc = capacity_curve(prices, tw, dvol, [1e7, 1e3, 1e5])
        assert list(cc.index) == sorted(cc.index)
        assert cc.index.name == "aum"

    def test_rejects_bad_aum(self):
        prices, tw, dvol = _panel()
        with pytest.raises(ValueError):
            capacity_curve(prices, tw, dvol, [])
        with pytest.raises(ValueError):
            capacity_curve(prices, tw, dvol, [1e3, -1.0])


class TestCostToll:
    def test_reconstructs_gross_and_reports_toll(self):
        prices, tw, dvol = _panel()
        exe = ExecutionConfig(adv_impact_coeff=10.0, adv_impact_model="sqrt", adv_floor_dollars=1.0)
        res = backtest_target_weights(prices, tw, execution=exe, initial_capital=1e7, dollar_volume=dvol)
        toll = cost_toll(res.returns, res.costs)
        assert 0.0 <= toll["toll_booth_fraction"] <= 1.0
        assert toll["cost_to_gross"] >= 0.0
        assert toll["total_cost"] == pytest.approx(res.costs["total"].sum())

    def test_zero_cost_gives_zero_toll(self):
        prices, tw, dvol = _panel()
        exe = ExecutionConfig(commission_bps=0.0, spread_bps=0.0, slippage_coeff=0.0, borrow_rate_bps_annual=0.0)
        res = backtest_target_weights(prices, tw, execution=exe, initial_capital=1.0)
        toll = cost_toll(res.returns, res.costs)
        assert toll["total_cost"] == pytest.approx(0.0)
        assert toll["toll_booth_fraction"] == pytest.approx(0.0)

    def test_empty_input_returns_nan_fields(self):
        toll = cost_toll(pd.Series(dtype=float), pd.DataFrame())
        assert np.isnan(toll["toll_booth_fraction"])
        assert np.isnan(toll["cost_to_gross"])


# ---------------------------------------------------------------------------
# Crowding re-pricing
# ---------------------------------------------------------------------------

RETAIL_V3 = Path(__file__).resolve().parents[1] / "examples" / "reference" / "costs" / "retail_v3.toml"


def _crowd_panel(n_bars=40, seed=1):
    """Own order notional on a few names, with gaps, plus matching volume and volatility panels."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-02", periods=n_bars, freq="B")
    names = ["AAA", "BBB", "CCC", "DDD"]
    trades = pd.DataFrame(rng.uniform(-3_000.0, 3_000.0, (n_bars, 4)), index=index, columns=names)
    trades = trades.where(rng.uniform(size=(n_bars, 4)) < 0.6, 0.0)
    trades.iloc[0, 0] = 2_500.0
    volume = pd.DataFrame(rng.uniform(5e7, 5e8, (n_bars, 4)), index=index, columns=names)
    sigma = pd.DataFrame(rng.uniform(0.01, 0.03, (n_bars, 4)), index=index, columns=names)
    return trades, volume, sigma


class TestCrowdScenario:
    def test_window_factor(self):
        assert CrowdScenario().window_factor == 1.0
        assert CrowdScenario(windows=((1.0, 0.01),)).window_factor == pytest.approx(10.0, rel=1e-15)
        split = CrowdScenario(windows=((0.5, 0.01), (0.5, 0.08)))
        assert split.window_factor == pytest.approx(0.5**1.5 * (10.0 + 1.0 / math.sqrt(0.08)), rel=1e-15)
        assert split.describe_windows() == "0.5@0.01+0.5@0.08"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"follower_capital": -1.0},
            {"follower_capital": float("nan")},
            {"synchronized_fraction": 0.0},
            {"synchronized_fraction": 1.5},
            {"impact_y": 0.0},
            {"windows": ()},
            {"windows": ((0.5, 0.01),)},
            {"windows": ((1.0, 0.0),)},
            {"windows": ((1.0, 1.5),)},
            {"windows": ((1.0,),)},
        ],
    )
    def test_refuses_an_undefined_scenario(self, kwargs):
        with pytest.raises(ValueError):
            CrowdScenario(**kwargs)


class TestCrowdImpactCost:
    def test_with_no_followers_it_is_the_charged_participation_term(self):
        """F = 0, one window at the full day's volume: CostModel.charge_trade's sqrt term, name by name.

        Y * sigma is set to the stack's coefficient, and the floor to the stack's impact floor;
        one name trades below the floor so the floor is exercised.
        """
        stack = CostStack.from_mapping(tomllib.loads(RETAIL_V3.read_text(encoding="utf-8")))
        equity = 100_000.0
        notional = pd.Series({"AAA": 4_000.0, "BBB": 1_500.0, "CCC": 900.0})
        volume = pd.Series({"AAA": 3.0e8, "BBB": 2.5e7, "CCC": 6.0e4})
        prices = pd.Series({"AAA": 200.0, "BBB": 50.0, "CCC": 12.5})
        model = CostModel(stack, CostInputs(as_of="2026-05-01", adv_impact=volume), equity=equity)
        fills = [
            Fill(
                execution_id=f"sim:{sid}",
                client_order_id=f"2026-05-01:{sid}",
                sid=sid,
                ts="2026-05-04",
                qty=(1.0 if sid != "BBB" else -1.0) * notional[sid] / prices[sid],
                fill_price=float(prices[sid]),
                reference_price=float(prices[sid]),
            )
            for sid in notional.index
        ]
        charged = model.charge_trade(fills).per_name["impact_participation"]

        y = 0.7
        sigma = stack.impact.participation_coeff / 1e4 / y
        bar = pd.Index(["2026-05-04"])
        repriced = crowd_impact_cost(
            pd.DataFrame([notional.to_numpy()], index=bar, columns=notional.index),
            pd.DataFrame([volume.to_numpy()], index=bar, columns=volume.index),
            sigma,
            equity,
            CrowdScenario(follower_capital=0.0, windows=((1.0, 1.0),), impact_y=y),
            floor_dollars=stack.dollar_volume["impact"].floor_dollars,
        ).iloc[0]
        assert charged["CCC"] > 0.0
        for sid in notional.index:
            assert repriced[sid] == pytest.approx(charged[sid], rel=1e-12)

    def test_cost_rises_with_follower_capital(self):
        trades, volume, sigma = _crowd_panel()
        totals = []
        previous = None
        for capital in (0.0, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10):
            cost = crowd_impact_cost(
                trades, volume, sigma, 100_000.0, CrowdScenario(follower_capital=capital, windows=((1.0, 0.01),))
            )
            if previous is not None:
                assert (cost.to_numpy() >= previous.to_numpy()).all()
            previous = cost
            totals.append(float(cost.to_numpy().sum()))
        assert all(later > earlier for earlier, later in zip(totals, totals[1:]))

    def test_uniform_phases_cut_cost_by_root_21_when_follower_flow_dominates(self):
        """Only phi * F enters, and once the followers' flow dwarfs the account's, phi = 1/21 divides cost by √21."""
        trades, volume, sigma = _crowd_panel()
        equity = 100_000.0

        def total(capital, phi):
            scenario = CrowdScenario(follower_capital=capital, synchronized_fraction=phi, windows=((1.0, 0.03),))
            return float(crowd_impact_cost(trades, volume, sigma, equity, scenario).to_numpy().sum())

        assert total(2.1e8, 1.0 / 21.0) == pytest.approx(total(1e7, 1.0), rel=1e-12)
        capital = 1e7
        assert total(capital, 1.0) / total(capital, 1.0 / 21.0) == pytest.approx(
            math.sqrt((1.0 + capital / equity) / (1.0 + capital / (21.0 * equity))), rel=1e-12
        )
        assert total(1e17, 1.0) / total(1e17, 1.0 / 21.0) == pytest.approx(math.sqrt(21.0), rel=1e-9)

    def test_untraded_cells_cost_nothing_and_the_sign_is_ignored(self):
        trades, volume, sigma = _crowd_panel()
        scenario = CrowdScenario(follower_capital=1e7)
        cost = crowd_impact_cost(trades, volume, sigma, 100_000.0, scenario)
        assert (cost.to_numpy()[trades.to_numpy() == 0.0] == 0.0).all()
        flipped = crowd_impact_cost(-trades, volume, sigma, 100_000.0, scenario)
        pd.testing.assert_frame_equal(cost, flipped)

    def test_refuses_inputs_it_cannot_price(self):
        trades, volume, sigma = _crowd_panel()
        scenario = CrowdScenario()
        nan_trades = trades.copy()
        nan_trades.iloc[3, 1] = float("nan")
        with pytest.raises(ValueError, match="pass 0.0"):
            crowd_impact_cost(nan_trades, volume, sigma, 1e5, scenario)
        with pytest.raises(ValueError, match="no non-zero order"):
            crowd_impact_cost(trades * 0.0, volume, sigma, 1e5, scenario)
        missing = volume.drop(columns=["AAA"])
        with pytest.raises(ValueError, match="dollar_volume is nan for 'AAA'"):
            crowd_impact_cost(trades, missing, sigma, 1e5, scenario)
        flat = sigma.copy()
        flat.iloc[0, 0] = 0.0
        with pytest.raises(ValueError, match="sigma is 0.0 for 'AAA'"):
            crowd_impact_cost(trades, volume, flat, 1e5, scenario)
        equity = pd.Series(1e5, index=trades.index).iloc[:-1]
        with pytest.raises(ValueError, match="equity is nan"):
            crowd_impact_cost(trades, volume, sigma, equity, scenario)
        with pytest.raises(ValueError, match="equity"):
            crowd_impact_cost(trades, volume, sigma, 0.0, scenario)
        with pytest.raises(ValueError, match="DataFrame"):
            crowd_impact_cost(trades, volume.iloc[0], sigma, 1e5, scenario)

    def test_a_volume_missing_where_nothing_trades_is_not_needed(self):
        trades, volume, sigma = _crowd_panel()
        holes = volume.mask(trades == 0.0)
        scenario = CrowdScenario(follower_capital=1e6)
        pd.testing.assert_frame_equal(
            crowd_impact_cost(trades, holes, sigma, 1e5, scenario),
            crowd_impact_cost(trades, volume, sigma, 1e5, scenario),
        )


class TestBreakeven:
    # One name, one order of $2,000 on each of 10 bars, constant equity $100,000.
    EQUITY = 100_000.0
    BARS = 10

    def _case(self):
        trades = pd.DataFrame({"AAA": [2_000.0] * self.BARS})
        scenario = CrowdScenario(synchronized_fraction=0.5, windows=((1.0, 0.04),), impact_y=0.5)
        return trades, scenario

    def test_reproduces_a_hand_computed_case(self):
        trades, scenario = self._case()
        # Own cost per bar: 0.5 * 0.02 * 2000 * sqrt(2000 / (1e8 * 0.04)) = 20 * sqrt(5e-4) dollars;
        # over equity, times 252: the annual own impact cost A.
        own_dollars = 0.5 * 0.02 * 2_000.0 * math.sqrt(2_000.0 / (1e8 * 0.04))
        assert own_dollars == pytest.approx(0.447213595499958, rel=1e-12)
        own_annual = own_dollars / self.EQUITY * 252.0
        margin = 0.01
        # A * sqrt(1 + 0.5 * F / E) = margin  =>  F = (E / 0.5) * ((margin / A) ** 2 - 1).
        expected = (self.EQUITY / 0.5) * ((margin / own_annual) ** 2 - 1.0)
        assert expected == pytest.approx(15_547_040.0, rel=1e-6)
        got = breakeven_follower_capital(trades, 1e8, 0.02, self.EQUITY, scenario, margin_annual=margin)
        assert got == pytest.approx(expected, rel=1e-12)

        # With the account's own impact already charged inside the margin, the budget is
        # margin + charged: here the charged term is exactly the own cost.
        expected_net = (self.EQUITY / 0.5) * (((margin + own_annual) / own_annual) ** 2 - 1.0)
        got_net = breakeven_follower_capital(
            trades, 1e8, 0.02, self.EQUITY, scenario, margin_annual=margin, charged_annual=own_annual
        )
        assert got_net == pytest.approx(expected_net, rel=1e-12)

    def test_the_cost_at_break_even_uses_up_the_budget(self):
        trades, volume, sigma = _crowd_panel()
        equity = pd.Series(np.linspace(90_000.0, 130_000.0, len(trades)), index=trades.index)
        scenario = CrowdScenario(synchronized_fraction=1.0 / 21.0, windows=((0.5, 0.01), (0.5, 0.08)))
        margin = 0.008
        capital = breakeven_follower_capital(trades, volume, sigma, equity, scenario, margin_annual=margin)
        at = CrowdScenario(capital, scenario.synchronized_fraction, scenario.windows, scenario.impact_y)
        cost = crowd_impact_cost(trades, volume, sigma, equity, at)
        annual = float((cost.sum(axis=1) / equity).mean()) * 252
        assert annual == pytest.approx(margin, rel=1e-9)

    def test_no_room_when_the_own_impact_already_uses_the_budget(self):
        trades, scenario = self._case()
        assert breakeven_follower_capital(trades, 1e8, 0.02, self.EQUITY, scenario, margin_annual=-0.01) == 0.0
        assert breakeven_follower_capital(trades, 1e8, 0.02, self.EQUITY, scenario, margin_annual=1e-7) == 0.0
        with pytest.raises(ValueError, match="charged_annual"):
            breakeven_follower_capital(
                trades, 1e8, 0.02, self.EQUITY, scenario, margin_annual=0.01, charged_annual=-0.001
            )

    def test_ignores_the_scenario_follower_capital(self):
        trades, scenario = self._case()
        loaded = CrowdScenario(5e9, scenario.synchronized_fraction, scenario.windows, scenario.impact_y)
        assert breakeven_follower_capital(
            trades, 1e8, 0.02, self.EQUITY, loaded, margin_annual=0.01
        ) == breakeven_follower_capital(trades, 1e8, 0.02, self.EQUITY, scenario, margin_annual=0.01)


class TestCrowdCapacityTable:
    def test_rows_agree_with_the_functions(self):
        trades, volume, sigma = _crowd_panel()
        volume.iloc[:, :3] = volume.iloc[:, :3].clip(lower=1.5e8)
        volume.iloc[:, 3] = 6e7  # one thin name throughout
        equity = 100_000.0
        scenarios = [
            CrowdScenario(follower_capital=capital, synchronized_fraction=phi, windows=windows, label=f"{capital:g}")
            for capital in (0.0, 1e8)
            for phi in (1.0, 1.0 / 21.0)
            for windows in (((1.0, 0.01),), ((0.5, 0.01), (0.5, 0.08)))
        ]
        margin, charged = 0.01, 0.0004
        table = crowd_capacity_table(
            trades, volume, sigma, equity, scenarios, margin_annual=margin, charged_annual=charged
        )
        assert len(table) == len(scenarios)
        assert list(table["label"]) == [s.label for s in scenarios]
        for row, scenario in zip(table.itertuples(index=False), scenarios):
            cost = crowd_impact_cost(trades, volume, sigma, equity, scenario)
            annual = float(cost.to_numpy().sum()) / equity / len(trades) * 252
            assert row.cost_annual == pytest.approx(annual, rel=1e-12)
            assert row.crowding_drag_annual == pytest.approx(row.cost_annual - row.own_cost_annual, abs=1e-15)
            assert row.excess_over_hurdle_annual == pytest.approx(margin + charged - row.cost_annual, abs=1e-15)
            breakeven = breakeven_follower_capital(
                trades, volume, sigma, equity, scenario, margin_annual=margin, charged_annual=charged
            )
            assert row.breakeven_follower_capital == pytest.approx(breakeven, rel=1e-12)
            assert row.breakeven_users_at_100000 == pytest.approx(breakeven / 1e5, rel=1e-12)
            assert row.breakeven_users_at_1000000 == pytest.approx(breakeven / 1e6, rel=1e-12)
            thin = float(cost["DDD"].sum()) / float(cost.to_numpy().sum())
            assert row.thin_name_cost_share == pytest.approx(thin, rel=1e-12)
            if scenario.follower_capital == 0.0:
                assert row.crowding_drag_annual == 0.0

    def test_own_capacity_is_where_a_scaled_account_uses_up_the_budget(self):
        trades, volume, sigma = _crowd_panel()
        equity = pd.Series(np.linspace(100_000.0, 110_000.0, len(trades)), index=trades.index)
        scenario = CrowdScenario(windows=((1.0, 0.01),))
        margin = 0.005
        row = crowd_capacity_table(trades, volume, sigma, equity, [scenario], margin_annual=margin).iloc[0]
        k = row["own_capacity_multiple"]
        assert k > 1.0
        assert row["own_capacity_initial_equity"] == pytest.approx(k * 100_000.0, rel=1e-12)
        scaled = crowd_impact_cost(trades * k, volume, sigma, equity * k, scenario)
        assert float((scaled.sum(axis=1) / (equity * k)).mean()) * 252 == pytest.approx(margin, rel=1e-9)

    def test_refuses_an_empty_grid_or_a_bad_account_size(self):
        trades, volume, sigma = _crowd_panel()
        with pytest.raises(ValueError, match="non-empty"):
            crowd_capacity_table(trades, volume, sigma, 1e5, [], margin_annual=0.01)
        with pytest.raises(ValueError, match="users_at"):
            crowd_capacity_table(trades, volume, sigma, 1e5, [CrowdScenario()], margin_annual=0.01, users_at=(0.0,))
