"""Unit tests for the tax-wedge asymmetry arithmetic (research tier).

Synthetic fixtures only — no network, no real run directories. Each spec §2
asymmetry is exercised deterministically against hand-computed values.
"""

import pytest

pytestmark = pytest.mark.research

from research.scripts.tax_wedge import (  # noqa: E402
    capital_engine,
    fifo_wash_replay,
    loss_cap_wedge,
    pil_wedge,
    state_exemption_wedge,
    synthetic_crash_cell,
    wash_wedge,
    worst_year_cell,
)


def fill(symbol: str, day: str, qty: float, price: float) -> dict:
    return {"symbol": symbol, "filled_bar": day, "qty": qty, "fill_price": price}


# ---------------------------------------------------------------------------
# asymmetry 1 — wash-sale window hit vs miss over real lots
# ---------------------------------------------------------------------------


def test_wash_sale_window_hit_defers_the_loss() -> None:
    # Buy, sell at a $200 loss, rebuy 29 days later: inside the 30-day window,
    # so the loss is disallowed and rides the replacement lot still open at end.
    fills = [
        fill("AAA", "2026-01-05", 100.0, 10.0),
        fill("AAA", "2026-02-16", -100.0, 8.0),
        fill("AAA", "2026-03-17", 100.0, 9.0),
    ]
    replay = fifo_wash_replay(fills, wash_window_days=30)
    assert replay["n_loss_events"] == 1
    assert replay["n_wash_events"] == 1
    assert replay["total_disallowed"] == pytest.approx(200.0)
    assert replay["deferred_open_at_end"] == pytest.approx(200.0)
    assert replay["naive_realized_by_year"][2026] == pytest.approx(-200.0)
    # Taxable income shows no loss: it is deferred, not destroyed.
    assert replay["taxable_realized_by_year"][2026] == pytest.approx(0.0)


def test_wash_sale_window_miss_recognizes_the_loss() -> None:
    # Same trades but the rebuy lands 31 days after the loss sale: no wash.
    fills = [
        fill("AAA", "2026-01-05", 100.0, 10.0),
        fill("AAA", "2026-02-16", -100.0, 8.0),
        fill("AAA", "2026-03-19", 100.0, 9.0),
    ]
    replay = fifo_wash_replay(fills, wash_window_days=30)
    assert replay["n_loss_events"] == 1
    assert replay["n_wash_events"] == 0
    assert replay["total_disallowed"] == 0.0
    assert replay["deferred_open_at_end"] == 0.0
    assert replay["taxable_realized_by_year"][2026] == pytest.approx(-200.0)


def test_wash_sale_deferral_surfaces_when_replacement_lot_closes() -> None:
    # The washed loss re-emerges through the replacement lot's basis: bought
    # back at 9, sold at 9 -> raw 0, but the $200 deferral makes it a $200
    # taxable loss, recognized in the replacement's disposal year.
    fills = [
        fill("AAA", "2026-01-05", 100.0, 10.0),
        fill("AAA", "2026-02-16", -100.0, 8.0),
        fill("AAA", "2026-03-17", 100.0, 9.0),
        fill("AAA", "2026-06-01", -100.0, 9.0),
    ]
    replay = fifo_wash_replay(fills, wash_window_days=30)
    assert replay["total_disallowed"] == pytest.approx(200.0)
    assert replay["recognized_deferral_in_sample"] == pytest.approx(200.0)
    assert replay["deferred_open_at_end"] == pytest.approx(0.0)
    # Conservation: with the chain closed, taxable equals naive in total.
    assert sum(replay["taxable_realized_by_year"].values()) == pytest.approx(
        sum(replay["naive_realized_by_year"].values())
    )


def test_wash_sale_applies_to_short_side_reentry() -> None:
    # Short at 100, cover at 110 (a $1000 loss), re-short 20 days later:
    # same-direction re-entry inside the window defers the short loss too.
    fills = [
        fill("BBB", "2026-01-05", -100.0, 100.0),
        fill("BBB", "2026-02-04", 100.0, 110.0),
        fill("BBB", "2026-02-24", -100.0, 105.0),
    ]
    replay = fifo_wash_replay(fills, wash_window_days=30)
    assert replay["n_wash_events"] == 1
    assert replay["deferred_open_at_end"] == pytest.approx(1000.0)


def test_wash_wedge_monetizes_prepaid_tax_at_the_hurdle() -> None:
    replay = {"deferred_open_at_end": 10_000.0}
    wedge = wash_wedge(replay, tau=0.5, hurdle_rate=0.04, avg_equity=1_000_000.0, years=0.5)
    assert wedge["prepaid_tax_usd"] == pytest.approx(5_000.0)
    assert wedge["carry_cost_usd_per_year"] == pytest.approx(200.0)
    assert wedge["wedge_bps_per_year"] == pytest.approx(-2.0)
    assert wedge["prepaid_one_time_bps"] == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# asymmetry 2 — PIL vs the long leg's dividend treatment
# ---------------------------------------------------------------------------


def test_pil_wedge_zero_in_all_gain_years_with_bounds() -> None:
    # Capitalized PIL is relieved at tau in net-gain years exactly as the
    # baseline relieves it: measured wedge 0, bracketed by the no-relief
    # bound -tau*PIL and the full-relief bound 0.
    price = {2024: 50_000.0, 2025: 60_000.0}
    pil = {2024: 10_000.0, 2025: 10_000.0}
    long_div = {2024: 6_000.0, 2025: 6_000.0}
    out = pil_wedge(price, pil, long_div, loss_cap=3_000.0, tau=0.5, hurdle_rate=0.04,
                    equity=1_000_000.0, years=2.0)
    assert out["wedge_bps_per_year"] == pytest.approx(0.0)
    assert out["bound_no_relief_bps_per_year"] == pytest.approx(-0.5 * 10_000.0 / 1_000_000.0 * 1e4)
    assert out["bound_full_relief_bps_per_year"] == 0.0
    # The long leg is ordinary at tau on both codes: a derived zero, stated.
    assert out["long_leg_qualified_wedge_bps_per_year"] == 0.0
    assert "recorded, not answered" in out["long_leg_note"]


def test_pil_wedge_worsens_the_crash_year() -> None:
    # In a loss year the capitalized PIL deepens the capped loss: the
    # attribution difference between engine(price - PIL) and engine(price)
    # is negative (a drag), and it scales with tau.
    price = {2024: -100_000.0, 2025: 20_000.0}
    pil = {2024: 10_000.0, 2025: 0.0}
    out = pil_wedge(price, pil, {}, loss_cap=3_000.0, tau=0.5, hurdle_rate=0.04,
                    equity=1_000_000.0, years=2.0)
    # Extra terminal carryforward from the PIL: 10k more unrelieved at end,
    # plus one year of carry on the extra prepaid tax (0.5*10k*4%).
    expected_cost = 0.5 * 10_000.0 + 0.5 * 10_000.0 * 0.04
    assert out["wedge_bps_per_year"] == pytest.approx(-expected_cost / 1_000_000.0 / 2.0 * 1e4)
    half_tau = pil_wedge(price, pil, {}, loss_cap=3_000.0, tau=0.25, hurdle_rate=0.04,
                         equity=1_000_000.0, years=2.0)
    assert half_tau["wedge_bps_per_year"] == pytest.approx(out["wedge_bps_per_year"] / 2.0)


# ---------------------------------------------------------------------------
# asymmetry 3 — the loss cap binds in the crash year, not steady-state
# ---------------------------------------------------------------------------


def test_loss_cap_not_binding_steady_state() -> None:
    # Small loss inside the cap: fully relieved in-year, no carryforward,
    # no wedge — the steady-state case.
    engine = capital_engine({2024: 5_000.0, 2025: -2_000.0}, loss_cap=3_000.0, tau=0.5, hurdle_rate=0.04)
    assert engine["carryforward_end"] == 0.0
    assert engine["terminal_unrelieved_tax"] == 0.0
    assert engine["timing_cost_total"] == 0.0
    wedge = loss_cap_wedge(engine, equity=1_000_000.0, years=2.0)
    assert wedge["wedge_bps_per_year"] == 0.0


def test_loss_cap_binds_in_the_crash_year() -> None:
    # A -300k year against a 3k cap: baseline refunds 150k, actual relieves
    # 1.5k in-year; the crash-year delta is -tau*(300k - 3k).
    engine = capital_engine({2024: -300_000.0, 2025: 5_000.0}, loss_cap=3_000.0, tau=0.5, hurdle_rate=0.04)
    crash = engine["per_year"][0]
    assert crash["after_tax_delta"] == pytest.approx(-0.5 * (300_000.0 - 3_000.0))
    # 2025 gains absorb 5k, cap another 3k: carryforward persists at end.
    assert engine["carryforward_end"] == pytest.approx(297_000.0 - 5_000.0 - 3_000.0)
    assert engine["terminal_unrelieved_tax"] == pytest.approx(0.5 * 289_000.0)
    # Conservation: deltas sum to minus the terminal unrelieved tax.
    assert engine["after_tax_delta_total"] == pytest.approx(-engine["terminal_unrelieved_tax"])


def test_carryforward_relief_is_timing_only() -> None:
    # Loss fully absorbed by later gains: no terminal component, only the
    # time value of the prepaid tax at the hurdle rate.
    engine = capital_engine({2024: -10_000.0, 2025: 20_000.0}, loss_cap=3_000.0, tau=0.5, hurdle_rate=0.04)
    assert engine["carryforward_end"] == 0.0
    assert engine["terminal_unrelieved_tax"] == 0.0
    # Year-end carryforward 7k -> prepaid tax 3.5k carried one year at 4%.
    assert engine["timing_cost_total"] == pytest.approx(0.5 * 7_000.0 * 0.04)
    assert engine["after_tax_delta_total"] == pytest.approx(0.0)


def test_worst_year_cell_reports_the_gap_before_recovery() -> None:
    engine = capital_engine(
        {2023: -100_000.0, 2024: 200_000.0}, loss_cap=3_000.0, tau=0.5, hurdle_rate=0.04
    )
    cell = worst_year_cell(engine, state_bps=-30.0, equity=1_000_000.0)
    assert cell["year"] == 2023
    assert cell["capital_cell_bps"] == pytest.approx(-0.5 * 97_000.0 / 1_000_000.0 * 1e4)
    assert cell["conditional_wedge_bps"] == pytest.approx(cell["capital_cell_bps"] - 30.0)
    assert cell["carryforward_relieved_later_in_sample"] is True
    assert cell["wash_bps"] is None and "unsupported" in cell["wash_note"]


def test_synthetic_crash_cell_matches_hand_arithmetic() -> None:
    # -30% on 1M plus 10k PIL: loss 310k, baseline refund tau*310k, actual
    # relief tau*3k -> capital gap -tau*307k = -1535 bps at tau=0.5.
    cell = synthetic_crash_cell(
        equity=1_000_000.0, loss_cap=3_000.0, tau=0.5, pil_annual_usd=10_000.0, state_bps=-30.0
    )
    assert cell["book_return"] == -0.30
    assert cell["capital_cell_bps"] == pytest.approx(-0.5 * 307_000.0 / 1_000_000.0 * 1e4)
    assert cell["conditional_wedge_bps"] == pytest.approx(cell["capital_cell_bps"] - 30.0)
    assert cell["wash_bps"] is None


# ---------------------------------------------------------------------------
# asymmetry 4 — the state-exemption term scales with the state rate
# ---------------------------------------------------------------------------


def test_state_exemption_scales_with_state_rate() -> None:
    base = state_exemption_wedge(hurdle_annual_pct=3.71, state_rate=0.093)
    assert base["wedge_bps_per_year"] == pytest.approx(-0.093 * 371.0)
    doubled = state_exemption_wedge(hurdle_annual_pct=3.71, state_rate=0.186)
    assert doubled["wedge_bps_per_year"] == pytest.approx(2.0 * base["wedge_bps_per_year"])
    assert state_exemption_wedge(3.71, 0.0)["wedge_bps_per_year"] == 0.0
