"""Tests for the shared after-cost Sharpe hurdle (``prism.validation.hurdle``).

The hurdle was computed three ways before this module existed. These tests pin
the formula of SPEC.md §3 (amended 2026-09-25), the exact reproduction of the
recorded zero-yield convention, the loud refusals, and the agreement with the
older unit-conversion helper that the recorded ledgers were written with.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from prism.validation.hurdle import (
    PACKET_BLOCK_KEYS,
    SLEEVE_BLOCK_KEYS,
    VOL_BASES,
    CashToll,
    HurdleInputs,
    HurdleResult,
    cash_toll,
    tbill_hurdle,
)
from prism.validation.metrics import after_cost_hurdle_periodic

REPO = Path(__file__).resolve().parents[1]
B1_SUMMARY = REPO / "results" / "demotion_b1" / "summary.json"


def test_zero_yield_reproduces_rf_over_vol_exactly():
    # The recorded vol-momentum hurdle: rf 4.21% at a 12% target, 0.351.
    found = tbill_hurdle(0.0421, 0.0, 0.5, 0.12, vol_basis="target")
    assert found.sharpe_annual == 0.0421 / 0.12
    assert found.return_hurdle_annual == 0.0421
    assert found.idle_fraction == 0.5  # recorded but not charged: the yield is zero
    assert found.sharpe_annual == pytest.approx(0.351, abs=5e-4)


def test_amended_formula_charges_idle_cash_at_the_yield():
    found = tbill_hurdle(0.04, 0.033, 0.5, 0.10)
    assert found.idle_fraction == 0.5
    assert found.return_hurdle_annual == pytest.approx(0.04 - 0.033 * 0.5)
    assert found.sharpe_annual == pytest.approx((0.04 - 0.033 * 0.5) / 0.10)


@pytest.mark.parametrize("long_over_equity", [1.0, 1.3, 2.0])
def test_idle_fraction_clamps_at_zero_when_fully_invested(long_over_equity):
    found = tbill_hurdle(0.04, 0.033, long_over_equity, 0.10)
    assert found.idle_fraction == 0.0
    assert found.sharpe_annual == pytest.approx(0.4)


def test_periodic_conversion_divides_by_root_periods():
    found = tbill_hurdle(0.04, 0.033, 0.5, 0.10, periods_per_year=252)
    assert found.sharpe_periodic == found.sharpe_annual / math.sqrt(252)
    monthly = tbill_hurdle(0.04, 0.033, 0.5, 0.10, periods_per_year=12)
    assert monthly.sharpe_annual == found.sharpe_annual
    assert monthly.sharpe_periodic == pytest.approx(found.sharpe_annual / math.sqrt(12))
    assert monthly.inputs.periods_per_year == 12


@pytest.mark.parametrize("vol", [0.0, -0.1, float("nan"), float("inf")])
def test_refuses_bad_vol(vol):
    with pytest.raises(ValueError, match="vol_annual"):
        tbill_hurdle(0.04, 0.033, 0.5, vol)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rf_annual": float("nan")}, "rf_annual"),
        ({"cash_yield": float("inf")}, "cash_yield"),
        ({"long_over_equity": -0.1}, "long_over_equity"),
        ({"long_over_equity": float("nan")}, "long_over_equity"),
        ({"periods_per_year": 0}, "periods_per_year"),
        ({"vol_basis": "annual"}, "vol_basis"),
    ],
)
def test_refuses_other_bad_inputs(kwargs, match):
    base = {"rf_annual": 0.04, "cash_yield": 0.033, "long_over_equity": 0.5, "vol_annual": 0.1}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        tbill_hurdle(**base)


def test_sources_and_basis_are_echoed_back():
    found = tbill_hurdle(
        0.0421, 0.033, 0.54, 0.12,
        rf_source="French RF, last 252 sessions of the 202607 file",
        cash_yield_source="examples/reference/costs/retail_v2.toml cash_income_bps_annual",
        vol_basis="target",
    )
    assert isinstance(found, HurdleResult) and isinstance(found.inputs, HurdleInputs)
    assert found.inputs.rf_source.startswith("French RF")
    assert found.inputs.cash_yield_source.endswith("cash_income_bps_annual")
    assert found.inputs.vol_basis == "target" and "target" in VOL_BASES
    flat = found.to_dict()
    for key in ("rf_annual", "cash_yield", "long_over_equity", "vol_annual", "periods_per_year",
                "rf_source", "cash_yield_source", "vol_basis", "idle_fraction", "return_hurdle_annual",
                "sharpe_annual", "sharpe_periodic"):
        assert key in flat
    block = found.packet_block()
    assert tuple(block) == PACKET_BLOCK_KEYS
    assert block["hurdle_sharpe_annual"] == found.sharpe_annual
    assert block["hurdle_sharpe_periodic"] == found.sharpe_periodic


def test_frozen_dataclasses_refuse_mutation():
    found = tbill_hurdle(0.04, 0.0, 1.0, 0.1)
    with pytest.raises(Exception):
        found.sharpe_annual = 0.0  # type: ignore[misc]
    with pytest.raises(Exception):
        found.inputs.rf_annual = 0.0  # type: ignore[misc]


def test_agrees_with_the_periodic_unit_conversion_helper():
    # after_cost_hurdle_periodic(h, periodic_vol) is the zero-yield, fully-invested case at
    # periodic_vol * sqrt(P). The two differ in the last bit on some recorded values, which is
    # why the older helper keeps its literal arithmetic; they agree to float precision.
    for h, vol_d in ((0.0371, 0.00725), (0.0504, 0.01), (0.0378, 0.0037)):
        shared = tbill_hurdle(h, 0.0, 1.0, vol_d * math.sqrt(252), vol_basis="periodic_realized")
        assert shared.sharpe_periodic == pytest.approx(after_cost_hurdle_periodic(h, vol_d), rel=1e-12)


def test_reproduces_the_live_book_record():
    # results/demotion_b1/summary.json: 3.71% T-bill at the book's realized vol.
    summary = json.loads(B1_SUMMARY.read_text())
    recorded = summary["after_cost_hurdle"]
    assert recorded["basis"] == "tbill_nominal"
    found = tbill_hurdle(recorded["annual_pct"] / 100.0, 0.0, 1.0, summary["annualized_vol"],
                         vol_basis="realized_annualized")
    assert found.sharpe_periodic == pytest.approx(recorded["periodic_sharpe_hurdle"], rel=1e-12)
    assert found.sharpe_annual == pytest.approx(0.0371 / summary["annualized_vol"])


# --------------------------------------------------------------------- T-bill ETF (sleeve) inputs

# The recorded vol-momentum inputs (results/2026-09-vol-momentum/summary.json): rf 4.21%, a 12%
# target, mean scale 1.0786 so the stock long side is 0.53931 of equity. The ETF holds idle equity
# above a 2% cash buffer, 0.98 - 0.53931 = 0.44069, and yields rf less a 10 bp fee, 4.11%.
RF, TARGET, LONG_E = 0.0421, 0.12, 0.53931
SLEEVE, SLEEVE_YIELD = 0.44069, 0.0411


def test_defaults_reproduce_the_recorded_hurdles():
    assert tbill_hurdle(RF, 0.0, LONG_E, TARGET).sharpe_annual == pytest.approx(0.35083, abs=5e-6)
    assert tbill_hurdle(RF, 0.033, LONG_E, TARGET).sharpe_annual == pytest.approx(0.2241, abs=5e-5)


@pytest.mark.parametrize("rf", [0.0, 0.0371, 0.0421, -0.001])
@pytest.mark.parametrize("y", [0.0, 0.033, 0.0411])
@pytest.mark.parametrize("long_e", [0.0, 0.3, 0.53931, 1.0, 1.4])
@pytest.mark.parametrize("vol", [0.07, 0.12, 0.127])
def test_sleeve_defaults_are_bit_identical(rf, y, long_e, vol):
    # The pre-sleeve formula, written out literally.
    idle = max(0.0, 1.0 - long_e)
    expected = (rf - y * idle) / vol
    found = tbill_hurdle(rf, y, long_e, vol)
    explicit = tbill_hurdle(rf, y, long_e, vol, sleeve_fraction=0.0, sleeve_yield=0.0)
    assert found.sharpe_annual == expected
    assert found.idle_fraction == idle
    assert found == explicit
    assert not found.inputs.has_sleeve
    assert tuple(found.packet_block()) == PACKET_BLOCK_KEYS
    assert set(found.to_dict()).isdisjoint(SLEEVE_BLOCK_KEYS)


def test_sleeve_hurdle_reproduces_the_planned_governing_figure():
    # (rf - y_s * s - 0 * buffer) / vol: the buffer earns nothing.
    found = tbill_hurdle(RF, 0.0, LONG_E, TARGET, sleeve_fraction=SLEEVE, sleeve_yield=SLEEVE_YIELD,
                         sleeve_yield_source="rf - 10 bp fee (assumption)", vol_basis="target")
    assert found.sharpe_annual == pytest.approx(0.19990, abs=5e-6)
    assert found.return_hurdle_annual == pytest.approx(RF - SLEEVE_YIELD * SLEEVE)
    assert found.idle_fraction == pytest.approx(0.02)  # only the buffer is idle cash
    assert found.inputs.has_sleeve
    assert found.inputs.sleeve_fraction == SLEEVE and found.inputs.sleeve_yield == SLEEVE_YIELD


def test_sleeve_and_buffer_yields_both_count():
    found = tbill_hurdle(RF, 0.033, LONG_E, TARGET, sleeve_fraction=SLEEVE, sleeve_yield=SLEEVE_YIELD)
    buffer = 1.0 - LONG_E - SLEEVE
    assert found.return_hurdle_annual == pytest.approx(RF - SLEEVE_YIELD * SLEEVE - 0.033 * buffer)


def test_sleeve_that_fills_idle_cash_at_the_bill_rate_leaves_rf_on_the_long_side():
    found = tbill_hurdle(0.04, 0.033, 0.6, 0.1, sleeve_fraction=0.4, sleeve_yield=0.04)
    assert found.idle_fraction == 0.0
    assert found.return_hurdle_annual == pytest.approx(0.04 * 0.6)


@pytest.mark.parametrize(
    ("long_e", "sleeve", "match"),
    [
        (0.5, -0.01, "sleeve_fraction must be >= 0"),
        (0.5, 0.5 + 1e-9, "exceeds the idle equity"),
        (0.5, 0.6, "exceeds the idle equity"),
        (1.0, 0.01, "exceeds the idle equity"),
        (1.3, 0.01, "exceeds the idle equity"),
        (0.5, float("nan"), "sleeve_fraction"),
        (0.5, float("inf"), "sleeve_fraction"),
    ],
)
def test_refuses_a_sleeve_outside_idle_equity(long_e, sleeve, match):
    with pytest.raises(ValueError, match=match):
        tbill_hurdle(0.04, 0.0, long_e, 0.1, sleeve_fraction=sleeve, sleeve_yield=0.039)


def test_refuses_a_non_finite_sleeve_yield():
    with pytest.raises(ValueError, match="sleeve_yield"):
        tbill_hurdle(0.04, 0.0, 0.5, 0.1, sleeve_fraction=0.3, sleeve_yield=float("nan"))


def test_sleeve_that_fills_idle_cash_up_to_rounding_is_accepted_and_clamped():
    # A mean of per-session (1 - L_t/E_t) can exceed 1 - mean(L_t/E_t) in the last place.
    found = tbill_hurdle(0.04, 0.033, 0.5, 0.1, sleeve_fraction=0.5 + 1e-13, sleeve_yield=0.039)
    assert found.idle_fraction == 0.0


def test_sleeve_inputs_go_in_an_optional_sub_block():
    found = tbill_hurdle(RF, 0.0, LONG_E, TARGET, sleeve_fraction=SLEEVE, sleeve_yield=SLEEVE_YIELD,
                         sleeve_yield_source="SGOV: rf - 10 bp")
    block = found.packet_block()
    assert tuple(block) == (*PACKET_BLOCK_KEYS, "sleeve")
    assert tuple(block["sleeve"]) == SLEEVE_BLOCK_KEYS
    assert block["sleeve"] == {"sleeve_fraction": SLEEVE, "sleeve_yield": SLEEVE_YIELD,
                               "sleeve_yield_source": "SGOV: rf - 10 bp"}
    flat = found.to_dict()
    assert flat["sleeve_fraction"] == SLEEVE and flat["sleeve_yield_source"] == "SGOV: rf - 10 bp"
    # A source alone declares a sleeve, so the record says which yield was assumed even at s = 0.
    assert tbill_hurdle(RF, 0.0, 1.0, TARGET, sleeve_yield_source="SGOV").inputs.has_sleeve


# --------------------------------------------------------------------- cash_toll


def test_cash_toll_dollar_neutral_arithmetic():
    found = cash_toll(0.35, 0.06, 0.5, 0.0412, 0.005)
    assert isinstance(found, CashToll)
    assert found.spread_per_long_dollar == pytest.approx(2 * 0.35 * 0.06)
    # Dollar neutral: one dollar short per long dollar, so the borrow per long dollar is the fee.
    assert found.hurdle_per_long_dollar == pytest.approx(0.0412 + 0.005)
    assert found.margin_per_long_dollar == pytest.approx(0.042 - 0.0462)
    assert found.excess_over_cash_annual == pytest.approx(0.5 * (0.042 - 0.0462))
    assert found.displaced_fraction == 1.0
    assert set(found.to_dict()) >= {"spread_per_long_dollar", "hurdle_per_long_dollar",
                                    "margin_per_long_dollar", "excess_over_cash_annual"}


def test_cash_toll_breakeven_volatility_per_unit_of_gross():
    # A net Sharpe of 0.35 covers a 4.12% cash yield only above 4.12% / (2 x 0.35) = 5.886%
    # volatility per unit of gross: diversification that lowers it raises the toll's bite.
    breakeven = 0.0412 / (2 * 0.35)
    assert cash_toll(0.35, 0.06, 0.5, 0.0412, 0.0).margin_per_long_dollar > 0.0
    assert cash_toll(0.35, 0.058, 0.5, 0.0412, 0.0).margin_per_long_dollar < 0.0
    assert cash_toll(0.35, breakeven, 0.5, 0.0412, 0.0).margin_per_long_dollar == pytest.approx(0.0, abs=1e-15)


def test_cash_toll_mid_case_matches_the_sleeve_hurdle():
    # Net Sharpe 0.305 at the 12% target and mean gross 1.0786 (L/E 0.5393), ETF at 4.11%, 2% buffer.
    gross, sharpe, buffer = 1.0786, 0.305, 0.02
    long_e = gross / 2
    toll = cash_toll(sharpe, TARGET / gross, long_e, SLEEVE_YIELD, 0.0)
    via_toll = toll.excess_over_cash_annual - (RF - SLEEVE_YIELD) - SLEEVE_YIELD * buffer
    hurdle = tbill_hurdle(RF, 0.0, long_e, TARGET, sleeve_fraction=1.0 - long_e - buffer,
                          sleeve_yield=SLEEVE_YIELD)
    via_hurdle = sharpe * TARGET - hurdle.return_hurdle_annual
    assert via_toll == pytest.approx(via_hurdle, rel=1e-12)
    assert via_toll == pytest.approx(0.0126, abs=5e-5)  # +1.26%/yr over bills


@pytest.mark.parametrize("long_e", [0.2, 0.53931, 0.9, 1.0, 1.2])
@pytest.mark.parametrize("sharpe", [-0.2, 0.1, 0.3, 0.6])
@pytest.mark.parametrize("borrow", [0.0, 0.005])
def test_cash_toll_margin_sign_agrees_with_the_hurdle_at_cash_earning_bills(long_e, sharpe, borrow):
    rf, vol_g = 0.0421, 0.11
    toll = cash_toll(sharpe, vol_g, long_e, rf, borrow)
    gross = long_e / 0.5
    vol = vol_g * gross
    # The book's Sharpe net of the borrow on its short side (short market value = L/E here).
    net_of_borrow = (sharpe * vol - borrow * long_e) / vol
    hurdle = tbill_hurdle(rf, rf, long_e, vol)
    assert (toll.margin_per_long_dollar > 0.0) == (net_of_borrow > hurdle.sharpe_annual)
    # And the gap is the same quantity in different units.
    assert toll.margin_per_long_dollar * long_e == pytest.approx((net_of_borrow - hurdle.sharpe_annual) * vol,
                                                                 abs=1e-15)


def test_cash_toll_above_one_times_equity_charges_cash_only_on_equity():
    found = cash_toll(0.3, 0.1, 1.25, 0.04, 0.0)
    assert found.displaced_fraction == pytest.approx(0.8)
    assert found.hurdle_per_long_dollar == pytest.approx(0.04 * 0.8)


def test_cash_toll_long_only_book_pays_no_borrow():
    found = cash_toll(0.3, 0.1, 0.8, 0.04, 0.005, long_share_of_gross=1.0)
    assert found.spread_per_long_dollar == pytest.approx(0.03)
    assert found.hurdle_per_long_dollar == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"net_sharpe": float("nan")}, "net_sharpe"),
        ({"vol_per_unit_gross": 0.0}, "vol_per_unit_gross"),
        ({"vol_per_unit_gross": -0.1}, "vol_per_unit_gross"),
        ({"long_over_equity": 0.0}, "long_over_equity"),
        ({"long_over_equity": float("inf")}, "long_over_equity"),
        ({"cash_yield": float("nan")}, "cash_yield"),
        ({"borrow_annual": -0.001}, "borrow_annual"),
        ({"long_share_of_gross": 0.0}, "long_share_of_gross"),
        ({"long_share_of_gross": 1.5}, "long_share_of_gross"),
    ],
)
def test_cash_toll_refuses_bad_inputs(kwargs, match):
    base = {"net_sharpe": 0.3, "vol_per_unit_gross": 0.1, "long_over_equity": 0.5, "cash_yield": 0.04,
            "borrow_annual": 0.005}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        cash_toll(**base)
