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
    VOL_BASES,
    HurdleInputs,
    HurdleResult,
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
