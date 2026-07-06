"""Per-bucket spread calibration from the fills ledger (I-9, SPEC §13 R2 exit).

Pins the estimator half of the cost instrument on synthetic fills: the
signed arrival-slippage convention, bucket partitioning on the schedule's
dollar-volume floors, notional weighting, the min-fills promotion guard
(an under-sampled bucket keeps its conservative fallback), and the
column-compatibility of the estimator with the ledger the live loop
actually writes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.execution.spread import (
    DEFAULT_BUCKET_FLOORS,
    arrival_slippage_bps,
    calibrated_bucket_schedule,
    spread_calibration_table,
)

# Mirrors research/arbitrage/residual_walk_forward.py::SPREAD_BUCKET_SCHEDULE_V1
# (the pre-registered conservative-upper fallback the calibration replaces).
_FALLBACK = ((500e6, 1.0), (100e6, 2.0), (25e6, 5.0), (0.0, 10.0))


def _fill(symbol: str, qty: float, fill_price: float, reference_price: float) -> dict:
    return {
        "client_order_id": f"d:{symbol}",
        "symbol": symbol,
        "qty": qty,
        "fill_price": fill_price,
        "reference_price": reference_price,
        "decision_bar": "d",
        "filled_bar": "d+1",
    }


# ---------------------------------------------------------------------------
# arrival_slippage_bps: sign convention and ledger validation
# ---------------------------------------------------------------------------


def test_slippage_sign_convention() -> None:
    fills = pd.DataFrame(
        [
            _fill("BUY_ADVERSE", 10.0, 100.10, 100.0),  # bought above reference: paid
            _fill("SELL_ADVERSE", -10.0, 99.90, 100.0),  # sold below reference: paid
            _fill("BUY_IMPROVED", 10.0, 99.90, 100.0),  # bought below: improvement
        ]
    )
    slippage = arrival_slippage_bps(fills)
    assert slippage.iloc[0] == pytest.approx(10.0)
    assert slippage.iloc[1] == pytest.approx(10.0)
    assert slippage.iloc[2] == pytest.approx(-10.0)


def test_slippage_rejects_malformed_ledger_rows() -> None:
    with pytest.raises(ValueError, match="missing ledger columns"):
        arrival_slippage_bps(pd.DataFrame({"symbol": ["A"], "qty": [1.0]}))
    bad_reference = pd.DataFrame([_fill("A", 1.0, 100.0, 0.0)])
    with pytest.raises(ValueError, match="unusable"):
        arrival_slippage_bps(bad_reference)
    zero_qty = pd.DataFrame([_fill("A", 0.0, 100.0, 100.0)])
    with pytest.raises(ValueError, match="unusable"):
        arrival_slippage_bps(zero_qty)


# ---------------------------------------------------------------------------
# spread_calibration_table: bucketing and statistics
# ---------------------------------------------------------------------------

# MEGA sits in the >=500e6 bucket, MID in [25e6, 100e6), SMALL in the catch-all.
_MDV = pd.Series({"MEGA": 1e9, "MID": 50e6, "SMALL": 1e6})


def test_table_buckets_and_weights() -> None:
    fills = pd.DataFrame(
        [
            _fill("MEGA", 10.0, 100.05, 100.0),  # +5 bps on $1000
            _fill("MEGA", -30.0, 99.90, 100.0),  # +10 bps on $3000
            _fill("SMALL", 5.0, 10.02, 10.0),  # +20 bps on $50
        ]
    )
    table = spread_calibration_table(fills, _MDV)
    assert list(table["floor"]) == list(DEFAULT_BUCKET_FLOORS)
    mega = table.iloc[0]
    assert mega["n_fills"] == 2 and mega["notional"] == pytest.approx(4000.0)
    # Notional-weighted mean: (5*1000 + 10*3000) / 4000 = 8.75 bps.
    assert mega["mean_bps"] == pytest.approx(8.75)
    assert mega["improve_frac"] == 0.0
    empty = table.iloc[1:3]  # no fills landed in the 100e6 or 25e6 buckets
    assert list(empty["n_fills"]) == [0, 0] and empty["mean_bps"].isna().all()
    small = table.iloc[3]
    assert small["n_fills"] == 1 and small["mean_bps"] == pytest.approx(20.0)
    assert np.isnan(small["se_bps"])  # one fill has no standard error


def test_table_requires_liquidity_for_every_traded_name() -> None:
    fills = pd.DataFrame([_fill("GHOST", 1.0, 100.0, 100.0)])
    with pytest.raises(ValueError, match="GHOST"):
        spread_calibration_table(fills, _MDV)


def test_table_rejects_non_descending_floors() -> None:
    fills = pd.DataFrame([_fill("MEGA", 1.0, 100.0, 100.0)])
    with pytest.raises(ValueError, match="descending"):
        spread_calibration_table(fills, _MDV, floors=(25e6, 100e6, 0.0))
    with pytest.raises(ValueError, match="catch-all"):
        spread_calibration_table(fills, _MDV, floors=(100e6, 25e6))


# ---------------------------------------------------------------------------
# calibrated_bucket_schedule: promotion guard and fallback merge
# ---------------------------------------------------------------------------


def _many_fills(symbol: str, n: int, fill_price: float, reference: float) -> pd.DataFrame:
    return pd.DataFrame([_fill(symbol, 10.0, fill_price, reference) for _ in range(n)])


def test_schedule_promotes_only_sampled_buckets() -> None:
    # 30 MEGA fills at +5 bps: promoted. 3 SMALL fills: below min_fills, fallback.
    fills = pd.concat(
        [_many_fills("MEGA", 30, 100.05, 100.0), _many_fills("SMALL", 3, 10.02, 10.0)],
        ignore_index=True,
    )
    table = spread_calibration_table(fills, _MDV)
    schedule = calibrated_bucket_schedule(table, _FALLBACK, min_fills=30)
    assert schedule[0] == (500e6, pytest.approx(5.0))  # measured
    assert schedule[1] == (100e6, 2.0)  # empty bucket: fallback
    assert schedule[2] == (25e6, 5.0)  # empty bucket: fallback
    assert schedule[3] == (0.0, 10.0)  # under-sampled: fallback, not the 20 bps noise


def test_schedule_floors_systematic_improvement_at_zero() -> None:
    # Systematic price improvement must never become a negative cost.
    fills = _many_fills("MEGA", 30, 99.95, 100.0)  # buys filled 5 bps below reference
    table = spread_calibration_table(fills, _MDV)
    schedule = calibrated_bucket_schedule(table, _FALLBACK, min_fills=30)
    assert schedule[0] == (500e6, 0.0)


def test_schedule_rejects_mismatched_floors() -> None:
    fills = _many_fills("MEGA", 5, 100.05, 100.0)
    table = spread_calibration_table(fills, _MDV)
    with pytest.raises(ValueError, match="partition liquidity identically"):
        calibrated_bucket_schedule(table, ((1e9, 1.0), (0.0, 10.0)))


# ---------------------------------------------------------------------------
# End-to-end: the ledger the live loop writes feeds the estimator directly
# ---------------------------------------------------------------------------


def test_live_fills_ledger_round_trips_into_calibration(tmp_path) -> None:
    from prism.live import (
        LiveLoopContext,
        StateStore,
        decide_and_submit,
        read_fills_ledger,
        settle,
    )
    from tests.test_live_loop import FakeBroker

    ctx = LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=FakeBroker(),
        fills_ledger=tmp_path / "fills.jsonl",
    )
    decide_and_submit(ctx, "d1", pd.Series({"MEGA": 0.5}), pd.Series({"MEGA": 100.0}))
    ctx.broker.open_prices["MEGA"] = 100.10  # next-open fill 10 bps above reference
    settle(ctx, "d1")

    ledger = read_fills_ledger(ctx.fills_ledger)
    table = spread_calibration_table(ledger, _MDV)
    assert table.iloc[0]["n_fills"] == 1
    assert table.iloc[0]["mean_bps"] == pytest.approx(10.0)
