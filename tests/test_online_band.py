"""Tests for the single-step online no-trade band (SPEC.md §7.3)."""

import numpy as np
import pandas as pd
import pytest

from prism.portfolio.construct import apply_no_trade_band, step_no_trade_band


class TestStepNoTradeBand:
    def test_holds_within_band_moves_outside(self):
        prev = pd.Series({"A": 0.10, "B": 0.10})
        tgt = pd.Series({"A": 0.11, "B": 0.20})  # A moves 0.01 (<0.05), B moves 0.10 (>0.05)
        out = step_no_trade_band(prev, tgt, 0.05)
        assert out["A"] == pytest.approx(0.10)  # held
        assert out["B"] == pytest.approx(0.20)  # moved

    def test_preserves_hysteresis_across_days(self):
        # The core online property: state carried forward, not reset to flat.
        band = 0.05
        held = pd.Series({"A": 0.10})
        # Targets oscillate within +/-band of the HELD weight every day, so a
        # correct online band never trades. The stateful step must remember 0.10
        # across days — a reset-to-flat variant would compute vs 0 and re-trade.
        for tgt_val in (0.11, 0.09, 0.13, 0.07, 0.12):
            held = step_no_trade_band(held, pd.Series({"A": tgt_val}), band)
            assert held["A"] == pytest.approx(0.10)

    def test_crosses_band_once_target_drifts_past_it(self):
        band = 0.05
        held = pd.Series({"A": 0.0})
        results = []
        for day in range(1, 4):  # target 0.02, 0.04, 0.06
            held = step_no_trade_band(held, pd.Series({"A": 0.02 * day}), band)
            results.append(held["A"])
        # Held stays 0 until the target passes the band (day 3: 0.06 > 0.05).
        assert results == pytest.approx([0.0, 0.0, 0.06])

    def test_matches_batch_band_when_fed_sequentially(self):
        rng = np.random.default_rng(3)
        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        targets = pd.DataFrame(rng.uniform(-0.3, 0.3, (30, 4)), index=idx, columns=list("ABCD"))
        band = 0.08
        batch = apply_no_trade_band(targets, band)
        held = pd.Series(0.0, index=targets.columns)
        online_rows = []
        for _, row in targets.iterrows():
            held = step_no_trade_band(held, row, band)
            online_rows.append(held)
        online = pd.DataFrame(online_rows, index=targets.index)
        pd.testing.assert_frame_equal(online[batch.columns], batch, check_names=False)

    def test_missing_target_holds_prior(self):
        prev = pd.Series({"A": 0.3, "B": -0.2})
        tgt = pd.Series({"A": 0.9})  # B has no decision today
        out = step_no_trade_band(prev, tgt, 0.05)
        assert out["B"] == pytest.approx(-0.2)
        assert out["A"] == pytest.approx(0.9)  # moved (0.6 > band)

    def test_per_name_band_series(self):
        prev = pd.Series({"A": 0.0, "B": 0.0})
        tgt = pd.Series({"A": 0.05, "B": 0.05})
        band = pd.Series({"A": 0.10, "B": 0.01})  # A tight-held, B loose-moves
        out = step_no_trade_band(prev, tgt, band)
        assert out["A"] == pytest.approx(0.0)   # 0.05 < 0.10 -> held
        assert out["B"] == pytest.approx(0.05)  # 0.05 > 0.01 -> moved

    def test_zero_band_always_moves(self):
        prev = pd.Series({"A": 0.1})
        tgt = pd.Series({"A": 0.1000001})
        out = step_no_trade_band(prev, tgt, 0.0)
        assert out["A"] == pytest.approx(0.1000001)
