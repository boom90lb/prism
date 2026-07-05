"""Tests for the single-step online no-trade band (SPEC.md §7.3)."""

import logging

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


class TestNaNBandPolicy:
    """Pin the ONE degenerate-band policy shared by both forms (SPEC N7):
    a NaN or -inf band value DISABLES the band for that name (trade straight
    to target) and warns loudly, while +inf is a legitimate explicit
    never-trade pin (no warning). Historically the batch form froze a
    NaN-band name forever while the online form silently disabled it. A
    future "freeze" policy variant for NaN would be one more parametrized
    branch here — see ``_band_step``'s docstring for where it would live.
    """

    _LOGGER = "prism.portfolio.construct"

    @staticmethod
    def _run_batch(band):
        targets = pd.DataFrame({"A": [0.10, 0.20], "B": [0.10, 0.20]})
        return apply_no_trade_band(targets, band)

    @staticmethod
    def _run_online(band):
        prev = pd.Series({"A": 0.10, "B": 0.10})
        tgt = pd.Series({"A": 0.11, "B": 0.11})
        return step_no_trade_band(prev, tgt, band)

    @pytest.mark.parametrize("form", ["batch", "online"])
    @pytest.mark.parametrize("degenerate", [np.nan, -np.inf])
    def test_degenerate_band_name_equals_disabled_band(self, form, degenerate, caplog):
        run = self._run_batch if form == "batch" else self._run_online
        bad_band = pd.Series({"A": degenerate, "B": 0.15})
        zero_band = pd.Series({"A": 0.0, "B": 0.15})
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            with_bad = run(bad_band)
        baseline = run(zero_band)
        if form == "batch":
            pd.testing.assert_frame_equal(with_bad, baseline)
        else:
            pd.testing.assert_series_equal(with_bad, baseline)
        assert any("NaN/-inf" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("form", ["batch", "online"])
    def test_posinf_band_pins_name_without_warning(self, form, caplog):
        run = self._run_batch if form == "batch" else self._run_online
        band = pd.Series({"A": np.inf, "B": 0.15})
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            out = run(band)
        if form == "batch":
            # A pinned at the flat start for the whole frame; B trades once
            # its target clears the 0.15 band.
            assert (out["A"] == 0.0).all()
            assert out["B"].tolist() == pytest.approx([0.0, 0.20])
        else:
            assert out["A"] == pytest.approx(0.10)  # held at prev
        assert not caplog.records

    @pytest.mark.parametrize("form", ["batch", "online"])
    def test_nan_scalar_band_disables_banding(self, form, caplog):
        run = self._run_batch if form == "batch" else self._run_online
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            with_nan = run(float("nan"))
        baseline = run(0.0)
        if form == "batch":
            pd.testing.assert_frame_equal(with_nan, baseline)
        else:
            pd.testing.assert_series_equal(with_nan, baseline)
        assert any("NaN/-inf" in r.getMessage() for r in caplog.records)

    def test_absent_name_has_no_band_and_no_warning(self, caplog):
        prev = pd.Series({"A": 0.10})
        tgt = pd.Series({"A": 0.11})
        band = pd.Series({"B": 0.5})  # A absent from the band Series
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            out = step_no_trade_band(prev, tgt, band)
        assert out["A"] == pytest.approx(0.11)  # no band -> moved
        assert not caplog.records
