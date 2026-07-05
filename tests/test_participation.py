"""Tests for the hard participation gate (SPEC.md §3 law 5, §7.4)."""

import numpy as np
import pandas as pd
import pytest

from prism.execution.participation import max_participation_of, participation_capped_targets


class TestParticipationGate:
    def test_caps_trade_to_participation_of_adv(self):
        prev = pd.Series({"A": 0.0})
        tgt = pd.Series({"A": 0.5})
        dvol = pd.Series({"A": 1e6})
        # allowed weight delta = max_part * adv / aum = 0.01 * 1e6 / 1e8 = 1e-4
        capped = participation_capped_targets(prev, tgt, dvol, aum=1e8, max_participation=0.01)
        assert capped["A"] == pytest.approx(1e-4)

    def test_small_trade_within_cap_unchanged(self):
        prev = pd.Series({"A": 0.10})
        tgt = pd.Series({"A": 0.11})
        dvol = pd.Series({"A": 1e12})  # effectively unlimited
        capped = participation_capped_targets(prev, tgt, dvol, aum=1e4, max_participation=0.05)
        assert capped["A"] == pytest.approx(0.11)

    def test_direction_preserved_on_sell(self):
        prev = pd.Series({"A": 0.5})
        tgt = pd.Series({"A": -0.5})  # wants to flip
        dvol = pd.Series({"A": 1e6})
        capped = participation_capped_targets(prev, tgt, dvol, aum=1e8, max_participation=0.01)
        # moves DOWN from 0.5 by at most 1e-4, never overshoots past target
        assert 0.5 - 1e-4 - 1e-9 <= capped["A"] <= 0.5
        assert capped["A"] < prev["A"]

    @pytest.mark.parametrize("bad_adv", [np.nan, 0.0], ids=["unknown_adv", "zero_adv"])
    def test_unusable_adv_holds_prior(self, bad_adv):
        prev = pd.Series({"A": 0.2})
        tgt = pd.Series({"A": 0.9})
        dvol = pd.Series({"A": bad_adv})
        capped = participation_capped_targets(prev, tgt, dvol, aum=1e6, max_participation=0.1)
        assert capped["A"] == pytest.approx(0.2)  # not traded on missing liquidity

    def test_missing_target_holds_prior(self):
        prev = pd.Series({"A": 0.3, "B": -0.2})
        tgt = pd.Series({"A": 0.3})  # no decision for B
        dvol = pd.Series({"A": 1e12, "B": 1e12})
        capped = participation_capped_targets(prev, tgt, dvol, aum=1e3, max_participation=0.5)
        assert capped["B"] == pytest.approx(-0.2)

    def test_never_exceeds_cap_after_capping(self):
        rng = np.random.default_rng(1)
        names = [f"S{i}" for i in range(50)]
        prev = pd.Series(rng.uniform(-0.3, 0.3, 50), index=names)
        tgt = pd.Series(rng.uniform(-0.3, 0.3, 50), index=names)
        dvol = pd.Series(rng.uniform(1e5, 1e8, 50), index=names)
        aum, cap = 5e6, 0.02
        capped = participation_capped_targets(prev, tgt, dvol, aum=aum, max_participation=cap)
        realized = max_participation_of(prev, capped, dvol, aum)
        assert realized <= cap + 1e-9

    def test_bad_args_raise(self):
        prev = pd.Series({"A": 0.0})
        with pytest.raises(ValueError):
            participation_capped_targets(prev, prev, pd.Series({"A": 1.0}), aum=0.0, max_participation=0.1)
        with pytest.raises(ValueError):
            participation_capped_targets(prev, prev, pd.Series({"A": 1.0}), aum=1.0, max_participation=0.0)
