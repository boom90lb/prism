"""Tests for the declared block bootstrap (engine specification §3.4, "Uncertainty").

The failures these are written to catch, in order of how much they would cost:

1. A resample that breaks a cross-section apart. The whole point of resampling decision
   *periods* is that a drawn period arrives with its names intact; a resampler that mixed
   names across dates would destroy the dependence structure and quietly narrow every
   interval built on it.
2. A block scheme that is not the declared one — wrong block count, wrong length, blocks
   that do not wrap circularly, or a resample shorter than the sample.
3. An insufficient bootstrap that returns a number anyway. Below ``min_blocks`` the answer
   is an exclusion with a reason, and reading an endpoint off it raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.validation.uncertainty import (
    BlockBootstrapSpec,
    BootstrapError,
    BootstrapExcluded,
    block_bootstrap,
    block_resample_indices,
    concat_units,
    period_units,
    plan_blocks,
    validate_block_span,
)

#: The reference monthly book of §3.4: 21-bar horizon, 21-bar decision cadence.
REFERENCE = BlockBootstrapSpec(block_periods=3, n_resamples=200, seed=20260906, min_blocks=4)


def _mean(units: list[float]) -> float:
    return float(np.mean(units))


class TestSpecValidation:
    def test_rejects_unknown_scheme(self):
        with pytest.raises(ValueError, match="block_scheme"):
            BlockBootstrapSpec(block_periods=3, n_resamples=10, seed=1, block_scheme="stationary")

    def test_rejects_degenerate_counts(self):
        with pytest.raises(ValueError, match="block_periods"):
            BlockBootstrapSpec(block_periods=0, n_resamples=10, seed=1)
        with pytest.raises(ValueError, match="n_resamples"):
            BlockBootstrapSpec(block_periods=3, n_resamples=0, seed=1)
        with pytest.raises(ValueError, match="min_blocks"):
            BlockBootstrapSpec(block_periods=3, n_resamples=10, seed=1, min_blocks=0)

    def test_method_block_records_every_declared_key(self):
        method = REFERENCE.method()
        assert method["block_scheme"] == "circular_moving"
        assert method["block_periods"] == 3
        assert method["n_resamples"] == 200
        assert method["seed"] == 20260906
        assert method["min_blocks"] == 4
        assert "directional" in method["coverage_statement"]


class TestBlockSpan:
    def test_reference_configuration_spans_a_label(self):
        # block_periods 3 * update_every_bars 21 = 63 bars >= horizon_bars 21.
        validate_block_span(3, 21, 21)

    def test_block_shorter_than_a_label_is_refused(self):
        # A daily book with a 21-bar horizon and a 1-period block spans 1 bar < 21.
        with pytest.raises(ValueError, match="shorter than horizon_bars"):
            validate_block_span(1, 1, 21)

    def test_minimum_for_the_reference_book_is_one_period(self):
        validate_block_span(1, 21, 21)


class TestPlanBlocks:
    def test_reference_fold_geometry(self):
        # §3.4 (Run): a 312-bar formation at a 21-bar cadence holds 13 labelled decision
        # periods, so the reference bootstrap draws ceil(13 / 3) = 5 blocks.
        plan = plan_blocks((312 - 21) // 21, REFERENCE)
        assert plan.n_units == 13
        assert plan.blocks_drawn == 5
        assert plan.sufficient
        assert plan.exclusion_reason is None

    def test_below_min_blocks_is_an_exclusion_with_a_reason(self):
        plan = plan_blocks(6, REFERENCE)  # ceil(6/3) = 2 < min_blocks 4
        assert plan.blocks_drawn == 2
        assert not plan.sufficient
        assert "min_blocks=4" in str(plan.exclusion_reason)

    def test_block_longer_than_the_sample_is_refused(self):
        with pytest.raises(ValueError, match="exceeds the sample"):
            plan_blocks(2, REFERENCE)


class TestResampleIndices:
    def test_shape_is_resamples_by_units(self):
        idx = block_resample_indices(13, REFERENCE)
        assert idx.shape == (200, 13)
        assert idx.min() >= 0 and idx.max() <= 12

    def test_blocks_are_circular_and_consecutive(self):
        idx = block_resample_indices(13, REFERENCE)
        for row in idx:
            for start in range(0, 12, 3):  # the last block is truncated at 13 units
                block = row[start : start + 3]
                expected = (block[0] + np.arange(len(block))) % 13
                assert list(block) == list(expected)

    def test_length_equals_the_sample_after_truncation(self):
        # 5 blocks of 3 is 15 draws truncated back to 13 periods.
        assert block_resample_indices(13, REFERENCE).shape[1] == 13

    def test_same_seed_reproduces_the_same_resamples(self):
        first = block_resample_indices(13, REFERENCE)
        second = block_resample_indices(13, REFERENCE)
        assert np.array_equal(first, second)
        other = BlockBootstrapSpec(block_periods=3, n_resamples=200, seed=20260907, min_blocks=4)
        assert not np.array_equal(first, block_resample_indices(13, other))

    def test_insufficient_plan_raises_rather_than_resampling(self):
        with pytest.raises(BootstrapExcluded, match="min_blocks"):
            block_resample_indices(6, REFERENCE)


class TestBlockBootstrap:
    def test_resampled_statistic_is_recorded_with_its_method(self):
        result = block_bootstrap(list(range(13)), _mean, REFERENCE)
        assert not result.excluded
        assert result.values.size == 200
        assert result.observed == pytest.approx(6.0)
        assert result.blocks_drawn == 5
        assert result.to_dict()["method"]["block_periods"] == 3

    def test_exclusion_is_data_not_an_exception(self):
        result = block_bootstrap(list(range(6)), _mean, REFERENCE)
        assert result.excluded
        assert result.values.size == 0
        assert "min_blocks=4" in str(result.exclusion_reason)
        assert result.to_dict()["blocks_drawn"] == 2

    def test_reading_an_endpoint_off_an_excluded_result_raises(self):
        result = block_bootstrap(list(range(6)), _mean, REFERENCE)
        with pytest.raises(BootstrapExcluded, match="not a NaN"):
            result.quantile(0.5)
        with pytest.raises(BootstrapExcluded):
            result.one_sided_endpoints(0.8)

    def test_one_sided_endpoints_follow_the_declared_rule(self):
        result = block_bootstrap(list(range(13)), _mean, REFERENCE)
        lower, upper = result.one_sided_endpoints(0.8)
        assert lower == pytest.approx(result.quantile(0.2))
        assert upper == pytest.approx(result.quantile(0.8))
        assert lower <= upper
        with pytest.raises(ValueError, match="confidence"):
            result.one_sided_endpoints(0.5)

    def test_non_finite_statistic_is_a_failure_not_a_wide_interval(self):
        def broken(units: list[float]) -> float:
            return float("nan")

        with pytest.raises(BootstrapError, match="non-finite"):
            block_bootstrap(list(range(13)), broken, REFERENCE)

    def test_empty_units_refused(self):
        with pytest.raises(ValueError, match="empty"):
            block_bootstrap([], _mean, REFERENCE)


class TestCrossSectionsStayIntact:
    """The failure most worth catching: a resample that breaks a cross-section apart."""

    @staticmethod
    def _panel() -> pd.DataFrame:
        rows = []
        for period in range(13):
            for name in ("AAA", "BBB", "CCC"):
                rows.append({"period": period, "sid": name, "score": period + hash(name) % 7})
        return pd.DataFrame(rows)

    def test_every_resampled_period_is_one_original_cross_section(self):
        panel = self._panel()
        units = period_units(panel, "period")
        assert len(units) == 13
        originals = {int(unit["period"].iloc[0]): unit[["sid", "score"]].reset_index(drop=True) for unit in units}

        seen: list[pd.DataFrame] = []

        def statistic(drawn: list[pd.DataFrame]) -> float:
            seen.append(concat_units(drawn))
            return float(np.mean([float(unit["score"].mean()) for unit in drawn]))

        spec = BlockBootstrapSpec(block_periods=3, n_resamples=5, seed=1, min_blocks=4)
        block_bootstrap(units, statistic, spec)

        for frame in seen:
            assert len(frame) == 13 * 3
            for label, group in frame.groupby("resample_period"):
                assert list(group["sid"]) == ["AAA", "BBB", "CCC"], f"cross-section {label} was broken apart"
                source = int(group["period"].iloc[0])
                assert (group["period"] == source).all(), "a resampled period mixed two decision dates"
                pd.testing.assert_frame_equal(
                    group[["sid", "score"]].reset_index(drop=True), originals[source], check_dtype=False
                )

    def test_a_period_drawn_twice_counts_twice_under_its_own_label(self):
        panel = self._panel()
        units = period_units(panel, "period")
        drawn = [units[4], units[4], units[7]]
        frame = concat_units(drawn)
        assert sorted(frame["resample_period"].unique()) == [0, 1, 2]
        assert (frame.groupby("resample_period")["period"].first().to_numpy() == np.array([4, 4, 7])).all()

    def test_period_units_refuses_a_missing_column(self):
        with pytest.raises(KeyError, match="decision"):
            period_units(self._panel(), "decision")
