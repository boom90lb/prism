"""Tests for comparison, repricing, and the studentized Sharpe intervals (§3.12).

The failures these are written to catch:

1. **A Sharpe compared with a yield.** The units error the specification forbids is easy to
   commit by accident — a scalar hurdle passed where a return series belongs — so the
   boundary refuses a scalar and says why.
2. **An interval that does not cover.** Plumbing tests pass on a bootstrap that ignores
   serial dependence. ``TestIntervalCoverage`` measures achieved coverage on AR(1) data
   with known dependence and reports the number, and it checks that a block length of one
   bar — the "dependence, what dependence" configuration — destroys that coverage.
3. **Two records compared that are not the same evidence class**: a legacy row against a
   post-overhaul one, two different visibility modes, an engineering-channel run, or two
   arms evaluated on different samples.
4. **A repricing that presents itself as an outcome** instead of a counterfactual attached
   to the record it reprices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.validation.compare import (
    ComparisonArm,
    ComparisonRefused,
    SharpeBootstrapSpec,
    attach_repricing,
    compare_reevaluation,
    compare_repricing,
    constant_hurdle_returns,
    excess_returns,
    excess_sharpe_interval,
    record_field,
    require_return_series,
    sharpe_difference_interval,
    sharpe_interval,
)
from prism.validation.metrics import periodic_sharpe
from prism.validation.uncertainty import BootstrapError

SPEC = SharpeBootstrapSpec(block_bars=5, n_resamples=199, seed=20260908)


def _ar1(rng: np.random.Generator, n: int, mean: float, sd: float, phi: float) -> np.ndarray:
    """AR(1) with stationary mean ``mean`` and stationary standard deviation ``sd``."""
    innovation = sd * np.sqrt(1.0 - phi**2)
    out = np.empty(n, dtype=float)
    out[0] = rng.normal(mean, sd)
    noise = rng.normal(0.0, innovation, n)
    for t in range(1, n):
        out[t] = mean + phi * (out[t - 1] - mean) + noise[t]
    return out


def _record(
    run_id: str,
    *,
    config_id: str | None = "cfg-a",
    arm_id: str = "arm-a",
    capital: float = 1.0,
    run_channel: str = "evidence",
    visibility_mode: str = "unavailable",
    accounting_engine: str = "historical_constant_weight",
    fold_table_sha256: str = "fold-sha",
    synthetic: bool = False,
) -> dict:
    """A record in the nested §3.12 layout, which ``record_field`` reads one level in."""
    return {
        "ids": {"run_id": run_id, "config_id": config_id, "arm_id": arm_id, "kind": "new"},
        "flags": {
            "synthetic": synthetic,
            "prospective": None,
            "fill_source": "simulated",
            "run_channel": run_channel,
        },
        "data": {"visibility_mode": visibility_mode, "vintage": "unavailable", "data_id": "data-1"},
        "conventions": {"executable_return": "open_to_open"},
        "evaluation": {
            "accounting_engine": accounting_engine,
            "capital": capital,
            "fold_table_sha256": fold_table_sha256,
        },
        "costs": {"stack": "retail_v1", "cost_refresh": "per_fold"},
        "coverage": {"untested_conclusions": [], "bias_status": {"state": "unquantified"}},
        "versions": {"engine": "0.4.0"},
    }


def _arm(run_id: str, returns: np.ndarray, **kwargs) -> ComparisonArm:
    index = pd.RangeIndex(len(returns))
    costs = kwargs.pop("costs", None)
    weights = kwargs.pop("weights", None)
    return ComparisonArm(
        record=_record(run_id, **kwargs),
        net_returns=pd.Series(returns, index=index),
        costs=None if costs is None else pd.Series(costs, index=index),
        weights=weights,
    )


class TestUnits:
    def test_a_scalar_hurdle_is_refused_with_the_reason(self):
        with pytest.raises(TypeError, match="never compared with a yield"):
            require_return_series(0.053, "hurdle")

    def test_a_yield_becomes_a_return_series_explicitly(self):
        hurdle = constant_hurdle_returns(0.0504, 252)
        assert len(hurdle) == 252
        assert hurdle.iloc[0] == pytest.approx(0.0504 / 252)

    def test_excess_returns_are_returns_minus_returns(self):
        net = pd.Series(np.full(10, 0.001))
        cash = constant_hurdle_returns(0.0252, 10)
        assert excess_returns(net, cash) == pytest.approx(np.full(10, 0.001 - 0.0001))

    def test_misaligned_hurdle_is_refused(self):
        net = pd.Series(np.full(10, 0.001), index=pd.RangeIndex(10))
        cash = pd.Series(np.full(10, 0.0001), index=pd.RangeIndex(1, 11))
        with pytest.raises(ValueError, match="different indexes"):
            excess_returns(net, cash)

    def test_non_finite_returns_are_a_defect(self):
        with pytest.raises(ValueError, match="non-finite"):
            require_return_series(np.array([0.1, np.nan, 0.2, 0.3]), "returns")

    def test_excess_sharpe_interval_reads_returns_against_returns(self):
        rng = np.random.default_rng(3)
        net = _ar1(rng, 300, 0.0008, 0.01, 0.3)
        cash = constant_hurdle_returns(0.0252, 300)
        interval = excess_sharpe_interval(net, cash, SPEC)
        assert interval.statistic == "sharpe"
        assert interval.sharpe_a == pytest.approx(periodic_sharpe(net - cash.to_numpy()))
        assert interval.lower < interval.sharpe_a < interval.upper


class TestSharpeIntervals:
    def test_spec_refuses_a_block_shorter_than_the_decision_interval(self):
        with pytest.raises(ValueError, match="decision interval"):
            SharpeBootstrapSpec(block_bars=5, n_resamples=99, seed=1, decision_interval_bars=21)

    def test_method_block_declares_every_parameter(self):
        method = SPEC.method()
        assert method["frequency"] == "daily"
        assert method["block_bars"] == 5
        assert method["hac_lags"] == 4
        assert method["n_resamples"] == 199
        assert method["seed"] == 20260908
        assert method["interval_method"] == "studentized_circular_block_bootstrap"

    def test_difference_matches_the_reported_sharpes(self):
        rng = np.random.default_rng(11)
        a = _ar1(rng, 400, 0.0006, 0.01, 0.2)
        b = _ar1(rng, 400, 0.0002, 0.01, 0.2)
        interval = sharpe_difference_interval(a, b, SPEC)
        assert interval.difference == pytest.approx(interval.sharpe_a - interval.sharpe_b)
        assert interval.mean_difference == pytest.approx(a.mean() - b.mean())
        assert interval.lower < interval.difference < interval.upper
        assert interval.blocks_drawn == 80

    def test_arms_are_resampled_on_the_same_blocks(self):
        # With b = -a the paired difference is exactly 2 * Sharpe(a) on every resample,
        # and the HAC influence vector doubles, so a *paired* bootstrap reproduces the
        # one-sample interval scaled by two. Resampling the arms independently would not.
        rng = np.random.default_rng(5)
        a = _ar1(rng, 300, 0.0005, 0.01, 0.4)
        paired = sharpe_difference_interval(a, -a, SPEC)
        single = sharpe_interval(a, SPEC)
        assert paired.difference == pytest.approx(2.0 * single.difference, rel=1e-12)
        assert paired.standard_error == pytest.approx(2.0 * single.standard_error, rel=1e-10)
        assert paired.lower == pytest.approx(2.0 * single.lower, rel=1e-10)
        assert paired.upper == pytest.approx(2.0 * single.upper, rel=1e-10)

    def test_unequal_samples_are_refused(self):
        rng = np.random.default_rng(2)
        with pytest.raises(ValueError, match="same sample"):
            sharpe_difference_interval(rng.normal(size=200), rng.normal(size=180), SPEC)

    def test_comparing_a_series_with_itself_fails_loud(self):
        rng = np.random.default_rng(7)
        a = _ar1(rng, 200, 0.0005, 0.01, 0.2)
        with pytest.raises(BootstrapError, match="standard error"):
            sharpe_difference_interval(a, a, SPEC)

    def test_too_few_blocks_is_refused(self):
        rng = np.random.default_rng(9)
        spec = SharpeBootstrapSpec(block_bars=5, n_resamples=99, seed=1, min_blocks=40)
        with pytest.raises(BootstrapError, match="min_blocks"):
            sharpe_difference_interval(rng.normal(size=50), rng.normal(size=50), spec)


class TestIntervalCoverage:
    """Achieved coverage on AR(1) data with known dependence, nominal 95%.

    Declared before running: 120 replications at T=250, phi=0.4, block length 5 bars,
    199 resamples, true Sharpe difference zero. The measured coverage is **0.908**, three
    to four points below the nominal 0.95 (Monte Carlo SE at 120 replications is 0.020).
    A wider study on the same estimator (400 replications per cell, scratch script) put
    coverage at 0.912-0.932 for T in {250, 500, 1266} and phi in {0.0, 0.4, 0.7} with the
    block length matched to the dependence, rising to 0.947 at a 20-bar block. The
    shortfall is the finite-sample behaviour of a studentized bootstrap on a ratio
    statistic, it is not removed by more resamples or a longer HAC lag, and it is reported
    rather than absorbed: a nominal 95% interval from this estimator covers about 91-93%
    at these sample sizes.
    """

    NOMINAL = 0.95
    ACHIEVED = 109 / 120  # measured: 109 of 120 replications covered (0.908)

    @staticmethod
    def _coverage(reps: int, n_periods: int, phi: float, block_bars: int, seed: int) -> float:
        rng = np.random.default_rng(seed)
        hits = 0
        for rep in range(reps):
            a = _ar1(rng, n_periods, 0.0005, 0.01, phi)
            b = _ar1(rng, n_periods, 0.0005, 0.01, phi)  # same true Sharpe: true difference 0
            spec = SharpeBootstrapSpec(block_bars=block_bars, n_resamples=199, seed=seed + rep)
            interval = sharpe_difference_interval(a, b, spec)
            hits += int(interval.lower <= 0.0 <= interval.upper)
        return hits / reps

    def test_achieved_coverage_is_reported_and_within_a_declared_band(self):
        achieved = self._coverage(reps=120, n_periods=250, phi=0.4, block_bars=5, seed=20260908)
        assert achieved == pytest.approx(self.ACHIEVED, abs=1e-9), (
            f"coverage moved to {achieved:.3f} from the recorded {self.ACHIEVED:.3f}; the estimator or the "
            "resampling stream changed"
        )
        assert 0.85 <= achieved <= 0.99, f"achieved coverage {achieved:.3f} against nominal {self.NOMINAL}"

    def test_ignoring_dependence_destroys_coverage(self):
        # phi = 0.7 with a one-bar block is the "resample as if iid" configuration.
        matched = self._coverage(reps=60, n_periods=250, phi=0.7, block_bars=10, seed=4242)
        ignored = self._coverage(reps=60, n_periods=250, phi=0.7, block_bars=1, seed=4242)
        assert matched >= 0.85, f"matched-block coverage {matched:.3f}"
        assert ignored <= 0.70, f"one-bar-block coverage {ignored:.3f} — dependence is being carried by accident"
        assert matched - ignored >= 0.20


class TestRecordAccess:
    def test_fields_are_found_one_level_in(self):
        record = _record("run-1")
        assert record_field(record, "run_id") == "run-1"
        assert record_field(record, "accounting_engine") == "historical_constant_weight"
        assert record_field(record, "missing", "fallback") == "fallback"
        with pytest.raises(KeyError):
            record_field(record, "missing")


class TestComparisonRefusals:
    @staticmethod
    def _returns(seed: int, n: int = 300) -> np.ndarray:
        return _ar1(np.random.default_rng(seed), n, 0.0005, 0.01, 0.3)

    def test_one_arm_is_not_a_comparison(self):
        with pytest.raises(ComparisonRefused, match="at least two arms"):
            compare_reevaluation([_arm("run-1", self._returns(1))], SPEC)

    def test_engineering_channel_may_not_be_cited(self):
        arms = [
            _arm("run-1", self._returns(1)),
            _arm("run-2", self._returns(2), run_channel="engineering"),
        ]
        with pytest.raises(ComparisonRefused, match="engineering channel"):
            compare_reevaluation(arms, SPEC)

    def test_a_legacy_row_is_never_comparable(self):
        arms = [
            _arm("run-1", self._returns(1)),
            _arm("imported-1", self._returns(2), config_id=None),
        ]
        with pytest.raises(ComparisonRefused, match="legacy config_id"):
            compare_reevaluation(arms, SPEC)

    def test_two_visibility_modes_are_two_evidence_classes(self):
        arms = [
            _arm("run-1", self._returns(1), visibility_mode="replay_recorded"),
            _arm("run-2", self._returns(2), visibility_mode="research_vintage"),
        ]
        with pytest.raises(ComparisonRefused, match="visibility modes"):
            compare_reevaluation(arms, SPEC)

    def test_different_samples_are_refused(self):
        arms = [_arm("run-1", self._returns(1, 300)), _arm("run-2", self._returns(2, 280))]
        with pytest.raises(ComparisonRefused, match="same-sample"):
            compare_reevaluation(arms, SPEC)

    def test_different_fold_geometry_is_refused(self):
        arms = [
            _arm("run-1", self._returns(1)),
            _arm("run-2", self._returns(2), fold_table_sha256="other-sha"),
        ]
        with pytest.raises(ComparisonRefused, match="fold geometries"):
            compare_reevaluation(arms, SPEC)

    def test_duplicate_run_id_is_refused(self):
        arms = [_arm("run-1", self._returns(1)), _arm("run-1", self._returns(2))]
        with pytest.raises(ComparisonRefused, match="appears twice"):
            compare_reevaluation(arms, SPEC)

    def test_an_unlabeled_record_is_refused(self):
        record = _record("run-2")
        del record["flags"]["fill_source"]
        arms = [
            _arm("run-1", self._returns(1)),
            ComparisonArm(record=record, net_returns=pd.Series(self._returns(2))),
        ]
        with pytest.raises(ComparisonRefused, match="provenance flags"):
            compare_reevaluation(arms, SPEC)


class TestReevaluationReport:
    @staticmethod
    def _arms() -> list[ComparisonArm]:
        rng = np.random.default_rng(21)
        index = pd.RangeIndex(300)
        arms = []
        for position, (run_id, mean, capital) in enumerate(
            [
                ("run-a", 0.0006, 100_000.0),
                ("run-b", 0.0003, 100_000.0),
                ("run-c", 0.0004, 250_000.0),
            ]
        ):
            returns = _ar1(rng, 300, mean, 0.01, 0.3)
            weights = pd.DataFrame(
                {"AAA": np.full(300, 0.5 - 0.1 * position), "BBB": np.full(300, -0.5 + 0.1 * position)},
                index=index,
            )
            arms.append(
                ComparisonArm(
                    record=_record(run_id, arm_id=run_id, capital=capital),
                    net_returns=pd.Series(returns, index=index),
                    turnover=pd.Series(np.full(300, 0.02), index=index),
                    weights=weights,
                    by_fold={"fold_0": {"gross": 0.01, "net": 0.008}},
                )
            )
        return arms

    def test_blocks_are_capital_scenarios_not_arms(self):
        report = compare_reevaluation(self._arms(), SPEC)
        capitals = [block["capital"] for block in report.blocks]
        assert sorted(capitals) == [100_000.0, 250_000.0]
        # One pair inside the 100k block, none inside the singleton 250k block.
        assert report.n_intervals == 1
        assert report.to_dict()["selection"]["n_intervals"] == 1

    def test_every_panel_carries_the_records_flags(self):
        report = compare_reevaluation(self._arms(), SPEC).to_dict()
        for arm in report["arms"]:
            assert set(arm["flags"]) == {"fill_source", "synthetic", "prospective", "run_channel"}
        difference = report["blocks"][0]["sharpe_differences"][0]
        assert difference["flags_a"]["fill_source"] == "simulated"
        assert difference["flags_b"]["run_channel"] == "evidence"

    def test_holdings_overlap_is_reported_between_the_pair(self):
        report = compare_reevaluation(self._arms(), SPEC)
        overlap = report.blocks[0]["sharpe_differences"][0]["holdings_overlap"]
        assert overlap["status"] == "computed"
        assert overlap["mean_l1_distance"] == pytest.approx(0.2)
        assert overlap["mean_name_jaccard"] == pytest.approx(1.0)

    def test_conditions_held_fixed_and_conditions_that_differ(self):
        report = compare_reevaluation(self._arms(), SPEC)
        assert report.conditions_held_fixed["accounting_engine"] == "historical_constant_weight"
        assert report.conditions_held_fixed["data_id"] == "data-1"
        assert set(report.conditions_that_differ) == {"capital"}

    def test_a_comparison_is_not_an_evidence_row(self):
        payload = compare_reevaluation(self._arms(), SPEC).to_dict()
        assert payload["is_evidence_row"] is False
        assert payload["writes_trial_row"] is False

    def test_unavailable_visibility_mode_is_disclosed(self):
        report = compare_reevaluation(self._arms(), SPEC)
        assert any("visibility_mode is unavailable" in warning for warning in report.warnings)

    def test_cash_series_adds_excess_return_intervals(self):
        arms = self._arms()
        cash = constant_hurdle_returns(0.0504, 300)
        report = compare_reevaluation(arms, SPEC, cash_returns=cash)
        excess = report.blocks[0]["excess_return_intervals"]
        assert {entry["arm"] for entry in excess} == {"run-a", "run-b"}
        assert report.n_intervals == 1 + 3


class TestRepricing:
    @staticmethod
    def _arm(engine: str = "historical_constant_weight") -> ComparisonArm:
        rng = np.random.default_rng(31)
        returns = _ar1(rng, 200, 0.0005, 0.01, 0.2)
        costs = np.full(200, 0.0002)
        return _arm_with_costs(returns, costs, engine)

    def test_zero_multiplier_recovers_the_gross_path(self):
        arm = self._arm()
        report = compare_repricing(arm, cost_multiplier=0.0)
        net = arm.net_returns.to_numpy()
        costs = arm.costs.to_numpy()
        assert report.counterfactual["mean_return"] == pytest.approx((net + costs).mean())
        assert report.counterfactual["final_equity"] == pytest.approx(float(np.prod(1.0 + net + costs)))
        assert report.deviation["mean_return"] == pytest.approx(costs.mean())

    def test_doubling_costs_moves_the_path_the_other_way(self):
        report = compare_repricing(self._arm(), cost_multiplier=2.0)
        assert report.deviation["final_equity"] < 0.0
        assert report.counterfactual["total_cost"] == pytest.approx(2.0 * report.recorded["total_cost"])

    def test_exact_only_under_the_reproduction_engine(self):
        assert compare_repricing(self._arm(), cost_multiplier=1.5).exact is True
        canonical = compare_repricing(self._arm(engine="canonical"), cost_multiplier=1.5)
        assert canonical.exact is False
        assert "not have produced" in canonical.exactness_note
        assert canonical.warnings

    def test_a_repricing_is_never_an_evidence_row(self):
        payload = compare_repricing(self._arm(), cost_multiplier=1.5).to_dict()
        assert payload["kind"] == "repricing"
        assert payload["label"] == "counterfactual"
        assert payload["is_evidence_row"] is False
        assert payload["writes_trial_row"] is False

    def test_costs_must_be_supplied_not_inferred(self):
        arm = ComparisonArm(record=_record("run-1"), net_returns=pd.Series(np.full(50, 0.001)))
        with pytest.raises(ValueError, match="charged-cost series"):
            compare_repricing(arm, cost_multiplier=0.5)

    def test_negative_costs_and_multipliers_are_refused(self):
        arm = self._arm()
        with pytest.raises(ValueError, match="cost_multiplier"):
            compare_repricing(arm, cost_multiplier=-1.0)
        broken = ComparisonArm(
            record=arm.record,
            net_returns=arm.net_returns,
            costs=pd.Series(np.full(200, -0.0001), index=arm.net_returns.index),
        )
        with pytest.raises(ValueError, match="positive debit"):
            compare_repricing(broken, cost_multiplier=1.0)

    def test_attaches_to_its_record_without_modifying_it(self):
        arm = self._arm()
        report = compare_repricing(arm, cost_multiplier=1.5)
        before = dict(arm.record)
        attached = attach_repricing(arm.record, report)
        assert "diagnostics" not in arm.record, "the source record was mutated"
        assert arm.record == before
        assert attached["diagnostics"]["repricings"][0]["cost_multiplier"] == 1.5
        assert attached["ids"] == arm.record["ids"]

    def test_will_not_attach_to_a_different_record(self):
        report = compare_repricing(self._arm(), cost_multiplier=1.5)
        with pytest.raises(ComparisonRefused, match="cannot attach"):
            attach_repricing(_record("some-other-run"), report)

    def test_interval_is_optional_and_paired_when_asked(self):
        report = compare_repricing(self._arm(), cost_multiplier=1.5, spec=SPEC)
        assert report.interval is not None
        assert report.interval["statistic"] == "sharpe_difference"
        assert compare_repricing(self._arm(), cost_multiplier=1.5).interval is None


def _arm_with_costs(returns: np.ndarray, costs: np.ndarray, engine: str) -> ComparisonArm:
    index = pd.RangeIndex(len(returns))
    return ComparisonArm(
        record=_record("run-priced", accounting_engine=engine),
        net_returns=pd.Series(returns, index=index),
        costs=pd.Series(costs, index=index),
    )
