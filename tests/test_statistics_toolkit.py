"""The statistics toolkit's public surface, its published reference values, and its example.

* Every function ``docs/statistics.md`` documents is importable from ``prism.validation``
  and is the object defined in its submodule.
* The numerical example of Bailey and López de Prado (2014), *The deflated Sharpe ratio*,
  section "A numerical example" (working paper of 31 July 2014, SSRN 2460551, pages 9-10),
  is reproduced to the four decimals the paper prints: SR0 = 0.1132 and DSR = 0.9004 at
  N = 100; DSR = 0.9505 at N = 46; DSR = 0.9505 at N = 88 with normal returns.
* ``examples/statistics/sharpe_inference.py`` runs and prints every section.
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import pytest

import prism.validation as validation
from prism.validation import anytime, compare, metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "statistics" / "sharpe_inference.py"

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

EXPORTS = {
    compare: (
        "SharpeBootstrapSpec",
        "SharpeInterval",
        "constant_hurdle_returns",
        "excess_sharpe_interval",
        "sharpe_difference_interval",
        "sharpe_interval",
    ),
    metrics: (
        "breadth_diagnostic",
        "deflated_sharpe_ratio",
        "deflated_sharpe_ratio_with_n",
        "effective_breadth",
        "effective_breadth_from_cov",
        "expected_max_sharpe",
        "expected_max_sharpe_with_n",
        "information_ratio_ceiling",
        "periodic_sharpe",
        "probabilistic_sharpe_ratio",
        "probability_backtest_overfitting",
        "rank_information_coefficient",
    ),
    anytime: (
        "anytime_confidence_sequence",
        "anytime_monitor_read",
        "empirical_bernstein_confidence_sequence",
        "empirical_bernstein_monitor_read",
    ),
}


@pytest.mark.parametrize(
    ("module", "name"),
    [(module, name) for module, names in EXPORTS.items() for name in names],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_statistics_functions_are_exported_from_the_package(module, name):
    assert name in validation.__all__
    assert getattr(validation, name) is getattr(module, name)


def test_every_name_in_all_resolves():
    missing = [name for name in validation.__all__ if not hasattr(validation, name)]
    assert not missing
    assert len(set(validation.__all__)) == len(validation.__all__)


# ---------------------------------------------------------------------------
# Bailey and López de Prado (2014), "A numerical example"
# ---------------------------------------------------------------------------

#: Daily observations per year in the paper's example ("non-annualized (with 250 observations
#: per year)"), and its sample length: five years of daily returns.
PAPER_PERIODS_PER_YEAR = 250
PAPER_T = 1250
#: The selected configuration's annualized Sharpe ratio, 2.5, de-annualized as the paper does.
PAPER_SR = 2.5 / math.sqrt(PAPER_PERIODS_PER_YEAR)
#: Variance across trials of the annualized Sharpe ratios, 1/2, de-annualized the same way.
PAPER_V = 0.5 / PAPER_PERIODS_PER_YEAR
#: Half a unit in the fourth decimal: the paper prints four decimals.
PRINTED = 5e-5


def _trial_sharpes(n_trials: int, variance: float) -> np.ndarray:
    """``n_trials`` Sharpe ratios whose ``ddof=1`` variance is exactly ``variance``.

    The functions take the trial Sharpe ratios, not their variance, and compute the variance
    with ``ddof=1``; any set with that variance gives the same expected maximum.
    """
    z = np.linspace(-1.0, 1.0, n_trials)
    z = (z - z.mean()) / z.std(ddof=1)
    return z * math.sqrt(variance)


def test_expected_maximum_sharpe_matches_the_paper():
    """SR0 = sqrt(1/(2*250)) * ((1 - gamma) Z^-1[1 - 1/100] + gamma Z^-1[1 - e^-1/100]) ~= 0.1132."""
    sr0 = metrics.expected_max_sharpe(_trial_sharpes(100, PAPER_V))
    assert sr0 == pytest.approx(0.1132, abs=PRINTED)
    # The declared-N form gives the same benchmark from two trials with the same dispersion.
    declared = metrics.expected_max_sharpe_with_n(_trial_sharpes(2, PAPER_V), n_trials=100)
    assert declared == pytest.approx(sr0, rel=1e-12)


@pytest.mark.parametrize(
    ("n_trials", "skewness", "kurtosis", "printed"),
    [
        pytest.param(100, -3.0, 10.0, 0.9004, id="example-N100"),
        pytest.param(46, -3.0, 10.0, 0.9505, id="example-N46"),
        pytest.param(88, 0.0, 3.0, 0.9505, id="normal-returns-N88"),
    ],
)
def test_deflated_sharpe_ratio_matches_the_paper(n_trials, skewness, kurtosis, printed):
    """The DSR from the paper's summary statistics.

    The paper gives moments, not a return series, so this reads the PSR core that
    ``deflated_sharpe_ratio`` and ``probabilistic_sharpe_ratio`` share. No series of 1,250
    returns has exactly skewness -3 and kurtosis 10: kurtosis = skewness^2 + 1 is attained
    only by a two-point distribution, whose weights here are irrational. The end-to-end test
    below goes through the public function with the nearest such series.

    The paper's 0.9004 is computed with the unrounded SR0 (0.113172); substituting the
    printed 0.1132 would give 0.9003.
    """
    sr0 = metrics.expected_max_sharpe(_trial_sharpes(n_trials, PAPER_V))
    dsr = metrics._psr_from_stats(PAPER_SR, PAPER_T, sr0, skewness, kurtosis)
    assert dsr == pytest.approx(printed, abs=PRINTED)


@pytest.mark.parametrize(
    ("last_n_above", "skewness", "kurtosis"),
    [pytest.param(46, -3.0, 10.0, id="example"), pytest.param(88, 0.0, 3.0, id="normal-returns")],
)
def test_the_papers_trial_counts_are_the_largest_that_clear_095(last_n_above, skewness, kurtosis):
    """The paper names N = 46 (and N = 88 for normal returns) as the trial count at which the
    DSR would still have cleared 0.95. Read as "the largest such N", one more trial fails."""

    def dsr(n_trials: int) -> float:
        sr0 = metrics.expected_max_sharpe(_trial_sharpes(n_trials, PAPER_V))
        return metrics._psr_from_stats(PAPER_SR, PAPER_T, sr0, skewness, kurtosis)

    assert dsr(last_n_above) >= 0.95 > dsr(last_n_above + 1)


def _series_with_sharpe(z: np.ndarray, sharpe: float) -> np.ndarray:
    """An affine map of ``z`` whose ``ddof=1`` Sharpe ratio is ``sharpe``; moments unchanged."""
    return 0.01 * (z / z.std(ddof=1) + sharpe)


def test_public_dsr_reproduces_the_normal_returns_case_exactly():
    """1,250 returns with skewness exactly 0 and kurtosis exactly 3, through the public function.

    Four symmetric levels: 1,000 observations at +-1 and 250 at +-sqrt(6 + sqrt(50)). The
    kurtosis equation 1250 (1000 + 250 x^2) / (1000 + 250 x)^2 = 3 has root x = 6 + sqrt(50).
    """
    outer = math.sqrt(6.0 + math.sqrt(50.0))
    z = np.concatenate([np.full(500, 1.0), np.full(500, -1.0), np.full(125, outer), np.full(125, -outer)])
    returns = _series_with_sharpe(z, PAPER_SR)
    sr, n, skewness, kurtosis = metrics._sharpe_and_moments(returns)
    assert sr == pytest.approx(PAPER_SR, rel=1e-12)
    assert n == PAPER_T
    assert skewness == pytest.approx(0.0, abs=1e-12)
    assert kurtosis == pytest.approx(3.0, abs=1e-12)

    dsr = validation.deflated_sharpe_ratio(returns, _trial_sharpes(88, PAPER_V))
    assert dsr == pytest.approx(0.9505, abs=PRINTED)
    assert validation.deflated_sharpe_ratio_with_n(returns, _trial_sharpes(2, PAPER_V), 88) == pytest.approx(
        dsr, rel=1e-12
    )


def test_public_dsr_reproduces_the_main_example_from_the_nearest_series():
    """1,145 of 1,250 returns at one level and 105 at a lower one: skewness -2.9994 and kurtosis
    9.9965, the closest a 1,250-point sample comes to the paper's -3 and 10. That moves the DSR
    by about 1e-5, so the tolerance is one unit in the fourth decimal instead of half."""
    z = np.where(np.arange(PAPER_T) < 1145, 1.0, 0.0)
    returns = _series_with_sharpe(z - z.mean(), PAPER_SR)
    _, _, skewness, kurtosis = metrics._sharpe_and_moments(returns)
    assert skewness == pytest.approx(-3.0, abs=1e-3)
    assert kurtosis == pytest.approx(10.0, abs=1e-2)

    assert validation.deflated_sharpe_ratio(returns, _trial_sharpes(100, PAPER_V)) == pytest.approx(0.9004, abs=1e-4)
    assert validation.deflated_sharpe_ratio(returns, _trial_sharpes(46, PAPER_V)) == pytest.approx(0.9505, abs=1e-4)


# ---------------------------------------------------------------------------
# The example
# ---------------------------------------------------------------------------


def test_the_synthetic_example_runs_and_prints_every_section(capsys):
    spec = importlib.util.spec_from_file_location("sharpe_inference_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    out = capsys.readouterr().out
    assert out.splitlines()[0] == module.HEADER
    for title in module.SECTIONS:
        assert title in out
    assert not re.search(r"\bnan\b", out, flags=re.IGNORECASE)
