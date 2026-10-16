"""Sharpe-ratio inference on synthetic data: intervals, PSR and DSR, PBO, and an anytime monitor.

Synthetic data; demonstrates the API only; makes no claim about any strategy.

Run it from a checkout, with the package installed (``uv sync``) or with ``PYTHONPATH=src``::

    python examples/statistics/sharpe_inference.py

Every series is drawn from one seeded numpy generator, so the output repeats exactly for a
given numpy version. It takes a few seconds, reads no files and uses no network.
``docs/statistics.md`` says what each function estimates, what it assumes, and its limits.

The four sections:

1. ``sharpe_interval`` and ``sharpe_difference_interval``: a studentized circular
   block-bootstrap interval for the Sharpe ratio of one return series, and for the
   difference between two series measured over the same days.
2. ``probabilistic_sharpe_ratio`` and ``deflated_sharpe_ratio``: the probability that the
   true Sharpe ratio is above zero, before and after accounting for picking the best of
   20 configurations.
3. ``probability_backtest_overfitting``: how often the in-sample winner of those 20
   configurations lands in the bottom half out of sample.
4. ``empirical_bernstein_monitor_read``: a confidence sequence for the mean daily return
   against a cash hurdle, valid however often it is checked.
"""

from __future__ import annotations

import math

import numpy as np

from prism.validation import (
    SharpeBootstrapSpec,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_with_n,
    empirical_bernstein_monitor_read,
    expected_max_sharpe,
    periodic_sharpe,
    probabilistic_sharpe_ratio,
    probability_backtest_overfitting,
    sharpe_difference_interval,
    sharpe_interval,
)

SEED = 7
N_DAYS = 2520  # ten years of trading days
PERIODS_PER_YEAR = 252
ANNUALIZE = math.sqrt(PERIODS_PER_YEAR)
DAILY_VOL = 0.01
TRUE_SHARPE_A = 0.8  # annualized
TRUE_SHARPE_B = 0.4  # annualized
N_CONFIGURATIONS = 20
CASH_RATE = 0.04  # annual cash yield used as the monitor's hurdle
RETURN_BOUND = 0.1  # assumed support of a daily return for the monitor, fixed before looking

HEADER = "Synthetic data; demonstrates the API only; makes no claim about any strategy."
SECTIONS = (
    "1. Sharpe-ratio intervals: studentized circular block bootstrap",
    "2. Probabilistic and deflated Sharpe ratio",
    "3. Probability of backtest overfitting",
    "4. Anytime-valid monitor against a cash hurdle",
)


def correlated_ar1(
    rng: np.random.Generator,
    n_days: int,
    annual_sharpes: tuple[float, ...],
    correlation: float,
    phi: float,
) -> np.ndarray:
    """``(n_days, k)`` daily returns: AR(1) with coefficient ``phi``, equicorrelated innovations.

    Each column has stationary standard deviation ``DAILY_VOL`` and stationary mean
    ``annual_sharpe * DAILY_VOL / sqrt(252)``, so its true annualized Sharpe ratio is the
    one given.
    """
    k = len(annual_sharpes)
    cov = np.full((k, k), correlation) + (1.0 - correlation) * np.eye(k)
    scale = DAILY_VOL * math.sqrt(1.0 - phi**2)
    innovations = rng.multivariate_normal(np.zeros(k), cov, size=n_days) * scale
    means = np.asarray(annual_sharpes) * DAILY_VOL / ANNUALIZE
    out = np.empty((n_days, k))
    out[0] = means + innovations[0] / math.sqrt(1.0 - phi**2)
    for t in range(1, n_days):
        out[t] = means + phi * (out[t - 1] - means) + innovations[t]
    return out


def trial_matrix(rng: np.random.Generator, n_days: int, annual_sharpes: np.ndarray) -> np.ndarray:
    """``(n_days, n)`` independent normal daily returns for ``n`` configurations.

    Independent trials are what the expected-maximum formula behind the DSR assumes;
    configurations from one real search are correlated, and then their effective number
    is below ``n`` (``docs/statistics.md``, limits).
    """
    noise = rng.normal(0.0, DAILY_VOL, size=(n_days, annual_sharpes.size))
    return annual_sharpes[None, :] * DAILY_VOL / ANNUALIZE + noise


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(HEADER)

    # 1. Two series measured over the same 2,520 days: true annualized Sharpe ratios 0.8 and
    # 0.4, innovation correlation 0.6, AR(1) coefficient 0.1. Blocks of 10 days carry the
    # serial dependence into the resamples; 999 resamples keep the run to a few seconds.
    section(SECTIONS[0])
    pair = correlated_ar1(rng, N_DAYS, annual_sharpes=(TRUE_SHARPE_A, TRUE_SHARPE_B), correlation=0.6, phi=0.1)
    series_a, series_b = pair[:, 0], pair[:, 1]
    spec = SharpeBootstrapSpec(block_bars=10, n_resamples=999, seed=SEED)
    one = sharpe_interval(series_a, spec, label="series_a")
    diff = sharpe_difference_interval(series_a, series_b, spec, label="a_minus_b")
    daily_true_a = TRUE_SHARPE_A / ANNUALIZE
    iid_se = math.sqrt((1.0 + 0.5 * daily_true_a**2) / N_DAYS) * ANNUALIZE
    print(f"True annualized Sharpe ratios: a {TRUE_SHARPE_A:.2f}, b {TRUE_SHARPE_B:.2f}.")
    print(f"Sample annualized Sharpe ratios: a {diff.sharpe_a * ANNUALIZE:.3f}, b {diff.sharpe_b * ANNUALIZE:.3f}")
    print(
        f"Standard error of one annualized Sharpe ratio over ten years of iid normal returns: "
        f"about {iid_se:.2f}, so sample values this far from the true ones are ordinary."
    )
    print(
        f"Sharpe(a), 95% interval, annualized: [{one.lower * ANNUALIZE:.3f}, {one.upper * ANNUALIZE:.3f}]; "
        f"p-value against zero {one.p_value_two_sided:.4f}"
    )
    print(
        f"Sharpe(a) - Sharpe(b): {diff.difference * ANNUALIZE:.3f}, 95% interval "
        f"[{diff.lower * ANNUALIZE:.3f}, {diff.upper * ANNUALIZE:.3f}], "
        f"p-value {diff.p_value_two_sided:.4f}"
    )
    print(
        f"Bootstrap: {diff.n_resamples} resamples of {diff.blocks_drawn} blocks of {spec.block_bars} days, "
        f"HAC lag {spec.lags}, seed {spec.seed}."
    )

    # 2. PSR of series a against zero, then a search: 20 configurations with no true edge,
    # of which the best in-sample Sharpe ratio is selected. Its PSR ignores the search; the
    # DSR replaces the zero benchmark with the Sharpe ratio the best of 20 skill-less trials
    # would reach by luck.
    section(SECTIONS[1])
    print(f"PSR(series a > 0): {probabilistic_sharpe_ratio(series_a):.4f}")
    null_trials = trial_matrix(rng, N_DAYS, np.zeros(N_CONFIGURATIONS))
    trial_sharpes = np.array([periodic_sharpe(null_trials[:, j]) for j in range(N_CONFIGURATIONS)])
    best = int(np.argmax(trial_sharpes))
    selected = null_trials[:, best]
    print(
        f"{N_CONFIGURATIONS} configurations, every true Sharpe ratio zero; best sample Sharpe ratio "
        f"{trial_sharpes[best] * ANNUALIZE:.3f} annualized (configuration {best})."
    )
    print(f"PSR(best > 0), ignoring the search: {probabilistic_sharpe_ratio(selected):.4f}")
    expected_max = expected_max_sharpe(trial_sharpes) * ANNUALIZE
    print(f"Expected maximum Sharpe ratio of {N_CONFIGURATIONS} skill-less trials, annualized: {expected_max:.3f}")
    dsr = deflated_sharpe_ratio(selected, trial_sharpes)
    print(f"DSR(best), deflated against the {N_CONFIGURATIONS} trials: {dsr:.4f}")
    dsr_200 = deflated_sharpe_ratio_with_n(selected, trial_sharpes, n_trials=200)
    print(f"DSR(best), had 200 configurations been searched and only these 20 kept: {dsr_200:.4f}")

    # 3. PBO by combinatorially symmetric cross-validation, 10 slices (252 splits). With no
    # true edge anywhere the in-sample winner is a coin flip out of sample; with one
    # configuration carrying a real edge it is not.
    section(SECTIONS[2])
    print(f"PBO, {N_CONFIGURATIONS} configurations with no edge: {probability_backtest_overfitting(null_trials):.3f}")
    one_edge = trial_matrix(rng, N_DAYS, np.array([1.0] + [0.0] * (N_CONFIGURATIONS - 1)))
    print(
        f"PBO, one of {N_CONFIGURATIONS} configurations with true annualized Sharpe ratio 1.0: "
        f"{probability_backtest_overfitting(one_edge):.3f}"
    )

    # 4. Empirical-Bernstein confidence sequence for series a's mean daily return, read
    # against a 4% annual cash rate. The reads at 1, 5 and 10 years are all valid together:
    # coverage holds uniformly over every day the sequence could have been checked. The
    # fixed-sample interval beside it is valid only at one sample size chosen in advance.
    section(SECTIONS[3])
    hurdle = CASH_RATE / PERIODS_PER_YEAR
    true_mean = TRUE_SHARPE_A * DAILY_VOL / ANNUALIZE
    print(
        f"Hurdle: {CASH_RATE:.0%} a year = {hurdle * 1e4:.2f} bp a day. True mean of series a: "
        f"{true_mean * 1e4:.2f} bp a day, an edge of {(true_mean - hurdle) * 1e4:.2f} bp a day."
    )
    for days in (252, 1260, N_DAYS):
        window = series_a[:days]
        read = empirical_bernstein_monitor_read(window, bound=RETURN_BOUND, hurdle=hurdle)
        mean_bp, lower_bp, upper_bp = (float(read[key]) * 1e4 for key in ("mean", "ci_lower", "ci_upper"))
        fixed_half = 1.96 * window.std(ddof=1) / math.sqrt(days)
        print(
            f"After {days:>4} days: mean {mean_bp:6.2f} bp, sequence [{lower_bp:6.2f}, {upper_bp:6.2f}] bp, "
            f"verdict {read['verdict']}; fixed-sample 95% half-width {fixed_half * 1e4:.2f} bp; "
            f"{read['n_winsorised']} winsorised"
        )
    print(
        "An inconclusive read is the expected answer here. An edge of a few basis points a day "
        "against 1% daily volatility is about the half-width of a fixed-sample interval even at "
        "ten years, and a sequence valid at every day is wider still."
    )


if __name__ == "__main__":
    main()
