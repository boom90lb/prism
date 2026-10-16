# Statistics toolkit: Sharpe-ratio inference, selection-bias corrections, sequential monitors and breadth

Date: 2026-10-15. This page covers the statistics functions exported from `prism.validation`:
what each one estimates, what it assumes, and where it stops being reliable. Every function
takes plain numpy arrays or pandas objects. None of them needs a Prism workspace, a dataset
or a broker account.

A runnable tour on synthetic data is `examples/statistics/sharpe_inference.py`. It reads no
files, uses no network and finishes in a few seconds:

```sh
PYTHONPATH=src python examples/statistics/sharpe_inference.py
```

Install from a clone of this repository. `docs/depending_on_prism.md` explains why: the
distribution `prism-trading` is not on PyPI, and the PyPI name `prism` belongs to an unrelated
project. Importing `prism.validation` also loads the rest of the validation package (the trial
ledger, the walk-forward splitters and scipy), which took 0.7–0.9 s on a warm cache on the
development machine and takes longer on a cold one.

## Conventions shared by every function

- **Sharpe ratios are periodic.** A Sharpe ratio is the mean return divided by the sample
  standard deviation (`ddof=1`) at the frequency of the input, with no annualization. Multiply
  by `sqrt(252)` to annualize daily figures. Trial Sharpe ratios, benchmarks and returns passed
  to one call must all be at the same frequency.
- **Excess over what.** No function subtracts a risk-free rate for you. To read a strategy
  against cash, subtract a cash return series first (`excess_sharpe_interval`), or turn a flat
  annual yield into a per-period return series with `constant_hurdle_returns`. A Sharpe ratio
  is never compared with a yield.
- **Missing values.** The interval functions refuse non-finite input and fewer than three
  periods. The functions in `prism.validation.metrics` (PSR, DSR, PBO and the rest) silently
  drop non-finite values. Align and clean a series yourself before calling either kind.
- **Degenerate input** (constant returns, a single trial, too short a sample) gives `NaN` from
  the metrics functions and an exception from the interval functions. Neither returns a number
  that looks valid.

## 1. Sharpe-ratio intervals

| Function | Estimates |
|---|---|
| `sharpe_interval(returns, spec)` | A confidence interval for one series' Sharpe ratio, and a two-sided p-value against zero. |
| `sharpe_difference_interval(returns_a, returns_b, spec)` | A confidence interval for `Sharpe(a) - Sharpe(b)` on two series observed over the same periods. |
| `excess_sharpe_interval(net_returns, cash_returns, spec)` | `sharpe_interval` applied to `net - cash`. |
| `SharpeBootstrapSpec(block_bars, n_resamples, seed, ...)` | The bootstrap settings, all fixed before the run. |
| `SharpeInterval` | The result: point estimates, standard error, interval, t-statistic, p-value and the full method block. |

**Method.** This is a studentized (bootstrap-t) circular block bootstrap. The standard error
comes from the delta method applied to the first two raw moments, with Sharpe =
mu / sqrt(gamma - mu^2), and a Bartlett-kernel (Newey–West) long-run covariance with
`hac_lags` lags. The default is `block_bars - 1`. Each resample draws circular blocks of
`block_bars` consecutive periods. For a difference, both series are resampled on the same
blocks, which keeps their correlation. The same standard error is recomputed on every
resample, and the interval is equal-tailed in the resampled t-statistics. The p-value is
symmetric: the share of resampled |t| at least the observed |t|. The same `spec` and inputs
reproduce the same interval bit for bit.

**Assumptions.** The return series is stationary enough that blocks drawn from anywhere in the
sample look like one another. `block_bars` must be long enough to carry the serial dependence:
at least the holding period of overlapping positions, and longer for slow-decaying
autocorrelation. Nothing chooses the block length for you. Pick it before looking at the
result and record it.

**Limits.**

- *Undercoverage at moderate sample sizes.* Measured on AR(1) data with a true difference of
  zero (`tests/test_compare.py::TestIntervalCoverage`), a nominal 95% interval covered 109 of
  120 replications (0.908) at 250 periods, AR coefficient 0.4 and 5-period blocks. A wider
  study recorded in that test's docstring found 0.912–0.932 for 250 to 1,266 periods and AR
  coefficients from 0 to 0.7, and 0.947 with 20-period blocks. Read a nominal 95% interval as
  roughly 91–93% at these sample sizes. More resamples do not fix this.
- *Ignoring dependence is costly.* With AR coefficient 0.7, 1-period blocks dropped coverage
  below 0.70 in the same test file.
- *Not the Ledoit–Wolf (2008) procedure exactly.* Their test differs in four places. They use
  a prewhitened quadratic-spectral kernel for the sample standard error. In the bootstrap they
  compute the standard error from block sums, where this code reuses the Bartlett estimator.
  They use a symmetric interval, where this one is equal-tailed. They calibrate the block
  length by an algorithm, where here the user picks it. Nobody has cross-checked the output
  against the authors' R code or the R package `PeerPerformance`, so do not report it as
  "the Ledoit–Wolf test".
- *Units.* The point Sharpe uses `ddof=1` and the standard error uses population moments.
  They differ by a factor of `sqrt(T / (T - 1))`, which is 0.2% at 250 periods and smaller
  after that.
- *Cost.* Time grows with `n_resamples × T × hac_lags`. On the development machine, ten years
  of daily data with 999 resamples and 9 lags took 0.8 s for one series and 1.7 s for a
  difference, with a peak resident memory of about 370 MB.

## 2. Probabilistic and deflated Sharpe ratio

| Function | Estimates |
|---|---|
| `periodic_sharpe(returns)` | Sample Sharpe ratio at the input frequency (`ddof=1`). |
| `probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)` | PSR: the probability that the true Sharpe ratio exceeds `sr_benchmark`, given the sample's length, skewness and kurtosis (Bailey and López de Prado 2012). |
| `expected_max_sharpe(trial_sharpes)` | The expected maximum sample Sharpe ratio of N skill-less trials, with N the number of trial Sharpe ratios passed. |
| `expected_max_sharpe_with_n(trial_sharpes, n_trials)` | The same with N declared, never below the number passed. Use it when more configurations were tried than you have Sharpe ratios for. |
| `deflated_sharpe_ratio(returns, trial_sharpes)` | DSR: the PSR of the selected strategy against that expected maximum (Bailey and López de Prado 2014). |
| `deflated_sharpe_ratio_with_n(returns, trial_sharpes, n_trials)` | The DSR against a declared N. |

**Formulas.** With SR the selected strategy's periodic Sharpe ratio, T its number of returns,
g3 its skewness and g4 its kurtosis (not excess; 3 for a normal distribution):

    PSR(SR*) = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )
    SR0      = sqrt(V) * ( (1 - e_gamma) * Phi^-1(1 - 1/N) + e_gamma * Phi^-1(1 - 1/(N e)) )
    DSR      = PSR(SR0)

V is the variance of the trial Sharpe ratios (computed with `ddof=1`) and e_gamma is the
Euler–Mascheroni constant.

**Checked against the paper.** `tests/test_statistics_toolkit.py` reproduces the numerical
example of Bailey and López de Prado (2014), section "A numerical example". The inputs are
N = 100, V = 1/2 annualized, T = 1,250 daily returns, g3 = -3, g4 = 10 and an annualized SR of
2.5. The tests get SR0 = 0.1132 and DSR = 0.9004, as printed. They also get the paper's two
other figures: 0.9505 at N = 46, and 0.9505 at N = 88 with normal returns. All match to the
paper's four decimals, so no reference test is marked as an expected failure.

**Limits.**

- *Independent trials.* The expected-maximum formula is for N independent trials. Trials from
  one search are usually correlated, and then neither the raw count nor the observed
  dispersion is what the formula assumes. The error has no fixed direction in general. The
  package has no estimator for the effective number of independent trials, so choose N
  deliberately, record how you counted it, and report the DSR at more than one N if the count
  is uncertain.
- *Unreported trials.* The DSR is only as honest as `trial_sharpes` and N. Leaving failed or
  abandoned configurations out makes the DSR too high. `deflated_sharpe_ratio_with_n` can raise
  N to the number tried, but it cannot recover the dispersion of trials you did not keep.
- *Serially independent returns.* The PSR's variance term is the iid formula (Mertens 2002).
  It corrects for skewness and kurtosis, but not for autocorrelation. With autocorrelated
  returns (overlapping holdings, smoothed marks) the PSR and DSR are miscalibrated. Use the
  block-bootstrap interval in section 1 instead.
- *Sample moments.* Skewness and kurtosis are sample estimates and are noisy with fat tails;
  one large day moves them. For a high Sharpe ratio with positive skewness the variance term
  can turn non-positive. The functions then return `NaN` instead of a probability.
- *Approximation.* SR0 is the expected maximum, not an upper quantile, and the formula is an
  extreme-value approximation that is looser for small N. A DSR above 0.95 is therefore not
  an exactly calibrated 5%-level test of "the best strategy has skill".

## 3. Probability of backtest overfitting

`probability_backtest_overfitting(returns_matrix, n_splits=10)` estimates PBO by
combinatorially symmetric cross-validation (Bailey, Borwein, López de Prado and Zhu 2017). The
rows of the T × N return matrix are cut into `n_splits` contiguous slices. For every way of
picking half the slices as in-sample, the configuration with the best in-sample Sharpe ratio
is ranked out of sample. PBO is the share of splits in which it lands below the out-of-sample
median.

**Conventions to know when comparing with other implementations.**

- When N is odd, a selection that lands exactly on the median counts as not overfit.
- Tied out-of-sample Sharpe ratios are ranked by column order.
- The last `T mod n_splits` rows are dropped so the slices are equal.
- `n_splits` must be even.

**Limits.**

- The matrix must hold every configuration tried, on the same dates. A matrix of only the
  survivors understates PBO.
- The selection rule is fixed: the best in-sample Sharpe ratio.
- In-sample and out-of-sample halves are not purged at slice boundaries, so serial dependence
  leaks a little across them.
- The cost is C(n_splits, n_splits/2) evaluations: 252 at 10 slices, 924 at 12 and 12,870 at 16.

## 4. Anytime-valid monitors for a live or paper return stream

| Function | Estimates |
|---|---|
| `empirical_bernstein_confidence_sequence(returns, bound=...)` | A confidence sequence for the mean per-period return whose width adapts to the observed variance (Waudby-Smith and Ramdas 2024, Theorem 2). |
| `empirical_bernstein_monitor_read(returns, bound=..., hurdle=...)` | The same, read against a per-period hurdle return: `confirmed`, `refuted` or `inconclusive`, with the count of clipped observations. |
| `anytime_confidence_sequence(returns, bound=..., opt_horizon=...)` | A normal-mixture confidence sequence (Robbins 1970; Howard, Ramdas, McAuliffe and Sekhon 2021). Its width depends on `bound` only. |
| `anytime_monitor_read(returns, bound=..., hurdle=...)` | The same, read against a hurdle. |

**Why a confidence sequence.** A fixed-sample interval or PSR is valid at one sample size
chosen in advance. Checking it every day and acting on the first good-looking day inflates the
error rate, much as hiding trials does. A confidence sequence covers the true mean at every
sample size at once (with probability at least 1 - alpha), so it can be recomputed after every
period and acted on at any time. `tests/test_anytime.py` checks that coverage holds over every
checked stopping time by simulation, for both sequences.

**Assumptions.** Each observation lies in `[-bound, bound]` and has the same conditional mean.
Observations outside the bound are winsorized, with a warning and a count, so coverage still
holds, but for the winsorized mean. That equals the true mean only if nothing was clipped.
Fix `bound` before the stream starts. Choosing it after seeing the data changes what is being
estimated.

**Limits.**

- *Width.* Time-uniform coverage costs width. The empirical-Bernstein half-width never falls
  below about 2 × `bound` × log(2/alpha) / (`lambda_max` × t) in return units. At
  `bound = 0.1`, alpha = 0.05 and the default `lambda_max = 0.75`, that floor is about 39 bp
  after a year of daily data and 3.9 bp after ten years. An edge of a few basis points a day
  stays `inconclusive` for years. That is the correct answer for so thin an edge, not a fault.
  The normal-mixture sequence is wider still when returns are small relative to `bound`,
  because its width ignores the variance.
- *A mean, not a Sharpe ratio.* The hurdle is a per-period return, for example an annual
  T-bill yield divided by 252. It is not a Sharpe ratio.
- *A constant mean.* If the edge decays during the stream, the sequence no longer covers a
  single fixed quantity. It is a test against the hurdle, not an estimate of a drifting mean.
- `opt_horizon` has no effect on the empirical-Bernstein sequence. It is accepted only so the
  two functions can be swapped.

## 5. Breadth diagnostics

| Function | Estimates |
|---|---|
| `rank_information_coefficient(scores, forward_returns)` | Spearman rank correlation between one cross-section of scores and the next period's returns. |
| `effective_breadth(avg_correlation, n_names)` | Effective number of independent bets for N names with one common pairwise correlation: N / (1 + (N - 1) rho). |
| `effective_breadth_from_cov(cov)` | Participation ratio of the covariance spectrum, (sum lambda)^2 / sum lambda^2. It tends to 1 when one factor dominates and equals N for a flat spectrum. |
| `information_ratio_ceiling(ic, n_eff)` | The information ratio the fundamental law of active management implies, \|IC\| × sqrt(N_eff) (Grinold 1989). |
| `breadth_diagnostic(realized_sharpe, ic, n_eff)` | The ceiling, the realized share of it, and whether the realized figure exceeds it. |

**Limits.**

- *Model-dependent ceiling.* The ceiling rests on the fundamental law's assumptions: a
  constant IC across names and periods, independent bets after the covariance is accounted
  for, and an unconstrained portfolio. Real constraints lower it (Clarke, de Silva and Thorley
  2002). It moves with the covariance universe, the weighting, the horizon and the breadth
  estimator, so report those with it.
- *Exceeding it is a prompt, not a proof.* A realized Sharpe ratio above the ceiling is a
  reason to look for look-ahead, a bug or a misestimated IC. It establishes none of them.
  Staying below it is the ordinary case and is evidence of nothing.
- *Sample covariance noise.* With a sample covariance estimated from not many more periods
  than names, the eigenvalues spread out, which pushes the participation ratio down. Compute
  it on the residuals the book actually trades, over a long enough window.
- *One cross-section at a time.* `rank_information_coefficient` reads one cross-section. A
  time series of ICs is autocorrelated when forward-return windows overlap, so its plain mean
  and standard error overstate precision.

## 6. Walk-forward splitters with scikit-learn

`PurgedWalkForward` and `RollingWalkForward` accept scikit-learn's call signatures,
`split(X, y=None, groups=None)` and `get_n_splits(X, y=None, groups=None)`, so either can be
passed as `cv=` to `cross_val_score` and the other model-selection functions. The folds
depend on `len(X)` only. scikit-learn is not a dependency of the core package.

**Limits.**

- *The embargo does nothing useful in a forward-only split.* In `PurgedWalkForward`,
  `embargo_pct` removes the bars just after each test fold from later training sets. Training
  data always come before the test window here, so those bars never follow a later test
  window. The embargo therefore prevents no leakage and only discards data. The embargo in
  López de Prado's *Advances in Financial Machine Learning* (2018, chapter 7) protects k-fold
  splits, where training data can come after the test fold. Set `embargo_pct=0.0` unless you
  mean to discard those bars.
- *Purging is the control that matters.* Set `purge_horizon` to the label horizon (5 for a
  five-day forward return), so no training label overlaps the test window.
- *Fold counts can shrink.* A fold whose training set is emptied is skipped with a warning
  (or refused with `strict=True`), and `get_n_splits` on `PurgedWalkForward` still reports
  the requested count. Pass `strict=True` when the fold count matters.

## What has not been checked

- Sharpe-difference intervals against the Ledoit–Wolf R code (see section 1).
- PBO against another implementation, such as `pypbo`.
- The confidence sequences against the `confseq` package. Their coverage is checked by
  simulation only.
- Any of these functions' behaviour on the project's recorded returns. This page and its
  example use synthetic data only.

## References

- Bailey, D. H., and M. López de Prado (2012). The Sharpe ratio efficient frontier. *Journal
  of Risk* 15(2), 3–44.
- Bailey, D. H., and M. López de Prado (2014). The deflated Sharpe ratio: correcting for
  selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*
  40(5), 94–107. The working-paper version of 31 July 2014 (SSRN 2460551), retrieved
  2026-10-15 from `https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf`, is the one
  the reference test transcribes. Its numerical example is on pages 9–10.
- Bailey, D. H., J. M. Borwein, M. López de Prado and Q. J. Zhu (2017). The probability of
  backtest overfitting. *Journal of Computational Finance* 20(4), 39–69.
- Clarke, R., H. de Silva and S. Thorley (2002). Portfolio constraints and the fundamental law
  of active management. *Financial Analysts Journal* 58(5), 48–66.
- Grinold, R. C. (1989). The fundamental law of active management. *Journal of Portfolio
  Management* 15(3), 30–37.
- Howard, S. R., A. Ramdas, J. McAuliffe and J. Sekhon (2021). Time-uniform, nonparametric,
  nonasymptotic confidence sequences. *Annals of Statistics* 49(2), 1055–1080.
- Ledoit, O., and M. Wolf (2008). Robust performance hypothesis testing with the Sharpe ratio.
  *Journal of Empirical Finance* 15(5), 850–859.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4), 36–52.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002). Working paper,
  University of Basel.
- Politis, D. N., and J. P. Romano (1992). A circular block-resampling procedure for
  stationary data. In *Exploring the Limits of Bootstrap*, Wiley, 263–270.
- Robbins, H. (1970). Statistical methods related to the law of the iterated logarithm.
  *Annals of Mathematical Statistics* 41(5), 1397–1409.
- Waudby-Smith, I., and A. Ramdas (2024). Estimating means of bounded random variables by
  betting. *Journal of the Royal Statistical Society, Series B* 86(1), 1–27.
