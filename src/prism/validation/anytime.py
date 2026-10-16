"""Anytime-valid confidence sequences for the mean per-period return of a live or paper stream.

The deflated-Sharpe machinery in :mod:`prism.validation.metrics` corrects for
*cross-sectional* multiplicity — the many trials searched in a sweep. It does
**not** cover the *temporal* multiplicity of a live monitor: looking at the
paper-loop equity stream every day and stopping the first day it looks good (or
bad) inflates type-I error exactly the way uncounted trials do. A fixed-sample
statistic (PSR at a chosen horizon) is only valid at the *one* horizon it was
computed for; peeking at it daily and acting on the peek is not.

A *confidence sequence* is the temporal analogue of the DSR's selection-set
deflation: an interval that covers the true mean *uniformly over all sample
sizes at once*, so you may inspect it after every new day and stop at any
data-dependent time without breaking the 1 − α guarantee. This module gives the
two-sided normal-mixture confidence sequence (Robbins 1970; Howard, Ramdas,
McAuliffe & Sekhon 2021, *Time-uniform, nonparametric, nonasymptotic confidence
sequences*) and the variance-adaptive empirical-Bernstein sequence (Waudby-Smith
and Ramdas, *Estimating means of bounded random variables by betting*), both
specialised to a bounded (winsorised) per-period net return. Usage, assumptions and
limits: ``docs/statistics.md``.

**Governance.** This is *additive telemetry* run beside the ratified rolling-PSR
promotion read of ``docs/momentum_design.md``. It changes no ratified statistic
and introduces no counted trial, so it needs no new pre-registration. It becomes
a binding promotion/kill read only if a future program's pre-registration adopts
it — at which point the estimand (mean per-period net return vs. a stated hurdle)
is itself the pre-registered quantity.

**The power cost is real and by design.** Time-uniform coverage is bought with a
union bound over every stopping time, so the interval is strictly wider at every
fixed ``n`` than a fixed-sample CI. For an edge as marginal as anything in this
book, the sequence may not exclude the hurdle within the paper-loop horizon —
that is the *correct* answer when the edge is that thin, not a defect. A
``verdict`` of ``"inconclusive"`` after months is information, not a bug.
"""

from __future__ import annotations

import logging
import math
from typing import Literal, Union, overload

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, pd.Series]

# Default sub-Gaussian support for a per-period *net return* (fraction). A book's
# daily NAV return does not move ±50% in a day, so the default never clips real
# data; it is deliberately loose (wide interval, zero clipping bias). Tightening
# ``bound`` toward the true return scale (e.g. 0.1) sharply narrows the sequence
# — the coverage guarantee is preserved by winsorisation regardless of ``bound``,
# the only cost of too tight a bound is clipping bias on genuine large moves.
DEFAULT_RETURN_BOUND = 0.5

# Default mixture-tuning horizon (trading days). The sequence is tightest near
# this sample size; validity is INDEPENDENT of it (any positive mixture variance
# yields a valid supermartingale), so this trades tightness across the horizon,
# never coverage.
DEFAULT_OPT_HORIZON = 252


def _finite(returns: ArrayLike) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    return arr[np.isfinite(arr)]


def anytime_confidence_sequence(
    returns: ArrayLike,
    *,
    alpha: float = 0.05,
    bound: float = DEFAULT_RETURN_BOUND,
    opt_horizon: int = DEFAULT_OPT_HORIZON,
) -> tuple[float, float]:
    """Two-sided normal-mixture confidence sequence for the mean per-period return.

    Returns ``(lower, upper)`` — an interval that, evaluated at any
    data-dependent stopping time, covers the true mean of the (winsorised)
    per-period return with probability at least ``1 - alpha`` *uniformly over all
    sample sizes*. This is the anytime-valid analogue of a confidence interval:
    safe to recompute and act on after every new observation.

    Construction. For a candidate mean ``μ`` and increments winsorised to
    ``[-bound, bound]`` (Hoeffding sub-Gaussian parameter ``σ = bound``), the
    normal mixture over the exponential test-supermartingale
    ``exp(λ Σ(Xᵢ−μ) − λ²σ²t/2)`` against a ``N(0, τ²)`` prior on ``λ`` is the
    nonnegative supermartingale

        ``M_t(μ) = (1 + τ²σ²t)^(−1/2) · exp( τ²(Σ(Xᵢ−μ))² / (2(1 + τ²σ²t)) )``

    with ``M_0 = 1``. By Ville's inequality ``P(∃t : M_t(μ) ≥ 1/α) ≤ α``, so the
    never-rejected set ``{μ : M_t(μ) < 1/α}`` is a level-``(1−α)`` confidence
    sequence. Solving the inequality gives the closed form
    ``mean ± R_t / t`` with
    ``R_t = sqrt( (2A/τ²)·(½ ln A + ln(1/α)) )`` and ``A = 1 + τ²σ²t``.
    ``τ² = 1/(σ²·opt_horizon)`` tunes tightness near ``opt_horizon``.

    ``bound`` is the assumed support: values outside ``[-bound, bound]`` are
    winsorised (with a warning) so the sub-Gaussian assumption — and hence
    coverage — holds by construction; the inference is then about the winsorised
    mean, which equals the true mean whenever no clipping occurs.

    Returns ``(nan, nan)`` for an empty / all-non-finite sample (n < 1). Raises
    ``ValueError`` on ``alpha ∉ (0, 1)``, ``bound ≤ 0``, or ``opt_horizon < 1``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (bound > 0.0) or not np.isfinite(bound):
        raise ValueError(f"bound must be a positive finite number, got {bound}")
    if opt_horizon < 1:
        raise ValueError(f"opt_horizon must be >= 1, got {opt_horizon}")

    arr = _finite(returns)
    n = arr.size
    if n < 1:
        return (float("nan"), float("nan"))

    clipped = np.clip(arr, -bound, bound)
    n_clipped = int(np.count_nonzero(clipped != arr))
    if n_clipped:
        logger.warning(
            "anytime_confidence_sequence winsorised %d/%d observation(s) to ±%g; "
            "the interval covers the winsorised mean (tighten or widen `bound` deliberately)",
            n_clipped,
            n,
            bound,
        )

    sigma = float(bound)
    tau2 = 1.0 / (sigma * sigma * opt_horizon)
    mean = float(clipped.mean())

    a = 1.0 + tau2 * sigma * sigma * n  # A = 1 + τ²σ²t
    # bracket = ½ ln A + ln(1/α) > 0 for all A >= 1, alpha in (0,1).
    bracket = 0.5 * math.log(a) - math.log(alpha)
    radius_sum = math.sqrt((2.0 * a / tau2) * bracket)  # R_t on the SUM scale
    half_width = radius_sum / n  # back to the mean scale

    return (mean - half_width, mean + half_width)


def anytime_monitor_read(
    returns: ArrayLike,
    *,
    alpha: float = 0.05,
    bound: float = DEFAULT_RETURN_BOUND,
    opt_horizon: int = DEFAULT_OPT_HORIZON,
    hurdle: float = 0.0,
) -> dict[str, float | int | bool | str]:
    """Anytime-valid monitor verdict on whether mean per-period net return beats ``hurdle``.

    Runs :func:`anytime_confidence_sequence` and reads it against ``hurdle``, which is
    a *per-period net return* in the units of ``returns`` — 0.0 tests "does the book
    make money net", and the per-period accrual of a cash rate (for example the T-bill
    yield divided by 252) tests it against its opportunity cost. Returns are compared
    with returns here, as in :mod:`prism.validation.compare`;
    ``after_cost_hurdle_periodic`` in :mod:`prism.validation.metrics` restates a yield
    in *Sharpe* units and is therefore not what this argument takes:

    * ``edge_confirmed`` — the whole sequence is above ``hurdle`` (lower > hurdle):
      the strategy beats the bar at level ``1 − alpha``, valid at this stopping time.
    * ``edge_refuted`` — the whole sequence is below ``hurdle`` (upper < hurdle):
      a kill signal, valid at this stopping time.
    * otherwise ``inconclusive`` — the interval straddles ``hurdle``; keep accruing.

    Because coverage is time-uniform, this may be recomputed after every new day
    and the first crossing acted on without inflating error — the property a
    daily-inspected PSR read lacks. Returns a JSON-friendly dict so it can be
    logged beside the rolling-PSR read or embedded in a monitor artifact.
    """
    lower, upper = anytime_confidence_sequence(returns, alpha=alpha, bound=bound, opt_horizon=opt_horizon)
    arr = _finite(returns)
    n = int(arr.size)
    mean = float(np.clip(arr, -bound, bound).mean()) if n else float("nan")

    edge_confirmed = bool(np.isfinite(lower) and lower > hurdle)
    edge_refuted = bool(np.isfinite(upper) and upper < hurdle)
    if edge_confirmed:
        verdict = "confirmed"
    elif edge_refuted:
        verdict = "refuted"
    else:
        verdict = "inconclusive"

    return {
        "n": n,
        "mean": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "alpha": float(alpha),
        "hurdle": float(hurdle),
        "edge_confirmed": edge_confirmed,
        "edge_refuted": edge_refuted,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Variance-adaptive confidence sequence (empirical Bernstein)
# ---------------------------------------------------------------------------

# Cap on the predictable bet size in the empirical-Bernstein sequence. The
# theorem allows any c in (0, 1); the width floor log(2/alpha) / (c * n) favours a
# larger c when the observations' variance is small relative to their support,
# which is the live monitor's situation (daily NAV volatility ~0.7% against a
# 10% support). Waudby-Smith and Ramdas use 1/2 and 3/4 in their experiments.
DEFAULT_EB_LAMBDA_MAX = 0.75


def _psi_e(lam: np.ndarray) -> np.ndarray:
    """psi_E(lambda) = (-log(1 - lambda) - lambda) / 4, the empirical-Bernstein CGF bound."""
    return (-np.log1p(-lam) - lam) / 4.0


@overload
def empirical_bernstein_running_intervals(
    paths: np.ndarray,
    *,
    alpha: float = ...,
    bound: float = ...,
    lambda_max: float = ...,
    return_center: Literal[False] = ...,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def empirical_bernstein_running_intervals(
    paths: np.ndarray,
    *,
    alpha: float = ...,
    bound: float = ...,
    lambda_max: float = ...,
    return_center: Literal[True],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def empirical_bernstein_running_intervals(
    paths: np.ndarray,
    *,
    alpha: float = 0.05,
    bound: float = DEFAULT_RETURN_BOUND,
    lambda_max: float = DEFAULT_EB_LAMBDA_MAX,
    return_center: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The empirical-Bernstein confidence sequence at every sample size, for many paths at once.

    ``paths`` is a 2-D array of shape ``(n_paths, n_steps)`` of per-period returns;
    the result is a pair of arrays of that shape holding the lower and upper
    endpoints of the sequence after each step. This is the vectorised core that
    :func:`empirical_bernstein_confidence_sequence` reads its last column from
    and that the coverage test checks at every stopping time. Non-finite entries
    are not accepted here (the public function drops them first). With
    ``return_center`` a third array, the unclipped bet-weighted centre
    ``sum lambda_i X_i / sum lambda_i`` in return units, is returned as well; the
    endpoints are clipped to the support, so their midpoint is not that centre
    while either endpoint is clipped.

    Construction (Waudby-Smith and Ramdas 2023, *Estimating means of bounded
    random variables by betting*, JRSS-B, Theorem 2, the predictable plug-in
    empirical-Bernstein confidence sequence, with the plug-in sequences defined
    just before it). Observations are winsorised to ``[-bound, bound]`` and
    mapped to ``X_i = (r_i + bound) / (2 bound)`` in ``[0, 1]``. With

    * ``mu_t = (1/2 + sum_{i<=t} X_i) / (t + 1)`` and
      ``sigma2_t = (1/4 + sum_{i<=t} (X_i - mu_i)^2) / (t + 1)`` (``mu_0 = 1/2``,
      ``sigma2_0 = 1/4``), both predictable one step ahead,
    * ``lambda_i = min( sqrt( 2 log(2/alpha) / (sigma2_{i-1} * i * log(i + 1)) ), lambda_max )``,
    * ``v_i = 4 (X_i - mu_{i-1})^2`` and ``psi_E(lambda) = (-log(1 - lambda) - lambda) / 4``,

    the interval after ``t`` observations is

        ``sum_i lambda_i X_i / sum_i lambda_i  +/-  (log(2/alpha) + sum_i v_i psi_E(lambda_i)) / sum_i lambda_i``

    intersected with ``[0, 1]`` and mapped back to return units. Theorem 2 states
    that for any sequence with conditional means ``mu`` in ``[0, 1]`` and any
    predictable ``lambda_i`` in ``[0, 1)`` this covers ``mu`` with probability at
    least ``1 - alpha`` *simultaneously over all t*: the validity is
    nonasymptotic (no normal approximation, only the support bound) and
    time-uniform (the sequence may be inspected after every observation and
    acted on at any data-dependent time). Unlike the normal-mixture sequence
    above, whose width is set by ``bound`` alone, this one's width adapts to the
    realised variance through ``v_i``, while the term ``log(2/alpha) / sum lambda_i``
    keeps a floor of about ``2 bound log(2/alpha) / (lambda_max t)`` in return
    units, which is the price of a nonasymptotic guarantee on a bounded support.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (bound > 0.0) or not np.isfinite(bound):
        raise ValueError(f"bound must be a positive finite number, got {bound}")
    if not (0.0 < lambda_max < 1.0):
        raise ValueError(f"lambda_max must be in (0, 1), got {lambda_max}")
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"paths must be a 2-D array with at least one step, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("paths must be finite; drop non-finite observations before calling")

    x = (np.clip(arr, -bound, bound) + bound) / (2.0 * bound)  # in [0, 1]
    n_paths, n_steps = x.shape
    t = np.arange(1, n_steps + 1, dtype=float)
    log_term = math.log(2.0 / alpha)

    # Predictable plug-ins: mu_hat_t and sigma2_hat_t after t observations; the
    # bet at step i uses their values after i - 1 observations.
    mu_hat = (0.5 + np.cumsum(x, axis=1)) / (t + 1.0)
    mu_prev = np.concatenate([np.full((n_paths, 1), 0.5), mu_hat[:, :-1]], axis=1)
    sigma2_hat = (0.25 + np.cumsum((x - mu_hat) ** 2, axis=1)) / (t + 1.0)
    sigma2_prev = np.concatenate([np.full((n_paths, 1), 0.25), sigma2_hat[:, :-1]], axis=1)

    lam = np.sqrt(2.0 * log_term / (sigma2_prev * t * np.log1p(t)))
    lam = np.minimum(lam, lambda_max)
    v = 4.0 * (x - mu_prev) ** 2

    sum_lam = np.cumsum(lam, axis=1)
    center = np.cumsum(lam * x, axis=1) / sum_lam
    width = (log_term + np.cumsum(v * _psi_e(lam), axis=1)) / sum_lam
    lower = np.clip(center - width, 0.0, 1.0)
    upper = np.clip(center + width, 0.0, 1.0)
    # Back to return units: r = 2 bound X - bound.
    lower_r, upper_r = 2.0 * bound * lower - bound, 2.0 * bound * upper - bound
    if return_center:
        return lower_r, upper_r, 2.0 * bound * center - bound
    return lower_r, upper_r


def empirical_bernstein_confidence_sequence(
    returns: ArrayLike,
    *,
    alpha: float = 0.05,
    bound: float = DEFAULT_RETURN_BOUND,
    opt_horizon: int = DEFAULT_OPT_HORIZON,
    lambda_max: float = DEFAULT_EB_LAMBDA_MAX,
) -> tuple[float, float]:
    """Variance-adaptive two-sided confidence sequence for the mean per-period return.

    Returns ``(lower, upper)`` for the observations seen so far, computed by
    :func:`empirical_bernstein_running_intervals` (which holds the construction
    and the citation). Coverage is ``1 - alpha`` uniformly over all sample sizes
    and nonasymptotic for observations in ``[-bound, bound]``; values outside are
    winsorised with a warning, so the inference is about the winsorised mean,
    which equals the true mean whenever nothing is clipped.

    ``opt_horizon`` is accepted so this function is a drop-in for
    :func:`anytime_confidence_sequence`; the predictable plug-in bets have no
    horizon parameter, so beyond the same ``>= 1`` validation it has no effect.

    Returns ``(nan, nan)`` for an empty or all-non-finite sample, matching the
    normal-mixture function. Raises ``ValueError`` on ``alpha`` outside ``(0, 1)``,
    a non-positive ``bound``, ``opt_horizon < 1`` or ``lambda_max`` outside ``(0, 1)``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (bound > 0.0) or not np.isfinite(bound):
        raise ValueError(f"bound must be a positive finite number, got {bound}")
    if opt_horizon < 1:
        raise ValueError(f"opt_horizon must be >= 1, got {opt_horizon}")

    arr = _finite(returns)
    n = arr.size
    if n < 1:
        return (float("nan"), float("nan"))
    n_clipped = int(np.count_nonzero(np.clip(arr, -bound, bound) != arr))
    if n_clipped:
        logger.warning(
            "empirical_bernstein_confidence_sequence winsorised %d/%d observation(s) to ±%g; "
            "the interval covers the winsorised mean (tighten or widen `bound` deliberately)",
            n_clipped,
            n,
            bound,
        )
    lower, upper = empirical_bernstein_running_intervals(
        arr.reshape(1, -1), alpha=alpha, bound=bound, lambda_max=lambda_max
    )
    return (float(lower[0, -1]), float(upper[0, -1]))


def empirical_bernstein_monitor_read(
    returns: ArrayLike,
    *,
    alpha: float = 0.05,
    bound: float = DEFAULT_RETURN_BOUND,
    hurdle: float = 0.0,
    lambda_max: float = DEFAULT_EB_LAMBDA_MAX,
) -> dict[str, float | int | bool | str]:
    """The :func:`anytime_monitor_read` verdict dict, computed from the empirical-Bernstein sequence.

    Same keys and the same reading against ``hurdle`` (a per-period return in the
    units of ``returns``): ``confirmed`` when the whole sequence is above the
    hurdle, ``refuted`` when it is entirely below, else ``inconclusive``. Adds
    ``method``, ``bound``, ``lambda_max`` and ``n_winsorised`` so a logged read says
    which sequence produced it and whether any observation was clipped. ``mean``
    is the arithmetic mean of the winsorised observations (what the operator
    expects to see); ``cs_center`` is the sequence's own bet-weighted centre
    ``sum lambda_i X_i / sum lambda_i`` in return units, taken before the
    endpoints are clipped to the support, so at small ``n`` (where the interval
    is the whole support) it still says where the bets sit rather than ``0``.
    """
    lower, upper = empirical_bernstein_confidence_sequence(returns, alpha=alpha, bound=bound, lambda_max=lambda_max)
    arr = _finite(returns)
    n = int(arr.size)
    clipped = np.clip(arr, -bound, bound)
    mean = float(clipped.mean()) if n else float("nan")
    if n:
        _, _, centre = empirical_bernstein_running_intervals(
            clipped.reshape(1, -1), alpha=alpha, bound=bound, lambda_max=lambda_max, return_center=True
        )
        cs_center = float(centre[0, -1])
    else:
        cs_center = float("nan")
    edge_confirmed = bool(np.isfinite(lower) and lower > hurdle)
    edge_refuted = bool(np.isfinite(upper) and upper < hurdle)
    verdict = "confirmed" if edge_confirmed else ("refuted" if edge_refuted else "inconclusive")
    return {
        "n": n,
        "mean": mean,
        "cs_center": cs_center,
        "ci_lower": lower,
        "ci_upper": upper,
        "alpha": float(alpha),
        "hurdle": float(hurdle),
        "bound": float(bound),
        "lambda_max": float(lambda_max),
        "n_winsorised": int(np.count_nonzero(clipped != arr)),
        "method": "empirical_bernstein",
        "edge_confirmed": edge_confirmed,
        "edge_refuted": edge_refuted,
        "verdict": verdict,
    }
