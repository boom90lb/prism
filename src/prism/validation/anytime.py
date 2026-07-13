"""Anytime-valid monitoring for the live paper loop (SPEC.md §10, I-9).

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
sequences*), specialised to a bounded (winsorised) per-period net return.

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
from typing import Union

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

    Runs :func:`anytime_confidence_sequence` and reads it against ``hurdle`` (a
    *per-period* net-return bar — 0.0 tests "does the book make money net"; pass a
    periodic T-bill/cost hurdle to test economic viability, consistent with
    ``after_cost_hurdle_periodic`` in :mod:`prism.validation.metrics`):

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
    lower, upper = anytime_confidence_sequence(
        returns, alpha=alpha, bound=bound, opt_horizon=opt_horizon
    )
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
