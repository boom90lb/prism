"""The shared-slope calibration: the mapping from a standardized score to expected return.

Section 3.4 of the engine specification, and decision D6.

**What is estimated.** One pooled coefficient, through the origin, of the executable
forward return divided by its horizon-matched volatility on the standardized score:

    theta_hat = sum_{p,i} w_{p,i} z_{p,i} y_{p,i} / sum_{p,i} w_{p,i} z_{p,i}^2
    y_{p,i}   = r_{p,i} / sigma_{i,h,t_p}

pooled over the training decision periods of one fold. ``r`` is the return the fills can
actually earn -- from the open after the decision close to the open after the close
``horizon_bars`` later, with cash distributions credited -- not a close-to-close return
nobody trades (:func:`executable_returns`).

**Units, because the constructor depends on getting this right.** ``z`` is dimensionless
and ``y`` is a return divided by a volatility at the same horizon, so ``theta`` is
dimensionless and the estimand is ``E[r_{i,h}] / sigma_{i,h} = theta * z_i``. **Expected
return is recovered only by multiplying back by the name's horizon volatility.** The
objective's coefficient vector is therefore

    a_i  = z_i * sigma_{i,h}          (return units per unit of weight)
    mu_i = theta_shrunk * a_i         (expected executable return over horizon_bars)

and :func:`apply_calibration` is the only place either is formed. An objective built on
``z`` alone is wrong by a factor of order ``sigma_h`` -- 0.0722 for a 25 percent
annualized name at a 21-bar horizon -- which would silently re-weight the cross-section
by the inverse of volatility. :func:`apply_calibration` multiplies by ``sigma_h``; nothing
else in this module does, and nothing downstream may skip it.

**Shrinkage** is declared before the experiment and is not a tuned parameter:
``theta_shrunk = theta_hat * n / (n + shrink_pseudo_periods)`` with ``n`` the number of
training decision periods carrying at least one complete observation. Under the frozen
fold geometry of §3.13 (14 decision periods in a 312-bar formation window at a 21-bar
cadence, 13 of them with a complete 21-bar forward label) the reference factor is
``13 / (13 + 12) = 0.52``: a property of the declared geometry, recorded per fold.

**Endpoints.** ``confidence`` absent is the *point* arm and both endpoints equal
``theta_shrunk``. ``confidence`` present gives one-sided endpoints
``lower = quantile(1 - confidence)`` and ``upper = quantile(confidence)`` of the
resampled shrunk slope, by linear interpolation between order statistics; the registered
uncertainty arm declares 0.80. The resampling is the declared block bootstrap of
:mod:`prism.validation.uncertainty`: whole decision periods, each cross-section intact,
``circular_moving`` blocks, the slope **refit including shrinkage** on every resample with
the shrink factor held at the original ``n`` so the spread reflects slope variation and
not a moving shrink factor.

The bootstrap is run for *both* arms even though the point arm reads no quantile off it.
That is deliberate and is what makes the same-sample comparison of D6 hold by
construction: the two arms then share their excluded folds, their ``blocks_drawn``, and
their estimation-failure behaviour, and differ in exactly one thing -- which quantiles are
read. :func:`require_same_exclusion_geometry` is the executable statement of that.

**Exclusion is not failure.** A fold whose training pool holds fewer than
``min_training_periods`` decision periods, or whose bootstrap would draw fewer than
``min_blocks`` blocks, is *excluded* -- recorded with its reason, no fit -- and it is
excluded **for every arm alike**, because a fold dropped for one arm and kept for another
would break the same-sample comparison the experiment depends on
(:func:`fold_exclusion`, callable without fitting, which is how the ``decile_control`` arm
gets the identical excluded set from the registration's shared calibration family). A
non-finite estimate, a degenerate denominator, or an empty input panel is an *estimation
failure* and raises.

**Sign policy [Q9].** A finite ``theta_shrunk`` whose sign contradicts ``expected_sign``
sets ``sign_ok = False`` and ``disabled_reason = "sign_contradiction"``, disables that
component's *new* directional exposure through :func:`objective_endpoints`, which returns
``(0.0, 0.0)``, and lets the evaluation continue as a research outcome. The slope is
**never** inverted, and the point estimate stays in the record with its own sign. Existing
positions follow ``on_sign_contradiction`` in the recovery policy, not this module. A
negative ``lower`` with a correct-sign point estimate is *uncertainty*, not a
contradiction: ``sign_ok`` stays true, ``straddles_zero`` records it, and §3.7's adverse
term then sets ``a' w = 0`` on that fold -- the designed output of the declared method.

**Interpretation notes**, where §3.4 leaves a boundary unstated and this module had to
choose (both choices are arm-independent, so neither can break the same-sample property):

* A cross-section is standardized over the finite scores that survive eligibility, and
  only then are names without a complete label dropped from the regression. That is the
  order the decision-time path uses -- at ``t`` there are no labels at all -- so the fit
  standardizes the same vector the deployment does.
* A non-empty panel in which no period carries a complete observation is an *exclusion*
  (``n_periods = 0`` is below any admissible ``min_training_periods``), not a raise: it is
  the extreme of a degraded geometry. An *empty* panel -- no periods or no names at all --
  is a caller defect and raises.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri

from prism.validation.uncertainty import (
    BLOCK_SCHEMES,
    BlockBootstrapSpec,
    BootstrapResult,
    block_resample_indices,
    plan_blocks,
    validate_block_span,
)

__all__ = [
    "CALIBRATION_KINDS",
    "EXPECTED_SIGNS",
    "EXCLUSION_KEYS",
    "LABEL_DISTRIBUTIONS",
    "EXECUTABLE_RETURNS",
    "SIGMA_ESTIMATORS",
    "SIGMA_SCALINGS",
    "STANDARDIZATIONS",
    "UNCERTAINTY_SCHEMES",
    "WEIGHTINGS",
    "CalibrationError",
    "CalibrationFit",
    "CalibrationSpec",
    "FoldExclusion",
    "SigmaHSpec",
    "apply_calibration",
    "ewma_bar_sigma",
    "executable_returns",
    "fit_calibration",
    "fold_exclusion",
    "horizon_sigma",
    "load_calibration",
    "objective_endpoints",
    "require_same_exclusion_geometry",
    "require_sigma_agreement",
    "standardize_cross_section",
    "standardize_panel",
    "validate_standardize",
]

#: The only calibration kind §3.4 declares.
CALIBRATION_KINDS: tuple[str, ...] = ("shared_slope",)
#: Cross-sectional standardizations (§3.4, "Standardization").
STANDARDIZATIONS: tuple[str, ...] = ("rank_normal", "zscore", "none")
#: Pooled-regression weighting rules. ``per_observation_equal`` must be declared to be used.
WEIGHTINGS: tuple[str, ...] = ("per_date_equal", "per_observation_equal")
#: The declared sign of the component's edge.
EXPECTED_SIGNS: tuple[str, ...] = ("positive", "negative")
#: The only uncertainty scheme.
UNCERTAINTY_SCHEMES: tuple[str, ...] = ("time_block_bootstrap",)
#: The only label convention: open after the decision close to open after the horizon close.
EXECUTABLE_RETURNS: tuple[str, ...] = ("open_to_open",)
#: Whether per-share cash distributions enter the label.
LABEL_DISTRIBUTIONS: tuple[str, ...] = ("credited", "ignored")
#: Horizon-volatility estimators (§3.4, "[sigma_h]").
SIGMA_ESTIMATORS: tuple[str, ...] = ("ewma_daily",)
#: Horizon scalings; must equal the risk snapshot's ``scaling`` (§3.5).
SIGMA_SCALINGS: tuple[str, ...] = ("per_bar_times_h", "estimated_at_horizon")

#: Every declared key that decides *which folds are excluded* and *what is estimated*.
#: ``confidence`` is deliberately absent: it is the one key the three arms may differ in,
#: and D6 requires every other key to agree so the comparison stays same-sample.
EXCLUSION_KEYS: tuple[str, ...] = (
    "kind",
    "standardize",
    "weighting",
    "shrink_pseudo_periods",
    "expected_sign",
    "uncertainty",
    "block_scheme",
    "block_periods",
    "n_resamples",
    "seed",
    "min_training_periods",
    "min_blocks",
    "inner_oos_required",
    "executable_return",
    "label_distributions",
    "sigma_h",
)

_REQUIRED_KEYS: frozenset[str] = frozenset(EXCLUSION_KEYS)
_OPTIONAL_KEYS: frozenset[str] = frozenset({"confidence"})


class CalibrationError(RuntimeError):
    """A calibration that cannot be estimated, or a declaration that cannot be honoured.

    Raised for an estimation *failure* -- a non-finite slope, a degenerate denominator, an
    empty panel, a refused declaration. A fold below the declared minimums is **not** this:
    it is a :class:`FoldExclusion` carried on the fit.
    """


@dataclass(frozen=True)
class SigmaHSpec:
    """The ``[sigma_h]`` block: the horizon volatility the label, ``a``, and the risk
    snapshot must all agree on (§3.4, §3.5)."""

    estimator: str
    window_bars: int
    scaling: str

    def __post_init__(self) -> None:
        if self.estimator not in SIGMA_ESTIMATORS:
            raise CalibrationError(f"[sigma_h] estimator must be one of {SIGMA_ESTIMATORS}, got {self.estimator!r}")
        if int(self.window_bars) < 2:
            raise CalibrationError(f"[sigma_h] window_bars must be >= 2, got {self.window_bars}")
        if self.scaling not in SIGMA_SCALINGS:
            raise CalibrationError(f"[sigma_h] scaling must be one of {SIGMA_SCALINGS}, got {self.scaling!r}")
        if self.estimator == "ewma_daily" and self.scaling != "per_bar_times_h":
            # §3.4 defines ``ewma_daily`` as a *per-bar* estimator scaled by sqrt(h). The
            # enum carries ``estimated_at_horizon`` because §3.5's risk snapshot may be
            # estimated that way, but §3.4 declares no horizon estimator for this one, so
            # the pair is refused at load rather than resolved into an invented estimator.
            raise CalibrationError(
                "[sigma_h] estimator = 'ewma_daily' is defined per bar and scaled by sqrt(horizon_bars); "
                f"scaling = {self.scaling!r} declares an estimator \u00a73.4 does not define"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"estimator": self.estimator, "window_bars": int(self.window_bars), "scaling": self.scaling}


@dataclass(frozen=True)
class CalibrationSpec:
    """One ``calibrations/*.toml`` file, resolved. Every key of §3.4 is explicit.

    ``confidence`` is ``None`` for the point arm and a float in ``(0.5, 1.0)`` for the
    uncertainty arm; it is the *only* key the two shipped files differ in, which
    :func:`require_same_exclusion_geometry` checks rather than assumes.
    """

    kind: str
    standardize: str
    weighting: str
    shrink_pseudo_periods: float
    expected_sign: str
    uncertainty: str
    block_scheme: str
    block_periods: int
    n_resamples: int
    seed: int
    min_training_periods: int
    min_blocks: int
    inner_oos_required: bool
    executable_return: str
    label_distributions: str
    sigma_h: SigmaHSpec
    confidence: float | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in CALIBRATION_KINDS:
            raise CalibrationError(f"kind must be one of {CALIBRATION_KINDS}, got {self.kind!r}")
        if self.standardize not in STANDARDIZATIONS:
            raise CalibrationError(f"standardize must be one of {STANDARDIZATIONS}, got {self.standardize!r}")
        if self.weighting not in WEIGHTINGS:
            raise CalibrationError(f"weighting must be one of {WEIGHTINGS}, got {self.weighting!r}")
        shrink = float(self.shrink_pseudo_periods)
        if not math.isfinite(shrink) or shrink < 0.0:
            raise CalibrationError(f"shrink_pseudo_periods must be finite and >= 0, got {self.shrink_pseudo_periods}")
        if self.expected_sign not in EXPECTED_SIGNS:
            raise CalibrationError(f"expected_sign must be one of {EXPECTED_SIGNS}, got {self.expected_sign!r}")
        if self.uncertainty not in UNCERTAINTY_SCHEMES:
            raise CalibrationError(f"uncertainty must be one of {UNCERTAINTY_SCHEMES}, got {self.uncertainty!r}")
        if self.block_scheme not in BLOCK_SCHEMES:
            raise CalibrationError(f"block_scheme must be one of {BLOCK_SCHEMES}, got {self.block_scheme!r}")
        if int(self.block_periods) < 1:
            raise CalibrationError(f"block_periods must be >= 1, got {self.block_periods}")
        if int(self.n_resamples) < 1:
            raise CalibrationError(f"n_resamples must be >= 1, got {self.n_resamples}")
        if int(self.min_training_periods) < 1:
            raise CalibrationError(f"min_training_periods must be >= 1, got {self.min_training_periods}")
        if int(self.min_blocks) < 1:
            raise CalibrationError(f"min_blocks must be >= 1, got {self.min_blocks}")
        if self.executable_return not in EXECUTABLE_RETURNS:
            raise CalibrationError(
                f"executable_return must be one of {EXECUTABLE_RETURNS}, got {self.executable_return!r}"
            )
        if self.label_distributions not in LABEL_DISTRIBUTIONS:
            raise CalibrationError(
                f"label_distributions must be one of {LABEL_DISTRIBUTIONS}, got {self.label_distributions!r}"
            )
        if self.confidence is not None and not 0.5 < float(self.confidence) < 1.0:
            raise CalibrationError(
                f"confidence must lie in (0.5, 1.0) when present, got {self.confidence}; absent is the point arm"
            )
        if not isinstance(self.inner_oos_required, bool):
            raise CalibrationError(f"inner_oos_required must be a boolean, got {self.inner_oos_required!r}")

    # -- construction ---------------------------------------------------------------

    @classmethod
    def from_mapping(cls, table: Mapping[str, Any], *, source: str | None = None) -> "CalibrationSpec":
        """Resolve a parsed calibration table. Unknown and missing keys are refused."""
        where = f"{source}: " if source else ""
        keys = set(table)
        missing = sorted(_REQUIRED_KEYS - keys)
        if missing:
            raise CalibrationError(f"{where}calibration is missing required keys {missing} (§3.4)")
        unknown = sorted(keys - _REQUIRED_KEYS - _OPTIONAL_KEYS)
        if unknown:
            raise CalibrationError(
                f"{where}calibration declares unknown keys {unknown}; §3.4 fixes the schema and an undeclared "
                "key is a silent setting"
            )
        sigma_table = table["sigma_h"]
        if not isinstance(sigma_table, Mapping):
            raise CalibrationError(f"{where}[sigma_h] must be a table, got {type(sigma_table).__name__}")
        sigma_unknown = sorted(set(sigma_table) - {"estimator", "window_bars", "scaling"})
        if sigma_unknown:
            raise CalibrationError(f"{where}[sigma_h] declares unknown keys {sigma_unknown}")
        sigma_missing = sorted({"estimator", "window_bars", "scaling"} - set(sigma_table))
        if sigma_missing:
            raise CalibrationError(f"{where}[sigma_h] is missing required keys {sigma_missing}")
        confidence = table.get("confidence")
        return cls(
            kind=str(table["kind"]),
            standardize=str(table["standardize"]),
            weighting=str(table["weighting"]),
            shrink_pseudo_periods=float(table["shrink_pseudo_periods"]),
            expected_sign=str(table["expected_sign"]),
            uncertainty=str(table["uncertainty"]),
            block_scheme=str(table["block_scheme"]),
            block_periods=int(table["block_periods"]),
            n_resamples=int(table["n_resamples"]),
            seed=int(table["seed"]),
            min_training_periods=int(table["min_training_periods"]),
            min_blocks=int(table["min_blocks"]),
            inner_oos_required=bool(table["inner_oos_required"]),
            executable_return=str(table["executable_return"]),
            label_distributions=str(table["label_distributions"]),
            sigma_h=SigmaHSpec(
                estimator=str(sigma_table["estimator"]),
                window_bars=int(sigma_table["window_bars"]),
                scaling=str(sigma_table["scaling"]),
            ),
            confidence=None if confidence is None else float(confidence),
            source=source,
        )

    # -- derived declarations -------------------------------------------------------

    def bootstrap_spec(self) -> BlockBootstrapSpec:
        """The declared resampler, in decision periods (§3.4, :mod:`~prism.validation.uncertainty`)."""
        return BlockBootstrapSpec(
            block_periods=int(self.block_periods),
            n_resamples=int(self.n_resamples),
            seed=int(self.seed),
            block_scheme=self.block_scheme,
            min_blocks=int(self.min_blocks),
        )

    def shrink_factor(self, n_periods: int) -> float:
        """``n / (n + shrink_pseudo_periods)`` -- the declared shrinkage toward zero."""
        n = int(n_periods)
        if n < 0:
            raise CalibrationError(f"n_periods must be >= 0, got {n_periods}")
        denominator = float(n) + float(self.shrink_pseudo_periods)
        if denominator <= 0.0:
            return 1.0
        return float(n) / denominator

    def exclusion_geometry(self) -> dict[str, Any]:
        """Every declared key except ``confidence``: what the arms must hold in common."""
        payload = {key: getattr(self, key) for key in EXCLUSION_KEYS if key != "sigma_h"}
        payload["shrink_pseudo_periods"] = float(self.shrink_pseudo_periods)
        payload["sigma_h"] = self.sigma_h.to_dict()
        return payload

    def method(self) -> dict[str, Any]:
        """The resolved method block, recorded beside every fit (§3.4, §3.12)."""
        block = self.exclusion_geometry()
        block["confidence"] = None if self.confidence is None else float(self.confidence)
        block["arm_convention"] = (
            "confidence absent is the point arm: lower and upper both equal theta_shrunk. "
            "confidence present gives one-sided endpoints quantile(1 - confidence) and quantile(confidence) "
            "of the resampled shrunk slope, by linear interpolation between order statistics."
        )
        block["label"] = (
            "r = (O[t + h + 1] + D) / O[t + 1] - 1 with O the as-traded open, h = horizon_bars, and D the "
            "per-share cash distributions with ex-date in (t, t + h] when label_distributions = 'credited'; "
            "the regressand is r / sigma_h and the estimand is E[r_h] / sigma_h = theta * z."
        )
        block["units"] = (
            "theta is dimensionless. Expected return is recovered only by multiplying back by the name's "
            "horizon volatility: a_i = z_i * sigma_{i,h} carries return units and mu_i = theta_shrunk * a_i "
            "is an expected executable return over horizon_bars."
        )
        block["coverage_statement"] = (
            "Each endpoint is a one-sided, directional statement about the shared coefficient under the "
            "resampling assumptions: that the training pool is stationary enough to resample and that "
            "block_periods is long enough to carry the dependence overlapping labels induce. It is not a "
            "simultaneous interval over both endpoints, it says nothing about the coverage of a realized "
            "portfolio return, it does not survive a regime change inside the training pool, and it is not a "
            "conformal outcome width."
        )
        block["source"] = self.source
        return block


def load_calibration(path: str | Path) -> CalibrationSpec:
    """Parse and resolve a ``calibrations/*.toml`` file (§3.4)."""
    resolved = Path(path)
    try:
        table = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - message is the value
        raise CalibrationError(f"calibration file not found: {resolved}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CalibrationError(f"{resolved}: calibration file is not valid TOML: {exc}") from exc
    return CalibrationSpec.from_mapping(table, source=str(resolved))


def require_same_exclusion_geometry(specs: Mapping[str, CalibrationSpec]) -> dict[str, Any]:
    """Refuse a set of arms whose calibrations could produce different excluded folds.

    D6: a fold is excluded for all arms alike, and a fold is never dropped for one arm and
    kept for another. The arms may differ in ``confidence`` and in nothing else; this is
    the executable form of that, and it returns the shared geometry so a caller can record
    it once for the whole comparison.
    """
    if len(specs) < 2:
        raise CalibrationError("require_same_exclusion_geometry needs at least two arms to compare")
    items = list(specs.items())
    reference_id, reference = items[0]
    geometry = reference.exclusion_geometry()
    for arm_id, spec in items[1:]:
        other = spec.exclusion_geometry()
        differing = sorted(key for key in geometry if geometry[key] != other.get(key))
        if differing:
            raise CalibrationError(
                f"arms {reference_id!r} and {arm_id!r} declare different {differing} in their calibration files; "
                "the arms may differ in 'confidence' and in nothing else, or the comparison is not same-sample "
                "(D6, §3.4)"
            )
    return geometry


def require_sigma_agreement(spec: CalibrationSpec, *, scaling: str, horizon_bars: int | None = None) -> None:
    """Refuse a run whose risk snapshot does not scale volatility as the calibration does.

    §3.4: the ``[sigma_h]`` block declares the volatility the label, the vector ``a``, and
    the risk snapshot must agree on, and a run whose two declarations differ is refused --
    otherwise the two convex arms do not share one mapping and ``mu`` is not commensurable
    with the variance term of §3.7. ``horizon_bars`` is checked when the caller passes the
    snapshot's horizon.
    """
    if scaling != spec.sigma_h.scaling:
        raise CalibrationError(
            f"the risk snapshot declares scaling={scaling!r} and the calibration declares "
            f"{spec.sigma_h.scaling!r}; mu would not be commensurable with the variance term (§3.4, §3.5)"
        )
    if horizon_bars is not None and int(horizon_bars) < 1:
        raise CalibrationError(f"horizon_bars must be >= 1, got {horizon_bars}")


def validate_standardize(standardize: str, output_kind: str) -> None:
    """Refuse a standardization the component's ``output_kind`` does not admit (§3.4).

    ``rank_score`` requires ``rank_normal`` or ``zscore``; ``standardized_forecast`` takes
    ``none`` by default, so the slope calibrates the forecast's own scale, and may declare
    ``zscore``.
    """
    if standardize not in STANDARDIZATIONS:
        raise CalibrationError(f"standardize must be one of {STANDARDIZATIONS}, got {standardize!r}")
    admissible: Mapping[str, tuple[str, ...]] = {
        "rank_score": ("rank_normal", "zscore"),
        "standardized_forecast": ("none", "zscore"),
    }
    if output_kind not in admissible:
        raise CalibrationError(
            f"output_kind {output_kind!r} has no declared standardization admissibility; flow F2 takes "
            f"{sorted(admissible)} (§3.3, §3.4)"
        )
    allowed = admissible[output_kind]
    if standardize not in allowed:
        raise CalibrationError(
            f"output_kind {output_kind!r} admits standardize {allowed}, not {standardize!r}; an alternative "
            "standardization is a declared arm with its own place in the trial budget, never a free "
            "diagnostic (§3.4 [Q8])"
        )


# ---------------------------------------------------------------------------------
# Standardization (§3.4, "Standardization")
# ---------------------------------------------------------------------------------


def standardize_cross_section(scores: pd.Series, standardize: str, *, label: object = None) -> pd.Series:
    """Standardize one decision cross-section over the finite scores it carries.

    ``rank_normal`` is ``Phi^-1((rank - 0.5) / n)`` with ranks ascending from 1, ties
    taking the average rank, and ``n`` the count of finite scores. ``zscore`` is
    ``(x - mean) / sd`` with ``sd`` the *sample* standard deviation, refused when ``sd`` is
    not positive. ``none`` passes the score through. Non-finite entries stay non-finite:
    they are names the cross-section has no opinion on, and they are dropped from the
    regression rather than imputed.
    """
    if standardize not in STANDARDIZATIONS:
        raise CalibrationError(f"standardize must be one of {STANDARDIZATIONS}, got {standardize!r}")
    where = "" if label is None else f" in decision period {label!r}"
    values = pd.to_numeric(scores, errors="coerce").astype(float)
    finite = np.isfinite(values.to_numpy(dtype=float))
    out = pd.Series(np.full(len(values), np.nan), index=values.index, dtype=float)
    count = int(finite.sum())
    if count == 0:
        return out
    present = values[finite]
    if standardize == "none":
        out[finite] = present.to_numpy(dtype=float)
        return out
    if standardize == "zscore":
        if count < 2:
            raise CalibrationError(
                f"zscore standardization needs at least two finite scores{where}, got {count}; a sample "
                "standard deviation is undefined on one point (§3.4)"
            )
        sd = float(present.std(ddof=1))
        if not math.isfinite(sd) or sd <= 0.0:
            raise CalibrationError(
                f"zscore standardization refused{where}: the sample standard deviation is {sd}, not positive " "(§3.4)"
            )
        out[finite] = (present.to_numpy(dtype=float) - float(present.mean())) / sd
        return out
    ranks = present.rank(method="average", ascending=True).to_numpy(dtype=float)
    out[finite] = ndtri((ranks - 0.5) / float(count))
    return out


def standardize_panel(scores: pd.DataFrame, standardize: str) -> pd.DataFrame:
    """Standardize every row (decision period) of a panel, row by row.

    Vectorized over rows for a 3,000-name cross-section, and equal cell for cell to
    :func:`standardize_cross_section` applied one period at a time -- which is what the
    decision-time path calls, so the test that pins the two together is what keeps the fit
    and the deployment on one definition.
    """
    if standardize not in STANDARDIZATIONS:
        raise CalibrationError(f"standardize must be one of {STANDARDIZATIONS}, got {standardize!r}")
    if scores.shape[0] == 0:
        return pd.DataFrame(index=scores.index, columns=scores.columns, dtype=float)
    numeric = scores.apply(pd.to_numeric, errors="coerce").astype(float)
    finite = numeric.where(np.isfinite(numeric))
    counts = finite.notna().sum(axis=1)
    if standardize == "none":
        return finite
    if standardize == "zscore":
        means = finite.mean(axis=1)
        sds = finite.std(axis=1, ddof=1)
        thin = counts[counts < 2]
        if len(thin):
            raise CalibrationError(
                f"zscore standardization needs at least two finite scores in decision period {thin.index[0]!r}, "
                f"got {int(thin.iloc[0])}; a sample standard deviation is undefined on one point (§3.4)"
            )
        degenerate = sds[~np.isfinite(sds) | (sds <= 0.0)]
        if len(degenerate):
            raise CalibrationError(
                f"zscore standardization refused in decision period {degenerate.index[0]!r}: the sample "
                f"standard deviation is {float(degenerate.iloc[0])}, not positive (§3.4)"
            )
        return finite.sub(means, axis=0).div(sds, axis=0)
    ranks = finite.rank(axis=1, method="average", ascending=True)
    quantiles = ranks.sub(0.5).div(counts.where(counts > 0), axis=0)
    return pd.DataFrame(
        ndtri(quantiles.to_numpy(dtype=float)), index=scores.index, columns=scores.columns, dtype=float
    )


# ---------------------------------------------------------------------------------
# The label and its volatility denominator (§3.4, "Label", "Horizon volatility")
# ---------------------------------------------------------------------------------


def executable_returns(
    opens: pd.DataFrame,
    decision_positions: Sequence[int],
    horizon_bars: int,
    *,
    distributions: pd.DataFrame | None = None,
    label_distributions: str = "credited",
) -> pd.DataFrame:
    """The return the fills can earn, per decision period (§3.4, "Label").

    ``r_{p,i} = (O[t_p + h + 1] + D_i) / O[t_p + 1] - 1`` with ``O`` the as-traded open,
    ``h = horizon_bars``, and ``D_i`` the per-share cash distributions with ex-date in
    ``(t_p, t_p + h]`` when ``label_distributions = "credited"``. The entry is the open
    *after* the decision close, so nothing in the label is earnable only by a fill at the
    decision bar's own close.

    ``opens`` is indexed by bar in ascending order and ``decision_positions`` are integer
    positions of the decision bars into that index. A period whose exit bar lies beyond the
    panel has no complete label and its row is left non-finite: that is the ordinary end of
    a training pool (13 of the reference geometry's 14 periods carry a label), not an
    error. A ``credited`` label with no distributions supplied is an error, because
    silently dropping dividends understates every long return.
    """
    if label_distributions not in LABEL_DISTRIBUTIONS:
        raise CalibrationError(
            f"label_distributions must be one of {LABEL_DISTRIBUTIONS}, got {label_distributions!r}"
        )
    if int(horizon_bars) < 1:
        raise CalibrationError(f"horizon_bars must be >= 1, got {horizon_bars}")
    if opens.empty or opens.shape[1] == 0:
        raise CalibrationError("opens panel is empty; a label needs bars and names")
    if not opens.index.is_monotonic_increasing or not opens.index.is_unique:
        raise CalibrationError("opens panel index must be unique and ascending in bar order")
    positions = [int(position) for position in decision_positions]
    if not positions:
        raise CalibrationError("decision_positions is empty; there is no decision period to label")
    if any(later <= earlier for earlier, later in zip(positions, positions[1:])):
        raise CalibrationError("decision_positions must be strictly increasing bar positions")
    n_bars = int(opens.shape[0])
    if positions[0] < 0 or positions[-1] >= n_bars:
        raise CalibrationError(
            f"decision_positions must lie in [0, {n_bars - 1}], got [{positions[0]}, {positions[-1]}]"
        )
    if label_distributions == "credited":
        if distributions is None:
            raise CalibrationError(
                "label_distributions = 'credited' but no distributions panel was supplied; a silently "
                "uncredited distribution understates every long label (§3.4)"
            )
        if not distributions.columns.equals(opens.columns):
            raise CalibrationError("distributions panel must carry the same names, in the same order, as opens")
        extra = distributions.index.difference(opens.index)
        if len(extra):
            raise CalibrationError(f"distributions carry {len(extra)} ex-date bars absent from the opens panel")
        cash = (
            distributions.reindex(index=opens.index)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    else:
        if distributions is not None:
            raise CalibrationError(
                "label_distributions = 'ignored' but a distributions panel was supplied; the panel would be "
                "dropped silently"
            )
        cash = np.zeros(opens.shape, dtype=float)

    prices = opens.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    horizon = int(horizon_bars)
    out = np.full((len(positions), opens.shape[1]), np.nan, dtype=float)
    for row, position in enumerate(positions):
        entry_bar = position + 1
        exit_bar = position + horizon + 1
        if exit_bar > n_bars - 1:
            continue
        entry = prices[entry_bar]
        exit_price = prices[exit_bar]
        credited = cash[position + 1 : position + horizon + 1].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_values = (exit_price + credited) / entry - 1.0
        row_values = np.where(np.isfinite(entry) & (entry > 0.0) & np.isfinite(exit_price), row_values, np.nan)
        out[row] = row_values
    return pd.DataFrame(out, index=opens.index[positions], columns=opens.columns, dtype=float)


def ewma_bar_sigma(prices: pd.DataFrame, window_bars: int) -> pd.DataFrame:
    """Per-bar EWMA standard deviation of daily simple returns at span ``window_bars``.

    Point-in-time by construction: ``pandas`` EWMA at bar ``t`` uses bars up to ``t`` and
    no later one, and ``min_periods`` keeps the warm-up non-finite rather than reporting a
    volatility from three observations.
    """
    if int(window_bars) < 2:
        raise CalibrationError(f"[sigma_h] window_bars must be >= 2, got {window_bars}")
    if prices.empty or prices.shape[1] == 0:
        raise CalibrationError("price panel is empty; a volatility needs bars and names")
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    returns = numeric.pct_change()
    span = int(window_bars)
    return returns.ewm(span=span, min_periods=span, adjust=False).std()


def horizon_sigma(
    prices: pd.DataFrame,
    sigma_h: SigmaHSpec,
    horizon_bars: int,
    *,
    decision_positions: Sequence[int] | None = None,
) -> pd.DataFrame:
    """``sigma_{i,h,t} = sigma_{i,bar,t} * sqrt(horizon_bars)`` under ``per_bar_times_h``.

    The value this returns is the denominator of the label, the multiplier that turns the
    standardized score into the coefficient vector ``a``, and the volatility the risk
    snapshot must agree with (:func:`require_sigma_agreement`). One estimator, one scaling,
    one number -- if the three disagreed, ``mu`` would not be commensurable with the
    variance term of §3.7.
    """
    if int(horizon_bars) < 1:
        raise CalibrationError(f"horizon_bars must be >= 1, got {horizon_bars}")
    if sigma_h.scaling != "per_bar_times_h":  # pragma: no cover - refused at load
        raise CalibrationError(f"scaling {sigma_h.scaling!r} has no declared estimator in §3.4")
    scaled = ewma_bar_sigma(prices, sigma_h.window_bars) * math.sqrt(float(int(horizon_bars)))
    if decision_positions is None:
        return scaled
    positions = [int(position) for position in decision_positions]
    if positions and (positions[0] < 0 or positions[-1] >= int(scaled.shape[0])):
        raise CalibrationError("decision_positions must be bar positions inside the price panel")
    return scaled.iloc[positions]


# ---------------------------------------------------------------------------------
# Fold exclusion (§3.4, "Minimum training length and fold exclusion"; D6)
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldExclusion:
    """Whether a fold is admitted, decided from the geometry alone.

    Nothing here depends on ``confidence``, which is the point: the three arms of the
    stage 4 registration read the identical decision off the shared calibration family,
    so a fold is never dropped for one arm and kept for another (D6).
    """

    excluded: bool
    reason: str | None
    n_periods: int
    blocks_drawn: int
    min_training_periods: int
    min_blocks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "excluded": bool(self.excluded),
            "exclusion_reason": self.reason,
            "n_periods": int(self.n_periods),
            "blocks_drawn": int(self.blocks_drawn),
            "min_training_periods": int(self.min_training_periods),
            "min_blocks": int(self.min_blocks),
        }


def fold_exclusion(n_periods: int, spec: CalibrationSpec) -> FoldExclusion:
    """Decide a fold's admission from ``n_periods`` and the declared minimums.

    Callable without fitting anything, which is how the ``decile_control`` arm -- which
    estimates no slope -- obtains the identical excluded set from the registration's
    ``[shared] calibration_family``.
    """
    n = int(n_periods)
    if n < 0:
        raise CalibrationError(f"n_periods must be >= 0, got {n_periods}")
    block = int(spec.block_periods)
    drawn = int(math.ceil(n / block)) if n > 0 else 0
    if n < int(spec.min_training_periods):
        reason = (
            f"training pool holds {n} decision periods with a complete observation, below the declared "
            f"min_training_periods={int(spec.min_training_periods)}"
        )
    elif block > n:
        reason = (
            f"block_periods={block} exceeds the {n} decision periods in the training pool; a block longer "
            "than the sample is not a resample"
        )
    elif drawn < int(spec.min_blocks):
        reason = (
            f"bootstrap would draw {drawn} blocks from {n} decision periods at block_periods={block}, below "
            f"the declared min_blocks={int(spec.min_blocks)}"
        )
    else:
        return FoldExclusion(
            excluded=False,
            reason=None,
            n_periods=n,
            blocks_drawn=drawn,
            min_training_periods=int(spec.min_training_periods),
            min_blocks=int(spec.min_blocks),
        )
    return FoldExclusion(
        excluded=True,
        reason=reason,
        n_periods=n,
        blocks_drawn=drawn,
        min_training_periods=int(spec.min_training_periods),
        min_blocks=int(spec.min_blocks),
    )


# ---------------------------------------------------------------------------------
# The fit (§3.4, "Slope", "Uncertainty", "Sign policy")
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationFit:
    """One fold's calibration, as it is recorded (§3.4, §3.12).

    The five estimate fields are ``None`` on an excluded fold rather than ``NaN``: the
    record is canonicalized by ``stable_json_dumps``, which refuses NaN, and a number that
    was never estimated must not be readable as one. :func:`objective_endpoints` and
    :func:`apply_calibration` both raise on an excluded fit.
    """

    theta_hat: float | None
    theta_shrunk: float | None
    lower: float | None
    upper: float | None
    straddles_zero: bool
    n_periods: int
    n_observations: int
    shrink_factor: float | None
    blocks_drawn: int
    fit_cutoff: str
    sign_ok: bool
    excluded: bool
    exclusion_reason: str | None
    disabled_reason: str | None
    confidence: float | None
    method: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        """The per-fold calibration record of §3.12 -- JSON-safe, no NaN, nothing implied."""
        return {
            "theta_hat": _record_float(self.theta_hat),
            "theta_shrunk": _record_float(self.theta_shrunk),
            "shrink_factor": _record_float(self.shrink_factor),
            "lower": _record_float(self.lower),
            "upper": _record_float(self.upper),
            "straddles_zero": bool(self.straddles_zero),
            "n_periods": int(self.n_periods),
            "n_observations": int(self.n_observations),
            "blocks_drawn": int(self.blocks_drawn),
            "fit_cutoff": self.fit_cutoff,
            "sign_ok": bool(self.sign_ok),
            "excluded": bool(self.excluded),
            "exclusion_reason": self.exclusion_reason,
            "disabled_reason": self.disabled_reason,
            "confidence": None if self.confidence is None else float(self.confidence),
            "method": dict(self.method),
        }


def _record_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):  # pragma: no cover - a non-finite estimate raises upstream
        raise CalibrationError(f"a non-finite estimate ({number}) reached the record; §3.4 raises on one")
    return number


def _require_panel(frame: object, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise CalibrationError(f"{name} must be a DataFrame of decision periods by names, got {type(frame).__name__}")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise CalibrationError(
            f"{name} is empty ({frame.shape[0]} periods by {frame.shape[1]} names); an empty panel is a caller "
            "defect, not a short fold"
        )
    return frame


def fit_calibration(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    sigma_h: pd.DataFrame,
    spec: CalibrationSpec,
    folds_inner: Sequence[Hashable] | None = None,
    *,
    fit_cutoff: str | None = None,
    horizon_bars: int | None = None,
    update_every_bars: int | None = None,
) -> CalibrationFit:
    """Fit the shared slope on one fold's training pool (§3.4).

    ``scores`` are the component's **raw** outputs -- this function standardizes them by
    the declared rule, so the fit standardizes exactly the vector the decision-time path
    does. ``returns`` are executable returns from :func:`executable_returns` and
    ``sigma_h`` the horizon volatilities from :func:`horizon_sigma`. All three are indexed
    by decision period, ascending, with one column per name, and must agree exactly: a
    silent reindex would align a label to the wrong date.

    ``folds_inner`` names the decision periods whose scores are *inner out-of-sample*
    predictions. A component declaring ``fitted_state = "train_only"`` calibrates on those
    and never on fitted training predictions, so ``inner_oos_required = true`` refuses a
    call that does not supply them. No outer test outcome may enter a fit; that boundary is
    the evaluator's, and this function never sees a test bar.

    ``horizon_bars`` and ``update_every_bars``, when supplied, are checked against
    ``block_periods``: a block must span at least one label, and a configuration that
    violates it is refused rather than silently widened.
    """
    scores = _require_panel(scores, "scores")
    returns = _require_panel(returns, "returns")
    sigma_frame = _require_panel(sigma_h, "sigma_h")
    for name, frame in (("returns", returns), ("sigma_h", sigma_frame)):
        if not frame.index.equals(scores.index):
            raise CalibrationError(f"{name} index does not equal the scores index; the panels must be aligned")
        if not frame.columns.equals(scores.columns):
            raise CalibrationError(f"{name} columns do not equal the scores columns; the panels must be aligned")
    if not scores.index.is_unique:
        raise CalibrationError("the decision-period index carries duplicates; a period is one cross-section")
    if not scores.index.is_monotonic_increasing:
        raise CalibrationError(
            "the decision-period index is not ascending; circular_moving blocks are runs of *consecutive* "
            "periods, so an unordered index would silently destroy the block structure (§3.4)"
        )
    if horizon_bars is not None and update_every_bars is not None:
        validate_block_span(int(spec.block_periods), int(update_every_bars), int(horizon_bars))

    if spec.inner_oos_required and folds_inner is None:
        raise CalibrationError(
            "inner_oos_required = true but no inner out-of-sample periods were supplied; a train_only "
            "component calibrates on inner out-of-sample predictions, never on fitted training predictions "
            "(§3.4)"
        )
    if folds_inner is not None:
        wanted = pd.Index(list(folds_inner))
        if not len(wanted):
            raise CalibrationError("folds_inner is empty; there is no inner out-of-sample period to calibrate on")
        missing = wanted.difference(scores.index)
        if len(missing):
            raise CalibrationError(f"folds_inner names {len(missing)} periods absent from the panels: {list(missing)}")
        keep_index = scores.index.intersection(wanted)
        scores = scores.loc[keep_index]
        returns = returns.loc[keep_index]
        sigma_frame = sigma_frame.loc[keep_index]

    cutoff = str(scores.index[-1]) if fit_cutoff is None else str(fit_cutoff)
    standardized = standardize_panel(scores, spec.standardize)

    z = standardized.to_numpy(dtype=float)
    r = returns.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    sigma = sigma_frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    score_ok = np.isfinite(z)
    return_ok = np.isfinite(r)
    sigma_ok = np.isfinite(sigma) & (sigma > 0.0)
    complete = score_ok & return_ok & sigma_ok
    per_period = complete.sum(axis=1)
    kept = per_period > 0
    n_periods = int(kept.sum())
    n_observations = int(per_period[kept].sum())

    exclusion = fold_exclusion(n_periods, spec)
    sample = {
        "n_cells": int(complete.size),
        "n_complete": n_observations,
        "n_missing_score": int((~score_ok).sum()),
        "n_missing_return": int((score_ok & ~return_ok).sum()),
        "n_invalid_sigma": int((score_ok & return_ok & ~sigma_ok).sum()),
        "n_periods_supplied": int(scores.shape[0]),
        "n_periods_used": n_periods,
        "inner_oos_periods": None if folds_inner is None else int(len(list(folds_inner))),
    }
    if exclusion.excluded:
        return _excluded_fit(spec, exclusion, cutoff, sample, n_observations, horizon_bars, update_every_bars)

    y = np.divide(r, sigma, out=np.full_like(r, np.nan), where=sigma_ok)
    weights = np.zeros_like(z)
    if spec.weighting == "per_date_equal":
        scale = np.divide(
            1.0, per_period.astype(float), out=np.zeros(per_period.shape, dtype=float), where=per_period > 0
        )
        weights[complete] = np.broadcast_to(scale[:, None], weights.shape)[complete]
    else:
        weights[complete] = 1.0

    products = np.where(complete, weights * z * y, 0.0)
    squares = np.where(complete, weights * z * z, 0.0)
    numerators = products.sum(axis=1)[kept]
    denominators = squares.sum(axis=1)[kept]

    denominator = float(denominators.sum())
    numerator = float(numerators.sum())
    if not math.isfinite(denominator) or not math.isfinite(numerator):
        raise CalibrationError(
            f"the pooled regression is not finite (numerator {numerator}, denominator {denominator}); a "
            "degenerate estimate is an estimation failure and the run is recorded failed (§3.4)"
        )
    if denominator <= 0.0:
        raise CalibrationError(
            f"the pooled regression's denominator sum(w z^2) is {denominator}, not positive; the standardized "
            "scores carry no cross-sectional variation and no slope is identified (§3.4)"
        )
    theta_hat = numerator / denominator
    if not math.isfinite(theta_hat):
        raise CalibrationError(f"theta_hat is not finite ({theta_hat}); an estimation failure raises (§3.4)")
    shrink = spec.shrink_factor(n_periods)
    theta_shrunk = theta_hat * shrink

    resampled = _resample_shrunk_slopes(numerators, denominators, shrink, theta_shrunk, spec)
    if resampled.excluded:  # pragma: no cover - fold_exclusion already agreed the plan is sufficient
        raise CalibrationError(
            f"the bootstrap plan disagrees with the fold's admission ({resampled.exclusion_reason}); the two "
            "read the same declared minimums and must agree"
        )
    if spec.confidence is None:
        lower = upper = theta_shrunk
    else:
        lower, upper = resampled.one_sided_endpoints(float(spec.confidence))

    sign_ok = theta_shrunk > 0.0 if spec.expected_sign == "positive" else theta_shrunk < 0.0
    method = _method_block(spec, sample, exclusion, horizon_bars, update_every_bars)
    method["bootstrap"] = resampled.to_dict()
    return CalibrationFit(
        theta_hat=float(theta_hat),
        theta_shrunk=float(theta_shrunk),
        lower=float(lower),
        upper=float(upper),
        straddles_zero=bool(lower <= 0.0 <= upper),
        n_periods=n_periods,
        n_observations=n_observations,
        shrink_factor=float(shrink),
        blocks_drawn=int(resampled.blocks_drawn),
        fit_cutoff=cutoff,
        sign_ok=bool(sign_ok),
        excluded=False,
        exclusion_reason=None,
        disabled_reason=None if sign_ok else "sign_contradiction",
        confidence=None if spec.confidence is None else float(spec.confidence),
        method=method,
    )


def _excluded_fit(
    spec: CalibrationSpec,
    exclusion: FoldExclusion,
    cutoff: str,
    sample: Mapping[str, Any],
    n_observations: int,
    horizon_bars: int | None,
    update_every_bars: int | None,
) -> CalibrationFit:
    """A fold below the declared minimums: recorded with its reason, and no estimate."""
    return CalibrationFit(
        theta_hat=None,
        theta_shrunk=None,
        lower=None,
        upper=None,
        straddles_zero=False,
        n_periods=int(exclusion.n_periods),
        n_observations=int(n_observations),
        shrink_factor=None,
        blocks_drawn=int(exclusion.blocks_drawn),
        fit_cutoff=cutoff,
        sign_ok=False,
        excluded=True,
        exclusion_reason=exclusion.reason,
        disabled_reason="fold_excluded",
        confidence=None if spec.confidence is None else float(spec.confidence),
        method=_method_block(spec, sample, exclusion, horizon_bars, update_every_bars),
    )


def _method_block(
    spec: CalibrationSpec,
    sample: Mapping[str, Any],
    exclusion: FoldExclusion,
    horizon_bars: int | None,
    update_every_bars: int | None,
) -> dict[str, Any]:
    method = spec.method()
    method["sample"] = dict(sample)
    method["admission"] = exclusion.to_dict()
    method["schedule"] = {
        "horizon_bars": None if horizon_bars is None else int(horizon_bars),
        "update_every_bars": None if update_every_bars is None else int(update_every_bars),
    }
    return method


def _resample_shrunk_slopes(
    numerators: np.ndarray,
    denominators: np.ndarray,
    shrink: float,
    observed: float,
    spec: CalibrationSpec,
) -> BootstrapResult:
    """Refit the shrunk slope on every block resample (§3.4, "Uncertainty").

    Because the slope is a ratio of sums over decision periods, refitting on a resample is
    the ratio of the resampled sums -- exactly the same number a period-by-period refit
    would produce, and :func:`prism.validation.uncertainty.block_bootstrap` is the
    reference the test pins this against. The shrink factor is held at the original ``n``,
    so the resampled spread reflects slope variation and not a varying shrink factor.
    """
    resampler = spec.bootstrap_spec()
    plan = plan_blocks(int(numerators.size), resampler)
    indices = block_resample_indices(int(numerators.size), resampler)
    resampled_numerator = numerators[indices].sum(axis=1)
    resampled_denominator = denominators[indices].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = float(shrink) * (resampled_numerator / resampled_denominator)
    if not np.all(np.isfinite(values)):
        bad = int(np.count_nonzero(~np.isfinite(values)))
        raise CalibrationError(
            f"{bad} of {values.size} resamples produced a non-finite slope; a degenerate estimate is a "
            "failure, not a wide interval (§3.4)"
        )
    return BootstrapResult(
        observed=float(observed),
        values=np.asarray(values, dtype=float),
        n_units=plan.n_units,
        blocks_drawn=plan.blocks_drawn,
        excluded=False,
        exclusion_reason=None,
        method=resampler.method(),
    )


# ---------------------------------------------------------------------------------
# Using a fit (§3.4, "Estimand and units", "Objective use", "Sign policy")
# ---------------------------------------------------------------------------------


def objective_endpoints(fit: CalibrationFit) -> tuple[float, float]:
    """The ``(lower, upper)`` §3.7 maximizes ``min(lower * s, upper * s)`` over, sign policy applied.

    On a fit whose sign contradicts ``expected_sign`` this returns ``(0.0, 0.0)``: the
    return term vanishes, so the component takes **no new directional exposure**, the
    evaluation continues, and the slope is never inverted. The existing book is not this
    function's business -- it follows ``on_sign_contradiction`` in the recovery policy,
    subject to executable limits.

    A negative ``lower`` with a correct-sign point estimate is *uncertainty*, not a
    contradiction, and is returned unchanged; §3.7's adverse term then sets ``a' w = 0`` on
    that fold, which is the designed output of the declared method.
    """
    if fit.excluded:
        raise CalibrationError(
            f"the fold is excluded ({fit.exclusion_reason}); it has no endpoints, and an excluded fold is "
            "excluded for every arm alike (D6)"
        )
    if fit.lower is None or fit.upper is None:  # pragma: no cover - only an excluded fit has none
        raise CalibrationError("the fit carries no endpoints")
    if not fit.sign_ok:
        return (0.0, 0.0)
    return (float(fit.lower), float(fit.upper))


def apply_calibration(
    fit: CalibrationFit,
    z_t: pd.Series | np.ndarray,
    sigma_h_t: pd.Series | np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    """``(a_t, mu_t)`` with ``a_t = z_t * sigma_h_t`` and ``mu_t = theta_shrunk * a_t``.

    **This multiplication is the units contract.** ``theta`` is dimensionless because the
    regressand was a return divided by a volatility at the same horizon, so expected return
    is recovered only by multiplying the standardized score back by the name's horizon
    volatility. The constructor consumes ``a`` -- never ``z`` -- and an objective built on
    ``z`` alone would be wrong by a factor of order ``sigma_h`` (0.0722 for a 25 percent
    annualized name at a 21-bar horizon) and would silently re-weight the cross-section by
    inverse volatility.

    ``z_t`` is already standardized: pass :func:`standardize_cross_section` output, so the
    decision-time vector is formed by the same code the fit used. Every entry must be
    finite and every volatility finite and positive; a name with no current estimate is the
    constructor's reduce-only case and is given ``a_i = 0`` *there*, explicitly, rather than
    arriving here as a silent NaN.
    """
    if fit.excluded:
        raise CalibrationError(
            f"the fold is excluded ({fit.exclusion_reason}); there is no mapping to apply, for any arm (D6)"
        )
    if fit.theta_shrunk is None:  # pragma: no cover - only an excluded fit has none
        raise CalibrationError("the fit carries no slope")
    index = z_t.index if isinstance(z_t, pd.Series) else None
    if isinstance(sigma_h_t, pd.Series):
        if index is None:
            raise CalibrationError("z_t and sigma_h_t must both be Series or both be arrays")
        if not sigma_h_t.index.equals(index):
            raise CalibrationError("z_t and sigma_h_t must share one index; a silent reindex misprices names")
    scores = np.asarray(z_t, dtype=float)
    sigma = np.asarray(sigma_h_t, dtype=float)
    if scores.shape != sigma.shape:
        raise CalibrationError(f"z_t {scores.shape} and sigma_h_t {sigma.shape} must have the same shape")
    if scores.size == 0:
        raise CalibrationError("z_t is empty; there is no cross-section to price")
    if not np.all(np.isfinite(scores)):
        raise CalibrationError(
            "z_t carries non-finite entries; a name without a standardized score is dropped from the "
            "cross-section or given a zero coefficient explicitly, never carried as NaN"
        )
    if not np.all(np.isfinite(sigma)) or not np.all(sigma > 0.0):
        raise CalibrationError(
            "sigma_h_t carries a non-finite or non-positive volatility; expected return would be undefined "
            "or zero for that name (§3.4)"
        )
    a_values = scores * sigma
    mu_values = float(fit.theta_shrunk) * a_values
    if index is None:
        index = pd.RangeIndex(len(a_values))
    return (
        pd.Series(a_values, index=index, dtype=float, name="a"),
        pd.Series(mu_values, index=index, dtype=float, name="mu"),
    )
