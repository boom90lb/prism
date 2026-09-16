"""Comparison and repricing reports (engine specification §3.12, "Comparison recording").

``compare`` distinguishes two kinds of input and produces one report.

* A **reevaluation** is a set of records produced by ``prism evaluate`` — new runs with new
  ids. Any change that reconstructs the portfolio is one, a confidence change and a
  constructor-respecting cost sensitivity included. This module *reads* such records; it
  never creates them.
* A **repricing** is produced here: fill quantities and fill prices are held fixed, cost
  debits are recomputed under a new multiplier, and the resulting equity path is labeled
  counterfactual with its deviation from the recorded path reported. It attaches to the
  original record as a diagnostic and is never a new evidence row. It is exact only under
  ``accounting_engine = "historical_constant_weight"``; under the canonical engine a
  fixed-share repricing produces a cash path the accounting would not have produced, which
  is why the deviation is reported rather than presented as an outcome.

**Units.** Returns are compared with returns. A hurdle is a *return series* — the declared
cash series accrued per period — and never a scalar yield or a Sharpe number
(§3.12 **[Q21]**). :func:`require_return_series` enforces that at the boundary:
:func:`constant_hurdle_returns` is the only sanctioned way to turn an annual yield into
something comparable, and it produces a series, not a bar for a Sharpe to clear.

**Intervals.** Sharpe differences are computed on each arm's periodic net return series with
paired block resampling — both arms resampled on the same blocks — and every parameter is
declared in :class:`SharpeBootstrapSpec` before the run: frequency, block length in bars (at
least the decision interval), resample count, seed, and scheme. The interval is studentized
(a bootstrap-t on the Sharpe difference, standardized by a HAC standard error from the
influence function of the moment functional, in the manner of Ledoit and Wolf 2008) and is
reported beside the difference in means.

Both the point Sharpe and every resampled Sharpe use the project's ``ddof=1`` estimator
(:func:`prism.validation.metrics.periodic_sharpe`), while the studentizing standard error is
the asymptotic population-moment quantity; the two conventions differ by a factor
``sqrt(T / (T - 1))`` that vanishes in the sample sizes this engine evaluates, and the
studentization stays internally consistent because observed and resampled statistics use one
convention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from prism.validation.metrics import periodic_sharpe
from prism.validation.uncertainty import (
    CIRCULAR_MOVING,
    BlockBootstrapSpec,
    BootstrapError,
    block_resample_indices,
    plan_blocks,
)

#: Record fields every compared panel is labeled with (§3.12).
PROVENANCE_FLAGS: tuple[str, ...] = ("fill_source", "synthetic", "prospective", "run_channel")

#: Fields whose equality across arms is reported as "held fixed" and whose difference is
#: reported as a changed condition.
CONDITION_FIELDS: tuple[str, ...] = (
    "data_id",
    "visibility_mode",
    "vintage",
    "accounting_engine",
    "executable_return",
    "capital",
    "engine",
    "cost_refresh",
)

#: The channel no comparison may cite (§3.12, "Which commands write rows").
ENGINEERING_CHANNEL = "engineering"

UNAVAILABLE = "unavailable"

#: Chunk of resamples held in memory at once by the studentized bootstrap.
_MAX_CHUNK_CELLS = 20_000_000


class ComparisonRefused(ValueError):
    """The comparison is refused because the inputs are not the same evidence class."""


# ---------------------------------------------------------------------------
# Units: returns are compared with returns.
# ---------------------------------------------------------------------------


def require_return_series(value: Any, name: str) -> np.ndarray:
    """Return ``value`` as a 1-D float array, or raise if it is not a return series.

    A scalar is refused with the reason spelled out: a yield or a Sharpe number is not a
    return series, and comparing one with a return series is the units error §3.12 forbids.
    Use :func:`constant_hurdle_returns` to accrue an annual yield into per-period returns.
    """
    if isinstance(value, (pd.Series, pd.Index)):
        arr = np.asarray(value.to_numpy(), dtype=float)
    elif isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float)
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float)
    else:
        raise TypeError(
            f"{name} must be a per-period return series, got {type(value).__name__}. Returns are compared "
            "with returns: a Sharpe number is never compared with a yield, and a scalar hurdle must be "
            "accrued into a return series first (constant_hurdle_returns)."
        )
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size < 3:
        raise ValueError(f"{name} needs at least 3 periods, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} carries non-finite values; a gap in a return series is a defect, not a zero")
    return arr


def constant_hurdle_returns(
    annual_rate: float,
    periods: int | pd.Index,
    *,
    periods_per_year: int = 252,
) -> pd.Series:
    """Accrue a flat annual yield into a per-period *return* series.

    This is the unit conversion that makes a yield comparable with a strategy's returns:
    ``annual_rate / periods_per_year`` per period. It is a convenience for a flat rate; a
    registration that names a cash series (``[benchmark].cash_series``, §3.13) accrues the
    realized series instead and records it, and that recorded series is what a promotion
    rule and a comparison read.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
    rate = float(annual_rate)
    if not np.isfinite(rate):
        raise ValueError(f"annual_rate must be finite, got {annual_rate}")
    index = pd.RangeIndex(int(periods)) if isinstance(periods, int) else pd.Index(periods)
    if len(index) < 1:
        raise ValueError("periods is empty")
    return pd.Series(np.full(len(index), rate / periods_per_year, dtype=float), index=index, name="hurdle_return")


def excess_returns(net_returns: Any, cash_returns: Any) -> np.ndarray:
    """``net - cash``, element-wise, on two aligned per-period return series.

    Both arguments must be return series: this is the return-versus-hurdle read that
    replaces comparing a Sharpe with a yield (§3.12 migration row for `SPEC.md:80`).
    """
    net = require_return_series(net_returns, "net_returns")
    cash = require_return_series(cash_returns, "cash_returns")
    if net.size != cash.size:
        raise ValueError(f"net_returns has {net.size} periods and cash_returns has {cash.size}; align them first")
    if isinstance(net_returns, pd.Series) and isinstance(cash_returns, pd.Series):
        if not net_returns.index.equals(cash_returns.index):
            raise ValueError("net_returns and cash_returns carry different indexes; align them before differencing")
    return net - cash


# ---------------------------------------------------------------------------
# The declared bootstrap.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharpeBootstrapSpec:
    """The registration's ``[sharpe_bootstrap]`` block, resolved (§3.12).

    ``block_bars`` is the block length in bars and must be at least the decision interval,
    so a block spans at least one decision. ``hac_lags`` defaults to ``block_bars - 1``, the
    dependence the block length already declares; it is recorded either way.
    """

    block_bars: int
    n_resamples: int
    seed: int
    frequency: str = "daily"
    scheme: str = CIRCULAR_MOVING
    confidence: float = 0.95
    decision_interval_bars: int = 1
    min_blocks: int = 2
    hac_lags: int | None = None

    def __post_init__(self) -> None:
        if int(self.block_bars) < 1:
            raise ValueError(f"block_bars must be >= 1, got {self.block_bars}")
        if int(self.block_bars) < int(self.decision_interval_bars):
            raise ValueError(
                f"block_bars={int(self.block_bars)} is shorter than the decision interval "
                f"{int(self.decision_interval_bars)} bars; a block must span at least one decision (§3.12)"
            )
        if not 0.5 < float(self.confidence) < 1.0:
            raise ValueError(f"confidence must lie in (0.5, 1.0), got {self.confidence}")
        if self.hac_lags is not None and int(self.hac_lags) < 0:
            raise ValueError(f"hac_lags must be >= 0, got {self.hac_lags}")

    @property
    def lags(self) -> int:
        """HAC truncation lag actually used."""
        return int(self.block_bars) - 1 if self.hac_lags is None else int(self.hac_lags)

    def block_spec(self) -> BlockBootstrapSpec:
        """The unit-level resampling scheme; the unit here is one bar."""
        return BlockBootstrapSpec(
            block_periods=int(self.block_bars),
            n_resamples=int(self.n_resamples),
            seed=int(self.seed),
            block_scheme=self.scheme,
            min_blocks=int(self.min_blocks),
        )

    def method(self) -> dict[str, Any]:
        block = self.block_spec().method()
        block.update(
            {
                "frequency": self.frequency,
                "block_bars": int(self.block_bars),
                "decision_interval_bars": int(self.decision_interval_bars),
                "hac_lags": self.lags,
                "confidence": float(self.confidence),
                "units": "bars",
                "interval_method": "studentized_circular_block_bootstrap",
            }
        )
        return block


@dataclass(frozen=True)
class SharpeInterval:
    """A Sharpe statistic with its studentized interval and the difference in means."""

    label: str
    statistic: str
    sharpe_a: float
    sharpe_b: float
    difference: float
    mean_difference: float
    standard_error: float
    lower: float
    upper: float
    confidence: float
    t_statistic: float
    p_value_two_sided: float
    n_periods: int
    blocks_drawn: int
    n_resamples: int
    method: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "statistic": self.statistic,
            "sharpe_a": self.sharpe_a,
            "sharpe_b": self.sharpe_b,
            "difference": self.difference,
            "mean_difference": self.mean_difference,
            "standard_error": self.standard_error,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "t_statistic": self.t_statistic,
            "p_value_two_sided": self.p_value_two_sided,
            "n_periods": self.n_periods,
            "blocks_drawn": self.blocks_drawn,
            "n_resamples": self.n_resamples,
            "method": dict(self.method),
        }


def _sharpe_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise ``ddof=1`` Sharpe of a ``(rows, T)`` matrix."""
    mean = matrix.mean(axis=1)
    std = matrix.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(std > 0.0, mean / np.where(std > 0.0, std, 1.0), np.nan)
    return out


def _hac_quadratic(moments: np.ndarray, gradient: np.ndarray, lags: int) -> np.ndarray:
    """``g' Psi g`` per row, with ``Psi`` a Bartlett-kernel HAC of ``moments``.

    ``moments`` is ``(rows, T, k)`` and ``gradient`` is ``(rows, k)``. The moment vector is
    centered here, so the caller passes raw moment contributions.
    """
    rows, n_periods, _ = moments.shape
    centered = moments - moments.mean(axis=1, keepdims=True)
    psi = np.einsum("btk,btm->bkm", centered, centered) / n_periods
    for lag in range(1, int(lags) + 1):
        if lag >= n_periods:
            break
        gamma = np.einsum("btk,btm->bkm", centered[:, lag:, :], centered[:, :-lag, :]) / n_periods
        weight = 1.0 - lag / (int(lags) + 1.0)
        psi = psi + weight * (gamma + np.transpose(gamma, (0, 2, 1)))
    quad = np.einsum("bk,bkm,bm->b", gradient, psi, gradient)
    return np.asarray(quad, dtype=float).reshape(rows)


def _one_sample_pieces(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(sharpe, moments, gradient)`` for a ``(rows, T)`` matrix of returns."""
    mu = matrix.mean(axis=1)
    gamma = (matrix**2).mean(axis=1)
    var = gamma - mu**2
    if np.any(var <= 0.0) or not np.isfinite(var).all():
        raise BootstrapError("a resampled series has non-positive variance; the Sharpe standard error is undefined")
    sigma3 = var**1.5
    moments = np.stack([matrix, matrix**2], axis=2)
    gradient = np.stack([gamma / sigma3, -mu / (2.0 * sigma3)], axis=1)
    return _sharpe_rows(matrix), moments, gradient


def _paired_pieces(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(difference, moments, gradient)`` for two ``(rows, T)`` matrices."""
    mu_a = left.mean(axis=1)
    mu_b = right.mean(axis=1)
    gamma_a = (left**2).mean(axis=1)
    gamma_b = (right**2).mean(axis=1)
    var_a = gamma_a - mu_a**2
    var_b = gamma_b - mu_b**2
    if np.any(var_a <= 0.0) or np.any(var_b <= 0.0):
        raise BootstrapError("a resampled series has non-positive variance; the Sharpe standard error is undefined")
    sigma3_a = var_a**1.5
    sigma3_b = var_b**1.5
    moments = np.stack([left, right, left**2, right**2], axis=2)
    gradient = np.stack(
        [gamma_a / sigma3_a, -gamma_b / sigma3_b, -mu_a / (2.0 * sigma3_a), mu_b / (2.0 * sigma3_b)],
        axis=1,
    )
    difference = _sharpe_rows(left) - _sharpe_rows(right)
    return difference, moments, gradient


def _chunk_rows(n_periods: int, width: int) -> int:
    return max(1, int(_MAX_CHUNK_CELLS // max(1, n_periods * width)))


def _studentized_interval(
    statistic: float,
    standard_error: float,
    t_star: np.ndarray,
    confidence: float,
) -> tuple[float, float, float, float]:
    """``(lower, upper, t_statistic, p_two_sided)`` from the bootstrap-t distribution."""
    alpha = 1.0 - float(confidence)
    q_low = float(np.quantile(t_star, alpha / 2.0))
    q_high = float(np.quantile(t_star, 1.0 - alpha / 2.0))
    lower = statistic - standard_error * q_high
    upper = statistic - standard_error * q_low
    t_hat = statistic / standard_error if standard_error > 0.0 else float("nan")
    p_value = float(np.mean(np.abs(t_star) >= abs(t_hat))) if np.isfinite(t_hat) else float("nan")
    return float(lower), float(upper), float(t_hat), p_value


def sharpe_difference_interval(
    returns_a: Any,
    returns_b: Any,
    spec: SharpeBootstrapSpec,
    *,
    label: str = "a_minus_b",
) -> SharpeInterval:
    """Studentized interval for ``Sharpe(a) - Sharpe(b)`` under paired block resampling.

    Both arms are resampled on the *same* blocks, so the interval describes the difference
    on one sample rather than two independently drawn ones. The series must be aligned and
    of equal length: excluding a fold from one arm and keeping it in the other is refused
    upstream (§3.4) and would be undetectable here.
    """
    left = require_return_series(returns_a, "returns_a")
    right = require_return_series(returns_b, "returns_b")
    if left.size != right.size:
        raise ValueError(
            f"paired resampling needs the same sample: returns_a has {left.size} periods and returns_b has "
            f"{right.size}"
        )
    n_periods = left.size
    plan = plan_blocks(n_periods, spec.block_spec())
    if not plan.sufficient:
        raise BootstrapError(str(plan.exclusion_reason))

    observed_diff, moments, gradient = _paired_pieces(left[None, :], right[None, :])
    se_hat = math.sqrt(float(_hac_quadratic(moments, gradient, spec.lags)[0]) / n_periods)
    if not np.isfinite(se_hat) or se_hat <= 0.0:
        raise BootstrapError(f"Sharpe-difference standard error is not positive ({se_hat}); the interval is undefined")

    indices = block_resample_indices(n_periods, spec.block_spec())
    t_star = np.empty(indices.shape[0], dtype=float)
    step = _chunk_rows(n_periods, 4)
    for start in range(0, indices.shape[0], step):
        block = indices[start : start + step]
        diff_star, moments_star, gradient_star = _paired_pieces(left[block], right[block])
        se_star = np.sqrt(_hac_quadratic(moments_star, gradient_star, spec.lags) / n_periods)
        if not np.isfinite(se_star).all() or np.any(se_star <= 0.0):
            raise BootstrapError("a resample produced a non-positive Sharpe-difference standard error")
        t_star[start : start + block.shape[0]] = (diff_star - float(observed_diff[0])) / se_star

    lower, upper, t_hat, p_value = _studentized_interval(
        float(observed_diff[0]), se_hat, t_star, float(spec.confidence)
    )
    return SharpeInterval(
        label=label,
        statistic="sharpe_difference",
        sharpe_a=periodic_sharpe(left),
        sharpe_b=periodic_sharpe(right),
        difference=float(observed_diff[0]),
        mean_difference=float(left.mean() - right.mean()),
        standard_error=se_hat,
        lower=lower,
        upper=upper,
        confidence=float(spec.confidence),
        t_statistic=t_hat,
        p_value_two_sided=p_value,
        n_periods=n_periods,
        blocks_drawn=plan.blocks_drawn,
        n_resamples=int(spec.n_resamples),
        method=spec.method(),
    )


def sharpe_interval(
    returns: Any,
    spec: SharpeBootstrapSpec,
    *,
    label: str = "sharpe",
) -> SharpeInterval:
    """Studentized interval for one series' Sharpe, reported against zero.

    Applied to an excess-return series (``net - cash``) this is the promotion rule's read:
    "periodic net excess Sharpe positive against the declared cash series", stated on
    returns rather than as a Sharpe compared with a yield.
    """
    values = require_return_series(returns, "returns")
    n_periods = values.size
    plan = plan_blocks(n_periods, spec.block_spec())
    if not plan.sufficient:
        raise BootstrapError(str(plan.exclusion_reason))

    observed, moments, gradient = _one_sample_pieces(values[None, :])
    se_hat = math.sqrt(float(_hac_quadratic(moments, gradient, spec.lags)[0]) / n_periods)
    if not np.isfinite(se_hat) or se_hat <= 0.0:
        raise BootstrapError(f"Sharpe standard error is not positive ({se_hat}); the interval is undefined")

    indices = block_resample_indices(n_periods, spec.block_spec())
    t_star = np.empty(indices.shape[0], dtype=float)
    step = _chunk_rows(n_periods, 2)
    for start in range(0, indices.shape[0], step):
        block = indices[start : start + step]
        sharpe_star, moments_star, gradient_star = _one_sample_pieces(values[block])
        se_star = np.sqrt(_hac_quadratic(moments_star, gradient_star, spec.lags) / n_periods)
        if not np.isfinite(se_star).all() or np.any(se_star <= 0.0):
            raise BootstrapError("a resample produced a non-positive Sharpe standard error")
        t_star[start : start + block.shape[0]] = (sharpe_star - float(observed[0])) / se_star

    lower, upper, t_hat, p_value = _studentized_interval(float(observed[0]), se_hat, t_star, float(spec.confidence))
    return SharpeInterval(
        label=label,
        statistic="sharpe",
        sharpe_a=float(observed[0]),
        sharpe_b=0.0,
        difference=float(observed[0]),
        mean_difference=float(values.mean()),
        standard_error=se_hat,
        lower=lower,
        upper=upper,
        confidence=float(spec.confidence),
        t_statistic=t_hat,
        p_value_two_sided=p_value,
        n_periods=n_periods,
        blocks_drawn=plan.blocks_drawn,
        n_resamples=int(spec.n_resamples),
        method=spec.method(),
    )


def excess_sharpe_interval(
    net_returns: Any,
    cash_returns: Any,
    spec: SharpeBootstrapSpec,
    *,
    label: str = "net_excess",
) -> SharpeInterval:
    """Studentized interval for the Sharpe of ``net - cash`` (returns against returns)."""
    return sharpe_interval(excess_returns(net_returns, cash_returns), spec, label=label)


# ---------------------------------------------------------------------------
# Record access. The record layout is nested by category (§3.12); a field is looked
# up at the top level first and then one level in, so a comparison does not depend on
# which grouping a producer chose.
# ---------------------------------------------------------------------------


_MISSING = object()


def record_field(record: Mapping[str, Any], name: str, default: Any = _MISSING) -> Any:
    """Read ``name`` from a record, top level first, then one nesting level in."""
    if name in record:
        return record[name]
    for value in record.values():
        if isinstance(value, Mapping) and name in value:
            return value[name]
    if default is _MISSING:
        raise KeyError(name)
    return default


def record_has(record: Mapping[str, Any], name: str) -> bool:
    """True when the record carries ``name``, at the top level or one level in.

    Distinguishes "absent" from "recorded as null": ``prospective`` is legitimately null on
    a record that is not prospective, and a missing field is a different fact.
    """
    if name in record:
        return True
    return any(isinstance(value, Mapping) and name in value for value in record.values())


@dataclass(frozen=True)
class ComparisonArm:
    """One arm of a comparison: its record and the series the comparison reads.

    ``net_returns`` is the arm's periodic net return series. ``costs`` is the per-period
    charged cost as a fraction of equity, positive as a debit, and is what a repricing
    rescales. ``weights`` is the decided weight panel (periods by sid) used for the holdings
    overlap panel. Everything else is read from ``record`` and passed through as recorded.
    """

    record: Mapping[str, Any]
    net_returns: pd.Series
    gross_returns: pd.Series | None = None
    costs: pd.Series | None = None
    weights: pd.DataFrame | None = None
    turnover: pd.Series | None = None
    by_fold: Mapping[str, Any] | None = None

    @property
    def run_id(self) -> str:
        return str(record_field(self.record, "run_id"))

    @property
    def arm_id(self) -> str:
        value = record_field(self.record, "arm_id", None)
        return self.run_id if value is None else str(value)

    def flags(self) -> dict[str, Any]:
        """The provenance labels every panel carries (§3.12)."""
        return {flag: record_field(self.record, flag, UNAVAILABLE) for flag in PROVENANCE_FLAGS}

    def conditions(self) -> dict[str, Any]:
        return {name: record_field(self.record, name, UNAVAILABLE) for name in CONDITION_FIELDS}


def _require_flags(arm: ComparisonArm) -> None:
    missing = [flag for flag in PROVENANCE_FLAGS if not record_has(arm.record, flag)]
    if missing:
        raise ComparisonRefused(
            f"record {arm.run_id} does not carry {missing}; every comparison panel is labeled with the "
            "record's provenance flags, and an unlabeled panel displays a synthetic demonstration and a "
            "live-fill result alike (§3.12)"
        )


def _refuse_incomparable(arms: Sequence[ComparisonArm]) -> None:
    if len(arms) < 2:
        raise ComparisonRefused(f"a comparison needs at least two arms, got {len(arms)}")
    seen: set[str] = set()
    for arm in arms:
        _require_flags(arm)
        if arm.run_id in seen:
            raise ComparisonRefused(f"run_id {arm.run_id} appears twice; a run is compared with a different run")
        seen.add(arm.run_id)
        if str(record_field(arm.record, "run_channel", UNAVAILABLE)) == ENGINEERING_CHANNEL:
            raise ComparisonRefused(
                f"record {arm.run_id} is on the engineering channel, which no promotion or comparison may "
                "cite (§3.12)"
            )
        if not record_has(arm.record, "config_id"):
            raise ComparisonRefused(f"record {arm.run_id} carries no config_id")
        if record_field(arm.record, "config_id", None) is None:
            raise ComparisonRefused(
                f"record {arm.run_id} carries config_id = null (an imported legacy row). A legacy config_id "
                "is never comparable with a post-overhaul config_id (§3.12)"
            )

    modes = {str(record_field(arm.record, "visibility_mode", UNAVAILABLE)) for arm in arms}
    if len(modes) > 1:
        raise ComparisonRefused(
            f"records carry different visibility modes {sorted(modes)}; a research_vintage result and a "
            "replay_recorded result are not the same evidence class (§10.1)"
        )

    lengths = {arm.run_id: len(arm.net_returns) for arm in arms}
    if len(set(lengths.values())) > 1:
        raise ComparisonRefused(f"arms carry different sample lengths {lengths}; the comparison is same-sample")
    reference = arms[0]
    for arm in arms[1:]:
        if not arm.net_returns.index.equals(reference.net_returns.index):
            raise ComparisonRefused(
                f"arm {arm.run_id} and arm {reference.run_id} carry different return indexes; the comparison "
                "is same-sample and paired resampling would silently misalign them"
            )

    geometries = {
        str(record_field(arm.record, "fold_table_sha256", record_field(arm.record, "fold_geometry", UNAVAILABLE)))
        for arm in arms
    }
    if len(geometries) > 1:
        raise ComparisonRefused(
            f"arms were evaluated on different fold geometries {sorted(geometries)}; a fold is never dropped "
            "for one arm and kept for another (§3.4)"
        )


def _holdings_overlap(left: ComparisonArm, right: ComparisonArm) -> dict[str, Any]:
    if left.weights is None or right.weights is None:
        return {"status": UNAVAILABLE, "reason": "one or both arms supplied no weight panel"}
    a = left.weights
    b = right.weights
    if not a.index.equals(b.index):
        raise ComparisonRefused(f"weight panels of {left.run_id} and {right.run_id} carry different decision indexes")
    columns = sorted(set(a.columns) | set(b.columns))
    aw = a.reindex(columns=columns).fillna(0.0)
    bw = b.reindex(columns=columns).fillna(0.0)
    l1 = (aw - bw).abs().sum(axis=1)
    same_sign = np.sign(aw) == np.sign(bw)
    common = (np.minimum(aw.abs(), bw.abs()) * same_sign).sum(axis=1)
    held_a = aw.abs() > 0.0
    held_b = bw.abs() > 0.0
    union = (held_a | held_b).sum(axis=1)
    intersection = (held_a & held_b).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union.to_numpy() > 0, intersection.to_numpy() / np.where(union > 0, union, 1), np.nan)
    return {
        "status": "computed",
        "mean_l1_distance": float(l1.mean()),
        "mean_common_weight": float(common.mean()),
        "mean_name_jaccard": float(np.nanmean(jaccard)) if len(jaccard) else float("nan"),
        "definition": (
            "l1 distance = sum |w_a - w_b| per decision; common weight = sum min(|w_a|, |w_b|) over names held "
            "on the same side; name jaccard = |held_a and held_b| / |held_a or held_b|"
        ),
    }


def _series_panel(arm: ComparisonArm) -> dict[str, Any]:
    panel: dict[str, Any] = {
        "run_id": arm.run_id,
        "arm_id": arm.arm_id,
        "flags": arm.flags(),
        "net_sharpe": periodic_sharpe(arm.net_returns),
        "net_mean": float(np.asarray(arm.net_returns, dtype=float).mean()),
        "n_periods": int(len(arm.net_returns)),
    }
    panel["gross_sharpe"] = periodic_sharpe(arm.gross_returns) if arm.gross_returns is not None else UNAVAILABLE
    panel["turnover_mean"] = (
        float(np.asarray(arm.turnover, dtype=float).mean()) if arm.turnover is not None else UNAVAILABLE
    )
    panel["by_fold"] = dict(arm.by_fold) if arm.by_fold is not None else UNAVAILABLE
    panel["exposures"] = record_field(arm.record, "exposures", UNAVAILABLE)
    panel["cost_decomposition"] = record_field(arm.record, "costs", UNAVAILABLE)
    panel["capital"] = record_field(arm.record, "capital", UNAVAILABLE)
    panel["coverage"] = {
        "untested_conclusions": record_field(arm.record, "untested_conclusions", UNAVAILABLE),
        "bias_status": record_field(arm.record, "bias_status", UNAVAILABLE),
    }
    return panel


@dataclass(frozen=True)
class ComparisonReport:
    """The reevaluation report of §3.12: one report, blocked by capital scenario."""

    kind: Literal["reevaluation"]
    arms: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    conditions_held_fixed: dict[str, Any]
    conditions_that_differ: dict[str, Any]
    n_intervals: int
    method: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "arms": deepcopy(self.arms),
            "blocks": deepcopy(self.blocks),
            "conditions_held_fixed": deepcopy(self.conditions_held_fixed),
            "conditions_that_differ": deepcopy(self.conditions_that_differ),
            "selection": {"n_intervals": self.n_intervals},
            "method": deepcopy(self.method),
            "warnings": list(self.warnings),
            "is_evidence_row": False,
            "writes_trial_row": False,
        }


def compare_reevaluation(
    arms: Sequence[ComparisonArm],
    spec: SharpeBootstrapSpec,
    *,
    cash_returns: Any | None = None,
) -> ComparisonReport:
    """Build the comparison report over records ``prism evaluate`` already produced.

    Arms are blocked by the ``capital`` recorded in each record: a capital scenario is a
    recorded condition of the experiment, not an additional arm, so differences are computed
    within a capital level and never across two. Every pairwise difference inside a block is
    reported, and the count of intervals computed is carried in the report's selection block
    because the multiplicity of comparisons is part of the selection history.

    ``cash_returns``, when given, adds each arm's excess-return Sharpe against that declared
    cash series — returns compared with returns.
    """
    _refuse_incomparable(arms)

    conditions = {arm.run_id: arm.conditions() for arm in arms}
    held_fixed: dict[str, Any] = {}
    differs: dict[str, Any] = {}
    for name in CONDITION_FIELDS:
        values = {arm.run_id: conditions[arm.run_id][name] for arm in arms}
        distinct = {repr(value) for value in values.values()}
        if len(distinct) == 1:
            held_fixed[name] = next(iter(values.values()))
        else:
            differs[name] = values

    warnings: list[str] = []
    for arm in arms:
        flags = arm.flags()
        if flags.get("synthetic") is True:
            warnings.append(f"{arm.run_id}: synthetic run; its panels are a demonstration, not evidence of edge")
        if str(flags.get("fill_source", UNAVAILABLE)) in {"simulated", UNAVAILABLE}:
            warnings.append(f"{arm.run_id}: fill_source={flags.get('fill_source')}; fills are not actual fills")
    if str(held_fixed.get("visibility_mode", UNAVAILABLE)) == UNAVAILABLE:
        warnings.append(
            "visibility_mode is unavailable on every arm; the comparison asserts nothing about what this "
            "workspace could have known at the decision (§3.2)"
        )

    by_capital: dict[str, list[ComparisonArm]] = {}
    for arm in arms:
        key = repr(record_field(arm.record, "capital", UNAVAILABLE))
        by_capital.setdefault(key, []).append(arm)

    blocks: list[dict[str, Any]] = []
    n_intervals = 0
    for capital_key in sorted(by_capital):
        members = by_capital[capital_key]
        differences: list[dict[str, Any]] = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                left, right = members[i], members[j]
                interval = sharpe_difference_interval(
                    left.net_returns,
                    right.net_returns,
                    spec,
                    label=f"{left.arm_id}_minus_{right.arm_id}",
                )
                n_intervals += 1
                payload = interval.to_dict()
                payload["arm_a"] = left.arm_id
                payload["arm_b"] = right.arm_id
                payload["flags_a"] = left.flags()
                payload["flags_b"] = right.flags()
                payload["holdings_overlap"] = _holdings_overlap(left, right)
                differences.append(payload)
        block: dict[str, Any] = {
            "capital": record_field(members[0].record, "capital", UNAVAILABLE),
            "arms": [arm.arm_id for arm in members],
            "sharpe_differences": differences,
        }
        if cash_returns is not None:
            excess: list[dict[str, Any]] = []
            for arm in members:
                interval = excess_sharpe_interval(
                    arm.net_returns, cash_returns, spec, label=f"{arm.arm_id}_net_excess"
                )
                n_intervals += 1
                entry = interval.to_dict()
                entry["arm"] = arm.arm_id
                entry["flags"] = arm.flags()
                excess.append(entry)
            block["excess_return_intervals"] = excess
        blocks.append(block)

    return ComparisonReport(
        kind="reevaluation",
        arms=[_series_panel(arm) for arm in arms],
        blocks=blocks,
        conditions_held_fixed=held_fixed,
        conditions_that_differ=differs,
        n_intervals=n_intervals,
        method=spec.method(),
        warnings=warnings,
    )


@dataclass(frozen=True)
class RepricingReport:
    """The repricing diagnostic of §3.12: a counterfactual attached to one record."""

    kind: Literal["repricing"]
    run_id: str
    cost_multiplier: float
    accounting_engine: str
    exact: bool
    exactness_note: str
    flags: dict[str, Any]
    recorded: dict[str, Any]
    counterfactual: dict[str, Any]
    deviation: dict[str, Any]
    interval: dict[str, Any] | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "cost_multiplier": self.cost_multiplier,
            "accounting_engine": self.accounting_engine,
            "exact": self.exact,
            "exactness_note": self.exactness_note,
            "flags": deepcopy(self.flags),
            "recorded": deepcopy(self.recorded),
            "counterfactual": deepcopy(self.counterfactual),
            "deviation": deepcopy(self.deviation),
            "interval": deepcopy(self.interval),
            "warnings": list(self.warnings),
            "label": "counterfactual",
            "is_evidence_row": False,
            "writes_trial_row": False,
        }


#: Repricing is exact only under this engine (§3.12).
EXACT_REPRICING_ENGINE = "historical_constant_weight"


def compare_repricing(
    arm: ComparisonArm,
    *,
    cost_multiplier: float,
    spec: SharpeBootstrapSpec | None = None,
) -> RepricingReport:
    """Recompute cost debits at ``cost_multiplier`` with fills held fixed.

    Fill quantities and fill prices are not re-simulated: the recorded per-period charged
    cost is rescaled and the net return becomes ``r_t + (1 - m) * c_t``. The resulting
    equity path is labeled counterfactual and its deviation from the recorded path is
    reported. The report is attached to the original record as a diagnostic
    (:func:`attach_repricing`); it is never a new evidence row and writes no trial row.

    Exact only under ``historical_constant_weight``. Under the canonical engine a
    fixed-share repricing produces a cash path the accounting would not have produced, so
    ``exact`` is False and the deviation is reported rather than presented as an outcome.
    """
    _require_flags(arm)
    multiplier = float(cost_multiplier)
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError(f"cost_multiplier must be finite and >= 0, got {cost_multiplier}")
    if arm.costs is None:
        raise ValueError(
            f"record {arm.run_id} supplied no per-period charged-cost series; a repricing rescales recorded "
            "cost debits and cannot infer them from a net return series"
        )
    net = require_return_series(arm.net_returns, "net_returns")
    costs = require_return_series(arm.costs, "costs")
    if net.size != costs.size:
        raise ValueError(f"net_returns has {net.size} periods and costs has {costs.size}; align them first")
    if np.any(costs < 0.0):
        raise ValueError("costs carries negative entries; the charged cost is a positive debit per period")

    repriced = net + (1.0 - multiplier) * costs
    recorded_equity = np.cumprod(1.0 + net)
    counterfactual_equity = np.cumprod(1.0 + repriced)
    deviation_path = counterfactual_equity - recorded_equity

    engine = str(record_field(arm.record, "accounting_engine", UNAVAILABLE))
    exact = engine == EXACT_REPRICING_ENGINE
    note = (
        "Fills are held fixed and cost debits rescaled; under historical_constant_weight the cash path is the "
        "recorded one with the new debits, so the repricing is exact."
        if exact
        else (
            f"accounting_engine={engine!r}: a fixed-share repricing produces a cash path the accounting would "
            "not have produced. The deviation below is reported as a deviation, not as an outcome."
        )
    )

    interval: dict[str, Any] | None = None
    if spec is not None:
        interval = sharpe_difference_interval(
            repriced, net, spec, label=f"{arm.arm_id}_repriced_minus_recorded"
        ).to_dict()

    warnings: list[str] = []
    if not exact:
        warnings.append("repricing is not exact under this accounting engine; read the deviation, not the level")
    return RepricingReport(
        kind="repricing",
        run_id=arm.run_id,
        cost_multiplier=multiplier,
        accounting_engine=engine,
        exact=exact,
        exactness_note=note,
        flags=arm.flags(),
        recorded={
            "sharpe": periodic_sharpe(net),
            "mean_return": float(net.mean()),
            "total_cost": float(costs.sum()),
            "final_equity": float(recorded_equity[-1]),
        },
        counterfactual={
            "sharpe": periodic_sharpe(repriced),
            "mean_return": float(repriced.mean()),
            "total_cost": float(multiplier * costs.sum()),
            "final_equity": float(counterfactual_equity[-1]),
        },
        deviation={
            "final_equity": float(deviation_path[-1]),
            "max_abs_equity": float(np.max(np.abs(deviation_path))),
            "mean_return": float(repriced.mean() - net.mean()),
            "sharpe": float(periodic_sharpe(repriced) - periodic_sharpe(net)),
            "n_periods": int(net.size),
        },
        interval=interval,
        warnings=warnings,
    )


def attach_repricing(record: Mapping[str, Any], report: RepricingReport) -> dict[str, Any]:
    """Return a copy of ``record`` with the repricing appended under ``diagnostics``.

    The input record is not modified: a record is append-only evidence, and a repricing is a
    diagnostic attached to it, never a new evidence row and never an edit of one.
    """
    if str(record_field(record, "run_id")) != report.run_id:
        raise ComparisonRefused(
            f"repricing was computed for run_id {report.run_id} and cannot attach to "
            f"{record_field(record, 'run_id')}"
        )
    attached = deepcopy(dict(record))
    diagnostics = attached.setdefault("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise ComparisonRefused("record's diagnostics field is not a mapping; refusing to overwrite it")
    repricings = diagnostics.setdefault("repricings", [])
    if not isinstance(repricings, list):
        raise ComparisonRefused("record's diagnostics.repricings field is not a list; refusing to overwrite it")
    repricings.append(report.to_dict())
    return attached


__all__ = [
    "CONDITION_FIELDS",
    "ENGINEERING_CHANNEL",
    "EXACT_REPRICING_ENGINE",
    "PROVENANCE_FLAGS",
    "ComparisonArm",
    "ComparisonRefused",
    "ComparisonReport",
    "RepricingReport",
    "SharpeBootstrapSpec",
    "SharpeInterval",
    "attach_repricing",
    "compare_reevaluation",
    "compare_repricing",
    "constant_hurdle_returns",
    "excess_returns",
    "excess_sharpe_interval",
    "record_field",
    "record_has",
    "require_return_series",
    "sharpe_difference_interval",
    "sharpe_interval",
]
