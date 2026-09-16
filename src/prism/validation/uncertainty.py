"""Block bootstrap over decision periods (engine specification §3.4, "Uncertainty").

One resampler serves every uncertainty read in the engine: the calibration slope's
endpoints (§3.4), the studentized Sharpe-difference intervals of
:mod:`prism.validation.compare` (§3.12), and any later statistic that must respect the
serial dependence overlapping labels induce.

The scheme is declared, never inferred:

* **Units are decision periods, not observations.** A resample draws whole periods, so
  every drawn period brings its entire cross-section intact. Resampling names inside a
  cross-section would destroy exactly the dependence the block scheme exists to carry.
* ``block_scheme = "circular_moving"`` takes the ``n`` circular windows of
  ``block_periods`` consecutive periods as the block set. Each resample draws
  ``ceil(n / block_periods)`` blocks with replacement, concatenates them **in draw
  order**, and truncates to ``n`` periods, so every resample has the original length.
* ``block_periods * update_every_bars`` must be at least ``horizon_bars`` so a block
  spans at least one label — :func:`validate_block_span` is the check, and it raises
  rather than warning.
* Randomness is one PCG64 stream seeded with ``seed`` and drawn once per fit, so a refit
  reproduces the same resamples byte for byte.

**Sufficiency.** A bootstrap that would draw fewer than ``min_blocks`` blocks is not run.
That is an *exclusion*, recorded with its reason, ``blocks_drawn``, and the resolved
method block (§3.4: "a fold below the declared minimums is an exclusion, not a failure"),
and it is returned as a :class:`BootstrapResult` with ``excluded=True`` and no values.
Reading an interval off an excluded result raises :class:`BootstrapExcluded`; a caller
that wants a number from an excluded fold has to say so in its own code, not inherit a
silent NaN. A non-finite statistic on a resample is the other case and is an estimation
*failure*: it raises :class:`BootstrapError`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import pandas as pd

#: The only block scheme this stage implements (§3.4).
CIRCULAR_MOVING = "circular_moving"
BLOCK_SCHEMES: tuple[str, ...] = (CIRCULAR_MOVING,)

T = TypeVar("T")


class BootstrapError(RuntimeError):
    """The bootstrap cannot be computed as declared (an estimation failure)."""


class BootstrapExcluded(RuntimeError):
    """A quantity was read off a bootstrap that was excluded for insufficient blocks."""


@dataclass(frozen=True)
class BlockBootstrapSpec:
    """The declared resampling scheme (§3.4: every key is recorded in the result).

    ``block_periods`` is a count of decision periods (or of bars, when the caller's unit
    is a bar — :mod:`prism.validation.compare` declares its block length in bars).
    ``min_blocks`` is the sufficiency floor: a sample that would draw fewer blocks is
    excluded rather than resampled.
    """

    block_periods: int
    n_resamples: int
    seed: int
    block_scheme: str = CIRCULAR_MOVING
    min_blocks: int = 1

    def __post_init__(self) -> None:
        if self.block_scheme not in BLOCK_SCHEMES:
            raise ValueError(f"block_scheme must be one of {BLOCK_SCHEMES}, got {self.block_scheme!r}")
        if int(self.block_periods) < 1:
            raise ValueError(f"block_periods must be >= 1, got {self.block_periods}")
        if int(self.n_resamples) < 1:
            raise ValueError(f"n_resamples must be >= 1, got {self.n_resamples}")
        if int(self.min_blocks) < 1:
            raise ValueError(f"min_blocks must be >= 1, got {self.min_blocks}")

    def blocks_drawn(self, n_units: int) -> int:
        """Blocks drawn per resample: ``ceil(n_units / block_periods)`` (§3.4)."""
        if int(n_units) < 1:
            raise ValueError(f"n_units must be >= 1, got {n_units}")
        return int(math.ceil(int(n_units) / int(self.block_periods)))

    def method(self) -> dict[str, Any]:
        """The resolved method block, as it is recorded beside every estimate."""
        return {
            "block_scheme": self.block_scheme,
            "block_periods": int(self.block_periods),
            "n_resamples": int(self.n_resamples),
            "seed": int(self.seed),
            "min_blocks": int(self.min_blocks),
            "units": "decision_periods",
            "coverage_statement": (
                "Endpoints are directional statements under the resampling assumptions: the pool is "
                "stationary enough to resample and block_periods carries the dependence of overlapping "
                "labels. They are not a simultaneous interval and say nothing about realized portfolio "
                "return coverage."
            ),
        }


@dataclass(frozen=True)
class BlockPlan:
    """What a bootstrap would draw on a sample of ``n_units``, and whether it is run."""

    n_units: int
    block_periods: int
    blocks_drawn: int
    min_blocks: int
    sufficient: bool
    exclusion_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_units": self.n_units,
            "block_periods": self.block_periods,
            "blocks_drawn": self.blocks_drawn,
            "min_blocks": self.min_blocks,
            "sufficient": self.sufficient,
            "exclusion_reason": self.exclusion_reason,
        }


def validate_block_span(block_periods: int, update_every_bars: int, horizon_bars: int) -> None:
    """Raise unless a block spans at least one label (§3.4).

    ``block_periods * update_every_bars >= horizon_bars`` is the constraint that makes the
    block scheme carry the overlap of consecutive labels. A configuration that violates it
    is refused, not silently widened.
    """
    if int(block_periods) < 1:
        raise ValueError(f"block_periods must be >= 1, got {block_periods}")
    if int(update_every_bars) < 1:
        raise ValueError(f"update_every_bars must be >= 1, got {update_every_bars}")
    if int(horizon_bars) < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    span = int(block_periods) * int(update_every_bars)
    if span < int(horizon_bars):
        raise ValueError(
            f"block_periods * update_every_bars = {span} bars is shorter than horizon_bars="
            f"{int(horizon_bars)}; a block must span at least one label (§3.4)"
        )


def plan_blocks(n_units: int, spec: BlockBootstrapSpec) -> BlockPlan:
    """Resolve the block geometry for ``n_units`` and say whether the bootstrap runs."""
    n = int(n_units)
    if n < 1:
        raise ValueError(f"n_units must be >= 1, got {n_units}")
    if int(spec.block_periods) > n:
        raise ValueError(
            f"block_periods={int(spec.block_periods)} exceeds the sample of {n} units; a block longer "
            "than the sample is not a resample"
        )
    drawn = spec.blocks_drawn(n)
    sufficient = drawn >= int(spec.min_blocks)
    reason = None
    if not sufficient:
        reason = (
            f"bootstrap would draw {drawn} blocks from {n} units at block_periods="
            f"{int(spec.block_periods)}, below the declared min_blocks={int(spec.min_blocks)}"
        )
    return BlockPlan(
        n_units=n,
        block_periods=int(spec.block_periods),
        blocks_drawn=drawn,
        min_blocks=int(spec.min_blocks),
        sufficient=sufficient,
        exclusion_reason=reason,
    )


def block_resample_indices(
    n_units: int,
    spec: BlockBootstrapSpec,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``(n_resamples, n_units)`` matrix of unit indices, one row per resample.

    Row ``b`` is the concatenation, in draw order, of ``ceil(n / block_periods)`` circular
    windows of ``block_periods`` consecutive units drawn with replacement, truncated to
    ``n`` units. ``rng`` defaults to a PCG64 stream seeded with ``spec.seed``, so the
    matrix is a deterministic function of ``(n_units, spec)``.

    Raises :class:`BootstrapExcluded` when the plan is insufficient — callers that want the
    exclusion as data call :func:`plan_blocks` or :func:`block_bootstrap` instead.
    """
    plan = plan_blocks(n_units, spec)
    if not plan.sufficient:
        raise BootstrapExcluded(str(plan.exclusion_reason))
    generator = np.random.Generator(np.random.PCG64(int(spec.seed))) if rng is None else rng
    n = plan.n_units
    block = plan.block_periods
    starts = generator.integers(0, n, size=(int(spec.n_resamples), plan.blocks_drawn))
    offsets = np.arange(block, dtype=np.int64)
    windows = (starts[:, :, None] + offsets[None, None, :]) % n
    return windows.reshape(int(spec.n_resamples), plan.blocks_drawn * block)[:, :n].astype(np.int64, copy=False)


@dataclass(frozen=True)
class BootstrapResult:
    """Resampled values of one statistic, or the recorded reason there are none."""

    observed: float
    values: np.ndarray
    n_units: int
    blocks_drawn: int
    excluded: bool
    exclusion_reason: str | None
    method: dict[str, Any]

    def _require_values(self) -> np.ndarray:
        if self.excluded:
            raise BootstrapExcluded(
                f"no resampled values: {self.exclusion_reason}. The exclusion is recorded; it is not a NaN."
            )
        return self.values

    def quantile(self, q: float) -> float:
        """Linear-interpolation quantile of the resampled statistic (§3.4 endpoints)."""
        if not 0.0 <= float(q) <= 1.0:
            raise ValueError(f"q must lie in [0, 1], got {q}")
        return float(np.quantile(self._require_values(), float(q)))

    def one_sided_endpoints(self, confidence: float) -> tuple[float, float]:
        """``(quantile(1 - confidence), quantile(confidence))`` — the §3.4 endpoints."""
        if not 0.5 < float(confidence) < 1.0:
            raise ValueError(f"confidence must lie in (0.5, 1.0), got {confidence}")
        values = self._require_values()
        return (
            float(np.quantile(values, 1.0 - float(confidence))),
            float(np.quantile(values, float(confidence))),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary; the resampled values themselves are not embedded."""
        payload: dict[str, Any] = {
            "observed": self.observed,
            "n_units": self.n_units,
            "blocks_drawn": self.blocks_drawn,
            "n_values": int(self.values.size),
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
            "method": dict(self.method),
        }
        if not self.excluded:
            payload["mean"] = float(np.mean(self.values))
            payload["std"] = float(np.std(self.values, ddof=1)) if self.values.size > 1 else float("nan")
        return payload


def _excluded_result(observed: float, plan: BlockPlan, spec: BlockBootstrapSpec) -> BootstrapResult:
    return BootstrapResult(
        observed=float(observed),
        values=np.empty(0, dtype=float),
        n_units=plan.n_units,
        blocks_drawn=plan.blocks_drawn,
        excluded=True,
        exclusion_reason=plan.exclusion_reason,
        method=spec.method(),
    )


def block_bootstrap(
    units: Sequence[T],
    statistic: Callable[[list[T]], float],
    spec: BlockBootstrapSpec,
    *,
    observed: float | None = None,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Resample whole units and recompute ``statistic`` on each resample.

    ``units`` is one entry per decision period — a cross-section, a fold's frame, a scalar
    return, whatever the statistic consumes. The unit is never split, so a drawn period
    arrives with its cross-section intact. ``statistic`` receives the resampled units as a
    list in draw order and returns one float; ``observed`` defaults to
    ``statistic(list(units))``.

    Returns an excluded :class:`BootstrapResult` when the plan draws fewer than
    ``min_blocks`` blocks. Raises :class:`BootstrapError` when a resample's statistic is
    not finite: that is an estimation failure, not an exclusion.
    """
    materialized = list(units)
    n = len(materialized)
    if n < 1:
        raise ValueError("units is empty; a bootstrap needs at least one decision period")
    point = float(statistic(materialized)) if observed is None else float(observed)
    plan = plan_blocks(n, spec)
    if not plan.sufficient:
        return _excluded_result(point, plan, spec)

    indices = block_resample_indices(n, spec, rng=rng)
    values = np.empty(indices.shape[0], dtype=float)
    for row, index_row in enumerate(indices):
        value = float(statistic([materialized[int(i)] for i in index_row]))
        if not np.isfinite(value):
            raise BootstrapError(
                f"resample {row} produced a non-finite statistic ({value}); a degenerate estimate is a "
                "failure, not a wide interval (§3.4)"
            )
        values[row] = value
    return BootstrapResult(
        observed=point,
        values=values,
        n_units=n,
        blocks_drawn=plan.blocks_drawn,
        excluded=False,
        exclusion_reason=None,
        method=spec.method(),
    )


def period_units(frame: pd.DataFrame, period_column: str) -> list[pd.DataFrame]:
    """Split a long panel into one frame per decision period, in period order.

    The unit of resampling is the whole cross-section of a decision period, so this is the
    canonical way to hand a panel to :func:`block_bootstrap`.
    """
    if period_column not in frame.columns:
        raise KeyError(f"period_column {period_column!r} is not a column of the panel: {list(frame.columns)}")
    if frame.empty:
        raise ValueError("panel is empty; a bootstrap needs at least one decision period")
    return [group for _, group in frame.groupby(period_column, sort=True)]


def concat_units(units: Sequence[pd.DataFrame], *, period_column: str = "resample_period") -> pd.DataFrame:
    """Concatenate resampled cross-sections, relabelling each draw as its own period.

    A period drawn twice must count twice, and per-date-equal weighting must weight each
    *draw* rather than pooling the repeats into one heavier cross-section, so the draw
    order becomes the new period label.
    """
    if not len(units):
        raise ValueError("no units to concatenate")
    stamped = []
    for position, unit in enumerate(units):
        block = unit.copy()
        block[period_column] = position
        stamped.append(block)
    return pd.concat(stamped, ignore_index=True)


__all__ = [
    "BLOCK_SCHEMES",
    "CIRCULAR_MOVING",
    "BlockBootstrapSpec",
    "BlockPlan",
    "BootstrapError",
    "BootstrapExcluded",
    "BootstrapResult",
    "block_bootstrap",
    "block_resample_indices",
    "concat_units",
    "period_units",
    "plan_blocks",
    "validate_block_span",
]
