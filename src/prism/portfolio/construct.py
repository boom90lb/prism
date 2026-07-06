"""Shared score-to-book construction helpers.

The functions in this module deliberately stop at close-time target weights.
Execution timing, rebalance suppression against filled weights, borrow, and
capacity costs remain in ``prism.execution.target_weights``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cap_book(targets: pd.DataFrame, max_gross: float, max_symbol_abs_weight: float) -> pd.DataFrame:
    """Scale rows over ``max_gross`` down proportionally, then clip per symbol.

    This never scales a low-gross row up. A directional scorer that emits only
    30% gross remains 30% gross, leaving the rest in cash.
    """
    if max_gross <= 0:
        raise ValueError(f"max_gross must be > 0, got {max_gross}")
    if max_symbol_abs_weight <= 0:
        raise ValueError(f"max_symbol_abs_weight must be > 0, got {max_symbol_abs_weight}")
    out = targets.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = out.abs().sum(axis=1)
    scale = (max_gross / gross.where(gross > max_gross)).fillna(1.0).clip(upper=1.0)
    return out.mul(scale, axis=0).clip(lower=-max_symbol_abs_weight, upper=max_symbol_abs_weight)


def _band_step(held: np.ndarray, target: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """The one hysteresis rule shared by the batch and online band forms: move a
    name to its target only where the target is more than ``thr`` from what is
    held. Thresholds arrive from ``_coerce_band_thresholds`` as finite values
    or +inf (an explicit never-trade pin — the comparison never triggers)."""
    return np.where(np.abs(target - held) > thr, target, held)


def _coerce_band_thresholds(band: float | pd.Series, index: pd.Index) -> np.ndarray:
    """One band-threshold policy for the batch and online forms.

    Numeric-coerce and clip negatives to 0. A +inf band is a legitimate
    explicit never-trade pin and passes through. A NaN (or -inf) band is
    DISABLED (0.0) with a loud warning: a degenerate estimate to surface, not
    a hold-forever instruction (SPEC N7). A name absent from a per-name
    Series simply has no band (0.0), without a warning.
    """
    if isinstance(band, pd.Series):
        aligned = pd.to_numeric(band.reindex(index), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(aligned) | (aligned == np.inf)
        if not valid.all():
            degenerate = index.isin(band.index) & ~valid
            n_degenerate = int(degenerate.sum())
            if n_degenerate:
                logger.warning(
                    "no-trade band disabled for %d name(s) with NaN/-inf band values: %s%s",
                    n_degenerate,
                    [str(s) for s in index[degenerate][:10]],
                    " ..." if n_degenerate > 10 else "",
                )
        return np.clip(np.where(valid, aligned, 0.0), 0.0, None)
    value = float(band)
    if np.isnan(value) or value == -np.inf:
        logger.warning("no-trade band disabled: NaN/-inf scalar band %r", band)
        value = 0.0
    return np.full(len(index), max(value, 0.0))


def apply_no_trade_band(targets: pd.DataFrame, band: float | pd.Series) -> pd.DataFrame:
    """Hold each name's weight until the target moves by more than ``band``.

    Batch/backtest form: replays hysteresis from a flat book (``held=0``) over the
    whole frame. For an ONLINE loop that owns yesterday's held weights, use
    ``step_no_trade_band`` instead — calling this on a one-row frame every day
    silently resets hysteresis to flat and defeats the band.

    Band policy (shared with ``step_no_trade_band``, pinned by
    ``tests/test_online_band.py``): +inf pins a name at its held weight;
    negative, NaN, or -inf values disable the band for that name (threshold
    0), the degenerate ones with a loud warning.
    """
    thr = _coerce_band_thresholds(band, targets.columns)
    # Scalar fast path only: a Series band always runs the loop so the output
    # is a fresh float frame with NaN targets coerced to held weights, exactly
    # like every banded path.
    if not isinstance(band, pd.Series) and not np.any(thr > 0.0):
        return targets
    mat = targets.to_numpy(dtype=float)
    out = np.empty_like(mat)
    held = np.zeros(mat.shape[1])
    for t in range(mat.shape[0]):
        held = _band_step(held, mat[t], thr)
        out[t] = held
    return pd.DataFrame(out, index=targets.index, columns=targets.columns)


def step_no_trade_band(
    prev_held: pd.Series,
    target: pd.Series,
    band: float | pd.Series,
) -> pd.Series:
    """One online step of the no-trade band (SPEC.md §7.3, the primary-lever seed).

    Given the weights actually held after yesterday's fill (``prev_held``) and
    today's fresh target (``target``), move a name only when the target has moved
    more than ``band`` away from what is held; otherwise keep the held weight.
    This is the stateful, restart-safe analogue of ``apply_no_trade_band`` — it
    takes the prior book as input rather than replaying from flat, so a daily loop
    that calls it once per session preserves hysteresis across days.

    ``band`` is a scalar half-width or a per-name Series, coerced by the shared
    policy (+inf -> never-trade pin; negative/NaN/-inf -> band disabled, the
    degenerate ones warned loudly — see ``_coerce_band_thresholds``). The two
    Series are aligned on the union of
    their indices; a name absent from ``prev_held`` is treated as held-flat
    (0.0), a name absent from ``target`` keeps its held weight (no decision
    today). Returns the new held weights.
    """
    index = prev_held.index.union(target.index)
    held = pd.to_numeric(prev_held.reindex(index), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    tgt = pd.to_numeric(target.reindex(index), errors="coerce").to_numpy(dtype=float)
    # No decision for a name (NaN target) -> hold prior.
    tgt = np.where(np.isfinite(tgt), tgt, held)

    thr = _coerce_band_thresholds(band, index)

    new_held = _band_step(held, tgt, thr)
    return pd.Series(new_held, index=index, name=getattr(target, "name", None))


def cost_aware_band(
    half_life_bars: np.ndarray, per_trade_cost_frac: float, gamma: float = 1.0
) -> np.ndarray:
    """Heuristic no-trade half-width from OU speed and round-trip cost."""
    hl = np.asarray(half_life_bars, dtype=float)
    valid = np.isfinite(hl) & (hl > 0.0)
    kappa = np.where(valid, np.log(2.0) / np.where(valid, hl, 1.0), np.nan)
    with np.errstate(invalid="ignore"):
        band = gamma * np.sqrt(max(per_trade_cost_frac, 0.0) / kappa)
    return np.where(np.isfinite(band), band, 0.0)


def closed_form_band(
    target_change_var: np.ndarray,
    per_trade_cost_frac: float | np.ndarray,
    gamma_risk: float = 1.0,
) -> np.ndarray:
    """Martin (2012) proportional-cost cube-root no-trade half-width.

    ``((3/2) * cost * target_change_var / gamma_risk) ** (1/3)`` elementwise —
    the proportional-cost tracking asymptotic: cost enters cube-root
    (shallower than the sqrt heuristic above) and the driver is the variance
    of day-over-day *target-weight changes*, not OU speed. ``gamma_risk`` is
    pre-registered at 1.0 and never fitted (docs/r2_design.md §1).
    Non-finite or non-positive inputs disable the band for that name (0.0) —
    the same finite ``>= 0`` output convention as ``cost_aware_band``.
    """
    var = np.asarray(target_change_var, dtype=float)
    cost = np.asarray(per_trade_cost_frac, dtype=float)
    var, cost = np.broadcast_arrays(var, cost)
    gamma_ok = bool(np.isfinite(gamma_risk)) and gamma_risk > 0.0
    valid = gamma_ok & np.isfinite(var) & (var > 0.0) & np.isfinite(cost) & (cost > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        band = (1.5 * cost * var / gamma_risk) ** (1.0 / 3.0)
    return np.where(valid & np.isfinite(band), band, 0.0)


def strength_multiplier(sscores: np.ndarray, entry_band: float, cap: float = 2.0) -> np.ndarray:
    """Conviction multiplier from absolute s-score distance past entry."""
    if entry_band <= 0:
        raise ValueError(f"entry_band must be > 0, got {entry_band}")
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")
    mult = (np.abs(np.asarray(sscores, dtype=float)) - entry_band) / entry_band
    return np.where(np.isfinite(mult), np.clip(mult, 0.0, cap), 0.0)


def build_residual_book_row(
    states: np.ndarray,
    beta_day: np.ndarray,
    eigenportfolios: np.ndarray,
    position_unit: float,
    size_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Net one residual-stat-arb bar into per-symbol target weights."""
    scale = 1.0 if size_scale is None else np.asarray(size_scale, dtype=float)
    stock_legs = position_unit * scale * np.asarray(states, dtype=float)
    factor_exposure = np.asarray(beta_day, dtype=float) @ stock_legs
    hedge_legs = -factor_exposure @ np.asarray(eigenportfolios, dtype=float)
    return stock_legs + hedge_legs


def construct_directional_targets(
    scores: pd.DataFrame,
    *,
    position_size: float,
    max_gross: float,
    max_symbol_abs_weight: float,
    no_trade_band: float | pd.Series = 0.0,
) -> pd.DataFrame:
    """Convert signed per-symbol convictions to unhedged target weights.

    ``scores`` are dimensionless convictions, typically in ``[-1, 1]``. The
    mapping is linear: ``target = position_size * score`` followed only by
    down-only gross/symbol caps. Missing scores stay missing so callers can use
    all-NaN rows as explicit fold-boundary no-op markers.
    """
    if position_size <= 0:
        raise ValueError(f"position_size must be > 0, got {position_size}")
    decision_mask = scores.notna()
    raw = (
        scores.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=-1.0, upper=1.0)
        .fillna(0.0)
        * position_size
    )
    targets = cap_book(raw, max_gross=max_gross, max_symbol_abs_weight=max_symbol_abs_weight)
    if isinstance(no_trade_band, pd.Series) or no_trade_band > 0:
        targets = apply_no_trade_band(targets, no_trade_band)
        targets = cap_book(targets, max_gross=max_gross, max_symbol_abs_weight=max_symbol_abs_weight)
    return targets.where(decision_mask)
