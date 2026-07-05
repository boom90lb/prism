"""Hard participation gate (SPEC.md §3 law 5, §7.4).

The sqrt-ADV term in ``target_weights`` *prices* impact; this module *bounds* it.
A production book must never send a name-day order that is a large fraction of
that name's average daily dollar volume, regardless of what the signal wants —
impact is concave but not free, and a single illiquid name can dominate realized
cost. The gate caps each name's per-bar *trade* (not its target level) so implied
participation stays under a cap.

Fail-safe by construction (N7): a name with unknown or non-positive ADV is not
tradeable this bar, so its weight is held at the prior level rather than traded
blindly against missing liquidity data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned(
    prev_weights: pd.Series,
    target_weights: pd.Series,
    dollar_volume: pd.Series,
) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    """Union-align the three legs of a rebalance decision.

    Missing prior weight -> flat (0.0); missing/NaN target -> hold prior (no new
    decision for that name); ADV is passed through as-is (NaN = unknown, judged
    by the caller). This is the one place the alignment contract lives — both
    the capping function and its independent monitor consume it.
    """
    index = prev_weights.index.union(target_weights.index).union(dollar_volume.index)
    prev = pd.to_numeric(prev_weights.reindex(index), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    target = pd.to_numeric(target_weights.reindex(index), errors="coerce").to_numpy(dtype=float)
    adv = pd.to_numeric(dollar_volume.reindex(index), errors="coerce").to_numpy(dtype=float)
    target = np.where(np.isfinite(target), target, prev)
    return index, prev, target, adv


def participation_capped_targets(
    prev_weights: pd.Series,
    target_weights: pd.Series,
    dollar_volume: pd.Series,
    aum: float,
    max_participation: float,
    *,
    adv_floor: float = 0.0,
) -> pd.Series:
    """Cap one cross-section's trades to ``max_participation`` of each name's ADV.

    For each name: the desired trade notional is ``|target - prev| * aum``; the
    allowed notional is ``max_participation * ADV`` (ADV = daily dollar volume,
    floored at ``adv_floor``). The trade is shrunk toward the prior weight so the
    filled notional never exceeds the allowed notional; the direction is
    preserved. A name with NaN / non-positive ADV holds its prior weight (no
    trade permitted on missing liquidity).

    ``adv_floor`` defaults to 0.0 (no floor) — deliberately *not* the pricer's
    ``ExecutionConfig.adv_floor_dollars``: the impact model floors tiny ADV **up**
    to stabilize pricing, but flooring ADV up in a *bound* would permit more
    trade in exactly the illiquid names this gate exists to protect. Pass a
    floor explicitly if pricer-consistent participation is wanted.

    All three Series are aligned on the union of their indices (missing prior
    weight = 0.0 flat; missing target = hold prior). Returns capped target
    weights indexed like the union.
    """
    if aum <= 0:
        raise ValueError(f"aum must be > 0, got {aum}")
    if max_participation <= 0:
        raise ValueError(f"max_participation must be > 0, got {max_participation}")
    if adv_floor < 0:
        raise ValueError(f"adv_floor must be >= 0, got {adv_floor}")

    index, prev, target, adv = _aligned(prev_weights, target_weights, dollar_volume)
    tradeable = np.isfinite(adv) & (adv > 0.0)
    # Untradeable names (unknown/zero ADV) get zero allowed delta, i.e. hold prior.
    allowed_weight_delta = np.where(tradeable, max_participation * np.maximum(adv, adv_floor) / aum, 0.0)
    desired_delta = target - prev
    capped_delta = np.sign(desired_delta) * np.minimum(np.abs(desired_delta), allowed_weight_delta)
    return pd.Series(prev + capped_delta, index=index, name=getattr(target_weights, "name", None))


def max_participation_of(
    prev_weights: pd.Series,
    target_weights: pd.Series,
    dollar_volume: pd.Series,
    aum: float,
) -> float:
    """Largest single-name participation implied by a proposed rebalance.

    A monitoring helper: ``max_i |target_i - prev_i| * aum / ADV_i`` over names
    with positive ADV. Returns 0.0 if nothing trades, NaN if no name has usable
    ADV. Lets the live loop assert the realized rebalance is under the cap before
    submitting, independent of the capping function.
    """
    if aum <= 0:
        raise ValueError(f"aum must be > 0, got {aum}")
    _, prev, target, adv = _aligned(prev_weights, target_weights, dollar_volume)
    tradeable = np.isfinite(adv) & (adv > 0.0)
    if not tradeable.any():
        return float("nan")
    participation = np.abs(target - prev) * aum / np.where(tradeable, adv, np.nan)
    finite = participation[np.isfinite(participation)]
    return float(finite.max()) if finite.size else 0.0
