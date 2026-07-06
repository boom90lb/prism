"""Per-bucket effective-spread calibration from the fills ledger (I-9).

The estimator half of the R2 cost-measurement instrument: the paper loop's
fills ledger (``prism.live.read_fills_ledger``) records every fill beside
its decision-close reference price, and this module turns those rows into
the per-liquidity-bucket spread table that replaces the pre-registered
conservative-upper schedule (the research WFO's ``SPREAD_BUCKET_SCHEDULE_V1``
carries the same ``(dollar-volume floor, one-way bps)`` shape and documents
itself as "replaced only by fill calibration when fills exist").

What the number measures — stated so it is not over-claimed: with N2
next-open fills, arrival slippage against the decision close is the
*overnight drift plus* the effective execution cost. Trade direction is
uncorrelated with the drift under the null, so the drift washes out of the
signed mean at large n but dominates its variance at small n. The table
therefore carries ``se_bps`` beside every estimate, and
:func:`calibrated_bucket_schedule` refuses to promote a bucket with fewer
than ``min_fills`` fills — an under-sampled bucket keeps its conservative
fallback rather than adopting a noise estimate (the same fail-safe posture
as the schedule it replaces).

Sign convention: positive slippage = the fill was adverse to the trade
(bought above / sold below the reference); negative = price improvement.
The *schedule* floors calibrated values at zero — a cost model must never
book a credit for trading — while the table reports the raw signed numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Dollar-volume floors of the pre-registered V1 bucket schedule
# (docs/r2_design.md §3), descending; the catch-all 0.0 floor is last.
# Duplicated by value from research (production never imports research, §9);
# pass explicit floors to calibrate on a different partition.
DEFAULT_BUCKET_FLOORS: tuple[float, ...] = (500e6, 100e6, 25e6, 0.0)

_REQUIRED_FILL_COLUMNS = ("symbol", "qty", "fill_price", "reference_price")


def arrival_slippage_bps(fills: pd.DataFrame) -> pd.Series:
    """Signed one-way arrival slippage per fill, in bps of the reference.

    ``sign(qty) * (fill_price - reference_price) / reference_price * 1e4``:
    positive = paid (adverse), negative = price improvement. Requires the
    fills-ledger columns; a missing column, non-positive reference price,
    non-finite fill price, or zero qty raises (N7) — a malformed ledger row
    is an accounting defect, not a row to skip.
    """
    missing = [c for c in _REQUIRED_FILL_COLUMNS if c not in fills.columns]
    if missing:
        raise ValueError(f"fills frame is missing ledger columns {missing} (N7)")
    qty = pd.to_numeric(fills["qty"], errors="coerce")
    fill_price = pd.to_numeric(fills["fill_price"], errors="coerce")
    reference = pd.to_numeric(fills["reference_price"], errors="coerce")
    bad = ~np.isfinite(qty) | (qty == 0.0) | ~np.isfinite(fill_price) | ~(reference > 0)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} fills rows have unusable qty/fill_price/reference_price "
            f"(first bad index: {fills.index[bad][0]!r}); repair the ledger, do not skip (N7)"
        )
    return np.sign(qty) * (fill_price - reference) / reference * 1e4


def spread_calibration_table(
    fills: pd.DataFrame,
    median_dollar_volume: pd.Series,
    *,
    floors: tuple[float, ...] = DEFAULT_BUCKET_FLOORS,
) -> pd.DataFrame:
    """Per-bucket arrival-slippage statistics from the fills ledger.

    ``median_dollar_volume`` maps each traded symbol to the liquidity
    statistic the buckets partition on (the same per-name median dollar
    volume the pre-registered schedule keys on); a traded symbol absent
    from it raises (N7) rather than landing silently in the wrong bucket.

    Returns one row per bucket floor (descending, including empty buckets):
    ``floor``, ``n_fills``, ``notional`` (at reference prices),
    ``mean_bps`` (notional-weighted — the economic toll per traded dollar),
    ``median_bps``, ``se_bps`` (unweighted, of the mean), and
    ``improve_frac`` (fraction of fills with price improvement).
    """
    if not floors or sorted(floors, reverse=True) != list(floors) or floors[-1] != 0.0:
        raise ValueError(f"floors must be descending and end at the 0.0 catch-all, got {floors}")
    slippage = arrival_slippage_bps(fills)

    symbols = fills["symbol"].astype(str)
    mdv = pd.to_numeric(median_dollar_volume, errors="coerce")
    unmapped = sorted(set(symbols) - set(mdv[np.isfinite(mdv)].index))
    if unmapped:
        raise ValueError(
            f"no finite median_dollar_volume for traded symbols {unmapped}; "
            "supply the liquidity statistic for every traded name (N7)"
        )
    fill_mdv = symbols.map(mdv).to_numpy(dtype=float)
    notional = (
        pd.to_numeric(fills["qty"], errors="coerce").abs()
        * pd.to_numeric(fills["reference_price"], errors="coerce")
    ).to_numpy(dtype=float)
    slip = slippage.to_numpy(dtype=float)

    rows = []
    ceiling = np.inf
    for floor in floors:
        in_bucket = (fill_mdv >= floor) & (fill_mdv < ceiling)
        ceiling = floor
        n = int(in_bucket.sum())
        bucket_slip = slip[in_bucket]
        bucket_notional = notional[in_bucket]
        rows.append(
            {
                "floor": float(floor),
                "n_fills": n,
                "notional": float(bucket_notional.sum()),
                "mean_bps": (
                    float(np.average(bucket_slip, weights=bucket_notional)) if n else np.nan
                ),
                "median_bps": float(np.median(bucket_slip)) if n else np.nan,
                "se_bps": (float(np.std(bucket_slip, ddof=1) / np.sqrt(n)) if n > 1 else np.nan),
                "improve_frac": float((bucket_slip < 0).mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibrated_bucket_schedule(
    table: pd.DataFrame,
    fallback_schedule: tuple[tuple[float, float], ...],
    *,
    min_fills: int = 30,
) -> tuple[tuple[float, float], ...]:
    """Merge a calibration table onto a conservative fallback schedule.

    A bucket with at least ``min_fills`` fills adopts its measured
    notional-weighted mean (floored at zero — trading never books a
    credit); an under-sampled bucket keeps the fallback's bps. The two
    inputs must partition liquidity on identical floors, or the merge
    would silently re-bucket (raises). The result is the
    ``(floor, one-way bps)`` schedule shape the cost model consumes.
    """
    if min_fills < 1:
        raise ValueError(f"min_fills must be >= 1, got {min_fills}")
    table_floors = [float(f) for f in table["floor"]]
    fallback_floors = [float(f) for f, _ in fallback_schedule]
    if table_floors != fallback_floors:
        raise ValueError(
            f"calibration floors {table_floors} != fallback floors {fallback_floors}; "
            "the schedules must partition liquidity identically"
        )
    schedule = []
    for (floor, fallback_bps), (_, row) in zip(fallback_schedule, table.iterrows()):
        if int(row["n_fills"]) >= min_fills and np.isfinite(row["mean_bps"]):
            schedule.append((floor, max(float(row["mean_bps"]), 0.0)))
        else:
            schedule.append((floor, float(fallback_bps)))
    return tuple(schedule)
