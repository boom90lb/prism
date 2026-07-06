"""Incremental local bar store (SPEC.md §7.0).

One canonical parquet per (symbol, interval) — ``{symbol}_{interval}_bars``
— instead of the legacy range-keyed cache's one-file-per-request-range. The
daily loop then fetches only the missing tail since the last stored bar
(delta fetch) and appends, so a 500-name universe costs ~500 requests/day of
the 800/day budget instead of 500 full-history refetches per widened range.

The store is pure local persistence: it never touches the network. The
delta-fetch orchestration (what range to request, when a full refetch is
required) lives with the loader; the store contributes the one decision that
must be local — **split detection**. A split-adjusted vendor series rewrites
*all* history when a split lands, so an appended tail that disagrees with
stored bars on their overlap means the whole series must be refetched, not
patched (`SPEC §5: split-driven back-rewrites are handled by the incremental
store, not masked by full refetch`).

The legacy range-keyed cache in ``prism.data_loader`` remains the default
read path until the R4 rework folds the loader into ``prism.io`` — this
module lands the mechanics ahead of that move, opt-in (nothing in the
current pipeline changes behavior).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical bar timezone. Duplicated from prism.data_loader deliberately:
# importing it from there would put the (partially initialized) prism.io
# package on data_loader's own import path; the constant moves here for good
# when the loader folds into io/ (R4).
BAR_TZ = "America/New_York"

# Overlap bars re-requested with every delta fetch and compared against the
# stored series: agreement validates a plain append, disagreement signals a
# split-driven back-rewrite (full refetch). 5 bars ≈ one trading week.
DEFAULT_OVERLAP_BARS = 5

# Relative tolerance for "the vendor rewrote history". Split adjustments
# move prices by integer ratios (2x, 4x, 1.5x); float jitter in a stable
# series is ~1e-12. 1e-6 separates the two by orders of magnitude.
REWRITE_RTOL = 1e-6


class SplitRewriteDetected(Exception):
    """Overlap bars disagree with the stored series: history was rewritten.

    The caller must refetch the full history and ``replace`` the store —
    patching the tail onto rewritten history would splice two different
    adjustment bases into one series.
    """


def _validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bar frame must have a DatetimeIndex")
    if bars.index.tz is None:
        raise ValueError(f"bar frame index must be tz-aware ({BAR_TZ})")
    if str(bars.index.tz) != BAR_TZ:
        bars = bars.tz_convert(BAR_TZ)
    if bars.index.has_duplicates:
        raise ValueError("bar frame has duplicate timestamps")
    if not bars.index.is_monotonic_increasing:
        bars = bars.sort_index()
    return bars


class IncrementalBarStore:
    """Per-(symbol, interval) canonical bar series on local parquet."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True, parents=True)

    def _path(self, symbol: str, interval: str) -> Path:
        safe = symbol.replace("/", "_")
        return self.directory / f"{safe}_{interval}_bars.parquet"

    def read(self, symbol: str, interval: str) -> pd.DataFrame:
        """The stored series (possibly empty), tz-aware and sorted."""
        path = self._path(symbol, interval)
        if not path.exists():
            return pd.DataFrame()
        return _validate_bars(pd.read_parquet(path))

    def last_timestamp(self, symbol: str, interval: str) -> pd.Timestamp | None:
        """Timestamp of the newest stored bar, or None when empty."""
        stored = self.read(symbol, interval)
        return None if stored.empty else stored.index[-1]

    def replace(self, symbol: str, interval: str, bars: pd.DataFrame) -> pd.DataFrame:
        """Overwrite the stored series wholesale (seed or post-rewrite refetch)."""
        bars = _validate_bars(bars)
        bars.to_parquet(self._path(symbol, interval))
        return bars

    def append_tail(
        self,
        symbol: str,
        interval: str,
        tail: pd.DataFrame,
        *,
        rewrite_rtol: float = REWRITE_RTOL,
    ) -> pd.DataFrame:
        """Append a delta-fetched tail; overlap bars must agree with the store.

        ``tail`` should start a few bars *before* the last stored bar (the
        delta fetch re-requests ``DEFAULT_OVERLAP_BARS`` of overlap). Where
        the tail overlaps stored history, closes are compared: agreement
        (within ``rewrite_rtol`` relative) validates the append and the
        vendor's fresher rows win; disagreement raises
        :class:`SplitRewriteDetected` and writes nothing (N7 — never splice
        two adjustment bases).

        Returns the merged stored series.
        """
        tail = _validate_bars(tail)
        stored = self.read(symbol, interval)
        if stored.empty:
            return self.replace(symbol, interval, tail)
        if tail.empty:
            return stored

        overlap_idx = stored.index.intersection(tail.index)
        if len(overlap_idx) and "close" in stored.columns and "close" in tail.columns:
            old = pd.to_numeric(stored.loc[overlap_idx, "close"], errors="coerce")
            new = pd.to_numeric(tail.loc[overlap_idx, "close"], errors="coerce")
            both = old.notna() & new.notna()
            if both.any():
                rel = ((old[both] - new[both]).abs() / old[both].abs().clip(lower=1e-12)).max()
                if rel > rewrite_rtol:
                    raise SplitRewriteDetected(
                        f"{symbol} {interval}: overlap closes diverge by {rel:.2e} "
                        f"(> {rewrite_rtol:.0e}) — history was back-rewritten "
                        "(split/re-adjustment); refetch the full series and replace()"
                    )
        elif not len(overlap_idx):
            gap_start, gap_end = stored.index[-1], tail.index[0]
            if gap_start < gap_end:
                logger.warning(
                    "%s %s: delta tail starts at %s with no overlap against stored "
                    "end %s — append is unvalidated (a back-rewrite would go "
                    "undetected); request the tail with overlap bars",
                    symbol,
                    interval,
                    gap_end,
                    gap_start,
                )

        merged = pd.concat([stored.loc[~stored.index.isin(tail.index)], tail]).sort_index()
        merged = _validate_bars(merged)
        merged.to_parquet(self._path(symbol, interval))
        return merged
