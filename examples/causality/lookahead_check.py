"""Look-ahead checks on synthetic prices: honest signal functions pass, leaky ones are caught.

Synthetic data; demonstrates the API only; makes no claim about any strategy.

Run it from a checkout, with the package installed (``uv sync``) or with ``PYTHONPATH=src``::

    python examples/causality/lookahead_check.py

The prices are a seeded random walk, so the output repeats exactly. It reads no files, uses
no network and takes about a second. ``docs/causality.md`` explains what each check does and
what it cannot see.

The cases:

* 12-1 momentum written two ways: as a function that returns one score per asset for the
  last bar it is handed, and as vectorized code that returns a whole frame. Both pass.
* A five-day forward return used as a feature, and a z-score that uses the full-sample mean
  and standard deviation. Both are caught.
* A class that computes a forward-looking feature on the first panel it sees and caches it.
  It is caught when ``factory`` gives the check a fresh instance for its reference. Without
  a factory the same leak passes, because the first, shortest panel fills the cache; that is
  why stateful code should always be checked with one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.causality import CausalityReport, check_causality

HEADER = "Synthetic data; demonstrates the API only; makes no claim about any strategy."
SEED = 7
N_BARS = 600
N_ASSETS = 30

#: Whether each case is expected to pass every check.
EXPECTED: dict[str, bool] = {
    "12-1 momentum, one score per asset": True,
    "12-1 momentum, vectorized": True,
    "five-day forward return": False,
    "full-sample z-score": False,
    "cached forward feature, fresh instance for the reference": False,
    "cached forward feature, no factory (blind spot)": True,
}


def synthetic_close() -> pd.DataFrame:
    """Daily closes for ``N_ASSETS`` assets over ``N_BARS`` business days."""
    rng = np.random.default_rng(SEED)
    index = pd.bdate_range("2015-01-01", periods=N_BARS)
    returns = rng.normal(0.0003, 0.015, (N_BARS, N_ASSETS))
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=index,
        columns=[f"asset_{i:02d}" for i in range(N_ASSETS)],
    )


def momentum_last_bar(close: pd.DataFrame) -> pd.Series:
    """12-1 momentum for the last bar handed in: the return from t-252 to t-21."""
    return close.iloc[-22] / close.iloc[-253] - 1.0


def momentum_vectorized(close: pd.DataFrame) -> pd.DataFrame:
    """The same signal for every bar at once."""
    return close.shift(21) / close.shift(252) - 1.0


def forward_return(close: pd.DataFrame) -> pd.DataFrame:
    """The next five days' return: a label, not a feature."""
    return close.shift(-5) / close - 1.0


def full_sample_zscore(close: pd.DataFrame) -> pd.DataFrame:
    """Prices standardized with a mean and standard deviation taken over the whole sample."""
    return (close - close.mean()) / close.std()


class CachedForwardFeature:
    """Computes a forward return on the first panel it is given, then answers from that cache by date."""

    def __init__(self, forward_bars: int = 5) -> None:
        self.forward_bars = forward_bars
        self._cache: pd.DataFrame | None = None

    def score(self, close: pd.DataFrame) -> pd.Series:
        if self._cache is None:
            self._cache = close.shift(-self.forward_bars) / close - 1.0
        stamp = close.index[-1]
        if stamp in self._cache.index and bool(self._cache.loc[stamp].notna().any()):
            return self._cache.loc[stamp]
        return momentum_last_bar(close)


def main() -> dict[str, CausalityReport]:
    close = synthetic_close()
    print(HEADER)
    print(f"{N_ASSETS} assets, {N_BARS} business days, seed {SEED}\n")

    reports = {
        "12-1 momentum, one score per asset": check_causality(momentum_last_bar, close),
        "12-1 momentum, vectorized": check_causality(momentum_vectorized, close),
        "five-day forward return": check_causality(forward_return, close),
        "full-sample z-score": check_causality(full_sample_zscore, close),
        "cached forward feature, fresh instance for the reference": check_causality(
            CachedForwardFeature().score, close, factory=lambda: CachedForwardFeature().score
        ),
        "cached forward feature, no factory (blind spot)": check_causality(CachedForwardFeature().score, close),
    }
    for name, report in reports.items():
        print(f"{'PASS' if report.ok else 'FAIL'}  {name}  (output read as {report.output})")
        for check in report.checks:
            if not check.passed:
                print(f"      {check.name}: {check.detail}")
    print("\nThe full report for the forward return:\n")
    print(reports["five-day forward return"].text())
    return reports


if __name__ == "__main__":
    main()
