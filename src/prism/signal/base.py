"""Signal contract (SPEC.md §7.1) — the pluggable alpha node.

A Signal consumes wide panels (dates × symbols, tz-aware) and emits one
standardized score per eligible name: the expected return over
``horizon_bars`` in horizon-sigma units, ``E[r_h] / (sigma_daily * sqrt(h))``
(I-3, with the sqrt(h) unit scaling explicit). Scores are *not* positions:
no clipping, no conviction shaping, no cash allocation — sizing folds into
the target weight exactly once, downstream in construction (I-4).

Contract obligations every implementation upholds:

* **Causal (N1).** ``score`` sees a panel whose last row is the decision bar
  *t* and may use nothing after it. Fitted state comes from ``fit`` on a
  train panel only; no transform is fit across the train/score boundary (I-2).
* **NaN = no opinion.** A name the signal cannot score (insufficient history,
  ineligible, degenerate estimate) gets NaN — never a fabricated zero. The
  returned Series is indexed by every panel column, so downstream can tell
  "no opinion" from "absent".
* **Fail loud (N7).** Configuration errors, unconstructable members, and
  unexpected fit/score failures raise. Silent degradation to an empty Series
  or all-zeros is a defect, not a fallback.
* **Scale-invariant.** Scores are unchanged (to float tolerance) under a
  global price-level rescale of the input panel — property-tested per node.
* **Horizon-tagged.** ``horizon_bars`` states the forward horizon the score
  is about; blending across horizons is the consumer's problem, not hidden
  inside the node.
* **JAX/torch-free (N8).** ``prism.signal`` is a new production module: its
  import closure must stay research-free (enforced by
  ``tests/test_import_hygiene.py``).

Portfolio knowledge is explicitly out of contract: a node never sees other
names' weights, never nets, never caps (SPEC §7.1 "never").
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Signal(ABC):
    """Abstract cross-sectional signal node.

    ``fit(close, volume)`` fits on the training panel and returns ``self``;
    ``score(close, volume)`` scores the *last row* of the panel it is given
    and returns one standardized score per column (NaN = no opinion).
    """

    @property
    @abstractmethod
    def horizon_bars(self) -> int:
        """Forward horizon (in bars) the emitted scores are about."""

    @property
    @abstractmethod
    def required_history(self) -> int:
        """Minimum trailing panel rows ``score`` needs to score the last bar."""

    @abstractmethod
    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "Signal":
        """Fit node state on the training panel only (I-2). Returns self."""

    @abstractmethod
    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        """Standardized scores for the panel's last row.

        The panel's last row is the decision bar *t*; only rows ≤ *t* may be
        consumed (N1). Returns a float Series indexed by ``close.columns``:
        ``E[r_h] / (sigma_daily * sqrt(horizon_bars))`` per name, NaN where
        the node has no opinion.
        """
