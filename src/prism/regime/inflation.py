"""Inflation-expectations regime block (SPEC.md §7.5 follow-ons, R4).

The direct observables of a "2% rhetoric, higher realized" financial-repression
regime, which the nominal-only curve block cannot see:

* **real yield** — FRED ``DFII10`` (10Y TIPS constant maturity). Suppressed or
  negative real rates are the repression signature.
* **breakeven** — FRED ``T10YIE`` (10Y breakeven inflation, = DGS10 − DFII10 by
  construction). Its divergence from the rhetorical target is the market's
  disagreement with the stated regime.

Both are **conditioning inputs only, IC-gated (I-8)** like every regime
feature; the dollar-neutral residual core is approximately inflation-neutral
by construction, so this block enters through regime inputs and the §10
hurdle, never through the alpha. Pure and causal; inputs are percent per
annum as FRED publishes them.
"""

from __future__ import annotations

import pandas as pd

#: The rhetorical policy target the breakeven divergence is measured against.
DEFAULT_INFLATION_TARGET_PCT = 2.0


def breakeven_divergence(
    breakeven: pd.Series, target_pct: float = DEFAULT_INFLATION_TARGET_PCT
) -> pd.Series:
    """Breakeven inflation minus the rhetorical target, in percentage points.

    Positive values mean the market prices inflation above the stated target —
    the breakeven-vs-target divergence of the repression playbook. Pure,
    causal, unit-preserving (percent per annum in, percentage points out).
    """
    return (
        pd.to_numeric(breakeven, errors="coerce") - float(target_pct)
    ).rename("breakeven_divergence")


def inflation_state(
    real_yield: pd.Series,
    breakeven: pd.Series,
    target_pct: float = DEFAULT_INFLATION_TARGET_PCT,
) -> pd.DataFrame:
    """Align the two observables on the union calendar, causally.

    Series are forward-filled *within* each series only (release-cadence
    mismatch), which is causal — a value known on its release date is carried
    forward, never backward; dates before a series' first observation stay
    NaN (no fabricated history). Columns: ``real_yield``, ``breakeven``,
    ``breakeven_divergence``.
    """
    frame = pd.DataFrame(
        {
            "real_yield": pd.to_numeric(real_yield, errors="coerce"),
            "breakeven": pd.to_numeric(breakeven, errors="coerce"),
        }
    ).sort_index()
    frame = frame.ffill()  # carry each last-known release forward (causal)
    frame["breakeven_divergence"] = frame["breakeven"] - float(target_pct)
    return frame
