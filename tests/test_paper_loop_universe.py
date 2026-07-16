"""The fetch universe must contain the held book; extras are valuation/exit-only.

Pins the 2026-07-15 failure class: a regenerated ``sp500_current.txt``
correctly dropped POOL (index removal 2026-06-22) while the live book still
held it from the prior refresh, and the mark step refused to value the
position (N7). The loop now fetches file ∪ held, masks the extras out of
scoring, and relies on the decile construct's explicit-flat pin to exit them
at the next refresh.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from prism.portfolio.construct import construct_decile_neutral
from prism.residual.factors import ResidualStatArbConfig
from prism.scripts.paper_loop import _with_held_names
from prism.signal.momentum_node import MomentumSignalNode


def test_with_held_names_unions_and_reports_extras() -> None:
    symbols = ["AAA", "BBB"]
    fetch, extras = _with_held_names(symbols, SimpleNamespace(positions={"POOL": -5.0, "AAA": 3.0}))
    assert fetch == ["AAA", "BBB", "POOL"]
    assert extras == ["POOL"]


def test_with_held_names_tolerates_fresh_state() -> None:
    symbols = ["AAA", "BBB"]
    assert _with_held_names(symbols, None) == (symbols, [])
    assert _with_held_names(symbols, SimpleNamespace(positions={})) == (symbols, [])


def test_masked_extra_scores_nan_and_gets_explicit_flat_target() -> None:
    idx = pd.date_range("2024-01-01", periods=300, freq="B", tz="America/New_York")
    names = [f"S{i:02d}" for i in range(12)] + ["POOL"]
    drift = np.linspace(-0.002, 0.002, len(names))
    steps = np.arange(len(idx))[:, None]
    close = pd.DataFrame(100.0 * (1.0 + drift[None, :]) ** steps, index=idx, columns=names)
    volume = pd.DataFrame(1_000_000.0, index=idx, columns=names)
    mask = pd.DataFrame(False, index=idx, columns=names)
    mask.loc[:, names[:-1]] = True  # POOL is valuation-only: masked ineligible

    node = MomentumSignalNode(
        ResidualStatArbConfig(), lookback_bars=252, skip_bars=21, horizon_bars=21, membership_mask=mask
    )
    scores = node.score(close, volume)
    assert np.isnan(scores["POOL"])
    assert scores.drop("POOL").notna().all()

    row = construct_decile_neutral(pd.DataFrame([scores]), decile=0.1).iloc[0]
    # The explicit-flat pin: the masked name gets weight 0.0 (an exit order for
    # a held position), never NaN (which would hold it forever).
    assert row["POOL"] == 0.0
    assert (row != 0.0).any()
