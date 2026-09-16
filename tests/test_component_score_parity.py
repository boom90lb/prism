"""Score parity for the momentum and trend components (Parity, §3.3).

Declaring kinds and units must not move a score. The live momentum book and the
recorded trend driver consume these two nodes, so a changed value here is a
changed book, not a refactor. The goldens below are the exact IEEE-754 doubles
the nodes emitted on a fixed, closed-form panel before the declarations landed,
stored as ``float.hex()`` so equality is bit-for-bit rather than approximate:
any change to the score — including one too small to see in decimal — fails.

The panel is deterministic and generator-free on purpose (no RNG, no fixture
file): the expected values are reproducible from this module alone.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from prism.residual.factors import ResidualStatArbConfig
from prism.signal import MomentumSignalNode, TrendSignalNode

N_DAYS, N_ASSETS = 340, 12
LOOKBACK, SKIP = 120, 10


def _golden(text: str) -> tuple[str, ...]:
    """Whitespace-separated ``float.hex()`` literals, one score each."""
    return tuple(text.split())


MOMENTUM_200 = _golden("""
    -0x1.aa360c15fb360p-5 -0x1.9aaa319e98f80p-6 0x1.c1c69b5bed400p-8
    0x1.efc780ae52c40p-6 0x1.1f239d612aac0p-5 0x1.5002e68fab400p-6
    -0x1.1684f4e55f880p-8 -0x1.a3c0d12c96740p-6 -0x1.08c6a88cebdc0p-5
    -0x1.4450cd75f1300p-6 0x1.115b77f1a4700p-7 0x1.5241c1fb8ff80p-5
    """)
MOMENTUM_270 = _golden("""
    0x1.0c9ae7a0d0000p-13 0x1.78ca17bf9c7c0p-6 0x1.be93d63c08c40p-6
    0x1.97a62e7a65c00p-7 -0x1.9046f4afe8d80p-7 -0x1.0fe093a5f7c20p-5
    -0x1.42ca646530d50p-5 -0x1.b28ad182b6840p-6 0x1.b5ca997b78000p-10
    0x1.192622aa03d20p-5 0x1.d7b5c1c1c6060p-5 0x1.f791603e63a20p-5
    """)
MOMENTUM_340 = _golden("""
    0x1.3fa697a569740p-6 0x1.2254ae576f000p-8 -0x1.49928e86876a0p-6
    -0x1.4d534dd807d90p-5 -0x1.7c49ab5ba9c50p-5 -0x1.0fef0474f95e0p-5
    -0x1.44c98b0c2b200p-8 0x1.c0bf4c9a394c0p-6 0x1.9a7220e586fa0p-5
    0x1.b5e4faad13b20p-5 0x1.331e353a8bf80p-5 0x1.8b165fd934880p-7
    """)
# S3 below the dollar-volume floor, S7 without lookback history: the eligibility
# mask and the history gate are part of the score and are pinned with it.
MOMENTUM_340_MASKED = _golden("""
    0x1.3fa697a569740p-6 0x1.2254ae576f000p-8 -0x1.49928e86876a0p-6
    nan -0x1.7c49ab5ba9c50p-5 -0x1.0fef0474f95e0p-5
    -0x1.44c98b0c2b200p-8 nan 0x1.9a7220e586fa0p-5
    0x1.b5e4faad13b20p-5 0x1.331e353a8bf80p-5 0x1.8b165fd934880p-7
    """)
TREND_300 = _golden("""
    -0x1.ef1dd16be1280p-5 -0x1.4231ff26e13a0p-5 -0x1.f292c73ce9040p-7
    0x1.066eabc407d00p-8 0x1.98951c9246600p-7 0x1.39acc643d9700p-7
    0x1.84eed8cf8f000p-12 -0x1.d2d9e80880380p-8 -0x1.7fdd655c00f80p-8
    0x1.daa6343f13100p-8 0x1.ec05428de5e00p-6 0x1.c97a864e32d40p-5
    """)
TREND_340 = _golden("""
    -0x1.9506b6d4ab670p-5 -0x1.c09320186d220p-5 -0x1.9d05df80ba170p-5
    -0x1.1daffdfaea830p-5 -0x1.7a866ad716740p-7 0x1.953e084ae2a00p-7
    0x1.db833dd9006c0p-6 0x1.153f2d9017d20p-5 0x1.ccc7de8251580p-6
    0x1.2cdd8cb2dcac0p-6 0x1.a549cc024c900p-7 0x1.23981e0a38f80p-6
    """)
TREND_340_MASKED = _golden("""
    -0x1.9506b6d4ab670p-5 -0x1.c09320186d220p-5 -0x1.9d05df80ba170p-5
    -0x1.1daffdfaea830p-5 -0x1.7a866ad716740p-7 0x1.953e084ae2a00p-7
    0x1.db833dd9006c0p-6 nan 0x1.ccc7de8251580p-6
    0x1.2cdd8cb2dcac0p-6 0x1.a549cc024c900p-7 0x1.23981e0a38f80p-6
    """)


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A closed-form panel: per-name drift plus a per-name phase-shifted cycle."""
    bar = np.arange(N_DAYS)[:, None]
    name = np.arange(N_ASSETS)[None, :]
    returns = 0.0004 * (name - 5.5) / 12.0 + 0.01 * np.sin(0.11 * bar + 0.7 * name)
    index = pd.date_range("2020-01-02", periods=N_DAYS, freq="B", tz="America/New_York")
    columns = [f"S{i}" for i in range(N_ASSETS)]
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=index, columns=columns)
    volume = pd.DataFrame(2_000_000.0, index=index, columns=columns)
    return close, volume


def _config() -> ResidualStatArbConfig:
    return ResidualStatArbConfig(
        corr_window=60,
        regr_window=20,
        n_factors=2,
        min_price=5.0,
        min_median_dollar_volume=1_000.0,
        dollar_volume_window=5,
    )


def _assert_bit_identical(scores: pd.Series, golden: tuple[str, ...]) -> None:
    assert len(scores) == len(golden)
    emitted = tuple("nan" if math.isnan(v) else float(v).hex() for v in scores.to_numpy(dtype=float))
    assert emitted == golden


def test_momentum_scores_are_bit_identical_to_the_recorded_book() -> None:
    close, volume = _panels()
    node = MomentumSignalNode(_config(), lookback_bars=LOOKBACK, skip_bars=SKIP, horizon_bars=21)
    for cut, golden in ((200, MOMENTUM_200), (270, MOMENTUM_270), (340, MOMENTUM_340)):
        _assert_bit_identical(node.score(close.iloc[:cut], volume.iloc[:cut]), golden)


def test_momentum_eligibility_and_history_masks_are_bit_identical() -> None:
    close, volume = _panels()
    volume = volume.copy()
    volume["S3"] = 1.0  # dollar volume below the floor
    close = close.copy()
    close.iloc[:230, close.columns.get_loc("S7")] = np.nan  # no lookback history
    node = MomentumSignalNode(_config(), lookback_bars=LOOKBACK, skip_bars=SKIP, horizon_bars=21)
    _assert_bit_identical(node.score(close, volume), MOMENTUM_340_MASKED)


def test_trend_scores_are_bit_identical_to_the_recorded_driver() -> None:
    close, _ = _panels()
    node = TrendSignalNode()  # the pinned 252 / 21 window
    _assert_bit_identical(node.score(close.iloc[:300]), TREND_300)
    _assert_bit_identical(node.score(close), TREND_340)


def test_trend_history_gate_is_bit_identical() -> None:
    close, _ = _panels()
    close = close.copy()
    close.iloc[:230, close.columns.get_loc("S7")] = np.nan
    _assert_bit_identical(TrendSignalNode().score(close), TREND_340_MASKED)
