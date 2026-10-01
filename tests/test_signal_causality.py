"""One causality test over every signal node (SPEC N1; signal contract in
``prism.signal.base``).

A node scores the last row of the panel it is given, so the only way a future
bar can reach a past score is through state the node keeps between calls: a
cached fit, a lazily built feature table, a frozen panel read at the wrong date.
The test therefore scores bar *t* twice, from a fresh node given the panel
truncated at *t* and from a node that has already been fitted on and has scored
the full panel, and requires the two to agree exactly; it also requires the
score to carry at least one opinion, so an all-NaN agreement cannot pass.

Every concrete ``Signal`` subclass under ``src/prism/signal/*_node.py`` is
discovered by import, not listed by hand. A new node fails this test until it
has a construction case here or an explicit skip reason, which is the point: the
per-node suites carry this property for the nodes whose authors remembered it,
and this file carries it for the ones who did not.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

import prism.signal
from prism.residual.factors import ResidualStatArbConfig
from prism.signal.base import Signal

# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

SIGNAL_DIR = Path(prism.signal.__file__).parent


def discover_nodes() -> dict[str, type[Signal]]:
    """``{"<module>.<Class>": class}`` for every concrete Signal in a ``*_node.py``."""
    found: dict[str, type[Signal]] = {}
    for path in sorted(SIGNAL_DIR.glob("*_node.py")):
        module = importlib.import_module(f"prism.signal.{path.stem}")
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, Signal)
                and obj.__module__ == module.__name__
                and not inspect.isabstract(obj)
            ):
                found[f"{path.stem}.{name}"] = obj
    return found


NODES = discover_nodes()

#: Nodes that cannot be exercised in isolation, with the reason the skip states.
SKIPPED: dict[str, str] = {}


# ---------------------------------------------------------------------------
# construction cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    make: Callable[[], Signal]
    close: pd.DataFrame
    volume: pd.DataFrame | None
    cut: int
    min_opinions: int = 1


def _small_config() -> ResidualStatArbConfig:
    return ResidualStatArbConfig(
        corr_window=60,
        regr_window=20,
        n_factors=2,
        min_price=5.0,
        min_median_dollar_volume=1_000.0,
        dollar_volume_window=5,
    )


def _price_panel(n_days: int, n_assets: int, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One common factor plus per-name drift and noise of distinct scale (the
    characteristics suite's panel), so every price-based node has opinions."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-02", periods=n_days, freq="B", tz="America/New_York")
    market = rng.normal(0.0, 0.01, size=(n_days, 1))
    betas = np.linspace(0.6, 1.4, n_assets)[None, :]
    drifts = np.linspace(-0.0008, 0.0008, n_assets)[None, :]
    scales = np.linspace(0.003, 0.012, n_assets)[None, :]
    returns = market * betas + drifts + rng.normal(0.0, 1.0, size=(n_days, n_assets)) * scales
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=index,
        columns=[f"S{i:02d}" for i in range(n_assets)],
    )
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, size=(n_days, n_assets)), index=index, columns=close.columns)
    return close, volume


def _frozen_panel(tmp_path: Path, close: pd.DataFrame) -> tuple[str, str]:
    """A point-in-time event/characteristic panel whose values differ on every
    date, so a node reading the wrong date is caught, with a name (``C``) the
    close panel has but the frozen panel does not."""
    days = pd.DatetimeIndex([stamp.tz_localize(None).normalize() for stamp in close.index])
    rows = []
    for i, day in enumerate(days):
        rows.append(
            {
                "date": day,
                "sid": "A",
                "ear": 0.01 * (i + 1),
                "age_sessions": i,
                "days_to_expected": 5.0 + i,
                "be_me": 0.5 + 0.1 * i,
                "at_gr1": 0.02 * i,
            }
        )
        rows.append(
            {
                "date": day,
                "sid": "B",
                "ear": -0.01 * (i + 1),
                "age_sessions": 2 * i,
                "days_to_expected": 40.0 - i,
                "be_me": 1.5 - 0.1 * i,
                "at_gr1": -0.02 * i,
            }
        )
    frame = pd.DataFrame(rows)
    path = tmp_path / "frozen_panel.parquet"
    frame.to_parquet(path)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _event_close() -> pd.DataFrame:
    index = pd.date_range("2024-03-01", periods=12, freq="B", tz="America/New_York")
    return pd.DataFrame(10.0, index=index, columns=["A", "B", "C"])


def _momentum(tmp_path: Path) -> Case:
    from prism.signal.momentum_node import MomentumSignalNode

    close, volume = _price_panel(360, 12)
    return Case(lambda: MomentumSignalNode(_small_config(), lookback_bars=120, skip_bars=10), close, volume, cut=299)


def _trend(tmp_path: Path) -> Case:
    from prism.signal.trend_node import TrendSignalNode

    close, volume = _price_panel(360, 12)
    return Case(lambda: TrendSignalNode(lookback_bars=120, skip_bars=10), close, volume, cut=299)


def _residual(tmp_path: Path) -> Case:
    from prism.signal.residual_node import ResidualSignalNode

    close, volume = _price_panel(360, 12)
    return Case(lambda: ResidualSignalNode(_small_config(), horizon_bars=5), close, volume, cut=299)


def _residual_momentum(tmp_path: Path) -> Case:
    from prism.signal.residual_momentum_node import ResidualMomentumSignalNode

    close, volume = _price_panel(360, 12)
    return Case(
        lambda: ResidualMomentumSignalNode(_small_config(), lookback_bars=120, skip_bars=10), close, volume, cut=299
    )


def _idiosyncratic_vol(tmp_path: Path) -> Case:
    from prism.signal.idiosyncratic_vol_node import IdiosyncraticVolSignalNode

    close, volume = _price_panel(360, 12)
    return Case(
        lambda: IdiosyncraticVolSignalNode(_small_config(), estimation_bars=120, vol_bars=30), close, volume, cut=299
    )


def _learned_xsection(tmp_path: Path) -> Case:
    from prism.signal.learned_xsection_node import LearnedCrossSectionNode

    close, volume = _price_panel(700, 40, seed=7)
    membership = pd.DataFrame(True, index=close.index, columns=close.columns)

    def make() -> Signal:
        node = LearnedCrossSectionNode(_small_config(), n_cross_sections=12)
        node.bind_membership(membership)
        return node

    # 252 feature bars + 12 formation bars 21 apart + a 21-bar target fit before the cut.
    return Case(make, close, volume, cut=640, min_opinions=10)


def _earnings_announcement(tmp_path: Path) -> Case:
    from prism.signal.earnings_node import EarningsAnnouncementNode

    close = _event_close()
    path, digest = _frozen_panel(tmp_path, close)
    return Case(lambda: EarningsAnnouncementNode(path, digest, max_age_sessions=60), close, None, cut=6)


def _announcement_proximity(tmp_path: Path) -> Case:
    from prism.signal.earnings_node import AnnouncementProximityNode

    close = _event_close()
    path, digest = _frozen_panel(tmp_path, close)
    return Case(lambda: AnnouncementProximityNode(path, digest), close, None, cut=6)


def _fundamental(tmp_path: Path) -> Case:
    from prism.signal.fundamental_node import FundamentalCharacteristicNode

    close = _event_close()
    path, digest = _frozen_panel(tmp_path, close)
    return Case(lambda: FundamentalCharacteristicNode("be_me", path, digest), close, None, cut=6)


CASES: dict[str, Callable[[Path], Case]] = {
    "earnings_node.AnnouncementProximityNode": _announcement_proximity,
    "earnings_node.EarningsAnnouncementNode": _earnings_announcement,
    "fundamental_node.FundamentalCharacteristicNode": _fundamental,
    "idiosyncratic_vol_node.IdiosyncraticVolSignalNode": _idiosyncratic_vol,
    "learned_xsection_node.LearnedCrossSectionNode": _learned_xsection,
    "momentum_node.MomentumSignalNode": _momentum,
    "residual_momentum_node.ResidualMomentumSignalNode": _residual_momentum,
    "residual_node.ResidualSignalNode": _residual,
    "trend_node.TrendSignalNode": _trend,
}


# ---------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------


def test_discovery_finds_the_shipped_nodes() -> None:
    assert {"momentum_node.MomentumSignalNode", "trend_node.TrendSignalNode"} <= set(NODES)
    stale = sorted((set(CASES) | set(SKIPPED)) - set(NODES))
    assert not stale, f"cases or skips name nodes that no longer exist: {stale}"


def _truncate(frame: pd.DataFrame | None, cut: int) -> pd.DataFrame | None:
    return None if frame is None else frame.iloc[: cut + 1]


@pytest.mark.parametrize("name", sorted(NODES))
def test_appending_future_bars_changes_nothing(name: str, tmp_path: Path) -> None:
    if name in SKIPPED:
        pytest.skip(SKIPPED[name])
    if name not in CASES:
        pytest.fail(f"{name} has no causality case: add a construction case to CASES or a reason to SKIPPED")
    case = CASES[name](tmp_path)
    assert isinstance(case.make(), NODES[name])
    close_t, volume_t = _truncate(case.close, case.cut), _truncate(case.volume, case.cut)
    assert len(close_t) < len(case.close), "the cut must leave future bars to append"

    fresh = case.make()
    fresh.fit(close_t, volume_t)
    at_t = fresh.score(close_t, volume_t)

    exposed = case.make()
    exposed.fit(case.close, case.volume)
    later = exposed.score(case.close, case.volume)  # bars after t have now been seen
    after_future = exposed.score(close_t, volume_t)

    pd.testing.assert_series_equal(at_t, after_future)
    assert at_t.index.equals(case.close.columns)
    assert int(at_t.notna().sum()) >= case.min_opinions, "an all-NaN agreement proves nothing"
    assert later.index.equals(case.close.columns)
