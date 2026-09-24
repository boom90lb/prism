"""Offline tests for the fundamental characteristic node (prism.signal.fundamental_node)."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from prism.signal.base import check_declarations
from prism.signal.fundamental_node import FundamentalCharacteristicNode


@pytest.fixture
def panel(tmp_path):
    days = pd.to_datetime(["2024-03-01", "2024-03-04"])
    frame = pd.DataFrame({"date": np.repeat(days, 2), "sid": ["A", "B"] * 2, "at_gr1": [0.1, -0.2, 0.3, np.nan],
                          "be_me": [0.5, 1.5, 0.6, 1.4]})
    path = tmp_path / "panel.parquet"
    frame.to_parquet(path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _close(day, columns=("A", "B", "C")):
    index = pd.DatetimeIndex([pd.Timestamp(day)]).tz_localize("America/New_York")
    return pd.DataFrame([[10.0] * len(columns)], index=index, columns=list(columns))


def test_scores_are_signed_to_the_published_direction(panel):
    path, digest = panel
    growth = FundamentalCharacteristicNode("at_gr1", str(path), digest)
    scores = growth.score(_close("2024-03-01"))
    assert scores["A"] == pytest.approx(-0.1) and scores["B"] == pytest.approx(0.2)  # low growth ranks high
    assert np.isnan(scores["C"])  # not in the panel: no opinion
    assert np.isnan(growth.score(_close("2024-03-04"))["B"])
    value = FundamentalCharacteristicNode("be_me", str(path), digest)
    assert value.score(_close("2024-03-04"))["B"] == pytest.approx(1.4)
    check_declarations(value, configured_id=None)


def test_refuses_wrong_hash_unknown_characteristic_and_dates_outside_the_panel(panel):
    path, digest = panel
    with pytest.raises(ValueError):
        FundamentalCharacteristicNode("be_me", str(path), "0" * 64)
    with pytest.raises(ValueError):
        FundamentalCharacteristicNode("momentum", str(path), digest)
    node = FundamentalCharacteristicNode("be_me", str(path), digest)
    with pytest.raises(ValueError):
        node.score(_close("2024-03-05"))


def test_scores_ignore_the_price_level(panel):
    path, digest = panel
    node = FundamentalCharacteristicNode("be_me", str(path), digest)
    base = node.score(_close("2024-03-01"))
    rescaled = node.score(_close("2024-03-01") * 7.0)
    pd.testing.assert_series_equal(base, rescaled)
