"""Offline tests for the announcement-return node (prism.signal.earnings_node)."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from prism.signal.base import check_declarations
from prism.signal.earnings_node import (
    AnnouncementProximityNode,
    EarningsAnnouncementNode,
)


@pytest.fixture
def panel(tmp_path):
    days = pd.to_datetime(["2024-03-01", "2024-03-04"])
    frame = pd.DataFrame({"date": np.repeat(days, 2), "sid": ["A", "B"] * 2, "ear": [0.05, -0.02, 0.05, -0.02],
                          "age_sessions": [60, 3, 61, 4], "days_to_expected": [5.0, 40.0, 2.0, np.nan]})
    path = tmp_path / "panel.parquet"
    frame.to_parquet(path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _close(day, columns=("A", "B", "C")):
    index = pd.DatetimeIndex([pd.Timestamp(day)]).tz_localize("America/New_York")
    return pd.DataFrame([[10.0] * len(columns)], index=index, columns=list(columns))


def test_scores_are_the_announcement_return_while_it_is_live(panel):
    path, digest = panel
    node = EarningsAnnouncementNode(str(path), digest, max_age_sessions=60)
    first = node.score(_close("2024-03-01"))
    assert first["A"] == pytest.approx(0.05) and first["B"] == pytest.approx(-0.02)
    assert np.isnan(first["C"])  # no announcement: no opinion
    later = node.score(_close("2024-03-04"))
    assert np.isnan(later["A"]) and later["B"] == pytest.approx(-0.02)  # A aged past 60 sessions
    check_declarations(node, configured_id=None)


def test_refuses_wrong_hash_negative_age_and_dates_outside_the_panel(panel):
    path, digest = panel
    with pytest.raises(ValueError):
        EarningsAnnouncementNode(str(path), "0" * 64, max_age_sessions=60)
    with pytest.raises(ValueError):
        EarningsAnnouncementNode(str(path), digest, max_age_sessions=-1)
    node = EarningsAnnouncementNode(str(path), digest, max_age_sessions=126)
    with pytest.raises(ValueError):
        node.score(_close("2024-03-05"))


def test_a_date_with_no_live_value_scores_all_nan_rather_than_refusing(panel):
    path, digest = panel
    node = EarningsAnnouncementNode(str(path), digest, max_age_sessions=2)
    assert node.score(_close("2024-03-01")).isna().all()


def test_scores_ignore_the_price_level(panel):
    path, digest = panel
    node = EarningsAnnouncementNode(str(path), digest, max_age_sessions=126)
    pd.testing.assert_series_equal(node.score(_close("2024-03-01")), node.score(_close("2024-03-01") * 7.0))


def test_proximity_ranks_the_imminent_announcer_highest(panel):
    path, digest = panel
    node = AnnouncementProximityNode(str(path), digest)
    first = node.score(_close("2024-03-01"))
    assert first["A"] == pytest.approx(-5.0) and first["B"] == pytest.approx(-40.0) and np.isnan(first["C"])
    assert np.isnan(node.score(_close("2024-03-04"))["B"])
    check_declarations(node, configured_id=None)
    with pytest.raises(ValueError):
        AnnouncementProximityNode(str(path), "0" * 64)
