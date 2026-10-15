"""Synthetic-data tests for the earnings reads beside the replication criteria (research tier).

They pin the next-open alignment of the forward returns (N2), the per-session rank IC and
its minimum cross-section, the Newey-West mean, and the discovery set's kill rule.
"""

import numpy as np
import pandas as pd
import pytest


from research.scripts import earnings_read as rd  # noqa: E402


def test_forward_returns_start_at_the_next_open():
    index = pd.bdate_range("2024-01-02", periods=5)
    opens = pd.DataFrame({"A": [1.0, 2.0, 4.0, 8.0, 16.0]}, index=index)
    forward = rd.forward_open_returns(opens, 1)
    assert forward.iloc[0, 0] == pytest.approx(1.0)  # open 2 -> open 4: nothing at or before session 0's open
    assert np.isnan(forward.iloc[3, 0]) and np.isnan(forward.iloc[4, 0])


def test_rank_ic_is_one_for_a_perfect_ranking_and_needs_enough_names():
    index = pd.bdate_range("2024-01-02", periods=2)
    names = [f"S{i}" for i in range(40)]
    scores = pd.DataFrame([np.arange(40.0), np.arange(40.0)], index=index, columns=names)
    forward = pd.DataFrame([np.arange(40.0) ** 2, -np.arange(40.0)], index=index, columns=names)
    ic = rd.rank_ic(scores, forward)
    assert ic.iloc[0] == pytest.approx(1.0) and ic.iloc[1] == pytest.approx(-1.0)
    assert rd.rank_ic(scores.iloc[:, :10], forward.iloc[:, :10]).isna().all()


def test_the_newey_west_mean_reduces_to_the_iid_error_without_lags():
    values = pd.Series(np.random.default_rng(2).normal(0.02, 0.1, 5000))
    got = rd.newey_west_mean(values, lags=0)
    assert got["mean"] == pytest.approx(values.mean())
    assert got["se"] == pytest.approx(values.std(ddof=0) / np.sqrt(len(values)))


def test_an_arm_is_killed_only_when_its_interval_lies_below_zero():
    rng = np.random.default_rng(4)
    boot = dict(rd.KILL_BOOTSTRAP, n_resamples=200)
    losing = pd.Series(rng.normal(-0.004, 0.01, 1200))
    flat = pd.Series(rng.normal(0.0, 0.01, 1200))
    assert rd.kill_read(losing, boot)["killed"] is True
    assert rd.kill_read(flat, boot)["killed"] is False
