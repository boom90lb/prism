"""The live monitor (prism.live.monitor): consecutive-session returns, the
method switch, the shadow return and the tracking read (docs/live_monitor_design.md).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from prism.io import sessions
from prism.live.loop import append_equity_ledger, append_shadow_ledger
from prism.live.monitor import (
    TRACKING_DIFFERENCE_BOUND,
    TRACKING_DIFFERENCE_TOLERANCE,
    consecutive_session_returns,
    paper_monitor_read,
    shadow_session_return,
    shadow_tracking_read,
)


def _session_bars(first: str, n: int) -> list[str]:
    start = dt.date.fromisoformat(first)
    end = start + dt.timedelta(days=int(n * 1.8) + 10)
    return [d.isoformat() for d in sessions.sessions_between(start, end)[:n]]


def _write_equity(path, bars, navs):
    for bar, nav in zip(bars, navs):
        append_equity_ledger(path, bar, float(nav), float(nav))


# ---------------------------------------------------------------------------
# Consecutive-session returns
# ---------------------------------------------------------------------------


def test_gap_between_ledger_rows_is_reported_not_averaged(tmp_path):
    bars = ["2026-08-12", "2026-08-13", "2026-08-14", "2026-09-25", "2026-09-28"]  # the real outage shape
    navs = [100.0, 101.0, 102.0, 90.0, 91.0]
    p = tmp_path / "equity.jsonl"
    _write_equity(p, bars, navs)
    read = paper_monitor_read(p)
    assert read["n"] == 3 and read["n_gaps"] == 1
    assert read["skipped_pairs"] == [{"prev_bar": "2026-08-14", "bar": "2026-09-25", "sessions_between": 28}]
    frame = pd.DataFrame({"decision_bar": bars, "equity": navs})
    series = consecutive_session_returns(frame)
    assert series["bars"] == ["2026-08-13", "2026-08-14", "2026-09-28"]
    np.testing.assert_allclose(series["returns"], [101.0 / 100.0 - 1.0, 102.0 / 101.0 - 1.0, 91.0 / 90.0 - 1.0])


def test_holiday_and_weekend_pairs_are_gaps_and_a_non_session_row_never_counts():
    # Memorial Day 2026-05-25: a row dated on it pairs with nothing.
    frame = pd.DataFrame(
        {"decision_bar": ["2026-05-22", "2026-05-25", "2026-05-26", "2026-05-27"], "equity": [1.0, 1.1, 1.2, 1.3]}
    )
    series = consecutive_session_returns(frame)
    assert series["bars"] == ["2026-05-27"] and series["n_gaps"] == 2


def test_method_switch_keeps_the_keys_and_the_old_read_is_wider(tmp_path):
    rng = np.random.default_rng(5)
    bars = _session_bars("2026-01-02", 120)
    navs = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.007, size=len(bars)))
    p = tmp_path / "equity.jsonl"
    _write_equity(p, bars, navs)
    new = paper_monitor_read(p)
    old = paper_monitor_read(p, method="normal_mixture")
    keys = {"n", "mean", "ci_lower", "ci_upper", "verdict", "hurdle", "alpha", "n_equity_points", "latest_equity",
            "method", "n_gaps", "bound"}
    assert keys <= set(new) and keys <= set(old)
    assert new["method"] == "empirical_bernstein" and old["method"] == "normal_mixture"
    assert new["n"] == old["n"] == 119 and new["n_gaps"] == 0
    assert (old["ci_upper"] - old["ci_lower"]) > 3.0 * (new["ci_upper"] - new["ci_lower"])
    with pytest.raises(ValueError):
        paper_monitor_read(p, method="bootstrap")


def test_two_points_without_a_consecutive_pair_is_inconclusive_at_n_zero(tmp_path):
    p = tmp_path / "equity.jsonl"
    _write_equity(p, ["2026-08-14", "2026-09-25"], [100.0, 90.0])
    read = paper_monitor_read(p)
    assert read["n"] == 0 and read["n_gaps"] == 1 and read["verdict"] == "inconclusive"
    assert read["n_equity_points"] == 2 and read["method"] == "empirical_bernstein"


# ---------------------------------------------------------------------------
# Shadow return
# ---------------------------------------------------------------------------


def _close(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York") for d in rows])
    return pd.DataFrame(list(rows.values()), index=idx)


def test_shadow_return_is_the_weighted_close_to_close_return():
    close = _close(
        {
            "2026-06-10": {"AAA": 100.0, "BBB": 50.0, "CCC": 20.0},
            "2026-06-11": {"AAA": 102.0, "BBB": 49.0, "CCC": 21.0},
        }
    )
    out = shadow_session_return(close, {"AAA": 0.5, "BBB": -0.5}, "2026-06-11", "2026-06-10")
    assert out["shadow_return"] == pytest.approx(0.5 * 0.02 - 0.5 * (-0.02))
    assert out["n_names_used"] == 2 and out["n_names_missing"] == 0 and out["gross_baseline"] == 1.0


def test_shadow_return_tolerates_a_small_missing_name_and_refuses_a_large_one():
    close = _close({"2026-06-10": {"AAA": 100.0, "BBB": 50.0}, "2026-06-11": {"AAA": 101.0, "BBB": np.nan}})
    out = shadow_session_return(close, {"AAA": 0.95, "BBB": -0.05}, "2026-06-11", "2026-06-10")
    assert out["n_names_missing"] == 1 and out["gross_missing"] == pytest.approx(0.05)
    assert out["shadow_return"] == pytest.approx(0.95 * 0.01)
    with pytest.raises(ValueError, match="unpriced"):
        shadow_session_return(close, {"AAA": 0.5, "BBB": -0.5}, "2026-06-11", "2026-06-10")
    with pytest.raises(KeyError):
        shadow_session_return(close, {"AAA": 0.5}, "2026-06-12", "2026-06-11")
    with pytest.raises(ValueError, match="no book"):
        shadow_session_return(close, {"AAA": 0.0}, "2026-06-11", "2026-06-10")


# ---------------------------------------------------------------------------
# Tracking read
# ---------------------------------------------------------------------------


def _ledgers(tmp_path, n, *, diff=0.0, noise=0.0, shadow_sd=0.007, corr_break=False, seed=1):
    """Paper NAV and shadow ledgers over n consecutive sessions; the first session is a refresh,
    so session two is a transition and the rest are steady."""
    rng = np.random.default_rng(seed)
    bars = _session_bars("2025-01-02", n + 1)
    shadow = rng.normal(0.0, shadow_sd, size=n)
    paper = (rng.normal(0.0, shadow_sd, size=n) if corr_break else shadow) + diff + rng.normal(0.0, noise, size=n)
    nav = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + paper)])
    eq, sh = tmp_path / "equity.jsonl", tmp_path / "shadow.jsonl"
    _write_equity(eq, bars, nav)
    for i in range(1, n + 1):
        append_shadow_ledger(
            sh,
            bars[i],
            {"prev_bar": bars[i - 1], "shadow_return": float(shadow[i - 1]), "paper_return": float(paper[i - 1]),
             "transition": i == 1, "refresh_bar": bars[0], "n_names_used": 10, "n_names_missing": 0},
        )
    return eq, sh


def test_tracking_consistent_when_paper_reproduces_the_shadow(tmp_path):
    eq, sh = _ledgers(tmp_path, 300, noise=0.00005)
    read = shadow_tracking_read(eq, sh)
    assert read["n"] == 299 and read["n_transition"] == 1 and read["n_unmatched"] == 0
    assert read["verdict"] == "consistent"
    assert -TRACKING_DIFFERENCE_TOLERANCE <= read["ci_lower"] <= read["ci_upper"] <= TRACKING_DIFFERENCE_TOLERANCE
    assert read["correlation"] > 0.99 and read["slope"] == pytest.approx(1.0, abs=0.05)
    assert list(read["wedge_by_refresh"]) == ["2025-01-02"]


def test_tracking_contradicted_by_a_persistent_shortfall(tmp_path):
    eq, sh = _ledgers(tmp_path, 100, diff=-0.001, noise=0.0001)  # -10 bp a day against a 2 bp tolerance
    read = shadow_tracking_read(eq, sh)
    assert read["verdict"] == "contradicted" and read["ci_upper"] < -TRACKING_DIFFERENCE_TOLERANCE
    assert read["correlation"] > 0.99  # the shortfall, not the correlation, fired


def test_tracking_contradicted_by_a_broken_correlation(tmp_path):
    eq, sh = _ledgers(tmp_path, 60, corr_break=True)
    read = shadow_tracking_read(eq, sh)
    assert read["n"] == 59 and read["correlation"] < 0.9
    assert read["verdict"] == "contradicted"


def test_tracking_inconclusive_below_the_sample_floor_and_the_width_floor(tmp_path):
    # 20 steady sessions: perfect reproduction, but the sequence's floor is far wider
    # than the tolerance and n is below the 42-session floor.
    eq, sh = _ledgers(tmp_path, 21)
    read = shadow_tracking_read(eq, sh)
    assert read["n"] == 20 and read["verdict"] == "inconclusive"
    assert read["ci_upper"] > TRACKING_DIFFERENCE_TOLERANCE
    # An uncorrelated book at n < 42 is also inconclusive: the correlation is not read yet.
    eq2, sh2 = _ledgers(tmp_path / "b", 30, corr_break=True)
    assert shadow_tracking_read(eq2, sh2)["verdict"] == "inconclusive"


def test_tracking_counts_a_shadow_row_whose_prev_bar_disagrees_as_unmatched(tmp_path):
    eq, sh = _ledgers(tmp_path, 10)
    append_shadow_ledger(sh, "2025-02-03", {"prev_bar": "2025-01-10", "shadow_return": 0.0, "paper_return": 0.0,
                                            "transition": False, "refresh_bar": "2025-01-02"})
    read = shadow_tracking_read(eq, sh)
    assert read["n_unmatched"] == 1 and read["n"] == 9


def test_tracking_power_floor_pins_the_registered_detection_time():
    """docs/live_monitor_design.md §2 states that a constant 5 bp a day shortfall first reads
    `contradicted` at about session 66: with the bet capped at 0.75 the half-width floor is
    2b*log(2/alpha)/(0.75 n), so upper < -tolerance needs n > 0.0197 / 0.0003. Pinned so the
    registration's power statement and the shipped sequence cannot drift apart."""
    from prism.validation.anytime import empirical_bernstein_running_intervals

    path = np.full((1, 120), -0.0005)
    _, upper = empirical_bernstein_running_intervals(path, alpha=0.05, bound=TRACKING_DIFFERENCE_BOUND)
    first = int(np.argmax(upper[0] < -TRACKING_DIFFERENCE_TOLERANCE)) + 1
    assert first == 66


def test_empty_ledgers_read_inconclusive_at_n_zero(tmp_path):
    read = shadow_tracking_read(tmp_path / "equity.jsonl", tmp_path / "shadow.jsonl")
    assert read["n"] == 0 and read["verdict"] == "inconclusive" and read["correlation"] is None
    read2 = paper_monitor_read(tmp_path / "equity.jsonl", shadow_ledger_path=tmp_path / "shadow.jsonl")
    assert read2["tracking"]["verdict"] == "inconclusive"
