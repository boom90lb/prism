"""Unit tests for the bar-vendor divergence arithmetic (research tier).

Pure helpers run offline; the network seam is exercised through the same
injectable fake-session pattern the AlpacaBarSource tests use, so only the
real HTTP transport stays network-gated.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.research

from prism.live.alpaca_data import DATA_BASE_URL, AlpacaBarSource  # noqa: E402
from research.scripts.bar_vendor_divergence import (  # noqa: E402
    CountingSession,
    align_panels,
    decile_legs,
    endpoint_scores,
    fetch_iex_closes,
    flag_adjustment_mismatch,
    iex_close_panel,
    leg_flip_report,
    month_end_decision_bars,
    nav_mark_report,
    panel_diff_stats,
    parse_universe,
    per_name_diff_stats,
    perturbed_flip_stats,
    score_endpoints,
)

KEY, SECRET = "test-key-id", "test-secret"


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        return self._payload


class FakeSession:
    """Routes (method, path) to a canned response or handler(json, params)."""

    def __init__(self):
        self.routes = {}
        self.calls = []

    def route(self, method, path, response):
        self.routes[(method, path)] = response

    def request(self, method, url, headers=None, timeout=None, json=None, params=None):
        assert url.startswith(DATA_BASE_URL)
        path = url[len(DATA_BASE_URL) :]
        self.calls.append({"method": method, "path": path, "params": params})
        handler = self.routes[(method, path)]
        if callable(handler):
            return handler(json=json, params=params)
        return handler


def _bar(t, c):
    return {"t": t, "o": c, "h": c, "l": c, "c": c, "v": 1000, "n": 1, "vw": c}


def naive_dates(*dates: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(dates)))


# ------------------------------------------------------------ universe / panel


def test_parse_universe_skips_comments_and_blanks() -> None:
    text = "# header\n# more\nAAPL\n\nBRK.B\n  \nZTS\n"
    assert parse_universe(text) == ["AAPL", "BRK.B", "ZTS"]


def test_iex_close_panel_normalizes_midnight_et_and_reports_empties() -> None:
    idx = pd.DatetimeIndex(
        pd.to_datetime(["2026-07-06T04:00:00Z", "2026-07-07T04:00:00Z"], utc=True)
    ).tz_convert("America/New_York")
    frames = {
        "AAA": pd.DataFrame({"close": [10.0, 11.0], "open": [1, 1]}, index=idx),
        "GONE": pd.DataFrame(),
    }
    panel, empty = iex_close_panel(frames)
    assert empty == ["GONE"]
    assert list(panel.columns) == ["AAA"]
    assert panel.index.tz is None
    assert [str(d.date()) for d in panel.index] == ["2026-07-06", "2026-07-07"]
    assert panel["AAA"].tolist() == [10.0, 11.0]


def test_fetch_seam_through_injected_session_counts_requests() -> None:
    fake = FakeSession()
    fake.route(
        "GET",
        "/v2/stocks/bars",
        FakeResponse(
            payload={
                "bars": {
                    "AAA": [_bar("2026-01-05T05:00:00Z", 100.0), _bar("2026-01-06T05:00:00Z", 101.0)],
                    "GONE": [],
                },
                "next_page_token": None,
            }
        ),
    )
    counting = CountingSession(fake)
    source = AlpacaBarSource(KEY, SECRET, session=counting)
    panel, empty = fetch_iex_closes(source, ["AAA", "GONE"], "2025-01-02")

    assert counting.n_requests == 1
    assert empty == ["GONE"]
    assert panel["AAA"].tolist() == [100.0, 101.0]
    params = fake.calls[0]["params"]
    assert params["feed"] == "iex" and params["adjustment"] == "split"
    assert params["start"] == "2025-01-02" and params["symbols"] == "AAA,GONE"


# --------------------------------------------------------------- align / stats


def test_align_panels_diff_bps_and_coverage_accounting() -> None:
    idx = naive_dates("2026-01-05", "2026-01-06", "2026-01-07")
    spine = pd.DataFrame(
        {
            "AAA": [100.0, 100.0, 100.0],
            "BBB": [50.0, np.nan, 50.0],
            "SPINEONLY": [1.0, 1.0, 1.0],
            # Priced by spine only on 01-06, a session IEX lacks -> zero overlap.
            "DEBUT": [np.nan, 9.0, np.nan],
        },
        index=idx,
    )
    iex = pd.DataFrame(
        {
            "AAA": [100.1, np.nan, 100.0],
            "BBB": [50.0, 50.05, 50.0],
            "IEXONLY": [2.0, 2.0, 2.0],
            "DEBUT": [np.nan, np.nan, 9.0],
        },
        index=naive_dates("2026-01-05", "2026-01-07", "2026-01-08"),  # no 01-06
    )
    diff, coverage = align_panels(spine, iex, "2026-01-05")

    assert coverage["n_common_sessions"] == 2  # 05 and 07
    assert coverage["n_common_names"] == 3
    assert coverage["names_no_valid_diff_days"] == ["DEBUT"]
    assert coverage["names_spine_only"] == ["SPINEONLY"]
    assert coverage["names_iex_only"] == ["IEXONLY"]
    assert coverage["sessions_spine_only"] == ["2026-01-06"]
    assert coverage["sessions_iex_only"] == ["2026-01-08"]
    assert coverage["window_effective"] == ["2026-01-05", "2026-01-07"]
    # AAA 01-05: (100.1/100 - 1) * 1e4 = 10 bps; AAA 01-07: IEX NaN -> NaN, counted.
    assert diff.loc[idx[0], "AAA"] == pytest.approx(10.0)
    assert np.isnan(diff.loc[idx[2], "AAA"])
    assert coverage["name_days_nan_iex"] == 1
    assert coverage["names_partial_iex"] == {"AAA": 1}
    # BBB 01-07: (50.05/50 - 1) * 1e4 = 10 bps.
    assert diff.loc[idx[2], "BBB"] == pytest.approx(10.0)


def test_align_panels_fails_loud_on_no_overlap() -> None:
    a = pd.DataFrame({"AAA": [1.0]}, index=naive_dates("2026-01-05"))
    b = pd.DataFrame({"AAA": [1.0]}, index=naive_dates("2026-02-05"))
    with pytest.raises(SystemExit, match="no overlap"):
        align_panels(a, b, "2026-01-01", "2026-03-01")


def test_per_name_stats_and_adjustment_flagging() -> None:
    idx = naive_dates("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")
    diff = pd.DataFrame(
        {
            "NORMAL": [0.0, 2.0, -4.0, 6.0],
            "SPLIT": [-5000.0, -5000.5, -4999.5, -5000.0],  # 2:1 split after cache freeze
            "EMPTY": [np.nan] * 4,
        },
        index=idx,
    )
    stats = per_name_diff_stats(diff)
    assert set(stats) == {"NORMAL", "SPLIT"}  # all-NaN name yields no stats row
    assert stats["NORMAL"]["mean_bps"] == pytest.approx(1.0)
    assert stats["NORMAL"]["median_bps"] == pytest.approx(1.0)
    assert stats["NORMAL"]["max_abs_bps"] == pytest.approx(6.0)
    assert stats["NORMAL"]["n_days"] == 4

    flagged = flag_adjustment_mismatch(stats, level_bps=250.0)
    assert [f["symbol"] for f in flagged] == ["SPLIT"]


def test_panel_diff_stats_tails_and_exclusion() -> None:
    idx = naive_dates("2026-01-05", "2026-01-06")
    diff = pd.DataFrame(
        {"AAA": [0.0, 10.0], "BBB": [-30.0, 2.0], "SPLIT": [-5000.0, -5000.0]}, index=idx
    )
    panel = panel_diff_stats(diff, thresholds_bps=(5.0, 25.0), exclude=["SPLIT"])
    assert panel["n_name_days"] == 4 and panel["n_names"] == 2
    assert panel["mean_bps"] == pytest.approx(-4.5)
    assert panel["median_abs_bps"] == pytest.approx(6.0)
    assert panel["max_abs_bps"] == pytest.approx(30.0)
    assert panel["frac_exact_equal"] == pytest.approx(0.25)
    assert panel["frac_abs_gt_5bps"] == pytest.approx(0.5)
    assert panel["frac_abs_gt_25bps"] == pytest.approx(0.25)
    assert panel["worst_names_by_max_abs"][0] == {"symbol": "BBB", "max_abs_bps": 30.0}


# ----------------------------------------------------------------- rank impact


def test_month_end_decision_bars_drop_truncated_final_month() -> None:
    calendar = pd.bdate_range("2026-01-02", "2026-03-13")
    bars = month_end_decision_bars(calendar, "2026-01-02")
    assert [str(b.date()) for b in bars] == ["2026-01-30", "2026-02-27"]

    # A calendar ending exactly on a business-month end keeps that month.
    full = pd.bdate_range("2026-01-02", "2026-02-27")
    bars = month_end_decision_bars(full, "2026-01-02")
    assert [str(b.date()) for b in bars] == ["2026-01-30", "2026-02-27"]


def test_score_endpoints_position_math_and_history_gate() -> None:
    calendar = pd.bdate_range("2025-01-02", periods=300)
    skip_date, look_date = score_endpoints(calendar, calendar[260], skip=21, lookback=252)
    assert skip_date == calendar[239] and look_date == calendar[8]
    with pytest.raises(SystemExit, match="prior sessions"):
        score_endpoints(calendar, calendar[100], skip=21, lookback=252)
    with pytest.raises(SystemExit, match="not on the spine calendar"):
        score_endpoints(calendar, pd.Timestamp("2031-01-01"), skip=21, lookback=252)


def test_endpoint_scores_ratio_and_missing_date() -> None:
    idx = naive_dates("2026-01-05", "2026-01-06")
    closes = pd.DataFrame({"AAA": [100.0, 110.0], "BAD": [-1.0, 5.0]}, index=idx)
    scores = endpoint_scores(closes, idx[1], idx[0])
    assert scores["AAA"] == pytest.approx(0.10)
    assert np.isnan(scores["BAD"])  # non-positive base is NaN, not a bogus score
    missing = endpoint_scores(closes, pd.Timestamp("2030-01-01"), idx[0])
    assert missing.isna().all()


def test_decile_legs_floor_count_and_stable_ties() -> None:
    scores = pd.Series({chr(65 + i): float(i) for i in range(10)})  # A..J ascending
    long, short, n_dec = decile_legs(scores, decile=0.10)
    assert n_dec == 1 and long == {"J"} and short == {"A"}

    # Ties break by symbol order identically on repeated calls (stable sort).
    tied = pd.Series({"B": 1.0, "A": 1.0, "C": 0.0, "D": 2.0})
    long1, short1, _ = decile_legs(tied, decile=0.25)
    long2, short2, _ = decile_legs(tied, decile=0.25)
    assert (long1, short1) == (long2, short2) == ({"D"}, {"C"})

    thin = pd.Series({"A": 1.0, "B": 2.0})
    assert decile_legs(thin, decile=0.10) == (frozenset(), frozenset(), 0)


def test_leg_flip_report_counts_boundary_swap() -> None:
    base = {chr(65 + i): float(i) for i in range(10)}  # A..J, long leg {J}, short {A}
    spine = pd.Series(base)
    identical = leg_flip_report(spine, spine.copy(), decile=0.10)
    assert identical["long_flips"] == 0 and identical["short_flips"] == 0
    assert identical["spearman"] == pytest.approx(1.0)
    assert identical["max_abs_score_diff_bps"] == pytest.approx(0.0)

    swapped = dict(base, I=9.5)  # IEX ranks I above J -> one long-leg swap
    report = leg_flip_report(spine, pd.Series(swapped), decile=0.10)
    assert report["long_flips"] == 1 and report["short_flips"] == 0
    assert report["long_out"] == ["J"] and report["long_in"] == ["I"]
    assert report["n_common"] == 10 and report["n_dec_per_leg"] == 1

    # A name scored on one side only is excluded and counted, never dropped silently.
    spine_extra = pd.Series(dict(base, ZZZ=99.0))
    report = leg_flip_report(spine_extra, pd.Series(base), decile=0.10)
    assert report["n_common"] == 10
    assert report["n_excluded_spine_score_only"] == 1
    assert report["excluded_spine_score_only"] == ["ZZZ"]


def test_perturbed_flip_stats_zero_noise_and_determinism() -> None:
    rng = np.random.default_rng(7)
    skip_close = pd.Series(rng.uniform(50, 150, 40), index=[f"S{i:02d}" for i in range(40)])
    look_close = pd.Series(rng.uniform(50, 150, 40), index=[f"S{i:02d}" for i in range(40)])

    zero = perturbed_flip_stats(
        skip_close, look_close, np.array([0.0]), decile=0.10, n_draws=20, seed=1
    )
    assert zero["mean_flips"] == 0.0 and zero["max_flips"] == 0
    assert zero["n_names"] == 40 and zero["n_dec_per_leg"] == 4

    noisy_a = perturbed_flip_stats(
        skip_close, look_close, np.array([-50.0, 0.0, 50.0]), decile=0.10, n_draws=50, seed=3
    )
    noisy_b = perturbed_flip_stats(
        skip_close, look_close, np.array([-50.0, 0.0, 50.0]), decile=0.10, n_draws=50, seed=3
    )
    assert noisy_a == noisy_b  # seeded MC is reproducible


# -------------------------------------------------------------------- NAV mark


def test_nav_mark_report_bps_and_coverage() -> None:
    idx = naive_dates("2026-01-05", "2026-01-06")
    spine = pd.DataFrame({"AAA": [100.0, 100.0], "BBB": [50.0, 50.0]}, index=idx)
    iex = pd.DataFrame({"AAA": [100.1, 100.0], "BBB": [50.0, np.nan]}, index=idx)
    positions = {"AAA": 10.0, "BBB": -5.0, "NEVER": 3.0}

    nav = nav_mark_report(positions, 1000.0, spine, iex, idx)
    # Day 1: spine NAV 1000 + 1000 - 250 = 1750; IEX NAV 1751 -> +5.714 bps.
    assert nav["per_session"][0]["nav_diff_bps"] == pytest.approx((1751.0 / 1750.0 - 1) * 1e4)
    assert nav["per_session"][0]["n_covered"] == 2
    # Day 2: BBB unpriceable on IEX -> excluded from BOTH marks; diff 0.
    assert nav["per_session"][1]["nav_diff_bps"] == pytest.approx(0.0)
    assert nav["per_session"][1]["n_covered"] == 1
    assert nav["per_session"][1]["n_excluded"] == 2
    assert nav["names_never_covered"] == ["NEVER"]
    assert nav["n_sessions_marked"] == 2
    assert nav["median_abs_nav_diff_bps"] == pytest.approx(
        abs((1751.0 / 1750.0 - 1) * 1e4) / 2.0
    )


def test_nav_mark_fails_loud_when_nothing_marks() -> None:
    idx = naive_dates("2026-01-05")
    spine = pd.DataFrame({"AAA": [100.0]}, index=idx)
    iex = pd.DataFrame({"AAA": [100.0]}, index=idx)
    with pytest.raises(SystemExit, match="no session"):
        nav_mark_report({"AAA": 1.0}, 0.0, spine, iex, naive_dates("2030-01-01"))
