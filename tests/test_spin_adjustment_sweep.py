"""Spin-adjustment-consistency sweep (docs/spin_adjustment_sweep.md) — offline classification math.

Pins the sweep's event classification on synthetic close panels and
synthetic action records: a clean back-adjusted split, a raw split step, a
raw spin distribution (the APTV class, including the market-netting that
recovers the mechanical step under a market move), and the loud
INDETERMINATE paths (missing bars, undeterminable distribution value,
sub-floor step). The network seam is an injectable session, so the fetch
shape — full-taxonomy grouping, pagination, batch dedupe, Retry-After —
is exercised offline too. No network in any test.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.scripts.spin_adjustment_sweep import (
    aptv_cross_check,
    build_events,
    classify,
    cross_event_read,
    dedupe_records,
    fetch_corporate_actions,
    spin_mechanical_ratio,
    split_mechanical_ratio,
    stock_dividend_mechanical_ratio,
    summarize,
    sweep_events,
)

# Ten business sessions; the event lands on index 5 (2026-03-09, a Monday).
IDX = pd.bdate_range("2026-03-02", periods=10)
EX = pd.Timestamp("2026-03-09")
FLAT = {"M1": [100.0] * 10, "M2": [50.0] * 10, "M3": [80.0] * 10}


def _panel(extra: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame({**FLAT, **extra}, index=IDX)


def _rows(panel: pd.DataFrame, by_type: dict, raw_closes: dict | None = None) -> list[dict]:
    events, _ = build_events(by_type, set(panel.columns), IDX[0], IDX[-1])
    return sweep_events(panel, events, raw_closes or {})


# ---------------------------------------------------------------------------
# The four required classes
# ---------------------------------------------------------------------------


def test_back_adjusted_split_shows_no_step() -> None:
    # The spine already folded the 4:1 forward split into history: the series
    # is level across the ex-date, so the event classifies BACK_ADJUSTED.
    record = {"symbol": "ADJ", "old_rate": 1, "new_rate": 4, "ex_date": "2026-03-09"}
    rows = _rows(_panel({"ADJ": [25.0] * 10}), {"forward_splits": [record]})
    assert len(rows) == 1
    row = rows[0]
    assert row["classification"] == "BACK_ADJUSTED"
    assert row["mechanical_ratio"] == pytest.approx(0.25)
    assert row["step_bps"] is None
    assert row["spine_cross_event_return_bps"] == pytest.approx(0.0)


def test_raw_split_step_is_flagged_with_its_size() -> None:
    # The spine did NOT adjust: the close falls 100 -> 25 across the ex-date,
    # exactly the mechanical 4:1 ratio — RAW_STEP at −7,500 bps.
    record = {"symbol": "RAW", "old_rate": 1, "new_rate": 4, "ex_date": "2026-03-09"}
    rows = _rows(_panel({"RAW": [100.0] * 5 + [25.0] * 5}), {"forward_splits": [record]})
    row = rows[0]
    assert row["classification"] == "RAW_STEP"
    assert row["step_bps"] == pytest.approx(-7500.0)
    assert row["mechanical_step_bps"] == pytest.approx(-7500.0)


def test_raw_spin_distribution_step() -> None:
    # 1-for-3 distribution, child debuts at 39 against a 100 parent close:
    # fraction 0.13, mechanical ratio 0.87. The spine shows the full raw step.
    record = {
        "source_symbol": "PAR",
        "new_symbol": "KID",
        "source_rate": 3.0,
        "new_rate": 1.0,
        "ex_date": "2026-03-09",
    }
    raw = {
        "PAR": pd.Series([100.0] * 5 + [87.0] * 5, index=IDX),
        "KID": pd.Series([39.0] * 5, index=IDX[5:]),
    }
    rows = _rows(_panel({"PAR": [100.0] * 5 + [87.0] * 5}), {"spin_offs": [record]}, raw)
    row = rows[0]
    assert row["classification"] == "RAW_STEP"
    assert row["mechanical_ratio"] == pytest.approx(0.87)
    assert row["step_bps"] == pytest.approx(-1300.0)
    assert row["spin_detail"]["child_debut_raw_close"] == pytest.approx(39.0)
    assert row["spin_detail"]["parent_prev_raw_close"] == pytest.approx(100.0)


def test_missing_bars_are_indeterminate_and_named() -> None:
    # No finite close for GAP anywhere near the ex-date: the event must be
    # INDETERMINATE with a missing-bars reason, never silently dropped (N7).
    closes = [100.0, 100.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    record = {"symbol": "GAP", "old_rate": 1, "new_rate": 2, "ex_date": "2026-03-09"}
    rows = _rows(_panel({"GAP": closes}), {"forward_splits": [record]})
    row = rows[0]
    assert row["classification"] == "INDETERMINATE"
    assert "missing bars" in row["reason"]


# ---------------------------------------------------------------------------
# Netting, floors, and the remaining INDETERMINATE paths
# ---------------------------------------------------------------------------


def test_market_move_is_netted_before_classification() -> None:
    # The APTV shape: mechanical ratio 0.867 but the observed cross-event
    # return is only −10.58% because the market rose ~3% that day. Netting
    # the panel median move recovers the mechanical step — RAW_STEP.
    market = {name: [100.0] * 5 + [103.0] * 5 for name in ("M1", "M2", "M3")}
    panel = pd.DataFrame({**market, "PAR": [100.0] * 5 + [89.42] * 5}, index=IDX)
    record = {
        "source_symbol": "PAR",
        "new_symbol": "KID",
        "source_rate": 3.0,
        "new_rate": 1.0,
        "ex_date": "2026-03-09",
    }
    raw = {
        "PAR": pd.Series([100.0] * 5 + [89.42] * 5, index=IDX),
        "KID": pd.Series([39.9] * 5, index=IDX[5:]),
    }
    rows = _rows(panel, {"spin_offs": [record]}, raw)
    row = rows[0]
    assert row["classification"] == "RAW_STEP"
    assert row["step_bps"] == pytest.approx(-1058.0, abs=1.0)  # reported step = the spine's own return
    assert row["market_move_bps"] == pytest.approx(300.0, abs=1.0)


def test_back_adjusted_spin_with_determinable_value() -> None:
    # The spine absorbed the spin (level series) while the raw parent shows
    # the step: mechanical value determinable, classification BACK_ADJUSTED.
    record = {
        "source_symbol": "PAR",
        "new_symbol": "KID",
        "source_rate": 3.0,
        "new_rate": 1.0,
        "ex_date": "2026-03-09",
    }
    raw = {
        "PAR": pd.Series([100.0] * 5 + [87.0] * 5, index=IDX),
        "KID": pd.Series([39.0] * 5, index=IDX[5:]),
    }
    rows = _rows(_panel({"PAR": [87.0] * 10}), {"spin_offs": [record]}, raw)
    assert rows[0]["classification"] == "BACK_ADJUSTED"


def test_undeterminable_distribution_value_is_indeterminate() -> None:
    # No child raw series: the distribution value cannot be computed and the
    # event says so by name — even though the spine shows a step.
    record = {
        "source_symbol": "PAR",
        "new_symbol": "KID",
        "source_rate": 3.0,
        "new_rate": 1.0,
        "ex_date": "2026-03-09",
    }
    raw = {"PAR": pd.Series([100.0] * 5 + [87.0] * 5, index=IDX)}
    rows = _rows(_panel({"PAR": [100.0] * 5 + [87.0] * 5}), {"spin_offs": [record]}, raw)
    row = rows[0]
    assert row["classification"] == "INDETERMINATE"
    assert "KID" in row["reason"] and "unavailable" in row["reason"]


def test_sub_floor_step_is_indeterminate_not_coinflipped() -> None:
    # A 1% stock dividend separates the two hypotheses by less than daily
    # noise can resolve: INDETERMINATE by the separability floor.
    record = {"symbol": "TINY", "rate": 0.01, "ex_date": "2026-03-09"}
    rows = _rows(_panel({"TINY": [100.0] * 10}), {"stock_dividends": [record]})
    row = rows[0]
    assert row["classification"] == "INDETERMINATE"
    assert "separability floor" in row["reason"]


def test_matches_neither_hypothesis_is_indeterminate() -> None:
    # Observed −60% against a mechanical −75%: nearer RAW_STEP but far outside
    # the noise budget — reported as matching neither, with the nearest named.
    verdict = classify(0.40, 0.25, 1.0)
    assert verdict["classification"] == "INDETERMINATE"
    assert "neither" in verdict["reason"] and "RAW_STEP" in verdict["reason"]


# ---------------------------------------------------------------------------
# Mechanical-ratio helpers
# ---------------------------------------------------------------------------


def test_split_ratio_reads_the_record() -> None:
    assert split_mechanical_ratio({"old_rate": 3, "new_rate": 1})[0] == pytest.approx(3.0)  # 3:1 reverse
    assert split_mechanical_ratio({"old_rate": 1, "new_rate": 4})[0] == pytest.approx(0.25)  # 1:4 forward
    ratio, reason = split_mechanical_ratio({"old_rate": None, "new_rate": 4})
    assert ratio is None and "not determinable" in reason


def test_stock_dividend_ratio() -> None:
    assert stock_dividend_mechanical_ratio({"rate": 0.25})[0] == pytest.approx(0.8)
    assert stock_dividend_mechanical_ratio({})[0] is None


def test_spin_ratio_defaults_missing_source_rate_to_one_noted() -> None:
    record = {"source_symbol": "PAR", "new_symbol": "KID", "new_rate": 0.25, "ex_date": "2026-03-09"}
    raw_parent = pd.Series([100.0] * 5 + [90.0] * 5, index=IDX)
    raw_child = pd.Series([40.0] * 5, index=IDX[5:])
    ratio, reason, detail = spin_mechanical_ratio(record, raw_parent, raw_child)
    assert reason is None
    assert ratio == pytest.approx(1.0 - 0.25 * 40.0 / 100.0)
    assert detail["source_rate_defaulted"] is True


# ---------------------------------------------------------------------------
# Windowing, coverage, and the summary
# ---------------------------------------------------------------------------


def test_out_of_window_and_out_of_panel_records_are_counted_not_dropped() -> None:
    by_type = {
        "forward_splits": [
            {"symbol": "ADJ", "old_rate": 1, "new_rate": 2, "ex_date": "2025-01-02"},  # before the window
            {"symbol": "ADJ", "old_rate": 1, "new_rate": 2, "ex_date": "2026-03-09"},  # in
            {"symbol": "ADJ", "old_rate": 1, "new_rate": 2},  # malformed: no ex-date
        ],
        "spin_offs": [
            {"source_symbol": "GHOST", "new_symbol": "KID", "new_rate": 1.0, "ex_date": "2026-03-09"},
        ],
        "cash_dividends": [{"symbol": "ADJ", "rate": 0.5, "ex_date": "2026-03-09"}],
    }
    events, coverage = build_events(by_type, {"ADJ"}, IDX[0], IDX[-1])
    assert len(events) == 1 and events[0]["symbol"] == "ADJ"
    tally = coverage["tested_types"]["forward_splits"]
    assert tally["n_records"] == 3 and tally["n_in_window"] == 1 and tally["n_malformed"] == 1
    assert coverage["not_in_panel"] == [{"type": "spin_offs", "symbol": "GHOST", "ex_date": "2026-03-09"}]
    # The untested type is tallied with its reason, never step-tested.
    assert coverage["untested_types"]["cash_dividends"]["n_records"] == 1
    assert "price-return" in coverage["untested_types"]["cash_dividends"]["note"]


def test_summary_counts_and_flagged_list() -> None:
    panel = _panel({"RAW": [100.0] * 5 + [25.0] * 5, "ADJ": [25.0] * 10})
    by_type = {
        "forward_splits": [
            {"symbol": "RAW", "old_rate": 1, "new_rate": 4, "ex_date": "2026-03-09"},
            {"symbol": "ADJ", "old_rate": 1, "new_rate": 4, "ex_date": "2026-03-09"},
        ]
    }
    rows = _rows(panel, by_type)
    summary, flagged = summarize(rows)
    assert summary["counts"] == {"BACK_ADJUSTED": 1, "RAW_STEP": 1, "INDETERMINATE": 0}
    assert [f["symbol"] for f in flagged] == ["RAW"]
    assert flagged[0]["step_bps"] == pytest.approx(-7500.0)


def test_aptv_cross_check_logic() -> None:
    good = [
        {
            "symbol": "APTV",
            "action_type": "spin_off",
            "ex_date": "2026-04-01",
            "classification": "RAW_STEP",
            "step_bps": -1058.3,
            "spine_cross_event_return_bps": -1058.3,
            "reason": None,
        }
    ]
    assert aptv_cross_check(good)["reproduced"] is True
    assert aptv_cross_check([])["reproduced"] is False
    wrong = [{**good[0], "classification": "BACK_ADJUSTED", "step_bps": None, "spine_cross_event_return_bps": -3.0}]
    assert aptv_cross_check(wrong)["reproduced"] is False


def test_cross_event_read_bridges_short_gaps_and_reports_span() -> None:
    closes = pd.Series([100.0, 100.0, 100.0, 100.0, np.nan, np.nan, 50.0, 50.0, 50.0, 50.0], index=IDX)
    market = pd.Series(0.0, index=IDX)
    read = cross_event_read(closes, market, EX)
    assert read["ok"] and read["r_obs"] == pytest.approx(0.5)
    assert read["n_sessions_spanned"] == 3  # index 3 -> 6, the NaN bars bridged and counted


# ---------------------------------------------------------------------------
# Fetch seam: full taxonomy, pagination, dedupe, Retry-After — all offline
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, payload: dict, status_code: int = 200, headers: dict | None = None) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class FakeSession:
    """Canned responses; records every request's params (creds must be header-only)."""

    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        assert "APCA-API-KEY-ID" in headers and "key" not in url
        self.requests.append(dict(params))
        return self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]


def test_fetch_groups_full_taxonomy_and_paginates() -> None:
    pages = [
        _Response(
            {
                "corporate_actions": {
                    "spin_offs": [{"id": "s1", "source_symbol": "PAR", "ex_date": "2026-03-09"}],
                    "forward_splits": [{"id": "f1", "symbol": "AAA", "ex_date": "2026-02-02"}],
                },
                "next_page_token": "tok",
            }
        ),
        _Response(
            {
                "corporate_actions": {"cash_dividends": [{"id": "d1", "symbol": "AAA", "ex_date": "2026-01-05"}]},
                "next_page_token": None,
            }
        ),
    ]
    session = FakeSession(pages)
    by_type, n_requests = fetch_corporate_actions(
        ["PAR", "AAA"], "2020-01-01", "2026-08-15", key_id="k", secret_key="s", session=session
    )
    assert n_requests == 2
    assert set(by_type) == {"spin_offs", "forward_splits", "cash_dividends"}
    # Full taxonomy: no types filter travels — whatever the endpoint returns is kept.
    assert "types" not in session.requests[0]
    assert session.requests[1]["page_token"] == "tok"


def test_fetch_honors_retry_after_on_429() -> None:
    ok = _Response({"corporate_actions": {"spin_offs": []}, "next_page_token": None})
    session = FakeSession([_Response({}, status_code=429, headers={"Retry-After": "2.5"}), ok])
    sleeps: list[float] = []
    by_type, n_requests = fetch_corporate_actions(
        ["AAA"], "2020-01-01", "2026-08-15", key_id="k", secret_key="s", session=session, sleep=sleeps.append
    )
    assert sleeps == [2.5]
    assert n_requests == 1  # one *successful* page; the 429 was retried, not counted as a page
    assert by_type == {"spin_offs": []}


def test_dedupe_drops_cross_batch_repeats_by_id() -> None:
    by_type = {
        "spin_offs": [
            {"id": "s1", "source_symbol": "PAR", "ex_date": "2026-03-09"},
            {"id": "s1", "source_symbol": "PAR", "ex_date": "2026-03-09"},
            {"source_symbol": "OTHER", "ex_date": "2026-03-10"},
            {"source_symbol": "OTHER", "ex_date": "2026-03-10"},  # no id: full-record identity
        ]
    }
    deduped, dropped = dedupe_records(by_type)
    assert dropped == 2
    assert len(deduped["spin_offs"]) == 2
