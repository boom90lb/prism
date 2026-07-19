"""Unit tests for the conditional-beta telemetry pure helpers (research tier)."""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.research

from research.scripts.beta_telemetry import (  # noqa: E402
    beta_cell,
    drawdown_state_mask,
    equal_weight_returns,
    rolling_beta,
    rolling_summary,
    strip_sessions,
    trailing_worst_decile_mask,
    worst_month_mask,
)


def sessions(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def alternating_market(n: int, scale: float = 0.01) -> pd.Series:
    return pd.Series([scale if i % 2 == 0 else -scale for i in range(n)], index=sessions(n))


def test_strip_sessions_preserves_dates_across_tz_forms() -> None:
    # tz-aware NY midnight across a DST boundary keeps its session dates.
    ny = pd.to_datetime(
        ["2024-03-07 00:00:00-05:00", "2024-03-11 00:00:00-04:00"], utc=True
    ).tz_convert("America/New_York")
    assert list(strip_sessions(ny)) == list(pd.to_datetime(["2024-03-07", "2024-03-11"]))
    # naive date index passes through unchanged.
    naive = pd.DatetimeIndex(["2026-01-02", "2026-01-05"])
    assert list(strip_sessions(naive)) == list(naive)
    # UTC representation of NY midnight resolves to the NY session date.
    utc = pd.DatetimeIndex(["2024-03-07 05:00:00+00:00"])
    assert strip_sessions(utc)[0] == pd.Timestamp("2024-03-07")


def test_beta_cell_recovers_planted_beta_and_alpha() -> None:
    market = alternating_market(100)
    book = 0.5 * market + 0.0002
    cell = beta_cell(book, market, min_obs=21)
    assert cell["n"] == 100
    assert cell["beta"] == pytest.approx(0.5)
    assert cell["alpha_daily"] == pytest.approx(0.0002)
    assert cell["r2"] == pytest.approx(1.0)


def test_beta_cell_null_below_min_obs_reports_n() -> None:
    market = alternating_market(10)
    cell = beta_cell(0.5 * market, market, min_obs=21)
    assert cell["beta"] is None and cell["n"] == 10
    assert "min_obs=21" in cell["note"]


def test_beta_cell_null_on_zero_variance_market() -> None:
    idx = sessions(30)
    market = pd.Series(0.001, index=idx)
    book = pd.Series(np.linspace(-0.01, 0.01, 30), index=idx)
    cell = beta_cell(book, market, min_obs=5)
    assert cell["beta"] is None and cell["n"] == 30
    assert "variance" in cell["note"]


def test_beta_cell_mask_recovers_planted_conditional_flip() -> None:
    # Book is +1x the market on calm days and -1x inside the masked regime:
    # the unconditional beta averages toward zero while the conditional cell
    # recovers the flip — exactly the false-reassurance the instrument exists
    # to expose.
    market = alternating_market(200)
    regime = pd.Series([i >= 100 for i in range(200)], index=market.index)
    book = market.where(~regime, -market)
    assert abs(beta_cell(book, market, min_obs=21)["beta"]) < 1e-12
    conditional = beta_cell(book, market, mask=regime, min_obs=21)
    assert conditional["n"] == 100
    assert conditional["beta"] == pytest.approx(-1.0)


def test_rolling_beta_endpoints_and_summary() -> None:
    market = alternating_market(126)
    book = pd.concat([2.0 * market.iloc[:63], -1.0 * market.iloc[63:]])
    betas = rolling_beta(book, market, window=63)
    assert len(betas) == 126 - 63 + 1
    assert betas.iloc[0] == pytest.approx(2.0)
    assert betas.iloc[-1] == pytest.approx(-1.0)
    summary = rolling_summary(betas, window=63)
    assert summary["n_windows"] == 64
    assert summary["min"] == pytest.approx(-1.0)
    assert summary["max"] == pytest.approx(2.0)
    assert summary["last"] == pytest.approx(-1.0)
    assert summary["last_date"] == str(betas.index[-1].date())
    empty = rolling_summary(betas.iloc[:0], window=63)
    assert empty["n_windows"] == 0 and empty["mean"] is None and "note" in empty


def test_trailing_worst_decile_mask_is_strictly_prior() -> None:
    # 100 calm sessions, a 21-session crash, 30 calm sessions.
    values = [0.001] * 100 + [-0.02] * 21 + [0.001] * 30
    market = pd.Series(values, index=sessions(151))
    mask, threshold, n_defined = trailing_worst_decile_mask(market, market.index, bars=21, q=0.10)
    assert n_defined == 151 - 21  # trailing defined from bar 21; prior shifts one more
    assert threshold < 0.0
    # The crash's first day is NOT flagged: its prior trailing window is calm.
    assert not mask.iloc[100]
    # The day after the deepest trailing window (crash fully inside) IS flagged.
    assert mask.iloc[121]
    # Roughly a decile of the defined days is flagged, and never none.
    assert 0 < int(mask.sum()) <= int(0.15 * n_defined) + 1
    # Every flagged day's conditioning really is prior: unflagging day 121
    # would require its previous-session trailing return to sit above the
    # threshold, but it is the sample minimum.
    trailing_ending_120 = (1.0 - 0.02) ** 21 - 1.0
    assert trailing_ending_120 <= threshold


def test_drawdown_state_mask_lags_one_session() -> None:
    values = [0.01] * 5 + [-0.03] * 3 + [0.01] * 5
    book = pd.Series(values, index=sessions(13))
    state, drawdown = drawdown_state_mask(book, threshold=0.05)
    # Drawdown after two -3% days is ~5.9%: crosses the 5% line at day 6,
    # so the *state* (prior-close reading) first fires on day 7.
    assert drawdown.iloc[6] == pytest.approx(1.0 - 0.97**2)
    assert not state.iloc[6]
    assert state.iloc[7]
    assert state.iloc[8]  # prior close (day 7) was ~8.7% down
    assert not state.iloc[0]  # first session has no prior close; fillna(0), not NaN


def test_worst_month_mask_selects_the_worst_month_days() -> None:
    idx = pd.date_range("2024-01-01", "2024-10-31", freq="B")
    book = pd.Series(0.002, index=idx)
    book[book.index.to_period("M") == pd.Period("2024-06", "M")] = -0.01
    mask, threshold, months, n_months = worst_month_mask(book, idx, q=0.10)
    assert n_months == 10
    assert months == ["2024-06"]
    june = idx.to_period("M") == pd.Period("2024-06", "M")
    # Linear-interpolation quantile sits between the two lowest months:
    # at or above the outlier, strictly below every calm month.
    assert threshold >= (1.0 - 0.01) ** int(june.sum()) - 1.0
    assert threshold < (1.0 + 0.002) ** 19 - 1.0
    assert mask[june].all() and not mask[~june].any()


def test_equal_weight_returns_skips_missing_names() -> None:
    idx = sessions(4)
    closes = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.01, 103.0301],
            "BBB": [50.0, 49.5, np.nan, np.nan],
            "CCC": [200.0, 202.0, 204.02, 206.0602],
        },
        index=idx,
    )
    proxy, counts = equal_weight_returns(closes)
    # Day 1: all three names; day 2: BBB has no bar -> mean of the two.
    assert proxy.loc[idx[1]] == pytest.approx((0.01 - 0.01 + 0.01) / 3)
    assert proxy.loc[idx[2]] == pytest.approx(0.01)
    assert counts.loc[idx[1]] == 3 and counts.loc[idx[2]] == 2
    # Day 0 has no prior bar for any name and is dropped, not read as zero.
    assert idx[0] not in proxy.index
