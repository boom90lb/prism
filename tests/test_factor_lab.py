"""Synthetic-data tests for the public factor lab (research tier).

They pin the properties the registration relies on: parsers read the vendors'
layouts, every sizing rule is causal (a scale decided at close t uses data
through t and applies from t+1), the crash rule follows its pinned state
machine, costs are charged as registered, and the verdict rules read as the
design doc states them.
"""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.research

from research.scripts import factor_lab as fl  # noqa: E402


def _zip(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def test_parse_french_reads_first_daily_section_in_decimals():
    text = (
        "This file was created by using the 202607 CRSP database.\n"
        "Some description line\n"
        "\n"
        ",Mkt-RF,RF\n"
        "19260701,    0.10,   0.01\n"
        "19260702,  -99.99,   0.01\n"
        "19260706,   -0.50,   0.01\n"
        "\n"
        "Copyright 2026 Eugene F. Fama and Kenneth R. French\n"
    )
    frame, vintage = fl.parse_french(_zip("x.csv", text))
    assert vintage is not None and vintage.startswith("This file was created by using the 202607 CRSP database")
    assert list(frame.columns) == ["Mkt-RF", "RF"]
    assert len(frame) == 3
    assert frame["Mkt-RF"].iloc[0] == pytest.approx(0.001)
    assert np.isnan(frame["Mkt-RF"].iloc[1])  # missing code, not a -99.99% day
    assert frame.index[2] == pd.Timestamp("1926-07-06")


def test_parse_jkp_pivots_long_file():
    text = "location,name,freq,weighting,date,ret\nusa,a,daily,vw_cap,2000-01-03,0.01\n" \
           "usa,b,daily,vw_cap,2000-01-03,-0.02\nusa,a,daily,vw_cap,2000-01-04,0.03\n"
    wide = fl.parse_jkp(_zip("x.csv", text))
    assert list(wide.columns) == ["a", "b"]
    assert wide.loc["2000-01-04", "a"] == pytest.approx(0.03)
    assert np.isnan(wide.loc["2000-01-04", "b"])


def test_require_contiguous_trims_edges_and_raises_inside():
    index = pd.bdate_range("2000-01-03", periods=5)
    edges = pd.Series([np.nan, 0.1, 0.2, 0.3, np.nan], index=index)
    assert len(fl.require_contiguous(edges, "edges")) == 3
    inside = pd.Series([0.1, np.nan, 0.2, 0.3, 0.4], index=index)
    with pytest.raises(fl.LabError):
        fl.require_contiguous(inside, "inside")


def test_compound_masks_periods_with_missing_days():
    index = pd.bdate_range("2000-01-03", "2000-02-29")
    returns = pd.Series(0.01, index=index)
    returns.loc["2000-02-15"] = np.nan
    monthly = fl.compound(returns, "ME")
    assert monthly.iloc[0] == pytest.approx(1.01 ** len(returns.loc["2000-01"]) - 1)
    assert np.isnan(monthly.iloc[1])


def _perturbed(values: np.ndarray, k: int, seed: int = 7) -> np.ndarray:
    changed = values.copy()
    changed[k:] = np.random.default_rng(seed).normal(0, 0.05, size=values.size - k)
    return changed


@pytest.mark.parametrize("k", [300, 421, 630])
def test_volatility_rules_are_causal(k):
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, size=900)
    changed = _perturbed(returns, k)
    for rule, kwargs in (
        (fl.scale_constant_vol, {"target_vol": 0.12, "lookback_sessions": 126}),
        (fl.scale_variance_managed, {"lookback_sessions": 21, "min_history_refreshes": 5}),
    ):
        left = rule(returns, refresh=21, max_scale=2.0, **kwargs)
        right = rule(changed, refresh=21, max_scale=2.0, **kwargs)
        # a change at session k can move only the scales of sessions after k
        np.testing.assert_array_equal(left[: k + 1], right[: k + 1])


def test_constant_vol_applies_from_the_next_session_and_caps():
    returns = np.full(300, 0.0005) * np.tile([1, -1], 150)  # tiny volatility => capped scale
    scale = fl.scale_constant_vol(returns, refresh=21, max_scale=2.0, target_vol=0.12, lookback_sessions=126)
    first_decision = next(i for i in range(0, 300, 21) if i + 1 >= 126)
    assert np.all(scale[: first_decision + 1] == 1.0)
    assert scale[first_decision + 1] == 2.0


def test_crash_state_uses_returns_through_the_prior_session():
    market = np.zeros(40)
    market[25] = -0.05  # a crash on session 25
    state = fl.crash_state(market, trailing=21, theta=-0.02)
    assert not state[25]  # the decision at session 25 cannot see session 25's return
    assert state[26:40].all()  # session 25 stays inside the 21-session trailing window


def test_crash_degross_state_machine():
    n = 120
    returns = np.zeros(n)
    market = np.zeros(n)
    market[30] = -0.05  # S = 1 on sessions 31..51 (window of 21 ending t-1 contains 30)
    scale = fl.scale_crash_degross(returns, market_total=market, refresh=21, theta=-0.02, trailing_sessions=21,
                                   multiplier=0.5, exit_clean_sessions=5)
    assert np.all(scale[:32] == 1.0)  # decided at close 31, effective 32
    assert scale[32] == 0.5
    # S returns to 0 at session 52; the first refresh with a trailing clean run >= 5 is 63 (clean since 52)
    assert np.all(scale[32:64] == 0.5)
    assert np.all(scale[64:] == 1.0)
    assert set(np.unique(scale)) == {0.5, 1.0}  # the multiplier never compounds


def test_crash_degross_is_causal():
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.01, size=400)
    market = rng.normal(0, 0.012, size=400)
    k = 200
    base = fl.scale_crash_degross(returns, market_total=market, refresh=21, theta=-0.03, trailing_sessions=21,
                                  multiplier=0.5, exit_clean_sessions=5)
    moved = fl.scale_crash_degross(returns, market_total=_perturbed(market, k), refresh=21, theta=-0.03,
                                   trailing_sessions=21, multiplier=0.5, exit_clean_sessions=5)
    np.testing.assert_array_equal(base[: k + 2], moved[: k + 2])


def test_apply_scale_charges_changes_and_base_cost():
    returns = np.array([0.01, 0.01, 0.01])
    scale = np.array([1.0, 0.5, 0.5])
    net, cost = fl.apply_scale(returns, scale, base_gross=2.0, change_cost_bps=10.0,
                               base_cost_per_gross_year=0.00499)
    base = 2.0 * 0.00499 / 252
    assert cost[0] == pytest.approx(base)
    assert cost[1] == pytest.approx(0.5 * 2.0 * 10 / 1e4 + 0.5 * base)
    assert net[2] == pytest.approx(0.005 - 0.5 * base)


def test_verdict_rules_follow_the_design_doc():
    window = {"max_drawdown": 0.30, "worst_21_session": -0.20}
    better = {"max_drawdown": 0.20, "worst_21_session": -0.10}
    worse_dd = {"max_drawdown": 0.35, "worst_21_session": -0.10}
    ok_interval = {"lower": -0.001, "upper": 0.002}
    below = {"lower": -0.003, "upper": -0.0001}
    assert fl.crash_rule_verdict(ok_interval, better, window) == "supported"
    assert fl.crash_rule_verdict(below, better, window) == "contradicted"
    assert fl.crash_rule_verdict(ok_interval, worse_dd, window) == "contradicted"
    assert fl.crash_rule_verdict(ok_interval, {"max_drawdown": 0.2, "worst_21_session": -0.25}, window) == \
        "inconclusive"

    above = {"lower": 0.001, "upper": 0.004, "difference": 0.002}
    post_ok = {"lower": -0.002, "upper": 0.003, "difference": 0.0005}
    post_neg = {"lower": -0.004, "upper": 0.001, "difference": -0.001}
    post_below = {"lower": -0.004, "upper": -0.001, "difference": -0.002}
    assert fl.published_rule_verdict(above, post_ok) == "supported"
    assert fl.published_rule_verdict(above, post_neg) == "inconclusive"
    assert fl.published_rule_verdict(above, post_below) == "contradicted"
    assert fl.combine_verdicts(["supported", "inconclusive"]) == "inconclusive"
    assert fl.combine_verdicts(["supported", "contradicted"]) == "contradicted"


def test_prior_record_applies_the_section_4_rule():
    cfg = {"sessions_per_year": 252, "publication_bar_t": 3.0, "publication_bar_min_years": 20.0,
           "post_publication_min_years": 10.0, "pre_publication_haircut": 0.4}
    index = pd.bdate_range("1970-01-01", "2020-12-31")
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0.0008, 0.01, size=len(index)), index=index)
    long_post = fl.prior_record(returns, "1990-01", cfg)
    assert long_post["prior_basis"] == "post_publication"
    assert long_post["prior_sharpe"] == pytest.approx(long_post["post"]["sharpe"])
    short_post = fl.prior_record(returns, "2015-01", cfg)
    assert short_post["prior_basis"] == "haircut_full_sample"
    assert short_post["prior_sharpe"] == pytest.approx(0.4 * short_post["full"]["sharpe"])
    assert fl.prior_record(returns, None, cfg)["prior_basis"] == "descriptive_only"


def _synthetic_public_data(reg):
    index = pd.bdate_range("2005-01-03", "2022-12-30")
    rng = np.random.default_rng(5)
    series = {base: pd.Series(rng.normal(0.0003, 0.01, size=len(index)), index=index) for base in ("UMD", "RESMOM")}
    market = pd.Series(rng.normal(0.0003, 0.012, size=len(index)), index=index).drop(index[100])
    return fl.PublicData(series=series, market_total=market, jkp_large_monthly={}, vintages={})


def test_sizing_driver_end_to_end_on_synthetic_data(tmp_path):
    reg = fl.load_registration()
    reg["sizing"]["bootstrap"]["n_resamples"] = 50
    data = _synthetic_public_data(reg)
    fetched = {name: {"sha256": "0" * 64} for name in ("ff_mom_daily", "ff3_daily", "jkp_resmom_daily")}
    log_path = tmp_path / "sizing.jsonl"
    summary = fl.run_sizing(reg, data, fetched, tmp_path, log_path, "new", "deadbeef")

    rows = fl.TrialLog(log_path).verify()
    assert [row["event"] for row in rows].count("started") == 8
    assert [row["event"] for row in rows].count("finished") == 8
    assert summary["calendar_drops"] == {"UMD": [str(data.series["UMD"].index[100].date())],
                                         "RESMOM": [str(data.series["RESMOM"].index[100].date())]}
    assert set(summary["verdicts"]) == {"crash_degross_v1", "constant_vol_bsc", "variance_managed_mm"}
    assert "post_publication" in summary["arms"]["UMD:constant_vol_bsc"]["intervals_vs_none"]
    assert (tmp_path / "sizing" / "returns.parquet").exists()

    with pytest.raises(fl.LabError):  # a second evaluation of a finished arm is not a new trial
        fl.run_sizing(reg, data, fetched, tmp_path, log_path, "new", "deadbeef")
    fl.run_sizing(reg, data, fetched, tmp_path, log_path, "retry", "deadbeef")  # nothing to retry
    assert len(fl.TrialLog(log_path).verify()) == 16


def test_priors_driver_on_synthetic_data(tmp_path):
    reg = fl.load_registration()
    data = _synthetic_public_data(reg)
    result = fl.run_priors(reg, data, tmp_path)
    assert result["priors"]["UMD"]["publication"] == "1993-03"
    assert result["priors"]["UMD"]["prior_basis"] == "post_publication"
    assert (tmp_path / "priors.json").exists()
    assert len(result["correlation_files"]) == 4
