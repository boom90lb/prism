"""Synthetic-data tests for the fundamentals replication read (research tier).

They pin the capped volatility rule (causal; the gross ceiling binds), the equal-risk
combination (parts averaged, costs summed without netting, total gross within the
ceiling), the per-arm criteria (tracking, and a sign mismatch recorded, not passed),
the part-eligibility rule (the combination is skipped when fewer than three parts pass
criteria 1 and 2), and the promotion callback.
"""

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.research

from research.scripts import fundamentals_replication as fr  # noqa: E402

CFG = {"target_vol": 0.12, "lookback_sessions": 126, "refresh_sessions": 21, "part_max_scale": 2.0,
       "max_total_gross": 2.0, "scale_change_cost_bps": 10.0}


def test_capped_scale_is_causal_and_honours_the_cap():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.002, size=600)  # ~3% vol: an uncapped scale would be ~4
    caps = np.full(600, 1.5)
    scale = fr.capped_scale(returns, caps, 0.12, 126, 21)
    assert scale.max() == pytest.approx(1.5) and (scale[:127] == 1.0).all()
    changed = returns.copy()
    changed[300:] = rng.normal(0, 0.05, size=300)
    np.testing.assert_array_equal(scale[:301], fr.capped_scale(changed, caps, 0.12, 126, 21)[:301])


def _part(index, seed, vol=0.0075, cost=0.0001, scale=1.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"net": rng.normal(0.0003, vol, size=len(index)), "charged_cost": cost, "scale": scale,
                         "gross": 1.0}, index=index)


def test_combine_averages_parts_and_keeps_gross_within_the_ceiling():
    index = pd.bdate_range("2020-01-01", periods=700)
    parts = {"a": _part(index, 1, scale=1.8), "b": _part(index, 2, scale=1.8), "c": _part(index, 3, scale=1.8)}
    combined = fr.combine(parts, CFG)
    assert combined.index[0] == index[127]
    assert (combined["gross"] <= 2.0 + 1e-9).all()  # 1.8 average part gross caps the combination scale at 2/1.8
    assert combined["scale"].max() <= 2.0 / 1.8 + 1e-9
    one_day = combined.index[5]
    expected_cost = combined.loc[one_day, "scale"] * 0.0001  # a scale-constant day: no change cost
    assert combined.loc[one_day, "charged_cost"] == pytest.approx(expected_cost)


def _jkp_zip(name, index, values, direction=1):
    frame = pd.DataFrame({"location": "usa", "name": name, "freq": "daily", "weighting": "vw_cap",
                          "direction": direction, "n_stocks": 100, "date": index.strftime("%Y-%m-%d"), "ret": values})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"[usa]_[{name}]_[daily]_[vw_cap].csv", frame.to_csv(index=False))
    return buffer.getvalue()


def _french_cash_zip(index):
    rows = "\n".join(f"{d:%Y%m%d},0.1,0.1,0.1,0.0150" for d in index)
    text = f"This file was created by using the 202607 CRSP database.\n\n,Mkt-RF,SMB,HML,RF\n{rows}\n\nCopyright\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_Factors_daily.csv", text)
    return buffer.getvalue()


def _setup(tmp_path, tracking: float | str = 0.9):
    rng = np.random.default_rng(7)
    public_index = pd.bdate_range("1960-01-04", "2025-12-31")
    house_index = pd.bdate_range("2022-03-29", "2025-12-31")
    chars = ["be_me", "gp_at", "at_gr1", "oaccruals_at", "niq_su", "chcsho_12m"]
    paths, arms = {}, {}
    for i, c in enumerate(chars):
        public = pd.Series(rng.normal(0.0004, 0.006, size=len(public_index)), index=public_index)
        path = tmp_path / f"{c}.zip"
        path.write_bytes(_jkp_zip(c, public_index, public.to_numpy()))
        paths[c] = path
        overlap = public.reindex(house_index).to_numpy()
        noise = rng.normal(0, 0.006, size=len(house_index))
        sign = -1.0 if (tracking == "one_flipped" and i == 0) else 1.0
        weight = 0.9 if tracking == "one_flipped" else tracking
        arms[c] = pd.DataFrame({"net": sign * weight * overlap + np.sqrt(1 - weight**2) * noise,
                                "charged_cost": 0.00002, "gross": 1.0}, index=house_index)
    cash = tmp_path / "cash.zip"
    cash.write_bytes(_french_cash_zip(public_index))
    momentum = pd.DataFrame(index=public_index)
    momentum["reference|scale"] = 1.0
    momentum["reference|gross"] = rng.normal(0.0004, 0.0075, size=len(public_index))
    momentum["in_house|scale"] = np.nan
    momentum["in_house|net"] = np.nan
    momentum["in_house|charged_cost"] = np.nan
    live = public_index >= "2022-09-29"
    momentum.loc[live, "in_house|scale"] = 1.0
    momentum.loc[live, "in_house|net"] = rng.normal(0.0004, 0.0075, size=int(live.sum()))
    momentum.loc[live, "in_house|charged_cost"] = 0.00002
    series = tmp_path / "momentum.csv"
    momentum.to_csv(series)
    reg = fr.load_registration(fr.REGISTRATION)
    reg["combination"]["momentum_series"] = str(series)
    reg["bootstrap"]["n_resamples"] = 100
    return reg, arms, paths, cash


def test_evaluate_end_to_end_and_the_sign_guard(tmp_path):
    reg, arms, paths, cash = _setup(tmp_path, tracking="one_flipped")
    summary = fr.evaluate(reg, arms, paths, cash)
    assert summary["rf_annual"] == pytest.approx(0.00015 * 252)
    flipped = summary["arms"]["be_me"]
    assert flipped["fidelity"]["direction_mismatch"] and not flipped["passes"]["fidelity"]
    assert all(summary["arms"][c]["passes"]["fidelity"] for c in ["gp_at", "at_gr1", "oaccruals_at", "niq_su"])
    combo = summary["combination"]
    assert combo["evaluated"] and "be_me" not in combo["parts"] and "vol_momentum" in combo["parts"]
    assert combo["prior"]["basis"] == "post_publication" and combo["prior"]["window_from"] == "2015-04-01"


def test_combination_is_skipped_when_too_few_parts_track(tmp_path):
    reg, arms, paths, cash = _setup(tmp_path, tracking=0.2)
    summary = fr.evaluate(reg, arms, paths, cash)
    assert not any(r["passes"]["fidelity"] for r in summary["arms"].values())
    assert summary["combination"]["evaluated"] is False and summary["combination_promoted"] is False


def test_promotion_outcome_reads_the_written_record(tmp_path):
    path = tmp_path / "replication.json"
    path.write_text(json.dumps({"promoted_arms": ["gp_at"], "combination_promoted": True,
                                "arms": {"gp_at": {"passes": {"fidelity": True}}},
                                "combination": {"passes": {"economics": True}}}))
    outcome = fr.promotion_outcome(path)
    assert outcome["promoted_arms"] == ["gp_at"] and outcome["combination_promoted"] is True
