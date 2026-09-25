"""Synthetic-data tests for the earnings-announcement replication read (research tier).

They pin the Chen and Zimmermann file parser (percent to decimal, empty portfolios,
refusals), the arm's criteria end to end on a tracking arm, the combination's prior
basis (the haircut when momentum's post-publication window is under ten years), the
skip when the arm does not track, and the promotion callback. The per-arm and
combination arithmetic is the fundamentals read's and is tested there.
"""

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.research

from research.scripts import earnings_replication as er  # noqa: E402
from research.scripts.factor_lab import LabError  # noqa: E402

MEMBER = "PredictorVW/AnnouncementReturn_ret.csv"


def _cz_zip(index, ls_percent, member=MEMBER):
    frame = pd.DataFrame({"date": index.strftime("%Y-%m-%d"), "port01": 0.1, "port02": "NA", "port03": 0.0,
                          "port04": 0.0, "port05": 0.1, "portLS": ls_percent})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("PredictorVW/", "")
        archive.writestr(member, frame.to_csv(index=False))
    return buffer.getvalue()


def test_the_parser_reads_percent_long_short_returns_as_decimals():
    index = pd.bdate_range("1990-01-02", periods=4)
    series = er.parse_cz_daily(_cz_zip(index, [0.5, np.nan, -0.25, 1.0]), MEMBER)
    assert list(series.round(6)) == [0.005, -0.0025, 0.01]  # the empty day is dropped
    with pytest.raises(LabError):
        er.parse_cz_daily(_cz_zip(index, [0.5, 0.1, -0.25, 1.0]), "PredictorVW/Other_ret.csv")
    with pytest.raises(LabError):
        er.parse_cz_daily(_cz_zip(index, [500.0, 600.0, 700.0, 800.0]), MEMBER)  # not percent returns


def _french_cash_zip(index):
    rows = "\n".join(f"{d:%Y%m%d},0.1,0.1,0.1,0.0150" for d in index)
    text = f"This file was created by using the 202607 CRSP database.\n\n,Mkt-RF,SMB,HML,RF\n{rows}\n\nCopyright\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_Factors_daily.csv", text)
    return buffer.getvalue()


def _setup(tmp_path, tracking):
    rng = np.random.default_rng(5)
    public_index = pd.bdate_range("1971-09-01", "2024-12-31")
    house_index = pd.bdate_range("2021-06-29", "2026-06-12")
    public = pd.Series(rng.normal(0.0005, 0.006, size=len(public_index)), index=public_index)
    cz = tmp_path / "cz.zip"
    cz.write_bytes(_cz_zip(public_index, public.to_numpy() * 100.0))
    overlap = public.reindex(house_index).fillna(0.0).to_numpy()
    noise = rng.normal(0, 0.006, size=len(house_index))
    arms = {"ear_monthly": pd.DataFrame({"net": tracking * overlap + np.sqrt(1 - tracking**2) * noise,
                                         "charged_cost": 0.00002, "gross": 1.0}, index=house_index)}
    cash = tmp_path / "cash.zip"
    cash.write_bytes(_french_cash_zip(public_index))
    momentum = pd.DataFrame(index=pd.bdate_range("1927-01-03", "2026-07-31"))
    momentum["reference|scale"] = 1.0
    momentum["reference|gross"] = rng.normal(0.0004, 0.0075, size=len(momentum))
    for column in ("in_house|scale", "in_house|net", "in_house|charged_cost"):
        momentum[column] = np.nan
    live = momentum.index >= "2022-09-29"
    momentum.loc[live, "in_house|scale"] = 1.0
    momentum.loc[live, "in_house|net"] = rng.normal(0.0004, 0.0075, size=int(live.sum()))
    momentum.loc[live, "in_house|charged_cost"] = 0.00002
    series = tmp_path / "momentum.csv"
    momentum.to_csv(series)
    reg = er.load_registration(er.REGISTRATION)
    reg["combination"]["momentum_series"] = str(series)
    # The fixture's weekday calendar carries holidays, which would stretch the real 9.74-year
    # window past ten; a 2016 publication keeps it unambiguously short.
    reg["combination"]["momentum_publication"] = "2016-01"
    reg["bootstrap"]["n_resamples"] = 100
    return reg, arms, cz, cash


def test_a_tracking_arm_is_read_on_all_four_criteria_and_the_combination_takes_the_haircut_prior(tmp_path):
    reg, arms, cz, cash = _setup(tmp_path, tracking=0.9)
    summary = er.evaluate(reg, arms, cz, cash)
    arm = summary["arms"]["ear_monthly"]
    assert arm["passes"]["fidelity"] and arm["publication_bar"]["prior_basis"] == "post_publication"
    assert summary["public_series"]["last"] == "2024-12-31"
    combo = summary["combination"]
    assert combo["evaluated"] and combo["parts"] == ["ear_monthly", "vol_momentum"]
    # under ten years of post-publication history: the prior is 0.4 x the whole common window's Sharpe
    assert combo["prior"]["basis"] == "haircut_full_window"
    assert combo["prior"]["sharpe"] == pytest.approx(0.4 * combo["prior"]["full_window"]["sharpe"])


def test_the_combination_is_skipped_when_the_arm_does_not_track(tmp_path):
    reg, arms, cz, cash = _setup(tmp_path, tracking=0.1)
    summary = er.evaluate(reg, arms, cz, cash)
    assert not summary["arms"]["ear_monthly"]["passes"]["fidelity"]
    assert summary["combination"]["evaluated"] is False and summary["combination_promoted"] is False


def test_promotion_outcome_reads_the_written_record(tmp_path):
    path = tmp_path / "replication.json"
    path.write_text(json.dumps({"promoted_arms": [], "combination_promoted": False,
                                "arms": {"ear_monthly": {"passes": {"fidelity": True}}},
                                "combination": {"passes": {"economics": False}}}))
    outcome = er.promotion_outcome(path)
    assert outcome["promoted_arms"] == [] and outcome["combination"] == {"economics": False}
