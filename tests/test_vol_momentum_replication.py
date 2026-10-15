"""Synthetic-data tests for the volatility-scaled momentum replication read (research tier).

They pin what the registration relies on: the in-house overlay scales returns and
recorded costs and charges scale changes on recorded gross, scoring starts after
the first full-lookback decision, the four criteria compute as the design doc
states, and the driver writes one row and refuses a second.
"""

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest


from research.scripts import vol_momentum_replication as vm  # noqa: E402

SOURCE = {"target_vol": 0.12, "lookback_sessions": 126, "refresh_sessions": 21, "max_scale": 2.0,
          "publication": "2015-04"}
CRITERIA = {"publication_bar_t": 3.0, "publication_bar_min_years": 20.0, "post_publication_min_years": 10.0,
            "pre_publication_haircut": 0.4, "fidelity_floor": 0.75, "hurdle_window_sessions": 252,
            "sessions_per_year": 252}


def test_first_scored_index_is_after_first_full_lookback_decision():
    assert vm.first_scored_index(SOURCE) == 127


def test_overlay_scales_returns_and_costs_and_charges_changes():
    index = pd.bdate_range("2022-03-29", periods=300)
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.0003, 0.02, size=300), index=index)  # ~32% vol => scale well below 1
    gross = pd.Series(1.0, index=index)
    total = pd.Series(0.0001, index=index)
    frame = vm.scaled_in_house(returns, gross, total, SOURCE, change_cost_bps=10.0)
    assert frame.index[0] == index[127]
    s = frame["scale"].iloc[0]
    assert 0 < s < 1
    # first scored session carries the move from scale 1 to s
    assert frame["charged_cost"].iloc[0] == pytest.approx(s * 0.0001 + abs(s - 1.0) * 1.0 * 10 / 1e4)
    assert frame["net"].iloc[0] == pytest.approx(s * returns.iloc[127] - abs(s - 1.0) * 10 / 1e4)
    # within a refresh window the scale is constant and only scaled costs are charged
    assert frame["scale"].iloc[1] == s
    assert frame["charged_cost"].iloc[1] == pytest.approx(s * 0.0001)


def test_overlay_refuses_misaligned_costs():
    index = pd.bdate_range("2022-03-29", periods=200)
    returns = pd.Series(0.001, index=index)
    with pytest.raises(vm.LabError):
        vm.scaled_in_house(returns, pd.Series(1.0, index=index[1:]), pd.Series(0.0, index=index), SOURCE, 10.0)


def test_publication_bar_uses_post_publication_prior_when_long_enough():
    index = pd.bdate_range("1960-01-01", "2026-07-31")
    rng = np.random.default_rng(4)
    reference = pd.Series(rng.normal(0.0004, 0.008, size=len(index)), index=index)
    bar = vm.publication_bar(reference, SOURCE, CRITERIA)
    assert bar["prior_basis"] == "post_publication"
    assert bar["prior_sharpe"] == pytest.approx(bar["post_publication"]["sharpe"])
    short = vm.publication_bar(reference, {**SOURCE, "publication": "2020-01"}, CRITERIA)
    assert short["prior_basis"] == "haircut_full_sample"
    assert short["prior_sharpe"] == pytest.approx(0.4 * short["full"]["sharpe"])


def test_economics_arithmetic_and_hurdle():
    index = pd.bdate_range("2023-01-02", periods=500)
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({"net": rng.normal(0.0004, 0.0075, size=500), "charged_cost": 0.0001}, index=index)
    econ = vm.economics(0.9, 0.5, frame, rf_annual=0.04, target_vol=0.12, long_over_equity=0.5)
    vol = frame["net"].std(ddof=1) * np.sqrt(252)
    assert econ["cost_drag_sharpe"] == pytest.approx(0.0001 * 252 / vol)
    assert econ["figure_f3"] == pytest.approx(0.45 - econ["cost_drag_sharpe"])
    assert econ["hurdle_sharpe"] == pytest.approx(0.04 / 0.12)
    assert econ["pass"] == (econ["figure_f3"] > econ["hurdle_sharpe"])
    # The measured L/E travels with the hurdle even though, at zero cash yield, it does not move the number.
    assert econ["hurdle_inputs"]["long_over_equity"] == 0.5
    assert econ["hurdle_inputs"]["idle_fraction"] == 0.5
    assert econ["hurdle_inputs"]["cash_yield"] == 0.0 and econ["hurdle_inputs"]["vol_basis"] == "target"
    with pytest.raises(TypeError):
        vm.economics(0.9, 0.5, frame, rf_annual=0.04, target_vol=0.12)  # L/E has no default (fail loud)


def test_fidelity_floor():
    index = pd.bdate_range("2023-01-02", periods=400)
    rng = np.random.default_rng(8)
    ref = pd.Series(rng.normal(0, 0.01, size=400), index=index)
    close = ref + pd.Series(rng.normal(0, 0.003, size=400), index=index)
    far = pd.Series(rng.normal(0, 0.01, size=400), index=index)
    assert vm.fidelity(close, ref, 0.75)["pass"]
    assert not vm.fidelity(far, ref, 0.75)["pass"]


def _french_zip(column: str, index: pd.DatetimeIndex, values: np.ndarray) -> bytes:
    rows = "\n".join(f"{d:%Y%m%d},{v * 100:.4f}" for d, v in zip(index, values))
    text = f"This file was created by using the 202607 CRSP database.\n\n,{column}\n{rows}\n\nCopyright\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("x.csv", text)
    return buffer.getvalue()


def test_driver_end_to_end_on_synthetic_inputs(tmp_path, monkeypatch):
    rng = np.random.default_rng(10)
    long_index = pd.bdate_range("1985-01-02", "2026-07-31")
    mom_zip = tmp_path / "mom.zip"
    mom_zip.write_bytes(_french_zip("Mom", long_index, rng.normal(0.0004, 0.009, size=len(long_index))))
    cash_zip = tmp_path / "cash.zip"
    cash_zip.write_bytes(_french_zip("RF", long_index, np.full(len(long_index), 0.00015)))

    house_index = pd.bdate_range("2022-03-29", "2026-06-12")
    stamps = [f"{d:%Y-%m-%d} 00:00:00-04:00" for d in house_index]
    returns_csv = tmp_path / "returns.csv"
    pd.DataFrame({"daily_return": rng.normal(0.0003, 0.008, size=len(house_index))}, index=stamps).to_csv(returns_csv)
    costs_csv = tmp_path / "costs.csv"
    pd.DataFrame({"gross": 1.0, "total": 0.00002}, index=stamps).to_csv(costs_csv)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    reg = vm.load_registration(vm.REGISTRATION)
    reg["reference"].update(momentum_sha256=digest(mom_zip), cash_sha256=digest(cash_zip))
    reg["implementation"].update(returns=str(returns_csv), returns_sha256=digest(returns_csv), costs=str(costs_csv),
                                 costs_sha256=digest(costs_csv))
    reg["bootstrap"]["n_resamples"] = 100
    monkeypatch.setattr(vm, "fetch_sources", lambda sources, cache: {
        "ff_mom_daily": {"path": str(mom_zip)}, "ff3_daily": {"path": str(cash_zip)}})

    log_path = tmp_path / "log.jsonl"
    summary = vm.run(reg, tmp_path / "out", log_path, "deadbeef")
    assert set(summary) >= {"publication_bar", "fidelity", "economics", "no_contradiction", "promoted"}
    assert summary["economics"]["rf_annual"] == pytest.approx(0.00015 * 252)
    assert (tmp_path / "out" / "series.csv").exists()
    rows = vm.TrialLog(log_path).verify()
    assert [row["event"] for row in rows] == ["started", "finished"]
    with pytest.raises(vm.LabError):
        vm.run(reg, tmp_path / "out", log_path, "deadbeef")

    reg["implementation"]["returns_sha256"] = "0" * 64
    with pytest.raises(vm.LabError):
        vm.run(reg, tmp_path / "out2", tmp_path / "log2.jsonl", "deadbeef")
