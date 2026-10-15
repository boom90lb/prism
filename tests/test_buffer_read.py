"""Synthetic-data tests for the rank-buffer reads (research tier)."""

import json

import numpy as np
import pandas as pd
import pytest


from research.scripts import buffer_read as br  # noqa: E402


def _read_dir(root, arm, returns, turnover, cost):
    root.mkdir(parents=True, exist_ok=True)
    (root / "returns").mkdir(exist_ok=True)
    index = pd.bdate_range("2022-01-03", periods=len(returns)).tz_localize("America/New_York")
    pd.Series(returns, index=index, name="daily_return").to_csv(root / "returns" / f"{arm}.csv")
    (root / "read.json").write_text(json.dumps({"arms": {arm: {
        "net_sharpe_annualized": 0.3, "gross_sharpe_annualized": 0.5, "turnover_total": turnover,
        "charged_cost_total": cost, "n_obs": len(returns)}}}))
    return root


def test_the_read_pairs_an_arm_with_its_counterpart_and_annualizes_turnover(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    base = rng.normal(0.0004, 0.005, 504)
    buffered = base + rng.normal(0.00005, 0.001, 504)
    monkeypatch.setattr(br, "BOOTSTRAP", dict(br.BOOTSTRAP, n_resamples=200))
    got = br.read_arm("arm_b", _read_dir(tmp_path / "b", "arm_b", buffered, 20.0, 0.02), 1,
                      _read_dir(tmp_path / "u", "arm_u", base, 60.0, 0.06), "arm_u")
    assert got["buffered"]["turnover_per_year"] == pytest.approx(10.0)
    assert got["unbuffered"]["turnover_per_year"] == pytest.approx(30.0)
    assert got["paired_difference"]["n_common"] == 504
    assert got["kill"]["killed"] is False


def test_a_losing_arm_is_killed(tmp_path, monkeypatch):
    rng = np.random.default_rng(8)
    monkeypatch.setattr(br, "BOOTSTRAP", dict(br.BOOTSTRAP, n_resamples=200))
    losing = rng.normal(-0.004, 0.01, 1200)
    got = br.read_arm("arm_b", _read_dir(tmp_path / "b", "arm_b", losing, 1.0, 0.01), 1,
                      _read_dir(tmp_path / "u", "arm_u", rng.normal(0, 0.01, 1200), 1.0, 0.01), "arm_u")
    assert got["kill"]["killed"] is True
