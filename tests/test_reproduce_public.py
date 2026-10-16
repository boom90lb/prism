"""Tests for the uncounted public-factor reproduction (``docs/public_reproduction_design.md``).

Offline only. Layer A runs on the tracked records and must reproduce the
recorded criteria within 1e-9. Layers B and the factor-lab part run against
synthetic French- and JKP-format zips with the download replaced, and must
write nothing outside the output directory.
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.scripts import reproduce_public as rp

REPO = rp.REPO
SUMMARY = json.loads((REPO / rp.VM_RECORD / "summary.json").read_text())


def _snapshot(*roots: Path) -> dict[str, tuple[int, int]]:
    files: dict[str, tuple[int, int]] = {}
    for root in roots:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    files[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return files


def _protected_snapshot() -> dict[str, tuple[int, int]]:
    return _snapshot(REPO / "results", REPO / "trials", REPO / "data")


# --------------------------------------------------------------------- layer A on the tracked record


def test_offline_run_reproduces_the_recorded_criteria(tmp_path):
    before = _protected_snapshot()
    out = tmp_path / "out"
    assert rp.main(["--out", str(out), "--offline"]) == 0
    assert _protected_snapshot() == before
    assert sorted(path.name for path in out.iterdir()) == ["manifest.json", "report.md", "vol_momentum.json"]
    assert (out / "report.md").read_text().splitlines()[0] == rp.FIRST_LINE

    vol = json.loads((out / "vol_momentum.json").read_text())
    a = vol["layer_a"]
    assert a["match"] and a["a1_criteria"]["comparison"]["leaves_differing"] == 0
    values = a["a1_criteria"]["values"]
    for keys in (("publication_bar", "prior_sharpe"), ("fidelity", "daily_corr"), ("economics", "cost_drag_sharpe"),
                 ("economics", "figure_f3"), ("economics", "hurdle_sharpe")):
        recorded, reproduced = SUMMARY, values
        for key in keys:
            recorded, reproduced = recorded[key], reproduced[key]
        assert abs(reproduced - recorded) <= 1e-9, keys
    assert values["promoted"] is SUMMARY["promoted"] is True
    assert a["a2_in_house_rebuild"]["match"] and a["a2_in_house_rebuild"]["same_sessions"]
    assert a["a3_reference_rule"]["match"] and a["a3_reference_rule"]["checked_sessions"] > 24_000

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["layers"] == ["A"] and manifest["downloads"] == {}
    impl = rp.fl.load_registration(REPO / rp.VM_REGISTRATION)["implementation"]
    assert manifest["inputs_sha256"][impl["costs"]] == impl["costs_sha256"]
    assert manifest["inputs_sha256"][impl["returns"]] == impl["returns_sha256"]


# --------------------------------------------------------------------- refusals


@pytest.mark.parametrize("relative", ["results/x", "trials/x", "data/public/x", "results", "data/public"])
def test_out_dir_under_a_protected_path_is_refused(relative):
    target = REPO / relative
    existed = target.exists()
    with pytest.raises(rp.ReproductionError, match="writes nothing there"):
        rp.prepare_out(target)
    assert target.exists() == existed


def test_out_dir_reached_through_a_symlink_is_refused(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(REPO / "results", target_is_directory=True)
    with pytest.raises(rp.ReproductionError, match="writes nothing there"):
        rp.prepare_out(link / "x")


def test_non_empty_out_dir_is_refused(tmp_path):
    (tmp_path / "occupied").mkdir()
    (tmp_path / "occupied" / "file").write_text("x")
    with pytest.raises(rp.ReproductionError, match="not empty"):
        rp.prepare_out(tmp_path / "occupied")
    assert rp.prepare_out(tmp_path / "fresh").is_dir()


def test_offline_with_factor_lab_is_refused(tmp_path):
    with pytest.raises(rp.ReproductionError, match="cannot run --offline"):
        rp.run(tmp_path / "out", offline=True, with_factor_lab=True)
    assert not (tmp_path / "out").exists()


def test_changed_in_house_input_is_refused():
    reg = rp.fl.load_registration(REPO / rp.VM_REGISTRATION)
    reg["implementation"]["costs_sha256"] = "0" * 64
    with pytest.raises(rp.ReproductionError, match="not the recorded state"):
        rp.rebuild_in_house(reg)


def test_failed_or_empty_download_raises(tmp_path, monkeypatch):
    def broken(url):
        raise OSError("connection refused")

    monkeypatch.setattr(rp, "download", broken)
    with pytest.raises(rp.ReproductionError, match="download failed for ff_mom_daily"):
        rp.fetch({"ff_mom_daily": "https://example.invalid/x.zip"}, tmp_path / "cache")
    monkeypatch.setattr(rp, "download", lambda url: (b"", {}))
    with pytest.raises(rp.ReproductionError, match="no bytes"):
        rp.fetch({"ff_mom_daily": "https://example.invalid/x.zip"}, tmp_path / "cache")


# --------------------------------------------------------------------- comparison helpers


def test_compare_tolerates_float_noise_and_flags_everything_else():
    recorded = {"a": 1.0, "b": {"c": 3, "d": "x", "e": True, "f": None}, "g": [1.0, 2.0]}
    same = {"a": 1.0 + 5e-10, "b": {"c": 3, "d": "x", "e": np.bool_(True), "f": None}, "g": [1.0, np.float64(2.0)],
            "extra": 9}
    assert rp.comparison_block(recorded, same)["match"]
    different = {"a": 1.0 + 2e-9, "b": {"c": 4, "d": "y", "e": False}, "g": [1.0]}
    block = rp.comparison_block(recorded, different)
    assert {row["path"] for row in block["differing"]} == {"a", "b.c", "b.d", "b.e", "b.f", "g"}


def test_drift_counts_changed_added_and_removed_sessions():
    index = pd.bdate_range("2020-01-01", periods=10)
    recorded = pd.Series(np.arange(10, dtype=float), index=index)
    current = recorded.iloc[1:].copy()
    current.iloc[3] += 0.5
    added = pd.Series([1.0, 2.0], index=pd.bdate_range(index[-1] + pd.offsets.BDay(), periods=2))
    current = pd.concat([current, added])
    result = rp.drift(recorded, current)
    assert result["common_sessions"] == 9 and result["changed_sessions"] == 1
    assert result["max_abs_difference"] == pytest.approx(0.5)
    assert result["removed_sessions"] == 1 and result["added_after_recorded_last"] == 2
    assert result["added_before_recorded_first"] == 0 and result["added_inside_recorded_window"] == 0
    assert result["first_changed"] == str(index[4].date())


def test_reference_consistency_detects_a_different_rule():
    reg = rp.fl.load_registration(REPO / rp.VM_REGISTRATION)
    index = pd.bdate_range("1990-01-01", periods=1800)
    momentum = pd.Series(np.random.default_rng(3).normal(0.0003, 0.01, size=index.size), index=index)
    reference = rp.vm.scaled_reference(momentum, reg["source"], int(reg["reference"]["warmup_sessions"]))[
        ["scale", "gross"]]
    assert rp.reference_consistency(reference, reg)["match"]
    other = {**reg, "source": {**reg["source"], "target_vol": 0.10}}
    assert not rp.reference_consistency(reference, other)["match"]


# --------------------------------------------------------------------- layer B and the factor lab, synthetic


def _zip(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", text)
    return buffer.getvalue()


def _french(columns: dict[str, np.ndarray], index: pd.DatetimeIndex) -> bytes:
    header = "," + ",".join(columns)
    rows = "\n".join(f"{day:%Y%m%d}," + ",".join(f"{columns[name][i] * 100:.4f}" for name in columns)
                     for i, day in enumerate(index))
    return _zip(f"This file was created by using the 209912 CRSP database.\n\n{header}\n{rows}\n\nCopyright 2099\n")


def _jkp(names: list[str], index: pd.DatetimeIndex, rng: np.random.Generator) -> bytes:
    frame = pd.DataFrame([(day.strftime("%Y-%m-%d"), name, rng.normal(0.0002, 0.006))
                          for day in index for name in names], columns=["date", "name", "ret"])
    return _zip(frame.to_csv(index=False))


def _synthetic_payloads() -> dict[str, bytes]:
    rng = np.random.default_rng(11)
    daily = pd.bdate_range("1996-01-02", "2026-08-31")
    jkp_daily = pd.bdate_range("1996-01-02", "2025-12-31")

    def noise(scale: float = 0.008) -> np.ndarray:
        return rng.normal(0.0002, scale, size=daily.size)

    reg = rp.fl.load_registration(REPO / rp.FL_REGISTRATION)
    urls = {source_id: spec["url"] for source_id, spec in reg["sources"].items()}
    payloads = {
        urls["ff3_daily"]: _french({"Mkt-RF": noise(), "SMB": noise(), "HML": noise(),
                                    "RF": np.full(daily.size, 0.00016)}, daily),
        urls["ff5_daily"]: _french({"Mkt-RF": noise(), "SMB": noise(), "HML": noise(), "RMW": noise(),
                                    "CMA": noise(), "RF": np.full(daily.size, 0.00016)}, daily),
        urls["ff_mom_daily"]: _french({"Mom": noise(0.01)}, daily),
        urls["ff_strev_daily"]: _french({"ST_Rev": noise()}, daily),
        urls["jkp_themes_daily"]: _jkp(list(reg["jkp_themes"]["names"]), jkp_daily, rng),
        urls["jkp_resmom_daily"]: _jkp(["resff3_12_1"], jkp_daily, rng),
        urls["jkp_resmom_large_monthly"]: _jkp(["resff3_12_1"], pd.date_range("1996-01-31", "2025-12-31",
                                                                              freq="ME"), rng),
    }
    return payloads


@pytest.fixture
def synthetic_downloads(monkeypatch):
    payloads = _synthetic_payloads()
    requested: list[str] = []

    def fake(url):
        requested.append(url)
        return payloads[url], {"Last-Modified": "Thu, 01 Oct 2099 00:00:00 GMT"}

    monkeypatch.setattr(rp, "download", fake)
    original = rp.vm.no_contradiction

    def quick(frame, figure, boot, per_year=252):
        return original(frame, figure, {**boot, "n_resamples": 50}, per_year)

    monkeypatch.setattr(rp.vm, "no_contradiction", quick)
    return payloads, requested


def test_layer_b_records_hashes_and_writes_only_inside_out(tmp_path, synthetic_downloads):
    payloads, requested = synthetic_downloads
    before = _protected_snapshot()
    out = tmp_path / "out"
    result = rp.run(out, offline=False, with_factor_lab=False)
    assert _protected_snapshot() == before
    assert [path.name for path in tmp_path.iterdir()] == ["out"]
    assert sorted(path.name for path in (out / "cache").iterdir()) == ["ff3_daily.zip", "ff_mom_daily.zip"]

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["layers"] == ["A", "B"] and len(requested) == 2
    for source_id in ("ff_mom_daily", "ff3_daily"):
        entry = manifest["downloads"][source_id]
        assert entry["sha256"] == rp.hashlib.sha256(payloads[entry["url"]]).hexdigest()
        assert entry["matches_registration"] is False and entry["matches_factor_lab_record"] is False
        assert entry["last_modified"] == "Thu, 01 Oct 2099 00:00:00 GMT"
        assert entry["vintage"].startswith("This file was created by using the 209912 CRSP database")
        assert entry["last_date"] == "2026-08-31"

    b = result["vol_momentum"]["layer_b"]
    assert b["windows"]["b1_recorded_window"]["reference_last"] == "2026-07-31"
    assert b["windows"]["b2_latest"]["reference_last"] == "2026-08-31"
    rf_new = b["windows"]["b2_latest"]["rf_recomputed"]
    assert rf_new["rf_annual"] == pytest.approx(0.00016 * 252)
    assert rf_new["values"]["economics"]["hurdle_inputs"]["rf_source"].startswith("French RF of the downloaded file")
    assert b["raw_momentum_drift"]["changed_sessions"] > 0  # synthetic data differs from the record
    assert "not a registered result" in b["windows"]["b1_recorded_window"]["label"]
    assert (out / "report.md").read_text().splitlines()[0] == rp.FIRST_LINE


def test_factor_lab_part_runs_into_out_and_compares(tmp_path, synthetic_downloads):
    _, requested = synthetic_downloads
    before = _protected_snapshot()
    out = tmp_path / "out"
    result = rp.run(out, offline=False, with_factor_lab=True)
    assert _protected_snapshot() == before
    assert len(requested) == 7  # ff3 and momentum are shared by both registrations and fetched once

    lab = json.loads((out / "priors_diff.json").read_text())
    for label in ("recorded_window", "current"):
        assert (out / "factor_lab" / label / "priors.json").exists()
        assert (out / "factor_lab" / label / "fidelity.json").exists()
        window = lab["windows"][label]
        assert set(window["priors"]) == set(json.loads((REPO / rp.FL_RECORD / "priors.json").read_text())["priors"])
        assert not window["priors"]["UMD"]["match"]  # synthetic data cannot match the record
    cut = json.loads((out / "factor_lab" / "recorded_window" / "priors.json").read_text())["priors"]
    assert cut["UMD"]["full"]["last"] == "2026-07-31"
    current = json.loads((out / "factor_lab" / "current" / "priors.json").read_text())["priors"]
    assert current["UMD"]["full"]["last"] == "2026-08-31"
    hashes = result["manifest"]["inputs_sha256"]
    assert hashes[rp.LIVE_BOOK_RETURNS.as_posix()] == rp.LIVE_BOOK_RETURNS_SHA256
    assert {entry["matches_factor_lab_record"] for entry in result["manifest"]["downloads"].values()} == {False}
