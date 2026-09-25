"""End-to-end offline test of the announcement panel builder (prism.scripts.build_earnings_panel)."""

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prism.io import earnings_events as ee
from prism.scripts import build_earnings_panel as bep
from prism.scripts.build_fundamentals import FILINGS_SCHEMA

NY = "America/New_York"


def _write_inputs(tmp_path):
    dataset, edgar_dir, out = tmp_path / "dataset", tmp_path / "edgar", tmp_path / "out"
    (dataset / "bars").mkdir(parents=True)
    edgar_dir.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({"id": "d1", "as_of": "2024-02-29T00:00:00-05:00"}))
    sids = ["XYZ", "XYZB", "MKT"]
    pd.DataFrame({"sid": sids, "valid_from": [None] * 3, "valid_to": [pd.Timestamp("2026-06-16")] * 3}).to_parquet(
        dataset / "membership.parquet")
    pd.DataFrame({"sid": sids, "ticker": ["XYZ", "XYZ.B", "MKT"]}).to_parquet(dataset / "listings.parquet")
    days = pd.bdate_range("2024-01-02", "2024-02-29")
    xyz = np.full(len(days), 10.0)
    jump = days.get_loc(pd.Timestamp("2024-01-11"))  # the session after an after-close release on 01-10
    xyz[jump:] = 11.0
    for sid, close, volume in (("XYZ", xyz, 2_000.0), ("XYZB", xyz, 500.0), ("MKT", np.full(len(days), 50.0), 1e6)):
        pd.DataFrame({"ts": days.tz_localize(NY), "close": close, "volume": volume}).to_parquet(
            dataset / "bars" / f"{sid}_1d.parquet")
    filings = pd.DataFrame({
        "cik": [7, 7, 7], "accn": ["q4", "q4a", "10k"], "form": ["8-K", "8-K/A", "10-K"],
        "filing_date": pd.to_datetime(["2024-01-10", "2024-01-12", "2024-02-20"]),
        "acceptance": pd.to_datetime(["2024-01-10 16:05", "2024-01-12 09:00", "2024-02-20 16:30"]).tz_localize(NY),
        "report_date": pd.to_datetime(["2023-12-31"] * 3), "items": ["2.02,9.01", "2.02", ""]})
    pq.write_table(pa.Table.from_pandas(filings, schema=FILINGS_SCHEMA, preserve_index=False),
                   edgar_dir / "filings.parquet")
    pd.DataFrame({"ticker": ["XYZ", "XYZ.B"], "start": [pd.Timestamp("2020-01-01")] * 2,
                  "end": [pd.Timestamp("2026-06-16")] * 2, "cik": [7, 7]}).to_parquet(edgar_dir / "ticker_cik.parquet")
    (edgar_dir / "manifest.json").write_text("{}")
    return dataset, edgar_dir, out


def test_build_writes_a_hashed_deduplicated_announcement_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(ee, "MIN_MARKET_NAMES", 1)
    dataset, edgar_dir, out = _write_inputs(tmp_path)
    freeze = bep.build(dataset, edgar_dir, out)
    events = pd.read_parquet(out / "events.parquet")
    assert list(events["accn"].unique()) == ["q4"]  # the amendment is not an event
    panel = pd.read_parquet(out / "panel.parquet")
    assert set(panel["sid"]) == {"XYZ"}  # two share classes, one company: the more liquid one is kept
    row = panel.set_index("date").loc[pd.Timestamp("2024-01-11")]
    # a 10% jump on 01-11 against an equal-weighted market of (10% + 10% + 0%) / 3
    assert row["ear"] == pytest.approx(0.10 - 0.20 / 3)
    assert row["age_sessions"] == 0 and row["d0"] == pd.Timestamp("2024-01-10")
    assert pd.Timestamp("2024-01-10") not in set(panel["date"])
    assert freeze["panel"]["sha256"] == bep.sha256_file(out / "panel.parquet")
    assert set(freeze["inputs"]) == {"filings", "ticker_cik", "edgar_manifest"}
    assert not list(out.glob("*.partial"))
