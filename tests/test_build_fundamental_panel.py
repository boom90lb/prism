"""End-to-end offline test of the characteristic panel builder (prism.scripts.build_fundamental_panel)."""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prism.scripts import build_fundamental_panel as bfp
from prism.scripts.build_fundamentals import FACTS_SCHEMA

NY = "America/New_York"


def _fact(concept, end, val, accepted, start=None, taxonomy="us-gaap", unit="USD"):
    return {"cik": 7, "taxonomy": taxonomy, "concept": concept, "unit": unit,
            "start": pd.Timestamp(start) if start else pd.NaT, "end": pd.Timestamp(end), "val": float(val),
            "accn": f"a-{accepted[:10]}", "fy": 2023, "fp": "FY", "form": "10-K", "filed": pd.Timestamp(accepted[:10]),
            "frame": None, "available_at": pd.Timestamp(accepted, tz=NY), "availability_basis": "acceptance"}


def _write_inputs(tmp_path):
    dataset, edgar_dir, out = tmp_path / "dataset", tmp_path / "edgar", tmp_path / "out"
    (dataset / "bars").mkdir(parents=True)
    edgar_dir.mkdir()
    out.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({"id": "d1", "as_of": "2024-06-14T00:00:00-04:00"}))
    pd.DataFrame({"sid": ["XYZ", "XYZB"], "valid_from": [None, None],
                  "valid_to": [pd.Timestamp("2026-06-16"), pd.Timestamp("2026-06-16")]}).to_parquet(
        dataset / "membership.parquet")
    pd.DataFrame({"sid": ["XYZ", "XYZB"], "ticker": ["XYZ", "XYZ.B"]}).to_parquet(dataset / "listings.parquet")
    days = pd.bdate_range("2024-02-05", "2024-02-16").tz_localize(NY)
    for sid, volume in (("XYZ", 2_000.0), ("XYZB", 500.0)):
        pd.DataFrame({"ts": days, "close": 10.0, "volume": volume}).to_parquet(dataset / "bars" / f"{sid}_1d.parquet")
    k = "2024-02-09 08:00"
    facts = pd.DataFrame([
        _fact("Assets", "2022-12-31", 1000, k), _fact("Assets", "2023-12-31", 1200, k),
        _fact("StockholdersEquity", "2023-12-31", 500, k),
        _fact("NetIncomeLoss", "2023-12-31", 120, k, start="2023-01-01"),
        _fact("NetCashProvidedByUsedInOperatingActivities", "2023-12-31", 150, k, start="2023-01-01"),
        _fact("GrossProfit", "2023-12-31", 360, k, start="2023-01-01"),
        _fact("EntityCommonStockSharesOutstanding", "2024-01-31", 420, k, taxonomy="dei", unit="shares"),
    ])
    pq.write_table(pa.Table.from_pandas(facts[FACTS_SCHEMA.names], schema=FACTS_SCHEMA, preserve_index=False),
                   edgar_dir / "facts.parquet")
    pd.DataFrame({"ticker": ["XYZ", "XYZ.B"], "start": [pd.Timestamp("2020-01-01")] * 2,
                  "end": [pd.Timestamp("2026-06-16")] * 2, "cik": [7, 7]}).to_parquet(edgar_dir / "ticker_cik.parquet")
    (edgar_dir / "manifest.json").write_text("{}")
    pd.DataFrame({"symbol": ["XYZ"], "type": ["forward_split"], "ex_date": [pd.Timestamp("2024-05-01")],
                  "old_rate": [1.0], "new_rate": [2.0], "ratio": [2.0]}).to_parquet(out / "splits.parquet")
    return dataset, edgar_dir, out


def test_build_writes_a_hashed_deduplicated_panel(tmp_path):
    dataset, edgar_dir, out = _write_inputs(tmp_path)
    freeze = bfp.build(dataset, edgar_dir, out, pd.Timestamp("2024-01-01"), fetch=False)
    panel = pd.read_parquet(out / "panel.parquet")
    assert set(panel["sid"]) == {"XYZ"}  # the two share classes map to one company; the more liquid one is kept
    day = panel.set_index("date").loc[pd.Timestamp("2024-02-12")]
    # 420 shares counted 2024-01-31, carried by the 2-for-1 split (2024-05-01) onto the 2024-06-14 price basis
    assert day["me"] == pytest.approx(840 * 10.0)
    assert day["be_me"] == pytest.approx(500 / 8400)
    assert day["gp_at"] == pytest.approx(360 / 1200)
    assert pd.isna(panel.set_index("date").loc[pd.Timestamp("2024-02-08"), "be"])  # before the 10-K's acceptance
    assert freeze["panel"]["sha256"] == bfp.sha256_file(out / "panel.parquet")
    assert set(freeze["inputs"]) == {"facts", "ticker_cik", "edgar_manifest", "splits"}
    assert not list(out.glob("*.partial"))


def test_build_refuses_without_split_history(tmp_path):
    dataset, edgar_dir, out = _write_inputs(tmp_path)
    (out / "splits.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        bfp.build(dataset, edgar_dir, out, pd.Timestamp("2024-01-01"), fetch=False)
