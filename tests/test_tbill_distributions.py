"""The T-bill ETF distribution histories (``data/distributions/SGOV.csv``, ``BIL.csv``) agree with their provenance.

Each file must load through the issuer-distribution reader the legacy import
uses (both files present, a source hash, a capture time), carry exactly the
record count and ex-date range its provenance states, and hold the zero and
out-of-order rows the provenance names, and no others.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from prism.io.events import issuer_distribution_events

DISTRIBUTIONS = Path(__file__).resolve().parents[1] / "data" / "distributions"

#: Per file: the rows with a zero amount and the rows whose pay date precedes the ex-date,
#: as each provenance file's parsing notes and anomalies describe them.
EXPECTED = {
    "SGOV": {"zero_rows": 1, "pay_before_ex": []},
    "BIL": {"zero_rows": 105, "pay_before_ex": ["2008-03-03"]},
}


@pytest.mark.parametrize("symbol", sorted(EXPECTED))
def test_the_history_loads_and_matches_its_provenance(symbol: str) -> None:
    provenance = json.loads((DISTRIBUTIONS / f"{symbol}.provenance.json").read_text(encoding="utf-8"))
    events = issuer_distribution_events(DISTRIBUTIONS, symbol)
    assert len(events) == provenance["n_records"]
    assert {event.payload["provenance_sha256"] for event in events} == {provenance["sha256"]}
    assert {event.ingested_at for event in events} == {provenance["captured_at"]}

    table = pd.read_csv(DISTRIBUTIONS / f"{symbol}.csv", dtype=str)
    assert list(table.columns) == ["ex_date", "pay_date", "amount"]
    assert table["ex_date"].is_unique and table["ex_date"].tolist() == sorted(table["ex_date"])
    assert (table["ex_date"].iloc[0], table["ex_date"].iloc[-1]) == (
        provenance["first_ex_date"],
        provenance["last_ex_date"],
    )
    amounts = table["amount"].astype(float)
    assert (amounts >= 0.0).all()
    assert int((amounts == 0.0).sum()) == EXPECTED[symbol]["zero_rows"]
    assert table.loc[table["pay_date"] < table["ex_date"], "ex_date"].tolist() == EXPECTED[symbol]["pay_before_ex"]
