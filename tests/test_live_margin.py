"""The read-only margin read (``prism.live.margin``) and the margin report (``prism.scripts.margin_report``).

Pinned here:

* the derived ratios: the cushion ``(equity - maintenance) / equity``, the two
  sides and gross over equity, and no cushion when equity is not positive;
* malformed reads are refused (N7), and a row's cushion is recomputed from its
  own numbers rather than read from its stored ratio;
* the record: one row per read keyed by its UTC timestamp, appended only when
  later than the last, with the New York date beside it;
* the report reads ``GET /v2/account`` and nothing else, prints or records the
  row, and refuses before any read when it has nowhere to record.

Nothing here opens a socket.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prism.live.alpaca import PAPER_BASE_URL, AlpacaBroker
from prism.live.lockfile import run_dir_lock
from prism.live.margin import (
    CUSHION_WARN_FLOOR,
    MARGIN_LEDGER_FILENAME,
    AccountMargin,
    MarginError,
    append_margin_row,
    margin_row,
    read_margin_rows,
    row_cushion,
)
from prism.scripts import margin_report

#: A long/short book at gross about 1 on 944,445 of equity, in Alpaca's string
#: encoding. Shaped on the documented ``GET /v2/account`` fields; not a capture
#: from a real account. Equity = cash + long + short, so the short side is negative.
ACCOUNT = {
    "account_number": "PA0TEST",
    "status": "ACTIVE",
    "currency": "USD",
    "cash": "950000.00",
    "equity": "944445.12",
    "portfolio_value": "944445.12",
    "long_market_value": "472000.00",
    "short_market_value": "-477554.88",
    "initial_margin": "474777.44",
    "maintenance_margin": "261266.46",
    "last_maintenance_margin": "259000.00",
    "last_equity": "940000.00",
    "regt_buying_power": "939335.36",
    "buying_power": "939335.36",
    "non_marginable_buying_power": "469667.68",
    "multiplier": "2",
    "sma": "0",
    "shorting_enabled": True,
    "balance_asof": "2026-10-15",
}


def _margin(**overrides) -> AccountMargin:
    values = {
        "equity": 100_000.0,
        "cash": 100_000.0,
        "long_market_value": 50_000.0,
        "short_market_value": 50_000.0,
        "initial_margin": 50_000.0,
        "maintenance_margin": 27_500.0,
        "regt_buying_power": 100_000.0,
        "buying_power": 100_000.0,
        "multiplier": 2.0,
    }
    values.update(overrides)
    return AccountMargin(**values)


READ_AT = datetime(2026, 10, 16, 1, 45, 30, tzinfo=timezone.utc)  # 21:45 ET on 2026-10-15


# ---------------------------------------------------------------------------
# The read and its ratios
# ---------------------------------------------------------------------------


def test_the_cushion_and_the_sides_are_read_over_equity() -> None:
    margin = _margin()
    # FINRA minimums at gross 1: 25% of the long side and 30% of the short side.
    assert margin.cushion == pytest.approx(0.725)
    assert margin.long_over_equity == pytest.approx(0.5)
    assert margin.short_over_equity == pytest.approx(0.5)
    assert margin.gross_over_equity == pytest.approx(1.0)
    assert margin.initial_margin_excess == pytest.approx(50_000.0)
    assert margin.maintenance_margin_excess == pytest.approx(72_500.0)


def test_equity_that_is_not_positive_has_no_cushion() -> None:
    margin = _margin(equity=-5_000.0, maintenance_margin=27_500.0)
    assert margin.cushion is None and margin.gross_over_equity is None
    assert margin.maintenance_margin_excess == pytest.approx(-32_500.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"equity": float("nan")},
        {"maintenance_margin": -1.0},
        {"short_market_value": -10.0},
        {"multiplier": 0.0},
        {"cash": "100"},
        {"sma": float("inf")},
    ],
)
def test_a_malformed_read_is_refused(overrides) -> None:
    with pytest.raises(MarginError):
        _margin(**overrides)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_a_row_carries_the_venue_numbers_the_ratios_the_floor_and_both_dates() -> None:
    row = margin_row(_margin(sma=0.0), read_at=READ_AT, mode="paper", source="alpaca")
    assert row["read_at"] == "2026-10-16T01:45:30+00:00"
    assert row["as_of"] == "2026-10-15"  # the New York date of the read
    assert row["mode"] == "paper" and row["source"] == "alpaca" and row["schema"] == 1
    assert row["equity"] == 100_000.0 and row["maintenance_margin"] == 27_500.0 and row["sma"] == 0.0
    assert row["cushion"] == pytest.approx(0.725) and row["cushion_floor"] == CUSHION_WARN_FLOOR
    assert row["last_equity"] is None  # optional venue fields stay explicit
    json.dumps(row)  # plain JSON


def test_a_naive_read_time_is_refused() -> None:
    with pytest.raises(MarginError, match="timezone-aware"):
        margin_row(_margin(), read_at=datetime(2026, 10, 16, 1, 45), mode="paper", source="alpaca")


def test_the_record_appends_one_row_per_later_read(tmp_path: Path) -> None:
    path = tmp_path / MARGIN_LEDGER_FILENAME
    first = margin_row(_margin(), read_at=READ_AT, mode="paper", source="alpaca")
    assert append_margin_row(path, first) is True
    assert append_margin_row(path, first) is False  # the same read again
    earlier = margin_row(_margin(), read_at=READ_AT.replace(hour=0), mode="paper", source="alpaca")
    assert append_margin_row(path, earlier) is False
    later = margin_row(_margin(equity=90_000.0), read_at=READ_AT.replace(day=17), mode="paper", source="alpaca")
    assert append_margin_row(path, later) is True
    assert [row["equity"] for row in read_margin_rows(path)] == [100_000.0, 90_000.0]


def test_a_rows_cushion_is_recomputed_from_its_own_numbers() -> None:
    row = margin_row(_margin(), read_at=READ_AT, mode="paper", source="alpaca")
    row["cushion"] = 0.99  # a stored ratio that disagrees with the row's numbers is not believed
    assert row_cushion(row) == pytest.approx(0.725)
    assert row_cushion({**row, "equity": 0.0}) is None
    with pytest.raises(MarginError, match="maintenance_margin"):
        row_cushion({"equity": 1.0})


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _AccountOnly:
    """A venue that answers GET /v2/account and records every request it is sent."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def request(self, method, url, headers=None, timeout=None, json=None, params=None):
        path = url[len(PAPER_BASE_URL) :]
        self.calls.append((method, path))
        if (method, path) == ("GET", "/v2/account"):
            return _Response(self.payload)
        raise AssertionError(f"the margin report sent {method} {path}")


def _factory(venue: _AccountOnly, modes: list[str]):
    def build(mode: str) -> AlpacaBroker:
        modes.append(mode)
        return AlpacaBroker("key", "secret", session=venue, sleep=lambda wait: None)

    return build


def test_the_report_reads_the_account_and_nothing_else(capsys) -> None:
    venue, modes = _AccountOnly(ACCOUNT), []
    assert margin_report.main([], broker_factory=_factory(venue, modes)) == 0
    assert venue.calls == [("GET", "/v2/account")]
    assert modes == ["paper"]
    out = capsys.readouterr().out
    assert "read-only, no orders" in out
    assert "472,000.00" in out and "477,554.88" in out  # the short side as a magnitude
    assert "72.3%" in out and "above the 25% warning floor" in out
    assert "PA0TEST" not in out  # the account number is not part of the report


def test_the_json_report_is_the_row(capsys) -> None:
    venue = _AccountOnly(ACCOUNT)
    assert margin_report.main(["--json"], broker_factory=_factory(venue, [])) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["short_market_value"] == pytest.approx(477_554.88)
    assert row["cushion"] == pytest.approx((944_445.12 - 261_266.46) / 944_445.12)
    assert row["regt_buying_power"] == pytest.approx(939_335.36) and row["multiplier"] == 2.0


def test_record_appends_the_read_to_the_run_directory(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    venue = _AccountOnly(ACCOUNT)
    assert margin_report.main(["--run-dir", str(run_dir), "--record"], broker_factory=_factory(venue, [])) == 0
    rows = read_margin_rows(run_dir / MARGIN_LEDGER_FILENAME)
    assert len(rows) == 1 and rows[0]["equity"] == pytest.approx(944_445.12)
    assert "recorded in" in capsys.readouterr().err
    # Only the record and the writer lock's own file, which the loop's lock leaves in place too; and the
    # lock was released.
    assert {p.name for p in run_dir.iterdir()} <= {MARGIN_LEDGER_FILENAME, ".lock"}
    with run_dir_lock(run_dir):
        pass


def test_record_needs_a_run_directory_and_refuses_a_missing_one_before_any_read(tmp_path: Path, capsys) -> None:
    venue, modes = _AccountOnly(ACCOUNT), []
    with pytest.raises(SystemExit):
        margin_report.main(["--record"], broker_factory=_factory(venue, modes))
    missing = tmp_path / "nope"
    assert margin_report.main(["--run-dir", str(missing), "--record"], broker_factory=_factory(venue, modes)) == 1
    assert modes == [] and venue.calls == [] and not missing.exists()


def test_a_read_the_venue_cannot_complete_exits_1_and_records_nothing(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {key: value for key, value in ACCOUNT.items() if key != "maintenance_margin"}
    venue = _AccountOnly(payload)
    assert margin_report.main(["--run-dir", str(run_dir), "--record"], broker_factory=_factory(venue, [])) == 1
    assert "maintenance_margin" in capsys.readouterr().err
    assert not (run_dir / MARGIN_LEDGER_FILENAME).exists()


def test_record_refuses_while_another_writer_holds_the_run_directory(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    venue = _AccountOnly(ACCOUNT)
    with run_dir_lock(run_dir):
        code = margin_report.main(["--run-dir", str(run_dir), "--record"], broker_factory=_factory(venue, []))
    assert code == 1
    assert "not recorded" in capsys.readouterr().err
    assert not (run_dir / MARGIN_LEDGER_FILENAME).exists()
