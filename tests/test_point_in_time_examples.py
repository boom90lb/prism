"""The point-in-time data examples, and the documentation they back.

* ``examples/point_in_time/edgar_fundamentals.py`` and ``examples/point_in_time/earnings_dates.py``
  exit 0 in a fresh interpreter with sockets disabled and without ``PRISM_SEC_USER_AGENT``,
  print their header first, and leave their working directory, their temporary directory and
  the builders' default output directories (``data/edgar``, ``data/universe``) untouched.
* What they demonstrate holds: a restatement replaces the original only from the restating
  filing's acceptance, a fact without a known acceptance is dated by the end of its filing
  date, a company with non-contiguous quarters has no trailing-twelve-month value, an 8-K/A and
  a same-week supplement are not announcements, and day 0 follows the acceptance date or the
  next session.
* The schema tables in ``docs/edgar_point_in_time.md`` and ``docs/earnings_dates.md`` equal the
  Arrow schemas of the files the examples write, and the documented fields and constants are
  the modules' own.
* ``--live`` refuses to start without a User-Agent, and with a cache directory inside a git
  checkout, before any request is made.

No test touches the network.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from prism.io import earnings_events as ee
from prism.io import edgar, sessions

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples" / "point_in_time"
EDGAR_EXAMPLE = EXAMPLES / "edgar_fundamentals.py"
EARNINGS_EXAMPLE = EXAMPLES / "earnings_dates.py"
EDGAR_DOC = REPO_ROOT / "docs" / "edgar_point_in_time.md"
EARNINGS_DOC = REPO_ROOT / "docs" / "earnings_dates.md"
HEADER = "Synthetic data; demonstrates the API only; makes no claim about any strategy."

# Runs an example with every socket call refused: python -P -c BOOTSTRAP <example>.
BOOTSTRAP = """
import runpy, socket, sys
def _refused(*args, **kwargs):
    raise OSError("network disabled by tests/test_point_in_time_examples.py")
socket.socket.connect = _refused
socket.socket.connect_ex = _refused
socket.create_connection = _refused
socket.getaddrinfo = _refused
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(module: ModuleType) -> tuple[dict[str, Any], str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = module.main([])
    return result, out.getvalue()


@pytest.fixture(scope="module")
def edgar_example() -> ModuleType:
    return _load(EDGAR_EXAMPLE, "edgar_fundamentals_example")


@pytest.fixture(scope="module")
def earnings_example() -> ModuleType:
    return _load(EARNINGS_EXAMPLE, "earnings_dates_example")


@pytest.fixture(scope="module")
def edgar_run(edgar_example: ModuleType) -> tuple[dict[str, Any], str]:
    return _run(edgar_example)


@pytest.fixture(scope="module")
def earnings_run(earnings_example: ModuleType) -> tuple[dict[str, Any], str]:
    return _run(earnings_example)


def _doc_table(doc: Path, heading: str) -> list[list[str]]:
    """The cells of the first table after ``heading`` (a line of the document), backticks removed."""
    lines = doc.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if heading in line)
    rows: list[list[str]] = []
    for line in lines[start + 1:]:
        text = line.strip()
        if text.startswith("| `"):
            rows.append([cell.strip().strip("`") for cell in text.strip("|").split("|")])
        elif rows and not text.startswith("|"):
            break
    assert rows, f"no table after {heading!r} in {doc.name}"
    return rows


def _doc_schema(doc: Path, filename: str) -> list[tuple[str, str]]:
    return [(row[0], row[1]) for row in _doc_table(doc, f"### `{filename}`")]


# --------------------------------------------------------------------- offline, isolated runs


#: Where the builders write by default (``--cache-dir``, ``--out-dir``, ``--ledger-dir``).
BUILDER_DEFAULT_DIRS = ("data/edgar", "data/universe")


def _snapshot() -> set[tuple[str, int, int]]:
    files = [p for name in BUILDER_DEFAULT_DIRS if (REPO_ROOT / name).exists()
             for p in (REPO_ROOT / name).rglob("*") if p.is_file()]
    return {(str(p.relative_to(REPO_ROOT)), p.stat().st_size, p.stat().st_mtime_ns) for p in files}


@pytest.mark.parametrize("example", [EDGAR_EXAMPLE, EARNINGS_EXAMPLE], ids=lambda p: p.stem)
def test_examples_run_offline_without_a_user_agent_and_write_nothing(example: Path, tmp_path: Path) -> None:
    work, scratch = tmp_path / "work", tmp_path / "tmp"
    work.mkdir()
    scratch.mkdir()
    env = {key: value for key, value in os.environ.items() if key != edgar.USER_AGENT_ENV}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])
    env["TMPDIR"] = str(scratch)
    data_before = _snapshot()

    proc = subprocess.run([sys.executable, "-P", "-c", BOOTSTRAP, str(example)], cwd=work, env=env,
                          capture_output=True, text=True, timeout=180)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[0] == HEADER
    assert list(work.iterdir()) == []
    assert list(scratch.iterdir()) == []
    assert _snapshot() == data_before


# --------------------------------------------------------------------- EDGAR fundamentals


def test_edgar_example_shows_the_restatement_only_after_its_acceptance(edgar_example, edgar_run) -> None:
    result, out = edgar_run
    assert out.splitlines()[0] == edgar_example.HEADER
    for day, expected in edgar_example.EXPECTED_AS_OF.items():
        got = result["as_of"][day]
        assert all((math.isnan(g) and math.isnan(e)) or g == e for g, e in zip(got, expected)), (day, got)

    facts = result["facts"]
    restated = facts[(facts["concept"] == "Assets") & (facts["end"] == pd.Timestamp("2023-12-31"))
                     & (facts["cik"] == edgar_example.CIK_A)].set_index("val")
    assert set(restated.index) == {1080.0, 1100.0}  # the original and the restatement are both kept
    accepted = restated.loc[1080.0, "available_at"]
    assert edgar.session_close("2024-04-15") < accepted <= edgar.session_close("2024-04-16")


def test_edgar_example_dates_facts_by_acceptance_else_by_the_end_of_the_filing_date(edgar_run) -> None:
    facts, filings = edgar_run[0]["facts"], edgar_run[0]["filings"]
    fallback = facts[facts["availability_basis"] == "filed_date_end"]
    assert len(fallback) == 2 and not fallback["accn"].isin(filings["accn"]).any()
    assert (fallback["available_at"] == pd.Timestamp("2023-02-09 23:59:59", tz=edgar.NEW_YORK)).all()
    dated = facts[facts["availability_basis"] == "acceptance"]
    assert (dated["available_at"] == dated["accn"].map(filings.set_index("accn")["acceptance"])).all()


def test_edgar_example_refuses_ttm_for_non_contiguous_quarters(edgar_example, edgar_run) -> None:
    result, _ = edgar_run
    ttm = result["ttm"]
    assert list(ttm["cik"]) == [edgar_example.CIK_A]
    assert float(ttm["ttm"].iloc[0]) == 380.0 and int(ttm["derived_quarters"].iloc[0]) == 2
    quarters = result["quarters"]
    assert len(quarters[quarters["cik"] == edgar_example.CIK_B]) == 4  # four quarters, but Q2 2023 is missing


def test_edgar_example_builds_the_store_with_every_documented_file(edgar_run) -> None:
    result, _ = edgar_run
    files = set(result["files"])
    assert {"facts.parquet", "filings.parquet", "ticker_cik.parquet", "manifest.json",
            "edgar_coverage_2024-06-03.json"} <= files
    assert "raw/companyfacts/CIK0009900001.json.gz" in files
    assert "raw/companyfacts/CIK0009900001.meta.json" in files
    assert not [name for name in files if name.endswith(".partial")]
    ledger = result["ledger"]
    assert ledger["status_counts"] == {"accepted": 3, "pre_xbrl": 1, "unresolved": 1}
    assert [row["ticker"] for row in ledger["not_accepted"]] == ["EXLOST"]
    assert list(result["ticker_cik"]["source"]) == ["sec_current_ticker", "sec_current_ticker", "review_decision"]


@pytest.mark.parametrize("filename", ["facts.parquet", "filings.parquet", "ticker_cik.parquet"])
def test_edgar_doc_schema_equals_the_written_file(edgar_run, filename: str) -> None:
    assert _doc_schema(EDGAR_DOC, filename) == edgar_run[0]["schemas"][filename]


def test_edgar_doc_lists_every_ledger_and_manifest_key(edgar_run) -> None:
    result, _ = edgar_run
    documented = [row[0] for row in _doc_table(EDGAR_DOC, "**The coverage ledger**")]
    assert sorted(documented) == sorted(result["ledger"])
    manifest_line = next(line for line in EDGAR_DOC.read_text().split("\n- ") if "`manifest.json`** has" in line)
    assert all(f"`{key}`" in manifest_line for key in result["manifest_keys"])


def test_edgar_doc_field_table_is_the_module_table() -> None:
    documented = {row[0]: tuple(name.strip() for name in row[1].split(","))
                  for row in _doc_table(EDGAR_DOC, "## Canonical fields")}
    assert documented == edgar.FIELDS


# --------------------------------------------------------------------- earnings dates


def test_earnings_example_dates_day_zero_from_the_acceptance(earnings_example, earnings_run) -> None:
    result, out = earnings_run
    assert out.splitlines()[0] == earnings_example.HEADER
    dates = result["dates"]
    assert {row.accn: str(row.d0.date()) for row in dates.itertuples()} == earnings_example.EXPECTED_D0
    assert not sessions.is_session(pd.Timestamp("2024-03-29").date())  # Good Friday: day 0 moves to Monday
    releases = result["releases"]
    amendment = releases.loc[releases["form"] == "8-K/A", "accn"]
    assert len(amendment) == 1 and amendment.iloc[0] not in set(dates["accn"])
    assert "0009900001-24-000005" in set(releases["accn"])  # the supplement is a release ...
    assert "0009900001-24-000005" not in set(dates["accn"])  # ... but not a separate announcement
    assert "0009900001-24-000009" not in set(releases["accn"])  # an 8-K without item 2.02 is not a release


def test_earnings_example_signal_date_is_the_session_after_day_zero(earnings_example, earnings_run) -> None:
    result, _ = earnings_run
    events = result["events_frame"]
    for row in events.itertuples():
        assert row.signal_date.date() == sessions.next_session(row.d0.date())
    panel = result["panel"]
    for row in events.itertuples():
        rows = panel[panel["sid"] == row.sid]
        before = rows[(rows["date"] < row.signal_date) & (rows["accn"] == row.accn)]
        assert before.empty  # no row shows an announcement return before its signal date
        on = rows[rows["date"] == row.signal_date].iloc[0]
        assert on["accn"] == row.accn and on["age_sessions"] == 0 and on["ear"] == row.ear
    assert result["days_to_expected"] == earnings_example.EXPECTED_DAYS_TO_EXPECTED


@pytest.mark.parametrize("filename", ["events.parquet", "panel.parquet"])
def test_earnings_doc_schema_equals_the_written_file(earnings_run, filename: str) -> None:
    assert _doc_schema(EARNINGS_DOC, filename) == earnings_run[0]["schemas"][filename]


def test_earnings_doc_states_the_module_constants() -> None:
    text = " ".join(EARNINGS_DOC.read_text().split())
    stated = {
        "WINDOW": ((-2, 1), "`WINDOW = (-2, 1)`"),
        "MIN_GAP_SESSIONS": (5, "**within 5 sessions** (`MIN_GAP_SESSIONS`)"),
        "MIN_MARKET_NAMES": (100, "fewer than 100 reporting members (`MIN_MARKET_NAMES`)"),
        "ANNIVERSARY_DAYS": (364, "364 days later (`ANNIVERSARY_DAYS`)"),
        "REALIZED_WITHIN_DAYS": (30, "no later than 30 days after the latest known release (`REALIZED_WITHIN_DAYS`)"),
        "OVERDUE_DAYS": (5, "more than 5 days overdue (`OVERDUE_DAYS`)"),
        "DUPLICATE_DAYS": (7, "within 7 calendar days of the last one kept is dropped (`DUPLICATE_DAYS`)"),
    }
    for name, (value, phrase) in stated.items():
        assert getattr(ee, name) == value, name
        assert phrase in text, name


# --------------------------------------------------------------------- live-mode refusals


@pytest.mark.parametrize("example", ["edgar_example", "earnings_example"])
def test_live_mode_refuses_without_a_user_agent(example: str, request, monkeypatch, tmp_path: Path) -> None:
    module = request.getfixturevalue(example)
    monkeypatch.delenv(edgar.USER_AGENT_ENV, raising=False)
    with pytest.raises(edgar.EdgarError, match=edgar.USER_AGENT_ENV):
        module.main(["--live", "--cik", "320193", "--cache-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("example", ["edgar_example", "earnings_example"])
def test_live_mode_refuses_a_cache_inside_a_checkout(example: str, request, monkeypatch) -> None:
    module = request.getfixturevalue(example)
    monkeypatch.setenv(edgar.USER_AGENT_ENV, "Prism tests nobody@example.invalid")
    inside = REPO_ROOT / "data" / "edgar-example-cache-that-must-not-exist"
    with pytest.raises(SystemExit, match="inside the git checkout"):
        module.main(["--live", "--cik", "320193", "--cache-dir", str(inside)])
    assert not inside.exists()


@pytest.mark.parametrize("example", ["edgar_example", "earnings_example"])
def test_live_mode_needs_a_cik_and_a_cache_dir(example: str, request, capsys) -> None:
    module = request.getfixturevalue(example)
    with pytest.raises(SystemExit):
        module.main(["--live"])
    assert "--live needs --cik and --cache-dir" in capsys.readouterr().err
