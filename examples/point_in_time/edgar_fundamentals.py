"""Point-in-time company fundamentals from SEC EDGAR XBRL, on made-up filings.

Synthetic data; demonstrates the API only; makes no claim about any strategy.

Run it from a checkout, with the package installed (``uv sync``) or with ``PYTHONPATH=src``::

    python examples/point_in_time/edgar_fundamentals.py

By default it is offline: the company-facts and filing-history payloads are made up and
served by a stand-in for the HTTP transport, so it needs no network and no
``PRISM_SEC_USER_AGENT``. It writes only inside a temporary directory, which it deletes.
The companies, tickers, CIKs, accession numbers and amounts are invented; any resemblance
to a real filer is accidental. ``docs/edgar_point_in_time.md`` explains the rules and
documents every column printed here.

Part 1 runs the library functions of ``prism.io.edgar`` on the payloads:

* a fact becomes usable at the close of the first session after its filing's EDGAR
  acceptance time, so a 10-K/A accepted at 16:20 changes nothing at that day's 16:00 close;
* a restated value replaces the original only from the restating filing's acceptance, and
  the original stays in the data as its own row;
* a fact whose filing is missing from the filing history falls back to 23:59:59 New York
  time on its filing date;
* quarters are derived from year-to-date differences, and a company whose latest four
  quarters are not contiguous gets no trailing-twelve-month value.

Part 2 runs the store builder, ``prism.scripts.build_fundamentals.build``, on the same
payloads through the real rate-limited, caching client, and prints the schema of every file
it writes, then checks that the stored facts give the same point-in-time answer.

To look at one real company instead (this fetches from SEC and caches what it fetches)::

    PRISM_SEC_USER_AGENT="Your Name you@example.com" \\
        python examples/point_in_time/edgar_fundamentals.py --live --cik 320193 \\
        --cache-dir ~/edgar-cache --as-of 2025-06-30

SEC's fair-access policy asks for a User-Agent naming you with a contact e-mail address,
and the client refuses to start without one. ``--cache-dir`` must be outside any git
checkout, so a live run never writes into this repository.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

from prism.io import edgar, sessions
from prism.scripts import build_fundamentals

HEADER = "Synthetic data; demonstrates the API only; makes no claim about any strategy."

#: Placeholder User-Agent for the offline transport. Nothing is sent anywhere.
OFFLINE_AGENT = "Prism offline example nobody@example.invalid"

CIK_A, CIK_B, CIK_GONE = 9900001, 9900002, 9900003

#: The sessions at whose 16:00 New York close Part 1 asks what was known.
AS_OF_DAYS = ("2024-02-07", "2024-02-08", "2024-04-15", "2024-04-16")
#: Expected answers for Example A Corp: FY2023 revenue, latest total assets, trailing-twelve-month revenue.
EXPECTED_AS_OF = {
    "2024-02-07": (math.nan, 1000.0, math.nan),  # 10-K not yet filed; three 2023 quarters only
    "2024-02-08": (400.0, 1100.0, 400.0),  # 10-K accepted 07:30 that morning
    "2024-04-15": (400.0, 1100.0, 400.0),  # 10-K/A accepted 16:20, after the close
    "2024-04-16": (380.0, 1080.0, 380.0),  # the restatement is in
}
TTM_DAY = "2024-06-03"
STORE_ASOF = "2024-06-03"


# --------------------------------------------------------------------- made-up payloads


def _fact(start: str | None, end: str, val: float, accn: str, fy: int, fp: str, form: str,
          filed: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start is not None:
        entry["start"] = start
    return entry


def _submissions(cik: int, name: str, rows: list[tuple[str, str, str, str, str, str]]) -> dict[str, Any]:
    """A filing-history payload in the shape of ``data.sec.gov/submissions/CIK##########.json``.

    Each row is (accession number, form, filing date, acceptance time in UTC, report date, 8-K items).
    """
    columns = ("accessionNumber", "form", "filingDate", "acceptanceDateTime", "reportDate", "items")
    recent = {column: [row[i] for row in rows] for i, column in enumerate(columns)}
    return {"cik": str(cik), "name": name, "filings": {"recent": recent, "files": []}}


def payloads() -> dict[str, dict[str, Any]]:
    """Company facts and filing histories for three made-up filers, keyed by SEC URL."""
    a10k, a10ka, a_old = "0009900001-24-000006", "0009900001-24-000015", "0009900001-23-000003"
    facts_a = {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                # In a filing that is not in the history below: dated by the fallback rule.
                _fact("2022-01-01", "2022-12-31", 360.0, a_old, 2022, "FY", "10-K", "2023-02-09"),
                _fact("2023-01-01", "2023-03-31", 90.0, "0009900001-23-000011", 2023, "Q1", "10-Q", "2023-05-04"),
                _fact("2023-04-01", "2023-06-30", 100.0, "0009900001-23-000021", 2023, "Q2", "10-Q", "2023-08-03"),
                _fact("2023-01-01", "2023-06-30", 190.0, "0009900001-23-000021", 2023, "Q2", "10-Q", "2023-08-03"),
                # The Q3 10-Q reports revenue year to date only, as cash-flow statements always do.
                _fact("2023-01-01", "2023-09-30", 290.0, "0009900001-23-000031", 2023, "Q3", "10-Q", "2023-11-02"),
                _fact("2023-01-01", "2023-12-31", 400.0, a10k, 2023, "FY", "10-K", "2024-02-08"),
                _fact("2023-01-01", "2023-12-31", 380.0, a10ka, 2023, "FY", "10-K/A", "2024-04-15"),  # restated
            ]}},
            "Assets": {"units": {"USD": [
                _fact(None, "2022-12-31", 1000.0, a_old, 2022, "FY", "10-K", "2023-02-09"),
                _fact(None, "2023-12-31", 1100.0, a10k, 2023, "FY", "10-K", "2024-02-08"),
                _fact(None, "2023-12-31", 1080.0, a10ka, 2023, "FY", "10-K/A", "2024-04-15"),  # restated
            ]}},
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {"units": {"shares": [
                _fact(None, "2024-01-31", 50.0, a10k, 2023, "FY", "10-K", "2024-02-08"),
            ]}},
        },
    }
    filings_a = _submissions(CIK_A, "Example A Corp", [
        ("0009900001-23-000011", "10-Q", "2023-05-04", "2023-05-04T20:15:00.000Z", "2023-03-31", ""),
        ("0009900001-23-000021", "10-Q", "2023-08-03", "2023-08-03T20:10:00.000Z", "2023-06-30", ""),
        ("0009900001-23-000031", "10-Q", "2023-11-02", "2023-11-02T20:05:00.000Z", "2023-09-30", ""),
        ("0009900001-24-000002", "8-K", "2024-01-30", "2024-01-30T21:30:00.000Z", "2024-01-30", "2.02,9.01"),
        (a10k, "10-K", "2024-02-08", "2024-02-08T12:30:00.000Z", "2023-12-31", ""),  # 07:30 New York
        (a10ka, "10-K/A", "2024-04-15", "2024-04-15T20:20:00.000Z", "2023-12-31", ""),  # 16:20 New York
    ])
    # Example B files revenue under the older ``Revenues`` concept and has no Q2 2023 value,
    # so its latest four quarters are not contiguous.
    facts_b = {"us-gaap": {
        "Revenues": {"units": {"USD": [
            _fact("2023-01-01", "2023-03-31", 50.0, "0009900002-23-000012", 2023, "Q1", "10-Q", "2023-05-10"),
            _fact("2023-07-01", "2023-09-30", 55.0, "0009900002-23-000032", 2023, "Q3", "10-Q", "2023-11-09"),
            _fact("2023-10-01", "2023-12-31", 60.0, "0009900002-24-000007", 2023, "FY", "10-K", "2024-02-20"),
            _fact("2024-01-01", "2024-03-31", 58.0, "0009900002-24-000020", 2024, "Q1", "10-Q", "2024-05-09"),
        ]}},
        "Assets": {"units": {"USD": [
            _fact(None, "2023-12-31", 500.0, "0009900002-24-000007", 2023, "FY", "10-K", "2024-02-20"),
        ]}},
    }}
    filings_b = _submissions(CIK_B, "Example B Inc", [
        ("0009900002-23-000012", "10-Q", "2023-05-10", "2023-05-10T20:30:00.000Z", "2023-03-31", ""),
        ("0009900002-23-000032", "10-Q", "2023-11-09", "2023-11-09T21:30:00.000Z", "2023-09-30", ""),
        ("0009900002-24-000007", "10-K", "2024-02-20", "2024-02-20T21:30:00.000Z", "2023-12-31", ""),
        ("0009900002-24-000020", "10-Q", "2024-05-09", "2024-05-09T20:30:00.000Z", "2024-03-31", ""),
    ])
    # A filer that stopped reporting in 2016, absent from SEC's current ticker map.
    facts_gone = {"us-gaap": {"Assets": {"units": {"USD": [
        _fact(None, "2013-12-31", 700.0, "0009900003-14-000004", 2013, "FY", "10-K", "2014-03-03"),
    ]}}}}
    filings_gone = _submissions(CIK_GONE, "Example Gone Co", [
        ("0009900003-14-000004", "10-K", "2014-03-03", "2014-03-03T21:10:00.000Z", "2013-12-31", ""),
    ])
    tickers = {"0": {"cik_str": CIK_A, "ticker": "EXA", "title": "Example A Corp"},
               "1": {"cik_str": CIK_B, "ticker": "EXB", "title": "Example B Inc"}}
    out = {edgar.TICKERS_URL: tickers}
    for cik, name, facts, history in ((CIK_A, "Example A Corp", facts_a, filings_a),
                                      (CIK_B, "Example B Inc", facts_b, filings_b),
                                      (CIK_GONE, "Example Gone Co", facts_gone, filings_gone)):
        out[edgar.COMPANYFACTS_URL.format(cik=cik)] = {"cik": cik, "entityName": name, "facts": facts}
        out[edgar.SUBMISSIONS_URL.format(cik=cik)] = history
    return out


def offline_transport(served: Mapping[str, Mapping[str, Any]]):
    """A stand-in for the HTTP transport: serves the made-up payloads, refuses anything else."""
    def transport(url: str, headers: Mapping[str, str], timeout: float) -> bytes:
        if url not in served:
            raise LookupError(f"the offline example has no payload for {url}")
        return json.dumps(served[url]).encode()
    return transport


# --------------------------------------------------------------------- part 1: the rules


def _close(day: str) -> pd.Timestamp:
    return edgar.session_close(pd.Timestamp(day))


def view_at(facts: pd.DataFrame, day: str) -> tuple[float, float, float]:
    """Example A's FY2023 revenue, latest total assets and TTM revenue, as known at ``day``'s close."""
    pit = edgar.point_in_time(facts, _close(day))
    revenue = edgar.field_facts(pit, "revenue")
    fy = revenue[(revenue["cik"] == CIK_A) & (revenue["kind"] == "annual")
                 & (revenue["end"] == pd.Timestamp("2023-12-31"))]
    assets = edgar.latest_instant(pit, "assets")
    assets = assets[assets["cik"] == CIK_A]
    ttm = edgar.trailing_twelve_months(pit, "revenue")
    ttm = ttm[ttm["cik"] == CIK_A]
    return (float(fy["val"].iloc[0]) if len(fy) else math.nan,
            float(assets["value"].iloc[0]) if len(assets) else math.nan,
            float(ttm["ttm"].iloc[0]) if len(ttm) else math.nan)


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def _dtypes(frame: pd.DataFrame) -> str:
    return "\n".join(f"  {name:<20} {dtype}" for name, dtype in frame.dtypes.items())


def part1(served: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    print("\nPart 1. The point-in-time rules, on two made-up filers\n")
    concepts = edgar.concept_names()
    facts = pd.concat([edgar.companyfacts_frame(served[edgar.COMPANYFACTS_URL.format(cik=cik)], concepts)
                       for cik in (CIK_A, CIK_B)], ignore_index=True)
    filings = pd.concat([edgar.submissions_filings(served[edgar.SUBMISSIONS_URL.format(cik=cik)])
                         for cik in (CIK_A, CIK_B)], ignore_index=True)
    facts = edgar.attach_availability(facts, filings)
    print(f"companyfacts_frame + attach_availability: {len(facts)} rows, one per fact per filing")
    print(_dtypes(facts))
    print(f"\nsubmissions_filings: {len(filings)} rows, acceptance converted from UTC to New York time")
    print(_dtypes(filings))

    print("\nWhen each of Example A's facts became usable (available_at, availability_basis):")
    field_of = {alias.split(":", 1)[-1]: name for name, aliases in edgar.FIELDS.items() for alias in aliases}
    shown = facts[facts["cik"] == CIK_A].sort_values(["available_at", "concept"])
    for row in shown.itertuples():
        period = f"{row.start.date()}..{row.end.date()}" if pd.notna(row.start) else f"at {row.end.date()}"
        print(f"  {field_of[row.concept]:<19} {period:<23} {row.val:>5.0f}  {row.form:<7} "
              f"{row.available_at:%Y-%m-%d %H:%M %Z}  {row.availability_basis}")
    fallback = facts[facts["availability_basis"] == "filed_date_end"]
    assert set(fallback["accn"]) == {"0009900001-23-000003"}, fallback
    assert str(fallback["available_at"].iloc[0]) == "2023-02-09 23:59:59-05:00"

    print("\nExample A as known at each session's 16:00 close (point_in_time):")
    print(f"  {'session':<12} {'FY2023 revenue':>15} {'total assets':>13} {'TTM revenue':>12}")
    as_of_rows = {}
    for day in AS_OF_DAYS:
        got = view_at(facts, day)
        as_of_rows[day] = got
        print(f"  {day:<12} " + " ".join(f"{v:>{w}.0f}" if not math.isnan(v) else f"{'-':>{w}}"
                                         for v, w in zip(got, (15, 13, 12))))
        assert all(_same(g, e) for g, e in zip(got, EXPECTED_AS_OF[day])), (day, got, EXPECTED_AS_OF[day])
    print("  The 10-K/A was accepted at 16:20 on 2024-04-15, so the restated values appear at the next close.")
    restated = facts[(facts["cik"] == CIK_A) & (facts["concept"] == "Assets")
                     & (facts["end"] == pd.Timestamp("2023-12-31"))]
    assert sorted(restated["val"]) == [1080.0, 1100.0]  # both reports kept

    pit = edgar.point_in_time(facts, _close(TTM_DAY))
    quarters = edgar.quarterly_values(edgar.field_facts(pit, "revenue"))
    print(f"\nquarterly_values for revenue at the {TTM_DAY} close:")
    for row in quarters.itertuples():
        print(f"  {row.cik}  {row.start.date()}..{row.end.date()}  {row.val:>5.0f}  {row.basis}")
    ttm = edgar.trailing_twelve_months(pit, "revenue")
    print(f"\ntrailing_twelve_months for revenue at the {TTM_DAY} close:")
    for row in ttm.itertuples():
        print(f"  {row.cik}  {row.ttm:.0f} through {row.period_end.date()} ({row.derived_quarters} derived quarters)")
    print(f"  {CIK_B} is absent: its latest four quarters skip Q2 2023, and a gap is never filled.")
    assert list(ttm["cik"]) == [CIK_A] and float(ttm["ttm"].iloc[0]) == 380.0
    a_quarters = quarters[quarters["cik"] == CIK_A]
    assert list(a_quarters["basis"]) == ["reported", "reported", "derived", "derived"]
    assert list(a_quarters["val"]) == [90.0, 100.0, 100.0, 90.0]
    return {"facts": facts, "filings": filings, "as_of": as_of_rows, "ttm": ttm, "quarters": quarters}


# --------------------------------------------------------------------- part 2: the store builder


def arrow_schema(path: Path) -> list[tuple[str, str]]:
    """(column, Arrow type) for a parquet file, as the documentation lists them."""
    return [(field.name, str(field.type)) for field in pq.read_schema(path)]


def part2(served: Mapping[str, Mapping[str, Any]], part1_facts: pd.DataFrame) -> dict[str, Any]:
    print("\nPart 2. The store builder (prism.scripts.build_fundamentals.build), offline\n")
    membership = pd.DataFrame({
        "ticker": ["EXA", "EXB", "EXGONE", "EXLOST", "EXOLD"],
        "start": pd.to_datetime(["2015-01-02", "2018-06-01", "2012-03-01", "2013-01-02", "2001-01-02"]),
        "end": pd.to_datetime(["2026-09-25", "2026-09-25", "2016-06-30", "2015-12-31", "2008-12-31"]),
    })
    with tempfile.TemporaryDirectory(prefix="prism-edgar-example-") as tmp:
        root = Path(tmp)
        # A reviewed decision assigning a CIK to a ticker SEC's current map no longer carries
        # (the format of the --review-decisions files under data/universe/).
        decisions = root / "review_decisions.json"
        decisions.write_text(json.dumps({"decisions": [{
            "ticker": "EXGONE", "start": "2012-03-01", "end": "2016-06-30", "kind": "cik", "cik": CIK_GONE,
            "name": "Example Gone Co", "reason": "made-up review: same name, filed 10-Ks during the interval"}]}))
        client = edgar.SecClient(OFFLINE_AGENT, root / "raw", transport=offline_transport(served))
        ledger = build_fundamentals.build(
            membership, client, root, root / f"edgar_coverage_{STORE_ASOF}.json", STORE_ASOF,
            name_matches=build_fundamentals.load_review_decisions(decisions))

        written = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        print("Files written under the output directory (raw/ is the --cache-dir):")
        for name in written:
            print(f"  {name}")
        schemas = {name: arrow_schema(root / name) for name in ("facts.parquet", "filings.parquet",
                                                                 "ticker_cik.parquet")}
        for name, columns in schemas.items():
            print(f"\n{name} ({pq.read_metadata(root / name).num_rows} rows)")
            for column, kind in columns:
                print(f"  {column:<20} {kind}")
        ticker_cik = pd.read_parquet(root / "ticker_cik.parquet")
        print("\nticker_cik.parquet rows:")
        for row in ticker_cik.itertuples():
            print(f"  {row.ticker:<7} {row.start.date()}..{row.end.date()}  {row.cik}  {row.source}")
        manifest = json.loads((root / "manifest.json").read_text())
        print(f"\nmanifest.json keys: {sorted(manifest)}")
        print(f"coverage ledger keys: {sorted(ledger)}")
        print(f"coverage ledger status_counts: {ledger['status_counts']}")
        for row in ledger["not_accepted"]:
            print(f"  not accepted: {row['ticker']} {row['start']}..{row['end']} {row['status']} ({row['reason']})")

        stored = pd.read_parquet(root / "facts.parquet")
        stored_view = view_at(stored, "2024-04-16")
        print("\nRead back from facts.parquet, Example A at the 2024-04-16 close: FY2023 revenue "
              f"{stored_view[0]:.0f}, total assets {stored_view[1]:.0f}, TTM revenue {stored_view[2]:.0f}")
        assert stored_view == view_at(part1_facts, "2024-04-16")
        assert ledger["status_counts"] == {"accepted": 3, "pre_xbrl": 1, "unresolved": 1}
        return {"schemas": schemas, "ledger": ledger, "manifest_keys": sorted(manifest), "files": written,
                "ticker_cik": ticker_cik}


# --------------------------------------------------------------------- live mode


def last_completed_session() -> dt.date:
    """The latest session whose 16:00 New York close has passed (within the bundled calendar)."""
    now = pd.Timestamp.now(tz=edgar.NEW_YORK)
    today = min(now.date(), sessions.coverage_end())
    if sessions.is_session(today) and (today < now.date() or now >= edgar.session_close(pd.Timestamp(today))):
        return today
    return sessions.previous_session(today)


def outside_checkout(path: Path) -> Path:
    """``path`` resolved, or SystemExit if it lies inside a git checkout (this one or any other)."""
    resolved = path.expanduser().resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists():
            raise SystemExit(f"--cache-dir {resolved} is inside the git checkout {parent}; "
                             "choose a directory outside any checkout")
    return resolved


def live(cik: int, cache_dir: Path, as_of: str | None) -> dict[str, Any]:
    agent = edgar.user_agent_from_env()  # raises without a contact e-mail address
    client = edgar.SecClient(agent, outside_checkout(cache_dir))
    history, _ = client.submissions(cik)
    pages = [client.submissions_page(page["name"])[0] for page in history.get("filings", {}).get("files", [])]
    filings = edgar.submissions_filings(history, pages)
    payload, meta = client.companyfacts(cik)
    facts = edgar.attach_availability(edgar.companyfacts_frame(payload, edgar.concept_names()), filings)
    day = pd.Timestamp(as_of) if as_of else pd.Timestamp(last_completed_session())
    pit = edgar.point_in_time(facts, edgar.session_close(day))
    period = ["cik", "taxonomy", "concept", "unit", "start", "end"]
    restated = int((facts.groupby(period, dropna=False)["val"].nunique() > 1).sum())
    print(f"{payload.get('entityName')} (CIK {cik}), as known at the {day.date()} 16:00 New York close")
    print(f"  {len(filings)} filings; {len(facts)} facts for the canonical fields, one per fact per filing; "
          f"{restated} periods reported with more than one value; company facts sha256 {meta['sha256'][:12]}")
    for name in ("revenue", "net_income", "cfo"):
        ttm = edgar.trailing_twelve_months(pit, name)
        text = (f"{ttm['ttm'].iloc[0]:,.0f} through {ttm['period_end'].iloc[0].date()}" if len(ttm)
                else "no value (fewer than four contiguous quarters)")
        print(f"  TTM {name:<23} {text}")
    for name in ("assets", "liabilities", "equity", "shares_outstanding"):
        latest = edgar.latest_instant(pit, name)
        text = (f"{latest['value'].iloc[0]:,.0f} at {pd.Timestamp(latest['period_end'].iloc[0]).date()}"
                if len(latest) else "no value")
        print(f"  latest {name:<20} {text}")
    return {"pit": pit, "facts": facts, "filings": filings}


# --------------------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="fetch one real company from SEC instead")
    parser.add_argument("--cik", type=int, default=None, help="with --live: the company's CIK, e.g. 320193")
    parser.add_argument("--cache-dir", type=Path, default=None, help="with --live: where fetched payloads are cached")
    parser.add_argument("--as-of", default=None,
                        help="with --live: YYYY-MM-DD; default: the latest session whose close has passed")
    args = parser.parse_args(argv)
    if args.live:
        if args.cik is None or args.cache_dir is None:
            parser.error("--live needs --cik and --cache-dir")
        return live(args.cik, args.cache_dir, args.as_of)
    print(HEADER)
    served = payloads()
    first = part1(served)
    second = part2(served, first["facts"])
    return {**first, **second}


if __name__ == "__main__":
    main()
