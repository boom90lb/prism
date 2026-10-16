"""Earnings announcement dates from 8-K item 2.02 filings, on made-up filings and prices.

Synthetic data; demonstrates the API only; makes no claim about any strategy.

Run it from a checkout, with the package installed (``uv sync``) or with ``PYTHONPATH=src``::

    python examples/point_in_time/earnings_dates.py

By default it is offline: the filing histories and prices are made up, it needs no network
and no ``PRISM_SEC_USER_AGENT``, and it writes only inside a temporary directory, which it
deletes. The companies, CIKs and accession numbers are invented; any resemblance to a real
filer is accidental. ``docs/earnings_dates.md`` explains the rules and documents every column
printed here. Sessions come from the bundled exchange calendar (``prism.io.sessions``).

The cases, all from ``prism.io.earnings_events`` and ``prism.io.edgar``:

* An original 8-K with item 2.02 is an announcement; an 8-K/A is never one, an 8-K without
  item 2.02 is not one, and a second item 2.02 8-K within five sessions belongs to the first.
* Day 0 is the New York date of the filing's acceptance. A release after the 16:00 close keeps
  that date as day 0, and its price reaction falls on day 0 + 1, inside the window. A release
  on a date the exchange is closed (here Good Friday, when EDGAR is open) moves to the next
  session.
* The announcement return sums the stock's return less the market's over sessions
  day 0 - 2 .. day 0 + 1, and it is first usable at the close of day 0 + 1.
* The days to the next expected announcement count only releases accepted by each session's
  16:00 close.

Part 2 builds the event list and the daily panel the way ``prism.scripts.build_earnings_panel``
does, writes them to parquet in a temporary directory and prints their schemas.

To list one real company's announcement dates instead (this fetches from SEC and caches what
it fetches)::

    PRISM_SEC_USER_AGENT="Your Name you@example.com" \\
        python examples/point_in_time/earnings_dates.py --live --cik 320193 --cache-dir ~/edgar-cache

``--cache-dir`` must be outside any git checkout, so a live run never writes into this
repository.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prism.io import earnings_events as ee
from prism.io import edgar, sessions

HEADER = "Synthetic data; demonstrates the API only; makes no claim about any strategy."

CIK_A, CIK_B = 9900001, 9900002
SID_A, SID_B = "EXA", "EXB"
FIRST, LAST = dt.date(2024, 1, 2), dt.date(2024, 6, 28)
SEED = 11
N_MARKET = 120  # synthetic index members besides the two companies; market_returns needs at least 100
JUMP = 0.06  # Example A's return on the session after its January release

#: Accession number -> expected day 0 for the announcements that are kept.
EXPECTED_D0 = {
    "0009900001-24-000002": "2024-01-30",  # accepted 16:30 Tuesday, after the close: day 0 is that Tuesday
    "0009900002-24-000008": "2024-04-01",  # accepted 08:00 on Good Friday: day 0 is Monday
    "0009900001-24-000011": "2024-04-30",  # accepted 16:05 Tuesday
}
#: Example A's days to its next expected announcement at three closes around its January release.
EXPECTED_DAYS_TO_EXPECTED = {"2024-01-26": 4.0, "2024-01-30": 0.0, "2024-01-31": 90.0}


def _submissions(cik: int, name: str, rows: list[tuple[str, str, str, str, str]]) -> dict[str, Any]:
    """A filing-history payload; each row is (accession number, form, filing date, acceptance in UTC, items)."""
    recent = {
        "accessionNumber": [r[0] for r in rows], "form": [r[1] for r in rows], "filingDate": [r[2] for r in rows],
        "acceptanceDateTime": [r[3] for r in rows], "reportDate": [r[2] for r in rows], "items": [r[4] for r in rows],
    }
    return {"cik": str(cik), "name": name, "filings": {"recent": recent, "files": []}}


def histories() -> list[dict[str, Any]]:
    a = _submissions(CIK_A, "Example A Corp", [
        # Last year's releases: before the sample, so never events, but they set the expected dates.
        ("0009900001-23-000002", "8-K", "2023-01-31", "2023-01-31T21:30:00.000Z", "2.02,9.01"),
        ("0009900001-23-000013", "8-K", "2023-05-02", "2023-05-02T20:30:00.000Z", "2.02,9.01"),
        ("0009900001-23-000023", "8-K", "2023-08-01", "2023-08-01T20:30:00.000Z", "2.02,9.01"),
        ("0009900001-23-000033", "8-K", "2023-10-31", "2023-10-31T20:30:00.000Z", "2.02,9.01"),
        ("0009900001-24-000002", "8-K", "2024-01-30", "2024-01-30T21:30:00.000Z", "2.02,9.01"),  # 16:30 EST
        ("0009900001-24-000004", "8-K/A", "2024-02-01", "2024-02-01T15:00:00.000Z", "2.02,9.01"),  # amendment
        ("0009900001-24-000005", "8-K", "2024-02-02", "2024-02-02T13:00:00.000Z", "2.02"),  # supplement
        ("0009900001-24-000009", "8-K", "2024-03-05", "2024-03-05T21:00:00.000Z", "5.02"),  # officer change
        ("0009900001-24-000011", "8-K", "2024-04-30", "2024-04-30T20:05:00.000Z", "2.02,9.01"),  # 16:05 EDT
        ("0009900001-24-000012", "10-Q", "2024-05-02", "2024-05-02T20:00:00.000Z", ""),
    ])
    b = _submissions(CIK_B, "Example B Inc", [
        ("0009900002-23-000005", "8-K", "2023-03-30", "2023-03-30T12:00:00.000Z", "2.02"),
        ("0009900002-24-000008", "8-K", "2024-03-29", "2024-03-29T12:00:00.000Z", "2.02,7.01,9.01"),  # Good Friday
    ])
    return [a, b]


def session_index() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(sessions.sessions_between(FIRST, LAST)))


def synthetic_returns(days: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Closes for the two companies and ``N_MARKET`` other index members, and the membership mask."""
    rng = np.random.default_rng(SEED)
    columns = [SID_A, SID_B, *(f"M{i:03d}" for i in range(N_MARKET))]
    daily = pd.DataFrame(rng.normal(0.0003, 0.012, (len(days), len(columns))), index=days, columns=columns)
    daily.loc[pd.Timestamp("2024-01-31"), SID_A] = JUMP
    daily.iloc[0] = 0.0
    close = 50.0 * (1.0 + daily).cumprod()
    members = pd.DataFrame(True, index=days, columns=columns)
    return close, members


def _fmt(value: float, spec: str = ".4f") -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def part1() -> dict[str, Any]:
    print("\nPart 1. Announcement dates, the window and the expected date\n")
    filings = pd.concat([edgar.submissions_filings(h) for h in histories()], ignore_index=True)
    releases = edgar.earnings_releases(filings)
    print(f"earnings_releases: {len(releases)} of {len(filings)} filings are 8-K or 8-K/A with item 2.02")
    for row in releases.itertuples():
        print(f"  {row.accn}  {row.form:<5}  accepted {row.acceptance:%a %Y-%m-%d %H:%M %Z}  items {row.items}")

    days = session_index()
    dates = ee.announcement_dates(releases, days)
    print(f"\nannouncement_dates over the {len(days)} sessions {days[0].date()}..{days[-1].date()}:")
    for row in dates.itertuples():
        print(f"  {row.accn}  CIK {row.cik}  day 0 {row.d0:%a %Y-%m-%d}")
    print("  Dropped: last year's releases (before the first session), the 8-K/A, and the 2024-02-02")
    print("  supplement, three sessions after a kept release.")
    got = {row.accn: str(row.d0.date()) for row in dates.itertuples()}
    assert got == EXPECTED_D0, got

    close, members = synthetic_returns(days)
    returns = close.pct_change(fill_method=None)
    market = ee.market_returns(returns, members)
    dates_by_cik = {int(part["cik"].iloc[0]): part for _, part in dates.groupby("cik")}
    intervals = {sid: pd.DataFrame({"start": [pd.Timestamp("2015-01-02")], "end": [pd.Timestamp("2026-09-25")],
                                    "cik": [cik]}) for sid, cik in ((SID_A, CIK_A), (SID_B, CIK_B))}
    events = {sid: ee.security_events(sid, intervals[sid], dates_by_cik, days, returns[sid].to_numpy(float),
                                      market.to_numpy(float)) for sid in (SID_A, SID_B)}

    first = events[SID_A].iloc[0]
    d0 = int(first["signal_pos"]) - ee.WINDOW[1]
    print(f"\nExample A's January announcement: day 0 {first['d0'].date()}, window {ee.WINDOW[0]}..+{ee.WINDOW[1]}")
    print(f"  {'session':<12} {'offset':>6} {'stock':>8} {'market':>8} {'excess':>8}")
    excess_sum = 0.0
    for offset in range(ee.WINDOW[0], ee.WINDOW[1] + 1):
        day = days[d0 + offset]
        stock, mkt = float(returns[SID_A].iloc[d0 + offset]), float(market.iloc[d0 + offset])
        excess_sum += stock - mkt
        print(f"  {day.date()!s:<12} {offset:>+6d} {stock:>8.4f} {mkt:>8.4f} {stock - mkt:>8.4f}")
    print(f"  announcement return (sum of excess) {first['ear']:.4f}; signal date {first['signal_date'].date()}: "
          "first usable at that close, traded at the next open")
    assert math.isclose(first["ear"], excess_sum, rel_tol=0, abs_tol=1e-12)
    assert first["signal_date"] == pd.Timestamp("2024-01-31")

    history_by_cik = {int(part["cik"].iloc[0]): part for _, part in ee.company_history(releases).groupby("cik")}
    probe = pd.DatetimeIndex([pd.Timestamp(d) for d in EXPECTED_DAYS_TO_EXPECTED])
    to_expected = ee.days_to_expected(history_by_cik[CIK_A], probe)
    print("\nExample A's days to its next expected announcement (days_to_expected):")
    for day, value in zip(probe, to_expected):
        print(f"  {day.date()}  {value:.0f}")
    print("  On 2024-01-30 the release (16:30) is not yet known at the 16:00 close, so the expected date is")
    print("  still that day; from 2024-01-31 it is the anniversary of last May's release.")
    assert dict(zip(EXPECTED_DAYS_TO_EXPECTED, to_expected.tolist())) == EXPECTED_DAYS_TO_EXPECTED
    return {"releases": releases, "dates": dates, "events": events, "close": close, "history": history_by_cik,
            "intervals": intervals, "days": days, "days_to_expected": dict(zip(EXPECTED_DAYS_TO_EXPECTED,
                                                                                to_expected.tolist()))}


def arrow_schema(path: Path) -> list[tuple[str, str]]:
    """(column, Arrow type) for a parquet file, as the documentation lists them."""
    return [(field.name, str(field.type)) for field in pq.read_schema(path)]


def part2(first: dict[str, Any]) -> dict[str, Any]:
    print("\nPart 2. The event list and the daily panel, as prism.scripts.build_earnings_panel writes them\n")
    days, close = first["days"], first["close"]
    panels = []
    for sid in (SID_A, SID_B):
        bars = pd.DataFrame({"ts": days.tz_localize(edgar.NEW_YORK), "close": close[sid].to_numpy(),
                             "volume": 1_000_000.0})
        panels.append(ee.security_panel(sid, bars, first["intervals"][sid], first["events"][sid], first["history"],
                                        days))
    members = {sid: (None, pd.Timestamp("2026-09-25")) for sid in (SID_A, SID_B)}
    events = pd.concat(list(first["events"].values()), ignore_index=True)[list(ee.EVENT_COLUMNS)]
    panel = ee.finalize(pd.concat(panels, ignore_index=True), members)
    with tempfile.TemporaryDirectory(prefix="prism-earnings-example-") as tmp:
        schemas = {}
        for name, frame in (("events.parquet", events), ("panel.parquet", panel)):
            path = Path(tmp) / name
            frame.to_parquet(path)
            schemas[name] = arrow_schema(path)
            print(f"{name} ({len(frame)} rows)")
            for column, kind in schemas[name]:
                print(f"  {column:<18} {kind}")
            print()
    print("events.parquet rows:")
    for row in events.itertuples():
        print(f"  {row.sid}  {row.accn}  day 0 {row.d0.date()}  signal date {row.signal_date.date()}  "
              f"ear {_fmt(row.ear)}")
    print()
    window = panel[(panel["sid"] == SID_A) & (panel["date"] >= "2024-01-26") & (panel["date"] <= "2024-02-02")]
    print("panel rows for Example A around its January announcement:")
    print(f"  {'date':<12} {'d0':<12} {'signal_date':<12} {'age':>4} {'ear':>8} {'days_to_expected':>17}")
    for row in window.itertuples():
        d0 = "-" if pd.isna(row.d0) else str(row.d0.date())
        signal = "-" if pd.isna(row.signal_date) else str(row.signal_date.date())
        print(f"  {row.date.date()!s:<12} {d0:<12} {signal:<12} {_fmt(row.age_sessions, '.0f'):>4} "
              f"{_fmt(row.ear):>8} {_fmt(row.days_to_expected, '.0f'):>17}")
    jan31 = window[window["date"] == pd.Timestamp("2024-01-31")].iloc[0]
    jan30 = window[window["date"] == pd.Timestamp("2024-01-30")].iloc[0]
    assert jan31["age_sessions"] == 0 and math.isclose(jan31["ear"], first["events"][SID_A].iloc[0]["ear"])
    assert math.isnan(jan30["ear"])  # the announcement return is not known before its signal date
    return {"schemas": schemas, "events_frame": events, "panel": panel}


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


def live(cik: int, cache_dir: Path, last: int) -> dict[str, Any]:
    agent = edgar.user_agent_from_env()  # raises without a contact e-mail address
    client = edgar.SecClient(agent, outside_checkout(cache_dir))
    history, _ = client.submissions(cik)
    pages = [client.submissions_page(page["name"])[0] for page in history.get("filings", {}).get("files", [])]
    releases = edgar.earnings_releases(edgar.submissions_filings(history, pages))
    days = pd.DatetimeIndex(pd.to_datetime(sessions.sessions_between(sessions.coverage()[0],
                                                                     last_completed_session())))
    dates = ee.announcement_dates(releases, days)
    expected = ee.days_to_expected(ee.company_history(releases), days[-1:])
    print(f"{history.get('name')} (CIK {cik}): {len(releases)} item 2.02 8-Ks, {len(dates)} announcements "
          f"on sessions {days[0].date()}..{days[-1].date()}")
    print(f"  {'accession':<22} {'accepted (New York)':<22} {'day 0':<12} signal date")
    for row in dates.tail(last).itertuples():
        position = int(row.d0_pos) + ee.WINDOW[1]
        signal = str(days[position].date()) if position < len(days) else "after the last session"
        print(f"  {row.accn:<22} {row.acceptance:%Y-%m-%d %H:%M %Z}   {row.d0.date()!s:<12} {signal}")
    print(f"  days to the next expected announcement at the {days[-1].date()} close: {_fmt(expected[0], '.0f')}")
    return {"releases": releases, "dates": dates}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="fetch one real company from SEC instead")
    parser.add_argument("--cik", type=int, default=None, help="with --live: the company's CIK, e.g. 320193")
    parser.add_argument("--cache-dir", type=Path, default=None, help="with --live: where fetched payloads are cached")
    parser.add_argument("--last", type=int, default=8, help="with --live: how many recent announcements to list")
    args = parser.parse_args(argv)
    if args.live:
        if args.cik is None or args.cache_dir is None:
            parser.error("--live needs --cik and --cache-dir")
        return live(args.cik, args.cache_dir, args.last)
    print(HEADER)
    first = part1()
    second = part2(first)
    return {**first, **second}


if __name__ == "__main__":
    main()
