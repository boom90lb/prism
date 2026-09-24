"""Point-in-time company fundamentals from SEC EDGAR XBRL (SPEC.md §5.3, N1, N7).

Three layers, each usable on its own.

*Fetch.* :class:`SecClient` reads ``data.sec.gov`` (company facts, filing
histories) and ``www.sec.gov`` (the ticker map). SEC's fair-access policy
requires a User-Agent naming the requester with a contact email, and refuses
undeclared traffic on ``www.sec.gov``; the client therefore takes its
User-Agent from ``PRISM_SEC_USER_AGENT`` and refuses to start without one that
carries an ``@`` (N7). Requests are spaced to at most five per second (SEC's
ceiling is ten), retried with backoff on 429 and 5xx, and cached gzip-compressed
with a sidecar recording URL, fetch time and sha256, so a vendor revision is a
changed hash, never a silent one.

*Normalize.* :func:`companyfacts_frame` flattens the company-facts payload to
one row per reported fact per filing: a value restated in a later filing
appears twice, once per filing, which is what makes point-in-time
reconstruction possible. :func:`submissions_filings` reads each filing's
acceptance timestamp (UTC in the payload; verified against issuers' known
16:30 ET earnings releases) and 8-K items, and :func:`attach_availability`
joins it to facts by accession number.

*Point in time.* A fact is available at ``available_at``: its filing's
acceptance time, or, where the acceptance time is unknown, the end of its
filing date in New York (conservative: EDGAR assigns filings accepted after
17:30 to the next day). A decision at the close of session *t* may use a fact
iff ``available_at <= close(t)`` (N1). :func:`point_in_time` keeps, for every
(company, concept, unit, period), the value from the latest available filing;
:func:`quarterly_values` derives quarter values from reported quarters and from
differences of year-to-date and annual values; :func:`trailing_twelve_months`
and :func:`latest_instant` read the canonical fields in :data:`FIELDS`.

Nothing here decides which company a ticker was on a given date: that mapping,
and its coverage ledger, belong to the builder
(``prism.scripts.build_fundamentals``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

USER_AGENT_ENV = "PRISM_SEC_USER_AGENT"
DATA_HOST = "https://data.sec.gov"
COMPANYFACTS_URL = DATA_HOST + "/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = DATA_HOST + "/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = DATA_HOST + "/submissions/{name}"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
MAX_REQUESTS_PER_SECOND = 5.0
NEW_YORK = "America/New_York"

FACT_COLUMNS = ("cik", "taxonomy", "concept", "unit", "start", "end", "val", "accn", "fy", "fp", "form", "filed",
                "frame")
FILING_COLUMNS = ("cik", "accn", "form", "filing_date", "acceptance", "report_date", "items")

#: Canonical fields and the XBRL concepts that report them, in preference order.
#: ``dei:`` marks the document-and-entity taxonomy; everything else is us-gaap.
#: Per period, the first alias with a value wins, so a company that moved from
#: ``SalesRevenueNet`` to the ASC 606 concept in 2018 reads as one series.
FIELDS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
    ),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "eps_diluted": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    "cfo": ("NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "rd_expense": ("ResearchAndDevelopmentExpense",),
    "dividends_paid": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    "buybacks": ("PaymentsForRepurchaseOfCommonStock",),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "shares_outstanding": ("dei:EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"),
}

#: Duration buckets in days (end − start + 1).
PERIOD_DAYS = {"quarter": (80, 100), "half": (170, 195), "nine_months": (260, 285), "annual": (350, 380)}


class EdgarError(RuntimeError):
    """An EDGAR request, payload or invariant failed (N7)."""


class EdgarNotFound(EdgarError):
    """The endpoint has no document for this key (HTTP 404)."""


def concept_names(fields: Iterable[str] = FIELDS) -> set[str]:
    """Bare concept names (taxonomy prefix stripped) behind the given canonical fields."""
    return {alias.split(":", 1)[-1] for name in fields for alias in FIELDS[name]}


# --------------------------------------------------------------------- fetch


def user_agent_from_env(env: Mapping[str, str] = os.environ) -> str:
    value = str(env.get(USER_AGENT_ENV, "")).strip()
    if "@" not in value:
        raise EdgarError(
            f"{USER_AGENT_ENV} must name the requester with a contact email, e.g. 'Prism research you@example.com'; "
            "SEC refuses undeclared traffic (https://www.sec.gov/os/accessing-edgar-data)"
        )
    return value


def _urllib_transport(url: str, headers: Mapping[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


@dataclass
class SecClient:
    """Rate-limited, cached, fail-loud JSON client for SEC endpoints."""

    user_agent: str
    cache_dir: Path
    transport: Callable[[str, Mapping[str, str], float], bytes] = _urllib_transport
    per_second: float = MAX_REQUESTS_PER_SECOND
    max_retries: int = 4
    timeout: float = 60.0
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _last_request: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if "@" not in self.user_agent:
            raise EdgarError("the SEC User-Agent must carry a contact email (fair-access policy)")
        if not 0 < self.per_second <= 10:
            raise EdgarError(f"per_second must lie in (0, 10], SEC's ceiling; got {self.per_second}")
        self.cache_dir = Path(self.cache_dir)

    def _space(self) -> None:
        if self._last_request is not None:
            wait = self._last_request + 1.0 / self.per_second - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last_request = self.clock()

    def _fetch(self, url: str) -> bytes:
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "identity"}
        for attempt in range(self.max_retries + 1):
            self._space()
            try:
                return self.transport(url, headers, self.timeout)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise EdgarNotFound(f"404 for {url}") from exc
                if exc.code == 403:
                    raise EdgarError(f"403 for {url}: SEC refused the request; check {USER_AGENT_ENV}") from exc
                if exc.code != 429 and exc.code < 500:
                    raise EdgarError(f"HTTP {exc.code} for {url}") from exc
                reason = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            if attempt == self.max_retries:
                raise EdgarError(f"{url} failed after {self.max_retries + 1} attempts ({reason})")
            backoff = 2.0**attempt
            logger.warning("EDGAR %s on %s; retrying in %.0fs", reason, url, backoff)
            self.sleep(backoff)
        raise AssertionError("unreachable")

    def get_json(self, url: str, cache_key: str, refresh: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        """The payload and its provenance (url, fetched_at, sha256, cache path)."""
        body_path = self.cache_dir / f"{cache_key}.json.gz"
        meta_path = self.cache_dir / f"{cache_key}.meta.json"
        if not refresh and body_path.exists() and meta_path.exists():
            raw = gzip.decompress(body_path.read_bytes())
            meta = json.loads(meta_path.read_text())
            if hashlib.sha256(raw).hexdigest() != meta.get("sha256"):
                raise EdgarError(f"cached {body_path} does not match its recorded sha256; refusing (N7)")
        else:
            raw = self._fetch(url)
            meta = {"url": url, "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            body_path.parent.mkdir(parents=True, exist_ok=True)
            body_path.write_bytes(gzip.compress(raw))
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EdgarError(f"{url} did not return JSON") from exc
        return payload, {**meta, "cache_path": str(body_path)}

    def companyfacts(self, cik: int, refresh: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.get_json(COMPANYFACTS_URL.format(cik=int(cik)), f"companyfacts/CIK{int(cik):010d}", refresh)

    def submissions(self, cik: int, refresh: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.get_json(SUBMISSIONS_URL.format(cik=int(cik)), f"submissions/CIK{int(cik):010d}", refresh)

    def submissions_page(self, name: str, refresh: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.get_json(SUBMISSIONS_PAGE_URL.format(name=name), f"submissions/{Path(name).stem}", refresh)

    def company_tickers(self, refresh: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.get_json(TICKERS_URL, "company_tickers", refresh)


# --------------------------------------------------------------------- normalize


def companyfacts_frame(payload: Mapping[str, Any], concepts: Iterable[str] | None = None) -> pd.DataFrame:
    """One row per reported fact per filing, for the requested bare concept names (all if None)."""
    try:
        cik = int(payload["cik"])
        facts = payload["facts"]
    except (KeyError, TypeError, ValueError) as exc:
        raise EdgarError("company-facts payload lacks cik/facts") from exc
    wanted = None if concepts is None else set(concepts)
    rows: list[tuple] = []
    for taxonomy, block in facts.items():
        if taxonomy not in ("us-gaap", "dei"):
            continue
        for concept, body in block.items():
            if wanted is not None and concept not in wanted:
                continue
            for unit, entries in body.get("units", {}).items():
                for entry in entries:
                    try:
                        rows.append((cik, taxonomy, concept, unit, entry.get("start"), entry["end"], entry["val"],
                                     entry["accn"], entry.get("fy"), entry.get("fp"), entry.get("form"),
                                     entry["filed"], entry.get("frame")))
                    except KeyError as exc:
                        raise EdgarError(f"CIK {cik} {taxonomy}:{concept} fact lacks {exc}") from exc
    frame = pd.DataFrame(rows, columns=list(FACT_COLUMNS))
    for column in ("start", "end", "filed"):
        frame[column] = pd.to_datetime(frame[column])
    frame["val"] = pd.to_numeric(frame["val"], errors="raise").astype(float)
    frame["fy"] = frame["fy"].astype("Int64")
    return frame


def submissions_filings(payload: Mapping[str, Any], pages: Sequence[Mapping[str, Any]] = ()) -> pd.DataFrame:
    """Every filing in a submissions payload (recent block plus any older pages)."""
    try:
        cik = int(payload["cik"])
        blocks = [payload["filings"]["recent"], *pages]
    except (KeyError, TypeError, ValueError) as exc:
        raise EdgarError("submissions payload lacks cik/filings.recent") from exc
    frames = []
    for block in blocks:
        frames.append(pd.DataFrame({
            "cik": cik,
            "accn": block["accessionNumber"],
            "form": block["form"],
            "filing_date": pd.to_datetime(block["filingDate"]),
            "acceptance": pd.to_datetime(block["acceptanceDateTime"], utc=True, errors="coerce"),
            "report_date": pd.to_datetime(block.get("reportDate", [None] * len(block["form"])), errors="coerce"),
            "items": block.get("items", [""] * len(block["form"])),
        }))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(FILING_COLUMNS))
    out["acceptance"] = out["acceptance"].dt.tz_convert(NEW_YORK)
    return out.drop_duplicates("accn", keep="first")[list(FILING_COLUMNS)]


def earnings_releases(filings: pd.DataFrame) -> pd.DataFrame:
    """8-K filings carrying item 2.02 (results of operations), with acceptance time."""
    items = filings["items"].fillna("").astype(str)
    mask = filings["form"].isin(["8-K", "8-K/A"]) & items.str.split(",").apply(lambda xs: "2.02" in xs)
    return filings.loc[mask, ["cik", "accn", "form", "filing_date", "acceptance", "items"]].reset_index(drop=True)


def attach_availability(facts: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Add ``available_at`` (tz-aware New York) and its basis to each fact."""
    acceptance = filings.drop_duplicates("accn").set_index("accn")["acceptance"]
    out = facts.copy()
    joined = pd.to_datetime(out["accn"].map(acceptance), utc=True).dt.tz_convert(NEW_YORK)
    fallback = (out["filed"] + pd.Timedelta(hours=23, minutes=59, seconds=59)).dt.tz_localize(NEW_YORK)
    known = joined.notna()
    out["available_at"] = joined.where(known, fallback)
    out["availability_basis"] = np.where(known, "acceptance", "filed_date_end")
    return out


# --------------------------------------------------------------------- point in time


def session_close(day: pd.Timestamp | str) -> pd.Timestamp:
    """16:00 New York on ``day``: the moment a decision at that session's close is made."""
    return pd.Timestamp(day).normalize().tz_localize(NEW_YORK) + pd.Timedelta(hours=16)


def point_in_time(facts: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Facts knowable at ``as_of``; per (company, concept, unit, period) the latest available report."""
    if "available_at" not in facts.columns:
        raise EdgarError("facts carry no available_at; call attach_availability first")
    if as_of.tzinfo is None:
        raise EdgarError("as_of must be timezone-aware")
    visible = facts[facts["available_at"] <= as_of]
    visible = visible.sort_values(["available_at", "accn"], kind="mergesort")
    key = ["cik", "taxonomy", "concept", "unit", "start", "end"]
    return visible.drop_duplicates(key, keep="last").reset_index(drop=True)


def period_kind(start: pd.Series, end: pd.Series) -> pd.Series:
    days = (end - start).dt.days + 1
    kind = pd.Series("other", index=start.index, dtype=object)
    kind[start.isna()] = "instant"
    for name, (low, high) in PERIOD_DAYS.items():
        kind[(days >= low) & (days <= high)] = name
    return kind


def field_facts(pit: pd.DataFrame, name: str) -> pd.DataFrame:
    """The PIT facts for one canonical field, alias-resolved per period."""
    aliases = FIELDS[name]
    frames = []
    for rank, alias in enumerate(aliases):
        taxonomy, concept = alias.split(":", 1) if ":" in alias else ("us-gaap", alias)
        part = pit[(pit["taxonomy"] == taxonomy) & (pit["concept"] == concept)].copy()
        part["alias_rank"] = rank
        frames.append(part)
    combined = pd.concat(frames, ignore_index=True) if frames else pit.iloc[0:0]
    if combined.empty:
        return combined
    combined = combined.sort_values(["cik", "start", "end", "alias_rank"], na_position="first", kind="mergesort")
    combined = combined.drop_duplicates(["cik", "start", "end"], keep="first")
    combined["kind"] = period_kind(combined["start"], combined["end"])
    return combined.reset_index(drop=True)


def quarterly_values(facts: pd.DataFrame) -> pd.DataFrame:
    """Quarter values per company from reported quarters and year-to-date differences.

    ``facts`` is one field's alias-resolved PIT duration facts. A quarter is taken as
    reported where a ~3-month fact exists; otherwise it is the difference of two
    reports sharing a start date whose ends are one quarter apart (6M − 3M, 9M − 6M,
    12M − 9M), iterated until nothing new is derived.
    """
    out = []
    for cik, group in facts[facts["kind"] != "instant"].groupby("cik"):
        values = {(row.start, row.end): row.val for row in group.itertuples()}
        quarters = {end: (start, val, "reported") for (start, end), val in values.items()
                    if PERIOD_DAYS["quarter"][0] <= (end - start).days + 1 <= PERIOD_DAYS["quarter"][1]}
        changed = True
        while changed:
            changed = False
            by_start: dict[pd.Timestamp, list[tuple[pd.Timestamp, float]]] = {}
            for (start, end), val in values.items():
                by_start.setdefault(start, []).append((end, val))
            for start, items in by_start.items():
                items.sort()
                for (end_a, val_a), (end_b, val_b) in zip(items, items[1:]):
                    gap = (end_b - end_a).days
                    if PERIOD_DAYS["quarter"][0] <= gap <= PERIOD_DAYS["quarter"][1] and end_b not in quarters:
                        q_start = end_a + pd.Timedelta(days=1)
                        quarters[end_b] = (q_start, val_b - val_a, "derived")
                        values.setdefault((q_start, end_b), val_b - val_a)
                        changed = True
        for end, (start, val, basis) in quarters.items():
            out.append((cik, start, end, val, basis))
    frame = pd.DataFrame(out, columns=["cik", "start", "end", "val", "basis"])
    return frame.sort_values(["cik", "end"]).reset_index(drop=True)


def trailing_twelve_months(pit: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per company: sum of the latest four contiguous quarters of a flow field, with its period end.

    Quarters are contiguous when each starts the day after the previous ends (±7 days
    for 52/53-week calendars). A company whose latest four quarters are not contiguous
    has no TTM value: it is absent from the result, never filled.
    """
    quarters = quarterly_values(field_facts(pit, name))
    rows = []
    for cik, group in quarters.groupby("cik"):
        last4 = group.tail(4)
        if len(last4) < 4:
            continue
        gaps = (last4["start"].iloc[1:].to_numpy() - last4["end"].iloc[:-1].to_numpy()) / np.timedelta64(1, "D")
        if np.all(np.abs(gaps - 1) <= 7):
            rows.append((cik, float(last4["val"].sum()), last4["end"].iloc[-1], int((last4["basis"] == "derived").sum())))
    return pd.DataFrame(rows, columns=["cik", "ttm", "period_end", "derived_quarters"])


def latest_instant(pit: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per company: the latest-dated instant value of a balance-sheet or share-count field."""
    facts = field_facts(pit, name)
    if facts.empty:
        return pd.DataFrame(columns=["cik", "value", "period_end"])
    instants = facts[facts["kind"] == "instant"].sort_values(["cik", "end"])
    last = instants.groupby("cik").tail(1)
    return pd.DataFrame({"cik": last["cik"].to_numpy(), "value": last["val"].to_numpy(),
                         "period_end": last["end"].to_numpy()})
