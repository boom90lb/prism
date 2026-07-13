"""Alpaca market-data adapter — daily bars for the live loop (SPEC §7.0/§7.4).

The R2 paper instrument decides after close *t* and fills at open *t+1* on
Alpaca; sourcing its **bars** from Alpaca too — rather than the Twelve Data
spine the research/backtest path uses — puts decision and fill on one venue and
one clock. That removes the cross-vendor EOD-latency lag that stalled the loop
on 2026-07-07: Twelve Data had not published that session's daily bar hours
after the close, so the driver kept re-deciding the prior bar and never settled.
Alpaca serves the just-closed session's daily bar the same evening because it is
the venue that printed it.

Duck-typed to :meth:`prism.io.loader.DataLoader.fetch_incremental` so it drops
into :func:`prism.live.daily.fetch_universe_panels` unchanged. For a universe of
more than a few dozen names, :meth:`fetch_batch` pulls the whole panel in a few
paginated multi-symbol requests — one request per name blows the IEX ~200 req/min
budget (the 429 a ~500-name momentum universe hits). Free accounts get the IEX
feed; recent SIP requires a paid subscription (a 403 the adapter surfaces loudly
rather than degrading). ``adjustment=split`` matches the split-only,
dividend-as-cash price convention of the rest of the stack (SPEC §5).

Written against an injectable requests-compatible session so every mapping is
tested offline against canned payloads; only the HTTP transport is
network-gated. Credentials are the same ``APCA_*`` env the broker uses and
travel only in request headers, never in logs or exceptions.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
import requests

from prism.live.alpaca import AlpacaAPIError

logger = logging.getLogger(__name__)

DATA_BASE_URL = "https://data.alpaca.markets"
# Free-tier feed. SIP (consolidated) needs a paid subscription; IEX is a ~2-3%
# volume sample of the tape but its daily OHLC is a faithful tradeable series,
# and it is the same venue the orders route to.
DEFAULT_FEED = "iex"

# Daily bar timezone — the midnight-ET convention the incremental store and the
# Twelve Data loader already use, so panels stay interchangeable. Alpaca stamps a
# 1Day bar at 00:00 ET expressed in UTC (04:00Z under EDT, 05:00Z under EST);
# converting to ET lands it on midnight of the session date either way.
BAR_TZ = "America/New_York"

# Alpaca's bars endpoint caps a page at 10000 rows; a page shorter than the cap
# (or a null next_page_token) is the last page. Daily history for one name fits
# in a single page, but pagination is honored for correctness.
_BARS_PAGE_LIMIT = 10000

# Twelvedata-style shorthand -> Alpaca timeframe. Only "1d" is used by the loop.
_VENDOR_TIMEFRAMES = {"1d": "1Day", "1wk": "1Week", "1mo": "1Month"}

# A generous lower bound when no start is requested: the vendor clamps to the
# history it actually has (IEX daily begins mid-2020), so we need not guess a
# listing date.
_DEFAULT_START = "2016-01-01"

# Retryable HTTP statuses: rate limit + transient server errors. Even a batched
# universe fetch can momentarily exceed the ~200 req/min IEX budget, so back off
# and retry rather than crash a nightly run.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

# Symbols per multi-symbol request — well under any URL-length limit; the total
# page count is ~total_bars / page_limit regardless of how symbols are chunked.
_BATCH_CHUNK = 100


class AlpacaBarSource:
    """Daily/period bars from Alpaca's v2 market-data API (IEX feed by default).

    Interchangeable with :class:`~prism.io.loader.DataLoader` for the live loop's
    read path: exposes :meth:`fetch_incremental` with the same signature and the
    same tz-aware, midnight-ET, lowercase-OHLCV contract, plus :meth:`fetch_batch`
    for universe-scale multi-symbol fetches.
    """

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        *,
        base_url: str = DATA_BASE_URL,
        session: Any | None = None,
        feed: str = DEFAULT_FEED,
        adjustment: str = "split",
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not key_id or not secret_key:
            raise ValueError("Alpaca key_id and secret_key must be non-empty")
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._base_url = base_url.rstrip("/")
        self._session = session if session is not None else requests.Session()
        self._feed = feed
        self._adjustment = adjustment
        self._timeout = timeout
        self._max_retries = int(max_retries)
        self._backoff_base = float(backoff_base)
        self._sleep = sleep or time.sleep

    @classmethod
    def from_env(cls, *, base_url: str | None = None, feed: str | None = None, **kwargs: Any) -> "AlpacaBarSource":
        """Construct from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``.

        ``APCA_DATA_URL`` overrides the data host and ``APCA_DATA_FEED`` the feed
        (default ``iex``). Missing credentials raise (N7) — the same fail-loud
        contract as the broker's ``from_env``.
        """
        key_id = os.environ.get("APCA_API_KEY_ID", "")
        secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
        if not key_id or not secret_key:
            raise RuntimeError(
                "APCA_API_KEY_ID / APCA_API_SECRET_KEY are not set; "
                "export the paper-account credentials before sourcing Alpaca bars (N7)"
            )
        resolved_url = base_url or os.environ.get("APCA_DATA_URL") or DATA_BASE_URL
        resolved_feed = feed or os.environ.get("APCA_DATA_FEED") or DEFAULT_FEED
        return cls(key_id, secret_key, base_url=resolved_url, feed=resolved_feed, **kwargs)

    def fetch_incremental(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        store: Any | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Bars for ``symbol`` as a tz-aware (ET) OHLCV frame, oldest first.

        Signature-compatible with ``DataLoader.fetch_incremental``. ``store`` is
        accepted for interface parity and **ignored**: a fresh full refetch
        sidesteps splicing two vendors' split-adjustment bases, so every run reads
        a self-consistent Alpaca series. One request per name is fine for a
        handful of symbols; for a universe use :meth:`fetch_batch` instead — a
        per-name loop over hundreds of names exhausts the IEX ~200 req/min budget.
        An empty result returns an empty frame (the caller fails loud per symbol, N7).
        """
        timeframe = _VENDOR_TIMEFRAMES.get(interval, interval)
        start = start_date or _DEFAULT_START

        rows: list[dict] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "timeframe": timeframe,
                "start": start,
                "adjustment": self._adjustment,
                "feed": self._feed,
                "limit": _BARS_PAGE_LIMIT,
            }
            if end_date:
                params["end"] = end_date
            if page_token:
                params["page_token"] = page_token
            payload = self._json_or_raise(
                self._request("GET", f"/v2/stocks/{symbol}/bars", params=params),
                f"GET /v2/stocks/{symbol}/bars",
            )
            rows.extend(payload.get("bars") or [])
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        return _bars_to_frame(rows)

    def fetch_batch(
        self,
        symbols: Sequence[str],
        interval: str = "1d",
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        chunk_size: int = _BATCH_CHUNK,
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """Bars for many symbols via Alpaca's multi-symbol endpoint.

        ``/v2/stocks/bars?symbols=…`` returns ``{symbol: [bars]}`` and collapses a
        universe fetch from one request per name (which blows the IEX ~200 req/min
        budget past a few dozen names — a ~500-name momentum universe 429s) into
        ``ceil(total_bars / page_limit)`` paginated requests across
        ``chunk_size``-symbol groups. Returns one frame per requested symbol,
        empty for a symbol the vendor had no bars for; the caller decides what an
        empty frame means (N7). Signature mirrors :meth:`fetch_incremental` so
        callers can duck-type on ``hasattr(loader, "fetch_batch")``.
        """
        timeframe = _VENDOR_TIMEFRAMES.get(interval, interval)
        start = start_date or _DEFAULT_START
        collected: dict[str, list] = {str(s): [] for s in symbols}
        for chunk in _chunk(list(symbols), chunk_size):
            page_token: str | None = None
            while True:
                params: dict[str, Any] = {
                    "symbols": ",".join(chunk),
                    "timeframe": timeframe,
                    "start": start,
                    "adjustment": self._adjustment,
                    "feed": self._feed,
                    "limit": _BARS_PAGE_LIMIT,
                }
                if end_date:
                    params["end"] = end_date
                if page_token:
                    params["page_token"] = page_token
                payload = self._json_or_raise(
                    self._request("GET", "/v2/stocks/bars", params=params),
                    "GET /v2/stocks/bars",
                )
                for sym, bars in (payload.get("bars") or {}).items():
                    collected.setdefault(str(sym), []).extend(bars or [])
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
        return {sym: _bars_to_frame(rows) for sym, rows in collected.items()}

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """One HTTP call, retrying rate-limit/transient statuses with backoff.

        A 429 (or transient 5xx) is retried up to ``max_retries`` times, honoring
        a ``Retry-After`` header when present and otherwise using exponential
        backoff. The final response (success or the last error) is returned for
        :meth:`_json_or_raise` to interpret — a genuine 4xx like 403 is not
        retried and surfaces immediately.
        """
        url = self._base_url + path
        response: Any = None
        for attempt in range(self._max_retries + 1):
            response = self._session.request(
                method,
                url,
                headers=self._headers,
                timeout=self._timeout,
                **kwargs,
            )
            if response.status_code not in _RETRY_STATUSES or attempt == self._max_retries:
                return response
            headers = getattr(response, "headers", None) or {}
            try:
                retry_after = float(headers.get("Retry-After", 0) or 0)
            except (TypeError, ValueError):
                retry_after = 0.0
            wait = retry_after if retry_after > 0 else self._backoff_base * (2.0**attempt)
            logger.warning(
                "Alpaca %s -> HTTP %s (attempt %d/%d); backing off %.1fs",
                path,
                response.status_code,
                attempt + 1,
                self._max_retries,
                wait,
            )
            self._sleep(wait)
        return response

    @staticmethod
    def _json_or_raise(response: Any, context: str) -> Any:
        if 200 <= response.status_code < 300:
            return response.json()
        body = (response.text or "")[:500]
        raise AlpacaAPIError(f"{context} -> HTTP {response.status_code}: {body}", response.status_code)


def _bars_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Alpaca bar dicts -> tz-aware (midnight-ET) OHLCV frame, oldest first."""
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    # .to_numpy() detaches the column values from raw's RangeIndex so they map
    # positionally onto the timestamp index; passing the Series directly would
    # make pandas align 0,1,2… labels against the datetimes and NaN everything.
    index = pd.DatetimeIndex(pd.to_datetime(raw["t"], utc=True).dt.tz_convert(BAR_TZ))
    frame = pd.DataFrame(
        {
            "open": pd.to_numeric(raw["o"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(raw["h"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(raw["l"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(raw["c"], errors="coerce").to_numpy(),
            "volume": pd.to_numeric(raw["v"], errors="coerce").to_numpy(),
        },
        index=index,
    ).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def _chunk(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), max(int(size), 1)):
        yield items[i : i + size]
