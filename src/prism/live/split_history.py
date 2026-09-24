"""Stock-split history from Alpaca's corporate-actions endpoint (SPEC.md §5.3, N1, N7).

Company filings report share counts as they stood on the report date, while the
price bars this program trades on are split-adjusted to the end of their series.
Anything that multiplies the two (market equity, net share issuance) needs the
splits in between. The frozen datasets record none, so this module reads them from
the corporate-actions endpoint the live loop already uses for spin-offs
(``prism.live.spinoff_mask``), with the same transport, batching and loud handling of
response-shape drift.

A split record becomes ``(symbol, ex_date, ratio)``, where ``ratio`` is new shares
per old share: 4.0 for a 4-for-1 forward split, 0.1 for a 1-for-10 reverse split.
The arithmetic that applies them lives with its consumer,
``prism.io.fundamental_panel.split_factor``, so the data layer never imports ``live``.

Production-import-path safe (N8): stdlib, pandas and the shared Alpaca transport only.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from prism.live.alpaca_transport import AlpacaSession

DATA_BASE_URL = "https://data.alpaca.markets"
CORPORATE_ACTIONS_PATH = "/v1/corporate-actions"
SYMBOL_BATCH = 50
PAGE_LIMIT = 1000
SPLIT_TYPES = ("forward_split", "reverse_split")
SPLIT_COLUMNS = ("symbol", "type", "ex_date", "old_rate", "new_rate", "ratio")


def fetch_splits(symbols: Sequence[str], start: str, end: str, *, key_id: str, secret_key: str,
                 session: Any | None = None, timeout: float = 30.0) -> pd.DataFrame:
    """Every forward and reverse split for ``symbols`` with ex-date in ``[start, end]``.

    A payload missing the ``corporate_actions`` container is response-shape drift
    and raises (N7); a present container without a split key is a valid "no splits".
    A record without a positive old/new rate or an ex-date raises rather than being
    skipped: a dropped split would silently mis-scale every market value after it.
    """
    api = AlpacaSession(key_id, secret_key, DATA_BASE_URL, session=session, timeout=timeout)
    rows: list[dict[str, Any]] = []
    symbols = list(symbols)
    for i in range(0, len(symbols), SYMBOL_BATCH):
        batch = symbols[i : i + SYMBOL_BATCH]
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {"symbols": ",".join(batch), "types": ",".join(SPLIT_TYPES), "start": start,
                                      "end": end, "limit": PAGE_LIMIT}
            if page_token:
                params["page_token"] = page_token
            payload = api.json_or_raise(api.request("GET", CORPORATE_ACTIONS_PATH, params=params),
                                        f"GET {CORPORATE_ACTIONS_PATH}")
            container = payload.get("corporate_actions")
            if container is None:
                raise ValueError(f"corporate-actions response lacks 'corporate_actions' (keys {sorted(payload)}); "
                                 "response-shape drift is a failure, not 'no splits' (N7)")
            for kind in SPLIT_TYPES:
                for record in container.get(f"{kind}s", []) or []:
                    rows.append(_split_row(kind, record))
            page_token = payload.get("next_page_token")
            if not page_token:
                break
    frame = pd.DataFrame(rows, columns=list(SPLIT_COLUMNS))
    frame["ex_date"] = pd.to_datetime(frame["ex_date"])
    return frame.drop_duplicates(["symbol", "ex_date", "ratio"]).sort_values(["symbol", "ex_date"]).reset_index(
        drop=True)


def _split_row(kind: str, record: dict[str, Any]) -> dict[str, Any]:
    try:
        old, new = float(record["old_rate"]), float(record["new_rate"])
        symbol, ex_date = str(record["symbol"]), str(record["ex_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{kind} record without symbol/ex_date/old_rate/new_rate: {record!r}") from exc
    if old <= 0 or new <= 0 or not ex_date:
        raise ValueError(f"{kind} record with a non-positive rate or empty ex_date: {record!r}")
    return {"symbol": symbol, "type": kind, "ex_date": ex_date, "old_rate": old, "new_rate": new,
            "ratio": new / old}
