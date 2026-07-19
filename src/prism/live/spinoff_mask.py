"""Spin-off eligibility mask (docs/bar_vendor_divergence.md §5) — live-loop only.

The live loop decides on Alpaca ``adjustment=split`` bars, which leave
spin-offs unadjusted while the certified spine back-adjusts them, so a name
with a spin-off inside the momentum lookback carries a distorted live-side
rank until the window clears the event (measured: bar_vendor_divergence §3 —
every systematic rank flip, ~1 name/leg/refresh at the short boundary; FTV
short-entered on a divergent rank at two refreshes). Remediation: such a name
is UNRANKABLE — removed from the scored cross-section at the refresh, no new
position may open on it, a held position is held (never liquidated) until the
event clears the lookback. The application seam is ``run_daily_cycle``'s
``unrankable`` provider (``prism.live.daily``); this module is the detection
side.

Detection reads the Alpaca corporate-actions endpoint — the surface the
dividend-wedge diagnostic already exercised on the paper key at $0
(``research/scripts/dividend_wedge.py``) — for ``spin_off`` records over the
trailing lookback window at decision time, through an injectable
requests-compatible session (the ``AlpacaBroker`` idiom) so every mapping is
tested offline. The per-decision-bar answer is cached as JSON in the run dir:
a same-bar rerun never refetches, and the cached flag lists are the durable
masked-name record the M6 divergence ledger consults. A detection failure is
a LOUD N7 warning naming every unchecked symbol, and the refresh proceeds
UNMASKED — the mask is a protection, not a correctness precondition.

Production-import-path safe (N8): stdlib + requests only.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Sequence

import requests

logger = logging.getLogger(__name__)

# Same endpoint, auth headers, batching, and pagination as the dividend-wedge
# fetch (research/scripts/dividend_wedge.py) — the $0 vendor and key the loop
# already reads bars from.
CORPORATE_ACTIONS_URL = "https://data.alpaca.markets/v1/corporate-actions"
SYMBOL_BATCH = 50
PAGE_LIMIT = 1000


def fetch_spinoffs(
    symbols: Sequence[str],
    start: str,
    end: str,
    *,
    key_id: str,
    secret_key: str,
    session: Any | None = None,
    timeout: float = 30.0,
) -> list[dict]:
    """All ``spin_off`` records for ``symbols`` in ``[start, end]``, paginated.

    Credentials travel only in request headers (the ``AlpacaBroker``
    convention) — never in URLs, exceptions, or logs.
    """
    sess = session if session is not None else requests.Session()
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}
    symbols = list(symbols)
    records: list[dict] = []
    for i in range(0, len(symbols), SYMBOL_BATCH):
        batch = symbols[i : i + SYMBOL_BATCH]
        page_token: str | None = None
        while True:
            params: dict = {
                "symbols": ",".join(batch),
                "types": "spin_off",
                "start": start,
                "end": end,
                "limit": PAGE_LIMIT,
            }
            if page_token:
                params["page_token"] = page_token
            resp = sess.get(CORPORATE_ACTIONS_URL, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            records.extend(payload.get("corporate_actions", {}).get("spin_offs", []) or [])
            page_token = payload.get("next_page_token")
            if not page_token:
                break
    return records


def _record_symbol(record: dict) -> str:
    # A spin_off record carries the parent as ``source_symbol`` — the name
    # whose pre-event bars the spine back-adjusts and Alpaca does not.
    # ``symbol`` is tolerated as an alternate payload shape.
    return str(record.get("source_symbol") or record.get("symbol") or "")


def _record_date(record: dict) -> str:
    # ``ex_date`` is the divergence-relevant date: the spine back-adjusts every
    # bar strictly before it. ``process_date`` is the fallback shape.
    return str(record.get("ex_date") or record.get("process_date") or "")


def spinoff_flags(
    records: Sequence[dict],
    symbols: Collection[str],
    window_start: str,
    decision_bar: str,
) -> dict[str, list[dict]]:
    """Universe names with a spin-off ex-date inside ``(window_start, decision_bar]``.

    Causality: an event dated after ``decision_bar`` never flags — masking at
    bar *t* uses only events known at *t*. An event on or before
    ``window_start`` is outside the lookback (both score endpoints postdate
    it, so the two vendors' ratio agrees). ISO date strings compare
    lexicographically. Records lacking a symbol or date are counted and warned,
    never silently dropped (N7).
    """
    universe = set(symbols)
    flagged: dict[str, list[dict]] = {}
    malformed = 0
    for record in records:
        symbol, ex_date = _record_symbol(record), _record_date(record)
        if not symbol or not ex_date:
            malformed += 1
            continue
        if symbol in universe and window_start < ex_date <= decision_bar:
            flagged.setdefault(symbol, []).append(record)
    if malformed:
        logger.warning(
            "%d spin-off record(s) lacked a symbol or date and could not be mapped "
            "to the universe (N7): vendor payload shape drift — inspect the raw "
            "corporate-actions response",
            malformed,
        )
    return flagged


def spinoff_unrankable(
    run_dir: str | Path,
    decision_bar: str,
    symbols: Collection[str],
    window_start: str,
    *,
    key_id: str | None = None,
    secret_key: str | None = None,
    session: Any | None = None,
    timeout: float = 30.0,
) -> list[str]:
    """Symbols unrankable at ``decision_bar`` (spin-off inside the lookback), cached.

    The per-bar JSON cache (``spinoff_mask_<bar>.json`` in ``run_dir``) makes a
    same-bar rerun fetch-free and records the flagged names beside their raw
    event records — the masked-name list the M6 divergence ledger consults. A
    cache whose window or checked universe differs from the request is stale
    (universe files regenerate) and is refetched. ``key_id``/``secret_key``
    default to the standard ``APCA_API_*`` environment variables.

    ANY detection failure — missing credentials, transport, venue error —
    warns loudly naming every unchecked symbol and returns the empty mask: the
    refresh proceeds UNMASKED, because the mask is a protection, not a
    correctness precondition. Failures are never cached (a same-bar rerun
    retries the fetch).
    """
    run_dir = Path(run_dir)
    cache_path = run_dir / f"spinoff_mask_{decision_bar}.json"
    checked = sorted(set(symbols))
    cached = _read_cache(cache_path, window_start, checked)
    if cached is not None:
        return cached
    key_id = key_id if key_id is not None else os.environ.get("APCA_API_KEY_ID", "")
    secret_key = secret_key if secret_key is not None else os.environ.get("APCA_API_SECRET_KEY", "")
    try:
        if not key_id or not secret_key:
            raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set")
        records = fetch_spinoffs(
            checked,
            window_start,
            decision_bar,
            key_id=key_id,
            secret_key=secret_key,
            session=session,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 — protection, not precondition
        logger.warning(
            "SPINOFF MASK UNAVAILABLE for %s (%s: %s) — refresh proceeds UNMASKED; "
            "ALL %d symbols unchecked for lookback spin-offs (N7): %s",
            decision_bar,
            type(exc).__name__,
            exc,
            len(checked),
            checked,
        )
        return []
    flagged = spinoff_flags(records, checked, window_start, decision_bar)
    payload = {
        "decision_bar": decision_bar,
        "window_start": window_start,
        "source": CORPORATE_ACTIONS_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_records": len(records),
        "symbols_checked": checked,
        "flagged": {s: flagged[s] for s in sorted(flagged)},
    }
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "spin-off mask cache write failed for %s (%s) — the mask still applies "
            "this refresh; a same-bar rerun refetches",
            cache_path,
            exc,
        )
    if flagged:
        logger.info(
            "spin-off mask %s: %d/%d name(s) unrankable (event inside (%s, %s]): %s",
            decision_bar,
            len(flagged),
            len(checked),
            window_start,
            decision_bar,
            sorted(flagged),
        )
    return sorted(flagged)


def _read_cache(cache_path: Path, window_start: str, checked: list[str]) -> list[str] | None:
    """Cached flag list for this exact question, else ``None`` (fetch)."""
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_window = payload["window_start"]
        cached_symbols = sorted(set(payload["symbols_checked"]))
        flagged = sorted(payload["flagged"])
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("unreadable spin-off mask cache %s (%s) — refetching", cache_path, exc)
        return None
    if cached_window != window_start or cached_symbols != checked:
        logger.info(
            "spin-off mask cache %s answers a different window/universe — refetching",
            cache_path.name,
        )
        return None
    logger.info("spin-off mask cache hit: %s (%d flagged)", cache_path.name, len(flagged))
    return flagged
