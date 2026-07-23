"""Spin-adjustment-consistency sweep over the certified spine panel (docs/bar_vendor_divergence.md §6 follow-up).

The §6 correction record found the certified spine's spin-off adjustment
NON-UNIFORM: APTV (VGNT spin, ex 2026-04-01) sits RAW in the certified basis
— the spine's own cross-event return is −1,058 bps, the mechanical
distribution step — while BDX, CMCSA, FDX, and FTV are back-adjusted.
Certified B1 momentum ranks are price ratios over this spine, so every raw
step inside a 252-bar lookback is a rank distortion the certified side
itself carries. This sweep mechanizes the APTV check for EVERY name in the
certified spine panel (``results/demotion_b1/config.json`` symbols, the
beta-telemetry loader with its quarantine fallback) over the panel's full
window:

- **Corporate actions**: the Alpaca v1 corporate-actions endpoint (the §5
  spinoff-mask surface, paper key, $0), full taxonomy (no ``types`` filter,
  so the sweep tests exactly the action set the vendor returns), batched and
  paginated, 429/Retry-After honored. The only committed local
  corporate-action record (``results/alpaca_cash_dividends_*.json``) covers
  cash dividends only and is used as an overlap cross-check; no local
  spin/split record exists (no ``runs/**/spinoff_mask_*.json`` cache has
  been written yet — the mask landed default-off), so the tested types come
  from the live fetch.
- **Mechanical expected cross-event ratio**: splits and unit splits from the
  record's ``old_rate/new_rate``; stock dividends from ``1/(1+rate)``;
  spin-offs from the distribution fraction ``(new_rate/source_rate) x
  child debut close / parent pre-event close``, both prices read from Alpaca
  ``adjustment=raw`` bars so a later rebasing on either series cannot
  contaminate the fraction (the spine's own closes are unusable here — they
  may be back-adjusted by the very convention under test).
- **Observed**: the spine panel's close-to-close ratio across the event,
  netted of the panel cross-sectional median move over the same sessions.
- **Classification per event**: BACK_ADJUSTED (net ratio nearer 1.0 — no
  step, the spine absorbed the event), RAW_STEP (net ratio nearer the
  mechanical ratio — the APTV class; step reported in bps), else
  INDETERMINATE with a named reason (missing bars around the event,
  distribution value not determinable, step below the separability floor,
  matches neither hypothesis) — never dropped (N7). Types with no
  adjustment expectation under the certified basis are tallied per type
  with the reason, not step-tested: cash dividends because the certified
  basis is price-return by design (I-7,
  ``split_adjusted_open_close_price_return_no_dividends`` — a dividend step
  is the convention, its wedge is measured by the dividend-wedge
  diagnostic), mergers / name changes / redemptions / worthless removals
  because they are identity or terminal events with no cross-event
  adjustment expectation testable against a surviving spine series.

Uncounted diagnostic: searches nothing, changes no trial code path, appends
nothing to the trials ledger, moves no ratified statistic; writes one JSON
(plus the raw action records beside it as evidence and refetch cache).
Read-only against the vendor; credentials come from the environment (source
``.env`` first) and travel only in request headers. Frame and results:
``docs/spin_adjustment_sweep.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from prism.live.alpaca_data import AlpacaBarSource
from research.scripts.bar_vendor_divergence import CountingSession, iex_close_panel
from research.scripts.beta_telemetry import load_panel_closes

CORPORATE_ACTIONS_URL = "https://data.alpaca.markets/v1/corporate-actions"
SYMBOL_BATCH = 50
PAGE_LIMIT = 1000
# Rate limit + transient server errors, retried with Retry-After honored —
# the AlpacaBarSource._request convention.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

# Response keys with a mechanical cross-event price-adjustment expectation
# under the certified basis: these are step-tested per event. Everything else
# the endpoint returns is tallied per type with a named reason (N7).
TESTED_KEYS = frozenset({"forward_splits", "reverse_splits", "unit_splits", "stock_dividends", "spin_offs"})

UNTESTED_NOTES = {
    "cash_dividends": (
        "certified basis is price-return by design (I-7 tag "
        "split_adjusted_open_close_price_return_no_dividends): a cash-dividend step is the "
        "convention, not an adjustment-consistency question — its wedge is measured by "
        "research/scripts/dividend_wedge.py"
    ),
}
UNTESTED_DEFAULT_NOTE = (
    "identity or terminal event (merger/rename/redemption class): no cross-event "
    "adjustment expectation testable against a surviving spine series"
)

# |log mechanical ratio| below this and the raw-step and back-adjusted
# hypotheses are closer together than daily idiosyncratic noise can separate:
# the event is reported INDETERMINATE, never coin-flipped.
MIN_STEP_BPS = 400.0
# Residual (market-netted) distance the winning hypothesis may sit from the
# observed ratio before the event is INDETERMINATE ("matches neither").
NOISE_BUDGET_BPS = 500.0
# Sessions the cross-event read may search on either side of the ex-date for
# a finite spine close before declaring the bars missing.
MAX_EVENT_GAP_SESSIONS = 5
# Calendar days a raw parent-prev / child-debut close may sit from the
# ex-date (IEX misses some debut bars — the §3 FDXF/HONA class).
MAX_RAW_LAG_DAYS = 10

# The §6 anchor this sweep must reproduce: APTV/VGNT ex 2026-04-01, spine
# cross-event return −1,058 bps, verified against the spine cache directly.
APTV_EXPECTED_STEP_BPS = -1058.0
APTV_STEP_TOLERANCE_BPS = 150.0


# ------------------------------------------------------------------ fetching


def _retrying_get(
    session: Any,
    url: str,
    params: dict,
    headers: dict,
    *,
    timeout: float,
    sleep: Callable[[float], None],
    max_retries: int = 5,
    backoff_base: float = 0.5,
) -> Any:
    """GET with 429/transient-5xx retry, honoring ``Retry-After`` when present."""
    response: Any = None
    for attempt in range(max_retries + 1):
        response = session.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code not in RETRY_STATUSES or attempt == max_retries:
            response.raise_for_status()
            return response
        resp_headers = getattr(response, "headers", None) or {}
        try:
            retry_after = float(resp_headers.get("Retry-After", 0) or 0)
        except (TypeError, ValueError):
            retry_after = 0.0
        sleep(retry_after if retry_after > 0 else backoff_base * (2.0**attempt))
    return response


def fetch_corporate_actions(
    symbols: list[str],
    start: str,
    end: str,
    *,
    key_id: str,
    secret_key: str,
    session: Any | None = None,
    timeout: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, list[dict]], int]:
    """All corporate-action records for ``symbols`` in ``[start, end]``, grouped by response key.

    No ``types`` filter: the response carries the full taxonomy the endpoint
    knows, so an action type this module has never heard of still lands in
    the tally rather than being silently unfetched (N7). Credentials travel
    only in request headers (the AlpacaBroker convention). Returns
    ``(records_by_type, n_http_requests)``.
    """
    sess = session if session is not None else __import__("requests").Session()
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}
    by_type: dict[str, list[dict]] = {}
    n_requests = 0
    for i in range(0, len(symbols), SYMBOL_BATCH):
        batch = symbols[i : i + SYMBOL_BATCH]
        page_token: str | None = None
        while True:
            params: dict = {"symbols": ",".join(batch), "start": start, "end": end, "limit": PAGE_LIMIT}
            if page_token:
                params["page_token"] = page_token
            resp = _retrying_get(
                sess, CORPORATE_ACTIONS_URL, params, headers, timeout=timeout, sleep=sleep
            )
            n_requests += 1
            payload = resp.json()
            for key, records in (payload.get("corporate_actions") or {}).items():
                by_type.setdefault(str(key), []).extend(records or [])
            page_token = payload.get("next_page_token")
            if not page_token:
                break
    return by_type, n_requests


def dedupe_records(by_type: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], int]:
    """Drop records repeated across symbol batches (a spin-off matches both its parent's and child's batch).

    Keyed on the vendor ``id`` when present, else the record's full item set.
    Returns ``(deduped, n_duplicates_dropped)``.
    """
    out: dict[str, list[dict]] = {}
    dropped = 0
    for key, records in by_type.items():
        seen: set = set()
        kept: list[dict] = []
        for record in records:
            marker = record.get("id") or tuple(sorted((str(k), str(v)) for k, v in record.items()))
            if marker in seen:
                dropped += 1
                continue
            seen.add(marker)
            kept.append(record)
        out[key] = kept
    return out, dropped


# ------------------------------------------------------------ event building


def _event_symbol(key: str, record: dict) -> str:
    if key == "spin_offs":
        # The parent is the name whose spine series does or does not absorb
        # the step — the spinoff_mask._record_symbol convention.
        return str(record.get("source_symbol") or record.get("symbol") or "")
    return str(record.get("symbol") or "")


def build_events(
    by_type: dict[str, list[dict]],
    panel_symbols: set[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> tuple[list[dict], dict]:
    """Tested-type records -> classifiable events, plus a full-taxonomy coverage tally.

    Every fetched record lands somewhere: a tested-type, in-window,
    in-panel record becomes an event; everything else is counted by type and
    by exclusion reason — malformed (no symbol/ex-date), out of window, or
    parent-not-in-panel — never silently dropped (N7).
    """
    events: list[dict] = []
    per_type: dict[str, dict] = {}
    untested: dict[str, dict] = {}
    parent_not_in_panel: list[dict] = []
    malformed = 0
    for key in sorted(by_type):
        records = by_type[key]
        ex_dates = sorted(str(r.get("ex_date")) for r in records if r.get("ex_date"))
        tally = {
            "n_records": len(records),
            "earliest_ex_date": ex_dates[0] if ex_dates else None,
            "latest_ex_date": ex_dates[-1] if ex_dates else None,
        }
        if key not in TESTED_KEYS:
            untested[key] = {**tally, "note": UNTESTED_NOTES.get(key, UNTESTED_DEFAULT_NOTE)}
            continue
        n_in_window = n_in_panel = n_malformed = 0
        for record in records:
            symbol = _event_symbol(key, record)
            ex_raw = record.get("ex_date") or record.get("process_date")
            if not symbol or not ex_raw:
                n_malformed += 1
                malformed += 1
                continue
            ex_date = pd.Timestamp(str(ex_raw))
            if not (window_start <= ex_date <= window_end):
                continue
            n_in_window += 1
            if symbol not in panel_symbols:
                parent_not_in_panel.append({"type": key, "symbol": symbol, "ex_date": str(ex_raw)})
                continue
            n_in_panel += 1
            events.append({"type": key, "symbol": symbol, "ex_date": ex_date, "record": record})
        per_type[key] = {
            **tally,
            "n_in_window": n_in_window,
            "n_in_panel": n_in_panel,
            "n_malformed": n_malformed,
        }
    events.sort(key=lambda e: (e["ex_date"], e["symbol"]))
    coverage = {
        "tested_types": per_type,
        "untested_types": untested,
        "n_events": len(events),
        "n_malformed_tested_records": malformed,
        "n_not_in_panel": len(parent_not_in_panel),
        "not_in_panel": parent_not_in_panel,
    }
    return events, coverage


# ------------------------------------------------------- mechanical ratios


def split_mechanical_ratio(record: dict) -> tuple[float | None, str | None]:
    """Expected raw cross-event price ratio ``old_rate/new_rate`` for a split-class record."""
    old_rate, new_rate = record.get("old_rate"), record.get("new_rate")
    try:
        old_f, new_f = float(old_rate), float(new_rate)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, "split rates not determinable from the record (old_rate/new_rate missing)"
    if old_f <= 0.0 or new_f <= 0.0:
        return None, "split rates not determinable from the record (non-positive rate)"
    return old_f / new_f, None


def stock_dividend_mechanical_ratio(record: dict) -> tuple[float | None, str | None]:
    """Expected raw cross-event price ratio ``1/(1+rate)`` for a stock dividend."""
    try:
        rate = float(record.get("rate"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, "stock-dividend rate not determinable from the record"
    if rate <= 0.0:
        return None, "stock-dividend rate not determinable from the record (non-positive rate)"
    return 1.0 / (1.0 + rate), None


def _raw_close_before(series: pd.Series | None, ex_date: pd.Timestamp, max_lag_days: int) -> tuple[float, str] | None:
    if series is None:
        return None
    prior = series.loc[series.index < ex_date].dropna()
    if prior.empty or (ex_date - prior.index[-1]).days > max_lag_days:
        return None
    return float(prior.iloc[-1]), str(prior.index[-1].date())


def _raw_close_on_or_after(
    series: pd.Series | None, ex_date: pd.Timestamp, max_lag_days: int
) -> tuple[float, str] | None:
    if series is None:
        return None
    onward = series.loc[series.index >= ex_date].dropna()
    if onward.empty or (onward.index[0] - ex_date).days > max_lag_days:
        return None
    return float(onward.iloc[0]), str(onward.index[0].date())


def spin_mechanical_ratio(
    record: dict,
    raw_parent: pd.Series | None,
    raw_child: pd.Series | None,
    *,
    max_lag_days: int = MAX_RAW_LAG_DAYS,
) -> tuple[float | None, str | None, dict]:
    """Expected raw cross-event ratio ``1 − (new_rate/source_rate)·child_debut/parent_prev``.

    Both prices come from vendor ``adjustment=raw`` series — the spine's own
    closes may be back-adjusted by the very convention under test, and an
    ``adjustment=split`` series would fold the child's or parent's later
    splits into the fraction. A missing ``source_rate`` defaults to 1.0
    (noted); a missing ``new_rate`` or missing price makes the distribution
    value undeterminable — INDETERMINATE, never guessed (N7).
    """
    detail: dict = {}
    child = str(record.get("new_symbol") or "")
    ex_date = pd.Timestamp(str(record.get("ex_date") or record.get("process_date")))
    try:
        new_rate = float(record.get("new_rate"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, "distribution value not determinable (new_rate missing from the record)", detail
    source_rate_raw = record.get("source_rate")
    try:
        source_rate = float(source_rate_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        source_rate = 1.0
        detail["source_rate_defaulted"] = True
    if new_rate <= 0.0 or source_rate <= 0.0:
        return None, "distribution value not determinable (non-positive rate)", detail
    if not child:
        return None, "distribution value not determinable (child symbol missing from the record)", detail
    parent_prev = _raw_close_before(raw_parent, ex_date, max_lag_days)
    if parent_prev is None:
        return None, f"parent raw pre-event close unavailable within {max_lag_days} days", detail
    child_debut = _raw_close_on_or_after(raw_child, ex_date, max_lag_days)
    if child_debut is None:
        return None, f"child ({child}) debut raw close unavailable within {max_lag_days} days", detail
    detail.update(
        child_symbol=child,
        rate_ratio=new_rate / source_rate,
        parent_prev_raw_close=parent_prev[0],
        parent_prev_raw_date=parent_prev[1],
        child_debut_raw_close=child_debut[0],
        child_debut_raw_date=child_debut[1],
    )
    fraction = (new_rate / source_rate) * child_debut[0] / parent_prev[0]
    detail["distribution_fraction"] = fraction
    if fraction >= 1.0:
        return None, "distribution value not determinable (implied fraction >= 1 of parent value)", detail
    return 1.0 - fraction, None, detail


# -------------------------------------------------------- spine-side reads


def cross_event_read(
    closes: pd.Series,
    market_returns: pd.Series,
    ex_date: pd.Timestamp,
    max_gap_sessions: int = MAX_EVENT_GAP_SESSIONS,
) -> dict:
    """Spine close-to-close ratio across ``ex_date``, plus the market factor over the same span.

    ``closes`` is one panel column on the full panel calendar (NaN where the
    name did not trade). The ex-side close is the first finite close on or
    after the ex-date and the prev-side close the last finite close before
    it, each within ``max_gap_sessions`` sessions; a gap larger than that is
    a missing-bars INDETERMINATE, named, never a silently stretched read
    (N7). The market factor compounds the panel median return over exactly
    the sessions the ratio spans, so a multi-session gap nets correctly.
    """
    calendar = closes.index
    pos = int(calendar.searchsorted(ex_date, side="left"))
    if pos >= len(calendar):
        return {"ok": False, "reason": "missing bars around the event: ex-date beyond the panel's last session"}
    if pos == 0:
        return {"ok": False, "reason": "missing bars around the event: no panel session before the ex-date"}
    values = closes.to_numpy()
    ex_i = next(
        (i for i in range(pos, min(pos + max_gap_sessions + 1, len(calendar))) if np.isfinite(values[i])), None
    )
    if ex_i is None:
        return {
            "ok": False,
            "reason": f"missing bars around the event: no finite spine close within "
            f"{max_gap_sessions} sessions on/after the ex-date",
        }
    prev_i = next((i for i in range(pos - 1, max(pos - 1 - max_gap_sessions, -1), -1) if np.isfinite(values[i])), None)
    if prev_i is None:
        return {
            "ok": False,
            "reason": f"missing bars around the event: no finite spine close within "
            f"{max_gap_sessions} sessions before the ex-date",
        }
    span = market_returns.iloc[prev_i + 1 : ex_i + 1]
    market_factor = float(np.prod(1.0 + span.fillna(0.0).to_numpy()))
    return {
        "ok": True,
        "reason": None,
        "r_obs": float(values[ex_i] / values[prev_i]),
        "market_factor": market_factor,
        "prev_session": str(calendar[prev_i].date()),
        "ex_session": str(calendar[ex_i].date()),
        "n_sessions_spanned": int(ex_i - prev_i),
        "ex_session_is_ex_date": bool(calendar[pos] == ex_date and ex_i == pos),
    }


def classify(
    r_obs: float,
    r_mech: float,
    market_factor: float,
    *,
    min_step_bps: float = MIN_STEP_BPS,
    noise_budget_bps: float = NOISE_BUDGET_BPS,
) -> dict:
    """BACK_ADJUSTED / RAW_STEP / INDETERMINATE for one event, in log space.

    The observed cross-event ratio, netted of the market factor, is compared
    against two hypothesis centers: 1.0 (the spine absorbed the event —
    BACK_ADJUSTED) and the mechanical ratio (the spine shows the raw step —
    RAW_STEP). The nearer center wins, provided the mechanical step exceeds
    the separability floor and the winner sits within the noise budget;
    otherwise the event is INDETERMINATE with the failing gate named.
    """
    if not (np.isfinite(r_obs) and r_obs > 0.0):
        return {"classification": "INDETERMINATE", "reason": "degenerate observed cross-event ratio"}
    if not (np.isfinite(r_mech) and r_mech > 0.0):
        return {"classification": "INDETERMINATE", "reason": "degenerate mechanical ratio"}
    separation_log = abs(float(np.log(r_mech)))
    floor_log = float(np.log1p(min_step_bps / 1e4))
    r_net = r_obs / market_factor
    d_raw = abs(float(np.log(r_net) - np.log(r_mech)))
    d_adj = abs(float(np.log(r_net)))
    out = {
        "net_cross_event_return_bps": (r_net - 1.0) * 1e4,
        "distance_raw_bps": d_raw * 1e4,
        "distance_back_adjusted_bps": d_adj * 1e4,
    }
    if separation_log < floor_log:
        out.update(
            classification="INDETERMINATE",
            reason=f"mechanical step {(r_mech - 1.0) * 1e4:+.0f} bps below the "
            f"{min_step_bps:.0f} bps separability floor",
        )
        return out
    winner, winner_distance = ("RAW_STEP", d_raw) if d_raw < d_adj else ("BACK_ADJUSTED", d_adj)
    if winner_distance > float(np.log1p(noise_budget_bps / 1e4)):
        out.update(
            classification="INDETERMINATE",
            reason=f"cross-event return matches neither hypothesis within the "
            f"{noise_budget_bps:.0f} bps noise budget (nearest: {winner})",
        )
        return out
    out.update(classification=winner, reason=None)
    return out


# --------------------------------------------------------------- the sweep


def sweep_events(
    panel: pd.DataFrame,
    events: list[dict],
    raw_closes: Mapping[str, pd.Series],
    *,
    min_step_bps: float = MIN_STEP_BPS,
    noise_budget_bps: float = NOISE_BUDGET_BPS,
    max_gap_sessions: int = MAX_EVENT_GAP_SESSIONS,
    max_raw_lag_days: int = MAX_RAW_LAG_DAYS,
) -> list[dict]:
    """Classify every event against the spine panel; one row per event, none dropped."""
    market_returns = panel.pct_change(fill_method=None).median(axis=1)
    rows: list[dict] = []
    for event in events:
        key, symbol, ex_date, record = event["type"], event["symbol"], event["ex_date"], event["record"]
        row: dict = {
            "symbol": symbol,
            "action_type": key[:-1] if key.endswith("s") else key,
            "ex_date": str(ex_date.date()),
            "mechanical_ratio": None,
            "mechanical_step_bps": None,
            "spine_cross_event_return_bps": None,
            "step_bps": None,
            "record": record,
        }
        detail: dict = {}
        if key == "spin_offs":
            child = str(record.get("new_symbol") or "")
            r_mech, mech_reason, detail = spin_mechanical_ratio(
                record, raw_closes.get(symbol), raw_closes.get(child), max_lag_days=max_raw_lag_days
            )
        elif key == "stock_dividends":
            r_mech, mech_reason = stock_dividend_mechanical_ratio(record)
        else:
            r_mech, mech_reason = split_mechanical_ratio(record)
        if detail:
            row["spin_detail"] = detail
        if r_mech is not None:
            row["mechanical_ratio"] = r_mech
            row["mechanical_step_bps"] = (r_mech - 1.0) * 1e4
        if symbol not in panel.columns:
            row.update(classification="INDETERMINATE", reason=f"no spine cache column for {symbol}")
            rows.append(row)
            continue
        read = cross_event_read(panel[symbol], market_returns, ex_date, max_gap_sessions)
        if read["ok"]:
            row.update(
                spine_cross_event_return_bps=(read["r_obs"] - 1.0) * 1e4,
                market_move_bps=(read["market_factor"] - 1.0) * 1e4,
                prev_session=read["prev_session"],
                ex_session=read["ex_session"],
                n_sessions_spanned=read["n_sessions_spanned"],
            )
        if not read["ok"]:
            row.update(classification="INDETERMINATE", reason=read["reason"])
        elif r_mech is None:
            row.update(classification="INDETERMINATE", reason=mech_reason)
        else:
            verdict = classify(
                read["r_obs"],
                r_mech,
                read["market_factor"],
                min_step_bps=min_step_bps,
                noise_budget_bps=noise_budget_bps,
            )
            row.update(verdict)
            if verdict["classification"] == "RAW_STEP":
                row["step_bps"] = row["spine_cross_event_return_bps"]
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Counts per class (overall, per type, per INDETERMINATE reason) and the flagged RAW_STEP list."""
    counts = {"BACK_ADJUSTED": 0, "RAW_STEP": 0, "INDETERMINATE": 0}
    per_type: dict[str, dict] = {}
    reasons: dict[str, int] = {}
    for row in rows:
        cls = row["classification"]
        counts[cls] = counts.get(cls, 0) + 1
        bucket = per_type.setdefault(row["action_type"], {"BACK_ADJUSTED": 0, "RAW_STEP": 0, "INDETERMINATE": 0})
        bucket[cls] = bucket.get(cls, 0) + 1
        if cls == "INDETERMINATE":
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    flagged = sorted(
        (
            {
                "symbol": r["symbol"],
                "action_type": r["action_type"],
                "ex_date": r["ex_date"],
                "step_bps": r["step_bps"],
                "mechanical_step_bps": r["mechanical_step_bps"],
            }
            for r in rows
            if r["classification"] == "RAW_STEP"
        ),
        key=lambda r: r["step_bps"],
    )
    summary = {
        "n_events": len(rows),
        "counts": counts,
        "per_type": per_type,
        "indeterminate_reasons": reasons,
    }
    return summary, flagged


def aptv_cross_check(rows: list[dict]) -> dict:
    """The §6 anchor: APTV/VGNT must classify RAW_STEP with a step near −1,058 bps."""
    matches = [
        r
        for r in rows
        if r["symbol"] == "APTV" and r["action_type"] == "spin_off" and r["ex_date"].startswith("2026-04")
    ]
    if not matches:
        return {
            "reproduced": False,
            "expected_step_bps": APTV_EXPECTED_STEP_BPS,
            "note": "no APTV spin-off event found in the fetched records (N7: coverage failure, not absence of the event)",
        }
    row = matches[0]
    step = row.get("step_bps") or row.get("spine_cross_event_return_bps")
    reproduced = (
        row["classification"] == "RAW_STEP"
        and step is not None
        and abs(step - APTV_EXPECTED_STEP_BPS) <= APTV_STEP_TOLERANCE_BPS
    )
    return {
        "reproduced": bool(reproduced),
        "expected_step_bps": APTV_EXPECTED_STEP_BPS,
        "tolerance_bps": APTV_STEP_TOLERANCE_BPS,
        "classification": row["classification"],
        "step_bps": step,
        "reason": row.get("reason"),
    }


# ---------------------------------------------------------- local cross-check


def local_dividend_cross_check(results_dir: Path, fetched: dict[str, list[dict]], panel_symbols: set[str]) -> dict:
    """Overlap check against the committed cash-dividend record (the only local corporate-actions probe).

    Every local record inside the fetch's coverage should reappear in the
    live response (matched by vendor ``id``); a miss means the vendor's
    history shifted under us and is reported, never absorbed (N7).
    """
    files = sorted(results_dir.glob("alpaca_cash_dividends_*.json"))
    if not files:
        return {"note": "no local cash-dividend record found; cross-check not run"}
    local = json.loads(files[-1].read_text())
    fetched_ids = {r.get("id") for r in fetched.get("cash_dividends", []) if r.get("id")}
    in_scope = [r for r in local if r.get("symbol") in panel_symbols and r.get("id")]
    missing = [r["id"] for r in in_scope if r["id"] not in fetched_ids]
    return {
        "local_file": files[-1].name,
        "n_local_records_in_scope": len(in_scope),
        "n_matched_in_live_fetch": len(in_scope) - len(missing),
        "n_missing_from_live_fetch": len(missing),
        "missing_ids_head": missing[:10],
    }


# ------------------------------------------------------------------------ main


def _clean(obj: Any) -> Any:
    """NaN → null recursively so the JSON never carries a bare NaN token."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="results/demotion_b1/config.json", help="Certified run config")
    parser.add_argument("--data_dir", default="data", help="Bar-cache directory")
    parser.add_argument("--quarantine_dir", default="data/quarantine", help="Quarantined-cache fallback directory")
    parser.add_argument(
        "--records_json",
        default=None,
        help="Reuse previously fetched corporate-action records (the evidence file) instead of hitting the API",
    )
    parser.add_argument(
        "--raw_closes_cache",
        default=None,
        help="Optional parquet path: reuse previously fetched adjustment=raw closes if present, else fetch and write",
    )
    parser.add_argument("--min_step_bps", type=float, default=MIN_STEP_BPS, help="Separability floor, bps")
    parser.add_argument("--noise_budget_bps", type=float, default=NOISE_BUDGET_BPS, help="Noise budget, bps")
    parser.add_argument(
        "--max_gap_sessions", type=int, default=MAX_EVENT_GAP_SESSIONS, help="Max sessions to bridge a missing bar"
    )
    parser.add_argument(
        "--fetch_end_pad_days",
        type=int,
        default=60,
        help="Days past the panel end to fetch actions (the vendor's window filter is not purely ex-date: the "
        "committed dividend record contains ex-dates before its requested start)",
    )
    parser.add_argument("--out", default=None, help="Output JSON (default: results/spin_adjustment_sweep_<today>.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    symbols = [str(s) for s in config["symbols"]]
    suffix = f"_1d_{config['start_date']}_{config['end_date']}.parquet"
    panel, missing_caches = load_panel_closes(symbols, Path(args.data_dir), Path(args.quarantine_dir), suffix)
    if missing_caches:
        print(f"LOUD: {len(missing_caches)} panel names have no spine cache (rows will say so): {missing_caches}")
    window_start, window_end = panel.index[0], panel.index[-1]
    fetch_start = str(window_start.date())
    fetch_end = str((window_end + pd.Timedelta(days=args.fetch_end_pad_days)).date())

    n_action_requests = 0
    if args.records_json:
        by_type = json.loads(Path(args.records_json).read_text())
        records_reused = True
    else:
        key_id, secret_key = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not key_id or not secret_key:
            raise SystemExit("APCA_API_KEY_ID / APCA_API_SECRET_KEY not in environment (source .env first)")
        by_type, n_action_requests = fetch_corporate_actions(
            symbols, fetch_start, fetch_end, key_id=key_id, secret_key=secret_key
        )
        raw_path = Path("results") / f"alpaca_corporate_actions_{fetch_start}_{fetch_end}.json"
        raw_path.write_text(json.dumps(by_type, indent=1))
        records_reused = False
    by_type, n_duplicates = dedupe_records(by_type)

    events, coverage = build_events(by_type, set(symbols), window_start, window_end)
    if coverage["n_malformed_tested_records"]:
        print(f"LOUD: {coverage['n_malformed_tested_records']} tested-type record(s) lacked a symbol or ex-date (N7)")

    # adjustment=raw closes for every spin parent and child — the mechanical
    # side of the distribution fraction. One batched fetch, full window.
    spin_symbols = sorted(
        {e["symbol"] for e in events if e["type"] == "spin_offs"}
        | {str(e["record"].get("new_symbol")) for e in events if e["type"] == "spin_offs" and e["record"].get("new_symbol")}
    )
    n_bar_requests = 0
    raw_closes: dict[str, pd.Series] = {}
    empty_raw: list[str] = []
    if spin_symbols:
        if args.raw_closes_cache and Path(args.raw_closes_cache).exists():
            raw_panel = pd.read_parquet(args.raw_closes_cache)
            raw_panel.index = pd.DatetimeIndex(raw_panel.index)
            empty_raw = sorted(set(spin_symbols) - set(raw_panel.columns))
        else:
            session = CountingSession(__import__("requests").Session())
            source = AlpacaBarSource.from_env(session=session, adjustment="raw")
            frames = source.fetch_batch(spin_symbols, "1d", start_date=fetch_start, end_date=fetch_end)
            raw_panel, empty_raw = iex_close_panel(frames)
            n_bar_requests = session.n_requests
            if args.raw_closes_cache:
                raw_panel.to_parquet(args.raw_closes_cache)
        raw_panel.index = pd.DatetimeIndex(raw_panel.index).tz_localize(None).normalize()
        raw_closes = {str(c): raw_panel[c] for c in raw_panel.columns}
        if empty_raw:
            print(f"LOUD: {len(empty_raw)} spin parent/child name(s) returned zero raw bars: {empty_raw}")

    rows = sweep_events(
        panel,
        events,
        raw_closes,
        min_step_bps=args.min_step_bps,
        noise_budget_bps=args.noise_budget_bps,
        max_gap_sessions=args.max_gap_sessions,
    )
    summary, flagged = summarize(rows)
    aptv = aptv_cross_check(rows)
    if not aptv["reproduced"]:
        print(f"LOUD: APTV cross-check NOT reproduced — treat the sweep as buggy until explained: {aptv}")
    dividend_check = local_dividend_cross_check(Path("results"), by_type, set(symbols))

    payload = {
        "meta": {
            "generated": date.today().isoformat(),
            "script": "research/scripts/spin_adjustment_sweep.py",
            "status": (
                "uncounted diagnostic: searches nothing, changes no trial code path, appends "
                "nothing to the trials ledger, moves no ratified statistic"
            ),
            "config": args.config,
            "panel": {
                "n_symbols": len(symbols),
                "n_missing_caches": len(missing_caches),
                "missing_caches": missing_caches,
                "window_effective": [str(window_start.date()), str(window_end.date())],
            },
            "parameters": {
                "min_step_bps": args.min_step_bps,
                "noise_budget_bps": args.noise_budget_bps,
                "max_gap_sessions": args.max_gap_sessions,
                "max_raw_lag_days": MAX_RAW_LAG_DAYS,
            },
            "sources": {
                "corporate_actions_url": CORPORATE_ACTIONS_URL,
                "actions_fetch_window": [fetch_start, fetch_end],
                "records_reused": records_reused,
                "n_http_requests_actions": n_action_requests,
                "n_http_requests_raw_bars": n_bar_requests,
                "n_duplicate_records_dropped": n_duplicates,
                "raw_bars_adjustment": "raw",
                "n_spin_symbols_raw_fetched": len(spin_symbols),
                "spin_symbols_zero_raw_bars": empty_raw,
                "local_dividend_cross_check": dividend_check,
            },
        },
        "coverage": coverage,
        "summary": summary,
        "raw_step_flagged": flagged,
        "aptv_cross_check": aptv,
        "events": rows,
    }
    out_path = Path(args.out) if args.out else Path("results") / f"spin_adjustment_sweep_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(_clean(payload), indent=2))
    compact = {k: v for k, v in payload.items() if k != "events"}
    print(json.dumps(_clean(compact), indent=2))


if __name__ == "__main__":
    main()
