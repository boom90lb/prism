"""Corporate events, adjustments, price basis, and the scoring-basis series (engine spec §3.2, §3.6).

Three things live here, and they are one subject because getting any of them
wrong moves equity silently.

*The event vocabulary.* §3.2 declares twelve ``event_type`` values and a payload
shape for each. :func:`validate_payload` is the single place those shapes are
checked, so a malformed spin-off ratio or a distribution with no amount is
refused at ingest rather than discovered as a wrong number in a ledger.
:class:`Event` and :class:`Adjustment` are the row types, and
:func:`events_frame` / :func:`adjustments_frame` write them in the declared
column order.

*The price basis, and the share change that follows from it.* ``bars`` declares
``price_basis`` per row, ``raw`` or ``split_adjusted_at_ingest`` (decision D3).
The rule that makes the accounting correct is
:func:`applied_share_factor`: on the effective date of a split the engine
multiplies shares by the factor **only** for a ``raw`` series, because a
``split_adjusted_at_ingest`` series already carries the ratio in its prices and
applying both would double-count it. Both vendors in this repository fetch
split-adjusted bars (``prism.io.loader.DataLoader.fetch_historical_data`` passes
``adjust=splits``; ``prism.live.alpaca_data`` defaults to ``adjustment="split"``),
so every imported cache is labelled ``split_adjusted_at_ingest`` and no share
change is applied on a split date for it.

The case that is not a split is a vendor back-rewrite: the vendor restates the
whole series onto a new basis. ``prism.io.store`` already detects it
(overlap closes diverging by more than ``REWRITE_RTOL`` raise
:class:`~prism.io.store.SplitRewriteDetected`) and forces a full refetch. Under
this contract that refetch arrives as a full-series bars revision, and
:func:`record_bars_revision` pairs it with an ``adjustments`` row of
``kind = "rebase"`` carrying the ratio, written by :func:`write_bars_revision`
in one transaction. The accounting kernel rebases held shares by that factor, so
equity is continuous: 100 shares marked at 200.00 is 20,000; after a 4-for-1
rebase the series marks 50.00, shares become 400, and equity is still 20,000.
Without the row the same revision would mark 100 shares at 50.00 for 5,000 — a
quarter of the true value, silently — which is why
:func:`validate_bars_revision` refuses a full-series revision whose closes
diverge beyond ``REWRITE_RTOL`` and that carries no matching adjustment.

*The scoring basis.* A component declares ``scoring_basis``, ``price`` or
``total_return`` (§3.3). :func:`total_return_close` derives the reinvested
series from the as-traded closes and the ``cash_distribution`` events;
:func:`scoring_close` is the resolution a caller applies to a declaration.
Costs, participation, fills, valuation, and every limit are computed on the
as-traded close whatever the scoring basis (§3.6) — nothing here returns a
price anything trades at.

Applying an event to portfolio state is stage 3b. This module is the recording
and derivation side: it says what the event is, what factor it implies against
a declared basis, and what a distribution entitles the holder to. It moves no
cash and holds no state.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.io.manifest import PRICE_BASES
from prism.io.store import REWRITE_RTOL

#: The ``event_type`` vocabulary of §3.2, in the order the specification lists it.
EVENT_TYPES: tuple[str, ...] = (
    "split",
    "cash_distribution",
    "spinoff",
    "ticker_change",
    "delisting_notice",
    "trading_cessation",
    "otc_continuation",
    "share_cancellation",
    "terminal_payout",
    "halt",
    "membership_add",
    "membership_remove",
)

#: ``distribution_kind`` values a ``cash_distribution`` payload may declare.
DISTRIBUTION_KINDS: tuple[str, ...] = ("dividend", "capital_gain", "return_of_capital", "mixed")

#: ``kind`` values of the ``adjustments`` table (§3.2).
ADJUSTMENT_KINDS: tuple[str, ...] = ("split", "reverse_split", "stock_distribution", "rebase")

#: The adjustment kinds that restate the share count against a ratio the vendor
#: may already have applied to the price series. ``rebase`` is deliberately not
#: one of them: a back-rewrite rebases shares whatever the declared basis,
#: because the prices themselves moved.
RATIO_KINDS: tuple[str, ...] = ("split", "reverse_split", "stock_distribution")

#: ``scoring_basis`` values a component may declare (§3.3). Duplicated by value
#: from ``prism.signal.base.SCORING_BASES`` rather than imported: ``prism.io``
#: is below ``prism.signal`` and must not depend on it.
#: ``tests/test_price_basis.py`` asserts the two agree.
SCORING_BASES: tuple[str, ...] = ("price", "total_return")

#: Column order of the ``events`` table (§3.2).
EVENT_COLUMNS: tuple[str, ...] = (
    "sid",
    "event_type",
    "effective_date",
    "announce_date",
    "payload",
    "available_at",
    "ingested_at",
    "source",
    "revision",
)

#: Column order of the ``adjustments`` table (§3.2).
ADJUSTMENT_COLUMNS: tuple[str, ...] = (
    "sid",
    "effective_date",
    "kind",
    "factor",
    "applies_from",
    "available_at",
    "ingested_at",
    "source",
    "revision",
)

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_ET = "America/New_York"


class EventError(ValueError):
    """An event, an adjustment, or a bars revision cannot be recorded as given."""


class PayloadError(EventError):
    """A payload does not match the shape §3.2 declares for its ``event_type``."""


class RebaseNotRecordedError(EventError):
    """A full-series bars revision restates the level and carries no ``rebase`` adjustment.

    Recording it would mark the unchanged share count at the restated price, so
    equity would jump by the rewrite ratio with nothing in the tables saying it
    had. Refusing is the whole point of the ``adjustments`` companion table
    (decision D3).
    """


# ---------------------------------------------------------------------------
# payloads
# ---------------------------------------------------------------------------


def _require_str(payload: Mapping[str, Any], field: str, event_type: str, *, allow_null: bool = False) -> str | None:
    value = payload.get(field)
    if value is None:
        if allow_null:
            return None
        raise PayloadError(f"{event_type} payload requires {field!r}, which is null")
    text = str(value)
    if not text:
        raise PayloadError(f"{event_type} payload requires a non-empty {field!r}")
    return text


def _require_date(payload: Mapping[str, Any], field: str, event_type: str, *, allow_null: bool = False) -> str | None:
    text = _require_str(payload, field, event_type, allow_null=allow_null)
    if text is None:
        return None
    if not _ISO_DATE.match(text):
        raise PayloadError(f"{event_type} payload {field!r} must be an ISO YYYY-MM-DD date, got {text!r}")
    return text


def _require_number(payload: Mapping[str, Any], field: str, event_type: str, *, positive: bool = True) -> float:
    value = payload.get(field)
    if value is None or isinstance(value, bool):
        raise PayloadError(f"{event_type} payload requires a numeric {field!r}, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PayloadError(f"{event_type} payload {field!r} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise PayloadError(f"{event_type} payload {field!r} must be finite, got {value!r}")
    if positive and number <= 0.0:
        raise PayloadError(f"{event_type} payload {field!r} must be positive, got {number}")
    return number


def validate_payload(event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check one payload against the shape §3.2 declares, and return it normalized.

    Normalized means numbers are floats, dates are ISO strings, and the keys the
    specification names are present; keys it does not name are carried through
    unchanged, because a vendor field that costs nothing to keep is provenance.
    """
    if event_type not in EVENT_TYPES:
        raise PayloadError(f"event_type must be one of {EVENT_TYPES}, got {event_type!r}")
    if not isinstance(payload, Mapping):
        raise PayloadError(f"{event_type} payload must be a mapping, got {type(payload).__name__}")
    out = dict(payload)

    if event_type == "split":
        out["factor"] = _require_number(payload, "factor", event_type)
    elif event_type == "cash_distribution":
        amount = _require_number(payload, "amount_per_share", event_type, positive=False)
        if amount < 0.0:
            raise PayloadError(f"cash_distribution amount_per_share must not be negative, got {amount}")
        out["amount_per_share"] = amount
        out["currency"] = _require_str(payload, "currency", event_type)
        out["pay_date"] = _require_date(payload, "pay_date", event_type, allow_null=True)
        kind = _require_str(payload, "distribution_kind", event_type)
        if kind not in DISTRIBUTION_KINDS:
            raise PayloadError(
                f"cash_distribution distribution_kind must be one of {DISTRIBUTION_KINDS}, got {kind!r}"
            )
        out["distribution_kind"] = kind
    elif event_type == "spinoff":
        out["child_sid"] = _require_str(payload, "child_sid", event_type, allow_null=True)
        out["ratio"] = _require_number(payload, "ratio", event_type)
        if "cash_in_lieu" not in payload:
            raise PayloadError("spinoff payload requires 'cash_in_lieu'")
        out["cash_in_lieu"] = bool(payload["cash_in_lieu"])
    elif event_type == "ticker_change":
        out["from_ticker"] = _require_str(payload, "from_ticker", event_type)
        out["to_ticker"] = _require_str(payload, "to_ticker", event_type)
    elif event_type == "delisting_notice":
        out["reason"] = _require_str(payload, "reason", event_type)
        out["expected_last_session"] = _require_date(payload, "expected_last_session", event_type)
    elif event_type == "trading_cessation":
        out["last_session"] = _require_date(payload, "last_session", event_type)
    elif event_type == "otc_continuation":
        out["venue"] = _require_str(payload, "venue", event_type)
        out["vendor_symbol"] = _require_str(payload, "vendor_symbol", event_type)
    elif event_type == "share_cancellation":
        fraction = _require_number(payload, "fraction_cancelled", event_type)
        if fraction > 1.0:
            raise PayloadError(f"share_cancellation fraction_cancelled must be at most 1.0, got {fraction}")
        out["fraction_cancelled"] = fraction
    elif event_type == "terminal_payout":
        # Two declared shapes: a known payout, or an explicit statement that the
        # terminal value is unknown. The second is not an empty payload — a
        # claim whose value nobody knows is recorded as incomplete (§3.6), and
        # only a payload that says so lets a record carry that flag honestly.
        if str(payload.get("status", "")) == "unknown":
            out = {"status": "unknown"}
        else:
            # Zero is a real terminal value (a wipeout); negative is not.
            payout = _require_number(payload, "amount_per_share", event_type, positive=False)
            if payout < 0.0:
                raise PayloadError(f"terminal_payout amount_per_share must not be negative, got {payout}")
            out["amount_per_share"] = payout
            out["currency"] = _require_str(payload, "currency", event_type)
            out["pay_date"] = _require_date(payload, "pay_date", event_type, allow_null=True)
    elif event_type == "halt":
        out["start"] = _require_str(payload, "start", event_type)
        out["end"] = _require_str(payload, "end", event_type, allow_null=True)
        out["reason"] = _require_str(payload, "reason", event_type)
    else:  # membership_add, membership_remove
        out["index_id"] = _require_str(payload, "index_id", event_type)
    return out


def _as_iso_date(value: str | dt.date | pd.Timestamp) -> str:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    text = str(value)
    if not _ISO_DATE.match(text[:10]):
        raise EventError(f"expected an ISO YYYY-MM-DD date, got {value!r}")
    return text[:10]


# ---------------------------------------------------------------------------
# row types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    """One row of the ``events`` table (§3.2), with its payload validated."""

    sid: str
    event_type: str
    effective_date: str
    payload: dict[str, Any]
    ingested_at: str
    source: str
    revision: int = 1
    announce_date: str | None = None
    available_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective_date", _as_iso_date(self.effective_date))
        if self.announce_date is not None:
            object.__setattr__(self, "announce_date", _as_iso_date(self.announce_date))
        object.__setattr__(self, "payload", validate_payload(self.event_type, self.payload))
        if int(self.revision) < 1:
            raise EventError(f"revision must be a positive integer, got {self.revision}")
        object.__setattr__(self, "revision", int(self.revision))

    def to_row(self) -> dict[str, Any]:
        """The row as the table stores it: the payload as canonical JSON text."""
        return {
            "sid": self.sid,
            "event_type": self.event_type,
            "effective_date": self.effective_date,
            "announce_date": self.announce_date,
            "payload": json.dumps(self.payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "available_at": self.available_at,
            "ingested_at": self.ingested_at,
            "source": self.source,
            "revision": self.revision,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Event":
        return cls(
            sid=str(row["sid"]),
            event_type=str(row["event_type"]),
            effective_date=str(row["effective_date"]),
            payload=decode_payload(row.get("payload")),
            ingested_at=str(row["ingested_at"]),
            source=str(row["source"]),
            revision=int(row.get("revision", 1)),
            announce_date=_optional_text(row.get("announce_date")),
            available_at=_optional_text(row.get("available_at")),
        )


@dataclass(frozen=True, slots=True)
class Adjustment:
    """One row of the ``adjustments`` table (§3.2).

    ``applies_from`` is the first ``ts`` the new basis covers. For a ``split``
    it is the effective date; for a ``rebase`` it is the first bar of the
    rewritten series, because the rewrite restated history back to there.
    """

    sid: str
    effective_date: str
    kind: str
    factor: float
    applies_from: str
    ingested_at: str
    source: str
    revision: int = 1
    available_at: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ADJUSTMENT_KINDS:
            raise EventError(f"adjustment kind must be one of {ADJUSTMENT_KINDS}, got {self.kind!r}")
        factor = float(self.factor)
        if not math.isfinite(factor) or factor <= 0.0:
            raise EventError(f"adjustment factor must be finite and positive, got {self.factor!r}")
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "effective_date", _as_iso_date(self.effective_date))
        object.__setattr__(self, "applies_from", _as_iso_date(self.applies_from))
        if int(self.revision) < 1:
            raise EventError(f"revision must be a positive integer, got {self.revision}")
        object.__setattr__(self, "revision", int(self.revision))

    def to_row(self) -> dict[str, Any]:
        return {
            "sid": self.sid,
            "effective_date": self.effective_date,
            "kind": self.kind,
            "factor": self.factor,
            "applies_from": self.applies_from,
            "available_at": self.available_at,
            "ingested_at": self.ingested_at,
            "source": self.source,
            "revision": self.revision,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Adjustment":
        return cls(
            sid=str(row["sid"]),
            effective_date=str(row["effective_date"]),
            kind=str(row["kind"]),
            factor=float(row["factor"]),
            applies_from=str(row["applies_from"]),
            ingested_at=str(row["ingested_at"]),
            source=str(row["source"]),
            revision=int(row.get("revision", 1)),
            available_at=_optional_text(row.get("available_at")),
        )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    return text or None


def decode_payload(value: Any) -> dict[str, Any]:
    """A payload cell as a dict, whether the table stored JSON text or a mapping."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value).strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PayloadError(f"payload is not JSON: {text[:120]!r}") from exc
    if not isinstance(loaded, dict):
        raise PayloadError(f"payload must decode to an object, got {type(loaded).__name__}")
    return loaded


def _rows_frame(
    rows: Sequence[Mapping[str, Any]], columns: tuple[str, ...], key: tuple[str, ...], what: str
) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=list(columns))
    if frame.empty:
        return frame
    duplicated = frame.duplicated(subset=list(key), keep=False)
    if bool(duplicated.any()):
        offending = frame.loc[duplicated, list(key)].drop_duplicates().head(3).to_dict("records")
        raise EventError(
            f"{what} carries more than one row for {offending}; two rows with one (key, source, revision) are an "
            "ingestion defect and the view refuses to read them (§3.2)"
        )
    return frame.reset_index(drop=True)


def events_frame(events: Iterable[Event]) -> pd.DataFrame:
    """The ``events`` table for ``events``, in the declared column order."""
    rows = [event.to_row() for event in events]
    return _rows_frame(rows, EVENT_COLUMNS, ("sid", "event_type", "effective_date", "source", "revision"), "events")


def adjustments_frame(adjustments: Iterable[Adjustment]) -> pd.DataFrame:
    """The ``adjustments`` table for ``adjustments``, in the declared column order."""
    rows = [adjustment.to_row() for adjustment in adjustments]
    return _rows_frame(
        rows, ADJUSTMENT_COLUMNS, ("sid", "effective_date", "kind", "source", "revision"), "adjustments"
    )


# ---------------------------------------------------------------------------
# price basis and the share change it implies
# ---------------------------------------------------------------------------


def require_price_basis(price_basis: str) -> str:
    if price_basis not in PRICE_BASES:
        raise EventError(f"price_basis must be one of {PRICE_BASES}, got {price_basis!r}")
    return price_basis


def series_price_basis(bars: pd.DataFrame) -> str:
    """The single ``price_basis`` a bar series declares; a mixed series is refused.

    The engine reads the basis from the series and never infers it from the data
    (§3.6), so a file that carries two bases has no answer to "does a split move
    shares here" and must not be given one by majority vote.
    """
    if "price_basis" not in bars.columns:
        raise EventError("bar series declares no price_basis column; §3.2 requires one on every row")
    declared = sorted({str(value) for value in bars["price_basis"].dropna().unique()})
    if not declared:
        raise EventError("bar series declares no price_basis on any row")
    if len(declared) > 1:
        raise EventError(
            f"bar series mixes price bases {declared}; one series carries one basis, and a split applies a share "
            "change under exactly one of them (§3.6)"
        )
    return require_price_basis(declared[0])


def applied_share_factor(kind: str, price_basis: str, factor: float) -> float:
    """The multiplier on held shares that ``kind`` implies against ``price_basis`` (§3.6, D3).

    ``raw`` prices are as printed, so a ratio event moves shares by ``factor``.
    ``split_adjusted_at_ingest`` prices already carry the ratio, so the same
    event moves shares by ``1.0``: applying both would double-count it, which is
    the error decision D3 exists to prevent.

    ``rebase`` is not a ratio event but a restatement of the whole series, so it
    moves shares by ``factor`` under either basis — the prices themselves moved,
    and the share count follows them or equity jumps.
    """
    require_price_basis(price_basis)
    if kind not in ADJUSTMENT_KINDS:
        raise EventError(f"adjustment kind must be one of {ADJUSTMENT_KINDS}, got {kind!r}")
    value = float(factor)
    if not math.isfinite(value) or value <= 0.0:
        raise EventError(f"factor must be finite and positive, got {factor!r}")
    if kind not in RATIO_KINDS:  # "rebase"
        return value
    return value if price_basis == "raw" else 1.0


def applied_price_factor(kind: str, price_basis: str, factor: float) -> float:
    """The multiplier on a carried reference price, the inverse of the share factor.

    A reference price recorded before a ratio event (the last decision close a
    live book carries, say) is divided by the factor on the effective date of a
    split against a ``raw`` series, and left alone against an adjusted one.
    ``share_factor * price_factor == 1`` by construction, which is the statement
    that neither leg changes equity.
    """
    return 1.0 / applied_share_factor(kind, price_basis, factor)


# ---------------------------------------------------------------------------
# vendor back-rewrites
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RewriteRatio:
    """What comparing a stored series with a revised one shows.

    ``factor`` is the common ratio ``previous / revised`` when the revision
    restated the whole level by one ratio, and ``None`` otherwise — either
    because nothing diverged beyond ``rtol`` or because the divergence is not
    one ratio, which is a source correction and not a rebase (§3.2, worked
    example B).

    ``full_series`` says the revision covers every stored timestamp, and only a
    full-series restatement is a rebase. A ratio that appears on the *tail* of a
    series is the market, not a rewrite: a raw series prints half its old level
    from a split's effective date onward, and rebasing shares for that would
    double-count the very split the price already carries.
    """

    factor: float | None
    max_divergence: float
    n_overlap: int
    n_diverging: int
    full_series: bool

    @property
    def is_rebase(self) -> bool:
        return self.factor is not None and self.full_series


def _close_series(bars: pd.DataFrame, what: str) -> pd.Series:
    if "ts" not in bars.columns or "close" not in bars.columns:
        raise EventError(f"{what} must carry 'ts' and 'close' columns")
    stamps = pd.DatetimeIndex(pd.to_datetime(bars["ts"], errors="raise", utc=True)).tz_convert(_ET)
    series = pd.Series(pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float), index=stamps)
    if series.index.has_duplicates:
        raise EventError(f"{what} carries more than one row for one ts; a revision is a replacement, not an append")
    return series.sort_index()


def rewrite_ratio(previous: pd.DataFrame, revised: pd.DataFrame, *, rtol: float = REWRITE_RTOL) -> RewriteRatio:
    """Compare a stored bar series with a revised one and classify the difference.

    ``rtol`` is ``prism.io.store.REWRITE_RTOL``, the tolerance the incremental
    store already uses to separate float jitter (~1e-12 on a stable series) from
    a vendor rewrite (an integer-ish ratio). The comparison is on closes over
    the overlapping timestamps, which is the same comparison
    ``IncrementalBarStore.append_tail`` makes.
    """
    old = _close_series(previous, "the stored series")
    new = _close_series(revised, "the revised series")
    overlap = old.index.intersection(new.index)
    both = old.reindex(overlap).notna() & new.reindex(overlap).notna()
    overlap = overlap[both.to_numpy()]
    if len(overlap) == 0:
        return RewriteRatio(None, 0.0, 0, 0, False)
    old_values = old.reindex(overlap).to_numpy(dtype=float)
    new_values = new.reindex(overlap).to_numpy(dtype=float)
    relative = np.abs(old_values - new_values) / np.clip(np.abs(old_values), 1e-12, None)
    diverging = int((relative > rtol).sum())
    covers_stored = bool(old.index.isin(new.index).all())
    if diverging == 0:
        return RewriteRatio(None, float(relative.max()), len(overlap), 0, covers_stored)
    ratios = old_values / np.where(new_values == 0.0, np.nan, new_values)
    if not np.isfinite(ratios).all():
        return RewriteRatio(None, float(relative.max()), len(overlap), diverging, covers_stored)
    factor = float(np.median(ratios))
    spread = float(np.max(np.abs(ratios - factor) / max(abs(factor), 1e-12)))
    if spread > rtol or diverging != len(overlap):
        # Some rows moved and others did not, or they moved by different
        # ratios: a source correction, which touches no share count.
        return RewriteRatio(None, float(relative.max()), len(overlap), diverging, covers_stored)
    return RewriteRatio(factor, float(relative.max()), len(overlap), diverging, covers_stored)


def matching_rebase(
    adjustments: Iterable[Adjustment] | pd.DataFrame, sid: str, factor: float, *, rtol: float = REWRITE_RTOL
) -> Adjustment | None:
    """The ``rebase`` row for ``sid`` whose factor matches ``factor``, or ``None``."""
    rows: list[Adjustment]
    if isinstance(adjustments, pd.DataFrame):
        rows = [Adjustment.from_row(row) for row in adjustments.to_dict("records")]
    else:
        rows = list(adjustments)
    for adjustment in rows:
        if adjustment.sid != sid or adjustment.kind != "rebase":
            continue
        if abs(adjustment.factor - factor) <= rtol * max(abs(factor), 1.0):
            return adjustment
    return None


def validate_bars_revision(
    previous: pd.DataFrame,
    revised: pd.DataFrame,
    adjustments: Iterable[Adjustment] | pd.DataFrame,
    *,
    sid: str,
    rtol: float = REWRITE_RTOL,
) -> RewriteRatio:
    """Refuse a full-series revision that restates the level with no ``rebase`` row (§3.2).

    Returns the classification when the revision is admissible, so a caller can
    record what it saw. A revision that diverges without a common ratio is a
    source correction and needs no adjustment: shares are untouched, which is
    the worked example B case (a 10 percent correction on one bar).
    """
    ratio = rewrite_ratio(previous, revised, rtol=rtol)
    if not ratio.is_rebase:
        return ratio
    factor = float(ratio.factor or 0.0)
    if matching_rebase(adjustments, sid, factor, rtol=rtol) is None:
        raise RebaseNotRecordedError(
            f"{sid}: the revision restates every one of {ratio.n_overlap} overlapping closes by a common factor of "
            f"{factor:.6g} (max divergence {ratio.max_divergence:.2e} > {rtol:.0e}) and carries no matching "
            "adjustments row with kind='rebase'. Recording it would mark the unchanged share count at the restated "
            "price and move equity by the ratio, silently (§3.2, decision D3). Write the rebase row in the same "
            "transaction, or record the revision as a correction with no common ratio."
        )
    return ratio


@dataclass(frozen=True, slots=True)
class BarsRevision:
    """A full-series bars revision and the adjustment that keeps equity continuous."""

    sid: str
    bars: pd.DataFrame
    adjustment: Adjustment | None
    ratio: RewriteRatio
    price_basis: str

    @property
    def factor(self) -> float | None:
        return self.ratio.factor

    def share_factor(self) -> float:
        """The multiplier stage 3b applies to held shares when this revision lands."""
        if self.adjustment is None:
            return 1.0
        return applied_share_factor(self.adjustment.kind, self.price_basis, self.adjustment.factor)


def record_bars_revision(
    previous: pd.DataFrame,
    revised: pd.DataFrame,
    *,
    sid: str,
    source: str,
    ingested_at: str,
    effective_date: str | dt.date,
    available_at: str | None = None,
    revision: int | None = None,
    rtol: float = REWRITE_RTOL,
) -> BarsRevision:
    """Stamp a refetched series as the next revision, with its ``rebase`` row when one is due.

    ``previous`` is the stored series for ``sid`` (the ``bars`` schema of §3.2);
    ``revised`` is the full refetch that
    :class:`~prism.io.store.SplitRewriteDetected` forced. The declared
    ``price_basis`` is carried from the stored series: a rewrite restates the
    level, not the basis, and inventing a basis here would decide the share
    question by accident.

    ``applies_from`` on the adjustment is the first ``ts`` the revised series
    covers, because the rewrite restated history back to there.
    """
    if previous.empty:
        raise EventError(f"{sid}: a revision needs a stored series to revise")
    price_basis = series_price_basis(previous)
    ratio = rewrite_ratio(previous, revised, rtol=rtol)
    stamped = revised.copy()
    stamps = pd.DatetimeIndex(pd.to_datetime(stamped["ts"], errors="raise", utc=True)).tz_convert(_ET)
    stamped["ts"] = stamps
    stamped["price_basis"] = price_basis
    stamped["source"] = source
    stamped["ingested_at"] = ingested_at
    stamped["available_at"] = available_at
    if revision is None:
        stored = pd.to_numeric(previous.get("revision", pd.Series([1])), errors="coerce")
        revision = int(stored.max()) + 1 if len(stored) else 2
    stamped["revision"] = int(revision)
    stamped = stamped.sort_values("ts").reset_index(drop=True)

    adjustment: Adjustment | None = None
    if ratio.is_rebase:
        adjustment = Adjustment(
            sid=sid,
            effective_date=_as_iso_date(effective_date),
            kind="rebase",
            factor=float(ratio.factor or 0.0),
            applies_from=_as_iso_date(stamps.min()),
            available_at=available_at,
            ingested_at=ingested_at,
            source=source,
            revision=int(revision),
        )
    return BarsRevision(sid=sid, bars=stamped, adjustment=adjustment, ratio=ratio, price_basis=price_basis)


def write_bars_revision(
    revision: BarsRevision, *, bars_path: Path, adjustments_path: Path
) -> tuple[Path, Path | None]:
    """Append the revision and its rebase row in one transaction (§3.2, D3).

    Both files are written to temporaries before either replaces its target, and
    the previous bars file is kept until the adjustment lands: a failure after
    the bars replace restores it, so the workspace never holds a restated price
    series whose share rebase was never recorded. The refusal comes first — a
    rebase revision with no adjustment is not written at all.
    """
    bars_path = Path(bars_path)
    adjustments_path = Path(adjustments_path)
    if revision.ratio.is_rebase and revision.adjustment is None:
        raise RebaseNotRecordedError(
            f"{revision.sid}: a full-series restatement by {revision.ratio.factor} carries no rebase adjustment"
        )
    stored = pd.read_parquet(bars_path) if bars_path.is_file() else pd.DataFrame(columns=revision.bars.columns)
    merged = pd.concat([stored, revision.bars], ignore_index=True) if len(stored) else revision.bars.copy()
    bars_tmp = bars_path.with_suffix(bars_path.suffix + ".tmp")
    merged.to_parquet(bars_tmp, index=False)

    if revision.adjustment is None:
        os.replace(bars_tmp, bars_path)
        return bars_path, None

    existing = pd.read_parquet(adjustments_path) if adjustments_path.is_file() else None
    row = pd.DataFrame([revision.adjustment.to_row()], columns=list(ADJUSTMENT_COLUMNS))
    table = pd.concat([existing, row], ignore_index=True) if existing is not None and len(existing) else row
    adjustments_tmp = adjustments_path.with_suffix(adjustments_path.suffix + ".tmp")
    table.to_parquet(adjustments_tmp, index=False)

    backup = bars_path.with_suffix(bars_path.suffix + ".prev")
    had_bars = bars_path.is_file()
    if had_bars:
        os.replace(bars_path, backup)
    try:
        os.replace(bars_tmp, bars_path)
        os.replace(adjustments_tmp, adjustments_path)
    except OSError:
        # The prices were restated and the rebase row did not land. Put the
        # stored series back: a workspace with neither is recoverable, one with
        # a restated series and no adjustment marks held shares at the wrong
        # level with nothing saying so.
        if had_bars:
            os.replace(backup, bars_path)
        bars_tmp.unlink(missing_ok=True)
        adjustments_tmp.unlink(missing_ok=True)
        raise
    if had_bars:
        backup.unlink(missing_ok=True)
    return bars_path, adjustments_path


# ---------------------------------------------------------------------------
# terminal claims
# ---------------------------------------------------------------------------

#: What the events table says about a holder's claim on an instrument (§3.2, §3.6).
CLAIM_STATUSES: tuple[str, ...] = ("trading", "open", "unresolved", "resolved")

#: The event kinds that end a series. A notice alone does not: it announces an
#: expected last session, and until the cessation lands the instrument trades.
CESSATION_TYPES: tuple[str, ...] = ("trading_cessation", "otc_continuation")


@dataclass(frozen=True, slots=True)
class TerminalClaim:
    """What is known about a holder's claim on one sid, read from the events table.

    Four states, and the difference between the middle two is what §3.2 means
    by "a strict claim that needs an unknown terminal value is recorded as
    incomplete":

    ``trading``
        No cessation is visible. The instrument prints and the position marks
        at its bar.
    ``open``
        Trading ceased and no ``terminal_payout`` has been recorded at all. The
        holder has a claim of unknown value.
    ``unresolved``
        A ``terminal_payout`` exists and says ``{"status": "unknown"}``: the
        payer has stated that the value is not yet known. The claim is
        unresolved for the same reason and with better provenance.
    ``resolved``
        A ``terminal_payout`` carries ``amount_per_share``. :meth:`cash_for`
        converts a share count to cash. Zero is a resolved value — a wipeout is
        a known outcome, not an unknown one.

    ``continues_at`` is set when an ``otc_continuation`` moved the instrument to
    another venue: the claim is not terminal, the series continues under a new
    vendor symbol, and treating it as terminal would write off a live position.

    Nothing here moves cash or touches a position. Stage 3b's
    ``execution/accounting.py`` applies the transition; this is the read that
    tells it, and the result record, which of the four cases holds.
    """

    sid: str
    status: Literal["trading", "open", "unresolved", "resolved"]
    last_session: str | None = None
    reason: str | None = None
    effective_date: str | None = None
    amount_per_share: float | None = None
    currency: str | None = None
    pay_date: str | None = None
    pay_date_assumed: bool = False
    continues_at: str | None = None

    def __post_init__(self) -> None:
        if self.status not in CLAIM_STATUSES:
            raise EventError(f"claim status must be one of {CLAIM_STATUSES}, got {self.status!r}")

    @property
    def complete(self) -> bool:
        """False exactly when a record built on this claim must be written ``incomplete``."""
        return self.status in ("trading", "resolved")

    @property
    def valuation_status(self) -> Literal["priced", "unresolved_claim"]:
        """The §3.6 ``Position.valuation_status`` this claim implies, ignoring bar staleness.

        ``stale`` is the other value and is a fact about bars, not about
        claims: :meth:`prism.io.dataset.DatasetView.marks` reports it. A caller
        composes the two — a stale mark on an unresolved claim is an
        ``unresolved_claim``, because the claim is the stronger statement.
        """
        return "priced" if self.complete else "unresolved_claim"

    def incomplete_reason(self) -> str | None:
        """The sentence a result record's ``incomplete_reasons`` carries, or ``None``."""
        if self.complete:
            return None
        if self.status == "unresolved":
            return f"{self.sid}: terminal_payout recorded with status 'unknown' as of {self.effective_date}"
        return (
            f"{self.sid}: trading ceased {self.last_session} with no terminal_payout recorded"
            if self.last_session
            else f"{self.sid}: delisting noticed with no terminal_payout recorded"
        )

    def cash_for(self, shares: float) -> float:
        """The cash a signed share count converts to, refusing an unresolved claim.

        Fail loud (SPEC N7): returning 0.0 for an unknown terminal value would
        write the position off at nothing and read, in every downstream total,
        exactly like a resolved wipeout.
        """
        if self.status != "resolved" or self.amount_per_share is None:
            raise EventError(
                f"{self.sid}: the terminal value is {self.status}, so no cash amount exists. A strict claim on it is "
                "recorded incomplete (§3.2); it is never settled at zero, which would be indistinguishable from a "
                "resolved wipeout"
            )
        quantity = float(shares)
        if not math.isfinite(quantity):
            raise EventError(f"{self.sid}: share count must be finite, got {shares!r}")
        return quantity * float(self.amount_per_share)


def terminal_claim(events: pd.DataFrame | Sequence[Event], sid: str) -> TerminalClaim:
    """Read one sid's claim state out of visible ``events`` rows.

    ``events`` is whatever the caller can see — pass a
    :meth:`prism.io.dataset.DatasetView.events` frame and the answer is as of
    that decision, which is the point: a payout announced after the decision
    must not resolve a claim the decision could not have known was resolved.

    Later effective dates win within a kind, so a payout that first landed as
    ``{"status": "unknown"}`` and was later amended to an amount resolves.
    """
    rows: list[Event]
    if isinstance(events, pd.DataFrame):
        frame = events if events.empty else events.loc[events["sid"].astype(str) == str(sid)]
        rows = [Event.from_row(row) for row in frame.to_dict("records")] if not frame.empty else []
    else:
        rows = [event for event in events if event.sid == sid]
    rows.sort(key=lambda event: (event.effective_date, event.revision))

    notice = _last_of(rows, "delisting_notice")
    cessation = _last_of(rows, "trading_cessation")
    continuation = _last_of(rows, "otc_continuation")
    payout = _last_of(rows, "terminal_payout")
    last_session = str(cessation.payload["last_session"]) if cessation is not None else None
    reason = str(notice.payload["reason"]) if notice is not None else None
    venue = str(continuation.payload["venue"]) if continuation is not None else None

    if payout is not None:
        if str(payout.payload.get("status", "")) == "unknown":
            return TerminalClaim(
                sid=str(sid),
                status="unresolved",
                last_session=last_session,
                reason=reason,
                effective_date=payout.effective_date,
                continues_at=venue,
            )
        pay_date = payout.payload.get("pay_date")
        return TerminalClaim(
            sid=str(sid),
            status="resolved",
            last_session=last_session,
            reason=reason,
            effective_date=payout.effective_date,
            amount_per_share=float(payout.payload["amount_per_share"]),
            currency=str(payout.payload.get("currency", "USD")),
            pay_date=None if pay_date is None else str(pay_date),
            pay_date_assumed=pay_date is None,
            continues_at=venue,
        )
    if cessation is not None or notice is not None:
        return TerminalClaim(
            sid=str(sid),
            status="open",
            last_session=last_session,
            reason=reason,
            effective_date=(cessation or notice).effective_date if (cessation or notice) else None,
            continues_at=venue,
        )
    return TerminalClaim(sid=str(sid), status="trading", continues_at=venue)


def _last_of(rows: Sequence[Event], event_type: str) -> Event | None:
    matches = [row for row in rows if row.event_type == event_type]
    return matches[-1] if matches else None


def terminal_claims(events: pd.DataFrame | Sequence[Event], sids: Iterable[str]) -> dict[str, TerminalClaim]:
    """:func:`terminal_claim` for several sids, in the order asked."""
    return {str(sid): terminal_claim(events, str(sid)) for sid in sids}


def incomplete_reasons(claims: Mapping[str, TerminalClaim] | Iterable[TerminalClaim]) -> tuple[str, ...]:
    """The ``incomplete_reasons`` a result record carries when any claim is unresolved (§3.12).

    Empty exactly when every claim is complete, so a caller writes
    ``status = "finished"`` on an empty tuple and ``"incomplete"`` otherwise
    without a second rule.
    """
    values = claims.values() if isinstance(claims, Mapping) else claims
    return tuple(reason for reason in (claim.incomplete_reason() for claim in values) if reason is not None)


# ---------------------------------------------------------------------------
# cash distributions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DistributionEntitlement:
    """What one ``cash_distribution`` entitles a holder to (§3.6).

    ``shares`` is the signed share count at the close of the bar *before* the
    ex-date, which is the canonical entitlement rule: a buyer filled at the
    ex-date open is not entitled and a seller filled at the ex-date open is.
    ``amount`` is signed the same way, so a short position produces a negative
    amount and a ``liability`` leg. Stage 3b books it: on the ex-date to
    ``receivables`` or ``liabilities`` with no cash movement, and on
    ``pay_date`` to cash.
    """

    sid: str
    ex_date: str
    pay_date: str | None
    pay_date_assumed: bool
    amount_per_share: float
    currency: str
    shares: float
    amount: float
    leg: Literal["receivable", "liability", "none"]


def distribution_entitlement(event: Event | Mapping[str, Any], shares: float) -> DistributionEntitlement:
    """The entitlement ``shares`` held into ``event``'s ex-date creates.

    ``shares`` is the signed count at the close of the bar before the ex-date.
    No cash moves here: this is the recording side, and the ledger transitions
    are stage 3b.
    """
    row = event if isinstance(event, Event) else Event.from_row(event)
    if row.event_type != "cash_distribution":
        raise EventError(f"{row.sid}: distribution_entitlement needs a cash_distribution, got {row.event_type!r}")
    amount_per_share = float(row.payload["amount_per_share"])
    quantity = float(shares)
    if not math.isfinite(quantity):
        raise EventError(f"{row.sid}: share count at the ex-date must be finite, got {shares!r}")
    amount = quantity * amount_per_share
    pay_date = row.payload.get("pay_date")
    leg: Literal["receivable", "liability", "none"] = "none"
    if amount > 0.0:
        leg = "receivable"
    elif amount < 0.0:
        leg = "liability"
    return DistributionEntitlement(
        sid=row.sid,
        ex_date=row.effective_date,
        pay_date=None if pay_date is None else str(pay_date),
        pay_date_assumed=pay_date is None,
        amount_per_share=amount_per_share,
        currency=str(row.payload.get("currency", "USD")),
        shares=quantity,
        amount=amount,
        leg=leg,
    )


def distribution_amounts(events: pd.DataFrame | Sequence[Event], close: pd.DataFrame) -> pd.DataFrame:
    """Per-share cash on the ex-date session, shaped like ``close`` and zero elsewhere.

    The conventions, which are the recorded trend driver's and are duplicated
    here by value with this citation rather than imported
    (``research/arbitrage/trend_walk_forward.py:425-486``; ``prism`` imports no
    research module, SPEC §7):

    * an ex-date that is not a session on ``close``'s calendar credits the next
      session, which is the same forward roll the availability derivation makes;
    * an ex-date after the last session is dropped — the amount is not knowable
      inside this panel and crediting it to the last bar would forge a date;
    * an ex-date before a column's first finite close is dropped, because there
      is no position in a name this dataset does not yet price;
    * two distributions on one session sum.

    A negative or non-finite amount raises: a distribution that takes cash out
    of a long is a defect in the source, not a number to net off (N7).
    """
    if isinstance(events, pd.DataFrame):
        # Filter before parsing: a scoring call answers about distributions, and
        # a halt or a spin-off whose payload this table stores differently is
        # not its business to refuse.
        frame_rows = events.loc[events["event_type"] == "cash_distribution"] if not events.empty else events
        rows = [] if frame_rows.empty else [Event.from_row(row) for row in frame_rows.to_dict("records")]
    else:
        rows = list(events)
    index = pd.DatetimeIndex(close.index)
    frame = pd.DataFrame(0.0, index=index, columns=list(close.columns), dtype=float)
    if not len(index):
        return frame
    if not index.is_monotonic_increasing or not index.is_unique:
        raise EventError("the close panel's index must be sorted and unique before distributions are placed on it")
    positions = {str(sid): number for number, sid in enumerate(close.columns)}
    first_valid = {str(sid): close[sid].first_valid_index() for sid in close.columns}
    # Accumulation needs an owned buffer; pandas copy-on-write views are read-only.
    values = frame.to_numpy(dtype=float, copy=True)
    for event in rows:
        if event.event_type != "cash_distribution":
            continue
        column = positions.get(event.sid)
        if column is None:
            continue
        amount = float(event.payload["amount_per_share"])
        if not math.isfinite(amount) or amount < 0.0:
            raise EventError(f"{event.sid} {event.effective_date}: distribution amount {amount!r} is not usable")
        stamp = (
            pd.Timestamp(event.effective_date, tz=index.tz)
            if index.tz is not None
            else pd.Timestamp(event.effective_date)
        )
        first = first_valid[event.sid]
        if first is None or stamp < first:
            continue
        position = int(index.searchsorted(stamp))
        if position >= len(index):
            continue
        values[position, column] += amount
    return pd.DataFrame(values, index=index, columns=list(close.columns))


# ---------------------------------------------------------------------------
# scoring basis
# ---------------------------------------------------------------------------


def total_return_close(close: pd.DataFrame, distributions: pd.DataFrame) -> pd.DataFrame:
    """The as-traded closes reinvested at each ex-date: the ``total_return`` scoring panel.

    Per column ``TR_t = TR_{t-1} * (close_t + div_t) / close_{t-1}``, rebased to
    ``TR_first = close_first`` and NaN before the first finite close, so
    ``TR_t / close_t`` equals ``prod_{ex <= t} (1 + div_ex / close_ex)``:
    reinvestment at the ex-date close. This is the recorded trend convention
    ``split_adjusted_total_return_dividends_as_cash``
    (``results/trend_t0/config.json``), duplicated by value from
    ``research/arbitrage/trend_walk_forward.py:494-511`` with this citation
    rather than imported (SPEC §7).

    Scoring only. Costs, participation, fills, valuation, and every limit are
    computed on the as-traded close whatever a component's ``scoring_basis``
    (§3.6): nothing trades at this level.
    """
    prices = close.apply(pd.to_numeric, errors="coerce").astype(float)
    cash = distributions.reindex(index=close.index, columns=close.columns)
    cash = cash.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    ratio = (prices + cash) / prices.shift(1)
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
    for column in close.columns:
        first = prices[column].first_valid_index()
        if first is None:
            continue
        steps = ratio[column].loc[first:].copy()
        steps.loc[first] = 1.0
        out.loc[first:, column] = float(prices.at[first, column]) * steps.cumprod()
    return out


def scoring_close(scoring_basis: str, close: pd.DataFrame, distributions: pd.DataFrame | None = None) -> pd.DataFrame:
    """The close panel a component's declared ``scoring_basis`` names (§3.3).

    ``price`` is the as-traded close, handed back unchanged. ``total_return`` is
    the reinvested series, which needs the distributions: passing ``None`` for
    them is refused rather than silently scoring a total-return component on a
    price series, because the two agree exactly on a name that never paid and
    diverge without a trace on one that did.
    """
    if scoring_basis not in SCORING_BASES:
        raise EventError(f"scoring_basis must be one of {SCORING_BASES}, got {scoring_basis!r}")
    if scoring_basis == "price":
        return close
    if distributions is None:
        raise EventError(
            "scoring_basis='total_return' needs the cash_distribution amounts; without them the panel would be the "
            "price series under a total-return label (§3.3)"
        )
    return total_return_close(close, distributions)


# ---------------------------------------------------------------------------
# issuer distribution histories
# ---------------------------------------------------------------------------

#: The source id §3.2 gives the issuer-hosted distribution histories in
#: ``data/distributions``. They are not redistributable; the manifest carries
#: their url and sha256 and a ``redistributable = false`` flag.
ISSUER_SOURCE = "issuer_csv"


def issuer_distribution_events(
    directory: Path | str,
    sid: str,
    *,
    symbol: str | None = None,
    source: str = ISSUER_SOURCE,
    distribution_kind: str = "mixed",
    currency: str = "USD",
) -> list[Event]:
    """``cash_distribution`` events from ``{SYMBOL}.csv`` and ``{SYMBOL}.provenance.json`` (§3.2).

    The mapping is the one the legacy import declares: ``effective_date`` from
    the CSV ``ex_date``, ``payload`` from ``amount``, ``pay_date`` and the
    provenance ``sha256``, ``ingested_at`` from the provenance ``captured_at``
    (the only data in this workspace with a genuine capture timestamp),
    ``available_at`` empty, ``revision = 1``. ``distribution_kind`` is
    ``mixed`` because the issuer files sum dividend and capital-gain columns
    into one amount, which the provenance records.

    Both files are required. A missing provenance file is a refusal, not a
    default: an amount with no capture timestamp and no source hash is a number
    with no origin.
    """
    root = Path(directory)
    name = symbol or sid
    csv_path = root / f"{name}.csv"
    provenance_path = root / f"{name}.provenance.json"
    for path in (csv_path, provenance_path):
        if not path.is_file():
            raise EventError(
                f"{name}: {path} is absent; a distribution history needs both the CSV and its provenance JSON "
                "before it can be recorded as events (§3.2)"
            )
    provenance = json.loads(provenance_path.read_text())
    captured_at = provenance.get("captured_at")
    if not captured_at:
        raise EventError(f"{name}: {provenance_path} records no captured_at, so the events have no ingestion time")
    # The digest of the document the issuer served, under either of the two key
    # names the captured files use. Hashing the provenance JSON instead would
    # put a hash of the description where the hash of the source belongs, and
    # the substitution would be invisible.
    digest = provenance.get("sha256") or provenance.get("raw_sha256")
    if not digest:
        raise EventError(
            f"{name}: {provenance_path} records neither 'sha256' nor 'raw_sha256', so the amounts have no source "
            "document to be traced to (§3.2, non-redistributable inputs)"
        )
    provenance_sha256 = str(digest)

    table = pd.read_csv(csv_path, dtype=str)
    missing = {"ex_date", "pay_date", "amount"} - set(table.columns)
    if missing:
        raise EventError(f"{name}: {csv_path} must carry columns ex_date, pay_date, amount; missing {sorted(missing)}")
    events: list[Event] = []
    for record in table.to_dict("records"):
        ex_date = str(record["ex_date"]).strip()
        pay_date = str(record["pay_date"] or "").strip() or None
        amount = float(str(record["amount"]).strip())
        if not math.isfinite(amount) or amount < 0.0:
            raise EventError(f"{name}: negative or non-finite distribution amount {record['amount']!r} in {csv_path}")
        events.append(
            Event(
                sid=sid,
                event_type="cash_distribution",
                effective_date=ex_date,
                payload={
                    "amount_per_share": amount,
                    "currency": currency,
                    "pay_date": pay_date,
                    "distribution_kind": distribution_kind,
                    "provenance_sha256": provenance_sha256,
                },
                ingested_at=str(captured_at),
                source=source,
                revision=1,
            )
        )
    return events


__all__ = [
    "ADJUSTMENT_COLUMNS",
    "ADJUSTMENT_KINDS",
    "Adjustment",
    "BarsRevision",
    "DISTRIBUTION_KINDS",
    "DistributionEntitlement",
    "EVENT_COLUMNS",
    "EVENT_TYPES",
    "Event",
    "EventError",
    "ISSUER_SOURCE",
    "PRICE_BASES",
    "PayloadError",
    "RATIO_KINDS",
    "REWRITE_RTOL",
    "RebaseNotRecordedError",
    "RewriteRatio",
    "CESSATION_TYPES",
    "CLAIM_STATUSES",
    "SCORING_BASES",
    "TerminalClaim",
    "adjustments_frame",
    "applied_price_factor",
    "applied_share_factor",
    "decode_payload",
    "distribution_amounts",
    "distribution_entitlement",
    "events_frame",
    "incomplete_reasons",
    "issuer_distribution_events",
    "matching_rebase",
    "record_bars_revision",
    "require_price_basis",
    "rewrite_ratio",
    "scoring_close",
    "series_price_basis",
    "terminal_claim",
    "terminal_claims",
    "total_return_close",
    "validate_bars_revision",
    "validate_payload",
    "write_bars_revision",
]
