"""The security master (engine spec §3.2, decision D2): identifiers, listings, membership.

``sid`` is an assigned surrogate key, minted once at the first ingest of an
instrument and never recomputed from current knowledge. It is deliberately
*not* a hash of a vendor tuple: the same instrument reached through the broker
(Alpaca symbology) and through the research vendor (Twelve Data) must resolve
to one identifier, or broker positions and research bars never join, and a hash
of ``(vendor, symbol)`` guarantees they do not. Verification changes
``entity_status`` and the provenance columns; it never changes the sid, so bar
file names, ledgers, positions and result records keyed by a provisional sid
stay valid when that sid is later verified.

What this module owns. The writer side of the three identity tables —
``securities``, ``listings`` and ``membership`` — in the schema of §3.2: the
minting rule, cross-vendor matching at ingest, the ``security_overrides.toml``
attestation file, the gap flag that surfaces the reuse case matching cannot
see, and the translation of point-in-time index intervals onto sids. The
*reader* side is :mod:`prism.io.dataset`: every as-of question about these
tables goes through ``DatasetView``, which applies the visibility predicate and
the revision-selection rule this module's output is written for.

Matching, exactly as §3.2 states it. A newly seen ``(vendor, vendor_symbol)``
series is split at every known ``ticker_change`` and ``delisting_notice`` into
segments, each with a ticker and a validity interval. A segment matches an
existing sid when its ``(ticker, interval)`` overlaps a ``listings`` row of
that sid *for any vendor*, or when ``security_overrides.toml`` asserts the
mapping. A segment matching more than one sid, or none, gets a new sid with
``entity_status = "provisional"`` and is flagged for attestation. Ticker reuse
after a delisting produces non-overlapping intervals and therefore two sids,
which is what keeps two instruments' holdings from merging into one position.

The stated limitation, and the flag that surfaces it. Reuse *inside* one vendor
series with no known event cannot be split, because nothing in the data says
where to cut. That case is flagged instead: a bar gap longer than
``gap_flag_sessions`` (default 15, counted on the bundled session table of
:mod:`prism.io.sessions`, never in calendar days) inside a segment becomes a
manifest ``coverage`` entry naming the sid, the segment and the gap length. The
threshold follows the integrity sweep already in the repository, which flags a
series with more than 15 missing sessions
(``docs/data_integrity_diagnostic.md:41-42``). A multi-session halt stays below
it; a multi-year stray series does not.

Failure is loud (SPEC.md N7). An unresolvable ticker raises rather than picking
a winner; an attestation naming an unknown sid raises; a gap computed outside
the session table's coverage raises rather than falling back to calendar days.
The one thing that is counted rather than refused is the provisional
population: ``n_provisional_sids`` and ``n_quarantined_sids`` are recorded on
the report and in the manifest coverage entries, because an identity awaiting
attestation is a known state of the workspace, not an error in a read.

Column conventions match :mod:`prism.io.dataset`: ``valid_from``, ``valid_to``,
``effective_date`` and ``announce_date`` are ISO ``YYYY-MM-DD`` date strings,
nullable where the interval is open; ``valid_at``, ``available_at`` and
``ingested_at`` are ISO 8601 instants with an offset, because they are compared
against ``cutoff(d)``, which is an instant. Intervals are inclusive of
``valid_to``.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import pandas as pd

from prism.io import sessions
from prism.io.universe_sp500 import HISTORY_FLOOR, normalize_ticker

#: The minting form of §3.2. The number is the next unused one in the
#: workspace; it carries no meaning beyond order of first sighting.
SID_FORMAT = "sec_%06d"

#: ``entity_status`` values of the ``securities`` table (§3.2).
ENTITY_STATUSES: tuple[str, ...] = ("provisional", "verified", "quarantined")

#: Asset classes the first version distinguishes (§3.2).
ASSET_CLASSES: tuple[str, ...] = ("equity", "etf")

#: Default of the ``[source]`` block's ``gap_flag_sessions`` (§3.2).
DEFAULT_GAP_FLAG_SESSIONS = 15

#: The attestation file, at ``workspace.data / OVERRIDES_FILENAME``.
OVERRIDES_FILENAME = "security_overrides.toml"

#: The only schema version the parser accepts.
OVERRIDES_SCHEMA = 1

#: ``source`` on every row an attestation writes. It is a source like any
#: other, so an attestation is as-of queryable and orderable against a vendor.
ATTESTATION_SOURCE = "security_overrides"

#: The events that split a vendor series into segments (§3.2). ``ticker_change``
#: renames the instrument; the other two close the segment on their last
#: session, after which the same ticker may denote a different instrument.
SPLITTING_EVENT_TYPES: tuple[str, ...] = ("ticker_change", "delisting_notice", "trading_cessation")

#: ``valid_at`` for an identity whose opening date the source does not record.
#: Availability for ``securities`` is ``valid_at`` itself (§3.2), so an epoch
#: ``valid_at`` states "known throughout" — which is what a cache that carries
#: no listing history actually supports, and what ``replay_recorded``'s
#: first-ingestion refusal still guards.
EPOCH_INSTANT = "1970-01-01T00:00:00-05:00"

SECURITIES_COLUMNS: tuple[str, ...] = (
    "sid",
    "valid_at",
    "entity_status",
    "quarantine_reason",
    "name",
    "asset_class",
    "share_class",
    "primary_listing",
    "available_at",
    "ingested_at",
    "source",
    "revision",
)

LISTINGS_COLUMNS: tuple[str, ...] = (
    "sid",
    "ticker",
    "vendor",
    "vendor_symbol",
    "valid_from",
    "valid_to",
    "announce_date",
    "available_at",
    "ingested_at",
    "source",
    "revision",
)

MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "index_id",
    "sid",
    "valid_from",
    "valid_to",
    "floor_unknown",
    "announce_date",
    "available_at",
    "ingested_at",
    "source",
    "revision",
)

TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "securities": SECURITIES_COLUMNS,
    "listings": LISTINGS_COLUMNS,
    "membership": MEMBERSHIP_COLUMNS,
}

TABLE_KEYS: dict[str, tuple[str, ...]] = {
    "securities": ("sid", "valid_at"),
    "listings": ("sid", "vendor", "valid_from"),
    "membership": ("index_id", "sid", "valid_from"),
}

TABLE_NAMES: tuple[str, ...] = tuple(TABLE_COLUMNS)

#: Fields excluded when deciding whether a re-ingested row equals the stored
#: one. ``revision`` is the answer, not an input; ``ingested_at`` is when *this
#: workspace* stored the row, not part of the claim the row makes, so an
#: unchanged fact re-read tomorrow keeps the date the workspace first held it.
_VOLATILE_FIELDS: frozenset[str] = frozenset({"revision", "ingested_at"})


class SecurityMasterError(ValueError):
    """An identity question cannot be answered as asked, and guessing would forge one.

    Raised rather than repaired: a wrong identity silently swaps one company's
    price history — and one account's position — for another's.
    """


# ---------------------------------------------------------------------------
# dates and instants
# ---------------------------------------------------------------------------


def sid_of(number: int) -> str:
    """The §3.2 minting form: ``sec_%06d``."""
    if number < 0:
        raise SecurityMasterError(f"sid number must be non-negative, got {number}")
    return SID_FORMAT % int(number)


def sid_number(sid: str) -> int:
    """The number inside a minted sid; refuses anything not of the minted form."""
    text = str(sid)
    if not text.startswith("sec_") or not text[4:].isdigit():
        raise SecurityMasterError(f"{sid!r} is not a minted sid of the form {SID_FORMAT % 0!r}")
    return int(text[4:])


def _as_date(value: str | dt.date | dt.datetime | pd.Timestamp) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def _opt_date(value: str | dt.date | dt.datetime | pd.Timestamp | None) -> dt.date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or (isinstance(value, str) and not value):
        return None
    if isinstance(value, pd.Timestamp) and pd.isna(value):
        return None
    return _as_date(value)


def _iso(day: dt.date | None) -> str | None:
    return None if day is None else day.isoformat()


def _instant(value: str | pd.Timestamp) -> str:
    """Validate and normalize an ISO 8601 instant with an offset."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise SecurityMasterError(
            f"{value!r} carries no UTC offset; ingested_at and valid_at are instants compared against cutoff(d), "
            "which is an instant"
        )
    return str(stamp.isoformat())


def _previous_day(day: dt.date) -> dt.date:
    """The day a segment ends on when the next segment opens at ``day``.

    The previous session where the bundled table covers it, else the previous
    calendar day. Segments abut and never overlap, which is what keeps
    :meth:`SecurityMaster.resolve` single-valued across a rename.
    """
    try:
        return sessions.previous_session(day)
    except sessions.SessionTableRangeError:
        return day - dt.timedelta(days=1)


def _next_day(day: dt.date) -> dt.date:
    """The day the segment after a cessation opens on: the next session, else the next calendar day."""
    try:
        return sessions.next_session(day)
    except sessions.SessionTableRangeError:
        return day + dt.timedelta(days=1)


def _session_close_instant(day: dt.date) -> str:
    """The close of ``day``'s own session as an instant, rolling a non-session forward.

    Rolling forward rather than back: a fact dated on a holiday becomes
    knowable at the next close, and dating it earlier would publish it before
    the day it belongs to. The rule mirrors ``prism.io.dataset``'s availability
    derivation.

    A date before the table's coverage takes :data:`EPOCH_INSTANT` — an
    identity opened before anything the workspace can date is knowable
    throughout — while a date after it raises, because there is no close to
    attribute it to and dating it 1970 would make it visible everywhere.
    """
    start, end = sessions.coverage()
    if day < start:
        return EPOCH_INSTANT
    if day > end:
        raise SecurityMasterError(
            f"{day.isoformat()} is after the bundled session table ends ({end.isoformat()}, version "
            f"{sessions.TABLE_VERSION}); it has no session close to become knowable at"
        )
    session = day if sessions.is_session(day) else sessions.next_session(day)
    stamp = pd.Timestamp(dt.datetime.combine(session, sessions.session_close(session)))
    return str(stamp.tz_localize(sessions.SESSION_TZ).isoformat())


@lru_cache(maxsize=1)
def _session_ordinals() -> dict[dt.date, int]:
    """Every covered session mapped onto its position, so a gap costs two lookups.

    A real import counts a gap between every pair of consecutive bars for every
    name; scanning the table for each pair would make the sweep quadratic in
    the sample and linear in the universe on top of it.
    """
    start, end = sessions.coverage()
    return {day: index for index, day in enumerate(sessions.sessions_between(start, end))}


def missing_sessions_between(first: dt.date, second: dt.date) -> int:
    """Sessions strictly between two bar dates, counted on the bundled table.

    Counted in sessions and never in calendar days: a Christmas week is not a
    gap, and a two-week suspension is. A range the table does not cover raises
    rather than falling back to a calendar count that would flag a different
    set of series.
    """
    if second < first:
        raise SecurityMasterError(f"bar dates are out of order: {second.isoformat()} precedes {first.isoformat()}")
    ordinals = _session_ordinals()
    opened, closed = ordinals.get(first), ordinals.get(second)
    if opened is not None and closed is not None:
        return closed - opened - 1
    try:
        between = sessions.sessions_between(first, second)
    except sessions.SessionTableRangeError as exc:
        start, end = sessions.coverage()
        raise SecurityMasterError(
            f"cannot count the gap {first.isoformat()}..{second.isoformat()} in sessions: the bundled table covers "
            f"{start.isoformat()}..{end.isoformat()} (version {sessions.TABLE_VERSION}). §3.2 counts gap_flag_sessions "
            "on that table, never in calendar days"
        ) from exc
    return sum(1 for day in between if first < day < second)


# ---------------------------------------------------------------------------
# segments and the events that cut them
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Segment:
    """One ticker over one validity interval, inclusive of ``valid_to``.

    ``valid_from`` is null when the interval opened before anything the source
    records, and ``valid_to`` is null when it is still open. ``opened_by``
    records which cut produced the segment, so a segment opened by
    ``reuse_after_cessation`` — the same ticker after another instrument
    stopped trading — is visible as such in the report.
    """

    ticker: str
    valid_from: dt.date | None
    valid_to: dt.date | None
    opened_by: Literal["series_start", "ticker_change", "reuse_after_cessation"] = "series_start"
    announce_date: dt.date | None = None

    def overlaps(self, valid_from: dt.date | None, valid_to: dt.date | None) -> bool:
        """Whether this interval shares at least one day with ``[valid_from, valid_to]``."""
        if self.valid_from is not None and valid_to is not None and self.valid_from > valid_to:
            return False
        if valid_from is not None and self.valid_to is not None and valid_from > self.valid_to:
            return False
        return True

    def contains(self, day: dt.date) -> bool:
        if self.valid_from is not None and day < self.valid_from:
            return False
        if self.valid_to is not None and day > self.valid_to:
            return False
        return True

    @property
    def label(self) -> str:
        """``from/to`` with an open end written as ``open``, the form a coverage entry carries."""
        return f"{self.valid_from.isoformat() if self.valid_from else 'open'}/" + (
            self.valid_to.isoformat() if self.valid_to else "open"
        )


@dataclass(frozen=True, slots=True)
class IdentityEvent:
    """A known event that cuts a vendor series into segments (§3.2).

    Keyed by ``ticker`` rather than by sid, because at ingest the series has no
    sid yet: identity is the question the event helps answer, not an input to
    it. ``closes_at`` is the last session the ticker denoted the instrument —
    ``expected_last_session`` on a ``delisting_notice``, ``last_session`` on a
    ``trading_cessation``.
    """

    event_type: str
    ticker: str
    effective_date: dt.date
    to_ticker: str | None = None
    last_session: dt.date | None = None
    announce_date: dt.date | None = None

    def __post_init__(self) -> None:
        if self.event_type not in SPLITTING_EVENT_TYPES:
            raise SecurityMasterError(
                f"event_type {self.event_type!r} does not split a series; §3.2 splits on {SPLITTING_EVENT_TYPES}"
            )
        if self.event_type == "ticker_change" and not self.to_ticker:
            raise SecurityMasterError("a ticker_change carries {from_ticker, to_ticker}; to_ticker is missing")

    @property
    def closes_at(self) -> dt.date:
        return self.last_session if self.last_session is not None else self.effective_date

    @classmethod
    def ticker_change(
        cls,
        from_ticker: str,
        to_ticker: str,
        effective_date: str | dt.date,
        *,
        announce_date: str | dt.date | None = None,
    ) -> "IdentityEvent":
        return cls(
            event_type="ticker_change",
            ticker=normalize_ticker(from_ticker),
            to_ticker=normalize_ticker(to_ticker),
            effective_date=_as_date(effective_date),
            announce_date=_opt_date(announce_date),
        )

    @classmethod
    def delisting(
        cls,
        ticker: str,
        *,
        notice_date: str | dt.date,
        expected_last_session: str | dt.date,
        announce_date: str | dt.date | None = None,
    ) -> "IdentityEvent":
        return cls(
            event_type="delisting_notice",
            ticker=normalize_ticker(ticker),
            effective_date=_as_date(notice_date),
            last_session=_as_date(expected_last_session),
            announce_date=_opt_date(announce_date) or _as_date(notice_date),
        )

    @classmethod
    def cessation(cls, ticker: str, last_session: str | dt.date) -> "IdentityEvent":
        day = _as_date(last_session)
        return cls(
            event_type="trading_cessation", ticker=normalize_ticker(ticker), effective_date=day, last_session=day
        )


def _event_order(event: IdentityEvent) -> tuple[dt.date, int]:
    # A rename and a cessation dated the same day: the rename applies first, so
    # the cessation closes the segment the rename opened rather than the one it
    # already closed.
    return event.effective_date, SPLITTING_EVENT_TYPES.index(event.event_type)


def segment_series(
    ticker: str,
    first_bar: dt.date,
    last_bar: dt.date,
    events: Sequence[IdentityEvent] = (),
) -> tuple[Segment, ...]:
    """Cut ``[first_bar, last_bar]`` under ``ticker`` into segments at the known events (§3.2).

    A ``ticker_change`` closes the segment on the day before it takes effect
    and opens the next one under the new ticker. A ``delisting_notice`` or
    ``trading_cessation`` closes the segment on its last session; bars after
    that date open a *new* segment under the same ticker, which is the ticker
    reuse case and is exactly why it yields a second sid.
    """
    if last_bar < first_bar:
        raise SecurityMasterError(f"series ends {last_bar.isoformat()} before it starts {first_bar.isoformat()}")
    current = normalize_ticker(ticker)
    opened = first_bar
    opened_by: Literal["series_start", "ticker_change", "reuse_after_cessation"] = "series_start"
    announce: dt.date | None = None
    out: list[Segment] = []
    for event in sorted(events, key=_event_order):
        if event.ticker != current:
            continue
        if event.event_type == "ticker_change":
            if not (opened < event.effective_date <= last_bar):
                continue
            out.append(Segment(current, opened, _previous_day(event.effective_date), opened_by, announce))
            current = str(event.to_ticker)
            opened = event.effective_date
            opened_by = "ticker_change"
            announce = event.announce_date
            continue
        closes = event.closes_at
        if closes < opened or closes >= last_bar:
            continue
        out.append(Segment(current, opened, closes, opened_by, announce))
        opened = _next_day(closes)
        opened_by = "reuse_after_cessation"
        announce = None
    out.append(Segment(current, opened, last_bar, opened_by, announce))
    return tuple(out)


def _chains(segments: Sequence[Segment]) -> tuple[tuple[Segment, ...], ...]:
    """Group segments into runs that one instrument spans.

    A rename keeps the instrument and the run; a cessation ends it, so the next
    segment — the same ticker under a different issuer — starts a new run and
    will take its own sid.
    """
    out: list[list[Segment]] = []
    for segment in segments:
        if not out or segment.opened_by == "reuse_after_cessation":
            out.append([segment])
        else:
            out[-1].append(segment)
    return tuple(tuple(chain) for chain in out)


# ---------------------------------------------------------------------------
# the attestation file
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Attestation:
    """One ``[[attestation]]`` entry: the owner's statement about one sid's identity."""

    sid: str
    action: Literal["verify", "quarantine"]
    attested_by: str
    attested_at: dt.date
    attested_identity: str | None = None
    reason: str | None = None
    evidence: str | None = None


@dataclass(frozen=True, slots=True)
class Merge:
    """One ``[[merge]]`` entry: ``absorb``'s vendor mappings move to ``keep``, and ``absorb`` retires."""

    keep: str
    absorb: str
    attested_by: str
    attested_at: dt.date
    evidence: str | None = None


@dataclass(frozen=True, slots=True)
class Alias:
    """One ``[[alias]]`` entry: a vendor mapping overlapping-ticker matching could not establish."""

    sid: str
    vendor: str
    vendor_symbol: str
    valid_from: dt.date | None
    valid_to: dt.date | None
    attested_by: str
    attested_at: dt.date
    evidence: str | None = None


_ATTESTATION_KEYS = {"sid", "action", "attested_identity", "attested_by", "attested_at", "evidence", "reason"}
_MERGE_KEYS = {"keep", "absorb", "attested_by", "attested_at", "evidence"}
_ALIAS_KEYS = {"sid", "vendor", "vendor_symbol", "valid_from", "valid_to", "attested_by", "attested_at", "evidence"}
_OVERRIDES_KEYS = {"schema", "attestation", "merge", "alias"}


def _require_keys(entry: dict[str, Any], required: Iterable[str], allowed: set[str], what: str) -> None:
    missing = [key for key in required if entry.get(key) in (None, "")]
    if missing:
        raise SecurityMasterError(f"{what} is missing {missing} in {OVERRIDES_FILENAME}: {entry}")
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise SecurityMasterError(
            f"{what} carries unknown keys {unknown} in {OVERRIDES_FILENAME}; a misspelled key would silently attest "
            "nothing"
        )


@dataclass(frozen=True)
class SecurityOverrides:
    """``security_overrides.toml`` parsed (§3.2). The owner's attestations, and nothing automatic."""

    schema: int = OVERRIDES_SCHEMA
    attestations: tuple[Attestation, ...] = ()
    merges: tuple[Merge, ...] = ()
    aliases: tuple[Alias, ...] = ()
    path: Path | None = None

    @classmethod
    def empty(cls) -> "SecurityOverrides":
        return cls()

    @classmethod
    def read(cls, path: Path) -> "SecurityOverrides":
        """Parse the attestation file, refusing an unknown schema, action or key."""
        path = Path(path)
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SecurityMasterError(
                f"{path} does not exist; use read_if_present when no attestation is expected"
            ) from exc
        except tomllib.TOMLDecodeError as exc:
            raise SecurityMasterError(f"{path} is not valid TOML: {exc}") from exc
        unknown = sorted(set(payload) - _OVERRIDES_KEYS)
        if unknown:
            raise SecurityMasterError(f"{path} carries unknown top-level keys {unknown}")
        schema = payload.get("schema")
        if schema != OVERRIDES_SCHEMA:
            raise SecurityMasterError(f"{path} declares schema {schema!r}; this reader accepts {OVERRIDES_SCHEMA}")
        attestations: list[Attestation] = []
        for entry in payload.get("attestation", []):
            action = entry.get("action")
            if action not in ("verify", "quarantine"):
                raise SecurityMasterError(
                    f"{path}: attestation action must be 'verify' or 'quarantine', got {action!r}"
                )
            required = ["sid", "action", "attested_by", "attested_at"] + (["reason"] if action == "quarantine" else [])
            _require_keys(entry, required, _ATTESTATION_KEYS, "an [[attestation]]")
            attestations.append(
                Attestation(
                    sid=str(entry["sid"]),
                    action=action,
                    attested_by=str(entry["attested_by"]),
                    attested_at=_as_date(entry["attested_at"]),
                    attested_identity=_opt_str(entry.get("attested_identity")),
                    reason=_opt_str(entry.get("reason")),
                    evidence=_opt_str(entry.get("evidence")),
                )
            )
        merges: list[Merge] = []
        for entry in payload.get("merge", []):
            _require_keys(entry, ["keep", "absorb", "attested_by", "attested_at"], _MERGE_KEYS, "a [[merge]]")
            if str(entry["keep"]) == str(entry["absorb"]):
                raise SecurityMasterError(f"{path}: a [[merge]] absorbs {entry['absorb']} into itself")
            merges.append(
                Merge(
                    keep=str(entry["keep"]),
                    absorb=str(entry["absorb"]),
                    attested_by=str(entry["attested_by"]),
                    attested_at=_as_date(entry["attested_at"]),
                    evidence=_opt_str(entry.get("evidence")),
                )
            )
        aliases: list[Alias] = []
        for entry in payload.get("alias", []):
            _require_keys(
                entry, ["sid", "vendor", "vendor_symbol", "attested_by", "attested_at"], _ALIAS_KEYS, "an [[alias]]"
            )
            aliases.append(
                Alias(
                    sid=str(entry["sid"]),
                    vendor=str(entry["vendor"]),
                    vendor_symbol=str(entry["vendor_symbol"]),
                    valid_from=_opt_date(entry.get("valid_from")),
                    valid_to=_opt_date(entry.get("valid_to")),
                    attested_by=str(entry["attested_by"]),
                    attested_at=_as_date(entry["attested_at"]),
                    evidence=_opt_str(entry.get("evidence")),
                )
            )
        return cls(
            schema=int(schema),
            attestations=tuple(attestations),
            merges=tuple(merges),
            aliases=tuple(aliases),
            path=path,
        )

    @classmethod
    def read_if_present(cls, path: Path) -> "SecurityOverrides":
        """The file's contents, or an empty attestation set when no file exists.

        An absent attestation file is a legitimate state — nothing has been
        attested yet — and is distinct from a file that exists and does not
        parse, which raises.
        """
        path = Path(path)
        return cls.read(path) if path.is_file() else cls.empty()

    def alias_for(self, vendor: str, vendor_symbol: str) -> tuple[Alias, ...]:
        return tuple(a for a in self.aliases if a.vendor == vendor and a.vendor_symbol == vendor_symbol)


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


# ---------------------------------------------------------------------------
# ingest results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GapFlag:
    """A bar gap longer than the declared threshold inside one segment (§3.2).

    The flag is the only signal available for reuse inside one vendor series
    with no known event, which matching cannot split. It names the sid, the
    segment and the gap, and becomes a manifest ``coverage`` entry.
    """

    sid: str
    vendor: str
    vendor_symbol: str
    ticker: str
    segment: str
    gap_from: dt.date
    gap_to: dt.date
    missing_sessions: int
    gap_flag_sessions: int

    def to_coverage_entry(self) -> dict[str, Any]:
        return {
            "sid": self.sid,
            "vendor": self.vendor,
            "vendor_symbol": self.vendor_symbol,
            "ticker": self.ticker,
            "segment": self.segment,
            "gap": f"{self.gap_from.isoformat()}/{self.gap_to.isoformat()}",
            "missing_sessions": int(self.missing_sessions),
            "gap_flag_sessions": int(self.gap_flag_sessions),
            "flagged": True,
        }


@dataclass(frozen=True, slots=True)
class AttestationFlag:
    """A segment that matched no existing sid, or more than one, and minted a provisional sid.

    Recorded rather than resolved: the engine never writes an attestation on
    the owner's behalf, so the flag is what the owner acts on when editing
    ``security_overrides.toml``.
    """

    sid: str
    vendor: str
    vendor_symbol: str
    ticker: str
    segment: str
    reason: Literal["no_match", "ambiguous_match"]
    candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sid": self.sid,
            "vendor": self.vendor,
            "vendor_symbol": self.vendor_symbol,
            "ticker": self.ticker,
            "segment": self.segment,
            "reason": self.reason,
            "candidates": list(self.candidates),
        }


@dataclass(frozen=True, slots=True)
class SegmentResolution:
    """What one segment of one vendor series resolved to."""

    segment: Segment
    sid: str
    minted: bool
    matched_by: Literal["ticker_interval_overlap", "alias", "minted"]
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SeriesResolution:
    """What one ``(vendor, vendor_symbol)`` series resolved to, segment by segment."""

    vendor: str
    vendor_symbol: str
    segments: tuple[SegmentResolution, ...]
    gap_flags: tuple[GapFlag, ...] = ()

    @property
    def sids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.sid for item in self.segments))

    def sid_on(self, day: str | dt.date) -> str:
        """The sid this series denoted on ``day``; refuses a date the series does not cover."""
        target = _as_date(day)
        for item in self.segments:
            if item.segment.contains(target):
                return item.sid
        raise SecurityMasterError(
            f"{self.vendor}:{self.vendor_symbol} covers no segment containing {target.isoformat()}"
        )


@dataclass(frozen=True)
class MembershipImport:
    """What one index-membership translation wrote, and what it could not resolve."""

    index_id: str
    n_intervals: int
    n_rows: int
    unresolved: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_id": self.index_id,
            "n_intervals": int(self.n_intervals),
            "n_rows": int(self.n_rows),
            "n_unresolved": len(self.unresolved),
            "unresolved": [{"ticker": t, "reason": r} for t, r in self.unresolved],
        }


@dataclass(frozen=True)
class SecurityMasterReport:
    """The counts §3.2 requires the result record and the manifest to carry."""

    n_sids: int
    n_provisional_sids: int
    n_verified_sids: int
    n_quarantined_sids: int
    provisional_sids: tuple[str, ...]
    quarantined_sids: tuple[str, ...]
    attestation_flags: tuple[AttestationFlag, ...]
    gap_flags: tuple[GapFlag, ...]
    unresolvable_tickers: tuple[tuple[str, tuple[str, ...]], ...]
    n_listings: int
    n_membership: int
    gap_flag_sessions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_sids": int(self.n_sids),
            "n_provisional_sids": int(self.n_provisional_sids),
            "n_verified_sids": int(self.n_verified_sids),
            "n_quarantined_sids": int(self.n_quarantined_sids),
            "provisional_sids": list(self.provisional_sids),
            "quarantined_sids": list(self.quarantined_sids),
            "attestation_flags": [flag.to_dict() for flag in self.attestation_flags],
            "gap_flags": [flag.to_coverage_entry() for flag in self.gap_flags],
            "unresolvable_tickers": [{"ticker": t, "sids": list(s)} for t, s in self.unresolvable_tickers],
            "n_listings": int(self.n_listings),
            "n_membership": int(self.n_membership),
            "gap_flag_sessions": int(self.gap_flag_sessions),
        }

    def coverage_entries(self) -> dict[str, Any]:
        """The manifest ``coverage`` fragment: gap flags, provisional counts, unresolvable tickers."""
        return {
            "gap_flag_sessions": int(self.gap_flag_sessions),
            "gap_flags": [flag.to_coverage_entry() for flag in self.gap_flags],
            "n_provisional_sids": int(self.n_provisional_sids),
            "provisional_sids": list(self.provisional_sids),
            "n_quarantined_sids": int(self.n_quarantined_sids),
            "quarantined_sids": list(self.quarantined_sids),
            "attestation_flags": [flag.to_dict() for flag in self.attestation_flags],
            "unresolvable_tickers": [{"ticker": t, "sids": list(s)} for t, s in self.unresolvable_tickers],
        }


# ---------------------------------------------------------------------------
# the master
# ---------------------------------------------------------------------------


class SecurityMaster:
    """Mints sids, writes ``securities``, ``listings`` and ``membership``, and resolves tickers.

    One master owns one workspace's identity tables. It is opened over what the
    workspace already holds (:meth:`open`), so a sid minted by an earlier
    ingest is reused rather than re-minted: that stability is the whole point
    of a surrogate key, and it is what lets a position recorded under
    ``sec_000412`` still be the same instrument a year later.

    Rows are appended, never edited. A re-ingested row identical to the stored
    one writes nothing; a differing one writes the next revision within its
    ``(key, source)``, which is the full-row replacement §3.2 defines.
    """

    def __init__(
        self,
        *,
        overrides: SecurityOverrides | None = None,
        gap_flag_sessions: int = DEFAULT_GAP_FLAG_SESSIONS,
        rows: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        if gap_flag_sessions < 0:
            raise SecurityMasterError(f"gap_flag_sessions must be non-negative, got {gap_flag_sessions}")
        self.overrides = overrides if overrides is not None else SecurityOverrides.empty()
        self.gap_flag_sessions = int(gap_flag_sessions)
        self._rows: dict[str, list[dict[str, Any]]] = {name: list((rows or {}).get(name, [])) for name in TABLE_NAMES}
        self._sources_by_sid: dict[str, set[str]] = {}
        for row in self._rows["securities"]:
            self._sources_by_sid.setdefault(str(row["sid"]), set()).add(str(row["source"]))
        self._attestation_flags: list[AttestationFlag] = []
        self._gap_flags: list[GapFlag] = []
        self._applied = False
        self._live_cache: list[dict[str, Any]] | None = None
        known = [sid_number(row["sid"]) for row in self._rows["securities"] if str(row["sid"]).startswith("sec_")]
        self._next_number = max(known) + 1 if known else 0

    # -- construction ----------------------------------------------------

    @classmethod
    def open(cls, root: Path, *, gap_flag_sessions: int = DEFAULT_GAP_FLAG_SESSIONS) -> "SecurityMaster":
        """Open the master over the tables already under ``root`` (``workspace.data``).

        Absent tables are an empty master, not an error: a workspace before its
        first ingest holds none. The attestation file is read if present.
        """
        root = Path(root)
        rows: dict[str, list[dict[str, Any]]] = {}
        for name in TABLE_NAMES:
            path = root / f"{name}.parquet"
            if not path.is_file():
                continue
            frame = pd.read_parquet(path)
            missing = [column for column in TABLE_KEYS[name] if column not in frame.columns]
            if missing:
                raise SecurityMasterError(f"{path} is missing key column(s) {missing}; it is not a §3.2 {name} table")
            rows[name] = [
                {column: _clean(record.get(column)) for column in TABLE_COLUMNS[name]}
                for record in frame.to_dict("records")
            ]
        return cls(
            overrides=SecurityOverrides.read_if_present(root / OVERRIDES_FILENAME),
            gap_flag_sessions=gap_flag_sessions,
            rows=rows,
        )

    # -- ingest ----------------------------------------------------------

    def ingest_series(
        self,
        vendor: str,
        vendor_symbol: str,
        *,
        ingested_at: str,
        ticker: str | None = None,
        bar_dates: Sequence[dt.date | str] | None = None,
        first_bar: dt.date | str | None = None,
        last_bar: dt.date | str | None = None,
        events: Sequence[IdentityEvent] = (),
        name: str | None = None,
        asset_class: str = "equity",
        share_class: str | None = None,
        primary_listing: str | None = None,
    ) -> SeriesResolution:
        """Resolve one vendor series to sids, minting where nothing matches (§3.2, D2).

        ``bar_dates`` are the sessions the series actually carries; giving them
        is what makes the gap flag computable, and a series ingested without
        them is resolved but never flagged. ``first_bar`` and ``last_bar``
        default to the first and last of ``bar_dates``.
        """
        if self._applied:
            raise SecurityMasterError(
                "the attestations have already been applied to this master; ingest before finalize, so that an "
                "attestation is never overtaken by a later ingest of the sid it decided"
            )
        if asset_class not in ASSET_CLASSES:
            raise SecurityMasterError(f"asset_class must be one of {ASSET_CLASSES}, got {asset_class!r}")
        stamp = _instant(ingested_at)
        vendor = str(vendor)
        vendor_symbol = str(vendor_symbol)
        days = sorted({_as_date(day) for day in (bar_dates or ())})
        opened = _opt_date(first_bar) or (days[0] if days else None)
        closed = _opt_date(last_bar) or (days[-1] if days else None)
        if opened is None or closed is None:
            raise SecurityMasterError(
                f"{vendor}:{vendor_symbol} was ingested with neither bar_dates nor first_bar/last_bar; a listing "
                "interval is not inferable from nothing (N7)"
            )
        segments = segment_series(ticker or vendor_symbol, opened, closed, events)
        resolutions: list[SegmentResolution] = []
        flags: list[GapFlag] = []
        for chain in _chains(segments):
            resolution = self._resolve_chain(chain, vendor, vendor_symbol)
            for segment in chain:
                self._write_listing(resolution.sid, segment, vendor, vendor_symbol, stamp)
                if days:
                    flags.extend(self._scan_gaps(resolution.sid, segment, vendor, vendor_symbol, days))
                resolutions.append(
                    SegmentResolution(
                        segment, resolution.sid, resolution.minted, resolution.matched_by, resolution.candidates
                    )
                )
            if resolution.minted:
                self._mint_security(
                    resolution.sid,
                    chain[0],
                    vendor=vendor,
                    ingested_at=stamp,
                    name=name,
                    asset_class=asset_class,
                    share_class=share_class,
                    primary_listing=primary_listing,
                )
            else:
                self._verify_by_second_source(resolution.sid, vendor=vendor, ingested_at=stamp)
        self._gap_flags.extend(flags)
        return SeriesResolution(vendor, vendor_symbol, tuple(resolutions), tuple(flags))

    def _resolve_chain(self, chain: tuple[Segment, ...], vendor: str, vendor_symbol: str) -> SegmentResolution:
        """Match one chain of segments to an existing sid, or mint one and flag it for attestation.

        A *chain* is the run of segments a series holds between two identity
        breaks: a ``ticker_change`` renames the instrument and keeps it, while a
        ``delisting_notice`` or ``trading_cessation`` ends it. So the segments
        of a chain are one instrument by the events' own assertion, and are
        matched together — the chain matches the sid its segments overlap under
        their tickers. Matching each segment on its own would mint a second sid
        at every rename, because the new ticker overlaps nothing the old one
        listed, which is precisely the identity break a rename is not.
        """
        aliases = [
            alias
            for alias in self.overrides.alias_for(vendor, vendor_symbol)
            if any(segment.overlaps(alias.valid_from, alias.valid_to) for segment in chain)
        ]
        alias_sids = sorted({alias.sid for alias in aliases})
        label = "|".join(segment.label for segment in chain)
        if len(alias_sids) > 1:
            raise SecurityMasterError(
                f"{OVERRIDES_FILENAME} asserts {alias_sids} for {vendor}:{vendor_symbol} over {label}; an "
                "attestation that is itself ambiguous cannot resolve an identity"
            )
        if alias_sids:
            sid = alias_sids[0]
            if not self._has_security(sid):
                raise SecurityMasterError(
                    f"{OVERRIDES_FILENAME} aliases {vendor}:{vendor_symbol} to {sid}, which no securities row names; "
                    "an alias asserts a mapping onto an existing identity and mints no sid"
                )
            return SegmentResolution(chain[0], sid, minted=False, matched_by="alias")
        candidates = self._candidates(chain)
        if len(candidates) == 1:
            return SegmentResolution(chain[0], candidates[0], minted=False, matched_by="ticker_interval_overlap")
        sid = self._next_sid()
        reason: Literal["no_match", "ambiguous_match"] = "no_match" if not candidates else "ambiguous_match"
        self._attestation_flags.append(
            AttestationFlag(sid, vendor, vendor_symbol, chain[0].ticker, label, reason, tuple(candidates))
        )
        return SegmentResolution(chain[0], sid, minted=True, matched_by="minted", candidates=tuple(candidates))

    def _candidates(self, chain: Sequence[Segment]) -> tuple[str, ...]:
        """Sids whose live listing overlaps any segment of the chain under that segment's ticker."""
        found: list[str] = []
        for row in self._live_listings():
            for segment in chain:
                if row["ticker"] != segment.ticker:
                    continue
                if not segment.overlaps(_opt_date(row["valid_from"]), _opt_date(row["valid_to"])):
                    continue
                sid = str(row["sid"])
                if sid not in found:
                    found.append(sid)
                break
        return tuple(sorted(found))

    def _next_sid(self) -> str:
        sid = sid_of(self._next_number)
        self._next_number += 1
        return sid

    def _scan_gaps(
        self, sid: str, segment: Segment, vendor: str, vendor_symbol: str, days: Sequence[dt.date]
    ) -> list[GapFlag]:
        inside = [day for day in days if segment.contains(day)]
        flags: list[GapFlag] = []
        for first, second in zip(inside, inside[1:]):
            missing = missing_sessions_between(first, second)
            if missing > self.gap_flag_sessions:
                flags.append(
                    GapFlag(
                        sid=sid,
                        vendor=vendor,
                        vendor_symbol=vendor_symbol,
                        ticker=segment.ticker,
                        segment=segment.label,
                        gap_from=first,
                        gap_to=second,
                        missing_sessions=missing,
                        gap_flag_sessions=self.gap_flag_sessions,
                    )
                )
        return flags

    # -- table writers ---------------------------------------------------

    def _write_listing(self, sid: str, segment: Segment, vendor: str, vendor_symbol: str, ingested_at: str) -> None:
        key = (sid, vendor, _iso(segment.valid_from))
        for existing in self._rows_for("listings", key):
            if existing["vendor_symbol"] not in (None, vendor_symbol):
                raise SecurityMasterError(
                    f"{vendor} lists {sid} from {key[2]} as both {existing['vendor_symbol']!r} and "
                    f"{vendor_symbol!r}; one vendor cannot denote one instrument by two symbols from one date"
                )
        self._append(
            "listings",
            {
                "sid": sid,
                "ticker": segment.ticker,
                "vendor": vendor,
                "vendor_symbol": vendor_symbol,
                "valid_from": _iso(segment.valid_from),
                "valid_to": _iso(segment.valid_to),
                "announce_date": _iso(segment.announce_date),
                "available_at": None,
                "ingested_at": ingested_at,
                "source": vendor,
                "revision": 1,
            },
        )

    def _mint_security(
        self,
        sid: str,
        segment: Segment,
        *,
        vendor: str,
        ingested_at: str,
        name: str | None,
        asset_class: str,
        share_class: str | None,
        primary_listing: str | None,
    ) -> None:
        self._append(
            "securities",
            {
                "sid": sid,
                "valid_at": _session_close_instant(segment.valid_from) if segment.valid_from else EPOCH_INSTANT,
                "entity_status": "provisional",
                "quarantine_reason": None,
                "name": name,
                "asset_class": asset_class,
                "share_class": share_class,
                "primary_listing": primary_listing,
                "available_at": None,
                "ingested_at": ingested_at,
                "source": vendor,
                "revision": 1,
            },
        )
        self._sources_by_sid.setdefault(sid, set()).add(vendor)

    def _verify_by_second_source(self, sid: str, *, vendor: str, ingested_at: str) -> None:
        """A second independent source covering a segment verifies the identity (§3.2).

        Dated by the ingest instant, not by the segment: agreement between two
        vendors is a fact about this workspace, established when the second
        vendor's series arrived. A reproduction as of an earlier date therefore
        still sees the provisional status the validator saw then.
        """
        sources = self._sources_by_sid.setdefault(sid, set())
        if vendor in sources:
            return
        sources.add(vendor)
        if self.status_of(sid) != "provisional":
            return
        self._append("securities", self._status_row(sid, "verified", valid_at=ingested_at, source=vendor))

    def _status_row(
        self,
        sid: str,
        entity_status: str,
        *,
        valid_at: str,
        source: str,
        quarantine_reason: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """A full ``securities`` row carrying the new status: every revision is a complete row (§3.2)."""
        if entity_status not in ENTITY_STATUSES:
            raise SecurityMasterError(f"entity_status must be one of {ENTITY_STATUSES}, got {entity_status!r}")
        latest = self._latest_security(sid)
        if latest is None:
            raise SecurityMasterError(f"no securities row names {sid}; a status change needs an identity to change")
        return {
            "sid": sid,
            "valid_at": valid_at,
            "entity_status": entity_status,
            "quarantine_reason": quarantine_reason,
            "name": name if name is not None else latest["name"],
            "asset_class": latest["asset_class"],
            "share_class": latest["share_class"],
            "primary_listing": latest["primary_listing"],
            "available_at": None,
            "ingested_at": latest["ingested_at"],
            "source": source,
            "revision": 1,
        }

    def _append(self, table: str, row: dict[str, Any]) -> None:
        """Append one row, deduping an identical re-ingest and bumping the revision otherwise."""
        columns = TABLE_COLUMNS[table]
        missing = [column for column in columns if column not in row]
        if missing:
            raise SecurityMasterError(f"a {table} row is missing {missing}")
        row = {column: row[column] for column in columns}
        existing = self._rows_for(table, tuple(row[column] for column in TABLE_KEYS[table]), source=str(row["source"]))
        if existing:
            latest = max(existing, key=lambda item: int(item["revision"]))
            if _content_equal(latest, row):
                return
            row = {**row, "revision": int(latest["revision"]) + 1}
        self._rows[table].append(row)
        if table == "listings":
            self._live_cache = None

    def _rows_for(self, table: str, key: tuple[Any, ...], *, source: str | None = None) -> list[dict[str, Any]]:
        columns = TABLE_KEYS[table]
        return [
            row
            for row in self._rows[table]
            if tuple(row[column] for column in columns) == key and (source is None or str(row["source"]) == source)
        ]

    # -- membership ------------------------------------------------------

    def add_membership(
        self,
        index_id: str,
        sid: str,
        *,
        valid_from: str | dt.date | None,
        valid_to: str | dt.date | None,
        source: str,
        ingested_at: str,
        floor_unknown: bool | None = None,
        announce_date: str | dt.date | None = None,
    ) -> None:
        """One ``membership`` row. ``floor_unknown`` defaults to "the interval opens before records"."""
        if not self._has_security(sid):
            raise SecurityMasterError(
                f"membership names {sid}, which no securities row names; an index cannot hold an instrument the "
                "master has never seen"
            )
        opened = _opt_date(valid_from)
        self._append(
            "membership",
            {
                "index_id": str(index_id),
                "sid": str(sid),
                "valid_from": _iso(opened),
                "valid_to": _iso(_opt_date(valid_to)),
                "floor_unknown": bool(opened is None if floor_unknown is None else floor_unknown),
                "announce_date": _iso(_opt_date(announce_date)),
                "available_at": None,
                "ingested_at": _instant(ingested_at),
                "source": str(source),
                "revision": 1,
            },
        )

    def import_membership_intervals(
        self,
        index_id: str,
        intervals: Iterable[tuple[str, Any, Any]],
        *,
        as_of: str | dt.date,
        source: str,
        ingested_at: str,
        floor: str | dt.date | pd.Timestamp = HISTORY_FLOOR,
        on_unresolved: Literal["raise", "skip"] = "raise",
    ) -> MembershipImport:
        """Translate ``(ticker, start, end)`` intervals onto sids, under the §3.2 conventions.

        Two conventions, translated rather than inherited. An ``end`` *equal to*
        the file's ``as_of`` is an *open* interval (``valid_to = null``), not a
        closure on the day the file was built — the file says "still a member",
        and closing it there would drop the name from every later decision. An
        end after the as-of is a scheduled removal the scrape already knew and
        is kept verbatim (Run: two such rows in
        ``data/universe/sp500_membership_2026-06-16.parquet``, CPB and POOL at
        2026-06-22); reading those as open would forget a known closure.

        A ``start`` at or before the reconstruction's history floor becomes
        ``valid_from = null`` with ``floor_unknown = true``: the interval opened
        before anything the source records. At or *before*, not equal to,
        because the file carries starts older than the floor (Run: BUD and DIS
        at 1976-07-01) and a record time that far outside the session table is
        not a date the availability derivation of §3.2 can walk.

        A ticker the master cannot resolve over the interval is a counted skip
        under ``on_unresolved="skip"`` — the same measured-survivorship-leak
        discipline the coverage ledger of :mod:`prism.io.universe_sp500` uses —
        and a refusal under the default. It is never silently dropped.
        """
        as_of_day = _as_date(as_of)
        floor_day = _as_date(floor)
        stamp = _instant(ingested_at)
        unresolved: list[tuple[str, str]] = []
        written = 0
        seen = 0
        keys: set[tuple[str, str | None]] = set()
        for ticker, start, end in intervals:
            seen += 1
            opened = _opt_date(start)
            closed = _opt_date(end)
            floor_unknown = opened is None or opened <= floor_day
            valid_from = None if floor_unknown else opened
            valid_to = None if closed == as_of_day else closed
            try:
                sid = self.resolve_interval(normalize_ticker(ticker), valid_from, valid_to)
            except SecurityMasterError as exc:
                if on_unresolved == "raise":
                    raise
                unresolved.append((normalize_ticker(ticker), str(exc)))
                continue
            key = (sid, _iso(valid_from))
            if key in keys:
                # Two intervals of one file that share a key are not a revision
                # of each other: the key is (index_id, sid, valid_from), so they
                # would overlap inside the sample. That is a defect in the
                # source, refused rather than folded into one row.
                raise SecurityMasterError(
                    f"{index_id} carries two intervals for {ticker!r} ({sid}) opening at {key[1]}; inside the sample "
                    "they overlap, which is a defect in the source, not a revision to choose between"
                )
            keys.add(key)
            self.add_membership(
                index_id,
                sid,
                valid_from=valid_from,
                valid_to=valid_to,
                floor_unknown=floor_unknown,
                source=source,
                ingested_at=stamp,
            )
            written += 1
        return MembershipImport(str(index_id), seen, written, tuple(unresolved))

    # -- attestations ----------------------------------------------------

    def apply_overrides(self) -> None:
        """Apply the attestation file: verify, quarantine, then merge (§3.2).

        Merges are applied last, so a sid that an attestation quarantined and a
        merge retired carries the merge's reason: the merge is the later
        statement about that identity.
        """
        if self._applied:
            return
        for attestation in self.overrides.attestations:
            if not self._has_security(attestation.sid):
                raise SecurityMasterError(
                    f"{OVERRIDES_FILENAME} attests {attestation.sid}, which no securities row names; an attestation "
                    "records a decision about an existing identity and mints no sid"
                )
            self._append(
                "securities",
                self._status_row(
                    attestation.sid,
                    "verified" if attestation.action == "verify" else "quarantined",
                    valid_at=_session_close_instant(attestation.attested_at),
                    source=ATTESTATION_SOURCE,
                    quarantine_reason=attestation.reason if attestation.action == "quarantine" else None,
                    name=attestation.attested_identity,
                ),
            )
        for merge in self.overrides.merges:
            self._apply_merge(merge)
        self._applied = True

    def _apply_merge(self, merge: Merge) -> None:
        """Move ``absorb``'s vendor mappings onto ``keep`` and retire ``absorb``.

        Both sids stay resolvable *as sids* — no earlier record is invalidated —
        but ``absorb``'s listing rows are superseded by null-valued full
        revisions, which is the deletion §3.2 defines, so the ticker they
        carried resolves to ``keep`` alone and stays single-valued.

        ``absorb`` retires as ``quarantined`` with the merge recorded as its
        reason. The ``entity_status`` enum has three values and none of them is
        "merged"; quarantined carries the operative consequence — the sid is
        never eligible in a new evaluation — while its rows stay readable, so a
        historical reproduction that consumed them still reproduces.
        """
        for sid in (merge.keep, merge.absorb):
            if not self._has_security(sid):
                raise SecurityMasterError(f"{OVERRIDES_FILENAME} merges {sid}, which no securities row names")
        stamp = _session_close_instant(merge.attested_at)
        for row in [row for row in self._live_listings() if str(row["sid"]) == merge.absorb]:
            self._append(
                "listings",
                {
                    **row,
                    "sid": merge.keep,
                    "ingested_at": stamp,
                    "source": ATTESTATION_SOURCE,
                    "revision": 1,
                },
            )
            self._append(
                "listings",
                {
                    **row,
                    "ticker": None,
                    "vendor_symbol": None,
                    "valid_to": None,
                    "announce_date": None,
                    "ingested_at": stamp,
                    "revision": 1,
                },
            )
        reason = f"merged into {merge.keep} by {OVERRIDES_FILENAME} ({merge.evidence or 'no evidence cited'})"
        self._append(
            "securities",
            self._status_row(
                merge.absorb, "quarantined", valid_at=stamp, source=ATTESTATION_SOURCE, quarantine_reason=reason
            ),
        )

    # -- reads -----------------------------------------------------------

    def _has_security(self, sid: str) -> bool:
        return any(str(row["sid"]) == str(sid) for row in self._rows["securities"])

    def _latest_security(self, sid: str) -> dict[str, Any] | None:
        rows = [row for row in self._rows["securities"] if str(row["sid"]) == str(sid)]
        if not rows:
            return None
        return max(rows, key=lambda row: (pd.Timestamp(row["valid_at"]), int(row["revision"])))

    def status_of(self, sid: str) -> str:
        """The current ``entity_status`` of ``sid``, from its latest ``securities`` row."""
        latest = self._latest_security(sid)
        if latest is None:
            raise SecurityMasterError(f"no securities row names {sid}")
        return str(latest["entity_status"])

    def sids(self) -> tuple[str, ...]:
        return tuple(sorted({str(row["sid"]) for row in self._rows["securities"]}))

    def _live_listings(self) -> list[dict[str, Any]]:
        """The highest revision of every ``listings`` key, with deleted rows dropped.

        Deletion is a null-valued full revision (§3.2), so a row whose latest
        revision carries no ticker no longer maps anything.

        Cached and invalidated on every listings append: an import of a real
        universe asks this question twice per name and once per membership
        interval, and rebuilding it each time is what makes the sweep
        quadratic.
        """
        if self._live_cache is not None:
            return self._live_cache
        latest: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in self._rows["listings"]:
            key = tuple(row[column] for column in TABLE_KEYS["listings"]) + (str(row["source"]),)
            current = latest.get(key)
            if current is None or int(row["revision"]) > int(current["revision"]):
                latest[key] = row
        self._live_cache = [row for row in latest.values() if row["ticker"]]
        return self._live_cache

    def resolve(self, ticker: str, on: str | dt.date) -> str:
        """The single sid whose listing for ``ticker`` is valid on ``on``.

        Single-valued by contract: zero or several candidates raise. Two vendor
        mappings for one instrument overlapping on that date are one answer,
        because they carry one sid — that is what the surrogate key buys.

        This is the writer-side resolve, over every row the master holds. The
        reader-side, as-of resolve is ``DatasetView.resolve``, which applies
        the visibility predicate first.
        """
        day = _as_date(on)
        wanted = normalize_ticker(ticker)
        candidates = sorted(
            {
                str(row["sid"])
                for row in self._live_listings()
                if row["ticker"] == wanted
                and Segment(wanted, _opt_date(row["valid_from"]), _opt_date(row["valid_to"])).contains(day)
            }
        )
        if len(candidates) != 1:
            raise SecurityMasterError(
                f"{wanted!r} on {day.isoformat()} resolves to {candidates or 'no sid'}; a ticker must be "
                "single-valued on a date"
            )
        return candidates[0]

    def resolve_interval(self, ticker: str, valid_from: dt.date | None, valid_to: dt.date | None) -> str:
        """The single sid whose listing for ``ticker`` overlaps ``[valid_from, valid_to]``.

        What a membership interval needs: an index holds an *instrument* over a
        period, and a ticker reused inside that period would name two, which is
        a question the data cannot answer and the caller must.
        """
        wanted = normalize_ticker(ticker)
        probe = Segment(wanted, valid_from, valid_to)
        candidates = sorted(
            {
                str(row["sid"])
                for row in self._live_listings()
                if row["ticker"] == wanted and probe.overlaps(_opt_date(row["valid_from"]), _opt_date(row["valid_to"]))
            }
        )
        if len(candidates) != 1:
            raise SecurityMasterError(
                f"{wanted!r} over {probe.label} resolves to {candidates or 'no sid'}; an index interval names one "
                "instrument"
            )
        return candidates[0]

    def sid_for_vendor_symbol(self, vendor: str, vendor_symbol: str, on: str | dt.date) -> str:
        """The sid a vendor's own symbol denoted on ``on`` — the broker-to-research join.

        A position the broker reports in its symbology and a bar series the
        research vendor supplies meet here, on one sid, which is the reason the
        identifier is vendor-neutral (D2).
        """
        day = _as_date(on)
        candidates = sorted(
            {
                str(row["sid"])
                for row in self._live_listings()
                if str(row["vendor"]) == str(vendor)
                and str(row["vendor_symbol"]) == str(vendor_symbol)
                and Segment(str(row["ticker"]), _opt_date(row["valid_from"]), _opt_date(row["valid_to"])).contains(day)
            }
        )
        if len(candidates) != 1:
            raise SecurityMasterError(
                f"{vendor}:{vendor_symbol} on {day.isoformat()} resolves to {candidates or 'no sid'}; a vendor symbol "
                "must be single-valued on a date"
            )
        return candidates[0]

    def unresolvable_tickers(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Tickers whose live listing intervals overlap across two sids, so :meth:`resolve` must refuse.

        Reported rather than refused at write time: a segment that matched more
        than one sid mints a third by design (§3.2), so the collision is a
        state awaiting attestation, and the refusal belongs at the query that
        would otherwise get a silent winner.
        """
        by_ticker: dict[str, list[dict[str, Any]]] = {}
        for row in self._live_listings():
            by_ticker.setdefault(str(row["ticker"]), []).append(row)
        out: list[tuple[str, tuple[str, ...]]] = []
        for ticker, rows in sorted(by_ticker.items()):
            colliding: set[str] = set()
            for index, row in enumerate(rows):
                probe = Segment(ticker, _opt_date(row["valid_from"]), _opt_date(row["valid_to"]))
                for other in rows[index + 1 :]:
                    if str(other["sid"]) == str(row["sid"]):
                        continue
                    if probe.overlaps(_opt_date(other["valid_from"]), _opt_date(other["valid_to"])):
                        colliding.update({str(row["sid"]), str(other["sid"])})
            if colliding:
                out.append((ticker, tuple(sorted(colliding))))
        return tuple(out)

    # -- frames, validation, output --------------------------------------

    def frame(self, name: str) -> pd.DataFrame:
        """One table as a DataFrame in the §3.2 column order; empty tables keep their columns."""
        if name not in TABLE_COLUMNS:
            raise SecurityMasterError(f"{name!r} is not one of {TABLE_NAMES}")
        columns = list(TABLE_COLUMNS[name])
        rows = self._rows[name]
        if not rows:
            return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
        frame = pd.DataFrame(rows, columns=columns)
        frame["revision"] = frame["revision"].astype("int64")
        if name == "membership":
            frame["floor_unknown"] = frame["floor_unknown"].astype(bool)
        return frame.sort_values(list(TABLE_KEYS[name]) + ["source", "revision"]).reset_index(drop=True)

    def validate(self) -> None:
        """Refuse the structural defects §3.2 says a view must never read.

        Two rows sharing ``(key, source, revision)``; a listing or membership
        row naming a sid the ``securities`` table does not; an unknown
        ``entity_status``; a quarantine with no reason.
        """
        for name in TABLE_NAMES:
            seen: set[tuple[Any, ...]] = set()
            for row in self._rows[name]:
                identity = tuple(row[column] for column in TABLE_KEYS[name]) + (row["source"], row["revision"])
                if identity in seen:
                    raise SecurityMasterError(
                        f"{name} carries two rows with the same (key, source, revision) {identity}; §3.2 calls that an "
                        "ingestion defect and the view refuses to read it"
                    )
                seen.add(identity)
        known = set(self.sids())
        for name in ("listings", "membership"):
            unknown = sorted({str(row["sid"]) for row in self._rows[name]} - known)
            if unknown:
                raise SecurityMasterError(f"{name} names sid(s) {unknown} that the securities table does not")
        for row in self._rows["securities"]:
            if row["entity_status"] not in ENTITY_STATUSES:
                raise SecurityMasterError(
                    f"securities row {row['sid']} carries entity_status {row['entity_status']!r}, "
                    f"not one of {ENTITY_STATUSES}"
                )
            if row["entity_status"] == "quarantined" and not row["quarantine_reason"]:
                raise SecurityMasterError(
                    f"securities row {row['sid']} is quarantined with no quarantine_reason; a quarantine that does "
                    "not say why cannot be reviewed"
                )

    def report(self) -> SecurityMasterReport:
        """The counts §3.2 requires, computed on the latest status of every sid."""
        statuses = {sid: self.status_of(sid) for sid in self.sids()}
        provisional = tuple(sid for sid, status in statuses.items() if status == "provisional")
        quarantined = tuple(sid for sid, status in statuses.items() if status == "quarantined")
        verified = tuple(sid for sid, status in statuses.items() if status == "verified")
        return SecurityMasterReport(
            n_sids=len(statuses),
            n_provisional_sids=len(provisional),
            n_verified_sids=len(verified),
            n_quarantined_sids=len(quarantined),
            provisional_sids=provisional,
            quarantined_sids=quarantined,
            attestation_flags=tuple(self._attestation_flags),
            gap_flags=tuple(self._gap_flags),
            unresolvable_tickers=self.unresolvable_tickers(),
            n_listings=len(self._rows["listings"]),
            n_membership=len(self._rows["membership"]),
            gap_flag_sessions=self.gap_flag_sessions,
        )

    def write(self, root: Path) -> tuple[Path, ...]:
        """Apply the attestations, validate, and write the three tables under ``root``.

        All three are written even when empty: a view reads only files the
        manifest lists, and "no listings table" and "no listings" are different
        statements.
        """
        self.apply_overrides()
        self.validate()
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for name in TABLE_NAMES:
            path = root / f"{name}.parquet"
            self.frame(name).to_parquet(path, index=False)
            written.append(path)
        return tuple(written)


def _content_equal(stored: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return {k: v for k, v in stored.items() if k not in _VOLATILE_FIELDS} == {
        k: v for k, v in candidate.items() if k not in _VOLATILE_FIELDS
    }


def _clean(value: Any) -> Any:
    """Normalize a value read back from Parquet: pandas nulls become ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return None if pd.isna(value) else value


__all__ = [
    "ASSET_CLASSES",
    "ATTESTATION_SOURCE",
    "Alias",
    "Attestation",
    "AttestationFlag",
    "DEFAULT_GAP_FLAG_SESSIONS",
    "ENTITY_STATUSES",
    "EPOCH_INSTANT",
    "GapFlag",
    "IdentityEvent",
    "LISTINGS_COLUMNS",
    "MEMBERSHIP_COLUMNS",
    "MembershipImport",
    "Merge",
    "OVERRIDES_FILENAME",
    "OVERRIDES_SCHEMA",
    "SECURITIES_COLUMNS",
    "SID_FORMAT",
    "Segment",
    "SecurityMaster",
    "SecurityMasterError",
    "SecurityMasterReport",
    "SecurityOverrides",
    "SegmentResolution",
    "SeriesResolution",
    "TABLE_COLUMNS",
    "TABLE_KEYS",
    "TABLE_NAMES",
    "missing_sessions_between",
    "segment_series",
    "sid_number",
    "sid_of",
]
