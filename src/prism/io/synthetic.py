"""Deterministic synthetic fixtures in the §3.2 schema (engine spec §3.2, "Synthetic fixtures").

``prism data synthetic --seed <n> --scenarios <list>`` writes a complete
dataset with a manifest whose ``provenance`` is ``synthetic``, so the worked
example of section 4 runs with no credential, no network call and no vendor
licence. Every row carries ``synthetic = true``; every record derived from
such a manifest inherits that flag (§3.2, [Q11]).

Deterministic means reproducible from the seed alone, not merely
"random-looking". Each sid draws from its own stream, keyed by the seed and
the sid, so adding a name or turning a scenario off does not move any other
name's prices, and regenerating with the same seed and scenario set gives
byte-identical files and therefore the same ``manifest.id``.

Scenarios and stages. The twelve scenario ids of §3.2 are written here in the
final schema; the stage that first *consumes* each differs from the stage that
writes it. Stage 1 exercises ``membership``, ``distribution``, ``costs`` and
``rounding``. The stage 3 scenarios (``spinoff``, ``ticker_reuse``,
``revision``, ``rebase``, ``delisting``) and the stage 4 ``recovery`` scenario
are generated as data and are not applied to *accounting*: they are written onto
sids that belong to no index, so the stage 1 universe never selects them, and
the manifest's ``assumptions`` says so by name. ``split`` and ``halt`` are
stage 1 scenarios that the retained reproduction engine cannot yet interpret
(section 4's second table), so their rows stay in the universe and the
manifest records what a stage 1 evaluation will make of them.

From stage 3a every scenario is applied *through the data layer*: each one is
read back through a :class:`~prism.io.dataset.DatasetView` and checked against
what §3.2 says the answer is, in ``tests/test_synthetic_scenarios.py``. Stage 3b
applies them to accounting through the canonical simulator, driving the same
cases by id as the adversarial session set.

The scenario catalog. Every scenario the generator writes also registers a
:class:`ScenarioCase` naming the sids it touches by role, the dates that decide
it, and the three-bar window a decision runs over — the prior close the starting
state is built at, the decision close ``t``, and the bar whose open fills and
whose close marks. The catalog is carried in the manifest's ``coverage["scenarios"]``
and read back with :func:`scenario_catalog`, so a scenario is *addressable by
id* rather than by a sid a caller had to rediscover. That is the contract the
stage 3b adversarial session set is built on (§5, "The adversarial session
set"): one scenario id, one three-bar sequence, and the ledger invariants the
sequence must satisfy.

Ingestion model. Every row declares ``ingested_at`` at 20:00 America/New_York
on its own *record* date — the announce date where one exists, otherwise the
effective date, the opening date of the interval, or the bar's own session.
That is what a workspace maintained nightly holds, and it is what makes
``replay_recorded`` answerable *inside* the sample: one constant ingestion
instant would leave every row invisible until the last session, so every
earlier decision would be a refusal and no scenario could be driven in that
mode. A bar revision declares its own availability and ingestion instants, so
a correction is invisible in both modes until the day it landed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from prism.io import sessions
from prism.io.manifest import (
    Manifest,
    ManifestError,
    canonical_json,
    hash_tree,
    sha256_bytes,
)

_ET = ZoneInfo("America/New_York")

BAR_INTERVAL = "1d"

# Source ids. Two, because the `split` scenario needs one series the vendor had
# already rebased and one printed raw, and `price_basis` is a source property
# as well as a row property (§3.2).
SOURCE_ADJUSTED = "synthetic"
SOURCE_RAW = "synthetic_raw"

# The default gap flag of §3.2: a segment with more than this many missing
# sessions is a coverage flag, not a silent hole.
GAP_FLAG_SESSIONS = 15

EQUITY_INDEX = "synthetic_100"
ETF_INDEX = "synthetic_etf_10"

VENDOR = "synthetic"
PRIMARY_LISTING = "XNYS"

# The hour a nightly-maintained workspace stores the day's rows, and the hour a
# source publishes a correction. Neither is a market event: ingestion time is
# never a knowledge timestamp (§3.2, "Availability and the decision-time
# cutoff"), which is why the two are separate fields and separate clocks.
INGESTION_HOUR = 20
CORRECTION_HOUR = 9


@dataclass(frozen=True, slots=True)
class Scenario:
    """One row of the §3.2 scenario table."""

    name: str
    exercises: str
    stage: int


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario("membership", "an add and a remove inside the sample", 1),
        Scenario("split", "a split event on a raw series and on an adjusted one", 1),
        Scenario("distribution", "a cash_distribution with ex and pay dates on either side of a decision", 1),
        Scenario("halt", "a multi-session halt, below gap_flag_sessions", 1),
        Scenario("costs", "spread buckets and a participation bound that binds", 1),
        Scenario("rounding", "a per-sid quantity_increment and a repair that scales the batch", 1),
        Scenario("spinoff", "a spinoff with a tracked child and cash in lieu", 3),
        Scenario("ticker_reuse", "a ticker reused after a delisting, yielding two sids", 3),
        Scenario("revision", "a late revision and a null-valued full revision", 3),
        Scenario("rebase", "a full-series bars revision with a matching adjustments row", 3),
        Scenario("delisting", "a delisting with a terminal payout and one without", 3),
        Scenario("recovery", "an inherited breach and an open obligation after a data gap", 4),
    )
}


@dataclass(frozen=True, slots=True)
class ScenarioCase:
    """One scenario, addressable by id: which sids, which dates, which bars.

    The three bars are the stage 3b adversarial session's window (§5, "The
    adversarial session set"). ``prior_bar`` is the close the starting
    portfolio state is built at, ``decision_bar`` is the close ``t`` a decision
    is taken on, and ``fill_bar`` is the bar whose open fills the resulting
    order and whose close marks the book. The scenario's effect lands on
    ``fill_bar``, so the session straddles it: a split, an ex-date, a halt or a
    cessation is between the decision and the mark, never comfortably before
    both. A distribution's entitlement is therefore fixed at ``decision_bar``,
    the close before the ex-date (§3.6, the ex-date Correction).

    ``sids`` is keyed by role (``"parent"``, ``"child"``, ``"paid"``,
    ``"unknown"``, …) so a caller names what it wants rather than indexing a
    list. ``dates`` carries the ISO dates the scenario turns on and nothing
    else; the scalars a caller needs to check an answer against — a split
    factor, an amount per share, a reused ticker — are in ``values``, kept
    separate so ``date()`` never hands back something that is not a date.
    ``expects`` is the sentence the data layer must satisfy, carried with the
    case so a fixture that stops exercising it is visible in the manifest
    rather than only in a test that stopped running.
    """

    scenario: str
    sids: dict[str, str]
    dates: dict[str, str]
    values: dict[str, str]
    prior_bar: str
    decision_bar: str
    fill_bar: str
    expects: str

    def __post_init__(self) -> None:
        for role, day in self.dates.items():
            try:
                dt.date.fromisoformat(day)
            except ValueError as exc:
                raise SyntheticError(
                    f"scenario {self.scenario!r} carries {role!r} = {day!r} in dates, which is not an ISO date; "
                    "a scalar belongs in values"
                ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "sids": dict(self.sids),
            "dates": dict(self.dates),
            "values": dict(self.values),
            "prior_bar": self.prior_bar,
            "decision_bar": self.decision_bar,
            "fill_bar": self.fill_bar,
            "expects": self.expects,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenarioCase":
        return cls(
            scenario=str(payload["scenario"]),
            sids={str(k): str(v) for k, v in dict(payload["sids"]).items()},
            dates={str(k): str(v) for k, v in dict(payload["dates"]).items()},
            values={str(k): str(v) for k, v in dict(payload.get("values", {})).items()},
            prior_bar=str(payload["prior_bar"]),
            decision_bar=str(payload["decision_bar"]),
            fill_bar=str(payload["fill_bar"]),
            expects=str(payload["expects"]),
        )

    def sid(self, role: str) -> str:
        """The sid playing ``role``, refusing a role this scenario does not cast."""
        try:
            return self.sids[role]
        except KeyError as exc:
            raise SyntheticError(
                f"scenario {self.scenario!r} casts no {role!r}; it names {sorted(self.sids)}"
            ) from exc

    def date(self, role: str) -> str:
        """The ISO date playing ``role``, refusing one this scenario does not carry."""
        try:
            return self.dates[role]
        except KeyError as exc:
            raise SyntheticError(
                f"scenario {self.scenario!r} carries no {role!r} date; it names {sorted(self.dates)}"
            ) from exc

    def value(self, role: str) -> str:
        """The scalar playing ``role``, as text; refusing one this scenario does not carry."""
        try:
            return self.values[role]
        except KeyError as exc:
            raise SyntheticError(
                f"scenario {self.scenario!r} carries no {role!r} value; it names {sorted(self.values)}"
            ) from exc

    def number(self, role: str) -> float:
        """:meth:`value` parsed as a float, refusing text that is not a number."""
        text = self.value(role)
        try:
            return float(text)
        except ValueError as exc:
            raise SyntheticError(f"scenario {self.scenario!r} value {role!r} = {text!r} is not a number") from exc


def scenario_catalog(manifest: Manifest) -> dict[str, ScenarioCase]:
    """The scenario cases a synthetic manifest carries, by scenario id.

    Refuses a manifest that is not synthetic: reading a market dataset for
    scenario cases would silently return nothing and read as "no scenario
    failed" rather than "no scenario ran".
    """
    if manifest.provenance != "synthetic":
        raise SyntheticError(
            f"manifest {manifest.id} has provenance {manifest.provenance!r}; only a synthetic dataset carries "
            "scenario cases"
        )
    payload = manifest.coverage.get("scenarios")
    if not isinstance(payload, dict):
        raise SyntheticError(f"manifest {manifest.id} carries no coverage['scenarios'] catalog")
    return {name: ScenarioCase.from_dict(case) for name, case in sorted(payload.items())}


def scenario_case(manifest: Manifest, scenario: str) -> ScenarioCase:
    """One scenario case by id, refusing an id this dataset did not write."""
    catalog = scenario_catalog(manifest)
    try:
        return catalog[scenario]
    except KeyError as exc:
        raise SyntheticError(f"this dataset wrote no {scenario!r} scenario; it wrote {sorted(catalog)}") from exc


ALL_SCENARIOS: tuple[str, ...] = tuple(SCENARIOS)
STAGE1_SCENARIOS: tuple[str, ...] = tuple(n for n, s in SCENARIOS.items() if s.stage == 1)

# The two scenarios whose sids sit inside the equity index -- so a stage 1
# evaluation selects and prices them -- while the effect the scenario exists to
# test is not applied to accounting. Section 4's "written but not exercised"
# table lists them alongside the later-stage scenarios, so the smoke summary
# must name them too; they are a separate list because the reason differs. A
# later-stage scenario is invisible to the evaluation; these two are visible
# and read wrong (a raw split reads as a price return, a halted session is
# forward-filled), which is the more dangerous of the two silences.
STAGE1_IN_UNIVERSE_UNAPPLIED: tuple[str, ...] = ("split", "halt")

# Scenarios stage 1 writes but does not apply to accounting, with the stage
# that applies them. Copied into the manifest so a reader of the dataset never
# has to infer it from the code that wrote it.
APPLIED_AT_STAGE: dict[str, int] = {name: s.stage for name, s in SCENARIOS.items()}

# Stage 1 scenario sids are drawn from the head of the equity universe, so a
# test can name them; the later-stage scenarios get sids past the end of both
# indices, which is what keeps them out of every stage 1 evaluation.
_STAGE1_SLOTS: dict[str, tuple[int, ...]] = {
    "split": (1, 2),
    "distribution": (3,),
    "halt": (4,),
    "costs": (5, 6, 7, 8),
    "rounding": (9,),
    "membership": (10, 11),
}
_EXTRA_SLOTS: dict[str, tuple[int, ...]] = {
    "spinoff": (1, 2),
    "ticker_reuse": (3, 4),
    "revision": (5,),
    "rebase": (6,),
    "delisting": (7, 8),
    "recovery": (9,),
}

MIN_EQUITIES = 12
MIN_ETFS = 2
MIN_SESSIONS = 60
MIN_SESSIONS_FOR_RECOVERY = 80

# The stage 4 recovery scenario needs a gap the coverage sweep flags; 20 > 15.
RECOVERY_GAP_SESSIONS = 20
HALT_SESSIONS = 3
ANNOUNCE_LEAD_SESSIONS = 5
SPLIT_FACTOR = 2.0
REBASE_FACTOR = 4.0
SPINOFF_RATIO = 0.25
# The terminal payout of the resolved delisting, as a multiple of the last
# close. Not 1.0: a payout equal to the last close could not distinguish "the
# claim settled at the payout" from "the last close was carried forward".
TERMINAL_PAYOUT_MULTIPLE = 1.05
# The late correction of the `revision` scenario, and the sessions after the
# corrected bar at which the correction and then its withdrawal land.
CORRECTION_MULTIPLE = 1.10
CORRECTION_LAG_SESSIONS = 4
WITHDRAWAL_LAG_SESSIONS = 8

# Dollar-volume levels that straddle the reference spread schedule of section 4
# ([[500000000, 1], [100000000, 2], [25000000, 5], [0, 10]]), plus one name thin
# enough that a 5 percent participation cap binds on a small account.
_COST_DOLLAR_VOLUMES: tuple[float, ...] = (8.0e8, 2.0e8, 5.0e7, 8.0e5)


class SyntheticError(ValueError):
    """The requested fixture cannot be generated as asked."""


@dataclass(frozen=True)
class SyntheticSpec:
    """What to generate. Two datasets with equal specs are byte-identical."""

    seed: int
    scenarios: tuple[str, ...]
    n_equities: int = 100
    n_etfs: int = 10
    start: dt.date | None = None
    end: dt.date | None = None
    vintage: str = "synthetic_v1"
    sessions: tuple[dt.date, ...] = field(default_factory=tuple, compare=False, repr=False)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.scenarios) - set(ALL_SCENARIOS))
        if unknown:
            raise SyntheticError(f"unknown scenarios {unknown}; the twelve ids are {list(ALL_SCENARIOS)}")
        if self.n_equities < MIN_EQUITIES:
            raise SyntheticError(f"n_equities must be >= {MIN_EQUITIES} to place the stage 1 scenario sids")
        if self.n_etfs < MIN_ETFS:
            raise SyntheticError(f"n_etfs must be >= {MIN_ETFS}")
        covered_start, covered_end = sessions.coverage()
        grid = sessions.sessions_between(self.start or covered_start, self.end or covered_end)
        if len(grid) < MIN_SESSIONS:
            raise SyntheticError(f"the fixture needs at least {MIN_SESSIONS} sessions, got {len(grid)}")
        if "recovery" in self.scenarios and len(grid) < MIN_SESSIONS_FOR_RECOVERY:
            raise SyntheticError(
                f"the recovery scenario writes a {RECOVERY_GAP_SESSIONS}-session gap above the "
                f"{GAP_FLAG_SESSIONS}-session flag and needs at least {MIN_SESSIONS_FOR_RECOVERY} sessions"
            )
        object.__setattr__(self, "scenarios", tuple(sorted(set(self.scenarios))))
        object.__setattr__(self, "sessions", tuple(grid))


def resolve_scenarios(value: str | Sequence[str]) -> tuple[str, ...]:
    """``"all"``, a comma-separated string, or a sequence, into the sorted scenario tuple."""
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in value if str(part).strip()]
    if names == ["all"]:
        return ALL_SCENARIOS
    if "all" in names:
        raise SyntheticError("'all' names every scenario and cannot be combined with a subset")
    unknown = sorted(set(names) - set(ALL_SCENARIOS))
    if unknown:
        raise SyntheticError(f"unknown scenarios {unknown}; the twelve ids are {list(ALL_SCENARIOS)}")
    if not names:
        raise SyntheticError("--scenarios takes 'all' or at least one scenario id")
    return tuple(sorted(set(names)))


def sid_of(number: int) -> str:
    """The §3.2 minting form: the ``sec_%06d`` surrogate key."""
    return f"sec_{number:06d}"


def _stream(seed: int, salt: str) -> np.random.Generator:
    """A generator keyed by ``(seed, salt)``, independent of every other salt."""
    entropy = int.from_bytes(hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([int(seed), entropy]))


def _instant_at(day: dt.date, hour: int = INGESTION_HOUR, minute: int = 0) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=_ET)


def _instant(day: dt.date, hour: int = INGESTION_HOUR, minute: int = 0) -> str:
    return _instant_at(day, hour, minute).isoformat()


def _session_close_at(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, sessions.session_close(day), tzinfo=_ET)


def _session_close_instant(day: dt.date) -> str:
    return _session_close_at(day).isoformat()


def _later(*instants: dt.datetime) -> str:
    """The latest of several instants, compared as datetimes rather than as text.

    Two ISO strings on either side of a daylight-saving change carry different
    offsets, so string order is not time order; a rebase restated across March
    would otherwise publish a bar before its own close.
    """
    return max(instants).isoformat()


def _payload_sha(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class _Instrument:
    sid: str
    ticker: str
    name: str
    asset_class: Literal["equity", "etf"]
    in_index: str | None
    source: str
    price_basis: str
    dollar_volume: float


def _price_frame(spec: SyntheticSpec, instrument: _Instrument) -> pd.DataFrame:
    """A geometric random walk with per-name drift and volatility, on the session grid."""
    grid = list(spec.sessions)
    n = len(grid)
    rng = _stream(spec.seed, instrument.sid)
    start_price = float(rng.uniform(15.0, 200.0))
    drift = float(rng.normal(0.0002, 0.0003))
    vol = float(rng.uniform(0.010, 0.030))
    shocks = rng.normal(drift, vol, size=n)
    close = start_price * np.exp(np.cumsum(shocks))
    gaps = rng.normal(0.0, vol / 3.0, size=n)
    open_ = np.empty(n, dtype=float)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1.0 + gaps[1:])
    span = np.abs(rng.normal(0.0, vol / 2.0, size=n))
    high = np.maximum(open_, close) * (1.0 + span)
    low = np.minimum(open_, close) * (1.0 - span)
    noise = np.exp(rng.normal(0.0, 0.35, size=n))
    volume = np.maximum(np.round(instrument.dollar_volume * noise / close), 1.0)
    return pd.DataFrame(
        {
            "ts": pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day in grid]),
            "open": np.round(open_, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "volume": volume,
        }
    )


def _instruments(spec: SyntheticSpec) -> list[_Instrument]:
    """The universe plus the sids the later-stage scenarios need, in mint order."""
    out: list[_Instrument] = []
    cost_sids = {sid_of(slot): level for slot, level in zip(_STAGE1_SLOTS["costs"], _COST_DOLLAR_VOLUMES)}
    raw_split_sid = sid_of(_STAGE1_SLOTS["split"][0])
    for number in range(1, spec.n_equities + 1):
        sid = sid_of(number)
        rng = _stream(spec.seed, f"dollar_volume:{sid}")
        level = float(np.exp(rng.uniform(np.log(2.0e7), np.log(2.0e9))))
        if "costs" in spec.scenarios and sid in cost_sids:
            level = cost_sids[sid]
        raw = "split" in spec.scenarios and sid == raw_split_sid
        out.append(
            _Instrument(
                sid=sid,
                ticker=f"SYN{number:03d}",
                name=f"Synthetic Equity {number:03d}",
                asset_class="equity",
                in_index=EQUITY_INDEX,
                source=SOURCE_RAW if raw else SOURCE_ADJUSTED,
                price_basis="raw" if raw else "split_adjusted_at_ingest",
                dollar_volume=level,
            )
        )
    for offset in range(1, spec.n_etfs + 1):
        number = spec.n_equities + offset
        sid = sid_of(number)
        rng = _stream(spec.seed, f"dollar_volume:{sid}")
        out.append(
            _Instrument(
                sid=sid,
                ticker=f"ETF{offset:03d}",
                name=f"Synthetic ETF {offset:03d}",
                asset_class="etf",
                in_index=ETF_INDEX,
                source=SOURCE_ADJUSTED,
                price_basis="split_adjusted_at_ingest",
                dollar_volume=float(np.exp(rng.uniform(np.log(1.0e8), np.log(3.0e9)))),
            )
        )
    base = spec.n_equities + spec.n_etfs
    for scenario, slots in _EXTRA_SLOTS.items():
        if scenario not in spec.scenarios:
            continue
        for position, slot in enumerate(slots):
            number = base + slot
            sid = sid_of(number)
            ticker = "RUSE" if scenario == "ticker_reuse" else f"X{scenario[:3].upper()}{position + 1}"
            rng = _stream(spec.seed, f"dollar_volume:{sid}")
            out.append(
                _Instrument(
                    sid=sid,
                    ticker=ticker,
                    name=f"Synthetic {scenario} instrument {position + 1}",
                    asset_class="equity",
                    in_index=None,
                    source=SOURCE_ADJUSTED,
                    price_basis="split_adjusted_at_ingest",
                    dollar_volume=float(np.exp(rng.uniform(np.log(2.0e7), np.log(4.0e8)))),
                )
            )
    return out


def _extra_sid(spec: SyntheticSpec, scenario: str, position: int) -> str:
    return sid_of(spec.n_equities + spec.n_etfs + _EXTRA_SLOTS[scenario][position])


class _Builder:
    """Accumulates the tables, then writes them and the manifest."""

    def __init__(self, root: Path, spec: SyntheticSpec) -> None:
        self.root = Path(root)
        self.spec = spec
        self.grid = list(spec.sessions)
        self.n = len(self.grid)
        self.instruments = _instruments(spec)
        self.by_sid = {inst.sid: inst for inst in self.instruments}
        self.bars: dict[str, pd.DataFrame] = {inst.sid: _price_frame(spec, inst) for inst in self.instruments}
        self.bar_revisions: dict[str, list[pd.DataFrame]] = {}
        self.securities: list[dict[str, Any]] = []
        self.listings: list[dict[str, Any]] = []
        self.membership: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.adjustments: list[dict[str, Any]] = []
        self.capabilities: list[dict[str, Any]] = []
        self.coverage_flags: list[dict[str, Any]] = []
        self.bar_gaps: list[dict[str, Any]] = []
        self.cases: dict[str, ScenarioCase] = {}

    # -- grid helpers ---------------------------------------------------

    def at(self, fraction: float) -> dt.date:
        return self.grid[max(0, min(self.n - 1, int(self.n * fraction)))]

    def index_of(self, day: dt.date) -> int:
        return self.grid.index(day)

    def shift(self, day: dt.date, sessions_back: int) -> dt.date:
        return self.grid[max(0, self.index_of(day) - sessions_back)]

    def ahead(self, day: dt.date, sessions_forward: int) -> dt.date:
        return self.grid[min(self.n - 1, self.index_of(day) + sessions_forward)]

    def window(self, fill_bar: dt.date) -> tuple[str, str, str]:
        """The three-bar adversarial window ending at ``fill_bar`` (§5)."""
        decision = self.shift(fill_bar, 1)
        prior = self.shift(fill_bar, 2)
        return prior.isoformat(), decision.isoformat(), fill_bar.isoformat()

    def record_case(
        self,
        scenario: str,
        *,
        sids: dict[str, str],
        dates: dict[str, str],
        fill_bar: dt.date,
        expects: str,
        values: dict[str, str] | None = None,
    ) -> None:
        prior, decision, fill = self.window(fill_bar)
        self.cases[scenario] = ScenarioCase(
            scenario=scenario,
            sids=dict(sids),
            dates=dict(dates),
            values=dict(values or {}),
            prior_bar=prior,
            decision_bar=decision,
            fill_bar=fill,
            expects=expects,
        )

    def note_gap(self, sid: str, missing: Sequence[dt.date]) -> None:
        """Record a run of absent sessions, flagged only when it exceeds ``gap_flag_sessions``.

        Both sides are recorded, not only the flagged one: a fixture that
        carries no unflagged gap cannot show that the threshold discriminates,
        and "the flag fires above 15 sessions and not below" is a two-sided
        claim (§3.2, "Tests").
        """
        entry = {
            "sid": sid,
            "segment": f"{missing[0].isoformat()}/{missing[-1].isoformat()}",
            "missing_sessions": len(missing),
            "gap_flag_sessions": GAP_FLAG_SESSIONS,
            "flagged": len(missing) > GAP_FLAG_SESSIONS,
        }
        self.bar_gaps.append(entry)
        if entry["flagged"]:
            self.coverage_flags.append(entry)

    # -- row builders ---------------------------------------------------

    def _stamp(self, source: str, on: dt.date, *, available: str | None = None) -> dict[str, Any]:
        """The four provenance columns every table carries, stamped on ``on``'s night."""
        return {
            "available_at": available,
            "ingested_at": _instant(on, hour=INGESTION_HOUR),
            "source": source,
            "revision": 1,
            "synthetic": True,
        }

    def add_security(self, inst: _Instrument, *, valid_on: dt.date | None = None, **overrides: Any) -> None:
        day = self.grid[0] if valid_on is None else valid_on
        row: dict[str, Any] = {
            "sid": inst.sid,
            "valid_at": _session_close_instant(day),
            "entity_status": "verified",
            "quarantine_reason": None,
            "name": inst.name,
            "asset_class": inst.asset_class,
            "share_class": "A",
            "primary_listing": PRIMARY_LISTING,
            **self._stamp(SOURCE_ADJUSTED, day),
        }
        row.update(overrides)
        self.securities.append(row)

    def add_listing(self, sid: str, ticker: str, valid_from: str | None, valid_to: str | None, **kw: Any) -> None:
        opens = self.grid[0] if valid_from is None else dt.date.fromisoformat(valid_from)
        announce = kw.get("announce_date")
        known_on = dt.date.fromisoformat(announce) if announce else opens
        row: dict[str, Any] = {
            "sid": sid,
            "ticker": ticker,
            "vendor": VENDOR,
            "vendor_symbol": ticker,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "announce_date": None,
            **self._stamp(SOURCE_ADJUSTED, known_on),
        }
        row.update(kw)
        self.listings.append(row)

    def add_membership(
        self, index_id: str, sid: str, valid_from: str | None, valid_to: str | None, announce: str | None = None
    ) -> None:
        opens = self.grid[0] if valid_from is None else dt.date.fromisoformat(valid_from)
        known_on = dt.date.fromisoformat(announce) if announce else opens
        self.membership.append(
            {
                "index_id": index_id,
                "sid": sid,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "floor_unknown": valid_from is None,
                "announce_date": announce,
                **self._stamp(SOURCE_ADJUSTED, known_on),
            }
        )

    def add_event(
        self, sid: str, event_type: str, effective: dt.date, payload: dict[str, Any], announce: dt.date | None = None
    ) -> None:
        self.events.append(
            {
                "sid": sid,
                "event_type": event_type,
                "effective_date": effective.isoformat(),
                "announce_date": None if announce is None else announce.isoformat(),
                "payload": canonical_json(payload),
                **self._stamp(SOURCE_ADJUSTED, announce or effective),
            }
        )

    def add_adjustment(self, sid: str, effective: dt.date, kind: str, factor: float, applies_from: dt.date) -> None:
        self.adjustments.append(
            {
                "sid": sid,
                "effective_date": effective.isoformat(),
                "kind": kind,
                "factor": float(factor),
                "applies_from": applies_from.isoformat(),
                **self._stamp(SOURCE_ADJUSTED, effective),
            }
        )

    def add_capability(self, sid: str, valid_on: dt.date, *, quantity_increment: float = 1.0, **kw: Any) -> None:
        row: dict[str, Any] = {
            "sid": sid,
            "valid_at": _session_close_instant(valid_on),
            "fractionable": False,
            "quantity_increment": quantity_increment,
            "min_notional": 1.0,
            "shortable": True,
            "easy_to_borrow": True,
            "lot_size": 100,
            "auction_eligible": True,
            "restrictions": json.dumps([]),
            "order_types": json.dumps(["market_on_open"]),
            "sessions": json.dumps(["regular"]),
            **self._stamp(SOURCE_ADJUSTED, valid_on),
        }
        row.update(kw)
        self.capabilities.append(row)

    # -- construction ---------------------------------------------------

    def build(self) -> None:
        first = self.grid[0].isoformat()
        for inst in self.instruments:
            self.add_security(inst)
            self.add_listing(inst.sid, inst.ticker, first, None)
            self.add_capability(inst.sid, self.grid[0])
            if inst.in_index is not None:
                self.add_membership(inst.in_index, inst.sid, first, None)
        for scenario in self.spec.scenarios:
            getattr(self, f"_scenario_{scenario}")()
        missing = sorted(set(self.spec.scenarios) - set(self.cases))
        if missing:  # a scenario that registers no case is a fixture that stopped being exercised
            raise SyntheticError(f"scenarios {missing} wrote no ScenarioCase; every scenario is addressable by id")

    def _scenario_membership(self) -> None:
        """One add and one remove inside the sample, each announced before it takes effect."""
        added = sid_of(_STAGE1_SLOTS["membership"][0])
        removed = sid_of(_STAGE1_SLOTS["membership"][1])
        add_day = self.at(0.35)
        last_day = self.at(0.70)
        remove_day = sessions.next_session(last_day)
        self.membership = [
            row for row in self.membership if not (row["index_id"] == EQUITY_INDEX and row["sid"] in {added, removed})
        ]
        self.add_membership(
            EQUITY_INDEX,
            added,
            add_day.isoformat(),
            None,
            announce=self.shift(add_day, ANNOUNCE_LEAD_SESSIONS).isoformat(),
        )
        # No announce_date on the removed name's row. ``announce_date`` on a
        # membership row is when *that row* became knowable, and the row opens
        # at the start of the sample; carrying the removal announcement here
        # would hide the whole interval until the removal was announced. The
        # removal's announcement timing lives on the membership_remove event.
        # Expressing "known open, later announced closed" needs two revisions
        # of one row, which is the stage 3 revision semantics.
        self.add_membership(EQUITY_INDEX, removed, self.grid[0].isoformat(), last_day.isoformat())
        self.add_event(
            added, "membership_add", add_day, {"index_id": EQUITY_INDEX}, self.shift(add_day, ANNOUNCE_LEAD_SESSIONS)
        )
        self.add_event(
            removed,
            "membership_remove",
            remove_day,
            {"index_id": EQUITY_INDEX},
            self.shift(remove_day, ANNOUNCE_LEAD_SESSIONS),
        )
        self.record_case(
            "membership",
            sids={"added": added, "removed": removed},
            dates={
                "add_effective": add_day.isoformat(),
                "add_announce": self.shift(add_day, ANNOUNCE_LEAD_SESSIONS).isoformat(),
                "remove_effective": remove_day.isoformat(),
                "last_member_session": last_day.isoformat(),
            },
            fill_bar=add_day,
            expects=(
                "membership() returns the added sid on and after add_effective and not on the announce date; the "
                "removed sid leaves the index on remove_effective and its holding is never sold by the data layer"
            ),
        )

    def _scenario_split(self) -> None:
        """A 2-for-1 on a raw series (the print halves) and on an adjusted one (it does not)."""
        raw_sid = sid_of(_STAGE1_SLOTS["split"][0])
        adjusted_sid = sid_of(_STAGE1_SLOTS["split"][1])
        effective = self.at(0.55)
        announce = self.shift(effective, ANNOUNCE_LEAD_SESSIONS)
        frame = self.bars[raw_sid]
        tail = frame["ts"].dt.date >= effective
        for column in ("open", "high", "low", "close"):
            frame.loc[tail, column] = (frame.loc[tail, column] / SPLIT_FACTOR).round(4)
        frame.loc[tail, "volume"] = (frame.loc[tail, "volume"] * SPLIT_FACTOR).round()
        for sid in (raw_sid, adjusted_sid):
            self.add_event(sid, "split", effective, {"factor": SPLIT_FACTOR}, announce)
            self.add_adjustment(sid, effective, "split", SPLIT_FACTOR, effective)
        self.record_case(
            "split",
            sids={"raw": raw_sid, "adjusted": adjusted_sid},
            dates={"effective": effective.isoformat(), "announce": announce.isoformat()},
            values={"factor": f"{SPLIT_FACTOR}"},
            fill_bar=effective,
            expects=(
                "the raw series halves its print on the effective date and applied_share_factor doubles the shares; "
                "the split_adjusted_at_ingest series prints continuously and its share count does not move (D3)"
            ),
        )

    def _scenario_distribution(self) -> None:
        """A cash distribution whose ex-date and pay date straddle a decision."""
        sid = sid_of(_STAGE1_SLOTS["distribution"][0])
        ex_date = self.at(0.60)
        pay_date = self.grid[min(self.n - 1, self.index_of(ex_date) + 2)]
        close = float(self.bars[sid].loc[self.bars[sid]["ts"].dt.date == ex_date, "close"].iloc[0])
        amount = round(close * 0.004, 4)
        payload = {
            "amount_per_share": amount,
            "currency": "USD",
            "pay_date": pay_date.isoformat(),
            "distribution_kind": "dividend",
        }
        payload["provenance_sha256"] = _payload_sha({"sid": sid, "ex_date": ex_date.isoformat(), **payload})
        announce = self.shift(ex_date, ANNOUNCE_LEAD_SESSIONS)
        self.add_event(sid, "cash_distribution", ex_date, payload, announce)
        self.record_case(
            "distribution",
            sids={"payer": sid},
            dates={
                "ex_date": ex_date.isoformat(),
                "pay_date": pay_date.isoformat(),
                "announce": announce.isoformat(),
            },
            values={"amount_per_share": f"{amount}"},
            fill_bar=ex_date,
            expects=(
                "entitlement is the signed share count at the close of decision_bar, the bar before the ex-date, so "
                "a buyer filled at the ex-date open is not entitled; total_return reinvests on the ex-date while "
                "the as-traded close does not"
            ),
        )

    def _scenario_halt(self) -> None:
        """A three-session halt: the bars are absent and a halt event says why."""
        sid = sid_of(_STAGE1_SLOTS["halt"][0])
        start = self.at(0.65)
        first = self.index_of(start)
        halted = self.grid[first : first + HALT_SESSIONS]
        frame = self.bars[sid]
        self.bars[sid] = frame.loc[~frame["ts"].dt.date.isin(set(halted))].reset_index(drop=True)
        resumed = self.grid[min(self.n - 1, first + HALT_SESSIONS)]
        self.add_event(
            sid,
            "halt",
            halted[0],
            {"start": halted[0].isoformat(), "end": resumed.isoformat(), "reason": "synthetic_volatility_halt"},
        )
        self.note_gap(sid, halted)
        self.record_case(
            "halt",
            sids={"halted": sid},
            dates={
                "halt_start": halted[0].isoformat(),
                "halt_end": halted[-1].isoformat(),
                "resumed": resumed.isoformat(),
                "last_priced": self.shift(halted[0], 1).isoformat(),
            },
            values={"halted_sessions": f"{len(halted)}"},
            fill_bar=halted[0],
            expects=(
                "marks() carries the holding at the close of last_priced with valuation_status 'stale' through the "
                "halt, never at zero and never sold; the gap stays below gap_flag_sessions so no coverage flag fires"
            ),
        )

    def _scenario_costs(self) -> None:
        """The dollar-volume levels are set when the instruments are minted; this registers the case."""
        slots = _STAGE1_SLOTS["costs"]
        sids = {
            "widest_volume": sid_of(slots[0]),
            "wide_volume": sid_of(slots[1]),
            "mid_volume": sid_of(slots[2]),
            "thin_volume": sid_of(slots[3]),
        }
        self.record_case(
            "costs",
            sids=sids,
            dates={},
            values={"dollar_volume_levels": ",".join(f"{level:.0f}" for level in _COST_DOLLAR_VOLUMES)},
            fill_bar=self.at(0.50),
            expects=(
                "the four sids land one in each step of the reference spread schedule when dollar volume is computed "
                "from the view's own close and volume columns, and thin_volume binds a 5 percent participation cap"
            ),
        )

    def _scenario_rounding(self) -> None:
        """A dated per-sid quantity increment: whole shares until mid-sample, then blocks of five."""
        sid = sid_of(_STAGE1_SLOTS["rounding"][0])
        change_day = self.at(0.50)
        self.add_capability(sid, change_day, quantity_increment=5.0, min_notional=5.0)
        self.record_case(
            "rounding",
            sids={"rounded": sid},
            dates={"increment_change": change_day.isoformat()},
            values={"increment_before": "1.0", "increment_after": "5.0"},
            fill_bar=change_day,
            expects=(
                "capabilities() returns quantity_increment 1.0 for a decision before increment_change and 5.0 on and "
                "after it; the increment is dated, never assumed current"
            ),
        )

    def _scenario_spinoff(self) -> None:
        """A tracked child that begins trading on the effective date, with cash in lieu of fractions."""
        parent = _extra_sid(self.spec, "spinoff", 0)
        child = _extra_sid(self.spec, "spinoff", 1)
        effective = self.at(0.55)
        announce = self.shift(effective, ANNOUNCE_LEAD_SESSIONS)
        self.add_event(
            parent,
            "spinoff",
            effective,
            {"child_sid": child, "ratio": SPINOFF_RATIO, "cash_in_lieu": True},
            announce,
        )
        # The child does not exist before the distribution. Its listing, its
        # security row and its bars all open on the effective date, so a view
        # taken before it refuses to resolve the child rather than pricing an
        # instrument that had not been created.
        self.listings = [row for row in self.listings if row["sid"] != child]
        self.securities = [row for row in self.securities if row["sid"] != child]
        self.add_listing(child, self.by_sid[child].ticker, effective.isoformat(), None)
        self.add_security(self.by_sid[child], valid_on=effective)
        frame = self.bars[child]
        self.bars[child] = frame.loc[frame["ts"].dt.date >= effective].reset_index(drop=True)
        self.record_case(
            "spinoff",
            sids={"parent": parent, "child": child},
            dates={
                "effective": effective.isoformat(),
                "announce": announce.isoformat(),
            },
            values={"ratio": f"{SPINOFF_RATIO}"},
            fill_bar=effective,
            expects=(
                "the child's first bar, listing and security row all open on the effective date; the parent's close "
                "on decision_bar is unchanged by the distribution and the child is a second holding, not a merge"
            ),
        )

    def _scenario_ticker_reuse(self) -> None:
        """One ticker, two instruments, non-overlapping listing intervals."""
        first = _extra_sid(self.spec, "ticker_reuse", 0)
        second = _extra_sid(self.spec, "ticker_reuse", 1)
        cease = self.at(0.45)
        reopen = self.grid[min(self.n - 1, self.index_of(cease) + 1)]
        notice = self.shift(cease, ANNOUNCE_LEAD_SESSIONS)
        self.listings = [row for row in self.listings if row["sid"] not in {first, second}]
        self.securities = [row for row in self.securities if row["sid"] != second]
        self.add_listing(first, "RUSE", self.grid[0].isoformat(), cease.isoformat())
        self.add_listing(second, "RUSE", reopen.isoformat(), None)
        # The second instrument is a different company that took the ticker
        # over; it is not a continuation of the first, so its security row
        # opens when it lists.
        self.add_security(self.by_sid[second], valid_on=reopen)
        self.add_event(
            first,
            "delisting_notice",
            notice,
            {"reason": "synthetic_acquisition", "expected_last_session": cease.isoformat()},
            notice,
        )
        self.add_event(first, "trading_cessation", cease, {"last_session": cease.isoformat()})
        frame = self.bars[first]
        self.bars[first] = frame.loc[frame["ts"].dt.date <= cease].reset_index(drop=True)
        frame = self.bars[second]
        self.bars[second] = frame.loc[frame["ts"].dt.date >= reopen].reset_index(drop=True)
        self.record_case(
            "ticker_reuse",
            sids={"first": first, "second": second},
            dates={
                "cease": cease.isoformat(),
                "reopen": reopen.isoformat(),
                "notice": notice.isoformat(),
            },
            values={"ticker": "RUSE"},
            fill_bar=reopen,
            expects=(
                "resolve('RUSE', on) is single-valued on every date and answers 'first' up to cease and 'second' "
                "from reopen; the two sids hold separate bar files and separate marks, so a holding in one is never "
                "merged into the other when the ticker changes hands"
            ),
        )

    def _scenario_revision(self) -> None:
        """A late correction and then a null-valued full revision of the same bar."""
        sid = _extra_sid(self.spec, "revision", 0)
        frame = self.bars[sid]
        target = self.at(0.40)
        corrected_day = self.ahead(target, CORRECTION_LAG_SESSIONS)
        withdrawn_day = self.ahead(target, WITHDRAWAL_LAG_SESSIONS)
        if not target < corrected_day < withdrawn_day:
            raise SyntheticError(
                "the revision scenario needs three distinct sessions for the original, the correction and the "
                f"withdrawal; the grid gave {target}, {corrected_day}, {withdrawn_day}"
            )
        row = frame.loc[frame["ts"].dt.date == target].iloc[0]
        original_close = float(row["close"])
        corrected = row.to_dict()
        corrected["close"] = round(original_close * CORRECTION_MULTIPLE, 4)
        withdrawn = {
            key: (float("nan") if key in {"open", "high", "low", "close", "volume"} else value)
            for key, value in row.to_dict().items()
        }
        self.bar_revisions[sid] = [
            pd.DataFrame([corrected]).assign(
                revision=2,
                available_at=_instant(corrected_day, hour=CORRECTION_HOUR),
                ingested_at=_instant(corrected_day, hour=INGESTION_HOUR),
            ),
            pd.DataFrame([withdrawn]).assign(
                revision=3,
                available_at=_instant(withdrawn_day, hour=CORRECTION_HOUR),
                ingested_at=_instant(withdrawn_day, hour=INGESTION_HOUR),
            ),
        ]
        self.record_case(
            "revision",
            sids={"revised": sid},
            dates={
                "revised_bar": target.isoformat(),
                "corrected_from": corrected_day.isoformat(),
                "withdrawn_from": withdrawn_day.isoformat(),
            },
            values={
                "original_close": f"{round(original_close, 4)}",
                "corrected_close": f"{corrected['close']}",
            },
            fill_bar=target,
            expects=(
                "a decision before corrected_from sees original_close; one on or after it sees corrected_close; one "
                "on or after withdrawn_from sees null, not corrected_close, and marks() carries the holding stale "
                "at the last surviving priced bar rather than at zero"
            ),
        )

    def _scenario_rebase(self) -> None:
        """A 4-for-1 vendor back-rewrite: a full-series revision plus the matching adjustments row."""
        sid = _extra_sid(self.spec, "rebase", 0)
        effective = self.at(0.50)
        frame = self.bars[sid].copy()
        # No rounding on the restated prices. Rounding a 4-decimal print to 4
        # decimals after dividing by the factor leaves a per-row ratio spread
        # of order 1e-5 on a ten-dollar stock, which is above REWRITE_RTOL
        # (1e-6): rewrite_ratio would then classify a genuine rebase as a
        # source correction and no share rebasing would be demanded.
        for column in ("open", "high", "low", "close"):
            frame[column] = frame[column] / REBASE_FACTOR
        frame["volume"] = (frame["volume"] * REBASE_FACTOR).round()
        # The rewrite restates history at `effective` and the vendor prints the
        # new basis natively from then on, so a restated row for a session
        # *after* the rewrite is knowable no earlier than that session's own
        # close. Stamping the whole revision at `effective` would publish a
        # future bar, which is exactly the leak the availability rule exists to
        # stop.
        landing = _instant_at(effective, hour=CORRECTION_HOUR)
        stored = _instant_at(effective, hour=INGESTION_HOUR)
        days = [stamp.date() for stamp in frame["ts"]]
        frame = frame.assign(
            revision=2,
            available_at=[_later(landing, _session_close_at(day)) for day in days],
            ingested_at=[_later(stored, _instant_at(day, hour=INGESTION_HOUR)) for day in days],
        )
        self.bar_revisions[sid] = [frame]
        self.add_adjustment(sid, effective, "rebase", REBASE_FACTOR, self.grid[0])
        self.record_case(
            "rebase",
            sids={"rebased": sid},
            dates={"effective": effective.isoformat(), "applies_from": self.grid[0].isoformat()},
            values={"factor": f"{REBASE_FACTOR}"},
            fill_bar=effective,
            expects=(
                "validate_bars_revision classifies the revision as a full-series rebase at factor 4.0 and finds the "
                "matching adjustments row; applied_share_factor multiplies held shares by 4.0 so equity is "
                "continuous across the rewrite, and the same revision with the adjustments row removed is refused"
            ),
        )

    def _scenario_delisting(self) -> None:
        """One delisting that pays a known terminal value and one whose terminal value is unknown."""
        paid = _extra_sid(self.spec, "delisting", 0)
        unknown = _extra_sid(self.spec, "delisting", 1)
        last = self.at(0.60)
        notice = self.shift(last, ANNOUNCE_LEAD_SESSIONS)
        pay_day = self.ahead(last, 10)
        close = float(self.bars[paid].loc[self.bars[paid]["ts"].dt.date == last, "close"].iloc[0])
        payout = round(close * TERMINAL_PAYOUT_MULTIPLE, 4)
        for sid in (paid, unknown):
            self.add_event(
                sid,
                "delisting_notice",
                notice,
                {"reason": "synthetic_going_private", "expected_last_session": last.isoformat()},
                notice,
            )
            self.add_event(sid, "trading_cessation", last, {"last_session": last.isoformat()})
            frame = self.bars[sid]
            self.bars[sid] = frame.loc[frame["ts"].dt.date <= last].reset_index(drop=True)
        # The payout lands on the pay date, not on the last session: a payout
        # dated at the cessation would be indistinguishable from the last close
        # being carried forward, and "no earlier close is overwritten by an
        # eventual payout" would be untestable on this fixture (§3.2).
        self.add_event(
            paid,
            "terminal_payout",
            pay_day,
            {"amount_per_share": payout, "currency": "USD", "pay_date": pay_day.isoformat()},
        )
        self.add_event(unknown, "terminal_payout", last, {"status": "unknown"})
        self.record_case(
            "delisting",
            sids={"paid": paid, "unknown": unknown},
            dates={
                "notice": notice.isoformat(),
                "last_session": last.isoformat(),
                "pay_date": pay_day.isoformat(),
            },
            values={"last_close": f"{round(close, 4)}", "amount_per_share": f"{payout}"},
            fill_bar=self.ahead(last, 1),
            expects=(
                "both sids stop printing after last_session and marks() carries each stale at its own last close; "
                "the 'unknown' claim stays unresolved so a strict claim on it is recorded incomplete, while the "
                "'paid' claim resolves to amount_per_share on pay_date and the last close is never overwritten"
            ),
        )

    def _scenario_recovery(self) -> None:
        """A data gap long enough to trip the coverage flag, on a name carrying a broker restriction."""
        sid = _extra_sid(self.spec, "recovery", 0)
        start_index = self.index_of(self.at(0.55))
        gap = self.grid[start_index : start_index + RECOVERY_GAP_SESSIONS]
        frame = self.bars[sid]
        self.bars[sid] = frame.loc[~frame["ts"].dt.date.isin(set(gap))].reset_index(drop=True)
        self.add_capability(
            sid,
            gap[-1],
            restrictions=json.dumps(["no_new_exposure"]),
            shortable=False,
            easy_to_borrow=False,
        )
        self.note_gap(sid, gap)
        self.record_case(
            "recovery",
            sids={"gapped": sid},
            dates={
                "gap_start": gap[0].isoformat(),
                "gap_end": gap[-1].isoformat(),
                "last_priced": self.shift(gap[0], 1).isoformat(),
            },
            values={"missing_sessions": f"{len(gap)}"},
            fill_bar=gap[0],
            expects=(
                "the gap exceeds gap_flag_sessions so a coverage flag naming the sid, the segment and the length is "
                "written; the holding is carried stale for the whole gap and the dated capability adds "
                "no_new_exposure from gap_end"
            ),
        )

    # -- writing --------------------------------------------------------

    def _bar_frame(self, sid: str) -> pd.DataFrame:
        inst = self.by_sid[sid]
        frame = self.bars[sid].copy()
        frame["price_basis"] = inst.price_basis
        frame["available_at"] = None
        # A nightly-maintained series stores each bar on its own evening, so a
        # replay view answers inside the sample instead of refusing every date
        # before one shared download instant.
        frame["ingested_at"] = [_instant(stamp.date(), hour=INGESTION_HOUR) for stamp in frame["ts"]]
        frame["source"] = inst.source
        frame["revision"] = 1
        frame["synthetic"] = True
        columns = [
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "price_basis",
            "available_at",
            "ingested_at",
            "source",
            "revision",
            "synthetic",
        ]
        frame = frame[columns]
        extra = self.bar_revisions.get(sid)
        if extra:
            aligned = [part.reindex(columns=columns) for part in extra]
            for part in aligned:
                part["price_basis"] = inst.price_basis
                part["source"] = inst.source
                part["synthetic"] = True
                if part["ingested_at"].isna().any() or part["available_at"].isna().any():
                    raise SyntheticError(
                        f"{sid}: a bar revision declares its own available_at and ingested_at; without them the "
                        "correction would be visible from the moment the original was, which is the whole point of "
                        "the scenario"
                    )
            frame = pd.concat([frame, *aligned], ignore_index=True)
        return frame.sort_values(["ts", "revision"]).reset_index(drop=True)

    def write(self) -> Manifest:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "bars").mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for sid in self.by_sid:
            path = self.root / "bars" / f"{sid}_{BAR_INTERVAL}.parquet"
            self._bar_frame(sid).to_parquet(path, index=False)
            written.append(path)
        tables = {
            "securities": self.securities,
            "listings": self.listings,
            "membership": self.membership,
            "events": self.events,
            "adjustments": self.adjustments,
            "capabilities": self.capabilities,
        }
        for name, rows in tables.items():
            if not rows:
                continue
            path = self.root / f"{name}.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            written.append(path)
        manifest = self._manifest(written)
        manifest.write(self.root / "manifest.json")
        return manifest

    def _first_ingested(self) -> dict[str, str]:
        """The earliest ingestion instant per table, which is the floor a replay view refuses below.

        ``replay_recorded`` refuses a decision earlier than the first ingestion
        of a table it reads (§3.2), so a reader of this fixture needs the floor
        without opening every file to find it. ``bars`` is the minimum over all
        bar files.
        """
        tables: dict[str, list[str]] = {
            "securities": [row["ingested_at"] for row in self.securities],
            "listings": [row["ingested_at"] for row in self.listings],
            "membership": [row["ingested_at"] for row in self.membership],
            "events": [row["ingested_at"] for row in self.events],
            "adjustments": [row["ingested_at"] for row in self.adjustments],
            "capabilities": [row["ingested_at"] for row in self.capabilities],
        }
        out = {
            name: min(pd.Timestamp(value) for value in values).isoformat() for name, values in tables.items() if values
        }
        stamps = [min(stamp.date() for stamp in frame["ts"]) for frame in self.bars.values() if not frame.empty]
        if stamps:
            out["bars"] = _instant(min(stamps), hour=INGESTION_HOUR)
        return dict(sorted(out.items()))

    def _manifest(self, written: list[Path]) -> Manifest:
        tables = sorted({path.name.split(".")[0] if path.parent == self.root else "bars" for path in written})
        priority_tables = [t for t in tables if t != "bars"] + ["bars"]
        sources = {
            SOURCE_ADJUSTED: {
                "lag_bars": 0,
                "revision_policy": "conflict_raise",
                "price_basis": "split_adjusted_at_ingest",
                "url": None,
                "sha256": None,
                "redistributable": True,
            },
            SOURCE_RAW: {
                "lag_bars": 0,
                "revision_policy": "conflict_raise",
                "price_basis": "raw",
                "url": None,
                "sha256": None,
                "redistributable": True,
            },
        }
        universe = {
            EQUITY_INDEX: sorted({row["sid"] for row in self.membership if row["index_id"] == EQUITY_INDEX}),
            ETF_INDEX: sorted({row["sid"] for row in self.membership if row["index_id"] == ETF_INDEX}),
        }
        written_not_applied = sorted(s for s in self.spec.scenarios if SCENARIOS[s].stage > 1)
        in_universe_unapplied = sorted(s for s in self.spec.scenarios if s in STAGE1_IN_UNIVERSE_UNAPPLIED)
        return Manifest(
            provenance="synthetic",
            as_of=str(pd.Timestamp(self.grid[-1], tz="America/New_York").isoformat()),
            files=tuple(hash_tree(self.root, written)),
            sources=sources,
            source_priority={table: [SOURCE_ADJUSTED, SOURCE_RAW] for table in priority_tables},
            universe=sha256_bytes(canonical_json(universe).encode("utf-8")),
            vintage=self.spec.vintage,
            coverage={
                "index_membership": {index: len(sids) for index, sids in universe.items()},
                "sessions": [self.grid[0].isoformat(), self.grid[-1].isoformat()],
                "n_sessions": self.n,
                "gap_flags": self.coverage_flags,
                "bar_gaps": self.bar_gaps,
                "sids_outside_every_index": sorted(inst.sid for inst in self.instruments if inst.in_index is None),
                "scenarios": {name: case.to_dict() for name, case in sorted(self.cases.items())},
                "first_ingested_at": self._first_ingested(),
            },
            transformations=[
                "synthetic_v1",
                f"seed={self.spec.seed}",
                f"scenarios={','.join(self.spec.scenarios)}",
                f"n_equities={self.spec.n_equities}",
                f"n_etfs={self.spec.n_etfs}",
            ],
            assumptions={
                "gap_flag_sessions": GAP_FLAG_SESSIONS,
                "availability": (
                    "available_at is empty on every row except a declared bar revision, which carries the instant "
                    "its source published it; availability otherwise derives from the session table"
                ),
                "availability_non_session_rolls_forward": True,
                "ingestion_model": (
                    "every row was stored at 20:00 America/New_York on its own record date -- the announce date "
                    "where one exists, otherwise the effective date, the interval's opening date, or the bar's own "
                    "session. A revision declares its own. This is a workspace maintained nightly, so "
                    "replay_recorded is answerable inside the sample; coverage.first_ingested_at gives the floor "
                    "each table refuses below."
                ),
                "session_table_version": sessions.TABLE_VERSION,
                "scenarios_written": list(self.spec.scenarios),
                "scenario_applied_at_stage": {s: APPLIED_AT_STAGE[s] for s in self.spec.scenarios},
                "scenarios_written_not_applied": written_not_applied,
                "scenarios_written_not_applied_note": (
                    "Rows for these scenarios are written in the final schema on sids that belong to no index, so "
                    "no evaluation over an index universe selects them or applies them to accounting. The data "
                    "layer reads them all -- revision selection, the adjustments table, terminal claims and the "
                    "two visibility modes (tests/test_synthetic_scenarios.py) -- and "
                    "tests/test_adversarial_sessions.py drives each one through the canonical simulator."
                ),
                "scenario_cases": (
                    "coverage.scenarios carries one ScenarioCase per scenario id: the sids by role, the dates that "
                    "decide it, and the three-bar window (prior_bar, decision_bar, fill_bar) a session runs over. "
                    "That is the address tests/test_adversarial_sessions.py drives each scenario by."
                ),
                "membership_removal_announcement": (
                    "A removal is a revision of one membership row, so a single-revision fixture cannot make the "
                    "closed interval invisible before it was announced. The membership row carries the final "
                    "interval and the membership_remove event carries the announcement date."
                ),
                "scenarios_unapplied_in_universe": in_universe_unapplied,
                "stage1_unapplied_in_universe": (
                    "The split and halt scenarios stay in the equity index. The reproduction engine "
                    "(historical_constant_weight) applies no share change for a split and forward-fills a halted "
                    "session, so an evaluation on it reads the raw split as a price return and cannot see the "
                    "halt. The canonical simulator handles both: a split moves no equity and a halted holding is "
                    "carried stale (tests/test_adversarial_sessions.py)."
                ),
            },
            fit_cutoff=None,
            feature_code_sha=None,
            feature_params=None,
        )


def write_synthetic_dataset(
    data_root: Path,
    *,
    seed: int,
    scenarios: str | Sequence[str] = "all",
    n_equities: int = 100,
    n_etfs: int = 10,
    start: dt.date | None = None,
    end: dt.date | None = None,
    vintage: str = "synthetic_v1",
) -> Manifest:
    """Write a complete synthetic dataset under ``data_root`` and return its manifest.

    ``data_root`` is ``workspace.data``. The bar files land under
    ``bars/``, the tables beside them, and ``manifest.json`` names every one
    with its hash. Regenerating with the same arguments rewrites the same
    bytes and therefore the same ``manifest.id``.
    """
    spec = SyntheticSpec(
        seed=int(seed),
        scenarios=resolve_scenarios(scenarios),
        n_equities=n_equities,
        n_etfs=n_etfs,
        start=start,
        end=end,
        vintage=vintage,
    )
    builder = _Builder(Path(data_root), spec)
    builder.build()
    return builder.write()


__all__ = [
    "ALL_SCENARIOS",
    "BAR_INTERVAL",
    "CORRECTION_MULTIPLE",
    "EQUITY_INDEX",
    "ETF_INDEX",
    "GAP_FLAG_SESSIONS",
    "INGESTION_HOUR",
    "Manifest",
    "ManifestError",
    "REBASE_FACTOR",
    "SCENARIOS",
    "SPINOFF_RATIO",
    "SPLIT_FACTOR",
    "STAGE1_SCENARIOS",
    "SOURCE_ADJUSTED",
    "SOURCE_RAW",
    "Scenario",
    "ScenarioCase",
    "SyntheticError",
    "SyntheticSpec",
    "TERMINAL_PAYOUT_MULTIPLE",
    "resolve_scenarios",
    "scenario_case",
    "scenario_catalog",
    "sid_of",
    "write_synthetic_dataset",
]
