"""The account's margin position, read from the venue, and its record ``{run-dir}/margin.jsonl``.

Read-only telemetry. :meth:`prism.live.alpaca.AlpacaBroker.account_margin`
returns an :class:`AccountMargin` from one ``GET /v2/account``; the margin
report (``python -m prism.scripts.margin_report``) prints it and, with
``--record``, appends one row to ``{run-dir}/margin.jsonl``; ``prism-doctor``
warns when the maintenance cushion is thin. Nothing on the order path reads
the account's margin: the loop decides, sizes and submits exactly as it did
before this module existed, and acting on a thin cushion (selling the T-bill
holding first) is not implemented.

**The cushion** is ``(equity - maintenance margin) / equity``: the share of
equity that can be lost before the account reaches its maintenance
requirement, at the requirement the venue reports. A margin call comes when it
reaches zero. :data:`CUSHION_WARN_FLOOR` (25%) is the doctor's warning line; it
is a monitoring choice, not a venue rule. A long/short book at gross 1 under
FINRA's minimum requirements (25% of the long side, 30% of the short side)
has a cushion near 72%; at gross 2 near 45%. A venue's house requirement can
be higher, which is why the venue's own number is read rather than computed.

**Sign convention.** Alpaca defines equity as ``cash + long_market_value +
short_market_value``, so a short side is reported as a negative market value.
:class:`AccountMargin` stores the short side's magnitude, so
``short_market_value >= 0`` whatever sign the payload carries.

**The record.** One row per read, keyed by the read's UTC timestamp
(``read_at``) and appended only when it is later than the last recorded one,
plus ``as_of``, the New York calendar date of the read. A row carries the
venue's numbers, the derived ratios, and the floor in force, so a later reader
needs no code to interpret it.

Production-import-path safe (N8): standard library and :mod:`prism.live.ledgers`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from prism.live import ledgers

__all__ = [
    "CUSHION_WARN_FLOOR",
    "MARGIN_LEDGER_FILENAME",
    "MARGIN_ROW_SCHEMA",
    "AccountMargin",
    "MarginError",
    "append_margin_row",
    "margin_row",
    "read_margin_rows",
    "row_cushion",
]

#: The record's file name inside the run directory.
MARGIN_LEDGER_FILENAME = "margin.jsonl"

#: The row schema version.
MARGIN_ROW_SCHEMA = 1

#: The maintenance cushion below which ``prism-doctor`` warns (a monitoring
#: line chosen by this program, not a venue or regulatory rule).
CUSHION_WARN_FLOOR = 0.25

#: The calendar ``as_of`` is expressed in: one date per New York session.
_AS_OF_TZ = ZoneInfo("America/New_York")

#: The fields every row carries from the venue, in the order the report prints them.
VENUE_FIELDS: tuple[str, ...] = (
    "equity",
    "cash",
    "long_market_value",
    "short_market_value",
    "initial_margin",
    "maintenance_margin",
    "regt_buying_power",
    "buying_power",
    "multiplier",
    "last_equity",
    "last_maintenance_margin",
    "non_marginable_buying_power",
    "sma",
    "shorting_enabled",
    "status",
    "balance_asof",
)


class MarginError(ValueError):
    """A margin read or row is malformed (N7)."""


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MarginError(f"account margin field {name} must be a finite number, got {value!r}")
    if minimum is not None and float(value) < minimum:
        raise MarginError(f"account margin field {name} must be >= {minimum}, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class AccountMargin:
    """The account's equity, its two sides, and the venue's margin figures, in dollars.

    ``initial_margin`` is the Reg T initial requirement and ``maintenance_margin``
    the maintenance requirement, both as the venue computes them now.
    ``regt_buying_power`` is the excess of equity over the initial requirement
    times the multiplier; ``buying_power`` is the venue's headline figure, which
    for a 4x (pattern day trader) account is intraday buying power. The
    ``last_*`` fields are the venue's previous-session snapshot, dated
    ``balance_asof``. Optional fields are ``None`` when the venue omits them.
    """

    equity: float
    cash: float
    long_market_value: float
    short_market_value: float
    initial_margin: float
    maintenance_margin: float
    regt_buying_power: float
    buying_power: float
    multiplier: float
    last_equity: float | None = None
    last_maintenance_margin: float | None = None
    non_marginable_buying_power: float | None = None
    sma: float | None = None
    shorting_enabled: bool | None = None
    status: str | None = None
    balance_asof: str | None = None

    def __post_init__(self) -> None:
        for name in ("equity", "cash", "regt_buying_power", "buying_power"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        for name in ("long_market_value", "short_market_value", "initial_margin", "maintenance_margin"):
            object.__setattr__(self, name, _finite(name, getattr(self, name), minimum=0.0))
        multiplier = _finite("multiplier", self.multiplier)
        if multiplier <= 0.0:
            raise MarginError(f"account margin field multiplier must be > 0, got {self.multiplier!r}")
        object.__setattr__(self, "multiplier", multiplier)
        for name in ("last_equity", "last_maintenance_margin", "non_marginable_buying_power", "sma"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(name, value))

    # ------------------------------------------------------------- derived

    def _over_equity(self, value: float) -> float | None:
        return value / self.equity if self.equity > 0.0 else None

    @property
    def cushion(self) -> float | None:
        """``(equity - maintenance margin) / equity``; ``None`` when equity is not positive."""
        return self._over_equity(self.equity - self.maintenance_margin)

    @property
    def long_over_equity(self) -> float | None:
        return self._over_equity(self.long_market_value)

    @property
    def short_over_equity(self) -> float | None:
        return self._over_equity(self.short_market_value)

    @property
    def gross_over_equity(self) -> float | None:
        return self._over_equity(self.long_market_value + self.short_market_value)

    @property
    def initial_margin_excess(self) -> float:
        """Equity above the Reg T initial requirement: what new positions can draw on."""
        return self.equity - self.initial_margin

    @property
    def maintenance_margin_excess(self) -> float:
        """Equity above the maintenance requirement: what losses can draw on before a call."""
        return self.equity - self.maintenance_margin

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in VENUE_FIELDS}


def margin_row(margin: AccountMargin, *, read_at: datetime, mode: str, source: str) -> dict[str, Any]:
    """One ``margin.jsonl`` row: when and where the read was made, the venue's numbers, the ratios, the floor.

    ``read_at`` must be timezone-aware; it is stored in UTC to the second, so
    rows order lexically.
    """
    if read_at.tzinfo is None or read_at.utcoffset() is None:
        raise MarginError(f"read_at must be timezone-aware, got {read_at!r}")
    return {
        "schema": MARGIN_ROW_SCHEMA,
        "read_at": read_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "as_of": read_at.astimezone(_AS_OF_TZ).date().isoformat(),
        "mode": str(mode),
        "source": str(source),
        **margin.to_dict(),
        "cushion": margin.cushion,
        "long_over_equity": margin.long_over_equity,
        "short_over_equity": margin.short_over_equity,
        "gross_over_equity": margin.gross_over_equity,
        "initial_margin_excess": margin.initial_margin_excess,
        "maintenance_margin_excess": margin.maintenance_margin_excess,
        "cushion_floor": CUSHION_WARN_FLOOR,
    }


def append_margin_row(path: Path | str, row: Mapping[str, Any]) -> bool:
    """Append ``row`` iff its ``read_at`` is later than the last recorded one; True when written."""
    if "read_at" not in row:
        raise MarginError("a margin row needs its read_at timestamp")
    return ledgers.append_monotone(Path(path), "read_at", dict(row))


def read_margin_rows(path: Path | str) -> list[dict]:
    """Every parseable row of a margin record, oldest first (empty when absent)."""
    return ledgers.read_rows(Path(path))


def row_cushion(row: Mapping[str, Any]) -> float | None:
    """The cushion of a recorded row, recomputed from its equity and maintenance margin.

    Recomputed rather than read, so a row whose stored ratio disagrees with its
    own numbers cannot pass. ``None`` when equity is not positive; raises
    :class:`MarginError` when either number is missing or not finite.
    """
    try:
        equity = _finite("equity", row["equity"])
        maintenance = _finite("maintenance_margin", row["maintenance_margin"], minimum=0.0)
    except KeyError as exc:
        raise MarginError(f"margin row has no {exc.args[0]}") from None
    if equity <= 0.0:
        return None
    return (equity - maintenance) / equity
