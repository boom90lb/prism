"""The after-cost Sharpe hurdle, computed in one place.

SPEC.md §3 (owner decision 2026-09-25) states the hurdle a long/short book must
clear to beat Treasury bills. Idle own cash, the equity the long side leaves
unused, earns the account's cash yield *y*; short-sale proceeds are collateral
and earn nothing. With a long side *L* on equity *E*:

    return_hurdle_annual = rf - y * max(0, 1 - L/E)
    hurdle_sharpe_annual = return_hurdle_annual / vol_annual
    hurdle_sharpe_periodic = hurdle_sharpe_annual / sqrt(periods_per_year)

Before this module the same quantity was computed at three call sites, each
with a different volatility in the denominator: the research driver that
produced the recorded momentum and trend ledgers converts an annual return
hurdle at the book's *realized periodic* standard deviation
(:func:`prism.validation.metrics.after_cost_hurdle_periodic`); the
volatility-scaled momentum replication divides the bill rate by the
*target* volatility (``research/scripts/vol_momentum_replication.economics``);
the forward reader divides the amended return hurdle by the *realized
annualized* volatility (``research/scripts/forward_read``). Same formula,
three volatility bases, three numbers for one book. ``vol_basis`` therefore
travels with the result so a reader can tell which volatility a hurdle was
stated at, and every caller must say which one it used.

With ``cash_yield=0`` the function reproduces ``rf / vol`` exactly: ``y * idle``
is ``0.0`` and ``rf - 0.0`` is ``rf``. That is the convention every recorded
verdict was decided under, so recorded numbers reproduce bit for bit when the
caller passes the same volatility.

The function refuses a non-finite input or a non-positive volatility loudly
(SPEC.md N7). A hurdle of NaN or infinity that a comparison silently fails or
passes is the failure mode this module exists to remove.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

#: The volatility a caller may state as the denominator's basis. ``target`` is
#: the volatility a scaling rule aims at (12% for the Barroso–Santa-Clara
#: rule); ``realized_annualized`` is the book's realized periodic standard
#: deviation times the square root of the periods per year;
#: ``periodic_realized`` is that same standard deviation before annualizing,
#: which is what the pre-overhaul research driver used. A caller that passes a
#: periodic volatility must annualize it before calling; the label records
#: where the number came from, not its units.
VOL_BASES: tuple[str, ...] = ("target", "realized_annualized", "periodic_realized")

#: The keys of the optional ``hurdle`` block of a research claim packet
#: (:mod:`prism.validation.trials`). Every key is present when the block is;
#: the block itself is optional so packets written before it still validate.
PACKET_BLOCK_KEYS: tuple[str, ...] = (
    "rf_annual",
    "rf_source",
    "cash_yield",
    "cash_yield_source",
    "long_over_equity",
    "vol_basis",
    "vol_annual",
    "hurdle_sharpe_annual",
    "hurdle_sharpe_periodic",
)


@dataclass(frozen=True)
class HurdleInputs:
    """What the hurdle was computed from, echoed back with its sources.

    ``rf_annual``, ``cash_yield`` and ``vol_annual`` are annual fractions
    (0.0421 for 4.21%). ``long_over_equity`` is the long side's notional over
    equity, averaged over the sessions the hurdle describes. The three source
    strings are free text naming where each figure came from, for example
    ``"FRED:DTB3 mean 2026-09-24..2027-03-24"`` or
    ``"examples/reference/costs/retail_v2.toml cash_income_bps_annual"``.
    """

    rf_annual: float
    cash_yield: float
    long_over_equity: float
    vol_annual: float
    periods_per_year: int = 252
    rf_source: str = ""
    cash_yield_source: str = ""
    vol_basis: str = ""


@dataclass(frozen=True)
class HurdleResult:
    """The hurdle in return and Sharpe units, with the inputs that produced it."""

    inputs: HurdleInputs
    idle_fraction: float
    return_hurdle_annual: float
    sharpe_annual: float
    sharpe_periodic: float

    def to_dict(self) -> dict[str, Any]:
        """Flat JSON-ready mapping: every input, every source and every output."""
        out: dict[str, Any] = asdict(self.inputs)
        out.update(
            idle_fraction=self.idle_fraction,
            return_hurdle_annual=self.return_hurdle_annual,
            sharpe_annual=self.sharpe_annual,
            sharpe_periodic=self.sharpe_periodic,
        )
        return out

    def packet_block(self) -> dict[str, Any]:
        """The ``hurdle`` block of a research claim packet (:data:`PACKET_BLOCK_KEYS`)."""
        i = self.inputs
        return {
            "rf_annual": i.rf_annual,
            "rf_source": i.rf_source,
            "cash_yield": i.cash_yield,
            "cash_yield_source": i.cash_yield_source,
            "long_over_equity": i.long_over_equity,
            "vol_basis": i.vol_basis,
            "vol_annual": i.vol_annual,
            "hurdle_sharpe_annual": self.sharpe_annual,
            "hurdle_sharpe_periodic": self.sharpe_periodic,
        }


def _finite(name: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def tbill_hurdle(
    rf_annual: float,
    cash_yield: float,
    long_over_equity: float,
    vol_annual: float,
    *,
    periods_per_year: int = 252,
    rf_source: str = "",
    cash_yield_source: str = "",
    vol_basis: str = "",
) -> HurdleResult:
    """SPEC.md §3 as amended 2026-09-25: ``(rf - y * max(0, 1 - L/E)) / vol``.

    ``cash_yield=0`` is the convention every verdict recorded before the
    amendment was decided under and returns ``rf_annual / vol_annual`` exactly.
    ``vol_basis`` names which volatility ``vol_annual`` is (:data:`VOL_BASES`);
    it may be left empty when the caller has nothing to say, but a non-empty
    value must be one of the known labels so a typo cannot pass as a basis.

    Raises ``ValueError`` on a non-finite input, a non-positive volatility, a
    negative long side or fewer than one period a year.
    """
    rf = _finite("rf_annual", rf_annual)
    y = _finite("cash_yield", cash_yield)
    long_e = _finite("long_over_equity", long_over_equity)
    vol = _finite("vol_annual", vol_annual)
    if vol <= 0.0:
        raise ValueError(f"vol_annual must be positive, got {vol!r}; a hurdle at zero volatility is undefined")
    if long_e < 0.0:
        raise ValueError(f"long_over_equity must be >= 0, got {long_e!r}")
    if not isinstance(periods_per_year, int) or isinstance(periods_per_year, bool) or periods_per_year < 1:
        raise ValueError(f"periods_per_year must be an integer >= 1, got {periods_per_year!r}")
    if vol_basis and vol_basis not in VOL_BASES:
        raise ValueError(f"vol_basis must be one of {VOL_BASES} or empty, got {vol_basis!r}")

    idle = max(0.0, 1.0 - long_e)
    excess = rf - y * idle
    sharpe_annual = excess / vol
    sharpe_periodic = sharpe_annual / math.sqrt(periods_per_year)
    inputs = HurdleInputs(
        rf_annual=rf,
        cash_yield=y,
        long_over_equity=long_e,
        vol_annual=vol,
        periods_per_year=periods_per_year,
        rf_source=str(rf_source),
        cash_yield_source=str(cash_yield_source),
        vol_basis=str(vol_basis),
    )
    return HurdleResult(
        inputs=inputs,
        idle_fraction=idle,
        return_hurdle_annual=excess,
        sharpe_annual=sharpe_annual,
        sharpe_periodic=sharpe_periodic,
    )


__all__ = [
    "HurdleInputs",
    "HurdleResult",
    "PACKET_BLOCK_KEYS",
    "VOL_BASES",
    "tbill_hurdle",
]
