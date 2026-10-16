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

**Cash held in a T-bill ETF.** When idle own equity is invested in a T-bill
ETF instead of left as broker cash, the ETF is a long position whose total
return is part of the account's return, and only the cash left beside it is
idle. With a fraction *s* of equity in the ETF earning ``sleeve_yield``
*y_s*, and *L* the stock long side alone (the ETF is not part of it):

    return_hurdle_annual = rf - y_s * s - y * max(0, 1 - L/E - s)

*s* must lie in ``[0, max(0, 1 - L/E)]``: the ETF is bought with idle own
equity and cannot exceed it. Counting the ETF as a long position, so that only
the cash beside it is idle, is a reading of SPEC.md §3 the owner has not yet
confirmed; a verdict stated at it should be reported beside the broker-cash
and zero-yield hurdles. With the defaults ``sleeve_fraction=0.0`` and
``sleeve_yield=0.0`` every output, ``to_dict()`` and ``packet_block()``
included, is bit-identical to the function without the ETF, and a packet
records the ETF inputs in an optional ``sleeve`` sub-block only when one was
declared, so packets written before it validate unchanged.

:func:`cash_toll` states the same arithmetic per dollar of long exposure: the
net long-short spread a long dollar earns against the cash yield it displaces
plus the borrow on its matching short. With the cash yield equal to the bill
rate, a book clears :func:`tbill_hurdle` exactly when that margin is positive.
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

#: The keys of the optional ``sleeve`` sub-block of a packet's ``hurdle``
#: block, present only when the hurdle was stated with idle equity held in a
#: T-bill ETF (``sleeve_fraction`` or ``sleeve_yield`` non-zero, or a sleeve
#: yield source given). Every key is present when the sub-block is.
SLEEVE_BLOCK_KEYS: tuple[str, ...] = (
    "sleeve_fraction",
    "sleeve_yield",
    "sleeve_yield_source",
)

#: Slack allowed when checking that the ETF fits in idle equity. A sleeve
#: fraction computed as the mean of per-session ``1 - L_t/E_t`` can exceed
#: ``1 - mean(L_t/E_t)`` by a few units in the last place; refusing that would
#: refuse an ETF that exactly fills idle cash. Idle cash is then clamped at
#: zero, and the ETF's yield is counted on at most this much excess equity.
_SLEEVE_FIT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class HurdleInputs:
    """What the hurdle was computed from, echoed back with its sources.

    ``rf_annual``, ``cash_yield`` and ``vol_annual`` are annual fractions
    (0.0421 for 4.21%). ``long_over_equity`` is the long side's notional over
    equity, averaged over the sessions the hurdle describes. The three source
    strings are free text naming where each figure came from, for example
    ``"FRED:DTB3 mean 2026-09-24..2027-03-24"`` or
    ``"examples/reference/costs/retail_v2.toml cash_income_bps_annual"``.

    ``sleeve_fraction`` is the fraction of equity held in a T-bill ETF and
    ``sleeve_yield`` the annual yield it earns net of its fee, with
    ``sleeve_yield_source`` naming where that yield came from. All three are at
    their defaults when no ETF was declared (:attr:`has_sleeve`).
    """

    rf_annual: float
    cash_yield: float
    long_over_equity: float
    vol_annual: float
    periods_per_year: int = 252
    rf_source: str = ""
    cash_yield_source: str = ""
    vol_basis: str = ""
    sleeve_fraction: float = 0.0
    sleeve_yield: float = 0.0
    sleeve_yield_source: str = ""

    @property
    def has_sleeve(self) -> bool:
        """True when any T-bill ETF input differs from its default."""
        return self.sleeve_fraction != 0.0 or self.sleeve_yield != 0.0 or bool(self.sleeve_yield_source)


@dataclass(frozen=True)
class HurdleResult:
    """The hurdle in return and Sharpe units, with the inputs that produced it."""

    inputs: HurdleInputs
    idle_fraction: float
    return_hurdle_annual: float
    sharpe_annual: float
    sharpe_periodic: float

    def to_dict(self) -> dict[str, Any]:
        """Flat JSON-ready mapping: every input, every source and every output.

        The T-bill ETF inputs appear only when one was declared, so a hurdle
        stated without one has exactly the keys it had before they existed.
        """
        out: dict[str, Any] = asdict(self.inputs)
        if not self.inputs.has_sleeve:
            for key in SLEEVE_BLOCK_KEYS:
                del out[key]
        out.update(
            idle_fraction=self.idle_fraction,
            return_hurdle_annual=self.return_hurdle_annual,
            sharpe_annual=self.sharpe_annual,
            sharpe_periodic=self.sharpe_periodic,
        )
        return out

    def packet_block(self) -> dict[str, Any]:
        """The ``hurdle`` block of a research claim packet (:data:`PACKET_BLOCK_KEYS`).

        A ``sleeve`` sub-block (:data:`SLEEVE_BLOCK_KEYS`) follows the required
        keys only when a T-bill ETF was declared.
        """
        i = self.inputs
        block: dict[str, Any] = {
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
        if i.has_sleeve:
            block["sleeve"] = {
                "sleeve_fraction": i.sleeve_fraction,
                "sleeve_yield": i.sleeve_yield,
                "sleeve_yield_source": i.sleeve_yield_source,
            }
        return block


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
    sleeve_fraction: float = 0.0,
    sleeve_yield: float = 0.0,
    sleeve_yield_source: str = "",
) -> HurdleResult:
    """SPEC.md §3 as amended 2026-09-25: ``(rf - y * max(0, 1 - L/E)) / vol``.

    ``cash_yield=0`` is the convention every verdict recorded before the
    amendment was decided under and returns ``rf_annual / vol_annual`` exactly.
    ``vol_basis`` names which volatility ``vol_annual`` is (:data:`VOL_BASES`);
    it may be left empty when the caller has nothing to say, but a non-empty
    value must be one of the known labels so a typo cannot pass as a basis.

    ``sleeve_fraction`` (*s*) and ``sleeve_yield`` (*y_s*) state idle equity
    held in a T-bill ETF: the return hurdle becomes
    ``rf - y_s * s - y * max(0, 1 - L/E - s)``, so ``cash_yield`` then applies
    only to the cash left beside the ETF, and ``long_over_equity`` is the stock
    long side without the ETF. A caller that averages over sessions passes the
    mean of the per-session ETF fractions, not a fraction derived from the mean
    long side. With both at ``0.0`` (the defaults) the result is bit-identical
    to the function without them.

    Raises ``ValueError`` on a non-finite input, a non-positive volatility, a
    negative long side, fewer than one period a year, or a sleeve fraction
    below zero or above the idle equity ``max(0, 1 - L/E)`` it is bought with.
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
    sleeve = _finite("sleeve_fraction", sleeve_fraction)
    sleeve_y = _finite("sleeve_yield", sleeve_yield)
    if sleeve < 0.0:
        raise ValueError(f"sleeve_fraction must be >= 0, got {sleeve!r}")
    room = max(0.0, 1.0 - long_e)
    if sleeve > room + _SLEEVE_FIT_TOLERANCE:
        raise ValueError(
            f"sleeve_fraction {sleeve!r} exceeds the idle equity max(0, 1 - long_over_equity) = {room!r}; "
            "a T-bill ETF is bought with idle own equity and cannot be larger than it"
        )

    # With sleeve == 0.0: 1.0 - long_e - 0.0 is 1.0 - long_e, and rf - 0.0 is rf,
    # so the default path is bit-identical to the formula without the ETF.
    idle = max(0.0, 1.0 - long_e - sleeve)
    excess = rf - sleeve_y * sleeve - y * idle
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
        sleeve_fraction=sleeve,
        sleeve_yield=sleeve_y,
        sleeve_yield_source=str(sleeve_yield_source),
    )
    return HurdleResult(
        inputs=inputs,
        idle_fraction=idle,
        return_hurdle_annual=excess,
        sharpe_annual=sharpe_annual,
        sharpe_periodic=sharpe_periodic,
    )


@dataclass(frozen=True)
class CashToll:
    """A long-short book's net spread per dollar of long exposure against the cash it displaces.

    Inputs, echoed back: ``net_sharpe`` is the book's annualized Sharpe net of
    trading costs; ``vol_per_unit_gross`` its annualized volatility divided by
    its gross exposure over equity; ``long_over_equity`` its long side over
    equity; ``cash_yield`` the annual yield of the cash a long dollar displaces
    (the broker's cash rate, or the T-bill ETF's yield net of its fee when idle
    equity is held in one); ``borrow_annual`` the annual borrow fee on short
    market value that ``net_sharpe`` does not already include; and
    ``long_share_of_gross`` the long side's share of gross (0.5 for a
    dollar-neutral book).

    Outputs, all annual fractions:

    * ``spread_per_long_dollar``: the book's net return per dollar of long
      exposure, ``net_sharpe * vol_per_unit_gross / long_share_of_gross``.
    * ``hurdle_per_long_dollar``: what that dollar must beat, the cash yield
      on the part of the long side that displaces cash plus the borrow on the
      short side that comes with it.
    * ``margin_per_long_dollar``: the spread less that hurdle. Positive means
      adding long-short exposure raises the account's expected return over
      holding the same equity in cash.
    * ``excess_over_cash_annual``: ``margin_per_long_dollar * long_over_equity``,
      the account's expected return per unit of equity above holding all of it
      in cash at ``cash_yield``.
    """

    net_sharpe: float
    vol_per_unit_gross: float
    long_over_equity: float
    cash_yield: float
    borrow_annual: float
    long_share_of_gross: float
    displaced_fraction: float
    spread_per_long_dollar: float
    hurdle_per_long_dollar: float
    margin_per_long_dollar: float
    excess_over_cash_annual: float

    def to_dict(self) -> dict[str, Any]:
        """Flat JSON-ready mapping of every input and output."""
        return asdict(self)


def cash_toll(
    net_sharpe: float,
    vol_per_unit_gross: float,
    long_over_equity: float,
    cash_yield: float,
    borrow_annual: float,
    *,
    long_share_of_gross: float = 0.5,
) -> CashToll:
    """Net spread per dollar of long exposure, the yield it must beat, and the margin.

    Under SPEC.md §3 short-sale proceeds are collateral and earn nothing, so
    every dollar of long exposure is paid for with own equity that would
    otherwise earn ``cash_yield``. A book with gross ``G`` (over equity) and
    long side ``L/E = long_share_of_gross * G`` returns
    ``net_sharpe * vol_per_unit_gross * G`` a year, which is
    ``net_sharpe * vol_per_unit_gross / long_share_of_gross`` per long dollar.
    That long dollar gives up ``cash_yield`` and its matching short pays
    ``borrow_annual * (1 - long_share_of_gross) / long_share_of_gross``.
    Diversification lowers ``vol_per_unit_gross``, so at a fixed volatility
    target a diversified book needs more gross and pays more of this toll.

    Above ``L/E = 1`` the long side beyond equity displaces no cash: it is
    financed by a margin debit whose interest a cost stack charges as a cost
    (``financing_bps_annual``), so it belongs in ``net_sharpe``. The cash part
    of the hurdle is therefore charged on ``min(1, L/E) / (L/E)`` of each long
    dollar (``displaced_fraction``), which is 1 for every dollar-neutral book
    at or below 2x gross.

    The statistic restates :func:`tbill_hurdle` per long dollar. With
    ``cash_yield`` equal to the bill rate, ``margin_per_long_dollar > 0``
    exactly when the book's Sharpe net of borrow clears
    ``tbill_hurdle(rf, rf, L/E, vol)``. With a T-bill ETF, pass the ETF's
    net yield as ``cash_yield``; while the ETF holds ``1 - L/E - buffer`` of
    equity beside a cash buffer that earns nothing, the account's excess over
    bills is ``excess_over_cash_annual - (rf - cash_yield) - cash_yield * buffer``,
    the same number as the book's return less ``tbill_hurdle``'s return hurdle
    with that ETF fraction.

    ``borrow_annual`` is the fee not already inside ``net_sharpe``: the
    canonical engine charges borrow on short market value in its net returns,
    so a ``net_sharpe`` read from those returns takes ``borrow_annual=0.0``.

    Raises ``ValueError`` (SPEC.md N7) on a non-finite input, a non-positive
    ``vol_per_unit_gross`` or ``long_over_equity`` (a book with no long side
    has no spread per long dollar), a negative ``borrow_annual``, or a
    ``long_share_of_gross`` outside ``(0, 1]``.
    """
    sharpe = _finite("net_sharpe", net_sharpe)
    vol_g = _finite("vol_per_unit_gross", vol_per_unit_gross)
    long_e = _finite("long_over_equity", long_over_equity)
    y = _finite("cash_yield", cash_yield)
    borrow = _finite("borrow_annual", borrow_annual)
    share = _finite("long_share_of_gross", long_share_of_gross)
    if vol_g <= 0.0:
        raise ValueError(f"vol_per_unit_gross must be positive, got {vol_g!r}")
    if long_e <= 0.0:
        raise ValueError(f"long_over_equity must be positive, got {long_e!r}; a book with no long side has no "
                         "spread per long dollar")
    if borrow < 0.0:
        raise ValueError(f"borrow_annual must be >= 0, got {borrow!r}")
    if not 0.0 < share <= 1.0:
        raise ValueError(f"long_share_of_gross must lie in (0, 1], got {share!r}")

    displaced = min(1.0, long_e) / long_e
    spread = sharpe * vol_g / share
    hurdle = y * displaced + borrow * (1.0 - share) / share
    margin = spread - hurdle
    return CashToll(
        net_sharpe=sharpe,
        vol_per_unit_gross=vol_g,
        long_over_equity=long_e,
        cash_yield=y,
        borrow_annual=borrow,
        long_share_of_gross=share,
        displaced_fraction=displaced,
        spread_per_long_dollar=spread,
        hurdle_per_long_dollar=hurdle,
        margin_per_long_dollar=margin,
        excess_over_cash_annual=margin * long_e,
    )


__all__ = [
    "CashToll",
    "HurdleInputs",
    "HurdleResult",
    "PACKET_BLOCK_KEYS",
    "SLEEVE_BLOCK_KEYS",
    "VOL_BASES",
    "cash_toll",
    "tbill_hurdle",
]
