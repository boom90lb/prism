"""Decision assembly (engine specification §3.8): one bar in, order intents out.

One function turns a decision bar into orders, and both the evaluator and the
operating loop call it. What used to be spread across the live driver — score,
mask, construct, cap, band, participation (``live/daily.py``) and the sizing,
zero-crossing clamp and dust rules (``live/loop.py``) — lives here, so the
backtest and the live book cannot drift apart by editing one and not the other.
``live/daily.py`` keeps what is genuinely operational: settle, mark,
concordance, the halt rails, the regime read and the ledgers.

Two things this module deliberately does not own. It never *fits*: it receives
components already fitted, because fitting has a window and a cadence the
caller declares (§3.8). And it never fetches the unrankable mask: the mask is
an input, supplied by ``live/spinoff_mask.py`` in operation and derived from
spin-off events in evaluation, so the two paths mask by the same rule and the
per-bar mask file stays the durable record.

**Parity.** The construction and sizing here are the recorded live path,
unchanged: ``score`` on the last panel row, the book construct's own caps, the
*online* no-trade band against held weights, then the participation gate, then
whole-share sizing with the zero-crossing clamp and the dust rules. The
composition step in front of it (``prism.composition``, flow F1) reduces to
exactly that vector for a single-component book that is due at every refresh,
which is what the live books are.

**Stage 1 boundaries**, each named where it bites:

* ``DecisionOutput.order_intents`` carries ``prism.live.broker.Order`` objects,
  not the ``OrderIntent`` of §3.6: ``increment``, ``order_type``,
  ``time_in_force`` and ``action_class`` come from the execution profile (§3.9)
  and the action classes (§3.10), neither of which exists yet, and inventing
  values for them would put fiction in the record.
* ``DecisionOutput.rounding`` is ``None``: batch rounding repair
  (``portfolio/rounding.py::round_batch``, §3.7) is not implemented, so the
  whole-share rounding stays inside sizing exactly as the live path does it.
* ``AccountState.positions`` of ``None`` means "not reconciled yet", and then
  no intents are sized and a warning says so. In the live cycle reconciliation
  happens inside ``decide_and_submit``, *after* the write-ahead pending check,
  so at the point ``run_daily_cycle`` assembles there is no reconciled position
  book to size against. Moving reconciliation in front of assembly, and having
  ``decide_and_submit`` consume intents (§3.8), changes the write-ahead
  protocol's restart semantics and is not a Parity change; it waits for the
  stage that rewrites that protocol.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.composition import (
    COMBINATION,
    ComponentOutput,
    ComponentPlan,
    ComponentState,
    compose,
)
from prism.execution.participation import participation_capped_targets
from prism.portfolio.construct import (
    cap_book,
    construct_decile_neutral,
    construct_directional_targets,
    construct_inverse_vol_targets,
    step_no_trade_band,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prism.live.broker import Order
    from prism.live.daily import DailyBookConfig

logger = logging.getLogger(__name__)

#: The declared step orders of the heuristic constructor (§3.3).
#: ``cap_band_participation`` is the reference momentum path: the
#: rank-to-weight rule caps, the band steps against the held book, the
#: participation gate applies. ``band_cap_participation`` re-caps after the
#: band, which the recorded trend and demotion driver does because the band
#: blends held and fresh weights per name and the blend can exceed max_gross.
STEP_ORDERS: tuple[str, ...] = ("cap_band_participation", "band_cap_participation")


class AssemblyError(ValueError):
    """An assembly input is unsupported, missing, or self-contradictory."""


@dataclass(frozen=True)
class AccountState:
    """The account as assembly sees it: marked equity, held weights, held shares.

    ``positions`` is the reconciled broker-truth share book. ``None`` is not an
    empty book — it means reconciliation has not run yet, so nothing may be
    sized against it (see the module docstring).
    """

    equity: float
    held_weights: pd.Series
    cash: float = 0.0
    positions: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.equity)) or float(self.equity) <= 0.0:
            raise AssemblyError(f"equity must be finite and > 0 to size a book, got {self.equity!r}")
        if not isinstance(self.held_weights, pd.Series):
            raise AssemblyError(
                f"held_weights must be a pandas Series of marked weights, got " f"{type(self.held_weights).__name__}"
            )


@dataclass(frozen=True)
class ConstructionOutput:
    """What the constructor produced (§3.7).

    The heuristic constructor runs no solver and evaluates no holding-state
    limit, so ``solver`` is always ``None``, ``path`` is always ``"normal"``,
    and ``violations`` is empty: it classifies nothing. The limit-evaluation
    shim that fills those in for this path is stage 3 and does not change the
    target.
    """

    w_target: pd.Series
    violations: Mapping[str, list] = field(default_factory=lambda: {"existing": [], "remaining": [], "introduced": []})
    path: str = "normal"
    solver: Mapping[str, Any] | None = None
    binding_condition: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionContext:
    """Everything one decision needs (§3.8), in the shape stage 1 can supply.

    ``scoring_panels`` maps a declared ``scoring_basis`` to its close panel;
    ``"price"`` is the as-traded close and is required, because costs,
    participation, sizing and reference prices always use it whatever a
    component scores on. ``unrankable`` is supplied, never derived here.

    The trailing fields are the §3.8 carriers whose types arrive at later
    stages (the risk snapshot, the cost model, the resolved limits and
    obligations, the execution profile, dated capabilities). They are declared
    so a caller that has one can pass it and the record can hold it; nothing on
    the heuristic path reads them.
    """

    decision_bar: str
    components: Sequence[ComponentPlan]
    scoring_panels: Mapping[str, pd.DataFrame]
    construction: "DailyBookConfig"
    state: AccountState
    volume: pd.DataFrame | None = None
    sessions: Sequence[str] | None = None
    component_state: Mapping[str, ComponentState] | None = None
    unrankable: Collection[str] = ()
    renormalize: bool = False
    step_order: str = "cap_band_participation"
    max_gross: float | None = None
    regime_scale: float | None = None
    order_id_prefix: str = ""
    view: object | None = None
    calibrations: Mapping[str, Any] | None = None
    risk: object | None = None
    costs: object | None = None
    limits: Sequence[Mapping[str, Any]] = ()
    obligations: Sequence[Any] = ()
    config: object | None = None
    execution_profile: object | None = None
    capabilities: object | None = None

    def __post_init__(self) -> None:
        if self.step_order not in STEP_ORDERS:
            raise AssemblyError(
                f"unknown construction step order {self.step_order!r}; the heuristic constructor "
                f"declares {STEP_ORDERS} (§3.3)"
            )
        if "price" not in self.scoring_panels:
            raise AssemblyError(
                "scoring_panels must carry the as-traded 'price' close panel: costs, "
                "participation, sizing and reference prices use it whatever a component "
                "scores on (§3.8)"
            )


@dataclass(frozen=True)
class DecisionOutput:
    """The decision (§3.8): the target, how it was made, and what to send."""

    targets: pd.Series
    component_outputs: tuple[ComponentOutput, ...]
    construction: ConstructionOutput
    rounding: object | None
    order_intents: list["Order"]
    diagnostics: Mapping[str, Any]
    warnings: list[str]


def assemble(ctx: DecisionContext) -> DecisionOutput:
    """Turn one decision bar into an account target and order intents.

    Composition (flow F1, §3.3) attributes a sub-target per component and sums
    them into the account target; the heuristic constructor then caps, bands
    against the held book and applies the participation gate in the declared
    step order; sizing diffs the result against the reconciled share book at
    the decision closes. A sid a component ordered liquidated is exempt from
    the band and the dust rules and is flat by the time sizing sees it.
    """
    price_panel = ctx.scoring_panels["price"]
    composition = compose(
        ctx.components,
        decision_bar=ctx.decision_bar,
        scoring_panels=ctx.scoring_panels,
        sessions=ctx.sessions,
        volume=ctx.volume,
        states=ctx.component_state,
        unrankable=ctx.unrankable,
        renormalize=ctx.renormalize,
    )
    warnings = list(composition.warnings)

    targets = construct_targets(
        composition.targets,
        prev_weights=ctx.state.held_weights,
        close=price_panel,
        volume=ctx.volume,
        config=ctx.construction,
        equity=float(ctx.state.equity),
        liquidate=composition.liquidate,
        step_order=ctx.step_order,
        max_gross=ctx.max_gross,
    )

    prices = price_panel.iloc[-1]
    intents: list["Order"] = []
    if ctx.state.positions is None:
        warnings.append(
            "no reconciled position book supplied (AccountState.positions is None): the account "
            "target was assembled but no order intents were sized against it"
        )
    else:
        intents = size_orders(
            targets,
            positions=dict(ctx.state.positions),
            prices=prices,
            equity=float(ctx.state.equity),
            decision_bar=ctx.decision_bar,
            min_order_notional=float(ctx.construction.min_order_notional),
            whole_shares=bool(ctx.construction.whole_shares),
            order_id_prefix=ctx.order_id_prefix,
            exempt=composition.liquidate,
        )

    masked = sorted({sid for output in composition.component_outputs for sid in output.masked})
    diagnostics: dict[str, Any] = {
        "decision_bar": ctx.decision_bar,
        "step_order": ctx.step_order,
        "combination": COMBINATION,
        "masked": masked,
        "liquidate": sorted(composition.liquidate),
        "renormalization": composition.renormalization,
        "regime_scale": ctx.regime_scale,
        "gross": float(targets.abs().sum()),
        "n_orders": len(intents),
        "component_states": dict(composition.states),
    }
    construction = ConstructionOutput(
        w_target=targets,
        diagnostics={
            "step_order": ctx.step_order,
            "max_gross": ctx.max_gross,
            "participation_applied": ctx.construction.max_participation is not None,
            "band": float(ctx.construction.no_trade_band),
            "gross": float(targets.abs().sum()),
        },
    )
    return DecisionOutput(
        targets=targets,
        component_outputs=composition.component_outputs,
        construction=construction,
        rounding=None,
        order_intents=intents,
        diagnostics=diagnostics,
        warnings=warnings,
    )


def heuristic_book_rule(
    config: "DailyBookConfig",
    close: pd.DataFrame,
    max_gross: float,
) -> Callable[[pd.Series], pd.Series]:
    """The live book's construct as a ``ComponentPlan.weights_fn`` (Parity).

    The recorded books are constructed from the whole close panel and a
    regime-scaled cap, neither of which a rank-to-weight rule name can carry,
    so the live path binds them here and composition applies the result as the
    component's rule. The returned callable takes one decision bar's score
    cross-section and returns that bar's raw target weights, pre-band and
    pre-gate — bit for bit the row ``run_daily_cycle`` built before this module
    existed.
    """

    def rule(scores: pd.Series) -> pd.Series:
        score_row = pd.DataFrame([scores], index=close.index[-1:])
        return raw_book_weights(score_row, close, config, max_gross)

    return rule


def raw_book_weights(
    score_row: pd.DataFrame,
    close: pd.DataFrame,
    config: "DailyBookConfig",
    effective_max_gross: float,
) -> pd.Series:
    """One scored row → the book's raw target weights (pre-band, pre-gate)."""
    if effective_max_gross <= 0.0:
        # A zero effective cap — regime_scale clamped to 0.0, the full
        # de-gross — admits no gross, and cap_book refuses a 0 cap, so
        # emit each book's own flat form directly: the decile /
        # inverse_vol books' explicit 0.0 for every name (NaN-score pin
        # included), the directional book's flat-where-decided (a NaN
        # score keeps carrying no decision, exactly as the construct
        # would emit). The band step then exits every held, decidable name.
        if config.book in ("decile_neutral", "inverse_vol"):
            return pd.Series(0.0, index=score_row.columns, dtype=float)
        return score_row.iloc[0] * 0.0
    if config.book == "decile_neutral":
        # The decile construct emits an explicit weight (0.0 outside the
        # decile) for every scored name, so a held name that falls out of the
        # decile — including a book inherited at cutover, as long as it is in
        # the fetched universe — is rebalanced to flat by the online band. A
        # held name absent from the universe cannot be priced and is refused
        # loudly at the mark step (N7): trade only what you can value, so a
        # cutover universe must contain the names it inherits.
        return construct_decile_neutral(
            score_row,
            decile=config.decile,
            max_gross=effective_max_gross,
            max_symbol_abs_weight=config.max_symbol_abs_weight,
        ).iloc[0]
    if config.book == "inverse_vol":
        # Trend sleeve (docs/trend_design.md §2). Vol needs trailing
        # close history; align the one-row score onto the full panel so
        # only the decision bar carries a sign and earlier bars stay
        # NaN (unused by the last-row construct read).
        scores_aligned = pd.DataFrame(np.nan, index=close.index, columns=score_row.columns, dtype=float)
        scores_aligned.iloc[-1] = score_row.iloc[0].reindex(score_row.columns)
        return construct_inverse_vol_targets(
            scores_aligned,
            close.reindex(columns=score_row.columns),
            vol_ewma_bars=config.vol_ewma_bars,
            max_gross=effective_max_gross,
            max_symbol_abs_weight=config.max_symbol_abs_weight,
        ).iloc[-1]
    return construct_directional_targets(
        score_row,
        position_size=config.position_size,
        max_gross=effective_max_gross,
        max_symbol_abs_weight=config.max_symbol_abs_weight,
        no_trade_band=0.0,  # the online band owns hysteresis (never the flat replay)
    ).iloc[0]


def construct_targets(
    raw: pd.Series,
    *,
    prev_weights: pd.Series,
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    config: "DailyBookConfig",
    equity: float,
    liquidate: Collection[str] = (),
    step_order: str = "cap_band_participation",
    max_gross: float | None = None,
) -> pd.Series:
    """The account target through the band and the participation gate (Parity).

    Sizing folds in exactly once (I-4): the rank-to-weight rule owns the
    weights and their caps, the online band owns hysteresis against *held*
    weights, and the participation gate shrinks trades over the %ADV cap. A sid
    under a ``liquidate`` instruction bypasses the band — the flat is the
    decision, not a target to be damped — but the participation gate still
    applies to it (§3.3).

    ``band_cap_participation`` re-caps after the band because a per-name blend
    of held and fresh weights can exceed ``max_gross``; the cap used is
    ``max_gross`` when given, else the book's configured one.
    """
    if step_order not in STEP_ORDERS:
        raise AssemblyError(f"unknown construction step order {step_order!r}; expected {STEP_ORDERS}")
    targets = step_no_trade_band(prev_weights, raw, config.no_trade_band)
    exempt = [sid for sid in dict.fromkeys(liquidate) if sid in raw.index]
    if exempt:
        # The band held these back toward what is held; a liquidate instruction
        # is exempt from it, so the flat stands.
        targets.loc[exempt] = raw.loc[exempt].to_numpy(dtype=float)
    if step_order == "band_cap_participation":
        cap = float(config.max_gross if max_gross is None else max_gross)
        targets = cap_book(
            targets.to_frame().T,
            max_gross=cap,
            max_symbol_abs_weight=config.max_symbol_abs_weight,
        ).iloc[0]
    if config.max_participation is not None:
        if volume is None:
            raise ValueError("max_participation is set but there is no volume panel to compute ADV (N7)")
        dollar_volume = (close * volume).rolling(config.adv_window_bars, min_periods=1).mean().iloc[-1]
        targets = participation_capped_targets(
            prev_weights,
            targets,
            dollar_volume,
            aum=equity,
            max_participation=config.max_participation,
        )
    return targets


def size_orders(
    target_weights: pd.Series,
    positions: Mapping[str, float],
    prices: pd.Series,
    equity: float,
    decision_bar: str,
    min_order_notional: float = 0.0,
    whole_shares: bool = False,
    order_id_prefix: str = "",
    exempt: Collection[str] = (),
) -> list["Order"]:
    """Diff target weights against held shares into orders (Parity).

    A NaN target is "no target" — the position is left alone (the signal
    layer's no-opinion never forces a liquidation; construction emits an
    explicit 0.0 when it wants flat). A symbol with a *non-flat* target but no
    finite positive price cannot be sized and raises (N7); a flat (0.0) target
    on an unheld name needs no price and is skipped. Orders below
    ``min_order_notional`` dollars are dropped as dust. ``whole_shares``
    rounds each share *delta* to the nearest integer (required by
    market-on-open venue order types, e.g. Alpaca OPG); a delta that rounds
    to zero is dropped like dust — and because a censored order is a silent
    book/target divergence (a $2,000+ name can never enter a $1,000-per-name
    book), every drop whose *intended* notional was economically real is
    logged loudly with its size (N7: loud, not silent). An order that would
    flip a held position through zero (short→long or long→short) is clamped
    to flat this bar — a venue rejects a side-crossing single order
    ("insufficient qty available"), so the opposite side opens next bar from
    a flat book. ``order_id_prefix`` namespaces the deterministic client id
    (``"{prefix}{decision_bar}:{symbol}"``) so two books on one venue account
    can never collide on the same bar and symbol.

    ``exempt`` names sids a ``liquidate`` instruction flattened: they are
    exempt from the dust rules (§3.3), so a small residual position is still
    exited. A delta that rounds to zero whole shares is still not an order —
    there is no such quantity to send.
    """
    # Imported here, not at module scope: `prism.live` imports the daily driver,
    # which imports this module, so a module-level import of a `prism.live`
    # submodule would make the two packages import-order dependent.
    from prism.live.broker import Order

    if equity <= 0:
        raise ValueError(f"equity must be > 0 to size a book, got {equity}")
    exempt_sids = {str(sid) for sid in exempt}
    orders: list[Order] = []
    dust_dropped: list[tuple[str, float]] = []  # (symbol, intended $ notional)
    for symbol, weight in target_weights.items():
        if isinstance(weight, float) and math.isnan(weight):
            continue
        held = positions.get(symbol, 0.0)
        if float(weight) == 0.0 and held == 0.0:
            # Flat target with nothing held: no trade to size, so no decision
            # price is needed. A large universe carries names the construct left
            # at 0.0 (ineligible, or a gapped latest bar) whose price is NaN — a
            # no-op here, not an N7 error.
            continue
        price = prices.get(symbol)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            raise ValueError(
                f"no usable decision price for {symbol!r} (got {price!r}); " "cannot size its target (N7)"
            )
        price = float(price)
        target_shares = float(weight) * equity / price
        if held != 0.0 and target_shares != 0.0 and (held > 0.0) != (target_shares > 0.0):
            # A single venue order may not cross zero — Alpaca rejects flipping a
            # position's side in one order ("insufficient qty available"). Trade
            # only to flat this bar; the opposite side opens next bar from flat.
            delta = -held
        else:
            delta = target_shares - held
        intended_notional = abs(delta) * price
        if whole_shares:
            delta = float(round(delta))
        floor = 0.0 if str(symbol) in exempt_sids else max(min_order_notional, 1e-9)
        if abs(delta) * price < floor or delta == 0.0:
            if intended_notional >= max(min_order_notional, 1.0):
                # The pre-round intent was a real order; rounding/dust censored
                # it. Silent censoring is a reproducible book/target divergence.
                dust_dropped.append((str(symbol), intended_notional))
            continue
        orders.append(
            Order(
                client_order_id=f"{order_id_prefix}{decision_bar}:{symbol}",
                symbol=str(symbol),
                qty=delta,
                decision_bar=decision_bar,
                reference_price=price,
            )
        )
    if dust_dropped:
        shown = sorted(dust_dropped, key=lambda item: -item[1])[:20]
        tail = "…" if len(dust_dropped) > len(shown) else ""
        logger.warning(
            "%s: %d intended orders dropped by whole-share/notional rounding "
            "(book/target divergence, N7-loud): %s%s",
            decision_bar,
            len(dust_dropped),
            [(sym, round(notional, 2)) for sym, notional in shown],
            tail,
        )
    return orders
