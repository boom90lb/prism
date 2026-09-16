"""The adversarial session set (engine spec §5, "The adversarial session set"; §3.2, §3.6, §3.7).

An adversarial session is a triple: one scenario id from the §3.2 list, a three-bar
sequence -- the prior close the starting state is built at, the decision close ``t``, the
bar whose open fills and whose close marks -- and the ledger invariants that sequence must
satisfy. The set is the full scenario list crossed with three portfolio starting states
(``flat``, ``long_only``, ``long_short`` with a short in the affected name), written by
``prism data synthetic --scenarios all`` and driven through
:func:`prism.assembly.assemble` and :func:`prism.execution.simulate.simulate`.

Twelve scenarios by three starting states is **36 crosses**. Thirty-four are constructed
and run; the two that cannot be constructed are declared in :data:`REFUSALS`, and
:func:`test_a_declared_refusal_raises_for_its_declared_reason` asserts each one raises for
that reason rather than being skipped quietly.

**How a session is priced, and why it is bar by bar.** Every bar of the window is read
through a :class:`~prism.io.dataset.DatasetView` dated at *that bar*, which is what a live
loop sees: the prior and decision bars are priced as they were known then, and the fill bar
is priced as it is known on the fill bar. That is the only construction under which the
``rebase`` scenario is a *revision* at all -- a single view taken at the end of the window
would show the vendor's back-rewrite applied to the whole history and there would be
nothing to be discontinuous across. The one departure is :data:`KNOWLEDGE_FLOOR`, which
floors the knowledge date for the ``revision`` scenario at the day the null-valued full
revision landed, because that scenario's whole content is a bar that later stops existing.

**Who enforces the action restrictions.** §3.7 puts them in ``portfolio/validate.py`` and
``portfolio/rounding.py``, not in the accounting kernel, and the canonical simulator
deliberately calls neither: :func:`prism.execution.accounting.decide` is the sizing half of
the decision. So the driver derives the restriction set for the decision bar from the dated
``capabilities`` row, the halt and cessation events visible at that close, and the printed
price, and masks the assembled target with it -- exactly the layer a live loop supplies.
:func:`test_no_order_the_action_restrictions_forbid` then checks every intent the engine
produced against that same set, so an engine that *manufactures* an order the caller did
not ask for is caught.

**What this set found.** Three defects, each of which loses money without saying so, each
fixed in ``execution/accounting.py`` and pinned by a named test here:

* a ``rebase`` adjustment -- the vendor back-rewrite of decision D3 -- had no path into the
  book at all, so a holding across a 4-for-1 rewrite lost three quarters of its value at
  the next mark and realised the loss in cash at the next fill
  (:func:`test_a_vendor_back_rewrite_rebases_shares_and_leaves_equity_continuous`);
* a split cancelled the orders resting on the name only when the name was *held*, so a
  flat book's decided buy survived the split and filled the pre-split share count at the
  post-split price, acquiring half the decided notional
  (:func:`test_a_split_cancels_the_orders_resting_on_the_name_even_when_it_is_not_held`);
* nothing cancelled the orders resting on an instrument that had stopped trading, so an
  order rested for ever against a dead name and ``committed_cash`` reserved cash against it
  permanently (:func:`test_a_dead_instrument_carries_no_resting_order`).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from prism.assembly import AccountState, DecisionContext, assemble, heuristic_book_rule
from prism.composition import ComponentPlan
from prism.execution.accounting import apply_events, apply_fills, expire_orders, mark_at_close, mark_prices, submit
from prism.execution.cost_model import CostInputs, CostModel, CostStack
from prism.execution.portfolio_state import (
    AccountingError,
    Fill,
    OrderIntent,
    PortfolioState,
    empty_state,
    state_from_shares,
)
from prism.execution.simulate import FillPolicy, SimulationResult, simulate
from prism.io.dataset import Dataset, DatasetView
from prism.io.events import series_price_basis
from prism.io.manifest import Manifest
from prism.io.synthetic import ALL_SCENARIOS, ScenarioCase, scenario_catalog, write_synthetic_dataset
from prism.live.daily import DailyBookConfig
from prism.portfolio.limits import (
    classify_action,
    restriction_blocks_all_trading,
    restriction_blocks_new_exposure,
)
from prism.signal.base import Signal

START = dt.date(2026, 1, 1)
END = dt.date(2026, 12, 31)
SEED = 7
VINTAGE = "synthetic_v1"

#: The account every session opens with, in dollars of marked equity at the prior close.
CAPITAL = 1_000_000.0

#: Two names no scenario touches, so every book holds something whose behaviour is
#: uneventful and a scenario's effect is never the only thing in the ledger.
BALLAST = ("sec_000012", "sec_000013")

#: The three declared starting states (§5).
START_STATES = ("flat", "long_only", "long_short")

#: The role each scenario's *affected name* plays in its :class:`ScenarioCase`. The
#: long/short state is short exactly this name, which is what makes the short leg of every
#: event -- a distribution the book *pays*, a borrow accrual, a claim that cannot be
#: covered -- part of the set rather than an afterthought.
AFFECTED_ROLE: dict[str, str] = {
    "membership": "removed",
    "split": "raw",
    "distribution": "payer",
    "halt": "halted",
    "costs": "thin_volume",
    "rounding": "rounded",
    "spinoff": "parent",
    "ticker_reuse": "first",
    "revision": "revised",
    "rebase": "rebased",
    "delisting": "unknown",
    "recovery": "gapped",
}

#: Scenarios whose knowledge date is floored past the window, with the case date that
#: floors it. Only ``revision``: its null-valued full revision lands six sessions after the
#: bar it withdraws, and a session priced strictly bar by bar would never see it, so the
#: scenario's own expectation -- "marks() carries the holding stale at the last surviving
#: priced bar rather than at zero" -- would go unexercised.
KNOWLEDGE_FLOOR: dict[str, str] = {"revision": "withdrawn_from"}

#: The crosses that cannot be constructed, and why. Each is asserted to raise, by
#: :func:`test_a_declared_refusal_raises_for_its_declared_reason`; none is skipped.
REFUSALS: dict[tuple[str, str], str] = {
    ("spinoff", "long_only"): (
        "the canonical engine declares no valuation for the spun-off child at the moment of distribution, so a "
        "spinoff on a held parent is refused rather than dropped (§3.6). A dropped spinoff loses the child's "
        "value silently, because the parent price marks down for it on the ex-date"
    ),
    ("spinoff", "long_short"): (
        "same refusal as the long-only cross, and the short leg makes it worse: a short parent owes the child, so "
        "dropping the distribution would understate a liability rather than only omitting an asset"
    ),
}

#: Event types that end an instrument's life, so that no order may rest on it afterwards.
TERMINAL_EVENT_TYPES = ("trading_cessation", "terminal_payout", "share_cancellation")


# ---------------------------------------------------------------------------
# the declared cost stack and the scoring component
# ---------------------------------------------------------------------------


def cost_stack() -> CostStack:
    """A fully declared stack, zeros included, with every term switched on.

    Nothing here is a reference value chosen for realism; it is chosen so that every
    charged term is nonzero and a session that silently stopped charging one would show.
    """
    dollar_volume = {
        "spread_bucket": {
            "estimator": "median",
            "window": "trailing",
            "window_bars": 20,
            "refresh": "decision",
            "floor_dollars": 0.0,
        },
        "participation": {
            "estimator": "median",
            "window": "trailing",
            "window_bars": 20,
            "refresh": "decision",
            "floor_dollars": 0.0,
            "base": "current_equity",
        },
        "impact": {
            "estimator": "same_bar",
            "window": "trailing",
            "window_bars": 1,
            "refresh": "bar",
            "floor_dollars": 100000.0,
        },
    }
    return CostStack.from_mapping(
        {
            "spread": {"mode": "flat", "flat_bps": 5.0},
            "impact": {
                "quadratic_notional_coeff": 10.0,
                "participation_model": "sqrt",
                "participation_coeff": 0.0,
            },
            "dollar_volume": dollar_volume,
            "commission_bps": 1.0,
            "borrow_bps_annual": 50.0,
            "cash_income_bps_annual": 0.0,
            "financing_bps_annual": 0.0,
            "short_proceeds": "collateral_no_rebate",
            "holding_cost_settlement": "per_bar",
            "trading_days_per_year": 252,
        }
    )


class TwentyOneBarMomentum(Signal):
    """A closed-form score: the 21-bar price ratio, no RNG and no fit.

    The component is not the point of this file -- the ledger is -- so it is the simplest
    thing that produces a decision on every name that printed and ``NaN`` on every name
    that did not, which is what makes "a name with no decision-bar price is not traded" a
    property of the session rather than of the score.
    """

    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    price_scale_invariant = True
    fitted_state = "none"
    estimated_decay_bars = None

    @property
    def horizon_bars(self) -> int:
        return 21

    @property
    def required_history(self) -> int:
        return 22

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "TwentyOneBarMomentum":
        return self

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        return (close.iloc[-1] / close.iloc[-22] - 1.0).astype(float)


def book_config() -> DailyBookConfig:
    return DailyBookConfig(
        book="directional",
        position_size=0.05,
        max_gross=1.0,
        max_symbol_abs_weight=0.10,
        no_trade_band=0.0,
        max_participation=0.05,
        adv_window_bars=20,
        min_order_notional=1.0,
        whole_shares=True,
    )


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Session:
    """One adversarial session: the triple, everything it was driven with, and what it did."""

    scenario: str
    start_state: str
    case: ScenarioCase
    affected: str
    universe: tuple[str, ...]
    bars: tuple[str, str, str]
    opens: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame
    events: pd.DataFrame
    adjustments: pd.DataFrame
    price_basis: Mapping[str, str]
    increments: Mapping[str, float]
    min_notionals: Mapping[str, float]
    capabilities: pd.DataFrame
    restrictions: Mapping[str, frozenset[str]]
    initial: PortfolioState
    targets: pd.DataFrame
    assembled: pd.Series
    result: SimulationResult
    warnings: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return f"{self.scenario}/{self.start_state}"

    @property
    def prior_bar(self) -> str:
        return self.bars[0]

    @property
    def decision_bar(self) -> str:
        return self.bars[1]

    @property
    def fill_bar(self) -> str:
        return self.bars[2]

    def traded(self, bar: str) -> frozenset[str]:
        """The names that printed on ``bar``: a price or a volume, and nothing else."""
        out: set[str] = set()
        for frame in (self.opens, self.closes, self.volumes):
            row = frame.loc[bar]
            for sid in self.universe:
                value = float(row.get(sid, np.nan))
                if math.isfinite(value) and value > 0.0:
                    out.add(sid)
        return frozenset(out)

    def price_row(self, frame: pd.DataFrame, bar: str) -> pd.Series:
        return pd.to_numeric(frame.loc[bar], errors="coerce").astype(float)


class _Views:
    """A per-date cache of dataset views: a session opens nine and there are thirty-four."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self._cache: dict[str, DatasetView] = {}

    def at(self, day: str) -> DatasetView:
        if day not in self._cache:
            self._cache[day] = self.dataset.view("research_vintage", day, vintage=VINTAGE)
        return self._cache[day]


def _capability_restrictions(row: pd.Series) -> set[str]:
    raw = row.get("restrictions")
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return set()
    if isinstance(raw, str):
        return {str(item) for item in json.loads(raw)}
    return {str(item) for item in raw}


def _restrictions_at_decision(
    view: DatasetView,
    *,
    universe: tuple[str, ...],
    decision_bar: str,
    start: str,
    capabilities: pd.DataFrame,
    close_row: pd.Series,
    volume_row: pd.Series,
) -> dict[str, frozenset[str]]:
    """The §3.7 action restriction set per name at the decision close.

    Four sources, all of them dated at or before the decision bar:

    * the ``capabilities`` row's own ``restrictions`` column, plus ``shortable``, which is
      an action restriction expressed as a flag;
    * a ``halt`` whose window covers the decision bar -- §3.6 says the participation
      restriction refuses any order on a halted name;
    * a ``trading_cessation``, ``terminal_payout`` or ``share_cancellation`` effective on
      or before the decision bar: the instrument's life ended, so no order placed at this
      close can fill at any later open;
    * no printed close, or no printed volume, at the decision bar -- the participation
      bound of §3.6 refuses a trade on a name the decision cannot price or whose dollar
      volume is not positive, and that restriction is ``PHYSICAL`` and never waived.
    """
    out: dict[str, set[str]] = {sid: set() for sid in universe}
    for sid in universe:
        row = capabilities.loc[sid]
        out[sid] |= _capability_restrictions(row)
        if not bool(row.get("shortable", True)):
            out[sid].add("no_short")
        price = float(close_row.get(sid, np.nan))
        volume = float(volume_row.get(sid, np.nan))
        if not (math.isfinite(price) and price > 0.0):
            out[sid].add("no_trade")
        if not (math.isfinite(volume) and volume > 0.0):
            out[sid].add("no_trade")
    events = view.events(list(universe), start, decision_bar)
    for record in events.to_dict("records"):
        sid = str(record["sid"])
        if sid not in out:
            continue
        kind = str(record["event_type"])
        if kind in TERMINAL_EVENT_TYPES:
            out[sid].add("no_trade")
        elif kind == "halt":
            payload = record.get("payload")
            body = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            end = str(body.get("end") or record["effective_date"])
            if str(record["effective_date"]) <= decision_bar <= end:
                out[sid].add("halted")
    return {sid: frozenset(values) for sid, values in out.items()}


def _mask_target(
    target: pd.Series,
    *,
    held_weights: pd.Series,
    restrictions: Mapping[str, frozenset[str]],
) -> pd.Series:
    """Apply the action restrictions to the assembled target (§3.7).

    A name the restrictions bar from trading is given the ``NaN`` no-decision marker, which
    holds the position exactly and emits no trade. A name barred from new exposure keeps
    only a target that is a strict reduction on the side already held, and is otherwise cut
    to zero -- so every surviving trade on it is a ``C1`` action in the §3.10
    classification. A name that may not be shorted is floored at the short it already has.
    """
    masked = target.copy()
    for sid in list(masked.index):
        rules = restrictions.get(str(sid), frozenset())
        held = float(held_weights.get(sid, 0.0))
        if restriction_blocks_all_trading(rules) or "no_trade" in rules:
            # ``NaN`` is the declared no-decision marker (§3.6): it holds the position
            # exactly. Writing the held *weight* instead would not, because the weight was
            # measured at the prior close and the decision is sized at the decision close,
            # so re-expressing it in shares emits a small trade -- on a name that may not
            # trade at all.
            masked.loc[sid] = np.nan
            continue
        wanted = float(masked.loc[sid])
        if not math.isfinite(wanted):
            continue
        if restriction_blocks_new_exposure(rules):
            # Reduce-only. A target of zero is a reduction whatever the held size, so it is
            # the safe expression of "this order may not open or add"; a requested target
            # that is already a strict reduction on the same side passes through.
            if held == 0.0 or (wanted > 0.0) != (held > 0.0) or abs(wanted) >= abs(held):
                masked.loc[sid] = 0.0
        if "no_short" in rules and float(masked.loc[sid]) < min(0.0, held):
            masked.loc[sid] = min(0.0, held)
    return masked


def build_session(dataset: Dataset, manifest: Manifest, views: _Views, scenario: str, start_state: str) -> Session:
    """Construct and run one adversarial session. Raises where the cross cannot be built."""
    case = scenario_catalog(manifest)[scenario]
    affected = case.sid(AFFECTED_ROLE[scenario])
    universe = tuple(sorted(set(case.sids.values()) | set(BALLAST)))
    bars = (case.prior_bar, case.decision_bar, case.fill_bar)
    floor = case.date(KNOWLEDGE_FLOOR[scenario]) if scenario in KNOWLEDGE_FLOOR else None

    # -- the panels, read bar by bar at each bar's own knowledge date ------
    opens: dict[str, dict[str, float]] = {}
    closes: dict[str, dict[str, float]] = {}
    volumes: dict[str, dict[str, float]] = {}
    event_frames: list[pd.DataFrame] = []
    adjustment_frames: list[pd.DataFrame] = []
    for bar in bars:
        view = views.at(max(bar, floor) if floor else bar)
        for field_name, sink in (("open", opens), ("close", closes), ("volume", volumes)):
            panel = view.panel(list(universe), bar, field_name)
            days = [str(stamp.date()) for stamp in panel.index]
            row: dict[str, float] = {}
            if bar in days:
                values = panel.iloc[days.index(bar)]
                row = {sid: float(values.get(sid, np.nan)) for sid in universe}
            sink[bar] = row
        event_frames.append(view.events(list(universe), bar, bar))
        adjustment_frames.append(view.adjustments(list(universe), bar, bar))
    opens_frame = pd.DataFrame(opens).T.reindex(index=list(bars), columns=list(universe))
    closes_frame = pd.DataFrame(closes).T.reindex(index=list(bars), columns=list(universe))
    volumes_frame = pd.DataFrame(volumes).T.reindex(index=list(bars), columns=list(universe))
    events = (
        pd.concat([f for f in event_frames if len(f)], ignore_index=True)
        if any(len(f) for f in event_frames)
        else pd.DataFrame()
    )
    adjustments = (
        pd.concat([f for f in adjustment_frames if len(f)], ignore_index=True)
        if any(len(f) for f in adjustment_frames)
        else pd.DataFrame()
    )

    decision_view = views.at(case.decision_bar)
    close_panel = decision_view.panel(list(universe), case.decision_bar, "close")
    volume_panel = decision_view.panel(list(universe), case.decision_bar, "volume")
    price_basis: dict[str, str] = {}
    for sid in universe:
        rows = decision_view.bars([sid], case.decision_bar)
        price_basis[sid] = series_price_basis(rows) if len(rows) else "split_adjusted_at_ingest"

    # -- the starting book, marked at the prior close ---------------------
    prior_close = pd.to_numeric(closes_frame.loc[case.prior_bar], errors="coerce").astype(float)
    shares: dict[str, float] = {}
    cash = CAPITAL
    if start_state != "flat":
        for sid in BALLAST:
            price = float(prior_close[sid])
            quantity = float(np.floor(0.10 * CAPITAL / price))
            shares[sid] = quantity
            cash -= quantity * price
        price = float(prior_close.get(affected, np.nan))
        if not (math.isfinite(price) and price > 0.0):
            raise AccountingError(
                f"{scenario}/{start_state}: {affected} printed no close at the prior bar {case.prior_bar}, so a "
                "starting book holding it cannot be marked and would have to be valued at nothing"
            )
        quantity = float(np.floor(0.10 * CAPITAL / price))
        if start_state == "long_only":
            shares[affected] = quantity
            cash -= quantity * price
        else:
            shares[affected] = -quantity
            cash += quantity * price
    initial = (
        state_from_shares(case.prior_bar, shares, prior_close, cash=cash)
        if shares
        else empty_state(case.prior_bar, cash=CAPITAL)
    )

    # -- the decision: assemble, then mask with the action restrictions ----
    equity = initial.equity(prior_close)
    held_weights = pd.Series(
        {sid: initial.shares(sid) * float(prior_close[sid]) / equity for sid in initial.sids()}, dtype=float
    )
    config = book_config()
    plan = ComponentPlan(
        component=TwentyOneBarMomentum(),
        component_id="adversarial",
        to_weights=config.book,
        weights_fn=heuristic_book_rule(config, close_panel, config.max_gross),
    )
    context = DecisionContext(
        decision_bar=case.decision_bar,
        components=(plan,),
        scoring_panels={"price": close_panel},
        volume=volume_panel,
        construction=config,
        state=AccountState(
            equity=equity,
            held_weights=held_weights if len(held_weights) else pd.Series(dtype=float),
            cash=initial.cash,
            positions={sid: initial.shares(sid) for sid in initial.sids()},
        ),
        sessions=[str(stamp.date()) for stamp in close_panel.index],
        max_gross=config.max_gross,
    )
    decision = assemble(context)

    capabilities = decision_view.capabilities(list(universe))
    increments = {sid: float(capabilities.loc[sid, "quantity_increment"]) for sid in universe}
    min_notionals = {sid: float(capabilities.loc[sid, "min_notional"]) for sid in universe}
    restrictions = _restrictions_at_decision(
        decision_view,
        universe=universe,
        decision_bar=case.decision_bar,
        start=case.prior_bar,
        capabilities=capabilities,
        close_row=pd.to_numeric(closes_frame.loc[case.decision_bar], errors="coerce").astype(float),
        volume_row=pd.to_numeric(volumes_frame.loc[case.decision_bar], errors="coerce").astype(float),
    )
    assembled = decision.targets.reindex(list(universe))
    masked = _mask_target(
        decision.targets.reindex(list(universe)), held_weights=held_weights, restrictions=restrictions
    )
    targets = pd.DataFrame(np.nan, index=list(bars), columns=list(universe))
    targets.loc[case.decision_bar] = masked

    model = CostModel(cost_stack(), CostInputs(as_of=case.decision_bar), equity=equity, horizon_bars=21)
    result = simulate(
        opens_frame,
        closes_frame,
        targets,
        cost_model=model,
        volumes=volumes_frame,
        events=events,
        adjustments=adjustments,
        price_basis=price_basis,
        initial_state=initial,
        fills=FillPolicy(),
        increment=increments,
        min_notional=min(min_notionals.values()),
        order_type="market_on_open",
        bars=list(bars),
    )
    return Session(
        scenario=scenario,
        start_state=start_state,
        case=case,
        affected=affected,
        universe=universe,
        bars=bars,
        opens=opens_frame,
        closes=closes_frame,
        volumes=volumes_frame,
        events=events,
        adjustments=adjustments,
        price_basis=price_basis,
        increments=increments,
        min_notionals=min_notionals,
        capabilities=capabilities,
        restrictions=restrictions,
        initial=initial,
        targets=targets,
        assembled=assembled,
        result=result,
        warnings=tuple(decision.warnings),
    )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Dataset, Manifest]:
    root = Path(tmp_path_factory.mktemp("adversarial")) / "data"
    manifest = write_synthetic_dataset(root, seed=SEED, scenarios="all", n_equities=14, n_etfs=3, start=START, end=END)
    return Dataset.open(root), manifest


@pytest.fixture(scope="module")
def sessions(dataset: tuple[Dataset, Manifest]) -> dict[tuple[str, str], Session]:
    """Every constructible cross, run once. The refusals of :data:`REFUSALS` are absent."""
    data, manifest = dataset
    views = _Views(data)
    out: dict[tuple[str, str], Session] = {}
    for scenario in sorted(ALL_SCENARIOS):
        for start_state in START_STATES:
            if (scenario, start_state) in REFUSALS:
                continue
            out[(scenario, start_state)] = build_session(data, manifest, views, scenario, start_state)
    return out


CROSSES = [(scenario, state) for scenario in sorted(ALL_SCENARIOS) for state in START_STATES]
RUNNABLE = [cross for cross in CROSSES if cross not in REFUSALS]
IDS = [f"{scenario}-{state}" for scenario, state in RUNNABLE]


@pytest.fixture(params=RUNNABLE, ids=IDS)
def session(request: pytest.FixtureRequest, sessions: dict[tuple[str, str], Session]) -> Session:
    return sessions[request.param]


# ---------------------------------------------------------------------------
# the set itself
# ---------------------------------------------------------------------------


def test_the_set_is_the_full_scenario_list_crossed_with_three_starting_states() -> None:
    """Twelve scenarios by three states, all accounted for: thirty-four run, two refused."""
    assert len(ALL_SCENARIOS) == 12
    assert len(CROSSES) == 36
    assert len(RUNNABLE) == 34
    assert len(REFUSALS) == 2
    assert set(AFFECTED_ROLE) == set(ALL_SCENARIOS)


def test_every_session_names_a_three_bar_window_in_ascending_order(session: Session) -> None:
    prior, decision, fill = session.bars
    assert prior < decision < fill
    assert list(session.result.ledger.index) == [prior, decision, fill]
    assert session.affected in session.universe


def test_no_session_is_vacuous(session: Session) -> None:
    """Each session decides, and moves something. A session that does nothing proves nothing."""
    result = session.result
    decided = [intent for batch in result.decisions for intent in batch.intents]
    assert decided, f"{session.id}: the decision produced no order at all"
    assert float(result.costs["total"].sum()) > 0.0, f"{session.id}: nothing was charged"
    assert len(result.fills) or not result.events.empty, f"{session.id}: neither a fill nor an event"
    assert session.affected in set(session.restrictions), "the affected name is not in the restriction set"
    if session.start_state == "long_only":
        assert session.initial.shares(session.affected) > 0.0
        assert all(session.initial.shares(sid) > 0.0 for sid in session.initial.sids())
    if session.start_state == "long_short":
        assert session.initial.shares(session.affected) < 0.0, "the long/short book must be short the affected name"
        prior = session.price_row(session.closes, session.prior_bar)
        assert session.initial.short_collateral(prior) > 0.0
    if session.start_state == "flat":
        assert session.initial.positions == {}
        assert session.initial.cash == CAPITAL


@pytest.mark.parametrize("cross", sorted(REFUSALS), ids=[f"{s}-{t}" for s, t in sorted(REFUSALS)])
def test_a_declared_refusal_raises_for_its_declared_reason(
    dataset: tuple[Dataset, Manifest], cross: tuple[str, str]
) -> None:
    """A cross that cannot be constructed says why, loudly, and is never skipped."""
    data, manifest = dataset
    scenario, start_state = cross
    with pytest.raises(AccountingError) as excinfo:
        build_session(data, manifest, _Views(data), scenario, start_state)
    message = str(excinfo.value)
    assert "spinoff" in message
    assert "no declared valuation" in message
    assert REFUSALS[cross]


# ---------------------------------------------------------------------------
# invariant 1: the balance sheet closes, before and after every transition
# ---------------------------------------------------------------------------


def _closes_to_the_cent(state: PortfolioState, prices: pd.Series | None, where: str) -> float:
    """``cash + marked positions + receivables - liabilities == equity``, to the cent.

    With no quotes the marks are the ones the state carries, which is the only valuation
    that exists *between* two bars -- after an event has been applied and before the bar
    has printed anything. Summed here from the positions themselves rather than taken from
    a helper, so that the identity is reconstructed and not restated.
    """
    if prices is None:
        equity = state.carried_equity()
        marked = sum(
            position.shares * float(position.last_price or float("nan")) for position in state.positions.values()
        )
    else:
        equity = state.equity(prices)
        marked = state.positions_value(prices)
    identity = state.cash + marked + state.receivables_total() - state.liabilities_total()
    assert equity == pytest.approx(identity, abs=1e-9), where
    return equity


def test_the_balance_sheet_closes_on_every_recorded_row(session: Session) -> None:
    ledger = session.result.ledger
    identity = ledger["cash"] + ledger["positions_value"] + ledger["receivables"] - ledger["liabilities"]
    worst = float((ledger["equity_close"] - identity).abs().max())
    assert worst < 0.01, f"{session.id}: the balance sheet is out by {worst} dollars"
    pd.testing.assert_series_equal(ledger["equity_close"], identity, check_names=False, rtol=0.0, atol=1e-9)


def test_the_balance_sheet_closes_before_and_after_every_transition(session: Session) -> None:
    """Walk the six transitions of one bar and check the identity and the expected delta.

    The fills replayed here are the ones the simulator recorded, so the fill *model* is not
    re-implemented; what is re-implemented is the **order** of the transitions, and the
    final state of this walk is compared with the simulator's in
    :func:`test_the_transition_replay_reproduces_the_simulated_book`. A divergence there is
    a statement that the simulator's order is not the declared one.
    """
    result = session.result
    model = CostModel(cost_stack(), CostInputs(as_of=session.decision_bar), equity=CAPITAL, horizon_bars=21)
    state = session.initial.copy()
    _closes_to_the_cent(state, None, f"{session.id}: opening book")

    for bar in session.bars:
        open_row = session.price_row(session.opens, bar)
        close_row = session.price_row(session.closes, bar)
        traded = session.traded(bar)

        before = state.carried_equity()
        cash_before = state.cash
        outcome = apply_events(
            state,
            bar=bar,
            events=session.events,
            adjustments=session.adjustments,
            price_basis=session.price_basis,
            prices=None,
        )
        state = outcome.state
        after = _closes_to_the_cent(state, None, f"{session.id} @ {bar}: after events")
        assert after - before == pytest.approx(sum(e.equity_delta for e in outcome.applied), abs=1e-6)
        assert state.cash - cash_before == pytest.approx(sum(e.cash_delta for e in outcome.applied), abs=1e-9)

        bar_fills = [_fill_from_row(row) for row in result.fills.to_dict("records") if str(row["bar"]) == bar]
        if bar_fills:
            before = state.carried_equity()
            cash_before = state.cash
            carried_marks = {sid: position.last_price for sid, position in state.positions.items()}
            carried_shares = {sid: position.shares for sid, position in state.positions.items()}
            applied = apply_fills(state, bar_fills)
            state = applied.state
            after = _closes_to_the_cent(state, None, f"{session.id} @ {bar}: after fills")
            # ``rebalance_conserves`` (formal/PrismFormal/Ledger.lean:40-43) under §3.6's Pi
            # says a rebalance **at fixed prices** changes wealth by exactly the charged
            # costs. Prices are not fixed here: a fill re-marks the whole position at the
            # executed price, so the shares already held move from the previous mark to the
            # fill price. That re-mark is priced separately and the residual is the cost.
            charged = sum(f.charged_cost for f in bar_fills)
            remark = 0.0
            for fill in bar_fills:
                held = float(carried_shares.get(fill.sid, 0.0))
                mark = carried_marks.get(fill.sid)
                if held and mark is not None:
                    remark += held * (fill.fill_price - float(mark))
            assert after - before == pytest.approx(remark - charged, abs=1e-6)
            assert state.cash - cash_before == pytest.approx(sum(f.cash_delta for f in bar_fills), abs=1e-9)

        before = state.carried_equity()
        cash_before = state.cash
        state, _ = expire_orders(state, bar=bar, phase="open", sids=traded)
        assert state.carried_equity() == pytest.approx(before, abs=1e-9), "an expiry moved equity"
        assert state.cash == pytest.approx(cash_before, abs=1e-9), "an expiry moved cash"

        marks_before = {
            sid: position.shares * (position.last_price or 0.0) for sid, position in state.positions.items()
        }
        before = state.carried_equity()
        state, _ = mark_prices(state, bar=bar, prices=open_row, stale_on_missing=False)
        after = _closes_to_the_cent(state, None, f"{session.id} @ {bar}: after the open mark")
        moved = sum(
            position.shares * (position.last_price or 0.0) - marks_before.get(sid, 0.0)
            for sid, position in state.positions.items()
        )
        assert after - before == pytest.approx(moved, abs=1e-6)

        marks_before = {
            sid: position.shares * (position.last_price or 0.0) for sid, position in state.positions.items()
        }
        before = state.carried_equity()
        cash_before = state.cash
        close_mark = mark_at_close(state, bar=bar, prices=close_row, cost_model=model)
        state = close_mark.state
        after = _closes_to_the_cent(state, close_row, f"{session.id} @ {bar}: after the close mark")
        moved = sum(
            position.shares * (position.last_price or 0.0) - marks_before.get(sid, 0.0)
            for sid, position in state.positions.items()
        )
        assert after - before == pytest.approx(moved - close_mark.accrual.holding_total, abs=1e-6)
        assert state.cash - cash_before == pytest.approx(close_mark.settled_cash, abs=1e-9)

        before = state.carried_equity()
        cash_before = state.cash
        state, _ = expire_orders(state, bar=bar, phase="close", sids=traded)
        for batch in result.decisions:
            if batch.decision_bar == bar and batch.intents:
                state = submit(state, batch.intents, submitted_at=bar)
        assert state.carried_equity() == pytest.approx(before, abs=1e-9), "deciding moved equity"
        assert state.cash == pytest.approx(cash_before, abs=1e-9), "deciding moved cash"

    session_end = state
    assert session_end.cash == pytest.approx(result.state.cash, abs=1e-9)


def _fill_from_row(row: Mapping[str, Any]) -> Fill:
    return Fill(
        execution_id=str(row["execution_id"]),
        client_order_id=str(row["client_order_id"]),
        sid=str(row["sid"]),
        ts=str(row["ts"]),
        qty=float(row["qty"]),
        fill_price=float(row["fill_price"]),
        reference_price=float(row["reference_price"]),
        commission=float(row["commission"]),
        spread_dollars=float(row["spread_dollars"]),
        impact_dollars=float(row["impact_dollars"]),
        source=str(row["source"]),
    )


def test_the_transition_replay_reproduces_the_simulated_book(session: Session) -> None:
    """An independent walk of the declared transition order ends at the simulator's book."""
    result = session.result
    model = CostModel(cost_stack(), CostInputs(as_of=session.decision_bar), equity=CAPITAL, horizon_bars=21)
    state = session.initial.copy()
    for bar in session.bars:
        traded = session.traded(bar)
        state = apply_events(
            state,
            bar=bar,
            events=session.events,
            adjustments=session.adjustments,
            price_basis=session.price_basis,
            prices=None,
        ).state
        bar_fills = [_fill_from_row(row) for row in result.fills.to_dict("records") if str(row["bar"]) == bar]
        if bar_fills:
            state = apply_fills(state, bar_fills).state
        state, _ = expire_orders(state, bar=bar, phase="open", sids=traded)
        state, _ = mark_prices(state, bar=bar, prices=session.price_row(session.opens, bar), stale_on_missing=False)
        state = mark_at_close(state, bar=bar, prices=session.price_row(session.closes, bar), cost_model=model).state
        state, _ = expire_orders(state, bar=bar, phase="close", sids=traded)
        for batch in result.decisions:
            if batch.decision_bar == bar and batch.intents:
                state = submit(state, batch.intents, submitted_at=bar)

    assert state.cash == pytest.approx(result.state.cash, abs=1e-9)
    assert sorted(state.sids()) == sorted(result.state.sids())
    for sid in state.sids():
        assert state.shares(sid) == pytest.approx(result.state.shares(sid), abs=1e-9), sid
        assert state.positions[sid].valuation_status == result.state.positions[sid].valuation_status, sid
    assert state.receivables == pytest.approx(result.state.receivables, abs=1e-9)
    assert state.liabilities == pytest.approx(result.state.liabilities, abs=1e-9)


def test_equity_moves_only_by_pnl_declared_flows_and_charged_costs(session: Session) -> None:
    """The equity path, rebuilt from shares, marks, fills, events and the cost rows alone."""
    result = session.result
    ledger = result.ledger
    shares = result.shares.fillna(0.0)
    marks = result.marks
    bars = list(ledger.index)
    rebuilt = [float(ledger["equity_close"].iloc[0])]
    factors = _share_factors(result)
    for previous, bar in zip(bars[:-1], bars[1:]):
        carried = 0.0
        for sid in shares.columns:
            held = float(shares.loc[previous, sid])
            if held == 0.0:
                continue
            before, after = float(marks.loc[previous, sid]), float(marks.loc[bar, sid])
            assert math.isfinite(before) and math.isfinite(after), f"{session.id}: {sid} lost its mark at {bar}"
            # A split or a rebase restates the basis both legs are counted in, and
            # ``share_factor * price_factor == 1`` by construction, so the carried PnL is
            # the restated share count against the new mark less the old count against the
            # old one. With no restatement the factor is 1 and this is ``held * (after -
            # before)``.
            factor = factors.get((bar, sid), 1.0)
            carried += held * factor * after - held * before
        traded_pnl = 0.0
        for fill in result.fills[result.fills["bar"] == bar].itertuples():
            traded_pnl += float(fill.qty) * (float(marks.loc[bar, fill.sid]) - float(fill.fill_price))
        flows = 0.0
        if not result.events.empty:
            flows = float(result.events.loc[result.events["effective_date"] == bar, "equity_delta"].sum())
        costs = float(ledger.loc[bar, "trade_cost"]) + float(ledger.loc[bar, "holding_cost"])
        rebuilt.append(rebuilt[-1] + carried + traded_pnl + flows - costs)
    expected = pd.Series(rebuilt, index=pd.Index(bars, name="bar"))
    worst = float((ledger["equity_close"] - expected).abs().max())
    assert worst < 0.01, f"{session.id}: the equity path is unexplained by {worst} dollars"


def _share_factors(result: SimulationResult) -> dict[tuple[str, str], float]:
    """``(bar, sid) -> `` the share factor the bar's events applied, defaulting to 1.0."""
    out: dict[tuple[str, str], float] = {}
    if result.events.empty:
        return out
    for row in result.events.to_dict("records"):
        detail = row["detail"] if isinstance(row["detail"], dict) else {}
        factor = float(detail.get("share_factor", 1.0))
        if factor == 1.0:
            continue
        key = (str(row["effective_date"]), str(row["sid"]))
        out[key] = out.get(key, 1.0) * factor
    return out


def test_cash_is_the_sum_of_its_recorded_movements(session: Session) -> None:
    """No unexplained cash term: fills, event cash and settled holding cost, added up."""
    result = session.result
    reconstructed = (
        session.initial.cash
        + float(result.fills["cash_delta"].sum())
        + (0.0 if result.events.empty else float(result.events["cash_delta"].sum()))
        - float(result.ledger["holding_cost"].sum())
    )
    assert result.state.cash == pytest.approx(reconstructed, abs=1e-9)


def test_the_charged_cost_components_sum_to_the_recorded_total(session: Session) -> None:
    costs = session.result.costs
    recomposed = (
        costs["commission"]
        + costs["spread"]
        + costs["impact_quadratic"]
        + costs["impact_participation"]
        + costs["borrow"]
        + costs["financing"]
        - costs["cash_income"]
    )
    pd.testing.assert_series_equal(costs["total"], recomposed, check_names=False, rtol=0.0, atol=1e-12)


def test_a_duplicate_fill_changes_the_ledger_not_at_all(session: Session) -> None:
    """§3.6's falsifiable assertion: a replayed fill moves nothing.

    A broker page re-read, a restarted loop, a settle path that ran twice -- each delivers
    the same execution id a second time, and applying it twice doubles the position and
    charges the cost twice. The execution id is the fills-ledger key and the idempotence
    key (§3.11), and it is passed in through ``known_executions`` because an order that
    filled completely has already left ``pending_orders`` and the state alone can no longer
    recognise the replay.
    """
    result = session.result
    if result.fills.empty:
        pytest.skip(f"{session.id} recorded no fill, so there is no duplicate to replay")
    model = CostModel(cost_stack(), CostInputs(as_of=session.decision_bar), equity=CAPITAL, horizon_bars=21)
    state = session.initial.copy()
    replayed: list[Fill] = []
    for bar in session.bars:
        state = apply_events(
            state,
            bar=bar,
            events=session.events,
            adjustments=session.adjustments,
            price_basis=session.price_basis,
            prices=None,
        ).state
        bar_fills = [_fill_from_row(row) for row in result.fills.to_dict("records") if str(row["bar"]) == bar]
        if bar_fills:
            state = apply_fills(state, bar_fills).state
            replayed.extend(bar_fills)
            before = state.copy()
            again = apply_fills(state, bar_fills, known_executions=[f.execution_id for f in replayed])
            assert again.duplicates == tuple(f.execution_id for f in bar_fills)
            assert again.applied == ()
            assert again.state.cash == pytest.approx(before.cash, abs=1e-12)
            for sid in before.sids():
                assert again.state.shares(sid) == pytest.approx(before.shares(sid), abs=1e-12), sid
            assert sorted(again.state.sids()) == sorted(before.sids())
        traded = session.traded(bar)
        state, _ = expire_orders(state, bar=bar, phase="open", sids=traded)
        state, _ = mark_prices(state, bar=bar, prices=session.price_row(session.opens, bar), stale_on_missing=False)
        state = mark_at_close(state, bar=bar, prices=session.price_row(session.closes, bar), cost_model=model).state
        state, _ = expire_orders(state, bar=bar, phase="close", sids=traded)
        for batch in result.decisions:
            if batch.decision_bar == bar and batch.intents:
                state = submit(state, batch.intents, submitted_at=bar)


# ---------------------------------------------------------------------------
# invariant 2: no fill without a submitted order and an available bar
# ---------------------------------------------------------------------------


def test_no_fill_exists_without_a_submitted_order_and_an_available_bar(session: Session) -> None:
    result = session.result
    decided = {f"{intent.decision_bar}:{intent.sid}": intent for batch in result.decisions for intent in batch.intents}
    for row in result.fills.to_dict("records"):
        sid, bar = str(row["sid"]), str(row["bar"])
        client_order_id = str(row["client_order_id"])
        intent = decided.get(client_order_id)
        assert intent is not None, f"{session.id}: fill {client_order_id} has no decided order"
        assert intent.sid == sid
        # Next-open causality: decided at a close, filled no earlier than the next open.
        assert intent.decision_bar < bar, f"{session.id}: {client_order_id} filled on its own decision bar"
        assert sid in session.traded(bar), f"{session.id}: {sid} filled on {bar}, a bar it did not trade"
        # "Never at a price the bar did not print", with the one declared exception: a bar
        # that traded but printed no open fills at the intent's own reference price under
        # ``missing_open_policy = "reference_price"``. Nothing else is a legal fill price.
        printed = float(session.price_row(session.opens, bar).get(sid, np.nan))
        legal = printed if math.isfinite(printed) and printed > 0.0 else intent.reference_price
        assert float(row["fill_price"]) == pytest.approx(
            legal, abs=1e-9
        ), f"{session.id}: {sid} filled at {row['fill_price']}, a price the bar did not print"
        assert float(row["reference_price"]) == pytest.approx(intent.reference_price, abs=1e-12)
        assert abs(float(row["qty"])) <= abs(intent.qty) * (1.0 + 1e-9)
        assert float(row["qty"]) * intent.qty > 0.0, "a fill took the opposite side of its order"
        assert float(row["commission"]) >= 0.0


def test_no_position_moved_on_a_bar_without_a_fill_or_an_event(session: Session) -> None:
    """Shares move for exactly two reasons: a fill, or an event that says they do."""
    result = session.result
    shares = result.shares.fillna(0.0)
    opening = {sid: session.initial.shares(sid) for sid in session.initial.sids()}
    previous = pd.Series({sid: float(opening.get(sid, 0.0)) for sid in shares.columns}, dtype=float)
    for bar in session.bars:
        filled = set(result.fills.loc[result.fills["bar"] == bar, "sid"].astype(str))
        touched: set[str] = set()
        if not result.events.empty:
            # Only the four transitions that are *declared* to move a share count excuse a
            # move: a rebase or a split restating the basis, a terminal payout clearing the
            # instrument, a cancellation writing shares off. A membership change, a halt or
            # a distribution moving the share count would be a defect, so they must not be
            # waved through here.
            for row in result.events.loc[result.events["effective_date"] == bar].to_dict("records"):
                detail = row["detail"] if isinstance(row["detail"], dict) else {}
                moves = float(detail.get("share_factor", 1.0)) != 1.0 or str(row["event_type"]) in (
                    "terminal_payout",
                    "share_cancellation",
                )
                if moves:
                    touched.add(str(row["sid"]))
        for sid in shares.columns:
            if sid in filled or sid in touched:
                continue
            assert float(shares.loc[bar, sid]) == pytest.approx(
                float(previous[sid]), abs=1e-9
            ), f"{session.id}: {sid} moved on {bar} with neither a fill nor an event"
        previous = shares.loc[bar]


# ---------------------------------------------------------------------------
# invariant 3: no order the §3.7 action restrictions forbid
# ---------------------------------------------------------------------------


def test_no_order_the_action_restrictions_forbid(session: Session) -> None:
    for batch in session.result.decisions:
        quantities = batch.quantities.set_index("sid")
        for intent in batch.intents:
            rules = session.restrictions.get(intent.sid, frozenset())
            assert not restriction_blocks_all_trading(
                rules
            ), f"{session.id}: an order on {intent.sid}, which {sorted(rules)} bars from trading"
            assert "no_trade" not in rules, (
                f"{session.id}: an order on {intent.sid}, which could not be priced or had no volume at the "
                "decision close; the participation restriction of §3.7 is PHYSICAL and never waived"
            )
            row = quantities.loc[intent.sid]
            held = float(row["held_shares"]) + float(row["outstanding_shares"])
            action = classify_action(held, intent.qty)
            if restriction_blocks_new_exposure(rules):
                assert (
                    action == "C1"
                ), f"{session.id}: {intent.sid} is reduce-only under {sorted(rules)} and the order is a {action}"
            if "no_short" in rules:
                assert held + intent.qty >= -1e-12, f"{session.id}: {intent.sid} is not shortable"
            step = session.increments[intent.sid]
            assert step > 0.0
            assert (
                abs(intent.qty / step - round(intent.qty / step)) < 1e-9
            ), f"{session.id}: {intent.sid} order of {intent.qty} is not a multiple of the dated increment {step}"
            assert abs(intent.notional) >= min(session.min_notionals.values()) - 1e-9
            assert intent.order_type == "market_on_open"
            assert intent.time_in_force == "opg"
            assert intent.decision_bar == session.decision_bar


def test_the_restriction_mask_is_load_bearing_and_not_decorative(sessions: dict[tuple[str, str], Session]) -> None:
    """The mask removes a trade the constructor actually wanted, in at least one session.

    Without this, :func:`test_no_order_the_action_restrictions_forbid` would be satisfied by
    a mask that never removed anything and by an engine that never proposed anything. Here
    the *unmasked* target the heuristic constructor produced is compared with the masked one
    on the restricted names, and the set is required to contain a session where the two
    differ -- that is, where the restriction stopped a trade the decision wanted to make.
    """
    bitten: list[str] = []
    for session in sessions.values():
        for sid, rules in session.restrictions.items():
            if not rules:
                continue
            wanted = float(session.assembled.get(sid, np.nan))
            got = float(session.targets.loc[session.decision_bar, sid])
            if math.isfinite(wanted) and not math.isfinite(got):
                bitten.append(f"{session.id}:{sid}")
            elif math.isfinite(wanted) and math.isfinite(got) and wanted != got:
                bitten.append(f"{session.id}:{sid}")
    assert bitten, "no restriction removed any trade the constructor wanted; the mask is untested"
    assert any(entry.startswith("delisting/") for entry in bitten)
    assert any(entry.startswith("ticker_reuse/") for entry in bitten)


def test_the_quantity_increment_the_order_used_is_the_one_dated_at_the_decision(session: Session) -> None:
    """A dated capability, read at the decision and never assumed current (§3.2)."""
    for sid in session.universe:
        assert session.increments[sid] == float(session.capabilities.loc[sid, "quantity_increment"])


# ---------------------------------------------------------------------------
# invariant 4: a missing bar leaves the holding stale and never at zero
# ---------------------------------------------------------------------------


def test_a_missing_bar_leaves_the_holding_stale_and_never_at_zero(session: Session) -> None:
    result = session.result
    shares = result.shares
    marks = result.marks
    for bar in session.bars:
        traded = session.traded(bar)
        for sid in shares.columns:
            held = shares.loc[bar, sid]
            if not math.isfinite(float(held)) or float(held) == 0.0:
                continue
            if sid in traded:
                continue
            mark = float(marks.loc[bar, sid])
            assert math.isfinite(mark) and mark > 0.0, (
                f"{session.id}: {sid} did not print on {bar} and is marked at {mark}. A missing bar never erases a "
                "holding and never marks it at zero (SPEC N7)"
            )
    final = session.fill_bar
    traded = session.traded(final)
    for sid, position in result.state.positions.items():
        if sid in traded:
            continue
        assert position.valuation_status in (
            "stale",
            "unresolved_claim",
        ), f"{session.id}: {sid} did not print on {final} and is carried {position.valuation_status!r}"
        assert position.last_price is not None and position.last_price > 0.0
        assert position.last_price_bar is not None and position.last_price_bar < final


def test_a_holding_is_never_sold_by_the_absence_of_a_bar(session: Session) -> None:
    """The affected name is still in the book unless a fill or an event removed it."""
    result = session.result
    if session.start_state == "flat":
        # Nothing to lose, and the claim is about a holding. Asserted rather than returned
        # silently, so the branch states what makes it inapplicable.
        assert session.initial.positions == {}
        return
    sid = session.affected
    removed_by_event = False
    if not result.events.empty:
        rows = result.events.loc[result.events["sid"] == sid]
        removed_by_event = any(
            str(row["event_type"]) in ("terminal_payout", "share_cancellation") for row in rows.to_dict("records")
        )
    sold = float(result.fills.loc[result.fills["sid"] == sid, "qty"].sum()) if len(result.fills) else 0.0
    opening = session.initial.shares(sid)
    if removed_by_event:
        # A declared terminal event may remove the position; the absence of a bar may not.
        # Which of the two happened is the whole distinction, so it is checked, not assumed.
        assert opening != 0.0
        return
    ending = result.state.shares(sid)
    assert ending == pytest.approx(
        _expected_shares(session, sid, opening, sold), abs=1e-6
    ), f"{session.id}: {sid} ended at {ending} shares, which is neither the opening book nor the fills applied to it"


def _expected_shares(session: Session, sid: str, opening: float, filled: float) -> float:
    """Opening shares, rebased by any share factor the window applied, plus the fills."""
    factor = 1.0
    if not session.result.events.empty:
        for row in session.result.events.to_dict("records"):
            if str(row["sid"]) != sid:
                continue
            detail = row["detail"] if isinstance(row["detail"], dict) else {}
            factor *= float(detail.get("share_factor", 1.0))
    return opening * factor + filled


# ---------------------------------------------------------------------------
# invariant 5: every event applied is recorded with its effective date and source revision
# ---------------------------------------------------------------------------


def test_every_applied_event_is_recorded_with_its_effective_date_and_source_revision(session: Session) -> None:
    applied = session.result.events
    for row in applied.to_dict("records"):
        effective = str(row["effective_date"])
        assert effective in session.bars, f"{session.id}: an event was applied outside the window at {effective}"
        if not str(row["sid"]):
            # The only sid-less row is the monthly holding-cost settlement, which is a
            # transition of this engine's own accrual and not a row of the events table.
            assert str(row["event_type"]) == "holding_cost_settlement"
            continue
        assert row["source"], f"{session.id}: {row['event_type']} on {row['sid']} carries no source"
        assert int(row["revision"]) >= 1, f"{session.id}: {row['event_type']} carries no source revision"


def test_every_visible_event_in_the_window_was_applied(session: Session) -> None:
    """Nothing visible is dropped: a dropped corporate action is a silent loss."""
    if session.events.empty:
        return
    applied = session.result.events
    seen = {
        (str(row["sid"]), str(row["event_type"]), str(row["effective_date"])) for row in applied.to_dict("records")
    }
    for row in session.events.to_dict("records"):
        sid, kind, effective = str(row["sid"]), str(row["event_type"]), str(row["effective_date"])
        if effective not in session.bars:
            continue
        assert (sid, kind, effective) in seen, f"{session.id}: the {kind} on {sid} at {effective} was never applied"
        recorded = applied.loc[
            (applied["sid"] == sid) & (applied["event_type"] == kind) & (applied["effective_date"] == effective)
        ]
        assert str(recorded.iloc[0]["source"]) == str(row["source"])
        assert int(recorded.iloc[0]["revision"]) == int(row["revision"])


def test_every_applied_adjustment_is_recorded_with_its_source_revision(session: Session) -> None:
    """A ``rebase`` row is a recorded transition, not an invisible restatement."""
    if session.adjustments.empty:
        return
    applied = session.result.events
    for row in session.adjustments.to_dict("records"):
        if str(row["kind"]) != "rebase" or str(row["effective_date"]) not in session.bars:
            continue
        recorded = applied.loc[(applied["sid"] == str(row["sid"])) & (applied["event_type"] == "rebase")]
        assert len(recorded) == 1, f"{session.id}: the rebase on {row['sid']} was not recorded"
        assert str(recorded.iloc[0]["source"]) == str(row["source"])
        assert int(recorded.iloc[0]["revision"]) == int(row["revision"])
        assert float(recorded.iloc[0]["equity_delta"]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# the three defects this set found, each pinned by name
# ---------------------------------------------------------------------------


def test_a_vendor_back_rewrite_rebases_shares_and_leaves_equity_continuous(
    sessions: dict[tuple[str, str], Session],
) -> None:
    """Decision D3's back-rewrite: prices and shares restate in one transaction (§3.6).

    Before the fix ``apply_events`` read only the ``events`` table, so the 4-for-1 rewrite
    of the ``rebase`` scenario restated the prices and nothing restated the shares: the
    long book lost 7.2% of its equity in one bar and the short book gained 7.1%, with no
    event, no fill and no cost row saying why. The assertion is equity continuity across
    the effective date, at the ordinary size of a day's move.
    """
    for start_state in ("long_only", "long_short"):
        session = sessions[("rebase", start_state)]
        sid = session.affected
        factor = float(session.case.number("factor"))
        assert session.case.date("effective") == session.fill_bar

        rebases = session.result.events.loc[session.result.events["event_type"] == "rebase"]
        assert len(rebases) == 1, "the rebase adjustment was not applied"
        assert str(rebases.iloc[0]["sid"]) == sid
        assert float(rebases.iloc[0]["detail"]["share_factor"]) == pytest.approx(factor)

        ledger = session.result.ledger
        before = float(ledger.loc[session.decision_bar, "equity_close"])
        after = float(ledger.loc[session.fill_bar, "equity_close"])
        assert abs(after / before - 1.0) < 0.02, (
            f"rebase/{start_state}: equity moved {after / before - 1.0:+.2%} across the rewrite. A vendor "
            "back-rewrite must move no equity beyond the bar's own return"
        )
        # And the shares themselves were restated, not merely the equity total.
        opening = session.initial.shares(sid)
        filled = float(session.result.fills.loc[session.result.fills["sid"] == sid, "qty"].sum())
        assert session.result.state.shares(sid) == pytest.approx(opening * factor + filled, abs=1e-6)


def test_a_split_cancels_the_orders_resting_on_the_name_even_when_it_is_not_held(
    sessions: dict[tuple[str, str], Session],
) -> None:
    """The flat book's decided buy does not survive a split on a ``raw`` series.

    ``raw`` prices halve on the effective date. An order decided at the pre-split close
    carries a pre-split share count, so leaving it alive fills the old quantity at the new
    price and acquires half the decided notional -- an error that conserves cash and shares
    and is therefore invisible to a conservation test. Before the fix the cancellation was
    inside the ``position is not None`` branch, so only a *held* name was protected.
    """
    session = sessions[("split", "flat")]
    raw = session.case.sid("raw")
    assert session.price_basis[raw] == "raw"
    assert session.case.date("effective") == session.fill_bar

    decided = {intent.sid for batch in session.result.decisions for intent in batch.intents}
    assert raw in decided, "the fixture must decide an order on the raw-split name for this to bite"
    filled = session.result.fills.loc[session.result.fills["sid"] == raw]
    assert filled.empty, "a pre-split quantity filled at the post-split open"
    assert raw not in {order.sid for order in session.result.state.pending_orders}
    assert raw not in session.result.state.positions

    split_rows = session.result.events.loc[
        (session.result.events["sid"] == raw) & (session.result.events["event_type"] == "split")
    ]
    assert len(split_rows) == 1
    assert split_rows.iloc[0]["detail"]["canceled_orders"], "the split recorded no cancellation"


def test_a_split_against_each_price_basis_moves_no_equity(sessions: dict[tuple[str, str], Session]) -> None:
    """The raw series rebases shares; the adjusted one does not; neither moves equity (D3)."""
    session = sessions[("split", "long_only")]
    raw, adjusted = session.case.sid("raw"), session.case.sid("adjusted")
    assert session.price_basis[raw] == "raw"
    assert session.price_basis[adjusted] == "split_adjusted_at_ingest"
    events = session.result.events
    splits = events.loc[events["event_type"] == "split"].set_index("sid")
    assert float(splits.loc[raw, "detail"]["share_factor"]) == pytest.approx(2.0)
    assert float(splits.loc[adjusted, "detail"]["share_factor"]) == pytest.approx(1.0)
    assert float(splits.loc[raw, "equity_delta"]) == pytest.approx(0.0, abs=1e-9)
    assert float(splits.loc[adjusted, "equity_delta"]) == pytest.approx(0.0, abs=1e-9)


def test_a_dead_instrument_carries_no_resting_order() -> None:
    """A cessation, a terminal payout and a share cancellation cancel the resting orders.

    Found by the ``delisting`` and ``ticker_reuse`` sessions, whose books ended carrying
    orders on instruments that had stopped printing. Such an order can never fill and never
    expires -- :func:`prism.execution.accounting.expire_orders` only expires orders on names
    that *traded* -- so it rests for ever and ``committed_cash`` reserves its notional
    against buying power permanently. This is asserted directly on the primitives, because
    in a session the event lands at the start of the bar and the decision is taken at its
    close, so the driver's restriction mask stops the order being decided at all.
    """
    bars = ("2026-03-02", "2026-03-03")
    state = state_from_shares(bars[0], {"AAA": 100.0}, pd.Series({"AAA": 10.0}), cash=1_000.0)
    intent = OrderIntent(
        sid="AAA",
        qty=-40.0,
        increment=1.0,
        reference_price=10.0,
        decision_bar=bars[0],
        order_type="market_on_open",
        time_in_force="opg",
        action_class="C1",
    )
    submitted = submit(state, [intent], submitted_at=bars[0])
    assert len(submitted.pending_orders) == 1

    outcome = apply_events(
        submitted,
        bar=bars[1],
        events=[
            {
                "sid": "AAA",
                "event_type": "trading_cessation",
                "effective_date": bars[1],
                "payload": {"last_session": bars[0]},
                "ingested_at": f"{bars[1]}T20:00:00-04:00",
                "source": "test",
                "revision": 1,
            }
        ],
    )
    assert outcome.state.pending_orders == [], "an order rests on an instrument that stopped trading"
    assert outcome.state.committed_cash() == 0.0
    assert outcome.applied[0].detail["canceled_orders"] == [f"{bars[0]}:AAA"]
    assert outcome.state.positions["AAA"].valuation_status == "unresolved_claim"
    assert outcome.state.positions["AAA"].shares == 100.0, "the cessation must not sell the holding"


def test_a_terminal_payout_clears_the_position_and_its_resting_orders() -> None:
    bars = ("2026-03-02", "2026-03-03")
    state = state_from_shares(bars[0], {"AAA": 100.0}, pd.Series({"AAA": 10.0}), cash=1_000.0)
    intent = OrderIntent(
        sid="AAA",
        qty=25.0,
        increment=1.0,
        reference_price=10.0,
        decision_bar=bars[0],
        order_type="market_on_open",
        time_in_force="opg",
        action_class="C3",
    )
    submitted = submit(state, [intent], submitted_at=bars[0])
    assert submitted.committed_cash() == pytest.approx(250.0)
    outcome = apply_events(
        submitted,
        bar=bars[1],
        events=[
            {
                "sid": "AAA",
                "event_type": "terminal_payout",
                "effective_date": bars[1],
                "payload": {"amount_per_share": 11.0, "currency": "USD", "pay_date": bars[1]},
                "ingested_at": f"{bars[1]}T20:00:00-04:00",
                "source": "test",
                "revision": 1,
            }
        ],
    )
    assert "AAA" not in outcome.state.positions
    assert outcome.state.pending_orders == []
    assert outcome.state.committed_cash() == 0.0
    assert outcome.state.cash == pytest.approx(1_000.0 + 1_100.0)


# ---------------------------------------------------------------------------
# per-scenario expectations, on top of the five invariants
# ---------------------------------------------------------------------------


def test_a_halt_carries_the_holding_stale_and_leaves_the_order_pending(
    sessions: dict[tuple[str, str], Session],
) -> None:
    """§3.6: the halted name is priced at its last bar, ``stale``, and nothing fills."""
    for start_state in ("long_only", "long_short"):
        session = sessions[("halt", start_state)]
        sid = session.affected
        assert session.case.date("halt_start") == session.fill_bar
        position = session.result.state.positions[sid]
        assert position.valuation_status == "stale"
        assert position.last_price_bar == session.case.date("last_priced")
        assert position.shares == pytest.approx(session.initial.shares(sid))
        assert session.result.fills.loc[session.result.fills["sid"] == sid].empty
        resting = [order for order in session.result.state.pending_orders if order.sid == sid]
        assert len(resting) == 1, "a halted name's order neither fills nor expires; it rests (§3.6)"
        assert resting[0].status == "pending"


def test_a_short_pays_the_distribution_it_is_entitled_against(sessions: dict[tuple[str, str], Session]) -> None:
    """Entitlement is the signed share count at the close before the ex-date (§3.6)."""
    session = sessions[("distribution", "long_short")]
    sid = session.affected
    assert session.case.date("ex_date") == session.fill_bar
    amount = session.case.number("amount_per_share")
    held = session.initial.shares(sid)
    assert held < 0.0
    events = session.result.events
    row = events.loc[(events["sid"] == sid) & (events["event_type"] == "cash_distribution")].iloc[0]
    assert row["detail"]["leg"] == "liability", "a short owes the distribution; it does not receive it"
    assert float(row["detail"]["shares"]) == pytest.approx(held), "entitlement is the prior close's share count"
    assert float(row["detail"]["amount"]) == pytest.approx(held * amount)
    assert float(row["cash_delta"]) == 0.0, "nothing moves in cash on the ex-date; the pay date settles it"
    key = f"cash_distribution:{sid}:{session.fill_bar}"
    assert session.result.state.liabilities[key] == pytest.approx(abs(held * amount))


def test_the_long_book_books_the_distribution_as_a_receivable_not_as_cash(
    sessions: dict[tuple[str, str], Session],
) -> None:
    session = sessions[("distribution", "long_only")]
    sid = session.affected
    events = session.result.events
    row = events.loc[(events["sid"] == sid) & (events["event_type"] == "cash_distribution")].iloc[0]
    assert row["detail"]["leg"] == "receivable"
    assert row["detail"]["pay_date"] == session.case.date("pay_date")
    assert float(row["cash_delta"]) == 0.0
    key = f"cash_distribution:{sid}:{session.fill_bar}"
    expected = session.initial.shares(sid) * session.case.number("amount_per_share")
    assert session.result.state.receivables[key] == pytest.approx(expected)


def test_a_delisting_with_an_unknown_terminal_value_makes_the_record_incomplete(
    sessions: dict[tuple[str, str], Session],
) -> None:
    """A strict claim of unknown value flags the state; it is never written off (§3.2)."""
    for start_state in ("long_only", "long_short"):
        session = sessions[("delisting", start_state)]
        sid = session.affected
        assert sid == session.case.sid("unknown")
        position = session.result.state.positions[sid]
        assert position.valuation_status == "unresolved_claim"
        assert position.shares == pytest.approx(session.initial.shares(sid))
        # Marked at its own last close, which is the decision bar's -- not at the *other*
        # delisting sid's, which is what ``case.value("last_close")`` records.
        assert position.last_price == pytest.approx(float(session.closes.loc[session.decision_bar, sid]), rel=1e-9)
        assert session.result.state.is_incomplete
        assert session.result.ledger.loc[session.fill_bar, "incomplete"]


def test_a_reused_ticker_never_merges_two_holdings(sessions: dict[tuple[str, str], Session]) -> None:
    """One ticker, two sids, two books; the first's holding survives the handover."""
    session = sessions[("ticker_reuse", "long_only")]
    first, second = session.case.sid("first"), session.case.sid("second")
    assert session.case.date("reopen") == session.fill_bar
    assert first in session.result.state.positions
    assert session.result.state.shares(first) == pytest.approx(session.initial.shares(first))
    assert session.result.state.positions[first].valuation_status == "unresolved_claim"
    assert session.result.state.shares(second) == 0.0 or second not in session.result.state.positions


def test_a_null_valued_full_revision_leaves_the_holding_stale_and_fills_nothing(
    sessions: dict[tuple[str, str], Session],
) -> None:
    """The withdrawn bar means null, not the previous value; nothing fills on a bar that is gone."""
    for start_state in ("long_only", "long_short"):
        session = sessions[("revision", start_state)]
        sid = session.affected
        assert session.case.date("revised_bar") == session.fill_bar
        assert not math.isfinite(float(session.closes.loc[session.fill_bar, sid]))
        position = session.result.state.positions[sid]
        assert position.valuation_status == "stale"
        assert position.last_price_bar == session.decision_bar
        assert position.last_price == pytest.approx(float(session.closes.loc[session.decision_bar, sid]))
        assert session.result.fills.loc[session.result.fills["sid"] == sid].empty


def test_a_multi_session_gap_carries_the_holding_and_does_not_flag_the_order(
    sessions: dict[tuple[str, str], Session],
) -> None:
    """The recovery scenario's gap opens on the fill bar; the holding is carried, not sold."""
    for start_state in ("long_only", "long_short"):
        session = sessions[("recovery", start_state)]
        sid = session.affected
        assert session.case.date("gap_start") == session.fill_bar
        position = session.result.state.positions[sid]
        assert position.valuation_status == "stale"
        assert position.shares == pytest.approx(session.initial.shares(sid))
        assert position.last_price_bar == session.case.date("last_priced")


def test_a_removed_index_member_is_never_sold_by_the_data_layer(sessions: dict[tuple[str, str], Session]) -> None:
    session = sessions[("membership", "long_only")]
    sid = session.affected
    assert sid == session.case.sid("removed")
    opening = session.initial.shares(sid)
    filled = float(session.result.fills.loc[session.result.fills["sid"] == sid, "qty"].sum())
    assert session.result.state.shares(sid) == pytest.approx(opening + filled, abs=1e-9)


def test_the_thin_name_is_charged_and_bounded_rather_than_traded_freely(
    sessions: dict[tuple[str, str], Session],
) -> None:
    """The ``costs`` scenario: four dollar-volume levels, all of them charged."""
    session = sessions[("costs", "flat")]
    costs = session.result.costs
    assert float(costs["commission"].sum()) > 0.0
    assert float(costs["spread"].sum()) > 0.0
    assert float(costs["impact_quadratic"].sum()) > 0.0
    assert float(costs["total"].sum()) > 0.0
    charged = float(session.result.fills[["commission", "spread_dollars", "impact_dollars"]].sum().sum())
    recorded = float(costs[["commission", "spread", "impact_quadratic", "impact_participation"]].sum().sum())
    assert charged == pytest.approx(recorded, abs=1e-9)


def test_the_short_book_accrues_borrow_on_the_marked_short_value(sessions: dict[tuple[str, str], Session]) -> None:
    """Borrow accrues on short market value as marked, not on entry notional (§3.6)."""
    session = sessions[("costs", "long_short")]
    stack = cost_stack()
    per_bar = stack.borrow_bps_annual / 1e4 / stack.trading_days_per_year
    ledger, costs = session.result.ledger, session.result.costs
    assert float(costs["borrow"].sum()) > 0.0
    charged_on_a_short = False
    for bar in session.bars:
        shares = session.result.shares.loc[bar].fillna(0.0)
        marks = session.result.marks.loc[bar]
        collateral = sum(-float(shares[sid]) * float(marks[sid]) for sid in shares.index if float(shares[sid]) < 0.0)
        assert float(costs.loc[bar, "borrow"]) == pytest.approx(per_bar * collateral, abs=1e-6), bar
        assert float(ledger.loc[bar, "holding_cost"]) >= 0.0
        charged_on_a_short = charged_on_a_short or collateral > 0.0
    assert charged_on_a_short, "no bar carried a short, so the borrow accrual was never tested against one"


def test_which_action_restrictions_the_set_actually_exercises(sessions: dict[tuple[str, str], Session]) -> None:
    """State the set's restriction coverage rather than letting an untested branch pass.

    The three-bar windows are fixed by the scenario catalog, and each scenario's event
    lands on the *fill* bar, so the only restriction that binds at a decision close in this
    set is the ``PHYSICAL`` one -- a name the decision cannot price or whose bar carried no
    volume. The halt begins after the decision it could restrict; the ``recovery``
    scenario's broker restriction is dated nineteen sessions past its window. Those two are
    covered by :func:`test_a_dated_broker_restriction_makes_the_name_reduce_only` and
    :func:`test_a_halt_carries_the_holding_stale_and_leaves_the_order_pending` instead, and
    this test fails if that ever silently changes.
    """
    fired: set[str] = set()
    restricted_names = 0
    for session in sessions.values():
        for rules in session.restrictions.values():
            fired |= set(rules)
            restricted_names += 1 if rules else 0
    assert fired == {"no_trade"}, f"the set now exercises {sorted(fired)}; widen or narrow this claim"
    assert restricted_names >= 7, "no name was restricted at any decision close; the mask is not being exercised"


def test_a_dated_broker_restriction_makes_the_name_reduce_only(dataset: tuple[Dataset, Manifest]) -> None:
    """The ``recovery`` scenario's ``no_new_exposure`` row, read on the date it is dated at.

    §3.7's reduce-only action restriction, and the mask the driver applies for it. It is
    tested here rather than inside a session because the row is dated at ``gap_end``, which
    the scenario's own three-bar window does not reach.
    """
    data, manifest = dataset
    case = scenario_catalog(manifest)["recovery"]
    sid = case.sid("gapped")
    view = data.view("research_vintage", case.date("gap_end"), vintage=VINTAGE)
    row = view.capabilities([sid]).loc[sid]
    rules = _capability_restrictions(row)
    assert rules == {"no_new_exposure"}
    assert not bool(row["shortable"])
    assert restriction_blocks_new_exposure(rules)
    assert not restriction_blocks_all_trading(rules)

    restrictions = {sid: frozenset(rules | {"no_short"})}
    held = pd.Series({sid: 0.06}, dtype=float)
    # An order that would add to the holding is refused and becomes a full exit; a strict
    # reduction on the same side survives; a short is refused outright.
    assert _mask_target(pd.Series({sid: 0.09}), held_weights=held, restrictions=restrictions)[sid] == 0.0
    assert _mask_target(pd.Series({sid: 0.02}), held_weights=held, restrictions=restrictions)[sid] == 0.02
    assert _mask_target(pd.Series({sid: -0.04}), held_weights=held, restrictions=restrictions)[sid] == 0.0
    flat = pd.Series(dtype=float)
    assert _mask_target(pd.Series({sid: 0.05}), held_weights=flat, restrictions=restrictions)[sid] == 0.0
    blocked = {sid: frozenset({"halted"})}
    assert math.isnan(float(_mask_target(pd.Series({sid: 0.05}), held_weights=held, restrictions=blocked)[sid]))


def test_the_dated_quantity_increment_changes_on_its_own_date(dataset: tuple[Dataset, Manifest]) -> None:
    """The ``rounding`` scenario's increment is dated, and the session used the dated value."""
    data, manifest = dataset
    case = scenario_catalog(manifest)["rounding"]
    sid = case.sid("rounded")
    change = case.date("increment_change")
    before = data.view("research_vintage", case.decision_bar, vintage=VINTAGE).capabilities([sid])
    on = data.view("research_vintage", change, vintage=VINTAGE).capabilities([sid])
    assert float(before.loc[sid, "quantity_increment"]) == pytest.approx(float(case.value("increment_before")))
    assert float(on.loc[sid, "quantity_increment"]) == pytest.approx(float(case.value("increment_after")))
    assert case.decision_bar < change
