"""Decision assembly (engine specification §3.8).

The construction and sizing in ``prism.assembly`` were moved out of
``live/daily.py`` and ``live/loop.py`` unchanged, so the test that matters most
is the one that catches a *silent* change to the live book: the goldens below
are ``float.hex()`` literals produced by the pre-extraction code on a
closed-form panel, for all three recorded books, with the online band, the
participation gate and the unrankable mask all active. They are bit-exact by
construction; regenerating them is only legitimate if a change to the live
book's weights has been authorised.

The rest pins the seams the move created: assembly sizes intents only against a
reconciled position book, a liquidate instruction bypasses the band and the
dust floor, and the declared step order actually changes where the cap runs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from prism.assembly import (
    AccountState,
    AssemblyError,
    DecisionContext,
    assemble,
    construct_targets,
    heuristic_book_rule,
    size_orders,
)
from prism.composition import ComponentPlan
from prism.live.daily import DailyBookConfig, _construct_refresh_targets
from prism.signal.base import Signal

N_BARS = 40
SYMBOLS = [f"S{i:02d}" for i in range(12)]
SCORES = [1.9, -0.4, 0.7, float("nan"), 2.4, -1.8, 0.15, -0.95, 1.1, -2.2, 0.35, -0.05]
HELD = {"S00": 0.08, "S01": -0.06, "S02": 0.03, "S05": -0.02, "S09": 0.05, "S11": 0.01}
EQUITY = 1_000_000.0
MASK = ["S02", "S09"]  # both are held, so the mask's "hold, don't flatten" shows

BOOKS = {
    "decile_neutral": dict(
        book="decile_neutral",
        decile=0.25,
        max_gross=1.0,
        max_symbol_abs_weight=0.30,
        no_trade_band=0.004,
        max_participation=0.02,
        adv_window_bars=20,
    ),
    "inverse_vol": dict(
        book="inverse_vol",
        vol_ewma_bars=10,
        max_gross=0.8,
        max_symbol_abs_weight=0.25,
        no_trade_band=0.002,
        max_participation=None,
    ),
    "directional": dict(
        book="directional",
        position_size=0.05,
        max_gross=1.0,
        max_symbol_abs_weight=0.10,
        no_trade_band=0.0,
        max_participation=None,
    ),
}

#: Pre-extraction targets, hex-exact (see the module docstring).
GOLDEN_TARGETS = {
    "decile_neutral": {
        "S00": "0x1.9c3b2ba539868p-4",
        "S01": "-0x1.13a5cdaedb37cp-6",
        "S02": "0x1.eb851eb851eb8p-6",
        "S03": "0x0.0p+0",
        "S04": "0x1.e96fb73e61b33p-4",
        "S05": "-0x1.59465f169ec76p-3",
        "S06": "0x0.0p+0",
        "S07": "-0x1.b68045b6088f6p-3",
        "S08": "0x0.0p+0",
        "S09": "0x1.999999999999ap-5",
        "S10": "0x0.0p+0",
        "S11": "0x0.0p+0",
    },
    "inverse_vol": {
        "S00": "0x1.62e9cf066d513p-4",
        "S01": "-0x1.7b5a781f4bf08p-4",
        "S02": "0x1.eb851eb851eb8p-6",
        "S03": "0x0.0p+0",
        "S04": "0x1.7ca7640f8cc85p-4",
        "S05": "-0x1.5bb0eb897aba1p-4",
        "S06": "0x1.588b2d43d3280p-4",
        "S07": "-0x1.7ba90a5dceecep-4",
        "S08": "0x1.61e839a7fa8c0p-4",
        "S09": "0x1.999999999999ap-5",
        "S10": "0x1.790a2c51e50ccp-4",
        "S11": "-0x1.670998728a5b2p-4",
    },
    "directional": {
        "S00": "0x1.999999999999ap-5",
        "S01": "-0x1.47ae147ae147cp-6",
        "S02": "0x1.eb851eb851eb8p-6",
        "S03": "0x0.0p+0",
        "S04": "0x1.999999999999ap-5",
        "S05": "-0x1.999999999999ap-5",
        "S06": "0x1.eb851eb851eb8p-8",
        "S07": "-0x1.851eb851eb852p-5",
        "S08": "0x1.999999999999ap-5",
        "S09": "0x1.999999999999ap-5",
        "S10": "0x1.1eb851eb851ebp-6",
        "S11": "-0x1.47ae147ae147cp-9",
    },
}


class Node(Signal):
    """A fixed cross-section — the panel drives the vol and the ADV, not the score."""

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
        return 5

    def fit(self, close, volume=None) -> "Node":
        return self

    def score(self, close, volume=None) -> pd.Series:
        return pd.Series(SCORES, index=close.columns, dtype=float)


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A closed-form panel: no RNG, so the goldens are reproducible anywhere."""
    index = pd.date_range("2026-01-02", periods=N_BARS, freq="B", tz="America/New_York")
    t = np.arange(N_BARS, dtype=float).reshape(-1, 1)
    i = np.arange(len(SYMBOLS), dtype=float).reshape(1, -1)
    close = (50.0 + 2.0 * i + 0.05 * t * (1.0 + 0.1 * i)) * (1.0 + 0.01 * np.sin(t + i))
    volume = 2.0e4 * (1.0 + i) * (1.0 + 0.1 * np.cos(t - i))
    return (
        pd.DataFrame(close, index=index, columns=SYMBOLS),
        pd.DataFrame(volume, index=index, columns=SYMBOLS),
    )


def _config(book: str, **overrides) -> DailyBookConfig:
    settings = dict(min_order_notional=1.0, whole_shares=True, decision_every=21)
    settings.update(BOOKS[book])
    settings.update(overrides)
    return DailyBookConfig(**settings)


def _context(config: DailyBookConfig, close, volume, *, positions=None, **overrides) -> DecisionContext:
    plan = ComponentPlan(
        component=Node(),
        component_id=config.book,
        to_weights=config.book,
        weights_fn=heuristic_book_rule(config, close, config.max_gross),
    )
    settings: dict = dict(
        decision_bar=str(close.index[-1].date()),
        components=(plan,),
        scoring_panels={"price": close},
        volume=volume,
        construction=config,
        state=AccountState(
            equity=EQUITY,
            held_weights=pd.Series(HELD, dtype=float),
            cash=EQUITY,
            positions=positions,
        ),
        sessions=[str(bar.date()) for bar in close.index],
        unrankable=frozenset(MASK),
        max_gross=config.max_gross,
    )
    settings.update(overrides)
    return DecisionContext(**settings)


def _hexes(series: pd.Series) -> dict[str, str]:
    return {str(k): float(v).hex() for k, v in series.items()}


# ---------------------------------------------------------------------------
# Parity: the live book's weights are unchanged, bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("book", sorted(GOLDEN_TARGETS))
def test_the_assembled_book_matches_the_pre_extraction_targets(book: str) -> None:
    close, volume = _panels()
    output = assemble(_context(_config(book), close, volume))
    assert _hexes(output.targets) == GOLDEN_TARGETS[book]


@pytest.mark.parametrize("book", sorted(GOLDEN_TARGETS))
def test_the_live_driver_step_produces_the_same_targets(book: str) -> None:
    # The driver's refresh step now calls the assembly; this is the same
    # comparison from the caller the live loop actually uses, including the
    # mask provider seam and the masked-name list it reports.
    close, volume = _panels()
    config = _config(book)
    targets, masked = _construct_refresh_targets(
        Node(),
        close,
        volume,
        config,
        pd.Series(HELD, dtype=float),
        EQUITY,
        config.max_gross,
        lambda bar: list(MASK),
        str(close.index[-1].date()),
    )
    assert _hexes(targets) == GOLDEN_TARGETS[book]
    assert masked == sorted(MASK)


def test_a_masked_held_name_is_held_and_a_masked_unheld_name_stays_flat() -> None:
    # The property the goldens encode, said out loud: masking pins nothing, it
    # withdraws the decision. S02 and S09 are held, so the band keeps them at
    # what is held; an unheld masked name has nothing to hold and stays flat.
    close, volume = _panels()
    output = assemble(_context(_config("decile_neutral"), close, volume))
    assert output.targets["S02"] == HELD["S02"]
    assert output.targets["S09"] == HELD["S09"]

    unheld = assemble(_context(_config("decile_neutral"), close, volume, unrankable=frozenset({"S04"})))
    assert unheld.targets["S04"] == 0.0
    assert unheld.diagnostics["masked"] == ["S04"]


def test_the_regime_de_gross_still_reaches_the_construct() -> None:
    close, volume = _panels()
    config = _config("decile_neutral", max_participation=None, no_trade_band=0.0)
    full = assemble(_context(config, close, volume, max_gross=config.max_gross, unrankable=frozenset()))
    half = assemble(_context(config, close, volume, max_gross=0.5, unrankable=frozenset()))
    # The cap is applied by the book construct the caller bound, so the context
    # value must reach it: half the gross, not the same book.
    plan = ComponentPlan(
        component=Node(),
        component_id="decile_neutral",
        to_weights="decile_neutral",
        weights_fn=heuristic_book_rule(config, close, 0.5),
    )
    halved = assemble(_context(config, close, volume, components=(plan,), unrankable=frozenset()))
    assert full.targets.abs().sum() == pytest.approx(1.0)
    assert halved.targets.abs().sum() == pytest.approx(0.5)
    assert half.diagnostics["step_order"] == "cap_band_participation"


# ---------------------------------------------------------------------------
# Sizing needs a reconciled book
# ---------------------------------------------------------------------------


def test_no_intents_are_sized_without_a_reconciled_position_book() -> None:
    close, volume = _panels()
    output = assemble(_context(_config("decile_neutral"), close, volume))
    assert output.order_intents == []
    assert any("no reconciled position book" in warning for warning in output.warnings)


def test_intents_are_sized_against_the_reconciled_book_at_the_decision_closes() -> None:
    close, volume = _panels()
    config = _config("decile_neutral")
    prices = close.iloc[-1]
    positions = {"S00": 100.0, "S05": -50.0}
    output = assemble(_context(config, close, volume, positions=positions))

    assert output.order_intents, "a decided book with a held position must produce intents"
    by_symbol = {order.symbol: order for order in output.order_intents}
    for symbol, order in by_symbol.items():
        assert order.reference_price == float(prices[symbol])
        assert order.decision_bar == str(close.index[-1].date())
        assert order.qty == float(round(order.qty))  # whole shares (OPG)
    expected = float(round(output.targets["S00"] * EQUITY / float(prices["S00"]) - 100.0))
    assert by_symbol["S00"].qty == expected
    assert output.diagnostics["n_orders"] == len(output.order_intents)


# ---------------------------------------------------------------------------
# Liquidate: exempt from the band and the dust floor, not from participation
# ---------------------------------------------------------------------------


class Liquidator(Node):
    """Scores everything, then orders one name out."""

    def instructions(self, close, volume=None) -> pd.Series:
        return pd.Series({"S00": "liquidate"}, dtype=object)


def test_a_liquidate_instruction_bypasses_the_band_and_the_dust_floor() -> None:
    close, volume = _panels()
    # A band wide enough to hold everything where it is: without the exemption
    # the flat would be damped straight back to the held weight.
    config = _config("decile_neutral", no_trade_band=0.5, max_participation=None)
    prices = close.iloc[-1]
    tiny = 1.0  # one share: a $50 position under a $10,000 dust floor
    plan = ComponentPlan(
        component=Liquidator(),
        component_id="decile_neutral",
        to_weights="decile_neutral",
        weights_fn=heuristic_book_rule(config, close, config.max_gross),
    )
    held_weight = tiny * float(prices["S00"]) / EQUITY
    ctx = _context(
        config,
        close,
        volume,
        components=(plan,),
        unrankable=frozenset(),
        positions={"S00": tiny},
        state=AccountState(
            equity=EQUITY,
            held_weights=pd.Series({"S00": held_weight}, dtype=float),
            cash=EQUITY,
            positions={"S00": tiny},
        ),
        construction=_config("decile_neutral", no_trade_band=0.5, max_participation=None, min_order_notional=10_000.0),
    )
    output = assemble(ctx)

    assert output.targets["S00"] == 0.0  # the band did not hold it back
    assert output.diagnostics["liquidate"] == ["S00"]
    exit_order = next(order for order in output.order_intents if order.symbol == "S00")
    assert exit_order.qty == -tiny  # dust floor waived: the position is exited


def test_the_dust_floor_still_applies_to_everything_else() -> None:
    close, _ = _panels()
    prices = close.iloc[-1]
    orders = size_orders(
        pd.Series({"S00": 0.0, "S01": 0.5}),
        positions={"S00": 1.0, "S01": 0.0},
        prices=prices,
        equity=EQUITY,
        decision_bar="2026-02-26",
        min_order_notional=10_000.0,
        whole_shares=True,
    )
    assert {order.symbol for order in orders} == {"S01"}  # S00's one share is dust


# ---------------------------------------------------------------------------
# The declared step order
# ---------------------------------------------------------------------------


def test_band_cap_participation_recaps_the_blend_the_band_produced() -> None:
    # The band blends held and fresh weights per name, so the blend can exceed
    # max_gross; the trend/demotion path re-caps after it and the momentum path
    # does not. Every name here moves less than the band, so the band holds the
    # 1.6-gross book and the two step orders disagree by 0.6 of gross.
    close, _ = _panels()
    config = _config("decile_neutral", no_trade_band=0.05, max_participation=None, max_gross=1.0)
    held = pd.Series({symbol: 0.2 for symbol in SYMBOLS[:8]}, dtype=float)
    raw = pd.Series({symbol: 0.17 for symbol in SYMBOLS[:8]}, dtype=float)

    kept = construct_targets(raw, prev_weights=held, close=close, volume=None, config=config, equity=EQUITY)
    recapped = construct_targets(
        raw,
        prev_weights=held,
        close=close,
        volume=None,
        config=config,
        equity=EQUITY,
        step_order="band_cap_participation",
    )

    assert kept.abs().sum() == pytest.approx(1.6)  # the band held every name
    assert recapped.abs().sum() == pytest.approx(1.0)  # re-capped to max_gross


def test_an_unknown_step_order_is_refused() -> None:
    close, volume = _panels()
    with pytest.raises(AssemblyError, match="step order"):
        _context(_config("decile_neutral"), close, volume, step_order="participation_first")


def test_the_as_traded_price_panel_is_required() -> None:
    close, volume = _panels()
    with pytest.raises(AssemblyError, match="'price' close panel"):
        _context(
            _config("decile_neutral"),
            close,
            volume,
            scoring_panels={"total_return": close},
        )


def test_a_non_positive_equity_cannot_size_a_book() -> None:
    with pytest.raises(AssemblyError, match="equity"):
        AccountState(equity=0.0, held_weights=pd.Series(dtype=float))


def test_the_participation_gate_without_a_volume_panel_is_a_loud_failure() -> None:
    close, _ = _panels()
    config = _config("decile_neutral")
    with pytest.raises(ValueError, match="no volume panel"):
        construct_targets(
            pd.Series({"S00": 0.5}),
            prev_weights=pd.Series(dtype=float),
            close=close,
            volume=None,
            config=config,
            equity=EQUITY,
        )


def test_the_decision_output_records_what_the_stage_cannot_yet_produce() -> None:
    # Batch rounding repair (§3.7) is not implemented at this stage; the field
    # is declared and None rather than quietly filled with something else.
    close, volume = _panels()
    output = assemble(_context(_config("decile_neutral"), close, volume))
    assert output.rounding is None
    assert output.construction.solver is None
    assert output.construction.path == "normal"
    assert output.diagnostics["combination"] == "position_space"
    assert math.isfinite(output.diagnostics["gross"])
