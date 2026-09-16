"""Parity of the extracted assembly against the recorded paper sessions.

``tests/fixtures/paper_replay`` is the stage 0 capture of the live momentum
book: two decision-bearing sessions (2026-07-13 and 2026-08-11) with their
decided targets, decision-close reference prices, marked equity, submitted
orders, unfilled residuals and the spin-off mask file the second session used.
What the capture does **not** hold is bar panels — its manifest records that no
local bar source covers those sessions — so the two parity claims here are
deliberately different in strength:

* **Executable now.** Feed the recorded targets, reference prices, equity and
  the position book implied by the recorded fills through ``assemble`` and
  through the live sizing entry point, and assert both reproduce the recorded
  orders exactly: every symbol, every signed quantity, every client order id,
  every reference price. That exercises composition, the band, and the sizing,
  zero-crossing and dust rules on real recorded numbers, including the seven
  orders the whole-share/dust rules censored at the second session. The masked
  session's mask is asserted separately, because the recorded targets are
  written after the band and already carry its effect.

* **Deferred.** Replaying the sessions end to end through
  ``replay_daily_cycles`` — score, rank, construct, decide — needs bar panels
  for the universe the sessions actually ranked. The test is written and skips
  on the manifest's own recorded reasons; it starts running the day those
  inputs exist.

Both read the fixture, never the live run directory.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from prism.assembly import AccountState, DecisionContext, assemble
from prism.composition import ComponentPlan
from prism.live.daily import DailyBookConfig
from prism.live.loop import targets_to_orders
from prism.signal.base import Signal

FIXTURE = Path(__file__).parent / "fixtures" / "paper_replay"

#: The recorded book's construction policy (``prism.scripts.paper_books``:
#: the momentum book's CLI defaults, which is what these sessions traded).
MOMENTUM = DailyBookConfig(
    book="decile_neutral",
    decile=0.10,
    max_gross=1.0,
    max_symbol_abs_weight=0.10,
    no_trade_band=0.0,
    max_participation=None,
    min_order_notional=1.0,
    whole_shares=True,
    decision_every=21,
)
ORDER_ID_PREFIX = "mom:"


def _manifest() -> dict:
    return json.loads((FIXTURE / "manifest.json").read_text())


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURE / name).read_text().splitlines() if line.strip()]


def _decision_bars() -> list[str]:
    return list(_manifest()["decision_bearing_bars"])


def _targets_row(bar: str) -> dict:
    rows = [row for row in _rows("targets.jsonl") if row["decision_bar"] == bar]
    assert len(rows) == 1, f"expected exactly one recorded targets row for {bar}, got {len(rows)}"
    return rows[0]


def _recorded_orders(bar: str) -> dict[str, dict]:
    """Every order the session submitted, from the two ledgers that record them.

    An order that did not fill completely is in ``unfilled.jsonl`` with its
    submitted quantity; one that filled outright is only in ``fills.jsonl``,
    where a partially-filled parent contributes its executed quantity and the
    morning sweep's re-submission carries a suffixed id (``...:S1``). Summing
    fills per symbol therefore recovers the parent order exactly when it is not
    in the unfilled ledger.
    """
    orders: dict[str, dict] = {}
    for row in _rows("unfilled.jsonl"):
        if row["decision_bar"] == bar:
            orders[row["symbol"]] = {
                "qty": float(row["order_qty"]),
                "reference_price": float(row["reference_price"]),
                "client_order_id": f"{ORDER_ID_PREFIX}{bar}:{row['symbol']}",
            }
    filled: dict[str, float] = defaultdict(float)
    prices: dict[str, float] = {}
    for row in _rows("fills.jsonl"):
        if row["decision_bar"] == bar:
            filled[row["symbol"]] += float(row["qty"])
            prices[row["symbol"]] = float(row["reference_price"])
    for symbol, qty in filled.items():
        orders.setdefault(
            symbol,
            {
                "qty": qty,
                "reference_price": prices[symbol],
                "client_order_id": f"{ORDER_ID_PREFIX}{bar}:{symbol}",
            },
        )
    return orders


def _positions_before(bar: str) -> dict[str, float]:
    """The share book the session decided against, from the fills that preceded it.

    The first recorded session starts flat (its equity mark is all cash), and
    every later one holds what the earlier decisions filled. The reconstruction
    is checked against the recorded equity mark before it is used.
    """
    positions: dict[str, float] = defaultdict(float)
    for row in _rows("fills.jsonl"):
        if row["filled_bar"] <= bar:
            positions[row["symbol"]] += float(row["qty"])
    return {symbol: qty for symbol, qty in positions.items() if qty != 0.0}


def _prices_for(bar: str) -> pd.Series:
    """Decision-close reference prices: the targets row plus each order's own."""
    prices = dict(_targets_row(bar)["reference_prices"])
    for symbol, order in _recorded_orders(bar).items():
        prices.setdefault(symbol, order["reference_price"])
    return pd.Series(prices, dtype=float)


def _masked(bar: str) -> frozenset[str]:
    path = FIXTURE / f"spinoff_mask_{bar}.json"
    if not path.exists():
        return frozenset()
    return frozenset(json.loads(path.read_text())["flagged"])


def _decided_book(bar: str, positions: dict[str, float]) -> pd.Series:
    """The target vector the session's construction emitted, as sizing saw it.

    The targets ledger records only the non-zero weights (``loop.py``:
    "absent names read as an explicit 0.0 downstream"), so a held name missing
    from it was decided flat — that is the exit leg. The ledger row is written
    *after* the band, so a masked name is already in it at the weight the band
    resolved its no-decision to (its held weight); no NaN reaches sizing from
    this reconstruction, which is why the mask needs its own assertion below.
    """
    book = {symbol: float(weight) for symbol, weight in _targets_row(bar)["targets"].items()}
    for symbol in positions:
        book.setdefault(symbol, 0.0)
    return pd.Series(book, dtype=float)


class RecordedBook(Signal):
    """The recorded decided book, replayed as a ``target_weights`` component."""

    output_kind = "target_weights"
    units = "weight"
    scoring_basis = "price"
    price_scale_invariant = False
    fitted_state = "none"
    estimated_decay_bars = None

    def __init__(self, weights: pd.Series) -> None:
        self._weights = weights

    @property
    def horizon_bars(self) -> int:
        return 21

    @property
    def required_history(self) -> int:
        return 1

    def fit(self, close, volume=None) -> "RecordedBook":
        return self

    def score(self, close, volume=None) -> pd.Series:
        return self._weights.reindex(close.columns)


def _as_dict(orders) -> dict[str, dict]:
    return {
        order.symbol: {
            "qty": order.qty,
            "reference_price": order.reference_price,
            "client_order_id": order.client_order_id,
        }
        for order in orders
    }


# ---------------------------------------------------------------------------
# The fixture is what it says it is
# ---------------------------------------------------------------------------


def test_the_fixture_files_match_their_recorded_hashes() -> None:
    manifest = _manifest()
    for name, recorded in manifest["files"].items():
        digest = hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()
        assert digest == recorded["sha256"], f"{name} is not the file the capture recorded"


def test_the_reconstructed_share_book_reproduces_the_recorded_equity_mark() -> None:
    # The parity claim below rests on this: the positions derived from the
    # fills ledger are the book the session marked. Checked against the
    # recorded equity at the recorded reference prices before it is trusted.
    equity_marks = {row["decision_bar"]: row for row in _rows("equity.jsonl")}
    for bar in _decision_bars():
        positions = _positions_before(bar)
        prices = _prices_for(bar)
        mark = equity_marks[bar]
        equity = float(mark["cash"]) + sum(qty * float(prices[s]) for s, qty in positions.items())
        assert equity == pytest.approx(float(mark["equity"]), abs=1e-6)
        assert float(mark["equity"]) == pytest.approx(float(_targets_row(bar)["equity"]), abs=1e-9)


# ---------------------------------------------------------------------------
# Executable parity: recorded targets -> recorded orders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bar", _decision_bars())
def test_the_assembly_reproduces_the_recorded_orders(bar: str) -> None:
    row = _targets_row(bar)
    positions = _positions_before(bar)
    prices = _prices_for(bar)
    book = _decided_book(bar, positions)
    equity = float(row["equity"])
    close = pd.DataFrame(
        [prices.to_numpy()],
        index=pd.DatetimeIndex([pd.Timestamp(bar, tz="America/New_York")]),
        columns=list(prices.index),
    )
    held_weights = pd.Series(
        {symbol: qty * float(prices[symbol]) / equity for symbol, qty in positions.items()},
        dtype=float,
    )
    plan = ComponentPlan(
        component=RecordedBook(book),
        component_id="momentum",
        to_weights="none",
        output_kind="target_weights",
        update_every_bars=21,
        valid_for_bars=26,
    )

    output = assemble(
        DecisionContext(
            decision_bar=bar,
            components=(plan,),
            scoring_panels={"price": close},
            construction=MOMENTUM,
            state=AccountState(
                equity=equity,
                held_weights=held_weights,
                cash=equity - float((held_weights * equity).sum()),
                positions=positions,
            ),
            sessions=[bar],
            order_id_prefix=ORDER_ID_PREFIX,
        )
    )

    recorded = _recorded_orders(bar)
    assert _as_dict(output.order_intents) == recorded
    assert len(output.order_intents) == len(recorded)


@pytest.mark.parametrize("bar", _decision_bars())
def test_the_live_sizing_entry_point_reproduces_the_recorded_orders(bar: str) -> None:
    # The same numbers through ``live/loop.py``'s public name, which the
    # write-ahead protocol, the safety rails and the venue adapter call.
    row = _targets_row(bar)
    positions = _positions_before(bar)
    prices = _prices_for(bar)
    orders = targets_to_orders(
        _decided_book(bar, positions),
        positions,
        prices,
        float(row["equity"]),
        bar,
        min_order_notional=MOMENTUM.min_order_notional,
        whole_shares=MOMENTUM.whole_shares,
        order_id_prefix=ORDER_ID_PREFIX,
    )
    assert _as_dict(orders) == _recorded_orders(bar)


@pytest.mark.parametrize("bar", _decision_bars())
def test_the_reproduced_orders_account_for_the_recorded_residuals(bar: str) -> None:
    # The unfilled ledger's arithmetic, closed against the reproduced orders:
    # what was submitted is what executed plus what was left over.
    reproduced = _recorded_orders(bar)
    residual_rows = [row for row in _rows("unfilled.jsonl") if row["decision_bar"] == bar]
    assert residual_rows, f"the capture records unfilled residuals for {bar}"
    for row in residual_rows:
        order = reproduced[row["symbol"]]
        assert order["qty"] == pytest.approx(float(row["executed_qty"]) + float(row["residual_qty"]), abs=1e-9)
        assert order["qty"] == pytest.approx(float(row["order_qty"]), abs=1e-9)


def test_the_recorded_mask_held_its_name_instead_of_flattening_it() -> None:
    # The masked session's evidence that a spin-off mask withdraws the decision
    # rather than pinning a flat: DD was held (-77 shares) and masked, and the
    # recorded target for it is exactly its held weight — the no-trade band's
    # resolution of "no decision" — so no order was submitted. Had the mask
    # been read as an explicit 0.0, the record would carry a +77-share exit.
    bar = "2026-08-11"
    masked = _masked(bar)
    positions = _positions_before(bar)
    held_and_masked = sorted(set(positions) & masked)
    assert held_and_masked == ["DD"], f"the recorded mask covers held {held_and_masked}"

    row = _targets_row(bar)
    prices = _prices_for(bar)
    for symbol in held_and_masked:
        held_weight = positions[symbol] * float(prices[symbol]) / float(row["equity"])
        assert float(row["targets"][symbol]) == pytest.approx(held_weight, rel=1e-12)
        assert symbol not in _recorded_orders(bar)


# ---------------------------------------------------------------------------
# Deferred parity: the full replay
# ---------------------------------------------------------------------------


def _replay_blockers(manifest: dict) -> list[str]:
    """Why the recorded sessions cannot be replayed end to end, from the manifest."""
    blockers: list[str] = []
    panels = manifest["panels"]
    if panels.get("status") != "available":
        blockers.append(f"no bar panels: {panels.get('reason')}")
    universe = manifest["universe"]
    if int(universe["fixture_size"]) != int(universe["recorded_scored_universe_size"]):
        blockers.append(
            f"the fixture universe is trimmed to {universe['fixture_size']} names but the "
            f"sessions ranked {universe['recorded_scored_universe_size']}: "
            f"{universe['note']}"
        )
    return blockers


def test_full_session_replay_reproduces_the_recorded_targets_and_orders(tmp_path) -> None:
    """Replay every recorded session through the real cycle and compare.

    Skips while the capture's own manifest records that the inputs do not
    exist. It is written against the interfaces it will use so that restoring
    a bar source for the recorded universe turns it on, rather than leaving a
    parity gate to be written after the fact.
    """
    manifest = _manifest()
    blockers = _replay_blockers(manifest)
    if blockers:
        pytest.skip("cannot replay the recorded sessions: " + "; ".join(blockers))

    from prism.live.replay import (
        align_replay_panels,
        load_local_bar_panels,
        replay_daily_cycles,
    )
    from prism.scripts.paper_books import BookParams, book_spec

    symbols = [line.strip() for line in (FIXTURE / "universe.txt").read_text().splitlines() if line.strip()]
    close, volume, open_ = align_replay_panels(
        *load_local_bar_panels(symbols, data_dir=manifest["source"]["store_dir"])
    )
    spec = book_spec("momentum")
    params = BookParams()
    decision_every = spec.decision_every(params)
    masks = {bar: sorted(_masked(bar)) for bar in _decision_bars()}

    results, _ = replay_daily_cycles(
        lambda: spec.signal(params, decision_every, None),
        close,
        volume,
        open_,
        spec.config(params, decision_every),
        tmp_path / "replay",
        start_bar=manifest["sessions"][0]["bar"],
        end_bar=manifest["sessions"][-1]["bar"],
        initial_cash=float(_targets_row(_decision_bars()[0])["equity"]),
        unrankable=lambda bar: masks.get(bar, []),
    )

    # Declared before the run: the decided book is compared at 1e-9 in weight
    # space and the orders exactly, because a share count is an integer.
    by_bar = {result.decision_bar: result for result in results}
    for bar in _decision_bars():
        replayed = by_bar[bar].target_weights
        for symbol, weight in _targets_row(bar)["targets"].items():
            assert float(replayed[symbol]) == pytest.approx(float(weight), abs=1e-9)
        assert _as_dict(by_bar[bar].submitted_orders) == _recorded_orders(bar)
