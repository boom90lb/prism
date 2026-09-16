"""Durable state at schema version 3, and the migration that must not lose a share.

The acceptance test of this stage runs on the real book: the live paper loop's
``state.json`` is a version-2 document with 99 positions and 937,669.47 in cash,
and it must reach version 3 with every position, the cash, the three bar
markers and every client order id intact. A migration that silently drops a
field is the failure these tests exist to catch, so they compare the migrated
state against the *source document* field by field rather than against a fixture
someone wrote by hand.

The live document is never written to: it is copied into ``tmp_path`` first, and
the copy's bytes are compared before and after to prove the loader read it
without touching it.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from prism.execution.accounting import mark_prices
from prism.execution.portfolio_state import (
    Fill,
    OrderIntent,
    OrderRecord,
    PortfolioState,
    UnpricedPosition,
)
from prism.live.broker import Order
from prism.live.state import (
    STATE_SCHEMA_VERSION,
    UNANCHORED_BAR,
    ComponentState,
    LoopState,
    StateStore,
    migrate_state_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_RUN_NAME = "paper_loop_momentum2"


def _live_run_dir() -> Path | None:
    """The live paper run directory, or ``None`` when this checkout cannot see it.

    ``PRISM_LIVE_RUN_DIR`` overrides the search. Otherwise the walk goes up from
    the repository root, because the tests run inside a git worktree whose
    parent checkout is where the operating run directories live; ``runs/`` is
    operating data and is not in the repository.
    """
    override = os.environ.get("PRISM_LIVE_RUN_DIR")
    if override:
        candidate = Path(override)
        return candidate if (candidate / "state.json").is_file() else None
    for base in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = base / "runs" / LIVE_RUN_NAME
        if (candidate / "state.json").is_file():
            return candidate
    return None


@pytest.fixture()
def live_v2_state(tmp_path: Path) -> tuple[dict, Path]:
    """A **copy** of the live v2 document, and the payload it holds."""
    run_dir = _live_run_dir()
    if run_dir is None:
        pytest.skip(f"no live run directory holding runs/{LIVE_RUN_NAME}/state.json is visible from {REPO_ROOT}")
    source = run_dir / "state.json"
    copy = tmp_path / "state.json"
    shutil.copy2(source, copy)
    payload = json.loads(copy.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        pytest.skip(f"{source} is at schema_version {payload.get('schema_version')!r}, not the version-2 document")
    return payload, copy


def _recorded_reference_prices(run_dir: Path) -> dict[str, float] | None:
    """The last refresh's recorded decision-close prices, or ``None``.

    These are real prices the loop wrote at its last refresh — recorded
    evidence, not a fixture — and they cover exactly the names the book holds.
    """
    ledger = run_dir / "targets.jsonl"
    if not ledger.is_file():
        return None
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None
    prices = rows[-1].get("reference_prices") or {}
    return {str(sid): float(price) for sid, price in prices.items()} or None


# ---------------------------------------------------------------------------
# the acceptance test: the live book migrates without loss
# ---------------------------------------------------------------------------


def test_the_live_v2_document_migrates_to_v3_without_loss(live_v2_state: tuple[dict, Path]) -> None:
    payload, copy = live_v2_state
    before = copy.read_bytes()

    state = StateStore(copy).load()

    assert state is not None
    assert state.schema_version == STATE_SCHEMA_VERSION == 3
    assert copy.read_bytes() == before, "loading a state file must never rewrite it"

    # Every position, at its exact share count, in both directions.
    v2_positions = {str(sid): float(shares) for sid, shares in payload["positions"].items()}
    assert state.positions == v2_positions
    assert len(state.portfolio.positions) == len(v2_positions) == 99
    assert set(state.portfolio.positions) == set(v2_positions)
    for sid, position in state.portfolio.positions.items():
        assert position.shares == v2_positions[sid]
        # No price is invented anywhere in the migration table (§3.11): the
        # book carries no mark until a close prices it.
        assert position.last_price is None and position.last_price_bar is None
        assert position.valuation_status == "stale"

    # Cash, to the cent and to the bit.
    assert state.cash == float(payload["cash"]) == 937669.47

    # The three bar markers keep their names and values, and the fills marker
    # starts from the settled bar (decision D4).
    assert state.pending_decision_bar == payload["pending_decision_bar"]
    assert state.last_settled_bar == payload["last_settled_bar"]
    assert state.last_refresh_bar == payload["last_refresh_bar"]
    assert state.fills_marker == payload["last_settled_bar"]
    assert state.portfolio.as_of == payload["last_settled_bar"]

    # Pending orders, with their client order ids verbatim.
    assert [order.client_order_id for order in state.pending_orders] == [
        order["client_order_id"] for order in payload["pending_orders"]
    ]

    # Nothing a v2 document never held is invented.
    assert state.portfolio.liabilities == {} and state.portfolio.receivables == {}
    assert state.portfolio.obligations == [] and state.portfolio.sequence == 0
    assert state.decision_id is None and state.hysteresis == {}


def test_the_live_book_values_the_same_at_v3_as_it_did_at_v2(live_v2_state: tuple[dict, Path]) -> None:
    """Equity from the migrated state equals equity from the v2 document, at the same prices.

    The prices are the ones the loop recorded at its last refresh, so this is
    the real book at real marks; the two equity figures are computed by
    independent routes (a sum over the v2 dict, and ``PortfolioState.equity``
    over the migrated positions) and must agree exactly.
    """
    payload, copy = live_v2_state
    run_dir = _live_run_dir()
    assert run_dir is not None
    prices = _recorded_reference_prices(run_dir)
    if prices is None or not set(payload["positions"]) <= set(prices):
        pytest.skip("the recorded targets ledger does not price every held name")

    v2_equity = float(payload["cash"]) + sum(
        float(shares) * prices[str(sid)] for sid, shares in payload["positions"].items()
    )

    state = StateStore(copy).load()
    assert state is not None
    # The migrated book carries no mark, so it must be priced before it can be
    # valued: that is the rule, not an inconvenience (§3.6).
    marked, stale = mark_prices(state.portfolio, bar=payload["last_settled_bar"], prices=prices)
    assert stale == ()
    assert marked.equity(prices) == pytest.approx(v2_equity, abs=1e-9)
    assert marked.equity(prices) == pytest.approx(928556.975, abs=1e-6)


def test_the_migrated_live_book_round_trips_through_the_store(live_v2_state: tuple[dict, Path], tmp_path) -> None:
    payload, copy = live_v2_state
    state = StateStore(copy).load()
    assert state is not None

    written = tmp_path / "v3.json"
    store = StateStore(written)
    store.save(state)
    reloaded = store.load()

    assert reloaded == state
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["schema_version"] == 3
    assert len(document["portfolio"]["positions"]) == len(payload["positions"])
    assert document["portfolio"]["cash"] == payload["cash"]
    # The v2 keys are gone from the document: one book, one representation.
    assert "positions" not in document and "cash" not in document


def test_pending_orders_survive_the_migration_with_their_client_order_ids(
    live_v2_state: tuple[dict, Path], tmp_path
) -> None:
    """The live document has no pending orders, so the case is exercised on it *with* some.

    Client order ids are the idempotency key: regenerating one would resubmit a
    live order, so the migration copies them verbatim rather than deriving them.
    """
    payload, _copy = live_v2_state
    held = sorted(payload["positions"])[:2]
    payload["pending_orders"] = [
        {
            "client_order_id": f"mom:{payload['pending_decision_bar']}:{symbol}",
            "symbol": symbol,
            "qty": 7.0 if index == 0 else -3.0,
            "decision_bar": payload["pending_decision_bar"],
            "reference_price": 101.5,
        }
        for index, symbol in enumerate(held)
    ]
    path = tmp_path / "with_pending.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    state = StateStore(path).load()

    assert state is not None
    assert [order.client_order_id for order in state.pending_orders] == [
        f"mom:{payload['pending_decision_bar']}:{held[0]}",
        f"mom:{payload['pending_decision_bar']}:{held[1]}",
    ]
    assert [order.qty for order in state.pending_orders] == [7.0, -3.0]
    assert [order.symbol for order in state.pending_orders] == held
    assert [order.decision_bar for order in state.pending_orders] == [payload["pending_decision_bar"]] * 2
    # They are open orders in the canonical book, at the venue's whole-share increment.
    records = state.portfolio.pending_orders
    assert [record.status for record in records] == ["pending", "pending"]
    assert [record.intent.increment for record in records] == [1.0, 1.0]
    assert state.portfolio.committed_cash() == pytest.approx(7.0 * 101.5)


# ---------------------------------------------------------------------------
# the migration steps themselves
# ---------------------------------------------------------------------------


def _v2(**overrides) -> dict:
    payload = {
        "positions": {"AAA": 10.0, "BBB": -4.0},
        "cash": 1_000.0,
        "pending_orders": [],
        "pending_decision_bar": None,
        "last_settled_bar": "2026-07-06",
        "last_refresh_bar": "2026-07-01",
        "schema_version": 2,
    }
    payload.update(overrides)
    return payload


def test_v1_reaches_v3_in_two_declared_hops() -> None:
    payload = _v2(schema_version=1)
    del payload["last_refresh_bar"]

    migrated = migrate_state_payload(payload)

    assert migrated["schema_version"] == 3
    assert migrated["last_refresh_bar"] is None  # the v1 -> v2 step, kept
    assert migrated["portfolio"]["positions"]["AAA"]["shares"] == 10.0
    assert migrated["fills_marker"] == "2026-07-06"


@pytest.mark.parametrize("version", [0, 4, "2", None])
def test_an_unknown_schema_version_is_a_hard_stop(version) -> None:
    with pytest.raises(ValueError, match="do not start flat"):
        migrate_state_payload(_v2(schema_version=version))


def test_the_migration_records_the_running_configuration_and_the_declared_components() -> None:
    migrated = migrate_state_payload(
        _v2(),
        components=["momentum", "trend"],
        config_id="cfg-1234",
        engine_version="9.9.9",
    )

    assert migrated["decision_id"] is None, "a migrated state was migrated, not decided"
    assert migrated["config_id"] == "cfg-1234" and migrated["engine_version"] == "9.9.9"
    components = [ComponentState.from_dict(row) for row in migrated["components"]]
    assert [component.component_id for component in components] == ["momentum", "trend"]
    # last_refresh_bar becomes every component's last_output_bar (§3.11 table).
    assert {component.last_output_bar for component in components} == {"2026-07-01"}
    assert all(component.fitted_state_ref is None and not component.expired for component in components)
    assert {component.fit_scope for component in components} == {"trailing_window"}


def test_a_symbol_that_resolves_to_no_sid_stops_the_migration_naming_it() -> None:
    resolver = {"AAA": "sec_000001"}.get

    with pytest.raises(ValueError, match="'BBB'"):
        migrate_state_payload(_v2(), resolve_sid=resolver)

    migrated = migrate_state_payload(
        _v2(positions={"AAA": 10.0}),
        resolve_sid=resolver,
    )
    assert list(migrated["portfolio"]["positions"]) == ["sec_000001"]


def test_two_symbols_resolving_to_one_sid_are_refused_rather_than_netted() -> None:
    with pytest.raises(ValueError, match="two v2 symbols resolve"):
        migrate_state_payload(_v2(), resolve_sid=lambda symbol: "sec_000001")


def test_a_fractional_pending_order_is_carried_as_one_lot_and_says_so(caplog) -> None:
    payload = _v2(
        pending_orders=[
            {
                "client_order_id": "book:2026-07-06:AAA",
                "symbol": "AAA",
                "qty": 2.5,
                "decision_bar": "2026-07-06",
                "reference_price": 40.0,
            }
        ]
    )

    with caplog.at_level("WARNING"):
        migrated = migrate_state_payload(payload)

    intent = migrated["portfolio"]["pending_orders"][0]["intent"]
    assert intent["qty"] == 2.5 and intent["increment"] == 2.5
    assert "fractional quantity" in caplog.text
    # And it survives the round trip at that quantity, which is what matters.
    assert LoopState.from_dict(migrated).pending_orders[0].qty == 2.5


# ---------------------------------------------------------------------------
# the version-3 document itself
# ---------------------------------------------------------------------------


def test_a_fresh_loop_state_is_unanchored_and_empty() -> None:
    state = LoopState()

    assert state.schema_version == 3
    assert state.positions == {} and state.cash == 0.0 and state.pending_orders == []
    assert state.portfolio.as_of == UNANCHORED_BAR
    assert state.fills_marker is None and state.components == [] and state.hysteresis == {}
    assert state.portfolio.as_of < "1970-01-01", "the sentinel sorts before every session"


def test_the_document_round_trips_byte_identically_with_fills_and_components(tmp_path) -> None:
    intent = OrderIntent(
        sid="AAA",
        qty=10.0,
        increment=1.0,
        reference_price=100.0,
        decision_bar="2026-07-06",
    )
    record = OrderRecord(
        intent=intent,
        client_order_id="mom:2026-07-06:AAA",
        status="partial",
        filled_qty=4.0,
        fills=[
            Fill(
                execution_id="exec-1",
                client_order_id="mom:2026-07-06:AAA",
                sid="AAA",
                ts="2026-07-07T09:30:00-04:00",
                qty=4.0,
                fill_price=100.25,
                reference_price=100.0,
                commission=0.02,
                source="broker",
            )
        ],
        submitted_at="2026-07-06T16:05:00-04:00",
    )
    portfolio = PortfolioState(as_of="2026-07-06", cash=500.0, pending_orders=[record])
    portfolio.set_shares("AAA", 10.0, price=100.0, bar="2026-07-06")
    portfolio.add_receivable("cash_distribution:AAA:2026-07-02", 12.5)
    state = LoopState(
        portfolio,
        decision_id="dec-7",
        config_id="cfg-9",
        engine_version="1.2.3",
        components=[
            ComponentState(
                component_id="momentum",
                last_output_bar="2026-07-06",
                decision_bar="2026-07-06",
                expires_bar="2026-08-03",
                sub_targets={"AAA": 0.25},
                fitted_state_ref="sha256:" + "ab" * 32,
                fitted_through_bar="2026-07-06",
            )
        ],
        hysteresis={"AAA": 0.25},
        fills_marker="2026-07-07T09:31:00Z",
        pending_decision_bar="2026-07-06",
        last_settled_bar="2026-07-02",
        last_refresh_bar="2026-07-06",
    )

    store = StateStore(tmp_path / "state.json")
    store.save(state)
    first = (tmp_path / "state.json").read_text(encoding="utf-8")
    reloaded = store.load()
    assert reloaded is not None
    store.save(reloaded)

    assert (tmp_path / "state.json").read_text(encoding="utf-8") == first
    assert reloaded == state
    order = reloaded.portfolio.pending_orders[0]
    assert order.client_order_id == "mom:2026-07-06:AAA" and order.status == "partial"
    assert order.filled_qty == 4.0 and order.fills[0].execution_id == "exec-1"
    assert order.fills[0].commission == 0.02 and order.fills[0].source == "broker"
    assert reloaded.components[0].fitted_state_ref == "sha256:" + "ab" * 32
    assert reloaded.hysteresis == {"AAA": 0.25} and reloaded.decision_id == "dec-7"
    assert reloaded.portfolio.receivables == {"cash_distribution:AAA:2026-07-02": 12.5}


def test_the_v2_views_write_into_the_portfolio_and_keep_what_they_should(tmp_path) -> None:
    """The loop's vocabulary still works, and it is a view — not a second copy of the book."""
    state = LoopState()
    state.positions = {"AAA": 10.0}
    state.cash = 250.0
    state.portfolio.positions["AAA"].last_price = 100.0
    state.portfolio.positions["AAA"].last_price_bar = "2026-07-06"
    state.portfolio.positions["AAA"].valuation_status = "priced"

    # Re-adopting a share count keeps the mark of a name that stays held.
    state.positions = {"AAA": 12.0, "BBB": 5.0}
    assert state.portfolio.positions["AAA"].last_price == 100.0
    assert state.portfolio.positions["AAA"].valuation_status == "priced"
    assert state.portfolio.positions["BBB"].last_price is None
    assert state.portfolio.positions["BBB"].valuation_status == "stale"
    # And an adopted name with no mark refuses to be valued rather than being
    # valued at zero, even when a price is supplied: it is stale until a mark
    # step prices it (§3.6, SPEC N7).
    with pytest.raises(UnpricedPosition, match="BBB"):
        state.portfolio.equity({"AAA": 100.0, "BBB": 20.0})

    # A name that leaves the book leaves nothing behind.
    state.positions = {"AAA": 12.0}
    assert list(state.portfolio.positions) == ["AAA"]

    # Rewriting the pending set keeps the record of an order that did not change,
    # so the executions already applied to it are not discarded.
    order = Order("mom:2026-07-06:AAA", "AAA", 5.0, "2026-07-06", 100.0)
    state.pending_orders = [order]
    state.portfolio.pending_orders[0].status = "partial"
    state.portfolio.pending_orders[0].filled_qty = 2.0
    state.pending_orders = list(state.pending_orders) + [Order("mom:2026-07-06:BBB", "BBB", 1.0, "2026-07-06", 20.0)]
    assert [record.status for record in state.portfolio.pending_orders] == ["partial", "pending"]
    assert state.portfolio.pending_orders[0].filled_qty == 2.0
