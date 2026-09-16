"""Corporate events against the canonical ledger (spec §3.6, decision D3).

The specification names exactly what this file must cover: entitlement at the ex-date
boundary in both directions, receivable and liability settlement on the pay date, the short
debit, both price bases under a split, terminal payout, cancellation, and halt.

The canonical ex-date rule is a **Correction**, stated rather than inherited, and it is the
reason two of these tests exist at all. Entitlement is the signed share count at the close
of the bar before the ex-date, so a buyer filled at the ex-date open is not entitled and a
seller filled at the ex-date open is. ``research/trading.py`` credits the ex-date buyer
(fills at step 2 of the bar, distributions at step 5b) and ``backtest_target_weights``
does not; ``tests/test_dividends.py`` fixes neither, and stays valid for the research
trader it tests. Cash timing moves from the ex-date to the pay date, which is
equity-neutral and changes cash, financing, and buying power on the bars between.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.execution.accounting import apply_events, mark_at_close
from prism.execution.portfolio_state import PortfolioState, Position, empty_state, state_from_shares
from prism.execution.simulate import simulate
from prism.io.events import Event
from tests.test_accounting_conservation import cost_model
from tests.test_accounting_timeline import zero_stack

B0, B1, B2, B3 = "2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09"
BARS = [B0, B1, B2, B3]


def _event(event_type: str, effective_date: str, payload: dict, sid: str = "XXX", **kwargs) -> Event:
    return Event(
        sid=sid,
        event_type=event_type,
        effective_date=effective_date,
        payload=payload,
        ingested_at="2026-01-01T00:00:00Z",
        source="test",
        **kwargs,
    )


def _distribution(effective_date: str, amount: float = 1.0, pay_date: str | None = B3, sid: str = "XXX") -> Event:
    return _event(
        "cash_distribution",
        effective_date,
        {"amount_per_share": amount, "currency": "USD", "pay_date": pay_date, "distribution_kind": "dividend"},
        sid=sid,
    )


def _flat_panel(prices: dict[str, list[float]], bars=BARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(prices, index=list(bars))
    return frame.copy(), frame.copy()


# ---------------------------------------------------------------------------
# the ex-date boundary, in both directions
# ---------------------------------------------------------------------------


def test_a_buyer_filled_at_the_ex_date_open_is_not_entitled() -> None:
    opens, closes = _flat_panel({"XXX": [100.0, 100.0, 99.0, 99.0]})
    targets = pd.DataFrame(np.nan, index=BARS, columns=["XXX"])
    targets.loc[B1] = [0.50]  # decided at close t1, fills at the open of the ex-date
    result = simulate(
        opens,
        closes,
        targets,
        cost_model=cost_model(zero_stack(), equity=10_000.0),
        events=[_distribution(B2)],
        initial_cash=10_000.0,
    )
    assert result.shares.loc[B2, "XXX"] > 0.0, "the fixture must actually fill at the ex-date open"
    assert result.ledger.loc[B2, "receivables"] == 0.0
    booked = result.events[result.events["event_type"] == "cash_distribution"]
    assert len(booked) == 1
    assert booked.iloc[0]["detail"]["shares"] == 0.0
    assert booked.iloc[0]["detail"]["leg"] == "none"


def test_a_seller_filled_at_the_ex_date_open_is_entitled() -> None:
    opens, closes = _flat_panel({"XXX": [100.0, 100.0, 99.0, 99.0]})
    targets = pd.DataFrame(np.nan, index=BARS, columns=["XXX"])
    targets.loc[B0] = [0.50]  # buy at the open of t1
    targets.loc[B1] = [0.00]  # sell it all at the open of the ex-date
    result = simulate(
        opens,
        closes,
        targets,
        cost_model=cost_model(zero_stack(), equity=10_000.0),
        events=[_distribution(B2)],
        initial_cash=10_000.0,
    )
    held_into_the_ex_date = float(result.shares.loc[B1, "XXX"])
    assert held_into_the_ex_date == 50.0
    assert float(result.shares.loc[B2].fillna(0.0).iloc[0]) == 0.0, "sold at the ex-date open"
    assert result.ledger.loc[B2, "receivables"] == pytest.approx(50.0, abs=1e-12)
    assert result.ledger.loc[B3, "cash"] - result.ledger.loc[B2, "cash"] == pytest.approx(50.0, abs=1e-9)


def test_nothing_is_booked_at_the_announce_date() -> None:
    """A receivable booked at announcement would double count the distribution on every bar
    between the announcement and the ex-date, beside a position that has not marked down."""
    opens, closes = _flat_panel({"XXX": [100.0, 100.0, 100.0, 99.0]})
    targets = pd.DataFrame(np.nan, index=BARS, columns=["XXX"])
    targets.loc[B0] = [0.50]
    announced = _event(
        "cash_distribution",
        B3,
        {"amount_per_share": 1.0, "currency": "USD", "pay_date": None, "distribution_kind": "dividend"},
        announce_date=B1,
    )
    result = simulate(
        opens,
        closes,
        targets,
        cost_model=cost_model(zero_stack(), equity=10_000.0),
        events=[announced],
        initial_cash=10_000.0,
    )
    assert result.ledger.loc[B1, "receivables"] == 0.0
    assert result.ledger.loc[B2, "receivables"] == 0.0
    # no pay date is known, so the entry settles on the ex-date and says so
    assert result.ledger.loc[B3, "receivables"] == 0.0
    booked = result.events[result.events["event_type"] == "cash_distribution"].iloc[0]
    assert booked["detail"]["pay_date_assumed"] is True
    assert booked["detail"]["leg"] == "cash"
    assert booked["cash_delta"] == pytest.approx(50.0, abs=1e-12)


def test_a_short_pays_the_distribution_as_a_liability() -> None:
    state = state_from_shares(B1, {"XXX": -10.0}, {"XXX": 50.0}, cash=10_500.0)
    outcome = apply_events(state, bar=B2, events=[_distribution(B2, amount=1.0)])
    after = outcome.state
    assert after.liabilities == {"cash_distribution:XXX:2026-04-08": 10.0}
    assert after.cash == pytest.approx(10_500.0, abs=0.0), "nothing moves in cash on the ex-date"
    assert after.carried_equity() == pytest.approx(state.carried_equity() - 10.0, abs=1e-12)
    settled = apply_events(after, bar=B3, events=[_distribution(B2, amount=1.0)]).state
    assert settled.liabilities == {}
    assert settled.cash == pytest.approx(10_490.0, abs=1e-12)
    assert settled.carried_equity() == pytest.approx(after.carried_equity(), abs=1e-12)


# ---------------------------------------------------------------------------
# splits, against each declared price basis
# ---------------------------------------------------------------------------


def test_a_split_against_a_raw_series_rebases_shares_and_moves_no_equity() -> None:
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 200.0}, cash=1_000.0)
    before = state.carried_equity()
    outcome = apply_events(state, bar=B2, events=[_event("split", B2, {"factor": 2.0})], price_basis="raw")
    after = outcome.state
    assert after.shares("XXX") == 200.0
    assert after.positions["XXX"].last_price == pytest.approx(100.0, abs=1e-12)
    assert after.carried_equity() == pytest.approx(before, abs=1e-12)
    assert outcome.applied[0].detail["share_factor"] == 2.0


def test_a_split_against_an_adjusted_series_changes_no_shares_and_moves_no_equity() -> None:
    """D3: the vendor already restated the price series, so applying the ratio again would
    double count it. Both current sources fetch split-adjusted bars, so this is the path
    every legacy cache takes."""
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 200.0}, cash=1_000.0)
    before = state.carried_equity()
    outcome = apply_events(
        state, bar=B2, events=[_event("split", B2, {"factor": 2.0})], price_basis="split_adjusted_at_ingest"
    )
    after = outcome.state
    assert after.shares("XXX") == 100.0
    assert after.positions["XXX"].last_price == pytest.approx(200.0, abs=1e-12)
    assert after.carried_equity() == pytest.approx(before, abs=1e-12)
    assert outcome.applied[0].detail["share_factor"] == 1.0
    assert outcome.applied[0].detail["declared_factor"] == 2.0


def test_a_raw_split_cancels_the_orders_decided_on_the_old_basis() -> None:
    """An order decided at a pre-split reference price is a stale quantity at a stale price.

    Cancelling is the declared response: the alternative, rounding a rebased quantity back
    to the increment, changes the decided notional silently, and the alternative to *that*
    is filling a pre-split quantity at a post-split price, which doubles the position.
    """
    from prism.execution.accounting import decide, submit

    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 200.0}, cash=1_000.0)
    batch = decide(state, pd.Series({"XXX": 0.9}), pd.Series({"XXX": 200.0}), decision_bar=B1)
    state = submit(state, batch.intents, submitted_at=B1)
    assert len(state.pending_orders) == 1
    after = apply_events(state, bar=B2, events=[_event("split", B2, {"factor": 2.0})], price_basis="raw").state
    assert after.pending_orders == []


def test_an_unadjusted_basis_split_leaves_a_pending_order_alone() -> None:
    from prism.execution.accounting import decide, submit

    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 200.0}, cash=1_000.0)
    batch = decide(state, pd.Series({"XXX": 0.9}), pd.Series({"XXX": 200.0}), decision_bar=B1)
    state = submit(state, batch.intents, submitted_at=B1)
    after = apply_events(
        state, bar=B2, events=[_event("split", B2, {"factor": 2.0})], price_basis="split_adjusted_at_ingest"
    ).state
    assert len(after.pending_orders) == 1


# ---------------------------------------------------------------------------
# terminal payout, cancellation, halt
# ---------------------------------------------------------------------------


def test_a_terminal_payout_converts_the_claim_to_cash_and_clears_the_position() -> None:
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 12.0}, cash=500.0)
    state.positions["XXX"].valuation_status = "unresolved_claim"
    assert state.is_incomplete
    payout = _event("terminal_payout", B2, {"amount_per_share": 12.5, "currency": "USD", "pay_date": None})
    after = apply_events(state, bar=B2, events=[payout]).state
    assert "XXX" not in after.positions
    assert after.cash == pytest.approx(500.0 + 1_250.0, abs=1e-12)
    assert not after.is_incomplete, "the payout resolves the flag on records written after it"


def test_a_terminal_payout_with_a_pay_date_is_a_receivable_until_it_pays() -> None:
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 12.0}, cash=500.0)
    payout = _event("terminal_payout", B2, {"amount_per_share": 12.5, "currency": "USD", "pay_date": B3})
    booked = apply_events(state, bar=B2, events=[payout]).state
    assert booked.cash == pytest.approx(500.0, abs=0.0)
    assert booked.receivables == {"terminal_payout:XXX:2026-04-08": 1_250.0}
    assert booked.carried_equity() == pytest.approx(500.0 + 1_250.0, abs=1e-12)
    settled = apply_events(booked, bar=B3, events=[payout]).state
    assert settled.receivables == {}
    assert settled.cash == pytest.approx(1_750.0, abs=1e-12)


def test_an_unknown_terminal_value_is_an_unresolved_claim_and_never_a_wipeout() -> None:
    """A strict claim on an unknown terminal value is recorded ``incomplete`` (§3.2).

    It is never settled at zero, which would be indistinguishable in every downstream total
    from a resolved wipeout.
    """
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 12.0}, cash=500.0)
    after = apply_events(state, bar=B2, events=[_event("terminal_payout", B2, {"status": "unknown"})]).state
    assert after.positions["XXX"].valuation_status == "unresolved_claim"
    assert after.positions["XXX"].shares == 100.0
    assert after.is_incomplete
    assert after.unresolved_claims() == ["XXX"]
    assert after.carried_equity() == pytest.approx(state.carried_equity(), abs=0.0)


def test_a_resolved_wipeout_is_a_known_outcome_and_clears_the_position() -> None:
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 12.0}, cash=500.0)
    after = apply_events(
        state,
        bar=B2,
        events=[_event("terminal_payout", B2, {"amount_per_share": 0.0, "currency": "USD", "pay_date": None})],
    ).state
    assert "XXX" not in after.positions
    assert after.cash == pytest.approx(500.0, abs=0.0)
    assert after.carried_equity() == pytest.approx(500.0, abs=1e-12)


def test_a_share_cancellation_removes_the_position_at_zero_proceeds() -> None:
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 12.0}, cash=500.0)
    outcome = apply_events(state, bar=B2, events=[_event("share_cancellation", B2, {"fraction_cancelled": 1.0})])
    after = outcome.state
    assert "XXX" not in after.positions
    assert after.cash == pytest.approx(500.0, abs=0.0)
    assert after.carried_equity() == pytest.approx(500.0, abs=1e-12)
    applied = outcome.applied[0]
    assert applied.detail["proceeds"] == 0.0
    assert applied.detail["realized"] == pytest.approx(-1_200.0, abs=1e-12)
    assert applied.equity_delta == pytest.approx(-1_200.0, abs=1e-12)


def test_a_partial_cancellation_scales_the_position() -> None:
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 12.0}, cash=500.0)
    after = apply_events(state, bar=B2, events=[_event("share_cancellation", B2, {"fraction_cancelled": 0.25})]).state
    assert after.shares("XXX") == pytest.approx(75.0, abs=1e-12)
    assert after.carried_equity() == pytest.approx(500.0 + 900.0, abs=1e-12)


def test_a_halt_leaves_the_position_priced_at_its_last_bar_as_stale() -> None:
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 12.0}, cash=500.0)
    after = apply_events(
        state, bar=B2, events=[_event("halt", B2, {"start": "09:30", "end": None, "reason": "news"})]
    ).state
    position = after.positions["XXX"]
    assert position.valuation_status == "stale"
    assert position.last_price == 12.0
    assert position.last_price_bar == B1
    assert after.carried_equity() == pytest.approx(state.carried_equity(), abs=0.0)


def test_a_halted_bar_creates_no_fill_and_leaves_the_order_pending() -> None:
    """ "Nothing fills at a bar that does not trade" -- the order waits, it does not lapse."""
    opens = pd.DataFrame({"XXX": [100.0, np.nan, 100.0, 100.0]}, index=BARS)
    closes = pd.DataFrame({"XXX": [100.0, np.nan, 100.0, 100.0]}, index=BARS)
    volumes = pd.DataFrame({"XXX": [1e6, 0.0, 1e6, 1e6]}, index=BARS)
    targets = pd.DataFrame(np.nan, index=BARS, columns=["XXX"])
    targets.loc[B0] = [0.50]
    result = simulate(
        opens,
        closes,
        targets,
        cost_model=cost_model(zero_stack(), equity=10_000.0),
        volumes=volumes,
        events=[_event("halt", B1, {"start": "09:30", "end": None, "reason": "news"})],
        initial_cash=10_000.0,
    )
    assert result.fills["bar"].tolist() == [B2], "the halted bar fills nothing; the next one does"
    assert result.ledger.loc[B1, "pending_orders"] == 1
    assert result.ledger.loc[B1, "n_expired"] == 0
    assert result.ledger.loc[B1, "equity_close"] == pytest.approx(10_000.0, abs=1e-9)


def test_a_spinoff_is_refused_rather_than_dropped() -> None:
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 12.0}, cash=500.0)
    spinoff = _event("spinoff", B2, {"child_sid": "YYY", "ratio": 0.5, "cash_in_lieu": True})
    with pytest.raises(Exception, match="spinoff"):
        apply_events(state, bar=B2, events=[spinoff])


def test_a_trading_cessation_makes_the_holding_an_unresolved_claim() -> None:
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 12.0}, cash=500.0)
    after = apply_events(state, bar=B2, events=[_event("trading_cessation", B2, {"last_session": B1})]).state
    assert after.positions["XXX"].valuation_status == "unresolved_claim"
    assert after.is_incomplete
    marked = mark_at_close(after, bar=B2, prices=pd.Series(dtype=float), cost_model=None)
    assert marked.state.positions["XXX"].last_price == 12.0
    assert marked.state.equity(pd.Series(dtype=float)) == pytest.approx(1_700.0, abs=1e-12)


def test_every_applied_event_is_recorded_with_its_effective_date_and_source_revision() -> None:
    """The adversarial-session invariant: no event is applied without a record of it."""
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 12.0}, cash=500.0)
    events = [
        _distribution(B2, amount=0.25),
        _event("halt", B2, {"start": "09:30", "end": None, "reason": "news"}),
    ]
    outcome = apply_events(state, bar=B2, events=events)
    frame = outcome.frame()
    assert set(frame["event_type"]) == {"cash_distribution", "halt"}
    assert (frame["effective_date"] == B2).all()
    assert (frame["source"] == "test").all()
    assert (frame["revision"] == 1).all()


def test_an_event_dated_on_another_bar_is_not_applied() -> None:
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 12.0}, cash=500.0)
    outcome = apply_events(state, bar=B1, events=[_distribution(B2)])
    assert outcome.applied == ()
    assert outcome.state.receivables == {}


def test_the_price_basis_is_per_series_and_never_borrowed_from_another_name() -> None:
    """A dataset may mix an as-printed source with an adjusted one; the basis is per sid."""
    prices = {"RAW": 200.0, "ADJ": 200.0}
    state = state_from_shares(B1, {"RAW": 100.0, "ADJ": 100.0}, prices, cash=0.0)
    events = [_event("split", B2, {"factor": 2.0}, sid="RAW"), _event("split", B2, {"factor": 2.0}, sid="ADJ")]
    after = apply_events(
        state, bar=B2, events=events, price_basis={"RAW": "raw", "ADJ": "split_adjusted_at_ingest"}
    ).state
    assert after.shares("RAW") == 200.0 and after.positions["RAW"].last_price == 100.0
    assert after.shares("ADJ") == 100.0 and after.positions["ADJ"].last_price == 200.0
    assert after.carried_equity() == pytest.approx(state.carried_equity(), abs=1e-12)


def test_a_missing_price_basis_declaration_is_refused() -> None:
    state = state_from_shares(B1, {"XXX": 100.0}, {"XXX": 200.0}, cash=0.0)
    with pytest.raises(Exception, match="price_basis"):
        apply_events(state, bar=B2, events=[_event("split", B2, {"factor": 2.0})], price_basis={"OTHER": "raw"})


def test_a_spinoff_on_a_name_that_is_not_held_is_a_recorded_no_op() -> None:
    state = empty_state(B1, cash=500.0)
    outcome = apply_events(
        state, bar=B2, events=[_event("spinoff", B2, {"child_sid": "YYY", "ratio": 0.5, "cash_in_lieu": True})]
    )
    assert outcome.state.cash == 500.0
    assert outcome.applied[0].detail["child_sid"] == "YYY"
