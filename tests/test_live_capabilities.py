"""The dated capabilities table and the submission gate (§3.2 table, §3.9 revalidation).

The gate is the last thing that runs before an order leaves the process, so the
tests here are written around the failures that cost money silently: a capability
row that is stale or missing being read as permissive, a short authorized by an
omitted flag, and a batch submitted against buying power the account does not
have. Every one of them must be a *fatal finding on a report*, not an exception
the caller can accidentally swallow and not a silent pass.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import pytest

from prism.execution.portfolio_state import OrderIntent
from prism.live.broker import AccountIdentity, Broker, CapabilityRow
from prism.live.capabilities import (
    SubmissionCheck,
    capabilities_frame,
    capability_ages_hours,
    refresh_capabilities,
    revalidate_for_submission,
    stale_capability_sids,
)
from prism.portfolio.limits import LimitContext

REPO_ROOT = Path(__file__).resolve().parents[1]
AS_OF = "2026-09-08T16:00:00-04:00"


@pytest.fixture(scope="module")
def execution_profile() -> dict:
    """The shipped reference execution profile, parsed. Its `[capabilities]` block is
    the declaration this gate reads, so the test binds the file rather than a copy."""
    path = REPO_ROOT / "examples" / "reference" / "profiles" / "execution_alpaca.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _row(sid: str, valid_at: str = AS_OF, **overrides) -> CapabilityRow:
    fields = dict(
        sid=sid,
        valid_at=valid_at,
        fractionable=False,
        quantity_increment=1.0,
        min_notional=0.0,
        shortable=True,
        easy_to_borrow=True,
        lot_size=None,
        auction_eligible=True,
        restrictions=(),
        order_types=("market_on_open", "market_day"),
        sessions=("regular",),
        available_at=valid_at,
        ingested_at=valid_at,
        source="alpaca",
        revision=1,
    )
    fields.update(overrides)
    return CapabilityRow(**fields)


class FakeBroker(Broker):
    """A venue that answers the section 3.9 reads and counts them."""

    def __init__(self, *, rows=(), buying_power=1_000_000.0, cash=100_000.0, account_id="PA1"):
        self._rows = {row.sid: row for row in rows}
        self._buying_power = buying_power
        self._cash = cash
        self._account_id = account_id
        self.capability_calls: list[tuple[tuple[str, ...], str]] = []

    def positions(self) -> dict[str, float]:
        return {}

    def cash(self) -> float:
        return self._cash

    def submit(self, order) -> None:  # pragma: no cover - the gate submits nothing
        raise AssertionError("the gate never submits")

    def submitted_order_ids(self) -> set[str]:
        return set()

    def fills_for(self, client_order_ids):
        return []

    def account_identity(self) -> AccountIdentity:
        return AccountIdentity(account_id=self._account_id, endpoint="https://fake", is_paper=True)

    def buying_power(self) -> float:
        return self._buying_power

    def capabilities(self, sids, as_of):
        self.capability_calls.append((tuple(str(s) for s in sids), as_of))
        return [self._rows[str(sid)] for sid in sids if str(sid) in self._rows]


def _intent(sid: str, qty: float, price: float = 100.0) -> OrderIntent:
    return OrderIntent(
        sid=sid,
        qty=qty,
        increment=1.0,
        reference_price=price,
        decision_bar="2026-09-08",
        order_type="market_on_open",
        time_in_force="opg",
        action_class="C3",
    )


def _gate(broker, intents, *, profile, **kwargs) -> SubmissionCheck:
    prices = pd.Series({str(i.sid): float(i.reference_price) for i in intents}, dtype=float)
    return revalidate_for_submission(
        broker,
        intents,
        execution_profile=profile,
        ctx=LimitContext(w0=pd.Series(dtype=float), equity=100_000.0),
        prices=prices,
        as_of=AS_OF,
        now=AS_OF,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_the_frame_is_the_section_32_table_indexed_by_sid() -> None:
    frame = capabilities_frame([_row("AAA"), _row("BBB")])
    assert frame.index.name == "sid" and list(frame.index) == ["AAA", "BBB"]
    for column in ("valid_at", "available_at", "ingested_at", "source", "revision"):
        assert column in frame.columns
    # The list columns are JSON-encoded, which is the form validate.capability_row parses.
    from prism.portfolio.validate import capability_row

    assert capability_row(frame, "AAA")["order_types"] == ("market_on_open", "market_day")


def test_two_rows_for_one_sid_are_refused_rather_than_resolved_here() -> None:
    with pytest.raises(ValueError, match="as-of selection"):
        capabilities_frame([_row("AAA"), _row("AAA", valid_at="2026-09-07T16:00:00-04:00")])


def test_an_empty_pull_is_an_empty_table_not_an_error() -> None:
    assert capabilities_frame([]).empty


def test_refresh_pulls_through_the_broker(execution_profile) -> None:
    broker = FakeBroker(rows=[_row("AAA")])
    frame = refresh_capabilities(broker, ["AAA"], as_of=AS_OF)
    assert list(frame.index) == ["AAA"]
    assert broker.capability_calls == [(("AAA",), AS_OF)]


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


def test_age_is_measured_from_valid_at() -> None:
    frame = capabilities_frame([_row("AAA", valid_at="2026-09-08T04:00:00-04:00")])
    ages = capability_ages_hours(frame, now="2026-09-08T16:00:00-04:00")
    assert ages["AAA"] == pytest.approx(12.0)
    assert stale_capability_sids(frame, now="2026-09-08T16:00:00-04:00", max_age_hours=24.0) == ()
    assert stale_capability_sids(frame, now="2026-09-08T16:00:00-04:00", max_age_hours=6.0) == ("AAA",)


def test_an_unreadable_valid_at_is_infinitely_old_not_fresh() -> None:
    frame = capabilities_frame([_row("AAA", valid_at="whenever")])
    assert capability_ages_hours(frame, now=AS_OF)["AAA"] == float("inf")
    assert stale_capability_sids(frame, now=AS_OF, max_age_hours=1e9) == ("AAA",)


def test_a_negative_max_age_is_refused_rather_than_read_as_never_stale() -> None:
    frame = capabilities_frame([_row("AAA")])
    with pytest.raises(ValueError, match="max_age_hours"):
        stale_capability_sids(frame, now=AS_OF, max_age_hours=-1.0)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_a_clean_batch_passes_and_records_what_it_checked(execution_profile) -> None:
    broker = FakeBroker(rows=[_row("AAA")], buying_power=50_000.0)
    check = _gate(broker, [_intent("AAA", 10.0)], profile=execution_profile, expect_account_id="PA1")
    assert check.ok and check.report.ok
    assert check.refreshed is True
    assert check.required_buying_power == pytest.approx(1000.0)
    assert check.buying_power == 50_000.0
    record = check.to_dict()
    assert record["sids"] == ["AAA"] and record["account"]["account_id"] == "PA1"


def test_the_profile_forces_a_refresh_even_when_rows_were_supplied(execution_profile) -> None:
    # `[capabilities] revalidate_at_submission = true` in the shipped profile: the
    # rows the batch was constructed against are not the rows it is submitted against.
    assert execution_profile["capabilities"]["revalidate_at_submission"] is True
    broker = FakeBroker(rows=[_row("AAA")])
    stale_rows = capabilities_frame([_row("AAA", valid_at="2026-01-02T16:00:00-05:00")])
    check = _gate(broker, [_intent("AAA", 1.0)], profile=execution_profile, capabilities=stale_rows)
    assert check.refreshed is True and broker.capability_calls
    assert check.ok


def test_supplied_rows_are_reused_only_while_they_are_inside_max_age_hours() -> None:
    profile = {
        "quantity_increment_default": 1.0,
        "min_notional": 1.0,
        "order_types": ["market_on_open"],
        "capabilities": {"source": "broker", "max_age_hours": 24, "revalidate_at_submission": False},
    }
    broker = FakeBroker(rows=[_row("AAA")])
    fresh = capabilities_frame([_row("AAA", valid_at="2026-09-08T09:30:00-04:00")])
    reused = _gate(broker, [_intent("AAA", 1.0)], profile=profile, capabilities=fresh)
    assert reused.refreshed is False and broker.capability_calls == []

    stale = capabilities_frame([_row("AAA", valid_at="2026-09-01T09:30:00-04:00")])
    refetched = _gate(broker, [_intent("AAA", 1.0)], profile=profile, capabilities=stale)
    assert refetched.refreshed is True and broker.capability_calls


def test_a_sid_with_no_dated_row_fails_the_batch(execution_profile) -> None:
    # The delisted-mid-session case: the venue no longer knows the name, so no row
    # comes back. That must fail the batch, not pass it unchecked.
    broker = FakeBroker(rows=[_row("AAA")])
    check = _gate(broker, [_intent("AAA", 1.0), _intent("GONE", 1.0)], profile=execution_profile)
    assert not check.ok
    checks = {finding.check for finding in check.report.fatal_findings}
    assert "capabilities" in checks


def test_a_batch_beyond_the_venues_buying_power_is_refused(execution_profile) -> None:
    broker = FakeBroker(rows=[_row("AAA")], buying_power=500.0)
    check = _gate(broker, [_intent("AAA", 10.0)], profile=execution_profile)
    assert not check.ok
    finding = next(f for f in check.report.fatal_findings if f.check == "buying_power")
    assert "1000.00" in finding.detail and "500.00" in finding.detail


def test_closing_a_long_consumes_no_buying_power_and_opening_a_short_does(execution_profile) -> None:
    broker = FakeBroker(rows=[_row("AAA"), _row("BBB")], buying_power=100.0)
    check = _gate(
        broker,
        [_intent("AAA", -10.0)],
        profile=execution_profile,
        positions={"AAA": 10.0},
    )
    assert check.required_buying_power == pytest.approx(0.0)

    opening = _gate(broker, [_intent("BBB", -3.0)], profile=execution_profile, positions={})
    assert opening.required_buying_power == pytest.approx(300.0)
    assert not opening.ok  # 300 > the venue's 100


def test_a_batch_for_another_account_is_refused(execution_profile) -> None:
    broker = FakeBroker(rows=[_row("AAA")], account_id="PA9")
    check = _gate(broker, [_intent("AAA", 1.0)], profile=execution_profile, expect_account_id="PA1")
    assert not check.ok
    assert any(f.check == "account_identity" for f in check.report.fatal_findings)


def test_the_gate_refuses_a_short_the_capability_row_does_not_permit(execution_profile) -> None:
    broker = FakeBroker(rows=[_row("AAA", shortable=False, easy_to_borrow=False)])
    check = _gate(broker, [_intent("AAA", -5.0)], profile=execution_profile, positions={})
    assert not check.ok
    assert any("shortable" in f.detail or "borrow" in f.detail for f in check.report.fatal_findings)


def test_the_gate_refuses_an_order_type_the_row_does_not_admit(execution_profile) -> None:
    broker = FakeBroker(rows=[_row("AAA", order_types=("market_day",), auction_eligible=False)])
    check = _gate(broker, [_intent("AAA", 1.0)], profile=execution_profile)
    assert not check.ok
    assert any(f.check == "order_type" for f in check.report.fatal_findings)


def test_a_halted_name_is_refused_by_its_restriction(execution_profile) -> None:
    broker = FakeBroker(rows=[_row("AAA", restrictions=("halted",))])
    check = _gate(broker, [_intent("AAA", 1.0)], profile=execution_profile)
    assert not check.ok


def test_two_intents_for_one_name_pull_one_capability_row(execution_profile) -> None:
    # The table is keyed (sid, valid_at); pulling the same name twice would build a
    # frame the as-of selection rule then has to arbitrate, which is not this
    # function's job.
    broker = FakeBroker(rows=[_row("AAA")])
    check = _gate(broker, [_intent("AAA", 3.0), _intent("AAA", 2.0)], profile=execution_profile)
    assert broker.capability_calls == [(("AAA",), AS_OF)]
    assert list(check.capabilities.index) == ["AAA"]
