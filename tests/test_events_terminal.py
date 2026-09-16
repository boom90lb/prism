"""Missing terminal values and the claims they leave behind (engine spec §3.2, §3.6; §6 stage 3).

The acceptance case this file carries: *stale holding, unresolved claim,
incomplete strict claim; no earlier close is overwritten by an eventual
payout.*

The failure it is written to catch is a silent one. When a name stops trading,
three states look identical to anything that only reads bars — the claim
settled at zero, the claim settled at a value nobody has published yet, and the
claim was never adjudicated at all. Booking any of them as zero produces a
number that is arithmetically consistent, propagates into equity, return and
every downstream ratio, and is wrong by the whole position. So
:class:`~prism.io.events.TerminalClaim` distinguishes the three, refuses to
produce cash for the two that have no value, and hands a caller the sentence a
result record's ``incomplete_reasons`` carries.

The read side is here; applying the transition to portfolio state is §3.6 and
arrives with ``execution/accounting.py`` at stage 3b. What this file fixes is
what that kernel is allowed to believe.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from prism.io.dataset import Dataset, DatasetError
from prism.io.events import (
    CLAIM_STATUSES,
    Event,
    EventError,
    TerminalClaim,
    events_frame,
    incomplete_reasons,
    terminal_claim,
    terminal_claims,
)
from prism.io.synthetic import scenario_case, write_synthetic_dataset

SID = "sec_000001"
OTHER = "sec_000002"
INGESTED = "2026-03-20T20:00:00-04:00"
SOURCE = "synthetic"

START = dt.date(2026, 1, 1)
END = dt.date(2026, 12, 31)
VINTAGE = "synthetic_v1"


def _event(event_type: str, effective: str, payload: dict, sid: str = SID, revision: int = 1) -> Event:
    return Event(
        sid=sid,
        event_type=event_type,
        effective_date=effective,
        payload=payload,
        ingested_at=INGESTED,
        source=SOURCE,
        revision=revision,
    )


NOTICE = _event("delisting_notice", "2026-03-02", {"reason": "going private", "expected_last_session": "2026-03-13"})
CESSATION = _event("trading_cessation", "2026-03-13", {"last_session": "2026-03-13"})
UNKNOWN_PAYOUT = _event("terminal_payout", "2026-03-13", {"status": "unknown"})
PAID = _event("terminal_payout", "2026-03-27", {"amount_per_share": 12.5, "currency": "USD", "pay_date": "2026-03-27"})


# ---- the four states -------------------------------------------------------


def test_a_sid_with_no_terminal_event_is_still_trading() -> None:
    claim = terminal_claim(events_frame([]), SID)
    assert claim.status == "trading" and claim.complete
    assert claim.incomplete_reason() is None
    assert claim.valuation_status == "priced"


def test_a_cessation_with_no_payout_leaves_an_open_claim() -> None:
    claim = terminal_claim(events_frame([NOTICE, CESSATION]), SID)
    assert claim.status == "open"
    assert not claim.complete
    assert claim.last_session == "2026-03-13"
    assert claim.reason == "going private"
    assert claim.valuation_status == "unresolved_claim"
    assert "no terminal_payout recorded" in str(claim.incomplete_reason())


def test_a_payout_declaring_the_value_unknown_leaves_an_unresolved_claim() -> None:
    """An explicit "nobody knows yet" is better provenance than silence, and no better a number."""
    claim = terminal_claim(events_frame([NOTICE, CESSATION, UNKNOWN_PAYOUT]), SID)
    assert claim.status == "unresolved"
    assert not claim.complete
    assert claim.amount_per_share is None
    assert "status 'unknown'" in str(claim.incomplete_reason())


def test_a_payout_with_an_amount_resolves_the_claim() -> None:
    claim = terminal_claim(events_frame([NOTICE, CESSATION, PAID]), SID)
    assert claim.status == "resolved" and claim.complete
    assert claim.amount_per_share == pytest.approx(12.5)
    assert claim.currency == "USD"
    assert claim.pay_date == "2026-03-27" and not claim.pay_date_assumed
    assert claim.last_session == "2026-03-13", "resolving the claim does not forget when trading stopped"


def test_a_zero_payout_is_a_resolved_wipeout_and_not_an_unknown() -> None:
    """Zero is a terminal value someone published; unknown is the absence of one."""
    wipeout = _event(
        "terminal_payout", "2026-03-27", {"amount_per_share": 0.0, "currency": "USD", "pay_date": "2026-03-27"}
    )
    claim = terminal_claim(events_frame([CESSATION, wipeout]), SID)
    assert claim.status == "resolved" and claim.complete
    assert claim.cash_for(100.0) == 0.0


def test_a_later_amendment_supersedes_an_earlier_unknown() -> None:
    claim = terminal_claim(events_frame([CESSATION, UNKNOWN_PAYOUT, PAID]), SID)
    assert claim.status == "resolved"
    assert claim.effective_date == "2026-03-27"


def test_an_otc_continuation_is_recorded_and_does_not_end_the_claim() -> None:
    """A name that kept trading over the counter has not been written off."""
    moved = _event("otc_continuation", "2026-03-16", {"venue": "OTCM", "vendor_symbol": "ACMEQ"})
    claim = terminal_claim(events_frame([NOTICE, CESSATION, moved]), SID)
    assert claim.continues_at == "OTCM"
    assert claim.status == "open", "the claim is open, not resolved: nobody has said what it is worth"


def test_every_status_the_reader_can_return_is_declared() -> None:
    produced = {
        terminal_claim(events_frame(rows), SID).status
        for rows in ([], [CESSATION], [CESSATION, UNKNOWN_PAYOUT], [CESSATION, PAID])
    }
    assert produced == set(CLAIM_STATUSES)


def test_an_undeclared_status_is_refused() -> None:
    with pytest.raises(EventError, match="claim status must be one of"):
        TerminalClaim(sid=SID, status="settled")  # type: ignore[arg-type]


# ---- cash, and the refusal that keeps it honest ----------------------------


def test_a_resolved_claim_converts_a_signed_share_count_to_cash() -> None:
    claim = terminal_claim(events_frame([CESSATION, PAID]), SID)
    assert claim.cash_for(100.0) == pytest.approx(1250.0)
    assert claim.cash_for(-100.0) == pytest.approx(-1250.0), "a short pays the terminal value"


@pytest.mark.parametrize("rows", ([CESSATION], [CESSATION, UNKNOWN_PAYOUT]))
def test_an_unpriced_claim_refuses_to_produce_cash_rather_than_settling_at_zero(rows: list[Event]) -> None:
    """Returning 0.0 here would be indistinguishable from a resolved wipeout in every total."""
    claim = terminal_claim(events_frame(rows), SID)
    assert not claim.complete
    with pytest.raises(EventError, match="recorded incomplete"):
        claim.cash_for(100.0)


def test_a_still_trading_sid_has_no_terminal_value_to_convert() -> None:
    claim = terminal_claim(events_frame([]), SID)
    with pytest.raises(EventError, match="the terminal value is trading"):
        claim.cash_for(100.0)


def test_a_non_finite_share_count_is_refused() -> None:
    claim = terminal_claim(events_frame([CESSATION, PAID]), SID)
    with pytest.raises(EventError, match="share count must be finite"):
        claim.cash_for(float("nan"))


# ---- the strict-claim rule -------------------------------------------------


def test_incomplete_reasons_is_empty_only_when_every_claim_is_complete() -> None:
    resolved = terminal_claim(events_frame([CESSATION, PAID]), SID)
    unresolved = terminal_claim(events_frame([CESSATION, UNKNOWN_PAYOUT]), SID)
    assert incomplete_reasons([resolved]) == ()
    assert incomplete_reasons([resolved, unresolved]) == (unresolved.incomplete_reason(),)
    assert len(incomplete_reasons([unresolved, unresolved])) == 2


def test_claims_are_read_per_sid_and_do_not_leak_across_names() -> None:
    frame = events_frame(
        [CESSATION, PAID, _event("trading_cessation", "2026-04-01", {"last_session": "2026-04-01"}, sid=OTHER)]
    )
    claims = terminal_claims(frame, [SID, OTHER])
    assert claims[SID].status == "resolved"
    assert claims[OTHER].status == "open"
    assert incomplete_reasons(claims) == (claims[OTHER].incomplete_reason(),)


# ---- through the as-of view, on the synthetic delisting scenario -----------


@pytest.fixture(scope="module")
def delisted(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("terminal") / "data"
    manifest = write_synthetic_dataset(
        root, seed=7, scenarios=["delisting"], n_equities=14, n_etfs=3, start=START, end=END
    )
    return Dataset.open(root), manifest


def test_the_payout_is_invisible_to_a_decision_taken_before_it_landed(delisted) -> None:
    """A claim the decision could not know was resolved must not read as resolved."""
    ds, manifest = delisted
    case = scenario_case(manifest, "delisting")
    paid = case.sid("paid")
    early = ds.view("research_vintage", case.fill_bar, vintage=VINTAGE).terminal_claims([paid])[paid]
    assert early.status == "open"
    with pytest.raises(EventError, match="recorded incomplete"):
        early.cash_for(100.0)
    late = ds.view("research_vintage", case.date("pay_date"), vintage=VINTAGE).terminal_claims([paid])[paid]
    assert late.status == "resolved"


def test_the_eventual_payout_overwrites_no_earlier_close(delisted) -> None:
    ds, manifest = delisted
    case = scenario_case(manifest, "delisting")
    paid = case.sid("paid")
    last_session, last_close = case.date("last_session"), float(case.value("last_close"))
    payout = float(case.value("amount_per_share"))
    assert payout != pytest.approx(last_close), "the fixture must distinguish a payout from a carried close"

    before = ds.view("research_vintage", case.fill_bar, vintage=VINTAGE)
    after = ds.view("research_vintage", case.date("pay_date"), vintage=VINTAGE)
    stamp = pd.Timestamp(last_session, tz="America/New_York")
    assert float(before.panel([paid], last_session, "close").loc[stamp, paid]) == pytest.approx(last_close)
    assert float(after.panel([paid], last_session, "close").loc[stamp, paid]) == pytest.approx(last_close)
    marks = after.marks([paid])
    assert marks.loc[paid, "last_price"] == pytest.approx(last_close)
    assert marks.loc[paid, "last_price_bar"] == last_session
    assert marks.loc[paid, "valuation_status"] == "stale"


def test_the_unresolved_holding_is_carried_stale_and_never_at_zero(delisted) -> None:
    """The whole acceptance case in one assertion set: stale mark, unresolved claim, incomplete record."""
    ds, manifest = delisted
    case = scenario_case(manifest, "delisting")
    unknown = case.sid("unknown")
    view = ds.view("research_vintage", case.date("pay_date"), vintage=VINTAGE)
    marks = view.marks([unknown])
    assert marks.loc[unknown, "valuation_status"] == "stale"
    assert marks.loc[unknown, "last_price"] > 0.0
    assert marks.loc[unknown, "last_price_bar"] == case.date("last_session")
    claims = view.terminal_claims([unknown])
    assert claims[unknown].status == "unresolved"
    assert incomplete_reasons(claims), "a strict claim on an unknown terminal value is recorded incomplete"


def test_a_dataset_with_no_events_table_refuses_rather_than_reading_every_claim_as_trading(
    tmp_path: Path,
) -> None:
    """Silence is not evidence that a name still trades."""
    root = tmp_path / "no_events"
    manifest = write_synthetic_dataset(
        root, seed=7, scenarios=["costs"], n_equities=14, n_etfs=3, start=START, end=END
    )
    assert not any(path == "events.parquet" for path, _ in manifest.files)
    view = Dataset.open(root).view("research_vintage", "2026-06-15", vintage=VINTAGE)
    with pytest.raises(DatasetError, match="lists no events table"):
        view.terminal_claims(["sec_000001"])
