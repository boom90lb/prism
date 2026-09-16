"""The §3.2 synthetic scenario set, applied through the data layer (engine spec §3.2, stage 3a).

``tests/test_synthetic.py`` asserts that the generator *writes* each scenario.
This file asserts that the data layer *answers* each one: every case is opened
as a :class:`~prism.io.dataset.Dataset`, read through a
:class:`~prism.io.dataset.DatasetView` at the dates the scenario turns on, and
checked against what §3.2 and §3.6 say the answer is. A fixture that is only
written is a fixture nobody has ever run.

Each scenario is addressed by id through the manifest's scenario catalog
(``prism.io.synthetic.scenario_case``), never by a sid a test rediscovered, so
the stage 3b adversarial session set can drive the same cases through the
canonical simulator without re-deriving which sid plays which role.

Five of these are where a trading system loses money quietly, and they are
asserted hardest:

* a halt leaves the holding priced at its last bar, ``stale``, never at zero
  and never sold;
* a delisting with no terminal payout leaves an unresolved claim, and a strict
  claim needing that unknown value is recorded incomplete;
* a terminal payout converts the claim to cash and never overwrites an earlier
  close;
* a null-valued full revision means null, not the previous value;
* a reused ticker resolves to two sids and never merges two holdings.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.execution.cost_model import SPREAD_BUCKET_SCHEDULE_V1, bucket_spread_bps
from prism.execution.participation import participation_capped_targets
from prism.io import sessions
from prism.io.dataset import Dataset, DatasetError, DatasetView
from prism.io.events import (
    RebaseNotRecordedError,
    applied_share_factor,
    distribution_entitlement,
    incomplete_reasons,
    series_price_basis,
    validate_bars_revision,
)
from prism.io.manifest import Manifest
from prism.io.synthetic import (
    ALL_SCENARIOS,
    EQUITY_INDEX,
    GAP_FLAG_SESSIONS,
    REBASE_FACTOR,
    SPLIT_FACTOR,
    ScenarioCase,
    SyntheticError,
    scenario_case,
    scenario_catalog,
    write_synthetic_dataset,
)

START = dt.date(2026, 1, 1)
END = dt.date(2026, 12, 31)
SEED = 7
VINTAGE = "synthetic_v1"
NIGHTLY = "21:30 America/New_York"

# The two visibility modes, parametrized wherever a scenario's answer must be
# the same under both. They disagree only where the cutoffs straddle a
# publication instant, and each such case says so explicitly.
MODES = ("research_vintage", "replay_recorded")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Dataset, Manifest]:
    root = tmp_path_factory.mktemp("scenarios") / "data"
    manifest = write_synthetic_dataset(root, seed=SEED, scenarios="all", n_equities=14, n_etfs=3, start=START, end=END)
    return Dataset.open(root), manifest


def _view(dataset: Dataset, day: str, mode: str = "research_vintage") -> DatasetView:
    if mode == "research_vintage":
        return dataset.view("research_vintage", day, vintage=VINTAGE)
    return dataset.view("replay_recorded", day, decision_time=NIGHTLY)


def _case(manifest: Manifest, scenario: str) -> ScenarioCase:
    return scenario_case(manifest, scenario)


def _close(view: DatasetView, sid: str, day: str) -> float:
    frame = view.bars([sid], day)
    row = frame.loc[frame["ts"].dt.tz_convert("America/New_York").dt.date == dt.date.fromisoformat(day)]
    assert len(row) == 1, f"{sid} has {len(row)} visible rows on {day}"
    return float(row.iloc[0]["close"])


def _before(day: str, n: int = 1) -> str:
    out = dt.date.fromisoformat(day)
    for _ in range(n):
        out = sessions.previous_session(out)
    return out.isoformat()


# ---- the catalog itself ----------------------------------------------------


def test_every_scenario_the_generator_wrote_is_addressable_by_id(dataset: tuple[Dataset, Manifest]) -> None:
    """The contract stage 3b builds its adversarial session set on."""
    _, manifest = dataset
    catalog = scenario_catalog(manifest)
    assert sorted(catalog) == sorted(ALL_SCENARIOS)
    for name, case in catalog.items():
        assert case.scenario == name
        assert case.sids, f"{name} casts no sid"
        assert case.expects, f"{name} states no expectation"


def test_each_case_names_three_ascending_sessions_inside_the_sample(dataset: tuple[Dataset, Manifest]) -> None:
    """The three-bar window of §5: decision close t, fill at t+1's open, mark at t+1's close."""
    _, manifest = dataset
    first, last = (dt.date.fromisoformat(day) for day in manifest.coverage["sessions"])
    for name, case in scenario_catalog(manifest).items():
        bars = [dt.date.fromisoformat(day) for day in (case.prior_bar, case.decision_bar, case.fill_bar)]
        assert bars == sorted(bars) and len(set(bars)) == 3, f"{name} window {bars} is not three ascending sessions"
        for day in bars:
            assert sessions.is_session(day), f"{name} names {day}, which is not a session"
            assert first <= day <= last, f"{name} names {day}, outside the sample"


def test_a_subset_registers_only_the_scenarios_it_wrote(tmp_path: Path) -> None:
    manifest = write_synthetic_dataset(
        tmp_path / "subset", seed=SEED, scenarios=["halt", "delisting"], n_equities=14, n_etfs=3, start=START, end=END
    )
    assert sorted(scenario_catalog(manifest)) == ["delisting", "halt"]
    with pytest.raises(SyntheticError, match="wrote no 'rebase' scenario"):
        scenario_case(manifest, "rebase")


def test_a_market_manifest_is_refused_rather_than_answered_empty(dataset: tuple[Dataset, Manifest]) -> None:
    """An empty catalog would read as "no scenario failed" rather than "no scenario ran"."""
    _, manifest = dataset
    market = Manifest(
        provenance="market",
        as_of=manifest.as_of,
        files=manifest.files,
        sources=manifest.sources,
        source_priority=manifest.source_priority,
        universe=manifest.universe,
    )
    with pytest.raises(SyntheticError, match="only a synthetic dataset carries"):
        scenario_catalog(market)


# ---- membership ------------------------------------------------------------


def test_membership_admits_the_addition_only_from_its_effective_date(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "membership")
    added = case.sid("added")
    announce, effective = case.date("add_announce"), case.date("add_effective")
    assert added not in _view(ds, announce).membership(EQUITY_INDEX)
    assert added not in _view(ds, _before(effective)).membership(EQUITY_INDEX)
    assert added in _view(ds, effective).membership(EQUITY_INDEX)


def test_membership_removal_leaves_the_index_without_touching_the_bars(dataset: tuple[Dataset, Manifest]) -> None:
    """Leaving an index is not a sale: the data layer stops listing the name, nothing more."""
    ds, manifest = dataset
    case = _case(manifest, "membership")
    removed = case.sid("removed")
    last_in, out = case.date("last_member_session"), case.date("remove_effective")
    assert removed in _view(ds, last_in).membership(EQUITY_INDEX)
    assert removed not in _view(ds, out).membership(EQUITY_INDEX)
    marks = _view(ds, out).marks([removed])
    assert marks.loc[removed, "valuation_status"] == "priced"
    assert marks.loc[removed, "last_price"] > 0.0


# ---- split, against both price bases (decision D3) -------------------------


def test_the_raw_split_halves_the_print_and_doubles_the_shares(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "split")
    raw = case.sid("raw")
    effective = case.date("effective")
    view = _view(ds, effective)
    basis = series_price_basis(view.bars([raw], effective))
    assert basis == "raw"
    before, after = _close(view, raw, _before(effective)), _close(view, raw, effective)
    assert after / before < 0.6, "a raw series prints the post-split level on the effective date"

    shares = 100.0
    factor = applied_share_factor("split", basis, SPLIT_FACTOR)
    assert factor == SPLIT_FACTOR
    # Equity is continuous across the split: the halved print times the doubled
    # share count is the pre-split value, up to the session's own return.
    assert (shares * factor * after) / (shares * before) == pytest.approx(1.0, rel=0.1)


def test_the_adjusted_split_moves_no_shares_and_prints_continuously(dataset: tuple[Dataset, Manifest]) -> None:
    """The error D3 exists to prevent: applying the ratio twice on an already-rebased series."""
    ds, manifest = dataset
    case = _case(manifest, "split")
    adjusted = case.sid("adjusted")
    effective = case.date("effective")
    view = _view(ds, effective)
    basis = series_price_basis(view.bars([adjusted], effective))
    assert basis == "split_adjusted_at_ingest"
    before, after = _close(view, adjusted, _before(effective)), _close(view, adjusted, effective)
    assert after / before == pytest.approx(1.0, rel=0.1), "the vendor already rebased this series"
    assert applied_share_factor("split", basis, SPLIT_FACTOR) == 1.0


# ---- cash distribution -----------------------------------------------------


def test_the_distribution_is_invisible_before_its_announcement(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "distribution")
    payer, ex_date = case.sid("payer"), case.date("ex_date")
    early = _before(case.date("announce"), 2)
    assert _view(ds, early).events([payer], early, early).empty
    visible = _view(ds, ex_date).events([payer], ex_date, ex_date)
    assert list(visible["event_type"]) == ["cash_distribution"]


def test_entitlement_is_the_share_count_at_the_close_before_the_ex_date(dataset: tuple[Dataset, Manifest]) -> None:
    """§3.6's Correction: an ex-date-open buyer is not entitled, an ex-date-open seller is."""
    ds, manifest = dataset
    case = _case(manifest, "distribution")
    payer, ex_date = case.sid("payer"), case.date("ex_date")
    row = _view(ds, ex_date).events([payer], ex_date, ex_date).iloc[0]
    holder = distribution_entitlement(row, 100.0)
    assert holder.amount == pytest.approx(100.0 * holder.amount_per_share)
    assert holder.leg == "receivable"
    assert holder.pay_date == case.date("pay_date") and not holder.pay_date_assumed
    buyer = distribution_entitlement(row, 0.0)
    assert buyer.amount == 0.0 and buyer.leg == "none"
    shorter = distribution_entitlement(row, -100.0)
    assert shorter.amount < 0.0 and shorter.leg == "liability"


def test_total_return_reinvests_on_the_ex_date_and_the_as_traded_close_does_not(
    dataset: tuple[Dataset, Manifest],
) -> None:
    ds, manifest = dataset
    case = _case(manifest, "distribution")
    payer, ex_date = case.sid("payer"), case.date("ex_date")
    view = _view(ds, ex_date)
    price = view.panel([payer], ex_date, "close")
    total = view.total_return([payer], ex_date)
    assert list(price.index) == list(total.index)
    before = pd.Timestamp(_before(ex_date), tz="America/New_York")
    on = pd.Timestamp(ex_date, tz="America/New_York")
    assert float(total.loc[before, payer]) == pytest.approx(float(price.loc[before, payer]))
    assert float(total.loc[on, payer]) > float(price.loc[on, payer])


# ---- halt: stale, never zero, never sold -----------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_halt_carries_the_holding_at_its_last_bar_and_never_at_zero(
    dataset: tuple[Dataset, Manifest], mode: str
) -> None:
    ds, manifest = dataset
    case = _case(manifest, "halt")
    sid = case.sid("halted")
    last_priced, halt_start, halt_end = (case.date(k) for k in ("last_priced", "halt_start", "halt_end"))
    reference = _close(_view(ds, last_priced, mode), sid, last_priced)

    for day, expected_stale in ((halt_start, 1), (halt_end, 3)):
        view = _view(ds, day, mode)
        marks = view.marks([sid])
        assert marks.loc[sid, "valuation_status"] == "stale"
        assert marks.loc[sid, "last_price"] == pytest.approx(reference)
        assert marks.loc[sid, "last_price_bar"] == last_priced
        assert marks.loc[sid, "sessions_stale"] == expected_stale
        # Nothing is forward-filled into the panel and nothing is zero: the
        # halted session simply has no row.
        panel = view.panel([sid], day, "close")
        assert pd.Timestamp(day, tz="America/New_York") not in panel.index


def test_the_halt_gap_stays_below_the_flag_and_the_recovery_gap_does_not(dataset: tuple[Dataset, Manifest]) -> None:
    """ "The gap flag fires above 15 sessions and not below" is a two-sided claim."""
    _, manifest = dataset
    halted = _case(manifest, "halt").sid("halted")
    gapped = _case(manifest, "recovery").sid("gapped")
    gaps = {entry["sid"]: entry for entry in manifest.coverage["bar_gaps"]}
    assert gaps[halted]["missing_sessions"] <= GAP_FLAG_SESSIONS and gaps[halted]["flagged"] is False
    assert gaps[gapped]["missing_sessions"] > GAP_FLAG_SESSIONS and gaps[gapped]["flagged"] is True
    flagged = {entry["sid"] for entry in manifest.coverage["gap_flags"]}
    assert flagged == {gapped}


# ---- costs and rounding ----------------------------------------------------


def test_the_cost_names_land_one_in_each_step_of_the_reference_schedule(dataset: tuple[Dataset, Manifest]) -> None:
    """Dollar volume is computed from the view's own close and volume, not from a stored column."""
    ds, manifest = dataset
    case = _case(manifest, "costs")
    day = case.decision_bar
    view = _view(ds, day)
    sids = [case.sid(role) for role in ("widest_volume", "wide_volume", "mid_volume", "thin_volume")]
    close = view.panel(sids, day, "close")
    volume = view.panel(sids, day, "volume")
    median = (close * volume).tail(20).median()
    assert list(bucket_spread_bps(median, schedule=SPREAD_BUCKET_SCHEDULE_V1)) == [1.0, 2.0, 5.0, 10.0]
    assert float(median[sids[-1]]) < 5.0e6, "the thin name must be thin enough for a participation cap to bind"


def test_the_participation_bound_binds_on_the_thin_name_and_not_on_the_widest(
    dataset: tuple[Dataset, Manifest],
) -> None:
    """The second half of the costs scenario: a bound that actually binds on this fixture."""
    ds, manifest = dataset
    case = _case(manifest, "costs")
    day = case.decision_bar
    thin, widest = case.sid("thin_volume"), case.sid("widest_volume")
    view = _view(ds, day)
    sids = [widest, thin]
    close = view.panel(sids, day, "close")
    volume = view.panel(sids, day, "volume")
    adv = (close * volume).tail(20).median()

    aum, cap, target = 1_000_000.0, 0.05, 0.10
    capped = participation_capped_targets(pd.Series(0.0, index=sids), pd.Series(target, index=sids), adv, aum, cap)
    assert capped[widest] == pytest.approx(target), "a mega-cap name is not participation constrained"
    assert capped[thin] < target, "the thin name's 5 percent-of-ADV bound must bind"
    assert capped[thin] == pytest.approx(cap * float(adv[thin]) / aum)


def test_the_quantity_increment_is_dated_and_never_assumed_current(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "rounding")
    sid, change = case.sid("rounded"), case.date("increment_change")
    before = _view(ds, _before(change)).capabilities([sid])
    after = _view(ds, change).capabilities([sid])
    assert float(before.loc[sid, "quantity_increment"]) == 1.0
    assert float(after.loc[sid, "quantity_increment"]) == 5.0
    assert not bool(after.loc[sid, "fractionable"])


# ---- spin-off --------------------------------------------------------------


def test_the_spun_off_child_does_not_exist_before_its_effective_date(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "spinoff")
    child, effective = case.sid("child"), case.date("effective")
    before = _view(ds, _before(effective))
    assert before.bars([child], _before(effective)).empty
    with pytest.raises(DatasetError, match="no priced bar"):
        before.marks([child])
    on = _view(ds, effective)
    marks = on.marks([child])
    assert marks.loc[child, "valuation_status"] == "priced"
    assert marks.loc[child, "last_price_bar"] == effective


def test_the_spin_off_creates_a_second_holding_and_overwrites_no_parent_close(
    dataset: tuple[Dataset, Manifest],
) -> None:
    ds, manifest = dataset
    case = _case(manifest, "spinoff")
    parent, child, effective = case.sid("parent"), case.sid("child"), case.date("effective")
    decision = case.decision_bar
    payload = _view(ds, effective).events([parent], effective, effective).iloc[0]
    assert payload["event_type"] == "spinoff"
    body = pd.read_json(pd.io.common.StringIO(str(payload["payload"])), typ="series")
    assert str(body["child_sid"]) == child
    assert bool(body["cash_in_lieu"]) is True
    # The parent's decision close is the same before and after the distribution:
    # the child is a new position, not a restatement of the parent's history.
    assert _close(_view(ds, decision), parent, decision) == _close(_view(ds, effective), parent, decision)
    marks = _view(ds, effective).marks([parent, child])
    assert set(marks.index) == {parent, child}
    assert marks.loc[parent, "last_price"] != marks.loc[child, "last_price"]


# ---- ticker reuse: two sids, never one holding -----------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_reused_ticker_resolves_to_one_sid_per_date_and_two_across_the_sample(
    dataset: tuple[Dataset, Manifest], mode: str
) -> None:
    ds, manifest = dataset
    case = _case(manifest, "ticker_reuse")
    first, second = case.sid("first"), case.sid("second")
    cease, reopen, ticker = case.date("cease"), case.date("reopen"), case.value("ticker")
    assert _view(ds, cease, mode).resolve(ticker, cease) == first
    assert _view(ds, reopen, mode).resolve(ticker, reopen) == second
    assert first != second


def test_the_two_instruments_sharing_a_ticker_never_merge_into_one_holding(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "ticker_reuse")
    first, second, cease, reopen = (case.sid("first"), case.sid("second"), case.date("cease"), case.date("reopen"))
    late = case.fill_bar
    view = _view(ds, late)
    bars = view.bars([first, second], late)
    ts_first = set(bars.loc[bars["sid"] == first, "ts"])
    ts_second = set(bars.loc[bars["sid"] == second, "ts"])
    assert ts_first and ts_second and not (ts_first & ts_second), "the two series must not overlap"
    marks = view.marks([first, second])
    assert marks.loc[first, "last_price_bar"] == cease
    assert marks.loc[first, "valuation_status"] == "stale"
    assert marks.loc[second, "last_price_bar"] == reopen
    assert marks.loc[first, "last_price"] != marks.loc[second, "last_price"]


# ---- revision: a correction, then a withdrawal -----------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_late_correction_is_invisible_until_it_lands(dataset: tuple[Dataset, Manifest], mode: str) -> None:
    ds, manifest = dataset
    case = _case(manifest, "revision")
    sid, bar = case.sid("revised"), case.date("revised_bar")
    original = float(case.value("original_close"))
    corrected = float(case.value("corrected_close"))
    landed = case.date("corrected_from")
    assert _close(_view(ds, _before(landed), mode), sid, bar) == pytest.approx(original)
    assert _close(_view(ds, landed, mode), sid, bar) == pytest.approx(corrected)


@pytest.mark.parametrize("mode", MODES)
def test_the_null_full_revision_returns_null_and_not_the_previous_value(
    dataset: tuple[Dataset, Manifest], mode: str
) -> None:
    """A revision is a complete replacement: an explicit null means null (§3.2)."""
    ds, manifest = dataset
    case = _case(manifest, "revision")
    sid, bar = case.sid("revised"), case.date("revised_bar")
    corrected = float(case.value("corrected_close"))
    withdrawn = case.date("withdrawn_from")
    view = _view(ds, withdrawn, mode)
    frame = view.bars([sid], bar)
    row = frame.loc[frame["ts"].dt.tz_convert("America/New_York").dt.date == dt.date.fromisoformat(bar)]
    assert len(row) == 1, "the withdrawn revision is a row, not an absence"
    assert np.isnan(float(row.iloc[0]["close"])), "the withdrawal is null, not the previous value"
    assert float(row.iloc[0]["revision"]) == 3.0

    panel = view.panel([sid], bar, "close")
    assert np.isnan(float(panel.loc[pd.Timestamp(bar, tz="America/New_York"), sid]))


@pytest.mark.parametrize("mode", MODES)
def test_the_withdrawn_bar_leaves_the_holding_stale_rather_than_erasing_it(
    dataset: tuple[Dataset, Manifest], mode: str
) -> None:
    ds, manifest = dataset
    case = _case(manifest, "revision")
    sid, bar = case.sid("revised"), case.date("revised_bar")
    corrected = float(case.value("corrected_close"))
    view = _view(ds, case.date("withdrawn_from"), mode)
    marks = view.marks([sid], end=bar)
    assert marks.loc[sid, "valuation_status"] == "stale"
    assert marks.loc[sid, "last_price_bar"] == _before(bar)
    assert marks.loc[sid, "last_price"] != pytest.approx(corrected)
    assert marks.loc[sid, "last_price"] > 0.0


# ---- rebase: a full-series rewrite with its adjustment ---------------------


def test_the_rebase_is_classified_as_a_full_series_restatement(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "rebase")
    sid, effective = case.sid("rebased"), case.date("effective")
    previous = _view(ds, _before(effective)).bars([sid], _before(effective))
    after = _view(ds, effective)
    revised = after.bars([sid], effective)
    adjustments = after.adjustments([sid], case.date("applies_from"), effective)
    ratio = validate_bars_revision(previous, revised, adjustments, sid=sid)
    assert ratio.is_rebase and ratio.full_series
    assert ratio.factor == pytest.approx(REBASE_FACTOR, rel=1e-3)
    assert list(adjustments["kind"]) == ["rebase"]


def test_the_rebase_keeps_equity_continuous_across_the_rewrite(dataset: tuple[Dataset, Manifest]) -> None:
    """100 shares at 200.00 is 20,000; a 4-for-1 rebase marks 50.00 with 400 shares (§3.2)."""
    ds, manifest = dataset
    case = _case(manifest, "rebase")
    sid, effective = case.sid("rebased"), case.date("effective")
    bar = case.date("applies_from")
    before_view = _view(ds, _before(effective))
    stored = _close(before_view, sid, bar)
    restated = _close(_view(ds, effective), sid, bar)
    basis = series_price_basis(before_view.bars([sid], bar))
    shares = 100.0
    factor = applied_share_factor("rebase", basis, REBASE_FACTOR)
    assert factor == REBASE_FACTOR, "a rebase moves shares under either price basis"
    assert shares * factor * restated == pytest.approx(shares * stored, rel=1e-3)


def test_the_same_rewrite_without_its_adjustment_row_is_refused(dataset: tuple[Dataset, Manifest]) -> None:
    """Without the row the revision would mark unchanged shares at a quarter of the value, silently."""
    ds, manifest = dataset
    case = _case(manifest, "rebase")
    sid, effective = case.sid("rebased"), case.date("effective")
    previous = _view(ds, _before(effective)).bars([sid], _before(effective))
    revised = _view(ds, effective).bars([sid], effective)
    with pytest.raises(RebaseNotRecordedError, match="carries no matching"):
        validate_bars_revision(previous, revised, [], sid=sid)


def test_the_adjustment_row_is_invisible_before_the_rewrite_landed(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "rebase")
    sid, effective = case.sid("rebased"), case.date("effective")
    early = _before(effective)
    assert _view(ds, early).adjustments([sid], case.date("applies_from"), early).empty


# ---- delisting: a resolved claim and an unresolved one ---------------------


def test_both_delisted_names_stop_printing_and_are_carried_stale(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "delisting")
    paid, unknown, last = case.sid("paid"), case.sid("unknown"), case.date("last_session")
    view = _view(ds, case.date("pay_date"))
    marks = view.marks([paid, unknown])
    for sid in (paid, unknown):
        assert marks.loc[sid, "last_price_bar"] == last
        assert marks.loc[sid, "valuation_status"] == "stale"
        assert marks.loc[sid, "last_price"] > 0.0
    assert marks.loc[paid, "last_price"] == pytest.approx(float(case.value("last_close")))


def test_the_unknown_terminal_value_leaves_an_unresolved_claim(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "delisting")
    unknown = case.sid("unknown")
    claims = _view(ds, case.date("pay_date")).terminal_claims([unknown])
    claim = claims[unknown]
    assert claim.status == "unresolved"
    assert not claim.complete
    assert claim.valuation_status == "unresolved_claim"
    assert incomplete_reasons(claims) == (claim.incomplete_reason(),)


def test_the_terminal_payout_resolves_the_claim_without_overwriting_an_earlier_close(
    dataset: tuple[Dataset, Manifest],
) -> None:
    ds, manifest = dataset
    case = _case(manifest, "delisting")
    paid = case.sid("paid")
    payout = float(case.value("amount_per_share"))
    last_close = float(case.value("last_close"))
    view = _view(ds, case.date("pay_date"))
    claim = view.terminal_claims([paid])[paid]
    assert claim.status == "resolved" and claim.complete
    assert claim.amount_per_share == pytest.approx(payout)
    assert claim.cash_for(100.0) == pytest.approx(100.0 * payout)
    assert incomplete_reasons([claim]) == ()
    # The payout is a cash amount, not a price: the last close stands.
    assert payout != pytest.approx(last_close)
    marks = view.marks([paid])
    assert marks.loc[paid, "last_price"] == pytest.approx(last_close)
    assert marks.loc[paid, "last_price_bar"] == case.date("last_session")


def test_the_claim_is_unresolved_until_the_payout_is_visible(dataset: tuple[Dataset, Manifest]) -> None:
    """A payout recorded after the decision must not resolve a claim retroactively."""
    ds, manifest = dataset
    case = _case(manifest, "delisting")
    paid = case.sid("paid")
    early = _view(ds, case.fill_bar).terminal_claims([paid])[paid]
    assert early.status == "open" and not early.complete
    assert early.last_session == case.date("last_session")
    late = _view(ds, case.date("pay_date")).terminal_claims([paid])[paid]
    assert late.status == "resolved"


# ---- recovery --------------------------------------------------------------


def test_the_recovery_gap_is_flagged_and_the_holding_is_carried_through_it(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "recovery")
    sid = case.sid("gapped")
    last_priced, gap_end = case.date("last_priced"), case.date("gap_end")
    reference = _close(_view(ds, last_priced), sid, last_priced)
    marks = _view(ds, gap_end).marks([sid])
    assert marks.loc[sid, "valuation_status"] == "stale"
    assert marks.loc[sid, "last_price"] == pytest.approx(reference)
    assert marks.loc[sid, "sessions_stale"] == int(case.value("missing_sessions"))
    flags = {entry["sid"]: entry for entry in manifest.coverage["gap_flags"]}
    assert flags[sid]["missing_sessions"] == int(case.value("missing_sessions"))


def test_the_no_new_exposure_restriction_is_dated_at_the_end_of_the_gap(dataset: tuple[Dataset, Manifest]) -> None:
    ds, manifest = dataset
    case = _case(manifest, "recovery")
    sid, gap_end = case.sid("gapped"), case.date("gap_end")
    before = _view(ds, case.date("last_priced")).capabilities([sid])
    after = _view(ds, gap_end).capabilities([sid])
    assert "no_new_exposure" not in str(before.loc[sid, "restrictions"])
    assert "no_new_exposure" in str(after.loc[sid, "restrictions"])
    assert not bool(after.loc[sid, "shortable"])


# ---- the synthetic flag ----------------------------------------------------


def test_every_scenario_row_the_view_returns_carries_the_synthetic_flag(dataset: tuple[Dataset, Manifest]) -> None:
    """§3.2 [Q11]: every derived record says the data is synthetic."""
    ds, manifest = dataset
    assert manifest.provenance == "synthetic"
    for name, case in scenario_catalog(manifest).items():
        view = _view(ds, case.fill_bar)
        sids = sorted(set(case.sids.values()))
        bars = view.bars(sids, case.fill_bar)
        if not bars.empty:
            assert bars["synthetic"].all(), name
        events = view.events(sids, manifest.coverage["sessions"][0], case.fill_bar)
        if not events.empty:
            assert events["synthetic"].all(), name
