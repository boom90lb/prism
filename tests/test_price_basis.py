"""Price basis, adjustments, and the scoring-basis series (engine spec §3.2, §3.6; decision D3).

The failure this file exists to catch is a silent one. A vendor rewrites a
series onto a new split basis; the workspace stores the restated prices; the
share count is left alone; every mark, every weight and every limit is then
computed on an equity a quarter (or four times) its true value, and nothing
raises. The two halves of the rule that prevents it are asserted in both
directions here: a split moves shares only against a ``raw`` series, and a
full-series restatement moves shares under either basis and is refused outright
unless the ``adjustments`` row that says so is written with it.

The last group is the reproduction. The recorded trend configuration scores on
``split_adjusted_total_return_dividends_as_cash`` — an as-traded close with cash
distributions reinvested at the ex-date — and stage 3b reproduces that run
through this data layer. The events built here from the tracked issuer CSVs must
therefore produce the recorded per-share cash frame exactly, and the reinvested
series must reproduce the recorded scoring panel; the assertions are against
``results/trend_t0/``, which was written by the recorded driver and is read-only
evidence here.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.io.dataset import Dataset, DatasetError
from prism.io.events import (
    ADJUSTMENT_KINDS,
    PRICE_BASES,
    REWRITE_RTOL,
    SCORING_BASES,
    Adjustment,
    Event,
    EventError,
    PayloadError,
    RebaseNotRecordedError,
    adjustments_frame,
    applied_price_factor,
    applied_share_factor,
    distribution_amounts,
    distribution_entitlement,
    events_frame,
    issuer_distribution_events,
    record_bars_revision,
    rewrite_ratio,
    scoring_close,
    series_price_basis,
    total_return_close,
    validate_bars_revision,
    validate_payload,
    write_bars_revision,
)
from prism.io.synthetic import SPLIT_FACTOR, sid_of, write_synthetic_dataset
from prism.signal.base import SCORING_BASES as DECLARED_SCORING_BASES

REPO = Path(__file__).resolve().parents[1]

#: The recorded trend run. Its five JSON artifacts are tracked; the four CSV
#: time series are untracked and regenerable, exactly as in
#: ``results/demotion_b1``, so the reproduction group skips when they are absent
#: and the tracked-config assertions still run.
TREND = REPO / "results" / "trend_t0"
DISTRIBUTIONS = REPO / "data" / "distributions"

#: The declared tolerance for the total-return reproduction. The recorded
#: driver and this module apply the same recursion in the same order, so the
#: only admitted difference is float association across a 4,981-bar cumulative
#: product; measured maximum relative difference over the ten names is 3.1e-16.
#: 1e-12 relative is four orders of magnitude of headroom over that and eight
#: below the smallest economically meaningful level difference.
TR_TOLERANCE = 1e-12

ET = "America/New_York"


def _bars(closes: dict[str, float], *, price_basis: str, source: str = "vendor", revision: int = 1) -> pd.DataFrame:
    """A minimal ``bars`` frame in the §3.2 schema: dated closes with a declared basis."""
    stamps = pd.DatetimeIndex([pd.Timestamp(day, tz=ET) for day in closes])
    return pd.DataFrame(
        {
            "ts": stamps,
            "open": list(closes.values()),
            "high": list(closes.values()),
            "low": list(closes.values()),
            "close": list(closes.values()),
            "volume": [1_000_000.0] * len(closes),
            "price_basis": price_basis,
            "available_at": None,
            "ingested_at": "2026-01-05T20:00:00-05:00",
            "source": source,
            "revision": revision,
        }
    )


def _distribution(sid: str, ex_date: str, amount: float, *, pay_date: str | None = "2026-03-20") -> Event:
    return Event(
        sid=sid,
        event_type="cash_distribution",
        effective_date=ex_date,
        payload={
            "amount_per_share": amount,
            "currency": "USD",
            "pay_date": pay_date,
            "distribution_kind": "dividend",
        },
        ingested_at="2026-03-01T20:00:00-05:00",
        source="issuer_csv",
    )


# ---------------------------------------------------------------------------
# a split, against each declared price basis
# ---------------------------------------------------------------------------


def test_a_split_multiplies_shares_only_on_a_raw_series() -> None:
    # 100 shares at 200.00 is 20,000 of equity. The raw print halves on the
    # effective date, so the share count doubles and equity is unchanged.
    shares, price = 100.0, 200.0
    factor = applied_share_factor("split", "raw", 2.0)
    assert factor == 2.0
    assert applied_price_factor("split", "raw", 2.0) == pytest.approx(0.5)
    assert shares * factor * (price * applied_price_factor("split", "raw", 2.0)) == pytest.approx(shares * price)


def test_a_split_against_an_already_adjusted_series_moves_no_share_and_the_double_count_is_absent() -> None:
    # The vendor restated the whole series at ingest, so the halved print is
    # already in the panel on the effective date. Applying the share leg as
    # well would state 200 shares against a 100.00 close: twice the equity,
    # with nothing in any table saying so. This is the error D3 exists to
    # prevent, and it is asserted absent, not merely unasserted.
    shares, price_before, price_after = 100.0, 200.0, 100.0
    equity_before = shares * price_before

    factor = applied_share_factor("split", "split_adjusted_at_ingest", SPLIT_FACTOR)
    assert factor == 1.0
    applied = shares * factor * price_after
    double_counted = shares * SPLIT_FACTOR * price_after

    assert applied == pytest.approx(equity_before / SPLIT_FACTOR), (
        "an adjusted series marks the halved close on the same shares: the price return of the split day, which is "
        "the -50% the panel prints and the accounting engine is entitled to see"
    )
    assert double_counted == pytest.approx(2.0 * applied)
    assert applied != pytest.approx(double_counted), "applying both legs would double-count the ratio"


def test_the_price_basis_vocabulary_is_closed_and_the_kinds_are_the_declared_four() -> None:
    assert PRICE_BASES == ("raw", "split_adjusted_at_ingest")
    assert ADJUSTMENT_KINDS == ("split", "reverse_split", "stock_distribution", "rebase")
    with pytest.raises(EventError, match="price_basis must be one of"):
        applied_share_factor("split", "adjusted", 2.0)
    with pytest.raises(EventError, match="adjustment kind must be one of"):
        applied_share_factor("dividend", "raw", 2.0)
    with pytest.raises(EventError, match="finite and positive"):
        applied_share_factor("split", "raw", 0.0)


def test_a_reverse_split_and_a_stock_distribution_follow_the_same_basis_rule() -> None:
    for kind in ("reverse_split", "stock_distribution"):
        assert applied_share_factor(kind, "raw", 0.1 if kind == "reverse_split" else 1.05) != 1.0
        assert applied_share_factor(kind, "split_adjusted_at_ingest", 1.05) == 1.0


def test_a_rebase_moves_shares_under_either_basis() -> None:
    # A rewrite is not a ratio event the vendor may already have applied: the
    # prices themselves moved, so the share count follows under both bases.
    for basis in PRICE_BASES:
        assert applied_share_factor("rebase", basis, 4.0) == 4.0


def test_the_declared_basis_is_read_from_the_series_and_a_mixed_series_is_refused() -> None:
    adjusted = _bars({"2026-01-02": 100.0, "2026-01-05": 101.0}, price_basis="split_adjusted_at_ingest")
    assert series_price_basis(adjusted) == "split_adjusted_at_ingest"
    mixed = adjusted.copy()
    mixed.loc[1, "price_basis"] = "raw"
    with pytest.raises(EventError, match="mixes price bases"):
        series_price_basis(mixed)
    with pytest.raises(EventError, match="declares no price_basis column"):
        series_price_basis(adjusted.drop(columns="price_basis"))


def test_the_synthetic_split_scenario_writes_one_series_of_each_basis(dataset: Dataset) -> None:
    raw = pd.read_parquet(dataset.root / f"bars/{sid_of(1)}_1d.parquet")
    adjusted = pd.read_parquet(dataset.root / f"bars/{sid_of(2)}_1d.parquet")
    assert series_price_basis(raw) == "raw"
    assert series_price_basis(adjusted) == "split_adjusted_at_ingest"


# ---------------------------------------------------------------------------
# a vendor back-rewrite
# ---------------------------------------------------------------------------

_STORED = {"2026-01-02": 200.0, "2026-01-05": 204.0, "2026-01-06": 196.0}


def _rewritten(factor: float) -> pd.DataFrame:
    return _bars(
        {day: close / factor for day, close in _STORED.items()},
        price_basis="split_adjusted_at_ingest",
        revision=2,
    )


def test_a_full_series_rewrite_is_classified_as_a_rebase_with_its_ratio() -> None:
    stored = _bars(_STORED, price_basis="split_adjusted_at_ingest")
    ratio = rewrite_ratio(stored, _rewritten(4.0))
    assert ratio.is_rebase
    assert ratio.factor == pytest.approx(4.0)
    assert ratio.n_overlap == 3 and ratio.n_diverging == 3
    assert ratio.max_divergence > REWRITE_RTOL


def test_a_rewrite_recorded_with_its_adjustment_keeps_equity_continuous() -> None:
    # The worked case of §3.2: 100 shares marked at 200.00 is 20,000; the
    # 4-for-1 rebase writes factor 4.0, the revised series marks 50.00, shares
    # become 400, and equity is 20,000 on both sides of the revision.
    stored = _bars(_STORED, price_basis="split_adjusted_at_ingest")
    shares, mark = 100.0, 200.0
    equity_before = shares * mark

    revision = record_bars_revision(
        stored,
        _rewritten(4.0),
        sid="sec_000001",
        source="vendor",
        ingested_at="2026-01-07T20:00:00-05:00",
        effective_date="2026-01-07",
    )
    assert revision.adjustment is not None
    assert revision.adjustment.kind == "rebase"
    assert revision.adjustment.factor == pytest.approx(4.0)
    assert revision.adjustment.applies_from == "2026-01-02", "a rewrite restates history back to its first bar"
    assert revision.price_basis == "split_adjusted_at_ingest", "a rewrite restates the level, never the basis"
    assert int(revision.bars["revision"].max()) == 2

    revised_mark = float(revision.bars.loc[revision.bars["ts"].dt.date == dt.date(2026, 1, 2), "close"].iloc[0])
    assert revised_mark == pytest.approx(50.0)
    rebased_shares = shares * revision.share_factor()
    assert rebased_shares == pytest.approx(400.0)
    assert rebased_shares * revised_mark == pytest.approx(equity_before)

    # And the failure the adjustment prevents, stated as a number: the same
    # revision against the untouched share count is a quarter of the equity.
    assert shares * revised_mark == pytest.approx(equity_before / 4.0)


def test_a_full_series_rewrite_without_its_adjustment_row_is_refused() -> None:
    stored = _bars(_STORED, price_basis="split_adjusted_at_ingest")
    with pytest.raises(RebaseNotRecordedError) as excinfo:
        validate_bars_revision(stored, _rewritten(4.0), [], sid="sec_000001")
    message = str(excinfo.value)
    assert "kind='rebase'" in message and "4" in message

    unrelated = Adjustment(
        sid="sec_000002",
        effective_date="2026-01-07",
        kind="rebase",
        factor=4.0,
        applies_from="2026-01-02",
        ingested_at="2026-01-07T20:00:00-05:00",
        source="vendor",
    )
    with pytest.raises(RebaseNotRecordedError):
        validate_bars_revision(stored, _rewritten(4.0), [unrelated], sid="sec_000001")

    wrong_factor = Adjustment(
        sid="sec_000001",
        effective_date="2026-01-07",
        kind="rebase",
        factor=2.0,
        applies_from="2026-01-02",
        ingested_at="2026-01-07T20:00:00-05:00",
        source="vendor",
    )
    with pytest.raises(RebaseNotRecordedError):
        validate_bars_revision(stored, _rewritten(4.0), [wrong_factor], sid="sec_000001")

    right = Adjustment(
        sid="sec_000001",
        effective_date="2026-01-07",
        kind="rebase",
        factor=4.0,
        applies_from="2026-01-02",
        ingested_at="2026-01-07T20:00:00-05:00",
        source="vendor",
    )
    ratio = validate_bars_revision(stored, _rewritten(4.0), adjustments_frame([right]), sid="sec_000001")
    assert ratio.factor == pytest.approx(4.0)


def test_a_source_correction_is_not_a_rebase_and_touches_no_share_count() -> None:
    # Worked example B of §3.2: revision 2 moves one close by 10 percent, which
    # is not an integer split ratio and carries no adjustments row, so shares
    # are untouched. A rule that demanded an adjustment for every divergence
    # would refuse a legitimate correction.
    stored = _bars(_STORED, price_basis="split_adjusted_at_ingest")
    corrected = _bars({"2026-01-05": 224.4}, price_basis="split_adjusted_at_ingest", revision=2)
    ratio = validate_bars_revision(stored, corrected, [], sid="sec_000001")
    assert not ratio.is_rebase
    assert ratio.n_diverging == 1
    revision = record_bars_revision(
        stored,
        corrected,
        sid="sec_000001",
        source="vendor",
        ingested_at="2026-01-06T20:00:00-05:00",
        effective_date="2026-01-06",
    )
    assert revision.adjustment is None
    assert revision.share_factor() == 1.0


def test_an_identical_refetch_is_neither_a_rebase_nor_a_correction() -> None:
    stored = _bars(_STORED, price_basis="split_adjusted_at_ingest")
    ratio = rewrite_ratio(stored, _bars(_STORED, price_basis="split_adjusted_at_ingest", revision=2))
    assert ratio.factor is None and ratio.n_diverging == 0


def test_the_revision_and_its_rebase_row_land_together_or_not_at_all(tmp_path: Path, monkeypatch) -> None:
    bars_path = tmp_path / "bars" / "sec_000001_1d.parquet"
    bars_path.parent.mkdir(parents=True)
    adjustments_path = tmp_path / "adjustments.parquet"
    stored = _bars(_STORED, price_basis="split_adjusted_at_ingest")
    stored.to_parquet(bars_path, index=False)

    revision = record_bars_revision(
        stored,
        _rewritten(4.0),
        sid="sec_000001",
        source="vendor",
        ingested_at="2026-01-07T20:00:00-05:00",
        effective_date="2026-01-07",
    )

    # The adjustment write fails. The stored series must be the one on disk
    # afterwards: a restated price series whose share rebase was never recorded
    # is exactly the silent quarter-value state.
    real_replace = __import__("os").replace
    calls = {"n": 0}

    def failing_replace(src, dst):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("simulated failure between the two replaces")
        return real_replace(src, dst)

    monkeypatch.setattr("prism.io.events.os.replace", failing_replace)
    with pytest.raises(OSError):
        write_bars_revision(revision, bars_path=bars_path, adjustments_path=adjustments_path)
    after = pd.read_parquet(bars_path)
    assert set(after["revision"]) == {1}
    assert float(after["close"].max()) == pytest.approx(204.0)
    assert not adjustments_path.is_file()
    assert not list(tmp_path.rglob("*.tmp")) and not list(tmp_path.rglob("*.prev")), "no half-written file survives"

    monkeypatch.undo()
    write_bars_revision(revision, bars_path=bars_path, adjustments_path=adjustments_path)
    landed = pd.read_parquet(bars_path)
    recorded = pd.read_parquet(adjustments_path)
    assert set(landed["revision"]) == {1, 2}
    assert len(recorded) == 1
    assert float(recorded["factor"].iloc[0]) == pytest.approx(4.0)
    assert str(recorded["kind"].iloc[0]) == "rebase"


# ---------------------------------------------------------------------------
# events and their payloads
# ---------------------------------------------------------------------------


def test_every_declared_event_type_validates_its_payload_shape() -> None:
    with pytest.raises(PayloadError, match="event_type must be one of"):
        validate_payload("dividend", {})
    with pytest.raises(PayloadError, match="factor"):
        validate_payload("split", {})
    with pytest.raises(PayloadError, match="must not be negative"):
        validate_payload(
            "cash_distribution",
            {"amount_per_share": -0.1, "currency": "USD", "pay_date": None, "distribution_kind": "dividend"},
        )
    with pytest.raises(PayloadError, match="distribution_kind must be one of"):
        validate_payload(
            "cash_distribution",
            {"amount_per_share": 0.5, "currency": "USD", "pay_date": None, "distribution_kind": "special"},
        )
    with pytest.raises(PayloadError, match="ISO YYYY-MM-DD"):
        validate_payload(
            "cash_distribution",
            {"amount_per_share": 0.5, "currency": "USD", "pay_date": "20/03/26", "distribution_kind": "dividend"},
        )
    assert validate_payload("terminal_payout", {"status": "unknown"}) == {"status": "unknown"}
    assert validate_payload("spinoff", {"child_sid": None, "ratio": 0.25, "cash_in_lieu": True})["child_sid"] is None
    assert validate_payload("halt", {"start": "2026-01-05T10:00", "end": None, "reason": "volatility"})["end"] is None


def test_an_event_round_trips_through_the_table_row() -> None:
    event = _distribution("sec_000003", "2026-03-06", 0.55)
    frame = events_frame([event])
    assert list(frame.columns) == [
        "sid",
        "event_type",
        "effective_date",
        "announce_date",
        "payload",
        "available_at",
        "ingested_at",
        "source",
        "revision",
    ]
    back = Event.from_row(frame.iloc[0].to_dict())
    assert back == event


def test_two_rows_for_one_key_source_and_revision_are_an_ingestion_defect() -> None:
    event = _distribution("sec_000003", "2026-03-06", 0.55)
    with pytest.raises(EventError, match="more than one row"):
        events_frame([event, event])


# ---------------------------------------------------------------------------
# cash distributions
# ---------------------------------------------------------------------------


def test_a_distribution_credits_a_long_and_debits_a_short() -> None:
    event = _distribution("sec_000003", "2026-03-06", 0.55)
    long = distribution_entitlement(event, 400.0)
    short = distribution_entitlement(event, -400.0)
    assert long.amount == pytest.approx(220.0) and long.leg == "receivable"
    assert short.amount == pytest.approx(-220.0) and short.leg == "liability"
    assert long.amount == -short.amount, "the sign follows the signed position; a short pays what a long receives"
    assert long.pay_date == "2026-03-20" and not long.pay_date_assumed

    # Entitlement is the share count at the close of the bar *before* the
    # ex-date, so a book that was flat into the ex-date open is owed nothing.
    assert distribution_entitlement(event, 0.0).leg == "none"


def test_a_distribution_with_no_pay_date_says_the_settlement_is_assumed() -> None:
    event = _distribution("sec_000003", "2026-03-06", 0.55, pay_date=None)
    entitlement = distribution_entitlement(event, 100.0)
    assert entitlement.pay_date is None and entitlement.pay_date_assumed


def test_only_a_cash_distribution_creates_an_entitlement() -> None:
    split = Event(
        sid="sec_000001",
        event_type="split",
        effective_date="2026-03-06",
        payload={"factor": 2.0},
        ingested_at="2026-03-01T20:00:00-05:00",
        source="vendor",
    )
    with pytest.raises(EventError, match="needs a cash_distribution"):
        distribution_entitlement(split, 100.0)


def _close_panel() -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(f"2026-01-0{d}", tz=ET) for d in (2, 5, 6, 7, 8)])
    return pd.DataFrame(
        {"sec_000003": [100.0, 101.0, 102.0, 103.0, 104.0], "sec_000004": [np.nan, np.nan, 50.0, 51.0, 52.0]},
        index=index,
    )


def test_distribution_amounts_place_the_cash_on_the_ex_date_session() -> None:
    close = _close_panel()
    frame = distribution_amounts([_distribution("sec_000003", "2026-01-06", 0.25)], close)
    assert frame.loc[close.index[2], "sec_000003"] == pytest.approx(0.25)
    assert frame.drop(index=close.index[2]).to_numpy().sum() == 0.0


def test_an_ex_date_off_the_calendar_credits_the_next_session_and_two_on_one_session_sum() -> None:
    # pandas 3 enables copy-on-write by default. Exercise the same read-only
    # array boundary on pandas 2, which remains supported by the release.
    with pd.option_context("mode.copy_on_write", True):
        close = _close_panel()
        original = close.copy(deep=True)
        # Both weekend ex-dates roll forward to Monday's session.
        frame = distribution_amounts(
            [_distribution("sec_000003", "2026-01-03", 0.10), _distribution("sec_000003", "2026-01-04", 0.20)],
            close,
        )
        pd.testing.assert_frame_equal(close, original)
    assert frame.loc[close.index[1], "sec_000003"] == pytest.approx(0.30)


def test_a_distribution_outside_the_panel_is_dropped_rather_than_credited_to_an_edge() -> None:
    close = _close_panel()
    beyond = distribution_amounts([_distribution("sec_000003", "2026-02-02", 0.40)], close)
    assert beyond.to_numpy().sum() == 0.0, "an ex-date after the last session is not knowable inside this panel"
    before_listing = distribution_amounts([_distribution("sec_000004", "2026-01-02", 0.40)], close)
    assert before_listing.to_numpy().sum() == 0.0, "there is no position in a name this panel does not yet price"


def test_a_negative_distribution_amount_is_refused_rather_than_netted() -> None:
    # Two guards, and both are asserted: the payload check at construction, and
    # the placement check that a caller cannot get past by mutating a payload
    # after the fact. A negative per-share amount is a source defect; netting it
    # against a real distribution would hide it inside a plausible total.
    with pytest.raises(PayloadError, match="must not be negative"):
        _distribution("sec_000003", "2026-01-06", -0.25)

    event = _distribution("sec_000003", "2026-01-06", 0.25)
    object.__setattr__(event, "payload", {**event.payload, "amount_per_share": -0.25})
    with pytest.raises(EventError, match="is not usable"):
        distribution_amounts([event], _close_panel())


# ---------------------------------------------------------------------------
# the total-return series and the scoring basis
# ---------------------------------------------------------------------------


def test_the_total_return_series_reinvests_at_the_ex_date_close() -> None:
    close = _close_panel()[["sec_000003"]]
    distributions = distribution_amounts([_distribution("sec_000003", "2026-01-06", 2.04)], close)
    total = total_return_close(close, distributions)

    assert total.iloc[0, 0] == pytest.approx(close.iloc[0, 0]), "the series is rebased to the first finite close"
    assert total.iloc[1, 0] == pytest.approx(close.iloc[1, 0]), "no distribution, no wedge"
    factor = 1.0 + 2.04 / 102.0
    ratio = total["sec_000003"] / close["sec_000003"]
    assert ratio.iloc[1] == pytest.approx(1.0)
    assert ratio.iloc[2] == pytest.approx(factor)
    assert ratio.iloc[4] == pytest.approx(factor), "the wedge is a level, held after the ex-date"
    assert (ratio.diff().dropna() >= -1e-15).all(), "reinvestment never lowers the total-return wedge"


def test_the_total_return_series_is_nan_before_the_first_close_and_equals_price_without_distributions() -> None:
    close = _close_panel()
    total = total_return_close(close, distribution_amounts([], close))
    assert total["sec_000004"].isna().tolist() == [True, True, False, False, False]
    pd.testing.assert_frame_equal(total, close, check_dtype=False)


def test_the_scoring_basis_resolution_is_the_declared_one() -> None:
    assert SCORING_BASES == DECLARED_SCORING_BASES == ("price", "total_return")
    close = _close_panel()[["sec_000003"]]
    distributions = distribution_amounts([_distribution("sec_000003", "2026-01-06", 2.04)], close)
    pd.testing.assert_frame_equal(scoring_close("price", close, distributions), close)
    reinvested = scoring_close("total_return", close, distributions)
    assert reinvested.iloc[-1, 0] > close.iloc[-1, 0]
    with pytest.raises(EventError, match="scoring_basis must be one of"):
        scoring_close("adjusted_close", close, distributions)
    with pytest.raises(EventError, match="needs the cash_distribution amounts"):
        scoring_close("total_return", close, None)


# ---------------------------------------------------------------------------
# through the as-of view
# ---------------------------------------------------------------------------

START = dt.date(2026, 1, 1)
END = dt.date(2026, 12, 31)
VINTAGE = "synthetic_v1"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    root = tmp_path_factory.mktemp("price_basis") / "data"
    write_synthetic_dataset(root, seed=11, scenarios="all", n_equities=14, n_etfs=3, start=START, end=END)
    return Dataset.open(root, verify_files=True)


def _ex_date(dataset: Dataset, sid: str) -> str:
    events = pd.read_parquet(dataset.root / "events.parquet")
    rows = events.loc[(events["sid"] == sid) & (events["event_type"] == "cash_distribution")]
    assert len(rows) == 1, "the distribution scenario writes exactly one cash distribution"
    return str(rows["effective_date"].iloc[0])


def test_the_view_reinvests_a_visible_distribution_and_nothing_else(dataset: Dataset) -> None:
    sid = sid_of(3)
    ex_date = _ex_date(dataset, sid)
    events = pd.read_parquet(dataset.root / "events.parquet")
    payload = json.loads(events.loc[(events["sid"] == sid) & (events["effective_date"] == ex_date), "payload"].iloc[0])
    view = dataset.view("research_vintage", "2026-12-30", vintage=VINTAGE)
    close = view.panel([sid], "2026-12-30", "close")
    total = view.total_return([sid], "2026-12-30")

    assert list(total.columns) == [sid]
    assert total.index.equals(close.index), "the scoring panel is on the calendar the price panel is on"
    stamp = pd.Timestamp(ex_date, tz=close.index.tz)
    ex_close = float(close.loc[stamp, sid])
    wedge = total[sid] / close[sid]
    assert wedge.loc[: stamp - pd.Timedelta(days=1)].max() == pytest.approx(1.0)
    assert float(wedge.loc[stamp]) == pytest.approx(1.0 + float(payload["amount_per_share"]) / ex_close)
    assert float(wedge.iloc[-1]) == pytest.approx(float(wedge.loc[stamp]))


def test_a_view_that_closes_before_the_ex_date_sees_no_reinvestment(dataset: Dataset) -> None:
    sid = sid_of(3)
    ex_date = dt.date.fromisoformat(_ex_date(dataset, sid))
    earlier = (ex_date - dt.timedelta(days=7)).isoformat()
    view = dataset.view("research_vintage", earlier, vintage=VINTAGE)
    close = view.panel([sid], earlier, "close")
    pd.testing.assert_frame_equal(view.total_return([sid], earlier), close)


def test_a_name_that_never_distributed_scores_on_its_own_close(dataset: Dataset) -> None:
    sid = sid_of(12)
    view = dataset.view("research_vintage", "2026-12-30", vintage=VINTAGE)
    close = view.panel([sid], "2026-12-30", "close")
    pd.testing.assert_frame_equal(view.total_return([sid], "2026-12-30"), close)


def test_total_return_refuses_a_dataset_with_no_events_table(dataset: Dataset, tmp_path: Path) -> None:
    # A manifest with no events table cannot answer "was there a distribution",
    # and answering with the price series would put a total-return label on a
    # price panel.
    from prism.io.manifest import Manifest

    manifest = Manifest.read(dataset.root / "manifest.json")
    stripped = Manifest(
        provenance=manifest.provenance,
        as_of=manifest.as_of,
        files=tuple((path, digest) for path, digest in manifest.files if path != "events.parquet"),
        sources=manifest.sources,
        source_priority=manifest.source_priority,
        universe=manifest.universe,
        vintage=manifest.vintage,
        coverage=manifest.coverage,
        transformations=list(manifest.transformations),
        assumptions=dict(manifest.assumptions),
    )
    view = Dataset(dataset.root, stripped).view("research_vintage", "2026-12-30", vintage=VINTAGE)
    with pytest.raises(DatasetError, match="lists no events table"):
        view.total_return([sid_of(3)], "2026-12-30")


# ---------------------------------------------------------------------------
# the recorded trend convention
# ---------------------------------------------------------------------------


def _recorded_config() -> dict:
    path = TREND / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"the tracked recorded record {path} is missing from this checkout")
    return json.loads(path.read_text())


def _recorded_series(name: str) -> pd.DataFrame:
    path = TREND / name
    if not path.is_file():
        pytest.skip(
            f"{path} is an untracked artifact of the recorded trend run (regenerable: "
            "`python -m research.scripts.trend_wfo --output_dir results/trend_t0`); the tracked config and "
            "summary assertions above still run"
        )
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(ET)
    return frame.astype(float)


def test_the_recorded_trend_configuration_declares_the_convention_this_module_builds() -> None:
    config = _recorded_config()
    assert config["price_convention"] == "total_return"
    assert config["data_convention"] == "split_adjusted_total_return_dividends_as_cash"
    assert config["distributions_dir"] == "data/distributions"
    assert config["price_return_fallback"] is False


def test_the_issuer_files_become_cash_distribution_events_with_their_provenance() -> None:
    config = _recorded_config()
    recorded = json.loads((TREND / "summary.json").read_text())["distributions"]
    for symbol in config["symbols"]:
        events = issuer_distribution_events(DISTRIBUTIONS, symbol)
        provenance = json.loads((DISTRIBUTIONS / f"{symbol}.provenance.json").read_text())
        assert len(events) == int(provenance["n_records"]), f"{symbol}: every issuer row becomes one event"
        digest = provenance.get("sha256") or provenance["raw_sha256"]
        for event in events:
            assert event.source == "issuer_csv"
            assert event.ingested_at == provenance["captured_at"]
            assert event.payload["provenance_sha256"] == digest
            assert event.payload["currency"] == "USD"
            assert event.payload["distribution_kind"] == "mixed"
        if events:
            assert min(event.effective_date for event in events) == provenance["first_ex_date"]
            assert max(event.effective_date for event in events) == provenance["last_ex_date"]
        else:
            # GLD has never distributed. A name with no events is a real answer
            # here, and the reproduction below asserts its column stays zero.
            assert int(provenance["n_records"]) == 0
        # The recorded run saw a subset of these rows: the ones inside the
        # sample. It can never have seen more of them than the file holds.
        assert int(recorded[symbol]["n_records_in_sample"]) <= len(events)


def test_the_events_reproduce_the_recorded_per_share_cash_frame_exactly() -> None:
    config = _recorded_config()
    close = _recorded_series("closes.csv")
    recorded_dividends = _recorded_series("dividends.csv")
    recorded_meta = json.loads((TREND / "summary.json").read_text())["distributions"]

    end = pd.Timestamp(config["end_date"])
    events = []
    for symbol in close.columns:
        events.extend(
            event
            for event in issuer_distribution_events(DISTRIBUTIONS, symbol)
            if pd.Timestamp(event.effective_date, tz=ET) <= end
        )
    frame = distribution_amounts(events, close)

    pd.testing.assert_frame_equal(frame, recorded_dividends, check_dtype=False, atol=0.0, rtol=0.0)
    for symbol in close.columns:
        credited = int((frame[symbol] != 0.0).sum())
        assert credited == int(recorded_meta[symbol]["n_sessions_credited"]), symbol


def test_the_total_return_series_reproduces_the_recorded_scoring_panel() -> None:
    close = _recorded_series("closes.csv")
    dividends = _recorded_series("dividends.csv")
    recorded = _recorded_series("score_close.csv")

    total = total_return_close(close, dividends)
    assert total.shape == recorded.shape
    assert (total.isna() == recorded.isna()).all().all(), "a name is unscored on exactly the sessions it was"
    difference = (total - recorded).abs()
    relative = (difference / recorded.abs()).max().max()
    assert float(relative) < TR_TOLERANCE, f"total-return panel differs by {relative:.2e} relative"

    # And the panel is not the price panel: the recorded run's whole point is
    # that the two differ on names that distributed.
    assert not np.allclose(total.to_numpy(), close.to_numpy(), equal_nan=True)
