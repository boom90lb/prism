"""``--no-submit``, the submission announcement, and the run-directory rail.

The defect these pin, from the 2026-09-09 incident: ``prism.scripts.paper_loop``
and ``prism.scripts.paper_sweep`` had exactly two states — "fails before the
credential check" and "trades the account". A verification step meaning to
observe the first performed the second, sending 116 market-on-open orders from a
throwaway ``--run-dir``. Three things had to become true, and each is asserted
here through ``main`` — the way the nightly wrapper reaches these shells:

1. ``--no-submit`` runs the whole cycle and submits nothing. The venue double
   below **fails the test if its** ``submit`` **is reached at all**, so the
   assertion is not "no orders were recorded" but "the submitting call was never
   made"; the matching test without the flag proves the same double's ``submit``
   *is* reached, so the first result is the flag's doing and not a broken path.
2. A run that submits announces it on one line naming the endpoint, the account,
   the decision bar and the order count — and a run that submits nothing never
   prints that line.
3. Submitting from a run directory with no prior session is refused, before any
   venue read, unless the operator says otherwise.

Nothing here opens a socket: the broker and the bar source are doubles injected
into the entry-point module, and the panels are in memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prism.live.authorization import PAPER_BASE_URL
from prism.live.broker import AccountIdentity, Broker, Fill, Order, transmits_to_venue
from prism.live.loop import SUBMISSION_ANNOUNCEMENT, LiveLoopContext, submit_pending
from prism.live.no_submit import (
    NO_SUBMIT,
    WOULD_SUBMIT,
    NoSubmitBroker,
    VenueWriteRefused,
    mirror_run_dir,
)
from prism.live.safety import SafetyViolation, require_established_run_dir, sample_run_dir_history
from prism.live.state import LoopState, StateStore
from prism.scripts import paper_loop, paper_sweep

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]
ACCOUNT_ID = "PA-TEST-0001"


class VenueWriteReached(AssertionError):
    """The double was asked to write to the venue when it must not have been."""


class VenueDouble(Broker):
    """In-memory venue that records reads and can fail the test on any write.

    ``forbid_writes`` is the whole point: under ``--no-submit`` the real
    adapter's ``submit`` must never be called, and the only way to assert that
    without trusting the code under test is to make the call itself the failure.
    """

    endpoint = PAPER_BASE_URL

    def __init__(
        self,
        *,
        forbid_writes: bool = False,
        positions: dict[str, float] | None = None,
        cash: float = 1_000_000.0,
        known_ids: set[str] | None = None,
    ) -> None:
        self.forbid_writes = forbid_writes
        self._positions = dict(positions or {})
        self._cash = float(cash)
        self.submitted: list[Order] = []
        self.cancelled: list[str] = []
        self.reads: list[str] = []
        self._known: set[str] = set(known_ids or ())
        self.open_ids: set[str] = set()
        self.fills: list[Fill] = []

    # ------------------------------------------------------------------ writes

    def submit(self, order: Order) -> None:
        if self.forbid_writes:
            raise VenueWriteReached(
                f"the venue double's submit() was reached for {order.client_order_id!r}; "
                "under --no-submit no order may leave the process"
            )
        self.submitted.append(order)
        self._known.add(order.client_order_id)

    def cancel(self, client_order_id: str) -> None:
        if self.forbid_writes:
            raise VenueWriteReached(f"the venue double's cancel() was reached for {client_order_id!r}")
        self.cancelled.append(client_order_id)

    # ------------------------------------------------------------------- reads

    def positions(self) -> dict[str, float]:
        self.reads.append("positions")
        return dict(self._positions)

    def cash(self) -> float:
        self.reads.append("cash")
        return self._cash

    def submitted_order_ids(self, since: str | None = None) -> set[str]:
        self.reads.append("submitted_order_ids")
        return set(self._known)

    def open_order_ids(self, client_order_ids: set[str]) -> set[str]:
        self.reads.append("open_order_ids")
        return set(client_order_ids) & self.open_ids

    def fills_for(self, client_order_ids: set[str]) -> list[Fill]:
        self.reads.append("fills_for")
        return [f for f in self.fills if f.client_order_id in client_order_ids]

    def account_identity(self) -> AccountIdentity:
        self.reads.append("account_identity")
        return AccountIdentity(account_id=ACCOUNT_ID, endpoint=self.endpoint, is_paper=True)


class BarSourceDouble:
    """A ``fetch_batch`` source over one in-memory panel."""

    def __init__(self, close: pd.DataFrame, volume: pd.DataFrame) -> None:
        self._close = close
        self._volume = volume

    def fetch_batch(self, symbols, interval="1d", start_date=None, end_date=None, **kwargs):
        frames = {}
        for symbol in symbols:
            if symbol not in self._close.columns:
                frames[symbol] = pd.DataFrame()
                continue
            frames[symbol] = pd.DataFrame({"close": self._close[symbol], "volume": self._volume[symbol]})
        return frames


def _panels(n_bars: int = 400) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic panels whose names have well-separated 12-1 momentum."""
    index = pd.bdate_range("2024-01-02", periods=n_bars)
    steps = np.linspace(-0.0006, 0.0006, len(SYMBOLS))
    close = pd.DataFrame(
        {sym: 100.0 * np.exp(np.arange(n_bars) * steps[i]) for i, sym in enumerate(SYMBOLS)},
        index=index,
    )
    volume = pd.DataFrame(1_000_000.0, index=index, columns=SYMBOLS)
    return close, volume


@pytest.fixture()
def venue(tmp_path: Path, monkeypatch) -> VenueDouble:
    """A wired paper loop: doubles injected, credentials present, no `.env` above."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret-key")
    monkeypatch.delenv("APCA_API_BASE_URL", raising=False)
    broker = VenueDouble()
    close, volume = _panels()

    def _from_env(**kwargs):
        # The rehearsal must still resolve credentials: a flag that skipped them
        # would not be exercising the path the incident ran.
        assert kwargs.get("mode") == "paper"
        import os

        assert os.environ["APCA_API_KEY_ID"] and os.environ["APCA_API_SECRET_KEY"]
        return broker

    for module in (paper_loop, paper_sweep):
        monkeypatch.setattr(module, "AlpacaBroker", SimpleNamespace(from_env=_from_env), raising=True)
    monkeypatch.setattr(
        paper_loop,
        "AlpacaBarSource",
        SimpleNamespace(from_env=lambda **kw: BarSourceDouble(close, volume)),
        raising=True,
    )
    return broker


def _loop_argv(run_dir: Path, *extra: str) -> list[str]:
    return [
        "--book",
        "momentum",
        "--symbols",
        ",".join(SYMBOLS),
        "--run-dir",
        str(run_dir),
        "--decile",
        "0.3",
        "--decision-every",
        "1",
        *extra,
    ]


def _announcements(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if SUBMISSION_ANNOUNCEMENT in r.getMessage()]


def _would_submit_lines(caplog) -> list[str]:
    """The per-order rehearsal lines only — not the banner that names the prefix."""
    marker = f"{NO_SUBMIT} {WOULD_SUBMIT} #"
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(marker)]


# ---------------------------------------------------------------------------
# 1. --no-submit runs everything and submits nothing
# ---------------------------------------------------------------------------


def test_no_submit_runs_the_cycle_and_never_calls_submit(venue: VenueDouble, tmp_path: Path, caplog) -> None:
    """The incident's exact invocation shape, made safe.

    A throwaway run directory, credentials present — the 2026-09-09 command —
    plus ``--no-submit``. The double fails the test if ``submit`` is reached, so
    passing means no order left the process; the venue was still read, the book
    was still constructed, and every order it would have sent is in the log.
    """
    run_dir = tmp_path / "throwaway"
    venue.forbid_writes = True
    with caplog.at_level(logging.INFO):
        assert paper_loop.main(_loop_argv(run_dir, "--no-submit")) == 0

    assert venue.submitted == []
    assert venue.cancelled == []
    # The real path ran: the venue was read for truth and the account identified.
    assert "positions" in venue.reads and "cash" in venue.reads
    # ... and it decided orders, order by order, every one of them logged.
    would = _would_submit_lines(caplog)
    assert would, "a refresh session decided no orders; the rehearsal proved nothing"
    assert all(line.startswith(f"{NO_SUBMIT} {WOULD_SUBMIT} #") for line in would)
    # The closing statement is unmissable and names the count.
    closing = [r.getMessage() for r in caplog.records if "NOTHING WAS SENT TO THE VENUE" in r.getMessage()]
    assert len(closing) == 1
    assert f"{len(would)} ORDERS WOULD HAVE BEEN SUBMITTED" in closing[0]


def test_no_submit_leaves_the_run_directory_without_a_decision(venue: VenueDouble, tmp_path: Path, caplog) -> None:
    """A rehearsal must not be able to arm a trade.

    The write-ahead protocol persists the decided orders *before* submitting
    them, and a resumed pending set is submitted without being re-decided — so a
    rehearsal that wrote ``state.json`` into the real run directory would leave a
    decision for the next real session to fire at stale prices. Nothing durable
    may be left behind.
    """
    run_dir = tmp_path / "throwaway"
    venue.forbid_writes = True
    with caplog.at_level(logging.INFO):
        assert paper_loop.main(_loop_argv(run_dir, "--no-submit")) == 0

    left_behind = sorted(p.name for p in run_dir.iterdir()) if run_dir.is_dir() else []
    # The writer lock is the run directory's own file and is released; nothing
    # else — no state, no ledger — may survive a rehearsal.
    assert [name for name in left_behind if name != ".lock"] == []


def test_no_submit_preserves_an_established_run_directory_byte_for_byte(
    venue: VenueDouble, tmp_path: Path, caplog
) -> None:
    """Rehearsing against the live run directory reads it and changes nothing."""
    run_dir = tmp_path / "runs" / "paper_loop_momentum"
    run_dir.mkdir(parents=True)
    StateStore(run_dir / "state.json").save(LoopState())
    (run_dir / "equity.jsonl").write_text('{"decision_bar": "2024-01-02", "equity": 1000000.0}\n', "utf-8")
    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.name != ".lock"}

    venue.forbid_writes = True
    with caplog.at_level(logging.INFO):
        assert paper_loop.main(_loop_argv(run_dir, "--no-submit")) == 0

    after = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.name != ".lock"}
    assert after == before
    assert venue.submitted == []


def test_a_rehearsal_still_obeys_the_kill_switch(venue: VenueDouble, tmp_path: Path, caplog) -> None:
    """The mirror carries the operator's controls, not just the ledgers.

    A rehearsal that ignored the kill switch would report a book the halted
    account would never take, which is a worse answer than no answer.
    """
    run_dir = tmp_path / "runs" / "paper_loop_momentum"
    run_dir.mkdir(parents=True)
    StateStore(run_dir / "state.json").save(LoopState())
    (run_dir / "KILL_SWITCH").write_text("", "utf-8")
    venue.forbid_writes = True
    with caplog.at_level(logging.INFO):
        assert paper_loop.main(_loop_argv(run_dir, "--no-submit")) == 2
    assert _would_submit_lines(caplog) == []
    assert (run_dir / "KILL_SWITCH").exists()


# ---------------------------------------------------------------------------
# 2. the matching control: the normal path really does submit
# ---------------------------------------------------------------------------


def test_the_normal_path_calls_submit_and_announces_the_venue(venue: VenueDouble, tmp_path: Path, caplog) -> None:
    """The control for every ``--no-submit`` assertion above.

    Same command, same doubles, flag removed: the double's ``submit`` is called
    once per order, and the run announces itself on exactly one line naming the
    endpoint, the account, the decision bar and the order count.
    """
    run_dir = tmp_path / "throwaway"
    with caplog.at_level(logging.INFO):
        assert paper_loop.main(_loop_argv(run_dir, "--allow-new-run-dir")) == 0

    assert venue.submitted, "the control submitted nothing; it controls for nothing"
    lines = _announcements(caplog)
    assert len(lines) == 1
    announcement = lines[0]
    assert PAPER_BASE_URL.rstrip("/") in announcement
    assert f"account={ACCOUNT_ID}" in announcement
    assert f"orders={len(venue.submitted)}" in announcement
    assert f"decision_bar={venue.submitted[0].decision_bar}" in announcement
    # And the rehearsal's vocabulary is nowhere in a run that traded.
    assert _would_submit_lines(caplog) == []


def test_the_announcement_is_absent_from_a_no_submit_run(venue: VenueDouble, tmp_path: Path, caplog) -> None:
    """The line that says a run traded is never printed by a run that did not."""
    venue.forbid_writes = True
    with caplog.at_level(logging.INFO):
        assert paper_loop.main(_loop_argv(tmp_path / "throwaway", "--no-submit")) == 0
    assert _announcements(caplog) == []


def test_an_established_run_directory_needs_no_opt_in(venue: VenueDouble, tmp_path: Path, caplog) -> None:
    """The nightly's own shape: a run directory with prior sessions just runs.

    ``ops/paper_loop_nightly.sh`` passes ``--run-dir "$(cat runs/ACTIVE_RUN_DIR)"``
    and no other flag, so the provenance rail must be inert for it.
    """
    run_dir = tmp_path / "runs" / "paper_loop_momentum"
    run_dir.mkdir(parents=True)
    StateStore(run_dir / "state.json").save(LoopState())
    with caplog.at_level(logging.INFO):
        assert paper_loop.main(_loop_argv(run_dir)) == 0
    assert venue.submitted
    assert len(_announcements(caplog)) == 1


# ---------------------------------------------------------------------------
# 3. the run-directory provenance rail
# ---------------------------------------------------------------------------


def test_submitting_from_a_directory_with_no_prior_session_is_refused(venue: VenueDouble, tmp_path: Path) -> None:
    """The rail fires before the venue is read at all."""
    venue.forbid_writes = True
    with pytest.raises(SafetyViolation, match="not the continuation of an established paper session"):
        paper_loop.main(_loop_argv(tmp_path / "throwaway"))
    assert venue.reads == [], "the rail must refuse before any venue read"
    assert venue.submitted == []


def test_the_sweep_refuses_the_same_directory(venue: VenueDouble, tmp_path: Path) -> None:
    venue.forbid_writes = True
    with pytest.raises(SafetyViolation, match="the paper sweep refuses to submit"):
        paper_sweep.main(["--run-dir", str(tmp_path / "throwaway")])
    assert venue.reads == []


def test_run_dir_history_is_sampled_from_disk(tmp_path: Path) -> None:
    absent = sample_run_dir_history(tmp_path / "nope")
    assert not absent.existed and not absent.established
    empty = tmp_path / "empty"
    empty.mkdir()
    assert sample_run_dir_history(empty).existed
    assert not sample_run_dir_history(empty).established
    (empty / "equity.jsonl").write_text("", "utf-8")
    established = sample_run_dir_history(empty)
    assert established.established and established.session_files == ("equity.jsonl",)
    # The opt-in is the only way past, and it is explicit.
    require_established_run_dir(absent, allow_new=True, entry_point="the paper loop")
    with pytest.raises(SafetyViolation):
        require_established_run_dir(absent, allow_new=False, entry_point="the paper loop")


# ---------------------------------------------------------------------------
# 4. the sweep
# ---------------------------------------------------------------------------


def _seed_pending(run_dir: Path, venue: VenueDouble) -> LoopState:
    """A run directory holding a decided-and-submitted batch that never printed."""
    run_dir.mkdir(parents=True, exist_ok=True)
    orders = [
        Order(
            client_order_id=f"mom:2024-07-01:{sym}",
            symbol=sym,
            qty=10.0,
            decision_bar="2024-07-01",
            reference_price=100.0,
        )
        for sym in ("AAA", "BBB")
    ]
    state = LoopState(pending_orders=orders, pending_decision_bar="2024-07-01")
    StateStore(run_dir / "state.json").save(state)
    venue._known = {o.client_order_id for o in orders}  # submitted; the auction did not print
    venue.open_ids = set()
    return state


def test_sweep_no_submit_never_calls_submit_and_preserves_the_run_dir(
    venue: VenueDouble, tmp_path: Path, caplog
) -> None:
    run_dir = tmp_path / "runs" / "paper_loop_momentum"
    _seed_pending(run_dir, venue)
    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.name != ".lock"}
    venue.forbid_writes = True

    with caplog.at_level(logging.INFO):
        assert paper_sweep.main(["--run-dir", str(run_dir), "--no-submit"]) == 0

    assert venue.submitted == []
    assert _announcements(caplog) == []
    assert len(_would_submit_lines(caplog)) == 2
    after = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.name != ".lock"}
    assert after == before


def test_sweep_normal_path_submits_and_announces(venue: VenueDouble, tmp_path: Path, caplog) -> None:
    run_dir = tmp_path / "runs" / "paper_loop_momentum"
    _seed_pending(run_dir, venue)
    with caplog.at_level(logging.INFO):
        assert paper_sweep.main(["--run-dir", str(run_dir)]) == 0
    assert [o.client_order_id for o in venue.submitted] == [
        "mom:2024-07-01:AAA:S1",
        "mom:2024-07-01:BBB:S1",
    ]
    lines = _announcements(caplog)
    assert len(lines) == 1
    assert "orders=2" in lines[0] and f"account={ACCOUNT_ID}" in lines[0]


# ---------------------------------------------------------------------------
# 5. the pieces, on their own
# ---------------------------------------------------------------------------


def test_no_submit_broker_reads_through_records_and_refuses_other_writes() -> None:
    inner = VenueDouble(positions={"AAA": 3.0}, cash=500.0, forbid_writes=True)
    double = NoSubmitBroker(inner)
    assert transmits_to_venue(inner) is True
    assert transmits_to_venue(double) is False
    assert double.positions() == {"AAA": 3.0} and double.cash() == 500.0
    assert double.account_identity().account_id == ACCOUNT_ID
    assert double.endpoint == PAPER_BASE_URL  # duck-typed attributes delegate
    order = Order(client_order_id="x:1", symbol="AAA", qty=-2.0, decision_bar="2024-07-01", reference_price=10.0)
    double.submit(order)  # the inner double would have raised if it were reached
    assert double.recorded == (order,) and inner.submitted == []
    with pytest.raises(VenueWriteRefused):
        double.cancel("x:1")


def test_mirror_run_dir_copies_the_record_and_skips_the_lock(tmp_path: Path) -> None:
    source = tmp_path / "run"
    source.mkdir()
    (source / "state.json").write_text("{}", "utf-8")
    (source / "equity.jsonl").write_text("{}\n", "utf-8")
    (source / ".lock").write_text("held", "utf-8")
    (source / "state.json.tmp").write_text("torn", "utf-8")
    mirrored = mirror_run_dir(source, tmp_path / "mirror")
    assert sorted(p.name for p in mirrored.iterdir()) == ["equity.jsonl", "state.json"]
    # A directory that does not exist mirrors to an empty one, not to an error.
    assert list(mirror_run_dir(tmp_path / "absent", tmp_path / "m2").iterdir()) == []


def test_a_pass_that_offers_nothing_announces_nothing(tmp_path: Path, caplog) -> None:
    """Resubmitting a batch the venue already holds is not a trade, and says so."""
    order = Order(client_order_id="x:1", symbol="AAA", qty=1.0, decision_bar="2024-07-01", reference_price=10.0)
    venue = VenueDouble(known_ids={"x:1"})
    ctx = LiveLoopContext(
        store=StateStore(tmp_path / "state.json"), broker=venue, fills_ledger=tmp_path / "fills.jsonl"
    )
    state = LoopState(pending_orders=[order], pending_decision_bar="2024-07-01")
    with caplog.at_level(logging.INFO):
        outcome = submit_pending(ctx, state, [order])
    assert outcome.submitted == () and outcome.accepted == ("x:1",)
    assert _announcements(caplog) == []
