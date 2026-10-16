"""The per-deployment seed and the seeded rebalance phase (``prism.live.deployment``).

Every deployment computes the same book, and until a seed existed the only thing
that set when a monthly book traded was its first session, so users who started
together traded together. Pinned here:

* the profile file: written atomically with mode 0600 whatever the umask, never
  overwritten, refused when malformed, and its seed kept out of ``repr``;
* the draws: HMAC-SHA256 of ``"<purpose>:<bar>"`` keyed by the seed, the same
  for one seed and different across seeds, and the phase offset in range and
  spread over the whole cycle;
* who creates the file: only a new deployment's first submitting session
  (``--submit --allow-new-run-dir``); a seeded mode with no file raises;
* the cycle (``run_daily_cycle(phase_offset=...)``): off is bit-identical; on,
  the book waits for its first grid date, refreshes on its grid, catches up
  once after a missed grid date, and agrees with composition's predicate on
  every bar; inputs it cannot honour raise before any state moves;
* the CLI (``paper_loop --rebalance-phase``): the default writes no seed and
  passes no offset; ``seeded`` creates the seed on a new deployment's first
  session and reads it afterwards, in a rehearsal too; a seeded run with no
  seed is refused before any venue read; the certified research_paper path and
  a one-session cadence are refused.

Nothing here opens a socket.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import stat
from collections import Counter
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from prism.composition import ComponentPlan, ComponentState, due_on_grid
from prism.io import sessions
from prism.live import (
    DailyBookConfig,
    DailyCycleResult,
    LiveLoopContext,
    StateStore,
    run_daily_cycle,
)
from prism.live.deployment import (
    DEPLOYMENT_FILENAME,
    DEPLOYMENT_SCHEMA,
    REBALANCE_PHASE_PURPOSE,
    DeploymentProfile,
    DeploymentProfileError,
    DeploymentProfileMissing,
    establish_deployment,
    read_deployment,
    write_deployment,
)
from prism.live.safety import RunDirHistory, sample_run_dir_history
from prism.scripts import paper_loop
from tests.test_live_daily import ConstSignal, _full_ctx
from tests.test_live_loop import FakeBroker
from tests.test_paper_no_submit import (
    BarSourceDouble,
    VenueDouble,
    _established,
    _loop_argv,
    _panels,
    _point_switch,
    _run_session,
)

SEED_A = "a" * 64
SEED_B = "0123456789abcdef" * 4


def _profile(seed: str = SEED_A, created_bar: str = "2026-10-15") -> DeploymentProfile:
    return DeploymentProfile(seed_hex=seed, created_bar=created_bar)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def test_the_profile_is_written_owner_only_and_reads_back(tmp_path: Path) -> None:
    profile = _profile()
    previous = os.umask(0)  # the most permissive umask: the mode must not depend on it
    try:
        path = write_deployment(tmp_path / "run", profile)
    finally:
        os.umask(previous)
    assert path == tmp_path / "run" / DEPLOYMENT_FILENAME
    assert _mode(path) == 0o600
    assert read_deployment(tmp_path / "run") == profile
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema": DEPLOYMENT_SCHEMA,
        "seed_hex": SEED_A,
        "created_bar": "2026-10-15",
    }
    assert not list((tmp_path / "run").glob("*.tmp"))


def test_a_leftover_temp_file_does_not_carry_its_mode_into_the_profile(tmp_path: Path) -> None:
    tmp = tmp_path / (DEPLOYMENT_FILENAME + ".tmp")
    tmp.write_text("torn", encoding="utf-8")
    tmp.chmod(0o644)
    path = write_deployment(tmp_path, _profile())
    assert _mode(path) == 0o600
    assert read_deployment(tmp_path).seed_hex == SEED_A


def test_an_existing_profile_is_never_overwritten(tmp_path: Path) -> None:
    write_deployment(tmp_path, _profile(SEED_A))
    before = (tmp_path / DEPLOYMENT_FILENAME).read_bytes()
    with pytest.raises(DeploymentProfileError, match="refusing to overwrite"):
        write_deployment(tmp_path, _profile(SEED_B))
    assert (tmp_path / DEPLOYMENT_FILENAME).read_bytes() == before


def test_a_missing_profile_raises(tmp_path: Path) -> None:
    with pytest.raises(DeploymentProfileMissing):
        read_deployment(tmp_path)


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"schema": 2, "seed_hex": SEED_A, "created_bar": "2026-10-15"}, "schema"),
        ({"schema": True, "seed_hex": SEED_A, "created_bar": "2026-10-15"}, "schema"),
        ({"schema": 1, "seed_hex": "A" * 64, "created_bar": "2026-10-15"}, "lowercase hex"),
        ({"schema": 1, "seed_hex": "a" * 63, "created_bar": "2026-10-15"}, "lowercase hex"),
        ({"schema": 1, "seed_hex": SEED_A, "created_bar": "2026-13-01"}, "ISO date"),
        ({"schema": 1, "seed_hex": SEED_A, "created_bar": 20261015}, "ISO date"),
        ({"schema": 1, "seed_hex": SEED_A}, "missing"),
        ({"schema": 1, "seed_hex": SEED_A, "created_bar": "2026-10-15", "phase": 3}, "unexpected"),
        ([1, 2, 3], "JSON object"),
    ],
)
def test_a_malformed_profile_is_refused(tmp_path: Path, payload, match) -> None:
    (tmp_path / DEPLOYMENT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DeploymentProfileError, match=match):
        read_deployment(tmp_path)


def test_a_profile_that_is_not_json_is_refused(tmp_path: Path) -> None:
    (tmp_path / DEPLOYMENT_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(DeploymentProfileError, match="unreadable"):
        read_deployment(tmp_path)


def test_a_readable_by_others_profile_loads_with_a_warning(tmp_path: Path, caplog) -> None:
    path = write_deployment(tmp_path, _profile())
    path.chmod(0o644)
    with caplog.at_level(logging.WARNING, logger="prism.live.deployment"):
        assert read_deployment(tmp_path).seed_hex == SEED_A
    assert any("should be 600" in r.getMessage() for r in caplog.records)


def test_the_seed_stays_out_of_repr() -> None:
    assert SEED_A not in repr(_profile())
    assert "2026-10-15" in repr(_profile())


def test_a_new_profile_draws_a_fresh_seed() -> None:
    first, second = DeploymentProfile.new("2026-10-15"), DeploymentProfile.new("2026-10-15")
    assert first.seed_hex != second.seed_hex
    assert len(first.seed_hex) == 64 and first.created_bar == "2026-10-15"


# ---------------------------------------------------------------------------
# The draws
# ---------------------------------------------------------------------------


def test_a_draw_is_hmac_sha256_of_purpose_and_bar_keyed_by_the_seed() -> None:
    profile = _profile(SEED_B)
    digest = hmac.new(bytes.fromhex(SEED_B), b"sweep_delay:2026-10-15", hashlib.sha256).digest()
    word = int.from_bytes(digest[:8], "big")
    assert profile.draw("sweep_delay", "2026-10-15") == (word >> 11) * 2.0**-53
    assert profile.draw_index("sweep_delay", "2026-10-15", 31) == (word * 31) >> 64


def test_draws_are_fixed_by_the_seed_and_differ_across_seeds() -> None:
    bars = [d.isoformat() for d in sessions.sessions_between(dt.date(2026, 1, 2), dt.date(2026, 3, 31))]
    a, a_again, b = _profile(SEED_A), _profile(SEED_A), _profile(SEED_B)
    draws_a = [a.draw("p", bar) for bar in bars]
    assert draws_a == [a_again.draw("p", bar) for bar in bars]
    assert draws_a != [b.draw("p", bar) for bar in bars]
    assert all(0.0 <= value < 1.0 for value in draws_a)
    assert len(set(draws_a)) == len(bars), "each bar gets its own draw"
    assert a.draw("p", bars[0]) != a.draw("q", bars[0]), "each purpose gets its own draw"


def test_a_draw_refuses_a_purpose_that_could_collide() -> None:
    profile = _profile()
    for purpose in ("", "a:b"):
        with pytest.raises(DeploymentProfileError, match="purpose"):
            profile.draw(purpose, "2026-10-15")
    for n in (0, -1, True, 2.0):
        with pytest.raises(DeploymentProfileError, match="positive integer"):
            profile.draw_index("p", "2026-10-15", n)


def test_the_phase_offset_is_fixed_in_range_and_spread_over_the_cycle() -> None:
    profile = _profile(SEED_B)
    offset = profile.phase_offset(21)
    assert offset == profile.draw_index(REBALANCE_PHASE_PURPOSE, profile.created_bar, 21)
    assert offset == _profile(SEED_B).phase_offset(21)
    # 2,100 deployments: every offset of the 21 occurs, none implausibly often
    # (expectation 100 each; the bounds are far outside sampling noise).
    seeds = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(2100)]
    counts = Counter(_profile(seed).phase_offset(21) for seed in seeds)
    assert set(counts) == set(range(21))
    assert 50 <= min(counts.values()) and max(counts.values()) <= 150


# ---------------------------------------------------------------------------
# Who creates the file
# ---------------------------------------------------------------------------


def _history(run_dir: Path) -> RunDirHistory:
    return sample_run_dir_history(run_dir)


def test_a_new_deployments_first_submitting_session_creates_the_profile(tmp_path: Path) -> None:
    run_dir = tmp_path / "new"
    history = _history(run_dir)
    profile = establish_deployment(
        run_dir, history=history, submit=True, allow_new_run_dir=True, created_bar="2026-10-15"
    )
    assert read_deployment(run_dir) == profile
    assert _mode(run_dir / DEPLOYMENT_FILENAME) == 0o600
    # A re-run of that first session (it crashed before any session record) keeps the seed.
    again = establish_deployment(
        run_dir, history=_history(run_dir), submit=True, allow_new_run_dir=True, created_bar="2026-10-16"
    )
    assert again == profile


def test_an_existing_profile_is_read_by_every_later_session_and_rehearsal(tmp_path: Path) -> None:
    run_dir = _established(tmp_path)
    write_deployment(run_dir, _profile())
    for submit, allow_new in ((True, False), (False, False), (True, True)):
        profile = establish_deployment(
            run_dir, history=_history(run_dir), submit=submit, allow_new_run_dir=allow_new, created_bar="2027-01-04"
        )
        assert profile == _profile()


@pytest.mark.parametrize(
    "established, submit, allow_new, match",
    [
        (False, False, False, "A rehearsal never creates one"),
        (True, True, False, "already holds sessions"),
        (True, True, True, "already holds sessions"),
        (False, True, False, "Only the first session of a new deployment"),
    ],
)
def test_a_seeded_mode_without_a_profile_raises(tmp_path: Path, established, submit, allow_new, match) -> None:
    run_dir = _established(tmp_path) if established else tmp_path / "runs" / "fresh"
    with pytest.raises(DeploymentProfileMissing, match=match):
        establish_deployment(
            run_dir, history=_history(run_dir), submit=submit, allow_new_run_dir=allow_new, created_bar="2026-10-15"
        )
    assert not (run_dir / DEPLOYMENT_FILENAME).exists()


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------

#: Sixty consecutive exchange sessions from 2026-07-06.
DAYS = [d.isoformat() for d in sessions.sessions_between(dt.date(2026, 7, 6), dt.date(2026, 12, 31))[:60]]
MONTHLY = DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=21)
SIGNAL = ConstSignal({"AAA": 1.0, "BBB": -1.0})


def _panel(through: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flat AAA/BBB closes on ``DAYS[: through + 1]``, so the last bar is ``DAYS[through]``."""
    index = pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day in DAYS[: through + 1]])
    close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=index)
    return close, pd.DataFrame(1e6, index=index, columns=close.columns)


def _ctx(root: Path) -> LiveLoopContext:
    return LiveLoopContext(
        store=StateStore(root / "state.json"), broker=FakeBroker(), fills_ledger=root / "fills.jsonl"
    )


def _offset_landing_on(bar: str, every: int = 21) -> int:
    return sessions.grid_position(dt.date.fromisoformat(bar)) % every


def _run(ctx: LiveLoopContext, numbers, phase_offset: int | None, config: DailyBookConfig = MONTHLY) -> list[str]:
    """Run one cycle per session number; return the bars on which the book refreshed."""
    refreshed = []
    for number in numbers:
        state = ctx.store.load()
        before = None if state is None else state.last_refresh_bar
        result = run_daily_cycle(ctx, SIGNAL, *_panel(number), config, phase_offset=phase_offset)
        after = ctx.store.load().last_refresh_bar
        if after != before:
            assert after == result.decision_bar
            refreshed.append(result.decision_bar)
    return refreshed


def test_off_the_cycle_is_bit_identical(tmp_path: Path) -> None:
    config = DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=5)
    results: dict[str, list[DailyCycleResult]] = {}
    for name, kwargs in (("bare", {}), ("off", {"phase_offset": None})):
        ctx = _full_ctx(tmp_path / name)
        results[name] = [run_daily_cycle(ctx, SIGNAL, *_panel(n), config, **kwargs) for n in range(5, 13)]
    for bare, off in zip(results["bare"], results["off"]):
        for field in dataclass_fields(DailyCycleResult):
            left, right = getattr(bare, field.name), getattr(off, field.name)
            if isinstance(left, pd.Series):
                pd.testing.assert_series_equal(left, right)
            else:
                assert left == right, field.name
    for ledger in ("state.json", "fills.jsonl", "equity.jsonl", "targets.jsonl", "concordance.jsonl"):
        assert (tmp_path / "bare" / ledger).exists() is (tmp_path / "off" / ledger).exists(), ledger
        if (tmp_path / "bare" / ledger).exists():
            assert (tmp_path / "bare" / ledger).read_bytes() == (tmp_path / "off" / ledger).read_bytes(), ledger
    assert (tmp_path / "bare" / "targets.jsonl").read_text(encoding="utf-8").count("\n") == 2  # two refreshes
    assert not (tmp_path / "off" / DEPLOYMENT_FILENAME).exists()


def test_a_new_seeded_book_waits_for_its_grid_date_then_keeps_to_its_grid(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    offset = _offset_landing_on(DAYS[10])
    refreshed = _run(ctx, range(3, 60), offset)
    assert refreshed == [DAYS[10], DAYS[31], DAYS[52]]
    # Before its first grid date the book holds nothing and decides nothing.
    fresh = _ctx(tmp_path / "fresh")
    for number in range(3, 10):
        result = run_daily_cycle(fresh, SIGNAL, *_panel(number), MONTHLY, phase_offset=offset)
        assert result.submitted_orders == [] and fresh.store.load().last_refresh_bar is None
    first = run_daily_cycle(fresh, SIGNAL, *_panel(10), MONTHLY, phase_offset=offset)
    assert len(first.submitted_orders) == 2 and fresh.store.load().last_refresh_bar == DAYS[10]


def test_the_first_session_rule_refreshes_at_once_where_the_seeded_one_waits(tmp_path: Path) -> None:
    offset = _offset_landing_on(DAYS[10])
    assert _run(_ctx(tmp_path / "first"), range(3, 60), None) == [DAYS[3], DAYS[24], DAYS[45]]
    assert _run(_ctx(tmp_path / "seeded"), range(3, 60), offset) == [DAYS[10], DAYS[31], DAYS[52]]


def test_a_missed_grid_date_is_caught_up_once_and_the_book_returns_to_its_grid(tmp_path: Path) -> None:
    offset = _offset_landing_on(DAYS[3])
    down = set(range(22, 26))  # an outage across the grid date DAYS[24]
    numbers = [n for n in range(3, 60) if n not in down]
    refreshed = _run(_ctx(tmp_path), numbers, offset)
    # DAYS[26] is the first session after more than 21 elapsed since DAYS[3]; DAYS[45] is back on the grid.
    assert refreshed == [DAYS[3], DAYS[26], DAYS[45]]

    # The same bars as composition's predicate for a plan with that cadence and offset.
    plan = ComponentPlan(
        component=SIGNAL,
        component_id="book",
        to_weights="top_k_equal_weight",
        to_weights_params={"k": 1, "max_gross": 1.0},
        update_every_bars=21,
        valid_for_bars=42,
        phase_offset_bars=offset,
    )
    last: int | None = None
    expected = []
    for number in numbers:
        state = ComponentState(component_id="book", last_output_bar=None if last is None else DAYS[last])
        if due_on_grid(plan, state, DAYS[number], elapsed=0 if last is None else number - last):
            expected.append(DAYS[number])
            last = number
    assert refreshed == expected


def test_a_same_bar_rerun_on_a_grid_date_does_not_refresh_twice(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    offset = _offset_landing_on(DAYS[10])
    first = run_daily_cycle(ctx, SIGNAL, *_panel(10), MONTHLY, phase_offset=offset)
    again = run_daily_cycle(ctx, SIGNAL, *_panel(10), MONTHLY, phase_offset=offset)
    assert [o.client_order_id for o in again.submitted_orders] == [o.client_order_id for o in first.submitted_orders]
    assert len(ctx.broker.orders) == 2


@pytest.mark.parametrize(
    "offset, every, match",
    [
        (21, 21, r"\[0, decision_every=21\)"),
        (-1, 21, r"\[0, decision_every=21\)"),
        (True, 21, r"\[0, decision_every=21\)"),
        (2.0, 21, r"\[0, decision_every=21\)"),
        (0, 1, "no phase to set"),
    ],
)
def test_a_schedule_the_cycle_cannot_honour_raises_before_any_state_moves(
    tmp_path: Path, offset, every, match
) -> None:
    ctx = _ctx(tmp_path)
    config = DailyBookConfig(position_size=0.1, min_order_notional=1.0, decision_every=every)
    with pytest.raises(ValueError, match=match):
        run_daily_cycle(ctx, SIGNAL, *_panel(10), config, phase_offset=offset)
    assert ctx.store.load() is None and not ctx.fills_ledger.exists()


def test_a_decision_bar_that_is_not_a_session_raises_before_any_state_moves(tmp_path: Path) -> None:
    ctx = _full_ctx(tmp_path)
    days = sessions.sessions_between(dt.date(2026, 6, 1), dt.date(2026, 7, 2)) + [dt.date(2026, 7, 3)]
    assert not sessions.is_session(dt.date(2026, 7, 3))  # Independence Day, observed
    index = pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day in days])
    close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=index)
    with pytest.raises(ValueError, match="no session-grid position"):
        run_daily_cycle(ctx, SIGNAL, close, None, MONTHLY, phase_offset=0)
    assert ctx.store.load() is None and not (tmp_path / "panels").exists()


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch) -> VenueDouble:
    """``paper_loop`` wired to in-memory doubles, the way ``tests/test_paper_no_submit.py`` wires it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret-key")
    monkeypatch.delenv("APCA_API_BASE_URL", raising=False)
    broker = VenueDouble()
    close, volume = _panels()
    monkeypatch.setattr(paper_loop, "AlpacaBroker", SimpleNamespace(from_env=lambda **kw: broker), raising=True)
    monkeypatch.setattr(
        paper_loop,
        "AlpacaBarSource",
        SimpleNamespace(from_env=lambda **kw: BarSourceDouble(close, volume)),
        raising=True,
    )
    return broker


@pytest.fixture()
def cycles(monkeypatch) -> list[dict]:
    """The keyword arguments of every ``run_daily_cycle`` call ``paper_loop.main`` makes."""
    calls: list[dict] = []
    real = paper_loop.run_daily_cycle

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(paper_loop, "run_daily_cycle", spy, raising=True)
    return calls


SEEDED = ("--rebalance-phase", "seeded", "--decision-every", "21")


def test_the_default_writes_no_seed_and_passes_no_offset(wired: VenueDouble, tmp_path: Path, cycles) -> None:
    run_dir = tmp_path / "runs" / "plain"
    _point_switch(run_dir)
    assert paper_loop.main(_loop_argv(run_dir, "--submit", "--allow-new-run-dir", "--decision-every", "21")) == 0
    assert cycles[-1]["phase_offset"] is None
    assert not (run_dir / DEPLOYMENT_FILENAME).exists()
    assert wired.submitted, "the first-session rule refreshes on the first session"


def test_seeded_creates_the_seed_on_the_first_session_and_reads_it_after(
    wired: VenueDouble, tmp_path: Path, cycles, monkeypatch
) -> None:
    run_dir = tmp_path / "runs" / "seeded"
    _point_switch(run_dir)
    _run_session(monkeypatch, run_dir, 400, "--submit", "--allow-new-run-dir", *SEEDED)
    path = run_dir / DEPLOYMENT_FILENAME
    assert _mode(path) == 0o600
    profile = read_deployment(run_dir)
    offset = profile.phase_offset(21)
    assert cycles[-1]["phase_offset"] == offset
    # The fixture's decision bar is 2025-07-14: the book refreshed there only if that is its grid date.
    on_grid = sessions.grid_position(dt.date(2025, 7, 14)) % 21 == offset
    assert bool(wired.submitted) is on_grid
    recorded = path.read_bytes()

    # The next session and a rehearsal of the one after read the same seed and pass the same offset.
    _run_session(monkeypatch, run_dir, 401, "--submit", *SEEDED)
    wired.forbid_writes = True
    _run_session(monkeypatch, run_dir, 402, "--no-submit", *SEEDED)
    assert [call["phase_offset"] for call in cycles] == [offset, offset, offset]
    assert path.read_bytes() == recorded and _mode(path) == 0o600


def test_a_seeded_rehearsal_of_a_new_directory_is_refused_before_any_venue_read(
    wired: VenueDouble, tmp_path: Path, cycles
) -> None:
    run_dir = tmp_path / "runs" / "fresh"
    wired.forbid_writes = True
    with pytest.raises(DeploymentProfileMissing, match="A rehearsal never creates one"):
        paper_loop.main(_loop_argv(run_dir, "--no-submit", *SEEDED))
    assert wired.reads == [] and cycles == []
    assert not (run_dir / DEPLOYMENT_FILENAME).exists()


def test_an_established_directory_without_a_seed_is_refused_before_any_venue_read(
    wired: VenueDouble, tmp_path: Path, cycles
) -> None:
    run_dir = _established(tmp_path)
    _point_switch(run_dir)
    with pytest.raises(DeploymentProfileMissing, match="already holds sessions"):
        paper_loop.main(_loop_argv(run_dir, "--submit", *SEEDED))
    assert wired.reads == [] and wired.submitted == [] and cycles == []
    assert not (run_dir / DEPLOYMENT_FILENAME).exists()


@pytest.mark.parametrize(
    "extra, match",
    [
        (("--rebalance-phase", "seeded", "--decision-every", "1"), "cadence above one session"),
        (("--rebalance-phase", "seeded", "--profile", "research_paper"), "research_paper"),
    ],
)
def test_seeded_is_refused_where_it_has_no_meaning(wired: VenueDouble, tmp_path: Path, cycles, extra, match) -> None:
    run_dir = _established(tmp_path)
    _point_switch(run_dir)
    with pytest.raises(SystemExit, match=match):
        paper_loop.main(_loop_argv(run_dir, "--submit", *extra))
    assert wired.reads == [] and cycles == []
