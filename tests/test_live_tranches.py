"""The tranched paper book (``run_daily_cycle(tranches=K)``, ``prism.live.tranches``).

Pinned here:

* off (``tranches=1``) is bit-identical and never enters the tranche code;
* a 21-tranche state survives a round trip through the state store, and a
  component without the tranche fields serializes exactly as before;
* staged entry: a new book builds one tranche per session on its grid dates,
  each tranche sized at one twenty-first of the equity of its own session;
* each tranche is held in shares between its refreshes: a session decides only
  the names the refreshing tranche held or holds (plus catch-ups), its orders
  are that tranche's change, a split is restated from the panel, and a state
  without share holdings is refused;
* every session of a built 21-tranche book has exactly one due tranche, and a
  3-tranche book refreshes on three sessions a cycle and holds in between;
* restarts: a same-bar re-run and a process restart in the middle of the build
  reproduce the uninterrupted run byte for byte, and a decision the order guard
  refused advances no tranche;
* catch-up: an outage is caught up on the return session and the tranches go
  back to their grid; an outage of more than two cycles expires and rebuilds;
* wind-down takes 21 sessions and leaves the account flat;
* a tranche's refresh is its whole book, and an entry the refreshing tranche
  cannot price is deferred instead of failing the sizing step;
* the volatility scale per tranche: the scale is the rule on an independently
  rebuilt tranched base book, and a failure is bounded by the tranche's own
  last good scale;
* refusals, all before any state moves, and the CLI flags.

Nothing here opens a socket.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prism.io import sessions
from prism.live import (
    DailyBookConfig,
    DailyCycleResult,
    LiveLoopContext,
    SafetyConfig,
    StateStore,
)
from prism.live import daily as daily_module
from prism.live import (
    run_daily_cycle,
)
from prism.live.replay import ReplayBroker, replay_daily_cycles
from prism.live.safety import SafetyViolation
from prism.live.state import ComponentState, LoopState
from prism.live.tranches import TARGETS_TRANCHES_KEY, TrancheError
from prism.live.vol_sizing import (
    SIZING_LEDGER_FILENAME,
    TARGETS_LEDGER_KEY,
    required_panel_bars,
)
from prism.portfolio.vol_scale import VolScaleRule
from prism.scripts import paper_loop
from prism.scripts.paper_books import resolve_tranches
from tests.test_live_daily import ConstSignal, RecordingRegime
from tests.test_live_loop import FakeBroker
from tests.test_live_vol_scale import TrailingReturn
from tests.test_paper_no_submit import (
    BarSourceDouble,
    VenueDouble,
    _established,
    _loop_argv,
    _panels,
    _point_switch,
)

N_SYM = 10
EVERY = 21
NAMES = [f"S{i}" for i in range(N_SYM)]
#: Real exchange sessions, so every bar has a session-grid position.
DAYS = sessions.sessions_between(dt.date(2025, 1, 2), dt.date(2026, 6, 30))
#: Distinct fixed scores: the decile book (two names a leg at decile 0.2) is the same at every refresh.
SCORES = {name: float(i) for i, name in enumerate(NAMES)}
CONFIG = DailyBookConfig(
    book="decile_neutral",
    decile=0.2,
    decision_every=EVERY,
    max_gross=1.0,
    max_symbol_abs_weight=1.0,
    min_order_notional=1.0,
    whole_shares=False,
)
#: The single decile book the constant scores give: +0.25 on S8 and S9, -0.25 on S0 and S1.
BOOK = pd.Series({**{name: 0.0 for name in NAMES}, "S0": -0.25, "S1": -0.25, "S8": 0.25, "S9": 0.25})


def _walk(n_bars: int = 120, *, seed: int = 11, flat_from: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    moves = rng.normal(0.0, 0.015, size=(n_bars, N_SYM))
    moves[0] = 0.0
    levels = 100.0 * np.exp(np.cumsum(moves, axis=0))
    if flat_from is not None:
        levels[flat_from:] = levels[flat_from - 1]
    index = pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day in DAYS[:n_bars]])
    close = pd.DataFrame(levels, index=index, columns=NAMES)
    return close, pd.DataFrame(1e6, index=index, columns=NAMES)


def _ctx(root: Path, broker=None) -> LiveLoopContext:
    root.mkdir(parents=True, exist_ok=True)
    return LiveLoopContext(
        store=StateStore(root / "state.json"),
        broker=broker if broker is not None else ReplayBroker(1_000_000.0),
        fills_ledger=root / "fills.jsonl",
        equity_ledger=root / "equity.jsonl",
        targets_ledger=root / "targets.jsonl",
        unfilled_ledger=root / "unfilled.jsonl",
        concordance_ledger=root / "concordance.jsonl",
    )


def _session(ctx: LiveLoopContext, close, volume, i: int, *, signal=None, config=CONFIG, **kwargs) -> DailyCycleResult:
    """One cycle at bar ``i``; a replay venue fills the previous decision at this bar's close."""
    if isinstance(ctx.broker, ReplayBroker):
        ctx.broker.set_fill(close.iloc[i], str(close.index[i].date()))
    return run_daily_cycle(
        ctx,
        signal if signal is not None else ConstSignal(SCORES),
        close.iloc[: i + 1],
        volume.iloc[: i + 1],
        config,
        **kwargs,
    )


def _bar(close: pd.DataFrame, i: int) -> str:
    return str(close.index[i].date())


def _grid(close: pd.DataFrame, i: int) -> int:
    return sessions.grid_position(close.index[i].date())


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _components(ctx: LiveLoopContext) -> dict[str, ComponentState]:
    state = ctx.store.load()
    return {} if state is None else {c.component_id: c for c in state.components}


def _tranche_id(j: int) -> str:
    return f"decile_neutral/t{j:02d}"


def _held_target(close: pd.DataFrame, refreshes: list[tuple[int, float]], at: int, *, k: int = 21) -> pd.Series:
    """The account target of tranches built from ``BOOK``, each held in shares since its refresh.

    ``refreshes`` are ``(bar, marked equity)`` of the tranches currently held:
    each holds ``BOOK / k * equity / close`` shares from that bar, and the
    target at ``at`` is their shares times the close at ``at`` over the equity
    there, which the caller supplies as the last entry's equity when ``at`` is
    the last refresh, or reads from the result.
    """
    shares = sum(BOOK.reindex(NAMES) / k * equity / close.iloc[bar][NAMES] for bar, equity in refreshes)
    return shares


def _ledger_target(ctx: LiveLoopContext, bar: str) -> pd.Series:
    row = next(row for row in _rows(Path(ctx.targets_ledger)) if row["refresh_bar"] == bar)
    return pd.Series(row["targets"], dtype=float).reindex(NAMES).fillna(0.0)


# ---------------------------------------------------------------------------
# Off is bit-identical
# ---------------------------------------------------------------------------


def test_off_is_bit_identical_and_never_enters_the_tranche_code(tmp_path: Path, monkeypatch) -> None:
    close, volume = _walk(60)
    monthly = replace(CONFIG, decision_every=5, whole_shares=True)
    signal = TrailingReturn()
    bare_ctx = _ctx(tmp_path / "bare", FakeBroker(cash=1_000_000.0))
    bare = [_session(bare_ctx, close, volume, i, signal=signal, config=monthly) for i in range(10, 30)]

    def refuse(*args, **kwargs):
        raise AssertionError("the single book entered the tranche code")

    monkeypatch.setattr(daily_module, "open_tranche_session", refuse)
    monkeypatch.setattr(daily_module, "decide_tranches", refuse)
    off_ctx = _ctx(tmp_path / "off", FakeBroker(cash=1_000_000.0))
    off = [
        _session(off_ctx, close, volume, i, signal=signal, config=monthly, tranches=1, wind_down=False)
        for i in range(10, 30)
    ]
    for left, right in zip(bare, off):
        for field in dataclass_fields(DailyCycleResult):
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, pd.Series):
                pd.testing.assert_series_equal(a, b)
            else:
                assert a == b, field.name
        assert right.tranches is None
    for ledger in ("state.json", "fills.jsonl", "equity.jsonl", "targets.jsonl", "concordance.jsonl"):
        assert (tmp_path / "bare" / ledger).read_bytes() == (tmp_path / "off" / ledger).read_bytes(), ledger
    assert json.loads((tmp_path / "off" / "state.json").read_text())["components"] == []
    assert TARGETS_TRANCHES_KEY not in (tmp_path / "off" / "targets.jsonl").read_text(encoding="utf-8")


def test_a_component_without_the_tranche_fields_serializes_as_before() -> None:
    plain = ComponentState(component_id="momentum", last_output_bar="2026-07-01", sub_targets={"AAA": 0.1})
    assert set(plain.to_dict()) == {
        "component_id",
        "last_output_bar",
        "decision_bar",
        "expires_bar",
        "expired",
        "sub_targets",
        "fitted_state_ref",
        "fitted_through_bar",
        "fit_scope",
    }
    assert ComponentState.from_dict(plain.to_dict()) == plain


def test_a_holding_round_trips_and_a_malformed_one_is_refused_on_load() -> None:
    entry = {"shares": -3.5, "bar": "2026-07-01", "close": 10.25}
    held = ComponentState(component_id=_tranche_id(0), holdings={"AAA": dict(entry)})
    assert held.to_dict()["holdings"] == {"AAA": entry}
    assert ComponentState.from_dict(json.loads(json.dumps(held.to_dict()))) == held
    for bad in (
        {"AAA": {"shares": 3.0, "bar": "2026-07-01"}},
        {"AAA": {**entry, "shares": "three"}},
        {"AAA": {**entry, "shares": float("nan")}},
        {"AAA": {**entry, "close": 0.0}},
        {"AAA": {**entry, "shares": True}},
        [entry],
    ):
        with pytest.raises(ValueError):
            ComponentState.from_dict({**held.to_dict(), "holdings": bad})


def test_a_21_tranche_state_survives_a_round_trip(tmp_path: Path) -> None:
    close, volume = _walk(60)
    ctx = _ctx(tmp_path)
    for i in range(30, 52):
        _session(ctx, close, volume, i, tranches=21)
    raw = (tmp_path / "state.json").read_bytes()
    state = ctx.store.load()
    assert len(state.components) == 21
    assert [c.phase_offset_bars for c in state.components] == list(range(21))
    copy = StateStore(tmp_path / "copy.json")
    copy.save(state)
    assert copy.load() == state
    assert (tmp_path / "copy.json").read_bytes() == raw


# ---------------------------------------------------------------------------
# Staged entry and the schedule
# ---------------------------------------------------------------------------


def test_staged_entry_builds_one_tranche_per_session_on_its_grid_date(tmp_path: Path) -> None:
    close, volume = _walk(80)
    ctx = _ctx(tmp_path)
    start = 30
    refreshes: list[tuple[int, float]] = []
    for n in range(1, 22):
        i = start + n - 1
        result = _session(ctx, close, volume, i, tranches=21)
        expected_due = _tranche_id(_grid(close, i) % EVERY)
        assert result.tranches["due"] == [expected_due]
        assert result.tranches["built"] == n
        built = [c for c in _components(ctx).values() if c.last_output_bar is not None]
        assert len(built) == n
        # Every tranche was first built on its own grid date.
        for component in built:
            assert sessions.grid_position(dt.date.fromisoformat(component.last_output_bar)) % EVERY == (
                component.phase_offset_bars
            )
        # Constant scores: each built tranche holds BOOK/21 of the equity of its own session, in shares,
        # so the target is those shares at this close over this equity.
        refreshes.append((i, result.equity))
        expected = _held_target(close, refreshes, i) * close.iloc[i][NAMES] / result.equity
        np.testing.assert_allclose(result.target_weights.reindex(NAMES).to_numpy(), expected.to_numpy(), atol=1e-12)
        np.testing.assert_allclose(_ledger_target(ctx, _bar(close, i)).to_numpy(), expected.to_numpy(), atol=1e-12)
        holding = _components(ctx)[expected_due].holdings
        assert set(holding) == {"S0", "S1", "S8", "S9"}
        assert all(entry["bar"] == _bar(close, i) for entry in holding.values())
        assert result.concordance is None
    assert not (tmp_path / "concordance.jsonl").exists()
    rows = _rows(tmp_path / "targets.jsonl")
    assert [row["refresh_bar"] for row in rows] == [_bar(close, i) for i in range(start, start + 21)]
    assert [row[TARGETS_TRANCHES_KEY]["built"] for row in rows] == list(range(1, 22))


def test_a_built_book_refreshes_exactly_one_tranche_every_session(tmp_path: Path) -> None:
    close, volume = _walk(120)
    results, _ = replay_daily_cycles(
        lambda: ConstSignal(SCORES),
        close,
        volume,
        close,
        CONFIG,
        tmp_path / "replay",
        start_bar=_bar(close, 20),
        initial_cash=1_000_000.0,
        tranches=21,
    )
    assert len(results) == 100
    assert all(len(r.tranches["due"]) == 1 for r in results)
    for r in results:
        offset = int(r.tranches["due"][0].rsplit("/t", 1)[1])
        assert sessions.grid_position(dt.date.fromisoformat(r.decision_bar)) % EVERY == offset
    assert [r.tranches["built"] for r in results[:21]] == list(range(1, 22))
    assert {r.tranches["built"] for r in results[21:]} == {21}


def test_a_three_tranche_book_refreshes_three_times_a_cycle_and_holds_between(tmp_path: Path) -> None:
    close, volume = _walk(80)
    ctx = _ctx(tmp_path)
    base = 4
    refreshed = []
    for i in range(30, 72):
        result = _session(ctx, close, volume, i, tranches=3, phase_offset=base)
        assert result.tranches["base_offset"] == base
        if result.tranches["due"]:
            refreshed.append((i, result.tranches["due"]))
    offsets = {_tranche_id(j): (base + 7 * j) % EVERY for j in range(3)}
    assert len(refreshed) == 6
    for i, due in refreshed:
        assert len(due) == 1 and (_grid(close, i) - offsets[due[0]]) % EVERY == 0
    assert [c.phase_offset_bars for c in ctx.store.load().components] == [4, 11, 18]
    assert len(_rows(tmp_path / "targets.jsonl")) == 6


# ---------------------------------------------------------------------------
# Restarts
# ---------------------------------------------------------------------------


def test_a_restart_in_the_middle_of_the_build_reproduces_the_uninterrupted_run(tmp_path: Path) -> None:
    close, volume = _walk(80)
    straight = _ctx(tmp_path / "straight")
    for i in range(30, 40):
        _session(straight, close, volume, i, tranches=21)

    broker = ReplayBroker(1_000_000.0)
    interrupted = _ctx(tmp_path / "interrupted", broker)
    for i in range(30, 35):
        _session(interrupted, close, volume, i, tranches=21)
    before = (tmp_path / "interrupted" / "state.json").read_bytes()
    again = _session(interrupted, close, volume, 34, tranches=21)  # same-bar re-run: resumes, decides nothing new
    assert again.tranches["due"] == []
    assert (tmp_path / "interrupted" / "state.json").read_bytes() == before
    restarted = _ctx(tmp_path / "interrupted", broker)  # a new process on the same run directory and venue
    for i in range(35, 40):
        _session(restarted, close, volume, i, tranches=21)

    for ledger in ("state.json", "fills.jsonl", "equity.jsonl", "targets.jsonl"):
        assert (tmp_path / "straight" / ledger).read_bytes() == (tmp_path / "interrupted" / ledger).read_bytes(), (
            ledger
        )


def test_a_decision_the_order_guard_refuses_advances_no_tranche(tmp_path: Path) -> None:
    close, volume = _walk(80)
    ctx = _ctx(tmp_path)
    for i in range(30, 33):
        _session(ctx, close, volume, i, tranches=21)
    before = [c.to_dict() for c in ctx.store.load().components]
    with pytest.raises(SafetyViolation):
        _session(ctx, close, volume, 33, tranches=21, safety=SafetyConfig(max_orders=1))
    # The previous decision settled (a state move of its own); the refused one persisted nothing.
    state = ctx.store.load()
    assert [c.to_dict() for c in state.components] == before
    assert state.pending_decision_bar != _bar(close, 33) and state.last_refresh_bar == _bar(close, 32)
    result = _session(ctx, close, volume, 33, tranches=21)
    assert result.tranches["built"] == 4 and result.submitted_orders


# ---------------------------------------------------------------------------
# Outages
# ---------------------------------------------------------------------------


def test_an_outage_is_caught_up_on_the_return_session_and_the_tranches_go_back_to_their_grid(
    tmp_path: Path,
) -> None:
    close, volume = _walk(110)
    ctx = _ctx(tmp_path)
    for i in range(30, 60):  # fully built by bar 50
        _session(ctx, close, volume, i, tranches=21)
    down = range(60, 65)  # five sessions dark
    back = _session(ctx, close, volume, 65, tranches=21)
    missed = {_tranche_id(_grid(close, i) % EVERY) for i in down}
    assert set(back.tranches["due"]) == missed | {_tranche_id(_grid(close, 65) % EVERY)}
    assert len(back.tranches["due"]) == 6
    for i in range(66, 95):
        result = _session(ctx, close, volume, i, tranches=21)
        assert result.tranches["due"] == [_tranche_id(_grid(close, i) % EVERY)], _bar(close, i)


def test_an_outage_of_more_than_two_cycles_expires_and_rebuilds_every_tranche(tmp_path: Path) -> None:
    close, volume = _walk(110)
    ctx = _ctx(tmp_path)
    for i in range(30, 52):
        _session(ctx, close, volume, i, tranches=21)
    back = _session(ctx, close, volume, 100, tranches=21)  # 48 sessions dark: every output is past its 42
    assert len(back.tranches["due"]) == 21
    target = back.target_weights.reindex(NAMES)
    np.testing.assert_allclose(target.to_numpy(), BOOK.reindex(NAMES).to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------
# Wind-down
# ---------------------------------------------------------------------------


def test_wind_down_takes_21_sessions_and_leaves_the_account_flat(tmp_path: Path) -> None:
    close, volume = _walk(110)
    ctx = _ctx(tmp_path)
    equity_at: dict[int, float] = {}
    for i in range(30, 51):
        equity_at[i] = _session(ctx, close, volume, i, tranches=21).equity
    assert all(c.last_output_bar is not None for c in _components(ctx).values())
    provider_calls: list[str] = []

    def unrankable(bar: str) -> list[str]:
        provider_calls.append(bar)
        return ["S9"]

    for j in range(1, 22):
        i = 50 + j
        result = _session(ctx, close, volume, i, tranches=21, wind_down=True, unrankable=unrankable)
        assert result.tranches["wind_down"] is True and len(result.tranches["due"]) == 1
        flat = [c for c in _components(ctx).values() if all(v == 0.0 for v in c.sub_targets.values())]
        assert len(flat) == j and all(c.holdings == {} for c in flat)
        # The tranches built at bars 30 + j .. 50 are still held, in the shares they were built with.
        remaining = [(b, equity_at[b]) for b in range(30 + j, 51)]
        expected = _held_target(close, remaining, i) * close.iloc[i][NAMES] / result.equity
        np.testing.assert_allclose(result.target_weights.reindex(NAMES).to_numpy(), expected.to_numpy(), atol=1e-12)
    assert provider_calls == []  # a wind-down ranks nothing, so the mask is not consulted
    assert (result.target_weights.fillna(0.0) == 0.0).all()
    _session(ctx, close, volume, 72, tranches=21, wind_down=True)  # the last decision's fills settle here
    assert ctx.broker.positions() == {}


# ---------------------------------------------------------------------------
# Whole-book refreshes and withheld entries
# ---------------------------------------------------------------------------


def _seeded_state(
    close: pd.DataFrame,
    decision: int,
    extra: dict[int, dict[str, float]],
    *,
    equity: float = 1_000_000.0,
    holdings: bool = True,
) -> LoopState:
    """A built 21-tranche state whose tranche ``j`` last refreshed on its latest grid date before ``decision``.

    Each tranche holds ``BOOK`` in shares sized at ``equity`` and the close of
    its refresh bar (``holdings=False`` leaves the share holdings out, as a
    state written before tranches were held in shares). ``extra`` adds
    sub-targets the tranche carries and does not hold.
    """
    components = []
    for j in range(21):
        last = max(p for p in range(decision) if _grid(close, p) % EVERY == j)
        sub = {**BOOK.to_dict(), **extra.get(j, {})}
        held = {
            sid: {"shares": weight / 21.0 * equity / float(close.iloc[last][sid]), "bar": _bar(close, last),
                  "close": float(close.iloc[last][sid])}
            for sid, weight in BOOK.items()
            if weight != 0.0
        }
        components.append(
            ComponentState(
                component_id=_tranche_id(j),
                last_output_bar=_bar(close, last),
                decision_bar=_bar(close, last),
                sub_targets=sub,
                phase_offset_bars=j,
                holdings=held if holdings else None,
            )
        )
    return LoopState(components=components, last_refresh_bar=_bar(close, decision - 1))


def _account_of(state: LoopState) -> dict[str, float]:
    """The account that holds exactly the tranches' shares."""
    totals: dict[str, float] = {}
    for component in state.components:
        for sid, entry in (component.holdings or {}).items():
            totals[sid] = totals.get(sid, 0.0) + entry["shares"]
    return totals


def test_a_session_decides_only_the_refreshing_tranches_change(tmp_path: Path) -> None:
    """Constant scores, a built book: the orders are the refreshing tranche's change and nothing else."""
    close, volume = _walk(120)
    results, _ = replay_daily_cycles(
        lambda: ConstSignal(SCORES),
        close,
        volume,
        close,
        CONFIG,
        tmp_path / "replay",
        start_bar=_bar(close, 30),
        initial_cash=1_000_000.0,
        tranches=21,
    )
    by_bar = {r.decision_bar: r for r in results}
    position = {_bar(close, i): i for i in range(len(close))}
    checked = 0
    for r in results[21:]:
        i = position[r.decision_bar]
        before = by_bar[_bar(close, i - 21)]  # the same tranche's previous refresh
        assert r.tranches["due"] == before.tranches["due"]
        fresh = BOOK / 21.0 * r.equity / close.iloc[i]
        previous = BOOK / 21.0 * before.equity / close.iloc[i - 21]
        change = (fresh - previous)[["S0", "S1", "S8", "S9"]]
        orders = {o.symbol: o.qty for o in r.submitted_orders}
        assert set(orders) <= {"S0", "S1", "S8", "S9"}
        np.testing.assert_allclose(
            [orders.get(sid, 0.0) for sid in change.index], change.to_numpy(), rtol=1e-9, atol=1e-9
        )
        assert r.tranches["catch_up"] == [] and r.tranches["decided"] == 4
        checked += 1
    assert checked == len(results) - 21 > 50


def test_a_changing_book_orders_only_names_the_refreshing_tranche_touches(tmp_path: Path) -> None:
    close, volume = _walk(120, seed=5)
    monthly = replace(CONFIG, whole_shares=True)
    ctx = _ctx(tmp_path, ReplayBroker(1_000_000.0))
    signal = TrailingReturn()
    for i in range(30, 100):
        before = _components(ctx)
        result = _session(ctx, close, volume, i, signal=signal, config=monthly, tranches=21)
        (tid,) = result.tranches["due"]
        touched = set((before.get(tid).holdings or {}) if tid in before else {}) | set(_components(ctx)[tid].holdings)
        ordered = {o.symbol for o in result.submitted_orders}
        assert ordered <= touched | set(result.tranches["catch_up"]), _bar(close, i)
        # The undecided names are held: their sized target is the held weight.
        for sid in set(NAMES) - touched - set(result.tranches["catch_up"]):
            held = ctx.broker.positions().get(sid, 0.0) * close.iloc[i][sid] / result.equity
            assert float(result.target_weights.get(sid, 0.0)) == pytest.approx(held, abs=1e-12)


def test_a_split_is_restated_from_the_panel_and_orders_nothing_extra(tmp_path: Path) -> None:
    close, volume = _walk(80)
    decision = 50
    state = _seeded_state(close, decision, {})
    account = _account_of(state)

    plain_ctx = _ctx(tmp_path / "plain", FakeBroker(cash=1_000_000.0))
    plain_ctx.broker._positions = dict(account)
    plain_ctx.store.save(state)
    plain = _session(plain_ctx, close, volume, decision, tranches=21)

    # A 2-for-1 split of S8 effective on the decision bar: the venue doubled the account's shares, and the
    # panel fetched now shows every S8 close halved (split-adjusted to its last session).
    split = close.copy()
    split["S8"] = split["S8"] / 2.0
    split_ctx = _ctx(tmp_path / "split", FakeBroker(cash=1_000_000.0))
    split_ctx.broker._positions = {**account, "S8": 2.0 * account["S8"]}
    split_ctx.store.save(state)
    after = _session(split_ctx, split, volume, decision, tranches=21)

    plain_orders = {o.symbol: o.qty for o in plain.submitted_orders}
    split_orders = {o.symbol: o.qty for o in after.submitted_orders}
    assert set(split_orders) == set(plain_orders) <= {"S0", "S1", "S8", "S9"}
    assert split_orders["S8"] == pytest.approx(2.0 * plain_orders["S8"], rel=1e-9)
    for sid in ("S0", "S1", "S9"):
        assert split_orders[sid] == pytest.approx(plain_orders[sid], rel=1e-9)
    assert after.tranches["catch_up"] == []
    pd.testing.assert_series_equal(
        after.target_weights.reindex(NAMES), plain.target_weights.reindex(NAMES), rtol=1e-9, atol=1e-12
    )


def test_a_state_without_share_holdings_is_refused(tmp_path: Path) -> None:
    close, volume = _walk(80)
    ctx = _ctx(tmp_path, FakeBroker(cash=1_000_000.0))
    ctx.store.save(_seeded_state(close, 50, {}, holdings=False))
    before = (tmp_path / "state.json").read_bytes()
    with pytest.raises(TrancheError, match="no share holdings"):
        _session(ctx, close, volume, 50, tranches=21)
    assert (tmp_path / "state.json").read_bytes() == before
    assert ctx.broker.submitted_order_ids() == set()


def test_a_carried_sub_target_the_tranche_does_not_hold_is_never_ordered_and_its_refresh_drops_it(
    tmp_path: Path,
) -> None:
    close, volume = _walk(80)
    decision = 50
    holder = (_grid(close, decision) + 3) % EVERY  # refreshes three sessions later
    state = _seeded_state(close, decision, {holder: {"GONE": 0.25}})  # not in the panel, not held
    ctx = _ctx(tmp_path, FakeBroker(cash=1_000_000.0))
    ctx.broker._positions = _account_of(state)
    ctx.store.save(state)

    result = _session(ctx, close, volume, decision, tranches=21)
    assert "GONE" not in result.target_weights.index or float(result.target_weights["GONE"]) == 0.0
    assert all(order.symbol != "GONE" for order in result.submitted_orders)
    assert _components(ctx)[_tranche_id(holder)].sub_targets["GONE"] == 0.25  # carried until its own refresh

    for i in range(decision + 1, decision + 4):
        result = _session(ctx, close, volume, i, tranches=21)
    assert result.tranches["due"] == [_tranche_id(holder)]
    assert _components(ctx)[_tranche_id(holder)].sub_targets["GONE"] == 0.0  # its refresh is its whole book
    assert result.tranches["deferred"] == []


def test_an_entry_without_a_price_is_deferred_until_the_tranches_next_refresh(tmp_path: Path) -> None:
    close, volume = _walk(80)
    i = 40
    gapped = close.copy()
    gapped["GAP"] = 100.0
    gapped.loc[gapped.index[i], "GAP"] = np.nan  # no print on the decision bar
    volume = volume.reindex(columns=[*NAMES, "GAP"], fill_value=1e6)
    ctx = _ctx(tmp_path, FakeBroker(cash=1_000_000.0))
    scores = {**SCORES, "GAP": 99.0}  # the top name: in the long leg of every refresh
    result = _session(ctx, gapped, volume, i, signal=ConstSignal(scores), tranches=21)
    (tid,) = result.tranches["due"]
    assert result.tranches["deferred"] == ["GAP"]
    assert all(order.symbol != "GAP" for order in result.submitted_orders)
    assert "GAP" not in _components(ctx)[tid].holdings
    assert _components(ctx)[tid].sub_targets["GAP"] > 0.0
    assert "GAP" not in _rows(tmp_path / "targets.jsonl")[-1]["targets"]


# ---------------------------------------------------------------------------
# The volatility scale per tranche
# ---------------------------------------------------------------------------

RULE = VolScaleRule(target_vol=0.12, lookback_bars=20, max_scale=2.0)
SCALED = replace(CONFIG, max_gross=2.0)


def _tranched_reference(close: pd.DataFrame, rule: VolScaleRule, offsets: list[int]) -> tuple[float, float]:
    """The rule on the tranched base book, rebuilt with plain loops and nothing from the package."""
    prices = close.to_numpy(dtype=float)
    last = len(prices) - 1
    grid = [sessions.grid_position(ts.date()) for ts in close.index]
    k = int(N_SYM * 0.2)

    def book(position: int) -> np.ndarray:
        momentum = prices[position] / prices[position - 5] - 1.0
        order = np.argsort(momentum, kind="stable")
        weights = np.zeros(N_SYM)
        weights[order[:k]] = -0.5 / k
        weights[order[-k:]] = 0.5 / k
        return weights

    returns = []
    for bar in range(last - rule.lookback_bars + 1, last + 1):
        held_at = bar - 1
        books = [book(max(p for p in range(held_at + 1) if (grid[p] - o) % EVERY == 0)) for o in offsets]
        held = sum(books) / len(offsets)
        returns.append(float(np.sum(held * (prices[bar] / prices[bar - 1] - 1.0))))
    sigma = float(np.std(returns, ddof=1) * math.sqrt(252))
    return min(rule.max_scale, rule.target_vol / sigma), sigma


@pytest.mark.parametrize("target_vol", [0.03, 0.08])  # the base book's sigma is about 0.05: neither cap binds
def test_each_tranche_is_scaled_by_the_rule_on_the_tranched_base_book(tmp_path: Path, target_vol: float) -> None:
    close, volume = _walk(90)
    rule = replace(RULE, target_vol=target_vol)
    signal = TrailingReturn()
    t = required_panel_bars(signal.required_history, rule, EVERY) - 1
    assert t == 45
    ctx = _ctx(tmp_path, FakeBroker(cash=1_000_000.0))
    result = _session(ctx, close, volume, t, signal=signal, config=SCALED, tranches=21, vol_scale=rule)
    (tid,) = result.tranches["due"]
    expected_scale, expected_sigma = _tranched_reference(close.iloc[: t + 1], rule, list(range(21)))
    assert 0.0 < expected_scale < rule.max_scale

    row = result.vol_scale
    assert row["schedule"] == "tranches" and list(row["tranches"]) == [tid]
    decision = row["tranches"][tid]
    assert decision["status"] == "ok" and decision["reason"] is None and decision["last_good"] is None
    assert decision["scale"] == pytest.approx(expected_scale, rel=1e-12)
    assert row["sigma"] == pytest.approx(expected_sigma, rel=1e-12)
    assert row["n_valid"] == rule.lookback_bars

    unscaled_ctx = _ctx(tmp_path / "unscaled", FakeBroker(cash=1_000_000.0))
    unscaled = _session(unscaled_ctx, close, volume, t, signal=signal, config=SCALED, tranches=21)
    pd.testing.assert_series_equal(
        result.target_weights.reindex(NAMES), unscaled.target_weights.reindex(NAMES) * decision["scale"]
    )
    assert row["target_gross"] == pytest.approx(decision["scale"] / 21.0)
    persisted = _components(ctx)[tid].sizing
    assert persisted["scale"] == decision["scale"] and persisted["last_good"] == decision["scale"]
    assert _rows(tmp_path / SIZING_LEDGER_FILENAME) == [json.loads(json.dumps(row, sort_keys=True))]
    assert _rows(tmp_path / "targets.jsonl")[0][TARGETS_LEDGER_KEY][TARGETS_TRANCHES_KEY][tid]["status"] == "ok"


def test_a_tranche_failure_is_bounded_by_its_own_last_good_scale(tmp_path: Path) -> None:
    # Flat from bar 66: the decision at 67 sees two zero returns in twenty, more than 5%.
    close, volume = _walk(100, flat_from=66)
    signal = TrailingReturn()
    ctx = _ctx(tmp_path, FakeBroker(cash=1_000_000.0))
    decided: dict[str, float] = {}
    for i in range(45, 67):
        result = _session(ctx, close, volume, i, signal=signal, config=SCALED, tranches=21, vol_scale=RULE)
        for tid, entry in result.vol_scale["tranches"].items():
            assert entry["status"] == "ok"
            decided.setdefault(tid, entry["scale"])
    result = _session(ctx, close, volume, 67, signal=signal, config=SCALED, tranches=21, vol_scale=RULE)
    (tid,) = result.tranches["due"]
    entry = result.vol_scale["tranches"][tid]
    assert entry["status"] == "fallback" and entry["reason"] == "zero_returns"
    assert entry["last_good"] == decided[tid]  # this tranche's own, from 21 sessions ago
    assert entry["scale"] == min(1.0, decided[tid])
    assert _components(ctx)[tid].sizing["last_good"] == decided[tid]  # a fallback keeps the bound


def test_the_tranched_account_never_exceeds_the_ceiling(tmp_path: Path) -> None:
    close, volume = _walk(90)
    signal = TrailingReturn()
    loud = replace(RULE, target_vol=5.0)  # far above the book's volatility: every tranche at its cap of 2
    ctx = _ctx(tmp_path, FakeBroker(cash=1_000_000.0))
    config = replace(SCALED, max_gross=1.5)
    for i in range(45, 70):
        result = _session(ctx, close, volume, i, signal=signal, config=config, tranches=21, vol_scale=loud)
        assert float(result.target_weights.abs().sum()) <= 1.5 + 1e-12
        for entry in result.vol_scale["tranche_targets"].values():
            assert entry["ceiling_factor"] == pytest.approx(0.75)


class StatelessConst(ConstSignal):
    """Fixed scores that declare no fitted state, which the volatility scale requires."""

    output_kind = "rank_score"
    units = "rank"
    scoring_basis = "price"
    price_scale_invariant = True
    fitted_state = "none"
    estimated_decay_bars = None

    @property
    def horizon_bars(self) -> int:
        return EVERY

    @property
    def required_history(self) -> int:
        return 6


def test_a_drifted_account_above_the_ceiling_is_scaled_whole_and_every_name_decided(tmp_path: Path) -> None:
    close, volume = _walk(110, seed=2)
    loud = replace(RULE, target_vol=5.0)  # every tranche at its cap: each refresh sized at the 1.5 ceiling
    config = replace(SCALED, max_gross=1.5)
    ctx = _ctx(tmp_path, ReplayBroker(1_000_000.0))
    bound = []
    for i in range(45, 105):
        result = _session(ctx, close, volume, i, signal=StatelessConst(SCORES), config=config, tranches=21,
                          vol_scale=loud)
        recorded = _ledger_target(ctx, _bar(close, i))
        assert float(recorded.abs().sum()) <= 1.5 * (1.0 + 1e-9)
        if result.tranches["ceiling_factor"] < 1.0:
            bound.append(i)
            assert result.tranches["decided"] == N_SYM  # every name, the drift included
            assert float(recorded.abs().sum()) == pytest.approx(1.5, rel=1e-9)
            # The persisted holdings were scaled with the target: at this close they make the capped book.
            totals = pd.Series(0.0, index=NAMES)
            for component in _components(ctx).values():
                for sid, entry in component.holdings.items():
                    totals[sid] += entry["shares"]
            held_target = totals * close.iloc[i][NAMES] / result.equity
            np.testing.assert_allclose(held_target.to_numpy(), recorded.to_numpy(), atol=1e-12)
    assert bound, "the drifted book never reached the ceiling, so this test exercised nothing"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case, match",
    [
        ("directional_book", "decile_neutral"),
        ("count_not_dividing", "must divide decision_every"),
        ("regime_hook", "regime gross-scale hook"),
        ("wind_down_single", "needs a tranched book"),
        ("layout_differs", "tranche layout"),
        ("untranched_book", "untranched book"),
        ("held_account", "flat account"),
        ("single_book_on_tranches", "holds a tranched book"),
        ("not_a_session", "no session-grid position"),
    ],
)
def test_inputs_the_tranched_book_cannot_honour_raise_before_any_state_moves(tmp_path: Path, case, match) -> None:
    close, volume = _walk(60)
    broker = FakeBroker(cash=1_000_000.0)
    ctx = _ctx(tmp_path, broker)
    config, kwargs = CONFIG, {"tranches": 21}
    if case == "directional_book":
        config = DailyBookConfig(position_size=0.1, decision_every=EVERY)
    elif case == "count_not_dividing":
        kwargs = {"tranches": 4}
    elif case == "regime_hook":
        kwargs.update(regime=RecordingRegime(), regime_gross_scale=lambda state: 1.0)
    elif case == "wind_down_single":
        kwargs = {"tranches": 1, "wind_down": True}
    elif case == "layout_differs":
        ctx.store.save(_seeded_state(close, 40, {}))
        kwargs = {"tranches": 21, "phase_offset": 5}
    elif case == "untranched_book":
        ctx.store.save(LoopState(last_refresh_bar=_bar(close, 30)))
    elif case == "held_account":
        broker._positions = {"S3": 10.0}
    elif case == "single_book_on_tranches":
        ctx.store.save(_seeded_state(close, 40, {}))
        kwargs = {}
    before = None if not (tmp_path / "state.json").exists() else (tmp_path / "state.json").read_bytes()
    panel = close.iloc[:41], volume.iloc[:41]
    if case == "not_a_session":
        index = close.index[:40].append(pd.DatetimeIndex([pd.Timestamp("2025-03-01", tz="America/New_York")]))
        panel = close.iloc[:41].set_axis(index), volume.iloc[:41].set_axis(index)
    with pytest.raises((TrancheError, ValueError), match=match):
        run_daily_cycle(ctx, ConstSignal(SCORES), *panel, config, **kwargs)
    after = None if not (tmp_path / "state.json").exists() else (tmp_path / "state.json").read_bytes()
    assert after == before
    assert not (tmp_path / "panels").exists()
    assert broker.submitted_order_ids() == set()


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


def test_the_flags_default_to_the_single_book() -> None:
    args = paper_loop._parse_args([])
    assert args.tranches == 1 and args.wind_down is False


def test_the_default_passes_the_single_book(wired: VenueDouble, tmp_path: Path, cycles) -> None:
    run_dir = tmp_path / "runs" / "plain"
    _point_switch(run_dir)
    assert paper_loop.main(_loop_argv(run_dir, "--submit", "--allow-new-run-dir", "--decision-every", "21")) == 0
    assert cycles[-1]["tranches"] == 1 and cycles[-1]["wind_down"] is False
    assert json.loads((run_dir / "state.json").read_text())["components"] == []


def test_a_tranched_session_builds_its_first_tranche(wired: VenueDouble, tmp_path: Path, cycles) -> None:
    run_dir = tmp_path / "runs" / "tranched"
    _point_switch(run_dir)
    argv = _loop_argv(run_dir, "--submit", "--allow-new-run-dir", "--decision-every", "21", "--tranches", "21")
    assert paper_loop.main(argv) == 0
    assert cycles[-1]["tranches"] == 21
    components = json.loads((run_dir / "state.json").read_text())["components"]
    assert len(components) == 21
    assert sum(1 for c in components if c["last_output_bar"] is not None) == 1
    assert wired.submitted
    assert _rows(run_dir / "targets.jsonl")[0][TARGETS_TRANCHES_KEY]["count"] == 21


@pytest.mark.parametrize(
    "extra, match",
    [
        (("--tranches", "21", "--decision-every", "21", "--profile", "research_paper"), "research_paper"),
        (("--wind-down", "--decision-every", "21"), "needs --tranches"),
        (("--tranches", "3", "--decision-every", "20"), "cadence it divides"),
    ],
)
def test_the_cli_refuses_before_any_venue_read(wired: VenueDouble, tmp_path: Path, cycles, extra, match) -> None:
    run_dir = _established(tmp_path)
    _point_switch(run_dir)
    with pytest.raises(SystemExit, match=match):
        paper_loop.main(_loop_argv(run_dir, "--submit", *extra))
    assert wired.reads == [] and wired.submitted == [] and cycles == []


def test_resolve_tranches() -> None:
    assert resolve_tranches(1, book="trend", decision_every=21, wind_down=False) == 1
    assert resolve_tranches(21, book="momentum", decision_every=21, wind_down=True) == 21
    with pytest.raises(SystemExit, match="--book momentum only"):
        resolve_tranches(21, book="trend", decision_every=21, wind_down=False)
    with pytest.raises(SystemExit, match="one of"):
        resolve_tranches(7, book="momentum", decision_every=21, wind_down=False)
