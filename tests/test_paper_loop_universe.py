"""The fetch universe must cover the BROKER's book; extras are valuation/exit-only.

Pins two failure classes, one nested inside the other.

The 2026-07-15 class: a regenerated ``sp500_current.txt`` correctly dropped
POOL (index removal 2026-06-22) while the live book still held it, and the mark
step refused to value the position (N7). The fix fetched universe ∪ held,
masked the extras out of scoring, and leaned on the decile construct's
explicit-flat pin to exit them at the next refresh.

The 2026-07-23 class: that fix read "held" from the *persisted state*, so it was
empty over a fresh run directory and stale (28 positions against 98 at the
venue) over an old one. Every scheduled cycle from 2026-07-23 through 07-29 died
on ``cannot value held position 'POOL': price None (N7)`` while the morning
sweep exited 0 and prism-doctor reported "8 pass, 0 fail". Broker truth is now
the authority (``resolve_fetch_universe``), a held name may not be dropped by
the max_missing tolerance (``fetch_universe_panels(required=...)``), and the
end-to-end case — fresh run directory over a pre-existing venue book — runs a
full cycle here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.live import (
    DailyBookConfig,
    LiveLoopContext,
    StateStore,
    fetch_universe_panels,
    resolve_fetch_universe,
    run_daily_cycle,
)
from prism.live.state import LoopState
from prism.portfolio.construct import construct_decile_neutral
from prism.residual.factors import ResidualStatArbConfig
from prism.signal.momentum_node import MomentumSignalNode
from tests.test_live_daily import ConstSignal
from tests.test_live_loop import FakeBroker


def _broker(positions: dict[str, float], cash: float = 100_000.0) -> FakeBroker:
    broker = FakeBroker(cash=cash)
    broker._positions = dict(positions)
    return broker


class DictLoader:
    """A ``fetch_batch`` source over in-memory frames; unknown names come back empty."""

    def __init__(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> None:
        self._close = close
        self._volume = volume

    def fetch_batch(self, symbols, interval="1d", start_date=None, end_date=None, **kwargs):
        frames = {}
        for symbol in symbols:
            if symbol not in self._close.columns:
                frames[symbol] = pd.DataFrame()
                continue
            frame = pd.DataFrame({"close": self._close[symbol]})
            if self._volume is not None:
                frame["volume"] = self._volume[symbol]
            frames[symbol] = frame
        return frames


# ---------------------------------------------------------------------------
# Broker truth is the authority (the 2026-07-23 class)
# ---------------------------------------------------------------------------


def test_broker_truth_survives_a_fresh_run_directory() -> None:
    # The specimen: no persisted state at all, a live book at the venue.
    universe = resolve_fetch_universe(["AAA", "BBB"], _broker({"POOL": -5.0, "AAA": 3.0}), state=None)
    assert universe.score == ("AAA", "BBB")
    assert universe.valuation_only == ("POOL",)
    assert universe.fetch == ("AAA", "BBB", "POOL")
    assert universe.held == {"POOL": -5.0, "AAA": 3.0}


def test_persisted_state_is_not_the_authority_for_holdings() -> None:
    # A stale state that has forgotten the venue's book: POOL must still be
    # fetched, and it must be *required* to price (it is genuinely held).
    universe = resolve_fetch_universe(
        ["AAA"], _broker({"POOL": -5.0}), state=LoopState(positions={"AAA": 1.0})
    )
    assert universe.valuation_only == ("POOL",)
    assert universe.held == {"POOL": -5.0}


def test_state_only_name_rides_along_but_is_never_required() -> None:
    # The venue closed GONE; the state has not caught up. Fetching it is free
    # insurance for the reconcile path, but a delisted name with no bars must
    # not be able to fail the fetch — only broker-held names are `required`.
    universe = resolve_fetch_universe(
        ["AAA"], _broker({"POOL": -5.0}), state=LoopState(positions={"GONE": 7.0})
    )
    assert universe.valuation_only == ("GONE", "POOL")
    assert set(universe.held) == {"POOL"}


def test_zero_share_positions_are_not_held() -> None:
    universe = resolve_fetch_universe(["AAA"], _broker({"AAA": 0.0, "POOL": 0.0}))
    assert universe.valuation_only == ()
    assert universe.held == {}


def test_configured_universe_is_deduped_and_upper_cased() -> None:
    universe = resolve_fetch_universe([" aaa ", "AAA", "bbb"], _broker({}))
    assert universe.score == ("AAA", "BBB")
    assert universe.fetch == ("AAA", "BBB")


def test_empty_configured_universe_raises() -> None:
    with pytest.raises(ValueError, match="empty configured universe"):
        resolve_fetch_universe([], _broker({"AAA": 1.0}))


def test_unattributed_book_is_loud(caplog) -> None:
    with caplog.at_level("WARNING"):
        resolve_fetch_universe(["AAA"], _broker({"AAA": 1.0}), state=None)
    assert "NO persisted book" in caplog.text


# ---------------------------------------------------------------------------
# A held name may not be dropped by the max_missing tolerance
# ---------------------------------------------------------------------------


def _panel(names: list[str], n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2026-05-01", periods=n, freq="B", tz="America/New_York")
    close = pd.DataFrame({s: [100.0 + i] * n for i, s in enumerate(names)}, index=idx)
    return close, pd.DataFrame(1e6, index=idx, columns=close.columns)


def test_fetch_refuses_to_drop_a_held_name_under_tolerance() -> None:
    close, volume = _panel(["AAA", "BBB"])
    loader = DictLoader(close, volume)
    # POOL has no bars. Under a 50% tolerance a plain fetch drops it happily —
    # and the mark step then dies on it. Declared `required`, it raises here,
    # naming the position and the remedy.
    with pytest.raises(RuntimeError, match=r"no bars for held position\(s\) \['POOL'\]"):
        fetch_universe_panels(loader, ["AAA", "BBB", "POOL"], max_missing=0.5, required=["POOL"])


def test_fetch_still_tolerates_a_dead_non_held_ticker() -> None:
    close, volume = _panel(["AAA", "BBB"])
    got, _ = fetch_universe_panels(
        DictLoader(close, volume), ["AAA", "BBB", "DEAD"], max_missing=0.5, required=["AAA"]
    )
    assert list(got.columns) == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# End to end: fresh run directory over a pre-existing venue book
# ---------------------------------------------------------------------------


def test_fresh_run_dir_over_a_live_book_completes_and_exits_the_leaver(tmp_path) -> None:
    """The regression: an empty run directory, a venue holding an index leaver.

    Before the fix this raised ``cannot value held position 'POOL'`` at the mark
    step — six consecutive nightly sessions. After it, the reconciliation puts
    POOL in the fetch universe (so it can be marked) and out of the scoring
    universe (so it can only leave), and the cycle submits its exit.
    """
    names = [f"S{i:02d}" for i in range(10)]
    close, volume = _panel(names + ["POOL"])
    configured = list(names)  # POOL has left the index — it is NOT in the file
    broker = _broker({"POOL": -5.0, "S00": 10.0})
    run_dir = tmp_path / "run"  # deliberately fresh: no state.json, no ledgers
    store = StateStore(run_dir / "state.json")
    assert store.load() is None

    universe = resolve_fetch_universe(configured, broker, state=store.load())
    assert "POOL" in universe.fetch and "POOL" in universe.held
    assert "POOL" not in universe.score

    panel, vol = fetch_universe_panels(
        DictLoader(close, volume), list(universe.fetch), max_missing=0.1, required=sorted(universe.held)
    )
    assert "POOL" in panel.columns  # the whole point: the leaver can be valued

    # Scoring is confined to `universe.score`, which is what the momentum shell's
    # membership mask achieves: a valuation-only name scores NaN and the decile
    # construct pins NaN to an explicit 0.0 — an exit, never a hold.
    signal = ConstSignal({s: float(i) for i, s in enumerate(universe.score)})

    ctx = LiveLoopContext(
        store=store,
        broker=broker,
        fills_ledger=run_dir / "fills.jsonl",
        equity_ledger=run_dir / "equity.jsonl",
        targets_ledger=run_dir / "targets.jsonl",
        order_id_prefix="mom:",
    )
    result = run_daily_cycle(
        ctx,
        signal,
        panel,
        vol,
        DailyBookConfig(book="decile_neutral", decile=0.2, min_order_notional=1.0),
    )

    assert result.halted is None
    assert result.equity > 0
    # POOL is targeted flat and its exit order is submitted (a -5 share short
    # closes with a +5 buy), while the scored book still trades.
    assert result.target_weights["POOL"] == pytest.approx(0.0)
    pool_orders = [o for o in result.submitted_orders if o.symbol == "POOL"]
    assert len(pool_orders) == 1 and pool_orders[0].qty == pytest.approx(5.0)
    assert any(o.symbol != "POOL" for o in result.submitted_orders)
    # And the run directory now has the NAV row whose absence is what
    # prism-doctor's equity-ledger check reports as a dark loop.
    assert (run_dir / "equity.jsonl").exists()


def test_a_universe_without_the_held_leaver_still_dies_at_the_mark(tmp_path) -> None:
    """The counterfactual that gives the test above its teeth.

    This is literally what the old state-derived universe produced over a fresh
    run directory — the verbatim nightly traceback of 2026-07-23..29. Keeping it
    asserted means the mark step's refusal stays a real refusal (N7): the fix is
    "fetch what the venue holds", never "let an unpriceable position slide".
    """
    names = [f"S{i:02d}" for i in range(10)]
    close, volume = _panel(names)  # POOL absent from the panel, held at the venue
    ctx = LiveLoopContext(
        store=StateStore(tmp_path / "state.json"),
        broker=_broker({"POOL": -5.0, "S00": 10.0}),
        fills_ledger=tmp_path / "fills.jsonl",
    )
    with pytest.raises(ValueError, match=r"cannot value held position 'POOL'"):
        run_daily_cycle(
            ctx,
            ConstSignal({s: float(i) for i, s in enumerate(names)}),
            close,
            volume,
            DailyBookConfig(book="decile_neutral", decile=0.2),
        )


def test_masked_extra_scores_nan_and_gets_explicit_flat_target() -> None:
    idx = pd.date_range("2024-01-01", periods=300, freq="B", tz="America/New_York")
    names = [f"S{i:02d}" for i in range(12)] + ["POOL"]
    drift = np.linspace(-0.002, 0.002, len(names))
    steps = np.arange(len(idx))[:, None]
    close = pd.DataFrame(100.0 * (1.0 + drift[None, :]) ** steps, index=idx, columns=names)
    volume = pd.DataFrame(1_000_000.0, index=idx, columns=names)
    mask = pd.DataFrame(False, index=idx, columns=names)
    mask.loc[:, names[:-1]] = True  # POOL is valuation-only: masked ineligible

    node = MomentumSignalNode(
        ResidualStatArbConfig(), lookback_bars=252, skip_bars=21, horizon_bars=21, membership_mask=mask
    )
    scores = node.score(close, volume)
    assert np.isnan(scores["POOL"])
    assert scores.drop("POOL").notna().all()

    row = construct_decile_neutral(pd.DataFrame([scores]), decile=0.1).iloc[0]
    # The explicit-flat pin: the masked name gets weight 0.0 (an exit order for
    # a held position), never NaN (which would hold it forever).
    assert row["POOL"] == 0.0
    assert (row != 0.0).any()
