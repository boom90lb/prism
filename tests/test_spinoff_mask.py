"""Spin-off eligibility mask (docs/bar_vendor_divergence.md §5).

Pins the owner-approved remediation for the measured bar-vendor divergence:
a name with a spin-off inside the momentum lookback is unrankable — masked
out of the scored cross-section, no new position, held-until-clear — with the
mechanic OFF by default and bit-identical to the unmasked loop, detection
cached per decision bar in the run dir, causal (only events known at *t*),
and every detection failure loud and unmasked (the mask is a protection, not
a correctness precondition).
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

from prism.live import (
    DailyBookConfig,
    LiveLoopContext,
    StateStore,
    fetch_spinoffs,
    run_daily_cycle,
    spinoff_flags,
    spinoff_unrankable,
)
from prism.scripts.paper_loop import _parse_args, main
from tests.test_live_daily import ConstSignal, _panels
from tests.test_live_loop import FakeBroker

_DECILE = DailyBookConfig(
    book="decile_neutral", decile=0.2, max_symbol_abs_weight=1.0, min_order_notional=1.0
)


def _ctx(tmp_path, name: str = "run") -> LiveLoopContext:
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    return LiveLoopContext(
        store=StateStore(d / "state.json"),
        broker=FakeBroker(),
        fills_ledger=d / "fills.jsonl",
    )


def _decile_panels(n: int = 30, n_sym: int = 10):
    return _panels(n=n, prices={f"S{i}": 100.0 for i in range(n_sym)})


def _ranked(n_sym: int = 10) -> ConstSignal:
    return ConstSignal({f"S{i}": float(i) for i in range(n_sym)})


# ---------------------------------------------------------------------------
# Parity: the mechanic lands default-off and changes nothing
# ---------------------------------------------------------------------------


def test_flag_off_is_bit_identical_to_empty_mask(tmp_path) -> None:
    # No provider (the default) and a provider flagging nothing produce the
    # same targets and orders bit-for-bit on the same panel.
    calls: list[str] = []

    def empty_provider(bar: str) -> list[str]:
        calls.append(bar)
        return []

    close, vol = _decile_panels()
    base = run_daily_cycle(_ctx(tmp_path, "off"), _ranked(), close, vol, _DECILE)
    empty = run_daily_cycle(
        _ctx(tmp_path, "empty"), _ranked(), close, vol, _DECILE, unrankable=empty_provider
    )
    pd.testing.assert_series_equal(base.target_weights, empty.target_weights, check_exact=True)
    assert base.submitted_orders == empty.submitted_orders
    assert base.masked is None and empty.masked == []
    assert calls == [base.decision_bar]  # consulted exactly once, on the refresh


# ---------------------------------------------------------------------------
# Mask semantics: no entry on a divergent rank; held names held, never flattened
# ---------------------------------------------------------------------------


def test_flagged_unheld_name_cannot_enter(tmp_path) -> None:
    close, vol = _decile_panels()
    result = run_daily_cycle(
        _ctx(tmp_path), _ranked(), close, vol, _DECILE, unrankable=lambda bar: ["S9"]
    )
    assert result.masked == ["S9"]
    assert "S9" not in {o.symbol for o in result.submitted_orders}  # no entry
    assert result.target_weights["S9"] == 0.0
    # Decile membership recomputes over the 9 rankable names: 1 per leg.
    longs = {o.symbol for o in result.submitted_orders if o.qty > 0}
    shorts = {o.symbol for o in result.submitted_orders if o.qty < 0}
    assert longs == {"S8"} and shorts == {"S0"}


def test_flagged_held_name_is_held_not_liquidated(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    run_daily_cycle(ctx, _ranked(), *_decile_panels(n=26), _DECILE)  # S8,S9 long; S0,S1 short
    # Next bar: S9's fresh score would drop it from the decile (the exit
    # scenario test_decile_book_exits_a_name_that_leaves_the_decile pins) AND
    # a spin-off flags it: the mask must hold the position, not act on the
    # divergent rank.
    dropped = {f"S{i}": float(i) for i in range(10)}
    dropped["S9"] = 4.5  # mid-pack: unmasked, the book would exit
    r2 = run_daily_cycle(
        ctx, ConstSignal(dropped), *_decile_panels(n=27), _DECILE, unrankable=lambda bar: ["S9"]
    )
    assert r2.masked == ["S9"]
    assert "S9" not in {o.symbol for o in r2.submitted_orders}
    assert r2.target_weights["S9"] == pytest.approx(0.25)  # the held weight, untouched
    # Window cleared (mask empty): the name ranks normally again and the book
    # exits on the now-divergence-free rank.
    r3 = run_daily_cycle(
        ctx, ConstSignal(dropped), *_decile_panels(n=28), _DECILE, unrankable=lambda bar: []
    )
    closed = {o.symbol: o.qty for o in r3.submitted_orders}
    assert "S9" in closed and closed["S9"] < 0  # sold to flat
    assert r3.target_weights["S9"] == pytest.approx(0.0)


def test_provider_not_consulted_off_refresh_sessions(tmp_path) -> None:
    # Off a refresh session nothing is scored, so nothing is fetched — the
    # detection budget is one call per refresh, and masking stays causal to
    # the bars that were actually decided on.
    ctx = _ctx(tmp_path)
    config = DailyBookConfig(
        book="decile_neutral",
        decile=0.2,
        decision_every=5,
        max_symbol_abs_weight=1.0,
        min_order_notional=1.0,
    )
    calls: list[str] = []

    def provider(bar: str) -> list[str]:
        calls.append(bar)
        return []

    r1 = run_daily_cycle(ctx, _ranked(), *_decile_panels(n=26), config, unrankable=provider)
    assert calls == [r1.decision_bar]
    r2 = run_daily_cycle(ctx, _ranked(), *_decile_panels(n=27), config, unrankable=provider)
    assert calls == [r1.decision_bar]  # hold session: no scoring, no detection
    assert r2.masked is None


# ---------------------------------------------------------------------------
# Detection: fetch shape, causal window, cache, loud failure
# ---------------------------------------------------------------------------

_SPIN = {
    "source_symbol": "FTV",
    "new_symbol": "RAL",
    "source_rate": 1.0,
    "new_rate": 0.25,
    "ex_date": "2026-05-05",
    "process_date": "2026-05-06",
}


class _Response:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class FakeSession:
    """Canned corporate-actions pages; records every request's params."""

    def __init__(self, pages: list[dict]) -> None:
        self.pages = list(pages)
        self.requests: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        # Credentials travel in headers, never the URL (the AlpacaBroker rule).
        assert "APCA-API-KEY-ID" in headers and "key" not in url
        self.requests.append(dict(params))
        return _Response(self.pages[min(len(self.requests) - 1, len(self.pages) - 1)])


class ExplodingSession:
    def get(self, *args, **kwargs):
        raise ConnectionError("venue down")


def test_fetch_paginates_and_pins_the_api_shape() -> None:
    pages = [
        {"corporate_actions": {"spin_offs": [_SPIN]}, "next_page_token": "tok"},
        {
            "corporate_actions": {"spin_offs": [{"source_symbol": "WDC", "ex_date": "2026-02-21"}]},
            "next_page_token": None,
        },
    ]
    session = FakeSession(pages)
    records = fetch_spinoffs(
        ["FTV", "WDC"], "2025-07-18", "2026-07-18", key_id="k", secret_key="s", session=session
    )
    assert [r.get("source_symbol") for r in records] == ["FTV", "WDC"]
    assert len(session.requests) == 2
    assert session.requests[0]["types"] == "spin_off"
    assert session.requests[0]["start"] == "2025-07-18"
    assert session.requests[0]["end"] == "2026-07-18"
    assert "page_token" not in session.requests[0]
    assert session.requests[1]["page_token"] == "tok"


def test_causality_and_window_bounds() -> None:
    records = [
        {"source_symbol": "FUT", "ex_date": "2026-07-30"},  # after t: unknown at decision time
        {"source_symbol": "OLD", "ex_date": "2025-07-18"},  # on window start: both endpoints postdate it
        {"source_symbol": "INW", "ex_date": "2026-05-05"},  # inside the lookback: flagged
        {"symbol": "ALT", "ex_date": "2026-06-01"},  # tolerated alternate payload shape
        {"source_symbol": "XXX", "ex_date": "2026-05-05"},  # not in the checked universe
    ]
    flagged = spinoff_flags(records, ["FUT", "OLD", "INW", "ALT"], "2025-07-18", "2026-07-18")
    assert sorted(flagged) == ["ALT", "INW"]


def test_cache_round_trip_skips_refetch(tmp_path) -> None:
    session = FakeSession([{"corporate_actions": {"spin_offs": [_SPIN]}, "next_page_token": None}])
    kwargs = dict(key_id="k", secret_key="s", session=session)
    first = spinoff_unrankable(tmp_path, "2026-07-18", ["AAA", "FTV"], "2025-07-18", **kwargs)
    assert first == ["FTV"]
    assert len(session.requests) == 1
    cache = json.loads((tmp_path / "spinoff_mask_2026-07-18.json").read_text(encoding="utf-8"))
    assert cache["symbols_checked"] == ["AAA", "FTV"]
    assert cache["flagged"]["FTV"][0]["ex_date"] == "2026-05-05"  # the M6-consultable record
    # Same-bar rerun (any symbol order): served from the cache, no refetch.
    second = spinoff_unrankable(tmp_path, "2026-07-18", ["FTV", "AAA"], "2025-07-18", **kwargs)
    assert second == ["FTV"] and len(session.requests) == 1
    # A different window (fresh panel vintage) is a different question: refetch.
    third = spinoff_unrankable(tmp_path, "2026-07-18", ["AAA", "FTV"], "2025-07-20", **kwargs)
    assert third == ["FTV"] and len(session.requests) == 2


def test_fetch_failure_is_loud_and_unmasked(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="prism.live.spinoff_mask"):
        out = spinoff_unrankable(
            tmp_path,
            "2026-07-18",
            ["AAA", "FTV"],
            "2025-07-18",
            key_id="k",
            secret_key="s",
            session=ExplodingSession(),
        )
    assert out == []  # unmasked proceed
    assert "UNMASKED" in caplog.text
    assert "AAA" in caplog.text and "FTV" in caplog.text  # every unchecked symbol named
    assert not (tmp_path / "spinoff_mask_2026-07-18.json").exists()  # failures never cached


def test_cycle_proceeds_unmasked_on_detection_failure(tmp_path, caplog) -> None:
    ctx = _ctx(tmp_path)

    def provider(bar: str) -> list[str]:
        return spinoff_unrankable(
            tmp_path / "run",
            bar,
            [f"S{i}" for i in range(10)],
            "2025-07-18",
            key_id="k",
            secret_key="s",
            session=ExplodingSession(),
        )

    with caplog.at_level(logging.WARNING, logger="prism.live.spinoff_mask"):
        result = run_daily_cycle(ctx, _ranked(), *_decile_panels(), _DECILE, unrankable=provider)
    assert "UNMASKED" in caplog.text
    assert result.masked == []
    longs = {o.symbol for o in result.submitted_orders if o.qty > 0}
    assert longs == {"S8", "S9"}  # the full unmasked book decided and submitted


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def test_cli_flag_default_off() -> None:
    assert _parse_args(["--symbols", "AAA"]).spinoff_mask is False
    assert _parse_args(["--symbols", "AAA", "--spinoff-mask"]).spinoff_mask is True


def test_cli_flag_requires_momentum_book() -> None:
    with pytest.raises(SystemExit, match="momentum"):
        main(["--symbols", "AAA", "--spinoff-mask"])  # ensemble book: refused pre-network
