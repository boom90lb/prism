"""Incremental bar store + delta fetch (SPEC §7.0).

Store mechanics are tested pure (no network); the DataLoader orchestration
is tested against a recorded fake of fetch_historical_data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from prism.data_loader import BAR_TZ, DataLoader
from prism.io import IncrementalBarStore, SplitRewriteDetected


def _bars(start: str, periods: int, base: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="B", tz=BAR_TZ)
    close = base + np.arange(periods, dtype=float)
    return pd.DataFrame(
        {"open": close - 0.5, "close": close, "volume": 1e6}, index=idx
    )


# ---------------------------------------------------------------------------
# Store mechanics (pure)
# ---------------------------------------------------------------------------


def test_replace_read_roundtrip(tmp_path) -> None:
    store = IncrementalBarStore(tmp_path)
    bars = _bars("2024-01-01", 10)
    store.replace("AAPL", "1d", bars)
    out = store.read("AAPL", "1d")
    pd.testing.assert_frame_equal(out, bars, check_freq=False)
    assert store.last_timestamp("AAPL", "1d") == bars.index[-1]
    assert store.read("MSFT", "1d").empty
    assert store.last_timestamp("MSFT", "1d") is None


def test_store_rejects_naive_and_duplicate_indexes(tmp_path) -> None:
    store = IncrementalBarStore(tmp_path)
    naive = _bars("2024-01-01", 5).tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        store.replace("AAPL", "1d", naive)
    dup = pd.concat([_bars("2024-01-01", 3), _bars("2024-01-01", 3)])
    with pytest.raises(ValueError, match="duplicate"):
        store.replace("AAPL", "1d", dup)


def test_append_tail_merges_with_agreeing_overlap(tmp_path) -> None:
    store = IncrementalBarStore(tmp_path)
    full = _bars("2024-01-01", 15)
    store.replace("AAPL", "1d", full.iloc[:10])
    # Tail re-requests 3 overlap bars (identical) + 5 new ones.
    merged = store.append_tail("AAPL", "1d", full.iloc[7:], rewrite_rtol=1e-6)
    pd.testing.assert_frame_equal(merged, full, check_freq=False)
    pd.testing.assert_frame_equal(store.read("AAPL", "1d"), full, check_freq=False)


def test_append_tail_detects_split_rewrite_and_writes_nothing(tmp_path) -> None:
    store = IncrementalBarStore(tmp_path)
    full = _bars("2024-01-01", 15)
    store.replace("AAPL", "1d", full.iloc[:10])
    rewritten = full / 4.0  # a 4:1 split back-rewrites all history
    with pytest.raises(SplitRewriteDetected, match="back-rewritten"):
        store.append_tail("AAPL", "1d", rewritten.iloc[7:])
    # The store still holds the original series untouched.
    pd.testing.assert_frame_equal(store.read("AAPL", "1d"), full.iloc[:10], check_freq=False)


def test_append_tail_edge_cases(tmp_path, caplog) -> None:
    store = IncrementalBarStore(tmp_path)
    full = _bars("2024-01-01", 12)
    # Empty store: append == replace.
    merged = store.append_tail("AAPL", "1d", full.iloc[:6])
    pd.testing.assert_frame_equal(merged, full.iloc[:6], check_freq=False)
    # Empty tail: no-op.
    merged = store.append_tail("AAPL", "1d", full.iloc[:0])
    pd.testing.assert_frame_equal(merged, full.iloc[:6], check_freq=False)
    # Gap with no overlap appends but warns (a rewrite would go undetected).
    with caplog.at_level(logging.WARNING):
        merged = store.append_tail("AAPL", "1d", full.iloc[8:])
    assert "no overlap" in caplog.text
    assert len(merged) == 10  # 6 + 4, gap preserved


# ---------------------------------------------------------------------------
# DataLoader.fetch_incremental orchestration
# ---------------------------------------------------------------------------


class _FakeFetcher:
    """Stands in for fetch_historical_data; records (start, end, force)."""

    def __init__(self, universe: pd.DataFrame) -> None:
        self.universe = universe
        self.calls: list[tuple] = []

    def __call__(self, symbol, interval="1d", start_date=None, end_date=None, *, force_refresh=False):
        self.calls.append((start_date, end_date, force_refresh))
        df = self.universe
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date, tz=BAR_TZ)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date, tz=BAR_TZ)]
        return df.copy()


def _loader(tmp_path) -> DataLoader:
    return DataLoader(api_key="test", cache_dir=tmp_path)


def test_incremental_seeds_then_delta_fetches_only_the_tail(tmp_path, monkeypatch) -> None:
    universe = _bars("2024-01-01", 30)
    loader = _loader(tmp_path)
    fake = _FakeFetcher(universe.iloc[:20])
    monkeypatch.setattr(loader, "fetch_historical_data", fake)

    end0 = universe.index[19].strftime("%Y-%m-%d")
    out = loader.fetch_incremental("AAPL", "1d", end_date=end0)
    assert len(out) == 20
    assert fake.calls == [(None, end0, False)]  # one full seed fetch

    # Ten more bars appear at the vendor; only the tail is requested.
    fake.universe = universe
    end1 = universe.index[-1].strftime("%Y-%m-%d")
    out = loader.fetch_incremental("AAPL", "1d", end_date=end1)
    assert len(out) == 30
    assert len(fake.calls) == 2
    delta_start, delta_end, forced = fake.calls[1]
    assert delta_end == end1 and not forced
    # Delta starts at the overlap window, not at history's beginning.
    assert pd.Timestamp(delta_start, tz=BAR_TZ) == universe.index[20 - 5]

    # Fully covered request: no new fetch at all.
    out = loader.fetch_incremental("AAPL", "1d", end_date=end1)
    assert len(fake.calls) == 2
    assert len(out) == 30


def test_incremental_split_rewrite_triggers_forced_full_refetch(tmp_path, monkeypatch) -> None:
    universe = _bars("2024-01-01", 30)
    loader = _loader(tmp_path)
    fake = _FakeFetcher(universe.iloc[:20])
    monkeypatch.setattr(loader, "fetch_historical_data", fake)

    end0 = universe.index[19].strftime("%Y-%m-%d")
    loader.fetch_incremental("AAPL", "1d", end_date=end0)

    # A 4:1 split lands: the vendor rewrites all history.
    fake.universe = universe / 4.0
    end1 = universe.index[-1].strftime("%Y-%m-%d")
    out = loader.fetch_incremental("AAPL", "1d", end_date=end1)

    # delta fetch -> divergence -> full force_refresh fetch.
    assert len(fake.calls) == 3
    assert fake.calls[2] == (None, end1, True)
    assert len(out) == 30
    stored = IncrementalBarStore(tmp_path / "store").read("AAPL", "1d")
    np.testing.assert_allclose(stored["close"].to_numpy(), (universe / 4.0)["close"].to_numpy())


def test_incremental_seeds_from_legacy_cache_without_network(tmp_path, monkeypatch) -> None:
    universe = _bars("2024-01-01", 25)
    end_cached = universe.index[19].strftime("%Y-%m-%d")
    # A legacy range-keyed cache file already on disk covers the first 20 bars.
    universe.iloc[:20].to_parquet(tmp_path / f"AAPL_1d_2024-01-01_{end_cached}.parquet")

    loader = _loader(tmp_path)
    fake = _FakeFetcher(universe)
    monkeypatch.setattr(loader, "fetch_historical_data", fake)

    out = loader.fetch_incremental("AAPL", "1d", end_date=universe.index[-1].strftime("%Y-%m-%d"))
    assert len(out) == 25
    # No full-history fetch happened: the single call is the overlap delta.
    assert len(fake.calls) == 1
    assert fake.calls[0][0] is not None


def test_incremental_empty_delta_serves_stored_with_warning(tmp_path, monkeypatch, caplog) -> None:
    universe = _bars("2024-01-01", 20)
    loader = _loader(tmp_path)
    fake = _FakeFetcher(universe)
    monkeypatch.setattr(loader, "fetch_historical_data", fake)

    end0 = universe.index[-1].strftime("%Y-%m-%d")
    loader.fetch_incremental("AAPL", "1d", end_date=end0)

    fake.universe = universe.iloc[:0]  # vendor suddenly returns nothing
    with caplog.at_level(logging.WARNING):
        out = loader.fetch_incremental("AAPL", "1d", end_date="2024-06-01")
    assert len(out) == 20  # stored series served
    assert "possibly stale" in caplog.text


def test_incremental_slices_to_requested_range(tmp_path, monkeypatch) -> None:
    universe = _bars("2024-01-01", 20)
    loader = _loader(tmp_path)
    monkeypatch.setattr(loader, "fetch_historical_data", _FakeFetcher(universe))

    start = universe.index[5].strftime("%Y-%m-%d")
    end = universe.index[9].strftime("%Y-%m-%d")
    out = loader.fetch_incremental("AAPL", "1d", start_date=start, end_date=end)
    assert len(out) == 5
    assert out.index[0] == universe.index[5]
    assert out.index[-1] == universe.index[9]
