"""Token-bucket rate limiter (SPEC §7.0) + DataLoader metering wiring.

All timing is driven through injected clocks — nothing here really sleeps.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from prism.data_loader import DataLoader
from prism.io import DataBudgetExhausted, TokenBucket


class _Clock:
    """Fake monotonic clock + sleep that advances the clock."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds


def _bucket(per_minute=8, per_day=800, today=lambda: date(2026, 7, 6)) -> tuple[TokenBucket, _Clock]:
    clock = _Clock()
    bucket = TokenBucket(
        per_minute=per_minute,
        per_day=per_day,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        today=today,
    )
    return bucket, clock


def test_burst_up_to_per_minute_never_sleeps() -> None:
    bucket, clock = _bucket()
    for _ in range(8):
        bucket.acquire()
    assert clock.sleeps == []
    assert bucket.used_today == 8


def test_ninth_call_sleeps_for_one_token() -> None:
    bucket, clock = _bucket()
    for _ in range(8):
        bucket.acquire()
    bucket.acquire()  # bucket empty -> waits for one token at 8/min
    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] == pytest.approx(60.0 / 8.0, rel=1e-9)


def test_tokens_refill_with_elapsed_time() -> None:
    bucket, clock = _bucket()
    for _ in range(8):
        bucket.acquire()
    clock.now += 30.0  # half a minute refills 4 tokens
    for _ in range(4):
        bucket.acquire()
    assert clock.sleeps == []
    bucket.acquire()  # the 5th needs a wait again
    assert len(clock.sleeps) == 1


def test_daily_budget_exhaustion_raises_and_rolls_over() -> None:
    day = {"value": date(2026, 7, 6)}
    bucket, clock = _bucket(per_minute=1000, per_day=5, today=lambda: day["value"])
    for _ in range(5):
        bucket.acquire()
    assert bucket.remaining_today == 0
    with pytest.raises(DataBudgetExhausted, match="Daily request budget"):
        bucket.acquire()
    # The next local date resets the budget.
    day["value"] = date(2026, 7, 7)
    bucket.acquire()
    assert bucket.used_today == 1


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="per_minute"):
        TokenBucket(per_minute=0)
    with pytest.raises(ValueError, match="per_day"):
        TokenBucket(per_day=0)


# ---------------------------------------------------------------------------
# DataLoader wiring
# ---------------------------------------------------------------------------


class _CountingBucket(TokenBucket):
    def __init__(self, *, exhausted: bool = False) -> None:
        super().__init__(per_minute=1000, per_day=1000)
        self.acquires = 0
        self._exhausted = exhausted

    def acquire(self) -> None:
        self.acquires += 1
        if self._exhausted:
            raise DataBudgetExhausted("Daily request budget spent (test)")
        super().acquire()


class _FakeResp:
    def __init__(self, payload) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_loader_meters_network_calls_but_not_cache_hits(tmp_path, monkeypatch) -> None:
    def fake_get(url, params=None, timeout=None):
        return _FakeResp(
            {"dividends": [{"ex_date": "2021-02-05", "amount": 0.2}]}
        )

    monkeypatch.setattr("prism.data_loader.requests.get", fake_get)
    bucket = _CountingBucket()
    loader = DataLoader(api_key="test", cache_dir=tmp_path, rate_limiter=bucket)

    loader.fetch_dividends("AAPL", "2021-01-01", "2021-12-31")
    assert bucket.acquires == 1
    # Second call is served from the cache: no token consumed.
    loader.fetch_dividends("AAPL", "2021-01-01", "2021-12-31")
    assert bucket.acquires == 1


def test_loader_budget_exhaustion_propagates(tmp_path, monkeypatch) -> None:
    """DataBudgetExhausted must escape the broad except handlers (N7):
    neither an empty frame nor a cached negative may absorb it."""

    def fake_get(url, params=None, timeout=None):  # pragma: no cover — never reached
        raise AssertionError("network call should not happen after budget exhaustion")

    monkeypatch.setattr("prism.data_loader.requests.get", fake_get)
    loader = DataLoader(
        api_key="test", cache_dir=tmp_path, rate_limiter=_CountingBucket(exhausted=True)
    )
    with pytest.raises(DataBudgetExhausted):
        loader.fetch_historical_data("AAPL", "1d", "2021-01-01", "2021-12-31")
    with pytest.raises(DataBudgetExhausted):
        loader.fetch_dividends("AAPL", "2021-01-01", "2021-12-31")
    assert not list(tmp_path.glob("*.parquet"))  # nothing fabricated, nothing cached


def test_default_loader_has_vendor_budget(tmp_path) -> None:
    loader = DataLoader(api_key="test", cache_dir=tmp_path)
    assert isinstance(loader.rate_limiter, TokenBucket)
    assert loader.rate_limiter.per_minute == 8
    assert loader.rate_limiter.per_day == 800
