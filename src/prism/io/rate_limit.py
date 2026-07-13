"""Token-bucket rate limiting for the $0 data stack (SPEC.md §7.0, §4).

The binding data constraint is the keyed Twelve Data tier: 8 requests/minute,
800/day. The per-minute side is a classic token bucket — ``acquire`` blocks
(sleeps) until a token refills — because a short wait is the correct response
to burst pressure. The per-day side is a hard budget: exhausting it raises
``DataBudgetExhausted`` (N7 — a run that would silently starve for hours must
fail loud instead), and the counter resets on the local-date rollover, which
is how the vendor meters it.

Clock, sleep, and today are injectable so the arithmetic is testable offline
without real waiting. Thread-safe; one bucket meters one API key, so a
process with several ``DataLoader`` instances against the same key should
share a single bucket explicitly.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date
from typing import Callable

logger = logging.getLogger(__name__)

# The verified Twelve Data free-tier budget (SPEC §4).
TWELVEDATA_PER_MINUTE = 8
TWELVEDATA_PER_DAY = 800


class DataBudgetExhausted(RuntimeError):
    """The hard daily request budget is spent; retrying today cannot succeed."""


class TokenBucket:
    """Blocking token bucket with a hard daily cap.

    ``acquire()`` consumes one request slot: it sleeps while the per-minute
    bucket is empty and raises :class:`DataBudgetExhausted` once the daily
    budget is spent. Tokens refill continuously at ``per_minute / 60`` per
    second up to a burst capacity of ``per_minute``.
    """

    def __init__(
        self,
        *,
        per_minute: int = TWELVEDATA_PER_MINUTE,
        per_day: int = TWELVEDATA_PER_DAY,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], date] = date.today,
    ) -> None:
        if per_minute < 1:
            raise ValueError(f"per_minute must be >= 1, got {per_minute}")
        if per_day < 1:
            raise ValueError(f"per_day must be >= 1, got {per_day}")
        self.per_minute = int(per_minute)
        self.per_day = int(per_day)
        self._monotonic = monotonic
        self._sleep = sleep
        self._today = today
        self._rate = self.per_minute / 60.0  # tokens per second
        self._tokens = float(self.per_minute)
        self._last_refill = monotonic()
        self._day = today()
        self._used_today = 0
        self._lock = threading.Lock()

    @property
    def used_today(self) -> int:
        """Requests consumed against today's budget."""
        with self._lock:
            self._roll_day()
            return self._used_today

    @property
    def remaining_today(self) -> int:
        """Requests left in today's budget."""
        with self._lock:
            self._roll_day()
            return self.per_day - self._used_today

    def acquire(self) -> None:
        """Consume one request slot, sleeping through per-minute pressure.

        Raises:
            DataBudgetExhausted: the daily budget is spent (resets at the
                local-date rollover).
        """
        with self._lock:
            self._roll_day()
            if self._used_today >= self.per_day:
                raise DataBudgetExhausted(
                    f"Daily request budget spent ({self.per_day}/{self.per_day}); "
                    f"resets on the next local-date rollover (currently {self._day})"
                )
            self._refill()
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                logger.info("Rate limit: sleeping %.1fs for a request token", wait)
                self._sleep(wait)
                self._refill()
                # Continuous refill over `wait` yields >= 1 token up to float
                # rounding; clamp so the accounting never goes negative.
                self._tokens = max(self._tokens, 1.0)
            self._tokens -= 1.0
            self._used_today += 1
            if self._used_today in (int(self.per_day * 0.8), self.per_day):
                logger.warning(
                    "Data budget at %d/%d requests today", self._used_today, self.per_day
                )

    def _refill(self) -> None:
        now = self._monotonic()
        elapsed = max(0.0, now - self._last_refill)
        self._last_refill = now
        self._tokens = min(float(self.per_minute), self._tokens + elapsed * self._rate)

    def _roll_day(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._used_today = 0
