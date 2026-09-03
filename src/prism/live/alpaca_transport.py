"""Shared Alpaca REST transport (SPEC.md §7.4) — one retry/auth/error discipline.

Every Alpaca surface the live loop touches — the trading API
(:mod:`prism.live.alpaca`), the market-data API
(:mod:`prism.live.alpaca_data`), and the corporate-actions endpoint
(:mod:`prism.live.spinoff_mask`) — speaks through this session object, so
credential handling, timeouts, and the transient-failure policy exist once
instead of three diverging copies (the pre-2026-07-29 state: the bar source
retried 429/5xx while the *broker* — the surface whose failed session cannot
be backfilled as live evidence, docs/operations.md — retried nothing).

Retry policy: 429 and transient 5xx are retried up to ``max_retries`` times,
honoring a ``Retry-After`` header when present and exponential backoff
otherwise. The final response (success or the last error) is returned for
:meth:`AlpacaSession.json_or_raise` to interpret — a genuine 4xx (403
buying-power, 422 duplicate-id) is never retried and surfaces immediately for
the caller's own mapping. Retrying a POST is safe *for this venue* because
every order carries a deterministic ``client_order_id``: a retry of an
already-accepted submit is rejected as a duplicate, which the write-ahead
protocol treats as success (``prism.live.broker``).

Credentials travel only in request headers — never in URLs, exceptions, or
logs (docs/security.md §2).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)

#: Rate limit + transient server errors — the statuses worth a bounded retry.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 0.5


class AlpacaAPIError(RuntimeError):
    """A non-2xx venue response the adapter cannot map onto its contract.

    Carries ``status_code`` and the (truncated) response body. For ``submit``
    the loop's crash-safety semantics apply: the order may or may not have
    been accepted, and the write-ahead protocol retries it idempotently on
    the next pass.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_env_credentials(context: str) -> tuple[str, str]:
    """``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``, or raise naming ``context``.

    Missing credentials fail loud here (N7) — a client that silently
    constructs unauthenticated would fail one request later with a less
    actionable error.
    """
    key_id = os.environ.get("APCA_API_KEY_ID", "")
    secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key_id or not secret_key:
        raise RuntimeError(
            "APCA_API_KEY_ID / APCA_API_SECRET_KEY are not set; "
            f"export the paper-account credentials before {context} (N7)"
        )
    return key_id, secret_key


class AlpacaSession:
    """Auth + timeout + bounded transient retry over an injectable session.

    ``session`` is any requests-compatible object (``request(method, url,
    headers=, timeout=, **kwargs)``), so every adapter mapping is tested
    offline against canned payloads; only the real HTTP transport is
    network-gated. ``sleep`` is injectable for the same reason.
    """

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        base_url: str,
        *,
        session: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not key_id or not secret_key:
            raise ValueError("Alpaca key_id and secret_key must be non-empty")
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self.base_url = base_url.rstrip("/")
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout
        self._max_retries = int(max_retries)
        self._backoff_base = float(backoff_base)
        self._sleep = sleep or time.sleep

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """One HTTP call, retrying rate-limit/transient statuses with backoff."""
        url = self.base_url + path
        response: Any = None
        for attempt in range(self._max_retries + 1):
            response = self._session.request(
                method,
                url,
                headers=self._headers,
                timeout=self._timeout,
                **kwargs,
            )
            if response.status_code not in RETRY_STATUSES or attempt == self._max_retries:
                return response
            headers = getattr(response, "headers", None) or {}
            try:
                retry_after = float(headers.get("Retry-After", 0) or 0)
            except (TypeError, ValueError):
                retry_after = 0.0
            wait = retry_after if retry_after > 0 else self._backoff_base * (2.0**attempt)
            logger.warning(
                "Alpaca %s -> HTTP %s (attempt %d/%d); backing off %.1fs",
                path,
                response.status_code,
                attempt + 1,
                self._max_retries,
                wait,
            )
            self._sleep(wait)
        return response

    @staticmethod
    def json_or_raise(response: Any, context: str) -> Any:
        if 200 <= response.status_code < 300:
            return response.json()
        body = (response.text or "")[:500]
        raise AlpacaAPIError(f"{context} -> HTTP {response.status_code}: {body}", response.status_code)
