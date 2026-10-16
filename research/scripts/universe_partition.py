"""Seeded universe partitions for holdings-diversity diagnostics (a library module, not a CLI).

A partition keeps a name when a keyed hash of its security id falls below ``keep``:
``HMAC-SHA256(key = seed as 8 big-endian bytes, message = sid in UTF-8)``, whose first
8 bytes read as an unsigned integer and divided by 2**64 give a draw in [0, 1). Each name
is kept or dropped independently, so a seed keeps about ``keep`` of any universe, and a
name's fate depends only on (seed, sid): it is the same on every bar, whatever else is in
the cross-section on that date, which is what lets a partition be applied to a
point-in-time membership that changes over time.

Seeds are integers in [0, 2**63) so that they fit a TOML integer; draw one with
``secrets.randbits(63)``. The draw is a hash, not a random stream, so the same seed gives
the same partition on any machine and in any Python version.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Iterable

__all__ = ["MAX_SEED", "draw", "partition"]

#: Seeds are below this so that a registration can store them as TOML integers.
MAX_SEED = 1 << 63


def _key(seed: int) -> bytes:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < MAX_SEED:
        raise ValueError(f"a partition seed is an integer in [0, 2**63), got {seed!r}")
    return seed.to_bytes(8, "big")


def draw(sid: str, seed: int) -> float:
    """The name's uniform draw in [0, 1) under ``seed``: HMAC-SHA256's first 8 bytes / 2**64."""
    if not isinstance(sid, str) or not sid:
        raise ValueError(f"a security id is a non-empty string, got {sid!r}")
    digest = hmac.new(_key(seed), sid.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def partition(sids: Iterable[str], seed: int, keep: float = 0.8) -> tuple[str, ...]:
    """The names of ``sids`` kept under ``seed``: those whose :func:`draw` is below ``keep``.

    Input order is preserved. A duplicated sid is refused, because a cross-section holds
    one row per name and a duplicate means the caller's universe is malformed.
    """
    if not 0.0 < float(keep) <= 1.0:
        raise ValueError(f"keep must be in (0, 1], got {keep!r}")
    _key(seed)  # refuse a bad seed even on an empty universe
    names = list(sids)
    if len(set(names)) != len(names):
        raise ValueError("partition received a duplicated security id")
    return tuple(sid for sid in names if draw(sid, seed) < float(keep))
