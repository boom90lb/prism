"""Durable live-loop state (SPEC.md §7.7: "state is durable").

One JSON document per loop, written atomically (temp file + ``os.replace``)
so a crash mid-write can never leave a torn state file. Loading is
fail-loud (N7): a missing file is a legitimate "fresh loop" (``None``), but
a corrupt or schema-incompatible file raises — silently starting flat over
a real book is the defect this module exists to prevent.

Positions are broker-truth shares (not weights) and cash is dollars: the
decision pipeline thinks in weights, but what must survive a restart is
what reconciles against the broker.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from prism.live.broker import Order

STATE_SCHEMA_VERSION = 1


@dataclass
class LoopState:
    """Everything the daily loop must not lose between processes.

    ``pending_orders`` is the write-ahead record: orders decided at close
    *t* that may or may not have reached the broker yet. While it is
    non-empty, the decision for ``pending_decision_bar`` is immutable — a
    restarted loop resumes submission from this record, it never
    re-decides (re-deciding after a partial submit would double-trade).
    """

    positions: dict[str, float] = field(default_factory=dict)  # shares
    cash: float = 0.0
    pending_orders: list[Order] = field(default_factory=list)
    pending_decision_bar: str | None = None
    last_settled_bar: str | None = None
    schema_version: int = STATE_SCHEMA_VERSION

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "LoopState":
        payload = json.loads(raw)
        version = payload.get("schema_version")
        if version != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"live state schema_version {version!r} != supported "
                f"{STATE_SCHEMA_VERSION}; migrate explicitly, do not start flat (N7)"
            )
        payload["pending_orders"] = [Order(**o) for o in payload.get("pending_orders", [])]
        return cls(**payload)


class StateStore:
    """Atomic-write JSON persistence for :class:`LoopState`."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> LoopState | None:
        """The persisted state, ``None`` for a genuinely fresh loop.

        Anything else — unreadable file, bad JSON, wrong schema — raises.
        """
        if not self.path.exists():
            return None
        raw = self.path.read_text(encoding="utf-8")
        try:
            return LoopState.from_json(raw)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError(
                f"corrupt live state at {self.path}: {exc}; refusing to start flat (N7)"
            ) from exc

    def save(self, state: LoopState) -> None:
        """Write-then-rename so the file on disk is always a whole document."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(state.to_json())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)
