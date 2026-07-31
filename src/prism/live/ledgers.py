"""Append-only JSONL ledger mechanics (SPEC.md §7.7; the I-9 record discipline).

Every durable record a run directory holds — fills, equity marks, refresh
targets, unfilled residuals, concordance and regime telemetry — is an
append-only JSONL file. This module is the one implementation of that
discipline; the schema-specific appenders in :mod:`prism.live.loop` delegate
here instead of each hand-rolling file I/O (the pre-2026-07-29 state: four
copies of the monotone-append read/compare/write and five copies of the
line-parse read).

Three properties the mechanics guarantee:

* **Durable appends.** Rows are flushed and fsynced before the append
  returns — the ledgers carry the loop's evidence stream, and the state file
  they sit beside is already fsynced (``prism.live.state``); a ledger that
  can silently lose its tail to a crash is a weaker record than the state
  that references it.
* **A torn tail never wedges the loop.** A crash mid-append can leave a
  partial final line. Readers skip unparseable lines with a loud ERROR
  naming the file and count (N7: loud, never silent — and never fatal,
  because ``halt_reason`` reads the equity ledger *before* every decision,
  so a reader that raises on a torn tail converts one crash into a
  permanently wedged book). Appends heal a torn tail first (a missing
  trailing newline gets one), so the fragment is quarantined as its own
  skipped line instead of corrupting the next row.
* **Monotone idempotency.** ``append_monotone`` writes a row only when its
  key is strictly greater than the last recorded key, so a same-bar restart
  (the write-ahead protocol's resume) never duplicates a session row.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def append_rows(path: Path, rows: list[dict]) -> None:
    """Durably append ``rows`` (one JSON object per line), healing a torn tail."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torn = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as check:
            check.seek(-1, os.SEEK_END)
            torn = check.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as handle:
        if torn:
            # Quarantine the partial line a crash mid-append left behind: with
            # its own newline it becomes one skipped (loudly) line instead of
            # a prefix that corrupts this row too.
            logger.error(
                "%s does not end in a newline (torn tail from a crash mid-append); "
                "sealing the fragment as its own line before appending (N7)",
                path,
            )
            handle.write("\n")
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict]:
    """Every parseable row, oldest first; unparseable lines are skipped loudly.

    A missing file is an empty ledger. A line that fails to parse (or parses
    to a non-object) is counted and reported at ERROR — one such line is the
    torn tail of a crash mid-append (healed by the next append); more than
    one means real corruption worth eyes. Either way the loop keeps its
    record and keeps running: the reader's job is to surface damage, not to
    convert it into a permanently wedged book.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if not isinstance(row, dict):
            bad += 1
            continue
        rows.append(row)
    if bad:
        logger.error(
            "%s: skipped %d unparseable line(s) — one is a crash's torn tail (self-healing); "
            "more than one is real corruption, inspect the file (N7: loud, never wedged)",
            path,
            bad,
        )
    return rows


def read_frame(path: Path) -> pd.DataFrame:
    """The ledger as a DataFrame (empty for a missing/empty ledger)."""
    return pd.DataFrame(read_rows(path))


def last_value(path: Path, field: str) -> Any | None:
    """The newest recorded value of ``field``, or ``None`` on an empty ledger."""
    for row in reversed(read_rows(path)):
        if field in row:
            return row[field]
    return None


def append_monotone(path: Path, key_field: str, row: dict) -> bool:
    """Append ``row`` iff its ``key_field`` exceeds the last recorded key.

    The idempotent per-session append shared by the equity, targets,
    concordance, and regime ledgers: a same-bar rerun or an out-of-order bar
    is skipped, so the ledger holds exactly one row per key and re-running
    the loop never double-counts a session. Keys are ISO date strings, so
    lexical comparison is chronological. Returns ``True`` when written.
    """
    key = str(row[key_field])
    last = last_value(path, key_field)
    if last is not None and key <= str(last):
        return False
    append_rows(path, [row])
    return True
