"""Append-only observatory capture store (W5; non-gating).

Capture is time-irreversible — every uncaptured day is gone — so this module
provides a tiny, production-safe surface for point-in-time *expectation state*
payloads (news flow, EDGAR cadence, coverage counts, or any future factory
input). Modeling is deferred until a factory pre-registration exists
(``docs/factory_amendment.md``); this module only **writes and reads**
verbatim JSON lines, optionally gzip-compressed.

Contract:

* Each record is one JSON object with at least ``captured_at`` (UTC ISO-8601
  string supplied by the caller) and ``lane`` (string namespace).
* Appends never rewrite prior bytes (open ``ab`` / gzip append).
* Reads are for inspection and tests; they do not mutate.
* No network I/O here — fetchers live at the call site and pass payloads in.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator


REQUIRED_FIELDS = ("captured_at", "lane")


def append_capture(
    path: Path | str,
    record: dict[str, Any],
    *,
    gzip_compress: bool | None = None,
) -> None:
    """Append one capture record to ``path`` (``.jsonl`` or ``.jsonl.gz``).

    ``gzip_compress`` defaults from the suffix (``.gz`` → True). Missing
    required fields raise (N7). Parent directories are created.
    """
    path = Path(path)
    for field in REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"capture record missing required field {field!r}")
        if not record[field]:
            raise ValueError(f"capture record field {field!r} must be non-empty")
    if gzip_compress is None:
        gzip_compress = path.suffix == ".gz" or path.name.endswith(".jsonl.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if gzip_compress:
        with gzip.open(path, "ab") as fh:
            fh.write(line)
    else:
        with path.open("ab") as fh:
            fh.write(line)


def iter_captures(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield records from a capture file in append order."""
    path = Path(path)
    if not path.exists():
        return
        yield  # pragma: no cover — makes this a generator even when empty
    gzip_compress = path.suffix == ".gz" or path.name.endswith(".jsonl.gz")
    opener = gzip.open if gzip_compress else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[arg-type]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_captures(path: Path | str) -> int:
    return sum(1 for _ in iter_captures(path))
