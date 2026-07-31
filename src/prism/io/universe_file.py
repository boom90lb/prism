"""The universe-file read contract (SPEC.md §7.0) — one parser, four consumers.

One symbol per line, blank lines and ``#`` comments skipped, upper-cased.
This is the format ``prism-build-universe`` writes and the paper loop, the
replay driver, and the doctor all read; before 2026-07-29 each of the three
readers hand-rolled the parse with divergent failure semantics. Stdlib only,
so every consumer (including the lazily-importing doctor) can use it without
weight.
"""

from __future__ import annotations

from pathlib import Path


def load_universe_symbols(path: Path, *, require_nonempty: bool = True) -> list[str]:
    """Symbols from a universe file, in file order (duplicates preserved —
    downstream reconciliation dedupes, ``resolve_fetch_universe``).

    ``require_nonempty`` raises on a file that parses to zero symbols (the
    trading path's N7 contract); the doctor passes ``False`` because "zero
    symbols" is a finding it reports, not an error it dies on.
    """
    symbols = [
        line.strip().upper()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if require_nonempty and not symbols:
        raise ValueError(f"universe file {path} has no symbols")
    return symbols
