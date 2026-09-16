"""Production-side scripts. Only the periodic universe builder lives here; the
batch WFO / sweep / stat-arb CLIs are quarantined under research/ (SPEC §9).

Every module here is a shell over a command of engine specification section
3.14, and each keeps working as an alias through stage 6 (the Vocabulary
(Correction) row: "`python -m prism.scripts.paper_loop`, `prism-doctor`, and
`prism-build-universe` keep working as aliases through this stage and print
their replacement"). :func:`announce_alias` is how they print it.
"""

from __future__ import annotations

import sys

__all__ = ["announce_alias"]


def announce_alias(alias: str, replacement: str, *, stream: object | None = None) -> None:
    """Name the `prism` command that replaces this entry point, on stderr.

    stderr and not stdout: the doctor's report, the universe builder's summary
    and the loop's log lines are the readable output of these commands, and a
    notice must not land inside them. It is one line, printed before any work,
    so the operator sees it whether or not the run succeeds.
    """
    print(
        f"note: `{alias}` is an alias kept through stage 6; its replacement is `{replacement}`.",
        file=stream if stream is not None else sys.stderr,
    )
