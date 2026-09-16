"""``python -m prism`` — the same entry point as the ``prism`` console script.

Useful before an install (from a checkout with ``src`` on the path) and inside
a CI job that has not put the script directory on ``PATH``.
"""

from __future__ import annotations

from prism.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
