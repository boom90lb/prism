"""CLI: anytime-valid monitor read over the paper-loop equity ledger (SPEC §10).

A thin shell over :func:`prism.live.monitor.paper_monitor_read`: it reads the
mark-to-market equity ledger the daily loop appends (``runs/paper_loop/equity.jsonl``)
and prints the time-uniform confidence-sequence verdict on mean daily net return
vs a hurdle. Additive telemetry beside the ratified rolling-PSR read — no counted
trial, no pre-registration. Safe to run any time; it only reads.

    python -m prism.scripts.paper_monitor --run-dir runs/paper_loop --hurdle 0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prism.live.monitor import DEFAULT_NAV_RETURN_BOUND, paper_monitor_read


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Anytime-valid confidence-sequence read over the paper equity ledger."
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/paper_loop"))
    parser.add_argument(
        "--hurdle",
        type=float,
        default=0.0,
        help="Per-period net-return bar (0 tests 'beats zero'; pass a periodic hurdle for viability).",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--bound",
        type=float,
        default=DEFAULT_NAV_RETURN_BOUND,
        help="Assumed |daily NAV return| support; tighter = narrower interval (validity holds regardless).",
    )
    parser.add_argument("--opt-horizon", type=int, default=252)
    args = parser.parse_args(argv)

    read = paper_monitor_read(
        args.run_dir / "equity.jsonl",
        hurdle=args.hurdle,
        alpha=args.alpha,
        bound=args.bound,
        opt_horizon=args.opt_horizon,
    )
    print(json.dumps(read, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
