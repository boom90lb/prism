"""Morning completion sweep for the paper loop (SPEC §13 R2; I-9 instrument).

The Alpaca *paper* venue's opening-auction simulation prints only ~20-25% of
OPG orders, so an unswept monthly refresh holds a fraction of the decided
book for a whole cadence period and the paper stream measures a different
portfolio than the decision constructed. This CLI runs
:func:`prism.live.loop.sweep_pending` once, after the open, while the
evening decision is still pending: every terminal-but-unexecuted residual is
re-submitted as a plain DAY market order under a ``:S1``-suffixed client id.
The evening cycle's settle then ledgers auction fills and sweep fills alike
against the same decision-close reference price — the total arrival cost of
executing the decision at this venue — with the id suffix keeping the two
fill populations segmentable for I-9 calibration
(docs/momentum_design.md, instrument amendment 2026-07-11).

Run once per session, after the open (the OPG auction must be terminal;
~09:35 ET is comfortable):

    python -m prism.scripts.paper_sweep --run-dir runs/paper_loop_momentum

A rerun is idempotent (deterministic suffixed ids; the venue's duplicate-id
rejection counts as submitted), and a sweep with nothing pending is a no-op.
Like :mod:`prism.scripts.paper_loop`, this shell holds no logic worth
testing; the sweep semantics live in ``prism.live.loop`` and are exercised
offline (tests/test_live_loop.py).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from prism.live import AlpacaBroker, LiveLoopContext, StateStore, sweep_pending

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-submit the pending decision's unexecuted residuals as DAY market orders."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/paper_loop"),
        help="The paper loop's run directory (state.json + ledgers) to sweep.",
    )
    parser.add_argument(
        "--sweep-suffix",
        default="S1",
        help="Client-id suffix for sweep orders ({original_id}:{suffix}); one "
        "completion generation per decision.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    ctx = LiveLoopContext(
        store=StateStore(args.run_dir / "state.json"),
        # DAY market orders: the auction the original OPG orders were for has
        # already happened; the sweep executes at the open market.
        broker=AlpacaBroker.from_env(time_in_force="day"),
        fills_ledger=args.run_dir / "fills.jsonl",
        unfilled_ledger=args.run_dir / "unfilled.jsonl",
    )
    swept = sweep_pending(ctx, sweep_suffix=args.sweep_suffix)
    logger.info("sweep: %d residual orders submitted", len(swept))
    return 0


if __name__ == "__main__":
    sys.exit(main())
