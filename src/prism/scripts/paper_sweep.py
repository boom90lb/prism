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

**Exercising the command without trading**: ``--no-submit`` reads the venue and
computes the residuals exactly as a real sweep does, logs every order it would
have sent, and sends none (``prism.live.no_submit``); the run directory's
durable record is untouched. Use it for any verification — the sweep, like the
loop, otherwise had no state between "fails before the credential check" and
"trades the account". A sweep that does submit says so on one line
(``prism.live.loop.SUBMISSION_ANNOUNCEMENT``) naming the endpoint, the account,
the decision bar and the order count.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from prism.live import AlpacaBroker, LiveLoopContext, StateStore, sweep_pending
from prism.live.authorization import refuse_an_unbindable_record, resolve_endpoint
from prism.live.lockfile import run_dir_lock
from prism.live.no_submit import no_submit_session
from prism.live.safety import require_established_run_dir, sample_run_dir_history
from prism.workspace import load_env

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
    parser.add_argument(
        "--kill-switch",
        type=Path,
        default=None,
        help="Halt-file path; its PRESENCE stops the sweep (exit 2). Default "
        "{run-dir}/KILL_SWITCH — the same file that halts the evening loop, so one "
        "touch stops every path to the venue. 'off' disables the rail.",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Run the whole sweep — credentials, broker reads, residual computation — log every "
        "order it WOULD submit, and submit nothing. The venue is read but never written, and the "
        "run directory's durable record is left untouched (writes go to a discarded scratch copy). "
        "Use this for any verification of the command.",
    )
    parser.add_argument(
        "--allow-new-run-dir",
        action="store_true",
        help="Permit submitting from a --run-dir that carries no prior session (no state.json and "
        "no equity.jsonl when the process started). An established run directory — every morning "
        "invocation — needs nothing. Ignored under --no-submit, which submits nothing wherever it "
        "points.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    # The provenance rail's evidence, sampled before the writer lock below
    # creates the directory it is asked about.
    run_dir_history = sample_run_dir_history(args.run_dir)
    # Credentials come from the project's .env, loaded here rather than at
    # import (spec §3.1) and before ``AlpacaBroker.from_env`` below.
    load_env()
    # §3.10, before a credential is read and before a client exists: this entry
    # point is the paper instrument's sweep, so its mode is `paper` and an
    # exported APCA_API_BASE_URL may set an endpoint but never the mode; and it
    # resolves no specification, so it computes no `config_id` and can bind
    # against no authorization record found in the run directory.
    resolve_endpoint("paper", from_environment=os.environ.get("APCA_API_BASE_URL"))
    refuse_an_unbindable_record(
        args.run_dir,
        entry_point="the paper-sweep entry point",
        instead="`prism run --spec <spec> --account paper --broker alpaca --once --stage sweep`",
    )
    kill_switch = args.run_dir / "KILL_SWITCH" if args.kill_switch is None else args.kill_switch
    if str(kill_switch) != "off" and Path(kill_switch).exists():
        logger.error("SAFETY HALT (kill switch present at %s): not sweeping", kill_switch)
        return 2
    # One writer per run directory (prism.live.lockfile): the sweep appends to
    # the same persisted pending set the evening loop owns, and the write-ahead
    # protocol protects against crashes, not concurrent writers. The lock is
    # taken on the REAL run directory even under --no-submit, whose mirror is
    # a snapshot of it.
    with run_dir_lock(args.run_dir), no_submit_session(args.run_dir, enabled=args.no_submit) as rehearsal:
        # With --no-submit this is a scratch mirror; otherwise it IS the run
        # directory, so one code path serves both.
        work_dir = rehearsal.work_dir
        # DAY market orders: the auction the original OPG orders were for has
        # already happened; the sweep executes at the open market.
        broker = AlpacaBroker.from_env(mode="paper", time_in_force="day")
        if args.no_submit:
            broker = rehearsal.wrap(broker)
        else:
            # The provenance rail, before the first venue read.
            require_established_run_dir(
                run_dir_history, allow_new=args.allow_new_run_dir, entry_point="the paper sweep"
            )
        ctx = LiveLoopContext(
            store=StateStore(work_dir / "state.json"),
            broker=broker,
            fills_ledger=work_dir / "fills.jsonl",
            unfilled_ledger=work_dir / "unfilled.jsonl",
        )
        swept = sweep_pending(ctx, sweep_suffix=args.sweep_suffix)
    logger.info(
        "sweep: %d residual orders %s",
        len(swept),
        "WOULD HAVE been submitted" if args.no_submit else "submitted",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
