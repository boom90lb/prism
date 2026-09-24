# Incident, 2026-09-09: 116 unauthorized orders on the paper account

Status: closed, no market impact. Written 2026-09-15 from the venue's own order history
and the session record, because the working notes taken at the time lived in a temporary
directory that has since been cleared. The venue record is the surviving primary source.

## What happened

During stage 6 of the engine overhaul, an automated verification step ran

    python -m prism.scripts.paper_loop --book momentum \
      --universe-file data/universe/sp500_current.txt --run-dir <temp> --spinoff-mask

intending only to confirm the command still parsed and reached its credential check after
three modules moved. It did not stop there. Credentials resolved, and the command ran a
complete decision cycle against Alpaca paper account `PA3RNDGP1KD3`, adopting broker truth
and submitting 116 market-on-open orders for decision bar 2026-09-08.

The run directory was a temporary one, so the account's real run directory
(`runs/paper_loop_momentum2`) received no record of the orders. That directory has not been
written since 2026-08-14 and was not written during this incident (Run, `find -newermt`).

## Timeline, from the venue

| Time (UTC) | Event |
|---|---|
| 2026-09-09 11:14:41 | 116 orders submitted, client order ids `mom:2026-09-08:ABT` through `mom:2026-09-08:ZTS`, all market, time in force `opg` |
| 2026-09-09 11:58 | Orders found still unfilled, status `new`, roughly 90 minutes before the open |
| 2026-09-09 11:59:39 | All 116 cancelled, HTTP 204 on every one |

Verified again on 2026-09-15 (Run, Alpaca orders API, window 2026-09-09 to 2026-09-10):
116 orders that day, all `canceled`, **zero fills**.

## Account state

Restored exactly. Before the incident the account held 99 positions and cash 937,669.47,
matching the owner's recorded `state.json`. After cancellation it held 99 positions and
cash 937,669.47. On 2026-09-15 it holds 0 open orders, 99 positions and cash 937,669.47.
Equity differs across those readings only by mark to market on the held book.

## Root causes, both now acted on

**1. A verification instruction assumed a command would stop where it does not.** The
instruction said to confirm the command "reaches its credential check". Nothing in
`paper_loop` stops there, and at the time it had no flag to run without trading. Fixed in
`ad00833`: `--no-submit` runs the real path, including credential resolution and venue
reads, and refuses only the write; a run that does submit now prints an unmistakable line
naming the endpoint, account, decision bar and order count; and submission is refused
outright when the run directory held neither `state.json` nor `equity.jsonl` at process
start, which is the property that made this incident large (an empty run directory
reconciles against nothing, so the whole book is emitted from flat).

**2. Credentials crossed the worktree isolation boundary.** `.env` discovery
walked up from the current directory without bound, so from
`/home/b90/prism/.claude/worktrees/engine-overhaul` it finds `/home/b90/prism/.env`, the
live checkout's credentials (Run, 2026-09-15: `dotenv.find_dotenv(usecwd=True)` from the
worktree returns `/home/b90/prism/.env`). An isolated development checkout therefore
inherits the live account. Stage 1 chose the cwd walk deliberately, because a
package-relative walk breaks under a wheel install, and recorded the tradeoff; the
consequence for worktrees was not drawn out. **Fixed 2026-09-16**: discovery now stops at the first directory holding a `.git` entry (`prism.workspace.find_dotenv_within_checkout`), so a linked worktree or nested checkout never reads a parent checkout's `.env`; `tests/test_workspace.py::test_load_env_never_crosses_a_checkout_boundary` reproduces the incident layout and asserts nothing is loaded.

## What was not done

No further account action was taken. Nothing was re-submitted, no position was touched,
and the scheduler was not started. Restarting the paper book remains an owner act under
the existing operating rules.

## What this costs

Nothing at the venue: no fill, no position change, no cash change. The cost is to the
evidence record. The paper book has been deliberately dark since 2026-08-14, and any
account activity in that window is now part of its history and has to be read alongside
the clean-session clock and any future observation window.
