# Quickstart — from clone to a nightly paper loop

This gets the project's paper book, monthly 12−1 cross-sectional momentum on
the S&P 500 ([design](momentum_design.md)), trading **paper money** on your own
free Alpaca account. Every fill, daily account value and difference between
held and target positions is written to append-only ledgers.

It does not claim you will make money. The book's recorded backtest earned a
net Sharpe ratio of 0.465 over its 2021–2026 out-of-sample window, on price
returns that exclude dividends. Its deflated
Sharpe ratio, the probability that the true Sharpe is above zero once the 17
configurations its experiment tried are allowed for, is 0.19
([limitations](limitations.md)). No configuration has met the project's
conditions for real money ([deployment gate](../SPEC.md#6-deployment-gate)).
The one configuration approved under the project's rules, volatility-scaled
momentum, beat its T-bill benchmark by 0.018 Sharpe on the evidence it was
approved on, and it is not what this quickstart trades. The project's first
research program, daily residual reversion, was published as a negative result.
Prism is a harness for finding out honestly. Nothing here is investment advice;
the licence is MIT, with no warranty. Terms you do not recognise are in the
[glossary](glossary.md).

## 1. Prerequisites

- Linux or WSL2, Python ≥ 3.12, [uv](https://docs.astral.sh/uv/).
- A free [Alpaca](https://alpaca.markets) account. Only the **paper** API
  keys are needed. Prism never touches bank credentials; funding and account
  linkage live at the broker ([security](security.md), section 1).
- Optional: a free [Twelve Data](https://twelvedata.com) key. The research
  scripts use it; the paper loop uses Alpaca's own data feed.

```sh
git clone https://github.com/boom90lb/prism && cd prism
uv sync    # core + dev; --extra research is Linux-only and not needed here
```

## 2. Credentials

Create `.env` in the repository root. It is gitignored; the rules are in
[security](security.md).

```
APCA_API_KEY_ID=<your paper key id>
APCA_API_SECRET_KEY=<your paper secret>
TWELVEDATA_API_KEY=<optional>
```

Leave `APCA_API_BASE_URL` unset. The paper endpoint is the default, and
pointing at the live endpoint is a deliberate act this quickstart does not
cover (section 7).

## 3. Preflight

```sh
uv run prism-doctor              # offline: env, universe, data dir, loop state, kill switch
uv run prism-doctor --network    # adds live probes: Alpaca account, Twelve Data (1 credit)
```

`prism-doctor` is the project's health check. Fix anything that FAILs; WARNs
are runnable but worth a look. Every first-run failure this catches would
otherwise appear as a crash in the middle of the evening's run.

## 4. Universe

The point-in-time S&P 500 membership file ships in-tree
(`data/universe/sp500_current.txt`). You do not need to rebuild it. If you
edit it: a held name missing from the file is still fetched for valuation
and exit, but edits take effect at the next valuation — run `prism-doctor`
after any edit.

## 5. One cycle by hand

Every invocation names its mode. `--no-submit` rehearses: it reads the venue,
runs the whole cycle, logs every order it would have sent and sends none.
`--submit` trades. Neither flag exits 2 and does nothing. Run against the
directory `runs/ACTIVE_RUN_DIR` names; a submitting run is refused from any
other directory, and from a bar that is still trading (run it after the close;
the scheduled wrapper runs at 18:30 PT).

Rehearse first, at any hour:

```sh
uv run python -m prism.scripts.paper_loop \
    --book momentum \
    --universe-file data/universe/sp500_current.txt \
    --run-dir "$(cat runs/ACTIVE_RUN_DIR)" \
    --no-submit
```

A real session, once after the market close (next-open auction orders must
reach Alpaca before ~09:28 ET the next morning):

```sh
uv run python -m prism.scripts.paper_loop \
    --book momentum \
    --universe-file data/universe/sp500_current.txt \
    --run-dir "$(cat runs/ACTIVE_RUN_DIR)" \
    --submit
```

The first run fetches about three years of bars for ~500 names (minutes),
scores, constructs the decile long/short book, and submits next-open orders.
Re-running the same evening is safe: the loop persists its decision before
submitting anything and a rerun resumes it — it never double-trades. A new
book starts by writing its directory to `runs/ACTIVE_RUN_DIR` and passing
`--allow-new-run-dir` with `--submit` on its first session only.

The next morning after the open, sweep the orders the opening auction did not
fill (the paper venue fills only about 20–25% of them; the scheduled wrapper of
section 6 runs this at 06:50 PT, 09:50 ET):

```sh
uv run python -m prism.scripts.paper_sweep --run-dir "$(cat runs/ACTIVE_RUN_DIR)" --no-submit   # rehearse
uv run python -m prism.scripts.paper_sweep --run-dir "$(cat runs/ACTIVE_RUN_DIR)" --submit      # trade
```

## 6. Schedule it

Two wrappers ship in-tree; point any scheduler — cron, systemd timers, or
Windows Task Scheduler via `wsl.exe` — at them, weekdays only:

```sh
ops/paper_loop_nightly.sh      # evening: one cycle, then the health verdict
ops/paper_sweep_morning.sh     # morning: the completion sweep, then the health verdict
```

Both read `runs/ACTIVE_RUN_DIR` (one line, the live run directory relative to
the repo) and `.env`, and pass `--submit`. A missing or empty switch refuses
the session (exit 4) rather than falling back to a retired directory. A holiday
run holds the previous session's completed bar and resumes that decision.

Each wrapper runs `prism-doctor` after its work and raises an alert if
*either* exits nonzero: a marker file at `{run-dir}/ALERT`, a loud log line and
a nonzero exit. This matters because a sweep with nothing pending legitimately
exits 0, so a quiet morning is not evidence that last night ran. Set
`PRISM_ALERT_CMD` in `.env` to route alerts somewhere you actually read
([operations](operations.md)); `ops/alert_curl.sh` is a minimal transport.

What accumulates in the run directory:

| File | What it is |
|---|---|
| `equity.jsonl` | one account-value row per session; the monitor's return series |
| `fills.jsonl` | every fill beside its decision-time reference price |
| `targets.jsonl` | the target portfolio at each rebalance |
| `concordance.jsonl` | the difference between held and target positions, per session |
| `unfilled.jsonl` | every order the venue did not print |
| `state.json` | durable loop state — never edit by hand |
| `nightly.log` / `sweep.log` | per-session wrapper log, verdict line per run |
| `ALERT` | present = an unresolved unhealthy session; a healthy nightly clears it |

## 7. Safety rails and stopping

- **Stop the book:** `touch <run-dir>/KILL_SWITCH`. The loop still settles
  fills and marks the account's value but submits nothing; delete the file to
  resume.
- **Drawdown halt:** by default the loop halts below 50% of its peak account
  value (`--max-drawdown`); tighten it if you care about the account.
- **Order guards:** per-order notional and count bounds veto a corrupted
  decision before anything is persisted or submitted.
- **Real money is out of scope here.** It requires explicitly setting
  `APCA_API_BASE_URL` to the live endpoint, and no configuration has met the
  project's conditions for real money (SPEC §6). Whole-share sizing also sets
  a measured account floor: about $100k holds the roughly 100-name book
  faithfully, while $10k trades a structurally different portfolio
  ([account-size floor](account_size_floor.md)).

## 8. Where to go next

- [README](../README.md): what the evidence says and what is in progress.
- [Limitations](limitations.md): what the published results do and do not establish.
- [Operations](operations.md): operational sharp edges.
- [Free-tier profile](free_tier_profile.md): exactly what free data and keys reproduce.
- [Operating contract](../SPEC.md): the project's rules, for maintainers.
- [Documents](README.md): an index of everything else.
