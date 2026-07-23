# Quickstart — from clone to a nightly paper loop

This gets the current candidate book — monthly cross-sectional momentum
(`docs/momentum_design.md`) — trading **paper money** on your own free Alpaca
account, with every fill, NAV mark, and book-versus-target divergence
recorded to append-only ledgers.

It does not claim you will make money. The project's own evidence bar
(`SPEC.md` §10) has never been cleared by any configuration: the first
candidate was certified uneconomic and archived, and the momentum candidate's
verdict is unreadable before mid-2027. Prism is a harness for finding out
honestly. Nothing here is investment advice; the license is MIT, no warranty.

## 1. Prerequisites

- Linux or WSL2, Python ≥ 3.12, [uv](https://docs.astral.sh/uv/).
- A free [Alpaca](https://alpaca.markets) account — only the **paper** API
  keys are needed. Prism never touches bank credentials; funding and account
  linkage live at the broker (`docs/security.md` §1).
- Optional: a free [Twelve Data](https://twelvedata.com) key — used by the
  research scripts, not the live loop, which defaults to Alpaca's own feed.

```sh
git clone https://github.com/boom90lb/prism && cd prism
uv sync    # core + dev; --extra research is Linux-only and not needed here
```

## 2. Credentials

Create `.env` in the repo root (gitignored; rules in `docs/security.md`):

```
APCA_API_KEY_ID=<your paper key id>
APCA_API_SECRET_KEY=<your paper secret>
TWELVEDATA_API_KEY=<optional>
```

Leave `APCA_API_BASE_URL` unset — the paper endpoint is the default, and
pointing at the live endpoint is a deliberate act this quickstart does not
cover (§7).

## 3. Preflight

```sh
uv run prism-doctor              # offline: env, universe, data dir, loop state, kill switch
uv run prism-doctor --network    # adds live probes: Alpaca account, Twelve Data (1 credit)
```

Fix anything that FAILs; WARNs are runnable but worth a look. Every
first-run failure class this catches otherwise appears as a mid-loop crash
in the evening.

## 4. Universe

The point-in-time S&P 500 membership file ships in-tree
(`data/universe/sp500_current.txt`). You do not need to rebuild it. If you
edit it: a held name missing from the file is still fetched for valuation
and exit, but edits take effect at the next valuation — run `prism-doctor`
after any edit.

## 5. One cycle by hand

Run once after the market close (next-open auction orders must reach Alpaca
before ~09:28 ET the next morning):

```sh
uv run python -m prism.scripts.paper_loop \
    --book momentum \
    --universe-file data/universe/sp500_current.txt \
    --run-dir runs/paper_loop_momentum
```

The first run fetches about three years of bars for ~500 names (minutes),
scores, constructs the decile long/short book, and submits next-open orders.
Re-running the same evening is safe: the loop persists its decision before
submitting anything and a rerun resumes it — it never double-trades.

Next morning after the open (~09:35 ET), sweep the orders the auction did
not fill (the paper venue prints only ~20–25% of them):

```sh
uv run python -m prism.scripts.paper_sweep --run-dir runs/paper_loop_momentum
```

## 6. Schedule it

Wrap the two commands in any scheduler — cron, systemd timers, or Windows
Task Scheduler via `wsl.exe`: an evening loop run (between the close and
~09:00 ET) and a morning sweep (~09:35 ET), weekdays only. A holiday run
no-ops on the stale-panel guard.

What accumulates in the run directory:

| File | What it is |
|---|---|
| `equity.jsonl` | one NAV row per session — the monitor's return stream |
| `fills.jsonl` | every fill beside its decision-time reference price |
| `targets.jsonl` | the constructed book at each refresh |
| `concordance.jsonl` | held-versus-target divergence per session |
| `unfilled.jsonl` | every order the venue did not print |
| `state.json` | durable loop state — never edit by hand |

## 7. Safety rails and stopping

- **Stop the book:** `touch <run-dir>/KILL_SWITCH`. The loop still settles
  and marks NAV but submits nothing; delete the file to resume.
- **Drawdown halt:** the loop halts below 50% of peak NAV by default
  (`--max-drawdown`); tighten it if you care about the account.
- **Order guards:** per-order notional and count bounds veto a corrupted
  decision before anything is persisted or submitted.
- **Real money is out of scope here.** It requires explicitly setting
  `APCA_API_BASE_URL` to the live endpoint, and the repo's own doctrine is
  that no configuration has cleared the deployment bar (`SPEC.md` §10).
  Whole-share sizing also sets a measured account floor: ~$100k holds the
  ~100-name book faithfully; $10k trades a structurally different portfolio
  (`docs/account_size_floor.md`).

## 8. Where to go next

- `README.md` — program status: what is certified, what is in flight.
- `SPEC.md` — the constitution.
- `docs/operations.md` — operational sharp edges.
- `docs/free_tier_profile.md` — exactly what $0 reproduces.
