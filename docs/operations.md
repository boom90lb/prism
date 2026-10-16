# Operations notes

Durable operational facts for running the pipeline on real data — the
"what will actually bite you" companion to the README. The deployment
gate is `SPEC.md` §6.

## Path to deployment (owner acts; agents cannot fund or arm)

These are the open items between the v0.3.4 checkpoint and an honest
v0.4.0. Record each result durably (ops log or dated section here) before
any deployment claim.

| Step | Owner decision | Evidence to attach |
|---|---|---|
| **Micro-account (A3)** | Fund ≤ $2k of cost-calibration equity (SPEC §10 carve-out) | Cap, funding date, venue account id — no secrets in git |
| **Venue check (i)** | Fractional short acceptance and per-name fractionability | API probe receipt; on failure, whole-share ≥ $100k is the only mode |
| **Venue check (ii)** | Day-order versus auction fill quality from the paper stream | Paper fill telemetry summary |
| **Capital mode** | Whole-share auction ≥ $100k, **or** fractional day orders if both venue checks pass | Chosen mode written down |
| **Gate G0** | ≥ 1 full rebalance cycle of real fills; spread tables recalibrated from them | Fill ledger + recalibrated buckets |
| **Regime clock** | ≥ 21 consecutive clean paper sessions with regime telemetry (`regime.jsonl`); any fail-loud event restarts the count | Session count + date of last event |
| **De-gross arming** | Dedicated commit, only after sizing pins and the regime clock both hold | Commit SHA |
| **First real order** | Named risk profile + a sleeve at its evidence bar + live URL set intentionally | Claim packet with profile id |

Paper is the default: `APCA_API_BASE_URL` unset means the paper endpoint
(`docs/security.md` §2.5). Risk profiles:
`python -m prism.scripts.paper_loop --profile research_paper ...`
(schema frozen — `docs/risk_profile_schema.md`).

Remaining ops hardening called out by the program: boot-resilient nightly
scheduling (the paper stream has already shown sleep/gap risk) and a one-page
deploy runbook covering capital mode × risk profile × venue × custody × kill
switch.

## The paper session: wrappers, health, alerting

The two scheduled entry points are versioned in-tree under `ops/`:

| Script | When | What it does |
|---|---|---|
| `ops/paper_loop_nightly.sh` | weekdays 18:30 PT | one `prism.scripts.paper_loop` cycle, then `prism-doctor --network`; alerts if either goes nonzero |
| `ops/paper_sweep_morning.sh` | weekdays 06:50 PT | one `prism.scripts.paper_sweep`, then offline `prism-doctor`; alerts if either goes nonzero |

Both resolve the live run directory from `runs/ACTIVE_RUN_DIR` (one line,
relative to the repo or absolute) and read `$REPO/.env`. Point the scheduler at these,
not at private copies — the July 2026 outage below was invisible to review
precisely because the wrappers lived only at `~/bin/`.

**Alerting** (`ops/_alert.sh`) fires three independent notifications, because
each one alone has already failed in production:

1. `{run-dir}/ALERT` — a durable marker, carried off-box by the artifact sync.
   A *healthy nightly* clears it; the morning sweep never does, so a
   good-looking morning cannot retire an alert raised by the session that
   trades. A present `ALERT` file always means unresolved.
2. A loud `ALERT:` line in the session log, beside the traceback.
3. A nonzero exit, so the scheduler's own last-result surface goes red.

Escalation is a seam, not a choice: export `PRISM_ALERT_CMD` in `.env` and it
is invoked as `"$PRISM_ALERT_CMD" "<subject>" "<body>"`. The repo does not pick
a transport, and a failing transport never fails the session.

### Exercising the commands without trading

**Every verification of `paper_loop` or `paper_sweep` uses `--no-submit`.** Both
commands otherwise had exactly two states — "fails before the credential check"
and "trades the account" — with nothing between them, and on 2026-09-09 a step
meaning to observe the first performed the second: it ran the command against a
throwaway `--run-dir` to see whether it still parsed, found the credentials in
`.env`, and sent 116 market-on-open orders to the paper account. (They were
cancelled and the account restored; the record is in the incident notes.)

Under `--no-submit` the command does everything a real session does — resolves
credentials, reads the venue's account, positions and cash, marks, scores,
constructs the book, diffs the orders, runs the safety rails — logs each order
it would have sent as a `NO-SUBMIT WOULD SUBMIT #n …` line, and sends none. Its
durable writes go to a discarded scratch copy of the run directory, so a
rehearsal cannot leave a decision persisted for the next real session to
resume, and pointing it at the live run directory changes nothing there. It
closes with an unmissable `NOTHING WAS SENT TO THE VENUE` banner.

**A run that does trade says so on one line**, at WARNING, before the first
order leaves the process:

```
*** SUBMITTING ORDERS TO THE VENUE *** endpoint=… account=… decision_bar=… orders=N — this run is trading: …
```

Grepping `SUBMITTING ORDERS TO THE VENUE` in any session log answers "did this
run trade, and against which account?". A run that offers no order — a
rehearsal, a replay, a resume whose orders the venue already holds — never
prints it.

**Submitting from a run directory with no prior session is refused.** A paper
session is a continuation: its run directory carries `state.json` and
`equity.jsonl` from the sessions before it, and both scheduled wrappers point at
one that does, so the rail is inert for them. An invocation pointed at a
directory that carries neither — the incident's shape — is refused before any
venue read, and must say which case it is: `--no-submit` to rehearse, or
`--allow-new-run-dir` for the deliberate first session of a new book.

**Health is read from the durable record of work done, never from an exit
code.** `prism-doctor` (offline; `--network` adds venue reconciliation) is the
signal:

| Check | Question | Verdict |
|---|---|---|
| `equity-ledger` | how many weekdays since the last completed cycle's NAV mark? | FAIL at ≥ 4, WARN at ≥ 2 |
| `nightly-log` | last wrapper verdict, and is a *successful* one stale? | FAIL on any nonzero, FAIL on a stale success |
| `regime-clock` | consecutive clean sessions on the precondition-(b) clock | WARN until 21 (`docs/regime_step.md` §4) |
| `alpaca-account-book` | does the venue's book belong to this run directory? | FAIL when the venue holds a book this directory has no record of |
| `alpaca-holdings-priceable` | can every held position be marked? | FAIL naming the unpriceable names |

The equity ledger is the load-bearing one: it gains exactly one row per
*completed* cycle, so nothing else in the surface stays honest when the
scheduler, the log, and the sweep all look fine. Session age is counted in
weekdays and is deliberately holiday-blind — a health check that needs a
holiday table breaks in January, and blindness only ever over-counts, which is
why FAIL sits at four.

`twelvedata-quote` is WARN-only by design: the live loop reads Alpaca's own
feed, and an alert that fires on something the live path does not use is an
alert the operator learns to ignore.

### Postmortem: the 2026-07-23 → 07-29 dark nightly

Every scheduled cycle in that window died with
`ValueError: cannot value held position 'POOL': price None (N7)`, plus one
manual retry — six consecutive failures. Nobody was told. The morning sweep
exited 0 on four of those mornings (it had nothing pending, so it had nothing
to fail on), and `prism-doctor` reported "8 pass, 0 fail" throughout.

Root cause, in two layers:

1. **The universe reconciliation read the wrong authority.** The 2026-07-15 fix
   for index leavers fetched *configured universe ∪ persisted held book*. On
   2026-07-23 `ACTIVE_RUN_DIR` was repointed from `runs/paper_loop_momentum2`
   (which persists the account's actual book — 98 positions including
   `POOL -48`) to a fresh `runs/paper_loop_2026-07-23`, which persists nothing.
   The union therefore added nothing, POOL was never fetched, and the mark step
   correctly refused to value it. Persisted state was never a safe authority in
   the first place: it is a cache of what the loop last reconciled, and it can
   be absent (this outage) or stale — the earlier retired
   `runs/paper_loop_momentum` still records 28 positions against the same
   venue's 98.
2. **Nothing was watching.** No check compared the venue's book to the run
   directory, no check read the age of the last NAV mark, and no path turned a
   nonzero nightly exit into a notification.

Both are fixed: `prism.live.daily.resolve_fetch_universe` queries the broker
*before* the fetch universe is decided and treats persisted state as advisory
only; `fetch_universe_panels(required=...)` refuses to let the `max_missing`
tolerance silently drop a held name; the doctor checks above catch the
condition offline; and the `ops/` wrappers alert. Pinned by
`tests/test_paper_loop_universe.py` (including the counterfactual — a panel
missing the held leaver must still die at the mark) and `tests/test_doctor.py`.

### Reattaching a run directory (owner action)

A `FAIL alpaca-account-book` means the venue holds a book this run directory
has no record of. The cycle *can* run through it — broker truth drives
valuation — but the directory's `equity.jsonl` / `regime.jsonl` would then be a
new evidence stream over an old book, and the session and regime clocks would
count sessions whose history lives elsewhere. Choosing is an owner act
(`SPEC.md` §6); an agent must not pick.

The two honest options:

- **Reattach.** Restore the ledgers of the directory that decided the book and
  point `runs/ACTIVE_RUN_DIR` back at it. The July 2026 case is recoverable:
  `runs/paper_loop_momentum2` — `state.json` (98 positions, `POOL -48`,
  `last_refresh 2026-07-13`), `equity.jsonl` (4 sessions: 07-13, 07-14, 07-16,
  07-17), `fills.jsonl`, `targets.jsonl`, `concordance.jsonl`,
  `unfilled.jsonl`, `nightly.log` — survives in the private `prism-artifacts`
  repo in the commit *before* `5b7194e` ("backup: 2026-07-24T01:30Z nightly
  sync"), which is where the local deletion propagated through
  `rsync --delete`. Its state matches the venue's book name for name, so
  reattaching preserves a continuous stream.
- **Start clean.** Flat the account at the venue, then start the new directory
  over a flat book. The paper clocks restart from zero, which is the honest
  price of a discontinuous stream.

Either way, failed sessions cannot be backfilled as live evidence. `rsync
--delete` in `ops/artifacts_sync.sh` (the versioned successor to
`~/bin/prism_artifacts_sync.sh`) means the off-box copy tracks local
deletions: a retired run directory is recoverable only from the backup repo's
*history*, not its worktree.

## Data vendor (Twelve Data)

- The key goes in a gitignored `.env`: `TWELVEDATA_API_KEY=...` (loaded by
  `prism.workspace.load_env` at the command entry point and passed to
  `DataLoader`).
- **Interval strings.** The vendor expects `1day`/`1week`/`1month`; the
  project keeps the short forms (`1d`, `1wk`, `1mo`) everywhere and
  normalizes only at the request boundary (`DataLoader._to_vendor_interval`).
- **The free tier bites.** ~8 requests/min and 800/day — pre-warm the cache
  before a multi-symbol run. The `/dividends` endpoint returns 403 for most
  symbols on this tier, so cross-symbol total return is inconsistent there:
  run backtests with `--no_dividends` for a uniform price-return comparison,
  or supply a higher-tier key.
- **Dividend caching is failure-aware.** A failed fetch is never cached, and
  a genuinely-empty "no dividends" answer expires after 7 days — so a tier
  outage cannot poison later runs and a tier upgrade heals on its own.

## Which ensemble members actually contribute

For the legacy research backtest: `xgboost` is the default forecast member;
`arima` and `prophet` are opt-in diagnostics only; the forecast `lstm` was
removed outright; the RL policy members run if requested but see the
performance note below. For a forecast-only backtest, train with
`--models xgboost`.

## Per-bar prediction cost

The backtest calls `predict` once per bar over the test range (~1,600
bars/symbol on a multi-year run):

- **JAX members re-trace on most bars** (input shape variation defeats the
  compile cache). A 4-symbol forecast+RL backtest can take ~80 minutes,
  almost all of it retracing; the same universe forecast-only runs in
  ~15–20 minutes. Don't pay for members you don't need.
- **statsmodels warns once per bar** and routes some warnings through
  logging, so expect chatty stderr regardless of `PYTHONWARNINGS`. Redirect
  output to a file and skip `--verbose`.

## Reproducing a run from scratch

```bash
echo "TWELVEDATA_API_KEY=..." > .env

python -m research.scripts.training \
  --symbols AAPL,MSFT,GOOG,AMZN \
  --start_date 2020-01-01 --end_date <today> \
  --horizon 5 --n_splits 5 --models xgboost

PYTHONWARNINGS=ignore python -m research.scripts.backtest \
  --training_run runs/<run_name> --no_dividends
```

Participation impact is opt-in (`--adv_impact_coeff`, `--adv_floor_dollars`);
dollar volume comes from the close × volume bars already on disk. Artifacts
land under `runs/` (per-fold models) and `results/` (backtest report); both
are gitignored, as is `logs/`.

## The synthetic-versus-real lesson

Every verification before the first real-data run used synthetic
random-walk prices, which masked three real bugs (vendor interval string,
ensemble member persistence, predict-frame column set). When changing
data/model/backtest plumbing, prefer a real or schema-faithful fixture;
`tests/test_real_pipeline_regressions.py` guards these specific paths.

## 2026-09-23: `--no-submit` rehearsal after the engine overhaul

Run from the repository root against the live run directory, exactly the
nightly wrapper's arguments plus `--no-submit`:

```sh
.venv/bin/python -m prism.scripts.paper_loop --book momentum \
  --universe-file data/universe/sp500_current.txt \
  --run-dir runs/paper_loop_momentum2 --spinoff-mask --regime --no-submit
```

Result (Run, commit `f477e5a` plus an uncommitted research script): exit 0;
panels 752 bars x 502 symbols through 2026-09-23 from Alpaca; state migrated
v2 -> v3 in the scratch copy (99 positions, cash 937,669.47, 0 pending);
equity marked 986,600.94; 7 names held out by the spin-off mask; decision bar
2026-09-23 would submit 119 orders, 627,630.21 notional, gross 1.0102, with 4
dust orders dropped by whole-share rounding; live monitor `inconclusive`
(n = 16). `NOTHING WAS SENT TO THE VENUE`; no `SUBMITTING ORDERS` line; the
run directory's ledgers are unchanged (only `.lock` was touched).

The first real session after re-arming is therefore a full rebalance of a book
last refreshed 2026-08-11, not an incremental one. Re-arming the Windows Task
Scheduler entries for `ops/paper_loop_nightly.sh` (weekdays 18:30 PT) and
`ops/paper_sweep_morning.sh` (weekdays 06:50 PT) remains the owner's act.

## 2026-09-25: why the loop went dark, and a second rehearsal

**Cause.** The scheduler did not fail; both Windows Task Scheduler entries were
disabled. `schtasks /query /v` reads `Status: Disabled` for
`\Prism paper loop (nightly)` and `\Prism paper sweep (morning)`, each with
`Last Run Time` 2026-08-14 and `Last Result 0`. Both task definition files under
`C:\Windows\System32\Tasks\` were last modified 2026-08-15 23:20:18 PDT, 30 ms
apart, the signature of one disable act; the Task Scheduler operational event
log is not enabled on this machine, so who disabled them is not recorded, and no
commit or tracked note from that weekend mentions it. The actions still run
`wsl.exe -d Ubuntu -e bash -c '~/bin/prism_paper_{loop_nightly,sweep_morning}.sh'`,
and both `~/bin` files are symlinks into `ops/`, so re-enabling restores the
versioned wrappers unchanged. Both tasks carry `DisallowStartIfOnBatteries`,
`StartWhenAvailable` and `WakeToRun`.

**Rehearsal** (Run, commit `3a3fca9`, `--no-submit`, the nightly's arguments):
exit 0; panels 753 bars x 502 symbols through 2026-09-24 from Alpaca; state
migrated v2 -> v3 in the scratch copy (99 positions, cash 937,669.47); equity
marked 991,104.81; 7 names held out by the spin-off mask; decision bar
2026-09-24 would submit 123 orders, 685,504.97 notional, gross 1.0101, with 3
orders dropped by whole-share rounding; live monitor `inconclusive` (n = 16).
`NOTHING WAS SENT TO THE VENUE`; no `SUBMITTING ORDERS` line. The health check
afterwards: 15 pass, 1 warn, 3 fail, the three being the stale NAV ledger (30
weekdays), 28 missed sessions and the stale nightly log; the venue's 99
positions reconcile against `state.json`.

**Drift.** `data/universe/sp500_current.txt` is the 2026-07-17 build. Since then
FERG, RDDT, BE, P and ILMN joined the index, EA, AVB, TAP, TTD and BLDR left,
and Equity Residential became Vivmark Residential (EQR -> VMRK, 2026-08-18). The
rehearsed book shorts BLDR and TTD, which are no longer members.

**Re-armed 2026-09-25** (owner decision, same day). Both tasks were enabled
with `schtasks /Change /TN "<name>" /ENABLE`; the first session is the 18:30 PT
nightly of 2026-09-25. Before it, `sp500_current.txt` was rebuilt as of
2026-09-25 (`sp500_membership_2026-09-25.parquet`, revisions recorded in
`sp500_sources_2026-09-25.json`): minus AVB, BLDR, EA, EQR, TAP, TTD; plus BE,
FERG, ILMN, P, RDDT, VMRK; 502 names. A `--no-submit` rehearsal on it exited 0
with 123 orders decided (705,817 notional, gross 1.0102), the held BLDR and TTD
shorts exiting as out-of-universe names; it ran during market hours, so its
decision bar was a partial one. The regime clock now restarts on unrecorded
sessions (docs/regime_step.md §4, `prism.scripts.doctor.regime_streak`), so it
reads 0/21 before this session and 1/21 after a clean one.

## 2026-09-30: the backup was failing silently; off-box alerting

**What was found.** The off-box backup of `runs/`, `logs/` and `results/`
(`~/bin/prism_artifacts_sync.sh`, an operator-local script wired into the
nightly) had not landed a push since 2026-08-14. From the night the loop
restarted, 2026-09-25, every push was refused by GitHub's pre-receive hook:
`results/run-a195851811a34615/record.json` is 128 MB, over the 100 MB blob
limit (GH001), and three more `record.json` files sit at 85–96 MB. The script
logged `PUSH-FAILED (kept local commit)` and exited 0; the nightly wrapper ran
it as `[ -x "$SYNC" ] && "$SYNC"` and discarded even that. So the ledgers had no
off-box copy for six weeks, the health check had no row that could say so, and
the wrapper's exit code was green throughout. Separately, `PRISM_ALERT_CMD` was
never set, so none of the three on-box notifications (marker file, log line,
exit code) could reach anyone, and nothing at all could report a scheduler that
does not fire, which is what happened between 2026-08-15 and 2026-09-25.

**The versioned sync.** `ops/artifacts_sync.sh` replaces the `~/bin` script and
is the nightly's default (`PRISM_ARTIFACT_SYNC` still overrides; a
non-executable path still disables). It mirrors `runs/` and `logs/` exactly
(`rsync --delete`, as before), mirrors `results/` additively up to
`PRISM_SYNC_MAX_MB` per file (default 45 MB), and ships every larger results
file as a gzip copy at the same path plus `.gz` (a 134 MB `record.json`
compresses to a few MB). Raw oversize copies the old script left in the backup
worktree are removed; each is derivable from the source. It commits only when
the tree changed, pushes every run (so commits left behind by an earlier refused
push land on the next quiet night), and exits nonzero on any failure. After
every attempt it writes `{run-dir}/sync_status.json` in this checkout, one JSON
object: `at`, `ok`, `stage` (`repo`, `rsync`, `commit`, `push`, `done`), a
one-line `detail`, `head`, `pushed`, `compressed`. Its log is
`{run-dir}/sync.log`. Neither file is mirrored, so the backup's content is a
function of the source and "no change" means no change. A `results/`-stage
error (an unreadable file, a failed gzip) is reported only after `runs/` and
`logs/` have been committed and pushed, so `ok: false, stage: "rsync"` with
`pushed: true` means the ledgers landed and only the `results/` copy is
incomplete. When a results file grows past the cap after being mirrored raw,
the raw copy is removed from the backup (logged `DROPPED stale raw copy`) and
the `.gz` is the current one.

The wrapper now logs `SYNC EXIT <rc>`, alerts on a nonzero sync (`prism
artifact backup failed`, pointing at the status file and the log), and folds the
code into its exit: loop result first, then doctor, then sync. The health check
gained three rows: `artifact-sync` (FAIL when the last report says `ok: false`,
WARN when none exists yet or the last success is more than one weekday old),
`alert-transport` (WARN when neither `PRISM_ALERT_CMD` nor `PRISM_HEARTBEAT_URL`
is set; names only, never values), and `universe-file` now reads the `as of`
date in the file's header and WARNs past 45 days, since the 2026-07-17 build
traded on 2026-09-25 holding shorts in two names that had left the index.

**Off-box alerting.** Two independent paths, both operator-configured in `.env`:

- *Heartbeat (dead-man's monitor).* Each wrapper ends by pinging
  `<url>/<final exit code>` (`curl -fsS -m 10 --retry 2`): `/0` is success,
  `/<nonzero>` reports a failure at once rather than after the grace period
  (the healthchecks.io convention). `PRISM_HEARTBEAT_URL` is the nightly's
  monitor, `PRISM_HEARTBEAT_URL_SWEEP` the morning sweep's. The monitor expects
  a ping on each weekday schedule and alarms when none arrives, which is the
  only mechanism here that can report a scheduler that never ran. A failed ping
  is logged as `HEARTBEAT-FAILED` and never changes the session's exit code.
- *Alert transport.* `ops/alert_curl.sh` is a minimal transport for the
  `PRISM_ALERT_CMD` seam: it POSTs the alert body to `PRISM_ALERT_URL` with the
  subject as the `Title` header (ntfy-compatible), exiting nonzero if the URL is
  unset or the request fails, which the wrappers log as
  `ALERT-TRANSPORT-FAILED` and otherwise ignore.

Creating the two monitors and the notification topic, and setting
`PRISM_HEARTBEAT_URL`, `PRISM_HEARTBEAT_URL_SWEEP`, `PRISM_ALERT_CMD`
(`<repo>/ops/alert_curl.sh`, with `<repo>` the checkout's absolute path) and `PRISM_ALERT_URL` in `.env`, are the
operator's acts; until then the `alert-transport` row warns on every health
read. The ping URLs and the topic are secrets in the same sense the API keys
are: whoever holds them can keep the monitor green or post to the topic. They
are never written to a log.

**One-time repair of the backup repository (operator's act).** The backup
clone at `~/prism-artifacts` holds unpushed local commits whose trees contain
the oversize blobs, so the remote will keep refusing until they are gone. The
new script does not rewrite history. As of this writing `git log
origin/main..HEAD --format=%s` there lists exactly four commits, all the old
script's own nightly backups (2026-09-26, -29, -30 and 10-01 UTC), each
re-derivable from the source tree; nothing else was committed locally. The
repair is to drop those commits while keeping the worktree, then let the new
script make one clean commit:

```bash
cd ~/prism-artifacts && git log origin/main..HEAD --format=%s   # confirm: only "backup: … nightly sync" rows
git -C ~/prism-artifacts reset --soft origin/main
"$REPO"/ops/artifacts_sync.sh && cat "$REPO"/runs/paper_loop_momentum2/sync_status.json   # REPO: the live checkout
```

This `reset --soft` repair applies only when the status `detail` says the
oversize blob is in unpushed history. When it instead names a file in the
current tree (`runs/` and `logs/` are mirrored without a size cap), that file
must be removed or capped at the source first; rewriting history would not
clear it.

The old script's tracked `sync.log` at the backup repo's root is no longer
written; `git -C ~/prism-artifacts rm -q sync.log` retires it (its content
stays in history) and the next sync commits the removal. Then repoint nothing:
the nightly already calls `ops/artifacts_sync.sh`. Retire
`~/bin/prism_artifacts_sync.sh` once the first `SYNCED` line appears in
`{run-dir}/sync.log`.

## 2026-09-30: trading is opt-in and bound to the active run directory

Three new refusals sit in front of every order, each one a condition that an
earlier session actually violated. None of them can fire on the scheduled
sessions: the wrappers pass the flag, point at the directory the switch names,
and run after the close.

**No mode flag.** `prism.scripts.paper_loop` and `prism.scripts.paper_sweep`
now require either `--submit` or `--no-submit` and exit 2 with a one-paragraph
message when given neither. Until today a bare invocation that found
credentials in `.env` traded (the 2026-09-09 incident was exactly that), and
`--no-submit` was the only thing standing between parsing the arguments and
sending orders. `--allow-new-run-dir` is accepted with `--submit` only; on a
rehearsal it is refused. Both wrappers pass `--submit`.

**Inactive run directory.** A submitting run must point at the directory
`runs/ACTIVE_RUN_DIR` names (`prism.live.safety.require_active_run_dir`,
checked right after the established-run-directory rail and before the first
venue read). The pointer's content is a path relative to the repository root or
absolute; both sides are resolved before comparison. A missing pointer or a
mismatch refuses, naming both paths and how to point the switch. There is no
flag past it: a new book starts by writing its directory to
`runs/ACTIVE_RUN_DIR` first, which the scheduler needs anyway. The old rail
asked whether *a* session had run in the directory; ten retired directories
under `runs/` pass that question. The wrappers themselves no longer fall back
to the retired `runs/paper_loop_momentum` when the pointer is missing or empty:
they log `FAIL`, raise the alert at `runs/ALERT` (there is no run directory to
put it in), send the heartbeat with code 4, and exit 4. The shell accepts the
same two forms of pointer the Python rail does, repo-relative or absolute, and
puts its log, status and `ALERT` files in the directory it names. `prism-doctor`
run by hand without `--run-dir` follows the same rule: it refuses when the
switch is missing or empty instead of checking a retired or current-directory
book.

**Partial bar.** `prism.live.daily.require_published_panel` runs after the
stale-panel warning. The panel's last bar must be dated on an exchange session,
and when the clock is on that date it must be at least 30 minutes past that
session's close (`prism.io.sessions.session_close`, which knows the 13:00 early
closes). A bar dated after the clock is refused as a feed or clock defect. The
bar fetch asks for no end date, so a catch-up run during market hours receives
today's row at its last print; the 2026-09-25 rehearsal decided on one and the
stale-panel guard could not see it, because a bar from this morning is zero
days old. A submitting run is refused (`PartialBarError`, a `SafetyViolation`);
a rehearsal is allowed through with a `PARTIAL BAR` warning so a daytime
`--no-submit` still exercises the command. The 18:30 PT nightly is 21:30 ET,
five hours past the latest possible publication, and a holiday or pre-open run
holds the previous session's completed bar, which passes at any hour.

**Run-scoped client order ids.** New decisions carry
`{book}{tag}:{decision_bar}:{symbol}` where `tag` is the first eight hex
characters of the SHA-256 of the run directory's basename
(`prism.scripts.paper_books.scoped_order_id_prefix`; the live book's prefix is
`mom:faf4b45b:`). The basename rather than the path, so a relocated or restored
checkout keeps producing the same ids; distinct per run directory, so two
directories trading one book can no longer submit ids the venue treats as each
other's duplicates. The longest id the scheme can produce (a `:S1` sweep child
of a five-character symbol) is 32 characters against the venue's limit of 48.
Nothing parses the prefix: the sweep appends its suffix to the persisted parent
id, the replay reader matches children to parents by `rpartition`, and the
parity gate compares orders by symbol. Orders recorded under the bare `mom:`
prefix on 2026-09-25 keep their ids and are swept and settled as before; the
sweep sets no prefix of its own. The live book holds no pending orders and its
next refresh is on or about 2026-10-26, so no order carries the new scheme
before then, and the two generations never meet in one pending set.

**The per-session panel record.** Since 2026-09-30 every cycle writes
the panel it scored to `runs/<run>/panels/`: a base parquet on the first
session, one delta parquet per session after it, and `index.jsonl` with one row
per record (decision bar, kind, file, SHA-256, chain position). Each refresh row
in `targets.jsonl` carries `panel_sha256` and `panel_file`. A rerun on a bar
that already has a record never makes a new decision (the write-ahead protocol
resumes or holds), so when its panel differs from the recorded one the cycle
keeps the recorded panel, logs both digests at ERROR, and proceeds; the fetch
window is anchored to the last published session rather than the wall clock
(`paper_books.default_panel_start`), so such a mismatch means the vendor
revised history, not that the window slid overnight. A conflicting panel for a
bar with no persisted decision still refuses (`PanelLogConflict`). To verify a
session's record without touching the run directory, open a Python shell from
the repository root:

```python
from prism.live.panel_log import latest_record, reconstruct_panel
record = latest_record("runs/paper_loop_momentum2")          # the newest index row, or None
panel = reconstruct_panel("runs/paper_loop_momentum2", record.decision_bar)
panel["close"].tail()                                        # exactly the frame the cycle scored
```

`reconstruct_panel` replays the chain from the base through every delta, checks
each file's digest against the index on the way, and raises on any mismatch.
Both functions only read.

## 2026-09-30: the live monitor reads two statistics; the panel record has a health row

**What the nightly logs.** After the NAV mark, `monitor <bar>: verdict=… n=…
mean=… ci=[…] method=empirical_bernstein gaps=…` is the return monitor over
consecutive-session NAV returns (a gap between ledger rows is counted, never
averaged in as one day). Read off the ledger on 2026-09-30, after that evening's
session (the last to run the previous all-pairs monitor): 20 NAV rows, 19 adjacent pairs, 3 gaps (a missed 2026-07-15, the
2026-07-17 → 07-30 break, the 2026-08-14 → 09-25 outage), n = 16, interval
±0.062 a day against the old method's ±0.25 on the same rows. `tracking <bar>: verdict=… n=… mean_diff=…
ci=[…] corr=… slope=… transitions=… unmatched=…` is the paper-minus-shadow read.
Both are registered in `docs/live_monitor_design.md`; neither gates an order,
and a failure in either is logged as `LIVE MONITOR FAILED` or `LIVE MONITOR:
SHADOW RECORD FAILED` with the verdict `unavailable`, never raised into the
cycle. `python -m prism.scripts.paper_monitor` prints the return read;
`--method normal_mixture` is not wired there yet (call
`prism.live.monitor.paper_monitor_read(..., method="normal_mixture")` from a
shell for the comparison).

**The shadow ledger.** `{run-dir}/shadow.jsonl` gains one row per session the
cycle can shadow: the previous session must be in the ledger and be the
preceding exchange session, and a refresh book must predate it. The row carries
the decided book's close-to-close return on the panel the cycle scored, the
paper return over the same pair, whether the session is a transition (the one
after a refresh bar), and how many baseline names were priced. The first row can
land on 2026-10-01 (previous session 2026-09-30 in the ledger and the preceding
exchange session; baseline the 2026-09-25 refresh, which predates it), provided
the fetched panel prices at least 90% of that book's gross on both bars. Each row
is recomputable from that session's `panels/` record and `targets.jsonl`.

**The `panel-record` health row.** PASS when `panels/index.jsonl`'s newest
decision bar is the equity ledger's last; WARN while no panel has been recorded
(the log starts with the first session after this change) or when a panel is
ahead of the ledger (a cycle recorded its inputs and did not complete); FAIL
when the record lags the ledger by a session or more, because those decisions
cannot be re-derived from their inputs.

**The `shadow-record` health row.** PASS when `shadow.jsonl`'s newest decision
bar is the equity ledger's last (the detail carries that session's paper minus
shadow return); WARN while no row has been written, or when none was owed for
the last session because the previous ledger bar is not the preceding exchange
session or no refresh book predates it; FAIL when a row was owed and is missing,
since the shadow step is shielded from the trading cycle and would otherwise
fail every session with nothing but a log line to show for it, leaving the
tracking read empty.

## 2026-10-16: a seeded rebalance phase for new paper deployments (off by default)

A monthly book's first session used to set its rebalance phase, so every
deployment of this repository that started on the same day rebalanced on the
same days and sent its orders into the same opening auctions.
`python -m prism.scripts.paper_loop --rebalance-phase seeded` instead puts the
book on a fixed session grid (`prism.io.sessions.grid_position`, session 0 is
2006-01-03) and refreshes it where `(grid position − offset) mod
decision_every = 0`. The offset is drawn from a random seed that belongs to the
run directory.

- **The seed.** `{run-dir}/deployment.json`, mode 0600, written once by the
  first submitting session of a new run directory (`--submit
  --allow-new-run-dir`) and never overwritten. Every later session and every
  `--no-submit` rehearsal reads it. The offset is HMAC-SHA256 of
  `rebalance_phase:<created_bar>` keyed by the seed, reduced to
  `[0, decision_every)` (`prism.live.deployment`). The seed's job is to make
  phases independent across deployments, not to keep a deployment's schedule
  secret: the offset it sets is written in plain text to the nightly log, to
  `state.json` (each tranche's grid offset), to the `tranches` record of
  `targets.jsonl` and to `sizing.jsonl`, and anyone who sees the account's
  orders sees it from the first refresh. The 0600 mode keeps the seed itself,
  and so every later draw it would make (none is used yet), to the owner. The off-box backup
  (`ops/artifacts_sync.sh`) mirrors `runs/`, so the seed is in the backup as
  well, and a restored run directory keeps its phase.
- **Refusals.** A seeded run with no seed file is refused before any venue read:
  a rehearsal of a new directory, or an established directory whose first
  session ran without the flag. Seeded is also refused with `--profile
  research_paper` and with a one-session cadence.
- **Behaviour.** A new seeded book places no orders until its first grid date,
  up to `decision_every − 1` sessions away, then builds in one session. A grid date missed
  to an outage is caught up once, on the first session after more than
  `decision_every` have elapsed, and the book is back on its grid at the next
  grid date. The cadence is `prism.composition.due_on_schedule`, the predicate
  the evaluator uses for grid-scheduled components.
- **Not armed.** `ops/paper_loop_nightly.sh` does not pass the flag, and the
  live book's schedule is unchanged. The default, `first-session`, decides
  exactly as before. Limits: the session table ends 2027-12-31, after which a
  seeded run refuses to decide; and the replay harness (`prism.live.replay`)
  does not take an offset yet, so it replays a seeded run directory on the
  first-session schedule.

## 2026-10-16: a volatility scale for the momentum paper book (off by default)

`python -m prism.scripts.paper_loop --book momentum --vol-scale bsc-12-126-2
--max-gross 2` sizes the decile momentum book by the Barroso and Santa-Clara
constant-volatility rule that selection set `vol_momentum_v1` registered: at
each refresh the scale is `min(2, 0.12 / sigma)`, where sigma is the annualized
standard deviation of the last 126 daily returns of the unscaled book, and the
account target is that scale times the unscaled book. The scale is held until
the next refresh. Code: `prism.live.vol_sizing` (the live seam) on top of
`prism.portfolio.vol_scale` (the rule).

- **Which returns.** Not the paper account's own: those are scaled, rounded to
  whole shares and charged costs. At every refresh the unscaled book is
  rebuilt from the panel the cycle already holds: the signal is scored at each
  past refresh date of the book's schedule on the panel truncated at that date,
  the same decile construct turns each score into weights, and the weights are
  held from one refresh date to the next, close to close, gross of costs. The
  schedule is the book's own: the seeded grid dates under `--rebalance-phase
  seeded`, otherwise every `decision_every`-th bar back from the refresh being
  decided.
- **History.** The rebuild needs the signal's history before the oldest
  refresh date the window reaches: 253 + 126 + 21 − 1 = 399 bars for 12−1
  momentum at the monthly cadence. A shorter panel is refused before any state
  moves; the loop's default three-year fetch holds about 750.
- **Failure path.** When the window cannot be trusted — the panel's last bar is
  not the decision bar, fewer than 126 finite returns, more than 5% of them
  exactly zero (stale or flat bars), or a degenerate estimate — the scale is
  `min(1, last good scale)`, or 1 with none, and the cycle logs a `FALLBACK`
  warning. A failure therefore never sizes the book above the unscaled book.
  The last good scale is read from the refresh rows of `targets.jsonl`, each of
  which carries the scale decision it was persisted with under `vol_scale`, so
  a decision the order guard refused never bounds a later one.
- **Record.** Each refresh appends one row to `{run-dir}/sizing.jsonl` before
  the write-ahead save: the rule, the scale, sigma, the status and reason, the
  base book's refresh dates, the unscaled and target gross, the long side over
  equity and the cash-sleeve fraction `max(0, 0.98 − L/E)`. A same-bar re-run
  that decides again finds its own row and writes nothing; a row that disagrees
  raises before anything is persisted.
- **Refusals.** Only `--book momentum`; `--max-gross` must be 2.0, the rule's
  gross ceiling; refused under every `--profile`, since each pins gross at or
  below 1.0 and profiles only tighten. In `run_daily_cycle` it also refuses a
  regime gross-scale hook (which has no flag), a signal with fitted state, and a
  context without a targets ledger. Regime telemetry (`--regime`) still reads
  every session beside it.
- **Regime hook guard.** Independently of the scale, `run_daily_cycle` now
  refuses the regime gross-scale hook whenever `max_gross > 1`: the hook
  multiplies the configured ceiling by a factor in [0, 1], so above 1 a
  de-gross could size a book above the unscaled one. The hook has no CLI path,
  so the live book is unaffected.
- **Not armed.** `ops/paper_loop_nightly.sh` does not pass the flag, and the
  live book is unchanged; off, the cycle decides exactly as before. Arming it,
  and changing the live book, are owner acts (SPEC §6). Limits: the spin-off
  mask applies to the decision being made but not to the rebuilt past books, so
  a corporate action the bars do not adjust for enters sigma as a return (a
  jump of that kind almost always raises sigma and lowers the scale); and the
  replay harness (`prism.live.replay`) does not take a scale, so it replays a
  scaled run directory unscaled.

## 2026-10-16: a tranched momentum paper book (off by default)

`python -m prism.scripts.paper_loop --book momentum --tranches 21` splits the
decile momentum book into 21 overlapping tranches. Each holds 1/21 of capital
and refreshes every 21 sessions on its own date of the fixed session grid
(tranche *j* where `(grid position − base − j) mod 21 = 0`), so one tranche
refreshes on every session. The base is the seeded offset under
`--rebalance-phase seeded`, otherwise 0; with 21 tranches it only relabels
them. `--tranches 3` gives tranches 7 sessions apart. Code:
`prism.live.tranches`, composed by `prism.composition.compose` with
`tranche_plans` and `due_on_grid`, the schedule the evaluator uses.

- **Held in shares.** Each tranche is held in shares between its refreshes, as
  the registered evaluation holds it (`prism.evaluate._TrancheHoldings`,
  `docs/public_baseline_design.md` §1.1). At its refresh a tranche's holding is
  reset to 1/21 × its book (times its volatility scale) × equity / close shares
  of each name; until its next refresh the shares are held and drift with
  prices. A session decides only the names the refreshing tranche held or now
  holds, plus any name whose account shares differ from the tranches' total by
  more than half a share (an unfilled order or a position no tranche holds);
  every other name is held. So a session's orders are the refreshing tranche's
  change and those catch-ups, never a rebalance of the drifted book back to
  constant weights. Each tranche's holding is persisted with the session and
  close it was sized at, and a split since then is restated from the panel
  (the live bars are split-adjusted, so the close the panel now shows for that
  session differs from the recorded one by the split ratio).

- **Staged entry.** A new tranched book starts empty and builds each tranche on
  its own grid date, so it reaches full size over 21 sessions; it never buys the
  whole book on its first day. It must start in its own run directory on a flat
  account: a run directory that holds an untranched book, or an account that
  holds positions no tranche holds (the T-bill holding aside), is refused before
  any state moves, because the first session would sell all but 1/21 of what
  is held.
- **State.** Each tranche's schedule anchor, sub-targets, holding in shares and
  grid offset (and, with the volatility scale, its scale decision) are
  `LoopState.components` in `state.json`, saved in the same write-ahead step as
  the decision, so a restart in the middle of the build resumes the same
  schedule. No schema bump: the three new component fields are written only
  when set. A built tranche state without share holdings (written before
  tranches were held in shares) is refused. A run directory
  whose persisted layout (count, ids, offsets) differs from the flags is
  refused, and `--tranches 1` (the default) refuses a run directory that holds
  tranches, so a forgotten flag fails rather than holding the book for a month.
- **Outages.** A tranche whose grid date was missed catches up once on the
  first session back and then returns to its grid, so after an outage of *n*
  sessions the return session refreshes *n* + 1 tranches; only then do trades
  net across tranches. A tranche output is valid for 42 sessions; after a longer
  outage every tranche is expired and rebuilt at once.
- **Whole-book refreshes and deferred entries.** A tranche's refresh is its
  whole book: a name it held that is no longer scored is set flat in it. A name
  the refreshing tranche targets that has no price at the close keeps the
  tranche's previous holding of it, or, if it held none, is not entered until
  that tranche's next refresh; it is logged and listed under `deferred` in the
  session record, instead of failing the sizing step. A name the spin-off mask
  holds out of ranking keeps the refreshing tranche's previous holding and is
  never entered.
- **Wind-down.** `--wind-down` with the same `--tranches` sets each tranche flat
  on its own date, so the book exits over 21 sessions; the spin-off mask is not
  consulted and the scale is not decided while winding down. The kill switch
  still stops everything at once.
- **With `--vol-scale bsc-12-126-2 --max-gross 2`.** Each tranche is scaled at
  its own refresh by the rule applied to the tranched base book rebuilt point in
  time from the panel: the mean of all 21 tranche books as last refreshed,
  `Σ D_j / 21` at raw gross 1.0. A tranche's sub-targets are its book times its
  scale, capped at the gross ceiling. Because held shares drift, the account
  target the holdings make is capped at the ceiling as well; on a session where
  that cap binds, every tranche's holding is scaled by the same factor and every
  name is decided (`ceiling_factor` in the session record). A failed estimate
  gives `min(1, that
  tranche's last good scale)`, read from its persisted state. `sizing.jsonl` gets
  one row per session (`schedule = "tranches"`, per-tranche decisions under
  `tranches`). The rebuild scores the signal at about 146 past dates per session
  (about 0.2 s each for 12−1 momentum on 500 names, so roughly half a minute).
- **Telemetry.** Every refresh session appends a `targets.jsonl` row: the
  account target the tranches' holdings make, before whole-share rounding, and
  a `tranches` record (count, base offset, due tranches, tranches built, names
  decided, names caught up, ceiling factor). The
  concordance read is skipped for a tranched book (its baseline would be
  yesterday's target, a fill-rate read). The shadow tracking read still runs,
  but with a refresh every session every row is a transition, so it carries no
  tracking information for a 21-tranche book.
- **Not armed.** `ops/paper_loop_nightly.sh` does not pass the flag, and the
  live book is unchanged; with `--tranches 1` the cycle decides exactly as before
  (checked byte for byte against the previous code on synthetic replays).
  Refused with `--profile research_paper`, with a book other than momentum, and
  with a cadence the count does not divide. Arming it, and changing the live
  book, are owner acts (SPEC §6). The replay harness takes `tranches` on the
  epoch grid (base 0); it takes no seeded phase. `prism-doctor` checks the
  tranche states (`tranche-state`, below).

## 2026-10-16: a T-bill ETF holding for idle equity and a buying-power check (off by default)

`python -m prism.scripts.paper_loop --book momentum --cash-sleeve SGOV` holds the
equity the stock long side does not use, less a 2% cash buffer, in a T-bill ETF
beside the decile book. The target is `max(0, 1 − buffer − L)` of equity, where
`L` is the stock long side of the book the cycle decided (after the caps, the
band and the participation gate). Short-sale proceeds are collateral and are not
invested; only idle own equity is. At the recorded mean volatility scale (long
side about 0.54 of equity) the holding is about 0.44 of equity; at a scale of
1.96 or more it is zero. Code: `prism.live.cash_holding`, applied in
`prism.live.daily.run_daily_cycle(cash_holding=...)`.

- **Not ranked.** The ETF is fetched and priced every session whether held or
  not (`resolve_fetch_universe(..., unscored=...)`, and a held position must
  price), and the momentum membership mask keeps it out of the ranked
  cross-section, so the stock book is the one decided without it. A symbol in
  the universe file is refused.
- **When it trades.** On every session that is not halted, refresh or not: the
  holding is left alone while it is within the band (default 0.02 of equity) of
  its target and otherwise moved to the target. A target of zero sells any
  holding, even inside the band, because the ETF would then be held on a margin
  debit. `--wind-down` sets the target to zero, so the holding is sold with the
  tranches.
- **Order size.** One session's change is at most the per-order bound
  (`--max-order-fraction`, default 2 × the per-name cap = 20% of equity) less
  one share, so the order passes that guard after whole-share rounding. A 0.44
  holding is therefore built over three sessions (0.199, 0.199, then the rest
  on 100,000 at $100 a share). During a 21-tranche staged entry the long side
  is small at first, so the holding is built toward a larger target and then
  trimmed as the tranches fill. An account where one share exceeds the bound is
  refused.
- **Record.** `DailyCycleResult.cash_holding` and one log line every session,
  and one row per decided session in `{run-dir}/cash_holding.jsonl` (decision
  bar, long side, held weight, target, applied weight, step bound, action
  `hold` / `rebalance` / `bounded`), written once the decision is persisted.
  Each refresh row of `targets.jsonl` also carries the decision under
  `cash_holding`, but for a monthly book that is once a cycle while the holding
  trades on other sessions too, so the doctor's drift check and the T-bill
  monitor read `cash_holding.jsonl`.
- **Stopping it.** Pass `--cash-sleeve-buffer 1` (target zero) until the ETF is
  sold, or run `--wind-down`. Simply removing `--cash-sleeve` while the ETF is
  held makes the next refresh exit it in one order, which the per-order bound
  refuses: the session stops before anything is persisted.
- **Distributions.** Alpaca paper accounts do not credit dividends, so the paper
  NAV omits the ETF's monthly distributions. Their histories are in
  `data/distributions/SGOV.csv` and `BIL.csv` (issuer files captured
  2026-10-16, provenance beside each), for forward reads to add back.

`--buying-power-check` (default off) reads the venue's buying power before a
new decision is persisted. When the orders that open or extend positions need
more notional than that, every opening quantity is scaled down by one factor
(rounded down to whole shares) and a `BUYING POWER` warning names the
shortfall; closing orders are unchanged, and a batch that fits is unchanged.
Alpaca states buying power as notional (twice the excess equity in a 2x
margin account), so this is the same test as the 50% initial requirement fitting
the excess. For a 4x account Alpaca reports intraday buying power, which can
exceed what the account may hold overnight; the check does not see that case.
Code: `prism.live.buying_power`, called from `decide_and_submit`.

Neither is armed. `ops/paper_loop_nightly.sh` passes neither flag, and with both
off the cycle decides exactly as before (tests compare every result field and
ledger byte). The holding is refused under any `--profile` and with a book
other than momentum. Holding the ETF changes the live book and is an owner act
(SPEC §6). The replay harness takes neither. `prism-doctor` checks the
holding's drift (`cash-holding-drift`, below).

## 2026-10-16: an account margin report and doctor warnings for the opt-in features

Read-only telemetry; nothing on the order path reads it, and nothing acts on
it.

- **The margin read.** `AlpacaBroker.account_margin()` reads `GET /v2/account`
  once and returns equity, cash, the long and short market values (the short
  side as a magnitude: Alpaca defines equity as cash + long + short, so it
  reports the short side negative), the Reg T initial and the maintenance
  requirement as the venue computes them, Reg T buying power, buying power and
  the multiplier, plus the previous-session snapshot fields when present. A
  missing or non-numeric required field raises. The deprecated
  `portfolio_value` is not read. Code: `prism.live.margin`, `prism.live.alpaca`.
  The field list follows Alpaca's documented account object (checked
  2026-10-16); whether a paper account fills every field has not been checked
  against a live response.
- **The report.** `python -m prism.scripts.margin_report` prints the read:
  both sides over equity, gross over equity, the excess over each requirement,
  and the cushion `(equity − maintenance margin) / equity`. `--json` prints the
  row. `--run-dir DIR --record` appends the row to `DIR/margin.jsonl` under the
  run directory's writer lock (so never during a loop session; a held lock
  refuses the record with exit 1), keyed by the read's UTC timestamp, with the
  New York date beside it; a directory that does not exist is refused before
  any read. It sends no order and calls no other endpoint. It is the place to
  check the broker facts the cash-holding estimates rest on: the requirement
  with shorts and the ETF held, the cushion at the gross the book actually
  runs, and, read beside the account statement, whether short proceeds offset
  a margin debit (`cash` includes the proceeds; the interest charged is only on
  the statement).
- **Not scheduled.** The nightly wrapper does not run the report, so
  `margin.jsonl` is written only when someone runs it. A daily record needs a
  line such as `"$PY" -m prism.scripts.margin_report --run-dir "$RUN_DIR"
  --record` after the loop in a wrapper variant; adding it to
  `ops/paper_loop_nightly.sh` is an owner act, with the other arming flags.
  The nightly's `prism-doctor --network` reads the cushion live every session
  regardless (`alpaca-margin-cushion`, one more free account read).
- **Doctor warnings.** Five offline checks, each a PASS when its feature has
  left no record in the run directory, and otherwise a WARN, never a FAIL.
  Promoting any of them to a failure is the owner's decision.
  - `margin-cushion`: the newest `margin.jsonl` row's cushion below 25%, or a
    read two or more weekdays old. With no record it passes, unless
    `sizing.jsonl`, `cash_holding.jsonl` or a `cash_holding` record in
    `targets.jsonl` shows a scaled book or a T-bill holding, which draw on
    margin. The 25% line is a
    monitoring choice: an unscaled long/short book at FINRA's minimums (25% of
    the long side, 30% of the short) sits near 72%, the same book at gross 2
    near 45%.
  - `vol-scale-fallback`: two or more consecutive `sizing.jsonl` rows whose
    scale fell back to `min(1, last good scale)`; a tranched row counts when any
    due tranche fell back, and a warm-up is not a fallback.
  - `cash-holding-drift`: after the newest decision in `cash_holding.jsonl`
    the holding is more than its band from target (it is being built or sold
    within the per-order bound, expected for a few sessions after a start or a
    large change in the long side), or, when the previous record is from the
    previous session, the holding marked now is more than the band away from
    what that session decided (an order that did not fill). The record is
    written on every session the holding is decided, refresh or not.
  - `tranche-state`: tranche states the cycle could not have written: ids
    other than `decile_neutral/t00..tK-1`, grid offsets that are not
    `(base + j·s) mod (s·K)`, non-finite sub-targets, a tranche holding a
    sub-target without share holdings, holdings sized after the last refresh,
    an invalid last good scale, output bars after the last refresh, expiry
    before output, other
    components mixed in, or a count or base offset that disagrees with the
    newest tranche record in `targets.jsonl`. The cadence is inferred from the
    offsets, since it is not persisted.
  - `deployment-profile`: `deployment.json` unreadable or readable by group or
    others; or absent while the state holds tranches or `sizing.jsonl` shows
    refreshes on the seeded grid. A nonzero tranche base offset or a seeded
    grid means the next session with the same flags will be refused; a base of
    0 can be an unseeded book on the epoch grid every unseeded deployment
    shares. A seeded single book without the volatility scale leaves no other
    record of its mode, so its missing profile is not seen here; the loop
    refuses that run before any venue read. The seed is never printed.
- **Acting on the cushion** (selling the T-bill holding first when the cushion
  is thin) is not implemented; the doctor says so in its warning.

## 2026-10-16: a report-only live-monitor read against the T-bill rate (off by default)

`python -m prism.scripts.paper_loop ... --tbill-monitor FILE` adds a second
return read to the live monitor. It is the same empirical-Bernstein sequence
over consecutive-session NAV returns (bound 0.1, α 0.05), read against the
per-session bill rate instead of zero (`prism.live.monitor.tbill_monitor_read`).
It is logged beside the registered read and recorded under
`monitor_read["tbill"]`. The registered hurdle-0 read is computed exactly as
before, and every registered "not in contradiction" conjunct still reads it
(`docs/live_monitor_design.md` §4). The new read gates nothing.

- **The hurdle.** SPEC §3 through `prism.validation.hurdle.tbill_hurdle`:
  `rf − (rf − fee)·s − y·max(0, 1 − L/E − s)`, divided by 252. `L/E` is the
  stock long side and `s` the T-bill ETF's share of equity of the book held
  through each session (the latest `targets.jsonl` row decided at or before the
  previous close), averaged over the sessions read. With no ETF and `y = 0` it
  is the bill rate.
- **The inputs.** `FILE` is a TOML file of `TbillReference` fields, each number
  with its source (`examples/reference/monitor/tbill_reference.toml`). An
  unknown key is refused, and so is a file without `rf_annual` and `rf_source`.
  The rate goes stale; the file is the operator's to keep current.
- **What the paper NAV leaves out.** `y` (default 0) is idle-cash interest the
  NAV does not record. Whether Alpaca paper accounts credit any is unverified,
  and entering a yield the NAV already carries would count it twice. Paper
  accounts do not simulate dividends, so with `--cash-sleeve` the ETF's
  distributions are missing from the NAV; the read credits the ETF's share at
  `rf − fee` (fee 10 bp by default), an assumption about the ETF's distribution
  yield. Borrow fees, which paper accounts do not charge either, are not
  added: this read is telemetry, and the registered forward read of a T21
  paper stream (`research/scripts/public_baseline_forward.py`) makes both
  corrections.
- **The detection horizon.** With `planning_sharpe` (and `planning_source`) in
  the file, the read states `(z_0.95 / S)²` years: the horizon at which the
  expected t-statistic of a stream whose true Sharpe over bills is `S` reaches
  the one-sided 5% critical value. That is about 30 years at 0.30 and about
  245 at 0.105, and the anytime sequence needs longer. It is a planning
  figure, not a measurement.
- **Failure.** A read that cannot be computed (a non-zero `y` with no targets
  ledger, an ETF share above the idle equity, an unreadable ledger) is logged
  at WARNING and recorded as `unavailable`; the cycle and the registered read
  are unaffected.
- **Not armed.** The nightly wrapper does not pass the flag; adding it is an
  owner act.
