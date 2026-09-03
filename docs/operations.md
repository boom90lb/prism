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
relative to the repo) and read `$REPO/.env`. Point the scheduler at these,
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
--delete` in `prism_artifacts_sync.sh` means the off-box copy tracks local
deletions: a retired run directory is recoverable only from the backup repo's
*history*, not its worktree.

## Data vendor (Twelve Data)

- The key goes in a gitignored `.env`: `TWELVEDATA_API_KEY=...`
  (loaded by `src/prism/config.py`).
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
