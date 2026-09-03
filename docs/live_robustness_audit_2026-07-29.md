# Live-path robustness & structure audit — 2026-07-29

Scope: the trading closure — `src/prism/live/*`, `src/prism/scripts/{paper_loop,paper_sweep,doctor,replay_loop}.py`,
`ops/*.sh`. Read in full at 697c9cc. Findings first (complete set), remediation map second
(AGENTS.md §2: discovery and remediation are separate passes).

## Findings — robustness

- **R1 (high) — `AlpacaBroker` has no transport retry, and HTTP 429 maps to `OrderRejected`.**
  `alpaca.py:120-127` sends one unretried request per call; `alpaca.py:172-181` classifies every
  4xx except duplicate-422 as a definitive per-order rejection. A 429 (rate limit) is a 4xx: under
  burst it is logged "rejected by the venue" and skipped — a transient misclassified against the
  broker.py taxonomy, and one transient 503 on `positions()` kills a session that "cannot be
  backfilled as live evidence" (docs/operations.md). Contrast `AlpacaBarSource._request`
  (`alpaca_data.py:231-267`), which retries 429/5xx with Retry-After + backoff.
- **R2 (high) — `fills_for` is one unretried GET per order id** (`alpaca.py:239-277`). A ~100-order
  refresh settle issues ~100 sequential requests against the same ~200 req/min budget as the bar
  fetch — peak 429 exposure at the settle step. R1's retry covers it.
- **R3 (high) — a torn ledger tail wedges every later cycle.** Ledger appends are buffered writes
  without fsync (`loop.py:557-563`); every production reader (`read_equity_ledger` etc.,
  `loop.py:486-528`) does `json.loads` per line with no tolerance. A crash mid-append leaves a torn
  final line; `halt_reason` → `read_equity_ledger` then raises on every subsequent cycle until
  manual repair. Doctor's `_last_ledger_bar` already tolerates bad lines; production readers don't.
- **R4 (med) — fills/unfilled appends are not crash-idempotent.** `settle` appends ledger rows
  *then* saves state (`loop.py:323-368`); a crash between the two re-appends the same rows on
  resume — duplicate rows in the I-9 calibration record. The other four ledgers are idempotent
  (monotone key); these two are not.
- **R5 (med) — no cross-process exclusion on the run directory.** Nightly loop, morning sweep, and
  manual invocations share `state.json` with no lock. The write-ahead protocol protects against
  crashes, not against two concurrent writers interleaving decide/settle (the runtime analogue of
  AGENTS.md §6's single-checkout hazard).
- **R6 (low) — `StateStore.save` fsyncs the file, not the directory** (`state.py:99-107`). On power
  loss the rename itself can be lost; the state then legally reverts one version, and a
  post-submit revert re-decides (client-id idempotency still protects the venue side).
- **R7 (low) — `submitted_order_ids` pages the account's entire order history on every submit pass**
  (`alpaca.py:184-207`). Latency and 429 exposure grow with account age; the pending set is always
  bounded to one decision bar.
- **R8 (low) — universe-file parsing exists four times** with divergent failure semantics:
  `paper_loop.py:233`, `replay_loop.py:150`, `doctor.py:134` (non-raising), plus the
  `load_universe` format contract in `io/universe_sp500.py`.

## Findings — structure / maintainability

- **M1 — ledger I/O is five hand-rolled copies of one concept.** Four monotone appenders + a
  raw appender + five near-identical `read_*` functions in `loop.py:481-663`; `daily.py:78-87`
  imports the private `_append_*`/`_require_price` across the module boundary.
- **M2 — `run_daily_cycle` is a 368-line monolith** (`daily.py:368-736`): settle, mark,
  concordance, halt, regime, cadence, de-gross, three-way construction dispatch, decide/submit,
  ledgers, monitor — all inline.
- **M3 — book dispatch is duplicated if/elif in two files**: construction on `config.book`
  (`daily.py:599-649`) and signal/config/prefix/cadence defaults on `args.book`
  (`paper_loop.py:294-521`).
- **M4 — `paper_loop.main` violates its own no-logic rule** (module docstring vs. 300 lines of
  profile-pin mutation of the argparse namespace, universe resolution, defaulting, safety
  derivation, staleness check — none of it unit-tested).
- **M5 — Alpaca HTTP transport is triplicated** with asymmetric behavior: broker (no retry), bar
  source (retry), `spinoff_mask.fetch_spinoffs` (third client, `raise_for_status`).

## Remediation map (this session)

| Finding | Change |
|---|---|
| R1, R2, R5(part), M5 | new `live/alpaca_transport.py`: shared retrying session (429/5xx, Retry-After, injectable sleep); broker + bar source rewired; 429 excluded from `OrderRejected` |
| R3, R4, M1 | new `live/ledgers.py`: fsync'd tolerant-tail JSONL discipline; `loop.py` delegates; settle appends made idempotent |
| R5 | new `live/lockfile.py`: advisory flock on `{run-dir}/.lock`, wired into paper_loop/paper_sweep CLIs |
| R6 | directory fsync in `StateStore.save` |
| R7 | optional `since=` on `submitted_order_ids`, wired via signature check in `_submit_pending` |
| R8 | new `io/universe_file.py`; all four parsers delegate |
| M2 | `run_daily_cycle` decomposed into per-step helpers (behavior-preserving moves) |
| M3, M4 | new `scripts/paper_books.py` book registry; `paper_loop.main` decomposed into testable builders |

Not fixed here, flagged only: spinoff rename-window blindness (documented vendor-convention gap,
`spinoff_mask.py:26-33`); doctor formatting/check coupling (acceptable as-is); `__init__.py`
hand-maintained `__all__`.

## Outcome (same session)

All mapped remediations landed. **Tested**: full core suite green — 901 passed, 144
research-deselected — including 30 new tests (tests/test_live_ledgers.py,
tests/test_lockfile.py, tests/test_paper_books.py, plus retry/idempotency additions to
tests/test_live_alpaca.py and tests/test_live_loop.py). The G6 pin is now itself a
regression test: BOOKS["momentum"] under CLI defaults must satisfy
assert_research_paper_bit_identity. mypy on the touched closure reports only the four
errors already present at HEAD (verified against a HEAD worktree). Offline doctor on
runs/paper_loop_momentum2 unchanged: 8 pass / 1 warn / 2 fail (the known dark-loop FAILs
that clear on the first completed session).

One deliberate behavior fix beyond pure refactor, in an uncounted path:
`--decision-every` was silently dropped for `--book ensemble` (constructed at cadence 1
regardless); the registry threads it through. Default behavior unchanged.

Deferred (flagged, not done): promote `docs/dev/agents_md_redraft_2026-07-29.md` if
ratified; consider `since`-scoping for `open_order_ids` (sweep path) symmetrical to
`submitted_order_ids`.
