# Prism — operating contract

Prism is a daily-bar, cross-sectional US equity trading system: score →
residualize → construct → execute, conditioned by a regime layer, evaluated by
a harness built to produce out-of-sample numbers that can be believed. This
file is the entire operating contract; code and docs defer to it. On
2026-09-02 it replaced the constitution, program document, doctrine,
amendments, pinned interpretations, ratification tags, timestamp anchors, and
agent contract that had accumulated through 2026-08 (≈13k lines of governing
prose over a 14k-line package). Registrations, read records, diagnostics, and
the certification under `docs/` remain as the research record.

## 1. State of the system (2026-09-03)

- **Live book.** 12−1 cross-sectional momentum (`prism.signal.momentum_node`:
  252-bar lookback, 21-bar skip, top/bottom decile equal-weight long−short,
  decisions every 21 bars) on the current S&P 500 universe, trading an Alpaca
  paper account. `runs/ACTIVE_RUN_DIR` names the run directory
  (`runs/paper_loop_momentum2`). Weekday schedule, Pacific time: 18:30
  `ops/paper_loop_nightly.sh` (decide at close, next-open orders), 06:50
  `ops/paper_sweep_morning.sh` (complete unfilled auction residuals). Both end
  with `prism-doctor`; a nonzero exit raises `{run-dir}/ALERT`.
- **Last completed session 2026-08-14** (EXIT 0; doctor 15 pass / 1 warn;
  equity 944,445; 99 positions; regime clock 12/21 clean sessions). No session
  has run since — the scheduler stopped firing, the code did not fail.
  Restarting the scheduler is the first operational task.
- **Evidence for the live book** (`results/demotion_b1/`, price-return
  convention, 2021-03 → 2026-06 out-of-sample, 21 purged folds): gross total
  return 32.4 %, total cost 3.7 %, net 27.6 %; net annualized Sharpe 0.465;
  DSR 0.191 against the 17-trial selection set it was discovered in; 21-bar
  rank IC 0.030 ± 0.024 (n = 62, one-sided lower-95 −0.009); N_eff ≈ 52.7;
  realized net Sharpe is 61 % of the IC·√N_eff ceiling. Read: at monthly
  cadence the book is **signal-bound, not cost-bound**; breadth and IC are the
  levers, cost is not.
- **Certified negative.** Daily residual reversion on the S&P cross-section is
  uneconomic at retail cost (`docs/certifications/001-…`, 17 trials, closed set).
- **Momentum fragility reads M1–M5 ran 2026-09-03**; the sleeve survives the
  pre-committed kill (median net Sharpe 0.477 vs B1 0.465, no sign flip;
  `docs/momentum_m_series_read_2026-09-03.md`; ledger
  `results/momentum_v1_trials.jsonl`, 6 rows; M2 and M4 sit at their N6
  ceiling; no configuration moved). **Trend T0–T4 ran 2026-09-03**: no
  standalone carry (net Sharpe 0.04, below the cash hurdle) and the convexity
  admission read fails on the B1 leg — admission refused
  (`docs/trend_t_series_read_2026-09-03.md`; 5 of 6 rows spent; T5 on its
  clock). Learned cross-section and replication: unbuilt.

## 2. Invariants — a change that violates one is wrong, not the invariant

- **N1 Point-in-time causality.** A value attributed to bar *t* uses only
  information available at *t*'s close. No across-bar ffill, no full-series
  fits before splitting.
- **N2 Next-open fills.** Decided at close *t*, filled at open *t+1* or later,
  in backtest and live.
- **N3 Costs before edge.** Half-spread per liquidity bucket, impact,
  commission, borrow, participation cap. Gross is a diagnostic, never a claim.
- **N4 The ledger conserves capital.** Equity moves only by PnL minus charged
  costs. Proven over exact integers in Lean (`formal/`, `rebalance_conserves`,
  `run_conserves`); the float implementation is property-tested against it
  (`tests/test_ledger_conservation.py`). Lean proves the algebra; pytest
  bridges the code to it. Nothing about markets is formalized.
- **N5 Every claim carries its deflation**, recomputable from the ledger (§4).
- **N6 Breadth is accounted.** Every cross-sectional claim reports N_eff
  (participation ratio of the post-residualization covariance) and the IC·√N_eff
  ceiling. Realized above ceiling ⇒ leak or bug. Ceiling below the after-cost
  hurdle at the one-sided lower-95 IC ⇒ not viable.
- **N7 Fail loud.** A failed fetch, an unfit model, a degenerate estimate
  raises or de-grosses explicitly. Silent zero / empty frame is a defect.
- **N8 The production import path is JAX/torch/prophet/mlflow/matplotlib-free**
  and `prism` never imports `research` (`tests/test_import_hygiene.py`, CI).

## 3. Evaluation standards

- Purged, embargoed walk-forward on every fit (`prism.validation.walk_forward`);
  all transforms train-only; no 80/20 split anywhere.
- Signals emit standardized scores with horizon metadata, never prices; one
  sizing function; blending in position space.
- Costs are calibrated per liquidity bucket (`SPREAD_BUCKET_SCHEDULE_V1`,
  `prism.execution.spread` from fills). Every net claim records its spread
  assumption, venue, fee, data convention, universe as-of, and coverage.
- The after-cost hurdle is the T-bill yield, stated in the packet.
- Regime features condition sizing only after positive incremental IC in purged
  WFO; regime never trades.
- The claim ladder is cumulative: `mechanics_clean` (pipeline runs, N4 holds)
  → `gross_edge` (gross > 0, N6 gates pass) → `net_edge` (net > 0 under bucket
  spreads) → `robust_edge` (net_edge and DSR against the selection set).
  No result is described above its tier. No capital is risked below `net_edge`.

## 4. The trial ledger is the whole authorization system

- **Register, then run.** Before the first configuration of a search is
  evaluated, write down the selection set: feature space, search procedure,
  universe, cost stack, sample, kill rule, promotion rule. That text is a
  tracked doc under `docs/` (the `*_design.md` / `*_preregistration.md`
  pattern). No owner ratification, budget cap, adjudication slot, tag, or
  timestamp is required. The registration is the commit that adds the doc.
- **Every evaluated configuration is a ledger row**, including NaN and
  degenerate outcomes, appended by the run itself (`--trial_ledger`,
  `research/scripts/stat_arb_residual_wfo.py` pattern) with config hash,
  sample, and periodic Sharpe; the claim packet carries the code commit.
  Ledgers are append-only; editing or deleting rows is falsification.
- **A claim deflates against its own selection set** — the effective
  independent count of every row in that set — never against a pooled ledger
  of incommensurable strategies, and never against fewer rows than were run.
- **Closed sets stay closed.** A set closes when its kill rule fires or its
  registration says it does. No new trial targets a closed set; a new idea is
  a new registration.
- **Promotion is out-of-sample by construction.** A pre-registered
  configuration promotes on data after its registration date (backtest
  extension plus the paper stream), at `net_edge` under bucket spreads, DSR
  > 0.5 against its set, periodic net Sharpe above the hurdle, and a live
  monitor read not in contradiction. Same-sample variants can kill, never
  promote.
- **Certified artifacts are immutable.** Anything under `results/` cited by a
  file in `docs/certifications/` is never rewritten.

## 5. Research direction

1. **The registered reads are done** (§1). M6 (≥ 2027-06 data) and T5
   (≥ 2027-07-17) stay on their clocks; nothing waits for them, and T5's B1
   leg is unreadable until M6 extends the B1 stream.
2. **Breadth.** N_eff ≈ 53 on ~96 names held is the binding constraint. The
   lever is a point-in-time universe beyond the S&P 500 with the survivorship
   leak counted, not hidden (`prism.scripts.build_sp500_universe` is the
   pattern). A named dataset may be bought if it closes a measured gap and
   total data spend stays ≤ ~$1,000/yr (`docs/data_purchase_evaluation.md`).
3. **IC.** A composite of slow, bar-derived characteristics at monthly cadence
   (momentum variants, long-horizon reversal, idiosyncratic volatility,
   residualized versions of each) as one registered selection set, deflated
   as a whole. Combination happens inside construction
   (`docs/aim_portfolio_preregistration.md`); the trend sleeve's convexity
   admission was refused 2026-09-03, so a second sleeve needs a new registration.
4. **Not this:** reviving residual reversion; intraday anything; model species
   (RL, forecaster zoos) in place of breadth; crypto before an equity sleeve
   promotes; quantum amplitude estimation (its own design concluded STOP).

## 6. Deployment gate

No real-money order until, all together: a sleeve is at `net_edge` on
out-of-sample data under §4; the crash-conditional de-gross sizing
(`docs/sizing_preregistration.md`) is armed by its own commit; the paper loop
has logged ≥ 21 consecutive clean sessions with regime telemetry
(`prism-doctor` regime-clock); a risk profile is chosen
(`docs/risk_profile_schema.md`, profiles only tighten pins). A ≤ $2,000
real-money micro-account mirroring the paper book for spread calibration is
measurement, not deployment, and is allowed before any of that. Opening,
funding, or resizing any real-money account, arming de-gross, and changing the
live book's configuration are owner acts. Everything else — including
registering and running trials — needs nobody's permission.

## 7. Working rules

- **The working tree is the live system.** The nightly runs the editable
  `.venv` install of whatever is in the tree at 18:30 PT; uncommitted changes
  trade that night. Do not leave the tree broken across that hour. To pause
  trading, create `{run-dir}/KILL_SWITCH`. One checkout, one writer at a time;
  re-check `git status` and HEAD before writing.
- **Tooling.** Python ≥ 3.12, uv. `uv sync` installs core + dev; `--extra
  research` adds the heavy stack (Linux, ~10 GB). Core suite:
  `.venv/bin/python -m pytest -q -m "not research"`. isort (black profile) on
  touched files only; never run black or isort repo-wide. mypy with targeted
  overrides. Commit style `type(scope): Sentence-case summary`, no
  attribution trailers.
- **Status on every non-trivial claim:** Tested (tests pass) / Run (executed,
  correctness not asserted) / Drafted (not executed) / Assumed. "Should work"
  is banned. Non-obvious claims carry a path, symbol, or artifact.
- **Code boundary.** `src/prism` is the shipped wheel; `research/` imports it
  and runs as `python -m research.scripts.<name>` from the repo root. Signal
  implementations are `*_node.py` under `src/prism/signal/`. Cross-boundary
  reuse is duplication by value with a citation, never an import.
- **`docs/dev/` is gitignored scratch.** Tracked files never cite it.
- **Audits produce the full finding set before any fix.** Refactors are
  authorized explicitly, never bundled with fixes.

## 8. Non-goals

No options or vol trading; no intraday alpha; no latency or order-book
modeling (fills are modeled adversarially); no RL in production; no direct
futures, cash Treasuries, or spot FX (ETF exposure only); no crypto shorting,
perps, or margin; no Hawkes / self-excitation modeling. US equities and ETFs
are the execution market; rates, FX, vol indices, and futures curves are
regime inputs only (`MARKETS.md`).

## 9. Map

```
src/prism/     io · signal · residual · portfolio · execution · regime · live ·
               validation · conformal · scripts (doctor, paper_loop, paper_sweep)
research/      quarantined: legacy forecasters, RL, batch WFO CLIs, diagnostics
formal/        Lean 4 kernel proofs (N4, band hysteresis, purge geometry, gate)
ops/           scheduled wrappers + alerting        tests/  offline suite
results/       ledgers and claim packets (cited ones immutable)
data/universe/ tracked PIT membership files         runs/   live run dirs (ignored)
```

`docs/`: registrations and designs (`*_design.md`, `*_preregistration.md`,
`risk_profile_schema`, `regime_step`), read records (`*_read_<date>.md`),
diagnostics and receipts, operations (`operations`, `quickstart`, `security`,
`free_tier_profile`), `certifications/001`, and `audit.md` (the 2026-06 audit
of the pre-Prism codebase, historical). `ARCHITECTURE.md` is the call graph;
`MARKETS.md` the market-structure analysis.
