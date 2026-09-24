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

## 1. State of the system (2026-09-23)

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
  clock). Replication: unbuilt (no survivorship-complete panel).
- **Reads of 2026-09-23** (canonical engine, $100,000, 1,056 sessions
  2022-03 → 2026-06 unless noted; every control reproduces the construction
  read's decile control exactly). `characteristics_v1` (6 of 6,
  `docs/characteristics_read_2026-09-23.md`): idiosyncratic volatility alone
  killed; **residual momentum** net Sharpe 0.829 vs 0.679, max DD 4.3% vs
  16.3%, N_eff 71 vs 47, not distinguishable from the control (p 0.79).
  `breadth_v1` (3 of 3, `docs/breadth_read_2026-09-23.md`): S&P 500 ∪ 400
  raises N_eff 37% and not net Sharpe (0.638). `learned_xsection_v1` X0-X4
  (5 of 6, 300 sessions, `docs/learned_xsection_read_2026-09-23.md`):
  fragility survives, admission refused (spanned by momentum, alpha t 0.59).
  `forward_v1` (`docs/forward_design.md`) registers the two residual-momentum
  candidates' forward read: interim kill ≥ 126 sessions after 2026-09-24,
  promotion ≥ 252, deflated against nine rows.
- **Paper loop rehearsed 2026-09-23** with `--no-submit`: exit 0, nothing
  sent, 119 orders decided (`docs/operations.md`). Re-arming the schedule is
  the owner's act.
- **Volatility-scaled momentum promoted 2026-09-24** (replication route,
  `vol_momentum_v1`, 1 of 1, `docs/vol_momentum_read_2026-09-24.md`): the live
  book's configuration scaled by the Barroso–Santa-Clara rule (12% target, 126
  sessions, monthly, cap 2). Prior 0.465 (since 2015); fidelity 0.895; F3 0.368
  against a T-bill hurdle of 0.351, a 0.018 margin. With idle cash earning
  nothing, the expected edge over T-bills is about 0.5% a year; if idle equity
  earns the bill rate, it is about 4.7%. Arming the scale, replacing §6's
  de-gross with it, and changing the live book are owner acts. The forward kill
  rule is live; its first read is at about 2027-03-24.
- **Public factor lab ran 2026-09-24** (`sizing_lab_v1`, 8 of 8,
  `docs/public_factor_lab_read_2026-09-24.md`). On French/JKP daily data
  (momentum 1931–2026, residual momentum 1958–2025), the ratified crash
  de-gross is **contradicted**: it lowers Sharpe before 2020 on both bases and
  holds half gross ~45% of sessions. Constant-volatility scaling is supported
  and roughly doubles momentum's Sharpe (0.37 → 0.78; +0.23 after
  publication). Post-publication priors: momentum 0.35, residual momentum
  0.18. In-house fidelity: momentum 0.88; residual momentum 0.495, below the
  §4 floor. §6 still names the contradicted de-gross; changing that is the
  owner's call.
- **Contract amended 2026-09-23** (owner decision): a published characteristic
  implemented as published can promote on long-sample public evidence plus
  in-house fidelity and cost tests (§4, replication route); research scope is
  open to any data source and technique that satisfies §§2–4 (§§5, 8).

## 2. Invariants — a change that violates one is wrong, not the invariant

- **N1 Point-in-time causality.** A value attributed to bar *t* uses only
  information available at *t*'s close. No across-bar ffill, no full-series
  fits before splitting.
- **N2 Next-open fills.** Decided at the close of bar *t*, filled at the open
  of bar *t+1* or later, in backtest and live; on daily bars, close *t* to open
  *t+1* or later.
- **N3 Costs before edge.** Half-spread per liquidity bucket, impact,
  commission, borrow, participation cap. Gross is a diagnostic, never a claim.
- **N4 The ledger conserves capital.** Equity moves only by PnL minus charged
  costs. Proven over exact integers in Lean (`formal/`, `rebalance_conserves`,
  `run_conserves`); the float implementation is property-tested against it
  (`tests/test_ledger_conservation.py`). Lean proves the algebra; pytest
  bridges the code to it. Nothing about markets is formalized.
- **N5 Every claim carries its deflation**, recomputable from the ledger and
  any public series it cites (§4).
- **N6 Breadth is accounted.** Every cross-sectional claim reports N_eff
  (participation ratio of the post-residualization covariance) and the IC·√N_eff
  ceiling. Realized above ceiling ⇒ leak or bug. On the discovery route (§4),
  a ceiling below the after-cost hurdle at the one-sided lower-95 IC ⇒ not
  viable.
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
- Regime features enter sizing or signals only after positive incremental IC
  in purged WFO.
- The claim ladder is cumulative: `mechanics_clean` (pipeline runs, N4 holds)
  → `gross_edge` (gross > 0, N6 gates pass) → `net_edge` (net > 0 under bucket
  spreads) → `robust_edge` (net_edge, and DSR against the selection set or the
  replication tests of §4).
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
  `research/scripts/trend_wfo.py` pattern) with config hash,
  sample, and periodic Sharpe; the claim packet carries the code commit.
  Ledgers are append-only; editing or deleting rows is falsification.
- **A claim deflates against its own selection set** — the effective
  independent count of every row in that set — never against a pooled ledger
  of incommensurable strategies, and never against fewer rows than were run.
- **Closed sets stay closed.** A set closes when its kill rule fires or its
  registration says it does. No new trial targets a closed set; a new idea is
  a new registration.
- **Promotion has two routes**, and a registration names its route before its
  first run. Promotion makes a configuration eligible for the live book. It is
  revocable: the registered kill rule runs on every later session.
- **Discovery route**: anything this program found by searching data,
  including weights fitted to returns and a choice among implementations made
  on in-house Sharpe. It promotes on data after its registration date
  (backtest extension plus the paper stream), at `net_edge` under bucket
  spreads, DSR > 0.5 against its set, periodic net Sharpe above the hurdle,
  and a live monitor read not in contradiction. Same-sample variants can kill,
  never promote.
- **Replication route**: a published characteristic implemented as its source
  defines it. Choices the source leaves open (universe, the factor model used
  to residualize, construction) are allowed; the fidelity test licenses them.
  Before any in-house run, the registration names the source, a public return
  series for it (Kenneth French, AQR, JKP, Chen–Zimmermann, or the source's
  own tables where no series exists), the implementation, the fidelity floor,
  and the forward kill rule. It promotes when all of these hold:
  1. *Publication bar.* The public long−short series has t ≥ 3 over at least
     20 years. Its prior Sharpe is the post-publication Sharpe where that
     history spans ten years or more, else 0.4 × the full-sample Sharpe
     (McLean and Pontiff 2016 measure a 58% post-publication decline). A
     prior Sharpe ≤ 0 fails.
  2. *Fidelity.* The implementation's in-house returns correlate with the
     public series at ρ ≥ the registered floor, never below 0.5, at the
     series' native frequency.
  3. *Economics at our costs.* ρ × prior Sharpe (the Sharpe of the component
     that tracks the public series), less the implementation's measured cost
     drag under bucket spreads, clears the hurdle.
  4. *No in-house contradiction.* The in-house net Sharpe's 95% interval does
     not lie entirely below the figure from 3.

  A combination of promoted configurations at weights pinned in its
  registration, not fitted to returns, is promoted when each part is and the
  combined book is `mechanics_clean`.
- **Certified artifacts are immutable.** Anything under `results/` cited by a
  file in `docs/certifications/` is never rewritten.

## 5. Research direction

1. **Clocks already running.** M6 (≥ 2027-06 data), T5 (≥ 2027-07-17), and
   `forward_v1` (from 2026-09-24) stay as registered; nothing waits for them.
   T5's B1 leg is unreadable until M6 extends the B1 stream.
2. **Priors from long public samples.** Free factor libraries (Kenneth French
   daily factors and industries from 1926, AQR, JKP, Chen–Zimmermann) supply
   what 2020–26 cannot: per-theme post-publication Sharpe, cross-theme
   correlations, and crash and volatility-scaling rules tested over the whole
   momentum history before any is armed. The replication route (§4) consumes
   them.
3. **New information, not new transforms.** Characteristics built from the
   daily price and volume path are saturated on this sample: the learned
   combiner was spanned by momentum (alpha t 0.59), and residual and raw
   momentum returns correlate 0.58 (`results/2026-09-characteristics/returns/`).
   The next information is fundamentals from SEC EDGAR XBRL, point-in-time by
   filing date (value, profitability, investment and issuance, accruals,
   earnings surprise), and earnings dates from 8-K item 2.02 (announcement
   returns, post-announcement drift). Any source that can be made
   point-in-time (N1) is admissible. Data may be bought while total spend
   stays ≤ ~$1,000/yr; `docs/data_purchase_evaluation.md` names the pick for
   survivorship-complete pre-2020 history.
4. **Themes at fixed risk weights.** Themes combine at risk weights pinned in
   the registration: not dollar weights (the ½ + ½ momentum blend correlates
   0.97 with raw momentum, which carries 2.6× the volatility), and not weights
   fitted to returns outside the discovery route. Each theme is neutralized to
   the factor model. Momentum splits into its stock-specific part and its
   factor/industry part, each with its own budget; the factor part may trade
   at short lookbacks through liquid ETFs.
5. **Horizon.** Every signal reports its IC at 1, 2, 5, 10, 21 and 63
   sessions. One daily cost-aware construction step (the Gârleanu–Pedersen
   design, `docs/aim_portfolio_preregistration.md`) replaces calendar
   rebalancing, and it never re-scales a summed book
   (`docs/b2_anti_netting_diagnostic.md`). Fast signals (earnings events,
   turnover-conditioned reversal and continuation, overnight/intraday return
   splits, short-term reversal) are judged by their marginal effect on the
   combined book's after-cost return, usually as trade timing. Certification
   001 tested standalone reversal books; it does not bear on that use.
6. **Past negatives are priors, not prohibitions.** Standalone daily residual
   reversion (certification 001), ETF trend as standalone carry (T-series),
   mid-cap breadth on free data, price-only learned combiners, low
   idiosyncratic volatility in 2022–26 large caps, and quantum amplitude
   estimation (its own design concluded stop) are evidence to weigh. Closed
   sets stay closed (§4); a new idea is a new registration.

## 6. Deployment gate

No real-money order until, all together: a configuration is promoted under
§4 (either route); the crash-conditional de-gross sizing
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

## 8. Scope

US equities and ETFs are the execution market: no options or vol trading; no
direct futures, cash Treasuries, or spot FX (ETF exposure only); no crypto
shorting, perps, or margin. Widening what capital is exposed to is an owner
decision. Inside that scope no technique is excluded: intraday signals,
order-book and latency models, reinforcement learning, point-process models,
alternative data, and anything else are in bounds when they satisfy §§2–4. A
new bar frequency defines its fill model under N2 before its first counted
run. N8 limits the production import path, not methods: a model trained with
any library ships as plain arrays. Any market series, including rates, FX,
vol indices, and futures curves, may be a signal input (`MARKETS.md`).

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
