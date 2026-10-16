# Public baseline: tranched, volatility-scaled 12−1 momentum — design and registration (`public_baseline_v1`, replication route)

**Status: registered 2026-10-16.** The registration is the commit that adds
this file and `registrations/2026-10-public-baseline.toml` (SPEC §4). Every
parameter is in the TOML, which the driver reads strictly (an unknown key is
refused). Before this commit no row of this set had been run: there is no
in-house run of any variant below, and no tracking correlation, cost drag,
hurdle or verdict of this set has been computed. The driver,
`research/scripts/public_baseline.py`, and the engine changes it runs on were
built and tested on synthetic data only. Two driver parts that evaluate
nothing were run to prepare this commit: `--part pin`, which downloaded the two
French files and hashed them, and `--part check`, which validated this TOML,
the dataset manifest, the fold table and all 30 row specifications. The French
files pinned here are **not unseen**: another work package computed
statistics on them before this commit, and §2 lists what it computed.

"Approval" below is what SPEC §4 calls promotion.

## 0. Summary

This set evaluates one strategy as the repository's public baseline: the
momentum book approved for live use under `vol_momentum_v1` (12−1 momentum,
top and bottom decile of the S&P 500, scaled to 12% volatility by the
Barroso–Santa-Clara rule), split into 21 overlapping tranches that each
rebalance monthly on a different session, with idle own equity held in a T-bill
ETF.

Tranching changes when trades happen, not what is held. On every session every
deployment refreshes the same one tranche, about 1/21 of the book, so orders
from many independent users spread over the month instead of landing on one
day.

Of the 30 rows in the set, one (T21) is the only row that can be approved. The
other 29 measure what tranching, execution choices, holdings diversity and
account size cost. Approval follows the replication route of SPEC §4. §9 prices
crowding from many deployments under stated assumptions; it is uncounted and
cannot choose a configuration.

## 1. What is registered

### 1.1 The candidate, T21

- **Signal.** `prism.signal.momentum_node.MomentumSignalNode`, lookback 252
  sessions, skip 21. Eligibility screen of the `characteristics_v1` control
  (`examples/characteristics/control.toml`): price ≥ $5, 20-session median
  close × volume ≥ $1M, 252 sessions of return history, index membership on the
  decision date.
- **Universe.** The S&P 500 point-in-time membership of the frozen dataset
  `characteristics_v1` used (`demotion_b1_2026-06-16`, manifest id in the TOML).
  It resolves 574 of the 633 names that were index members over the dataset's
  span (coverage 0.907, from the manifest's coverage block).
- **Per-tranche book.** Rank-to-weight rule `decile_long_short`
  (`prism.portfolio.construct.construct_decile_neutral`): the top and bottom
  decile of eligible names by score, equal weight, long leg +0.5 and short leg
  −0.5 of the tranche's capital (raw gross 1.0), per-name cap 0.10.
- **Tranches.** 21, each holding 1/21 of the book's weight.
  - The grid position of a session is the number of NYSE sessions since
    2006-01-03 (`prism.io.sessions.grid_position`). 2022-03-29, the first
    decision date, is position 4087, which is 13 modulo 21.
  - Tranche *j* (*j* = 0…20) refreshes on the sessions where grid position ≡
    13 + *j* (mod 21), so exactly one tranche refreshes on each session
    (`prism.composition.tranche_plans`, `due_on_grid`).
  - A tranche that misses its grid date refreshes once on the first later
    session on which more than 21 sessions have passed since its last refresh,
    then returns to its grid. Its output stays valid for 42 sessions, so a
    missed date makes the refresh late, not expired.
- **Entry.** The account is empty until the first test session, 2022-03-29;
  nothing is decided before it. Each tranche is first built on its own grid
  date, so the book reaches full size over 21 sessions. This evaluation has no
  wind-down; a live wind-down is not part of this set.
- **Holdings between refreshes.** Each tranche is held in shares
  (`prism.evaluate._TrancheHoldings`), as overlapping Jegadeesh–Titman cohorts
  are held.
  - At tranche *j*'s refresh at close *t*, its holding is reset to
    (1/21) · *s_j* · *w_j,i* · *E_t* / *P_t,i* shares of each name *i*, where
    *w_j* is its decile book, *s_j* its volatility scale, *E_t* the marked equity
    and *P_t,i* the close. A name the tranche targets that has no price on the
    session keeps the tranche's previous holding of it, or is not entered.
  - Between refreshes the shares are held and drift with prices; a split
    restates them by the factor the account's shares move by.
  - The account target at close *t* is Σ_j shares_j · *P_t* / *E_t*. A session
    decides only the names the refreshing tranche held or now holds, plus any
    name whose account shares differ from the tranches' total by more than half
    a share (an unfilled order, or a share lost to rounding). Every other name is
    held. So a session's orders are the refreshing tranche's change and those
    catch-ups, never a rebalance of the drifted book.
  - Orders are sized and rounded to whole shares by the execution profile's
    batch rounding (`prism.decisions.decide_bar`).
- **Volatility scale, per tranche.**
  - Each component has one *base book*: the weight-averaged unscaled decile book
    of the tranches that currently hold one (a single-tranche row's own book),
    held at constant weights from close to close, gross of costs, each return
    using the weights of the prior close only
    (`prism.portfolio.vol_scale.base_book_returns`). Averaging over the tranches
    that hold a book keeps a book that is still being built from reading as a
    quiet one.
  - At tranche *j*'s refresh at close *t*, σ̂_t is the sample standard deviation
    (ddof 1) of the base book's last 126 daily returns, times √252, taken before
    anything decided at *t* enters the base book (N1). Then *s* = min(2, 0.12 / σ̂_t),
    held by that tranche until its next refresh; the orders it sizes fill from
    the next open (N2). This is `research/scripts/factor_lab.py::scale_constant_vol`
    by value (`prism.portfolio.vol_scale.decide_scale`).
  - The base book is built from this run's own decisions, so its first return
    is on 2022-03-30. Until a tranche's window holds 126 returns and it has never
    had a good decision, its scale is 1, recorded as warm-up.
- **Gross ceiling.** The decided account target is capped at gross 2.0
  (`prism.portfolio.vol_scale.cap_gross`). When the cap binds, every tranche's
  holding is scaled by the same factor and the whole book is decided on that
  session. The ceiling bounds each decided target, not the held book: between
  decisions held gross drifts with prices and can close above 2.0, which is
  reported beside criterion 5.
- **Scale failure.** `decide_scale` checks, in this order: the price panel's
  last bar is the decision bar; the window holds 126 finite returns; no more
  than 5% of the 126 returns are exactly zero (a stale or flat panel); σ̂ is
  finite and above 1e-10. A failure gives min(1, the tranche's last good scale),
  or 1 if it has none, and is recorded with its reason in
  `sizing_decisions.csv`. No failure can raise a scale above 1 or above the last
  good scale.
- **Construction and policy files.** Those of the `characteristics_v1`
  `momentum_control` arm, with the cost stack replaced (§3):
  - construction: the control's heuristic block with the no-trade band off
    (the account-state path applies none,
    `docs/construction_band_audit_2026-09-24.md`), max gross 2.0, per-name cap
    0.10;
  - profile `examples/reference/profiles/experiment_v1.toml`: its preference
    limits (max gross 1.0, max name weight 0.10, net band 0.02) are evaluated and
    recorded at each decision and do not change a target whose rounded batch
    validates (recovery policy `preference_response = "record"`). Its variance
    budget needs a risk snapshot, which this specification does not build (no
    risk model is declared), so it is recorded as unchecked. Its physical limit,
    each name's trade at most 5% of the name's trailing 20-session median dollar
    volume, is enforced. If a rounded batch fails validation for another reason,
    the recovery policy's repair step minimizes a measure that includes the
    preference violations, so it can trim gross toward 1.0; any such repair is in
    the run record;
  - execution profile `examples/reference/profiles/execution_alpaca.toml`: whole
    shares, $1 minimum order, no order that crosses zero in one step;
  - `examples/reference/policies/monitoring.toml` and
    `examples/reference/policies/recovery.toml`.
- **Fills.** `full_next_open`: an order decided at close *t* is a
  market-on-open order that fills in full at the open price of *t*+1. No fill is
  created on a session the name did not trade or at a price it did not print
  (`prism.execution.simulate`). Initial capital $100,000.
- **Cash.**
  - Idle own equity above a 2% buffer is held in a T-bill ETF (SGOV; BIL as the
    alternate). In this evaluation the holding is not simulated: it enters
    through the hurdle (§6) as a long position earning rf − 10 bp. The 10 bp is
    an assumption: SGOV's expense ratio is about 0.09% according to secondary
    sources, not yet checked at the issuer.
  - The buffer earns nothing.
  - Short-sale proceeds are collateral and earn nothing.

### 1.2 Diagnostic rows

These measure costs and alternatives and cannot be approved. See §4.

## 2. Why this configuration, and what was seen before registering

- `vol_momentum_v1` is the only configuration in the program with a positive
  pre-registered forward figure: tracking correlation 0.895 × post-publication
  Sharpe 0.465 (t 1.56) − cost drag 0.047 = 0.368, against a hurdle of 0.351
  (`results/2026-09-vol-momentum/summary.json`, 202607 French vintage).
  - Every other replication and both pinned combinations fail economics
    (`docs/hurdle_restatement_2026-09-30.md`).
  - Cut at 2025-12-31, the post-publication Sharpe was 0.401 and the figure
    0.312 (`docs/fundamentals_read_2026-09-24.md`, "Vol-scaled momentum's prior
    depends on the window's end").
- Its recorded evidence is one rebalance phase. Rebalance timing luck in monthly
  momentum books is large in the literature (Hoffstein, Faber and Braun 2020)
  and unmeasured here.
- Every deployment of the repository computes the same book. The live loop's
  phase is set by each deployment's first session
  (`prism.live.daily._is_refresh_session`: the first cycle always refreshes), so
  users who start together trade together.
- Overlapping cohorts are how Jegadeesh and Titman (1993) built momentum
  returns. `docs/prism_engine_decisions.md` classifies cohort staggering as a
  strategy choice that must be evaluated ("Cohort staggering is a separate
  evaluated strategy choice"). This registration is that evaluation.

**Prior exposure, disclosed.** The following were known when this set was
written:

- every number in `results/2026-09-vol-momentum/summary.json`, including the
  in-house scaled net Sharpe 0.663 over 929 sessions and the unscaled 0.681 on
  the same window;
- the unscaled control's figures (`characteristics_v1`) and every other read
  of this program on the same 2020–2026 in-house sample;
- the 17-row set the live configuration came from (`momentum_v1`) and the 8-row
  sizing study that chose the rule (`sizing_lab_v1`);
- the paper account's fill mix: 185 of the 249 fills in
  `tests/fixtures/paper_replay/fills.jsonl` (74%) are morning sweep orders
  (client order ids ending `:S<n>`) rather than opening-auction fills;
- the planning arithmetic for this set, which derived a range of forward
  expectations from the recorded numbers above (no market data was read for it);
- **statistics of the French files pinned here.** On 2026-10-14 at 21:40 PT,
  after the plan for this set was written and before this commit, the keyless
  reproduction package (`docs/public_reproduction_design.md`, addendum dated
  2026-10-16, commit `7c62f4a`, not yet merged into this branch) fetched these
  same two files (the same sha256, byte counts and `Last-Modified`) and
  recomputed `vol_momentum_v1`'s criteria on them, in output kept outside the
  repository. Its addendum records, through 2026-08-31, with
  `vol_momentum_v1`'s rule, warm-up and 2015-04-01 publication cut:
  - post-publication Sharpe of the scaled reference 0.4446 (t 1.50), and
    full-sample Sharpe 0.8986. This set's criterion-1 prior is the same
    computation on the same file;
  - `vol_momentum_v1`'s economics figure 0.3506, using that set's recorded
    tracking correlation (0.895) and cost drag (0.0474), against its hurdle of
    0.3508 at its recorded rf of 4.21% (a fail by 0.0002), or 0.333 at the rf of
    4.00% recomputed from this file's last 252 RF rows (a pass). That 4.00% is
    the rf this set's hurdle uses (§6);
  - cut at 2026-07-31, a prior of 0.46464 against the recorded 0.46457, so the
    revision between vintages moved it by less than 0.0001.

  The same run computed every public factor-lab series' Sharpe and prior on the
  202608 French files; this set reads only the `Mom` and `RF` columns. No
  in-house figure of this set was computed by it. The registering session wrote
  the TOML before it learned of that run, saw only the run's qualitative
  outcome (the economics test failing at one rf and passing at the other) before
  writing this document's prose, and read the figures themselves after the
  whole document was drafted. They were then copied into this paragraph; later
  edits only corrected wording to match the implementation, and no parameter or
  rule was changed. Every parameter is the planning document's (finished
  2026-10-14 at 21:18 PT, before that run) or a correction that makes the text
  match the driver as implemented. This registration pins the vintage
  current at its commit, as the plan specified. Registering instead on a later
  vintage that nobody has read would be a new registration; since no row of this
  set has run, that choice stays open to the owner until the first row runs.

No row of this set has been run.

## 3. Data, sample, engine, costs

- **Public series.** Kenneth French's daily momentum factor (`Mom`) and the
  daily three-factor file (`RF`), at the vintage served on 2026-10-16: the
  202608 CRSP vintage, daily rows through 2026-08-31.
  - The sha256, byte count, `Last-Modified` header and first line ("vintage
    line") of each file are pinned in the TOML. A file whose sha256, byte count
    or vintage line differs is refused; `Last-Modified` is recorded, not
    checked.
  - The files are cached under `data/public/vintages/<sha256[:12]>/`, which is
    git-ignored and never overwritten. A file missing from the cache is
    downloaded and must hash to its pin, or nothing is written.
  - The scaled reference is `Mom` scaled by the same rule (decisions every 21
    rows counted from the file's first row, 126-row window, cap 2), gross of
    costs, scored from its 1,261st row, exactly as in `vol_momentum_v1`
    (`research/scripts/vol_momentum_replication.py::scaled_reference`).
- **In-house data.** The frozen dataset `demotion_b1_2026-06-16` of
  `characteristics_v1` (manifest id pinned): 578 files, daily bars 2020-01-02 →
  2026-06-15. The driver checks the manifest id and vintage and does not re-hash
  the files at run time; on 2026-10-16 every file of the copy at
  `data/frozen/demotion_b1` hashed to its manifest entry
  (`prism.io.manifest.Manifest.verify_files`). `--dataset` may name another copy;
  the manifest id and vintage are what is checked.
- **Sample.**
  - Decisions start on the first test session, 2022-03-29, the date
    `vol_momentum_v1`'s scale decisions start from (its `refresh_phase_start`, a
    rebalance date of the unscaled control book). The last scored session is
    2026-06-12 (1,056 scored sessions). The engine runs from 2020-12-31 (the first
    fold's formation start) to 2026-06-15, one session past the last scored one so
    the last decision fills.
  - **Common window.** Every row is scored from the first session on which
    every tranche of every row holds a scale from a full-window decision, to
    2026-06-12. From the session calendar alone this is 2022-10-27 (909
    sessions): the last tranche to be built (20 sessions after the anchor) first
    has 126 base-book returns at its refresh 146 sessions after the anchor. If a
    first full-window decision falls back, the window starts later; the driver
    computes it from the recorded decisions, and refuses if the rows' series
    differ in length on it.
  - **Parity window.** P00 is also read on the parity series' own 929 sessions,
    2022-09-29 → 2026-06-12. P00's first full-window decision is 126 sessions
    after the anchor, so it is scaled from 2022-09-29, as `vol_momentum_v1` was.
- **Engine.** The canonical engine (`prism.evaluate.walk_forward_evaluate`) with
  `decision_coupling = "account_state"` (each decision is made against marked
  equity and held shares), `fold_portfolio = "carry"` (one continuous account),
  `cost_refresh = "fold"` and `visibility_mode = "research_vintage"`.
  - The fold table: 17 contiguous test windows of 63 sessions from 2022-03-29
    (the last is 48), each after a 312-session formation window. These are folds
    4–20 of `folds/demotion_b1.toml`, the geometry `vol_momentum_v1`'s in-house
    book was charged under. The driver builds it from the session calendar,
    writes it to `results/2026-10-public-baseline/folds.toml` and records its
    content hash (`59ddc263…`) in every row record.
  - No parameter is fitted. A fold's formation window sets only its spread
    buckets: each name's median close × volume over that window, for the index
    members at the window's end. A name that was not a member then is charged
    the widest step for the fold.
- **Return convention.** The in-house series are price returns, as in every
  recorded in-house figure; the frozen dataset carries no cash distributions.
  The public series are total returns. The mismatch is stated in every output.
- **Cost stack:** `examples/reference/costs/retail_v3.toml`, which is
  `retail_v2` with two changes:
  - `cash_income_bps_annual = 0`, so idle cash enters only through the hurdle
    and nothing is counted twice;
  - a square-root participation term, `participation_coeff = 126` bp
    (Y = 0.7 times an assumed daily volatility of 1.8%, class "assumption"):
    cost per traded dollar 126 bp × √(q / V), with q the order's decided notional
    and V the close × full-day volume of the session it fills in, floored at
    $100,000.
  - Unchanged from `retail_v2`: half spreads of 1, 2, 5 and 10 bp per fill by
    formation-window median dollar volume (≥ $500M, ≥ $100M, ≥ $25M, below),
    1 bp commission, 50 bp/yr borrow on short market value, the quadratic impact
    term, 6.5%/yr financing on debit cash, and short proceeds held as collateral
    without rebate.
  - The stack declares its spread statistic refreshed at each decision; the
    run's fold cadence is what is applied, and each row record's warnings say
    so.
- **Bootstrap.** Studentized circular block bootstrap, 42-session blocks, 2,000
  resamples, seed 20261015, 95%, with the 21-session decision interval
  (`prism.validation.compare`). It gives criterion 4's interval and every paired
  interval.

## 4. The set: 30 counted rows

Every attempt at a row is appended to `trials/public_baseline_v1.jsonl`: the
engine writes a `started` row and one terminal row per attempt, with `arm_id` =
the row id, `kind` = `new` or `retry`, and the run's `config_id` and `data_id`;
the run record it points to carries this registration's id and hash. A row
whose specification cannot be built never reaches the engine, and the driver
appends its `failed` row.

- **Order.** P00 runs first, then the parity gate is read (§5, criterion 5),
  then T21, then the other rows in the TOML's order, one at a time, each with
  its memory use logged.
- **Retries.** `--retry failed` or `--retry <ids>` re-runs an attempt that
  failed, never ended, or finished with a mechanics failure (for P00, a failed
  parity gate). A retry spends nothing. A finished row whose mechanics passed is
  never re-run. Every attempt is listed in `summary.json`; a superseded
  attempt's outputs are kept under `rows/<id>.superseded-<time>/`.
- **Closing.** The set closes when all 30 rows have finished. The forward stop
  rule (§10) withdraws eligibility; it does not reopen the set.

| Row(s) | What it is | Can be approved |
|---|---|---|
| P00–P20 | Single-tranche scaled book refreshing every 21 sessions, phase k = 0…20 sessions after the anchor 2022-03-29 (grid offset 13 + k mod 21). σ̂ from that row's own base book. P00 is `vol_momentum_v1`'s configuration run inside the engine (differences in §13): a control and parity check, not a new trial on that closed set, and its result does not change that set's verdict. | No |
| T21 | §1.1 | **Yes (only row)** |
| D-split | T21 with each order filled half at the open of *t*+1 (rounded to whole shares, ties to even) and the rest at the close of *t*+1; each leg is charged as a fill (fill model `split_open_close`, declared here under N2) | No |
| D-liq | T21 with weights within each leg proportional to √(20-session median close × volume, ending at the decision close), legs at ±0.5, per-name cap 0.10 (rule `decile_neutral_sqrt_dv`) | No |
| D-part1–3 | T21 on three seeded subsets of the S&P 500 membership: a name is kept on every date when HMAC-SHA256(seed, security id) / 2^64 < 0.8 (`research/scripts/universe_partition.py`; seeds in the TOML, drawn with `secrets.randbits(63)`) | No |
| T21-25k | T21 at $25,000 initial capital | No |
| T21-1m | T21 at $1,000,000 initial capital | No |
| T3-25k | 3 tranches at grid offsets 13, 20 and 6 (the anchor, +7 and +14 sessions), $25,000 | No |

**Choosing a diagnostic row after seeing results is ruled out.** Adopting any
diagnostic variant later requires a new registration that approves it on data
after its own registration date.

## 5. Criteria and approval rule for T21 (SPEC §4, replication route)

Source: Jegadeesh and Titman (1993) for the characteristic; Barroso and
Santa-Clara (2015), JFE 116(1), for the sizing rule. The public series is §3's
scaled reference. Criteria 1–4 reuse `publication_bar`, `fidelity` and
`no_contradiction` from `research/scripts/vol_momentum_replication.py`. T21 is
approved only if all six hold.

1. **Publication bar.** Over the full scaled reference, t = annualized Sharpe ×
   √years ≥ 3 and years ≥ 20. The prior is the annualized Sharpe from
   2015-04-01 to 2026-08-31, because that span is at least 10 years (otherwise
   it would be 0.4 × the full-sample Sharpe). A prior ≤ 0 fails. Reported beside
   it: the same figure cut at 2025-12-31, and 0.4 × the full-sample Sharpe.
2. **Tracking.** The daily Pearson correlation ρ of T21's daily net returns with
   the scaled reference over the common window is ≥ 0.75. The weekly
   correlation (Friday-ending compounded returns) is reported beside it.
3. **Economics.** ρ × prior − cost drag > the governing hurdle Sharpe (§6),
   strictly. Cost drag is T21's annualized mean charged cost (as a fraction of
   the prior session's equity) divided by the annualized volatility of its daily
   net returns, both on the common window.
4. **No in-house contradiction.** The upper end of the bootstrap 95% interval
   of T21's annualized net Sharpe on the common window is at least the figure
   from 3.
5. **Mechanics.**
   - N4: on every session of T21's simulation, the change in equity equals held
     shares × the change in marks, plus fills × (mark − fill price), plus event
     cash, less charged costs, within $10⁻⁶ + 10⁻⁹ × equity, and no held name is
     ever left without a mark.
   - Every decided account target of T21 has gross ≤ 2.0 (relative slack 10⁻⁹),
     and no fallback scale exceeds min(1, the last good scale). The held book's
     maximum gross and the number of sessions it closed above 2.0 are reported
     beside the criterion and are not part of it.
   - P00 passes the parity gate: on the 929 sessions of
     `results/2026-09-vol-momentum/series.csv`, column `in_house|net`, P00's
     daily net returns have correlation ≥ 0.99 with it and an annualized Sharpe
     within 0.05 of it. If the gate fails, the run stops before any other row,
     P00 is recorded as a mechanics failure, the summary's verdict is
     `mechanics_failure`, and no verdict is read. After a diagnosed fix P00 may
     be retried.
   - The same N4, ceiling and fallback checks are recorded for every row; only
     T21's and P00's gate enter the criterion.
6. **Tranching does not cost return.** The paired bootstrap 95% interval of the
   annualized net-Sharpe difference T21 − P̄ does not lie entirely below zero
   (its upper end is ≥ 0), where P̄ is the equal-weight average of the daily net
   returns of P00–P20 on the common window
   (`prism.validation.compare.sharpe_difference_interval`). If the two series
   are identical the difference is recorded as zero without an interval. T21 −
   P00 is reported beside it.

**Consequences.**
- Approval makes T21 with its T-bill holding eligible for the live book. Arming
  the scale, holding the ETF, choosing the risk profile, replacing the
  crash-triggered gross reduction named in SPEC §6 and changing the live book
  are all owner acts.
- If T21 is deployed without the T-bill holding, the applicable verdict is
  criterion 3 at the hurdle of the account's verified cash yield (§6).
- If any criterion fails, T21 is not approved and nothing in this set is.
  `vol_momentum_v1`'s own verdict and stop rule are unchanged either way.

## 6. Hurdle and cash conventions

Computed with `prism.validation.hurdle.tbill_hurdle` and its T-bill ETF inputs
(`sleeve_fraction`, `sleeve_yield`).

- **rf** is the mean of the last 252 rows of French `RF` in the pinned file
  (the 252 sessions ending 2026-08-31), annualized × 252. No other rate is read.
- **Governing hurdle.** [rf − (rf − 0.0010) · s̄ − 0 · b̄] / vol, where:
  - L_t/E_t is T21's stock long side over equity at each close (recorded every
    session in `sizing.csv`), and L̄/E its mean over the common window;
  - s̄ is the mean over the common window of max(0, 0.98 − L_t/E_t), the T-bill
    ETF's share of equity beside the 2% buffer;
  - b̄ = max(0, 1 − L̄/E − s̄) is the idle cash left beside the ETF, which earns
    `buffer_cash_yield` = 0;
  - vol is whichever of the 12% target or T21's realized volatility on the
    common window gives the larger hurdle Sharpe; both are reported.
  - `tbill_hurdle` refuses an ETF share larger than the idle equity
    max(0, 1 − L̄/E). That can only happen if L_t/E_t exceeds 1 on some
    sessions; the summary then raises instead of reading a hurdle (N7).
- **Reading of SPEC §3.** The T-bill ETF is a long position. Its total return
  is in the account's return, and idle cash is only the buffer. That is the
  arithmetic of the formula above. The owner has not yet confirmed this reading.
- **Also reported, as cases rather than criteria**, each at both volatility
  bases and each with criterion 3's pass or fail, the excess over bills in %/yr
  (figure × vol − return hurdle) and the margin in %/yr:
  - broker cash yield 3.3% with no ETF: (rf − 0.033 · max(0, 1 − L̄/E)) / vol;
  - idle cash earning nothing: rf / vol (the convention under which
    `vol_momentum_v1` was decided).
- **Cash toll.** For every row: net long–short return per dollar of long
  exposure (annualized net Sharpe × volatility per unit of gross ÷ 0.5), against
  the cash yield a long dollar displaces, at the ETF's rf − 10 bp, at 0 and at
  3.3% (`prism.validation.hurdle.cash_toll`; borrow is already in the net
  returns, so none is added).

## 7. Deflation and breadth

- **Deflated Sharpe ratio** for T21 (`prism.validation.metrics.deflated_sharpe_ratio_with_n`)
  against the periodic net Sharpes of all 30 rows on the common window with
  N = 30, and against their effective count: max(2, the participation ratio of
  the 30 rows' daily net-return correlation matrix), through the probabilistic
  Sharpe ratio at the expected maximum Sharpe for that fractional N. Both are
  reported.
- Approval does not depend on it (replication route), but every claim carries
  it (N5).
- Inherited selections are disclosed and not pooled: `momentum_v1` (17 rows),
  `sizing_lab_v1` (8), `vol_momentum_v1` (1).
- **N6.** T21's effective number of independent positions, N_eff, computed as
  the participation ratio of the covariance of position-weighted daily return
  contributions over held names (`research/scripts/registered_read.py::contribution_breadth`,
  the estimator that reproduces the recorded B1 N_eff of 52.7), from T21's
  decided target carried forward between sessions. This is not the
  post-residualization covariance SPEC N6 names; T21 is a decile book with no
  residualization step. The ceiling is |IC₂₁| × √N_eff × √(252/21), with IC₂₁
  the mean 21-session rank IC below. A realized net Sharpe above the ceiling is
  flagged for a look for a leak; the flag establishes nothing by itself.
- **Horizon profile.** The rank IC of T21's decided target weights against each
  name's next h-session close-to-close return, over names with a non-zero
  target, at every decision session of the common window, for h = 1, 2, 5, 10,
  21 and 63 (SPEC §5): the mean over all sessions (overlapping windows) and a
  standard error from every h-th session.

## 8. What is measured and reported (all rows, on the common window)

1. **Timing luck across phases (P00–P20).**
   - Median, interquartile range and range of annualized net Sharpe, annual
     (geometric) return and maximum drawdown.
   - The standard deviation across phases of annual return, in %/yr.
   - Where T21 and P̄ fall among the phases' Sharpes.
   - Stated: a single-phase user carries this dispersion; a T21 user carries
     almost none.
   - Stated: 3.6 years of in-house data estimate it noisily.
2. **Tranching mechanics (T21 against the mean of the P rows).**
   - Annual turnover and cost drag.
   - Orders per session and median order notional, from the fills.
   - Wash-sale pairs: consecutive trades in one name of opposite sign within 30
     calendar days, counted from the fills (a tranched row's recorded target
     moves with prices without a trade, so targets cannot say when the account
     bought or sold); total, sell-then-buy count, and pairs per year.
   - The scale path of the account scale (the weight-averaged scale of the
     tranches holding a book): mean, minimum, maximum and fraction of sessions at
     the cap; for T21, the fallback count by reason.
   - Mean L/E and mean T-bill ETF share.
   - The Reg T path, counting the ETF as a long position: initial excess
     1 − 0.5 × (gross + ETF share) and maintenance excess 1 − 0.3 × (gross + ETF
     share), minimum and mean.
   - Stated: 50% and 30% are Alpaca's documented minimums; the ETF's margin
     treatment is unverified.
3. **Rounding by account size** (T21 at $25k, $100k and $1M; T3 at $25k).
   - Names held against names targeted.
   - The annualized tracking error of the realized gross return against the
     return of the decided continuous-share target, held from the session after
     its fill.
   - The paired net-Sharpe interval against T21 at $100k.
4. **Venue** (D-split against T21): tracking, the full criterion-3 arithmetic
   at every hurdle case, the paired net-Sharpe interval, and the tracking loss.
5. **Liquidity weighting** (D-liq against T21): as for D-split, plus, for both
   rows, each order's notional over the name's trailing 20-session median
   dollar volume: the median over all orders, and the median and 95th
   percentile over the orders whose name's median dollar volume is at or below
   the 10th percentile across all orders.
6. **Holdings diversity** (D-part1–3 against T21).
   - As for D-split, plus each row's loss in tracking correlation against T21
     and its cost in the economics figure (loss × prior).
   - Name overlap (Jaccard) between the partitions' kept universes, and the
     mean daily overlap of held names.
   - Aggregate per-name flow with deployments split evenly over the three
     partitions, against all on T21: the ratio of total flow and of flow^1.5
     (the weighting square-root impact gives).
7. **Flow across deployments** (arithmetic on the recorded turnover and fills):
   how evenly flow spreads over sessions (peak to mean, the busiest 1/21 of
   sessions' share, and a concentration ratio) for
   - (i) single-phase deployments that all start on one day (P00),
   - (ii) single-phase deployments with uniform random phases (the mean of
     P00–P20),
   - (iii) T21, and
   - (iv) T21's own first 21 sessions, the staged build a burst of new
     deployments would make together, against T21's mean.
   Also the fraction of one single-phase deployment's traded notional that a
   deployment on another phase trades in the same name, direction and session
   (mean and maximum over pairs of P rows). Two deployments on one phase, or two
   T21 deployments of one size, make the same trades by construction.
8. **The cap on public data.** The fraction of sessions on which the reference
   scale is ≥ 1.8 and at 2.0, over the full sample and from 2015-04-01. This
   describes the volatility series. It is reported, not a criterion.

## 9. Crowding and capacity scenarios (uncounted; cannot choose a configuration)

Following the precedent of `docs/cost_frontier.md`, this section re-prices the
recorded per-name trades of T21, P00, D-split and D-liq on the common window
(`--part capacity`, `prism.validation.capacity`). Decisions do not depend on
costs in this construction, so re-pricing keeps every decision; it ignores only
the small change different costs would make to the equity path and so to later
order sizes. It adds no ledger row.

- **Crowding cost per own dollar traded in name i on session t**, for each
  execution window that takes a fraction f of the order:
  Y · σ_i · √[f · q · (1 + φ · F / E) / (V_i · w)], where
  - q is the order's decided notional (quantity × the decision close), which is
    what the engine charges the participation term on;
  - σ_i is the name's trailing 20-session daily close-to-close volatility and
    V_i its trailing 20-session median dollar volume, both ending at the prior
    close and needing 10 valid sessions (otherwise the name's last earlier value
    is carried and counted);
  - w is the window's share of daily volume;
  - F is aggregate follower capital and E own equity;
  - φ is the synchronized fraction.
  The stack's own participation charge is recomputed on the same trades and
  replaced by the scenario's cost, not added to it.
- **Grid:**
  - F ∈ {0, $1M, $10M, $100M, $1B, $10B};
  - windows: open at 1%, open at 3%, close at 8% and the full day at 100% of
    daily volume, each alone, for the next-open rows; for D-split, half at the
    open (1% or 3%) and half at the close (8%);
  - synchronization: P00 with φ = 1 (everyone on one phase) and φ = 1/21
    (uniform phases), T21 with φ = 1, D-split and D-liq with φ = 1;
  - Y ∈ {0.5, 0.7, 1.0}.
- **Reported for each scenario:** own impact cost and cost with followers in
  %/yr, the crowding drag, the excess over the row's governing hurdle (its own
  criterion-3 arithmetic), the break-even F and that capital as a number of
  users at $100k and at $1M, the largest multiple of own capital with no
  followers at which the excess stays ≥ 0, and the share of cost in names under
  $100M daily volume. Also, per row, the binding names: the count and list of
  traded names whose median dollar volume is under $100M, and the 5th and 10th
  percentiles of traded names' median dollar volume.
- **Labels.** Every coefficient and window share is labelled an assumption:
  the auction shares come from 2015–17 industry notes and are not checked for
  2026. Paper fills cannot measure impact.

## 10. Forward stop rule (runs from this commit; also after approval)

The reader that applies this rule is not part of this commit's driver; it will
be committed before the first read and cannot change the rule.

**Forward stream.** T21 re-run with the canonical engine on data after the
registration commit, by the `forward_v1` method (`docs/forward_design.md`: S&P
500 point-in-time membership rebuilt at each as-of date and Twelve Data daily
bars frozen into a dataset). Earlier bars enter only the formation and σ̂
windows. If the owner arms T21 on a paper account, that stream is read too:
- the T-bill ETF's distributions are added from `data/distributions/SGOV.csv`
  (not in the repository at this commit; it must exist, with its provenance,
  before a paper-stream read), because paper accounts do not simulate
  dividends;
- borrow is charged at 50 bp/yr, because paper accounts charge none.

**Reads.** The first at the first as-of date with at least 126 forward
sessions, then every 63 sessions. Any of the following removes eligibility;
reinstatement is a new registration.

- (a) The forward net Sharpe's 95% interval lies entirely below zero.
- (b) Forward drawdown from the forward peak exceeds 25%.
- (c) Daily correlation with the scaled reference on the latest French vintage
  (hash recorded) is below 0.5.
- (d) On two consecutive reads, the forward tracking correlation × the
  post-publication Sharpe on the latest vintage − the forward cost drag is below
  the governing hurdle.

(a)–(c) are carried from `vol_momentum_v1`.

Report only:
- (e) an empirical-Bernstein anytime-valid read (bound 0.1, α = 0.05) of the
  daily account excess over bills (T21 net + the ETF share at rf − 10 bp − rf),
  against zero, with the expected detection horizon printed beside it. The
  planning arithmetic put it at about 29 years to reject a zero Sharpe at 0.30
  and about 250 years to detect the expected margin over bills; those are
  derived from recorded figures, not measured.

## 11. Outputs

- Ledger: `trials/public_baseline_v1.jsonl`.
- Results under `results/2026-10-public-baseline/`:
  - `summary.json`: the six criteria and verdict, every input and hash (French
    files, dataset, cost stack; per row its run id, `config_id`, `data_id`, code
    commit, specification hash, record hash, fold-table hash, fill model and
    return convention), every reported case of §§6–8, and every attempt;
  - `series.csv`: the scaled reference (scale, return) and T21 (net return,
    cost, scale, gross, target gross, L/E, ETF share) per session;
  - `rows/<id>/`: `row.json` (record, mechanics, memory), `spec.toml`,
    `series.csv`, `targets.parquet`, `holdings.parquet`, `fills.parquet`,
    `sizing_decisions.csv`, and `partition_kept.txt` for D-part rows;
  - `parity.json`, `folds.toml`, `phase_dispersion.json`, `flow.json`,
    `capacity/` (`summary.json` and one table per row), `invocations.jsonl`
    and `memory.log`.
- The engine also writes each run's record under `results/<run_id>/` and its
  working files under `state/`, `runs/` and `logs/`.
- Read record: `docs/public_baseline_read_<date>.md`, written whatever the
  outcome.
- Driver: `python -m research.scripts.public_baseline --part rows|summary|capacity|all`
  (`--part check` validates without evaluating). It refuses to run unless this
  file, the TOML, the cost stack, the four policy files and the parity series
  are committed unchanged at HEAD; it then checks the French files' hashes and
  the dataset's manifest id and vintage. It records HEAD's commit in every row
  record; it does not check for, or record, uncommitted changes to code, so the
  run is to be made from a clean checkout.

## 12. What this does not do

- It does not arm any sizing rule, hold any ETF, change the live book, edit
  SPEC §6 or `docs/sizing_preregistration.md`, or choose a risk profile.
- It does not reopen or extend any closed set: `vol_momentum_v1`,
  `fundamentals_v1`, `earnings_v1`, `sizing_lab_v1`.
- It changes no clock: `forward_v1`, M6, T5 and the `vol_momentum_v1` stop rule
  keep running as registered.
- It tests no other theme, target, lookback, cap or crash rule.
- It does not adopt closing-auction orders, crowded-date avoidance, a T-bill
  holding funded by short proceeds, or full-day execution. Each of those is a
  new registration.

## 13. Method limits, stated before the result

- The in-house sample is 3.6 years. The phase dispersion and every paired
  interval are imprecise.
- In-house books are price returns, the public series total returns. The
  dividend wedge (−36 bp/yr) is not charged.
- Membership coverage is 0.907. The missing names flatter tracking and cost.
- **P00 is not `vol_momentum_v1`'s computation.** The parity gate bounds these
  differences; it does not remove them:
  - σ̂ uses the base book's returns gross of costs at constant weights, while
    `vol_momentum_v1` used the recorded book's net returns;
  - P00 trades its scaled book in whole shares inside the engine, while
    `vol_momentum_v1` multiplied the unscaled book's recorded net return by the
    scale and charged 10 bp one-way on |Δscale| × gross for each change;
  - P00 is charged `retail_v3` (participation term, 6.5% financing on debit
    cash); the parity book was charged `retail_v1`, which has neither;
  - P00 starts empty on 2022-03-29, while the parity book had been trading since
    2021-03-30;
  - the per-name cap is 0.10 here and 0.35 in the control; at about 50 names a
    leg neither binds.
- The French files pinned here were already read by another package before this
  commit (§2).
- The T-bill holding is not simulated. Its yield is an assumption until the
  forward stream measures it.
- The experiment profile's preference limits, including max gross 1.0, are
  recorded rather than enforced; a batch repaired for another reason can be
  trimmed toward them (§1.1).
- The driver does not re-hash the dataset's files at run time (§3).
- Paper fills are not the modelled open fills: 74% are morning sweep orders.
- §9 is driven by its assumptions, not measured.
- French rewrites history with each vintage. The prior is pinned here, and
  stop-rule reads use the latest vintage.
