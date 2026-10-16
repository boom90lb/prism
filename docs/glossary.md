# Glossary

Prism's records were written over several months by one project, and they use
in-house words for ordinary quantitative and engineering ideas. This page maps
each of those words to the standard term. Use it when a record, a code comment
or a command's output uses a word that is not explained where it appears.

Identifiers are never renamed: run ids, experiment ids such as `momentum_v1`,
file names such as `results/demotion_b1/` and the tier names in records keep
their original spelling, because other records cite them. Newer documents use
the plain term and give the identifier beside it.

The [operating contract](../SPEC.md) is the authority for the rules these terms
name. Where this page and the contract differ, the contract governs.

## Experiments and approval

| Term in the records | Plain meaning |
|---|---|
| registration, registered, pre-registration | A document committed before an experiment runs. It fixes what will be tried, the data, the universe, the cost assumptions, and the rules that will stop or approve a configuration. The commit that adds it is the registration ([SPEC §4](../SPEC.md#4-the-trial-ledger-is-the-whole-authorization-system)). Files follow `docs/*_design.md`, `docs/*_preregistration.md` and `registrations/*.toml`. |
| selection set | The complete set of configurations tried in one registered experiment, named by an id such as `momentum_v1` or `vol_momentum_v1`. Multiple-testing adjustments are computed against it. |
| trial, counted trial, cell | One evaluated configuration. Every trial is a row in its experiment's ledger, including failed, NaN and degenerate runs. |
| trial ledger, trial log | The append-only record of every evaluated configuration: JSONL files under `trials/` and `results/*_trials.jsonl`, and the hash-chained log the engine writes for its own runs (`prism.validation.trials`). Editing or deleting a row is falsification. |
| arm | One variant inside an experiment. |
| closed set | A completed experiment. Nothing new may be added to it; a new idea needs a new registration. |
| kill rule | The stop criterion stated in a registration. When it fires, the configuration loses its approval or the experiment closes. |
| promotion rule, promote, promoted | Approval under SPEC §4 that makes a configuration eligible to become the live paper book. It is revocable by the kill rule, it is not deployment, and it moves no money. |
| replication route | The approval path for a published factor implemented as its source defines it. It relies on the factor's long public return series (Kenneth French, AQR, JKP, Chen–Zimmermann) plus tracking and cost tests on in-house data. |
| discovery route | The approval path for anything the project found by searching data. It needs data from after the registration date and a deflated Sharpe ratio above 0.5. |
| same-sample | Evaluated on the same data the idea was found on. A same-sample result can reject a configuration and can never approve one. |
| read (noun), read record, `*_read_<date>.md` | The results report of a registered experiment, written whatever the outcome. |
| forward read, forward stream | An evaluation on data that arrived after the registration: an extension of the backtest, the paper account's record, or both. |
| fragility read | A robustness test: does the result survive small, pre-stated changes to the configuration? |
| admission, convexity admission | A diversification test for adding a strategy to an existing book, for example whether it earns money in the existing book's worst periods. |
| uncounted diagnostic | A measurement that adds no ledger row and may not be used to choose a configuration. |
| certification, certified | A result published together with artifacts that may never be rewritten (SPEC §4). The only one, [certification 001](certifications/001-residual-reversion-daily-negative.md), is a negative result: daily residual reversion is uneconomic at retail cost. |
| ratified, RATIFIED | Approved by the owner. Before 2026-09-02 approval was a separate step with its own commit or tag; the current contract needs none. |
| owner act, owner decision | A decision reserved to the maintainer who owns the accounts: opening or funding an account, arming a sizing rule, changing the live book, publishing a release. No code change or review can make it. |
| constitution, doctrine, amendments, handoff, program document | Governing documents that [SPEC.md](../SPEC.md) replaced on 2026-09-02. Older records cite them; none of them is in the repository any longer. |
| launch brief, Q1–Q20, `[Q7]` | The numbered questions of [the engine design brief](prism_engine_decisions.md) of 2026-09-06. |
| engine specification, spec §3.x | Sections of the engine overhaul's implementation specification. It is an unpublished draft that is not in the repository, so code comments that cite it cannot be followed from a clone. |
| stage 0–6 | The delivery phases of the 2026-09 engine overhaul, all complete ([completion record](engine_overhaul_completion_2026-09-15.md)). Older messages and documents use them as a schedule. |
| Tested, Run, Drafted, Assumed | Status labels on claims (SPEC §7): tests exist and pass; executed, correctness not asserted; written, not executed; not checked. |
| receipt | A recorded output of a verification run, such as [the onboarding receipt](onboarding_verification.json). |
| erratum | A correction to how a recorded result is described. It changes no result, verdict or ledger row ([2026-10-14](erratum_2026-10-14.md)). |

## Statistics

| Term in the records | Plain meaning |
|---|---|
| claim tier, claim ladder | How strong a result is, from weakest to strongest (SPEC §3): `no_claim`; `mechanics_clean`, the pipeline ran and the accounting conserves capital; `gross_edge`, profitable before costs and the breadth checks pass; `net_edge`, profitable after modeled costs; `robust_edge`, also significant after the multiple-testing adjustment, or passing the replication tests of the published-factor route. In the momentum record's rules the adjustment means a deflated Sharpe ratio of at least 0.95. |
| DSR | Deflated Sharpe ratio (Bailey and López de Prado, 2014): the probability that the true Sharpe ratio is above zero after correcting for the number of configurations tried and for non-normal returns. |
| hurdle, after-cost hurdle | The T-bill benchmark expressed as a Sharpe ratio: what a strategy must earn to beat holding bills, after crediting what its idle cash earns. The formula is in SPEC §3 and the code is `prism.validation.hurdle.tbill_hurdle`. |
| cash convention | Which cash earns interest in a backtest. From 2026-09-25, idle own cash earns the account's cash yield and short-sale proceeds earn nothing; earlier experiments assumed idle cash earned nothing. |
| prior, post-publication prior | The expected Sharpe ratio of a published factor after its publication date, measured on its public return series: the post-publication Sharpe when ten or more years exist, otherwise 0.4 times the full-sample Sharpe. |
| publication bar | The requirement that a published factor's public series has a t-statistic of at least 3 over at least 20 years. |
| fidelity | Tracking correlation: how closely the in-house implementation's returns follow the published factor's returns. |
| F3, "the figure from criterion 3" | Expected net Sharpe after decay and costs: tracking correlation × post-publication prior − cost drag. It is criterion 3 of the replication route. |
| cost drag | Annualized modeled cost divided by annualized volatility, so that it is in Sharpe units. |
| N_eff, breadth | The effective number of independent bets: the participation ratio of the covariance matrix after factor exposures are removed. |
| IC, rank IC | Information coefficient: the rank correlation between a signal and the returns that follow it. |
| IC·√N_eff ceiling | The fundamental-law upper bound on the Sharpe ratio implied by the IC and the breadth. A realized Sharpe above it points to look-ahead or a bug. |
| purged, embargoed walk-forward | Out-of-sample testing that trains on the past, tests on the next period, drops training observations whose outcome windows overlap the test period (purge) and leaves a gap after it (embargo). |
| fold table | The frozen, content-hashed table of training and test date ranges an evaluation uses. |
| bucket spreads | Half-spread cost assumptions per liquidity bucket (`SPREAD_BUCKET_SCHEDULE_V1`). |
| next-open fill | An order decided at the close of day *t* is filled at the open of day *t*+1 or later, in backtests and in live trading. |
| price return, total return | Returns without or with dividends. The in-house books are price returns; the public factor series are total returns. |
| membership coverage | The fraction of an index's past and present members for which usable prices exist; 0.907 for the momentum record. Missing names are mostly delisted ones, which flatters a backtest. |
| L/E, long side over equity | The long market value divided by account equity. It sets how much idle cash the account has. |

## Strategies and recorded results

| Term in the records | Plain meaning |
|---|---|
| B1, `demotion_b1` | The 12−1 cross-sectional momentum backtest on the S&P 500: buy the top decile and sell the bottom decile by trailing twelve-month return, skipping the latest month, rebalanced every 21 sessions. It is recorded at `results/demotion_b1/` and is the evidence for the paper book. |
| demotion, `demotion_design` | The seven-trial experiment of July 2026 that followed the stop of the residual-reversion program ([design](demotion_design.md)). Its rows are D1–D5, B1 and B2. |
| D1–D5 | Daily residual-reversion variants of that experiment. All were found uneconomic ([certification 001](certifications/001-residual-reversion-daily-negative.md)). |
| B2 | The residual-reversion and momentum targets summed into one book. A diagnostic, also uneconomic. |
| live book, reference book, reference strategy | The configuration the paper account trades: unscaled B1 momentum on the current S&P 500. |
| M1–M6, M-series | The momentum robustness tests: M1 no skip month, M2 six-month lookback, M3 20% deciles, M4 quarterly rebalancing, M5 two-month skip, and M6, a re-run of B1 on data through at least 2027-06 ([design](momentum_design.md)). |
| T0–T5, T-series | The trend-following tests on ETFs ([design](trend_design.md)). T5 is a re-run on later data. |
| X0–X4 | The learned cross-sectional ranking tests ([design](learned_xsection_design.md)). |
| `vol_momentum_v1`, volatility-scaled momentum | B1 scaled to a 12% volatility target by the Barroso and Santa-Clara (2015) rule, with a 126-session window and a 2× gross cap. It was approved on 2026-09-24 under the replication route by a margin of 0.018 Sharpe over its hurdle, and it is not traded ([results](vol_momentum_read_2026-09-24.md)). |
| `forward_v1` | The forward evaluation of the two residual-momentum candidates ([design](forward_design.md)). |
| sleeve | A sub-strategy inside a larger account, such as "the trend sleeve". A "cash sleeve" is idle equity held in a T-bill ETF. |
| de-gross, crash-conditional de-gross | Cutting gross exposure. The crash-conditional rule is the sizing rule of [the sizing registration](sizing_preregistration.md); public data contradicted it on 2026-09-24. |
| arm (a control), arming | Switching on a default-off control or sizing rule in live operation, by its own commit. An owner act. |

## Engine and operations

| Term in the records | Plain meaning |
|---|---|
| book, target book | A portfolio; the target book is the set of target weights a decision produces. |
| engine, production spine | The `prism` package under `src/prism`, which ships as the `prism-trading` wheel. `research/` imports it and is never imported by it. |
| specification, spec | The TOML file that declares a strategy: universe, components, construction, cost and risk policies (`prism.spec`). |
| `config_id` | The hash that identifies a resolved specification. It covers the engine version and the content of every policy file the specification uses. |
| component, node | A signal implementation, such as `prism.signal.momentum_node`. |
| canonical engine | The share-holding simulator (`prism.execution.simulate`). Weights drift with prices between decisions, and cash is a real balance. |
| `historical_constant_weight` | The constant-weight simulator that reproduces the recorded results. It re-levers to its target weights every day without charging for that trading. |
| workspace | The only directory the engine writes to: data, logs, results and state. |
| dataset, manifest, vintage | A dataset is the workspace's tables of bars, membership and events; `manifest.json` identifies it; the vintage labels one frozen version of it. |
| research_vintage, replay_recorded | The two ways to read a dataset as of a date: what a frozen vintage says was known then, or what this workspace had actually ingested by then. |
| engineering channel | Runs that check machinery. Their records are capped at `mechanics_clean` and excluded from trial counts. |
| claim packet | The result manifest of a recorded run (`claim_packet.json`): configuration, data, costs, statistics, tier and code commit. |
| result record | The engine's record of one evaluation (`record.json`, `prism.validation.trials.ResultRecord`). |
| run directory | The directory holding one paper or live book's ledgers and state, such as `runs/paper_loop_momentum2`. `runs/ACTIVE_RUN_DIR` names the one the scheduled jobs trade. |
| `KILL_SWITCH`, `ALERT` | Files in a run directory. `KILL_SWITCH` stops order submission; `ALERT` marks an unhealthy session. |
| doctor, `prism-doctor` | The health check that ends every scheduled job. Its exit code is the alert condition. |
| write-ahead | The loop saves its decision before it submits any order, so a restart resumes rather than trading twice. |
| sweep | The morning job (06:50 PT, 09:50 ET) that resubmits unfilled opening-auction orders as market orders. |
| concordance | The ledger of differences between held and target positions. |
| regime layer, regime telemetry | Market and macro state (yield curve, volatility, liquidity) recorded each session. It conditions sizing only after a test shows it helps. |
| regime clock | The count of consecutive clean paper sessions with regime telemetry. The deployment gate needs 21 (SPEC §6). |
| deployment gate | The conditions in SPEC §6 that must all hold before any real-money order. |
| parity gate | The check that an engine upgrade decides the same orders as the previous version over recorded sessions. |
| capture | A directory of recorded sessions that `prism upgrade-diff` compares. |
| authorization record | The file in a run directory, written by the account owner, that permits a given process, host and configuration to trade a given account. |
| S0–S9 | Operating states of the authorization layer (`prism.live.authorization.OperatingState`). For example S4b means no authorization record exists, so no broker write is permitted. |
| C0–C5, OB | Classes of broker action the authorization layer permits or refuses: C0 cancel an open order, C1 reduce a position, C3 open or add, C4 and C5 hedges, OB obligations such as covering a short. |
| observatory | Tools for capturing point-in-time observations. `research/observatory.py` stores captured news and filing counts. Older documents also use the name for captures run from a separate `prism-observatory` repository, which is not part of this one. |

## The invariants N1–N8

SPEC §2 names eight rules every change must keep. Records cite them by tag.

| Tag | Rule |
|---|---|
| N1 | No look-ahead: a value for day *t* uses only information available at *t*'s close. |
| N2 | Next-open fills: decided at the close, filled at the next open or later. |
| N3 | Costs before edge: spreads, impact, commission, borrow and participation limits are charged; gross returns are never a claim. |
| N4 | Capital conservation: equity moves only by profit and loss minus charged costs, proved in Lean (`formal/`) and property-tested. |
| N5 | Every claim carries its multiple-testing adjustment. |
| N6 | Breadth accounting: every cross-sectional claim reports its effective number of bets and the bound it implies. |
| N7 | Fail loudly: a failed fetch, an unfit model or a degenerate estimate raises or cuts exposure explicitly; a silent zero is a defect. |
| N8 | Import hygiene: the production import path loads no heavy research library, and `prism` never imports `research`. |
