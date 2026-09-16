# Prism engine: finalized question answers and launch brief

Date: 2026-09-06. Repository inspected at `5fcac33869e9c8c35362c98b00924b3b3f5afafb`.

**Decision status: finalized as the input to Fable's specification and implementation plan. Implementation status: Drafted.** Performance and onboarding targets below are acceptance goals, not measured achievements.

This document consolidates the product discussion, Fable's draft 0, the corrected construction analysis, and the reviewed Gemini advice. It replaces their competing working assumptions for the new design. Fable can make routine implementation choices within these decisions without reopening the product questions. Empirical findings, external package eligibility, and actual capital decisions remain distinct from design choices.

The current operating contract continues to govern the existing system until its focused migration is applied. This brief is the input to that migration and the new specification, not another permanent layer of governing prose.

The product promise is to make private strategy research, comparison, and operation substantially easier to do correctly. Users should improve the quality and speed of their decisions, preserve the evidence behind them, and receive useful engine improvements through ordinary package upgrades. Realized returns and expected Sharpe are not guaranteed to improve monotonically.

## Q1. Who is the user of the public engine?

**A Python-capable independent researcher, often assisted by an AI agent, developing private strategies.** Small teams should fit the same workflow. Optimize for users who want reliable research and operation without maintaining an entire trading infrastructure.

Use $25,000–$1,000,000 as the initial live-account design range, not an eligibility rule or a promise that every strategy works throughout it. Research and paper operation require no investment capital. Include a clearly illustrative long-only ETF example for smaller accounts alongside the momentum reference. Capital suitability belongs to a strategy, account, broker, and execution profile together; Q22 defines how it is established.

## Q2. Is the engine or the momentum book the product?

**The engine is the product. Momentum is its reproducible reference strategy and construction control.** A user must be able to substitute a private signal, dataset, risk policy, and cost assumptions without editing engine internals.

Keep three boundaries: a public versioned engine, private strategy projects that depend on it, and a public reference project using the same extension interfaces. Version the reference configuration independently from the engine. An engine release does not implicitly change the reference strategy or promote a new configuration.

## Q3. What should the release produce for the owner?

**Successful private users, supported by the owner's daily use of the same interfaces.** Credibility follows from reproducible evidence and dependable operation; contributions should improve that experience.

Initial acceptance goals are a credential-free example in 15 minutes, a first private signal evaluation in one hour once data is available, and three independent users completing the workflow without engine edits. Record hardware, dataset size, and excluded download time when measuring these goals. Track upgrade effort, failed operating sessions, and time spent on integration alongside onboarding.

Keep private code, trials, fills, and datasets local by default. Later, offer an opt-in contribution path for narrowly described negative results and appropriately aggregated execution observations. Treat privacy and data rights explicitly; anonymization is not automatically sufficient. The shared knowledge service is not a first-release dependency.

## Q4. Which strategy class is in scope for version one?

**Daily-bar US equities and ETFs, long-only or long-short, with one account-level portfolio and one decision schedule per strategy configuration.** Several components may contribute at that schedule when their forecast validity supports it.

Support both a simple constructor and a convex constructor behind one interface. Keep single-period optimization initially. Forecast horizon, update schedule, and estimated decay are separate fields. A one-day forecast cannot silently remain active through a monthly holding interval.

Retain the current exclusions for intraday alpha, options, direct futures, cash FX, and crypto. Intraday order reconciliation and execution recovery are operating mechanics, not intraday alpha. Multi-period control, tail-measure optimization, tax optimization, learned-model expansion, and compiled simulation kernels require later measured justification. Preserve event detail needed for future accounting and reporting extensions.

## Q5. Does the public engine support real-money operation?

**Yes. Shared execution safeguards are part of the product.** Default the mode to paper, but require an explicitly named account and verify its actual identity and endpoint. An environment variable alone cannot switch operation to real money.

Live authorization binds the account, strategy code and configuration, engine version, and declared risk and operating policy. The order path enforces that policy and checks reconciliation and current broker restrictions. Local authorization establishes what the account owner approved; it is not a profitability certificate or a sandbox against arbitrary private Python code.

Separate authorization to add strategy exposure from authorized cancellation and recovery. A missing signal or expired research approval must not disable management of existing positions. A recovery action still needs trustworthy inputs and must be executable.

The owner's existing deployment rules, including the clean-session requirement, risk-profile selection, and separately enabled sizing policy, remain the owner's project policy. The existing micro-account exception retains its measurement-only purpose and owner-controlled account actions. Public live support does not itself deploy the current book.

## Q6. Which brokers, data sources, and execution capabilities are required?

**Alpaca execution; preserve Alpaca and Twelve Data ingestion; add documented local Parquet/CSV ingestion.** Extend the existing interfaces where possible. Add a broker when a real operating project requires one. Add a data provider when it closes a measured coverage or quality gap, including for the owner's project. The owner's existing approximately $1,000/year data budget remains a project constraint.

Record an execution profile: quantity increments, minimum sizes, fractional eligibility, shortability, permitted order types and sessions, price conventions, and account restrictions. Distinguish quantity increments, round-lot classification, and auction eligibility. Revalidate changeable capabilities and buying power before submission; historical assumptions belong in the evaluation record.

As checked on 2026-09-06, FINRA's new intraday-margin framework took effect June 4, with a transition period for brokers through October 20, 2027. Alpaca has deprecated its PDT fields. Do not encode a universal $25,000 PDT rule. [FINRA notice](https://www.finra.org/rules-guidance/notices/26-10), [Alpaca API change](https://docs.alpaca.markets/us/changelog/2026-06-03-pdt-f3c04da)

Alpaca documents fractional orders with day time-in-force and no fractional short sales. Model only supported combinations, and verify adapter behavior against current documentation. No blanket fractional-share spread surcharge or assumed auction fill is adopted. Costs require calibration or explicitly labeled sensitivity assumptions. [Alpaca fractional trading](https://docs.alpaca.markets/us/docs/fractional-trading)

## Q7. What does the engine enforce, and what does a project declare?

**The engine enforces mechanical correctness and the policies a project explicitly selects. Projects choose economic and statistical policy.**

| Engine responsibility | Project decision |
|---|---|
| Information-availability checks, purged/embargoed walk-forward, train-only fits and causal cache access | Features, estimators, training windows, forecast horizons |
| Close-time decisions with next-open-or-later execution | Decision schedule and supported execution policy |
| Cash, positions, events, fills, and cost accounting | Forecast assumptions, risk budgets, benchmarks |
| Consistent construction/simulation cost semantics and recorded calibration | Cost values, uncertainty methods and confidence levels |
| Independent pre-trade and post-rounding validation | Mandatory limits, preferences, recovery priorities |
| Durable state, explicit account identity, broker reconciliation | Registration, search budgets when used, promotion and monitoring policies |
| Automatic result/trial recording, provenance and reproducibility | Which configuration to operate and when to adopt an upgrade |

Exploratory evaluation may run without a promotion policy and records that fact. If a project selects strict registration or a budget, enforce it; do not merely warn and continue. Missing cost inputs prevent a complete net claim. Policy changes receive new identities and cannot silently rewrite historical evidence. Arbitrary private plugins are trusted executable code; interface restrictions and conformance tests do not prove that they cannot fetch future data elsewhere.

Use one shared decision assembly for evaluation and operation: validated inputs, fitted state, portfolio state, configuration, risk, and costs produce a target and diagnostics. Simulation generates fills; operation observes broker fills. Sharing code does not remove that boundary. Modeled costs and observed execution costs must be compared without charging spread or impact twice.

Classify constraints by meaning. Physical action restrictions and mandatory account requirements cannot be waived by a penalty. Preferred holding or turnover limits may have declared tradeoffs. When current holdings breach limits, enter a separate recovery path: identify executable actions, reduce authorized violations in declared priority order, then optimize economic preferences within the achieved limits. Use successive optimization stages when priorities must be strict. Report existing breaches, proposed changes, and newly introduced violations separately. No-trade and the zero portfolio are not universally safe, feasible anchors.

Apply observed fills and confirmed corporate actions to state. Recalls and delisting notices create obligations or events, not invented fills. If no executable improvement can be established, use the declared cancellation, hold, or escalation response. Solver status alone never authorizes an order; validate finite outputs, constraints, cash, and tolerances independently.

Every evaluated configuration and failed/degenerate attempt is recorded automatically with its selection-set identity. Preserve complete histories across repository splits. Distinguish replay from new configurations; repetition creates no new independent evidence. Report observed trial counts, dependence adjustments and their method, and unobserved-search limitations. An append-only log or hash chain detects changes only relative to a trusted record; it does not prove the search was exhaustive.

## Q8. What confidence level and mapping does construction use?

**No universal construction confidence level. Register exactly three initial comparison arms:** the existing decile constructor; a convex constructor using a calibrated, shrunk point forecast; and that convex constructor with a declared 80% one-sided uncertainty adjustment. Report all three. Confidence is not selected afterward by whichever backtest wins.

Freeze the score panel, universe, outer folds, executable return convention, and charged cost model across the comparison. The two convex arms share their mapping, risk model, and declared construction settings. Account-size scenarios are recorded conditions of the experiment. Any additional parameter search is declared and included in its selection history.

The initial mapping is a train-only pooled slope from a declared standardized score to volatility-normalized forward returns, with shrinkage toward zero specified before the experiment. In the simple shared-coefficient model, write expected returns as `mu = theta * a`, where `a` contains the score and horizon-matched volatility. Rank IC remains a diagnostic; it is not substituted for the regression coefficient without establishing the required relationship. For learned signals, calibration uses inner out-of-sample training predictions, not fitted training predictions. No outer test outcome fits the mapping.

Estimate coefficient uncertainty using preregistered time-block resampling that preserves each sampled cross-section and refits calibration. Declare treatment of overlapping labels, block selection, shrinkage, and limited sample size. Use lower and upper one-sided 80% coefficient bounds as the adverse endpoints for positive and negative aggregate signal exposure respectively. State their directional coverage; they do not constitute a simultaneous 80% interval, and their actual coverage depends on the estimation assumptions. Tail-shape fitting to individual residuals does not automatically establish coefficient uncertainty. Student-t, Cornish–Fisher, EVT, or alternative calibration methods are later research choices.

Preserve the uncertainty structure. For a symmetric shared coefficient interval `theta_hat +/- r`, worst-case expected portfolio return is `theta_hat * (a' w) - r * abs(a' w)`. The independent per-name box penalty `r * sum(abs(a_i * w_i))` represents a different uncertainty set. Use the shared-coefficient interval initially, allowing asymmetric endpoints when produced by the declared method. Conformal outcome widths are not automatically confidence intervals for conditional means.

Use a factor covariance model with diagonal residual risk and declared shrinkage as the initial risk representation. Hold it fixed across the two convex arms. A dense shrunk residual is a separate research alternative. Include the same linear costs and convex three-halves total-impact term the simulator charges, cash and financing conventions, and borrow over the declared horizon. Unspent cash is an allowed economic outcome subject to project constraints.

Same-sample improvements do not promote the reference book. Use the existing prospective research rules for promotion. Do not infer a live decision, guaranteed empty portfolio, or round-trip break-even result from the earlier IC arithmetic table.

## Q9. What if the fitted mapping contradicts the declared signal sign?

**Disable that component's new directional exposure and record the condition; never silently invert it.** A finite contradictory estimate is a research outcome, distinct from failed estimation or invalid data. Continue evaluation through the declared disabled-component behavior and retain its result in the trial history.

Existing positions follow an explicit retention or unwind policy subject to executable constraints. Missing opinion is distinct from an intentional zero target. A negative confidence endpoint expresses uncertainty, not automatically a failed fit. Learning or reversing signal sign is permitted only as a declared research choice with its own evaluation.

## Q10. Can reference results with survivorship gaps be published?

**Yes, as conditional historical evidence with prominent coverage limitations.** Publication does not make the data complete or qualify a strategy for promotion. Record known membership coverage, missing securities and terminal outcomes, return conventions, and which conclusions remain untested. Unknown bias is disclosed, not described as quantified.

The first data contract includes a small security master: stable internal security identifiers, dated ticker/vendor mappings, and distinct share-class/listing semantics. Separate bars, membership/eligibility history, and corporate actions. Keep event/effective time, source availability time when established, local ingestion time, source identity, and revisions. An as-of query selects one complete revision with deterministic tie rules. Download time cannot manufacture historical knowledge timestamps.

Missing bars never erase a holding or simulate a sale at the last close. Track trading cessation, OTC continuation, share cancellation, distributions, and unresolved claims explicitly. Retain valuation uncertainty; strict claims that require an unknown outcome remain incomplete. Never replace an earlier closing price with an eventual payout. [CRSP delisting definitions](https://www.crsp.org/crsp_pdf/crsp-us-stock-indexes-databases-data-descriptions-guide-crspaccess/)

Use content-hashed columnar caches and manifests from the first milestone. Include data revisions, universe, transformations, fit cutoff/state, feature code and parameters in cache identity as applicable. Hashes establish identity, not truth. New source revisions create new inputs and derived results; preserved historical records remain reproducible rather than being rewritten.

## Q11. Should the repository ship market data?

**Ship deterministic synthetic fixtures immediately and a small real-data fixture only when redistribution rights are established.** Synthetic fixtures exercise changing membership, corporate actions, missing/revised observations, halts, costs, rounding, and recovery. Label synthetic results everywhere they could be mistaken for market evidence.

Provide real-data fetch commands, source manifests, and local-file ingestion. A permissibly licensed real fixture is valuable but is not a dependency for the credential-free example. Keep the data layer usable with the owner's affordable sources while reporting their limitations.

## Q12. Does the owner's research become a private project?

**Yes. The owner uses the same pinned-package workflow as public users.** Public code holds the engine, reference project, template, and preserved research history. Active private work holds proprietary signals, registrations, costs, policies, trial histories, and operating state.

Migrate incrementally through evaluation and recorded-decision parity. No phase deliberately breaks active research. Preserve selection-set lineage and immutable artifacts; a reproduction run does not reset the trial log. Explicit workspace paths replace package-relative data/results locations. Importing the package never creates directories, fetches data, or reads broker credentials.

## Q13. What happens to legacy models and bespoke evaluation drivers?

**Replace useful behavior before removing its implementation, then archive unused research.** Reproduce representative historical momentum and trend cases through the generic evaluator before retiring their drivers. Preserve tagged source, dependency information, data identities and reproduction instructions; document external data dependencies that cannot be redistributed.

Archive unused forecasters and RL experiments outside the supported default workflow. Evaluate the ensemble separately: lightweight combination primitives can remain public, while specific learned models belong in optional strategy packages. Dependency declarations follow audited imports, including lazy imports. Keep removals, extraction refactors, and behavioral fixes separately reviewable. Historical results and cited artifacts retain their identity and links.

## Q14. Do the Lean proofs stay?

**Yes. Preserve their current scope and maintain the implementation bridge.** Update bridge tests if accounting representations change. Extend executable accounting coverage to corporate distributions, short liabilities, price changes, duplicate fills, and interrupted replay where relevant.

Formal algebra, implementation tests, and market assumptions have separate evidence roles. No new formal-research program is required for this overhaul. Describe the exact theorem and implementation checks supporting each accounting claim.

## Q15. What are the dependencies and numerical implementation?

**Python 3.12+, NumPy, pandas, PyArrow and SciPy form the shared data/evaluation base.** Retain requests and python-dotenv where integrations use them, with explicit initialization. Remove unrelated model dependencies from core only after a complete import audit. Keep the production import path free of the existing prohibited research stacks; optional packages load only when used.

Put CVXPY and Clarabel together in a `construction` extra, included in the recommended optimizer installation. Explicitly select Clarabel and initially use the same formulation in evaluation and operation. Avoid separate research and production solver implementations. Clarabel supports the needed cones; CVXPY exposes compilation controls, including `ignore_dpp`, that must be benchmarked before replacement. [CVXPY solver documentation](https://www.cvxpy.org/tutorial/solvers/)

Profile first compilation, repeated solves, memory, and complete evaluations on declared hardware and shapes. Compare DPP and non-DPP compilation and appropriate native cone formulations. The earlier synthetic OOM is not evidence that 3,000 names inherently require direct assembly. Do not carry that conclusion into the roadmap.

Use factor/diagonal operators without constructing dense diagonal matrices. Reuse problem shapes when beneficial, but do not fix the universe dimension by mandate. Held names leaving eligibility remain in state with valid trading restrictions; a zero holding mask must not force an unavailable liquidation. Use stated numerical tolerances and deterministic ties. A tiny regularizer is a declared modeling term, not permission to ignore solver error.

Continuous construction followed by deterministic quantity rounding, repair, and full revalidation is the initial execution path. Fractional and whole-share assumptions are explicit. Mixed-integer methods, Rust, Polars migration, GPU sweeps and direct cone assembly remain demand-driven choices.

## Q16. One book per account, or several?

**One account-level target and one execution authority, with multiple attributed components.** Initially combine declared component targets in position space, then apply account-level construction, risk checks, rounding and order generation. Compute costs on the resulting net account trades.

Component outputs identify score/forecast/target semantics, units, decision time, horizon and expiry. Preserve zero forecasts, absent opinions, and explicit liquidation instructions as different states. Heuristic rank signals need no return mapping; a constructor using return units does. Keep the legacy momentum rank order unchanged in the control arm.

Opposing quantities net normally. Component urgency changes require an evaluated forecast/scheduling model; do not multiply final orders by the proposed half-life formula. Evaluate additions jointly with the incumbent portfolio, including costs, correlated uncertainty and search history. Diversification does not guarantee that every accepted component improves expected Sharpe.

## Q17. What is the operating target?

**Portable, restartable `run --once` commands with durable state, invoked by an external scheduler.** Document cron, systemd and Windows Task Scheduler/WSL operation. Use explicit stages for decision, submission and residual execution where needed; a daemon is not required.

Persist decision identity, fitted state, positions/cash observations, open orders, fills, sequence/checkpoint information and any hysteresis state. Write durable intent before submission, use stable order identities, protect each account from concurrent writers, and reconcile uncertain submissions before retries. Do not promise network-level exactly-once delivery; enforce idempotent accounting and recover ambiguous outcomes.

Freshness uses expected exchange sessions and dataset publication deadlines, with required asset coverage and known halt status. A single maximum timestamp is insufficient. Separate data freshness, broker-state freshness, clock health and scheduler liveness. Replay uses its recorded decision clock. A separate missed-run monitor must detect that the trading job never fired.

Missing alpha data blocks new alpha-driven exposure. Authorized cancellation and recovery remain possible when their own required inputs are trustworthy. Statistical deterioration uses a declared evidence window and sequential monitoring response; ordinary drawdowns or repeated unadjusted significance checks do not automatically terminate a strategy.

## Q18. How does the paper system operate during the overhaul?

**A stable paper deployment continues separately from overhaul work in an isolated worktree and environment.** Before restoring the stalled scheduler, reconcile positions, open orders, persisted state, expected sessions and missed-session handling. Do not fabricate fills or clean sessions for the interruption.

Move scheduled operation toward a pinned immutable release rather than an editable development checkout. Replay representative sessions before switching. Extraction refactors require parity within justified tolerances; intentional corrections require explained differences and their applicable operating approval. Missing historical inputs are a documented evidence gap, not a reason to invent a passing replay.

Keep one writer per checkout and keep the existing paper path usable through every stage. This launch brief settles the worktree arrangement; actual changes to the operating configuration and capital retain their existing authorization requirements.

## Q19. Which vocabulary and breadth diagnostics are adopted?

**Use ordinary language and retain standard financial terms.** Prefer strategy, portfolio, component, evaluation, result record, trial log, health check, and owner decision. Preserve old research IDs and citations. Rename public interfaces when useful, with aliases until supported consumers migrate. Cosmetic internal renaming is not a milestone.

Separate pipeline correctness, historical profitability, statistical evidence and authorization to operate. DSR is a selection-aware evidence statistic, not a reduction applied to observed returns. Avoid “positive after deflation.” [Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

Report covariance participation ratio, portfolio risk concentration, and independent evidence over time as distinct diagnostics. State the covariance universe, weighting, horizon and estimator. The current covariance participation ratio is a useful starting point, not a universal count of independent forecasts. Entropy-based risk concentration may be optional with its factor basis disclosed; PCA-basis ambiguity prevents its use as a universal replacement.

Call `IC * sqrt(N_eff)` a model-dependent performance diagnostic. It is not a Sharpe ceiling, viability theorem, annualization rule, or confidence correction. Exceeding it motivates investigation rather than establishing a leak. Do not infer which research lever binds solely from that ratio. Diagnose should rank supported explanations and recommend a discriminating comparison; “insufficient evidence” is a valid answer.

## Q20. Name, publication, license and documentation?

**Keep Prism, retain MIT, prefer the distribution name `prism-trading`, and preserve `import prism`.** The name `prism` is already occupied on PyPI; eligibility of the preferred distribution remains an external check before publication. Resolve it early enough for consistent templates, without delaying architecture work. [Existing PyPI distribution](https://pypi.org/project/prism/)

Use versioned releases, an explicit compatibility/deprecation policy, private-project version pins and lockfiles. Users review an upgrade comparison before adopting changed behavior. Keep secrets outside source, avoid import-time side effects, and separate evaluation environments from live credentials. Pin runnable environments and verify release/dependency provenance without promising that a lockfile makes dependencies safe.

Organize onboarding around four journeys: first evaluation, private signal, interpreting a comparison, and paper/operating workflow. Add reference pages where useful rather than enforcing exactly four documents. Preserve an indexed research archive. Contribution guidance distinguishes engine changes from reference-policy changes; merging code never implicitly authorizes a new live configuration.

## Q21. What may the project claim publicly?

**Publish reproducible historical evidence and conditional forecasts with the assumptions each requires.** Prohibit guarantees and unsupported reliable-profitability claims. Label synthetic demonstrations, exploratory historical analysis, prospective evaluation, paper fills and actual fills distinctly.

A result records implementation and configuration identities; data and revision manifests; return/cash/financing conventions; coverage; cost assumptions; evaluation folds and transforms; selection history and deflation method; performance, uncertainty, turnover, exposures, violations and account-size conditions. Report tails and drawdowns beside Sharpe. Preserve the actual periodic return series so calculations can be recomputed. Record “unavailable” when an uncertainty estimate or search history is not established.

Compare returns with returns. Define the excess-return series used for Sharpe and the hurdle's horizon and units. Benchmark cash and the relevant passive/control portfolio under consistent accounting; do not compare a Sharpe number directly with a Treasury yield. For Sharpe comparisons, use inference appropriate to serial dependence and non-normal returns, such as a declared studentized time-series bootstrap. [Ledoit and Wolf, 2008](https://www.ledoit.net/Robust_Sharpe_2008.pdf)

Use structured records with readable explanations rather than repeated mandatory prose. Operational consistency, cost savings, reduced engineering effort and reproducibility are valid product outcomes even when a candidate has insufficient investment evidence.

## Q22. What minimum account size does the reference profile declare?

**The reference strategy's minimum supported live capital is undetermined until measured.** Do not encode $250,000 from one synthetic rounding exercise. Research and paper profiles can explore explicit capital scenarios without claiming live suitability.

Evaluate multiple historical rebalance dates with actual prices, quantity increments, repair rules, cash, borrow/financing, trading restrictions and costs. Start with $10k, $25k, $50k, $100k, $250k and $1m as recorded scenarios. State which profile, broker and execution model the supported range applies to, how often repair fails, and how results change with capital. Feasibility alone does not establish an economically suitable account size; an all-cash solution cannot certify that the intended strategy is implementable.

Revalidate each actual rebalance. If the profile cannot execute, explain which condition binds. A smaller-account ETF or reduced-holdings strategy is a separate configuration with its own evidence. The owner's measurement micro-account remains distinct from deploying the reference strategy.

## Build order and acceptance

Fable should implement complete usable slices. Registration and research runs follow the current contract; synthetic engineering checks do not become market evidence. No phase intentionally leaves the reference workflow broken.

| Stage | Deliverable | Exit condition |
|---|---|---|
| 0. Preserve operation and define migration | Reconciled paper-state assessment, isolated development checkout, baseline replay inputs, focused contract migration | Existing paper path is usable; behavior and evidence gaps are recorded; engine/reference policy boundaries are explicit |
| 1. Installed private-project workflow | Workspace paths, extensible specification loader, template, synthetic fixtures, initial data/event identities and shared decision assembly with the legacy constructor | A separate directory uses an installed wheel, evaluates a private signal and replays a synthetic session without engine edits or credentials |
| 2. One evaluator and evidence history | Train-only transforms/calibration, conformance, caches, automatic trial/result records, comparison report, migrated reference history | Representative historical cases reproduce where inputs exist; history is retained; failures are recorded; private CI runs an offline smoke workflow |
| 3. Execution and accounting contract | Security/event handling, broker capability profile, cash/cost conventions, fractional/whole-share rules, rounding/repair, durable reconciliation and recovery scenarios | Adversarial synthetic sessions conserve accounting and never create unsupported fills or orders; shared cost semantics are checked numerically |
| 4. Construction experiment | CVXPY/Clarabel normal and recovery paths, calibration and uncertainty, factor risk model, preregistered three-arm comparison | All arms and capital scenarios are reported with identical charged cost conventions; constraints and rounding are independently checked; no automatic promotion |
| 5. Public operating workflow | Explicit account authorization, freshness and missed-run checks, restart recovery, upgrade comparison, controlled paper transition | Recorded scenarios and the declared paper observation window pass; intended behavior changes are reviewed; real-money controls work in offline tests |
| 6. Release and maintainability | Package publication preparation, onboarding journeys, compatibility guidance, verified private migrations, legacy retirement | Fresh installed-package workflow passes; public limitations are clear; target users can evaluate, operate in paper and assess an upgrade without maintaining a fork |

The foundational contracts in stage 1 may begin with synthetic implementations; stage 3 completes the real execution/event behavior before market comparisons depend on it. Extract shared decision assembly early, not after separate evaluator and live paths have accumulated new behavior. Documentation evolves with each slice.

At minimum, acceptance covers ticker reuse; late and null-valued revisions; missing terminal values; splits/distributions and short liabilities; held assets leaving eligibility; unsupported fractional orders; halted markets; gap-induced limit conflicts; rounding repair failure; expired forecasts; partial feed outages and holidays; duplicate fills; submission/restart ambiguity; invalid covariance/solver outputs; and direct attempts to submit live orders without authorization.

Cost consistency is tested on supplied trades in financial units. Causality checks cover fit/score boundaries and future-data extension, including cache reuse. Apply transformations such as price-scale invariance only where the declared signal semantics require them. Private CI should use deterministic offline examples; it must not contact a live account or rerun a large market search on every push.

## Required operating-contract migration

The implementation specification must include a small, reviewable migration rather than copying the old contract into the engine:

| Current statement or assumption | Required destination |
|---|---|
| N6 makes `IC * sqrt(N_eff)` a hard ceiling and viability gate | Replace with the diagnostic semantics in Q19; preserve historical records and add corrections without rewriting certified artifacts |
| Every signal has one sigma-normalized score interpretation | Declare score units/kind; preserve rank-consumed momentum; require calibrated return units only where a constructor needs them |
| Owner deployment thresholds and tighten-only profiles appear universal | Keep them in the owner's/reference policy; engine enforces the explicit policy chosen by each project |
| Sharpe is compared directly with a cash yield | Specify returns, excess returns, Sharpe units and hurdle comparisons consistently |
| Nightly operation consumes the editable checkout | Migrate to a pinned deployment with explicit upgrades and recorded state |

No migration erases trial history, reopens closed selection sets, advances future-dated research, or treats an engine release as a capital decision. If a historical result changes after a correctness fix, publish the reason and new result identity; do not widen tolerances or overwrite the old artifact.

## Instructions for Fable's next deliverable

Write the implementation specification from these answers, retaining Q1–Q22 traceability. Supply the smallest sufficient interfaces and schemas, staged code changes with their dependencies, acceptance cases, and the focused contract migration. Prefer adapting existing modules over a wholesale directory rename. Distinguish extraction parity from intentional behavioral corrections and from new strategy research.

Use the command vocabulary `init`, `data`, `conform`, `evaluate`, `compare`, `run --once`, `reconcile`, `check`, and `upgrade-diff` as the intended user journeys; finalize exact CLI details during specification. One strategy configuration plus its private dependencies must drive both evaluation and operation. Config identity includes referenced policy and cost content and strategy implementation identity, not just the top-level TOML bytes.

Carry empirical questions into registrations and acceptance work: calibration adequacy, uncertainty, actual execution costs, capital suitability, prospective investment evidence, and measured performance bottlenecks. Do not recast those unknowns as product decisions still awaiting an owner answer.

Do not carry forward the prior claims of monotone expected Sharpe, exact cost forecasts, universal solver feasibility, automatic zero holdings at a confidence threshold, a synthetic $250k capital floor, mandatory native code at 3,000 names, entropy as a universal statistical-breadth replacement, ad hoc urgency-weighted orders, or a private rebalance seed eliminating an individual portfolio's timing luck. Cohort staggering is a separate evaluated strategy choice. The reviewed Gemini snippets are illustrative material to replace with tested implementations against these contracts.

Keep the resulting specification and implementation record cohesive. After they incorporate this brief, retain it as a dated decision record rather than adding ongoing ratifications, parallel contracts, or duplicated enforcement prose.
