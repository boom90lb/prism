# Documents

An index of everything under `docs/`, grouped by what a reader wants from it.
Terms the records use without explanation are in the [glossary](glossary.md).

## Start here

| Document | What it is |
|---|---|
| [Quickstart](quickstart.md) | From a clone to the momentum book trading a free Alpaca paper account, scheduled nightly. |
| [Onboarding](onboarding.md) | Four walkthroughs with an installed wheel: a synthetic evaluation, a private signal, reading a comparison, and an offline operating review. |
| [Limitations](limitations.md) | What the published results do and do not establish. Read it before the numbers. |
| [Glossary](glossary.md) | In-house terms mapped to standard ones. |
| [Erratum, 2026-10-14](erratum_2026-10-14.md) | Corrections to how recorded results are described. |
| [Architecture](../ARCHITECTURE.md) | What calls what in the code. |
| [Markets](../MARKETS.md) | The mid-2026 analysis of which markets a retail trader can work in at no data cost. |
| [Operating contract](../SPEC.md) | The project's rules: invariants, evaluation standards, how experiments are registered and approved, and the deployment gate. Written for maintainers. |

## Using the engine

| Document | What it is |
|---|---|
| [Depending on a release](depending_on_prism.md) | Pinning `prism-trading`, what a lockfile does and does not buy, and the release check. |
| [Versioning](versioning.md) | What a version number promises, the upgrade comparison, deprecation, and what to re-verify after an upgrade. |
| [Contributing](contributing.md) | How engine changes and strategy-policy changes are reviewed, and why merging code never authorizes trading an account. |
| [Security](security.md) | How credentials are held, transported and logged. |
| [Free-tier profile](free_tier_profile.md) | What a clean checkout reproduces with free data and keys, and what it cannot. |
| [Operations](operations.md) | Operational facts for running the paper loop on real data. |
| [Risk-profile schema](risk_profile_schema.md) | The operator's risk settings, which may only tighten the project's fixed limits. |
| [Research from an installed release](owner_migration.md) | Reproducing a recorded result from a private project that pins the wheel ([receipt](owner_migration.json)). |
| [Research tree](research_tree.md) | Every module under `research/`, with its status, records, tests and dependencies. |
| [Research archive](research_archive.md) | Where the implementations retired by the engine overhaul can be found and reproduced. |

## Results reports

Each report is the record of one pre-registered experiment, written whatever the outcome.

| Report | Experiment and result |
|---|---|
| [Momentum robustness, 2026-09-03](momentum_m_series_read_2026-09-03.md) | Five changes to the 12−1 momentum book (M1–M5). No sign flip; median net Sharpe 0.477 against 0.465. |
| [ETF trend following, 2026-09-03](trend_t_series_read_2026-09-03.md) | Trend tests T0–T4. Net Sharpe 0.04 with no return above cash, and the diversification test failed. |
| [Portfolio construction, 2026-09-08](construction_experiment_read_2026-09-08.md) | Three construction methods at six account sizes. All eighteen comparisons were inconclusive. |
| [Stock characteristics, 2026-09-23](characteristics_read_2026-09-23.md) | Idiosyncratic volatility and residual momentum against the momentum control. Residual momentum could not be told apart from it. |
| [Universe breadth, 2026-09-23](breadth_read_2026-09-23.md) | Adding the S&P 400 raised effective breadth 37% and did not raise net Sharpe. |
| [Learned ranking, 2026-09-23](learned_xsection_read_2026-09-23.md) | A learned cross-sectional ranking (X0–X4). Explained by momentum; not added. |
| [Volatility-scaled momentum, 2026-09-24](vol_momentum_read_2026-09-24.md) | Momentum scaled to 12% volatility. Approved on the published-factor route by 0.018 Sharpe over its hurdle; not traded. |
| [Public factor lab, 2026-09-24](public_factor_lab_read_2026-09-24.md) | Sizing rules tested on Kenneth French and JKP data. The crash-conditional cut was contradicted; constant-volatility scaling was supported. |
| [Fundamentals, 2026-09-24](fundamentals_read_2026-09-24.md) | Six accounting-based factors and one combination. None cleared its hurdle. |
| [Earnings announcements, 2026-09-24](earnings_read_2026-09-24.md) | The announcement return, monthly and daily. Nothing approved; two daily variants carried to a forward evaluation. |
| [Rank buffer, 2026-09-25](buffer_read_2026-09-25.md) | Entering at 10% and holding to 20%. Turnover fell 12–36% with no distinguishable change in net Sharpe. |

## Published negative result

| Document | What it is |
|---|---|
| [Certification 001](certifications/001-residual-reversion-daily-negative.md) | Daily residual reversion on the S&P 500 is uneconomic at retail cost, over a closed set of 17 trials. Its cited artifacts are never rewritten. |

## Experiment designs

Each design was committed before its experiment ran and fixes what was tried and how it is judged.

| Design | What it registers |
|---|---|
| [Momentum](momentum_design.md) | `momentum_v1`: robustness tests M1–M5 and the later re-run M6. |
| [Trend](trend_design.md) | ETF trend following, T0–T5. |
| [July 2026 budget](demotion_design.md) | The seven trials D1–D5, B1 and B2 that followed the stop of the residual-reversion program. |
| [R2](r2_design.md) | A closed-form no-trade band, a participation limit and per-bucket spread calibration, all default-off. |
| [Replication](replication_preregistration.md) | The frozen momentum book on other samples; waits on a qualifying price panel. |
| [Sizing](sizing_preregistration.md) | The crash-conditional exposure cut, later contradicted by public data. |
| [Aim portfolio](aim_portfolio_preregistration.md) | Multi-signal Gârleanu–Pedersen construction; drafted and gated. |
| [Covariance cleaning](covariance_cleaning_preregistration.md) | Ledoit–Wolf nonlinear shrinkage for the risk model; drafted and gated. |
| [EDGE spreads](edge_preregistration.md) | The bar-based spread estimator, fixed as a cross-check, never a calibration. |
| [Learned ranking](learned_xsection_design.md) | `learned_xsection_v1`. |
| [Characteristics](characteristics_design.md) | `characteristics_v1`. |
| [Breadth](breadth_design.md) | `breadth_v1`. |
| [Public factor lab](public_factor_lab_design.md) | `sizing_lab_v1`. |
| [Volatility-scaled momentum](vol_momentum_design.md) | `vol_momentum_v1`, published-factor route. |
| [Fundamentals](fundamentals_design.md) | `fundamentals_v1`, published-factor route. |
| [Earnings](earnings_design.md) | `earnings_v1` and `earnings_daily_v1`. |
| [Rank buffer](buffer_design.md) | `buffer_monthly_v1` and `buffer_daily_v1`. |
| [Forward evaluation](forward_design.md) | `forward_v1`: the two residual-momentum candidates on data after 2026-09-24. |
| [Live monitor](live_monitor_design.md) | The statistics computed nightly over the paper account's ledgers. |
| [Delisted-ticker prices](dead_ticker_price_design.md) | Accumulating prices of delisted names from now on. |
| [Tax wedge specification](tax_wedge_spec.md) | How the tax measurement is computed. |

## Measurements and diagnostics

None of these adds a trial or can choose a configuration.

| Document | What it measures |
|---|---|
| [Account-size floor](account_size_floor.md) | How whole-share rounding changes the momentum book at each account size. |
| [Cost frontier](cost_frontier.md) | Net performance against the assumed one-way spread. |
| [Tax wedge](tax_wedge.md) | Measured tax asymmetries against a symmetric baseline. |
| [Market beta](beta_telemetry.md) | The momentum book's realized market beta. |
| [Joint crash](joint_crash_diagnostic.md) | The momentum book alone and with the trend book under stress. |
| [B2 netting](b2_anti_netting_diagnostic.md) | Why the combined residual-and-momentum book traded more than its halves. |
| [Bar-vendor divergence](bar_vendor_divergence.md) | Differences between price vendors' bars. |
| [Spin-off adjustment](spin_adjustment_sweep.md) | Whether spin-off price adjustments are consistent across the recorded panel. |
| [Data integrity](data_integrity_diagnostic.md) | Vendor symbols that resolve to the wrong instrument, enumerated and cleared. |
| [Fold-flatten counterfactual](carry_flatten_diagnostic.md) | What the backtest's forced flattening at each fold boundary costs, given that the live loop never flattens. |
| [Replay concordance](replay_concordance_diagnostic.md) | The live loop's replay against the backtest. |
| [Regime step](regime_step.md) | How regime telemetry is wired into the live cycle; its exposure-cut hook is not armed. |
| [Stat-arb shrinkage](stat_arb_shrinkage_experiment.md) | A pairs-trading gate against portfolio-level shrinkage; negative after deflation. |

## Audits, reviews and incidents

| Document | What it is |
|---|---|
| [Construction band audit, 2026-09-24](construction_band_audit_2026-09-24.md) | Records that described a no-trade band the engine did not apply. |
| [Hurdle restatement, 2026-09-30](hurdle_restatement_2026-09-30.md) | Every recorded hurdle recomputed under the 2026-09-25 cash convention. No verdict changes. |
| [Incident, 2026-09-09](incident_2026-09-09_unauthorized_paper_orders.md) | 116 unauthorized orders on the paper account, and the fixes. |
| [Live-path audit, 2026-07-29](live_robustness_audit_2026-07-29.md) | Robustness findings in the trading code. |
| [Program review, 2026-07](program_review_2026-07.md) | A verified review of the research program. |
| [Data-purchase evaluation](data_purchase_evaluation.md) | Whether to buy survivorship-complete pre-2020 price history. |
| [Audit of the predecessor](audit.md) | The 2026-06 audit of the code base Prism replaced. Historical. |
| [Statistical arbitrage](stat_arb.md) | The pairs-trading research harness. Historical. |

## Engine overhaul, 2026-09

| Document | What it is |
|---|---|
| [Design brief](prism_engine_decisions.md) | The questions Q1–Q20 and their answers that shaped the engine. |
| [Operating changes](engine_overhaul_operating_changes.md) | Every intended difference in live behaviour, and the decision-parity check ([result](engine_overhaul_parity_gate.json)). |
| [Completion record](engine_overhaul_completion_2026-09-15.md) | The release engineering completed before the overhaul was merged. |
| [Onboarding receipt](onboarding_verification.json) | The recorded run of the onboarding walkthroughs. |
