# Characteristics composite — design and registration (`characteristics_v1`)

**Status: registered 2026-09-16.** The registration is the commit that adds this
file and `registrations/2026-09-characteristics.toml` (SPEC §4). It answers
SPEC §5.3: a composite of slow, bar-derived characteristics at monthly cadence
as one registered selection set, deflated as a whole. Nothing here has been
run at the time of registration; the read record is a separate file.

## 1. Why this set, and why now

The live book is 12-1 momentum alone. Its recorded evidence
(`results/demotion_b1/`, net annualized Sharpe 0.465, DSR 0.19, 21-bar rank
IC 0.030 with a one-sided lower-95 below zero) says the book is signal-bound:
breadth and IC are the levers, cost is not (SPEC §1). Of the two, IC is the
one that can move on the data already in the tree. This set asks whether two
further characteristics, each computed from the same bars under the same
screen and cadence, add to that book.

Long-horizon reversal (De Bondt and Thaler 1985; months 13–60) is named in
SPEC §5.3 and is **not** in this set. The frozen dataset opens 2020-01-02
(`[sample].history_start`), so a 36-to-60-month formation window has no
observations before 2023–2025 and the cell would be evaluated on a fraction
of the sample the other arms use. It waits for a history-complete panel
(`docs/data_purchase_evaluation.md` records what that costs and that the
owner declined the purchase on 2026-07-17). Registering it on a truncated
sample now, and re-registering it later on a full one, would be two chances
at one idea.

## 2. Feature space — three characteristics, fixed parameters

All three are `rank_score` components consumed by the control's own
`decile_long_short` rule (decile 0.10, per-name cap 0.35, gross 1.0), on the
B1 eligibility screen (price ≥ 5, median dollar volume ≥ $1M over 20 bars,
252 finite trailing returns, point-in-time S&P 500 membership).

| id | node | score at bar *t* | parameters |
|---|---|---|---|
| `momentum` | `prism.signal.momentum_node:MomentumSignalNode` | `close[t−21] / close[t−252] − 1` | B1's own (252, 21) |
| `residual_momentum` | `prism.signal.residual_momentum_node:ResidualMomentumSignalNode` | `Σ ε[t−252 : t−21] / sd(ε[t−252 : t−21])`, ε the return net of a 15-eigenportfolio fit on the same 252 bars, intercept retained | (252, 21), 15 factors (`ResidualStatArbConfig` defaults) |
| `idiosyncratic_vol` | `prism.signal.idiosyncratic_vol_node:IdiosyncraticVolSignalNode` | `−sd(ε[t−63 : t])`, ε as above | 252 estimation bars, 63 dispersion bars |

Residualization is one trailing-window fit per decision bar
(`prism.residual.factors.trailing_factor_residuals`): eigenportfolios struck on
the window's own correlation matrix, factor returns formed over the window,
one batched OLS. The intercept stays in the residual; subtracting it would
make every window residual sum to zero and turn residual momentum into
short-term reversal (`tests/test_characteristics_nodes.py` pins this).
The volatility node emits the *negative* standard deviation so that the
top decile is the low-volatility leg and every component carries
`expected_sign = "positive"`.

The parameters are the published characteristics' conventional ones (Blitz,
Huij and Martens 2011; Ang, Hodrick, Xing and Zhang 2006) restated on the
momentum book's own windows. None was chosen against a result; none may be
changed inside this set.

## 3. Search procedure — six declared arms, one weight table each

Arms differ only in `[arms.components]`, the weight each component carries in
the position-space sum (`prism.composition`, `renormalize = false`; each
component's decile book has gross 1.0, so the composed gross is at most 1.0).

| arm | momentum | residual_momentum | idiosyncratic_vol |
|---|---|---|---|
| `momentum_control` | 1 | 0 | 0 |
| `residual_momentum_only` | 0 | 1 | 0 |
| `idiosyncratic_vol_only` | 0 | 0 | 1 |
| `momentum_residual_momentum` | ½ | ½ | 0 |
| `momentum_idiosyncratic_vol` | ½ | 0 | ½ |
| `composite_three` | ⅓ | ⅓ | ⅓ |

Equal weights, not fitted weights. A weight fitted on this sample would be a
free parameter the deflation could not see. The control runs the same
three-component book with the other two at zero, so every arm executes one
code path and the freeze covers every panel once.

The registration machinery gained one field for this set: `[arms.components]`
(`prism.registration._parse_component_weights`), which must list every
declared component (zero excludes, omission is refused) with at least one
positive weight. `arm_specification` writes the arm's weights over the
control's. The freeze step now materializes one panel per component and
digests each (`component_panel_sha256`); a single-component registration
freezes exactly what it did before.

## 4. Universe, cost stack, sample, engine — the B1 evidence's own

Frozen dataset `data/frozen/demotion_b1` (574 resolved of 633 ever-members,
coverage 0.907, the gap enumerated in `data/universe/sp500_coverage_2026-06-16.json`);
fold table `folds/demotion_b1.toml` (21 folds, 312 formation bars, 63 test
bars, last 48); cost stack `examples/reference/costs/retail_v1.toml`;
execution profile `examples/reference/profiles/execution_alpaca.toml`;
canonical accounting, `research_vintage` visibility, `fold_portfolio = "carry"`;
experiment profile `examples/reference/profiles/experiment_v1.toml` (the
construction experiment's, so the control here is that experiment's
`decile_control` at $100,000 and can be checked against it). One capital
scenario, $100,000: the construction experiment showed capital moving the
control's Sharpe without ordering, and this set spends its multiplicity on
arms. Price-return convention, no distributions, as recorded for B1.

## 5. Kill rule and closure

Same-sample. Each arm is read against `momentum_control` by the declared paired
studentized block bootstrap (`[sharpe_bootstrap]`: daily, circular blocks of
42 bars, 2,000 resamples, seed 20260916). **An arm whose two-sided 95% interval
for (arm − control) net periodic Sharpe lies entirely below zero is dead.**
The set closes when the six arms have run, whatever the outcome. Five
intervals against the control (plus the ten other pairs the comparison record
carries for completeness) are the multiplicity; the read reports all of them
and counts them.

Deflation: every arm is a row in `trials/characteristics_v1.jsonl` and the set
deflates against its own six rows (SPEC §4). The standalone arms are counted
even though their purpose is diagnostic; the alternative, running them
"outside" the set, is the pooled-ledger error §4 forbids.

## 6. Promotion rule

None. A same-sample comparison cannot promote. A composite arm that survives
the kill rule *and* beats the control in point estimate is a candidate for a
new registration whose evidence is prospective: the backtest extension after
2026-06-16 plus the paper stream, at `net_edge` under bucket spreads, DSR
> 0.5 against this set, periodic net Sharpe above the T-bill hurdle. The live
book's configuration is an owner act (SPEC §6) and nothing here changes it.

## 7. Relation to the other registrations

- `docs/learned_xsection_design.md` (`learned_xsection_v1`, frozen, unrun)
  registers a *learned* ridge combination of seven features (`mom_12_1,
  mom_6_1, rev_1, vol_63, maxret_21, high_52w, dv_63`). This set is fixed
  equal weights over three characteristics, two of which (`residual_momentum`,
  `idiosyncratic_vol`) are not in that feature list, and `vol_63` there is
  total, not idiosyncratic, volatility. Different feature space, different
  procedure, different set; neither cites the other's rows.
- `docs/aim_portfolio_preregistration.md` governs Gârleanu–Pedersen blending
  of *sleeves* and is gated on a sleeve promoting. This set combines
  characteristics inside one sleeve by the engine's existing position-space
  rule, which that document names as the rule it would eventually replace. No
  GP parameter is fitted here and nothing in that document is spent.
- `docs/momentum_design.md` (`momentum_v1`) holds the momentum book's own
  fragility reads; `momentum_control` reproduces that book's configuration
  under the experiment profile and is not a row in that set.

## 8. What the set cannot establish

Modeled fills; a 0.907-coverage universe biased toward survivors; no
dividends; 2020-01 to 2026-06, one market, one vendor; no momentum crash in
sample. A composite that survives here has shown it is not worse than the
control on this sample under these conventions, and nothing more.
