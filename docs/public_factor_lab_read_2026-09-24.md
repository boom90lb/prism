# Public factor lab — read record, 2026-09-24

**Kind:** read record for `registrations/2026-09-factor-lab.toml`
(`docs/public_factor_lab_design.md`). **Registration commit:** `c6817f2` (the
first commit to carry the registration; the run refused to start until it was
committed unchanged). **Code commit of the run:** `c6817f2`. **Selection set:**
`sizing_lab_v1`, budget 8 declared, 8 spent, trial log
`trials/sizing_lab_v1.jsonl` (8 started, 8 finished, chain verified).
**Outputs:** `results/2026-09-factor-lab/` (`sources.json`, `priors.json`,
`correlations/`, `fidelity.json`, `sizing/summary.json`,
`sizing/returns.parquet`).

**Data.** French files built on the 202607 CRSP database (through
2026-07-31); JKP files through 2025-12-31. The sha256 prefixes are:

| source | sha256 prefix |
|---|---|
| `ff3_daily` | `1916d331c2c5` |
| `ff5_daily` | `478350b8d608` |
| `ff_mom_daily` | `b039d986db27` |
| `ff_strev_daily` | `5b491fb0d2b4` |
| `jkp_themes_daily` | `6d479c0ed02e` |
| `jkp_resmom_daily` | `d8eb0e51260b` |
| `jkp_resmom_large_monthly` | `b8e293d68d1c` |

The full hashes and fetch times are in `sources.json`. As registered, one
session (1985-09-27) was dropped from the residual-momentum arms.

## Verdict

- **The ratified crash de-gross is contradicted on both bases.** It is
  `docs/sizing_preregistration.md` §3 with every pin as ratified. It was read
  before 2020, which is out of sample for its threshold. On momentum it lowered
  Sharpe from 0.399 to 0.298 (difference −0.101, 95% interval −0.203 to
  −0.005). On residual momentum it lowered Sharpe from 0.973 to 0.800
  (difference −0.172, −0.276 to −0.046). It held half gross on about 45% of
  sessions, including in its own 2020–26 calibration window. It cut the 1932
  and 2009 losses by roughly seven points each.
- **Constant-volatility scaling (Barroso–Santa-Clara) is supported on both
  bases.**
  - On momentum it is the strongest result here. Sharpe roughly doubled over
    1931–2026, from 0.368 to 0.776 (difference +0.409, interval +0.267 to
    +0.561). After its 2015 publication it still added +0.228 (interval +0.076
    to +0.428). The 1932 loss fell from 51% to 27% and the 2009 loss from 46%
    to 16%.
  - On residual momentum the support is knife-edge. The post-publication
    difference is +0.0002, against a threshold of ≥ 0. Its content is
    leverage, not timing: the mean scale is 1.84, and maximum drawdown rose
    from 42% to 56%.
- **Variance-managed scaling (Moreira–Muir) is inconclusive overall.** It is
  supported on momentum (full-sample difference +0.308) and inconclusive on
  residual momentum. On momentum it deepens the worst drawdown: 78% against
  76%, with a worst 21-session loss of −50.5% and a worst calendar year of
  −51.5% (1939).
- **Priors (uncounted).** Every classic premium shrank after publication. The
  published residual-momentum factor earned 1.40 before 2011 and 0.18 in the
  14.6 years since. Momentum fell from 0.63 to 0.35 after 1993.
- **Fidelity (uncounted).** The in-house momentum book tracks the public
  momentum factor at ρ = 0.88. The in-house residual-momentum book tracks the
  published residual-momentum factor at ρ = 0.495, below the §4 floor of 0.5.

## Part A — priors

Annualized Sharpe of the daily long−short return. t = Sharpe × √years, which
assumes independent days. Prior = the §4 rule's prior Sharpe. The 13 JKP
themes are descriptive only and appear in the correlation files.

| series | from | years | Sharpe | t | pre-pub | post-pub | post years | §4 bar | prior |
|---|---|---|---|---|---|---|---|---|---|
| UMD (momentum) | 1926-11 | 104 | 0.51 | 5.2 | 0.63 | 0.35 | 33.4 | pass | **0.35** |
| RESMOM (residual momentum) | 1953-05 | 73 | 1.10 | 9.4 | 1.40 | 0.18 | 14.6 | pass | **0.18** |
| HML (value) | 1963-07 | 63 | 0.40 | 3.2 | 0.90 | 0.21 | 34.1 | pass | 0.21 |
| RMW (profitability) | 1963-07 | 63 | 0.48 | 3.8 | 0.58 | 0.24 | 13.3 | pass | 0.24 |
| CMA (investment) | 1963-07 | 63 | 0.49 | 3.9 | 0.75 | 0.03 | 21.6 | pass | 0.03 |
| SMB (size) | 1963-07 | 63 | 0.18 | 1.4 | 0.88 | 0.01 | 45.4 | fail | 0.01 |
| STREV (short-term reversal) | 1926-01 | 105 | 3.78 | 38.7 | 5.10 | 1.22 | 36.1 | pass | 1.22 |
| MKT_RF | 1926-07 | 104 | 0.46 | 4.7 | — | — | — | pass | — |

The figures are gross paper-portfolio returns. STREV's 1.22 is the textbook
case of a gross premium that turnover consumes: certification 001 found its
nearest in-house relative uneconomic at retail cost. The sizing arms below
carry the registered costs.

**Correlation with momentum** (monthly, 2000-01 → 2025-12,
`correlations/corr_2000_2025_monthly.csv`):

| theme | ρ with UMD |
|---|---|
| JKP value | −0.18 |
| HML | −0.16 |
| JKP accruals | −0.18 |
| JKP investment | +0.02 |
| JKP debt issuance | +0.16 |
| JKP seasonality | +0.27 |
| JKP low risk | +0.34 |
| JKP quality | +0.36 |
| JKP profit growth | +0.47 |
| RESMOM | +0.74 |

Profit growth sitting at +0.47 fits the argument that price momentum is
largely earnings momentum.

## Part B — fidelity

Pearson correlation of in-house daily net returns with the reference series,
over their overlap (`fidelity.json`):

| in-house book | reference | overlap | daily ρ | weekly ρ | β | implied tracking Sharpe (ρ × prior) |
|---|---|---|---|---|---|---|
| live book B1 (historical engine) | UMD | 2021-03 → 2026-06 | 0.644 | 0.827 | 0.44 | 0.22 |
| momentum control (canonical engine) | UMD | 2022-03 → 2026-06 | **0.880** | 0.881 | 0.63 | **0.30** |
| momentum control | JKP momentum theme | 2022-03 → 2025-12 | 0.869 | 0.878 | 0.74 | — |
| residual momentum | RESMOM daily | 2022-03 → 2025-12 | **0.495** | 0.474 | 0.24 | 0.09 |
| residual momentum | JKP large-cap RESMOM, monthly | 46 months | 0.129 | — | — | — |

The canonical engine's momentum book is a faithful implementation of the
public factor.

The in-house residual momentum is not the published characteristic. It
residualizes on 15 trailing principal components, where the paper uses the
Fama–French three factors, and its correlation with the large-cap published
series is 0.13. Its correlation with the all-stock series is 0.495, below the
0.5 floor. At the 0.18 prior, the implied tracking Sharpe is 0.09 even before
costs. The replication route (SPEC §4) is therefore not open to it as
implemented.

The live book's lower daily ρ (0.644 against 0.827 weekly) is the timing
offset the design anticipated in the historical constant-weight engine.

## Part C — sizing rules

Net of the registered costs, which are gross paper-portfolio returns less the
in-house cost rates. Pre-1990 costs were higher than charged, so the rows are
read against each other. "Post-pub" is from the rule's publication; the
control's figures in that column are computed from the recorded series
(`sizing/returns.parquet`), because the registration did not name a
post-publication window for the control. Crash columns are compounded returns
over 1932-06→08, 2009-03→05 and 2020-11.

**UMD (momentum), 1931-02 → 2026-07.**

| rule | Sharpe full | pre-2020 | since 2020 | post-pub | mean scale | share de-grossed | max DD | worst 21 | 1932 | 2009 | 2020-11 | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.368 | 0.399 | 0.178 | 0.161 / 0.217 | 1.00 | — | 75.5% | −32.6% | −50.8% | −45.8% | −12.1% | 0.51 |
| crash_degross_v1 | 0.273 | 0.298 | 0.142 | — | 0.77 | 46% | 68.9% | −32.3% | −43.7% | −38.4% | −6.7% | 0.18 |
| constant_vol_bsc | **0.776** | 0.803 | 0.421 | **0.388** | 1.40 | — | 54.6% | −27.0% | −26.7% | −16.3% | −3.7% | 1.00 |
| variance_managed_mm | 0.676 | 0.682 | 0.572 | 0.529 | 1.57 | — | 78.2% | −50.5% | −22.1% | −3.6% | −3.4% | 1.00 |

**RESMOM (residual momentum), 1958-05 → 2025-12.**

| rule | Sharpe full | pre-2020 | since 2020 | post-pub | mean scale | share de-grossed | max DD | worst 21 | 2009 | 2020-11 | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.880 | 0.973 | 0.341 | 0.097 / 0.170 | 1.00 | — | 42.1% | −14.7% | −17.2% | −3.6% | 1.00 |
| crash_degross_v1 | 0.724 | 0.800 | 0.320 | — | 0.78 | 44% | 36.0% | −9.1% | −13.2% | −1.0% | 1.00 |
| constant_vol_bsc | 0.986 | 1.075 | 0.346 | 0.097 | 1.84 | — | 56.2% | −25.3% | −18.7% | −5.3% | 1.00 |
| variance_managed_mm | 0.859 | 0.928 | 0.241 | 0.101 | 1.23 | — | 43.5% | −9.1% | −5.5% | −2.8% | 1.00 |

"Post-pub" for `none` is shown from 2015-04 / 2017-08, matching the two
published rules.

**Paired Sharpe differences against `none`**: annualized, 95% studentized
circular block bootstrap, 63-session blocks, 1,000 resamples.

| arm | window | difference | interval | p |
|---|---|---|---|---|
| UMD crash_degross_v1 | pre-2020 | −0.101 | [−0.203, −0.005] | 0.048 |
| UMD constant_vol_bsc | full | +0.409 | [+0.267, +0.561] | <0.001 |
| UMD constant_vol_bsc | post-pub | +0.228 | [+0.076, +0.428] | 0.018 |
| UMD variance_managed_mm | full | +0.308 | [+0.132, +0.500] | 0.001 |
| UMD variance_managed_mm | post-pub | +0.312 | [−0.038, +0.742] | 0.126 |
| RESMOM crash_degross_v1 | pre-2020 | −0.172 | [−0.276, −0.046] | 0.004 |
| RESMOM constant_vol_bsc | full | +0.106 | [+0.018, +0.196] | 0.018 |
| RESMOM constant_vol_bsc | post-pub | +0.0002 | [−0.122, +0.125] | 0.998 |
| RESMOM variance_managed_mm | full | −0.021 | [−0.229, +0.201] | 0.843 |
| RESMOM variance_managed_mm | post-pub | −0.068 | [−0.371, +0.241] | 0.646 |

**Verdicts, applied as registered:**

| rule | UMD | RESMOM | overall |
|---|---|---|---|
| `crash_degross_v1` | contradicted | contradicted | **contradicted** |
| `constant_vol_bsc` | supported | supported (post-pub +0.0002) | **supported** |
| `variance_managed_mm` | supported | inconclusive | **inconclusive** |

## What the numbers say, and what they do not

- **Why the de-gross fails.** The state (trailing 21-session market return
  below −1.8%) persists for weeks, and the exit waits for five clean sessions
  and then a refresh. So the rule halves the book for long stretches of
  ordinary markets and gives up more return than it saves in crashes. Its
  "worst decile" origin describes the threshold, not the duty cycle, which is
  about 45%. The rule is read here as registered, including the exit reading
  pinned in the design (§5). A looser exit would shorten the duty cycle, but
  that is a new rule and a new registration.
- **Volatility scaling helps where volatility forecasts losses.** Momentum's
  volatility is time-varying and its crashes arrive in high-volatility
  rebounds. Scaling to constant volatility cut its worst episodes by half and
  kept a significant gain after publication. That is the Barroso–Santa-Clara
  and Daniel–Moskowitz result, reproduced on data through 2026-07 net of
  in-house costs. Residual momentum is already low-volatility, so the same
  rule mostly levers it up.
- **Post-publication premia are small.** Classic factors keep 0.2–0.35 Sharpe
  after publication. Residual momentum keeps 0.18. The in-house 2022–26
  figures (momentum 0.68, residual momentum 0.83) sit well above the long-run
  post-publication record. The 2022–26 in-house sample should be read as a
  favourable draw, not as the expectation.
- **Not established here.**
  - Nothing in this set promotes or arms anything.
  - The public series are all-CRSP paper portfolios, not S&P 500 books.
  - t-statistics in Part A assume independent days.
  - JKP themes are descriptive.

## What this changes

- **De-gross.** This is recorded evidence against arming
  `docs/sizing_preregistration.md` §3 as ratified. SPEC §6 requires that rule
  to be armed before any real-money order, so the gate as written now requires
  a sizing rule this record contradicts. Changing the gate or the rule is the
  owner's decision. The evidence points to constant-volatility scaling as the
  replacement, and adopting it is a separate registration.
- **Residual momentum.** `forward_v1` continues unchanged; its forward clock
  is unaffected. The replication route is closed to the current
  implementation. A version built on the published definition (Fama–French 3
  residuals) would be a new registration and would face a 0.18 prior.
- **Momentum.** This is the candidate with a public record behind it:
  fidelity 0.88, prior 0.35, and 0.388 post-publication with volatility
  scaling net of in-house costs. A replication-route registration for the
  in-house momentum book with constant-volatility scaling is the next
  registration this record supports.
- **Open definitional point for that registration.** SPEC §4 criterion 3 says
  "clears the hurdle". In-house books assume zero cash income
  (`examples/reference/costs/retail_v1.toml`, `cash_income_bps_annual = 0.0`,
  a stated placeholder), so the T-bill hurdle in Sharpe units depends on the
  volatility the book runs at. The registration has to pin it.
