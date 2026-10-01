# After-cost hurdle restatement — 2026-09-30

**Kind:** restatement, not a decision. **No verdict changes. Closed sets stay closed**
(SPEC §1, §4: a closed set keeps its recorded verdict). **The cash yield used below, 3.3%,
is a provisional listed rate** (`examples/reference/costs/retail_v2.toml`,
`cash_income_bps_annual = 330`, "replace with the account statement's rate") and every
restated number inherits that status. Nothing under `results/` or `trials/` was touched.

**Why this exists.** The after-cost hurdle was computed three ways: the research driver that
wrote the momentum, trend and certification ledgers converts an annual bill yield at each
book's realized periodic volatility (`prism.validation.metrics.after_cost_hurdle_periodic`);
the volatility-scaled momentum replication divides the yield by the 12% *target*
(`research/scripts/vol_momentum_replication.economics`); the forward reader divides the
amended return hurdle by realized annualized volatility (`research/scripts/forward_read`).
From today all three route through `prism.validation.hurdle.tbill_hurdle`, which records
the volatility basis and the rate sources, and the claim packet carries an optional
`hurdle` block (`prism.validation.trials`). Recorded numbers reproduce exactly
(`tests/test_hurdle.py`, `tests/test_vol_momentum_replication.py`).

**Convention.** Recorded (every verdict below): idle cash earns nothing, so the Sharpe hurdle is
rf / vol. Amended (SPEC §3, owner decision 2026-09-25): (rf − y·max(0, 1 − L/E)) / vol, with
y the account's cash yield and L/E the long side over equity. Short proceeds earn nothing
either way. Where L/E is taken from a record it is (avg_gross + avg_net) / 2 of that record's
`summary.json`, or 0.5 × the arm's scale for a 1×-gross equal long/short book; where neither
is recorded the table says so. All Sharpe figures are annualized; periodic figures are the
recorded `periodic_sharpe_hurdle` (daily) where the verdict was read in those units.

## Discovery-route records (hurdle at realized annualized volatility; rf = 3.71% or 3.78% T-bill, `after_cost_hurdle.basis = tbill_nominal`)

Source per row: `results/<record>/summary.json` (`after_cost_hurdle`, `annualized_vol`, `avg_gross`,
`avg_net`, `sharpe`). Verdict columns: `docs/momentum_m_series_read_2026-09-03.md` ("clears 3.71% T-bill"),
`docs/trend_t_series_read_2026-09-03.md` §Results, `docs/certifications/001-*.md` §4.

| record | rf | vol | L/E (recorded) | recorded hurdle (periodic) | restated, y = 3.3% | realized net Sharpe | margin recorded → restated | recorded verdict |
|---|---|---|---|---|---|---|---|---|
| `demotion_b1` (live book) | 3.71% | 11.51% | 0.484 | 0.322 (0.02031) | 0.174 (0.01099) | 0.465 | +0.143 → +0.291 | clears; `net_edge` |
| `momentum_m1` | 3.71% | 11.81% | 0.484 | 0.314 (0.01979) | 0.170 | 0.477 | +0.162 → +0.307 | clears |
| `momentum_m2` | 3.71% | 10.78% | 0.484 | 0.344 (0.02168) | 0.186 | 0.324 | −0.021 → +0.137 | does not clear |
| `momentum_m3` | 3.71% | 8.71% | 0.484 | 0.426 (0.02682) | 0.230 | 0.409 | −0.017 → +0.179 | does not clear |
| `momentum_m4` | 3.71% | 10.85% | 0.484 | 0.342 (0.02154) | 0.185 | 0.552 | +0.210 → +0.367 | clears |
| `momentum_m5` | 3.71% | 10.85% | 0.484 | 0.342 (0.02153) | 0.185 | 0.530 | +0.188 → +0.345 | clears |
| `trend_t0` (pinned) | 3.78% | 5.87% | 0.677 (net-long ETF book) | 0.644 (0.04058) | 0.462 | 0.042 | −0.602 → −0.420 | no carry |
| `trend_t1` | 3.78% | 6.28% | 0.649 | 0.602 (0.03791) | 0.417 | 0.393 | −0.209 → −0.024 | does not clear |
| `trend_t2` | 3.78% | 6.04% | 0.688 | 0.626 (0.03941) | 0.455 | 0.104 | −0.522 → −0.352 | does not clear |
| `trend_t3` | 3.78% | 6.56% | 0.688 | 0.577 (0.03632) | 0.420 | 0.044 | −0.533 → −0.376 | does not clear |
| `trend_t4` | 3.78% | 7.64% | 0.644 | 0.495 (0.03118) | 0.341 | 0.024 | −0.471 → −0.317 | does not clear |
| `demotion_d1` | 3.71% | 2.13% | 0.493 | 1.744 (0.1098) | 0.958 | −0.384 | −2.13 → −1.34 | certified uneconomic |
| `demotion_d2` | 3.71% | 2.21% | 0.491 | 1.682 (0.1059) | 0.920 | 0.091 | −1.59 → −0.83 | certified uneconomic |
| `demotion_d3` | 3.71% | 2.16% | 0.494 | 1.720 (0.1084) | 0.946 | −0.120 | −1.84 → −1.07 | certified uneconomic |
| `demotion_d4` | 3.71% | 2.69% | 0.495 | 1.380 (0.0869) | 0.760 | −0.951 | −2.33 → −1.71 | certified uneconomic |
| `demotion_d5` | 3.71% | 2.50% | 0.430 | 1.485 (0.0936) | 0.732 | 0.335 | −1.15 → −0.40 | certified uneconomic |
| `demotion_b2` | 3.71% | 3.25% | 0.494 | 1.141 (0.0719) | 0.627 | 0.081 | −1.06 → −0.55 | certified uneconomic |

M2 and M3 would clear a 3.3%-yield hurdle. That column carried no registered consequence
(the M-series read says so: the hurdle is a promotion conjunct at M6, not a fragility test),
so nothing about `momentum_v1` moves. T1 comes within 0.024 of the restated hurdle and no
trend cell clears it; the T-series admission refusal stands. The certification set fails by
a wide margin under both conventions.

## Replication-route records (rf = 4.21%, French RF, last 252 sessions of the pinned 202607 file)

Source per row: `results/2026-09-vol-momentum/summary.json` (`economics`, `in_house.mean_scale`);
`results/2026-09-fundamentals/replication/replication.json` and
`results/2026-09-earnings/replication/replication.json` (`arms.<arm>.economics`:
`hurdle_sharpe`, `runnable_vol`, `own_vol_annual`, `figure_f3`; `combination`). The margin is
F3 − hurdle, F3 = ρ × prior − cost drag. Each arm's volatility basis is its *runnable*
volatility, min(12%, 2 × own volatility), which is the 12% target for `vol_momentum`, `be_me`,
`gp_at` and both combinations. The arm's scale is runnable ÷ own volatility; L/E = scale / 2.

| record | vol (basis) | L/E (source) | recorded hurdle | restated, y = 3.3% | F3 | margin recorded → restated | recorded verdict |
|---|---|---|---|---|---|---|---|
| `vol_momentum` | 12.0% (target) | 0.539 (0.5 × mean_scale 1.079; the read's "46% idle") | 0.351 | 0.224 | 0.368 | +0.018 → +0.144 | promoted |
| `vol_momentum` at the assumed L/E | 12.0% (target) | 0.5 (assumed, 1×-gross equal long/short) | 0.351 | 0.213 | 0.368 | +0.018 → +0.155 | — |
| `be_me` | 12.0% (runnable) | 0.758 (scale 1.517) | 0.351 | 0.284 | 0.041 | −0.309 → −0.243 | fails bar, economics |
| `gp_at` | 12.0% (runnable) | 0.967 (scale 1.935) | 0.351 | 0.342 | 0.192 | −0.159 → −0.150 | fails economics |
| `at_gr1` | 11.54% (runnable) | 1.0 (scale 2) | 0.365 | 0.365 | 0.099 | −0.266 → −0.266 | fails economics |
| `oaccruals_at` | 8.84% (runnable) | 1.0 (scale 2) | 0.476 | 0.476 | 0.116 | −0.360 → −0.360 | fails fidelity, economics |
| `niq_su` | 9.49% (runnable) | 1.0 (scale 2) | 0.444 | 0.444 | 0.028 | −0.416 → −0.416 | fails fidelity, economics |
| `chcsho_12m` | 9.90% (runnable) | 1.0 (scale 2) | 0.425 | 0.425 | 0.046 | −0.379 → −0.379 | fails economics |
| fundamentals combination | 12.0% (target; realized 6.24%) | 1.0 (record: at the 2× gross ceiling) | 0.351 | 0.351 | 0.087 | −0.264 → −0.264 | fails economics |
| `ear_monthly` | 9.52% (runnable) | 1.0 (scale 2) | 0.442 | 0.442 | 0.117 | −0.325 → −0.325 | fails economics |
| earnings combination | 12.0% (target; realized 11.9%) | 1.0 (record: at the 2× gross ceiling) | 0.351 | 0.351 | 0.275 | −0.076 → −0.076 | fails economics |

The vol-momentum margin widens from 0.018 to about 0.14–0.15 Sharpe, which is roughly
1.7–1.9% a year at 12% volatility, consistent with the corrected "about 2% a year" in
`docs/vol_momentum_read_2026-09-24.md`. The fundamentals read's addendum put `gp_at`'s
move at about 0.01 (0.009 at 3.3%); the table reproduces 0.009. Every arm or combination
that runs at L = E is unchanged by construction, so no replication verdict is near a flip.

## Records with no hurdle number to restate

- **Discovery arms carried to the forward read** (`earnings_daily_v1`: `ear_event`,
  `announcement_premium`; `buffer_daily_v1`, `buffer_monthly_v1`). Their viability test
  compared the IC ceiling at the one-sided lower-95 IC with cost drag *plus* a hurdle, and
  each failed before any hurdle was added (`docs/earnings_read_2026-09-24.md`,
  `docs/buffer_read_2026-09-25.md`). No hurdle number is recorded; a lower hurdle changes none
  of them. Their first forward read computes it under the amended convention.
- **`forward_v1`** (`docs/forward_design.md` §4): "periodic net Sharpe above the T-bill
  hurdle, converted as the record declares". `research/scripts/forward_read.py` now states it
  through the shared function: rf = mean FRED DTB3 bond-equivalent yield over the forward
  sessions, y = `retail_v2.toml` `cash_income_bps_annual`, L/E = the mean long weight sum,
  volatility = realized annualized over the forward sessions. No number exists before the
  first scheduled read (≥ 126 forward sessions after 2026-09-24, about 2027-03-24).
- **Comparison reads against a control** (`characteristics_v1`, `breadth_v1`,
  `learned_xsection_v1`, `construction`, `sizing_lab_v1`) record no hurdle.

## What remains open

The account statement's cash and margin rates (`retail_v2.toml` carries listed figures
retrieved 2026-09-25). Whether the live book's idle cash in fact earns anything in the paper
account is unverified; every restated margin above is conditional on it.
