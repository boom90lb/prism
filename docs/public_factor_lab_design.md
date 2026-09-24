# Public factor lab — design and registration (`sizing_lab_v1`)

**Status: registered 2026-09-24.** The registration is the commit that adds
this file and `registrations/2026-09-factor-lab.toml` (SPEC §4). Every
parameter lives in the TOML; the script reads it and this document explains it.
Nothing has been evaluated at registration. The only public data inspected
beforehand was file schemas and date ranges (below).

## 1. Why

The in-house sample is 2020–2026. On it, telling residual momentum from raw
momentum apart by 0.15 Sharpe at their measured return correlation of 0.58
would take about 150 years of daily data, and reaching t = 2 on the level of a
0.8 Sharpe takes about 8. Public factor libraries hold 70–100 years of daily
long−short returns at no cost. They answer three questions the in-house sample
cannot:

- **A. Priors.** What each published premium earned before and after its
  publication, whether it clears the SPEC §4 publication bar, and how the
  premia correlate. This is what the replication route consumes.
- **B. Fidelity.** How closely the in-house books track the public series they
  implement (SPEC §4 criterion 2).
- **C. Sizing.** Whether the ratified crash de-gross
  (`docs/sizing_preregistration.md` §3) and two published volatility-scaling
  rules help or hurt momentum over its full history, including 1932 and 2009.
  Before this lab, residual momentum's crash behaviour was treated as untestable
  without buying data (`docs/forward_design.md` §6). JKP's free daily US
  residual-momentum series (1953-05 → 2025-12) makes it testable now.

## 2. Data

| source | series | first date | file |
|---|---|---|---|
| Kenneth French, 3-factor daily | Mkt-RF, RF | 1926-07 | `F-F_Research_Data_Factors_daily` |
| Kenneth French, 5-factor 2×3 daily | SMB, HML, RMW, CMA | 1963-07 | `F-F_Research_Data_5_Factors_2x3_daily` |
| Kenneth French, momentum daily | Mom (UMD) | 1926-11 | `F-F_Momentum_Factor_daily` |
| Kenneth French, short-term reversal daily | ST_Rev | 1926-01 | `F-F_ST_Reversal_Factor_daily` |
| JKP, US themes, daily, capped value weight | 13 themes | 1926–1972 by theme | `[usa]_[all_themes]_[daily]_[vw_cap]` |
| JKP, US residual momentum, daily | resff3_12_1 | 1953-05 | `[usa]_[resff3_12_1]_[daily]_[vw_cap]` |
| JKP, US large-cap residual momentum | resff3_12_1 | monthly | `[usa]_[resff3_12_1]_[large]` |

Raw files are cached in `data/public/` (gitignored). Every output records each
file's sha256, its download time, and French's CRSP vintage line, so a vendor
revision shows up as a changed hash, never silently. Returns are the vendors'
percent or decimal figures converted to decimal. A missing-value code inside a
window raises (N7).

**Prior exposure, disclosed.** On 2026-09-23, while checking availability, the
French momentum file and the two JKP daily files were downloaded and their
column names, row counts and first and last dates printed. `head(2)` printed two
rows of each (four daily returns in all). On 2026-09-24 every source was
parsed and checked for gaps and calendar alignment (§5, Bases). No statistic
was computed on any public series. The published results these series come from (Jegadeesh–Titman,
Blitz–Huij–Martens, Barroso–Santa-Clara, Moreira–Muir, Daniel–Moskowitz) are
known and are why the rules below exist. That is literature exposure, which
this lab measures rather than hides.

## 3. Part A — priors (uncounted, nothing selected)

For each series in `[[series]]`, over its full history:

- annualized Sharpe of the daily long−short return, and its t-statistic
  (Sharpe × √years);
- where a publication month is pinned, the same statistics before and after
  it;
- the §4 publication bar: t ≥ 3 over at least 20 years;
- the §4 prior Sharpe: the post-publication Sharpe when that history spans ten
  years or more, else 0.4 × the full-sample Sharpe; a prior ≤ 0 fails.

For the eight series and the thirteen JKP themes: correlation matrices over
2000-01 → 2025-12 and 2020-01 → 2025-12, at daily and monthly frequency.
Monthly returns are compounded daily returns. JKP themes are descriptive only:
a theme averages many papers' factors, so it has no single publication date.

## 4. Part B — fidelity (uncounted measurement)

The Pearson correlation of each in-house book's daily net returns with its
reference series, over their overlap:

| in-house book | reference |
|---|---|
| live book B1 (`results/demotion_b1/returns.csv`) | UMD; JKP momentum theme |
| `momentum_control` (characteristics read) | UMD; JKP momentum theme |
| `residual_momentum_only` (characteristics read) | JKP resff3_12_1 daily; JKP large-cap resff3_12_1 monthly (reported beside it) |

Daily is the SPEC §4 criterion-2 statistic. Weekly correlation (Friday-close
compounding) is reported beside it, because in-house returns follow next-open
fills and a one-session timing offset lowers daily correlation without
changing what the book holds. For each pair the record also reports the
regression beta and the implementation Sharpe implied by the tracking
component, ρ × the reference's prior Sharpe.

**This part decides nothing.** A candidate takes the replication route only
through its own registration, which names the floor (≥ 0.5) and the kill rule.
Both in-house candidates were already run in-house (`characteristics_v1`). A
replication registration for either must disclose that. Their implementations
were pinned before those runs (`docs/characteristics_design.md` §2), so the
in-house selection among six arms did not choose the implementation.

## 5. Part C — sizing rules on public momentum (counted: `sizing_lab_v1`)

**Selection set.** Two base series × four rules = eight rows, the whole budget.
The set closes when the eight have run. A retry after a crash is `kind =
"retry"` and spends nothing. Every row is appended to
`trials/sizing_lab_v1.jsonl` by the run itself.

**Bases.** UMD (French, daily, 1926-11 →) and RESMOM (JKP resff3_12_1, daily,
1953-05 → 2025-12). Every arm of a base runs on the base's sessions that also
appear in the French market series. A structural check before registration
(parsing, gaps and calendars only; no statistics) found every series free of
gaps inside its range. It found one session present in RESMOM and absent from
the market series: 1985-09-27, when the NYSE was closed for Hurricane Gloria.
That session is dropped from every RESMOM arm alike, and the output lists it.

**Rules.** Scale *s* multiplies the base's daily return. It is decided at the
close of session *t* from data through *t* and applies from session *t+1*
(N1, N2). Scale is re-set on every 21st session counted from the series'
first session, the refresh schedule, except where a rule says otherwise.
*s* ≤ 2 (the Reg T gross ceiling relative to a 1× base).

| rule | definition |
|---|---|
| `none` | *s* = 1. The control. |
| `crash_degross_v1` | `docs/sizing_preregistration.md` §3, pins verbatim. State *S_t* = 1 iff the compounded French total market return (Mkt-RF + RF) over the 21 sessions ending *t−1* is below θ = −0.017981751321183567. On the first session with *S_t* = 1, *s* = 0.5 from *t+1*. At a refresh, a de-grossed book returns to *s* = 1 when the run of consecutive *S* = 0 sessions ending at that refresh is at least 5. This is the reading of §3.2's "after 5 consecutive sessions … at the next scheduled refresh" that never re-grosses into a live crash state. The multiplier does not compound. |
| `constant_vol_bsc` | Barroso and Santa-Clara (2015): at each refresh, *s* = min(2, 0.12 / σ̂), with σ̂ the annualized standard deviation of the base's 126 daily returns ending *t*. |
| `variance_managed_mm` | Moreira and Muir (2017), made causal: at each refresh, *s* = min(2, RV̄ / RV), with RV the realized variance of the base's 21 daily returns ending *t* and RV̄ the mean of every refresh's RV to date. The rule scales only from the 60th refresh; before that, *s* = 1. |

**Evaluation window.** Every arm is scored from session 1,261 of its base (the
longest warm-up, 60 refreshes) to the base's last session. Before that the
arms are identical by construction. Sub-windows: pre-2020 (→ 2019-12-31);
2020 →; and, for the two published rules, post-publication (from 2015-04 and
2017-08).

**Costs (N3).** The public series are gross paper portfolios, so each arm is
charged two things:

- **Scale changes:** |Δ*s*| × 2 (the base's gross) × 10 bp one-way, on the
  session the new scale takes effect.
- **Base rebalancing:** *s* × 2 × the in-house measured cost per unit of gross
  per year, spread evenly per session. That rate is 0.499%/yr for momentum and
  0.594%/yr for residual momentum (characteristics read: 2.09% and 2.49% over
  4.19 years at gross 1).

Pre-1990 costs were higher than this. So absolute pre-1990 net figures are
indicative; comparisons between rules on the same base are what the set reads.

**Statistics per arm and window.** These are:

- annualized net Sharpe, volatility and mean scale;
- maximum drawdown;
- worst 21-session return and worst calendar year;
- returns over the crash windows (1932-06 → 08 for UMD only, 2009-03 → 05,
  2020-11);
- the fraction of sessions de-grossed (`crash_degross_v1`);
- total cost.

The rule arms are compared with `none` on the same base using the paired
studentized circular block bootstrap
(`prism.validation.compare.sharpe_difference_interval`: 63-session blocks,
1,000 resamples, seed 20260924, 95%). Each row also carries its DSR against
the eight rows' periodic Sharpes (N5). N6 does not apply: these are factor
time series, with no holdings to account.

**Verdicts, per base, fixed now.**

- `crash_degross_v1`, read on pre-2020 (its θ was measured on 2020–2026):
  - **contradicted** if the Sharpe-difference interval lies entirely below
    zero, or its maximum drawdown is not below `none`'s;
  - **supported** if the interval does not lie entirely below zero *and* both
    its maximum drawdown and its worst 21-session loss are smaller than
    `none`'s;
  - the rule is supported overall only if supported on both bases, and
    contradicted overall if contradicted on either.
- `constant_vol_bsc` and `variance_managed_mm`, read on the full evaluation
  window and post-publication:
  - **contradicted** if the full-window interval lies entirely below zero, or
    the post-publication interval does;
  - **supported** if the full-window interval lies entirely above zero *and*
    the post-publication point difference is ≥ 0;
  - otherwise **inconclusive**.

**Kill rule.** A rule contradicted on a base is dead for that base in this set.
**Promotion rule.** None: nothing in this set changes the live book.

- A supported `crash_degross_v1` is evidence for arming it. Arming stays the
  owner's act under `docs/sizing_preregistration.md` §5, which this file does
  not edit.
- A contradicted verdict is recorded as evidence against arming.
- Adopting a volatility-scaling rule into any book is a separate registration.

## 6. Outputs

Everything goes under `results/2026-09-factor-lab/`:

- `sources.json`: hashes, download times, vintages;
- `priors.json` and `correlations/*.csv` (Part A);
- `fidelity.json` (Part B);
- `sizing/summary.json` (every arm × window, the paired intervals, the DSRs,
  the verdicts);
- `sizing/returns.parquet`: every arm's daily net returns and scale, so every
  figure is recomputable.

The read record is a separate `docs/public_factor_lab_read_<date>.md`.

## 7. What this does not do

- It does not promote anything, arm the de-gross, or change the live book.
- It does not test residual momentum *as implemented in-house* before 2020.
  That needs a survivorship-complete panel (`docs/forward_design.md` §6). It
  tests the published characteristic on CRSP through JKP; Part B measures how
  far the in-house implementation is from it.
- It does not test factor momentum, industry momentum, or any fundamental
  theme as a strategy. Part A supplies their priors; strategies are later
  registrations.
