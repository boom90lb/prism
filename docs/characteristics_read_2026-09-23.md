# Characteristics composite — read record, 2026-09-23

**Kind:** read record for `registrations/2026-09-characteristics.toml`
(`docs/characteristics_design.md`). **Selection set:** `characteristics_v1`.
**Registration commit:** `04dcb34` (the first commit to carry the registration,
before any arm ran). **Budget:** 6 declared, 6 spent. **Records:** six run
records, one per arm, read by `research/scripts/registered_read.py`; the
comparison and per-arm read are `results/2026-09-characteristics/comparison.json`
and `read.json`, and each arm's daily net returns are under `returns/`.
**Trial log:** `trials/characteristics_v1.jsonl` (6 started, 6 finished).

## Verdict

**The set closes. Idiosyncratic volatility alone is killed; the other five
survive. Residual momentum is the one candidate worth carrying forward, and
it carries forward only to a prospective registration: nothing here
promotes.**

## Sample and conventions

Canonical engine, $100,000 marked equity at the first decision, frozen dataset
`data/frozen/demotion_b1` (`data_id bedf8c4c…`), fold table
`folds/demotion_b1.toml`, cost stack `examples/reference/costs/retail_v1.toml`,
experiment profile, `research_vintage` visibility, fold portfolio carried.
Folds 0-3 are excluded for every arm alike by the shared calibration family's
admission rule (the same exclusion the construction read recorded), so the
scored sample is 1,056 sessions, 2022-03-29 to 2026-06-12. Modeled fills,
price-return convention, no distributions.

**Parity (Run).** `momentum_control` reproduces the construction read's
`decile_control` at $100,000 exactly (annualized net Sharpe 0.678903), and its
frozen momentum panel digest equals the construction freeze's
(`23fda286…`). The two zero-weight components change nothing in the control
book.

## Results

| arm | net Sharpe | gross Sharpe | ann. return | ann. vol | max DD | charged cost | N_eff | held-book IC ± se | DSR vs set |
|---|---|---|---|---|---|---|---|---|---|
| `momentum_control` | 0.679 | 0.719 | 7.95% | 12.4% | 16.3% | 2.09% | 46.7 | 0.049 ± 0.041 | 0.507 |
| `residual_momentum_only` | **0.829** | 0.958 | 3.83% | 4.7% | **4.3%** | 2.49% | **71.0** | 0.038 ± 0.022 | **0.626** |
| `idiosyncratic_vol_only` | −0.529 | −0.486 | −6.30% | 11.1% | 32.8% | 2.02% | 32.1 | −0.006 ± 0.041 | 0.007 |
| `momentum_residual_momentum` | 0.759 | 0.828 | 5.75% | 7.8% | 9.1% | 2.22% | 53.2 | 0.043 ± 0.026 | 0.572 |
| `momentum_idiosyncratic_vol` | 0.144 | 0.197 | 0.86% | 8.4% | 12.3% | 1.84% | 32.9 | 0.032 ± 0.034 | 0.142 |
| `composite_three` | 0.271 | 0.342 | 1.56% | 6.5% | 8.6% | 1.92% | 43.2 | 0.032 ± 0.026 | 0.209 |

Net Sharpe is the record's own (annualized, excess of the declared cash
convention). `N_eff` is the participation ratio of the covariance of
position-weighted daily return contributions, the certified diagnostic's
definition, which the read tool reproduces on the certified B1 record to
1e-13. The held-book IC is the rank correlation of target weight with the next
21-session return over held names, one cross-section per decision (50); it
measures leg separation and is not the full-cross-section IC B1 reported.
DSR deflates each arm against the six rows of this set.

## Kill rule

Registered: an arm whose paired studentized 95% interval for (arm − control)
net periodic Sharpe lies entirely below zero is dead.

| arm − control, per bar | lower | upper | two-sided p | verdict |
|---|---|---|---|---|
| `residual_momentum_only` | −0.064 | 0.073 | 0.79 | survives |
| `idiosyncratic_vol_only` | −0.175 | **−0.001** | 0.083 | **killed** |
| `momentum_residual_momentum` | −0.014 | 0.022 | 0.56 | survives |
| `momentum_idiosyncratic_vol` | −0.076 | 0.004 | 0.105 | survives |
| `composite_three` | −0.067 | 0.011 | 0.189 | survives |

The kill is marginal and the record says so: the interval's upper end is
−0.0010 per bar while the reported two-sided p is 0.083. The two come from
different constructions inside `prism.validation.compare`; the registration
names the interval, so the interval decides.

## What the numbers say, and what they do not

- **Idiosyncratic volatility had the wrong sign on this sample.** The
  low-volatility leg lost to the high-volatility leg across 2022-2026, a period
  that rewarded high-beta, high-volatility large caps. Every arm that holds it
  is dragged down roughly in proportion to its weight.
- **Residual momentum is a different book, not a better-levered one.** It runs
  at a third of the control's volatility (4.7% against 12.4%), with a
  quarter of its drawdown and half again its breadth (N_eff 71.0 against
  46.7). Its return per unit of gross is lower. Its IC's one-sided 95% lower
  bound is 0.038 − 1.645 × 0.022 = +0.002, where the certified B1 book's was
  −0.009; that is the first bound in this program's record to sit above zero,
  and it sits there by 0.002.
- **None of the survivors is distinguishable from the control.** The largest
  point improvement, residual momentum's +0.15 annualized, has an interval
  from −1.15 to +1.02. 1,056 daily observations cannot separate these books,
  as the construction read found for its arms.
- **DSR 0.63 is against six same-sample rows.** It is not the promotion
  statistic: promotion requires DSR above 0.5 on data after the candidate's
  registration (SPEC §4), and a candidate registered today has none yet.

## Carried forward

`residual_momentum_only` and `momentum_residual_momentum` are candidates for
a prospective registration whose evidence is data after that registration's
commit. No configuration of the live book moves; changing it is the owner's
act (SPEC §6).
