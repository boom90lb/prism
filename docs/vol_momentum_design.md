# Volatility-scaled momentum — design and registration (`vol_momentum_v1`, replication route)

**Status: registered 2026-09-24.** The registration is the commit that adds
this file and `registrations/2026-09-vol-momentum.toml` (SPEC §4). It is the
program's first use of the replication route. Every parameter is in the TOML.
None of the quantities the criteria read has been computed: the scaled
reference series, the scaled in-house series, their correlation, the cost
drag, the hurdle, the in-house interval.

## 1. What is registered

The live book's configuration (12−1 momentum, top/bottom decile, S&P 500,
canonical engine) with its gross scaled by the published constant-volatility
rule:

> At each decision bar *t*, σ̂ = the annualized standard deviation of the
> book's 126 unscaled daily net returns ending *t*. The scale is
> *s* = min(2, 0.12 / σ̂). It applies from session *t+1* until the next
> decision bar (N1, N2).

The source is Barroso and Santa-Clara (2015), *Momentum has its moments*, JFE
116(1). The paper scales the momentum factor to a 12% target using six months
of daily returns, re-set monthly. The 2× cap is not the paper's. It is the
implementation's Reg T gross ceiling, and it is applied to the reference
series as well, so the two follow one rule.

## 2. Why this configuration

The public factor lab (`docs/public_factor_lab_read_2026-09-24.md`) found:

- The rule **supported** on French momentum over 1931–2026: Sharpe 0.368 →
  0.776, net of in-house cost rates. After publication it still added +0.228
  (95% interval +0.076 to +0.428). It halved the 1932 and 2009 losses.
- The in-house momentum book tracks French momentum at ρ = 0.88.
- The ratified crash de-gross that SPEC §6 names was **contradicted**.

This configuration is the candidate that has a public record behind it.

**Prior exposure, disclosed.**

- The lab evaluated this rule, the variance-managed rule and the ratified
  de-gross on public series (`sizing_lab_v1`, 8 rows).
- This rule is registered here because its registered lab verdict was
  "supported" on both bases. That was not a choice on post-publication level:
  the variance-managed rule had the higher post-publication Sharpe on momentum
  (0.529 against 0.388) and was not chosen. So the choice does not inflate the
  post-publication prior this route reads.
- The unscaled in-house book's figures are known: net Sharpe 0.679, 12.4%
  volatility, fidelity 0.88 to unscaled momentum.
- Constant-volatility scaling was named as a candidate before the lab ran.

## 3. The four criteria (SPEC §4), as computed

1. **Publication bar and prior.**
   - The reference is French daily momentum at the lab's vintage (sha256
     pinned), scaled by the rule, gross of costs, scored from its session
     1,261 (as in the lab) to 2026-07-31.
   - The bar is t = Sharpe × √years ≥ 3 over ≥ 20 years.
   - The prior is the reference's annualized Sharpe from 2015-04-01, since
     11.3 years of post-publication history is ≥ 10.
   - A prior ≤ 0 fails.
2. **Fidelity.**
   - The scaled in-house series is the committed `momentum_control` daily net
     returns (sha256 pinned), multiplied by the in-house scale.
   - The scale is decided every 21st session from 2022-03-29, which coincides
     with the book's decision bars (`folds/demotion_b1.toml`). The first
     decision is at scored session 126, the first with 126 returns, so the
     series is scored from the session after it.
   - Criterion: the daily Pearson correlation with the reference over their
     overlap must be ≥ **0.75**. The registered floor is above the contract
     minimum of 0.5, because a book tracking below 0.75 would be a different
     object from the one the public record describes.
   - Weekly correlation is reported beside it.
3. **Economics at our costs.**
   - The scaled book is charged *s_t* × its recorded daily cost (commission,
     spread, impact and borrow, from its run record `costs.csv`, sha256
     pinned). Scale changes are charged at 10 bp one-way on |Δ*s*| × the
     book's recorded gross exposure.
   - Cost drag = annualized mean charged cost ÷ annualized volatility of the
     scaled net returns.
   - Figure F3 = ρ × prior − drag.
   - **Hurdle:** in-house books earn nothing on cash (`retail_v1.toml`,
     `cash_income_bps_annual = 0.0`). So the book must out-earn T-bills in
     total return at the volatility it runs at: hurdle = rf / 0.12, with rf the
     annualized mean French RF over the pinned file's last 252 sessions.
     F3 must exceed it.
   - The same comparison with a zero hurdle is reported beside it. That is
     what applies if idle equity earns the bill rate, for example held in a
     T-bill fund. It is not the criterion.
4. **No in-house contradiction.** The studentized circular block bootstrap
   interval (42-session blocks, 2,000 resamples, 95%) for the scaled in-house
   book's annualized net Sharpe must not lie entirely below F3.

**Promotion.** All four hold → promoted, meaning eligible for the live book.

- Changing the live book, arming any sizing rule, and replacing the de-gross
  that SPEC §6 names are owner acts. A promotion here is evidence for them,
  not the act.
- Any criterion failing → not promoted.
- The row stays in the trial log either way, and a changed parameter is a new
  registration.

**Method limits, stated before the result.**

- The in-house scaled series is an overlay on the recorded book: daily return
  × scale, costs × scale, plus scale-change costs. It is not a re-run of the
  engine, which has no time-varying gross scale in backtest. The overlay
  ignores the small compounding difference between a book rebalanced to scale
  at each decision and one scaled daily.
- The live path is the regime step's gross-scale hook, which fires on refresh
  bars only (`src/prism/live/daily.py`). That matches this rule's cadence.
  Wiring it is the arming act.
- N6: the unscaled book's N_eff is 46.7 (characteristics read). Proportional
  scaling leaves it unchanged, and it is reported by citation.

## 4. Forward kill rule (runs from this commit, and after promotion)

**Stream.** The forward stream is `forward_v1`'s FB arm
(`docs/forward_design.md`), the same configuration, on sessions from
2026-09-24, scaled by this rule. FB's pre-forward sessions from the same read's
dataset enter only the volatility estimate, never a forward statistic.

**Reads.** The first read comes at the first as-of date with ≥ 126 forward
sessions (about 2027-03-24), then one every 63 sessions. Any one of these
kills:

- (a) the scaled forward net Sharpe's 95% interval (same bootstrap) lies
  entirely below zero;
- (b) the scaled forward book's drawdown from its forward peak exceeds 25%;
- (c) the scaled forward book's daily correlation with the reference (the
  latest French vintage, hash recorded) is below 0.5 over their overlap.

If the owner arms the rule on the paper book, the paper stream is read under
the same three conditions as well.

## 5. Outputs

Everything goes under `results/2026-09-vol-momentum/`:

- `summary.json`: every criterion's inputs and outcome, hashes, code commit;
- `series.csv`: the daily reference and in-house scaled series with their
  scales.

The trial log is `trials/vol_momentum_v1.jsonl`, with one row. The read
record is a separate `docs/vol_momentum_read_<date>.md`.

## 6. What this does not do

- It does not arm anything, change the live book, or edit SPEC §6 or
  `docs/sizing_preregistration.md`.
- It does not test residual momentum, other targets, other lookbacks, or the
  variance-managed rule. Each of those is a new registration.
- It does not replace `forward_v1`. That registration's arms and reads are
  unchanged; this one reads FB's forward stream through its own rule.
