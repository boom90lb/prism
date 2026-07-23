# Tax wedge — measurement spec (uncounted diagnostic)

> **Status: SPEC ratified 2026-07-19 (owner, in-turn, as Q4 of the desk-review
> adjudication); implementation landed** (`research/scripts/tax_wedge.py`,
> measured artifact `results/tax_wedge_2026-07-19.json`, write-up
> `docs/tax_wedge.md`). Governs a ledger-computed diagnostic in the
> `account_size_floor` / `carry_flatten` pattern: one script, one tracked
> JSON, one doc section. Searches nothing, moves no ratified statistic,
> appends no trials-ledger row.

## 1. The architecture this spec implements

- **The constitutional viability gate stays pre-tax.** Spreads and impact hit
  every participant; tax is operator-specific (rate, state, account
  structure). A certification that bakes in one operator's marginal rate is
  portable to nobody — under the enterprise-expandability doctrine the
  certified numbers stay pre-tax and the tax read is a parameterized overlay.
- **The deployment read goes after-tax for this operator.** The A3
  micro-account and any GO-branch sizing decision consume the diagnostic's
  output, not the certified pre-tax number.

## 2. Why this is a wedge measurement, not a haircut

First-order, tax does not move the viability *sign*: strategy gains at
monthly cadence are ordinary-rate short-term gains, and the T-bill hurdle is
*also* ordinary income. Under symmetric contemporaneous taxation at rate τ,
excess-over-cash and its volatility both scale by (1−τ) and the periodic
Sharpe comparison is τ-invariant. The measurable wedge is the asymmetries:

1. **Wash-sale deferral.** Monthly re-entry into recently-lossed names of an
   overlapping decile book systematically defers loss recognition (timing
   drag, not loss destruction). Computed from the fills ledger's actual lot
   sequence — no model, replay of the wash-sale window over real lots.
2. **Payments-in-lieu on the short leg.** Non-qualified, asymmetric against
   the long leg's dividends; interacts with the measured −36.1 bps/yr
   dividend wedge (worsens after tax). Computed from the corporate-actions
   records the dividend-wedge instrument already pulls.
3. **Capital-loss netting cap.** Gains are taxed in the year earned; a large
   net loss banks at ~$3k/yr against ordinary income. Steady-state this is
   small; **in the crash year it is not** — see §3.
4. **State asymmetry.** Treasury interest is state-tax-exempt; trading gains
   are not. Scales with the state-rate parameter.

## 3. Outputs (both mandatory)

- **Steady-state wedge:** annualized after-tax drag vs the symmetric-τ
  baseline, computed over the certified window's ledger. Expectation recorded
  ex ante: tens of bps, the same order as the flatten (−19.4) and dividend
  (−36.1) wedges. The diagnostic exists to replace this expectation with a
  number.
- **Crash-year conditional wedge:** the same quantity computed on the
  worst-drawdown year of the sample and on a synthetic −30% book-return
  year (asymmetry 3 dominates: current-year gains taxed, crash losses
  banked at $3k/yr). This is a convexity penalty in after-tax space that
  co-moves with the momentum-crash state; averaging it into the steady-state
  number would hide exactly the tail the sizing pre-registration needs to
  price. The GO-branch sizing read (handoff §8 preconditions) consumes this
  number, not the average.

## 4. Parameters and portability

Federal marginal rate, state marginal rate, and filing-year loss-cap are
CLI parameters with no defaults baked into any committed artifact; the
tracked JSON records the parameter vector it was computed under. Entity and
election questions (trader-status mark-to-market under §475(f), which would
change the wash-sale treatment and which a monthly-cadence book very likely
does not qualify for) are real-money-adjacent decisions for a CPA and are
out of scope here — recorded, not answered.
