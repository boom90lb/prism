# The public baseline

This page is for anyone thinking of running the strategy this repository offers as
its baseline. It says what the strategy is, what return to expect and how uncertain
that is, what the expectation depends on, how many people can run it before they
compete with each other, and when the project stops treating it as approved.

It is a research result, not investment advice. Every figure comes from the
registered evaluation of 2026-10-16 ([results report](public_baseline_read_2026-10-16.md),
[design](public_baseline_design.md)) and its outputs under
`results/2026-10-public-baseline/`. Figures marked *derived* are plain arithmetic on
those outputs; the report shows the arithmetic.

## What it is

- **Universe.** Current S&P 500 members (point-in-time membership in the backtest).
- **Signal.** 12−1 momentum: each stock's return over the last twelve months,
  skipping the most recent month (Jegadeesh and Titman, 1993).
- **Portfolio.** Long the top tenth of stocks by that signal and short the bottom
  tenth, equal weight within each side.
- **Volatility scaling.** Exposure is scaled so that the portfolio's volatility,
  estimated from its last 126 daily returns, targets 12% a year, with gross
  exposure capped at twice equity (Barroso and Santa-Clara, 2015).
- **21 overlapping sub-portfolios.** The capital is split into 21 parts. Each part
  rebalances every 21 trading days on its own day, so exactly one part rebalances on
  each trading day. A new account builds up over its first 21 trading days. This
  spreads trading over the month instead of putting it all on one day, so the
  result depends much less on which day you happened to start.
- **Cash.** Idle equity above a 2% cash buffer is held in a T-bill ETF (SGOV, or BIL).
  Proceeds from short sales are collateral and earn nothing.
- **Orders.** Decided at the close and filled at the next day's open.

## Status on 2026-10-16

- **Approved under the project's rules.** The evaluation was registered before it
  ran and passed all six of its criteria. Approval makes the strategy eligible for
  the project's paper-trading account. It moves no money.
- **Not what the project's paper account trades.** That account trades plain,
  unscaled momentum rebalanced on one day a month. Running this strategy in the
  paper loop needs support for the 21-part book, the volatility scaling and the
  T-bill holding. That support is separate work, off by default, and switching it on
  for the project's own account is the account owner's decision.

## What to expect

The estimate starts from the public record of the same strategy, not from this
repository's backtest. The backtest covers only 3.6 years; the public series runs
from 1931, and its last 11 years come after the scaling rule was published.

| Step | Value |
|---|---|
| Sharpe ratio of Kenneth French's momentum factor scaled by the same rule, 2015-04-01 → 2026-08-31 (after the rule's publication) | 0.445 |
| × how closely this implementation tracks it (daily correlation) | × 0.903 |
| − modelled trading and borrowing costs, in Sharpe units | − 0.044 |
| **= expected Sharpe ratio of the long–short portfolio** | **0.358** |
| × 12% volatility = expected long–short return | 4.30%/yr *(derived)* |
| + T-bill ETF on 42% of equity, earning the bill rate less a 0.10% fee | + 1.65%/yr *(derived)* |
| = expected account return | 5.95%/yr *(derived)* |
| − T-bill rate (French one-month rate, last 252 trading days) | − 4.00%/yr |
| **= expected return above T-bills** | **+1.95%/yr** |

The account's Sharpe ratio above T-bills is therefore about 0.16 (*derived*), not
0.36. That is the number to compare with simply holding T-bills.

**Where your idle cash sits matters more than any setting.** About 44% of equity is
idle on average, because the long side uses only about 56% of it.

| Idle equity | Expected return above T-bills |
|---|---|
| In a T-bill ETF | **+1.95%/yr** |
| In broker cash paying 3.3% (a provisional listed rate, not verified for a margin account holding shorts) | +1.76%/yr |
| In cash earning nothing | **+0.30%/yr** |

**The uncertainty is larger than the edge** (*derived*, normal approximation).

- One standard error of the expected Sharpe ratio is about 0.27, which is about
  ±3.2%/yr at 12% volatility. It comes from the 11 years of public data after
  publication; uncertainty in tracking, costs and future decay comes on top.
- The chance that the true expected return above T-bills is positive is about 0.73
  with the T-bill ETF and about 0.54 with cash earning nothing.
- The chance that one account is ahead of T-bills after 5 years is about 0.62 with
  the T-bill ETF (0.65 after 10 years); with cash earning nothing it is about 0.52.
- Your own results cannot settle the question on any useful horizon: about 21 years
  to show the long–short Sharpe ratio is above zero, and about 100 years to show the
  edge over T-bills is positive.

**What would make it worse** (*derived*):

| Change | With the T-bill ETF | With cash earning nothing |
|---|---|---|
| None (the registered estimate) | +1.95%/yr | +0.30%/yr |
| Public record cut at 2025-12-31 (Sharpe 0.401 instead of 0.445) | +1.48%/yr | −0.17%/yr |
| Further decay to 0.4 × the full-history Sharpe (0.359) | +1.03%/yr | −0.63%/yr |
| T-bill rate of 5% instead of 4%, all else unchanged | +1.38%/yr | −0.70%/yr |
| Fills worse than the modelled open, per 0.01 of Sharpe lost | −0.12%/yr each | −0.12%/yr each |

- The estimate leans on January–August 2026: with cash earning nothing, it is
  positive only with those months included.
- Published strategies decay. Momentum's own Sharpe ratio fell from 0.63 before its
  1993 publication to 0.35 after (`results/2026-09-factor-lab/priors.json`). The
  scaling rule was published in 2015, and its Sharpe ratio since then (0.445) is about
  half its full-history figure (0.899).
- Not included: taxes; the effect of stocks missing from the backtest's history
  (it covers 574 of the 633 stocks that were ever members, which flatters tracking
  and cost); and any shortfall between the modelled opening fills and real ones.

## What it needs

- **A margin account that allows short selling.** Retirement accounts generally do
  not.
- **About $100,000 or more.** Orders are in whole shares. At $25,000, about a fifth of
  the stocks the strategy targets round to zero shares, and the account drifts from
  its target by about 1.5% a year, against 0.5% at $100,000 and 0.4% at $1,000,000.
  The measured return difference at $25,000 is not distinguishable from zero, but
  the account holds noticeably fewer of the strategy's stocks.
- **Idle equity in a T-bill ETF, or a verified cash yield.** Without either, the
  expected edge is about +0.30% a year. How a broker treats the ETF for margin is
  not verified. In the backtest, Reg T initial-margin headroom fell to 12% of
  equity at its tightest, with the ETF counted as a long position.
- **Short borrowing at about 0.5% a year**, the modelled rate. Stocks that are hard
  to borrow cost more; the rate is not verified against a real account.
- **Opening-auction orders filled at the open.** In the project's own paper account
  74% of fills (185 of 249) came from a 09:50 ET follow-up order rather than the
  opening auction.
- **Tolerance for frequent trading.** About 38 small orders a day (median about $72
  at $100,000). A buy and a sell of the same stock within 30 days of each other
  happen about 1,100 times a year, four times as often as with a single monthly
  rebalance. In a taxable
  account the wash-sale rule then defers losses; the tax cost is not estimated.
- **On a paper account, adjustments.** Alpaca's paper accounts credit no dividends
  and charge no borrow fees (Alpaca's paper-trading documentation, checked
  2026-10-16), so a paper account shows the T-bill ETF earning nothing and the short
  side costing less than it would.

## How many people can run it

Every copy of this strategy computes the same portfolio and trades the same stocks.
The more capital follows it, the more each user's orders move prices against the
others. The table shows how many other users with $100,000 each it takes for that
cost to use up the whole +1.95%/yr, under an assumed square-root market-impact
model. At $1,000,000 per user, divide by ten.

| Where the orders execute | This strategy (21 parts) | One monthly rebalance, everyone on the same day | One monthly rebalance, days spread at random |
|---|---|---|---|
| Opening auction, 1% of the day's volume | **548** | **65** | 1,375 |
| Opening auction, 3% | 1,646 | 199 | 4,170 |
| Closing auction, 8% | 4,391 | 531 | 11,157 |
| Over the whole day | 54,904 | 6,653 | 139,720 |

- **Read the first row.** The live loop's orders are opening-auction orders. It has no
  closing-auction orders and no way to spread an order over the day.
- **The edge shrinks long before that.** In the opening auction at 1%, 10 other users
  cost each user about 0.19%/yr, 100 cost 0.75%/yr, and 1,000 cost 2.56%/yr, more
  than the whole edge.
- **The figures are assumptions, not measurements.** The impact coefficient is taken
  from 0.5 to 1.0 around the middle value of 0.7 used in the table; across that range
  each entry roughly doubles or halves. The auction volume shares come from 2015–17
  industry notes. Paper trading cannot measure market impact.
- **Spreading rebalance days at random looks better on paper, but only works if
  people actually spread them.** A copy that rebalances on the first day it runs
  puts everyone who starts together in the "same day" column, and a single
  rebalance day carries luck: over the backtest the same strategy returned anywhere
  from 4.7% to 8.2% a year depending only on the day chosen. The 21-part version
  needs no coordination. At $100,000 its orders come in whole-share lumps, which is
  why it prices below the random-day case in this model.
- **Timing never removes the overlap in holdings.** Everyone holds the same stocks,
  so a momentum crash hits every copy on the same day. The scaled public momentum
  series has lost 48% from a peak (1937–1942) and 22% since 2015 (2016–2017)
  (*derived* from `results/2026-10-public-baseline/series.csv`).

## When approval is withdrawn

The project re-runs the strategy on data that arrives after 2026-10-16. The first
check is after 126 new trading days, about **2027-04-20**, and then every 63 trading
days. Approval is withdrawn, and only a new registration can restore it, if:

- the new data's Sharpe ratio is below zero with 95% confidence;
- the loss from the new data's peak exceeds 25%;
- the daily correlation with the scaled public momentum series falls below 0.5; or
- on two checks in a row, the expected Sharpe ratio recomputed on the newest public
  data falls below the T-bill benchmark.

## Where the numbers come from

- [Results report](public_baseline_read_2026-10-16.md): every criterion, every
  variant, the arithmetic behind the derived figures, and how to check each number.
- [Design and registration](public_baseline_design.md), committed before anything ran.
- `results/2026-10-public-baseline/summary.json`: the machine-readable results;
  `capacity/` holds the crowding tables.
- [SPEC.md](../SPEC.md) §§3–4: the project's rules for costs, the T-bill benchmark
  and approval.
