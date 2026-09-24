# Forward read of the residual-momentum candidates (`forward_v1`)

**Status: registered 2026-09-23.** The registration is the commit that adds
this file (SPEC §4). It spends nothing today. It pins, before a single
qualifying observation exists, the configurations, the data, the geometry and
the rules under which the two candidates carried forward by
`docs/characteristics_read_2026-09-23.md` can be killed or promoted. SPEC §4:
"a pre-registered configuration promotes on data after its registration date".
This file is that registration date.

## 1. The configurations, frozen

Each is an arm of `registrations/2026-09-characteristics.toml`, run exactly as
recorded there (same control specification, same component source at commit
`87153fd`, same construction, profile, cost stack, execution profile,
canonical engine, $100,000):

| id | arm of `characteristics_v1` | weights (momentum, residual_momentum, idiosyncratic_vol) |
|---|---|---|
| `F1` | `residual_momentum_only` | 0, 1, 0 |
| `F2` | `momentum_residual_momentum` | ½, ½, 0 |
| `FB` | `momentum_control` (the benchmark: the live book's configuration) | 1, 0, 0 |

Any change to a component's source, a parameter, the construction, the
profile or the cost stack makes a different configuration, and a different
configuration is not this registration's.

## 2. Data

The S&P 500 point-in-time membership rebuilt by
`prism.scripts.build_sp500_universe` at each read's as-of date, and Twelve
Data daily bars from 2020-01-01 to that date, frozen into a dataset by the
same legacy import the certified dataset used. Coverage is recorded, as it is
for every dataset in this program. Bars before 2026-09-24 enter only formation
windows (cost buckets, the closed-form band, eligibility history); no return
before 2026-09-24 enters a forward statistic.

## 3. Geometry

Rolling folds of the certified shape: 312 formation sessions, 63 test sessions
(the last fold truncated at the as-of date), step 63. The first test window
opens on the first session on or after **2026-09-24**. The fold table is
written at read time by this rule and content-hashed like every other; it may
not be written any other way.

## 4. Reads

- **Interim read**, at the first as-of date with at least 126 forward
  sessions (on or after about 2027-03-24): reports every number
  `research/scripts/registered_read.py` reports. It can kill and cannot
  promote. **Kill:** a candidate whose paired studentized 95% interval for
  (candidate − `FB`) net periodic Sharpe lies entirely below zero is dead.
- **Promotion read**, at the first as-of date with at least 252 forward
  sessions (on or after about 2027-09-24). A candidate promotes only if all of
  these hold on the forward sessions alone:
  1. `net_edge` under bucket spreads;
  2. DSR above 0.5, deflated against **nine** rows: the six rows of
     `characteristics_v1` that selected these candidates and the three rows
     of this set, so the selection is paid for;
  3. periodic net Sharpe above the T-bill hurdle, converted as the record
     declares;
  4. the live paper book's monitor not in contradiction.
- **Neither fires:** the set waits for more forward data. Nothing refills.

Promotion makes a candidate eligible to become or join the live book. Changing
the live book is the owner's act (SPEC §6), and a combination with the momentum
book is `docs/aim_portfolio_preregistration.md`'s jurisdiction.

## 5. Power, stated before any data

One year of daily returns estimates an annualized Sharpe with a standard error
of about √((1 + SR²/2)/T) ≈ 1.1 at SR ≈ 0.8 and T = 1. The central case is that
the promotion read is inconclusive at one year and needs two or three; the
momentum design recorded the same arithmetic for M6. Time is the weakest
evidence axis this program has.

## 6. The faster axis, conditional on data the program does not hold

A survivorship-complete panel back to 1995
(`docs/data_purchase_evaluation.md`: Norgate US Stocks Platinum, $630 a year,
reversed by the owner on 2026-07-17) would add 25 years outside the selection
cell, including 2008-09 and the 2009 momentum crash, where residual momentum's
claimed crash robustness is directly testable. If such a panel lands, `F1`,
`F2` and `FB` run on the S&P 500 point-in-time universe from the panel's start
to 2019-12 under the data gates of `docs/replication_preregistration.md` §3,
as three counted rows of this set that can kill and cannot promote. That is
written here so it cannot be designed after the panel is seen.
