# Prism

Prism is an open-source system for researching and paper-trading daily-bar,
cross-sectional US equity strategies. It ranks stocks on a signal, builds a
long–short portfolio, simulates trading it with next-open fills and modeled
costs, and evaluates the result with purged walk-forward testing and
multiple-testing adjustments. The same code trades an Alpaca paper account.

The project exists to produce numbers that can be believed. Each experiment is
written down and committed before it runs. Every configuration tried is kept in
an append-only ledger, and negative results are published beside positive ones.
Nothing here has been traded with real money, and nothing here is investment
advice.

Terms the records use without explanation are defined in the
[glossary](docs/glossary.md). Read the [limitations](docs/limitations.md)
before the numbers.

## What the evidence says (as of 2026-10-16)

The table shows recorded results. Each figure can be read from the record
cited beside it, except the few marked (derived), which are plain arithmetic on
recorded figures.

| Strategy | Recorded result | Status | Record |
|---|---|---|---|
| **12−1 momentum** on the S&P 500: buy the top decile and sell the bottom decile by trailing 12-month return, skipping the latest month, rebalanced every 21 sessions. This is the book the paper account trades. | Net Sharpe 0.465 out of sample, 2021-03-30 to 2026-06-12 (21 purged folds, 1,308 sessions). Deflated Sharpe 0.19 against the 17 configurations its experiments tried. Price returns, without dividends. | Profitable after modeled costs (`net_edge`). Not significant after the multiple-testing adjustment, which needs a deflated Sharpe of 0.95. | [`results/demotion_b1/summary.json`](results/demotion_b1/summary.json), [`claim_packet.json`](results/demotion_b1/claim_packet.json) |
| **Volatility-scaled momentum** (`vol_momentum_v1`): the same book scaled to 12% volatility by the Barroso–Santa-Clara rule (126-session window, gross capped at 2×). | Expected net Sharpe 0.368 (tracking correlation 0.895 × post-publication Sharpe 0.465 − cost drag 0.047) against a T-bill benchmark of 0.351 if idle cash earns nothing. The margin is +0.018 Sharpe, about +0.2%/yr at the 12% target volatility (derived: 0.018 × 12%). The results report's "about half a percent a year" uses another basis: the 4.7%/yr that 0.368 gives at the book's 12.7% realized volatility, less 4.2% bills. At a provisional 3.3% cash yield the benchmark is 0.224 and the margin +0.144 Sharpe, about +1.7%/yr at 12% (derived: 0.144 × 12%), or about +2.0%/yr on the report's basis. The post-publication Sharpe has t = 1.56 over 11.3 years, a standard error of about 0.30, or about 0.27 after tracking (derived). If the public series is cut at 2025-12, the figure is 0.312, below the benchmark. On the 202608 French vintage the figure is 0.3506 against 0.3508 at the recorded rf of 4.21% (passes at the 4.00% recomputed from that file; [read record](docs/public_baseline_read_2026-10-16.md)). | Approved under the project's rules; not deployed. | [`results/2026-09-vol-momentum/summary.json`](results/2026-09-vol-momentum/summary.json), [results report](docs/vol_momentum_read_2026-09-24.md) §3, [benchmark restatement](docs/hurdle_restatement_2026-09-30.md), [fundamentals report](docs/fundamentals_read_2026-09-24.md) |
| **Public baseline** (`public_baseline_v1`, T21): volatility-scaled momentum as 21 overlapping tranches, idle equity in a T-bill ETF. | Expected net Sharpe 0.358 (0.903 × 0.445 − 0.044) against a T-bill benchmark of 0.195; about +1.95%/yr over bills (derived: margin × 12%), +0.30%/yr if idle cash earns nothing. Deflated Sharpe 0.79 against 30 configurations. | Approved under the project's rules; not deployed. | [read record](docs/public_baseline_read_2026-10-16.md), [`summary.json`](results/2026-10-public-baseline/summary.json) |
| **Daily residual reversion** on the S&P 500 | Uneconomic at retail cost over a completed experiment of 17 trials. | Published negative result. | [Certification 001](docs/certifications/001-residual-reversion-daily-negative.md) |
| **ETF trend following** (T0–T4) | Net Sharpe 0.04, and it lost money in the momentum book's worst periods. | Not added. | [results report](docs/trend_t_series_read_2026-09-03.md) |
| **Six accounting factors** and one combination | Tracking correlation × post-publication Sharpe of 0.08–0.24 per factor, against benchmarks of 0.35–0.48, before any cost. | None approved. | [results report](docs/fundamentals_read_2026-09-24.md) |
| **Earnings-announcement return** | Monthly: 0.117 against 0.442. Combined with volatility-scaled momentum: 0.275 against 0.351. | None approved. Two daily variants wait for data from after their registration. | [results report](docs/earnings_read_2026-09-24.md) |
| **Residual momentum, a wider universe, a learned ranking, a rank buffer** | Each was compared with an unchanged control on the same data. None was distinguishable from its control, and the learned ranking was explained by momentum. | None approved. Same-sample comparisons can reject a configuration and never approve one. | [characteristics](docs/characteristics_read_2026-09-23.md), [breadth](docs/breadth_read_2026-09-23.md), [learned ranking](docs/learned_xsection_read_2026-09-23.md), [buffer](docs/buffer_read_2026-09-25.md) |

In-house backtest Sharpes measured on the same data a configuration was chosen
on are not quoted here. They are not expectations.
[`erratum_2026-10-14.md`](docs/erratum_2026-10-14.md) corrects how several
recorded results were described. It changes no verdict.

### The public baseline: tranched, volatility-scaled momentum (`public_baseline_v1`)

The baseline this repository offers is the volatility-scaled momentum book
above, unchanged in signal, universe and parameters, run as **21 overlapping
tranches**: each holds 1/21 of capital and rebalances on its own session, so one
tranche trades each day. Idle equity above a 2% cash buffer is held in a
**T-bill ETF**, because a retail account earns nothing on short-sale proceeds.
It was registered before any run ([design](docs/public_baseline_design.md)),
evaluated on 30 pre-declared configurations, and **approved on all six of its
replication criteria** ([read record](docs/public_baseline_read_2026-10-16.md),
[`summary.json`](results/2026-10-public-baseline/summary.json)). Nothing here
switches it on: the paper account still trades unscaled momentum, and changing
that is the account owner's decision. [baseline.md](docs/baseline.md) is the
page for users.

- **Expected net Sharpe 0.358** for the long–short book: tracking correlation
  0.903 × post-publication Sharpe of scaled French momentum 0.445 (2015-04 to
  2026-08, t 1.50) − cost drag 0.044. This is an expectation built from the
  public series, not the in-house backtest.
- **Against T-bills** (rf 4.00%): about **+1.95%/yr** with the T-bill ETF
  holding, +1.76%/yr if idle cash earns a 3.3% broker yield, and only
  **+0.30%/yr** if idle cash earns nothing. The account's excess-over-bills
  Sharpe is about **0.16**, not 0.358.
- **Uncertainty.** One standard error is about 0.27 Sharpe (±3.2%/yr), larger
  than the edge. The probability that the true expected excess over bills is
  positive is about 0.73 with the ETF and 0.54 if idle cash earns nothing.
  Cut at 2025-12 the post-publication Sharpe is 0.401, and the idle-cash case
  turns negative (−0.17%/yr). A deployment's own returns would take about 21
  years to reject a zero Sharpe.
- **Decay.** Momentum was published in 1993; Kenneth French's momentum factor
  kept a Sharpe of 0.35 after publication against 0.63 before it, a 45% fall
  close to the literature's average, and the volatility-scaling rule (2015) is
  the newer and less tested part. Low decay cannot be confirmed from a
  deployment's own returns: rejecting a zero Sharpe would take about 21 years.
  The deflated Sharpe ratio against the 30 configurations tried is 0.79.
- **Tranching costs nothing measurable and removes timing luck.** Across the 21
  possible rebalance dates a single-date book's annual return ranged from 4.7%
  to 8.2% (standard deviation 0.97%/yr); the tranched book sits at the middle,
  and its paired Sharpe difference against the average date is +0.014
  [−0.009, +0.037].
- **Crowding (scenarios, not measurements).** Every deployment holds the same
  names, so tranching spreads trading across days but does not diversify
  holdings. Under assumed square-root impact, other users' trades use up the
  whole edge at about **65 users of $100k** if everyone rebalances a
  single-date book on the same day in the opening auction, about **548** for
  the tranched book, and about 4,400 if the tranched book traded in the closing
  auction. Users at $1M each: divide by ten. See the read record §9.

## What the evidence does not cover

- **Real money.** No result comes from a funded account. This repository
  documents paper trading only, and no configuration has met the project's
  own conditions for real money ([`SPEC.md` §6](SPEC.md#6-deployment-gate)).
- **Actual fills.** Every performance figure is simulated: decided at the
  close, filled at the next open, with modeled costs.
- **Dividends.** The in-house backtests are price returns. Momentum's net
  dividend flow was measured at −36 bp/yr (erratum E3), so the recorded books
  are not total-return figures.
- **Delisted stocks.** The momentum record covers 574 of the 633 stocks that
  were ever in the index over its window (coverage 0.907). The missing names
  are mostly ones that left the index, so the gap flatters the backtest.
- **Size.** Before the public baseline, modeled cost did not grow with
  assets: the impact term that depends on traded volume is zero in every
  earlier record. `public_baseline_v1` charges it at an assumed coefficient
  (`examples/reference/costs/retail_v3.toml`), and its capacity figures are
  scenarios under assumed parameters; paper fills cannot measure impact
  ([limitations §13](docs/limitations.md#13-capacity-and-crowding-measured-only-under-assumptions)).
- **The approved rule is not what trades.** The paper account trades unscaled
  momentum. The volatility-scaled rule is approved but not deployed. Changing
  the live book is the account owner's decision.
- **Paper fills.** 74% of the paper account's fills (185 of 249 in the tracked
  capture) are market orders sent at 09:50 ET by the morning completion sweep,
  not opening-auction fills. Alpaca paper accounts credit no dividends and
  charge no borrow fees.
- **Small accounts.** Whole-share opening-auction orders need about $100k to
  hold the roughly 100-name book faithfully. At $10k the account holds a
  different portfolio ([account-size floor](docs/account_size_floor.md)).
- **Taxes and selection.** No figure is after tax, and tranching about
  quadruples wash-sale pairs (1,102 a year against 274 for a single-date
  book). The volatility scaling rule was chosen in a sizing experiment of 8
  configurations, and no figure is deflated for that choice.

## Quickstart: no keys, about a minute

This runs the engine on a synthetic dataset it generates itself. It needs no
account, no API key and no network after installation. The output checks the
machinery; it says nothing about markets.

```sh
git clone https://github.com/boom90lb/prism.git && cd prism
uv sync --locked --no-dev            # core only: numpy, pandas, scipy, pyarrow, requests, python-dotenv
export PRISM_DISABLE_DOTENV=1        # never read a .env file, even if one exists
PRISM="$PWD/.venv/bin/prism"
"$PRISM" init ../prism-demo && cd ../prism-demo
"$PRISM" data synthetic --seed 7 --scenarios all --equities 12 --etfs 2
"$PRISM" conform specs/book.toml
"$PRISM" evaluate specs/book.toml --engine historical_constant_weight --smoke
"$PRISM" run --spec specs/book.toml --account paper --once --broker synthetic
```

- `prism init` writes a private project: an example signal, two specifications
  and their cost, fold and policy files. Everything the later commands write
  stays inside it.
- `prism data synthetic` writes a deterministic 22-year fixture of 12 stocks
  and 2 ETFs, including splits, delistings, halts and ticker reuse.
- `prism conform` checks the example signal against the component contract:
  no look-ahead on sampled prefixes, determinism, short-history handling and
  price-scale invariance.
- `prism evaluate --smoke` runs the specification over the whole fixture,
  reading each decision date through its own point-in-time view, and prints
  the summary. It writes no result record and no ledger row, and its Sharpe is
  an engineering diagnostic.
- `prism run --broker synthetic` runs the paper loop's own daily cycle for the
  fixture's last session against a simulated broker. It decides the target
  book and the next-open orders and records the loop's state. Nothing reaches
  a venue.

Measured on 2026-10-16 on a WSL2 machine, in a fresh clone with no `.env`:
`uv sync --locked --no-dev` took 3 s, downloading into an empty cache. The
five `prism` commands then ran in an environment with no broker keys and no
network access and took 70 s together, about 60 s of it in `evaluate`.

The [onboarding guide](docs/onboarding.md) continues from here: adding your own
signal, reading a comparison, and an offline operating review.

## Paper trading

[docs/quickstart.md](docs/quickstart.md) takes a clone to the momentum book
trading a free Alpaca **paper** account on a nightly schedule. It needs paper
API keys in a `.env` file ([security](docs/security.md)). Every command names
its mode: `--no-submit` rehearses a full session and sends nothing, and
`--submit` trades. Read [operations](docs/operations.md) and the
[free-tier profile](docs/free_tier_profile.md) for what the free data and
venue can and cannot do.

## Components you can use on their own

| Module | What it does |
|---|---|
| `prism.validation.metrics` | Probabilistic and deflated Sharpe ratios, probability of backtest overfitting, effective number of independent bets, information-coefficient ceiling. |
| `prism.validation.walk_forward` | Purged, embargoed walk-forward splits (`PurgedWalkForward`) and content-hashed fold tables (`FoldTable`). |
| `prism.validation.hurdle` | `tbill_hurdle`: the T-bill benchmark in Sharpe units, given the rate, the cash yield and the long side's share of equity. |
| `prism.validation.uncertainty` | Block-bootstrap intervals for dependent returns. They undercover; the measured shortfall is in [limitations §5](docs/limitations.md#5-the-calibration-bootstrap-undercovers-and-this-was-measured-before-the-experiment-ran). |
| `prism.execution.cost_model` | Declared cost stacks: half-spreads by liquidity bucket, impact, commission, borrow and cash income. |
| `prism.conformance` (`prism conform`) | Contract checks for any signal component: look-ahead on sampled prefixes, determinism, history handling. |
| [`formal/`](formal/README.md) | Lean 4 proofs that the ledger's rebalance and run algebra conserves capital, property-tested against the Python ledger. |

[ARCHITECTURE.md](ARCHITECTURE.md) shows what calls what. The distribution is
`prism-trading`, and the import name is `prism`. The unrelated PyPI project
`prism` also installs a top-level `prism` package, so the two cannot share an
environment. `prism-trading` is not on PyPI (checked 2026-10-16);
[depending on a release](docs/depending_on_prism.md) covers installing from a
built wheel.

## Development

Work in a separate clone that holds no `.env` and runs no paper loop. The test
suite sets `PRISM_DISABLE_DOTENV=1` itself, but a checkout that holds broker
keys is still the wrong place to develop.

```sh
git clone https://github.com/boom90lb/prism.git prism-dev && cd prism-dev
uv sync --locked --extra construction          # core, dev tools and the optional solver
uv run --extra construction pytest -q -m "not research"
```

The `research` extra installs the heavy, Linux-only stack that the archived
experiments under `research/` use. Nothing above needs it.
[Contributing](docs/contributing.md) covers how engine and strategy changes
are reviewed. [Versioning](docs/versioning.md) covers what a version
promises and how to compare decisions before upgrading.

## Documents

- [Documents index](docs/README.md): every document, grouped by purpose.
- [Limitations](docs/limitations.md): what the published results do and do not
  establish.
- [Glossary](docs/glossary.md): in-house terms mapped to standard ones.
- [Operating contract](SPEC.md): the project's rules for maintainers.
  It covers invariants, evaluation standards, how experiments are registered
  and approved, and the conditions for real money.
- [Markets](MARKETS.md): which markets a retail trader can work in at no data
  cost.
- [Research tree](docs/research_tree.md): every module under `research/`, with
  its status.

## Licence and data

The code and the project's own documentation are under the MIT licence
([LICENSE](LICENSE)), Copyright (c) 2025 Brendon Reperttang, with no warranty.
Third-party data in this repository, and series derived from it, stays under
its providers' terms, listed source by source in [NOTICE](NOTICE). That
material includes index membership lists, public factor returns, SEC filing
data, series derived from price bars, and broker records. "S&P", "S&P 500" and
"S&P MidCap 400" are trademarks of Standard & Poor's Financial Services LLC.
This project is not affiliated with or endorsed by S&P Dow Jones Indices.
Nothing in this repository is investment advice or a forecast of future
returns.
