# Free-tier profile — what free data and keys reproduce

> **Status: reference (drafted 2026-07-18, updated 2026-10-15).** The project
> started on free data only. The current contract allows buying data while
> total spend stays at or below about $1,000 a year (`SPEC.md` section 5,
> item 3), and the one purchase evaluated so far was reversed in favour of
> accumulating data going forward (docs/data_purchase_evaluation.md §6). This
> file makes "usable for free" a specified claim: which capabilities a clean
> checkout plus free keys reproduces, which it cannot, and why. Terms are
> explained in the [glossary](glossary.md).

## 1. The free stack

| Source | Access | What it feeds |
|---|---|---|
| Alpaca paper account | free key; IEX feed; paper venue; the corporate-actions API works on the paper key | the whole paper loop: bars, orders, fills, the dividend estimate |
| Twelve Data | free key: 8 requests a minute, 800 a day (`prism/io/rate_limit.py`) | the research drivers' bar and dividend caches |
| Kenneth French Data Library | no key; daily factor files as zip downloads (checked 2026-10-15) | the published-factor evaluations: momentum, the three- and five-factor files, the risk-free rate |
| JKP factor data | no key; files on a public S3 bucket | the published-factor evaluations (licence terms in `NOTICE`) |
| FRED | the JSON API needs a free key; the `fredgraph.csv` download needs none (checked 2026-10-15) | the regime layer's macro series and the T-bill rate |
| SEC EDGAR | no key; a declared User-Agent header | point-in-time fundamentals and earnings dates (`prism.io.edgar`) |
| Wikipedia | no key | the S&P 500 constituent and change tables `prism-build-universe` reconstructs membership from |
| SEC EDGAR, Nasdaq directories, SSGA daily holdings | free, no key | membership captures (S&P 500/400 via SPY/MDY holdings) run from the separate `prism-observatory` repository |
| Alpha Vantage | free key, 25 requests a day | that repository's monthly capture of active and delisted listings |

**Stooq is not usable.** Older documents list Stooq as a free daily source. It
now answers a scripted request with a JavaScript proof-of-work page instead of
CSV (checked 2026-10-15: `https://stooq.com/q/d/l/?s=aapl.us&i=d` returned a
796-byte HTML page whose script asks the browser to verify itself). Prism does
not use it, and getting around a deliberate bot check is not an option.

## 2. Reproducible for free today

- **The paper loop, end to end** ([quickstart](quickstart.md)): Alpaca keys
  only. Bars come from the broker's own IEX feed; no Twelve Data key is needed.
- **The preflight, replay and diagnostics**: `prism-doctor`, the replay driver
  over local caches, and the paper loop's own ledgers.
- **The point-in-time membership data**: the universe files ship in-tree
  (`data/universe/`).
- **The published-factor downloads**: `python -m research.scripts.factor_lab
  --part fetch` downloads the Kenneth French and JKP files with no key. Kenneth
  French replaces files at fixed URLs with each new vintage, so the exact files
  a recorded evaluation pinned may no longer be downloadable (erratum E6).
- **The research drivers' caches, slowly**: about 570 member histories at 800
  requests a day rebuild over a few days of scripted fetching; the loader's
  token bucket and backoff on HTTP 429 keep this within the tier.

## 3. Not reproducible for free — recorded, not hidden

- **Pre-2020 prices of delisted mid- and large-cap stocks.** No known free
  source. The replication pre-registration records this as the known cost of
  the free-data stance (docs/replication_preregistration.md §8): its crash-era
  samples, cells C2 and C3, stay unscheduled until a source exists or enough
  data has been accumulated.
- **Delisted-ticker bars generally.** The published backtests carry no price
  rows for delisted tickers. The fix going forward is accumulation, not
  purchase (docs/dead_ticker_price_design.md). Any free stock-level backtest is
  therefore survivorship-biased, which is why the published-factor evaluations
  lean on long public return series.
- **Bond-ETF bars on the Twelve Data free tier.** The vendor returns no bars
  for bond ETFs at this tier (measured 2026-07-18 while resolving the trend
  universe, docs/trend_design.md §1); the trend tests' credit buckets read the
  Alpaca feed instead.
- **Real-time consolidated data, sub-daily history and redistribution
  rights.** None is free. The IEX feed's median share of about 5% of volume is
  an accepted constraint on execution (MARKETS.md). `SPEC.md` section 8 admits
  intraday signals once a fill model for that bar frequency is defined.

## 4. Why it matters

The data budget is part of the evidence, not an inconvenience to route around.
A result reproducible only with paid data is a different claim from one
reproducible for free, and every net result records its data source, its
coverage and its data convention so a reader can tell which is being made
(`SPEC.md` section 3). When a gap matters, such as pre-2020 delisted prices,
the program records it as a named limitation with a trigger for revisiting,
rather than quietly narrowing the question (docs/replication_preregistration.md
§8 is the worked example).
