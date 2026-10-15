# Universe files

Index membership lists and the reviews that map tickers to SEC filers.

- **S&P 500 and S&P 400 membership** (`sp*_membership_*.parquet`, `sp*_pit_*.txt`,
  `sp*_coverage_*.json`, `sp400_exits_*.json`, `sp500_sources_*.json`). Rebuilt from pinned
  Wikipedia revisions by `src/prism/io/universe_sp500.py`. Membership is keyed by ticker, the
  `*_pit_*` lists are unions of every past member, and the coverage files list which members had
  price data. These are approximations built from public text, not index data from S&P Dow Jones
  Indices.
- **`sp500_current.txt`**: current members that have local price data. The paper-trading loop
  trades this list, so a change to it changes the live book.
- **`sp500-2026-06-11.txt`**: an older current-constituents list from the
  `datasets/s-and-p-500-companies` project. It is survivorship-biased, and no current code reads it.
- **`cik_*`, `edgar_coverage_*.json`**: ticker-to-CIK matches and their manual reviews, built from
  SEC EDGAR.
- **`2026-05-30.txt`, `sp500_density_trials.jsonl`**: a hand-written symbol list and a trial
  ledger, both the project's own.

**Licences.** The Wikipedia-derived files are under CC BY-SA 4.0, not the repository's MIT
License. The terms, sources and required attribution for every file here are in
[`NOTICE`](../../NOTICE), entries 1, 2 and 8, with the S&P trademark notice at its end.
