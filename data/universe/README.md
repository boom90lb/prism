# Universe files

Index membership lists and the reviews that map tickers to SEC filers.

- **S&P 500 and S&P 400 membership** (`sp*_membership_*.parquet`, `sp*_pit_*.txt`,
  `sp*_coverage_*.json`, `sp400_exits_*.json`, `sp500_sources_*.json`). Built from Wikipedia's
  S&P 500, 400 and 600 tables with the parsers in `src/prism/io/universe_sp500.py` (earlier
  versions of them for the older builds): the S&P 500 files by
  `src/prism/scripts/build_sp500_universe.py` or its predecessors, the S&P 400 files by
  `src/prism/scripts/build_sp400_resolved.py`, and the S&P 400-only and combined S&P 900 files by
  `research/scripts/build_breadth_dataset.py`. Only the 2026-09-25 S&P 500 build pins the article
  revisions it read, in `sp500_sources_2026-09-25.json`; the earlier S&P 500 builds and the
  S&P 400 and S&P 900 files do not. Membership is keyed by ticker, the
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
