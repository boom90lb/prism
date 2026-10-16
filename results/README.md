# Results

Recorded outputs of evaluations: claim packets, run records, return series, summary statistics and
some trial ledgers. The documents under `docs/` that read these results cite them by path.

- **Do not rewrite recorded files.** A file cited by a document in `docs/certifications/` is never
  changed, and ledgers (`*.jsonl`) are append-only (`SPEC.md` §4). Corrections to how a result is
  described go in a dated erratum (for example `docs/erratum_2026-10-14.md`), not into the record.
- **New output is ignored by default.** `.gitignore` excludes `results/`; a record is tracked only
  when it is added on purpose (`git add -f`).
- **Third-party data.** Some files hold series or statistics derived from the Kenneth R. French
  Data Library, JKP factor data (CC BY-NC 4.0), Chen and Zimmermann's open-source portfolios,
  SEC EDGAR filings, Alpaca records (some with CUSIP identifiers) and Twelve Data price bars, and
  some records embed ticker lists copied from the Wikipedia-derived membership files
  (CC BY-SA 4.0). The repository's MIT License does not cover those parts. Which files are
  affected, and on what terms, is in [`NOTICE`](../NOTICE), entries 1, 3 to 8 and 10.
