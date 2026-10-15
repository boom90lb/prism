# Reproducing the published-factor evidence from a clone — design (uncounted)

Written 2026-10-14, before the script it describes had been run against any
downloaded file. This document covers `research/scripts/reproduce_public.py`.

## 0. What this is, and what it is not

This is a **reproduction, not an evaluation.** It recomputes numbers that are
already recorded, from the files a fresh clone has plus files anyone can
download without a key, and it reports how far they differ from the record.

- It is uncounted. It writes no row to `trials/`, calls no `RunRecorder`, and
  opens no selection set.
- It writes nothing under `results/`, `trials/` or `data/public/`. Every output,
  including every downloaded file, goes into one output directory chosen by
  the caller, which must be empty and must not be under those three paths.
- It changes no recorded verdict. `vol_momentum_v1` (1 of 1) and
  `sizing_lab_v1` (8 of 8) are closed sets and stay closed (SPEC §4). The
  registered drivers (`research/scripts/vol_momentum_replication.py`,
  `research/scripts/factor_lab.py`) are not edited. Their hash checks and
  one-run budgets stay exactly as registered.
- A number computed on a vendor file that differs from the registered one is
  labelled with that file's vintage and hash. It is reported as drift beside
  the recorded number, and it never replaces the recorded number in any
  document. Whether a criterion "would pass" on a new vintage is reported as a
  description of that vintage, not as a verdict. Only a new registration can
  produce a verdict.

## 1. Why it is needed

The only configuration approved under the project's rules, volatility-scaled
12−1 momentum (`registrations/2026-09-vol-momentum.toml`), was read against
Kenneth French's daily momentum and three-factor files of the "202607 CRSP"
vintage. The registration pins their sha256 (`b039d986…` and `1916d331…`).

- French rewrites these files in place at fixed URLs, about once a month, and
  each rewrite can revise history (CRSP revisions), not only append to it.
- The registered bytes are no longer served. A query of the Internet Archive's
  capture index on 2026-10-14 found captures of the momentum file only from
  2025-09-18 and 2026-03-09, and of the three-factor file only up to
  2026-06-13. Both precede July 2026 data, so neither can be the 202607
  vintage, which the factor lab downloaded on 2026-09-24
  (`results/2026-09-factor-lab/sources.json`). Layer B records the
  `Last-Modified` header of the files served now.
- The registered driver therefore refuses to run from a clone: its hash check
  fails, and it also refuses a second run once its one ledger row exists.
- The factor lab's `priors` and `fidelity` parts would overwrite tracked
  records under `results/2026-09-factor-lab/` if re-run, and its default
  download cache (`data/public/`) would overwrite the only local copy of the
  pinned vintage.

So a third party can today check the recorded promotion only by trusting the
recorded hashes. This package gives them two things they can run instead: an
offline recheck of the arithmetic and the in-house side, and a measured
statement of how much the current public files differ from the recorded ones.

## 2. Inputs

All tracked, read only:

| Input | Role | Integrity check |
|---|---|---|
| `registrations/2026-09-vol-momentum.toml` | parameters of the rule and the four criteria | read, never edited |
| `registrations/2026-09-factor-lab.toml` | sources, series and fidelity pairs of the factor lab | read, never edited |
| `results/2026-09-vol-momentum/summary.json` | recorded criteria | compared against |
| `results/2026-09-vol-momentum/series.csv` | recorded scaled reference and scaled in-house series | read |
| `results/2026-09-factor-lab/priors.json`, `fidelity.json`, `sources.json`, `correlations/*.csv` | recorded factor-lab measurements and source hashes | compared against |
| `results/2026-09-characteristics/returns/momentum_control.csv` | unscaled in-house momentum book | sha256 must equal the registration's `returns_sha256` |
| `results/run-9f425894430542de/costs.csv` | that book's recorded gross exposure and charged cost | sha256 must equal the registration's `costs_sha256` (`349b7a7d…`); tracked since the registration commit `79ce54b` |
| `results/demotion_b1/returns.csv` | the live book's recorded returns, used by the factor lab's fidelity part | added to the index by this package as a byte-for-byte copy; sha256 must equal `90956649…`, recorded in `results/history_import/records.jsonl` line 1 |
| `results/2026-09-characteristics/returns/residual_momentum_only.csv` | residual-momentum book for the factor lab's fidelity part | sha256 recorded in the output |

A mismatch on any in-house input above stops the run with an error (SPEC N7).
These files are tracked and immutable, so a mismatch means the clone is not
the recorded state.

Downloaded, keyless, into `<out>/cache/` only: the two French files of the
registration (layer B), and with `--with-factor-lab` the seven sources of the
factor-lab registration (four French, three JKP). A failed download or an
unparseable file stops the run. A hash that differs from the pin does not
stop the run: recording that difference is the point of layer B.

## 3. Layer A: offline, always run

**A1. Arithmetic of the four criteria.** From `series.csv` and the registration
alone, recompute with the registered functions
(`publication_bar`, `fidelity`, `economics`, `no_contradiction` in
`research/scripts/vol_momentum_replication.py`):

- publication bar: full-sample and post-publication (from 2015-04-01) Sharpe,
  t-statistic, years, first and last date; pass flag; prior Sharpe and its
  basis;
- tracking: daily and weekly correlation of the scaled in-house net return
  with the scaled reference, sessions, window, pass flag;
- economics: cost drag in Sharpe units, annual charged cost, realized
  volatility, correlation × prior, the criterion-3 figure (correlation × prior
  − cost drag), the hurdle Sharpe and the pass flag. The risk-free rate is the
  recorded `economics.rf_annual`, labelled "rf from record", because the
  three-factor file that produced it cannot be downloaded any more. The long
  side over equity is 0.5 × the mean recorded scale, as in the registered run;
- no contradiction: the block-bootstrap Sharpe interval of the scaled in-house
  net return with the registered seed and settings, and the pass flag;
- the in-house summary (mean, minimum and maximum scale, annualized statistics
  scaled and unscaled) and the conjunction of the four pass flags.

Each value is compared with `summary.json`. Floats must agree within 1e-9
absolute; integers, strings and booleans must be equal. This checks the
arithmetic and the published series file, not the French bytes.

**A2. The in-house side, rebuilt from its inputs.** Rebuild the scaled in-house
series from the hash-checked `momentum_control.csv` and `costs.csv` with the
registered overlay (`scaled_in_house`) and compare scale, net return, charged
cost and unscaled return session by session with `series.csv`. Expected:
identical within 1e-9. This checks the in-house half of the evidence end to
end from tracked inputs.

**A3. Consistency of the recorded reference with the registered rule.** The
recorded scaled reference stores the scale and the scaled return. Their ratio
is the unscaled French return. Re-applying the rule to that implied series
must give back the recorded scale for every decision whose 126-session
lookback lies wholly inside the recorded window (the recorded window starts
at the warm-up's end, so the first six decisions in it are not checkable).
This shows the recorded reference was produced by the
registered rule, refresh grid and cap. It cannot show that the unscaled
returns were French's, because those bytes are gone.

## 4. Layer B: network, keyless, skipped with `--offline`

1. Download the momentum and three-factor files into `<out>/cache/` and record,
   per file: URL, bytes, sha256, the `Last-Modified` header, the fetch time,
   the "CRSP database" vintage line, and the first and last data dates. Compare
   each sha256 with the registration's pin and with
   `results/2026-09-factor-lab/sources.json`.
2. **Raw drift.** Compare the new unscaled momentum return with the implied
   recorded one (A3) on common dates: sessions compared, sessions that differ
   by more than 1e-9, largest absolute difference, first and last date that
   differs, and sessions added or removed.
3. **Scaled drift.** Recompute the scaled reference with the registered rule
   and warm-up, and compare it with the recorded scaled reference in the same
   way.
4. **Criteria on the new vintage, two windows.**
   - B1: the new file cut at the recorded reference's last date (2026-07-31).
     This isolates history revisions.
   - B2: the new file through its last date. This adds the extension.

   Each is computed twice: with the recorded rf, and with rf recomputed from
   the new file's RF column over the registered 252 sessions ending at the
   window's last date. The in-house side is the A2 series in every case; it
   ends 2026-06-12, so the tracking window is the same in B1 and B2 and only
   history revisions can move the tracking correlation.

## 5. Factor-lab measurements, with `--with-factor-lab` (network)

Download the seven factor-lab sources, then call the registered
`build_public_data`, `run_priors` and `run_fidelity` with the output directory
set inside `<out>/`, twice:

- recorded window: each series cut at the last date recorded for it in
  `priors.json`;
- current: every series through its last date.

For every series, compare full-sample Sharpe, t-statistic, sessions, last date,
post-publication Sharpe, prior Sharpe, prior basis and both pass flags with
`priors.json`. For every fidelity pair, compare sessions, daily and weekly
correlation, implied tracking Sharpe and the floor flag with `fidelity.json`.
For each correlation matrix, report the largest absolute difference from the
tracked file. Hash matches are reported per source; the three JKP files are
expected to match while JKP has not published a new release.

The factor lab's counted sizing set is not recomputed. Its per-session
returns are tracked (`results/2026-09-factor-lab/sizing/returns.parquet`),
but recomputing it is not part of this package.

## 6. Outputs

All under `<out>/`:

- `manifest.json`: command, code commit and whether the tree was clean, the
  layers run, every input path with its sha256, and every download with the
  fields of §4.1 and its match flags;
- `vol_momentum.json`: per criterion, the recorded value, A1, and B1 and B2
  with both rf choices, plus A2, A3 and the drift statistics;
- `priors_diff.json` (with `--with-factor-lab`): §5's comparisons, plus the
  factor lab's own outputs under `<out>/factor_lab/`;
- `report.md`: a short readable summary. Its first line is exactly
  "Reproduction, not an evaluation: uncounted, no ledger row, no recorded
  verdict changes."

The command exits 0 when layer A matches within tolerance and every requested
layer completed. It exits non-zero when layer A does not match. Drift in layer
B is reported, not treated as a failure.

## 7. Refusals

- An output directory that resolves (following symlinks) under the
  repository's `results/`, `trials/` or `data/public/`.
- A non-empty output directory.
- `--offline` together with `--with-factor-lab`.
- An in-house input whose sha256 differs from its record (§2).
- A failed download, an empty payload, or a file that does not parse.

## 8. Limits, stated before the first run

- Layer A cannot reproduce the registered reference series from its source,
  because the registered French bytes are not retrievable. A3 checks that it
  follows the registered rule; it does not check its origin.
- Layer B measures a different vintage. Its numbers describe that vintage.
  The registered verdict and the forward stop rule
  (`docs/vol_momentum_design.md`, registration `[kill]`) are untouched by it.
- French revises history at each release, so layer B's numbers will change
  from month to month. A drift record is useful only with its vintage line and
  hashes, which every output carries.
- Archiving the vendor files themselves would make the registered evaluation
  reproducible byte for byte. That is an external publication and depends on
  the data licences, which is the owner's decision. Until then, the hashes in
  the record, layer A, and layer B's drift report are what a third party can
  check.
- If the owner later wants a drift record kept in the repository, it goes into
  new paths (for example `results/reproductions/<date>-french-<vintage>/` with
  a short read record under `docs/`). Nothing in this package writes there.

## 9. Acceptance

- This document is committed before the first run that downloads anything.
- `tests/test_reproduce_public.py` passes offline: A1 reproduces
  `publication_bar.prior_sharpe`, `fidelity.daily_corr`,
  `economics.cost_drag_sharpe`, `economics.figure_f3` and
  `economics.hurdle_sharpe` within 1e-9; the output-directory refusals hold;
  layer B's bookkeeping works against a synthetic French-format zip with the
  download replaced, and nothing is written outside the output directory.
- `python -m research.scripts.reproduce_public --out <empty dir> --offline`
  exits 0 with no network.
- From a fresh clone, `python -m research.scripts.reproduce_public --out <empty
  dir> --with-factor-lab` exits 0, the clone's `git status --porcelain` stays
  empty, and no `data/public/` directory appears.
- No file under `results/2026-09-*` or `trials/` changes.
