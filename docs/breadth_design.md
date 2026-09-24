# Breadth — the B1 book on a wider point-in-time universe (`breadth_v1`)

**Status: registered 2026-09-23.** The registration is the commit that adds
this file and `registrations/2026-09-breadth.toml` (SPEC §4). It answers SPEC
§5.2: "N_eff ≈ 53 on ~96 names held is the binding constraint. The lever is a
point-in-time universe beyond the S&P 500 with the survivorship leak counted,
not hidden." Every universe rule below was fixed before any mid-cap return
was read; the bar pull read prices for existence and last date only.

## 1. The question

The certified B1 book realizes 61% of its `IC·√N_eff` ceiling and its IC is
not distinguishable from zero at the one-sided 95% bound
(`results/demotion_b1/breadth_diagnostic.json`). The fundamental law says the
ceiling scales with `√N_eff`. Does the same frozen book, run on the S&P 500
plus the S&P MidCap 400, hold more independent bets, and does its net Sharpe
move with them?

## 2. Universe construction, pinned

- **S&P 500.** `data/universe/sp500_membership_2026-06-16.parquet`, unchanged:
  the certified book's own membership and its 574 price-resolved names.
- **S&P 400.** Reconstructed from Wikipedia's "List of S&P 400 companies"
  (fetched 2026-09-23; current constituents plus 544 dated changes back to
  2012) by the S&P 500 builder's own method
  (`prism.io.universe_sp500.reconstruct_membership`). That changes table is
  selected, not complete: a forward sweep left 53 tickers open to the as-of
  date that are not current constituents. `prism.scripts.build_sp400_resolved`
  closes each such interval on the day before the earliest dated evidence
  that the ticker had left: the start of its next S&P 500 interval
  (promotion), its addition in the S&P 600 page's changes table (demotion),
  or its vendor series ending at least 14 days before the pull's end
  (acquisition or delisting). An interval with no evidence is dropped. Every
  closed and dropped interval is in `data/universe/sp400_exits_2026-09-23.json`.
- **Exclusivity.** The S&P 500, 400 and 600 are mutually exclusive. Every day
  a ticker holds in the S&P 500 is removed from its S&P 400 membership
  (`prism.io.universe_sp500.subtract_membership`). This corrects names the
  reconstruction treats as mid-cap members from the history floor while they
  were in the 500.
- **The union, `sp900`.** Per-ticker union of the S&P 500 and the exclusive
  S&P 400 memberships, touching intervals merged
  (`prism.io.universe_sp500.union_membership`).
- **Bars.** Twelve Data daily bars 2020-01-01 to 2026-06-16, the certified
  window. S&P 500 names read their existing caches (fetched 2026-06). Mid-cap
  names were fetched on 2026-09-23, so their split adjustment is three months
  newer; a split between 2026-06-16 and 2026-09-23 rescales a name's whole
  history by one factor, which moves the absolute $5 screen for that name and
  nothing else.
- **Coverage.** Mid-cap ever-members with no retrievable bars are the
  measured survivorship leak, enumerated in
  `data/universe/sp400_coverage_2026-09-23.json` beside the S&P 500's 59.

## 3. Arms

One frozen dataset, `data/frozen/breadth_v1`, holding all three indices, so
every arm reads identical bars. The book is B1 exactly
(`examples/breadth/control.toml`); the arms differ only in
`[[arms]].universe_index`.

| arm | universe | role |
|---|---|---|
| `sp500_control` | S&P 500 | the certified universe; must reproduce `characteristics_v1`'s `momentum_control` |
| `sp400_only` | exclusive S&P 400 | the mid-cap cross-section alone |
| `sp900_union` | S&P 500 ∪ S&P 400 | the breadth lever |

Budget 3. Canonical engine, $100,000, the experiment profile and cost stack
of the construction registration, fold table `folds/demotion_b1.toml`.

## 4. Reads, kill rule, closure

- **Breadth (N6).** For each arm, `N_eff` as the participation ratio of the
  covariance of position-weighted daily return contributions, the retired
  diagnostic's definition (`git show c08e3d0^:research/scripts/breadth_diagnostic.py`),
  reported with names held and the top eigenvalue shares.
- **Kill.** An arm whose paired studentized 95% interval for (arm − control)
  net periodic Sharpe lies entirely below zero is dead (declared bootstrap:
  daily, circular blocks of 42, 2,000 resamples, seed 20260923).
- **Closure.** The set closes when the three arms have run.

## 5. Promotion and what this is not

No arm promotes: this is a same-sample read. A surviving union arm is a
candidate for a prospective registration on post-2026-06-16 data plus paper.

This is **not** cell C1 of `docs/replication_preregistration.md`. C1 is gated
on a survivorship-complete panel, and this panel is not one: it carries the
free vendor's delisted-name gap, counted. Nothing here may be cited for C1.

## 6. What the set cannot establish

Modeled fills; a free-vendor panel whose delisted names are missing in
proportion to the coverage ledgers; a reconstructed mid-cap membership whose
exits rest on the evidence rules above; 2020-01 to 2026-06, one market; no
dividends; no momentum crash in sample.
