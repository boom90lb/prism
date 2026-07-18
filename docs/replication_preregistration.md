# Replication pre-registration — frozen-B1 reads outside the selection cell (DRAFTED, GATED on the A2 purchase)

**Status: drafted 2026-07-17 under the owner's scope-expansion directive; not
ratified.** Ratification is the owner's dedicated commit/push (AGENTS.md §2;
the amendment convention of `docs/handoff.md` §7). This document spends
nothing until (i) it is ratified, (ii) a qualifying survivorship-complete
panel is delivered — the purchase was reversed 2026-07-17 and delivery is
unscheduled, see §8 — and (iii) the §3 data gates pass. **The design below is frozen before any purchased return
data is inspected — that ordering is the entire evidential content of this
instrument.** A design adjusted after seeing the panel is a search, not a
replication.

## 0. Why this exists, and the honest prior

B1 was selected as the best of 17 searched trials on one fixed cell: the PIT
S&P 500 universe, 2020-01 → 2026-06 (`docs/momentum_design.md` §0). Every
statistic computed on that cell is selection-contaminated, and the ratified
cure — the M6 extension year — is the *minimum credible quarantine*, not an
evidence engine: one added year moves a ~0.47-Sharpe read by roughly half a
standard error, and the pre-registration's own expectation section says the
central case "fails the deflated bar on one added year and needs two"
(`docs/momentum_design.md` §4). Time is the weakest evidence axis available
to this program.

Data outside the selection cell is a different axis. Selection contamination
is *local to the searched cell* — the specific universe and era the 17 trials
saw — while the premium claim is cross-sectional and temporal. The two
hypotheses the M6 judgment must weigh therefore make divergent predictions
off-cell:

- **H_premium** (B1 rides the documented momentum premium): the frozen
  configuration shows a positive gross momentum spread on universes and eras
  the selection never touched.
- **H_mined** (B1 is the max of 17 draws from noise): off-cell reads center
  on zero.

Against *that specific alternative* the likelihood ratio of an off-cell read
is far stronger than the ~1.34 the extension year buys against a generic
zero-edge null (`docs/program_review_2026-07.md` §1.4). Additionally, the
purchased panel's pre-2020 depth contains what the selection cell
structurally cannot: a momentum crash. Finding #2 of the program review
records that 2020–2026 lacks the strategy's known failure mode and that the
extension year is equally unlikely to contain it
(`docs/program_review_2026-07.md` §1.2). A pre-2020 large-cap cell measures
the frozen configuration's 2008–09 behavior directly — turning the crash
paragraph of `docs/momentum_design.md` §4 from deferred prose into recorded
numbers available to the sizing decision the GO branch will consume.

This instrument is the A2 amendment's standing first candidate made
operational: the survivorship-complete US equity history
(`docs/amendments_2026-07.md` §A2) exists to be read, and this document pins
how it is read before it exists on disk.

## 1. The grid

Four cells; universe × era. One is a data gate, three are counted trials.

| cell | universe | era | standing |
|------|----------|-----|----------|
| C0 | PIT large-cap (S&P 500) | 2020-01 → 2026-06 | **uncounted data gate** — cross-vendor reproduction of the certified cell (§3.iii) |
| C1 | PIT mid-cap (§2 rule) | 2020-01 → delivery month-end | **counted trial 1** — the cross-sectional replication |
| C2 | PIT large-cap | panel start (floor 1995-01) → 2019-12 | **counted trial 2** — the temporal replication, containing 2008–09 |
| C3 | PIT mid-cap (§2 rule) | panel start (floor 1995-01) → 2019-12 | **counted trial 3** — the joint cell |

Era endpoints are pinned as follows, so no era can be chosen after seeing
results: C1's end is the last complete calendar month before the purchased
panel's delivery date, recorded at gate time; C2/C3 start as deep as the
panel's survivorship-complete coverage supports, floor 1995-01-01, actual
start recorded at gate time, and run as **single unbroken cells** — no
sub-era reads are registered, so none may be reported.

## 2. The frozen configuration — transplanted, not re-tuned

Every cell runs **B1's ratified configuration exactly** (`docs/momentum_design.md`
§0): 12−1 cross-sectional momentum (252-bar lookback, 21-bar skip),
top/bottom decile equal-weight long−short, `decision_every=21`, closed-form
band, `SPREAD_BUCKET_SCHEDULE_V1` bucket spreads, 5% participation cap. Zero
free parameters. The bucket schedule extends to mid-caps mechanically — it
keys one-way bps to dollar-volume floors (500M/100M/25M/catch-all;
`research/arbitrage/residual_walk_forward.py:69`, duplicated by value at
`src/prism/execution/spread.py:37`) — so mid-cap names price into the wider
conservative-upper buckets with no improvised cost model. Any deviation from
the transplanted configuration is a new discovery event and out of this
document's scope.

The only definitions that are new, pinned here before data:

- **Mid-cap universe rule (decision tree, branch selected by panel
  *contents*, a data-availability fact readable without touching outcome
  data):** (a) if the panel carries point-in-time S&P MidCap 400 membership,
  the universe is PIT S&P 400 — the expected branch: owner verification of
  the vendor's published data-content tables (2026-07-17) confirms daily
  S&P MidCap 400 constituents back to the June-1991 inception,
  quarterly-reconstituted with immediate delisting replacement;
  (b) otherwise, PIT capitalization band — ranks
  501–1000 by point-in-time market cap from the panel's shares × price —
  reconstituted on the decision grid, under the same eligibility hygiene
  (price floor, ADV floor) as the production builder
  (`prism.scripts.build_sp500_universe` parameters, reused unchanged). The
  branch taken is recorded at gate time.
- **Runner semantics:** the identical code path that produced the certified
  B1 artifacts, with two additive pieces of uncounted mechanics — a panel
  ingestion adapter and the universe producer above — and its **own ledger
  file and namespace** (`results/momentum_replication_trials.jsonl`,
  `strategy: momentum_replication_v1`). The standing defect that
  `research/scripts/stat_arb_residual_wfo.py` appends a trials-ledger row
  unconditionally makes ledger isolation a pinned requirement, not a
  preference.
- **Terminal-bar rule (P-TB).** No daily-bar dataset carries post-delisting
  recovery values — the purchased panel included. A position whose security
  ceases trading mid-hold exits at the security's final *unpadded* quoted
  close, on that bar, flagged `terminal_exit` in the run ledger. An index
  removal while the series continues trading is **not** a terminal exit —
  the vendor carries OTC-relegated names as *currently listed* (its
  "delisted" means untradeable on any venue it tracks; owner verification,
  2026-07-17) — and such a position exits by the normal universe-exit rule
  at the next decision bar, at market prices. All registered cell
  statistics use terminal exits at final quoted price with **no
  imputation**; the delisting-return sensitivity lives in §5 as an
  uncounted note, computable only via gate 5's reason classification.

## 3. Data gates (uncounted; all precede any counted run)

A failed gate blocks every counted cell: the response is purchase
remediation, and nothing is spent.

1. **Integrity sweep.** `research/scripts/data_integrity_sweep.py`
   generalized to the purchased panel: zero unexplained wrong-instrument
   suspects, by the same evidence classes as the 2026-07 collision sweep
   (`docs/data_integrity_diagnostic.md`).
2. **Delisting canon.** The panel must contain terminal histories for the
   known-answer cases the free vendor failed: ECHO pins at its 2021-09
   buyout terms; the FB→META and PCLN→BKNG renames resolve as renames, not
   new listings; SBNY and WLTW terminate correctly. The 2026-07 quarantine
   class is the test suite.
3. **C0 cross-vendor reproduction.** Frozen B1 on the purchased panel's
   large-cap 2020-01 → 2026-06 cell against the certified/remediated lineage
   (`docs/data_integrity_diagnostic.md` §7): mean per-refresh active share
   ≤ 0.05 and |Δ net annualized Sharpe| ≤ 0.10 versus the remediated
   reproduction. This is a *data* comparison on an already-spent cell — it
   contributes no new strategy evidence and is not a trial.
4. **Provenance and license** of the panel enter the SPEC §5 coverage-ledger
   discipline, per A2(iii).
5. **Delisting-event reconciliation (G-DR).** Before any cell's spread is
   read, the panel's terminal events must reconcile against an independent
   canon assembled by `prism-observatory` from sources sharing no vendor
   with the panel: (i) EDGAR Forms 25 / 25-NSE / 15-\* (filed and effective
   dates; Form 25 effectiveness is filing + 10 calendar days per Rule
   12d2-2) and (ii) Alpha Vantage `LISTING_STATUS` point-in-time snapshots
   (2010→present). Every panel security whose series terminates inside a
   cell's window must match a canon event with |vendor last-quoted date −
   canon effective date| ≤ **5 trading days** — a pinned constant, not a
   knob, absorbing suspension-before-delisting gaps and filed-vs-effective
   lag. **Reason classification sources from the canon only**: the vendor
   publishes `LastQuotedDate`/`SecondLastQuotedDate` and no delisting-reason
   field anywhere (owner inspection of the published data-content tables,
   2026-07-17, settling the purchase memo's open item negatively), so
   merger/acquisition vs deficiency/bankruptcy/exchange-initiated comes
   from canon form types and filing context, recorded per event, feeding
   only the §5 sensitivity note — never a registered statistic. The item-2
   known-answer set must reconcile exactly, plus a per-cell 50-name sample
   of terminal events drawn with **pinned seed 20260717**; every
   unreconciled event is a named residual, and the cell is blocked if
   unreconciled events exceed **2%** of its terminal events. Universe exit
   ≠ terminal event (the §2 P-TB distinction): only series-terminal events
   reconcile here. The canon is evidence, not calibration — it moves no
   price, edits no series, and fires no recompute; a reconciliation failure
   quarantines the affected names under the `QUARANTINE_TABLE` discipline
   with the evidence row cited, and the gate artifact records the
   observatory commit hash it was read from.

## 4. Trial accounting

Own selection set, namespace `momentum_replication_v1`; budget **exactly 3
counted trials** (C1, C2, C3), never refilled; degenerate or NaN outcomes
count. Prior counted programs at drafting, recorded per amended SPEC §10:
two (residual reversion, closed at 17, cert 001; momentum, ≤ 8, in flight).
There is no configuration search anywhere in this family: three runs of one
frozen configuration on three disjoint cells.

## 5. Adjudication, pre-committed

Per-cell reads, all reported: **gross** annualized Sharpe of the spread (the
premium-existence read), **net** under the frozen stack (the economics read,
carrying the pre-stated caveat that sub-large-cap buckets are
conservative-upper and uncalibrated — a net haircut on C1/C3 is expected and
uninformative about H_premium vs H_mined), and for C2/C3 the crash block:
maximum drawdown, worst 21-bar return, and monthly-return skew of the frozen
book through 2008–09 (the measurement `docs/momentum_design.md` §4 defers to
sizing).

**Uncounted delisting-return sensitivity note** (companion to the §2
terminal-bar rule; reported beside — never inside — each cell's registered
read): recompute the spread appending a Shumway-style delisting return
after the final bar for **performance-classified** terminal events only
(gate-5 canon reason ∈ {deficiency, bankruptcy, exchange-initiated}):
−30% for NYSE/NYSE-American listings, −55% for Nasdaq (Shumway 1997;
Shumway & Warther 1999). Merger/acquisition terminal events receive no
imputation — the final quoted price already embeds the deal. Direction,
recorded ex ante: performance delistings concentrate in the loser decile,
which the 12−1 book is short, so the registered no-imputation number
truncates short-leg profit and is **conservative for the spread**; the
residual non-conservative exposure — a long-leg name collapsing mid-hold —
has its realized frequency counted and reported in the same note, so the
rarity claim is measured, not assumed. The registered number is the number.

Interpretation, pinned now so it cannot be negotiated later:

- **Supports H_premium:** gross > 0 in all three counted cells.
- **Supports H_mined:** pooled read ≤ 0 (equal-weight mean of the three
  cells' gross Sharpes).
- **Mixed:** recorded as mixed; no forced call.

Interpretive limits, equally pinned: this instrument's output **cannot move
the M6 adjudication in either direction** — `docs/momentum_design.md` §3 is
ratified and stands as written. What it can do: (a) enter the
divergence-ledger context available to the GO/WAIT deliberation; (b) supply
the crash statistics to the GO-branch sizing decision; (c) seed — not spend —
a future mid-cap expansion program's prior, which would carry its own
pre-registration and budget. A strong replication does not promote B1, and a
failed replication does not kill it; it re-weights the judgment M6 was always
going to be.

## 6. Sequencing under one-counted-program-at-a-time

Amended SPEC §10 pins "at most one counted program runs at a time," and the
momentum program holds that slot until its §3 verdict. The rule's two objects
are across-family selection pressure and the ledger discipline of an
operator of one (`docs/handoff.md` §8, the crypto-lane precedent). This
family is a budget-3, zero-degrees-of-freedom instrument whose output is
barred from every promotion read — there is nothing to select and nothing to
iterate — so the drafted position is that running it at data delivery is
within the rule's purpose, and this document's ratification commit is the
place the owner grants that sequencing exception explicitly. The
alternative — queueing the counted cells behind the momentum verdict —
forfeits the timeliness that motivates the purchase; it remains the owner's
call, and neither branch moves any other rule.

## 7. What this document does not do

No momentum trial value or M6 criterion moves. No live-loop universe file
changes. No re-tuning, no new signal, no sub-era or sub-universe reads beyond
the four cells. The closed residual selection set stays closed. The budget
does not refill. The purchased panel feeds the *live* spine only under its
own future amendment; this document governs research reads only.

## 8. Amendment 2026-07-17: the purchase reversed, the design unchanged

Owner decision, same day as drafting: the Norgate purchase is reversed in
favor of slowly accumulating survivorship-bias-free data at enterprise
quality in-house — prospective capture via `prism-observatory` as the
backbone, a dead-ticker price layer as future design work
(`docs/data_purchase_evaluation.md` §6 records the outcome against the
evaluation). Consequences, stated now so they cannot be discovered later:

- **Every §3 gate is already source-agnostic** — the gates define panel
  *quality*, not vendor. Nothing in §§1–7 moves. The instrument's trigger
  changes from "purchase delivered" to "a panel covering a cell's
  universe × era exists and passes §3," evaluated **per cell** — C1 may
  activate years before C2/C3.
- **Known source map at amendment time:** C1 membership is coverable by
  the observatory's IJH point-in-time holdings (~2010 onward); C1
  delisted-name *prices* are the accumulation target (recent deaths are
  the recoverable class). C2/C3 have **no known free source for pre-2020
  delisted prices**: those cells stay registered and unscheduled, and the
  crash-era read (C2) is deferred indefinitely. Recording this is the
  honest price of the reversal, paid knowingly.
- **An accumulated multi-source panel faces the same discipline as any
  vendor panel**: the §3 integrity sweep treats splice seams between
  sources as first-class wrong-instrument suspects, and adjustment-basis
  agreement across sources is a gate input, not an assumption.
- **The frozen design ages well under this.** A configuration pinned in
  2026-07 and first evaluated on a panel assembled years later is a
  *stronger* quarantine than the purchased path offered — provided no
  interim backtest touches any cell. The §4 accounting is unchanged; the
  budget stays exactly 3.
- **A2 is untouched.** The amendment permits purchases; it does not
  mandate them. The budget goes unspent and remains available if the
  decision is ever revisited.
