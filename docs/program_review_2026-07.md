# Program review 2026-07: bird's-eye findings, verified and corrected

**Status: review record, recorded 2026-07-13.** Not a pre-registration; no
ratified statistic moves here. The seven findings below were produced as an
expert bird's-eye review, then adversarially re-verified against primary
artifacts (seven independent verification passes over 22 factual claims).
Claims that failed verification are corrected inline and listed in §2; two
findings changed character under verification and were executed the same day
(§3, §4).

## 1. The seven findings, corrected form

1. **The apparatus is world-class; the prize it guards is ~40 bps of Sharpe
   over cash.** B1 records net Sharpe 0.465 at 11.5% vol ≈ 480 bps/yr
   (`results/demotion_b1/summary.json`) against a ~4% cash rate, and the
   breadth diagnostic's lower-CI read leaves viability unpinned (IC 0.030 ±
   0.024, viability margin −0.024 at the lower-95 IC,
   `results/demotion_b1/breadth_diagnostic.json`). Mega-cap monthly momentum
   is the most arbitraged, edge-poor premium in equities; the $0-data,
   retail-cost constitution made it the only survivor. Cost-bound before
   signal-bound is sound reasoning whose terminal consequence should be
   stated plainly: as constituted, the pipeline can only certify slow,
   low-Sharpe premia. The §4 measurement below trims the prize further:
   under total-return accounting B1's Sharpe is ≈ 0.434.

2. **The sample structurally cannot contain the strategy's known failure
   mode.** 2020–2026 has no momentum crash, and the promotion criteria carry
   no skew or convexity term. Both facts live in the ratified
   pre-registration — `docs/momentum_design.md` §3 (conjuncts) and §4 (the
   crash paragraph, deferring crash exposure to sizing at deployment) — not
   in SPEC §3/§4 as the original review said; SPEC's own §10 tier gates also
   carry no tail term, so the finding survives either attribution. The 2026–27
   extension year is equally unlikely to contain the event. The real
   mitigation is portfolio-level: the second sleeve should be chosen for
   crash-convexity, which argues for time-series ETF trend over any
   standalone-Sharpe comparison.

3. **The certified equity process is not the live equity process.** The
   research backtest force-flattens the book on the last two bars of every
   fold (`research/arbitrage/walk_forward.py:264-271`; 20 folds of 63 bars,
   one of 48), charging full costs; the live loop never flattens. The
   "~63% of B1 turnover" attribution is a *derivation* (≈2.0 of 3.15 gross
   per fold, consistent with `folds.json` aggregates), not a recorded
   statistic, and it overstates the incremental cost because part of each
   fold-start re-buy duplicates refresh churn. A carry-mode counterfactual is
   impossible in-tree (`walk_forward.py:92-93` raises), so quantifying what
   flattening does to the drawdown/vol/skew statistics the eventual sizing
   decision will consume requires new code. Open; worth doing before M6.

4. **The promotion conjunct that waits on a year of paper data is nearly
   uninformative.** Rolling PSR > 0.5 at ~1y on a true-0.465 strategy passes
   with p ≈ 0.67 versus 0.5 under zero edge — likelihood ratio ≈ 1.34 and a
   ~33% chance of blocking a genuinely working strategy. It is pinned, so it
   stands; plan explicitly for the WAIT branch, and treat the concordance
   ledger (which resolves mechanical fidelity at any n) as the real content
   of "not in contradiction."

5. **What paper fills can and cannot calibrate.** Alpaca paper simulates the
   opening auction and prints only ~20–25% of OPG orders — the fills are
   simulator prints, not the official cross. Ledgered arrival slippage on the
   22 fills to date spans **−233 to +884 bps** (three price improvements) —
   overnight-gap noise, not spread. No sweep fill exists yet (the 2026-07-13
   sweep submitted zero residuals); the first genuine spread-bearing data
   arrives with the first real residual sweep. Fills from the paper venue
   validate the pipeline, not the market; real cost calibration wants a small
   real-money instrument, which is an owner decision requiring an amendment.

6. **The budget arithmetic starves the enterprise.** 17 counted trials spent
   plus 8 reserved for momentum = 25 of SPEC §10's ~30, leaving ~5 for any
   future program (derived from `docs/demotion_design.md` §"Budget" and
   `docs/aim_portfolio_preregistration.md`). The flat cap conflates
   within-selection-set multiplicity (DSR already deflates it) with
   across-family multiplicity (per-family budgets + OOS confirmation are the
   honest control). The $0-data clause (SPEC.md §1 mandate, §4 stack)
   forecloses a survivorship-complete dataset that is now argued for by three
   independent facts: the mid-cap expansion blocker, the Twelve Data
   `/dividends` 403 that blocked wedge measurement on the spine vendor, and
   the §3 collision class. Both clauses deserve amendments argued on their
   merits.

7. **Throughput and the ops SPOF.** Seventeen sessions, superb
   infrastructure, one live candidate with a mid-2027 verdict, and the
   concordance stream lives on a laptop that has already slept through a run.
   The cheap-uncounted-sandbox / strict-confirmatory split exists in the repo
   (mechanics_clean, uncounted diagnostics) and is underused. A $5/mo
   always-on box protects the one asset on the critical path that cannot be
   regenerated.

## 2. Corrections ledger

Verification corrected the review in these places; corrected values are the
ones cited anywhere in this document.

| As originally stated | Verified state | Evidence |
|---|---|---|
| `src/prism/data/alpaca_data.py:18` | `src/prism/live/alpaca_data.py:18-19` (docstring); the setting itself is the constructor default at `:93` | sweep of `src/` (no `prism/data/` exists) |
| Promotion conjuncts and crash-deferral in SPEC §3/§4 | Both live in `docs/momentum_design.md` §3/§4; SPEC §3 is the six-laws triage | `momentum_design.md:102-128` |
| "Periodic-Sharpe conjunct cleared 0.0293 vs 0.0203" | That was the *discovery-run viability lens* (cert 001 §8); the promotion conjunct is unadjudicated until M6; margin +0.0090 | `results/demotion_b1/summary.json:33,38` |
| "Fold-flattening = 63% of total turnover" | A derivation from aggregates, not a recorded stat; B1-only (residual runs ≈14%); overstates incremental cost | `folds.json` fold-0 `avg_turnover` = 3.0/63 exactly |
| "Slippage +135 to +884 bps" | Derived signed range **−233 to +884 bps**, 3 price improvements; no slippage field is ledgered | `runs/paper_loop_momentum/fills.jsonl` |
| "Saturday's sweep fills" | The sweep ran Monday 2026-07-13 06:50 PT and submitted **zero** residuals; no sweep fills exist | `runs/paper_loop_momentum2/sweep.log` |
| "OPG fills print at the official opening cross" | Alpaca paper *simulates* the auction (22/101 filled); fills are sim prints | `src/prism/live/loop.py:360-361` |
| "Dividends internally consistent on both evidence streams" | **False** — B1's ledger is price-return-only (`no_dividends` tag per I-7); paper equity accrues dividend cash via the broker | `research/scripts/stat_arb_residual_wfo.py:591`; §4 below |
| "ADS is a probable ticker-reuse" | Wrong company (Adidas AG, Xetra) for the entire range — vendor collision, not temporal reuse | `docs/data_integrity_diagnostic.md` |
| "IEX $1M floor ⇒ effective $30–50M, unmeasured" | Confirmed no correction factor exists; bite is bounded on the current S&P 500 book and binds for the mid-cap expansion | `src/prism/residual/factors.py:253-258`, `alpaca_data.py:42-44` |

## 3. Upgrade 1, executed: the collision sweep

Full frame, results, and remediation proposals:
`docs/data_integrity_diagnostic.md`. Summary: **8 suspects in 574 caches**
(6 wrong-instrument, 1 corrupted splice, 1 benign false positive). **B1 is
clean** — zero contaminated days held. The cert 001 residual family held INFO
and WLTW for 61–235 days at ±10 bps total contribution against −0.35 to −0.75
annualized certified results: no verdict moves. FB (retired ticker) sits in
the live universe file; the sweep becomes an M6 pre-flight; remediation
(cache quarantine, `RENAME_TABLE` seeding, membership-interval closure,
`sp500_current.txt` regeneration) awaits owner decision.

## 4. Upgrade 2, executed: the dividend wedge, measured

**Instrument:** `research/scripts/dividend_wedge.py` (tests in
`tests/test_dividend_wedge.py`). **Source:** the Alpaca corporate-actions
endpoint — a data surface the repo had not touched, but the same admitted $0
vendor and key the live loop already reads bars from. Uncounted diagnostic.
**Evidence:** `results/dividend_wedge_2026-07-13.json` (raw records beside it
locally, refetchable via `--records_json`).

**Result:** on B1's decided book over 2021-03-30 → 2026-06-12, the net
dividend flow is **−36.1 bps/yr**. Long leg portfolio yield **1.28%/yr**,
short leg **2.00%/yr** — the anti-yield tilt of momentum, measured. 7,519
records; 87 off-calendar ex-dates dropped (~1.2% undercount); specials and
foreign records included; renamed-symbol keying may undercount slightly;
short-side dividend pass-through and withholding asymmetries not modeled.

**Implications (arithmetic on measured artifacts):**

- A total-return B1 ledger would read ≈ **Sharpe 0.434** (0.465 − 36.1 bps /
  11.5% vol, vol unchanged). DSR/PSR shift the same direction; not
  recomputed here.
- The discovery viability lens still clears under total-return accounting:
  periodic ≈ 0.0273 vs hurdle 0.0203 — the wedge consumes ~22% of that
  margin, not the margin.
- **M6 divergence pin (stated now, before any comparison is read):** the
  paper equity stream is total-return and should run ≈ 36 bps/yr *below* the
  certified price-return process systematically. A marginal conjunct-#4
  (rolling PSR) read at M6 must not be misattributed to spine mechanics or
  venue effects — the spine is concordant
  (`docs/replay_concordance_diagnostic.md`) and this wedge is measured. This
  is divergence-ledger context, not an amendment to `momentum_design.md` §3;
  folding it into the pre-registration is an owner decision.

## 5. What was executed and what was not

Executed 2026-07-13: the sweep and its book-exposure verdict (§3), the wedge
measurement and its M6 pin (§4), both instruments committed with tests.
**Not** executed, by scope: no cache quarantined, no `RENAME_TABLE` entry
seeded, no universe file regenerated, no amendment drafted, no counted trial
touched. Standing owner decisions, sharpened but unchanged: the amendment
stack (per-family trial budgets; the data purchase, now carried by three
independent bricks; a real-money cost-calibration micro-account), the
crash-convex ETF-trend second sleeve, an always-on box for the paper stream,
and a carry-mode fold-flattening quantification before M6.
