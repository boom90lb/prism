# Constitutional amendments, 2026-07-14

**Status: drafted under owner delegation 2026-07-14 ("the amendment stack I
leave to you"); RATIFIED by the owner's 2026-07-17 push of the dedicated
amendment commit (`2162806` — since collapsed into the v0.3.3 release
commit; the before/after diff resolves on branch
`backup/pre-squash-v0.3.3`, pushed to origin).** Three amendments. None moves a pre-registered trial value, reopens a
closed selection set, or touches a certified artifact: the residual ledger
stays closed at 17 rows (cert 001 §9), momentum's budget stays exactly 8
(`docs/momentum_design.md` §2), and every adjudication rule in ratified
pre-registrations stands as written. These are maintenance of the
constitution's own machinery, argued from measured evidence.

## A1. Per-family trial budgets (SPEC §10)

**Problem.** SPEC §10 pins "the cumulative construction-trial budget
(pre-registered, ≤ ~30 counted trials)" — written for the residual-reversion
sleeve's kill-criterion, then generalized downstream into an enterprise
lifetime cap (`docs/aim_portfolio_preregistration.md` §4 reads it that way).
Arithmetic today: 17 spent + 8 reserved for momentum = 25, leaving ~5 counted
trials for every future program combined. The flat cap conflates two
different multiplicities: within-selection-set search (which DSR already
deflates, N5) and across-family search (which a shared counter does not
control — it just starves later programs regardless of merit).

**Change.** The budget becomes per-program: each candidate family receives a
fixed, never-refilled budget in its own ratified pre-registration, counted in
its own selection-set ledger (exactly the momentum precedent). Across-family
multiplicity gets the controls that actually bind it: every pre-registration
records the count of prior counted programs; at most one counted program runs
at a time (`docs/handoff.md` "one verdict at a time"); and promotion always
requires genuinely out-of-sample confirmation, never same-sample selection
across families.

**Edits.** SPEC §10 third kill-criterion disjunct and the budget-logging
paragraph (SPEC.md ~L552-557) — before/after recorded in the diff of the
amendment commit. `docs/aim_portfolio_preregistration.md` §4 (an unratified
draft) reads on the old model; it re-derives its budget from this amendment
at its own ratification, and is deliberately not edited here.

## A2. Bounded data budget (SPEC §1 mandate, §4)

**Problem.** The mandate's "zero-data-budget … survives on free data tiers"
clause forecloses purchases whose absence is now a measured cost, three times
over: (i) the mid-cap breadth expansion — the one capacity lever the breadth
diagnostic identifies — is blocked solely on survivorship-complete history
(`docs/momentum_design.md` §0); (ii) the free spine's symbol resolution
produced the vendor-collision class — six wrong-instrument caches inside the
PIT universe (`docs/data_integrity_diagnostic.md`); (iii) the dividend wedge
could not be measured on the spine vendor at all (`/dividends` 403,
`docs/operations.md`) and required a second vendor's corporate-actions
endpoint. The $0 constraint was the right founding discipline; as a permanent
identity it now buys integrity risk to save ~$1k/yr.

**Change.** $0 stays the default; a named dataset may be purchased under a
ratified amendment when (i) it closes a measured integrity or capacity gap
free tiers cannot, (ii) total annual data spend stays ≤ $1,000, and (iii) the
dataset's provenance and license enter the §5 coverage-ledger discipline. The
standing first candidate is a survivorship-complete US equity history. The
purchase itself is an owner act; this amendment only makes it constitutional.

**Edits.** SPEC §1 mandate sentence (~L69-72); one amendment paragraph in §4
after the stack-table intro (~L226). The §4 heading keeps its name — "$0"
remains the default posture and external references point at it.

## A3. Real-money cost-calibration micro-account (SPEC §10 carve-out)

**Problem.** "No live capital is risked below `net_edge`" (SPEC §10) plus the
R2 designation of the *paper* loop as the I-9 cost instrument together imply
cost calibration waits on simulator prints. But the simulator cannot price
spread or impact: OPG fills are auction-sim prints (~20-25% fill rate), sweep
fills cross a simulated book, and every observed "slippage" number to date is
overnight-drift noise (`docs/program_review_2026-07.md` §1.5). The recompute
trigger ("paper fills contradict the 1bp assumption", `docs/handoff.md` §8)
can therefore fire on artifacts or never fire at all. Measuring cost is not
claiming edge; the gate was written for the latter.

**Change.** A real-money micro-account operated purely as the I-9
cost-measurement instrument is a measurement expenditure, not a deployment,
and does not violate the `net_edge` gate, under pins: total equity ≤ $2,000
(default cap; owner sets the actual figure at funding and it is recorded);
orders mirror the paper book's decisions at trivial size; fills are
venue-tagged `real` in the calibration ledger and preferred over paper prints
once a bucket reaches `min_fills`; its P&L never enters any claim packet or
promotion read. Opening, funding, and any change to its size are owner acts.

**Edits.** One carve-out paragraph in SPEC §10 after the no-live-capital
gate (~L539-541); one cross-reference sentence in the R2 paper-instrument
paragraph (~L656-663). The `docs/handoff.md` §8 recompute row is untouched —
real fills contradicting the 1bp assumption trigger exactly the recompute it
prescribes, which is the point.

## What these amendments do not do

No cache, ledger, certification, ratified pre-registration, or adjudication
rule changes under any of the three. The momentum program's promotion still
reads only at M6 + paper under `docs/momentum_design.md` §3 as ratified. The
SPEC status stamp gains an amendment note; the full before/after text lives
in this commit's diff, which is the auditable record.
