# Dead-ticker price layer — forward accumulation design

> **Status: DRAFT — not ratified.** Infrastructure design for the
> survivorship-free data program (the observatory's price companion), drafted
> 2026-07-18. Ratification is the owner's dedicated commit (AGENTS.md §2).
> This designs *accumulation machinery*, not a trial: nothing here touches
> counted budgets, and the C2/C3 registration (no known free source for
> pre-2020 delisted prices, docs/replication_preregistration.md §8) stands —
> this layer is how the C1-era gap closes going forward, not a claim about
> the past.

## 1. Problem

The membership layer is solved forward: the observatory captures S&P 500/400
constituents daily (SSGA holdings) and the full active+delisted listing
universe monthly (Alpha Vantage), with append-only manifests. But membership
without prices cannot backtest: the certified runs carry **zero
delisted-ticker price rows** (standing fact; the two apparent exceptions were
wrong-instrument collisions, docs/data_integrity_diagnostic.md). Every
future ex-member is currently on track to be exactly as unpriceable as every
past one. Survivorship bias in prices is fought *forward*: capture the whole
active universe's bars while the names are alive, and when a name dies its
history up to the death date is already on disk.

## 2. Goals / non-goals

- **Goal:** from a start date T0, every name in the observatory's active
  universe has daily bars accumulating, so the delisted-price gap closes at
  one day per day from T0 with zero backfill dependence.
- **Goal:** identity discipline strong enough to survive ticker reuse — the
  measured hazard (FB/PCLN/SBNY/WLTW rename-reuse strays; RENAME_TABLE and
  QUARANTINE_TABLE seeds exist in-tree).
- **Non-goal:** purchasing (the A2 evaluation was reversed for accumulation,
  docs/data_purchase_evaluation.md §6). **Non-goal:** pre-2020
  reconstruction (registered as unclosable at $0, replication pre-reg §8).
  **Non-goal:** intraday anything.

## 3. Design

**Source.** Alpaca IEX daily bars for the active universe (free key, batch
endpoint, ~200 req/min — the live loop already reads this feed; ~14k names at
multi-symbol pagination is a few hundred requests). The Twelve Data spine
stays the research-side source for member histories; this layer is breadth
over depth. Names that leave the IEX-feed universe (delisting) stop
producing rows — which is the event being recorded, not a failure.

**Identity.** The key is `(ticker, listing_interval)`, not `ticker`: a row
joins the observatory's listing snapshots (first_seen/last_seen from the AV
delisted universe and the Nasdaq directories), with the RENAME_TABLE chain
applied before storage so a renamed continuation (SATS→ECHO class) extends
one series and a reuse (PCLN class) starts a new one. SEC CIK from the
observatory's tickers capture is stored beside the interval where available
— the durable cross-check identity.

**Storage.** Parquet per `(ticker, interval_start_year)` under the
observatory repo's data tree, with the same append-only manifest convention
as the membership capture (unchanged-duplicate rows recorded as no-change
evidence, gap_days visible by construction). SQLite derivation joins prices
to membership at read time; prism consumes exported per-name parquet, never
the observatory's internals.

**Cadence.** Piggyback the existing two daily crons (13:45/22:45 UTC): the
price capture appends after the membership capture in the same workflow.
A missed day is a one-day gap in a delta feed — the capture fetches a
trailing window (e.g. 5 sessions) so single-run failures self-heal, and only
a multi-day outage widens gap_days.

**Verification.** C0-style cross-vendor spot check: a rotating sample of N
names/day compared against the Twelve Data spine on overlapping sessions
(the ECHO settlement's fingerprint-close protocol, 5/5 exact closes, is the
precedent for what "same instrument" means). Adjustment-basis discipline:
store unadjusted + split factors if the feed provides them; never store
dividend-adjusted series (SPEC §5's dividends-as-cash doctrine).

## 4. What this buys, concretely

- **C1 (mid-cap 2020–26 replication cell):** delisted-price coverage begins
  accruing at T0; the cell's delisted-bar gap stops growing.
- **M-series extensions:** future counted windows (M6 ≥ 2027-06) overlap the
  accumulation era — the first counted runs where index leavers can remain
  priceable to their exit.
- **The eligibility screen:** held-name valuation for names that leave the
  index AND the venue (the POOL class) gains a second source.

## 5. Open questions for ratification

1. **Home:** prism-observatory (recommended — it owns identity + manifests)
   vs a third repo. The prism side only ever reads exports.
2. **IEX-feed completeness for thin names:** the feed's ~5% share is an
   execution constraint for prism; whether its *daily bar coverage* of
   thin/mid-cap names is complete enough for a price archive needs a probe
   (compare a week of IEX bars vs the spine on the current S&P 400).
3. **Start date T0:** on ratification, or aligned to the next observatory
   release tag.
4. **Alpha Vantage backfill lane:** AV serves daily history for *some*
   delisted tickers; whether that is worth a 25-req/day drip alongside the
   monthly universe pull is a separate probe with its own manifest lane —
   opportunistic backfill must never gate the forward capture.
