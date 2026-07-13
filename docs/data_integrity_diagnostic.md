# Data integrity: the vendor symbol-collision class, enumerated and cleared against the books

**Status: uncounted data-hygiene diagnostic, recorded 2026-07-13.** Claims no
economics, enters no selection set, writes no ledger row. Instrument:
`research/scripts/data_integrity_sweep.py` (offline, deterministic; unit
tests in `tests/test_data_integrity_sweep.py`). Evidence:
`results/data_integrity_sweep_2026-07-13.json`.

## 1. The finding it generalizes

The 2026-07 program review surfaced two bad caches by hand.
`data/ADS_1d_2020-01-01_2026-06-16.parquet` holds **Adidas AG (Xetra, EUR)
for its entire range** — prices track the Adidas trajectory throughout,
volumes print in the low thousands, and the series never matches Alliance
Data Systems, the constituent the PIT universe means by ADS. There is no
regime break, so nothing in-band could have caught it.
`data/INFO_1d_*.parquet` carries **105 duplicated dates** (2020-01 → 2021-03,
paired rows with slightly different OHLCV) plus bars four years past IHS
Markit's 2022-02 merger into S&P Global.

The mechanism is vendor symbol resolution: for retired, renamed, or reused US
tickers, Twelve Data answers with whatever instrument currently owns the
symbol — a foreign local line, a new listing, or a sparse remnant — precisely
on the delisted/renamed names where this program had assumed "zero
delisted-ticker bars." That assumption is falsified in the worst direction:
some delisted names have bars, and the bars belong to other instruments. Two
hand-found incidents implied a class; this sweep enumerates it and clears it
against every finished book.

## 2. Method

Five offline tests per cache, thresholds pinned as constants in the script:

1. **Duplicated dates** (vendor series-merge smell).
2. **Bars on NYSE full-day closures** (≥ 3 flags): a US-listed series prints
   none; a foreign-venue line (Xetra) trades straight through them. The
   closure table is hardcoded 2020–2026 and self-audited: the output's
   `closure_bar_counts` counts caches printing a bar on each closure date — a
   near-universal count would indict the table, a handful indicts the caches.
   Observed maximum: **3 of 574**. The table is sound.
3. **Session coverage** vs the expected NYSE calendar (> 15 missing sessions
   flags a sparse/spliced series; halts that long are themselves news).
4. **Volume scale** (median share volume < 50k): an S&P constituent prints
   millions of shares, a foreign local line prints thousands.
5. **Screen passability**: max 20-bar-median dollar volume vs the trials'
   $1M floor, computed from the cache's own numbers — exactly what the
   backtest's eligibility screen saw.

Suspects are then cross-referenced against all thirteen
`results/*/target_weights.csv` panels (ground truth for what each book held,
with return contribution computed from the suspect's own cache closes — the
same series the trial's panel was built from) and against the PIT membership
intervals (`data/universe/sp500_membership_2026-06-16.parquet`).

## 3. Results — eight suspects in 574 caches

| Symbol | Flags | What the cache actually is | Held in any run? |
|---|---|---|---|
| ADS | off-calendar (45), thin, gappy | Adidas AG (Xetra) full-range; membership ended 2020-06-22, before any traded fold | no |
| INFO | duplicates (105), off-calendar (14), gappy (658 missing) | Dual-feed merge of IHS Markit while listed (all closure bars ≤ 2022-02-21), sparse wrong-instrument tail after the SPGI merger | **yes** |
| WLTW | gappy (978 missing) | Genuine Willis Towers Watson to the 2022-01 WTW rename, sparse remnant after; the membership interval never closes | **yes** |
| FB | thin | 244 rows starting 2025-06-26 at ~650 sh/day — wrong instrument; the real Facebook lives in META (full 1621-row history) | no |
| PCLN | thin | 166 rows from 2025-10-16 at ~100 sh/day — wrong instrument; BKNG carries the real series | no |
| SBNY | thin | Rows begin 2024-08-15 — Signature Bank failed 2023-03-15 with **no bars at all** during its actual membership; the 2024+ rows are a different instrument | no |
| SOLS | thin, gappy | Median dollar volume $703/day with a recorded single-day \|return\| of 459,999× — a corrupted splice; membership only 2025-10-30 → 2025-12-22 | no |
| NVR | thin (benign) | False positive by design: a $5–8k/share stock prints ~20k shares but $117M median dollar volume on a perfect US calendar | yes (legitimately) |

Near-threshold, recorded for completeness (no silent caps): single stray
closure-date bars on BSX and DE (2024-05-27) and CPT, FBHS, TTWO
(2025-12-25); LUMN's max daily return of 0.93 is plausibly genuine. FBHS is
otherwise a full, healthy series — the vendor maps some retired symbols onto
the continuing company (FBHS-style) and others onto remnants or foreign
instruments (WLTW-, ADS-style). The inconsistency is the reason to verify by
sweep rather than assume either behavior.

## 4. Certified-book exposure — the load-bearing read

**B1 (the momentum candidate, the program's only live claim): clean.** Zero
days held for every wrong-instrument name across
`results/demotion_b1/target_weights.csv`. Its only flagged holding is NVR
(benign, 143 days, +12.4 bps total contribution on genuine data). The
protection was structural but incidental: momentum's 252-bar full-history
eligibility and the $1M screen happen to exclude gappy and thin series; that
is luck-adjacent, not a designed gate.

**Residual family (the cert 001 evidence set): touched, immaterially.** INFO
was held 61–235 days (2021-03-30 → 2022-03-24) across b2/d1–d5/r2_t1–t4 and
the phaseA reruns; during that window the series is single-feed (the
duplicate region ends 2021-03-11, before the first hold) but still prints
US-holiday bars — foreign-line pricing of the same company. WLTW was held
61–115 days, and the phaseA band rerun held it 2026-04-01 → 2026-06-10,
entirely inside the post-rename tail. Total return contributions per run:
INFO −2.7 to +8.3 bps, WLTW −0.1 to +9.7 bps — over a 5.2-year sample, in
runs certified negative at −0.35 to −0.75 annualized. No sign, margin, or
verdict can move. Certification 001 stands; the exposure is recorded here,
not relitigated.

Membership masks bounded the damage everywhere they could: ADS's membership
ended before the traded window, so the Adidas series was structurally
excluded from every book regardless of its content.

## 5. Live surface

Live decisions read Alpaca IEX bars, not these caches — tonight's book is not
exposed to any of this. Two live-adjacent facts still need eyes: **FB (a
retired ticker) sits in `data/universe/sp500_current.txt`**, the live loop's
universe file; and the M6 extension re-runs the research backtest on these
caches with refreshed tails, so the collision class re-enters the certified
path unless remediated first. This sweep is cheap and offline; it should run
as an M6 pre-flight.

## 6. Remediation proposals (owner decisions; none applied here)

1. **Quarantine the wrong-instrument caches** (ADS, FB, PCLN, SBNY, SOLS; the
   INFO and WLTW post-rename tails). Absence is the honest state under the
   zero-delisted-bars doctrine — a wrong series is strictly worse than no
   series. INFO and WLTW carry genuine pre-event segments, so the decision
   per name is truncate vs remove.
2. **Seed `RENAME_TABLE`** (`src/prism/io/universe_sp500.py:52`, empty by
   design pending reviewed entries) with FB→META, PCLN→BKNG, WLTW→WTW, and
   close rename intervals in the membership reconstruction — FB, PCLN, and
   WLTW intervals currently run to 2026-06-16 as if the old symbols were
   still members.
3. **Regenerate `sp500_current.txt`** without retired tickers.
4. **Adopt this sweep as an M6 pre-flight** and a periodic ops check; any new
   suspect gets the §4 exposure treatment before a counted run consumes the
   cache.
