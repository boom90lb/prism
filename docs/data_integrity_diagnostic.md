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

## 6. Remediation proposals (as posed to the owner 2026-07-13)

1. **Quarantine the wrong-instrument caches** (ADS, FB, PCLN, SBNY, SOLS; the
   INFO and WLTW post-rename tails). Absence is the honest state under the
   zero-delisted-bars doctrine — a wrong series is strictly worse than no
   series. INFO and WLTW carry genuine pre-event segments, so the decision
   per name is truncate vs remove.
2. **Seed `RENAME_TABLE`** (`src/prism/io/universe_sp500.py`, empty by
   design pending reviewed entries) with FB→META, PCLN→BKNG, WLTW→WTW, and
   close rename intervals in the membership reconstruction — FB, PCLN, and
   WLTW intervals ran to 2026-06-16 as if the old symbols were still members.
3. **Regenerate `sp500_current.txt`** without retired tickers.
4. **Adopt this sweep as an M6 pre-flight** and a periodic ops check; any new
   suspect gets the §4 exposure treatment before a counted run consumes the
   cache.

## 7. Remediation executed (2026-07-14, owner decision: "no wrong data — regenerate")

**Mechanism, not one-off cleanup.** `QUARANTINE_TABLE` (symbol → reason) now
sits beside `RENAME_TABLE` in `prism.io.universe_sp500`: quarantined names
are never fetched and can never resolve — they join the measured skip-list
with their reasons echoed in the coverage ledger. `RENAME_TABLE` is seeded
with the three reviewed remaps (FB→META, PCLN→BKNG, WLTW→WTW), which also
merges the old symbols' membership intervals into their successors' — the
never-closing rename intervals dissolve rather than needing a parser fix.
The universe builder now also *produces* `sp500_pit_resolved_<asof>.txt` and
`sp500_current.txt`, which previously had no producer script. Both tables
are content-pinned by tests.

**Regeneration (asof 2026-07-14, pull window unchanged 2020-01-01 →
2026-06-16).** 634 window members → **570 resolved, 64 skips (5
quarantined)**; the frozen 2026-06-16 artifacts are untouched and certified
configs still reference them. `sp500_current.txt`: 502 names, no retired
tickers. Three late-June index additions entered with verified-clean caches
(FLEX, MRVL; HONA is a one-bar nascent listing). The fourth, **ECHO, failed
verification and joined the quarantine**: its continuous 2020–2026 series
never shows Echo Global Logistics' September-2021 buyout pin at ~$48 or the
December-2021 going-private stop — whatever the vendor is serving, the
pre-2022 segment is not the company the PIT universe means, and the current
listing's identity is unverified (owner can clear it with one broker lookup
and a refetch scoped to the new listing's life).

**Caches.** All eight contaminated files moved to `data/quarantine/`
(reversible; delete at will): ADS, ECHO, FB, PCLN, SBNY, SOLS, INFO, WLTW.
INFO's genuine 2020–2022 segment and WLTW's pre-rename segment are preserved
there; in the tradeable universe INFO is quarantined at the symbol level and
WLTW is carried by WTW's full genuine history.

**The reproduction gate (pre-stated: books materially unchanged → proceed;
else stop).** Control first: re-running B1's exact config on original
universe files and original caches reproduces the certified run **bit-exact**
— same `config_hash 000b74941cfd`, identical summary to full float precision,
`target_weights` max |Δw| = 0.0 (1308×574). Then the remediated run
(`sp500_pit_resolved_2026-07-14`, quarantined caches absent, scratch trial
ledger — **not** a counted trial): no removed name was ever held, no added
name enters (memberships start 2026-06-22+), and the only differences on
common names are decile-boundary substitutions — mean daily active share
**0.0038**, max **0.0214**, 307 of 1308 days touched. Headline stats move
slightly favorably (Sharpe 0.4654 → 0.4790, total +27.6% → +28.6%, periodic
0.0293 → 0.0302): the contamination had been mild drag. **Gate: PASS.**
Evidence: `results/b1_remediation_repro_diff.json`; run dirs
`results/demotion_b1_repro_{control,remediated}` (local).

**Standing consequences.** The certified B1 numbers stand as certified under
`config_hash 000b74941cfd`; the remediated lineage is `583b9155eab7`, and the
M6 extension runs on it citing this section as the certified-window bridge.
Post-remediation sweep over all 570 caches: **zero wrong-instrument
suspects** (only the two benign share-count flags, NVR and HONA) —
`results/data_integrity_sweep_2026-07-14.json`. Residual risks, stated: the
vendor can mint new collisions at any future delisting/rename (the sweep is
the standing M6 pre-flight for exactly this), and the loader's fetch-error
logging exposes the vendor API key in URLs (flagged separately for fix +
key rotation; out of this diagnostic's scope).
