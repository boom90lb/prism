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

## 8. ECHO identity settled: genuine EchoStar, a rename artifact (2026-07-17)

**Status: uncounted data-hygiene diagnostic; read-only.** Instrument: a
cross-vendor diff of the quarantined cache against Alpaca daily bars (SIP
feed, `adjustment=split` — the stack's price convention) via
`AlpacaBarSource`; no repo data touched. Evidence:
`results/echo_identity_diff_2026-07-17.json`.

**Verdict: the quarantined series is the continuous, genuine EchoStar
Corporation history.** Five fingerprint closes match exactly on both vendors
(2020-01-02 43.21; 2021-12-31 26.35; 2023-12-29 16.57; 2025-08-26 50.87 —
the +70.25% EchoStar/AT&T spectrum-deal day; 2025-12-31 108.70). Volumes
agree at consolidated-tape scale (cache rounds to hundreds: 17,427,700 vs
17,427,709 on 2023-12-29; 46,579,100 vs 46,579,122 on 2025-08-26). Full
overlap, 1,621 sessions (2020-01-02 → 2026-06-15): 1,612 closes within
$0.005; max |Δclose| $0.02 on three sessions (2020-01-22, 2021-01-22,
2021-01-25) — cross-vendor close-rounding noise, not an identity signal.
Symbology cross-check: Alpaca serves a bit-identical series (1,643 sessions,
max |Δclose| 0.0) for `ECHO` under current symbology and `SATS` under
`asof=2026-06-01` — one instrument chain across the rename.

**Mechanism.** SATS → ECHO ticker change effective 2026-06-24 (company
release 2026-06-22; CUSIP unchanged — a pure rename), and the vendor
backfills the entire chain under the new symbol. The missing 2021-09 pin
that quarantined the cache is therefore *expected*, not indicting: the
ticker ECHO belonged to Echo Global Logistics only until The Jordan
Company's $48.25 cash buyout **completed 2021-11-23** (announced
2021-09-10; §7 said "December-2021" — corrected here). On 2021-09-10 the
cache prints an ordinary −2.1% EchoStar day, exactly as it should. §7's
"the pre-2022 segment is not the company the PIT universe means by ECHO"
remains true in *ticker* space; the *data* is genuine EchoStar, so the
quarantine reason (identity unverified) dissolves.

**What the quarantine actually costs.** `data/SATS_1d_*.parquet` — resolved
in the 2026-07-14 lineage — is close-identical to the quarantined ECHO cache
on all 1,621 sessions (same vendor chain under both symbols), so EchoStar's
genuine bars already serve the SATS membership interval (2026-03-23 →
2026-06-24; it joined the index at the March-2026 rebalance, S&P DJI release
2026-03-06). What is lost is the continuation: the 2026-07-14 membership
carries `ECHO 2026-06-24 → open` as a *separate, unresolved* interval, and
EchoStar is absent from `sp500_current.txt` — **a legitimate current S&P 500
member excluded from the live tradeable universe** since the rename date.

## 9. ECHO un-quarantine packet (as posed to the owner 2026-07-17; nothing executed)

1. **`RENAME_TABLE` gains `"SATS": "ECHO"`** (`src/prism/io/universe_sp500.py:56`)
   — the FB→META mechanism: vendor fetch under ECHO carries the full genuine
   chain, and the two membership intervals (SATS 2026-03-23 → 2026-06-24,
   ECHO 2026-06-24 → open) merge into one continuous ECHO interval.
2. **Delete the `QUARANTINE_TABLE` `"ECHO"` entry** (line 72) — its stated
   reason is now refuted by §8.
3. **PIT ticker-semantics guard, recorded not mechanized:** before
   2021-11-23 the ticker ECHO denotes Echo Global Logistics — never an
   S&P 500 member (verified: no pre-2026 ECHO interval in either membership
   parquet), no genuine series retrievable. No membership rows change and
   membership masking already confines tradeable exposure to 2026-03-23+;
   the guard lives as a comment on the rename entry so no future consumer
   reads the vendor's backfilled pre-rename ECHO bars as "ticker ECHO,
   then an index member."
4. **Update the content pin** `tests/test_universe_sp500.py:170-172`
   (RENAME_TABLE gains SATS; QUARANTINE_TABLE set drops ECHO).
5. **Regenerate universe artifacts** (new asof): ECHO leaves the skip-list
   and enters `sp500_pit_resolved_<asof>.txt` and `sp500_current.txt`;
   restore the cache from `data/quarantine/` (content verified §8) or
   refetch; retire the superseded SATS cache; re-run the integrity sweep
   (the standing M6 pre-flight) expecting zero suspects. The 2026-07-15
   lesson stands (live-loop fix `64d7ea1`): universe-file changes bite at
   the next run's *valuation*, and ECHO becomes fetchable — and tradeable at
   the next refresh — immediately after regeneration.
6. **Decision context, not data integrity:** EchoStar's DBS/wireless
   subsidiaries filed prepackaged Chapter 11 on 2026-06-30; the violent
   2026-06-11/12 bars (+11.2% on 16.4M shares, then −11.0% on 50.1M — both
   vendors concur) are that event's run-up. The parent equity remains listed
   and an index member; if un-quarantined, the momentum sleeve can hold it
   subject to the ordinary screens.

## 10. ACT: the seat that never closed — a rename-invisible membership defect (2026-07-17)

**Status: found by the IEX-eligibility pre-flight (commit `7393cee`); posed
and approved 2026-07-17; execution record §11.** The membership tables carry
`ACT 1999-04-12 → open` and `sp500_current.txt` lists ACT, but the vendor's
ACT is Enact Holdings (NASDAQ, IPO 2021-09-16; joined the S&P SmallCap 600
in April 2025, never the 500) — the cache is clean Enact throughout.

**Identity and mechanism.** The row is the Watson Pharmaceuticals → Actavis
→ Allergan plc seat, keyed under its 2013–2015 ticker: WPI → ACT effective
2013-01-24 (Actavis FY2013 10-K); ACT → AGN effective at the 2015-06-15
open (MIAX corporate-action alert, 2015-06-12); seat ended when DexCom
replaced Allergan plc prior to the 2020-05-12 open (S&P DJI release
2020-05-06, cited for AbbVie's acquisition). Wikipedia's changes table
records the addition under the backfilled chain ticker (`1999-04-12 added
ACT`, removed cell empty) and the removal under the post-rename ticker
(`2020-05-12 added DXCM removed AGN`); the 2015-06-15 rename itself is
invisible. `reconstruct_membership` therefore dropped the 2020 removal as
inconsistent — AGN was inactive, its legacy Allergan Inc. interval having
closed 2015-03-23 against `added AAL` — and the ACT interval ran open
forever. Same family as the §6 FB/PCLN/WLTW never-closing renames, with two
twists that rule out the `RENAME_TABLE` relabel mechanism: the successor
ticker previously belonged to a *different company* (legacy Allergan Inc.),
and the chain *ended* in 2020 — a bare ACT→AGN relabel would merge two
companies' intervals, leave the seat open, and invite an un-reviewed vendor
fetch under AGN.

**Exposure (quantified at discovery, commit `7393cee`; no verdict moves).**
B1 and both §7 reproduction runs held ACT zero days; the residual
demotion/R2 family held it 6–906 days for +4 to +27 bps total per run. The
live loop never traded it — the IEX $1M ADV screen excludes ACT, the only
name it excludes.

**Remediation (approved 2026-07-17).** (1) `CHANGES_PATCH_TABLE`, a third
reviewed table in `prism.io.universe_sp500`: primary-sourced membership
events the Wikipedia table omits, appended to the parsed changes frame
before reconstruction (`apply_changes_patches`); seeded with the single
rename event `2015-06-15 added AGN removed ACT`. Reconstruction then yields
`ACT 1999-04-12 → 2015-06-15` and `AGN 2015-06-15 → 2020-05-12` (legacy
`AGN 1990-01-01 → 2015-03-23` untouched; every other interval unchanged).
PIT guards recorded, not mechanized (the §9-item-3 pattern): 1999-04-12 →
2013-01-24 the seat traded as WPI, and vendor ACT from 2021-09-16 is Enact.
(2) `QUARANTINE_TABLE` gains ACT (the ADS precedent: genuine bars, wrong
instrument for what the universe means by the symbol) — this alone keeps
ACT out of `sp500_current.txt` at any regeneration and keeps Enact bars out
of any future deep-history pull against the 1999–2015 interval. (3) The
Enact cache moves to `data/quarantine/`. (4) Content pins updated; the
patch mechanism unit-tested — the defect is reproduced in miniature and
closed (`tests/test_universe_sp500.py`). (5) Artifacts regenerated under
the standing reproduction gate (§11). AGN is deliberately *not*
pre-quarantined — no fetch evidence existed; its probe outcome is recorded
in §11. Norgate cross-check hook: when the A2 trial lands, diff its PIT
membership for the chain against the patched intervals — an independent
check on the 1999-04-12 start date, which is Wikipedia-sourced like every
event this builder consumes.

## 11. Execution record: §9 + §10 remediation (2026-07-17, owner sign-off)

**Mechanism landed** (suite **782 passed, 1 skipped** after): `RENAME_TABLE`
gains `SATS → ECHO` with the §9-item-3 PIT guard as its comment;
`QUARANTINE_TABLE` drops ECHO and gains ACT; `CHANGES_PATCH_TABLE` +
`apply_changes_patches` wired into the builder; content pins and the
§10-in-miniature reconstruction tests updated (`tests/test_universe_sp500.py`).
Ops fix folded in: `_pull_prices` exempts covering-cache hits from the
rate-limit pacing (`DataLoader.has_cached`, unit-tested) — pacing exists to
protect API calls and a cache hit makes none; a mostly-cached regeneration
drops from ~90 minutes (the 8.5 s/name sleep, all 633 names) to ~12.

**Caches.** ECHO restored from `data/quarantine/` (identity verified §8);
SATS (superseded duplicate — close-identical to the ECHO cache on all 1,621
sessions) and ACT (clean Enact Holdings, wrong instrument for the seat)
moved in. Quarantine now holds nine files; the two new arrivals are
*genuine-but-wrong-for-the-symbol* parkings, not corrupted series.

**Regeneration** (asof 2026-07-17, pull window unchanged 2020-01-01 →
2026-06-16): 633 window members (SATS+ECHO merged to one name; ACT left the
window entirely, its interval now ending 2015-06-15), 869 ever-members, 897
intervals; **569 resolved, 64 skips (4 in-window quarantined)**.
`sp500_current.txt`: 502 names; the delta against 2026-07-14 is exactly
{−ACT, +ECHO}. Membership carries `ACT 1999-04-12 → 2015-06-15`,
`AGN 2015-06-15 → 2020-05-12` beside the untouched legacy Allergan Inc.
interval, and ECHO's two adjacent intervals `2026-03-23 → 2026-06-24 →
open` — one instrument chain across the rename. **AGN probe: the vendor
returned nothing** — no cache created, AGN joins the measured skip-list as
an ordinary unretrievable delisted name; no quarantine entry needed.

**Post-regeneration sweep** (`results/data_integrity_sweep_2026-07-17.json`):
570 caches, **zero wrong-instrument suspects**; only the two benign
share-count flags (NVR, HONA). ECHO's restored cache passes on a full US
calendar.

**Reproduction gate — PASS** (pre-stated: mean daily active share ≤ 0.01,
max ≤ 0.05, no removed-name holdings beyond membership deltas). B1's exact
config on the 2026-07-17 artifacts (scratch trial ledger — not a counted
trial) against the `583b9155eab7` lineage: ACT is the only removed column
and was **never held** in the base book; the SATS→ECHO relabel is exact
identity (zero phantom turnover); 117 of 1,308 days differ, every one a
single decile-boundary substitution (max |Δw| 0.0104 ≈ one slot), mean
daily active share **0.00122**, max **0.0208**. Headline drift: Sharpe
0.4790 → 0.4855, avg turnover 0.05005 → 0.05006. Evidence:
`results/b1_repro_diff_2026-07-17.json`; run dir
`results/demotion_b1_repro_2026-07-17` (local). **The remediated lineage is
now `37ed61308aca`; the M6 extension runs on it citing this section — the
bridge chain reads 000b74941cfd (certified, §7 control bit-exact) →
583b9155eab7 (§7) → 37ed61308aca (§11).**

**Live surface.** ACT leaves the live universe file having never been held
(the IEX $1M screen's only exclusion, §10); no held-name departure is
involved, so the 64d7ea1 valuation mechanics have nothing to absorb. ECHO
becomes fetchable at the next nightly run and tradeable at the next
refresh, subject to the ordinary screens — the §9.6 context (DBS/wireless
subsidiaries' prepackaged Ch11 of 2026-06-30) stands as owner-acknowledged
decision context.

**Anomaly, recorded:** the first gate invocation exited 1 with no output;
the identical rerun (output file-redirected) completed cleanly and is the
run recorded above. Both invocations pointed at the scratch ledger.
