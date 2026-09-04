# Trend T-series read — 2026-09-03

**Kind:** read record for `docs/trend_design.md` §4 (the fragility kill and the
convexity admission read). **Counted cells:** T0–T4, run 2026-09-03 21:09 PDT
(ledger timestamps 2026-09-04 04:09 UTC) at commit `56eb3bf` — the commit that
landed the driver and the §2 seam pin at 21:09:18 PDT, three seconds before
T0's row at 21:09:21; tracked tree clean, 11 untracked root dotfiles, none
imported. **Ledger:** `results/trend_v1_trials.jsonl`, 5 rows, T0…T4 in order
(rows carry the config hash and Sharpe; the code commit is in each packet).
**Budget:** 6 registered; 5 spent; T5 remains. **Driver:**
`research/scripts/trend_wfo.py`; **adjudicator:** `research/scripts/trend_adjudicate.py`
→ `results/trend_t0/adjudication.json`.

**Sample.** 4,981 sessions 2006-09-27 → 2026-07-17 (the last bar before family
ratification), 74 folds of B1's geometry (formation 312, test 63, the last two
target rows of each fold flat), 4,662 scored return rows; the last scored
session is 2026-07-08 and the 7 trailing sessions fall in no test window. Entry
at first *cached* bar + 252: SPY 2007-09-28, EFA 2008-01-03, EEM 2008-12-31, TLT 2008-01-03, IEF 2008-01-03, LQD 2008-01-03, HYG 2008-04-10, GLD 2008-01-03, PDBC 2015-11-09, UUP 2008-02-29. Convention: the registered trailing
252-bar **total** return on a total-return close built from the issuer
distribution histories (`data/distributions`, `c55cbc1`), dividends credited as
cash. Hurdle 3.78 % T-bill (FRED DTB3, 2026-09-02; the same-day momentum record
kept B1's July 3.71 % for comparability with the certified packet).

## Verdicts

1. **No standalone carry.** T0: net annualized Sharpe **0.042**, net total
   return **+1.4 %** over 18.5 scored years (0.08 %/yr). By the ladder's
   letter T0 is `net_edge` (net return > 0, periodic Sharpe > 0); the cash
   hurdle is a promotion conjunct, not a tier rung, and T0 earns 6.6 % of it
   (periodic 0.0027 vs 0.0406). Decomposition: gross +15.6 % =
   distribution cash +28.9 % on a net-long book (average net exposure +0.39)
   plus a **negative price leg (−13.5 %)**, minus 13.1 % cost. The registered
   verdict string's "positive-carry" is that nominal +1.4 %; the TSMOM signal's
   price P&L is negative in the pinned cell. Only T1 has a positive price leg.
2. **Fragility clause: survives under the pinned reading; the reading is decisive.**
   Median net annualized Sharpe of T1–T4 = **0.074** (clause (a) not
   negative). Clause (b), "net result" read as the statistic clause (a) names
   (net annualized Sharpe; pinned in `docs/momentum_m_series_read_2026-09-03.md`
   at 14:50 PDT, before these runs): no probe is negative → no flip. Under a
   net-total-return or claim-tier reading, **T4 flips** (-2.0 % vs T0
   +1.4 %, `gross_edge` vs `net_edge`) at a magnitude above T0's estimate and the
   clause would fire; T4's flip is variance drag on a +0.18 %/yr arithmetic
   mean, sign-equivalent to its positive Sharpe. With T0's point estimate
   economically zero, the magnitude qualifier has no discriminating power.
3. **Convexity admission read: FAILS.** Non-overlapping 21-bar windows on T0's
   decision grid (221 complete windows; sleeve window = return rows d+1…d+21,
   i.e. open(d+1) → open(d+22); the engine is fed opens and this was rebuilt
   from the artifacts to 1.8e-16). Decile rule as pinned in the adjudicator:
   m = ⌊0.1 n⌋ smallest conditioning values, pass iff the conditional mean sleeve
   return is no worse than the unconditional mean, on **both** legs.
   - SPY leg (close(d) → close(d+21), pre-committed): n = 221, m = 22, conditional
     +0.06 % vs unconditional +0.02 % → passes by +0.04 pp, a tie (0.13 se).
     Alignment-dependent: on MOO-aligned open(d+1) → open(d+22) SPY the leg
     fails (−0.29 % vs +0.02 %); on price-return closes it fails; at 5 % or 20 %
     tails it fails. No smile either way.
   - B1 momentum-book leg (2021-03-31 → 2026-05-07 overlap, the B1 stream's
     span): n = 62, m = 6, conditional **-1.70 %** vs unconditional +0.32 %
     over the same 62 windows → **fails by -2.02 pp**. This is a clear fail,
     not a power problem: the 62 window returns have sd 1.72 %, so the six-window
     mean has se 0.70 % and the shortfall is 2.9 se; a permutation test gives
     P(random 6 of 62 ≤ −1.70 %) ≈ 0.002 (200k draws); sleeve–B1 window
     correlation is +0.42 (Spearman +0.35, p = 0.005). Robust to every variant
     tested by three independent readers: quantile decile (7 windows), in-fold
     alignment, rolling windows (m = 1,288), the full-sample unconditional
     benchmark (−1.72 pp), 5 % and 20 % tails, every leave-one-out. The six
     conditioning windows (2022-06-30, 2022-12-29, 2023-11-30, 2024-07-03,
     2025-02-04, 2025-06-05) are momentum-reversal months, mostly with SPY up:
     the sleeve is positively exposed to the momentum book's drawdowns. The
     sample contains no momentum crash.

   Registered verdict (§4): **"positive-carry, non-convex; portfolio admission refused"**. N6
   does not apply: a 10-name time-series sleeve has no cross-sectional rank.

## Results

| cell | net ann. Sharpe | net total return | gross total return | price-only gross | dividend cash | total cost | turnover/day | max DD | periodic net SR | periodic hurdle | DSR as written / uniform (N = 6) | ladder tier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 (pinned cell) | 0.042 | +1.4 % | +15.6 % | -13.5 % | +28.9 % | 13.1 % | 0.041 | 19.8 % | 0.0027 | 0.0406 | NaN (first row) / 0.247 | net_edge |
| T1 lookback 126 | 0.393 | +52.3 % | +77.1 % | +37.9 % | +25.1 % | 15.1 % | 0.045 | 16.1 % | 0.0248 | 0.0379 | 0.619 / 0.794 | net_edge |
| T2 skip 0 | 0.104 | +8.5 % | +23.5 % | -8.4 % | +29.9 % | 12.9 % | 0.040 | 22.1 % | 0.0065 | 0.0394 | 0.274 / 0.337 | net_edge |
| T3 decision_every 63 | 0.044 | +1.3 % | +12.5 % | -14.9 % | +27.9 % | 10.5 % | 0.032 | 24.9 % | 0.0027 | 0.0363 | 0.227 / 0.249 | net_edge |
| T4 equal notional | 0.024 | -2.0 % | +10.5 % | -13.1 % | +24.0 % | 12.0 % | 0.039 | 26.7 % | 0.0015 | 0.0312 | 0.223 / 0.223 | gross_edge |

"DSR as written" is the driver's sequence-dependent value (T0's is NaN by
construction as the first row); "uniform" recomputes every row against the
final 5-row ledger at N = 6. Neither is an adjudication input. Per probe: 6−1
(T1) is far stronger than 12−1; the skip month is not load-bearing (T2);
quarterly cadence changes nothing (T3); equal-notional sizing is slightly worse
(T4). No cell clears the cash hurdle (T1 at 65 % of its hurdle).

**T1 cannot be adopted.** Adopting a different cell is a new discovery event
(§3) and repurposing as a standalone-Sharpe sleeve is a new pre-registration
(§4). Further: T1's convexity was not read (registered on T0 only); in the
sibling momentum set 6−1 was the *weak* axis (M2 0.32 vs B1 0.47) while here it
is the strong one, which argues against a robust horizon effect; and any
future trend registration must declare these five rows and the rehearsals
below as prior exposure — T1's 0.39 is already seen.

## What the registration says happens now

- Portfolio admission is refused. `docs/aim_portfolio_preregistration.md`
  (G4a names B1 + trend_v1 as its example second sleeve) and
  `docs/learned_xsection_design.md` (names T5 among firewalled promotions)
  lose that second sleeve; the crash-convexity case for trend is closed.
- T5 stays registered (budget slot 6) as T0 re-run on data ≥ 2027-07-17. §4's
  promotion bullet requires "the convexity read passing on the OOS segment"
  (the full sample is re-read and reported). Two structural facts: the OOS B1
  leg is unreadable until the B1 stream extends past 2026-07-20 (M6 must run
  first; `results/demotion_b1/returns.csv` ends 2026-06-12), and one OOS year is
  ≈ 12 decision bars, so m = 1. The adjudicator's verdict string currently reads
  the full sample only; fix before T5. Nothing waits for T5.
- SPEC §5's breadth and IC levers for the momentum book are unaffected.

## Pinned readings and disclosures (no trial value moved)

- **Per-fold bucket accounting** (`spread_accounting="per_fold_bucket"`, seam pin
  `docs/trend_design.md` §2, 2026-09-03): each fold's formation-window buckets price
  that fold's fills; the sibling froze fold 0's buckets for the whole sample.
  Fold 0 (2007) puts six names at 10 bps; by the last fold four of them are at
  1 bps and PDBC/UUP at 5 bps. Favourable to the sleeve: on the sub-sample
  rehearsal the frozen counterfactual doubled the spread bill (0.1215 vs
  0.0604). Decided from the bucket table after that rehearsal had been seen.
- **Convexity reading pins:** both legs must pass; the B1 leg's unconditional
  mean is over the 62 overlap windows (+0.32 %); SPY conditions on close-to-close
  by pre-committed choice while the sleeve window is open-to-open.
- **Prior exposure (all breaches of §0's "none runs after ratification outside
  the §3 counted set"; disclosed, not counted).** (i) The joint-crash receipt of
  2026-07-22/23 ran T0's mechanics over 2018-01-02 → 2026-07-20 at flat 1 bp
  spread, 1 bp commission, 50 bp borrow on price-return bars, four days after
  ratification: trend max DD −11.1 %, bear-2022 +9.3 %, covid-2020-03 −2.5 %,
  0.7/0.3 blend max DD −12.1 %. (ii) Three scratch rehearsals of the pinned
  configuration on 2006-09-27 → 2021-12-31 (56 of 74 folds) on the real caches,
  in run order: 15:54 PDT total return with frozen fold-0 accounting, packet
  hash `6a3bb0a14c86`, net Sharpe −0.218 (artifacts overwritten and its scratch
  ledger row discarded when the directory was reset); 15:54 price-return
  fallback, packet `9b92e3d66c72` / ledger row `424ba26030a1`, −0.431; 20:45
  total return with per-fold accounting, packet `65101fd55b63` / ledger row
  `bfaa229b53c2`, −0.144. The driver now refuses any rehearsal on the pinned
  cache (guard 8). No full-sample number was seen before the counted runs.
- **Entry rule as realized:** §1 says listing + 252; the driver enters at the
  first *cached* session + 252, and the sample starts at the live loop's SPY
  cache (2006-09-27). SPY (listed 1993) therefore enters 2007-09-28 and five
  names listed by 2004 enter 2008-01-03; fold 0 trades 6 of 10 names and its
  2007-12-24 window (sleeve −10.5 %) is a ~96 % single-name SPY book. Excluding
  that window lifts the SPY-leg conditional mean to ≈ +0.5 %; the B1 leg is
  untouched. Deeper caches would move the realization toward the text.
- **Fold geometry mirrors B1** for comparability and because the band is defined on
  a formation window. It forces a full liquidation and re-entry every 63
  sessions (74 times; each fold's first return row is 0.0), a round trip a live
  monthly sleeve would not pay, so 13.1 % is conservative for the sleeve as it
  would trade. Trend decision bars sit exactly one session after B1's over the
  overlap (offset 1, zero coincidences): "trade the same bars" is delivered as
  cadence, not phase.
- **Data caveats:** EEM's 2008 rows are real closes with missing intraday fields
  and zero volume (first EEM fill 2009 by the entry rule; 82 duplicate closes
  moved ≤ $0.035 by keep-last); PDBC's 2014–15 bars are thin; GLD pays no
  distributions; EFA's 2002–2007 distributions are annual per the issuer table
  (the verification's one "usable_with_gaps"); Alpaca-side holes on 2016-08-01
  (four bond funds) and two SPY dates were resolved from the issuer files.
  Packets record `dirty=true` from untracked root dotfiles only.

## Verification performed

Before the counted runs: three adversarial lenses on the driver (production
diff accepted; registration parity accepted; causality on the real caches
reproduced every accounting quantity of the rehearsal to 1e-15 and found no
look-ahead) and the guard rehearsals. After: five packets validate; ledger
rows equal summaries; each T1–T4 config differs from T0's in exactly its
registered key at its registered value (asserted by the adjudicator).

Blind verification (workflow, four agents plus a reconciler, none reading the
adjudicator's output): two independent derivations reproduced every number
from the raw artifacts (Sharpe and total return to 1e-11, commission spread per
fold to 1e-16, dividend credits to the issuer ex-dates) and reached the same
verdicts on both reads; the fidelity audit passed all nine checks; the critic's
corrections adopted here are the tier wording, the price-leg decomposition, the
B1-leg power statistics, the T5 promotion conjunct and its unreadable OOS leg,
the complete prior-exposure list, the entry-rule realization gap, and the
quarterly-liquidation note. One critic claim was rejected on evidence: the
sleeve window is open-to-open, not close-to-close.

## Command lines

From the repo root: T0 `python -m research.scripts.trend_wfo --output_dir
results/trend_t0`; T1 `--lookback 126`; T2 `--skip 0`; T3 `--decision_every 63`;
T4 `--sizing equal_notional` (each with its own `--output_dir`); then `python -m
research.scripts.trend_adjudicate --t0 results/trend_t0 --trials
results/trend_t1,results/trend_t2,results/trend_t3,results/trend_t4 --ledger
results/trend_v1_trials.jsonl --b1_returns results/demotion_b1/returns.csv --out
results/trend_t0/adjudication.json`.
