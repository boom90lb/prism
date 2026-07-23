# Spin-adjustment consistency — sweep of the certified spine panel

> **Status: uncounted diagnostic (run 2026-07-19).** The follow-up named in
> `docs/bar_vendor_divergence.md` §6, mechanized: searches nothing, changes
> no trial code path, appends nothing to the trials ledger, moves no
> ratified statistic. Instrument:
> `research/scripts/spin_adjustment_sweep.py` (tests in
> `tests/test_spin_adjustment_sweep.py`; the network seam is an injectable
> session, so every classification runs offline). Evidence:
> `results/spin_adjustment_sweep_2026-07-19.json`. The raw action records
> (`results/alpaca_corporate_actions_2020-01-02_2026-08-14.json`) were
> untracked in the v0.3.4 fat-cull — 136k lines of re-fetchable vendor
> payload, not a diagnostic result; the file stays on the operator box and in
> the nightly `prism-artifacts` sync, sha256-pinned in §7.
> No remediation is decided here; that is an owner call.

## 1. Question

`docs/bar_vendor_divergence.md` §6 found the certified spine's spin-off
adjustment **non-uniform**: APTV (VGNT spin, ex 2026-04-01) sits raw in the
certified basis — the spine's own cross-event return is −1,058 bps, the
mechanical distribution step — while BDX, CMCSA, FDX, and FTV are
back-adjusted. Certified B1 momentum ranks are price ratios over this spine
(`close[t−21]/close[t−252] − 1`), so **every raw step inside a 252-bar
lookback is a rank distortion the certified side itself carries**, invisible
to any live-vs-certified comparison. That finding was one name, probed by
hand. This sweep asks the general question: for every name in the certified
spine panel and every corporate action touching it inside the panel window,
does the spine absorb the event (back-adjusted) or show the mechanical step
(raw)?

## 2. Method and sources

**Panel.** The certified run's own bar-cache panel
(`results/demotion_b1/config.json` symbols, 574 names, window effective
2020-01-02 → 2026-06-15, quarantine fallback included, zero missing caches
— the `beta_telemetry.load_panel_closes` loader).

**Actions.** Alpaca v1 corporate-actions endpoint (the §5 spinoff-mask
surface, paper key, $0), **full taxonomy** — no `types` filter, so the
tested set is exactly what the vendor returns: 19 paginated batch requests,
14 cross-batch duplicates dropped by vendor id. Local-record check first:
the only committed local corporate-action record is the cash-dividend file
(`results/alpaca_cash_dividends_2021-03-30_2026-06-12.json`); no local
spin/split record exists (no `runs/**/spinoff_mask_*.json` cache has been
written — the mask landed default-off), so tested types come from the live
fetch, cross-checked against the local record on its overlap: **7,519 of
7,519** local dividend ids reappear in the live response.

**Mechanical expected ratio.** Splits from the record
(`old_rate/new_rate`); stock dividends from `1/(1+rate)`; spin-offs from
`1 − (new_rate/source_rate) · child_debut / parent_prev`, both prices from
Alpaca `adjustment=raw` bars (6 requests, 54 symbols) — the spine's own
closes are unusable for the fraction because they may be back-adjusted by
the very convention under test, and a `split`-adjusted series would fold
later rebasings into it.

**Classification.** Observed = the spine's close-to-close ratio across the
ex-date (gaps ≤ 5 sessions bridged and counted), netted of the panel
cross-sectional median move over the same sessions. In log space the net
ratio is compared to two centers: 1.0 (**BACK_ADJUSTED**) and the
mechanical ratio (**RAW_STEP**, step reported in bps). The nearer center
wins if the mechanical step exceeds a 400 bps separability floor and the
winner sits within a 500 bps noise budget; every other outcome is
**INDETERMINATE with a named reason** — nothing is dropped (N7). Types with
no adjustment expectation under the certified basis are tallied, not
step-tested: cash dividends (11,079 records) because the certified basis is
price-return by design (I-7, `…_no_dividends`; the wedge is measured by
`research/scripts/dividend_wedge.py`), and mergers / name changes
(70 + 78 + 17 + 58 records) as identity or terminal events.

## 3. Results

101 events step-tested: **81 BACK_ADJUSTED, 1 RAW_STEP, 19 INDETERMINATE.**

| Type | events | BACK_ADJUSTED | RAW_STEP | INDETERMINATE |
|---|---|---|---|---|
| forward_split | 63 | 59 | 0 | 4 |
| reverse_split | 7 | 4 | 0 | 3 |
| spin_off | 30 | 17 | **1** | 12 |
| stock_dividend | 1 | 1 | 0 | 0 |

**Flagged RAW_STEP list (complete): APTV, spin-off VGNT, ex 2026-04-01,
step −1,058 bps** (mechanical −1,333 bps; parent raw pre-event close
69.445, VGNT debut 27.77, fraction 0.1333 — the §6 quotes, reproduced
independently by this instrument). The §6 anchor is reproduced within 1 bps
of the hand probe; the sweep found **no additional raw step anywhere in the
panel window**.

**The split dimension is uniform.** Zero raw split steps in 70 split
events. The four forward-split INDETERMINATEs (ODFL 2020-03-25, TSLA
2020-08-31, AIV 2020-12-15, TTD 2021-06-17) are large-idiosyncratic-move
split days: each observed net return sits 39–90 **percentage points** from
the raw-step hypothesis (raw is excluded) but beyond the 500 bps budget
from exactly 1.0, so the instrument reports nearest-BACK_ADJUSTED rather
than claiming it.

**Back-adjusted spin-offs (17):** PFE→VTRS 2020-11-17, DTE→DTM 2021-07-01,
IP→SLVM 2021-10-01, ADS→LYLT 2021-11-08, EXC→CEG 2022-02-02, GE→GEHC
2023-01-04, LH→FTRE 2023-07-03, BWA→PHIN 2023-07-05, DHR→VLTO 2023-10-02,
GE→GEV 2024-04-02, J→AMTM 2024-09-30, FTV→RAL 2025-06-30, HON→SOLS
2025-10-30, DD→Q 2025-11-03, CMCSA→VSNT 2026-01-05, BDX→WAT 2026-02-10,
FDX→FDXF 2026-06-01. This independently confirms every §6 attribution:
BDX, CMCSA, FDX, FTV back-adjusted; DD's and HON's own spins absorbed;
APTV alone raw.

**INDETERMINATE (19, all named).**

- *Below the 400 bps separability floor (9)* — even if raw, each is
  bounded below the floor by construction: SLG reverse adjustments +292
  (2021-01-21) and +306 bps (2022-01-24); spins O→ONL −292 (2021-11-15),
  ZBH→ZIMV −201 (2022-03-01), BDX→EMBC −229 (2022-04-01), ILMN→GRAL −255
  (2024-06-25), J→AMTM −103 (2025-05-16), XRX→XRXDW −345 (2026-02-09,
  twice — the vendor carries two distinct-id records for one event; both
  reported).
- *Matches neither hypothesis within the noise budget (7)*: the four split
  days above, plus SOLS 120:1 reverse ex 2020-11-02 — a **ticker-reuse
  artifact**: the record belongs to the prior SOLS lister, measured against
  the quarantined wrong-instrument series (the quarantine class,
  `data/quarantine/SOLS_…`) — and two spins, MMM→SOLV 2024-04-01 (net
  +672 bps: MMM rallied through its spin day) and WDC→SNDK 2025-02-24 (net
  −572 bps, 72 bps outside the budget; WDC is independently evidenced
  back-adjusted by §3's 8,648 bps spine-vs-IEX score read across this
  event). The thresholds were declared, not tuned; neither name is
  reclassified to fit.
- *Distribution value not determinable (3)*: AIV→AIRC.WI 2020-12-15
  (when-issued line, no IEX debut bar), FTI→THNPF 2021-02-16 (foreign OTC
  line), LEN→MRP 2025-01-21 (no MRP raw close within 10 days of the
  ex-date).

**Counted, not classified:** two spin parents outside the panel (T→WBD
2022-04-11; FBIN→MBC 2022-12-15) and one malformed `unit_splits` record
(the PANW/CYBR merger consideration, no `symbol`/`ex_date` fields — a
merger in split clothing). The FBIN record is a genuine coverage gap: the
panel holds that series under its pre-rename ticker FBHS, so the FBHS
series' 2022-12-15 cross-event was **not** tested against an action record
(the sweep does not resolve rename chains).

## 4. Consequence for B1 rank integrity

Factually: **exactly one certified name carries a verified raw spin step —
APTV, −1,058 bps at 2026-04-01.** Every B1 momentum score whose 252-bar
lookback straddles that date prices APTV off a series containing a −10.6%
non-economic step, understating its score by roughly the step; within the
certified window that is every refresh from 2026-04-01 to the 2026-06-16
end, and on the frozen spine any replay decision until the event clears the
lookback (~2027-04). Whether the distortion flips decile membership at a
given refresh is a boundary question measured per refresh by the §3
instrument, not here. On the live side the §5 mask already renders APTV
unrankable while the event sits in the trailing window (it is in the
live-verified flag set). The remaining exposure is bounded and named: the
nine sub-floor events distort by under 400 bps each even in the worst case;
the three undeterminable distributions (ex-dates 2020-12-15, 2021-02-16,
2025-01-21) are unknowns that certified refreshes within 252 bars after
those dates were exposed to — the latest cleared the lookback around
2026-01; MMM 2024-04-01 and WDC 2025-02-24 are nearest-back-adjusted with
independent corroboration for WDC; and the FBHS/FBIN rename gap leaves one
real 2022-12-15 spin on a certified series untested. What follows from any
of this is an owner decision; this record decides nothing.

## 5. Limits

Vendor coverage of Jan–Feb 2020 cannot be proven from absence (the
earliest returned ex-date is 2020-03-02); the action set is what the
endpoint returns, with no independent completeness check beyond the
dividend-record overlap (7,519/7,519) and the §6 hand probes (all
reproduced). The single `stock_dividends` record (RJF 2023-04-13, vendor
rate 1.3) classified BACK_ADJUSTED at an 85 bps observed move regardless of
rate semantics, which were taken at face value. The fetch window is padded
+60 days past the panel end because the vendor's window filter is not
purely ex-date (the committed dividend record contains ex-dates before its
requested start); window membership is enforced locally on ex-date. The
sweep is rerunnable offline against the saved records
(`--records_json results/alpaca_corporate_actions_2020-01-02_2026-08-14.json`
— untracked since the v0.3.4 fat-cull; the file remains on the operator box
and in the nightly `prism-artifacts` sync, and the instrument re-fetches it
from the vendor when absent; integrity pin sha256
`beeed3856ffeb8af82af5007500ee86e0b03e021ad592435008a486e432d345d`)
and rerunnable against any fresher cache vintage via `--config`.
