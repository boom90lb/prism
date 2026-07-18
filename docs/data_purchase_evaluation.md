# A2 data-purchase evaluation — survivorship-complete US equity history (decision memo)

**Status: procurement memo, recorded 2026-07-17. Not a pre-registration; no
statistic moves here.** The A2 amendment (`docs/amendments_2026-07.md` §A2,
ratified with the 2026-07 push of the amendment commit) permits a named
dataset purchase when it closes a measured gap free tiers cannot, total
annual data spend stays ≤ $1,000, and provenance/license enter the SPEC §5
coverage-ledger discipline. The purchase itself is an owner act; this memo
exists so that act reduces to a decision. Nine vendors were researched
2026-07-17 by parallel agents, and every load-bearing price and
delisting-coverage claim of the viable candidates was adversarially
re-verified by fresh fetches the same day (verdicts below are from that
second pass; full citations in the session record).

## 1. What the dataset must serve

The consumer is `docs/replication_preregistration.md`: three counted cells
of frozen-B1 on (mid-cap, 2020–26), (large-cap, pre-2020 incl. 2008–09),
(mid-cap, pre-2020), plus the C0 cross-vendor gate. Requirements, in gate
order: (i) daily bars for delisted names' full histories (survivorship
completeness); (ii) point-in-time mid-cap universe construction — branch
(a) of the pre-registration's decision tree wants PIT S&P MidCap 400
membership, branch (b) falls back to cap bands from shares history;
(iii) history to the 1995-01 era floor; (iv) corporate actions
(splits/dividends/renames); (v) ≤ $1,000/yr total; (vi) a license
compatible with private single-user research and personal trading.

## 2. Verified field

| vendor / product | price (verified) | delisted history | PIT mid-cap universe | era floor | verdict |
|---|---|---|---|---|---|
| **Norgate US Stocks Platinum** | **$630 / 12 mo** (live cart, CONFIRMED) | yes, to 1990; delisted names keyed `SYM-YYYYMM` (CONFIRMED) | **yes — PIT S&P 400 membership from Jun-1991 inception**, per-day queryable via Python API | 1990 (Diamond: 1950, $787.50) | **pick** |
| Sharadar SEP / SFA bundle (Nasdaq Data Link) | $588/yr SEP; $948/yr SFA (plans API, CONFIRMED) | yes, to 1998; ACTIONS carries delist reasons + ticker changes, permaticker keys (CONFIRMED) | no S&P 400 table; cap bands need SFA's DAILY marketcap ($948 tier) | 1998 | runner-up |
| EODHD All World + constituents add-on | $199 + ~$360/yr (CONFIRMED) | tiered: delisted **before 2018 = "EOD only"** (CONFIRMED) | S&P 400 membership only ~12 yr deep | 30+ yr claimed, but see tiering | fails C2/C3 |
| Tiingo Power | $300/yr (CONFIRMED) | only until a ticker is **recycled** (own docs, CONFIRMED) | none | 1962 claimed | fails (i)(ii) |
| Polygon/Massive Developer | $948/yr (CONFIRMED) | yes, point-in-time model (CONFIRMED) | none | **10 yr (~2016+)** at this tier | fails (iii) |
| FirstRate Stocks Complete | $499.95 one-time (CONFIRMED) | yes, 7,000+ delisted, 2000+ (CONFIRMED) | none — no membership or shares data | 2000 | fails (ii) |
| Algoseek (ADX daily) | $660–$1,224/yr | delisted coverage explicit only on out-of-budget tiers | S&P 400 not offered | 2007 | not viable ≤ $1k |
| FMP Premium | $708/yr (CONFIRMED) | own FAQ: delisted price history for "select US companies" | S&P 500 only | 30 yr claimed | self-refuted (i) |
| QuantRocket bundled US prices | login-walled, UNVERIFIED | yes (explicit) | none documented | 2007 | fails (iii), price unknown |

## 3. The pick: Norgate Data, US Stocks Platinum, $630/12 months

The only candidate that clears every gate, and the only one offering PIT
S&P MidCap 400 membership at all — which resolves the pre-registration's
universe decision tree to branch (a) outright (no cap-band construction,
no shares-history dependency). Delisted securities to 1990 covers the
1995-01 era floor with a five-year formation runway; membership from the
index's June-1991 inception covers C3. Corporate-action handling includes
total-return adjustment modes (useful for the dividend-wedge lens, though
the certified ledger remains price-return per I-7). Rename handling
consolidates history under the final ticker (`AOL-201506` pattern), which
matches the RENAME_TABLE discipline. Price verified on the live cart at
$630.00/12 mo — 37% under the A2 ceiling. Access is the vendor's NDU
updater on a Windows host with the `norgatedata` package inside WSL — the
vendor's own documentation describes exactly this repo's operating
arrangement.

**Known deltas, recorded before purchase:** no historical
shares-outstanding/market-cap *series* (current values only — irrelevant
under branch (a), disqualifying only for branch (b) uses); the
delisting-*reason* question is **settled negatively** (owner inspection of
the published data-content tables, 2026-07-17): the metadata carries
`LastQuotedDate`/`SecondLastQuotedDate` and no reason field anywhere, so
merger-vs-deficiency classification sources from the independent
EDGAR/Alpha-Vantage canon (`prism-observatory`; replication pre-reg §3
gate 5), and the vendor's "delisted" means untradeable on any venue it
tracks — OTC-relegated names remain *currently listed*, so index removal
while trading is a universe exit, not a terminal event (pinned in the
pre-registration's terminal-bar rule); subscriptions are non-cancellable and
non-refundable; ingestion runs through the proprietary NDU application
(no raw bulk files), so the replication's ingestion adapter targets the
Python API. A 3-week free trial at Platinum features (2-year history
cap) permits validating the ingestion path and the §3 data-gate tooling
before money moves — recommended sequencing: open the trial, run the
delisting-canon spot checks that fit inside two years of history, then
purchase.

**Not recommended:** Diamond (+$157.50 for history to 1950) — no
registered cell reads before the 1995-01 floor; unspent budget is not a
reason to buy history the pre-registration cannot use. Sharadar SFA
($948) is the runner-up if Norgate's trial fails validation: richer
relational model (permaticker, machine-readable delist reasons) but no
S&P 400 table (forces branch (b)) and a 1998 floor that truncates C2/C3
formation windows.

## 4. Coverage-ledger entry (drafted for the purchase, per A2(iii))

- **Provenance:** Norgate Investor Services (norgatedata.com), US Stocks
  Platinum subscription; delivery via Norgate Data Updater + `norgatedata`
  PyPI package; purchased under the A2 amendment.
- **License:** EULA — one individual natural person; two personal
  computers; no redistribution "in any way or form"; personal
  investment/trading use permitted; commercial use and use "as the basis
  for a financial instrument" excluded; non-refundable. Consequence for
  claim packets: committed artifacts may carry derived aggregates and
  statistics, never redistributable raw vendor series.
- **Spend:** $630.00 of the ≤ $1,000 annual A2 budget; remainder $370
  unallocated.

## 5. What this memo does not do

It buys nothing (owner act), registers nothing, and moves no adjudication.
The replication pre-registration's data gates (`docs/replication_preregistration.md`
§3) bind regardless of vendor; a Norgate panel that fails them is
remediated or returned to this memo's runner-up, and nothing is spent
from any trial budget either way.
