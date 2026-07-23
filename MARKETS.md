# MARKETS — zero-budget market structure analysis (mid-2026)

Companion to `SPEC.md`; the analysis behind the market-scope table in
`SPEC.md` §4. The lens throughout: a US individual, retail latency
(seconds to minutes), daily-to-weekly horizon, exactly $0 data and
infrastructure budget, in a regulatory environment in flux. Regulatory
claims are tagged **verified / likely / uncertain**; rule numbers and
thresholds are asserted only where a primary source was checked. Everything
is as of mid-2026 — re-confirm before it drives capital.

**Bottom line.** Two markets are executable at $0 with a viable retail
horizon — **US cash equities/ETFs** and **crypto spot majors**. Four more —
rates, FX, options/vol, futures — offer no $0 retail edge but emit free,
high-value *regime* data that conditions the core book. The recurring
structural fact: retail latency excludes the trader from every intraday
game, and the daily-to-weekly band is the only one where a zero-budget
actor is not structurally dead.

---

## 1. US cash equities & ETFs — **CORE**

**Structure.** Highly fragmented: ~16 lit exchanges, 30+ dark pools, and a
few wholesalers buying retail marketable flow via payment for order flow.
Off-exchange volume is roughly half of consolidated share volume (~45–55%,
*likely*). A retail systematic trader is structurally a liquidity taker
whose orders are internalized at or marginally inside the national best
bid/offer. Settlement is T+1 (since May 2024).

**Regulation.**
- **The pattern-day-trader rule is eliminated** — the SEC approved FINRA
  amendments removing the $25,000 minimum and the PDT designation,
  effective ~2026-06-04 with an ~18-month broker phase-in; a $2,000 margin
  minimum remains. *Verified (FINRA Regulatory Notice 26-10; reconfirm
  before relying).* This is a secondary tailwind, not the linchpin: Prism's
  next-open overnight-hold pattern never day-trades. Note the long-short
  book runs in a Reg-T margin account with locates — "$0 cash account"
  applies only to long-only contexts.
- **Reg NMS tick/access-fee amendments** (fee cap 30→10 mils, half-cent
  ticks for constrained names) were upheld in court (2025-10-14) but
  compliance is pushed to ~November 2026 — not yet in force, and possibly
  revised before it binds. *Verified; final parameters uncertain.* Monitor
  the effect on spreads before scaling size.
- Payment-for-order-flow reform appears deprioritized; treat the
  internalization status quo as intact. *Uncertain.*

**Data at $0.** Alpaca's free tier is IEX-only — measured at ~5% of
consolidated volume across the 2026 S&P 500 (per-name median 4.8%,
`results/iex_eligibility_2026-07-17.json`) — thin and not representative of
the true best bid/offer. Also free: historical daily/minute bars, Stooq
daily backfill, SEC EDGAR, FINRA transparency files. Real-time consolidated
data is paid. **The binding scarcity is data quality, not access.**

**Execution at $0.** Alpaca: free paper environment on the full API,
commission-free live equities/ETFs, $0 borrow on 5,000+ easy-to-borrow
names. Fills route through wholesalers at or near the NBBO — adverse
selection that is tolerable at daily/weekly horizons and fatal intraday.
IBKR is the fallback for borrow depth and routing.

**Verdict — CORE.** The canonical zero-budget systematic venue, and the
only market with enough names for the cross-sectional machinery to matter.
Minimum viable horizon ~1 trading day; at that horizon retail-size market
impact is effectively zero. The binding limitation is free-data fidelity,
which argues for daily-bar signals — not intraday alpha.

---

## 2. Crypto spot (BTC/ETH majors) — **CORE-CANDIDATE (time-series lane)**

**Structure.** Fragmented across venues with no consolidated tape and no
cross-venue price protection — the bot picks one home venue rather than
routing. BTC/ETH liquidity is deep (~1 bp top of book). Settlement is
internal-ledger and effectively instant against pre-funded balances; there
is no SIPC/FDIC, assets are custodied by the venue, and **counterparty
solvency is the dominant structural risk** (FTX precedent). Markets run
24/7 with no close and no circuit breakers.

**Regulation.** The CLARITY market-structure bill passed the House
(July 2025) and is on the Senate calendar, but is **not law** as of
July 2026 — the SEC/CFTC split is unresolved. *Verified.* Spot BTC/ETH are
treated in practice as commodities, making them the least legally contested
crypto assets regardless of the bill's fate. *Likely.*

**Data and execution at $0.** Genuinely production-grade: public REST and
WebSocket feeds with full book/trades/candles and no key (Binance.US;
Coinbase and Kraken with lower limits) — real-time, a major advantage over
equities' delayed feeds. Trading APIs carry no subscription; per-fill fees
drive venue choice: Binance.US ~0%/0.02% maker/taker (*verified*,
Apr 2026), Kraken ~0.25/0.40%, Coinbase ~0.40/0.60% (nearly disqualifying
at small size). No day-trading rule; instant settlement. **US retail spot
shorting is largely unavailable, so the book is long/flat only.** Sweep
profits off-venue; keep a minimal working float.

**Verdict — CORE-CANDIDATE, time-series lane, not cross-sectional core.**
Access is best-in-class, but two correlated majors give effectively one
independent bet — so crypto cannot use the residual/breadth machinery that
is the system's thesis and must carry its own net-edge evidence bar
(`SPEC.md` §7.1 time-series carve-out). Name one US execution venue and
price the book off *its* real fee; Alpaca crypto is a paper/validation
fallback, not the priced home. Minimum hold ~1 hour on the cheapest venue
(round-trip ~4–6 bp); the 24/7 clock demands a continuously monitored
kill switch. No live capital yet.

---

## 3. US Treasuries & rates — **SIGNAL_ONLY**

Dealer-centric OTC market, closed to a $0 retail bot: interdealer platforms
and dealer-to-client request-for-quote, with the SEC's central-clearing
mandate (cash by 2026-12-31, repo by 2027-06-30, dates holding as of
April 2026 — *verified/likely*) raising the access bar further. The one $0
path to rates *exposure* is duration ETFs on a commission-free equity
broker.

The value here is **the curve as regime state**, not tradable alpha: FRED
constant-maturity yields (daily, history to 1962), slope series, real and
breakeven rates, and Treasury.gov's daily par curve — all free, all EOD
with ~1-day lag. Level, slope, and curvature imprint the
recession/expansion and easing/tightening state and condition every other
sleeve. A duration-ETF sleeve would just be macro beta competing for the
same capital; it stays a documented, deferrable satellite.

---

## 4. Spot FX (G10 majors) — **SIGNAL_ONLY**

A decentralized dealer market — structurally the opposite of equities'
regulatory framework: no exchange, no consolidated tape, no statutory
best-execution mandate. "Last look" — the dealer's millisecond window to
reject a trade that moved against them — is pervasive and governed only by
a voluntary code. *Verified.* US retail leverage is capped at 50:1 majors /
20:1 minors, with FIFO and anti-hedging rules that break naive grid
strategies. *Verified.*

Data is genuinely free and clean (ECB reference rates; OANDA's practice API
as both feed and paper venue; Dukascopy historical tick exports). But a
retail bot has no microstructure edge, ~7–9 majors provide no
cross-sectional breadth, and dealer-marked-up rollover confiscates most of
the carry — the one robust daily-horizon FX return source. Wire the free
information as regime features instead: dollar regime, G10 carry and rate
differentials, JPY/CHF risk-off. No live FX book.

---

## 5. US listed options & the vol surface — **SIGNAL_ONLY**

The most venue-fragmented US market (~17–18 exchanges), with liquidity
almost entirely from designated market makers and much retail flow
internalized through auctions. Spreads are wide in percentage terms; a
retail spread-taker pays a structural liquidity tax every round trip. The
PDT elimination covers options too — a real unlock for small accounts
(*verified*) — but the binding wall is data: **the consolidated real-time
options feed (OPRA) needed to reconstruct a live implied-vol surface is not
free** (*likely — reconfirm the current fee schedule*).

What $0 does buy is first-class regime signal: VIX and VIX3M daily closes
on FRED (history to 1990), the VIX9D/VIX/VIX3M term-structure slope from
Cboe's pages, and realized vol from free underlying bars. The
term-structure slope and the realized-versus-implied spread are high-value
overlays for sizing and crisis de-grossing. Direct options trading is
deferred: the spread-plus-fee tax kills daily-horizon spread-taking, and
short-premium adds margin/assignment/tail complexity the stack is not
built to manage.

---

## 6. Futures & commodities (CME complex) — **SIGNAL_ONLY**

Structurally centralized — one order book per contract, professional
market-maker liquidity, central clearing with daily cash variation margin,
and fixed expiries that force a roll. Micro contracts are liquid enough
that retail size has negligible impact. But there is **no $0 execution
path**: every futures broker charges per-contract commission
(~$0.25–$2.00/side on micros plus fees), live trading needs $2–5k of
posted margin, and CME terminated its free settlement-data licenses in 2025
(*likely*), leaving all CME data free only on a 10-minute delay.

Trade the data, not the contracts: the entire forward curve is observable
at $0 (delayed), so contango/backwardation and roll yield feed the regime
layer, and any commodity/rates/index *exposure* worth holding is available
more cheaply through commission-free ETFs already in the equities lane.
Revisit only if capital grows past a few thousand dollars and a specific
carry/trend edge clears the commission floor in paper.

---

## 7. Cross-market synthesis

Which structural laws bind, by market:

- **The latency light-cone** binds everywhere by exclusion — it is why the
  horizon floor is ~1 day in every market. A fence, not an edge.
- **Market impact** is non-binding *in the retail trader's favor* in
  equities, crypto, and liquid ETFs (retail size ≈ zero impact) — the one
  structural advantage a small trader holds over institutions. It binds
  only as the spread/adverse-selection tax in options and FX.
- **The yield curve** is the dominant tradeable-signal law in rates, FX
  (carry), and futures (term structure) — exactly why those three are
  signal layers, not venues.
- **Breadth** only has cross-sectional bite in equities — the sole market
  with hundreds of names to residualize. Crypto, FX, and rates are
  low-rank, so the residual machinery earns its keep only on the equity
  cross-section. This is the structural reason equities is the core alpha
  venue and everything else is context.
- **Counterparty/custody risk** is the binding operational law in crypto;
  variation-margin path dependence plays the same role in futures.

**The $0 data spine** (full table in `SPEC.md` §4): one keyed pair —
Twelve Data + Alpaca — for the price/execution spine, plus unauthenticated
official sources (FRED, Treasury, Cboe, SEC EDGAR) for the regime layer.
One measured caveat: Twelve Data resolves retired/renamed tickers to
whatever instrument currently owns the symbol (8 of ~574 caches quarantined
2026-07; a standing pre-flight sweep exists —
`docs/data_integrity_diagnostic.md`). Fragile sources (yfinance,
Binance.com) are confined to quarantined cross-checks, never production
dependencies.

**Live uncertainties to monitor:** the ~Nov-2026 Reg NMS tick/fee change
and the crypto market-structure bill. Neither threatens the current plan —
equities' daily-horizon economics and spot BTC/ETH tradability survive
their most likely resolutions — but both should be re-checked before
scaling size.
