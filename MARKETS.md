# MARKETS — zero-budget market structure analysis (mid-2026)

Companion to `SPEC.md`. This is the qualitative and structural analysis behind
the market-scope table in `SPEC.md §4`. The lens throughout: a **US individual,
retail latency (seconds-to-minutes), daily-to-weekly horizon, exactly $0 data
and infrastructure budget**, in a macro environment of regulatory flux. Every
regulatory claim is tagged **verified / likely / uncertain**; specific rule
numbers and thresholds are only asserted where a primary source was checked.
Treat everything as of mid-2026 and re-confirm before it drives capital.

**Bottom line.** Two markets are executable at $0 with a genuine retail-viable
horizon — **US cash equities/ETFs** and **crypto spot majors** — and are the
system's core execution venues. Four more — **rates, FX, options/vol, futures**
— have no $0 retail edge but emit free, high-value *regime* data that conditions
the core book. The recurring structural fact is that retail latency exiles the
trader from every intraday microstructure game, and the daily-to-weekly band is
the only one where a zero-budget actor is not structurally dead.

---

## 1. US cash equities & ETFs — **CORE**

**Structure.** Highly fragmented: ~16 lit exchanges, 30+ ATS/dark pools, and a
few wholesale internalizers (Citadel Securities, Virtu, Jane Street) buying
retail marketable flow via payment-for-order-flow. Off-exchange (TRF) volume is
roughly half of consolidated share volume (~45–55%, *likely*); wholesaler
internalization of retail flow is the single largest bucket. A retail systematic
trader is structurally a **liquidity taker** whose marketable orders are
internalized at or marginally inside the NBBO. Consolidated quotes flow through
the SIPs; full depth and the fastest signals live in paid exchange proprietary
feeds. Settlement is **T+1** (since May 2024); no T+0 mandate is scheduled.

**Regulatory shifts that matter.**
- **PDT rule eliminated** — the SEC approved FINRA amendments removing the
  $25,000 minimum-equity requirement and the "pattern day trader" designation,
  reported effective ~2026-06-04 with an ~18-month phase-in (**brokers on their
  own schedule through ~2027**), replacing end-of-day day-trading buying-power
  limits with real-time intraday margin excess (a $2,000 margin minimum remains).
  *Verified (FINRA Regulatory Notice 26-10; broker notices — reconfirm before
  relying).* A **secondary** tailwind, not the linchpin: Prism's N2 next-open
  overnight-hold pattern never day-trades the same security intraday, so PDT was
  not its binding constraint. Note the residual book runs in a **Reg-T margin
  account with locates** (it shorts) — the "$0 cash account" applies only to
  long-only / crypto-long-flat contexts.
- **Reg NMS tick/access-fee amendments** (Rules 610/612: access-fee cap 30→10
  mils, sub-penny/half-cent ticks for tick-constrained names, new round-lot and
  odd-lot dissemination) were upheld by the D.C. Circuit (2025-10-14) but
  compliance is pushed to ~first business day of November 2026 — **not yet in
  force** as of July 2026, and possibly revised before it binds. *Verified;
  final parameters/date uncertain.*
- PFOF/order-competition and Reg Best-Ex proposals appear deprioritized under the
  current SEC; treat the internalization status quo as intact. *Uncertain.*

**Data at $0.** Alpaca Basic real-time = **IEX only** (measured ~5% of
consolidated volume on the 2026 S&P 500 book — per-name median 4.8%, p5–p95
3.2–6.6%, `results/iex_eligibility_2026-07-17.json` — thin,
non-representative of true NBBO); historical minute/daily bars available;
~200 req/min. yfinance (EOD + delayed intraday, ToS-gray, fragile),
Stooq (clean daily EOD backfill), SEC EDGAR (fundamentals, free, full history),
FINRA transparency files (delayed short-sale/ATS volume), Ken French factor
library. Real-time SIP consolidated data requires a paid tier — **the binding
scarcity is data quality, not access**.

**Execution at $0.** Alpaca: free paper environment exercising the full API,
commission-free live equities/ETFs, $0 to open a cash account, $0 borrow on
5,000+ easy-to-borrow names with programmatic locate support. PDT no longer
enforced. Fills routed through PFOF wholesalers at/near NBBO — adverse selection,
tolerable daily/weekly, fatal intraday. IBKR is the fallback for borrow depth and
routing quality.

**Minimum viable horizon.** ~1 trading day; comfortable at days-to-weeks.
Intraday is structurally dead (retail latency + IEX-only/delayed data + PFOF
fills = always adversely selected inside someone's light-cone). At daily horizon,
cross-sectional residual alpha decays over days so execution lag is negligible,
and retail-notional market impact is ~zero — the one microstructure law that
normally taxes you does not bind.

**Verdict — CORE.** The canonical zero-budget systematic venue; access barriers
that historically killed small systematic accounts just fell away, and the
existing stack is already residual stat-arb on equities. Binding limitation is
free-data fidelity (argues for daily-bar signals executed intraday, not intraday
alpha). Monitor the ~Nov-2026 Reg NMS tick/fee change to spreads and rebates.

---

## 2. Crypto spot (BTC/ETH majors) — **CORE**

**Structure.** Fragmented across dozens of global CEXs and on-chain DEXs, but a
US individual realistically uses US-domiciled venues (Coinbase, Kraken,
Binance.US, Gemini). Each runs its own CLOB with **no cross-venue NBBO, no
consolidated tape, no Reg-NMS protection** — the bot picks a home venue rather
than routes. Liquidity on BTC/ETH majors is deep (~1 bp top-of-book, books absorb
retail clips with no measurable impact). Settlement is internal-ledger and
effectively instant against pre-funded balances — **no T+1, no clearinghouse, no
SIPC/FDIC**. Assets are custodied by the venue; rehypothecation/commingling are
possible; **counterparty solvency is the dominant structural risk** (FTX
precedent). Markets run 24/7/365 with no close and no circuit breakers.

**Regulatory shifts that matter.**
- **CLARITY Act** (H.R.3633, digital-asset market structure) passed the House
  (July 2025, 294–134), advanced by Senate Banking Committee (~15–9, May 2026),
  placed on the Senate calendar (~June 2026), but **not signed into law** as of
  July 2026 — the SEC/CFTC jurisdictional split is unresolved. *Verified.*
- SEC permitted in-kind creation/redemption for spot BTC/ETH ETPs, tightening the
  ETF-to-spot arbitrage loop. *Verified.* A joint SEC/CFTC release (~2026-03-17)
  classified staking rewards as non-securities across ~16 digital commodities
  incl. ETH; staked-ETH ETFs launched ~March 2026. *Likely.*
- **Spot BTC/ETH are treated in practice as commodities** (CFTC-leaning), making
  them the least legally contested crypto assets regardless of CLARITY's fate.
  *Likely.*

**Data at $0.** Genuinely production-grade: Binance/Binance.US public REST +
WebSocket give full L2 book/trades/klines with **no API key** (weight-based
limits; WebSocket depth/trade streams effectively real-time). Coinbase Advanced
(~10 req/s public, free WebSocket), Kraken (≤1 req/s public). History depth per
request is capped (~720–1000 candles) requiring pagination; deep tick/L2 history
is not free, but free daily-weekly bars are ample. **Public feeds are real-time
— a major advantage over equities' delayed SIP.**

**Execution at $0.** All three US venues expose trading APIs with no subscription
fee — you pay only per-fill trading fees, and those drive venue choice: Binance.US
~0% maker / 0.02% taker (near-zero, announced Apr 2026, *verified*), Kraken Pro
~0.25/0.40%, Coinbase Advanced ~0.40/0.60% (nearly disqualifying at small size).
**No PDT rule** (equities-only); unlimited round-trips; instant settlement.
Constraint: **spot shorting needs margin, largely unavailable to US retail** — so
the bot is **long/flat only** (park in USD/USDC). Custody is the binding
operational risk — sweep profits off-venue, keep minimal working float.

**Minimum viable horizon.** ~1 hour floor; daily-to-weekly comfortable. On the
cheapest venue round-trip is ~4–6 bp incl. spread, so any hold over ~1h clears
cost for signals with >~10 bp edge. Below ~1 minute you enter the light-cone
against co-located makers and are dead. The 24/7 clock never stops — the loop
must run continuously with a monitored kill-switch and overnight/weekend exposure
controls rather than relying on a market close.

**Verdict — CORE-CANDIDATE (time-series lane), not cross-sectional core.**
Uniquely delivers production-grade real-time data *and* trading API at $0 (fees
only), near-zero execution cost on the right venue, deep liquidity, no PDT, 24/7 —
so *access* is best-in-class. **But** FLAM/breadth is thin (two correlated majors
≈ rank-1, $N_{\text{eff}}\approx1$): crypto is a **time-series** book, not a
cross-sectional one, so it cannot use the residual/breadth machinery that is the
rest of the system's thesis and must carry its **own** `net_edge` evidence bar
(`SPEC.md §7.1` time-series carve-out). Name **one** US execution venue and price
the book off *its* real fee — Binance.US (~2 bp taker) or Coinbase (higher);
Alpaca crypto is a paper/validation + bars fallback, not the priced home. Trade
long/flat (US retail spot shorting restricted — this forecloses BTC/ETH
relative-value), counterparty risk capped by design. No live capital in v0.3.0.

---

## 3. US Treasuries & rates — **SIGNAL_ONLY**

**Structure.** Dealer-centric OTC, not an exchange. Two tiers: interdealer CLOBs
(BrokerTec/CME, Dealerweb, Fenics) in on-the-run issues, and dealer-to-client RFQ
(Tradeweb, Bloomberg, MarketAxess) for the ~500-CUSIP long tail. Fragmentation is
by venue-type, not competing exchanges. Cash settlement is T+1 book-entry over
Fedwire. The liquid listed complement is CME Treasury futures (ZT/ZF/ZN/US/UB),
centrally cleared — where much price discovery and hedging actually occurs.

**Regulatory shifts that matter.**
- **SEC Treasury central-clearing mandate** (adopted Dec 2023): eligible
  secondary cash and repo transactions must centrally clear. *Verified.*
- Compliance **extended ~1 year** (Feb/Mar 2025): eligible cash by **2026-12-31**,
  repo by **2027-06-30**; as of the April 2026 Uyeda update these dates remain in
  effect with no further blanket extension signaled. *Verified / likely.* FICC,
  CME Securities Clearing, and ICE Clear Credit are the competing CCPs. *Likely.*
- **Implication for retail: the mandate raises the access bar, not lowers it** —
  direct cash Treasuries remain closed to a $0 retail bot.

**Data at $0.** First-class and deep: FRED constant-maturity yields
(`DGS1MO…DGS30`, daily, history to 1962, ~120 req/min, ~1-day lag), FRED slope
series (`T10Y2Y`, `T10Y3M`), real/breakeven (`DFII10`, `T10YIE`), `SOFR`, `DFF`;
Treasury.gov Daily Par Yield Curve (CMT, XML/CSV, **no key**, EOD) — the canonical
level/slope/curvature source. Rate-ETF history (SHY/IEF/TLT/GOVT) free via Alpaca
/ Stooq. Real-time cash-Treasury quotes and live futures data are paid/entitled.

**Execution at $0.** Direct cash Treasuries and futures: none at $0 (dealer/FICC
access, RFQ/voice, institutional minimums; futures need funded margin + paid
data). The one $0 path to rates exposure is **duration ETFs** on a commission-free
equity broker (BIL/SHV ultra-short, SHY front, IEI/IEF belly, TLT long, GOVT
broad; curve trades as ETF pairs). PDT non-binding at daily horizon.

**Minimum viable horizon.** ~1 day, comfortable multi-day/weekly; rate ETFs are
among the most liquid ETFs so retail latency is irrelevant. The ~1-day FRED/
Treasury data lag naturally floors the horizon at daily.

**Verdict — SIGNAL_ONLY.** The distinctive value is **the curve as regime
state**, not tradable alpha. Level/slope/curvature (3M, 2s10s, 2s5s10s butterfly)
imprint recession/expansion, easing/tightening, risk-on/off — free, 60+ years,
and should condition every other sleeve. A duration/curve ETF sleeve is just
macro beta competing for the same capital; keep it a documented, deferrable
satellite, not core alpha.

---

## 4. Spot FX (G10 majors) — **SIGNAL_ONLY**

**Structure.** Decentralized OTC dealer market — structurally the *opposite* of
Reg NMS: **no exchange, no NBBO, no consolidated tape, no legal best-execution
mandate**. Tiered liquidity (interdealer EBS/Refinitiv → ECNs → retail RFED/FCM
dealers who internalize/B-book). Every venue streams its own price, so there is no
single "the price." **Last look** — the LP's millisecond window to reject a trade
that moved against them — is pervasive, asymmetric against fast/informed flow, and
governed only by the *voluntary* FX Global Code, not law. Interbank spot settles
T+2 via CLS; **retail leveraged FX never settles** — positions are marked and
rolled nightly with a financing swap carrying a dealer markup.

**Regulatory shifts that matter.**
- US retail FX leverage capped **50:1 majors / 20:1 minors** (CFTC/NFA); no 2026
  change surfaced. *Verified.* **NFA FIFO + anti-hedging** rules break naive
  grid/hedged strategies and force net-position bookkeeping. *Verified.*
- FX Global Code last updated Dec 2024; still voluntary — **retail has no
  statutory last-look protection**. *Verified.* No US spot-FX consolidated tape is
  proposed. *Likely.*

**Data at $0.** Genuinely free and clean: ECB euro reference rates (one EOD
fixing, no key), OANDA v20 practice API (real streaming + candles, indefinitely
free after KYC — simultaneously data feed and paper execution venue), Twelve Data
(FX daily + intraday), Dukascopy free historical tick/bar export (deep, for
backtest corpus). Alpha Vantage too throttled (~25/day).

**Execution at $0.** Paper is fully free/unlimited (OANDA v20 demo). Live needs a
funded RFED account (no free live path). Structural pluses: **no PDT** (equities
concept), **symmetric shorting**, tiny minimums. The unavoidable cost is not
commission or impact (majors far too deep for retail to move) but the **spread +
last-look adverse selection**, and the **marked-up rollover confiscates most of
the carry** — the one robust daily-horizon FX return source.

**Minimum viable horizon.** ~1 day minimum, multi-day comfortable; retail latency
exiles you from the intraday band that bank algos/HFT own, and last look taxes
exactly the fast orders a short-horizon strategy sends. At daily+ the ~1-pip
round-trip amortizes and the tradeable signal (carry, trend, dollar regime) is
slow enough that latency is irrelevant.

**Verdict — SIGNAL_ONLY.** The most efficient, most last-look-protected market in
existence; a zero-budget retail bot has no microstructure edge and only ~7–9
majors (no cross-sectional breadth for the residual machinery this stack has).
But FX information is free and high-value: **dollar regime (DXY/DTWEXBGS), G10
carry / rate differentials, JPY/CHF risk-off** — wire as regime features, keep an
OANDA paper loop as a cheap future option, do not allocate a live FX book.

---

## 5. US listed options & the vol surface — **SIGNAL_ONLY**

**Structure.** ~17–18 exchanges (Cboe, Nasdaq/PHLX/ISE, NYSE, MIAX, BOX) — more
venue-fragmented than equities, linked by the Options Order Protection plan.
Liquidity is almost entirely designated/lead market makers; much retail flow is
internalized/auctioned (Cboe AIM, PHLX PIXL, NYSE CUBE). Quotes consolidate
through **OPRA**, whose real-time full feed is **the core cost barrier**. Clearing
is monopolized by OCC; premium and cash-settled index options settle T+1. Spreads
are wide in percentage terms and depth thin away from the most active names —
a retail spread-taker pays a structural liquidity tax every round trip.

**Regulatory shifts that matter.**
- PDT eliminated (FINRA 26-10, ~2026-06-04, 18-month phase-in) covers equity
  options through member brokers — a real unlock for sub-$25k accounts. *Verified.*
- SEC Options Market Structure Roundtable (2026-04-16) signals incremental reform
  (routing/allocation, specialist "five-lot" guarantees), nothing adopted yet.
  *Verified.* Nasdaq ORF change (2026-01-02) is sub-cent-to-low-cents per
  contract but nonzero on every fill. *Likely.*
- **OPRA real-time display fees** (~$1.25/mo non-pro, ~$31.50/mo pro per feed,
  plus redistributor fees); delayed/historical (>~15 min) is exempt. *Likely —
  reconfirm current OPRA schedule.*

**Data at $0.** FRED `VIXCLS` (VIX) and `VXVCLS` (VIX3M) daily close (full history
to 1990, ~1-session lag); Cboe index pages for VIX/VIX9D/VIX3M/VIX6M (VIX9D is
Cboe-only, not on FRED; delayed ~15–20 min + free EOD CSVs); realized vol from
free underlying OHLC; broker delayed chains (Alpaca ~15-min, Tradier/IBKR gated).
**What $0 cannot buy: the consolidated real-time OPRA feed to reconstruct a live
full-chain implied-vol surface.** That is the binding wall.

**Execution at $0.** Commission-free options exist (Alpaca, etc.) but every fill
carries pass-through fees (ORF, OCC ~$0.02, exchange) plus the real cost of
crossing a wide spread twice. Selling vol needs options approval + margin;
defined-risk spreads work in a small account, naked shorts need higher tiers.

**Minimum viable horizon.** ~1 week+ for directly trading options at $0 — the
round-trip spread is often 2–10%+ of premium and a retail trader is a pure
spread-taker, so intraday/single-day directional options are structurally dead.
As a **regime signal** the surface is usable at daily granularity (EOD series with
a one-session lag are fine).

**Verdict — SIGNAL_ONLY.** A first-class free regime signal, a poor $0 execution
venue. **VIX level, the VIX9D/VIX/VIX3M term-structure slope (backwardation vs
contango), and the realized-vs-implied spread** are all $0 and high-value overlays
for sizing/gating/crisis de-grossing the equity book. Direct options trading is
deferred: no affordable real-time surface, spread+fee tax kills daily-horizon
spread-taking, short-premium adds margin/assignment/tail complexity the stack is
not built to manage.

---

## 6. Futures & commodities (CME complex) — **SIGNAL_ONLY**

**Structure.** Structurally **centralized**, not fragmented: essentially all
liquidity for a contract sits on one designated market (CME/CBOT/NYMEX/COMEX on
Globex; ICE for some energy/ag) — one CLOB per contract, no NBBO fragmentation, no
maker-taker rebate ecosystem. Liquidity is professional HFT/PTF market makers.
Clearing is centralized (CME Clearing) with **daily mark-to-market**: variation
margin moves in cash every session (functionally T+0/T+1 cash). Fixed expiries
force a roll → term-structure/roll cost. **Micro contracts** (MES, MNQ, MGC, MCL,
micro Treasury-yield) are liquid enough that retail size has negligible impact.
Nearly 24/6.

**Regulatory / data-licensing shifts that matter.**
- **CME terminated free EOD/settlement data licenses** (2025), reclassifying
  settlement data as "delayed" (free only after ~8h) and consolidating non-display
  licenses, with retroactive billing from ~June 2025. *Likely.* CME's Jan 2026 fee
  list bundles delayed/historical into a real-time license for direct licensees
  (not via Bloomberg/LSEG intermediaries). *Likely.* Retail non-pro real-time
  top-of-book is cheap (~$7/mo promo on TradingView) but not $0; all CME data is
  free on a **10-minute delay**. *Likely.*
- CFTC prediction-markets NPRM (RIN 3038-AF65, June 2026, comments due 2026-07-27)
  is adjacent to event contracts, tangential to core commodity/rates/index
  futures. *Verified.*

**Data at $0.** CME 10-minute delayed quotes (cmegroup.com, TradingView free) —
**the entire forward curve is observable at $0, just not real-time**, so
contango/backwardation and roll yield are visible; Yahoo continuous futures
(ES=F, CL=F, GC=F — EOD, unofficial); TurtleTrader historical archive (back to
1970s); EIA (energy) and USDA (ags) fundamentals; broker demo feeds (time-boxed).

**Execution at $0.** **None.** Unlike equities, no futures broker offers
commission-free trading — every fill costs ~$0.25–$2.00/side on micros plus
exchange/NFA fees, a hard floor that eats daily-to-weekly edges on small size;
live also needs ~$2–5k posted margin capital. Two $0 upsides: paper/sim is free
and realistic (NinjaTrader/Tradovate/IBKR paper), and there is **no PDT and no
$25k minimum** in futures. But the commission floor + margin capital make direct
futures execution non-viable at zero budget.

**Minimum viable horizon.** ~1 day+, real edge at 1 week–months (carry/roll/trend
are latency-insensitive); intraday scalping dead. The commission floor forces
holds long enough that the edge dominates round-trip cost.

**Verdict — SIGNAL_ONLY.** *Trade the data, not the contracts.* Direct execution
has no $0 path and the exposure it offers (commodity, rates, equity-index) is
captured more cheaply via **commission-free ETFs** already in the equities lane
(broad commodity DBC/GSG/PDBC; oil USO/USL/BNO; metals GLD/SLV; rates SHY/IEF/TLT;
index SPY/QQQ/IWM), with tradeable term-structure proxies (USO/USL for oil carry,
VIXY/VIXM for the VIX curve). What futures uniquely give at $0 is the **forward
curve as a regime signal** feeding the yield-curve-state law. Revisit "core" only
if capital grows past a few thousand dollars and a specific carry/trend edge
clears the commission floor in paper.

---

## 7. Cross-market synthesis

**Which laws bind, by market.** The pattern is consistent and disciplining:

- **Latency light-cone** binds everywhere by *exclusion* — it is why the horizon
  floor is ~1 day in every market and why no intraday game is reachable. It is a
  fence, not an edge.
- **Sqrt-impact** is *non-binding in the retail trader's favor* in equities,
  crypto, and liquid futures/ETFs (retail size ≈ zero impact) — this is the one
  structural advantage the small trader holds over institutions. It binds only as
  the *spread/adverse-selection* tax in options and FX.
- **Yield-curve state** is the dominant *tradeable-signal* law in rates, FX
  (carry), and futures (term structure) — which is exactly why those three are
  signal layers, not execution venues.
- **Diffusion/vol** governs sizing everywhere and is the object of interest in
  options (the surface as regime).
- **FLAM/breadth** only has cross-sectional bite in **equities** — the sole market
  with hundreds of names to residualize. Crypto, FX, rates are low-rank
  (one-to-few factors), so the residual/RMT machinery earns its keep only on the
  equity cross-section. This is the structural reason equities is the core alpha
  venue and everything else is context.
- **Capital conservation** binds as *counterparty/custody risk* in crypto (no
  SIPC, rehypothecation) and as *variation-margin path dependence* in futures —
  operational-controls problems, not alpha.

**The $0 data spine** (full table in `SPEC.md §4`): one keyed reliable pair
(Twelve Data + Alpaca) for the price/execution spine, plus unauthenticated
official sources (FRED, Treasury, CBOE, SEC EDGAR) for macro/rates/vol/liquidity
regime — the fragile-source exposure (yfinance, Binance.com) is confined to
quarantined cross-checks, never a production dependency. Alpaca is the
highest-leverage single account: free equity + crypto data *and* the $0
commission-free execution venue for both core markets.

**Regulatory-flux takeaways for the build.** The 2026 PDT repeal is a genuine
tailwind (daily rebalancing under $25k is now unconstrained in equities and
options). The Reg NMS tick/fee change (~Nov 2026) and the crypto market-structure
bill (CLARITY, not yet law) are the two live uncertainties to monitor; neither
threatens the v0.3.0 plan (equities daily-horizon economics and spot-BTC/ETH
tradability both survive their most likely resolutions), but both should be
re-checked before scaling size. The Treasury clearing mandate (2026-12/2027-06)
only reinforces that rates stays a signal layer, not an execution venue.
