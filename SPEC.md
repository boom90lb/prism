# SPEC — Prism: a cross-sectional systematic trading engine

**Status:** Ratified for v0.3.0. This document is the constitution. Code, docs,
and roadmap defer to it; where they disagree with it, they are wrong. It
supersedes the "ensemble of forecasters" framing of v0.1–v0.2.

**Audience:** the operator (senior quant, solo, zero external budget) and any
agent extending the system. It assumes AFML-level familiarity with purged CV,
DSR/PBO, market-neutral construction, and market microstructure. It does not
re-teach them.

**Codename.** The system is *Prism* — the lecture's own metaphor: hold the
market up to a prism and decompose returns across frequencies and into
factor / residual / regime components. The importable package is `prism` under
src-layout (`src/prism/`, shipped in v0.3.0 — §11 item 8). The Python
distribution and the GitHub repository were renamed to `prism` in the single
publication-time identity break, 2026-07-05 (§12). Where this spec says "the
engine" it means Prism.

---

## 0. Why this rewrite exists

v0.2.2 is an unusually *honest research backtester* wrapped around a *weak and
mis-abstracted alpha*. The evaluation harness (purged walk-forward, next-open
costed fills, DSR/PBO, claim packets, leakage tests) is genuinely good and is
kept almost intact. The thing it evaluates — a per-symbol ensemble of
forecasters and reinforcement-learning policies — is the wrong object:

1. **The ensemble is not a portfolio engine.** `EnsembleModel` consumes one
   symbol's series and emits one position; there is no cross-name ranking,
   covariance, netting, or risk budget anywhere inside it. It can only ever be
   *one signal node* invoked once per name, with all portfolio construction
   built outside it. "Ensemble of models" is therefore a limiting abstraction,
   confirmed by audit — it names the least load-bearing layer as if it were the
   system.
2. **The binding constraint is breadth and cost, not member quality.** The
   fundamental law of active management ($IR \approx IC\sqrt{BR}$, correlation
   ceiling $IC/\sqrt{\rho}$) says a 4–6 name daily book cannot be made
   competitive by improving its parts. The project's own history confirms this:
   the megacap directional path is thin; the survivorship-free S&P residual
   slice reaches raw breadth (≈487 names/fold — but *effective* breadth
   $N_{\text{eff}}$ is far smaller once factor correlation is counted, N6) and
   finds a *real gross edge* (gross **daily** Sharpe ≈ +0.23) that **nets
   negative** (≈ −0.65 daily-annualized) after costs. A grid-searched
   cost-aware no-trade band lifts net only to ≈ −0.01 — **still net-negative,
   un-deflated (one in-sample optimum), and it raises *gross* to ≈ +0.42 as
   well**, so the band is acting as an in-sample selection filter, not a pure
   cost reducer. **Net-positive has never been reached.** At the operator's
   actual scale (~$2k–$10k) the binding cost is **turnover × effective spread**,
   not market impact. The system is **cost-bound before it is signal-bound** —
   and may yet prove *signal-dead* at daily frequency, which is why §10 carries a
   numeric kill-criterion, not only a promotion gate.
3. **Half the machinery is research-only ballast in the production path.** The
   RL trio (JAX/`flax.nnx`, path-memorizing, hardcoded-loss lineage), the batch
   `prism/scripts/` WFO orchestration, and even the feature scalers gratuitously
   wrapped in `nnx.Module` drag `jax[cuda12]` + `torch` + `prophet/cmdstan`
   into any import of the alpha path. A lean daily loop must not pay that.

Prism keeps the honest harness, retires the ensemble-as-system abstraction, and
re-founds the codebase on the one architecture into which both the directional
and stat-arb paths converge: **score → residualize → construct → execute**,
conditioned by a **regime** layer and gated by the **harness**.

---

## 1. Mandate and non-negotiables

**Mandate.** A production-grade, single-operator, zero-data-budget systematic
trading bot that trades a **cross-section** at a **daily-to-weekly** horizon,
survives on **free data tiers**, and only ever risks capital on edges that have
cleared an explicit, deflation-adjusted, cost-aware evidence bar.

**Non-negotiables (invariants that gate every merge):**

- **N1 · Point-in-time causality.** Every value attributed to bar *t* uses only
  information available at *t*'s close. No across-bar ffill, no full-series
  statistics fit before splitting, no future row reachable from a decision row.
- **N2 · Next-open fills, never same-bar.** A target decided at close *t* fills
  at open *t+1* or later. PnL accrues only after the fill. This holds in
  backtest *and* live.
- **N3 · Costs are charged before any edge is claimed.** Half-spread + impact +
  commission + borrow + (where panels exist) sqrt-ADV participation impact, on a
  fold-aligned net equity curve. A "gross" number is a diagnostic, never a claim.
- **N4 · The ledger conserves capital.** Equity moves only by realized PnL minus
  charged costs; no cash is created or destroyed across a rebalance. This is the
  single defensible residue of the "capital conservation law". Enforced twice:
  the accounting *algebra* is machine-checked in Lean 4 over an exact-integer
  kernel (`formal/PrismFormal/Ledger.lean` — `rebalance_conserves`,
  `run_conserves`), and the float implementation is property-tested against it
  (`tests/test_ledger_conservation.py`). Lean proves the algorithm; pytest
  bridges the float64 code to the algorithm.
- **N5 · Every claim carries its deflation.** A reported edge embeds the trial
  count it was deflated against (DSR / expected-max-Sharpe), and the deflation is
  recomputable from the trial ledger. The ledger is **one canonical table** (one
  periodic Sharpe per trial, a common frequency, and the selection-set id each
  trial belongs to); a candidate is deflated only against **its own selection
  set**, using an *effective* independent-trial count (estimated from trial-return
  correlation), never a raw grid size pooled across heterogeneous
  universes/frequencies. There is no "global" deflation across incommensurable
  strategies — the False Strategy Theorem's $\sigma_{SR}$ is only meaningful
  within one selection set.
- **N6 · Breadth is accounted, not assumed.** Any cross-sectional claim reports
  effective breadth $N_{\text{eff}}$ and the implied $IR$ ceiling alongside the
  realized (periodic, frequency-tagged) number, so "we have 500 names" cannot
  masquerade as "we have 500 bets." $N_{\text{eff}}$ is a **diagnostic**, and the
  ceiling is used two ways (§10): as a *falsification* gate (a realized Sharpe
  *above* the ceiling signals leak/bug) and a *viability* gate (the ceiling must
  clear the after-cost hurdle with margin). Being *below* the ceiling is the
  normal case and is not evidence of edge.
- **N7 · Fail loud, not silent-zero.** A data fetch failure, an unfit member, or
  a degenerate estimate raises or de-grosses explicitly. Silent degradation to
  an empty frame / zero position is a defect (it is the current failure mode in
  `data_loader`, the forecast members, and sentiment).
- **N8 · The production import path is JAX/torch-free.** Importing the live loop
  must not transitively import `jax`, `torch`, `prophet`, `mlflow`, or
  `matplotlib`. Research heavyweights live behind the quarantine boundary (§9).
  *Scope:* enforced on **new** modules (`regime/`, the `validation` additions,
  `live/`) and on the live import path once `live/` lands (R4). The legacy
  `ensemble.py → lstm_ppo → jax` and `features.py → flax.nnx` couplings are
  *grandfathered* until the R1 decoupling. Enforcement mechanism is a CI
  import-linter check, so N8 is a test, not a slogan.

If a proposed change violates one of N1–N8, the change is wrong, not the
invariant.

---

## 2. The organizing abstraction

Prism is a pipeline of pure-ish stages over a **wide panel** (dates × symbols),
not a bag of per-symbol models. Each stage has a typed contract (§7). The
stages:

```
            ┌─────────────────────────── REGIME (§7.5) ───────────────────────────┐
            │  curve state · VRP / vol term structure · dollar · net liquidity      │
            │  → conditioning + gross-scaling + kill signals (never a tradable book) │
            └───────────────────────────────┬──────────────────────────────────────┘
                                             ▼ (conditions sizing/gating)
 DATA ──► SIGNAL ──────► RESIDUALIZE ──────► CONSTRUCT ──────► EXECUTE ──────► LEDGER
 (§7.1)   per-name /      factor-neutralize   breadth- &        t+1 fills,      net equity,
          x-sectional     (RMT-cleaned) →     cost-aware book:  sqrt-ADV,       costs, PnL
          scores          residual scores     caps, no-trade    participation   (N3, N4)
          (ensemble is    (Avellaneda–Lee,    bands, netting    gate, borrow
          ONE optional     PCA/ETF factors)   (§7.3)            (§7.4)
          node, §7.1)     (§7.2)
                                             ▲
                        HARNESS (§7.6): purged WFO · DSR/PBO/PSR · claim tiers · N_eff ·
                        capacity curve · leakage & ledger property tests · live monitor
```

Key consequences of this shape:

- **The forecaster ensemble is demoted to a plug-in signal node.** It is not
  deleted (the XGBoost/Prophet/ARIMA forecast core + vol-mapping + conformal
  blend is real engineering), but it is *one* implementation of the Signal
  contract, sitting beside residual-reversion and any future signal, decoupled
  from JAX (§9). The RL policies are removed from the production path entirely.
- **Residualization is first-class, not a stat-arb-only detail.** Every
  cross-sectional signal passes through factor neutralization so that "breadth"
  means independent residual bets, not co-moving factor bets (audit S-4). This
  is where Marchenko–Pastur covariance cleaning lives.
- **Construction is the primary lever.** A cost-aware no-trade band moved the
  residual slice's net Sharpe from −0.65 to −0.01 (still net-negative) — but it
  also lifted *gross* from +0.23 to +0.42, so it acts as an in-sample selection
  filter, not a pure cost reducer, and its band was grid-searched (un-deflated).
  Construction (§7.3) is nonetheless where the marginal engineering hour is best
  spent — via a **closed-form** band, deflated as a counted trial — ahead of any
  new alpha, because the cost side is where the gap lives.
- **Regime conditions; it does not trade.** Rates curve, VIX term structure, the
  dollar, and net liquidity enter as sizing/gating features and de-gross
  triggers, never as a standalone traded sleeve (§4, §7.5).

---

## 3. The six "laws," triaged into artifacts

The lecture's six physics-analogy laws are the *seed*, not the skeleton. Taken
literally they are a mix of one load-bearing constraint, several cheap feature
ideas, one correctness invariant, and one piece of narrative symmetry. The
honest mapping — what each law is *actually allowed to become* in this codebase:

| # | Law | Verdict | Concrete artifact in Prism | Why not more |
|---|-----|---------|----------------------------|--------------|
| 3 | **FLAM + RMT** — $IR=IC\sqrt{BR}$, ceiling $IC/\sqrt{\rho}$, Marchenko–Pastur cleaning | **Directly actionable — the load-bearing law** | *(TO-BUILD in 0.3.0)* (a) `effective_breadth` / `information_ratio_ceiling` / `fundamental_law_diagnostic` in `validation/metrics.py`: report $N_{\text{eff}}$ + the periodic $IC\sqrt{N_{\text{eff}}}$ ceiling beside the realized periodic Sharpe (N6). The ceiling is **not** a promotion threshold (being below a maximum is the normal case, §10) — it is a *falsification* check (realized > ceiling ⇒ leak/bug) and a *viability* check (ceiling vs after-cost hurdle). (b) MP factor-count selection in `residual/` via the MP edge $\lambda_+=(1+\sqrt{N/T})^2$. *(SHIPPED)* (c) James–Stein signal shrinkage (`arbitrage/pairs.py`). | The law manufactures no alpha; it only tells you when an alpha is too thin to survive cost. That *is* the project's whole decision. |
| 5 | **Sqrt impact** $\sim \sigma\sqrt{Q/\text{ADV}}$ **+ Hawkes** $n\approx0.9$ | **Impact half: capacity is a scaling-readiness ruler, not the binding cost at this scale; Hawkes is a non-goal** | *(SHIPPED)* sqrt-ADV impact in `execution/target_weights.py` (`adv_impact_model='sqrt'`, referenced to `initial_capital`). *(TO-BUILD in 0.3.0)* a `capacity_curve` (net Sharpe vs AUM) and a **participation gate** (deny/downscale any name-day trade over a %ADV cap). **But at ~$2k–$10k the sqrt-ADV term does not bind** — the −0.65 result was produced with `adv_impact_coeff=0`; the binding cost is a *flat* per-name spread × turnover ≈ 0.31. So capacity is cheap scaling insurance, and the **primary construction objective is turnover × effective (per-liquidity-bucket) spread vs gross**, not impact. | Hawkes self-excitation lives at second-to-minute scale, invisible in daily bars; fitting it to ~250 obs/yr is numerology, and $n\approx0.9$ is itself empirically contested. |
| 2 | **Yield curve as equation of state** — PCA level/slope/curvature | **Partially — a cheap regime feature** | `regime/curve.py`: level/slope/curvature from free FRED/Treasury CMT yields via **fixed** Litterman–Scheinkman-shaped contrasts (not rolling PCA), gated on measured incremental IC. | "Equation of state" oversells: the curve is not a sufficient statistic for equity returns (2022–24 was the longest false recession signal on record). Worth an afternoon, not a pillar. |
| 4 | **Diffusion / Black–Scholes** — vol smile, realized-vs-implied | **Partially — one feature, one hard non-goal** | `regime/vol.py`: variance risk premium ($\text{VIX}^2-$ realized var) and VIX term slope (VIX9D/VIX/VIX3M) as sizing/gating features. **Non-goal:** no options positions, no vol selling, no delta hedging (§8). | Trading the gap needs OPRA surface data and intraday hedging infra that do not exist at $0; unhedged retail vol-selling is the canonical blow-up. |
| 1 | **Capital conservation + rehypothecation** $1/(1-v)$ | **Partially — a correctness invariant, not alpha** | N4 ledger-conservation property test on `execution/target_weights.py`. Optionally: net liquidity (WALCL − RRP − TGA) from FRED as a *monitored* regime diagnostic, gated on IC before it touches sizing. | The rehypothecation multiplier needs collateral velocity that is unobservable at retail frequency; net-liquidity↔SPX is a contested QE-era artifact at weekly cadence — negligible breadth. |
| 6 | **Latency light-cones** — speed-of-light, midpoint placement | **Decorative — deleted from the skeleton** | None. Its only residue is a design *assumption* already encoded: pessimistic, adversarial taker-side fills (cross the spread, pay impact, never model queue priority or rebates), plus one non-goal sentence. | A retail daily bot is not slow in this race — it is not a participant. Presenting it as a pillar is the clearest tell the six-law framing was assembled for narrative symmetry, not this builder's decision surface. |

**The skeleton, honestly, is two load-bearing laws — breadth, and cost
(turnover × effective spread; capacity/impact is a scaling-readiness ruler that
does not bind at this AUM) — three cheap regime feature blocks (curve, vol,
liquidity), and two invariants (ledger, adversarial fills).** Everything else is
a non-goal. §7 turns this row set into contracts.

---

## 4. Market scope and the $0 data stack

Analyzed for a US individual, retail latency, daily-to-weekly horizon, mid-2026
regulatory state. Full qualitative + structural analysis in `MARKETS.md`; the
operative verdicts:

| Market | Verdict | Role in v0.3.0 | Decisive reason |
|--------|---------|----------------|-----------------|
| **US equities & ETFs** | **CORE (cross-sectional)** | Primary execution market; the one venue with cross-sectional breadth | Commission-free live+paper via Alpaca; $0 borrow on 5k+ ETB names; daily horizon is exactly where retail latency + PFOF stop disqualifying; the existing stack is already residual stat-arb on equities. Runs in a **Reg-T margin account with locates** (the residual book shorts), not a cash account. PDT repeal (phasing in through ~2027) is a secondary tailwind — an N2 next-open overnight-hold pattern was never a day-trading pattern to begin with. |
| **Crypto spot (BTC/ETH majors)** | **CORE-CANDIDATE (time-series lane)** | Distinct long/flat time-series book — **bypasses the §7.2 residualize stage** | Uniquely delivers production-grade real-time L2 data **and** trading API at exactly $0 (fees only). But it is **rank-1** (two correlated majors, $N_{\text{eff}}\approx1$) — a *time-series* book, not cross-sectional, so it does **not** inherit the breadth thesis or the cross-sectional tier language and carries its **own** `net_edge` bar. Execution home is **one** named US venue at its real fee (§4 stack); custody/counterparty is the survival risk — sweep profits, minimize float. No live capital in v0.3.0. |
| **US Treasuries / rates** | **SIGNAL_ONLY** | Curve as regime state; optional ETF satellite | Direct cash Treasuries closed to retail at $0 (clearing mandate 2026-12 / 2027-06 raises the bar). Curve level/slope/curvature is first-class free regime context (FRED/Treasury, 60+ yr). |
| **FX (G10 majors)** | **SIGNAL_ONLY** | Dollar regime, carry, risk-on/off | Deepest, most last-look-protected market on earth; no retail microstructure edge, ~7–9 names give no cross-sectional breadth. Data is genuinely $0 and macro-informative. |
| **Options / vol surface** | **SIGNAL_ONLY** | VIX term structure + RV-IV spread overlay | Real-time OPRA surface is unaffordable at $0 (the binding wall); but VIX / VIX9D / VIX3M and realized-vs-implied are free EOD and high-value for gating. Direct options trading deferred (spread+fee tax, margin/assignment complexity). |
| **Futures / commodities** | **SIGNAL_ONLY** | Term-structure (contango/backwardation) regime; ETF proxies | No $0 execution path (universal per-contract commission, $2–5k margin, tightened CME data licensing). Take commodity/rates/index exposure via ETFs; consume the forward curve as a free regime signal. |

**Net:** 1 cross-sectional core (US equities/ETFs) + 1 time-series core-candidate
(crypto majors, its own evidence bar, residualize-bypassed); 4 free regime/context
layers (rates curve, FX dollar, VIX term structure, futures term structure). No
live capital leaves these markets in v0.3.0, and none leaves any market below the
`net_edge` tier (§10).

**The $0 data stack (verified free tiers; one keyed spine + official sources):**

| Job | Primary → fallbacks |
|-----|---------------------|
| Equity EOD bars | Stooq bulk (deep-history backfill) → **Twelve Data** `/time_series 1day` (incumbent, keyed; also `/dividends`) → Alpaca IEX daily → yfinance (last resort, quarantined) |
| Equity intraday (execution timing) | Alpaca IEX minute bars → Twelve Data intraday |
| Crypto bars | **Binance.US** public REST/WebSocket (unauth, US-accessible) → Coinbase (~10/s) → Kraken (≤1/s). **Binance.com** is data cross-check only (not US-accessible). |
| Yield curve | Treasury Fiscal Data XML (no key) → FRED `DGS*` |
| Vol indices | FRED `VIXCLS`/`VXVCLS` → CBOE daily CSV (VIX9D is CBOE-only) |
| Macro / liquidity | FRED `WALCL`, `RRPONTSYD`, `WTREGEN`/`WDTGAL` |
| Fundamentals / events | SEC EDGAR (≤10 req/s, User-Agent required) |
| Equity execution | **Alpaca** (equities, paper → live, commission-free); IBKR fallback |
| Crypto execution | **one** named US venue = its data home: **Binance.US** (~0.02% ≈ 2 bp taker at small size) *or* Coinbase Advanced (higher fee) — pick one and price the book off *its* real fee. Alpaca crypto is a **paper/validation + bars fallback**, not the priced execution home (its retail crypto fills are materially more expensive). |

**Binding constraint:** Twelve Data 800 credits/day, 8/min. Mitigation: range-keyed
cache (already implemented), incremental delta-fetch (to be built, §7.0), Stooq
for deep history, hard universe cap. `yfinance` and `Binance.com` are quarantined
as non-production cross-checks (ToS / US-jurisdiction risk). **Every net claim
must record the venue and per-fill fee it was priced against** (I-7); a crypto
`net_edge` priced off Binance.US's 2 bp does not transfer to a Coinbase-default
fee tier.

---

## 5. Layered architecture and the data spine

```
src/prism/                     (the importable package — src-layout as of v0.3.0)
  io/          data access + universe + $0 source adapters + incremental store   [ADAPT data_loader]
  regime/      curve.py · vol.py · liquidity.py · dollar.py  → RegimeState        [NEW]
  signal/      base Signal contract · ensemble_node (JAX-free) · residual_node    [ADAPT]
  residual/    factor model + RMT cleaning + neutralization                       [ADAPT factors/residual]
  construct/   breadth/cost-aware book: caps, no-trade bands, netting (online)    [ADAPT construct]
  execution/   costs · target-weight accounting · participation gate · borrow     [KEEP+extend]
  validation/  purged WFO · PSR/DSR/PBO · N_eff · capacity · claim packets         [KEEP+extend]
  live/        the daily loop: state, reconcile, decide, order, monitor, kill      [NEW]
  ops/         logging · (research-only) mlflow                                    [KEEP logging]
research/      quarantined: RL members, batch prism/scripts, sweep, stat-arb WFO CLIs [MOVE, JAX-heavy]
```

**The canonical data object is a tz-aware (America/New_York) wide panel**:
`close`, `open`, `volume`, plus a point-in-time `membership` mask (dates ×
symbols). Every stage consumes and/or produces panels or per-day cross-sections;
nothing downstream of `io/` re-fetches. Corporate actions: prices split-adjusted
but **not** dividend-adjusted; dividends credited as explicit cash (the correct
total-return treatment). Split-driven back-rewrites of stored bars are handled by
the incremental store, not masked by full refetch.

---

## 6. Standing invariants (the standard every component upholds)

Beyond N1–N8 (§1), these are the *methodological* standards the harness pins:

- **I-1 · Purge + embargo on every fit.** Training rows whose forward-label
  window overlaps a test slice are dropped; a buffer after each test slice is
  embargoed from later folds (`validation.walk_forward.PurgedWalkForward`). No
  80/20 split exists anywhere.
- **I-2 · Train-only transforms.** Scalers, clip bounds, factor loadings, and
  covariance estimates are fit on train data only and applied — never bfill,
  never fit on the full series before splitting.
- **I-3 · Score, not price.** Signals emit a standardized score (expected return
  in per-bar σ units) with horizon metadata, never a raw price; the price→position
  vol mapping uses $\sigma_h=\sigma_{\text{daily}}\sqrt{h}$ (fixes the audit D-2
  $\sqrt{h}$ unit break). Blending happens in position space only.
- **I-4 · One sizing function.** Conviction, cross-model agreement, and conformal
  band width fold into the target weight exactly once. No stacked multiplicative
  shrinkage.
- **I-5 · Deflate against reality.** DSR uses an honestly enumerated trial count;
  PBO runs over the actual selection set (the sweep's $T\times N$ return matrix),
  not a decorative 4-strategy set. Every knob that was searched is a counted
  trial in the ledger.
- **I-6 · Claim tiers are the vocabulary of results** (§10). No result is
  described above the tier its metrics support.
- **I-7 · Convention tags travel with artifacts.** Every results artifact records
  its data convention (price-return vs total-return, universe as-of, survivorship
  coverage %) so cross-run comparisons cannot silently mix conventions.
- **I-8 · Regime features earn their place.** A regime/context feature ships only
  after it shows positive incremental IC in purged WFO; otherwise it is logged as
  a monitored diagnostic, not fed to sizing.
- **I-9 · Cost is calibrated, not a flat constant.** The dominant realized cost at
  this scale is turnover × spread, and a flat `spread_bps` across 500 names (mega-
  through small-cap) understates the real toll on illiquid names. The effective
  spread is calibrated **per liquidity bucket** (by dollar-volume), plus an
  arrival-slippage / adverse-selection term estimated from paper/live fills once
  they exist; until then half-spread stands as a *conservative upper* proxy on
  liquid ETB retail-notional fills. Every net claim records its spread assumption.

---

## 7. Component contracts

Each stage is defined by *what it consumes, what it guarantees, and what it must
never do*. Interfaces are sketched in Python-ish pseudotype; the point is the
contract, not the signature.

### 7.0 IO / data (`io/`) — ADAPT of `data_loader.py`, `universe_sp500.py`
- **Consumes:** symbol list, interval, date range, $0 source config.
- **Guarantees:** tz-aware panels from a local **incremental store** — delta
  fetch of only the missing tail since last close, appended; rate-limited
  (token bucket against the 8/min–800/day budget) with backoff; split-driven
  back-rewrite handling; dividends as cash with a negative-cache TTL that
  distinguishes "no dividends" from "fetch failed" (fixes the poisoning bug).
- **Never:** silently return an empty frame on failure (N7); refetch full
  history daily; let the Wikipedia universe scrape sit on the runtime hot path
  (it is a supervised periodic build emitting a coverage-ledgered snapshot).
- **Keeps as-is:** `universe_sp500.py` PIT reconstruction + `CoverageReport`
  (the survivorship leak is *counted*, not hidden) — a production-grade asset.

### 7.1 Signal (`signal/`) — the pluggable alpha node
- **Contract:** `fit(panel_train) -> None; score(cross_section_asof_t) -> Series`
  of standardized scores (I-3), one per eligible name, NaN where no opinion.
- **Guarantees:** causal (N1), horizon-tagged, scale-invariant under a synthetic
  price-level shift (property-tested).
- **Implementations in v0.3.0:** (a) `residual_node` — Avellaneda–Lee residual
  reversion (the credible breadth path); (b) `ensemble_node` — the salvaged
  XGBoost/Prophet/ARIMA forecast blend + conformal band, **decoupled from JAX**
  (lazy/removed policy branch, pure-numpy scalers). Both are *nodes*, not the
  system. RL policies are not production signals.
- **Never:** construct a portfolio, know about other names' weights, or import a
  research heavyweight into the production path (N8).
- **Time-series lane (crypto).** A rank-1 book ($N_{\text{eff}}\approx1$) has no
  cross-section to residualize, so the crypto node **bypasses §7.2** and is a
  pure time-series signal (trend / vol-state / carry-if-ever). It does not use
  `residual_node` (which needs breadth) or `ensemble_node` (the demoted equity
  forecaster); it is its own node with its own `net_edge` bar. This is the one
  sanctioned exception to the unconditional score→**residualize**→construct spine.

### 7.2 Residualize (`residual/`) — ADAPT of `arbitrage/factors.py`, `residual.py`
- **Contract:** `neutralize(scores, returns_window) -> residual_scores` and
  `factor_model(returns_window) -> (loadings, factor_returns)`.
- **Guarantees:** factor loadings and covariance estimated causally on trailing
  windows; **factor count chosen by the Marchenko–Pastur edge** $\lambda_+$, not
  a fixed `n_factors=15`; bulk eigenvalues clipped when residual covariance is
  used downstream. PCA eigenportfolios or fixed sector-ETF factors.
- **Purpose:** make "breadth" mean independent residual bets (N6, audit S-4), so
  the correlation ceiling $IC/\sqrt{\rho}$ is not silently breached.
- **Breadth is measured on the traded residuals.** $N_{\text{eff}}$ for the
  ceiling (N6) is computed as the participation ratio $(\sum\lambda)^2/\sum\lambda^2$
  of the **post-residualization** covariance — the covariance of what the book
  actually trades — not from a pre-neutralization average $\rho$. This closes the
  endogeneity the reviewer flagged: residualization deliberately drives residual
  correlation toward zero, which *raises* $N_{\text{eff}}$, so the ceiling must be
  read on the residuals or it is non-binding by construction.

### 7.3 Construct (`construct/`) — ADAPT of `portfolio/construct.py` (online)
- **Contract:** `build(residual_scores, state) -> target_weights` where `state`
  carries **yesterday's held/filled weights** (the online fix — the current
  `apply_no_trade_band` replays hysteresis from held=0 over a whole frame and
  would reset every day in a live loop).
- **Guarantees:** down-only gross/per-symbol caps; **cost-aware no-trade band**
  sized from the OU half-life and round-trip cost by a **closed-form
  Gârleanu–Pedersen rule**, not grid-searched — each swept band value is a counted
  DSR trial that inflates the deflation hurdle, so the band's half-life/cost inputs
  are pre-registered or counted; residual-book netting of stock + hedge legs.
- **This is the primary lever** (§2), so v0.3.0 ships the seed of it now: a
  **single-step online no-trade band** `step_no_trade_band(prev_held, target,
  band)` (the online fix for the held=0 replay bug), additive and tested. The full
  closed-form-band online rebalancer is R2. The band's *effect* must be read as a
  return-stream change (it moved the slice from −0.65 to −0.01 net **and** +0.23 to
  +0.42 gross — a searched in-sample filter, still net-negative), never as a proven
  cost-only win.

### 7.4 Execute (`execution/`) — KEEP + extend
- **Contract:** `account(target_weights, open_prices, adv, ...) -> (fills, costs,
  returns, equity)` at t+1 (N2), net of all costs (N3).
- **Keeps:** the t+1 convention, cost primitives (`costs.py`), sqrt-ADV impact,
  borrow on shorts — all property-tested, the cleanest subsystem in the repo.
- **Adds:** a **hard participation gate** (deny/downscale any name-day trade over
  a %ADV cap — the missing completion of law 5), impact referenced to *current*
  equity not static initial capital, and a **capacity-curve** output (net Sharpe
  vs deployed AUM).
- **Live extension (`live/`):** a broker adapter (Alpaca) with **durable order
  state** — a daily loop that restarts between decision (close *t*) and fill
  (open *t+1*) must not lose queued orders (the current in-memory queue does).

### 7.5 Regime (`regime/`) — NEW, the honest residue of laws 1/2/4
- **Contract:** `regime_state(asof_t) -> RegimeState` = {curve level/slope/
  curvature, VRP, vol term slope, dollar factor, net liquidity}, all from free
  EOD sources (§4), all causal.
- **Consumers:** construction gross-scaling and de-gross/kill triggers; signal
  conditioning features (gated by I-8).
- **Never:** become a tradable book. Regime conditions; it does not trade.
- **Specified follow-ons (macro-playbook conditioning; sequenced R4, not
  v0.3.0):** an **inflation-expectations block** — real yields (FRED `DFII10`)
  and breakevens (`T10YIE`), the direct observables of a "2% rhetoric, higher
  realized" financial-repression regime (breakeven-vs-target divergence,
  real-rate suppression; the nominal-only curve block cannot see this); and a
  **stablecoin-float term** extending the net-liquidity identity — aggregate
  stablecoin market cap (DefiLlama, free/no key) as structural T-bill demand
  outside the Fed's balance sheet, a fourth term beside `WALCL − RRP − TGA`
  for the post-GENIUS-Act issuance regime. Both are conditioning inputs only,
  IC-gated (I-8) like every regime feature; the dollar-neutral residual core
  is approximately inflation-neutral by construction, so the playbook enters
  through regime inputs and the hurdle (§10), not through the alpha.

### 7.6 Harness (`validation/`) — KEEP + extend
- **Keeps:** `PurgedWalkForward`, PSR/DSR/PBO/expected-max-Sharpe, claim packets
  with hashed git state — pure, tested, near-zero coupling; also the live monitor
  (rolling PSR/DSR on the live return stream for the kill-switch).
- **Adds:** `effective_breadth` + `information_ratio_ceiling` +
  `fundamental_law_diagnostic` (N6). The ceiling is specified operationally: **rank
  IC** at a stated forward horizon and estimation window, with a bootstrap CI;
  gate on the **lower** CI bound; IC and the gated Sharpe are never estimated on
  the same fold. Both sides are **periodic** (ceiling $=IC_{\text{periodic}}\sqrt{N_{\text{eff}}}$
  vs periodic Sharpe), or annual with $BR=N_{\text{eff}}\times$ periods/yr — never a
  daily IC against an annualized Sharpe ($\sqrt{252}$ mismatch). Also: `capacity_curve`
  (scaling-readiness, §3 law 5); a `cost_toll` diagnostic (turnover × effective
  spread vs gross, and the fold-fraction where toll ≥ gross — the primary retail
  cost lens); PBO over the *real* selection set; the canonical trial ledger (N5)
  feeding `expected_max_sharpe` per selection set.

### 7.7 Live loop (`live/`) — NEW, the thing v0.2 never had
- **Cadence:** once per session (crypto: a UTC-daily bar with a continuously-armed
  kill-switch, since there is no close). Steps: reconcile broker state → delta
  fetch (7.0) → regime (7.5) → score (7.1) → residualize (7.2) → construct
  against *held* weights (7.3) → participation-gate + submit (7.4) → persist
  state → monitor (7.6).
- **State is durable.** Positions, cash, pending orders, ACI band queue, and last
  processed bar survive process restarts. Nothing important lives only in memory.
- **Model staleness is explicit.** A defined retrain cadence owns which model
  serves "today"; a drift check gates promotion of a refit. No "last WFO fold
  silently becomes production."

---

## 8. Non-goals (explicit, with reasons)

Written down so they are not relitigated:

- **No options / vol trading, no vol selling, no delta hedging.** Free data
  cannot reconstruct a real-time surface; unhedged retail short-vol is the
  canonical blow-up. VIX term structure is consumed as *regime only*.
- **No sub-daily / intraday alpha.** At retail latency + PFOF/last-look fills you
  are inside everyone's light-cone; the horizon floor is ~1 day by structure, not
  by preference.
- **No latency, venue-map, or order-book microstructure modeling.** Law 6 is a
  non-participant for this builder; fills are modeled adversarially and that is
  the whole lesson.
- **No reinforcement-learning policies in production.** On one realized daily path
  per name, model-free RL learns little a supervised signal does not, and it
  imports the entire JAX stack. The RL trio is quarantined to `research/`.
- **No direct futures, cash Treasuries, or spot-FX execution.** No $0 path;
  exposure (when wanted) comes through ETFs in the equities lane.
- **No crypto shorting / perps / margin in v0.3.0.** US retail spot shorting is
  restricted; crypto is long/flat, counterparty-capped. This knowingly forecloses
  the one market-neutral crypto trade (BTC/ETH relative-value) and leaves only
  directional/vol-state timing — an accepted cost of the no-margin, no-perp
  constraint, revisited only if a cleared venue is later justified.
- **No delisting-complete universe beyond the counted coverage leak.** CRSP/Norgate
  is out of scope; the survivorship leak (~9%) is disclosed on every claim (I-7),
  not hidden.
- **No Hawkes / self-excitation / branching-ratio modeling.** Daily bars cannot
  see it; $n\approx0.9$ is contested even for HFT.

---

## 9. Production / research boundary (salvage map)

Derived from the per-subsystem audit. **KEEP** = production-usable ~as-is;
**ADAPT** = valuable, needs online rework; **QUARANTINE** = sound research, out of
the production import path; **DROP** = delete/archive.

| Subsystem | Disposition | Note |
|-----------|-------------|------|
| `validation/` (walk_forward, metrics, trials) | **KEEP** | Cleanest subsystem; pure numpy/pandas/scipy; 54 tests. metrics→live monitor, walk_forward→retrain, trials→research provenance. |
| `execution/` (costs, target_weights) | **KEEP + extend** | t+1 + sqrt-ADV, best-tested; add participation gate, capacity curve, live broker adapter. |
| `portfolio/construct.py` | **KEEP + adapt** | Add the single-step online no-trade-band variant taking prior held weights. |
| `conformal/` (enbpi, aci) | **KEEP + adapt** | Persist the ACI band queue across restarts. |
| `universe_sp500.py` | **KEEP** | Hidden gem; PIT + counted survivorship leak. Runs as periodic build, not hot path. |
| `logging_utils.py` | **KEEP** | Production-ready as-is. |
| `data_loader.py`, `config.py` | **ADAPT** | Incremental store, rate limiting, dividend-cache fix, fail-loud; split config (data vs ensemble). |
| Forecast members (xgboost, prophet, arima), `mapping.py`, ensemble OOF+conformal core | **ADAPT** | Salvage as a JAX-free `ensemble_node`; fix ARIMA per-bar refit; fail-loud. |
| `features.py` | **ADAPT** | Strip the gratuitous `nnx.Module`/`nnx.Param` scaler wrapping (the only reason the feature path imports JAX). |
| `sentiment_analysis.py` | **ADAPT, low priority** | Clean PIT bucketing + FinBERT, but unpaginated 100-article fetch → recency bias; optional signal, not a v0.3.0 dependency. |
| RL members (lstm_ppo, xlstm_ppo, xlstm_grpo) | **QUARANTINE** | JAX-heavy, path-memorizing; `research/` only. |
| `prism/scripts/` batch WFO (training, backtest, sweep, rl_seed_eval) | **QUARANTINE** | Offline fold context; becomes the validation harness that *gates* live, not part of it. |
| `arbitrage/` stat-arb WFO CLIs | **QUARANTINE** | Signal core (factors/residual) promotes to `residual/`; the WFO/ledger CLIs stay research. Net-negative in every config to date. |
| `baselines/` | **QUARANTINE** | Research comparators. |
| `tracking/mlflow_utils.py` | **QUARANTINE** | Heavy dep; research provenance only. |
| `prism/scripts/prediction.py` | **DROP** | Feeds unscaled features (train/serve skew), dead pickle path. |
| `pyproject` `jax[cuda12]` + `torch` as hard deps | **DROP from core** | Move to a `research` optional-dependency extra. |
| Dead code: `sentiment_cache`, `_cost_row`/`_target_from_row` twins, `TradingConfig.stop_loss`/`take_profit`, `DEFAULT_MODEL_WEIGHTS` RL entries | **DROP** | Phantom surface. |

---

## 10. Claim tiers, promotion gates, and the kill-criterion

The existing five-tier ladder (`validation/trials.py`) is the vocabulary; Prism
adds the **breadth, cost, and falsification** gates that make a tier bankable.

**The ceiling is not a promotion threshold.** $IC\sqrt{N_{\text{eff}}}$ is the
*maximum* achievable IR — a realized Sharpe *below* it is the normal case and does
zero gating work. So it enters as two distinct gates: a **falsification** gate (a
realized Sharpe *above* the ceiling ⇒ leak/bug/overfit — flag and reject) and a
**viability** gate (the ceiling minus the after-cost hurdle must be positive with
margin, else the edge is too thin to ever survive cost). The **after-cost hurdle
is a stated, real-terms choice, never an implicit nominal zero**: anchor it at
minimum to the prevailing T-bill yield (the cash book's opportunity cost).
Under a financial-repression regime — rhetorical 2% target, higher realized
inflation, suppressed bill yields — a nominal-zero hurdle systematically
overstates viability, so the chosen hurdle and its basis (nominal vs real) are
recorded in the claim packet. All Sharpes below are **periodic** unless
annualization is stated.

| Tier | Rule (existing) | Prism gate added |
|------|-----------------|------------------|
| `no_claim` | < min obs | — |
| `mechanics_clean` | pipeline runs, ledger conserves (N4) | leakage + ledger property tests green |
| `gross_edge` | gross return > 0 and gross Sharpe > 0 | $N_{\text{eff}}$ + ceiling reported; **falsification** gate passes (realized ≤ ceiling); **viability** gate passes (ceiling − hurdle > 0) |
| `net_edge` | net return > 0 and net Sharpe > 0 | net of the **calibrated per-bucket spread** cost (I-9), participation gate active; venue+fee recorded (I-7) |
| `robust_edge` | `net_edge` and DSR ≥ threshold | DSR against the trial's **own selection set** (N5), not a pooled global ledger; capacity curve shows a positive-net AUM band |

**No live capital is risked below `net_edge`.** The current best across all configs
is net ≈ −0.01 (daily-annualized) — **net-negative, not breakeven, and un-deflated**
— so by this spec **nothing is deployable today.**

**Kill-criterion (co-equal with the promotion gate — a program has a STOP, not
only a GO).** The residual-reversion equity sleeve is declared **uneconomic at
daily frequency** — demoted to a lower rebalance frequency or abandoned, *not*
iterated further — when, after R2 lands the closed-form G-P band + participation
gate + calibrated per-bucket spreads:

- its best **deflated** net Sharpe (against the counted construction-trial budget)
  is still < 0, **or**
- the cost toll ≥ gross in more than ~40% of folds (a structural toll-booth), **or**
- the cumulative construction-trial budget (pre-registered, ≤ ~30 counted trials)
  is exhausted without a `net_edge`-tier claim.

The construction-trial budget is logged in the ledger and every swept band/factor
value counts against it, so "cost-bound before signal-bound" cannot become infinite
runway. **Frequency-demotion and score-smoothing experiments are admissible
now, before R2** — 2–3-day/weekly rebalance cadence, EWMA-smoothed s-scores,
continuous sizing in s in place of the ±1.25/−0.5 threshold state machine —
each a counted trial against the same budget. They are cheap, and the criterion
must not wait on expensive machinery for information that hours can buy. The
near-term target is *the first `net_edge` claim under a calibrated cost
model* — and the honest alternative outcome, that the signal is simply too weak net
of realistic retail cost, is a first-class, ledgered result, not a failure to hide.

---

## 11. What v0.3.0 actually ships vs. specifies

To keep the release honest, v0.3.0 lands the **spec + analysis + the additive
load-bearing modules**, plus the **src-layout package migration** (item 8 —
originally sequenced as a follow-on, pulled forward once the packaging defect it
fixes was found). Items 1–7 are additive; item 8 is a mechanical rename whose
only test changes are import paths. Every claim is verified by a committed
test, not asserted.

**v0.3.0 acceptance criteria (implemented + tested this release):**
1. `SPEC.md` and `MARKETS.md` (this constitution + the market/data analysis).
2. `validation/metrics.py`: `effective_breadth`, `effective_breadth_from_cov`
   (participation ratio on post-residual covariance), `information_ratio_ceiling`,
   `fundamental_law_diagnostic` (returns $N_{\text{eff}}$, periodic ceiling,
   realized, falsification flag, viability margin) — N6, law 3a.
3. `validation/capacity.py`: `capacity_curve` (net Sharpe vs AUM via the shipped
   sqrt-ADV model — scaling-readiness) and `cost_toll` (turnover × effective spread
   vs gross, plus the fold-fraction where toll ≥ gross — the primary retail cost
   lens, law 5).
4. `execution/participation.py`: the hard **participation gate**
   (`participation_capped_targets`) + `tests/test_ledger_conservation.py` (the N4
   property test on `backtest_target_weights`).
5. `portfolio/construct.py`: the **single-step online no-trade band**
   (`step_no_trade_band`) — the online fix seed of the primary lever (§7.3).
6. `regime/`: pure, offline-tested feature math — `curve` (level/slope/curvature via
   fixed Litterman–Scheinkman contrasts), `vol` (VRP, VIX term slope), `liquidity`
   (net liquidity = WALCL − RRP − TGA) — plus a documented free-source map. The
   live FRED/Treasury/CBOE fetch adapter is a thin, network-gated shell (untestable
   in-sandbox); the math it feeds is tested on synthetic series.
7. README + ARCHITECTURE rewritten to the cross-sectional spine; version → 0.3.0.
8. **src-layout migration**: the importable package is `prism` at `src/prism/`.
   Previously the wheel config (`packages = ["src", "scripts"]`) installed
   top-level packages literally named `src` and `scripts` into site-packages —
   a genuine packaging defect, not just style. `scripts/` moved to
   `prism.scripts` (console entry points updated); `PROJECT_DIR` and the two
   script-side repo-root derivations re-anchored to the new depth; tests import
   the source tree via `pythonpath = ["src"]`. Full suite green post-move.

**Specified, sequenced for follow-on (does NOT gate this tag):**
- The `research/` quarantine relocation (the package move itself shipped —
  item 8 above).
- The JAX/torch decoupling of `features.py` / `ensemble.py`, the `pyproject`
  research-extras split, and the N8 import-linter CI check (mechanical; must keep
  the full existing suite green).
- The `io/` incremental store + rate limiter + dividend-cache fix.
- The **full** online cost-aware rebalancer (closed-form G-P band) — R2, the
  primary lever; v0.3.0 ships only its single-step seed (item 5).
- The `live/` daily loop and the concrete Alpaca / Binance.US broker adapters.

Crypto (the time-series lane) ships **no** code in v0.3.0 — it is scoped in the
spec and deferred to R4; it is not counted toward the release's tested surface.

Each follow-on item is a self-contained change with a pinned exit test, in the
sequence R0→R4 of `docs/audit.md` (which this spec subsumes and does not replace).

---

## 12. Versioning and naming

- **Package:** `prism`, under src-layout (`src/prism/`), as of v0.3.0 (§11
  item 8). This fixed a real packaging defect — the wheel previously installed
  generically-named top-level `src` and `scripts` packages.
- **Distribution & repository:** renamed to `prism` in the **single
  publication-time identity break** (2026-07-05) — GitHub repo rename (old
  URLs redirect) + `[project] name` + `[project.urls]` in one commit. Bare
  "prism" is overloaded across the software ecosystem (Prism.js, Stoplight
  Prism, …); the owner weighed that and chose it anyway. If PyPI publication
  is ever pursued and the name is unavailable there, qualify the
  *distribution* name only — the import package stays `prism`.
- **System identity:** *Prism* in all docs and new modules.
- **SemVer intent:** 0.3.0 is a foundation-level re-architecture (new organizing
  abstraction, new invariants, quarantine boundary) plus a mechanical package
  rename, with the full suite green — a minor bump, consistent with the 0.x
  line.

---

## 13. Roadmap (subsumes `docs/audit.md` R0–R4)

- **R0 · Foundation (this release).** Spec, market analysis, breadth + capacity +
  ledger + regime modules, claim-tier gates, src-layout migration (`prism` at
  `src/prism/`). Exit: all new modules tested, full suite green (463 tests as
  of v0.3.1; 411 without the `[research]` extra).
- **R1 · Representation.** JAX-free signal path; `ensemble_node` as a plug-in;
  ARIMA per-bar refit; score-not-price everywhere; scale-invariance property test.
- **R2 · Decision layer + cost calibration.** Online cost-aware rebalancer
  against held weights; meta-learner with a cash vertex + turnover penalty;
  participation gate wired into construction; the slow-sleeve netting book
  (see the gating paragraph below); and the **Alpaca paper loop at trivial
  size as the I-9 cost-measurement instrument** — pulled forward from R4
  because it needs no edge, it forces durable order state into existence, and
  every fill is per-bucket spread calibration data. Without it the §10 verdict
  is judged under an uncalibrated flat spread, inside the cost model's own
  error bars in both directions. Exit: turnover responds monotonically to the
  band parameter, and a per-liquidity-bucket effective-spread table exists
  from paper fills.
- **R3 · Statistical honesty.** PBO over the real selection set; the canonical
  per-selection-set trial ledger (N5) → `expected_max_sharpe` with an effective
  independent-trial count; stat-arb multiplicity fixes. Exit: every artifact
  embeds a recomputable deflation count against its own selection set.
- **R4 · Live + breadth.** `io/` incremental store; `live/` loop hardening
  (the R2 paper instrument grows into the full unattended daily loop);
  crypto lane; MP-cleaned factor count; capacity curves per strategy; the
  inflation-expectations regime block + stablecoin-float liquidity term (§7.5
  follow-ons). Exit: a paper-traded loop runs unattended for a full cycle and
  reconciles.
- **RF · Formal track (parallel, additive).** `formal/` — a core-only Lean 4
  package machine-checking the spec's exact-arithmetic kernel: N4 ledger
  conservation (single- and multi-step), the §7.3 band's hysteresis plus the
  batch-replay-from-zero divergence witness, I-1 purge/embargo geometry, the
  §7.4 participation gate's attenuation properties. Lean proves the algebra;
  pytest bridges the float implementation to it. Next targets in value order:
  the `live/` crash-safety state machine, the R2 G-P band's monotonicity,
  trial-ledger append-only monotonicity. Charter: `docs/handoff.md §5`.

The gating discipline is fixed: the system is **cost-bound before signal-bound**,
so R2/R3 (construction + honesty) precede signal-side sophistication. No new
**fast** alpha is added until a result clears `net_edge` under the **calibrated
per-bucket spread** cost model (I-9) — *or* the §10 kill-criterion fires and the
daily residual sleeve is demoted/abandoned. **Slow signals are construction
machinery, not new alpha**, for a specific economic reason: turnover is a
property of the signal *set*, not of construction alone. A slow,
negatively-turnover-correlated sleeve (first candidate: 1–3-month
cross-sectional momentum from bars already on disk) cuts cost per unit of
gross through internal netting — reversion buys what momentum sells into, and
the trades cross before touching the market. That is the heterogeneous-decay
case Gârleanu–Pedersen is actually derived for; a single fast signal is the
framework's degenerate case. Such a sleeve is therefore admissible in R2 and
counted against the same trial budget. "Construction first" runs on a
pre-registered, ledgered budget so it terminates in a verdict (edge or kill)
rather than open-ended iteration. Long-horizon sequencing and rationale:
`docs/handoff.md`.
