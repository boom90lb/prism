# Tax wedge — measured asymmetries vs the symmetric-τ baseline

> **Status: uncounted diagnostic (run 2026-07-19).** Implements
> `docs/tax_wedge_spec.md` (ratified 2026-07-19, desk-review Q4): searches
> nothing, moves no ratified statistic, appends no trials-ledger row.
> Instrument: `research/scripts/tax_wedge.py`; artifact:
> `results/tax_wedge_2026-07-19.json` (tracked). The artifact was computed
> under a labeled **REFERENCE parameter vector — 0.37 federal, 0.093 state,
> $3,000 loss cap, $1,000,000 book equity — for the record's
> reproducibility, not the operator's rates**; the certified numbers stay
> pre-tax and the overlay re-runs under any vector (spec §1, §4). Local
> committed artifacts only, no network I/O (the corporate-action records are
> read from the dividend-wedge instrument's cache, never refetched).

## 1. Question

First-order, tax does not move the viability sign: strategy gains at monthly
cadence are ordinary short-term gains and the T-bill hurdle is also ordinary
income, so under symmetric contemporaneous taxation at τ the periodic Sharpe
comparison is τ-invariant (spec §2). What is measurable is the four
asymmetries — wash-sale deferral, payments-in-lieu on the short leg, the
capital-loss netting cap, and the state exemption on Treasury interest —
each computed on a recorded ledger, against the symmetric-τ baseline. Sign
convention throughout: **negative bps/yr = after-tax drag vs the baseline**,
the same orientation as the flatten (−19.4) and dividend (−36.1) wedges.

## 2. Method and ledger map

| Asymmetry | Ledger | Window |
|---|---|---|
| Wash-sale deferral | `runs/replay_floor_1000000/fills.jsonl` (per-lot fills, modeled fills through live-loop mechanics) | 2026-01-02 → 2026-06-15, 112 sessions |
| Payments-in-lieu | `results/demotion_b1/target_weights.csv` + cached Alpaca records (`results/alpaca_cash_dividends_2021-03-30_2026-06-12.json`) | 2021-03-30 → 2026-06-12 |
| Loss-netting cap | `results/demotion_b1/returns.csv` (calendar-year P&L, constant $1M base) | 2021-03-30 → 2026-06-12 |
| State exemption | `results/demotion_b1/summary.json` `after_cost_hurdle` (3.71% T-bill nominal) | — |

- **Wash sale**: FIFO lot replay of the fills ledger, 30-day window both
  sides, disallowed losses attached to replacement-lot basis, re-washed
  chains compounding naturally; applies to short-side re-entry too. **The
  certified run directory cannot support this asymmetry** — it records
  weight-space accounting only, no per-lot fills — so the replay ledger is
  the measurement surface, recorded here and in the JSON meta, never a
  silent substitution. Timing is monetized as prepaid tax on the sample-end
  deferral balance carried at the run's own recorded T-bill hurdle; no new
  rate constants enter.
- **PIL**: the dividend-wedge accrual (`w[t-1]·amount[t]/close[t-1]`) split
  by leg and calendar year. Validation gate (passed): the recomputed leg
  flows reproduce the committed `results/dividend_wedge_2026-07-13.json`
  artifact (short −99.99, long +63.93 bps/yr). Treatment: short PIL is
  capitalized into the short's capital result (IRC §263(h), sub-46-day
  shorts at monthly cadence) — relieved at τ in net-gain years exactly as
  the baseline relieves it, deepening the capped loss in the loss year. The
  long leg's dividends are ordinary at τ (the 61-day qualified holding
  window fails at monthly cadence), which is the baseline's treatment too:
  the qualified-side wedge is a **derived zero, stated rather than
  omitted**. One close cache is absent (`SATS`, the SATS→ECHO rename chain)
  — its records drop out of the accrual, counted in the JSON.
- **Loss cap**: gains taxed at τ in the year earned; a net-loss year deducts
  at most the cap against ordinary income and carries the rest forward
  against later gains. The wedge is terminal unrelieved carryforward plus
  the time value of prepaid tax at the hurdle; in-sample reversals net to
  zero by construction, which is the point — the asymmetry *is* timing plus
  terminal.
- **State**: T-bill interest is state-exempt, trading gains are not:
  −τ_state × hurdle.

## 3. Results (reference vector: 0.37 / 0.093 / $3,000 / $1M)

**Steady-state wedge (spec §3 output 1): −37.3 bps/yr**, decomposed:

| Component | bps/yr | Source ledger |
|---|---|---|
| State exemption on the hurdle | **−34.5** | certified summary |
| Loss-netting cap (2023 carryforward timing) | −2.2 | certified returns |
| PIL capitalization (2023 deepening + timing) | −0.6 | certified weights + records cache |
| Wash-sale deferral (carry on $242 deferred at sample end) | −0.04 | replay fills |

The ex-ante expectation recorded in the spec — tens of bps, same order as
the flatten and dividend wedges — is replaced by −37.3, and its composition
is lopsided: the state exemption is essentially the whole steady-state
wedge. The PIL worst-case bound (no relief at all: −46.3 bps/yr = −τ × the
short leg's 100 bps/yr dividend flow) and best case (0) bracket the measured
−0.6; the measured number is small precisely because capitalized PIL rides
the capital channel in net-gain years. Wash-sale machinery fired 30 times on
127 loss events ($5,395 disallowed) but 95% re-recognized in-sample — at
monthly cadence with ~30-day re-entry gaps the book sits at the wash
window's edge, and most deferrals resolve within the year.

**Crash-year conditional wedge (spec §3 output 2):**

| Cell | Capital cell | + State | Conditional wedge |
|---|---|---|---|
| (a) Worst sample year (2023, −$82.4k incl. PIL) | −367.8 bps | −34.5 | **−402.3 bps** |
| (b) Synthetic −30% book-return year (spec-pinned constant, §3) | −1421.4 bps | −34.5 | **−1455.9 bps** |

The 2023 cell is the year's own after-tax gap — the baseline refunds
τ·$82.4k that year, the actual code relieves τ·$3k — reported *before* the
later recovery (the carryforward was fully relieved by 2024 gains, flagged
in the JSON), because averaging the reversal back in would hide exactly the
convexity the sizing pre-registration needs to price. The synthetic −30%
year puts the penalty near −14.6% of book equity in after-tax space: this
co-moves with the momentum-crash state, and the GO-branch sizing read
consumes this number, not the steady-state average. Both cells carry a null
(not zero) wash entry: no per-lot ledger exists for 2023 or for a synthetic
year (N7).

## 4. Reading

Steady-state, tax is a −37 bps/yr overlay dominated by a term that exists
for any state-taxed operator holding the T-bill alternative — it scales
linearly with the state rate and vanishes for a no-income-tax state. The
loss cap and PIL terms are data-driven small on this sample because 2023's
loss was shallow (−7%) and fully absorbed by 2024. The asymmetry that
matters is conditional: one −30% year costs ~40× the annual steady-state
wedge in the year it lands, with recovery stretched across later gain years
at $3k/yr plus gain-absorption. After-tax, the book is short a tax
straddle on its own crash state; the steady-state number is not the risk.

## 5. Limits, and what the certified ledger cannot support

- The certified run dir has **no per-lot fills**, so the wash-sale replay
  runs on the 112-session replay ledger: a different window (2026 H1), a
  $1M whole-share account, modeled fills, spanning no tax-year boundary.
  The sample-end deferral balance ($242) is a boundary snapshot, not a
  steady-state flow; the honest wash claim is "the mechanism fires ~30
  times per half-year at this cadence and mostly self-resolves in-sample."
- Lot holding periods for the qualified-dividend test are likewise
  unsupported on the certified ledger; the long leg is treated ordinary-at-τ
  (which matches the baseline, so the recorded wedge contribution is an
  explicit zero), and the foregone qualified-rate benefit is recorded, not
  computed — it needs a qualified-rate parameter outside the ratified set.
- Constant-base P&L convention (each year's compounded return × $1M): both
  tax codes consume the same series, so the wedge is first-order insensitive
  to it; recorded in the JSON meta.
- The loss cap is a dollar figure, so it only has meaning against an equity
  base: `--book-equity` is a required parameter beside the spec §4 three,
  and the cap terms scale roughly as 1/equity — a $10k A3-style account has
  a proportionally larger cap shield, a larger book a smaller one.
- Entity and election questions — trader-status mark-to-market under
  §475(f), which would change the wash-sale treatment and which a
  monthly-cadence book very likely does not qualify for — are
  real-money-adjacent decisions for a CPA: **recorded, not answered**
  (spec §4).
