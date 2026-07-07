# Momentum-program pre-registration (DRAFT — ratify before any counted run)

**Status: draft, 2026-07-06.** Committed so the frame exists *before* any
momentum trial runs; **no counted trial may execute until the owner ratifies
(or amends) this document in a dedicated commit.** Amendments after
ratification follow the same rule as `docs/demotion_design.md`: seams may be
pinned, trial values may not move.

## 0. Provenance and the honest prior

The candidate is B1 of the residual demotion budget
(`docs/certifications/001-residual-reversion-daily-negative.md §8`): 12−1
cross-sectional momentum (252-bar lookback, 21-bar skip), top/bottom decile
equal-weight long−short, monthly decisions (`decision_every=21`), on the PIT
S&P 500 universe under the frozen cost stack (closed-form band, bucket
spreads, 5% participation cap). Measured there: net annualized Sharpe +0.465,
+27.6% total (2020-01 → 2026-06), turnover 0.050/day, periodic Sharpe 0.0293
vs cash hurdle 0.0203 — the only cash-hurdle-clearing result in program
history — and DSR 0.191 against that 17-trial selection set.

The prior, stated so it cannot inflate later: **B1 was the best of 17
searched trials in another program's selection set.** Its point estimate is
selection-biased upward — being the max of 17 is precisely the situation
expected-max-Sharpe deflation describes — and one backtest of the most
published anomaly in finance (Jegadeesh–Titman 12−1) proves little:
post-publication decay is documented, and the classic failure mode (momentum
crashes — short-leg squeezes in sharp reversals, 2009-style) is
under-represented in a 5.2-year sample whose worst reversal (2020-03) the
strategy happened to survive. B1 therefore enters at `mechanics_clean`
(mechanics already parity- and property-tested by the Arm-B suite), imported
as **trial #1 of this program's ledger** — spent, not free.

## 1. What a same-sample trial can and cannot show

Every new configuration run on 2020-01 → 2026-06-16 is *in-sample relative to
the discovery of B1*. The robustness set below therefore exists for
**fragility detection only** — it can kill the candidate, it cannot promote
it. Promotion evidence must be out-of-sample by construction:

- **Extension rule:** the promotion read re-runs the *ratified* configuration
  (B1's, unmodified) on the sample extended through at least **2027-06**
  (≥ 1 year of genuinely new data), with the OOS segment's contribution
  reported separately.
- **Paper confirmation:** the momentum book runs on the Alpaca paper loop
  (`prism.scripts.paper_loop`) at trivial size while the extension accrues —
  fills calibrate the bucket schedule (I-9) and the live monitor's rolling
  PSR/DSR provides the second, sample-independent read.

## 2. Proposed counted trial set (ratify or amend, then frozen)

All on the frozen cost stack; each is one counted ledger trial in a **new
selection set** (`strategy: momentum_v1` ledger namespace); budget **exactly
8 counted trials** including the B1 import. Axes chosen to probe the
canonical fragilities (skip month, horizon, breadth, cadence), not to search
for a better backtest — the pre-registered configuration remains B1's
regardless of which robustness trial scores best (adopting a different cell
would be a new discovery event, restarting §1).

| id | delta vs B1 | probes |
|----|-------------|--------|
| M0 | — (the B1 import; not re-run) | the discovery record |
| M1 | `mom_skip_bars=0` | is the skip month load-bearing? |
| M2 | `mom_lookback_bars=126` | horizon sensitivity (6−1) |
| M3 | `mom_decile=0.2` | breadth/concentration sensitivity |
| M4 | `decision_every=63` | cadence sensitivity (quarterly) |
| M5 | `mom_skip_bars=42` | skip robustness, other direction |
| M6 | extension re-run of B1 (≥ 2027-06 data) | the §1 promotion read |
| M7 | reserve — may be ratified later for one seam discovered by M1–M6; unspent otherwise | — |

## 3. Adjudication (pre-committed)

- **Fragility kill (readable after M1–M5):** the sleeve is dropped without
  appeal if the *median* net annualized Sharpe of M1–M5 is negative under
  bucket spreads, or if any single knob move flips the sign of the net
  result at magnitude > B1's own point estimate (a signal that thin
  configuration choices, not the anomaly, produced B1).
- **Promotion (readable only at M6 + paper):** `net_edge` under bucket
  spreads on the extended sample, with DSR > 0.5 against this program's own
  ledger (≤ 8 trials), periodic net Sharpe above the then-current cash
  hurdle, **and** a live-monitor read not in contradiction (rolling PSR on
  the paper stream above 0.5 at horizon). Then deployment follows the GO
  branch of `docs/handoff.md §8` at minimum size.
- **Neither fires:** if M6 is positive but deflation-failing, the program
  waits for more OOS data — it does not buy more same-sample trials. The
  budget does not refill.

## 4. Expectation, recorded honestly

Momentum's literature prior is favorable but decayed; the realistic central
case is a modest positive that fails the deflated bar on one added year and
needs two. The failure worth designing against is the crash tail: a
short-leg squeeze can erase multiple years of this premium in weeks, and the
5% participation cap + monthly cadence do nothing to hedge it. If the
program promotes, crash exposure is priced into sizing (the capacity/vol
inputs at deployment), not hand-waved. A wildly positive robustness set
should raise suspicion (§10 falsification gate), not excitement.
