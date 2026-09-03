# Momentum-program pre-registration (RATIFIED)

**Status: ratified, 2026-07-06** (this commit is the dedicated ratification
commit; the ratification decision was delegated by the owner to the
2026-07-06 session in the same turn that supplied the new input). One
amendment was folded in during the free pre-ratification window: the §0
breadth accounting and its §4 expectation update, sourced from
`results/demotion_b1/breadth_diagnostic.json` (committed alongside as
evidence). No value in the §2 trial set or §3 adjudication rules moved.
From this point amendments follow the same rule as
`docs/demotion_design.md`: seams may be pinned, trial values may not move.

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

**Breadth accounting (folded in pre-ratification, 2026-07-06).** The N6
fundamental-law diagnostic over the finished B1 run
(`research/scripts/breadth_diagnostic.py`; reading committed at
`results/demotion_b1/breadth_diagnostic.json`):

- $N_{\text{eff}} \approx 52.7$ — participation ratio of the book's
  per-name PnL-contribution covariance (491 ever-held names, ~96 held per
  day). The raw held-name *return* covariance is far more concentrated
  (participation ratio ≈ 11.8, top eigenvalue 7.3%): co-movement is real,
  but the book is not the rank-1 crash-factor bet the prior feared — in
  sample-covariance terms. A momentum crash is a regime event a sample
  covariance cannot exhibit, so this number does **not** soften §4's crash
  paragraph.
- 21-bar rank IC **0.030 ± 0.024** (n = 62 non-overlapping cross-sections;
  one-sided lower-95 **−0.009**). 5.2 years of monthly cross-sections cannot
  pin the IC's sign. This is the single most load-bearing unknown in the
  program, and no same-sample trial can shrink it — only calendar time can.
- Realized net Sharpe already captures **61%** of the
  $IC\sqrt{N_{\text{eff}}}$ ceiling (gross 69%): **breadth binds, not IC
  capture.** Configuration search has little headroom to manufacture —
  consistent with §2's fragility-only framing — and the real capacity lever
  is universe expansion (mid-caps), which is *not* one of the 8 trials;
  pursuing it would be a new discovery event under §2's own rule.
- Viability (N6 lens, 21-bar periodic): the ceiling clears the cash hurdle
  at the point IC (0.219 vs 0.093, margin +0.126) but **fails at the
  lower-CI IC** (ceiling 0.069, margin −0.024). Viability is *unpinned, not
  established* — exactly the situation the §1 extension rule exists for.

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

## 5. Instrument amendment — paper-loop execution seams (2026-07-11)

**Amendment under the §-banner rule: seams pinned, no §2 trial value and no
§3 adjudication rule moves.** Owner-directed 2026-07-11 (execution delegated
to the session in the same turn, the ratification-delegation precedent of
this document's own banner). Motivating reading, recorded so the seam pins
are auditable: the first live refresh (decision 2026-07-08) filled 22/101
OPG orders at the 07-09 open — Alpaca *paper*'s opening-auction simulation
prints ~20-25% of OPG orders across books, sides, and price tiers — leaving
the instrument holding a 0.19-gross, ~2%-net-short partial book for a whole
monthly cadence period, contaminated by six positions originating from the
retired ensemble run-dir's in-flight orders on the shared account, and
silently unable to hold any name priced above ~2× the per-name notional
(whole-share zero-rounds: the NVR/AZO class at $100k NAV). §1's paper
confirmation reads the paper stream as evidence about *B1*; these are
instrument defects, not strategy properties, and the seams below repair the
instrument.

1. **Completion sweep.** OPG remains the primary order type (the N2
   next-open convention; the KEEP-OPG decision stands). The morning after a
   decision, while it is still pending, every *terminal-but-unexecuted*
   residual is re-submitted once as a DAY market order under the suffixed
   client id `{original_id}:S1` (`prism.live.loop.sweep_pending`;
   `prism.scripts.paper_sweep`). Sweep orders carry the **original
   decision-close reference price**, so the fills ledger records the total
   arrival cost of executing the decision at this venue; the id suffix keeps
   the auction and sweep fill populations segmentable. One completion
   generation per decision; an order still live at the venue is never swept.
2. **Fills-inclusion rules for I-9 calibration, pre-stated before any bucket
   approaches the `min_fills=30` promotion guard.** The momentum calibration
   pool contains: opening-auction fills and `:S1` sweep fills of the momentum
   book, reported jointly and separably by id suffix. It excludes: fills of
   the retired ensemble book (`runs/paper_loop/fills.jsonl` — different book,
   and its 3 fills carry 2-session arrival latency, a different estimand than
   the N2 next-open convention `prism.execution.spread` documents), and
   book-establishment/cutover fills (reference prices stale by construction;
   tagged by their run context, see seam 3). Deciding inclusion *after*
   seeing which rule promotes a bucket would be ex-post selection; this
   paragraph forecloses it.
3. **Size and segment reset.** The paper account is reset to **$1,000,000**
   and the momentum loop restarts in a fresh run-dir with a fresh equity
   ledger. "Trivial size" (§1) is unchanged in meaning — $1M is still
   capacity-irrelevant — and the dollar NAV was never a registered value; the
   reset exists because at $100k the whole-share quantum censors high-priced
   names from the book entirely and puts up to 2× weight error on names near
   the threshold. The monitor stream restarts at n=0 on the new segment:
   equity ledgers from different segments are **never concatenated** (the
   monitor forms raw NAV pct-changes with no external-flow adjustment, so a
   cross-segment join would inject a fictitious return). The first decision
   in the fresh run-dir re-anchors the monthly cadence grid; the grid phase
   is an instrument property (the prior anchor was itself the arbitrary
   2026-07-08 cutover date), not a §2 value. Establishment fills of that
   first refresh enter `fills.jsonl` (the write-ahead machinery does not
   fork) but are excluded from calibration pools per seam 2, identified by
   the first refresh bar of the segment.
4. **Order-id namespacing.** Client order ids carry a per-book prefix
   (`mom:{bar}:{symbol}` for this instrument). Two books sharing one venue
   account with the bare `{bar}:{symbol}` scheme silently substitute each
   other's same-bar orders, because the venue's duplicate-id rejection is
   (correctly) treated as submission success by the write-ahead protocol.
5. **Fidelity telemetry (additive, gates nothing).** Each refresh persists
   its decided target book (`targets.jsonl`, written inside the write-ahead
   step); each settle persists every unexecuted residual (`unfilled.jsonl`);
   each cycle records active share / weight correlation / gross ratio of the
   held book against the last refresh targets (`concordance.jsonl`,
   `prism.live.monitor.book_concordance`). The §3 promotion conjunct still
   reads the rolling PSR on the paper stream — unchanged — but the
   concordance ledger records *how much of B1* that stream was measuring,
   bar by bar, so the read at M6 is interpretable at any n.

What this amendment deliberately does not do: it does not touch the B1
configuration, the §2 trial set, the §3 adjudication rules, the M6
extension window, or the closed residual ledger; it does not admit replay
or EDGE output into calibration or the concordance stream
(`src/prism/live/replay.py` boundaries; `docs/edge_preregistration.md` §2).

The §0 breadth accounting sharpens this expectation: with realized capture
already at ~61% of the fundamental-law ceiling and the ceiling's own
viability unpinned at the lower-CI IC, the plausible upside from
configuration choices is small, and the program's outcome is dominated by
whether one more year of data pins the IC above zero. That is an argument
for the M6 read and the paper monitor, not for spending M7.
