# Conditional-beta telemetry — realized market beta of the B1 book

> **Status: uncounted diagnostic (run 2026-07-19).** Measures the realized
> market exposure of a recorded return stream, searches nothing, moves no
> ratified statistic. Instrument: `research/scripts/beta_telemetry.py` over
> the certified B1 stream (`results/demotion_b1/returns.csv`, config hash
> `000b74941cfd`); artifact: `results/beta_telemetry_2026-07-19.json`
> (tracked). Desk-review Q3 tier 1; local artifacts only, no network I/O.

## 1. Question

B1 is dollar-neutral by balanced decile legs, not factor-neutralized — the
SPEC §7.2 residualize stage is unwired by design
(`src/prism/live/daily.py:9-15`). Dollar-neutral is not beta-neutral, and
the classic momentum-crash mechanism is a *conditional* exposure: after a
market fall the short leg fills with crashed high-beta names, so the book's
beta flips exactly when the market whipsaws back. An unconditional beta
averaging near zero is therefore the number most likely to falsely
reassure; the conditional cells are the point. This converts "dollar-neutral
≠ beta-neutral" from an argument into numbers, and those numbers feed the
sizing pre-registration's crash-conditional de-gross term (`docs/handoff.md`
§8 GO preconditions — precondition (a)).

## 2. Method

Book = the certified B1 OOS daily-return stream (1,308 sessions, 2021-03-30
→ 2026-06-12). Market, two series, because the local SPY bar cache begins
2025-06-01 (trend-program ADV fetch — no longer SPY history exists on disk):

- **SPY** (`data/SPY_1d_2025-06-01_2026-07-17.parquet`): joint sample
  2025-06-03 → 2026-06-12, n = 259 — the last ~12.5 months only.
- **EW proxy**: cap-blind equal-weight daily mean return over the certified
  run's own 574-name bar-cache panel (quarantine fallback included — the run
  consumed those bars; zero missing caches), full joint sample n = 1,308.
  Validation on the overlap: β(proxy on SPY) = 0.81, corr = 0.78, n = 260.
  The 0.78 is real tracking difference — equal weight versus a cap-weighted
  index in a concentrated tape — so proxy cells read as "the market the book
  trades in", not as SPY.

Conditioning is strictly prior wherever a state is claimed: the trailing
21-bar market window ends the session *before* the conditioned return, and
the drawdown state is the prior close's reading, so no cell conditions on a
bar it contains. The worst-months cell is the exception by construction —
months are selected on the same returns then conditioned on — and reads as
ex-post attribution, not a tradeable state. Any cell with n < 21 reports
beta as null with its n (N7); in this run no cell fell below the floor.
Cross-check: the 112-session replay stream through live-loop mechanics
(`runs/replay_floor_1000000/equity.jsonl`) shows β = 0.69 on SPY over
2026-01 → 2026-06 — same sign and regime as the certified-stream rolling
read for that window, higher point estimate on a third of the observations.

## 3. Results

| Cell | Market | β | n | Notes |
|---|---|---|---|---|
| Unconditional OLS | SPY (2025-06 → 2026-06) | **+0.24** | 259 | corr +0.20 |
| Unconditional OLS | EW proxy (2021-03 → 2026-06) | **−0.08** | 1,308 | corr −0.12 |
| Rolling 63d — mean / min / max / last | SPY | +0.24 / −0.17 / +0.50 / +0.31 | 197 wins | max 2026-02-04 |
| Rolling 63d — mean / min / max / last | EW proxy | −0.06 / **−0.49** / **+0.47** / +0.17 | 1,246 wins | min 2024-01-29 |
| Worst-decile trailing 21-bar market windows | SPY | **+0.42** | 24 | threshold −1.8%; n near the floor |
| Worst-decile trailing 21-bar market windows | EW proxy | −0.01 | 131 | threshold −5.0% |
| Book drawdown state (>5% below peak, prior close) | SPY | +0.15 | 85 | 603 state days full sample; max dd 14.0% |
| Book drawdown state (>5% below peak, prior close) | EW proxy | −0.10 | 603 | |
| Worst-decile book months | SPY | **+0.42** | 41 | 2025-07, 2025-11 (2 of 13 months) |
| Worst-decile book months | EW proxy | −0.16 | 145 | 7 of 64 months, incl. 2022-01/07/11 |

## 4. Reading

The full-sample unconditional beta (−0.08) is exactly the falsely
reassuring number the instrument exists to look past: the rolling 63d
series swings from −0.49 to +0.47, so the book routinely carries a
half-unit of market exposure in one direction or the other, and the sign of
every conditional cell depends on which regime the window samples. In the
current regime (the SPY overlap year) the book is *long* beta precisely
where it hurts — +0.42 in the worst-decile down-windows and +0.42 in its
own worst months, versus +0.24 unconditionally — while over the full sample
the same cells sit near zero to mildly negative because the 2022 bear
happened to catch the book on the other tack. For the sizing
pre-registration, the de-gross term should price the conditional exposure
as *regime-dependent with realized excursions to ±0.5*, not as the
full-sample −0.08; a crash-conditional term calibrated to the unconditional
number would under-degross by roughly half a unit of beta in the measured
worst case.

## 5. Limits

A sample covariance cannot exhibit a regime event — the identical caveat
`docs/momentum_design.md` §0 pins on N_eff applies verbatim here. The
2021-2026 sample's worst market episode (the 2022 bear, worst proxy
trailing-21-bar threshold −5.0%) is mild next to the 2009/2020-style
reversal the momentum-crash literature describes, and the near-zero
full-sample conditional cells measure that absence, not immunity: the
post-crash beta flip these cells exist to price is under-represented in
sample, so the measured conditional betas are floors on regime severity,
not estimates of it. The SPY cells cover 12.5 calm months (the worst-decile
threshold there is only −1.8%, and its n = 24 sits near the reporting
floor); the full-sample cells lean on a cap-blind equal-weight proxy that
tracks SPY at corr 0.78. Alpaca-fee/dividend wedges and execution seams are
out of scope — this measures the certified price-return stream, and one
replay cross-check of the live-mechanics stream.
