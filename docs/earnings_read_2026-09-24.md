# Earnings announcements — read record, 2026-09-24

**Kind:** read record for `docs/earnings_design.md`.

**Selection sets and budgets:**
- `earnings_v1`, replication route: one arm and one combination declared; both spent.
- `earnings_daily_v1`, discovery route: two arms declared; both spent.

**Commits:**
- Registration: `f2efe9e`. It is preceded by `ab6e1f6`, the engine's dollar-volume statistic restricted to its last window, whose output is unchanged.
- Every record below carries `f2efe9e` as its code commit.

**Records:**
- `results/2026-09-earnings/`: `read.json`, `comparison.json`, `returns/`, and `replication/` (`replication.json`, `combination_series.csv`).
- `results/2026-09-earnings-daily/`: `read.json`, `comparison.json`, `returns/`, `earnings_read.json`.

**Trial logs:**
- `trials/earnings_v1.jsonl`: 1 started, 1 finished.
- `trials/earnings_v1_combination.jsonl`: 1 started, 1 finished.
- `trials/earnings_daily_v1.jsonl`: 2 started, 2 finished.

## Verdict

- **`earnings_v1`: nothing promotes; the set closes; nothing is killed.**
  - The arm fails economics (F3 0.117 against a hurdle of 0.442).
  - The combination with vol-scaled momentum fails economics by 0.076 (F3 0.275 against 0.351).
  - Before costs the combination clears the hurdle (0.365). Costs decide it.
- **`earnings_daily_v1`: neither arm is killed by the registered rule, and both carry to
  the forward read.**
  - At this sample's precision, neither is viable under SPEC N6 at its declared horizon
    (§6).
  - The design omitted that invariant. It governs, and the forward read must meet it.
- **Costs are the binding constraint across the family.** The premium's gross Sharpe of
  0.753 nets to 0.274; the daily announcement return's 0.462 nets to 0.276.

## Sample and conventions

The engine setup matches the fundamentals arms: canonical engine, $100,000, frozen
dataset `data/frozen/demotion_b1`, folds `folds/demotion_b1.toml`, cost stack
`retail_v1`, modeled fills, price-return convention.

| set | scored sessions | span | excluded folds |
|---|---|---|---|
| monthly | 1,245 | 2021-06-29 → 2026-06-12 | fold 0: 10 decision periods, below the minimum of 12 |
| daily | 1,308 | 2021-03-30 → 2026-06-12 | none: daily decisions give fold 0 enough periods |

The two sets' samples differ by those 63 sessions, and no comparison across the sets was
registered.

**Runtime and memory:**

| run | time | peak memory |
|---|---|---|
| monthly arm | 65 s | 0.6 GB |
| `ear_event` | 610 s | 1.0 GB |
| `announcement_premium` | 633 s | 1.06 GB |

## Engine read (net of `retail_v1` costs)

| arm | set | net Sharpe | gross Sharpe | ann. return | ann. vol | max DD | charged cost (total) | turnover (total) | N_eff | held-book IC ± se | DSR vs set |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ear_monthly` | `earnings_v1` | 0.426 | 0.537 | 1.93% | 4.8% | 6.6% | 2.61% | 42 | 79.5 | 0.023 ± 0.018 | — (one row) |
| `ear_event` | `earnings_daily_v1` | 0.276 | 0.462 | 1.22% | 4.8% | 7.3% | 4.67% | 101 | 75.5 | 0.007 ± 0.004 | 0.734 |
| `announcement_premium` | `earnings_daily_v1` | 0.274 | 0.753 | 1.13% | 4.5% | 12.2% | 11.12% | 297 | 101.2 | 0.034 ± 0.004 | 0.733 |

- Column definitions are the characteristics read's.
- The DSR is undefined for a set of one row. For the replication route, the four
  replication tests stand in for it (SPEC §3 claim ladder).
- The paired Sharpe difference between the two daily arms is 0.002 a year, 95%
  interval [−1.30, 1.28].

## Replication read (SPEC §4)

rf = 4.21%. The public series is Chen and Zimmermann's daily value-weighted
`AnnouncementReturn` long−short, 1971-09-01 → 2024-12-31.

| | public t (years) | prior | ρ daily (weekly) | ρ × prior | cost drag | F3 | hurdle | in-house Sharpe [95%] | result |
|---|---|---|---|---|---|---|---|---|---|
| `ear_monthly` | 5.00 (53) | 0.410 (post-1996, t 2.17) | 0.556 (0.522) | 0.228 | 0.111 | 0.117 | 0.442 | 0.426 [−0.35, 1.19] | fails economics |
| combination | — | 0.474 (0.4 × 1.185, full window from 1972-08) | 0.770 | 0.365 | 0.090 | 0.275 | 0.351 | 0.818 [−0.07, 1.65] | fails economics |

- **The arm's other tests pass.** Tracking over 2021-06-29 → 2024-12-31 (883 sessions)
  clears the 0.5 floor; the publication bar passes; the arm is not contradicted.
- **Why its hurdle is 0.442.** Its own volatility is 4.8%, so under the 2× gross ceiling
  it can run at 9.5%, not 12%.
- **Combination details.**
  - Parts: `ear_monthly` and vol-scaled momentum.
  - The post-publication window is 9.74 years, so the SPEC haircut applies to its prior.
  - Tracking was measured over 440 sessions.
  - It ran at 11.9% volatility, mean scale 1.18, with costs of 1.07% a year.

**Cash yield on idle equity.** The registration records this beside the verdict. At a
zero hurdle (idle equity earning the bill rate), the arm and the combination both pass
all criteria.

## Daily arms: kill rule and diagnostics

| arm | net Sharpe | 95% interval | killed | daily corr. with vol-scaled momentum |
|---|---|---|---|---|
| `ear_event` | 0.276 | [−0.44, 0.96] | no | 0.56 |
| `announcement_premium` | 0.274 | [−0.88, 1.40] | no | −0.10 |

- The interval is studentized, with 42-session circular blocks and daily decisions.
- The correlations are over the 929 sessions both arms share with vol-scaled momentum.

## Horizon profile (SPEC §5.5)

Mean cross-sectional rank IC with the open-to-open return from the next open, over
index members, sessions from 2021-03-30. The t-statistic is Newey–West.

| signal | 1 | 2 | 5 | 10 | 21 | 63 |
|---|---|---|---|---|---|---|
| `ear_monthly` (≤ 126 sessions old) | 0.0074 (t 3.1) | 0.0086 (2.9) | 0.0069 (1.7) | 0.0072 (1.3) | 0.0066 (0.9) | 0.0032 (0.3) |
| `ear_event` (≤ 60 sessions old) | 0.0076 (3.1) | 0.0090 (3.0) | 0.0079 (1.9) | 0.0083 (1.5) | 0.0082 (1.1) | 0.0046 (0.5) |
| `announcement_premium` | 0.0062 (3.3) | 0.0067 (2.9) | 0.0090 (2.7) | 0.0109 (2.2) | 0.0148 (2.4) | 0.0089 (1.6) |

The announcement return's predictability is concentrated in the first two sessions. The
premium's is positive at every horizon, largest in size at 21 sessions and most precise
at 1.

## Breadth (N6)

The ceiling is IC × √(N_eff × 252 / horizon). The horizon is each component's declared
`horizon_bars` (21, 21, 5), fixed before the run. The bets per year follow from it, as
the B1 ceiling used 12 a year.

| arm | route | IC at horizon (se) | ceiling | ceiling, 95% range | ceiling at one-sided lower-95 IC | realized gross / net |
|---|---|---|---|---|---|---|
| `ear_monthly` | replication | 0.0066 (0.0074) at 21 | 0.202 | [−0.24, 0.65] | −0.171 | 0.537 / 0.426 |
| `ear_event` | discovery | 0.0082 (0.0074) at 21 | 0.248 | [−0.19, 0.69] | −0.119 | 0.462 / 0.276 |
| `announcement_premium` | discovery | 0.0090 (0.0034) at 5 | 0.643 | [0.17, 1.12] | 0.243 | 0.753 / 0.274 |

- **`ear_monthly`: realized is above the point ceiling.** N6 treats that as a leak or bug
  until shown otherwise.
  - Realized gross (0.537) exceeds the point ceiling (0.202) but lies inside its 95%
    range. At t 0.9 the 21-session IC cannot resolve the check.
  - The causality tests pass:
    - an announcement's value is unchanged by any return after its signal date;
    - it first appears at the close after day +1;
    - releases count only once accepted by the decision's close;
    - fills are at the next open.
  - The held-book IC (0.023) gives a ceiling of 0.71, against which the realized net is
    60%, the same ratio as the B1 evidence.
  - **Status:** flagged for the forward read; not resolved here.
- **The discovery arms fail the viability clause.** N6: "a ceiling below the after-cost
  hurdle at the one-sided lower-95 IC ⇒ not viable."
  - `ear_event`'s ceiling at the lower-95 IC is negative.
  - `announcement_premium`'s is 0.243, below its measured cost drag of 0.48 (gross 0.753
    less net 0.274) before any hurdle is added.
  - **Neither is viable on this sample's precision.** The forward read must establish
    viability on forward data before promotion.

## What the numbers say

1. **The public announcement-return premium survived publication.** The value-weighted
   series earned Sharpe 0.41 over 1997–2024 (t 2.17), against the design's central
   expectation of a small or zero prior.
2. **The standalone arm fails on volatility.** At 4.8% own volatility and a 2× gross
   ceiling, it must reach Sharpe 0.44 to beat T-bills.
3. **The combination is the closest miss the program has recorded.**
   - It clears the hurdle before costs and misses by 0.076 after them.
   - In-house, adding the announcement return to vol-scaled momentum gives net Sharpe 0.82
     at 11.9% volatility. The vol-momentum read's in-house scaled Sharpe was 0.66.
   - This is the same sample, so it cannot promote anything.
4. **Timing.**
   - The announcement return's information is front-loaded.
   - The daily arm captures it, but pays about 0.9% a year in costs; the monthly arm pays
     0.53%.
   - Its gross Sharpe (0.462) is below the monthly arm's (0.537), on samples 63 sessions
     apart.
5. **The premium is a gross effect that costs erase.**
   - IC is positive at every horizon, and it is nearly uncorrelated with momentum (−0.10).
   - But it turns over about 57 times a year, and costs of about 2.1% a year take its
     Sharpe from 0.753 to 0.274.
6. **Costs are the lever.**
   - The construction band's width is fixed at 0.0 on this engine path
     (`src/prism/evaluate.py:2515`), although the record reports `closed_form`. Nothing
     damps the re-sorting.
   - A working no-trade band, or the daily cost-aware construction step of SPEC §5.5
     (Gârleanu–Pedersen), acts directly on the binding constraint. Either would be a new
     registration.
7. **Cash yield on idle equity** turns both earnings replication results into passes, as
   it did for vol-scaled momentum.

## Disclosures

- **N6 was omitted from the design.**
  - The design's kill rule for the daily arms named only the interval. N6's
    discovery-route clause applies regardless and is read above at the horizons the
    components declared before the run.
  - The IC-horizon profile shows the announcement return is most precise at one session.
    Re-basing N6 on that horizon now would choose a horizon after seeing results, so it
    is not done.
- **A duplicate run was launched and stopped.**
  - The daily runs were started once in the background. A second copy was launched by
    mistake while the first appeared idle; the sandbox cannot see other commands'
    processes.
  - It was stopped during its freeze step, before it wrote a freeze record or any trial
    row. Each daily arm has exactly one started and one finished row.
- **`registered_read` changed after registration.**
  - It refused a one-arm registration at its pairwise comparison. It now records the
    comparison as skipped; nothing else it computes changed.
  - The change was made after the monthly arm ran and before its read. It is committed
    with this record.
- **The fidelity window** is 2021-06-29 → 2024-12-31. The design left its start to the
  admission rule.
- **The horizon profile** uses index members without the engine's price, liquidity and
  history screen.

## What closes and what remains

- **`earnings_v1` is closed:** one of one arm and one of one combination spent. No new
  trial targets it. Its forward kill rule runs after a promotion, and there is none.
- **`earnings_daily_v1` is closed to new arms.** `ear_event` and `announcement_premium`
  carry to the forward read of `docs/earnings_design.md` §6:
  - an interim read at ≥ 126 forward sessions, about 2027-03;
  - promotion at ≥ 252 forward sessions, about 2027-09. It must also pass N6 on forward
    data.
- **The forward reader is not built.**

## Limits

- The in-house sample is one regime family, 2021 → 2026. The replication fidelity overlap
  ends 2024-12.
- The in-house books are S&P 500 large caps. The public series is all-stock,
  value-weighted quintiles.
- Item 2.02 includes some non-earnings releases. The market leg is an equal-weighted
  member mean.
- The combination is an overlay of its parts' returns and costs, not one engine book.
- Fills are modeled.
