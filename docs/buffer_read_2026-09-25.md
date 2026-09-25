# Rank buffer — read record, 2026-09-25

**Kind:** read record for `docs/buffer_design.md`.

**Selection sets:** `buffer_monthly_v1` and `buffer_daily_v1`, discovery route. Each
declared two arms and spent both.

**Commits:** registration `47290e7`. It was preceded by `7504758` (the band refusal) and
`2feefe9` (the cash convention). Every record below carries `47290e7`.

**Records:**
- `results/2026-09-buffer-monthly/` and `results/2026-09-buffer-daily/`: `read.json`,
  `comparison.json`, `returns/`.
- `results/2026-09-buffer/buffer_read.json`.

**Trial logs:** `trials/buffer_monthly_v1.jsonl` and `trials/buffer_daily_v1.jsonl`, each
with 2 started and 2 finished.

## Verdict

**No arm is killed, and all four carry to the forward read. The buffer did not help:**
- It cut turnover by 12–36%.
- It gave up gross return roughly in proportion.
- The four paired net-Sharpe differences against the unbuffered records are −0.089 to
  +0.072 a year. None is distinguishable from zero (p 0.56–0.87).

On this sample, a 10%-entry, 20%-hold buffer is not the cost lever for these books.

## Sample and conventions

The engine setup matches every counterpart's: canonical engine, `account_state`, $100,000,
`data/frozen/demotion_b1`, folds `folds/demotion_b1.toml`, `retail_v1`, band `off`,
modeled fills and the price-return convention.

| book | excluded folds | scored sample | reason |
|---|---|---|---|
| monthly | 0–3 | 1,056 sessions from 2022-03-29 | momentum's admission rule, as in the characteristics read |
| daily | none | 1,308 sessions from 2021-03-30 | — |

Paired comparisons use the sessions both arms scored.

| arm | runtime | peak memory |
|---|---|---|
| `momentum_buffered` | 172 s | 0.7 GB |
| `ear_monthly_buffered` | 204 s | 0.7 GB |
| `ear_event_buffered` | 558 s | 1.0 GB |
| `premium_buffered` | 531 s | 1.2 GB |

## Buffered against unbuffered

Turnover and cost are per year, each over its own record's sample. The paired difference
is buffered − unbuffered net Sharpe on common sessions, annualized, with a studentized 95%
interval (42-session blocks, same resamples).

| arm | turnover/yr | cost/yr | gross Sharpe | net Sharpe (own sample) | paired difference [95%], sessions |
|---|---|---|---|---|---|
| `momentum_buffered` | 7.6 → 4.9 | 0.50% → 0.41% | 0.719 → 0.686 | 0.679 → 0.648 | −0.031 [−0.21, +0.14], 1,056 |
| `ear_monthly_buffered` | 8.5 → 7.5 | 0.53% → 0.49% | 0.537 → 0.742 | 0.426 → 0.638 | +0.072 [−0.17, +0.34], 1,056 |
| `ear_event_buffered` | 19.4 → 16.0 | 0.90% → 0.79% | 0.462 → 0.422 | 0.276 → 0.253 | −0.024 [−0.26, +0.28], 1,308 |
| `premium_buffered` | 57.1 → 49.6 | 2.14% → 1.90% | 0.753 → 0.643 | 0.274 → 0.185 | −0.089 [−0.41, +0.24], 1,308 |

**Reading `ear_monthly_buffered` correctly.** Its own-sample columns compare different
windows: the unbuffered record starts in 2021-06, the buffered one in 2022-03. On their
common sessions the unbuffered arm earned 0.566, so the buffer's effect is the paired
+0.072, not 0.638 − 0.426.

**Momentum.** The buffer cut its turnover by 36%, the largest cut, and it lost 0.03 of
Sharpe net. The live book's configuration keeps its construction.

## Kill rule

Studentized 95% interval for net periodic Sharpe; an arm is dead if the interval lies
entirely below zero.

| arm | net Sharpe | 95% interval | killed |
|---|---|---|---|
| `momentum_buffered` | 0.648 | [−0.17, 1.42] | no |
| `ear_monthly_buffered` | 0.638 | [−0.22, 1.50] | no |
| `ear_event_buffered` | 0.253 | [−0.54, 0.99] | no |
| `premium_buffered` | 0.185 | [−0.98, 1.45] | no |

## Breadth (N6)

The ceiling is IC × √(N_eff × 252 / horizon), at each component's declared horizon.

| arm | IC (se) at horizon | N_eff | ceiling [95% range] | at one-sided lower-95 IC | realized gross |
|---|---|---|---|---|---|
| `momentum_buffered` | 0.030 (0.024) at 21 | 44.1 | 0.69 [−0.39, 1.77] | −0.21 | 0.686 |
| `ear_monthly_buffered` | 0.0111 (0.0079) at 21 | 71.8 | 0.33 [−0.13, 0.78] | −0.06 | 0.742 |
| `ear_event_buffered` | 0.0082 (0.0074) at 21 | 74.5 | 0.25 [−0.19, 0.68] | −0.12 | 0.422 |
| `announcement_premium` signal, `premium_buffered` | 0.0090 (0.0034) at 5 | 105.3 | 0.65 [0.17, 1.14] | 0.25 | 0.643 |

**Where the ICs come from:**
- `momentum_buffered` uses the B1 evidence's 21-bar IC (SPEC §1). Momentum's horizon profile
  has not been computed on this engine.
- `ear_monthly_buffered` uses the same score's IC recomputed on this arm's own sessions
  (from 2022-03-29), with `research.scripts.earnings_read`'s functions, on 2026-09-25. It
  is not written under `results/`. Over the earnings read's longer window the IC is
  0.0066, which would put realized gross above the whole 95% range (upper end 0.62). The
  window difference, not the arm, accounts for that.
- The daily arms use the earnings read's horizon profile, which covers the same sessions.

**Realized above the point ceiling.** For `ear_monthly_buffered`, realized gross is above
the point ceiling but inside its range. N6 flags this as a possible leak or bug until shown
otherwise. The same causality tests apply as in the earnings read, and the buffer rule
reads only the component's own previous output (`tests/test_buffered_decile.py`). The
held-book IC (0.026) puts the ceiling at 0.76, which the realized gross reaches at 97%.
Flagged for the forward read.

**Viability on the discovery route.** No arm is viable on this sample's precision. At the
one-sided lower-95 IC, every arm's ceiling is below its cost drag plus the hurdle:
- three ceilings are negative;
- the premium's is 0.25, against a cost drag of 0.46.

The forward read must establish viability on forward data. This matches the live book's
own B1 evidence: its IC's one-sided lower-95 is −0.009.

## What the numbers say

1. **The buffer trades less and earns less.**
   - Holding names that have slipped from the top decile to the top quintile keeps weaker
     positions.
   - On momentum and both daily earnings arms, the gross return given up roughly equals
     or exceeds the cost saved.
   - The one positive difference, `ear_monthly_buffered`, is +0.07 with p 0.59.
2. **The premium's churn is structural, as the design expected.** Turnover fell only 13%,
   because its score moves with the calendar: names enter as their announcement nears and
   fall to the bottom after it. The buffer cannot hold a name whose rank has dropped past
   the hold band.
3. **Cost is not the only binding constraint.** For momentum at monthly cadence, cost is
   0.41–0.50% a year against a gross Sharpe near 0.7. The live book remains signal-bound,
   as its B1 evidence said.
4. **The earnings combination is not rescued.** A buffered monthly announcement arm would
   cut the combination's cost by roughly 0.04% a year on that part, against a 0.076 Sharpe
   miss.

## What closes and what remains

- **Both sets are closed to new arms.** All four arms carry to the forward read
  (`docs/buffer_design.md` §5): interim at ≥ 126 forward sessions, about 2027-03;
  promotion at ≥ 252, deflated against eight rows and required to meet N6 on forward data.
- **The forward reader is not built.** The same reader serves `earnings_daily_v1`, vol-scaled
  momentum's kill rule and `forward_v1`.

## Disclosures

- **The read script's annualized block for the paired intervals** carries arm A's Sharpe
  and the difference's bounds. The difference's point estimate is in
  `paired_difference.interval.difference` (periodic). The table above annualizes it by
  √252.
- **The background command that ran the daily arms exited 1.** Its final `tail` call took
  an option this shell's `tail` rejects with several files. Both arms had finished, each
  with one started and one finished row, and both records are complete.
- **Two N6 inputs were recomputed after the registration:** the IC on the monthly arm's own
  sessions, and momentum's IC taken from B1. Neither changes a kill or promotion decision
  here.

## Limits

- The in-house sample is one regime family, and it is the same sample every counterpart
  was read on.
- Costs are `retail_v1` with modeled fills.
- The buffer's 20% hold was not varied, and no other width was computed.
