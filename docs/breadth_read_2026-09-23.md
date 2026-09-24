# Breadth — read record, 2026-09-23

**Kind:** read record for `registrations/2026-09-breadth.toml`
(`docs/breadth_design.md`). **Selection set:** `breadth_v1`.
**Registration commit:** `ef2256b`, before the combined dataset existed and
before any mid-cap return was read. **Budget:** 3 declared, 3 spent.
**Records:** `results/2026-09-breadth/comparison.json`, `read.json`,
`returns/`. **Trial log:** `trials/breadth_v1.jsonl` (3 started, 3 finished).
**Dataset:** `data/frozen/breadth_v1` (`data_id d8e74fc4…`), 1,065 names,
1,621 sessions.

## Verdict

**The set closes with no arm killed. The union raises breadth by a third and
does not raise net Sharpe. On this sample the breadth lever, pulled with free
mid-cap data, does not pay.**

## Universe as built

- S&P 500: the certified membership and its 574 names, unchanged. Per-date
  membership in the new dataset equals the certified dataset's on every
  sampled date (471 on 2020-06-01, 507 on 2026-06-15).
- S&P 400: 53 phantom intervals resolved (5 promotions, 6 demotions, 42
  unresolved and dropped; `data/universe/sp400_exits_2026-09-23.json`).
  Twelve Data resolves 570 of 665 window-active members, coverage 0.857; the
  95 skips are the measured mid-cap survivorship leak, against the S&P 500's
  59 of 633. Exclusive S&P 400 membership with bars runs from 340 names
  (2020-06-01) to 391 (2026-06-15); it never overlaps the S&P 500.

**Parity (Run).** `sp500_control`, read from this dataset through index id
`sp500`, reproduces `characteristics_v1`'s `momentum_control` bar for bar:
every daily return and every target weight is identical.

## Results

Same conventions as the characteristics read: canonical engine, $100,000,
folds 0-3 excluded by the shared admission rule, 1,056 scored sessions
(2022-03-29 to 2026-06-12), modeled fills.

| arm | net Sharpe | gross Sharpe | ann. return | ann. vol | max DD | charged cost | N_eff | names held | held-book IC ± se | DSR vs set |
|---|---|---|---|---|---|---|---|---|---|---|
| `sp500_control` | 0.679 | 0.719 | 7.95% | 12.4% | 16.3% | 2.09% | 46.7 | 96 | 0.049 ± 0.041 | 0.900 |
| `sp400_only` | 0.575 | 0.633 | 6.86% | 13.0% | 17.5% | 3.15% | 61.0 | 72 | 0.046 ± 0.034 | 0.859 |
| `sp900_union` | 0.638 | 0.689 | 7.27% | 12.2% | 16.2% | 2.57% | **63.8** | 169 | 0.050 ± 0.036 | 0.885 |

Paired studentized differences (per bar; declared bootstrap, seed 20260923):

| difference | lower | upper | p |
|---|---|---|---|
| control − sp400 | −0.030 | 0.039 | 0.72 |
| control − union | −0.017 | 0.018 | 0.78 |
| sp400 − union | −0.024 | 0.016 | 0.71 |

No interval excludes zero; no arm is killed.

## Reading

- **Breadth moved as the fundamental law wanted.** The union holds 169 names
  against 96 and its N_eff is 63.8 against 46.7, +37%. At an unchanged IC the
  ceiling `IC·√N_eff` would rise by 17%.
- **Edge per name fell by more.** Mid-cap momentum earned less gross (0.633
  against 0.719) and cost half again as much to trade (3.15% against 2.09% of
  equity over the sample, the wider spread buckets doing what they are
  declared to do). The union's net Sharpe lands between its parts.
- **The DSRs near 0.9 are against three same-sample rows** and carry no
  promotion weight.
- **The mid-cap panel is optimistic in a known direction.** Its coverage
  (0.857) is lower than the S&P 500's (0.907), and the missing names are the
  ones that left by delisting or acquisition. A panel with them would be more
  likely to lower the mid-cap result than raise it.

This is not cell C1 of `docs/replication_preregistration.md` and cannot be
cited for it.
