# Fundamentals — read record, 2026-09-24

**Kind:** read record for `registrations/2026-09-fundamentals.toml` and
`registrations/2026-09-fundamentals-replication.toml` (`docs/fundamentals_design.md`).
**Selection set:** `fundamentals_v1`. **Route:** replication (SPEC §4).
**Registration commit:** `8ded2b2` (the commit that added both registrations, before
any arm ran); every record below carries it as its code commit. **Budget:** six arms
declared, six spent; one combination declared, one spent. **Records:** six run
records read by `research/scripts/registered_read.py` into
`results/2026-09-fundamentals/` (`read.json`, `comparison.json`, `returns/`); the
replication read by `research/scripts/fundamentals_replication.py` into
`results/2026-09-fundamentals/replication/` (`replication.json`,
`combination_series.csv`). **Trial logs:** `trials/fundamentals_v1.jsonl` (6
started, 6 finished), `trials/fundamentals_v1_combination.jsonl` (1 started, 1
finished).

## Verdict

**Nothing promotes: no arm and not the combination. The set closes, and no arm is
killed (every in-house interval covers its own F3).** The binding constraint is the
public prior times tracking, before any cost. For every arm, and for the combination,
ρ × prior is below the hurdle with cost set to zero.

## Sample and conventions

The engine setup matches the momentum control of the characteristics read:

- canonical engine, $100,000;
- frozen dataset `data/frozen/demotion_b1` (`data_id bedf8c4c…`), folds
  `folds/demotion_b1.toml`;
- cost stack `retail_v1`, `research_vintage` visibility;
- modeled fills, price-return convention.

The characteristic panel is `data/frozen/fundamentals_v1/panel.parquet` (sha256
`06ea92de…5df3`). The freeze step reproduced the dry run's component hashes.

**Scored sample: 1,245 sessions, 2021-06-29 → 2026-06-12.** The shared admission
rule excluded fold 0 only: its training pool held 10 decision periods, below the
declared minimum of 12. The design's statement of the sample (1,056 sessions from
2022-03-29, folds 0–3) was wrong. It was carried over from the characteristics read,
where residual momentum's longer warm-up excluded folds 0–3. The rule that ran is the
registered one. The design's statement is wrong and the rule is right; the
consequence is under *Disclosures*.

## Engine read (in-house, net of `retail_v1` costs)

| arm | net Sharpe | gross Sharpe | ann. return | ann. vol | max DD | charged cost (total) | N_eff | held-book IC ± se | DSR vs set |
|---|---|---|---|---|---|---|---|---|---|
| `be_me` | 0.111 | 0.154 | 0.57% | 7.9% | 15.0% | 1.68% | 17.0 | 0.010 ± 0.035 | 0.460 |
| `gp_at` | 0.036 | 0.086 | 0.03% | 6.2% | 10.5% | 1.52% | 19.4 | 0.026 ± 0.025 | 0.395 |
| `at_gr1` | 0.116 | 0.178 | 0.50% | 5.8% | 16.9% | 1.75% | 37.8 | −0.000 ± 0.024 | 0.465 |
| `oaccruals_at` | 0.038 | 0.115 | 0.07% | 4.4% | 12.6% | 1.68% | 31.0 | −0.021 ± 0.016 | 0.396 |
| `niq_su` | **0.356** | 0.459 | 1.59% | 4.7% | 7.3% | 2.41% | **68.6** | 0.032 ± 0.019 | **0.671** |
| `chcsho_12m` | 0.192 | 0.262 | 0.83% | 5.0% | 5.9% | 1.73% | 32.9 | 0.031 ± 0.022 | 0.531 |

Column definitions are those of `docs/characteristics_read_2026-09-23.md`. DSR deflates
each arm against this set's six rows. Every arm is at `net_edge` (net > 0) and no
higher. The 15 pairwise Sharpe-difference intervals all straddle zero, with p from
0.50 to 1.00 (`comparison.txt`).

## Replication read, per arm (SPEC §4)

rf = 4.21% (French RF, last 252 sessions of the pinned 202607 file). The hurdle is
rf ÷ min(12%, 2 × own vol). Fidelity is the daily correlation with JKP's
direction-adjusted capped-value-weight series over 2021-06-29 → 2025-12-31 (1,133
sessions).

| arm | public t (years) | post-pub Sharpe = prior | ρ daily (weekly) | ρ × prior | cost drag | F3 | hurdle | in-house Sharpe [95%] | fails |
|---|---|---|---|---|---|---|---|---|---|
| `be_me` | 2.48 (75) | 0.105 | 0.805 (0.823) | 0.084 | 0.043 | 0.041 | 0.351 | 0.111 [−0.67, 0.95] | bar, economics |
| `gp_at` | 4.29 (75) | 0.426 | 0.567 (0.628) | 0.242 | 0.050 | 0.192 | 0.351 | 0.036 [−0.79, 0.83] | economics |
| `at_gr1` | 3.03 (74) | 0.221 | 0.725 (0.764) | 0.160 | 0.061 | 0.099 | 0.365 | 0.116 [−0.78, 1.16] | economics |
| `oaccruals_at` | 7.26 (74) | 0.387 | 0.499 (0.420) | 0.193 | 0.077 | 0.116 | 0.476 | 0.038 [−0.74, 0.84] | fidelity, economics |
| `niq_su` | 4.97 (61) | 0.452 | 0.290 (0.342) | 0.131 | 0.103 | 0.028 | 0.444 | 0.356 [−0.33, 1.15] | fidelity, economics |
| `chcsho_12m` | 4.00 (103) | 0.219 | 0.534 (0.568) | 0.117 | 0.071 | 0.046 | 0.425 | 0.192 [−0.49, 0.85] | economics |

Every source predates 2016, so every prior is the post-publication Sharpe. No arm is
contradicted: each interval covers its F3.

## Combination (SPEC §4, Combinations)

- **Parts** (passing criteria 1 and 2): `at_gr1`, `chcsho_12m`, `gp_at`, plus
  `vol_momentum`. Equal risk at 12%, re-scaled to 12% under the total-gross ceiling of 2.
- **Prior:** the public combination's Sharpe from 2015-04 to 2025-12-31 (10.7
  years): **0.393**. Over its full window (1952–2025) it is 1.31.
- **Fidelity:** 0.548 daily (0.603 weekly), 690 sessions, 2023-04-03 → 2025-12-31.
- **Economics:** ρ × prior 0.215 − cost drag 0.128 = **F3 0.087** against a hurdle of
  0.351. **Fails.**
- **In-house:** net Sharpe 0.406 at 6.2% vol, 802 sessions (2023-04-03 → 2026-06-12),
  95% interval [−0.54, 1.35]. Not contradicted.
- **Scaling:** mean combination scale 1.14. Total gross was at the ceiling on 65% of
  sessions.

## Why nothing promotes

1. **Prior × tracking, before cost.**
   - Across the arms, ρ × prior ranges from 0.08 to 0.24 against hurdles of 0.35 to
     0.48, with cost set to zero.
   - Cost drag (0.04–0.10) widens the gaps; it does not create them.
   - For the combination, cost is about half the gap: 0.215 falls to 0.087 against
     0.351.
2. **The fundamental parts earned little after 2015.** The table gives each part's
   public Sharpe over the combination's prior window, scaled to 12% by the registered
   rule.

   | part | Sharpe, 2015-04 → 2025-12 |
   |---|---|
   | `gp_at` | 0.469 |
   | `niq_su` | 0.354 |
   | `oaccruals_at` | 0.172 |
   | `chcsho_12m` | 0.044 |
   | `at_gr1` | −0.176 |
   | `be_me` | −0.224 |
   | `vol_momentum` | 0.401 |

   - Momentum alone has 0.401 over this window. The registered combination has 0.393:
     adding the three fundamental parts lowered the public prior, because two of them
     earned nothing.
   - The full-window 1.31 shows what diversification across these themes delivered
     historically. The post-publication rule reads only the last decade, as it should.
3. **The gross ceiling caps the diversified book's volatility.**
   - The fundamental books run at 4–8% unscaled vol, so each part needs a scale near
     2 to reach 12%.
   - After averaging, the combination re-levers only to scale 1.14 before hitting the
     2× gross ceiling, which leaves it at 6.2% vol.
   - At 6.2% vol, beating T-bills takes a Sharpe of 0.68, not the registered 0.351
     (see *Disclosures*).

**Cash yield on idle equity.** The registration records this beside the verdict. If
idle equity earned the bill rate, the hurdle would be zero. Then `at_gr1`,
`chcsho_12m`, `gp_at` and the combination would pass all their criteria.

The expected excess over bills would still be small. It is F3 × the volatility the
book can run at:

| book | F3 × volatility | excess over bills |
|---|---|---|
| `gp_at` | 0.192 × 12% | 2.3% a year |
| `at_gr1` | 0.099 × 11.5% | 1.1% a year |
| `chcsho_12m` | 0.046 × 9.9% | 0.5% a year |
| combination | 0.087 × 6.2% realized | 0.5% a year |

The cash convention is the owner's decision (SPEC §1, vol-scaled momentum). This read
does not change it.

## Disclosures

- **Fidelity window.** The design stated fidelity over 2022-03-29 → 2025-12-31. The
  read used the full overlap of the scored sample, from 2021-06-29 (see *Sample*).
  - On the stated window the per-arm correlations are:

    | arm | stated window | full overlap |
    |---|---|---|
    | `be_me` | 0.786 | 0.805 |
    | `gp_at` | 0.458 | 0.567 |
    | `at_gr1` | 0.711 | 0.725 |
    | `oaccruals_at` | 0.517 | 0.499 |
    | `niq_su` | 0.381 | 0.290 |
    | `chcsho_12m` | 0.515 | 0.534 |

  - Two outcomes would flip: `gp_at` would fail fidelity and `oaccruals_at` would
    pass. No per-arm verdict changes, because every arm fails economics either way.
  - The combination's parts would change to `at_gr1`, `chcsho_12m`, `oaccruals_at`
    and `vol_momentum`. That set's public prior (2015-04 → 2025-12) is **0.339**.
    F3 cannot exceed the prior, so this combination fails the 0.351 hurdle even at
    ρ = 1 and zero cost.
  - The bound uses public data only. No second combination was evaluated and no row
    was added.
  - To recompute: `fundamentals_replication.combine` over `scale_public` of the pinned
    JKP files plus `momentum_part` of the pinned momentum record, Sharpe of `net`
    from 2015-04-01.
- **The combination's hurdle is lenient.**
  - The registered hurdle is rf ÷ 12%. It assumes the combination reaches its target.
  - It did not: realized vol was 6.2%, set by the gross ceiling. The per-arm hurdle
    accounts for this ceiling; the combination's does not.
  - The verdict is unaffected.
- **Gross drifts between re-sets.** The ceiling is enforced at each 21-session re-set.
  In between, realized total gross drifted from 1.87 to 2.20.
- **Vol-scaled momentum's prior depends on the window's end.**
  - Its promoted record (`docs/vol_momentum_read_2026-09-24.md`) used French data
    through 2026-07-31 and reported a prior of 0.465.
  - Cut at 2025-12-31 to match the JKP window, the same series gives 0.401.
  - At that prior its F3 would be 0.895 × 0.401 − 0.047 = 0.312, below the 0.351
    hurdle. Its 0.018 margin rests on January–July 2026.
  - The promotion stands under its registration, which pinned that file. This is
    recorded so the forward kill rule's first read is weighed with it.
- **`niq_su` is not the published characteristic as implemented.**
  - It is the best in-house arm (net 0.356, DSR 0.671, N_eff 68.6), but it tracks JKP
    at 0.29.
  - The in-house definition is year-on-year change in quarterly net income over its
    8-quarter standard deviation. JKP's standardized unexpected earnings differs in
    construction, and a monthly decile book captures little of a drift concentrated
    around announcements.
  - A same-sample result cannot promote on either route. It is evidence for an
    event-time earnings registration using 8-K item 2.02 dates (SPEC §5.3, §5.5),
    which would be a new registration.

## What closes and what remains

- **Closed.** `fundamentals_v1` is closed: six of six arms and one of one combination
  spent. No new trial targets it.
- **Forward kill rule.** Nothing was promoted, so the registered rule has no subject
  to monitor.
- **Remaining clocks.** Vol-scaled momentum's forward kill rule (first read about
  2027-03-24) and `forward_v1` are unchanged.

## Limits

- The in-house sample is 2021-06 → 2026-06, one regime family.
- The in-house books are S&P 500 large caps. JKP's references are all-stock,
  capped-value-weight terciles, and fidelity measures that gap.
- The combination overlays the parts' returns and costs; it is not one engine book.
  Netting across parts would lower its cost drag, but it cannot raise ρ × prior
  (0.215) above the hurdle.
- Gross profitability covers about 61% of names; most financials report no cost of
  revenue.
- Fills are modeled and returns use the price-return convention.
- **How the numbers were obtained.**
  - The tables come from the committed run records.
  - The window sensitivities and the public-only bound were run on 2026-09-24 and
    are not written under `results/`. They can be recomputed as described above.

## Addendum 2026-09-25: the cash-yield paragraph, corrected

"If idle equity earned the bill rate, the hurdle would be zero" assumed the short-sale
proceeds earn bills. Under the cost stack's terms (`short_proceeds =
"collateral_no_rebate"`) and a retail account's, only the equity the long side leaves unused
earns: 1 − L/E. SPEC §3 as amended on 2026-09-25 gives the hurdle as
(rf − y·max(0, 1 − L/E)) / vol.

- **The arms.** `at_gr1` and `chcsho_12m` run at a scale of 2 to reach their volatility, so
  L = E and nothing is idle; their hurdle is the registered one. `gp_at` runs at scale 1.94,
  leaving 3% idle, which moves its hurdle by about 0.01 (0.009 at a 3.3% yield, 0.011 at
  4.21%).
- **The combination** runs at the 2× gross ceiling, so its hurdle is the registered one.
- **So none of the four passes stated in the paragraph holds under the corrected
  convention.** The verdict recorded above (nothing promotes) is unchanged.
