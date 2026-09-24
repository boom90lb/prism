# Volatility-scaled momentum — read record, 2026-09-24

**Kind:** read record for `registrations/2026-09-vol-momentum.toml`
(`docs/vol_momentum_design.md`), the program's first replication-route read
(SPEC §4). **Registration commit:** `79ce54b`, which is also the code commit of
the run; the run refused to start until the registration was committed
unchanged. **Selection set:** `vol_momentum_v1`, budget 1, spent 1; trial log
`trials/vol_momentum_v1.jsonl` (started, finished, chain verified).
**Outputs:** `results/2026-09-vol-momentum/summary.json`, `series.csv`.
**Inputs:** all four matched their registered sha256. They are French momentum
and 3-factor daily files (202607 CRSP vintage), `momentum_control` returns, and
its run record's `costs.csv`.

## Verdict

**Promoted: all four criteria hold.** The live book's configuration with
Barroso–Santa-Clara constant-volatility scaling is eligible for the live book.
Changing the live book, wiring the scale, and replacing the sizing rule SPEC §6
names are owner acts; this record is the evidence for them.

The economics criterion passes by 0.018 Sharpe, and what that margin means for
money is the most important thing in this record (§3).

## 1. The four criteria

| criterion | computed | threshold | result |
|---|---|---|---|
| 1. Publication bar | scaled French momentum, gross, 1931-02 → 2026-07: Sharpe 0.901, t 8.97 over 98.9 years | t ≥ 3 over ≥ 20 years | pass |
| 1. Prior | Sharpe since 2015-04 (11.3 years): **0.465** (t 1.56 on its own) | > 0 | pass |
| 2. Fidelity | daily ρ between the scaled in-house book and the scaled reference: **0.895**, over 929 sessions, 2022-09-29 → 2026-06-12; weekly 0.910 | ≥ 0.75 | pass |
| 3. Economics | F3 = ρ × prior − cost drag = 0.416 − 0.047 = **0.368** | > hurdle 0.351 (rf 4.21% ÷ 0.12) | pass, by 0.018 |
| 4. No contradiction | in-house scaled net Sharpe 0.663, 95% interval [−0.194, +1.564] | upper ≥ 0.368 | pass |

The cost drag of 0.047 Sharpe is an annualized charged cost of 0.60% at 12.7%
volatility. That is the book's recorded daily costs times the scale, plus 10 bp
one-way on each scale change. rf is the mean French one-month T-bill rate over
the file's last 252 sessions (2025-08 → 2026-07), annualized.

## 2. What the scaling did in-house

| 2022-09-29 → 2026-06-12 | scaled | unscaled |
|---|---|---|
| annualized net Sharpe | 0.663 | 0.681 |
| annualized return | 8.42% | 8.38% |
| annualized volatility | 12.7% | 12.3% |
| scale: mean (min – max) | 1.08 (0.68 – 1.41) | 1 |

Over this window the rule was close to neutral. The book ran near its 12%
target, so the scale hovered around 1. The in-house sample contains no
momentum crash, and the rule's value is concentrated in crashes. On French
momentum it halved the 1932 and 2009 losses
(`docs/public_factor_lab_read_2026-09-24.md`). The promotion rests on that
public record plus the 0.895 fidelity, as the replication route intends. It
does not rest on anything the in-house window shows.

## 3. What 0.018 of margin means

F3 = 0.368 is this route's central estimate of the book's net Sharpe going
forward. That estimate is the post-publication prior, discounted for tracking
and costs. It is not the 0.66 of the in-house window.

At the book's ~12.7% volatility, a Sharpe of 0.368 is about **4.7% a year**
of total return. The criterion compares that with T-bills at 4.2%, because the
cost stack assumes idle cash earns nothing
(`examples/reference/costs/retail_v1.toml`, `cash_income_bps_annual = 0.0`).
Read literally, **the expected edge over holding T-bills is about half a
percent a year**. A bill rate ~20 bp higher would have failed the criterion.

That assumption is also where the largest lever sits. A dollar-neutral book at
1× gross leaves roughly all of its equity as a cash balance. If that cash
earns the bill rate, through the broker's cash interest or by holding the
collateral in a T-bill fund within Reg T limits, the book earns about 4.7% *on
top of* the bills. That is roughly a tenfold difference in the edge, and it
comes from treasury management, not from the signal.

Whether the deployed account can earn the bill rate on its idle equity is an
operational fact this program has not established. It is now the most valuable
unknown in the deployment path.

## 4. What this changes

- **SPEC §6 deployment gate:**
  - **Promoted configuration: met.** This record satisfies "a configuration is
    promoted under §4 (either route)".
  - **Sizing rule: blocked.** §6 still names the crash-conditional de-gross,
    which the factor lab contradicted. This promoted rule is the evidence-backed
    replacement. Substituting it, and wiring it through the live gross-scale
    hook (`src/prism/live/daily.py`, refresh bars only), are owner acts.
  - **Clean paper sessions: 12 of 21.** The scheduler has been dark since
    2026-08-14; re-arming it is the owner's act.
  - **Risk profile: not chosen.**
- **Forward kill rule is live** (`docs/vol_momentum_design.md` §4). It reads
  `forward_v1`'s FB stream scaled by this rule. The first read is at ≥ 126
  forward sessions (about 2027-03-24), then every 63 sessions. It kills if the
  forward Sharpe interval lies entirely below zero, the forward drawdown
  exceeds 25%, or daily correlation with scaled French momentum falls below
  0.5. The reader for it is not yet built and is due before that date.
- **Cash yield.** A documented answer is needed on whether idle equity earns
  the bill rate: the broker's cash-interest terms, or the margin treatment of a
  T-bill fund alongside the book. It moves the expected edge by about 4% a
  year. A changed cash assumption would change this criterion's hurdle, and
  that would be a new registration.

## 5. Limits

- The in-house scaled series is an overlay on the recorded book (returns and
  costs × scale, plus scale-change costs), not an engine re-run
  (`docs/vol_momentum_design.md` §3).
- The prior's post-publication window is 11.3 years with t 1.56 on its own.
  The route leans on the 99-year record for the bar and on those 11 years for
  the level.
- The reference is an all-CRSP paper portfolio. The 0.895 fidelity measures
  how closely the S&P 500 book follows it. It does not make them the same
  book.
- Modeled fills throughout.
