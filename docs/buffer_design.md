# Rank buffer — design and registration (`buffer_monthly_v1`, `buffer_daily_v1`)

**Status: registered 2026-09-25.** The registration is the commit that adds this file
with:
- `registrations/2026-09-buffer-monthly.toml` and `registrations/2026-09-buffer-daily.toml`;
- the books `examples/buffer/monthly.toml` and `examples/buffer/daily.toml`;
- the buffered decile rule (`prism.portfolio.construct.buffered_decile_row`; spec rule
  `decile_long_short_buffered`) and its tests;
- the read `research/scripts/buffer_read.py`.

At registration, no buffered book's weights, turnover or return had been computed on real
data.

## 1. What is registered

Four arms. Each is a registered decile book with its decile rule replaced by a buffered
one, every other setting unchanged:

| selection set | arm | signal | decisions | unbuffered counterpart |
|---|---|---|---|---|
| `buffer_monthly_v1` | `momentum_buffered` | 12-1 momentum, the live book's signal | every 21 sessions | `momentum_control`, `characteristics_v1` |
| `buffer_monthly_v1` | `ear_monthly_buffered` | announcement return, ≤ 126 sessions old | every 21 sessions | `ear_monthly`, `earnings_v1` |
| `buffer_daily_v1` | `ear_event_buffered` | announcement return, ≤ 60 sessions old | every session | `ear_event`, `earnings_daily_v1` |
| `buffer_daily_v1` | `premium_buffered` | minus the days to the next expected announcement | every session | `announcement_premium`, `earnings_daily_v1` |

**Route: discovery.** The buffer was chosen after the earnings reads found
decile-membership churn binding. That makes it a choice among implementations made on
in-house results (SPEC §4). The in-house read can kill an arm and never promote one;
promotion is the forward read (§5).

## 2. Why, and what to expect

**Costs decide these books.** The facts that bind (`docs/earnings_read_2026-09-24.md`):
- The announcement premium's gross Sharpe of 0.753 nets to 0.274 at 57× turnover a year.
- The daily announcement return's gross 0.462 nets to 0.276.
- The announcement-return combination with vol-scaled momentum cleared its hurdle before
  costs and missed it after them, by 0.076.
- The live book's evidence (B1) charged 3.7% of cost over its sample.

**The no-trade band is not the fix.** It is 0.1–0.2% of equity wide and leaves
decile-membership churn untouched (`docs/construction_band_audit_2026-09-24.md`, F5).

**The rule** is the buy/hold spread of Novy-Marx and Velikov (2016, *Review of Financial
Studies*):
- a name enters the long leg in the top 10% of scores and is held while it stays in the
  top 20%;
- the short leg mirrors this at the bottom.

The 10% entry is every counterpart's decile. The 20% hold is pinned here and not tuned: no
other width was computed.

**Expectations:**
- Turnover should fall most where ranks persist between decisions: momentum, and the
  announcement return, whose score for a name is constant between announcements.
- The premium's score moves by the calendar. A name climbs into the top decile as its
  announcement nears and falls to the bottom after it, so the buffer delays few of its
  exits. Expect little change there.
- A buffer holds names whose signal has weakened, so gross return can fall. The read
  measures both sides.

## 3. Implementation

**The rule.**
- The rule's state is the component's own previous sub-targets (`ComponentState.sub_targets`,
  passed by `prism.composition`). It never sees fills or held shares, so it is
  deterministic in the scores.
- Each leg is equal-weighted at half the gross, so the book stays dollar-neutral while
  the legs' name counts differ.
- A name without a finite score is flat.
- With no previous legs it reproduces the plain decile rule exactly
  (`tests/test_buffered_decile.py`).
- The live daily cycle (`prism run`) refuses the rule (`LaterStage`) until it is built
  there.

**Engine runs.** Identical to the counterparts:
- frozen dataset `data/frozen/demotion_b1`, folds `folds/demotion_b1.toml`;
- cost stack `retail_v1`, so the buffer is the only difference;
- execution profile `execution_alpaca`, canonical engine, `account_state` coupling,
  $100,000;
- the same eligibility screen;
- construction with the band declared `off`, which the engine now requires on this path.

The monthly book's admission is set by its components, as every book's is. Its sample
can therefore differ from a counterpart's, and paired comparisons use the sessions both
scored.

**Mechanics check.** The freeze step was run to scratch; it computes scores, not weights
or returns. The monthly book produced 63 decision rows in 49 s and the daily book 1,308
rows in 216 s, both with no warnings.

## 4. The in-house read (`research/scripts/buffer_read.py`)

Per arm:
- **Kill.** The studentized 95% interval for net periodic Sharpe (42-session circular
  blocks, 2,000 resamples, seed 20261001). An arm whose interval lies entirely below zero
  is dead.
- **Reported beside it:**
  - turnover and charged cost a year, buffered against the counterpart's committed record;
  - the paired studentized interval for (buffered − unbuffered) net periodic Sharpe on
    their common sessions.
- Each registration's `registered_read` also reports N_eff and the held-book IC.

## 5. Forward read (promotion)

It uses the geometry and data rules of `docs/forward_design.md` §2–3: the first test
window opens on or after 2026-09-25. The announcement panel is rebuilt at each read's
as-of by `prism.scripts.build_earnings_panel`.

- **Interim**, at ≥ 126 forward sessions: the kill rule of §4 on forward sessions only. It
  cannot promote.
- **Promotion**, at ≥ 252 forward sessions. All of these must hold on forward sessions
  only:
  1. `net_edge` under bucket spreads;
  2. DSR above 0.5, deflated against **eight** rows: the four buffered arms and their four
     unbuffered counterparts, so the choice between them is paid for;
  3. periodic net Sharpe above the hurdle of SPEC §3 as amended on 2026-09-25,
     (rf − y·max(0, 1 − L/E)) / vol, with the account's cash yield y and the book's
     recorded average L/E;
  4. N6 viability at the declared horizon;
  5. the live monitor not in contradiction.
- **The forward reader is not built.** It is due before the first interim read, about
  2027-03.
- **A promoted `momentum_buffered` only becomes eligible** to replace the live book's
  construction. Replacing it is the owner's act (SPEC §6).

## 6. Run order

```
python -m prism.scripts.construction_experiment --registration registrations/2026-09-buffer-monthly.toml \
    --dataset data/frozen/demotion_b1 --freeze
python -m prism.scripts.construction_experiment --registration registrations/2026-09-buffer-monthly.toml \
    --dataset data/frozen/demotion_b1 --arm <arm> --capital 100000   # momentum_buffered, ear_monthly_buffered
python -m research.scripts.registered_read --registration registrations/2026-09-buffer-monthly.toml \
    --dataset data/frozen/demotion_b1 --out results/2026-09-buffer-monthly
# the same three steps for registrations/2026-09-buffer-daily.toml
#   (arms ear_event_buffered, premium_buffered; --out results/2026-09-buffer-daily)
python -m research.scripts.buffer_read
```

## 7. Prior exposure, disclosed

- **Every counterpart's in-house result is known**: net and gross Sharpe, turnover, cost
  and N_eff. This design exists because of them.
- **The band audit's turnover replay** used the counterparts' recorded target paths.
- **The buffered rule** has run only on synthetic scores (its tests) and through the
  score-only freeze step. No buffered weight, turnover or return on real data was
  computed before this registration.
- **The 20% hold width** was taken from the rule's published form and not searched.

## 8. Limits

- The in-house sample is the same one every counterpart was read on.
- Costs are modeled under `retail_v1`, with modeled fills.
- The buffer changes the legs' name counts. Concentration and N_eff are reported, not
  constrained.
- The rule is not yet in the live daily cycle.

## 9. What this does not do

- It does not change the live book, the cost stack of any earlier registration, or any
  closed set's verdict.
- It does not search buffer widths or apply the buffer to other signals: each is a new
  registration.
- It does not build the Gârleanu–Pedersen construction step
  (`docs/aim_portfolio_preregistration.md`), which would need the convex constructor and
  a registered alpha and risk model.
