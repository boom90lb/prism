# R2 design + pre-registration (drafted 2026-07-05, before any counted run)

Scope (SPEC §7.3/§13 R2): closed-form no-trade band, participation gate wired
into construction, per-bucket spread calibration (I-9). Everything lands
default-off (frozen-v1 parity); enabling any piece is a counted ledger trial.

## 1. Closed-form band (replaces the sqrt heuristic as an OPTION, not in place)

Formula (Martin 2012 proportional-cost tracking asymptotic — the cube-root
law; the SPEC's "closed-form Gârleanu–Pedersen rule" in the proportional-cost
setting):

    half_width_i = ( (3/2) * c_i * sigma2_target_i / gamma_risk )^(1/3)

- `c_i` — round-trip proportional cost for name i: 2 × (commission_bps +
  spread_bps_i) / 1e4, with spread_bps_i from the bucket schedule (§3) when
  active, else the flat spread.
- `sigma2_target_i` — daily variance of the name's *target-weight changes*,
  estimated on the formation window only (causal): var(diff(w*_i)) over
  formation bars. Degenerate (<2 obs, NaN) → band 0 (disabled, warned via the
  shared coercion policy).
- `gamma_risk` — pre-registered constant **1.0**. NOT fitted, NOT swept. If it
  is ever swept, every value is a counted trial.

Contrast with the shipped heuristic `cost_aware_band` (gamma·sqrt(c/kappa)):
cost enters cube-root (shallower), and the driver is target-weight volatility
rather than 1/kappa. Both remain available; `--band_mode closed_form` is a
new config hash → new trial.

## 2. Participation gate wiring

`participation_capped_targets` (exists, tested, currently unwired) is applied
per test day INSIDE the online loop, after `step_no_trade_band` + `cap_book`,
against the held weights and that day's trailing dollar volume:

    capped = participation_capped_targets(held, target, dollar_volume_asof,
                                          aum=initial_capital,
                                          max_participation=P, adv_floor=0.0)

- `P` via `--max_participation`, default **0.0 = off** (frozen-v1 parity).
  Pre-registered enable value for the R2 run: **0.05** (5% ADV).
- `adv_floor=0.0` deliberately: flooring ADV *up* would loosen the cap for
  illiquid names (the pricer's `adv_floor_dollars` serves impact pricing and
  stays divergent by design — reconciliation is documentation, not code).
- Gate BEFORE cost accounting so charged turnover reflects gated trades.

## 3. Per-bucket effective spread (I-9)

Buckets by formation-window median dollar volume (same screen statistic the
WFO already computes), conservative-upper schedule, recorded in every claim
packet:

    ADV >= $500M   : 1.0 bps     $100M-$500M : 2.0 bps
    $25M-$100M     : 5.0 bps     < $25M      : 10.0 bps

- `--spread_mode {flat,bucket}`, default **flat** (frozen-v1 parity).
- Mechanics: per-name spread vector enters `_cost_values` elementwise
  (commission stays scalar); `backtest_target_weights` accepts an optional
  `spread_bps_per_name` aligned to columns. Flat mode must be bit-identical
  to today (parity pinned by test).
- Basis recorded in claim packet: "conservative upper proxy, no fill data
  yet" per I-9; replaced by paper/live fill calibration when fills exist.

## 4. Pre-registered R2 trial set (counted; ≤ the §10 budget of ~30)

After the Phase-A re-run lands, exactly these configs run, in order:

1. closed_form band, flat spread, no gate      (isolates the band)
2. closed_form band, bucket spread, no gate    (band under honest costs)
3. closed_form band, bucket spread, P=0.05     (the full R2 stack)
4. sqrt-heuristic band (existing cost_aware), bucket spread, P=0.05
   (is the closed form actually better than the heuristic?)

`--design_trials` on each run counts the FULL design grid above plus the two
Phase-A re-measurements. No other knob moves; gamma_risk, bucket edges,
schedule values, and P are fixed by this document.

## 5. Exit criteria (SPEC §13 R2)

- Turnover responds monotonically to the band parameter (test: synthetic
  panel, band grid, realized turnover non-increasing).
- Participation gate property tests already pin cap semantics.
- Flat-spread parity test: spread_mode=flat reproduces current numbers
  bit-for-bit.
- §10 adjudication happens on trial 3's claim packet (deflated against the
  ledger + design_trials), with cost_toll and capacity_curve attached.
