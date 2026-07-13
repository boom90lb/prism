# B2 anti-netting: the cap-rescale wiggle, confirmed

**Status: uncounted mechanics diagnostic, recorded 2026-07-11.** This note
settles the hypothesis `docs/certifications/001` §8 recorded as
"(hypothesis, unverified)". It claims no economics, enters no selection set,
writes no ledger row, and appeals nothing — the residual program's 17-row
ledger and its verdict are untouched (cert 001 §9). The certification itself
scheduled this work: "Worth resolving before any future multi-sleeve book;
per pre-registration it buys B2 no re-run here."

## 1. The finding it explains

B2 (the two-speed book: residual targets + frozen momentum targets, summed
per decision bar then capped/banded/gated as one book) *anti*-netted against
the gross-weighted sum of its standalone halves: certified cost
0.0761 vs ~0.0593 (+28.5%) and turnover 0.1358 vs ~0.0987 (+37.7%)
(`results/demotion_b2/summary.json` vs 0.5·(D1+B1);
`docs/demotion_design.md` §7). The netting hypothesis was the demotion
doctrine's mechanism; its failure was load-bearing in the §4 verdict.

## 2. Mechanism, from code

Two sites, both in the shipped construction path:

1. **Pre-band re-cap.** In `two_speed` mode the frozen momentum row is summed
   with the residual row each decision bar
   (`research/arbitrage/residual_walk_forward.py:328`) and the combined frame
   is proportionally re-capped (`:331` via `cap_book`,
   `src/prism/portfolio/construct.py:29-31`). Momentum's raw gross is exactly
   1.0 and residual raw gross varies, so the scale factor
   `max_gross / gross_t` — and with it the *capped momentum component* —
   moves on every decision bar the residual sleeve moves.
2. **Post-band, band-bypassing re-cap.** The online loop applies
   `step_no_trade_band` first (`residual_walk_forward.py:239`) and then
   `cap_book` on the stepped row (`:240-244`): when the stepped book's gross
   exceeds `max_gross`, the proportional rescale moves *every* name —
   including band-held momentum names — generating trades the band never
   approved. B2's emitted gross averaged ~0.985, i.e. the cap bound nearly
   always; standalone B1 (raw gross exactly 1.0, never above the cap) is
   never rescaled, so the asymmetry vs B2 is structural.

## 3. Empirical confirmation (day-class attribution)

Read-only pandas over the committed artifacts
`results/demotion_{b2,b1,d1}/target_weights.csv` + `demotion_b2/costs.csv`
(1308×574 frames, identical indexes; drop-first-establishment-day diff
convention). Computed independently twice this session (analysis and
adversarial re-derivation); every figure reproduced exactly.

| day class (of 1307 diff days) | B2 |Δw| | 0.5·(D1+B1) baseline | excess |
|---|---|---|---|
| B1 static, D1 traded (249 days) | 137.16 | 76.27 | **+60.89** |
| B1 moved — refresh/fold days (83) | 39.42 | 51.73 | **−12.31** |
| both dormant (975) | 0.00 | 0.00 | 0.00 |
| total | 176.58 | 128.00 | +48.58 (ratio 1.380) |

- The **entire** anti-netting excess sits on the days the momentum sleeve
  should have been dormant — the cap-rescale wiggle, exactly as hypothesized.
- On refresh days netting **works** (−12.31): the mechanism the doctrine
  predicted is real when both sleeves actually move.
- ~77% of B2's commission+spread+impact (0.0495 of 0.0642) attributes to
  wiggle-day decisions.
- The churn is mostly *amplification* of cells the residual sleeve also
  traded, not trades in residual-untouched names (strictly-manufactured
  cells: 3.30 |Δw|, 1.9% of B2 turnover) — consistent with D1 holding ~484
  of 574 names on average.

**Caveat (scope of the Tier-0 read).** The standalone emitted books are
*proxies* for B2's internal components — band widths differ inside B2. A
definitive per-component split would require an instrumented re-run of the
recorded B2 config via `run_residual_stat_arb_walk_forward` directly (never
via `research/scripts/stat_arb_residual_wfo.py`, whose driver appends a
ledger row unconditionally). The day-class attribution above does not depend
on that split: zero-turnover days and B1-static days are identified from the
emitted target frames themselves.

## 4. What this changes

- `docs/aim_portfolio_preregistration.md` §5's netting exit-test (drafted,
  unratified) now pins the confirmed regime: cap-binding pressure with a
  static slow component, asserting zero slow-component turnover on fast-only
  days. Aggregate-turnover bounds alone can pass while this failure mode
  survives.
- Any future multi-sleeve construction must not proportionally re-cap a
  summed row across sleeves of different cadences — or must demonstrate, by
  the pinned test, that its trade-rate machinery does not manufacture
  turnover in a dormant sleeve.
- Nothing else. The residual program stays archived; B2 is not re-run; no
  counted trial is spent or refunded.
