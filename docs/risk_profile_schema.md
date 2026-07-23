# Risk-profile schema — operator surface (W6) (DRAFT)

> **Status: DRAFT — schema may land early; freeze lags code**
> (`docs/v040_program.md` §5 queue item 6 / W6; handbook J6). Profiles are
> **product**; ratified pins are **constitution**. No profile may loosen a
> pin. Code draft lives at `prism.live.risk_profile` (provisional names) with
> G6 soft-gate tests in `tests/test_risk_profile.py`: `research_paper`
> resolves to `CERTIFIED_B1_PAPER_CONFIG` and fails loud on divergence.
> Paper CLI: `--profile research_paper` pins construction and writes
> `profile.json` under the run dir. Freeze of this schema is the owner act
> that stops free renaming of profile fields; until freeze, field names
> below are provisional and non-paper profiles are not a live-deploy path.

## 0. Product law

Operators never free-knob a counted statistic (gross that was searched,
g that was sized, participation that was costed). They choose a **named
risk profile** that maps onto already-ratified pins or explicitly
*stricter* subsets, plus a **hedge composition policy** over sleeves that
exist.

| Ranking | Who wins conflict |
|---|---|
| SPEC / ratified family pins | always |
| This schema (once frozen) | product surface only |
| Operator day-discretion | never (no ad-hoc hedge ratios) |

## 1. Profile enum (provisional)

| Profile id | Intent | Gross / participation | Crash term (de-gross) | Sleeve mix default | Who |
|---|---|---|---|---|---|
| `research_paper` | Free-tier learn-the-loop; promotion instrument | As certified B1 paper path | Telemetry only (hook **not** armed) | B1 only | default paper |
| `conservative` | First real-money default | Tightened `max_gross` allowed (≤ certified pin) | g=0.5 pin when GO-armed (`docs/sizing_preregistration.md` §3) | B1 + trend weight *floor* if trend admitted | first deploy |
| `balanced` | Default deploy after multi-sleeve evidence | Certified gross | g=0.5 when GO-armed | Multi-sleeve preferred weights (policy bands) | standard deploy |
| `assertive` | Higher risk *within* pins | Certified gross (never above) | Only as later-ratified pins allow — **never freer than ratified** | Multi-sleeve | explicit opt-in |

**Hard rules:**

1. Profiles may only **tighten** relative to ratified pins, never loosen
   (no g > 0.5 without a new sizing ratification; no participation above
   the certified gate; no gross above the certified construction pin).
2. `research_paper` is bit-identity-bound to today's certified B1 paper
   loop: same config hash surface for the promotion instrument. Any
   divergence is a defect, not a profile feature.
3. Enabling trend / crypto sleeves is a **composition policy** choice
   under §2, gated on sleeve existence and GO doctrine labels — not a
   free boost inside `assertive`.
4. Unknown profile id → fail loud (N7). Silent fallback to
   `research_paper` is forbidden outside tests that assert the fallback
   path explicitly.

## 2. Hedge composition policy

One declared policy object (not three mental ledgers):

```text
HedgePolicy:
  equity_sleeve:  { enabled: bool, capital_band: [lo, hi] }  # B1 or X-supersession
  trend_sleeve:   { enabled: bool, capital_band: [lo, hi] }  # convexity
  crypto_book:    { enabled: bool, capital_band: [lo, hi] }  # optional validation
  de_gross:       { armed: bool }  # only after GO preconditions + arming commit
```

Constraints:

- `sum(hi)` over enabled sleeves ≤ 1.0 of deployable capital after custody
  float (or the policy is rejected at load).
- `lo`/`hi` are static bands for a profile, not daily discretion.
- `crypto_book.enabled` default **false** for all v0.4.0 equity-first
  profiles; enabling is operator-explicit and never required for preferred
  GO (`docs/v040_program.md` §3).
- `trend_sleeve` weight floor under `conservative` is a *minimum* allocation
  when the sleeve is admitted at its bar — it is not a backtest-optimized
  weight (G4b owns counted construction search).
- De-gross is sizing law, not a sleeve; `armed` is independent of profile
  id and requires the sizing deployment commit.

Composition instruments (doctrine table — `docs/v040_program.md` §8):

| Instrument | Profile surface | Not |
|---|---|---|
| De-gross | `de_gross.armed` + sizing pins | Convexity |
| Trend sleeve | `trend_sleeve.*` | Standalone Sharpe knob |
| Construction netting | G4b / aim-portfolio when open | Sum-then-cap |

## 3. Mapping to code surfaces (implementation sketch; not yet authorized)

Provisional target (code lands only after schema freeze + G6 work):

| Concern | Surface | Profile effect |
|---|---|---|
| Paper / live loop config | run dir / profile yaml | selects pins + policy |
| Construction | `max_gross`, symbol cap, band | tighten-only overrides |
| Participation | gate %ADV | tighten-only |
| Regime gross scale | hook arm + g | arm only when allowed; g never freer |
| Multi-book routing | book prefixes | enable flags from policy |
| Claim packets | profile id recorded | every GO/deploy claim names profile |

**G6 soft gate (from program):** documented schema (this file) + paper loop
honors a profile without bit-breaking certified B1 under `research_paper`.
No counted trial spent.

## 4. Freeze criteria

This schema freezes when all hold:

1. Owner dedicated commit flips banner DRAFT → FROZEN (or RATIFIED if
   treated as a design doc).
2. `research_paper` paper path is demonstrated bit-identical to the
   certified B1 instrument on a fixed fixture (test receipt).
3. Field names in §1–§2 stop drifting; further renames need a new
   dedicated commit.

Code may draft against provisional names on the execution branch; shipping
profile-aware deploy without freeze is not "deserved" v0.4.0 product.

## 5. Explicit non-goals

- No free risk knobs that loosen pins.
- No "assertive = higher leverage than certified."
- No crypto-first defaults.
- No profile that bypasses GO preconditions (sizing; §7.7 + 21 clean
  sessions; capital mode).
- No silent un-gating of aim-portfolio counted trials via profile mix.
