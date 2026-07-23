# Regime step — wiring record (SPEC §7.7 step 3, §7.5 blocks)

**Status: telemetry LIVE-WIRABLE on the certified path; the de-gross ACTION
hook exists and is UNARMED.** This is GO-branch precondition (b) of
`docs/handoff.md` §8: *"the SPEC §7.7 regime step is wired into the live
cycle and has run in the paper loop for ≥ 21 consecutive sessions (one full
decision cycle) without an N7 event."* The step runs as telemetry every
cycle; the gross-scale action arms in code only after the sizing
pre-registration (`docs/sizing_preregistration.md`) is ratified with
its crash-conditional de-gross term. There is deliberately no CLI path to
the action hook. With the feature off, the certified B1 paper book is
bit-identical to the pre-wiring loop; with telemetry on, it is identical
except for the regime read and its ledger
(`tests/test_live_daily.py::test_bit_identity_with_both_regime_params_none`).

## 1. Contract

**Provider** (`prism.live.regime_step.RegimeTelemetry`): a callable
`provider(decision_bar: str) -> dict` over injectable `FredClient` /
`DefiLlamaClient`-shaped clients (`prism.regime.fetch`), so every path runs
offline on canned series. The returned dict:

```json
{
  "decision_bar": "YYYY-MM-DD",
  "blocks":   {"curve": {...}, "liquidity": {...}, "inflation": {...}, "vol": {...}},
  "failures": [{"block": "<name>", "error": "<exception class name>"}],
  "absent":   ["dollar_factor", "vix9d_slope", "vrp"]
}
```

Every fetch is causal (`end = decision_bar`); each block reports the `asof`
date of the row it read, so a lagging source is visible as a lagging `asof`,
never disguised as fresh. Non-finite values become `null` — every read is
strict-JSON. The FRED key comes from the environment (`FRED_API_KEY`,
`FredClient.from_env`), travels only in the client's request params, and
appears in no telemetry dict, ledger row, or log line.

**Seam** (`run_daily_cycle(..., regime=, regime_gross_scale=)`,
`prism.live.daily`): the `unrankable` pattern — optional params, `None` =
bit-identical. The provider is consulted on EVERY cycle: refresh, hold, and
halted sessions alike, because the precondition-(b) clock counts sessions
and a halted cycle must still record regime state. The read lands in
`DailyCycleResult.regime` and is appended to the regime ledger
(`LiveLoopContext.regime_ledger`, `{run-dir}/regime.jsonl`) — one row per
decision bar, idempotent and monotone (the equity-ledger discipline), each
row carrying `clean`, `gross_scale`, `blocks`, `failures`
(`read_regime_ledger` in `prism.live.loop`).

## 2. Blocks: wired vs named-absent

Wired — all through the existing `prism.regime` surface; nothing new is
adapted:

| Block | Fields | Source path |
|---|---|---|
| `curve` | level / slope / curvature | `fetch_curve_state` (FRED DGS tenors, `regime.sources` fred_dgs) |
| `liquidity` | net_liquidity, net_liquidity_change, stablecoin_term | `fetch_net_liquidity` + `net_liquidity_change` (FRED WALCL − RRPONTSYD − WTREGEN) |
| `inflation` | real_yield, breakeven, breakeven_divergence | `fetch_inflation_state` (FRED DFII10 + T10YIE) |
| `vol` | vix, vix3m, term_ratio | `FredClient.series` on the registered fred_vix row (VIXCLS, VXVCLS) + `vix_term_slope` |

Named absent (`ABSENT_BLOCKS`, recorded in every read) — §7.5 contract
members with no free fetch path through `regime.fetch`; building a new
external dependency for any of them is a separate, owner-gated change:

* **`vrp`** — `variance_risk_premium` needs the market proxy's realized leg;
  `regime.fetch` adapts no equity price series, and the provider deliberately
  does not reach into the loop's bar panel (the seam passes only the decision
  bar).
* **`vix9d_slope`** — VIX9D is CBOE-only; the CBOE CSV row in
  `regime.sources` is registry-only, unadapted.
* **`dollar_factor`** — no shipped feature math in `prism.regime` computes it.

The DefiLlama stablecoin fourth term (§7.5 R4) is *not* wired in the live
provider: `RegimeTelemetry` defaults to the three-term identity
(`stablecoin_term: false` in the ledger) and takes the fourth term only by
injected client — one fewer keyless third-party failure surface on the
precondition-(b) clock. Arming it live is an explicit constructor change,
recorded here when made.

## 3. Failure policy (N7: loud, named, never silent, never fatal)

* A block that fails — transport, venue, malformed payload, or an all-NaN
  fetch window (absence is not data) — produces `{"error": "<exception
  class name>"}` in `blocks`, one matching entry in `failures`, and one loud
  `logger.warning`. Class name only: no message text that could carry a URL
  or credential into the ledger.
* Every configured block appears in every read — values or a named failure.
  A silently-empty dict is a contract violation, not an output.
* The provider never raises out of the cycle. If a provider violates that
  contract anyway (a raise, or an empty result), the seam converts it to a
  named `provider` failure entry with a loud warning — telemetry can never
  take down the certified book's cycle, and the violation is still recorded
  as not-clean.
* Telemetry failure never touches orders: the cycle's decisions are
  byte-identical to the unwired loop under any provider failure
  (`test_raising_provider_is_contained_named_and_loud`).

## 4. The clean-session rule (handoff §8 precondition (b))

The 21-session clock reads `{run-dir}/regime.jsonl`. A session **counts as
clean** iff its cycle ran to completion and appended a regime row with
`"clean": true` — i.e. the read carried **zero** failure entries (block-level
or provider-level). Any failure entry — one dark FRED series, one malformed
payload, one provider contract violation — makes that session not clean, and
the consecutive count restarts from the next clean session. A session with
no regime row at all (loop dark, cycle crashed before the append) is not
clean either: absence of the record is absence of the session. Halted and
non-refresh sessions count on the same terms as trading sessions — the row,
not the order flow, is the unit. An N7 event anywhere else in the cycle
(per the precondition's own wording) also disqualifies the session
regardless of a clean regime row.

## 5. The action hook stays unarmed

`regime_gross_scale` is the §7.5 consumer seam ("construction
gross-scaling"): armed, it fires only on a refresh session whose regime read
has zero failure entries, multiplies `config.max_gross` for that refresh's
construction, is clamped to `[0, config.max_gross]`, and its applied
multiplier is recorded in `DailyCycleResult.regime_scale` and the ledger's
`gross_scale`. On a not-clean read the refresh constructs at the configured
gross with a loud warning — a telemetry failure de-arms the action; it never
silently de-grosses. The hook requires the telemetry provider (a hook
without telemetry is a dead switch and is refused, N7). A clamp to 0.0 is
the full de-gross: construction emits each book's explicit flat form and the
online band exits every held, decidable name — except a spin-off-masked
(unrankable) name, whose no-decision rule takes precedence and holds it (the
mask contract, `docs/bar_vendor_divergence.md` §5).

**It is unarmed everywhere.** No caller passes it; `--regime` wires
telemetry only; there is no CLI flag that can construct it. It arms in code
— a reviewed change citing this section — only after
`docs/sizing_preregistration.md` is ratified with its
crash-conditional de-gross term. Telemetry is live evidence-gathering;
sizing authority waits on the pre-registration, which is precondition (a) of
the same handoff row.

## 6. Operation

```
python -m prism.scripts.paper_loop --book momentum \
    --universe-file data/universe/sp500_current.txt --run-dir runs/paper_loop \
    --regime
```

`--regime` (default off) constructs `RegimeTelemetry.from_env()` — a missing
`FRED_API_KEY` fails loud at startup, before any venue call: the operator
armed the regime record, so a cycle without one must not run quietly. The
nightly wrapper (`/home/b90/bin/prism_paper_loop_nightly.sh`) is not touched
by this change; the integrator activates the flag there.

## 7. Evidence

`tests/test_regime_step.py` (provider: block values, named failures,
JSON-safety, causal windows, never-raises, never-empty) and the regime-step
section of `tests/test_live_daily.py` (seam: every-cycle execution incl.
hold and halted sessions, provider-failure containment, bit-identity with
both params `None` down to ledger bytes, hook clamp/recording/de-arm rules,
ledger idempotence).
