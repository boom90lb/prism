# Construction band audit — 2026-09-24

**Kind:** audit, findings only (SPEC §7: the full finding set before any fix). Nothing in
the code, specs or records was changed by this audit. The fix options at the end await
the owner's authorization.

**Trigger:** the earnings read (`docs/earnings_read_2026-09-24.md`) found costs binding
across the family, and the run records reported a `closed_form` no-trade band whose
width was 0.0.

## Findings

**F1. The registered runs' engine path applies no no-trade band.**
- `prism.scripts.construction_experiment` runs every arm with
  `decision_coupling="account_state"` (`src/prism/scripts/construction_experiment.py:170`,
  `:334`), which routes to `_evaluate_stateful` (`src/prism/evaluate.py:1285`).
- On that path:
  - the heuristic arm's construction is `_heuristic_construction`
    (`src/prism/decisions.py:547`), which applies the participation cap and nothing else;
  - the fold cost inputs are built with `band=0.0` (`src/prism/evaluate.py:2517`).
- The spec's `[construction.band]` block is parsed and validated
  (`src/prism/evaluate.py:1829-1845`) and has no effect.
- The band is estimated and applied only on the other coupling, `_evaluate_folds` →
  `_fold_cost_inputs` → `_apply_construction` (`src/prism/evaluate.py:3162-3222`,
  `:2170`, `:1994`). No registered run uses it.

**F2. The records describe a band that did not act (N7).**
- Every such record reports:
  - `evaluation.construction.heuristic_settings.band = {mode: closed_form, estimated_on: fold_formation_window, width: 0.0}`;
  - `evaluation.construction.policies.construction.band.mode = closed_form`.
- Its cost block reports `costs.cost_refresh.per_fold[*].band = {kind: scalar, width: 0.0}`.
- A declared behaviour silently not performed is a fail-loud defect.
- Registrations whose control declares `closed_form` and ran on this path:
  - `2026-09-characteristics`
  - `2026-09-breadth`
  - `2026-09-learned-xsection`
  - `2026-09-fundamentals`
  - `2026-09-earnings`
  - `2026-09-earnings-daily`
- `2026-09-construction` declared `off` (`examples/momentum/reference.toml`) and is
  consistent.

**F3. Corroboration already in the record.** The characteristics read's `momentum_control`
declares `closed_form`, yet it reproduces the construction read's decile control
(declared `off`) exactly, at net Sharpe 0.678903
(`docs/characteristics_read_2026-09-23.md`, Parity). That is what F1 predicts.

**F4. The certified B1 evidence was built with a band, and the canonical reads are not.**
- `results/demotion_b1/config.json` records `walk.band_mode = "closed_form"`.
- The pre-overhaul driver that produced it applied the band through
  `step_no_trade_band`, sized by `closed_form_band` per fold
  (`research/arbitrage/residual_walk_forward.py:239`, `:444-460`, recoverable at
  `52281e1^`; the file was removed in `52281e1`).
- Every canonical read since differs from B1 in construction. No read states this, and
  `docs/construction_experiment_read_2026-09-08.md` does not mention bands.

**F5. Magnitude: the declared band is too narrow to matter for these books.** This is a
diagnostic on turnover only, run 2026-09-24 and not written under `results/`.

Method:
- Each run's recorded target path is replayed through `step_no_trade_band`.
- The per-name half-width comes from `closed_form_band`, using:
  - the variance of the target path's own decision-to-decision changes (the whole path,
    not the fold's formation window);
  - a round trip at the recorded mean half-spread of 2.6 bp;
  - zero commission;
  - `gamma_risk` = 1.
- Drift between decisions is ignored.

| run | decisions | median half-width | target-path turnover | with the band |
|---|---|---|---|---|
| `momentum_control` (`characteristics_v1`) | 51 | 0.0020 | 28.3 | 28.3 (100%) |
| `ear_monthly` | 60 | 0.0023 | 39.7 | 39.6 (100%) |
| `ear_event` | 1,308 | 0.0010 | 86.6 | 78.7 (91%) |
| `announcement_premium` | 1,308 | 0.0016 | 298.2 | 297.8 (100%) |

- **Why it barely acts.** The half-width is 0.1–0.2% of equity. Decile weights are about
  1% per name. The band suppresses only small re-weightings, not entries and exits,
  which carry the turnover.
- **Consequence.** Recorded results of the affected reads are very likely unchanged in
  substance; the defect is in what the records say.
- **Caveat.** This is an approximation, not a rerun.

**F6. The live paper loop applies no band either.** `prism.scripts.paper_loop` takes
`--band` with default 0.0 (`src/prism/scripts/paper_loop.py:179-182`), and the `ops/`
wrappers pass none. This is consistent with its own records, but it differs from B1's
certified construction (F4). By F5 the difference is negligible for a monthly decile
momentum book.

**F7. The earnings read overstates the band as the cost lever.**
- `docs/earnings_read_2026-09-24.md` ("What the numbers say", item 6) says a working
  no-trade band "acts directly on the binding constraint". F5 contradicts that for the
  declared closed-form band.
- The earnings costs come from decile-membership churn, which a band of this width does
  not touch.
- The SPEC §1 bullet of 2026-09-24 places the band beside the cost finding.

**Not a defect:** the experiment profile's `net_band` is a preference limit, recorded when
breached rather than enforced (`preference_response = "record"`, `examples/reference/policies/recovery.toml:10`). That is by design.

**Adjacent, not part of this audit:** the fundamentals read did not report SPEC §5.5's IC
horizon profile.

## Fix options (not applied; for authorization)

1. **Make the declaration honest (recommended first).**
   - Refuse `[construction.band] mode != "off"` on the `account_state` path with a clear
     error (N7).
   - Leave the existing specs and records as committed, since closed sets stay
     reproducible from their commits.
   - Record this audit next to each affected read.
   - Future specs declare `off` until a band is implemented on this path.
2. **Implement the band on the `account_state` path.** Apply the fold's closed-form
   half-width in `_heuristic_construction` against the held book.
   - This changes the behaviour of any configuration that declares it, so it needs a
     versioned semantics flag so earlier records reproduce.
   - By F5 it would change little for decile books.
3. **Correct F7.** Add a dated addendum to the earnings read and reword the SPEC §1 bullet.
   Costs are driven by decile-membership churn; the declared band is not the lever.
4. **The actual cost lever for the earnings family is a construction change, and a new
   registration.** Candidates:
   - a rank buffer: enter in the top decile, exit only below the top quintile;
   - the Gârleanu–Pedersen partial-trade step of SPEC §5.5
     (`docs/aim_portfolio_preregistration.md`);
   - a slower re-formation cadence for the premium.
