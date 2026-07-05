# HANDOFF — publication + deferred work

State (2026-07-05, post-quarantine): **v0.3.0 is committed and tagged locally**
(`bfc1772`, tag `v0.3.0`, on `release/v0.3.0`, `main` fast-forwarded; nothing
pushed). The wheel build is verified: the only top-level import package is
`prism` (B.4 below — done). The **`research/` quarantine relocation (C.1) is
done** on `refactor/research-quarantine`: RL members, batch WFO/sweep/stat-arb
CLIs, baselines, and mlflow tracking now live in top-level `research/`
(outside the wheel; `research` imports `prism`, never the reverse at module
level), the stat-arb signal core promoted to `src/prism/residual/`, and
`scripts/prediction.py` was dropped per SPEC §9. Full suite after the move:
452 passed, 1 skipped.

---

## A. Human actions (cannot be done by an agent)

1. ~~Approve the v0.3.0 commit + tag.~~ **Done** — `bfc1772` + tag `v0.3.0`
   (local only). Remaining human step: **push** branch(es) + tag when ready
   (`git push origin main release/v0.3.0 refactor/research-quarantine v0.3.0`
   or your preferred subset).

2. **Pick the public name and rename the GitHub repo** (Settings → General →
   Repository name). GitHub redirects old clone/web URLs automatically. This is the *single identity
   break* — do step B.1 in the same stroke.

3. **Refresh the local editable install** so subprocess imports resolve
   without the pytest `pythonpath` shim (environment-modifying, so left to
   you):

   ```
   uv pip install -e .        # or: python -m pip install -e .
   ```

   The venv currently holds a stale editable install
   (`_editable_impl_trading_ensemble.pth` + `trading_ensemble-0.2.2.dist-info`)
   that predates v0.3.0, the package move, and the quarantine.

## B. Agent prompt — publication session

Paste to an agent after A.2 is decided:

> In ~/trading-ensemble (may already be renamed on GitHub): complete the Prism
> identity break. (1) In `pyproject.toml`, set `[project] name` to the chosen
> distribution name and update both `[project.urls]` entries to the renamed
> repo. (2) Console scripts: the five `trading-*` entry points were **removed**
> at the quarantine (their targets are research-only now; rationale recorded in
> pyproject) — the only remaining decision is whether the retained universe
> builder (`prism.scripts.build_sp500_universe`) gets a `prism-build-universe`
> entry point. (3) Update `git remote set-url origin` to the new URL
> (redirects work, but explicit is better). (4) ~~Verify packaging~~ — **done
> 2026-07-05**: `uv build` succeeds and the wheel's only top-level import
> package is `prism`; re-verify only if pyproject changes. (5) Sweep remaining
> `trading-ensemble` strings in README.md badges/links, SPEC.md §12, and
> `src/prism/__init__.py`'s docstring. (6) Run the full test suite. Do not
> commit without showing the diff.

## C. Deferred engineering (sequenced in SPEC §11/§13 — normal queue, in order)

- ~~**`research/` quarantine relocation** (R1 prerequisite)~~ **Done**
  (post-v0.3.0, `refactor/research-quarantine`): moved RL members, batch WFO
  scripts, sweep, stat-arb CLIs, baselines, and mlflow tracking to top-level
  `research/`; promoted `factors.py`/`residual.py` to `src/prism/residual/`;
  dropped `prediction.py` and the RL entries in `DEFAULT_MODEL_WEIGHTS`.
  Ensemble still constructs RL members by name via a lazy fail-soft import
  from `research.models.*` — R1 replaces that seam with the plug-in
  signal-node contract.
- **JAX/torch decouple + `research` optional-dependency extra** (R1): today
  `jax[cuda12]` + `torch` are hard deps of the core install. The remaining
  core JAX importer is `features.py`'s `nnx` scaler wrapping (SPEC §9); with
  the quarantine done, this plus the pyproject extras split is the whole item.
  Add the N8 import-linter check (`prism` must not import `research`) here.
- **NaN-band policy unification** (before R2): batch `apply_no_trade_band`
  freezes a NaN-band name forever; online `step_no_trade_band` treats NaN as
  no-band. One policy, both functions, pinned by a test. (A session chip for
  this was already spawned: "Unify NaN-band policy across no-trade band forms".)
- **R2**: closed-form Gârleanu–Pedersen online rebalancer; participation gate
  wired into construction (and reconcile its ADV-floor semantics with the
  pricer's `ExecutionConfig.adv_floor_dollars` — currently documented-divergent
  by design).
- **R4 macro-regime additions** (agreed 2026-07-05, conditioning inputs only,
  IC-gated per I-8):
  - `regime/inflation.py`: real yields (FRED `DFII10`) + breakevens (`T10YIE`)
    — the observables of the "2% rhetoric, higher realized" repression regime.
  - Stablecoin-float term extending `net_liquidity` (aggregate stablecoin
    market cap, DefiLlama API, free/no key) — structural T-bill demand outside
    `WALCL − RRP − TGA`.
  - Use the SPEC §10 real-terms hurdle: `after_cost_hurdle` anchored at least
    to the prevailing T-bill yield, basis recorded in claim packets.

Discipline unchanged: cost-bound before signal-bound — R2/R3 precede any new
signal work, on the pre-registered trial budget, until `net_edge` or the §10
kill-criterion fires.
