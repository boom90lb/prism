# HANDOFF — publication + deferred work

State at handoff (2026-07-05): **v0.3.0 is complete in the working tree,
uncommitted.** It contains the Prism re-founding (SPEC.md, MARKETS.md, six new
modules + tests), a post-review cleanup pass, and the **src-layout migration**
(importable package `prism` at `src/prism/`, `scripts/` → `prism.scripts`).
Full suite: 452 passed, 1 skipped. Nothing here blocks daily research work;
everything below is publication mechanics or sequenced engineering.

---

## A. Human actions (cannot be done by an agent)

1. **Approve the v0.3.0 commit + tag.** Suggested sequence:

   ```
   git checkout -b release/v0.3.0
   git add -A
   git commit   # "feat: Re-found as Prism cross-sectional engine, src-layout (v0.3.0)"
   git tag v0.3.0
   ```

   Note: `git status` will show `src/* → src/prism/*` as deletes+adds until
   commit; rename detection recovers history (`git log --follow`).

2. **Pick the public name and rename the GitHub repo** (Settings → General →
   Repository name). GitHub redirects old clone/web URLs automatically.
   Recommendation from the naming review: bare `prism` is heavily overloaded
   (Prism.js, Azure Prism, Stoplight Prism); prefer **`prism-trading`** if the
   repo will ever be public or published to PyPI. This is the *single identity
   break* — do step B.1 in the same stroke.

3. **Refresh the local editable install** so console entry points and
   subprocess imports resolve without the pytest `pythonpath` shim
   (environment-modifying, so left to you):

   ```
   uv pip install -e .        # or: python -m pip install -e .
   ```

   The venv currently holds a stale editable install
   (`_editable_impl_trading_ensemble.pth` + `trading_ensemble-0.2.2.dist-info`)
   that predates both v0.3.0 and the package move.

## B. Agent prompt — publication session

Paste to an agent after A.2 is decided:

> In ~/trading-ensemble (may already be renamed on GitHub): complete the Prism
> identity break. (1) In `pyproject.toml`, set `[project] name` to the chosen
> distribution name and update both `[project.urls]` entries to the renamed
> repo. (2) Decide/confirm console-script names: the five `[project.scripts]`
> entries still carry the `trading-*` prefix (`trading-backtest`, etc.) —
> rename to a `prism-*` prefix or record why not. (3) Update `git remote
> set-url origin` to the new URL (redirects work, but explicit is better).
> (4) **Verify packaging**: build a wheel (`uv build` or `pip wheel --no-deps`)
> and confirm the only top-level import package in it is `prism` — this checks
> the `packages = ["src/prism"]` hatchling mapping, which was configured but
> could not be build-verified at migration time (no build tool on PATH).
> (5) Sweep remaining `trading-ensemble` strings in README.md badges/links,
> SPEC.md §12, and `src/prism/__init__.py`'s docstring. (6) Run the full test
> suite. Do not commit without showing the diff.

## C. Deferred engineering (sequenced in SPEC §11/§13 — normal queue, in order)

- **`research/` quarantine relocation** (R1 prerequisite): move RL members,
  batch WFO scripts, sweep, stat-arb CLIs per the SPEC §9 salvage map.
- **JAX/torch decouple + `research` optional-dependency extra** (R1): today
  `jax[cuda12]` + `torch` are hard deps of the core install.
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
