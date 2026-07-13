# formal/ — machine-checked kernel invariants

A core-only Lean 4 package (no Mathlib; builds in seconds) that
machine-checks the *checkable algebra* behind `SPEC.md`'s invariants, over
exact integer arithmetic (micro-dollars / micro-weights):

| Module | Invariant | Theorems |
|---|---|---|
| `Ledger.lean` | **N4** | `rebalance_conserves`, `run_conserves` — trading and multi-period accounting create/destroy no cash; `mark_to_market_pnl` — PnL is held-quantity × price move, nothing else |
| `Band.lean` | **§7.3** | `stepBand_idem`, `stepBand_trades_only_past_band`, `stepBand_cases`; `batch_replay_from_zero_diverges` — the batch-replay defect as a checked counterexample |
| `Purge.lean` | **I-1** | `purge_label_disjoint`, `purge_train_before_test`, `train_avoids_embargo` — purge/embargo index geometry |
| `Participation.lean` | **§7.4** | `capTrade_le_cap`, `capTrade_down_only`, `capTrade_sign_*` — the gate is a pure attenuator |

## Division of labor

Lean proves the **algorithm** over ℤ. Pytest proves the float64
**implementation** tracks the algorithm (`tests/test_ledger_conservation.py`
and friends). The Lean proofs do *not* verify the Python code — they verify
the algebra the Python is property-tested against. Both halves together are
the guarantee; neither alone is.

Deliberately **not** formalized: float error propagation (bounded empirically
in the property tests), the DSR/PBO statistics derivations (literature
results; formalizing them is research, not engineering), anything about
market behavior. Formal methods here verify *bookkeeping*, never *beliefs*.

## Build

```bash
# toolchain pinned in lean-toolchain (leanprover/lean4:v4.31.0)
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh   # once
cd formal && lake build
```

A green `lake build` with zero `sorry` warnings is the acceptance criterion.
The package stays core-only: a Mathlib dependency is added only for a named
theorem that needs it and that a property test cannot cover.

Next targets, in value order (charter: `docs/handoff.md §5`): the
`live/` crash-safety state machine (`src/prism/live/` now ships), the R2
Gârleanu–Pedersen band's monotonicity properties, trial-ledger append-only
monotonicity.
