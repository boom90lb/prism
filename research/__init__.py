"""Quarantined research code (SPEC.md §9) — NOT part of the `prism` distribution.

Everything under `research/` is sound research kept out of the production
import path: JAX-heavy RL members, batch WFO scripts, the sweep, stat-arb
WFO/ledger CLIs, baselines, and MLflow provenance. `research` may import
`prism`; `prism` must never import `research` at module level (RL member
construction in the ensemble uses a lazy, fail-soft import until R1 replaces
it with the plug-in signal-node contract).

Run CLIs from the repo root, e.g. `python -m research.scripts.backtest`.
"""
