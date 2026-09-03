"""Quarantined research code (SPEC.md §9) — NOT part of the `prism` distribution.

Everything under `research/` is sound research kept out of the production
import path: the legacy per-symbol forecaster stack (trading.py, models/,
features.py, sentiment_analysis.py, config.py — the v0.2 system, moved out
of src/prism by the R1 vocabulary convergence; it still serves the batch
WFO CLIs), the JAX-heavy RL members, the sweep, stat-arb WFO/ledger CLIs,
baselines, and MLflow provenance. `research` may import `prism`; `prism`
never imports `research` (enforced by tests/test_import_hygiene.py).

Run CLIs from the repo root, e.g. `python -m research.scripts.backtest`.
"""
