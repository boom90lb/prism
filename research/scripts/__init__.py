"""Quarantined batch CLIs: WFO training/backtest, sweep, RL seed eval, stat-arb.

These are the offline validation harness that *gates* live trading, not part
of it (SPEC §9). No console entry points ship for them; run as modules from
the repo root, e.g. `python -m research.scripts.training --help`.
"""
