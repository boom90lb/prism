"""The legacy model stack: forecast members + quarantined RL members.

The forecast members (arima, prophet, xgboost — expected h-bar returns),
their ``EnsembleModel`` orchestrator, the registry, and the vol-sizing
mapping moved here from ``src/prism/models`` in the R1 vocabulary
convergence; they serve the research WFO CLIs only. The production signal
node is ``prism.signal.ensemble_node``. The RL members (LSTM-PPO,
xLSTM-PPO, xLSTM-GRPO) are JAX-heavy and path-memorizing (docs/audit.md);
research-only per SPEC §9.
"""

__all__: list[str] = []
