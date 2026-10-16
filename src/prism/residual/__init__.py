"""Residual signal core: factor model + causal OU s-scores + book builder.

Promoted from the quarantined stat-arb research path (SPEC §9): `factors`
holds the eigenportfolio/ETF factor construction and batched factor OLS;
`residual` holds the Avellaneda-Lee s-score state machine and the hedged
book builder whose output rows feed
``prism.execution.target_weights.backtest_target_weights``.

SPEC §5 scopes this package to grow RMT cleaning and neutralization (R4).
The research walk-forward harnesses that consumed it are gone:
``research/arbitrage/residual_walk_forward.py`` (removed; last present at commit 5fcac33869e9) and
``research/scripts/stat_arb_residual_wfo.py`` (removed; last present at commit 5fcac33869e9).
Several research modules still import it, ``research/arbitrage/pairs.py`` among
them, and so does the engine: the signal nodes, ``prism.risk_model``,
``prism.scripts.paper_books``, ``prism.scripts.replay_loop`` and
``prism.io.legacy_import``.
"""

__all__: list[str] = []
