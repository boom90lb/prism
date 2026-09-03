"""Residual signal core: factor model + causal OU s-scores + book builder.

Promoted from the quarantined stat-arb research path (SPEC §9): `factors`
holds the eigenportfolio/ETF factor construction and batched factor OLS;
`residual` holds the Avellaneda-Lee s-score state machine and the hedged
book builder whose output rows feed
``prism.execution.target_weights.backtest_target_weights``.

SPEC §5 scopes this package to grow RMT cleaning and neutralization (R4);
the WFO harnesses that consume it live in ``research/arbitrage``.
"""

__all__: list[str] = []
