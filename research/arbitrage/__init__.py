"""Quarantined stat-arb WFO machinery (pairs scan + fold ledgers).

The signal core (factor model, OU s-scores, book builder) promoted to
`prism.residual`; what stays here is the research harness around it: the
Engle-Granger pairs path and the rolling formation/test walk-forward with
per-fold selection ledgers. Net-negative in every configuration to date —
retained as ledgered evidence, not as a production path.
"""

__all__: list[str] = []
