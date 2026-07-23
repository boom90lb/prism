# Joint crash diagnostic — B1 alone vs B1+trend stress receipts (DRAFT instrument)

> **Status: uncounted engineering instrument (W1 / G4a).** Pure arithmetic in
> `prism.validation.joint_crash`; tests in `tests/test_joint_crash.py`.
> Searches nothing, moves no ratified statistic, does not open G4b.
> Preferred product narrative (`docs/v040_program.md` §3 / W1) requires this
> receipt before a multi-sleeve GO story — not before uncounted mechanics.

## 1. Question

Does adding the trend (ETF TSMOM) sleeve improve crash-window outcomes and
joint max drawdown relative to B1 alone under **fixed** capital weights
(sensitivity, not optimized aim-portfolio weights)?

## 2. Method

| Input | Source |
|---|---|
| B1 daily returns | Certified stream `results/demotion_b1/returns.csv` or a replay equity ledger converted to simple returns |
| Trend daily returns | Offline construct from local ETF bars via `TrendSignalNode` + `construct_inverse_vol_targets`, or a paper sleeve ledger once the trend paper instrument runs |
| Stress windows | Named intervals; program pins **2020-03** and **2022** as the first targets |
| Blend | Fixed weights (default equal; product profiles may pass `{b1: 0.7, trend: 0.3}` etc.) |

Outputs (JSON-serializable):

- per sleeve: `n_sessions`, full-sample `max_drawdown`, per-window `{n, total_return}`
- blend: same + the weights used
- empty windows report `total_return: null` with `n: 0` (unmeasured, not zero)

## 3. Data honesty

Local free-tier ETF caches on this checkout begin ~2025-06 for the trend
universe members present on disk; **2020-03 / 2022 windows are not runnable
from those caches alone.** The instrument is still landable: arithmetic is
tested on synthetic crashes; historical stress runs wait on a panel that
covers those eras (free ETF history pull, or owner data). Do not invent
crash-window numbers.

## 4. What this is not

- Not a counted trial; not G4b construction search.
- Not optimized multi-sleeve weights (aim-portfolio / G4b).
- Not a promotion of trend; convexity admission remains `docs/trend_design.md` §4.
- Not a silent un-gating of aim-portfolio counted trials.

## 5. CLI / usage (when bars exist)

```python
from prism.validation.joint_crash import joint_crash_report

report = joint_crash_report(
    {"b1": b1_returns, "trend": trend_returns},
    {
        "covid_2020_03": ("2020-03-01", "2020-03-31"),
        "bear_2022": ("2022-01-01", "2022-12-31"),
    },
    blend_weights={"b1": 0.7, "trend": 0.3},
)
```

A research CLI wrapper may land later; the pure module is the contract.
