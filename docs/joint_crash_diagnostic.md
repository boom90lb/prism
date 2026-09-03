# Joint crash diagnostic — B1 alone vs B1+trend stress receipts

> **Status: uncounted engineering instrument (W1 / G4a).** Pure arithmetic in
> `prism.validation.joint_crash`; offline receipt CLI in
> `research/scripts/joint_crash_receipt.py`; tests in `tests/test_joint_crash.py`.
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
| Trend daily returns | Offline construct from local ETF bars: full-panel 12−1 TSMOM (matches `TrendSignalNode` last-bar formula) → `construct_inverse_vol_targets` → next-open `backtest_target_weights` on `decision_every=21` |
| Stress windows | Named intervals; program pins **2020-03** and **2022** as the first targets |
| Blend | Fixed weights (default product profile `{b1: 0.7, trend: 0.3}`) |

Outputs (JSON-serializable):

- per sleeve: `n_sessions`, full-sample `max_drawdown`, per-window `{n, total_return}`
- blend: same + the weights used
- empty windows report `total_return: null` with `n: 0` (unmeasured, not zero)

## 3. Data honesty

**Trend ETF deep history (this checkout):** free-tier Twelve Data pull
2018-01-01 → present for the pinned TREND_V1 ten (`SPY EFA EEM TLT IEF LQD
HYG GLD PDBC UUP`) as range-keyed caches under `data/{SYM}_1d_2018-01-01_*.parquet`.
Caches are gitignored; re-pull with `DataLoader.fetch_historical_data` if
absent. Vendor duplicate session rows (observed on EEM) are dropped
keep-last at load.

**B1 certified stream:** `results/demotion_b1/returns.csv` on this checkout
spans **2021-03-30 → 2026-06-12**. Therefore **`covid_2020_03` is empty for
B1 by construction** — do not invent a pre-sample B1 stream. The 2022 window
is jointly measurable. Blend rows treat a missing sleeve as cash (0 return)
for that session; read B1's own window block for joint eligibility.

**First offline receipt (local, uncounted, 2026-07-22):**

| Series | max DD | covid_2020_03 | bear_2022 |
|---|---|---|---|
| B1 | −14.0% | empty (n=0) | +2.9% (n=251) |
| trend (T0 offline) | −11.1% | −2.5% (n=22) | +9.3% (n=251) |
| blend 0.7/0.3 | −12.1% | −0.7% (n=22; B1 cash) | +5.0% (n=251) |

Read as sensitivity only: trend helped 2022 total return and full-sample
max DD vs B1 alone under these weights; March 2020 has **no joint B1
evidence** on the certified stream. Not a promotion; not G4b.

## 4. What this is not

- Not a counted trial; not G4b construction search.
- Not optimized multi-sleeve weights (aim-portfolio / G4b).
- Not a promotion of trend; convexity admission remains `docs/trend_design.md` §4.
- Not a silent un-gating of aim-portfolio counted trials.

## 5. CLI

```bash
# requires local deep-history ETF caches (see §3)
uv run python -m research.scripts.joint_crash_receipt
# → results/joint_crash_receipt_YYYY-MM-DD.json  (gitignored scratch)

# G4a fixed-weight capital allocation sensitivity (b1 weight 0..1 step 0.1)
uv run python -m research.scripts.joint_crash_receipt --sensitivity
```

Library surface:

```python
from prism.validation.joint_crash import (
    capital_allocation_sensitivity,
    joint_crash_report,
)

report = joint_crash_report(
    {"b1": b1_returns, "trend": trend_returns},
    {
        "covid_2020_03": ("2020-03-01", "2020-03-31"),
        "bear_2022": ("2022-01-01", "2022-12-31"),
    },
    blend_weights={"b1": 0.7, "trend": 0.3},
)
grid = capital_allocation_sensitivity(
    {"b1": b1_returns, "trend": trend_returns},
    {"bear_2022": ("2022-01-01", "2022-12-31")},
    primary="b1",
)
```
