# Look-ahead checker: does a signal's value at bar t change once later bars exist?

Date: 2026-10-15. This page covers `prism.causality`, a check for look-ahead bias in any
signal function. A signal computed at the close of bar *t* may use bars up to and including
*t* and nothing after. The usual way to break that rule leaves no trace in a backtest's summary
statistics. A feature is computed once over the whole series and then read row by row, so the
value at bar *t* carries information from bar *t + k*. Typical culprits are a forward return
used as a feature, a full-sample mean or z-score, a centred rolling window, a backward fill and
a cache filled from a longer panel.

The checker runs a function on prefixes of a panel and on the whole panel and compares the
values it gives for the same bars. It imports only numpy and pandas (about 5 ms on top of
pandas on the development machine), and it needs no Prism workspace, dataset, broker account
or class hierarchy: any callable that takes one or more panels works.

A runnable tour on synthetic prices is `examples/causality/lookahead_check.py`. It reads no
files, uses no network and finishes in under a second:

```sh
PYTHONPATH=src python examples/causality/lookahead_check.py
```

Install from a clone of this repository; `docs/depending_on_prism.md` explains why.

## Quick start

```python
import pandas as pd
from prism.causality import assert_causal, check_causality

def momentum(close: pd.DataFrame) -> pd.DataFrame:
    return close.shift(21) / close.shift(252) - 1.0      # honest

def leaky(close: pd.DataFrame) -> pd.DataFrame:
    return (close - close.mean()) / close.std()          # full-sample z-score

report = check_causality(leaky, close)
print(report.ok)       # False
print(report.text())   # one line per check, naming the bar, the column and both values

# In a test suite: returns the report on success, raises an AssertionError otherwise.
def test_momentum_has_no_lookahead():
    assert_causal(momentum, close)
```

`close` is a DataFrame of prices indexed by bar (sorted, no duplicate bars), with one column
per asset. Pass several panels in the order the function takes them:
`check_causality(fn, close, volume)` calls `fn(close_prefix, volume_prefix)`. A Series works
too, for a single asset. Every panel is cut at the same row.

## Two output shapes

| `output=` | The function returns | The check reads |
|---|---|---|
| `"cross_section"` | Values for the *last* bar it was handed: one per asset (a Series or a 1-D array) or one number. This is the contract of `prism.signal.base.Signal.score`. | The whole output. |
| `"panel"` | A result indexed by the bars it was handed: a DataFrame, a Series, or an array with one row per bar. This is what vectorized pandas code returns. | The row of each bar being compared. A bar missing from the output (for example after `dropna()`) counts as NaN. |
| `"auto"` (default) | Decided once, from the first output. A DataFrame or 2-D array is a panel. A Series whose labels are all bars of the panel handed in is a panel. Any other Series, a 1-D array or a number is a cross section. | |

`"auto"` refuses to guess when a Series's labels are all bars *and* also all columns, or are
all bars of an index that is not made of dates (for example a 0..n-1 integer index, where a
positional cross section looks like a time series). Pass `output=` explicitly in those cases.
A panel output must be indexed by the panel's own bars. A result indexed by other dates, for
example month-ends from a `resample`, is not supported: `"panel"` refuses it when no label is a
bar and reads NaN for the bars it lacks, and `"auto"` reads it as a cross section, which catches
only leaks through retained state. Return the result on the panel's bars before checking it.

## Geometry

The panel is cut into three blocks:

```
| rows before the window | check window (5 bars) | withheld future (21 bars) |
  training_rows              compared                 seen only by full-panel runs
```

`check_window_bars` (default 5) and `future_bars` (default 21) set the last two. The panel needs
at least `check_window_bars + future_bars + 1` rows. The withheld block exists because a
forward-looking feature is NaN at the end of any panel it is computed on, so a check window at
the very end of the panel would see nothing. A function that looks *k* bars ahead can be caught
when *k* is at most `future_bars`.

A function that needs a fitted model is fitted inside `factory`, on the rows before the window
only. Their count is `report.training_rows`, which equals
`len(close) - check_window_bars - future_bars`.

## The four checks

They run in this order, because the order is what exposes a cache.

1. **`causality`.** First the *causal reference*: for each check bar *t*, the function is
   called on the panel cut at *t*. When `factory` is given, each of these calls uses a new
   function from it. Then the full panel is scored. In panel mode, the full panel's row for
   each check bar must equal the reference. This is the comparison that catches a forward
   return, a full-sample statistic, a centred window or a backward fill. In both modes, each
   prefix is then rescored, newest first, and must still equal the reference. A function that
   cached a feature computed on the full panel answers the shorter prefixes from that cache
   and is caught here.
2. **`cache_reuse`.** The prefixes are rescored again, oldest first, with whatever state the
   function kept, and must equal the reference.
3. **`determinism`.** The full panel is scored twice more by the same function and the two
   outputs must agree (in panel mode, at the check bars). A nondeterministic function also
   fails the comparisons above, so fix this first; its other failures may be noise.
4. **`coverage`.** The reference must hold at least one finite value at some check bar. A
   function that is NaN everywhere in the window would pass every comparison while having been
   tested on nothing, so that case fails. Coverage also fails a purely forward-looking feature,
   which is NaN at the end of every prefix however far ahead it looks.

`report.ok` is true only when all four pass. Comparisons are bitwise by default, with NaN equal
to NaN. `rtol=` sets a relative tolerance for every comparison instead. Use it only when you
know why prefix and full-panel results differ in the last bits; a difference that grows with
the panel's length is look-ahead, whatever its size. Outputs are aligned on their labels, and
a label present in one output and missing from the other counts as NaN where it is missing.

The checker calls the function on copies of the prefixes, so a function that modifies its
input in place cannot change the panel the later calls see. The number of calls is at most
`3 × check_window_bars + 3` (18 with the defaults), each on up to the full panel.

## Reading a report

`check_causality` returns a `CausalityReport`. Its fields are `ok`, `checks` (four `Check`
objects with `name`, `status`, `detail` and, for a failed comparison, a `difference`),
`output` (the mode used), `check_bars`, `training_rows`, `future_bars`, `isolated` (whether a
factory was given) and `rtol`. `report.check("causality")` looks a check up by name.
`difference` gives `bar`, `label`, `reference` (the causal value) and `observed`.

```
look-ahead check: FAIL
output       panel: rows read at each check bar from a time-indexed output
check bars   5, 2017-03-15 00:00:00 to 2017-03-21 00:00:00; 574 row(s) before them, 21 withheld after them
reference    the function under test itself (no `factory`, so state kept between calls is not isolated)
comparison   bitwise, NaN equal to NaN
  FAIL causality   the full panel's value at a check bar differs from the value computed on data ending at that bar, ...: bar 2017-03-21 00:00:00: 'asset_00' scored nan, then 0.0388...
  PASS cache_reuse 5 prefix(es) rescored in order with state retained: unchanged
  PASS determinism the full panel scored twice by the same function gave the same output
  FAIL coverage    handed data ending at each check bar, the function produced no finite value at any of them, ...
```

A failed check is a result, not an exception. `CausalityCheckError` (a `ValueError`) means the
check could not be run: the panels are unsorted or misaligned, the panel is too short, the
function raised (its exception is chained as `__cause__`, and the message names the call and
the bar it ended at), or it returned something that cannot be compared (text, duplicate
labels, a frame where one value per asset was expected). `assert_causal` raises
`LookaheadDetected`, an `AssertionError` that carries the report as `.report`.

## Stateful code: always pass a factory

If the function is a method of an object, or a closure with a cache, pass `factory`: a
zero-argument callable that returns a new score function that has never been called.

```python
check_causality(model.score, close, factory=lambda: MyModel().score)
```

Without one, the causal reference is computed by the same object that is later handed the
full panel. A cache filled on the first, shortest prefix then hides the leak. The example and
`tests/test_causality.py::test_without_a_factory_a_cached_feature_can_hide` show a cached
forward feature that passes without a factory and fails with one. The report's `reference`
line says which way it ran.

## What it cannot see

A passing report is evidence about this function, on this panel, with these settings. It is
not a proof.

- **Data the function does not get from its arguments.** A function that reads a module-level
  frame, a file or a network service is not tested, because cutting the panels changes nothing
  it reads.
- **Code paths the panel never exercises.** A backward fill on a panel with no gaps does
  nothing, so it passes. The same applies to a corporate-action branch that never fires on the
  sample. `tests/test_causality.py::test_a_backward_fill_is_caught_only_where_the_panel_has_a_gap`
  pins this.
- **Look-ahead longer than `future_bars`.** A ten-bar look-ahead with its NaN tail filled is
  caught with 21 bars withheld and missed with 5
  (`test_a_look_ahead_longer_than_the_withheld_future_is_not_seen`). Raise `future_bars` if
  your features have long forward windows.
- **Look-ahead that only shows at other bars.** Only `check_window_bars` bars are compared.
  A leak confined to, say, month-end bars is missed unless one falls in the window. Widen the
  window or run the check at several cut points.
- **Isolation without a factory**, as above.
- **Whether the panel is point-in-time.** The checker trusts the panels it is handed. A price
  history that was revised after the fact, or a universe chosen with hindsight (survivorship),
  carries the future into every prefix and is invisible to any prefix test.

## Relation to `prism conform`

`prism.conformance.check_component`, behind `prism conform`, runs the same prefix comparisons
on a `prism.signal.base.Signal`. It uses the same order, the same defaults (5 check bars, 21
withheld) and the same cut: it fits the component on the rows before the window.
`tests/test_causality.py` runs both on the components the conformance suite's own tests use:
an honest trailing ratio, the shipped momentum node, a cached forward leak at two horizons
and two withheld lengths, a nondeterministic scorer and a stateful scorer. It checks that
they reach the same verdicts on causality, cache reuse and score determinism, that they name
the same first differing bar, asset and values, and that they cut the panel at the same bar.

What `prism conform` checks in addition, because it can read a component's declarations:
that a stateless component scores before `fit` and a stateful one refuses to; that a panel
shorter than `required_history` raises; that a name with too little history scores NaN rather
than zero; price-scale invariance when declared; that two fits write the same artifact; that a
declared feature-cache key names every element it must; and that every declaration is present.

What this checker does in addition: it takes plain functions; it reads time-indexed (panel)
outputs and compares the full panel's rows directly, which is what catches look-ahead in
vectorized code; it fails a check window with no finite value (coverage); it aligns outputs on
their labels rather than failing on a changed index; it refuses non-numeric output instead of
coercing it to NaN; and it hands the function copies of the panels.

The two are separate implementations. `prism conform` was left unchanged, and the parity test
is what keeps the shared comparisons in step.
