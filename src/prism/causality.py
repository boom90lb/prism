"""Look-ahead check for a signal function: does its value at bar t change once later bars exist?

A signal computed at the close of bar *t* may use bars up to and including *t* and
nothing after. The usual way to break that rule leaves no trace in a backtest's summary
statistics: a feature is computed once over the whole series (a forward return, a
full-sample mean or z-score, a centred rolling window, a backward fill) and then read row
by row, so the value at bar *t* carries information from bar *t + k*. This module runs a
function on prefixes of a panel and on the whole panel and compares the values it gives
for the same bars. It imports only numpy and pandas, and it takes any callable.

Two output shapes are understood:

* ``output="cross_section"``: ``fn(close, ...)`` returns the scores for the *last* bar of
  the panel it is handed, one value per asset (a Series) or a single number. This is the
  contract of :meth:`prism.signal.base.Signal.score`.
* ``output="panel"``: ``fn(close, ...)`` returns a DataFrame (or a Series) indexed by the
  bars it was handed, as vectorized pandas code usually does. The check reads the row of
  each bar it compares.

``output="auto"`` (the default) chooses from the first output: see :func:`check_causality`.

**Geometry.** The panel is cut into three blocks: the rows before the check window, the
``check_window_bars`` bars whose values are compared, and ``future_bars`` bars after the
window that only the full-panel run sees. A forward-looking feature is NaN at the end of
any panel it is computed on, so a check window at the very end of the panel would miss it.
Withholding a block of later bars is what lets the comparison catch a function that looks
up to ``future_bars`` bars ahead.

**The four checks**, run in this order because the order is what exposes a cache:

1. ``causality``. The causal reference is computed first: for each check bar *t*, the
   function is called on the panel cut at *t* (by a freshly made function per bar when a
   ``factory`` is given). Then the full panel is scored. In panel mode, the full panel's
   row for each check bar must equal the reference. In both modes, each prefix is then
   rescored, newest first, and must still equal the reference: a function that cached a
   feature computed on the full panel answers the shorter prefixes from that cache and
   is caught here.
2. ``cache_reuse``. The prefixes are rescored again, oldest first, with whatever state the
   function kept, and must equal the reference.
3. ``determinism``. The full panel is scored twice more by the same function and the two
   outputs must agree (in panel mode, at the check bars). A nondeterministic function also
   fails the comparisons above, so fix this one first.
4. ``coverage``. The reference must hold at least one finite value at some check bar.
   A function that is NaN everywhere in the window passes every comparison while having
   been tested on nothing, so that case fails instead of passing. A purely forward-looking
   feature is NaN at the end of every prefix, however far ahead it looks, so it fails here
   too.

Comparisons are bitwise by default (NaN equals NaN); ``rtol`` relaxes them to a relative
tolerance. Values are aligned on their labels, and a label missing from one output counts
as NaN there.

**What this cannot see.** A function that reads data from somewhere other than the panels
it is handed (a module-level frame, a file, a network call) is not tested, because
cutting the panels changes nothing it reads. A look-ahead on a code path the data never
exercises (a backward fill on a panel without gaps, a corporate-action branch that never
fires) is not seen. Look-ahead longer than ``future_bars`` is not seen, nor is one that
shows only at bars outside the check window. The panels are trusted: a revised price history
or a universe chosen with hindsight carries the future into every prefix. Without a
``factory``, the causal reference is computed by the same function object that is later
handed the full panel; state kept between calls is then not isolated, and a cache filled
on the first, shortest prefix can hide a leak. A passing report is evidence about this
panel and these settings, not a proof.

:func:`prism.conformance.check_component` (``prism conform``) runs the same prefix
comparisons, in the same order and with the same default window, on a
:class:`~prism.signal.base.Signal`, plus checks that need the component's declarations.
``tests/test_causality.py`` checks that the two give the same verdicts and name the same
first difference. ``docs/causality.md`` is the user guide.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CHECKS",
    "DEFAULT_CHECK_WINDOW_BARS",
    "DEFAULT_FUTURE_BARS",
    "FAIL",
    "OUTPUTS",
    "PASS",
    "SCALAR_LABEL",
    "CausalityCheckError",
    "CausalityReport",
    "Check",
    "Difference",
    "LookaheadDetected",
    "assert_causal",
    "check_causality",
]

#: Check bars compared. The same default as ``prism.conformance``.
DEFAULT_CHECK_WINDOW_BARS = 5

#: Bars after the check window that only the full-panel run sees. A function that looks
#: *k* bars ahead can be caught when *k* <= this. The same default as ``prism.conformance``.
DEFAULT_FUTURE_BARS = 21

#: Check names, in the order they run and are reported.
CHECKS: tuple[str, ...] = ("causality", "cache_reuse", "determinism", "coverage")

#: Accepted values of ``output``.
OUTPUTS: tuple[str, ...] = ("auto", "cross_section", "panel")

PASS = "PASS"
FAIL = "FAIL"

#: The label a single-number output is compared under.
SCALAR_LABEL = "value"

ScoreFunction = Callable[..., Any]
Panel = pd.DataFrame | pd.Series


class CausalityCheckError(ValueError):
    """The check could not be run, so there is no verdict.

    Raised for unusable inputs (unsorted or misaligned panels, a panel too short for the
    window) and when the function under test raises or returns something that cannot be
    compared. The error names the call that failed; the function's own exception, when
    there is one, is chained as ``__cause__``.
    """


@dataclass(frozen=True)
class Difference:
    """The first value that moved between the causal reference and another run."""

    bar: Any
    label: Any
    reference: float
    observed: float

    def __str__(self) -> str:
        return f"bar {self.bar}: {self.label!r} scored {self.reference!r}, then {self.observed!r}"


@dataclass(frozen=True)
class Check:
    """One check's outcome. ``difference`` is set when a comparison failed."""

    name: str
    status: str
    detail: str
    difference: Difference | None = None

    @property
    def passed(self) -> bool:
        return self.status == PASS


@dataclass(frozen=True)
class CausalityReport:
    """Every check against one function, with the geometry it was run under."""

    checks: tuple[Check, ...]
    output: str
    check_bars: tuple[Any, ...]
    training_rows: int
    future_bars: int
    isolated: bool
    rtol: float

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def check(self, name: str) -> Check:
        """The check called ``name`` (one of :data:`CHECKS`)."""
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(f"no check named {name!r}; the checks are {list(CHECKS)}")

    def text(self) -> str:
        """A plain-text report, one line per check."""
        reading = (
            "rows read at each check bar from a time-indexed output"
            if self.output == "panel"
            else "the output for the last bar of each panel handed in"
        )
        reference = (
            "a fresh function from `factory` for each check bar"
            if self.isolated
            else "the function under test itself (no `factory`, so state kept between calls is not isolated)"
        )
        comparison = "bitwise, NaN equal to NaN" if self.rtol == 0.0 else f"relative tolerance {self.rtol:g}"
        lines = [
            f"look-ahead check: {'PASS' if self.ok else 'FAIL'}",
            f"output       {self.output}: {reading}",
            f"check bars   {len(self.check_bars)}, {self.check_bars[0]} to {self.check_bars[-1]}; "
            f"{self.training_rows} row(s) before them, {self.future_bars} withheld after them",
            f"reference    {reference}",
            f"comparison   {comparison}",
        ]
        for check in self.checks:
            lines.append(f"  {check.status:<4} {check.name:<11} {check.detail}")
        return "\n".join(lines)


class LookaheadDetected(AssertionError):
    """Raised by :func:`assert_causal` when any check fails. ``report`` holds the full result."""

    def __init__(self, report: CausalityReport) -> None:
        super().__init__(report.text())
        self.report = report


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def check_causality(
    score: ScoreFunction,
    *panels: Panel,
    factory: Callable[[], ScoreFunction] | None = None,
    check_window_bars: int = DEFAULT_CHECK_WINDOW_BARS,
    future_bars: int = DEFAULT_FUTURE_BARS,
    output: str = "auto",
    rtol: float = 0.0,
) -> CausalityReport:
    """Run the four checks against ``score`` and return the report.

    ``score`` is called as ``score(*prefixes)``, where each prefix is a copy of the first
    rows of the corresponding entry of ``panels`` (for example ``close`` and ``volume``).
    Every panel is a DataFrame or Series indexed by bar, and all of them share one sorted,
    unique index.

    ``factory``, when given, is a zero-argument callable returning a new score function
    that has never been called. It is used for the causal reference, so state that
    ``score`` keeps between calls cannot leak into the values everything else is compared
    with. Pass one whenever ``score`` is a method of an object or a closure with a cache.
    A model that must be fitted is fitted inside the factory, on the rows before the
    check window only (``report.training_rows`` of them; the count is
    ``len(panel) - check_window_bars - future_bars``).

    ``output`` says how to read the function's result: ``"cross_section"`` (values for
    the last bar handed in), ``"panel"`` (a time-indexed result whose row at each check
    bar is read) or ``"auto"``. In ``"auto"`` the first output decides once for the whole
    run: a DataFrame or 2-D array is a panel; a Series whose labels are all bars of the
    panel handed in is a panel; any other Series, a 1-D array or a number is a cross
    section. A Series whose labels are all bars is refused as ambiguous when they are also
    the first panel's columns, or when the bars are not dates (with a 0..n-1 index a
    positional cross section looks like a time series); pass ``output`` explicitly then.

    ``rtol`` is the relative tolerance of every comparison, default 0 (bitwise).

    Raises :class:`CausalityCheckError` when the check cannot be run. A failed check is
    not an exception: read ``report.ok``, or call :func:`assert_causal`.
    """
    if not callable(score):
        raise CausalityCheckError(f"score must be callable, got {type(score).__name__}")
    if factory is not None and not callable(factory):
        raise CausalityCheckError(f"factory must be a zero-argument callable, got {type(factory).__name__}")
    if output not in OUTPUTS:
        raise CausalityCheckError(f"output must be one of {list(OUTPUTS)}, got {output!r}")
    tolerance = _validated_rtol(rtol)
    frames = _validated_panels(panels)
    window = _positive_int(check_window_bars, "check_window_bars")
    future = _positive_int(future_bars, "future_bars")
    index = frames[0].index
    n_rows = len(index)
    training_rows = n_rows - window - future
    if training_rows < 1:
        raise CausalityCheckError(
            f"the check window is {window} bar(s) and {future} bar(s) are withheld after it, so the panel needs at "
            f"least {window + future + 1} rows (at least one before the window); it has {n_rows}"
        )
    positions = tuple(range(training_rows, training_rows + window))
    run = _Run(frames, output, tolerance)

    # The causal reference: every check bar scored on data ending at that bar, oldest first.
    reference: dict[int, pd.Series] = {}
    for position in positions:
        scorer = score if factory is None else _fresh(factory)
        reference[position] = run.values_at(scorer, position, "the causal reference")

    # The full panel first, so that anything it leaves behind is in place for the rescoring.
    full = run.call(score, n_rows, "the full panel")
    full_rows: Difference | None = None
    if run.mode == "panel":
        for position in reversed(positions):
            got = run.extract(full, n_rows, position, "the full panel")
            full_rows = run.compare(reference[position], got, index[position])
            if full_rows is not None:
                break
    rescored = run.sweep(score, reversed(positions), reference, "a prefix rescored after the full panel")
    warm = run.sweep(score, positions, reference, "a prefix rescored in order with state retained")

    first = run.call(score, n_rows, "the full panel (determinism, first run)")
    second = run.call(score, n_rows, "the full panel (determinism, second run)")
    determinism: Difference | None = None
    for position in positions if run.mode == "panel" else (n_rows - 1,):
        determinism = run.compare(
            run.extract(first, n_rows, position, "the full panel (determinism, first run)"),
            run.extract(second, n_rows, position, "the full panel (determinism, second run)"),
            index[position],
        )
        if determinism is not None:
            break

    covered = sum(1 for position in positions if bool(np.isfinite(reference[position].to_numpy(dtype=float)).any()))
    mode = run.mode
    assert mode is not None  # resolved by the first call of the causal reference
    checks = (
        _causality_check(mode, window, full_rows, rescored),
        _cache_check(window, warm),
        _determinism_check(determinism),
        _coverage_check(covered, window, training_rows),
    )
    return CausalityReport(
        checks=checks,
        output=mode,
        check_bars=tuple(index[position] for position in positions),
        training_rows=training_rows,
        future_bars=future,
        isolated=factory is not None,
        rtol=tolerance,
    )


def assert_causal(score: ScoreFunction, *panels: Panel, **kwargs: Any) -> CausalityReport:
    """:func:`check_causality`, raising :class:`LookaheadDetected` (an ``AssertionError``) on any failed check.

    Returns the report when every check passes, so it fits in a test::

        def test_my_signal_has_no_lookahead():
            assert_causal(my_signal, close, volume, factory=make_signal)
    """
    report = check_causality(score, *panels, **kwargs)
    if not report.ok:
        raise LookaheadDetected(report)
    return report


# --------------------------------------------------------------------------
# Check verdicts
# --------------------------------------------------------------------------


def _causality_check(mode: str, window: int, full_rows: Difference | None, rescored: Difference | None) -> Check:
    if full_rows is not None:
        return Check(
            "causality",
            FAIL,
            "the full panel's value at a check bar differs from the value computed on data ending at that bar, "
            f"so the function uses bars after the one it scores: {full_rows}",
            full_rows,
        )
    if rescored is not None:
        return Check(
            "causality",
            FAIL,
            "rescoring a prefix after the full panel was scored changed its value, so something kept from the "
            f"longer panel reaches earlier bars: {rescored}",
            rescored,
        )
    what = (
        "the full panel's rows, and each prefix rescored after it, matched"
        if mode == "panel"
        else "each prefix rescored after the full panel matched"
    )
    return Check("causality", PASS, f"{window} check bar(s): {what} the causal reference")


def _cache_check(window: int, warm: Difference | None) -> Check:
    if warm is not None:
        return Check(
            "cache_reuse",
            FAIL,
            "rescoring the prefixes in order, with whatever state the function kept, disagreed with the causal "
            f"reference, so retained state serves stale or future-contaminated values: {warm}",
            warm,
        )
    return Check("cache_reuse", PASS, f"{window} prefix(es) rescored in order with state retained: unchanged")


def _determinism_check(difference: Difference | None) -> Check:
    if difference is not None:
        return Check(
            "determinism",
            FAIL,
            "the same function scored the same panel twice and disagreed, so its output is not reproducible; this "
            f"alone also fails the comparisons above, so fix it first: {difference}",
            difference,
        )
    return Check("determinism", PASS, "the full panel scored twice by the same function gave the same output")


def _coverage_check(covered: int, window: int, training_rows: int) -> Check:
    if covered == 0:
        return Check(
            "coverage",
            FAIL,
            "handed data ending at each check bar, the function produced no finite value at any of them, so a "
            "passing comparison would mean nothing; either it needs more history than the "
            f"{training_rows} row(s) before the window, or its values exist only once later bars are present",
        )
    return Check("coverage", PASS, f"the causal reference has finite values at {covered} of {window} check bar(s)")


# --------------------------------------------------------------------------
# One run: calling, reading and comparing outputs
# --------------------------------------------------------------------------


class _Run:
    """The panels, the resolved output mode and the tolerance shared by every call of one check."""

    def __init__(self, frames: Sequence[Panel], output: str, rtol: float) -> None:
        self.frames = tuple(frames)
        self.index = frames[0].index
        self.mode: str | None = None if output == "auto" else output
        self.rtol = rtol

    def call(self, fn: ScoreFunction, rows: int, arm: str) -> Any:
        """``fn`` on copies of the first ``rows`` rows, so a function that mutates its input cannot alter the panel."""
        prefixes = [frame.iloc[:rows].copy() for frame in self.frames]
        try:
            return fn(*prefixes)
        except Exception as exc:  # noqa: BLE001 - re-raised with the call that failed
            raise CausalityCheckError(
                f"the score function raised {type(exc).__name__} on {arm} ({rows} row(s), ending at "
                f"{self.index[rows - 1]}): {exc}"
            ) from exc

    def values_at(self, fn: ScoreFunction, position: int, arm: str) -> pd.Series:
        """The values ``fn`` gives for the bar at ``position`` when handed data ending at that bar."""
        out = self.call(fn, position + 1, arm)
        return self.extract(out, position + 1, position, arm)

    def sweep(
        self, fn: ScoreFunction, positions: Any, reference: Mapping[int, pd.Series], arm: str
    ) -> Difference | None:
        """Rescore prefixes in the given order; stop at the first that differs from the reference."""
        for position in positions:
            difference = self.compare(reference[position], self.values_at(fn, position, arm), self.index[position])
            if difference is not None:
                return difference
        return None

    def extract(self, out: Any, rows: int, position: int, arm: str) -> pd.Series:
        """Read one bar's values out of an output, resolving the mode on the first output seen."""
        handed = self.index[:rows]
        if self.mode is None:
            self.mode = _resolve_mode(out, handed, self.frames[0], arm)
        if self.mode == "cross_section":
            values = _cross_section_values(out, arm)
        else:
            values = _panel_values(out, handed, self.index[position], arm)
        return _numeric(values, arm)

    def compare(self, reference: pd.Series, observed: pd.Series, bar: Any) -> Difference | None:
        return _first_difference(reference, observed, bar=bar, rtol=self.rtol)


def _fresh(factory: Callable[[], ScoreFunction]) -> ScoreFunction:
    try:
        scorer = factory()
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise CausalityCheckError(f"factory() raised {type(exc).__name__}: {exc}") from exc
    if not callable(scorer):
        raise CausalityCheckError(f"factory() returned {type(scorer).__name__}, not a callable score function")
    return scorer


def _resolve_mode(out: Any, handed: pd.Index, first: Panel, arm: str) -> str:
    if isinstance(out, pd.DataFrame):
        return "panel"
    if isinstance(out, np.ndarray):
        return "panel" if out.ndim == 2 else "cross_section"
    if isinstance(out, pd.Series):
        labels = out.index
        if len(labels) > 0 and bool(labels.isin(handed).all()):
            dated_bars = isinstance(handed, (pd.DatetimeIndex, pd.PeriodIndex))
            also_columns = isinstance(first, pd.DataFrame) and bool(labels.isin(first.columns).all())
            if also_columns or not dated_bars:
                raise CausalityCheckError(
                    f"{arm} returned a Series whose labels are all bars of the panel, but "
                    + ("they are also its columns" if also_columns else "the bars are not dates")
                    + ", so it is unclear whether it holds one value per asset or one value per bar; pass "
                    "output='cross_section' or output='panel'"
                )
            return "panel"
    return "cross_section"


def _cross_section_values(out: Any, arm: str) -> pd.Series:
    if isinstance(out, pd.Series):
        return out
    if isinstance(out, pd.DataFrame):
        raise CausalityCheckError(
            f"{arm} returned a DataFrame, but with output='cross_section' a function returns one value per asset "
            "(a Series) or one number for the last bar; for a time-indexed result use output='panel'"
        )
    if isinstance(out, np.ndarray):
        if out.ndim != 1:
            raise CausalityCheckError(f"{arm} returned a {out.ndim}-D array; a cross section is 1-D")
        return pd.Series(out)
    if _is_number(out):
        return pd.Series([out], index=[SCALAR_LABEL])
    raise CausalityCheckError(f"{arm} returned {type(out).__name__}; expected a Series, a 1-D array or a number")


def _panel_values(out: Any, handed: pd.Index, stamp: Any, arm: str) -> pd.Series:
    if isinstance(out, np.ndarray):
        if out.ndim not in (1, 2) or out.shape[0] != len(handed):
            raise CausalityCheckError(
                f"{arm} returned an array of shape {out.shape}; a panel output has one row per bar handed in "
                f"({len(handed)})"
            )
        out = pd.Series(out, index=handed) if out.ndim == 1 else pd.DataFrame(out, index=handed)
    if not isinstance(out, (pd.DataFrame, pd.Series)):
        raise CausalityCheckError(
            f"{arm} returned {type(out).__name__}; with output='panel' a function returns a DataFrame or Series "
            "indexed by bar"
        )
    if out.index.has_duplicates:
        raise CausalityCheckError(f"{arm} returned an output whose bar index has duplicates")
    if len(out.index) > 0 and len(out.index.intersection(handed)) == 0:
        raise CausalityCheckError(
            f"{arm} returned an output none of whose index labels is a bar of the panel handed in; with "
            "output='panel' the result must be indexed by bar (for one value per asset use output='cross_section')"
        )
    if isinstance(out, pd.DataFrame):
        if stamp in out.index:
            return out.loc[stamp]
        return pd.Series(np.nan, index=out.columns, dtype=float)
    value = out.loc[stamp] if stamp in out.index else np.nan
    return pd.Series([value], index=[SCALAR_LABEL])


def _numeric(values: pd.Series, arm: str) -> pd.Series:
    if values.index.has_duplicates:
        duplicated = values.index[values.index.duplicated()][0]
        raise CausalityCheckError(f"{arm} returned the label {duplicated!r} more than once")
    numeric = pd.to_numeric(values, errors="coerce")
    lost = numeric.isna() & values.notna()
    if bool(lost.any()):
        label = values.index[np.flatnonzero(lost.to_numpy())[0]]
        raise CausalityCheckError(
            f"{arm} returned a non-numeric value for {label!r}: {values[label]!r}; only numbers can be compared"
        )
    return numeric.astype(float)


def _first_difference(reference: pd.Series, observed: pd.Series, *, bar: Any, rtol: float) -> Difference | None:
    """The first label whose value moved, or None. ``rtol=0`` means bitwise, with NaN equal to NaN."""
    labels = reference.index
    if not observed.index.equals(labels):
        labels = labels.union(observed.index, sort=False)
    left = reference.reindex(labels).to_numpy(dtype=float)
    right = observed.reindex(labels).to_numpy(dtype=float)
    same = (np.isnan(left) & np.isnan(right)) | (left == right)
    if rtol > 0.0:
        with np.errstate(invalid="ignore"):
            same = same | (np.abs(left - right) <= rtol * np.maximum(np.abs(left), np.abs(right)))
    if bool(same.all()):
        return None
    at = int(np.flatnonzero(~same)[0])
    return Difference(bar=bar, label=labels[at], reference=float(left[at]), observed=float(right[at]))


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _validated_panels(panels: Sequence[Any]) -> tuple[Panel, ...]:
    if not panels:
        raise CausalityCheckError("pass at least one panel (for example close) after the score function")
    for position, panel in enumerate(panels):
        if not isinstance(panel, (pd.DataFrame, pd.Series)):
            raise CausalityCheckError(
                f"panel {position} is {type(panel).__name__}; every panel is a DataFrame or Series indexed by bar"
            )
    index = panels[0].index
    if len(index) == 0:
        raise CausalityCheckError("the first panel is empty")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise CausalityCheckError("the panels' bar index must be sorted ascending with no duplicates")
    for position, panel in enumerate(panels[1:], start=1):
        if not panel.index.equals(index):
            raise CausalityCheckError(f"panel {position} does not share the first panel's bar index")
    return tuple(panels)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 1:
        raise CausalityCheckError(f"{name} must be an integer >= 1, got {value!r}")
    return int(value)


def _validated_rtol(rtol: Any) -> float:
    if isinstance(rtol, bool) or not isinstance(rtol, numbers.Real):
        raise CausalityCheckError(f"rtol must be a number, got {rtol!r}")
    value = float(rtol)
    if not math.isfinite(value) or value < 0.0:
        raise CausalityCheckError(f"rtol must be finite and >= 0, got {rtol!r}")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (numbers.Real, np.bool_))
