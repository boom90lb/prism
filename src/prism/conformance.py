"""The component conformance suite of the engine specification §3.3, behind ``prism conform``.

A private component is trusted executable code: the engine imports it and runs
it in the same process, with the same privileges, as the code that submits
orders. Conformance is therefore not a sandbox and cannot make an untrusted
plugin safe. What it does is check, on real bars, the behavioural promises the
engine relies on when it composes a component into a book — the promises whose
violation is invisible in a backtest's summary statistics and fatal in
production:

* **Causality (N1).** The value a component emits for bar *t* is the same
  whether the panel it was handed ends at *t* or extends past it. The failure
  this catches is the classic one: a derived feature computed once over the
  whole series and then sliced per bar, so bar *t* carries information from
  *t + k*. The suite scores the long panel first, precisely so a component that
  caches a non-causal feature answers from the poisoned cache and is caught.
* **NaN, never a fabricated zero.** A name without enough history scores NaN.
  Zero is a position-space instruction ("hold nothing") and a component that
  emits it for "I don't know" silently trades on ignorance.
* **Determinism.** The same fitted object, on the same panel, twice, bitwise.
* **Declaration completeness**, since the engine reads behaviour off the
  declarations (§3.3) and an absent one is an unanswerable question.

A component declaring ``fitted_state = "train_only"`` is measured, not refused.
The refusal that stood here existed for one reason — nothing could persist a
fit, so a fit run during conformance was lost between the check and the
decision it was supposed to drive — and :mod:`prism.fits` now writes that
artifact under ``workspace.state / "fits"`` and names it by content hash. Two
checks change with it: check 1 requires a stateful component to *raise* when
scored before ``fit`` (the reverse of the stateless requirement), and check 6
adds fit determinism — two independent fits on the same declared window produce
the same ``fitted_state_ref``, which is checkable only because the artifact is
real.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism import fits
from prism.fits import FitArtifactError
from prism.io.dataset import Dataset
from prism.signal.base import DeclarationError, Signal, check_declarations
from prism.spec import Specification, instantiate_component
from prism.workspace import Workspace

#: The five elements §3.3 requires a feature cache key to carry. A component
#: that caches derived features declares its key by implementing
#: ``feature_cache_key()``; a key omitting any element can collide across
#: datasets, params, fit cutoffs, or scoring bases, and serving a colliding
#: entry is exactly the leak check 2 exists to prevent.
CACHE_KEY_ELEMENTS: tuple[str, ...] = (
    "data_id",
    "implementation_identity",
    "params",
    "fit_cutoff",
    "scoring_basis",
)

#: The checks of §3.3, in specification order. The report prints them in it.
CHECKS: tuple[str, ...] = (
    "fit_boundary",
    "causality",
    "cache_reuse",
    "history",
    "price_scale_invariance",
    "determinism",
    "declarations",
)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

#: Decision bars the causality sweep scores. Five is enough for a prefix sweep
#: to cross a decision cadence and small enough that the suite stays a few
#: seconds on a hundred-name panel.
DEFAULT_CHECK_WINDOW_BARS = 5

#: Bars withheld after the check window, which the contaminating arm of check 2
#: is allowed to see and the causal reference is not. This is the lookahead the
#: suite can detect: a component peeking k bars past its decision bar is caught
#: when k <= DEFAULT_FUTURE_BARS. A window sitting at the very end of the panel
#: would catch nothing, because a forward-looking feature is NaN there.
DEFAULT_FUTURE_BARS = 21

#: The rescale factor of check 5. It is > 1 deliberately: a component's
#: eligibility screen gates on absolute price and dollar volume, so scaling the
#: close panel *up* stays inside the eligible region, while scaling it down
#: would cross an eligibility boundary and refute a property the component
#: never claimed (§3.3).
DEFAULT_RESCALE_FACTOR = 3.0

#: Tolerance of check 5. ``(a * c) / (b * c)`` is not bitwise ``a / b``, so the
#: invariance claim is a numerical one and is checked as such.
RESCALE_RTOL = 1e-9

TRUSTED_PLUGIN_STATEMENT = (
    "A private plugin is trusted executable code [Q7]. It is imported into this process and runs with the "
    "engine's privileges: it can read the workspace, reach the network, and submit orders through any credential "
    "this process holds. `prism conform` checks declared behaviour on a sample of real bars, not intent; it is "
    "not a sandbox, it does not isolate the component, and a passing report is not a safety verdict. Install "
    "only components you or your team wrote and reviewed."
)


class ConformanceError(ValueError):
    """A component is refused before its behaviour can be checked.

    Refusal is not a failed check: a component whose declarations are unreadable
    or whose fitted state this stage cannot persist is not measurable, and
    reporting a verdict on it would be a fabricated result.
    """


@dataclass(frozen=True)
class CheckResult:
    """One §3.3 check against one component."""

    number: int
    name: str
    status: str
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == FAIL


@dataclass(frozen=True)
class ComponentReport:
    """Every check against one component, with the declarations they were read under."""

    component_id: str
    class_name: str
    declarations: Mapping[str, Any]
    checks: tuple[CheckResult, ...]
    cache_declared: bool

    @property
    def ok(self) -> bool:
        return not any(check.failed for check in self.checks)


@dataclass(frozen=True)
class ConformanceReport:
    """The result of ``prism conform``: one component report per declared component."""

    spec_path: Path | None
    dataset: Mapping[str, Any]
    components: tuple[ComponentReport, ...]

    @property
    def ok(self) -> bool:
        return all(component.ok for component in self.components)

    def text(self) -> str:
        """The printed report, including the trusted-plugin statement (§3.3)."""
        lines: list[str] = ["prism conform"]
        if self.spec_path is not None:
            lines.append(f"specification  {self.spec_path}")
        for key, value in self.dataset.items():
            lines.append(f"{key:<14} {value}")
        for component in self.components:
            lines.append("")
            lines.append(f"component {component.component_id!r}  ({component.class_name})")
            declared = ", ".join(f"{k}={v!r}" for k, v in component.declarations.items())
            lines.append(f"  declarations  {declared}")
            lines.append(f"  feature cache {'declared' if component.cache_declared else 'not declared'}")
            for check in component.checks:
                lines.append(f"  {check.status:<4} {check.number} {check.name:<23} {check.detail}")
        failed = sum(1 for c in self.components for check in c.checks if check.failed)
        lines.append("")
        lines.append(f"{len(self.components)} component(s), {failed} failed check(s)")
        lines.append("")
        lines.append(TRUSTED_PLUGIN_STATEMENT)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------


def check_component(
    component: Signal,
    *,
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    component_id: str | None = None,
    factory: Callable[[], Signal] | None = None,
    check_window_bars: int = DEFAULT_CHECK_WINDOW_BARS,
    future_bars: int = DEFAULT_FUTURE_BARS,
    rescale_factor: float = DEFAULT_RESCALE_FACTOR,
) -> ComponentReport:
    """Run the seven §3.3 checks against ``component`` on ``close`` (and ``volume``).

    The panel is cut into three: a training slice ending at the fit cutoff, then
    ``check_window_bars`` decision bars the sweep scores, then ``future_bars``
    bars that only the contaminating arm of check 2 ever sees. That last cut is
    what gives the causality check teeth — a component that peeks *k* bars ahead
    emits NaN at the end of any panel, so a check window sitting at the panel's
    end would exonerate it. ``factory`` returns a freshly constructed component
    and is what makes the cache-cold arm genuinely cold; without one the same
    instance is reused and the report says so, which is weaker.

    Raises ``ConformanceError`` when the component cannot be measured at all:
    unreadable declarations, or a panel too short for the component's own
    ``required_history``. A component declaring fitted state is measured like
    any other; what changes is checks 1 and 6, which read the declaration.
    """
    label = f"component {component_id!r} ({type(component).__name__})" if component_id else type(component).__name__
    try:
        declarations = check_declarations(component, configured_id=component_id)
    except DeclarationError as exc:
        raise ConformanceError(f"{label} is refused: {exc}") from exc

    close, volume = _validated_panels(close, volume, label=label)
    required = int(declarations["required_history"])
    window = int(check_window_bars)
    future = int(future_bars)
    if window < 1:
        raise ConformanceError(f"{label}: check_window_bars must be >= 1, got {check_window_bars}")
    if future < 1:
        raise ConformanceError(
            f"{label}: future_bars must be >= 1, or the causality sweep has no future to be contaminated by, "
            f"got {future_bars}"
        )
    if len(close) < required + window + future:
        raise ConformanceError(
            f"{label} declares required_history={required}, the check window is {window} bar(s) and {future} bar(s) "
            f"are withheld as the future, so the suite needs a panel of at least {required + window + future} rows; "
            f"it was given {len(close)}"
        )

    train_rows = len(close) - window - future
    positions = list(range(train_rows, train_rows + window))
    train_close, train_volume = _prefix(close, volume, train_rows)
    fit_cutoff = close.index[train_rows - 1]

    checks: list[CheckResult] = []
    checks.append(
        _check_fit_boundary(
            component,
            factory,
            train_close,
            train_volume,
            label=label,
            stateful=declarations["fitted_state"] != "none",
        )
    )
    component.fit(train_close, train_volume)

    cold, cold_detail = _cold_pass(component, factory, close, volume, positions, train_close, train_volume)
    causality, cache_reuse = _causality_checks(component, close, volume, positions, cold, cold_detail)
    checks.append(causality)
    cache_declared, cache_key_check = _cache_key_check(component)
    if cache_key_check is not None and cache_reuse.status == PASS:
        checks.append(cache_key_check)
    else:
        checks.append(cache_reuse)
    checks.append(_check_history(component, factory, close, volume, positions, cold, required, train_rows))
    checks.append(
        _check_price_scale_invariance(
            component,
            factory,
            close,
            volume,
            positions,
            cold,
            declared=bool(declarations["price_scale_invariant"]),
            factor=float(rescale_factor),
        )
    )
    checks.append(
        _check_determinism(
            component,
            factory,
            close,
            volume,
            declarations,
            train_close=train_close,
            train_volume=train_volume,
            component_id=component_id or str(declarations["component_id"]) or type(component).__name__,
        )
    )
    checks.append(_check_declarations(declarations, fit_cutoff))
    return ComponentReport(
        component_id=component_id or declarations["component_id"] or type(component).__name__,
        class_name=type(component).__name__,
        declarations=dict(declarations),
        checks=tuple(sorted(checks, key=lambda c: c.number)),
        cache_declared=cache_declared,
    )


def _check_fit_boundary(
    component: Signal,
    factory: Callable[[], Signal] | None,
    train_close: pd.DataFrame,
    train_volume: pd.DataFrame | None,
    *,
    label: str,
    stateful: bool,
) -> CheckResult:
    """Check 1, both directions of the declaration (§3.3).

    A stateless component must score before ``fit`` without raising. A component
    declaring ``train_only`` must raise: a stateful node that quietly scores
    unfitted emits numbers from an unfitted model, and every later check in this
    suite — and every decision downstream — would be measuring that.
    """
    unfitted = factory() if factory is not None else component
    if stateful:
        if factory is None:
            return CheckResult(
                1,
                "fit boundary",
                SKIP,
                "declares fitted state, and the boundary can only be checked on a component that has never "
                "been fitted; supply a factory so the suite can construct one",
            )
        try:
            unfitted.score(train_close, train_volume)
        except Exception as exc:  # noqa: BLE001 - the check is precisely "does this raise?"
            return CheckResult(
                1,
                "fit boundary",
                PASS,
                f"declares fitted state and score() before fit() raised {type(exc).__name__}, as it must; "
                f"fitted on {len(train_close)} training bars",
            )
        return CheckResult(
            1,
            "fit boundary",
            FAIL,
            "declares fitted state but score() before fit() returned a value instead of raising: an unfitted "
            "model emitting scores is a book built on nothing (§3.3 check 1)",
        )
    try:
        unfitted.score(train_close, train_volume)
    except Exception as exc:  # noqa: BLE001 - the check is precisely "does this raise?"
        return CheckResult(
            1,
            "fit boundary",
            FAIL,
            f"declares fitted_state='none' but score() before fit() raised {type(exc).__name__}: {exc}",
        )
    return CheckResult(
        1,
        "fit boundary",
        PASS,
        f"fitted_state='none': scored before fit() without raising; fitted on {len(train_close)} training bars",
    )


def _cold_pass(
    component: Signal,
    factory: Callable[[], Signal] | None,
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    positions: Sequence[int],
    train_close: pd.DataFrame,
    train_volume: pd.DataFrame | None,
) -> tuple[dict[int, pd.Series], str]:
    """The causal reference: each prefix scored by a component that has seen nothing longer.

    Ascending order, one freshly constructed and fitted component per prefix
    where a factory is available. This is the "no future bar was ever in the
    room" arm; every later arm is compared against it.
    """
    values: dict[int, pd.Series] = {}
    for position in positions:
        scorer = factory() if factory is not None else component
        if factory is not None:
            scorer.fit(train_close, train_volume)
        prefix_close, prefix_volume = _prefix(close, volume, position + 1)
        values[position] = scorer.score(prefix_close, prefix_volume)
    detail = (
        "one freshly constructed component per prefix"
        if factory is not None
        else "the same instance per prefix (no factory given; a component holding hidden state is not isolated)"
    )
    return values, detail


def _causality_checks(
    component: Signal,
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    positions: Sequence[int],
    cold: Mapping[int, pd.Series],
    cold_detail: str,
) -> tuple[CheckResult, CheckResult]:
    """Checks 2 and 3: prefix equality after a longer panel, then again with the cache warm.

    The long panel is scored *first*. A component that computes a derived
    feature over everything it is handed and caches it by name will then answer
    the earlier prefixes out of a cache built from bars after those prefixes
    ended — which is the leak, and it shows up as a value difference against the
    cold reference.
    """
    component.score(close, volume)
    long_first = _compare_prefixes(component, close, volume, list(reversed(list(positions))), cold)
    warm = _compare_prefixes(component, close, volume, positions, cold)

    if long_first is None:
        causality = CheckResult(
            2,
            "causality",
            PASS,
            f"{len(positions)} prefix bar(s): unchanged after the full panel was scored first; "
            f"reference from {cold_detail}",
        )
    else:
        causality = CheckResult(
            2,
            "causality",
            FAIL,
            "appending future bars changed an earlier score, so the component is not causal (N1): " + long_first,
        )
    if warm is None:
        cache = CheckResult(
            3,
            "cache reuse",
            PASS,
            f"{len(positions)} prefix bar(s) rescored with the component's state warm: unchanged",
        )
    else:
        cache = CheckResult(
            3,
            "cache reuse",
            FAIL,
            "a warm rerun disagreed with the cold reference, so retained state is serving stale or "
            "future-contaminated features: " + warm,
        )
    return causality, cache


def _compare_prefixes(
    component: Signal,
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    positions: Sequence[int],
    cold: Mapping[int, pd.Series],
) -> str | None:
    for position in positions:
        prefix_close, prefix_volume = _prefix(close, volume, position + 1)
        got = component.score(prefix_close, prefix_volume)
        difference = _first_difference(cold[position], got)
        if difference is not None:
            return f"bar {close.index[position]}: {difference}"
    return None


def _cache_key_check(component: Signal) -> tuple[bool, CheckResult | None]:
    """Check 3's key hygiene, for a component that declares a feature cache.

    A component declares one by implementing ``feature_cache_key()``; the key it
    returns must name every element of ``CACHE_KEY_ELEMENTS``, because a key
    omitting one collides across datasets, params, fit cutoffs, or scoring
    bases and hands back a neighbour's features.
    """
    getter = getattr(component, "feature_cache_key", None)
    if getter is None:
        return False, None
    try:
        key = getter()
    except Exception as exc:  # noqa: BLE001 - a key that cannot be produced is a failed key
        return True, CheckResult(3, "cache reuse", FAIL, f"feature_cache_key() raised {type(exc).__name__}: {exc}")
    if not isinstance(key, Mapping):
        return True, CheckResult(
            3, "cache reuse", FAIL, f"feature_cache_key() returned {type(key).__name__}, not a mapping"
        )
    missing = [element for element in CACHE_KEY_ELEMENTS if element not in key]
    if missing:
        return True, CheckResult(
            3,
            "cache reuse",
            FAIL,
            f"the declared feature cache key omits {missing}; a key missing any of {list(CACHE_KEY_ELEMENTS)} "
            "collides across datasets, params, fit cutoffs, or scoring bases",
        )
    return True, None


def _check_history(
    component: Signal,
    factory: Callable[[], Signal] | None,
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    positions: Sequence[int],
    cold: Mapping[int, pd.Series],
    required: int,
    train_rows: int,
) -> CheckResult:
    """Check 4: a short panel raises (N7); inside a long one, a short name scores NaN."""
    short_close, short_volume = _prefix(close, volume, max(required - 1, 1))
    if required > 1:
        try:
            component.score(short_close, short_volume)
        except Exception:  # noqa: BLE001 - raising is the required behaviour
            pass
        else:
            return CheckResult(
                4,
                "history",
                FAIL,
                f"a panel of {len(short_close)} rows, below required_history={required}, returned a score "
                "instead of raising (N7)",
            )

    last = positions[-1]
    reference = cold[last]
    finite = [sid for sid, value in reference.items() if np.isfinite(pd.to_numeric(value, errors="coerce"))]
    if not finite:
        return CheckResult(
            4,
            "history",
            FAIL,
            "no name scored a finite value at the last check bar, so the NaN-on-short-history half of the check "
            "would be vacuous; the panel or the component's eligibility screen admits nothing",
        )
    sid = finite[0]
    keep = required - 1
    blank_close = close.copy()
    blank_close.loc[blank_close.index[:-keep], sid] = np.nan
    blank_volume = None
    if volume is not None:
        blank_volume = volume.copy()
        blank_volume.loc[blank_volume.index[:-keep], sid] = np.nan
    scorer = factory() if factory is not None else component
    train_close, train_volume = _prefix(blank_close, blank_volume, train_rows)
    try:
        if factory is not None:
            scorer.fit(train_close, train_volume)
        scored = scorer.score(blank_close, blank_volume)
    except Exception as exc:  # noqa: BLE001 - a raise here is a verdict, not a crash
        return CheckResult(
            4,
            "history",
            FAIL,
            f"a name with {keep} of {required} required bars raised {type(exc).__name__} inside a panel of "
            f"sufficient length; §3.3 asks for NaN on that name, not a refusal of the whole cross-section: {exc}",
        )
    value = pd.to_numeric(scored.get(sid), errors="coerce")
    if not pd.isna(value):
        return CheckResult(
            4,
            "history",
            FAIL,
            f"sid {sid!r} with {keep} of {required} required bars scored {float(value)!r}; a name the component "
            "cannot score is NaN, never a fabricated value (0.0 is a position instruction, not an opinion)",
        )
    return CheckResult(
        4,
        "history",
        PASS,
        f"a panel below required_history={required} raises; sid {sid!r} truncated to {keep} bars scores NaN",
    )


def _check_price_scale_invariance(
    component: Signal,
    factory: Callable[[], Signal] | None,
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    positions: Sequence[int],
    cold: Mapping[int, pd.Series],
    *,
    declared: bool,
    factor: float,
) -> CheckResult:
    """Check 5, run exactly when ``price_scale_invariant`` is True; the declaration is recorded either way."""
    if not declared:
        return CheckResult(
            5,
            "price-scale invariance",
            SKIP,
            "price_scale_invariant=False: the rescale test is not run and no value of units implies it (§3.3)",
        )
    if factor <= 1.0:
        raise ConformanceError(
            f"rescale_factor must be > 1 so the rescaled panel stays inside the eligible region, got {factor}"
        )
    last = positions[-1]
    scaled_close = close * factor
    scorer = factory() if factory is not None else component
    train_rows = positions[0]
    train_close, train_volume = _prefix(scaled_close, volume, train_rows)
    if factory is not None:
        scorer.fit(train_close, train_volume)
    prefix_close, prefix_volume = _prefix(scaled_close, volume, last + 1)
    scored = scorer.score(prefix_close, prefix_volume)

    # The claim holds inside the eligible region, and only there: a component's
    # eligibility screen gates on absolute price and dollar volume, both
    # monotone in the rescale factor, so a factor above 1 can only *admit*
    # names. A name the screen admits at 3x and refuses at 1x has no reference
    # value to be invariant against; a name with a reference value that goes
    # NaN or moves is a genuine scale dependence and fails.
    reference = pd.to_numeric(cold[last], errors="coerce")
    eligible = reference.index[np.isfinite(reference.to_numpy(dtype=float))]
    if len(eligible) == 0:
        return CheckResult(
            5,
            "price-scale invariance",
            FAIL,
            "no name scored a finite value at the last check bar, so the rescale comparison would be vacuous",
        )
    admitted = int(pd.to_numeric(scored, errors="coerce").notna().sum()) - len(eligible)
    difference = _first_difference(reference.loc[eligible], scored.reindex(eligible), rtol=RESCALE_RTOL)
    if difference is not None:
        return CheckResult(
            5,
            "price-scale invariance",
            FAIL,
            f"multiplying every close by {factor} changed the score, against the declaration "
            f"price_scale_invariant=True: {difference}",
        )
    note = ""
    if admitted:
        note = (
            f"; {admitted} further name(s) score only at the rescaled price level and are outside the comparison, "
            "because eligibility gates on absolute price and dollar volume"
        )
    return CheckResult(
        5,
        "price-scale invariance",
        PASS,
        f"every close multiplied by {factor} leaves the {len(eligible)} scored name(s) equal to within "
        f"rtol={RESCALE_RTOL:g}{note}",
    )


def _check_determinism(
    component: Signal,
    factory: Callable[[], Signal] | None,
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    declarations: Mapping[str, Any],
    *,
    train_close: pd.DataFrame,
    train_volume: pd.DataFrame | None,
    component_id: str,
) -> CheckResult:
    """Check 6, split in two (§3.3).

    Score determinism on a fixed fitted state is required of every component.
    Fit determinism — two independent fits on the same declared window producing
    the same ``fitted_state_ref`` — is required of every component declaring
    fitted state, and is checkable because the fit is a real artifact
    (:mod:`prism.fits`): the two fits are written into a scratch store and their
    content hashes compared. A legitimately randomized fit satisfies it by
    hashing its declared seed into the artifact.
    """
    first = component.score(close, volume)
    second = component.score(close, volume)
    difference = _first_difference(first, second)
    if difference is not None:
        return CheckResult(
            6,
            "determinism",
            FAIL,
            "the same fitted object scored the same panel twice and disagreed, so the emitted book is not "
            "reproducible from its recorded inputs: " + difference,
        )
    if declarations["fitted_state"] == "none":
        return CheckResult(
            6,
            "determinism",
            PASS,
            "score is bitwise identical on a rerun; fit determinism does not apply to "
            f"fitted_state={declarations['fitted_state']!r}",
        )
    if factory is None:
        return CheckResult(
            6,
            "determinism",
            SKIP,
            "score is bitwise identical on a rerun, but fit determinism needs two independently constructed "
            "components and no factory was supplied; it was not run rather than assumed",
        )
    refs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="prism-conform-fits-") as scratch:
        for _ in range(2):
            fresh = factory()
            fitted = fresh.fit(train_close, train_volume)
            if fitted is None:
                return CheckResult(
                    6,
                    "determinism",
                    FAIL,
                    "fit() returned None; the contract is 'fits on the training panel and returns self' (§3.3), "
                    "so the fitted object cannot be persisted or compared",
                )
            try:
                ref = fits.save(fitted, component_id=component_id, fits_dir=Path(scratch))
            except FitArtifactError as exc:
                return CheckResult(
                    6,
                    "determinism",
                    FAIL,
                    f"declares fitted_state={declarations['fitted_state']!r} and its fit could not be "
                    f"persisted, so it has no fitted_state_ref to compare or to replay: {exc}",
                )
            refs.append(str(ref))
    if refs[0] != refs[1]:
        return CheckResult(
            6,
            "determinism",
            FAIL,
            "two independent fits on the same training window wrote different artifacts "
            f"({refs[0]} vs {refs[1]}), so a recorded fitted_state_ref does not identify the fit that "
            "produced a book; a randomized fit satisfies this by hashing its declared seed",
        )
    return CheckResult(
        6,
        "determinism",
        PASS,
        f"score is bitwise identical on a rerun, and two independent fits on {len(train_close)} training bars "
        f"wrote the same artifact ({refs[0]})",
    )


def _check_declarations(declarations: Mapping[str, Any], fit_cutoff: Any) -> CheckResult:
    """Check 7. ``check_declarations`` already refused an incomplete set; this records the complete one."""
    return CheckResult(
        7,
        "declarations",
        PASS,
        f"all {len(declarations)} declared field(s) present and consistent (estimated_decay_bars="
        f"{declarations['estimated_decay_bars']!r} satisfies presence); checked at fit cutoff {fit_cutoff}",
    )


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def _validated_panels(
    close: pd.DataFrame, volume: pd.DataFrame | None, *, label: str
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not isinstance(close, pd.DataFrame):
        raise ConformanceError(f"{label}: close must be a wide DataFrame panel, got {type(close).__name__}")
    if close.empty:
        raise ConformanceError(f"{label}: the close panel is empty; conformance is checked on real bars")
    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
        raise ConformanceError(f"{label}: the close panel index must be sorted ascending and unique")
    if volume is not None and (not close.index.equals(volume.index) or not close.columns.equals(volume.columns)):
        raise ConformanceError(f"{label}: the close and volume panels must share index and columns")
    return close, volume


def _prefix(close: pd.DataFrame, volume: pd.DataFrame | None, rows: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    return close.iloc[:rows], None if volume is None else volume.iloc[:rows]


def _first_difference(reference: pd.Series, other: pd.Series, *, rtol: float = 0.0) -> str | None:
    """The first sid whose value moved, or None. ``rtol=0`` means bitwise (NaN equals NaN)."""
    if not isinstance(other, pd.Series):
        return f"score returned {type(other).__name__}, not a Series"
    if not other.index.equals(reference.index):
        return f"the score index changed: {list(other.index)[:3]}... against {list(reference.index)[:3]}..."
    left = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(other, errors="coerce").to_numpy(dtype=float)
    same = (np.isnan(left) & np.isnan(right)) | (left == right)
    if rtol > 0.0:
        with np.errstate(invalid="ignore"):
            close_enough = np.abs(left - right) <= rtol * np.maximum(np.abs(left), np.abs(right))
        same = same | np.nan_to_num(close_enough, nan=False)
    if bool(same.all()):
        return None
    index = int(np.flatnonzero(~same)[0])
    return f"sid {reference.index[index]!r} scored {left[index]!r}, then {right[index]!r}"


# --------------------------------------------------------------------------
# Specification entry point
# --------------------------------------------------------------------------


def conform_specification(
    spec: Specification,
    *,
    workspace: Workspace,
    dataset: Dataset | None = None,
    check_window_bars: int = DEFAULT_CHECK_WINDOW_BARS,
    future_bars: int = DEFAULT_FUTURE_BARS,
    decision_date: str | None = None,
) -> ConformanceReport:
    """Run the suite over every component of ``spec`` on the workspace's dataset.

    The panel is read through a ``research_vintage`` view opened for the last
    session the manifest covers, which is the same as-of path an evaluation
    reads; ``check_window_bars`` decision bars sit ``future_bars`` before its
    end, so the sweep has a real future to be contaminated by.
    """
    ds = dataset if dataset is not None else Dataset.open(workspace.data)
    vintage = ds.manifest.vintage
    if vintage is None:
        raise ConformanceError(
            f"the dataset at {ds.root} carries no vintage label, so no research_vintage view can be opened over "
            "it; write one with `prism data manifest --freeze` (§3.2)"
        )
    end = decision_date or _last_session(ds)
    view = ds.view("research_vintage", end, vintage=vintage)
    sids = _universe(spec, view)
    close = view.panel(sids, end, "close")
    volume = view.panel(sids, end, "volume")

    reports = []
    for loaded in spec.components:
        reports.append(
            check_component(
                loaded.component,
                close=close,
                volume=volume,
                component_id=loaded.component_id,
                factory=lambda ls=loaded: instantiate_component(ls.spec, workspace=workspace).component,
                check_window_bars=check_window_bars,
                future_bars=future_bars,
            )
        )
    dataset_block = {
        "dataset": str(ds.root),
        "manifest": ds.manifest.id,
        "vintage": vintage,
        "decision date": end,
        "universe": f"{len(sids)} sid(s), {len(close)} bar(s)",
        "check window": (
            f"{check_window_bars} decision bar(s) from {close.index[-check_window_bars - future_bars]}, "
            f"fit cutoff {close.index[-check_window_bars - future_bars - 1]}, "
            f"{future_bars} bar(s) withheld as the future"
        ),
    }
    return ConformanceReport(spec_path=spec.path, dataset=dataset_block, components=tuple(reports))


def _last_session(dataset: Dataset) -> str:
    sessions = dataset.manifest.coverage.get("sessions") if dataset.manifest.coverage else None
    if not sessions:
        raise ConformanceError(
            f"manifest {dataset.manifest.id} records no coverage.sessions, so the suite cannot tell which "
            "decision date the dataset ends at; pass decision_date explicitly"
        )
    return str(sessions[-1])


def _universe(spec: Specification, view: Any) -> list[str]:
    if spec.strategy.universe_index is not None:
        sids = sorted(view.membership(spec.strategy.universe_index))
        if not sids:
            raise ConformanceError(
                f"index {spec.strategy.universe_index!r} has no members at {view.decision_date}; conformance is "
                "checked on the bars the strategy would actually trade"
            )
        return sids
    listed = list(spec.strategy.universe_list or ())
    if not listed:
        raise ConformanceError(f"{spec.path} declares neither a universe index nor a universe list")
    return listed
