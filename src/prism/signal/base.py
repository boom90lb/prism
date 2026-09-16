"""Signal contract (SPEC.md §7.1; engine specification §3.3) — the pluggable
component.

A Signal consumes wide panels (dates × symbols, tz-aware) and emits one value
per eligible name at the decision bar. What that value *means* is declared, not
inferred: ``output_kind`` names the emission (a rank score, a standardized
forecast, or a target weight vector) and ``units`` names what one unit of it
is. The historical default — an expected return over ``horizon_bars`` in
horizon-sigma units, ``E[r_h] / (sigma_daily * sqrt(h))`` (I-3) — is the
``standardized_forecast`` / ``sigma_h`` pair; the momentum and trend books
declare ``rank_score`` / ``rank`` because their construction rules consume only
cross-sectional order. Scores are *not* positions: no clipping, no conviction
shaping, no cash allocation — sizing folds into the target weight exactly once,
downstream in construction (I-4).

Contract obligations every implementation upholds:

* **Causal (N1).** ``score`` sees a panel whose last row is the decision bar
  *t* and may use nothing after it. Fitted state comes from ``fit`` on a
  train panel only; no transform is fit across the train/score boundary (I-2).
* **NaN = no opinion.** A name the signal cannot score (insufficient history,
  ineligible, degenerate estimate) gets NaN — never a fabricated zero. The
  returned Series is indexed by every panel column, so downstream can tell
  "no opinion" from "absent".
* **Fail loud (N7).** Configuration errors, unconstructable members, and
  unexpected fit/score failures raise. Silent degradation to an empty Series
  or all-zeros is a defect, not a fallback.
* **Declared.** Every field of ``DECLARED_FIELDS`` is a class attribute or a
  property, never taken from configuration, and ``check_declarations`` refuses
  a component that omits one. The engine reads behaviour off the declaration
  (which close panel to supply, whether the rescale property holds, whether a
  fit artifact exists), so an undeclared field is an unanswerable question, not
  a default.
* **Horizon-tagged.** ``horizon_bars`` states the forward horizon the value is
  about; ``valid_for_bars`` states when it stops driving a target and
  ``estimated_decay_bars`` is recorded metadata only. None of the three ever
  scales an order.
* **JAX/torch-free (N8).** ``prism.signal`` is a production module: its import
  closure must stay research-free (enforced by ``tests/test_import_hygiene.py``).

Portfolio knowledge is explicitly out of contract: a node never sees other
names' weights, never nets, never caps (SPEC §7.1 "never").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import pandas as pd

OutputKind = Literal["rank_score", "standardized_forecast", "target_weights"]
Units = Literal["rank", "sigma_h", "weight"]
ScoringBasis = Literal["price", "total_return"]
FittedState = Literal["none", "train_only"]

#: One unit of each emission, per §3.3. The mapping is 1:1: a rank score in
#: weight units, or a target weight in rank units, has no meaning the engine
#: could act on, so the pair is checked rather than assumed.
UNITS_FOR_KIND: dict[str, str] = {
    "rank_score": "rank",
    "standardized_forecast": "sigma_h",
    "target_weights": "weight",
}
OUTPUT_KINDS: tuple[str, ...] = tuple(UNITS_FOR_KIND)
UNITS: tuple[str, ...] = tuple(UNITS_FOR_KIND.values())
SCORING_BASES: tuple[str, ...] = ("price", "total_return")
FITTED_STATES: tuple[str, ...] = ("none", "train_only")

#: The fields §3.3 requires a component to declare, in specification order.
#: ``check_declarations`` reads exactly these.
DECLARED_FIELDS: tuple[str, ...] = (
    "component_id",
    "output_kind",
    "units",
    "scoring_basis",
    "price_scale_invariant",
    "fitted_state",
    "estimated_decay_bars",
    "horizon_bars",
    "update_every_bars",
    "valid_for_bars",
    "refit_every_bars",
    "training_window_bars",
    "required_history",
)


class DeclarationError(ValueError):
    """A component's declarations are missing, mistyped, or inconsistent."""


class Signal(ABC):
    """Abstract cross-sectional component.

    ``fit(close, volume)`` fits on the training panel and returns ``self``;
    ``score(close, volume)`` scores the *last row* of the panel it is given
    and returns one value per column (NaN = no opinion).

    The declarations below are the component's half of the contract. The four
    that describe the emission (``output_kind``, ``units``, ``scoring_basis``,
    ``price_scale_invariant``), the two that describe fitted state
    (``fitted_state``, ``estimated_decay_bars``), and the two abstract
    properties (``horizon_bars``, ``required_history``) carry no default: a
    subclass states them or ``check_declarations`` refuses it, because every
    one of them changes what the engine does. The schedule fields carry the
    defaults §3.3 and §3.8 name — score at every decision, hold an output for
    one bar, refit at every decision, train on every bar the view exposes —
    which is what the live books do today (Parity).
    """

    # Stable id. The configuration table's `id` is authoritative and the loader
    # assigns it after construction; "" means the class declares none, which is
    # the normal case. A class-declared id that differs from the configured one
    # is a conformance failure, not a silent override (§3.3).
    component_id: str = ""

    output_kind: OutputKind
    units: Units
    scoring_basis: ScoringBasis
    price_scale_invariant: bool
    fitted_state: FittedState
    estimated_decay_bars: int | None

    # Schedule. The loader sets `update_every_bars` and `valid_for_bars` from
    # the component table; a class that also declares them must agree with it.
    update_every_bars: int = 1
    valid_for_bars: int = 1
    refit_every_bars: int = 1
    training_window_bars: int = 0

    @property
    @abstractmethod
    def horizon_bars(self) -> int:
        """Forward horizon (in bars) the emitted values are about."""

    @property
    @abstractmethod
    def required_history(self) -> int:
        """Minimum trailing panel rows ``score`` needs to score the last bar."""

    @abstractmethod
    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "Signal":
        """Fit node state on the training panel only (I-2). Returns self."""

    @abstractmethod
    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        """Emitted values for the panel's last row.

        The panel's last row is the decision bar *t*; only rows ≤ *t* may be
        consumed (N1). Returns a float Series indexed by ``close.columns`` in
        the component's declared ``units``, NaN where the component has no
        opinion.
        """

    def instructions(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series | None:
        """Per-sid marks in ``{"liquidate", "no_decision"}``; None when there are none.

        An absent index entry is no instruction. The default is None: a
        component that never asks for an explicit exit or an explicit
        abstention says so by not overriding this.
        """
        return None

    def save_state(self, path: Path) -> None:
        """Write the fitted state to ``path`` (required when ``fitted_state != "none"``).

        A component declaring itself stateless implements neither this nor
        ``load_state``, and its ``fitted_state_ref`` is null (§3.12); the
        default therefore refuses rather than writing an empty artifact that a
        later load would silently accept.
        """
        raise NotImplementedError(_no_state_message(self, "save_state"))

    def load_state(self, path: Path) -> "Signal":
        """Restore the fitted state written by ``save_state`` on the same engine version."""
        raise NotImplementedError(_no_state_message(self, "load_state"))


def _no_state_message(component: Signal, method: str) -> str:
    declared = getattr(component, "fitted_state", None)
    name = type(component).__name__
    if declared == "none":
        return (
            f"{name} declares fitted_state='none' and has no fitted state to persist; "
            f"{method} is required only of a component declaring 'train_only'"
        )
    return (
        f"{name} declares fitted_state={declared!r} and must implement {method}: "
        "fitted state is persisted as a real artifact, never as a dangling reference"
    )


def check_declarations(component: Signal, *, configured_id: str | None = None) -> dict[str, object]:
    """Read every field of ``DECLARED_FIELDS`` off ``component``, or refuse it.

    Returns the declarations as a plain dict, which is what the conformance
    report and the ``ResultRecord`` component block record. Raises
    ``DeclarationError`` naming the offending field when one is missing, of the
    wrong type, outside its permitted values, or inconsistent with another
    (``units`` must match ``output_kind``; an output may not expire before its
    own next refresh). ``configured_id``, when given, is the configuration
    table's ``id``: a class-declared ``component_id`` that differs from it is a
    conformance failure, not a silent override.
    """
    name = type(component).__name__
    values: dict[str, object] = {}
    for field in DECLARED_FIELDS:
        try:
            values[field] = getattr(component, field)
        except AttributeError as exc:
            raise DeclarationError(
                f"{name} declares no {field!r}; the component contract (§3.3) requires "
                f"every component to declare {field!r} as a class attribute or a property"
            ) from exc

    _require_str(name, "component_id", values["component_id"])
    _require_choice(name, "output_kind", values["output_kind"], OUTPUT_KINDS)
    _require_choice(name, "units", values["units"], UNITS)
    _require_choice(name, "scoring_basis", values["scoring_basis"], SCORING_BASES)
    _require_choice(name, "fitted_state", values["fitted_state"], FITTED_STATES)
    if not isinstance(values["price_scale_invariant"], bool):
        raise DeclarationError(
            f"{name} declares price_scale_invariant={values['price_scale_invariant']!r}; "
            "it must be a bool (the conformance rescale check runs exactly when it is True)"
        )
    _require_int(name, "horizon_bars", values["horizon_bars"], minimum=1)
    _require_int(name, "refit_every_bars", values["refit_every_bars"], minimum=1)
    update_every = _require_int(name, "update_every_bars", values["update_every_bars"], minimum=1)
    valid_for = _require_int(name, "valid_for_bars", values["valid_for_bars"], minimum=1)
    _require_int(name, "training_window_bars", values["training_window_bars"], minimum=0)
    _require_int(name, "required_history", values["required_history"], minimum=1)
    if values["estimated_decay_bars"] is not None:
        _require_int(name, "estimated_decay_bars", values["estimated_decay_bars"], minimum=1)

    expected_units = UNITS_FOR_KIND[str(values["output_kind"])]
    if values["units"] != expected_units:
        raise DeclarationError(
            f"{name} declares units={values['units']!r} with output_kind="
            f"{values['output_kind']!r}, which is measured in {expected_units!r}"
        )
    if valid_for < update_every:
        raise DeclarationError(
            f"{name} declares valid_for_bars={valid_for} below update_every_bars={update_every}; "
            "an output that expires before its own next refresh has no reachable valid state"
        )
    if configured_id is not None and values["component_id"] not in ("", configured_id):
        raise DeclarationError(
            f"{name} declares component_id={values['component_id']!r}, which differs from the "
            f"configured id {configured_id!r}; the configuration id is authoritative and the "
            "class declaration is not silently overridden"
        )
    return values


def _require_str(cls_name: str, field: str, value: object) -> None:
    if not isinstance(value, str):
        raise DeclarationError(f"{cls_name} declares {field}={value!r}; it must be a str, got {type(value).__name__}")


def _require_choice(cls_name: str, field: str, value: object, permitted: tuple[str, ...]) -> None:
    if value not in permitted:
        allowed = ", ".join(repr(p) for p in permitted)
        raise DeclarationError(f"{cls_name} declares {field}={value!r}; permitted values are {allowed}")


def _require_int(cls_name: str, field: str, value: object, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DeclarationError(f"{cls_name} declares {field}={value!r}; it must be an int, got {type(value).__name__}")
    if value < minimum:
        raise DeclarationError(f"{cls_name} declares {field}={value!r}; it must be >= {minimum}")
    return value
