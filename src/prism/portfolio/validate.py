"""Independent validation of a solved target and of a rounded order batch (spec §3.7).

**Solver status never authorizes an order.** Nothing in this module reads a solver's
``status``, ``iters``, or ``tol``; :func:`validate_batch` takes no solver argument at all.
A program that reports ``optimal`` and a program that reports nothing are validated
identically, from the numbers they produced.

Validation runs twice -- after the solve and again after rounding -- and checks
finiteness of every output, every limit to its tolerance, cash, and the cost expression
against the cost model. It partitions violations exactly as ``ConstructionOutput.violations``
does, and its accept condition is stated rather than implied, because a pass/fail over
every limit would reject every recovery output that has not fully cured an inherited
breach -- the normal case, and one neither repair can cure (scaling moves the result back
toward the breaching held book, dropping leaves the breach untouched):

* any **introduced** violation of a ``PHYSICAL`` or ``MANDATORY_ACTION`` restriction fails,
  always, on every path;
* any violation of a bound R1 achieved, beyond ``bound_slack``, fails;
* an introduced ``PREFERENCE`` violation fails when ``preference_response = "block"`` and
  is recorded when it is ``"record"``;
* an **inherited** violation, present in the marked state and not made worse, never fails;
  it is reported as ``remaining`` with its measure.

``mode = "report_only"`` exists for exactly one use, the stage 5 replay parity gate: every
check still runs and every finding is still recorded, and only the *verdict* softens.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.portfolio.limits import (
    ACTION_RESTRICTION_CLASSES,
    Limit,
    LimitClass,
    LimitContext,
    LimitError,
    binding_condition,
    partition_violations,
)

__all__ = [
    "ValidationError",
    "ValidationSettings",
    "Finding",
    "ValidationReport",
    "capability_row",
    "quantity_increment_for",
    "validate_targets",
    "validate_batch",
]


class ValidationError(ValueError):
    """An input validation itself cannot proceed on -- never a finding about a batch."""


_MODES = ("blocking", "report_only")
_PREFERENCE_RESPONSES = ("record", "block")


@dataclass(frozen=True)
class ValidationSettings:
    """The declared ``[validation]`` block, plus the two recovery-policy values it reads.

    Tolerances are declared, not implied: ``limit_tolerance`` is in weight units
    (fractions of marked equity), ``cash_tolerance`` in dollars, and
    ``cost_tolerance_bps`` in basis points of traded notional.
    """

    limit_tolerance: float = 1.0e-6
    cash_tolerance: float = 0.01
    cost_tolerance_bps: float = 0.1
    mode: Literal["blocking", "report_only"] = "blocking"
    preference_response: Literal["record", "block"] = "record"
    bound_slack: float = 1.0e-6
    #: When true, a sid without a dated ``capabilities`` row is a failure rather than an
    #: unchecked note. False during construction, true at submission revalidation (§3.9).
    require_capabilities: bool = False

    def __post_init__(self) -> None:
        for name in ("limit_tolerance", "cash_tolerance", "cost_tolerance_bps", "bound_slack"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValidationError(f"{name} must be finite and >= 0, got {getattr(self, name)!r}")
            object.__setattr__(self, name, value)
        if self.mode not in _MODES:
            raise ValidationError(f"validation mode must be one of {_MODES}, got {self.mode!r}")
        if self.preference_response not in _PREFERENCE_RESPONSES:
            raise ValidationError(
                f"preference_response must be one of {_PREFERENCE_RESPONSES}, got {self.preference_response!r}"
            )

    @classmethod
    def from_profile(
        cls,
        execution_profile: Any,
        *,
        recovery_policy: Mapping[str, Any] | None = None,
        require_capabilities: bool = False,
    ) -> "ValidationSettings":
        """Read the ``[validation]`` block off an execution profile (object or mapping).

        Accepts either the parsed ``ExecutionProfile`` of §3.9 (an attribute per block) or
        the plain mapping ``tomllib`` produces, so that a caller holding the file's
        contents needs no adapter. Unknown keys in the block are refused: a mis-spelled
        tolerance that silently kept the default is exactly the defect this module exists
        to prevent.
        """
        block = _block(execution_profile, "validation") or {}
        unknown = set(block) - {"limit_tolerance", "cash_tolerance", "cost_tolerance_bps", "mode"}
        if unknown:
            raise ValidationError(f"unknown keys in the [validation] block: {sorted(unknown)}")
        policy = dict(recovery_policy or {})
        return cls(
            limit_tolerance=float(block.get("limit_tolerance", 1.0e-6)),
            cash_tolerance=float(block.get("cash_tolerance", 0.01)),
            cost_tolerance_bps=float(block.get("cost_tolerance_bps", 0.1)),
            mode=str(block.get("mode", "blocking")),  # type: ignore[arg-type]
            preference_response=str(policy.get("preference_response", "record")),  # type: ignore[arg-type]
            bound_slack=float(policy.get("bound_slack", 1.0e-6)),
            require_capabilities=require_capabilities,
        )


def _block(profile: Any, name: str) -> Mapping[str, Any] | None:
    """One block of an execution profile, whether it is an object or a mapping."""
    if profile is None:
        return None
    if isinstance(profile, Mapping):
        value = profile.get(name)
    else:
        value = getattr(profile, name, None)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return {f.name: getattr(value, f.name) for f in getattr(value, "__dataclass_fields__", {}).values()}


def _setting(profile: Any, name: str, default: Any) -> Any:
    """One top-level key of an execution profile, whether it is an object or a mapping."""
    if profile is None:
        return default
    if isinstance(profile, Mapping):
        return profile.get(name, default)
    return getattr(profile, name, default)


@dataclass(frozen=True)
class Finding:
    """One validation finding. ``fatal`` is the verdict this finding carries by itself."""

    check: str
    detail: str
    fatal: bool
    sid: str | None = None
    limit_name: str | None = None
    limit_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The record shape."""
        return {
            "check": self.check,
            "detail": self.detail,
            "fatal": self.fatal,
            "sid": self.sid,
            "limit_name": self.limit_name,
            "limit_class": self.limit_class,
        }


@dataclass(frozen=True)
class ValidationReport:
    """The partitioned report §3.7 names, from one validation pass."""

    stage: Literal["post_solve", "post_rounding"]
    mode: Literal["blocking", "report_only"]
    violations: Mapping[str, Sequence[Mapping[str, Any]]]
    findings: tuple[Finding, ...] = ()
    unchecked: tuple[str, ...] = ()
    cash: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, Any] = field(default_factory=dict)
    binding_condition: str | None = None

    @property
    def fatal_findings(self) -> tuple[Finding, ...]:
        """The findings that fail the batch under this report's accept condition."""
        return tuple(finding for finding in self.findings if finding.fatal)

    @property
    def ok(self) -> bool:
        """The verdict. ``report_only`` accepts every batch and still records every finding."""
        if self.mode == "report_only":
            return True
        return not self.fatal_findings

    @property
    def blocked(self) -> bool:
        """True when a blocking-mode validation refuses the batch."""
        return not self.ok

    def to_dict(self) -> dict[str, Any]:
        """The JSON-safe record."""
        return {
            "stage": self.stage,
            "mode": self.mode,
            "ok": self.ok,
            "violations": {key: [dict(entry) for entry in value] for key, value in self.violations.items()},
            "findings": [finding.to_dict() for finding in self.findings],
            "unchecked": list(self.unchecked),
            "cash": dict(self.cash),
            "cost": dict(self.cost),
            "binding_condition": self.binding_condition,
        }


# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------

_LIST_COLUMNS = ("restrictions", "order_types", "sessions")


def capability_row(capabilities: Any, sid: str) -> dict[str, Any] | None:
    """The dated ``capabilities`` row for one sid, with its JSON list columns parsed.

    Returns ``None`` when no row exists. The caller decides whether that is a note or a
    failure (:attr:`ValidationSettings.require_capabilities`); this function never invents
    a permissive default.
    """
    if capabilities is None:
        return None
    if isinstance(capabilities, pd.DataFrame):
        if sid not in capabilities.index:
            return None
        raw = capabilities.loc[sid]
        row = dict(raw.iloc[0]) if isinstance(raw, pd.DataFrame) else dict(raw)
    elif isinstance(capabilities, Mapping):
        value = capabilities.get(sid)
        if value is None:
            return None
        row = dict(value)
    else:  # pragma: no cover - defensive
        raise ValidationError(f"capabilities must be a DataFrame or a mapping, got {type(capabilities).__name__}")
    for column in _LIST_COLUMNS:
        row[column] = _as_list(row.get(column))
    return row


def _as_list(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("["):
            try:
                return tuple(str(item) for item in json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"capability list column is not JSON: {value!r} ({exc})") from exc
        return (text,)
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
        return tuple(str(item) for item in value)
    return (str(value),)


def quantity_increment_for(
    sid: str,
    *,
    execution_profile: Any,
    capabilities: Any = None,
) -> float:
    """The per-sid quantity increment, derived rather than assumed (§3.7).

    It is ``quantity_increment_default`` from the execution profile, unless the sid's
    ``capabilities`` row has ``fractionable = true`` **and** the profile's
    ``[fractional] allowed = true``, in which case it is the profile's
    ``[rounding] fractional_increment``. ``lot_size`` is a round-lot classification and is
    never the increment; auction eligibility is the presence of ``market_on_open`` in the
    sid's ``order_types`` and is never assumed to produce a fill.
    """
    default = float(_setting(execution_profile, "quantity_increment_default", 1.0))
    if not math.isfinite(default) or default < 0.0:
        raise ValidationError(f"quantity_increment_default must be finite and >= 0, got {default!r}")
    fractional = _block(execution_profile, "fractional") or {}
    if not bool(fractional.get("allowed", False)):
        return default
    row = capability_row(capabilities, sid)
    if row is None or not bool(row.get("fractionable", False)):
        return default
    rounding = _block(execution_profile, "rounding") or {}
    increment = float(rounding.get("fractional_increment", default))
    if not math.isfinite(increment) or increment <= 0.0:
        raise ValidationError(f"[rounding] fractional_increment must be finite and > 0, got {increment!r}")
    return increment


# --------------------------------------------------------------------------
# Post-solve validation
# --------------------------------------------------------------------------


def validate_targets(
    w_target: pd.Series,
    limits: Iterable[Limit],
    ctx: LimitContext,
    *,
    settings: ValidationSettings,
    achieved_bounds: Mapping[str, float] | None = None,
    attributed_to: str = "R2",
) -> ValidationReport:
    """Validate a solved target, before any rounding.

    ``achieved_bounds`` maps a limit name to the violation measure R1 achieved for it; a
    target that exceeds one of those by more than ``bound_slack`` fails, which is the
    checkable form of "R2 cannot increase any violation beyond the value R1 achieved".
    """
    limits = tuple(limits)
    findings: list[Finding] = []
    target = pd.Series(w_target, dtype=float)
    values = target.to_numpy(dtype=float)
    non_finite = ~np.isfinite(values)
    if non_finite.any():
        offenders = [str(sid) for sid in target.index[non_finite]][:10]
        findings.append(
            Finding(
                check="finiteness",
                detail=f"target weights are not finite for {offenders}",
                fatal=True,
            )
        )
        return ValidationReport(
            stage="post_solve",
            mode=settings.mode,
            violations={"existing": [], "remaining": [], "introduced": []},
            findings=tuple(findings),
        )
    partition = partition_violations(
        limits,
        target,
        ctx,
        limit_tolerance=settings.limit_tolerance,
        attributed_to=attributed_to,
    )
    findings.extend(_findings_from_partition(partition, settings))
    findings.extend(_findings_from_bounds(partition, achieved_bounds, settings))
    return ValidationReport(
        stage="post_solve",
        mode=settings.mode,
        violations=partition,
        findings=tuple(findings),
        binding_condition=binding_condition(partition),
    )


def _findings_from_partition(
    partition: Mapping[str, Sequence[Mapping[str, Any]]],
    settings: ValidationSettings,
) -> list[Finding]:
    findings: list[Finding] = []
    for entry in partition.get("introduced", ()):
        cls = LimitClass(str(entry["class"]))
        fatal = cls in ACTION_RESTRICTION_CLASSES or settings.preference_response == "block"
        findings.append(
            Finding(
                check="introduced_violation",
                detail=(
                    f"{entry['limit_name']} ({cls.value}) is breached by v={float(entry['v']):.6g} on a book "
                    f"the marked state satisfied, attributed to {entry['attributed_to']}"
                ),
                fatal=fatal,
                limit_name=str(entry["limit_name"]),
                limit_class=cls.value,
            )
        )
    for existing, remaining in zip(partition.get("existing", ()), partition.get("remaining", ())):
        before, after = float(existing["v"]), float(remaining["v"])
        worse = after > before + settings.limit_tolerance
        cls = LimitClass(str(remaining["class"]))
        findings.append(
            Finding(
                check="inherited_violation_worsened" if worse else "inherited_violation",
                detail=(
                    f"{remaining['limit_name']} ({cls.value}) was already breached at v={before:.6g} and the "
                    f"target measures v={after:.6g}"
                ),
                fatal=worse and (cls in ACTION_RESTRICTION_CLASSES or settings.preference_response == "block"),
                limit_name=str(remaining["limit_name"]),
                limit_class=cls.value,
            )
        )
    return findings


def _findings_from_bounds(
    partition: Mapping[str, Sequence[Mapping[str, Any]]],
    achieved_bounds: Mapping[str, float] | None,
    settings: ValidationSettings,
) -> list[Finding]:
    if not achieved_bounds:
        return []
    measured: dict[str, float] = {}
    for key in ("remaining", "introduced"):
        for entry in partition.get(key, ()):
            measured[str(entry["limit_name"])] = float(entry["v"])
    findings: list[Finding] = []
    for name, bound in achieved_bounds.items():
        achieved = float(measured.get(name, 0.0))
        if achieved > float(bound) + settings.bound_slack:
            findings.append(
                Finding(
                    check="achieved_bound",
                    detail=(
                        f"{name} measures v={achieved:.6g} against the bound R1 achieved ({float(bound):.6g}) "
                        f"plus bound_slack {settings.bound_slack:.6g}"
                    ),
                    fatal=True,
                    limit_name=name,
                )
            )
    return findings


# --------------------------------------------------------------------------
# Post-rounding validation
# --------------------------------------------------------------------------


def validate_batch(
    intents: Sequence[Any],
    *,
    limits: Iterable[Limit],
    ctx: LimitContext,
    prices: pd.Series,
    settings: ValidationSettings,
    execution_profile: Any = None,
    capabilities: Any = None,
    cash: float = 0.0,
    committed_cash: float = 0.0,
    positions: Mapping[str, float] | None = None,
    cost_model: Any = None,
    modeled_cost_dollars: float | None = None,
    achieved_bounds: Mapping[str, float] | None = None,
    hedge_sids: Iterable[str] = (),
    exempt_sids: Iterable[str] = (),
    check_zero_crossing: bool = True,
    attributed_to: str = "rounding",
) -> ValidationReport:
    """Validate the order batch **and the portfolio it implies**, after rounding.

    The validated object is the portfolio the batch implies together with everything
    already outstanding, which is why current positions and the cash already committed to
    pending buys are inputs and not derived from the intents.

    No argument of this function is a solver status, and none is consulted: an order is
    authorized by these numbers or it is not authorized.
    """
    limits = tuple(limits)
    prices = pd.Series(prices, dtype=float)
    held = pd.Series(dict(positions or {}), dtype=float) if positions else pd.Series(dtype=float)
    findings: list[Finding] = []
    unchecked: list[str] = []

    findings.extend(_finiteness_findings(intents, prices))
    if any(finding.fatal for finding in findings):
        return ValidationReport(
            stage="post_rounding",
            mode=settings.mode,
            violations={"existing": [], "remaining": [], "introduced": []},
            findings=tuple(findings),
            unchecked=tuple(unchecked),
        )

    order_findings, order_unchecked = _order_findings(
        intents,
        prices=prices,
        held=held,
        execution_profile=execution_profile,
        capabilities=capabilities,
        settings=settings,
        hedge_sids=frozenset(str(sid) for sid in hedge_sids),
        exempt_sids=frozenset(str(sid) for sid in exempt_sids),
        check_zero_crossing=check_zero_crossing,
    )
    findings.extend(order_findings)
    unchecked.extend(order_unchecked)

    implied = _implied_weights(intents, ctx=ctx, prices=prices)
    partition = partition_violations(
        limits, implied, ctx, limit_tolerance=settings.limit_tolerance, attributed_to=attributed_to
    )
    findings.extend(_findings_from_partition(partition, settings))
    findings.extend(_findings_from_bounds(partition, achieved_bounds, settings))

    cash_report, cash_findings = _cash_findings(
        intents,
        cash=cash,
        committed_cash=committed_cash,
        held=held,
        prices=prices,
        settings=settings,
        modeled_cost_dollars=modeled_cost_dollars,
    )
    findings.extend(cash_findings)

    cost_report, cost_findings = _cost_findings(
        intents,
        prices=prices,
        cost_model=cost_model,
        modeled_cost_dollars=modeled_cost_dollars,
        settings=settings,
    )
    findings.extend(cost_findings)

    return ValidationReport(
        stage="post_rounding",
        mode=settings.mode,
        violations=partition,
        findings=tuple(findings),
        unchecked=tuple(unchecked),
        cash=cash_report,
        cost=cost_report,
        binding_condition=_binding(findings, partition),
    )


def _binding(findings: Sequence[Finding], partition: Mapping[str, Sequence[Mapping[str, Any]]]) -> str | None:
    for finding in findings:
        if not finding.fatal:
            continue
        if finding.limit_name:
            return finding.limit_name
        return f"{finding.check}:{finding.sid}" if finding.sid else finding.check
    return binding_condition(partition)


def _finiteness_findings(intents: Sequence[Any], prices: pd.Series) -> list[Finding]:
    findings: list[Finding] = []
    for intent in intents:
        sid = str(getattr(intent, "sid"))
        qty = float(getattr(intent, "qty"))
        reference = float(getattr(intent, "reference_price"))
        if not math.isfinite(qty) or qty == 0.0:
            findings.append(
                Finding(
                    check="finiteness",
                    detail=f"order quantity for {sid} is {qty!r}; a zero or non-finite quantity is not an order",
                    fatal=True,
                    sid=sid,
                )
            )
        if not math.isfinite(reference) or reference <= 0.0:
            findings.append(
                Finding(
                    check="finiteness",
                    detail=f"reference price for {sid} is {reference!r}; a size computed from it is meaningless",
                    fatal=True,
                    sid=sid,
                )
            )
        price = prices.get(sid, float("nan"))
        if price is None or not math.isfinite(float(price)) or float(price) <= 0.0:
            findings.append(
                Finding(
                    check="finiteness",
                    detail=f"no usable price for {sid} ({price!r}); the implied portfolio cannot be marked",
                    fatal=True,
                    sid=sid,
                )
            )
    return findings


def _order_findings(
    intents: Sequence[Any],
    *,
    prices: pd.Series,
    held: pd.Series,
    execution_profile: Any,
    capabilities: Any,
    settings: ValidationSettings,
    hedge_sids: frozenset[str],  # C4/C5 classification is the caller's; validation checks numbers
    exempt_sids: frozenset[str],
    check_zero_crossing: bool,
) -> tuple[list[Finding], list[str]]:
    """Per-order action restrictions: increment, notional, order type, borrow, halts.

    These are ``PHYSICAL`` and ``MANDATORY_ACTION`` restrictions, so every one of them is
    fatal in blocking mode. They are enforced here, not in the adapter's ``submit``, which
    sees no positions and so cannot decide "never short" (§3.9).
    """
    from prism.portfolio.limits import restriction_blocks_all_trading, restriction_blocks_new_exposure

    findings: list[Finding] = []
    unchecked: list[str] = []
    profile_min_notional = float(_setting(execution_profile, "min_notional", 0.0))
    allowed_types = tuple(str(item) for item in _setting(execution_profile, "order_types", ()) or ())
    fractional = _block(execution_profile, "fractional") or {}
    fractional_tif = tuple(str(item) for item in fractional.get("time_in_force", ()) or ())
    seen: set[str] = set()
    for intent in intents:
        sid = str(getattr(intent, "sid"))
        if sid in seen:
            findings.append(
                Finding(
                    check="duplicate_order",
                    detail=f"two orders in one batch for {sid}; a batch carries at most one order per sid",
                    fatal=True,
                    sid=sid,
                )
            )
        seen.add(sid)
        qty = float(getattr(intent, "qty"))
        reference = float(getattr(intent, "reference_price"))
        increment = float(getattr(intent, "increment", 0.0))
        order_type = str(getattr(intent, "order_type", "") or "")
        tif = str(getattr(intent, "time_in_force", "") or "")
        row = capability_row(capabilities, sid)
        if row is None:
            note = f"{sid}: no dated capabilities row; increment, shortability and halt status not evaluated"
            if settings.require_capabilities:
                findings.append(Finding(check="capabilities", detail=note, fatal=True, sid=sid))
            else:
                unchecked.append(note)
        # -- quantity increment
        expected = quantity_increment_for(sid, execution_profile=execution_profile, capabilities=capabilities)
        if increment > 0.0 and abs(increment - expected) > 1e-12:
            findings.append(
                Finding(
                    check="quantity_increment",
                    detail=f"{sid}: order carries increment {increment}, the derived increment is {expected}",
                    fatal=True,
                    sid=sid,
                )
            )
        if expected > 0.0 and not _on_increment(qty, expected):
            findings.append(
                Finding(
                    check="quantity_increment",
                    detail=f"{sid}: quantity {qty} is not a multiple of the increment {expected}",
                    fatal=True,
                    sid=sid,
                )
            )
        # -- minimum notional
        row_min = float(row["min_notional"]) if row is not None and row.get("min_notional") is not None else None
        venue_floor = row_min if row_min is not None else 0.0
        # The profile's min_notional is a *policy* dust floor; the capabilities row's is the
        # venue's own minimum. An obligation leg is exempt from the policy floor -- a recall
        # on two shares is still covered -- and never from the venue's, which no declaration
        # can waive.
        floor = venue_floor if sid in exempt_sids else max(profile_min_notional, venue_floor)
        notional = abs(qty) * reference
        if floor > 0.0 and notional + 1e-12 < floor:
            findings.append(
                Finding(
                    check="min_notional",
                    detail=f"{sid}: order notional {notional:.4f} is below the minimum {floor:.4f}",
                    fatal=True,
                    sid=sid,
                )
            )
        # -- order type and time in force
        if allowed_types and order_type and order_type not in allowed_types:
            findings.append(
                Finding(
                    check="order_type",
                    detail=f"{sid}: order type {order_type!r} is outside the profile's {list(allowed_types)}",
                    fatal=True,
                    sid=sid,
                )
            )
        if row is not None and row["order_types"] and order_type and order_type not in row["order_types"]:
            findings.append(
                Finding(
                    check="order_type",
                    detail=f"{sid}: the capabilities row admits {list(row['order_types'])}, not {order_type!r}",
                    fatal=True,
                    sid=sid,
                )
            )
        # -- fractional eligibility and the fractional-short prohibition
        current = float(held.get(sid, 0.0))
        ends_at = current + qty
        if expected > 0.0 and expected < 1.0:
            if not bool(fractional.get("allowed", False)):
                findings.append(
                    Finding(
                        check="fractional",
                        detail=f"{sid}: a fractional increment under a profile whose [fractional] allowed is false",
                        fatal=True,
                        sid=sid,
                    )
                )
            if fractional_tif and tif and tif not in fractional_tif:
                findings.append(
                    Finding(
                        check="fractional",
                        detail=f"{sid}: fractional orders take {list(fractional_tif)} time in force, not {tif!r}",
                        fatal=True,
                        sid=sid,
                    )
                )
            if not bool(fractional.get("short_allowed", False)) and (qty < 0.0 and ends_at < 0.0):
                findings.append(
                    Finding(
                        check="fractional_short",
                        detail=f"{sid}: a fractional order may never open or extend a short ({current} -> {ends_at})",
                        fatal=True,
                        sid=sid,
                    )
                )
        # -- shortability and borrow
        opens_short = ends_at < 0.0 and (current >= 0.0 or qty < 0.0)
        if row is not None and opens_short:
            if not bool(row.get("shortable", True)):
                findings.append(
                    Finding(
                        check="shortable",
                        detail=f"{sid}: the capabilities row marks the name not shortable, and the batch ends short",
                        fatal=True,
                        sid=sid,
                    )
                )
            elif current >= 0.0 and not bool(row.get("easy_to_borrow", True)):
                findings.append(
                    Finding(
                        check="borrow",
                        detail=f"{sid}: no borrow is available, so no new short may be opened",
                        fatal=True,
                        sid=sid,
                    )
                )
        # -- halts and reduce-only restrictions
        if row is not None:
            restrictions = row["restrictions"]
            if restriction_blocks_all_trading(restrictions):
                findings.append(
                    Finding(
                        check="halted",
                        detail=f"{sid}: the name carries {list(restrictions)} and may not be traded this bar",
                        fatal=True,
                        sid=sid,
                    )
                )
            elif restriction_blocks_new_exposure(restrictions) and not _is_reduction(current, qty):
                findings.append(
                    Finding(
                        check="reduce_only",
                        detail=(
                            f"{sid}: the name is pinned reduce-only by {list(restrictions)}, and the order "
                            f"moves {current} to {ends_at}"
                        ),
                        fatal=True,
                        sid=sid,
                    )
                )
        # -- zero crossing
        if check_zero_crossing and current != 0.0 and ends_at != 0.0 and (current > 0.0) != (ends_at > 0.0):
            findings.append(
                Finding(
                    check="zero_crossing",
                    detail=(
                        f"{sid}: a single order may not cross zero ({current} -> {ends_at}); a crossing is a "
                        "C2 action decomposed into a C1 leg and a C3 leg, each authorized on its own"
                    ),
                    fatal=True,
                    sid=sid,
                )
            )
    return findings, unchecked


def _on_increment(qty: float, increment: float) -> bool:
    if increment <= 0.0:
        return True
    steps = qty / increment
    return abs(steps - round(steps)) <= 1e-6


def _is_reduction(current: float, qty: float) -> bool:
    """True when the order strictly reduces the magnitude of the position without crossing."""
    if current == 0.0 or qty == 0.0:
        return False
    ends_at = current + qty
    if (current > 0.0) == (qty > 0.0):
        return False
    return abs(ends_at) < abs(current) and (ends_at == 0.0 or (ends_at > 0.0) == (current > 0.0))


def _implied_weights(intents: Sequence[Any], *, ctx: LimitContext, prices: pd.Series) -> pd.Series:
    """The weights the batch implies: the marked weights plus each order's weight delta.

    Sized at the *marked* equity of ``ctx``, which is the equity every limit in the §3.7
    table is expressed in fractions of.
    """
    implied = ctx.w0.copy()
    for intent in intents:
        sid = str(getattr(intent, "sid"))
        qty = float(getattr(intent, "qty"))
        price = float(prices.get(sid, float("nan")))
        if not math.isfinite(price):
            raise LimitError(f"no price for {sid}; the implied portfolio cannot be marked")
        delta = qty * price / ctx.equity
        implied[sid] = float(implied.get(sid, 0.0)) + delta
    return implied


def _cash_findings(
    intents: Sequence[Any],
    *,
    cash: float,
    committed_cash: float,
    held: pd.Series,
    prices: pd.Series,
    settings: ValidationSettings,
    modeled_cost_dollars: float | None,
) -> tuple[dict[str, Any], list[Finding]]:
    """Buying power after the batch, with pending buys already deducted.

    ``buying_power = cash - committed_cash - short_collateral``: a pending buy reduces
    available funds before its fill, so a submitted batch is never validated against cash
    it has already promised.
    """
    buy_notional = 0.0
    sell_proceeds = 0.0
    ending: dict[str, float] = {str(sid): float(qty) for sid, qty in held.items()}
    for intent in intents:
        sid = str(getattr(intent, "sid"))
        qty = float(getattr(intent, "qty"))
        reference = float(getattr(intent, "reference_price"))
        if qty > 0.0:
            buy_notional += qty * reference
        else:
            sell_proceeds += -qty * reference
        ending[sid] = ending.get(sid, 0.0) + qty
    costs = 0.0 if modeled_cost_dollars is None else float(modeled_cost_dollars)
    cash_after = float(cash) + sell_proceeds - buy_notional - costs
    short_collateral = 0.0
    for sid, qty in ending.items():
        if qty < 0.0:
            price = float(prices.get(sid, float("nan")))
            if math.isfinite(price):
                short_collateral += -qty * price
    buying_power = cash_after - float(committed_cash) - short_collateral
    report = {
        "cash_before": float(cash),
        "cash_after": cash_after,
        "buy_notional": buy_notional,
        "sell_proceeds": sell_proceeds,
        "charged_costs": costs,
        "committed_cash": float(committed_cash),
        "short_collateral": short_collateral,
        "buying_power": buying_power,
        "tolerance": settings.cash_tolerance,
    }
    findings: list[Finding] = []
    if buying_power < -settings.cash_tolerance:
        findings.append(
            Finding(
                check="cash",
                detail=(
                    f"the batch implies buying power {buying_power:.2f} (cash {cash_after:.2f} less committed "
                    f"{float(committed_cash):.2f} less short collateral {short_collateral:.2f}), beyond the "
                    f"declared tolerance {settings.cash_tolerance:.2f}"
                ),
                fatal=True,
            )
        )
    return report, findings


def _cost_findings(
    intents: Sequence[Any],
    *,
    prices: pd.Series,
    cost_model: Any,
    modeled_cost_dollars: float | None,
    settings: ValidationSettings,
) -> tuple[dict[str, Any], list[Finding]]:
    """The constructor's cost expression against the cost model, in basis points.

    When the caller declares no cost expression the check is recorded as ``not_supplied``
    and nothing is asserted -- the heuristic constructor of stage 3 evaluates no cost term.
    When a cost expression *is* declared and the model cannot recompute it, the check is
    ``unavailable`` and fails in blocking mode: an unverifiable cost expression is not a
    verified one.
    """
    traded_notional = sum(
        abs(float(getattr(intent, "qty"))) * float(getattr(intent, "reference_price")) for intent in intents
    )
    report: dict[str, Any] = {
        "status": "not_supplied",
        "traded_notional": traded_notional,
        "tolerance_bps": settings.cost_tolerance_bps,
        "modeled_dollars": modeled_cost_dollars,
    }
    if modeled_cost_dollars is None:
        return report, []
    if cost_model is None or not hasattr(cost_model, "expected_for_intents"):
        report["status"] = "unavailable"
        report["detail"] = "the cost model exposes no expected_for_intents to recompute the expression against"
        return report, [
            Finding(check="cost", detail=str(report["detail"]), fatal=True),
        ]
    try:
        breakdown = cost_model.expected_for_intents(list(intents), prices)
    except NotImplementedError as exc:
        report["status"] = "unavailable"
        report["detail"] = f"the cost model cannot recompute the expression: {exc}"
        return report, [Finding(check="cost", detail=str(report["detail"]), fatal=True)]
    recomputed = _cost_total(breakdown)
    report["status"] = "checked"
    report["recomputed_dollars"] = recomputed
    difference = abs(recomputed - float(modeled_cost_dollars))
    report["difference_dollars"] = difference
    bps = 0.0 if traded_notional <= 0.0 else difference / traded_notional * 10_000.0
    report["difference_bps"] = bps
    findings: list[Finding] = []
    if bps > settings.cost_tolerance_bps + 1e-12:
        findings.append(
            Finding(
                check="cost",
                detail=(
                    f"the declared cost expression ({float(modeled_cost_dollars):.4f}) and the cost model's "
                    f"recomputation ({recomputed:.4f}) differ by {bps:.4f} bps of traded notional, beyond the "
                    f"declared {settings.cost_tolerance_bps} bps"
                ),
                fatal=True,
            )
        )
    return report, findings


def _cost_total(breakdown: Any) -> float:
    """The dollar total of a ``CostBreakdown``, whatever shape it arrives in."""
    if isinstance(breakdown, (int, float)):
        return float(breakdown)
    if isinstance(breakdown, Mapping):
        if "total" in breakdown:
            return float(breakdown["total"])
        return float(sum(float(value) for value in breakdown.values()))
    total = getattr(breakdown, "total", None)
    if total is None:
        raise ValidationError(f"cannot read a dollar total off {type(breakdown).__name__}")
    return float(total() if callable(total) else total)
