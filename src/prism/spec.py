"""Strategy specifications: TOML in, a resolved object and a ``config_id`` out (spec §3.3, §3.12, §4).

A specification is the whole statement of what a strategy *is*: the universe it
trades, the components it runs with their constructor arguments and schedules,
the eligibility screen, the construction path, and the policy files that price
and constrain it. This module turns that file into a resolved object, imports
and instantiates the components it names, and computes the identity every
recorded result is filed under.

Three things are deliberate here.

**Defaults are resolved before hashing.** A field left implicit in the file —
an eligibility floor, a rank-to-weight cap — is written into the resolved
specification with the value the engine will actually use, so ``config_id``
covers it. Two files that differ only in what they leave unsaid, and mean the
same thing, hash the same; two runs that behave differently never do.

**Identity is over content, not over paths.** ``config_id`` hashes the resolved
specification, the *content* of every referenced policy file, the
implementation identity of every component (the source that computes, its
distribution and version, its repository commit), and the engine version
(§3.12). Renaming a directory does not mint a new identity; editing a cost
stack does. The top-level TOML bytes are never hashed: a comment is not a
configuration change, and a formatting pass must not revoke an authorization.

**Component code is imported, not described.** ``workspace.root`` goes on the
front of ``sys.path`` before a component is imported, which is what makes a
private project's ``strategies`` package importable from an installed console
script whose own ``sys.path[0]`` is the script directory (§4). A path that
cannot be imported, does not name a ``Signal``, or will not accept its declared
parameters is refused with a message naming the path — never an ImportError
traceback from inside somebody else's module.

The four rank-to-weight rules live here too, because a rule name and its
``to_weights_params`` are configuration: they are declared in the specification,
they enter ``config_id``, and the engine may not run a rule it cannot name.
``decile_long_short`` and ``sign_inverse_vol`` delegate to
``prism.portfolio.construct``, unchanged, so the reference books keep their
recorded weights exactly (Parity, §3.3).
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import logging
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.portfolio.construct import (
    buffered_decile_row,
    cap_book,
    construct_decile_neutral,
    construct_inverse_vol_targets,
)
from prism.signal.base import Signal, check_declarations
from prism.validation.trials import UNAVAILABLE as RECORD_UNAVAILABLE
from prism.validation.trials import current_git_commit, stable_json_dumps
from prism.workspace import Workspace, resolve_workspace

logger = logging.getLogger(__name__)

#: The combination flows of §3.3 and the construction kinds of §3.7. The flow is
#: selected by the pair and by no other key.
COMBINATIONS: tuple[str, ...] = ("position_space", "calibrated_convex")
CONSTRUCTION_KINDS: tuple[str, ...] = ("heuristic", "convex")
KIND_FOR_COMBINATION: dict[str, str] = {"position_space": "heuristic", "calibrated_convex": "convex"}

#: Every key of the ``[policy]`` block. All five are required: a specification
#: that cannot name its monitoring or recovery policy cannot be conformed
#: against one, and a missing policy is a silent default (§4).
POLICY_KEYS: tuple[str, ...] = ("profile", "costs", "execution", "monitoring", "recovery")

#: The rank-to-weight vocabulary of §3.3. Four names, no extension point: a rule
#: the engine cannot name is a rule nothing can reproduce.
TO_WEIGHTS_RULES: tuple[str, ...] = (
    "decile_long_short", "decile_long_short_buffered", "sign_inverse_vol", "top_k_equal_weight", "none",
)

#: The three schedule fields the component table carries (§3.3), all required.
SCHEDULE_FIELDS: tuple[str, ...] = ("horizon_bars", "update_every_bars", "valid_for_bars")

#: ``[strategy.eligibility]`` with the defaults §3.3 declares. Resolved into the
#: specification before hashing so a field left implicit still enters ``config_id``.
ELIGIBILITY_DEFAULTS: dict[str, Any] = {
    "min_price": 0.0,
    "min_median_dollar_volume": 0.0,
    "dollar_volume_window": 20,
    "return_history_bars": 0,
    "listing_seasoning_bars": 0,
    "membership_required": False,
    "exclude_asset_classes": (),
}

#: Per-rule ``to_weights_params``: the required keys and the defaults for the
#: rest. The two caps default to the values ``prism.portfolio.construct`` uses
#: (down-only, never binding at 1.0), and the inverse-vol window defaults to the
#: recorded trend sleeve's 63-bar span annualized by 252.
_REQUIRED = object()
TO_WEIGHTS_PARAMS: dict[str, dict[str, Any]] = {
    "decile_long_short": {"decile": _REQUIRED, "max_gross": 1.0, "max_symbol_abs_weight": 1.0},
    "decile_long_short_buffered": {
        "decile": _REQUIRED,
        "hold_quantile": _REQUIRED,
        "max_gross": 1.0,
        "max_symbol_abs_weight": 1.0,
    },
    "sign_inverse_vol": {
        "vol_ewma_bars": 63,
        "annualization_bars": 252,
        "max_gross": 1.0,
        "max_symbol_abs_weight": 1.0,
    },
    "top_k_equal_weight": {"k": _REQUIRED, "max_gross": 1.0, "max_symbol_abs_weight": 1.0},
    "none": {},
}
_INT_PARAMS: frozenset[str] = frozenset({"k", "vol_ewma_bars", "annualization_bars"})

#: The distribution the engine version is read from (§3.12). ``prism`` is the
#: pre-rename name and is consulted second so the loader works in a checkout
#: whose packaging correction has not landed yet.
ENGINE_DISTRIBUTION = "prism-trading"
_LEGACY_ENGINE_DISTRIBUTION = "prism"

#: What an identity field takes when the fact has no source, per §3.12's rule
#: for records: never a null that reads as a configured absence, never a
#: plausible default. Defined by ``validation/trials.py``, which owns the
#: record schema, and re-exported here so a caller reading a specification does
#: not need the record module; there is one literal, not two.
UNAVAILABLE = RECORD_UNAVAILABLE

_HASH_CHUNK_BYTES = 1 << 20


class SpecError(ValueError):
    """A specification is malformed, incomplete, or names something unloadable.

    Raised rather than defaulted. Every field of a specification changes what
    the engine does, so an unreadable one is refused at load rather than
    silently reduced to whatever the loader could parse.
    """


# --------------------------------------------------------------------------
# The resolved specification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Eligibility:
    """The strategy-level screen of §3.3, with every default made explicit."""

    min_price: float = 0.0
    min_median_dollar_volume: float = 0.0
    dollar_volume_window: int = 20
    return_history_bars: int = 0
    listing_seasoning_bars: int = 0
    membership_required: bool = False
    exclude_asset_classes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """The screen as plain JSON values, in specification order."""
        return {
            "min_price": self.min_price,
            "min_median_dollar_volume": self.min_median_dollar_volume,
            "dollar_volume_window": self.dollar_volume_window,
            "return_history_bars": self.return_history_bars,
            "listing_seasoning_bars": self.listing_seasoning_bars,
            "membership_required": self.membership_required,
            "exclude_asset_classes": list(self.exclude_asset_classes),
        }


@dataclass(frozen=True)
class ComponentSpec:
    """One ``[[strategy.components]]`` table, resolved.

    ``params`` is forwarded to ``__init__`` verbatim and carries constructor
    arguments only; the declarations (``output_kind``, ``units``,
    ``scoring_basis``, ...) are class attributes and are never taken from
    configuration (§3.3).
    """

    id: str
    path: str
    weight: float
    to_weights: str
    horizon_bars: int
    update_every_bars: int
    valid_for_bars: int
    params: Mapping[str, Any]
    to_weights_params: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """The component table as plain JSON values, defaults resolved."""
        return {
            "id": self.id,
            "path": self.path,
            "weight": self.weight,
            "to_weights": self.to_weights,
            "horizon_bars": self.horizon_bars,
            "update_every_bars": self.update_every_bars,
            "valid_for_bars": self.valid_for_bars,
            "params": dict(self.params),
            "to_weights_params": dict(self.to_weights_params),
        }


@dataclass(frozen=True)
class StrategySpec:
    """The ``[strategy]`` block: universe, combination flow, and components."""

    name: str
    combination: str
    renormalize: bool
    components: tuple[ComponentSpec, ...]
    eligibility: Eligibility
    universe_index: str | None = None
    universe_list: tuple[str, ...] | None = None
    universe_as_of: str | None = None
    universe_registration: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The strategy block as plain JSON values, defaults resolved."""
        payload: dict[str, Any] = {
            "name": self.name,
            "combination": self.combination,
            "renormalize": self.renormalize,
            "components": [component.to_dict() for component in self.components],
            "eligibility": self.eligibility.to_dict(),
        }
        if self.universe_index is not None:
            payload["universe_index"] = self.universe_index
        else:
            payload["universe_list"] = list(self.universe_list or ())
            payload["universe_as_of"] = self.universe_as_of
            payload["universe_registration"] = self.universe_registration
        return payload


@dataclass(frozen=True)
class ConstructionSpec:
    """The ``[construction]`` block.

    ``calibration`` and ``risk_model`` are resolved file paths; their *content*
    enters ``config_id`` and their paths do not (§3.12). ``options`` carries
    every other key of the block verbatim — ``step_order``, the band table,
    ``risk_aversion``, ``eps`` — because §3.7 owns their meaning and the loader
    only has to preserve them into the identity.
    """

    kind: str
    calibration: Path | None
    risk_model: Path | None
    options: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """The construction block as plain JSON values, file paths excluded."""
        return {"kind": self.kind, **{key: _jsonable(value) for key, value in self.options.items()}}


@dataclass(frozen=True)
class PolicyRefs:
    """The five resolved ``[policy]`` file paths."""

    profile: Path
    costs: Path
    execution: Path
    monitoring: Path
    recovery: Path

    def paths(self) -> dict[str, Path]:
        """The five files by policy key, in ``POLICY_KEYS`` order."""
        return {key: getattr(self, key) for key in POLICY_KEYS}


@dataclass(frozen=True)
class LoadedComponent:
    """An instantiated component, its configuration table, and its identity."""

    spec: ComponentSpec
    component: Signal
    declarations: Mapping[str, Any]
    identity: Mapping[str, Any]

    @property
    def component_id(self) -> str:
        """The configured id, which is authoritative over any class declaration."""
        return self.spec.id


@dataclass(frozen=True)
class Specification:
    """A loaded strategy specification and everything its identity is computed from."""

    path: Path
    root: Path
    strategy: StrategySpec
    construction: ConstructionSpec
    policy: PolicyRefs
    components: tuple[LoadedComponent, ...]
    instantiated: bool

    def resolved(self) -> dict[str, Any]:
        """The specification after defaults, as the plain dict ``config_id`` hashes.

        Policy paths, the calibration path, and the risk-model path are absent
        by construction: those files enter the identity as content (§3.12).
        """
        return {"strategy": self.strategy.to_dict(), "construction": self.construction.to_dict()}

    def identity_payload(
        self,
        *,
        evaluation_context: Mapping[str, Any] | None = None,
        engine_version: str | None = None,
    ) -> dict[str, Any]:
        """The ``{common, evaluation_context}`` payload of §3.12, ready to canonicalize.

        ``evaluation_context`` is the evaluation's half of the identity and is
        ``None`` for an operating session, where capital is an observation
        rather than a setting.
        """
        if not self.instantiated:
            raise SpecError(
                f"{self.path} was loaded with instantiate=False; config_id covers component "
                "implementation identity (source hash, distribution, version), which is only "
                "available from the imported component"
            )
        common = {
            "spec_resolved": self.resolved(),
            "policies": {key: _file_content_digest(path) for key, path in self.policy.paths().items()},
            "construction": {
                "kind": self.construction.kind,
                "calibration_content": _optional_content_digest(self.construction.calibration),
                "risk_model_content": _optional_content_digest(self.construction.risk_model),
            },
            "components": [dict(loaded.identity) for loaded in self.components],
            "engine_version": engine_version if engine_version is not None else _installed_engine_version(),
        }
        return {"common": common, "evaluation_context": _jsonable(evaluation_context)}

    def config_id(
        self,
        *,
        evaluation_context: Mapping[str, Any] | None = None,
        engine_version: str | None = None,
    ) -> str:
        """``sha256(stable_json_dumps({common, evaluation_context}))`` (§3.12)."""
        payload = self.identity_payload(evaluation_context=evaluation_context, engine_version=engine_version)
        return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()

    def component(self, component_id: str) -> LoadedComponent:
        """The loaded component with this configured id."""
        for loaded in self.components:
            if loaded.component_id == component_id:
                return loaded
        known = ", ".join(repr(loaded.component_id) for loaded in self.components) or "none"
        raise SpecError(f"{self.path} declares no component with id {component_id!r}; declared ids: {known}")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_spec(
    path: Path | str,
    *,
    workspace: Workspace | None = None,
    instantiate: bool = True,
) -> Specification:
    """Load, resolve, and (by default) instantiate the specification at ``path``.

    Relative policy, calibration, and risk-model paths resolve against
    ``workspace.root`` — the project directory ``prism init`` writes
    ``profiles/``, ``costs/``, and ``policies/`` into — not against the
    specification's own directory, so ``specs/book.toml`` naming
    ``profiles/paper.toml`` resolves the way §4 shows it.

    ``instantiate=False`` parses and resolves without importing any component
    code, which is what a template check wants; the resulting specification
    refuses ``config_id``, since implementation identity is only readable off
    the imported component.
    """
    spec_path = Path(path).expanduser()
    if not spec_path.is_file():
        raise SpecError(f"no specification at {spec_path}")
    ws = workspace if workspace is not None else resolve_workspace()
    try:
        table = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise SpecError(f"{spec_path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise SpecError(f"{spec_path} cannot be read: {exc}") from exc
    return load_spec_table(table, path=spec_path, workspace=ws, instantiate=instantiate)


def load_spec_table(
    table: Mapping[str, Any],
    *,
    path: Path | str,
    workspace: Workspace | None = None,
    instantiate: bool = True,
) -> Specification:
    """Resolve an already-parsed specification table, as :func:`load_spec` does its file.

    ``path`` is provenance: it names the file the table came from in every error
    message and on the returned :class:`Specification`, and it never enters
    ``config_id`` (§3.12 hashes content, not paths). It does not have to exist,
    which is what lets a registration resolve a declared arm's configuration —
    a control specification plus the arm's declared overrides — without writing
    a specification file nobody would ever read again (§3.13).
    """
    ws = workspace if workspace is not None else resolve_workspace()
    spec_path = Path(path).expanduser()

    _reject_unknown(table, ("strategy", "construction", "policy"), where=str(spec_path))
    strategy = _parse_strategy(_require_table(table, "strategy", where=str(spec_path)), where=str(spec_path))
    construction = _parse_construction(
        _require_table(table, "construction", where=str(spec_path)), root=ws.root, where=str(spec_path)
    )
    policy = _parse_policy(_require_table(table, "policy", where=str(spec_path)), root=ws.root, where=str(spec_path))

    expected_kind = KIND_FOR_COMBINATION[strategy.combination]
    if construction.kind != expected_kind:
        raise SpecError(
            f"{spec_path}: [strategy] combination={strategy.combination!r} pairs with "
            f"[construction] kind={expected_kind!r}, not {construction.kind!r}; the flow is "
            "selected by that pair and by no other key (§3.3)"
        )

    components: tuple[LoadedComponent, ...] = ()
    if instantiate:
        components = tuple(
            instantiate_component(component, workspace=ws, where=str(spec_path)) for component in strategy.components
        )
        if strategy.combination == "position_space":
            for loaded in components:
                if loaded.declarations["output_kind"] == "rank_score" and loaded.spec.to_weights == "none":
                    raise SpecError(
                        f"{spec_path}: component {loaded.component_id!r} declares "
                        "output_kind='rank_score' with to_weights='none' under "
                        "combination='position_space'; a rank score has no weight meaning until a "
                        "rank-to-weight rule runs over it (§3.3)"
                    )
    return Specification(
        path=spec_path,
        root=ws.root,
        strategy=strategy,
        construction=construction,
        policy=policy,
        components=components,
        instantiated=instantiate,
    )


def instantiate_component(
    spec: ComponentSpec,
    *,
    workspace: Workspace,
    where: str | None = None,
) -> LoadedComponent:
    """Import, construct, and check one component; return it with its identity.

    The declared ``path`` is ``module:Class``. ``workspace.root`` is placed at
    the front of ``sys.path`` first (§4). After construction the loader compares
    every schedule field the class also declares with the table value and
    refuses a mismatch, then sets ``update_every_bars`` and ``valid_for_bars``
    on the instance, and finally reads the declarations of §3.3 off it.
    """
    context = f"{where}: " if where else ""
    cls = _import_component(spec.path, workspace=workspace, context=context)
    try:
        component = cls(**dict(spec.params))
    except SpecError:
        raise
    except Exception as exc:
        raise SpecError(
            f"{context}component {spec.id!r} at {spec.path!r} refused its "
            f"[strategy.components.params] {dict(spec.params)!r}: {type(exc).__name__}: {exc}"
        ) from None

    _check_schedule(spec, component, context=context)
    component.update_every_bars = spec.update_every_bars
    component.valid_for_bars = spec.valid_for_bars
    declarations = check_declarations(component, configured_id=spec.id)
    component.component_id = spec.id

    if spec.to_weights != "none" and declarations["output_kind"] == "target_weights":
        raise SpecError(
            f"{context}component {spec.id!r} declares output_kind='target_weights' with "
            f"to_weights={spec.to_weights!r}; a target-weight vector is already account-relative "
            "and takes to_weights='none' (§3.3)"
        )
    return LoadedComponent(
        spec=spec,
        component=component,
        declarations=dict(declarations),
        identity=_component_identity(spec, component, declarations),
    )


def _import_component(path: str, *, workspace: Workspace, context: str) -> type[Signal]:
    """Resolve ``module:Class`` to a ``Signal`` subclass, or refuse naming the path."""
    if path.count(":") != 1 or path.startswith(":") or path.endswith(":"):
        raise SpecError(
            f"{context}component path {path!r} is not 'module:Class'; a component is named by "
            "its import path and its class, separated by one colon"
        )
    module_name, _, attribute = path.partition(":")
    _insert_workspace_on_sys_path(workspace.root)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise SpecError(
            f"{context}component path {path!r} is not importable: module {module_name!r} raised "
            f"{type(exc).__name__}: {exc}. Component paths resolve against the workspace "
            f"({workspace.root}) and the installed environment"
        ) from None
    try:
        obj = getattr(module, attribute)
    except AttributeError:
        raise SpecError(
            f"{context}component path {path!r} is unknown: module {module_name!r} "
            f"({getattr(module, '__file__', 'no file')}) defines no {attribute!r}"
        ) from None
    if not (isinstance(obj, type) and issubclass(obj, Signal)):
        raise SpecError(
            f"{context}component path {path!r} names {obj!r}, which is not a subclass of "
            "prism.signal.base.Signal; a component is a Signal or it is not a component"
        )
    return obj


def _insert_workspace_on_sys_path(root: Path) -> str:
    """Put ``workspace.root`` at the front of ``sys.path`` (§4), idempotently.

    An installed console script's ``sys.path[0]`` is the script directory, not
    the project, so without this the project's ``strategies`` package is
    invisible to the engine that has to import it.
    """
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return entry


def _check_schedule(spec: ComponentSpec, component: Signal, *, context: str) -> None:
    """Refuse a class-declared schedule field that contradicts the table (§3.3)."""
    cls = type(component)
    for field_name in SCHEDULE_FIELDS:
        declared = _class_declared(cls, field_name)
        if declared is None:
            continue
        try:
            value = getattr(component, field_name)
        except Exception as exc:
            raise SpecError(
                f"{context}component {spec.id!r} at {spec.path!r} declares {field_name!r} but "
                f"reading it raised {type(exc).__name__}: {exc}"
            ) from None
        table_value = getattr(spec, field_name)
        if value != table_value:
            raise SpecError(
                f"{context}component {spec.id!r} at {spec.path!r} declares {field_name}={value!r} "
                f"while its configuration table says {table_value!r}; the two must agree "
                "(the declaration is not silently overridden, §3.3)"
            )


def _class_declared(cls: type, field_name: str) -> Any | None:
    """The declaration of ``field_name`` below ``Signal``, or None when the class
    inherits the base default and therefore declares nothing."""
    for klass in cls.__mro__:
        if klass is Signal:
            return None
        if field_name in vars(klass):
            return vars(klass)[field_name]
    return None


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _parse_strategy(table: Mapping[str, Any], *, where: str) -> StrategySpec:
    permitted = (
        "name",
        "combination",
        "renormalize",
        "universe_index",
        "universe_list",
        "universe_as_of",
        "universe_registration",
        "components",
        "eligibility",
    )
    _reject_unknown(table, permitted, where=f"{where} [strategy]")
    name = _require_str(table, "name", where=f"{where} [strategy]")
    combination = _require_choice(table, "combination", COMBINATIONS, where=f"{where} [strategy]")
    renormalize = _optional_bool(table, "renormalize", default=False, where=f"{where} [strategy]")

    has_index = table.get("universe_index") is not None
    has_list = table.get("universe_list") is not None
    if has_index == has_list:
        raise SpecError(
            f"{where} [strategy] declares "
            f"{'both universe_index and universe_list' if has_index else 'neither universe_index nor universe_list'}; "
            "a strategy declares exactly one (§3.3)"
        )
    universe_index: str | None = None
    universe_list: tuple[str, ...] | None = None
    universe_as_of: str | None = None
    universe_registration: str | None = None
    if has_index:
        universe_index = _require_str(table, "universe_index", where=f"{where} [strategy]")
        for key in ("universe_as_of", "universe_registration"):
            if key in table:
                raise SpecError(
                    f"{where} [strategy] declares {key!r} with universe_index; it belongs to universe_list"
                )
    else:
        universe_list = tuple(_require_str_list(table, "universe_list", where=f"{where} [strategy]"))
        if not universe_list:
            raise SpecError(f"{where} [strategy] declares an empty universe_list")
        universe_as_of = _require_scalar_str(table, "universe_as_of", where=f"{where} [strategy]")
        universe_registration = _require_scalar_str(table, "universe_registration", where=f"{where} [strategy]")

    raw_components = table.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise SpecError(f"{where} declares no [[strategy.components]]; a strategy runs at least one component")
    components = tuple(_parse_component(entry, index=index, where=where) for index, entry in enumerate(raw_components))
    ids = [component.id for component in components]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicates:
        raise SpecError(
            f"{where} declares component id(s) {duplicates} more than once; component state is "
            "keyed by id, so two components cannot share one"
        )
    eligibility = _parse_eligibility(table.get("eligibility", {}), where=where)
    return StrategySpec(
        name=name,
        combination=combination,
        renormalize=renormalize,
        components=components,
        eligibility=eligibility,
        universe_index=universe_index,
        universe_list=universe_list,
        universe_as_of=universe_as_of,
        universe_registration=universe_registration,
    )


def _parse_component(entry: Any, *, index: int, where: str) -> ComponentSpec:
    label = f"{where} [[strategy.components]] #{index}"
    if not isinstance(entry, Mapping):
        raise SpecError(f"{label} is {entry!r}, not a table")
    permitted = ("id", "path", "weight", "to_weights", *SCHEDULE_FIELDS, "params", "to_weights_params")
    _reject_unknown(entry, permitted, where=label)
    component_id = _require_str(entry, "id", where=label)
    path = _require_str(entry, "path", where=label)
    to_weights = _require_choice(entry, "to_weights", TO_WEIGHTS_RULES, where=label)
    weight = _optional_float(entry, "weight", default=1.0, where=label)
    schedule = {field: _require_int(entry, field, minimum=1, where=label) for field in SCHEDULE_FIELDS}
    if schedule["valid_for_bars"] < schedule["update_every_bars"]:
        raise SpecError(
            f"{label} declares valid_for_bars={schedule['valid_for_bars']} below "
            f"update_every_bars={schedule['update_every_bars']}; an output that expires before "
            "its own next refresh has no reachable valid state (§3.3)"
        )
    params = entry.get("params", {})
    if not isinstance(params, Mapping):
        raise SpecError(f"{label} declares params={params!r}, which is not a table")
    to_weights_params = resolve_to_weights_params(to_weights, entry.get("to_weights_params"), where=label)
    return ComponentSpec(
        id=component_id,
        path=path,
        weight=weight,
        to_weights=to_weights,
        params=dict(params),
        to_weights_params=to_weights_params,
        **schedule,
    )


def _parse_eligibility(table: Any, *, where: str) -> Eligibility:
    label = f"{where} [strategy.eligibility]"
    if not isinstance(table, Mapping):
        raise SpecError(f"{label} is {table!r}, not a table")
    _reject_unknown(table, tuple(ELIGIBILITY_DEFAULTS), where=label)
    return Eligibility(
        min_price=_optional_float(table, "min_price", default=0.0, where=label, minimum=0.0),
        min_median_dollar_volume=_optional_float(
            table, "min_median_dollar_volume", default=0.0, where=label, minimum=0.0
        ),
        dollar_volume_window=_optional_int(table, "dollar_volume_window", default=20, where=label, minimum=1),
        return_history_bars=_optional_int(table, "return_history_bars", default=0, where=label, minimum=0),
        listing_seasoning_bars=_optional_int(table, "listing_seasoning_bars", default=0, where=label, minimum=0),
        membership_required=_optional_bool(table, "membership_required", default=False, where=label),
        exclude_asset_classes=tuple(_optional_str_list(table, "exclude_asset_classes", where=label)),
    )


def _parse_construction(table: Mapping[str, Any], *, root: Path, where: str) -> ConstructionSpec:
    label = f"{where} [construction]"
    kind = _require_choice(table, "kind", CONSTRUCTION_KINDS, where=label)
    calibration = _resolve_file(table.get("calibration"), root=root, key="calibration", where=label)
    risk_model = _resolve_file(table.get("risk_model"), root=root, key="risk_model", where=label)
    options = {key: value for key, value in table.items() if key not in ("kind", "calibration", "risk_model")}
    return ConstructionSpec(kind=kind, calibration=calibration, risk_model=risk_model, options=options)


def _parse_policy(table: Mapping[str, Any], *, root: Path, where: str) -> PolicyRefs:
    label = f"{where} [policy]"
    _reject_unknown(table, POLICY_KEYS, where=label)
    resolved: dict[str, Path] = {}
    for key in POLICY_KEYS:
        if key not in table:
            raise SpecError(
                f"{label} declares no {key!r}; all five policy files are required, because a "
                "policy the specification does not name is a silent default (§4)"
            )
        path = _resolve_file(table[key], root=root, key=key, where=label)
        assert path is not None  # _resolve_file only returns None for an absent value
        resolved[key] = path
    return PolicyRefs(**resolved)


def _resolve_file(value: Any, *, root: Path, key: str, where: str) -> Path | None:
    """Resolve a declared file reference against the workspace root, or refuse it."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{where} declares {key}={value!r}; it must be a non-empty path string")
    candidate = Path(value).expanduser()
    path = candidate if candidate.is_absolute() else (root / candidate)
    if not path.is_file():
        raise SpecError(
            f"{where} names {key}={value!r}, which does not exist at {path}; relative "
            f"references resolve against the workspace root ({root})"
        )
    return path


# --------------------------------------------------------------------------
# Rank-to-weight rules (§3.3)
# --------------------------------------------------------------------------


def resolve_to_weights_params(
    rule: str,
    params: Mapping[str, Any] | None,
    *,
    where: str = "to_weights_params",
) -> dict[str, Any]:
    """Validate ``to_weights_params`` for ``rule`` and fill in its defaults.

    The result is what enters the resolved specification and therefore
    ``config_id``: a cap left implicit is recorded at the value the rule will
    use, never left to be inferred from the code.
    """
    if rule not in TO_WEIGHTS_RULES:
        raise SpecError(
            f"{where}: unknown to_weights rule {rule!r}; permitted rules are {', '.join(TO_WEIGHTS_RULES)}"
        )
    given = dict(params or {})
    schema = TO_WEIGHTS_PARAMS[rule]
    unknown = sorted(set(given) - set(schema))
    if unknown:
        permitted = ", ".join(sorted(schema)) or "no parameters"
        raise SpecError(f"{where}: to_weights={rule!r} takes {permitted}; it does not take {unknown}")
    resolved: dict[str, Any] = {}
    for key, default in schema.items():
        if key not in given:
            if default is _REQUIRED:
                raise SpecError(
                    f"{where}: to_weights={rule!r} requires {key!r} in [strategy.components.to_weights_params]"
                )
            resolved[key] = default
            continue
        value = given[key]
        if key in _INT_PARAMS:
            resolved[key] = _coerce_int(value, key=key, where=where, minimum=1 if key == "k" else 2)
        else:
            resolved[key] = _coerce_float(value, key=key, where=where)
    if rule == "decile_long_short" and not 0.0 < float(resolved["decile"]) <= 0.5:
        raise SpecError(f"{where}: decile must be in (0, 0.5], got {resolved['decile']!r}")
    if rule == "decile_long_short_buffered" and not (
        0.0 < float(resolved["decile"]) <= float(resolved["hold_quantile"]) <= 0.5
    ):
        raise SpecError(
            f"{where}: need 0 < decile <= hold_quantile <= 0.5, got decile={resolved['decile']!r}, "
            f"hold_quantile={resolved['hold_quantile']!r}"
        )
    for cap in ("max_gross", "max_symbol_abs_weight"):
        if cap in resolved and not float(resolved[cap]) > 0.0:
            raise SpecError(f"{where}: {cap} must be > 0, got {resolved[cap]!r}")
    if rule == "sign_inverse_vol" and int(resolved["annualization_bars"]) < 1:
        raise SpecError(f"{where}: annualization_bars must be >= 1, got {resolved['annualization_bars']!r}")
    return resolved


def apply_to_weights(
    rule: str,
    scores: pd.Series,
    params: Mapping[str, Any] | None = None,
    *,
    close: pd.DataFrame | None = None,
    no_decision: Iterable[str] | None = None,
    previous: Mapping[str, float] | None = None,
) -> pd.Series:
    """Turn one decision bar's cross-section of scores into target weights.

    ``scores`` is the component's output for the decision bar, indexed by sid;
    ``close`` is the close panel on the component's ``scoring_basis``, ending at
    that bar, required by ``sign_inverse_vol`` and unused by the others. The
    returned Series is indexed like ``scores``: an explicit ``0.0`` where the
    rule declines a name (so a name leaving the book fully exits) and NaN only
    where the component marked ``no_decision``, which carries no decision at all.

    ``decile_long_short`` and ``sign_inverse_vol`` delegate to
    ``prism.portfolio.construct``, so the recorded books' weights are reproduced
    exactly rather than re-derived.

    ``decile_long_short_buffered`` is the one stateful rule: ``previous`` is the
    component's own sub-targets from its last decision (sid -> weight), whose signs say
    which leg each name was in. It never sees fills or held shares, only the rule's own
    last output.
    """
    if not isinstance(scores, pd.Series):
        raise SpecError(f"scores must be a pandas Series indexed by sid, got {type(scores).__name__}")
    if scores.index.has_duplicates:
        raise SpecError("scores index has duplicate sids; a cross-section holds one score per name")
    resolved = resolve_to_weights_params(rule, params)
    values = _numeric_scores(scores)

    if rule == "none":
        weights = values
    elif rule == "decile_long_short":
        frame = construct_decile_neutral(
            values.to_frame().T,
            decile=float(resolved["decile"]),
            max_gross=float(resolved["max_gross"]),
            max_symbol_abs_weight=float(resolved["max_symbol_abs_weight"]),
        )
        weights = frame.iloc[0]
    elif rule == "decile_long_short_buffered":
        prior = pd.Series(dict(previous or {}), dtype=float).reindex(values.index).fillna(0.0)
        row = buffered_decile_row(
            values.to_numpy(dtype=float),
            np.sign(prior.to_numpy(dtype=float)),
            decile=float(resolved["decile"]),
            hold=float(resolved["hold_quantile"]),
        )
        capped = cap_book(
            pd.DataFrame([row], columns=values.index),
            float(resolved["max_gross"]),
            float(resolved["max_symbol_abs_weight"]),
        )
        weights = capped.iloc[0]
    elif rule == "sign_inverse_vol":
        weights = _sign_inverse_vol(values, close=close, params=resolved)
    else:
        weights = _top_k_equal_weight(values, params=resolved)

    weights = weights.astype(float)
    weights.index = scores.index
    weights.name = scores.name
    if no_decision is not None:
        marked = [sid for sid in dict.fromkeys(no_decision) if sid in weights.index]
        if marked:
            weights.loc[marked] = np.nan
    return weights


def _numeric_scores(scores: pd.Series) -> pd.Series:
    """Coerce a score cross-section to float, non-finite values becoming NaN."""
    return pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)


def _sign_inverse_vol(scores: pd.Series, *, close: pd.DataFrame | None, params: Mapping[str, Any]) -> pd.Series:
    """One decision bar of ``construct_inverse_vol_targets`` (§3.3).

    The whole close panel is required, not a trailing slice: the EWMA is
    recursive from the panel's first bar (``adjust=False``), so truncating the
    history would silently change the volatility and therefore the weights.
    """
    if close is None:
        raise SpecError(
            "to_weights='sign_inverse_vol' needs the close panel on the component's scoring_basis: "
            "its declared inputs are the score and the close panel, not the score alone (§3.3)"
        )
    if not close.columns.equals(scores.index):
        raise SpecError(
            "the close panel's columns must match the score index exactly for "
            "to_weights='sign_inverse_vol'; got "
            f"{len(close.columns)} columns against {len(scores.index)} scores"
        )
    if len(close) == 0:
        raise SpecError("to_weights='sign_inverse_vol' needs a non-empty close panel")
    frame = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
    frame.iloc[-1] = scores.to_numpy(dtype=float)
    targets = construct_inverse_vol_targets(
        frame,
        close,
        vol_ewma_bars=int(params["vol_ewma_bars"]),
        max_gross=float(params["max_gross"]),
        max_symbol_abs_weight=float(params["max_symbol_abs_weight"]),
        annualization_bars=int(params["annualization_bars"]),
    )
    return targets.iloc[-1]


def _top_k_equal_weight(scores: pd.Series, *, params: Mapping[str, Any]) -> pd.Series:
    """The long-only rule of §3.3: the ``k`` highest finite scores, equally weighted.

    Ties break by ascending sid, so the rule is deterministic under any column
    order. A cross-section with fewer than ``k`` finite scores spreads
    ``max_gross`` over the finite scores it has and records the shortfall,
    rather than levering the survivors or silently under-investing.
    """
    k = int(params["k"])
    max_gross = float(params["max_gross"])
    weights = pd.Series(0.0, index=scores.index, dtype=float)
    finite = scores[np.isfinite(scores.to_numpy(dtype=float))]
    if finite.empty:
        return weights
    if len(finite) < k:
        logger.warning(
            "top_k_equal_weight: %d finite score(s) for k=%d; spreading max_gross=%.6g over the %d available",
            len(finite),
            k,
            max_gross,
            len(finite),
        )
    ordered = finite.sort_index(kind="stable").sort_values(ascending=False, kind="stable")
    taken = ordered.index[: min(k, len(ordered))]
    weights.loc[taken] = max_gross / float(len(taken))
    capped = cap_book(
        weights.to_frame().T,
        max_gross=max_gross,
        max_symbol_abs_weight=float(params["max_symbol_abs_weight"]),
    )
    return capped.iloc[0]


# --------------------------------------------------------------------------
# Identity (§3.12)
# --------------------------------------------------------------------------


def engine_version() -> str:
    """The installed engine version, ``importlib.metadata.version("prism-trading")``.

    The pre-rename distribution name is consulted second so the loader works in
    a checkout whose stage 1 packaging correction has not landed; the package
    ``__version__`` is the last resort, for a source tree with no installed
    distribution at all.
    """
    return _installed_engine_version()


def _installed_engine_version() -> str:
    for distribution in (ENGINE_DISTRIBUTION, _LEGACY_ENGINE_DISTRIBUTION):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    import prism

    return str(prism.__version__)


def _component_identity(spec: ComponentSpec, component: Signal, declarations: Mapping[str, Any]) -> dict[str, Any]:
    """The component block of §3.12's ``common``.

    ``source_sha256`` hashes the module that *defines* the class, not the module
    the path names: a thin re-export is not an implementation, and the reference
    project's ``strategies/momentum.py`` re-export must identify the engine's
    momentum node rather than the shim.
    """
    cls = type(component)
    source = _source_file(cls, spec=spec)
    distribution, version = _distribution_of(cls)
    return {
        "component_id": spec.id,
        "import_path": spec.path,
        "distribution": distribution,
        "version": version,
        "source_sha256": _sha256_file(source),
        "git_commit": _git_commit(source.parent),
        "output_kind": declarations["output_kind"],
        "units": declarations["units"],
        "scoring_basis": declarations["scoring_basis"],
        "horizon_bars": int(declarations["horizon_bars"]),
        "update_every_bars": int(declarations["update_every_bars"]),
        "valid_for_bars": int(declarations["valid_for_bars"]),
    }


def _source_file(cls: type, *, spec: ComponentSpec) -> Path:
    source = inspect.getsourcefile(cls) or getattr(sys.modules.get(cls.__module__), "__file__", None)
    if not source or not Path(source).is_file():
        raise SpecError(
            f"component {spec.id!r} at {spec.path!r} resolves to {cls!r}, whose source file is not "
            "on disk; a component with no hashable implementation has no identity (§3.12)"
        )
    return Path(source)


def _distribution_of(cls: type) -> tuple[str, str]:
    """The distribution and version providing ``cls``, or ``unavailable`` twice.

    A component living in the project's own ``strategies`` package is normally
    not an installed distribution; that fact is recorded as ``unavailable``
    rather than as a null, which would read as a configured absence (§3.12).
    """
    top_level = cls.__module__.split(".", 1)[0]
    for name in _distributions_for(top_level):
        try:
            return name, importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return UNAVAILABLE, UNAVAILABLE


@functools.lru_cache(maxsize=None)
def _distributions_for(top_level: str) -> tuple[str, ...]:
    try:
        mapping = importlib.metadata.packages_distributions()
    except Exception:  # pragma: no cover - a broken environment metadata store
        return ()
    return tuple(sorted(mapping.get(top_level, ())))


@functools.lru_cache(maxsize=None)
def _git_commit(directory: Path) -> str | None:
    """The HEAD commit of the repository holding a component's source, or None."""
    return current_git_commit(directory)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_content_digest(path: Path) -> str:
    """A referenced file's content as it enters the identity: its sha256."""
    if not path.is_file():
        raise SpecError(f"referenced file {path} has disappeared; config_id covers its content, not its path")
    return _sha256_file(path)


def _optional_content_digest(path: Path | None) -> str | None:
    return None if path is None else _file_content_digest(path)


def _jsonable(value: Any) -> Any:
    """Plain JSON values for the identity payload; dates become ISO strings."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


# --------------------------------------------------------------------------
# TOML field helpers
# --------------------------------------------------------------------------


def _reject_unknown(table: Mapping[str, Any], permitted: Sequence[str], *, where: str) -> None:
    unknown = sorted(set(table) - set(permitted))
    if unknown:
        raise SpecError(f"{where} declares unknown key(s) {unknown}; permitted keys are {sorted(permitted)}")


def _require_table(table: Mapping[str, Any], key: str, *, where: str) -> Mapping[str, Any]:
    value = table.get(key)
    if not isinstance(value, Mapping):
        raise SpecError(f"{where} declares no [{key}] block" if value is None else f"{where} [{key}] is not a table")
    return value


def _require_str(table: Mapping[str, Any], key: str, *, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{where} declares {key}={value!r}; it must be a non-empty string")
    return value


def _require_scalar_str(table: Mapping[str, Any], key: str, *, where: str) -> str:
    """Like ``_require_str`` but accepts a TOML date, which parses to ``datetime.date``."""
    value = table.get(key)
    if value is None:
        raise SpecError(f"{where} declares no {key!r}")
    if isinstance(value, str):
        return _require_str(table, key, where=where)
    return str(value)


def _require_choice(table: Mapping[str, Any], key: str, permitted: Sequence[str], *, where: str) -> str:
    value = table.get(key)
    if value not in permitted:
        raise SpecError(f"{where} declares {key}={value!r}; permitted values are {', '.join(permitted)}")
    return str(value)


def _require_str_list(table: Mapping[str, Any], key: str, *, where: str) -> list[str]:
    value = table.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise SpecError(f"{where} declares {key}={value!r}; it must be a list of non-empty strings")
    return list(value)


def _optional_str_list(table: Mapping[str, Any], key: str, *, where: str) -> list[str]:
    if key not in table:
        return []
    return _require_str_list(table, key, where=where)


def _require_int(table: Mapping[str, Any], key: str, *, minimum: int, where: str) -> int:
    if key not in table:
        raise SpecError(f"{where} declares no {key!r}; the component table carries all three schedule fields (§3.3)")
    return _coerce_int(table[key], key=key, where=where, minimum=minimum)


def _optional_int(table: Mapping[str, Any], key: str, *, default: int, minimum: int, where: str) -> int:
    if key not in table:
        return default
    return _coerce_int(table[key], key=key, where=where, minimum=minimum)


def _coerce_int(value: Any, *, key: str, where: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpecError(f"{where} declares {key}={value!r}; it must be an integer")
    if value < minimum:
        raise SpecError(f"{where} declares {key}={value!r}; it must be >= {minimum}")
    return int(value)


def _optional_float(
    table: Mapping[str, Any], key: str, *, default: float, where: str, minimum: float | None = None
) -> float:
    if key not in table:
        return default
    value = _coerce_float(table[key], key=key, where=where)
    if minimum is not None and value < minimum:
        raise SpecError(f"{where} declares {key}={value!r}; it must be >= {minimum}")
    return value


def _coerce_float(value: Any, *, key: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{where} declares {key}={value!r}; it must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise SpecError(f"{where} declares {key}={value!r}; it must be finite")
    return number


def _optional_bool(table: Mapping[str, Any], key: str, *, default: bool, where: str) -> bool:
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, bool):
        raise SpecError(f"{where} declares {key}={value!r}; it must be true or false")
    return value
