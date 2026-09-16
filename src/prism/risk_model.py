"""The factor risk model: one snapshot, one variance, one cone (spec section 3.5, decision D9).

A :class:`RiskSnapshot` is the whole risk surface the constructor sees on a decision bar:
factor loadings ``B``, a **full** factor covariance ``omega``, a diagonal residual ``d``, and
the horizon and scaling those two are stated at. ``Sigma = B omega B' + diag(d)`` is never
formed: :meth:`RiskSnapshot.variance` evaluates ``w' Sigma w`` through the factor projection
``B' w``, and :meth:`RiskSnapshot.cone_operator` hands the solver the two blocks of a
second-order cone whose squared norm *is* that quadratic form.

**The covariance is full, and that is the point.** Exposure one half to each of two
unit-variance factors correlated at 0.9 carries variance
``0.25 + 0.25 + 2 * 0.25 * 0.9 = 0.95``; a diagonal-only representation reports ``0.50`` and
under-prices the book by nearly half. Any positive residual adds ``sum_i d_i w_i^2`` on top.
The cheap representation comes from the small factor dimension and the diagonal residual, not
from dropping factor correlations. ``tests/test_risk_model.py`` holds that check.

**Eigendecomposition, never Cholesky** (decision D9). A PCA factor covariance is routinely
positive semidefinite and *singular*: ``estimate_eigenportfolios`` gives a zero-variance column
zero weight in every eigenportfolio (``residual/factors.py:341-346``), so a constant return
column contributes a factor with zero variance, and ``numpy.linalg.cholesky`` raises
``LinAlgError`` on exactly that matrix. The factorization is ``omega = V diag(lam) V'`` by
``eigh`` with ``lam`` clipped from below at ``eigenvalue_floor_relative * max(lam)``, and the
cone operator is ``(B @ V @ diag(sqrt(lam_clipped)), sqrt(d))``. The clip is a declared
modeling term and is recorded whenever the operator is built, with whether it bound [Q15].

**Names without a current estimate.** The estimators in ``residual/factors.py`` refuse exactly
the names the constructor keeps in the program under reduce-only restrictions:

* ``compute_eligibility`` (``factors.py:227``) requires a full ``corr_window`` of finite
  trailing returns, a close at or above ``min_price``, and a trailing median dollar volume
  clearing ``min_median_dollar_volume``, ANDed with point-in-time index membership when a
  membership mask is supplied; declared sector ETFs are unconditionally ineligible as legs;
* ``estimate_eigenportfolios`` (``factors.py:339``) raises on a window that is not fully
  finite, and zeroes a zero-variance column rather than letting it poison the correlation;
* ``batched_factor_ols`` (``factors.py:412-414``) requires more observations than factors plus
  intercept, and full finiteness of both windows.

A name in the snapshot with no current estimate therefore takes the declared substitution:
zero factor loadings, a residual variance at the declared conservative floor
(``residual_floor_var_annual / annualization_bars * horizon_bars``, 0.0075 at the reference
values, an 8.66 percent standard deviation over 21 bars), a reduce-only pin, and a per-name
entry in the snapshot's ``method`` block naming the reason. Stale, halted, and
unresolved-claim holdings take the same substitution and are excluded from estimation
entirely, per the policy's ``[substitution]`` block: a holding carried at its last valuation
presents a constant return column, for which ``batched_factor_ols`` returns a zero loading
vector and zero residual variance, so a book of stale names would satisfy any variance budget
trivially. The residual floor is the second defence and applies to every name, estimated or
substituted, after shrinkage.

**Validation before any solve.** :func:`validate_snapshot` is the gate, and it *reports*
rather than raises: a rejection is the recorded acceptance outcome
``"validation rejects, no orders"``, not an exception thrown from inside the cone operator.
It rejects a snapshot whose ``sids`` do not cover the constructor universe, one with a
non-finite entry in ``B``, ``omega`` or ``d``, a negative ``d``, an ``omega`` whose asymmetry
exceeds ``symmetry_tolerance``, and an ``omega`` with an eigenvalue below
``-eigenvalue_reject_relative * max(lam)`` or with a non-positive largest eigenvalue.

This module computes ``omega`` and ``d`` itself; ``residual/factors.py`` supplies inputs only
and returns no covariance anywhere.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from prism.residual.factors import (
    batched_factor_ols,
    estimate_eigenportfolios,
    factor_returns_from_window,
)

__all__ = [
    "FACTOR_MODES",
    "REQUIRED_POLICY_KEYS",
    "SCALINGS",
    "SHRINK_TARGETS",
    "ConeOperator",
    "RiskModelError",
    "RiskPolicy",
    "RiskSnapshot",
    "SnapshotValidation",
    "SubstitutionPolicy",
    "build_snapshot",
    "validate_snapshot",
]

#: The two declared factor constructions of section 3.5. ``omega`` is estimated under both.
FACTOR_MODES: tuple[str, ...] = ("pca", "etf")

#: The two declared horizon conventions. ``per_bar_times_h`` multiplies a per-bar variance by
#: ``horizon_bars`` and therefore ignores autocorrelation over the horizon, which is acceptable
#: because it is declared; ``estimated_at_horizon`` estimates on overlapping horizon returns.
SCALINGS: tuple[str, ...] = ("per_bar_times_h", "estimated_at_horizon")

#: Declared shrinkage targets for the factor covariance.
SHRINK_TARGETS: tuple[str, ...] = ("diagonal", "identity")

#: Every key a ``risk/*.toml`` declares (section 3.13). A key omitted is refused rather than
#: defaulted: a shrinkage or a floor that was never declared is indistinguishable in the
#: record from one that was declared at that value.
REQUIRED_POLICY_KEYS: tuple[str, ...] = (
    "kind",
    "factor_mode",
    "n_factors",
    "loadings_window_bars",
    "covariance_window_bars",
    "omega_shrink",
    "omega_shrink_target",
    "eigenvalue_floor_relative",
    "eigenvalue_reject_relative",
    "symmetry_tolerance",
    "residual_shrink",
    "residual_floor_var_annual",
    "scaling",
    "horizon_bars",
    "annualization_bars",
    "refresh_every_bars",
)

#: The recorded acceptance outcome of a rejected snapshot (section 3.5).
REJECTED_OUTCOME = "validation rejects, no orders"


class RiskModelError(ValueError):
    """A risk policy, a snapshot, or an estimation input that cannot be used as written."""


def _content_id(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")).hexdigest()


def _positive_float(value: object, *, name: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number <= 0.0:
        raise RiskModelError(f"{name} must be finite and > 0, got {value!r}")
    return number


def _non_negative_float(value: object, *, name: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number < 0.0:
        raise RiskModelError(f"{name} must be finite and >= 0, got {value!r}")
    return number


def _positive_int(value: object, *, name: str) -> int:
    number = int(value)  # type: ignore[arg-type]
    if number <= 0:
        raise RiskModelError(f"{name} must be an integer > 0, got {value!r}")
    return number


def _unit_interval(value: object, *, name: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RiskModelError(f"{name} must be a shrinkage intensity in [0, 1], got {value!r}")
    return number


# --------------------------------------------------------------------------
# The declared policy file
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SubstitutionPolicy:
    """The ``[substitution]`` block: which holdings are excluded from estimation entirely.

    A holding carried at a stale valuation, or one with an unresolved corporate-action claim,
    presents a constant or fabricated return column. Estimating it is worse than not
    estimating it, so it takes the declared substitution instead.
    """

    exclude_valuation_status: tuple[str, ...] = ()
    exclude_halted: bool = True

    def __post_init__(self) -> None:
        statuses = tuple(str(item) for item in self.exclude_valuation_status)
        object.__setattr__(self, "exclude_valuation_status", statuses)
        object.__setattr__(self, "exclude_halted", bool(self.exclude_halted))

    def declaration(self) -> dict[str, Any]:
        return {
            "exclude_valuation_status": list(self.exclude_valuation_status),
            "exclude_halted": self.exclude_halted,
        }


@dataclass(frozen=True)
class RiskPolicy:
    """The parsed form of a ``risk/*.toml`` file (section 3.5).

    Every setting here is held fixed across the two convex arms, which is what makes "the arms
    share their risk model" checkable rather than asserted: the file's content enters
    ``config_id`` through the ``[construction] risk_model`` key of section 3.3, so two arms
    that share a risk model share a hash.
    """

    kind: str
    factor_mode: Literal["pca", "etf"]
    n_factors: int
    loadings_window_bars: int
    covariance_window_bars: int
    omega_shrink: float
    omega_shrink_target: Literal["diagonal", "identity"]
    eigenvalue_floor_relative: float
    eigenvalue_reject_relative: float
    symmetry_tolerance: float
    residual_shrink: float
    residual_floor_var_annual: float
    scaling: Literal["per_bar_times_h", "estimated_at_horizon"]
    horizon_bars: int
    annualization_bars: int
    refresh_every_bars: int
    substitution: SubstitutionPolicy = field(default_factory=SubstitutionPolicy)
    source: str = ""
    source_sha256: str = ""
    policy_id: str = ""

    def __post_init__(self) -> None:
        if self.kind != "factor_diagonal":
            raise RiskModelError(f"risk policy kind must be 'factor_diagonal', got {self.kind!r}")
        if self.factor_mode not in FACTOR_MODES:
            raise RiskModelError(f"factor_mode must be one of {FACTOR_MODES}, got {self.factor_mode!r}")
        if self.scaling not in SCALINGS:
            raise RiskModelError(f"scaling must be one of {SCALINGS}, got {self.scaling!r}")
        if self.omega_shrink_target not in SHRINK_TARGETS:
            raise RiskModelError(
                f"omega_shrink_target must be one of {SHRINK_TARGETS}, got {self.omega_shrink_target!r}"
            )
        for name in (
            "n_factors",
            "loadings_window_bars",
            "covariance_window_bars",
            "horizon_bars",
            "annualization_bars",
            "refresh_every_bars",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        for name in ("omega_shrink", "residual_shrink"):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name=name))
        for name in ("eigenvalue_floor_relative", "eigenvalue_reject_relative", "symmetry_tolerance"):
            object.__setattr__(self, name, _non_negative_float(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "residual_floor_var_annual",
            _positive_float(self.residual_floor_var_annual, name="residual_floor_var_annual"),
        )
        if self.loadings_window_bars > self.covariance_window_bars:
            raise RiskModelError(
                f"loadings_window_bars ({self.loadings_window_bars}) exceeds covariance_window_bars "
                f"({self.covariance_window_bars}): the loadings are regressed on the factor returns "
                "computed over the covariance window, so the shorter window is the regression's"
            )
        if self.loadings_window_bars < self.n_factors + 2:
            raise RiskModelError(
                f"loadings_window_bars ({self.loadings_window_bars}) leaves no residual degrees of freedom "
                f"for {self.n_factors} factors plus an intercept; batched_factor_ols refuses it "
                "(factors.py:412) and a residual variance divided by a non-positive df is not a variance"
            )
        if not isinstance(self.substitution, SubstitutionPolicy):
            raise RiskModelError(f"substitution must be a SubstitutionPolicy, got {type(self.substitution).__name__}")
        if not self.policy_id:
            object.__setattr__(self, "policy_id", _content_id(self.declaration()))

    # -- derived, declared quantities -------------------------------------------

    @property
    def residual_floor_variance(self) -> float:
        """The conservative residual variance of a name with no current estimate, at the horizon.

        ``residual_floor_var_annual / annualization_bars * horizon_bars``: at the reference
        values ``0.09 / 252 * 21 = 0.0075``, an 8.66 percent standard deviation over 21 bars,
        deliberately conservative because the alternative is pricing the position as riskless.
        There is no second no-estimate key; this floor is also the clip applied to every
        estimated residual after shrinkage.
        """
        return self.residual_floor_var_annual / self.annualization_bars * self.horizon_bars

    def declaration(self) -> dict[str, Any]:
        """The declared content, as the mapping the policy file carries. Excludes provenance."""
        return {
            "kind": self.kind,
            "factor_mode": self.factor_mode,
            "n_factors": self.n_factors,
            "loadings_window_bars": self.loadings_window_bars,
            "covariance_window_bars": self.covariance_window_bars,
            "omega_shrink": self.omega_shrink,
            "omega_shrink_target": self.omega_shrink_target,
            "eigenvalue_floor_relative": self.eigenvalue_floor_relative,
            "eigenvalue_reject_relative": self.eigenvalue_reject_relative,
            "symmetry_tolerance": self.symmetry_tolerance,
            "residual_shrink": self.residual_shrink,
            "residual_floor_var_annual": self.residual_floor_var_annual,
            "scaling": self.scaling,
            "horizon_bars": self.horizon_bars,
            "annualization_bars": self.annualization_bars,
            "refresh_every_bars": self.refresh_every_bars,
            "substitution": self.substitution.declaration(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, source: str = "", source_sha256: str = "") -> "RiskPolicy":
        """Parse a declared risk policy, refusing an omitted key and an unknown one alike."""
        missing = [key for key in REQUIRED_POLICY_KEYS if key not in mapping]
        if missing:
            raise RiskModelError(
                f"the risk policy omits {missing}; every setting is declared explicitly, because a "
                "shrinkage or a floor that was never declared is indistinguishable in the record from "
                "one declared at that value (section 3.5)"
            )
        unknown = [key for key in mapping if key not in REQUIRED_POLICY_KEYS and key != "substitution"]
        if unknown:
            raise RiskModelError(
                f"the risk policy declares unknown keys {sorted(unknown)}; the schema of section 3.5 is "
                "closed, and a mistyped key silently ignored is a modeling term nobody declared"
            )
        block = mapping.get("substitution", {})
        if not isinstance(block, Mapping):
            raise RiskModelError(f"[substitution] must be a table, got {type(block).__name__}")
        unknown_sub = [key for key in block if key not in ("exclude_valuation_status", "exclude_halted")]
        if unknown_sub:
            raise RiskModelError(f"[substitution] declares unknown keys {sorted(unknown_sub)}")
        substitution = SubstitutionPolicy(
            exclude_valuation_status=tuple(block.get("exclude_valuation_status", ())),
            exclude_halted=bool(block.get("exclude_halted", True)),
        )
        return cls(
            kind=str(mapping["kind"]),
            factor_mode=str(mapping["factor_mode"]),  # type: ignore[arg-type]
            n_factors=mapping["n_factors"],
            loadings_window_bars=mapping["loadings_window_bars"],
            covariance_window_bars=mapping["covariance_window_bars"],
            omega_shrink=mapping["omega_shrink"],
            omega_shrink_target=str(mapping["omega_shrink_target"]),  # type: ignore[arg-type]
            eigenvalue_floor_relative=mapping["eigenvalue_floor_relative"],
            eigenvalue_reject_relative=mapping["eigenvalue_reject_relative"],
            symmetry_tolerance=mapping["symmetry_tolerance"],
            residual_shrink=mapping["residual_shrink"],
            residual_floor_var_annual=mapping["residual_floor_var_annual"],
            scaling=str(mapping["scaling"]),  # type: ignore[arg-type]
            horizon_bars=mapping["horizon_bars"],
            annualization_bars=mapping["annualization_bars"],
            refresh_every_bars=mapping["refresh_every_bars"],
            substitution=substitution,
            source=source,
            source_sha256=source_sha256,
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "RiskPolicy":
        """Load and parse a ``risk/*.toml``, carrying its path and content hash for the record."""
        import tomllib

        file_path = Path(path)
        raw = file_path.read_bytes()
        try:
            table = tomllib.loads(raw.decode("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise RiskModelError(f"{file_path}: not parseable TOML: {exc}") from exc
        return cls.from_mapping(table, source=str(file_path), source_sha256=hashlib.sha256(raw).hexdigest())


# --------------------------------------------------------------------------
# The snapshot
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConeOperator:
    """The two blocks of the second-order cone that stands for ``sqrt(w' Sigma w)``.

    ``factor`` is ``B @ V @ diag(sqrt(lam_clipped))`` with shape ``(N, K)`` and ``residual`` is
    ``sqrt(d)`` with shape ``(N,)``, so
    ``||factor' w||_2^2 + ||diag(residual) w||_2^2 = w' Sigma w`` exactly. The constructor
    stacks ``factor.T @ w`` and ``multiply(residual, w)`` inside one norm; nothing here forms
    an ``N x N`` matrix. It unpacks as the two-tuple section 3.5 writes.
    """

    factor: np.ndarray
    residual: np.ndarray

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield self.factor
        yield self.residual

    def quadratic_form(self, w: np.ndarray) -> float:
        """``||factor' w||^2 + ||residual * w||^2`` -- the value the cone stands for."""
        vector = np.asarray(w, dtype=float).reshape(-1)
        if vector.shape[0] != self.residual.shape[0]:
            raise RiskModelError(
                f"weights of length {vector.shape[0]} against an operator of {self.residual.shape[0]}"
            )
        projected = self.factor.T @ vector
        return float(projected @ projected + float(np.dot(self.residual * vector, self.residual * vector)))


@dataclass(frozen=True)
class RiskSnapshot:
    """The risk surface of one decision bar (section 3.5).

    ``sids`` fixes the ordering of ``w`` and must cover the constructor universe; a name it
    does not cover has no defined risk and :func:`validate_snapshot` rejects the snapshot
    rather than treating that name as riskless. ``B`` is ``(N, K)``, ``omega`` is the ``(K, K)``
    **full** factor covariance, and ``d`` is the ``(N,)`` diagonal residual, all stated over
    ``horizon_bars`` under the declared ``scaling``. ``method`` carries the estimator settings,
    the per-name substitutions with their reasons, and the fallbacks and clips that bound.

    The dataclass checks shapes only. Value checks -- finiteness, non-negativity, symmetry,
    the eigenvalue floor -- belong to :func:`validate_snapshot`, because a snapshot that fails
    them is a recorded acceptance outcome, not an exception at construction.
    """

    sids: tuple[str, ...]
    B: np.ndarray
    omega: np.ndarray
    d: np.ndarray
    horizon_bars: int
    scaling: Literal["per_bar_times_h", "estimated_at_horizon"]
    as_of: str
    method: dict[str, Any] = field(default_factory=dict)
    _cache: list[tuple[float, ConeOperator]] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        sids = tuple(str(sid) for sid in self.sids)
        if not sids:
            raise RiskModelError("a risk snapshot covers at least one sid")
        seen: set[str] = set()
        duplicates: set[str] = set()
        for sid in sids:
            if sid in seen:
                duplicates.add(sid)
            seen.add(sid)
        if duplicates:
            raise RiskModelError(
                f"sids fixes the ordering of w and must be unique; duplicated {sorted(duplicates)[:5]}"
            )
        object.__setattr__(self, "sids", sids)
        loadings = np.asarray(self.B, dtype=float)
        if loadings.ndim != 2:
            raise RiskModelError(f"B must be (N, K), got shape {loadings.shape}")
        covariance = np.asarray(self.omega, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise RiskModelError(f"omega must be square (K, K), got shape {covariance.shape}")
        residual = np.asarray(self.d, dtype=float).reshape(-1)
        n_names, n_factors = loadings.shape
        if n_names != len(sids):
            raise RiskModelError(f"B has {n_names} rows against {len(sids)} sids")
        if residual.shape[0] != len(sids):
            raise RiskModelError(f"d has {residual.shape[0]} entries against {len(sids)} sids")
        if covariance.shape[0] != n_factors:
            raise RiskModelError(f"omega is {covariance.shape} against {n_factors} loading columns")
        if n_factors < 1:
            raise RiskModelError("a snapshot carries at least one factor; a pure-diagonal risk model is not §3.5")
        object.__setattr__(self, "B", loadings)
        object.__setattr__(self, "omega", covariance)
        object.__setattr__(self, "d", residual)
        object.__setattr__(self, "horizon_bars", _positive_int(self.horizon_bars, name="horizon_bars"))
        if self.scaling not in SCALINGS:
            raise RiskModelError(f"scaling must be one of {SCALINGS}, got {self.scaling!r}")
        if not isinstance(self.method, dict):
            raise RiskModelError(f"method must be a dict, got {type(self.method).__name__}")

    # -- shape ------------------------------------------------------------------

    @property
    def n_names(self) -> int:
        return len(self.sids)

    @property
    def n_factors(self) -> int:
        return int(self.B.shape[1])

    @property
    def factor_ids(self) -> tuple[str, ...]:
        """The declared factor names, in the column order of ``B`` and ``omega``."""
        named = self.method.get("factor_ids")
        if named is None:
            return tuple(f"factor_{index + 1}" for index in range(self.n_factors))
        return tuple(str(item) for item in named)

    @property
    def reduce_only(self) -> tuple[str, ...]:
        """The names the substitution pinned reduce-only: no estimate, so no new exposure."""
        return tuple(str(sid) for sid in self.method.get("reduce_only", ()))

    @property
    def substitutions(self) -> dict[str, dict[str, Any]]:
        """Per-name substitution records, keyed by sid, each naming its reason."""
        recorded = self.method.get("substitutions", {})
        return {str(sid): dict(entry) for sid, entry in dict(recorded).items()}

    def loadings_frame(self) -> pd.DataFrame:
        """``B`` as a frame indexed by sid with the declared factor ids as columns."""
        return pd.DataFrame(self.B, index=list(self.sids), columns=list(self.factor_ids))

    # -- the two things the constructor asks for --------------------------------

    def align(self, w: Any) -> np.ndarray:
        """``w`` as a dense vector in ``sids`` order, refusing a name the snapshot does not cover."""
        if isinstance(w, pd.Series) or isinstance(w, Mapping):
            series = pd.Series(w, dtype=float)
            index = [str(sid) for sid in series.index]
            if len(set(index)) != len(index):
                raise RiskModelError("weights carry a duplicated sid; the ordering of w must be unambiguous")
            covered = set(self.sids)
            unknown = sorted({sid for sid in index if sid not in covered})
            if unknown:
                raise RiskModelError(
                    f"weights name {unknown[:5]}, which the snapshot does not cover; a name without an "
                    "estimate is not a name with zero risk (D9)"
                )
            declared = series.to_numpy(dtype=float)
            if not np.isfinite(declared).all():
                offenders = [sid for sid, value in zip(index, declared) if not np.isfinite(value)][:5]
                raise RiskModelError(
                    f"weights must be finite to carry a variance; non-finite at {offenders}. A name the "
                    "caller omits carries zero weight, but a name it names as NaN is a defect, not a zero"
                )
            series.index = pd.Index(index)
            # Only a name the caller did not mention is filled at zero; a NaN it did name raised above.
            vector = series.reindex(list(self.sids)).fillna(0.0).to_numpy(dtype=float)
        else:
            vector = np.asarray(w, dtype=float).reshape(-1)
            if vector.shape[0] != len(self.sids):
                raise RiskModelError(f"weights of length {vector.shape[0]} against {len(self.sids)} sids")
        if not np.isfinite(vector).all():
            raise RiskModelError("weights must be finite to carry a variance")
        return vector

    def variance(self, w: Any) -> float:
        """``w' (B omega B' + diag(d)) w``, never forming a dense ``N x N`` matrix.

        The factor term is evaluated on the ``K``-vector ``B' w`` and the residual term on the
        elementwise product, so the work is ``O(N K + K^2)`` and the memory ``O(N K)``.
        """
        vector = self.align(w)
        exposure = self.B.T @ vector
        factor_term = float(exposure @ self.omega @ exposure)
        residual_term = float(np.dot(self.d, vector * vector))
        return factor_term + residual_term

    def cone_operator(self, *, eigenvalue_floor_relative: float | None = None) -> ConeOperator:
        """``(B @ V @ diag(sqrt(clip(lam, floor))), sqrt(d))`` from ``omega = V diag(lam) V'``.

        By ``eigh``, never by Cholesky (decision D9): a PCA factor covariance is routinely
        positive semidefinite and singular, and ``numpy.linalg.cholesky`` raises
        ``LinAlgError`` on that input while ``eigh`` factorizes it. Eigenvalues are clipped
        from below at ``eigenvalue_floor_relative * max(lam)``, a declared modeling term taken
        from the risk policy file, and the clip is recorded in ``method['eigenvalue_clip']``
        with whether it bound [Q15].
        """
        floor_relative = eigenvalue_floor_relative
        if floor_relative is None:
            declared = self.method.get("eigenvalue_floor_relative")
            if declared is None:
                raise RiskModelError(
                    "the snapshot's method block declares no eigenvalue_floor_relative and none was passed; "
                    "the clip is a declared modeling term of the risk policy file, never a default (D9)"
                )
            floor_relative = float(declared)
        floor_relative = _non_negative_float(floor_relative, name="eigenvalue_floor_relative")
        for cached_floor, operator in self._cache:
            if cached_floor == floor_relative:
                return operator

        if not np.isfinite(self.omega).all():
            raise RiskModelError("omega carries a non-finite entry; validate_snapshot rejects it before any solve")
        if not np.isfinite(self.d).all() or (self.d < 0.0).any():
            raise RiskModelError(
                "d is non-finite or negative; validate_snapshot rejects it before any solve, so that the "
                "rejection is a recorded acceptance outcome rather than an exception from the cone operator"
            )
        eigenvalues, vectors = np.linalg.eigh(self.omega)
        largest = float(eigenvalues[-1])
        if not math.isfinite(largest) or largest <= 0.0:
            raise RiskModelError(
                f"omega's largest eigenvalue is {largest!r}; a factor covariance with no positive eigenvalue "
                "carries no risk at all and validate_snapshot rejects it before any solve"
            )
        floor = floor_relative * largest
        clipped = np.clip(eigenvalues, floor, None)
        n_clipped = int(np.count_nonzero(clipped > eigenvalues))
        self.method["eigenvalue_clip"] = {
            "floor_relative": floor_relative,
            "floor_absolute": floor,
            "max_eigenvalue": largest,
            "min_eigenvalue": float(eigenvalues[0]),
            "clipped": n_clipped,
            "binds": bool(n_clipped),
        }
        operator = ConeOperator(
            factor=self.B @ (vectors * np.sqrt(clipped)[np.newaxis, :]),
            residual=np.sqrt(self.d),
        )
        self._cache.append((floor_relative, operator))
        return operator

    # -- record ------------------------------------------------------------------

    def record(self) -> dict[str, Any]:
        """The snapshot as the record of section 3.12 carries it: shape, horizon, method."""
        return {
            "as_of": self.as_of,
            "n_names": self.n_names,
            "n_factors": self.n_factors,
            "horizon_bars": self.horizon_bars,
            "scaling": self.scaling,
            "method": self.method,
        }


# --------------------------------------------------------------------------
# Validation, before any solve
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotValidation:
    """The verdict on a snapshot, as a record rather than an exception (section 3.5)."""

    accepted: bool
    outcome: str
    findings: tuple[str, ...] = ()
    measures: dict[str, Any] = field(default_factory=dict)

    def require(self) -> None:
        """Raise when the verdict is a rejection -- for callers with no record to write to."""
        if not self.accepted:
            raise RiskModelError(f"{self.outcome}: " + "; ".join(self.findings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "outcome": self.outcome,
            "findings": list(self.findings),
            "measures": self.measures,
        }


def validate_snapshot(
    snapshot: RiskSnapshot,
    *,
    universe: Sequence[str],
    policy: RiskPolicy | None = None,
) -> SnapshotValidation:
    """Reject a snapshot the constructor must not solve against (section 3.5, decision D9).

    Rejections, in the order the section states them: ``sids`` that do not cover the
    constructor universe; a non-finite entry in ``B``, ``omega`` or ``d``; a negative ``d``; an
    ``omega`` whose asymmetry exceeds ``symmetry_tolerance``; and an ``omega`` with an
    eigenvalue below ``-eigenvalue_reject_relative * max(lam)`` or with a non-positive largest
    eigenvalue. When a ``policy`` is supplied the snapshot's horizon and scaling are checked
    against it too, since a snapshot stated at a different horizon than the one the calibration
    and the budget were declared at is not the risk the budget bounds.

    A residual variance below the floor is *not* a rejection: :func:`build_snapshot` clips it up
    and records the clip.
    """
    findings: list[str] = []
    measures: dict[str, Any] = {}

    covered = set(snapshot.sids)
    missing = [str(sid) for sid in universe if str(sid) not in covered]
    measures["universe"] = len({str(sid) for sid in universe})
    measures["covered"] = len(covered)
    measures["uncovered"] = len(missing)
    if missing:
        findings.append(
            f"the snapshot covers no estimate for {len(missing)} name(s) of the constructor universe, "
            f"including {missing[:5]}; sids fixes the ordering of w and a missing name has no defined risk"
        )

    for name, array in (("B", snapshot.B), ("omega", snapshot.omega), ("d", snapshot.d)):
        finite = np.isfinite(array)
        if not finite.all():
            findings.append(f"{name} carries {int((~finite).sum())} non-finite entr(ies)")

    negative = np.asarray(snapshot.d) < 0.0
    measures["min_residual_variance"] = float(np.min(snapshot.d)) if snapshot.d.size else float("nan")
    if bool(negative.any()):
        offenders = [str(sid) for sid, flag in zip(snapshot.sids, negative) if flag][:5]
        findings.append(
            f"d is negative for {int(negative.sum())} name(s), including {offenders}; that is not a variance"
        )

    symmetry_tolerance = 1.0e-12 if policy is None else policy.symmetry_tolerance
    reject_relative = 1.0e-08 if policy is None else policy.eigenvalue_reject_relative
    omega = np.asarray(snapshot.omega, dtype=float)
    if np.isfinite(omega).all():
        asymmetry = float(np.max(np.abs(omega - omega.T))) if omega.size else 0.0
        measures["asymmetry"] = asymmetry
        if asymmetry > symmetry_tolerance:
            findings.append(
                f"omega's asymmetry {asymmetry:.3e} exceeds symmetry_tolerance {symmetry_tolerance:.3e}; "
                "an asymmetric covariance is not a covariance and eigh would silently read one triangle"
            )
        eigenvalues = np.linalg.eigvalsh(0.5 * (omega + omega.T))
        largest = float(eigenvalues[-1])
        smallest = float(eigenvalues[0])
        measures["max_eigenvalue"] = largest
        measures["min_eigenvalue"] = smallest
        if not math.isfinite(largest) or largest <= 0.0:
            findings.append(
                f"omega's largest eigenvalue is {largest:.3e}, not positive; a factor block that carries no "
                "risk would price every factor exposure as free"
            )
        elif smallest < -reject_relative * largest:
            findings.append(
                f"omega's smallest eigenvalue {smallest:.3e} is below -{reject_relative:.3e} * "
                f"{largest:.3e}; that is an indefinite covariance, not the PSD-singular one the floor admits"
            )
    else:
        measures["asymmetry"] = float("nan")

    if policy is not None:
        if snapshot.horizon_bars != policy.horizon_bars:
            findings.append(
                f"the snapshot is stated over {snapshot.horizon_bars} bars against the policy's "
                f"{policy.horizon_bars}; the variance budget bounds risk at the declared horizon"
            )
        fallback = snapshot.method.get("scaling_fallback")
        if snapshot.scaling != policy.scaling and not fallback:
            findings.append(
                f"the snapshot declares scaling {snapshot.scaling!r} against the policy's {policy.scaling!r} "
                "and records no fallback"
            )
        if snapshot.n_factors != policy.n_factors:
            findings.append(
                f"the snapshot carries {snapshot.n_factors} factors against the policy's {policy.n_factors}"
            )

    accepted = not findings
    return SnapshotValidation(
        accepted=accepted,
        outcome="accepted" if accepted else REJECTED_OUTCOME,
        findings=tuple(findings),
        measures=measures,
    )


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------


def _shrunk_covariance(sample: np.ndarray, *, intensity: float, target: str) -> np.ndarray:
    """``(1 - s) * S + s * T`` with ``T`` the declared target, symmetrized exactly."""
    matrix = np.atleast_2d(np.asarray(sample, dtype=float))
    if target == "diagonal":
        prior = np.diag(np.diag(matrix))
    else:  # identity, scaled to the mean factor variance so the target is on the same scale
        prior = np.eye(matrix.shape[0]) * float(np.mean(np.diag(matrix)))
    shrunk = (1.0 - intensity) * matrix + intensity * prior
    return 0.5 * (shrunk + shrunk.T)


def _overlapping_sums(window: np.ndarray, horizon: int) -> np.ndarray:
    """Overlapping ``horizon``-bar factor returns as the arithmetic sum of the per-bar returns.

    A factor return here is a self-financing long-short portfolio return, so it aggregates by
    addition rather than by compounding; compounding would also be undefined for the
    ``<= -100 percent`` values a levered long-short leg can print.
    """
    cumulative = np.cumsum(window, axis=0)
    padded = np.vstack([np.zeros((1, window.shape[1])), cumulative])
    return padded[horizon:] - padded[:-horizon]


def _refuse_look_ahead(index: pd.Index, *, as_of: str) -> None:
    """The panel ends at the decision bar, never after it (SPEC: point-in-time causality).

    A snapshot attributed to bar ``t`` may read bars ``<= t`` only. When the index is
    datetime-like and ``as_of`` parses as a timestamp, a later bar in the panel is a
    look-ahead defect and is refused here rather than discovered in a suspiciously good
    backtest.
    """
    if len(index) == 0 or not isinstance(index, pd.DatetimeIndex):
        return
    try:
        decision = pd.Timestamp(as_of)
    except (TypeError, ValueError):
        return
    if index.tz is not None and decision.tz is None:
        decision = decision.tz_localize(index.tz)
    last = index[-1]
    if last > decision:
        raise RiskModelError(
            f"the return panel ends at {last} against as_of={as_of}; a snapshot attributed to a decision "
            "bar may read that bar and no later one"
        )


def _substitution_reasons(
    sids: Sequence[str],
    returns: pd.DataFrame,
    policy: RiskPolicy,
    *,
    eligible: Mapping[str, bool] | pd.Series | None,
    valuation_status: Mapping[str, str] | None,
    halted: Iterable[str] | None,
) -> dict[str, str]:
    """Why each name cannot be estimated, in the order the screens are declared."""
    reasons: dict[str, str] = {}
    columns = {str(column) for column in returns.columns}
    halted_set = {str(sid) for sid in (halted or ())}
    status = {str(sid): str(value) for sid, value in dict(valuation_status or {}).items()}
    excluded_status = set(policy.substitution.exclude_valuation_status)
    eligible_map = {str(sid): bool(value) for sid, value in dict(eligible).items()} if eligible is not None else None
    covariance_window = returns.tail(policy.covariance_window_bars)
    for sid in sids:
        if sid not in columns:
            reasons[sid] = "absent_from_return_panel"
            continue
        if policy.substitution.exclude_halted and sid in halted_set:
            reasons[sid] = "halted"
            continue
        if status.get(sid) in excluded_status:
            reasons[sid] = f"valuation_status:{status[sid]}"
            continue
        if eligible_map is not None and not eligible_map.get(sid, False):
            reasons[sid] = "ineligible"
            continue
        column = covariance_window[sid].to_numpy(dtype=float)
        if not np.isfinite(column).all():
            reasons[sid] = "incomplete_return_window"
    return reasons


def build_snapshot(
    returns: pd.DataFrame,
    policy: RiskPolicy,
    *,
    as_of: str,
    universe: Sequence[str],
    held: Sequence[str] = (),
    factor_returns: pd.DataFrame | None = None,
    eligible: Mapping[str, bool] | pd.Series | None = None,
    valuation_status: Mapping[str, str] | None = None,
    halted: Iterable[str] | None = None,
) -> RiskSnapshot:
    """Estimate one snapshot from a trailing return panel, point in time (section 3.5).

    ``returns`` is a ``(bars x sids)`` frame of simple per-bar returns whose last row is the
    decision bar: everything read here ends at that bar, so the snapshot attributed to it is
    computable at its close. The covariance window is its last ``covariance_window_bars`` rows
    and the loadings window the last ``loadings_window_bars`` of those.

    ``omega`` is the sample covariance of the factor return window, shrunk toward the declared
    target; ``d`` is the residual variance per name from the same regression, shrunk toward the
    cross-sectional median, scaled to the horizon, and clipped up at the declared floor. Names
    that cannot be estimated -- absent, halted, stale, ineligible, or short of a finite window
    -- take the declared substitution and are pinned reduce-only.

    The snapshot's ``sids`` are the constructor universe followed by any held name not in it,
    so coverage is a property of construction and :func:`validate_snapshot` checks it rather
    than repairs it.
    """
    if not isinstance(returns, pd.DataFrame):
        raise RiskModelError(f"returns must be a DataFrame of per-bar returns, got {type(returns).__name__}")
    ordered: list[str] = []
    for sid in list(universe) + list(held):
        text = str(sid)
        if text not in ordered:
            ordered.append(text)
    if not ordered:
        raise RiskModelError("build_snapshot needs a universe; an empty snapshot covers nothing")
    if len(returns) < policy.covariance_window_bars:
        raise RiskModelError(
            f"the return panel carries {len(returns)} bars against covariance_window_bars="
            f"{policy.covariance_window_bars}; a short window is a smaller estimate, never a silent one"
        )
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise RiskModelError(
            "the return panel's index must be strictly increasing: the covariance window is its last "
            "covariance_window_bars rows, so an unsorted index silently estimates on the wrong bars"
        )
    _refuse_look_ahead(returns.index, as_of=as_of)
    panel = returns.copy()
    panel.columns = [str(column) for column in panel.columns]
    if panel.columns.duplicated().any():
        offenders = sorted({str(column) for column in panel.columns[panel.columns.duplicated()]})
        raise RiskModelError(
            f"the return panel carries duplicated columns {offenders[:5]}; a name's window must be one column"
        )

    reasons = _substitution_reasons(
        ordered, panel, policy, eligible=eligible, valuation_status=valuation_status, halted=halted
    )
    estimable = [sid for sid in ordered if sid not in reasons]
    if len(estimable) <= policy.n_factors:
        raise RiskModelError(
            f"{len(estimable)} name(s) are estimable against n_factors={policy.n_factors}; a factor model "
            "with at least as many factors as names is not identified, and substituting the whole book "
            "would hand the constructor a risk surface nobody estimated"
        )

    covariance_window = panel.tail(policy.covariance_window_bars)
    estimation_window = covariance_window[estimable].to_numpy(dtype=float)
    horizon = policy.horizon_bars

    explained: list[float] | None = None
    if policy.factor_mode == "pca":
        weights, explained_shares = estimate_eigenportfolios(estimation_window, policy.n_factors)
        factor_window = factor_returns_from_window(estimation_window, weights)
        explained = [float(value) for value in explained_shares]
        factor_ids = [f"pca_{index + 1}" for index in range(policy.n_factors)]
    else:
        if factor_returns is None:
            raise RiskModelError("factor_mode = 'etf' needs the exogenous factor return panel; none was supplied")
        exogenous = factor_returns.tail(policy.covariance_window_bars)
        if len(exogenous) != policy.covariance_window_bars:
            raise RiskModelError(
                f"the factor return panel carries {len(exogenous)} bars against covariance_window_bars="
                f"{policy.covariance_window_bars}"
            )
        if not exogenous.index.equals(covariance_window.index):
            raise RiskModelError("the factor return panel and the return panel must share their bar index")
        if exogenous.shape[1] != policy.n_factors:
            raise RiskModelError(
                f"the factor return panel carries {exogenous.shape[1]} factors against n_factors="
                f"{policy.n_factors}; a factor the policy did not declare is not estimated silently"
            )
        factor_window = exogenous.to_numpy(dtype=float)
        if not np.isfinite(factor_window).all():
            raise RiskModelError("the factor return panel must be fully finite over the covariance window")
        factor_ids = [str(column) for column in exogenous.columns]

    # -- loadings and residuals, on the shorter regression window ------------------
    loadings_rows = policy.loadings_window_bars
    _alpha, beta, residuals = batched_factor_ols(estimation_window[-loadings_rows:], factor_window[-loadings_rows:])
    residual_df = loadings_rows - policy.n_factors - 1
    residual_variance_per_bar = np.einsum("ij,ij->j", residuals, residuals) / residual_df

    # -- the factor covariance, at the declared horizon ---------------------------
    scaling = policy.scaling
    fallback: dict[str, Any] | None = None
    if scaling == "estimated_at_horizon":
        if policy.covariance_window_bars / horizon > policy.n_factors:
            horizon_window = _overlapping_sums(factor_window, horizon)
            sample = np.atleast_2d(np.cov(horizon_window, rowvar=False, ddof=1))
            horizon_multiplier = 1.0
            overlapping_obs = int(horizon_window.shape[0])
        else:
            fallback = {
                "declared": "estimated_at_horizon",
                "applied": "per_bar_times_h",
                "reason": (
                    f"covariance_window_bars / horizon_bars = {policy.covariance_window_bars / horizon:.4g} "
                    f"is not greater than n_factors = {policy.n_factors}"
                ),
            }
            scaling = "per_bar_times_h"
            sample = np.atleast_2d(np.cov(factor_window, rowvar=False, ddof=1))
            horizon_multiplier = float(horizon)
            overlapping_obs = 0
    else:
        sample = np.atleast_2d(np.cov(factor_window, rowvar=False, ddof=1))
        horizon_multiplier = float(horizon)
        overlapping_obs = 0
    omega = _shrunk_covariance(sample, intensity=policy.omega_shrink, target=policy.omega_shrink_target)
    omega = omega * horizon_multiplier

    # -- the residual, shrunk toward the cross-sectional median then floored -------
    median = float(np.median(residual_variance_per_bar))
    lambda_d = policy.residual_shrink
    shrunk_residual = (1.0 - lambda_d) * residual_variance_per_bar + lambda_d * median
    # The residual regression is on per-bar returns under either scaling, so the residual is
    # always carried to the horizon by per_bar_times_h; the record states it.
    residual_at_horizon = shrunk_residual * horizon

    floor = policy.residual_floor_variance
    n_names = len(ordered)
    loadings = np.zeros((n_names, policy.n_factors), dtype=float)
    residual = np.full(n_names, floor, dtype=float)
    position = {sid: index for index, sid in enumerate(ordered)}
    clipped: list[str] = []
    for column, sid in enumerate(estimable):
        row = position[sid]
        loadings[row] = beta[:, column]
        value = float(residual_at_horizon[column])
        if value < floor:
            clipped.append(sid)
            value = floor
        residual[row] = value

    substitutions = {
        sid: {
            "reason": reason,
            "loadings": "zero",
            "residual_variance": floor,
            "restriction": "reduce_only",
        }
        for sid, reason in reasons.items()
    }
    method: dict[str, Any] = {
        "kind": policy.kind,
        "factor_mode": policy.factor_mode,
        "factor_ids": factor_ids,
        "n_factors": policy.n_factors,
        "loadings_window_bars": policy.loadings_window_bars,
        "covariance_window_bars": policy.covariance_window_bars,
        "omega_shrink": policy.omega_shrink,
        "omega_shrink_target": policy.omega_shrink_target,
        "eigenvalue_floor_relative": policy.eigenvalue_floor_relative,
        "eigenvalue_reject_relative": policy.eigenvalue_reject_relative,
        "symmetry_tolerance": policy.symmetry_tolerance,
        "residual_shrink": policy.residual_shrink,
        "residual_shrink_target": "cross_sectional_median",
        "residual_median_per_bar": median,
        "residual_floor_var_annual": policy.residual_floor_var_annual,
        "residual_floor_variance": floor,
        "residual_scaling": "per_bar_times_h",
        "annualization_bars": policy.annualization_bars,
        "refresh_every_bars": policy.refresh_every_bars,
        "scaling": scaling,
        "scaling_declared": policy.scaling,
        "scaling_fallback": fallback,
        "overlapping_observations": overlapping_obs,
        "horizon_bars": horizon,
        "policy_id": policy.policy_id,
        "policy_source": policy.source,
        "policy_sha256": policy.source_sha256,
        "as_of": str(as_of),
        "estimated": list(estimable),
        "n_estimated": len(estimable),
        "explained_share": explained,
        "substitutions": substitutions,
        "reduce_only": sorted(substitutions),
        "residual_floor_clipped": {
            "count": len(clipped),
            "fraction": len(clipped) / len(estimable) if estimable else 0.0,
            "sids": sorted(clipped),
        },
    }
    return RiskSnapshot(
        sids=tuple(ordered),
        B=loadings,
        omega=omega,
        d=residual,
        horizon_bars=horizon,
        scaling=scaling,  # type: ignore[arg-type]
        as_of=str(as_of),
        method=method,
    )
