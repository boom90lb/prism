"""The factor risk model (spec section 3.5, decision D9).

The failures these tests exist to catch, in the order they would cost the most:

1. a diagonal-only covariance that under-prices a correlated book by nearly half;
2. a Cholesky factorization that raises on the PSD-singular covariance a PCA factor model
   routinely produces, taking the whole decision down with it;
3. a held name with no current estimate priced as riskless, so a book of stale names
   satisfies any variance budget trivially;
4. a snapshot that does not cover the constructor universe reaching a solve at all.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.portfolio.limits import (
    Limit,
    LimitClass,
    LimitContext,
    LimitError,
    RiskLike,
    violation,
)
from prism.residual.factors import batched_factor_ols
from prism.risk_model import (
    ConeOperator,
    RiskModelError,
    RiskPolicy,
    RiskSnapshot,
    build_snapshot,
    validate_snapshot,
)

REFERENCE_POLICY = Path(__file__).resolve().parents[1] / "examples" / "reference" / "risk" / "factor_diagonal.toml"


def _policy(**overrides: object) -> RiskPolicy:
    """The shipped reference policy, with the overrides a test needs declared explicitly."""
    declared = RiskPolicy.from_toml(REFERENCE_POLICY).declaration()
    declared.update(overrides)
    return RiskPolicy.from_mapping(declared)


def _snapshot(
    *,
    sids: list[str],
    B: np.ndarray,
    omega: np.ndarray,
    d: np.ndarray,
    horizon_bars: int = 21,
    method: dict | None = None,
) -> RiskSnapshot:
    return RiskSnapshot(
        sids=tuple(sids),
        B=np.asarray(B, dtype=float),
        omega=np.asarray(omega, dtype=float),
        d=np.asarray(d, dtype=float),
        horizon_bars=horizon_bars,
        scaling="per_bar_times_h",
        as_of="2026-05-01",
        method={"eigenvalue_floor_relative": 1.0e-10, **(method or {})},
    )


def _as_of(panel: pd.DataFrame) -> str:
    """The decision bar of a panel: its last row, the only bar a snapshot may be attributed to."""
    return str(panel.index[-1].date())


def _returns_panel(
    *, n_names: int = 40, bars: int = 300, n_factors: int = 3, seed: int = 11
) -> tuple[pd.DataFrame, list[str]]:
    """A synthetic factor-plus-noise return panel; the estimator is exercised, not evaluated."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, 0.01, size=(bars, n_factors))
    loadings = rng.normal(0.0, 1.0, size=(n_factors, n_names))
    noise = rng.normal(0.0, 0.012, size=(bars, n_names))
    sids = [f"S{index:03d}" for index in range(n_names)]
    panel = pd.DataFrame(factor @ loadings + noise, index=pd.bdate_range("2024-01-02", periods=bars), columns=sids)
    return panel, sids


# --------------------------------------------------------------------------
# The full covariance
# --------------------------------------------------------------------------


def test_the_full_factor_covariance_is_not_a_diagonal_one() -> None:
    """The correctness check section 3.5 names: 0.95, where a diagonal reports 0.50.

    Exposure one half to each of two unit-variance factors correlated at 0.9 carries
    ``0.25 + 0.25 + 2 * 0.25 * 0.9 = 0.95``. Dropping the off-diagonal reports ``0.50`` and
    under-prices the book by 47 percent -- the difference between a variance budget that binds
    and one that does not.
    """
    correlated = _snapshot(
        sids=["AAA"],
        B=[[0.5, 0.5]],
        omega=[[1.0, 0.9], [0.9, 1.0]],
        d=[0.0],
    )
    diagonal_only = _snapshot(
        sids=["AAA"],
        B=[[0.5, 0.5]],
        omega=[[1.0, 0.0], [0.0, 1.0]],
        d=[0.0],
    )
    w = pd.Series({"AAA": 1.0})
    assert correlated.variance(w) == pytest.approx(0.95, abs=1e-12)
    assert diagonal_only.variance(w) == pytest.approx(0.50, abs=1e-12)

    # The residual condition is part of the check: any positive d adds sum_i d_i w_i^2.
    with_residual = _snapshot(sids=["AAA"], B=[[0.5, 0.5]], omega=[[1.0, 0.9], [0.9, 1.0]], d=[0.04])
    assert with_residual.variance(w) == pytest.approx(0.99, abs=1e-12)


def test_variance_matches_the_dense_quadratic_form_it_stands_for() -> None:
    """``w' (B omega B' + diag(d)) w``, checked against the dense matrix on a small case."""
    rng = np.random.default_rng(3)
    n, k = 12, 4
    B = rng.normal(size=(n, k))
    root = rng.normal(size=(k, k))
    omega = root @ root.T
    d = rng.uniform(0.001, 0.02, size=n)
    sids = [f"N{i}" for i in range(n)]
    snapshot = _snapshot(sids=sids, B=B, omega=omega, d=d)
    w = pd.Series(rng.normal(size=n) * 0.05, index=sids)

    dense = B @ omega @ B.T + np.diag(d)
    expected = float(w.to_numpy() @ dense @ w.to_numpy())
    assert snapshot.variance(w) == pytest.approx(expected, rel=1e-12)
    # Order of the Series must not matter: sids fixes the ordering, not the caller.
    assert snapshot.variance(w.sample(frac=1.0, random_state=0)) == pytest.approx(expected, rel=1e-12)
    # A name the caller does not mention carries zero weight, not a dropped row.
    assert snapshot.variance(w.iloc[:5]) == pytest.approx(
        float(
            w.iloc[:5].reindex(sids).fillna(0.0).to_numpy() @ dense @ w.iloc[:5].reindex(sids).fillna(0.0).to_numpy()
        ),
        rel=1e-12,
    )


def test_variance_never_forms_a_dense_covariance() -> None:
    """100,000 names against 3 factors: dense would be 80 GB, so forming it fails the test.

    This is the specification's "never forming a dense N x N matrix" as an executable claim
    rather than a comment. An implementation that materializes ``B omega B' + diag(d)`` raises
    ``MemoryError`` (or takes the machine down) here; this one runs in milliseconds.
    """
    n = 100_000
    B = np.ones((n, 3))
    B[:, 1] = 0.5
    B[:, 2] = -0.25
    omega = np.array([[4.0e-4, 1.0e-4, 0.0], [1.0e-4, 2.0e-4, 0.0], [0.0, 0.0, 1.0e-4]])
    d = np.full(n, 1.0e-4)
    snapshot = _snapshot(sids=[f"X{i}" for i in range(n)], B=B, omega=omega, d=d)

    w = np.full(n, 1.0 / n)
    exposure = B.T @ w
    expected = float(exposure @ omega @ exposure + float(np.dot(d, w * w)))
    assert snapshot.variance(w) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# The cone operator
# --------------------------------------------------------------------------


def test_the_cone_operator_reproduces_the_quadratic_form_it_stands_for() -> None:
    """``||F' w||^2 + ||sqrt(d) * w||^2 == w' Sigma w`` for the operator handed to the solver."""
    rng = np.random.default_rng(17)
    n, k = 30, 5
    B = rng.normal(size=(n, k))
    root = rng.normal(size=(k, k))
    omega = root @ root.T
    d = rng.uniform(0.0, 0.03, size=n)
    sids = [f"N{i:02d}" for i in range(n)]
    snapshot = _snapshot(sids=sids, B=B, omega=omega, d=d)

    operator = snapshot.cone_operator()
    assert isinstance(operator, ConeOperator)
    factor, residual = operator  # it unpacks as the two-tuple section 3.5 writes
    assert factor.shape == (n, k)
    assert residual.shape == (n,)

    for _ in range(5):
        w = rng.normal(size=n) * 0.03
        assert operator.quadratic_form(w) == pytest.approx(snapshot.variance(w), rel=1e-10, abs=1e-18)
        # The solver's own expression, written the way convex.py stacks it.
        stacked = np.concatenate([factor.T @ w, residual * w])
        assert float(stacked @ stacked) == pytest.approx(snapshot.variance(w), rel=1e-10, abs=1e-18)


def test_a_singular_factor_covariance_factorizes_rather_than_raising() -> None:
    """Decision D9, on the specification's own input: ``diag([0.04, 0.0, 0.01])``.

    Cholesky raises on it -- asserted here, so the reason the code takes the eigendecomposition
    path is part of the test rather than a claim in a docstring -- and ``eigh`` with the clipped
    floor reproduces it to within 4e-12.
    """
    omega = np.diag([0.04, 0.0, 0.01])
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(omega)

    snapshot = _snapshot(sids=["A", "B", "C"], B=np.eye(3), omega=omega, d=np.zeros(3))
    operator = snapshot.cone_operator()
    reconstructed = operator.factor @ operator.factor.T
    # The whole discrepancy IS the declared floor, 1e-10 * 0.04 = 4e-12, sitting in the
    # direction whose eigenvalue was zero; nothing else moved.
    assert np.abs(reconstructed - omega).max() <= 4.0e-12
    assert reconstructed[1, 1] == pytest.approx(4.0e-12, rel=1e-9)

    clip = snapshot.method["eigenvalue_clip"]
    assert clip["binds"] is True
    assert clip["clipped"] == 1
    assert clip["floor_relative"] == 1.0e-10
    assert clip["floor_absolute"] == pytest.approx(1.0e-10 * 0.04)
    # The clipped direction carries the floor, not zero, and not a negative under sqrt.
    assert np.isfinite(operator.factor).all()
    w = np.array([1.0, 1.0, 1.0])
    assert operator.quadratic_form(w) == pytest.approx(0.05, rel=1e-9)


def test_the_cone_operator_never_calls_cholesky(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regression guard on D9: the factorization route is the decision, not an implementation detail."""

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("cone_operator must factorize by eigendecomposition, never by Cholesky (D9)")

    monkeypatch.setattr(np.linalg, "cholesky", _forbidden)
    snapshot = _snapshot(sids=["A", "B", "C"], B=np.eye(3), omega=np.diag([0.04, 0.0, 0.01]), d=np.zeros(3))
    assert snapshot.cone_operator().factor.shape == (3, 3)


def test_the_eigenvalue_floor_is_declared_never_defaulted() -> None:
    """A snapshot carrying no declared floor refuses to invent one."""
    snapshot = RiskSnapshot(
        sids=("A",),
        B=np.array([[1.0]]),
        omega=np.array([[0.04]]),
        d=np.array([0.0]),
        horizon_bars=21,
        scaling="per_bar_times_h",
        as_of="2026-05-01",
        method={},
    )
    with pytest.raises(RiskModelError, match="declares no eigenvalue_floor_relative"):
        snapshot.cone_operator()
    assert snapshot.cone_operator(eigenvalue_floor_relative=1.0e-10).factor.shape == (1, 1)


def test_the_cone_operator_refuses_a_snapshot_validation_would_have_rejected() -> None:
    """The operator is not the place a bad snapshot is discovered, but it never proceeds on one."""
    negative = _snapshot(sids=["A"], B=[[1.0]], omega=[[0.04]], d=[-1.0e-6])
    with pytest.raises(RiskModelError, match="validate_snapshot rejects it before any solve"):
        negative.cone_operator()

    dead = _snapshot(sids=["A"], B=[[1.0]], omega=[[0.0]], d=[0.01])
    with pytest.raises(RiskModelError, match="largest eigenvalue"):
        dead.cone_operator()


# --------------------------------------------------------------------------
# Validation, before any solve
# --------------------------------------------------------------------------


def test_validation_rejects_a_snapshot_that_does_not_cover_the_universe() -> None:
    """``sids`` fixes the ordering of ``w``; a missing name has no defined risk (D9)."""
    snapshot = _snapshot(sids=["A", "B"], B=np.ones((2, 1)), omega=[[0.04]], d=[0.01, 0.01])
    accepted = validate_snapshot(snapshot, universe=["A", "B"], policy=None)
    assert accepted.accepted is True
    assert accepted.outcome == "accepted"

    rejected = validate_snapshot(snapshot, universe=["A", "B", "C"], policy=None)
    assert rejected.accepted is False
    assert rejected.outcome == "validation rejects, no orders"
    assert "C" in rejected.findings[0]
    assert rejected.measures["uncovered"] == 1
    with pytest.raises(RiskModelError, match="validation rejects, no orders"):
        rejected.require()


def test_validation_rejects_a_non_finite_or_negative_variance_snapshot() -> None:
    base = dict(sids=["A", "B"], B=np.ones((2, 1)), omega=[[0.04]], d=[0.01, 0.01])

    non_finite_b = _snapshot(**{**base, "B": np.array([[1.0], [np.nan]])})
    report = validate_snapshot(non_finite_b, universe=["A", "B"])
    assert not report.accepted and "B carries 1 non-finite" in report.findings[0]

    non_finite_omega = _snapshot(**{**base, "omega": [[np.inf]]})
    report = validate_snapshot(non_finite_omega, universe=["A", "B"])
    assert not report.accepted and any("omega carries" in finding for finding in report.findings)

    non_finite_d = _snapshot(**{**base, "d": [0.01, np.nan]})
    report = validate_snapshot(non_finite_d, universe=["A", "B"])
    assert not report.accepted and any("d carries" in finding for finding in report.findings)

    negative_d = _snapshot(**{**base, "d": [0.01, -1.0e-9]})
    report = validate_snapshot(negative_d, universe=["A", "B"])
    assert not report.accepted and any("that is not a variance" in finding for finding in report.findings)
    assert report.measures["min_residual_variance"] == pytest.approx(-1.0e-9)


def test_validation_rejects_an_asymmetric_or_indefinite_covariance() -> None:
    policy = _policy(n_factors=2)
    sids = ["A", "B"]
    B = np.ones((2, 2))
    d = np.array([0.01, 0.01])

    asymmetric = _snapshot(sids=sids, B=B, omega=[[0.04, 0.01], [0.02, 0.04]], d=d)
    report = validate_snapshot(asymmetric, universe=sids, policy=policy)
    assert not report.accepted and any("asymmetry" in finding for finding in report.findings)

    indefinite = _snapshot(sids=sids, B=B, omega=[[0.04, 0.10], [0.10, 0.04]], d=d)
    report = validate_snapshot(indefinite, universe=sids, policy=policy)
    assert not report.accepted and any("indefinite covariance" in finding for finding in report.findings)

    # PSD-singular is admitted: it is the normal case the floor exists for.
    singular = _snapshot(sids=sids, B=B, omega=[[0.04, 0.0], [0.0, 0.0]], d=d)
    assert validate_snapshot(singular, universe=sids, policy=policy).accepted is True

    dead = _snapshot(sids=sids, B=B, omega=np.zeros((2, 2)), d=d)
    report = validate_snapshot(dead, universe=sids, policy=policy)
    assert not report.accepted and any("not positive" in finding for finding in report.findings)


def test_validation_rejects_a_snapshot_stated_at_the_wrong_horizon() -> None:
    """The variance budget bounds risk at the declared horizon; a 5-bar snapshot is not it."""
    policy = _policy()
    snapshot = _snapshot(sids=["A"], B=[[1.0]], omega=[[0.04]], d=[0.01], horizon_bars=5)
    report = validate_snapshot(snapshot, universe=["A"], policy=policy)
    assert not report.accepted and any("stated over 5 bars" in finding for finding in report.findings)


def test_validation_reports_rather_than_raising() -> None:
    """A rejection is a recorded acceptance outcome, not an exception (section 3.5)."""
    snapshot = _snapshot(sids=["A"], B=[[1.0]], omega=[[0.04]], d=[np.nan])
    report = validate_snapshot(snapshot, universe=["A", "Z"])
    assert report.to_dict()["outcome"] == "validation rejects, no orders"
    assert len(report.findings) == 2  # coverage and finiteness, both reported, neither raised


# --------------------------------------------------------------------------
# The declared policy file
# --------------------------------------------------------------------------


def test_the_reference_policy_file_parses_at_its_declared_values() -> None:
    policy = RiskPolicy.from_toml(REFERENCE_POLICY)
    assert policy.kind == "factor_diagonal"
    assert policy.factor_mode == "pca"
    assert (policy.n_factors, policy.loadings_window_bars, policy.covariance_window_bars) == (15, 60, 252)
    assert (policy.omega_shrink, policy.omega_shrink_target) == (0.10, "diagonal")
    assert (policy.eigenvalue_floor_relative, policy.eigenvalue_reject_relative) == (1.0e-10, 1.0e-08)
    assert (policy.residual_shrink, policy.residual_floor_var_annual) == (0.20, 0.09)
    assert (policy.scaling, policy.horizon_bars, policy.annualization_bars) == ("per_bar_times_h", 21, 252)
    assert policy.refresh_every_bars == 5
    assert policy.substitution.exclude_valuation_status == ("stale", "unresolved_claim")
    assert policy.substitution.exclude_halted is True
    # 0.09 / 252 * 21 = 0.0075, an 8.66 percent standard deviation over 21 bars.
    assert policy.residual_floor_variance == pytest.approx(0.0075)
    assert math.sqrt(policy.residual_floor_variance) == pytest.approx(0.0866, abs=5e-5)
    assert policy.source_sha256 and policy.policy_id


def test_the_policy_refuses_an_omitted_key_and_an_unknown_one() -> None:
    declared = RiskPolicy.from_toml(REFERENCE_POLICY).declaration()

    for key in ("residual_shrink", "eigenvalue_floor_relative", "scaling"):
        without = {name: value for name, value in declared.items() if name != key}
        with pytest.raises(RiskModelError, match="omits"):
            RiskPolicy.from_mapping(without)

    with pytest.raises(RiskModelError, match="unknown keys"):
        RiskPolicy.from_mapping({**declared, "residual_shrinkage": 0.2})

    with pytest.raises(RiskModelError, match="must be a shrinkage intensity"):
        RiskPolicy.from_mapping({**declared, "omega_shrink": 1.5})

    with pytest.raises(RiskModelError, match="exceeds covariance_window_bars"):
        RiskPolicy.from_mapping({**declared, "loadings_window_bars": 300})

    with pytest.raises(RiskModelError, match="no residual degrees of freedom"):
        RiskPolicy.from_mapping({**declared, "loadings_window_bars": 16, "n_factors": 15})

    # Two files declaring the same content share a policy_id; one different value does not.
    assert RiskPolicy.from_mapping(declared).policy_id == RiskPolicy.from_mapping(dict(declared)).policy_id
    assert RiskPolicy.from_mapping({**declared, "residual_shrink": 0.25}).policy_id != (
        RiskPolicy.from_mapping(declared).policy_id
    )


# --------------------------------------------------------------------------
# Estimation and the reduce-only substitution
# --------------------------------------------------------------------------


def test_a_held_name_without_an_estimate_takes_the_declared_substitution() -> None:
    """D9: zero loadings, the declared floor, a reduce-only pin, recorded per name with its reason.

    These are precisely the names ``residual/factors.py`` refuses -- absent from the panel,
    short of a finite window, halted, or carried at a stale valuation -- so the path is live.
    """
    panel, sids = _returns_panel()
    panel["GAPPY"] = np.nan
    panel["HALTED"] = panel["S000"].to_numpy()
    panel["STALE"] = panel["S001"].to_numpy()
    policy = _policy(n_factors=3)

    snapshot = build_snapshot(
        panel,
        policy,
        as_of=_as_of(panel),
        universe=sids,
        held=["GAPPY", "HALTED", "STALE", "DELISTED"],
        halted=["HALTED"],
        valuation_status={"STALE": "stale"},
    )

    assert set(snapshot.reduce_only) == {"GAPPY", "HALTED", "STALE", "DELISTED"}
    reasons = {sid: entry["reason"] for sid, entry in snapshot.substitutions.items()}
    assert reasons == {
        "GAPPY": "incomplete_return_window",
        "HALTED": "halted",
        "STALE": "valuation_status:stale",
        "DELISTED": "absent_from_return_panel",
    }
    floor = policy.residual_floor_variance
    for sid in snapshot.reduce_only:
        row = snapshot.sids.index(sid)
        assert np.all(snapshot.B[row] == 0.0)
        assert snapshot.d[row] == pytest.approx(floor)
        assert snapshot.substitutions[sid]["restriction"] == "reduce_only"
        assert snapshot.substitutions[sid]["residual_variance"] == pytest.approx(floor)

    # A substituted name is not riskless: it carries the floor, and the floor is conservative.
    assert snapshot.variance(pd.Series({"DELISTED": 1.0})) == pytest.approx(floor)
    assert validate_snapshot(snapshot, universe=sids, policy=policy).accepted is True
    # And the snapshot covers the held names too, so a book holding them is measurable.
    assert validate_snapshot(snapshot, universe=[*sids, "DELISTED"], policy=policy).accepted is True


def test_a_stale_holding_would_be_estimated_as_riskless_which_is_why_it_is_substituted() -> None:
    """The specification's argument, executed: a constant column regresses to nothing.

    ``batched_factor_ols`` returns a zero loading vector and zero residual variance for a
    constant column, so a book of stale names would satisfy any variance budget trivially.
    Two defences follow, and both are checked: ``[substitution]`` keeps the name out of
    estimation, and the residual floor catches an unflagged one.
    """
    rng = np.random.default_rng(5)
    window = rng.normal(0.0, 0.01, size=(60, 4))
    window[:, 3] = 0.0  # a holding carried at its last valuation
    factors = rng.normal(0.0, 0.01, size=(60, 3))
    _alpha, beta, residuals = batched_factor_ols(window, factors)
    assert np.allclose(beta[:, 3], 0.0)
    assert float((residuals[:, 3] ** 2).sum()) == pytest.approx(0.0, abs=1e-30)

    panel, sids = _returns_panel(n_names=20)
    panel["FROZEN"] = 0.0
    policy = _policy(n_factors=3)

    flagged = build_snapshot(
        panel,
        policy,
        as_of=_as_of(panel),
        universe=sids,
        held=["FROZEN"],
        valuation_status={"FROZEN": "stale"},
    )
    assert flagged.substitutions["FROZEN"]["reason"] == "valuation_status:stale"

    # Unflagged, the name is estimated -- and the residual floor is what stops it pricing as
    # riskless. Its loadings are zero either way, so the factor block adds nothing.
    unflagged = build_snapshot(panel, policy, as_of=_as_of(panel), universe=sids, held=["FROZEN"])
    row = unflagged.sids.index("FROZEN")
    assert "FROZEN" not in unflagged.substitutions
    assert np.allclose(unflagged.B[row], 0.0)
    assert unflagged.d[row] == pytest.approx(policy.residual_floor_variance)
    assert unflagged.variance(pd.Series({"FROZEN": 1.0})) > 0.0
    assert "FROZEN" in unflagged.method["residual_floor_clipped"]["sids"]


def test_a_rank_deficient_factor_covariance_is_reachable_end_to_end() -> None:
    """Section 3.5's claim, produced by the estimators rather than asserted.

    ``estimate_eigenportfolios`` gives a zero-variance column zero weight in every
    eigenportfolio (``factors.py:341-346``), so two constant columns and ``n_factors`` deep
    enough to reach their eigenvectors leave a factor with identically zero returns and an
    ``omega`` with a zero eigenvalue. Validation admits it; the cone operator factorizes it.
    """
    rng = np.random.default_rng(23)
    bars = 80
    common = rng.normal(0.0, 0.01, size=(bars, 2))
    live = common @ rng.normal(size=(2, 4)) + rng.normal(0.0, 0.005, size=(bars, 4))
    panel = pd.DataFrame(
        np.column_stack([live, np.zeros((bars, 2))]),
        index=pd.bdate_range("2025-01-01", periods=bars),
        columns=["A", "B", "C", "D", "FLAT1", "FLAT2"],
    )
    policy = _policy(n_factors=5, loadings_window_bars=30, covariance_window_bars=60)
    snapshot = build_snapshot(panel, policy, as_of=_as_of(panel), universe=list(panel.columns))

    eigenvalues = np.linalg.eigvalsh(snapshot.omega)
    assert eigenvalues.min() == pytest.approx(0.0, abs=1e-18)
    assert eigenvalues.max() > 0.0
    assert validate_snapshot(snapshot, universe=list(panel.columns), policy=policy).accepted is True

    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(snapshot.omega)
    operator = snapshot.cone_operator()
    assert np.isfinite(operator.factor).all()
    assert snapshot.method["eigenvalue_clip"]["binds"] is True
    w = np.full(len(snapshot.sids), 0.1)
    assert operator.quadratic_form(w) == pytest.approx(snapshot.variance(w), rel=1e-9)


def test_the_estimated_snapshot_is_a_full_covariance_at_the_declared_horizon() -> None:
    panel, sids = _returns_panel(n_names=30, bars=400)
    policy = _policy(n_factors=4)
    snapshot = build_snapshot(panel, policy, as_of=_as_of(panel), universe=sids)

    assert snapshot.B.shape == (30, 4)
    assert snapshot.omega.shape == (4, 4)
    assert snapshot.horizon_bars == 21 and snapshot.scaling == "per_bar_times_h"
    # Full, not diagonal: the loadings are estimated on 60 bars and the correlation structure
    # on 252, so the factor returns over the regression window are not uncorrelated.
    off_diagonal = snapshot.omega - np.diag(np.diag(snapshot.omega))
    assert np.abs(off_diagonal).max() > 0.0
    assert np.abs(snapshot.omega - snapshot.omega.T).max() == 0.0
    assert (snapshot.d > 0.0).all()
    assert snapshot.method["scaling_fallback"] is None
    assert snapshot.method["residual_shrink_target"] == "cross_sectional_median"
    assert len(snapshot.method["factor_ids"]) == 4
    assert snapshot.factor_ids == ("pca_1", "pca_2", "pca_3", "pca_4")

    # Determinism: the same panel and policy give the same snapshot, to the bit.
    again = build_snapshot(panel, policy, as_of=_as_of(panel), universe=sids)
    assert np.array_equal(snapshot.B, again.B)
    assert np.array_equal(snapshot.omega, again.omega)
    assert np.array_equal(snapshot.d, again.d)


def test_the_declared_shrinkages_move_the_estimate_in_the_declared_direction() -> None:
    panel, sids = _returns_panel(n_names=30, bars=400)
    unshrunk = build_snapshot(
        panel, _policy(n_factors=4, omega_shrink=0.0, residual_shrink=0.0), as_of=_as_of(panel), universe=sids
    )
    shrunk = build_snapshot(
        panel, _policy(n_factors=4, omega_shrink=0.5, residual_shrink=0.5), as_of=_as_of(panel), universe=sids
    )

    # omega_shrink toward the diagonal target halves the off-diagonal and leaves the diagonal.
    off_unshrunk = unshrunk.omega - np.diag(np.diag(unshrunk.omega))
    off_shrunk = shrunk.omega - np.diag(np.diag(shrunk.omega))
    assert np.allclose(off_shrunk, 0.5 * off_unshrunk)
    assert np.allclose(np.diag(shrunk.omega), np.diag(unshrunk.omega))

    # residual_shrink pulls each name halfway to the cross-sectional median, before the floor.
    floor = shrunk.method["residual_floor_variance"]
    median = shrunk.method["residual_median_per_bar"] * shrunk.horizon_bars
    for row in range(len(sids)):
        expected = max(floor, 0.5 * unshrunk.d[row] + 0.5 * median)
        assert shrunk.d[row] == pytest.approx(expected, rel=1e-12)


def test_the_horizon_scaling_falls_back_and_records_the_fallback() -> None:
    """``estimated_at_horizon`` below its declared guard falls back and says so."""
    panel, sids = _returns_panel(n_names=30, bars=400)
    # 252 / 21 = 12 is not greater than n_factors = 12.
    fell_back = build_snapshot(
        panel,
        _policy(n_factors=12, scaling="estimated_at_horizon"),
        as_of=_as_of(panel),
        universe=sids,
    )
    assert fell_back.scaling == "per_bar_times_h"
    fallback = fell_back.method["scaling_fallback"]
    assert fallback["declared"] == "estimated_at_horizon" and fallback["applied"] == "per_bar_times_h"
    assert (
        validate_snapshot(
            fell_back, universe=sids, policy=_policy(n_factors=12, scaling="estimated_at_horizon")
        ).accepted
        is True
    )

    # Above the guard the covariance is estimated on overlapping horizon returns instead.
    at_horizon = build_snapshot(
        panel,
        _policy(n_factors=4, scaling="estimated_at_horizon"),
        as_of=_as_of(panel),
        universe=sids,
    )
    assert at_horizon.scaling == "estimated_at_horizon"
    assert at_horizon.method["scaling_fallback"] is None
    assert at_horizon.method["overlapping_observations"] == 252 - 21 + 1


def test_estimation_refuses_what_it_cannot_estimate() -> None:
    panel, sids = _returns_panel(n_names=20, bars=400)
    policy = _policy(n_factors=3)

    with pytest.raises(RiskModelError, match="carries 400 bars against covariance_window_bars"):
        build_snapshot(
            panel.tail(400), _policy(n_factors=3, covariance_window_bars=500), as_of=_as_of(panel), universe=sids
        )

    with pytest.raises(RiskModelError, match="are estimable against n_factors"):
        build_snapshot(panel, policy, as_of=_as_of(panel), universe=sids[:3])

    with pytest.raises(RiskModelError, match="needs the exogenous factor return panel"):
        build_snapshot(panel, _policy(n_factors=3, factor_mode="etf"), as_of=_as_of(panel), universe=sids)

    with pytest.raises(RiskModelError, match="empty snapshot covers nothing"):
        build_snapshot(panel, policy, as_of=_as_of(panel), universe=[])


def test_estimation_refuses_a_panel_that_could_not_have_been_read_at_the_decision_bar() -> None:
    """Point-in-time causality: the window ends at the decision bar, never after it."""
    panel, sids = _returns_panel(n_names=20, bars=300)
    policy = _policy(n_factors=3)

    with pytest.raises(RiskModelError, match="may read that bar and no later one"):
        build_snapshot(panel, policy, as_of=str(panel.index[-2].date()), universe=sids)
    # The decision bar itself is readable; the estimate at its close uses bars <= t.
    assert build_snapshot(panel, policy, as_of=str(panel.index[-1].date()), universe=sids).as_of

    shuffled = panel.iloc[np.r_[5:300, 0:5]]
    with pytest.raises(RiskModelError, match="strictly increasing"):
        build_snapshot(shuffled, policy, as_of="2030-01-01", universe=sids)


def test_an_eligibility_mask_is_honoured_as_a_substitution() -> None:
    """The screen ``compute_eligibility`` computes is applied here as a substitution, not a drop."""
    panel, sids = _returns_panel(n_names=20)
    policy = _policy(n_factors=3)
    mask = {sid: sid not in {"S000", "S001"} for sid in sids}
    snapshot = build_snapshot(panel, policy, as_of=_as_of(panel), universe=sids, eligible=mask)

    assert set(snapshot.reduce_only) == {"S000", "S001"}
    assert snapshot.substitutions["S000"]["reason"] == "ineligible"
    assert snapshot.sids == tuple(sids)  # coverage of the universe is preserved, not trimmed


def test_the_etf_mode_estimates_omega_from_the_declared_factor_panel() -> None:
    panel, sids = _returns_panel(n_names=20, bars=300)
    rng = np.random.default_rng(31)
    etf = pd.DataFrame(rng.normal(0.0, 0.008, size=(300, 3)), index=panel.index, columns=["XLK", "XLF", "XLE"])
    policy = _policy(n_factors=3, factor_mode="etf")
    snapshot = build_snapshot(panel, policy, as_of=_as_of(panel), universe=sids, factor_returns=etf)

    assert snapshot.factor_ids == ("XLK", "XLF", "XLE")
    assert snapshot.method["explained_share"] is None
    assert snapshot.omega.shape == (3, 3)

    with pytest.raises(RiskModelError, match="against n_factors"):
        build_snapshot(
            panel, _policy(n_factors=2, factor_mode="etf"), as_of=_as_of(panel), universe=sids, factor_returns=etf
        )


# --------------------------------------------------------------------------
# The one channel the risk model acts through
# --------------------------------------------------------------------------


def test_the_snapshot_is_what_the_variance_budget_measures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decision D8: the budget is the only channel, so the snapshot must satisfy §3.7's surface."""
    panel, sids = _returns_panel(n_names=20)
    snapshot = build_snapshot(panel, _policy(n_factors=3), as_of=_as_of(panel), universe=sids)
    assert isinstance(snapshot, RiskLike)

    w = pd.Series(0.05, index=sids)
    ctx = LimitContext(w0=pd.Series(0.0, index=sids), equity=1_000_000.0, risk=snapshot)
    sigma = math.sqrt(snapshot.variance(w))

    tight = Limit(name="variance_budget", cls=LimitClass.PREFERENCE, priority=1, params={"value": sigma / 2.0})
    loose = Limit(name="variance_budget", cls=LimitClass.PREFERENCE, priority=1, params={"value": sigma * 2.0})
    assert violation(tight, w, ctx) == pytest.approx(sigma - sigma / 2.0, rel=1e-12)
    assert violation(loose, w, ctx) == 0.0

    # A target naming a name the snapshot does not cover is refused before any solve.
    uncovered = pd.concat([w, pd.Series({"UNKNOWN": 0.05})])
    ctx_uncovered = LimitContext(w0=pd.Series(0.0, index=uncovered.index), equity=1_000_000.0, risk=snapshot)
    with pytest.raises(LimitError, match="covers no estimate"):
        violation(tight, uncovered, ctx_uncovered)


def test_variance_refuses_weights_it_cannot_place() -> None:
    snapshot = _snapshot(sids=["A", "B"], B=np.ones((2, 1)), omega=[[0.04]], d=[0.01, 0.01])
    with pytest.raises(RiskModelError, match="does not cover"):
        snapshot.variance(pd.Series({"A": 0.1, "Z": 0.1}))
    # A NaN the caller named is a defect, not a zero: the fill is for names it omitted.
    with pytest.raises(RiskModelError, match="non-finite at \\['A'\\]"):
        snapshot.variance(pd.Series({"A": np.nan}))
    assert snapshot.variance(pd.Series({"A": 0.1})) == pytest.approx(0.04 * 0.01 + 0.01 * 0.01)
    with pytest.raises(RiskModelError, match="against 2 sids"):
        snapshot.variance(np.array([0.1, 0.2, 0.3]))


def test_a_snapshot_refuses_an_incoherent_shape() -> None:
    with pytest.raises(RiskModelError, match="must be unique"):
        _snapshot(sids=["A", "A"], B=np.ones((2, 1)), omega=[[0.04]], d=[0.01, 0.01])
    with pytest.raises(RiskModelError, match="rows against"):
        _snapshot(sids=["A", "B"], B=np.ones((3, 1)), omega=[[0.04]], d=[0.01, 0.01])
    with pytest.raises(RiskModelError, match="d has"):
        _snapshot(sids=["A", "B"], B=np.ones((2, 1)), omega=[[0.04]], d=[0.01])
    with pytest.raises(RiskModelError, match="loading columns"):
        _snapshot(sids=["A", "B"], B=np.ones((2, 2)), omega=[[0.04]], d=[0.01, 0.01])
