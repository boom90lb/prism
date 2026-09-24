"""Learned cross-section Signal node (docs/learned_xsection_design.md, RATIFIED 2026-07-19).

The mechanics of the frozen `learned_xsection_v1` family. Every value the
design pins is a default here, and a registration may only select among the
design's own probes (X0-X4); nothing in this module is chosen against a result.

**Features (design §2, F0).** At bar ``b`` and per name, from closes and volume
through ``b`` only:

1. ``mom_12_1``  = close[b-21] / close[b-252] - 1
2. ``mom_6_1``   = close[b-21] / close[b-126] - 1
3. ``rev_1``     = close[b] / close[b-21] - 1
4. ``vol_63``    = sample std of the 63 daily returns ending at b
5. ``maxret_21`` = max of the 21 daily returns ending at b
6. ``high_52w``  = close[b] / max(close over the 252 bars ending at b)
7. ``dv_63``     = median of close * volume over the 63 bars ending at b

**Retained rows.** A name is retained at bar ``b`` when it passes the B1
eligibility screen at ``b`` (``prism.residual.factors.compute_eligibility``),
is a point-in-time index member at ``b`` (the membership mask the engine binds,
see below), and every selected feature is finite. A training row additionally
needs a finite target. Each feature is then rank-transformed over the retained
rows of its own bar to ``(rank - 1) / (n - 1) - 0.5``, average ranks for ties,
so every column of every cross-section is exactly mean-zero.

**Learner (design §2, L0).** At decision bar ``t`` the pool is the
``n_cross_sections`` (36) formation bars ``s_k = t - spacing * k``, ``k = 1..K``,
so ``s + 21 <= t`` holds for every row by construction and every target is
realized at ``t``. The target is the name's close-to-close return over
``(s, s + 21]``, rank-transformed the same way over the training rows of ``s``.
A cross-section is realizable when ``s_k`` has 252 bars of history in the panel
and retains at least one row; fewer than ``K`` realizable cross-sections means
no fit and an all-NaN output (runway, not an error). Ridge fits
``beta = (X'X + lambda I)^-1 X'y`` with no intercept (all columns are
cross-sectionally centered by the rank transform); the score at ``t`` is
``X_t beta``. The GBM probe (X4) is squared-loss gradient boosting of depth-2
regression trees, pinned hyperparameters, no subsampling; it is implemented
here in numpy so the production import path takes no new dependency (SPEC N8)
and the pinned seed is recorded but has nothing random to seed.

**Point-in-time membership.** The training pool reaches 36 months back, and the
panel the engine hands a component covers every sid in the evaluated universe,
including names that joined the index after ``t``. Training on them would put
future index inclusion into the fit. The node therefore declares
``uses_membership = True`` and refuses to score until the engine has bound a
bar x sid membership mask (``bind_membership``); a mask row for ``s <= t``
depends only on whether ``s`` lies inside an interval, so the rows the node
reads are knowable at ``t``.

**Bypass.** ``learner = "bypass"`` pins the feature set to ``mom_12_1`` and emits
its rank at ``t`` with no fit: the parity arm whose decile book must equal the
momentum node's (design §5, the family's central parity test).

Stateless across the train/score boundary: every fit is a trailing refit at
the decision bar, so ``fitted_state = "none"`` like the residual node.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.residual.factors import (
    ResidualStatArbConfig,
    compute_eligibility,
    compute_returns,
)
from prism.signal.base import FittedState, OutputKind, ScoringBasis, Signal, Units

#: F0, exactly these seven, in the design's order.
FEATURES: tuple[str, ...] = ("mom_12_1", "mom_6_1", "rev_1", "vol_63", "maxret_21", "high_52w", "dv_63")

#: Bars of history the widest feature needs before its bar (mom_12_1, high_52w).
FEATURE_HISTORY_BARS = 252

LEARNERS: tuple[str, ...] = ("ridge", "gbm", "bypass")


class LearnedCrossSectionError(RuntimeError):
    """The node cannot score as configured (a missing membership binding, a bad panel)."""


class LearnedCrossSectionNode(Signal):
    """Cross-sectional learned ranking over seven slow characteristics, refit at every decision."""

    output_kind: OutputKind = "rank_score"
    units: Units = "rank"
    scoring_basis: ScoringBasis = "price"
    # Every feature is scale-invariant, but the fit is not: the B1 screen gates
    # on an absolute $5 price over a 36-month training pool, so a global price
    # rescale admits different names into past cross-sections and moves the
    # coefficients. The conformance suite measured it (x1 and x2 moved by about
    # 4e-6 relative under a 3x rescale), so the property is declared absent
    # rather than claimed "within the eligible region".
    price_scale_invariant: bool = False
    fitted_state: FittedState = "none"
    estimated_decay_bars: int | None = None
    #: The engine binds a point-in-time membership mask before scoring.
    uses_membership: bool = True

    def __init__(
        self,
        config: ResidualStatArbConfig | None = None,
        *,
        learner: str = "ridge",
        features: list[str] | tuple[str, ...] = FEATURES,
        ridge_lambda: float = 1.0,
        n_cross_sections: int = 36,
        spacing_bars: int = 21,
        horizon_bars: int = 21,
        gbm_trees: int = 200,
        gbm_depth: int = 2,
        gbm_learning_rate: float = 0.05,
        gbm_min_leaf: int = 50,
        gbm_seed: int = 20260718,
        membership_mask: pd.DataFrame | None = None,
    ) -> None:
        if learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}, got {learner!r}")
        chosen = tuple(str(name) for name in features)
        unknown = [name for name in chosen if name not in FEATURES]
        if unknown:
            raise ValueError(f"features {unknown} are not in F0 {FEATURES}")
        if len(set(chosen)) != len(chosen) or not chosen:
            raise ValueError(f"features must be a non-empty set without repeats, got {chosen}")
        if learner == "bypass" and chosen != ("mom_12_1",):
            # Bypass is the parity arm and nothing else; it scores mom_12_1 alone.
            chosen = ("mom_12_1",)
        if not (np.isfinite(ridge_lambda) and ridge_lambda >= 0.0):
            raise ValueError(f"ridge_lambda must be finite and >= 0, got {ridge_lambda}")
        if n_cross_sections < 1:
            raise ValueError(f"n_cross_sections must be >= 1, got {n_cross_sections}")
        if spacing_bars < 1 or horizon_bars < 1:
            raise ValueError("spacing_bars and horizon_bars must be >= 1")
        if spacing_bars < horizon_bars:
            raise ValueError(
                f"spacing_bars ({spacing_bars}) < horizon_bars ({horizon_bars}) would pool overlapping targets"
            )
        if gbm_trees < 1 or gbm_depth < 1 or gbm_min_leaf < 1 or not gbm_learning_rate > 0.0:
            raise ValueError("GBM hyperparameters must be positive")
        self._config = config or ResidualStatArbConfig()
        self._learner = learner
        self._features = chosen
        self._lambda = float(ridge_lambda)
        self._k = int(n_cross_sections)
        self._spacing = int(spacing_bars)
        self._horizon = int(horizon_bars)
        self._gbm = (int(gbm_trees), int(gbm_depth), float(gbm_learning_rate), int(gbm_min_leaf), int(gbm_seed))
        self._membership: pd.DataFrame | None = membership_mask
        #: The last fit's coefficients (ridge) or tree count (gbm), for the record and the tests.
        self.last_fit: dict[str, object] | None = None

    # ------------------------------------------------------------ contract

    @property
    def horizon_bars(self) -> int:
        return self._horizon

    @property
    def required_history(self) -> int:
        # The decision bar's own features need 252 bars before it, and the
        # eligibility screen needs corr_window finite returns. The 36-cross-
        # section training runway is a no-opinion condition, not a refusal.
        return max(FEATURE_HISTORY_BARS, self._config.corr_window) + 1

    def bind_membership(self, mask: pd.DataFrame | None) -> None:
        """Bind the bar x sid point-in-time membership mask (engine seam)."""
        if mask is not None and not isinstance(mask, pd.DataFrame):
            raise TypeError(f"membership mask must be a DataFrame, got {type(mask).__name__}")
        self._membership = mask

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "LearnedCrossSectionNode":
        """Validate the panel; every fit is a trailing refit inside ``score``."""
        _validate_panels(close, volume, required=self.required_history)
        return self

    def instructions(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series | None:
        """``liquidate`` for an eligible member whose decision-bar features are not all finite.

        Design §2: a NaN-featured name takes no position. NaN alone would read
        as "hold what you have" to the composition state machine.
        """
        volume = _validate_panels(close, volume, required=self.required_history)
        mask = self._aligned_membership(close)
        eligible = compute_eligibility(close, volume, self._config, mask).iloc[-1].astype(bool)
        frames = _feature_frames(close, volume, self._features, rows=[len(close) - 1])
        finite = np.ones(close.shape[1], dtype=bool)
        for name in self._features:
            finite &= np.isfinite(frames[name][0])
        flagged = eligible.to_numpy() & ~finite
        if not flagged.any():
            return None
        return pd.Series("liquidate", index=close.columns[flagged], dtype=object)

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        """The learned score at the panel's last row; all-NaN before the runway is met."""
        volume = _validate_panels(close, volume, required=self.required_history)
        mask = self._aligned_membership(close)
        eligible = compute_eligibility(close, volume, self._config, mask).to_numpy(dtype=bool)
        t = len(close) - 1
        out = pd.Series(np.nan, index=close.columns, dtype=float)

        if self._learner == "bypass":
            frames = _feature_frames(close, volume, self._features, rows=[t])
            x_t, keep_t = _ranked_cross_section(frames, self._features, 0, eligible[t])
            out.iloc[np.flatnonzero(keep_t)] = x_t[:, 0]
            self.last_fit = {"learner": "bypass", "n_rows": 0}
            return out

        formation = [t - self._spacing * k for k in range(1, self._k + 1)]
        realizable = [s for s in formation if s - FEATURE_HISTORY_BARS >= 0]
        if len(realizable) < self._k:
            self.last_fit = {"learner": self._learner, "runway": len(realizable), "needed": self._k}
            return out
        rows = sorted(realizable) + [t]
        frames = _feature_frames(close, volume, self._features, rows=rows)
        closes = close.to_numpy(dtype=float)

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for position, s in enumerate(sorted(realizable)):
            with np.errstate(invalid="ignore", divide="ignore"):
                base = closes[s]
                target = closes[s + self._horizon] / np.where(base > 0.0, base, np.nan) - 1.0
            usable = eligible[s] & np.isfinite(target)
            x_s, keep = _ranked_cross_section(frames, self._features, position, usable)
            if keep.sum() == 0:
                self.last_fit = {"learner": self._learner, "runway": position, "needed": self._k}
                return out
            xs.append(x_s)
            ys.append(_rank_centered(target[keep]))
        x_t, keep_t = _ranked_cross_section(frames, self._features, len(rows) - 1, eligible[t])
        if keep_t.sum() == 0:
            self.last_fit = {"learner": self._learner, "n_rows": 0}
            return out

        X = np.vstack(xs)
        y = np.concatenate(ys)
        pooled = {
            "formation_positions": sorted(realizable),
            "max_target_position": max(realizable) + self._horizon,
            "decision_position": t,
            "n_rows": len(y),
        }
        if self._learner == "ridge":
            beta = _ridge(X, y, self._lambda)
            prediction = x_t @ beta
            self.last_fit = {"learner": "ridge", "beta": dict(zip(self._features, beta.tolist())), **pooled}
        else:
            trees, depth, rate, min_leaf, seed = self._gbm
            model = _fit_gbm(X, y, n_trees=trees, depth=depth, learning_rate=rate, min_leaf=min_leaf)
            prediction = _predict_gbm(model, x_t)
            self.last_fit = {"learner": "gbm", "n_trees": trees, "seed": seed, **pooled}
        out.iloc[np.flatnonzero(keep_t)] = prediction
        return out

    # ------------------------------------------------------------ internals

    def _aligned_membership(self, close: pd.DataFrame) -> pd.DataFrame:
        if self._membership is None:
            raise LearnedCrossSectionError(
                "LearnedCrossSectionNode needs a point-in-time membership mask bound before it scores: its "
                "training pool reaches 36 cross-sections back, and without the mask names that joined the index "
                "after the decision bar would enter the fit (SPEC N1). The engine binds it for components "
                "declaring uses_membership; a caller outside the engine passes membership_mask"
            )
        aligned = self._membership.reindex(index=close.index, columns=close.columns)
        return aligned.fillna(False).astype(bool)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def _feature_frames(
    close: pd.DataFrame, volume: pd.DataFrame, features: tuple[str, ...], *, rows: list[int]
) -> dict[str, np.ndarray]:
    """Each selected feature at the requested row positions, shape (len(rows), n_names)."""
    c = close.to_numpy(dtype=float)
    c = np.where(c > 0.0, c, np.nan)
    idx = np.asarray(rows, dtype=int)
    out: dict[str, np.ndarray] = {}
    needs_returns = any(name in ("vol_63", "maxret_21") for name in features)
    returns = compute_returns(close) if needs_returns else pd.DataFrame(index=close.index)

    def lag(offset: int) -> np.ndarray:
        positions = idx - offset
        values = np.full((len(idx), c.shape[1]), np.nan)
        valid = positions >= 0
        values[valid] = c[positions[valid]]
        return values

    with np.errstate(invalid="ignore", divide="ignore"):
        for name in features:
            if name == "mom_12_1":
                out[name] = lag(21) / lag(252) - 1.0
            elif name == "mom_6_1":
                out[name] = lag(21) / lag(126) - 1.0
            elif name == "rev_1":
                out[name] = lag(0) / lag(21) - 1.0
            elif name == "vol_63":
                rolled = returns.rolling(63, min_periods=63).std(ddof=1).to_numpy(dtype=float)
                out[name] = rolled[idx]
            elif name == "maxret_21":
                rolled = returns.rolling(21, min_periods=21).max().to_numpy(dtype=float)
                out[name] = rolled[idx]
            elif name == "high_52w":
                peak = pd.DataFrame(c).rolling(252, min_periods=252).max().to_numpy(dtype=float)
                out[name] = c[idx] / peak[idx]
            elif name == "dv_63":
                dollars = pd.DataFrame(c * volume.to_numpy(dtype=float))
                rolled = dollars.rolling(63, min_periods=63).median().to_numpy(dtype=float)
                out[name] = rolled[idx]
            else:  # pragma: no cover - guarded in __init__
                raise ValueError(name)
    return out


def _ranked_cross_section(
    frames: dict[str, np.ndarray], features: tuple[str, ...], position: int, usable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-transformed features of the retained names at one bar, and the retained mask."""
    keep = np.asarray(usable, dtype=bool).copy()
    for name in features:
        keep &= np.isfinite(frames[name][position])
    columns = [_rank_centered(frames[name][position][keep]) for name in features]
    matrix = np.column_stack(columns) if columns else np.zeros((int(keep.sum()), 0))
    return matrix, keep


def _rank_centered(values: np.ndarray) -> np.ndarray:
    """``(rank - 1) / (n - 1) - 0.5`` with average ranks for ties; a single value maps to 0."""
    n = len(values)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.zeros(1)
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    return (ranks - 1.0) / (n - 1.0) - 0.5


# --------------------------------------------------------------------------
# Learners
# --------------------------------------------------------------------------


def _ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    p = X.shape[1]
    gram = X.T @ X + lam * np.eye(p)
    return np.linalg.solve(gram, X.T @ y)


def _fit_gbm(
    X: np.ndarray, y: np.ndarray, *, n_trees: int, depth: int, learning_rate: float, min_leaf: int
) -> dict[str, object]:
    """Squared-loss gradient boosting of greedy depth-limited regression trees, no subsampling.

    Each feature is argsorted once per fit; a node's order is the global order
    filtered by the node's membership, so a split scan is linear. A split is
    admissible only between distinct feature values and when both children keep
    ``min_leaf`` rows. The gain is the standard ``S_L^2/n_L + S_R^2/n_R - S^2/n``
    on the current residuals.
    """
    n, p = X.shape
    base = float(y.mean()) if n else 0.0
    prediction = np.full(n, base)
    orders = [np.argsort(X[:, j], kind="mergesort") for j in range(p)]
    trees: list[list[tuple]] = []
    for _ in range(n_trees):
        residual = y - prediction
        tree: list[tuple] = []
        _grow(X, residual, orders, np.ones(n, dtype=bool), depth, min_leaf, learning_rate, tree)
        trees.append(tree)
        prediction += _predict_tree(tree, X)
    return {"base": base, "trees": trees}


def _grow(
    X: np.ndarray,
    residual: np.ndarray,
    orders: list[np.ndarray],
    member: np.ndarray,
    depth: int,
    min_leaf: int,
    rate: float,
    tree: list[tuple],
) -> int:
    """Append the subtree rooted at ``member`` to ``tree`` (preorder); return its node id."""
    node_id = len(tree)
    count = int(member.sum())
    value = rate * float(residual[member].mean()) if count else 0.0
    if depth == 0 or count < 2 * min_leaf:
        tree.append(("leaf", value))
        return node_id
    best = _best_split(X, residual, orders, member, min_leaf)
    if best is None:
        tree.append(("leaf", value))
        return node_id
    feature, threshold = best
    tree.append(("split", feature, threshold, -1, -1))
    left_member = member & (X[:, feature] <= threshold)
    right_member = member & ~(X[:, feature] <= threshold)
    left = _grow(X, residual, orders, left_member, depth - 1, min_leaf, rate, tree)
    right = _grow(X, residual, orders, right_member, depth - 1, min_leaf, rate, tree)
    tree[node_id] = ("split", feature, threshold, left, right)
    return node_id


def _best_split(
    X: np.ndarray, residual: np.ndarray, orders: list[np.ndarray], member: np.ndarray, min_leaf: int
) -> tuple[int, float] | None:
    total = float(residual[member].sum())
    count = int(member.sum())
    parent = total * total / count
    best_gain = 1e-12
    best: tuple[int, float] | None = None
    for j, order in enumerate(orders):
        idx = order[member[order]]
        xs = X[idx, j]
        cumulative = np.cumsum(residual[idx])
        left_n = np.arange(1, count + 1)
        # A split after position i puts rows 0..i left; admissible where the
        # value changes and both sides keep min_leaf rows.
        candidates = np.flatnonzero((xs[:-1] < xs[1:]) & (left_n[:-1] >= min_leaf) & (count - left_n[:-1] >= min_leaf))
        if candidates.size == 0:
            continue
        s_left = cumulative[candidates]
        n_left = left_n[candidates].astype(float)
        s_right = total - s_left
        n_right = count - n_left
        gain = s_left * s_left / n_left + s_right * s_right / n_right - parent
        winner = int(np.argmax(gain))
        if gain[winner] > best_gain:
            best_gain = float(gain[winner])
            i = candidates[winner]
            best = (j, float(0.5 * (xs[i] + xs[i + 1])))
    return best


def _predict_tree(tree: list[tuple], X: np.ndarray) -> np.ndarray:
    out = np.zeros(X.shape[0])
    stack = [(0, np.arange(X.shape[0]))]
    while stack:
        node_id, rows = stack.pop()
        node = tree[node_id]
        if node[0] == "leaf":
            out[rows] = node[1]
            continue
        _, feature, threshold, left, right = node
        goes_left = X[rows, feature] <= threshold
        stack.append((left, rows[goes_left]))
        stack.append((right, rows[~goes_left]))
    return out


def _predict_gbm(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    prediction = np.full(X.shape[0], float(model["base"]))
    for tree in model["trees"]:  # type: ignore[union-attr]
        prediction += _predict_tree(tree, X)
    return prediction


def _validate_panels(close: pd.DataFrame, volume: pd.DataFrame | None, *, required: int) -> pd.DataFrame:
    """Shared fit/score panel checks; returns the volume panel (never None)."""
    if not isinstance(close, pd.DataFrame):
        raise TypeError(f"expected a wide DataFrame close panel, got {type(close).__name__}")
    if volume is None:
        raise ValueError(
            "the learned cross-section node requires the volume panel: dv_63 and the eligibility screen read it "
            "(N7, no silent pass)"
        )
    if not close.columns.equals(volume.columns) or not close.index.equals(volume.index):
        raise ValueError("close and volume panels must share index and columns")
    if close.columns.has_duplicates:
        raise ValueError("panel has duplicate symbol columns")
    if not close.index.is_monotonic_increasing:
        raise ValueError("panel index must be sorted ascending")
    if len(close) < required:
        raise ValueError(f"panel has {len(close)} rows; the learned cross-section node needs >= {required}")
    return volume
