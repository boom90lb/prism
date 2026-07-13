"""The forecast-ensemble signal node (SPEC.md §7.1 implementation (b)).

The demotion of the v0.2 ensemble-as-system: the salvaged forecast core
(XGBoost + ARIMA members, purged-OOF member weighting, conformal residual
band) re-founded as one plug-in implementation of the ``Signal`` contract.
What changed relative to the legacy ``research.models.ensemble.EnsembleModel``:

* **Score, not position (I-3/I-4).** The node emits unclipped standardized
  scores ``E[r_h] / (sigma_daily * sqrt(h))`` — with the sqrt(h) horizon
  scaling the legacy vol-mapping omitted (audit D-2). No position cap, no
  vol-targeting: sizing happens once, in construction.
* **Per-bar-causal ARIMA.** The legacy member forecast ``len(X)`` steps from
  the *end of its training data*, so every test bar deep in a fold consumed
  an increasingly stale extrapolation. The node's ARIMA member models daily
  log returns and, at score time, applies the fitted parameters to the
  observed returns up to the decision bar (``ARIMAResults.apply`` — a
  Kalman state update, no refit) before forecasting ``h`` steps — every
  score is conditioned on data through *t* and nothing after (N1).
* **No silent members (N7).** An unknown or unconstructable member raises at
  construction; a member failure during fit raises with symbol context. The
  legacy path logged and dropped. Names with insufficient history are the
  one sanctioned "no opinion" (NaN) case, and they are counted and logged.
* **Scale-invariant features.** All features are built from returns and
  ratios, so a global price-level rescale leaves scores unchanged (to float
  tolerance) — property-tested, per the contract.
* **N8.** Import closure is numpy/pandas/statsmodels/xgboost plus
  prism-internal pure modules. The prophet member is deliberately not
  offered here: the prophet library hard-depends on matplotlib, and
  ``prism.signal`` is a new module where N8's prophet/matplotlib clause
  binds. Prophet remains available on the legacy per-symbol path only.

All OOF bookkeeping (member weights, conformal residuals) happens in
*standardized* units — per-horizon-sigma, the same units the node emits —
so member MAEs are comparable across symbols and vol regimes, and the
conformal quantile can be added directly to a score.

Persistence (save/load) is deliberately absent: the node is refit on the
retrain cadence; durable model state is the live/ loop's job (R4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from prism.conformal import EnbPICalibrator
from prism.signal.base import Signal
from prism.validation.walk_forward import PurgedWalkForward

logger = logging.getLogger(__name__)

_SUPPORTED_MEMBERS = ("xgboost", "arima")

# Feature lookbacks (bars). Ratio/return-based only — price-level invariant.
_MOM_FAST, _MOM_SLOW = 5, 21
_RSI_WINDOW = 14
_VOLUME_WINDOW = 21

# Max ARIMA OOF evaluations per fold per symbol. Each evaluation is a Kalman
# pass over the symbol's history; the pooled MAE is a weight estimate, not an
# exhaustive backtest, so the val grid is strided down to this budget.
_ARIMA_OOF_BUDGET = 25


@dataclass(frozen=True)
class EnsembleNodeConfig:
    """Configuration for :class:`EnsembleSignalNode`.

    ``members`` may only name supported forecast members — configuration
    errors raise at construction (N7), they are not logged away. Weighting
    is pooled inverse OOF-MAE across the cross-section: one weight per
    member, estimated on purged walk-forward folds of the training panel.
    """

    members: tuple[str, ...] = ("xgboost", "arima")
    horizon_bars: int = 5
    vol_window: int = 21
    vol_floor: float = 5e-3
    min_train_rows: int = 100
    oof_splits: int = 3
    oof_embargo_pct: float = 0.01
    conformal_alpha: float = 0.10
    # XGBoost member knobs. nthread=1 + fixed seed keep member fits
    # deterministic so the scale-invariance contract is testable.
    xgb_estimators: int = 150
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    # ARIMA member order on daily log returns (already differenced -> d=0).
    arima_order: tuple[int, int, int] = (1, 0, 1)

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("EnsembleNodeConfig.members must not be empty")
        for name in self.members:
            if name == "prophet":
                raise ValueError(
                    "prophet is not available in prism.signal (N8: the prophet "
                    "library hard-depends on matplotlib); it remains a legacy "
                    "research.models path member only"
                )
            if name not in _SUPPORTED_MEMBERS:
                raise ValueError(
                    f"Unknown signal member {name!r}; supported: {_SUPPORTED_MEMBERS}"
                )
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"Duplicate members in {self.members}")
        if self.horizon_bars < 1:
            raise ValueError(f"horizon_bars must be >= 1, got {self.horizon_bars}")
        if self.vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {self.vol_window}")
        if self.min_train_rows < 30:
            raise ValueError(f"min_train_rows must be >= 30, got {self.min_train_rows}")
        if not 0.0 < self.conformal_alpha < 1.0:
            raise ValueError(f"conformal_alpha must be in (0, 1), got {self.conformal_alpha}")


def _log_prices(close: pd.Series) -> pd.Series:
    """Log prices with non-positive values treated as missing, never -inf."""
    values = pd.to_numeric(close, errors="coerce")
    return pd.Series(
        np.where(values > 0, np.log(values.where(values > 0)), np.nan),
        index=close.index,
        dtype=float,
    )


def build_features(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """Causal, scale-invariant per-name features for the forecast members.

    Everything is a function of log-return windows (plus a volume *ratio*),
    so a global price-level rescale changes nothing but float rounding.
    Leading rows are NaN until each lookback fills — never backfilled (I-2).
    """
    log_ret = _log_prices(close).diff()
    feats = pd.DataFrame(index=close.index)
    feats["ret_1"] = log_ret
    feats["mom_fast"] = log_ret.rolling(_MOM_FAST, min_periods=_MOM_FAST).sum()
    feats["mom_slow"] = log_ret.rolling(_MOM_SLOW, min_periods=_MOM_SLOW).sum()
    feats["vol"] = log_ret.rolling(_MOM_SLOW, min_periods=_MOM_SLOW).std()
    gains = log_ret.clip(lower=0.0).rolling(_RSI_WINDOW, min_periods=_RSI_WINDOW).mean()
    magnitude = log_ret.abs().rolling(_RSI_WINDOW, min_periods=_RSI_WINDOW).mean()
    feats["rsi"] = gains / magnitude.where(magnitude > 0)
    if volume is not None:
        vol_med = (
            pd.to_numeric(volume, errors="coerce")
            .rolling(_VOLUME_WINDOW, min_periods=_VOLUME_WINDOW)
            .median()
        )
        feats["volume_ratio"] = volume / vol_med.where(vol_med > 0)
    return feats


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Realized log return over the next ``horizon`` bars (the fit target)."""
    log_price = _log_prices(close)
    return log_price.shift(-horizon) - log_price


class _XGBMember:
    """Thin deterministic XGBoost regressor on the node's feature block."""

    def __init__(self, config: EnsembleNodeConfig) -> None:
        self._config = config
        self._booster = None
        self._columns: list[str] = []

    def _params(self) -> dict:
        return {
            "objective": "reg:squarederror",
            "max_depth": self._config.xgb_max_depth,
            "learning_rate": self._config.xgb_learning_rate,
            "tree_method": "hist",
            "nthread": 1,
            "seed": 0,
        }

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "_XGBMember":
        import xgboost as xgb

        self._columns = list(features.columns)
        dtrain = xgb.DMatrix(features.to_numpy(dtype=float), label=target.to_numpy(dtype=float))
        self._booster = xgb.train(self._params(), dtrain, num_boost_round=self._config.xgb_estimators)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        if self._booster is None:
            raise RuntimeError("XGBoost member predict before fit")
        aligned = features[self._columns].to_numpy(dtype=float)
        return np.asarray(self._booster.predict(xgb.DMatrix(aligned)), dtype=float)


class _ARIMAMember:
    """ARMA on daily log returns; h-bar expectation via causal state update.

    ``fit`` estimates parameters on the training returns. ``expected_h``
    applies those *fixed* parameters to the observed returns through the
    decision bar (no refit) and sums the h-step forecast of daily log
    returns into an h-bar expected log return. This is the per-bar-refit
    fix: the legacy member extrapolated from the end of its training
    sample for a whole fold.
    """

    def __init__(self, order: tuple[int, int, int], horizon_bars: int) -> None:
        self._order = order
        self._horizon = horizon_bars
        self._results = None

    def fit(self, train_returns: pd.Series) -> "_ARIMAMember":
        from statsmodels.tsa.arima.model import ARIMA

        clean = train_returns.dropna()
        self._results = ARIMA(
            clean.to_numpy(dtype=float), order=self._order, trend="c"
        ).fit()
        return self

    def expected_h(self, returns_through_t: pd.Series) -> float:
        if self._results is None:
            raise RuntimeError("ARIMA member scored before fit")
        clean = returns_through_t.dropna()
        if len(clean) < 3:
            return float("nan")
        applied = self._results.apply(clean.to_numpy(dtype=float))
        return float(np.sum(applied.forecast(self._horizon)))


@dataclass
class _PerSymbolState:
    xgb: _XGBMember | None = None
    arima: _ARIMAMember | None = None


@dataclass
class _SymbolOOF:
    """Standardized OOF predictions for one symbol.

    ``member_*`` live on each member's own evaluation grid (XGBoost dense,
    ARIMA strided) and feed the pooled inverse-MAE weights. ``aligned_*``
    live on one common grid so the members can be *blended* per row for the
    conformal residual pool.
    """

    member_preds: dict[str, np.ndarray] = field(default_factory=dict)
    member_tgts: dict[str, np.ndarray] = field(default_factory=dict)
    aligned_preds: dict[str, np.ndarray] = field(default_factory=dict)
    aligned_tgts: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


class EnsembleSignalNode(Signal):
    """Cross-sectional forecast-blend node under the Signal contract."""

    def __init__(self, config: EnsembleNodeConfig | None = None) -> None:
        self._config = config or EnsembleNodeConfig()
        self._states: dict[str, _PerSymbolState] = {}
        self._weights: dict[str, float] = {}
        self._volume_used = False
        self.weight_basis_: str = "unfit"
        self.conformal_: EnbPICalibrator | None = None
        self.fitted_symbols_: list[str] = []
        self.skipped_symbols_: list[str] = []
        self._is_fitted = False

    # ------------------------------------------------------------------ meta

    @property
    def horizon_bars(self) -> int:
        return self._config.horizon_bars

    @property
    def required_history(self) -> int:
        # Longest feature lookback (+1 for the return diff) and the vol
        # window, plus one bar of slack for the decision row itself.
        return max(_MOM_SLOW, _RSI_WINDOW, _VOLUME_WINDOW, self._config.vol_window) + 2

    # ------------------------------------------------------------------- fit

    def fit(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> "EnsembleSignalNode":
        close = _validate_panel(close)
        volume = _align_optional(volume, close)
        cfg = self._config

        self._volume_used = volume is not None
        self._states.clear()
        self.fitted_symbols_ = []
        self.skipped_symbols_ = []
        symbol_oofs: list[_SymbolOOF] = []

        for symbol in close.columns:
            prepared = self._prepare_symbol(close[symbol], None if volume is None else volume[symbol])
            if prepared is None:
                self.skipped_symbols_.append(symbol)
                continue
            features, target, log_ret, sigma = prepared
            folds = self._oof_folds(len(features))

            state = _PerSymbolState()
            try:
                oof = self._fit_symbol(state, features, target, log_ret, sigma, folds)
            except Exception as exc:  # noqa: BLE001 — re-raise with context (N7)
                raise RuntimeError(f"signal member fit failed for symbol {symbol!r}: {exc}") from exc

            self._states[symbol] = state
            self.fitted_symbols_.append(symbol)
            symbol_oofs.append(oof)

        if not self._states:
            raise ValueError(
                f"No symbol had >= {cfg.min_train_rows} usable training rows; "
                "cannot fit the ensemble node (N7)"
            )
        if self.skipped_symbols_:
            logger.warning(
                "ensemble node: %d/%d symbols below min_train_rows were skipped (NaN scores): %s",
                len(self.skipped_symbols_),
                close.shape[1],
                ", ".join(self.skipped_symbols_[:8]) + ("…" if len(self.skipped_symbols_) > 8 else ""),
            )

        self._fit_weights(symbol_oofs)
        self._fit_conformal(symbol_oofs)
        self._is_fitted = True
        return self

    def _prepare_symbol(
        self, close: pd.Series, volume: pd.Series | None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series] | None:
        """Features/target/returns/sigma on the symbol's usable rows, or None.

        ``log_ret`` keeps the full history (the ARIMA member manages its own
        NaNs); the other three are restricted to rows where every feature,
        the forward target, and the trailing sigma are finite.
        """
        cfg = self._config
        features = build_features(close, volume)
        target = forward_log_return(close, cfg.horizon_bars)
        log_ret = _log_prices(close).diff()
        sigma = (
            log_ret.rolling(cfg.vol_window, min_periods=cfg.vol_window)
            .std()
            .clip(lower=cfg.vol_floor)
        )
        usable = features.notna().all(axis=1) & target.notna() & sigma.notna()
        if int(usable.sum()) < cfg.min_train_rows:
            return None
        return features.loc[usable], target.loc[usable], log_ret, sigma.loc[usable]

    def _oof_folds(self, n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
        splitter = PurgedWalkForward(
            n_splits=self._config.oof_splits,
            purge_horizon=self._config.horizon_bars,
            embargo_pct=self._config.oof_embargo_pct,
            expanding=True,
        )
        return list(splitter.split(np.zeros(n_rows)))

    def _fit_symbol(
        self,
        state: _PerSymbolState,
        features: pd.DataFrame,
        target: pd.Series,
        log_ret: pd.Series,
        sigma: pd.Series,
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> _SymbolOOF:
        """Fit the symbol's members and collect standardized OOF rows."""
        cfg = self._config
        # Per-row standardizer: per-horizon sigma, the score's denominator.
        std = (sigma * np.sqrt(cfg.horizon_bars)).to_numpy(dtype=float)
        tgt = target.to_numpy(dtype=float)
        oof = _SymbolOOF()

        # One strided grid shared by the aligned (blendable) pool. The stride
        # keeps ARIMA's per-row Kalman passes within _ARIMA_OOF_BUDGET.
        aligned_positions: list[np.ndarray] = []
        xgb_dense: dict[int, float] = {}
        arima_strided: dict[int, float] = {}

        for train_idx, val_idx in folds:
            stride = max(1, len(val_idx) // _ARIMA_OOF_BUDGET)
            aligned_positions.append(val_idx[::stride])

            if "xgboost" in cfg.members:
                member = _XGBMember(cfg).fit(features.iloc[train_idx], target.iloc[train_idx])
                preds = member.predict(features.iloc[val_idx])
                oof.member_preds.setdefault("xgboost", np.array([], dtype=float))
                oof.member_preds["xgboost"] = np.concatenate(
                    [oof.member_preds["xgboost"], preds / std[val_idx]]
                )
                oof.member_tgts.setdefault("xgboost", np.array([], dtype=float))
                oof.member_tgts["xgboost"] = np.concatenate(
                    [oof.member_tgts["xgboost"], tgt[val_idx] / std[val_idx]]
                )
                for pos, pred in zip(val_idx, preds):
                    xgb_dense[int(pos)] = float(pred)

            if "arima" in cfg.members:
                train_end = target.index[train_idx[-1]]
                member = _ARIMAMember(cfg.arima_order, cfg.horizon_bars).fit(log_ret.loc[:train_end])
                for pos in val_idx[::stride]:
                    asof = target.index[pos]
                    arima_strided[int(pos)] = member.expected_h(log_ret.loc[:asof])

        if "xgboost" in cfg.members:
            state.xgb = _XGBMember(cfg).fit(features, target)
        if "arima" in cfg.members:
            state.arima = _ARIMAMember(cfg.arima_order, cfg.horizon_bars).fit(log_ret)

        if arima_strided:
            positions = np.array(sorted(arima_strided), dtype=int)
            oof.member_preds["arima"] = np.array(
                [arima_strided[p] for p in positions]
            ) / std[positions]
            oof.member_tgts["arima"] = tgt[positions] / std[positions]

        # Aligned pool on the strided grid: every member must have an
        # opinion at the row for the blend to be meaningful.
        common = np.concatenate(aligned_positions) if aligned_positions else np.array([], dtype=int)
        common = np.array(
            [
                p
                for p in common
                if ("xgboost" not in cfg.members or int(p) in xgb_dense)
                and ("arima" not in cfg.members or int(p) in arima_strided)
            ],
            dtype=int,
        )
        if len(common):
            if "xgboost" in cfg.members:
                oof.aligned_preds["xgboost"] = np.array(
                    [xgb_dense[int(p)] for p in common]
                ) / std[common]
            if "arima" in cfg.members:
                oof.aligned_preds["arima"] = np.array(
                    [arima_strided[int(p)] for p in common]
                ) / std[common]
            oof.aligned_tgts = tgt[common] / std[common]
        return oof

    def _fit_weights(self, symbol_oofs: list[_SymbolOOF]) -> None:
        cfg = self._config
        maes: dict[str, float] = {}
        counts: dict[str, int] = {}
        for member in cfg.members:
            preds = np.concatenate(
                [o.member_preds[member] for o in symbol_oofs if member in o.member_preds]
                or [np.array([], dtype=float)]
            )
            tgts = np.concatenate(
                [o.member_tgts[member] for o in symbol_oofs if member in o.member_tgts]
                or [np.array([], dtype=float)]
            )
            finite = np.isfinite(preds) & np.isfinite(tgts)
            counts[member] = int(finite.sum())
            if counts[member] >= 5:
                maes[member] = float(np.mean(np.abs(preds[finite] - tgts[finite])))
        if len(maes) == len(cfg.members) and all(v > 0 for v in maes.values()):
            inv = {m: 1.0 / v for m, v in maes.items()}
            total = sum(inv.values())
            self._weights = {m: v / total for m, v in inv.items()}
            self.weight_basis_ = "inverse_oof_mae"
        else:
            # Explicit, logged, inspectable — not a silent default (N7).
            self._weights = {m: 1.0 / len(cfg.members) for m in cfg.members}
            self.weight_basis_ = "equal_fallback"
            logger.warning(
                "ensemble node: insufficient finite OOF rows for inverse-MAE "
                "weighting (finite rows per member: %s); using equal weights",
                counts,
            )

    def _fit_conformal(self, symbol_oofs: list[_SymbolOOF]) -> None:
        """EnbPI over the pooled standardized blended OOF residuals.

        Predictions and targets are already in per-horizon-sigma units, so
        the calibrated quantile adds directly to an emitted score.
        """
        cfg = self._config
        blended: list[np.ndarray] = []
        realized: list[np.ndarray] = []
        for oof in symbol_oofs:
            if len(oof.aligned_preds) != len(cfg.members) or not len(oof.aligned_tgts):
                continue
            stacked = np.stack([oof.aligned_preds[m] for m in cfg.members])
            w = np.array([self._weights[m] for m in cfg.members])[:, None]
            blended.append((stacked * w).sum(axis=0))
            realized.append(oof.aligned_tgts)
        if not blended:
            logger.warning("ensemble node: no aligned OOF rows; conformal band disabled")
            self.conformal_ = None
            return
        try:
            self.conformal_ = EnbPICalibrator().fit(
                oof_predictions=np.concatenate(blended),
                targets=np.concatenate(realized),
                position_cap=float("inf"),
            )
        except ValueError as exc:
            logger.warning("ensemble node: conformal calibration skipped (%s)", exc)
            self.conformal_ = None

    # ----------------------------------------------------------------- score

    def score(self, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("EnsembleSignalNode.score called before fit (N7)")
        close = _validate_panel(close)
        if self._volume_used and volume is None:
            raise ValueError(
                "node was fit with volume features; score requires the volume panel (N7)"
            )
        volume = _align_optional(volume, close)
        cfg = self._config

        out = pd.Series(np.nan, index=close.columns, dtype=float)
        for symbol in close.columns:
            state = self._states.get(symbol)
            if state is None:
                continue
            series = close[symbol]
            log_ret = _log_prices(series).diff()
            sigma_tail = log_ret.iloc[-cfg.vol_window :]
            if len(sigma_tail) < cfg.vol_window or sigma_tail.isna().any():
                continue
            sigma = max(float(sigma_tail.std()), cfg.vol_floor)

            member_preds: dict[str, float] = {}
            if state.xgb is not None:
                feats = build_features(series, None if volume is None else volume[symbol])
                row = feats.iloc[[-1]]
                if row.notna().all(axis=1).iloc[0]:
                    member_preds["xgboost"] = float(state.xgb.predict(row)[0])
            if state.arima is not None:
                expected = state.arima.expected_h(log_ret)
                if np.isfinite(expected):
                    member_preds["arima"] = expected
            if len(member_preds) != len(cfg.members):
                continue  # a member has no opinion -> the node has no opinion

            blend = sum(self._weights[m] * member_preds[m] for m in cfg.members)
            out[symbol] = blend / (sigma * np.sqrt(cfg.horizon_bars))
        return out

    def score_band(
        self, close: pd.DataFrame, volume: pd.DataFrame | None = None
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """(lower, point, upper) conformal score bands at ``conformal_alpha``.

        When no calibrator could be fit the bands are NaN — an explicit
        "no calibrated interval", never a fabricated one.
        """
        point = self.score(close, volume)
        if self.conformal_ is None:
            nan = pd.Series(np.nan, index=point.index, dtype=float)
            return nan, point, nan.copy()
        q = self.conformal_.quantile(self._config.conformal_alpha)
        return point - q, point, point + q


def _validate_panel(close: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(close, pd.DataFrame):
        raise TypeError(f"expected a wide DataFrame panel, got {type(close).__name__}")
    if close.columns.has_duplicates:
        raise ValueError("panel has duplicate symbol columns")
    if not close.index.is_monotonic_increasing:
        raise ValueError("panel index must be sorted ascending")
    return close


def _align_optional(volume: pd.DataFrame | None, close: pd.DataFrame) -> pd.DataFrame | None:
    if volume is None:
        return None
    return _validate_panel(volume).reindex(index=close.index, columns=close.columns)
