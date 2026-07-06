"""Rolling walk-forward evaluation for residual (Avellaneda-Lee) stat-arb.

Mirrors the pairs walk-forward, with one structural difference: there is no
per-fold model selection to freeze. Eigenportfolios, betas, and OU fits re-roll
causally through test bars on trailing windows; folds exist for reporting, for
forcing positions flat at fold boundaries (carry rules are still intentionally
absent), and so the fold geometry matches the pairs path. Only hyperparameters
are fixed — and in v1 those are frozen at the config defaults.

Within-run there is exactly one trial (one frozen config), so the deflated
Sharpe for this path is computed *across runs* from the persisted trial ledger
by the CLI script, not here; the core summary exposes ``oos_periodic_sharpe``
as the ledger entry's input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prism.residual.factors import (
    ResidualStatArbConfig,
    compute_eligibility,
    consensus_trading_days,
)
from prism.execution.participation import participation_capped_targets
from prism.execution.target_weights import PortfolioBacktestResult, backtest_target_weights
from prism.residual.residual import (
    ResidualSignalPanel,
    compute_residual_signal_panel,
    next_states,
)
from prism.portfolio.construct import (
    build_residual_book_row,
    cap_book,
    closed_form_band,
    cost_aware_band,
    step_no_trade_band,
    strength_multiplier,
)
from research.arbitrage.walk_forward import (
    StatArbWalkForwardConfig,
    _empty_targets,
    _fold_metrics_from_result,
    _FoldSlices,
    _force_fold_flat,
    _numeric_prices,
    _slice_portfolio_result,
    iter_walk_forward_slices,
)
from prism.config import ExecutionConfig
from prism.validation.metrics import periodic_sharpe

# Conviction multiplier cap for sizing_mode="strength": a documented
# modeling constant, NOT fitted on the backtest.
_SIZE_CAP = 2.0

# Risk-aversion constant of the closed-form (Martin 2012 cube-root) band:
# pre-registered at 1.0, never fitted or swept (docs/r2_design.md §1).
GAMMA_RISK = 1.0

# Pre-registered conservative-upper spread schedule (docs/r2_design.md §3,
# I-9): (median formation-window dollar-volume floor, one-way spread bps),
# descending floors. Not fitted; replaced only by fill calibration when fills
# exist. Unknown liquidity (NaN median) lands in the widest bucket — fail-safe,
# never cheap.
SPREAD_BUCKET_SCHEDULE_V1: tuple[tuple[float, float], ...] = (
    (500e6, 1.0),
    (100e6, 2.0),
    (25e6, 5.0),
    (0.0, 10.0),
)


def bucket_spread_bps(median_dollar_volume: pd.Series) -> pd.Series:
    """Map per-name median dollar volume through ``SPREAD_BUCKET_SCHEDULE_V1``."""
    mdv = pd.to_numeric(median_dollar_volume, errors="coerce").to_numpy(dtype=float)
    floors = SPREAD_BUCKET_SCHEDULE_V1[:-1]
    with np.errstate(invalid="ignore"):
        out = np.select(
            [mdv >= floor for floor, _ in floors],
            [bps for _, bps in floors],
            default=SPREAD_BUCKET_SCHEDULE_V1[-1][1],
        )
    return pd.Series(out, index=median_dollar_volume.index, dtype=float)


@dataclass(frozen=True)
class ResidualFoldResult:
    """Diagnostics and metrics for one residual walk-forward fold."""

    fold: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    formation_rows: int
    test_rows: int
    names_traded: int
    cost_to_gross_pnl: float
    signal: dict[str, float]
    metrics: dict[str, float]


@dataclass(frozen=True)
class ResidualStatArbWalkForwardResult:
    """End-to-end residual walk-forward result."""

    folds: tuple[ResidualFoldResult, ...]
    portfolio: PortfolioBacktestResult
    panel: ResidualSignalPanel
    summary: dict[str, float]


def _fold_cost_share(result: PortfolioBacktestResult, test_index: pd.Index) -> float:
    """Costs as a share of gross (pre-cost) PnL magnitude over the fold.

    Values above 1 mean costs exceeded everything the signal made before
    costs — the most direct "this fold was a toll booth" diagnostic.
    (Denominator is ``|sum(gross_t)|`` — net PnL magnitude; the per-period lens
    ``cost_to_gross`` in ``prism.validation.capacity.cost_toll`` divides by
    ``sum(|gross_t|)`` instead. Different statistics; they will not reconcile.)
    """
    rows = result.returns.index.intersection(test_index)
    if rows.empty:
        return float("nan")
    costs = float(result.costs.loc[rows, "total"].sum())
    gross_pnl = float((result.returns.loc[rows] + result.costs.loc[rows, "total"]).sum())
    return costs / max(abs(gross_pnl), 1e-9)


def _momentum_scores(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    signal_config: ResidualStatArbConfig,
    walk_config: StatArbWalkForwardConfig,
    membership_mask: pd.DataFrame | None,
) -> np.ndarray:
    """Cross-sectional momentum score panel (demotion_design.md §2b).

    ``score[t] = close[t - skip] / close[t - lookback] - 1`` — strictly
    trailing, so appending future bars never changes a past score — masked to
    the SAME eligibility screen the residual sleeve trades under. Bars without
    ``lookback`` bars of history (and non-positive base prices) are NaN, which
    downstream means "no momentum position", never "trade anyway".
    """
    lookback = walk_config.mom_lookback_bars
    skip = walk_config.mom_skip_bars
    px = closes.to_numpy(dtype=float)
    n_days, n_symbols = px.shape
    scores = np.full((n_days, n_symbols), np.nan)
    if n_days > lookback:
        with np.errstate(invalid="ignore", divide="ignore"):
            base = px[: n_days - lookback]
            recent = px[lookback - skip : n_days - skip]
            scores[lookback:] = np.where(base > 0.0, recent / base - 1.0, np.nan)
    elig = compute_eligibility(closes, volumes, signal_config, membership_mask)
    scores[~elig.to_numpy(dtype=bool)] = np.nan
    return scores


def _momentum_row(scores_t: np.ndarray, decile: float) -> np.ndarray:
    """Equal-weight top/bottom-decile long-short row at raw gross 1.0.

    ``n_dec = floor(n_finite * decile)``; each winner gets ``+0.5/n_dec``,
    each loser ``-0.5/n_dec`` (stable sort, so ties are deterministic). A
    cross-section too thin for one name per side emits a zero row.
    """
    row = np.zeros(scores_t.shape[0])
    finite = np.flatnonzero(np.isfinite(scores_t))
    n_dec = int(len(finite) * decile)
    if n_dec < 1:
        return row
    order = np.argsort(scores_t[finite], kind="stable")
    row[finite[order[:n_dec]]] = -0.5 / n_dec
    row[finite[order[-n_dec:]]] = 0.5 / n_dec
    return row


def _smoothed_sscores(sscore: pd.DataFrame, config: ResidualStatArbConfig) -> np.ndarray:
    """Causal EWMA of the s-score panel (demotion_design.md §2); 0 = raw scores.

    Recursive form (``adjust=False``) so a live incremental EWMA reproduces it
    exactly; decay is per bar regardless of gaps (``ignore_na=False``), and the
    smoothed panel is re-masked to the raw panel's NaNs — smoothing carries
    state across an invalid-OU gap but never resurrects a day that had no
    valid fit.
    """
    if config.sscore_ewma_halflife_bars <= 0.0:
        return sscore.to_numpy()
    smoothed = sscore.ewm(
        halflife=config.sscore_ewma_halflife_bars, adjust=False, ignore_na=False
    ).mean()
    return smoothed.where(sscore.notna()).to_numpy()


def _online_banded_targets(
    targets: pd.DataFrame,
    band: float | pd.Series,
    walk_config: StatArbWalkForwardConfig,
    *,
    dollar_volume: pd.DataFrame | None = None,
    aum: float = 1.0,
    decision_every: int = 1,
) -> pd.DataFrame:
    """Apply the no-trade band with online (held-state) semantics over a fold.

    Each day's fresh target is banded against the weights the book actually
    holds, via ``step_no_trade_band`` — the same call a live daily loop makes
    (SPEC §7.3) — and the post-cap row is fed back as the next day's held
    state. The retired batch form banded the whole frame first and re-capped
    after, so its hysteresis state tracked weights the book never held; the
    related batch-replay-from-zero divergence is machine-checked in
    ``formal/PrismFormal/Band.lean``. The gross/per-symbol caps are enforced
    last because they are hard risk limits: a capped scale-down is visible to
    the band tomorrow as the true held book. Folds genuinely start flat, so
    the held state starts at zero.

    When ``walk_config.max_participation > 0`` the hard participation gate
    (r2_design.md §2) runs after band + caps against the held weights and that
    day's trailing dollar volume, and the *gated* row becomes both the emitted
    target and tomorrow's held state — the backtest charges gated trades, and
    unfilled residual demand must re-clear the band tomorrow.
    """
    gate = walk_config.max_participation > 0.0
    if gate and dollar_volume is None:
        raise ValueError("max_participation > 0 requires a dollar_volume panel")
    held = pd.Series(0.0, index=targets.columns)
    out = np.empty((len(targets.index), len(targets.columns)))
    for i, day in enumerate(targets.index):
        if i % decision_every != 0:
            # Non-decision bar (demotion_design.md §2): no band step, no gate,
            # no trade — the emitted row repeats the held book exactly, which
            # the weight-space backtest prices as zero trade.
            out[i] = held.to_numpy(dtype=float)
            continue
        stepped = step_no_trade_band(held, targets.loc[day], band)
        capped = cap_book(
            stepped.to_frame().T,
            walk_config.max_gross,
            walk_config.max_symbol_abs_weight,
        ).iloc[0]
        if gate:
            # adv_floor=0.0 deliberately: flooring ADV up would loosen the cap
            # for exactly the illiquid names the gate protects (r2_design.md §2).
            capped = participation_capped_targets(
                held,
                capped,
                dollar_volume.loc[day],
                aum=aum,
                max_participation=walk_config.max_participation,
                adv_floor=0.0,
            )
        held = capped.reindex(targets.columns).fillna(0.0)
        out[i] = held.to_numpy(dtype=float)
    return pd.DataFrame(out, index=targets.index, columns=targets.columns)


def _windowed_targets(
    window: slice,
    index: pd.Index,
    symbols: list[str],
    panel: ResidualSignalPanel,
    sscores: np.ndarray,
    tradeable: np.ndarray,
    signal_config: ResidualStatArbConfig,
    walk_config: StatArbWalkForwardConfig,
    mom_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    """Capped close-time targets over one panel window, state machine from flat.

    One construction for both consumers: the test-window build, and the
    closed-form band's formation-window replay — which estimates target-change
    variance on formation bars only and feeds NOTHING forward (r2_design.md §1).

    ``signal_config.decision_every`` restricts state-machine decisions to bars
    where ``(t - window.start) % decision_every == 0`` (the first bar of every
    window is a decision bar); between decision bars the target row is frozen
    at the last decision row (demotion_design.md §2). At the default 1 every
    bar is a decision bar and this is the frozen-v1 path bit-for-bit. Both
    consumers share the rule, so the closed-form band's ``sigma2_target``
    reflects cadence-consistent target changes.

    ``walk_config.sleeve_mode`` adds the Arm-B momentum sleeve
    (demotion_design.md §2b): its component refreshes from ``mom_scores`` on
    its own (slower) ``mom_decision_every`` cadence, the emitted row picks the
    refresh up at the next trading decision bar, and in ``two_speed`` the
    sleeves are summed pre-cap so the combined book is capped/banded/gated as
    one.
    """
    sleeve = walk_config.sleeve_mode
    if sleeve != "off" and mom_scores is None:
        raise ValueError("sleeve_mode != 'off' requires a momentum score panel")
    s_win = sscores[window]
    ok_win = tradeable[window]
    current = np.zeros(len(symbols), dtype=np.int8)
    mom_current = np.zeros(len(symbols))
    rows = np.zeros((window.stop - window.start, len(symbols)))
    for offset, t in enumerate(range(window.start, window.stop)):
        if sleeve != "off" and offset % walk_config.mom_decision_every == 0:
            mom_current = _momentum_row(mom_scores[t], walk_config.mom_decile)
        if offset % signal_config.decision_every != 0:
            rows[offset] = rows[offset - 1]
            continue
        row = np.zeros(len(symbols))
        if sleeve != "momentum_only":
            current = next_states(current, s_win[offset], ok_win[offset], signal_config)
            q_idx = int(panel.q_index[t])
            if q_idx >= 0 and current.any():
                size_scale = None
                if signal_config.sizing_mode == "strength":
                    # Conviction multiplier in [0, _SIZE_CAP]: 0 at the entry band,
                    # growing with |s| past it (Da-Nagel-Xiu shrinkage of weak
                    # signals); cap_book then redistributes the fixed gross budget
                    # toward conviction.
                    entry = min(abs(signal_config.s_entry_long), abs(signal_config.s_entry_short))
                    size_scale = strength_multiplier(sscores[t], entry, cap=_SIZE_CAP)
                row = build_residual_book_row(
                    current,
                    panel.beta[t],
                    panel.eigenportfolios[q_idx],
                    signal_config.position_unit,
                    size_scale=size_scale,
                )
        if sleeve != "off":
            row = row + mom_current
        rows[offset] = row
    targets = pd.DataFrame(rows, index=index[window], columns=symbols)
    return cap_book(targets, walk_config.max_gross, walk_config.max_symbol_abs_weight)


def run_residual_stat_arb_walk_forward(
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    volumes: pd.DataFrame,
    signal_config: ResidualStatArbConfig | None = None,
    walk_config: StatArbWalkForwardConfig | None = None,
    execution: ExecutionConfig | None = None,
    membership_mask: pd.DataFrame | None = None,
    initial_capital: float = 1.0,
) -> ResidualStatArbWalkForwardResult:
    """Run causal residual signal generation, fold accounting, and net backtest.

    ``membership_mask`` (optional, day x symbol bool) enforces point-in-time
    index membership in eligibility; it is aligned to the (trading-day-filtered)
    panel inside ``compute_eligibility``. ``None`` reproduces frozen-v1 output.
    """
    signal_config = signal_config or ResidualStatArbConfig()
    walk_config = walk_config or StatArbWalkForwardConfig()
    execution = execution or ExecutionConfig()
    if walk_config.formation_bars < signal_config.warmup_bars:
        raise ValueError(
            "formation_bars must cover the estimator warmup: need >= corr_window + regr_window = "
            f"{signal_config.warmup_bars}, got {walk_config.formation_bars}"
        )

    closes = _numeric_prices(close_prices)
    opens = _numeric_prices(open_prices)
    vols = _numeric_prices(volumes)
    common_index = closes.index.intersection(opens.index).intersection(vols.index)
    symbols = sorted(set(closes.columns) & set(opens.columns) & set(vols.columns))
    etf_set = set(signal_config.etf_symbols)
    stock_symbols = [s for s in symbols if s not in etf_set]
    if signal_config.factor_mode == "etf":
        missing = etf_set - set(symbols)
        if missing:
            raise ValueError(f"ETF factors missing from price panels: {sorted(missing)}")
        if len(stock_symbols) < 2:
            raise ValueError(f"need >= 2 tradeable stock columns in ETF mode, got {len(stock_symbols)}")
    elif len(symbols) < signal_config.n_factors + 2:
        raise ValueError(f"need at least n_factors + 2 = {signal_config.n_factors + 2} symbols with full data")
    closes = closes.loc[common_index, symbols]
    opens = opens.loc[common_index, symbols]
    vols = vols.loc[common_index, symbols]

    # Drop phantom (non-trading) union dates before any estimator sees them: a
    # single stray holiday bar would otherwise inject a panel-wide NaN row and
    # bench the whole cross-section for a full corr_window under the strict
    # full-history eligibility rule.
    trading_days = consensus_trading_days(closes)
    n_dropped_calendar_days = int((~trading_days).sum())
    closes = closes.loc[trading_days]
    opens = opens.loc[trading_days]
    vols = vols.loc[trading_days]
    if len(closes) < walk_config.formation_bars + walk_config.min_test_bars + 1:
        raise ValueError("Not enough aligned rows for the requested residual stat-arb WFO")

    panel = compute_residual_signal_panel(closes, vols, signal_config, membership_mask)

    full_targets = _empty_targets(closes.index, symbols)
    fold_shells: list[dict[str, object]] = []
    all_test_index = pd.Index([])
    # Smoothed once for the whole panel (causal, so pre-window history is fair
    # game); the state machine AND strength sizing both consume the smoothed
    # scores (demotion_design.md §2).
    sscores = _smoothed_sscores(panel.sscore, signal_config)
    tradeable = panel.tradeable.to_numpy()
    mom_scores: np.ndarray | None = None
    if walk_config.sleeve_mode != "off":
        mom_scores = _momentum_scores(closes, vols, signal_config, walk_config, membership_mask)

    raw_dollar_volume = closes * vols
    gate_active = walk_config.max_participation > 0.0
    adv_trailing: pd.DataFrame | None = None
    if gate_active:
        # The same trailing statistic the eligibility screen uses
        # (compute_eligibility): rolling median dollar volume over
        # dollar_volume_window bars, full-window min_periods — causal as of
        # each close, never same-day volume alone.
        adv_trailing = raw_dollar_volume.rolling(
            signal_config.dollar_volume_window, min_periods=signal_config.dollar_volume_window
        ).median()
    backtest_spread_bps: pd.Series | None = None

    for slices in iter_walk_forward_slices(len(closes), walk_config):
        test_index = closes.index[slices.test]
        targets = _windowed_targets(
            slices.test, closes.index, symbols, panel, sscores, tradeable, signal_config, walk_config,
            mom_scores=mom_scores,
        )
        fold_spread_bps: pd.Series | None = None
        if walk_config.spread_mode == "bucket":
            fold_spread_bps = bucket_spread_bps(
                raw_dollar_volume.iloc[slices.formation].median(axis=0, skipna=True)
            )
            if backtest_spread_bps is None:
                # One backtest prices the whole sample, so it carries the FIRST
                # formation window's buckets — causal for every test bar (the
                # same convention as the CLI's --top_liquid screen). Per-fold
                # buckets still drive each fold's band cost below.
                backtest_spread_bps = fold_spread_bps
        band: float | pd.Series | None = None
        if walk_config.band_mode == "cost_aware":
            # Per-name band from each name's most-recent causal OU half-life (formation
            # window only, never the test window) and the linear round-trip cost.
            per_trade_cost = (execution.commission_bps + execution.spread_bps) * 2.0 / 10_000.0
            hl_hist = panel.half_life_bars.iloc[: slices.test.start].replace([np.inf, -np.inf], np.nan)
            hl_recent = (
                hl_hist.ffill().iloc[-1].to_numpy() if len(hl_hist) else np.full(len(symbols), np.nan)
            )
            band = pd.Series(cost_aware_band(hl_recent, per_trade_cost), index=symbols)
        elif walk_config.band_mode == "closed_form":
            # sigma2_target from a formation-only replay of the exact test-window
            # construction; the replay is discarded after the variance estimate
            # (r2_design.md §1). Names with <2 finite diffs or zero variance -> band 0.
            formation_targets = _windowed_targets(
                slices.formation, closes.index, symbols, panel, sscores, tradeable, signal_config, walk_config,
                mom_scores=mom_scores,
            )
            sigma2 = formation_targets.diff().iloc[1:].var(ddof=1)
            spread_bps_i: float | np.ndarray = (
                fold_spread_bps.reindex(symbols).to_numpy(dtype=float)
                if fold_spread_bps is not None
                else execution.spread_bps
            )
            round_trip = 2.0 * (execution.commission_bps + spread_bps_i) / 10_000.0
            band = pd.Series(
                closed_form_band(sigma2.to_numpy(dtype=float), round_trip, gamma_risk=GAMMA_RISK),
                index=symbols,
            )
        elif walk_config.no_trade_band > 0:
            band = walk_config.no_trade_band
        if band is not None or gate_active:
            # band 0.0 = pass-through banding so the gate still runs with
            # held-state feedback when no band is configured.
            targets = _online_banded_targets(
                targets,
                0.0 if band is None else band,
                walk_config,
                dollar_volume=None if adv_trailing is None else adv_trailing.loc[test_index],
                aum=initial_capital,
                decision_every=signal_config.decision_every,
            )
        targets = _force_fold_flat(targets)

        full_targets.loc[test_index, symbols] = targets
        all_test_index = all_test_index.union(test_index)
        fold_shells.append(
            {
                "slices": slices,
                "names_traded": int((targets[stock_symbols].abs() > 0).any(axis=0).sum()),
                "signal": panel.diagnostics(slices.test.start, slices.test.stop),
            }
        )

    if not fold_shells:
        raise ValueError("No walk-forward folds were produced")

    dollar_volume = raw_dollar_volume.reindex(index=opens.index, columns=opens.columns)
    full_portfolio = backtest_target_weights(
        opens,
        full_targets,
        execution=execution,
        dollar_volume=dollar_volume,
        initial_capital=initial_capital,
        spread_bps_per_name=backtest_spread_bps,
    )
    portfolio = _slice_portfolio_result(full_portfolio, all_test_index)

    folds: list[ResidualFoldResult] = []
    for shell in fold_shells:
        slices = shell["slices"]
        if not isinstance(slices, _FoldSlices):
            raise TypeError("internal fold slice metadata corrupted")
        formation_index = closes.index[slices.formation]
        test_index = closes.index[slices.test]
        folds.append(
            ResidualFoldResult(
                fold=slices.fold,
                formation_start=formation_index[0],
                formation_end=formation_index[-1],
                test_start=test_index[0],
                test_end=test_index[-1],
                formation_rows=len(formation_index),
                test_rows=len(test_index),
                names_traded=int(shell["names_traded"]),  # type: ignore[arg-type]
                cost_to_gross_pnl=_fold_cost_share(full_portfolio, test_index),
                signal=shell["signal"],  # type: ignore[arg-type]
                metrics=_fold_metrics_from_result(full_portfolio, test_index),
            )
        )

    evaluations = float(sum(f.signal["signal_evaluations"] for f in folds))
    weighted = {"invalid_ou_rate": float("nan"), "slow_ou_rate": float("nan")}
    if evaluations > 0:
        for key in weighted:
            weighted[key] = (
                sum(f.signal[key] * f.signal["signal_evaluations"] for f in folds if np.isfinite(f.signal[key]))
                / evaluations
            )

    summary = dict(portfolio.metrics)
    summary.update(
        {
            "n_folds": float(len(folds)),
            "n_symbols": float(len(stock_symbols)),
            "n_factor_etfs": float(len(symbols) - len(stock_symbols)),
            "n_dropped_calendar_days": float(n_dropped_calendar_days),
            "avg_names_traded": float(np.mean([f.names_traded for f in folds])),
            "signal_evaluations": evaluations,
            "invalid_ou_rate": float(weighted["invalid_ou_rate"]),
            "slow_ou_rate": float(weighted["slow_ou_rate"]),
            "n_rebalances": float(len(panel.rebalance_positions)),
            "skipped_rebalances": float(panel.skipped_rebalances),
            "avg_eligible_at_rebalance": (
                float(np.mean(panel.eligible_at_rebalance)) if panel.eligible_at_rebalance else float("nan")
            ),
            "avg_explained_variance": (
                float(np.mean(panel.explained_at_rebalance)) if panel.explained_at_rebalance else float("nan")
            ),
            "oos_periodic_sharpe": float(periodic_sharpe(portfolio.returns)),
        }
    )
    return ResidualStatArbWalkForwardResult(folds=tuple(folds), portfolio=portfolio, panel=panel, summary=summary)


def residual_fold_to_dict(fold: ResidualFoldResult) -> dict[str, object]:
    """Convert a residual fold result to a deterministic JSON-compatible dict."""
    return {
        "fold": int(fold.fold),
        "formation_start": fold.formation_start.isoformat(),
        "formation_end": fold.formation_end.isoformat(),
        "test_start": fold.test_start.isoformat(),
        "test_end": fold.test_end.isoformat(),
        "formation_rows": int(fold.formation_rows),
        "test_rows": int(fold.test_rows),
        "names_traded": int(fold.names_traded),
        "cost_to_gross_pnl": float(fold.cost_to_gross_pnl),
        "signal": {k: float(v) for k, v in sorted(fold.signal.items())},
        "metrics": {k: float(v) for k, v in sorted(fold.metrics.items())},
    }
