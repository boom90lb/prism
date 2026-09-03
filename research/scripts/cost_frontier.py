"""Cost frontier: net performance vs effective one-way spread level (uncounted diagnostic).

The corrected-axis answer to the capacity question. In the shipped cost model
the spread leg is charged per dollar traded on weight-space turnover
(``prism.execution.target_weights._cost_values``: ``commission_spread =
sum(|trade|) * (commission_bps + spread_bps) / 1e4`` with weights as fractions
of capital), so spread cost per dollar traded is AUM-invariant; the only
AUM-dependent term is the ADV participation-impact leg, which every certified
residual run disabled (``adv_impact_coeff = 0.0``), making the
``capacity_curve`` AUM axis flat by construction
(``src/prism/validation/capacity.py:55-68``). The axis an outside reader needs
is therefore the *effective one-way spread level*: at what spread does this
edge net positive, or does no spread level rescue it?

Method — pure arithmetic over STORED artifacts, no backtest rerun, no ledger
row, no counted machinery invoked:

* **Daily granularity (frozen-stack trial 3, ``results/r2_t3_full``, cert-001
  row 9):** the run's ``returns.csv``/``costs.csv`` record net daily returns
  and the per-day cost decomposition (``commission_spread``, ``impact``,
  ``borrow``, ``turnover``). Gross is reconstructed as ``net + total``; the
  commission leg (flat 1 bp on turnover) is split out of ``commission_spread``,
  leaving the realized bucket-schedule spread cost. Re-pricing at a flat
  one-way spread ``s`` is then ``net(s) = gross - turnover*(commission_bps+s)/1e4
  - impact - borrow`` — commission, quadratic impact, and borrow are held at
  the shipped model; only the spread axis moves. A validation gate asserts the
  reconstruction reproduces the stored summary before any frontier row is read.
* **Aggregate granularity (ETF-factor run, ``config_hash a2c5538e70b1``):**
  the run's per-day artifacts are no longer on disk (its output_dir lived in a
  deleted worktree), so the frontier is computed from the recorded aggregates —
  the trials-ledger row (``results/stat_arb_residual_trials.jsonl``: net
  annualized Sharpe -0.2261, 1304 obs) and ``docs/stat_arb.md:218-232`` (gross
  cumulative +6.1%, cumulative costs 8.6%, ~0.26/day turnover, flat 1 bp
  spread + 1 bp commission as run). Annualized volatility is implied from the
  ledger Sharpe and held constant across levels (stated approximation); a
  cross-check gate asserts the implied gross Sharpe matches the documented
  +0.54 within rounding of the recorded inputs.

Per spread level: net annualized Sharpe, net total return, cost/gross ratio
(the ``cost_toll`` lens for the daily config); per config: the break-even
one-way spread. The zero-cost gross ceiling row is flagged explicitly with its
deflated read — the best residual selection-set DSR is 0.145
(``docs/certifications/001-residual-reversion-daily-negative.md:108-109``), so
the multiplicity bar survives even at zero cost.

Uncounted diagnostic: searches nothing (every level is reported, none is
selected), changes no trial code path, moves no ratified statistic, writes one
JSON. Frame and results: ``docs/cost_frontier.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from prism.validation.capacity import cost_toll
from prism.validation.trials import compute_trial_metrics

# Flat one-way spread levels re-priced for every config, in bps. The bucket
# schedule (SPREAD_BUCKET_SCHEDULE_V1) enters as the daily config's as-run row.
FLAT_LEVELS_ONE_WAY_BPS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)

# Zero-cost ceiling caveat, attached to every gross-ceiling row: the
# multiplicity bar survives even at zero cost.
GROSS_CEILING_DEFLATED_READ = (
    "gross ceiling, not a claim: the best residual selection-set DSR is 0.145 "
    "(docs/certifications/001-residual-reversion-daily-negative.md:108-109); "
    "deflation fails the sleeve before costs are even charged"
)

# Recorded aggregates for the ETF-factor run. Per-day artifacts are gone (the
# ledger row's output_dir pointed into a since-deleted worktree), so these are
# the surviving primary records:
# - net_annualized_sharpe, n_obs: results/stat_arb_residual_trials.jsonl row
#   with config_hash a2c5538e70b1 (exact values).
# - gross_cum_return_arith / cost_cum_arith / avg_daily_turnover /
#   doc_gross_annualized_sharpe: docs/stat_arb.md:218-232 ("gross cumulative
#   +6.1% vs 8.6% in costs", "~0.26/day turnover", "gross annualized Sharpe
#   +0.54"). Cumulatives are read as arithmetic sums over the OOS window, the
#   same convention as the summary/costs `total_cost` (sum of daily costs).
# - commission_bps / spread_bps_as_run / borrow / slippage: the ledger row's
#   execution config (flat 1 bp spread — this run predates the bucket schedule).
ETF_FACTOR_RECORD: dict[str, float | int | str] = {
    "config_hash": "a2c5538e70b1",
    "net_annualized_sharpe": -0.22606475156753253,
    "n_obs": 1304,
    "gross_cum_return_arith": 0.061,
    "cost_cum_arith": 0.086,
    "avg_daily_turnover": 0.26,
    "commission_bps": 1.0,
    "spread_bps_as_run": 1.0,
    "doc_gross_annualized_sharpe": 0.54,
    "residual_set_dsr_recorded": 0.029,
}

# Tolerance on |implied - documented| gross annualized Sharpe for the
# aggregate reconstruction: the three doc inputs carry 2 significant figures.
AGGREGATE_CROSS_CHECK_TOL = 0.03


def require_finite(values: pd.Series, column: str, context: str) -> None:
    """Raise loudly on any non-finite value in a gated series.

    Every validation gate here is ``if error > tol: raise`` and NaN compares
    False against any tolerance, so a NaN-bearing column would sail through
    the exact gates the doc advertises and then be dropped silently
    downstream. Finiteness is therefore asserted explicitly before any
    tolerance gate runs. Every stored run to date is finite — this hardens
    the gates for future inputs; it is not a data-corruption alarm on the
    committed artifacts.
    """
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        where = [str(ix) for ix in values.index[bad][:10]]
        raise ValueError(
            f"{context}: column {column!r} carries {int(bad.sum())} non-finite value(s) "
            f"(first at {where}); the tolerance gates cannot evaluate non-finite inputs, "
            "so this input is rejected before any frontier row is read"
        )


def load_run_series(run_dir: Path) -> tuple[pd.Series, pd.DataFrame, dict, dict]:
    """Stored per-day artifacts of a finished run, on a shared calendar."""
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    returns = pd.read_csv(run_dir / "returns.csv", index_col=0)["daily_return"].astype(float)
    costs = pd.read_csv(run_dir / "costs.csv", index_col=0)
    assert (returns.index == costs.index).all(), "returns.csv and costs.csv calendars differ"
    return returns, costs, config, summary


def decompose_costs(returns: pd.Series, costs: pd.DataFrame, commission_bps: float) -> pd.DataFrame:
    """Split the stored per-day cost stream into re-priceable legs.

    Columns: ``gross_return`` (net + total, the pre-cost stream),
    ``turnover`` (one-way, fraction of capital), ``commission`` (flat leg on
    turnover), ``spread_cost`` (what the run's spread pricing actually
    charged, bucket or flat), ``fixed`` (impact + borrow — spread-invariant),
    plus exposure pass-throughs for the canonical metrics frame. Raises if
    the decomposition does not reconstruct the stored accounting exactly.
    """
    for column in ("commission_spread", "impact", "borrow", "total", "turnover", "gross", "net"):
        if column not in costs.columns:
            raise ValueError(f"costs frame lacks column {column!r}; not a stored run costs.csv")
    # Finiteness before the reconstruction gate: a NaN residual passes the
    # ``> 1e-12`` check (NaN compares False) and the NaN days then vanish
    # downstream — require_finite rejects such inputs loudly instead.
    require_finite(returns, "daily_return", "cost decomposition")
    for column in ("commission_spread", "impact", "borrow", "total", "turnover", "gross", "net"):
        require_finite(costs[column], column, "cost decomposition")
    commission = costs["turnover"] * commission_bps / 1e4
    spread_cost = costs["commission_spread"] - commission
    if (spread_cost < -1e-15).any():
        raise ValueError(
            "commission_spread < turnover * commission_bps on some day; "
            "commission_bps does not match the stored run"
        )
    decomp = pd.DataFrame(
        {
            "gross_return": returns + costs["total"],
            "turnover": costs["turnover"],
            "commission": commission,
            "spread_cost": spread_cost,
            "fixed": costs["impact"] + costs["borrow"],
            "gross_exposure": costs["gross"],
            "net_exposure": costs["net"],
        }
    )
    residual = (
        decomp["gross_return"]
        - decomp["commission"]
        - decomp["spread_cost"]
        - decomp["fixed"]
        - returns
    )
    max_residual = float(residual.abs().max())
    if max_residual > 1e-12:
        raise ValueError(
            f"cost decomposition does not reconstruct stored net returns "
            f"(max abs residual {max_residual:.3e}); total != commission_spread + impact + borrow?"
        )
    return decomp


def realized_effective_spread_bps(decomp: pd.DataFrame) -> float:
    """Realized one-way spread paid per dollar traded, in bps (turnover-weighted)."""
    turnover_sum = float(decomp["turnover"].sum())
    if turnover_sum <= 0.0:
        return float("nan")
    return float(decomp["spread_cost"].sum() / turnover_sum * 1e4)


def reprice(decomp: pd.DataFrame, one_way_spread_bps: float) -> tuple[pd.Series, pd.DataFrame]:
    """Net daily returns and a canonical costs frame at a flat one-way spread level.

    Only the spread leg moves: ``net(s) = gross - turnover*s/1e4 - commission
    - impact - borrow``. Commission, quadratic impact, and borrow stay at the
    shipped model.
    """
    if one_way_spread_bps < 0.0:
        raise ValueError(f"one_way_spread_bps must be >= 0, got {one_way_spread_bps}")
    commission_spread = decomp["commission"] + decomp["turnover"] * one_way_spread_bps / 1e4
    total = commission_spread + decomp["fixed"]
    net = (decomp["gross_return"] - total).rename("daily_return")
    costs = pd.DataFrame(
        {
            "commission_spread": commission_spread,
            "total": total,
            "turnover": decomp["turnover"],
            "gross": decomp["gross_exposure"],
            "net": decomp["net_exposure"],
        }
    )
    return net, costs


def break_even_one_way_spread_bps(decomp: pd.DataFrame) -> float:
    """The flat one-way spread (bps) at which the mean net daily return is zero.

    ``net(s)`` is linear in ``s``, so the break-even is closed-form:
    ``1e4 * mean(gross - commission - fixed) / mean(turnover)``. Negative
    means no spread level rescues the config — it is net-negative even at
    zero spread, with commission/impact/borrow still at the shipped model.
    """
    mean_turnover = float(decomp["turnover"].mean())
    if mean_turnover <= 0.0:
        return float("nan")
    base = decomp["gross_return"] - decomp["commission"] - decomp["fixed"]
    return float(1e4 * base.mean() / mean_turnover)


def _row_metrics(net: pd.Series, costs: pd.DataFrame) -> dict:
    """Canonical per-level metrics: trial metrics + the SPEC §10 cost_toll lens."""
    trial = compute_trial_metrics(net, costs)
    toll = cost_toll(net, costs)
    n_obs = int(trial["n_obs"])
    total_return = float(trial["total_return"])
    return {
        "net_annualized_sharpe": trial["annualized_sharpe"],
        "net_total_return": total_return,
        "net_annualized_return": float((1.0 + total_return) ** (252.0 / max(n_obs, 1)) - 1.0),
        "gross_annualized_sharpe": trial["gross_annualized_sharpe"],
        "total_cost": trial["total_cost"],
        "cost_to_gross": toll["cost_to_gross"],
        "toll_booth_fraction": toll["toll_booth_fraction"],
    }


def daily_frontier(
    returns: pd.Series,
    costs: pd.DataFrame,
    commission_bps: float,
    levels: tuple[float, ...] = FLAT_LEVELS_ONE_WAY_BPS,
) -> dict:
    """Full frontier for a run with stored per-day artifacts (daily granularity)."""
    decomp = decompose_costs(returns, costs, commission_bps)
    rows: list[dict] = []
    for level in levels:
        net, level_costs = reprice(decomp, level)
        rows.append({"level": f"flat_{level:g}bp", "one_way_spread_bps": level, **_row_metrics(net, level_costs)})
    # As-run row: the stored pricing itself (bucket schedule for R2 runs) —
    # reported at its realized turnover-weighted effective spread.
    rows.append(
        {
            "level": "bucket_v1_as_run",
            "one_way_spread_bps": realized_effective_spread_bps(decomp),
            **_row_metrics(returns.rename("daily_return"), costs),
        }
    )
    # Zero-cost gross ceiling: every cost leg off, flagged, never a claim.
    gross = decomp["gross_return"].rename("daily_return")
    zero_costs = pd.DataFrame(
        {
            "commission_spread": 0.0,
            "total": 0.0,
            "turnover": decomp["turnover"],
            "gross": decomp["gross_exposure"],
            "net": decomp["net_exposure"],
        }
    )
    rows.append(
        {
            "level": "zero_cost_gross_ceiling",
            "one_way_spread_bps": None,
            "gross_ceiling": True,
            "deflated_read": GROSS_CEILING_DEFLATED_READ,
            **_row_metrics(gross, zero_costs),
        }
    )
    return {
        "granularity": "daily",
        "realized_effective_one_way_spread_bps": realized_effective_spread_bps(decomp),
        "break_even_one_way_spread_bps": break_even_one_way_spread_bps(decomp),
        "frontier": rows,
    }


def validate_against_summary(returns: pd.Series, costs: pd.DataFrame, summary: dict) -> dict:
    """Gate: the loaded series must reproduce the stored summary statistics.

    Finiteness is asserted first: the tolerance comparison below is NaN-blind
    (NaN compares False against any tolerance), so a non-finite input would
    otherwise pass the gate unexamined.
    """
    require_finite(returns, "daily_return", "summary validation gate")
    for column in ("total", "turnover"):
        require_finite(costs[column], column, "summary validation gate")
    recomputed_sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0))
    recomputed_total_cost = float(costs["total"].sum())
    recomputed_turnover = float(costs["turnover"].mean())
    checks = {
        "sharpe": (recomputed_sharpe, float(summary["sharpe"])),
        "total_cost": (recomputed_total_cost, float(summary["total_cost"])),
        "avg_turnover": (recomputed_turnover, float(summary["avg_turnover"])),
    }
    for name, (got, stored) in checks.items():
        if not (np.isfinite(got) and np.isfinite(stored)):
            raise ValueError(
                f"validation gate cannot evaluate {name}: recomputed {got!r} vs stored "
                f"{stored!r} carries a non-finite value the tolerance comparison would "
                "silently pass; input rejected, frontier not computed"
            )
        if abs(got - stored) > 1e-9 * max(1.0, abs(stored)):
            raise ValueError(
                f"validation gate FAILED on {name}: recomputed {got!r} != stored {stored!r}; "
                "the loaded series is not the run's accounting, frontier not computed"
            )
    return {
        "recomputed_sharpe": recomputed_sharpe,
        "stored_sharpe": float(summary["sharpe"]),
        "recomputed_total_cost": recomputed_total_cost,
        "stored_total_cost": float(summary["total_cost"]),
        "recomputed_avg_turnover": recomputed_turnover,
        "stored_avg_turnover": float(summary["avg_turnover"]),
        "n_obs": int(len(returns)),
    }


def aggregate_frontier(
    record: dict,
    levels: tuple[float, ...] = FLAT_LEVELS_ONE_WAY_BPS,
    *,
    cross_check_tol: float = AGGREGATE_CROSS_CHECK_TOL,
) -> dict:
    """Frontier from run-level aggregates when per-day artifacts are gone.

    Approximations, stated: annualized vol is implied from the recorded net
    annualized Sharpe and held constant across spread levels (the daily-level
    computation on trial 3 shows the vol shift across levels is immaterial);
    returns are arithmetic per-year means, not compounded. The as-run row
    reproduces the ledger Sharpe by construction; the implied gross Sharpe
    must match the documented value within ``cross_check_tol`` or the
    reconstruction is rejected.
    """
    years = float(record["n_obs"]) / 252.0
    tau = float(record["avg_daily_turnover"])
    commission_bps = float(record["commission_bps"])
    spread_as_run = float(record["spread_bps_as_run"])
    ledger_sharpe = float(record["net_annualized_sharpe"])

    gross_ann = float(record["gross_cum_return_arith"]) / years
    cost_ann = float(record["cost_cum_arith"]) / years
    commission_spread_ann = 252.0 * tau * (commission_bps + spread_as_run) / 1e4
    fixed_ann = cost_ann - commission_spread_ann  # impact + borrow, spread-invariant
    if fixed_ann < 0.0:
        raise ValueError("recorded cumulative cost is below the commission+spread leg alone")
    net_ann_as_run = gross_ann - cost_ann
    vol_ann = net_ann_as_run / ledger_sharpe
    if not np.isfinite(vol_ann) or vol_ann <= 0.0:
        raise ValueError(f"implied annualized vol is unusable ({vol_ann!r})")
    implied_gross_sharpe = gross_ann / vol_ann
    doc_gross_sharpe = float(record["doc_gross_annualized_sharpe"])
    if abs(implied_gross_sharpe - doc_gross_sharpe) > cross_check_tol:
        raise ValueError(
            f"aggregate cross-check FAILED: implied gross annualized Sharpe "
            f"{implied_gross_sharpe:.4f} vs documented {doc_gross_sharpe} "
            f"(tol {cross_check_tol}); recorded aggregates are inconsistent"
        )

    def net_ann_at(spread_bps: float) -> float:
        return gross_ann - 252.0 * tau * (commission_bps + spread_bps) / 1e4 - fixed_ann

    rows: list[dict] = []
    for level in levels:
        net_ann = net_ann_at(level)
        cost_at = 252.0 * tau * (commission_bps + level) / 1e4 + fixed_ann
        rows.append(
            {
                "level": f"flat_{level:g}bp",
                "one_way_spread_bps": level,
                "net_annualized_sharpe": net_ann / vol_ann,
                "net_annualized_return_arith": net_ann,
                "net_window_return_arith": net_ann * years,
                "cost_ann_arith": cost_at,
                "cost_to_gross_pnl": cost_at / gross_ann,
            }
        )
    as_run = net_ann_at(spread_as_run) / vol_ann
    if abs(as_run - ledger_sharpe) > 1e-12:
        raise ValueError("as-run reconstruction does not reproduce the ledger Sharpe")
    rows.append(
        {
            "level": "zero_cost_gross_ceiling",
            "one_way_spread_bps": None,
            "gross_ceiling": True,
            "deflated_read": (
                f"{GROSS_CEILING_DEFLATED_READ}; this run's own recorded residual_set_dsr "
                f"~{record['residual_set_dsr_recorded']} (docs/stat_arb.md:218-232)"
            ),
            "net_annualized_sharpe": implied_gross_sharpe,
            "net_annualized_return_arith": gross_ann,
            "net_window_return_arith": gross_ann * years,
            "cost_ann_arith": 0.0,
            "cost_to_gross_pnl": 0.0,
        }
    )
    break_even = (gross_ann - fixed_ann - 252.0 * tau * commission_bps / 1e4) / (252.0 * tau / 1e4)
    return {
        "granularity": "aggregate",
        "implied_annualized_vol": vol_ann,
        "implied_gross_annualized_sharpe": implied_gross_sharpe,
        "as_run_net_annualized_sharpe": as_run,
        "break_even_one_way_spread_bps": break_even,
        "frontier": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run_dir",
        default="results/r2_t3_full",
        help="Finished run directory with per-day artifacts (default: the adjudicated frozen stack, cert-001 row 9)",
    )
    parser.add_argument(
        "--skip_etf",
        action="store_true",
        help="Skip the aggregate-granularity ETF-factor reconstruction",
    )
    parser.add_argument(
        "--out", default=None, help="Output JSON (default: results/cost_frontier_<today>.json)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    returns, costs, config, summary = load_run_series(run_dir)
    commission_bps = float(config["execution"]["commission_bps"])
    validation = validate_against_summary(returns, costs, summary)

    frozen = {
        "run_dir": str(run_dir),
        "config_hash": summary.get("config_hash"),
        "cert_reference": "docs/certifications/001-residual-reversion-daily-negative.md row 9 (frozen stack)",
        "commission_bps_held": commission_bps,
        "adv_impact_coeff": float(config["execution"]["adv_impact_coeff"]),
        "validation": validation,
        **daily_frontier(returns, costs, commission_bps),
    }

    payload: dict = {
        "meta": {
            "generated": date.today().isoformat(),
            "script": "research/scripts/cost_frontier.py",
            "status": (
                "uncounted diagnostic: re-prices stored artifacts, searches nothing, "
                "appends no ledger row, moves no ratified statistic"
            ),
            "axis_note": (
                "spread cost per dollar traded is AUM-invariant in the shipped model and the "
                "certified runs set adv_impact_coeff=0, so the capacity_curve AUM axis is flat "
                "by construction (src/prism/validation/capacity.py:55-68); the meaningful axis "
                "is the effective one-way spread level"
            ),
            "flat_levels_one_way_bps": list(FLAT_LEVELS_ONE_WAY_BPS),
            "doc": "docs/cost_frontier.md",
        },
        "frozen_stack_trial3": frozen,
    }
    if not args.skip_etf:
        payload["etf_factor_run"] = {
            "config_hash": ETF_FACTOR_RECORD["config_hash"],
            "artifact_status": (
                "per-day artifacts not on disk (ledger output_dir was in a deleted worktree); "
                "reconstructed from results/stat_arb_residual_trials.jsonl aggregates + "
                "docs/stat_arb.md:218-232"
            ),
            "inputs": dict(ETF_FACTOR_RECORD),
            **aggregate_frontier(ETF_FACTOR_RECORD),
        }

    out_path = (
        Path(args.out)
        if args.out
        else Path("results") / f"cost_frontier_{date.today().isoformat()}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
