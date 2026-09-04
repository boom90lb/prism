"""trend_v1 counted walk-forward driver (docs/trend_design.md §§1-4).

Flag defaults ARE the T0 pinned cell; T1-T4 move exactly one registered knob each;
T5 moves exactly ``end_date`` under ``--allow_post_ratification``. Every completed
run appends one row to the family ledger BEFORE the deflated Sharpe is computed
against it, and the driver refuses to run once the ledger holds the budget of 6
parseable rows. Ledgers are append-only; deleting one is self-deception.

    python -m research.scripts.trend_wfo --output_dir results/trend_t0                    # T0
    python -m research.scripts.trend_wfo --lookback 126 --output_dir results/trend_t1     # T1
    python -m research.scripts.trend_wfo --skip 0 --output_dir results/trend_t2           # T2
    python -m research.scripts.trend_wfo --decision_every 63 --output_dir results/trend_t3
    python -m research.scripts.trend_wfo --sizing equal_notional --output_dir results/trend_t4

Pinned readings (every path:line pointer read at authoring time; the core's own
readings are in ``research/arbitrage/trend_walk_forward.py``):

* **Uncounted end_date cap (Issue A).** ``docs/trend_design.md`` §0, last paragraph: "no
  trend backtest of any configuration runs before ratification, and none runs after it
  outside the §3 counted set — the budget-amnesia failure mode". The driver is the only
  mechanical enforcement point, so a scratch invocation (non-``results/`` output dir AND
  non-default ledger) must end at or before ``TREND_UNCOUNTED_END_MAX = 2021-12-31`` —
  the latest date that bounds a rehearsal well inside the pre-ratification sample and
  still contains the PDBC 2021-12-03 $5.39 credit the smoke needs. The constant is a
  guard, not a knob: it is NOT in the hashed payload (test 9 asserts its absence). The
  pure core has no such guard by design; a hand-written script importing it could still
  run full-sample — a governance matter, disclosed, not a driver defect.
* **end_date localisation and the post-load cutoff guard (Issue B).** The cache index dtype
  is ``datetime64[ns, America/New_York]`` (``src/prism/io/loader.py:45-55``); a tz-naive
  ``pd.Timestamp("2026-07-17")`` raises ``TypeError`` on comparison and ``tz="UTC"`` silently
  drops the 2026-07-17 bar (UTC midnight is 20:00 NY the previous evening). ``end_date`` is
  resolved ONCE via ``resolve_end_date`` (bare ``YYYY-MM-DD``, ``tz=TREND_TZ``) and passed as
  that object to every loader, which refuses tz-naive input rather than localising. Guard 6
  (after load, before any write) requires ``closes.index[-1] == end_date`` on the counted
  path and ``end_date in closes.index`` on the uncounted path, so a UTC implementation, a
  stale cache, or a non-session end_date cannot run T0 with ``config.end_date=2026-07-17``
  on a panel ending 2026-07-16. ``config.end_date`` / ``data.end_date`` carry
  ``end_date.isoformat()`` with offset; ``data.sample_cutoff.last_session`` records the
  actual last session.
* **Fallback never on the counted path (Issue C).** ``_load_trial_sharpes``
  (``research/scripts/stat_arb_residual_wfo.py:376-405``, copied by value below) counts EVERY
  parseable row toward ``n_trials_searched``, so a ``price_return`` fallback row on the
  default ledger would burn one of six budget slots (``docs/trend_design.md`` §3) on a run
  the adjudicator refuses to read; and the "missing distribution file" predicate is
  trivially satisfiable by pointing ``--distributions_dir`` at an empty directory, so it
  cannot gate a counted write. Guard 4 refuses ``--allow_price_return_fallback`` whenever
  ``counted`` is true, before the missing-file predicate is evaluated and before any file is
  opened for writing. ``config.price_return_fallback`` therefore cannot be True on any row
  of the default ledger; the adjudicator's readability check is defence in depth.
* **Price-convention default.** ``docs/trend_design.md`` §2 pins the trailing 252-bar TOTAL
  return and ``data/distributions`` holds ``<SYM>.csv`` + ``<SYM>.provenance.json`` for all
  ten names (GLD header-only by issuer statement), so ``total_return`` is the runnable
  registered pin and the CLI default; T0-T4 lines carry no ``--price_convention`` flag.
* **Counted-path end_date rule.** T0-T4 are registered at one cutoff (2026-07-17, the last
  bar before ratification 2026-07-18) and T5 at >= 2027-07-17 (``docs/trend_design.md`` §3,
  "sample extended >= 1 year past ratification"). ``counted = (trial_ledger resolves to
  results/trend_v1_trials.jsonl) OR (output_dir resolves under results/)``; on the counted
  path ``end_date`` must equal the default unless ``--allow_post_ratification``, in which
  case it must be ``>= TREND_T5_END_MIN = 2027-07-17`` (the registered T5 shape — a date
  merely later than the default would burn a budget slot on a non-registered cell); earlier
  than the default is never accepted. ``TREND_T5_END_MIN`` is a guard, not a knob: not
  hashed. All comparisons are tz-aware Timestamp comparisons in ``America/New_York``, never
  string comparisons.
* **Constants, not flags.** ``StatArbWalkForwardConfig(formation_bars=312, test_bars=63,
  step_bars=None, min_test_bars=20, max_gross=1.0, max_symbol_abs_weight=1.0,
  no_trade_band=0.0, band_mode="closed_form", spread_mode="bucket", max_participation=0.05,
  sleeve_mode="off")``; ``ExecutionConfig()`` (all nine fields equal B1's
  ``results/demotion_b1/config.json`` execution block, ``src/prism/config.py:38-61``);
  ``GAMMA_RISK = 1.0``; ``SPREAD_BUCKET_SCHEDULE_V1``; ``initial_capital = 1.0`` (the 5%
  participation gate is inert by magnitude, as in B1, and recorded as such). The
  registration pins only the gross cap; ``max_symbol_abs_weight=1.0`` is recorded in
  ``walk.*`` of the hash.
* **Registered-cell whitelist (guard 7) and budget (guard 3).** A counted run's hashed payload
  must equal T0's, or differ by exactly one of ``REGISTERED_CELLS`` (T1-T4), or be T0 with a
  later ``end_date`` under ``--allow_post_ratification`` (T5); ``--design_trials`` must be 6 on
  the counted path. Rehearsals (guard 8) never read the pinned cache under ``data/``.
  ``distributions_dir`` is hashed as a repo-relative normalised path, not the CLI spelling.
* **Ledger.** ``results/trend_v1_trials.jsonl``; row keys and types exactly
  ``stat_arb_residual_wfo.py:523-538``; ``_append_trial``/``_load_trial_sharpes`` copied by
  value from ``:370-405``; ``--design_trials 6`` in every row and in the hash; the ledger
  ``config_hash`` (``sha256(json.dumps(payload, sort_keys=True, default=str))[:12]``,
  ``stat_arb_residual_wfo.py:331-332``) and the packet ``config_hash`` are two digests of one
  configuration; ``summary.json`` carries the packet hash (sibling wart kept for
  row<->packet comparability).
* **Hurdle.** ``--hurdle_annual_pct 3.78 --hurdle_basis tbill_nominal`` (FRED DTB3
  2026-09-02), summary-only, not hashed, not in the packet (sibling parity).
* **Sample start** = first cached session of any pinned file (2006-09-27, the SPY min file);
  no ``--start_date`` flag; the resolved date is written to ``config.start_date``.
* **code_state** is ``current_git_state(PROJECT_DIR)`` explicitly (repo root, not cwd); a
  dirty tree records ``dirty=true`` — nothing rejects a dirty packet.
* **Prior exposure to disclose in T0's read record:** the joint-crash receipt's offline run
  of T0's mechanics (``results/joint_crash_receipt_2026-07-22/23.json``; flat costs,
  intersection calendar). No retro-appended row; its loader is not reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from prism.config import PROJECT_DIR, RESULTS_DIR, ExecutionConfig
from prism.signal.trend_node import TREND_V1_UNIVERSE
from prism.validation.metrics import after_cost_hurdle_periodic, deflated_sharpe_ratio_with_n
from prism.validation.trials import (
    current_git_state,
    emit_research_claim_packet,
    summary_claim_fields,
    validate_claim_packet_dir,
)
from research.arbitrage.residual_walk_forward import GAMMA_RISK, SPREAD_BUCKET_SCHEDULE_V1
from research.arbitrage.trend_walk_forward import (
    DATA_CONVENTION,
    ENTRY_BARS,
    PINNED_CACHE_FILES,
    STRATEGY_TAG,
    TREND_END_DATE_DEFAULT,
    TREND_T5_END_MIN,
    TREND_TZ,
    TREND_UNCOUNTED_END_MAX,
    TrendSleeveConfig,
    distributions_missing_message,
    load_distributions,
    load_pinned_panels,
    missing_distribution_files,
    resolve_end_date,
    run_trend_walk_forward,
    total_return_close,
    trend_fold_to_dict,
)
from research.arbitrage.walk_forward import StatArbWalkForwardConfig

logger = logging.getLogger(__name__)

TREND_LEDGER_DEFAULT = RESULTS_DIR / "trend_v1_trials.jsonl"
DESIGN_TRIALS_DEFAULT = 6
# The registered cells relative to T0 (docs/trend_design.md §3): exactly one hashed key moves,
# to exactly this value. T5 is T0 with a later end_date under --allow_post_ratification.
REGISTERED_CELLS: tuple[tuple[str, object], ...] = (
    ("signal.lookback_bars", 126),  # T1
    ("signal.skip_bars", 0),  # T2
    ("signal.decision_every", 63),  # T3
    ("construction.sizing", "equal_notional"),  # T4
)
HURDLE_ANNUAL_PCT_DEFAULT = 3.78  # FRED DTB3 2026-09-02
HURDLE_BASIS_DEFAULT = "tbill_nominal"
INITIAL_CAPITAL = 1.0
VOL_EWMA_BARS = 63
ANNUALIZATION_BARS = 252
DOLLAR_VOLUME_WINDOW = 20
UNIVERSE_META: dict[str, str] = {
    "name": "TREND_V1_UNIVERSE",
    "asof": "2026-07-17",
    "registration": "docs/trend_design.md §1",
    "policy": "fixed_registered_list",
}
DATA_CAVEATS: tuple[str, ...] = (
    "EEM 2008: the vendor's O=H=L=C zero-volume year is accepted as cached (never filled: the first "
    "EEM fill is >= the 2009-01-02 open, and the zero-ADV hold-prior rule keeps EEM flat until its "
    "20-bar median dollar volume is positive)",
    "EEM within-file duplicate stamps resolved keep-last (82 closes move <= $0.035)",
    "PDBC thin 2014-15 bars",
    "GLD distributions header-only by issuer statement",
    'data/distributions/verification_2026-09-03.json overall = "partial"',
)
ARTIFACTS: dict[str, str] = {
    "folds": "folds.json",
    "returns": "returns.csv",
    "equity": "equity.csv",
    "target_weights": "target_weights.csv",
    "costs": "costs.csv",
    "score_close": "score_close.csv",
    "closes": "closes.csv",
    "dividends": "dividends.csv",
    "config": "config.json",
    "summary": "summary.json",
    "claim_packet": "claim_packet.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="trend_v1 counted walk-forward driver (defaults = T0).")
    p.add_argument(
        "--price_convention",
        type=str,
        default="total_return",
        choices=tuple(DATA_CONVENTION),
        help="Registered pin is total_return (hashed). price_return is a scratch-only data-missing fallback.",
    )
    p.add_argument(
        "--allow_price_return_fallback",
        action="store_true",
        help="Required for --price_convention price_return; refused on the counted path unconditionally "
        "and on the uncounted path unless a distribution file is missing.",
    )
    p.add_argument("--data_dir", type=str, default="data", help="Read-only parquet cache directory.")
    p.add_argument(
        "--distributions_dir",
        type=str,
        default="data/distributions",
        help="<SYM>.csv + <SYM>.provenance.json per name; coverage checked under both conventions.",
    )
    p.add_argument(
        "--end_date",
        type=str,
        default=TREND_END_DATE_DEFAULT,
        help="Bare YYYY-MM-DD, resolved once as America/New_York midnight. Counted path: must equal "
        f"{TREND_END_DATE_DEFAULT} (T0-T4) or be later under --allow_post_ratification (T5). Uncounted "
        f"path: must be <= {TREND_UNCOUNTED_END_MAX}.",
    )
    p.add_argument(
        "--allow_post_ratification",
        action="store_true",
        help="T5 only: permits end_date > 2026-07-17 on the counted path.",
    )
    p.add_argument("--lookback", type=int, default=252, help="T1 -> 126.")
    p.add_argument("--skip", type=int, default=21, help="T2 -> 0.")
    p.add_argument("--decision_every", type=int, default=21, help="T3 -> 63.")
    p.add_argument("--sizing", type=str, default="inverse_vol", choices=("inverse_vol", "equal_notional"))
    p.add_argument("--output_dir", type=str, default=None, help="Default results/trend_wfo_<UTC ts>.")
    p.add_argument("--trial_ledger", type=str, default=None, help=f"Default {TREND_LEDGER_DEFAULT}.")
    p.add_argument("--design_trials", type=int, default=DESIGN_TRIALS_DEFAULT, help="Budget (hashed).")
    p.add_argument("--hurdle_annual_pct", type=float, default=HURDLE_ANNUAL_PCT_DEFAULT)
    p.add_argument(
        "--hurdle_basis",
        type=str,
        default=HURDLE_BASIS_DEFAULT,
        choices=("nominal_zero", "tbill_nominal", "tbill_real"),
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# Pinned constants
# --------------------------------------------------------------------------------------


def _walk_config() -> StatArbWalkForwardConfig:
    return StatArbWalkForwardConfig(
        formation_bars=312,
        test_bars=63,
        step_bars=None,
        min_test_bars=20,
        max_gross=1.0,
        max_symbol_abs_weight=1.0,
        no_trade_band=0.0,
        band_mode="closed_form",
        spread_mode="bucket",
        max_participation=0.05,
        sleeve_mode="off",
    )


def _sleeve_config(args: argparse.Namespace) -> TrendSleeveConfig:
    return TrendSleeveConfig(
        price_convention=args.price_convention,
        lookback_bars=int(args.lookback),
        skip_bars=int(args.skip),
        decision_every=int(args.decision_every),
        entry_bars=ENTRY_BARS,
        sizing=args.sizing,
        vol_ewma_bars=VOL_EWMA_BARS,
        annualization_bars=ANNUALIZATION_BARS,
        dollar_volume_window=DOLLAR_VOLUME_WINDOW,
    )


def _config_payload(
    args: argparse.Namespace,
    *,
    start_date: str | None = None,
    symbols: Sequence[str] = TREND_V1_UNIVERSE,
) -> dict[str, Any]:
    """The ledger-hashed configuration. T1-T4 move exactly one key; T5 moves exactly end_date."""
    sleeve = _sleeve_config(args)
    walk = _walk_config()
    execution = ExecutionConfig()
    end_date = resolve_end_date(args.end_date)
    return {
        "strategy": STRATEGY_TAG,
        "signal": {
            "lookback_bars": int(sleeve.lookback_bars),
            "skip_bars": int(sleeve.skip_bars),
            "decision_every": int(sleeve.decision_every),
            "entry_bars": int(sleeve.entry_bars),
            "dollar_volume_window": int(sleeve.dollar_volume_window),
        },
        "construction": {
            "sizing": sleeve.sizing,
            "vol_ewma_bars": int(sleeve.vol_ewma_bars),
            "annualization_bars": int(sleeve.annualization_bars),
        },
        "walk": asdict(walk),
        "execution": asdict(execution),
        "band_gamma_risk": float(GAMMA_RISK),
        "spread_schedule": {
            "name": "SPREAD_BUCKET_SCHEDULE_V1",
            "buckets": [[float(floor), float(bps)] for floor, bps in SPREAD_BUCKET_SCHEDULE_V1],
        },
        "spread_accounting": "per_fold_bucket",
        "price_convention": args.price_convention,
        "data_convention": DATA_CONVENTION[args.price_convention],
        "price_return_fallback": bool(args.allow_price_return_fallback),
        "cache_files": {sym: list(PINNED_CACHE_FILES[sym]) for sym in symbols},
        "distributions_dir": _normalized_relpath(args.distributions_dir),
        "initial_capital": float(INITIAL_CAPITAL),
        "design_trials": int(args.design_trials),
        "start_date": start_date,
        "end_date": end_date.isoformat(),
        "timezone": TREND_TZ,
        "symbols": sorted(symbols),
        "universe": dict(UNIVERSE_META),
    }


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = json.dumps(v, sort_keys=True) if isinstance(v, list) else v
    return out


def _normalized_relpath(text: str) -> str:
    """Path as hashed: relative to the repo root when inside it, else absolute; never the raw CLI spelling."""
    path = Path(text).resolve()
    root = Path(PROJECT_DIR).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _config_hash(payload: dict[str, Any]) -> str:
    # stat_arb_residual_wfo.py:331-332
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _append_trial(ledger_path: Path, entry: dict[str, Any]) -> None:
    # stat_arb_residual_wfo.py:370-373
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True, allow_nan=True) + "\n")


def _load_trial_sharpes(ledger_path: Path) -> tuple[list[float], int]:
    """``(finite_sharpes, n_trials_searched)`` — stat_arb_residual_wfo.py:376-405, copied by value.

    Every parseable ledger entry counts toward ``n_trials_searched`` (a NaN-Sharpe trial
    was still a searched configuration); only the dispersion uses the finite Sharpes.
    """
    if not ledger_path.exists():
        return [], 0
    sharpes: list[float] = []
    n_searched = 0
    for lineno, line in enumerate(ledger_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Trial ledger {ledger_path} line {lineno} is corrupt; DSR will understate trials")
            continue
        n_searched += 1
        sharpe = entry.get("oos_periodic_sharpe")
        if isinstance(sharpe, (int, float)) and np.isfinite(sharpe):
            sharpes.append(float(sharpe))
    return sharpes, n_searched


# --------------------------------------------------------------------------------------
# Guards (each SystemExit; 1-5 before any file is opened for writing, 6 after loading)
# --------------------------------------------------------------------------------------


def _is_under(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)


def _is_counted(ledger_path: Path, out_dir: Path) -> bool:
    """Guard 1: the counted-path predicate."""
    default_ledger = Path(TREND_LEDGER_DEFAULT).resolve()
    return ledger_path.resolve() == default_ledger or _is_under(out_dir.resolve(), Path(RESULTS_DIR).resolve())


def _guard_end_date(end_date: pd.Timestamp, *, counted: bool, allow_post_ratification: bool) -> None:
    """Guard 2: end_date on both paths (tz-aware Timestamp comparisons only)."""
    end_default = resolve_end_date(TREND_END_DATE_DEFAULT)
    uncounted_max = resolve_end_date(TREND_UNCOUNTED_END_MAX)
    if counted:
        if allow_post_ratification:
            # T5 is registered as "sample extended >= 1 year past ratification" (docs/trend_design.md:167):
            # anything earlier than TREND_T5_END_MIN would burn a budget slot on a non-registered cell.
            if end_date < resolve_end_date(TREND_T5_END_MIN):
                raise SystemExit(
                    f"end_date {end_date.isoformat()} is not a T5 shape: --allow_post_ratification requires "
                    f"end_date >= {TREND_T5_END_MIN} on the counted path (sample extended >= 1 year past "
                    "ratification, docs/trend_design.md:167)"
                )
            return
        if end_date != end_default:
            raise SystemExit(
                f"end_date {end_date.isoformat()} is not a registered cell on the counted path (T0–T4 are "
                f"{TREND_END_DATE_DEFAULT}; T5 needs --allow_post_ratification); sub-sample runs must pass both a "
                "non-results/ --output_dir and a non-default --trial_ledger and end at or before "
                f"{TREND_UNCOUNTED_END_MAX}"
            )
        return
    if allow_post_ratification:
        raise SystemExit(
            "--allow_post_ratification is a counted-path (T5) flag; it is refused on the uncounted path "
            "(no silent no-op flags)"
        )
    if end_date > uncounted_max:
        raise SystemExit(
            f"end_date {end_date.isoformat()} exceeds TREND_UNCOUNTED_END_MAX {TREND_UNCOUNTED_END_MAX}: "
            "docs/trend_design.md:73-74 forbids any trend backtest outside the §3 counted set after "
            "ratification; the counted cells are run with the default --trial_ledger and an --output_dir "
            "under results/"
        )


def _guard_budget(ledger_path: Path, design_trials: int, *, counted: bool) -> None:
    """Guard 3: the family budget — exactly 6 on the counted path, never refilled, not a knob."""
    if counted and int(design_trials) != DESIGN_TRIALS_DEFAULT:
        raise SystemExit(
            f"--design_trials {design_trials} is refused on the counted path: the trend_v1 budget is exactly "
            f"{DESIGN_TRIALS_DEFAULT} (docs/trend_design.md §3), never refilled"
        )
    _, n_searched = _load_trial_sharpes(ledger_path)
    if n_searched >= design_trials:
        raise SystemExit(
            f"trend_v1 budget of {design_trials} is exhausted ({n_searched} parseable rows in {ledger_path}); "
            "ledgers are append-only"
        )


def _guard_convention(args: argparse.Namespace, *, counted: bool) -> dict[str, Any] | None:
    """Guard 4: price convention and the fallback flag. Returns ``fallback_reason`` (None unless it fires)."""
    dist_dir = Path(args.distributions_dir)
    if counted and args.allow_price_return_fallback:
        raise SystemExit(
            "price_return fallback is not a registered cell; run it only with a non-results/ --output_dir "
            "and a non-default --trial_ledger"
        )
    if args.price_convention == "total_return" and args.allow_price_return_fallback:
        raise SystemExit(
            "--allow_price_return_fallback is meaningless under --price_convention total_return "
            "(no silent no-op flags)"
        )
    missing = missing_distribution_files(dist_dir, TREND_V1_UNIVERSE)
    if args.price_convention == "total_return":
        if missing:
            raise SystemExit(distributions_missing_message(missing))
        return None
    if not args.allow_price_return_fallback:
        raise SystemExit(
            "price_return is a fallback, not a cell; pass --allow_price_return_fallback (scratch only; "
            "refused unless a distribution file is missing)"
        )
    if not missing:
        raise SystemExit(
            f"--allow_price_return_fallback refused: distributions are complete under {dist_dir} for all ten "
            "names; the registered convention is total_return"
        )
    reason = {"missing_distribution_files": sorted(missing), "distributions_dir": str(dist_dir)}
    print(f"WARNING: price_return fallback active (scratch only): {json.dumps(reason, sort_keys=True)}")
    return reason


def registered_cell_diff(args: argparse.Namespace) -> dict[str, tuple[Any, Any]]:
    """Flattened hashed-payload keys where ``args`` differ from the T0 defaults: ``{key: (t0, mine)}``."""
    mine = _flat(_config_payload(args))
    t0 = _flat(_config_payload(parse_args([])))
    return {k: (t0.get(k), mine.get(k)) for k in sorted(set(mine) | set(t0)) if mine.get(k) != t0.get(k)}


def _guard_registered_cell(args: argparse.Namespace, *, counted: bool) -> str:
    """Guard 7: a counted run is exactly T0, exactly one of T1-T4, or T5 (T0 + a later end_date).

    Returns the cell label (``"T0"``..``"T5"``; ``"uncounted:<label|unregistered>"`` off the
    counted path, where nothing is refused).
    """
    diff = registered_cell_diff(args)
    # Data location is provenance, not a knob: it stays in the hash (a different directory is a different
    # configuration) but does not decide cell membership — the per-file sha256s in the packet do that job.
    diff.pop("distributions_dir", None)
    label: str | None = None
    if args.allow_post_ratification:
        rest = {k: v for k, v in diff.items() if k != "end_date"}
        if not rest:
            label = "T5"
    elif not diff:
        label = "T0"
    elif len(diff) == 1:
        ((key, (_, value)),) = diff.items()
        for i, (cell_key, cell_value) in enumerate(REGISTERED_CELLS, start=1):
            if key == cell_key and value == cell_value:
                label = f"T{i}"
    if label is None and counted:
        cells = ", ".join(f"T{i} {k}={v!r}" for i, (k, v) in enumerate(REGISTERED_CELLS, start=1))
        raise SystemExit(
            f"not a registered cell on the counted path: the hashed config differs from T0 by {diff}; "
            f"registered cells are T0 (no knob flags), {cells}, and T5 (T0 with --allow_post_ratification)"
        )
    return label or "uncounted:unregistered"


def _guard_uncounted_data(args: argparse.Namespace, *, counted: bool) -> None:
    """Guard 8: rehearsals never read the pinned cache (docs/trend_design.md §0: no trend backtest outside §3)."""
    if counted:
        return
    data_root = Path(args.data_dir).resolve()
    pinned = (Path(PROJECT_DIR) / "data").resolve()
    if data_root == pinned or data_root.is_relative_to(pinned):
        raise SystemExit(
            f"uncounted rehearsals must not read the pinned cache ({pinned}); point --data_dir at synthetic "
            "or copied files — the registration forbids any trend backtest outside the §3 counted set"
        )


def _guard_universe(symbols: Sequence[str]) -> None:
    """Guard 5: the universe is the registered list."""
    if sorted(symbols) != sorted(TREND_V1_UNIVERSE):
        raise SystemExit(f"universe {sorted(symbols)} != TREND_V1_UNIVERSE {sorted(TREND_V1_UNIVERSE)}")


def _guard_sample_cutoff(closes: pd.DataFrame, end_date: pd.Timestamp, *, counted: bool) -> pd.Timestamp:
    """Guard 6: the loaded panel actually ends on (counted) / contains (uncounted) end_date."""
    last = pd.Timestamp(closes.index[-1])
    if counted:
        if last != end_date:
            raise SystemExit(
                f"cache ends {last.isoformat()} but end_date is {end_date.isoformat()}: the counted cell must be "
                "scored on a panel that ends exactly on its registered cutoff"
            )
        return last
    if end_date not in closes.index:
        raise SystemExit(
            f"end_date {end_date.isoformat()} is not a session of the loaded panel (last session {last.isoformat()})"
        )
    return last


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    end_date = resolve_end_date(args.end_date)
    ledger_path = Path(args.trial_ledger) if args.trial_ledger else Path(TREND_LEDGER_DEFAULT)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(RESULTS_DIR) / f"trend_wfo_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}"
    )
    symbols = list(TREND_V1_UNIVERSE)

    counted = _is_counted(ledger_path, out_dir)  # guard 1
    _guard_end_date(end_date, counted=counted, allow_post_ratification=bool(args.allow_post_ratification))
    _guard_budget(ledger_path, int(args.design_trials), counted=counted)
    fallback_reason = _guard_convention(args, counted=counted)
    _guard_universe(symbols)
    cell_label = _guard_registered_cell(args, counted=counted)
    _guard_uncounted_data(args, counted=counted)

    sleeve = _sleeve_config(args)
    walk = _walk_config()
    execution = ExecutionConfig()

    closes, opens, volumes, cache_meta = load_pinned_panels(Path(args.data_dir), symbols, end_date=end_date)
    last = _guard_sample_cutoff(closes, end_date, counted=counted)

    first_session = {sym: pd.Timestamp(closes[sym].first_valid_index()) for sym in symbols}
    dist_meta: dict[str, Any] | None = None
    if args.price_convention == "total_return":
        dividends, dist_meta = load_distributions(
            Path(args.distributions_dir),
            symbols,
            pd.DatetimeIndex(closes.index),
            first_session=first_session,
            end_date=end_date,
        )
        score_close = total_return_close(closes, dividends)
        backtest_dividends: pd.DataFrame | None = dividends
    else:
        dividends = pd.DataFrame(0.0, index=closes.index, columns=symbols)
        score_close = closes
        backtest_dividends = None

    result = run_trend_walk_forward(
        closes,
        opens,
        volumes,
        score_close=score_close,
        dividends=backtest_dividends,
        sleeve=sleeve,
        walk=walk,
        execution=execution,
        initial_capital=INITIAL_CAPITAL,
    )
    portfolio = result.portfolio

    # ---- writes begin here (all guards passed) ----
    out_dir.mkdir(parents=True, exist_ok=True)
    start_date = pd.Timestamp(closes.index[0]).date().isoformat()
    payload = _config_payload(args, start_date=start_date, symbols=symbols)
    config_hash = _config_hash(payload)
    _append_trial(
        ledger_path,
        {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "strategy": STRATEGY_TAG,
            "config_hash": config_hash,
            "oos_periodic_sharpe": result.summary["oos_periodic_sharpe"],
            "oos_annualized_sharpe": result.summary["sharpe"],
            "n_obs": int(len(portfolio.returns)),
            "n_folds": int(result.summary["n_folds"]),  # type: ignore[call-overload]
            "n_symbols": int(len(symbols)),
            "design_trials": int(args.design_trials),
            "output_dir": str(out_dir),
            "config": payload,
        },
    )
    trial_sharpes, n_trials_searched = _load_trial_sharpes(ledger_path)
    n_deflation = max(int(args.design_trials), n_trials_searched)
    trend_set_dsr = deflated_sharpe_ratio_with_n(portfolio.returns, np.asarray(trial_sharpes), n_deflation)

    oos_vol = float(portfolio.returns.dropna().std(ddof=1))
    hurdle_block = {
        "annual_pct": float(args.hurdle_annual_pct),
        "basis": str(args.hurdle_basis),
        "periodic_sharpe_hurdle": float(after_cost_hurdle_periodic(args.hurdle_annual_pct / 100.0, oos_vol)),
    }
    if counted and args.allow_post_ratification:
        cutoff_rule = "post-ratification extension (T5)"
    elif counted:
        cutoff_rule = "last session before family ratification 2026-07-18"
    else:
        cutoff_rule = "uncounted sub-sample (<= TREND_UNCOUNTED_END_MAX)"

    last_scored = pd.Timestamp(portfolio.returns.index[-1])
    n_unscored_trailing = int(len(closes) - (int(closes.index.get_loc(last_scored)) + 1))
    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "registered_cell": cell_label,
        "config_hash": config_hash,
        "trial_ledger": str(ledger_path),
        "ledger_trials": n_trials_searched,
        "ledger_trials_finite_sharpe": len(trial_sharpes),
        "deflation_trials": int(n_deflation),
        "design_trials": int(args.design_trials),
        "initial_capital": float(INITIAL_CAPITAL),
        **result.summary,
        "dsr": float(trend_set_dsr),
        "after_cost_hurdle": hurdle_block,
        "universe": dict(UNIVERSE_META),
        "data_caveats": list(DATA_CAVEATS),
        "fallback_reason": fallback_reason,
        "spread_accounting": "per_fold_bucket",
        "spread_bps_first_fold": dict(result.folds[0].spread_bps),
        "spread_bps_last_fold": dict(result.folds[-1].spread_bps),
        "entry_sessions": dict(result.entry_sessions),
        "distributions": dist_meta,
        "counted": bool(counted),
        "end_date": end_date.isoformat(),
        "last_session": last.isoformat(),
        "last_scored_session": last_scored.isoformat(),
        "n_unscored_trailing_sessions": n_unscored_trailing,
    }

    (out_dir / ARTIFACTS["folds"]).write_text(
        json.dumps([trend_fold_to_dict(f) for f in result.folds], indent=2, allow_nan=True)
    )
    (out_dir / ARTIFACTS["config"]).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    portfolio.returns.to_csv(out_dir / ARTIFACTS["returns"])
    portfolio.equity.to_csv(out_dir / ARTIFACTS["equity"])
    portfolio.target_weights.to_csv(out_dir / ARTIFACTS["target_weights"])
    portfolio.costs.to_csv(out_dir / ARTIFACTS["costs"])
    score_close.to_csv(out_dir / ARTIFACTS["score_close"])
    closes.to_csv(out_dir / ARTIFACTS["closes"])
    dividends.to_csv(out_dir / ARTIFACTS["dividends"])

    data_payload: dict[str, Any] = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date.isoformat(),
        "timezone": TREND_TZ,
        "source": "twelvedata_parquet_cache_explicit_file_list",
        "bar_interval": "1d",
        "data_convention": DATA_CONVENTION[args.price_convention],
        "price_convention": args.price_convention,
        "fallback_reason": fallback_reason,
        "universe_policy": "fixed_registered_list",
        "universe": dict(UNIVERSE_META),
        "sample_cutoff": {
            "end_date": end_date.isoformat(),
            "last_session": last.isoformat(),
            "last_scored_session": last_scored.isoformat(),
            "n_unscored_trailing_sessions": n_unscored_trailing,
            "rule": cutoff_rule,
        },
        "registered_cell": cell_label,
        "n_sessions": int(len(closes)),
        "cache_files": {sym: cache_meta[sym]["files"] for sym in symbols},
        "overlap_checks": {
            sym: {
                "overlap_sessions_checked": cache_meta[sym]["overlap_sessions_checked"],
                "max_rel_close_diff": cache_meta[sym]["max_rel_close_diff"],
            }
            for sym in symbols
        },
        "listing_entry": {
            sym: {
                "first_cached_session": cache_meta[sym]["first_session"],
                "entry_session": result.entry_sessions[sym],
            }
            for sym in symbols
        },
        "distributions": dist_meta,
        "data_caveats": list(DATA_CAVEATS),
    }
    packet = emit_research_claim_packet(
        out_dir,
        filename=ARTIFACTS["claim_packet"],
        strategy=STRATEGY_TAG,
        config=payload,
        data=data_payload,
        returns=portfolio.returns,
        costs=portfolio.costs,
        target_weights=portfolio.target_weights,
        summary=summary,
        artifacts=ARTIFACTS,
        code_state=current_git_state(PROJECT_DIR),
    )
    summary_payload = {**summary, **summary_claim_fields(packet, packet_filename=ARTIFACTS["claim_packet"])}
    (out_dir / ARTIFACTS["summary"]).write_text(json.dumps(summary_payload, indent=2, allow_nan=True))
    validate_claim_packet_dir(out_dir)
    print(json.dumps(summary_payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
